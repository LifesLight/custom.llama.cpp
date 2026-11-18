// Inkling 2: the logits of the banded flash attention path (FA on) vs the dense bias path (FA off), same token stream
// prompt in several ubatches, single-token decodes past the SWA window and the relative extent, a verification batch,
// a rollback of the tail (seq_rm) and its re-decode
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

struct attention_counts {
    int contiguous = 0;
    int gathered = 0;
};

static bool count_attention(ggml_tensor * node, bool ask, void * user_data) {
    if (node->op != GGML_OP_FLASH_ATTN_EXT_BANDED) {
        return false;
    }
    if (!ask) {
        auto & counts = *static_cast<attention_counts *>(user_data);
        const auto * k = node->src[1];
        if (k->op == GGML_OP_PERMUTE && k->src[0]) {
            if (k->src[0]->op == GGML_OP_VIEW) {
                ++counts.contiguous;
            } else if (k->src[0]->op == GGML_OP_RESHAPE) {
                ++counts.gathered;
            }
        }
    }
    return true;
}

static std::vector<float> run(llama_model * model, bool fa, int n_ub, int n_ctx, attention_counts & counts) {
    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ub; cp.n_ubatch = n_ub; cp.n_seq_max = 1;
    // Keep the recurrent snapshots for the seven-token verification batch.
    cp.n_rs_seq = 6;
    cp.swa_full = false;
    cp.cb_eval = count_attention;
    cp.cb_eval_user_data = &counts;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "no ctx\n"); exit(1); }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::mt19937 rng(7);
    std::vector<float> out;
    int pos = 0;
    auto dec = [&](int n) {
        llama_batch_ext * b = llama_batch_ext_init(ctx);
        for (int i = 0; i < n; ++i) {
            const int idx = llama_batch_ext_add_token(b, 0, rng() % (n_vocab - 2));
            llama_pos p = pos++;
            llama_batch_ext_set_pos(b, idx, &p);
            llama_batch_ext_set_output_logits(b, idx, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
        for (int i = 0; i < n; ++i) {
            const float * l = llama_get_logits_ith(ctx, i);
            if (!l) { fprintf(stderr, "missing logits\n"); exit(1); }
            out.insert(out.end(), l, l + n_vocab);
        }
        llama_batch_ext_free(b);
    };
    const int n_prompt = getenv("N_PROMPT") ? atoi(getenv("N_PROMPT")) : 300;
    for (int done = 0; done < n_prompt; ) { const int n = std::min(n_ub, n_prompt - done); dec(n); done += n; }
    for (int i = 0; i < 40; ++i) dec(1);
    dec(7);
    // Roll back five positions from the verification batch before decoding again.
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, pos - 5, -1)) {
        fprintf(stderr, "rollback failed\n");
        exit(1);
    }
    pos -= 5;
    for (int i = 0; i < 30; ++i) dec(i % 9 == 4 ? 3 : 1);
    llama_free(ctx);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: model [n_ubatch n_ctx]\n"); return 1; }
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const int n_ub  = argc > 2 ? atoi(argv[2]) : 64;
    const int n_ctx = argc > 3 ? atoi(argv[3]) : 512;
    if (n_ub < 7 || n_ctx <= 0) {
        fprintf(stderr, "expected n_ubatch >= 7 and n_ctx > 0\n");
        return 1;
    }
    attention_counts fa_counts, dense_counts;
    const auto a = run(model, true,  n_ub, n_ctx, fa_counts);
    const auto b = run(model, false, n_ub, n_ctx, dense_counts);
    llama_model_free(model);
    printf("FA on: %d contiguous, %d gathered; FA off: %d contiguous, %d gathered\n",
           fa_counts.contiguous, fa_counts.gathered, dense_counts.contiguous, dense_counts.gathered);
    if (a.empty() || a.size() != b.size() || dense_counts.contiguous != 0 || dense_counts.gathered != 0) {
        fprintf(stderr, "expected matching rows and dense attention with FA off\n");
        return 1;
    }
    if (fa_counts.contiguous == 0 || fa_counts.gathered == 0) {
        fprintf(stderr, "expected both banded layouts with FA on; use a longer N_PROMPT or smaller n_ubatch to wrap the SWA cache\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    double max_d = 0, max_a = 0, sum2 = 0, ref2 = 0; int agree = 0, rows = (int) (a.size()/n_vocab);
    for (int r = 0; r < rows; ++r) {
        int ia = 0, ib = 0;
        for (int i = 0; i < n_vocab; ++i) {
            const double x = a[r*n_vocab + i], y = b[r*n_vocab + i];
            max_d = std::max(max_d, std::fabs(x - y)); max_a = std::max(max_a, std::fabs(y));
            sum2 += (x - y)*(x - y); ref2 += y*y;
            if (a[r*n_vocab + i] > a[r*n_vocab + ia]) ia = i;
            if (b[r*n_vocab + i] > b[r*n_vocab + ib]) ib = i;
        }
        agree += ia == ib;
    }
    const double nmse = sum2/ref2;
    printf("rows %d  max|diff| %.3g (logit scale %.3g)  nmse %.3g  argmax agree %.1f%%\n", rows, max_d, max_a, nmse, 100.0*agree/rows);
    return std::isfinite(nmse) && nmse <= 1e-9 ? 0 : 1;
}
