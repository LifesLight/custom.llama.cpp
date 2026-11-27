#include "common.h"
#include "log.h"
#include "speculative.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"
#include "../src/llama-arch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void set_env_str(const char * name, const char * val) {
#ifdef _WIN32
    _putenv_s(name, val ? val : "");
#else
    if (val) {
        setenv(name, val, 1);
    } else {
        unsetenv(name);
    }
#endif
}

struct kv_writer {
    struct gguf_context * gguf_ctx;
    const llm_arch        arch;

    kv_writer(enum llm_arch arch, struct gguf_context * gguf_ctx) : gguf_ctx(gguf_ctx), arch(arch) {}

    void add_kv(enum llm_kv key, uint32_t     value) { gguf_set_val_u32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, int32_t      value) { gguf_set_val_i32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, uint64_t     value) { gguf_set_val_u64 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, float        value) { gguf_set_val_f32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, bool         value) { gguf_set_val_bool(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, const char * value) { gguf_set_val_str (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }

    template <typename T>
    void add_kv(enum llm_kv key, const std::vector<T> & value) {
        if (value.empty()) {
            return;
        }
        if (std::is_same<T, uint32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT32, value.data(), value.size());
        } else if (std::is_same<T, int32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_INT32, value.data(), value.size());
        } else if (std::is_same<T, uint64_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT64, value.data(), value.size());
        } else if (std::is_same<T, float>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_FLOAT32, value.data(), value.size());
        } else {
            GGML_ABORT("unsupported vector type");
        }
    }

    void add_kv(enum llm_kv key, const std::vector<std::string> & value) {
        std::vector<const char *> tmp(value.size());
        for (size_t i = 0; i < value.size(); ++i) {
            tmp[i] = value[i].c_str();
        }
        gguf_set_arr_str(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), tmp.data(), tmp.size());
    }
};

static std::vector<float> g_normal_pool;

static void init_normal_pool(size_t seed, float stdev) {
    constexpr size_t pool_size = 4 * 1024 * 1024;
    g_normal_pool.resize(pool_size);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, stdev);
    for (float & v : g_normal_pool) {
        v = dist(gen);
    }
}

static void set_tensor_data(struct ggml_tensor * tensor, void * /*userdata*/) {
    std::vector<float> tmp(ggml_nelements(tensor));
    if (strstr(tensor->name, "exp_probs_b") != nullptr) {
        const float step = strstr(tensor->name, "exp_probs_b_vl") ? -2.0f : 2.0f;
        for (size_t i = 0; i < tmp.size(); ++i) {
            tmp[i] = step * (float) i;
        }
    } else {
        const size_t pool_size = g_normal_pool.size();
        size_t pos = std::hash<std::string>{}(tensor->name) % pool_size;
        size_t i = 0;
        while (i < tmp.size()) {
            const size_t chunk = std::min(pool_size - pos, tmp.size() - i);
            std::memcpy(tmp.data() + i, g_normal_pool.data() + pos, chunk * sizeof(float));
            i += chunk;
            pos = 0;
        }
    }

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp16(tmp.size());
        for (size_t i = 0; i < tmp.size(); ++i) {
            tmp16[i] = ggml_fp32_to_fp16(tmp[i]);
        }
        ggml_backend_tensor_set(tensor, tmp16.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("unexpected tensor type in test fixture");
    }
}

static gguf_context_ptr make_dsv41_ced_gguf(uint32_t indexer_top_k) {
    gguf_context_ptr ret(gguf_init_empty());
    kv_writer ms(LLM_ARCH_DEEPSEEK41, ret.get());

    const uint32_t n_vocab     = 64;
    const uint32_t n_ctx       = 512;
    const uint32_t n_embd      = 512;
    const uint32_t n_head      = 8;
    const uint32_t n_embd_head = n_embd / n_head; // 64
    const uint32_t n_ff        = 512;
    const uint32_t n_layer     = 10; // L_enc: 0..3, L_dec: 4..9

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llama_arch_name(LLM_ARCH_DEEPSEEK41));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       n_ff);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,      n_embd_head);
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,    n_embd_head);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,      n_embd_head / 2);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,     1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,     1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,  uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,       uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,      uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,    uint32_t(8));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,     std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));

    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        n_ff);
    ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);
    ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,              n_ff);
    ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,         uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_COUNT,                      uint32_t(4));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,                 uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,               uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,                uint32_t(4));
    ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,                1.0f);
    ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,                 uint32_t(1));

    ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,            uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,              uint32_t(32));
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,               std::vector<uint32_t>({0, 0, 2, 2, 1, 1, 1, 1, 1, 1}));
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,       160000.0f);
    ms.add_kv(LLM_KV_ATTENTION_KV_SOURCE_LAYERS,              std::vector<uint32_t>({2, 4}));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_SOURCE_LAYERS,         std::vector<uint32_t>({2, 4, 6, 8}));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, int32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, uint32_t(4096));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,            uint32_t(16));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,            uint32_t(128));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,                 indexer_top_k);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,            uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,                 uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL,     true);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS,          uint32_t(1));

    ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1e-6f);
    ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                     uint32_t(0));
    ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                     10.0f);
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                 1.0f);
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                  true);

    std::vector<int32_t> token_map(n_vocab);
    for (uint32_t i = 0; i < n_vocab; ++i) {
        token_map[i] = (int32_t) (i % 61);
    }
    ms.add_kv(LLM_KV_PLE_LAYERS,                 std::vector<uint32_t>({1, 3}));
    ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,             uint32_t(3));
    ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,        uint32_t(2));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, uint32_t(64));
    ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,      std::vector<uint64_t>({
        76632096046245ull, 4839876093313ull, 35959672319349ull,
        67716810739261ull, 51510806800915ull, 30921347202721ull }));
    ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,           std::vector<uint64_t>({0, 101, 204, 311, 420, 533, 660, 791}));
    ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,       std::vector<uint64_t>({101, 103, 107, 109, 113, 127, 131, 137}));
    ms.add_kv(LLM_KV_ENGRAM_PAD_TOKEN_ID,        uint32_t(2));
    ms.add_kv(LLM_KV_ENGRAM_TOKEN_MAP,           token_map);

    std::vector<std::string> tokenizer_list(n_vocab);
    std::vector<float>       tokenizer_scores(n_vocab, 0.0f);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "test");
    for (uint32_t i = 0; i < n_vocab; ++i) {
        tokenizer_list[i] = "tok_" + std::to_string(i);
    }
    ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokenizer_list);
    ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);

    return ret;
}

static gguf_context_ptr make_dflash_swa_gguf() {
    gguf_context_ptr ret(gguf_init_empty());
    kv_writer ms(LLM_ARCH_DFLASH, ret.get());

    const uint32_t n_vocab     = 64;
    const uint32_t n_ctx       = 512;
    const uint32_t n_embd      = 512;
    const uint32_t n_head      = 8;
    const uint32_t n_head_kv   = 2;
    const uint32_t n_embd_head = n_embd / n_head; // 64
    const uint32_t n_ff        = 512;
    const uint32_t n_layer     = 2;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,       llama_arch_name(LLM_ARCH_DFLASH));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                 n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,             n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,            n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                n_layer);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,        n_ff);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,       n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,    n_head_kv);
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       n_embd_head);
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     n_embd_head);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       n_embd_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_DFLASH_BLOCK_SIZE,          uint32_t(4));
    ms.add_kv(LLM_KV_TARGET_LAYERS,              std::vector<uint32_t>({5, 8}));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,   uint32_t(8));
    {
        const uint8_t pattern[2] = { 1, 1 };
        gguf_set_arr_data(ret.get(), llama_arch_kv_name(LLM_ARCH_DFLASH, LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN).c_str(),
                GGUF_TYPE_BOOL, pattern, 2);
    }

    std::vector<std::string> tokenizer_list(n_vocab);
    std::vector<float>       tokenizer_scores(n_vocab, 0.0f);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "test");
    ms.add_kv(LLM_KV_TOKENIZER_MASK_ID, uint32_t(0));
    for (uint32_t i = 0; i < n_vocab; ++i) {
        tokenizer_list[i] = "tok_" + std::to_string(i);
    }
    ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokenizer_list);
    ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);

    return ret;
}

static llama_model_ptr load_test_model(gguf_context * gguf_ctx, ggml_backend_dev_t dev) {
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { dev, nullptr };
    mp.devices = devs;
    llama_model_ptr m(llama_model_init_from_user(gguf_ctx, set_tensor_data, nullptr, mp));
    if (!m) {
        throw std::runtime_error("failed to create DSV41 CED test model");
    }
    return m;
}

static llama_context_ptr make_ctx(
        llama_model * model,
        bool ced_enabled,
        uint32_t n_ubatch,
        uint32_t n_batch = 256,
        uint32_t n_seq_max = 1,
        uint32_t n_rs_seq = 0,
        bool ubatch_auto = false,
        ggml_type type_k = GGML_TYPE_F16,
        bool embeddings = false,
        bool causal_attn = true,
        ggml_backend_sched_eval_callback cb_eval = nullptr,
        void * cb_eval_user_data = nullptr) {
    set_env_str("LLAMA_DSV41_CED", ced_enabled ? "1" : "0");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 512 * n_seq_max;
    cp.n_batch         = n_batch;
    cp.n_ubatch        = n_ubatch;
    cp.n_seq_max       = n_seq_max;
    cp.n_rs_seq        = n_rs_seq;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.ubatch_auto     = ubatch_auto;
    cp.type_k          = type_k;
    cp.type_v          = type_k;
    cp.embeddings      = embeddings;
    cp.cb_eval         = cb_eval;
    cp.cb_eval_user_data = cb_eval_user_data;

    llama_context_ptr ctx(llama_init_from_model(model, cp));
    if (!ctx) {
        throw std::runtime_error("failed to initialize llama_context");
    }
    if (!causal_attn) {
        llama_set_causal_attn(ctx.get(), false);
    }
    return ctx;
}

static std::vector<llama_token> make_tokens(uint32_t n, uint32_t n_vocab, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dist(0, n_vocab - 1);
    std::vector<llama_token> toks(n);
    for (uint32_t i = 0; i < n; ++i) {
        toks[i] = (llama_token) dist(gen);
    }
    return toks;
}

static std::vector<float> make_embd(uint32_t n_rows, uint32_t n_embd, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 0.2f);
    std::vector<float> v((size_t) n_rows * n_embd);
    for (float & x : v) {
        x = dist(gen);
    }
    return v;
}

static float  g_worst_abs  = 0.0f;
static double g_worst_nmse = 0.0;
static double g_tol_nmse   = 1e-6;

static float max_abs_diff(const float * a, const float * b, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        GGML_ASSERT(!std::isnan(a[i]) && !std::isnan(b[i]));
        mx = std::max(mx, std::fabs(a[i] - b[i]));
    }
    return mx;
}

static double calc_nmse(const float * a, const float * b, size_t n) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        den += (double) a[i] * (double) a[i];
    }
    return num / std::max(den, 1e-30);
}

static int argmax(const float * a, size_t n) {
    return (int) (std::max_element(a, a + n) - a);
}

static void assert_vec_close(const float * ref, const float * got, size_t n, float tol, const char * label) {
    const float  d  = max_abs_diff(ref, got, n);
    const double nm = calc_nmse(ref, got, n);
    g_worst_abs  = std::max(g_worst_abs, d);
    g_worst_nmse = std::max(g_worst_nmse, nm);
    if (d > tol || nm > g_tol_nmse) {
        fprintf(stderr, "FAIL [%s]: max_abs=%.6e (tol=%.6e), nmse=%.6e (tol=%.6e), argmax ref=%d vs got=%d\n",
                label, d, tol, nm, g_tol_nmse, argmax(ref, n), argmax(got, n));
        std::abort();
    }
}

static void feed_tokens_split(
        llama_context * ctx,
        const std::vector<llama_token> & toks,
        llama_pos pos0,
        llama_seq_id seq_id,
        const std::vector<uint32_t> & batch_ends,
        bool output_last,
        std::vector<float> & out_logits,
        uint32_t n_vocab) {
    uint32_t beg = 0;
    for (size_t b = 0; b < batch_ends.size(); ++b) {
        const uint32_t end = batch_ends[b];
        common_batch batch(ctx);
        int32_t out_idx = -1;
        for (uint32_t i = beg; i < end; ++i) {
            const bool out = output_last && (b + 1 == batch_ends.size()) && (i + 1 == end);
            batch.add(toks[i], pos0 + (llama_pos) i, seq_id, out);
            if (out) {
                out_idx = (int32_t) (i - beg);
            }
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("feed_tokens_split llama_process failed");
        }
        if (out_idx >= 0) {
            const float * l = llama_get_logits_ith(ctx, out_idx);
            GGML_ASSERT(l != nullptr);
            out_logits.assign(l, l + n_vocab);
        }
        beg = end;
    }
}

static void check_decode_steps(
        llama_context * ctx_ref,
        llama_context * ctx_ced,
        llama_pos start_pos,
        uint32_t n_steps,
        llama_seq_id seq_id,
        uint32_t n_vocab,
        float tol,
        const char * tag) {
    for (uint32_t step = 0; step < n_steps; ++step) {
        const llama_token tok = (llama_token) ((step * 13 + 7 + seq_id) % n_vocab);
        const llama_pos   pos = start_pos + (llama_pos) step;
        for (auto * ctx : { ctx_ref, ctx_ced }) {
            common_batch batch(ctx);
            batch.add(tok, pos, seq_id, true);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                throw std::runtime_error("decode step failed");
            }
        }
        const float * l_ref = llama_get_logits_ith(ctx_ref, 0);
        const float * l_ced = llama_get_logits_ith(ctx_ced, 0);
        char lbl[128];
        snprintf(lbl, sizeof(lbl), "%s step %u", tag, step);
        assert_vec_close(l_ref, l_ced, n_vocab, tol, lbl);
    }
}

struct eval_counter {
    int32_t enc_tokens = 0;
    int32_t dec_tokens = 0;
};

static bool count_decoder_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (!ask) {
        return true;
    }
    auto * ec = (eval_counter *) user_data;
    const char * name = ggml_get_name(t);
    int il = -1;
    if (sscanf(name, "attn_out-%d", &il) == 1) {
        if (il < 4) {
            ec->enc_tokens += (int32_t) t->ne[1];
        } else {
            ec->dec_tokens += (int32_t) t->ne[1];
        }
    }
    return false;
}

static void run_suite_for_device(ggml_backend_dev_t dev, const char * dev_name, bool is_gpu) {
    printf("=== Running DSV41 CED Exactness Suite on %s ===\n", dev_name);
    gguf_context_ptr gguf = make_dsv41_ced_gguf(is_gpu ? 131072u : 16u);
    llama_model_ptr model = load_test_model(gguf.get(), dev);

    const uint32_t n_vocab = (uint32_t) llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const uint32_t n_embd  = (uint32_t) llama_model_n_embd(model.get());
    g_worst_abs  = 0.0f;
    g_worst_nmse = 0.0;
    g_tol_nmse   = is_gpu ? 2e-5 : 1e-6;
    const float tol = is_gpu ? 2e-3f : 1e-4f;

    // 1. Long prompt P=200, n_ubatch=16 (< 43): multi-chunk ENC_ONLY + multi-chunk DEC_ONLY (single batch & split batch)
    {
        auto toks = make_tokens(200, n_vocab, 101);
        auto ctx_ref = make_ctx(model.get(), false, 16, 256);
        auto ctx_ced = make_ctx(model.get(), true,  16, 256);
        std::vector<float> l_ref, l_ced;
        feed_tokens_split(ctx_ref.get(), toks, 0, 0, { 200 }, true, l_ref, n_vocab);
        feed_tokens_split(ctx_ced.get(), toks, 0, 0, { 200 }, true, l_ced, n_vocab);
        assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "1a P=200 ub=16 single batch");
        check_decode_steps(ctx_ref.get(), ctx_ced.get(), 200, 16, 0, n_vocab, tol, "1a P=200 ub=16");
        printf("[PASS] Scenario 1: P=200, n_ubatch=16 multi-chunk DEC_ONLY\n");
    }

    // 2. Long prompt P=200, n_ubatch=64 (> 43): tail_keep_max truncation in ENC_ONLY + single-chunk DEC_ONLY
    {
        auto toks = make_tokens(200, n_vocab, 202);
        auto ctx_ref = make_ctx(model.get(), false, 64, 256);
        auto ctx_ced = make_ctx(model.get(), true,  64, 256);
        std::vector<float> l_ref, l_ced;
        feed_tokens_split(ctx_ref.get(), toks, 0, 0, { 128, 200 }, true, l_ref, n_vocab);
        feed_tokens_split(ctx_ced.get(), toks, 0, 0, { 128, 200 }, true, l_ced, n_vocab);
        assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "2 P=200 ub=64");
        check_decode_steps(ctx_ref.get(), ctx_ced.get(), 200, 16, 0, n_vocab, tol, "2 P=200 ub=64");
        printf("[PASS] Scenario 2: P=200, n_ubatch=64 tail_keep_max truncation\n");
    }

    // 3. Final 4-token batch after deferred ENC_ONLY prefill
    {
        auto toks = make_tokens(200, n_vocab, 303);
        auto ctx_ref = make_ctx(model.get(), false, 32, 256);
        auto ctx_ced = make_ctx(model.get(), true,  32, 256);
        std::vector<float> l_ref, l_ced;
        feed_tokens_split(ctx_ref.get(), toks, 0, 0, { 96, 196, 200 }, true, l_ref, n_vocab);
        feed_tokens_split(ctx_ced.get(), toks, 0, 0, { 96, 196, 200 }, true, l_ced, n_vocab);
        assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "3 final 4-token batch");
        check_decode_steps(ctx_ref.get(), ctx_ced.get(), 200, 16, 0, n_vocab, tol, "3 final 4-token batch");
        printf("[PASS] Scenario 3: Final 4-token batch after deferred ENC_ONLY\n");
    }

    // 4. Short prompt P=25 <= n_ubatch=64 (single-ubatch pruned FULL)
    {
        auto toks = make_tokens(25, n_vocab, 404);
        eval_counter ec_ref, ec_ced;
        auto ctx_ref = make_ctx(model.get(), false, 64, 256, 1, 0, false, GGML_TYPE_F16, false, true, count_decoder_eval_cb, &ec_ref);
        auto ctx_ced = make_ctx(model.get(), true,  64, 256, 1, 0, false, GGML_TYPE_F16, false, true, count_decoder_eval_cb, &ec_ced);
        std::vector<float> l_ref, l_ced;
        feed_tokens_split(ctx_ref.get(), toks, 0, 0, { 25 }, true, l_ref, n_vocab);
        feed_tokens_split(ctx_ced.get(), toks, 0, 0, { 25 }, true, l_ced, n_vocab);
        assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "4 short pruned FULL");
        GGML_ASSERT(ec_ced.dec_tokens < ec_ref.dec_tokens);
        check_decode_steps(ctx_ref.get(), ctx_ced.get(), 25, 16, 0, n_vocab, tol, "4 short pruned FULL");
        printf("[PASS] Scenario 4: Short prompt P=25 pruned FULL (dec ops %d vs ref %d)\n", ec_ced.dec_tokens, ec_ref.dec_tokens);
    }

    // 5. MIXED MTMD (token + embd interleaved with distinct exp_probs_b vs exp_probs_b_vl)
    {
        auto toks = make_tokens(120, n_vocab, 505);
        auto embd = make_embd(120, n_embd, 506);
        auto ctx_ref = make_ctx(model.get(), false, 32, 256);
        auto ctx_ced = make_ctx(model.get(), true,  32, 256);
        auto run_mixed = [&](llama_context * ctx) {
            llama_batch_ext_ptr batch(llama_batch_ext_init(ctx));
            // 0..39 text, 40..84 embd (spans decoder tail boundary), 85..119 text
            for (int32_t i = 0; i < 120; ++i) {
                const bool is_emb = (i >= 40 && i < 85);
                const int32_t idx = is_emb
                    ? llama_batch_ext_add_embd(batch.get(), 0, { embd.data() + (size_t) i * n_embd, 1, n_embd })
                    : llama_batch_ext_add_token(batch.get(), 0, toks[i]);
                GGML_ASSERT(idx >= 0);
                const llama_pos p[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
                llama_batch_ext_set_pos(batch.get(), idx, p);
                llama_batch_ext_set_output_logits(batch.get(), idx, i == 119);
            }
            GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
            const float * l = llama_get_logits_ith(ctx, 119);
            return std::vector<float>(l, l + n_vocab);
        };
        auto l_ref = run_mixed(ctx_ref.get());
        auto l_ced = run_mixed(ctx_ced.get());
        assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "5 mixed MTMD");
        check_decode_steps(ctx_ref.get(), ctx_ced.get(), 120, 16, 0, n_vocab, tol, "5 mixed MTMD");
        printf("[PASS] Scenario 5: MIXED MTMD with distinct exp_probs_b_vl\n");
    }

    // 6. Prefix caching (short suffix reusing kv_raw_dec + long suffix resetting kv_raw_dec)
    {
        auto toks = make_tokens(200, n_vocab, 606);
        // 6a: prefix 0..99, then short suffix 100..109
        {
            auto ctx_ref = make_ctx(model.get(), false, 32, 256);
            auto ctx_ced = make_ctx(model.get(), true,  32, 256);
            std::vector<float> l_ref, l_ced;
            feed_tokens_split(ctx_ref.get(), std::vector<llama_token>(toks.begin(), toks.begin() + 100), 0, 0, { 100 }, true, l_ref, n_vocab);
            feed_tokens_split(ctx_ced.get(), std::vector<llama_token>(toks.begin(), toks.begin() + 100), 0, 0, { 100 }, true, l_ced, n_vocab);
            assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "6a prefix");
            std::vector<llama_token> suf(toks.begin() + 100, toks.begin() + 110);
            feed_tokens_split(ctx_ref.get(), suf, 100, 0, { 10 }, true, l_ref, n_vocab);
            feed_tokens_split(ctx_ced.get(), suf, 100, 0, { 10 }, true, l_ced, n_vocab);
            assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "6a short suffix");
            check_decode_steps(ctx_ref.get(), ctx_ced.get(), 110, 16, 0, n_vocab, tol, "6a short suffix");
        }
        // 6b: prefix 0..99, then long suffix 100..199 (resets kv_raw_dec cleanly)
        {
            auto ctx_ref = make_ctx(model.get(), false, 32, 256);
            auto ctx_ced = make_ctx(model.get(), true,  32, 256);
            std::vector<float> l_ref, l_ced;
            feed_tokens_split(ctx_ref.get(), std::vector<llama_token>(toks.begin(), toks.begin() + 100), 0, 0, { 100 }, true, l_ref, n_vocab);
            feed_tokens_split(ctx_ced.get(), std::vector<llama_token>(toks.begin(), toks.begin() + 100), 0, 0, { 100 }, true, l_ced, n_vocab);
            std::vector<llama_token> suf(toks.begin() + 100, toks.end());
            feed_tokens_split(ctx_ref.get(), suf, 100, 0, { 100 }, true, l_ref, n_vocab);
            feed_tokens_split(ctx_ced.get(), suf, 100, 0, { 100 }, true, l_ced, n_vocab);
            assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "6b long suffix");
            check_decode_steps(ctx_ref.get(), ctx_ced.get(), 200, 16, 0, n_vocab, tol, "6b long suffix");
        }
        printf("[PASS] Scenario 6: Prefix caching (short reuse + long reset)\n");
    }

    // 7. Checkpoints & state_seq_get/set_data (both full and PARTIAL_ONLY) mid-deferred-prefill + global state
    {
        auto toks = make_tokens(200, n_vocab, 707);
        auto ctx_ref = make_ctx(model.get(), false, 32, 256);
        auto ctx_ced = make_ctx(model.get(), true,  32, 256);
        std::vector<float> l_ref, l_ced;
        feed_tokens_split(ctx_ref.get(), toks, 0, 0, { 200 }, true, l_ref, n_vocab);
        // deferred up to 179
        feed_tokens_split(ctx_ced.get(), std::vector<llama_token>(toks.begin(), toks.begin() + 180), 0, 0, { 180 }, false, l_ced, n_vocab);

        for (llama_state_seq_flags flags : { (llama_state_seq_flags) 0, (llama_state_seq_flags) LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY }) {
            const size_t sz = llama_state_seq_get_size_ext(ctx_ced.get(), 0, flags);
            std::vector<uint8_t> buf(sz);
            const size_t nw = llama_state_seq_get_data_ext(ctx_ced.get(), buf.data(), buf.size(), 0, flags);
            GGML_ASSERT(nw == sz);

            if (flags == 0) {
                auto ctx_restored = make_ctx(model.get(), true, 32, 256);
                const size_t nr = llama_state_seq_set_data_ext(ctx_restored.get(), buf.data(), buf.size(), 0, flags);
                GGML_ASSERT(nr == nw);

                std::vector<llama_token> suf(toks.begin() + 180, toks.end());
                feed_tokens_split(ctx_restored.get(), suf, 180, 0, { 20 }, true, l_ced, n_vocab);
                assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "7 state_seq FULL");

                auto ctx_ref2 = make_ctx(model.get(), false, 32, 256);
                std::vector<float> l_ref2;
                feed_tokens_split(ctx_ref2.get(), toks, 0, 0, { 200 }, true, l_ref2, n_vocab);
                check_decode_steps(ctx_ref2.get(), ctx_restored.get(), 200, 16, 0, n_vocab, tol, "7 state_seq FULL decode");
            } else {
                // PARTIAL_ONLY omits global compressed KV (kv_csa/hca/lid) and rewinds the active sequence in-place.
                std::vector<llama_token> suf(toks.begin() + 180, toks.end());
                feed_tokens_split(ctx_ced.get(), suf, 180, 0, { 20 }, true, l_ced, n_vocab);
                assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "7 pre-restore");

                const size_t nr = llama_state_seq_set_data_ext(ctx_ced.get(), buf.data(), buf.size(), 0, flags);
                GGML_ASSERT(nr == nw);

                feed_tokens_split(ctx_ced.get(), suf, 180, 0, { 20 }, true, l_ced, n_vocab);
                assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "7 state_seq PARTIAL_ONLY");

                auto ctx_ref2 = make_ctx(model.get(), false, 32, 256);
                std::vector<float> l_ref2;
                feed_tokens_split(ctx_ref2.get(), toks, 0, 0, { 200 }, true, l_ref2, n_vocab);
                check_decode_steps(ctx_ref2.get(), ctx_ced.get(), 200, 16, 0, n_vocab, tol, "7 state_seq PARTIAL_ONLY decode");
            }
        }

        // Global state save/load mid-deferred prefill
        auto ctx_g_src = make_ctx(model.get(), true, 32, 256);
        feed_tokens_split(ctx_g_src.get(), std::vector<llama_token>(toks.begin(), toks.begin() + 180), 0, 0, { 180 }, false, l_ced, n_vocab);
        const size_t gsz = llama_state_get_size(ctx_g_src.get());
        std::vector<uint8_t> gbuf(gsz);
        const size_t gnw = llama_state_get_data(ctx_g_src.get(), gbuf.data(), gbuf.size());
        GGML_ASSERT(gnw == gsz);
        auto ctx_g_dst = make_ctx(model.get(), true, 32, 256);
        GGML_ASSERT(llama_state_set_data(ctx_g_dst.get(), gbuf.data(), gbuf.size()) == gnw);
        feed_tokens_split(ctx_g_dst.get(), std::vector<llama_token>(toks.begin() + 180, toks.end()), 180, 0, { 20 }, true, l_ced, n_vocab);
        assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "7 global state");
        printf("[PASS] Scenario 7: State save/load (full, PARTIAL_ONLY, global) mid-deferred prefill\n");
    }

    // 8. Speculative rollback (n_rs_seq=4) pre-flush (+ immediate checkpoint!) & post-flush
    {
        auto toks = make_tokens(180, n_vocab, 808);
        auto ctx_ref = make_ctx(model.get(), false, 32, 256, 1, 4);
        auto ctx_ced = make_ctx(model.get(), true,  32, 256, 1, 4);
        std::vector<float> l_ref, l_ced;

        // Pre-flush rollback on [0..123] -> rollback pos >= 121
        for (auto * ctx : { ctx_ref.get(), ctx_ced.get() }) {
            std::vector<float> dummy;
            feed_tokens_split(ctx, std::vector<llama_token>(toks.begin(), toks.begin() + 124), 0, 0, { 120, 124 }, false, dummy, n_vocab);
            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx), 0, 121, -1));
        }

        // Immediate checkpoint (PARTIAL_ONLY) right after pre-flush rollback!
        {
            const size_t sz = llama_state_seq_get_size_ext(ctx_ced.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            std::vector<uint8_t> buf(sz);
            GGML_ASSERT(llama_state_seq_get_data_ext(ctx_ced.get(), buf.data(), buf.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == sz);
        }

        // Continue after pre-flush rollback to 160 with output, then do post-flush rollback at 161..164 -> keep 160
        std::vector<llama_token> cont(toks.begin() + 121, toks.begin() + 160);
        feed_tokens_split(ctx_ref.get(), cont, 121, 0, { (uint32_t) cont.size() }, true, l_ref, n_vocab);
        feed_tokens_split(ctx_ced.get(), cont, 121, 0, { (uint32_t) cont.size() }, true, l_ced, n_vocab);
        assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "8a pre-flush rollback");

        for (auto * ctx : { ctx_ref.get(), ctx_ced.get() }) {
            common_batch batch(ctx);
            for (int i = 0; i < 4; ++i) {
                batch.add(toks[160 + i], 160 + i, 0, i == 3);
            }
            GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx), 0, 161, -1));
        }
        check_decode_steps(ctx_ref.get(), ctx_ced.get(), 161, 12, 0, n_vocab, tol, "8b post-flush rollback");
        printf("[PASS] Scenario 8: Speculative rollback pre-flush (+checkpoint) & post-flush\n");
    }

    // 9. Multi-seq + seq_cp mid-deferred prefill + divergent suffixes
    {
        auto toks0 = make_tokens(180, n_vocab, 909);
        auto toks1 = make_tokens(190, n_vocab, 910);
        // shared prefix 0..149
        std::copy(toks0.begin(), toks0.begin() + 150, toks1.begin());

        auto ctx_ref = make_ctx(model.get(), false, 32, 256, 2, 0);
        auto ctx_ced = make_ctx(model.get(), true,  32, 256, 2, 0);

        for (auto * ctx : { ctx_ref.get(), ctx_ced.get() }) {
            std::vector<float> dummy;
            feed_tokens_split(ctx, std::vector<llama_token>(toks0.begin(), toks0.begin() + 150), 0, 0, { 150 }, false, dummy, n_vocab);
            llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, -1, -1);

            common_batch batch(ctx);
            for (int i = 150; i < 180; ++i) {
                batch.add(toks0[i], i, 0, i == 179);
            }
            for (int i = 150; i < 190; ++i) {
                batch.add(toks1[i], i, 1, i == 189);
            }
            GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
        }
        // seq 0 out idx is 29, seq 1 out idx is 30 + 39 = 69
        assert_vec_close(llama_get_logits_ith(ctx_ref.get(), 29), llama_get_logits_ith(ctx_ced.get(), 29), n_vocab, tol, "9 seq0");
        assert_vec_close(llama_get_logits_ith(ctx_ref.get(), 69), llama_get_logits_ith(ctx_ced.get(), 69), n_vocab, tol, "9 seq1");
        check_decode_steps(ctx_ref.get(), ctx_ced.get(), 180, 8, 0, n_vocab, tol, "9 seq0 decode");
        check_decode_steps(ctx_ref.get(), ctx_ced.get(), 190, 8, 1, n_vocab, tol, "9 seq1 decode");
        printf("[PASS] Scenario 9: Multi-seq + seq_cp mid-deferred prefill\n");
    }

    // 10. ubatch_auto + LLAMA_DECODE_WORKSPACE=4 + Q8_0 K-cache (Hadamard k_rot check)
    {
        set_env_str("LLAMA_DECODE_WORKSPACE", "4");
        auto toks = make_tokens(200, n_vocab, 1010);
        auto ctx_ref = make_ctx(model.get(), false, 32, 256, 1, 0, true, GGML_TYPE_Q8_0);
        auto ctx_ced = make_ctx(model.get(), true,  32, 256, 1, 0, true, GGML_TYPE_Q8_0);
        std::vector<float> l_ref, l_ced;
        feed_tokens_split(ctx_ref.get(), toks, 0, 0, { 196, 200 }, true, l_ref, n_vocab);
        feed_tokens_split(ctx_ced.get(), toks, 0, 0, { 196, 200 }, true, l_ced, n_vocab);
        assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "10 ubatch_auto + workspace=4 + q8_0");
        check_decode_steps(ctx_ref.get(), ctx_ced.get(), 200, 16, 0, n_vocab, tol, "10 ubatch_auto + q8_0");
        set_env_str("LLAMA_DECODE_WORKSPACE", nullptr);
        printf("[PASS] Scenario 10: ubatch_auto + LLAMA_DECODE_WORKSPACE=4 + Q8_0 K-cache\n");
    }

    // 11. Fallbacks & DSpark layer_inp
    {
        auto toks = make_tokens(120, n_vocab, 1111);

        // 11a: Multi-output batch
        {
            auto ctx_ref = make_ctx(model.get(), false, 32, 256);
            auto ctx_ced = make_ctx(model.get(), true,  32, 256);
            for (auto * ctx : { ctx_ref.get(), ctx_ced.get() }) {
                common_batch batch(ctx);
                for (int i = 0; i < 120; ++i) {
                    batch.add(toks[i], i, 0, i == 60 || i == 119);
                }
                GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
            }
            assert_vec_close(llama_get_logits_ith(ctx_ref.get(), 60),  llama_get_logits_ith(ctx_ced.get(), 60),  n_vocab, tol, "11a multi-out 60");
            assert_vec_close(llama_get_logits_ith(ctx_ref.get(), 119), llama_get_logits_ith(ctx_ced.get(), 119), n_vocab, tol, "11a multi-out 119");
        }

        // 11b: embeddings=true fallback
        {
            auto ctx_ref = make_ctx(model.get(), false, 32, 256, 1, 0, false, GGML_TYPE_F16, true);
            auto ctx_ced = make_ctx(model.get(), true,  32, 256, 1, 0, false, GGML_TYPE_F16, true);
            for (auto * ctx : { ctx_ref.get(), ctx_ced.get() }) {
                common_batch batch(ctx);
                for (int i = 0; i < 80; ++i) {
                    batch.add(toks[i], i, 0, true);
                }
                GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
            }
            assert_vec_close(llama_get_embeddings_ith(ctx_ref.get(), 79), llama_get_embeddings_ith(ctx_ced.get(), 79), n_embd, tol, "11b embeddings");
        }

        // 11c: !causal_attn fallback
        {
            auto ctx_ref = make_ctx(model.get(), false, 256, 256, 1, 0, false, GGML_TYPE_F16, false, false);
            auto ctx_ced = make_ctx(model.get(), true,  256, 256, 1, 0, false, GGML_TYPE_F16, false, false);
            std::vector<float> l_ref, l_ced;
            feed_tokens_split(ctx_ref.get(), std::vector<llama_token>(toks.begin(), toks.begin() + 64), 0, 0, { 64 }, true, l_ref, n_vocab);
            feed_tokens_split(ctx_ced.get(), std::vector<llama_token>(toks.begin(), toks.begin() + 64), 0, 0, { 64 }, true, l_ced, n_vocab);
            assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "11c non-causal");
        }

        // 11d: DSpark decoder layer_inp (layer 9, tail=8)
        {
            auto toks200 = make_tokens(200, n_vocab, 1112);
            auto ctx_ref = make_ctx(model.get(), false, 32, 256);
            auto ctx_ced = make_ctx(model.get(), true,  32, 256);
            llama_set_embeddings_layer_inp(ctx_ref.get(), 9, true);
            llama_set_embeddings_layer_inp(ctx_ced.get(), 9, true);
            llama_set_embeddings_layer_inp_tail(ctx_ced.get(), 8);

            // Feed identical split {128, 200}; host embd_layer_inp on ref holds [128..199]
            std::vector<float> l_ref, l_ced;
            feed_tokens_split(ctx_ref.get(), toks200, 0, 0, { 128, 200 }, true, l_ref, n_vocab);
            feed_tokens_split(ctx_ced.get(), toks200, 0, 0, { 128, 200 }, true, l_ced, n_vocab);

            assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "11d dspark logits");
            llama_pos pos_beg = -1;
            int32_t   row_beg = -1;
            int32_t   n_rows  = -1;
            GGML_ASSERT(llama_ced_layer_inp_status(ctx_ced.get(), 0, &pos_beg, &row_beg, &n_rows) == 1);
            GGML_ASSERT(pos_beg == 192 && row_beg == 0 && n_rows == 8);

            const float * li_ref = llama_get_embeddings_layer_inp(ctx_ref.get(), 9);
            const float * li_ced = llama_get_embeddings_layer_inp(ctx_ced.get(), 9);
            assert_vec_close(li_ref + (size_t) (192 - 128) * n_embd, li_ced + (size_t) row_beg * n_embd, (size_t) n_rows * n_embd, tol, "11d dspark layer_inp rows");
            llama_ced_clear_layer_inp_status(ctx_ced.get(), 0);
            GGML_ASSERT(llama_ced_layer_inp_status(ctx_ced.get(), 0, nullptr, nullptr, nullptr) == -1);
        }

        // 11e: Encoder layer_inp fallback (layer 2)
        {
            auto ctx_ref = make_ctx(model.get(), false, 32, 256);
            auto ctx_ced = make_ctx(model.get(), true,  32, 256);
            llama_set_embeddings_layer_inp(ctx_ref.get(), 2, true);
            llama_set_embeddings_layer_inp(ctx_ced.get(), 2, true);
            std::vector<float> l_ref, l_ced;
            feed_tokens_split(ctx_ref.get(), std::vector<llama_token>(toks.begin(), toks.begin() + 64), 0, 0, { 64 }, true, l_ref, n_vocab);
            feed_tokens_split(ctx_ced.get(), std::vector<llama_token>(toks.begin(), toks.begin() + 64), 0, 0, { 64 }, true, l_ced, n_vocab);
            assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "11e enc layer_inp fallback");
            assert_vec_close(llama_get_embeddings_layer_inp(ctx_ref.get(), 2),
                             llama_get_embeddings_layer_inp(ctx_ced.get(), 2), 64 * (size_t) n_embd, tol, "11e enc layer_inp rows");
        }
        printf("[PASS] Scenario 11: Fallbacks (multi-out, embeddings, !causal) & DSpark layer_inp\n");
    }

    // 12. Verify decoder compute skip via cb_eval on P=200
    {
        auto toks = make_tokens(200, n_vocab, 1212);
        eval_counter ec_ref, ec_ced;
        auto ctx_ref = make_ctx(model.get(), false, 32, 256, 1, 0, false, GGML_TYPE_F16, false, true, count_decoder_eval_cb, &ec_ref);
        auto ctx_ced = make_ctx(model.get(), true,  32, 256, 1, 0, false, GGML_TYPE_F16, false, true, count_decoder_eval_cb, &ec_ced);
        std::vector<float> l_ref, l_ced;
        feed_tokens_split(ctx_ref.get(), toks, 0, 0, { 200 }, true, l_ref, n_vocab);
        feed_tokens_split(ctx_ced.get(), toks, 0, 0, { 200 }, true, l_ced, n_vocab);
        GGML_ASSERT(ec_ref.enc_tokens == ec_ced.enc_tokens);
        GGML_ASSERT(ec_ced.dec_tokens < ec_ref.dec_tokens / 4);
        printf("[PASS] Scenario 12: Decoder compute skip (CED=%d vs naive=%d decoder token-ops, enc=%d)\n",
                ec_ced.dec_tokens, ec_ref.dec_tokens, ec_ced.enc_tokens);
    }

    // 13. Uniform-SWA DFlash/DSpark draft flow under CED: mid-prompt checkpoints, deferred prefill, and context reverts
    {
        gguf_context_ptr gguf_dft = make_dflash_swa_gguf();
        llama_model_ptr  model_dft = load_test_model(gguf_dft.get(), dev);
        GGML_ASSERT(llama_model_is_swa_all(model_dft.get()));
        GGML_ASSERT(llama_model_n_swa(model_dft.get()) == 8);

        auto make_dft_ctx = [&](llama_context * ctx_tgt) -> llama_context_ptr {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx           = 512;
            cp.n_batch         = 256;
            cp.n_ubatch        = 32;
            cp.n_seq_max       = 1;
            cp.n_threads       = 4;
            cp.n_threads_batch = 4;
            cp.type_k          = GGML_TYPE_F16;
            cp.type_v          = GGML_TYPE_F16;
            cp.ctx_other       = ctx_tgt;
            llama_context_ptr ctx(llama_init_from_model(model_dft.get(), cp));
            if (!ctx) {
                throw std::runtime_error("failed to initialize uniform-SWA DFlash draft context");
            }
            return ctx;
        };

        auto make_spec = [&](llama_context * ctx_tgt, llama_context * ctx_dft) -> common_speculative_ptr {
            common_params_speculative sp;
            sp.types               = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
            sp.draft.ctx_tgt       = ctx_tgt;
            sp.draft.ctx_dft       = ctx_dft;
            sp.draft.n_max         = 3;
            sp.draft.n_min         = 0;
            sp.draft.p_min         = 0.0f;
            sp.draft.n_ubatch      = 32;
            common_speculative_ptr s(common_speculative_init(sp, 1));
            GGML_ASSERT(s != nullptr);
            return s;
        };

        auto feed_with_spec = [&](
                llama_context * ctx_tgt,
                common_speculative * spec,
                const std::vector<llama_token> & toks,
                llama_pos pos0,
                const std::vector<uint32_t> & ends,
                bool output_last,
                std::vector<float> & out_logits) {
            uint32_t beg = 0;
            for (size_t b = 0; b < ends.size(); ++b) {
                const uint32_t end = ends[b];
                common_batch batch(ctx_tgt);
                int32_t out_idx = -1;
                for (uint32_t i = beg; i < end; ++i) {
                    const bool out = output_last && (b + 1 == ends.size()) && (i + 1 == end);
                    batch.add(toks[i], pos0 + (llama_pos) i, 0, out);
                    if (out) {
                        out_idx = (int32_t) (i - beg);
                    }
                }
                GGML_ASSERT(llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
                if (out_idx >= 0) {
                    const float * l = llama_get_logits_ith(ctx_tgt, out_idx);
                    GGML_ASSERT(l != nullptr);
                    out_logits.assign(l, l + n_vocab);
                }
                GGML_ASSERT(common_speculative_process(spec, batch));
                beg = end;
            }
        };

        auto probe_dft_logits = [&](llama_context * ctx_dft, llama_pos pos0, llama_token tok_last) -> std::vector<float> {
            common_batch b(ctx_dft);
            for (int32_t i = 0; i < 4; ++i) {
                b.add(i == 0 ? tok_last : 0, pos0 + i, 0, i == 1);
            }
            GGML_ASSERT(llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, b.get()) == 0);
            const float * l = llama_get_logits_ith(ctx_dft, 1);
            GGML_ASSERT(l != nullptr);
            std::vector<float> res(l, l + n_vocab);
            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, pos0, -1));
            return res;
        };

        for (int dev_feat : { 1, 0 }) {
            set_env_str("LLAMA_DFLASH_DEV", dev_feat ? "1" : "0");
            const auto base_toks  = make_tokens(200, n_vocab, 1300 + dev_feat);
            const auto alt_long   = make_tokens(80,  n_vocab, 1320 + dev_feat);
            const auto alt_short  = make_tokens(4,   n_vocab, 1340 + dev_feat);
            const auto alt_mid    = make_tokens(20,  n_vocab, 1360 + dev_feat);

            auto ctx_tgt_ref = make_ctx(model.get(), false, 24, 256);
            auto ctx_dft_ref = make_dft_ctx(ctx_tgt_ref.get());
            auto spec_ref    = make_spec(ctx_tgt_ref.get(), ctx_dft_ref.get());

            auto ctx_tgt_ced = make_ctx(model.get(), true,  24, 256);
            auto ctx_dft_ced = make_dft_ctx(ctx_tgt_ced.get());
            auto spec_ced    = make_spec(ctx_tgt_ced.get(), ctx_dft_ced.get());

            // 13a: Mid-prompt PARTIAL_ONLY checkpoint at pos=119, then long deferred prefill [120..199]
            std::vector<float> l_ref, l_ced;
            feed_with_spec(ctx_tgt_ref.get(), spec_ref.get(), std::vector<llama_token>(base_toks.begin(), base_toks.begin() + 120), 0, { 120 }, false, l_ref);
            feed_with_spec(ctx_tgt_ced.get(), spec_ced.get(), std::vector<llama_token>(base_toks.begin(), base_toks.begin() + 120), 0, { 120 }, false, l_ced);

            common_prompt_checkpoint ckpt_ced;
            ckpt_ced.update_pos(120, 0, 119);
            ckpt_ced.update_tgt(ctx_tgt_ced.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            GGML_ASSERT(common_speculative_flush_ced(spec_ced.get(), 0));
            ckpt_ced.update_dft(ctx_dft_ced.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(ctx_dft_ced.get()), 0) == 119);

            std::vector<llama_token> suf13a(base_toks.begin() + 120, base_toks.end());
            feed_with_spec(ctx_tgt_ref.get(), spec_ref.get(), suf13a, 120, { 40, 80 }, true, l_ref);
            feed_with_spec(ctx_tgt_ced.get(), spec_ced.get(), suf13a, 120, { 40, 80 }, true, l_ced);
            assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, dev_feat ? "13a dev tgt" : "13a host tgt");
            auto d_ref_a = probe_dft_logits(ctx_dft_ref.get(), 200, base_toks.back());
            auto d_ced_a = probe_dft_logits(ctx_dft_ced.get(), 200, base_toks.back());
            assert_vec_close(d_ref_a.data(), d_ced_a.data(), n_vocab, tol, dev_feat ? "13a dev dft" : "13a host dft");

            // 13b: Restore PARTIAL_ONLY checkpoint pos=119 on ctx_tgt + ctx_dft, then long different continuation [120..199]
            GGML_ASSERT(ckpt_ced.load_tgt(ctx_tgt_ced.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            GGML_ASSERT(ckpt_ced.load_dft(ctx_dft_ced.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx_tgt_ced.get()), 0, 120, -1));
            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx_dft_ced.get()), 0, 120, -1));
            GGML_ASSERT(llama_ced_layer_inp_status(ctx_tgt_ced.get(), 0, nullptr, nullptr, nullptr) == -1);
            feed_with_spec(ctx_tgt_ced.get(), spec_ced.get(), alt_long, 120, { 40, 80 }, true, l_ced);
            {
                auto ctx_t2 = make_ctx(model.get(), false, 24, 256);
                auto ctx_d2 = make_dft_ctx(ctx_t2.get());
                auto spec2  = make_spec(ctx_t2.get(), ctx_d2.get());
                std::vector<llama_token> full_b(base_toks.begin(), base_toks.begin() + 120);
                full_b.insert(full_b.end(), alt_long.begin(), alt_long.end());
                feed_with_spec(ctx_t2.get(), spec2.get(), full_b, 0, { 120, 160, 200 }, true, l_ref);
                assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, dev_feat ? "13b dev tgt" : "13b host tgt");
                auto d_ref_b = probe_dft_logits(ctx_d2.get(), 200, alt_long.back());
                auto d_ced_b = probe_dft_logits(ctx_dft_ced.get(), 200, alt_long.back());
                assert_vec_close(d_ref_b.data(), d_ced_b.data(), n_vocab, tol, dev_feat ? "13b dev dft" : "13b host dft");
            }

            // 13c: Restore PARTIAL_ONLY checkpoint pos=119 on ctx_tgt + ctx_dft, then short consecutive continuation [120..123]
            GGML_ASSERT(ckpt_ced.load_tgt(ctx_tgt_ced.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            GGML_ASSERT(ckpt_ced.load_dft(ctx_dft_ced.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx_tgt_ced.get()), 0, 120, -1));
            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx_dft_ced.get()), 0, 120, -1));
            feed_with_spec(ctx_tgt_ced.get(), spec_ced.get(), alt_short, 120, { 4 }, true, l_ced);
            {
                auto ctx_t2 = make_ctx(model.get(), false, 24, 256);
                auto ctx_d2 = make_dft_ctx(ctx_t2.get());
                auto spec2  = make_spec(ctx_t2.get(), ctx_d2.get());
                std::vector<llama_token> full_c(base_toks.begin(), base_toks.begin() + 120);
                full_c.insert(full_c.end(), alt_short.begin(), alt_short.end());
                feed_with_spec(ctx_t2.get(), spec2.get(), full_c, 0, { 120, 124 }, true, l_ref);
                assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, dev_feat ? "13c dev tgt" : "13c host tgt");
                auto d_ref_c = probe_dft_logits(ctx_d2.get(), 124, alt_short.back());
                auto d_ced_c = probe_dft_logits(ctx_dft_ced.get(), 124, alt_short.back());
                assert_vec_close(d_ref_c.data(), d_ced_c.data(), n_vocab, tol, dev_feat ? "13c dev dft" : "13c host dft");
            }

            // 13d: Context revert mid-deferred prefill (defer [120..159], restore pos=119 without flush, then alt_mid [120..139])
            GGML_ASSERT(ckpt_ced.load_tgt(ctx_tgt_ced.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            GGML_ASSERT(ckpt_ced.load_dft(ctx_dft_ced.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx_tgt_ced.get()), 0, 120, -1));
            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx_dft_ced.get()), 0, 120, -1));
            feed_with_spec(ctx_tgt_ced.get(), spec_ced.get(), std::vector<llama_token>(alt_long.begin(), alt_long.begin() + 40), 120, { 40 }, false, l_ced);
            GGML_ASSERT(llama_ced_layer_inp_status(ctx_tgt_ced.get(), 0, nullptr, nullptr, nullptr) == 0);

            GGML_ASSERT(ckpt_ced.load_tgt(ctx_tgt_ced.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            GGML_ASSERT(ckpt_ced.load_dft(ctx_dft_ced.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx_tgt_ced.get()), 0, 120, -1));
            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx_dft_ced.get()), 0, 120, -1));
            GGML_ASSERT(llama_ced_layer_inp_status(ctx_tgt_ced.get(), 0, nullptr, nullptr, nullptr) == -1);
            feed_with_spec(ctx_tgt_ced.get(), spec_ced.get(), alt_mid, 120, { 20 }, true, l_ced);
            {
                auto ctx_t2 = make_ctx(model.get(), false, 24, 256);
                auto ctx_d2 = make_dft_ctx(ctx_t2.get());
                auto spec2  = make_spec(ctx_t2.get(), ctx_d2.get());
                std::vector<llama_token> full_d(base_toks.begin(), base_toks.begin() + 120);
                full_d.insert(full_d.end(), alt_mid.begin(), alt_mid.end());
                feed_with_spec(ctx_t2.get(), spec2.get(), full_d, 0, { 120, 140 }, true, l_ref);
                assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, dev_feat ? "13d dev tgt" : "13d host tgt");
                auto d_ref_d = probe_dft_logits(ctx_d2.get(), 140, alt_mid.back());
                auto d_ced_d = probe_dft_logits(ctx_dft_ced.get(), 140, alt_mid.back());
                assert_vec_close(d_ref_d.data(), d_ced_d.data(), n_vocab, tol, dev_feat ? "13d dev dft" : "13d host dft");
            }
        }
        set_env_str("LLAMA_DFLASH_DEV", nullptr);
        printf("[PASS] Scenario 13: Uniform-SWA DFlash/DSpark checkpoints, deferred prefill & context reverts\n");
    }

    // 14. Split CED reserve (n_ubatch=128 > ced_tail_cap=44, ubatch_auto=true):
    //     deferred prompt, single-call prompt, short follow-up, multi-out fallback,
    //     DSpark tail layer_inp, non-causal toggle, and llama_ubatch_measure
    {
        auto toks200 = make_tokens(200, n_vocab, 1401);
        auto toks100 = make_tokens(100, n_vocab, 1402);
        auto toks16  = make_tokens(16,  n_vocab, 1403);
        auto toks32  = make_tokens(32,  n_vocab, 1404);

        // 14a: Deferred prompt P=200 with llama_ubatch_prepare
        {
            auto ctx_ref = make_ctx(model.get(), false, 128, 256, 1, 0, true);
            auto ctx_ced = make_ctx(model.get(), true,  128, 256, 1, 0, true);
            llama_ubatch_prepare(ctx_ref.get(), 200, false);
            llama_ubatch_prepare(ctx_ced.get(), 200, false);
            std::vector<float> l_ref, l_ced;
            feed_tokens_split(ctx_ref.get(), toks200, 0, 0, { 128, 200 }, true, l_ref, n_vocab);
            feed_tokens_split(ctx_ced.get(), toks200, 0, 0, { 128, 200 }, true, l_ced, n_vocab);
            assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "14a deferred split");
        }

        // 14b & 14c: Single-call P=100 (> ced_tail_cap) + short follow-up 16 + decode steps
        {
            auto ctx_ref = make_ctx(model.get(), false, 128, 256, 1, 0, true);
            auto ctx_ced = make_ctx(model.get(), true,  128, 256, 1, 0, true);
            std::vector<float> l_ref, l_ced;
            feed_tokens_split(ctx_ref.get(), toks100, 0, 0, { 100 }, true, l_ref, n_vocab);
            feed_tokens_split(ctx_ced.get(), toks100, 0, 0, { 100 }, true, l_ced, n_vocab);
            assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "14b single-call P=100");

            feed_tokens_split(ctx_ref.get(), toks16, 100, 0, { 16 }, true, l_ref, n_vocab);
            feed_tokens_split(ctx_ced.get(), toks16, 100, 0, { 16 }, true, l_ced, n_vocab);
            assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "14c short follow-up");
            check_decode_steps(ctx_ref.get(), ctx_ced.get(), 116, 8, 0, n_vocab, tol, "14c decode");
        }

        // 14d: Multi-output Branch-3 fallback (n_tokens=100 > ced_full_n_ubatch)
        {
            auto ctx_ref = make_ctx(model.get(), false, 128, 256, 1, 0, true);
            auto ctx_ced = make_ctx(model.get(), true,  128, 256, 1, 0, true);
            for (auto * ctx : { ctx_ref.get(), ctx_ced.get() }) {
                common_batch batch(ctx);
                for (int i = 0; i < 100; ++i) {
                    batch.add(toks100[i], i, 0, i == 10 || i == 50 || i == 99);
                }
                GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
            }
            assert_vec_close(llama_get_logits_ith(ctx_ref.get(), 10), llama_get_logits_ith(ctx_ced.get(), 10), n_vocab, tol, "14d multi-out 10");
            assert_vec_close(llama_get_logits_ith(ctx_ref.get(), 50), llama_get_logits_ith(ctx_ced.get(), 50), n_vocab, tol, "14d multi-out 50");
            assert_vec_close(llama_get_logits_ith(ctx_ref.get(), 99), llama_get_logits_ith(ctx_ced.get(), 99), n_vocab, tol, "14d multi-out 99");
        }

        // 14e: DSpark tail layer_inp + split reserve + llama_ubatch_measure
        {
            auto ctx_ref = make_ctx(model.get(), false, 128, 256, 1, 0, true);
            auto ctx_ced = make_ctx(model.get(), true,  128, 256, 1, 0, true);
            llama_set_embeddings_layer_inp(ctx_ref.get(), 9, true);
            llama_set_embeddings_layer_inp(ctx_ced.get(), 9, true);
            llama_set_embeddings_layer_inp_tail(ctx_ced.get(), 8);

            ggml_backend_dev_t mdevs[8] = {};
            size_t msizes[8] = {};
            const int32_t nm = llama_ubatch_measure(ctx_ced.get(), 256, 128, mdevs, msizes, 8);
            GGML_ASSERT(nm > 0);

            std::vector<float> l_ref, l_ced;
            feed_tokens_split(ctx_ref.get(), toks200, 0, 0, { 128, 200 }, true, l_ref, n_vocab);
            feed_tokens_split(ctx_ced.get(), toks200, 0, 0, { 128, 200 }, true, l_ced, n_vocab);
            assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "14e dspark logits");
            llama_pos pos_beg = -1;
            int32_t   row_beg = -1;
            int32_t   n_rows  = -1;
            GGML_ASSERT(llama_ced_layer_inp_status(ctx_ced.get(), 0, &pos_beg, &row_beg, &n_rows) == 1);
            GGML_ASSERT(pos_beg == 192 && row_beg == 0 && n_rows == 8);
            const float * li_ref = llama_get_embeddings_layer_inp(ctx_ref.get(), 9);
            const float * li_ced = llama_get_embeddings_layer_inp(ctx_ced.get(), 9);
            assert_vec_close(li_ref + (size_t) (192 - 128) * n_embd, li_ced + (size_t) row_beg * n_embd, (size_t) n_rows * n_embd, tol, "14e dspark layer_inp rows");
        }

        // 14f: Non-causal toggle (causal -> non-causal -> causal)
        {
            auto ctx_ref = make_ctx(model.get(), false, 128, 256, 1, 0, true);
            auto ctx_ced = make_ctx(model.get(), true,  128, 256, 1, 0, true);
            llama_set_causal_attn(ctx_ref.get(), false);
            llama_set_causal_attn(ctx_ced.get(), false);
            std::vector<float> l_ref, l_ced;
            feed_tokens_split(ctx_ref.get(), toks32, 0, 0, { 32 }, true, l_ref, n_vocab);
            feed_tokens_split(ctx_ced.get(), toks32, 0, 0, { 32 }, true, l_ced, n_vocab);
            assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "14f non-causal");

            llama_set_causal_attn(ctx_ref.get(), true);
            llama_set_causal_attn(ctx_ced.get(), true);
            feed_tokens_split(ctx_ref.get(), toks16, 32, 0, { 16 }, true, l_ref, n_vocab);
            feed_tokens_split(ctx_ced.get(), toks16, 32, 0, { 16 }, true, l_ced, n_vocab);
            assert_vec_close(l_ref.data(), l_ced.data(), n_vocab, tol, "14f back to causal");
        }

        printf("[PASS] Scenario 14: Split CED reserve (enc_ub > dec_ub), fallbacks, non-causal toggle & ubatch_measure\n");
    }

    printf("  [%s summary] worst max_abs=%.3e, worst NMSE=%.3e\n", dev_name, g_worst_abs, g_worst_nmse);
}

int main(int argc, char ** argv) {
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);

    setvbuf(stdout, nullptr, _IONBF, 0);
    common_log_set_verbosity_thold(LOG_LEVEL_ERROR);
    common_init();
    llama_backend_init();
    init_normal_pool(42, 0.1f);

    std::vector<std::pair<ggml_backend_dev_t, std::string>> devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU || type == GGML_BACKEND_DEVICE_TYPE_GPU) {
            devs.emplace_back(dev, ggml_backend_dev_description(dev));
        }
    }

    // Run CPU first, plus CUDA/GPU if present
    for (const auto & [dev, name] : devs) {
        const bool is_gpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
        run_suite_for_device(dev, name.c_str(), is_gpu);
    }

    printf("All DeepSeek-V4.1 CED exactness tests PASSED!\n");
    llama_backend_free();
    return 0;
}
