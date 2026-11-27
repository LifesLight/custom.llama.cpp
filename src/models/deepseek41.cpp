#include "models.h"
#include "llama-impl.h"

#include "llama-kv-cache-dsv4.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

// ref: deepseek-ai/DeepSeek-V4.1-Flash-1017 inference/model.py, engram.py
// LLAMA_DSV41_SPARSE=0 disables the sparse attention path (see build_attention41)

static constexpr int64_t DSV41_ENGRAM_DEAD = -1;

static bool dsv41_env_on(const char * name) {
    const char * v = getenv(name);
    return !v || atoi(v) != 0;
}

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    llama_model_deepseek4::load_arch_hparams(ml);

    hparams.n_embd_out_impl = hparams.n_embd; // no hc_head
    if (hparams.indexer_top_k == 0) {
        throw std::runtime_error("DeepSeek-V4.1 indexer top_k must be > 0");
    }

    // each compressed layer reads the KV / top-k of the latest source layer at or before it
    std::vector<uint32_t> kv_src;
    std::vector<uint32_t> idx_src;
    ml.get_arr(LLM_KV_ATTENTION_KV_SOURCE_LAYERS,      kv_src);
    ml.get_arr(LLM_KV_ATTENTION_INDEXER_SOURCE_LAYERS, idx_src);

    const uint32_t n_layer = hparams.n_layer();
    const auto out_of_range = [&](const std::vector<uint32_t> & v) {
        return std::any_of(v.begin(), v.end(), [&](uint32_t il) { return il >= n_layer; });
    };
    if (out_of_range(kv_src) || out_of_range(idx_src)) {
        throw std::runtime_error("DeepSeek-V4.1 kv/index source layer out of range");
    }
    hparams.dsv41_kv_src.fill(-1);
    hparams.dsv41_topk_src.fill(-1);
    int32_t last_kv  = -1;
    int32_t last_idx = -1;
    for (uint32_t il = 0; il < n_layer; ++il) {
        const uint32_t ratio = hparams.dsv4_compress_ratios[il];
        const bool is_kv  = std::find(kv_src.begin(),  kv_src.end(),  il) != kv_src.end();
        const bool is_idx = std::find(idx_src.begin(), idx_src.end(), il) != idx_src.end();
        if ((is_kv || is_idx) && ratio == 0) {
            throw std::runtime_error(format("DeepSeek-V4.1 source layer %u has compress ratio 0", il));
        }
        last_kv  = is_kv  ? (int32_t) il : last_kv;
        last_idx = is_idx ? (int32_t) il : last_idx;
        if (ratio == 0) {
            continue;
        }
        if (ratio > 2 || last_kv < 0 || last_idx < last_kv ||
                hparams.dsv4_compress_ratios[last_kv] != ratio || hparams.dsv4_compress_ratios[last_idx] != ratio) {
            throw std::runtime_error(format("DeepSeek-V4.1 layer %u: ratio %u without a matching kv/index source", il, ratio));
        }
        hparams.dsv41_kv_src[il]   = last_kv;
        hparams.dsv41_topk_src[il] = last_idx;
    }

    ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, hparams.dsv41_cand_src, false);
    if (hparams.dsv41_cand_src >= 0) {
        ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, hparams.dsv41_cand_topk_blocks);
        ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,   hparams.dsv41_cand_block_size);
        const int32_t cs = hparams.dsv41_cand_src;
        if ((uint32_t) cs >= n_layer || hparams.dsv41_topk_src[cs] != cs || hparams.dsv41_cand_block_size == 0 ||
                hparams.dsv41_cand_topk_blocks == 0) {
            throw std::runtime_error("DeepSeek-V4.1 invalid candidate selection (source must be an index source, block size and top-k blocks > 0)");
        }
        for (uint32_t il = cs + 1; il < n_layer; ++il) {
            if (hparams.dsv41_topk_src[il] == (int32_t) il && hparams.dsv4_compress_ratios[il] != hparams.dsv4_compress_ratios[cs]) {
                throw std::runtime_error("DeepSeek-V4.1 candidate consumers must share the source's compress ratio");
            }
        }
    }

    // Engram: the PLE n-gram hash of qwen4exp with one set of hash constants per Engram layer, stored back to back
    std::vector<uint32_t> e_layers;
    ml.get_arr(LLM_KV_PLE_LAYERS, e_layers, false);
    if (!e_layers.empty()) {
        std::vector<uint64_t> mult;
        std::vector<uint64_t> offsets;
        std::vector<uint64_t> primes;
        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,             hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM,        hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.ple_head_dim);
        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS,      mult);
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,           offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES,       primes);
        ml.get_key(LLM_KV_ENGRAM_PAD_TOKEN_ID,        hparams.engram_pad_id);
        ml.get_arr(LLM_KV_ENGRAM_TOKEN_MAP,           engram_token_map);

        const size_t n_e    = e_layers.size();
        const size_t n_gram = hparams.ple_ngram_size;
        const size_t n_cols = n_gram >= 2 ? (n_gram - 1)*hparams.ple_heads_per_ngram : 0;
        if (n_cols == 0 || hparams.ple_head_dim == 0 ||
                mult.size() != n_e*n_gram || primes.size() != n_e*n_cols || offsets.size() != n_e*n_cols ||
                mult.size() > LLAMA_MAX_PLE_NGRAM || primes.size() > LLAMA_MAX_PLE_HEADS ||
                hparams.engram_pad_id >= engram_token_map.size()) {
            throw std::runtime_error("DeepSeek-V4.1 engram metadata is inconsistent");
        }
        for (size_t e = 0; e < n_e; ++e) {
            // module e belongs to the e-th Engram layer
            if (e_layers[e] >= n_layer || (e > 0 && e_layers[e] <= e_layers[e - 1])) {
                throw std::runtime_error(format("DeepSeek-V4.1 engram layer %u is out of range or out of order", e_layers[e]));
            }
            hparams.is_ple_impl.set(e_layers[e]);
        }
        for (size_t c = 0; c < n_e*n_cols; ++c) {
            // a bucket count of 0 traps in the hash; row ids are int32
            if (primes[c] < 2 || offsets[c] + primes[c] > (uint64_t) INT32_MAX) {
                throw std::runtime_error(format("DeepSeek-V4.1 engram head %zu: bucket range out of range", c));
            }
            hparams.ple_head_offsets[c]     = (uint32_t) offsets[c];
            hparams.ple_head_vocab_sizes[c] = (uint32_t) primes[c];
        }
        std::copy(mult.begin(), mult.end(), hparams.ple_layer_multipliers.begin());
        hparams.ple_n_heads   = (uint32_t) n_cols;
        hparams.ple_n_modules = (uint32_t) n_e;
    }

    // Causal Encoder-Decoder (CED) split: largest kv source layer L_dec such that all layers >= L_dec
    // read global KV exclusively from L_dec (projected from encoder output), use no Engram, and have
    // no cross-boundary top-k or candidate dependency back to encoder layers < L_dec.
    hparams.dsv41_ced_dec_layer = -1;
    int32_t l_dec = -1;
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (hparams.dsv41_kv_src[il] == (int32_t) il) {
            l_dec = (int32_t) il;
        }
    }
    if (l_dec > 0 && l_dec < (int32_t) n_layer &&
            hparams.n_swa > 0 && hparams.swa_type == LLAMA_SWA_TYPE_STANDARD && hparams.n_pos_per_embd() == 1 &&
            (hparams.dsv41_cand_src < 0 || hparams.dsv41_cand_src >= l_dec)) {
        bool ced_ok = true;
        for (uint32_t il = (uint32_t) l_dec; il < n_layer; ++il) {
            if (hparams.dsv4_compress_ratios[il] != 1 ||
                    hparams.dsv41_kv_src[il] != l_dec ||
                    hparams.dsv41_topk_src[il] < l_dec ||
                    hparams.is_ple(il)) {
                ced_ok = false;
                break;
            }
        }
        if (ced_ok) {
            hparams.dsv41_ced_dec_layer = l_dec;
        }
    }
}

void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t n_embd_head     = hparams.n_embd_head_k();
    const int64_t o_groups        = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank     = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult         = hparams.dsv4_hc_mult;
    const int64_t hc_dim          = hc_mult * n_embd;
    const int64_t hc_mix_dim      = (2 + hc_mult) * hc_mult;
    const int64_t n_idx_embd      = hparams.indexer_head_size;
    const int64_t n_idx_head      = hparams.indexer_n_head;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    // all Engram tables in one lazily read tensor, like the qwen4exp n-gram table
    if (hparams.ple_n_heads > 0) {
        int64_t n_rows = 0;
        for (size_t c = 0; c < (size_t) hparams.ple_n_modules*hparams.ple_n_heads; ++c) {
            n_rows = std::max(n_rows, (int64_t) hparams.ple_head_offsets[c] + hparams.ple_head_vocab_sizes[c]);
        }
        const std::string name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * w = ml.get_weight(name.c_str())) {
            if (w->tensor->ne[1] < n_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the Engram heads (%" PRId64 ")",
                                                name.c_str(), w->tensor->ne[1], n_rows));
            }
            n_rows = w->tensor->ne[1];
        }
        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"), {(int64_t) hparams.ple_head_dim, n_rows}, TENSOR_READ_LAZY);
    }

    // an FP8 overlay (convert_deepseek41_fp8_overlay.py) gives a weight the checkpoint's E4M3 values and one F32 scale
    // per bs x bs block (bs = 32 in DeepSeek-V4.1), computed by ggml_mul_mat_fp8_block; groups > 1: wo_a, whose
    // [k, n/groups, groups] view takes the scales of its n/groups rows per group
    auto create_fp8 = [&](llm_tensor tid, int i, int64_t k, int64_t n, int64_t groups = 1) {
        ggml_tensor * w = groups > 1
            ? create_tensor(tn(tid, "weight", i), {k, n/groups, groups}, TENSOR_ALLOW_RESHAPE)
            : create_tensor(tn(tid, "weight", i), {k, n}, 0);
        const auto scale_name = tn(tid, "fp8_block_scale", i);
        const ggml_tensor * meta = ml.get_tensor_meta(scale_name.str().c_str());
        if (!meta) {
            return w;
        }
        if (split_mode() == LLAMA_SPLIT_MODE_ROW || split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("DeepSeek-V4.1 block-scaled FP8 supports none and layer split modes");
        }
        const int64_t bs = meta->ne[0] > 0 && k % meta->ne[0] == 0 ? k / meta->ne[0] : 0;
        if ((w && w->type != GGML_TYPE_F8_E4M3) || meta->type != GGML_TYPE_F32 ||
            bs == 0 || bs % 32 != 0 || bs > GGML_FP8_BLOCK_SIZE || (n/groups) % bs != 0) {
            throw std::runtime_error(format("invalid DeepSeek-V4.1 block-scaled FP8 tensor %s", scale_name.str().c_str()));
        }
        ggml_tensor * scale = groups > 1
            ? create_tensor(scale_name, {k/bs, n/groups/bs, groups}, TENSOR_ALLOW_RESHAPE)
            : create_tensor(scale_name, {k/bs, n/bs}, 0);
        if (w && scale) {
            fp8_block_scales.emplace(w, scale);
        }
        return w;
    };

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
        layer.wq_a          = create_fp8(LLM_TENSOR_ATTN_Q_A, i, n_embd, q_lora_rank);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_fp8(LLM_TENSOR_ATTN_Q_B, i, q_lora_rank, n_head * n_embd_head);
        layer.wkv           = create_fp8(LLM_TENSOR_ATTN_KV,  i, n_embd, n_embd_head);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
        layer.wo_a          = create_fp8(LLM_TENSOR_ATTN_OUT_A, i, n_head * n_embd_head / o_groups, o_lora_rank * o_groups, o_groups);
        layer.wo_b          = create_fp8(LLM_TENSOR_ATTN_OUT_B, i, o_groups * o_lora_rank, n_embd);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        if (hparams.dsv41_kv_src[i] == i) {
            layer.attn_comp_wkv  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,  "weight", i), {n_embd, n_embd_head}, 0);
            layer.attn_comp_norm = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM, "weight", i), {n_embd_head}, 0);
            if (hparams.dsv4_compress_ratios[i] > 1) {
                layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, n_embd_head}, 0);
            }
            layer.indexer_attn_k = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K, "weight", i), {n_embd_head, n_idx_embd}, 0);
            layer.indexer_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", i), {n_idx_embd}, 0);
        }
        if (hparams.dsv41_topk_src[i] == i) {
            layer.indexer_attn_q_b = create_fp8(LLM_TENSOR_INDEXER_ATTN_Q_B, i, q_lora_rank, n_idx_head * n_idx_embd);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, n_idx_head}, 0);
        }

        if (engram_index(i) >= 0) {
            layer.engram_kv   = create_fp8(LLM_TENSOR_ENGRAM_KV, i, (int64_t) hparams.ple_n_heads * hparams.ple_head_dim, n_embd * (hc_mult + 1));
            layer.engram_gate = create_tensor(tn(LLM_TENSOR_ENGRAM_GATE, "weight", i), {n_embd, hc_mult}, 0);
        }

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,        "weight", i), {n_embd}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp = create_fp8(LLM_TENSOR_FFN_GATE_SHEXP, i, n_embd, n_ff_exp * n_expert_shared);
        layer.ffn_down_shexp = create_fp8(LLM_TENSOR_FFN_DOWN_SHEXP, i, n_ff_exp * n_expert_shared, n_embd);
        layer.ffn_up_shexp   = create_fp8(LLM_TENSOR_FFN_UP_SHEXP,   i, n_embd, n_ff_exp * n_expert_shared);
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        throw std::runtime_error("DeepSeek-V4.1 has no MTP graph: its DSpark draft is a separate DFLASH GGUF "
                "(convert_hf_to_gguf.py --dspark), used with --spec-type draft-dspark");
    }
    return std::make_unique<graph>(params);
}

// Engram n-gram hash (engram.py NgramHashState.forward)
int llama_model_deepseek41::engram_index(int il) const {
    if (hparams.ple_n_heads == 0 || !hparams.is_ple(il)) {
        return -1;
    }
    int e = 0;
    for (int j = 0; j < il; ++j) {
        e += hparams.is_ple(j);
    }
    return e;
}

void llama_model_deepseek41::engram_hash(int e, const llama_ubatch & ubatch, const std::vector<llama_token> & prev, std::vector<int32_t> & out) const {
    const int64_t n_gram = hparams.ple_ngram_size;
    const int64_t n_prev = n_gram - 1;
    const int64_t n_head = hparams.ple_heads_per_ngram;
    const int64_t n_cols = hparams.ple_n_heads;

    const uint64_t * mult   = hparams.ple_layer_multipliers.data() + (size_t) e*n_gram;
    const uint32_t * primes = hparams.ple_head_vocab_sizes.data()  + (size_t) e*n_cols;
    const int64_t    pad    = engram_token_map[hparams.engram_pad_id];
    const uint32_t * offs   = hparams.ple_head_offsets.data()      + (size_t) e*n_cols;

    const auto cid = [&](llama_token t) -> int64_t {
        return t < 0 || t >= (llama_token) engram_token_map.size() ? DSV41_ENGRAM_DEAD : engram_token_map[t];
    };

    out.resize(ubatch.n_tokens*n_cols);
    std::vector<uint64_t> toks(n_gram);
    for (int64_t i = 0; i < ubatch.n_tokens; ++i) {
        // shift 0 is the token itself; once a shift is blocked (sequence start or dead token) every longer one is too
        const bool is_embd = !ubatch.token || (ubatch.is_mixed() && ubatch.type[i]);
        int64_t c = is_embd ? DSV41_ENGRAM_DEAD : cid(ubatch.token[i]);
        bool blocked = c == DSV41_ENGRAM_DEAD;
        toks[0] = blocked ? pad : c;
        for (int64_t s = 1; s < n_gram; ++s) {
            c = ubatch.pos[i] - s < 0 ? DSV41_ENGRAM_DEAD : cid(prev[i*n_prev + (n_prev - s)]);
            blocked = blocked || c == DSV41_ENGRAM_DEAD;
            toks[s] = blocked ? pad : c;
        }

        uint64_t rolling = toks[0]*mult[0];
        for (int64_t o = 1; o < n_gram; ++o) {
            rolling ^= toks[o]*mult[o];
            for (int64_t h = 0; h < n_head; ++h) {
                const int64_t col = (o - 1)*n_head + h;
                out[i*n_cols + col] = (int32_t) (rolling % primes[col] + offs[col]);
            }
        }
    }
}

// cand_pin: +1e30 on the block with the newest compressed row of each query, so a partly filled block is always kept; the reference uses +inf, but a finite pin keeps fully masked blocks at -inf instead of NaN
class llm_graph_input_dsv41 : public llm_graph_input_i {
public:
    llm_graph_input_dsv41(const llama_model_deepseek41 & model, const llama_kv_cache_dsv4_context * mctx) :
        model(model), mctx(mctx) {}

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    std::vector<ggml_tensor *> engram_ids;
    ggml_tensor * engram_keep = nullptr;    // 0 for embedding (image) tokens; only in mixed ubatches
    ggml_tensor * router_bias_ids = nullptr;
    ggml_tensor * cand_pin = nullptr;
    ggml_tensor * ced_inp_h   = nullptr;    // DEC_ONLY: [n_embd, hc_mult, n_tokens] F32
    ggml_tensor * ced_inp_pre = nullptr;    // DEC_ONLY: [hc_mult, n_tokens] F32
    ggml_tensor * ced_out_ids = nullptr;    // [n_outputs] I32 relative to ced_in_start
    bool cand_csa = false;

    llama_dsv4_ced_mode ced_mode = LLAMA_DSV4_CED_FULL;
    std::vector<int32_t> ced_k_start;
    std::vector<int32_t> ced_o_start;
    bool     ced_dec_has_vl    = false;
    bool     ced_dec_all_vl    = false;
    int32_t  ced_in_start      = 0;
    int32_t  ced_li_start      = -1;
    uint32_t ced_tail_keep_max = 0;
    int64_t  ced_n_outputs     = -1;

    const llama_model_deepseek41 & model;
    const llama_kv_cache_dsv4_context * mctx;

private:
    const llama_kv_cache_dsv4_context::comp_plan & cand_plan(const llama_ubatch & ubatch) const {
        return cand_csa ? mctx->get_csa_plan(ubatch) : mctx->get_hca_plan(ubatch);
    }
};

void llm_graph_input_dsv41::set_input(const llama_ubatch * ubatch) {
    if (!engram_ids.empty()) {
        // a token shared by several sequences has the same history in each, so its first sequence resolves it
        std::vector<llama_token> prev;
        mctx->get_raw()->get_prev_tokens(*ubatch, model.hparams.ple_ngram_size - 1, prev);
        // hash all Engram layers first so a lazily read table gets one batched readahead before the first gather
        std::vector<std::vector<int32_t>> ids(engram_ids.size());
        std::vector<int32_t> all;
        for (size_t e = 0; e < engram_ids.size(); ++e) {
            model.engram_hash((int) e, *ubatch, prev, ids[e]);
            all.insert(all.end(), ids[e].begin(), ids[e].end());
        }
        if (model.can_prefetch.count(model.per_layer_tok_embd)) {
            llama_prefetch_rows(model.per_layer_tok_embd, all.data(), all.size());
        }
        for (size_t e = 0; e < engram_ids.size(); ++e) {
            if (engram_ids[e]->buffer) {
                ggml_backend_tensor_set(engram_ids[e], ids[e].data(), 0, ids[e].size()*sizeof(int32_t));
            }
        }
    }

    // embedding (image) tokens are hashed DEAD and get no Engram contribution (gate 0)
    if (engram_keep && engram_keep->buffer) {
        std::vector<float> keep(ubatch->n_tokens);
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            const bool is_embd = !ubatch->token || (ubatch->is_mixed() && ubatch->type[i]);
            keep[i] = is_embd ? 0.0f : 1.0f;
        }
        ggml_backend_tensor_set(engram_keep, keep.data(), 0, keep.size()*sizeof(float));
    }

    const auto & ced = mctx->get_ced_plan();

    if (ced_inp_h && ced_inp_h->buffer) {
        GGML_ASSERT(ced.dec_h_enc != nullptr && ced.dec_pre_prev != nullptr);
        ggml_backend_tensor_set(ced_inp_h,   ced.dec_h_enc,    0, ggml_nbytes(ced_inp_h));
        ggml_backend_tensor_set(ced_inp_pre, ced.dec_pre_prev, 0, ggml_nbytes(ced_inp_pre));
    }

    if (router_bias_ids && router_bias_ids->buffer) {
        if (ced.mode == LLAMA_DSV4_CED_DEC_ONLY) {
            GGML_ASSERT(ced.dec_is_embd != nullptr);
            std::vector<int32_t> ids(ubatch->n_tokens);
            for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
                ids[i] = ced.dec_is_embd[i];
            }
            ggml_backend_tensor_set(router_bias_ids, ids.data(), 0, ggml_nbytes(router_bias_ids));
        } else {
            std::vector<int32_t> ids(ubatch->type, ubatch->type + ubatch->n_tokens);
            ggml_backend_tensor_set(router_bias_ids, ids.data(), 0, ggml_nbytes(router_bias_ids));
        }
    }

    if (ced_out_ids && ced_out_ids->buffer) {
        std::vector<int32_t> oids;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            if (ubatch->output[i]) {
                GGML_ASSERT((int32_t) i >= ced_in_start);
                oids.push_back((int32_t) i - ced_in_start);
            }
        }
        GGML_ASSERT((int64_t) oids.size() == ced_out_ids->ne[0]);
        ggml_backend_tensor_set(ced_out_ids, oids.data(), 0, oids.size()*sizeof(int32_t));
    }

    if (cand_pin && cand_pin->buffer) {
        const auto & plan = cand_plan(*ubatch);
        const int64_t n_blk = cand_pin->ne[0];
        const int64_t bs    = model.hparams.dsv41_cand_block_size;
        GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) ubatch->n_tokens);
        std::vector<float> pin(n_blk*ubatch->n_tokens, 0.0f);
        for (int64_t i = 0; i < (int64_t) ubatch->n_tokens; ++i) {
            const int64_t nv = plan.n_visible[i];
            if (nv > 0) {
                GGML_ASSERT((nv - 1)/bs < n_blk);
                pin[i*n_blk + (nv - 1)/bs] = 1e30f;
            }
        }
        ggml_backend_tensor_set(cand_pin, pin.data(), 0, pin.size()*sizeof(float));
    }
}

bool llm_graph_input_dsv41::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);
    const auto & ced = mctx->get_ced_plan();

    if (ced.mode != ced_mode || ced.k_start != ced_k_start || ced.o_start != ced_o_start ||
            ced.dec_has_vl != ced_dec_has_vl || ced.dec_all_vl != ced_dec_all_vl ||
            ced.li_start != ced_li_start || ced.tail_keep_max != ced_tail_keep_max ||
            ced_n_outputs != (int64_t) params.n_outputs) {
        return false;
    }

    const bool want_rbi = ced.mode == LLAMA_DSV4_CED_DEC_ONLY
        ? (ced.dec_has_vl && !ced.dec_all_vl)
        : params.ubatch.is_mixed();
    if ((router_bias_ids != nullptr) != want_rbi) {
        return false;
    }
    if (router_bias_ids && router_bias_ids->ne[0] != (int64_t) params.ubatch.n_tokens) {
        return false;
    }
    if ((ced_inp_h != nullptr) != (ced.mode == LLAMA_DSV4_CED_DEC_ONLY)) {
        return false;
    }
    if (ced_inp_h && ced_inp_h->ne[2] != (int64_t) params.ubatch.n_tokens) {
        return false;
    }
    if (ced_out_ids && ced_out_ids->ne[0] != (int64_t) params.n_outputs) {
        return false;
    }

    bool res = true;
    for (auto * t : engram_ids) {
        res &= t->ne[0] == (int64_t) model.hparams.ple_n_heads*params.ubatch.n_tokens;
    }
    res &= (engram_keep != nullptr) == (!engram_ids.empty() && params.ubatch.embd != nullptr);
    if (engram_keep) {
        res &= engram_keep->ne[2] == (int64_t) params.ubatch.n_tokens;
    }
    if (cand_pin) {
        const auto & plan = cand_plan(params.ubatch);
        const int64_t bs = model.hparams.dsv41_cand_block_size;
        res &= plan.n_kv > 0 && cand_pin->ne[0] == (plan.n_kv + bs - 1)/bs;
        res &= cand_pin->ne[1]*cand_pin->ne[3] == (int64_t) params.ubatch.n_tokens;
    }
    return res;
}

ggml_tensor * llama_model_deepseek41::graph::build_engram(
        const llama_model_deepseek41 & model,
        ggml_tensor * x,
        ggml_tensor * rows,
        ggml_tensor * keep,
        int il) const {
    const auto & layer = model.layers[il];
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    GGML_ASSERT(rows->ne[0] == (int64_t) (hparams.ple_head_dim*hparams.ple_n_heads) && rows->ne[1] == nt);

    ggml_tensor * kv = build_lora_mm(layer.engram_kv, rows);
    cb(kv, "engram_kv", il);

    ggml_tensor * key   = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, n_embd, hc, nt, ggml_row_size(kv->type, n_embd), kv->nb[1], 0));
    ggml_tensor * value = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, n_embd, 1,  nt, ggml_row_size(kv->type, n_embd), kv->nb[1],
            ggml_row_size(kv->type, hc*n_embd)));

    // quantized copy has no binary-op kernel
    ggml_tensor * egate = ggml_is_quantized(layer.engram_gate->type) ? ggml_cast(ctx0, layer.engram_gate, GGML_TYPE_F32) : layer.engram_gate;
    ggml_tensor * dot = ggml_mul(ctx0, ggml_mul(ctx0, ggml_rms_norm(ctx0, x, norm_rms_eps), ggml_rms_norm(ctx0, key, norm_rms_eps)), egate);
    dot = ggml_scale(ctx0, ggml_sum_rows(ctx0, dot), 1.0f/sqrtf((float) n_embd));
    cb(dot, "engram_dot", il);

    // sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)); sign(0) = +1 as in copysign
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, dot), 1e-6f, INFINITY));
    ggml_tensor * sign = ggml_scale_bias(ctx0, ggml_step(ctx0, ggml_neg(ctx0, dot)), -2.0f, 1.0f);
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, mag, sign));
    if (keep) {
        gate = ggml_mul(ctx0, gate, keep);
    }
    cb(gate, "engram_gate", il);

    return ggml_add(ctx0, x, ggml_mul(ctx0, ggml_repeat_4d(ctx0, value, n_embd, hc, nt, 1), gate));
}

void llama_model_deepseek41::graph::build_comp_kv_write(
        const llama_model_deepseek41 & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        int il) {
    const auto & layer = model.layers[il];
    const int64_t ratio = hparams.dsv4_compress_ratios[il];
    if (ratio == 0 || hparams.dsv41_kv_src[il] != il) {
        return;
    }

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_idx_embd       = hparams.indexer_head_size;

    const float   freq_scale_l     = freq_scale;
    const float   ext_factor_l     = ext_factor;
    const float   attn_factor_l    = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const auto rope = [&](ggml_tensor * t, ggml_tensor * pos, int64_t n_nope) {
        t = ggml_rope_ext(ctx0, t, pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
                hparams.dsv4_compress_rope_base, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast, beta_slow);
        return ggml_rope_set_offset(t, n_nope);
    };

    const auto & ci         = ratio == 2 ? inp_dsv4->get_csa()             : inp_dsv4->get_hca();
    const auto * comp_ctx   = ratio == 2 ? inp_dsv4->mctx->get_csa()       : inp_dsv4->mctx->get_hca();
    const auto * comp_state = ratio == 2 ? inp_dsv4->mctx->get_csa_state() : inp_dsv4->mctx->get_hca_state();

    if (!ci.state_pos) {
        return;
    }

    ggml_tensor * state_kv    = build_lora_mm(layer.attn_comp_wkv, cur);
    ggml_tensor * state_score = ratio > 1 ? build_lora_mm(layer.attn_comp_wgate, cur) : ggml_fill(ctx0, state_kv, 0.0f);

    ggml_tensor * restored_kv    = comp_state->get_kv_all(ctx0, il);
    ggml_tensor * restored_score = comp_state->get_score_all(ctx0, il);
    if (ci.state_restore_src_idxs && ci.state_restore_dst_idxs) {
        restored_kv    = comp_state->cpy_kv   (ctx0, ggml_get_rows(ctx0, restored_kv,    ci.state_restore_src_idxs), ci.state_restore_dst_idxs, il);
        restored_score = comp_state->cpy_score(ctx0, ggml_get_rows(ctx0, restored_score, ci.state_restore_src_idxs), ci.state_restore_dst_idxs, il);
    }

    if (ci.state_write_idxs) {
        const int64_t n_blocks = ci.state_write_pos->ne[0];
        const int64_t n_state  = comp_state->get_n_rows();
        GGML_ASSERT(ci.state_read_idxs->ne[0] == ratio*n_blocks);

        ggml_tensor * src_kv    = ggml_concat(ctx0, ggml_view_2d(ctx0, restored_kv,    restored_kv->ne[0],    n_state, restored_kv->nb[1],    0), state_kv,    1);
        ggml_tensor * src_score = ggml_concat(ctx0, ggml_view_2d(ctx0, restored_score, restored_score->ne[0], n_state, restored_score->nb[1], 0), state_score, 1);

        ggml_tensor * v = ggml_get_rows(ctx0, src_kv,    ci.state_read_idxs);
        ggml_tensor * s = ggml_get_rows(ctx0, src_score, ci.state_read_idxs);
        v = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, v, n_embd_head, ratio, n_blocks), 1, 0, 2, 3));
        s = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, s, n_embd_head, ratio, n_blocks), 1, 0, 2, 3));

        ggml_tensor * comp = ggml_sum_rows(ctx0, ggml_mul(ctx0, v, ggml_soft_max(ctx0, s)));
        comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
        comp = build_norm(comp, layer.attn_comp_norm, nullptr, LLM_NORM_RMS, il);
        cb(comp, "comp_latent", il);

        // index K from the RoPE-free latent; a latent stands for the first token of its group
        ggml_tensor * ik = build_norm(build_lora_mm(layer.indexer_attn_k, comp), layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
        ik = rope(ik, ci.state_write_pos, n_idx_embd - n_embd_head_rope);
        comp = rope(comp, ci.state_write_pos, n_embd_head_nope);

        ggml_tensor * row = ggml_concat(ctx0, comp, ik, 0);
        ggml_build_forward_expand(gf, comp_ctx->cpy_k(ctx0, row, ci.state_write_idxs, il));
        comp_row_of[il] = row;
    }

    if (ci.state_snapshot_src_idxs && ci.state_snapshot_dst_idxs) {
        ggml_tensor * snap_kv    = ggml_concat(ctx0, restored_kv,    state_kv,    1);
        ggml_tensor * snap_score = ggml_concat(ctx0, restored_score, state_score, 1);
        ggml_build_forward_expand(gf, comp_state->cpy_kv   (ctx0, ggml_get_rows(ctx0, snap_kv,    ci.state_snapshot_src_idxs), ci.state_snapshot_dst_idxs, il));
        ggml_build_forward_expand(gf, comp_state->cpy_score(ctx0, ggml_get_rows(ctx0, snap_score, ci.state_snapshot_src_idxs), ci.state_snapshot_dst_idxs, il));
    }
    ggml_build_forward_expand(gf, comp_state->cpy_kv   (ctx0, ggml_get_rows(ctx0, state_kv,    ci.state_persist_src_idxs), ci.state_persist_dst_idxs, il));
    ggml_build_forward_expand(gf, comp_state->cpy_score(ctx0, ggml_get_rows(ctx0, state_score, ci.state_persist_src_idxs), ci.state_persist_dst_idxs, il));
}

void llama_model_deepseek41::graph::build_comp_kv_mirrors(
        const llama_model_deepseek41 & model,
        llm_graph_input_dsv4 * inp_dsv4,
        int il_src) const {
    GGML_UNUSED(model);
    if (!comp_row_of[il_src]) {
        return;
    }
    for (int j = il_src + 1; j < n_layer; ++j) {
        if (hparams.dsv41_kv_src[j] != il_src || inp_dsv4->mctx->get_comp_read_layer(j) != j) {
            continue;
        }
        const int64_t r = hparams.dsv4_compress_ratios[j];
        const auto & cj = r == 2 ? inp_dsv4->get_csa() : inp_dsv4->get_hca();
        const auto * cc = r == 2 ? inp_dsv4->mctx->get_csa() : inp_dsv4->mctx->get_hca();
        ggml_build_forward_expand(gf, cc->cpy_k(ctx0, comp_row_of[il_src], cj.state_write_idxs, j));
    }
}

ggml_tensor * llama_model_deepseek41::graph::build_attention41(
        const llama_model_deepseek41 & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur_k,
        ggml_tensor * cur_o,
        ggml_tensor * pos_k,
        ggml_tensor * pos_o,
        ggml_tensor * k_idxs_dec,
        ggml_tensor * raw_mask_o,
        ggml_tensor * hca_mask_o,
        ggml_tensor * cand_pin_o,
        int32_t       ks,
        int32_t       os,
        bool          skip_comp_write,
        bool          kv_only,
        int           il) {
    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t n_idx_embd       = hparams.indexer_head_size;
    const int64_t n_idx_head       = hparams.indexer_n_head;
    const int64_t nk               = cur_k->ne[1];
    const float   kq_scale         = 1.0f/sqrtf(float(n_embd_head));
    const int64_t ratio            = hparams.dsv4_compress_ratios[il];

    const bool    comp_rope     = ratio != 0;
    const float   freq_base_l   = comp_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float   freq_scale_l  = comp_rope ? freq_scale : 1.0f;
    const float   ext_factor_l  = comp_rope ? ext_factor : 0.0f;
    const float   attn_factor_l = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float   beta_fast_l   = comp_rope ? beta_fast : 0.0f;
    const float   beta_slow_l   = comp_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l  = comp_rope ? n_ctx_orig : 0;

    const auto rope = [&](ggml_tensor * t, ggml_tensor * pos, int64_t n_nope) {
        t = ggml_rope_ext(ctx0, t, pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        return ggml_rope_set_offset(t, n_nope);
    };

    const auto & ci         = ratio == 2 ? inp_dsv4->get_csa()             : inp_dsv4->get_hca();
    const auto * comp_ctx   = ratio == 2 ? inp_dsv4->mctx->get_csa()       : inp_dsv4->mctx->get_hca();
    const auto * comp_state = ratio == 2 ? inp_dsv4->mctx->get_csa_state() : inp_dsv4->mctx->get_hca_state();

    const int32_t rd = inp_dsv4->mctx->get_comp_read_layer(il);

    if (ratio != 0 && !skip_comp_write) {
        GGML_ASSERT(hparams.dsv41_kv_src[il] >= 0 && (int64_t) comp_state->get_ratio() == ratio);
        GGML_ASSERT(ks == 0);
        build_comp_kv_write(model, inp_dsv4, cur_k, il);

        if (rd == il && hparams.dsv41_kv_src[il] != il && comp_row_of[hparams.dsv41_kv_src[il]]) {
            // first consumer on this device: append the source's new rows to the local mirror
            ggml_build_forward_expand(gf, comp_ctx->cpy_k(ctx0, comp_row_of[hparams.dsv41_kv_src[il]], ci.state_write_idxs, il));
        }
    }

    ggml_tensor * kv = build_norm(build_lora_mm(layer.wkv, cur_k), layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = rope(ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nk), pos_k, n_embd_head_nope);
    cb(kv, "kv", il);

    ggml_tensor * k_rot = inp_attn->self_k_rot[il];

    if (kv_only) {
        if (k_rot) {
            kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
        }
        ggml_build_forward_expand(gf, kv);
        ggml_tensor * k_idxs = k_idxs_dec ? k_idxs_dec : inp_attn->get_k_idxs();
        ggml_tensor * k_idxs_k = ks > 0 ? ggml_view_1d(ctx0, k_idxs, nk, (size_t) ks * ggml_element_size(k_idxs)) : k_idxs;
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, kv, k_idxs_k, il));
        return nullptr;
    }

    const int64_t nt = cur_o->ne[1];
    ggml_tensor * qr = build_norm(build_lora_mm(layer.wq_a, cur_o), layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr_norm", il);

    ggml_tensor * q = ggml_reshape_3d(ctx0, build_lora_mm(layer.wq_b, qr), n_embd_head, n_head, nt);
    q = rope(q, pos_o, n_embd_head_nope);
    cb(q, "q", il);

    ggml_tensor * out = nullptr;
    ggml_tensor * comp_mask = hca_mask_o ? hca_mask_o : ci.kq_mask;
    if (ratio == 0 || !comp_mask) {
        if (!k_idxs_dec && !raw_mask_o && ks == 0) {
            // no compressed row visible yet: window only (uses encoder kv_raw; build_raw_attention applies k_rot)
            out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks, kq_scale, il);
        } else {
            if (k_rot) {
                q  = llama_mul_mat_hadamard(ctx0, q,  k_rot);
                kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
            }
            ggml_build_forward_expand(gf, q);
            ggml_build_forward_expand(gf, kv);
            ggml_tensor * k_idxs   = k_idxs_dec ? k_idxs_dec : inp_attn->get_k_idxs();
            ggml_tensor * k_idxs_k = ks > 0 ? ggml_view_1d(ctx0, k_idxs, nk, (size_t) ks * ggml_element_size(k_idxs)) : k_idxs;
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, kv, k_idxs_k, il));
            ggml_tensor * raw_mask = raw_mask_o ? raw_mask_o : inp_attn->self_attn_mask.raw;
            ggml_tensor * k        = inp_attn->mctx->get_k(ctx0, il);
            out = build_attn_mha(q, k, k, nullptr, raw_mask, nullptr, layer.attn_sinks, nullptr, 0, kq_scale, il);
            if (k_rot) {
                out = llama_mul_mat_hadamard(ctx0, out, k_rot);
            }
            cb(out, "attn_raw", il);
        }
    } else {
        const int64_t n_comp = comp_mask->ne[0];

        ggml_tensor * comp_all = comp_ctx->get_k(ctx0, rd);
        GGML_ASSERT(comp_all->ne[0] == n_embd_head + n_idx_embd && n_comp <= comp_all->ne[2]);
        const int64_t n_stream = comp_all->ne[3];

        const int32_t topk_src = hparams.dsv41_topk_src[il];
        ggml_tensor * top_k = topk_of[topk_src];
        if (top_k && os > topk_o_start[topk_src]) {
            const int32_t diff = os - topk_o_start[topk_src];
            top_k = ggml_cont(ctx0, ggml_view_2d(ctx0, top_k, top_k->ne[0], nt, top_k->nb[1], (size_t) diff * top_k->nb[1]));
        }
        if (topk_src == il) {
            ggml_tensor * ik = ggml_view_4d(ctx0, comp_all, n_idx_embd, 1, n_comp, n_stream,
                    comp_all->nb[1], comp_all->nb[2], comp_all->nb[3], ggml_row_size(comp_all->type, n_embd_head));
            // the fused indexer reads the strided view in place, the unfused path needs it contiguous; CUDA has no same-type copy of quantized views, so dequantize them
            const bool fused_idx = sparse && cparams.fused_lid && comp_mask->type == GGML_TYPE_F16;
            if (ggml_is_quantized(ik->type)) {
                ik = ggml_cast(ctx0, ik, GGML_TYPE_F32);
            } else if (!fused_idx) {
                ik = ggml_cont(ctx0, ik);
            }

            ggml_tensor * iq = ggml_reshape_3d(ctx0, build_lora_mm(layer.indexer_attn_q_b, qr), n_idx_embd, n_idx_head, nt);
            iq = rope(iq, pos_o, n_idx_embd - n_embd_head_rope);
            iq = ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], nt/n_stream, n_stream, iq->nb[1], iq->nb[2], iq->nb[2]*(nt/n_stream), 0);

            ggml_tensor * iw = ggml_scale(ctx0, build_lora_mm(layer.indexer_proj, cur_o), 1.0f/sqrtf(float(n_idx_embd*n_idx_head)));
            iw = ggml_view_4d(ctx0, iw, iw->ne[0], nt/n_stream, 1, n_stream, iw->nb[1], iw->nb[1]*(nt/n_stream), iw->nb[1]*(nt/n_stream), 0);

            // chunk queries so the [n_ih, tokens, n_comp] scores stay near 2^26 elements (at most 64 chunks); the fused indexer runs as one chunk
            const int64_t n_tok = nt/n_stream;
            GGML_ASSERT(comp_mask->ne[1] == n_tok);
            const int64_t chunk = fused_idx ? n_tok :
                std::max({ (n_tok + 63)/64, (INT64_C(1) << 26)/(n_idx_head*n_comp*n_stream), INT64_C(1) });

            const int32_t cand_src = hparams.dsv41_cand_src;
            const bool    pick_blk = cand_src == il && cand_pin_o;              // level one: pick the candidate blocks
            const bool    use_blk  = cand_src >= 0 && cand_src < il && cand_sel; // level two: score inside them
            const int64_t bs       = hparams.dsv41_cand_block_size;
            const int64_t n_blk    = pick_blk || use_blk ? (n_comp + bs - 1)/bs : 0;
            GGML_ASSERT(!pick_blk || cand_pin_o->ne[0] == n_blk);
            GGML_ASSERT(!use_blk  || cand_n_comp == n_comp);

            ggml_tensor * cand_sel_cur = cand_sel;
            if (use_blk && os > cand_o_start) {
                const int32_t diff = os - cand_o_start;
                cand_sel_cur = ggml_view_4d(ctx0, cand_sel, cand_sel->ne[0], n_tok, 1, n_stream,
                        cand_sel->nb[1], cand_sel->nb[2], cand_sel->nb[3], (size_t) diff * cand_sel->nb[1]);
            }

            ggml_tensor * ik_p = fused_idx ? nullptr : ggml_permute(ctx0, ik, 0, 2, 1, 3);
            std::vector<ggml_tensor *> parts_top_k;
            std::vector<ggml_tensor *> parts_sel;
            for (int64_t t0 = 0; t0 < n_tok; t0 += chunk) {
                const int64_t nc = std::min(chunk, n_tok - t0);
                const auto rows = [&](ggml_tensor * t) {
                    return ggml_view_4d(ctx0, t, t->ne[0], nc, 1, n_stream, t->nb[1], t->nb[2], t->nb[3], t0*t->nb[1]);
                };
                ggml_tensor * score = nullptr;
                if (fused_idx) {
                    score = ggml_lightning_indexer(ctx0, iq, ik, iw, comp_mask);
                    res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
                } else {
                    ggml_tensor * iqc = ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], nc, n_stream, iq->nb[1], iq->nb[2], iq->nb[3], t0*iq->nb[2]);

                    ggml_tensor * kq = ggml_mul_mat(ctx0, ik_p, ggml_permute(ctx0, iqc, 0, 2, 1, 3));
                    kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));
                    score = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_relu(ctx0, kq), rows(iw)));
                    ggml_tensor * mask  = rows(comp_mask);
                    score = ggml_add(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3)),
                            mask->type == GGML_TYPE_F32 ? mask : ggml_cast(ctx0, mask, GGML_TYPE_F32));
                }
                cb(score, "idx_score", il);

                if (pick_blk) {
                    // best position per block, pin the newest block, keep the top blocks
                    ggml_tensor * s2 = ggml_reshape_2d(ctx0, score, n_comp, nc*n_stream);
                    if (n_blk*bs != n_comp) {
                        s2 = ggml_concat(ctx0, s2, ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_blk*bs - n_comp, nc*n_stream), -INFINITY), 0);
                    }
                    ggml_tensor * bmax = ggml_pool_1d(ctx0, s2, GGML_OP_POOL_MAX, bs, bs, 0);
                    bmax = ggml_add(ctx0, ggml_reshape_4d(ctx0, bmax, n_blk, nc, 1, n_stream), rows(cand_pin_o));
                    cb(bmax, "cand_block_score", il);
                    parts_sel.push_back(ggml_top_k(ctx0, bmax, std::min<int64_t>(hparams.dsv41_cand_topk_blocks, n_blk)));
                } else if (use_blk) {
                    // own scores, restricted to the source's candidate blocks: 0 inside them, -inf elsewhere
                    ggml_tensor * bsel = ggml_cont(ctx0, rows(cand_sel_cur));
                    const int64_t k_b = bsel->ne[0];
                    ggml_tensor * bm    = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_blk, nc*n_stream, 1), -INFINITY);
                    ggml_tensor * zeros = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, k_b, nc*n_stream, 1), 0.0f);
                    bm = ggml_set_rows(ctx0, bm, zeros, ggml_reshape_3d(ctx0, bsel, k_b, nc*n_stream, 1));
                    bm = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, bm, bs, n_blk, nc*n_stream, 1), bs*n_blk, nc*n_stream);
                    bm = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, bm, n_comp, nc*n_stream, bm->nb[1], 0)), n_comp, nc, 1, n_stream);
                    cb(bm, "cand_mask", il);
                    score = ggml_add(ctx0, score, bm);
                }

                parts_top_k.push_back(ggml_top_k(ctx0, score, std::min<int64_t>(n_comp, hparams.indexer_top_k)));
            }

            const auto join = [&](std::vector<ggml_tensor *> v) {
                while (v.size() > 1) {
                    std::vector<ggml_tensor *> w;
                    for (size_t i = 0; i < v.size(); i += 2) {
                        w.push_back(i + 1 < v.size() ? ggml_concat(ctx0, v[i], v[i + 1], 1) : v[i]);
                    }
                    v.swap(w);
                }
                return v[0];
            };
            if (pick_blk) {
                cand_sel     = join(parts_sel);
                cand_n_comp  = n_comp;
                cand_o_start = os;
            }
            top_k = ggml_cont(ctx0, join(parts_top_k));
            cb(top_k, "idx_top_k", il);
            topk_of[il]      = top_k;
            topk_o_start[il] = os;
        }
        GGML_ASSERT(top_k && "DeepSeek-V4.1: top-k source layer produced no top-k");

        // a quantized K cache stores the window Hadamard-rotated (as build_raw_attention)
        if (k_rot) {
            q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
            kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
        }
        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, kv);
        ggml_tensor * k_idxs = k_idxs_dec ? k_idxs_dec : inp_attn->get_k_idxs();
        ggml_tensor * k_idxs_k = ks > 0 ? ggml_view_1d(ctx0, k_idxs, nk, (size_t) ks * ggml_element_size(k_idxs)) : k_idxs;
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, kv, k_idxs_k, il));

        ggml_tensor * raw_k    = inp_attn->mctx->get_k(ctx0, il);
        ggml_tensor * raw_mask = raw_mask_o ? raw_mask_o : inp_attn->self_attn_mask.raw;

        const int64_t k_sel    = top_k->ne[0];
        const int64_t n_raw    = raw_k->ne[2];

        // few queries per stream: attend over [window | top-k rows of the query], one FA stream per query; same keys and mask as dense, different key order
        const bool gather = sparse && cparams.flash_attn && raw_mask->type == GGML_TYPE_F16 && comp_mask->type == GGML_TYPE_F16 &&
                ((n_stream == 1 && nt <= 32) || nt == n_stream);

        if (gather) {
            // ggml_top_k returns the rows in no fixed order and FA sums the keys in that order: sort them so the result is deterministic
            ggml_tensor *& sel = gather_sel_of[{topk_src, os}];
            if (!sel) {
                GGML_ASSERT(n_comp <= (INT64_C(1) << 24)); // row ids are exact in F32
                ggml_tensor * ids = ggml_reshape_2d(ctx0, top_k, k_sel, nt);
                ggml_tensor * ord = ggml_argsort(ctx0, ggml_cast(ctx0, ids, GGML_TYPE_F32), GGML_SORT_ORDER_ASC);
                sel = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, ids, 1, k_sel, nt), ord), k_sel, nt);
                cb(sel, "comp_top_k_sorted", il);
            }

            // built once per (storage, top-k, o_start) tuple
            ggml_tensor *& comp_k = gather_k_of[{rd, topk_src, os}];
            if (!comp_k) {
                if (n_stream == 1) {
                    ggml_tensor * src_rows = ggml_view_3d(ctx0, comp_all, n_embd_head, n_comp, 1, comp_all->nb[2], comp_all->nb[3], 0);
                    comp_k = ggml_get_rows(ctx0, src_rows, ggml_reshape_1d(ctx0, sel, k_sel*nt));
                } else {
                    ggml_tensor * src_rows = ggml_view_3d(ctx0, comp_all, n_embd_head, n_comp, n_stream, comp_all->nb[2], comp_all->nb[3], 0);
                    comp_k = ggml_get_rows(ctx0, src_rows, sel);
                }
                comp_k = ggml_reshape_4d(ctx0, comp_k, n_embd_head, 1, k_sel, nt); // F32 (get_rows dequantizes)
                if (k_rot) {
                    // quantized K cache: the window is stored rotated, the compressed rows are not
                    comp_k = llama_mul_mat_hadamard(ctx0, comp_k, k_rot);
                }
                if (comp_k->type != raw_k->type) {
                    comp_k = ggml_cast(ctx0, comp_k, raw_k->type); // exact: rows are raw_k->type values
                }
                cb(comp_k, "comp_k_sel", il);
            }

            if (n_stream == 1 && nt > 1) {
                if (ggml_is_quantized(raw_k->type)) {
                    // no REPEAT kernel for quantized types: byte-exact copies by doubling concats
                    ggml_tensor * rep = nullptr;
                    ggml_tensor * pw  = raw_k;
                    for (int64_t n = nt; n > 0; n >>= 1) {
                        if (n & 1) {
                            rep = rep ? ggml_concat(ctx0, rep, pw, 3) : pw;
                        }
                        if (n > 1) {
                            pw = ggml_concat(ctx0, pw, pw, 3);
                        }
                    }
                    raw_k = rep;
                } else {
                    raw_k = ggml_repeat_4d(ctx0, raw_k, n_embd_head, 1, n_raw, nt);
                }
            }
            ggml_tensor * k_all = ggml_concat(ctx0, raw_k, comp_k, 2);

            // window mask, then the comp mask at the selected rows (-inf where top-k picked an invisible row); one per (top-k source, o_start)
            ggml_tensor *& kq_mask = gather_mask_of[{topk_src, os}];
            if (!kq_mask) {
                ggml_tensor * sel_mask = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, comp_mask, 1, n_comp, nt), sel);
                sel_mask = ggml_cast(ctx0, ggml_reshape_4d(ctx0, sel_mask, k_sel, 1, 1, nt), GGML_TYPE_F16);
                kq_mask = ggml_concat(ctx0, ggml_reshape_4d(ctx0, raw_mask, raw_mask->ne[0], 1, 1, nt), sel_mask, 0);
                cb(kq_mask, "comp_kq_mask_sel", il);
            }

            out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, nullptr, layer.attn_sinks, nullptr, 0, kq_scale, il);
        } else {
            // dense attention over [window | all compressed rows], selection applied as a mask
            ggml_tensor * comp_k = ggml_view_4d(ctx0, comp_all, n_embd_head, 1, n_comp, n_stream,
                    comp_all->nb[1], comp_all->nb[2], comp_all->nb[3], 0);
            if (k_rot) {
                // compressed rows are stored unrotated: rotate in F32, then cast to the cache type
                comp_k = comp_k->type != GGML_TYPE_F32 ? ggml_cast(ctx0, comp_k, GGML_TYPE_F32) : ggml_cont(ctx0, comp_k);
                comp_k = llama_mul_mat_hadamard(ctx0, comp_k, k_rot);
                if (comp_k->type != raw_k->type) {
                    comp_k = ggml_cast(ctx0, comp_k, raw_k->type);
                }
            } else if (ggml_is_quantized(comp_k->type)) {
                // quantized cache without rotation (LLAMA_ATTN_ROT_DISABLE=1): via F32 for the same reason
                comp_k = ggml_cast(ctx0, ggml_cast(ctx0, comp_k, GGML_TYPE_F32), raw_k->type);
            } else {
                comp_k = comp_k->type != raw_k->type ? ggml_cast(ctx0, comp_k, raw_k->type) : ggml_cont(ctx0, comp_k);
            }
            cb(comp_k, "comp_k", il);
            ggml_tensor * k_all = ggml_concat(ctx0, raw_k, comp_k, 2);

            ggml_tensor * sel_mask = build_top_k_mask(comp_mask, top_k, "comp_top_k_mask", il);
            ggml_tensor * kq_mask  = ggml_concat(ctx0, raw_mask, sel_mask, 0);
            cb(kq_mask, "comp_kq_mask", il);

            // at most window + top-k visible columns per query: lets CUDA FA gather only those
            const int64_t n_kv_max = sparse ? std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + k_sel : 0;
            out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, nullptr, layer.attn_sinks, nullptr, n_kv_max, kq_scale, il);
        }
        if (k_rot) {
            out = llama_mul_mat_hadamard(ctx0, out, k_rot);
        }
        cb(out, "attn_comp", il);
    }

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    out = ggml_rope_ext_back(ctx0, out, pos_o, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    out = ggml_rope_set_offset(out, n_embd_head_nope);
    cb(out, "attn_derope", il);

    out = ggml_permute(ctx0, ggml_reshape_3d(ctx0, out, n_head/n_groups*n_embd_head, n_groups, nt), 0, 2, 1, 3);
    ggml_tensor * oa = nullptr;
    if (const auto fp8 = model.fp8_block_scales.find(layer.wo_a); fp8 != model.fp8_block_scales.end()) {
        oa = ggml_mul_mat_fp8_block(ctx0, layer.wo_a, ggml_cont(ctx0, out), fp8->second);
    } else {
        oa = ggml_mul_mat(ctx0, layer.wo_a, out);
    }
    cb(oa, "attn_wo_a", il);
    oa = ggml_cont_2d(ctx0, ggml_permute(ctx0, oa, 0, 2, 1, 3), o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

llama_model_deepseek41::graph::graph(const llm_graph_params & params) :
    llama_model_deepseek4::graph(params, no_build{}) {
    const auto & model = static_cast<const llama_model_deepseek41 &>(this->model);

    static const bool sparse_env = dsv41_env_on("LLAMA_DSV41_SPARSE");
    sparse = sparse_env;

    topk_of.assign(n_layer, nullptr);
    topk_o_start.assign(n_layer, 0);
    comp_row_of.assign(n_layer, nullptr);

    ggml_tensor * inp_pos = build_inp_pos();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    const auto & ced = inp_dsv4->mctx->get_ced_plan();
    const auto   ced_mode = ced.mode;
    if (inp_dsv4->get_raw()->self_attn_mask.raw) {
        ggml_build_forward_expand(gf, inp_dsv4->get_raw()->self_attn_mask.raw);
    }
    if (inp_dsv4->get_raw()->self_attn_mask_dec.raw) {
        ggml_build_forward_expand(gf, inp_dsv4->get_raw()->self_attn_mask_dec.raw);
    }

    auto inp41 = std::make_unique<llm_graph_input_dsv41>(model, inp_dsv4->mctx);
    inp41->ced_mode          = ced_mode;
    inp41->ced_k_start       = ced.k_start;
    inp41->ced_o_start       = ced.o_start;
    inp41->ced_dec_has_vl    = ced.dec_has_vl;
    inp41->ced_dec_all_vl    = ced.dec_all_vl;
    inp41->ced_li_start      = ced.li_start;
    inp41->ced_tail_keep_max = ced.tail_keep_max;
    inp41->ced_n_outputs     = n_outputs;

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL     = nullptr;
    ggml_tensor * pre_prev = nullptr;
    ggml_tensor * inp_out_ids = nullptr;

    std::vector<ggml_tensor *> engram_rows;
    ggml_tensor * engram_keep = nullptr;
    if (ced_mode != LLAMA_DSV4_CED_DEC_ONLY) {
        ggml_tensor * inp = build_inp_embd(model.tok_embd);
        inpL = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
        cb(inpL, "hc_init", -1);

        // embedding input has no token ids: it skips Engram, like the reference's masked (image) positions
        // the gathers go right after the token embeddings, so they share the first graph split (the table stays on the host)
        for (int il = 0; ubatch.token && hparams.ple_n_heads > 0 && il < n_layer; ++il) {
            if (!hparams.is_ple(il)) {
                continue;
            }
            ggml_tensor * ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) hparams.ple_n_heads*n_tokens);
            ggml_set_name(ids, ("engram_ids-" + std::to_string(il)).c_str());
            ggml_set_input(ids);
            inp41->engram_ids.push_back(ids);

            ggml_tensor * rows = ggml_get_rows(ctx0, model.per_layer_tok_embd, ids);
            rows = ggml_reshape_2d(ctx0, rows, hparams.ple_head_dim*hparams.ple_n_heads, n_tokens);
            cb(rows, "engram_rows", il);
            ggml_build_forward_expand(gf, rows);
            engram_rows.push_back(rows);
        }
        if (hparams.ple_n_heads > 0 && ubatch.token && ubatch.embd) {
            engram_keep = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
            ggml_set_input(engram_keep);
            ggml_set_name(engram_keep, "engram_keep");
            inp41->engram_keep = engram_keep;
        }
    } else {
        ggml_tensor * h_in = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, hc, n_tokens);
        ggml_set_input(h_in);
        ggml_set_name(h_in, "ced_inp_h");
        inp41->ced_inp_h = h_in;

        ggml_tensor * p_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc, n_tokens);
        ggml_set_input(p_in);
        ggml_set_name(p_in, "ced_inp_pre");
        inp41->ced_inp_pre = p_in;

        inpL     = h_in;
        pre_prev = p_in;
    }

    ggml_tensor * cand_pin = nullptr;
    if (ced_mode != LLAMA_DSV4_CED_ENC_ONLY && hparams.dsv41_cand_src >= 0) {
        inp41->cand_csa = hparams.dsv4_compress_ratios[hparams.dsv41_cand_src] == 2;
        const auto & ci = inp41->cand_csa ? inp_dsv4->get_csa() : inp_dsv4->get_hca();
        if (ci.kq_mask) {
            const int64_t bs = hparams.dsv41_cand_block_size;
            cand_pin = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, (ci.kq_mask->ne[0] + bs - 1)/bs, ci.kq_mask->ne[1], 1, ci.kq_mask->ne[3]);
            ggml_set_input(cand_pin);
            ggml_set_name(cand_pin, "cand_pin");
            inp41->cand_pin = cand_pin;
        }
    }

    ggml_tensor * router_bias_ids = nullptr;
    const bool need_rbi = ced_mode == LLAMA_DSV4_CED_DEC_ONLY
        ? (ced.dec_has_vl && !ced.dec_all_vl)
        : ubatch.is_mixed();
    if (need_rbi) {
        router_bias_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_input(router_bias_ids);
        ggml_set_name(router_bias_ids, "router_bias_ids");
        inp41->router_bias_ids = router_bias_ids;
    }

    const int32_t L_dec = hparams.dsv41_ced_dec_layer;
    const int32_t enc_end = ced_mode == LLAMA_DSV4_CED_ENC_ONLY ? L_dec : (ced_mode == LLAMA_DSV4_CED_DEC_ONLY ? 0 : n_layer);
    const bool has_bounds = !ced.k_start.empty() && !ced.o_start.empty();
    const int32_t split_dec = (L_dec >= 0 && ced_mode == LLAMA_DSV4_CED_FULL && has_bounds) ? L_dec : enc_end;

    // 1) Encoder loop: [0 .. split_dec) (for FULL unpruned: [0 .. n_layer); ENC_ONLY: [0 .. L_dec); DEC_ONLY: empty)
    for (int il = 0; il < split_dec; ++il) {
        const auto & layer = model.layers[il];

        const int e = model.engram_index(il);
        if (e >= 0 && !engram_rows.empty()) {
            inpL = build_engram(model, inpL, engram_rows[e], engram_keep, il);
            cb(inpL, "engram_out", il);
        }

        // DSpark target features: mean over the hc copies of the layer input
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = build_hc_mean(inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        ggml_tensor * residual = inpL;
        const hc_mixes attn_mix = build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);

        // the first layer's attention collapses with the identity pre-mix: copy 0
        ggml_tensor * cur = pre_prev ? build_hc_pre(inpL, pre_prev, il) : ggml_cont(ctx0, ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[2], 0));
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * k_idxs_dec = inp_dsv4->get_raw()->get_k_idxs_dec();
        ggml_tensor * raw_mask   = (k_idxs_dec && L_dec >= 0 && il >= L_dec)
            ? inp_dsv4->get_raw()->self_attn_mask_dec.raw
            : inp_dsv4->get_raw()->self_attn_mask.raw;
        ggml_tensor * k_idxs_layer = (L_dec >= 0 && il >= L_dec) ? k_idxs_dec : nullptr;

        cur = build_attention41(model, inp_dsv4, cur, cur, inp_pos, inp_pos, k_idxs_layer,
                raw_mask, nullptr, cand_pin, 0, 0, false, false, il);

        inpL = build_hc_post(cur, residual, attn_mix.post, attn_mix.comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        const hc_mixes ffn_mix = build_hc_mixes(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il);
        pre_prev = ffn_mix.pre;

        cur = build_hc_pre(inpL, attn_mix.pre, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, ffn_mix.post);
        ggml_build_forward_expand(gf, ffn_mix.comb);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // the reference computes the router in F32: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py#L813
        ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur);
        ggml_prec_set_acc(logits, GGML_PREC_F32);
        cb(logits, "ffn_moe_logits", il);

        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;
        if (!router_bias_ids && ubatch.embd && layer.ffn_exp_probs_b_vl) {
            exp_probs_b = layer.ffn_exp_probs_b_vl;
        }

        if (ced_mode == LLAMA_DSV4_CED_ENC_ONLY && il + 1 == L_dec) {
            moe_stream_next_n_tokens = 0;
        } else if (ced_mode == LLAMA_DSV4_CED_FULL && has_bounds && il + 1 == L_dec) {
            moe_stream_next_n_tokens = std::max<int32_t>(0, (int32_t) n_tokens - ced.o_start[L_dec]);
        } else {
            moe_stream_next_n_tokens = -1;
        }

        // the reference normalizes only when it routes to more than one expert
        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm && hparams.n_expert_used() > 1,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il, logits, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1e-5f, router_bias_ids);
        cb(moe_out, "ffn_moe_out", il);

        moe_stream_next_n_tokens = -1;

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, ffn_mix.post, ffn_mix.comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    // 2) ENC_ONLY: export encoder tail + compute global KV write & mirrors at L_dec, then return early
    if (ced_mode == LLAMA_DSV4_CED_ENC_ONLY) {
        GGML_ASSERT(L_dec > 0 && pre_prev != nullptr);
        res->ced_skip_layer_inp = true;

        ggml_tensor * out_h   = inpL;
        ggml_tensor * out_pre = pre_prev;
        if (ced.tail_keep_max > 0 && ubatch.n_seq_tokens > ced.tail_keep_max) {
            GGML_ASSERT(ubatch.equal_seqs() && ubatch.n_seqs > 0);
            const uint32_t keep = ced.tail_keep_max;
            const uint32_t ns   = ubatch.n_seqs;
            const uint32_t nst  = ubatch.n_seq_tokens;
            const uint32_t drop = nst - keep;
            if (ns == 1) {
                out_h   = ggml_cont(ctx0, ggml_view_3d(ctx0, inpL,     n_embd, hc, keep, inpL->nb[1],     inpL->nb[2],     (size_t) drop * inpL->nb[2]));
                out_pre = ggml_cont(ctx0, ggml_view_2d(ctx0, pre_prev, hc,     keep,     pre_prev->nb[1], (size_t) drop * pre_prev->nb[1]));
            } else {
                ggml_tensor * ch = nullptr;
                ggml_tensor * cp = nullptr;
                for (uint32_t s = 0; s < ns; ++s) {
                    const uint32_t off = s * nst + drop;
                    ggml_tensor * vh = ggml_view_3d(ctx0, inpL,     n_embd, hc, keep, inpL->nb[1],     inpL->nb[2],     (size_t) off * inpL->nb[2]);
                    ggml_tensor * vp = ggml_view_2d(ctx0, pre_prev, hc,     keep,     pre_prev->nb[1], (size_t) off * pre_prev->nb[1]);
                    ch = ch ? ggml_concat(ctx0, ch, vh, 2) : vh;
                    cp = cp ? ggml_concat(ctx0, cp, vp, 1) : vp;
                }
                out_h   = ggml_cont(ctx0, ch);
                out_pre = ggml_cont(ctx0, cp);
            }
        } else {
            out_pre = ggml_cont(ctx0, pre_prev);
        }
        res->t_ced_h   = out_h;
        res->t_ced_pre = out_pre;
        ggml_build_forward_expand(gf, out_h);
        ggml_build_forward_expand(gf, out_pre);

        ggml_tensor * cur = build_hc_pre(inpL, pre_prev, L_dec);
        cb(cur, "hc_attn_pre", L_dec);
        cur = build_norm(cur, model.layers[L_dec].attn_norm, nullptr, LLM_NORM_RMS, L_dec);
        cb(cur, "attn_norm", L_dec);
        build_comp_kv_write(model, inp_dsv4, cur, L_dec);
        build_comp_kv_mirrors(model, inp_dsv4, L_dec);

        res->add_input(std::move(inp41));
        return;
    }

    // 3) Global KV write for pruned FULL before entering sliced decoder pyramid
    if (ced_mode == LLAMA_DSV4_CED_FULL && has_bounds && L_dec >= 0) {
        ggml_tensor * cur = build_hc_pre(inpL, pre_prev, L_dec);
        cb(cur, "hc_attn_pre", L_dec);
        cur = build_norm(cur, model.layers[L_dec].attn_norm, nullptr, LLM_NORM_RMS, L_dec);
        cb(cur, "attn_norm", L_dec);
        build_comp_kv_write(model, inp_dsv4, cur, L_dec);
        build_comp_kv_mirrors(model, inp_dsv4, L_dec);
    }

    // 4) Decoder pyramid loop (used for DEC_ONLY and pruned FULL; unpruned FULL already ran split_dec == n_layer above)
    int32_t in_start = 0;
    const int dec_beg = ced_mode == LLAMA_DSV4_CED_DEC_ONLY ? L_dec : split_dec;
    for (int il = dec_beg; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const int32_t ks = has_bounds ? ced.k_start[il] : 0;
        const int32_t os = has_bounds ? ced.o_start[il] : 0;
        if (ks >= (int32_t) n_tokens) {
            break;
        }

        GGML_ASSERT(ks >= in_start && os >= ks);
        if (ks > in_start) {
            const int64_t nk = n_tokens - ks;
            const int32_t diff = ks - in_start;
            inpL     = ggml_cont(ctx0, ggml_view_3d(ctx0, inpL,     n_embd, hc, nk, inpL->nb[1],     inpL->nb[2],     (size_t) diff * inpL->nb[2]));
            pre_prev = ggml_cont(ctx0, ggml_view_2d(ctx0, pre_prev, hc,     nk,     pre_prev->nb[1], (size_t) diff * pre_prev->nb[1]));
            in_start = ks;
        }

        const int64_t nk = n_tokens - ks;

        // DSpark target features: mean over the hc copies of the layer input
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            if (ced.li_start < 0) {
                GGML_ASSERT(ks == 0);
                res->t_layer_inp[il] = build_hc_mean(inpL);
                cb(res->t_layer_inp[il], "layer_inp", il);
                ggml_build_forward_expand(gf, res->t_layer_inp[il]);
            } else if (ced.li_start < (int32_t) n_tokens) {
                GGML_ASSERT(ced.li_start >= ks);
                const int32_t li_diff = ced.li_start - ks;
                ggml_tensor * li_inp = li_diff > 0
                    ? ggml_cont(ctx0, ggml_view_3d(ctx0, inpL, n_embd, hc, n_tokens - ced.li_start, inpL->nb[1], inpL->nb[2], (size_t) li_diff * inpL->nb[2]))
                    : inpL;
                res->t_layer_inp[il] = build_hc_mean(li_inp);
                cb(res->t_layer_inp[il], "layer_inp", il);
                ggml_build_forward_expand(gf, res->t_layer_inp[il]);
            }
        }

        ggml_tensor * cur_k = build_hc_pre(inpL, pre_prev, il);
        cb(cur_k, "hc_attn_pre", il);
        cur_k = build_norm(cur_k, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur_k, "attn_norm", il);

        ggml_tensor * pos_k = ks > 0 ? ggml_view_1d(ctx0, inp_pos, nk, (size_t) ks * ggml_element_size(inp_pos)) : inp_pos;
        ggml_tensor * k_idxs_dec = inp_dsv4->get_raw()->get_k_idxs_dec();

        const bool kv_only = os >= (int32_t) n_tokens;
        if (kv_only) {
            build_attention41(model, inp_dsv4, cur_k, nullptr, pos_k, nullptr,
                    k_idxs_dec, nullptr, nullptr, nullptr, ks, os, true, true, il);
            break;
        }

        const int64_t no = n_tokens - os;
        ggml_tensor * cur_o = cur_k;
        if (os > ks) {
            const int32_t diff = os - ks;
            inpL  = ggml_cont(ctx0, ggml_view_3d(ctx0, inpL,  n_embd, hc, no, inpL->nb[1],  inpL->nb[2],  (size_t) diff * inpL->nb[2]));
            cur_o = ggml_cont(ctx0, ggml_view_2d(ctx0, cur_k, n_embd,     no, cur_k->nb[1],               (size_t) diff * cur_k->nb[1]));
            in_start = os;
        }

        ggml_tensor * pos_o = os > 0 ? ggml_view_1d(ctx0, inp_pos, no, (size_t) os * ggml_element_size(inp_pos)) : inp_pos;
        ggml_tensor * raw_mask_all = inp_dsv4->get_raw()->self_attn_mask_dec.raw
            ? inp_dsv4->get_raw()->self_attn_mask_dec.raw
            : inp_dsv4->get_raw()->self_attn_mask.raw;
        ggml_tensor * raw_mask_o = (os > 0 && raw_mask_all)
            ? ggml_cont(ctx0, ggml_view_4d(ctx0, raw_mask_all, raw_mask_all->ne[0], no, 1, raw_mask_all->ne[3],
                        raw_mask_all->nb[1], raw_mask_all->nb[2], raw_mask_all->nb[3], (size_t) os * raw_mask_all->nb[1]))
            : raw_mask_all;

        const auto & hca_ci = inp_dsv4->get_hca();
        ggml_tensor * hca_mask_o = (os > 0 && hca_ci.kq_mask)
            ? ggml_cont(ctx0, ggml_view_4d(ctx0, hca_ci.kq_mask, hca_ci.kq_mask->ne[0], no, 1, hca_ci.kq_mask->ne[3],
                        hca_ci.kq_mask->nb[1], hca_ci.kq_mask->nb[2], hca_ci.kq_mask->nb[3], (size_t) os * hca_ci.kq_mask->nb[1]))
            : hca_ci.kq_mask;

        ggml_tensor * cand_pin_o = (os > 0 && cand_pin)
            ? ggml_cont(ctx0, ggml_view_4d(ctx0, cand_pin, cand_pin->ne[0], no, 1, cand_pin->ne[3],
                        cand_pin->nb[1], cand_pin->nb[2], cand_pin->nb[3], (size_t) os * cand_pin->nb[1]))
            : cand_pin;

        ggml_tensor * rbi_o = (os > 0 && router_bias_ids)
            ? ggml_view_1d(ctx0, router_bias_ids, no, (size_t) os * ggml_element_size(router_bias_ids))
            : router_bias_ids;

        ggml_tensor * residual = inpL;
        const hc_mixes attn_mix = build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);

        ggml_tensor * cur = build_attention41(model, inp_dsv4, cur_k, cur_o, pos_k, pos_o, k_idxs_dec,
                raw_mask_o, hca_mask_o, cand_pin_o, ks, os, true, false, il);

        inpL = build_hc_post(cur, residual, attn_mix.post, attn_mix.comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        const hc_mixes ffn_mix = build_hc_mixes(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il);
        pre_prev = ffn_mix.pre;

        cur = build_hc_pre(inpL, attn_mix.pre, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, ffn_mix.post);
        ggml_build_forward_expand(gf, ffn_mix.comb);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur);
        ggml_prec_set_acc(logits, GGML_PREC_F32);
        cb(logits, "ffn_moe_logits", il);

        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;
        const bool all_vl = ced_mode == LLAMA_DSV4_CED_DEC_ONLY ? ced.dec_all_vl : (ubatch.embd != nullptr);
        if (!rbi_o && all_vl && layer.ffn_exp_probs_b_vl) {
            exp_probs_b = layer.ffn_exp_probs_b_vl;
        }

        if (il + 1 >= n_layer) {
            moe_stream_next_n_tokens = 0;
        } else if (has_bounds) {
            moe_stream_next_n_tokens = std::max<int32_t>(0, (int32_t) n_tokens - ced.o_start[il + 1]);
        } else {
            moe_stream_next_n_tokens = -1;
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm && hparams.n_expert_used() > 1,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il, logits, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1e-5f, rbi_o);
        cb(moe_out, "ffn_moe_out", il);

        moe_stream_next_n_tokens = -1;

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, ffn_mix.post, ffn_mix.comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    // 5) Output head
    res->ced_skip_layer_inp = ced.li_start >= (int32_t) n_tokens;
    if (!res->ced_skip_layer_inp && ced_mode == LLAMA_DSV4_CED_DEC_ONLY) {
        for (size_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
            if (cparams.embeddings_layer_inp[il] && ((int32_t) il < L_dec || (int32_t) il >= n_layer)) {
                res->ced_skip_layer_inp = true;
                break;
            }
        }
    }
    res->ced_li_start       = ced.li_start;
    res->ced_li_dst_offset  = ced.li_dst_offset;

    if (n_outputs == 0) {
        res->add_input(std::move(inp41));
        return;
    }

    const int64_t n_cur = n_tokens - in_start;
    if (in_start == 0) {
        inp_out_ids = build_inp_out_ids();
        if (inp_out_ids) {
            inpL     = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_cur), inp_out_ids), n_embd, hc, inp_out_ids->ne[0]);
            pre_prev = ggml_get_rows(ctx0, pre_prev, inp_out_ids);
        }
    } else if (n_outputs < n_cur) {
        ggml_tensor * out_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
        ggml_set_input(out_ids);
        ggml_set_name(out_ids, "ced_out_ids");
        inp41->ced_in_start = in_start;
        inp41->ced_out_ids  = out_ids;
        inpL     = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_cur), out_ids), n_embd, hc, n_outputs);
        pre_prev = ggml_get_rows(ctx0, pre_prev, out_ids);
    } else {
        GGML_ASSERT(n_outputs == n_cur);
    }

    res->add_input(std::move(inp41));

    // no hc_head: collapse with the last FFN's pre-mix
    ggml_tensor * cur = build_hc_pre(inpL, pre_prev, -1);
    cb(cur, "hc_out", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
