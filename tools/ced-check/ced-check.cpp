// llama-ced-check: exactness and prefill speed of the DeepSeek-V4.1 exact CED prefill (LLAMA_DSV41_CED)
//
// For every prompt length (and a multi-turn script) the same tokens are prefilled with the full decoder (old) and
// with CED (new), then compared:
//   - the sequence state (llama_state_seq_get_data): the window K of every layer (the state holds the last n_swa
//     positions), the compressed rows of every kv source (layer 20: the decoder's global KV, all positions) and the
//     compressor states
//   - the logits of the last prompt token, and of S teacher-forced decode steps (max abs diff, top-1, KLD)
//   - the DSpark target features (layer inputs) of the last n_swa rows of the last call
// The noise floor is the old path with the server's split of the prompt (calls of n - 4 - ub, ub and 4 tokens),
// which changes the ubatch shapes only; the same split with CED tests CED across calls.
//
// LLAMA_DSV41_CED is read when the KV cache is built, so the harness sets it per run and recreates the context
// (llama_free + llama_init_from_model) whenever the mode of the live context differs; the model stays loaded.
// The sequence states of the two modes differ in format and are then not compared (a note is printed instead).
//
// usage: llama-ced-check -m MODEL [model/context flags] -f TEXT -c CTX -ub UB --no-warmup
// env:   CED_LENS=1024,4096,16384,65536  prompt lengths
//        CED_STEPS=128                   teacher-forced decode steps after each prefill
//        CED_TURNS=8192:64:600,8192:64:4000  multi-turn scripts: prompt, decode steps, appended turn (one call each)
//        CED_SPLIT=1                     also run the server split (noise floor and CED across calls)
//        CED_TARGET=0                    1 (with CED_SPLIT): the split in calls of n_ubatch (the rest, then the last 4
//                                        tokens): CED defers the decoder across calls by itself
//        CED_REPS=1                      timing repetitions (old/new alternate)
//        CED_FEAT=37,38,39               layer inputs to extract (DSpark targets), empty = none
//        CED_DEC0=20                     first decoder layer (for the report)
//        CED_OUT=ced.json                machine-readable results
//        CED_AA=0                        1: run the old path twice (old vs old: the run-to-run noise floor)
//        CED_DUMP=                       prefix: write each sequence state to PREFIX<run>.state
//        CED_RESTORE=0                   1 (with CED_SPLIT): a checkpoint before the last call of the split, as the
//                                        server takes it, restored and the last call run again: must be identical

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static std::vector<int> env_ints(const char * name, const char * def) {
    const char * v = getenv(name);
    std::string s = v ? v : def;
    std::vector<int> res;
    size_t p = 0;
    while (p < s.size()) {
        size_t q = s.find_first_of(",:", p);
        if (q == std::string::npos) {
            q = s.size();
        }
        if (q > p) {
            res.push_back(atoi(s.substr(p, q - p).c_str()));
        }
        p = q + 1;
    }
    return res;
}

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

//
// logits metrics
//

struct logit_cmp {
    double max_abs = 0.0;
    double kld     = 0.0; // KL(ref || test)
    bool   top1    = true;
};

static logit_cmp cmp_logits(const float * a, const float * b, int n) {
    logit_cmp r;
    double ma = -INFINITY, mb = -INFINITY;
    int ia = 0, ib = 0;
    for (int i = 0; i < n; ++i) {
        r.max_abs = std::max(r.max_abs, (double) std::fabs(a[i] - b[i]));
        if (a[i] > ma) { ma = a[i]; ia = i; }
        if (b[i] > mb) { mb = b[i]; ib = i; }
    }
    double sa = 0.0, sb = 0.0;
    for (int i = 0; i < n; ++i) {
        sa += std::exp(a[i] - ma);
        sb += std::exp(b[i] - mb);
    }
    const double la = ma + std::log(sa), lb = mb + std::log(sb);
    for (int i = 0; i < n; ++i) {
        const double lpa = a[i] - la, lpb = b[i] - lb;
        r.kld += std::exp(lpa)*(lpa - lpb);
    }
    r.top1 = ia == ib;
    return r;
}

//
// sequence state parsing (llama_kv_cache_dsv4::state_write, full mode, one sequence)
//

struct reader {
    const uint8_t * p;
    size_t n;
    size_t o  = 0;
    bool   ok = true;

    template <typename T> T get() {
        T v{};
        if (o + sizeof(T) > n) { ok = false; return v; }
        memcpy(&v, p + o, sizeof(T));
        o += sizeof(T);
        return v;
    }
    template <typename T> T peek(size_t at) const {
        T v{};
        if (o + at + sizeof(T) <= n) {
            memcpy(&v, p + o + at, sizeof(T));
        }
        return v;
    }
    const uint8_t * take(size_t k) {
        if (o + k > n) { ok = false; return nullptr; }
        const uint8_t * r = p + o;
        o += k;
        return r;
    }
};

struct rows_block {
    int32_t         type = 0;
    uint64_t        row_size = 0;
    uint64_t        n_rows = 0;
    const uint8_t * data = nullptr;
};

struct kv_state {
    std::vector<int32_t>    pos;    // per cell
    std::vector<rows_block> layers; // K per layer (rows = cells)
};

struct comp_layer {
    uint32_t                il;
    std::vector<rows_block> blocks; // k-cache: 1 (rows), comp state: kv, score
};

struct dsv4_state {
    kv_state                base, swa;
    std::vector<std::vector<comp_layer>> kcache; // csa, hca, lid
    std::vector<std::vector<comp_layer>> cstate; // csa, hca, lid
};

static bool parse_kv(reader & r, bool has_ext, kv_state & out) {
    const uint32_t n_stream = r.get<uint32_t>();
    if (!r.ok || n_stream != 1) {
        return false;
    }
    const uint32_t n_cells = r.get<uint32_t>();
    if (n_cells == 0) {
        return r.ok;
    }
    for (uint32_t i = 0; i < n_cells && r.ok; ++i) {
        out.pos.push_back(r.get<int32_t>());
        const uint32_t n_seq = r.get<uint32_t>();
        if (has_ext) {
            r.take(12); // llama_kv_cell_ext: x, y, tok
        }
        if (n_seq > 64) {
            return false;
        }
        r.take(4*n_seq);
    }
    const uint32_t v_trans = r.get<uint32_t>();
    const uint32_t n_layer = r.get<uint32_t>();
    r.get<uint32_t>(); // n_rot_k
    r.get<uint32_t>(); // n_rot_v
    if (!r.ok || v_trans > 1 || n_layer > 4096) {
        return false;
    }
    for (uint32_t il = 0; il < n_layer && r.ok; ++il) {
        rows_block b;
        b.type     = r.get<int32_t>();
        b.row_size = r.get<uint64_t>();
        b.n_rows   = n_cells;
        if (b.type < 0 || b.type >= GGML_TYPE_COUNT || b.row_size == 0 || b.row_size > (1u << 20)) {
            return false;
        }
        b.data = r.take(n_cells*b.row_size);
        out.layers.push_back(b);
    }
    // V blocks (non-MLA caches): one per layer with the same row size, recognized by their header
    if (!v_trans && n_layer > 0 && r.peek<uint64_t>(4) == out.layers[0].row_size) {
        for (uint32_t il = 0; il < n_layer && r.ok; ++il) {
            r.get<int32_t>();
            const uint64_t rs = r.get<uint64_t>();
            r.take(n_cells*rs);
        }
    }
    return r.ok;
}

static bool parse_streams(reader & r, rows_block & b) {
    b.type = r.get<int32_t>();
    r.get<uint64_t>(); // ne0
    b.n_rows   = r.get<uint64_t>();
    b.row_size = r.get<uint64_t>();
    if (!r.ok || b.type < 0 || b.type >= GGML_TYPE_COUNT) {
        return false;
    }
    b.data = r.take(b.n_rows*b.row_size); // one sequence: one stream
    return r.ok;
}

static bool parse_dsv4(const std::vector<uint8_t> & buf, bool has_ext, dsv4_state & st) {
    reader r { buf.data(), buf.size() };
    r.get<uint32_t>(); // llama_context io magic
    r.get<int32_t>();  // seq_id
    const uint32_t magic = r.get<uint32_t>();
    const uint32_t version = r.get<uint32_t>(); // 2 = the CED layout (decoder cache, encoder tail): not parsed
    const uint32_t mode = r.get<uint32_t>();
    if (!r.ok || magic != 0x34565344 || version != 1 || mode != 0) {
        return false;
    }
    if (!parse_kv(r, has_ext, st.base) || !parse_kv(r, has_ext, st.swa)) {
        return false;
    }
    st.kcache.resize(3);
    for (int c = 0; c < 3; ++c) {
        r.get<uint32_t>(); // version
        r.get<uint32_t>(); // n_rows
        const uint32_t ns      = r.get<uint32_t>();
        const uint32_t n_layer = r.get<uint32_t>();
        if (!r.ok || ns != 1 || n_layer > 4096) {
            return false;
        }
        for (uint32_t l = 0; l < n_layer; ++l) {
            comp_layer cl { r.get<uint32_t>(), { rows_block{} } };
            if (!parse_streams(r, cl.blocks[0])) {
                return false;
            }
            st.kcache[c].push_back(cl);
        }
    }
    st.cstate.resize(3);
    for (int c = 0; c < 3; ++c) {
        for (int k = 0; k < 4; ++k) {
            r.get<uint32_t>(); // version, ratio, state_size, n_embd_state
        }
        const uint32_t ns      = r.get<uint32_t>();
        const uint32_t n_layer = r.get<uint32_t>();
        if (!r.ok || ns != 1 || n_layer > 4096) {
            return false;
        }
        for (uint32_t l = 0; l < n_layer; ++l) {
            comp_layer cl { r.get<uint32_t>(), { rows_block{}, rows_block{} } };
            if (!parse_streams(r, cl.blocks[0]) || !parse_streams(r, cl.blocks[1])) {
                return false;
            }
            st.cstate[c].push_back(cl);
        }
    }
    return r.ok && r.o == r.n;
}

static bool parse_dsv4_any(const std::vector<uint8_t> & buf, dsv4_state & st) {
    for (bool ext : { true, false }) {
        st = {};
        if (parse_dsv4(buf, ext, st)) {
            return true;
        }
    }
    return false;
}

static void row_to_float(const rows_block & b, uint64_t i, std::vector<float> & out) {
    const ggml_type t  = (ggml_type) b.type;
    const int64_t   ne = (int64_t) (b.row_size/ggml_type_size(t))*ggml_blck_size(t);
    out.resize(ne);
    const uint8_t * src = b.data + i*b.row_size;
    if (t == GGML_TYPE_F32) {
        memcpy(out.data(), src, ne*sizeof(float));
    } else {
        ggml_get_type_traits(t)->to_float(src, out.data(), ne);
    }
}

struct rows_cmp {
    double   max_abs = 0.0;
    double   max_ref = 0.0;
    uint64_t n_rows  = 0;
    uint64_t n_diff  = 0; // rows that are not bit-identical
    bool     shape_ok = true;
};

// rows ia[k] of a against rows ib[k] of b
static rows_cmp cmp_rows(const rows_block & a, const rows_block & b, const std::vector<std::pair<uint64_t, uint64_t>> & pairs) {
    rows_cmp r;
    if (a.type != b.type || a.row_size != b.row_size) {
        r.shape_ok = false;
        return r;
    }
    std::vector<float> fa, fb;
    for (const auto & [ia, ib] : pairs) {
        r.n_rows++;
        if (memcmp(a.data + ia*a.row_size, b.data + ib*b.row_size, a.row_size) == 0) {
            continue;
        }
        r.n_diff++;
        row_to_float(a, ia, fa);
        row_to_float(b, ib, fb);
        for (size_t j = 0; j < fa.size(); ++j) {
            r.max_abs = std::max(r.max_abs, (double) std::fabs(fa[j] - fb[j]));
            r.max_ref = std::max(r.max_ref, (double) std::fabs(fa[j]));
        }
    }
    return r;
}

static std::vector<std::pair<uint64_t, uint64_t>> same_rows(uint64_t n) {
    std::vector<std::pair<uint64_t, uint64_t>> res(n);
    for (uint64_t i = 0; i < n; ++i) {
        res[i] = { i, i };
    }
    return res;
}

//
// runs
//

struct run_result {
    std::string                     name;
    double                          t_prefill = 0.0; // seconds of the measured prompt call(s)
    int                             n_prefill = 0;   // tokens in them
    std::vector<float>              logits_last;
    std::vector<float>              logits_restore; // CED_RESTORE: the last call again, from the checkpoint before it
    std::vector<std::vector<float>> logits_steps;
    std::vector<uint8_t>            state;
    std::vector<std::vector<float>> feats; // per CED_FEAT layer: the last rows of the last call
    int                             n_feat_rows = 0;
};

struct harness {
    llama_context * ctx;
    int             n_vocab;
    int             n_embd;
    int             n_batch;
    int             n_ubatch;
    std::vector<int> feat_layers;
    bool             restore = false;

    std::vector<uint8_t> * ckpt = nullptr;
    llama_pos              ckpt_pos = -1;
    size_t                 ckpt_i   = 0;

    bool      use_target = false; // CED_TARGET: the split in calls of n_ubatch

    // the context is recreated when the CED mode (LLAMA_DSV41_CED, read when the KV cache is built) changes
    llama_model *          model = nullptr;
    llama_context_params   cparams {};
    bool                   ced_cur = true;

    void set_mode(bool ced) {
        setenv("LLAMA_DSV41_CED", ced ? "1" : "0", 1);
        if (ced == ced_cur) {
            return;
        }
        llama_free(ctx);
        ctx = llama_init_from_model(model, cparams);
        GGML_ASSERT(ctx && "failed to recreate the context");
        for (int l : feat_layers) {
            llama_set_embeddings_layer_inp(ctx, (uint32_t) l, true);
        }
        ced_cur = ced;
    }

    // decode toks[i0, i1) at pos0 + (i - i0), in calls of the given sizes (0 = one call); only the last token outputs
    bool decode(const std::vector<llama_token> & toks, size_t i0, size_t i1, llama_pos pos0, const std::vector<int> & calls, int & n_last) {
        common_batch batch(ctx);
        size_t i = i0;
        size_t c = 0;
        while (i < i1) {
            const size_t n = c < calls.size() ? std::min<size_t>(calls[c++], i1 - i) : i1 - i;
            if (n == 0) {
                continue;
            }
            GGML_ASSERT((int) n <= n_batch);
            if (ckpt && i + n == i1 && i > i0) {
                // the server's context checkpoint: taken before the last batch of the prompt
                llama_synchronize(ctx);
                ckpt->resize(llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                ckpt->resize(llama_state_seq_get_data_ext(ctx, ckpt->data(), ckpt->size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                ckpt_pos = pos0 + (llama_pos) (i - i0);
                ckpt_i   = i;
            }
            batch.clear();
            for (size_t k = 0; k < n; ++k) {
                batch.add(toks[i + k], pos0 + (llama_pos) (i + k - i0), 0, i + k + 1 == i1);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                LOG_ERR("decode failed at %zu (+%zu)\n", i, n);
                return false;
            }
            n_last = (int) n;
            i += n;
        }
        llama_synchronize(ctx);
        return true;
    }

    // a prefill script: segments of {n tokens, kind} with kind 0 = one prompt call (or the split), 1 = single-token
    // decode steps; the measurements follow the last prompt call
    bool run(run_result & res, bool ced, const std::vector<llama_token> & toks, const std::vector<std::pair<int, int>> & script,
             bool split, int n_steps) {
        set_mode(ced);
        llama_memory_clear(llama_get_memory(ctx), true);

        size_t i = 0;
        int n_last = 0;
        for (size_t s = 0; s < script.size(); ++s) {
            const auto [n, kind] = script[s];
            if (kind == 1) {
                for (int k = 0; k < n; ++k, ++i) {
                    if (!decode(toks, i, i + 1, (llama_pos) i, {}, n_last)) {
                        return false;
                    }
                }
                continue;
            }
            std::vector<int> calls;
            if (split && use_target && n > n_ubatch + 4) {
                // as a server that reads no state before the output: calls of n_ubatch, the rest, the last 4 tokens
                for (int r = n - 4; r > 0; r -= n_ubatch) {
                    calls.push_back(std::min(r, n_ubatch));
                }
                calls.push_back(4);
            } else if (split && n > n_ubatch + 4) {
                calls = { n - 4 - n_ubatch, n_ubatch, 4 };
            }
            const double t0 = now_s();
            std::vector<uint8_t> cp;
            ckpt     = restore && !calls.empty() && s + 1 == script.size() ? &cp : nullptr;
            ckpt_pos = -1;
            if (!decode(toks, i, i + n, (llama_pos) i, calls, n_last)) {
                return false;
            }
            ckpt = nullptr;
            if (ckpt_pos >= 0) {
                // as the server: restore the checkpoint, drop the later positions, run the last batch again
                const std::vector<float> straight(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + n_vocab);
                if (llama_state_seq_set_data_ext(ctx, cp.data(), cp.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != cp.size()) {
                    LOG_ERR("checkpoint restore failed\n");
                    return false;
                }
                llama_memory_seq_rm(llama_get_memory(ctx), 0, ckpt_pos, -1);
                int n_tmp = 0;
                if (!decode(toks, ckpt_i, i + n, ckpt_pos, {}, n_tmp)) {
                    return false;
                }
                res.logits_restore.assign(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + n_vocab);
                const logit_cmp c = cmp_logits(straight.data(), res.logits_restore.data(), n_vocab);
                printf("  %s: last batch straight vs from the checkpoint at %d: max |d| %.3e, KLD %.3e, top-1 %s\n", res.name.c_str(),
                        ckpt_pos, c.max_abs, c.kld, c.top1 ? "same" : "DIFFERENT");
            }
            if (s + 1 == script.size() || script[s + 1].second == 1) {
                res.t_prefill = now_s() - t0;
                res.n_prefill = n;
            }
            i += n;
        }

        res.logits_last.assign(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + n_vocab);

        res.n_feat_rows = std::min(n_last, 128);
        res.feats.clear();
        for (int l : feat_layers) {
            const float * f = llama_get_embeddings_layer_inp(ctx, (uint32_t) l);
            res.feats.emplace_back(f + (size_t) (n_last - res.n_feat_rows)*n_embd, f + (size_t) n_last*n_embd);
        }

        res.state.resize(llama_state_seq_get_size(ctx, 0));
        res.state.resize(llama_state_seq_get_data(ctx, res.state.data(), res.state.size(), 0));

        res.logits_steps.clear();
        for (int k = 0; k < n_steps; ++k, ++i) {
            if (!decode(toks, i, i + 1, (llama_pos) i, {}, n_last)) {
                return false;
            }
            const float * l = llama_get_logits_ith(ctx, -1);
            res.logits_steps.emplace_back(l, l + n_vocab);
        }
        return true;
    }
};

struct report {
    FILE * json = nullptr;
    bool   first = true;

    void kv(const std::string & k, double v) {
        if (json) {
            fprintf(json, "%s\"%s\": %.9g", first ? "" : ", ", k.c_str(), v);
            first = false;
        }
    }
};

// test against ref: logits, steps, features, state
static void compare(const run_result & ref, const run_result & test, int dec0, report & rep) {
    printf("\n  %s vs %s\n", test.name.c_str(), ref.name.c_str());

    const int n_vocab = (int) ref.logits_last.size();
    const logit_cmp l0 = cmp_logits(ref.logits_last.data(), test.logits_last.data(), n_vocab);
    printf("    last token logits : max |d| %.3e, KLD %.3e, top-1 %s\n", l0.max_abs, l0.kld, l0.top1 ? "same" : "DIFFERENT");
    rep.kv(test.name + ".last.max_abs", l0.max_abs);
    rep.kv(test.name + ".last.kld", l0.kld);
    rep.kv(test.name + ".last.top1", l0.top1);

    if (!ref.logits_steps.empty()) {
        double kmax = 0.0, ksum = 0.0, amax = 0.0;
        int n_top1 = 0;
        const size_t n = std::min(ref.logits_steps.size(), test.logits_steps.size());
        for (size_t s = 0; s < n; ++s) {
            const logit_cmp c = cmp_logits(ref.logits_steps[s].data(), test.logits_steps[s].data(), n_vocab);
            kmax = std::max(kmax, c.kld);
            ksum += c.kld;
            amax = std::max(amax, c.max_abs);
            n_top1 += c.top1;
        }
        printf("    %3zu decode steps   : max |d| %.3e, KLD mean %.3e max %.3e, top-1 %d/%zu\n", n, amax, ksum/n, kmax, n_top1, n);
        rep.kv(test.name + ".steps.max_abs", amax);
        rep.kv(test.name + ".steps.kld_mean", ksum/n);
        rep.kv(test.name + ".steps.kld_max", kmax);
        rep.kv(test.name + ".steps.top1", (double) n_top1/n);
    }

    for (size_t f = 0; f < ref.feats.size() && f < test.feats.size(); ++f) {
        double m = 0.0;
        const size_t n = std::min(ref.feats[f].size(), test.feats[f].size());
        for (size_t j = 0; j < n; ++j) {
            m = std::max(m, (double) std::fabs(ref.feats[f][j] - test.feats[f][j]));
        }
        printf("    features #%zu (%d rows): max |d| %.3e\n", f, ref.n_feat_rows, m);
        rep.kv(test.name + ".feat" + std::to_string(f) + ".max_abs", m);
    }

    dsv4_state sa, sb;
    if (!parse_dsv4_any(ref.state, sa) || !parse_dsv4_any(test.state, sb)) {
        printf("    state: not compared (format; %zu / %zu bytes, byte-identical %s)\n", ref.state.size(), test.state.size(),
                ref.state == test.state ? "yes" : "no");
        return;
    }

    // window K: the cells of both states by position (the state holds the last n_swa positions of each layer)
    std::map<int32_t, uint64_t> cell_b;
    for (size_t i = 0; i < sb.swa.pos.size(); ++i) {
        cell_b[sb.swa.pos[i]] = i;
    }
    std::vector<std::pair<uint64_t, uint64_t>> pairs;
    for (size_t i = 0; i < sa.swa.pos.size(); ++i) {
        const auto it = cell_b.find(sa.swa.pos[i]);
        if (it != cell_b.end()) {
            pairs.emplace_back(i, it->second);
        }
    }
    const int32_t pmin = sa.swa.pos.empty() ? -1 : *std::min_element(sa.swa.pos.begin(), sa.swa.pos.end());
    const int32_t pmax = sa.swa.pos.empty() ? -1 : *std::max_element(sa.swa.pos.begin(), sa.swa.pos.end());
    printf("    window K (positions %d..%d, %zu/%zu cells matched):\n", pmin, pmax, pairs.size(), sa.swa.pos.size());
    double enc_max = 0.0, dec_max = 0.0;
    for (size_t il = 0; il < sa.swa.layers.size() && il < sb.swa.layers.size(); ++il) {
        const rows_cmp c = cmp_rows(sa.swa.layers[il], sb.swa.layers[il], pairs);
        double & m = il >= (size_t) dec0 ? dec_max : enc_max;
        m = std::max(m, c.max_abs);
        if (il >= (size_t) dec0 || c.n_diff > 0) {
            printf("      layer %2zu %s: max |d| %.3e (max |ref| %.3e), %" PRIu64 "/%" PRIu64 " rows differ%s\n", il,
                    il >= (size_t) dec0 ? "dec" : "enc", c.max_abs, c.max_ref, c.n_diff, c.n_rows, c.shape_ok ? "" : ", SHAPE MISMATCH");
        }
        rep.kv(test.name + ".win.l" + std::to_string(il) + ".max_abs", c.max_abs);
    }
    printf("      encoder max |d| %.3e, decoder max |d| %.3e\n", enc_max, dec_max);

    // compressed rows of every kv source (all positions), and the compressor states
    static const char * names[3] = { "csa", "hca", "lid" };
    for (int c = 0; c < 3; ++c) {
        for (size_t l = 0; l < sa.kcache[c].size() && l < sb.kcache[c].size(); ++l) {
            const auto & la = sa.kcache[c][l];
            const auto & lb = sb.kcache[c][l];
            const uint64_t n = std::min(la.blocks[0].n_rows, lb.blocks[0].n_rows);
            const rows_cmp r = cmp_rows(la.blocks[0], lb.blocks[0], same_rows(n));
            printf("    %s rows layer %2u%s: %" PRIu64 " rows, max |d| %.3e (max |ref| %.3e), %" PRIu64 " differ%s\n", names[c], la.il,
                    (int) la.il == dec0 ? " (decoder global KV)" : "", n, r.max_abs, r.max_ref, r.n_diff,
                    la.blocks[0].n_rows == lb.blocks[0].n_rows && r.shape_ok ? "" : ", SHAPE MISMATCH");
            rep.kv(test.name + "." + names[c] + ".l" + std::to_string(la.il) + ".max_abs", r.max_abs);
        }
        for (size_t l = 0; l < sa.cstate[c].size() && l < sb.cstate[c].size(); ++l) {
            for (int k = 0; k < 2; ++k) {
                const auto & ba = sa.cstate[c][l].blocks[k];
                const auto & bb = sb.cstate[c][l].blocks[k];
                const rows_cmp r = cmp_rows(ba, bb, same_rows(std::min(ba.n_rows, bb.n_rows)));
                if (r.n_diff > 0 || !r.shape_ok) {
                    printf("    %s state layer %2u %s: max |d| %.3e, %" PRIu64 " rows differ\n", names[c], sa.cstate[c][l].il,
                            k ? "score" : "kv", r.max_abs, r.n_diff);
                }
            }
        }
    }
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx = 73728;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }

    const std::vector<int> lens   = env_ints("CED_LENS", "1024,4096,16384,65536");
    const std::vector<int> turns  = env_ints("CED_TURNS", "8192:64:600,8192:64:4000");
    const std::vector<int> feats  = env_ints("CED_FEAT", "37,38,39");
    const int n_steps = env_int("CED_STEPS", 128);
    const int split   = env_int("CED_SPLIT", 1);
    const int reps    = std::max(1, env_int("CED_REPS", 1));
    const int dec0    = env_int("CED_DEC0", 20);
    const int aa      = env_int("CED_AA", 0);
    const int ced_only = env_int("CED_ONLY", 0); // only the CED runs (with CED_DUMP: compare runs of two processes)
    const char * dump = getenv("CED_DUMP");
    const auto save = [&](const run_result & r) {
        if (dump && *dump) {
            const std::string fn = std::string(dump) + r.name + ".state";
            FILE * f = fopen(fn.c_str(), "wb");
            if (f) {
                fwrite(r.state.data(), 1, r.state.size(), f);
                fclose(f);
            }
            FILE * g = fopen((std::string(dump) + r.name + ".logits").c_str(), "wb");
            if (g) {
                fwrite(r.logits_last.data(), sizeof(float), r.logits_last.size(), g);
                fclose(g);
            }
        }
    };

    // one prompt call holds the longest prompt
    int n_need = 0;
    for (int l : lens) {
        n_need = std::max(n_need, l);
    }
    for (size_t i = 0; i + 2 < turns.size(); i += 3) {
        n_need = std::max({ n_need, turns[i], turns[i + 2] });
    }
    params.n_batch    = std::max(params.n_batch, n_need);
    params.n_parallel = 1;
    params.warmup     = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    // the context may be recreated by the harness (CED mode change), so the init result must not free it: it is
    // released (leaked) and the harness frees the live context at the end
    const char * ced_env = getenv("LLAMA_DSV41_CED");
    const bool ced_start = !(ced_env && !strcmp(ced_env, "0")); // default on

    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to load the model\n");
        return 1;
    }

    harness h;
    h.ctx         = ctx;
    h.model       = model;
    h.cparams     = common_context_params_to_llama(params);
    h.ced_cur     = ced_start;
    h.n_vocab     = llama_vocab_n_tokens(llama_model_get_vocab(model));
    h.n_embd      = llama_model_n_embd(model);
    h.n_batch     = (int) llama_n_batch(ctx);
    h.n_ubatch    = (int) llama_n_ubatch(ctx);
    h.feat_layers = feats;
    h.restore     = env_int("CED_RESTORE", 0) != 0;
    h.use_target  = env_int("CED_TARGET", 0) != 0;
    for (int l : feats) {
        llama_set_embeddings_layer_inp(ctx, (uint32_t) l, true);
    }

    // the text, repeated if it is too short for the longest script
    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, true);
    int n_max = 0;
    for (int l : lens) {
        n_max = std::max(n_max, l + n_steps);
    }
    for (size_t i = 0; i + 2 < turns.size(); i += 3) {
        n_max = std::max(n_max, turns[i] + turns[i + 1] + turns[i + 2] + n_steps);
    }
    if (toks.empty()) {
        LOG_ERR("no prompt text (-f FILE)\n");
        return 1;
    }
    while ((int) toks.size() < n_max) {
        const std::vector<llama_token> copy(toks.begin(), toks.begin() + std::min(toks.size(), (size_t) n_max - toks.size()));
        toks.insert(toks.end(), copy.begin(), copy.end());
    }
    if (n_max > (int) llama_n_ctx(ctx)) {
        LOG_ERR("-c %u is too small, the longest script needs %d\n", llama_n_ctx(ctx), n_max);
        return 1;
    }

    report rep;
    const char * out = getenv("CED_OUT");
    rep.json = fopen(out ? out : "ced.json", "w");
    if (rep.json) {
        fprintf(rep.json, "{");
    }

    printf("llama-ced-check: n_batch %d, n_ubatch %d, steps %d, split %d, reps %d\n", h.n_batch, h.n_ubatch, n_steps, split, reps);

    std::vector<std::pair<std::string, std::vector<std::pair<int, int>>>> scripts;
    for (int l : lens) {
        scripts.push_back({ "p" + std::to_string(l), { { l, 0 } } });
    }
    for (size_t i = 0; i + 2 < turns.size(); i += 3) {
        scripts.push_back({ "t" + std::to_string(turns[i]) + "+" + std::to_string(turns[i + 1]) + "+" + std::to_string(turns[i + 2]),
                { { turns[i], 0 }, { turns[i + 1], 1 }, { turns[i + 2], 0 } } });
    }

    std::vector<std::string> bench;
    for (const auto & [name, script] : scripts) {
        printf("\n== %s\n", name.c_str());

        run_result a, b, c, d;
        a.name = name + ".old";
        b.name = name + ".ced";
        c.name = name + ".old_split";
        d.name = name + ".ced_split";

        double t_old = 0.0, t_new = 0.0;
        for (int r = 0; r < reps; ++r) {
            // only the first repetition keeps its results
            run_result tmp_a, tmp_b;
            run_result & ra = r == 0 ? a : tmp_a;
            run_result & rb = r == 0 ? b : tmp_b;
            ra.name = a.name;
            rb.name = b.name;
            if ((!ced_only && !h.run(ra, false, toks, script, false, r == 0 ? n_steps : 0)) ||
                !h.run(rb, true,  toks, script, false, r == 0 ? n_steps : 0)) {
                return 1;
            }
            t_old += ra.t_prefill;
            t_new += rb.t_prefill;
            printf("  rep %d: last prompt call %d tokens: old %.2f s (%.1f t/s), CED %.2f s (%.1f t/s), x%.2f\n", r, ra.n_prefill,
                    ra.t_prefill, ra.n_prefill/ra.t_prefill, rb.t_prefill, rb.n_prefill/rb.t_prefill, ra.t_prefill/rb.t_prefill);
        }
        rep.kv(name + ".t_old", t_old/reps);
        rep.kv(name + ".t_ced", t_new/reps);
        char line[256];
        snprintf(line, sizeof(line), "%-22s %6d tok  old %8.2f s  CED %8.2f s  x%.2f", name.c_str(), a.n_prefill, t_old/reps, t_new/reps, t_old/t_new);
        bench.push_back(line);

        save(a);
        save(b);
        if (ced_only) {
            if (split) {
                if (!h.run(d, true, toks, script, true, n_steps)) {
                    return 1;
                }
                save(d);
            }
            continue;
        }
        compare(a, b, dec0, rep);

        if (aa) {
            run_result a2;
            a2.name = name + ".old2";
            if (!h.run(a2, false, toks, script, false, n_steps)) {
                return 1;
            }
            save(a2);
            printf("  old again: %.2f s\n", a2.t_prefill);
            compare(a, a2, dec0, rep);
        }

        if (split) {
            if (!h.run(c, false, toks, script, true, n_steps) || !h.run(d, true, toks, script, true, n_steps)) {
                return 1;
            }
            printf("  split: old %.2f s, CED %.2f s\n", c.t_prefill, d.t_prefill);
            compare(a, c, dec0, rep); // noise floor: other ubatch shapes, same math
            compare(a, d, dec0, rep);
            compare(c, d, dec0, rep); // the same split, CED against the full decoder
        }
    }

    printf("\n== prefill (mean of %d)\n", reps);
    for (const auto & l : bench) {
        printf("  %s\n", l.c_str());
    }

    if (rep.json) {
        fprintf(rep.json, "}\n");
        fclose(rep.json);
    }

    llama_free(h.ctx);
    init.release(); // the model and its context are freed by the process exit
    llama_backend_free();
    return 0;
}
