#pragma once

#include "llama.h"

#include "ggml-backend.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
class llama_moe_hot;

struct llama_moe_stream_params {
    int32_t     n_threads; // IO threads
    size_t      resident;  // bytes of the expert cache in RAM, SIZE_MAX = from the available RAM
    size_t      reserve;   // with resident = SIZE_MAX: RAM left available
    enum llama_moe_stream_policy policy;
    std::string state;     // file to save and restore the cached experts, empty = next to the model, "none" = disabled
    float       pool = -1.0f; // >= 0: layers with the same expert layout share a pool of cache slots, each keeping at least
                              // this fraction of its uniform share; < 0: fixed per-layer caches
};

// keeps the MoE experts of host-resident layers in an expert cache in RAM, filled with O_DIRECT reads,
// bypassing the page cache of the memory mapped model files entirely
//
// - every layer has the same fraction of its experts as cache slots (gate, up and down of an expert together),
//   or with a pool (--moe-stream-pool F), the layers with the same experts layout share their slots with one
//   replacement policy, each keeping at least F of its uniform share; the size of the cache does not change
// - batches of up to LLAMA_MOE_STREAM_CPU_MAX tokens are computed on the CPU from the cache: an op between the
//   router and the expert matmuls reads the missing experts in parallel into the slots of evicted ones (lru, slru
//   or lfu policy), or into an overflow window over the staging buffers, and remaps the selected experts
// - larger batches are offloaded to the GPU: the cached experts are uploaded from the cache, the others are read
//   with O_DIRECT by a pool of IO threads into a ring of pinned staging buffers, one tensor per buffer, most used
//   first while the previous tensors are computed; once the router selected the experts, only the missing selected
//   ones are read. the cache does not change during prompt processing
// the router of the next layer, to predict its experts from the input of the router of a layer (cross-layer prefetch)
struct llama_moe_stream_prefetch {
    ggml_tensor * gate_inp    = nullptr;
    ggml_tensor * gate_inp_b  = nullptr;
    ggml_tensor * exp_probs_b = nullptr;
    ggml_tensor * norm        = nullptr; // [n_embd] multiplies the input: norm weights of the next layer / of this layer
    int           k           = 0;       // number of experts to predict
};

class llama_moe_stream {
public:
    // backends are the backends of the context, the first GPU is the one the experts are offloaded to
    llama_moe_stream(const llama_model & model, const std::vector<ggml_backend_t> & backends, const llama_moe_stream_params & params);
    ~llama_moe_stream();

    // false if there is nothing to stream (no memory mapped experts)
    bool active() const;

    //
    // GPU expert tier (llama_moe_hot): called between synchronized graphs, except the observer callbacks
    //

    // the usage of the experts of a layer: their decayed selections (LLAMA_MOE_STREAM_HALFLIFE) by the batches
    // computed on the CPU, or by all the batches early on (as hot_experts), empty if the layer is not streamed
    std::vector<double> usage(int il) const;
    // the expert is in the RAM cache
    bool cached(int il, int expert) const;
    // save the cached experts and their usage now (with the experts of the GPU tier, which are not in RAM)
    void save_state();
    // the streamed expert tensors of a layer, sorted by name, empty if the layer is not streamed
    std::vector<const ggml_tensor *> layer_tensors(int il) const;
    // the tier observes the routing of the CPU path, its experts bypass the RAM cache and its prefetches, and their
    // RAM copies are the first victims of its misses
    void set_hot_observer(llama_moe_hot * observer);
    // all the layers compute a batch of that many tokens on the CPU
    bool cpu_batch(int64_t tokens) const;
    // the CPU path takes batches of up to that many tokens, if the cache allows, true if it changed
    bool set_hot_cpu_max(int64_t tokens);
    // finish the prefetches, so that the cache can change
    void quiesce_hot();

    //
    // lending RAM out, e.g. while the GPUs are suspended for another program: between synchronized graphs
    //

    // release about bytes of the expert cache: its least used experts are evicted (each layer keeps its least
    // slots) and the memory of their slots is unlocked and dropped. returns the bytes released
    size_t shrink_cache(size_t bytes);
    // take the released memory back in a background thread, as far as the available RAM minus the reserve allows: its
    // slots are free as soon as they are faulted in, and the misses take them before they evict. returns the bytes
    // released at the call
    size_t grow_cache();
    // read the most used experts that are not cached into up to bytes of the free slots. returns the bytes of the
    // free slots left, and of the released ones while they are taken back (0: nothing more to fill)
    size_t fill_free(size_t bytes);

    using expert_writer = std::function<void(const ggml_tensor *, const void *, size_t)>;
    // the tensors of an expert, from the RAM cache or read from the model file, false if not streamed
    bool with_expert(int il, int expert, const expert_writer & writer);
    // the tensors of the experts (il, expert) for writer(index, ...): from the RAM cache, the others read from the
    // model file in parallel, while the previous ones are written. false if one is not streamed
    using expert_batch_writer = std::function<void(size_t, const ggml_tensor *, const void *, size_t)>;
    bool with_experts(const std::vector<std::pair<int, int>> & experts, const expert_batch_writer & writer);
    // the experts of the tier changed
    void hot_changed();
    // expert e of layer il joined or left the tier: its RAM copy, if any, changes place in the eviction order
    void hot_changed(int il, int expert);

    // asynchronous transfers of the tier
    struct hold_t {
        int     pool = -1;
        int32_t slot = -1;
    };
    struct expert_part {
        const ggml_tensor * t;
        void *              data; // in the RAM slot, or in the buffer of read_expert
        size_t              size;
    };
    // between graphs: the RAM copy of the expert, held (not evicted) until release_hold, false if it is not cached
    // or its slot is being read into
    bool hold_expert(int il, int expert, std::vector<expert_part> & parts, hold_t & hold);
    // any thread
    void release_hold(hold_t & hold);
    // between graphs: a RAM slot for the expert, written by any thread (parts) before commit_ram: the slot that the
    // policy evicts first, its expert is evicted now. false if the expert is cached or no slot can take it
    bool reserve_ram(int il, int expert, std::vector<expert_part> & parts, hold_t & hold);
    // between graphs: the reserved slot holds the expert
    void commit_ram(int il, int expert, hold_t & hold);
    // any thread: read the tensors of the expert from the model file into buf, at the lowest priority of the IO
    // threads, false on a read error
    bool read_expert(int il, int expert, std::vector<uint8_t> & buf, std::vector<expert_part> & parts);

    // true if src is an expert tensor handled by the stream
    bool is_streamed(const ggml_tensor * src) const;

    // the cache of the experts of w, nullptr if w is not cached
    ggml_tensor * get_bank(const ggml_tensor * w) const;

    // the backend the expert matmuls of w are computed on when they are not computed from the cache, nullptr if not streamed
    // (small batches would otherwise be computed on the CPU from the memory map)
    ggml_backend_t offload_backend(const ggml_tensor * w) const;

    // for a batch computed on the CPU from the cache: the slots of the selected experts of layer il, as an op that
    // makes them resident when it is computed. nullptr if the layer cannot be computed from the cache.
    ggml_tensor * build_slots(ggml_context * ctx, ggml_tensor * selected_experts, int il) const;

    // LLAMA_MOE_STREAM_OVERLAP: the slots of build_slots are -1 for the experts being read, the cached experts are
    // computed while they are read. build_wait waits for them once deps are computed and gives [k, n, 2] slots:
    // [:, :, 0] -1 except for the experts that were read, [:, :, 1] all the slots. nullptr if not overlapping
    bool overlap() const;

    // LLAMA_MOE_STREAM_PROFILE=1: an op computed right after the expert matmuls of layer il on the CPU (cur is their
    // result) that times them, for the profile of the decode logged with the decode stats. nullptr if not profiling
    ggml_tensor * build_profile_mark(ggml_context * ctx, ggml_tensor * cur, int il) const;

    // cross-layer prefetch (LLAMA_MOE_STREAM_PREFETCH=K): for a batch of n_tokens of layer il computed on the CPU from
    // the cache, the router of the next layer, false if its experts are not prefetched
    bool prefetch_info(int il, int64_t n_tokens, llama_moe_stream_prefetch & out) const;

    // prefill forecast (LLAMA_MOE_STREAM_FORECAST=1): for a batch of n_tokens of layer il computed on the GPU, the
    // router of the next layer (to apply to the router input of layer il), false if the next layer is not forecast
    bool forecast_info(int il, int64_t n_tokens, llama_moe_stream_prefetch & out) const;

    // the op (on the CPU) that gives the forecast ids (top-k of the router of the next layer) to the stream, which
    // reads ahead the predicted experts of the next layer first
    ggml_tensor * build_forecast(ggml_context * ctx, ggml_tensor * ids, int il) const;

    // the op that reads the experts predicted by ids (top-k of the router of the next layer) into the cache, computed
    // after `after` (the slots of layer il) so that the reads of layer il go first
    ggml_tensor * build_prefetch(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * after, int il) const;

    // with overlap: the cached experts also go through down before the wait (LLAMA_MOE_STREAM_OVERLAP_DOWN=1, default off)
    bool overlap_down() const;

    // true if a batch of n_tokens of layer il is computed on the CPU from the cache (see build_slots)
    bool cpu_path(int il, int64_t n_tokens) const;
    ggml_tensor * build_wait(ggml_context * ctx, ggml_tensor * slots, const std::vector<ggml_tensor *> & deps, int il) const;

    // called before each graph compute, after the graph is allocated: the tensors are prefetched in the order
    // of the expert matmuls that the scheduler assigned to the GPU
    // expect_more is a hint that another ubatch will follow, so its first tensors can be prefetched
    void begin_graph(ggml_backend_sched_t sched, ggml_cgraph * gf, int64_t n_tokens, bool expect_more);
    void end_graph(bool success = true);

    // ggml_backend_sched copy callback for the experts of src
    // counts: the tokens per expert selected by the router (nullptr = copy all), dst: the copy of src on the device
    void copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, const std::vector<int32_t> * counts);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_stream_ptr = std::unique_ptr<llama_moe_stream>;
