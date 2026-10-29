#pragma once

#include "llama.h"

#include "ggml-backend.h"

#include <memory>
#include <string>
#include <vector>

struct llama_model;

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
class llama_moe_stream {
public:
    // backends are the backends of the context, the first GPU is the one the experts are offloaded to
    llama_moe_stream(const llama_model & model, const std::vector<ggml_backend_t> & backends, const llama_moe_stream_params & params);
    ~llama_moe_stream();

    // false if there is nothing to stream (no memory mapped experts)
    bool active() const;

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

    // true if a batch of n_tokens of layer il is computed on the CPU from the cache (see build_slots)
    bool cpu_path(int il, int64_t n_tokens) const;
    ggml_tensor * build_wait(ggml_context * ctx, ggml_tensor * slots, const std::vector<ggml_tensor *> & deps, int il) const;

    // called before each graph compute, after the graph is allocated: the tensors are prefetched in the order
    // of the expert matmuls that the scheduler assigned to the GPU
    // expect_more is a hint that another ubatch will follow, so its first tensors can be prefetched
    void begin_graph(ggml_backend_sched_t sched, ggml_cgraph * gf, int64_t n_tokens, bool expect_more);
    void end_graph();

    // ggml_backend_sched copy callback for the experts of src
    // counts: the tokens per expert selected by the router (nullptr = copy all), dst: the copy of src on the device
    void copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, const std::vector<int32_t> * counts);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_stream_ptr = std::unique_ptr<llama_moe_stream>;
