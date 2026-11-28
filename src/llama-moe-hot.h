#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct llama_model;
class llama_moe_stream;

struct llama_moe_hot_params {
    size_t   headroom;        // VRAM left free on each GPU after the banks are allocated
    uint64_t update_interval; // processed tokens between replacement checks, 0 = contents fixed for the phase
    size_t   update_bytes;    // GPU upload + download bytes planned per replacement check
    size_t   inflight_bytes;  // bytes of the transfers in progress at a time
    bool     verify;          // read every upload back and compare it with its source
};

// a GPU tier for the experts of the MoE stream during decode: the most used streamed experts are executed from
// banks on the GPUs, the others keep the CPU path of the stream. the banks are allocated once per small-batch
// phase, one bank per weight layout (role, type, shape) and device, and the GGUF stays the backing store of every
// expert: the banks are released without copies
//
// the contents change without stopping decode: a thread per GPU copies the experts on its own stream while the
// batches are computed, and the maps of the graph (which expert is in which slot) only change between batches:
// - an expert is published once its upload completed
// - an expert that leaves keeps its slot until it was downloaded into the RAM slot reserved for it (if it has no RAM
//   copy); its slot is written again once the graphs computed before it left are done (an event of the compute stream)
class llama_moe_hot {
public:
    // the banks and the GPU branches of each layer are planned with the most used experts of the stream, which are
    // then uploaded in the background; call between graphs
    llama_moe_hot(const llama_model & model, const std::vector<ggml_backend_t> & backends,
            llama_moe_stream & stream, const llama_moe_hot_params & params);
    ~llama_moe_hot();

    bool empty() const;
    int  n_devices() const;
    ggml_backend_t backend(int device) const;
    // the GPU backend that owns layer il, nullptr if the layer is not on a GPU in the tier
    ggml_backend_t home_backend(int il) const;

    // the layer has a GPU branch, on the device or on any device (-1). a branch stays once opened
    bool has_layer(int il, int device = -1) const;
    // the expert is in the tier (on any device)
    bool contains(int il, int expert) const;

    // the bank of the device with the layout of the expert tensor w, nullptr if none
    ggml_tensor * bank(const ggml_tensor * w, int device) const;

    // the bank slots on the device of the selected experts [k, n]: the slot of the experts it holds, -1 for the others
    ggml_tensor * build_ids(ggml_context * ctx, ggml_tensor * selected, int il, int device) const;

    // routing observations from the CPU path of the stream
    void begin_batch();
    void record_routes(int il, const ggml_tensor * selected);
    void record_ram_hits(int64_t routes); // routes of the CPU path that found their expert in RAM
    void abort_batch();
    void end_batch(int64_t tokens, int64_t interval);

    // between batches: publish the completed transfers, retire the slots of the downloaded experts and start new
    // transfers. true if a GPU branch was opened (the graph topology changed)
    bool tick(llama_moe_stream & stream);
    // between batches: finish the transfers in progress, drop the planned ones. true if the topology changed
    bool drain(llama_moe_stream & stream);

    // between batches: plan the replacements (started by the next ticks)
    bool update_due() const;
    void update(llama_moe_stream & stream);

    // totals since the start: experts published to the GPU tier, failed transfers, bytes RAM to GPU, GPU to RAM, file to RAM
    void totals(uint64_t & promoted, uint64_t & failed, uint64_t & up, uint64_t & down, uint64_t & file) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
