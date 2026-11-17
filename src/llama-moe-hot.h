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
    size_t   update_bytes;    // GPU upload + download bytes per replacement check
};

// a GPU tier for the experts of the MoE stream during decode: the most used streamed experts are executed from
// banks on the GPUs, the others keep the CPU path of the stream. the banks are allocated once per small-batch
// phase, one bank per weight layout (role, type, shape) and device; their contents change only between
// synchronized batches, and the GGUF stays the backing store of every expert
class llama_moe_hot {
public:
    // seeded with the most used experts of the stream; call between synchronized graphs
    llama_moe_hot(const llama_model & model, const std::vector<ggml_backend_t> & backends,
            llama_moe_stream & stream, const llama_moe_hot_params & params);
    ~llama_moe_hot();

    bool empty() const;
    int  n_devices() const;
    ggml_backend_t backend(int device) const;

    // the layer has experts in the tier, on the device or on any device (-1)
    bool has_layer(int il, int device = -1) const;
    // the expert is in the tier (on any device)
    bool contains(int il, int expert) const;

    // the bank of the device with the layout of the expert tensor w, nullptr if none
    ggml_tensor * bank(const ggml_tensor * w, int device) const;

    // the bank slots on the device of the selected experts [1, n]: the slot of the experts it holds, the zero
    // expert for the others
    ggml_tensor * build_ids(ggml_context * ctx, ggml_tensor * selected, int il, int device) const;

    // routing observations from the CPU path of the stream
    void begin_batch();
    void record_routes(int il, const ggml_tensor * selected);
    void record_ram_hits(int64_t routes); // routes of the CPU path that found their expert in RAM
    void abort_batch();
    void end_batch(int64_t tokens, int64_t interval);

    // between synchronized batches
    bool update_due() const;
    bool update(llama_moe_stream & stream, double synchronize_ms); // true if the graph topology changed
    void make_exclusive(llama_moe_stream & stream);                // repurpose the RAM copies of GPU experts
    void demote_all(llama_moe_stream & stream);                    // admit the GPU experts to RAM before release

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
