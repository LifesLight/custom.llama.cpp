#pragma once

#include "llama.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

// event ring and expert counters behind the llama_monitor_* API. owned by llama_context. nothing is allocated
// until a topic is set, and with no topic every emit site is one relaxed load
class llama_monitor {
public:
    llama_monitor(int32_t n_layer, int32_t n_expert, int32_t n_expert_used);

    bool on(uint32_t topic) const { return (topics.load(std::memory_order_relaxed) & topic) != 0; }

    void set_topics(uint32_t t);

    // ring: when full the oldest events are dropped. call only when on(TIMELINE)
    void emit(llama_monitor_event ev);
    void emit(uint8_t type, uint8_t kind, int64_t t_us, int64_t dur_us, int32_t n0 = 0, int32_t n1 = 0, int32_t n2 = 0, int32_t n3 = 0,
            int64_t x0 = 0, int64_t x1 = 0, int64_t x2 = 0);
    int32_t drain(llama_monitor_event * out, int32_t max, uint64_t * n_dropped);

    // the experts of layer il selected by a ubatch: counts[e] selections for each distinct expert in ids
    // (call only when on(EXPERTS)); the per layer outcome of the distinct experts
    void count(int il, int32_t e, uint32_t n) {
        if (il >= 0 && il < n_layer && e >= 0 && e < n_expert && heat) {
            heat[(size_t) il*n_expert + e].fetch_add(n, std::memory_order_relaxed);
        }
    }
    void layer_done(int il, int distinct, int hit_gpu, int hit_ram, int miss, uint64_t routes);

    // static facts of the layer, set by the context
    void set_layer(int il, int n_expert_l, int n_cached, int n_hot);

    void layout(llama_monitor_layout * out) const;
    // counts / layers copied, where is filled by the caller
    void snapshot(llama_monitor_layer * layers, uint32_t * counts) const;

    const int32_t n_layer, n_expert, n_expert_used;

private:
    struct layer_stat {
        std::atomic<int32_t>  distinct{0};
        std::atomic<uint64_t> routes{0}, hit_gpu{0}, hit_ram{0}, miss{0};
        std::atomic<int32_t>  moe{0}, n_expert{0}, n_cached{0}, n_hot{0};
    };

    std::atomic<uint32_t> topics{0};

    std::mutex                          mtx;
    std::vector<llama_monitor_event>    ring; // capacity fixed at the first topic
    size_t                              head = 0, size = 0;
    uint32_t                            next_seq = 0;
    uint64_t                            dropped = 0;

    std::unique_ptr<std::atomic<uint32_t>[]> heat;
    std::unique_ptr<layer_stat[]>            stat;
};
