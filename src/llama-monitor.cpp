#include "llama-monitor.h"

#include "llama-context.h"

#include "ggml.h"

#include <algorithm>
#include <cstring>

static constexpr size_t MONITOR_RING = 8192;

llama_monitor::llama_monitor(int32_t n_layer, int32_t n_expert, int32_t n_expert_used)
    : n_layer(n_layer), n_expert(n_expert), n_expert_used(n_expert_used) {}

void llama_monitor::set_topics(uint32_t t) {
    std::lock_guard<std::mutex> lk(mtx);
    if ((t & LLAMA_MONITOR_TOPIC_TIMELINE) && ring.empty()) {
        ring.resize(MONITOR_RING);
    }
    if (!stat && n_layer > 0) {
        stat.reset(new layer_stat[n_layer]);
    }
    if ((t & LLAMA_MONITOR_TOPIC_EXPERTS) && !heat && n_layer > 0 && n_expert > 0) {
        heat.reset(new std::atomic<uint32_t>[(size_t) n_layer*n_expert]());
    }
    if (t == 0) {
        head = size = 0;
    }
    topics.store(t, std::memory_order_relaxed);
}

void llama_monitor::emit(llama_monitor_event ev) {
    std::lock_guard<std::mutex> lk(mtx);
    if (ring.empty()) {
        return;
    }
    ev.seq = next_seq++;
    if (size == ring.size()) {
        head = (head + 1) % ring.size();
        size--;
        dropped++;
    }
    ring[(head + size) % ring.size()] = ev;
    size++;
}

void llama_monitor::emit(uint8_t type, uint8_t kind, int64_t t_us, int64_t dur_us, int32_t n0, int32_t n1, int32_t n2, int32_t n3,
        int64_t x0, int64_t x1, int64_t x2) {
    llama_monitor_event ev = {};
    ev.type   = type;
    ev.kind   = kind;
    ev.t_us   = t_us;
    ev.dur_us = dur_us;
    ev.n[0] = n0; ev.n[1] = n1; ev.n[2] = n2; ev.n[3] = n3;
    ev.x[0] = x0; ev.x[1] = x1; ev.x[2] = x2;
    emit(ev);
}

int32_t llama_monitor::drain(llama_monitor_event * out, int32_t max, uint64_t * n_dropped) {
    std::lock_guard<std::mutex> lk(mtx);
    const int32_t n = (int32_t) std::min<size_t>(size, std::max<int32_t>(max, 0));
    for (int32_t i = 0; i < n; ++i) {
        out[i] = ring[(head + i) % ring.size()];
    }
    head = ring.empty() ? 0 : (head + n) % ring.size();
    size -= n;
    if (n_dropped) {
        *n_dropped = dropped;
    }
    dropped = 0;
    return n;
}

void llama_monitor::layer_done(int il, int distinct, int hit_gpu, int hit_ram, int miss, uint64_t routes) {
    if (il < 0 || il >= n_layer || !stat) {
        return;
    }
    auto & s = stat[il];
    s.distinct.store(distinct, std::memory_order_relaxed);
    s.routes .fetch_add(routes,  std::memory_order_relaxed);
    s.hit_gpu.fetch_add(hit_gpu, std::memory_order_relaxed);
    s.hit_ram.fetch_add(hit_ram, std::memory_order_relaxed);
    s.miss   .fetch_add(miss,    std::memory_order_relaxed);
}

void llama_monitor::set_layer(int il, int n_expert_l, int n_cached, int n_hot) {
    if (il < 0 || il >= n_layer || !stat) {
        return;
    }
    auto & s = stat[il];
    s.moe.store(1, std::memory_order_relaxed);
    s.n_expert.store(n_expert_l, std::memory_order_relaxed);
    s.n_cached.store(n_cached, std::memory_order_relaxed);
    s.n_hot.store(n_hot, std::memory_order_relaxed);
}

void llama_monitor::layout(llama_monitor_layout * out) const {
    out->n_layer       = n_layer;
    out->n_expert      = n_expert;
    out->n_expert_used = n_expert_used;
    out->n_moe_layer   = 0;
}

void llama_monitor::snapshot(llama_monitor_layer * layers, uint32_t * counts) const {
    for (int il = 0; il < n_layer; ++il) {
        llama_monitor_layer l = {};
        if (stat) {
            const auto & s = stat[il];
            l.moe      = s.moe.load(std::memory_order_relaxed);
            l.n_expert = s.n_expert.load(std::memory_order_relaxed);
            l.n_cached = s.n_cached.load(std::memory_order_relaxed);
            l.n_hot    = s.n_hot.load(std::memory_order_relaxed);
            l.distinct = s.distinct.load(std::memory_order_relaxed);
            l.routes   = s.routes.load(std::memory_order_relaxed);
            l.hit_gpu  = s.hit_gpu.load(std::memory_order_relaxed);
            l.hit_ram  = s.hit_ram.load(std::memory_order_relaxed);
            l.miss     = s.miss.load(std::memory_order_relaxed);
        }
        layers[il] = l;
    }
    if (counts) {
        for (size_t i = 0; i < (size_t) n_layer*n_expert; ++i) {
            counts[i] = heat ? heat[i].load(std::memory_order_relaxed) : 0;
        }
    }
}

//
// C API
//

int64_t llama_monitor_now_us(void) {
    return ggml_time_us();
}

void llama_monitor_set_topics(llama_context * ctx, uint32_t topics) {
    ctx->monitor_set_topics(topics);
}

void llama_monitor_get_layout(const llama_context * ctx, llama_monitor_layout * out) {
    ctx->monitor_layout(out);
}

int32_t llama_monitor_drain(llama_context * ctx, llama_monitor_event * out, int32_t max, uint64_t * n_dropped) {
    return ctx->monitor->drain(out, max, n_dropped);
}

void llama_monitor_emit(llama_context * ctx, const llama_monitor_event * ev) {
    if (ctx->monitor->on(LLAMA_MONITOR_TOPIC_TIMELINE)) {
        ctx->monitor->emit(*ev);
    }
}

void llama_monitor_snapshot(llama_context * ctx, llama_monitor_layer * layers, uint32_t * counts, uint8_t * where) {
    ctx->monitor_snapshot(layers, counts, where);
}
