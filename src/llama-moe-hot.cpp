#include "llama-moe-hot.h"
#include "llama-moe-stream.h"
#include "llama-model.h"
#include "llama-impl.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

// the usage of an expert is the one of the stream: its decayed selections, the order of its lfu policy. a candidate
// replaces a victim if it is used more by this factor and by this many selections: routing noise does not churn
static constexpr double HOT_MARGIN   = 1.25;
static constexpr double HOT_MIN_GAIN = 2.0;

static constexpr double MiB = 1024.0*1024.0;

// copy threads per GPU, each with its own stream: one reads or copies while the other waits for its transfer
static constexpr int HOT_COPIERS = 2;

struct llama_moe_hot::impl {
    // the slots of one weight layout on one device: a bank per weight tensor of the layout, [ne0, ne1, n_slots + 1],
    // the slots, then a trailing padding expert
    struct group {
        std::vector<const ggml_tensor *> refs; // the tensors of the first layer with the layout, per bank
        std::vector<ggml_tensor *>       banks;
        int32_t n_slots = 0;

        std::vector<std::pair<int32_t, int32_t>> owner; // [n_slots] (il, expert), il = -1 if empty
        std::vector<uint8_t>                     busy;  // [n_slots] a transfer uses the slot

        // update scratch: the slots in victim order (empty, then the coldest) and the next one to take
        std::vector<std::pair<double, int32_t>> victims;
        size_t next = 0;
    };

    // stages of a transfer: the thread of the device runs QUEUED (download the old expert into its RAM slot, or upload
    // if the slot is empty) and RETIRED (upload), the batches run DOWNLOADED (retire the slot) and UPLOADED (publish)
    enum stage { PENDING, QUEUED, DOWNLOADED, RETIRED, UPLOADED, FAILED };

    struct transfer {
        int      device;
        group *  g;
        int32_t  slot;
        int32_t  il, e;           // the expert uploaded into the slot
        int32_t  vil = -1, ve = -1; // the expert in the slot, admitted to RAM before the slot is written
        size_t   bytes = 0;       // of the experts moved, for the window of the transfers in progress
        bool     seed  = false;
        stage    st    = PENDING;

        llama_moe_stream::hold_t                   hold;   // the RAM slot the upload reads from
        std::vector<llama_moe_stream::expert_part> parts;  // the source of the upload
        std::vector<uint8_t>                       buf;    // the upload read from the model file
        llama_moe_stream::hold_t                   vhold;  // the RAM slot reserved for the old expert
        std::vector<llama_moe_stream::expert_part> vparts; // its tensors in that slot
    };

    struct device {
        ggml_backend_t     backend = nullptr;
        ggml_backend_dev_t dev     = nullptr;
        size_t budget  = 0;
        size_t used    = 0; // planned, then allocated bytes
        size_t payload = 0; // bytes of the expert slots

        std::map<std::string, group> groups; // by layout, node addresses are stable

        ggml_context_ptr        ctx;
        ggml_backend_buffer_ptr buffer;

        std::unordered_map<const ggml_tensor *, ggml_tensor *> bindings; // expert tensor of the model -> bank

        // the transfers: backends of the same GPU (their own streams) and the threads that drive them
        std::vector<ggml_backend_t> copies;
        std::vector<std::thread>    threads;
        ggml_backend_event_t    retire_ev = nullptr; // recorded on the compute stream when slots are retired
        std::mutex              mtx;
        std::condition_variable cv;
        bool                    stop = false;
        std::deque<transfer *>  queue; // QUEUED and RETIRED, guarded by mtx
        std::vector<transfer *> done;  // for the next tick, guarded by mtx
        double                  busy_ms = 0.0;

        // statistics: cumulative, at the last report, at the start of the batch
        uint64_t hits = 0, hit_bytes = 0;
        uint64_t logged_hits = 0, logged_hit_bytes = 0;
        uint64_t batch_hits = 0, batch_hit_bytes = 0;
    };

    struct layer {
        std::vector<const ggml_tensor *> refs; // the streamed expert tensors, sorted by name
        std::string key;                       // their layout, empty if the layer cannot be in the tier
        size_t      bytes = 0;                 // of an expert, all its tensors
        int         home_dev = -1;             // index in devices of model.dev_layer(il), -1 if none

        std::vector<int32_t> owner;    // [n_expert] device, -1 if not in the tier
        std::vector<int32_t> slot;     // [n_expert] slot in the group of its device
        std::vector<uint8_t> moving;   // [n_expert] a transfer uploads the expert
        std::vector<int32_t> n_owned;  // [n_devices]
        std::vector<uint8_t> branch;   // [n_devices] the graph has a GPU branch for the layer on the device
        std::vector<group *> group_of; // [n_devices] the group of its layout, nullptr if none

        std::vector<uint32_t> seen; // [n_expert] the distinct experts of a batch, for the byte statistics
        uint32_t generation = 0;
    };

    struct op_ctx {
        impl * self;
        int    il;
        int    device;
    };

    llama_moe_stream & stream;

    std::deque<device>               devices; // stable addresses, for the threads
    std::vector<layer>               layers;  // [il]
    std::vector<std::vector<op_ctx>> ops;     // [il][device]

    size_t n_cached = 0;

    uint64_t update_interval = 0;
    size_t   update_bytes    = 0;
    size_t   inflight_cap    = 0;
    bool     verify          = false;
    uint64_t last_update     = 0;

    std::deque<transfer *> pending; // planned, not started
    size_t inflight = 0;            // bytes of the started transfers

    // seeding
    size_t  n_seed = 0, n_seeded = 0, seed_file_bytes = 0;
    int64_t t_seed_start = 0;

    // statistics: cumulative, at the last report, at the start of the batch
    uint64_t tokens = 0, routes = 0, bytes = 0, ram_hits = 0;
    uint64_t logged_tokens = 0, logged_routes = 0, logged_bytes = 0, logged_ram_hits = 0;
    uint64_t batch_routes = 0, batch_bytes = 0, batch_ram_hits = 0;

    uint64_t n_updates = 0, promotions = 0, demotions = 0, blocked = 0, failed = 0; // demotions: GPU experts admitted to RAM
    std::atomic<size_t> upload_bytes { 0 }, download_bytes { 0 }, file_bytes { 0 };
    double   tick_ms = 0.0;

    // the layout of a set of expert tensors: the role (the name without the block prefix), type, shape and stride
    static std::string layout(const std::vector<const ggml_tensor *> & refs) {
        std::string key;
        for (const ggml_tensor * t : refs) {
            const char * role = strchr(t->name, '.');
            role = role ? strchr(role + 1, '.') : nullptr;
            key += format("%s:%d:%" PRId64 ":%" PRId64 ":%zu;", role ? role : t->name, (int) t->type, t->ne[0], t->ne[1], t->nb[2]);
        }
        return key;
    }

    // the allocation of a group with n_slots slots on a device
    static size_t group_size(const device & d, const std::vector<const ggml_tensor *> & refs, int32_t n_slots) {
        if (n_slots == 0) {
            return 0;
        }
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(d.backend);
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        size_t res = 0;
        for (const ggml_tensor * ref : refs) {
            ggml_tensor t = *ref;
            t.ne[2] = n_slots + 1;
            t.nb[3] = t.nb[2]*t.ne[2];
            res += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, &t), alignment);
        }
        return res;
    }

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, llama_moe_stream & stream,
            const llama_moe_hot_params & params) :
        stream(stream), layers(model.layers.size()), update_interval(params.update_interval), update_bytes(params.update_bytes),
        inflight_cap(params.inflight_bytes), verify(params.verify) {
        t_seed_start = ggml_time_us();

        for (ggml_backend_t backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            const auto type = ggml_backend_dev_type(dev);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                continue;
            }
            size_t free = 0;
            size_t total = 0;
            ggml_backend_dev_memory(dev, &free, &total);

            device & d = devices.emplace_back();
            d.backend = backend;
            d.dev     = dev;
            d.budget  = free > params.headroom ? free - params.headroom : 0;
            LLAMA_LOG_INFO("moe_hot: %s: %.0f MiB free, %.0f MiB headroom, budget %.0f MiB\n",
                    ggml_backend_name(backend), free/MiB, params.headroom/MiB, d.budget/MiB);
        }
        const size_t n_dev = devices.size();

        for (size_t il = 0; il < layers.size(); ++il) {
            auto & l = layers[il];
            l.refs = stream.layer_tensors((int) il);
            if (l.refs.empty()) {
                continue;
            }
            const int64_t n_expert = l.refs[0]->ne[2];
            bool uniform = true;
            for (const ggml_tensor * t : l.refs) {
                uniform = uniform && t->ne[2] == n_expert;
                l.bytes += t->nb[2];
            }
            ggml_backend_dev_t ldev = model.dev_layer((int) il);
            for (size_t id = 0; id < n_dev; ++id) {
                if (devices[id].dev == ldev) {
                    l.home_dev = (int) id;
                    break;
                }
            }
            l.key = uniform && n_expert > 0 ? layout(l.refs) : std::string();
            l.owner   .assign(n_expert, -1);
            l.slot    .assign(n_expert, -1);
            l.moving  .assign(n_expert, 0);
            l.seen    .assign(n_expert, 0);
            l.n_owned .assign(n_dev, 0);
            l.branch  .assign(n_dev, 0);
            l.group_of.assign(n_dev, nullptr);
        }

        if (n_dev == 0) {
            return;
        }

        // the most used experts first, wherever they are (an expert that is not in RAM is read from the model file,
        // e.g. after a prompt on another topic), each on the device with the most budget left after it: one owner per
        // expert, whatever device the layer is on. without usage, only the experts in RAM
        struct seed {
            double  usage;
            bool    cached;
            int32_t il;
            int32_t e;
        };
        std::vector<seed> seeds;
        for (size_t il = 0; il < layers.size(); ++il) {
            const auto & l = layers[il];
            if (l.key.empty()) {
                continue;
            }
            std::vector<double> usage = stream.usage((int) il);
            usage.resize(l.owner.size(), 0.0);
            for (size_t e = 0; e < l.owner.size(); ++e) {
                const bool cached = stream.cached((int) il, (int) e);
                if (usage[e] > 0.0 || cached) {
                    seeds.push_back({ usage[e], cached, (int32_t) il, (int32_t) e });
                }
            }
        }
        std::stable_sort(seeds.begin(), seeds.end(), [](const seed & a, const seed & b) {
            return a.usage != b.usage ? a.usage > b.usage : a.cached > b.cached;
        });

        struct placement {
            int32_t il;
            int32_t e;
            int     device;
            int32_t slot;
        };
        std::vector<placement> plan;
        for (const auto & sd : seeds) {
            auto & l = layers[sd.il];
            auto fits = [&](int id, size_t & cost_out) -> bool {
                const auto & d = devices[id];
                const auto it = d.groups.find(l.key);
                const int32_t n = it == d.groups.end() ? 0 : it->second.n_slots;
                const size_t cost = group_size(d, l.refs, n + 1) - group_size(d, l.refs, n);
                if (d.used + cost <= d.budget) {
                    cost_out = cost;
                    return true;
                }
                return false;
            };

            // place on the layer's home GPU first so secondary GPUs (e.g. PCIe 3.0 x4) stay local to their own layers
            int    best      = -1;
            size_t best_cost = 0;
            if (l.home_dev >= 0 && fits(l.home_dev, best_cost)) {
                best = l.home_dev;
            } else {
                size_t best_left = 0;
                for (size_t id = 0; id < n_dev; ++id) {
                    if ((int) id == l.home_dev) {
                        continue;
                    }
                    size_t cost = 0;
                    if (fits((int) id, cost)) {
                        const size_t left = devices[id].budget - devices[id].used - cost;
                        if (best < 0 || left > best_left) {
                            best      = (int) id;
                            best_cost = cost;
                            best_left = left;
                        }
                    }
                }
            }
            if (best < 0) {
                continue;
            }
            auto & d = devices[best];
            auto & g = d.groups[l.key];
            if (g.refs.empty()) {
                g.refs = l.refs;
            }
            d.used += best_cost;
            l.branch[best] = 1;
            plan.push_back({ sd.il, sd.e, best, g.n_slots++ });
            seed_file_bytes += sd.cached ? 0 : l.bytes;
        }

        for (auto & d : devices) {
            if (d.groups.empty()) {
                continue;
            }
            size_t n_tensors = 0;
            for (const auto & [key, g] : d.groups) {
                n_tensors += g.refs.size();
            }
            ggml_init_params ip = { n_tensors*ggml_tensor_overhead(), nullptr, true };
            d.ctx.reset(ggml_init(ip));
            if (!d.ctx) {
                throw std::runtime_error("moe_hot: metadata allocation failed");
            }
            for (auto & [key, g] : d.groups) {
                g.owner.assign(g.n_slots, { -1, -1 });
                g.busy .assign(g.n_slots, 0);
                for (const ggml_tensor * ref : g.refs) {
                    ggml_tensor * t = ggml_new_tensor_3d(d.ctx.get(), ref->type, ref->ne[0], ref->ne[1], g.n_slots + 1);
                    if (t->nb[1] != ref->nb[1] || t->nb[2] != ref->nb[2]) {
                        throw std::runtime_error("moe_hot: nonstandard expert layout");
                    }
                    g.banks.push_back(t);
                }
            }
            d.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), ggml_backend_get_default_buffer_type(d.backend)));
            if (!d.buffer) {
                throw std::runtime_error("moe_hot: device allocation failed");
            }
            d.used = ggml_backend_buffer_get_size(d.buffer.get());
            if (d.used > d.budget) {
                throw std::runtime_error("moe_hot: allocation exceeded the budget");
            }
            ggml_backend_buffer_set_usage(d.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_clear(d.buffer.get(), 0);
        }

        // every layer of a layout is bound to its banks, so that replacements never allocate
        for (auto & l : layers) {
            if (l.key.empty()) {
                continue;
            }
            for (size_t id = 0; id < n_dev; ++id) {
                auto & d = devices[id];
                const auto it = d.groups.find(l.key);
                if (it == d.groups.end()) {
                    continue;
                }
                l.group_of[id] = &it->second;
                for (size_t j = 0; j < l.refs.size(); ++j) {
                    d.bindings[l.refs[j]] = it->second.banks[j];
                }
            }
        }

        // the copy threads, then the seeds as transfers, most used first. the destructor does not run if the
        // constructor throws: the threads already started must be joined here
        try {
            for (auto & d : devices) {
                if (d.groups.empty()) {
                    continue;
                }
                d.retire_ev = ggml_backend_event_new(d.dev);
                for (int i = 0; i < HOT_COPIERS; ++i) {
                    d.copies.push_back(ggml_backend_dev_init(d.dev, nullptr));
                    if (!d.copies.back() || !d.retire_ev) {
                        throw std::runtime_error("moe_hot: copy stream unavailable");
                    }
                }
                for (ggml_backend_t copy : d.copies) {
                    d.threads.emplace_back([this, &d, copy] { copy_thread(d, copy); });
                }
            }
        } catch (...) {
            stop_threads();
            free_streams();
            throw;
        }
        for (const auto & p : plan) {
            auto & l = layers[p.il];
            auto * t = new transfer;
            t->device = p.device;
            t->g      = l.group_of[p.device];
            t->slot   = p.slot;
            t->il     = p.il;
            t->e      = p.e;
            t->bytes  = l.bytes;
            t->seed   = true;
            t->g->busy[p.slot] = 1;
            l.moving[p.e] = 1;
            pending.push_back(t);
        }
        n_seed = plan.size();

        ops.resize(layers.size());
        for (size_t il = 0; il < layers.size(); ++il) {
            ops[il].resize(n_dev);
            for (size_t id = 0; id < n_dev; ++id) {
                ops[il][id] = { this, (int) il, (int) id };
            }
        }

        for (const auto & d : devices) {
            int32_t n_slots = 0;
            for (const auto & [key, g] : d.groups) {
                n_slots += g.n_slots;
            }
            LLAMA_LOG_INFO("moe_hot: %s: %d slots in %zu layouts, %.0f MiB\n", ggml_backend_name(d.backend), n_slots, d.groups.size(), d.used/MiB);
        }
        LLAMA_LOG_INFO("moe_hot: seeding %zu experts in the background (%.0f MiB from the model file), planned in %.0f ms\n",
                n_seed, seed_file_bytes/MiB, (ggml_time_us() - t_seed_start)/1000.0);
    }

    void stop_threads() {
        for (auto & d : devices) {
            {
                std::lock_guard<std::mutex> lk(d.mtx);
                d.stop = true;
            }
            d.cv.notify_all();
            for (auto & th : d.threads) {
                th.join();
            }
            d.threads.clear();
        }
    }

    void free_streams() {
        for (auto & d : devices) {
            for (ggml_backend_t copy : d.copies) {
                if (copy) {
                    ggml_backend_synchronize(copy);
                    ggml_backend_free(copy);
                }
            }
            d.copies.clear();
            if (d.retire_ev) {
                ggml_backend_event_free(d.retire_ev);
                d.retire_ev = nullptr;
            }
        }
    }

    ~impl() {
        stop_threads();
        auto drop = [&](transfer * t) {
            stream.release_hold(t->hold);
            if (t->vhold.pool >= 0) {
                admit(*t, t->st != DOWNLOADED);
            }
            delete t;
        };
        for (auto * t : pending) { drop(t); }
        for (auto & d : devices) {
            for (auto * t : d.queue) { drop(t); }
            for (auto * t : d.done)  { drop(t); }
        }
        free_streams();
    }

    //
    // the copy thread of a device
    //

    void copy_thread(device & d, ggml_backend_t copy) {
        for (;;) {
            transfer * t = nullptr;
            {
                std::unique_lock<std::mutex> lk(d.mtx);
                d.cv.wait(lk, [&] { return d.stop || !d.queue.empty(); });
                if (d.stop) {
                    return;
                }
                t = d.queue.front();
                d.queue.pop_front();
            }
            const int64_t t0 = ggml_time_us();
            try {
                run(d, copy, *t);
            } catch (const std::exception & e) {
                LLAMA_LOG_WARN("moe_hot: transfer of expert %d of layer %d failed: %s\n", t->e, t->il, e.what());
                t->st = FAILED;
            }
            std::lock_guard<std::mutex> lk(d.mtx);
            d.busy_ms += (ggml_time_us() - t0)/1000.0;
            d.done.push_back(t);
        }
    }

    void run(device & d, ggml_backend_t copy, transfer & t) {
        if (t.st == QUEUED && t.vhold.pool >= 0) {
            // the old expert is still published: its slot is only read meanwhile
            for (const auto & p : t.vparts) {
                ggml_tensor * bank = d.bindings.at(p.t);
                GGML_ASSERT(p.size == bank->nb[2]);
                ggml_backend_tensor_get_async(copy, bank, p.data, t.slot*bank->nb[2], p.size);
            }
            ggml_backend_synchronize(copy);
            download_bytes += layers[t.vil].bytes;
            t.st = DOWNLOADED;
            return;
        }

        GGML_ASSERT(t.st == QUEUED || t.st == RETIRED);
        if (t.st == RETIRED) {
            // the graphs that read the old expert are done before the slot is written
            ggml_backend_event_wait(copy, d.retire_ev);
        }
        if (t.parts.empty()) {
            if (!stream.read_expert(t.il, t.e, t.buf, t.parts)) {
                throw std::runtime_error("read of the model file failed");
            }
            file_bytes += layers[t.il].bytes;
        }
        for (const auto & p : t.parts) {
            ggml_tensor * bank = d.bindings.at(p.t);
            GGML_ASSERT(p.size == bank->nb[2]);
            ggml_backend_tensor_set_async(copy, bank, p.data, t.slot*bank->nb[2], p.size);
        }
        ggml_backend_synchronize(copy);
        if (verify) {
            std::vector<uint8_t> back;
            for (const auto & p : t.parts) {
                ggml_tensor * bank = d.bindings.at(p.t);
                back.resize(p.size);
                ggml_backend_tensor_get(bank, back.data(), t.slot*bank->nb[2], p.size);
                if (memcmp(back.data(), p.data, p.size) != 0) {
                    GGML_ABORT("moe_hot: verify: expert %d of layer %d differs in slot %d of %s", t.e, t.il, t.slot, ggml_backend_name(d.backend));
                }
            }
        }
        stream.release_hold(t.hold);
        upload_bytes += layers[t.il].bytes;
        t.st = UPLOADED;
    }

    //
    // between batches
    //

    // the bank slots on a device of the selected experts
    static void map_op(ggml_tensor * dst, int ith, int nth, void * user_data) {
        GGML_UNUSED(nth);
        if (ith != 0) {
            return;
        }
        const auto & c = *static_cast<const op_ctx *>(user_data);
        auto & d = c.self->devices[c.device];
        const auto & l = c.self->layers[c.il];

        const ggml_tensor * selected = dst->src[0];
        int32_t * out = static_cast<int32_t *>(dst->data);
        for (int64_t i1 = 0; i1 < selected->ne[1]; ++i1) {
            for (int64_t i0 = 0; i0 < selected->ne[0]; ++i0) {
                const int32_t e = *(const int32_t *) ((const char *) selected->data + i0*selected->nb[0] + i1*selected->nb[1]);
                GGML_ASSERT(e >= 0 && e < (int32_t) l.owner.size());
                const bool hit = l.owner[e] == c.device;
                out[i1*selected->ne[0] + i0] = hit ? l.slot[e] : -1;
                d.hits += hit;
            }
        }
    }

    void record(int il, const ggml_tensor * selected) {
        if (il < 0 || il >= (int) layers.size() || layers[il].owner.empty()) {
            return;
        }
        auto & l = layers[il];
        if (++l.generation == 0) {
            std::fill(l.seen.begin(), l.seen.end(), 0);
            l.generation = 1;
        }
        routes += ggml_nelements(selected);
        for (int64_t i1 = 0; i1 < selected->ne[1]; ++i1) {
            for (int64_t i0 = 0; i0 < selected->ne[0]; ++i0) {
                const int32_t e = *(const int32_t *) ((const char *) selected->data + i0*selected->nb[0] + i1*selected->nb[1]);
                GGML_ASSERT(e >= 0 && e < (int32_t) l.owner.size());
                if (l.seen[e] == l.generation) {
                    continue;
                }
                l.seen[e] = l.generation;
                bytes += l.bytes;
                if (l.owner[e] >= 0) {
                    devices[l.owner[e]].hit_bytes += l.bytes;
                }
            }
        }
    }

    void report() {
        const uint64_t nt = tokens - logged_tokens;
        const uint64_t nr = routes - logged_routes;
        const uint64_t nb = bytes  - logged_bytes;
        const uint64_t rh = ram_hits   - logged_ram_hits;
        if (nt == 0) {
            return;
        }
        uint64_t hits = 0;
        uint64_t hit_bytes = 0;
        std::string per_device;
        std::string busy;
        for (auto & d : devices) {
            const uint64_t h  = d.hits      - d.logged_hits;
            const uint64_t hb = d.hit_bytes - d.logged_hit_bytes;
            hits      += h;
            hit_bytes += hb;
            per_device += format(", %s %.1f%%", ggml_backend_name(d.backend), nr ? 100.0*h/nr : 0.0);
            d.logged_hits      = d.hits;
            d.logged_hit_bytes = d.hit_bytes;
            std::lock_guard<std::mutex> lk(d.mtx);
            busy += format("%s%s %.0f ms", busy.empty() ? "" : ", ", ggml_backend_name(d.backend), d.busy_ms);
        }
        // percentages of all the routes of the streamed layers
        LLAMA_LOG_INFO("moe_hot: %" PRIu64 " tokens: GPU %.1f%% of routes (%.1f%% of expert bytes%s), RAM %.1f%%, GPU+RAM %.1f%%; "
                "%" PRIu64 " updates: %" PRIu64 " promoted, %" PRIu64 " blocked, %" PRIu64 " failed, up %.0f MiB, down %.0f MiB, file %.0f MiB, "
                "%zu planned, %.0f MiB in flight; copies %s, ticks %.0f ms\n",
                nt, nr ? 100.0*hits/nr : 0.0, nb ? 100.0*hit_bytes/nb : 0.0, per_device.c_str(),
                nr ? 100.0*rh/nr : 0.0, nr ? 100.0*(hits + rh)/nr : 0.0,
                n_updates, promotions, blocked, failed, upload_bytes/MiB, download_bytes/MiB, file_bytes/MiB,
                pending.size(), inflight/MiB, busy.c_str(), tick_ms);
        logged_tokens     = tokens;
        logged_routes     = routes;
        logged_bytes      = bytes;
        logged_ram_hits   = ram_hits;
    }

    void remove(int device, group & g, int32_t s) {
        const auto [il, e] = g.owner[s];
        auto & l = layers[il];
        l.owner[e] = -1;
        l.slot [e] = -1;
        l.n_owned[device]--;
        g.owner[s] = { -1, -1 };
        n_cached--;
    }

    // true if the layer gets a new GPU branch
    bool insert(int device, group & g, int32_t s, int il, int e) {
        auto & l = layers[il];
        l.owner[e] = device;
        l.slot [e] = s;
        l.n_owned[device]++;
        g.owner[s] = { il, e };
        n_cached++;
        const bool opened = !l.branch[device];
        l.branch[device] = 1;
        return opened;
    }

    // the old expert of the transfer is in its reserved RAM slot, downloaded now if not yet (the slot is still its own)
    void admit(transfer & t, bool download) {
        if (download) {
            auto & d = devices[t.device];
            for (const auto & p : t.vparts) {
                ggml_tensor * bank = d.bindings.at(p.t);
                ggml_backend_tensor_get(bank, p.data, t.slot*bank->nb[2], p.size);
            }
            download_bytes += layers[t.vil].bytes;
        }
        stream.commit_ram(t.vil, t.ve, t.vhold);
    }

    // the old expert has a RAM copy: its slot is retired, the upload waits for the graphs that may still read it
    void retire(transfer * t, std::vector<uint8_t> & retired) {
        remove(t->device, *t->g, t->slot);
        stream.hot_changed(t->vil, t->ve);
        demotions++;
        t->st = RETIRED;
        retired[t->device] = 1;
    }

    // the transfer is over: its slot and its expert can be planned again
    void finish(transfer * t) {
        stream.release_hold(t->hold);
        t->g->busy[t->slot] = 0;
        layers[t->il].moving[t->e] = 0;
        inflight -= t->bytes;
        delete t;
    }

    void push(transfer * t, bool front) {
        auto & d = devices[t->device];
        {
            std::lock_guard<std::mutex> lk(d.mtx);
            if (front) {
                d.queue.push_front(t);
            } else {
                d.queue.push_back(t);
            }
        }
        d.cv.notify_one();
    }

    bool tick() {
        const int64_t t_start = ggml_time_us();
        bool changed = false;

        std::vector<transfer *> done;
        for (auto & d : devices) {
            std::lock_guard<std::mutex> lk(d.mtx);
            done.insert(done.end(), d.done.begin(), d.done.end());
            d.done.clear();
        }
        bool published = false;
        std::vector<uint8_t>    retired(devices.size(), 0);
        std::vector<transfer *> uploads;
        for (auto * t : done) {
            switch (t->st) {
                case DOWNLOADED:
                    admit(*t, false);
                    retire(t, retired);
                    uploads.push_back(t);
                    break;
                case UPLOADED: {
                    changed |= insert(t->device, *t->g, t->slot, t->il, t->e);
                    stream.hot_changed(t->il, t->e);
                    promotions += !t->seed;
                    n_seeded   += t->seed;
                    published = true;
                    if (t->seed && n_seeded == n_seed) {
                        LLAMA_LOG_INFO("moe_hot: seeded %zu experts in %.0f ms\n", n_seed, (ggml_time_us() - t_seed_start)/1000.0);
                    }
                    finish(t);
                } break;
                default:
                    // a failed download: the old expert stays on the GPU, and gets its RAM slot
                    if (t->vhold.pool >= 0) {
                        admit(*t, true);
                    }
                    failed++;
                    finish(t);
                    break;
            }
        }

        // start the planned transfers within the window
        while (!pending.empty() && (inflight == 0 || inflight + pending.front()->bytes <= inflight_cap)) {
            transfer * t = pending.front();
            pending.pop_front();
            inflight += t->bytes;
            t->st = QUEUED;
            if (t->vil >= 0) {
                if (stream.cached(t->vil, t->ve)) {
                    retire(t, retired); // nothing to download
                } else if (!stream.reserve_ram(t->vil, t->ve, t->vparts, t->vhold)) {
                    blocked++; // no RAM slot can take it: it stays on the GPU
                    finish(t);
                    continue;
                }
            }
            stream.hold_expert(t->il, t->e, t->parts, t->hold); // otherwise read from the model file
            if (t->st == RETIRED) {
                uploads.push_back(t);
            } else {
                push(t, false);
            }
        }

        for (size_t id = 0; id < devices.size(); ++id) {
            if (retired[id]) {
                ggml_backend_event_record(devices[id].retire_ev, devices[id].backend);
            }
        }
        for (auto * t : uploads) {
            push(t, true);
        }

        if (published) {
            stream.hot_changed();
        }
        tick_ms += (ggml_time_us() - t_start)/1000.0;
        return changed;
    }

    bool drain() {
        for (auto * t : pending) {
            t->g->busy[t->slot] = 0;
            layers[t->il].moving[t->e] = 0;
            delete t;
        }
        pending.clear();
        bool changed = false;
        for (;;) {
            changed |= tick();
            bool idle = true;
            for (auto & d : devices) {
                std::lock_guard<std::mutex> lk(d.mtx);
                idle = idle && d.queue.empty() && d.done.empty();
            }
            if (idle && inflight == 0) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return changed;
    }

    void update() {
        last_update = tokens;
        n_updates++;
        if (!pending.empty()) {
            return; // the previous plan is not started yet
        }

        std::vector<std::vector<double>> usage(layers.size());
        for (size_t il = 0; il < layers.size(); ++il) {
            if (!layers[il].owner.empty()) {
                usage[il] = stream.usage((int) il);
                usage[il].resize(layers[il].owner.size(), 0.0);
            }
        }

        // the candidates: the most used experts not in the tier first
        struct candidate {
            double  heat;
            int32_t il;
            int32_t e;
        };
        std::vector<candidate> candidates;
        for (size_t il = 0; il < layers.size(); ++il) {
            const auto & l = layers[il];
            if (l.key.empty() || std::none_of(l.group_of.begin(), l.group_of.end(), [](const group * g) { return g != nullptr; })) {
                continue;
            }
            for (size_t e = 0; e < l.owner.size(); ++e) {
                if (l.owner[e] < 0 && !l.moving[e] && usage[il][e] >= HOT_MIN_GAIN) {
                    candidates.push_back({ usage[il][e], (int32_t) il, (int32_t) e });
                }
            }
        }
        if (candidates.empty()) {
            return;
        }
        std::sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) { return a.heat > b.heat; });

        // the victims of each group, the empty slots first, then the least used. a slot that takes a candidate is not
        // a victim of a later one: the later candidates are used less than it
        for (auto & d : devices) {
            for (auto & [key, g] : d.groups) {
                g.victims.clear();
                g.next = 0;
                for (int32_t s = 0; s < g.n_slots; ++s) {
                    if (g.busy[s]) {
                        continue;
                    }
                    const auto [il, e] = g.owner[s];
                    g.victims.emplace_back(il < 0 ? -1.0 : usage[il][e], s);
                }
                std::sort(g.victims.begin(), g.victims.end());
            }
        }

        size_t spent = 0;
        for (const auto & c : candidates) {
            auto & l = layers[c.il];
            if (spent + l.bytes > update_bytes) {
                continue;
            }
            int    best      = -1;
            double best_heat = std::min(c.heat/HOT_MARGIN, c.heat - HOT_MIN_GAIN); // the most a victim can have
            auto consider = [&](size_t id) {
                const group * g = l.group_of[id];
                if (g == nullptr || g->next >= g->victims.size()) {
                    return;
                }
                const auto & [heat, s] = g->victims[g->next];
                const size_t cost = g->owner[s].first < 0 ? l.bytes : 2*l.bytes;
                if (heat < best_heat && spent + cost <= update_bytes) {
                    best      = (int) id;
                    best_heat = heat;
                }
            };

            // prefer the layer's home GPU; only spill onto another GPU if the home GPU has no eligible victim,
            // and never open a new remote branch on a smaller-budget GPU (avoids spraying CUDA0 layers onto CUDA1)
            if (l.home_dev >= 0) {
                consider((size_t) l.home_dev);
            }
            if (best < 0) {
                for (size_t id = 0; id < devices.size(); ++id) {
                    if ((int) id == l.home_dev) {
                        continue;
                    }
                    if (l.home_dev >= 0 && !l.branch[id] && l.group_of[l.home_dev] != nullptr &&
                            devices[id].budget < devices[l.home_dev].budget) {
                        continue;
                    }
                    consider(id);
                }
            }
            if (best < 0) {
                continue;
            }
            auto & g = *l.group_of[best];
            const int32_t s = g.victims[g.next++].second;

            auto * t = new transfer;
            t->device = best;
            t->g      = &g;
            t->slot   = s;
            t->il     = c.il;
            t->e      = c.e;
            t->bytes  = l.bytes;
            if (g.owner[s].first >= 0) {
                t->vil    = g.owner[s].first;
                t->ve     = g.owner[s].second;
                t->bytes += layers[t->vil].bytes;
            }
            g.busy[s]     = 1;
            l.moving[c.e] = 1;
            pending.push_back(t);
            spent += t->bytes;
        }
    }
};

llama_moe_hot::llama_moe_hot(const llama_model & model, const std::vector<ggml_backend_t> & backends,
        llama_moe_stream & stream, const llama_moe_hot_params & params) :
    pimpl(std::make_unique<impl>(model, backends, stream, params)) {}

llama_moe_hot::~llama_moe_hot() {
    pimpl->report();
}

bool llama_moe_hot::empty() const {
    return pimpl->n_seed == 0 && pimpl->n_cached == 0;
}

int llama_moe_hot::n_devices() const {
    return (int) pimpl->devices.size();
}

ggml_backend_t llama_moe_hot::backend(int device) const {
    return pimpl->devices.at(device).backend;
}

ggml_backend_t llama_moe_hot::home_backend(int il) const {
    if (il < 0 || il >= (int) pimpl->layers.size()) {
        return nullptr;
    }
    const int id = pimpl->layers[il].home_dev;
    return id >= 0 && id < (int) pimpl->devices.size() ? pimpl->devices[id].backend : nullptr;
}

bool llama_moe_hot::has_layer(int il, int device) const {
    if (il < 0 || il >= (int) pimpl->layers.size()) {
        return false;
    }
    const auto & branch = pimpl->layers[il].branch;
    if (branch.empty()) {
        return false;
    }
    if (device >= 0) {
        return branch[device] != 0;
    }
    return std::any_of(branch.begin(), branch.end(), [](uint8_t b) { return b != 0; });
}

bool llama_moe_hot::contains(int il, int expert) const {
    if (il < 0 || il >= (int) pimpl->layers.size()) {
        return false;
    }
    const auto & owner = pimpl->layers[il].owner;
    return expert >= 0 && expert < (int) owner.size() && owner[expert] >= 0;
}

ggml_tensor * llama_moe_hot::bank(const ggml_tensor * w, int device) const {
    const auto & bindings = pimpl->devices.at(device).bindings;
    const auto it = bindings.find(w);
    return it == bindings.end() ? nullptr : it->second;
}

ggml_tensor * llama_moe_hot::build_ids(ggml_context * ctx, ggml_tensor * selected, int il, int device) const {
    GGML_ASSERT(has_layer(il, device));
    ggml_tensor * args[] = { selected };
    return ggml_custom_4d(ctx, GGML_TYPE_I32, selected->ne[0], selected->ne[1], 1, 1, args, 1,
            impl::map_op, 1, &pimpl->ops[il][device]);
}

void llama_moe_hot::begin_batch() {
    auto & p = *pimpl;
    p.batch_routes     = p.routes;
    p.batch_bytes      = p.bytes;
    p.batch_ram_hits   = p.ram_hits;
    for (auto & d : p.devices) {
        d.batch_hits      = d.hits;
        d.batch_hit_bytes = d.hit_bytes;
    }
}

void llama_moe_hot::record_routes(int il, const ggml_tensor * selected) {
    pimpl->record(il, selected);
}

void llama_moe_hot::record_ram_hits(int64_t routes) {
    pimpl->ram_hits += routes;
}

// the statistics of an aborted batch are dropped (its routing still heats the experts)
void llama_moe_hot::abort_batch() {
    auto & p = *pimpl;
    p.routes     = p.batch_routes;
    p.bytes      = p.batch_bytes;
    p.ram_hits   = p.batch_ram_hits;
    for (auto & d : p.devices) {
        d.hits      = d.batch_hits;
        d.hit_bytes = d.batch_hit_bytes;
    }
}

void llama_moe_hot::end_batch(int64_t tokens, int64_t interval) {
    auto & p = *pimpl;
    p.tokens += tokens;
    if (interval > 0 && p.tokens - p.logged_tokens >= (uint64_t) interval) {
        p.report();
    }
}

bool llama_moe_hot::tick(llama_moe_stream & stream) {
    GGML_ASSERT(&stream == &pimpl->stream);
    return pimpl->tick();
}

bool llama_moe_hot::drain(llama_moe_stream & stream) {
    GGML_ASSERT(&stream == &pimpl->stream);
    return pimpl->drain();
}

bool llama_moe_hot::update_due() const {
    return pimpl->update_interval > 0 && pimpl->update_bytes > 0 &&
        pimpl->tokens - pimpl->last_update >= pimpl->update_interval;
}

void llama_moe_hot::update(llama_moe_stream & stream) {
    GGML_ASSERT(&stream == &pimpl->stream);
    pimpl->update();
}
