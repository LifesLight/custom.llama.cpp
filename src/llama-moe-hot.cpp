#include "llama-moe-hot.h"
#include "llama-moe-stream.h"
#include "llama-model.h"
#include "llama-impl.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>

// the usage of an expert is the one of the stream: its decayed selections, the order of its lfu policy. a candidate
// replaces a victim if it is used more by this factor and by this many selections: routing noise does not churn
static constexpr double HOT_MARGIN   = 1.25;
static constexpr double HOT_MIN_GAIN = 2.0;

static constexpr double MiB = 1024.0*1024.0;

struct llama_moe_hot::impl {
    // the slots of one weight layout on one device: a bank per weight tensor of the layout, [ne0, ne1, n_slots + 2],
    // the slots, then a zero expert that the routes of the other experts read, then a padding expert
    struct group {
        std::vector<const ggml_tensor *> refs; // the tensors of the first layer with the layout, per bank
        std::vector<ggml_tensor *>       banks;
        int32_t n_slots = 0;

        std::vector<std::pair<int32_t, int32_t>> owner; // [n_slots] (il, expert), il = -1 if empty

        // update scratch: the slots in victim order (empty, then the coldest) and the next one to take
        std::vector<std::pair<double, int32_t>> victims;
        size_t next = 0;
    };

    struct device {
        ggml_backend_t backend = nullptr;
        size_t budget  = 0;
        size_t used    = 0; // planned, then allocated bytes
        size_t payload = 0; // bytes of the expert slots

        std::map<std::string, group> groups; // by layout, node addresses are stable

        ggml_context_ptr        ctx;
        ggml_backend_buffer_ptr buffer;

        std::unordered_map<const ggml_tensor *, ggml_tensor *> bindings; // expert tensor of the model -> bank

        // statistics: cumulative, at the last report, at the start of the batch
        uint64_t hits = 0, hit_bytes = 0;
        uint64_t logged_hits = 0, logged_hit_bytes = 0;
        uint64_t batch_hits = 0, batch_hit_bytes = 0;
    };

    struct layer {
        std::vector<const ggml_tensor *> refs; // the streamed expert tensors, sorted by name
        std::string key;                       // their layout, empty if the layer cannot be in the tier
        size_t      bytes = 0;                 // of an expert, all its tensors

        std::vector<int32_t> owner;    // [n_expert] device, -1 if not in the tier
        std::vector<int32_t> slot;     // [n_expert] slot in the group of its device
        std::vector<int32_t> n_owned;  // [n_devices]
        std::vector<group *> group_of; // [n_devices] the group of its layout, nullptr if none

        std::vector<uint32_t> seen; // [n_expert] the distinct experts of a batch, for the byte statistics
        uint32_t generation = 0;
    };

    struct op_ctx {
        impl * self;
        int    il;
        int    device;
    };

    std::vector<device>              devices;
    std::vector<layer>               layers; // [il]
    std::vector<std::vector<op_ctx>> ops;    // [il][device]

    size_t n_cached = 0;

    uint64_t update_interval = 0;
    size_t   update_bytes    = 0;
    uint64_t last_update     = 0;

    // statistics: cumulative, at the last report, at the start of the batch
    uint64_t tokens = 0, routes = 0, bytes = 0, ram_hits = 0;
    uint64_t logged_tokens = 0, logged_routes = 0, logged_bytes = 0, logged_ram_hits = 0;
    uint64_t batch_routes = 0, batch_bytes = 0, batch_ram_hits = 0;

    uint64_t n_updates = 0, promotions = 0, demotions = 0, blocked = 0; // demotions: GPU experts admitted to RAM
    size_t   upload_bytes = 0, download_bytes = 0, refill_bytes = 0;
    double   update_ms = 0.0, synchronize_ms = 0.0;

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
            t.ne[2] = n_slots + 2;
            t.nb[3] = t.nb[2]*t.ne[2];
            res += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, &t), alignment);
        }
        return res;
    }

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, llama_moe_stream & stream,
            const llama_moe_hot_params & params) :
        layers(model.layers.size()), update_interval(params.update_interval), update_bytes(params.update_bytes) {
        const int64_t t_start = ggml_time_us();

        for (ggml_backend_t backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            const auto type = ggml_backend_dev_type(dev);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                continue;
            }
            size_t free = 0;
            size_t total = 0;
            ggml_backend_dev_memory(dev, &free, &total);

            device d;
            d.backend = backend;
            d.budget  = free > params.headroom ? free - params.headroom : 0;
            LLAMA_LOG_INFO("moe_hot: %s: %.0f MiB free, %.0f MiB headroom, budget %.0f MiB\n",
                    ggml_backend_name(backend), free/MiB, params.headroom/MiB, d.budget/MiB);
            devices.push_back(std::move(d));
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
            l.key = uniform && n_expert > 0 ? layout(l.refs) : std::string();
            l.owner   .assign(n_expert, -1);
            l.slot    .assign(n_expert, -1);
            l.seen    .assign(n_expert, 0);
            l.n_owned .assign(n_dev, 0);
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
            bool    cached;
            int     device;
            int32_t slot;
        };
        std::vector<placement> plan;
        for (const auto & sd : seeds) {
            auto & l = layers[sd.il];

            int    best      = -1;
            size_t best_cost = 0;
            size_t best_left = 0;
            for (size_t id = 0; id < n_dev; ++id) {
                const auto & d = devices[id];
                const auto it = d.groups.find(l.key);
                const int32_t n = it == d.groups.end() ? 0 : it->second.n_slots;
                const size_t cost = group_size(d, l.refs, n + 1) - group_size(d, l.refs, n);
                if (d.used + cost <= d.budget && (best < 0 || d.budget - d.used - cost > best_left)) {
                    best      = (int) id;
                    best_cost = cost;
                    best_left = d.budget - d.used - cost;
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
            l.owner[sd.e] = best;
            l.slot [sd.e] = g.n_slots++;
            plan.push_back({ sd.il, sd.e, sd.cached, best, l.slot[sd.e] });
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
                for (const ggml_tensor * ref : g.refs) {
                    ggml_tensor * t = ggml_new_tensor_3d(d.ctx.get(), ref->type, ref->ne[0], ref->ne[1], g.n_slots + 2);
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

        size_t from_ssd = 0;
        for (const auto & p : plan) {
            auto & d = devices[p.device];
            auto & l = layers[p.il];
            auto & g = *l.group_of[p.device];
            const bool ok = stream.with_expert(p.il, p.e, [&](const ggml_tensor * w, const void * src, size_t size) {
                ggml_tensor * bank = d.bindings.at(w);
                GGML_ASSERT(size == bank->nb[2]);
                ggml_backend_tensor_set(bank, src, p.slot*bank->nb[2], size);
            });
            if (!ok) {
                throw std::runtime_error("moe_hot: seed expert unavailable");
            }
            from_ssd += p.cached ? 0 : l.bytes;
            g.owner[p.slot] = { p.il, p.e };
            l.n_owned[p.device]++;
            d.payload += l.bytes;
            n_cached++;
        }

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
            LLAMA_LOG_INFO("moe_hot: %s: %d experts in %zu layouts, %.0f MiB (%.0f MiB zero/padding)\n",
                    ggml_backend_name(d.backend), n_slots, d.groups.size(), d.used/MiB, (d.used - d.payload)/MiB);
        }
        LLAMA_LOG_INFO("moe_hot: %zu experts filled in %.0f ms (%.0f MiB read from the model file)\n",
                n_cached, (ggml_time_us() - t_start)/1000.0, from_ssd/MiB);
    }

    // the bank slots on a device of the selected experts
    static void map_op(ggml_tensor * dst, int ith, int nth, void * user_data) {
        GGML_UNUSED(nth);
        if (ith != 0) {
            return;
        }
        const auto & c = *static_cast<const op_ctx *>(user_data);
        auto & d = c.self->devices[c.device];
        const auto & l = c.self->layers[c.il];
        const group * g = l.group_of[c.device];
        const int32_t zero = g ? g->n_slots : 0;

        const ggml_tensor * selected = dst->src[0];
        int32_t * out = static_cast<int32_t *>(dst->data);
        for (int64_t i1 = 0; i1 < selected->ne[1]; ++i1) {
            for (int64_t i0 = 0; i0 < selected->ne[0]; ++i0) {
                const int32_t e = *(const int32_t *) ((const char *) selected->data + i0*selected->nb[0] + i1*selected->nb[1]);
                GGML_ASSERT(e >= 0 && e < (int32_t) l.owner.size());
                const bool hit = l.owner[e] == c.device;
                out[i1*selected->ne[0] + i0] = hit ? l.slot[e] : zero;
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
        for (auto & d : devices) {
            const uint64_t h  = d.hits      - d.logged_hits;
            const uint64_t hb = d.hit_bytes - d.logged_hit_bytes;
            hits      += h;
            hit_bytes += hb;
            per_device += format(", %s %.1f%%", ggml_backend_name(d.backend), nr ? 100.0*h/nr : 0.0);
            d.logged_hits      = d.hits;
            d.logged_hit_bytes = d.hit_bytes;
        }
        // percentages of all the routes of the streamed layers
        LLAMA_LOG_INFO("moe_hot: %" PRIu64 " tokens: GPU %.1f%% of routes (%.1f%% of expert bytes%s), RAM %.1f%%, GPU+RAM %.1f%%; "
                "%" PRIu64 " updates: %" PRIu64 " promoted, %" PRIu64 " blocked, up %.0f MiB, down %.0f MiB, refilled %.0f MiB, %.0f ms + sync %.0f ms\n",
                nt, nr ? 100.0*hits/nr : 0.0, nb ? 100.0*hit_bytes/nb : 0.0, per_device.c_str(),
                nr ? 100.0*rh/nr : 0.0, nr ? 100.0*(hits + rh)/nr : 0.0,
                n_updates, promotions, blocked, upload_bytes/MiB, download_bytes/MiB, refill_bytes/MiB, update_ms, synchronize_ms);
        logged_tokens     = tokens;
        logged_routes     = routes;
        logged_bytes      = bytes;
        logged_ram_hits   = ram_hits;
    }

    // the occupancy of each (layer, device), to detect a change of the graph topology
    std::vector<uint8_t> topology() const {
        std::vector<uint8_t> res;
        res.reserve(layers.size()*devices.size());
        for (const auto & l : layers) {
            for (size_t id = 0; id < devices.size(); ++id) {
                res.push_back(!l.n_owned.empty() && l.n_owned[id] > 0);
            }
        }
        return res;
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

    void insert(int device, group & g, int32_t s, int il, int e) {
        auto & l = layers[il];
        l.owner[e] = device;
        l.slot [e] = s;
        l.n_owned[device]++;
        g.owner[s] = { il, e };
        n_cached++;
    }

    // a reader that downloads the expert of a slot, for its RAM admission
    llama_moe_stream::expert_reader download(device & d, int32_t s) {
        return [this, &d, s](const ggml_tensor * w, void * dst, size_t size) {
            ggml_tensor * bank = d.bindings.at(w);
            GGML_ASSERT(size == bank->nb[2]);
            ggml_backend_tensor_get(bank, dst, s*bank->nb[2], size);
            download_bytes += size;
        };
    }

    bool update(llama_moe_stream & stream, const llama_moe_hot & hot, double sync_ms) {
        last_update = tokens;
        n_updates++;
        synchronize_ms += sync_ms;

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
                if (l.owner[e] < 0 && usage[il][e] >= HOT_MIN_GAIN) {
                    candidates.push_back({ usage[il][e], (int32_t) il, (int32_t) e });
                }
            }
        }
        if (candidates.empty()) {
            return false;
        }

        const int64_t t_start = ggml_time_us();
        stream.quiesce_hot();

        std::sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) { return a.heat > b.heat; });

        // the victims of each group, the empty slots first, then the least used. a slot that takes a candidate is not
        // a victim of a later one: the later candidates are used less than it
        for (auto & d : devices) {
            for (auto & [key, g] : d.groups) {
                g.victims.clear();
                g.next = 0;
                for (int32_t s = 0; s < g.n_slots; ++s) {
                    const auto [il, e] = g.owner[s];
                    g.victims.emplace_back(il < 0 ? -1.0 : usage[il][e], s);
                }
                std::sort(g.victims.begin(), g.victims.end());
            }
        }

        const std::vector<uint8_t> before = topology();
        const uint64_t promotions0 = promotions;
        const size_t up0 = upload_bytes;
        const size_t down0 = download_bytes;

        size_t spent = 0;
        for (const auto & c : candidates) {
            auto & l = layers[c.il];
            if (spent + l.bytes > update_bytes) {
                continue;
            }
            int    best      = -1;
            double best_heat = std::min(c.heat/HOT_MARGIN, c.heat - HOT_MIN_GAIN); // the most a victim can have
            for (size_t id = 0; id < devices.size(); ++id) {
                const group * g = l.group_of[id];
                if (g == nullptr || g->next >= g->victims.size()) {
                    continue;
                }
                const auto & [heat, s] = g->victims[g->next];
                const size_t cost = g->owner[s].first < 0 ? l.bytes : 2*l.bytes;
                if (heat < best_heat && spent + cost <= update_bytes) {
                    best      = (int) id;
                    best_heat = heat;
                }
            }
            if (best < 0) {
                continue;
            }
            auto & d = devices[best];
            auto & g = *l.group_of[best];
            const int32_t s = g.victims[g.next++].second;

            if (g.owner[s].first >= 0) {
                // the victim is admitted to RAM before its slot is reused, otherwise it is kept
                const auto [vil, ve] = g.owner[s];
                if (!stream.restore_expert(vil, ve, download(d, s))) {
                    blocked++;
                    continue;
                }
                remove(best, g, s);
                demotions++;
                spent += l.bytes;
            }

            // the whole expert is uploaded before the maps publish it
            const bool ok = stream.with_expert(c.il, c.e, [&](const ggml_tensor * w, const void * src, size_t size) {
                ggml_tensor * bank = d.bindings.at(w);
                GGML_ASSERT(size == bank->nb[2]);
                ggml_backend_tensor_set(bank, src, s*bank->nb[2], size);
                upload_bytes += size;
            });
            if (!ok) {
                throw std::runtime_error("moe_hot: promotion source unavailable");
            }
            insert(best, g, s, c.il, c.e);
            promotions++;
            spent += l.bytes;
        }

        const size_t refilled = promotions > promotions0 ? stream.reclaim_hot_ram(hot) : 0;
        refill_bytes += refilled;

        const bool changed = topology() != before;
        const double elapsed = (ggml_time_us() - t_start)/1000.0;
        update_ms += elapsed;
        if (promotions > promotions0) {
            LLAMA_LOG_DEBUG("moe_hot: update: %" PRIu64 " promoted, up %.0f MiB, down %.0f MiB, refilled %.0f MiB, %.1f ms + sync %.1f ms%s\n",
                    promotions - promotions0, (upload_bytes - up0)/MiB, (download_bytes - down0)/MiB, refilled/MiB,
                    elapsed, sync_ms, changed ? ", topology changed" : "");
        }
        return changed;
    }

    void demote_all(llama_moe_stream & stream) {
        stream.quiesce_hot();
        std::vector<std::vector<double>> usage(layers.size());
        for (size_t il = 0; il < layers.size(); ++il) {
            usage[il] = stream.usage((int) il);
            usage[il].resize(layers[il].owner.size(), 0.0);
        }
        const int64_t t_start = ggml_time_us();
        const size_t down0 = download_bytes;

        // the most used first: a later admission can evict an earlier one
        struct entry {
            double  heat;
            int     device;
            group * g;
            int32_t slot;
        };
        std::vector<entry> owned;
        for (size_t id = 0; id < devices.size(); ++id) {
            for (auto & [key, g] : devices[id].groups) {
                for (int32_t s = 0; s < g.n_slots; ++s) {
                    const auto [il, e] = g.owner[s];
                    if (il >= 0) {
                        owned.push_back({ usage[il][e], (int) id, &g, s });
                    }
                }
            }
        }
        std::sort(owned.begin(), owned.end(), [](const entry & a, const entry & b) { return a.heat > b.heat; });

        uint64_t restored = 0;
        for (const auto & o : owned) {
            const auto [il, e] = o.g->owner[o.slot];
            restored += stream.restore_expert(il, e, download(devices[o.device], o.slot));
        }
        demotions += restored;
        const double elapsed = (ggml_time_us() - t_start)/1000.0;
        update_ms += elapsed;
        LLAMA_LOG_INFO("moe_hot: released: %" PRIu64 " of %zu GPU experts admitted to RAM (%.0f MiB) in %.0f ms\n",
                restored, owned.size(), (download_bytes - down0)/MiB, elapsed);
    }
};

llama_moe_hot::llama_moe_hot(const llama_model & model, const std::vector<ggml_backend_t> & backends,
        llama_moe_stream & stream, const llama_moe_hot_params & params) :
    pimpl(std::make_unique<impl>(model, backends, stream, params)) {}

llama_moe_hot::~llama_moe_hot() {
    pimpl->report();
}

bool llama_moe_hot::empty() const {
    return pimpl->n_cached == 0;
}

int llama_moe_hot::n_devices() const {
    return (int) pimpl->devices.size();
}

ggml_backend_t llama_moe_hot::backend(int device) const {
    return pimpl->devices.at(device).backend;
}

bool llama_moe_hot::has_layer(int il, int device) const {
    if (il < 0 || il >= (int) pimpl->layers.size()) {
        return false;
    }
    const auto & n_owned = pimpl->layers[il].n_owned;
    if (n_owned.empty()) {
        return false;
    }
    if (device >= 0) {
        return n_owned[device] > 0;
    }
    return std::any_of(n_owned.begin(), n_owned.end(), [](int32_t n) { return n > 0; });
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

bool llama_moe_hot::update_due() const {
    return pimpl->update_interval > 0 && pimpl->update_bytes > 0 &&
        pimpl->tokens - pimpl->last_update >= pimpl->update_interval;
}

bool llama_moe_hot::update(llama_moe_stream & stream, double synchronize_ms) {
    return pimpl->update(stream, *this, synchronize_ms);
}

void llama_moe_hot::make_exclusive(llama_moe_stream & stream) {
    stream.quiesce_hot();
    const int64_t t_start = ggml_time_us();
    const size_t refilled = stream.reclaim_hot_ram(*this);
    const double elapsed = (ggml_time_us() - t_start)/1000.0;
    pimpl->refill_bytes += refilled;
    pimpl->update_ms    += elapsed;
    LLAMA_LOG_INFO("moe_hot: refilled %.0f MiB of RAM copies of GPU experts with other experts in %.0f ms\n",
            refilled/MiB, elapsed);
}

void llama_moe_hot::demote_all(llama_moe_stream & stream) {
    pimpl->demote_all(stream);
}
