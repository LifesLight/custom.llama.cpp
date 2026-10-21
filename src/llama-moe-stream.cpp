#include "llama-moe-stream.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

// O_DIRECT needs the file offset, the length and the buffer aligned to the logical block size of the device
constexpr size_t STREAM_ALIGN = 4096;

// size of one read request into a staging buffer, large enough to be efficient, small enough to be spread over the IO threads
constexpr size_t STREAM_CHUNK = 4ull*1024*1024;

// size of one read request into the cache: an expert is split into several, so that its reads run in parallel
constexpr size_t BOUNCE_CHUNK = 1ull*1024*1024;

constexpr double GiB = 1024.0*1024.0*1024.0;

// generation statistics are logged every that many tokens
constexpr int DECODE_STATS_TOKENS = 128;

// the cached experts are saved every that many generated tokens
constexpr int STATE_SAVE_TOKENS = 256;

size_t align_down(size_t x, size_t a) { return x / a * a; }
size_t align_up  (size_t x, size_t a) { return (x + a - 1) / a * a; }

double t_now() { return ggml_time_us() * 1e-6; }

const char * policy_name(llama_moe_stream_policy p) {
    switch (p) {
        case LLAMA_MOE_STREAM_POLICY_LRU:  return "lru";
        case LLAMA_MOE_STREAM_POLICY_SLRU: return "slru";
    }
    return "?";
}

} // namespace

struct llama_moe_stream::impl {
    struct file {
        std::string path;
        int    fd_direct   = -1; // O_DIRECT, -1 if not supported by the file system
        int    fd_buffered = -1; // fallback and fadvise
        size_t size        = 0;
    };

    // a run of consecutive streamed experts of a tensor, read into a staging buffer as one aligned range of the file
    struct run {
        size_t rd_beg;
        size_t rd_end;
        size_t slot_offs; // offset of rd_beg in the staging buffer
    };

    struct tensor {
        const ggml_tensor * t;

        int     layer;       // index in layers
        int     file;        // index in files
        size_t  file_offs;   // offset of the data in the file
        size_t  nbytes;
        size_t  expert_size;
        int64_t n_expert;

        // selective reads: offset in the staging buffer of the experts read for the current batch, SIZE_MAX if not read
        std::vector<size_t> sel_exp;
        int                 sel_slot = -1;    // the staging buffer, -1 if not read yet
        bool                sel_full = false; // the whole layout was read instead (exp_slot)

        // the cache: [ne0, ne1, n_slots], an expert per slot
        ggml_tensor * bank      = nullptr;
        uint8_t *     bank_data = nullptr;

        // layout of the experts that are not cached in a staging buffer, rebuilt when the cached experts change
        std::vector<run>    runs;
        std::vector<size_t> exp_slot;   // [n_expert] offset of the expert in the staging buffer, SIZE_MAX if cached
        size_t              slot_bytes = 0;
        uint32_t            layout     = 0; // version of the layout
    };

    struct layer {
        int il;
        int64_t n_expert = 0;
        int64_t n_slots  = 0; // cache slots
        std::vector<int> tensors;

        std::vector<int32_t> slot_of;   // [n_expert] cache slot of the expert, -1 if not cached
        std::vector<int32_t> expert_of; // [n_slots]  expert in the cache slot

        // replacement policy
        std::vector<uint64_t> stamp;    // [n_expert] last use
        std::vector<uint8_t>  prot;     // [n_expert] in the protected segment (slru)
        int64_t n_prot = 0;

        std::vector<uint32_t> mark;     // [n_expert] selected by the current batch
        uint32_t              mark_gen = 0;

        bool dirty = false; // the cached experts changed since the staging layouts were built

        bool streamed() const { return n_slots < n_expert; }
    };

    // a set of reads to wait for
    struct wait_group {
        int  pending = 0;     // guarded by mtx
        bool error   = false; // guarded by mtx
    };

    struct slot {
        uint8_t * data = nullptr;

        int      tensor   = -1;    // the tensor held (or being read) by the staging buffer
        uint32_t layout   = 0;     // the layout of the tensor when it was read
        bool     consumed = false; // the tensor was uploaded since it was read
        wait_group wg;

        // the uploads from this buffer must finish before it can be overwritten
        ggml_backend_event_t event      = nullptr;
        ggml_backend_t       ev_backend = nullptr;
        bool                 ev_valid   = false; // the event was recorded after the last upload
        bool                 uploading  = false;

        int sel_owner = -1; // selective reads: the tensor read into the buffer
    };

    struct job {
        int          file;
        size_t       offs;
        size_t       len;
        uint8_t *    dst;
        wait_group * wg;
        bool         aligned; // offs, len and dst are aligned: read directly, otherwise through a bounce buffer
    };

    // user data of the cache ops
    struct op_ctx {
        impl * self;
        int    li;
    };

    bool is_active = false;

    llama_moe_stream_policy policy = LLAMA_MOE_STREAM_POLICY_LRU;

    std::vector<file>   files;
    std::vector<tensor> tensors;
    std::vector<layer>  layers;
    std::vector<int>    layer_of_il; // [il] index in layers, -1 if not streamed
    std::vector<op_ctx> op_ctxs;     // [layers.size()]
    std::unordered_map<const ggml_tensor *, int> tensor_idx;
    std::unordered_map<const ggml_tensor *, int> bank_idx;
    std::unordered_map<std::string, int>         name_idx;

    ggml_backend_dev_t target_dev     = nullptr;
    ggml_backend_t     target_backend = nullptr;
    int64_t            n_expert_used  = 0;

    // batches of up to that many tokens read only the experts selected by the router (LLAMA_MOE_STREAM_SELECTIVE)
    int64_t selective_max = 128;

    // expert cache
    void *                arena      = nullptr; // anonymous mapping, unless pinned
    size_t                arena_size = 0;
    ggml_backend_buffer_t arena_buf  = nullptr; // pinned host buffer, if requested
    ggml_backend_buffer_t bank_buf   = nullptr;
    ggml_context *        bank_ctx   = nullptr;
    bool                  pinned     = false;

    // staging buffers
    ggml_backend_buffer_t host_buf  = nullptr;
    void *                host_mem  = nullptr; // fallback if the device has no pinned host buffer type
    size_t                slot_size = 0;       // the largest streamed part of a tensor
    std::vector<slot>     slots;

    std::string state_path;

    // IO threads
    std::mutex               mtx;
    std::condition_variable  cv_jobs;
    std::condition_variable  cv_done;
    std::deque<job>          jobs;
    std::vector<std::thread> workers;
    bool                     stop = false;
    std::atomic<bool>        warned_dio_fallback { false };

    // graph state
    std::vector<int> order;     // the tensors streamed by the current graph, in the order of their matmuls
    std::vector<int> order_pos; // [tensors.size()] position of a tensor in order, -1 if not streamed
    bool    expect_more   = false;
    bool    last_streamed = false;
    bool    selective     = false; // the current graph reads only the experts selected by the router
    int     sel_layer     = -1;
    std::vector<bool> sel_used;    // the experts selected in sel_layer
    int64_t n_tokens      = 0;
    uint64_t clock        = 0;

    std::vector<int32_t> needed;
    std::vector<int32_t> missing;

    std::vector<uint8_t> zeros = std::vector<uint8_t>(512, 0); // padding of experts that are not read

    // stats of the current graph
    std::atomic<size_t> bytes_read { 0 }; // updated by the IO threads
    size_t bytes_read_start = 0;
    size_t bytes_up_slot    = 0;
    size_t bytes_up_res     = 0;
    double t_first   = -1.0;
    double t_stalled = 0.0;
    int    n_misses  = 0;
    int    n_wasted  = 0; // prefetched tensors evicted before they were used

    // generation stats, since the last report
    int64_t dec_tokens = 0;
    int64_t dec_access = 0;
    int64_t dec_hits   = 0;
    size_t  dec_bytes  = 0;
    double  dec_t_read = 0.0;

    bool    graph_cached = false; // the current graph was computed from the cache
    int64_t state_tokens = 0;     // generated tokens since the state was saved
    bool    state_dirty  = false;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const llama_moe_stream_params & params) {
#ifndef __linux__
        GGML_UNUSED(model);
        GGML_UNUSED(backends);
        GGML_UNUSED(params);
        LLAMA_LOG_WARN("moe_stream: %s: MoE streaming is only supported on Linux, disabled\n", __func__);
#else
        policy        = params.policy;
        n_expert_used = model.hparams.n_expert_used_max();

        // the experts are offloaded to the first GPU (see ggml_backend_sched_backend_id_from_cur)
        for (auto * backend : backends) {
            auto * dev = ggml_backend_get_device(backend);
            const auto type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                target_dev     = dev;
                target_backend = backend;
                break;
            }
        }
        if (!target_dev) {
            LLAMA_LOG_WARN("moe_stream: %s: no GPU to stream the experts to, MoE streaming disabled\n", __func__);
            return;
        }

        if (!collect(model)) {
            return;
        }

        int n_slots = 3;
        if (const char * env = getenv("LLAMA_MOE_STREAM_SLOTS")) {
            n_slots = std::max(2, atoi(env));
        }
        if (const char * env = getenv("LLAMA_MOE_STREAM_SELECTIVE")) {
            selective_max = std::max(0, atoi(env));
        }
        slots.resize(n_slots);

        const int n_threads = std::max(1, params.n_threads);

        plan(params.resident, params.reserve, n_threads);

        if (params.state != "none") {
            state_path = params.state.empty() ? files[0].path + ".moe-stream" : params.state;
        }

        for (int i = 0; i < n_threads; ++i) {
            workers.emplace_back([this] { worker(); });
        }

        // the experts are only read with O_DIRECT from now on, drop them from the page cache
        for (const auto & ti : tensors) {
            posix_fadvise(files[ti.file].fd_buffered, ti.file_offs, ti.nbytes, POSIX_FADV_DONTNEED);
        }

        if (!alloc_cache() || !fill_cache() || !alloc_slots()) {
            return;
        }

        check_offload();

        is_active = true;
#endif
    }

    ~impl() {
#ifdef __linux__
        if (is_active && state_dirty) {
            save_state();
        }
#endif
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop = true;
        }
        cv_jobs.notify_all();
        for (auto & w : workers) {
            w.join();
        }

#ifdef __linux__
        for (auto & s : slots) {
            if (s.event) {
                ggml_backend_event_free(s.event);
            }
        }
        if (host_buf) {
            ggml_backend_buffer_free(host_buf);
        }
        free(host_mem);
        if (bank_ctx) {
            ggml_free(bank_ctx);
        }
        if (bank_buf && bank_buf != arena_buf) {
            ggml_backend_buffer_free(bank_buf);
        }
        if (arena_buf) {
            ggml_backend_buffer_free(arena_buf);
        }
        if (arena) {
            munmap(arena, arena_size);
        }
        for (auto & f : files) {
            if (f.fd_direct >= 0) {
                close(f.fd_direct);
            }
            if (f.fd_buffered >= 0) {
                close(f.fd_buffered);
            }
        }
#endif
    }

#ifdef __linux__
    int open_file(const std::string & path) {
        for (size_t i = 0; i < files.size(); ++i) {
            if (files[i].path == path) {
                return (int) i;
            }
        }

        file f;
        f.path = path;
        f.fd_buffered = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (f.fd_buffered < 0) {
            LLAMA_LOG_WARN("moe_stream: %s: failed to open %s: %s\n", __func__, path.c_str(), strerror(errno));
            return -1;
        }
        f.fd_direct = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (f.fd_direct < 0) {
            LLAMA_LOG_WARN("moe_stream: %s: O_DIRECT is not supported for %s (%s), using buffered reads\n", __func__, path.c_str(), strerror(errno));
        }
        struct stat st;
        if (fstat(f.fd_buffered, &st) == 0) {
            f.size = st.st_size;
        }

        files.push_back(std::move(f));
        return (int) files.size() - 1;
    }

    // find the memory mapped expert tensors in host memory and group them by layer
    bool collect(const llama_model & model) {
        std::map<int, std::vector<tensor>> by_layer;

        int n_unmapped = 0;

        for (const auto & [name, t] : model.tensors_by_name) {
            if (name.find("_exps.weight") == std::string::npos) {
                continue;
            }
            if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer) ||
                ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                continue; // not in host memory
            }
            if (ggml_n_dims(t) != 3 || ggml_nbytes(t) != (size_t) t->nb[2]*t->ne[2]) {
                continue; // not a [.., .., n_expert] tensor with contiguous experts
            }

            int il = -1;
            if (sscanf(name.c_str(), "blk.%d.", &il) != 1 || il < 0) {
                continue;
            }

            std::string path;
            size_t offs = 0;
            if (!model.get_mapped_location(t, path, offs)) {
                n_unmapped++;
                continue;
            }

            const int fi = open_file(path);
            if (fi < 0) {
                n_unmapped++;
                continue;
            }

            tensor ti;
            ti.t           = t;
            ti.layer       = -1;
            ti.file        = fi;
            ti.file_offs   = offs;
            ti.nbytes      = ggml_nbytes(t);
            ti.expert_size = t->nb[2];
            ti.n_expert    = t->ne[2];

            by_layer[il].push_back(ti);
        }

        if (n_unmapped > 0) {
            LLAMA_LOG_WARN("moe_stream: %s: %d expert tensors in host memory are not memory mapped from the model files, they are not streamed (is mmap disabled?)\n",
                    __func__, n_unmapped);
        }

        for (auto & [il, ts] : by_layer) {
            std::sort(ts.begin(), ts.end(), [](const tensor & a, const tensor & b) {
                return a.file != b.file ? a.file < b.file : a.file_offs < b.file_offs;
            });

            layer l;
            l.il       = il;
            l.n_expert = ts[0].n_expert;
            bool ok = true;
            for (const auto & ti : ts) {
                ok = ok && ti.n_expert == l.n_expert;
            }
            if (!ok) {
                LLAMA_LOG_WARN("moe_stream: %s: layer %d has expert tensors with different expert counts, not streamed\n", __func__, il);
                continue;
            }

            const int li = (int) layers.size();
            for (auto & ti : ts) {
                ti.layer = li;
                l.tensors.push_back((int) tensors.size());
                tensor_idx[ti.t]     = (int) tensors.size();
                name_idx[ti.t->name] = (int) tensors.size();
                tensors.push_back(ti);
            }
            if ((int) layer_of_il.size() <= il) {
                layer_of_il.resize(il + 1, -1);
            }
            layer_of_il[il] = li;
            layers.push_back(std::move(l));
        }

        if (layers.empty()) {
            LLAMA_LOG_WARN("moe_stream: %s: no memory mapped expert tensors in host memory, MoE streaming disabled\n", __func__);
            return false;
        }

        op_ctxs.resize(layers.size());
        for (size_t li = 0; li < layers.size(); ++li) {
            op_ctxs[li] = { this, (int) li };
        }

        return true;
    }

    static size_t mem_available() {
        FILE * f = fopen("/proc/meminfo", "r");
        if (!f) {
            return 0;
        }
        char line[256];
        size_t kb = 0;
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) {
                break;
            }
        }
        fclose(f);
        return kb*1024;
    }

    // the worst case size of the streamed part of a tensor in a staging buffer, whichever experts are cached:
    // every run of streamed experts can need up to 2 alignment blocks more
    static size_t max_slot_bytes(const tensor & ti, int64_t n_cached) {
        const int64_t n_streamed = ti.n_expert - n_cached;
        if (n_streamed <= 0) {
            return 0;
        }
        const int64_t n_runs = std::min(n_streamed, n_cached + 1);
        return n_streamed*ti.expert_size + n_runs*2*STREAM_ALIGN;
    }

    size_t staging_size(double frac) const {
        size_t res = 0;
        for (const auto & l : layers) {
            const int64_t n_cached = (int64_t) (frac * (double) l.n_expert);
            for (int idx : l.tensors) {
                res = std::max(res, max_slot_bytes(tensors[idx], n_cached));
            }
        }
        return slots.size()*align_up(res, STREAM_ALIGN);
    }

    // every layer caches the same fraction of its experts
    // all the experts of a layer have the same size, so the same fraction of experts is the same fraction of bytes
    void plan(size_t resident, size_t reserve, int n_threads) {
        size_t total = 0;
        for (const auto & ti : tensors) {
            total += ti.nbytes;
        }

        double frac;
        if (resident == SIZE_MAX) {
            // what is available, minus the reserve, the staging buffers (which shrink as more experts are cached)
            // and the bounce buffers of the IO threads
            const size_t avail  = mem_available();
            const size_t bounce = n_threads*(BOUNCE_CHUNK + 2*STREAM_ALIGN);
            frac = 0.0;
            for (int i = 0; i < 8; ++i) {
                const double budget = (double) avail - (double) reserve - (double) staging_size(frac) - (double) bounce;
                frac = std::clamp(budget / (double) total, 0.0, 1.0);
            }
            LLAMA_LOG_INFO("moe_stream: %s: %.2f GiB available, %.2f GiB reserved, %.2f GiB for the staging buffers -> %.2f GiB of cached experts\n",
                    __func__, avail/GiB, reserve/GiB, staging_size(frac)/GiB, frac*total/GiB);
        } else {
            frac = std::min(1.0, (double) resident / (double) total);
        }

        size_t total_res = 0;
        for (auto & l : layers) {
            l.n_slots = (int64_t) (frac * (double) l.n_expert);

            l.slot_of  .assign(l.n_expert, -1);
            l.expert_of.assign(l.n_slots,  -1);
            l.stamp    .assign(l.n_expert, 0);
            l.prot     .assign(l.n_expert, 0);
            l.mark     .assign(l.n_expert, 0);

            for (int idx : l.tensors) {
                auto & ti = tensors[idx];
                total_res += (size_t) l.n_slots * ti.expert_size;
                slot_size = std::max(slot_size, align_up(max_slot_bytes(ti, l.n_slots), STREAM_ALIGN));
            }
        }

        LLAMA_LOG_INFO("moe_stream: %s: %zu layers, %.2f GiB of experts, %.2f GiB cached (%" PRId64 "/%" PRId64 " experts per layer), %.2f GiB streamed per ubatch, %s policy\n",
                __func__, layers.size(), total/GiB, total_res/GiB, layers[0].n_slots, layers[0].n_expert, (total - total_res)/GiB, policy_name(policy));
    }

    //
    // expert cache
    //

    bool alloc_cache() {
        size_t total = 0;
        size_t n_banks = 0;
        for (const auto & l : layers) {
            for (int idx : l.tensors) {
                if (l.n_slots > 0) {
                    total += align_up(l.n_slots*tensors[idx].expert_size, STREAM_ALIGN);
                    n_banks++;
                }
            }
        }
        if (total == 0) {
            return true;
        }

        const double t_start = t_now();

        uint8_t * base = nullptr;

        // pinned memory makes the uploads of the cached experts DMA, but pinning that much can take long
        const char * env_pin = getenv("LLAMA_MOE_STREAM_PIN");
        if (env_pin && atoi(env_pin) != 0) {
            if (auto * buft = ggml_backend_dev_host_buffer_type(target_dev)) {
                arena_buf = ggml_backend_buft_alloc_buffer(buft, total + STREAM_ALIGN);
            }
            if (arena_buf) {
                base     = (uint8_t *) align_up((size_t) ggml_backend_buffer_get_base(arena_buf), STREAM_ALIGN);
                bank_buf = arena_buf;
                pinned   = true;
            } else {
                LLAMA_LOG_WARN("moe_stream: %s: failed to allocate %.2f GiB of pinned memory, using normal memory\n", __func__, total/GiB);
            }
        }

        if (!base) {
            arena_size = total;
            arena = mmap(nullptr, arena_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (arena == MAP_FAILED) {
                arena = nullptr;
                LLAMA_LOG_ERROR("moe_stream: %s: failed to allocate %.2f GiB for the expert cache: %s\n", __func__, total/GiB, strerror(errno));
                return false;
            }
            // locking also allocates all the pages now
            if (mlock(arena, arena_size) != 0) {
                LLAMA_LOG_WARN("moe_stream: %s: failed to lock the expert cache (%s), it may be swapped out\n", __func__, strerror(errno));
                LLAMA_LOG_WARN("moe_stream: %s: raise the limit with 'ulimit -l unlimited' (or LimitMEMLOCK=infinity for a systemd service)\n", __func__);
            }
            base = (uint8_t *) arena;
            bank_buf = ggml_backend_cpu_buffer_from_ptr(arena, arena_size);
        }

        ggml_backend_buffer_set_usage(bank_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        ggml_init_params ip = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(n_banks + 1),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        bank_ctx = ggml_init(ip);

        uint8_t * p = base;
        for (const auto & l : layers) {
            if (l.n_slots == 0) {
                continue;
            }
            for (int idx : l.tensors) {
                auto & ti = tensors[idx];
                ti.bank = ggml_new_tensor_3d(bank_ctx, ti.t->type, ti.t->ne[0], ti.t->ne[1], l.n_slots);
                GGML_ASSERT((size_t) ti.bank->nb[2] == ti.expert_size);
                ggml_format_name(ti.bank, "%s.cache", ti.t->name);
                ggml_backend_tensor_alloc(bank_buf, ti.bank, p);
                ti.bank_data = p;
                bank_idx[ti.bank] = idx;
                p += align_up(l.n_slots*ti.expert_size, STREAM_ALIGN);
            }
        }

        LLAMA_LOG_INFO("moe_stream: %s: allocated %.2f GiB of %s memory for the expert cache in %.1f s\n", __func__,
                total/GiB, pinned ? "pinned" : "locked", t_now() - t_start);

        return true;
    }

    // read the experts of the slots [s0, s0 + n) of a layer, which hold consecutive experts from e0
    void read_experts(const layer & l, int64_t e0, int64_t s0, int64_t n, wait_group & wg) {
        for (int idx : l.tensors) {
            const auto & ti = tensors[idx];
            submit_bounce(ti.file, ti.file_offs + e0*ti.expert_size, n*ti.expert_size, ti.bank_data + s0*ti.expert_size, wg);
        }
    }

    // fill the cache with the saved experts, or the first ones
    bool fill_cache() {
        const int n_loaded = load_state();

        const double t_start = t_now();
        const size_t read_start = bytes_read;

        wait_group wg;
        for (auto & l : layers) {
            // the remaining slots get the first experts that are not cached yet
            int64_t e = 0;
            for (int64_t s = 0; s < l.n_slots; ++s) {
                if (l.expert_of[s] >= 0) {
                    continue;
                }
                while (l.slot_of[e] >= 0) {
                    e++;
                }
                l.expert_of[s] = (int32_t) e;
                l.slot_of[e]   = (int32_t) s;
            }

            // read runs of consecutive experts in consecutive slots together
            for (int64_t s = 0; s < l.n_slots; ) {
                int64_t n = 1;
                while (s + n < l.n_slots && l.expert_of[s + n] == l.expert_of[s] + n) {
                    n++;
                }
                read_experts(l, l.expert_of[s], s, n, wg);
                s += n;
            }

            build_layout(l);
        }
        if (!wait(wg)) {
            LLAMA_LOG_ERROR("moe_stream: %s: failed to read the experts into the cache\n", __func__);
            return false;
        }

        const double t    = std::max(t_now() - t_start, 1e-6);
        const size_t read = bytes_read - read_start;
        LLAMA_LOG_INFO("moe_stream: %s: read %.2f GiB of experts into the cache in %.1f s (%.2f GiB/s), %d layers from %s\n", __func__,
                read/GiB, t, read/GiB/t, n_loaded, n_loaded > 0 ? state_path.c_str() : "nowhere (first experts)");

        return true;
    }

    // log how batches are computed
    void check_offload() {
        int64_t cpu_max = INT64_MAX;
        for (const auto & l : layers) {
            if (l.streamed()) {
                cpu_max = std::min<int64_t>(cpu_max, n_expert_used > 0 ? l.n_slots / n_expert_used : 0);
            }
        }
        if (cpu_max == INT64_MAX) {
            return; // everything cached
        }
        if (cpu_max == 0) {
            LLAMA_LOG_WARN("moe_stream: %s: less than %" PRId64 " cached experts per layer, generation is offloaded too\n", __func__, n_expert_used);
        }
        LLAMA_LOG_INFO("moe_stream: %s: batches of up to %" PRId64 " tokens are computed on the CPU from the cache, up to %" PRId64
                " on %s reading the selected experts, larger ones on %s streaming all the experts\n", __func__,
                cpu_max, selective_max, ggml_backend_name(target_backend), ggml_backend_name(target_backend));
    }

    //
    // saved state: the cached experts of each layer, most recently used first
    //

    int load_state() {
        if (state_path.empty()) {
            return 0;
        }
        std::ifstream in(state_path);
        if (!in) {
            return 0;
        }

        std::string line;
        if (!std::getline(in, line) || line != "moe-stream-state 1") {
            LLAMA_LOG_WARN("moe_stream: %s: %s is not a state file, ignored\n", __func__, state_path.c_str());
            return 0;
        }

        int n_loaded = 0;
        while (std::getline(in, line)) {
            std::istringstream ss(line);
            int il = -1;
            int64_t n_expert = 0;
            int64_t count = 0;
            if (!(ss >> il >> n_expert >> count) || il < 0 || il >= (int) layer_of_il.size() || layer_of_il[il] < 0) {
                continue;
            }
            auto & l = layers[layer_of_il[il]];
            if (n_expert != l.n_expert) {
                continue;
            }
            int64_t s = 0;
            for (int64_t i = 0; i < count && s < l.n_slots; ++i) {
                int64_t e = -1;
                if (!(ss >> e) || e < 0 || e >= l.n_expert || l.slot_of[e] >= 0) {
                    continue;
                }
                l.expert_of[s] = (int32_t) e;
                l.slot_of[e]   = (int32_t) s;
                l.stamp[e]     = (uint64_t) (count - i); // most recently used first
                s++;
            }
            n_loaded++;
        }
        clock = 1ull << 20;

        return n_loaded;
    }

    void save_state() {
        if (state_path.empty()) {
            return;
        }
        const std::string tmp = state_path + ".tmp";
        {
            std::ofstream out(tmp);
            if (!out) {
                static bool warned = false;
                if (!warned) {
                    LLAMA_LOG_WARN("moe_stream: %s: cannot write %s, the cached experts are not saved\n", __func__, tmp.c_str());
                    warned = true;
                }
                return;
            }
            out << "moe-stream-state 1\n";
            std::vector<int32_t> es;
            for (const auto & l : layers) {
                es.clear();
                for (int64_t s = 0; s < l.n_slots; ++s) {
                    es.push_back(l.expert_of[s]);
                }
                std::sort(es.begin(), es.end(), [&](int32_t a, int32_t b) { return l.stamp[a] > l.stamp[b]; });
                out << l.il << " " << l.n_expert << " " << es.size();
                for (int32_t e : es) {
                    out << " " << e;
                }
                out << "\n";
            }
        }
        if (rename(tmp.c_str(), state_path.c_str()) != 0) {
            LLAMA_LOG_WARN("moe_stream: %s: failed to save %s: %s\n", __func__, state_path.c_str(), strerror(errno));
            return;
        }
        state_dirty  = false;
        state_tokens = 0;
    }

    //
    // replacement policy, for the batches computed on the CPU
    //

    // the least recently used cached expert that is not selected by the batch, in the given segment (-1 = any)
    int64_t victim(const layer & l, int seg) const {
        int64_t best = -1;
        for (int64_t s = 0; s < l.n_slots; ++s) {
            const int64_t e = l.expert_of[s];
            if (l.mark[e] == l.mark_gen || (seg >= 0 && l.prot[e] != seg)) {
                continue;
            }
            if (best < 0 || l.stamp[e] < l.stamp[best]) {
                best = e;
            }
        }
        return best;
    }

    // make the experts selected by a batch cached and write their slots
    void ensure(int li, const ggml_tensor * ids, ggml_tensor * out) {
        auto & l = layers[li];

        const int64_t k = ids->ne[0];
        const int64_t n = ids->ne[1];

        graph_cached = true;

        auto id_at = [&](int64_t i0, int64_t i1) {
            return *(const int32_t *) ((const char *) ids->data + i0*ids->nb[0] + i1*ids->nb[1]);
        };

        if (++l.mark_gen == 0) {
            std::fill(l.mark.begin(), l.mark.end(), 0);
            l.mark_gen = 1;
        }

        clock++;

        needed.clear();
        for (int64_t i1 = 0; i1 < n; ++i1) {
            for (int64_t i0 = 0; i0 < k; ++i0) {
                const int32_t e = id_at(i0, i1);
                GGML_ASSERT(e >= 0 && e < l.n_expert);
                if (l.mark[e] != l.mark_gen) {
                    l.mark[e] = l.mark_gen;
                    needed.push_back(e);
                }
            }
        }
        GGML_ASSERT((int64_t) needed.size() <= l.n_slots && "more experts selected than cache slots");

        missing.clear();
        for (int32_t e : needed) {
            l.stamp[e] = clock;
            dec_access++;
            if (l.slot_of[e] < 0) {
                missing.push_back(e);
                continue;
            }
            dec_hits++;
            if (policy == LLAMA_MOE_STREAM_POLICY_SLRU && !l.prot[e]) {
                // a second use promotes to the protected half, its least recently used expert goes back to probation
                l.prot[e] = 1;
                l.n_prot++;
                if (l.n_prot > l.n_slots/2) {
                    int64_t d = -1;
                    for (int64_t s = 0; s < l.n_slots; ++s) {
                        const int64_t x = l.expert_of[s];
                        if (l.prot[x] && x != e && (d < 0 || l.stamp[x] < l.stamp[d])) {
                            d = x;
                        }
                    }
                    if (d >= 0) {
                        l.prot[d] = 0;
                        l.n_prot--;
                    }
                }
            }
        }

        if (!missing.empty()) {
            // evict all the victims at once, then read all the missing experts in parallel
            const double t0 = t_now();
            const size_t read_start = bytes_read;

            wait_group wg;
            for (int32_t e : missing) {
                int64_t v = policy == LLAMA_MOE_STREAM_POLICY_SLRU ? victim(l, 0) : -1;
                if (v < 0) {
                    v = victim(l, -1);
                }
                GGML_ASSERT(v >= 0);

                const int32_t s = l.slot_of[v];
                if (l.prot[v]) {
                    l.prot[v] = 0;
                    l.n_prot--;
                }
                l.slot_of[v]   = -1;
                l.slot_of[e]   = s;
                l.expert_of[s] = e;

                read_experts(l, e, s, 1, wg);
            }
            if (!wait(wg)) {
                GGML_ABORT("moe_stream: failed to read the experts of layer %d", l.il);
            }

            l.dirty     = true;
            state_dirty = true;
            dec_bytes  += bytes_read - read_start;
            dec_t_read += t_now() - t0;
        }

        int32_t * dst = (int32_t *) out->data;
        for (int64_t i1 = 0; i1 < n; ++i1) {
            for (int64_t i0 = 0; i0 < k; ++i0) {
                dst[i1*k + i0] = l.slot_of[id_at(i0, i1)];
            }
        }
    }

    static void ensure_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
        GGML_UNUSED(nth);
        if (ith != 0) {
            return;
        }
        auto * c = (op_ctx *) userdata;
        c->self->ensure(c->li, dst->src[0], dst);
    }

    ggml_tensor * build_slots(ggml_context * ctx, ggml_tensor * selected, int il) const {
        if (il < 0 || il >= (int) layer_of_il.size() || layer_of_il[il] < 0) {
            return nullptr;
        }
        const int li = layer_of_il[il];
        const auto & l = layers[li];

        const int64_t k = selected->ne[0];
        const int64_t n = selected->ne[1];
        if (l.n_slots == 0 || (l.streamed() && l.n_slots < k*n)) {
            return nullptr; // the batch could select more experts than there are slots
        }

        ggml_tensor * args[1] = { selected };
        return ggml_custom_4d(ctx, GGML_TYPE_I32, k, n, 1, 1, args, 1, ensure_op, 1, const_cast<op_ctx *>(&op_ctxs[li]));
    }

    //
    // IO threads
    //

    void submit(const job & j) {
        jobs.push_back(j);
        j.wg->pending++;
    }

    // read an aligned range of a file into an aligned buffer
    void submit_aligned(int fi, size_t beg, size_t end, uint8_t * dst, wait_group & wg) {
        std::lock_guard<std::mutex> lk(mtx);
        for (size_t o = beg; o < end; o += STREAM_CHUNK) {
            submit({ fi, o, std::min(STREAM_CHUNK, end - o), dst + (o - beg), &wg, true });
        }
        cv_jobs.notify_all();
    }

    // read any range of a file into any buffer, through the bounce buffers of the IO threads
    void submit_bounce(int fi, size_t offs, size_t len, uint8_t * dst, wait_group & wg) {
        std::lock_guard<std::mutex> lk(mtx);
        for (size_t o = 0; o < len; o += BOUNCE_CHUNK) {
            submit({ fi, offs + o, std::min(BOUNCE_CHUNK, len - o), dst + o, &wg, false });
        }
        cv_jobs.notify_all();
    }

    bool wait(wait_group & wg) {
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [&] { return wg.pending == 0; });
        const bool ok = !wg.error;
        wg.error = false;
        return ok;
    }

    // read len bytes at offs, or up to the end of the file, returns the bytes read or -1
    static ssize_t pread_full(int fd, uint8_t * dst, size_t len, size_t offs) {
        size_t done = 0;
        while (done < len) {
            const ssize_t r = pread(fd, dst + done, len - done, offs + done);
            if (r < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return -1;
            }
            if (r == 0) {
                break;
            }
            done += r;
        }
        return done;
    }

    bool read_job(const job & j, uint8_t * bounce) {
        const file & f = files[j.file];

        // the bytes of the job that are in the file: an aligned range can end past the end of the file
        const size_t need = j.offs < f.size ? std::min(j.len, f.size - j.offs) : 0;

        if (f.fd_direct >= 0) {
            ssize_t r;
            if (j.aligned) {
                r = pread_full(f.fd_direct, j.dst, j.len, j.offs);
                if (r >= 0 && (size_t) r >= need) {
                    bytes_read += need;
                    return true;
                }
            } else {
                const size_t a0 = align_down(j.offs, STREAM_ALIGN);
                const size_t a1 = align_up(j.offs + j.len, STREAM_ALIGN);
                r = pread_full(f.fd_direct, bounce, a1 - a0, a0);
                if (r >= 0 && a0 + (size_t) r >= j.offs + need) {
                    memcpy(j.dst, bounce + (j.offs - a0), need);
                    bytes_read += need;
                    return true;
                }
            }
            if (r < 0 && errno != EINVAL) {
                LLAMA_LOG_ERROR("moe_stream: %s: read of %zu bytes at %zu of %s failed: %s\n", __func__, j.len, j.offs, f.path.c_str(), strerror(errno));
                return false;
            }
            if (!warned_dio_fallback.exchange(true)) {
                LLAMA_LOG_WARN("moe_stream: %s: O_DIRECT read failed for %s, falling back to buffered reads\n", __func__, f.path.c_str());
            }
        }

        const ssize_t r = pread_full(f.fd_buffered, j.dst, need, j.offs);
        if (r < 0 || (size_t) r != need) {
            LLAMA_LOG_ERROR("moe_stream: %s: read of %zu bytes at %zu of %s failed: %s\n", __func__, need, j.offs, f.path.c_str(),
                    r < 0 ? strerror(errno) : "unexpected end of file");
            return false;
        }
        bytes_read += need;
        return true;
    }

    void worker() {
        void * bounce = nullptr;
        if (posix_memalign(&bounce, STREAM_ALIGN, BOUNCE_CHUNK + 2*STREAM_ALIGN) != 0) {
            bounce = nullptr;
        }

        for (;;) {
            job j;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_jobs.wait(lk, [this] { return stop || !jobs.empty(); });
                if (jobs.empty()) {
                    break; // stop
                }
                j = jobs.front();
                jobs.pop_front();
            }

            const bool ok = bounce && read_job(j, (uint8_t *) bounce);

            {
                std::lock_guard<std::mutex> lk(mtx);
                j.wg->error = j.wg->error || !ok;
                if (--j.wg->pending == 0) {
                    cv_done.notify_all();
                }
            }
        }

        free(bounce);
    }

    //
    // staging buffers
    //

    bool alloc_slots() {
        if (slot_size == 0) {
            LLAMA_LOG_INFO("moe_stream: %s: all the experts are cached, nothing to stream\n", __func__);
            return true;
        }

        const size_t n_slots = slots.size();
        const size_t size    = n_slots*slot_size + STREAM_ALIGN;

        uint8_t * base = nullptr;

        auto * buft = ggml_backend_dev_host_buffer_type(target_dev);
        if (buft) {
            host_buf = ggml_backend_buft_alloc_buffer(buft, size);
            if (host_buf) {
                base = (uint8_t *) ggml_backend_buffer_get_base(host_buf);
            }
        }
        if (!base) {
            LLAMA_LOG_WARN("moe_stream: %s: no pinned host buffer for %s, uploads from the staging buffers will be slower\n", __func__, ggml_backend_dev_name(target_dev));
            if (posix_memalign(&host_mem, STREAM_ALIGN, size) != 0) {
                host_mem = nullptr;
            }
            base = (uint8_t *) host_mem;
        }
        if (!base) {
            LLAMA_LOG_ERROR("moe_stream: %s: failed to allocate %.2f GiB for the staging buffers, MoE streaming disabled\n", __func__, size/GiB);
            return false;
        }

        base = (uint8_t *) align_up((size_t) base, STREAM_ALIGN);
        for (size_t s = 0; s < n_slots; ++s) {
            slots[s].data  = base + s*slot_size;
            slots[s].event = ggml_backend_event_new(target_dev);
        }

        LLAMA_LOG_INFO("moe_stream: %s: %zu x %.2f GiB %s staging buffers (%.2f GiB), %zu IO threads\n", __func__,
                n_slots, slot_size/GiB, host_buf ? ggml_backend_buffer_name(host_buf) : "CPU", n_slots*slot_size/GiB, workers.size());

        return true;
    }

    // the runs of streamed experts of the tensors of a layer and their place in a staging buffer
    void build_layout(layer & l) {
        for (int idx : l.tensors) {
            auto & ti = tensors[idx];
            const size_t es = ti.expert_size;

            ti.runs.clear();
            ti.exp_slot.assign(ti.n_expert, SIZE_MAX);
            ti.slot_bytes = 0;
            ti.layout++;

            for (int64_t first = 0; first < ti.n_expert; ) {
                if (l.slot_of[first] >= 0) {
                    first++;
                    continue;
                }
                int64_t last = first;
                while (last + 1 < ti.n_expert && l.slot_of[last + 1] < 0) {
                    last++;
                }

                run r;
                r.rd_beg    = align_down(ti.file_offs + first*es,      STREAM_ALIGN);
                r.rd_end    = align_up  (ti.file_offs + (last + 1)*es, STREAM_ALIGN);
                r.slot_offs = ti.slot_bytes;
                for (int64_t e = first; e <= last; ++e) {
                    ti.exp_slot[e] = r.slot_offs + (ti.file_offs + e*es - r.rd_beg);
                }
                ti.slot_bytes += r.rd_end - r.rd_beg;
                ti.runs.push_back(r);

                first = last + 1;
            }

            GGML_ASSERT(ti.slot_bytes <= slot_size);
        }
    }

    // wait until a staging buffer can be overwritten
    void release(int s) {
        auto & sl = slots[s];
        if (sl.uploading) {
            if (sl.ev_valid) {
                ggml_backend_event_synchronize(sl.event);
            } else {
                ggml_backend_synchronize(sl.ev_backend);
            }
            sl.uploading = false;
        }
        wait(sl.wg);
    }

    void load(int s, int t) {
        release(s);

        auto & sl = slots[s];
        if (sl.tensor >= 0 && !sl.consumed) {
            n_wasted++;
        }

        const auto & ti = tensors[t];

        sl.tensor   = t;
        sl.layout   = ti.layout;
        sl.consumed = false;
        for (const auto & r : ti.runs) {
            submit_aligned(ti.file, r.rd_beg, r.rd_end, sl.data + r.slot_offs, sl.wg);
        }
    }

    int find_slot(int t) const {
        for (size_t s = 0; s < slots.size(); ++s) {
            if (slots[s].tensor == t && slots[s].layout == tensors[t].layout) {
                return (int) s;
            }
        }
        return -1;
    }

    // the staging buffer to read a new tensor into: not one of the wanted tensors, preferably empty or already uploaded
    int pick_slot(const std::vector<int> & wanted) const {
        int best = -1;
        int best_score = -1;
        for (size_t s = 0; s < slots.size(); ++s) {
            const auto & sl = slots[s];
            if (std::find(wanted.begin(), wanted.end(), sl.tensor) != wanted.end()) {
                continue;
            }
            const int score = sl.tensor < 0 ? 2 : sl.consumed ? 1 : 0;
            if (score > best_score) {
                best       = (int) s;
                best_score = score;
            }
        }
        return best;
    }

    // make sure tensor t is in a staging buffer, and read the next tensors of the graph into the other ones
    // the buffers of the tensors already uploaded are reused once their uploads are done, which in the steady
    // state is the previous tensor, whose upload is queued before the upload of t anyway
    int prepare(int t) {
        // the tensors wanted in the buffers: t and the next ones, wrapping to the next ubatch if one is expected
        std::vector<int> wanted = { t };
        int p = t < (int) order_pos.size() ? order_pos[t] : -1;
        while (p >= 0 && wanted.size() < slots.size()) {
            p = p + 1 < (int) order.size() ? p + 1 : expect_more ? 0 : -1;
            if (p < 0 || std::find(wanted.begin(), wanted.end(), order[p]) != wanted.end()) {
                break;
            }
            wanted.push_back(order[p]);
        }

        int s = find_slot(t);
        if (s < 0) {
            // not prefetched: the first tensor of a prompt, or the tensors are copied in an unexpected order
            n_misses++;
            s = pick_slot(wanted);
            if (s < 0) {
                s = 0;
            }
            load(s, t);
        }

        for (size_t i = 1; i < wanted.size(); ++i) {
            if (find_slot(wanted[i]) >= 0) {
                continue;
            }
            const int o = pick_slot(wanted);
            if (o < 0) {
                break;
            }
            load(o, wanted[i]);
        }

        return s;
    }

    //
    // selective reads: for small batches, only the selected experts that are not cached are read, once the router
    // has selected them. the reads of all the tensors of a layer are issued together.
    //

    void issue_selective(int t, const std::vector<bool> * used, int s) {
        release(s);

        auto & sl = slots[s];
        if (sl.tensor >= 0 && !sl.consumed) {
            n_wasted++; // a prefetch of a large batch
        }
        if (sl.sel_owner >= 0) {
            tensors[sl.sel_owner].sel_slot = -1;
        }
        sl.tensor    = -1;
        sl.consumed  = false;
        sl.sel_owner = t;

        auto & ti = tensors[t];
        const auto & l = layers[ti.layer];
        const size_t es = ti.expert_size;

        ti.sel_slot = s;
        if (used) {
            sel_used = *used;
        }

        // the runs of selected experts that are not cached
        std::vector<run> rs;
        size_t bytes = 0;
        ti.sel_exp.assign(ti.n_expert, SIZE_MAX);
        for (int64_t first = 0; first < ti.n_expert && used; ) {
            if (!(*used)[first] || l.slot_of[first] >= 0) {
                first++;
                continue;
            }
            int64_t last = first;
            while (last + 1 < ti.n_expert && (*used)[last + 1] && l.slot_of[last + 1] < 0) {
                last++;
            }
            run r;
            r.rd_beg    = align_down(ti.file_offs + first*es,      STREAM_ALIGN);
            r.rd_end    = align_up  (ti.file_offs + (last + 1)*es, STREAM_ALIGN);
            r.slot_offs = bytes;
            for (int64_t e = first; e <= last; ++e) {
                ti.sel_exp[e] = r.slot_offs + (ti.file_offs + e*es - r.rd_beg);
            }
            bytes += r.rd_end - r.rd_beg;
            rs.push_back(r);
            first = last + 1;
        }

        // without the selection, or too fragmented to fit: read all the experts that are not cached
        ti.sel_full = !used || bytes > slot_size;
        for (const auto & r : ti.sel_full ? ti.runs : rs) {
            submit_aligned(ti.file, r.rd_beg, r.rd_end, sl.data + r.slot_offs, sl.wg);
        }
    }

    // a staging buffer for selective reads: free, or holding a tensor that was uploaded
    int pick_selective() const {
        int best = -1;
        int best_score = -1;
        for (size_t s = 0; s < slots.size(); ++s) {
            const auto & sl = slots[s];
            const bool other = sl.sel_owner < 0 || tensors[sl.sel_owner].layer != sel_layer;
            const int score = other ? 2 : sl.consumed ? 1 : -1;
            if (score > best_score) {
                best       = (int) s;
                best_score = score;
            }
        }
        return best_score >= 0 ? best : -1;
    }

    int prepare_selective(int t, const std::vector<bool> * used) {
        const auto & ti = tensors[t];

        if (ti.layer != sel_layer || (used && *used != sel_used)) {
            // a new layer: read the selected experts of all its tensors, t first
            sel_layer = ti.layer;
            for (int idx : layers[sel_layer].tensors) {
                tensors[idx].sel_slot = -1;
            }
            std::vector<int> ts = { t };
            for (int idx : layers[sel_layer].tensors) {
                if (idx != t) {
                    ts.push_back(idx);
                }
            }
            for (int u : ts) {
                const int s = pick_selective();
                if (s < 0) {
                    break;
                }
                issue_selective(u, used, s);
            }
        }

        if (ti.sel_slot < 0) {
            // not enough staging buffers for the whole layer
            n_misses++;
            int s = pick_selective();
            issue_selective(t, used, s < 0 ? 0 : s);
        }

        return ti.sel_slot;
    }

    void copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, const std::vector<bool> * used) {
        const int    t  = tensor_idx.at(src);
        const auto & ti = tensors[t];
        const auto & l  = layers[ti.layer];
        const size_t es = ti.expert_size;

        int s = -1;
        if (l.streamed()) {
            if (t_first < 0.0) {
                t_first = t_now();
                bytes_read_start = bytes_read;
            }

            s = selective ? prepare_selective(t, used) : prepare(t);

            const double t0 = t_now();
            const bool ok = wait(slots[s].wg);
            t_stalled += t_now() - t0;

            if (!ok) {
                GGML_ABORT("moe_stream: %s: failed to read %s", __func__, ti.t->name);
            }
        }

        // where the experts that are not cached are in the staging buffer
        const bool sel = s >= 0 && selective && !ti.sel_full;
        const std::vector<size_t> & ex = sel ? ti.sel_exp : ti.exp_slot;
        if (s >= 0) {
            if (selective) {
                GGML_ASSERT(ti.sel_slot == s && slots[s].sel_owner == t);
            } else {
                GGML_ASSERT(slots[s].tensor == t && slots[s].layout == ti.layout);
            }
        }

        // copy [off, off + size) of the tensor, from the cache or from the staging buffer
        auto put = [&](size_t off, size_t size) {
            while (size > 0) {
                const int64_t e   = off / es;
                const bool    res = l.slot_of[e] >= 0;

                // extend over the next experts with the same source that are contiguous in it
                int64_t e2  = e;
                size_t  end = (e + 1)*es;
                while (end < off + size && e2 + 1 < ti.n_expert && (l.slot_of[e2 + 1] >= 0) == res &&
                        (res ? l.slot_of[e2 + 1] == l.slot_of[e2] + 1 :
                               ex[e2] != SIZE_MAX && ex[e2 + 1] == ex[e2] + es)) {
                    e2++;
                    end += es;
                }
                const size_t n = std::min(size, end - off);

                if (res) {
                    const uint8_t * p = ti.bank_data + l.slot_of[e]*es + (off - e*es);
                    ggml_backend_tensor_set_async(backend, dst, p, off, n);
                    bytes_up_res += n;
                } else if (ex[e] == SIZE_MAX) {
                    // the padding after the last selected expert, its content does not matter as long as it is finite
                    GGML_ASSERT(sel && n <= zeros.size() && "an expert was not read");
                    ggml_backend_tensor_set_async(backend, dst, zeros.data(), off, n);
                } else {
                    GGML_ASSERT(s >= 0);
                    const size_t so = ex[e] + (off - e*es);
                    GGML_ASSERT(so + n <= slot_size);
                    ggml_backend_tensor_set_async(backend, dst, slots[s].data + so, off, n);
                    bytes_up_slot += n;
                }

                off  += n;
                size -= n;
            }
        };

        if (!used) {
            put(0, ti.nbytes);
        } else {
            const int64_t n_expert = ti.n_expert;
            for (int64_t first = 0; first < n_expert; ) {
                if (!(*used)[first]) {
                    first++;
                    continue;
                }
                int64_t last = first;
                while (last + 1 < n_expert && (*used)[last + 1]) {
                    last++;
                }

                // copy a bit extra to ensure there are no NaNs in the padding of the last expert, this is necessary for MMQ in the CUDA backend
                const size_t padding = last < n_expert - 1 ? std::min<size_t>(es, 512) : 0;
                put(first*es, (last + 1 - first)*es + padding);

                first = last + 1;
            }
        }

        if (s >= 0) {
            auto & sl = slots[s];
            sl.ev_backend = backend;
            sl.ev_valid   = sl.event && ggml_backend_get_device(backend) == target_dev;
            if (sl.ev_valid) {
                ggml_backend_event_record(sl.event, backend);
            }
            sl.uploading = true;
            sl.consumed  = true;
        }
    }

    //
    // graph
    //

    void begin_graph(ggml_backend_sched_t sched, ggml_cgraph * gf, int64_t n_tok, bool more) {
        order.clear();
        order_pos.assign(tensors.size(), -1);

        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->op != GGML_OP_MUL_MAT_ID || !node->src[0]) {
                continue;
            }
            // the scheduler replaces the weights of the offloaded nodes with their copies on the device,
            // named "<backend>#<weight>#<copy>", the weights of the nodes computed on the CPU stay the same
            const char * name = node->src[0]->name;
            const char * beg  = strchr(name, '#');
            const char * end  = beg ? strrchr(name, '#') : nullptr;
            if (!beg || end <= beg) {
                continue;
            }
            auto it = name_idx.find(std::string(beg + 1, end));
            if (it == name_idx.end() || !layers[tensors[it->second].layer].streamed()) {
                continue;
            }
            ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, node);
            if (!backend || ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                continue;
            }
            if (order_pos[it->second] < 0) {
                order_pos[it->second] = (int) order.size();
                order.push_back(it->second);
            }
        }

        if (!order.empty()) {
            // the slots of the cached experts may have changed since the last prompt
            for (size_t li = 0; li < layers.size(); ++li) {
                auto & l = layers[li];
                if (!l.dirty) {
                    continue;
                }
                for (size_t s = 0; s < slots.size(); ++s) {
                    const int st = slots[s].tensor;
                    if (st >= 0 && tensors[st].layer == (int) li) {
                        release((int) s);
                        slots[s].tensor = -1;
                    }
                }
                build_layout(l);
                l.dirty = false;
            }
        } else {
            if (last_streamed) {
                // the uploads from the cache must be done before the cache changes
                ggml_backend_sched_synchronize(sched);
                cancel_prefetch();
            }
        }

        const bool sel = !order.empty() && n_tok <= selective_max;
        if (sel && !selective) {
            cancel_prefetch(); // reads of a large batch that is not coming
        }
        selective = sel;
        sel_layer = -1;
        for (auto & ti : tensors) {
            ti.sel_slot = -1;
        }
        for (auto & sl : slots) {
            sl.sel_owner = -1;
        }

        last_streamed = !order.empty();

        n_tokens      = n_tok;
        expect_more   = more;
        graph_cached  = false;
        t_first       = -1.0;
        t_stalled     = 0.0;
        n_misses      = 0;
        n_wasted      = 0;
        bytes_up_slot = 0;
        bytes_up_res  = 0;
    }

    // drop the reads queued for the staging buffers, generation needs the IO threads for the cache
    void cancel_prefetch() {
        std::lock_guard<std::mutex> lk(mtx);
        for (auto it = jobs.begin(); it != jobs.end(); ) {
            bool staging = false;
            for (auto & sl : slots) {
                if (it->wg == &sl.wg) {
                    staging = true;
                    sl.wg.pending--;
                    sl.tensor = -1; // incomplete
                    break;
                }
            }
            it = staging ? jobs.erase(it) : it + 1;
        }
        cv_done.notify_all();
    }

    void end_graph() {
        if (graph_cached && !last_streamed) {
            // generation (in prompt processing, only the last layers of ubatches with few outputs use the cache)
            dec_tokens   += n_tokens;
            state_tokens += n_tokens;
            if (dec_tokens >= DECODE_STATS_TOKENS && dec_access > 0) {
                LLAMA_LOG_INFO("moe_stream: generation: %" PRId64 " tokens, %.1f%% of the selected experts cached, %.1f missing per token, read %.2f GiB in %.1f s (%.2f GiB/s)\n",
                        dec_tokens, 100.0*dec_hits/dec_access, (double) (dec_access - dec_hits)/dec_tokens,
                        dec_bytes/GiB, dec_t_read, dec_bytes/GiB/std::max(dec_t_read, 1e-6));
                dec_tokens = 0;
                dec_access = 0;
                dec_hits   = 0;
                dec_bytes  = 0;
                dec_t_read = 0.0;
            }
            if (state_dirty && state_tokens >= STATE_SAVE_TOKENS) {
                save_state();
            }
        }

        if (t_first < 0.0) {
            return; // no experts were streamed
        }

        const double t    = std::max(t_now() - t_first, 1e-6);
        const size_t read = bytes_read - bytes_read_start;

        LLAMA_LOG_INFO("moe_stream: %" PRId64 " tokens%s: %.1f s, read %.2f GiB (%.2f GiB/s), uploaded %.2f GiB staged + %.2f GiB cached, stalled %.1f s, %d misses, %d wasted\n",
                n_tokens, selective ? " (selected experts)" : "", t, read/GiB, read/GiB/t, bytes_up_slot/GiB, bytes_up_res/GiB, t_stalled, n_misses, n_wasted);

        t_first = -1.0;
    }
#endif // __linux__
};

llama_moe_stream::llama_moe_stream(const llama_model & model, const std::vector<ggml_backend_t> & backends, const llama_moe_stream_params & params)
    : pimpl(std::make_unique<impl>(model, backends, params)) {}

llama_moe_stream::~llama_moe_stream() = default;

bool llama_moe_stream::active() const {
    return pimpl->is_active;
}

bool llama_moe_stream::is_streamed(const ggml_tensor * src) const {
    return pimpl->is_active && pimpl->tensor_idx.count(src) > 0;
}

ggml_tensor * llama_moe_stream::get_bank(const ggml_tensor * w) const {
    if (!pimpl->is_active) {
        return nullptr;
    }
    auto it = pimpl->tensor_idx.find(w);
    return it == pimpl->tensor_idx.end() ? nullptr : pimpl->tensors[it->second].bank;
}

ggml_backend_t llama_moe_stream::offload_backend(const ggml_tensor * w) const {
    if (!pimpl->is_active || !pimpl->tensor_idx.count(w)) {
        return nullptr;
    }
    return pimpl->target_backend;
}

ggml_tensor * llama_moe_stream::build_slots(ggml_context * ctx, ggml_tensor * selected_experts, int il) const {
#ifdef __linux__
    return pimpl->is_active ? pimpl->build_slots(ctx, selected_experts, il) : nullptr;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(selected_experts);
    GGML_UNUSED(il);
    return nullptr;
#endif
}

void llama_moe_stream::begin_graph(ggml_backend_sched_t sched, ggml_cgraph * gf, int64_t n_tokens, bool expect_more) {
#ifdef __linux__
    pimpl->begin_graph(sched, gf, n_tokens, expect_more);
#else
    GGML_UNUSED(sched);
    GGML_UNUSED(gf);
    GGML_UNUSED(n_tokens);
    GGML_UNUSED(expect_more);
#endif
}

void llama_moe_stream::end_graph() {
#ifdef __linux__
    pimpl->end_graph();
#endif
}

void llama_moe_stream::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, const std::vector<bool> * used) {
#ifdef __linux__
    pimpl->copy(backend, src, dst, used);
#else
    GGML_UNUSED(backend);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    GGML_UNUSED(used);
    GGML_ABORT("not supported");
#endif
}
