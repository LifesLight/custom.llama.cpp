#include "llama-moe-stream.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cmath>
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

// statistics of the batches computed on the CPU are logged every that many tokens
constexpr int CPU_STATS_TOKENS = 128;

// the cached experts are saved every that many tokens computed on the CPU
constexpr int STATE_SAVE_TOKENS = 256;

// the usage scores are rescaled when their weight reaches 2^SCORE_RESCALE
constexpr double SCORE_RESCALE = 512.0;

size_t align_down(size_t x, size_t a) { return x / a * a; }
size_t align_up  (size_t x, size_t a) { return (x + a - 1) / a * a; }

double t_now() { return ggml_time_us() * 1e-6; }

const char * policy_name(llama_moe_stream_policy p) {
    switch (p) {
        case LLAMA_MOE_STREAM_POLICY_LRU:  return "lru";
        case LLAMA_MOE_STREAM_POLICY_SLRU: return "slru";
        case LLAMA_MOE_STREAM_POLICY_LFU:  return "lfu";
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

    struct tensor {
        const ggml_tensor * t;

        int     layer;       // index in layers
        int     index;       // index in the tensors of the layer, also its staging buffer for the CPU overflow
        int     file;        // index in files
        size_t  file_offs;   // offset of the data in the file
        size_t  nbytes;
        size_t  expert_size;
        int64_t n_expert;

        // the cache: [ne0, ne1, n_slots + n_window], an expert per slot
        // the window after the cache slots is an alias of the staging buffer of the tensor, for the experts of
        // a batch computed on the CPU that do not fit in the cache
        ggml_tensor * bank      = nullptr;
        uint8_t *     bank_data = nullptr;
    };

    struct layer {
        int il;
        int64_t n_expert = 0;
        int64_t n_slots  = 0;     // cache slots
        bool    window   = false; // the cache has an overflow window
        std::vector<int> tensors;

        std::vector<int32_t> slot_of;   // [n_expert] cache slot of the expert, -1 if not cached
        std::vector<int32_t> expert_of; // [n_slots]  expert in the cache slot

        // replacement policy
        std::vector<uint64_t> stamp;    // [n_expert] last use
        std::vector<uint8_t>  prot;     // [n_expert] in the protected segment (slru)
        int64_t n_prot = 0;
        std::vector<double>   score;    // [n_expert] usage, decayed

        std::vector<uint32_t> mark;     // [n_expert] selected by the current batch
        uint32_t              mark_gen = 0;

        // the experts selected in the current graph, for the batches computed on the GPU
        std::vector<int32_t> counts;
        uint32_t             counts_gen = 0;

        bool dirty = false; // the cached experts changed since the staging buffers were filled

        // when each expert was last evicted (n_evict at the time), 0 = never
        std::vector<uint64_t> evicted_at;
        uint64_t n_evict = 0;

        bool streamed() const { return n_slots < n_expert; }
    };

    // a set of reads to wait for
    struct wait_group {
        int  pending = 0;     // guarded by mtx
        bool error   = false; // guarded by mtx
    };

    enum expert_state : uint8_t {
        EXP_NONE      = 0, // not in the staging buffer
        EXP_PLACED    = 1, // read or being read
        EXP_CANCELLED = 2, // has a place, but its reads were cancelled
    };

    struct slot {
        uint8_t * data = nullptr;

        int    tensor   = -1;    // the tensor held by the staging buffer
        bool   consumed = false; // the tensor was uploaded since it was assigned
        size_t fill     = 0;     // bytes of the buffer that are used

        std::vector<size_t>     off;   // [max n_expert] offset of the expert in the buffer
        std::vector<uint8_t>    state; // [max n_expert] expert_state
        std::vector<wait_group> wg;    // [max n_expert] the reads of the expert

        // the uploads from this buffer must finish before it can be overwritten
        ggml_backend_event_t event      = nullptr;
        ggml_backend_t       ev_backend = nullptr;
        bool                 ev_valid   = false; // the event was recorded after the last upload
        bool                 uploading  = false;
    };

    struct job {
        int          file;
        size_t       offs;
        size_t       len;
        uint8_t *    dst;
        wait_group * wg;
        bool         aligned; // offs, len and dst are aligned: read directly, otherwise through a bounce buffer
        int          slot   = -1; // the staging buffer and the expert, for the reads that can be cancelled
        int          expert = -1;
    };

    // speculative reads, by rank of the expert in the usage order of its layer, then by submission
    using spec_key = std::pair<int32_t, uint64_t>;

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
    std::unordered_map<std::string, int>         name_idx;
    int64_t max_n_expert = 0;

    ggml_backend_dev_t target_dev     = nullptr;
    ggml_backend_t     target_backend = nullptr;
    int64_t            n_expert_used  = 0;

    // batches of up to that many tokens are computed on the CPU (LLAMA_MOE_STREAM_CPU_MAX)
    int64_t cpu_max = 32;

    // half-life of the usage scores in tokens (LLAMA_MOE_STREAM_HALFLIFE)
    double halflife = 512.0;

    // compute the cached experts while the missing ones are read (LLAMA_MOE_STREAM_OVERLAP)
    bool overlap_mode = false;

    // the reads of the batch computed on the CPU between the start and the wait ops
    int                  pend_li = -1;
    wait_group           pend_wg;
    std::vector<int32_t> pend_all;   // [k*n] the slot of each selected expert
    std::vector<uint8_t> pend_read;  // [k*n] the expert is being read
    size_t               pend_read_start = 0;

    size_t page_size = 4096;

    // expert cache
    void *         arena      = nullptr; // the cache slots, and the windows mapped over the staging buffers
    size_t         arena_size = 0;
    ggml_backend_buffer_t bank_buf = nullptr;
    ggml_context * bank_ctx   = nullptr;
    std::vector<std::pair<void *, size_t>> bank_ranges; // the parts of the arena that are cache slots
    bool           pinned     = false;

    // staging buffers
    int                   memfd       = -1;      // the staging buffers, so that they can be mapped as windows
    void *                staging     = nullptr; // their mapping
    size_t                staging_size_ = 0;
    ggml_backend_buffer_t host_buf    = nullptr; // fallback without memfd: pinned host buffer
    void *                host_mem    = nullptr; // fallback: normal memory
    size_t                slot_size   = 0;       // the largest part of a tensor that is not cached, with alignment
    std::vector<slot>     slots;

    // pinned memory registration
    bool (*register_fn)(void *, size_t) = nullptr;
    void (*unregister_fn)(void *)       = nullptr;
    std::vector<void *> registered;

    std::string state_path;

    // IO threads
    std::mutex               mtx;
    std::condition_variable  cv_jobs;
    std::condition_variable  cv_done;
    std::deque<job>          jobs;    // reads that are needed now, first
    std::map<spec_key, job>  spec;    // speculative reads, when there is nothing else to read
    uint64_t                 spec_seq = 0;
    std::vector<std::thread> workers;
    bool                     stop = false;
    std::atomic<bool>        warned_dio_fallback { false };

    // graph state
    std::vector<int> order;     // the tensors streamed by the current graph, in the order of their matmuls
    std::vector<int> order_pos; // [tensors.size()] position of a tensor in order, -1 if not streamed
    bool     expect_more   = false;
    bool     last_streamed = false;
    int64_t  n_tokens      = 0;
    uint32_t graph_gen     = 0;
    uint64_t clock         = 0;

    // usage scores: an expert selected by a token gets 2^(tclock/halflife - sbase)
    double tclock = 0.0;
    double sbase  = 0.0;

    std::vector<int32_t> needed;
    std::vector<int32_t> missing;
    std::vector<int32_t> overflow;
    std::vector<int32_t> cnt;       // [max n_expert] tokens per expert in the current batch
    std::vector<int32_t> ov_index;  // [max n_expert] index in the window, -1 if not in it
    std::vector<uint8_t> being_read; // [max n_expert] read for the current batch
    std::vector<int32_t> rank;

    std::vector<uint8_t> zeros = std::vector<uint8_t>(512, 0); // padding of experts that are not read

    // stats of the current graph
    std::atomic<size_t> bytes_read { 0 }; // updated by the IO threads
    size_t bytes_read_start = 0;
    size_t bytes_up_slot    = 0;
    size_t bytes_up_res     = 0;
    double t_first   = -1.0;
    double t_stalled = 0.0;
    int    n_misses  = 0;
    int    n_wasted  = 0; // tensors in staging buffers reused before they were uploaded
    int64_t pf_need  = 0; // selected experts that were not cached
    int64_t pf_done  = 0; // of which the reads were done when the router selected them
    int64_t pf_queue = 0; // of which the reads were queued or in progress

    // stats of the batches computed on the CPU, since the last report
    int64_t cpu_tokens  = 0;
    int64_t cpu_batches = 0;
    int64_t cpu_access  = 0;
    int64_t cpu_hits    = 0;
    int64_t cpu_window  = 0;
    size_t  cpu_bytes   = 0;
    double  cpu_t_read  = 0.0;
    // how predictable the misses were: among the 2 / 8 most used experts that were not cached, or most recently evicted
    int64_t cpu_pred_used[2]    = { 0, 0 };
    int64_t cpu_pred_evicted[2] = { 0, 0 };

    bool    graph_cached = false; // the current graph was computed from the cache
    int64_t state_tokens = 0;     // tokens computed on the CPU since the state was saved
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
        page_size     = sysconf(_SC_PAGESIZE);

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
        if (const char * env = getenv("LLAMA_MOE_STREAM_CPU_MAX")) {
            cpu_max = std::max(0, atoi(env));
        }
        if (const char * env = getenv("LLAMA_MOE_STREAM_HALFLIFE")) {
            halflife = std::max(1.0, atof(env));
        }
        if (const char * env = getenv("LLAMA_MOE_STREAM_OVERLAP")) {
            overlap_mode = atoi(env) != 0;
        }
        slots.resize(n_slots);
        for (auto & sl : slots) {
            sl.off  .assign(max_n_expert, SIZE_MAX);
            sl.state.assign(max_n_expert, EXP_NONE);
            sl.wg = std::vector<wait_group>(max_n_expert);
        }
        cnt     .assign(max_n_expert, 0);
        ov_index.assign(max_n_expert, -1);
        being_read.assign(max_n_expert, 0);

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

        auto * reg = ggml_backend_dev_backend_reg(target_dev);
        if (reg) {
            // the registration of host memory is disabled by default in the CUDA backend, which only uses it for
            // this: enable it unless the user disabled it
            setenv("GGML_CUDA_REGISTER_HOST", "1", 0);
            register_fn   = (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_register_host_buffer");
            unregister_fn = (void (*)(void *))         ggml_backend_reg_get_proc_address(reg, "ggml_backend_unregister_host_buffer");
        }

        if (!alloc_staging() || !alloc_cache() || !fill_cache()) {
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
        for (void * p : registered) {
            unregister_fn(p);
        }
        if (bank_ctx) {
            ggml_free(bank_ctx);
        }
        if (bank_buf) {
            ggml_backend_buffer_free(bank_buf);
        }
        if (arena) {
            munmap(arena, arena_size);
        }
        if (staging) {
            munmap(staging, staging_size_);
        }
        if (memfd >= 0) {
            close(memfd);
        }
        if (host_buf) {
            ggml_backend_buffer_free(host_buf);
        }
        free(host_mem);
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
            ti.index       = -1;
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
                ti.index = (int) l.tensors.size();
                l.tensors.push_back((int) tensors.size());
                tensor_idx[ti.t]     = (int) tensors.size();
                name_idx[ti.t->name] = (int) tensors.size();
                tensors.push_back(ti);
            }
            if ((int) layer_of_il.size() <= il) {
                layer_of_il.resize(il + 1, -1);
            }
            layer_of_il[il] = li;
            max_n_expert = std::max(max_n_expert, l.n_expert);
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

    // the ZFS ARC is not counted in MemAvailable, but it shrinks down to its minimum size when memory is needed
    static size_t arc_evictable() {
        FILE * f = fopen("/proc/spl/kstat/zfs/arcstats", "r");
        if (!f) {
            return 0;
        }
        char line[256];
        char name[128];
        unsigned type = 0;
        unsigned long long value = 0;
        unsigned long long size = 0;
        unsigned long long c_min = 0;
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "%127s %u %llu", name, &type, &value) != 3) {
                continue;
            }
            if (strcmp(name, "size") == 0) {
                size = value;
            } else if (strcmp(name, "c_min") == 0) {
                c_min = value;
            }
        }
        fclose(f);
        return size > c_min ? (size_t) (size - c_min) : 0;
    }

    // the size of a staging buffer for the experts of a tensor that are not cached: each is read as its own
    // aligned range of the file, which can need up to 2 alignment blocks more
    size_t max_slot_bytes(const tensor & ti, int64_t n_cached) const {
        const int64_t n_streamed = ti.n_expert - n_cached;
        if (n_streamed <= 0) {
            return 0;
        }
        return align_up(n_streamed*(ti.expert_size + 2*STREAM_ALIGN), page_size);
    }

    size_t staging_size(double frac) const {
        size_t res = 0;
        for (const auto & l : layers) {
            const int64_t n_cached = (int64_t) (frac * (double) l.n_expert);
            for (int idx : l.tensors) {
                res = std::max(res, max_slot_bytes(tensors[idx], n_cached));
            }
        }
        return slots.size()*res;
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
            const size_t arc    = arc_evictable();
            const size_t avail  = mem_available() + arc;
            const size_t bounce = n_threads*(BOUNCE_CHUNK + 2*STREAM_ALIGN);
            frac = 0.0;
            for (int i = 0; i < 8; ++i) {
                const double budget = (double) avail - (double) reserve - (double) staging_size(frac) - (double) bounce;
                frac = std::clamp(budget / (double) total, 0.0, 1.0);
            }
            LLAMA_LOG_INFO("moe_stream: %s: %.2f GiB available (%.2f GiB of it ZFS ARC above its minimum), %.2f GiB reserved, %.2f GiB for the staging buffers -> %.2f GiB of cached experts\n",
                    __func__, avail/GiB, arc/GiB, reserve/GiB, staging_size(frac)/GiB, frac*total/GiB);
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
            l.score    .assign(l.n_expert, 0.0);
            l.mark     .assign(l.n_expert, 0);
            l.counts   .assign(l.n_expert, 0);
            l.evicted_at.assign(l.n_expert, 0);

            for (int idx : l.tensors) {
                auto & ti = tensors[idx];
                total_res += (size_t) l.n_slots * ti.expert_size;
                slot_size = std::max(slot_size, max_slot_bytes(ti, l.n_slots));
            }
        }

        LLAMA_LOG_INFO("moe_stream: %s: %zu layers, %.2f GiB of experts, %.2f GiB cached (%" PRId64 "/%" PRId64 " experts per layer), %.2f GiB streamed per ubatch, %s policy\n",
                __func__, layers.size(), total/GiB, total_res/GiB, layers[0].n_slots, layers[0].n_expert, (total - total_res)/GiB, policy_name(policy));
    }

    //
    // memory
    //

    bool register_pinned(void * p, size_t n) {
        if (!register_fn || !unregister_fn || n == 0) {
            return false;
        }
        if (!register_fn(p, n)) {
            return false;
        }
        registered.push_back(p);
        return true;
    }

    // the staging buffers: in a memfd, so that each can also be mapped as the overflow window of the cache of the
    // tensors computed on the CPU, and registered as pinned memory for the uploads
    bool alloc_staging() {
        if (slot_size == 0) {
            LLAMA_LOG_INFO("moe_stream: %s: all the experts are cached, nothing to stream\n", __func__);
            return true;
        }

        const size_t n_slots = slots.size();
        staging_size_ = n_slots*slot_size;

        uint8_t * base = nullptr;
        bool      pin  = false;

        memfd = memfd_create("moe-stream-staging", MFD_CLOEXEC);
        if (memfd >= 0 && ftruncate(memfd, staging_size_) == 0) {
            void * p = mmap(nullptr, staging_size_, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0);
            if (p != MAP_FAILED) {
                staging = p;
                base    = (uint8_t *) p;
                if (mlock(staging, staging_size_) != 0) {
                    LLAMA_LOG_WARN("moe_stream: %s: failed to lock the staging buffers: %s\n", __func__, strerror(errno));
                }
                pin = register_pinned(staging, staging_size_);
            }
        }
        if (!base) {
            LLAMA_LOG_WARN("moe_stream: %s: memfd failed (%s), batches computed on the CPU are limited to the cache size\n", __func__, strerror(errno));
            if (memfd >= 0) {
                close(memfd);
                memfd = -1;
            }
            if (auto * buft = ggml_backend_dev_host_buffer_type(target_dev)) {
                host_buf = ggml_backend_buft_alloc_buffer(buft, staging_size_ + STREAM_ALIGN);
            }
            if (host_buf) {
                base = (uint8_t *) align_up((size_t) ggml_backend_buffer_get_base(host_buf), STREAM_ALIGN);
                pin  = true;
            } else if (posix_memalign(&host_mem, STREAM_ALIGN, staging_size_) == 0) {
                base = (uint8_t *) host_mem;
            }
        }
        if (!base) {
            LLAMA_LOG_ERROR("moe_stream: %s: failed to allocate %.2f GiB for the staging buffers, MoE streaming disabled\n", __func__, staging_size_/GiB);
            return false;
        }

        for (size_t s = 0; s < n_slots; ++s) {
            slots[s].data  = base + s*slot_size;
            slots[s].event = ggml_backend_event_new(target_dev);
        }

        if (!pin) {
            LLAMA_LOG_WARN("moe_stream: %s: the staging buffers are not pinned, uploads from them will be slower\n", __func__);
        }
        LLAMA_LOG_INFO("moe_stream: %s: %zu x %.2f GiB %s staging buffers (%.2f GiB), %zu IO threads\n", __func__,
                n_slots, slot_size/GiB, pin ? "pinned" : "normal", staging_size_/GiB, workers.size());

        return true;
    }

    // the cache: for each tensor its slots, ending at a page boundary, followed by its overflow window if any
    bool alloc_cache() {
        struct region {
            int    t;
            size_t bank;   // bytes of the cache slots
            size_t padded; // rounded up to pages
            bool   window;
        };
        std::vector<region> regions;
        size_t total = 0;
        size_t total_banks = 0;
        int    n_windows = 0;

        for (auto & l : layers) {
            // a window per tensor of the layer, each over its own staging buffer
            l.window = memfd >= 0 && l.streamed() && l.tensors.size() <= slots.size();
            for (int idx : l.tensors) {
                const auto & ti = tensors[idx];
                region r;
                r.t      = idx;
                r.bank   = l.n_slots*ti.expert_size;
                r.padded = align_up(r.bank, page_size);
                r.window = l.window;
                if (r.bank == 0 && !r.window) {
                    continue;
                }
                total       += r.padded + (r.window ? slot_size : 0);
                total_banks += r.bank;
                n_windows   += r.window;
                regions.push_back(r);
            }
        }
        if (regions.empty()) {
            return true;
        }

        const double t_start = t_now();

        arena_size = total;
        arena = mmap(nullptr, arena_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (arena == MAP_FAILED) {
            arena = nullptr;
            LLAMA_LOG_ERROR("moe_stream: %s: failed to allocate %.2f GiB for the expert cache: %s\n", __func__, total_banks/GiB, strerror(errno));
            return false;
        }

        const char * env_pin = getenv("LLAMA_MOE_STREAM_PIN");
        const bool want_pin = env_pin && atoi(env_pin) != 0;

        bool lock_failed = false;
        size_t n_pinned = 0;

        uint8_t * cur = (uint8_t *) arena;
        std::vector<uint8_t *> starts(regions.size());
        for (size_t i = 0; i < regions.size(); ++i) {
            const auto & r  = regions[i];
            const auto & ti = tensors[r.t];

            if (r.padded > 0) {
                // locking also allocates all the pages now
                if (!lock_failed && mlock(cur, r.padded) != 0) {
                    lock_failed = true;
                    LLAMA_LOG_WARN("moe_stream: %s: failed to lock the expert cache (%s), it may be swapped out\n", __func__, strerror(errno));
                    LLAMA_LOG_WARN("moe_stream: %s: raise the limit with 'ulimit -l unlimited' (or LimitMEMLOCK=infinity for a systemd service)\n", __func__);
                }
                if (want_pin && register_pinned(cur, r.padded)) {
                    n_pinned += r.padded;
                }
                bank_ranges.emplace_back(cur, r.padded);
            }
            starts[i] = cur + r.padded - r.bank;
            cur += r.padded;

            if (r.window) {
                void * p = mmap(cur, slot_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, memfd, (off_t) (ti.index*slot_size));
                if (p == MAP_FAILED) {
                    LLAMA_LOG_ERROR("moe_stream: %s: failed to map an overflow window: %s\n", __func__, strerror(errno));
                    return false;
                }
                cur += slot_size;
            }
        }

        bank_buf = ggml_backend_cpu_buffer_from_ptr(arena, arena_size);
        ggml_backend_buffer_set_usage(bank_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        ggml_init_params ip = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(regions.size() + 1),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        bank_ctx = ggml_init(ip);

        for (size_t i = 0; i < regions.size(); ++i) {
            const auto & r  = regions[i];
            auto       & ti = tensors[r.t];
            const auto & l  = layers[ti.layer];

            const int64_t n_window = r.window ? (int64_t) (slot_size / ti.expert_size) : 0;
            GGML_ASSERT(!r.window || n_window >= l.n_expert - l.n_slots);

            ti.bank = ggml_new_tensor_3d(bank_ctx, ti.t->type, ti.t->ne[0], ti.t->ne[1], l.n_slots + n_window);
            GGML_ASSERT((size_t) ti.bank->nb[2] == ti.expert_size);
            ggml_format_name(ti.bank, "%s.cache", ti.t->name);
            ggml_backend_tensor_alloc(bank_buf, ti.bank, starts[i]);
            ti.bank_data = starts[i];
        }

        pinned = want_pin && n_pinned > 0;

        LLAMA_LOG_INFO("moe_stream: %s: allocated %.2f GiB of %s memory for the expert cache in %.1f s, %d overflow windows\n", __func__,
                total_banks/GiB, pinned ? "pinned" : "locked", t_now() - t_start, n_windows);

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

    // the largest batch of a layer computed on the CPU
    int64_t cpu_limit(const layer & l) const {
        if (!l.streamed() || l.window) {
            return cpu_max;
        }
        // without a window, all the experts of a batch must fit in the cache
        return std::min<int64_t>(cpu_max, n_expert_used > 0 ? l.n_slots / n_expert_used : 0);
    }

    // log how batches are computed
    void check_offload() {
        int64_t lim = INT64_MAX;
        for (const auto & l : layers) {
            lim = std::min(lim, cpu_limit(l));
        }
        LLAMA_LOG_INFO("moe_stream: %s: batches of up to %" PRId64 " tokens are computed on the CPU from the cache (LLAMA_MOE_STREAM_CPU_MAX=%" PRId64 "),"
                " larger ones on %s, prefetching the experts in the order of their usage\n", __func__,
                lim, cpu_max, ggml_backend_name(target_backend));
        if (lim < cpu_max) {
            LLAMA_LOG_WARN("moe_stream: %s: without overflow windows, batches on the CPU are limited to %" PRId64 " tokens\n", __func__, lim);
        }
    }

    //
    // saved state: the cached experts of each layer, most recently used first, and the usage scores
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
        if (!std::getline(in, line) || (line != "moe-stream-state 1" && line != "moe-stream-state 2")) {
            LLAMA_LOG_WARN("moe_stream: %s: %s is not a state file, ignored\n", __func__, state_path.c_str());
            return 0;
        }

        int n_loaded = 0;
        while (std::getline(in, line)) {
            std::istringstream ss(line);
            if (line.rfind("s ", 0) == 0) {
                // scores
                std::string tag;
                int il = -1;
                int64_t n_expert = 0;
                if (!(ss >> tag >> il >> n_expert) || il < 0 || il >= (int) layer_of_il.size() || layer_of_il[il] < 0) {
                    continue;
                }
                auto & l = layers[layer_of_il[il]];
                if (n_expert != l.n_expert) {
                    continue;
                }
                for (int64_t e = 0; e < l.n_expert; ++e) {
                    double v = 0.0;
                    if (!(ss >> v)) {
                        break;
                    }
                    l.score[e] = std::isfinite(v) && v > 0.0 ? v : 0.0;
                }
                continue;
            }
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
            out << "moe-stream-state 2\n";
            out.precision(6);
            const double w = weight();
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

                // relative to the current weight, so that they are restored at the scale of a fresh clock
                out << "s " << l.il << " " << l.n_expert;
                for (int64_t e = 0; e < l.n_expert; ++e) {
                    out << " " << l.score[e]/w;
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
    // usage scores: each selection adds 2^(tclock/halflife), relative to sbase, the scores are rescaled before
    // the weights get large, so they never overflow
    //

    double weight() const {
        return std::exp2(tclock/halflife - sbase);
    }

    void advance(int64_t n) {
        tclock += (double) n;
        if (tclock/halflife - sbase > SCORE_RESCALE) {
            const double f = std::exp2(-SCORE_RESCALE);
            for (auto & l : layers) {
                for (auto & v : l.score) {
                    v *= f;
                }
            }
            sbase += SCORE_RESCALE;
        }
    }

    // the order in which the experts of a layer that are not cached are read: most used first
    void ranked(const layer & l, std::vector<int32_t> & out) const {
        out.clear();
        for (int64_t e = 0; e < l.n_expert; ++e) {
            if (l.slot_of[e] < 0) {
                out.push_back((int32_t) e);
            }
        }
        std::stable_sort(out.begin(), out.end(), [&](int32_t a, int32_t b) { return l.score[a] > l.score[b]; });
    }

    //
    // replacement policy, for the batches computed on the CPU
    //

    // the cached expert to evict that is not selected by the batch, -1 if there is none
    int64_t victim(const layer & l) const {
        int64_t best = -1;
        int64_t best_prob = -1; // slru: the least recently used of the probation half first
        for (int64_t s = 0; s < l.n_slots; ++s) {
            const int64_t e = l.expert_of[s];
            if (l.mark[e] == l.mark_gen) {
                continue;
            }
            if (best < 0) {
                best = e;
            } else if (policy == LLAMA_MOE_STREAM_POLICY_LFU) {
                if (l.score[e] < l.score[best] || (l.score[e] == l.score[best] && l.stamp[e] < l.stamp[best])) {
                    best = e;
                }
            } else if (l.stamp[e] < l.stamp[best]) {
                best = e;
            }
            if (policy == LLAMA_MOE_STREAM_POLICY_SLRU && !l.prot[e] && (best_prob < 0 || l.stamp[e] < l.stamp[best_prob])) {
                best_prob = e;
            }
        }
        return best_prob >= 0 ? best_prob : best;
    }

    // a second use promotes to the protected half (slru), its least recently used expert goes back to probation
    void promote(layer & l, int64_t e) {
        if (policy != LLAMA_MOE_STREAM_POLICY_SLRU || l.prot[e]) {
            return;
        }
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

    // make the experts selected by a batch computed on the CPU available and write their slots:
    // the missing experts replace cached ones, or go to the overflow window when the cache is full of experts
    // selected by the batch (or, with lfu, when they are used less than the expert they would replace)
    void ensure(int li, const ggml_tensor * ids, ggml_tensor * out) {
        auto & l = layers[li];

        GGML_ASSERT(pend_li < 0 && "the previous layer did not wait for its reads");

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
                    cnt[e] = 0;
                    needed.push_back(e);
                }
                cnt[e]++;
            }
        }

        const double w = weight();
        missing.clear();
        for (int32_t e : needed) {
            cpu_access++;
            if (l.slot_of[e] < 0) {
                missing.push_back(e);
                continue;
            }
            cpu_hits++;
            promote(l, e);
        }

        // how the misses could have been predicted, with the scores before this batch
        for (int32_t e : missing) {
            int64_t higher = 0;
            for (int64_t x = 0; x < l.n_expert; ++x) {
                higher += l.slot_of[x] < 0 && l.score[x] > l.score[e];
            }
            cpu_pred_used[0] += higher < 2;
            cpu_pred_used[1] += higher < 8;
            if (l.evicted_at[e] > 0) {
                const uint64_t ago = l.n_evict - l.evicted_at[e];
                cpu_pred_evicted[0] += ago < 2;
                cpu_pred_evicted[1] += ago < 8;
            }
        }

        for (int32_t e : needed) {
            l.stamp[e]  = clock;
            l.score[e] += cnt[e]*w;
        }

        overflow.clear();
        const size_t read_start = bytes_read;
        wait_group & wg = pend_wg;
        if (!missing.empty()) {
            const double t0 = t_now();

            // the most used first, so that with lfu they are the ones admitted
            std::stable_sort(missing.begin(), missing.end(), [&](int32_t a, int32_t b) { return l.score[a] > l.score[b]; });

            bool changed = false;
            for (int32_t e : missing) {
                const int64_t v = victim(l);
                if (v < 0 || (l.window && policy == LLAMA_MOE_STREAM_POLICY_LFU && l.score[e] <= l.score[v])) {
                    overflow.push_back(e);
                    continue;
                }

                const int32_t s = l.slot_of[v];
                if (l.prot[v]) {
                    l.prot[v] = 0;
                    l.n_prot--;
                }
                l.slot_of[v]   = -1;
                l.slot_of[e]   = s;
                l.expert_of[s] = e;
                l.mark[v]      = l.mark_gen; // the slot is taken, not a victim again
                l.evicted_at[v] = ++l.n_evict;
                changed        = true;

                read_experts(l, e, s, 1, wg);
            }

            if (!overflow.empty()) {
                GGML_ASSERT(l.window && "more experts selected than cache slots");

                // the windows are the staging buffers of the tensors: wait for what uses them and drop their content
                for (int idx : l.tensors) {
                    const int s = tensors[idx].index;
                    release(s);
                    reset(s, -1);
                }
                for (size_t j = 0; j < overflow.size(); ++j) {
                    const int32_t e = overflow[j];
                    ov_index[e] = (int32_t) j;
                    for (int idx : l.tensors) {
                        const auto & ti = tensors[idx];
                        uint8_t * dst = ti.bank_data + (l.n_slots + j)*ti.expert_size;
                        submit_bounce(ti.file, ti.file_offs + e*ti.expert_size, ti.expert_size, dst, wg);
                    }
                }
                cpu_window += overflow.size();
            }

            if (changed) {
                l.dirty     = true;
                state_dirty = true;
            }

            if (!overlap_mode) {
                if (!wait(wg)) {
                    GGML_ABORT("moe_stream: failed to read the experts of layer %d", l.il);
                }
                cpu_bytes  += bytes_read - read_start;
                cpu_t_read += t_now() - t0;
            }
        }

        // the experts read for this batch: the missing ones, now in a cache slot or in the window
        for (int32_t e : missing) {
            being_read[e] = 1;
        }

        int32_t * dst = (int32_t *) out->data;
        pend_all .resize(k*n);
        pend_read.resize(k*n);
        for (int64_t i1 = 0; i1 < n; ++i1) {
            for (int64_t i0 = 0; i0 < k; ++i0) {
                const int32_t e    = id_at(i0, i1);
                const int32_t slot = l.slot_of[e] >= 0 ? l.slot_of[e] : (int32_t) l.n_slots + ov_index[e];
                const bool    rd   = being_read[e] != 0;
                pend_all [i1*k + i0] = slot;
                pend_read[i1*k + i0] = rd;
                dst[i1*k + i0] = overlap_mode && rd ? -1 : slot;
            }
        }

        for (int32_t e : missing) {
            being_read[e] = 0;
        }
        for (int32_t e : overflow) {
            ov_index[e] = -1;
        }

        if (overlap_mode) {
            pend_li         = li;
            pend_read_start = read_start;
        }
    }

    // the cached experts are computed: wait for the missing ones
    void wait_reads(int li, ggml_tensor * out) {
        GGML_ASSERT(pend_li == li);
        pend_li = -1;

        const double t0 = t_now();
        if (!wait(pend_wg)) {
            GGML_ABORT("moe_stream: failed to read the experts of layer %d", layers[li].il);
        }
        cpu_t_read += t_now() - t0;
        cpu_bytes  += bytes_read - pend_read_start;

        const size_t kn = pend_all.size();
        int32_t * dst = (int32_t *) out->data;
        for (size_t i = 0; i < kn; ++i) {
            dst[i]      = pend_read[i] ? pend_all[i] : -1;
            dst[kn + i] = pend_all[i];
        }
    }

    static void wait_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
        GGML_UNUSED(nth);
        if (ith != 0) {
            return;
        }
        auto * c = (op_ctx *) userdata;
        c->self->wait_reads(c->li, dst);
    }

    ggml_tensor * build_wait(ggml_context * ctx, ggml_tensor * slots_t, const std::vector<ggml_tensor *> & deps, int il) const {
        if (!overlap_mode || il < 0 || il >= (int) layer_of_il.size() || layer_of_il[il] < 0) {
            return nullptr;
        }
        const int li = layer_of_il[il];

        // the first parts are sources, so that the wait is computed after them
        std::vector<ggml_tensor *> args = { slots_t };
        for (ggml_tensor * d : deps) {
            if (args.size() < GGML_MAX_SRC) {
                args.push_back(d);
            }
        }
        return ggml_custom_4d(ctx, GGML_TYPE_I32, slots_t->ne[0], slots_t->ne[1], 2, 1, args.data(), (int) args.size(),
                wait_op, 1, const_cast<op_ctx *>(&op_ctxs[li]));
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
        if (n > cpu_limit(l) || n == 0) {
            return nullptr;
        }
        for (int idx : l.tensors) {
            if (!tensors[idx].bank) {
                return nullptr;
            }
        }

        ggml_tensor * args[1] = { selected };
        return ggml_custom_4d(ctx, GGML_TYPE_I32, k, n, 1, 1, args, 1, ensure_op, 1, const_cast<op_ctx *>(&op_ctxs[li]));
    }

    //
    // IO threads
    //

    // read any range of a file into any buffer, through the bounce buffers of the IO threads
    void submit_bounce(int fi, size_t offs, size_t len, uint8_t * dst, wait_group & wg) {
        std::lock_guard<std::mutex> lk(mtx);
        for (size_t o = 0; o < len; o += BOUNCE_CHUNK) {
            job j;
            j.file    = fi;
            j.offs    = offs + o;
            j.len     = std::min(BOUNCE_CHUNK, len - o);
            j.dst     = dst + o;
            j.wg      = &wg;
            j.aligned = false;
            jobs.push_back(j);
            wg.pending++;
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
                cv_jobs.wait(lk, [this] { return stop || !jobs.empty() || !spec.empty(); });
                if (!jobs.empty()) {
                    j = jobs.front();
                    jobs.pop_front();
                } else if (!spec.empty()) {
                    j = spec.begin()->second;
                    spec.erase(spec.begin());
                } else {
                    break; // stop
                }
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
    // staging buffers, for the batches computed on the GPU: the experts of a tensor that are not cached are read
    // into its buffer most used first, before the router selected them; once it did, the reads of the experts
    // that are not selected are cancelled and the selected ones that are missing are read first
    //

    // cancel the queued reads of a staging buffer, of all its experts or only of those that keep[] rejects
    // call with mtx locked
    template <typename F>
    void cancel_locked(int s, F && drop) {
        auto & sl = slots[s];
        std::deque<job> kept;
        for (auto & j : jobs) {
            if (j.slot == s && drop(j.expert)) {
                j.wg->pending--;
                sl.state[j.expert] = EXP_CANCELLED;
                continue;
            }
            kept.push_back(j);
        }
        jobs.swap(kept);
        for (auto it = spec.begin(); it != spec.end(); ) {
            job & j = it->second;
            if (j.slot == s && drop(j.expert)) {
                j.wg->pending--;
                sl.state[j.expert] = EXP_CANCELLED;
                it = spec.erase(it);
                continue;
            }
            ++it;
        }
        cv_done.notify_all();
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
        std::unique_lock<std::mutex> lk(mtx);
        cancel_locked(s, [](int) { return true; });
        cv_done.wait(lk, [&] {
            for (const auto & wg : sl.wg) {
                if (wg.pending > 0) {
                    return false;
                }
            }
            return true;
        });
    }

    // give a staging buffer to tensor t (-1 = none), after release()
    void reset(int s, int t) {
        auto & sl = slots[s];
        if (sl.tensor >= 0 && !sl.consumed && t >= 0) {
            n_wasted++;
        }
        sl.tensor   = t;
        sl.consumed = false;
        sl.fill     = 0;
        std::fill(sl.off.begin(),   sl.off.end(),   SIZE_MAX);
        std::fill(sl.state.begin(), sl.state.end(), (uint8_t) EXP_NONE);
        for (auto & wg : sl.wg) {
            wg.error = false;
        }
    }

    // read expert e into its place in the staging buffer: needed now (into head, rank < 0), or speculatively
    // call with mtx locked
    void place_locked(int s, int64_t e, int32_t rank_e, std::vector<job> & head) {
        auto & sl = slots[s];
        const auto & ti = tensors[sl.tensor];
        const size_t es = ti.expert_size;

        const size_t beg = align_down(ti.file_offs + e*es,       STREAM_ALIGN);
        const size_t end = align_up  (ti.file_offs + (e + 1)*es, STREAM_ALIGN);

        uint8_t * dst;
        if (sl.state[e] == EXP_NONE) {
            GGML_ASSERT(sl.fill + (end - beg) <= slot_size);
            sl.off[e] = sl.fill + (ti.file_offs + e*es - beg);
            dst = sl.data + sl.fill;
            sl.fill += end - beg;
        } else {
            // cancelled: read again at the same place, its reads in flight write the same data
            dst = sl.data + sl.off[e] - (ti.file_offs + e*es - beg);
        }
        sl.state[e] = EXP_PLACED;

        for (size_t o = beg; o < end; o += STREAM_CHUNK) {
            job j;
            j.file    = ti.file;
            j.offs    = o;
            j.len     = std::min(STREAM_CHUNK, end - o);
            j.dst     = dst + (o - beg);
            j.wg      = &sl.wg[e];
            j.aligned = true;
            j.slot    = s;
            j.expert  = (int) e;
            if (rank_e < 0) {
                head.push_back(j);
            } else {
                spec.emplace(spec_key(rank_e, spec_seq++), j);
            }
            sl.wg[e].pending++;
        }
    }

    // read the experts that are not cached, most used first, interleaved with the other tensors read ahead by rank
    // all: needed now (the router did not give the selection)
    void speculate(int s, bool all = false) {
        auto & sl = slots[s];
        const auto & l = layers[tensors[sl.tensor].layer];

        ranked(l, rank);

        std::vector<job> head;
        std::lock_guard<std::mutex> lk(mtx);
        if (all) {
            cancel_locked(s, [](int) { return true; }); // move them to the reads needed now
        }
        for (size_t r = 0; r < rank.size(); ++r) {
            const int32_t e = rank[r];
            if (sl.state[e] != EXP_PLACED) {
                place_locked(s, e, all ? -1 : (int32_t) r, head);
            }
        }
        jobs.insert(jobs.begin(), head.begin(), head.end());
        cv_jobs.notify_all();
    }

    // the router selected the experts: cancel the reads of the others, read the missing ones first
    void fill(int s, const std::vector<int32_t> & counts) {
        auto & sl = slots[s];
        const auto & l = layers[tensors[sl.tensor].layer];

        std::vector<job> head;
        std::lock_guard<std::mutex> lk(mtx);
        cancel_locked(s, [&](int e) { return counts[e] == 0; });
        // the speculative reads left are of selected experts: they are needed now, most used first
        for (auto it = spec.begin(); it != spec.end(); ) {
            if (it->second.slot == s) {
                head.push_back(it->second);
                it = spec.erase(it);
                continue;
            }
            ++it;
        }
        for (int64_t e = 0; e < l.n_expert; ++e) {
            if (counts[e] > 0 && l.slot_of[e] < 0 && sl.state[e] != EXP_PLACED) {
                place_locked(s, e, -1, head);
            }
        }
        jobs.insert(jobs.begin(), head.begin(), head.end());
        cv_jobs.notify_all();
    }

    int find_slot(int t) const {
        for (size_t s = 0; s < slots.size(); ++s) {
            if (slots[s].tensor == t) {
                return (int) s;
            }
        }
        return -1;
    }

    // the staging buffer to give to a new tensor: not one of the wanted tensors, preferably empty or already uploaded
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

    // give a staging buffer to tensor t and start its reads
    void assign(int s, int t) {
        release(s);
        reset(s, t);
        const auto & l = layers[tensors[t].layer];
        if (l.counts_gen == graph_gen) {
            fill(s, l.counts); // the router of this layer already selected the experts
        } else {
            speculate(s);
        }
    }

    // make sure tensor t has a staging buffer, and start the reads of the next tensors of the graph in the others
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
            assign(s, t);
        }

        for (size_t i = 1; i < wanted.size(); ++i) {
            if (find_slot(wanted[i]) >= 0) {
                continue;
            }
            const int o = pick_slot(wanted);
            if (o < 0) {
                break;
            }
            assign(o, wanted[i]);
        }

        return s;
    }

    bool expert_ready(int s, int64_t e) {
        const auto & sl = slots[s];
        if (sl.state[e] != EXP_PLACED) {
            return false;
        }
        std::lock_guard<std::mutex> lk(mtx);
        return sl.wg[e].pending == 0 && !sl.wg[e].error;
    }

    void copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, const std::vector<int32_t> * counts) {
        const int    t  = tensor_idx.at(src);
        const auto & ti = tensors[t];
        auto       & l  = layers[ti.layer];
        const size_t es = ti.expert_size;

        // the router selected the experts of this layer: record their usage, once per graph
        if (counts && l.counts_gen != graph_gen) {
            l.counts_gen = graph_gen;
            l.counts     = *counts;
            const double w = weight();
            for (int64_t e = 0; e < l.n_expert; ++e) {
                l.score[e] += l.counts[e]*w;
            }
        }

        int s = -1;
        if (l.streamed()) {
            if (t_first < 0.0) {
                t_first = t_now();
                bytes_read_start = bytes_read;
            }

            if (counts) {
                // count how many of the selected experts that are not cached were already being read
                const int s0 = find_slot(t);
                std::lock_guard<std::mutex> lk(mtx);
                for (int64_t e = 0; e < l.n_expert; ++e) {
                    if ((*counts)[e] > 0 && l.slot_of[e] < 0) {
                        pf_need++;
                        if (s0 >= 0 && slots[s0].state[e] == EXP_PLACED) {
                            if (slots[s0].wg[e].pending == 0) {
                                pf_done++;
                            } else {
                                pf_queue++;
                            }
                        }
                    }
                }
            }

            s = prepare(t);

            if (counts) {
                // this tensor and the other tensors of the layer that already have a buffer
                for (size_t o = 0; o < slots.size(); ++o) {
                    if (slots[o].tensor >= 0 && tensors[slots[o].tensor].layer == ti.layer) {
                        fill((int) o, *counts);
                    }
                }
            } else {
                speculate(s, true); // all of them
            }

            const double t0 = t_now();
            bool ok = true;
            for (int64_t e = 0; e < l.n_expert; ++e) {
                if (l.slot_of[e] >= 0 || (counts && (*counts)[e] == 0)) {
                    continue;
                }
                ok = wait(slots[s].wg[e]) && ok;
            }
            t_stalled += t_now() - t0;

            if (!ok) {
                GGML_ABORT("moe_stream: %s: failed to read %s", __func__, ti.t->name);
            }
        }

        // copy [off, off + size) of the tensor, from the cache or from the staging buffer, whole experts
        // except for the padding after the last one, which is zeros if the expert is not available
        auto put = [&](size_t off, size_t size, bool padding) {
            while (size > 0) {
                const int64_t e   = off / es;
                const bool    res = l.slot_of[e] >= 0;

                // extend over the next cached experts that are in consecutive slots
                int64_t e2  = e;
                size_t  end = (e + 1)*es;
                while (res && end < off + size && e2 + 1 < ti.n_expert && l.slot_of[e2 + 1] == l.slot_of[e2] + 1) {
                    e2++;
                    end += es;
                }
                const size_t n = std::min(size, end - off);

                if (res) {
                    const uint8_t * p = ti.bank_data + l.slot_of[e]*es + (off - e*es);
                    ggml_backend_tensor_set_async(backend, dst, p, off, n);
                    bytes_up_res += n;
                } else if (s >= 0 && expert_ready(s, e)) {
                    const size_t so = slots[s].off[e] + (off - e*es);
                    GGML_ASSERT(so + n <= slot_size);
                    ggml_backend_tensor_set_async(backend, dst, slots[s].data + so, off, n);
                    bytes_up_slot += n;
                } else {
                    GGML_ASSERT(padding && n <= zeros.size() && "an expert was not read");
                    ggml_backend_tensor_set_async(backend, dst, zeros.data(), off, n);
                }

                off  += n;
                size -= n;
            }
        };

        const int64_t n_expert = ti.n_expert;
        for (int64_t first = 0; first < n_expert; ) {
            if (counts && (*counts)[first] == 0) {
                first++;
                continue;
            }
            int64_t last = first;
            while (last + 1 < n_expert && (!counts || (*counts)[last + 1] > 0)) {
                last++;
            }

            put(first*es, (last + 1 - first)*es, false);

            // copy a bit extra to ensure there are no NaNs in the padding of the last expert, this is necessary for MMQ in the CUDA backend
            if (last < n_expert - 1) {
                put((last + 1)*es, std::min<size_t>(es, 512), true);
            }

            first = last + 1;
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

        graph_gen++;
        advance(n_tok);

        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->op != GGML_OP_MUL_MAT_ID || !node->src[0] || ggml_nelements(node) == 0) {
                continue; // no tokens (e.g. the last layer of a ubatch without outputs): its experts are not copied
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
            // the cached experts may have changed since the staging buffers were filled
            for (size_t li = 0; li < layers.size(); ++li) {
                auto & l = layers[li];
                if (!l.dirty) {
                    continue;
                }
                for (size_t s = 0; s < slots.size(); ++s) {
                    const int st = slots[s].tensor;
                    if (st >= 0 && tensors[st].layer == (int) li) {
                        release((int) s);
                        reset((int) s, -1);
                    }
                }
                l.dirty = false;
            }
        } else if (last_streamed) {
            // the uploads from the cache and the staging buffers must be done before the cache changes,
            // and generation needs the IO threads
            ggml_backend_sched_synchronize(sched);
            cancel_prefetch();
        }
        last_streamed = !order.empty();

        n_tokens      = n_tok;
        expect_more   = more;
        graph_cached  = false;
        t_first       = -1.0;
        t_stalled     = 0.0;
        n_misses      = 0;
        n_wasted      = 0;
        pf_need       = 0;
        pf_done       = 0;
        pf_queue      = 0;
        bytes_up_slot = 0;
        bytes_up_res  = 0;
    }

    // drop the queued reads into the staging buffers
    void cancel_prefetch() {
        std::lock_guard<std::mutex> lk(mtx);
        for (size_t s = 0; s < slots.size(); ++s) {
            cancel_locked((int) s, [](int) { return true; });
        }
    }

    void end_graph() {
        if (graph_cached && !last_streamed) {
            cpu_tokens   += n_tokens;
            cpu_batches  += 1;
            state_tokens += n_tokens;
            if (cpu_tokens >= CPU_STATS_TOKENS && cpu_access > 0) {
                const double n_miss = (double) std::max<int64_t>(cpu_access - cpu_hits, 1);
                LLAMA_LOG_INFO("moe_stream: cpu: %" PRId64 " tokens in %" PRId64 " batches, %.1f%% of the selected experts cached, %.1f missing per batch"
                        " (%.1f in the overflow window), read %.2f GiB, %s %.1f s\n",
                        cpu_tokens, cpu_batches, 100.0*cpu_hits/cpu_access, (double) (cpu_access - cpu_hits)/cpu_batches, (double) cpu_window/cpu_batches,
                        cpu_bytes/GiB, overlap_mode ? "waited" : "in", cpu_t_read);
                LLAMA_LOG_INFO("moe_stream: cpu: of the missing experts, %.0f%% / %.0f%% were among the 2 / 8 most used not cached, %.0f%% / %.0f%% among the 2 / 8 last evicted\n",
                        100.0*cpu_pred_used[0]/n_miss, 100.0*cpu_pred_used[1]/n_miss, 100.0*cpu_pred_evicted[0]/n_miss, 100.0*cpu_pred_evicted[1]/n_miss);
                cpu_pred_used[0] = cpu_pred_used[1] = 0;
                cpu_pred_evicted[0] = cpu_pred_evicted[1] = 0;
                cpu_tokens  = 0;
                cpu_batches = 0;
                cpu_access  = 0;
                cpu_hits    = 0;
                cpu_window  = 0;
                cpu_bytes   = 0;
                cpu_t_read  = 0.0;
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

        LLAMA_LOG_INFO("moe_stream: %" PRId64 " tokens: %.1f s, read %.2f GiB (%.2f GiB/s), uploaded %.2f GiB staged + %.2f GiB cached, stalled %.1f s,"
                " selected experts read ahead %.0f%% (+%.0f%% in progress), %d misses, %d wasted\n",
                n_tokens, t, read/GiB, read/GiB/t, bytes_up_slot/GiB, bytes_up_res/GiB, t_stalled,
                100.0*pf_done/std::max<int64_t>(pf_need, 1), 100.0*pf_queue/std::max<int64_t>(pf_need, 1), n_misses, n_wasted);

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

bool llama_moe_stream::overlap() const {
    return pimpl->is_active && pimpl->overlap_mode;
}

ggml_tensor * llama_moe_stream::build_wait(ggml_context * ctx, ggml_tensor * slots, const std::vector<ggml_tensor *> & deps, int il) const {
#ifdef __linux__
    return pimpl->is_active ? pimpl->build_wait(ctx, slots, deps, il) : nullptr;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(slots);
    GGML_UNUSED(deps);
    GGML_UNUSED(il);
    return nullptr;
#endif
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

void llama_moe_stream::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, const std::vector<int32_t> * counts) {
#ifdef __linux__
    pimpl->copy(backend, src, dst, counts);
#else
    GGML_UNUSED(backend);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    GGML_UNUSED(counts);
    GGML_ABORT("not supported");
#endif
}
