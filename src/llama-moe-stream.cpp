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
#include <set>
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
        int     index;       // index in the tensors of the layer
        int     role;        // index in the banks of the pool, also the staging buffer of its overflow window
        int     file;        // index in files
        size_t  file_offs;   // offset of the data in the file
        size_t  nbytes;
        size_t  expert_size;
        int64_t n_expert;

        // the bank of its role in the pool of the layer (shared by all the layers of the pool)
        ggml_tensor * bank      = nullptr;
        uint8_t *     bank_data = nullptr;
    };

    struct layer {
        int il;
        int64_t n_expert = 0;
        bool    window   = false; // the cache has an overflow window
        std::vector<int> tensors;

        int     pool      = -1;
        int64_t n_uniform = 0; // cache slots with the uniform split
        int64_t n_min     = 0; // the least cache slots it keeps (n_uniform without a shared pool)
        int64_t n_cached  = 0; // cache slots it has

        std::vector<int32_t> slot_of;   // [n_expert] cache slot of the expert in the pool, -1 if not cached

        // replacement policy
        std::vector<uint64_t> stamp;    // [n_expert] last use
        std::vector<uint8_t>  prot;     // [n_expert] in the protected segment (slru)
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

        // statistics of the batches computed on the CPU, since the start
        std::vector<uint64_t> last_use;  // [n_expert] use_tick of the last batch that selected the expert, 0 = never
        uint64_t              use_tick = 0;
        std::vector<uint64_t> sd_hist;   // [n_expert + 1] LRU stack distance of the selected experts, [n_expert] = first use
        std::vector<uint64_t> picks;     // [n_expert] tokens that selected the expert
        uint64_t st_access = 0;
        uint64_t st_hits   = 0;
        uint64_t st_window = 0;

        bool streamed() const { return n_uniform < n_expert; }
    };

    // the eviction order of a cache slot: slru: the probation segment first; lfu: the least used first; then
    // the least recently used first
    struct okey {
        uint8_t  prot;
        double   score;
        uint64_t stamp;
        int32_t  slot;

        bool operator<(const okey & o) const {
            if (prot  != o.prot)  { return prot  < o.prot;  }
            if (score != o.score) { return score < o.score; }
            if (stamp != o.stamp) { return stamp < o.stamp; }
            return slot < o.slot;
        }
    };

    // the cache slots of layers with the same experts layout (all of a layer without a shared pool)
    // a bank per role (gate, up, down...): [ne0, ne1, n_slots + n_window], an expert per slot, the window after
    // the slots is an alias of the staging buffer of the role, for the experts of a batch computed on the CPU that
    // do not fit in the cache
    struct pool {
        std::vector<int> layers;
        int64_t n_slots = 0;
        bool    window  = false;

        std::vector<int32_t> slot_layer;  // [n_slots] the layer of the expert in the slot
        std::vector<int32_t> slot_expert; // [n_slots] the expert in the slot

        std::set<std::pair<uint64_t, int32_t>> prot; // slru: the protected segment, by (stamp, slot)

        // shared pools: the slots in eviction order, rebuilt when it is not valid
        std::set<okey>    order;
        std::vector<okey> key_of; // [n_slots] the key of the slot in order
        bool              order_valid = false;

        std::vector<ggml_tensor *> bank;      // [n_roles]
        std::vector<uint8_t *>     bank_data; // [n_roles]
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
    std::vector<pool>   pools;
    std::vector<int32_t> cand; // victim candidates
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

    // the layers with the same experts layout share their cache slots, keeping at least that fraction of their
    // uniform share each (--moe-stream-pool), < 0 = per-layer caches
    double pool_min = -1.0;

    // the statistics of the batches computed on the CPU are logged (and written to the stats file) every that many
    // tokens (LLAMA_MOE_STREAM_INTERVAL)
    int64_t stats_interval = 1024;

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
    double  cpu_t_policy = 0.0; // in the cache bookkeeping of ensure, without waiting for its reads
    // how predictable the misses were: among the 2 / 8 most used experts that were not cached, or most recently evicted
    int64_t cpu_pred_used[2]    = { 0, 0 };
    int64_t cpu_pred_evicted[2] = { 0, 0 };

    // stats file (LLAMA_MOE_STREAM_STATS: a path, or 1 for next to the model; disabled by default)
    std::string stats_path;
    double      alloc_min = 0.5; // minimum slots of a layer in the projected split, relative to the uniform split (LLAMA_MOE_STREAM_ALLOC_MIN)
    double      t_start_all = 0.0;

    struct totals {
        int64_t cpu_tokens = 0, cpu_batches = 0, cpu_access = 0, cpu_hits = 0, cpu_window = 0;
        size_t  cpu_bytes  = 0;
        double  cpu_t_read = 0.0;
        double  cpu_t_policy = 0.0;
        int64_t cpu_pred_used[2] = { 0, 0 }, cpu_pred_evicted[2] = { 0, 0 };
        int64_t gpu_graphs = 0, gpu_tokens = 0, gpu_misses = 0, gpu_wasted = 0;
        int64_t gpu_pf_need = 0, gpu_pf_done = 0, gpu_pf_queue = 0;
        size_t  gpu_read = 0, gpu_up_slot = 0, gpu_up_res = 0;
        double  gpu_time = 0.0, gpu_stalled = 0.0;
    };
    totals tot;  // since the start
    totals last; // of the last interval
    totals agg;  // gpu: the ubatches of the current prompt, logged together

    bool    graph_cached = false; // the current graph was computed from the cache
    int64_t state_tokens = 0;     // tokens computed on the CPU since the state was saved
    bool    state_dirty  = false;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const llama_moe_stream_params & params) {
#ifndef __linux__
        GGML_UNUSED(model);
        GGML_UNUSED(backends);
        GGML_UNUSED(params);
        LLAMA_LOG_WARN("moe_stream: MoE streaming is only supported on Linux, disabled\n");
#else
        policy        = params.policy;
        pool_min      = params.pool;
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
            LLAMA_LOG_WARN("moe_stream: no GPU to stream the experts to, MoE streaming disabled\n");
            return;
        }

        if (!collect(model)) {
            return;
        }

        if (pool_min >= 0.0 && (!model.hdmd.rot.empty() || !model.hdmd.inv.empty())) {
            LLAMA_LOG_WARN("moe_stream: the model has folded Hadamard rotations, the expert cache is not pooled\n");
            pool_min = -1.0;
        }
        make_pools();

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
        if (const char * env = getenv("LLAMA_MOE_STREAM_ALLOC_MIN")) {
            alloc_min = std::clamp(atof(env), 0.0, 1.0);
        }
        if (const char * env = getenv("LLAMA_MOE_STREAM_INTERVAL")) {
            stats_interval = std::max<int64_t>(1, atoll(env));
        }
        t_start_all = t_now();
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
        if (const char * env = getenv("LLAMA_MOE_STREAM_STATS")) {
            const std::string v = env;
            if (v == "1") {
                stats_path = files[0].path + ".moe-stream-stats.json";
            } else if (!v.empty() && v != "0" && v != "none") {
                stats_path = v;
            }
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
        if (is_active) {
            log_gpu();
        }
        if (is_active && state_dirty) {
            save_state();
        }
        if (is_active && tot.cpu_tokens + tot.gpu_tokens > 0) {
            write_stats();
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
            LLAMA_LOG_WARN("moe_stream: failed to open %s: %s\n", path.c_str(), strerror(errno));
            return -1;
        }
        f.fd_direct = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (f.fd_direct < 0) {
            LLAMA_LOG_WARN("moe_stream: O_DIRECT is not supported for %s (%s), using buffered reads\n", path.c_str(), strerror(errno));
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
            LLAMA_LOG_WARN("moe_stream: %d expert tensors in host memory are not memory mapped from the model files, they are not streamed (is mmap disabled?)\n",
                    n_unmapped);
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
                LLAMA_LOG_WARN("moe_stream: layer %d has expert tensors with different expert counts, not streamed\n", il);
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
            LLAMA_LOG_WARN("moe_stream: no memory mapped expert tensors in host memory, MoE streaming disabled\n");
            return false;
        }

        op_ctxs.resize(layers.size());
        for (size_t li = 0; li < layers.size(); ++li) {
            op_ctxs[li] = { this, (int) li };
        }

        return true;
    }

    // group the layers in pools: without a shared pool, one per layer, otherwise by experts layout (the type and
    // the shape of each of their tensors)
    void make_pools() {
        pools.clear();
        std::map<std::string, int> by_key;
        for (size_t li = 0; li < layers.size(); ++li) {
            auto & l = layers[li];
            if (pool_min < 0.0) {
                pool p;
                p.layers = { (int) li };
                l.pool = (int) pools.size();
                for (int idx : l.tensors) {
                    tensors[idx].role = tensors[idx].index;
                }
                pools.push_back(std::move(p));
                continue;
            }

            // the roles by name without the layer prefix
            std::vector<std::pair<std::string, int>> roles;
            for (int idx : l.tensors) {
                const char * name = tensors[idx].t->name;
                const char * p    = strchr(name, '.');
                p = p ? strchr(p + 1, '.') : nullptr;
                roles.emplace_back(p ? p + 1 : name, idx);
            }
            std::sort(roles.begin(), roles.end());

            std::string key;
            for (size_t r = 0; r < roles.size(); ++r) {
                const auto * t = tensors[roles[r].second].t;
                char buf[256];
                snprintf(buf, sizeof(buf), "%s:%d:%" PRId64 "x%" PRId64 "x%" PRId64 ":%zu;", roles[r].first.c_str(), (int) t->type,
                        t->ne[0], t->ne[1], t->ne[2], (size_t) t->nb[2]);
                key += buf;
                tensors[roles[r].second].role = (int) r;
            }

            auto it = by_key.find(key);
            if (it == by_key.end()) {
                it = by_key.emplace(key, (int) pools.size()).first;
                pools.emplace_back();
            }
            l.pool = it->second;
            pools[it->second].layers.push_back((int) li);
        }
    }

    bool shared(const layer & l) const {
        return pools[l.pool].layers.size() > 1;
    }

    // the least cache slots of a layer with n slots in the uniform split
    int64_t min_slots(const layer & l, int64_t n) const {
        if (!shared(l)) {
            return n;
        }
        return std::min<int64_t>(n, (int64_t) std::ceil(pool_min*(double) n - 1e-9));
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
            const int64_t n_cached = min_slots(l, (int64_t) (frac * (double) l.n_expert));
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
            if (arc > 0) {
                LLAMA_LOG_INFO("moe_stream: auto size: %.2f GiB available (%.2f GiB of it ZFS ARC), %.2f GiB reserved\n",
                        avail/GiB, arc/GiB, reserve/GiB);
            } else {
                LLAMA_LOG_INFO("moe_stream: auto size: %.2f GiB available, %.2f GiB reserved\n", avail/GiB, reserve/GiB);
            }
        } else {
            frac = std::min(1.0, (double) resident / (double) total);
        }

        for (auto & p : pools) {
            p.n_slots = 0;
        }

        size_t total_res = 0;
        for (auto & l : layers) {
            l.n_uniform = (int64_t) (frac * (double) l.n_expert);
            l.n_min     = min_slots(l, l.n_uniform);
            pools[l.pool].n_slots += l.n_uniform;

            l.slot_of  .assign(l.n_expert, -1);
            l.stamp    .assign(l.n_expert, 0);
            l.prot     .assign(l.n_expert, 0);
            l.score    .assign(l.n_expert, 0.0);
            l.mark     .assign(l.n_expert, 0);
            l.counts   .assign(l.n_expert, 0);
            l.evicted_at.assign(l.n_expert, 0);
            l.last_use  .assign(l.n_expert, 0);
            l.sd_hist   .assign(l.n_expert + 1, 0);
            l.picks     .assign(l.n_expert, 0);

            for (int idx : l.tensors) {
                auto & ti = tensors[idx];
                total_res += (size_t) l.n_uniform * ti.expert_size;
                slot_size = std::max(slot_size, max_slot_bytes(ti, l.n_min));
            }
        }
        for (auto & p : pools) {
            p.slot_layer .assign(p.n_slots, -1);
            p.slot_expert.assign(p.n_slots, -1);
        }

        LLAMA_LOG_INFO("moe_stream: %zu layers, %.2f GiB of experts, %.2f GiB cached (%" PRId64 "/%" PRId64 " per layer), %s policy\n",
                layers.size(), total/GiB, total_res/GiB, layers[0].n_uniform, layers[0].n_expert, policy_name(policy));

        if (pool_min >= 0.0) {
            int n_shared = 0;
            int n_own    = 0;
            std::string sizes;
            for (const auto & p : pools) {
                if (p.layers.size() > 1) {
                    n_shared++;
                    sizes += (sizes.empty() ? "" : " + ") + std::to_string(p.layers.size());
                } else {
                    n_own++;
                }
            }
            std::string own = n_own == 0 ? "" : n_own == 1 ? ", 1 layer with a unique layout keeps its own cache" :
                ", " + std::to_string(n_own) + " layers with a unique layout keep their own cache";
            LLAMA_LOG_INFO("moe_stream: pooled cache: %d pool%s (%s layers), each layer keeps >= %.0f%% of its share%s\n",
                    n_shared, n_shared == 1 ? "" : "s", sizes.empty() ? "0" : sizes.c_str(), 100.0*pool_min, own.c_str());
        }
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
            LLAMA_LOG_INFO("moe_stream: all the experts are cached, nothing to stream\n");
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
                    LLAMA_LOG_WARN("moe_stream: failed to lock the staging buffers: %s\n", strerror(errno));
                }
                pin = register_pinned(staging, staging_size_);
            }
        }
        if (!base) {
            LLAMA_LOG_WARN("moe_stream: memfd failed (%s), batches computed on the CPU are limited to the cache size\n", strerror(errno));
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
            LLAMA_LOG_ERROR("moe_stream: failed to allocate %.2f GiB for the staging buffers, MoE streaming disabled\n", staging_size_/GiB);
            return false;
        }

        for (size_t s = 0; s < n_slots; ++s) {
            slots[s].data  = base + s*slot_size;
            slots[s].event = ggml_backend_event_new(target_dev);
        }

        if (!pin) {
            LLAMA_LOG_WARN("moe_stream: the staging buffers are not pinned, uploads from them will be slower\n");
        }
        LLAMA_LOG_INFO("moe_stream: staging: %zu x %.2f GiB %s buffers, %zu IO threads\n",
                n_slots, slot_size/GiB, pin ? "pinned" : "unpinned", workers.size());

        return true;
    }

    // the cache: for each role of each pool its slots, ending at a page boundary, followed by its overflow window if any
    bool alloc_cache() {
        struct region {
            int    pool;
            int    role;
            int    t;      // a tensor of the role
            size_t bank;   // bytes of the cache slots
            size_t padded; // rounded up to pages
            bool   window;
        };
        std::vector<region> regions;
        size_t total = 0;
        size_t total_banks = 0;
        int    n_windows = 0;

        for (size_t pi = 0; pi < pools.size(); ++pi) {
            auto & p = pools[pi];
            const auto & l0 = layers[p.layers[0]];
            // a window per role, each over its own staging buffer
            p.window = memfd >= 0 && l0.streamed() && l0.tensors.size() <= slots.size();
            p.bank     .assign(l0.tensors.size(), nullptr);
            p.bank_data.assign(l0.tensors.size(), nullptr);
            for (int li : p.layers) {
                layers[li].window = p.window;
            }
            for (int idx : l0.tensors) {
                const auto & ti = tensors[idx];
                region r;
                r.pool   = (int) pi;
                r.role   = ti.role;
                r.t      = idx;
                r.bank   = p.n_slots*ti.expert_size;
                r.padded = align_up(r.bank, page_size);
                r.window = p.window;
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
            LLAMA_LOG_ERROR("moe_stream: failed to allocate %.2f GiB for the expert cache: %s\n", total_banks/GiB, strerror(errno));
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

            if (r.padded > 0) {
                // locking also allocates all the pages now
                if (!lock_failed && mlock(cur, r.padded) != 0) {
                    lock_failed = true;
                    LLAMA_LOG_WARN("moe_stream: failed to lock the expert cache (%s), it may be swapped out\n", strerror(errno));
                    LLAMA_LOG_WARN("moe_stream: raise the limit with 'ulimit -l unlimited' (or LimitMEMLOCK=infinity for a systemd service)\n");
                }
                if (want_pin && register_pinned(cur, r.padded)) {
                    n_pinned += r.padded;
                }
                bank_ranges.emplace_back(cur, r.padded);
            }
            starts[i] = cur + r.padded - r.bank;
            cur += r.padded;

            if (r.window) {
                void * p = mmap(cur, slot_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, memfd, (off_t) (r.role*slot_size));
                if (p == MAP_FAILED) {
                    LLAMA_LOG_ERROR("moe_stream: failed to map an overflow window: %s\n", strerror(errno));
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
            const auto & ti = tensors[r.t];
            auto       & p  = pools[r.pool];

            const int64_t n_window = r.window ? (int64_t) (slot_size / ti.expert_size) : 0;
            for (int li : p.layers) {
                GGML_ASSERT(!r.window || n_window >= layers[li].n_expert - layers[li].n_min);
            }

            ggml_tensor * bank = ggml_new_tensor_3d(bank_ctx, ti.t->type, ti.t->ne[0], ti.t->ne[1], p.n_slots + n_window);
            GGML_ASSERT((size_t) bank->nb[2] == ti.expert_size);
            if (p.layers.size() > 1) {
                const char * name = ti.t->name;
                const char * dot  = strchr(name, '.');
                dot = dot ? strchr(dot + 1, '.') : nullptr;
                ggml_format_name(bank, "pool%d.%s.cache", r.pool, dot ? dot + 1 : name);
            } else {
                ggml_format_name(bank, "%s.cache", ti.t->name);
            }
            ggml_backend_tensor_alloc(bank_buf, bank, starts[i]);
            p.bank     [r.role] = bank;
            p.bank_data[r.role] = starts[i];

            for (int li : p.layers) {
                for (int idx : layers[li].tensors) {
                    if (tensors[idx].role == r.role) {
                        tensors[idx].bank      = bank;
                        tensors[idx].bank_data = starts[i];
                    }
                }
            }
        }

        pinned = want_pin && n_pinned > 0;

        LLAMA_LOG_INFO("moe_stream: cache: %.2f GiB %s in %.1f s, %d overflow windows\n",
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

    // the slots of each layer of a pool, from the number of experts it wants: the same proportions, each layer
    // keeping at least n_min, the pool full
    void split_pool(const pool & p, const std::vector<int64_t> & want, std::vector<int64_t> & n) const {
        const size_t m = p.layers.size();
        n.assign(m, 0);
        if (m == 1) {
            n[0] = p.n_slots;
            return;
        }
        double sum = 0.0;
        for (size_t i = 0; i < m; ++i) {
            sum += (double) want[i];
        }
        const double scale = sum > 0.0 ? (double) p.n_slots / sum : 0.0;
        std::vector<double> ideal(m);
        int64_t used = 0;
        for (size_t i = 0; i < m; ++i) {
            const auto & l = layers[p.layers[i]];
            ideal[i] = sum > 0.0 ? (double) want[i]*scale : (double) l.n_uniform;
            n[i] = std::clamp<int64_t>((int64_t) ideal[i], l.n_min, l.n_expert);
            used += n[i];
        }
        while (used != p.n_slots) {
            // the layer furthest from its ideal size in the direction needed
            int64_t best = -1;
            double  bd   = 0.0;
            for (size_t i = 0; i < m; ++i) {
                const auto & l = layers[p.layers[i]];
                const double d = used < p.n_slots ? ideal[i] - (double) n[i] : (double) n[i] - ideal[i];
                const bool ok = used < p.n_slots ? n[i] < l.n_expert : n[i] > l.n_min;
                if (ok && (best < 0 || d > bd)) {
                    best = (int64_t) i;
                    bd   = d;
                }
            }
            GGML_ASSERT(best >= 0);
            const int64_t step = used < p.n_slots ? 1 : -1;
            n[best] += step;
            used    += step;
        }
    }

    // fill the cache with the saved experts, or the first ones
    bool fill_cache() {
        std::vector<std::vector<int32_t>> saved(layers.size());
        std::vector<uint8_t> has_saved(layers.size(), 0);
        const int n_loaded = load_state(saved, has_saved);

        const double t_start = t_now();
        const size_t read_start = bytes_read;

        wait_group wg;
        std::vector<int64_t> want;
        std::vector<int64_t> n;
        for (auto & p : pools) {
            // a run of slots per layer, sized like the saved lists
            want.clear();
            for (int li : p.layers) {
                want.push_back(has_saved[li] ? (int64_t) saved[li].size() : layers[li].n_uniform);
            }
            split_pool(p, want, n);

            int64_t base = 0;
            for (size_t i = 0; i < p.layers.size(); ++i) {
                const int li = p.layers[i];
                auto & l = layers[li];
                l.n_cached = n[i];

                // the saved experts first, most recently used first, then the first experts that are not cached yet
                int64_t s = base;
                for (size_t j = 0; j < saved[li].size() && s < base + n[i]; ++j) {
                    const int32_t e = saved[li][j];
                    l.slot_of[e]     = (int32_t) s;
                    p.slot_layer [s] = li;
                    p.slot_expert[s] = e;
                    l.stamp[e]       = (uint64_t) (saved[li].size() - j);
                    s++;
                }
                int64_t e = 0;
                for (; s < base + n[i]; ++s) {
                    while (l.slot_of[e] >= 0) {
                        e++;
                    }
                    l.slot_of[e]     = (int32_t) s;
                    p.slot_layer [s] = li;
                    p.slot_expert[s] = (int32_t) e;
                }

                // read runs of consecutive experts in consecutive slots together
                for (int64_t s0 = base; s0 < base + n[i]; ) {
                    int64_t k = 1;
                    while (s0 + k < base + n[i] && p.slot_expert[s0 + k] == p.slot_expert[s0] + k) {
                        k++;
                    }
                    read_experts(l, p.slot_expert[s0], s0, k, wg);
                    s0 += k;
                }
                base += n[i];
            }
        }
        if (!wait(wg)) {
            LLAMA_LOG_ERROR("moe_stream: failed to read the experts into the cache\n");
            return false;
        }

        const double t    = std::max(t_now() - t_start, 1e-6);
        const size_t read = bytes_read - read_start;
        LLAMA_LOG_INFO("moe_stream: cache: filled %.2f GiB in %.1f s (%.2f GiB/s), %s\n",
                read/GiB, t, read/GiB/t, n_loaded > 0 ? ("experts of " + std::to_string(n_loaded) + " layers from " + state_path).c_str() : "first experts");

        return true;
    }

    // the largest batch of a layer computed on the CPU
    int64_t cpu_limit(const layer & l) const {
        if (!l.streamed() || l.window) {
            return cpu_max;
        }
        // without a window, all the experts of a batch must fit in the cache
        return std::min<int64_t>(cpu_max, n_expert_used > 0 ? l.n_min / n_expert_used : 0);
    }

    // log how batches are computed
    void check_offload() {
        int64_t lim = INT64_MAX;
        for (const auto & l : layers) {
            lim = std::min(lim, cpu_limit(l));
        }
        LLAMA_LOG_INFO("moe_stream: batches of up to %" PRId64 " tokens on the CPU from the cache%s, larger ones on %s\n",
                lim, overlap_mode ? " (reads overlapped)" : "", ggml_backend_name(target_backend));
        if (lim < cpu_max) {
            LLAMA_LOG_WARN("moe_stream: without overflow windows, batches on the CPU are limited to %" PRId64 " tokens\n", lim);
        }
        if (!stats_path.empty()) {
            LLAMA_LOG_INFO("moe_stream: statistics written to %s every %" PRId64 " tokens\n", stats_path.c_str(), stats_interval);
        }
    }

    //
    // saved state: the cached experts of each layer, most recently used first, and the usage scores
    //

    // the saved experts of each layer, most recently used first
    int load_state(std::vector<std::vector<int32_t>> & saved, std::vector<uint8_t> & has_saved) {
        if (state_path.empty()) {
            return 0;
        }
        std::ifstream in(state_path);
        if (!in) {
            return 0;
        }

        std::string line;
        if (!std::getline(in, line) || (line != "moe-stream-state 1" && line != "moe-stream-state 2")) {
            LLAMA_LOG_WARN("moe_stream: %s is not a state file, ignored\n", state_path.c_str());
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
            const int li = layer_of_il[il];
            auto & l = layers[li];
            if (n_expert != l.n_expert) {
                continue;
            }
            std::vector<uint8_t> seen(l.n_expert, 0);
            auto & list = saved[li];
            list.clear();
            for (int64_t i = 0; i < count; ++i) {
                int64_t e = -1;
                if (!(ss >> e) || e < 0 || e >= l.n_expert || seen[e]) {
                    continue;
                }
                seen[e] = 1;
                list.push_back((int32_t) e);
            }
            has_saved[li] = 1;
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
                    LLAMA_LOG_WARN("moe_stream: cannot write %s, the cached experts are not saved\n", tmp.c_str());
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
                const auto & p = pools[l.pool];
                for (int64_t s = 0; s < p.n_slots; ++s) {
                    if (p.slot_layer[s] == &l - layers.data()) {
                        es.push_back(p.slot_expert[s]);
                    }
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
            LLAMA_LOG_WARN("moe_stream: failed to save %s: %s\n", state_path.c_str(), strerror(errno));
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
            for (auto & p : pools) {
                p.order_valid = false;
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

    // the order of eviction of the cache slots of a pool: lru: the least recently used first; slru: the same, the
    // probation segment first; lfu: the least used first
    bool evict_before(const pool & p, int32_t a, int32_t b) const {
        const auto & la = layers[p.slot_layer[a]];
        const auto & lb = layers[p.slot_layer[b]];
        const int32_t ea = p.slot_expert[a];
        const int32_t eb = p.slot_expert[b];
        if (policy == LLAMA_MOE_STREAM_POLICY_SLRU && la.prot[ea] != lb.prot[eb]) {
            return la.prot[ea] < lb.prot[eb];
        }
        if (policy == LLAMA_MOE_STREAM_POLICY_LFU && la.score[ea] != lb.score[eb]) {
            return la.score[ea] < lb.score[eb];
        }
        if (la.stamp[ea] != lb.stamp[eb]) {
            return la.stamp[ea] < lb.stamp[eb];
        }
        return a < b;
    }

    // a slot of the pool of layer li that a missing expert of li can take: an expert of li that the batch did not
    // select, or, if steal, an expert of another layer that has more than its least slots
    bool can_evict(const pool & p, int li, int32_t s, bool steal) const {
        const int lo = p.slot_layer[s];
        if (lo == li) {
            return layers[li].mark[p.slot_expert[s]] != layers[li].mark_gen;
        }
        return steal && layers[lo].n_cached > layers[lo].n_min;
    }

    okey make_key(const pool & p, int32_t s) const {
        const auto & l = layers[p.slot_layer[s]];
        const int32_t e = p.slot_expert[s];
        okey k;
        k.prot  = policy == LLAMA_MOE_STREAM_POLICY_SLRU ? l.prot[e] : 0;
        k.score = policy == LLAMA_MOE_STREAM_POLICY_LFU  ? l.score[e] : 0.0;
        k.stamp = l.stamp[e];
        k.slot  = s;
        return k;
    }

    // the policy state of the expert in slot s changed
    void rekey(pool & p, int32_t s) {
        if (!p.order_valid) {
            return;
        }
        p.order.erase(p.key_of[s]);
        p.key_of[s] = make_key(p, s);
        p.order.insert(p.key_of[s]);
    }

    void build_order(pool & p) {
        p.order.clear();
        p.key_of.resize(p.n_slots);
        for (int32_t s = 0; s < (int32_t) p.n_slots; ++s) {
            p.key_of[s] = make_key(p, s);
            p.order.insert(p.key_of[s]);
        }
        p.order_valid = true;
    }

    // the victim candidates of layer li among its own experts, the first n_want in eviction order at the front
    void collect_victims(const pool & p, int li, size_t n_want) {
        cand.clear();
        const auto & l = layers[li];
        for (int64_t e = 0; e < l.n_expert; ++e) {
            if (l.slot_of[e] >= 0 && l.mark[e] != l.mark_gen) {
                cand.push_back(l.slot_of[e]);
            }
        }
        auto less = [&](int32_t a, int32_t b) { return evict_before(p, a, b); };
        if (cand.size() > n_want) {
            std::nth_element(cand.begin(), cand.begin() + n_want, cand.end(), less);
            cand.resize(n_want);
        }
        std::sort(cand.begin(), cand.end(), less);
    }

    // a second use promotes to the protected half of the pool (slru), its least recently used expert goes back to probation
    void promote(pool & p, layer & l, int64_t e) {
        if (policy != LLAMA_MOE_STREAM_POLICY_SLRU || l.prot[e]) {
            return;
        }
        const int32_t s = l.slot_of[e];
        l.prot[e] = 1;
        rekey(p, s);
        p.prot.emplace(l.stamp[e], s);
        if ((int64_t) p.prot.size() > p.n_slots/2) {
            auto it = p.prot.begin();
            if (it->second == s) {
                ++it;
            }
            if (it != p.prot.end()) {
                const int32_t d = it->second;
                layers[p.slot_layer[d]].prot[p.slot_expert[d]] = 0;
                p.prot.erase(it);
                rekey(p, d);
            }
        }
    }

    // make the experts selected by a batch computed on the CPU available and write their slots:
    // the missing experts replace cached ones, or go to the overflow window when the cache is full of experts
    // selected by the batch (or, with lfu, when they are used less than the expert they would replace)
    void ensure(int li, const ggml_tensor * ids, ggml_tensor * out) {
        auto & l = layers[li];
        auto & p = pools[l.pool];

        GGML_ASSERT(pend_li < 0 && "the previous layer did not wait for its reads");

        const double t_ensure = t_now();
        double t_waited = 0.0;

        // the experts of the other layers of the pool can be evicted when no layer of the graph is streamed:
        // then nothing uploads from the cache while it changes. the eviction order of the pool is kept for that
        const bool steal = shared(l) && order.empty();
        if (steal && !p.order_valid) {
            build_order(p);
        }
        if (!steal) {
            p.order_valid = false;
        }

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
            promote(p, l, e);
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
            if (l.prot[e]) {
                p.prot.erase({ l.stamp[e], l.slot_of[e] });
                p.prot.emplace(clock, l.slot_of[e]);
            }
            l.stamp[e]  = clock;
            l.score[e] += cnt[e]*w;
            if (l.slot_of[e] >= 0) {
                rekey(p, l.slot_of[e]);
            }
        }

        // LRU stack distances: how many other experts were selected since the last batch that selected e
        // (all the selections of a batch are at the same distance)
        for (int32_t e : needed) {
            int64_t d = l.n_expert;
            if (l.last_use[e] > 0) {
                d = 0;
                for (int64_t x = 0; x < l.n_expert; ++x) {
                    d += l.last_use[x] > l.last_use[e];
                }
            }
            l.sd_hist[d]++;
            l.picks[e] += cnt[e];
        }
        l.use_tick++;
        for (int32_t e : needed) {
            l.last_use[e] = l.use_tick;
        }
        l.st_access += needed.size();
        l.st_hits   += needed.size() - missing.size();

        overflow.clear();
        const size_t read_start = bytes_read;
        wait_group & wg = pend_wg;
        if (!missing.empty()) {
            const double t0 = t_now();

            // the most used first, so that with lfu they are the ones admitted
            std::stable_sort(missing.begin(), missing.end(), [&](int32_t a, int32_t b) { return l.score[a] > l.score[b]; });

            bool changed = false;
            size_t ci = 0;
            auto it = p.order.begin(); // steal: the next candidate in the eviction order of the pool
            if (!steal) {
                collect_victims(p, li, missing.size());
            }
            for (int32_t e : missing) {
                // the next candidate that can be evicted: the candidates skipped stay unavailable during the
                // batch (selected by it, or of layers at their least slots, which only lose more)
                int32_t s = -1;
                if (steal) {
                    while (it != p.order.end() && !can_evict(p, li, it->slot, true)) {
                        ++it;
                    }
                    s = it != p.order.end() ? it->slot : -1;
                } else {
                    while (ci < cand.size() && !can_evict(p, li, cand[ci], false)) {
                        ci++;
                    }
                    s = ci < cand.size() ? cand[ci] : -1;
                }
                if (s < 0) {
                    overflow.push_back(e);
                    continue;
                }
                const int lo = p.slot_layer[s];
                auto & lv = layers[lo];
                const int32_t v = p.slot_expert[s];
                if (l.window && policy == LLAMA_MOE_STREAM_POLICY_LFU && l.score[e] <= lv.score[v]) {
                    overflow.push_back(e);
                    continue;
                }
                if (steal) {
                    ++it; // the key of s changes below, the new one is after the current position or not a candidate
                } else {
                    ci++;
                }

                if (lv.prot[v]) {
                    lv.prot[v] = 0;
                    p.prot.erase({ lv.stamp[v], s });
                }
                lv.slot_of[v]    = -1;
                lv.evicted_at[v] = ++lv.n_evict;
                if (lo != li) {
                    lv.n_cached--;
                    l.n_cached++;
                    lv.dirty = true;
                }
                l.slot_of[e]     = s;
                p.slot_layer [s] = li;
                p.slot_expert[s] = e;
                changed          = true;
                rekey(p, s);

                read_experts(l, e, s, 1, wg);
            }

            if (!overflow.empty()) {
                GGML_ASSERT(l.window && "more experts selected than cache slots");

                // the windows are the staging buffers of the tensors: wait for what uses them and drop their content
                for (int idx : l.tensors) {
                    const int s = tensors[idx].role;
                    release(s);
                    reset(s, -1);
                }
                for (size_t j = 0; j < overflow.size(); ++j) {
                    const int32_t e = overflow[j];
                    ov_index[e] = (int32_t) j;
                    for (int idx : l.tensors) {
                        const auto & ti = tensors[idx];
                        uint8_t * dst = ti.bank_data + (p.n_slots + j)*ti.expert_size;
                        submit_bounce(ti.file, ti.file_offs + e*ti.expert_size, ti.expert_size, dst, wg);
                    }
                }
                cpu_window += overflow.size();
                l.st_window += overflow.size();
            }

            if (changed) {
                l.dirty     = true;
                state_dirty = true;
            }

            if (!overlap_mode) {
                const double tw = t_now();
                if (!wait(wg)) {
                    GGML_ABORT("moe_stream: failed to read the experts of layer %d", l.il);
                }
                cpu_bytes  += bytes_read - read_start;
                cpu_t_read += t_now() - t0;
                t_waited    = t_now() - tw;
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
                const int32_t slot = l.slot_of[e] >= 0 ? l.slot_of[e] : (int32_t) p.n_slots + ov_index[e];
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

        cpu_t_policy += t_now() - t_ensure - t_waited;
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

    bool cpu_path(int il, int64_t n) const {
        if (il < 0 || il >= (int) layer_of_il.size() || layer_of_il[il] < 0) {
            return false;
        }
        const auto & l = layers[layer_of_il[il]];
        if (n > cpu_limit(l) || n == 0) {
            return false;
        }
        for (int idx : l.tensors) {
            if (!tensors[idx].bank) {
                return false;
            }
        }
        return true;
    }

    ggml_tensor * build_slots(ggml_context * ctx, ggml_tensor * selected, int il) const {
        const int64_t k = selected->ne[0];
        const int64_t n = selected->ne[1];
        if (!cpu_path(il, n)) {
            return nullptr;
        }
        const int li = layer_of_il[il];

        ggml_tensor * args[] = { selected };
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
                LLAMA_LOG_ERROR("moe_stream: read of %zu bytes at %zu of %s failed: %s\n", j.len, j.offs, f.path.c_str(), strerror(errno));
                return false;
            }
            if (!warned_dio_fallback.exchange(true)) {
                LLAMA_LOG_WARN("moe_stream: O_DIRECT read failed for %s, falling back to buffered reads\n", f.path.c_str());
            }
        }

        const ssize_t r = pread_full(f.fd_buffered, j.dst, need, j.offs);
        if (r < 0 || (size_t) r != need) {
            LLAMA_LOG_ERROR("moe_stream: read of %zu bytes at %zu of %s failed: %s\n", need, j.offs, f.path.c_str(),
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
            if (policy == LLAMA_MOE_STREAM_POLICY_LFU) {
                pools[l.pool].order_valid = false;
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
                GGML_ABORT("moe_stream: failed to read %s", ti.t->name);
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
            log_gpu();
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

    static void add_cpu(totals & a, const totals & b) {
        a.cpu_tokens  += b.cpu_tokens;
        a.cpu_batches += b.cpu_batches;
        a.cpu_access  += b.cpu_access;
        a.cpu_hits    += b.cpu_hits;
        a.cpu_window  += b.cpu_window;
        a.cpu_bytes   += b.cpu_bytes;
        a.cpu_t_read  += b.cpu_t_read;
        a.cpu_t_policy += b.cpu_t_policy;
        for (int i = 0; i < 2; ++i) {
            a.cpu_pred_used[i]    += b.cpu_pred_used[i];
            a.cpu_pred_evicted[i] += b.cpu_pred_evicted[i];
        }
    }

    // the bytes of an expert of a layer, all its tensors
    size_t expert_bytes(const layer & l) const {
        size_t b = 0;
        for (int idx : l.tensors) {
            b += tensors[idx].expert_size;
        }
        return b;
    }

    // LRU hit rate of the batches computed on the CPU from the stack distances, with the current uniform split of
    // the cache, and with the split of the same bytes that maximizes the hits, each layer keeping at least alloc_min
    // of its current slots (greedy by hits per byte of the next slot)
    void project(double & cur, double & best, std::vector<int64_t> * split) const {
        uint64_t access = 0, hits_cur = 0;
        size_t   budget = 0;
        std::vector<int64_t> c(layers.size());
        for (size_t li = 0; li < layers.size(); ++li) {
            const auto & l = layers[li];
            for (int64_t d = 0; d < l.n_uniform; ++d) {
                hits_cur += l.sd_hist[d];
            }
            for (uint64_t v : l.sd_hist) {
                access += v;
            }
            budget += l.n_uniform*expert_bytes(l);
            c[li] = std::min<int64_t>(l.n_expert, (int64_t) std::ceil(alloc_min*l.n_uniform));
            budget -= c[li]*expert_bytes(l) <= budget ? c[li]*expert_bytes(l) : budget;
        }

        uint64_t hits_best = 0;
        for (size_t li = 0; li < layers.size(); ++li) {
            for (int64_t d = 0; d < c[li]; ++d) {
                hits_best += layers[li].sd_hist[d];
            }
        }
        for (;;) {
            int64_t bl = -1;
            double  bv = -1.0;
            for (size_t li = 0; li < layers.size(); ++li) {
                const auto & l = layers[li];
                const size_t eb = expert_bytes(l);
                if (c[li] >= l.n_expert || eb > budget) {
                    continue;
                }
                // the hits of the next few slots, so that a gap in the histogram does not stop a layer
                double gain = 0.0;
                for (int64_t d = c[li]; d < std::min<int64_t>(l.n_expert, c[li] + 4); ++d) {
                    gain += l.sd_hist[d];
                }
                gain /= 4.0*eb;
                if (gain > bv) {
                    bv = gain;
                    bl = (int64_t) li;
                }
            }
            if (bl < 0) {
                break;
            }
            hits_best += layers[bl].sd_hist[c[bl]];
            c[bl]++;
            budget -= expert_bytes(layers[bl]);
        }

        cur  = access > 0 ? (double) hits_cur /access : 0.0;
        best = access > 0 ? (double) hits_best/access : 0.0;
        if (split) {
            *split = c;
        }
    }

    void write_stats() {
        if (stats_path.empty()) {
            return;
        }

        double cur = 0.0, best = 0.0;
        std::vector<int64_t> split;
        project(cur, best, &split);

        auto cpu_json = [](std::ostream & o, const totals & t) {
            const double n_miss = (double) std::max<int64_t>(t.cpu_access - t.cpu_hits, 1);
            o << "{\"tokens\": " << t.cpu_tokens << ", \"batches\": " << t.cpu_batches
              << ", \"selected\": " << t.cpu_access << ", \"cached\": " << t.cpu_hits << ", \"overflow_window\": " << t.cpu_window
              << ", \"read_gib\": " << t.cpu_bytes/GiB << ", \"read_or_wait_s\": " << t.cpu_t_read << ", \"policy_s\": " << t.cpu_t_policy
              << ", \"misses_in_top2_used\": " << t.cpu_pred_used[0]/n_miss << ", \"misses_in_top8_used\": " << t.cpu_pred_used[1]/n_miss
              << ", \"misses_in_last2_evicted\": " << t.cpu_pred_evicted[0]/n_miss << ", \"misses_in_last8_evicted\": " << t.cpu_pred_evicted[1]/n_miss << "}";
        };

        // the interval in progress is part of the totals
        totals cur_iv;
        cur_iv.cpu_tokens  = cpu_tokens;
        cur_iv.cpu_batches = cpu_batches;
        cur_iv.cpu_access  = cpu_access;
        cur_iv.cpu_hits    = cpu_hits;
        cur_iv.cpu_window  = cpu_window;
        cur_iv.cpu_bytes   = cpu_bytes;
        cur_iv.cpu_t_read  = cpu_t_read;
        cur_iv.cpu_t_policy = cpu_t_policy;
        for (int i = 0; i < 2; ++i) {
            cur_iv.cpu_pred_used[i]    = cpu_pred_used[i];
            cur_iv.cpu_pred_evicted[i] = cpu_pred_evicted[i];
        }
        totals all = tot;
        if (cpu_tokens > 0) {
            add_cpu(all, cur_iv);
        }

        const std::string tmp = stats_path + ".tmp";
        {
            std::ofstream o(tmp);
            if (!o) {
                static bool warned = false;
                if (!warned) {
                    LLAMA_LOG_WARN("moe_stream: cannot write %s\n", tmp.c_str());
                    warned = true;
                }
                return;
            }
            o.precision(6);
            o << "{\n";
            o << "  \"version\": 1,\n";
            o << "  \"uptime_s\": " << t_now() - t_start_all << ",\n";
            o << "  \"policy\": \"" << policy_name(policy) << "\", \"halflife\": " << halflife << ", \"overlap\": " << (overlap_mode ? "true" : "false")
              << ", \"cpu_max\": " << cpu_max << ", \"staging_buffers\": " << slots.size() << ", \"io_threads\": " << workers.size()
              << ", \"pool_min\": " << pool_min << ", \"pools\": " << pools.size() << ",\n";
            o << "  \"cpu_total\": ";
            cpu_json(o, all);
            o << ",\n  \"cpu_last_interval\": ";
            cpu_json(o, last.cpu_tokens > 0 ? last : cur_iv);
            o << ",\n";
            o << "  \"gpu_total\": {\"ubatches\": " << all.gpu_graphs << ", \"tokens\": " << tot.gpu_tokens << ", \"time_s\": " << tot.gpu_time
              << ", \"stalled_s\": " << tot.gpu_stalled << ", \"read_gib\": " << tot.gpu_read/GiB
              << ", \"uploaded_staged_gib\": " << tot.gpu_up_slot/GiB << ", \"uploaded_cached_gib\": " << tot.gpu_up_res/GiB
              << ", \"selected_not_cached\": " << tot.gpu_pf_need << ", \"read_ahead_done\": " << tot.gpu_pf_done << ", \"read_ahead_in_progress\": " << tot.gpu_pf_queue
              << ", \"misses\": " << tot.gpu_misses << ", \"wasted\": " << tot.gpu_wasted << "},\n";
            o << "  \"projection\": {\"lru_hit_rate_uniform\": " << cur << ", \"lru_hit_rate_best\": " << best << ", \"min_slots_fraction\": " << alloc_min << "},\n";
            o << "  \"layers\": [\n";
            for (size_t li = 0; li < layers.size(); ++li) {
                const auto & l = layers[li];
                o << "    {\"il\": " << l.il << ", \"n_expert\": " << l.n_expert << ", \"pool\": " << l.pool << ", \"slots\": " << l.n_cached
                  << ", \"uniform_slots\": " << l.n_uniform << ", \"min_slots\": " << l.n_min
                  << ", \"best_slots\": " << split[li] << ", \"expert_bytes\": " << expert_bytes(l)
                  << ", \"selected\": " << l.st_access << ", \"cached\": " << l.st_hits << ", \"overflow_window\": " << l.st_window;
                o << ",\n     \"stack_distance\": [";
                for (size_t d = 0; d < l.sd_hist.size(); ++d) {
                    o << (d ? "," : "") << l.sd_hist[d];
                }
                o << "],\n     \"picks\": [";
                for (size_t x = 0; x < l.picks.size(); ++x) {
                    o << (x ? "," : "") << l.picks[x];
                }
                o << "]}" << (li + 1 < layers.size() ? "," : "") << "\n";
            }
            o << "  ]\n}\n";
        }
        if (rename(tmp.c_str(), stats_path.c_str()) != 0) {
            LLAMA_LOG_WARN("moe_stream: failed to write %s: %s\n", stats_path.c_str(), strerror(errno));
        }
    }

    void end_graph() {
        if (graph_cached && !last_streamed) {
            cpu_tokens   += n_tokens;
            cpu_batches  += 1;
            state_tokens += n_tokens;
            if (cpu_tokens >= stats_interval && cpu_access > 0) {
                last.cpu_tokens  = cpu_tokens;
                last.cpu_batches = cpu_batches;
                last.cpu_access  = cpu_access;
                last.cpu_hits    = cpu_hits;
                last.cpu_window  = cpu_window;
                last.cpu_bytes   = cpu_bytes;
                last.cpu_t_read  = cpu_t_read;
                last.cpu_t_policy = cpu_t_policy;
                for (int i = 0; i < 2; ++i) {
                    last.cpu_pred_used[i]    = cpu_pred_used[i];
                    last.cpu_pred_evicted[i] = cpu_pred_evicted[i];
                }
                add_cpu(tot, last);

                std::string spread;
                if (pool_min >= 0.0) {
                    // how far the layers of the shared pools moved from their uniform share
                    int64_t lo = INT64_MAX, hi = 0;
                    for (const auto & l : layers) {
                        if (shared(l) && l.n_uniform > 0) {
                            lo = std::min<int64_t>(lo, 100*l.n_cached/l.n_uniform);
                            hi = std::max<int64_t>(hi, 100*l.n_cached/l.n_uniform);
                        }
                    }
                    if (lo <= hi) {
                        spread = ", layers at " + std::to_string(lo) + "-" + std::to_string(hi) + "% of their share";
                    }
                }
                LLAMA_LOG_INFO("moe_stream: decode: %" PRId64 " tokens, %.1f%% hits, %.2f misses/token, read %.2f GiB, %s %.1f s, cache policy %.2f ms/token%s\n",
                        cpu_tokens, 100.0*cpu_hits/cpu_access, (double) (cpu_access - cpu_hits)/std::max<int64_t>(cpu_tokens, 1),
                        cpu_bytes/GiB, overlap_mode ? "waited" : "read time", cpu_t_read, 1e3*cpu_t_policy/std::max<int64_t>(cpu_tokens, 1), spread.c_str());

                cpu_pred_used[0] = cpu_pred_used[1] = 0;
                cpu_pred_evicted[0] = cpu_pred_evicted[1] = 0;
                cpu_tokens  = 0;
                cpu_batches = 0;
                cpu_access  = 0;
                cpu_hits    = 0;
                cpu_window  = 0;
                cpu_bytes   = 0;
                cpu_t_read  = 0.0;
                cpu_t_policy = 0.0;

                write_stats();
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

        for (totals * a : { &tot, &agg }) {
            a->gpu_graphs++;
            a->gpu_tokens   += n_tokens;
            a->gpu_time     += t;
            a->gpu_stalled  += t_stalled;
            a->gpu_read     += read;
            a->gpu_up_slot  += bytes_up_slot;
            a->gpu_up_res   += bytes_up_res;
            a->gpu_pf_need  += pf_need;
            a->gpu_pf_done  += pf_done;
            a->gpu_pf_queue += pf_queue;
            a->gpu_misses   += n_misses;
            a->gpu_wasted   += n_wasted;
        }

        t_first = -1.0;

        // one line per prompt: after its last ubatch
        if (!expect_more) {
            log_gpu();
        }
    }

    void log_gpu() {
        if (agg.gpu_graphs == 0) {
            return;
        }
        const double t = std::max(agg.gpu_time, 1e-6);
        LLAMA_LOG_INFO("moe_stream: prompt: %" PRId64 " tokens in %" PRId64 " ubatches, %.1f s (%.1f t/s), read %.2f GiB (%.2f GiB/s), stalled %.1f s,"
                " read ahead %.0f%% (+%.0f%% in progress)%s\n",
                agg.gpu_tokens, agg.gpu_graphs, t, agg.gpu_tokens/t, agg.gpu_read/GiB, agg.gpu_read/GiB/t, agg.gpu_stalled,
                100.0*agg.gpu_pf_done/std::max<int64_t>(agg.gpu_pf_need, 1), 100.0*agg.gpu_pf_queue/std::max<int64_t>(agg.gpu_pf_need, 1),
                agg.gpu_misses + agg.gpu_wasted > 0 ? (", " + std::to_string(agg.gpu_misses) + " misses, " + std::to_string(agg.gpu_wasted) + " wasted").c_str() : "");
        agg = totals();
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

bool llama_moe_stream::cpu_path(int il, int64_t n_tokens) const {
#ifdef __linux__
    return pimpl->is_active && pimpl->cpu_path(il, n_tokens);
#else
    GGML_UNUSED(il);
    GGML_UNUSED(n_tokens);
    return false;
#endif
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
