#include "llama-memory.h"

#include "llama-ext.h"
#include "llama-io.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static thread_local uint32_t g_reserve_n_cells = 0;

uint32_t llama_memory_reserve_n_cells() {
    return g_reserve_n_cells;
}

llama_memory_reserve_scope::llama_memory_reserve_scope(uint32_t n_cells) : prev(g_reserve_n_cells) {
    g_reserve_n_cells = n_cells;
}

llama_memory_reserve_scope::~llama_memory_reserve_scope() {
    g_reserve_n_cells = prev;
}

static thread_local bool g_sparse = false;

bool llama_memory_sparse() {
    return g_sparse;
}

llama_memory_sparse_scope::llama_memory_sparse_scope(bool sparse) : prev(g_sparse) {
    g_sparse = sparse;
}

llama_memory_sparse_scope::~llama_memory_sparse_scope() {
    g_sparse = prev;
}

static ggml_backend_reg_t buffer_type_reg(ggml_backend_buffer_type_t buft) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    return dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
}

llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

bool llama_memory_status_is_fail(llama_memory_status status) {
    switch (status) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}

//
// llama_memory_bufs
//

ggml_backend_buffer_ptr llama_memory_bufs::alloc_buf(ggml_context * ctx, ggml_backend_buffer_type_t buft, bool no_alloc, const char * name) {
    ggml_backend_buffer_ptr buf;
    if (no_alloc) {
        buf.reset(ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0)); // dummy buffer
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            t->buffer = buf.get(); // set dummy buffer so that the backend scheduler won't try to allocate the tensors
        }
    } else {
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)); // real buffer
    }
    if (!buf) {
        throw std::runtime_error(std::string("failed to allocate buffer for ") + name);
    }
    return buf;
}

ggml_backend_buffer_t llama_memory_bufs::emplace(ggml_context_ptr ctx, ggml_backend_buffer_type_t buft, const char * name) {
    ggml_backend_buffer_ptr buf = alloc_buf(ctx.get(), buft, no_alloc, name);

    // initialize the buffer to avoid NaNs in the padding
    ggml_backend_buffer_clear(buf.get(), 0);

    bufs.emplace_back(std::move(ctx), std::move(buf));

    return bufs.back().second.get();
}

ggml_backend_buffer_t llama_memory_bufs::emplace_sparse(ggml_context_ptr ctx, ggml_backend_buffer_type_t buft, const char * name) {
    ggml_backend_reg_t reg = no_alloc ? nullptr : buffer_type_reg(buft);
    auto alloc = reg ? (ggml_backend_cuda_buffer_sparse_alloc_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_buffer_sparse_alloc") : nullptr;
    const size_t size = alloc ? ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft) : 0;
    ggml_backend_buffer_ptr buf(size > 0 ? alloc(buft, size) : nullptr);
    if (!buf) {
        return emplace(std::move(ctx), buft, name);
    }

    ggml_tallocr talloc = ggml_tallocr_new(buf.get());
    for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
        if (t->data != nullptr) {
            continue;
        }
        const ggml_status st = t->view_src == nullptr ? ggml_tallocr_alloc(&talloc, t) : ggml_backend_view_init(t);
        if (st != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("failed to allocate buffer for ") + name);
        }
    }

    sparse.push_back(buf.get());
    bufs.emplace_back(std::move(ctx), std::move(buf));

    return bufs.back().second.get();
}

bool llama_memory_bufs::is_sparse(ggml_backend_buffer_t buf) const {
    return std::find(sparse.begin(), sparse.end(), buf) != sparse.end();
}

bool llama_memory_bufs::commit(ggml_backend_buffer_t buf, const std::vector<size_t> & ranges) {
    ggml_backend_reg_t reg = buffer_type_reg(ggml_backend_buffer_get_type(buf));
    auto fn = reg ? (ggml_backend_cuda_buffer_sparse_commit_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_buffer_sparse_commit") : nullptr;
    return fn && fn(buf, ranges.data(), ranges.size()/2);
}

size_t llama_memory_bufs::committed(ggml_backend_buffer_t buf) {
    ggml_backend_reg_t reg = buffer_type_reg(ggml_backend_buffer_get_type(buf));
    auto fn = reg ? (ggml_backend_cuda_buffer_sparse_committed_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_buffer_sparse_committed") : nullptr;
    return fn ? fn(buf) : ggml_backend_buffer_get_size(buf);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_bufs::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }
    return ret;
}

// collects the buffers of the tensors that a state write visits, without keeping the data
struct llama_memory_buffers_writer : public llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem) {
    if (mem == nullptr) {
        return {};
    }

    llama_memory_buffers_writer writer;

    mem->state_write(writer);

    return std::vector<ggml_backend_buffer_t>(writer.buffers.begin(), writer.buffers.end());
}
