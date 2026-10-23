#pragma once

#include "server-task.h"

#include <cstdint>
#include <functional>
#include <string>

// Same rollback rules as prompt processing, without changing the context.
size_t server_cache_reusable(const server_prompt & prompt, const server_tokens & input, llama_pos pos_min, int32_t n_swa);
bool server_cache_should_save(size_t n_cached, size_t n_reusable, size_t min_loss);
std::string server_cache_model_key(const common_params & params, llama_context * ctx);
std::string server_cache_adapter_key(const std::string & model_key, const std::vector<common_adapter_lora_info> & adapters);

class server_disk_cache {
public:
    // A zero limit adopts the pool's limit, or creates a 64 GiB pool.
    server_disk_cache(const std::string & path, uint64_t limit_bytes = 0);

    bool save(const std::string & cache_key, const server_prompt & prompt, llama_pos pos_min, int32_t n_swa, uint64_t reserve_bytes,
              const std::function<bool(const std::string &)> & write_state);

    bool load(const std::string & cache_key, const server_tokens & input, size_t n_ctx, bool has_mtmd,
              int32_t n_swa, size_t n_checkpoints, size_t min_reuse, size_t n_reusable,
              const std::function<bool(const std::string &)> & read_state);

    uint64_t limit() const { return limit_bytes; }

private:
    std::string path;
    uint64_t limit_bytes;

    // Called under the pool lock. Busy readers/writers cannot be evicted.
    bool make_room(uint64_t needed, const std::string & keep = "");
};
