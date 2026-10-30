#include "server-cache.h"

#include "hash/hash.h"

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr uint32_t CACHE_VERSION = 1;
constexpr uint64_t CACHE_LIMIT_DEFAULT = 64ull << 30;
constexpr uint64_t CACHE_PROMPT_LIMIT = 64ull << 20;
constexpr uint64_t CACHE_META_LIMIT = 1ull << 20;

// The pool lock only protects metadata. Entry locks cover the slow state I/O.
class cache_lock {
public:
    cache_lock(const fs::path & path, bool wait = true, bool exclusive = true) {
#if defined(_WIN32)
        handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                             FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                             nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("cannot open disk cache lock");
        }
        OVERLAPPED overlap = {};
        if (!LockFileEx(handle, (exclusive ? LOCKFILE_EXCLUSIVE_LOCK : 0) | (wait ? 0 : LOCKFILE_FAIL_IMMEDIATELY),
                        0, 1, 0, &overlap)) {
            const auto error = GetLastError();
            CloseHandle(handle);
            handle = INVALID_HANDLE_VALUE;
            if (!wait && error == ERROR_LOCK_VIOLATION) {
                return;
            }
            throw std::runtime_error("cannot lock disk cache");
        }
#else
        fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open disk cache lock");
        }
        int result;
        do {
            result = flock(fd, (exclusive ? LOCK_EX : LOCK_SH) | (wait ? 0 : LOCK_NB));
        } while (result != 0 && errno == EINTR);
        if (result != 0) {
            const int error = errno;
            close(fd);
            fd = -1;
            if (!wait && (error == EWOULDBLOCK || error == EAGAIN)) {
                return;
            }
            throw std::system_error(error, std::generic_category(), "lock disk cache");
        }
#endif
    }

    ~cache_lock() {
#if defined(_WIN32)
        if (handle != INVALID_HANDLE_VALUE) {
            OVERLAPPED overlap = {};
            UnlockFileEx(handle, 0, 1, 0, &overlap);
            CloseHandle(handle);
        }
#else
        if (fd >= 0) {
            flock(fd, LOCK_UN);
            close(fd);
        }
#endif
    }

    cache_lock(const cache_lock &) = delete;
    cache_lock & operator=(const cache_lock &) = delete;

    explicit operator bool() const {
#if defined(_WIN32)
        return handle != INVALID_HANDLE_VALUE;
#else
        return fd >= 0;
#endif
    }

private:
#if defined(_WIN32)
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
};

bool cache_entry_name(const std::string & name) {
    return name.size() == 64 && name.find_first_not_of("0123456789abcdef") == std::string::npos;
}

bool cache_regular(const fs::path & path) {
    return fs::is_regular_file(fs::symlink_status(path));
}

json cache_read_json(const fs::path & path, uint64_t max_size = CACHE_META_LIMIT) {
    if (!cache_regular(path) || fs::file_size(path) > max_size) {
        throw std::runtime_error("invalid disk cache metadata");
    }
    std::ifstream file(path, std::ios::binary);
    std::string data(fs::file_size(path), '\0');
    if (!file.read(data.data(), data.size())) {
        throw std::runtime_error("cannot read disk cache metadata");
    }
    return json::parse(data);
}

// Flushes a file to stable storage. Snapshots are published by rename, and a superseded
// snapshot is deleted right after, so the new files must be durable before that happens.
void cache_sync_file(const fs::path & path) {
#if defined(_WIN32)
    HANDLE handle = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("cannot open disk cache file for sync");
    }
    const BOOL ok = FlushFileBuffers(handle);
    CloseHandle(handle);
    if (!ok) {
        throw std::runtime_error("cannot sync disk cache file");
    }
#else
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::system_error(errno, std::generic_category(), "open disk cache file for sync");
    }
    int result;
#if defined(__APPLE__)
    // fsync does not flush the drive cache on macOS. F_FULLFSYNC is not supported everywhere.
    result = fcntl(fd, F_FULLFSYNC);
    if (result != 0) {
        result = fsync(fd);
    }
#else
    do {
        result = fsync(fd);
    } while (result != 0 && errno == EINTR);
#endif
    const int error = errno;
    close(fd);
    if (result != 0) {
        throw std::system_error(error, std::generic_category(), "sync disk cache file");
    }
#endif
}

// Makes created, renamed and removed names in a directory durable. Best effort: some
// filesystems reject directory syncs, and Windows has no portable equivalent.
void cache_sync_dir(const fs::path & path) {
#if !defined(_WIN32)
    const int fd = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        return;
    }
    int result;
    do {
        result = fsync(fd);
    } while (result != 0 && errno == EINTR);
    close(fd);
#else
    (void) path;
#endif
}

void cache_write_json(const fs::path & path, const json & data) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << data.dump();
    file.close();
    if (!file) {
        throw std::runtime_error("cannot write disk cache metadata");
    }
    cache_sync_file(path);
}

// The native loader reports every failure the same way. A bad header is permanent corruption,
// so it is checked here; any later restore failure keeps the snapshot.
bool cache_state_header_ok(const fs::path & path) {
    std::ifstream file(path, std::ios::binary);
    uint32_t header[2] = {};
    if (!file.read(reinterpret_cast<char *>(header), sizeof(header))) {
        return false;
    }
    return header[0] == LLAMA_STATE_SEQ_MAGIC && header[1] == LLAMA_STATE_SEQ_VERSION;
}

uint64_t cache_entry_size(const fs::path & path) {
    uint64_t size = 0;
    for (const auto & item : fs::directory_iterator(path)) {
        if (fs::is_regular_file(item.symlink_status())) {
            const auto n = item.file_size();
            if (n > std::numeric_limits<uint64_t>::max() - size) {
                throw std::runtime_error("disk cache size overflow");
            }
            size += n;
        }
    }
    return size;
}

void cache_touch(const fs::path & entry) {
    fs::last_write_time(entry / "meta.json", fs::file_time_type::clock::now());
}

// A single-sequence state save leaves out the SWA-masked cells (llama_kv_cache::state_write), so a restored
// state starts n_swa positions before its last one, not at the pos_min the live slot had when it was saved.
llama_pos cache_restored_pos_min(const server_prompt & prompt, llama_pos pos_min, int32_t n_swa) {
    if (n_swa <= 0 || pos_min < 0) {
        return pos_min;
    }
    return std::max(pos_min, prompt.tokens.pos_next() - n_swa);
}

server_prompt cache_read_prompt(const fs::path & entry, const json & meta, size_t n_ctx, bool has_mtmd, size_t n_checkpoints) {
    const auto n_tokens = meta.at("n_tokens").get<uint64_t>();
    const fs::path path = entry / "prompt.bin";
    if (n_tokens == 0 || n_tokens > n_ctx || !cache_regular(path)) {
        throw std::runtime_error("invalid cached prompt length");
    }
    const uint64_t size = fs::file_size(path);
    if (size == 0 || size > CACHE_PROMPT_LIMIT || size % sizeof(llama_token) != 0) {
        throw std::runtime_error("invalid cached prompt size");
    }
    llama_tokens packed(size / sizeof(llama_token));
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(packed.data()), size)) {
        throw std::runtime_error("cannot read cached prompt");
    }
    const auto identity = meta.at("cache_key").get<std::string>() + ":" + hash_sha256_hex(packed.data(), size);
    if (entry.filename().u8string() != hash_sha256_hex(identity.data(), identity.size())) {
        throw std::runtime_error("cached prompt checksum mismatch");
    }
    server_prompt prompt;
    prompt.tokens = server_tokens::deserialize(packed, has_mtmd);
    if (prompt.tokens.size() != n_tokens) {
        throw std::runtime_error("cached prompt length mismatch");
    }
    for (const auto & item : meta.at("checkpoints")) {
        common_prompt_checkpoint checkpoint;
        checkpoint.n_tokens = item.at(0).get<int64_t>();
        checkpoint.pos_min  = item.at(1).get<llama_pos>();
        checkpoint.pos_max  = item.at(2).get<llama_pos>();
        if (checkpoint.n_tokens < 0 || (uint64_t) checkpoint.n_tokens > n_tokens ||
                checkpoint.pos_min < 0 || checkpoint.pos_max < checkpoint.pos_min ||
                checkpoint.pos_max == std::numeric_limits<llama_pos>::max()) {
            throw std::runtime_error("invalid cached checkpoint");
        }
        prompt.checkpoints.push_back(std::move(checkpoint));
        if (prompt.checkpoints.size() > n_checkpoints) {
            prompt.checkpoints.pop_front();
        }
    }
    return prompt;
}

json cache_file_identity(const std::string & path) {
    if (path.empty()) {
        return nullptr;
    }
    const auto file_path = fs::u8path(path);
    const auto size = fs::file_size(file_path);
    // Content identity only: moving, renaming or copying a file keeps its snapshots, while a
    // different quant or a re-converted model changes the size or the header and tail bytes.
    // Hashing the ends avoids reading all weights at startup.
    std::ifstream file(file_path, std::ios::binary);
    const size_t n = (size_t) std::min<uint64_t>(size, 1 << 20);
    std::vector<char> sample(2 * n);
    if (!file.read(sample.data(), n) || !file.seekg(size - n) || !file.read(sample.data() + n, n)) {
        throw std::runtime_error("cannot fingerprint disk cache model file");
    }
    return {
        {"size", size},
        {"sample", hash_sha256_hex(sample.data(), sample.size())},
    };
}

json cache_model_files(const std::string & path) {
    json result = json::array();
    if (path.empty()) {
        return result;
    }
    // Split GGUF names are standardized. Include every shard, not just the first.
    const std::string name = fs::u8path(path).filename().u8string();
    const std::string marker = "-00001-of-";
    const auto split = name.rfind(marker);
    if (split != std::string::npos && name.size() == split + marker.size() + 5 + 5 &&
            name.substr(name.size() - 5) == ".gguf") {
        const std::string count = name.substr(split + marker.size(), 5);
        if (count.find_first_not_of("0123456789") == std::string::npos) {
            const int n = std::stoi(count);
            if (n > 1) {
                const auto prefix = fs::u8path(path).parent_path() / fs::u8path(name.substr(0, split));
                for (int i = 0; i < n; ++i) {
                    result.push_back(cache_file_identity(prefix.u8string() + string_format("-%05d-of-%05d.gguf", i + 1, n)));
                }
                return result;
            }
        }
    }
    result.push_back(cache_file_identity(path));
    return result;
}

} // namespace

std::string server_cache_model_key(const common_params & params, llama_context * ctx) {
    if (params.model.path.empty()) {
        throw std::runtime_error("disk cache requires a resolved model file path");
    }
    // Only settings that change the stored KV values or the token sequence belong here.
    // Context size, batch sizes and the build do not: the native loader validates the state
    // version and cache layout, n_tokens is checked against n_ctx, and a snapshot that still
    // fails to restore is skipped rather than deleted.
    (void) ctx;
    json data = {
        {"version", CACHE_VERSION}, {"state_version", LLAMA_STATE_SEQ_VERSION},
        {"model", cache_model_files(params.model.path)},
        {"projector", cache_model_files(params.mmproj.path)},
        {"draft", cache_model_files(params.speculative.draft.mparams.path)},
        {"type_k", params.cache_type_k}, {"type_v", params.cache_type_v},
        {"flash_attn", params.flash_attn_type},
        {"draft_type_k", params.speculative.draft.cache_type_k}, {"draft_type_v", params.speculative.draft.cache_type_v},
        {"rope", json::array({params.rope_scaling_type, params.rope_freq_base, params.rope_freq_scale})},
        {"yarn", json::array({params.yarn_ext_factor, params.yarn_attn_factor, params.yarn_beta_fast, params.yarn_beta_slow, params.yarn_orig_ctx})},
        {"swa_full", params.swa_full}, {"attention", params.attention_type},
        {"control_layers", json::array({params.control_vector_layer_start, params.control_vector_layer_end})},
        {"image_tokens", json::array({params.image_min_tokens, params.image_max_tokens})},
    };
    data["speculative"] = json::array();
    for (const auto type : params.speculative.types) {
        data["speculative"].push_back(type);
    }
    for (const auto & adapter : params.lora_adapters) {
        data["adapters"].push_back(cache_file_identity(adapter.path));
    }
    for (const auto & control : params.control_vectors) {
        data["control_vectors"].push_back(json::array({cache_file_identity(control.fname), control.strength}));
    }
    for (const auto & override : params.kv_overrides) {
        if (override.key[0] == 0) {
            continue;
        }
        json value;
        switch (override.tag) {
            case LLAMA_KV_OVERRIDE_TYPE_INT:   value = override.val_i64;  break;
            case LLAMA_KV_OVERRIDE_TYPE_FLOAT: value = override.val_f64;  break;
            case LLAMA_KV_OVERRIDE_TYPE_BOOL:  value = override.val_bool; break;
            case LLAMA_KV_OVERRIDE_TYPE_STR:   value = override.val_str;  break;
        }
        data["overrides"].push_back(json::array({override.key, override.tag, value}));
    }
    const auto text = data.dump();
    return hash_sha256_hex(text.data(), text.size());
}

std::string server_cache_adapter_key(const std::string & model_key, const std::vector<common_adapter_lora_info> & adapters) {
    json scales = json::array();
    for (const auto & adapter : adapters) {
        scales.push_back(adapter.scale);
    }
    const auto text = model_key + scales.dump();
    return hash_sha256_hex(text.data(), text.size());
}

size_t server_cache_reusable(const server_prompt & prompt, const server_tokens & input, llama_pos pos_min, int32_t n_swa) {
    size_t n = prompt.tokens.get_common_prefix(input);
    if (n == 0 || pos_min < 0) {
        return 0;
    }
    const llama_pos pos_next = prompt.tokens.pos_next(n);
    const llama_pos pos_eval = pos_next - (n < input.size() ? 0 : 1);
    const llama_pos threshold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);
    if (pos_min >= threshold) {
        bool found = false;
        for (auto it = prompt.checkpoints.rbegin(); it != prompt.checkpoints.rend(); ++it) {
            if (it->pos_max <= pos_next && (it->pos_min < threshold || it->pos_min == 0)) {
                const auto pos = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                n = std::min(prompt.tokens.size_up_to_pos(pos), (size_t) it->n_tokens);
                found = true;
                break;
            }
        }
        if (!found) {
            return 0;
        }
    }
    // Prompt processing always evaluates at least one token to produce logits.
    return n == input.size() ? n - 1 : n;
}

bool server_cache_should_save(size_t n_input_cached, size_t n_reusable, size_t min_loss) {
    const size_t lost = n_input_cached - std::min(n_input_cached, n_reusable);
    return lost > 0 && lost >= min_loss;
}

server_disk_cache::server_disk_cache(const std::string & path, uint64_t requested_limit) : path(path), limit_bytes(0) {
    fs::create_directories(fs::u8path(path));
    const fs::path root = fs::u8path(path);
    cache_lock lock(root / ".lock");
    const fs::path config = root / "pool.json";
    if (fs::exists(config)) {
        const auto data = cache_read_json(config);
        if (data.at("version").get<uint32_t>() != CACHE_VERSION) {
            throw std::runtime_error("unsupported disk cache pool version");
        }
        limit_bytes = data.at("limit_bytes").get<uint64_t>();
        if (limit_bytes == 0 || (requested_limit != 0 && requested_limit != limit_bytes)) {
            throw std::runtime_error("--cache-disk-size differs from pool.json; stop the pool's servers before changing its limit");
        }
    } else {
        limit_bytes = requested_limit ? requested_limit : CACHE_LIMIT_DEFAULT;
        cache_write_json(root / "pool.json.tmp", {{"version", CACHE_VERSION}, {"limit_bytes", limit_bytes}});
        fs::rename(root / "pool.json.tmp", config);
        cache_sync_dir(root);
    }
    make_room(0);
}

bool server_disk_cache::make_room(uint64_t needed, const std::string & keep) {
    if (needed > limit_bytes) {
        return false;
    }
    struct entry_info {
        fs::path path;
        uint64_t size;
        fs::file_time_type last_used;
    };
    std::vector<entry_info> entries;
    uint64_t total = 0;
    for (const auto & item : fs::directory_iterator(fs::u8path(path))) {
        if (!cache_entry_name(item.path().filename().string()) || !fs::is_directory(item.symlink_status())) {
            continue;
        }
        const auto entry = item.path();
        uint64_t size = cache_entry_size(entry);
        if (!cache_regular(entry / "meta.json")) {
            cache_lock lock(entry / ".lock", false);
            if (lock) {
                fs::remove_all(entry);
                continue;
            }
            // A reservation includes the full write, even while its file is empty.
            const auto reservation = cache_read_json(entry / "reservation.json");
            size = std::max(size, reservation.at("bytes").get<uint64_t>());
        } else if (entry.filename().string() != keep) {
            entries.push_back({entry, size, fs::last_write_time(entry / "meta.json")});
        }
        if (size > std::numeric_limits<uint64_t>::max() - total) {
            throw std::runtime_error("disk cache pool size overflow");
        }
        total += size;
    }
    std::sort(entries.begin(), entries.end(), [](const entry_info & a, const entry_info & b) {
        return a.last_used < b.last_used;
    });
    for (const auto & entry : entries) {
        if (total <= limit_bytes - needed) {
            break;
        }
        cache_lock lock(entry.path / ".lock", false);
        if (!lock) {
            continue;
        }
        fs::remove_all(entry.path);
        total -= entry.size;
    }
    return total <= limit_bytes - needed;
}

bool server_disk_cache::save(const std::string & cache_key, const server_prompt & prompt, llama_pos pos_min, int32_t n_swa, uint64_t reserve_bytes,
                             const std::function<bool(const std::string &)> & write_state) {
    if (cache_key.empty() || prompt.tokens.empty()) {
        return false;
    }
    const fs::path root = fs::u8path(path);
    fs::path entry;
    std::unique_ptr<cache_lock> entry_lock;
    try {
        const auto packed = prompt.tokens.serialize();
        if (packed.size() > CACHE_PROMPT_LIMIT || reserve_bytes > limit_bytes || 2 * packed.size() > limit_bytes - reserve_bytes) {
            SRV_WRN("%s", "disk cache snapshot exceeds pool or metadata limit, skipping\n");
            return false;
        }
        reserve_bytes += 2 * packed.size();
        const auto prompt_hash = hash_sha256_hex(packed.data(), packed.size());
        const auto identity = cache_key + ":" + prompt_hash;
        entry = root / hash_sha256_hex(identity.data(), identity.size());
        {
            cache_lock lock(root / ".lock");
            if (fs::exists(entry)) {
                // An identical snapshot that failed to restore in this process is rewritten.
                if (cache_regular(entry / "meta.json") && !failed.count(entry.filename().string())) {
                    try {
                        const auto meta = cache_read_json(entry / "meta.json");
                        if (cache_regular(entry / "state.bin") && fs::file_size(entry / "state.bin") == meta.at("state_bytes").get<uint64_t>()) {
                            cache_read_prompt(entry, meta, prompt.tokens.size(), prompt.tokens.has_mtmd, 0);
                            cache_touch(entry);
                            return true;
                        }
                    } catch (const std::exception & error) {
                        SRV_WRN("replacing invalid disk cache entry: %s\n", error.what());
                    }
                }
                cache_lock abandoned(entry / ".lock", false);
                if (!abandoned) {
                    return false;
                }
                fs::remove_all(entry);
            }
            if (!make_room(reserve_bytes)) {
                SRV_WRN("%s", "disk cache pool has no room for snapshot, skipping\n");
                return false;
            }
            fs::create_directory(entry);
            entry_lock = std::make_unique<cache_lock>(entry / ".lock");
            cache_write_json(entry / "reservation.json", {{"bytes", reserve_bytes}});
        }
        {
            std::ofstream file(entry / "prompt.bin", std::ios::binary);
            file.write(packed.data(), packed.size());
            file.close();
            if (!file || !write_state((entry / "state.bin").u8string())) {
                throw std::runtime_error("cannot write disk cache snapshot");
            }
            cache_sync_file(entry / "prompt.bin");
            cache_sync_file(entry / "state.bin");
        }
        json checkpoints = json::array();
        for (const auto & checkpoint : prompt.checkpoints) {
            checkpoints.push_back(json::array({checkpoint.n_tokens, checkpoint.pos_min, checkpoint.pos_max}));
        }
        cache_write_json(entry / "meta.json.tmp", {
            {"version", CACHE_VERSION}, {"cache_key", cache_key},
            {"n_tokens", prompt.tokens.size()}, {"pos_min", pos_min}, {"checkpoints", checkpoints},
            {"state_bytes", fs::file_size(entry / "state.bin")},
        });
        // The entry directory and its file names must be durable before it is published.
        cache_sync_dir(entry);
        cache_sync_dir(root);
        {
            cache_lock lock(root / ".lock");
            // No partially written state is visible to readers.
            fs::remove(entry / "reservation.json");
            fs::rename(entry / "meta.json.tmp", entry / "meta.json");
            // Publish durably before any superseded snapshot is removed below.
            cache_sync_dir(entry);
            // A continuation replaces older snapshots only when it can still resume them.
            for (const auto & item : fs::directory_iterator(root)) {
                const auto older = item.path();
                if (older == entry || !cache_entry_name(older.filename().u8string()) ||
                        !fs::is_directory(item.symlink_status()) || !cache_regular(older / "meta.json")) {
                    continue;
                }
                try {
                    const auto meta = cache_read_json(older / "meta.json");
                    if (meta.at("cache_key").get<std::string>() != cache_key ||
                            meta.at("n_tokens").get<uint64_t>() >= prompt.tokens.size()) {
                        continue;
                    }
                    const auto previous = cache_read_prompt(older, meta, prompt.tokens.size(), prompt.tokens.has_mtmd, 0);
                    if (prompt.tokens.get_common_prefix(previous.tokens) != previous.tokens.size() ||
                            server_cache_reusable(prompt, previous.tokens, cache_restored_pos_min(prompt, pos_min, n_swa), n_swa) + 1 < previous.tokens.size()) {
                        continue;
                    }
                    cache_lock older_lock(older / ".lock", false);
                    if (older_lock) {
                        fs::remove_all(older);
                    }
                } catch (const std::exception & error) {
                    SRV_WRN("cannot prune disk cache entry: %s\n", error.what());
                }
            }
            if (!make_room(0, entry.filename().string())) {
                fs::remove_all(entry);
                return false;
            }
        }
        failed.erase(entry.filename().string());
        SRV_INF("disk cache saved %zu tokens (%.3f MiB)\n", prompt.tokens.size(), cache_entry_size(entry) / (1024.0 * 1024.0));
        return true;
    } catch (const std::exception & error) {
        SRV_WRN("disk cache save failed: %s\n", error.what());
        if (entry_lock) {
            try {
                cache_lock lock(root / ".lock");
                fs::remove_all(entry);
            } catch (const std::exception & cleanup_error) {
                SRV_WRN("disk cache cleanup failed: %s\n", cleanup_error.what());
            }
        }
        return false;
    }
}

bool server_disk_cache::load(const std::string & cache_key, const server_tokens & input, size_t n_ctx, bool has_mtmd,
                             int32_t n_swa, size_t n_checkpoints, size_t min_reuse, size_t n_reusable,
                             const std::function<bool(const std::string &)> & read_state) {
    const fs::path root = fs::u8path(path);
    fs::path best;
    std::unique_ptr<cache_lock> entry_lock;
    try {
        {
            cache_lock lock(root / ".lock");
            for (const auto & item : fs::directory_iterator(root)) {
                if (!cache_entry_name(item.path().filename().string()) || !fs::is_directory(item.symlink_status())) {
                    continue;
                }
                const auto entry = item.path();
                if (!cache_regular(entry / "meta.json") || failed.count(entry.filename().string())) {
                    continue;
                }
                try {
                    const auto meta = cache_read_json(entry / "meta.json");
                    if (meta.at("version").get<uint32_t>() != CACHE_VERSION ||
                            meta.at("cache_key").get<std::string>() != cache_key || !cache_regular(entry / "state.bin") ||
                            fs::file_size(entry / "state.bin") != meta.at("state_bytes").get<uint64_t>() ||
                            meta.at("n_tokens").get<uint64_t>() <= n_reusable ||
                            meta.at("n_tokens").get<uint64_t>() < min_reuse) {
                        continue;
                    }
                    if (!cache_state_header_ok(entry / "state.bin")) {
                        SRV_WRN("removing disk cache entry '%s': invalid state header\n", entry.filename().u8string().c_str());
                        cache_lock invalid(entry / ".lock", false);
                        if (invalid) {
                            fs::remove_all(entry);
                        }
                        continue;
                    }
                    const auto prompt = cache_read_prompt(entry, meta, n_ctx, has_mtmd, n_checkpoints);
                    const size_t reuse = server_cache_reusable(prompt, input,
                        cache_restored_pos_min(prompt, meta.at("pos_min").get<llama_pos>(), n_swa), n_swa);
                    if (reuse > n_reusable && reuse >= min_reuse) {
                        best = entry;
                        n_reusable = reuse;
                    }
                } catch (const std::exception & error) {
                    SRV_WRN("ignoring disk cache entry '%s': %s\n", entry.filename().u8string().c_str(), error.what());
                }
            }
            if (best.empty()) {
                return false;
            }
            entry_lock = std::make_unique<cache_lock>(best / ".lock", false, false);
            if (!*entry_lock) {
                return false;
            }
            cache_touch(best);
        }
        if (!read_state((best / "state.bin").u8string())) {
            // The header and size were valid, so this may be a transient or configuration-specific
            // failure (KV space, allocation, a changed cache layout). Keep the snapshot for a later
            // run, but do not re-read it for every request in this process. LRU eviction removes
            // it if it is never usable again.
            SRV_WRN("keeping disk cache entry '%s' after a failed restore; skipping it until restart\n",
                    best.filename().u8string().c_str());
            failed.insert(best.filename().string());
            return false;
        }
        SRV_INF("disk cache restored a prefix of %zu tokens\n", n_reusable);
        return true;
    } catch (const std::exception & error) {
        SRV_WRN("disk cache restore failed: %s\n", error.what());
        return false;
    }
}
