// the compute buffers of a model at a context and ubatch size, without allocating (no_alloc)
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void log_cb(ggml_log_level, const char * text, void *) {
    if (strstr(text, "compute buffer size") || strstr(text, "KV buffer size") || strstr(text, "error") || strstr(text, "failed")) {
        fputs(text, stderr);
    }
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: model n_ctx n_ubatch fa(0/1)\n"); return 1; }
    llama_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.no_alloc = true;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = atoi(argv[2]); cp.n_batch = cp.n_ubatch = atoi(argv[3]); cp.n_seq_max = 1;
    cp.flash_attn_type = atoi(argv[4]) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    fprintf(stderr, "ctx %s\n", ctx ? "ok" : "FAILED");
    const int status = ctx ? 0 : 1;
    if (ctx) llama_free(ctx);
    llama_model_free(model);
    return status;
}
