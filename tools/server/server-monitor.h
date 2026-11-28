#pragma once

#include "llama.h"

#include <functional>
#include <memory>

struct common_params;

// the live monitor: a second HTTP listener (--monitor-port) with one page, GET /api/layout and the SSE feed
// GET /api/stream?topics=timeline,experts of the llama_monitor_* events. shapes, counts, sizes and timings only
class server_monitor {
public:
    // get_ctx: the context of the model, which may change while the server runs (nullptr when there is none)
    server_monitor(const common_params & params, std::function<llama_context *()> get_ctx);
    ~server_monitor();

    bool start();
    void stop();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
