#include "server-monitor.h"

#include "common.h"
#include "log.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cpp-httplib/httplib.h>

#include "monitor-page.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static constexpr int TICK_MS = 150;

// printf into a string
static void append(std::string & s, const char * fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n > 0) {
        s.append(buf, std::min<size_t>(n, sizeof(buf) - 1));
    }
}

static uint32_t parse_topics(const std::string & q) {
    uint32_t t = 0;
    if (q.find("timeline") != std::string::npos) {
        t |= LLAMA_MONITOR_TOPIC_TIMELINE;
    }
    if (q.find("experts") != std::string::npos) {
        t |= LLAMA_MONITOR_TOPIC_EXPERTS;
    }
    return t;
}

static std::string json_str(const std::string & s) {
    std::string r = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') {
            r += '\\';
            r += c;
        } else if ((unsigned char) c >= 0x20) {
            r += c;
        }
    }
    return r + "\"";
}

struct server_monitor::impl {
    common_params params;
    std::function<llama_context *()> get_ctx;

    httplib::Server svr;
    std::thread     t_listen, t_tick;
    std::atomic<bool> stopping{false};
    int64_t base_us = 0;

    // one message of the feed, shared by the clients, each takes the parts of its topics
    struct tick {
        uint64_t    id;
        std::string head; // always
        std::string tl;   // timeline
        std::string ex;   // experts
    };
    std::mutex              mtx;
    std::condition_variable cv;
    std::deque<tick>        ticks;
    uint64_t                next_id = 1;
    std::map<int, uint32_t> clients; // id -> topics
    int                     next_client = 0;
    std::string             last_where;
    // the timeline is recorded while the monitor runs (one event per ubatch), a page that connects gets this history
    struct hist_entry { uint64_t id; std::string tl; int32_t n; };
    std::deque<hist_entry>  hist;
    size_t                  hist_n = 0;
    static constexpr size_t HIST_MAX = 200000; // events

    uint32_t topics_all() const {
        uint32_t t = 0;
        for (const auto & [id, c] : clients) {
            t |= c;
        }
        return t;
    }

    double ms(int64_t us) const { return (us - base_us)/1000.0; }

    std::string layout_json() {
        llama_context * ctx = get_ctx();
        std::string r = "{";
        if (!ctx) {
            return r + "\"error\":\"no model\"}";
        }
        const llama_model * model = llama_get_model(ctx);
        char buf[256] = { 0 };
        llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf));
        append(r, "\"arch\":%s,", json_str(buf).c_str());
        buf[0] = 0;
        llama_model_desc(model, buf, sizeof(buf));
        append(r, "\"model\":%s,", json_str(buf).c_str());
        llama_monitor_layout lo = {};
        llama_monitor_get_layout(ctx, &lo);
        append(r, "\"n_layer\":%d,\"n_expert\":%d,\"n_expert_used\":%d,\"n_moe_layer\":%d,", lo.n_layer, lo.n_expert, lo.n_expert_used, lo.n_moe_layer);
        append(r, "\"n_ctx\":%u,\"n_batch\":%u,\"n_ubatch\":%u,", llama_n_ctx(ctx), llama_n_batch(ctx), llama_n_ubatch(ctx));
        r += "\"devices\":[";
        bool first = true;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                continue;
            }
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            append(r, "%s{\"name\":%s,\"total\":%zu}", first ? "" : ",", json_str(ggml_backend_dev_name(dev)).c_str(), total);
            first = false;
        }
        r += "],\"layers\":[";
        std::vector<llama_monitor_layer> layers(lo.n_layer);
        llama_monitor_snapshot(ctx, layers.data(), nullptr, nullptr);
        for (int il = 0; il < lo.n_layer; ++il) {
            const auto & l = layers[il];
            append(r, "%s[%d,%d,%d,%d]", il ? "," : "", l.moe, l.n_expert, l.n_cached, l.n_hot);
        }
        return r + "]}";
    }

    std::string mem_json() {
        std::string r = "[";
        bool first = true;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                continue;
            }
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            append(r, "%s[%s,%zu,%zu]", first ? "" : ",", json_str(ggml_backend_dev_name(dev)).c_str(), total - free, total);
            first = false;
        }
        return r + "]";
    }

    // the broadcaster: while a client is connected, drain the events and snapshot the counters, once per tick
    void tick_loop() {
        std::vector<llama_monitor_event> evs(2048);
        std::vector<uint32_t> counts, prev;
        std::vector<uint8_t>  where;
        std::vector<llama_monitor_layer> layers;
        uint32_t cur_topics = 0;
        int n_tick = 0;
        while (!stopping) {
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait_for(lk, std::chrono::milliseconds(TICK_MS), [&]{ return stopping.load(); });
                if (stopping) {
                    break;
                }
                const uint32_t t = topics_all() | LLAMA_MONITOR_TOPIC_TIMELINE;
                llama_context * ctx = get_ctx();
                if (ctx && t != cur_topics) {
                    llama_monitor_set_topics(ctx, t);
                    cur_topics = t;
                    prev.clear();
                    n_tick = 0;
                }
            }
            llama_context * ctx = get_ctx();
            if (!ctx) {
                continue;
            }
            tick tk;
            append(tk.head, "\"now\":%.1f", ms(llama_monitor_now_us()));
            if (n_tick % 7 == 0) {
                append(tk.head, ",\"mem\":%s", mem_json().c_str());
            }
            if (cur_topics & LLAMA_MONITOR_TOPIC_TIMELINE) {
                uint64_t dropped = 0;
                int32_t  n = llama_monitor_drain(ctx, evs.data(), (int32_t) evs.size(), &dropped);
                if (dropped) {
                    append(tk.head, ",\"dropped\":%llu", (unsigned long long) dropped);
                }
                tk.tl = "[";
                for (int32_t i = 0; i < n; ++i) {
                    const auto & e = evs[i];
                    append(tk.tl, "%s[%u,%u,%u,%.3f,%.3f,%d,%d,%d,%d,%lld,%lld,%lld]", i ? "," : "", e.type, e.kind, e.seq, ms(e.t_us), e.dur_us/1000.0,
                            e.n[0], e.n[1], e.n[2], e.n[3], (long long) e.x[0], (long long) e.x[1], (long long) e.x[2]);
                }
                tk.tl += "]";
            }
            if (cur_topics & LLAMA_MONITOR_TOPIC_EXPERTS) {
                llama_monitor_layout lo = {};
                llama_monitor_get_layout(ctx, &lo);
                const size_t ne = lo.n_expert;
                layers.resize(lo.n_layer);
                counts.resize((size_t) lo.n_layer*ne);
                const bool with_where = n_tick % 7 == 0;
                where.resize((size_t) lo.n_layer*ne);
                llama_monitor_snapshot(ctx, layers.data(), counts.data(), with_where ? where.data() : nullptr);
                tk.ex = "{\"l\":[";
                for (int il = 0; il < lo.n_layer; ++il) {
                    const auto & l = layers[il];
                    append(tk.ex, "%s[%d,%llu,%llu,%llu,%llu,%d,%d,%d]", il ? "," : "", l.distinct, (unsigned long long) l.hit_gpu, (unsigned long long) l.hit_ram,
                            (unsigned long long) l.miss, (unsigned long long) l.routes, l.n_cached, l.n_hot, l.n_expert);
                }
                tk.ex += "],\"h\":{";
                bool first = true;
                if (prev.size() == counts.size()) {
                    for (int il = 0; il < lo.n_layer; ++il) {
                        bool any = false;
                        for (size_t e = 0; e < ne && !any; ++e) {
                            any = counts[il*ne + e] != prev[il*ne + e];
                        }
                        if (!any) {
                            continue;
                        }
                        append(tk.ex, "%s\"%d\":[", first ? "" : ",", il);
                        first = false;
                        for (size_t e = 0; e < ne; ++e) {
                            append(tk.ex, "%s%u", e ? "," : "", counts[il*ne + e] - prev[il*ne + e]);
                        }
                        tk.ex += "]";
                    }
                }
                tk.ex += "}";
                if (with_where) {
                    std::string w = ",\"w\":[";
                    for (int il = 0; il < lo.n_layer; ++il) {
                        w += il ? ",\"" : "\"";
                        for (size_t e = 0; e < ne; ++e) {
                            w += (char) ('0' + where[il*ne + e]);
                        }
                        w += "\"";
                    }
                    w += "]";
                    tk.ex += w;
                    std::lock_guard<std::mutex> lk(mtx);
                    last_where = w;
                }
                tk.ex += "}";
                prev = counts;
            }
            n_tick++;
            {
                std::lock_guard<std::mutex> lk(mtx);
                tk.id = next_id++;
                if (tk.tl.size() > 2) {
                    const int32_t n = (int32_t) std::count(tk.tl.begin(), tk.tl.end(), ']') - 1;
                    hist.push_back({ tk.id, tk.tl, n });
                    hist_n += n;
                    while (hist_n > HIST_MAX && !hist.empty()) {
                        hist_n -= hist.front().n;
                        hist.pop_front();
                    }
                }
                ticks.push_back(std::move(tk));
                while (ticks.size() > 64) {
                    ticks.pop_front();
                }
            }
            cv.notify_all();
        }
        llama_context * ctx = get_ctx();
        if (ctx) {
            llama_monitor_set_topics(ctx, 0);
        }
    }

    // one SSE message of the ticks after last for a client of the topics
    bool message(uint64_t & last, uint32_t topics, bool & first, std::string & out) {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait_for(lk, std::chrono::milliseconds(1000), [&]{ return stopping || (!ticks.empty() && ticks.back().id > last); });
        if (stopping) {
            return false;
        }
        out.clear();
        // a new page: the recorded timeline before the ticks it gets below
        if (last == 0 && (topics & LLAMA_MONITOR_TOPIC_TIMELINE) && !hist.empty()) {
            const uint64_t id0 = ticks.empty() ? next_id : ticks.front().id;
            std::string ev;
            for (const auto & h : hist) {
                if (h.id >= id0) {
                    break;
                }
                ev += ev.empty() ? "" : ",";
                ev.append(h.tl, 1, h.tl.size() - 2);
            }
            if (!ev.empty()) {
                out += "data: {\"history\":true,\"ev\":[" + ev + "]}\n\n";
            }
        }
        for (const auto & tk : ticks) {
            if (tk.id <= last) {
                continue;
            }
            last = tk.id;
            out += "data: {" + tk.head;
            if ((topics & LLAMA_MONITOR_TOPIC_TIMELINE) && !tk.tl.empty()) {
                out += ",\"ev\":" + tk.tl;
            }
            if ((topics & LLAMA_MONITOR_TOPIC_EXPERTS) && !tk.ex.empty()) {
                std::string ex = tk.ex;
                if (first && !last_where.empty() && ex.find("\"w\":") == std::string::npos) {
                    ex.insert(ex.size() - 1, last_where);
                }
                out += ",\"ex\":" + ex;
                first = false;
            }
            out += "}\n\n";
        }
        if (out.empty()) {
            out = ": ping\n\n";
        }
        return true;
    }

    void routes() {
        svr.new_task_queue = [] { return new httplib::ThreadPool(8); };
        svr.Get("/", [](const httplib::Request &, httplib::Response & res) {
            res.set_content((const char *) monitor_page, sizeof(monitor_page), "text/html; charset=utf-8");
        });
        svr.Get("/api/layout", [this](const httplib::Request &, httplib::Response & res) {
            res.set_content(layout_json(), "application/json");
        });
        svr.Get("/api/stream", [this](const httplib::Request & req, httplib::Response & res) {
            uint32_t topics = req.has_param("topics") ? parse_topics(req.get_param_value("topics")) : LLAMA_MONITOR_TOPIC_TIMELINE;
            int id;
            {
                std::lock_guard<std::mutex> lk(mtx);
                id = next_client++;
                clients[id] = topics;
            }
            cv.notify_all();
            struct state_t { uint64_t last = 0; bool first = true; };
            auto st = std::make_shared<state_t>();
            res.set_header("Cache-Control", "no-cache");
            res.set_chunked_content_provider("text/event-stream",
                [this, st, topics](size_t, httplib::DataSink & sink) {
                    std::string out;
                    if (!message(st->last, topics, st->first, out)) {
                        return false;
                    }
                    return sink.write(out.data(), out.size());
                },
                [this, id](bool) {
                    std::lock_guard<std::mutex> lk(mtx);
                    clients.erase(id);
                    cv.notify_all();
                });
        });
    }
};

server_monitor::server_monitor(const common_params & params, std::function<llama_context *()> get_ctx) : pimpl(new impl) {
    pimpl->params  = params;
    pimpl->get_ctx = std::move(get_ctx);
}

server_monitor::~server_monitor() {
    stop();
}

bool server_monitor::start() {
    auto & p = *pimpl;
    p.base_us = llama_monitor_now_us();
    p.routes();
    if (!p.svr.bind_to_port(p.params.monitor_host, p.params.monitor_port)) {
        LOG_ERR("monitor: failed to bind %s:%d\n", p.params.monitor_host.c_str(), p.params.monitor_port);
        return false;
    }
    p.t_tick   = std::thread([&p] { p.tick_loop(); });
    p.t_listen = std::thread([&p] { p.svr.listen_after_bind(); });
    LOG_INF("monitor: listening on http://%s:%d/\n", p.params.monitor_host.c_str(), p.params.monitor_port);
    return true;
}

void server_monitor::stop() {
    auto & p = *pimpl;
    if (p.stopping.exchange(true)) {
        return;
    }
    p.cv.notify_all();
    p.svr.stop();
    if (p.t_listen.joinable()) {
        p.t_listen.join();
    }
    if (p.t_tick.joinable()) {
        p.t_tick.join();
    }
}
