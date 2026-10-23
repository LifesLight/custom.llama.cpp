#include "testing.h"

#include "../tools/server/server-common.h"
#include "../tools/server/server-decision.h"
#include "../tools/server/server-cache.h"

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>
#endif

// unit tests for server internal components, no model required
// NOTE: end-to-end tests live in tools/server/tests
// see skills/code-review/SKILL.md for more

//
// server_tokens (text only, has_mtmd = false)
//

static void test_server_tokens_basic(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3}, false);
    t.assert_equal("size", (size_t) 3, st.size());
    t.assert_true("not empty", !st.empty());
    t.assert_equal("operator[]", 2, st[1]);

    st.push_back(4);
    t.assert_equal("push_back", (size_t) 4, st.size());
    t.assert_true("get_tokens", st.get_tokens() == llama_tokens({1, 2, 3, 4}));

    st.insert({5, 6});
    t.assert_true("insert appends", st.get_tokens() == llama_tokens({1, 2, 3, 4, 5, 6}));

    st.set_token(0, 9);
    t.assert_equal("set_token", 9, st[0]);

    bool thrown = false;
    try {
        st.push_back(LLAMA_TOKEN_NULL);
    } catch (const std::exception &) {
        thrown = true;
    }
    t.assert_true("push_back(LLAMA_TOKEN_NULL) throws", thrown);
    t.assert_equal("size unchanged after throw", (size_t) 6, st.size());

    st.clear();
    t.assert_true("clear", st.empty());
}

static void test_server_tokens_pos(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3, 4, 5}, false);
    t.assert_equal("pos_next()", (llama_pos) 5, st.pos_next());
    t.assert_equal("pos_next(0)", (llama_pos) 0, st.pos_next(0));
    t.assert_equal("pos_next(3)", (llama_pos) 3, st.pos_next(3));

    t.assert_equal("size_up_to_pos(0)", (size_t) 0, st.size_up_to_pos(0));
    t.assert_equal("size_up_to_pos(3)", (size_t) 3, st.size_up_to_pos(3));
    t.assert_equal("size_up_to_pos clamps", (size_t) 5, st.size_up_to_pos(100));
}

static void test_server_tokens_keep_first(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3, 4, 5}, false);
    st.keep_first(5);
    t.assert_equal("keep all", (size_t) 5, st.size());
    st.keep_first(2);
    t.assert_true("keep 2", st.get_tokens() == llama_tokens({1, 2}));
    st.keep_first(0);
    t.assert_true("keep 0", st.empty());
}

static void test_server_tokens_common_prefix(testing & t) {
    server_tokens a(llama_tokens{1, 2, 3, 4}, false);
    server_tokens empty(llama_tokens{}, false);

    t.assert_equal("both empty",  (size_t) 0, empty.get_common_prefix(empty));
    t.assert_equal("one empty",   (size_t) 0, a.get_common_prefix(empty));
    t.assert_equal("identical",   (size_t) 4, a.get_common_prefix(a));
    t.assert_equal("diverge at 2", (size_t) 2, a.get_common_prefix(server_tokens(llama_tokens{1, 2, 9, 4}, false)));
    t.assert_equal("diverge at 0", (size_t) 0, a.get_common_prefix(server_tokens(llama_tokens{9, 2, 3, 4}, false)));
    t.assert_equal("b is prefix",  (size_t) 2, a.get_common_prefix(server_tokens(llama_tokens{1, 2}, false)));
    t.assert_equal("a is prefix",  (size_t) 4, a.get_common_prefix(server_tokens(llama_tokens{1, 2, 3, 4, 5, 6}, false)));
}

static void test_server_tokens_serialize(testing & t) {
    // same packing as slot save/restore in server-context.cpp
    auto pack = [](const std::vector<char> & data) {
        llama_tokens packed(data.size() / sizeof(llama_token));
        memcpy(packed.data(), data.data(), data.size());
        return packed;
    };

    t.test("roundtrip", [&](testing & t) {
        server_tokens st(llama_tokens{10, 20, 30}, false);
        std::vector<char> data = st.serialize();
        t.assert_equal("aligned to llama_token", (size_t) 0, data.size() % sizeof(llama_token));

        server_tokens restored = server_tokens::deserialize(pack(data), false);
        t.assert_true("tokens", restored.get_tokens() == st.get_tokens());
    });

    t.test("roundtrip empty", [&](testing & t) {
        server_tokens st(llama_tokens{}, false);
        server_tokens restored = server_tokens::deserialize(pack(st.serialize()), false);
        t.assert_true("empty", restored.empty());
    });

    t.test("legacy plain token list", [&](testing & t) {
        server_tokens restored = server_tokens::deserialize(llama_tokens{7, 8, 9}, false);
        t.assert_true("tokens", restored.get_tokens() == llama_tokens({7, 8, 9}));
    });

    auto expect_throw = [](testing & t, const std::string & msg, const llama_tokens & packed) {
        bool thrown = false;
        try {
            server_tokens::deserialize(packed, false);
        } catch (const std::exception &) {
            thrown = true;
        }
        t.assert_true(msg, thrown);
    };

    t.test("corrupted", [&](testing & t) {
        llama_tokens good = pack(server_tokens(llama_tokens{10, 20, 30}, false).serialize());

        llama_tokens bad_version = good;
        bad_version[1] = 999;
        expect_throw(t, "bad version", bad_version);

        llama_tokens truncated = good;
        truncated.pop_back();
        expect_throw(t, "truncated", truncated);

        llama_tokens trailing = good;
        trailing.push_back(42);
        expect_throw(t, "trailing data", trailing);

        llama_tokens huge_count = good;
        huge_count[2] = 0x7fffffff;
        expect_throw(t, "huge token count", huge_count);

        expect_throw(t, "marker only", llama_tokens{LLAMA_TOKEN_NULL});
    });
}

static void test_server_tokens_clone(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3}, false);
    server_tokens cl = st.clone();
    cl.push_back(4);
    t.assert_equal("original untouched", (size_t) 3, st.size());
    t.assert_equal("clone grew", (size_t) 4, cl.size());
    t.assert_true("get_text_tokens", cl.get_text_tokens() == llama_tokens({1, 2, 3, 4}));
}

static void test_server_tokens(testing & t) {
    t.test("basic",         test_server_tokens_basic);
    t.test("pos",           test_server_tokens_pos);
    t.test("keep_first",    test_server_tokens_keep_first);
    t.test("common_prefix", test_server_tokens_common_prefix);
    t.test("serialize",     test_server_tokens_serialize);
    t.test("clone",         test_server_tokens_clone);
}

//
// string / json helpers
//

static void test_utf8(testing & t) {
    t.test("validate_utf8", [](testing & t) {
        t.assert_equal("empty",            (size_t) 0, validate_utf8(""));
        t.assert_equal("ascii",            (size_t) 3, validate_utf8("abc"));
        t.assert_equal("full 2-byte",      (size_t) 3, validate_utf8("a\xC3\xA9"));
        t.assert_equal("full 3-byte",      (size_t) 4, validate_utf8("a\xE2\x82\xAC"));
        t.assert_equal("full 4-byte",      (size_t) 5, validate_utf8("a\xF0\x9F\x98\x80"));
        t.assert_equal("cut 2-byte (1/2)", (size_t) 1, validate_utf8("a\xC3"));
        t.assert_equal("cut 3-byte (1/3)", (size_t) 1, validate_utf8("a\xE2"));
        t.assert_equal("cut 3-byte (2/3)", (size_t) 1, validate_utf8("a\xE2\x82"));
        t.assert_equal("cut 4-byte (1/4)", (size_t) 1, validate_utf8("a\xF0"));
        t.assert_equal("cut 4-byte (2/4)", (size_t) 1, validate_utf8("a\xF0\x9F"));
        t.assert_equal("cut 4-byte (3/4)", (size_t) 1, validate_utf8("a\xF0\x9F\x98"));
        t.assert_equal("cut only",         (size_t) 0, validate_utf8("\xF0\x9F\x98"));
        t.assert_equal("full then cut",    (size_t) 4, validate_utf8("\xF0\x9F\x98\x80\xC3"));
    });

    t.test("is_valid_utf8", [](testing & t) {
        t.assert_true("empty",          is_valid_utf8(""));
        t.assert_true("ascii",          is_valid_utf8("hello"));
        t.assert_true("multi-byte",     is_valid_utf8("\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80"));
        t.assert_true("truncated",      !is_valid_utf8("\xE2\x82"));
        t.assert_true("bad cont. byte", !is_valid_utf8("\xC3\x41"));
        t.assert_true("lone cont. byte", !is_valid_utf8("\x80"));
        t.assert_true("invalid lead",   !is_valid_utf8("\xFF"));
    });
}

static void test_sse(testing & t) {
    t.test("oai", [](testing & t) {
        t.assert_equal("single", std::string("data: {\"a\":1}\n\n"), format_oai_sse(json{{"a", 1}}));
        t.assert_equal("array",  std::string("data: {\"a\":1}\n\ndata: {\"b\":2}\n\n"),
                format_oai_sse(json::array({json{{"a", 1}}, json{{"b", 2}}})));
        t.assert_equal("invalid utf8 replaced", std::string("data: \"x\xEF\xBF\xBD\"\n\n"), format_oai_sse(json("x\xC3")));
    });

    t.test("oai_resp", [](testing & t) {
        json ev = {{"event", "response.created"}, {"data", {{"id", "r1"}}}};
        t.assert_equal("single", std::string("event: response.created\ndata: {\"id\":\"r1\"}\n\n"), format_oai_resp_sse(ev));
        t.assert_equal("array",  std::string("event: response.created\ndata: {\"id\":\"r1\"}\n\nevent: response.created\ndata: {\"id\":\"r1\"}\n\n"),
                format_oai_resp_sse(json::array({ev, ev})));
    });

    t.test("anthropic", [](testing & t) {
        json ev = {{"event", "message_start"}, {"data", {{"type", "message_start"}}}};
        t.assert_equal("with event", std::string("event: message_start\ndata: {\"type\":\"message_start\"}\n\n"), format_anthropic_sse(ev));
        t.assert_equal("data only",  std::string("data: {\"type\":\"ping\"}\n\n"), format_anthropic_sse(json{{"type", "ping"}}));
    });
}

static void test_json_helpers(testing & t) {
    t.test("json_is_array_of_numbers", [](testing & t) {
        t.assert_true("ints",         json_is_array_of_numbers(json::array({1, 2, 3})));
        t.assert_true("empty array",  json_is_array_of_numbers(json::array()));
        t.assert_true("float",        !json_is_array_of_numbers(json::array({1, 2.5})));
        t.assert_true("string",       !json_is_array_of_numbers(json::array({1, "a"})));
        t.assert_true("not an array", !json_is_array_of_numbers(json(1)));
    });

    t.test("json_is_array_of_mixed_numbers_strings", [](testing & t) {
        t.assert_true("mixed",        json_is_array_of_mixed_numbers_strings(json::array({"a", 1})));
        t.assert_true("numbers only", !json_is_array_of_mixed_numbers_strings(json::array({1, 2})));
        t.assert_true("strings only", !json_is_array_of_mixed_numbers_strings(json::array({"a", "b"})));
        t.assert_true("empty",        !json_is_array_of_mixed_numbers_strings(json::array()));
    });

    t.test("json_is_array_and_contains_numbers", [](testing & t) {
        t.assert_true("contains",     json_is_array_and_contains_numbers(json::array({"a", 1})));
        t.assert_true("strings only", !json_is_array_and_contains_numbers(json::array({"a"})));
        t.assert_true("not an array", !json_is_array_and_contains_numbers(json(1)));
    });

    t.test("json_get_nested_values", [](testing & t) {
        json js = {{"a", {{"b", {{"c", 1}}}}}, {"x", 2}};
        json res = json_get_nested_values({"a/b/c", "x", "a/b", "a/missing", "x/y"}, js);
        t.assert_equal("nested",  1, res.at("a/b/c").get<int>());
        t.assert_equal("top",     2, res.at("x").get<int>());
        t.assert_true("object",   res.at("a/b") == json{{"c", 1}});
        t.assert_true("missing",  !res.contains("a/missing"));
        t.assert_true("not obj",  !res.contains("x/y"));
        t.assert_equal("count", (size_t) 3, res.size());
    });
}

static void test_lora_helpers(testing & t) {
    // only pointer identity is used, never dereferenced
    auto * p0 = reinterpret_cast<llama_adapter_lora *>(0x10);
    auto * p1 = reinterpret_cast<llama_adapter_lora *>(0x20);

    auto make = [](llama_adapter_lora * ptr, float scale) {
        common_adapter_lora_info info;
        info.scale = scale;
        info.ptr   = ptr;
        return info;
    };

    t.test("parse_lora_request", [](testing & t) {
        auto res = parse_lora_request(json::array({
            json{{"id", 0}, {"scale", 0.5}},
            json{{"id", 2}, {"scale", 1.0}},
            json{{"id", 0}, {"scale", 0.25}}, // last one wins
        }));
        t.assert_equal("count", (size_t) 2, res.size());
        t.assert_equal("id 0", 0.25f, res.at(0));
        t.assert_equal("id 2", 1.0f,  res.at(2));

        auto defaults = parse_lora_request(json::array({json::object()}));
        t.assert_equal("default id/scale", 0.0f, defaults.at(-1));
    });

    t.test("are_lora_equal", [&](testing & t) {
        t.assert_true("both empty",   are_lora_equal({}, {}));
        t.assert_true("same",         are_lora_equal({make(p0, 1.0f)}, {make(p0, 1.0f)}));
        t.assert_true("diff scale",   !are_lora_equal({make(p0, 1.0f)}, {make(p0, 0.5f)}));
        t.assert_true("diff ptr",     !are_lora_equal({make(p0, 1.0f)}, {make(p1, 1.0f)}));
        t.assert_true("diff size",    !are_lora_equal({make(p0, 1.0f)}, {}));
    });

    t.test("lora_get_enabled_ids", [&](testing & t) {
        auto ids = lora_get_enabled_ids({make(p0, 0.0f), make(p1, 0.5f), make(p0, -1.0f), make(p1, 2.0f)});
        t.assert_true("ids", ids == std::vector<size_t>({1, 3}));
    });
}

//
// server_pipe
//

static void test_server_pipe(testing & t) {
    auto stop_now = []() { return true; };

    t.test("drain then eof", [](testing & t) {
        server_pipe<int> p;
        t.assert_true("write 1", p.write(1));
        t.assert_true("write 2", p.write(2));
        p.close_write();

        int v = 0;
        t.assert_true("read 1", p.read(v, nullptr) && v == 1);
        t.assert_true("read 2", p.read(v, nullptr) && v == 2);
        t.assert_true("eof", !p.read(v, nullptr));
    });

    t.test("broken pipe", [](testing & t) {
        server_pipe<int> p;
        p.close_read();
        t.assert_true("write fails", !p.write(1));
    });

    t.test("max_size drops oldest", [](testing & t) {
        server_pipe<int> p;
        p.max_size = 2;
        p.write(1);
        p.write(2);
        p.write(3);
        p.close_write();

        int v = 0;
        t.assert_true("read 2", p.read(v, nullptr) && v == 2);
        t.assert_true("read 3", p.read(v, nullptr) && v == 3);
        t.assert_true("eof", !p.read(v, nullptr));
    });

    t.test("queued data wins over should_stop", [&](testing & t) {
        server_pipe<int> p;
        p.write(1);
        int v = 0;
        t.assert_true("read", p.read(v, stop_now) && v == 1);
    });

    t.test("should_stop closes the pipe", [&](testing & t) {
        server_pipe<int> p;
        int v = 0;
        t.assert_true("read stopped", !p.read(v, stop_now));
        t.assert_true("write fails", !p.write(1));
    });

    t.test("should_stop as deadline keeps the pipe open", [&](testing & t) {
        server_pipe<int> p;
        int v = 0;
        t.assert_true("read stopped", !p.read(v, stop_now, /* close_on_stop */ false));
        t.assert_true("write ok", p.write(1));
        t.assert_true("read again", p.read(v, nullptr) && v == 1);
    });

    t.test("cross thread", [](testing & t) {
        constexpr int n = 1000;
        server_pipe<int> p;
        std::thread producer([&]() {
            for (int i = 0; i < n; i++) {
                p.write(int(i));
            }
            p.close_write();
        });

        int v = 0;
        int count = 0;
        bool in_order = true;
        while (p.read(v, nullptr)) {
            in_order &= v == count;
            count++;
        }
        producer.join();

        t.assert_equal("count", n, count);
        t.assert_true("in order", in_order);
    });
}

//
// server_decision_escape_special_tokens
//

// "<|name|>" with U+00A6 in place of "|"
static std::string escaped(const std::string & name) {
    return "<\xC2\xA6" + name + "\xC2\xA6>";
}

static void test_decision_escape(testing & t) {
    t.test("cases", [](testing & t) {
        const std::vector<std::pair<std::string, std::string>> cases = {
            { "",                 ""                               },
            { "no tokens",        "no tokens"                      },
            { "<|im_start|>user", escaped("im_start") + "user"     },
            { "<|a_1|><|B|>",     escaped("a_1") + escaped("B")    },
            { "<|<|a|>",          "<|" + escaped("a")              },
            { "<|a|>|>",          escaped("a") + "|>"              },
            { "<||>",             "<||>"                           },
            { "<|a b|>",          "<|a b|>"                        },
            { "<|a|b|>",          "<|a|b|>"                        },
            { "<|caf\xC3\xA9|>",  "<|caf\xC3\xA9|>"                },
            { "<|abc",            "<|abc"                          },
        };
        for (const auto & [text, expected] : cases) {
            t.assert_equal(text, expected, server_decision_escape_special_tokens(text));
        }
    });

    // every string of up to 6 chars from "<|>a-" gives the text of the regex
    t.test("same as std::regex", [](testing & t) {
        static const std::regex re_special("<\\|([A-Za-z0-9_]+)\\|>");
        const std::string chars = "<|>a-";
        size_t n_strings = 0;
        size_t n_diff    = 0;
        for (size_t len = 0; len <= 6; len++) {
            size_t n = 1;
            for (size_t k = 0; k < len; k++) {
                n *= chars.size();
            }
            for (size_t idx = 0; idx < n; idx++) {
                std::string text;
                for (size_t k = 0, v = idx; k < len; k++, v /= chars.size()) {
                    text += chars[v % chars.size()];
                }
                if (server_decision_escape_special_tokens(text) != std::regex_replace(text, re_special, "<\xC2\xA6$1\xC2\xA6>")) {
                    n_diff++;
                }
                n_strings++;
            }
        }
        t.assert_equal("strings that differ, of " + std::to_string(n_strings), (size_t) 0, n_diff);
    });

    // std::regex overflowed the stack on these
    t.test("long input", [](testing & t) {
        const std::string name(1000000, 'a');
        t.assert_true("not closed", server_decision_escape_special_tokens("<|" + name) == "<|" + name);
        t.assert_true("closed", server_decision_escape_special_tokens("<|" + name + "|>") == escaped(name));
    });
}

struct cache_test_directory {
    std::filesystem::path path;

    cache_test_directory() {
        do {
            path = std::filesystem::temp_directory_path() / ("llama-cache-test-" + std::to_string(std::random_device{}()));
        } while (!std::filesystem::create_directory(path));
    }

    ~cache_test_directory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

static std::string cache_test_executable;

#if !defined(_WIN32)
static int cache_test_child(const std::filesystem::path & path, const std::string & mode) {
    const std::string directory = path.u8string();
    const char * const args[] = {cache_test_executable.c_str(), "--disk-cache-child", directory.c_str(), mode.c_str(), nullptr};
    const auto pid = fork();
    if (pid < 0) {
        throw std::runtime_error("cannot fork disk cache test child");
    }
    if (pid == 0) {
        execv(args[0], const_cast<char * const *>(args));
        _exit(127);
    }
    int status = 0;
    int result;
    do {
        result = waitpid(pid, &status, 0);
    } while (result < 0 && errno == EINTR);
    return result == pid && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
#endif

static server_prompt cache_test_prompt(std::initializer_list<llama_token> tokens) {
    server_prompt prompt;
    prompt.tokens = server_tokens(llama_tokens(tokens), false);
    return prompt;
}

static bool cache_test_write(const std::string & path) {
    std::ofstream file(std::filesystem::u8path(path), std::ios::binary);
    std::string data(2048, 's');
    const uint32_t header[2] = {LLAMA_STATE_SEQ_MAGIC, LLAMA_STATE_SEQ_VERSION};
    std::memcpy(data.data(), header, sizeof(header));
    file.write(data.data(), data.size());
    file.close();
    return bool(file);
}

static size_t cache_test_count(const std::filesystem::path & path) {
    size_t count = 0;
    for (const auto & item : std::filesystem::directory_iterator(path)) {
        count += item.is_directory() && std::filesystem::exists(item.path() / "meta.json");
    }
    return count;
}

static void test_disk_cache(testing & t) {
    t.test("loss thresholds", [](testing & t) {
        t.assert_true("small edit", !server_cache_should_save(32768, 30000, 8192));
        t.assert_true("8191 tokens", !server_cache_should_save(32768, 24577, 8192));
        t.assert_true("8192 tokens", server_cache_should_save(32768, 24576, 8192));
        t.assert_true("exactly half", !server_cache_should_save(8000, 4000, 8192));
        t.assert_true("majority below threshold", !server_cache_should_save(8000, 3999, 8192));
        t.assert_true("short context entirely replaced", !server_cache_should_save(8000, 0, 8192));
        t.assert_true("unchanged", !server_cache_should_save(8000, 8000, 0));
        t.assert_true("generated tail discarded, input retained", !server_cache_should_save(16384, 16384, 8192));
        t.assert_true("reuse extends into generated tail", !server_cache_should_save(16384, 24576, 8192));
    });

    t.test("checkpoint reuse", [](testing & t) {
        auto prompt = cache_test_prompt({1, 2, 3, 4, 5, 6, 7, 8});
        server_tokens input(llama_tokens{1, 2, 3, 4, 9}, false);
        t.assert_equal("full attention", (size_t) 4, server_cache_reusable(prompt, input, 0, 0));
        t.assert_equal("recurrent state too recent", (size_t) 0, server_cache_reusable(prompt, input, 7, 0));
        common_prompt_checkpoint checkpoint;
        checkpoint.n_tokens = 3;
        checkpoint.pos_min = 2;
        checkpoint.pos_max = 3;
        prompt.checkpoints.push_back(checkpoint);
        checkpoint.n_tokens = 7;
        checkpoint.pos_min = 6;
        checkpoint.pos_max = 7;
        prompt.checkpoints.push_back(checkpoint);
        t.assert_equal("older checkpoint", (size_t) 3, server_cache_reusable(prompt, input, 7, 0));
        prompt.checkpoints.clear();
        t.assert_equal("SWA window present", (size_t) 4, server_cache_reusable(prompt, input, 1, 4));
        t.assert_equal("SWA window missing", (size_t) 0, server_cache_reusable(prompt, input, 2, 4));
        t.assert_equal("reevaluate final token", (size_t) 7, server_cache_reusable(prompt, prompt.tokens, 0, 0));
    });

    t.test("restart, namespaces and duplicate saves", [](testing & t) {
        cache_test_directory dir;
        auto prompt = cache_test_prompt({1, 2, 3});
        size_t writes = 0;
        auto writer = [&](const std::string & path) {
            ++writes;
            return cache_test_write(path);
        };
        {
            server_disk_cache cache(dir.path.u8string(), 65536);
            t.assert_true("save", cache.save("model-a", prompt, 0, 0, 2300, writer));
            t.assert_true("duplicate", cache.save("model-a", prompt, 0, 0, 2300, writer));
            t.assert_equal("one payload write", (size_t) 1, writes);
        }
        server_disk_cache cache(dir.path.u8string());
        t.assert_equal("adopt pool quota", (uint64_t) 65536, cache.limit());
        server_tokens input(llama_tokens{1, 2, 3, 4}, false);
        size_t reads = 0;
        auto reader = [&](const std::string & path) {
            ++reads;
            return std::filesystem::file_size(std::filesystem::u8path(path)) == 2048;
        };
        t.assert_true("restore after restart", cache.load("model-a", input, 64, false, 0, 32, 1, 0, reader));
        t.assert_true("different model misses", !cache.load("model-b", input, 64, false, 0, 32, 1, 0, reader));
        t.assert_true("short prefix misses", !cache.load("model-a", input, 64, false, 0, 32, 4, 0, reader));
        t.assert_true("live slot wins a tie", !cache.load("model-a", input, 64, false, 0, 32, 1, 3, reader));
        t.assert_equal("one payload read", (size_t) 1, reads);
        bool rejected = false;
        try {
            server_disk_cache conflict(dir.path.u8string(), 32768);
        } catch (const std::exception &) {
            rejected = true;
        }
        t.assert_true("conflicting pool limit rejected", rejected);
    });

    t.test("global LRU and pinned reader", [](testing & t) {
        cache_test_directory dir;
        server_disk_cache first(dir.path.u8string(), 6000);
        server_disk_cache second(dir.path.u8string());
        auto a = cache_test_prompt({1, 2, 3});
        auto b = cache_test_prompt({4, 5, 6});
        auto c = cache_test_prompt({7, 8, 9});
        t.assert_true("first model", first.save("a", a, 0, 0, 2300, cache_test_write));
        t.assert_true("second model", second.save("b", b, 0, 0, 2300, cache_test_write));
        for (const auto & item : std::filesystem::directory_iterator(dir.path)) {
            if (item.is_directory()) {
                std::filesystem::last_write_time(item.path() / "meta.json",
                    std::filesystem::file_time_type::clock::now() - std::chrono::hours(1));
            }
        }
        server_tokens input(llama_tokens{1, 2, 3, 10}, false);
        t.assert_true("reader stays pinned while another pool instance saves",
            first.load("a", input, 64, false, 0, 32, 1, 0, [&](const std::string & path) {
                std::filesystem::last_write_time(std::filesystem::u8path(path).parent_path() / "meta.json",
                    std::filesystem::file_time_type::clock::now() - std::chrono::hours(2));
                t.assert_true("save third model", second.save("c", c, 0, 0, 2300, cache_test_write));
                return std::filesystem::exists(std::filesystem::u8path(path));
            }));
        t.assert_equal("two snapshots fit", (size_t) 2, cache_test_count(dir.path));
        server_tokens old(llama_tokens{4, 5, 6, 10}, false);
        t.assert_true("other model evicted", !second.load("b", old, 64, false, 0, 32, 1, 0, [](const std::string &) { return true; }));
        t.assert_true("oversize skipped", !first.save("large", a, 0, 0, 6001, [](const std::string &) { return true; }));
    });

    t.test("in-flight reservations count toward shared quota", [](testing & t) {
        cache_test_directory dir;
        server_disk_cache first(dir.path.u8string(), 6000);
        server_disk_cache second(dir.path.u8string());
        auto a = cache_test_prompt({1, 2, 3});
        auto b = cache_test_prompt({4, 5, 6});
        t.assert_true("first writer", first.save("a", a, 0, 0, 4000, [&](const std::string & path) {
            bool wrote = false;
            t.assert_true("second writer cannot overbook", !second.save("b", b, 0, 0, 4000,
                [&](const std::string &) { wrote = true; return true; }));
            t.assert_true("second payload never written", !wrote);
            return cache_test_write(path);
        }));
        t.assert_equal("one snapshot published", (size_t) 1, cache_test_count(dir.path));
    });

#if !defined(_WIN32)
    t.test("independent processes respect reservations and recover crashes", [](testing & t) {
        cache_test_directory dir;
        server_disk_cache cache(dir.path.u8string(), 6000);
        auto prompt = cache_test_prompt({1, 2, 3});
        t.assert_true("parent write", cache.save("parent", prompt, 0, 0, 4000, [&](const std::string & path) {
            t.assert_equal("child cannot overbook parent's reservation", 10, cache_test_child(dir.path, "save"));
            return cache_test_write(path);
        }));
        t.assert_equal("one complete snapshot", (size_t) 1, cache_test_count(dir.path));
        t.assert_equal("child exits while writing", 23, cache_test_child(dir.path, "crash"));
        server_disk_cache restarted(dir.path.u8string());
        for (const auto & item : std::filesystem::directory_iterator(dir.path)) {
            if (item.is_directory()) {
                t.assert_true("no orphan reservation", !std::filesystem::exists(item.path() / "reservation.json"));
            }
        }
        t.assert_true("write after recovery", restarted.save("recovered", prompt, 0, 0, 2300, cache_test_write));
    });
#endif

    t.test("failed writes and interrupted saves", [](testing & t) {
        cache_test_directory dir;
        server_disk_cache cache(dir.path.u8string(), 65536);
        auto prompt = cache_test_prompt({1, 2, 3});
        t.assert_true("failed write", !cache.save("model", prompt, 0, 0, 2300, [](const std::string & path) {
            cache_test_write(path);
            return false;
        }));
        t.assert_equal("no incomplete snapshot published", (size_t) 0, cache_test_count(dir.path));
        const auto abandoned = dir.path / std::string(64, 'a');
        std::filesystem::create_directory(abandoned);
        std::ofstream(abandoned / "state.bin") << "partial";
        std::ofstream(abandoned / "reservation.json") << "{\"bytes\":60000}";
        server_disk_cache restarted(dir.path.u8string());
        t.assert_true("abandoned write removed", !std::filesystem::exists(abandoned));
        t.assert_true("retry succeeds", restarted.save("model", prompt, 0, 0, 2300, cache_test_write));
        for (const auto & item : std::filesystem::directory_iterator(dir.path)) {
            if (item.is_directory()) {
                std::ofstream(item.path() / "state.bin", std::ios::trunc) << "truncated";
            }
        }
        server_tokens input(llama_tokens{1, 2, 3, 4}, false);
        bool read = false;
        t.assert_true("truncated payload skipped", !restarted.load("model", input, 64, false, 0, 32, 1, 0,
            [&](const std::string &) { read = true; return true; }));
        t.assert_true("invalid payload not loaded", !read);
    });

    t.test("continuations replace snapshots, branches survive", [](testing & t) {
        cache_test_directory dir;
        server_disk_cache cache(dir.path.u8string(), 65536);
        auto old = cache_test_prompt({1, 2, 3});
        auto extended = cache_test_prompt({1, 2, 3, 4});
        auto branch = cache_test_prompt({1, 2, 9});
        t.assert_true("old", cache.save("model", old, 0, 0, 2300, cache_test_write));
        t.assert_true("extended", cache.save("model", extended, 0, 0, 2300, cache_test_write));
        t.assert_equal("superseded snapshot removed", (size_t) 1, cache_test_count(dir.path));
        t.assert_true("branch", cache.save("model", branch, 0, 0, 2300, cache_test_write));
        t.assert_equal("fork retained", (size_t) 2, cache_test_count(dir.path));
    });

    t.test("failed restores keep the snapshot", [](testing & t) {
        cache_test_directory dir;
        auto prompt = cache_test_prompt({1, 2, 3});
        server_tokens input(llama_tokens{1, 2, 3, 4}, false);
        size_t writes = 0;
        auto writer = [&](const std::string & path) {
            ++writes;
            return cache_test_write(path);
        };
        {
            server_disk_cache cache(dir.path.u8string(), 65536);
            t.assert_true("save", cache.save("model", prompt, 0, 0, 2300, writer));
            size_t reads = 0;
            auto failing = [&](const std::string &) { ++reads; return false; };
            t.assert_true("restore fails", !cache.load("model", input, 64, false, 0, 32, 1, 0, failing));
            t.assert_equal("snapshot kept", (size_t) 1, cache_test_count(dir.path));
            t.assert_true("not retried in this process", !cache.load("model", input, 64, false, 0, 32, 1, 0, failing));
            t.assert_equal("one failed read", (size_t) 1, reads);
            t.assert_true("identical save rewrites it", cache.save("model", prompt, 0, 0, 2300, writer));
            t.assert_equal("rewritten", (size_t) 2, writes);
            t.assert_true("usable after rewrite", cache.load("model", input, 64, false, 0, 32, 1, 0,
                [](const std::string &) { return true; }));
        }
        server_disk_cache restarted(dir.path.u8string());
        t.assert_true("retried after restart", restarted.load("model", input, 64, false, 0, 32, 1, 0,
            [](const std::string &) { return true; }));
    });

    t.test("invalid state header is removed", [](testing & t) {
        cache_test_directory dir;
        server_disk_cache cache(dir.path.u8string(), 65536);
        auto prompt = cache_test_prompt({1, 2, 3});
        t.assert_true("save", cache.save("model", prompt, 0, 0, 2300, cache_test_write));
        for (const auto & item : std::filesystem::directory_iterator(dir.path)) {
            if (item.is_directory()) {
                std::fstream state(item.path() / "state.bin", std::ios::binary | std::ios::in | std::ios::out);
                state.write("BAD!", 4);
            }
        }
        server_tokens input(llama_tokens{1, 2, 3, 4}, false);
        bool read = false;
        t.assert_true("not loaded", !cache.load("model", input, 64, false, 0, 32, 1, 0,
            [&](const std::string &) { read = true; return true; }));
        t.assert_true("reader not called", !read);
        t.assert_equal("removed", (size_t) 0, cache_test_count(dir.path));
    });
}

int main(int argc, char ** argv) {
    if (argc == 4 && std::string(argv[1]) == "--disk-cache-child") {
        server_disk_cache cache(argv[2]);
        auto prompt = cache_test_prompt({4, 5, 6});
        const bool crash = std::string(argv[3]) == "crash";
        const bool saved = cache.save("child", prompt, 0, 0, 4000, [&](const std::string & path) {
            if (crash) {
                std::ofstream(std::filesystem::u8path(path)) << "incomplete state";
                std::_Exit(23);
            }
            return cache_test_write(path);
        });
        return saved ? 0 : 10;
    }
    cache_test_executable = std::filesystem::absolute(argv[0]).u8string();
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("server_tokens", test_server_tokens);
    t.test("utf8",          test_utf8);
    t.test("sse",           test_sse);
    t.test("json_helpers",  test_json_helpers);
    t.test("lora_helpers",  test_lora_helpers);
    t.test("server_pipe",   test_server_pipe);
    t.test("decision_escape", test_decision_escape);
    t.test("disk_cache", test_disk_cache);

    return t.summary();
}
