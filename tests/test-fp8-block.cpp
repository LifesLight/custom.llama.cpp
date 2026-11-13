#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

static float decode(uint8_t code) {
    const int exponent = (code >> 3) & 15;
    const int mantissa = code & 7;
    const float magnitude = exponent == 0 ? std::ldexp(float(mantissa), -9)
                                         : std::ldexp(float(8 + mantissa), exponent - 10);
    return code & 128 ? -magnitude : magnitude;
}

static bool check(ggml_backend_t backend, int tokens, bool unit_vectors) {
    constexpr int k = 384;
    constexpr int n = 640;
    constexpr int blocks_k = k / 128;
    constexpr int blocks_n = n / 128;

    ggml_init_params params = { 64 * 1024, nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F8_E4M3, k, n);
    ggml_tensor * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, tokens);
    ggml_tensor * s = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, blocks_k, blocks_n);
    ggml_tensor * y = ggml_mul_mat_fp8_block(ctx.get(), w, x, s);
    if (y->ne[0] != n || y->ne[1] != tokens) {
        std::fprintf(stderr, "unexpected FP8 matmul output shape\n");
        return false;
    }
    if (!ggml_backend_supports_op(backend, y)) {
        std::fprintf(stderr, "%s does not support block-scaled FP8\n", ggml_backend_name(backend));
        return false;
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 8, false);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        return false;
    }

    std::vector<uint8_t> weights(n*k);
    std::vector<float> input(tokens*k);
    std::vector<float> scales(blocks_n*blocks_k);
    std::vector<float> output(tokens*n);
    for (int i = 0; i < n*k; ++i) {
        weights[i] = (1 + (i*43 + i/31) % 126) | (((i + i/13) % 2) << 7);
    }
    for (int i = 0; i < blocks_n*blocks_k; ++i) {
        scales[i] = 0.00213f * (1 + i);
    }
    if (unit_vectors) {
        const int columns[] = { 127, 128, 255, 383 };
        for (int t = 0; t < tokens; ++t) {
            input[t*k + columns[t]] = 1.0f;
        }
    } else {
        for (int i = 0; i < tokens*k; ++i) {
            input[i] = ((i*7 + i/17) % 29 - 14) / 17.0f;
        }
    }

    ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
    if (tokens > 0) {
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    }
    ggml_backend_tensor_set(s, scales.data(), 0, ggml_nbytes(s));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        return false;
    }
    if (tokens > 0) {
        ggml_backend_tensor_get(y, output.data(), 0, ggml_nbytes(y));
    }

    for (int t = 0; t < tokens; ++t) {
        for (int row = 0; row < n; ++row) {
            double expected = 0.0;
            for (int col = 0; col < k; ++col) {
                const float weight = decode(weights[row*k + col]) * scales[row/128*blocks_k + col/128];
                expected += double(weight) * input[t*k + col];
            }
            const double actual = output[t*n + row];
            const double tolerance = unit_vectors ? 0.0 : 0.0002 * (1 + std::abs(expected));
            if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
                std::fprintf(stderr, "%s: tokens=%d t=%d row=%d got %.9g expected %.9g\n",
                             ggml_backend_name(backend), tokens, t, row, actual, expected);
                return false;
            }
        }
    }
    std::printf("%s: %d tokens passed\n", ggml_backend_name(backend), tokens);
    return true;
}

int main() {
    ggml_backend_load_all();
    bool ran = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU && std::string_view(reg) != "CUDA") {
            continue;
        }
        ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
        if (!backend || !check(backend.get(), 0, false) || !check(backend.get(), 4, true) || !check(backend.get(), 13, false)) {
            return 1;
        }
        ran = true;
    }
    return ran ? 0 : 1;
}
