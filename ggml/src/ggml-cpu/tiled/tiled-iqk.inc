// Tiled matmul support for the IQ3_KT and IQ_K quants (ported from ik_llama.cpp, Copyright (C) 2024-2025 Iwan Kawrakow,
// MIT license). Included once from tiled.cpp, after the other tiled_unpack_src0 overloads.

#include "quants.h"

// iq3_kt: trellis values in [-126, 126] stored as value + 128 (BIAS = 128), int scale per 32, no min
// also used by the IQ_K types: grid values in [-127, 127] stored as value + 128, NB int scales per block of 256
template <typename block_kt, int NB = 8>
static void tiled_unpack_src0_kt(const block_kt * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile, int num_k,
                                 void (*unpack)(const void *, uint8_t *, int32_t *, float *)) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);

    const int qk_stride = num_k * TILED_TILE_K;
    const int nb_stride = NB * num_k;
    for (int slab = 0; slab < num_k; slab++) {
        for (int r = 0; r < n_rows; r++) {
            const int d_off = slab * TILED_MICRO + r;
            const int q_off = r * qk_stride + slab * TILED_TILE_K;
            const int s_off = r * nb_stride + slab * NB;
            unpack(&rows[r * row_stride + slab], &tile->q[q_off], &tile->scales[s_off], &tile->d[d_off]);
        }
    }
}

static void tiled_unpack_src0(const block_iq3_kt * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile, int num_k) {
    tiled_unpack_src0_kt(rows, row_stride, n_rows, tile, num_k, ggml_kt_unpack_iq3_kt);
}

// IQ_K types: IQ2_K, IQ3_K, IQ4_K have a scale per 16, the others per 32
static void tiled_unpack_src0(const block_iq2_k * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile, int num_k) {
    tiled_unpack_src0_kt<block_iq2_k, 16>(rows, row_stride, n_rows, tile, num_k, ggml_iqk_unpack_iq2_k);
}

static void tiled_unpack_src0(const block_iq3_k * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile, int num_k) {
    tiled_unpack_src0_kt<block_iq3_k, 16>(rows, row_stride, n_rows, tile, num_k, ggml_iqk_unpack_iq3_k);
}

static void tiled_unpack_src0(const block_iq4_k * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile, int num_k) {
    tiled_unpack_src0_kt<block_iq4_k, 16>(rows, row_stride, n_rows, tile, num_k, ggml_iqk_unpack_iq4_k);
}

static void tiled_unpack_src0(const block_iq2_ks * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile, int num_k) {
    tiled_unpack_src0_kt<block_iq2_ks, 8>(rows, row_stride, n_rows, tile, num_k, ggml_iqk_unpack_iq2_ks);
}

static void tiled_unpack_src0(const block_iq3_ks * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile, int num_k) {
    tiled_unpack_src0_kt<block_iq3_ks, 8>(rows, row_stride, n_rows, tile, num_k, ggml_iqk_unpack_iq3_ks);
}

static void tiled_unpack_src0(const block_iq4_ks * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile, int num_k) {
    tiled_unpack_src0_kt<block_iq4_ks, 8>(rows, row_stride, n_rows, tile, num_k, ggml_iqk_unpack_iq4_ks);
}

static void tiled_unpack_src0(const block_iq4_kss * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile, int num_k) {
    tiled_unpack_src0_kt<block_iq4_kss, 8>(rows, row_stride, n_rows, tile, num_k, ggml_iqk_unpack_iq4_kss);
}

static void tiled_unpack_src0(const block_iq2_kl * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile, int num_k) {
    tiled_unpack_src0_kt<block_iq2_kl, 8>(rows, row_stride, n_rows, tile, num_k, ggml_iqk_unpack_iq2_kl);
}

// cases of ggml_tiled_matmul_type_dispatch(). Trellis values reach +-126 and the IQ_K grids +-127, too large for the
// act-bias MAC: these use the iq4_xs sign trick (ACTBIAS = false)
#define GGML_TILED_IQK_DISPATCH_CASES \
    case GGML_TYPE_IQ3_KT: return tiled_matmul_dispatch<block_iq3_kt, 32, false, 128, false>(params, dst, expert_rows, cur_a, cne1, scratch); \
    case GGML_TYPE_IQ2_K: return tiled_matmul_dispatch<block_iq2_k, 16, false, 128, false>(params, dst, expert_rows, cur_a, cne1, scratch); \
    case GGML_TYPE_IQ3_K: return tiled_matmul_dispatch<block_iq3_k, 16, false, 128, false>(params, dst, expert_rows, cur_a, cne1, scratch); \
    case GGML_TYPE_IQ4_K: return tiled_matmul_dispatch<block_iq4_k, 16, false, 128, false>(params, dst, expert_rows, cur_a, cne1, scratch); \
    case GGML_TYPE_IQ2_KS: return tiled_matmul_dispatch<block_iq2_ks, 32, false, 128, false>(params, dst, expert_rows, cur_a, cne1, scratch); \
    case GGML_TYPE_IQ3_KS: return tiled_matmul_dispatch<block_iq3_ks, 32, false, 128, false>(params, dst, expert_rows, cur_a, cne1, scratch); \
    case GGML_TYPE_IQ4_KS: return tiled_matmul_dispatch<block_iq4_ks, 32, false, 128, false>(params, dst, expert_rows, cur_a, cne1, scratch); \
    case GGML_TYPE_IQ4_KSS: return tiled_matmul_dispatch<block_iq4_kss, 32, false, 128, false>(params, dst, expert_rows, cur_a, cne1, scratch); \
    case GGML_TYPE_IQ2_KL: return tiled_matmul_dispatch<block_iq2_kl, 32, false, 128, false>(params, dst, expert_rows, cur_a, cne1, scratch);
