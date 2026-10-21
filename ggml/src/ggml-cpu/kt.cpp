// CPU dot products for the trellis quant IQ3_KT (see ggml-kt.cpp)
// Ported from ik_llama.cpp (https://github.com/ikawrakow/ik_llama.cpp), Copyright (C) 2025 Iwan Kawrakow, MIT license

#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "quants.h"
#include "ggml-quants.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

namespace {

constexpr uint32_t kKTMul        = 0xCBAC1FEDu;
constexpr uint32_t kIQ3KTOffset  = 4096;

inline int kt_next(uint32_t & x) {
    x *= kKTMul;
    const uint32_t s = x & 0x3f3f3f3fu;
    return int(s & 0xff) + int((s >> 8) & 0xff) + int((s >> 16) & 0xff) + int(s >> 24) - 126;
}

#if defined(__AVX2__)

constexpr uint32_t kKTMul2 = kKTMul*kKTMul;
constexpr uint32_t kKTMul3 = kKTMul2*kKTMul;
constexpr uint32_t kKTMul4 = kKTMul3*kKTMul;
constexpr uint32_t kKTMul5 = kKTMul4*kKTMul;
constexpr uint32_t kKTMul6 = kKTMul5*kKTMul;
constexpr uint32_t kKTMul7 = kKTMul6*kKTMul;
constexpr uint32_t kKTMul8 = kKTMul7*kKTMul;

// trellis values of 8 lanes of seeds, as int32
inline __m256i kt_gen8(__m256i seeds, __m256i mul) {
    const __m256i v = _mm256_and_si256(_mm256_mullo_epi32(seeds, mul), _mm256_set1_epi32(0x3f3f3f3f));
    const __m256i s = _mm256_madd_epi16(_mm256_maddubs_epi16(v, _mm256_set1_epi8(1)), _mm256_set1_epi16(1));
    return _mm256_add_epi32(s, _mm256_set1_epi32(-126));
}

// pack 4 x 8 int32 (elements 0-7, 8-15, 16-23, 24-31) into 32 int8 in order
inline __m256i kt_pack32(__m256i v0, __m256i v1, __m256i v2, __m256i v3) {
    const __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(v0, v1), _mm256_packs_epi32(v2, v3));
    return _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

inline float hsum_float_8(__m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

#endif

} // namespace

extern "C" {

void quantize_row_iq3_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq3_kt(x, y, 1, k, NULL);
}

void ggml_vec_dot_iq3_kt_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    const block_iq3_kt * GGML_RESTRICT x = (const block_iq3_kt *) vx;
    const block_q8_K   * GGML_RESTRICT y = (const block_q8_K *) vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * GGML_KT_D_SCALE * y[i].d;
        const int8_t * q8 = y[i].qs;
        int sumi = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = (x[i].scales[ib%4] >> 4*(ib/4)) & 0xf;
            int sumb = 0;
            for (int ig = 0; ig < 4; ++ig) {
                const int g = 4*ib + ig;
                uint32_t idx = (x[i].ql[2*g] | (x[i].ql[2*g+1] << 8)) + kIQ3KTOffset;
                for (int j = 0; j < 8; ++j) {
                    const int l = 8*ig + j;
                    const int v = abs(kt_next(idx));
                    sumb += (x[i].qh[l] >> ib) & 1 ? -v*q8[32*ib + l] : v*q8[32*ib + l];
                }
            }
            sumi += ls*sumb;
        }
        sumf += d*sumi;
    }
    *s = sumf;
}

// decode one block of 256 for the tiled matmul: codes are value + 128, one int scale per 32, block scale d
void ggml_kt_unpack_iq3_kt(const void * GGML_RESTRICT vx, uint8_t * GGML_RESTRICT q, int32_t * GGML_RESTRICT scales, float * GGML_RESTRICT d) {
    const block_iq3_kt * x = (const block_iq3_kt *) vx;
    *d = GGML_CPU_FP16_TO_FP32(x->d) * GGML_KT_D_SCALE;
    for (int ib = 0; ib < QK_K/32; ++ib) {
        scales[ib] = (x->scales[ib%4] >> 4*(ib/4)) & 0xf;
    }
    uint16_t ql[QK_K/8];
    memcpy(ql, x->ql, sizeof(ql));
#if defined(__AVX2__)
    const __m256i mul    = _mm256_setr_epi32(kKTMul, kKTMul2, kKTMul3, kKTMul4, kKTMul5, kKTMul6, kKTMul7, kKTMul8);
    const __m256i offset = _mm256_set1_epi32(kIQ3KTOffset);
    const __m256i signs  = _mm256_loadu_si256((const __m256i *) x->qh);
    for (int ib = 0; ib < QK_K/32; ++ib) {
        const __m256i seeds = _mm256_add_epi32(_mm256_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(ql + 4*ib))), offset);
        const __m256i v0 = kt_gen8(_mm256_permutevar8x32_epi32(seeds, _mm256_set1_epi32(0)), mul);
        const __m256i v1 = kt_gen8(_mm256_permutevar8x32_epi32(seeds, _mm256_set1_epi32(1)), mul);
        const __m256i v2 = kt_gen8(_mm256_permutevar8x32_epi32(seeds, _mm256_set1_epi32(2)), mul);
        const __m256i v3 = kt_gen8(_mm256_permutevar8x32_epi32(seeds, _mm256_set1_epi32(3)), mul);
        const __m256i vabs = _mm256_abs_epi8(kt_pack32(v0, v1, v2, v3));
        const __m256i bit  = _mm256_set1_epi8(1 << ib);
        const __m256i neg  = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit), bit);
        const __m256i v    = _mm256_sub_epi8(_mm256_xor_si256(vabs, neg), neg);
        _mm256_storeu_si256((__m256i *)(q + 32*ib), _mm256_xor_si256(v, _mm256_set1_epi8((char) 0x80)));
    }
#else
    for (int ib = 0; ib < QK_K/32; ++ib) {
        for (int ig = 0; ig < 4; ++ig) {
            uint32_t idx = ql[4*ib + ig] + kIQ3KTOffset;
            for (int j = 0; j < 8; ++j) {
                const int l = 8*ig + j;
                const int v = abs(kt_next(idx));
                q[32*ib + l] = (uint8_t) (128 + ((x->qh[l] >> ib) & 1 ? -v : v));
            }
        }
    }
#endif
}

void ggml_vec_dot_iq3_kt_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    const block_iq3_kt * GGML_RESTRICT x = (const block_iq3_kt *) vx;
    const block_q8_K   * GGML_RESTRICT y = (const block_q8_K *) vy;

    const int nb = n / QK_K;

    const __m256i mul    = _mm256_setr_epi32(kKTMul, kKTMul2, kKTMul3, kKTMul4, kKTMul5, kKTMul6, kKTMul7, kKTMul8);
    const __m256i offset = _mm256_set1_epi32(kIQ3KTOffset);
    const __m256i one    = _mm256_set1_epi8(1);

    __m256 accf = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * GGML_KT_D_SCALE * y[i].d;

        uint16_t ql[QK_K/8];
        memcpy(ql, x[i].ql, sizeof(ql));
        uint32_t aux32; memcpy(&aux32, x[i].scales, 4);

        const __m256i signs = _mm256_loadu_si256((const __m256i *) x[i].qh);
        const int8_t * q8 = y[i].qs;

        __m256i sumi = _mm256_setzero_si256();
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const __m256i seeds = _mm256_add_epi32(_mm256_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(ql + 4*ib))), offset);
            const __m256i v0 = kt_gen8(_mm256_permutevar8x32_epi32(seeds, _mm256_set1_epi32(0)), mul);
            const __m256i v1 = kt_gen8(_mm256_permutevar8x32_epi32(seeds, _mm256_set1_epi32(1)), mul);
            const __m256i v2 = kt_gen8(_mm256_permutevar8x32_epi32(seeds, _mm256_set1_epi32(2)), mul);
            const __m256i v3 = kt_gen8(_mm256_permutevar8x32_epi32(seeds, _mm256_set1_epi32(3)), mul);
            const __m256i vabs = _mm256_abs_epi8(kt_pack32(v0, v1, v2, v3));

            const __m256i bit  = _mm256_set1_epi8(1 << ib);
            const __m256i neg  = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit), bit);
            const __m256i q8b  = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(q8 + 32*ib)), _mm256_or_si256(neg, one));

            const int ls = (aux32 >> (8*(ib%4) + 4*(ib/4))) & 0xf;
            const __m256i p16 = _mm256_maddubs_epi16(vabs, q8b);
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p16, _mm256_set1_epi16(ls)));
        }
        accf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accf);
    }

    *s = hsum_float_8(accf);
#else
    ggml_vec_dot_iq3_kt_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

}
