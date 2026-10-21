// CPU dot products for the IQ_K quants (IQ2_K, IQ3_K, IQ4_K, IQ2_KS, IQ3_KS, IQ4_KS, IQ4_KSS, IQ2_KL), see ggml-iqk.cpp
// Formats from ik_llama.cpp (https://github.com/ikawrakow/ik_llama.cpp), Copyright (C) 2024-2025 Iwan Kawrakow, MIT license

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

//
// exact integer decoding of one block of 256: q = the 256 grid values, scales = one int per sub-block of 16 (NB = 16)
// or 32 (NB = 8), d = block scale. Value = d * scales[j / (256/NB)] * q[j]. Used by the generic dot products,
// the tiled matmul and the tests
//

inline void decode_iq2_k(const block_iq2_k * x, int8_t * q, int32_t * scales, float * d) {
    *d = GGML_CPU_FP16_TO_FP32(x->d);
    const uint16_t extra = x->extra;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        scales[2*ib32+0] = int(x->scales[ib32] & 0xf) - 8;
        scales[2*ib32+1] = int(x->scales[ib32] >>  4) - 8;
        const int8_t * v1 = iq2nl_values + 4*((extra >> (2*ib32+0)) & 1);
        const int8_t * v2 = iq2nl_values + 4*((extra >> (2*ib32+1)) & 1);
        const uint8_t * qs = x->qs + 32*(ib32/4);
        const int shift = 2*(ib32%4);
        for (int j = 0; j < 16; ++j) {
            q[32*ib32 + j +  0] = v1[(qs[j+ 0] >> shift) & 3];
            q[32*ib32 + j + 16] = v2[(qs[j+16] >> shift) & 3];
        }
    }
}

inline void decode_iq3_k(const block_iq3_k * x, int8_t * q, int32_t * scales, float * d) {
    *d = GGML_CPU_FP16_TO_FP32(x->d);
    const uint16_t extra = x->extra;
    const uint16_t sh    = x->scales_h;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const int ls1 = 2*(x->scales_l[ib32] & 0xf) + 1;
        const int ls2 = 2*(x->scales_l[ib32] >>  4) + 1;
        scales[2*ib32+0] = (sh >> (2*ib32+0)) & 1 ? -ls1 : ls1;
        scales[2*ib32+1] = (sh >> (2*ib32+1)) & 1 ? -ls2 : ls2;
        const int8_t * v1 = iq3nl_values + 8*((extra >> (2*ib32+0)) & 1);
        const int8_t * v2 = iq3nl_values + 8*((extra >> (2*ib32+1)) & 1);
        const uint8_t * qs = x->qs + 32*(ib32/4);
        const int shift = 2*(ib32%4);
        for (int j = 0; j < 16; ++j) {
            q[32*ib32 + j +  0] = v1[((qs[j+ 0] >> shift) & 3) | (((x->qh[j+ 0] >> ib32) & 1) << 2)];
            q[32*ib32 + j + 16] = v2[((qs[j+16] >> shift) & 3) | (((x->qh[j+16] >> ib32) & 1) << 2)];
        }
    }
}

inline void decode_iq4_k(const block_iq4_k * x, int8_t * q, int32_t * scales, float * d) {
    *d = GGML_CPU_FP16_TO_FP32(x->d);
    const uint16_t extra = x->extra;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const uint8_t sh = x->scales_h[ib32/2] >> 4*(ib32%2);
        scales[2*ib32+0] = int((x->scales_l[ib32] & 0xf) | ((sh << 4) & 0x30)) - 32;
        scales[2*ib32+1] = int((x->scales_l[ib32] >>  4) | ((sh << 2) & 0x30)) - 32;
        const int8_t * v1 = iq4k_values + 16*((extra >> (2*ib32+0)) & 1);
        const int8_t * v2 = iq4k_values + 16*((extra >> (2*ib32+1)) & 1);
        const uint8_t * qs = x->qs + 16*ib32;
        for (int j = 0; j < 16; ++j) {
            q[32*ib32 + j +  0] = v1[qs[j] & 0xf];
            q[32*ib32 + j + 16] = v2[qs[j] >>  4];
        }
    }
}

inline void decode_iq2_ks(const block_iq2_ks * x, int8_t * q, int32_t * scales, float * d) {
    *d = GGML_CPU_FP16_TO_FP32(x->d);
    const uint16_t extra = x->extra;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        scales[ib32] = int(((x->scales[ib32/2] >> 4*(ib32%2)) & 0xf) | (((extra >> (8 + ib32)) & 1) << 4)) - 16;
        const int8_t * v = iq2nl_values + 4*((extra >> ib32) & 1);
        const uint8_t * qs = x->qs + 32*(ib32/4);
        const int shift = 2*(ib32%4);
        for (int j = 0; j < 32; ++j) {
            q[32*ib32 + j] = v[(qs[j] >> shift) & 3];
        }
    }
}

inline void decode_iq3_ks(const block_iq3_ks * x, int8_t * q, int32_t * scales, float * d) {
    *d = GGML_CPU_FP16_TO_FP32(x->d);
    const uint16_t extra = x->extra;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        scales[ib32] = int(((x->scales[ib32%4] >> 4*(ib32/4)) & 0xf) | (((extra >> ib32) & 1) << 4)) - 16;
        const int8_t * v = iq3nl_values + 8*((extra >> (8 + ib32)) & 1);
        const uint8_t * qs = x->qs + 32*(ib32/4);
        const int shift = 2*(ib32%4);
        for (int j = 0; j < 32; ++j) {
            q[32*ib32 + j] = v[((qs[j] >> shift) & 3) | (((x->qh[j] >> ib32) & 1) << 2)];
        }
    }
}

inline void decode_iq4_ks(const block_iq4_ks * x, int8_t * q, int32_t * scales, float * d) {
    *d = GGML_CPU_FP16_TO_FP32(x->d);
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        scales[ib32] = int(x->scales[ib32] & 254) - 127;
        const int8_t * v = iq4k_values + 16*(x->scales[ib32] & 1);
        const uint8_t * qs = x->qs + 16*ib32;
        for (int j = 0; j < 16; ++j) {
            q[32*ib32 + j +  0] = v[qs[j] & 0xf];
            q[32*ib32 + j + 16] = v[qs[j] >>  4];
        }
    }
}

inline void decode_iq4_kss(const block_iq4_kss * x, int8_t * q, int32_t * scales, float * d) {
    *d = GGML_CPU_FP16_TO_FP32(x->d);
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const uint16_t * qs = x->qs + 8*ib32;
        uint8_t aux8[16];
        int ls = 0;
        for (int k = 0; k < 8; ++k) {
            uint16_t a = qs[k] & 0xfffe;
            a ^= (a >> 1);
            aux8[2*k+0] = a & 0xff;
            aux8[2*k+1] = a >> 8;
            ls |= (qs[k] & 1) << k;
        }
        scales[ib32] = (ls & 254) - 127;
        const int8_t * v = iq4k_values + 16*(ls & 1);
        for (int j = 0; j < 16; ++j) {
            q[32*ib32 + j +  0] = v[aux8[j] & 0xf];
            q[32*ib32 + j + 16] = v[aux8[j] >>  4];
        }
    }
}

inline void decode_iq2_kl(const block_iq2_kl * x, int8_t * q, int32_t * scales, float * d) {
    *d = GGML_CPU_FP16_TO_FP32(x->d);
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        scales[ib32] = int(((x->scales_l[ib32%4] >> 4*(ib32/4)) & 0xf) | (((x->scales_h >> 2*ib32) & 3) << 4)) - 32;
        const uint8_t * qs = x->qs + 16*(ib32/2);
        const int shift = 4*(ib32%2);
        for (int j = 0; j < 16; ++j) {
            const int idx = ((qs[j] >> shift) & 0xf) | (((x->qh[j] >> ib32) & 1) << 4);
            const uint16_t v = iq2kl_values[idx];
            q[32*ib32 + 2*j + 0] = (int8_t)(v & 0xff);
            q[32*ib32 + 2*j + 1] = (int8_t)(v >> 8);
        }
    }
}

template <typename block_t, int NB, void (*decode)(const block_t *, int8_t *, int32_t *, float *)>
void vec_dot_generic(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_t   * x = (const block_t *) vx;
    const block_q8_K * y = (const block_q8_K *) vy;
    constexpr int kSub = QK_K/NB;
    int8_t  q[QK_K];
    int32_t scales[NB];
    float   d;
    float sumf = 0;
    for (int i = 0; i < n/QK_K; ++i) {
        decode(x + i, q, scales, &d);
        int sumi = 0;
        for (int ib = 0; ib < NB; ++ib) {
            int sumb = 0;
            for (int j = 0; j < kSub; ++j) sumb += q[kSub*ib + j] * y[i].qs[kSub*ib + j];
            sumi += scales[ib]*sumb;
        }
        sumf += d * y[i].d * sumi;
    }
    *s = sumf;
}

template <typename block_t, void (*decode)(const block_t *, int8_t *, int32_t *, float *)>
void unpack_tiled(const void * GGML_RESTRICT vx, uint8_t * GGML_RESTRICT q, int32_t * GGML_RESTRICT scales, float * GGML_RESTRICT d) {
    int8_t aux[QK_K];
    decode((const block_t *) vx, aux, scales, d);
    for (int j = 0; j < QK_K; ++j) q[j] = (uint8_t)(aux[j] + 128);
}

#if defined(__AVX2__)

inline float hsum_float_8(__m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// sum over pairs of v*q8 as 16 x int16. |v| <= 127 and |q8| <= 127, so the pair sums fit
inline __m256i dot32_i16(__m256i v, __m256i q8) {
    return _mm256_maddubs_epi16(_mm256_sign_epi8(v, v), _mm256_sign_epi8(q8, v));
}

inline __m256i scales16(int ls1, int ls2) {
    return _mm256_set_m128i(_mm_set1_epi16((int16_t) ls2), _mm_set1_epi16((int16_t) ls1));
}

inline __m256i load_table16(const int8_t * values) {
    const __m128i t = _mm_loadu_si128((const __m128i *) values);
    return _mm256_set_m128i(t, t);
}

inline __m256i load_tables16(const int8_t * v1, const int8_t * v2) {
    return _mm256_set_m128i(_mm_loadu_si128((const __m128i *) v2), _mm_loadu_si128((const __m128i *) v1));
}

// 2 bit fields of 32 bytes: (x >> shift) & 3
inline __m256i bits2(__m256i x, int shift) {
    return _mm256_and_si256(_mm256_srl_epi16(x, _mm_cvtsi32_si128(shift)), _mm256_set1_epi8(3));
}

// bit `bit` of 32 bytes, moved to bit position `pos`
inline __m256i bit_to(__m256i x, int bit, int pos) {
    const __m256i b = _mm256_and_si256(_mm256_srl_epi16(x, _mm_cvtsi32_si128(bit)), _mm256_set1_epi8(1));
    return _mm256_sll_epi16(b, _mm_cvtsi32_si128(pos));
}

// 16 bytes of 4 bit codes: low nibbles are values 0...15, high nibbles 16...31 of the sub-block
inline __m256i nibbles(const uint8_t * qs) {
    const __m128i q4 = _mm_loadu_si128((const __m128i *) qs);
    const __m128i m4 = _mm_set1_epi8(0xf);
    return _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q4, 4), m4), _mm_and_si128(q4, m4));
}

#endif

} // namespace

extern "C" {

void quantize_row_iq2_k  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) { assert(k % QK_K == 0); quantize_iq2_k  (x, y, 1, k, NULL); }
void quantize_row_iq3_k  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) { assert(k % QK_K == 0); quantize_iq3_k  (x, y, 1, k, NULL); }
void quantize_row_iq4_k  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) { assert(k % QK_K == 0); quantize_iq4_k  (x, y, 1, k, NULL); }
void quantize_row_iq2_ks (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) { assert(k % QK_K == 0); quantize_iq2_ks (x, y, 1, k, NULL); }
void quantize_row_iq3_ks (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) { assert(k % QK_K == 0); quantize_iq3_ks (x, y, 1, k, NULL); }
void quantize_row_iq4_ks (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) { assert(k % QK_K == 0); quantize_iq4_ks (x, y, 1, k, NULL); }
void quantize_row_iq4_kss(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) { assert(k % QK_K == 0); quantize_iq4_kss(x, y, 1, k, NULL); }
void quantize_row_iq2_kl (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) { assert(k % QK_K == 0); quantize_iq2_kl (x, y, 1, k, NULL); }

#define IQK_GENERIC(name, NB) \
void ggml_vec_dot_##name##_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) { \
    assert(nrc == 1); GGML_UNUSED(nrc); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(bs); \
    vec_dot_generic<block_##name, NB, decode_##name>(n, s, vx, vy); \
} \
void ggml_iqk_unpack_##name(const void * GGML_RESTRICT vx, uint8_t * GGML_RESTRICT q, int32_t * GGML_RESTRICT scales, float * GGML_RESTRICT d) { \
    unpack_tiled<block_##name, decode_##name>(vx, q, scales, d); \
}

IQK_GENERIC(iq2_k,   16)
IQK_GENERIC(iq3_k,   16)
IQK_GENERIC(iq4_k,   16)
IQK_GENERIC(iq2_ks,   8)
IQK_GENERIC(iq3_ks,   8)
IQK_GENERIC(iq4_ks,   8)
IQK_GENERIC(iq4_kss,  8)
IQK_GENERIC(iq2_kl,   8)

#undef IQK_GENERIC

#define IQK_VEC_DOT_PROLOGUE(btype) \
    assert(n % QK_K == 0); assert(nrc == 1); \
    GGML_UNUSED(nrc); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(bs); \
    const btype      * GGML_RESTRICT x = (const btype *) vx; \
    const block_q8_K * GGML_RESTRICT y = (const block_q8_K *) vy; \
    const int nb = n / QK_K;

void ggml_vec_dot_iq2_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    IQK_VEC_DOT_PROLOGUE(block_iq2_k)
    __m256 accf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint16_t extra = x[i].extra;
        __m256i sumi = _mm256_setzero_si256();
        for (int i128 = 0; i128 < QK_K/128; ++i128) {
            const __m256i qs = _mm256_loadu_si256((const __m256i *)(x[i].qs + 32*i128));
            for (int k = 0; k < 4; ++k) {
                const int ib32 = 4*i128 + k;
                const __m256i tab = load_tables16(iq2nl_values + 4*((extra >> (2*ib32+0)) & 1), iq2nl_values + 4*((extra >> (2*ib32+1)) & 1));
                const __m256i v   = _mm256_shuffle_epi8(tab, bits2(qs, 2*k));
                const __m256i q8  = _mm256_loadu_si256((const __m256i *)(y[i].qs + 32*ib32));
                const __m256i sc  = scales16(int(x[i].scales[ib32] & 0xf) - 8, int(x[i].scales[ib32] >> 4) - 8);
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot32_i16(v, q8), sc));
            }
        }
        accf = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d), _mm256_cvtepi32_ps(sumi), accf);
    }
    *s = hsum_float_8(accf);
#else
    ggml_vec_dot_iq2_k_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq3_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    IQK_VEC_DOT_PROLOGUE(block_iq3_k)
    const __m256i tab0 = load_tables16(iq3nl_values, iq3nl_values);
    __m256 accf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint16_t extra = x[i].extra;
        const uint16_t sh    = x[i].scales_h;
        const __m256i qh = _mm256_loadu_si256((const __m256i *) x[i].qh);
        __m256i sumi = _mm256_setzero_si256();
        for (int i128 = 0; i128 < QK_K/128; ++i128) {
            const __m256i qs = _mm256_loadu_si256((const __m256i *)(x[i].qs + 32*i128));
            for (int k = 0; k < 4; ++k) {
                const int ib32 = 4*i128 + k;
                // bit 3 of the index selects the shifted half of the table
                const __m256i shift = _mm256_set_m128i(_mm_set1_epi8((char)(8*((extra >> (2*ib32+1)) & 1))),
                                                       _mm_set1_epi8((char)(8*((extra >> (2*ib32+0)) & 1))));
                const __m256i idx = _mm256_or_si256(_mm256_or_si256(bits2(qs, 2*k), bit_to(qh, ib32, 2)), shift);
                const __m256i v   = _mm256_shuffle_epi8(tab0, idx);
                const __m256i q8  = _mm256_loadu_si256((const __m256i *)(y[i].qs + 32*ib32));
                const int ls1 = 2*(x[i].scales_l[ib32] & 0xf) + 1;
                const int ls2 = 2*(x[i].scales_l[ib32] >>  4) + 1;
                const __m256i sc = scales16((sh >> (2*ib32+0)) & 1 ? -ls1 : ls1, (sh >> (2*ib32+1)) & 1 ? -ls2 : ls2);
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot32_i16(v, q8), sc));
            }
        }
        accf = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d), _mm256_cvtepi32_ps(sumi), accf);
    }
    *s = hsum_float_8(accf);
#else
    ggml_vec_dot_iq3_k_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq4_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    IQK_VEC_DOT_PROLOGUE(block_iq4_k)
    // the shifted half of iq4k_values is the first half + 4: one table, the shift is a per-lane add
    const __m256i tab = load_table16(iq4k_values);
    const __m128i four = _mm_set1_epi8(4);
    __m256 accf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint16_t extra = x[i].extra;
        int16_t ls[QK_K/16];
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const uint8_t sh = x[i].scales_h[ib32/2] >> 4*(ib32%2);
            ls[2*ib32+0] = int16_t(((x[i].scales_l[ib32] & 0xf) | ((sh << 4) & 0x30)) - 32);
            ls[2*ib32+1] = int16_t(((x[i].scales_l[ib32] >>  4) | ((sh << 2) & 0x30)) - 32);
        }
        __m256i sumi = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const __m128i add1 = _mm_and_si128(four, _mm_set1_epi8((char) -((extra >> (2*ib32+0)) & 1)));
            const __m128i add2 = _mm_and_si128(four, _mm_set1_epi8((char) -((extra >> (2*ib32+1)) & 1)));
            const __m256i v   = _mm256_add_epi8(_mm256_shuffle_epi8(tab, nibbles(x[i].qs + 16*ib32)), _mm256_set_m128i(add2, add1));
            const __m256i q8  = _mm256_loadu_si256((const __m256i *)(y[i].qs + 32*ib32));
            const __m256i sc  = scales16(ls[2*ib32+0], ls[2*ib32+1]);
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot32_i16(v, q8), sc));
        }
        accf = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d), _mm256_cvtepi32_ps(sumi), accf);
    }
    *s = hsum_float_8(accf);
#else
    ggml_vec_dot_iq4_k_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq2_ks_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    IQK_VEC_DOT_PROLOGUE(block_iq2_ks)
    __m256 accf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint16_t extra = x[i].extra;
        __m256i sumi = _mm256_setzero_si256();
        for (int i128 = 0; i128 < QK_K/128; ++i128) {
            const __m256i qs = _mm256_loadu_si256((const __m256i *)(x[i].qs + 32*i128));
            for (int k = 0; k < 4; ++k) {
                const int ib32 = 4*i128 + k;
                const __m256i tab = load_table16(iq2nl_values + 4*((extra >> ib32) & 1));
                const __m256i v   = _mm256_shuffle_epi8(tab, bits2(qs, 2*k));
                const __m256i q8  = _mm256_loadu_si256((const __m256i *)(y[i].qs + 32*ib32));
                const int ls = int(((x[i].scales[ib32/2] >> 4*(ib32%2)) & 0xf) | (((extra >> (8 + ib32)) & 1) << 4)) - 16;
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot32_i16(v, q8), _mm256_set1_epi16((int16_t) ls)));
            }
        }
        accf = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d), _mm256_cvtepi32_ps(sumi), accf);
    }
    *s = hsum_float_8(accf);
#else
    ggml_vec_dot_iq2_ks_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq3_ks_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    IQK_VEC_DOT_PROLOGUE(block_iq3_ks)
    const __m256i tab0 = load_table16(iq3nl_values);
    __m256 accf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint16_t extra = x[i].extra;
        const __m256i qh = _mm256_loadu_si256((const __m256i *) x[i].qh);
        __m256i sumi = _mm256_setzero_si256();
        for (int i128 = 0; i128 < QK_K/128; ++i128) {
            const __m256i qs = _mm256_loadu_si256((const __m256i *)(x[i].qs + 32*i128));
            for (int k = 0; k < 4; ++k) {
                const int ib32 = 4*i128 + k;
                const __m256i shift = _mm256_set1_epi8((char)(8*((extra >> (8 + ib32)) & 1)));
                const __m256i idx = _mm256_or_si256(_mm256_or_si256(bits2(qs, 2*k), bit_to(qh, ib32, 2)), shift);
                const __m256i v   = _mm256_shuffle_epi8(tab0, idx);
                const __m256i q8  = _mm256_loadu_si256((const __m256i *)(y[i].qs + 32*ib32));
                const int ls = int(((x[i].scales[ib32%4] >> 4*(ib32/4)) & 0xf) | (((extra >> ib32) & 1) << 4)) - 16;
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot32_i16(v, q8), _mm256_set1_epi16((int16_t) ls)));
            }
        }
        accf = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d), _mm256_cvtepi32_ps(sumi), accf);
    }
    *s = hsum_float_8(accf);
#else
    ggml_vec_dot_iq3_ks_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq4_ks_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    IQK_VEC_DOT_PROLOGUE(block_iq4_ks)
    __m256 accf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        __m256i sumi = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const uint8_t sc = x[i].scales[ib32];
            const __m256i tab = load_table16(iq4k_values + 16*(sc & 1));
            const __m256i v   = _mm256_shuffle_epi8(tab, nibbles(x[i].qs + 16*ib32));
            const __m256i q8  = _mm256_loadu_si256((const __m256i *)(y[i].qs + 32*ib32));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot32_i16(v, q8), _mm256_set1_epi16((int16_t)(int(sc & 254) - 127))));
        }
        accf = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d), _mm256_cvtepi32_ps(sumi), accf);
    }
    *s = hsum_float_8(accf);
#else
    ggml_vec_dot_iq4_ks_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq4_kss_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    IQK_VEC_DOT_PROLOGUE(block_iq4_kss)
    __m256 accf = _mm256_setzero_ps();
    const __m128i mask = _mm_set1_epi16((short) 0xfffe);
    for (int i = 0; i < nb; ++i) {
        __m256i sumi = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const uint16_t * qs = x[i].qs + 8*ib32;
            __m128i a = _mm_and_si128(_mm_loadu_si128((const __m128i *) qs), mask);
            a = _mm_xor_si128(a, _mm_srli_epi16(a, 1));
            int ls = 0;
            for (int k = 0; k < 8; ++k) ls |= (qs[k] & 1) << k;
            uint8_t q4[16];
            _mm_storeu_si128((__m128i *) q4, a);
            const __m256i tab = load_table16(iq4k_values + 16*(ls & 1));
            const __m256i v   = _mm256_shuffle_epi8(tab, nibbles(q4));
            const __m256i q8  = _mm256_loadu_si256((const __m256i *)(y[i].qs + 32*ib32));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot32_i16(v, q8), _mm256_set1_epi16((int16_t)((ls & 254) - 127))));
        }
        accf = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d), _mm256_cvtepi32_ps(sumi), accf);
    }
    *s = hsum_float_8(accf);
#else
    ggml_vec_dot_iq4_kss_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq2_kl_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    IQK_VEC_DOT_PROLOGUE(block_iq2_kl)
    // low and high bytes of the 32 value pairs, split in entries 0...15 and 16...31
    uint8_t lo[32], hi[32];
    for (int j = 0; j < 32; ++j) {
        lo[j] = iq2kl_values[j] & 0xff;
        hi[j] = iq2kl_values[j] >> 8;
    }
    const __m128i lo0 = _mm_loadu_si128((const __m128i *)(lo +  0)), lo1 = _mm_loadu_si128((const __m128i *)(lo + 16));
    const __m128i hi0 = _mm_loadu_si128((const __m128i *)(hi +  0)), hi1 = _mm_loadu_si128((const __m128i *)(hi + 16));
    const __m128i m4  = _mm_set1_epi8(0xf);
    const __m128i one = _mm_set1_epi8(1);
    __m256 accf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const __m128i qh = _mm_loadu_si128((const __m128i *) x[i].qh);
        __m256i sumi = _mm256_setzero_si256();
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            const __m128i qs = _mm_loadu_si128((const __m128i *)(x[i].qs + 16*ib64));
            for (int k = 0; k < 2; ++k) {
                const int ib32 = 2*ib64 + k;
                const __m128i idx  = _mm_and_si128(k ? _mm_srli_epi16(qs, 4) : qs, m4);
                const __m128i high = _mm_cmpeq_epi8(_mm_and_si128(_mm_srl_epi16(qh, _mm_cvtsi32_si128(ib32)), one), one);
                const __m128i vlo  = _mm_blendv_epi8(_mm_shuffle_epi8(lo0, idx), _mm_shuffle_epi8(lo1, idx), high);
                const __m128i vhi  = _mm_blendv_epi8(_mm_shuffle_epi8(hi0, idx), _mm_shuffle_epi8(hi1, idx), high);
                const __m256i v    = _mm256_set_m128i(_mm_unpackhi_epi8(vlo, vhi), _mm_unpacklo_epi8(vlo, vhi));
                const __m256i q8   = _mm256_loadu_si256((const __m256i *)(y[i].qs + 32*ib32));
                const int ls = int(((x[i].scales_l[ib32%4] >> 4*(ib32/4)) & 0xf) | (((x[i].scales_h >> 2*ib32) & 3) << 4)) - 32;
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot32_i16(v, q8), _mm256_set1_epi16((int16_t) ls)));
            }
        }
        accf = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d), _mm256_cvtepi32_ps(sumi), accf);
    }
    *s = hsum_float_8(accf);
#else
    ggml_vec_dot_iq2_kl_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

#undef IQK_VEC_DOT_PROLOGUE

}
