// CPU code for the IQ3_KT and IQ_K quants (kt.cpp, iqk.cpp), ported from ik_llama.cpp
// (https://github.com/ikawrakow/ik_llama.cpp), Copyright (C) 2024-2025 Iwan Kawrakow, MIT license. Included from quants.h.
#pragma once

void quantize_row_iq3_kt (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq2_k  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq3_k  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq4_k  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq2_ks (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq3_ks (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq4_ks (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq4_kss(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq2_kl (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void ggml_vec_dot_iq3_kt_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_ks_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_ks_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_ks_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_kss_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_kl_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
// decode one trellis block of 256 for the tiled matmul: q = value + 128, int scale per 32, block scale d
void ggml_kt_unpack_iq3_kt(const void * GGML_RESTRICT x, uint8_t * GGML_RESTRICT q, int32_t * GGML_RESTRICT scales, float * GGML_RESTRICT d);
void ggml_vec_dot_iq3_kt_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// IQ_K quants (iqk.cpp): unpack one block of 256 for the tiled matmul, codes are value + 128
void ggml_iqk_unpack_iq2_k(const void * GGML_RESTRICT x, uint8_t * GGML_RESTRICT q, int32_t * GGML_RESTRICT scales, float * GGML_RESTRICT d);
void ggml_iqk_unpack_iq3_k(const void * GGML_RESTRICT x, uint8_t * GGML_RESTRICT q, int32_t * GGML_RESTRICT scales, float * GGML_RESTRICT d);
void ggml_iqk_unpack_iq4_k(const void * GGML_RESTRICT x, uint8_t * GGML_RESTRICT q, int32_t * GGML_RESTRICT scales, float * GGML_RESTRICT d);
void ggml_iqk_unpack_iq2_ks(const void * GGML_RESTRICT x, uint8_t * GGML_RESTRICT q, int32_t * GGML_RESTRICT scales, float * GGML_RESTRICT d);
void ggml_iqk_unpack_iq3_ks(const void * GGML_RESTRICT x, uint8_t * GGML_RESTRICT q, int32_t * GGML_RESTRICT scales, float * GGML_RESTRICT d);
void ggml_iqk_unpack_iq4_ks(const void * GGML_RESTRICT x, uint8_t * GGML_RESTRICT q, int32_t * GGML_RESTRICT scales, float * GGML_RESTRICT d);
void ggml_iqk_unpack_iq4_kss(const void * GGML_RESTRICT x, uint8_t * GGML_RESTRICT q, int32_t * GGML_RESTRICT scales, float * GGML_RESTRICT d);
void ggml_iqk_unpack_iq2_kl(const void * GGML_RESTRICT x, uint8_t * GGML_RESTRICT q, int32_t * GGML_RESTRICT scales, float * GGML_RESTRICT d);
void ggml_vec_dot_iq2_k_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_k_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_k_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_ks_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_ks_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_ks_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_kss_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_kl_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// entries of type_traits_cpu[] in ggml-cpu.c
#define GGML_IQK_CPU_TRAITS \
    [GGML_TYPE_IQ3_KT] = { .from_float = quantize_row_iq3_kt, .vec_dot = ggml_vec_dot_iq3_kt_q8_K, .vec_dot_type = GGML_TYPE_Q8_K, .nrows = 1, }, \
    [GGML_TYPE_IQ2_K] = { .from_float = quantize_row_iq2_k, .vec_dot = ggml_vec_dot_iq2_k_q8_K, .vec_dot_type = GGML_TYPE_Q8_K, .nrows = 1, }, \
    [GGML_TYPE_IQ3_K] = { .from_float = quantize_row_iq3_k, .vec_dot = ggml_vec_dot_iq3_k_q8_K, .vec_dot_type = GGML_TYPE_Q8_K, .nrows = 1, }, \
    [GGML_TYPE_IQ4_K] = { .from_float = quantize_row_iq4_k, .vec_dot = ggml_vec_dot_iq4_k_q8_K, .vec_dot_type = GGML_TYPE_Q8_K, .nrows = 1, }, \
    [GGML_TYPE_IQ2_KS] = { .from_float = quantize_row_iq2_ks, .vec_dot = ggml_vec_dot_iq2_ks_q8_K, .vec_dot_type = GGML_TYPE_Q8_K, .nrows = 1, }, \
    [GGML_TYPE_IQ3_KS] = { .from_float = quantize_row_iq3_ks, .vec_dot = ggml_vec_dot_iq3_ks_q8_K, .vec_dot_type = GGML_TYPE_Q8_K, .nrows = 1, }, \
    [GGML_TYPE_IQ4_KS] = { .from_float = quantize_row_iq4_ks, .vec_dot = ggml_vec_dot_iq4_ks_q8_K, .vec_dot_type = GGML_TYPE_Q8_K, .nrows = 1, }, \
    [GGML_TYPE_IQ4_KSS] = { .from_float = quantize_row_iq4_kss, .vec_dot = ggml_vec_dot_iq4_kss_q8_K, .vec_dot_type = GGML_TYPE_Q8_K, .nrows = 1, }, \
    [GGML_TYPE_IQ2_KL] = { .from_float = quantize_row_iq2_kl, .vec_dot = ggml_vec_dot_iq2_kl_q8_K, .vec_dot_type = GGML_TYPE_Q8_K, .nrows = 1, },
