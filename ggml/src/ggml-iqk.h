// IQ3_KT and IQ_K quants, ported from ik_llama.cpp (https://github.com/ikawrakow/ik_llama.cpp),
// Copyright (C) 2024-2025 Iwan Kawrakow, MIT license. Implemented in ggml-kt.cpp and ggml-iqk.cpp.
// Included from ggml-quants.h. The macros below keep the hooks in ggml.c and ggml-quants.c to one line each.
#pragma once

// trellis quants, implemented in ggml-kt.cpp
GGML_API void quantize_row_iq3_kt_ref(const float * GGML_RESTRICT x, block_iq3_kt * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_kt(const block_iq3_kt * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq3_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// IQ_K quants (ggml-iqk.cpp)
GGML_API void quantize_row_iq2_k_ref(const float * GGML_RESTRICT x, block_iq2_k * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq3_k_ref(const float * GGML_RESTRICT x, block_iq3_k * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_k_ref(const float * GGML_RESTRICT x, block_iq4_k * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_ks_ref(const float * GGML_RESTRICT x, block_iq2_ks * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq3_ks_ref(const float * GGML_RESTRICT x, block_iq3_ks * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_ks_ref(const float * GGML_RESTRICT x, block_iq4_ks * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_kss_ref(const float * GGML_RESTRICT x, block_iq4_kss * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_kl_ref(const float * GGML_RESTRICT x, block_iq2_kl * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_k(const block_iq2_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_k(const block_iq3_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_k(const block_iq4_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_ks(const block_iq2_ks * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_ks(const block_iq3_ks * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_ks(const block_iq4_ks * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_kss(const block_iq4_kss * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_kl(const block_iq2_kl * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq2_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_kss(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_kl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// entries of type_traits[] in ggml.c
#define GGML_IQK_TYPE_TRAITS \
    [GGML_TYPE_IQ3_KT] = { .type_name = "iq3_kt", .blck_size = QK_K, .type_size = sizeof(block_iq3_kt), .is_quantized = true, .to_float = (ggml_to_float_t) dequantize_row_iq3_kt, .from_float_ref = (ggml_from_float_t) quantize_row_iq3_kt_ref, }, \
    [GGML_TYPE_IQ2_K] = { .type_name = "iq2_k", .blck_size = QK_K, .type_size = sizeof(block_iq2_k), .is_quantized = true, .to_float = (ggml_to_float_t) dequantize_row_iq2_k, .from_float_ref = (ggml_from_float_t) quantize_row_iq2_k_ref, }, \
    [GGML_TYPE_IQ3_K] = { .type_name = "iq3_k", .blck_size = QK_K, .type_size = sizeof(block_iq3_k), .is_quantized = true, .to_float = (ggml_to_float_t) dequantize_row_iq3_k, .from_float_ref = (ggml_from_float_t) quantize_row_iq3_k_ref, }, \
    [GGML_TYPE_IQ4_K] = { .type_name = "iq4_k", .blck_size = QK_K, .type_size = sizeof(block_iq4_k), .is_quantized = true, .to_float = (ggml_to_float_t) dequantize_row_iq4_k, .from_float_ref = (ggml_from_float_t) quantize_row_iq4_k_ref, }, \
    [GGML_TYPE_IQ2_KS] = { .type_name = "iq2_ks", .blck_size = QK_K, .type_size = sizeof(block_iq2_ks), .is_quantized = true, .to_float = (ggml_to_float_t) dequantize_row_iq2_ks, .from_float_ref = (ggml_from_float_t) quantize_row_iq2_ks_ref, }, \
    [GGML_TYPE_IQ3_KS] = { .type_name = "iq3_ks", .blck_size = QK_K, .type_size = sizeof(block_iq3_ks), .is_quantized = true, .to_float = (ggml_to_float_t) dequantize_row_iq3_ks, .from_float_ref = (ggml_from_float_t) quantize_row_iq3_ks_ref, }, \
    [GGML_TYPE_IQ4_KS] = { .type_name = "iq4_ks", .blck_size = QK_K, .type_size = sizeof(block_iq4_ks), .is_quantized = true, .to_float = (ggml_to_float_t) dequantize_row_iq4_ks, .from_float_ref = (ggml_from_float_t) quantize_row_iq4_ks_ref, }, \
    [GGML_TYPE_IQ4_KSS] = { .type_name = "iq4_kss", .blck_size = QK_K, .type_size = sizeof(block_iq4_kss), .is_quantized = true, .to_float = (ggml_to_float_t) dequantize_row_iq4_kss, .from_float_ref = (ggml_from_float_t) quantize_row_iq4_kss_ref, }, \
    [GGML_TYPE_IQ2_KL] = { .type_name = "iq2_kl", .blck_size = QK_K, .type_size = sizeof(block_iq2_kl), .is_quantized = true, .to_float = (ggml_to_float_t) dequantize_row_iq2_kl, .from_float_ref = (ggml_from_float_t) quantize_row_iq2_kl_ref, },

// cases of ggml_validate_row_data() in ggml-quants.c
#define GGML_IQK_VALIDATE_CASES \
    case GGML_TYPE_IQ3_KT: { VALIDATE_ROW_DATA_D_F16_IMPL(block_iq3_kt, data, nb); } break; \
    case GGML_TYPE_IQ2_K: { VALIDATE_ROW_DATA_D_F16_IMPL(block_iq2_k, data, nb); } break; \
    case GGML_TYPE_IQ3_K: { VALIDATE_ROW_DATA_D_F16_IMPL(block_iq3_k, data, nb); } break; \
    case GGML_TYPE_IQ4_K: { VALIDATE_ROW_DATA_D_F16_IMPL(block_iq4_k, data, nb); } break; \
    case GGML_TYPE_IQ2_KS: { VALIDATE_ROW_DATA_D_F16_IMPL(block_iq2_ks, data, nb); } break; \
    case GGML_TYPE_IQ3_KS: { VALIDATE_ROW_DATA_D_F16_IMPL(block_iq3_ks, data, nb); } break; \
    case GGML_TYPE_IQ4_KS: { VALIDATE_ROW_DATA_D_F16_IMPL(block_iq4_ks, data, nb); } break; \
    case GGML_TYPE_IQ4_KSS: { VALIDATE_ROW_DATA_D_F16_IMPL(block_iq4_kss, data, nb); } break; \
    case GGML_TYPE_IQ2_KL: { VALIDATE_ROW_DATA_D_F16_IMPL(block_iq2_kl, data, nb); } break;

// cases of ggml_quantize_chunk() in ggml.c
#define GGML_IQK_QUANTIZE_CASES \
    case GGML_TYPE_IQ3_KT: result = quantize_iq3_kt(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break; \
    case GGML_TYPE_IQ2_K: result = quantize_iq2_k(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break; \
    case GGML_TYPE_IQ3_K: result = quantize_iq3_k(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break; \
    case GGML_TYPE_IQ4_K: result = quantize_iq4_k(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break; \
    case GGML_TYPE_IQ2_KS: result = quantize_iq2_ks(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break; \
    case GGML_TYPE_IQ3_KS: result = quantize_iq3_ks(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break; \
    case GGML_TYPE_IQ4_KS: result = quantize_iq4_ks(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break; \
    case GGML_TYPE_IQ4_KSS: result = quantize_iq4_kss(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break; \
    case GGML_TYPE_IQ2_KL: result = quantize_iq2_kl(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
