#pragma once

#include "common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
void ggml_cuda_mul_mat_fp8_block(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif

bool ggml_cuda_mul_mat_fp8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * weight_scale = nullptr);
