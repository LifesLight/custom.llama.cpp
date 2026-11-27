#include "fp8.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// a warp takes nrows rows and all tokens at once, so the weights are read once and each input value feeds nrows
// rows; a lane takes 4 consecutive weights per step (k and n are multiples of bs >= 32, so the 4 share one scale)
template <int ntokens, int nrows>
static __global__ void mul_mat_vec_fp8_block(
        const uint8_t * weights, const float * inputs, const float * scales, float * out, int k, int n, int bs) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE * nrows;
    const int batch = blockIdx.z;
    if (row0 >= n) {
        return;
    }

    const uint8_t * w = weights + ((int64_t) batch*n + row0)*k;
    const float * x = inputs + (int64_t) batch*ntokens*k;
    const float * s = scales + (int64_t) batch*(n/bs)*(k/bs);
    float sum[nrows][ntokens] = {};
    for (int col = 4*lane; col < k; col += 4*WARP_SIZE) {
        float4 xv[ntokens];
#pragma unroll
        for (int t = 0; t < ntokens; ++t) {
            xv[t] = *(const float4 *) (x + (int64_t) t*k + col);
        }
#pragma unroll
        for (int r = 0; r < nrows; ++r) {
            const uint32_t q = *(const uint32_t *) (w + (int64_t) r*k + col);
            const float d = s[(row0 + r)/bs*(k/bs) + col/bs];
            float wf[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wf[j] = __fmul_rn(ggml_cuda_f8_e4m3_to_fp32((q >> (8*j)) & 0xFF), d);
            }
#pragma unroll
            for (int t = 0; t < ntokens; ++t) {
                sum[r][t] = fmaf(wf[0], xv[t].x, sum[r][t]);
                sum[r][t] = fmaf(wf[1], xv[t].y, sum[r][t]);
                sum[r][t] = fmaf(wf[2], xv[t].z, sum[r][t]);
                sum[r][t] = fmaf(wf[3], xv[t].w, sum[r][t]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < nrows; ++r) {
#pragma unroll
        for (int t = 0; t < ntokens; ++t) {
            const float v = warp_reduce_sum(sum[r][t]);
            if (lane == 0) {
                out[((int64_t) batch*ntokens + t)*n + row0 + r] = v;
            }
        }
    }
}

template <int ntokens>
static void launch_mul_mat_vec_fp8_block(
        const uint8_t * weights, const float * inputs, const float * scales, float * out,
        int k, int n, int n_batch, int bs, cudaStream_t stream) {
    constexpr int nrows = 4; // divides n, a multiple of bs
    constexpr int threads = 128;
    const int warps = n/nrows;
    const dim3 grid((warps + threads/WARP_SIZE - 1)/(threads/WARP_SIZE), 1, n_batch);
    mul_mat_vec_fp8_block<ntokens, nrows><<<grid, threads, 0, stream>>>(weights, inputs, scales, out, k, n, bs);
}

static __global__ void dequantize_fp8_block_rows(
        const uint8_t * weights, const float * scales, float * out, int k, int rows, int row_start, int bs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k*rows) {
        return;
    }

    const int row = row_start + i/k;
    const int col = i % k;
    const float d = scales[row/bs*(k/bs) + col/bs];
    out[i] = ggml_cuda_f8_e4m3_to_fp32(weights[(int64_t) row*k + col]) * d;
}

void ggml_cuda_mul_mat_fp8_block(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * weight = dst->src[0];
    const ggml_tensor * input = dst->src[1];
    const ggml_tensor * scale = dst->src[2];
    const auto * weights = static_cast<const uint8_t *>(weight->data);
    const auto * inputs = static_cast<const float *>(input->data);
    const auto * scales = static_cast<const float *>(scale->data);
    auto * out = static_cast<float *>(dst->data);
    const int k = weight->ne[0];
    const int n = weight->ne[1];
    const int n_batch = weight->ne[2];
    const int tokens = input->ne[1];
    const int bs = k / scale->ne[0];

    if (tokens == 0) {
        return;
    }

    // the vector kernel reads 16-byte input and 4-byte weight words
    if (tokens <= 8 && ((uintptr_t) inputs | (uintptr_t) weights) % 16 == 0) {
        cudaStream_t stream = ctx.stream();
        switch (tokens) {
            case 1: launch_mul_mat_vec_fp8_block<1>(weights, inputs, scales, out, k, n, n_batch, bs, stream); break;
            case 2: launch_mul_mat_vec_fp8_block<2>(weights, inputs, scales, out, k, n, n_batch, bs, stream); break;
            case 3: launch_mul_mat_vec_fp8_block<3>(weights, inputs, scales, out, k, n, n_batch, bs, stream); break;
            case 4: launch_mul_mat_vec_fp8_block<4>(weights, inputs, scales, out, k, n, n_batch, bs, stream); break;
            case 5: launch_mul_mat_vec_fp8_block<5>(weights, inputs, scales, out, k, n, n_batch, bs, stream); break;
            case 6: launch_mul_mat_vec_fp8_block<6>(weights, inputs, scales, out, k, n, n_batch, bs, stream); break;
            case 7: launch_mul_mat_vec_fp8_block<7>(weights, inputs, scales, out, k, n, n_batch, bs, stream); break;
            default: launch_mul_mat_vec_fp8_block<8>(weights, inputs, scales, out, k, n, n_batch, bs, stream); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Dequantize at most 512 rows and 32 MiB at a time.
    constexpr int max_rows = 512;
    constexpr int threads = 256;
    const int chunk = std::min({ n, max_rows, (8 * 1024 * 1024) / k });
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) k*chunk);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle = ctx.cublas_handle();

    for (int b = 0; b < n_batch; ++b) {
        const uint8_t * w = weights + (int64_t) b*n*k;
        const float * s = scales + (int64_t) b*(n/bs)*(k/bs);
        const float * x = inputs + (int64_t) b*tokens*k;
        float * y = out + (int64_t) b*tokens*n;
        for (int row = 0; row < n; row += chunk) {
            const int rows = std::min(chunk, n - row);
            dequantize_fp8_block_rows<<<(k*rows + threads - 1)/threads, threads, 0, ctx.stream()>>>(
                    w, s, scratch.ptr, k, rows, row, bs);
            CUDA_CHECK(cudaGetLastError());
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, k, &alpha,
                    scratch.ptr, CUDA_R_32F, k, x, CUDA_R_32F, k,
                    &beta, y + row, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
    }
}

#endif

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080

struct fp8_abs_src {
    const float * x;
    int64_t ne0;
    int64_t ne1;
    int64_t ne2;
    int64_t s1;
    int64_t s2;
    int64_t s3;

    __device__ float operator()(int64_t i) const {
        const int64_t i0 = i % ne0;
        i /= ne0;
        const int64_t i1 = i % ne1;
        i /= ne1;
        const int64_t i2 = i % ne2;
        const int64_t i3 = i / ne2;
        const float value = fabsf(x[i0 + i1*s1 + i2*s2 + i3*s3]);
        return isfinite(value) ? value : 448.0f;
    }
};

static __global__ void fp8_amax_partials(fp8_abs_src src, int64_t ne, float * partials) {
    float amax = 0.0f;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < ne; i += (int64_t) blockDim.x*gridDim.x) {
        amax = fmaxf(amax, src(i));
    }

    __shared__ float shared[WARP_SIZE];
    amax = block_reduce<block_reduce_method::MAX, 256>(amax, shared);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = amax;
    }
}

static __global__ void fp8_amax_final(const float * partials, int n, float * amax) {
    float value = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        value = fmaxf(value, partials[i]);
    }

    __shared__ float shared[WARP_SIZE];
    value = block_reduce<block_reduce_method::MAX, 256>(value, shared);
    if (threadIdx.x == 0) {
        *amax = value;
    }
}

static __global__ void quantize_fp8_e4m3(
        const float * __restrict__ x, uint8_t * __restrict__ y, const float * __restrict__ amax,
        float * __restrict__ scale, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne, int64_t s1, int64_t s2, int64_t s3) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ne) {
        return;
    }

    int64_t tmp = i / ne0;
    const int64_t i0 = i - tmp*ne0;
    const int64_t i1 = tmp % ne1;
    tmp /= ne1;
    const int64_t i2 = tmp % ne2;
    const int64_t i3 = tmp / ne2;

    const float d = *amax > 0.0f ? *amax / 448.0f : 1.0f;
    y[i] = ggml_cuda_fp32_to_f8_e4m3(x[i0 + i1*s1 + i2*s2 + i3*s3] / d);
    if (i == 0) {
        *scale = d;
    }
}

static void fp8_set_batch(cublasLtMatrixLayout_t desc, int32_t count, int64_t stride) {
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
}

static void fp8_destroy_matmul(
        cublasLtMatmulDesc_t op_desc, cublasLtMatrixLayout_t a_desc, cublasLtMatrixLayout_t b_desc,
        cublasLtMatrixLayout_t d_desc) {
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(d_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));
}

bool ggml_cuda_mul_mat_fp8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * weight_scale) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!fp8_mma_hardware_available(cc) || src0->type != GGML_TYPE_F8_E4M3 || src1->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || src0->ne[0] % 16 != 0 || src0->ne[1] % 16 != 0 ||
            src0->nb[0] != sizeof(uint8_t) || src0->nb[1] != (size_t) src0->ne[0] || src1->nb[0] != sizeof(float) ||
            !ggml_cuda_is_aligned(src0, 16) || !ggml_cuda_is_aligned(dst, 16)) {
        return false;
    }

    GGML_ASSERT(!weight_scale || (weight_scale->type == GGML_TYPE_F32 && ggml_is_contiguous(weight_scale) && ggml_nelements(weight_scale) == 1));

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne10 == ne00);
    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    cudaStream_t stream = ctx.stream();
    const int64_t ne_src1 = ggml_nelements(src1);
    ggml_cuda_pool_alloc<uint8_t> src1_fp8(ctx.pool(), ne_src1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool(), 1);
    ggml_cuda_pool_alloc<float> src1_amax(ctx.pool(), 1);

    const fp8_abs_src abs_src = {
        (const float *) src1->data, ne10, ne11, ne12,
        (int64_t) (nb11 / sizeof(float)), (int64_t) (nb12 / sizeof(float)), (int64_t) (nb13 / sizeof(float))
    };
    const int reduce_blocks = std::min<int64_t>((ne_src1 + 255)/256, 1024);
    ggml_cuda_pool_alloc<float> reduce_tmp(ctx.pool(), reduce_blocks);
    fp8_amax_partials<<<reduce_blocks, 256, 0, stream>>>(abs_src, ne_src1, reduce_tmp.ptr);
    fp8_amax_final<<<1, 256, 0, stream>>>(reduce_tmp.ptr, reduce_blocks, src1_amax.ptr);

    quantize_fp8_e4m3<<<(ne_src1 + 255)/256, 256, 0, stream>>>(
        (const float *) src1->data, src1_fp8.ptr, src1_amax.ptr, src1_scale.ptr, ne10, ne11, ne12, ne_src1,
        nb11 / sizeof(float), nb12 / sizeof(float), nb13 / sizeof(float));
    CUDA_CHECK(cudaGetLastError());

    cublasLtMatmulDesc_t op_desc;
    cublasLtMatrixLayout_t a_desc;
    cublasLtMatrixLayout_t b_desc;
    cublasLtMatrixLayout_t d_desc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &src1_scale.ptr, sizeof(src1_scale.ptr)));
    if (weight_scale) {
        const float * scale_ptr = (const float *) weight_scale->data;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_ptr, sizeof(scale_ptr)));
    }
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    // src1 batches that share a src0 matrix are adjacent in the compact src1 copy and in dst, fold them into N
    const int64_t n = ne11*(ne02 == 1 ? r2*r3 : r2);

    // batch the remaining matmuls with a fixed src0 stride, or issue one batched call per i3 if there is none
    int64_t batch_count = ne02;
    int64_t stride_a    = nb02;
    int64_t n_calls     = ne13;
    if (ne02 == 1) {
        batch_count = ne03;
        stride_a    = nb03;
        n_calls     = 1;
    } else if (r3 == 1 && (ne03 == 1 || nb03 == ne02*nb02)) {
        batch_count = ne02*ne03;
        n_calls     = 1;
    }

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, ne00, ne01, ne00));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, ne10, n, ne10));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_32F, ne0, n, ne0));
    if (batch_count > 1) {
        fp8_set_batch(a_desc, batch_count, stride_a);
        fp8_set_batch(b_desc, batch_count, ne10*n);
        fp8_set_batch(d_desc, batch_count, ne0*n);
    }

    // TODO: Allow workspace in the heuristic search for potential performance gains.
    // cuBLASLt takes caller-owned workspace per matmul call.
    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    cublasLtMatmulHeuristicResult_t heuristic;
    int returned = 0;
    const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        ctx.cublaslt_handle(), op_desc, a_desc, b_desc, d_desc, d_desc, preference, 1, &heuristic, &returned);
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0) {
        fp8_destroy_matmul(op_desc, a_desc, b_desc, d_desc);
        return false;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int64_t i3 = 0; i3 < n_calls; ++i3) {
        const char * a = (const char *) src0->data + (i3/r3)*nb03;
        const uint8_t * b = src1_fp8.ptr + i3*ne12*ne11*ne10;
        float * d = (float *) ((char *) dst->data + i3*dst->nb[3]);
        CUBLAS_CHECK(cublasLtMatmul(ctx.cublaslt_handle(), op_desc, &alpha, a, a_desc, b, b_desc,
            &beta, d, d_desc, d, d_desc, &heuristic.algo, nullptr, 0, stream));
    }

    fp8_destroy_matmul(op_desc, a_desc, b_desc, d_desc);
    return true;
}

#else

bool ggml_cuda_mul_mat_fp8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * weight_scale) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst, weight_scale);
    return false;
}

#endif
