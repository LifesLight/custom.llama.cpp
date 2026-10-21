// CUDA kernels for the trellis quants IQ3_KT (see ggml-kt.cpp)
// Ported from ik_llama.cpp (https://github.com/ikawrakow/ik_llama.cpp), Copyright (C) 2025 Iwan Kawrakow, MIT license

#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"

#include <cstdint>

#define VDR_IQ3_KT_Q8_1_MMVQ 4

static constexpr uint32_t KT_IQ3_OFFSET  = 4096;

// next value of the integer trellis, result is in [-126, 126]
static __device__ __forceinline__ int kt_next_int(uint32_t & val) {
    val *= 0xCBAC1FEDu;
    return ggml_cuda_dp4a(int(val & 0x3f3f3f3fu), 0x01010101, -126);
}

// 4 absolute trellis values with signs from bit ib of each byte of signs, packed as int8
static __device__ __forceinline__ int kt_values4_signed(uint32_t & val, const uint32_t signs, const int ib) {
    int v4 = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int s = -int((signs >> (8*k + ib)) & 1);
        const int v = (abs(kt_next_int(val)) ^ s) - s;
        v4 |= (v & 0xff) << 8*k;
    }
    return v4;
}

static __device__ __forceinline__ float vec_dot_iq3_kt_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_kt * bq3 = (const block_iq3_kt *) vbq + kbx;

    // iqs is 0...28, one sub-block of 32 per call
    const int ib32 = iqs/4;
    const int ls   = (bq3->scales[ib32%4] >> 4*(ib32/4)) & 0xf;
    const uint16_t * ql = (const uint16_t *) bq3->ql + 4*ib32;
    const int * q8 = (const int *) bq8_1[ib32].qs;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t val = ql[j] + KT_IQ3_OFFSET;
        sumi = ggml_cuda_dp4a(kt_values4_signed(val, get_int_b2(bq3->qh, 2*j+0), ib32), q8[2*j+0], sumi);
        sumi = ggml_cuda_dp4a(kt_values4_signed(val, get_int_b2(bq3->qh, 2*j+1), ib32), q8[2*j+1], sumi);
    }
    return __half2float(bq3->d) * GGML_KT_D_SCALE * ls * __low2float(bq8_1[ib32].ds) * sumi;
}

// one CUDA block of 32 threads per block of 256, each thread writes 8 values
template<typename dst_t>
static __global__ void dequantize_block_iq3_kt(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const int     tid = threadIdx.x;
    const block_iq3_kt * x = (const block_iq3_kt *) vx + i;

    const int ib = tid/4;
    const float dl = __half2float(x->d) * GGML_KT_D_SCALE * ((x->scales[ib%4] >> 4*(ib/4)) & 0xf);
    uint32_t val = ((const uint16_t *) x->ql)[tid] + KT_IQ3_OFFSET;
    dst_t * y = yy + i*QK_K + 8*tid;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float v = dl * abs(kt_next_int(val));
        y[j] = ggml_cuda_cast<dst_t>((x->qh[8*(tid%4) + j] >> ib) & 1 ? -v : v);
    }
}

template<typename dst_t>
static void dequantize_row_iq3_kt_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int64_t nb = k / QK_K;
    dequantize_block_iq3_kt<<<nb, 32, 0, stream>>>(vx, y);
}
