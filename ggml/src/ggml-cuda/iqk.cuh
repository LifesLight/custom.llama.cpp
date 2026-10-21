// CUDA kernels for the IQ_K quants (IQ2_K, IQ3_K, IQ4_K, IQ2_KS, IQ3_KS, IQ4_KS, IQ4_KSS, IQ2_KL), see ggml-iqk.cpp
// The MMVQ dot products are ported from ik_llama.cpp (https://github.com/ikawrakow/ik_llama.cpp),
// Copyright (C) 2024-2025 Iwan Kawrakow, MIT license. The KS/KSS/KL types read their scale from the block instead of the row.

#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"
#include "kt.cuh"

#include <cstdint>

#define VDR_IQ2_K_Q8_1_MMVQ   4
#define VDR_IQ3_K_Q8_1_MMVQ   4
#define VDR_IQ4_K_Q8_1_MMVQ   4
#define VDR_IQ2_KS_Q8_1_MMVQ  4
#define VDR_IQ3_KS_Q8_1_MMVQ  4
#define VDR_IQ4_KS_Q8_1_MMVQ  4
#define VDR_IQ4_KSS_Q8_1_MMVQ 4
#define VDR_IQ2_KL_Q8_1_MMVQ  4

// all MMVQ dot products below: iqs is 0, 4, ..., 28, 8 threads per block of 256

static __device__ __forceinline__ float vec_dot_iq2_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    // i4/4 selects the half of the block (4 q8 blocks of 32), i4%4 the 8 quants within each of the 4 sub-blocks
    const int i4 = iqs/4;
    const int32_t * q8_1 = (const int *)bq8_1[4*(i4/4)+0].qs + 2*(i4%4);
    const int32_t * q8_2 = (const int *)bq8_1[4*(i4/4)+1].qs + 2*(i4%4);
    const int32_t * q8_3 = (const int *)bq8_1[4*(i4/4)+2].qs + 2*(i4%4);
    const int32_t * q8_4 = (const int *)bq8_1[4*(i4/4)+3].qs + 2*(i4%4);

    const block_iq2_k * bq2 = (const block_iq2_k *) vbq + kbx;
    const uint32_t * q2 = (const uint32_t *)bq2->qs + 8*(i4/4) + 2*(i4%4);
    const uint16_t extra = bq2->extra >> (8*(i4/4) + (i4%4)/2);

    const uint32_t * scales = (const uint32_t *)bq2->scales;
    const uint32_t s32 = __vsub4((scales[i4/4] >> 4*(((i4%4)/2)%2)) & 0x0f0f0f0f, 0x08080808);
    const int8_t * s8 = (const int8_t *)&s32;

    const uint32_t extra32   = uint32_t(extra & 0xff) * 0x01010101;
    const uint32_t extra32_1 = (extra32 << 2) & 0x44444444;
    const uint32_t extra32_2 = (extra32 << 0) & 0x44444444;

    uint32_t val1 = ((q2[0] >> 0) & 0x33333333) | extra32_1;
    uint32_t val2 = ((q2[1] >> 0) & 0x33333333) | extra32_1;
    int2 v1 = get_int_from_table_16(val1, iq2nl_values);
    int2 v2 = get_int_from_table_16(val2, iq2nl_values);
    const int sumi1 = ggml_cuda_dp4a(v2.x, q8_1[1], ggml_cuda_dp4a(v1.x, q8_1[0], 0)) * s8[0];
    const int sumi3 = ggml_cuda_dp4a(v2.y, q8_3[1], ggml_cuda_dp4a(v1.y, q8_3[0], 0)) * s8[2];

    val1 = ((q2[0] >> 2) & 0x33333333) | extra32_2;
    val2 = ((q2[1] >> 2) & 0x33333333) | extra32_2;
    v1 = get_int_from_table_16(val1, iq2nl_values);
    v2 = get_int_from_table_16(val2, iq2nl_values);
    const int sumi2 = ggml_cuda_dp4a(v2.x, q8_2[1], ggml_cuda_dp4a(v1.x, q8_2[0], 0)) * s8[1];
    const int sumi4 = ggml_cuda_dp4a(v2.y, q8_4[1], ggml_cuda_dp4a(v1.y, q8_4[0], 0)) * s8[3];

    return __half2float(bq2->d) * (__low2float(bq8_1[4*(i4/4)+0].ds) * sumi1
                                +  __low2float(bq8_1[4*(i4/4)+1].ds) * sumi2
                                +  __low2float(bq8_1[4*(i4/4)+2].ds) * sumi3
                                +  __low2float(bq8_1[4*(i4/4)+3].ds) * sumi4);
}

static __device__ __forceinline__ float vec_dot_iq2_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_ks * bq2 = (const block_iq2_ks *) vbq + kbx;
    const float scale = __half2float(bq2->d);

    const int i4 = iqs/4;
    const int32_t * q8_1 = (const int *)bq8_1[4*(i4/4)+0].qs + 2*(i4%4);
    const int32_t * q8_2 = (const int *)bq8_1[4*(i4/4)+1].qs + 2*(i4%4);
    const int32_t * q8_3 = (const int *)bq8_1[4*(i4/4)+2].qs + 2*(i4%4);
    const int32_t * q8_4 = (const int *)bq8_1[4*(i4/4)+3].qs + 2*(i4%4);

    const uint16_t * q2 = (const uint16_t *)bq2->qs + 16*(i4/4) + 4*(i4%4);
    const uint16_t extra = bq2->extra >> 4*(i4/4);

    const uint32_t val1 = q2[0] | (q2[1] << 16), val2 = q2[2] | (q2[3] << 16);

    const uint16_t * scales16 = (const uint16_t *)bq2->scales;
    int32_t scales32 = __vsub4((scales16[i4/4] | (scales16[i4/4] << 12)) & 0x0f0f0f0f, 0x10101010);
    int8_t * s8 = (int8_t *)&scales32;
    s8[0] += ((extra >> 4) & 0x10);
    s8[1] += ((extra >> 6) & 0x10);
    s8[2] += ((extra >> 5) & 0x10);
    s8[3] += ((extra >> 7) & 0x10);

    const uint32_t extra32 = uint32_t(extra & 0xf) * 0x01010101;

    uint32_t this_extra = ((extra32 << 2) & 0x04040404) | ((extra32 << 4) & 0x40404040);
    uint32_t idx1 = ((val1 >> 0) & 0x33333333) | this_extra;
    uint32_t idx2 = ((val2 >> 0) & 0x33333333) | this_extra;
    int2 v1 = get_int_from_table_16(idx1, iq2nl_values);
    int2 v2 = get_int_from_table_16(idx2, iq2nl_values);

    const int sumi1 = ggml_cuda_dp4a(v2.x, q8_1[1], ggml_cuda_dp4a(v1.x, q8_1[0], 0)) * s8[0];
    const int sumi3 = ggml_cuda_dp4a(v2.y, q8_3[1], ggml_cuda_dp4a(v1.y, q8_3[0], 0)) * s8[1];

    this_extra = ((extra32 << 1) & 0x04040404) | ((extra32 << 3) & 0x40404040);
    idx1 = ((val1 >> 2) & 0x33333333) | this_extra;
    idx2 = ((val2 >> 2) & 0x33333333) | this_extra;
    v1 = get_int_from_table_16(idx1, iq2nl_values);
    v2 = get_int_from_table_16(idx2, iq2nl_values);

    const int sumi2 = ggml_cuda_dp4a(v2.x, q8_2[1], ggml_cuda_dp4a(v1.x, q8_2[0], 0)) * s8[2];
    const int sumi4 = ggml_cuda_dp4a(v2.y, q8_4[1], ggml_cuda_dp4a(v1.y, q8_4[0], 0)) * s8[3];

    return scale * (__low2float(bq8_1[4*(i4/4)+0].ds) * sumi1
                 +  __low2float(bq8_1[4*(i4/4)+1].ds) * sumi2
                 +  __low2float(bq8_1[4*(i4/4)+2].ds) * sumi3
                 +  __low2float(bq8_1[4*(i4/4)+3].ds) * sumi4);
}

static __device__ __forceinline__ float vec_dot_iq2_kl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs) {

    const block_iq2_kl * bq2 = (const block_iq2_kl *) vbq + kbx;
    const float d = __half2float(bq2->d);

    const int iqs  = iiqs/4;
    const int ib64 = iqs/2;  // 0...3: quants 64*ib64 ... 64*ib64+63
    const int il16 = iqs%2;  // 0...1: 16 quants in each of the 2 sub-blocks of 32

    const uint16_t * ql = (const uint16_t *)bq2->qs + 8*ib64 + 4*il16;
    const uint16_t * qh = (const uint16_t *)bq2->qh + 4*il16;

    int32_t aux32;
    const uint8_t * aux8 = (const uint8_t *)&aux32;

    const int * q8l = (const int *)bq8_1[2*ib64+0].qs + 4*il16;
    const int * q8h = (const int *)bq8_1[2*ib64+1].qs + 4*il16;

    int sumi1 = 0, sumi2 = 0;
    int v1, v2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t vl =  ql[2*i+0] | (ql[2*i+1] << 16);
        const uint32_t vh = (qh[2*i+0] | (qh[2*i+1] << 16)) >> 2*ib64;

        aux32 = (vl & 0x0f0f0f0f) | ((vh << 4) & 0x10101010);
        v1 = iq2kl_values[aux8[0]] | (iq2kl_values[aux8[1]] << 16);
        v2 = iq2kl_values[aux8[2]] | (iq2kl_values[aux8[3]] << 16);
        sumi1 = ggml_cuda_dp4a(v1, q8l[2*i+0], ggml_cuda_dp4a(v2, q8l[2*i+1], sumi1));

        aux32 = ((vl >> 4) & 0x0f0f0f0f) | ((vh << 3) & 0x10101010);
        v1 = iq2kl_values[aux8[0]] | (iq2kl_values[aux8[1]] << 16);
        v2 = iq2kl_values[aux8[2]] | (iq2kl_values[aux8[3]] << 16);
        sumi2 = ggml_cuda_dp4a(v1, q8h[2*i+0], ggml_cuda_dp4a(v2, q8h[2*i+1], sumi2));
    }

    const int sh  = bq2->scales_h >> 4*ib64;
    const int ls1 = int(((bq2->scales_l[(2*ib64+0)%4] >> 4*(ib64/2)) & 0xf) | ((sh << 4) & 0x30)) - 32;
    const int ls2 = int(((bq2->scales_l[(2*ib64+1)%4] >> 4*(ib64/2)) & 0xf) | ((sh << 2) & 0x30)) - 32;

    return d * (__low2float(bq8_1[2*ib64+0].ds) * ls1 * sumi1 + __low2float(bq8_1[2*ib64+1].ds) * ls2 * sumi2);
}

static __device__ __forceinline__ float vec_dot_iq3_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs) {

    const block_iq3_k * bq3 = (const block_iq3_k *) vbq + kbx;

    const int iqs   = iiqs/4;
    const int ib128 = iqs/4;  // 0 or 1: quants 0...127 or 128...255
    const int il8   = iqs%4;  // 8 quants in each of the 4 sub-blocks of 32
    const int shift = 4*(il8/2);

    const uint16_t * ql = (const uint16_t *)bq3->qs + 16*ib128 + 4*il8;
    const uint16_t * qh = (const uint16_t *)bq3->qh + 4*il8;

    uint32_t aux32;
    const uint8_t * aux8 = (const uint8_t *)&aux32;

    const int hshift = 4*(1-ib128);
    const uint16_t sh = bq3->scales_h >> (8*ib128 + il8/2);

    const uint8_t extra = bq3->extra >> (8*ib128 + il8/2);
    const uint32_t extra32   = uint32_t(extra) * 0x01010101;
    const uint32_t extra32_1 = (extra32 << 3) & 0x88888888;
    const uint32_t extra32_2 = (extra32 << 1) & 0x88888888;

    const int * q8;
    int sumi[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t vl = ql[2*i+0] | (ql[2*i+1] << 16);
        const uint32_t vh = ((qh[2*i+0] | (qh[2*i+1] << 16)) << hshift);

        const uint32_t val1 = ((vl >> 0) & 0x33333333) | extra32_1 | ((vh >> 2) & 0x04040404) | ((vh >> 0) & 0x40404040);
        const uint32_t val2 = ((vl >> 2) & 0x33333333) | extra32_2 | ((vh >> 3) & 0x04040404) | ((vh >> 1) & 0x40404040);
        const int2 v1 = get_int_from_table_16(val1, iq3nl_values);
        const int2 v2 = get_int_from_table_16(val2, iq3nl_values);

        q8 = (const int *)bq8_1[4*ib128+0].qs + 2*il8;
        sumi[0] = ggml_cuda_dp4a(v1.x, q8[i], sumi[0]);

        q8 += sizeof(block_q8_1)/4;
        sumi[1] = ggml_cuda_dp4a(v2.x, q8[i], sumi[1]);

        q8 += sizeof(block_q8_1)/4;
        sumi[2] = ggml_cuda_dp4a(v1.y, q8[i], sumi[2]);

        q8 += sizeof(block_q8_1)/4;
        sumi[3] = ggml_cuda_dp4a(v2.y, q8[i], sumi[3]);
    }
    const float d = __half2float(bq3->d);
    const uint16_t * sl16 = (const uint16_t *)bq3->scales_l + 2*ib128;
    aux32 = ((((sl16[0] | (sl16[1] << 16)) >> shift) & 0x0f0f0f0f) << 1) | 0x01010101;
    return d * (__low2float(bq8_1[4*ib128+0].ds) * aux8[0] * (sh & 0x01 ? -1 : 1) * sumi[0] +
                __low2float(bq8_1[4*ib128+1].ds) * aux8[1] * (sh & 0x04 ? -1 : 1) * sumi[1] +
                __low2float(bq8_1[4*ib128+2].ds) * aux8[2] * (sh & 0x10 ? -1 : 1) * sumi[2] +
                __low2float(bq8_1[4*ib128+3].ds) * aux8[3] * (sh & 0x40 ? -1 : 1) * sumi[3]);
}

static __device__ __forceinline__ float vec_dot_iq3_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs) {

    const block_iq3_ks * bq3 = (const block_iq3_ks *) vbq + kbx;
    const float d = __half2float(bq3->d);

    const int iqs   = iiqs/4;
    const int ib128 = iqs/4;  // 0 or 1: quants 0...127 or 128...255
    const int il8   = iqs%4;  // 8 quants in each of the 4 sub-blocks of 32

    const uint16_t * ql = (const uint16_t *)bq3->qs + 16*ib128 + 4*il8;
    const uint16_t * qh = (const uint16_t *)bq3->qh + 4*il8;

    const uint16_t extra = bq3->extra >> 4*ib128;
    const uint32_t extra_v = uint32_t(extra >> 8) * 0x01010101;

    const uint32_t extra32_1 = ((extra_v << 3) & 0x08080808) | ((extra_v << 5) & 0x80808080);
    const uint32_t extra32_2 = ((extra_v << 2) & 0x08080808) | ((extra_v << 4) & 0x80808080);

    const int * q8;
    int sumi[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t vl = ql[2*i+0] | (ql[2*i+1] << 16);
        const uint32_t vh = ((qh[2*i+0] | (qh[2*i+1] << 16)) >> 4*ib128);

        const uint32_t val1 = ((vl >> 0) & 0x33333333) | extra32_1 | ((vh << 2) & 0x04040404) | ((vh << 4) & 0x40404040);
        const uint32_t val2 = ((vl >> 2) & 0x33333333) | extra32_2 | ((vh << 1) & 0x04040404) | ((vh << 3) & 0x40404040);
        const int2 v1 = get_int_from_table_16(val1, iq3nl_values);
        const int2 v2 = get_int_from_table_16(val2, iq3nl_values);

        q8 = (const int *)bq8_1[4*ib128+0].qs + 2*il8;
        sumi[0] = ggml_cuda_dp4a(v1.x, q8[i], sumi[0]);

        q8 += sizeof(block_q8_1)/4;
        sumi[1] = ggml_cuda_dp4a(v2.x, q8[i], sumi[1]);

        q8 += sizeof(block_q8_1)/4;
        sumi[2] = ggml_cuda_dp4a(v1.y, q8[i], sumi[2]);

        q8 += sizeof(block_q8_1)/4;
        sumi[3] = ggml_cuda_dp4a(v2.y, q8[i], sumi[3]);
    }
    const uint16_t * sl16 = (const uint16_t *)bq3->scales;
    int32_t aux32 = __vsub4(((sl16[0] | (sl16[1] << 16)) >> 4*ib128) & 0x0f0f0f0f, 0x10101010);
    const int8_t * a8 = (const int8_t *)&aux32;
    return d * (__low2float(bq8_1[4*ib128+0].ds) * (a8[0] + ((extra << 4) & 0x10)) * sumi[0] +
                __low2float(bq8_1[4*ib128+1].ds) * (a8[1] + ((extra << 3) & 0x10)) * sumi[1] +
                __low2float(bq8_1[4*ib128+2].ds) * (a8[2] + ((extra << 2) & 0x10)) * sumi[2] +
                __low2float(bq8_1[4*ib128+3].ds) * (a8[3] + ((extra << 1) & 0x10)) * sumi[3]);
}

static __device__ __forceinline__ float vec_dot_iq4_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_k * bq4 = (const block_iq4_k *) vbq + kbx;

    const int ib32 = iqs/4;
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const uint16_t * q4 = (const uint16_t *)bq4->qs + 8*ib32;
    const uint16_t extra = bq4->extra >> 2*ib32;
    // the shifted half of iq4k_values is the first half + 4, no value reaches 0xFC: select it with a packed byte add
    const int add1 = 0x04040404 & -( extra       & 1);
    const int add2 = 0x04040404 & -((extra >> 1) & 1);
    int sumi1 = 0, sumi2 = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux32 = q4[2*j+0] | (q4[2*j+1] << 16);
        const int2 v = get_int_from_table_16(aux32, kvalues_iq4nl);
        sumi1 = ggml_cuda_dp4a(v.x + add1, q8[j+0], sumi1);
        sumi2 = ggml_cuda_dp4a(v.y + add2, q8[j+4], sumi2);
    }
    const float d = __half2float(bq4->d) * __low2float(bq8_1[ib32].ds);
    const uint8_t sh = bq4->scales_h[ib32/2] >> 4*(ib32%2);
    const int ls1 = ((bq4->scales_l[ib32] & 0xf) | ((sh << 4) & 0x30)) - 32;
    const int ls2 = ((bq4->scales_l[ib32] >>  4) | ((sh << 2) & 0x30)) - 32;
    return d * (sumi1 * ls1 + sumi2 * ls2);
}

static __device__ __forceinline__ float vec_dot_iq4_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_ks * bq4 = (const block_iq4_ks *) vbq + kbx;

    const int ib32 = iqs/4;
    const int32_t * q8 = (const int *)bq8_1[ib32].qs;
    const float dl = __half2float(bq4->d) * ((bq4->scales[ib32] & 254) - 127);
    const int8_t * values = iq4k_values + ((bq4->scales[ib32] & 1) << 4);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        // qs is only 2 byte aligned in this block
        const int2 v = get_int_from_table_16(get_int_b2(bq4->qs, 4*ib32 + j), values);
        sumi = ggml_cuda_dp4a(v.x, q8[j+0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[j+4], sumi);
    }
    return dl * __low2float(bq8_1[ib32].ds) * sumi;
}

static __device__ __forceinline__ float vec_dot_iq4_kss_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_kss * bq4 = (const block_iq4_kss *) vbq + kbx;

    const int ib32 = iqs/4;
    const int32_t * q8 = (const int *)bq8_1[ib32].qs;
    uint32_t q4[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) q4[j] = get_int_b2(bq4->qs, 4*ib32 + j);
    const uint32_t s32 = (q4[0] & 0x00010001) | ((q4[1] & 0x00010001) << 2) | ((q4[2] & 0x00010001) << 4) | ((q4[3] & 0x00010001) << 6);
    const uint8_t ls = (s32 | (s32 >> 15)) & 0xff;
    const float dl = __half2float(bq4->d) * ((ls & 254) - 127);
    const int8_t * values = iq4k_values + ((ls & 1) << 4);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t aux32 = q4[j] & 0xfffefffe;
        aux32 ^= (aux32 >> 1);
        const int2 v = get_int_from_table_16(aux32, values);
        sumi = ggml_cuda_dp4a(v.x, q8[j+0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[j+4], sumi);
    }
    return dl * __low2float(bq8_1[ib32].ds) * sumi;
}

//
// dequantization: one CUDA block of 32 threads per block of 256, thread t writes values 8*t ... 8*t+7.
// The value of quant j is computed exactly like the CPU reference (dequantize_row_* in ggml-iqk.cpp).
//

static __device__ __forceinline__ float iqk_value(const block_iq2_k * x, int j) {
    const int ib32 = j/32, l = j%32, half = l/16;
    const int ls = int((x->scales[ib32] >> 4*half) & 0xf) - 8;
    const int8_t * v = iq2nl_values + 4*((x->extra >> (2*ib32 + half)) & 1);
    return __half2float(x->d) * ls * v[(x->qs[32*(ib32/4) + l] >> 2*(ib32%4)) & 3];
}

static __device__ __forceinline__ float iqk_value(const block_iq3_k * x, int j) {
    const int ib32 = j/32, l = j%32, half = l/16;
    const int ls = 2*((x->scales_l[ib32] >> 4*half) & 0xf) + 1;
    const int sl = (x->scales_h >> (2*ib32 + half)) & 1 ? -ls : ls;
    const int8_t * v = iq3nl_values + 8*((x->extra >> (2*ib32 + half)) & 1);
    const int idx = ((x->qs[32*(ib32/4) + l] >> 2*(ib32%4)) & 3) | (((x->qh[l] >> ib32) & 1) << 2);
    return __half2float(x->d) * sl * v[idx];
}

static __device__ __forceinline__ float iqk_value(const block_iq4_k * x, int j) {
    const int ib32 = j/32, l = j%32, half = l/16;
    const uint8_t sh = x->scales_h[ib32/2] >> 4*(ib32%2);
    const int ls = half ? int((x->scales_l[ib32] >> 4) | ((sh << 2) & 0x30)) - 32 : int((x->scales_l[ib32] & 0xf) | ((sh << 4) & 0x30)) - 32;
    const int8_t * v = iq4k_values + 16*((x->extra >> (2*ib32 + half)) & 1);
    const uint8_t q = x->qs[16*ib32 + l%16];
    return __half2float(x->d) * ls * v[half ? q >> 4 : q & 0xf];
}

static __device__ __forceinline__ float iqk_value(const block_iq2_ks * x, int j) {
    const int ib32 = j/32, l = j%32;
    const int ls = int(((x->scales[ib32/2] >> 4*(ib32%2)) & 0xf) | (((x->extra >> (8 + ib32)) & 1) << 4)) - 16;
    const int8_t * v = iq2nl_values + 4*((x->extra >> ib32) & 1);
    return __half2float(x->d) * ls * v[(x->qs[32*(ib32/4) + l] >> 2*(ib32%4)) & 3];
}

static __device__ __forceinline__ float iqk_value(const block_iq3_ks * x, int j) {
    const int ib32 = j/32, l = j%32;
    const int ls = int(((x->scales[ib32%4] >> 4*(ib32/4)) & 0xf) | (((x->extra >> ib32) & 1) << 4)) - 16;
    const int8_t * v = iq3nl_values + 8*((x->extra >> (8 + ib32)) & 1);
    const int idx = ((x->qs[32*(ib32/4) + l] >> 2*(ib32%4)) & 3) | (((x->qh[l] >> ib32) & 1) << 2);
    return __half2float(x->d) * ls * v[idx];
}

static __device__ __forceinline__ float iqk_value(const block_iq4_ks * x, int j) {
    const int ib32 = j/32, l = j%32;
    const int ls = int(x->scales[ib32] & 254) - 127;
    const int8_t * v = iq4k_values + 16*(x->scales[ib32] & 1);
    const uint8_t q = x->qs[16*ib32 + l%16];
    return __half2float(x->d) * ls * v[l < 16 ? q & 0xf : q >> 4];
}

static __device__ __forceinline__ float iqk_value(const block_iq4_kss * x, int j) {
    const int ib32 = j/32, l = j%32;
    const uint16_t * qs = x->qs + 8*ib32;
    int ls = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) ls |= (qs[k] & 1) << k;
    uint16_t a = qs[(l%16)/2] & 0xfffe;
    a ^= (a >> 1);
    const uint8_t q = (l%2) ? a >> 8 : a & 0xff;
    const int8_t * v = iq4k_values + 16*(ls & 1);
    return __half2float(x->d) * ((ls & 254) - 127) * v[l < 16 ? q & 0xf : q >> 4];
}

static __device__ __forceinline__ float iqk_value(const block_iq2_kl * x, int j) {
    const int ib32 = j/32, l = j%32;
    const int ls = int(((x->scales_l[ib32%4] >> 4*(ib32/4)) & 0xf) | (((x->scales_h >> 2*ib32) & 3) << 4)) - 32;
    const int idx = ((x->qs[16*(ib32/2) + l/2] >> 4*(ib32%2)) & 0xf) | (((x->qh[l/2] >> ib32) & 1) << 4);
    const uint16_t p = iq2kl_values[idx];
    return __half2float(x->d) * ls * (int8_t)(l%2 ? p >> 8 : p & 0xff);
}

template<typename block_t, typename dst_t>
static __global__ void dequantize_block_iqk(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const int     tid = threadIdx.x;
    const block_t * x = (const block_t *) vx + i;
    dst_t * y = yy + i*QK_K + 8*tid;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        y[k] = ggml_cuda_cast<dst_t>(iqk_value(x, 8*tid + k));
    }
}

template<typename block_t, typename dst_t>
static void dequantize_row_iqk_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int64_t nb = k / QK_K;
    dequantize_block_iqk<block_t><<<nb, 32, 0, stream>>>(vx, y);
}

// hooks for the upstream CUDA files: the type traits, and one-line case lists for convert.cu and mmvq.cu

#define GGML_CUDA_IQK_TRAITS(T, B) \
template<> struct ggml_cuda_type_traits<T> { \
    static constexpr int qk = QK_K; \
    static constexpr int qr = QR4_XS; \
    static constexpr int qi = QI4_XS; \
    static constexpr int bs = sizeof(B); \
};
GGML_CUDA_IQK_TRAITS(GGML_TYPE_IQ3_KT, block_iq3_kt)
GGML_CUDA_IQK_TRAITS(GGML_TYPE_IQ2_K, block_iq2_k)
GGML_CUDA_IQK_TRAITS(GGML_TYPE_IQ3_K, block_iq3_k)
GGML_CUDA_IQK_TRAITS(GGML_TYPE_IQ4_K, block_iq4_k)
GGML_CUDA_IQK_TRAITS(GGML_TYPE_IQ2_KS, block_iq2_ks)
GGML_CUDA_IQK_TRAITS(GGML_TYPE_IQ3_KS, block_iq3_ks)
GGML_CUDA_IQK_TRAITS(GGML_TYPE_IQ4_KS, block_iq4_ks)
GGML_CUDA_IQK_TRAITS(GGML_TYPE_IQ4_KSS, block_iq4_kss)
GGML_CUDA_IQK_TRAITS(GGML_TYPE_IQ2_KL, block_iq2_kl)

// ggml_get_to_fp32_cuda / to_fp16 / to_bf16 in convert.cu
#define GGML_CUDA_IQK_DEQUANT_CASES \
    case GGML_TYPE_IQ3_KT: return dequantize_row_iq3_kt_cuda; \
    case GGML_TYPE_IQ2_K: return dequantize_row_iqk_cuda<block_iq2_k>; \
    case GGML_TYPE_IQ3_K: return dequantize_row_iqk_cuda<block_iq3_k>; \
    case GGML_TYPE_IQ4_K: return dequantize_row_iqk_cuda<block_iq4_k>; \
    case GGML_TYPE_IQ2_KS: return dequantize_row_iqk_cuda<block_iq2_ks>; \
    case GGML_TYPE_IQ3_KS: return dequantize_row_iqk_cuda<block_iq3_ks>; \
    case GGML_TYPE_IQ4_KS: return dequantize_row_iqk_cuda<block_iq4_ks>; \
    case GGML_TYPE_IQ4_KSS: return dequantize_row_iqk_cuda<block_iq4_kss>; \
    case GGML_TYPE_IQ2_KL: return dequantize_row_iqk_cuda<block_iq2_kl>;

// get_vec_dot_q_cuda in mmvq.cu
#define GGML_CUDA_IQK_VEC_DOT_CASES \
    case GGML_TYPE_IQ3_KT: return vec_dot_iq3_kt_q8_1; \
    case GGML_TYPE_IQ2_K: return vec_dot_iq2_k_q8_1; \
    case GGML_TYPE_IQ3_K: return vec_dot_iq3_k_q8_1; \
    case GGML_TYPE_IQ4_K: return vec_dot_iq4_k_q8_1; \
    case GGML_TYPE_IQ2_KS: return vec_dot_iq2_ks_q8_1; \
    case GGML_TYPE_IQ3_KS: return vec_dot_iq3_ks_q8_1; \
    case GGML_TYPE_IQ4_KS: return vec_dot_iq4_ks_q8_1; \
    case GGML_TYPE_IQ4_KSS: return vec_dot_iq4_kss_q8_1; \
    case GGML_TYPE_IQ2_KL: return vec_dot_iq2_kl_q8_1;

// get_vdr_mmvq in mmvq.cu
#define GGML_CUDA_IQK_VDR_CASES \
    case GGML_TYPE_IQ3_KT: return VDR_IQ3_KT_Q8_1_MMVQ; \
    case GGML_TYPE_IQ2_K: return VDR_IQ2_K_Q8_1_MMVQ; \
    case GGML_TYPE_IQ3_K: return VDR_IQ3_K_Q8_1_MMVQ; \
    case GGML_TYPE_IQ4_K: return VDR_IQ4_K_Q8_1_MMVQ; \
    case GGML_TYPE_IQ2_KS: return VDR_IQ2_KS_Q8_1_MMVQ; \
    case GGML_TYPE_IQ3_KS: return VDR_IQ3_KS_Q8_1_MMVQ; \
    case GGML_TYPE_IQ4_KS: return VDR_IQ4_KS_Q8_1_MMVQ; \
    case GGML_TYPE_IQ4_KSS: return VDR_IQ4_KSS_Q8_1_MMVQ; \
    case GGML_TYPE_IQ2_KL: return VDR_IQ2_KL_Q8_1_MMVQ;

// mul_mat_vec_q_switch_type in mmvq.cu (uses the argument names of that function)
#define GGML_CUDA_IQK_MMVQ_CASES \
    case GGML_TYPE_IQ3_KT: mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_KT>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream); break; \
    case GGML_TYPE_IQ2_K: mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_K>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream); break; \
    case GGML_TYPE_IQ3_K: mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_K>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream); break; \
    case GGML_TYPE_IQ4_K: mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_K>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream); break; \
    case GGML_TYPE_IQ2_KS: mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_KS>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream); break; \
    case GGML_TYPE_IQ3_KS: mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_KS>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream); break; \
    case GGML_TYPE_IQ4_KS: mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_KS>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream); break; \
    case GGML_TYPE_IQ4_KSS: mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_KSS>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream); break; \
    case GGML_TYPE_IQ2_KL: mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_KL>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream); break;
