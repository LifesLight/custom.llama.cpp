// IQ_K quants (IQ2_K, IQ3_K, IQ4_K, IQ2_KS, IQ3_KS, IQ4_KS, IQ4_KSS, IQ2_KL), ported from ik_llama.cpp
// (https://github.com/ikawrakow/ik_llama.cpp), Copyright (C) 2024-2025 Iwan Kawrakow, MIT license.
//
// The quantizers are the scalar code paths of ik_llama.cpp, unchanged except:
//   - the KS/KSS/KL types have one scale per row in ik_llama.cpp. ggml has no per-row data, so here every block
//     of 256 is quantized as if it were a row of its own and stores that scale as fp16 in the block
//   - no quantize_user_data, no repacked variants

#include "ggml-quants.h"
#include "ggml-impl.h"

#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace {

struct quantize_user_data;

// the KS/KSS/KL blocks as they are in ik_llama.cpp, without the per-row scale
typedef struct {
    uint16_t extra;
    uint8_t  scales[QK_K/64];
    uint8_t  qs[QK_K/4];
} ik_block_iq2_ks;
static_assert(sizeof(ik_block_iq2_ks) + sizeof(ggml_half) == sizeof(block_iq2_ks), "iq2_ks layout");

typedef struct {
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/4];
    uint8_t  qh[QK_K/16];
} ik_block_iq2_kl;
static_assert(sizeof(ik_block_iq2_kl) + sizeof(ggml_half) == sizeof(block_iq2_kl), "iq2_kl layout");

typedef struct {
    uint16_t extra;
    uint8_t  scales[QK_K/64];
    uint8_t  qs[QK_K/4];
    uint8_t  qh[QK_K/8];
} ik_block_iq3_ks;
static_assert(sizeof(ik_block_iq3_ks) + sizeof(ggml_half) == sizeof(block_iq3_ks), "iq3_ks layout");

typedef struct {
    uint8_t  scales[QK_K/32];
    uint8_t  qs[QK_K/2];
} ik_block_iq4_ks;
static_assert(sizeof(ik_block_iq4_ks) + sizeof(ggml_half) == sizeof(block_iq4_ks), "iq4_ks layout");

typedef struct {
    uint32_t qs[QK_K/8];
} ik_block_iq4_kss;
static_assert(sizeof(ik_block_iq4_kss) + sizeof(ggml_half) == sizeof(block_iq4_kss), "iq4_kss layout");

inline int popcount(uint8_t x)  { return __builtin_popcount(x); }
inline int popcount(uint16_t x) { return __builtin_popcount(x); }

inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

struct QHelper {
    QHelper(const float * imatrix, int n_per_row, int block_size) :
        m_imatrix(imatrix), m_n_per_row(n_per_row), m_block_size(block_size) {
        if (m_imatrix) {
            m_weight.resize(m_n_per_row);
        }
    }
    const float * row_weights(const float * x) {
        constexpr float kEps  = 1e-9f;
        constexpr float kEps2 = kEps*kEps;
        if (!m_imatrix) return m_imatrix;
        int nblock = m_n_per_row / m_block_size;
        for (int ib = 0; ib < nblock; ++ib) {
            auto wb_in = m_imatrix + ib*m_block_size;
            auto xb = x + ib*m_block_size;
            auto wb = m_weight.data() + ib*m_block_size;
            float sumw2 = 0, sumx2 = 0, sumwx = 0;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = wb_in[j];
                sumw2 += wb[j]*wb[j];
                sumx2 += xb[j]*xb[j];
                sumwx += wb[j]*std::abs(xb[j]);
            }
            if (sumw2 > m_block_size*kEps2 && sumx2 > m_block_size*kEps2 && sumwx > m_block_size*kEps2) continue;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = kEps;
            }
        }
        return m_weight.data();
    }
    template <typename Func>
    void quantize(int64_t nrows, const float * src, void * dst, size_t row_size, const Func& qfunc) {
        auto cdst = (char *)dst;
        for (int64_t row = 0; row < nrows; ++row) {
            auto weights = row_weights(src);
            qfunc(src, cdst, m_n_per_row, weights);
            src  += m_n_per_row;
            cdst += row_size;
        }
    }
private:
    const float * m_imatrix;
    const int     m_n_per_row;
    const int     m_block_size;
    std::vector<float> m_weight;
};

float make_qx_quants(int n, int nmax, const float * x, int8_t * L, const float * qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (!amax) { // all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = -nmax / max;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax-1, l));
        L[i] = l + nmax;
        sumlx += qw[i]*x[i]*l;
        suml2 += qw[i]*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax-1, l));
            sumlx += qw[i]*x[i]*l;
            suml2 += qw[i]*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + std::max(-nmax, std::min(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

//
// ============================================== iq2_K
//
inline int best_index_iq2nl(const int8_t * values, float x) {
    int idx = x < values[1] ? 0 : x > values[2] ? 2 : 1;
    return x - values[idx] < values[idx+1] - x ? idx : idx + 1;
}

void quantize_row_iq2_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights,
        [[maybe_unused]] const quantize_user_data * user_data) {

    constexpr int kBlockSize = 16;

    block_iq2_k * y = (block_iq2_k *)vy;

    float scales[QK_K/kBlockSize];
    float weight[kBlockSize];
    float sumx[kBlockSize+1], sumw[kBlockSize+1];
    float sw[QK_K/kBlockSize];
    int8_t Ls[QK_K/kBlockSize];

    std::array<std::pair<float,int>, kBlockSize> pairs;

    const int8_t * shifted_values = iq2nl_values + 4;

    const float fudge = 1.03f;

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float * xb = xbl + kBlockSize*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            sw[ib] = 0;
            float amax = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                sw[ib] += weight[j];
                pairs[j] = {xb[j], j};
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            std::sort(pairs.begin(), pairs.end());
            sumx[0] = sumw[0] = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                int jj = pairs[j].second;
                sumw[j+1] = sumw[j] + weight[jj];
                sumx[j+1] = sumx[j] + weight[jj]*xb[jj];
            }
            float best = 0, d = 0;
            bool is_shifted = false;
            float sumqx, sumq2;
            for (int i1 = 0; i1 < kBlockSize; ++i1) {
                for (int i2 = i1; i2 < kBlockSize; ++i2) {
                    for (int i3 = i2; i3 < kBlockSize; ++i3) {
                        sumqx = (sumx[i1] - sumx[ 0])*iq2nl_values[0] + (sumx[i2] - sumx[i1])*iq2nl_values[1]
                              + (sumx[i3] - sumx[i2])*iq2nl_values[2] + (sumx[kBlockSize] - sumx[i3])*iq2nl_values[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*iq2nl_values[0]*iq2nl_values[0] + (sumw[i2] - sumw[i1])*iq2nl_values[1]*iq2nl_values[1]
                              + (sumw[i3] - sumw[i2])*iq2nl_values[2]*iq2nl_values[2] + (sumw[kBlockSize] - sumw[i3])*iq2nl_values[3]*iq2nl_values[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*shifted_values[0] + (sumx[i2] - sumx[i1])*shifted_values[1]
                              + (sumx[i3] - sumx[i2])*shifted_values[2] + (sumx[kBlockSize] - sumx[i3])*shifted_values[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*shifted_values[0]*shifted_values[0] + (sumw[i2] - sumw[i1])*shifted_values[1]*shifted_values[1]
                              + (sumw[i3] - sumw[i2])*shifted_values[2]*shifted_values[2] + (sumw[kBlockSize] - sumw[i3])*shifted_values[3]*shifted_values[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*iq2nl_values[3] + (sumx[i2] - sumx[i1])*iq2nl_values[2]
                              + (sumx[i3] - sumx[i2])*iq2nl_values[1] + (sumx[kBlockSize] - sumx[i3])*iq2nl_values[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*iq2nl_values[3]*iq2nl_values[3] + (sumw[i2] - sumw[i1])*iq2nl_values[2]*iq2nl_values[2]
                              + (sumw[i3] - sumw[i2])*iq2nl_values[1]*iq2nl_values[1] + (sumw[kBlockSize] - sumw[i3])*iq2nl_values[0]*iq2nl_values[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*shifted_values[3] + (sumx[i2] - sumx[i1])*shifted_values[2]
                              + (sumx[i3] - sumx[i2])*shifted_values[1] + (sumx[kBlockSize] - sumx[i3])*shifted_values[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*shifted_values[3]*shifted_values[3] + (sumw[i2] - sumw[i1])*shifted_values[2]*shifted_values[2]
                              + (sumw[i3] - sumw[i2])*shifted_values[1]*shifted_values[1] + (sumw[kBlockSize] - sumw[i3])*shifted_values[0]*shifted_values[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                    }
                }
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            max_abs_scale = std::max(max_abs_scale, abs_scale);
        }

        if (!max_abs_scale) continue;
        float d = make_qx_quants(QK_K/kBlockSize, 8, scales, Ls, sw);
        if (!d) continue;

        //float d = -max_scale/8;
        y[ibl].extra = extra;
        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            int ls = nearest_int(id*scales[ib]);
            ls = std::max(-8, std::min(7, ls));
            y[ibl].scales[ib/2] |= ((ls + 8) << 4*(ib%2));
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq2nl_values;
                const float * xb = xbl + kBlockSize*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/4) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq2nl(block_values, al);
                    qs[j] |= (ibest << 2*(ib32%4));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        y[ibl].d = GGML_FP32_TO_FP16(fudge*(sumq2 > 0 ? sumqx/sumq2 : d));

    }
}

//
// ============================================== iq2_KS
//
void quantize_row_iq2_ks_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_sw, int8_t * all_Ls) {

    constexpr int kBlockSize = 32;
    constexpr int kMax_i1 = 3*kBlockSize/4;
    constexpr int kMin_i3 = kBlockSize/4;

    ggml_half * dptr = (ggml_half *)vy;
    *dptr = GGML_FP32_TO_FP16(0.f);

    ik_block_iq2_ks * y = (ik_block_iq2_ks *)(dptr + 1);

    float weight[kBlockSize];
    float sumx[kBlockSize+1], sumw[kBlockSize+1];

    std::array<std::pair<float,int>, kBlockSize> pairs;

    float val [4] = {float(iq2nl_values[0]), float(iq2nl_values[1]), float(iq2nl_values[2]), float(iq2nl_values[3])};
    float sval[4] = {float(iq2nl_values[4]), float(iq2nl_values[5]), float(iq2nl_values[6]), float(iq2nl_values[7])};

    const int8_t * shifted_values = iq2nl_values + 4;

    const int nblock = n_per_row/QK_K;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(ik_block_iq2_ks));

        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto sw = all_sw + ibl*(QK_K/kBlockSize);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float * xb = xbl + kBlockSize*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            sw[ib] = 0;
            float amax = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                sw[ib] += weight[j];
                pairs[j] = {xb[j], j};
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            std::sort(pairs.begin(), pairs.end());
            sumx[0] = sumw[0] = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                int jj = pairs[j].second;
                sumw[j+1] = sumw[j] + weight[jj];
                sumx[j+1] = sumx[j] + weight[jj]*xb[jj];
            }
            float best = 0, d = 0;
            bool is_shifted = false;
            float sumqx, sumq2;
            for (int i1 = 0; i1 < kMax_i1; ++i1) {
                for (int i2 = i1; i2 < kBlockSize; ++i2) {
                    for (int i3 = std::max(i2, kMin_i3); i3 < kBlockSize; ++i3) {
                        sumqx = (sumx[i1] - sumx[ 0])*val[0] + (sumx[i2] - sumx[i1])*val[1]
                              + (sumx[i3] - sumx[i2])*val[2] + (sumx[kBlockSize] - sumx[i3])*val[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*val[0]*val[0] + (sumw[i2] - sumw[i1])*val[1]*val[1]
                              + (sumw[i3] - sumw[i2])*val[2]*val[2] + (sumw[kBlockSize] - sumw[i3])*val[3]*val[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*sval[0] + (sumx[i2] - sumx[i1])*sval[1]
                              + (sumx[i3] - sumx[i2])*sval[2] + (sumx[kBlockSize] - sumx[i3])*sval[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*sval[0]*sval[0] + (sumw[i2] - sumw[i1])*sval[1]*sval[1]
                              + (sumw[i3] - sumw[i2])*sval[2]*sval[2] + (sumw[kBlockSize] - sumw[i3])*sval[3]*sval[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*val[3] + (sumx[i2        ] - sumx[i1])*val[2]
                              + (sumx[i3] - sumx[i2])*val[1] + (sumx[kBlockSize] - sumx[i3])*val[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*val[3]*val[3] + (sumw[i2        ] - sumw[i1])*val[2]*val[2]
                              + (sumw[i3] - sumw[i2])*val[1]*val[1] + (sumw[kBlockSize] - sumw[i3])*val[0]*val[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*sval[3] + (sumx[i2        ] - sumx[i1])*sval[2]
                              + (sumx[i3] - sumx[i2])*sval[1] + (sumx[kBlockSize] - sumx[i3])*sval[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*sval[3]*sval[3] + (sumw[i2        ] - sumw[i1])*sval[2]*sval[2]
                              + (sumw[i3] - sumw[i2])*sval[1]*sval[1] + (sumw[kBlockSize] - sumw[i3])*sval[0]*sval[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                    }
                }
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
        }
        y[ibl].extra = extra;
    }

    float d = make_qx_quants(nblock*(QK_K/kBlockSize), 16, all_scales, all_Ls, all_sw);

    if (!d) return;

    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;
        auto Ls = all_Ls + ibl*(QK_K/kBlockSize);
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            int ls = Ls[ib];
            y[ibl].scales[ib/2] |= ((ls & 0xf) << 4*(ib%2));
            y[ibl].extra |= ((ls >> 4) << (8 + ib));
            ls -= 16;
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq2nl_values;
                const float * xb = xbl + kBlockSize*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                uint8_t * qs = y[ibl].qs + 32*(ib/4);
                for (int j = 0; j < 32; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq2nl(block_values, al);
                    qs[j] |= (ibest << 2*(ib%4));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
    }
    *dptr = GGML_FP32_TO_FP16(1.030f*(sumq2 > 0 ? sumqx/sumq2 : d));
}

//
// ======================================== iq2_KL
//
const int8_t iq3nl_index[111] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  8,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  9,
  9,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 10, 10,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 11, 11,  4,  4,  4,  4,
  4,  4,  4,  4,  4,  4, 12,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 13, 13,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
  6,  6,  6,  6, 14, 14,  7,  7,  7,  7,  7,  7,  7,  7, 7
};
inline int best_index_iq3nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 111) return ix < 0 ? 0 : 7;
    ix = iq3nl_index[ix];
    return ix < 8 ? ix : x - values[ix-8] < values[ix-7] - x ? ix-8 : ix-7;
}
#if 0 // AVX2 search of ik_llama.cpp not ported, the scalar search is equivalent
static const int8_t k_simple_index_iq3nl[128] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,
    6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
};
#endif

void quantize_row_iq2_kl_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales) {
    constexpr int kBlockSize = 32;
    constexpr float kSigmaFactor = 2.25f;
    constexpr int ntry = 5;
    static const int k_index[64] = {-1, -2, 0, -3, -4, 1, -5, -6, 2, -7, -8, 3, -9, 4, -10, 5, -11, 6, 7, -12, 8, 9, 10, -13, 11, -14, -15, -16, 12, 13, -17,
        14, -18, -19, 15, 16, 17, 18, 19, -20, -21, 20, 21, 22, 23, 24, -22, -23, 25, -24, 26, -25, 27, -26, 28, 29, -27, -28, 30, -29, -30, 31, -31, -32};
    static const std::vector<std::vector<int>> k_neighbours = {
        { 2, 0, 6, 11, 7, 3, 8, 15,  },
        { 0, 2, 3, 6, 7, 1, 8, 4,  },
        { 0, 1, 3, 4, 8, 7, 9, 6,  },
        { 1, 0, 3, 4, 8, 9, 7, 10,  },
        { 1, 4, 5, 10, 9, 3, 8, 0,  },
        { 5, 1, 4, 10, 9, 14, 8, 3,  },
        { 6, 2, 7, 0, 3, 11, 8, 15,  },
        { 3, 7, 0, 6, 8, 4, 12, 9,  },
        { 3, 4, 8, 9, 1, 7, 12, 10,  },
        { 4, 10, 5, 9, 1, 8, 13, 14,  },
        { 11, 2, 6, 7, 20, 15, 25, 21,  },
        { 8, 7, 3, 12, 9, 16, 17, 13,  },
        { 14, 5, 10, 19, 9, 13, 4, 18,  },
        { 6, 15, 7, 11, 20, 21, 16, 2,  },
        { 15, 7, 16, 6, 21, 12, 17, 22,  },
        { 12, 16, 17, 8, 15, 7, 13, 22,  },
        { 19, 10, 13, 18, 14, 9, 12, 24,  },
        { 11, 20, 25, 6, 15, 2, 21, 7,  },
        { 20, 15, 21, 6, 11, 7, 16, 26,  },
        { 14, 19, 29, 10, 28, 18, 13, 24,  },
        { 25, 11, 20, 21, 15, 6, 26, 30,  },
        { 19, 24, 28, 18, 29, 23, 13, 17,  },
        { 29, 19, 14, 28, 24, 18, 10, 13,  },
        { 20, 26, 21, 25, 30, 15, 22, 16,  },
        { 27, 26, 22, 23, 21, 30, 16, 24,  },
        { 27, 24, 28, 31, 23, 18, 22, 17,  },
        { 25, 30, 20, 26, 21, 11, 15, 22,  },
        { 30, 26, 25, 20, 21, 27, 22, 15,  },
        { 30, 27, 31, 26, 22, 23, 21, 24,  },
        { 31, 27, 30, 26, 28, 23, 22, 24,  },
        { 31, 28, 29, 27, 24, 23, 19, 18,  },
        { 29, 28, 31, 24, 19, 27, 14, 18,  },
    };
    auto values = iq3nl_values;
    std::pair<int8_t, int8_t> grid[32];
    for (int j = 0; j < 64; ++j) {
        if (int i = k_index[j]; i >= 0) {
            int i1 = j/8, i2 = j%8;
            grid[i] = {values[i1], values[i2]};
        }
    }

    ggml_half * dptr = (ggml_half *)vy;
    auto y = (ik_block_iq2_kl *)(dptr + 1);

    float weight[kBlockSize];

    auto index = [&grid, values] (float id, float x1, float x2, float w1, float w2) {
        float sx1 = id*x1;
        float sx2 = id*x2;
        int l1 = best_index_iq3nl(values, sx1);
        int l2 = best_index_iq3nl(values, sx2);
        int i = k_index[8*l1 + l2];
        if (i >= 0) return i;
        auto& neigh = k_neighbours[-i-1];
        float best = std::numeric_limits<float>::max();
        int ibest = -1;
        for (auto& n : neigh) {
            float diff1 = grid[n].first  - sx1;
            float diff2 = grid[n].second - sx2;
            float score = w1*diff1*diff1 + w2*diff2*diff2;
            if (score < best) {
                best = score; ibest = n;
            }
        }
        GGML_ASSERT(ibest >= 0);
        return ibest;
    };

    float max_scale = 0, max_abs_scale = 0;

    const float fudge = 1.025f;

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
        std::memset(&y[ibl], 0, sizeof(ik_block_iq2_kl));
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto xbl = x + ibl*QK_K;
        float sigma2 = 0;
        for (int j = 0; j < QK_K; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= kSigmaFactor/QK_K;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            auto xb = xbl + ib*kBlockSize;
            if (quant_weights) {
                auto qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j]*sqrt(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = std::abs(xb[j]); //xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < kBlockSize; j += 2) {
                float w1 = weight[j+0];
                float w2 = weight[j+1];
                int idx = index(id, xb[j+0], xb[j+1], w1, w2);
                float q1 = grid[idx].first ;
                float q2 = grid[idx].second;
                sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
                sumq2_p += w1*q1*q1 + w2*q2*q2;
                idx = index(-id, xb[j+0], xb[j+1], w1, w2);
                q1 = grid[idx].first ;
                q2 = grid[idx].second;
                sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
                sumq2_m += w1*q1*q1 + w2*q2*q2;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < kBlockSize; j += 2) {
                    float w1 = weight[j+0];
                    float w2 = weight[j+1];
                    int idx = index(id, xb[j+0], xb[j+1], w1, w2);
                    float q1 = grid[idx].first ;
                    float q2 = grid[idx].second;
                    sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
                    sumq2_p += w1*q1*q1 + w2*q2*q2;
                    idx = index(-id, xb[j+0], xb[j+1], w1, w2);
                    q1 = grid[idx].first ;
                    q2 = grid[idx].second;
                    sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
                    sumq2_m += w1*q1*q1 + w2*q2*q2;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m;
                }
            }
            scales[ib] = d;
            float ad = std::abs(d);
            if (ad > max_abs_scale) {
                max_abs_scale = ad; max_scale = d;
            }
        }
    }

    if (!max_abs_scale) {
        dptr[0] = GGML_FP32_TO_FP16(0.f);
        return;
    }

    float d = -max_scale/32;
    float id = 1/d;

    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto xbl = x + ibl*QK_K;
        float sigma2 = 0;
        for (int j = 0; j < QK_K; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= kSigmaFactor/QK_K;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            auto xb = xbl + ib*kBlockSize;
            if (quant_weights) {
                auto qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j]*sqrt(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = std::abs(xb[j]); //xb[j]*xb[j];
            }
            int ls = nearest_int(id*scales[ib]);
            ls = std::max(-32, std::min(31, ls));
            int lsmin = std::max(-32, ls-1);
            int lsmax = std::min( 31, ls+1);
            float best_score = std::numeric_limits<float>::max();
            int best_ls = ls;
            for (int ils = lsmin; ils <= lsmax; ++ils) {
                float dl = d*ils;
                float idl = dl ? 1/dl : 0.f;
                float score = 0;
                for (int j = 0; j < kBlockSize/2; ++j) {
                    float w1 = weight[2*j+0];
                    float w2 = weight[2*j+1];
                    int idx = index(idl, xb[2*j+0], xb[2*j+1], w1, w2);
                    float diff1 = dl*grid[idx].first  - xb[2*j+0];
                    float diff2 = dl*grid[idx].second - xb[2*j+1];
                    score += w1*diff1*diff1 + w2*diff2*diff2;
                }
                if (score < best_score) {
                    best_score = score;
                    best_ls = ils;
                }
            }
            ls = best_ls;
            int uls = ls + 32;
            y[ibl].scales_l[ib%4] |= ((uls & 0xf) << 4*(ib/4));
            y[ibl].scales_h |= ((uls >> 4) << 2*ib);
            if (ls == 0) continue;
            float dl = d*ls;
            float idl = 1/dl;
            for (int j = 0; j < kBlockSize/2; ++j) {
                float w1 = weight[2*j+0];
                float w2 = weight[2*j+1];
                int idx = index(idl, xb[2*j+0], xb[2*j+1], w1, w2);
                y[ibl].qs[16*(ib/2) + j] |= ((idx & 0xf) << 4*(ib%2));
                y[ibl].qh[j] |= ((idx >> 4) << ib);
                float q1 = ls*grid[idx].first ;
                float q2 = ls*grid[idx].second;
                sumqx += w1*q1*xb[2*j] + w2*q2*xb[2*j+1];
                sumq2 += w1*q1*q1 + w2*q2*q2;
            }
        }
    }
    if (sumq2 > 0) d = sumqx/sumq2;

    dptr[0] = GGML_FP32_TO_FP16(fudge * d);

}

//
// ============================================== iq3_K
//
static void quantize_row_iq3_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights,
        [[maybe_unused]] const quantize_user_data * user_data) {

    constexpr int ntry = 3;

#if 0 // AVX2 search of ik_llama.cpp not ported, the scalar search is equivalent
    float f_values[16];
    for (int j = 0; j < 16; ++j) f_values[j] = iq3nl_values[j];
#endif

    block_iq3_k * y = (block_iq3_k *)vy;

    float scales[QK_K/16];
    float weight[16];

    const int8_t * shifted_values = iq3nl_values + 8;

    const float fudge = 1.01f;

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        float max_abs_scale = 0;

#if 0 // AVX2 search of ik_llama.cpp not ported, the scalar search is equivalent
        for (int ib = 0; ib < QK_K/16; ++ib) {
            auto xb = xbl + ib*16;
            auto [d, _, is_shifted] = find_best_scale_avx2<16, 127>(8, iq3nl_values, f_values, k_simple_index_iq3nl, xb,
                    quant_weights ? quant_weights + QK_K*ibl + 16*ib : nullptr, sigma2, ntry, true, 2.0f);

            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
            float abs_scale = fabsf(scales[ib]);
            max_abs_scale = std::max(max_abs_scale, abs_scale);
        }
#else
        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*16;
                for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < 16; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/iq3nl_values[0] : max/iq3nl_values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            float best = 0;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq3nl(iq3nl_values, al);
                float q = iq3nl_values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq3nl(iq3nl_values, -al);
                q = iq3nl_values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0) {
                d = sumqx_p/sumq2_p;
                best = d*sumqx_p;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (2*itry + iq3nl_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(iq3nl_values, al);
                    float q = iq3nl_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(iq3nl_values, -al);
                    q = iq3nl_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (2*itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
            }
            if (!d) {
                scales[ib] = 0; continue;
            }

            scales[ib] = d;

            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            max_abs_scale = MAX(max_abs_scale, abs_scale);
        }
#endif

        if (!max_abs_scale) continue;

        float d = max_abs_scale/31;
        y[ibl].extra = extra;
        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int ls = nearest_int(0.5f*(id*fabsf(scales[ib])-1));
            ls = MAX(0, MIN(15, ls));
            y[ibl].scales_l[ib/2] |= (ls << 4*(ib%2));
            if (scales[ib] < 0) y[ibl].scales_h |= (1 << ib);
            ls = (2*ls + 1) * (scales[ib] < 0 ? -1 : 1);
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq3nl_values;
                const float * xb = xbl + 16*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*16;
                    for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/4) + offset;
                uint8_t * qh = y[ibl].qh + 32*(ib32/8) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq3nl(block_values, al);
                    qs[j] |= ((ibest &  3) << 2*(ib32%4));
                    qh[j] |= ((ibest >> 2) << (ib32%8));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        y[ibl].d = GGML_FP32_TO_FP16(fudge*(sumq2 > 0 ? sumqx/sumq2 : d));

    }
}

//
// ============================================== iq3_KS
//
static void quantize_row_iq3_ks_impl(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

#if 0 // AVX2 search of ik_llama.cpp not ported, the scalar search is equivalent
    float f_values[16];
    for (int j = 0; j < 16; ++j) f_values[j] = values[j];
#endif

    ggml_half * dptr = (ggml_half *)cy;
    ik_block_iq3_ks * y = (ik_block_iq3_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 8;

    float amax_scale = 0;
    float max_scale = 0;

    const float fudge = 1.0f;

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(ik_block_iq3_ks));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
#if 0 // AVX2 search of ik_llama.cpp not ported, the scalar search is equivalent
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            auto [d, _, is_shifted] = find_best_scale_avx2<block_size, 127>(8, values, f_values, k_simple_index_iq3nl, xbl + ib*block_size,
                    quant_weights ? quant_weights + super_block_size*ibl + block_size*ib : nullptr, sigma2, ntry);
            if (is_shifted) y[ibl].extra |= (1 << (8 + ib));
            scales[ib] = d;
            float ascale = std::abs(d);
            if (ascale > amax_scale) {
                amax_scale = ascale; max_scale = d;
            }
        }
#else
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            float best = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq3nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq3nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0) {
                d = sumqx_p/sumq2_p;
                best = d*sumqx_p;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(values, -al);
                    q = values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
            }
            if (is_shifted) y[ibl].extra |= (1 << (8 + ib));
            scales[ib] = d;
            float ascale = std::abs(d);
            if (ascale > amax_scale) {
                amax_scale = ascale; max_scale = d;
            }
        }
#endif
    }
    float d = -max_scale/16;
    *dptr = GGML_FP32_TO_FP16(d);
    if (!d) return;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = (y[ibl].extra >> (8 + ib)) & 0x01 ? shifted_values : values;
            int l = nearest_int(id*scales[ib]);
            l = std::max(-16, std::min(15, l));
            uint8_t ul = l + 16;
            y[ibl].scales[ib%4] |= (ul & 0xf) << 4*(ib/4);
            y[ibl].extra |= (ul >> 4) << ib;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            auto qs = y[ibl].qs + (ib/4)*block_size;
            auto qh = y[ibl].qh + (ib/8)*block_size;
            for (int j = 0; j < block_size; ++j) {
                uint8_t i = best_index_iq3nl(block_values, idl*xb[j]);
                qs[j] |= ((i &  3) << 2*(ib%4));
                qh[j] |= ((i >> 2) << (ib%8));
                float w = weight[j];
                float q = block_values[i]*l;
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
        }
    }
    if (sumq2 > 0) *dptr = GGML_FP32_TO_FP16(fudge*sumqx/sumq2);
}

//
// ============================================== iq4_K
//
const int8_t iq4nl_index[241] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16, 16,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1, 17, 17,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 18,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3, 19,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 20,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     5,  5, 21, 21,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 22,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 23, 23,  8,  8,  8,  8,
     8,  8,  8,  8,  8,  8, 24,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
};
inline int best_index_iq4nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = iq4nl_index[ix];
    return ix < 16 ? ix : x - values[ix-16] < values[ix-15] - x ? ix-16 : ix-15;
}
static void quantize_row_iq4_k_impl_bs16(const float * x,
        block_iq4_k * y,
        float * scales, float * weight, uint8_t * L,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 16;

    float sigma2 = 0;
    for (int j = 0; j < super_block_size; ++j) sigma2 += x[j]*x[j];
    sigma2 *= 2.f/super_block_size;

    memset(y, 0, sizeof(block_iq4_k));
    y->d = GGML_FP32_TO_FP16(0.f);

    uint16_t * scales_h = (uint16_t *)y->scales_h;

    const int8_t * shifted_values = values + 16;

    const float fudge = 1.0f;

#if 0 // AVX2 search of ik_llama.cpp not ported, the scalar search is equivalent
    float f_values[32];
    for (int j = 0; j < 32; ++j) f_values[j] = values[j];
#endif

    float max_scale = 0, amax_scale = 0;
    uint16_t extra = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const float * xb = x + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        float amax = 0, max = 0;
        for (int j = 0; j < block_size; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax; max = xb[j];
            }
        }
        if (amax < 1e-16f) {
            scales[ib] = 0;
            continue;
        }
        float d = ntry > 0 ? -max/values[0] : max/values[0];
        float id = 1/d;
        bool is_shifted = false;
#if 0 // AVX2 search of ik_llama.cpp not ported, the scalar search is equivalent
        float best = 0;
        __m256 vx[block_size/8], vw[block_size/8];
        for (int j = 0; j < block_size/8; ++j) vx[j] = _mm256_loadu_ps(xb + 8*j);
        for (int j = 0; j < block_size/8; ++j) vw[j] = _mm256_loadu_ps(weight + 8*j);
        if (!evaluate_block_iq4k<block_size>(id, values, f_values, vx, vw, d, best)) {
            GGML_ABORT("Fatal error");
        }
#else
        float sumqx_p = 0, sumq2_p = 0;
        float sumqx_m = 0, sumq2_m = 0;
        for (int j = 0; j < block_size; ++j) {
            float w = weight[j];
            float al = id*xb[j];
            int l = best_index_iq4nl(values, al);
            float q = values[l];
            sumqx_p += w*q*xb[j];
            sumq2_p += w*q*q;
            l = best_index_iq4nl(values, -al);
            q = values[l];
            sumqx_m += w*q*xb[j];
            sumq2_m += w*q*q;
        }
        d = sumqx_p/sumq2_p;
        float best = d*sumqx_p;
        if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
            d = sumqx_m/sumq2_m; best = d*sumqx_m;
        }
#endif
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + values[0])/max;
#if 0 // AVX2 search of ik_llama.cpp not ported, the scalar search is equivalent
            if (evaluate_block_iq4k<block_size>(id, values, f_values, vx, vw, d, best)) {
                is_shifted = false;
            }
#else
            sumqx_p = sumq2_p = 0;
            sumqx_m = sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq4nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq4nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
            }
#endif
            id = (itry + shifted_values[0])/max;
#if 0 // AVX2 search of ik_llama.cpp not ported, the scalar search is equivalent
            if (evaluate_block_iq4k<block_size>(id, values+16, f_values+16, vx, vw, d, best)) {
                is_shifted = true;
            }
#else
            sumqx_p = sumq2_p = 0;
            sumqx_m = sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq4nl(shifted_values, al);
                float q = shifted_values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq4nl(shifted_values, -al);
                q = shifted_values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
            }
#endif
        }
        if (is_shifted) extra |= (1 << ib);
        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }
    float d = -max_scale/32;
    y->d = GGML_FP32_TO_FP16(fudge*d);
    y->extra = extra;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const int8_t * block_values = extra & (1 << ib) ? shifted_values : values;
        int l = nearest_int(id*scales[ib]);
        l = MAX(-32, MIN(31, l));
        float dl = d * l;
        float idl = dl ? 1/dl : 0.f;
        uint8_t * Lb = L + ib*block_size;
        const float * xb = x + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        for (int j = 0; j < block_size; ++j) {
            Lb[j] = best_index_iq4nl(block_values, idl*xb[j]);
            float w = weight[j];
            float q = block_values[Lb[j]]*l;
            sumqx += w*q*xb[j];
            sumq2 += w*q*q;
        }
        l += 32;
        uint8_t l_l = l & 0xf;
        uint8_t l_h = l >>  4;
        if (ib%2 == 0) y->scales_l[ib/2] = l_l;
        else y->scales_l[ib/2] |= (l_l << 4);
        scales_h[ib/8] |= (l_h << 2*(ib%8));
    }
    if (sumq2 > 0) y->d = GGML_FP32_TO_FP16(fudge*sumqx/sumq2);

    for (int i = 0; i < super_block_size/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            y->qs[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
        }
    }
}

//
// ============================================== iq4_KS
//
static void quantize_row_iq4_k_impl_bs32(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

    float * dptr = (float *)cy;
    ik_block_iq4_ks * y = (ik_block_iq4_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 16;

    float amax_scale = 0;

    const float fudge = 1.0f;

#if 0 // AVX2 search of ik_llama.cpp not ported, the scalar search is equivalent
    float f_values[32];
    for (int j = 0; j < 32; ++j) f_values[j] = values[j];
#endif

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(ik_block_iq4_ks));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            bool is_shifted = false;
            float best = 0;
#if 0 // AVX2 search of ik_llama.cpp not ported, the scalar search is equivalent
            __m256 vx[block_size/8], vw[block_size/8];
            for (int j = 0; j < block_size/8; ++j) vx[j] = _mm256_loadu_ps(xb + 8*j);
            for (int j = 0; j < block_size/8; ++j) vw[j] = _mm256_loadu_ps(weight + 8*j);
            if (!evaluate_block_iq4k<block_size>(id, values, f_values, vx, vw, d, best)) {
                GGML_ABORT("Fatal error");
            }
#else
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq4nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq4nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
#endif
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
#if 0 // AVX2 search of ik_llama.cpp not ported, the scalar search is equivalent
                if (evaluate_block_iq4k<block_size>(id, values, f_values, vx, vw, d, best)) {
                    is_shifted = false;
                }
                id = (itry + shifted_values[0])/max;
                if (evaluate_block_iq4k<block_size>(id, values+16, f_values+16, vx, vw, d, best)) {
                    is_shifted = true;
                }
#else
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq4nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq4nl(values, -al);
                    q = values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq4nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq4nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
#endif
            }
            if (is_shifted) y[ibl].scales[ib] = 0x01;
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
    }
    float d = amax_scale/127;
    *dptr = fudge*d;
    if (!d) return;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = y[ibl].scales[ib] & 0x01 ? shifted_values : values;
            int l = nearest_int(0.5f*(id*scales[ib]+127.f));
            l = std::max(0, std::min(127, l)) << 1;
            y[ibl].scales[ib] |= l;
            l -= 127;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            auto qs = y[ibl].qs + ib*(block_size/2);
            for (int j = 0; j < block_size/2; ++j) {
                uint8_t i1 = best_index_iq4nl(block_values, idl*xb[j]);
                uint8_t i2 = best_index_iq4nl(block_values, idl*xb[j+block_size/2]);
                qs[j] = i1 | (i2 << 4);
                float w1 = weight[j];
                float w2 = weight[j+block_size/2];
                float q1 = block_values[i1]*l;
                float q2 = block_values[i2]*l;
                sumqx += w1*q1*xb[j] + w2*q2*xb[j+block_size/2];
                sumq2 += w1*q1*q1 + w2*q2*q2;
            }
        }
    }
    if (sumq2 > 0) *dptr = fudge*sumqx/sumq2;
}

//
// ============================================== iq4_KSS
//
const uint16_t * scramble_table() {
    static std::mutex mutex;
    static std::vector<uint16_t> table;
    std::lock_guard<std::mutex> lock(mutex);
    if (table.empty()) {
        table.resize(1 << 15);
        for (int i = 0; i < int(table.size()); ++i) {
            uint16_t val = i;
            int non = popcount(val);
            if (non%2) val |= (1 << 15);
            bool found = false;
            for (int j = 0; j < int(table.size()); ++j) {
                if ((j ^ (j << 1)) == val) {
                    table[i] = j; found = true; break;
                }
            }
            if (!found) {
                printf("Oops: did not find for %d %u\n", i, val);
                exit(1);
            }
        }
    }
    return table.data();
}
uint16_t prune_iq4ks(uint16_t v, const int8_t * values, const float * x, const float * w, float dl) {
    if (popcount(v)%2 == 0) return v;
    float best_score = std::numeric_limits<float>::max();
    uint8_t q4[4];
    int jbest = -1;
    uint8_t bestq = 0;
    for (int j = 0; j < 4; ++j) {
        uint8_t q = (v >> 4*j) & 0xf;
        q4[j] = q;
        auto pc = popcount(q);
        float diff0 = dl*iq4k_values[q] - x[j];
        int qmin = std::max(int(q)-2,  0);
        int qmax = std::min(int(q)+2, 15);
        for (int iq = qmin; iq <= qmax; ++iq) {
            uint8_t qq = iq;
            if (qq == q) continue;
            int pci = popcount(qq);
            if (std::abs(pci - pc)%2) {
                float diff1 = dl*values[qq] - x[j];
                float score = w[j]*(diff1*diff1 - diff0*diff0);
                if (score < best_score) {
                    best_score = score; jbest = j; bestq = qq;
                }
            }
        }
    }
    GGML_ASSERT(jbest >= 0);
    q4[jbest] = bestq;
    return (q4[0] | (q4[1] << 4) | (q4[2] << 8) | (q4[3] << 12));
}
static void quantize_row_iq4_kss_impl(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const uint16_t * table,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

    float * dptr = (float *)cy;
    *dptr = 0;
    ik_block_iq4_kss * y = (ik_block_iq4_kss *)(dptr + 1);

    const int8_t * shifted_values = values + 16;

    uint16_t vps[block_size/2], vms[block_size/2], vs[block_size/2];
    float xv[4], wv[4];

    float amax_scale = 0;

    const float fudge = 1.01f;

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(ik_block_iq4_kss));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float d = -max/iq4k_values[0];
            std::memset(vs, 0, block_size);
            for (int itry = -ntry; itry <= ntry; ++itry) {
                float id = (itry + values[0])/max;
                float sumqx_p = 0, sumq2_p = 0;
                float sumqx_m = 0, sumq2_m = 0;
                float this_d = 1/id;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = id*xv[j];
                        vp |= (best_index_iq4nl(values,  al) << 4*j);
                        vm |= (best_index_iq4nl(values, -al) << 4*j);
                    }
                    vp = prune_iq4ks(vp, values, xv, wv,  this_d);
                    vm = prune_iq4ks(vm, values, xv, wv,  this_d);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = values[(vp >> 4*j) & 0xf];
                        sumqx_p += w*q*xv[j];
                        sumq2_p += w*q*q;
                        q = values[(vm >> 4*j) & 0xf];
                        sumqx_m += w*q*xv[j];
                        sumq2_m += w*q*q;
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                bool copy_p = false, copy_m = false;
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; copy_p = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; copy_m = true;
                }
                if (copy_m) {
                    std::memcpy(vs, vms, block_size);
                } else if (copy_p) {
                    std::memcpy(vs, vps, block_size);
                }

                id = (itry + shifted_values[0])/max;
                this_d = 1/id;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = id*xv[j];
                        vp |= (best_index_iq4nl(shifted_values,  al) << 4*j);
                        vm |= (best_index_iq4nl(shifted_values, -al) << 4*j);
                    }
                    vp = prune_iq4ks(vp, shifted_values, xv, wv,  this_d);
                    vm = prune_iq4ks(vm, shifted_values, xv, wv,  this_d);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = shifted_values[(vp >> 4*j) & 0xf];
                        sumqx_p += w*q*xv[j];
                        sumq2_p += w*q*q;
                        q = shifted_values[(vm >> 4*j) & 0xf];
                        sumqx_m += w*q*xv[j];
                        sumq2_m += w*q*q;
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                copy_p = copy_m = false;
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; copy_p = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; copy_m = true;
                }
                if (copy_m) {
                    std::memcpy(vs, vms, block_size);
                } else if (copy_p) {
                    std::memcpy(vs, vps, block_size);
                }
            }
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
    }
    float d = amax_scale/127;
    *dptr = fudge*d;
    if (!d) return;
    float id = 1/d;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            int l = nearest_int(0.5f*(id*scales[ib]+127.f));
            l = (std::max(0, std::min(127, l)) << 1) - 127;
            if (l) {
                float dl = d*l;
                float idl = 1/dl;
                float mse_p = 0, mse_m = 0;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = idl*xv[j];
                        vp |= (best_index_iq4nl(        values, al) << 4*j);
                        vm |= (best_index_iq4nl(shifted_values, al) << 4*j);
                    }
                    vp = prune_iq4ks(vp,         values, xv, wv,  dl);
                    vm = prune_iq4ks(vm, shifted_values, xv, wv,  dl);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = values[(vp >> 4*j) & 0xf];
                        mse_p += w*(xv[j] - dl*q)*(xv[j] - dl*q);
                        q = shifted_values[(vm >> 4*j) & 0xf];
                        mse_m += w*(xv[j] - dl*q)*(xv[j] - dl*q);
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                const uint16_t * v = vps;
                const int8_t * block_values = values;
                if (mse_m < mse_p) {
                    v = vms;
                    block_values = values + 16;
                }
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    for (int j = 0; j < 4; ++j) {
                        float q = block_values[(v[k] >> 4*j) & 0xf] * l;
                        sumqx += wv[j]*q*xv[j];
                        sumq2 += wv[j]*q*q;
                    }
                }
                l += 127;
                if (mse_m < mse_p) l |= 1;
                uint16_t * q16 = (uint16_t *)y[ibl].qs + (block_size/4)*ib;
                for (int k = 0; k < block_size/4; ++k) {
                    auto val = table[v[k] & 0x7fff];
                    q16[k] = (val << 1) | ((l >> k) & 1);
                }
            } else {
                l += 127;
                uint16_t * q16 = (uint16_t *)y[ibl].qs + (block_size/4)*ib;
                for (int k = 0; k < block_size/4; ++k) {
                    q16[k] = ((l >> k) & 1);
                }
            }
        }
    }
    if (sumq2 > 0) *dptr = fudge*sumqx/sumq2;
}

} // namespace

//
// ============================================== public functions
//

extern "C" {

size_t quantize_iq2_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    const size_t row_size = (n_per_row/QK_K)*sizeof(block_iq2_k);
    QHelper helper(imatrix, n_per_row, 16);
    helper.quantize(nrows, src, dst, row_size, [] (const float * x, void * vy, int n, const float * w) {
        quantize_row_iq2_k_impl(x, vy, n, w, nullptr);
    });
    return nrows*row_size;
}

size_t quantize_iq3_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    const size_t row_size = (n_per_row/QK_K)*sizeof(block_iq3_k);
    QHelper helper(imatrix, n_per_row, 16);
    helper.quantize(nrows, src, dst, row_size, [] (const float * x, void * vy, int n, const float * w) {
        quantize_row_iq3_k_impl(x, vy, n, w, nullptr);
    });
    return nrows*row_size;
}

size_t quantize_iq4_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    const size_t row_size = (n_per_row/QK_K)*sizeof(block_iq4_k);
    uint8_t L[QK_K];
    float weight[16];
    float scales[QK_K/16];
    QHelper helper(imatrix, n_per_row, 16);
    helper.quantize(nrows, src, dst, row_size, [&] (const float * x, void * vy, int n, const float * w) {
        block_iq4_k * y = (block_iq4_k *) vy;
        for (int ibl = 0; ibl < n/QK_K; ++ibl) {
            quantize_row_iq4_k_impl_bs16(x + QK_K*ibl, y + ibl, scales, weight, L, iq4k_values, w ? w + QK_K*ibl : nullptr, 7);
        }
    });
    return nrows*row_size;
}

// the KS types: each block of 256 is quantized like a row of 256 in ik_llama.cpp

size_t quantize_iq2_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    const size_t row_size = (n_per_row/QK_K)*sizeof(block_iq2_ks);
    float all_scales[QK_K/32], all_sw[QK_K/32];
    int8_t all_Ls[QK_K/32];
    QHelper helper(imatrix, n_per_row, 32);
    helper.quantize(nrows, src, dst, row_size, [&] (const float * x, void * vy, int n, const float * w) {
        block_iq2_ks * y = (block_iq2_ks *) vy;
        for (int ibl = 0; ibl < n/QK_K; ++ibl) {
            quantize_row_iq2_ks_impl(x + QK_K*ibl, y + ibl, QK_K, w ? w + QK_K*ibl : nullptr, all_scales, all_sw, all_Ls);
        }
    });
    return nrows*row_size;
}

size_t quantize_iq2_kl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    const size_t row_size = (n_per_row/QK_K)*sizeof(block_iq2_kl);
    float all_scales[QK_K/32];
    QHelper helper(imatrix, n_per_row, 32);
    helper.quantize(nrows, src, dst, row_size, [&] (const float * x, void * vy, int n, const float * w) {
        block_iq2_kl * y = (block_iq2_kl *) vy;
        for (int ibl = 0; ibl < n/QK_K; ++ibl) {
            quantize_row_iq2_kl_impl(x + QK_K*ibl, y + ibl, QK_K, w ? w + QK_K*ibl : nullptr, all_scales);
        }
    });
    return nrows*row_size;
}

size_t quantize_iq3_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    const size_t row_size = (n_per_row/QK_K)*sizeof(block_iq3_ks);
    float all_scales[QK_K/32], weight[32];
    QHelper helper(imatrix, n_per_row, 32);
    helper.quantize(nrows, src, dst, row_size, [&] (const float * x, void * vy, int n, const float * w) {
        block_iq3_ks * y = (block_iq3_ks *) vy;
        for (int ibl = 0; ibl < n/QK_K; ++ibl) {
            quantize_row_iq3_ks_impl(QK_K, x + QK_K*ibl, (char *)(y + ibl), all_scales, weight, iq3nl_values, w ? w + QK_K*ibl : nullptr, 5);
        }
    });
    return nrows*row_size;
}

size_t quantize_iq4_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    const size_t row_size = (n_per_row/QK_K)*sizeof(block_iq4_ks);
    float all_scales[QK_K/32], weight[32];
    QHelper helper(imatrix, n_per_row, 32);
    helper.quantize(nrows, src, dst, row_size, [&] (const float * x, void * vy, int n, const float * w) {
        block_iq4_ks * y = (block_iq4_ks *) vy;
        // ik_llama.cpp writes a float scale followed by the block
        alignas(4) char tmp[sizeof(float) + sizeof(ik_block_iq4_ks)];
        for (int ibl = 0; ibl < n/QK_K; ++ibl) {
            quantize_row_iq4_k_impl_bs32(QK_K, x + QK_K*ibl, tmp, all_scales, weight, iq4k_values, w ? w + QK_K*ibl : nullptr, 7);
            float d; std::memcpy(&d, tmp, sizeof(float));
            y[ibl].d = GGML_FP32_TO_FP16(d);
            std::memcpy(y[ibl].scales, tmp + sizeof(float), sizeof(ik_block_iq4_ks));
        }
    });
    return nrows*row_size;
}

size_t quantize_iq4_kss(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    const size_t row_size = (n_per_row/QK_K)*sizeof(block_iq4_kss);
    float all_scales[QK_K/32], weight[32];
    const uint16_t * table = scramble_table();
    QHelper helper(imatrix, n_per_row, 32);
    helper.quantize(nrows, src, dst, row_size, [&] (const float * x, void * vy, int n, const float * w) {
        block_iq4_kss * y = (block_iq4_kss *) vy;
        alignas(4) char tmp[sizeof(float) + sizeof(ik_block_iq4_kss)];
        for (int ibl = 0; ibl < n/QK_K; ++ibl) {
            quantize_row_iq4_kss_impl(QK_K, x + QK_K*ibl, tmp, all_scales, weight, iq4k_values, w ? w + QK_K*ibl : nullptr, table, 7);
            float d; std::memcpy(&d, tmp, sizeof(float));
            y[ibl].d = GGML_FP32_TO_FP16(d);
            std::memcpy(y[ibl].qs, tmp + sizeof(float), sizeof(ik_block_iq4_kss));
        }
    });
    return nrows*row_size;
}

void quantize_row_iq2_k_ref  (const float * GGML_RESTRICT x, block_iq2_k   * GGML_RESTRICT y, int64_t k) { quantize_iq2_k  (x, y, 1, k, nullptr); }
void quantize_row_iq3_k_ref  (const float * GGML_RESTRICT x, block_iq3_k   * GGML_RESTRICT y, int64_t k) { quantize_iq3_k  (x, y, 1, k, nullptr); }
void quantize_row_iq4_k_ref  (const float * GGML_RESTRICT x, block_iq4_k   * GGML_RESTRICT y, int64_t k) { quantize_iq4_k  (x, y, 1, k, nullptr); }
void quantize_row_iq2_ks_ref (const float * GGML_RESTRICT x, block_iq2_ks  * GGML_RESTRICT y, int64_t k) { quantize_iq2_ks (x, y, 1, k, nullptr); }
void quantize_row_iq2_kl_ref (const float * GGML_RESTRICT x, block_iq2_kl  * GGML_RESTRICT y, int64_t k) { quantize_iq2_kl (x, y, 1, k, nullptr); }
void quantize_row_iq3_ks_ref (const float * GGML_RESTRICT x, block_iq3_ks  * GGML_RESTRICT y, int64_t k) { quantize_iq3_ks (x, y, 1, k, nullptr); }
void quantize_row_iq4_ks_ref (const float * GGML_RESTRICT x, block_iq4_ks  * GGML_RESTRICT y, int64_t k) { quantize_iq4_ks (x, y, 1, k, nullptr); }
void quantize_row_iq4_kss_ref(const float * GGML_RESTRICT x, block_iq4_kss * GGML_RESTRICT y, int64_t k) { quantize_iq4_kss(x, y, 1, k, nullptr); }

//
// dequantization. IQ2_K, IQ3_K and IQ4_K are the functions of ik_llama.cpp
//

void dequantize_row_iq2_k(const block_iq2_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            float dl1 = d * ((x[i].scales[ib32] & 0xf) - 8);
            float dl2 = d * ((x[i].scales[ib32] >>  4) - 8);
            const int8_t * values1 = extra & 1 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = extra & 2 ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 2;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[(qs[j+ 0] >> shift) & 3];
                y[j+16] = dl2 * values2[(qs[j+16] >> shift) & 3];
            }
            y += 32;
            shift += 2;
            if (shift == 8) { qs += 32; shift = 0; }
        }

    }

}

void dequantize_row_iq3_k(const block_iq3_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;

        uint16_t sh = x[i].scales_h;
        uint16_t extra = x[i].extra;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            float dl1 = d * ((2*(x[i].scales_l[ib32] & 0xf) + 1) * ((sh & 1) ? -1 : 1));
            float dl2 = d * ((2*(x[i].scales_l[ib32] >>  4) + 1) * ((sh & 2) ? -1 : 1));
            sh >>= 2;
            const int8_t * values1 = extra & 1 ? iq3nl_values + 8 : iq3nl_values;
            const int8_t * values2 = extra & 2 ? iq3nl_values + 8 : iq3nl_values;
            extra >>= 2;
            int shift_l = 2*(ib32%4);
            int shift_h = ib32%8;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[((qs[j+ 0] >> shift_l) & 3) | (((qh[j+ 0] >> shift_h) & 1) << 2)];
                y[j+16] = dl2 * values2[((qs[j+16] >> shift_l) & 3) | (((qh[j+16] >> shift_h) & 1) << 2)];
            }
            y += 32;
            if (shift_l == 6) qs += 32;
        }

    }
}

void dequantize_row_iq4_k(const block_iq4_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = GGML_FP16_TO_FP32(x[i].d);

        uint16_t extra = x[i].extra;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const uint8_t sh = x[i].scales_h[ib/2] >> 4*(ib%2);
            const float dl1 = d * (((x[i].scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32);
            const float dl2 = d * (((x[i].scales_l[ib] >>  4) | ((sh << 2) & 0x30)) - 32);
            const int8_t * values1 = extra & 1 ? iq4k_values + 16 : iq4k_values;
            const int8_t * values2 = extra & 2 ? iq4k_values + 16 : iq4k_values;
            extra >>= 2;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[qs[j] & 0xf];
                y[j+16] = dl2 * values2[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

void dequantize_row_iq2_ks(const block_iq2_ks * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        uint16_t extra = x[i].extra;
        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            const float dl1 = d * (((x[i].scales[ib64] & 0xf) | ((extra >> 4) & 0x10)) - 16);
            const float dl2 = d * (((x[i].scales[ib64] >>  4) | ((extra >> 5) & 0x10)) - 16);
            const int8_t * values1 = extra & 1 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = extra & 2 ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 2;
            for (int j = 0; j < 32; ++j) {
                y[j+ 0] = dl1 * values1[(qs[j] >> (shift+0)) & 3];
                y[j+32] = dl2 * values2[(qs[j] >> (shift+2)) & 3];
            }
            y += 64;
            shift += 4;
            if (shift == 8) { qs += 32; shift = 0; }
        }
    }
}

void dequantize_row_iq2_kl(const block_iq2_kl * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint16_t scales_h = x[i].scales_h;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            const float dl1 = d * (int(((x[i].scales_l[(2*ib64+0)%4] >> 4*(ib64/2)) & 0xf) | (((scales_h >> (4*ib64+0)) & 3) << 4)) - 32);
            const float dl2 = d * (int(((x[i].scales_l[(2*ib64+1)%4] >> 4*(ib64/2)) & 0xf) | (((scales_h >> (4*ib64+2)) & 3) << 4)) - 32);
            for (int j = 0; j < 16; ++j) {
                const uint16_t v1 = iq2kl_values[(qs[j] & 0xf) | (((qh[j] >> (2*ib64+0)) & 1) << 4)];
                const uint16_t v2 = iq2kl_values[(qs[j] >>  4) | (((qh[j] >> (2*ib64+1)) & 1) << 4)];
                y[2*j+ 0] = dl1 * (int8_t)(v1 & 0xff);
                y[2*j+ 1] = dl1 * (int8_t)(v1 >> 8);
                y[2*j+32] = dl2 * (int8_t)(v2 & 0xff);
                y[2*j+33] = dl2 * (int8_t)(v2 >> 8);
            }
            y  += 64;
            qs += 16;
        }
    }
}

void dequantize_row_iq3_ks(const block_iq3_ks * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    float dl[8];
    for (int64_t ibl = 0; ibl < nb; ++ibl) {
        const float d = GGML_FP16_TO_FP32(x[ibl].d);
        for (int j = 0; j < 4; ++j) {
            const int ls1 = (x[ibl].scales[j] & 0xf) | (((x[ibl].extra >> (j+0)) & 1) << 4);
            const int ls2 = (x[ibl].scales[j] >>  4) | (((x[ibl].extra >> (j+4)) & 1) << 4);
            dl[j+0] = d*(ls1 - 16);
            dl[j+4] = d*(ls2 - 16);
        }
        const uint8_t * qs = x[ibl].qs;
        const uint8_t * qh = x[ibl].qh;
        for (int i128 = 0; i128 < QK_K/128; ++i128) {
            for (int ib = 0; ib < 4; ++ib) {
                const int8_t * values = iq3nl_values + ((x[ibl].extra >> (8 + (4*i128+ib)) & 1) << 3);
                for (int j = 0; j < 32; ++j) {
                    y[j] = dl[4*i128 + ib] * values[((qs[j] >> 2*ib) & 3) | (((qh[j] >> (4*i128+ib)) & 1) << 2)];
                }
                y += 32;
            }
            qs += 32;
        }
    }
}

void dequantize_row_iq4_ks(const block_iq4_ks * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int64_t ibl = 0; ibl < nb; ++ibl) {
        const float d = GGML_FP16_TO_FP32(x[ibl].d);
        const uint8_t * qs = x[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float dl = d * ((int)(x[ibl].scales[ib] & 254) - 127);
            const int8_t * values = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl * values[qs[j] & 0xf];
                y[j+16] = dl * values[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

void dequantize_row_iq4_kss(const block_iq4_kss * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    uint16_t aux16[8];
    const uint8_t * aux8 = (const uint8_t *)aux16;
    for (int64_t ibl = 0; ibl < nb; ++ibl) {
        const float d = GGML_FP16_TO_FP32(x[ibl].d);
        const uint16_t * qs = x[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int16_t ls = 0;
            for (int j = 0; j < 8; ++j) {
                aux16[j] = qs[j] & 0xfffe;
                aux16[j] ^= (aux16[j] >> 1);
                ls |= (qs[j] & 1) << j;
            }
            const int8_t * values = iq4k_values + ((ls & 1) << 4);
            const float dl = d * ((ls & 254) - 127);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl * values[aux8[j] & 0xf];
                y[j+16] = dl * values[aux8[j] >>  4];
            }
            y  += 32;
            qs += 8;
        }
    }
}

} // extern "C"
