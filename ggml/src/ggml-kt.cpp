// Trellis quant IQ3_KT, ported from ik_llama.cpp (https://github.com/ikawrakow/ik_llama.cpp)
// Copyright (C) 2025 Iwan Kawrakow, MIT license
//
// Differences to ik_llama.cpp:
//   - the scale is stored as fp16 in each block of 256 instead of one float per row (no row meta data in ggml)
//   - the scale is optimized per block of 256 instead of per row
//   - rows must be a multiple of 256 (no tail blocks)

#include "ggml-quants.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// GGML_KT_NO_SIMD forces the scalar search, for testing
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(GGML_KT_NO_SIMD)
#include <immintrin.h>
#define GGML_KT_X86_DISPATCH 1
#endif

namespace {

constexpr int kSuperBlockSize = QK_K;
constexpr int kBlockSize      = 32;
constexpr int kNblock         = kSuperBlockSize/kBlockSize;

inline int kt_nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i; std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// next value of the integer trellis, result is in [-126, 126]
inline int kt_next(uint32_t & x) {
    x *= 0xCBAC1FEDu;
    const uint32_t s = x & 0x3f3f3f3fu;
    return int(s & 0xff) + int((s >> 8) & 0xff) + int((s >> 16) & 0xff) + int(s >> 24) - 126;
}

inline float kt_fp16_scale(float d) {
    // d is stored as fp16(d * 2^16)
    const float v = d / GGML_KT_D_SCALE;
    return std::min(std::max(v, -65504.f), 65504.f);
}

#ifdef GGML_KT_X86_DISPATCH
bool kt_has_avx2() {
    static const bool result = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return result;
}

// index of the point with the smallest weighted distance to x. Points are stored as soa[k*npoint + p], npoint%8 == 0
template <int ndim>
__attribute__((target("avx2,fma")))
int kt_best_point_avx2(const float * soa, int npoint, const float * x, const float * w) {
    __m256 vx[ndim], vw[ndim];
    for (int k = 0; k < ndim; ++k) {
        vx[k] = _mm256_set1_ps(x[k]);
        vw[k] = _mm256_set1_ps(w[k]);
    }
    __m256  best  = _mm256_set1_ps(INFINITY);
    __m256i ibest = _mm256_set1_epi32(-1);
    __m256i idx   = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i add8 = _mm256_set1_epi32(8);
    for (int p = 0; p < npoint; p += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(soa + p), vx[0]);
        __m256 s = _mm256_mul_ps(_mm256_mul_ps(vw[0], d), d);
        for (int k = 1; k < ndim; ++k) {
            d = _mm256_sub_ps(_mm256_loadu_ps(soa + k*npoint + p), vx[k]);
            s = _mm256_fmadd_ps(_mm256_mul_ps(vw[k], d), d, s);
        }
        const __m256 mask = _mm256_cmp_ps(s, best, _CMP_LT_OQ);
        best  = _mm256_min_ps(s, best);
        ibest = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(ibest), _mm256_castsi256_ps(idx), mask));
        idx   = _mm256_add_epi32(idx, add8);
    }
    float sb[8]; int ib[8];
    _mm256_storeu_ps(sb, best);
    _mm256_storeu_si256((__m256i *)ib, ibest);
    float smin = INFINITY; int imin = -1;
    for (int i = 0; i < 8; ++i) {
        if (sb[i] < smin) { smin = sb[i]; imin = ib[i]; }
    }
    return imin;
}
#endif

template <int ndim>
int kt_best_point_scalar(const float * soa, int npoint, const float * x, const float * w) {
    float best = INFINITY; int ibest = -1;
    for (int p = 0; p < npoint; ++p) {
        float s = 0;
        for (int k = 0; k < ndim; ++k) {
            const float d = soa[k*npoint + p] - x[k];
            s += w[k]*d*d;
        }
        if (s < best) { best = s; ibest = p; }
    }
    return ibest;
}

template <int ndim>
inline int kt_best_point(const float * soa, int npoint, const float * x, const float * w) {
#ifdef GGML_KT_X86_DISPATCH
    if (kt_has_avx2()) {
        return kt_best_point_avx2<ndim>(soa, npoint, x, w);
    }
#endif
    return kt_best_point_scalar<ndim>(soa, npoint, x, w);
}

// Codebook + search structure. Each seed i (0...2^num_bits-1) generates group_size values via kt_next().
// The codebook points are binned into clusters, and each point is added to its num_neighbours closest clusters.
// A search first picks the cluster from the bin of the input, then scans the points of that cluster.
template <int group_size, int num_bits, bool is_abs>
class KTQuantizer {
public:
    static constexpr int kGroupSize = group_size;
    static constexpr int kNg        = kBlockSize/kGroupSize;
    static constexpr int kNumVal    = 1 << num_bits;

    KTQuantizer(int num_clusters, int num_neighbours, uint32_t offset) {
        m_values.resize(kNumVal*kGroupSize);
        for (int i = 0; i < kNumVal; ++i) {
            set_values(i, m_values.data() + i*kGroupSize, 1.0f, offset);
        }
        cluster_points(num_clusters);
        finalize_clusters(num_neighbours);
    }

    const float * values() const { return m_values.data(); }

    static void set_values(uint32_t i, float * result, float scale, uint32_t offset) {
        uint32_t x = i + offset;
        for (int k = 0; k < kGroupSize; ++k) {
            const int v = kt_next(x);
            result[k] = scale*(is_abs ? std::abs(v) : v);
        }
    }

    void find_best_match(float d, const float * xb, const float * weight, int * best_idx) const {
        if (!d) {
            std::memset(best_idx, 0, kNg*sizeof(int));
            return;
        }
        const float id = 1/d;
        float xs[kGroupSize];
        for (int l = 0; l < kNg; ++l) {
            const float * xl = xb + kGroupSize*l;
            const float * wl = weight + kGroupSize*l;
            for (int k = 0; k < kGroupSize; ++k) xs[k] = id*xl[k];
            const int ic = cluster_of(xs);
            const Cluster & c = m_cluster[ic];
            if (c.points.empty()) {
                best_idx[l] = kt_best_point<kGroupSize>(m_all_soa.data(), kNumVal, xs, wl);
            } else {
                const int j = kt_best_point<kGroupSize>(c.soa.data(), int(c.points.size()), xs, wl);
                GGML_ASSERT(j >= 0);
                best_idx[l] = c.points[j];
            }
        }
    }

    // returns the best scale and the corresponding score for the given seeds
    std::pair<float, float> find_best_scale(const float * xb, const float * weight, const int * best_idx) const {
        float sumqx = 0, sumq2 = 0;
        for (int l = 0; l < kNg; ++l) {
            const float * xl = xb + kGroupSize*l;
            const float * wl = weight + kGroupSize*l;
            const float * ql = m_values.data() + kGroupSize*best_idx[l];
            for (int k = 0; k < kGroupSize; ++k) {
                sumqx += wl[k]*ql[k]*xl[k];
                sumq2 += wl[k]*ql[k]*ql[k];
            }
        }
        return sumq2 > 0 ? std::make_pair(sumqx/sumq2, sumqx*sumqx/sumq2) : std::make_pair(0.f, 0.f);
    }

    // importance weights for one block of 256
    static void set_weights(float sigma2_scale, const float * x, const float * imatrix, float * w) {
        constexpr float kEps2   = 1e-14f;
        constexpr float kWeight = 1e-4f;
        float sumx2 = 0;
        for (int j = 0; j < kSuperBlockSize; ++j) sumx2 += x[j]*x[j];
        if (sumx2 < kEps2*kSuperBlockSize) {
            for (int j = 0; j < kSuperBlockSize; ++j) w[j] = kWeight;
            return;
        }
        const float sigma2 = sigma2_scale*sumx2/kSuperBlockSize;
        if (imatrix) {
            for (int ib = 0; ib < kNblock; ++ib) {
                const float * qw = imatrix + ib*kBlockSize;
                const float * xb = x + ib*kBlockSize;
                float * wb = w + ib*kBlockSize;
                float sumwx = 0, sumw2 = 0, sumbx2 = 0;
                for (int j = 0; j < kBlockSize; ++j) {
                    wb[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                    sumwx  += wb[j]*std::abs(xb[j]);
                    sumw2  += wb[j]*wb[j];
                    sumbx2 += xb[j]*xb[j];
                }
                if (sumbx2 < kEps2 || sumw2 < kEps2 || sumwx < kEps2) {
                    for (int j = 0; j < kBlockSize; ++j) wb[j] = kWeight;
                }
            }
        } else {
            for (int j = 0; j < kSuperBlockSize; ++j) w[j] = 0.25f*sigma2 + x[j]*x[j];
        }
    }

private:
    struct Cluster {
        std::vector<int>   points;
        std::vector<float> soa;
    };

    static int bin5(float x) {
        return x < -48.f ? 0 : x < -16.f ? 1 : x < 16.f ? 2 : x < 48.f ? 3 : 4;
    }

    int cluster_of(const float * x) const {
        int u = 0;
        if constexpr (kGroupSize == 8) {
            for (int k = 0; k < kGroupSize; ++k) if (x[k] > m_mid[k]) u |= (1 << k);
        } else {
            int s = 1;
            for (int k = 0; k < kGroupSize; ++k) { u += s*bin5(x[k]); s *= 5; }
        }
        return u;
    }

    void cluster_points(int ncluster) {
        static_assert(kGroupSize == 8 || kGroupSize == 4, "unsupported group size");
        // only the binned variants: 8 dims with 256 clusters (IQ3_KT) or 4 dims with 625 clusters
        GGML_ASSERT((kGroupSize == 8 && ncluster == 256) || (kGroupSize == 4 && ncluster == 625));
        const int npoint = kNumVal;
        if constexpr (kGroupSize == 8) {
            std::vector<int> P(npoint);
            for (int k = 0; k < kGroupSize; ++k) {
                for (int ip = 0; ip < npoint; ++ip) P[ip] = int(m_values[ip*kGroupSize + k]);
                std::sort(P.begin(), P.end());
                m_mid[k] = npoint%2 == 0 ? 0.5f*(P[npoint/2] + P[npoint/2-1]) : P[npoint/2];
            }
        }
        std::vector<float> sump(ncluster*kGroupSize, 0.f);
        std::vector<int>   counts(ncluster, 0);
        for (int ip = 0; ip < npoint; ++ip) {
            const float * vp = m_values.data() + kGroupSize*ip;
            const int u = cluster_of(vp);
            ++counts[u];
            for (int k = 0; k < kGroupSize; ++k) sump[kGroupSize*u + k] += vp[k];
        }
        m_clusters.assign(ncluster*kGroupSize, 0.f);
        for (int ic = 0; ic < ncluster; ++ic) {
            if (!counts[ic]) continue;
            for (int k = 0; k < kGroupSize; ++k) m_clusters[ic*kGroupSize + k] = sump[ic*kGroupSize + k]/counts[ic];
        }
    }

    void finalize_clusters(int num_neighbours) {
        const int ncluster = int(m_clusters.size())/kGroupSize;
        std::vector<std::vector<int>> p_in_cluster(ncluster);
        std::vector<int>   which_cluster(num_neighbours*kNumVal);
        std::vector<int>   ibest(num_neighbours);
        std::vector<float> best(num_neighbours);
        for (int ip = 0; ip < kNumVal; ++ip) {
            const float * vp = m_values.data() + ip*kGroupSize;
            for (int j = 0; j < num_neighbours; ++j) {
                best[j] = INFINITY; ibest[j] = -1;
            }
            for (int ic = 0; ic < ncluster; ++ic) {
                const float * vc = m_clusters.data() + ic*kGroupSize;
                float dist2 = 0;
                for (int k = 0; k < kGroupSize; ++k) {
                    const float d = vp[k] - vc[k]; dist2 += d*d;
                }
                for (int j = 0; j < num_neighbours; ++j) {
                    if (dist2 < best[j]) {
                        for (int k = num_neighbours-1; k > j; --k) {
                            best[k] = best[k-1]; ibest[k] = ibest[k-1];
                        }
                        best[j] = dist2; ibest[j] = ic;
                        break;
                    }
                }
            }
            for (int j = 0; j < num_neighbours; ++j) {
                GGML_ASSERT(ibest[j] >= 0);
                p_in_cluster[ibest[j]].push_back(ip);
            }
            std::memcpy(which_cluster.data() + num_neighbours*ip, ibest.data(), num_neighbours*sizeof(int));
        }
        // pad each non-empty cluster to a multiple of 8 points with the closest points not yet in it
        std::vector<std::pair<float, int>> extra;
        extra.reserve(kNumVal);
        for (int ic = 0; ic < ncluster; ++ic) {
            auto & points = p_in_cluster[ic];
            if (points.size()%8 == 0) continue;
            extra.clear();
            const float * vc = m_clusters.data() + ic*kGroupSize;
            for (int ip = 0; ip < kNumVal; ++ip) {
                bool can_add = true;
                for (int j = 0; j < num_neighbours; ++j) {
                    if (which_cluster[num_neighbours*ip+j] == ic) { can_add = false; break; }
                }
                if (!can_add) continue;
                const float * vp = m_values.data() + ip*kGroupSize;
                float dist2 = 0;
                for (int k = 0; k < kGroupSize; ++k) {
                    const float d = vp[k] - vc[k]; dist2 += d*d;
                }
                extra.push_back(std::make_pair(dist2, ip));
            }
            std::sort(extra.begin(), extra.end());
            const int nadd = 8*((points.size()+7)/8) - points.size();
            for (int i = 0; i < nadd; ++i) points.push_back(extra[i].second);
            GGML_ASSERT(points.size()%8 == 0);
        }
        m_cluster.resize(ncluster);
        for (int ic = 0; ic < ncluster; ++ic) {
            Cluster & c = m_cluster[ic];
            c.points = std::move(p_in_cluster[ic]);
            const int n = int(c.points.size());
            c.soa.resize(n*kGroupSize);
            for (int p = 0; p < n; ++p) {
                for (int k = 0; k < kGroupSize; ++k) c.soa[k*n + p] = m_values[c.points[p]*kGroupSize + k];
            }
        }
        m_all_soa.resize(kNumVal*kGroupSize);
        for (int p = 0; p < kNumVal; ++p) {
            for (int k = 0; k < kGroupSize; ++k) m_all_soa[k*kNumVal + p] = m_values[p*kGroupSize + k];
        }
    }

    std::vector<float>   m_values;
    std::vector<float>   m_clusters;
    std::vector<Cluster> m_cluster;
    std::vector<float>   m_all_soa;
    float                m_mid[kGroupSize] = {};
};

using QuantizerIQ3KT = KTQuantizer<8, 16, true>;

constexpr uint32_t kIQ3KTOffset  = 4096;

// ik_llama.cpp applies this factor to the IQ3_KT scale at run time, we fold it into the stored scale
constexpr float kIQ3KTFudge = 1.01f;

const QuantizerIQ3KT & iq3kt_quantizer() {
    static const QuantizerIQ3KT quantizer(256, 8, kIQ3KTOffset);
    return quantizer;
}

// x, weights: one block of 256. qtmp: scratch of 256 floats
void quantize_block_iq3_kt(const float * x, block_iq3_kt * y, const float * weights, float amax_row, float * qtmp) {
    using Q = QuantizerIQ3KT;
    constexpr float kStep = 8.0f;

    const Q & quantizer = iq3kt_quantizer();

    std::memset(y, 0, sizeof(block_iq3_kt));

    float scales[kNblock];
    int   best_idx[2*Q::kNg];
    float xaux[kBlockSize];

    float max_scale = 0;

    for (int ib = 0; ib < kNblock; ++ib) {
        const float * xb = x + kBlockSize*ib;
        const float * wb = weights + kBlockSize*ib;
        float * qb = qtmp + kBlockSize*ib;
        float amax = 0;
        for (int j = 0; j < kBlockSize; ++j) {
            xaux[j] = std::abs(xb[j]);
            amax = std::max(amax, xaux[j]);
        }
        scales[ib] = 0;
        std::memset(qb, 0, kBlockSize*sizeof(float));
        if (amax < 1e-16f) {
            continue;
        }
        const float scale_0 = std::max(84.f, 123.f*amax/amax_row);
        float best = 0;
        for (int itry = -3; itry <= 3; ++itry) {
            quantizer.find_best_match(amax/(scale_0 + kStep*itry), xaux, wb, best_idx);
            auto [d, score] = quantizer.find_best_scale(xaux, wb, best_idx);
            if (score > best) {
                best = score;
                scales[ib] = d;
                std::memcpy(best_idx + Q::kNg, best_idx, Q::kNg*sizeof(int));
            }
        }
        if (best > 0) {
            for (int ig = 0; ig < Q::kNg; ++ig) {
                std::memcpy(qb + Q::kGroupSize*ig, quantizer.values() + Q::kGroupSize*best_idx[Q::kNg + ig], Q::kGroupSize*sizeof(float));
            }
        }
        max_scale = std::max(max_scale, scales[ib]);
    }

    if (max_scale <= 0) {
        return;
    }

    float d = max_scale/15;
    float best = 0;
    for (int itry = -9; itry <= 9; ++itry) {
        const float id = (itry*0.2f + 15)/max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < kNblock; ++ib) {
            const float * xb = x + kBlockSize*ib;
            const float * qb = qtmp + kBlockSize*ib;
            const float * wb = weights + kBlockSize*ib;
            const float dl = std::max(0, std::min(15, kt_nearest_int(id*scales[ib])));
            for (int j = 0; j < kBlockSize; ++j) {
                const float q = dl*qb[j];
                sumqx += wb[j]*std::abs(xb[j])*q;
                sumq2 += wb[j]*q*q;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
        }
    }

    const float id = d ? 1/d : 0.f;
    for (int ib = 0; ib < kNblock/2; ++ib) {
        const int ls1 = std::max(0, std::min(15, kt_nearest_int(id*scales[ib])));
        const int ls2 = std::max(0, std::min(15, kt_nearest_int(id*scales[ib + kNblock/2])));
        y->scales[ib] = ls1 | (ls2 << 4);
    }

    uint16_t ql[QK_K/8];
    float sumqx = 0, sumq2 = 0;
    for (int ib = 0; ib < kNblock; ++ib) {
        const float * xb = x + kBlockSize*ib;
        const float * wb = weights + kBlockSize*ib;
        for (int j = 0; j < kBlockSize; ++j) {
            xaux[j] = std::abs(xb[j]);
            if (xb[j] < 0) y->qh[j] |= (1 << ib);
        }
        const int ls = (y->scales[ib%(kNblock/2)] >> 4*(ib/(kNblock/2))) & 0xf;
        quantizer.find_best_match(d*ls, xaux, wb, best_idx);
        for (int j = 0; j < Q::kNg; ++j) {
            ql[ib*Q::kNg + j] = best_idx[j];
            const float * xl = xaux + Q::kGroupSize*j;
            const float * wl = wb + Q::kGroupSize*j;
            const float * vl = quantizer.values() + best_idx[j]*Q::kGroupSize;
            for (int k = 0; k < Q::kGroupSize; ++k) {
                const float q = vl[k]*ls;
                sumqx += wl[k]*xl[k]*q;
                sumq2 += wl[k]*q*q;
            }
        }
    }
    std::memcpy(y->ql, ql, sizeof(ql));
    if (sumq2 > 0) {
        d = sumqx/sumq2;
    }
    y->d = GGML_FP32_TO_FP16(kt_fp16_scale(d*kIQ3KTFudge));
}

} // namespace

extern "C" {

size_t quantize_iq3_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row % QK_K == 0);
    constexpr float kSigmaScale = 2.0f;
    const int64_t nblock = n_per_row/QK_K;
    std::vector<float> weights(QK_K), qtmp(QK_K);
    block_iq3_kt * y = (block_iq3_kt *) dst;
    for (int64_t row = 0; row < nrows; ++row) {
        const float * x = src + row*n_per_row;
        float amax_row = 0;
        for (int64_t j = 0; j < n_per_row; ++j) amax_row = std::max(amax_row, std::abs(x[j]));
        for (int64_t ibl = 0; ibl < nblock; ++ibl) {
            if (!amax_row) {
                std::memset(y, 0, sizeof(block_iq3_kt));
            } else {
                QuantizerIQ3KT::set_weights(kSigmaScale, x + ibl*QK_K, imatrix ? imatrix + ibl*QK_K : nullptr, weights.data());
                quantize_block_iq3_kt(x + ibl*QK_K, y, weights.data(), amax_row, qtmp.data());
            }
            ++y;
        }
    }
    return nrows*nblock*sizeof(block_iq3_kt);
}

void quantize_row_iq3_kt_ref(const float * GGML_RESTRICT x, block_iq3_kt * GGML_RESTRICT y, int64_t k) {
    quantize_iq3_kt(x, y, 1, k, nullptr);
}

void dequantize_row_iq3_kt(const block_iq3_kt * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * GGML_KT_D_SCALE;
        for (int ib = 0; ib < kNblock; ++ib) {
            const float dl = d * ((x[i].scales[ib%4] >> 4*(ib/4)) & 0xf);
            for (int ig = 0; ig < 4; ++ig) {
                const int g = 4*ib + ig;
                uint32_t idx = (x[i].ql[2*g] | (x[i].ql[2*g+1] << 8)) + kIQ3KTOffset;
                for (int j = 0; j < 8; ++j) {
                    const int v = std::abs(kt_next(idx));
                    const int l = 8*ig + j;
                    y[32*ib + l] = (x[i].qh[l] >> ib) & 1 ? -dl*v : dl*v;
                }
            }
        }
        y += QK_K;
    }
}

}
