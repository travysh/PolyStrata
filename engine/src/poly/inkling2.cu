// src/poly/inkling2.cu - the kernels of Inkling 2's layers (inkling2_kernels.hpp).  The reference for the math is
// llama.cpp's inkling2 graph (src/models/inkling2.cpp, inkling2-loop.cpp).
//
// A token (or a token's head) a block, sums in a fixed order so that a token comes out the same in any company.
// The attention of a few tokens reads a key or a value with a warp, its lanes side by side: ink_attend().
// q81_div(): a block's scale divided as native_mmvq.cu's --use_fast_math division does it (q8_1_finite.hpp)
#define STRATA_Q81_FASTMATH_DIV 1
#include "inkling2_kernels.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if !defined(__HIPCC__)
#include <cuda_fp8.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#if defined(__HIPCC__)
#include <atomic>

#include "strata/kernels/gfx_arch.hpp"
#endif

namespace strata::poly {
namespace {

inline cudaStream_t cs(void* s) { return (cudaStream_t) s; }
using strata::kernels::launch_pdl;
using strata::kernels::pdl_trigger;
using strata::kernels::pdl_wait;
inline int grid(size_t n, int per) { return (int) ((n + (size_t) per - 1) / (size_t) per); }

// A launch the driver refused leaves its error for whoever asks next: ask here, so that it names its kernel.
void launched(const char* what) {
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

#if defined(__HIPCC__)
// HIP has __ldcs and __ldcg for halves only.  The DSpark screen's are cache hints (a copy's words read once, the
// tiles' candidates from earlier launches), so on HIP they are plain loads.
template <typename T> __device__ __forceinline__ T __ldcs(const T* p) { return *p; }
template <typename T> __device__ __forceinline__ T __ldcg(const T* p) { return *p; }
#endif

constexpr int WARP = 32;   // the lanes a shuffle spans: a warp, or half of a 64-lane wavefront

// The warp's sum of x, in every lane.
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o, WARP));
    return x;
}

// A controller's weight: F32, or BF16 (--loop-bf16) widened as bf16_to_f32() does it.
__device__ __forceinline__ float wval(const float* w, size_t i) { return w[i]; }
__device__ __forceinline__ float wval(const uint16_t* w, size_t i) { return __uint_as_float(uint32_t(w[i]) << 16); }

struct QB {   // a q8_1 block, as native_quantize_q8_1() writes it
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(QB) == 36, "q8_1 block");

// Value i of a row into its q8_1 blocks, native_quantize_q8_1_kernel's arithmetic; a warp's 32 lanes hold one block.
__device__ __forceinline__ void q8_put(QB* y, int i, float xi) {
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = strata::kernels::q8_1_finite(strata::kernels::q81_div(amax, 127.0f));
    const int8_t q = strata::kernels::q8_1_quant(xi, d, amax);
    y[i / 32].qs[i % 32] = q;
    if (i % 32 == 0) y[i / 32].ds = strata::kernels::q8_1_ds(d, sum);
}

// native_add_rms_norm_q8_1_kernel<blockDim>'s norm (no add) of a row x, a thread over its values, into y and q.
__device__ __forceinline__ void add_rms_q8(const float* x, const float* __restrict__ w, float* y, QB* q, int n,
                                           float eps, float* part) {
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = x[i];
        ss = __fmaf_rn(v, v, ss);
    }
    ss = warp_sum(ss);
    if ((tid & 31) == 0) part[tid >> 5] = ss;
    __syncthreads();
    float sum = 0.0f;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) sum = __fadd_rn(sum, part[k]);
    __syncthreads();
    const float scale = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, float(n)), eps)));
    for (int i = tid; i < n; i += blockDim.x) {
        const float xi = __fmul_rn(__fmul_rn(x[i], scale), w[i]);
        y[i] = xi;
        q8_put(q, i, xi);
    }
}

constexpr int kNormHeld = 4;   // the values a thread of a norm keeps: sum_threads() gives it at most 4 below 4,096

// add_rms_q8() of the H values a thread holds (value h is the row's threadIdx.x + h * blockDim.x), the same arithmetic
template <int H>
__device__ __forceinline__ void add_rms_q8_held(const float (&v)[H], const float* __restrict__ w, float* y, QB* q,
                                                int n, float eps, float* part) {
    const int tid = threadIdx.x;
    float ss = 0.0f;
#pragma unroll
    for (int h = 0; h < H; ++h)
        if (tid + h * int(blockDim.x) < n) ss = __fmaf_rn(v[h], v[h], ss);
    ss = warp_sum(ss);
    if ((tid & 31) == 0) part[tid >> 5] = ss;
    __syncthreads();
    float sum = 0.0f;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) sum = __fadd_rn(sum, part[k]);
    __syncthreads();
    const float scale = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, float(n)), eps)));
#pragma unroll
    for (int h = 0; h < H; ++h) {
        const int i = tid + h * int(blockDim.x);
        if (i < n) {
            const float xi = __fmul_rn(__fmul_rn(v[h], scale), w[i]);
            y[i] = xi;
            q8_put(q, i, xi);
        }
    }
}

// The block's sum of x, the same in every thread: the warps' sums added in their order.  blockDim.x is a multiple
// of 32, and `part` has a float a warp.
__device__ __forceinline__ float block_sum(float x, float* part) {
    x = warp_sum(x);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float s = 0.0f;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) s += part[k];
    __syncthreads();   // part is free again
    return s;
}

__device__ __forceinline__ float block_max(float x, float* part) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o, WARP));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float m = part[0];
    for (int k = 1; k < int(blockDim.x) / WARP; ++k) m = fmaxf(m, part[k]);
    __syncthreads();
    return m;
}

// ---- the attention caches in FP8 E4M3 (InkAttn::kv8): a row the n_kv heads' dim codes, then a float scale a head,
// the head's largest magnitude over 448, so that its codes span E4M3's range ----
__device__ __forceinline__ size_t kv8_row(int n_kv, int dim) { return (size_t) n_kv * dim + (size_t) n_kv * 4; }

#if !defined(__HIPCC__)
__device__ __forceinline__ uint8_t e4m3_of(float x) { return (uint8_t) __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3); }
__device__ __forceinline__ float e4m3_val(uint8_t c) { return __half2float(__half(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3))); }
#else
__device__ __forceinline__ uint8_t e4m3_of(float) { return 0; }   // (ink_kv8_ok() is false under HIP)
__device__ __forceinline__ float e4m3_val(uint8_t) { return 0.0f; }
#endif

// A head of dim values v (a thread's values those of i = tid, tid + blockDim.x, ...) into row `cell` of an FP8
// cache, head h of its n_kv.  Every thread of the block takes part (block_max()).
__device__ void kv8_store(const float* v, uint8_t* cache, long long cell, int n_kv, int dim, int h, float* part) {
    float a = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) a = fmaxf(a, fabsf(v[i]));
    a = block_max(a, part);
    const float sc = a > 0.0f ? a / 448.0f : 1.0f, inv = 1.0f / sc;
    uint8_t* row = cache + (size_t) cell * kv8_row(n_kv, dim);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) row[(size_t) h * dim + i] = e4m3_of(v[i] * inv);
    if (threadIdx.x == 0) reinterpret_cast<float*>(row + (size_t) n_kv * dim)[h] = sc;
}

// ---- the attention caches in Q8_0 (InkAttn::kvq8, --kv q8_0): a row the n_kv heads' dim int8 codes, then a half
// scale for each block of 32 values (n_kv * dim / 32 of them, in the codes' order), ink_kvq8_row() bytes.  A block's
// scale is its largest magnitude over 127 and a code round(x / scale), as llama.cpp's quantize_row_q8_0 makes them.
// A row's codes start 4-aligned (dp4a can read them as they lie) and its scales 2-aligned. ----
constexpr int kKvF16 = 0, kKvFp8 = 1, kKvQ8 = 2;   // the token kernels' cache formats (InkAttn::kv8, ::kvq8)
// ink_score_kernel() of a Q8_0 cache with InkAttn::int8_dot (STRATA_KV_INT8): a token that reads kInt8From keys or
// more takes its scores by __dp4a, its queries quantized to Q8_0 codes too; the other tokens kKvQ8's way
constexpr int kKvQ8Dot = 3;
constexpr int kInt8From = 32768;
__device__ __forceinline__ size_t kvq8_row(int n_kv, int dim) { return (size_t) n_kv * dim + (size_t) n_kv * dim / 16; }
__device__ __forceinline__ const uint16_t* kvq8_scales(const uint8_t* row, int n_kv, int dim) {
    return reinterpret_cast<const uint16_t*>(row + (size_t) n_kv * dim);
}

// kv8_store() for a Q8_0 cache.  A block's 32 values are one warp's in one pass (dim and blockDim.x multiples of 32),
// so a warp finds its block's largest magnitude by itself.
__device__ void kvq8_store(const float* v, uint8_t* cache, long long cell, int n_kv, int dim, int h) {
    uint8_t* row = cache + (size_t) cell * kvq8_row(n_kv, dim);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float x = v[i];
        float a = fabsf(x);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o, 32));
        const float d = a / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        reinterpret_cast<int8_t*>(row)[(size_t) h * dim + i] = (int8_t) roundf(x * id);
        if ((i & 31) == 0)
            reinterpret_cast<uint16_t*>(row + (size_t) n_kv * dim)[((size_t) h * dim + i) >> 5] =
                __half_as_ushort(__float2half(d));
    }
}

template <int KV>
__device__ __forceinline__ void kv_store(const float* v, void* cache, long long cell, int n_kv, int dim, int h,
                                         float* part) {
    if (KV == kKvFp8) kv8_store(v, (uint8_t*) cache, cell, n_kv, dim, h, part);
    else if (KV == kKvQ8) kvq8_store(v, (uint8_t*) cache, cell, n_kv, dim, h);
}

constexpr int kQueries = 4;
constexpr int kMixThreads = 1024;
constexpr int kMixSums = 1 + kQueries;   // a state's sum of squares, and its product with each query

struct MixArgs {
    const float* s[InkStates::kMax];
    int n;
    const float* query;
    int nq;
    float* out[kQueries];
    const float* norm_w;   // with norm_out: out[0]'s rms_norm()
    float* norm_out;
    float eps;
    QB* q8;     // and the norm quantized (one row)
    int ffn;    // the norm is native_add_rms_norm_q8_1()'s
};

// A token a block.  A thread has every kMixThreads-th value of the row and reads each query's and each state's
// once: the states' squares and their products with the queries are summed over the warp, then over the warps in
// their order.
__device__ __forceinline__ void mix_row(const MixArgs& a, int n_embd) {
    __shared__ float part[InkStates::kMax * kMixSums][kMixThreads / WARP];
    __shared__ float whole[InkStates::kMax * kMixSums];
    __shared__ float coef[InkStates::kMax * kQueries];
    const size_t row = size_t(blockIdx.x) * n_embd;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float sum[InkStates::kMax][kMixSums];
#pragma unroll
    for (int s = 0; s < InkStates::kMax; ++s) {
#pragma unroll
        for (int k = 0; k < kMixSums; ++k) sum[s][k] = 0.0f;
    }
    for (int i = tid; i < n_embd; i += blockDim.x) {
        float q[kQueries];
#pragma unroll
        for (int j = 0; j < kQueries; ++j) q[j] = j < a.nq ? a.query[size_t(j) * n_embd + i] : 0.0f;
#pragma unroll
        for (int s = 0; s < InkStates::kMax; ++s) {
            if (s < a.n) {
                const float v = a.s[s][row + i];
                sum[s][0] += v * v;
#pragma unroll
                for (int j = 0; j < kQueries; ++j) sum[s][1 + j] += v * q[j];
            }
        }
    }
#pragma unroll
    for (int s = 0; s < InkStates::kMax; ++s) {
#pragma unroll
        for (int k = 0; k < kMixSums; ++k) {
            if (s < a.n && k <= a.nq) {
                const float w = warp_sum(sum[s][k]);
                if (lane == 0) part[s * kMixSums + k][warp] = w;
            }
        }
    }
    __syncthreads();
    if (tid < a.n * kMixSums && tid % kMixSums <= a.nq) {
        float t = 0.0f;
        for (int w = 0; w < (int) blockDim.x / 32; ++w) t += part[tid][w];
        whole[tid] = t;
    }
    __syncthreads();
    if (tid < a.nq) {
        const int j = tid;
        const float sc = 1.0f / sqrtf((float) n_embd);
        float v[InkStates::kMax], inv[InkStates::kMax];
        float m = -INFINITY;
        for (int s = 0; s < a.n; ++s) {
            inv[s] = 1.0f / sqrtf(whole[s * kMixSums] / (float) n_embd + 1e-6f);
            v[s] = whole[s * kMixSums + 1 + j] * inv[s] * sc;
            m = fmaxf(m, v[s]);
        }
        float total = 0.0f;
        for (int s = 0; s < a.n; ++s) {
            v[s] = expf(v[s] - m);
            total += v[s];
        }
        for (int s = 0; s < a.n; ++s) coef[s * kQueries + j] = v[s] / total * inv[s];
    }
    __syncthreads();
    for (int i = tid; i < n_embd; i += blockDim.x) {
        float x[InkStates::kMax];
#pragma unroll
        for (int s = 0; s < InkStates::kMax; ++s) x[s] = s < a.n ? a.s[s][row + i] : 0.0f;
#pragma unroll
        for (int j = 0; j < kQueries; ++j) {
            if (j < a.nq) {
                float acc = 0.0f;
#pragma unroll
                for (int s = 0; s < InkStates::kMax; ++s) {
                    if (s < a.n) acc += coef[s * kQueries + j] * x[s];
                }
                a.out[j][row + i] = acc;
            }
        }
    }
    if (a.norm_out && a.ffn) {
        __shared__ float fpart[kMixThreads / WARP];
        add_rms_q8(a.out[0] + row, a.norm_w, a.norm_out + row, a.q8 + size_t(blockIdx.x) * (n_embd / 32), n_embd,
                   a.eps, fpart);
    } else if (a.norm_out) {   // rms_norm_kernel's arithmetic, a thread over the values it wrote
        __shared__ float npart[kMixThreads / WARP];
        float ss = 0.0f;
        for (int i = tid; i < n_embd; i += blockDim.x) {
            const float v = a.out[0][row + i];
            ss += v * v;
        }
        ss = block_sum(ss, npart);
        const float scale = 1.0f / sqrtf(ss / float(n_embd) + a.eps);
        for (int i = tid; i < n_embd; i += blockDim.x) {
            const float v = a.out[0][row + i];
            const float y = a.norm_w ? v * scale * a.norm_w[i] : v * scale;
            a.norm_out[row + i] = y;
            if (a.q8) q8_put(a.q8 + size_t(blockIdx.x) * (n_embd / 32), i, y);
        }
    }
}

constexpr int kMixHeld = 3;   // the values of a row a thread of mix_held() holds: rows of up to 3 * kMixThreads

// mix_row() for NS states and NQ queries, a row of up to kMixHeld * kMixThreads values: a thread keeps the states'
// values it reads, and the outputs it writes, for what comes after (the outputs and the norm) instead of reading
// them again.  Every sum is mix_row()'s, term for term and in its order.
template <int NS, int NQ>
__device__ __forceinline__ void mix_held(const MixArgs& a, int n_embd) {
    __shared__ float part[InkStates::kMax * kMixSums][kMixThreads / WARP];
    __shared__ float whole[InkStates::kMax * kMixSums];
    __shared__ float coef[InkStates::kMax * kQueries];
    const size_t row = size_t(blockIdx.x) * n_embd;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float x[kMixHeld][NS];
    float sum[NS][1 + NQ];
#pragma unroll
    for (int s = 0; s < NS; ++s) {
#pragma unroll
        for (int k = 0; k <= NQ; ++k) sum[s][k] = 0.0f;
    }
#pragma unroll
    for (int h = 0; h < kMixHeld; ++h) {
        const int i = tid + h * kMixThreads;
        if (i < n_embd) {
            float q[NQ];
#pragma unroll
            for (int j = 0; j < NQ; ++j) q[j] = a.query[size_t(j) * n_embd + i];
#pragma unroll
            for (int s = 0; s < NS; ++s) {
                const float v = a.s[s][row + i];
                x[h][s] = v;
                sum[s][0] += v * v;
#pragma unroll
                for (int j = 0; j < NQ; ++j) sum[s][1 + j] += v * q[j];
            }
        }
    }
#pragma unroll
    for (int s = 0; s < NS; ++s) {
#pragma unroll
        for (int k = 0; k <= NQ; ++k) {
            const float w = warp_sum(sum[s][k]);
            if (lane == 0) part[s * kMixSums + k][warp] = w;
        }
    }
    __syncthreads();
    if (tid < NS * kMixSums && tid % kMixSums <= NQ) {
        float t = 0.0f;
        for (int w = 0; w < kMixThreads / WARP; ++w) t += part[tid][w];
        whole[tid] = t;
    }
    __syncthreads();
    if (tid < NQ) {
        const int j = tid;
        const float sc = 1.0f / sqrtf((float) n_embd);
        float v[NS], inv[NS];
        float m = -INFINITY;
        for (int s = 0; s < NS; ++s) {
            inv[s] = 1.0f / sqrtf(whole[s * kMixSums] / (float) n_embd + 1e-6f);
            v[s] = whole[s * kMixSums + 1 + j] * inv[s] * sc;
            m = fmaxf(m, v[s]);
        }
        float total = 0.0f;
        for (int s = 0; s < NS; ++s) {
            v[s] = expf(v[s] - m);
            total += v[s];
        }
        for (int s = 0; s < NS; ++s) coef[s * kQueries + j] = v[s] / total * inv[s];
    }
    __syncthreads();
    float out0[kMixHeld];
#pragma unroll
    for (int h = 0; h < kMixHeld; ++h) {
        const int i = tid + h * kMixThreads;
        if (i < n_embd) {
#pragma unroll
            for (int j = 0; j < NQ; ++j) {
                float acc = 0.0f;
#pragma unroll
                for (int s = 0; s < NS; ++s) acc += coef[s * kQueries + j] * x[h][s];
                a.out[j][row + i] = acc;
                if (j == 0) out0[h] = acc;
            }
        }
    }
    if (!a.norm_out) return;
    __shared__ float npart[kMixThreads / WARP];
    if (a.ffn) {   // add_rms_q8()'s arithmetic on the values written
        float ss = 0.0f;
#pragma unroll
        for (int h = 0; h < kMixHeld; ++h)
            if (tid + h * kMixThreads < n_embd) ss = __fmaf_rn(out0[h], out0[h], ss);
        ss = warp_sum(ss);
        if (lane == 0) npart[warp] = ss;
        __syncthreads();
        float total = 0.0f;
        for (int k = 0; k < kMixThreads / WARP; ++k) total = __fadd_rn(total, npart[k]);
        const float scale = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, float(n_embd)), a.eps)));
        QB* q = a.q8 + size_t(blockIdx.x) * (n_embd / 32);
#pragma unroll
        for (int h = 0; h < kMixHeld; ++h) {
            const int i = tid + h * kMixThreads;
            if (i < n_embd) {
                const float xi = __fmul_rn(__fmul_rn(out0[h], scale), a.norm_w[i]);
                a.norm_out[row + i] = xi;
                q8_put(q, i, xi);
            }
        }
        return;
    }
    // rms_norm_kernel's arithmetic, a thread over the values it wrote
    float ss = 0.0f;
#pragma unroll
    for (int h = 0; h < kMixHeld; ++h)
        if (tid + h * kMixThreads < n_embd) ss += out0[h] * out0[h];
    ss = block_sum(ss, npart);
    const float scale = 1.0f / sqrtf(ss / float(n_embd) + a.eps);
#pragma unroll
    for (int h = 0; h < kMixHeld; ++h) {
        const int i = tid + h * kMixThreads;
        if (i < n_embd) {
            const float v = out0[h];
            const float y = a.norm_w ? v * scale * a.norm_w[i] : v * scale;
            a.norm_out[row + i] = y;
            if (a.q8) q8_put(a.q8 + size_t(blockIdx.x) * (n_embd / 32), i, y);
        }
    }
}

// mix_held() where it takes the row, else mix_row()
template <int NS, int NQ>
__device__ __forceinline__ void mix_any(const MixArgs& a, int n_embd) {
    if (n_embd <= kMixHeld * kMixThreads) mix_held<NS, NQ>(a, n_embd);
    else mix_row(a, n_embd);
}

template <int NS, int NQ>
__global__ void __launch_bounds__(kMixThreads) ink_mix_kernel(MixArgs a, int n_embd) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    mix_any<NS, NQ>(a, n_embd);
}

constexpr int kMixJobs = 12;
struct MixMany {
    MixArgs j[kMixJobs];
};

// blockIdx.y the job
template <int NS, int NQ>
__global__ void __launch_bounds__(kMixThreads) ink_mix_many_kernel(MixMany m, int n_embd) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    mix_any<NS, NQ>(m.j[blockIdx.y], n_embd);
}

// kernel<NS, NQ> for a call's n states and nq queries
template <template <int, int> class K, class... Args>
void launch_mix(int n, int nq, dim3 grid, cudaStream_t s, Args... args) {
    const auto go = [&](auto ns_c) {
        constexpr int NS = decltype(ns_c)::value;
        switch (nq) {
            case 1: K<NS, 1>::launch(grid, s, args...); break;
            case 2: K<NS, 2>::launch(grid, s, args...); break;
            case 3: K<NS, 3>::launch(grid, s, args...); break;
            default: K<NS, 4>::launch(grid, s, args...); break;
        }
    };
    switch (n) {
        case 1: go(std::integral_constant<int, 1>{}); break;
        case 2: go(std::integral_constant<int, 2>{}); break;
        case 3: go(std::integral_constant<int, 3>{}); break;
        case 4: go(std::integral_constant<int, 4>{}); break;
        case 5: go(std::integral_constant<int, 5>{}); break;
        default: go(std::integral_constant<int, 6>{}); break;
    }
}

template <int NS, int NQ>
struct MixOne {
    static void launch(dim3 grid, cudaStream_t s, const MixArgs& a, int n_embd) {
        launch_pdl(ink_mix_kernel<NS, NQ>, dim3(grid), dim3(kMixThreads), 0, s, a, n_embd);
    }
};

template <int NS, int NQ>
struct MixJobs {
    static void launch(dim3 grid, cudaStream_t s, const MixMany& m, int n_embd) {
        launch_pdl(ink_mix_many_kernel<NS, NQ>, dim3(grid), dim3(kMixThreads), 0, s, m, n_embd);
    }
};

constexpr int kSet = 3;   // the controllers one launch takes

struct CatNormSet {
    const float* cur[kSet];
    const float* msg[kSet];
    float* out[kSet];
};

constexpr int kCatThreads = 256;   // ink_cat_norm_kernel's block

// Thread tid's shares of the squares of a row of cur and of msg, in ink_cat_norm_kernel's block: every
// kCatThreads-th value from tid.
__device__ __forceinline__ void cat_squares(const float* cur, const float* msg, int n_embd,
                                            int tid, float& a, float& b) {
    a = 0.0f, b = 0.0f;
    for (int i = tid; i < n_embd; i += kCatThreads) {
        a += cur[i] * cur[i];
        b += msg[i] * msg[i];
    }
}

__device__ __forceinline__ float cat_scale(float sum, int n_embd) { return 1.0f / sqrtf(sum / (float) n_embd + 1e-6f); }

// A token's pair a block, blockIdx.y the pair.
__global__ void ink_cat_norm_kernel(CatNormSet s, int n_embd) {
    __shared__ float part[kCatThreads / WARP];
    const float* __restrict__ cur = s.cur[blockIdx.y];
    const float* __restrict__ msg = s.msg[blockIdx.y];
    float* __restrict__ out = s.out[blockIdx.y];
    const size_t row = size_t(blockIdx.x) * n_embd;
    const int tid = threadIdx.x;
    float a, b;
    cat_squares(cur + row, msg + row, n_embd, tid, a, b);
    const float ia = cat_scale(block_sum(a, part), n_embd);
    const float ib = cat_scale(block_sum(b, part), n_embd);
    float* o = out + 2 * row;
    for (int i = tid; i < n_embd; i += blockDim.x) {
        o[i] = cur[row + i] * ia;
        o[n_embd + i] = msg[row + i] * ib;
    }
}

struct CtrlDown {
    const float* cur[kSet];
    const float* msg[kSet];
    const void* w[kSet];   // [2 * n_embd] an output, F32 or BF16
    float* y[kSet];
};

constexpr int kDownThreads = 512;   // ink_ctrl_down_kernel's block: half of gemv_set_kernel's

// ink_cat_norm_kernel and then gemv_set_kernel's product with the controller's down matrix in one launch: output o
// of controller j (blockIdx.x = j * rank + o) of token blockIdx.y.  Its first kCatThreads threads make the two norms
// as ink_cat_norm_kernel's block does, and the product takes each normed value as that kernel writes it.  The
// product is gemv_set_kernel's block of 1024 threads (2 * n_embd values) in kDownThreads: thread t has the sums of
// its threads t and t + 512, warps w and w + 16, and the 32 warps' sums are added in their order.  The same bits,
// and with half the threads a block, a launch's 3 * 64 blocks are one wave on the card.
template <typename W>
__global__ void __launch_bounds__(kDownThreads) ink_ctrl_down_kernel(CtrlDown d, int n_embd, int rank) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    __shared__ float pa[kCatThreads / WARP], pb[kCatThreads / WARP], part[32];
    const int j = blockIdx.x / rank, o = blockIdx.x % rank, tid = threadIdx.x;
    const size_t row = size_t(blockIdx.y) * n_embd;
    const float* cur = d.cur[j] + row;
    const float* msg = d.msg[j] + row;
    if (tid < kCatThreads) {
        float a, b;
        cat_squares(cur, msg, n_embd, tid, a, b);
        a = warp_sum(a);
        b = warp_sum(b);
        if ((tid & 31) == 0) pa[tid >> 5] = a, pb[tid >> 5] = b;
    }
    __syncthreads();
    float sa = 0.0f, sb = 0.0f;
    for (int k = 0; k < kCatThreads / WARP; ++k) sa += pa[k];
    for (int k = 0; k < kCatThreads / WARP; ++k) sb += pb[k];
    const float ia = cat_scale(sa, n_embd), ib = cat_scale(sb, n_embd);
    const W* __restrict__ wr = static_cast<const W*>(d.w[j]) + size_t(o) * 2 * n_embd;
    const int n_in = 2 * n_embd;
    float acc0 = 0.0f, acc1 = 0.0f;   // gemv_set_kernel's threads tid and tid + kDownThreads
    for (int i = tid; i < n_in; i += 2 * kDownThreads) {
        const float x0 = i < n_embd ? cur[i] * ia : msg[i - n_embd] * ib;
        acc0 += wval(wr, i) * x0;
        const int i1 = i + kDownThreads;
        if (i1 < n_in) {
            const float x1 = i1 < n_embd ? cur[i1] * ia : msg[i1 - n_embd] * ib;
            acc1 += wval(wr, i1) * x1;
        }
    }
    acc0 = warp_sum(acc0);
    acc1 = warp_sum(acc1);
    if ((tid & 31) == 0) part[tid >> 5] = acc0, part[(tid >> 5) + kDownThreads / WARP] = acc1;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int k = 0; k < 2 * kDownThreads / WARP; ++k) sum += part[k];
        d.y[j][size_t(blockIdx.y) * rank + o] = sum;
    }
}

// STRATA_INK_FAST_CTRL=1 (opt-in, other bits): ink_ctrl_down_kernel in one pass over cur and msg.  An output's block
// of kFastDownThreads takes, in one read of the two rows, the down row's products with cur and with msg and the two
// rows' squares, and scales the products by the norms after the sums: down . [cur * ia, msg * ib] as
// (down_l . cur) * ia + (down_r . msg) * ib.  The same values up to rounding (where the norm's scale is applied, and
// the sums' order); half the threads of a block, and cur and msg are read once a block instead of twice.  From our
// earlier Inkling engine's ctrl_down_kernel (PR 200), written anew on this kernel's arguments.
constexpr int kFastDownThreads = 256;
template <typename W>
__global__ void __launch_bounds__(kFastDownThreads) ink_ctrl_down_fast_kernel(CtrlDown d, int n_embd, int rank) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    __shared__ float part[kFastDownThreads / WARP][4];
    const int j = blockIdx.x / rank, o = blockIdx.x % rank, tid = threadIdx.x;
    const size_t row = size_t(blockIdx.y) * n_embd;
    const float* __restrict__ cur = d.cur[j] + row;
    const float* __restrict__ msg = d.msg[j] + row;
    const W* __restrict__ wl = static_cast<const W*>(d.w[j]) + size_t(o) * 2 * n_embd;
    const W* __restrict__ wr = wl + n_embd;
    float dl = 0.0f, dr = 0.0f, cc = 0.0f, mm = 0.0f;
    for (int i = tid; i < n_embd; i += kFastDownThreads) {
        const float c = cur[i], m = msg[i];
        dl += wval(wl, i) * c;
        dr += wval(wr, i) * m;
        cc += c * c;
        mm += m * m;
    }
    dl = warp_sum(dl), dr = warp_sum(dr), cc = warp_sum(cc), mm = warp_sum(mm);
    if ((tid & 31) == 0) part[tid >> 5][0] = dl, part[tid >> 5][1] = dr, part[tid >> 5][2] = cc, part[tid >> 5][3] = mm;
    __syncthreads();
    if (tid == 0) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int w = 0; w < kFastDownThreads / WARP; ++w)
#pragma unroll
            for (int k = 0; k < 4; ++k) v[k] += part[w][k];
        d.y[j][size_t(blockIdx.y) * rank + o] = v[0] * cat_scale(v[2], n_embd) + v[1] * cat_scale(v[3], n_embd);
    }
}

int g_fast_ctrl = -1;   // -1: STRATA_INK_FAST_CTRL decides (ink_set_fast_ctrl() for the kernel test)
bool fast_ctrl_on() {
#if defined(__HIPCC__)
    return false;   // CUDA only: not measured on AMD cards yet
#else
    if (g_fast_ctrl < 0) {
        const char* v = std::getenv("STRATA_INK_FAST_CTRL");
        g_fast_ctrl = v != nullptr && v[0] == '1';
    }
    return g_fast_ctrl == 1;
#endif
}

struct GateSet {
    const float* cur[kSet];
    const float* msg[kSet];
    const float* gain[kSet];
    const float* forget[kSet];
    float* out[kSet];
    int count;
};

// A value a thread, of every controller of the set: all its cur are read before an out is written.
__global__ void ink_gate_kernel(GateSet s, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float c[kSet];
#pragma unroll
    for (int j = 0; j < kSet; ++j)
        if (j < s.count) c[j] = s.cur[j][i];
#pragma unroll
    for (int j = 0; j < kSet; ++j)
        if (j < s.count) s.out[j][i] = c[j] - tanhf(s.forget[j][i]) * c[j] + tanhf(s.gain[j][i]) * s.msg[j][i];
}

// Token t's convolved value of channel c (ink_conv()).
__device__ __forceinline__ float conv_value(const float* x, const float* state, const float* __restrict__ w,
                                            const float* dyn, int channels,
                                            int heads, int taps, int t, int c, float norm) {
    float past[8];   // past[k]: the input k tokens back
    for (int k = 0; k < taps; ++k) {
        const int u = t - k;
        past[k] = u >= 0 ? x[size_t(u) * channels + c] : state[size_t(taps - 1 + u) * channels + c];
    }
    float acc = 0.0f;
    for (int j = 0; j < taps; ++j) acc += past[taps - 1 - j] * w[size_t(c) * taps + j];
    if (dyn) {
        const int head = c / (channels / heads);
        const float* d = dyn + (size_t(t) * heads + head) * taps;
        for (int lag = 0; lag < taps; ++lag) acc += past[lag] * d[lag];
    }
    return (past[0] + acc) * norm;
}

// Channel c's state after a call of `rows` tokens: the last taps - 1 inputs, oldest first.
__device__ __forceinline__ void conv_state_at(const float* x, float* state, int channels, int taps,
                                              int rows, int c) {
    const int keep = taps - 1;
    for (int k = 0; k < keep; ++k) {
        const int u = rows - keep + k;
        state[size_t(k) * channels + c] =
            u >= 0 ? x[size_t(u) * channels + c] : state[size_t(keep + u) * channels + c];
    }
}

// Channel c's state after the first `rows` tokens of a call (conv_state_at()'s), from the state before the call into
// out, which may be elsewhere.
__device__ __forceinline__ void conv_state_into(const float* x, const float* state, float* out, int channels, int taps,
                                                int rows, int c) {
    const int keep = taps - 1;
    for (int k = 0; k < keep; ++k) {
        const int u = rows - keep + k;
        out[size_t(k) * channels + c] = u >= 0 ? x[size_t(u) * channels + c] : state[size_t(keep + u) * channels + c];
    }
}

// One head's clip (ink_clip()), the whole block taking part; p is the token's position.
__device__ __forceinline__ void clip_head(float* v, const float* __restrict__ sc, float thr, int dim, float alpha,
                                          float floor_, long long p, float* part) {
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        const float y = sc ? v[i] * sc[i] : v[i];
        ss += y * y;
    }
    const float tot = block_sum(ss, part);
    float f = fminf(thr / (sqrtf(tot) + 1e-6f), 1.0f);
    if (alpha > 0.0f) f *= 1.0f + alpha * logf(fmaxf(1.0f, (float) (p + 1) / floor_));
    for (int i = tid; i < dim; i += blockDim.x) v[i] = (sc ? v[i] * sc[i] : v[i]) * f;
}

// One head's RMS (ink_head_rms()), the whole block taking part.
__device__ __forceinline__ void rms_head(float* v, int dim, float eps, float* part) {
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) ss += v[i] * v[i];
    const float f = 1.0f / sqrtf(block_sum(ss, part) / (float) dim + eps);
    for (int i = tid; i < dim; i += blockDim.x) v[i] *= f;
}

__global__ void ink_conv_kernel(const float* __restrict__ x, const float* __restrict__ state,
                                const float* __restrict__ w, const float* __restrict__ dyn, float* __restrict__ out,
                                int channels, int heads, int taps, int rows, float norm) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size_t(rows) * channels) return;
    const int t = (int) (idx / channels), c = (int) (idx % channels);
    out[idx] = conv_value(x, state, w, dyn, channels, heads, taps, t, c, norm);
}

// A channel a thread: the state after each token of the call that bits names (bit t: after token t), the j-th such
// state at snap + j * stride.  Before ink_conv_state_kernel, which overwrites the state.
__global__ void ink_conv_snaps_kernel(const float* x, const float* state, float* snap, size_t stride, unsigned bits,
                                      int channels, int taps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    int j = 0;
    for (unsigned b = bits; b; b &= b - 1, ++j)
        conv_state_into(x, state, snap + size_t(j) * stride, channels, taps, __ffs(b), c);
}

// A channel a thread, a call of a few tokens, job blockIdx.y: what ink_conv_kernel, ink_conv_snaps_kernel and
// ink_conv_state_kernel do in turn, then the stream's update ink_conv_add_kernel makes, the same bits in one launch
__global__ void ink_conv_few_kernel(InkConv j0, InkConv j1, size_t stride, unsigned bits, int channels, int heads,
                                    int taps, int rows) {
    const InkConv& j = blockIdx.y ? j1 : j0;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    for (int t = 0; t < rows; ++t) {
        const size_t i = size_t(t) * channels + c;
        const float y = conv_value(j.x, j.state, j.w, j.dyn, channels, heads, taps, t, c, j.norm);
        j.out[i] = y;
        if (!j.acc) continue;
        if (j.mode == 0) {
            j.acc[i] = j.acc[i] + y * j.a;
        } else {
            if (j.mode == 1) j.keep[i] = j.acc[i];
            j.acc[i] = y;
        }
    }
    int k = 0;
    if (j.snap)
        for (unsigned b = bits; b; b &= b - 1, ++k)
            conv_state_into(j.x, j.state, j.snap + size_t(k) * stride, channels, taps, __ffs(b), c);
    conv_state_at(j.x, j.state, channels, taps, rows, c);
}

// a channel a thread
__global__ void ink_conv_state_kernel(const float* __restrict__ x, float* state, int channels, int taps, int rows) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    conv_state_at(x, state, channels, taps, rows, c);
}

// a token's head a block
__global__ void ink_clip_kernel(float* x, const float* __restrict__ scale, float thr, int dim, int heads,
                                long long pos, float alpha, float floor_, const long long* __restrict__ at) {
    __shared__ float part[128 / WARP];
    const int row = blockIdx.x / heads, head = blockIdx.x % heads;
    clip_head(x + (size_t(row) * heads + head) * dim, scale ? scale + size_t(head) * dim : nullptr, thr, dim, alpha,
              floor_, at ? at[2 * row] : pos + row, part);
}

// ink_clip() and ink_head_rms() of InkClips' four jobs in one launch, a block a token's head as each of them
template <int KV>
__global__ void ink_clips_kernel(InkClips c) {
    __shared__ float part[128 / WARP];
    int b = blockIdx.x, j = 0;
    while (j < 3 && b >= c.rows * c.heads[j]) b -= c.rows * c.heads[j++];
    const int row = b / c.heads[j], head = b % c.heads[j];
    float* v = c.x[j] + (size_t(row) * c.heads[j] + head) * c.dim[j];
    if (c.rms[j]) rms_head(v, c.dim[j], c.eps, part);
    else
        clip_head(v, c.scale[j] ? c.scale[j] + size_t(head) * c.dim[j] : nullptr, c.thr[j], c.dim[j], c.alpha[j],
                  c.floor[j], c.at ? c.at[2 * row] : c.pos + row, part);
    if (!c.cache[j]) return;   // each thread its own values, as clip_head() and rms_head() left them
    long long cell = (c.at ? c.at[0] : c.pos) + row;
    if (c.ring > 0) cell %= c.ring;
    if (KV != kKvF16) {
        kv_store<KV>(v, c.cache[j], cell, c.heads[j], c.dim[j], head, part);
        return;
    }
    __half* to = (__half*) c.cache[j] + (size_t) cell * c.heads[j] * c.dim[j] + (size_t) head * c.dim[j];
    for (int i = threadIdx.x; i < c.dim[j]; i += blockDim.x) to[i] = __float2half(v[i]);
}

__global__ void ink_head_rms_kernel(float* x, int dim, int heads, float eps) {
    __shared__ float part[128 / WARP];
    rms_head(x + size_t(blockIdx.x) * dim, dim, eps, part);
}

// ink_conv_add()'s channel c of one token
__device__ __forceinline__ void conv_add_at(const float* x, float* state, const float* __restrict__ w,
                                            float* __restrict__ out, float* acc, float* keep, float a, int mode,
                                            int channels, int taps, float norm, int c) {
    const float y = conv_value(x, state, w, nullptr, channels, 1, taps, 0, c, norm);
    out[c] = y;
    conv_state_at(x, state, channels, taps, 1, c);
    if (mode == 0) {
        acc[c] = acc[c] + y * a;
    } else {
        if (mode == 1) keep[c] = acc[c];
        acc[c] = y;
    }
}

__global__ void ink_conv_add_kernel(const float* x, float* state, const float* __restrict__ w,
                                    float* __restrict__ out, float* acc, float* keep, float a, int mode, int channels,
                                    int taps, float norm) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    conv_add_at(x, state, w, out, acc, keep, a, mode, channels, taps, norm, c);
}

// ink_conv_add_rows(): token blockIdx.y's channels, a row of them on, with its own state
__global__ void ink_conv_add_rows_kernel(const float* x, InkStateRows state, const float* __restrict__ w,
                                         float* __restrict__ out, float* acc, float* keep, float a, int mode,
                                         int channels, int taps, float norm) {
    pdl_trigger();
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const size_t t = size_t(blockIdx.y) * channels;
    conv_add_at(x + t, state.p[blockIdx.y], w, out + t, acc + t, keep ? keep + t : nullptr, a, mode, channels, taps,
                norm, c);
}

__global__ void ink_moe_conv_add_kernel(const float* x, const float* w, const float* __restrict__ scale,
                                        const int* ids, int n_used,
                                        float* fo, float* state, const float* __restrict__ cw, float* __restrict__ out,
                                        float* acc, float a, int channels, int taps, float norm) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    float s = 0.0f;
    for (int k = 0; k < n_used; ++k) {
        const float wk = scale ? w[k] * scale[ids[k]] : w[k];   // expert_weights_kernel's product
        s += wk * x[size_t(k) * channels + i];
    }
    fo[i] = 0.0f + s;   // moe_sum() onto the zeros fo was
    const float y = conv_value(fo, state, cw, nullptr, channels, 1, taps, 0, i, norm);
    out[i] = y;
    conv_state_at(fo, state, channels, taps, 1, i);
    acc[i] = acc[i] + y * a;
}

struct QkvArgs {
    InkQkv a;
    float k_norm, v_norm;
};

// ink_qkv(): blocks [0, n_head) Q's heads, then K's n_kv, V's n_kv, and the relative positions' n_head; 128 threads.
template <int KV>
__device__ __forceinline__ void qkv_block(const InkQkv& a, float k_norm, float v_norm, float* part) {
    const long long p = a.at ? a.at[0] : (long long) a.pos;
    int b = blockIdx.x;
    if (b < a.n_head) {
        clip_head(a.q + size_t(b) * a.dim, a.q_scale ? a.q_scale + size_t(b) * a.dim : nullptr, a.qk_thr, a.dim,
                  a.alpha, a.floor, p, part);
        return;
    }
    b -= a.n_head;
    if (b < 2 * a.n_kv) {
        const bool is_v = b >= a.n_kv;
        const int h = is_v ? b - a.n_kv : b, channels = a.n_kv * a.dim, i = threadIdx.x;
        const float* x = is_v ? a.vraw : a.kraw;
        float* state = is_v ? a.vstate : a.kstate;
        float* out = is_v ? a.v : a.k;
        if (i < a.dim) {
            const int c = h * a.dim + i;
            out[c] = conv_value(x, state, is_v ? a.vw : a.kw, is_v ? a.vdyn : a.kdyn, channels, a.n_kv, a.taps, 0, c,
                                is_v ? v_norm : k_norm);
            conv_state_at(x, state, channels, a.taps, 1, c);
        }
        float* head = out + size_t(h) * a.dim;
        if (is_v) rms_head(head, a.dim, a.eps, part);
        else clip_head(head, nullptr, a.qk_thr, a.dim, 0.0f, 1.0f, p, part);
        const long long row = a.ring > 0 ? p % a.ring : p;
        if (!(is_v ? a.vals : a.keys)) return;   // a full layer under the ZHA index keeps its keys itself
        if (KV != kKvF16) kv_store<KV>(head, is_v ? a.vals : a.keys, row, a.n_kv, a.dim, h, part);
        else if (i < a.dim)
            ((__half*) (is_v ? a.vals : a.keys))[size_t(row) * channels + size_t(h) * a.dim + i] = __float2half(head[i]);
        return;
    }
    b -= 2 * a.n_kv;
    clip_head(a.r + size_t(b) * a.n_rel, a.r_scale ? a.r_scale + size_t(b) * a.n_rel : nullptr, a.rel_thr, a.n_rel,
              a.alpha, a.floor, p, part);
}

template <int KV>
__global__ void ink_qkv_kernel(QkvArgs g) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    __shared__ float part[128 / WARP];
    qkv_block<KV>(g.a, g.k_norm, g.v_norm, part);
}

// ink_qkv_rows(): token blockIdx.y's blocks, its rows of the products and outputs, its states, caches and position
template <int KV>
__global__ void ink_qkv_rows_kernel(QkvArgs g, InkQkvRows r) {
    pdl_trigger();
    pdl_wait();
    __shared__ float part[128 / WARP];
    const int t = blockIdx.y;
    InkQkv a = g.a;
    const size_t kv = size_t(a.n_kv) * a.dim, dyn = size_t(a.taps) * a.n_kv;
    a.q += size_t(t) * a.n_head * a.dim;
    a.r += size_t(t) * a.n_head * a.n_rel;
    a.kraw += t * kv;
    a.vraw += t * kv;
    if (a.kdyn) a.kdyn += t * dyn;
    if (a.vdyn) a.vdyn += t * dyn;
    a.k += t * kv;
    a.v += t * kv;
    a.kstate = r.kstate[t];
    a.vstate = r.vstate[t];
    a.keys = r.keys[t];
    a.vals = r.vals[t];
    a.at += 2 * t;
    qkv_block<KV>(a, g.k_norm, g.v_norm, part);
}

__global__ void ink_mul_sigmoid_kernel(float* x, const float* __restrict__ gate, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

__global__ void ink_store_ring_kernel(const float* __restrict__ x, __half* __restrict__ cache, int n, long long pos,
                                      int rows, int ring) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size_t(rows) * n) return;
    const long long t = (long long) (idx / n);
    cache[size_t((pos + t) % ring) * n + idx % n] = __float2half(x[idx]);
}

__global__ void ink_softcap_kernel(float* logits, int n_keep, int n_all, int rows, float cap) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size_t(rows) * n_all) return;
    logits[idx] = (int) (idx % n_all) < n_keep ? cap * tanhf(logits[idx] / cap) : -INFINITY;
}

constexpr int kAttnThreads = 1024;   // a block of the softmax kernel
constexpr int kMaxPer = 4;           // the most query heads on one key head
constexpr int kLaneHalves = 4;       // the most halves of a cache row's head a lane holds: heads of up to 128 values
constexpr int kScoreWarps = 8;       // a block of the score kernel
constexpr int kScoreKeys = 4;        // the keys a warp of it takes, one after the other
constexpr int kValueKeys = 8;        // the same for the value kernel: a lane has a weight of the warp's each
constexpr size_t kValueShared = size_t(48) << 10;   // the shared memory a block has without asking for more
static_assert(kMaxPer * kValueKeys <= 32, "a warp's weights are a lane each");

// A lane's share of a head's row of the cache, which the warp reads side by side: n halves (2 or 4) of the row's
// 32 * n, as floats.  The rest of out is 0.
// lane_halves() of an FP8 cache: row `row`'s head g, its codes times its scale
__device__ __forceinline__ void lane_fp8(const void* cache, long long row, int n_kv, int dim, int g, int lane, int n,
                                         float* out) {
    const uint8_t* r = (const uint8_t*) cache + (size_t) row * kv8_row(n_kv, dim);
    const float sc = reinterpret_cast<const float*>(r + (size_t) n_kv * dim)[g];
    const uint8_t* p = r + (size_t) g * dim + lane * n;
#pragma unroll
    for (int c = 0; c < kLaneHalves; ++c) out[c] = c < n ? e4m3_val(p[c]) * sc : 0.0f;
}

// lane_halves() of a Q8_0 cache: row `row`'s head g, its codes times their block's scale (a lane's n values lie in
// one block)
__device__ __forceinline__ void lane_q8(const void* cache, long long row, int n_kv, int dim, int g, int lane, int n,
                                        float* out) {
    const uint8_t* r = (const uint8_t*) cache + (size_t) row * kvq8_row(n_kv, dim);
    const size_t at = (size_t) g * dim + lane * n;
    const float d = __half2float(__ushort_as_half(kvq8_scales(r, n_kv, dim)[at >> 5]));
    const int8_t* p = reinterpret_cast<const int8_t*>(r) + at;
#pragma unroll
    for (int c = 0; c < kLaneHalves; ++c) out[c] = c < n ? (float) p[c] * d : 0.0f;
}

// A lane's n (2 or 4) codes of row `row`'s head g in a Q8_0 cache as one word for __dp4a (the rest of it 0), and
// their block's scale
__device__ __forceinline__ int lane_q8_codes(const void* cache, long long row, int n_kv, int dim, int g, int lane,
                                             int n, float* d) {
    const uint8_t* r = (const uint8_t*) cache + (size_t) row * kvq8_row(n_kv, dim);
    const size_t at = (size_t) g * dim + lane * n;
    *d = __half2float(__ushort_as_half(kvq8_scales(r, n_kv, dim)[at >> 5]));
    return n == 4 ? *reinterpret_cast<const int*>(r + at) : (int) *reinterpret_cast<const uint16_t*>(r + at);
}

__device__ __forceinline__ void lane_halves(const __half* head, int lane, int n, float* out) {
    const __half2* p = (const __half2*) (head + lane * n);
#pragma unroll
    for (int c = 0; c < kLaneHalves / 2; ++c) {
        float2 f = make_float2(0.0f, 0.0f);
        if (2 * c < n) f = __half22float2(p[c]);
        out[2 * c] = f.x;
        out[2 * c + 1] = f.y;
    }
}

// The scores of a token's heads on one key head (blockIdx.x) against a block of keys (blockIdx.y).  A warp has
// kScoreKeys keys, one at a time: its lanes read the key side by side, once for all of the group's heads, and
// their products with the queries are summed across the warp.  The bias's products are the first n_rel lanes'.
// t0 is the launch's first token among the call's, nk_max the scores a head has room for.  ink_attend_rows()'s path
// (ink_score_rows_kernel); ink_attend() takes ink_score_kernel below, the same bits a key in fewer instructions.
template <int KV>
__device__ __forceinline__ void score_block(const InkAttn& a, const void* keys, int t0, int nk_max) {
    __shared__ float qs[kMaxPer * 128];
    __shared__ float rs[kMaxPer * 32];
    __shared__ float sinks[kMaxPer];
    const int per = a.n_head / a.n_kv, lh = a.dim / 32;
    const int tt = blockIdx.x / a.n_kv, g = blockIdx.x % a.n_kv, t = t0 + tt;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long long p = a.at ? a.at[2 * t] : a.pos + t;
    const long long lo = a.window > 0 ? (p - a.window + 1 > 0 ? p - a.window + 1 : 0) : 0;
    const int nk = (int) (p - lo + 1);
    if ((int) (blockIdx.y * kScoreWarps * kScoreKeys) >= nk) return;   // (the whole block)
    const size_t head0 = size_t(t) * a.n_head + size_t(g) * per;       // the group's heads lie together
    for (int i = tid; i < per * a.dim; i += blockDim.x) qs[i] = a.q[head0 * a.dim + i] * a.scale;
    for (int i = tid; i < per * a.n_rel; i += blockDim.x) rs[i] = a.r[head0 * a.n_rel + i];
    if (tid < per) sinks[tid] = a.sink[head0 + tid];
    __syncthreads();
    const int j0 = (blockIdx.y * kScoreWarps + warp) * kScoreKeys, j1 = j0 + kScoreKeys < nk ? j0 + kScoreKeys : nk;
    if (j0 >= nk) return;   // (a whole warp)
    float q[kMaxPer][kLaneHalves], r[kMaxPer];
#pragma unroll
    for (int hp = 0; hp < kMaxPer; ++hp) {
#pragma unroll
        for (int c = 0; c < kLaneHalves; ++c) q[hp][c] = hp < per && c < lh ? qs[hp * a.dim + lane * lh + c] : 0.0f;
        r[hp] = hp < per && lane < a.n_rel ? rs[hp * a.n_rel + lane] : 0.0f;
    }
    float* s = a.score + (size_t(tt) * a.n_head + size_t(g) * per) * nk_max;
    // the warp's keys and their bias values first, so that their reads are on their way together (the scores'
    // stores between them could be to the same memory, as far as the compiler knows)
    float k[kScoreKeys][kLaneHalves], rb[kScoreKeys];
    long long row = a.ring > 0 ? (lo + j0) % a.ring : lo + j0;
#pragma unroll
    for (int u = 0; u < kScoreKeys; ++u) {
        if (j0 + u < j1) {
            if (KV == kKvFp8) lane_fp8(keys, row, a.n_kv, a.dim, g, lane, lh, k[u]);
            else if (KV == kKvQ8) lane_q8(keys, row, a.n_kv, a.dim, g, lane, lh, k[u]);
            else lane_halves((const __half*) keys + (size_t(row) * a.n_kv + g) * a.dim, lane, lh, k[u]);
            const long long dist = p - (lo + j0 + u);
            rb[u] = lane < a.n_rel && dist < a.extent ? a.rel_bias[size_t(dist) * a.n_rel + lane] : 0.0f;
            if (++row == a.ring) row = 0;
        }
    }
#pragma unroll
    for (int u = 0; u < kScoreKeys; ++u) {
        if (j0 + u < j1) {
#pragma unroll
            for (int hp = 0; hp < kMaxPer; ++hp) {
                if (hp < per) {
                    float d = rb[u] * r[hp];
#pragma unroll
                    for (int c = 0; c < kLaneHalves; ++c) d += q[hp][c] * k[u][c];
                    d = warp_sum(d);
                    if (lane == 0) s[size_t(hp) * nk_max + j0 + u] = d - sinks[hp];
                }
            }
        }
    }
}

// The scores of a token's heads on one key head (blockIdx.x) against a block of keys (blockIdx.y).  A key's score
// is the sum of 32 shares, share j the bias's product for relative value j (0 from n_rel on) plus the dot of the
// query's values LH * j .. LH * j + LH - 1 with the key's, and it is added up as a butterfly over 32 lanes adds:
// j with j + 16, then 8 apart, 4, 2, 1.  A lane takes the four shares of one key that sum first (j, j + 8, j + 16,
// j + 24, for j its lane in the key's 8), so a warp makes four keys' scores at once with three shuffles a head, and
// takes `steps` such steps (the block's warps in turn).  Every product and sum is the one a lane of a 32-lane
// warp a key made before, in its order (pinned with __fmul_rn / __fmaf_rn as NVVM contracted that code), so the
// scores are the same bits.  A step whose keys are all past the bias's extent (most of a long context) reads no
// bias weights: theirs are 0, so is their product.  t0 is the launch's first token among the call's, nk_max the
// scores a head has room for.
constexpr int kScoreLanes = 32 / kScoreKeys;     // a key's lanes
constexpr int kScoreShares = 32 / kScoreLanes;   // a lane's shares of a score
constexpr int kScoreSteps = 8;                   // the most steps of kScoreKeys keys a warp takes
static_assert(kScoreShares == 4, "a lane adds the butterfly's first two rounds itself");

// ink_score_kernel()'s steps for a full layer's token of kInt8From keys or more with InkAttn::int8_dot
// (STRATA_KV_INT8), Q8_0 keys: each lane's LH values of a query head (already times the scale), at the shares it
// takes, become Q8_0 codes, a 32-value block's delta its largest |value| / 127 and a code the value times 1 / delta
// rounded to the nearest (about as llama.cpp's flash-attention kernels quantize their queries for a quantized K).  A
// share is then the bias's product plus one __dp4a of its codes with the key's times the two deltas; the shares and
// lanes are added as in the default steps.  Other arithmetic, so other bits.
template <int LH>
__device__ __forceinline__ void score_int8(const InkAttn& a, const float* qs, const float (&rr)[kMaxPer][kScoreShares],
                                           float* s, const float* sinks, int per, int g, long long p, long long lo,
                                           int nk, int nk_max, int j, long long row, int sub, int mine, int steps) {
    // a 32-value block is 32 / LH shares: in this lane's 8 for LH 4; for LH 2 also the lane's other share of it
    int qc[kMaxPer][kScoreShares];
    float qd[kMaxPer][kScoreShares];
#pragma unroll
    for (int hp = 0; hp < kMaxPer; ++hp) {
        float x[kScoreShares][LH], amax[kScoreShares];
#pragma unroll
        for (int k = 0; k < kScoreShares; ++k) {
            const int vl = sub + kScoreLanes * k;
            amax[k] = 0.0f;
#pragma unroll
            for (int c = 0; c < LH; ++c) {
                x[k][c] = hp < per ? qs[hp * a.dim + vl * LH + c] : 0.0f;
                // a NaN or infinite value makes its block's delta infinite and the block's codes 0, so the scores it
                // is in are NaN (0 x Inf): not finite, as the default steps' would not be either (fmaxf skips a NaN,
                // and a NaN's code is 0, which would hide it)
                const float ax = fabsf(x[k][c]);
                amax[k] = fmaxf(amax[k], isfinite(ax) ? ax : INFINITY);
            }
        }
#pragma unroll
        for (int k = 0; k < kScoreShares; ++k) {
            float m = LH == 2 ? fmaxf(amax[k], amax[k ^ 1]) : amax[k];
#pragma unroll
            for (int o = 1; o < kScoreLanes; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o, WARP));
            const float d = m / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
            int w = 0;
#pragma unroll
            for (int c = 0; c < LH; ++c) w |= (int(roundf(x[k][c] * id)) & 0xff) << (8 * c);
            qc[hp][k] = w;
            qd[hp][k] = d;
        }
    }
    for (int st = 0; st < steps; ++st) {   // the launch's steps (score_steps())
        const int jw = j - mine;   // the step's first key
        if (jw >= nk) break;       // (a whole warp)
        const bool here = j < nk;
        const long long dist = p - (lo + j);
        const bool far = p - (lo + jw + kScoreKeys - 1) >= a.extent;   // (the whole warp)
        int kc[kScoreShares];
        float kd[kScoreShares], rb[kScoreShares];
#pragma unroll
        for (int k = 0; k < kScoreShares; ++k) {
            const int vl = sub + kScoreLanes * k;
            kc[k] = 0, kd[k] = 0.0f;
            if (here) kc[k] = lane_q8_codes(a.keys, row, a.n_kv, a.dim, g, vl, LH, &kd[k]);
            rb[k] = !far && here && vl < a.n_rel && dist < a.extent ? a.rel_bias[size_t(dist) * a.n_rel + vl] : 0.0f;
        }
#pragma unroll
        for (int hp = 0; hp < kMaxPer; ++hp) {
            if (hp < per) {
                float sh[kScoreShares];
#pragma unroll
                for (int k = 0; k < kScoreShares; ++k)
                    sh[k] = rb[k] * rr[hp][k] + (float) __dp4a(qc[hp][k], kc[k], 0) * (qd[hp][k] * kd[k]);
                float v = (sh[0] + sh[2]) + (sh[1] + sh[3]);
#pragma unroll
                for (int o = kScoreLanes / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o, WARP);
                if (sub == 0 && here) s[size_t(hp) * nk_max + j] = v - sinks[hp];
            }
        }
        j += kScoreWarps * kScoreKeys;
        row += kScoreWarps * kScoreKeys;
        if (a.ring > 0)
            while (row >= a.ring) row -= a.ring;
    }
}

template <int KV, int LH>
__global__ void __launch_bounds__(kScoreWarps * 32) ink_score_kernel(InkAttn a, int t0, int nk_max, int steps) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    __shared__ __align__(16) float qs[kMaxPer * 128];
    __shared__ float rs[kMaxPer * 32];
    __shared__ float sinks[kMaxPer];
    const int per = a.n_head / a.n_kv;
    const int tt = blockIdx.x / a.n_kv, g = blockIdx.x % a.n_kv, t = t0 + tt;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int sub = lane % kScoreLanes, mine = lane / kScoreLanes;   // this lane's key of a step, and its 8th of it
    const long long p = a.at ? a.at[2 * t] : a.pos + t;
    const long long lo = a.window > 0 ? (p - a.window + 1 > 0 ? p - a.window + 1 : 0) : 0;
    const int nk = (int) (p - lo + 1);
    const int jb = blockIdx.y * kScoreWarps * kScoreKeys * steps;
    if (jb >= nk) return;   // (the whole block)
    const size_t head0 = size_t(t) * a.n_head + size_t(g) * per;   // the group's heads lie together
    for (int i = tid; i < per * a.dim; i += blockDim.x) qs[i] = a.q[head0 * a.dim + i] * a.scale;
    for (int i = tid; i < per * a.n_rel; i += blockDim.x) rs[i] = a.r[head0 * a.n_rel + i];
    if (tid < per) sinks[tid] = a.sink[head0 + tid];
    __syncthreads();
    // share j's relative-position weight of each head: 0 from n_rel on, as a lane past n_rel had
    float rr[kMaxPer][kScoreShares];
#pragma unroll
    for (int hp = 0; hp < kMaxPer; ++hp)
#pragma unroll
        for (int k = 0; k < kScoreShares; ++k) {
            const int vl = sub + kScoreLanes * k;
            rr[hp][k] = hp < per && vl < a.n_rel ? rs[hp * a.n_rel + vl] : 0.0f;
        }
    float* s = a.score + (size_t(tt) * a.n_head + size_t(g) * per) * nk_max;
    // the warp's first key and its cache row; a ring's row is advanced, not divided for, from step to step
    int j = jb + warp * kScoreKeys + mine;
    long long row = a.ring > 0 ? (lo + j) % a.ring : lo + j;
    if (KV == kKvQ8Dot && a.window == 0 && nk >= kInt8From) {   // full layers only (the whole block)
        score_int8<LH>(a, qs, rr, s, sinks, per, g, p, lo, nk, nk_max, j, row, sub, mine, steps);
        return;
    }
    for (int st = 0; st < steps; ++st) {
        const int jw = j - mine;   // the step's first key
        if (jw >= nk) break;       // (a whole warp)
        const bool here = j < nk;
        const long long dist = p - (lo + j);
        const bool far = p - (lo + jw + kScoreKeys - 1) >= a.extent;   // (the whole warp)
        // this lane's four shares of key j: its values and bias weights first, so that their reads are on their way
        float kx[kScoreShares][kLaneHalves], rb[kScoreShares];
#pragma unroll
        for (int k = 0; k < kScoreShares; ++k) {
            const int vl = sub + kScoreLanes * k;
            if (here) {
                if (KV == kKvFp8) lane_fp8(a.keys, row, a.n_kv, a.dim, g, vl, LH, kx[k]);
                else if (KV == kKvQ8 || KV == kKvQ8Dot) lane_q8(a.keys, row, a.n_kv, a.dim, g, vl, LH, kx[k]);
                else lane_halves((const __half*) a.keys + (size_t(row) * a.n_kv + g) * a.dim, vl, LH, kx[k]);
            } else {
#pragma unroll
                for (int c = 0; c < kLaneHalves; ++c) kx[k][c] = 0.0f;
            }
            rb[k] = !far && here && vl < a.n_rel && dist < a.extent ? a.rel_bias[size_t(dist) * a.n_rel + vl] : 0.0f;
        }
#pragma unroll
        for (int hp = 0; hp < kMaxPer; ++hp) {
            if (hp < per) {
                float sh[kScoreShares];
#pragma unroll
                for (int k = 0; k < kScoreShares; ++k) {
                    const int vl = sub + kScoreLanes * k;
                    float q[kLaneHalves];
                    if (LH == 4) {
                        const float4 v = *reinterpret_cast<const float4*>(&qs[hp * a.dim + vl * 4]);
                        q[0] = v.x, q[1] = v.y, q[2] = v.z, q[3] = v.w;
                    } else {
#pragma unroll
                        for (int c = 0; c < kLaneHalves; ++c) q[c] = c < LH ? qs[hp * a.dim + vl * LH + c] : 0.0f;
                    }
                    // the old lane's share as NVVM contracted it: q0 k0, the bias's product added by an fma, then
                    // the other values' products in their order
                    float d = __fmaf_rn(rb[k], rr[hp][k], __fmul_rn(q[0], kx[k][0]));
#pragma unroll
                    for (int c = 1; c < kLaneHalves; ++c) d = __fmaf_rn(q[c], kx[k][c], d);
                    sh[k] = d;
                }
                // shares j and j + 16, j + 8 and j + 24, then those two; then the key's other lanes
                float v = __fadd_rn(__fadd_rn(sh[0], sh[2]), __fadd_rn(sh[1], sh[3]));
#pragma unroll
                for (int o = kScoreLanes / 2; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffff, v, o, WARP));
                if (sub == 0 && here) s[size_t(hp) * nk_max + j] = v - sinks[hp];
            }
        }
        j += kScoreWarps * kScoreKeys;
        row += kScoreWarps * kScoreKeys;
        if (a.ring > 0)
            while (row >= a.ring) row -= a.ring;
    }
}

// ink_attend_rows(): the score kernel's blocks of token t0 + blockIdx.x / n_kv over that token's own keys
template <int KV>
__global__ void ink_score_rows_kernel(InkAttn a, int t0, int nk_max, InkAttnRows c) {
    pdl_trigger();
    pdl_wait();
    score_block<KV>(a, c.keys[t0 + blockIdx.x / a.n_kv], t0, nk_max);
}

// A token's head a block: the scores' softmax with its extra term of score 0.  The scores become the weights'
// numerators (exp against the largest), and norm gets what they are divided by, a token's head each.
__global__ void ink_softmax_kernel(InkAttn a, int t0, int nk_max, float* norm) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    __shared__ float part[kAttnThreads / WARP];
    const int tt = blockIdx.x / a.n_head, h = blockIdx.x % a.n_head, t = t0 + tt, tid = threadIdx.x;
    const long long p = a.at ? a.at[2 * t] : a.pos + t;
    const long long lo = a.window > 0 ? (p - a.window + 1 > 0 ? p - a.window + 1 : 0) : 0;
    const int nk = (int) (p - lo + 1);
    float* s = a.score + (size_t(tt) * a.n_head + h) * nk_max;
    float m = -INFINITY;
    for (int j = tid; j < nk; j += blockDim.x) m = fmaxf(m, s[j]);
    m = fmaxf(block_max(m, part), 0.0f);
    float sum = 0.0f;
    for (int j = tid; j < nk; j += blockDim.x) {
        s[j] = expf(s[j] - m);
        sum += s[j];
    }
    sum = block_sum(sum, part);
    if (tid == 0) norm[blockIdx.x] = sum + expf(-m);
}

// The values' sums under the weights for a token's heads on one key head (blockIdx.x), a block of keys
// (blockIdx.y), into parts: `dim` values a head and a block.  A warp has kValueKeys values, one at a time, read
// as the score kernel reads a key; a lane holds one of the warp's weights, a head and a key each.  The warps'
// sums are added in their order.  value_warps() launches up to 32 warps, so the registers must fit 1,024 threads on
// every card: sm_86 and sm_89 gave this kernel 66 a thread without the bound, and the launch failed there.
template <int KV>
__device__ __forceinline__ void value_block(const InkAttn& a, const void* vals, int t0, int nk_max, float* parts,
                                            int n_parts) {
    extern __shared__ float sums[];   // a warp's, a head of the group: dim
    const int per = a.n_head / a.n_kv, lh = a.dim / 32;
    const int tt = blockIdx.x / a.n_kv, g = blockIdx.x % a.n_kv, t = t0 + tt;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, warps = blockDim.x / 32;
    const long long p = a.at ? a.at[2 * t] : a.pos + t;
    const long long lo = a.window > 0 ? (p - a.window + 1 > 0 ? p - a.window + 1 : 0) : 0;
    const int nk = (int) (p - lo + 1);
    const int j0 = (blockIdx.y * warps + warp) * kValueKeys, j1 = j0 + kValueKeys < nk ? j0 + kValueKeys : nk;
    const size_t head0 = size_t(tt) * a.n_head + size_t(g) * per;
    const float* s = a.score + head0 * nk_max;
    const int mine = j0 + lane % kValueKeys;   // the key of this lane's weight, for head lane / kValueKeys
    const float weight = lane < per * kValueKeys && mine < nk ? s[size_t(lane / kValueKeys) * nk_max + mine] : 0.0f;
    float acc[kMaxPer][kLaneHalves];
#pragma unroll
    for (int hp = 0; hp < kMaxPer; ++hp) {
#pragma unroll
        for (int c = 0; c < kLaneHalves; ++c) acc[hp][c] = 0.0f;
    }
    // the warp's values first, so that their reads are on their way together; then each in its turn
    long long row = a.ring > 0 ? (lo + j0) % a.ring : lo + j0;
    float v[kValueKeys][kLaneHalves];
#pragma unroll
    for (int u = 0; u < kValueKeys; ++u) {
        if (j0 + u < j1) {
            if (KV == kKvFp8) lane_fp8(vals, row, a.n_kv, a.dim, g, lane, lh, v[u]);
            else if (KV == kKvQ8) lane_q8(vals, row, a.n_kv, a.dim, g, lane, lh, v[u]);
            else lane_halves((const __half*) vals + (size_t(row) * a.n_kv + g) * a.dim, lane, lh, v[u]);
            if (++row == a.ring) row = 0;
        }
    }
#pragma unroll
    for (int u = 0; u < kValueKeys; ++u) {
        if (j0 + u < j1) {
#pragma unroll
            for (int hp = 0; hp < kMaxPer; ++hp) {
                if (hp < per) {
                    const float w = __shfl_sync(0xffffffff, weight, hp * kValueKeys + u, WARP);
#pragma unroll
                    for (int c = 0; c < kLaneHalves; ++c) acc[hp][c] += w * v[u][c];
                }
            }
        }
    }
#pragma unroll
    for (int hp = 0; hp < kMaxPer; ++hp) {
#pragma unroll
        for (int c = 0; c < kLaneHalves; ++c) {
            if (hp < per && c < lh) sums[(size_t(warp) * per + hp) * a.dim + lane * lh + c] = acc[hp][c];
        }
    }
    __syncthreads();
    for (int e = tid; e < per * a.dim; e += blockDim.x) {   // value e % dim of head e / dim
        float total = 0.0f;
        for (int w = 0; w < warps; ++w) total += sums[size_t(w) * per * a.dim + e];
        parts[((head0 + e / a.dim) * n_parts + blockIdx.y) * a.dim + e % a.dim] = total;
    }
}

template <int KV>
__global__ void __launch_bounds__(1024) ink_value_kernel(InkAttn a, int t0, int nk_max, float* parts, int n_parts) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    value_block<KV>(a, a.vals, t0, nk_max, parts, n_parts);
}

// ink_attend_rows(): the value kernel's blocks of token t0 + blockIdx.x / n_kv over that token's own values
template <int KV>
__global__ void __launch_bounds__(1024) ink_value_rows_kernel(InkAttn a, int t0, int nk_max, float* parts, int n_parts,
                                                              InkAttnRows c) {
    pdl_trigger();
    pdl_wait();
    value_block<KV>(a, c.vals[t0 + blockIdx.x / a.n_kv], t0, nk_max, parts, n_parts);
}

// A token's head a block: its first `mine` parts added in their order (of n_parts a head in parts), over its
// softmax's norm.
__device__ __forceinline__ void join_block(const InkAttn& a, int t0, const float* parts, int n_parts, int mine,
                                           const float* norm) {
    const int tt = blockIdx.x / a.n_head, h = blockIdx.x % a.n_head, t = t0 + tt;
    const float inv = 1.0f / norm[blockIdx.x];
    const float* pt = parts + size_t(blockIdx.x) * n_parts * a.dim;
    for (int i = threadIdx.x; i < a.dim; i += blockDim.x) {
        float total = 0.0f;
        for (int k = 0; k < mine; ++k) total += pt[size_t(k) * a.dim + i];
        const size_t at = (size_t(t) * a.n_head + h) * a.dim + i;
        float v = total * inv;
        if (a.gate) v *= 1.0f / (1.0f + expf(-a.gate[at]));   // ink_mul_sigmoid_kernel's
        a.out[at] = v;
        if (a.q8) q8_put((QB*) a.q8, (int) at, v);   // one token's row, for the output product
    }
}

__global__ void ink_join_kernel(InkAttn a, int t0, const float* parts, int n_parts, const float* norm) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    join_block(a, t0, parts, n_parts, n_parts, norm);
}

// ink_attend_rows(): a token's head adds the parts it has alone (InkAttnRows::parts)
__global__ void ink_join_rows_kernel(InkAttn a, int t0, const float* parts, int n_parts, const float* norm,
                                     InkAttnRows c) {
    pdl_trigger();
    pdl_wait();
    join_block(a, t0, parts, n_parts, c.parts[t0 + blockIdx.x / a.n_head], norm);
}

// The tensor cores' m16n8k16 step, ldmatrix and the copies into shared memory, as kernels.cu has them for its
// attn_flash kernels: the step and ldmatrix from Turing on (INK_MMA; Turing runs the step as two of k = 8 on the
// same registers), the copies from Ampere on (INK_FLASH).  ink_attend_batch() asks the kernels' PTX version.
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800)
#define INK_FLASH 0
#else
#define INK_FLASH 1
#endif
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750)
#define INK_MMA 0
#else
#define INK_MMA 1
#endif
#if INK_MMA
__device__ __forceinline__ void ink_mma(float* c, const uint32_t* a, const uint32_t* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ void ink_ldsm4(uint32_t* r, const __half* p, bool trans) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    if (trans)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ uint32_t ink_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}
#endif

#if INK_FLASH
// 16 bytes from global to shared memory, or zeros where `whole` is false, a thread's copies a group until
// ink_cp_wait() (cp.async)
__device__ __forceinline__ void ink_cp16(__half* dst, const __half* src, bool whole) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src), "r"(whole ? 16 : 0));
}

__device__ __forceinline__ void ink_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int ahead> __device__ __forceinline__ void ink_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(ahead));
}
#endif

// ink_attend() of a prompt batch on the tensor cores (ink_attend_batch()), FlashAttention-2's way, as kernels.cu's
// attn_flash128_kernel goes: a launch block takes TOKENS tokens of a key head (32 when LEAN), a warp 16 tokens of one of the
// head's G query heads, and the keys go kFlashKeys at a time, two buffers of keys and values in turn, each row's
// softmax kept as it goes, so no score leaves the registers.  A score is what attend_batch() with an AttnBias makes
// of it: the query rounded to halves, its product with the key summed in floats and times the scale, plus the bias
// by distance (r . rel_bias's column, in floats, in the same order), less the sink.  A weight is rounded to a half
// for the values' product and summed as that half, and the softmax has one more term of score 0.  Fragment layouts
// are PTX's for mma.m16n8k16: a thread holds rows gid and gid + 8 and columns 2 tig and 2 tig + 1 of each block of 8.
// Heads of 128 values, 16 relative values; the block's r is in shared memory beside the keys and values.
// LEAN (A100-class cards, sm_80 only; ink_attend_batch() picks it): a block takes 32 tokens (6 warps on Inkling 2's
// 3 query heads a key head; with 64 its 12 warps of 384 threads get at most 168 registers each and spill); with up to
// 3 query heads on a key head the bias table's columns of the distances a tile spans are in shared memory too,
// copied in with its keys and values; the blocks of the last tokens, which read the most keys, start first; a warp
// skips the tiles none of its rows sees, tests each score only on the tiles a row's diagonal, window edge or end
// crosses, and leaves its output as it is when no row's top moved.  None of that changes a bit: a tile no row sees
// would add weights of 0 and scale by 1, every tile still starts at a multiple of kFlashKeys, and a row's tiles and
// the order of its sums are the same.  On the other cards (sm_86, sm_89, sm_120) it was slower per layer, so they
// keep the kernel without LEAN.
// TOKENS without LEAN (64, 48 or 32; ink_flash_tokens() picks it by how the blocks fill the card's last wave): a
// block's rows.  A row's scores, weights and sums are the same whatever block it is in: its tiles of keys still start
// at multiples of kFlashKeys and go in the same order, and a tile none of whose keys the row sees (the block's wider
// span) adds weights of 0 and scales by 1, or keeps the zeros before the row's first key, so TOKENS changes no bit.
constexpr int kFlashKeys = 64, kFlashRow = 128 + 8, kFlashRel = 16;
constexpr int flash_tokens(bool lean) { return lean ? 32 : 64; }   // the kernel's default TOKENS
constexpr size_t kFlashKv = size_t(4) * kFlashKeys * kFlashRow * sizeof(__half);
// the shared memory of ink_flash_kernel<G, LEAN, TOKENS>: keys and values, the table's columns (LEAN, G < 4), r
constexpr size_t flash_smem(int g, bool lean, int tokens) {
    return kFlashKv + (lean && g < 4 ? size_t(2) * (tokens + kFlashKeys) * kFlashRel * sizeof(float) : 0) +
           size_t(tokens) * g * kFlashRel * sizeof(float);
}

template <int G, bool LEAN, int TOKENS = flash_tokens(LEAN)>
__global__ void __launch_bounds__(TOKENS / 16 * G * 32, 1) ink_flash_kernel(InkAttn a) {
    static_assert(!LEAN || TOKENS == 32, "LEAN takes 32 tokens a block");
    static_assert(TOKENS % 16 == 0, "a warp takes 16 tokens");
#if INK_FLASH
    constexpr int DIM = 128, KEYS = kFlashKeys, ROW = kFlashRow, NR = kFlashRel, THREADS = TOKENS / 16 * G * 32;
    constexpr int SPAN = TOKENS + KEYS;   // a tile's distances (one more than it spans, for 16 bytes)
    extern __shared__ __align__(16) __half ink_flash_sm[];
    __half(*ks)[KEYS][ROW] = reinterpret_cast<__half(*)[KEYS][ROW]>(ink_flash_sm);
    __half(*vs)[KEYS][ROW] = reinterpret_cast<__half(*)[KEYS][ROW]>(ink_flash_sm + 2 * KEYS * ROW);
    float(*tb)[SPAN][NR] = reinterpret_cast<float(*)[SPAN][NR]>(ink_flash_sm + 4 * KEYS * ROW);
    constexpr bool TB = LEAN && G < 4;   // the table's columns in shared memory
    float* rs = &tb[TB ? 2 : 0][0][0];   // [token of the block][head of G][NR]
    const int g = blockIdx.y, t0 = (LEAN ? gridDim.x - 1 - blockIdx.x : blockIdx.x) * TOKENS,
              lane = threadIdx.x % 32, w = threadIdx.x / 32;
    const int gid = lane / 4, tig = lane % 4, hg = w % G, sub = w / G, head = g * G + hg;
    const int tok[2] = {t0 + 16 * sub + gid, t0 + 16 * sub + gid + 8};
    for (int i = threadIdx.x; i < TOKENS * G * NR; i += THREADS) {
        const int t = t0 + i / (G * NR);
        rs[i] = t < a.rows ? a.r[(size_t(t) * a.n_head + g * G + i / NR % G) * NR + i % NR] : 0.0f;
    }
    uint32_t qa[8][4];   // a block of 16 values a step
#pragma unroll
    for (int k = 0; k < 8; ++k)
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                float2 x = make_float2(0.0f, 0.0f);
                if (tok[i] < a.rows)
                    x = *reinterpret_cast<const float2*>(a.q + (size_t(tok[i]) * a.n_head + head) * DIM + 16 * k +
                                                         8 * hi + 2 * tig);
                qa[k][i + 2 * hi] = ink_h2(x.x, x.y);
            }
    float sink[2];
    for (int i = 0; i < 2; ++i) sink[i] = tok[i] < a.rows ? a.sink[size_t(tok[i]) * a.n_head + head] : 0.0f;
    float o[16][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    // the block's tokens attend positions [lo, hi): from the first one's window to the last one
    const long long hi = a.pos + min(t0 + TOKENS, a.rows);
    const long long lo = a.window > 0 ? max(0LL, (long long) (a.pos + t0 - a.window + 1)) : 0;
    const __half* kc = reinterpret_cast<const __half*>(a.keys);
    const __half* vc = reinterpret_cast<const __half*>(a.vals);
    const auto fetch = [&](int buf, long long k0) {
        // the table's columns of distances d0 + e (as many as the tile spans), zeros past the extent and before 0
        const long long d0 = a.pos + t0 - (k0 + KEYS - 1);
        if (TB && d0 < a.extent)
            for (int i = threadIdx.x; i < SPAN * NR / 4; i += THREADS) {
                const long long d = d0 + i / (NR / 4);
                const bool in = d >= 0 && d < a.extent;
                ink_cp16(reinterpret_cast<__half*>(&tb[buf][i / (NR / 4)][4 * (i % (NR / 4))]),
                         reinterpret_cast<const __half*>(a.rel_bias + (in ? d : 0) * NR + 4 * (i % (NR / 4))), in);
            }
        for (int i = threadIdx.x; i < KEYS * DIM / 8; i += THREADS) {
            const int r = i / (DIM / 8), c = 8 * (i % (DIM / 8));
            const long long key = k0 + r;
            const bool whole = key >= lo && key < hi;   // the others are zeros, which no weight reads
            if (a.kv8) {   // eight codes of the key and of the value, as halves times their head's scale
                uint4 kh = make_uint4(0, 0, 0, 0), vh = make_uint4(0, 0, 0, 0);
                if (whole) {
                    const size_t rb = (size_t) (a.ring > 0 ? key % a.ring : key) * kv8_row(a.n_kv, DIM);
                    for (int side = 0; side < 2; ++side) {
                        const uint8_t* row = (const uint8_t*) (side ? a.vals : a.keys) + rb;
                        const float sc = reinterpret_cast<const float*>(row + (size_t) a.n_kv * DIM)[g];
                        const uint2 q = *reinterpret_cast<const uint2*>(row + (size_t) g * DIM + c);
                        const uint8_t* b = reinterpret_cast<const uint8_t*>(&q);
                        uint32_t h[4];
#pragma unroll
                        for (int e = 0; e < 4; ++e) h[e] = ink_h2(e4m3_val(b[2 * e]) * sc, e4m3_val(b[2 * e + 1]) * sc);
                        (side ? vh : kh) = make_uint4(h[0], h[1], h[2], h[3]);
                    }
                }
                *reinterpret_cast<uint4*>(&ks[buf][r][c]) = kh;
                *reinterpret_cast<uint4*>(&vs[buf][r][c]) = vh;
                continue;
            }
            if (a.kvq8) {   // eight codes of the key and of the value, as halves times their block's scale
                uint4 kh = make_uint4(0, 0, 0, 0), vh = make_uint4(0, 0, 0, 0);
                if (whole) {
                    const size_t rb = (size_t) (a.ring > 0 ? key % a.ring : key) * kvq8_row(a.n_kv, DIM);
                    for (int side = 0; side < 2; ++side) {
                        const uint8_t* row = (const uint8_t*) (side ? a.vals : a.keys) + rb;
                        const size_t at = (size_t) g * DIM + c;
                        const float d = __half2float(__ushort_as_half(kvq8_scales(row, a.n_kv, DIM)[at >> 5]));
                        const uint2 q = *reinterpret_cast<const uint2*>(row + at);
                        const int8_t* b = reinterpret_cast<const int8_t*>(&q);
                        uint32_t h[4];
#pragma unroll
                        for (int e = 0; e < 4; ++e) h[e] = ink_h2((float) b[2 * e] * d, (float) b[2 * e + 1] * d);
                        (side ? vh : kh) = make_uint4(h[0], h[1], h[2], h[3]);
                    }
                }
                *reinterpret_cast<uint4*>(&ks[buf][r][c]) = kh;
                *reinterpret_cast<uint4*>(&vs[buf][r][c]) = vh;
                continue;
            }
            const size_t at = whole ? (size_t(a.ring > 0 ? key % a.ring : key) * a.n_kv + g) * DIM + c : 0;
            ink_cp16(&ks[buf][r][c], kc + at, whole);
            ink_cp16(&vs[buf][r][c], vc + at, whole);
        }
        ink_cp_commit();
    };
    const long long first = lo / KEYS * KEYS;
    fetch(0, first);
    int buf = 0;
    for (long long k0 = first; k0 < hi; k0 += KEYS, buf ^= 1) {
        __syncthreads();   // the buffer the last block was worked on in is free (and rs is written)
        if (k0 + KEYS < hi) {
            fetch(buf ^ 1, k0 + KEYS);
            ink_cp_wait<1>();
        } else {
            ink_cp_wait<0>();
        }
        __syncthreads();
        // the warp's rows' positions own0..own1: whether one sees a key of the tile, and whether some row does not
        // see some key (a score test each)
        const int w0 = t0 + 16 * sub, w1 = min(w0 + 15, a.rows - 1);
        const long long own0 = a.pos + w0, own1 = a.pos + w1, d0 = a.pos + t0 - (k0 + KEYS - 1);
        if (LEAN && (w0 >= a.rows || k0 > own1 || (a.window > 0 && k0 + KEYS - 1 <= own0 - a.window))) continue;
        const bool edge = !LEAN || w0 + 15 >= a.rows || k0 + KEYS - 1 > own0 || (a.window > 0 && k0 <= own1 - a.window);
        float s[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < 8; k += 2) {
                uint32_t b[4];   // keys 8 n.., values 16 k.. and 16 (k + 1)..
                ink_ldsm4(b, &ks[buf][8 * n + lane % 8][16 * k + 8 * (lane / 8)], false);
                ink_mma(s[n], qa[k], b);
                ink_mma(s[n], qa[k + 1], b + 2);
            }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const long long own = a.pos + tok[i];   // the row's token attends positions (own - window, own]
            float rr[NR];
            const float* rp = rs + ((16 * sub + gid + 8 * i) * G + hg) * NR;
#pragma unroll
            for (int q = 0; q < NR; ++q) rr[q] = rp[q];
            float mx = -INFINITY;
#pragma unroll
            for (int n = 0; n < 8; ++n)
#pragma unroll
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    const long long key = k0 + 8 * n + 2 * tig + j % 2;
                    float v = -INFINITY;
                    if (!edge || (tok[i] < a.rows && key <= own && (a.window <= 0 || key > own - a.window))) {
                        const long long d = own - key;
                        float add = 0.0f;
                        if (d < a.extent) {
                            const float4* col = TB ? reinterpret_cast<const float4*>(&tb[buf][d - d0][0])
                                                   : reinterpret_cast<const float4*>(a.rel_bias + size_t(d) * NR);
#pragma unroll
                            for (int q = 0; q < NR / 4; ++q) {
                                const float4 c = TB ? col[q] : __ldg(col + q);
                                add += c.x * rr[4 * q];
                                add += c.y * rr[4 * q + 1];
                                add += c.z * rr[4 * q + 2];
                                add += c.w * rr[4 * q + 3];
                            }
                        }
                        v = s[n][j] * a.scale;
                        v += add - sink[i];
                    }
                    s[n][j] = v;
                    mx = fmaxf(mx, v);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float now = fmaxf(m[i], mx), base = now == -INFINITY ? 0.0f : now;
            const float f = expf(m[i] - base);
            m[i] = now;
            l[i] *= f;
            if (!LEAN || __any_sync(0xffffffffu, f != 1.0f))   // times 1 changes no bit
#pragma unroll
                for (int n = 0; n < 16; ++n) {
                    o[n][2 * i] *= f;
                    o[n][2 * i + 1] *= f;
                }
#pragma unroll
            for (int n = 0; n < 8; ++n)
#pragma unroll
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    s[n][j] = __half2float(__float2half(expf(s[n][j] - base)));   // the weight the values get
                    l[i] += s[n][j];
                }
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {   // keys 16 k..: the weights as an A fragment
            const uint32_t pa[4] = {ink_h2(s[2 * k][0], s[2 * k][1]), ink_h2(s[2 * k][2], s[2 * k][3]),
                                    ink_h2(s[2 * k + 1][0], s[2 * k + 1][1]), ink_h2(s[2 * k + 1][2], s[2 * k + 1][3])};
#pragma unroll
            for (int n = 0; n < 16; n += 2) {
                uint32_t b[4];   // keys 16 k.., values 8 n.. and 8 (n + 1)..
                ink_ldsm4(b, &vs[buf][16 * k + lane % 8 + 8 * (lane / 8 % 2)][8 * n + 8 * (lane / 16)], true);
                ink_mma(o[n], pa, b);
                ink_mma(o[n + 1], pa, b + 2);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        if (tok[i] >= a.rows) continue;
        const float inv = 1.0f / (l[i] + expf(0.0f - m[i]));   // the term of score 0, against the row's top
        const size_t at = (size_t(tok[i]) * a.n_head + head) * DIM + 2 * tig;
#pragma unroll
        for (int n = 0; n < 16; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                float v = o[n][2 * i + e] * inv;
                if (a.gate) v *= 1.0f / (1.0f + expf(-a.gate[at + 8 * n + e]));   // ink_mul_sigmoid_kernel's
                a.out[at + 8 * n + e] = v;
            }
    }
#else
    (void) a;   // ink_attend_batch() launches it only where the PTX has the steps
#endif
}

// ink_flash_kernel's attention on Turing, which has the tensor cores' steps but neither cp.async nor the shared
// memory for its two 64-key buffers beside 64 tokens: a launch block takes TOKENS tokens of a key head, a warp 16
// tokens of one of its G query heads (as there), the keys go KEYS at a time through one buffer that all the threads
// fill with plain copies, and the blocks of the last tokens, which read the most keys, start first.  Each score,
// weight and sum is what ink_flash_kernel makes of it, in the same order; the bias table's rows of the distances a
// tile spans are copied to shared memory once for the block's warps.  Only the tiles' bounds differ, so a row's
// running top (and the weights rounded against it) can be another one's, as between two of ink_flash_kernel's
// launches of other lengths.
constexpr int kFlash75Tokens = 32, kFlash75Keys = 32;

template <int G>
constexpr size_t flash75_smem() {
    return size_t(2) * kFlash75Keys * kFlashRow * sizeof(__half) +
           (size_t(kFlash75Tokens) * G + kFlash75Tokens + kFlash75Keys - 1) * kFlashRel * sizeof(float);
}

template <int G>
__global__ void __launch_bounds__(kFlash75Tokens / 16 * G * 32) ink_flash75_kernel(InkAttn a) {
#if INK_MMA && !INK_FLASH
    constexpr int DIM = 128, KEYS = kFlash75Keys, TOKENS = kFlash75Tokens, ROW = kFlashRow, NR = kFlashRel;
    constexpr int THREADS = TOKENS / 16 * G * 32, SPAN = TOKENS + KEYS - 1;
    extern __shared__ __align__(16) __half ink_flash75_sm[];
    __half(*ks)[ROW] = reinterpret_cast<__half(*)[ROW]>(ink_flash75_sm);
    __half(*vs)[ROW] = reinterpret_cast<__half(*)[ROW]>(ink_flash75_sm + KEYS * ROW);
    float* rs = reinterpret_cast<float*>(ink_flash75_sm + 2 * KEYS * ROW);   // [token of the block][head of G][NR]
    float* tb = rs + TOKENS * G * NR;   // [distance of the tile's span][NR]: rel_bias's columns
    const int g = blockIdx.y, t0 = (gridDim.x - 1 - blockIdx.x) * TOKENS, lane = threadIdx.x % 32,
              w = threadIdx.x / 32;
    const int gid = lane / 4, tig = lane % 4, hg = w % G, sub = w / G, head = g * G + hg;
    const int tok[2] = {t0 + 16 * sub + gid, t0 + 16 * sub + gid + 8};
    for (int i = threadIdx.x; i < TOKENS * G * NR; i += THREADS) {
        const int t = t0 + i / (G * NR);
        rs[i] = t < a.rows ? a.r[(size_t(t) * a.n_head + g * G + i / NR % G) * NR + i % NR] : 0.0f;
    }
    uint32_t qa[8][4];   // a block of 16 values a step
#pragma unroll
    for (int k = 0; k < 8; ++k)
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                float2 x = make_float2(0.0f, 0.0f);
                if (tok[i] < a.rows)
                    x = *reinterpret_cast<const float2*>(a.q + (size_t(tok[i]) * a.n_head + head) * DIM + 16 * k +
                                                         8 * hi + 2 * tig);
                qa[k][i + 2 * hi] = ink_h2(x.x, x.y);
            }
    float sink[2];
    for (int i = 0; i < 2; ++i) sink[i] = tok[i] < a.rows ? a.sink[size_t(tok[i]) * a.n_head + head] : 0.0f;
    float o[16][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    // the block's tokens attend positions [lo, hi): from the first one's window to the last one
    const long long hi = a.pos + min(t0 + TOKENS, a.rows);
    const long long lo = a.window > 0 ? max(0LL, (long long) (a.pos + t0 - a.window + 1)) : 0;
    const __half* kc = reinterpret_cast<const __half*>(a.keys);
    const __half* vc = reinterpret_cast<const __half*>(a.vals);
    for (long long k0 = lo / KEYS * KEYS; k0 < hi; k0 += KEYS) {
        __syncthreads();   // the last tile's keys, values and table rows are read (and rs is written)
        for (int i = threadIdx.x; i < KEYS * DIM / 8; i += THREADS) {
            const int r = i / (DIM / 8), c = 8 * (i % (DIM / 8));
            const long long key = k0 + r;
            uint4 kx = make_uint4(0, 0, 0, 0), vx = kx;   // zeros outside [lo, hi), which no weight reads
            if (key >= lo && key < hi) {
                const size_t at = (size_t(a.ring > 0 ? key % a.ring : key) * a.n_kv + g) * DIM + c;
                kx = *reinterpret_cast<const uint4*>(kc + at);
                vx = *reinterpret_cast<const uint4*>(vc + at);
            }
            *reinterpret_cast<uint4*>(&ks[r][c]) = kx;
            *reinterpret_cast<uint4*>(&vs[r][c]) = vx;
        }
        // the distances own - key of the tile's scores are d0 .. d0 + SPAN - 1
        const long long d0 = a.pos + t0 - (k0 + KEYS - 1);
        const bool near = d0 < a.extent;
        if (near)
            for (int i = threadIdx.x; i < SPAN * NR / 4; i += THREADS) {
                const long long d = d0 + i / (NR / 4);
                float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (d >= 0 && d < a.extent) x = __ldg(reinterpret_cast<const float4*>(a.rel_bias + size_t(d) * NR) + i % (NR / 4));
                reinterpret_cast<float4*>(tb)[i] = x;
            }
        __syncthreads();
        float s[KEYS / 8][4];
#pragma unroll
        for (int n = 0; n < KEYS / 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < 8; k += 2) {
                uint32_t b[4];   // keys 8 n.., values 16 k.. and 16 (k + 1)..
                ink_ldsm4(b, &ks[8 * n + lane % 8][16 * k + 8 * (lane / 8)], false);
                ink_mma(s[n], qa[k], b);
                ink_mma(s[n], qa[k + 1], b + 2);
            }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const long long own = a.pos + tok[i];   // the row's token attends positions (own - window, own]
            float rr[NR];
            const float* rp = rs + ((16 * sub + gid + 8 * i) * G + hg) * NR;
#pragma unroll
            for (int q = 0; q < NR; ++q) rr[q] = rp[q];
            float mx = -INFINITY;
#pragma unroll
            for (int n = 0; n < KEYS / 8; ++n)
#pragma unroll
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    const long long key = k0 + 8 * n + 2 * tig + j % 2;
                    float v = -INFINITY;
                    if (tok[i] < a.rows && key <= own && (a.window <= 0 || key > own - a.window)) {
                        const long long d = own - key;
                        float add = 0.0f;
                        if (d < a.extent) {
                            const float4* col = reinterpret_cast<const float4*>(tb + size_t(d - d0) * NR);
#pragma unroll
                            for (int q = 0; q < NR / 4; ++q) {
                                const float4 c = col[q];
                                add += c.x * rr[4 * q];
                                add += c.y * rr[4 * q + 1];
                                add += c.z * rr[4 * q + 2];
                                add += c.w * rr[4 * q + 3];
                            }
                        }
                        v = s[n][j] * a.scale;
                        v += add - sink[i];
                    }
                    s[n][j] = v;
                    mx = fmaxf(mx, v);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float now = fmaxf(m[i], mx), base = now == -INFINITY ? 0.0f : now;
            const float f = expf(m[i] - base);
            m[i] = now;
            l[i] *= f;
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                o[n][2 * i] *= f;
                o[n][2 * i + 1] *= f;
            }
#pragma unroll
            for (int n = 0; n < KEYS / 8; ++n)
#pragma unroll
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    s[n][j] = __half2float(__float2half(expf(s[n][j] - base)));   // the weight the values get
                    l[i] += s[n][j];
                }
        }
#pragma unroll
        for (int k = 0; k < KEYS / 16; ++k) {   // keys 16 k..: the weights as an A fragment
            const uint32_t pa[4] = {ink_h2(s[2 * k][0], s[2 * k][1]), ink_h2(s[2 * k][2], s[2 * k][3]),
                                    ink_h2(s[2 * k + 1][0], s[2 * k + 1][1]), ink_h2(s[2 * k + 1][2], s[2 * k + 1][3])};
#pragma unroll
            for (int n = 0; n < 16; n += 2) {
                uint32_t b[4];   // keys 16 k.., values 8 n.. and 8 (n + 1)..
                ink_ldsm4(b, &vs[16 * k + lane % 8 + 8 * (lane / 8 % 2)][8 * n + 8 * (lane / 16)], true);
                ink_mma(o[n], pa, b);
                ink_mma(o[n + 1], pa, b + 2);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        if (tok[i] >= a.rows) continue;
        const float inv = 1.0f / (l[i] + expf(0.0f - m[i]));   // the term of score 0, against the row's top
        const size_t at = (size_t(tok[i]) * a.n_head + head) * DIM + 2 * tig;
#pragma unroll
        for (int n = 0; n < 16; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                float v = o[n][2 * i + e] * inv;
                if (a.gate) v *= 1.0f / (1.0f + expf(-a.gate[at + 8 * n + e]));   // ink_mul_sigmoid_kernel's
                a.out[at + 8 * n + e] = v;
            }
    }
#else
    (void) a;   // ink_attend_batch() launches it only where the PTX has the steps
#endif
}
// ink_attend() of a prompt batch on RDNA4's matrix cores (ink_attend_batch() on gfx12): v_wmma_f32_16x16x16_f16
// (wave32) steps, done transposed, S^T = K Q^T and O^T = V^T P^T.  A launch block takes kWmTokens tokens of a key
// head's 3 query heads, 4 warps of 16 rows (row = token * 3 + head of the group; 21 tokens, the last row unused), and
// the keys go kWmKeys at a time counted from position 0, so a token's tiles are the same whichever call it is in.
// gfx12's fragments: a lane l holds A[l % 16][(l / 16) * 8 + i] and B[(l / 16) * 8 + i][l % 16], and its C/D is
// D[(l / 16) * 8 + i][l % 16].  So lane l's scores are its row l % 16's at 8 keys, which is what the same lane's B
// fragment of P^T holds: a row's online softmax lives in the two lanes l and l ^ 16, and the weights never leave the
// registers.  The queries (times the scale, in log2 units), R, the bias table's rows and the weights go into the
// products as a half and the half of what it leaves (hi + lo, two or three steps a product), so a sum is float-exact
// to about 2^-22 of each term; the cache's halves are exact.  The bias is R times the table's rows of the distances a
// warp's rows reach in the tile, through the warp's shared memory, read back at each score's distance; only tiles
// with a distance below the extent compute it.  The softmax has the sink's term, and the gate goes on at the end.
#if defined(__HIPCC__)
#if defined(__gfx1200__) || defined(__gfx1201__)
#define INK_WMMA 1
#else
#define INK_WMMA 0
#endif
constexpr int kWmPer = 3, kWmDim = 128, kWmRel = 16;   // Inkling 2 Mini's heads
constexpr int kWmWarps = 4, kWmKeys = 32, kWmTokens = 16 * kWmWarps / kWmPer;
constexpr int kWmSlice = 64;        // the block's distances a tile (rows of the table)
constexpr int kWmDist = 48;         // a warp's distances a tile: 3 steps of 16
static_assert(16 * (kWmWarps - 1) / kWmPer + kWmDist <= kWmSlice, "ink_wmma_kernel: the slice covers every warp");
static_assert(16 / kWmPer + 1 + kWmKeys <= kWmDist, "ink_wmma_kernel: a warp's distances fit its steps");
static_assert(kWmKeys * kWmDim % (16 * kWmWarps * WARP) == 0, "ink_wmma_kernel: the tiles' copies fill the block");

#if INK_WMMA
constexpr int kWmKr = kWmDim + 8;   // a key row of shared memory, halves
constexpr int kWmVs = kWmKeys + 8;  // a dim's row of the values, halves
constexpr int kWmBe = kWmDist + 4;  // a row's bias of a tile, floats
// The weights go into the products times 2^12 and the output comes out divided by it (both exact): a weight as small
// as 1e-6 is then a normal half, and its lo half is not lost below 6e-8.
constexpr float kWmPScale = 4096.0f;

typedef _Float16 ink_wh8 __attribute__((ext_vector_type(8)));
typedef float ink_wf8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ ink_wf8 ink_wmma16(ink_wh8 a, ink_wh8 b, ink_wf8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

__device__ __forceinline__ ink_wf8 ink_wf8_zero() { return ink_wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; }

// x as a half and the half of what that leaves
__device__ __forceinline__ void ink_hilo(const float* x, ink_wh8& hi, ink_wh8& lo) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        hi[i] = (_Float16) x[i];
        lo[i] = (_Float16) (x[i] - (float) hi[i]);
    }
}

__device__ __forceinline__ void ink_load8(const float* p, float* x) {
    const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
    x[0] = a.x, x[1] = a.y, x[2] = a.z, x[3] = a.w, x[4] = b.x, x[5] = b.y, x[6] = b.z, x[7] = b.w;
}

// The warp's shared-memory writes before its lanes read each other's
__device__ __forceinline__ void ink_wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
#endif

__global__ void __launch_bounds__(kWmWarps * WARP) ink_wmma_kernel(InkAttn a) {
#if INK_WMMA
    constexpr int DIM = kWmDim, PER = kWmPer, KEYS = kWmKeys;
    __shared__ __align__(16) _Float16 ks[KEYS * kWmKr];   // [KEYS][KR]
    __shared__ __align__(16) _Float16 vt[DIM * kWmVs];    // [DIM][VS]: a dim's keys, an A fragment's row
    __shared__ __align__(16) float slice[kWmSlice * kWmRel];   // the table's rows of the block's distances
    __shared__ __align__(16) float bias[kWmWarps][16 * kWmBe];
    constexpr float LOG2E = 1.44269504f;
    const int tid = threadIdx.x, lane = tid % WARP, w = tid / WARP, col = lane % 16, kh = lane / 16;
    const int g = blockIdx.y, t0 = (gridDim.x - 1 - blockIdx.x) * kWmTokens;   // the longest blocks first
    const int nt = min(kWmTokens, a.rows - t0), wt0 = 16 * w / PER;
    const int r = 16 * w + col, tok = r / PER, head = g * PER + r % PER;   // the lane's row
    const bool row_ok = tok < nt;
    const long long p0 = a.pos + t0, own = p0 + tok, ext = a.extent;
    const long long kb = a.window > 0 && p0 >= a.window ? p0 - a.window + 1 : 0, ke = p0 + nt;
    const size_t th = (size_t(t0) + (row_ok ? tok : 0)) * a.n_head + head;
    const __half* kc = (const __half*) a.keys;
    const __half* vc = (const __half*) a.vals;

    // the row's query (B fragments of Q^T, 16 dims a step) and R, in log2 units
    ink_wh8 qh[DIM / 16], ql[DIM / 16], rh, rl;
    {
        const float c = a.scale * LOG2E;
#pragma unroll
        for (int kk = 0; kk < DIM / 16; ++kk) {
            float x[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            if (row_ok) ink_load8(a.q + th * DIM + kk * 16 + kh * 8, x);
#pragma unroll
            for (int i = 0; i < 8; ++i) x[i] *= c;
            ink_hilo(x, qh[kk], ql[kk]);
        }
        float x[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (row_ok) ink_load8(a.r + th * kWmRel + kh * 8, x);
#pragma unroll
        for (int i = 0; i < 8; ++i) x[i] *= LOG2E;
        ink_hilo(x, rh, rl);
    }
    ink_wf8 acc[DIM / 16];   // O^T: dims dt * 16 + kh * 8 + i of the row
#pragma unroll
    for (int dt = 0; dt < DIM / 16; ++dt) acc[dt] = ink_wf8_zero();
    float m = -INFINITY, l = 0.0f;

    for (long long kt = kb < ke ? kb / KEYS * KEYS : ke; kt < ke; kt += KEYS) {
        __syncthreads();   // the last tile's keys, values and table rows are read
        // the keys, a 16-byte copy a thread; a key outside [kb, ke) is zeros (its cache row may be anything)
#pragma unroll
        for (int it = 0; it < KEYS * (DIM / 8) / (kWmWarps * WARP); ++it) {
            const int i = tid + it * kWmWarps * WARP, j = i / (DIM / 8), c = i % (DIM / 8);
            const long long kp = kt + j;
            const bool live = kp >= kb && kp < ke;
            const size_t at = (size_t(a.ring > 0 ? kp % a.ring : kp) * a.n_kv + g) * DIM + c * 8;
            *reinterpret_cast<uint4*>(ks + j * kWmKr + c * 8) =
                live ? *reinterpret_cast<const uint4*>(kc + at) : make_uint4(0, 0, 0, 0);
        }
        // the values by dim, two keys a thread: lanes of a half-warp take 16 key pairs, so a write is 16 words
#pragma unroll
        for (int it = 0; it < KEYS / 2 * (DIM / 8) / (kWmWarps * WARP); ++it) {
            const int i = tid + it * kWmWarps * WARP, j = 2 * (i % (KEYS / 2)), c = i / (KEYS / 2);
            uint4 x[2];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const long long kp = kt + j + u;
                const bool live = kp >= kb && kp < ke;
                const size_t at = (size_t(a.ring > 0 ? kp % a.ring : kp) * a.n_kv + g) * DIM + c * 8;
                x[u] = live ? *reinterpret_cast<const uint4*>(vc + at) : make_uint4(0, 0, 0, 0);
            }
            const uint32_t w0[4] = {x[0].x, x[0].y, x[0].z, x[0].w}, w1[4] = {x[1].x, x[1].y, x[1].z, x[1].w};
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                const uint32_t sh = 16 * (d % 2);
                const uint32_t v = (w0[d / 2] >> sh & 0xffffu) | (w1[d / 2] >> sh & 0xffffu) << 16;
                *reinterpret_cast<uint32_t*>(vt + (c * 8 + d) * kWmVs + j) = v;
            }
        }
        // the table's rows of distances p0 - (kt + KEYS - 1) + e, zeros outside [0, extent)
        const long long d0 = p0 - (kt + KEYS - 1);
        if (d0 < ext)
            for (int i = tid; i < kWmSlice * 4; i += kWmWarps * WARP) {
                const long long d = d0 + i / 4;
                float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
                if (d >= 0 && d < ext) x = *reinterpret_cast<const float4*>(a.rel_bias + d * kWmRel + (i % 4) * 4);
                *reinterpret_cast<float4*>(slice + i * 4) = x;
            }
        __syncthreads();

        // S^T = K Q^T: s[t16][i] is the row's score at key kt + 16 t16 + 8 kh + i
        ink_wf8 s[KEYS / 16];
#pragma unroll
        for (int t16 = 0; t16 < KEYS / 16; ++t16) {
            const _Float16* kr = ks + (t16 * 16 + col) * kWmKr + kh * 8;
            ink_wf8 c = ink_wf8_zero();
#pragma unroll
            for (int kk = 0; kk < DIM / 16; ++kk) {
                const ink_wh8 kf = *reinterpret_cast<const ink_wh8*>(kr + kk * 16);
                c = ink_wmma16(kf, qh[kk], c);
                c = ink_wmma16(kf, ql[kk], c);
            }
            s[t16] = c;
        }

        // the bias: the warp's distances dw + e, e = 16 T + 8 kh + i, times the rows' R, put by row and read back
        const long long dw = d0 + wt0;
        const bool near = dw < ext;
        float* bw = bias[w];
        if (near) {
#pragma unroll
            for (int T = 0; T < kWmDist / 16; ++T) {
                float x[8];
                ink_load8(slice + (wt0 + T * 16 + col) * kWmRel + kh * 8, x);
                ink_wh8 th_, tl_;
                ink_hilo(x, th_, tl_);
                ink_wf8 c = ink_wmma16(th_, rh, ink_wf8_zero());
                c = ink_wmma16(tl_, rh, c);
                c = ink_wmma16(th_, rl, c);
#pragma unroll
                for (int i = 0; i < 8; ++i) bw[col * kWmBe + T * 16 + kh * 8 + i] = c[i];
            }
            ink_wave_sync();
        }

        // the mask and the online softmax, in log2 units
        const int e0 = tok - wt0 + KEYS - 1;   // the row's distance to key kt, less dw
        float mx = -INFINITY;
#pragma unroll
        for (int t16 = 0; t16 < KEYS / 16; ++t16)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int k = t16 * 16 + kh * 8 + i;
                const long long kp = kt + k, dist = own - kp;
                const bool seen = row_ok && kp >= kb && kp < ke && dist >= 0 && (a.window <= 0 || dist < a.window);
                float v = -INFINITY;
                if (seen) v = near ? s[t16][i] + bw[col * kWmBe + e0 - k] : s[t16][i];
                s[t16][i] = v;
                mx = fmaxf(mx, v);
            }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
        const float now = fmaxf(m, mx), base = now == -INFINITY ? 0.0f : now;
        const float f = exp2f(m - base);
        m = now;
        if (f != 1.0f) {
            l *= f;
#pragma unroll
            for (int dt = 0; dt < DIM / 16; ++dt)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[dt][i] *= f;
        }
#pragma unroll
        for (int t16 = 0; t16 < KEYS / 16; ++t16)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p = exp2f(s[t16][i] - base);
                s[t16][i] = p;
                l += p;
            }

        // O^T += V^T P^T: keys 16 kg.. of the dims 16 dt..
#pragma unroll
        for (int kg = 0; kg < KEYS / 16; ++kg) {
            const _Float16* vr = vt + col * kWmVs + kg * 16 + kh * 8;
            float p[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) p[i] = s[kg][i] * kWmPScale;
            ink_wh8 ph, pl;
            ink_hilo(p, ph, pl);
#pragma unroll
            for (int dt = 0; dt < DIM / 16; ++dt) {
                const ink_wh8 vf = *reinterpret_cast<const ink_wh8*>(vr + dt * 16 * kWmVs);
                acc[dt] = ink_wmma16(vf, ph, acc[dt]);
                acc[dt] = ink_wmma16(vf, pl, acc[dt]);
            }
        }
    }

    l += __shfl_xor_sync(0xffffffffu, l, 16);
    if (!row_ok) return;
    // the sink, in log2 units like the scores: the softmax's term of score 0 once the sink is taken off the scores
    const float sink = a.sink[th] * LOG2E, now = fmaxf(m, sink), f = exp2f(m - now);
    l = l * f + exp2f(sink - now);
    const size_t at = th * DIM + kh * 8;
#pragma unroll
    for (int dt = 0; dt < DIM / 16; ++dt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const size_t e = at + dt * 16 + i;
            float v = acc[dt][i] * f / (l * kWmPScale);
            if (a.gate) v *= 1.0f / (1.0f + expf(-a.gate[e]));   // ink_mul_sigmoid_kernel's
            a.out[e] = v;
        }
#else
    (void) a;
    __builtin_trap();   // ink_attend_batch() launches it on gfx12 only (ink_wmma_card())
#endif
}
#endif

// STRATA_FAST_ATTN=1 (opt-in; other bits): a full layer's prompt batch on ink_fast_kernel instead.  A block takes
// the 64 rows of kFastTokens(G) tokens of a key head (row r: token r / G, query head r % G, the last rows unused
// when G does not divide 64), a warp 16 of them with their queries in registers, keys kFlashKeys at a time: the
// values of a tile come in by cp.async while the warps take its scores, the next keys while they take its weights
// (one buffer each, two barriers a tile, two blocks an SM).  The scores are in log2 units (the queries times the
// scale and log2(e), and r times log2(e), rounded to halves) and 2^x is the special-function unit's ex2.approx; the
// bias goes into the scores on the tensor cores, for each token u of a warp r with the other tokens' rows zeroed
// times the table's columns of u's distances (halves), only on tiles with a distance below the extent; the sink is
// the softmax's one more term.  A row's output is rescaled only when its top moves.  HACC (cards whose tensor
// cores sum fp16 products into floats at half rate: all but sm_80, sm_90 and sm_100) sums a tile's weighted values
// in halves and adds them to the float output.  Tiles start at multiples of kFlashKeys, so a token's result does
// not depend on the tokens beside it in the call.
constexpr int kFastWarps = 4, kFastThreads = kFastWarps * 32, kFastTrow = kFlashRel + 8;
__host__ __device__ constexpr int fast_tokens(int g) { return 16 * kFastWarps / g; }
// a tile's distances: from a warp's last row's token (the last row may be an unused one) back over the tile's keys
__host__ __device__ constexpr int fast_span(int g) { return (16 * kFastWarps - 1) / g + kFlashKeys; }

#if INK_FLASH
// 2^x by the special-function unit (ex2.approx: about 2 ulp, subnormals flushed; 2^-inf = 0)
__device__ __forceinline__ float ink_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// the m16n8k16 step summing into halves
__device__ __forceinline__ void ink_mma_h(uint32_t* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#endif

template <int G, bool HACC>
__global__ void __launch_bounds__(kFastThreads, 2) ink_fast_kernel(InkAttn a) {
#if INK_FLASH
    constexpr int DIM = 128, KEYS = kFlashKeys, ROW = kFlashRow, NR = kFlashRel, TOKENS = fast_tokens(G);
    constexpr int SPAN = fast_span(G), THREADS = kFastThreads;
    constexpr float LOG2E = 1.44269504f;
    __shared__ __align__(16) __half ks[KEYS][ROW], vs[KEYS][ROW], slice[SPAN][kFastTrow];
    const int lane = threadIdx.x % 32, w = threadIdx.x / 32, gid = lane / 4, tig = lane % 4, g = blockIdx.y;
    const int t0 = (gridDim.x - 1 - blockIdx.x) * TOKENS;   // the longest blocks first
    const int nt = min(TOKENS, a.rows - t0), wt0 = 16 * w / G, wt1 = (16 * w + 15) / G;
    const long long p0 = a.pos + t0;
    int tok[2];   // the thread's rows gid and gid + 8, as tokens of the block
    bool row_ok[2];
    uint32_t qa[8][4], ra[4];   // a block of 16 values a step, in log2 units; r the same
    for (int i = 0; i < 2; ++i) {
        const int r = 16 * w + gid + 8 * i;
        tok[i] = r / G;
        row_ok[i] = tok[i] < nt;
        const size_t th = (size_t(t0) + tok[i]) * a.n_head + g * G + r % G;
        for (int u = 0; u < 2; ++u) {
            for (int k = 0; k < 8; ++k) {
                float2 x = make_float2(0.0f, 0.0f);
                if (row_ok[i]) x = *reinterpret_cast<const float2*>(a.q + th * DIM + 16 * k + 8 * u + 2 * tig);
                qa[k][i + 2 * u] = ink_h2(x.x * a.scale * LOG2E, x.y * a.scale * LOG2E);
            }
            const float* rr = a.r + th * NR + 8 * u + 2 * tig;
            ra[i + 2 * u] = row_ok[i] ? ink_h2(rr[0] * LOG2E, rr[1] * LOG2E) : 0u;
        }
    }
    float o[16][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
#pragma unroll
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;

    // the block's tokens attend positions [lo, hi); the others are zeros (their cache rows may be anything)
    const long long hi = p0 + nt, lo = a.window > 0 ? max(0LL, p0 - a.window + 1) : 0;
    const long long first = lo / KEYS * KEYS;
    const int tiles = (int) ((hi - first + KEYS - 1) / KEYS);
    const auto load = [&](__half(*dst)[ROW], const uint16_t* src, long long k0) {
        for (int i = threadIdx.x; i < KEYS * DIM / 8; i += THREADS) {
            const int r = i / (DIM / 8), c = 8 * (i % (DIM / 8));
            const long long key = k0 + r;
            const bool in = key >= lo && key < hi;
            const size_t cell = in ? (size_t) (a.ring > 0 ? key % a.ring : key) : 0;
            if (a.kv8) {   // eight codes as halves times their head's scale, as ink_flash_kernel's
                uint4 h4 = make_uint4(0, 0, 0, 0);
                if (in) {
                    const uint8_t* row = (const uint8_t*) src + cell * kv8_row(a.n_kv, DIM);
                    const float sc = reinterpret_cast<const float*>(row + (size_t) a.n_kv * DIM)[g];
                    const uint2 q = *reinterpret_cast<const uint2*>(row + (size_t) g * DIM + c);
                    const uint8_t* b = reinterpret_cast<const uint8_t*>(&q);
                    uint32_t h[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) h[e] = ink_h2(e4m3_val(b[2 * e]) * sc, e4m3_val(b[2 * e + 1]) * sc);
                    h4 = make_uint4(h[0], h[1], h[2], h[3]);
                }
                *reinterpret_cast<uint4*>(&dst[r][c]) = h4;
            } else {
                ink_cp16(&dst[r][c], reinterpret_cast<const __half*>(src) + (cell * a.n_kv + g) * DIM + c, in);
            }
        }
        ink_cp_commit();
    };
    load(ks, a.keys, first);
    for (int it = 0; it < tiles; ++it) {
        const long long k0 = first + (long long) it * KEYS;
        ink_cp_wait<0>();
        __syncthreads();   // the tile's keys are in; the last tile's values and slice are read
        load(vs, a.vals, k0);
        const long long own0 = p0 + wt0, own1 = p0 + wt1;
        // whether a row of the warp sees a key of the tile, whether some do not see some (a score test each),
        // whether a distance is below the extent (the warp's, the block's)
        const bool act = wt0 < nt && k0 <= own1 && (a.window <= 0 || k0 + KEYS - 1 > own0 - a.window);
        const bool edge = k0 + KEYS - 1 > own0 || (a.window > 0 && k0 <= own1 - a.window);
        const bool near = own0 - (k0 + KEYS - 1) < a.extent, any_near = p0 - (k0 + KEYS - 1) < a.extent;
        float s[KEYS / 8][4];
#pragma unroll
        for (int n = 0; n < KEYS / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
        if (act)
#pragma unroll
            for (int k = 0; k < 8; k += 2)
#pragma unroll
                for (int n = 0; n < KEYS / 8; ++n) {
                    uint32_t b[4];   // keys 8 n.., values 16 k.. and 16 (k + 1)..
                    ink_ldsm4(b, &ks[8 * n + lane % 8][16 * k + 8 * (lane / 8)], false);
                    ink_mma(s[n], qa[k], b);
                    ink_mma(s[n], qa[k + 1], b + 2);
                }
        if (any_near) {   // the table's columns of distances p0 - (k0 + KEYS - 1) + e, zeros past the extent and before 0
            const long long d0 = p0 - (k0 + KEYS - 1);
            for (int i = threadIdx.x; i < SPAN * NR / 2; i += THREADS) {
                const int e = i / (NR / 2), j = 2 * (i % (NR / 2));
                const long long d = d0 + e;
                float2 x = make_float2(0.0f, 0.0f);
                if (d >= 0 && d < a.extent) x = *reinterpret_cast<const float2*>(a.rel_bias + d * NR + j);
                *reinterpret_cast<uint32_t*>(&slice[e][j]) = ink_h2(x.x, x.y);
            }
        }
        ink_cp_wait<0>();
        __syncthreads();   // the tile's values and slice are in; its keys are read
        if (it + 1 < tiles) load(ks, a.keys, k0 + KEYS);
        if (!act) continue;
        if (near)   // token u's rows of r times the table's columns of its distances u + KEYS - 1 - c to key c
            for (int u = wt0; u <= wt1; ++u) {
                const uint32_t au[4] = {tok[0] == u ? ra[0] : 0u, tok[1] == u ? ra[1] : 0u,
                                        tok[0] == u ? ra[2] : 0u, tok[1] == u ? ra[3] : 0u};
#pragma unroll
                for (int n = 0; n < KEYS / 8; ++n) {
                    const __half* tr = &slice[u + KEYS - 1 - (8 * n + gid)][2 * tig];
                    const uint32_t b[2] = {*reinterpret_cast<const uint32_t*>(tr),
                                           *reinterpret_cast<const uint32_t*>(tr + 8)};
                    ink_mma(s[n], au, b);
                }
            }
        float f[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int own = (int) (p0 + tok[i] - k0);   // the row's position, from the tile's first key
            float mx = -INFINITY;
#pragma unroll
            for (int n = 0; n < KEYS / 8; ++n)
#pragma unroll
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    const int c = 8 * n + 2 * tig + j % 2;
                    if (edge && (c > own || (a.window > 0 && c <= own - a.window))) s[n][j] = -INFINITY;
                    mx = fmaxf(mx, s[n][j]);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float now = fmaxf(m[i], mx), base = now == -INFINITY ? 0.0f : now;
            f[i] = ink_ex2(m[i] - base);
            m[i] = now;
            float sum = 0.0f;
#pragma unroll
            for (int n = 0; n < KEYS / 8; ++n)
#pragma unroll
                for (int j = 2 * i; j < 2 * i + 2; ++j) sum += s[n][j] = ink_ex2(s[n][j] - base);
            l[i] = l[i] * f[i] + sum;
        }
        if constexpr (HACC) {
            uint32_t t[16][2] = {};   // the tile's weighted values in halves, rows gid and gid + 8
#pragma unroll
            for (int k = 0; k < KEYS / 16; ++k) {   // keys 16 k..: the weights as an A fragment
                const uint32_t pa[4] = {ink_h2(s[2 * k][0], s[2 * k][1]), ink_h2(s[2 * k][2], s[2 * k][3]),
                                        ink_h2(s[2 * k + 1][0], s[2 * k + 1][1]), ink_h2(s[2 * k + 1][2], s[2 * k + 1][3])};
#pragma unroll
                for (int n = 0; n < 16; n += 2) {
                    uint32_t b[4];   // keys 16 k.., values 8 n.. and 8 (n + 1)..
                    ink_ldsm4(b, &vs[16 * k + lane % 8 + 8 * (lane / 8 % 2)][8 * n + 8 * (lane / 16)], true);
                    ink_mma_h(t[n], pa, b);
                    ink_mma_h(t[n + 1], pa, b + 2);
                }
            }
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&t[n][0]));
                const float2 y = __half22float2(*reinterpret_cast<const __half2*>(&t[n][1]));
                o[n][0] = fmaf(o[n][0], f[0], x.x);
                o[n][1] = fmaf(o[n][1], f[0], x.y);
                o[n][2] = fmaf(o[n][2], f[1], y.x);
                o[n][3] = fmaf(o[n][3], f[1], y.y);
            }
        } else {
            if (!__all_sync(0xffffffffu, f[0] == 1.0f && f[1] == 1.0f))
#pragma unroll
                for (int n = 0; n < 16; ++n) {
                    o[n][0] *= f[0];
                    o[n][1] *= f[0];
                    o[n][2] *= f[1];
                    o[n][3] *= f[1];
                }
#pragma unroll
            for (int k = 0; k < KEYS / 16; ++k) {
                const uint32_t pa[4] = {ink_h2(s[2 * k][0], s[2 * k][1]), ink_h2(s[2 * k][2], s[2 * k][3]),
                                        ink_h2(s[2 * k + 1][0], s[2 * k + 1][1]), ink_h2(s[2 * k + 1][2], s[2 * k + 1][3])};
#pragma unroll
                for (int n = 0; n < 16; n += 2) {
                    uint32_t b[4];
                    ink_ldsm4(b, &vs[16 * k + lane % 8 + 8 * (lane / 8 % 2)][8 * n + 8 * (lane / 16)], true);
                    ink_mma(o[n], pa, b);
                    ink_mma(o[n + 1], pa, b + 2);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        if (!row_ok[i]) continue;
        const int r = 16 * w + gid + 8 * i;
        const size_t th = (size_t(t0) + tok[i]) * a.n_head + g * G + r % G;
        // the sink, in log2 units like the scores: the term of score 0 of ink_flash_kernel's scores less the sink
        const float sink = a.sink[th] * LOG2E, now = fmaxf(m[i], sink), f = ink_ex2(m[i] - now);
        const float scale = f / (l[i] * f + ink_ex2(sink - now));
        const size_t at = th * DIM + 2 * tig;
#pragma unroll
        for (int n = 0; n < 16; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                float v = o[n][2 * i + e] * scale;
                if (a.gate) v *= 1.0f / (1.0f + expf(-a.gate[at + 8 * n + e]));   // ink_mul_sigmoid_kernel's
                a.out[at + 8 * n + e] = v;
            }
    }
#else
    (void) a;   // ink_attend_batch() launches it only where the PTX has the steps
#endif
}

// The same attention on Turing in the shape of attn_mma_kernel from our earlier Inkling engine (#200's 7f3f4574f,
// 16f3585bd): a launch block is 4 warps of 16 query rows, row r of the block token r / G and query head r % G of the
// key head (16 * 4 / G tokens, a row unused for G = 3), keys go 32 at a time through one buffer.  It differs from
// ink_flash75_kernel in what costs it time: the relative-position bias is a product on the tensor cores too, the
// warp's R (in log2 units) times the table's rows of the distances the tile spans, each split into a half and the
// half of what that leaves (hi*hi + lo*hi + hi*lo, summed in floats: about a float product's rounding), a warp's 16
// rows by mma75_bias(G) distances in shared memory as floats, each score reading its one float; the softmax runs in
// log2 units (exp2f); and a tile that every row of a warp attends whole skips the masks.  A score is the query rounded
// to halves times the key summed in floats, times the scale, plus the bias, less the sink; a weight is rounded to a
// half for the values' product and summed as that half; the softmax has one more term of score 0, then the gate.  The
// tiles are counted from position 0 and a row's sums read nothing of the block's other rows, so a token's result does
// not depend on the call it is in (ink_flash_test's split calls).
constexpr int kMma75Warps = 4, kMma75Keys = 32;

__host__ __device__ constexpr int mma75_tokens(int g) { return 16 * kMma75Warps / g; }
// a warp's distances a tile: its rows' tokens (at most 15 / G + 2) and the keys, in blocks of 8
__host__ __device__ constexpr int mma75_bias(int g) { return (15 / g + 2 + kMma75Keys - 1 + 7) / 8 * 8; }

template <int G>
__global__ void __launch_bounds__(kMma75Warps * 32) ink_mma75_kernel(InkAttn a) {
#if INK_MMA && !INK_FLASH
    constexpr int DIM = 128, KEYS = kMma75Keys, ROW = kFlashRow, NR = kFlashRel, W = kMma75Warps, THREADS = W * 32;
    constexpr int TOKENS = mma75_tokens(G), BIAS = mma75_bias(G);
    constexpr float LOG2E = 1.44269504f;
    static_assert(NR == 16, "the bias product is one k = 16 step");
    __shared__ __align__(16) __half ks[KEYS][ROW], vs[KEYS][ROW];
    __shared__ __align__(16) float bias[W][16][BIAS];
    const int lane = threadIdx.x % 32, w = threadIdx.x / 32, gid = lane / 4, tig = lane % 4;
    const int g = blockIdx.y, t0 = (gridDim.x - 1 - blockIdx.x) * TOKENS;   // the longest blocks first
    const int nt = min(TOKENS, a.rows - t0), wt0 = 16 * w / G, wt1 = min((16 * w + 15) / G, TOKENS - 1);
    const long long p0 = a.pos + t0;
    int tok[2];
    bool ok[2];
    size_t th[2];   // the thread's rows gid and gid + 8 of the warp's: token, and its head's place in q, r, out
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = 16 * w + gid + 8 * i;
        tok[i] = r / G;
        ok[i] = tok[i] < nt;
        th[i] = ok[i] ? size_t(t0 + tok[i]) * a.n_head + g * G + r % G : 0;
    }
    uint32_t qa[8][4], ra[4], rl[4];   // a block of 16 values a step; R in log2 units, and what its halves leave
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int u = 0; u < 2; ++u) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float2 x = make_float2(0.0f, 0.0f);
                if (ok[i]) x = *reinterpret_cast<const float2*>(a.q + th[i] * DIM + 16 * k + 8 * u + 2 * tig);
                qa[k][i + 2 * u] = ink_h2(x.x, x.y);
            }
            float2 x = make_float2(0.0f, 0.0f);
            if (ok[i]) x = *reinterpret_cast<const float2*>(a.r + th[i] * NR + 8 * u + 2 * tig);
            const float2 y = make_float2(x.x * LOG2E, x.y * LOG2E);
            const __half2 h = __floats2half2_rn(y.x, y.y);
            ra[i + 2 * u] = *reinterpret_cast<const uint32_t*>(&h);
            rl[i + 2 * u] = ink_h2(y.x - __low2float(h), y.y - __high2float(h));
        }
    float sink[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) sink[i] = ok[i] ? a.sink[th[i]] * LOG2E : 0.0f;
    const float sl = a.scale * LOG2E;
    float o[16][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
#pragma unroll
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    // the block's tokens attend positions [lo, hi): from the first one's window to the last one
    const long long hi = p0 + nt, lo = a.window > 0 ? max(0LL, p0 - a.window + 1) : 0;
    const bool wfull = wt1 < nt;   // every row of the warp is a token of the call
    const __half* kc = reinterpret_cast<const __half*>(a.keys);
    const __half* vc = reinterpret_cast<const __half*>(a.vals);
    for (long long k0 = lo / KEYS * KEYS; k0 < hi; k0 += KEYS) {
        __syncthreads();   // the last tile's keys, values and table rows are read
        for (int i = threadIdx.x; i < KEYS * DIM / 8; i += THREADS) {
            const int r = i / (DIM / 8), c = 8 * (i % (DIM / 8));
            const long long key = k0 + r;
            uint4 kx = make_uint4(0, 0, 0, 0), vx = kx;   // zeros outside [lo, hi), which no weight reads
            if (key >= lo && key < hi) {
                const size_t at = (size_t(a.ring > 0 ? key % a.ring : key) * a.n_kv + g) * DIM + c;
                kx = *reinterpret_cast<const uint4*>(kc + at);
                vx = *reinterpret_cast<const uint4*>(vc + at);
            }
            *reinterpret_cast<uint4*>(&ks[r][c]) = kx;
            *reinterpret_cast<uint4*>(&vs[r][c]) = vx;
        }
        // the block's distances own - key are d0 .. ; a tile with one below the extent has a bias
        const long long d0 = p0 - (k0 + KEYS - 1);
        const bool near = d0 < a.extent;
        __syncthreads();
        float s[KEYS / 8][4];
#pragma unroll
        for (int n = 0; n < KEYS / 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < 8; k += 2) {
                uint32_t b[4];   // keys 8 n.., values 16 k.. and 16 (k + 1)..
                ink_ldsm4(b, &ks[8 * n + lane % 8][16 * k + 8 * (lane / 8)], false);
                ink_mma(s[n], qa[k], b);
                ink_mma(s[n], qa[k + 1], b + 2);
            }
        }
        if (near) {   // the warp's rows times its distances: column e is distance d0 + wt0 + e
#pragma unroll
            for (int n = 0; n < BIAS / 8; ++n) {
                // the B fragment: the table's row of distance d0 + wt0 + 8 n + gid, values 2 tig.. and 8 + 2 tig..,
                // zeros before 0 and from the extent on; as halves, and the halves of what they leave
                const long long d = d0 + wt0 + 8 * n + gid;
                float2 x0 = make_float2(0.0f, 0.0f), x1 = x0;
                if (d >= 0 && d < a.extent) {
                    x0 = __ldg(reinterpret_cast<const float2*>(a.rel_bias + size_t(d) * NR + 2 * tig));
                    x1 = __ldg(reinterpret_cast<const float2*>(a.rel_bias + size_t(d) * NR + 8 + 2 * tig));
                }
                const __half2 h0 = __floats2half2_rn(x0.x, x0.y), h1 = __floats2half2_rn(x1.x, x1.y);
                const uint32_t b[2] = {*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1)};
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const uint32_t bl[2] = {ink_h2(x0.x - __low2float(h0), x0.y - __high2float(h0)),
                                        ink_h2(x1.x - __low2float(h1), x1.y - __high2float(h1))};
                ink_mma(c, ra, b);    // hi * hi
                ink_mma(c, rl, b);    // + lo * hi
                ink_mma(c, ra, bl);   // + hi * lo: about what the products in floats give
                *reinterpret_cast<float2*>(&bias[w][gid][8 * n + 2 * tig]) = make_float2(c[0], c[1]);
                *reinterpret_cast<float2*>(&bias[w][gid + 8][8 * n + 2 * tig]) = make_float2(c[2], c[3]);
            }
            __syncwarp();
        }
        // whether every row of the warp attends every key of the tile: then no score needs its mask
        const bool whole = wfull && k0 + KEYS - 1 <= p0 + wt0 && (a.window <= 0 || k0 > p0 + wt1 - a.window);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            // the row's token attends positions (own - window, own]: columns c of the tile with lo_c < c <= hi_c
            const long long own = p0 + tok[i];
            const int hi_c = ok[i] ? (int) min(own - k0, (long long) KEYS) : -1;
            const int lo_c = a.window > 0 ? (int) max(own - a.window - k0, -1LL) : -1;
            const int e0 = tok[i] - wt0 + KEYS - 1;
            float mx = -INFINITY;
#pragma unroll
            for (int n = 0; n < KEYS / 8; ++n)
#pragma unroll
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    const int c = 8 * n + 2 * tig + j % 2;
                    const float b = near ? bias[w][gid + 8 * i][e0 - c] : 0.0f;
                    float v = fmaf(s[n][j], sl, b) - sink[i];
                    if (!whole && (c > hi_c || c <= lo_c)) v = -INFINITY;
                    s[n][j] = v;
                    mx = fmaxf(mx, v);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float now = fmaxf(m[i], mx), base = now == -INFINITY ? 0.0f : now;
            const float f = exp2f(m[i] - base);
            m[i] = now;
            l[i] *= f;
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                o[n][2 * i] *= f;
                o[n][2 * i + 1] *= f;
            }
#pragma unroll
            for (int n = 0; n < KEYS / 8; ++n)
#pragma unroll
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    s[n][j] = __half2float(__float2half(exp2f(s[n][j] - base)));   // the weight the values get
                    l[i] += s[n][j];
                }
        }
#pragma unroll
        for (int k = 0; k < KEYS / 16; ++k) {   // keys 16 k..: the weights as an A fragment
            const uint32_t pa[4] = {ink_h2(s[2 * k][0], s[2 * k][1]), ink_h2(s[2 * k][2], s[2 * k][3]),
                                    ink_h2(s[2 * k + 1][0], s[2 * k + 1][1]), ink_h2(s[2 * k + 1][2], s[2 * k + 1][3])};
#pragma unroll
            for (int n = 0; n < 16; n += 2) {
                uint32_t b[4];   // keys 16 k.., values 8 n.. and 8 (n + 1)..
                ink_ldsm4(b, &vs[16 * k + lane % 8 + 8 * (lane / 8 % 2)][8 * n + 8 * (lane / 16)], true);
                ink_mma(o[n], pa, b);
                ink_mma(o[n + 1], pa, b + 2);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        if (!ok[i]) continue;
        const float inv = 1.0f / (l[i] + exp2f(0.0f - m[i]));   // the term of score 0, against the row's top
        const size_t at = th[i] * DIM + 2 * tig;
#pragma unroll
        for (int n = 0; n < 16; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                float v = o[n][2 * i + e] * inv;
                if (a.gate) v *= 1.0f / (1.0f + expf(-a.gate[at + 8 * n + e]));   // ink_mul_sigmoid_kernel's
                a.out[at + 8 * n + e] = v;
            }
    }
#else
    (void) a;   // ink_attend_batch() launches it only where the PTX has the steps
#endif
}

__device__ __forceinline__ float2 situ_scales(const ExpertScales& s, int row) {
    if (!s.gate && !s.up) return make_float2(1.0f, 1.0f);
    const int e = s.ids ? s.ids[row] : s.e[row];
    return make_float2(s.gate ? s.gate[e] : 1.0f, s.up ? s.up[e] : 1.0f);
}

// with q8: the rows also quantized, back to back as one vector (n a multiple of 32: a warp's values are a block)
__global__ void situ_rows_kernel(const float* gate_up, float* __restrict__ out, int n, float beta,
                                 float lb, ExpertScales s, QB* q8) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = gate_up + size_t(blockIdx.y) * 2 * n;
    const float2 sc = situ_scales(s, blockIdx.y);
    const float g = row[i] * sc.x;
    float u = row[n + i] * sc.y;
    if (lb > 0.0f) u = lb * tanhf(u / lb);
    const float h = beta * tanhf(g / beta) / (1.0f + expf(-g)) * u;
    out[size_t(blockIdx.y) * n + i] = h;
    if (q8) q8_put(q8, (int) (size_t(blockIdx.y) * n + i), h);
}

struct Control3 {
    const float* cur[3];
    const float* msg[3];
    const float* ch[3];
    const void* gw[3];   // F32 or BF16
    const void* fw[3];
    float* out[3];
    int count;
};

// a warp value i of the three controllers: 6 products of rank values (gemv_set_warp_kernel's), then the gates
template <typename W>
__global__ void ink_control3_gate_kernel(Control3 c, int n, int rank) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    const int i = int(blockIdx.x) * 8 + int(threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (i >= n) return;
    float g[3], f[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        if (j >= c.count) break;
        const W* gr = static_cast<const W*>(c.gw[j]) + size_t(i) * rank;
        const W* fr = static_cast<const W*>(c.fw[j]) + size_t(i) * rank;
        float ga = 0.0f, fa = 0.0f;
        for (int k = lane; k < rank; k += 32) {
            ga += wval(gr, k) * c.ch[j][k];
            fa += wval(fr, k) * c.ch[j][k];
        }
        g[j] = 0.0f + warp_sum(ga);
        f[j] = 0.0f + warp_sum(fa);
    }
    if (lane != 0) return;
    float cv[3];
#pragma unroll
    for (int j = 0; j < 3; ++j)
        if (j < c.count) cv[j] = c.cur[j][i];
#pragma unroll
    for (int j = 0; j < 3; ++j)
        if (j < c.count) c.out[j][i] = cv[j] - tanhf(f[j]) * cv[j] + tanhf(g[j]) * c.msg[j][i];
}

}  // namespace

struct NormMany {
    const float* x[3];
    const float* w[3];
    float* y[3];
    QB* q[3];
};

// rms_norm_many_kernel's row (blockIdx.y the input), each value also quantized
__global__ void ink_norm_many_q8_kernel(NormMany m, int n, float eps) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    __shared__ float part[32];
    const float* xr = m.x[blockIdx.y];
    const float* __restrict__ w = m.w[blockIdx.y];
    float* yr = m.y[blockIdx.y];
    if (n <= kNormHeld * int(blockDim.x)) {   // the values read once and kept for the norm
        float v[kNormHeld];
        float ss = 0.0f;
#pragma unroll
        for (int h = 0; h < kNormHeld; ++h) {
            const int i = threadIdx.x + h * blockDim.x;
            if (i < n) {
                v[h] = xr[i];
                ss += v[h] * v[h];
            }
        }
        ss = block_sum(ss, part);
        const float scale = 1.0f / sqrtf(ss / float(n) + eps);
#pragma unroll
        for (int h = 0; h < kNormHeld; ++h) {
            const int i = threadIdx.x + h * blockDim.x;
            if (i < n) {
                const float y = w ? v[h] * scale * w[i] : v[h] * scale;
                yr[i] = y;
                q8_put(m.q[blockIdx.y], i, y);
            }
        }
        return;
    }
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, part);
    const float scale = 1.0f / sqrtf(ss / float(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float y = w ? xr[i] * scale * w[i] : xr[i] * scale;
        yr[i] = y;
        q8_put(m.q[blockIdx.y], i, y);
    }
}

// ink_gate_kernel's values of one row, then add_rms_q8() of them
__global__ void ink_gate_norm_q8_kernel(const float* cur, const float* msg, const float* gain, const float* forget,
                                        float* out, const float* __restrict__ w,
                                        float* y, QB* q, int n, float eps) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    __shared__ float part[32];
    if (n <= kNormHeld * int(blockDim.x)) {   // the gated values kept for the norm
        float o[kNormHeld];
#pragma unroll
        for (int h = 0; h < kNormHeld; ++h) {
            const int i = threadIdx.x + h * blockDim.x;
            if (i < n) {
                const float c = cur[i];
                o[h] = c - tanhf(forget[i]) * c + tanhf(gain[i]) * msg[i];
                out[i] = o[h];
            }
        }
        add_rms_q8_held(o, w, y, q, n, eps, part);
        return;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float c = cur[i];
        out[i] = c - tanhf(forget[i]) * c + tanhf(gain[i]) * msg[i];
    }
    add_rms_q8(out, w, y, q, n, eps, part);
}

void ink_norm_many_q8(const float* const* x, const float* const* w, float* const* y, void* const* q8, int count, int n,
                      float eps, void* stream) {
    if (count < 1 || count > 3 || n % 32 != 0) throw std::invalid_argument("ink_norm_many_q8: 1 to 3 rows of 32k");
    NormMany m{};
    for (int j = 0; j < count; ++j) {
        m.x[j] = x[j];
        m.w[j] = w[j];
        m.y[j] = y[j];
        m.q[j] = (QB*) q8[j];
    }
    launch_pdl(ink_norm_many_q8_kernel, dim3(1, count), dim3(sum_threads(n)), 0, cs(stream), m, n, eps);
    launched("ink_norm_many_q8");
}

void ink_gate_norm_q8(const float* cur, const float* msg, const float* gain, const float* forget, float* out,
                      const float* w, float* y, void* q8, int n, float eps, void* stream) {
    const int threads = sum_threads(n);
    if (n % 32 != 0 || threads != std::min(1024, std::max(threads, n)))   // add_rms_norm_q8_1's block is this one
        throw std::invalid_argument("ink_gate_norm_q8: not add_rms_norm_q8_1's block");
    launch_pdl(ink_gate_norm_q8_kernel, dim3(1), dim3(threads), 0, cs(stream), cur, msg, gain, forget, out, w, y,
               (QB*) q8, n, eps);
    launched("ink_gate_norm_q8");
}

void ink_mix_norm(const InkStates& states, const float* query, int n_query, float* const* out, int n_embd, int rows,
                  const float* norm_w, float* norm_out, float eps, void* stream, void* q8, bool ffn) {
    if (states.n < 1 || states.n > InkStates::kMax || n_query < 1 || n_query > kQueries)
        throw std::invalid_argument("ink_mix: 1 to 6 states and 1 to 4 queries");
    if (rows <= 0) return;
    if (norm_out && sum_threads(n_embd) != kMixThreads)   // rms_norm()'s block must be this one
        throw std::invalid_argument("ink_mix_norm: a row of this width is normed by another block");
    MixArgs a{};
    for (int s = 0; s < states.n; ++s) a.s[s] = states.p[s];
    a.n = states.n;
    a.query = query;
    a.nq = n_query;
    for (int j = 0; j < n_query; ++j) a.out[j] = out[j];
    a.norm_w = norm_w;
    a.norm_out = norm_out;
    a.eps = eps;
    a.q8 = (QB*) q8;
    a.ffn = ffn;
    if ((q8 || ffn) && (rows != 1 || !norm_out || (ffn && (!q8 || !norm_w))))
        throw std::invalid_argument("ink_mix_norm: a quantized norm is one row's");
    launch_mix<MixOne>(a.n, a.nq, dim3(rows), cs(stream), a, n_embd);
    launched("ink_mix");
}

void ink_mix_many(const InkStates& states, const float* const* query, int n_query, float* const* out, int jobs,
                  int n_embd, void* stream) {
    if (states.n < 1 || states.n > InkStates::kMax || n_query < 1 || n_query > kQueries || jobs < 1 || jobs > kMixJobs)
        throw std::invalid_argument("ink_mix_many: 1 to 6 states, 1 to 4 queries, 1 to 12 jobs");
    MixMany m{};
    for (int k = 0; k < jobs; ++k) {
        MixArgs& a = m.j[k];
        for (int s = 0; s < states.n; ++s) a.s[s] = states.p[s];
        a.n = states.n;
        a.query = query[k];
        a.nq = n_query;
        for (int j = 0; j < n_query; ++j) a.out[j] = out[k * n_query + j];
    }
    launch_mix<MixJobs>(states.n, n_query, dim3(1, jobs), cs(stream), m, n_embd);
    launched("ink_mix_many");
}

void ink_mix(const InkStates& states, const float* query, int n_query, float* const* out, int n_embd, int rows,
             void* stream) {
    ink_mix_norm(states, query, n_query, out, n_embd, rows, nullptr, nullptr, 0.f, stream);
}

void ink_cat_norm_set(const float* const* cur, const float* const* msg, float* const* out, int count, int n_embd,
                      int rows, void* stream) {
    if (count < 1 || count > kSet) throw std::invalid_argument("ink_cat_norm_set: 1 to 3 pairs");
    if (rows <= 0) return;
    CatNormSet s{};
    for (int j = 0; j < count; ++j) {
        s.cur[j] = cur[j];
        s.msg[j] = msg[j];
        s.out[j] = out[j];
    }
    ink_cat_norm_kernel<<<dim3(rows, count), 256, 0, cs(stream)>>>(s, n_embd);
    launched("ink_cat_norm");
}

void ink_cat_norm(const float* cur, const float* msg, float* out, int n_embd, int rows, void* stream) {
    ink_cat_norm_set(&cur, &msg, &out, 1, n_embd, rows, stream);
}

void ink_set_fast_ctrl(bool on) { g_fast_ctrl = on ? 1 : 0; }

bool ink_ctrl_down(const float* const* cur, const float* const* msg, const void* const* down, float* const* ch,
                   int count, int n_embd, int rank, int rows, void* stream, bool bf16) {
    if (count < 1 || count > kSet) throw std::invalid_argument("ink_ctrl_down: 1 to 3 controllers");
    if (sum_threads(2 * n_embd) != 1024) return false;   // gemv_set_kernel's block is another
    if (rows <= 0) return true;
    CtrlDown d{};
    for (int j = 0; j < count; ++j) {
        d.cur[j] = cur[j];
        d.msg[j] = msg[j];
        d.w[j] = down[j];
        d.y[j] = ch[j];
    }
    if (fast_ctrl_on()) {
        if (bf16)
            launch_pdl(ink_ctrl_down_fast_kernel<uint16_t>, dim3(count * rank, rows), dim3(kFastDownThreads), 0,
                       cs(stream), d, n_embd, rank);
        else
            launch_pdl(ink_ctrl_down_fast_kernel<float>, dim3(count * rank, rows), dim3(kFastDownThreads), 0,
                       cs(stream), d, n_embd, rank);
        launched("ink_ctrl_down_fast");
        return true;
    }
    if (bf16)
        launch_pdl(ink_ctrl_down_kernel<uint16_t>, dim3(count * rank, rows), dim3(kDownThreads), 0, cs(stream), d,
                   n_embd, rank);
    else
        launch_pdl(ink_ctrl_down_kernel<float>, dim3(count * rank, rows), dim3(kDownThreads), 0, cs(stream), d, n_embd,
                   rank);
    launched("ink_ctrl_down");
    return true;
}

void ink_gate_set(const float* const* cur, const float* const* msg, const float* const* gain,
                  const float* const* forget, float* const* out, int count, size_t n, void* stream) {
    if (count < 1 || count > kSet) throw std::invalid_argument("ink_gate_set: 1 to 3 controllers");
    if (n == 0) return;
    GateSet s{};
    for (int j = 0; j < count; ++j) {
        s.cur[j] = cur[j];
        s.msg[j] = msg[j];
        s.gain[j] = gain[j];
        s.forget[j] = forget[j];
        s.out[j] = out[j];
    }
    s.count = count;
    ink_gate_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(s, n);
    launched("ink_gate");
}

void ink_gate(const float* cur, const float* msg, const float* gain, const float* forget, float* out, size_t n,
              void* stream) {
    ink_gate_set(&cur, &msg, &gain, &forget, &out, 1, n, stream);
}

void ink_conv(const float* x, float* state, const float* w, const float* dyn, float* out, int channels, int heads,
              int taps, int rows, void* stream, float* snap, size_t snap_stride, unsigned snap_bits) {
    if (taps < 2 || taps > 8) throw std::invalid_argument("ink_conv: 2 to 8 taps");
    if (rows <= 0) return;
    const float norm = 1.0f / std::sqrt((float) ((dyn ? 2 : 1) * taps + 1));
    if (rows <= kConvFew) {   // a window's: 384 launches fewer in Inkling 2 Mini's pass
        InkConv j;
        j.x = x, j.state = state, j.w = w, j.dyn = dyn, j.out = out, j.snap = snap_bits ? snap : nullptr;
        ink_conv_few(j, InkConv{}, channels, heads, taps, rows, stream, snap_stride, snap_bits);
        return;
    }
    ink_conv_kernel<<<grid((size_t) rows * channels, 256), 256, 0, cs(stream)>>>(x, state, w, dyn, out, channels,
                                                                                  heads, taps, rows, norm);
    launched("ink_conv");
    if (snap && snap_bits) {
        ink_conv_snaps_kernel<<<grid((size_t) channels, 256), 256, 0, cs(stream)>>>(x, state, snap, snap_stride,
                                                                                    snap_bits, channels, taps);
        launched("ink_conv_snaps");
    }
    ink_conv_state_kernel<<<grid((size_t) channels, 256), 256, 0, cs(stream)>>>(x, state, channels, taps, rows);
    launched("ink_conv_state");
}

void ink_conv_few(const InkConv& a, const InkConv& b, int channels, int heads, int taps, int rows, void* stream,
                  size_t snap_stride, unsigned snap_bits) {
    if (taps < 2 || taps > 8 || rows < 1 || rows > kConvFew)
        throw std::invalid_argument("ink_conv_few: 2 to 8 taps, 1 to 8 rows");
    InkConv ja = a, jb = b;   // ink_conv()'s norm, made on the host as it makes it
    ja.norm = 1.0f / std::sqrt((float) ((a.dyn ? 2 : 1) * taps + 1));
    jb.norm = 1.0f / std::sqrt((float) ((b.dyn ? 2 : 1) * taps + 1));
    ink_conv_few_kernel<<<dim3(grid((size_t) channels, 256), b.x ? 2 : 1), 256, 0, cs(stream)>>>(
        ja, jb, snap_stride, snap_bits, channels, heads, taps, rows);
    launched("ink_conv_few");
}

void ink_conv_add(const float* x, float* state, const float* w, float* out, float* acc, float* keep, float a, int mode,
                  int channels, int taps, void* stream) {
    if (taps < 2 || taps > 8) throw std::invalid_argument("ink_conv_add: 2 to 8 taps");
    const float norm = 1.0f / std::sqrt((float) (taps + 1));
    launch_pdl(ink_conv_add_kernel, dim3(grid((size_t) channels, 256)), dim3(256), 0, cs(stream), x, state, w, out,
               acc, keep, a, mode, channels, taps, norm);
    launched("ink_conv_add");
}

void ink_conv_add_rows(const float* x, const InkStateRows& state, const float* w, float* out, float* acc, float* keep,
                       float a, int mode, int channels, int taps, int rows, void* stream) {
    if (taps < 2 || taps > 8) throw std::invalid_argument("ink_conv_add_rows: 2 to 8 taps");
    if (rows < 1 || rows > kInkRows) throw std::invalid_argument("ink_conv_add_rows: 1 to kInkRows tokens");
    const float norm = 1.0f / std::sqrt((float) (taps + 1));   // ink_conv_add()'s
    launch_pdl(ink_conv_add_rows_kernel, dim3(grid((size_t) channels, 256), rows), dim3(256), 0, cs(stream), x, state,
               w, out, acc, keep, a, mode, channels, taps, norm);
    launched("ink_conv_add_rows");
}

void ink_moe_conv_add(const float* x, const float* w, const float* scale, const int* ids, int n_used, float* fo,
                      float* state, const float* cw, float* out, float* acc, float a, int channels, int taps,
                      void* stream) {
    if (taps < 2 || taps > 8) throw std::invalid_argument("ink_moe_conv_add: 2 to 8 taps");
    const float norm = 1.0f / std::sqrt((float) (taps + 1));
    launch_pdl(ink_moe_conv_add_kernel, dim3(grid((size_t) channels, 256)), dim3(256), 0, cs(stream), x, w, scale, ids,
               n_used, fo, state, cw, out, acc, a, channels, taps, norm);
    launched("ink_moe_conv_add");
}

#if defined(__HIPCC__)
// HIP: ink_clips() as the launches its contract names (ink_clip(), ink_head_rms() and fp16 stores of its jobs), and
// ink_qkv() as ink_conv() of K and V and then those; the fused kernels are not built there.  ink_clips_kernel and
// ink_qkv_kernel never finished on an RX 7900 XT (gfx1100) or an RX 9070 XT (gfx1201) under ROCm 7.2.4, the first
// prompt or token of Inkling 2 Mini with them, and the same code built with any change to ink_clips_kernel finished:
// the compiler's code for these two kernels, not their arithmetic.  The separate kernels give the same bits
// (poly_inkling2_test's clips and qkv cases compare them).
// A Q8_0 cache's rows (kvq8_store()'s), for clips_apart(): a block a row's head, its cell pos + row (at[0] + row
// with at), % ring when ring > 0.
__global__ void ink_store_q8_kernel(const float* x, uint8_t* cache, int n_kv, int dim, long long pos,
                                    const long long* at, int ring) {
    const int row = blockIdx.x / n_kv, h = blockIdx.x % n_kv;
    long long cell = (at ? at[0] : pos) + row;
    if (ring > 0) cell %= ring;
    kvq8_store(x + (size_t(row) * n_kv + h) * dim, cache, cell, n_kv, dim, h);
}

static void clips_apart(const InkClips& c, void* stream) {
    if (c.kv8) throw std::logic_error("ink_clips: no FP8 caches under HIP");
    for (int j = 0; j < 4; ++j) {
        if (c.heads[j] <= 0) continue;
        if (c.rms[j])
            ink_head_rms(c.x[j], c.dim[j], c.heads[j], c.rows, c.eps, stream);
        else
            ink_clip(c.x[j], c.scale[j], c.thr[j], c.dim[j], c.heads[j], c.rows, c.pos, c.alpha[j], c.floor[j], stream,
                     c.at);
        if (!c.cache[j]) continue;
        const int n = c.heads[j] * c.dim[j];
        if (c.kvq8) {
            if (c.dim[j] % 32) throw std::invalid_argument("ink_clips: Q8_0 caches need heads of whole 32-value blocks");
            ink_store_q8_kernel<<<c.rows * c.heads[j], 128, 0, cs(stream)>>>(c.x[j], (uint8_t*) c.cache[j], c.heads[j],
                                                                           c.dim[j], c.pos, c.at, c.ring);
        } else if (c.at) store_f16_at(c.x[j], c.cache[j], n, c.at, c.rows, c.ring, stream);
        else if (c.ring > 0) ink_store_ring(c.x[j], c.cache[j], n, c.pos, c.rows, c.ring, stream);
        else store_f16(c.x[j], c.cache[j], n, c.pos, c.rows, stream);
        launched("ink_clips' stores");
    }
}
#endif

void ink_qkv(const InkQkv& a, void* stream) {
    if (a.dim > 128 || a.n_rel > 128 || a.taps < 2 || a.taps > 8 || a.n_kv < 1)
        throw std::invalid_argument("ink_qkv: heads of up to 128 values, 2 to 8 taps");
#if defined(__HIPCC__)
    if (a.kv8) throw std::logic_error("ink_qkv: no FP8 caches under HIP");
    const int kv = a.n_kv * a.dim;
    ink_conv(a.kraw, a.kstate, a.kw, a.kdyn, a.k, kv, a.n_kv, a.taps, 1, stream);
    ink_conv(a.vraw, a.vstate, a.vw, a.vdyn, a.v, kv, a.n_kv, a.taps, 1, stream);
    InkClips c;
    float* xs[4] = {a.q, a.k, a.v, a.r};
    const float* sc[4] = {a.q_scale, nullptr, nullptr, a.r_scale};
    const float thr[4] = {a.qk_thr, a.qk_thr, 0.f, a.rel_thr}, al[4] = {a.alpha, 0.f, 0.f, a.alpha};
    const float fl[4] = {a.floor, 1.f, 1.f, a.floor};
    const int dims[4] = {a.dim, a.dim, a.dim, a.n_rel}, hs[4] = {a.n_head, a.n_kv, a.n_kv, a.n_head};
    for (int j = 0; j < 4; ++j) {
        c.x[j] = xs[j], c.scale[j] = sc[j], c.thr[j] = thr[j], c.alpha[j] = al[j], c.floor[j] = fl[j];
        c.dim[j] = dims[j], c.heads[j] = hs[j];
    }
    c.rms[2] = true;
    c.eps = a.eps;
    c.pos = a.pos;
    c.at = a.at;
    c.rows = 1;
    c.cache[1] = a.keys;
    c.cache[2] = a.vals;
    c.ring = a.ring;
    c.kv8 = a.kv8;
    c.kvq8 = a.kvq8;
    clips_apart(c, stream);
#else
    QkvArgs g{a, 1.0f / std::sqrt((float) ((a.kdyn ? 2 : 1) * a.taps + 1)),
              1.0f / std::sqrt((float) ((a.vdyn ? 2 : 1) * a.taps + 1))};
    launch_pdl(a.kvq8  ? ink_qkv_kernel<kKvQ8>
               : a.kv8 ? ink_qkv_kernel<kKvFp8>
                       : ink_qkv_kernel<kKvF16>, dim3(2 * a.n_head + 2 * a.n_kv), dim3(128), 0,
               cs(stream), g);
    launched("ink_qkv");
#endif
}

void ink_qkv_rows(const InkQkv& a, const InkQkvRows& r, int rows, void* stream) {
    if (a.dim > 128 || a.n_rel > 128 || a.taps < 2 || a.taps > 8 || a.n_kv < 1)
        throw std::invalid_argument("ink_qkv_rows: heads of up to 128 values, 2 to 8 taps");
    if (rows < 1 || rows > kInkRows || !a.at) throw std::invalid_argument("ink_qkv_rows: 1 to kInkRows tokens, positions on the card");
    QkvArgs g{a, 1.0f / std::sqrt((float) ((a.kdyn ? 2 : 1) * a.taps + 1)),   // ink_qkv()'s
              1.0f / std::sqrt((float) ((a.vdyn ? 2 : 1) * a.taps + 1))};
    launch_pdl(a.kv8 ? ink_qkv_rows_kernel<kKvFp8> : ink_qkv_rows_kernel<kKvF16>, dim3(2 * a.n_head + 2 * a.n_kv, rows),
               dim3(128), 0, cs(stream), g, r);
    launched("ink_qkv_rows");
}

void ink_clip(float* x, const float* scale, float thr, int dim, int heads, int rows, int64_t pos, float alpha,
              float floor, void* stream, const long long* at) {
    if (rows <= 0) return;
    ink_clip_kernel<<<rows * heads, 128, 0, cs(stream)>>>(x, scale, thr, dim, heads, (long long) pos, alpha, floor,
                                                          at);
    launched("ink_clip");
}

void ink_clips(const InkClips& c, void* stream) {
    if (c.rows <= 0) return;
#if defined(__HIPCC__)
    clips_apart(c, stream);
#else
    auto* k = c.kvq8 ? ink_clips_kernel<kKvQ8> : c.kv8 ? ink_clips_kernel<kKvFp8> : ink_clips_kernel<kKvF16>;
    k<<<c.rows * (c.heads[0] + c.heads[1] + c.heads[2] + c.heads[3]), 128, 0, cs(stream)>>>(c);
    launched("ink_clips");
#endif
}

void ink_head_rms(float* x, int dim, int heads, int rows, float eps, void* stream) {
    if (rows <= 0) return;
    ink_head_rms_kernel<<<rows * heads, 128, 0, cs(stream)>>>(x, dim, heads, eps);
    launched("ink_head_rms");
}

void ink_mul_sigmoid(float* x, const float* gate, size_t n, void* stream) {
    if (n == 0) return;
    ink_mul_sigmoid_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(x, gate, n);
    launched("ink_mul_sigmoid");
}

void ink_store_ring(const float* x, uint16_t* cache, int n, int64_t pos, int rows, int ring, void* stream) {
    if (rows <= 0) return;
    ink_store_ring_kernel<<<grid((size_t) rows * n, 256), 256, 0, cs(stream)>>>(x, (__half*) cache, n,
                                                                                (long long) pos, rows, ring);
    launched("ink_store_ring");
}

void ink_softcap(float* logits, int n_keep, int n_all, int rows, float cap, void* stream) {
    if (rows <= 0) return;
    ink_softcap_kernel<<<grid((size_t) rows * n_all, 256), 256, 0, cs(stream)>>>(logits, n_keep, n_all, rows, cap);
    launched("ink_softcap");
}

// The warps of a block of the value kernel: as many as have their sums in a block's shared memory.
static int value_warps(int n_head, int n_kv, int dim) {
    return (int) std::min<size_t>(32, kValueShared / ((size_t) (n_head / n_kv) * dim * sizeof(float)));
}

// ink_score_kernel() for a call's caches and heads
template <int LH> static auto score_kernel_lh(const InkAttn& a) {
    if (a.kvq8 && a.int8_dot) return ink_score_kernel<kKvQ8Dot, LH>;
    return a.kvq8 ? ink_score_kernel<kKvQ8, LH> : a.kv8 ? ink_score_kernel<kKvFp8, LH> : ink_score_kernel<kKvF16, LH>;
}
static auto score_kernel(const InkAttn& a) { return a.dim == 128 ? score_kernel_lh<4>(a) : score_kernel_lh<2>(a); }

// ink_score_kernel()'s steps for `rows` blocks of up to nk_max keys: kScoreSteps (a block 256 keys) when that still
// gives each multiprocessor kScoreFill blocks, else the most that do, down to 1 (a block 32 keys, the launch of a
// warp a step).  A short context gets the blocks to keep the card's multiprocessors busy (302 keys made 16 blocks
// of 256 for 68 multiprocessors), a long one a block's queries loaded for more keys.  A key's score is the same bits
// in every shape.  STRATA_INK_SCORE_STEPS=1, 2, 4 or 8 sets it (for poly_inkling2_attend_test).
constexpr int kScoreFill = 2;
static int score_steps(int rows, int nk_max) {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_INK_SCORE_STEPS");
        const int n = v ? std::atoi(v) : 0;
        return n == 1 || n == 2 || n == 4 || n == 8 ? n : 0;
    }();
    if (forced) return forced;
    int dev = 0, sms = 1;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
        return kScoreSteps;
    int steps = kScoreSteps;
    for (; steps > 1; steps /= 2) {
        const long long keys = (long long) kScoreWarps * kScoreKeys * steps;
        if ((long long) rows * ((nk_max + keys - 1) / keys) >= (long long) kScoreFill * sms) break;
    }
    return steps;
}

size_t ink_attend_floats(int n_head, int n_kv, int dim, int64_t keys) {
    // a head's scores, its softmax's norm, and its values' sums a block of the value kernel
    const int64_t block = (int64_t) value_warps(n_head, n_kv, dim) * kValueKeys;
    return (size_t) n_head * ((size_t) keys + 1 + (size_t) ((keys + block - 1) / block) * dim);
}

int ink_attend_parts(int n_head, int n_kv, int dim, int window, int64_t last) {
    const int nk_max = (int) (window > 0 ? std::min<int64_t>(window, last) : last);
    const int warps = value_warps(n_head, n_kv, dim);
    return (nk_max + warps * kValueKeys - 1) / (warps * kValueKeys);
}

// ink_attend(), or with rows ink_attend_rows()'s tokens over their own caches
static void attend(const InkAttn& a, const InkAttnRows* rows, void* stream) {
    if (a.rows <= 0) return;
    if (a.n_kv < 1 || a.n_head % a.n_kv != 0 || a.n_head / a.n_kv > kMaxPer || (a.dim != 64 && a.dim != 128) ||
        a.n_rel < 1 || a.n_rel > 32)
        throw std::invalid_argument("ink_attend: heads of 64 or 128 values, up to 4 on a key head, up to 32 relative "
                                    "values");
    // the most keys a query of the call reads: with the positions on the card, what the launches are sized for
    const int64_t last = a.at ? a.room : a.pos + a.rows;
    if (a.at && a.room < 1) throw std::invalid_argument("ink_attend: positions on the card need the rows to size for");
    const int nk_max = (int) (a.window > 0 ? std::min<int64_t>(a.window, last) : last);
    const int group = a.n_head / a.n_kv, warps = value_warps(a.n_head, a.n_kv, a.dim);
    const int n_parts = ink_attend_parts(a.n_head, a.n_kv, a.dim, a.window, last);
    const int score_blocks = (nk_max + kScoreWarps * kScoreKeys - 1) / (kScoreWarps * kScoreKeys);
    const size_t per = ink_attend_floats(a.n_head, a.n_kv, a.dim, nk_max);
    if (a.score_n < per) throw std::logic_error("ink_attend: the scores of one token do not fit the scratch");
    const int chunk = (int) std::min<size_t>(a.score_n / per, (size_t) a.rows);
    float* norm = a.score + (size_t) chunk * a.n_head * nk_max;
    float* parts = norm + (size_t) chunk * a.n_head;
    const size_t value_shared = (size_t) warps * group * a.dim * sizeof(float);
    for (int t0 = 0; t0 < a.rows; t0 += chunk) {
        const int m = std::min(chunk, a.rows - t0);
        if (rows)
            launch_pdl(a.kv8 ? ink_score_rows_kernel<kKvFp8> : ink_score_rows_kernel<kKvF16>,
                       dim3(m * a.n_kv, score_blocks), dim3(kScoreWarps * 32), 0, cs(stream), a, t0, nk_max, *rows);
        else {
            const int steps = score_steps(m * a.n_kv, nk_max), keys = kScoreWarps * kScoreKeys * steps;
            launch_pdl(score_kernel(a), dim3(m * a.n_kv, (nk_max + keys - 1) / keys), dim3(kScoreWarps * 32), 0,
                       cs(stream), a, t0, nk_max, steps);
        }
        launched("ink_score");
        launch_pdl(ink_softmax_kernel, dim3(m * a.n_head), dim3(kAttnThreads), 0, cs(stream), a, t0, nk_max, norm);
        launched("ink_softmax");
        if (rows)
            launch_pdl(a.kv8 ? ink_value_rows_kernel<kKvFp8> : ink_value_rows_kernel<kKvF16>, dim3(m * a.n_kv, n_parts),
                       dim3(warps * 32), value_shared, cs(stream), a, t0, nk_max, parts, n_parts, *rows);
        else
            launch_pdl(a.kvq8  ? ink_value_kernel<kKvQ8>
                       : a.kv8 ? ink_value_kernel<kKvFp8>
                               : ink_value_kernel<kKvF16>, dim3(m * a.n_kv, n_parts),
                       dim3(warps * 32), value_shared, cs(stream), a, t0, nk_max, parts, n_parts);
        launched("ink_value");
        if (rows)
            launch_pdl(ink_join_rows_kernel, dim3(m * a.n_head), dim3(128), 0, cs(stream), a, t0, parts, n_parts, norm,
                       *rows);
        else
            launch_pdl(ink_join_kernel, dim3(m * a.n_head), dim3(128), 0, cs(stream), a, t0, parts, n_parts, norm);
        launched("ink_join");
    }
}

#if !defined(__HIPCC__)
// ink_attend_batch() on Turing, fp16 caches: ink_mma75_kernel, or ink_flash75_kernel with STRATA_INK_MMA75=0.  On
// upstream's accuracy check (3 prompts of 302 to 8,192 tokens, 64 tokens, 3 files) ink_mma75_kernel's most likely
// token is ink_flash75_kernel's greedy answer at 98.3% of the steps and attend_batch()'s at 98.4%.
static bool flash75(const InkAttn& a, int per, void* stream) {
    static const bool mma = [] {
        const char* v = std::getenv("STRATA_INK_MMA75");
        return !(v && v[0] == '0');
    }();
    if (mma) {
        const auto launch = [&](auto kernel, int g) {
            kernel<<<dim3(grid((size_t) a.rows, mma75_tokens(g)), a.n_kv), kMma75Warps * 32, 0, cs(stream)>>>(a);
        };
        switch (per) {
            case 1: launch(ink_mma75_kernel<1>, 1); break;
            case 2: launch(ink_mma75_kernel<2>, 2); break;
            case 3: launch(ink_mma75_kernel<3>, 3); break;
            default: launch(ink_mma75_kernel<4>, 4); break;
        }
        launched("ink_mma75");
        return true;
    }
    const auto launch = [&](auto kernel, size_t smem, int g) {
        const cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        if (e != cudaSuccess) throw std::runtime_error(std::string("ink_attend_batch: shared memory: ") + cudaGetErrorString(e));
        kernel<<<dim3(grid((size_t) a.rows, kFlash75Tokens), a.n_kv), kFlash75Tokens / 16 * g * 32, smem, cs(stream)>>>(a);
    };
    switch (per) {
        case 1: launch(ink_flash75_kernel<1>, flash75_smem<1>(), 1); break;
        case 2: launch(ink_flash75_kernel<2>, flash75_smem<2>(), 2); break;
        case 3: launch(ink_flash75_kernel<3>, flash75_smem<3>(), 3); break;
        default: launch(ink_flash75_kernel<4>, flash75_smem<4>(), 4); break;
    }
    launched("ink_flash75");
    return true;
}
#endif

void ink_attend(const InkAttn& a, void* stream) { attend(a, nullptr, stream); }

void ink_attend_rows(const InkAttn& a, const InkAttnRows& c, void* stream) {
    if (a.rows > kInkRows || !a.at || a.q8) throw std::invalid_argument("ink_attend_rows: 1 to kInkRows tokens, positions on the card, no q8");
    for (int t = 0; t < a.rows; ++t)
        if (c.parts[t] < 1 || c.parts[t] > ink_attend_parts(a.n_head, a.n_kv, a.dim, a.window, a.room))
            throw std::invalid_argument("ink_attend_rows: a token's parts are more than the launch's");
    attend(a, &c, stream);
}

#if defined(__HIPCC__)
// Whether the current card is gfx12, whose instructions ink_wmma_kernel has.  The arch macros are the device pass's
// alone, so the host asks the card, once a device.
static bool ink_wmma_card() {
    static std::atomic<int> cards[16] = {};   // a device's: 1 gfx12, 2 not, 0 not yet asked
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) {
        (void) cudaGetLastError();
        return false;
    }
    int card = cards[dev].load(std::memory_order_relaxed);
    if (!card) {
        cudaDeviceProp p{};
        const bool gfx12 = cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
                           strata::kernels::gfx_arch_is_gfx12_wmma(p.gcnArchName);
        (void) cudaGetLastError();
        card = gfx12 ? 1 : 2;
        cards[dev].store(card, std::memory_order_relaxed);
    }
    return card == 1;
}
#endif

bool ink_attend_batch(const InkAttn& a, void* stream) {
#if defined(__HIPCC__)
    // RDNA4's matrix cores (ink_wmma_kernel): fp16 caches, Inkling 2 Mini's heads (128 values, 3 on a key head, 16
    // relative values), positions on the host, no q8 output and no Q8_0 cache (PR 214's rows go to attend_batch()).  Elsewhere, or with STRATA_INK_FLASH=0, attend_batch()'s path.
    if (a.rows <= 0) return true;
    static const bool off = [] {
        const char* v = std::getenv("STRATA_INK_FLASH");
        return v && v[0] == '0';
    }();
    if (off || a.kv8 || a.kvq8 || a.dim != kWmDim || a.n_rel != kWmRel || a.n_kv < 1 || a.n_head != kWmPer * a.n_kv || a.at || a.q8 ||
        !ink_wmma_card())
        return false;
    ink_wmma_kernel<<<dim3(grid((size_t) a.rows, kWmTokens), a.n_kv), kWmWarps * WARP, 0, cs(stream)>>>(a);
    launched("ink_wmma");
    return true;
#else
    return ink_attend_batch_rows(a, stream, 0);
#endif
}

#if !defined(__HIPCC__)
namespace {

int flash_ptx() {
    static const int ptx = [] {
        cudaFuncAttributes f{};
        return cudaFuncGetAttributes(&f, ink_flash_kernel<1, false>) == cudaSuccess ? f.ptxVersion : 0;
    }();
    return ptx;
}

bool flash_lean() {   // ink_flash_kernel's LEAN on sm_80 (A100 class) only: the card it was faster on
    static const bool lean = [] {
        int dev = 0, major = 0, minor = 0;
        return cudaGetDevice(&dev) == cudaSuccess &&
               cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
               cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess && major == 8 &&
               minor == 0;
    }();
    return lean;
}

bool flash_fast() {   // STRATA_FAST_ATTN=1: ink_fast_kernel for full layers (opt-in, other bits)
    static const bool fast = [] {
        const char* v = std::getenv("STRATA_FAST_ATTN");
        const bool on = v && v[0] == '1';
        if (on) std::fprintf(stderr, "strata: STRATA_FAST_ATTN=1: Inkling 2's full layers read prompts on ink_fast_kernel (other bits)\n");
        return on;
    }();
    return fast;
}

// the query heads on a key head when ink_attend_batch() takes the call at all, else 0
int flash_per(const InkAttn& a) {
    static const bool off = [] {   // STRATA_INK_FLASH=0: attend_batch()'s path
        const char* v = std::getenv("STRATA_INK_FLASH");
        return v && v[0] == '0';
    }();
    const int per = a.n_kv > 0 && a.n_head % a.n_kv == 0 ? a.n_head / a.n_kv : 0;
    if (off || a.dim != 128 || a.n_rel != kFlashRel || per < 1 || per > kMaxPer || a.at || a.q8) return 0;
    return per;
}

constexpr int kFlashSizes[] = {64, 48, 32};   // ink_flash_kernel's TOKENS without LEAN, main's first

int flash_size_index(int tokens) { return tokens == 64 ? 0 : tokens == 48 ? 1 : 2; }

template <int G, int T> void* flash_fn() { return reinterpret_cast<void*>(ink_flash_kernel<G, false, T>); }

void* flash_kernel(int g, int tokens) {   // ink_flash_kernel<g, false, tokens>
    static void* const fns[4][3] = {{flash_fn<1, 64>(), flash_fn<1, 48>(), flash_fn<1, 32>()},
                                    {flash_fn<2, 64>(), flash_fn<2, 48>(), flash_fn<2, 32>()},
                                    {flash_fn<3, 64>(), flash_fn<3, 48>(), flash_fn<3, 32>()},
                                    {flash_fn<4, 64>(), flash_fn<4, 48>(), flash_fn<4, 32>()}};
    return fns[g - 1][flash_size_index(tokens)];
}

// The kernel's shared memory set (once a kernel and device), and its blocks a multiprocessor
int flash_resident(int g, int tokens) {
    constexpr int kDevs = 16;
    static std::atomic<int> cache[kDevs][4][3] = {};   // 0: not asked yet; else 1 + blocks
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
    std::atomic<int>* c = dev >= 0 && dev < kDevs ? &cache[dev][g - 1][flash_size_index(tokens)] : nullptr;
    if (c && c->load(std::memory_order_relaxed) > 0) return c->load(std::memory_order_relaxed) - 1;
    void* fn = flash_kernel(g, tokens);
    const size_t smem = flash_smem(g, false, tokens);
    const cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    if (e != cudaSuccess) throw std::runtime_error(std::string("ink_attend_batch: shared memory: ") + cudaGetErrorString(e));
    int n = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, tokens / 16 * g * 32, smem) != cudaSuccess) n = 0;
    if (c) c->store(1 + n, std::memory_order_relaxed);
    return n;
}

int flash_sms() {
    int dev = 0, n = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    return cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess ? n : 0;
}

// The rows a block for a call ink_flash_kernel without LEAN takes: STRATA_INK_FLASH_TOKENS=64/48/32 fixes them (64:
// main's shape); else the size whose launch ends soonest by its waves: blocks over the multiprocessors' resident
// slots, rounded up, times a block's time against one of 64 rows (kFlashCost, measured on an RTX 5080 Laptop at 2K to
// 32K keys: a block of 48 takes 0.9 of one of 64, as nvcc gives both 168 registers, and one of 32 0.65).  Another size
// than 64 is taken only when that is at least kFlashGain less, so a card whose last wave is full or nearly so keeps
// 64.  Inkling 2 Mini's 1,024-row calls (128 blocks of 64, one a multiprocessor) move to 48 on 59 to 63
// multiprocessors (a third wave of 64 nearly empty), 88 to 127 and 176 to 199, and keep 64 elsewhere.
int flash_pick(const InkAttn& a, int per) {
    static const int fixed = [] {
        const char* v = std::getenv("STRATA_INK_FLASH_TOKENS");
        const int t = v ? std::atoi(v) : 0;
        return t == 64 || t == 48 || t == 32 ? t : 0;
    }();
    if (fixed) return fixed;
    constexpr double kFlashCost[] = {1.0, 0.9, 0.65}, kFlashGain = 0.05;   // kFlashSizes' blocks
    const int sms = flash_sms();
    if (sms < 1 || a.rows <= 0) return 64;
    double best = 0.0, main_cost = 0.0;
    int pick = 64;
    for (int i = 0; i < 3; ++i) {
        const int t = kFlashSizes[i], res = flash_resident(per, t);
        if (res < 1) continue;
        const long long blocks = (long long) grid((size_t) a.rows, t) * a.n_kv, slots = (long long) sms * res;
        const double cost = (double) ((blocks + slots - 1) / slots) * res * kFlashCost[i];
        if (t == 64) {
            main_cost = best = cost;
        } else if (main_cost > 0.0 && cost < best && cost <= main_cost * (1.0 - kFlashGain)) {
            best = cost;
            pick = t;
        }
    }
    return pick;
}

// whether the call goes to ink_flash_kernel without LEAN (not Turing's kernel, LEAN's or ink_fast_kernel)
bool flash_sized(const InkAttn& a, int per) {
    return per > 0 && flash_ptx() >= 80 && !flash_lean() && !(flash_fast() && a.window <= 0 && !a.kvq8);
}

}  // namespace

int ink_flash_tokens(const InkAttn& a) {
    if (a.rows <= 0) return 0;
    const int per = flash_per(a);
    return flash_sized(a, per) ? flash_pick(a, per) : 0;
}
#else
int ink_flash_tokens(const InkAttn&) { return 0; }
#endif

bool ink_attend_batch_rows(const InkAttn& a, void* stream, int tokens) {
#if defined(__HIPCC__)
    if (tokens == 64 || tokens == 48 || tokens == 32) return false;   // no ink_flash_kernel on HIP
    return ink_attend_batch(a, stream);
#else
    if (a.rows <= 0) return true;
    const int per = flash_per(a);
    if (per == 0) return false;
    const int ptx = flash_ptx();
    // Turing: ink_flash75_kernel reads fp16 caches only (FP8 and Q8_0 rows go to attend_batch())
    if (ptx < 80) return ptx >= 75 && !a.kv8 && !a.kvq8 && flash75(a, per, stream);
    const bool lean = flash_lean();
    const auto launch = [&](auto kernel, int g, bool ln) {
        const int t = flash_tokens(ln);
        const size_t smem = flash_smem(g, ln, t);
        const cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        if (e != cudaSuccess) throw std::runtime_error(std::string("ink_attend_batch: shared memory: ") + cudaGetErrorString(e));
        kernel<<<dim3(grid((size_t) a.rows, t), a.n_kv), t / 16 * g * 32, smem, cs(stream)>>>(a);
    };
    if (flash_fast() && a.window <= 0 && !a.kvq8) {   // ink_fast_kernel reads fp16 and FP8 caches
        // HACC where the tensor cores sum fp16 products into floats at half rate: all but sm_80, sm_90 and sm_100,
        // whose whitepapers give the same dense FP16 TFLOPS with FP32 accumulation as with FP16
        static const bool hacc = [] {
            int dev = 0, major = 0, minor = 0;
            if (cudaGetDevice(&dev) != cudaSuccess ||
                cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
                cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess)
                return true;
            return !((major == 8 && minor == 0) || major == 9 || major == 10);
        }();
        const auto run = [&](auto kernel, int g) {
            kernel<<<dim3(grid((size_t) a.rows, fast_tokens(g)), a.n_kv), kFastThreads, 0, cs(stream)>>>(a);
        };
        switch (per * 2 + hacc) {
            case 2: run(ink_fast_kernel<1, false>, 1); break;
            case 3: run(ink_fast_kernel<1, true>, 1); break;
            case 4: run(ink_fast_kernel<2, false>, 2); break;
            case 5: run(ink_fast_kernel<2, true>, 2); break;
            case 6: run(ink_fast_kernel<3, false>, 3); break;
            case 7: run(ink_fast_kernel<3, true>, 3); break;
            case 8: run(ink_fast_kernel<4, false>, 4); break;
            default: run(ink_fast_kernel<4, true>, 4); break;
        }
        launched("ink_fast");
        return true;
    }
    if (lean) {
        switch (per) {
            case 1: launch(ink_flash_kernel<1, true>, 1, true); break;
            case 2: launch(ink_flash_kernel<2, true>, 2, true); break;
            case 3: launch(ink_flash_kernel<3, true>, 3, true); break;
            default: launch(ink_flash_kernel<4, true>, 4, true); break;
        }
        launched("ink_flash");
        return true;
    }
    // without LEAN: the rows a block by the card's last wave (flash_pick()), or `tokens`; the same bits each way
    if (tokens != 64 && tokens != 48 && tokens != 32) tokens = flash_pick(a, per);
    flash_resident(per, tokens);   // sets the kernel's shared memory
    InkAttn arg = a;
    void* args[] = {&arg};
    (void) cudaLaunchKernel(flash_kernel(per, tokens), dim3(grid((size_t) a.rows, tokens), a.n_kv),
                            dim3(tokens / 16 * per * 32), args, flash_smem(per, false, tokens),
                            cs(stream));   // as a <<<>>> launch: an error shows at the next check
    launched("ink_flash");
    return true;
#endif
}

bool ink_kv8_ok() {
#if defined(__HIPCC__)
    return false;
#else
    static const bool ok = [] {
        const char* v = std::getenv("STRATA_INK_FLASH");
        cudaFuncAttributes f{};
        return !(v && v[0] == '0') && cudaFuncGetAttributes(&f, ink_flash_kernel<1, false>) == cudaSuccess && f.ptxVersion >= 80;
    }();
    return ok;
#endif
}

void ink_control3_gate(const float* const* cur, const float* const* msg, const float* const* ch,
                       const void* const* gain_w, const void* const* forget_w, float* const* out, int n, int rank,
                       void* stream, int count, bool bf16) {
    if (rank >= 128) throw std::invalid_argument("ink_control3_gate: gemv_set() takes products this wide otherwise");
    if (count < 1 || count > 3) throw std::invalid_argument("ink_control3_gate: 1 to 3 controllers");
    Control3 c{};
    c.count = count;
    for (int j = 0; j < count; ++j) {
        c.cur[j] = cur[j];
        c.msg[j] = msg[j];
        c.ch[j] = ch[j];
        c.gw[j] = gain_w[j];
        c.fw[j] = forget_w[j];
        c.out[j] = out[j];
    }
    if (bf16)
        launch_pdl(ink_control3_gate_kernel<uint16_t>, dim3((n + 7) / 8), dim3(256), 0, cs(stream), c, n, rank);
    else
        launch_pdl(ink_control3_gate_kernel<float>, dim3((n + 7) / 8), dim3(256), 0, cs(stream), c, n, rank);
    launched("ink_control3_gate");
}

void situ_rows(const float* gate_up, float* out, int n, int rows, float beta, float lb, void* stream, void* q8,
               const ExpertScales* scales) {
    if (rows <= 0) return;
    if (q8 && n % 32 != 0) throw std::invalid_argument("situ_rows: quantized rows of a multiple of 32");
    launch_pdl(situ_rows_kernel, dim3(grid((size_t) n, 256), rows), dim3(256), 0, cs(stream), gate_up, out, n, beta, lb,
               scales ? *scales : ExpertScales{}, (QB*) q8);
    launched("situ_rows");
}

// ---- the DSpark drafter's kernels (Inkling2::guess_block(), ds_commit()) ----
//
// The drafter's forward (the features' mixes, the context keys and values, the block of the anchor and mask rows, the
// Markov head's recurrence and its fold into the output matrix) follows Basalt's src/models/inkling2/dspark.cpp and
// kernels/dspark.cu (github.com/jesdga95/basalt), which follow vLLM's Inkling 2 DSpark draft (Apache-2.0); the kernels
// here are this engine's own.  Basalt is under the MIT license:
// MIT License
// Copyright (c) 2026 Jesús Gulfo
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

namespace {

constexpr int kDsHead = 128;   // a drafter head's values

__device__ __forceinline__ float ds_bf16(uint16_t v) { return __uint_as_float(uint32_t(v) << 16); }

__device__ __forceinline__ long long ds_pos(const DsPos& p, int t) { return p.at ? p.at[2 * t] : p.pos0 + t; }

// row t (blockIdx.x), mix j (blockIdx.y): v = sum_k alpha[j][k] f_k, then rms(v) * norm[j]; mixes 0..2 to out, mix 3 to
// the ring row of the row's position
__global__ void ds_mix_kernel(DsMix m, int n, int rows, float eps) {
    __shared__ float part[32];
    extern __shared__ float v[];
    const int t = blockIdx.x, j = blockIdx.y;
    const float* a = m.alpha + j * 4;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float x = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) x += a[k] * m.tap[(size_t) k * m.tap_stride + (size_t) t * n + i];
        v[i] = x;
        ss += x * x;
    }
    ss = block_sum(ss, part);
    const float scale = 1.0f / sqrtf(ss / (float) n + eps);
    float* out = j < 3 ? m.out + ((size_t) j * rows + t) * n
                       : m.ring + (size_t) (ds_pos(m.pos, t) % m.ring_rows) * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = v[i] * scale * m.norm[j][i];
}

// row t (blockIdx.x), head h (blockIdx.y), a thread a value: x's head RMS-normed with w, then rotated at the row's
// position (NeoX halves, base theta); into the fp16 cache's cell of the position, or back into x.  With v, v's head
// goes to its cache too.
__global__ void ds_rope_kernel(float* x, int heads, const float* __restrict__ w, float eps, float theta, DsPos p,
                               __half* cache, long long cap, const float* v, __half* vcache, long long* tags) {
    __shared__ float part[kDsHead / 32];
    __shared__ float y[kDsHead];
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    float* row = x + ((size_t) t * heads + h) * kDsHead;
    const float xi = row[i];
    const float scale = 1.0f / sqrtf(block_sum(xi * xi, part) / (float) kDsHead + eps);
    y[i] = xi * scale * w[i];
    __syncthreads();
    const long long pos = ds_pos(p, t);
    const int half = kDsHead / 2, k = i % half;
    const float freq = powf(theta, -2.0f * (float) k / (float) kDsHead), ang = (float) pos * freq;
    const float c = cosf(ang), s = sinf(ang);
    const float out = i < half ? y[i] * c - y[i + half] * s : y[i - half] * s + y[i] * c;
    if (!cache) {
        row[i] = out;
        return;
    }
    const size_t cell = (size_t) (pos % cap) * heads * kDsHead + (size_t) h * kDsHead + i;
    cache[cell] = __float2half(out);
    if (v) vcache[cell] = __float2half(v[((size_t) t * heads + h) * kDsHead + i]);
    if (tags && h == 0 && i == 0) tags[pos % cap] = pos;
}

// The block's attention, the keys split among blocks so that a long context is read at once: block (g, s) takes kv
// head g's query rows (block row t's head g * grp + k as row t * grp + k) over split s of the keys [lo, hi), a warp a
// key, and softmaxes them online; ds_join_kernel joins the splits.  lo and hi follow the block's first position on
// the card (a recorded block takes any): a window layer's window before the block and the block, a full layer every
// position.  The split: the keys over kDsSplits, a whole number of tiles each.
constexpr int kDsRows = 24;    // a kv head's query rows at most: 8 block rows of 3 heads
constexpr int kDsTile = 32;    // keys a tile, read into shared memory at once
static_assert(kDsTile == 32, "ds_part_kernel scores a tile's keys a lane each");
constexpr int kDsSplits = 64;  // splits at most
constexpr int kDsPart = kDsHead + 2;   // a row's part: its values' weighted sum, its max, its sum

__device__ __forceinline__ void ds_span(const DsPos& p, int nb, int window, long long& lo, long long& nk, int& per) {
    const long long p0 = ds_pos(p, 0);
    lo = window > 0 ? max(0LL, p0 - window) : 0LL;
    nk = p0 + nb - lo;
    per = (int) (((nk + kDsSplits - 1) / kDsSplits + kDsTile - 1) / kDsTile * kDsTile);
}

__global__ void __launch_bounds__(256) ds_part_kernel(const float* q, const __half* keys, const __half* vals,
                                                      const long long* tags, long long cap, DsPos p, int nb,
                                                      int window, int heads, int kv_heads, float scale, float* part) {
    __shared__ __align__(16) float qs[kDsRows][kDsHead];
    __shared__ half2 ks[kDsTile][kDsHead / 2 + 1];   // a key a row, a word longer: a warp's lanes read 32 rows
    __shared__ __align__(16) __half vs[kDsTile][kDsHead];
    __shared__ float sc[kDsRows][kDsTile];
    __shared__ float mrun[kDsRows], lrun[kDsRows], corr[kDsRows];
    __shared__ bool held[kDsTile];   // the key's cell holds its position (a ring after going back holds later ones)
    const int g = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int grp = heads / kv_heads, R = nb * grp;
    long long lo, nk;
    int per;
    ds_span(p, nb, window, lo, nk, per);
    const long long k0 = (long long) s * per;
    if (k0 >= nk) return;
    const long long k1 = min(nk, k0 + per);
    for (int i = tid; i < R * kDsHead; i += blockDim.x) {
        const int r = i / kDsHead, d = i % kDsHead;
        qs[r][d] = q[((size_t) (r / grp) * heads + g * grp + r % grp) * kDsHead + d] * scale;
    }
    if (tid < R) mrun[tid] = -INFINITY, lrun[tid] = 0.0f;
    const int d = tid % kDsHead, r0 = tid / kDsHead;   // the values: d of rows r0, r0 + 2, ...
    float acc[kDsRows / 2];
#pragma unroll
    for (int i = 0; i < kDsRows / 2; ++i) acc[i] = 0.0f;
    const size_t at_g = (size_t) g * kDsHead, cell = (size_t) kv_heads * kDsHead;
    __syncthreads();
    for (long long t0 = k0; t0 < k1; t0 += kDsTile) {
        const int nt = (int) min((long long) kDsTile, k1 - t0);
        for (int i = tid; i < nt * (kDsHead / 8); i += blockDim.x) {   // the tile's keys and values, 8 halves a load
            const int j = i / (kDsHead / 8), c = i % (kDsHead / 8) * 8;
            const size_t at = (size_t) ((lo + t0 + j) % cap) * cell + at_g + c;
            const uint4 kw = *(const uint4*) (keys + at);
            ks[j][c / 2] = *(const half2*) &kw.x, ks[j][c / 2 + 1] = *(const half2*) &kw.y;
            ks[j][c / 2 + 2] = *(const half2*) &kw.z, ks[j][c / 2 + 3] = *(const half2*) &kw.w;
            *(uint4*) &vs[j][c] = *(const uint4*) (vals + at);
        }
        if (tid < nt) held[tid] = !tags || tags[(lo + t0 + tid) % cap] == lo + t0 + tid;
        __syncthreads();
        for (int r = warp; r < R && lane < nt; r += 8) {   // a warp a query row, a lane a key
            float x = 0.0f;
#pragma unroll 16
            for (int c = 0; c < kDsHead / 2; ++c) {
                const float2 k = __half22float2(ks[lane][c]), qv = *(const float2*) &qs[r][2 * c];
                x += qv.x * k.x + qv.y * k.y;
            }
            sc[r][lane] = held[lane] ? x : -INFINITY;
        }
        __syncthreads();
        if (tid < R) {
            float m = mrun[tid];
            for (int j = 0; j < nt; ++j) m = fmaxf(m, sc[tid][j]);
            corr[tid] = m == -INFINITY ? 1.0f : expf(mrun[tid] - m);
            mrun[tid] = m;
        }
        __syncthreads();
        for (int i = tid; i < R * kDsTile; i += blockDim.x) {
            const int r = i / kDsTile, j = i % kDsTile;
            if (j < nt) sc[r][j] = sc[r][j] == -INFINITY ? 0.0f : expf(sc[r][j] - mrun[r]);
        }
        __syncthreads();
        if (tid < R) {
            float l = 0.0f;
            for (int j = 0; j < nt; ++j) l += sc[tid][j];
            lrun[tid] = lrun[tid] * corr[tid] + l;
        }
#pragma unroll
        for (int i = 0; i < kDsRows / 2; ++i)
            if (r0 + 2 * i < R) acc[i] *= corr[r0 + 2 * i];
        for (int j = 0; j < nt; ++j) {
            const float v = __half2float(vs[j][d]);
#pragma unroll
            for (int i = 0; i < kDsRows / 2; ++i)
                if (r0 + 2 * i < R) acc[i] += sc[r0 + 2 * i][j] * v;
        }
        __syncthreads();
    }
    float* out = part + ((size_t) g * kDsSplits + s) * kDsRows * kDsPart;
#pragma unroll
    for (int i = 0; i < kDsRows / 2; ++i)
        if (r0 + 2 * i < R) out[(r0 + 2 * i) * kDsPart + d] = acc[i];
    if (tid < R) out[tid * kDsPart + kDsHead] = mrun[tid], out[tid * kDsPart + kDsHead + 1] = lrun[tid];
}

// query row t (blockIdx.x), head h (blockIdx.y), a thread a value: the splits' parts joined
__global__ void ds_join_kernel(const float* part, DsPos p, int nb, int window, int heads, int kv_heads, float* out) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x, grp = heads / kv_heads;
    long long lo, nk;
    int per;
    ds_span(p, nb, window, lo, nk, per);
    const int ns = (int) ((nk + per - 1) / per);
    const size_t step = (size_t) kDsRows * kDsPart;
    const float* in = part + (size_t) (h / grp) * kDsSplits * step + (size_t) (t * grp + h % grp) * kDsPart;
    float m = -INFINITY;
    for (int s = 0; s < ns; ++s) m = fmaxf(m, in[s * step + kDsHead]);
    float l = 0.0f, a = 0.0f;
    for (int s = 0; s < ns; ++s) {
        const float ms = in[s * step + kDsHead];
        const float w = ms == -INFINITY ? 0.0f : expf(ms - m);   // a split of cells that hold other positions
        l += w * in[s * step + kDsHead + 1];
        a += w * in[s * step + d];
    }
    out[((size_t) t * heads + h) * kDsHead + d] = a / l;
}

__global__ void ds_swiglu_kernel(float* g, const float* __restrict__ u, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}

// The block's inputs beside its first row: rows 1 .. nb - 1 of x the mask; the Markov head's start (its state zeros,
// the anchor *token as the token before the first guess) and its context, the ring row of the position before the
// block's
__global__ void ds_block_in_kernel(float* x, const float* __restrict__ mask, int n, int nb, DsPos p,
                                   const long long* token, float* state, int rank, int32_t* prev,
                                   const float* ring, int ring_rows, float* ctx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int r = 1; r < nb; ++r) x[(size_t) r * n + i] = mask[i];
    ctx[i] = ring[(size_t) ((ds_pos(p, 0) - 1) % ring_rows) * n + i];
    if (i < rank) state[i] = 0.0f;
    if (i == 0) *prev = (int32_t) *token;
}

// [state ; w1's row of the token before] (rank + rank values), the token *prev
__global__ void ds_markov_in_kernel(const float* __restrict__ state, const uint16_t* __restrict__ w1,
                                    const int32_t* __restrict__ prev, float* in, int rank) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rank) return;
    in[r] = state[r];
    in[rank + r] = ds_bf16(w1[(size_t) *prev * rank + r]);
}

// raw = a + b + c + d; gate = sigmoid(raw[0:r]), state = gate * state + (1 - gate) * tanh(raw[r:2r]),
// latent = tanh(raw[2r:3r])
__global__ void ds_markov_step_kernel(const float* __restrict__ ctx_h, const float* __restrict__ ctx_f,
                                      const float* __restrict__ bias, const float* __restrict__ sp, float* state,
                                      float* latent, int rank) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rank) return;
    const auto raw = [&](int k) { return ctx_h[k] + ctx_f[k] + bias[k] + sp[k]; };
    const float gate = 1.0f / (1.0f + expf(-raw(r)));
    state[r] = gate * state[r] + (1.0f - gate) * tanhf(raw(rank + r));
    latent[r] = tanhf(raw(2 * rank + r));
}

// A draft's pick in two launches: each block of the first takes a slice of the vocabulary, its scores
// cap tanh((base + fy + fb) / cap), and leaves its largest (the lowest id of equals) and its softmax sum against it;
// the second's one block joins the slices: the pick, its probability, and the pick as the next step's token.
constexpr int kDsPickBlocks = 128;

struct DsPart {
    float m, s;
    int id;
};

__global__ void ds_pick_part_kernel(const float* __restrict__ base, const float* __restrict__ fy,
                                    const float* __restrict__ fb, int n_keep, float cap, DsPart* part) {
    __shared__ float bm[256 / 32], bs[256 / 32];
    __shared__ int bi[256 / 32];
    const int per = (n_keep + gridDim.x - 1) / gridDim.x, v0 = blockIdx.x * per, v1 = min(n_keep, v0 + per);
    float m = -INFINITY, s = 0.0f;
    int id = 0x7fffffff;
    for (int v = v0 + threadIdx.x; v < v1; v += blockDim.x) {
        const float z = cap * tanhf((base[v] + fy[v] + fb[v]) / cap);
        if (z > m) {
            s = s * expf(m - z) + 1.0f;
            m = z;
            id = v;
        } else {
            s += expf(z - m);
        }
    }
    const auto join = [](float& m, float& s, int& id, float m2, float s2, int id2) {
        if (m2 > m || (m2 == m && id2 < id)) {
            s = (m == -INFINITY ? 0.0f : s * expf(m - m2)) + s2;
            m = m2;
            id = id2;
        } else if (m2 != -INFINITY) {
            s += s2 * expf(m2 - m);
        }
    };
    for (int o = 16; o > 0; o >>= 1) {
        const float m2 = __shfl_xor_sync(0xffffffff, m, o), s2 = __shfl_xor_sync(0xffffffff, s, o);
        const int id2 = __shfl_xor_sync(0xffffffff, id, o);
        join(m, s, id, m2, s2, id2);
    }
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) bm[w] = m, bs[w] = s, bi[w] = id;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < (int) blockDim.x / 32; ++k) join(m, s, id, bm[k], bs[k], bi[k]);
        part[blockIdx.x] = {m, s, id};
    }
}

__global__ void ds_pick_join_kernel(const DsPart* __restrict__ part, int parts, Likeliest* out, int32_t* prev) {
    if (threadIdx.x != 0) return;
    float m = -INFINITY, s = 0.0f;
    int id = 0x7fffffff;
    for (int k = 0; k < parts; ++k) {
        const DsPart p = part[k];
        if (p.m > m || (p.m == m && p.id < id)) {
            s = (m == -INFINITY ? 0.0f : s * expf(m - p.m)) + p.s;
            m = p.m;
            id = p.id;
        } else if (p.m != -INFINITY) {
            s += p.s * expf(p.m - m);
        }
    }
    out->token = id;
    out->prob = 1.0f / s;
    *prev = id;
}

// fold rows' Q8_0 block b (values 32b .. 32b + 31 of each row) from 32 rows of the transposed fold, ft[k][v]
__global__ void ds_fold_q8_kernel(const float* __restrict__ ft, int n_rows, int blocks, int b, uint8_t* fold) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_rows) return;
    float x[32], amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        x[k] = ft[(size_t) k * n_rows + v];
        amax = fmaxf(amax, fabsf(x[k]));
    }
    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    uint8_t* blk = fold + ((size_t) v * blocks + b) * 34;
    *(__half*) blk = __float2half(d);
#pragma unroll
    for (int k = 0; k < 32; ++k) blk[2 + k] = (uint8_t) (int8_t) roundf(x[k] * id);
}

// ---- the Markov head's pick, screened (ds_head_base(), ds_head_pick()) ----
//
// Basalt's include/basalt/kernels/head_proxy.cuh and src/models/inkling2/kernels/dspark.cu (MIT, the notice above):
// draft i's scores l[v] = cap tanh((head_v . h_i + fold_v . lat_i + fold_b[v]) / cap) are first scored from 2-bit
// copies of the head's and the fold's rows (the head term once for all the block's rows, the fold term a guess), each
// tile of 32 copy rows keeps its best kScrPer and its softmax partial; the rows within kScrDelta of the best are then
// scored exactly from their Q8_0 rows and the exact best is the guess.  The copy holds the tokens transposed over its
// tiles (copy row r is token (r % 32) T + r / 32, T the tiles), so that a tokenizer's frequent tokens, which sit
// together, fall in different tiles.

constexpr int kScrPer = 4;          // candidates a tile keeps
constexpr int kScrMaxCand = 64;     // rows scored exactly at most
constexpr float kScrDelta = 5.0f;   // a row is scored exactly when its copy's score is within this of the best
constexpr int kDsDraftsMax = 7;

__host__ __device__ inline int64_t scr_tiles(int64_t n) { return (n + 31) / 32; }
__host__ __device__ constexpr int scr_nw4(int n_in) { return n_in / 64; }   // uint4 words of a copy row
inline size_t scr_bytes(int64_t n, int n_in) { return (size_t) scr_tiles(n) * scr_nw4(n_in) * 32 * (16 + 4); }
inline uint16_t* scr_scales(const void* copy, int64_t n, int n_in) {
    return (uint16_t*) ((char*) copy + (size_t) scr_tiles(n) * scr_nw4(n_in) * 32 * 16);
}
__host__ __device__ __forceinline__ int64_t scr_token(int64_t r, int64_t tiles) { return (r & 31) * tiles + (r >> 5); }

__device__ __forceinline__ int scr_level(float w, float sh) { return min(3, max(0, (int) floorf(w / sh + 2.0f))); }

// A 32-weight group's scale for four levels: of a few candidates, the one of least squared error, then stretched by
// sqrt(|w|^2 / <w, w_hat>) (the least-squares copy scores the rows near the best low)
__device__ __forceinline__ __half scr_fit(const float* w, float& sh_out) {
    float amax = 1e-9f;
    for (int l = 0; l < 32; ++l) amax = fmaxf(amax, fabsf(w[l]));
    float best_s = amax / 1.5f, best_e = INFINITY;
    for (int t = 0; t < 6; ++t) {
        const float s = amax / 1.5f * (1.0f - 0.1f * (float) t);
        const float sh = __half2float(__float2half(s));
        float e = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float r = ((float) scr_level(w[l], sh) - 1.5f) * sh - w[l];
            e += r * r;
        }
        if (e < best_e) best_e = e, best_s = s;
    }
    __half hs = __float2half(best_s);
    const float sh = __half2float(hs);
    sh_out = sh;
    float ww = 0.0f, wr = 0.0f;
    for (int l = 0; l < 32; ++l) {
        const float r = ((float) scr_level(w[l], sh) - 1.5f) * sh;
        ww += w[l] * w[l];
        wr += w[l] * r;
    }
    if (wr > 0.0f) hs = __float2half(sh * sqrtf(ww / wr));
    return hs;
}

// a thread a 32-weight group g of copy row `row`: its token's Q8_0 group into the copy
template <int N_IN>
__global__ void scr_pack_kernel(const uint8_t* __restrict__ q8, int64_t n_rows, uint4* __restrict__ wq,
                                uint16_t* __restrict__ sc) {
    constexpr int NB = N_IN / 32, kNw4 = scr_nw4(N_IN);
    const int64_t tiles = scr_tiles(n_rows);
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tiles * 32 * NB) return;
    const int64_t row = i / NB, tok = scr_token(row, tiles);
    const int g = (int) (i % NB);
    if (tok >= n_rows) return;   // past the vocabulary: zeros, masked when scored
    const uint8_t* b = q8 + (tok * NB + g) * 34;
    const float d = __half2float(*(const __half*) b);
    float w[32];
    for (int l = 0; l < 32; ++l) w[l] = d * (float) (int8_t) b[2 + l];
    float sh;
    const __half hs = scr_fit(w, sh);
    const int tile = (int) (row >> 5), lane = (int) (row & 31), w4 = g / 2, h = g & 1;
    uint32_t words[2] = {0u, 0u};
    for (int c = 0; c < 2; ++c)
        for (int ii = 0; ii < 4; ++ii)
            for (int bb = 0; bb < 4; ++bb)
                words[c] |= (uint32_t) scr_level(w[c * 16 + 4 * ii + bb], sh) << (8 * bb + 2 * ii);
    uint32_t* wd = (uint32_t*) wq + (((size_t) tile * kNw4 + w4) * 32 + lane) * 4 + 2 * h;
    wd[0] = words[0];
    wd[1] = words[1];
    sc[(((size_t) tile * kNw4 + w4) * 32 + lane) * 2 + h] = __half_as_ushort(hs);
}

// by_row[r] = by_token[the token of copy row r], 0 past the vocabulary
__global__ void scr_order_kernel(const float* __restrict__ by_token, int64_t n, int64_t tiles, float* by_row) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= tiles * 32) return;
    const int64_t tok = scr_token(r, tiles);
    by_row[r] = tok < n ? by_token[tok] : 0.0f;
}

// a copy row's score (a lane a row) over this thread's uint4 words j0 .. j0 + NW - 1
template <int NW>
__device__ __forceinline__ float scr_acc(const uint4 (&w)[NW], const uint32_t (&s)[NW], int j0, const int* xi,
                                         const float* xs, const float* xdd) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < NW; ++j) {
        const int w4 = j0 + j;
        const uint32_t ww[4] = {w[j].x, w[j].y, w[j].z, w[j].w};
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int g = 2 * w4 + hh;
            int P = 0;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const uint32_t v = ww[2 * hh + c];
                const int4 x0 = *(const int4*) &xi[g * 8 + c * 4];
                P = __dp4a((int) (v & 0x03030303u), x0.x, P);
                P = __dp4a((int) ((v >> 2) & 0x03030303u), x0.y, P);
                P = __dp4a((int) ((v >> 4) & 0x03030303u), x0.z, P);
                P = __dp4a((int) ((v >> 6) & 0x03030303u), x0.w, P);
            }
            const float sg = __half2float(__ushort_as_half((unsigned short) (hh ? (s[j] >> 16) : (s[j] & 0xffff))));
            acc += sg * (xdd[g] * (float) P - 1.5f * xs[g]);
        }
    }
    return acc;
}

// q8_1 blocks into the scoring layout: the codes as ints, each block's scale and scaled code sum
__device__ __forceinline__ void scr_x(const QB* x, int nb, int* xi, float* xs, float* xdd, int tid, int nthreads) {
    for (int i = tid; i < nb * 8; i += nthreads) xi[i] = ((const int*) x[i / 8].qs)[i % 8];
    for (int b = tid; b < nb; b += nthreads) {
        const int* q = (const int*) x[b].qs;
        int sum = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) sum = __dp4a(0x01010101, q[i], sum);
        const float d8 = __low2float(x[b].ds);
        xdd[b] = d8;
        xs[b] = d8 * (float) sum;
    }
}

// tiles kScrBaseTiles blockIdx.x on: approx[r][copy row] = the head copy's score of q8_1 row r of x (r < rows), four
// warps splitting the row, lane = copy row; the copy is read once for all the rows, the rows once for the tiles
constexpr int kScrBaseTiles = 4;
template <int N_IN>
__global__ void __launch_bounds__(128) scr_base_kernel(const uint4* __restrict__ wq, const uint32_t* __restrict__ sc,
                                                       const QB* x, int rows, int64_t n_tiles, int64_t n_pad,
                                                       float* approx) {
    constexpr int NB = N_IN / 32, kNw4 = scr_nw4(N_IN), NW = kNw4 / 4;
    __shared__ __align__(16) int xi[kDsDraftsMax][N_IN / 4];
    __shared__ float xs[kDsDraftsMax][NB], xdd[kDsDraftsMax][NB];
    __shared__ float red[kDsDraftsMax][4][32];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int r = 0; r < rows; ++r) scr_x(x + (size_t) r * NB, NB, xi[r], xs[r], xdd[r], tid, 128);
    for (int64_t tile = (int64_t) blockIdx.x * kScrBaseTiles;
         tile < min(n_tiles, (int64_t) (blockIdx.x + 1) * kScrBaseTiles); ++tile) {
        const size_t base = ((size_t) tile * kNw4 + warp * NW) * 32 + lane;
        uint4 w[NW];
        uint32_t s[NW];
#pragma unroll
        for (int j = 0; j < NW; ++j) {
            w[j] = __ldcs(wq + base + (size_t) j * 32);
            s[j] = __ldcs(sc + base + (size_t) j * 32);
        }
        __syncthreads();   // the rows are in, and red is free again
#pragma unroll
        for (int r = 0; r < kDsDraftsMax; ++r)
            if (r < rows) red[r][warp][lane] = scr_acc<NW>(w, s, warp * NW, xi[r], xs[r], xdd[r]);
        __syncthreads();
        const int64_t row = tile * 32 + lane;
        for (int r = warp; r < rows; r += 4)
            approx[r * n_pad + row] = red[r][0][lane] + red[r][1][lane] + red[r][2][lane] + red[r][3][lane];
    }
}

// a warp a tile of 32 copy rows (lane = row): the fold copy's score of the latent plus the head term and the bias,
// capped; the tile's softmax partial and its best kScrPer rows
constexpr int kScrTilesPerBlock = 4;
template <int RANK>
__global__ void __launch_bounds__(32 * kScrTilesPerBlock)
    scr_step_kernel(const uint4* __restrict__ wq, const uint32_t* __restrict__ sc, const QB* lat, const float* approx,
                    const float* __restrict__ fold_bp, float cap, int64_t n_rows, int n_tiles, float2* cand,
                    float2* part) {
    constexpr int kNw4 = scr_nw4(RANK), NB = RANK / 32;
    __shared__ __align__(16) int xi[RANK / 4];
    __shared__ float xs[NB], xdd[NB];
    const int tid = threadIdx.x, lane = tid & 31;
    const int tile = (int) blockIdx.x * kScrTilesPerBlock + (tid >> 5);
    const int64_t row = (int64_t) tile * 32 + lane;
    const bool live = tile < n_tiles, real = live && scr_token(row, n_tiles) < n_rows;
    uint4 w[kNw4];
    uint32_t s[kNw4];
#pragma unroll
    for (int j = 0; j < kNw4; ++j) {
        const size_t at = ((size_t) (live ? tile : 0) * kNw4 + j) * 32 + lane;
        w[j] = __ldcs(wq + at);
        s[j] = __ldcs(sc + at);
    }
    const float fb = live ? __ldg(fold_bp + row) : 0.0f;
    scr_x(lat, NB, xi, xs, xdd, tid, 32 * kScrTilesPerBlock);
    __syncthreads();
    if (!live) return;
    const float ninf = -INFINITY;
    float v = ninf;
    if (real) v = cap * tanhf((approx[row] + scr_acc<kNw4>(w, s, 0, xi, xs, xdd) + fb) / cap);
    const float m = warp_max(v);
    const float e = warp_sum(v > ninf ? __expf(v - m) : 0.0f);
    if (lane == 0) part[tile] = make_float2(m, e);
    for (int k = 0; k < kScrPer; ++k) {
        float bv = v;
        int bi = lane;
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
        }
        if (lane == 0) cand[tile * kScrPer + k] = make_float2(bv, __int_as_float(tile * 32 + bi));
        if (lane == bi) v = ninf;
    }
}

struct ScrFinal {
    float wmax[16];
    int list[kScrMaxCand];
    float lp[kScrMaxCand], ex[kScrMaxCand];
    int cnt;
    int hist[256];
    unsigned long long want;
    int need;
};

// a candidate's key: its approximate score, then the lower row on a tie
__device__ __forceinline__ unsigned long long scr_key(float2 c) {
    const unsigned u = __float_as_uint(c.x);
    const unsigned o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long) o << 32) | (unsigned) (0xffffffffu - (unsigned) __float_as_int(c.y));
}

// the kScrMaxCand-th largest key of the nc candidates within kScrDelta of m: a radix select over the key's bytes
__device__ unsigned long long scr_kth(ScrFinal& f, int nc, const float2* cand, float m) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) f.want = 0ull, f.need = kScrMaxCand;
    unsigned long long mask = 0ull;
    for (int shift = 56; shift >= 0; shift -= 8) {
        if (tid < 256) f.hist[tid] = 0;
        __syncthreads();
        const unsigned long long want = f.want;
        for (int i = tid; i < nc; i += 512) {
            const float2 c = __ldcg(&cand[i]);
            if (!(c.x > -INFINITY && c.x >= m - kScrDelta)) continue;
            const unsigned long long k = scr_key(c);
            if ((k & mask) == want) atomicAdd(&f.hist[(int) ((k >> shift) & 255u)], 1);
        }
        __syncthreads();
        if (warp == 0) {   // lane l holds bins 255 - 8l .. 248 - 8l
            int b[8], s = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) b[j] = f.hist[255 - 8 * lane - j], s += b[j];
            int incl = s;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += v;
            }
            const int need = f.need, excl = incl - s;
            if (excl < need && incl >= need) {
                int acc = excl, j = 0;
                while (acc + b[j] < need) acc += b[j++];
                f.want = want | ((unsigned long long) (255 - 8 * lane - j) << shift);
                f.need = need - acc;
            }
        }
        mask |= 255ull << shift;
        __syncthreads();
    }
    return f.want;
}

// the tiles' candidates within kScrDelta of the best into f.list (ascending copy row), the kScrMaxCand best of them
// where there are more; the count
__device__ int scr_collect(ScrFinal& f, int n_tiles, const float2* cand, const float2* part) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float m = -INFINITY;
    for (int t = tid; t < n_tiles; t += 512) m = fmaxf(m, __ldcg(&part[t].x));
    m = warp_max(m);
    if (lane == 0) f.wmax[warp] = m;
    __syncthreads();
    m = f.wmax[0];
    for (int w = 1; w < 16; ++w) m = fmaxf(m, f.wmax[w]);
    if (m == -INFINITY) return 0;
    for (int t = tid; t < n_tiles; t += 512) {
        const float tm = __ldcg(&part[t].x);
        if (!(tm > -INFINITY && tm >= m - kScrDelta)) continue;
        for (int k = 0; k < kScrPer; ++k) {
            const float2 c = __ldcg(&cand[t * kScrPer + k]);
            if (c.x > -INFINITY && c.x >= m - kScrDelta) {
                const int p = atomicAdd(&f.cnt, 1);
                if (p < kScrMaxCand) f.list[p] = __float_as_int(c.y), f.lp[p] = c.x;
            }
        }
    }
    __syncthreads();
    const int total = f.cnt;
    if (total > kScrMaxCand) {
        const int nc = n_tiles * kScrPer;
        const unsigned long long kth = scr_kth(f, nc, cand, m);
        if (tid == 0) f.cnt = 0;
        __syncthreads();
        for (int i = tid; i < nc; i += 512) {
            const float2 c = __ldcg(&cand[i]);
            if (c.x > -INFINITY && c.x >= m - kScrDelta && scr_key(c) >= kth) {
                const int p = atomicAdd(&f.cnt, 1);
                if (p < kScrMaxCand) f.list[p] = __float_as_int(c.y), f.lp[p] = c.x;
            }
        }
        __syncthreads();
    }
    const int n = min(total, kScrMaxCand);
    int row = 0, rank = 0;
    float lp = 0.0f;
    if (tid < n) {
        row = f.list[tid];
        lp = f.lp[tid];
        for (int j = 0; j < n; ++j) rank += f.list[j] < row;
    }
    __syncthreads();
    if (tid < n) f.list[rank] = row, f.lp[rank] = lp;
    __syncthreads();
    return n;
}

// a Q8_0 row of nb blocks by a q8_1 row, on a warp: lane L takes words 2 (L % 4), +1 of blocks L / 4 + 8 j
__device__ __forceinline__ float scr_dot(const uint8_t* w, const QB* x, int nb, int lane) {
    const int k = 2 * (lane & 3);
    float s = 0.0f;
    for (int b = lane >> 2; b < nb; b += 8) {
        const uint8_t* blk = w + (size_t) b * 34;
        const uint16_t* q16 = (const uint16_t*) (blk + 2);
        const int* xq = (const int*) x[b].qs;
        const int w0 = (int) q16[2 * k] | ((int) q16[2 * k + 1] << 16);
        const int w1 = (int) q16[2 * k + 2] | ((int) q16[2 * k + 3] << 16);
        const int sumi = __dp4a(w1, xq[k + 1], __dp4a(w0, xq[k], 0));
        s += __half2float(*(const __half*) blk) * __low2float(x[b].ds) * (float) sumi;
    }
    return warp_sum(s);
}

// One block: the candidates scored exactly (a warp a row), the exact best is guess i: out[i] and *prev, with its
// probability from the tiles' softmax sums with the rescored rows made exact
template <int NE, int NR>
__global__ void __launch_bounds__(512) scr_final_kernel(DsScreen h, const QB* xh_g, const QB* xl_g, int n_tiles,
                                                         const float2* cand, const float2* part, int32_t* prev,
                                                         Likeliest* out) {
    __shared__ QB xh[NE], xl[NR];
    __shared__ ScrFinal f;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) f.cnt = 0;
    for (int k = tid; k < (NE + NR) * 9; k += 512) {   // nine words a block
        const int b = k / 9, j = k % 9;
        ((int*) (b < NE ? xh + b : xl + (b - NE)))[j] = ((const int*) (b < NE ? xh_g + b : xl_g + (b - NE)))[j];
    }
    __syncthreads();
    const int n = scr_collect(f, n_tiles, cand, part);
    if (n == 0) {   // never: the vocabulary has rows
        if (tid == 0) out->token = *prev, out->prob = 0.0f;
        return;
    }
    for (int c = warp; c < n; c += 16) {   // by token from here: a tie goes to the lower token
        const int64_t tok = scr_token(f.list[c], n_tiles);
        const float a = scr_dot((const uint8_t*) h.head + tok * NE * 34, xh, NE, lane);
        const float b = scr_dot((const uint8_t*) h.fold + tok * NR * 34, xl, NR, lane);
        if (lane == 0) f.ex[c] = h.cap * tanhf((a + b + h.fold_b[tok]) / h.cap), f.list[c] = (int) tok;
    }
    __syncthreads();
    if (warp == 0) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int c = lane; c < n; c += 32)
            if (f.ex[c] > bv || (f.ex[c] == bv && f.list[c] < bi)) bv = f.ex[c], bi = f.list[c];
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
        }
        if (lane == 0) f.wmax[0] = bv, f.cnt = bi;
    }
    __syncthreads();
    const float bv = f.wmax[0];
    const int bi = f.cnt;
    float sum = 0.0f;
    for (int t = tid; t < n_tiles; t += 512) {
        const float2 p = __ldcg(&part[t]);
        if (p.y > 0.0f) sum += p.y * __expf(p.x - bv);
    }
    for (int c = tid; c < n; c += 512) sum += __expf(f.ex[c] - bv) - __expf(f.lp[c] - bv);
    sum = warp_sum(sum);
    __syncthreads();   // f.wmax is free again
    if (lane == 0) f.wmax[warp] = sum;
    __syncthreads();
    if (tid == 0) {
        sum = 0.0f;
        for (int w = 0; w < 16; ++w) sum += f.wmax[w];
        out->token = bi;
        out->prob = 1.0f / fmaxf(sum, 1.0f);
        *prev = bi;
    }
}

}  // namespace

void ds_mix(const DsMix& m, int n, int rows, float eps, void* stream) {
    if (rows <= 0) return;
    ds_mix_kernel<<<dim3(rows, 4), 256, (size_t) n * sizeof(float), cs(stream)>>>(m, n, rows, eps);
    launched("ds_mix");
}

void ds_rope(float* x, int heads, int rows, const float* w, float eps, float theta, const DsPos& p, uint16_t* cache,
             int64_t cap, const float* v, uint16_t* vcache, void* stream, long long* tags) {
    if (rows <= 0) return;
    ds_rope_kernel<<<dim3(rows, heads), kDsHead, 0, cs(stream)>>>(x, heads, w, eps, theta, p, (__half*) cache,
                                                                  (long long) cap, v, (__half*) vcache, tags);
    launched("ds_rope");
}

size_t ds_attend_floats(int kv_heads) { return (size_t) kv_heads * kDsSplits * kDsRows * kDsPart; }

void ds_attend(const float* q, const uint16_t* keys, const uint16_t* vals, const long long* tags, int64_t cap,
               const DsPos& p, int window, int rows, int heads, int kv_heads, float scale, float* part, float* out,
               void* stream) {
    if (heads % kv_heads || rows * (heads / kv_heads) > kDsRows)
        throw std::logic_error("ds_attend: more query rows a kv head than it holds");
    ds_part_kernel<<<dim3(kv_heads, kDsSplits), 256, 0, cs(stream)>>>(
        q, (const __half*) keys, (const __half*) vals, tags, (long long) cap, p, rows, window, heads, kv_heads, scale,
        part);
    launched("ds_part");
    ds_join_kernel<<<dim3(rows, heads), kDsHead, 0, cs(stream)>>>(part, p, rows, window, heads, kv_heads, out);
    launched("ds_join");
}

void ds_swiglu(float* g, const float* u, size_t n, void* stream) {
    ds_swiglu_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(g, u, n);
    launched("ds_swiglu");
}

void ds_block_in(float* x, const float* mask, int n, int rows, const DsPos& p, const long long* token, float* state,
                 int rank, int32_t* prev, const float* ring, int ring_rows, float* ctx, void* stream) {
    ds_block_in_kernel<<<grid((size_t) n, 256), 256, 0, cs(stream)>>>(x, mask, n, rows, p, token, state, rank, prev,
                                                                       ring, ring_rows, ctx);
    launched("ds_block_in");
}

void ds_markov_in(const float* state, const uint16_t* w1, const int32_t* prev, float* in, int rank, void* stream) {
    ds_markov_in_kernel<<<grid((size_t) rank, 256), 256, 0, cs(stream)>>>(state, w1, prev, in, rank);
    launched("ds_markov_in");
}

void ds_markov_step(const float* ctx_h, const float* ctx_f, const float* bias, const float* sp, float* state,
                    float* latent, int rank, void* stream) {
    ds_markov_step_kernel<<<grid((size_t) rank, 256), 256, 0, cs(stream)>>>(ctx_h, ctx_f, bias, sp, state, latent,
                                                                           rank);
    launched("ds_markov_step");
}

size_t ds_pick_floats() { return kDsPickBlocks * sizeof(DsPart) / sizeof(float); }

void ds_pick(const float* base, const float* fy, const float* fb, int n_keep, float cap, float* scratch, Likeliest* out,
             int32_t* prev, void* stream) {
    ds_pick_part_kernel<<<kDsPickBlocks, 256, 0, cs(stream)>>>(base, fy, fb, n_keep, cap, (DsPart*) scratch);
    launched("ds_pick_part");
    ds_pick_join_kernel<<<1, 32, 0, cs(stream)>>>((const DsPart*) scratch, kDsPickBlocks, out, prev);
    launched("ds_pick_join");
}

void ds_fold_q8(const float* ft, int n_rows, int blocks, int b, void* fold, void* stream) {
    ds_fold_q8_kernel<<<grid((size_t) n_rows, 256), 256, 0, cs(stream)>>>(ft, n_rows, blocks, b, (uint8_t*) fold);
    launched("ds_fold_q8");
}

bool ds_screens(int n_embd, int rank) { return n_embd == 2304 && rank == 256; }

size_t ds_screen_bytes(int n_in, int64_t n_rows) { return scr_bytes(n_rows, n_in); }

int64_t ds_screen_rows(int64_t n_rows) { return scr_tiles(n_rows) * 32; }

size_t ds_screen_scratch_bytes(int64_t n_rows) { return (size_t) scr_tiles(n_rows) * (kScrPer + 1) * sizeof(float2); }

void ds_screen_pack(const void* q8_0, int n_in, int64_t n_rows, void* copy, void* stream) {
    if (n_in != 2304 && n_in != 256) throw std::logic_error("ds_screen_pack: rows of 2,304 or 256");
    // the last tile's rows past n_rows score as zeros
    if (cudaMemsetAsync(copy, 0, scr_bytes(n_rows, n_in), cs(stream)) != cudaSuccess) launched("ds_screen_pack");
    const int64_t total = scr_tiles(n_rows) * 32 * (n_in / 32);
    const unsigned blocks = (unsigned) ((total + 127) / 128);
    uint16_t* sc = scr_scales(copy, n_rows, n_in);
    const uint8_t* q = (const uint8_t*) q8_0;
    if (n_in == 2304) scr_pack_kernel<2304><<<blocks, 128, 0, cs(stream)>>>(q, n_rows, (uint4*) copy, sc);
    else scr_pack_kernel<256><<<blocks, 128, 0, cs(stream)>>>(q, n_rows, (uint4*) copy, sc);
    launched("ds_screen_pack");
}

void ds_screen_order(const float* by_token, int64_t n_rows, float* by_row, void* stream) {
    const int64_t n = ds_screen_rows(n_rows);
    scr_order_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, cs(stream)>>>(by_token, n_rows, scr_tiles(n_rows), by_row);
    launched("ds_screen_order");
}

void ds_head_base(const DsScreen& h, const void* hq, int rows, float* approx, void* stream) {
    if (rows < 1 || rows > kDsDraftsMax) throw std::logic_error("ds_head_base: 1 to 7 rows");
    const int64_t tiles = scr_tiles(h.n_rows);
    scr_base_kernel<2304><<<(unsigned) ((tiles + kScrBaseTiles - 1) / kScrBaseTiles), 128, 0, cs(stream)>>>(
        (const uint4*) h.head_sc, (const uint32_t*) scr_scales(h.head_sc, h.n_rows, 2304), (const QB*) hq, rows, tiles,
        ds_screen_rows(h.n_rows), approx);
    launched("ds_head_base");
}

void ds_head_pick(const DsScreen& h, const void* hq, const void* latq, const float* approx, int i, void* scratch,
                  int32_t* prev, Likeliest* out, void* stream) {
    const int tiles = (int) scr_tiles(h.n_rows);
    float2* cand = (float2*) scratch;
    float2* part = cand + (size_t) tiles * kScrPer;
    scr_step_kernel<256><<<(unsigned) ((tiles + kScrTilesPerBlock - 1) / kScrTilesPerBlock), 32 * kScrTilesPerBlock, 0,
                           cs(stream)>>>((const uint4*) h.fold_sc,
                                         (const uint32_t*) scr_scales(h.fold_sc, h.n_rows, 256), (const QB*) latq,
                                         approx + (size_t) i * ds_screen_rows(h.n_rows), h.fold_bp, h.cap, h.n_rows,
                                         tiles, cand, part);
    launched("ds_head_step");
    scr_final_kernel<2304 / 32, 256 / 32><<<1, 512, 0, cs(stream)>>>(h, (const QB*) hq + (size_t) i * (2304 / 32),
                                                                     (const QB*) latq, tiles, cand, part, prev, out);
    launched("ds_head_final");
}

}  // namespace strata::poly
