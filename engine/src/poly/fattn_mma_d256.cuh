// src/poly/fattn_mma_d256.cuh - fattn_mma.cuh's flash-attention kernel (llama.cpp's, with the causal mask computed
// in the kernel and each tile's softmax max and sum left in dst_meta) cut down to the one case src/poly/fattn.cu
// launches, so that a build without llama.cpp has it: heads of 256 values for K, V and the result, a tile of 64
// query rows (8 tokens x 8 heads of a KV head's group, or 4 x 16), fp16 K and V, no softcap, sinks or sparse keys,
// one CUDA block a tile (the stream-k fixup is never taken), on Ampere or later.  The launch values are llama.cpp's
// Ampere configuration for that case (128 threads, 32 keys a step, the whole head of K or V a load, two cp.async
// stages, Q in registers, swizzled K and V tiles), which every architecture from 8.0 up takes.
//
// Every step is the original's arithmetic in the original's order, so the results are the same bits as the kernel
// a -DSTRATA_POLY_MMQ=ON build compiles from fattn_mma.cuh, provided it is compiled with --use_fast_math as that one
// is (cmake/strata_poly.cmake); poly_fattn_test prints a digest of every case's results for comparing the two.
//
// Specialized from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d,
// ggml/src/ggml-cuda/{fattn-mma-f16.cuh,mma.cuh,fattn-swizzle.cuh,cp-async.cuh,fattn-common.cuh,common.cuh}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
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
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace strata::poly::fattn_d256 {

constexpr int kD = 256, kD2 = kD / 2;          // values of a head, and as half2
constexpr int kCols = 64;                       // query rows of a tile: ncols1 tokens x ncols2 heads
constexpr int kThreads = 128, kWarps = 4, kOccupancy = 2;
constexpr int kBatchFa = 32;                    // keys a softmax step (nbatch_fa)
constexpr int kBatchCombine = 128;              // half2 of a result row combined at a time (nbatch_combine)
constexpr int kColsPerWarp = 16;                // query rows a warp (np = 1: one warp a row)
constexpr int kStrideQ = kD2 + 4;               // a row of tile_Q, in half2
constexpr int kStrideKV = kD2;                  // a row of the K and V tiles: swizzled, so not padded
constexpr int kStrideCombine = kBatchCombine + 4;
constexpr float kSoftmaxFtz = -20.0f;           // SOFTMAX_FTZ_THRESHOLD
#define STRATA_FATTN_D256_MAX_OFFSET (3.0f*0.6931f)   // FATTN_KQ_MAX_OFFSET, folded as the original folds it

// The dynamic shared memory of a launch, as ggml_cuda_flash_attn_ext_mma_f16_case() works it out: Q is held in
// registers, so its tile is reused for the two stages of K and V and the mask, and at the end for combining.
template <int ncols1>
constexpr size_t smem_bytes() {
    const size_t kv = size_t(kBatchFa) * (kStrideKV + kStrideKV) * sizeof(half2);
    const size_t q = size_t(kCols) * (kD2 + 4) * sizeof(half2);
    const size_t mask = size_t(ncols1) * (kBatchFa / 2 + 4) * sizeof(half2);
    const size_t combine = size_t(kWarps) * kColsPerWarp * (kBatchCombine + 4) * sizeof(half2);
    return std::max(combine, std::max(q, kv + mask));
}

// n / d as (mulhi(n, mp) + n) >> L, the divisor kept in .z (common.cuh's init_fastdiv_values(), d >= 1)
static uint3 init_fastdiv_values(uint64_t d_64) {
    const uint32_t d = (uint32_t) d_64;
    uint32_t L = 0;
    while (L < 32 && (uint32_t{1} << L) < d) L++;
    const uint32_t mp = (uint32_t) ((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
    return make_uint3(mp, L, d);
}

// The kernel has an image for the current card compiled for Ampere or later, which is what the original's
// ampere_mma_available() asks of the architectures compiled for; no image at all is a no too.
template <typename Kernel>
bool runs_here(Kernel kernel) {
    cudaFuncAttributes a{};
    const bool ok = cudaFuncGetAttributes(&a, (const void*) kernel) == cudaSuccess && a.ptxVersion >= 80;
    cudaGetLastError();
    return ok;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800   // (an older card's image only traps)

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

// mma.cuh's tile<16, 8, half2> (Q, K, V and the results) and tile<16, 16, float> (the scores) on Turing and later:
// a warp's 32-bit elements, l of a thread at row get_i(l) and 32-bit column get_j(l)
struct tile_h2 {
    static constexpr int I = 16, J = 8, ne = 4;
    half2 x[ne] = {{0.0f, 0.0f}};
    static __device__ __forceinline__ int get_i(const int l) { return ((l % 2) * 8) + (threadIdx.x / 4); }
    static __device__ __forceinline__ int get_j(const int l) { return ((l / 2) * 4) + (threadIdx.x % 4); }
};

struct tile_f {
    static constexpr int I = 16, J = 16, ne = 8;
    float x[ne] = {0};
    static __device__ __forceinline__ int get_i(const int l) { return (((l / 2) % 2) * 8) + (threadIdx.x / 4); }
    static __device__ __forceinline__ int get_j(const int l) {
        return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
    }
};

// D += A B, the scores: floats from halves
static __device__ __forceinline__ void mma(tile_f& D, const tile_h2& A, const tile_h2& B) {
    const int* Axi = (const int*) A.x;
    const int* Bxi = (const int*) B.x;
    int* Dxi = (int*) D.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[2]));
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(Dxi[4]), "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]), "r"(Bxi[3]));
}

// D += A B, the values: summed in halves, as llama.cpp does on NVIDIA cards
static __device__ __forceinline__ void mma(tile_h2& D, const tile_h2& A, const tile_h2& B) {
    const int* Axi = (const int*) A.x;
    const int* Bxi = (const int*) B.x;
    int* Dxi = (int*) D.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(Dxi[0]), "+r"(Dxi[1])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[2]));
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]), "r"(Bxi[3]));
}

// the scores' tiles as the halves the values are multiplied by (mma.cuh's get_half2())
static __device__ __forceinline__ tile_h2 get_half2(const tile_f& tile_float) {
    tile_h2 ret;
#pragma unroll
    for (int l0 = 0; l0 < tile_float.ne; l0 += 2) {
        ret.x[l0 / 2] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
    }
    return ret;
}

// Q from tile_Q, which is not swizzled
static __device__ __forceinline__ void load_ldmatrix(tile_h2& t, const half2* __restrict__ xs0, const int stride) {
    int* xi = (int*) t.x;
    const int* xs = (const int*) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
                 : "l"(xs));
}

// fattn-swizzle.cuh: a K or V tile's half2 (row, col_h2) at this byte offset, the same map for writes and reads
static __device__ __forceinline__ int bytes_rc(const int row, const int col_h2) {
    return ((row * kStrideKV + col_h2) * (int) sizeof(half2)) ^ ((row & 7) << 4);
}

// a lane's address in a swizzled tile for an ldmatrix of a tile_h2 at (base_row, base_col_h2)
static __device__ __forceinline__ const half2* lane_addr(const half2* tile_base, const int base_row,
                                                         const int base_col_h2) {
    const int lane_row = threadIdx.x % tile_h2::I;
    const int lane_col = (threadIdx.x / tile_h2::I) * (tile_h2::J / 2);
    uint32_t byte_off = (uint32_t) ((base_row + lane_row) * kStrideKV + base_col_h2 + lane_col) * (uint32_t) sizeof(half2);
    byte_off ^= (uint32_t) (((base_row + lane_row) & 7) << 4);
    return (const half2*) ((const char*) tile_base + byte_off);
}

// K from its swizzled tile
static __device__ __forceinline__ void load_k(tile_h2& t, const half2* tile_base, const int base_row,
                                              const int base_col_h2) {
    int* xi = (int*) t.x;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
                 : "l"(lane_addr(tile_base, base_row, base_col_h2)));
}

// V from its swizzled tile, transposed on the way (it lies keys by values in shared memory)
static __device__ __forceinline__ void load_v_trans(tile_h2& t, const half2* tile_base, const int off_h2) {
    int* xi = (int*) t.x;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
                 : "l"(lane_addr(tile_base, off_h2 / kStrideKV, off_h2 % kStrideKV)));
}

static __device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;"); }

// Keys k_VKQ_0 .. +kBatchFa of K or V, whole heads (128 half2, 32 chunks of 16 bytes: a warp a row at a time, a
// thread a chunk), into a swizzled tile with cp.async (flash_attn_ext_f16_load_tile() with use_cp_async)
static __device__ __forceinline__ void load_tile(const half2* __restrict__ KV, half2* __restrict__ tile_KV,
                                                 const int stride_KV, const int k_VKQ_0) {
    const unsigned int tile_KV_32 = __cvta_generic_to_shared(tile_KV);
#pragma unroll
    for (int i0 = 0; i0 < kBatchFa; i0 += kWarps) {
        const int i = i0 + threadIdx.y;
        const int k = threadIdx.x;
        const int64_t i_KV = k_VKQ_0 + i;
        asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;"
                     : : "r"(tile_KV_32 + bytes_rc(i, k * 4)), "l"(KV + i_KV * stride_KV + k * 4));
    }
}

// fattn_mma.cuh's change 1: the causal mask of keys k_VKQ_0 .. +kBatchFa for the tile's tokens from j0 on, as the
// mask's loads would have left it: query j sees the keys up to j + stride_mask
template <int ncols1>
static __device__ __forceinline__ void load_mask(half* __restrict__ tile_mask, const int stride_mask,
                                                 const int k_VKQ_0, const int j0, const uint3 ne01) {
#pragma unroll
    for (int j1 = 0; j1 < ncols1; j1 += kWarps) {
        const int j_sram = j1 + threadIdx.y;
        if (j1 + kWarps > ncols1 && j_sram >= ncols1) {
            break;
        }
        const int j_vram = fastmodulo(j0 + j_sram, ne01);
        const int i = threadIdx.x;
        tile_mask[j_sram * (kBatchFa + 8) + i] = k_VKQ_0 + i > j_vram + stride_mask ? half(-INFINITY) : half(0.0f);
    }
}

// One step of kBatchFa keys (flash_attn_ext_f16_iter()): the scores, the mask, the softmax with the running max and
// sum, the values.  The step's keys and mask are in shared memory already; its values are loaded here while the
// scores are worked out, and the next step's keys and mask while the values are.
template <int ncols1, int ncols2, bool last_iter>
static __device__ __forceinline__ void iter(
        const half2* __restrict__ K_h2, const half2* __restrict__ V_h2, const float slope, const uint3 ne01,
        const int stride_K, const int stride_V, const int stride_mask, half2* __restrict__ tile_K,
        half2* __restrict__ tile_V, half* __restrict__ tile_mask, const tile_h2* __restrict__ Q_B,
        tile_h2* __restrict__ VKQ_C, float* __restrict__ KQ_max, float* __restrict__ KQ_rowsum, const int jt,
        const int kb0) {
    constexpr int cols_per_thread = 2;
    const int k_VKQ_0 = kb0 * kBatchFa;
    tile_f KQ_C[kBatchFa / tile_f::J];

    cp_async_wait_all();
    __syncthreads();
    load_tile(V_h2, tile_V, stride_V, k_VKQ_0);

    // KQ, A and B swapped so that KQ_C is column-major
#pragma unroll
    for (int i_KQ_00 = 0; i_KQ_00 < kBatchFa; i_KQ_00 += tile_h2::I) {
        const int i_KQ_0 = i_KQ_00;
#pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < kD2; k_KQ_0 += tile_h2::J) {
            tile_h2 K_A;
            load_k(K_A, tile_K, i_KQ_0, k_KQ_0);
            mma(KQ_C[i_KQ_00 / tile_h2::I], Q_B[k_KQ_0 / tile_h2::J], K_A);
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

#pragma unroll
    for (int i00 = 0; i00 < kBatchFa; i00 += tile_f::J) {
        const int i0 = i00;
#pragma unroll
        for (int l0 = 0; l0 < tile_f::ne; l0 += 2) {
            const int i = (i0 + tile_f::get_j(l0)) / 2;
            const int j = (threadIdx.y * kColsPerWarp + tile_f::get_i(l0)) / ncols2;

            const float2 tmp = __half22float2(((const half2*) tile_mask)[j * (kBatchFa / 2 + 4) + i]);
            KQ_C[i00 / tile_f::J].x[l0 + 0] += slope * tmp.x;
            KQ_C[i00 / tile_f::J].x[l0 + 1] += slope * tmp.y;
        }
    }

    // The softmax with the current max; the divisor is kept in KQ_rowsum and applied at the end
#pragma unroll
    for (int k0 = 0; k0 < kBatchFa; k0 += tile_f::J) {
#pragma unroll
        for (int l = 0; l < tile_f::ne; ++l) {
            const int KQ_idx = (l / 2) % 2;
            KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0 / tile_f::J].x[l] + STRATA_FATTN_D256_MAX_OFFSET);
        }
    }

    // a query's scores are spread across 4 threads
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
        for (int offset = 2; offset >= 1; offset >>= 1) {
            KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, 32));
        }
    }

#pragma unroll
    for (int k0 = 0; k0 < kBatchFa; k0 += tile_f::J) {
#pragma unroll
        for (int l = 0; l < tile_f::ne; ++l) {
            const int KQ_idx = (l / 2) % 2;
            KQ_C[k0 / tile_f::J].x[l] = expf(KQ_C[k0 / tile_f::J].x[l] - KQ_max_new[KQ_idx]);
            KQ_rowsum_add[KQ_idx] += KQ_C[k0 / tile_f::J].x[l];
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];

            *((uint32_t*) &KQ_max_scale[col]) *= KQ_max_diff >= kSoftmaxFtz;

            // the sum so far scaled to the new max
            KQ_rowsum[col] = KQ_max_scale[col] * KQ_rowsum[col] + KQ_rowsum_add[col];
        }

#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
            for (int i = 0; i < kD2 / tile_h2::J; ++i) {
#pragma unroll
                for (int l0 = 0; l0 < tile_h2::ne; l0 += 2) {
                    VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                }
            }
        }
    }

    tile_h2 B[kBatchFa / (2 * tile_h2::J)];
    for (int k = 0; k < kBatchFa / (2 * tile_h2::J); ++k) {
        B[k] = get_half2(KQ_C[k]);
    }

    // the next step's mask and keys
    cp_async_wait_all();
    __syncthreads();
    if (!last_iter) {
        load_mask<ncols1>(tile_mask, stride_mask, k_VKQ_0 + kBatchFa, jt * ncols1, ne01);
        load_tile(K_h2, tile_K, stride_K, k_VKQ_0 + kBatchFa);
    }

    // VKQ, A and B swapped as for KQ
#pragma unroll
    for (int i_VKQ_0 = 0; i_VKQ_0 < kD; i_VKQ_0 += tile_h2::I) {
#pragma unroll
        for (int k00 = 0; k00 < kBatchFa / 2; k00 += tile_h2::J) {
            const int k0 = k00;
            tile_h2 A;
            load_v_trans(A, tile_V, 2 * k0 * kStrideKV + i_VKQ_0 / 2);
            mma(VKQ_C[i_VKQ_0 / tile_h2::I], B[k00 / tile_h2::J], A);
        }
    }
}

// A tile of ncols1 tokens x ncols2 heads over the keys of steps kb0_start .. kb0_stop - 1
// (flash_attn_ext_f16_process_tile() for a tile done whole), its max and sum to dstk_meta (fattn_mma.cuh change 2)
template <int ncols1, int ncols2>
static __device__ __forceinline__ void process_tile(
        const float2* __restrict__ Q_f2, const half2* __restrict__ K_h2, const half2* __restrict__ V_h2,
        float2* __restrict__ dstk, float2* __restrict__ dstk_meta, const float scale, const float slope,
        const uint3 ne01, const int ne02, const int gqa_ratio, const int stride_Q1, const int stride_Q2,
        const int stride_K, const int stride_V, const int stride_mask, const int jt, const int zt_gqa,
        const int kb0_start, const int kb0_stop) {
    constexpr int ncols = ncols1 * ncols2;
    constexpr int cols_per_thread = 2;

    extern __shared__ half2 tile_Q[];
    half2* tile_K = tile_Q;
    half2* tile_V = tile_K + kBatchFa * kStrideKV;
    half* tile_mask = (half*) (tile_V + kBatchFa * kStrideKV);

    tile_h2 Q_B[kD / (2 * tile_h2::J)];
    tile_h2 VKQ_C[kD / (2 * tile_h2::J)];

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX / 2.0f;
    }

    // Q scaled, as halves, through tile_Q into the registers
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int jc0 = 0; jc0 < ncols; jc0 += kWarps) {
        const int jc = jc0 + threadIdx.y;
        const int j = jc / ncols2;
        const int c = jc % ncols2;
        if (jt * ncols1 + j < int(ne01.z) && zt_gqa * ncols2 + c < gqa_ratio) {
#pragma unroll
            for (int k0 = 0; k0 < kD2; k0 += 32) {
                const int k = k0 + threadIdx.x;
                const float2 tmp = Q_f2[(jt * ncols1 + j) * stride_Q1 + c * stride_Q2 + k];
                tile_Q[jc * kStrideQ + k] = scale_h2 * make_half2(tmp.x, tmp.y);
            }
        } else {
#pragma unroll
            for (int k0 = 0; k0 < kD2; k0 += 32) {
                const int k = k0 + threadIdx.x;
                tile_Q[jc * kStrideQ + k] = make_half2(0.0f, 0.0f);
            }
        }
    }

    __syncthreads();

    {
        const int j0 = threadIdx.y * kColsPerWarp;
#pragma unroll
        for (int k0 = 0; k0 < kD2; k0 += tile_h2::J) {
            load_ldmatrix(Q_B[k0 / tile_h2::J], tile_Q + j0 * kStrideQ + k0, kStrideQ);
        }
    }

    __syncthreads();

    int kb0 = kb0_start;

    // the first step's mask and keys
    load_mask<ncols1>(tile_mask, stride_mask, kb0 * kBatchFa, jt * ncols1, ne01);
    load_tile(K_h2, tile_K, stride_K, kb0 * kBatchFa);

    // kb0_start < kb0_stop always, so the last step runs unconditionally
    for (; kb0 < kb0_stop - 1; ++kb0) {
        iter<ncols1, ncols2, false>(K_h2, V_h2, slope, ne01, stride_K, stride_V, stride_mask, tile_K, tile_V,
                                    tile_mask, Q_B, VKQ_C, KQ_max, KQ_rowsum, jt, kb0);
    }
    iter<ncols1, ncols2, true>(K_h2, V_h2, slope, ne01, stride_K, stride_V, stride_mask, tile_K, tile_V, tile_mask,
                               Q_B, VKQ_C, KQ_max, KQ_rowsum, jt, kb0);

    // the steps end without a barrier, and the combining below reuses their shared memory
    __syncthreads();

    // a query's partial row sums are spread across 4 threads
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
        for (int offset = 2; offset >= 1; offset >>= 1) {
            KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, 32);
        }
    }

    // the max and sum of each query in the padding of its tile_Q row (jc_cwm: jc combine write meta), and in
    // dstk_meta for merging blocks of keys
    const int jc_cwm = threadIdx.y * kColsPerWarp + tile_h2::get_i(threadIdx.x % 4);
    const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
    const bool thread_should_write = threadIdx.x % 4 < cols_per_thread;

    __syncthreads();

    if (thread_should_write) {
        ((float2*) tile_Q)[jc_cwm * (kStrideCombine / 2) + kBatchCombine / 2] = KQ_cmr;
    }

    if (thread_should_write && dstk_meta) {
        const int j = jc_cwm / ncols2, c = jc_cwm % ncols2;
        if (jt * ncols1 + j < int(ne01.z) && zt_gqa * ncols2 + c < gqa_ratio) {
            dstk_meta[(jt * ncols1 + j) * ne02 + c] = KQ_cmr;
        }
    }

    __syncthreads();

    // the results through shared memory, column-major, then each divided by its row sum on the way out
    {
        const int j0 = threadIdx.y * kColsPerWarp;
#pragma unroll
        for (int k1 = 0; k1 < kBatchCombine; k1 += tile_h2::J) {
#pragma unroll
            for (int l = 0; l < tile_h2::ne; ++l) {
                const int j = j0 + tile_h2::get_i(l);
                const int k = k1 + tile_h2::get_j(l);

                tile_Q[j * kStrideCombine + k] = VKQ_C[k1 / tile_h2::J].x[l];
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += kWarps) {
        const int jc_dst = jc0_dst + threadIdx.y;

        const int j_dst = jc_dst / ncols2;
        const int c_dst = jc_dst % ncols2;

        if (jt * ncols1 + j_dst >= int(ne01.z) || zt_gqa * ncols2 + c_dst >= gqa_ratio) {
            continue;
        }

        const float* meta_j = (const float*) tile_Q + jc_dst * kStrideCombine + kBatchCombine;
#pragma unroll
        for (int k0 = 0; k0 < kBatchCombine; k0 += 32) {
            const int k = k0 + threadIdx.x;

            // one warp a query (np == 1): its part is weighed by 1
            float2 dstk_val = make_float2(0.0f, 0.0f);
            const float KQ_crs = 1.0f;
            const float2 dstk_val_add = __half22float2(tile_Q[jc_dst * kStrideCombine + k]);
            dstk_val.x += dstk_val_add.x * KQ_crs;
            dstk_val.y += dstk_val_add.y * KQ_crs;

            const float KQ_rowsum_j = meta_j[1];
            dstk_val.x /= KQ_rowsum_j;
            dstk_val.y /= KQ_rowsum_j;

            dstk[((jt * ncols1 + j_dst) * ne02 + c_dst) * kD2 + k] = dstk_val;
        }
    }
}

#endif   // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

// fattn_mma.cuh's flash_attn_ext_f16() with its parameters, for this one case.  The launch gives each tile of
// ncols1 tokens and a KV head's ncols2 query heads (gqa_ratio == ncols2) a CUDA block of its own, blockIdx.x =
// KV head * tiles + tile, over all of the keys: the original's stream-k split of a tile between blocks never happens
// then, and its other paths (a mask read from memory, sinks, ALiBi, sequences) are not taken.  KV_max: the last step
// of keys a tile reads, or null for all of them.  nb31: twice the distance query j's last key lies beyond j
// (change 1).
template <int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool V_is_K_view, bool use_sparse>
__launch_bounds__(kThreads, kOccupancy)
static __global__ void flash_attn_ext_f16(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    static_assert(DKQ == kD && DV == kD && ncols1 * ncols2 == kCols && ncols2 > 1 && !use_logit_softcap &&
                  !V_is_K_view && !use_sparse, "fattn_mma_d256.cuh has the kernel for heads of 256 and 64 query rows only");
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    const char * __restrict__ Q      = Q_ptr;
    const char * __restrict__ K      = K_ptr;
    const char * __restrict__ V      = V_ptr;
    const int  * __restrict__ KV_max = KV_max_ptr;
    float      * __restrict__ dst    = dst_ptr;
    float2     * __restrict__ dst_meta = dst_meta_ptr;

    const int gqa_ratio = ne02 / ne12;   // query heads a KV head

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);
    const int stride_V    = nb21 / sizeof(half2);

    const int iter_k = (ne11   + (kBatchFa - 1)) / kBatchFa;
    const int iter_j = (ne01.z + (ncols1   - 1)) / ncols1;

    // z_KV: the KV head, jt: the tile of tokens, zt_gqa: the tile of the KV head's query heads (one of them)
    const int z_KV   = blockIdx.x / iter_j;
    const int jt     = blockIdx.x % iter_j;
    const int zt_gqa = 0;

    const int zt_Q = z_KV * gqa_ratio + zt_gqa * ncols2;   // the tile's first query head

    const float2 * Q_f2 = (const float2 *) (Q + nb02 * zt_Q);
    const half2  * K_h2 = (const half2  *) (K + nb12 * z_KV);
    const half2  * V_h2 = (const half2  *) (V + nb22 * z_KV);
    float2       * dstk = ((float2 *) dst) + zt_Q * (DV / 2);
    float2  * dstk_meta = dst_meta ? dst_meta + zt_Q : nullptr;

    const float slope = 1.0f;   // no ALiBi with ncols2 > 1

    int kb0_stop = iter_k;
    if (KV_max) {
        kb0_stop = min(kb0_stop, KV_max[jt] / kBatchFa);
    }
    process_tile<ncols1, ncols2>(Q_f2, K_h2, V_h2, dstk, dstk_meta, scale, slope, ne01, ne02, gqa_ratio, stride_Q1,
                                 stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, 0, kb0_stop);
#else
    __trap();
#endif
}

}  // namespace strata::poly::fattn_d256

#undef STRATA_FATTN_D256_MAX_OFFSET
