// src/kernels/cuda/i8_mma.cu - the int8 tensor-core prompt products (gemm_iq_i8_*, declared in
// include/strata/kernels/iq_kernels.hpp; the kernel is i8_mma.cuh).  A file of their own so that a change to them
// builds without iq_kernels.cu.  The weight sub-blocks are decoded by i8_expert.cuh, which this file includes as
// iq_kernels.cu does, in an anonymous namespace (only what i8_mma.cuh instantiates of it is compiled here).
// CUDA only: a HIP build compiles this file empty and takes iq_kernels.cu's stubs.
#if !defined(__HIPCC__)
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <cstdint>
#include <cstdio>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <type_traits>

#define ggml_cuda_dp4a(a, b, c) STRATA_DP4A((a), (b), (c))

namespace strata::kernels {
namespace {
#include "i8_expert.cuh"
#include "i8_mma.cuh"

// The types this kernel takes only once a model's engine has asked for them (gemm_iq_i8_allow()): IQ1_S and MXFP4,
// whose prompt products went to the FP16 kernel before, so that another model's stay as they were
constexpr bool i8m_asked_type(int t) { return t == 19 || t == 39; }
std::atomic<uint64_t> i8m_allowed{0};   // bit t: type t asked for

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
}  // namespace

// K whole stages of 256, or for a type of 32-value blocks whole blocks: the kernel leaves out what lies past K in its
// last stage (Gemma 4 26B's down, K = 704: Q8_0, Q5_1, IQ4_NL, Q4_0; Inkling 2 Mini's, K = 1,152: MXFP4); NVFP4 whole blocks of
// 64 (Inkling 2 Mini's down, K = 1,152)
bool gemm_iq_i8_takes(int ty, int K) {
    if (i8m_asked_type(ty) && !((i8m_allowed.load(std::memory_order_relaxed) >> ty) & 1u)) return false;
    const int step = ty == 40 ? 64 : ty == 39 || i8_block_vals(ty) == 32 ? 32 : 256;
    return i8m_type(ty) && K > 0 && K % step == 0 && i8m_card().on;
}

void gemm_iq_i8_turing(bool on) { i8m_turing.store(on, std::memory_order_relaxed); }
void gemm_iq_i8_allow(int ty, bool on) {
    if (!i8m_asked_type(ty)) return;
    if (on) i8m_allowed.fetch_or(uint64_t{1} << ty, std::memory_order_relaxed);
    else i8m_allowed.fetch_and(~(uint64_t{1} << ty), std::memory_order_relaxed);
}

void gemm_iq_i8_quantize(const float* x, const int32_t* ids, int64_t ld, int64_t rows, int K, void* xq, void* stream) {
    if (rows <= 0) return;
    float* d = (float*) ((char*) xq + (size_t) rows * (size_t) K);
    const unsigned gx = (unsigned) (rows < 32768 ? rows : 32768);   // a block a row, over the grid's two sides
    const dim3 grid(gx, (unsigned) ((rows + gx - 1) / gx));
    if ((uintptr_t) x % 16 == 0 && ld % 4 == 0)
        i8m_quantize4_kernel<<<grid, I8M_NT, 0, (cudaStream_t) stream>>>(x, ids, (long long) ld, K, (long long) rows, (uint32_t*) xq, d,
                                                                         d + (size_t) rows * (size_t) (K / 32));
    else
        i8m_quantize_kernel<<<grid, I8M_NT, 0, (cudaStream_t) stream>>>(x, ids, (long long) ld, K, (long long) rows, (uint32_t*) xq, d,
                                                                        d + (size_t) rows * (size_t) (K / 32));
    check("gemm_iq_i8_quantize");
}

bool gemm_iq_i8_grouped(int ty, const void* W, size_t mat_bytes, int n_out, int K, const void* xq, int64_t total_rows,
                        const int32_t* bounds, int n_mats, int max_rows, const int32_t* ids, float* Y, int64_t ldy, void* stream,
                        const int32_t* from, const int64_t* at) {
    if (!gemm_iq_i8_takes(ty, K) || n_out <= 0 || n_mats <= 0 || max_rows <= 0 || total_rows <= 0) return false;
    const I8mCard card = i8m_card();
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    const size_t row_bytes = iq_row_bytes(ty, K);
    const uint8_t* X = (const uint8_t*) xq;
    const float* XD = (const float*) (X + (size_t) total_rows * (size_t) K);
    const float* XS = XD + (size_t) total_rows * (size_t) (K / 32);
    cudaStream_t s = (cudaStream_t) stream;
    switch (ty) {
#define STRATA_I8M_CASE(T) \
        case T: \
            return i8m_launch_type<T>(card, dev, s, (const uint8_t*) W, mat_bytes, row_bytes, n_out, K, X, XD, XS, bounds, n_mats, max_rows, \
                                      ids, from, (const long long*) at, Y, (long long) ldy);
        STRATA_I8M_TYPES(STRATA_I8M_CASE)
#undef STRATA_I8M_CASE
        default: return false;
    }
}
}  // namespace strata::kernels
#endif
