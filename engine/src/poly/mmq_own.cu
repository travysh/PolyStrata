// src/poly/mmq_own.cu - mmq.hpp's prompt products without llama.cpp, on the kernel library's two grouped kernels
// (src/kernels/cuda).  The weights stay in their GGUF blocks and are decoded into shared memory a tile at a time.
// mmq.cu is the llama.cpp form of the same class (-DSTRATA_POLY_MMQ=ON); one of the two is built.
//
// On Ampere and newer, a type the int8 decoders have and rows of whole stages of 256 values (of whole blocks, for a
// type of 32-value blocks) go to the int8 tensor cores (gemm_iq_i8_grouped, i8_mma.cuh): the activations are int8 with
// a scale per 32 values, MMQ's numbers, and a matrix may have any number of rows.  quantize() and mul() choose by the type and the row length alone, so the rows
// one wrote are the rows the other reads.
//
// Everything else goes to the FP16 kernel (gemm_iq_f16_grouped): the activations FP16 rows, the sums FP32.  What
// differs there from MMQ for the callers:
//   - the activations are halves, not Q8_1 blocks.  A half holds 11 bits from 6e-5 to 65504, and a model's rows are
//     not all in that range (Flint's are far above it, a hidden row behind NVFP4's scales far below), so quantize()
//     multiplies each row by the power of two that puts its largest value near 1024, which is exact, and mul()
//     divides the row's products by it again.  The engine's guard for MMQ's half sums has nothing to do here;
//   - a row's product does not depend on the launch it is in (split_exact()): each output value is its own sum
//     over the row's tiles in their order.
//
// In a HIP build that kernel runs on RDNA3/RDNA4's matrix cores (the same FP16 operands and FP32 sums, in the
// hardware's own order within a 16 x 16 x 16 step), with the tiles that fit their 64 KiB of LDS a block: those of
// 128 values a row, which there take IQ4_NL, NVFP4, Q2_0 and Q8_0 matrices of rows of any whole 128s (Inkling 2
// Mini's experts: IQ4_NL or Q8_0, rows of 2,304 and 1,152).
//
// Where that kernel has no tile (a card before Volta, Turing's 64 KiB of shared memory a block, an AMD card
// without the matrix cores or a tile past its LDS, a matrix whose rows are not whole tiles of 128), each matrix of a launch is dequantized to halves
// (iq_dequant_f16) and multiplied with the same halves of the rows by cuBLAS (hipBLAS / rocBLAS through
// hip_compat), FP32 sums: the rows of a matrix take its bounds as the host keeps them (mirror()).  A row's product
// there depends on how many rows the library multiplies at once (split_exact() is false).  F32 matrices are not
// taken: their halves would be less exact than the products the engine makes of them otherwise.
//
// On an RTX 50 card (STRATA_PREFILL_NVFP4 unset or w4a4x2): an NVFP4 matrix with rows of a multiple of 128 values goes to the
// block-scaled FP4 tensor cores instead, the activations two FP4 terms under a float a row (nvfp4_x2.hpp, about 8
// bits against a half's 11), its products straight to dst's rows and its rows read by index as the int8 ones are.
// NVFP4 takes the int8 kernel only with STRATA_PREFILL_NVFP4=w4a8 (nvfp4_w4a8()), on any card that has it.
#include "mmq.hpp"
#if STRATA_POLY_X2
#include "nvfp4_x2.hpp"
#endif

#include "strata/kernels/iq_kernels.hpp"
#if defined(GGML_USE_HIP) || defined(STRATA_USE_HIP) || defined(__HIPCC__)
#include "strata/kernels/gfx_arch.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

#if defined(GGML_USE_HIP) || defined(STRATA_USE_HIP) || defined(__HIPCC__)
#define STRATA_MMQ_OWN_HIP 1
#endif

namespace strata::poly {
namespace {

constexpr int kTileRows = 128;        // gemm_iq_f16_grouped's WG_BN: weight rows a block
constexpr float kHalfMax = 65504.f;   // the largest finite half

// The types iq_kernels.cu's dq_dispatch dequantizes for the grouped kernel (iq_dequant_supported()) less the ones
// it takes only in its own builds: the i-quants, the K-quants from Q3_K, the older 32-value types, NVFP4, Q2_0, and
// strata-poly's F16, BF16, Q2_K, IQ1_S and MXFP4.
bool dequantized(int t) {
    switch (t) {
        case 2: case 3: case 6: case 7: case 8:                       // Q4_0 Q4_1 Q5_0 Q5_1 Q8_0
        case 10: case 11: case 12: case 13: case 14:                  // Q2_K Q3_K Q4_K Q5_K Q6_K
        case 16: case 17: case 18: case 19: case 20: case 21: case 22: case 23: case 29:   // the i-quants
        case 39: case 40: case 42:                                    // MXFP4 NVFP4 Q2_0
        case 1: case 30:                                              // F16 BF16
            return true;
        default:
            return false;
    }
}

// values of a block of the type: a row of the matrix is whole blocks
int block_values(int t) {
    switch (t) {
        case 1: case 30: return 1;
        case 2: case 3: case 6: case 7: case 8: case 20: case 39: return 32;
        case 40: case 42: return 64;
        default: return 256;
    }
}

// values of a row the kernel reads at a time: whole superblocks of 256, or blocks of 128 for the two types that have them
// NVFP4 rows of whole 128s and not 256s (Inkling 2's downs, K = 1,152) go in 128-value tiles, decoded in shared
// memory as the 256-value ones are, where they went to a dequantized copy and cuBLAS; STRATA_POLY_NV4_TILE=0: the copy
bool nv4_off() {
    static const bool off = [] {
        const char* v = std::getenv("STRATA_POLY_NV4_TILE");
        return v && v[0] == '0';
    }();
    return off;
}

#ifndef STRATA_MMQ_OWN_HIP
bool whole_tiles(int t, int64_t cols) {
    if (t == 40 && cols % 256 != 0 && nv4_off()) return false;
    if (t == 42 || ((t == 20 || t == 40) && cols % 256 != 0)) return cols % 128 == 0;
    return cols % 256 == 0;
}

// shared memory of a block (the kernel's own sum: BN weight rows of WK + 8 halves, BM = 64 rows of XK + 8 = 136)
size_t tile_bytes(int t, int64_t cols) {
    const int wk = (t == 42 || ((t == 20 || t == 40) && cols % 256 != 0)) ? 128 : 256;
    return (size_t) kTileRows * (wk + 8) * 2 + (size_t) 64 * (128 + 8) * 2;
}
#else
// RDNA: Q2_0, IQ4_NL, NVFP4 and Q8_0 in the 128-value tiles whatever the row length (gemm_iq_f16_grouped's HIP lanes);
// the other types' 256-value tiles need more LDS than a block has
bool lane128(int t) { return t == 42 || t == 20 || t == 40 || t == 8; }
bool whole_tiles(int t, int64_t cols) {
    if (t == 40 && cols % 256 != 0 && nv4_off()) return false;
    return lane128(t) ? cols % 128 == 0 : cols % 256 == 0;
}
size_t tile_bytes(int t, int64_t cols) {
    (void) cols;
    return (size_t) kTileRows * ((lane128(t) ? 128 : 256) + 8) * 2 + (size_t) 64 * (128 + 8) * 2;
}
#endif

#ifdef STRATA_MMQ_OWN_HIP
// STRATA_POLY_HIP_GROUPED: the tiles on AMD's matrix cores.  Unset: on for gfx11 (RDNA3), off for gfx12 (RDNA4);
// 1 / true / yes / on: on for both; any other value (0, off, ...): off for both (the BLAS loop).  RX 7900 XTX, Inkling 2 Mini IQ4_NL, every expert in VRAM:
// prompts 39-65% faster.  RX 9070 XT (2,240 of 3,072 experts in VRAM, a noisy node): no clear gain, so gfx12 stays
// opt-in until a clean run; rocBLAS there hands these GEMMs to hipBLASLt.
bool hip_grouped(bool gfx12) {
    static const int v = [] {
        const char* e = std::getenv("STRATA_POLY_HIP_GROUPED");
        if (!e || !*e) return -1;
        std::string v(e);
        for (char& ch : v) ch = (char) std::tolower((unsigned char) ch);
        return v == "1" || v == "true" || v == "yes" || v == "on" ? 1 : 0;
    }();
    return v < 0 ? !gfx12 : v == 1;
}
#endif

struct Card {
    int major = 0;
    size_t shared = 0;
};

// the current card's (a layer split asks for each of its cards): read once a card
Card card() {
    static Card known[64];
    static bool read[64];
    int dev = 0, v = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return Card{};
    if (!read[dev]) {
        Card c;
#ifndef STRATA_MMQ_OWN_HIP
        if (cudaDeviceGetAttribute(&v, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess) c.major = v;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) == cudaSuccess) c.shared = (size_t) v;
#else
        // an AMD card with the matrix cores the kernel library's tiles are built for, where they are on
        // (hip_grouped()): its LDS a block (major stays 0)
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
            ((kernels::gfx_arch_is_gfx11_wmma(p.gcnArchName) && hip_grouped(false)) ||
             (kernels::gfx_arch_is_gfx12_wmma(p.gcnArchName) && hip_grouped(true))))
            c.shared = (size_t) p.sharedMemPerBlock;
#endif
        (void) v;
        known[dev] = c;
        read[dev] = true;
    }
    return known[dev];
}

// STRATA_POLY_PROMPT_BLAS=1: every product by the BLAS path, as on Turing and on an AMD card without the tiles: how that path is
// checked on a card that has the tiles
bool blas_only() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_PROMPT_BLAS");
        return v && v[0] == '1';
    }();
    return on;
}

// gemm_iq_f16_grouped has a tile for the matrix on this card
bool grouped(int t, int64_t rows, int64_t cols) {
    const Card c = card();
#ifdef STRATA_MMQ_OWN_HIP
    return !blas_only() && c.shared > 0 && dequantized(t) && rows % kTileRows == 0 && whole_tiles(t, cols) &&
           tile_bytes(t, cols) <= c.shared;
#else
    return !blas_only() && c.major >= 7 && dequantized(t) && rows % kTileRows == 0 && whole_tiles(t, cols) &&
           tile_bytes(t, cols) <= c.shared;
#endif
}

// a matrix of the type is dequantized whole to halves and multiplied by the BLAS library (the dequantizer reads
// the matrix as one run of values, 256 at a time)
bool by_blas(int t, int64_t rows, int64_t cols) {
    return dequantized(t) && cols % block_values(t) == 0 && (rows * cols) % 256 == 0;
}

// The two-term FP4 products for an NVFP4 matrix of rows this long, on a card that has them: the default, and what
// STRATA_PREFILL_NVFP4=w4a4x2 names; w4a8 asks for the int8 kernel (int8_rows()) and any other value (fp16, w4a4,
// off) keeps the halves.  Inkling 2 Mini UD-Q4_K_XL on the RTX 5090, following the halves' greedy answers (8 prompts of 19 to 10,124 tokens, 256 tokens each):
// the same most likely token at 98.5% of the steps (95.3% to 100% a prompt), against 100% for the halves run again.
bool x2_rows(int t, int64_t cols) {
#if STRATA_POLY_X2
    static const bool asked = [] {
        const char* v = std::getenv("STRATA_PREFILL_NVFP4");
        return !v || !*v || std::string(v) == "w4a4x2";
    }();
    return t == 40 && cols % x2::kK == 0 && asked && nvfp4_x2_built();
#else
    (void) t, (void) cols;
    return false;
#endif
}

// a block a row; the rows go over the grid's two sides, which hold 65,535 each
dim3 rows_grid(int64_t rows) {
    const unsigned gx = (unsigned) (rows < 32768 ? rows : 32768);
    return dim3(gx, (unsigned) ((rows + gx - 1) / gx));
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("strata-poly prompt product: ") + what + ": " + cudaGetErrorString(e));
}

// Row i of xq = row ids[i] of x (row i without ids) as halves, times a power of two f that puts the row's largest
// value in [1024, 2048): the multiplication is exact, so what is lost is a half's own rounding and nothing to the
// row's size.  back[i] = 1 / f, which mul() multiplies the row's products by.  A row of zeros, or one with a value
// that is not finite, is left as it is (and a value beyond a half's range saturated).
__global__ void to_halves(const float* __restrict__ x, const int32_t* __restrict__ ids, __half* __restrict__ xq, int cols,
                          int64_t ld, int64_t rows, float* __restrict__ back) {
    const int64_t row = (int64_t) blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows) return;
    const float* src = x + (ids ? (int64_t) ids[row] : row) * ld;
    __shared__ float most[256];
    float m = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, fabsf(src[c]));   // (a NaN is passed over by fmaxf)
    most[threadIdx.x] = m;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (threadIdx.x < step) most[threadIdx.x] = fmaxf(most[threadIdx.x], most[threadIdx.x + step]);
        __syncthreads();
    }
    int e = 11;
    if (most[0] > 0.f && most[0] <= 3.0e38f) {
        frexpf(most[0], &e);                      // most = m * 2^e, m in [0.5, 1)
        e = max(-100, min(127, e));
    }
    const float f = ldexpf(1.f, 11 - e);
    __half* dst = xq + row * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        dst[c] = __float2half(fminf(fmaxf(src[c] * f, -kHalfMax), kHalfMax));
    if (threadIdx.x == 0) back[row] = ldexpf(1.f, e - 11);
}

// Row ids[i] of dst (row i without ids) = row i of y times back[i], for the rows of this launch's matrices only:
// [bounds[0], bounds[n]).  The other rows of y are an earlier launch's (another run of experts, or a mixed layer's
// other half), and dst's are not this launch's to write.  y may be dst itself where no row moves.
__global__ void finish_rows(const float* __restrict__ y, int64_t ldy, const int32_t* __restrict__ ids,
                            const float* __restrict__ back, float* dst, int64_t ld_dst, int cols, int64_t rows,
                            const int32_t* __restrict__ bounds, int n) {
    const int64_t row = (int64_t) blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows || row < bounds[0] || row >= bounds[n]) return;
    const float* src = y + row * ldy;
    float* out = dst + (ids ? (int64_t) ids[row] : row) * ld_dst;
    const float b = back[row];
    for (int c = threadIdx.x; c < cols; c += blockDim.x) out[c] = src[c] * b;
}

// STRATA_PREFILL_NVFP4=w4a8: an NVFP4 matrix goes to the int8 kernel where the card has it, its weights exact as int8
// codes times a float a 16 values and the activations int8 with a scale per 32 values, as every other type there (and
// as llama.cpp's NVFP4 tiles take them off RTX 50).  Otherwise NVFP4 keeps the halves (and RTX 50 its FP4 products):
// the int8 activations keep about 8 bits of a value where the halves keep 11.
bool nvfp4_w4a8() {
    static const bool asked = [] {
        const char* v = std::getenv("STRATA_PREFILL_NVFP4");
        return v && std::string(v) == "w4a8";
    }();
    return asked;
}

// the int8 kernel takes a matrix of this type with rows this long, whatever their number
bool int8_rows(int t, int64_t cols) {
    return !blas_only() && cols <= 0x7fffffff && (t != 40 || nvfp4_w4a8()) && kernels::gemm_iq_i8_takes(t, (int) cols);
}

// the rows' halves, then a float a row (back)
size_t halves_bytes(int64_t rows, int64_t cols) { return ((size_t) rows * (size_t) cols * 2 + 3) / 4 * 4; }

// One prompt path's buffers on the card: where a single matrix's two bounds lie, the products of a launch whose
// rows go elsewhere, and the BLAS path's handle and a matrix's halves.  The last three are the caller's when it
// gave them (give()), and this path's own when it had to make them.
struct Ctx {
    int32_t* bounds = nullptr;
    int32_t bounds_host[2] = {0, 0};
    float* y = nullptr;
    size_t y_floats = 0;
    cublasHandle_t blas = nullptr;
    __half* dq = nullptr;
    size_t dq_halves = 0;
    bool own_y = true, own_dq = true, own_blas = true;
};

// a buffer of the card grown to `bytes`; the launches before may still read the old one
template<class T>
void grow(T*& p, size_t& have, bool& own, size_t need, cudaStream_t s, const char* what) {
    if (need <= have) return;
    ck(cudaStreamSynchronize(s), "waiting for the launches before");
    if (own) cudaFree(p);
    p = nullptr;
    have = 0;
    own = true;
    ck(cudaMalloc(&p, need * sizeof(T)), what);
    have = need;
}

}  // namespace

Mmq::Mmq() {
    Ctx* c = new Ctx;
    if (cudaMalloc(&c->bounds, 2 * sizeof(int32_t)) != cudaSuccess) {
        delete c;
        throw std::runtime_error("strata-poly prompt product: no memory on the card for two numbers");
    }
    ctx_ = c;
}

Mmq::~Mmq() {
    Ctx* c = (Ctx*) ctx_;
    cudaFree(c->bounds);
    if (c->own_y) cudaFree(c->y);
    if (c->own_dq) cudaFree(c->dq);
    if (c->blas && c->own_blas) cublasDestroy(c->blas);
    delete c;
}

void Mmq::mul_at(const void* w, const int64_t* at, int ggml_type, int64_t w_rows, int64_t w_cols, int n, const void* xq,
                 const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
                 int64_t ld_dst, void* stream, const int32_t* from) {
    if (n <= 0 || total_rows <= 0) return;
    if (!int8_rows(ggml_type, w_cols) ||
        !kernels::gemm_iq_i8_grouped(ggml_type, w, 0, (int) w_rows, (int) w_cols, xq, total_rows, bounds, n, (int) max_rows,
                                     ids, dst, ld_dst, stream, from, at))
        throw std::runtime_error("strata-poly prompt product: the int8 kernel did not start for tensor type " +
                                 std::to_string(ggml_type) + " with [" + std::to_string(w_rows) + ", " +
                                 std::to_string(w_cols) + "] at offsets");
}

Mmq::Work Mmq::work(int ggml_type, int64_t w_rows, int64_t w_cols, int64_t rows, bool by_ids, int64_t ld_dst) {
    Work w;
    // int8 and FP4: straight to dst's rows
    if (!fits(ggml_type, w_rows, w_cols) || int8_rows(ggml_type, w_cols) || x2_rows(ggml_type, w_cols)) return w;
    if (by_ids || ld_dst % 4 != 0) w.floats = (size_t) rows * (size_t) ((w_rows + 3) / 4 * 4);
    if (!grouped(ggml_type, w_rows, w_cols)) w.halves = (size_t) w_rows * (size_t) w_cols;
    return w;
}

void Mmq::give(float* y, size_t floats, void* dq, size_t halves, void* blas) {
    Ctx& c = *(Ctx*) ctx_;
    if (c.own_y) cudaFree(c.y);
    if (c.own_dq) cudaFree(c.dq);
    if (c.blas && c.own_blas) cublasDestroy(c.blas);
    c.y = y;
    c.y_floats = y ? floats : 0;
    c.dq = (__half*) dq;
    c.dq_halves = dq ? halves : 0;
    c.blas = (cublasHandle_t) blas;
    c.own_y = c.own_dq = c.own_blas = false;
}

bool Mmq::fits(int ggml_type, int64_t w_rows, int64_t w_cols) {
    return x2_rows(ggml_type, w_cols) || int8_rows(ggml_type, w_cols) || grouped(ggml_type, w_rows, w_cols) ||
           by_blas(ggml_type, w_rows, w_cols);
}

size_t Mmq::q8_bytes(int64_t rows, int64_t cols) {
    return (halves_bytes(rows, cols) + (size_t) rows * sizeof(float) + 255) / 256 * 256;
}

bool Mmq::half_sums(int) { return false; }

bool Mmq::split_exact(int ggml_type, int64_t w_rows) {
    (void) w_rows;
    // (by the type: a call's rows are long)
    return x2_rows(ggml_type, 256) || int8_rows(ggml_type, 256) || grouped(ggml_type, w_rows, 256);
}

bool Mmq::reads_by_index(int ggml_type, int64_t w_cols) {
    return int8_rows(ggml_type, w_cols) || x2_rows(ggml_type, w_cols);
}

bool Mmq::int8(int ggml_type, int64_t w_cols) { return int8_rows(ggml_type, w_cols); }

void Mmq::quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows,
                   void* stream) {
    if (rows <= 0) return;
    if (int8_rows(ggml_type, cols)) return kernels::gemm_iq_i8_quantize(x, ids, ld, rows, (int) cols, xq, stream);
#if STRATA_POLY_X2
    if (x2_rows(ggml_type, cols)) return nvfp4_x2_quantize(x, ids, xq, cols, ld, rows, stream);   // (fewer bytes)
#endif
    float* back = (float*) ((char*) xq + halves_bytes(rows, cols));
    to_halves<<<rows_grid(rows), 256, 0, (cudaStream_t) stream>>>(x, ids, (__half*) xq, (int) cols, ld, rows, back);
    ck(cudaGetLastError(), "the rows as halves");
}

void Mmq::mul(const void* w, int ggml_type, int64_t w_rows, int64_t w_cols, size_t stride, int n, const void* xq,
              const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
              int64_t ld_dst, void* stream, const int32_t* from) {
    if (n <= 0 || total_rows <= 0) return;
#if STRATA_POLY_X2
    if (x2_rows(ggml_type, w_cols))
        return nvfp4_x2_mul(w, w_rows, w_cols, stride, n, xq, bounds, ids, total_rows, max_rows, dst, ld_dst, stream,
                            from);
#endif
    Ctx& c = *(Ctx*) ctx_;
    const cudaStream_t s = (cudaStream_t) stream;
    const int32_t* host_bounds = c.bounds_host;   // the BLAS path's, on the host
    if (!bounds) {   // one matrix takes every row
        if (n != 1) throw std::runtime_error("strata-poly prompt product: several matrices need their bounds");
        c.bounds_host[0] = 0;
        c.bounds_host[1] = (int32_t) total_rows;
        ck(cudaMemcpyAsync(c.bounds, c.bounds_host, sizeof c.bounds_host, cudaMemcpyHostToDevice, s), "a matrix's bounds");
        bounds = c.bounds;
    } else {
        const char* d = (const char*) bounds;
        const char* lo = (const char*) mirror_dev_;
        if (lo && d >= lo && d + (size_t) (n + 1) * sizeof(int32_t) <= lo + mirror_bytes_)
            host_bounds = (const int32_t*) ((const char*) mirror_host_ + (d - lo));
        else
            host_bounds = nullptr;
    }
    if (int8_rows(ggml_type, w_cols)) {   // straight to dst's rows
        if (!kernels::gemm_iq_i8_grouped(ggml_type, w, stride, (int) w_rows, (int) w_cols, xq, total_rows, bounds, n,
                                         (int) max_rows, ids, dst, ld_dst, stream, from))
            throw std::runtime_error("strata-poly prompt product: the int8 kernel did not start for tensor type " +
                                     std::to_string(ggml_type) + " with [" + std::to_string(w_rows) + ", " +
                                     std::to_string(w_cols) + "]");
        return;
    }
    if (from) throw std::logic_error("strata-poly prompt product: rows by index need the int8 kernel");
    // straight into dst where row i is row i (finish_rows then works in place); else by way of a buffer of this path's own
    float* y = dst;
    int64_t ldy = ld_dst;
    if (ids || ld_dst % 4 != 0) {
        ldy = (w_rows + 3) / 4 * 4;
        grow(c.y, c.y_floats, c.own_y, (size_t) total_rows * (size_t) ldy, s, "memory for a launch's products");
        y = c.y;
    }
    if (grouped(ggml_type, w_rows, w_cols)) {
        if (!kernels::gemm_iq_f16_grouped(ggml_type, w, stride, (int) w_rows, (int) w_cols, xq, (int) w_cols, bounds, n,
                                          (int) max_rows, y, (int) ldy, stream))
            throw std::runtime_error("strata-poly prompt product: no kernel for tensor type " + std::to_string(ggml_type) +
                                     " with [" + std::to_string(w_rows) + ", " + std::to_string(w_cols) + "]");
    } else {
        // The BLAS path: each matrix with rows, dequantized and multiplied on its own: Y[r0, r1) = X[r0, r1) W^T,
        // as column-major C (w_rows x r) = W^T (row-major W read transposed) times X^T.
        if (!by_blas(ggml_type, w_rows, w_cols))
            throw std::runtime_error("strata-poly prompt product: no product for tensor type " + std::to_string(ggml_type) +
                                     " with [" + std::to_string(w_rows) + ", " + std::to_string(w_cols) + "]");
        if (!host_bounds)
            throw std::logic_error("strata-poly prompt product: the matrices' bounds are not where mirror() says");
        if (!c.blas) {
            if (cublasCreate(&c.blas) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("strata-poly prompt product: the BLAS library did not start");
            c.own_blas = true;
        }
        if (cublasSetStream(c.blas, s) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("strata-poly prompt product: the BLAS library took no stream");
        const int64_t values = w_rows * w_cols;
        grow(c.dq, c.dq_halves, c.own_dq, (size_t) values, s, "memory for a matrix's halves");
        const __half* x = (const __half*) xq;
        const float one = 1.f, zero = 0.f;
        for (int e = 0; e < n; ++e) {
            const int64_t r0 = host_bounds[e], r = host_bounds[e + 1] - r0;
            if (r <= 0) continue;
            kernels::iq_dequant_f16(ggml_type, (const char*) w + (size_t) e * stride, values, (uint16_t*) c.dq, stream);
            if (cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) w_rows, (int) r, (int) w_cols, &one, c.dq, CUDA_R_16F,
                             (int) w_cols, x + r0 * w_cols, CUDA_R_16F, (int) w_cols, &zero, y + r0 * ldy, CUDA_R_32F,
                             (int) ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("strata-poly prompt product: the BLAS product failed for tensor type " +
                                         std::to_string(ggml_type));
        }
    }
    const float* back = (const float*) ((const char*) xq + halves_bytes(total_rows, w_cols));
    finish_rows<<<rows_grid(total_rows), 256, 0, s>>>(y, ldy, ids, back, dst, ld_dst, (int) w_rows, total_rows, bounds, n);
    ck(cudaGetLastError(), "the products to their rows");
}

}  // namespace strata::poly
