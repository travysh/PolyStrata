// src/poly/inkling2_zha.cu - the ZHA index of Inkling 2's full layers (STRATA_INK_ZHA, docs/INKLING_ZHA.md): the
// summaries of 64-key chunks, their scores, the chunks a token reads exactly, the tail through the summaries, and the
// answer's chunk cache in VRAM.  The method: "ZHA" (Zyphra Hierarchical Attention), arXiv 2610.42112.
//
// The kernels use mma.sync and cp.async: NVIDIA only (compute capability 7.5 and up; the load refuses the index
// elsewhere).  A HIP build gets entry points that say so.
#include "inkling2_zha.hpp"

#include <stdexcept>

#if defined(__HIPCC__) || defined(STRATA_USE_HIP)

namespace strata::poly::ink {
namespace {
[[noreturn]] void nvidia_only() { throw std::runtime_error("the ZHA index (STRATA_INK_ZHA) runs on NVIDIA cards only"); }
}  // namespace
void store_q4(const float*, int8_t*, uint16_t*, int, long long, int, long long, void*) { nvidia_only(); }
void q4_to_q8(const int8_t*, const uint16_t*, long long, int8_t*, uint16_t*, long long, long long, int, int, void*) {
    nvidia_only();
}
void zha_cache_clear(ZhaCache&, int, int, void*) { nvidia_only(); }
void zha_build(const float*, const ZhaKv&, long long, int, int, int, int, int, int, uint16_t*, uint16_t*, float*, void*,
               size_t) {
    nvidia_only();
}
void zha_attend(void*, const Zha&, const float*, const ZhaKv&, const ZhaKv&, long long, const uint16_t*, const uint16_t*,
                const float*, int, int, long long, int, int, int, const float*, const float*, int, int, const float*,
                const ZhaWork&, float*, void*, ZhaCache*, bool, int) {
    nvidia_only();
}
}  // namespace strata::poly::ink

#else

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace strata::poly::ink {
namespace {

inline cudaStream_t cs(void* s) { return (cudaStream_t) s; }
inline unsigned grid(size_t n, unsigned b) { return (unsigned) ((n + b - 1) / b); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ---- ZHA: summaries, chunk scores, the attention through them --------------------------------------------------
constexpr int kZhaDim = 128;
// an answer's chunk the cache has just taken (zha_assign*): its place in cmap with this bit, until the fill is done
constexpr int kZhaMiss = 1 << 30;

__device__ __forceinline__ void load4h(const __half* p, float (&x)[4]) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const __half2 a = *reinterpret_cast<const __half2*>(&u.x), b = *reinterpret_cast<const __half2*>(&u.y);
    const float2 fa = __half22float2(a), fb = __half22float2(b);
    x[0] = fa.x, x[1] = fa.y, x[2] = fb.x, x[3] = fb.y;
}

__device__ __forceinline__ void store4h(__half* p, const float (&x)[4]) {
    __half2 a = __floats2half2_rn(x[0], x[1]), b = __floats2half2_rn(x[2], x[3]);
    uint2 u;
    u.x = *reinterpret_cast<uint32_t*>(&a);
    u.y = *reinterpret_cast<uint32_t*>(&b);
    *reinterpret_cast<uint2*>(p) = u;
}

// Key k's 4 values from `lane` of KV head g (offset g * 128 when the store is a cache's, else 0), and its value's.
__device__ __forceinline__ void zload(const ZhaKv& s, long long k, int g, int lane, float (&kx)[4], float (&vx)[4]) {
    const size_t at = (size_t) k * s.row + (s.headed ? (size_t) g * kZhaDim : 0) + 4 * lane;
    if (s.q4) {   // 4 codes in 2 bytes
        const uint16_t a = *reinterpret_cast<const uint16_t*>(s.kc + at / 2), b = *reinterpret_cast<const uint16_t*>(s.vc + at / 2);
        const float da = __half2float(__ushort_as_half(s.ks[at / 32])), db = __half2float(__ushort_as_half(s.vs[at / 32]));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            kx[i] = (float) ((int) ((a >> (4 * i)) & 15) - 8) * da;
            vx[i] = (float) ((int) ((b >> (4 * i)) & 15) - 8) * db;
        }
    } else if (s.kc) {
        const char4 a = *reinterpret_cast<const char4*>(s.kc + at), b = *reinterpret_cast<const char4*>(s.vc + at);
        const float da = __half2float(__ushort_as_half(s.ks[at / 32])), db = __half2float(__ushort_as_half(s.vs[at / 32]));
        kx[0] = a.x * da, kx[1] = a.y * da, kx[2] = a.z * da, kx[3] = a.w * da;
        vx[0] = b.x * db, vx[1] = b.y * db, vx[2] = b.z * db, vx[3] = b.w * db;
    } else {
        load4h((const __half*) s.k16 + at, kx);
        load4h((const __half*) s.v16 + at, vx);
    }
}

__host__ __device__ __forceinline__ size_t zq_rows(size_t q8n) { return q8n & 0xffffffffull; }
// The summaries' chunks a KV head (their room, a slot's own since it grows with the context): q8n's bits 32-61, or
// max_chunks where they are 0 (fp16 summaries, or a caller that gives none)
__host__ __device__ __forceinline__ int zq_cap(size_t q8n, int max_chunks) {
    const int c = (int) ((q8n >> 32) & 0x3fffffffull);
    return c > 0 ? c : max_chunks;
}
__host__ __device__ __forceinline__ bool zq_4(size_t q8n) { return (q8n & kZhaQ4) != 0; }
__device__ __forceinline__ const uint16_t* zq_scales(const void* codes, size_t q8n) {
    return reinterpret_cast<const uint16_t*>(reinterpret_cast<const int8_t*>(codes) +
                                             zq_rows(q8n) * (zq_4(q8n) ? kZhaDim / 2 : kZhaDim));
}

// Summary row `row` (of khat or vhat: fp16, or with q8n its rows' int8 or 4-bit codes then their fp16 scales a 32
// values).
__device__ __forceinline__ void zsum4(const __half* base, size_t q8n, size_t row, int lane, float (&x)[4]) {
    if (q8n && zq_4(q8n)) {
        const uint8_t* codes = reinterpret_cast<const uint8_t*>(base);
        const uint16_t b2 = *reinterpret_cast<const uint16_t*>(codes + row * (kZhaDim / 2) + 2 * lane);
        const float d = __half2float(__ushort_as_half(zq_scales(base, q8n)[row * 4 + lane / 8]));
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = (float) ((int) ((b2 >> (4 * i)) & 15) - 8) * d;
    } else if (q8n) {
        const int8_t* codes = reinterpret_cast<const int8_t*>(base);
        const uint16_t* sc = reinterpret_cast<const uint16_t*>(codes + q8n * kZhaDim);
        const char4 c = *reinterpret_cast<const char4*>(codes + row * kZhaDim + 4 * lane);
        const float d = __half2float(__ushort_as_half(sc[row * 4 + lane / 8]));
        x[0] = c.x * d, x[1] = c.y * d, x[2] = c.z * d, x[3] = c.w * d;
    } else {
        load4h(base + row * kZhaDim + 4 * lane, x);
    }
}

__device__ __forceinline__ uint4 zsum8h(const __half* base, size_t q8n, size_t row, int d) {   // 8 values as fp16
    if (!q8n) return *reinterpret_cast<const uint4*>(base + row * kZhaDim + d);
    if (zq_4(q8n)) {
        const uint32_t a4 = *reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(base) +
                                                               row * (kZhaDim / 2) + d / 2);
        const float s = __half2float(__ushort_as_half(zq_scales(base, q8n)[row * 4 + d / 32]));
        uint4 out;
        uint32_t* o = &out.x;
        for (int e = 0; e < 4; ++e) {
            __half2 h = __floats2half2_rn((float) ((int) ((a4 >> (8 * e)) & 15) - 8) * s,
                                          (float) ((int) ((a4 >> (8 * e + 4)) & 15) - 8) * s);
            o[e] = *reinterpret_cast<uint32_t*>(&h);
        }
        return out;
    }
    const int8_t* codes = reinterpret_cast<const int8_t*>(base);
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(codes + q8n * kZhaDim);
    const uint2 a = *reinterpret_cast<const uint2*>(codes + row * kZhaDim + d);
    const int8_t* b = reinterpret_cast<const int8_t*>(&a);
    const float s = __half2float(__ushort_as_half(sc[row * 4 + d / 32]));
    uint4 out;
    uint32_t* o = &out.x;
    for (int e = 0; e < 4; ++e) {
        __half2 h = __floats2half2_rn(b[2 * e] * s, b[2 * e + 1] * s);
        o[e] = *reinterpret_cast<uint32_t*>(&h);
    }
    return out;
}

// A launch block a chunk and KV head, a warp a reference at a time, a lane 4 values: the online softmax of the
// reference's scores over the chunk's keys (from the ring), its keys and values summed with those weights.
__global__ void zha_build_kernel(const float* __restrict__ refs, ZhaKv ring, long long n_ring, int c0, int C, int R,
                                 int max_chunks, __half* __restrict__ khat, __half* __restrict__ vhat,
                                 float* __restrict__ bhat, size_t q8n) {
    const int c = c0 + blockIdx.x, g = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const float sc = rsqrtf((float) kZhaDim);
    for (int r = w; r < R; r += blockDim.x >> 5) {
        const float4 qr = *reinterpret_cast<const float4*>(refs + ((size_t) g * R + r) * kZhaDim + 4 * lane);
        float m = -INFINITY, l = 0.f, ak[4] = {0.f, 0.f, 0.f, 0.f}, av[4] = {0.f, 0.f, 0.f, 0.f};
        for (int j = 0; j < C; ++j) {
            float k[4], v[4];
            zload(ring, ((long long) c * C + j) % n_ring, g, lane, k, v);
            const float sv = warp_sum(qr.x * k[0] + qr.y * k[1] + qr.z * k[2] + qr.w * k[3]) * sc;
            const float mn = fmaxf(m, sv), f = expf(m - mn), e = expf(sv - mn);
            l = l * f + e;
#pragma unroll
            for (int i = 0; i < 4; ++i) ak[i] = ak[i] * f + e * k[i], av[i] = av[i] * f + e * v[i];
            m = mn;
        }
        float kh[4], vh[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) kh[i] = ak[i] / l, vh[i] = av[i] / l;
        const float dot = warp_sum(qr.x * kh[0] + qr.y * kh[1] + qr.z * kh[2] + qr.w * kh[3]) * sc;
        const size_t row = ((size_t) g * zq_cap(q8n, max_chunks) + c) * R + r, o = row * kZhaDim + 4 * lane;
        if (q8n && zq_4(q8n)) {   // 4-bit codes, a scale a 32 values (8 lanes)
            float ak2 = fmaxf(fmaxf(fabsf(kh[0]), fabsf(kh[1])), fmaxf(fabsf(kh[2]), fabsf(kh[3])));
            float av2 = fmaxf(fmaxf(fabsf(vh[0]), fabsf(vh[1])), fmaxf(fabsf(vh[2]), fabsf(vh[3])));
            for (int x = 1; x < 8; x <<= 1) {
                ak2 = fmaxf(ak2, __shfl_xor_sync(0xffffffffu, ak2, x));
                av2 = fmaxf(av2, __shfl_xor_sync(0xffffffffu, av2, x));
            }
            const float dk = ak2 / 7.f, dv = av2 / 7.f, ik = dk > 0.f ? 1.f / dk : 0.f, iv = dv > 0.f ? 1.f / dv : 0.f;
            uint16_t ck = 0, cv = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ck |= (uint16_t) (min(7, max(-8, (int) lrintf(kh[i] * ik))) + 8) << (4 * i);
                cv |= (uint16_t) (min(7, max(-8, (int) lrintf(vh[i] * iv))) + 8) << (4 * i);
            }
            uint8_t* kc = reinterpret_cast<uint8_t*>(khat);
            uint8_t* vc = reinterpret_cast<uint8_t*>(vhat);
            *reinterpret_cast<uint16_t*>(kc + row * (kZhaDim / 2) + 2 * lane) = ck;
            *reinterpret_cast<uint16_t*>(vc + row * (kZhaDim / 2) + 2 * lane) = cv;
            if ((lane & 7) == 0) {
                const_cast<uint16_t*>(zq_scales(kc, q8n))[row * 4 + lane / 8] = __half_as_ushort(__float2half(dk));
                const_cast<uint16_t*>(zq_scales(vc, q8n))[row * 4 + lane / 8] = __half_as_ushort(__float2half(dv));
            }
        } else if (q8n) {   // int8 codes, a scale a 32 values (8 lanes)
            float ak2 = fmaxf(fmaxf(fabsf(kh[0]), fabsf(kh[1])), fmaxf(fabsf(kh[2]), fabsf(kh[3])));
            float av2 = fmaxf(fmaxf(fabsf(vh[0]), fabsf(vh[1])), fmaxf(fabsf(vh[2]), fabsf(vh[3])));
            for (int x = 1; x < 8; x <<= 1) {
                ak2 = fmaxf(ak2, __shfl_xor_sync(0xffffffffu, ak2, x));
                av2 = fmaxf(av2, __shfl_xor_sync(0xffffffffu, av2, x));
            }
            const float dk = ak2 / 127.f, dv = av2 / 127.f, ik = dk > 0.f ? 1.f / dk : 0.f, iv = dv > 0.f ? 1.f / dv : 0.f;
            int8_t* kc = reinterpret_cast<int8_t*>(khat);
            int8_t* vc = reinterpret_cast<int8_t*>(vhat);
            *reinterpret_cast<char4*>(kc + o) = make_char4((signed char) lrintf(kh[0] * ik), (signed char) lrintf(kh[1] * ik),
                                                           (signed char) lrintf(kh[2] * ik), (signed char) lrintf(kh[3] * ik));
            *reinterpret_cast<char4*>(vc + o) = make_char4((signed char) lrintf(vh[0] * iv), (signed char) lrintf(vh[1] * iv),
                                                           (signed char) lrintf(vh[2] * iv), (signed char) lrintf(vh[3] * iv));
            if ((lane & 7) == 0) {
                reinterpret_cast<uint16_t*>(kc + q8n * kZhaDim)[row * 4 + lane / 8] = __half_as_ushort(__float2half(dk));
                reinterpret_cast<uint16_t*>(vc + q8n * kZhaDim)[row * 4 + lane / 8] = __half_as_ushort(__float2half(dv));
            }
        } else {
            store4h(khat + o, kh);
            store4h(vhat + o, vh);
        }
        if (lane == 0) bhat[((size_t) g * zq_cap(q8n, max_chunks) + c) * R + r] = m + logf(l) - dot;
    }
}

// A 32-value block of a row a warp's 8 lanes (4 values each): its largest magnitude / 7 the scale, codes
// round(x / scale) + 8 in [0, 15]; the block's 16 bytes go out in one store (the codes may be in RAM over PCIe).
__global__ void store_q4_kernel(const float* __restrict__ x, int8_t* __restrict__ codes, uint16_t* __restrict__ scales,
                                int n, long long pos, int rows, long long ring) {
    const size_t i = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) / 8, per = n / 32;
    const int sub = threadIdx.x & 7;   // the lane's 4 values of the block
    const bool live = i < (size_t) rows * per;
    const int r = live ? (int) (i / per) : 0, b = live ? (int) (i % per) : 0;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (live) v = *reinterpret_cast<const float4*>(x + (size_t) r * n + 32 * b + 4 * sub);
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o, 8));
    const float d = amax / 7.f, inv = d > 0.f ? 1.f / d : 0.f;
    const auto q = [&](float f) { return (uint32_t) min(15, max(0, (int) lrintf(f * inv) + 8)); };
    const uint32_t mine = q(v.x) | (q(v.y) << 4) | (q(v.z) << 8) | (q(v.w) << 12);   // 2 bytes
    // lanes sub 0..7 hold bytes 2 sub, 2 sub + 1: gather them in lane 0 of the 8 as 4 words
    uint32_t w0 = mine | (__shfl_down_sync(0xffffffffu, mine, 1, 8) << 16);
    uint32_t w1 = __shfl_down_sync(0xffffffffu, mine, 2, 8) | (__shfl_down_sync(0xffffffffu, mine, 3, 8) << 16);
    uint32_t w2 = __shfl_down_sync(0xffffffffu, mine, 4, 8) | (__shfl_down_sync(0xffffffffu, mine, 5, 8) << 16);
    uint32_t w3 = __shfl_down_sync(0xffffffffu, mine, 6, 8) | (__shfl_down_sync(0xffffffffu, mine, 7, 8) << 16);
    if (!live || sub != 0) return;
    long long row = pos + r;
    if (ring > 0) row %= ring;
    *reinterpret_cast<uint4*>(codes + (size_t) row * (n / 2) + 16 * b) = make_uint4(w0, w1, w2, w3);
    scales[(size_t) row * per + b] = __half_as_ushort(__float2half(d));
}

// A block of 32 values a thread: 4-bit codes back to values, then to Q8_0 (scale = largest magnitude / 127).
__global__ void q4_to_q8_kernel(const int8_t* __restrict__ sc, const uint16_t* __restrict__ ss, long long from,
                                int8_t* __restrict__ dc, uint16_t* __restrict__ ds, long long to, long long ring, int n,
                                int rows) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x, per = n / 32;
    if (i >= (size_t) rows * per) return;
    const int r = (int) (i / per), b = (int) (i % per);
    const uint8_t* c = reinterpret_cast<const uint8_t*>(sc) + (size_t) (from + r) * (n / 2) + 16 * b;
    const float d = __half2float(__ushort_as_half(ss[(size_t) (from + r) * per + b]));
    float v[32], amax = 0.f;
    for (int j = 0; j < 16; ++j) {
        v[2 * j] = (float) ((int) (c[j] & 15) - 8) * d;
        v[2 * j + 1] = (float) ((int) (c[j] >> 4) - 8) * d;
    }
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(v[j]));
    const float d8 = amax / 127.f, inv = d8 > 0.f ? 1.f / d8 : 0.f;
    const long long row = ring > 0 ? (to + r) % ring : to + r;
    int8_t* o = dc + (size_t) row * n + 32 * b;
    for (int j = 0; j < 32; ++j) o[j] = (int8_t) lrintf(v[j] * inv);
    ds[(size_t) row * per + b] = __half_as_ushort(__float2half(d8));
}

__device__ __forceinline__ int zha_far(long long own, int C, int recent, long long* fr) {
    *fr = max(1LL, own + 1 - recent) / C * C;   // the first recent key: a chunk's first
    return max(0, (int) (*fr / C) - 1);         // far chunks 1..nf
}

// The selection's scores of a KV group: for each far chunk, the best of the group's heads' scores less that head's
// best, so a chunk one head needs ranks as high as one another head needs.  One block a token.
__global__ void zha_pick_kernel(const float* __restrict__ score, int heads, size_t head_stride, int max_chunks,
                                long long pos, int C, int recent, float* __restrict__ pick) {
    __shared__ float best[8][4];
    const int t = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    score += (size_t) blockIdx.y * heads * head_stride;   // group blockIdx.y's heads, and its pick
    pick += (size_t) blockIdx.y * head_stride;
    long long fr;
    const int nf = zha_far(pos + t, C, recent, &fr);
    float top[8];
    for (int h = 0; h < heads; ++h) {
        float m = -INFINITY;
        for (int i = threadIdx.x; i < nf; i += blockDim.x) m = fmaxf(m, score[h * head_stride + (size_t) t * max_chunks + 1 + i]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (lane == 0) best[h][w] = m;
    }
    __syncthreads();
    for (int h = 0; h < heads; ++h) top[h] = fmaxf(fmaxf(best[h][0], best[h][1]), fmaxf(best[h][2], best[h][3]));
    for (int i = threadIdx.x; i < nf; i += blockDim.x) {
        float v = -INFINITY;
        for (int h = 0; h < heads; ++h) v = fmaxf(v, score[h * head_stride + (size_t) t * max_chunks + 1 + i] - top[h]);
        pick[(size_t) t * max_chunks + 1 + i] = v;
    }
}

// A token's selection: the B best of its far chunks by `score` (bisection on the threshold), a byte each into sel
// (chunk c at c; chunks past the far ones 0).  One block of 128 a token.
__global__ void zha_select_kernel(const float* __restrict__ score, int max_chunks, long long pos, int C, int recent,
                                  float budget, int min_chunks, int max_sel, uint8_t* __restrict__ sel, size_t stride) {
    __shared__ float part[4], part2[4], lo_s, hi_s;
    const int t = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    score += blockIdx.y * stride;   // selection blockIdx.y's scores and bytes
    sel += blockIdx.y * stride;
    long long fr;
    const int nf = zha_far(pos + t, C, recent, &fr);
    const float* s = score + (size_t) t * max_chunks + 1;
    uint8_t* out = sel + (size_t) t * max_chunks;
    float mn = INFINITY, mx = -INFINITY;
    for (int i = threadIdx.x; i < nf; i += blockDim.x) mn = fminf(mn, s[i]), mx = fmaxf(mx, s[i]);
    for (int o = 16; o > 0; o >>= 1) {
        mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    if (lane == 0) part[w] = mn, part2[w] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        lo_s = fminf(fminf(part[0], part[1]), fminf(part[2], part[3])) - 1.f;
        hi_s = fmaxf(fmaxf(part2[0], part2[1]), fmaxf(part2[2], part2[3]));
    }
    __syncthreads();
    const int B = nf ? min(nf, max(min_chunks, min(max_sel, (int) lrintf(budget * nf)))) : 0;
    for (int it = 0; it < 30 && nf; ++it) {
        const float mid = 0.5f * (lo_s + hi_s);
        int cnt = 0;
        for (int i = threadIdx.x; i < nf; i += blockDim.x) cnt += s[i] > mid;
        for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
        __syncthreads();
        if (lane == 0) part[w] = (float) cnt;
        __syncthreads();
        if (threadIdx.x == 0) {
            if (part[0] + part[1] + part[2] + part[3] >= B) lo_s = mid;
            else hi_s = mid;
        }
        __syncthreads();
    }
    for (int c = threadIdx.x; c < max_chunks; c += blockDim.x) out[c] = c >= 1 && c <= nf && s[c - 1] > lo_s;
}

// A tile of `tile` tokens chooses together: for each far chunk, the best over the tile's tokens of (its score less the
// token's best), the B best of those (bisection), each token reading the tile's chunks among its own far ones.
// One block of 128 a tile; selection blockIdx.y (scores and bytes `stride` apart).
__global__ void zha_select_tile_kernel(const float* __restrict__ score, int max_chunks, long long pos, int rows,
                                       int tile, int C, int recent, float budget, int min_chunks, int max_sel,
                                       uint8_t* __restrict__ sel, size_t stride, float* __restrict__ comb_buf) {
    __shared__ float part[4], part2[4], lo_s, hi_s, best[32];
    const int t0 = blockIdx.x * tile, t1 = min(rows, t0 + tile), lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    score += blockIdx.y * stride;
    sel += blockIdx.y * stride;
    float* comb = comb_buf + ((size_t) blockIdx.y * gridDim.x + blockIdx.x) * max_chunks;
    long long fr;
    const int nf0 = zha_far(pos + t0, C, recent, &fr), nf1 = zha_far(pos + t1 - 1, C, recent, &fr);
    for (int t = t0; t < t1; ++t) {   // each token's best far chunk
        const int nf = zha_far(pos + t, C, recent, &fr);
        float m = -INFINITY;
        for (int i = threadIdx.x; i < nf; i += blockDim.x) m = fmaxf(m, score[(size_t) t * max_chunks + 1 + i]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (lane == 0) part[w] = m;
        __syncthreads();
        if (threadIdx.x == 0) best[t - t0] = fmaxf(fmaxf(part[0], part[1]), fmaxf(part[2], part[3]));
        __syncthreads();
    }
    float mn = INFINITY, mx = -INFINITY;
    for (int i = threadIdx.x; i < nf1; i += blockDim.x) {
        float v = -INFINITY;
        for (int t = t0; t < t1; ++t) {
            const int nf = t == t0 ? nf0 : zha_far(pos + t, C, recent, &fr);
            if (i < nf) v = fmaxf(v, score[(size_t) t * max_chunks + 1 + i] - best[t - t0]);
        }
        comb[i] = v;
        mn = fminf(mn, v), mx = fmaxf(mx, v);
    }
    for (int o = 16; o > 0; o >>= 1) {
        mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    if (lane == 0) part[w] = mn, part2[w] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        lo_s = fminf(fminf(part[0], part[1]), fminf(part[2], part[3])) - 1.f;
        hi_s = fmaxf(fmaxf(part2[0], part2[1]), fmaxf(part2[2], part2[3]));
    }
    __syncthreads();
    const int B = nf1 ? min(nf1, max(min_chunks, min(max_sel, (int) lrintf(budget * nf1)))) : 0;
    for (int it = 0; it < 30 && nf1; ++it) {
        const float mid = 0.5f * (lo_s + hi_s);
        int cnt = 0;
        for (int i = threadIdx.x; i < nf1; i += blockDim.x) cnt += comb[i] > mid;
        for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
        __syncthreads();
        if (lane == 0) part[w] = (float) cnt;
        __syncthreads();
        if (threadIdx.x == 0) {
            if (part[0] + part[1] + part[2] + part[3] >= B) lo_s = mid;
            else hi_s = mid;
        }
        __syncthreads();
    }
    for (int t = t0; t < t1; ++t) {
        const int nf = zha_far(pos + t, C, recent, &fr);
        uint8_t* out = sel + (size_t) t * max_chunks;
        for (int c = threadIdx.x; c < max_chunks; c += blockDim.x) out[c] = c >= 1 && c <= nf && comb[c - 1] > lo_s;
    }
}

// The chunks any token of the call (any of `sels` selections) reads, and chunk 0: each not staged yet (cmap < 0)
// gets the next place (base + count) and goes on glist to be copied.  A thread a word of 32 chunks: 32 bytes of each
// token's selection.
__global__ void zha_union_kernel(const uint8_t* __restrict__ sel, int sels, size_t sel_stride, int rows, int max_chunks,
                                 int nch, int* __restrict__ cmap, int* __restrict__ count, int base,
                                 int* __restrict__ glist) {
    const int wi = blockIdx.x * blockDim.x + threadIdx.x;
    if (32 * wi >= max_chunks) return;
    if (gridDim.y > 1) {   // every group at once: group blockIdx.y's selections, map, count and list
        const int g = blockIdx.y;
        sel += (size_t) g * sels * sel_stride;
        cmap += (size_t) g * max_chunks;
        count += g;
        base = g * max_chunks;
        glist += (size_t) g * (max_chunks + 1);
    }
    uint4 acc0 = make_uint4(0, 0, 0, 0), acc1 = acc0;
    if (32 * wi < nch)
        for (int h = 0; h < sels; ++h)
            for (int t = 0; t < rows; ++t) {
                const uint4* r = reinterpret_cast<const uint4*>(sel + h * sel_stride + (size_t) t * max_chunks + 32 * (size_t) wi);
                const uint4 a = r[0], b = r[1];
                acc0.x |= a.x, acc0.y |= a.y, acc0.z |= a.z, acc0.w |= a.w;
                acc1.x |= b.x, acc1.y |= b.y, acc1.z |= b.z, acc1.w |= b.w;
            }
    const uint32_t v[8] = {acc0.x, acc0.y, acc0.z, acc0.w, acc1.x, acc1.y, acc1.z, acc1.w};
    for (int k = 0; k < 32; ++k) {
        const int c = 32 * wi + k;
        const bool any = c < nch && (c == 0 || ((v[k >> 2] >> (8 * (k & 3))) & 0xffu));
        if (!any || cmap[c] >= 0) continue;
        cmap[c] = base + atomicAdd(count, 1);
        glist[1 + atomicAdd(&glist[0], 1)] = c;
    }
}

// The staged chunks' keys and values of KV head g, from the leaves into the staging buffer (128 values a key).
// A block a chunk, a thread 8 bytes of codes or 4 fp16 values.
__global__ void zha_gather_kernel(ZhaKv leaves, int g, int C, const int* __restrict__ cmap,
                                  const int* __restrict__ glist, ZhaKv stage) {
    if ((int) blockIdx.x >= glist[0]) return;
    const int c = glist[1 + blockIdx.x];
    const int slot = cmap[c];
    for (int i = threadIdx.x; i < C * (kZhaDim / 4); i += blockDim.x) {
        const int j = i / (kZhaDim / 4), d = 4 * (i % (kZhaDim / 4));
        const size_t from = (size_t) ((long long) c * C + j) * leaves.row + (size_t) g * kZhaDim + d;
        const size_t to = ((size_t) slot * C + j) * kZhaDim + d;
        if (leaves.q4) {   // 4 codes in 2 bytes
            *reinterpret_cast<uint16_t*>(const_cast<int8_t*>(stage.kc) + to / 2) = *reinterpret_cast<const uint16_t*>(leaves.kc + from / 2);
            *reinterpret_cast<uint16_t*>(const_cast<int8_t*>(stage.vc) + to / 2) = *reinterpret_cast<const uint16_t*>(leaves.vc + from / 2);
        } else if (leaves.kc) {
            *reinterpret_cast<char4*>(const_cast<int8_t*>(stage.kc) + to) = *reinterpret_cast<const char4*>(leaves.kc + from);
            *reinterpret_cast<char4*>(const_cast<int8_t*>(stage.vc) + to) = *reinterpret_cast<const char4*>(leaves.vc + from);
        } else {
            *reinterpret_cast<uint2*>(const_cast<uint16_t*>(stage.k16) + to) = *reinterpret_cast<const uint2*>(leaves.k16 + from);
            *reinterpret_cast<uint2*>(const_cast<uint16_t*>(stage.v16) + to) = *reinterpret_cast<const uint2*>(leaves.v16 + from);
        }
        if (leaves.kc && d % 32 == 0) {
            const_cast<uint16_t*>(stage.ks)[to / 32] = leaves.ks[from / 32];
            const_cast<uint16_t*>(stage.vs)[to / 32] = leaves.vs[from / 32];
        }
    }
}

// ---- an answer's token through the index (the cache's step, rows 1): scores, selection, exact part, tail ------

// Chunk scores from int8 summaries, R 4: a thread a (chunk, reference) row of group blockIdx.y, 16-byte reads, the
// group's heads' dots; the best of a chunk's 4 references (4 lanes) and which one into head * hs + c.
__global__ void __launch_bounds__(128) zha_score_dec_kernel(const float* __restrict__ q, const int8_t* __restrict__ codes,
                                                            size_t q8n, const float* __restrict__ bhat, int nch,
                                                            int max_chunks, int n_head, int per,
                                                            float* __restrict__ score, uint8_t* __restrict__ rstar,
                                                            size_t hs) {
    __shared__ float qs[4][kZhaDim];
    const int g = blockIdx.y, lane = threadIdx.x & 31;
    const float sc = rsqrtf((float) kZhaDim);
    for (int i = threadIdx.x; i < per * kZhaDim; i += blockDim.x)
        qs[i / kZhaDim][i % kZhaDim] = q[((size_t) g * per + i / kZhaDim) * kZhaDim + i % kZhaDim] * sc;
    __syncthreads();
    const int c = blockIdx.x * 32 + (threadIdx.x >> 2), r = threadIdx.x & 3;
    const bool live = c < nch;
    const size_t row = ((size_t) g * zq_cap(q8n, max_chunks) + (live ? c : 0)) * 4 + r;
    const uint16_t* scales = zq_scales(codes, q8n);
    const uint2 su = *reinterpret_cast<const uint2*>(scales + row * 4);
    const uint16_t sv[4] = {(uint16_t) (su.x & 0xffff), (uint16_t) (su.x >> 16), (uint16_t) (su.y & 0xffff),
                            (uint16_t) (su.y >> 16)};
    float acc[3] = {0.f, 0.f, 0.f};
    if (zq_4(q8n)) {   // 4-bit codes: 64 bytes a row, a uint4 a 32-value block
        const uint4* cr = reinterpret_cast<const uint4*>(codes + row * (kZhaDim / 2));
        uint4 x[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = cr[i];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t w4[4] = {x[b].x, x[b].y, x[b].z, x[b].w};
            float part[3] = {0.f, 0.f, 0.f};
#pragma unroll
            for (int k = 0; k < 4; ++k)
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    const int d0 = 32 * b + 8 * k + n;
                    const float a0 = (float) ((int) ((w4[k] >> (4 * n)) & 15) - 8);
#pragma unroll
                    for (int p = 0; p < 3; ++p)
                        if (p < per) part[p] += a0 * qs[p][d0];
                }
            const float d = __half2float(__ushort_as_half(sv[b]));
#pragma unroll
            for (int p = 0; p < 3; ++p) acc[p] += part[p] * d;
        }
    } else {
    const uint4* cr = reinterpret_cast<const uint4*>(codes + row * kZhaDim);
    uint4 x[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = cr[i];
#pragma unroll
    for (int b = 0; b < 4; ++b) {   // a 32-value block: 2 uint4s
        float part[3] = {0.f, 0.f, 0.f};
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            const uint4 u = x[2 * b + h2];
            const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int d0 = 32 * b + 16 * h2 + 4 * k;
                const float a0 = (float) (int8_t) (w4[k] & 0xff), a1 = (float) (int8_t) ((w4[k] >> 8) & 0xff),
                            a2 = (float) (int8_t) ((w4[k] >> 16) & 0xff), a3 = (float) (int8_t) (w4[k] >> 24);
#pragma unroll
                for (int p = 0; p < 3; ++p)
                    if (p < per)
                        part[p] += a0 * qs[p][d0] + a1 * qs[p][d0 + 1] + a2 * qs[p][d0 + 2] + a3 * qs[p][d0 + 3];
            }
        }
        const float d = __half2float(__ushort_as_half(sv[b]));
#pragma unroll
        for (int p = 0; p < 3; ++p) acc[p] += part[p] * d;
    }
    }
    const float bb = bhat[row];
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        if (p >= per) break;
        float v = acc[p] + bb;
        int arg = r;
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oa = __shfl_xor_sync(0xffffffffu, arg, o);
            if (ov > v || (ov == v && oa < arg)) v = ov, arg = oa;
        }
        if (live && r == 0) {
            const size_t at = (size_t) (g * per + p) * hs + c;
            score[at] = v;
            rstar[at] = (uint8_t) arg;
        }
    }
    (void) lane;
}

// The block's max (every thread gets it); red: 32 floats of shared memory.
__device__ __forceinline__ float zha_block_max(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    v = -INFINITY;
    for (int i = 0; i < warps; ++i) v = fmaxf(v, red[i]);
    return v;
}

__device__ __forceinline__ float zha_block_min(float v, float* red) {
    return -zha_block_max(-v, red);
}

// The threshold that keeps the B largest of v[0..n) (shared memory; [lo, hi] their range): two rounds of a 1024-bin
// histogram, the first round's bin that crosses B split again.  A block of 1024 threads; every thread gets it.
__device__ float zha_hist_threshold(const float* v, int n, int B, float lo, float hi, int* hist, int* sums,
                                    int* found, int* above) {
    if (B >= n || !(hi > lo)) return -INFINITY;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, nt = blockDim.x;
    float a = lo, width = (hi - lo) / 1024.f, b_end = INFINITY;
    bool top_bin = true;
    int need = B;
    for (int level = 0; level < 2; ++level) {
        for (int k = tid; k < 1024; k += nt) hist[k] = 0;
        if (tid == 0) *found = 0, *above = 0;
        __syncthreads();
        for (int i = tid; i < n; i += nt) {
            const float x = v[i];
            if (x < a || x >= b_end) continue;
            atomicAdd(&hist[min(1023, max(0, (int) ((x - a) / width)))], 1);
        }
        __syncthreads();
        int c = hist[1023 - tid];   // a suffix scan: thread k' holds bins >= 1023 - k'
        for (int o = 1; o < 32; o <<= 1) {
            const int t2 = __shfl_up_sync(0xffffffffu, c, o);
            if (lane >= o) c += t2;
        }
        if (lane == 31) sums[w] = c;
        __syncthreads();
        if (w == 0) {
            int s2 = lane < (nt >> 5) ? sums[lane] : 0;
            for (int o = 1; o < 32; o <<= 1) {
                const int t2 = __shfl_up_sync(0xffffffffu, s2, o);
                if (lane >= o) s2 += t2;
            }
            sums[lane] = s2;
        }
        __syncthreads();
        const int S = c + (w ? sums[w - 1] : 0), Sabove = S - hist[1023 - tid];
        if (S >= need && Sabove < need) *found = 1023 - tid, *above = Sabove;
        __syncthreads();
        const int k = *found;
        need -= *above;
        a += k * width;
        if (!(top_bin && k == 1023)) b_end = a + width;
        top_bin = top_bin && k == 1023;
        width /= 1024.f;
        __syncthreads();
    }
    return a;
}

// zha_select_tile_kernel with a block of 1024 and the tile's picks in shared memory, the threshold by histograms:
// for each far chunk the best over the tile's tokens of (its score less the token's best; with zero_best the scores
// are a group's picks, whose best is 0), the B best of those, each token reading the tile's chunks among its own.
__global__ void __launch_bounds__(1024) zha_select_tile2_kernel(const float* __restrict__ score, int max_chunks,
                                                                long long pos, int rows, int tile, int C, int recent,
                                                                float budget, int min_chunks, int max_sel,
                                                                uint8_t* __restrict__ sel, size_t stride,
                                                                bool zero_best) {
    extern __shared__ float comb[];
    __shared__ float red[32], best[64];
    __shared__ int nfs[64], hist[1024], sums[32], found, above;
    const int t0 = blockIdx.x * tile, t1 = min(rows, t0 + tile), tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    score += blockIdx.y * stride;
    sel += blockIdx.y * stride;
    long long fr;
    for (int t = t0 + tid; t < t1; t += blockDim.x) nfs[t - t0] = zha_far(pos + t, C, recent, &fr);
    __syncthreads();
    const int nf1 = nfs[t1 - 1 - t0];
    for (int t = t0 + w; t < t1; t += 32) {   // a warp a token: its best far chunk
        float m = 0.f;
        if (!zero_best) {
            m = -INFINITY;
            for (int i = lane; i < nfs[t - t0]; i += 32) m = fmaxf(m, score[(size_t) t * max_chunks + 1 + i]);
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        }
        if (lane == 0) best[t - t0] = m;
    }
    __syncthreads();
    float mn = INFINITY, mx = -INFINITY;
    for (int i = tid; i < nf1; i += blockDim.x) {
        float v = -INFINITY;
        for (int t = t0; t < t1; ++t)
            if (i < nfs[t - t0]) v = fmaxf(v, score[(size_t) t * max_chunks + 1 + i] - best[t - t0]);
        comb[i] = v;
        mn = fminf(mn, v), mx = fmaxf(mx, v);
    }
    const float lo = zha_block_min(mn, red), hi = zha_block_max(mx, red);
    const int B = nf1 ? min(nf1, max(min_chunks, min(max_sel, (int) lrintf(budget * nf1)))) : 0;
    const float thr = zha_hist_threshold(comb, nf1, B, lo, hi, hist, sums, &found, &above);
    for (int t = t0; t < t1; ++t) {
        const int nf = nfs[t - t0];
        uint8_t* out = sel + (size_t) t * max_chunks;
        for (int c = tid; c < max_chunks; c += blockDim.x) out[c] = c >= 1 && c <= nf && comb[c - 1] >= thr;
    }
}


// a float's atomic max (the word starts all ones, a NaN, which every value replaces)
__device__ __forceinline__ void zha_atomic_max(float* at, float v) {
    if (__float_as_int(v) >= 0) atomicMax(reinterpret_cast<int*>(at), __float_as_int(v));
    else atomicMin(reinterpret_cast<unsigned int*>(at), __float_as_uint(v));
}

// zha_select_tile2_kernel on a group's picks, the picks made here (zha_pick_kernel no more): for each far chunk the best
// over the tile's tokens and the group's heads of (the head's score less the head's best, top: zha_score_kernel's), the
// B best of those, each token reading the tile's chunks among its own.  The same bytes as pick + select_tile2.
__global__ void __launch_bounds__(1024) zha_select_tile3_kernel(const float* __restrict__ score, int per, size_t hs,
                                                                const float* __restrict__ top, int max_chunks,
                                                                long long pos, int rows, int tile, int C, int recent,
                                                                float budget, int min_chunks, int max_sel,
                                                                uint8_t* __restrict__ sel) {
    extern __shared__ float comb[];
    __shared__ float red[32], tp[4][64];
    __shared__ int nfs[64], hist[1024], sums[32], found, above;
    const int t0 = blockIdx.x * tile, t1 = min(rows, t0 + tile), tid = threadIdx.x;
    long long fr;
    for (int t = t0 + tid; t < t1; t += blockDim.x) {
        nfs[t - t0] = zha_far(pos + t, C, recent, &fr);
        for (int h = 0; h < per; ++h) tp[h][t - t0] = top[(size_t) h * rows + t];
    }
    __syncthreads();
    const int nf1 = nfs[t1 - 1 - t0];
    float mn = INFINITY, mx = -INFINITY;
    for (int i = tid; i < nf1; i += blockDim.x) {
        float v = -INFINITY;
        for (int t = t0; t < t1; ++t)
            if (i < nfs[t - t0]) {
                float pk = -INFINITY;   // zha_pick_kernel's value, then less 0 as select_tile2's zero_best
                for (int h = 0; h < per; ++h) pk = fmaxf(pk, score[h * hs + (size_t) t * max_chunks + 1 + i] - tp[h][t - t0]);
                v = fmaxf(v, pk - 0.f);
            }
        comb[i] = v;
        mn = fminf(mn, v), mx = fmaxf(mx, v);
    }
    const float lo = zha_block_min(mn, red), hi = zha_block_max(mx, red);
    const int B = nf1 ? min(nf1, max(min_chunks, min(max_sel, (int) lrintf(budget * nf1)))) : 0;
    const float thr = zha_hist_threshold(comb, nf1, B, lo, hi, hist, sums, &found, &above);
    for (int t = t0; t < t1; ++t) {
        const int nf = nfs[t - t0];
        uint8_t* out = sel + (size_t) t * max_chunks;
        for (int c = tid; c < max_chunks; c += blockDim.x) out[c] = c >= 1 && c <= nf && comb[c - 1] >= thr;
    }
}

// A group's selection of an answer's token (blockIdx.x the group, 1024 threads, the far chunks' pick values in
// dynamic shared memory): each head's best, the pick (best of score less the head's best), then the B best by two
// rounds of a 1024-bin histogram; the bytes into sel + g * hs.
__global__ void __launch_bounds__(1024) zha_pick_select_dec_kernel(const float* __restrict__ score, int per, size_t hs,
                                                                   int max_chunks, long long pos, int C, int recent,
                                                                   float budget, int min_chunks, int max_sel,
                                                                   uint8_t* __restrict__ sel,
                                                                   const int* __restrict__ pmap, float stick) {
    extern __shared__ float pk[];
    __shared__ float red[32];
    __shared__ int hist[1024], sums[32];
    __shared__ int found, above_s;
    const int g = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, nt = blockDim.x;
    score += (size_t) g * per * hs;
    sel += (size_t) g * hs;
    long long fr;
    const int nf = zha_far(pos, C, recent, &fr);
    float top[4];
    for (int h = 0; h < per; ++h) {
        float m = -INFINITY;
        for (int i = tid; i < nf; i += nt) m = fmaxf(m, score[h * hs + 1 + i]);
        top[h] = zha_block_max(m, red);
    }
    float mn = INFINITY, mx = -INFINITY;
    if (pmap) pmap += (size_t) g * max_chunks;
    for (int i = tid; i < nf; i += nt) {
        float v = -INFINITY;
        for (int h = 0; h < per; ++h) v = fmaxf(v, score[h * hs + 1 + i] - top[h]);
        if (pmap && pmap[1 + i] >= 0) v += stick;   // a chunk the cache holds: kept over a close one off the card
        pk[i] = v;
        mn = fminf(mn, v), mx = fmaxf(mx, v);
    }
    const float lo = zha_block_min(mn, red), hi = zha_block_max(mx, red);
    const int B = nf ? min(nf, max(min_chunks, min(max_sel, (int) lrintf(budget * nf)))) : 0;
    float thr = -INFINITY;
    if (B < nf && hi > lo) {
        float a = lo, width = (hi - lo) / 1024.f;
        bool top_bin = true;   // the range's upper end open (the largest values clamp into the last bin)
        float b_end = INFINITY;
        int need = B;
        for (int level = 0; level < 2; ++level) {
            for (int k = tid; k < 1024; k += nt) hist[k] = 0;
            if (tid == 0) found = 0, above_s = 0;
            __syncthreads();
            for (int i = tid; i < nf; i += nt) {
                const float v = pk[i];
                if (v < a || v >= b_end) continue;
                const int bin = min(1023, max(0, (int) ((v - a) / width)));
                atomicAdd(&hist[bin], 1);
            }
            __syncthreads();
            // suffix sums: thread k' = 1023 - k scans hist[k] (an inclusive prefix over k')
            int v = tid < 1024 ? hist[1023 - tid] : 0;
            for (int o = 1; o < 32; o <<= 1) {
                const int t2 = __shfl_up_sync(0xffffffffu, v, o);
                if (lane >= o) v += t2;
            }
            if (lane == 31) sums[w] = v;
            __syncthreads();
            if (w == 0) {
                int s = lane < (nt >> 5) ? sums[lane] : 0;
                for (int o = 1; o < 32; o <<= 1) {
                    const int t2 = __shfl_up_sync(0xffffffffu, s, o);
                    if (lane >= o) s += t2;
                }
                sums[lane] = s;
            }
            __syncthreads();
            const int S = v + (w ? sums[w - 1] : 0);   // the count of bins >= 1023 - tid
            const int Sabove = S - (tid < 1024 ? hist[1023 - tid] : 0);
            if (tid < 1024 && S >= need && Sabove < need) found = 1023 - tid, above_s = Sabove;
            __syncthreads();
            const int k = found;
            need -= above_s;
            a = a + k * width;
            b_end = (top_bin && k == 1023) ? b_end : a + width;
            top_bin = top_bin && k == 1023;
            width /= 1024.f;
            __syncthreads();
        }
        thr = a;
    }
    for (int c = tid; c < max_chunks; c += nt) sel[c] = c >= 1 && c <= nf && pk[c - 1] >= thr;
}

// The exact part of an answer's token for head blockIdx.x: chunk 0 and the selected far chunks (the group's list,
// from the cache's pool) and the recent ones (from the ring), split over gridDim.y segments; a warp a chunk, a lane a
// key for the scores (64 keys: 2 a lane), a lane 4 dims for the values.  Parts (top, sum, values) at seg.
__global__ void __launch_bounds__(128) zha_exact_dec_kernel(const float* __restrict__ q, ZhaKv pool,
                                                            const int* __restrict__ cmap, const int* __restrict__ glist,
                                                            ZhaKv ring, long long n_ring, int max_chunks, long long pos,
                                                            int n_head, int kv, int C, int recent,
                                                            const float* __restrict__ rr, const float* __restrict__ table,
                                                            int extent, int nrel, float* __restrict__ part, int seg_total,
                                                            int nl_fix) {
    __shared__ float qs[kZhaDim], rv[32], wm[4], wl[4], wo[4][kZhaDim];
    const int head = blockIdx.x, seg = blockIdx.y, segs = gridDim.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int g = head / (n_head / kv);
    const float sc = rsqrtf((float) kZhaDim);
    for (int i = threadIdx.x; i < kZhaDim; i += blockDim.x) qs[i] = q[(size_t) head * kZhaDim + i] * sc;
    if ((int) threadIdx.x < nrel) rv[threadIdx.x] = rr[(size_t) head * nrel + threadIdx.x];
    __syncthreads();
    const long long own = pos;
    long long fr;
    zha_far(own, C, recent, &fr);
    const int* gl = glist + (size_t) g * (max_chunks + 1);
    const int nl = nl_fix >= 0 ? nl_fix : gl[0];   // nl_fix: every key in the ring, the list not read
    const int rc0 = (int) (fr / C), last = (int) (own / C), nr = last - rc0 + 1;
    const int items = nl + nr;
    const int per_seg = (items + segs - 1) / segs, i0 = seg * per_seg, i1 = min(items, i0 + per_seg);
    cmap += (size_t) g * max_chunks;
    float m = -INFINITY, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int it = i0 + w; it < i1; it += 4) {
        int c;
        bool from_ring;
        if (it < nl) {
            if (nl_fix >= 0) continue;
            c = gl[1 + it];
            if ((long long) c * C >= fr) continue;   // a recent chunk: the ring's
            from_ring = false;
        } else {
            c = rc0 + (it - nl);
            from_ring = true;
        }
        const long long j0 = (long long) c * C, j1 = min(j0 + C, own + 1);
        const int slot = from_ring ? 0 : cmap[c] & ~kZhaMiss;
        // the scores: lane keys j0 + lane and j0 + 32 + lane
        float s2[2];
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            const long long j = j0 + lane + 32 * h2;
            float s = -INFINITY;
            if (j < j1) {
                float dot = 0.f;
                if (from_ring) {
                    const size_t at = (size_t) (j % n_ring) * ring.row + (size_t) g * kZhaDim;
                    const uint4* kr = reinterpret_cast<const uint4*>(ring.kc + at);
                    const uint2 su = *reinterpret_cast<const uint2*>(ring.ks + at / 32);
                    const uint16_t sv[4] = {(uint16_t) (su.x & 0xffff), (uint16_t) (su.x >> 16),
                                            (uint16_t) (su.y & 0xffff), (uint16_t) (su.y >> 16)};
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        float pt = 0.f;
#pragma unroll
                        for (int h3 = 0; h3 < 2; ++h3) {
                            const uint4 u = kr[2 * b + h3];
                            const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                const int d0 = 32 * b + 16 * h3 + 4 * k;
                                pt += (float) (int8_t) (w4[k] & 0xff) * qs[d0] +
                                      (float) (int8_t) ((w4[k] >> 8) & 0xff) * qs[d0 + 1] +
                                      (float) (int8_t) ((w4[k] >> 16) & 0xff) * qs[d0 + 2] +
                                      (float) (int8_t) (w4[k] >> 24) * qs[d0 + 3];
                            }
                        }
                        dot += pt * __half2float(__ushort_as_half(sv[b]));
                    }
                } else {
                    const size_t kk = (size_t) slot * C + (size_t) (j - j0);   // the pool's key, 128 values
                    const uint4* kr = reinterpret_cast<const uint4*>(pool.kc + kk * (kZhaDim / 2));
                    const uint2 su = *reinterpret_cast<const uint2*>(pool.ks + kk * 4);
                    const uint16_t sv[4] = {(uint16_t) (su.x & 0xffff), (uint16_t) (su.x >> 16),
                                            (uint16_t) (su.y & 0xffff), (uint16_t) (su.y >> 16)};
#pragma unroll
                    for (int b = 0; b < 4; ++b) {   // 32 values: 16 bytes
                        const uint4 u = kr[b];
                        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
                        float pt = 0.f;
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
#pragma unroll
                            for (int n = 0; n < 8; ++n) {
                                const int d = 32 * b + 8 * k + n;
                                pt += (float) ((int) ((w4[k] >> (4 * n)) & 15) - 8) * qs[d];
                            }
                        }
                        dot += pt * __half2float(__ushort_as_half(sv[b]));
                    }
                }
                s = dot;
                const long long dd = own - j;
                if (dd < extent) {
                    const float* tt = table + (size_t) dd * nrel;
                    float bb = 0.f;
                    for (int k2 = 0; k2 < nrel; ++k2) bb += rv[k2] * tt[k2];
                    s += bb;
                }
            }
            s2[h2] = s;
        }
        float cm = fmaxf(s2[0], s2[1]);
        for (int o = 16; o > 0; o >>= 1) cm = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, o));
        const float p0 = s2[0] == -INFINITY ? 0.f : expf(s2[0] - cm), p1 = s2[1] == -INFINITY ? 0.f : expf(s2[1] - cm);
        const float cz = warp_sum(p0 + p1);
        // the values: a lane dims 4 lane .. 4 lane + 3, the chunk's keys in turn
        float o4[4] = {0.f, 0.f, 0.f, 0.f};
        const int nk = (int) (j1 - j0);
#pragma unroll 8
        for (int k = 0; k < nk; ++k) {
            const float p = __shfl_sync(0xffffffffu, k < 32 ? p0 : p1, k & 31);
            float v[4];
            if (from_ring) {
                const size_t at = (size_t) ((j0 + k) % n_ring) * ring.row + (size_t) g * kZhaDim + 4 * lane;
                const char4 b4 = *reinterpret_cast<const char4*>(ring.vc + at);
                const float d = __half2float(__ushort_as_half(ring.vs[at / 32]));
                v[0] = b4.x * d, v[1] = b4.y * d, v[2] = b4.z * d, v[3] = b4.w * d;
            } else {
                const size_t at = ((size_t) slot * C + k) * kZhaDim + 4 * lane;
                const uint16_t b2 = *reinterpret_cast<const uint16_t*>(pool.vc + at / 2);
                const float d = __half2float(__ushort_as_half(pool.vs[at / 32]));
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] = (float) ((int) ((b2 >> (4 * i)) & 15) - 8) * d;
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) o4[i] += p * v[i];
        }
        // into the warp's running sums
        const float nm = fmaxf(m, cm), f = m == -INFINITY ? 0.f : expf(m - nm), e = expf(cm - nm);
        l = l * f + cz * e;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = acc[i] * f + o4[i] * e;
        m = nm;
    }
    if (lane == 0) wm[w] = m, wl[w] = l;
#pragma unroll
    for (int i = 0; i < 4; ++i) wo[w][4 * lane + i] = acc[i];
    __syncthreads();
    if (w != 0) return;
    float M = -INFINITY;
    for (int k = 0; k < 4; ++k) M = fmaxf(M, wm[k]);
    float Z = 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < 4; ++k) {
        if (wm[k] == -INFINITY) continue;
        const float f = expf(wm[k] - M);
        Z += wl[k] * f;
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] += wo[k][4 * lane + i] * f;
    }
    float* pp = part + ((size_t) head * seg_total + seg) * (kZhaDim + 2);
#pragma unroll
    for (int i = 0; i < 4; ++i) pp[4 * lane + i] = o[i];
    if (lane == 0) pp[kZhaDim] = M, pp[kZhaDim + 1] = Z;
}

// The same for a KV group's 3 heads at once (blockIdx.x the group): a key's and a value's dequantized once for them.
__device__ __forceinline__ void zha_key_ring(const ZhaKv& ring, size_t at, const float (&qs)[3][kZhaDim], float (&dot)[3]) {
    const uint4* kr = reinterpret_cast<const uint4*>(ring.kc + at);
    const uint2 su = *reinterpret_cast<const uint2*>(ring.ks + at / 32);
    const uint16_t sv[4] = {(uint16_t) (su.x & 0xffff), (uint16_t) (su.x >> 16), (uint16_t) (su.y & 0xffff),
                            (uint16_t) (su.y >> 16)};
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        float pt[3] = {0.f, 0.f, 0.f};
#pragma unroll
        for (int h3 = 0; h3 < 2; ++h3) {
            const uint4 u = kr[2 * b + h3];
            const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const int d = 32 * b + 16 * h3 + 4 * k + n;
                    const float x = (float) (int8_t) ((w4[k] >> (8 * n)) & 0xff);
#pragma unroll
                    for (int p = 0; p < 3; ++p) pt[p] += x * qs[p][d];
                }
            }
        }
        const float sc = __half2float(__ushort_as_half(sv[b]));
#pragma unroll
        for (int p = 0; p < 3; ++p) dot[p] += pt[p] * sc;
    }
}

__device__ __forceinline__ void zha_key_pool(const ZhaKv& pool, size_t kk, const float (&qs)[3][kZhaDim], float (&dot)[3]) {
    const uint4* kr = reinterpret_cast<const uint4*>(pool.kc + kk * (kZhaDim / 2));
    const uint2 su = *reinterpret_cast<const uint2*>(pool.ks + kk * 4);
    const uint16_t sv[4] = {(uint16_t) (su.x & 0xffff), (uint16_t) (su.x >> 16), (uint16_t) (su.y & 0xffff),
                            (uint16_t) (su.y >> 16)};
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const uint4 u = kr[b];
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
        float pt[3] = {0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int d = 32 * b + 8 * k + n;
                const float x = (float) ((int) ((w4[k] >> (4 * n)) & 15) - 8);
#pragma unroll
                for (int p = 0; p < 3; ++p) pt[p] += x * qs[p][d];
            }
        }
        const float sc = __half2float(__ushort_as_half(sv[b]));
#pragma unroll
        for (int p = 0; p < 3; ++p) dot[p] += pt[p] * sc;
    }
}

// zha_exact_dec_kernel for a KV group's 3 heads at once (blockIdx.x the group, blockIdx.y the segment): the selected
// chunks from the cache's pool, chunk 0 and the recent keys from the ring, each key read once for the three.
__global__ void __launch_bounds__(128) zha_exact_dec_group_kernel(const float* __restrict__ q, ZhaKv pool,
                                                                  const int* __restrict__ cmap,
                                                                  const int* __restrict__ glist, ZhaKv ring,
                                                                  long long n_ring, int max_chunks, long long pos,
                                                                  int n_head, int C, int recent,
                                                                  const float* __restrict__ rr,
                                                                  const float* __restrict__ table, int extent, int nrel,
                                                                  float* __restrict__ part, int seg_total, int nl_fix) {
    constexpr int PER = 3, KPS = 4;   // KPS: keys a step of the values (8 lanes a key, 16 values a lane)
    __shared__ float qs[PER][kZhaDim], rv[PER][16], wm[4][PER], wl[4][PER], wo[4][PER][kZhaDim];
    const int g = blockIdx.x, seg = blockIdx.y, segs = gridDim.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const float sc = rsqrtf((float) kZhaDim);
    for (int i = threadIdx.x; i < PER * kZhaDim; i += blockDim.x)
        qs[i / kZhaDim][i % kZhaDim] = q[((size_t) g * PER + i / kZhaDim) * kZhaDim + i % kZhaDim] * sc;
    for (int i = threadIdx.x; i < PER * nrel; i += blockDim.x)
        rv[i / nrel][i % nrel] = rr[((size_t) g * PER + i / nrel) * nrel + i % nrel];
    __syncthreads();
    const long long own = pos;
    long long fr;
    zha_far(own, C, recent, &fr);
    const int* gl = glist + (size_t) g * (max_chunks + 1);
    const int nl = nl_fix >= 0 ? nl_fix : gl[0];   // nl_fix: every key in the ring, the list not read
    const int rc0 = (int) (fr / C), last = (int) (own / C), nr = last - rc0 + 1;
    const int items = nl + nr;
    const int per_seg = (items + segs - 1) / segs, i0 = seg * per_seg, i1 = min(items, i0 + per_seg);
    cmap += (size_t) g * max_chunks;
    constexpr int LPK = 32 / KPS, DPL = kZhaDim / LPK;   // lanes a key, values a lane
    const int sub = lane / LPK, dg = lane % LPK;          // the values' step: key sub, values [DPL dg, + DPL)
    float m[PER], l[PER], acc16[PER][DPL];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        m[p] = -INFINITY, l[p] = 0.f;
#pragma unroll
        for (int i = 0; i < DPL; ++i) acc16[p][i] = 0.f;
    }
    for (int it = i0 + w; it < i1; it += 4) {
        int c;
        bool from_ring;
        if (it < nl) {
            if (nl_fix >= 0) continue;
            c = gl[1 + it];
            if ((long long) c * C >= fr) continue;
            from_ring = false;
        } else {
            c = rc0 + (it - nl);
            from_ring = true;
        }
        const long long j0 = (long long) c * C, j1 = min(j0 + C, own + 1);
        const int slot = from_ring ? 0 : cmap[c] & ~kZhaMiss;
        float s[2][PER];
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            const long long j = j0 + lane + 32 * h2;
            float dot[PER] = {0.f, 0.f, 0.f};
            if (j < j1) {
                if (from_ring) zha_key_ring(ring, (size_t) (j % n_ring) * ring.row + (size_t) g * kZhaDim, qs, dot);
                else zha_key_pool(pool, (size_t) slot * C + (size_t) (j - j0), qs, dot);
                const long long dd = own - j;
                if (dd < extent) {
                    const float* tt = table + (size_t) dd * nrel;
                    for (int k2 = 0; k2 < nrel; ++k2) {
                        const float t = tt[k2];
#pragma unroll
                        for (int p = 0; p < PER; ++p) dot[p] += rv[p][k2] * t;
                    }
                }
            }
#pragma unroll
            for (int p = 0; p < PER; ++p) s[h2][p] = j < j1 ? dot[p] : -INFINITY;
        }
        float cm[PER], p0[PER], p1[PER], cz[PER];
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            float v = fmaxf(s[0][p], s[1][p]);
            for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
            cm[p] = v;
            p0[p] = s[0][p] == -INFINITY ? 0.f : expf(s[0][p] - v);
            p1[p] = s[1][p] == -INFINITY ? 0.f : expf(s[1][p] - v);
            cz[p] = warp_sum(p0[p] + p1[p]);
        }
        // the values: 4 keys a step, 8 lanes a key and 16 values a lane (sub, dg), straight into acc16 with the
        // chunk's weights under the running largest
        float pe0[PER], pe1[PER];
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const float nm = fmaxf(m[p], cm[p]), f = m[p] == -INFINITY ? 0.f : expf(m[p] - nm), e = expf(cm[p] - nm);
            l[p] = l[p] * f + cz[p] * e;
#pragma unroll
            for (int i = 0; i < DPL; ++i) acc16[p][i] *= f;
            pe0[p] = p0[p] * e, pe1[p] = p1[p] * e;
            m[p] = nm;
        }
        const int nk = (int) (j1 - j0);
        constexpr int STEPS = 16 / KPS;   // the steps whose reads go together: 16 keys
        for (int kb = 0; kb < nk; kb += 16) {
            uint4 cw[STEPS];
            float vsc[STEPS];
#pragma unroll
            for (int st = 0; st < STEPS; ++st) {
                const int kk = kb + KPS * st + sub;
                cw[st] = make_uint4(0u, 0u, 0u, 0u), vsc[st] = 0.f;
                if (kk < nk) {
                    if (from_ring) {
                        const size_t at = (size_t) ((j0 + kk) % n_ring) * ring.row + (size_t) g * kZhaDim + DPL * dg;
                        if constexpr (DPL == 16) cw[st] = *reinterpret_cast<const uint4*>(ring.vc + at);
                        else {
                            const uint2 u = *reinterpret_cast<const uint2*>(ring.vc + at);
                            cw[st] = make_uint4(u.x, u.y, 0u, 0u);
                        }
                        vsc[st] = __half2float(__ushort_as_half(ring.vs[at / 32]));
                    } else {
                        const size_t at = ((size_t) slot * C + kk) * kZhaDim + DPL * dg;
                        if constexpr (DPL == 16) {
                            const uint2 u = *reinterpret_cast<const uint2*>(pool.vc + at / 2);
                            cw[st] = make_uint4(u.x, u.y, 0u, 0u);
                        } else {
                            cw[st].x = *reinterpret_cast<const uint32_t*>(pool.vc + at / 2);
                        }
                        vsc[st] = __half2float(__ushort_as_half(pool.vs[at / 32]));
                    }
                }
            }
#pragma unroll
            for (int st = 0; st < STEPS; ++st) {
                const int k0 = kb + KPS * st, kk = k0 + sub;
                float wv[PER];
#pragma unroll
                for (int p = 0; p < PER; ++p) wv[p] = __shfl_sync(0xffffffffu, k0 < 32 ? pe0[p] : pe1[p], (k0 & 31) + sub);
                if (kk < nk) {
                    float v[DPL];
                    const uint32_t ww[4] = {cw[st].x, cw[st].y, cw[st].z, cw[st].w};
                    if (from_ring) {
#pragma unroll
                        for (int i = 0; i < DPL; ++i) v[i] = (float) (int8_t) ((ww[i / 4] >> (8 * (i % 4))) & 0xff) * vsc[st];
                    } else {
#pragma unroll
                        for (int i = 0; i < DPL; ++i) v[i] = (float) ((int) ((ww[i / 8] >> (4 * (i % 8))) & 15) - 8) * vsc[st];
                    }
#pragma unroll
                    for (int p = 0; p < PER; ++p)
#pragma unroll
                        for (int i = 0; i < DPL; ++i) acc16[p][i] += wv[p] * v[i];
                }
            }
        }
    }
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        if (lane == 0) wm[w][p] = m[p], wl[w][p] = l[p];
#pragma unroll
        for (int i = 0; i < DPL; ++i) {   // the step's keys' sums of each value: lanes dg + LPK * sub
            float x = acc16[p][i];
#pragma unroll
            for (int o = LPK; o < 32; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
            if (sub == 0) wo[w][p][DPL * dg + i] = x;
        }
    }
    __syncthreads();
    if (w >= PER) return;
    const int p = w;   // warp p writes head p's part
    float M = -INFINITY;
    for (int k = 0; k < 4; ++k) M = fmaxf(M, wm[k][p]);
    float Z = 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < 4; ++k) {
        if (wm[k][p] == -INFINITY) continue;
        const float f = expf(wm[k][p] - M);
        Z += wl[k][p] * f;
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] += wo[k][p][4 * lane + i] * f;
    }
    float* pp = part + ((size_t) (g * PER + p) * seg_total + seg) * (kZhaDim + 2);
#pragma unroll
    for (int i = 0; i < 4; ++i) pp[4 * lane + i] = o[i];
    if (lane == 0) pp[kZhaDim] = M, pp[kZhaDim + 1] = Z;
}

// An answer's token's tail (blockIdx.x the head, blockIdx.y the segment): each far chunk it does not read exactly as
// one key, the values' summary of the chunk's best reference (4-bit codes) weighted by exp(score).  Parts at
// seg_base + seg.
__global__ void __launch_bounds__(128) zha_tail_dec_kernel(const float* __restrict__ score,
                                                           const uint8_t* __restrict__ rstar,
                                                           const uint8_t* __restrict__ sel, const __half* __restrict__ vhat,
                                                           size_t q8n, int max_chunks, long long pos, int n_head, int kv,
                                                           int R, int C, int recent, size_t hs, int sel_heads,
                                                           float* __restrict__ part, int seg_base, int seg_total) {
    __shared__ float red[32], wl[4], wo[4][kZhaDim];
    const int head = blockIdx.x, seg = blockIdx.y, segs = gridDim.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int per = n_head / kv, g = head / per;
    score += (size_t) head * hs;
    rstar += (size_t) head * hs;
    sel += (size_t) (sel_heads == 1 ? g : head) * hs;
    long long fr;
    const int nf = zha_far(pos, C, recent, &fr);
    const int cps = (nf + segs - 1) / segs, c0 = 1 + seg * cps, c1 = min(nf + 1, c0 + cps);   // chunks [c0, c1)
    float mx = -INFINITY;
    for (int c = c0 + threadIdx.x; c < c1; c += blockDim.x)
        if (!sel[c]) mx = fmaxf(mx, score[c]);
    const float M = zha_block_max(mx, red);
    // 4-bit summaries: 4 chunks a step, 8 lanes a chunk's row and 16 values a lane (sub, dg); the 8 steps of a warp's
    // 32 chunks read together
    const int sub = lane >> 3, dg = lane & 7;
    const uint8_t* codes = reinterpret_cast<const uint8_t*>(vhat);
    const uint16_t* scl = zq_scales(vhat, q8n);
    float z = 0.f, acc16[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc16[i] = 0.f;
    if (M != -INFINITY)
        for (int b0 = c0 + 32 * w; b0 < c1; b0 += 128) {
            const int c = b0 + lane;
            const bool live = c < c1 && !sel[c];
            const float wt = live ? expf(score[c] - M) : 0.f;
            const int r = live ? rstar[c] : 0;
            z += wt;
            uint2 cw[8];
            float sc[8], wv[8];
#pragma unroll
            for (int st = 0; st < 8; ++st) {
                const int jj = 4 * st + sub;
                wv[st] = __shfl_sync(0xffffffffu, wt, jj);
                const int rj = __shfl_sync(0xffffffffu, r, jj);
                const int cj = min(b0 + jj, c1 - 1);
                const size_t row = ((size_t) g * zq_cap(q8n, max_chunks) + cj) * R + rj;
                cw[st] = *reinterpret_cast<const uint2*>(codes + row * (kZhaDim / 2) + 8 * dg);
                sc[st] = __half2float(__ushort_as_half(scl[row * 4 + dg / 2]));
            }
#pragma unroll
            for (int st = 0; st < 8; ++st) {
                const float f = wv[st] * sc[st];
                const uint32_t ww[2] = {cw[st].x, cw[st].y};
#pragma unroll
                for (int i = 0; i < 16; ++i) acc16[i] += f * (float) ((int) ((ww[i / 8] >> (4 * (i % 8))) & 15) - 8);
            }
        }
    z = warp_sum(z);
    if (lane == 0) wl[w] = z;
#pragma unroll
    for (int i = 0; i < 16; ++i) {   // the 4 chunks' sums of each value, lanes dg, dg + 8, dg + 16, dg + 24
        float x = acc16[i];
        x += __shfl_xor_sync(0xffffffffu, x, 8);
        x += __shfl_xor_sync(0xffffffffu, x, 16);
        if (sub == 0) wo[w][16 * dg + i] = x;
    }
    __syncthreads();
    if (w != 0) return;
    float Z = 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < 4; ++k) {
        Z += wl[k];
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] += wo[k][4 * lane + i];
    }
    float* pp = part + ((size_t) head * seg_total + seg_base + seg) * (kZhaDim + 2);
#pragma unroll
    for (int i = 0; i < 4; ++i) pp[4 * lane + i] = o[i];
    if (lane == 0) pp[kZhaDim] = Z > 0.f ? M : -INFINITY, pp[kZhaDim + 1] = Z;
}

// The parts of zha_attend_kernel's segments for a token's head (blockIdx.x, blockIdx.y + head0), with its sink.
__global__ void zha_join_kernel(const float* __restrict__ part, int segs, int n_head, int head0,
                                const float* __restrict__ sinks, float* __restrict__ out) {
    const int t = blockIdx.x, head = head0 + blockIdx.y, lane = threadIdx.x;
    const float* pp = part + ((size_t) t * n_head + head) * segs * (kZhaDim + 2);
    const float sink = sinks ? sinks[(size_t) t * n_head + head] : -INFINITY;
    float M = sink;
    for (int s = 0; s < segs; ++s) M = fmaxf(M, pp[s * (kZhaDim + 2) + kZhaDim]);
    float Z = sinks ? expf(sink - M) : 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < segs; ++s) {
        const float* x = pp + s * (kZhaDim + 2);
        if (x[kZhaDim] == -INFINITY) continue;
        const float f = expf(x[kZhaDim] - M);
        Z += x[kZhaDim + 1] * f;
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] += x[4 * lane + i] * f;
    }
    float* y = out + ((size_t) t * n_head + head) * kZhaDim + 4 * lane;
#pragma unroll
    for (int i = 0; i < 4; ++i) y[i] = o[i] / Z;
}

// The partials of a group's heads (part [rows][per][segs][130], natural-log tops) joined with each head's sink.
__global__ void zha_join_group_kernel(const float* __restrict__ part, int segs, int per, int n_head, int head0,
                                      const float* __restrict__ sinks, float* __restrict__ out) {
    const int t = blockIdx.x, p = blockIdx.y, head = head0 + p, lane = threadIdx.x;
    const float* pp = part + ((size_t) t * per + p) * segs * (kZhaDim + 2);
    const float sink = sinks ? sinks[(size_t) t * n_head + head] : -INFINITY;
    float M = sink;
    for (int s = 0; s < segs; ++s) M = fmaxf(M, pp[s * (kZhaDim + 2) + kZhaDim]);
    float Z = sinks ? expf(sink - M) : 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < segs; ++s) {
        const float* x = pp + s * (kZhaDim + 2);
        if (x[kZhaDim] == -INFINITY) continue;
        const float f = expf(x[kZhaDim] - M);
        Z += x[kZhaDim + 1] * f;
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] += x[4 * lane + i] * f;
    }
    float* y = out + ((size_t) t * n_head + head) * kZhaDim + 4 * lane;
#pragma unroll
    for (int i = 0; i < 4; ++i) y[i] = Z > 0.f ? o[i] / Z : 0.f;
}

// ---- the exact keys of a prompt batch on the tensor cores ----------------------------------------------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
#define ZHA_MMA 0
#else
#define ZHA_MMA 1
#endif

#if ZHA_MMA
__device__ __forceinline__ void zmma(float* c, const uint32_t* a, const uint32_t* b) {
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

__device__ __forceinline__ void zldm4(uint32_t* r, const __half* p, bool trans) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    if (trans)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ uint32_t zh2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}
#endif

constexpr int kZhaTokens = 64, kZhaKeys = 64, kZhaRow = kZhaDim + 8, kZhaWarps = 12;

// The exact keys of a prompt batch's tokens for KV group g, flash style: a launch block 64 tokens of the group's 3
// heads (12 warps: 4 of 16 tokens a head), walking the chunks any of its tokens reads exactly (chunk 0, the
// selected far ones from the staging buffer, the recent ones from the ring) 64 keys at a time; a token masks the
// keys it does not read.  bias [per][rows][extent] (natural units) adds the bias by distance.  Out: each token's
// head's (top, sum, values) unnormalized, part [rows][per][segs][130] at segment 0.

// The helpers of zha_exact_kernel (leaves in RAM as 4-bit, the ring Q8_0): a tile's raw bytes, [K codes][K scales]
// [V codes][V scales], 4-bit (64 bytes a key) from the staging buffer or Q8_0 (128) from the ring.
constexpr int kZhaRaw = 2 * (64 * 128 + 64 * 4 * 2);

__device__ __forceinline__ void zha_cp(void* dst, const void* src, int bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned d = (unsigned) __cvta_generic_to_shared(dst);
    if (bytes == 16) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
    else asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(d), "l"(src));
#else
    if (bytes == 16) *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    else *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(src);
#endif
}

__device__ __forceinline__ void zha_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void zha_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__device__ __forceinline__ void zha_raw_issue(int c, uint8_t* raw, const ZhaKv& stage, const int* cmap, const ZhaKv& ring,
                                              long long n_ring, int g, int C, int c_ring) {
    if (c >= c_ring) {   // Q8_0 rows of the ring: 128 code bytes and 4 scales of head g a key
        const long long r0 = ((long long) c * C) % n_ring;
        const size_t row = (size_t) ring.row;   // values a ring row (every KV head's)
        uint8_t* kcd = raw;
        uint8_t* ksd = raw + 64 * 128;
        uint8_t* vcd = raw + 64 * 128 + 64 * 8;
        uint8_t* vsd = vcd + 64 * 128;
        for (int i = threadIdx.x; i < 64 * 8; i += blockDim.x) {
            const int j = i >> 3, part = i & 7;
            const size_t at = (size_t) (r0 + j) * row + (size_t) g * kZhaDim + 16 * part;
            zha_cp(kcd + j * 128 + 16 * part, ring.kc + at, 16);
            zha_cp(vcd + j * 128 + 16 * part, ring.vc + at, 16);
        }
        for (int j = threadIdx.x; j < 64; j += blockDim.x) {
            const size_t at = ((size_t) (r0 + j) * row + (size_t) g * kZhaDim) / 32;
            zha_cp(ksd + j * 8, ring.ks + at, 8);
            zha_cp(vsd + j * 8, ring.vs + at, 8);
        }
    } else {   // 4-bit chunk of the staging buffer: its keys back to back (64 code bytes, 4 scales a key)
        const size_t slot = (size_t) cmap[c];
        const uint8_t* kc = reinterpret_cast<const uint8_t*>(stage.kc) + slot * C * 64;
        const uint8_t* vc = reinterpret_cast<const uint8_t*>(stage.vc) + slot * C * 64;
        const uint8_t* ks = reinterpret_cast<const uint8_t*>(stage.ks) + slot * C * 8;
        const uint8_t* vs = reinterpret_cast<const uint8_t*>(stage.vs) + slot * C * 8;
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            zha_cp(raw + 16 * i, kc + 16 * i, 16);
            zha_cp(raw + 64 * 128 + 64 * 8 + 16 * i, vc + 16 * i, 16);
        }
        for (int i = threadIdx.x; i < 32; i += blockDim.x) {
            zha_cp(raw + 64 * 128 + 16 * i, ks + 16 * i, 16);
            zha_cp(raw + 64 * 128 + 64 * 8 + 64 * 128 + 16 * i, vs + 16 * i, 16);
        }
    }
}

// A tile's raw bytes to the fp16 tiles: 8 values a thread at a time.
__device__ __forceinline__ void zha_raw_dequant(const uint8_t* raw, bool q8, __half (*ks)[kZhaDim + 8],
                                                __half (*vs)[kZhaDim + 8]) {
    const uint8_t* kcd = raw;
    const uint16_t* ksd = reinterpret_cast<const uint16_t*>(raw + 64 * 128);
    const uint8_t* vcd = raw + 64 * 128 + 64 * 8;
    const uint16_t* vsd = reinterpret_cast<const uint16_t*>(vcd + 64 * 128);
    for (int i = threadIdx.x; i < 64 * (kZhaDim / 8); i += blockDim.x) {
        const int r = i / (kZhaDim / 8), d = 8 * (i % (kZhaDim / 8));
        const float dk = __half2float(__ushort_as_half(ksd[r * 4 + d / 32]));
        const float dv = __half2float(__ushort_as_half(vsd[r * 4 + d / 32]));
        float xk[8], xv[8];
        if (q8) {
            const uint2 a = *reinterpret_cast<const uint2*>(kcd + r * 128 + d);
            const uint2 b = *reinterpret_cast<const uint2*>(vcd + r * 128 + d);
            const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
            const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
            for (int e = 0; e < 8; ++e) xk[e] = pa[e] * dk, xv[e] = pb[e] * dv;
        } else {
            const uint32_t a = *reinterpret_cast<const uint32_t*>(kcd + r * 64 + d / 2);
            const uint32_t b = *reinterpret_cast<const uint32_t*>(vcd + r * 64 + d / 2);
            for (int e = 0; e < 8; ++e) {
                xk[e] = (float) ((int) ((a >> (4 * e)) & 15) - 8) * dk;
                xv[e] = (float) ((int) ((b >> (4 * e)) & 15) - 8) * dv;
            }
        }
        for (int e = 0; e < 8; e += 2) {
            *reinterpret_cast<__half2*>(&ks[r][d + e]) = __floats2half2_rn(xk[e], xk[e + 1]);
            *reinterpret_cast<__half2*>(&vs[r][d + e]) = __floats2half2_rn(xv[e], xv[e + 1]);
        }
    }
}

__global__ void __launch_bounds__(kZhaWarps * 32)
zha_exact_kernel(const float* __restrict__ q, ZhaKv stage, const int* __restrict__ cmap, ZhaKv ring, long long n_ring,
                 const uint8_t* __restrict__ sel, int sel_heads, size_t hs, int max_chunks, long long pos, int rows,
                 int n_head, int g, int per, int C, int recent, const float* __restrict__ bias, int extent,
                 float* __restrict__ part, int segs, const uint32_t* __restrict__ bits, int words) {
#if ZHA_MMA
    __shared__ __align__(16) __half ks[kZhaKeys][kZhaRow];
    __shared__ __align__(16) __half vs[kZhaKeys][kZhaRow];
    __shared__ uint32_t want[32];   // a window of 1024 chunks: a bit each
    __shared__ int lst[1024], nlst; // the window's chunks in order
    extern __shared__ __align__(16) uint8_t zraw[];   // two stages of a tile's raw bytes (kZhaRaw each)
    constexpr float LOG2E = 1.44269504f;
    const int t0 = blockIdx.x * kZhaTokens, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int gid = lane >> 2, tig = lane & 3, p = w % per, head = g * per + p, tw = t0 + 16 * (w / per);
    const bool warp_live = w < per * 4;
    const int tok[2] = {tw + gid, tw + gid + 8};
    const float sc = rsqrtf((float) kZhaDim) * LOG2E;
    uint32_t qa[8][4];
    for (int k = 0; k < 8; ++k)
        for (int i = 0; i < 2; ++i)
            for (int hi = 0; hi < 2; ++hi) {
                float2 x = make_float2(0.f, 0.f);
                if (warp_live && tok[i] < rows)
                    x = *reinterpret_cast<const float2*>(q + ((size_t) tok[i] * n_head + head) * kZhaDim + 16 * k + 8 * hi + 2 * tig);
                qa[k][i + 2 * hi] = zh2(x.x * sc, x.y * sc);
            }
    long long own[2], fr[2];
    const uint8_t* srow[2];
    for (int i = 0; i < 2; ++i) {
        own[i] = pos + tok[i];
        zha_far(own[i], C, recent, &fr[i]);
        srow[i] = sel + (sel_heads > 1 ? (size_t) p * hs : 0) + (size_t) min(tok[i], rows - 1) * max_chunks;
    }
    const int t_last = min(t0 + kZhaTokens, rows) - 1;
    long long fr_first, fr_last;
    zha_far(pos + t0, C, recent, &fr_first);
    const int nf_last = zha_far(pos + t_last, C, recent, &fr_last);
    const int c_ring = (int) (fr_first / C);          // from here the ring holds the chunks
    const int c_end = (int) ((pos + t_last) / C) + 1;  // past the last key
    float o[16][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    const auto tile = [&](const int c) {
        // a warp's 16 tokens: the chunk only if one of them reads it exactly (the block's other tokens may)
        bool ex[2];
        for (int i = 0; i < 2; ++i)
            ex[i] = tok[i] < rows && ((long long) c * C >= fr[i] || c == 0 || srow[i][c]);
        if (warp_live && __any_sync(0xffffffffu, ex[0] || ex[1])) {
            float s[8][4];
            for (int n = 0; n < 8; ++n) {
                s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
                for (int k = 0; k < 8; k += 2) {
                    uint32_t bb[4];
                    zldm4(bb, &ks[8 * n + (lane & 7)][16 * k + 8 * (lane >> 3)], false);
                    zmma(s[n], qa[k], bb);
                    zmma(s[n], qa[k + 1], bb + 2);
                }
            }
            for (int i = 0; i < 2; ++i) {
                const bool row_live = tok[i] < rows;
                const bool exact_c = ex[i];
                float mx = -INFINITY;
                for (int n = 0; n < 8; ++n)
                    for (int j = 2 * i; j < 2 * i + 2; ++j) {
                        const long long key = (long long) c * C + 8 * n + 2 * tig + (j & 1);
                        if (!row_live || !exact_c || key > own[i]) {
                            s[n][j] = -INFINITY;
                        } else {
                            const long long dd = own[i] - key;
                            if (dd < extent) s[n][j] += bias[((size_t) p * rows + tok[i]) * extent + dd] * LOG2E;
                        }
                        mx = fmaxf(mx, s[n][j]);
                    }
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                const float now = fmaxf(m[i], mx), base = now == -INFINITY ? 0.f : now;
                const float f = exp2f(m[i] - base);
                m[i] = now;
                l[i] *= f;
                for (int n = 0; n < 16; ++n) o[n][2 * i] *= f, o[n][2 * i + 1] *= f;
                for (int n = 0; n < 8; ++n)
                    for (int j = 2 * i; j < 2 * i + 2; ++j) {
                        s[n][j] = exp2f(s[n][j] - base);
                        l[i] += s[n][j];
                    }
            }
            for (int k = 0; k < 4; ++k) {
                const uint32_t pa[4] = {zh2(s[2 * k][0], s[2 * k][1]), zh2(s[2 * k][2], s[2 * k][3]),
                                        zh2(s[2 * k + 1][0], s[2 * k + 1][1]), zh2(s[2 * k + 1][2], s[2 * k + 1][3])};
                for (int n = 0; n < 16; n += 2) {
                    uint32_t bb[4];
                    zldm4(bb, &vs[16 * k + (lane & 7) + 8 * ((lane >> 3) & 1)][8 * n + 8 * (lane >> 4)], true);
                    zmma(o[n], pa, bb);
                    zmma(o[n + 1], pa, bb + 2);
                }
            }
        }
    };
    for (int w0 = 0; w0 < c_end; w0 += 1024) {
        // the window's chunks any token reads: chunk 0, selected far ones, every recent one
        if (threadIdx.x < 32) {   // the block's selected far chunks (zha_bits_kernel), chunk 0 and the recent ones
            const int wi = (w0 >> 5) + threadIdx.x;
            uint32_t x = wi < words ? bits[(size_t) blockIdx.x * words + wi] : 0u;
            for (int k = 0; k < 32; ++k) {
                const int c = 32 * wi + k;
                if (c >= c_end) x &= ~(1u << k);
                else if (c == 0 || c >= c_ring) x |= 1u << k;
                else if (c > nf_last) x &= ~(1u << k);
            }
            want[threadIdx.x] = x;
        }
        __syncthreads();
        // the window's chunks a stage ahead: tile k+1's bytes copy while tile k computes
        if (threadIdx.x == 0) {   // blockIdx.y of gridDim.y splits takes every gridDim.y-th chunk
            int n = 0, k = 0;
            for (int b = 0; b < 32; ++b)
                for (uint32_t bits = want[b]; bits; bits &= bits - 1, ++k)
                    if (k % gridDim.y == blockIdx.y) lst[n++] = w0 + 32 * b + __ffs(bits) - 1;
            nlst = n;
        }
        __syncthreads();
        const int n_items = nlst;
        if (n_items > 0) zha_raw_issue(lst[0], zraw, stage, cmap, ring, n_ring, g, C, c_ring);
        zha_commit();
        for (int it = 0; it < n_items; ++it) {
            const int c = lst[it];
            if (it + 1 < n_items) {
                zha_raw_issue(lst[it + 1], zraw + ((it + 1) & 1) * kZhaRaw, stage, cmap, ring, n_ring, g, C, c_ring);
                zha_commit();
                zha_wait<1>();
            } else {
                zha_wait<0>();
            }
            __syncthreads();
            zha_raw_dequant(zraw + (it & 1) * kZhaRaw, c >= c_ring, ks, vs);
            __syncthreads();
            tile(c);
            __syncthreads();
        }
    }
    if (!warp_live) return;
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        if (tok[i] >= rows) continue;
        float* pp = part + (((size_t) tok[i] * per + p) * segs + blockIdx.y) * (kZhaDim + 2);   // split y's segment
        for (int n = 0; n < 16; ++n) {
            pp[8 * n + 2 * tig] = o[n][2 * i];
            pp[8 * n + 2 * tig + 1] = o[n][2 * i + 1];
        }
        if (tig == 0) {
            pp[kZhaDim] = m[i] == -INFINITY ? -INFINITY : m[i] * 0.69314718f;   // log2 units to natural
            pp[kZhaDim + 1] = l[i];
        }
    }
#else
    (void) q, (void) stage, (void) cmap, (void) ring, (void) n_ring, (void) sel, (void) sel_heads, (void) hs,
        (void) max_chunks, (void) pos, (void) rows, (void) n_head, (void) g, (void) per, (void) C, (void) recent,
        (void) bias, (void) extent, (void) part, (void) segs, (void) bits, (void) words;
#endif
}

// The chunk scores of a prompt batch for KV group g, fused: a launch block 64 tokens of the group's 3 heads (12
// warps: 4 of 16 tokens a head) walks the summaries 64 (chunk, reference) rows at a time (64 / R chunks), takes
// q.khat * scale + bhat on the tensor cores and the largest over each chunk's R references in registers; score and
// rstar [per][rows][max_chunks] (the best reference of each).
template <int R>
__global__ void __launch_bounds__(kZhaWarps * 32)
zha_score_kernel(const float* __restrict__ q, const __half* __restrict__ khat, const float* __restrict__ bhat,
                 int nch, int max_chunks, int rows, int n_head, int g, int per, float* __restrict__ score,
                 uint8_t* __restrict__ rstar, size_t hs, size_t q8n, long long pos = 0, int C = 0, int recent = 0,
                 float* __restrict__ top = nullptr) {
#if ZHA_MMA
    __shared__ __align__(16) __half ks[kZhaKeys][kZhaRow];
    __shared__ float bs[kZhaKeys];
    // top: each (head, token)'s best score over its far chunks 1..nf (zha_pick_kernel's), by atomics at the end
    int nft[2] = {0, 0};
    float tmax[2] = {-INFINITY, -INFINITY};
    constexpr int CPT = kZhaKeys / R;   // chunks a tile
    // a tile's scores and choices of the block's 64 tokens and 3 heads, written out a row of CPT chunks at a time
    __shared__ float sbuf[3][kZhaTokens][CPT];
    __shared__ uint8_t rbuf[3][kZhaTokens][CPT];
    const int t0 = blockIdx.x * kZhaTokens, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int gid = lane >> 2, tig = lane & 3, p = w % per, head = g * per + p, tw = t0 + 16 * (w / per);
    const bool warp_live = w < per * 4;
    const int tok[2] = {tw + gid, tw + gid + 8};
    const float sc = rsqrtf((float) kZhaDim);
    uint32_t qa[8][4];
    for (int k = 0; k < 8; ++k)
        for (int i = 0; i < 2; ++i)
            for (int hi = 0; hi < 2; ++hi) {
                float2 x = make_float2(0.f, 0.f);
                if (warp_live && tok[i] < rows)
                    x = *reinterpret_cast<const float2*>(q + ((size_t) tok[i] * n_head + head) * kZhaDim + 16 * k + 8 * hi + 2 * tig);
                qa[k][i + 2 * hi] = zh2(x.x * sc, x.y * sc);
            }
    const __half* kg = khat + (size_t) g * zq_cap(q8n, max_chunks) * R * kZhaDim;
    const float* bg = bhat + (size_t) g * zq_cap(q8n, max_chunks) * R;
    // blockIdx.y: a part of the chunks (a launch of a few token blocks fills the card that way)
    if (top)
        for (int i = 0; i < 2; ++i) {
            long long fr;
            nft[i] = tok[i] < rows ? zha_far(pos + tok[i], C, recent, &fr) : 0;
        }
    const int span = ((nch + gridDim.y - 1) / gridDim.y + CPT - 1) / CPT * CPT;
    const int cb = blockIdx.y * span, ce = min(nch, cb + span);
    for (int c0 = cb; c0 < ce; c0 += CPT) {
        const int nr = min(CPT, ce - c0) * R;   // the tile's rows
        for (int i = threadIdx.x; i < kZhaKeys * (kZhaDim / 8); i += blockDim.x) {
            const int r = i / (kZhaDim / 8), d = 8 * (i % (kZhaDim / 8));
            uint4 x = make_uint4(0, 0, 0, 0);
            if (r < nr) x = zsum8h(khat, q8n, ((size_t) g * zq_cap(q8n, max_chunks) + c0) * R + r, d);
            *reinterpret_cast<uint4*>(&ks[r][d]) = x;
        }
        for (int i = threadIdx.x; i < kZhaKeys; i += blockDim.x) bs[i] = i < nr ? bg[(size_t) c0 * R + i] : -INFINITY;
        __syncthreads();
        if (warp_live) {
            float s[8][4];
            for (int n = 0; n < 8; ++n) {
                s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
                for (int k = 0; k < 8; k += 2) {
                    uint32_t bb[4];
                    zldm4(bb, &ks[8 * n + (lane & 7)][16 * k + 8 * (lane >> 3)], false);
                    zmma(s[n], qa[k], bb);
                    zmma(s[n], qa[k + 1], bb + 2);
                }
            }
            // column 8 n + 2 tig + (j & 1) is (chunk c0 + col / R, reference col % R); a thread's two columns share
            // a chunk.  R 4: lanes tig 0, 1 hold one chunk of an n-tile, 2, 3 the other; R 8: the 4 lanes one chunk;
            // R 16: the 4 lanes of n-tiles n and n + 1 (n even).
            for (int i = 0; i < 2; ++i) {
                float best[8];
                int arg[8];
                for (int n = 0; n < 8; ++n) {
                    const int col = 8 * n + 2 * tig;
                    const float v0 = s[n][2 * i] + bs[col], v1 = s[n][2 * i + 1] + bs[col + 1];
                    best[n] = v1 > v0 ? v1 : v0;
                    arg[n] = v1 > v0 ? col + 1 : col;
                    const int span = R == 4 ? 2 : 4;   // lanes a chunk within the n-tile
                    for (int o = 1; o < span; o <<= 1) {
                        const float ob = __shfl_xor_sync(0xffffffffu, best[n], o);
                        const int oa = __shfl_xor_sync(0xffffffffu, arg[n], o);
                        if (ob > best[n] || (ob == best[n] && oa < arg[n])) best[n] = ob, arg[n] = oa;
                    }
                }
                if constexpr (R == 16)
                    for (int n = 0; n < 8; n += 2)
                        if (best[n + 1] > best[n]) best[n] = best[n + 1], arg[n] = arg[n + 1];
                if (!warp_live || tok[i] >= rows) continue;
                for (int n = 0; n < 8; ++n) {
                    bool writer;
                    if constexpr (R == 4) writer = (tig & 1) == 0;
                    else if constexpr (R == 8) writer = tig == 0;
                    else writer = tig == 0 && (n & 1) == 0;
                    const int c = c0 + arg[n] / R;
                    if (!writer || c >= ce) continue;
                    if (c >= 1 && c <= nft[i]) tmax[i] = fmaxf(tmax[i], best[n]);
                    sbuf[p][tok[i] - t0][c - c0] = best[n];
                    rbuf[p][tok[i] - t0][c - c0] = (uint8_t) (arg[n] % R);
                }
            }
        }
        __syncthreads();
        // the tile out: a (head, token) row's CPT chunks side by side
        const int nc = min(CPT, ce - c0), tn = min(kZhaTokens, rows - t0);
        for (int i = threadIdx.x; i < per * kZhaTokens * CPT; i += blockDim.x) {
            const int cc = i % CPT, t = (i / CPT) % kZhaTokens, pp = i / (CPT * kZhaTokens);
            if (cc >= nc || t >= tn) continue;
            const size_t at = (size_t) pp * hs + (size_t) (t0 + t) * max_chunks + c0 + cc;
            score[at] = sbuf[pp][t][cc];
            rstar[at] = rbuf[pp][t][cc];
        }
    }
    if (top && warp_live)   // the 4 lanes of a token row, then one atomic a (head, token) and block
        for (int i = 0; i < 2; ++i) {
            float m = tmax[i];
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
            if (tig == 0 && tok[i] < rows && m != -INFINITY) zha_atomic_max(top + (size_t) p * rows + tok[i], m);
        }
#else
    (void) q, (void) khat, (void) bhat, (void) nch, (void) max_chunks, (void) rows, (void) n_head, (void) g, (void) per,
        (void) score, (void) rstar, (void) hs, (void) q8n, (void) pos, (void) C, (void) recent, (void) top;
#endif
}

// A few tokens' chunk scores from int8 summaries: a thread a chunk of group blockIdx.y and token blockIdx.z, the
// group's heads' queries in shared memory, each head's best tangent and its reference into head * hs.
template <int R>
__global__ void zha_score_few_kernel(const float* __restrict__ q, const __half* __restrict__ khat, size_t q8n,
                                     const float* __restrict__ bhat, int nch, int max_chunks, int n_head, int per,
                                     float* __restrict__ score, uint8_t* __restrict__ rstar, size_t hs) {
    __shared__ float qs[4][kZhaDim];
    const int g = blockIdx.y, t = blockIdx.z;
    const float sc = rsqrtf((float) kZhaDim);
    for (int i = threadIdx.x; i < per * kZhaDim; i += blockDim.x)
        qs[i / kZhaDim][i % kZhaDim] = q[((size_t) t * n_head + g * per + i / kZhaDim) * kZhaDim + i % kZhaDim] * sc;
    __syncthreads();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nch) return;
    const int8_t* codes = reinterpret_cast<const int8_t*>(khat);
    const uint16_t* scales = zq_scales(codes, q8n);
    const bool q4 = zq_4(q8n);
    float best[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    int arg[4] = {0, 0, 0, 0};
    for (int r = 0; r < R; ++r) {
        const size_t row = ((size_t) g * zq_cap(q8n, max_chunks) + c) * R + r;
        const int8_t* cr = codes + row * (q4 ? kZhaDim / 2 : kZhaDim);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int b = 0; b < 4; ++b) {   // 32-value blocks
            const float d = __half2float(__ushort_as_half(scales[row * 4 + b]));
            float part[4] = {0.f, 0.f, 0.f, 0.f};
            if (q4) {
                const uint4 u = *reinterpret_cast<const uint4*>(cr + 16 * b);
                const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
                for (int k = 0; k < 4; ++k)
                    for (int n = 0; n < 8; ++n) {
                        const float x = (float) ((int) ((w4[k] >> (4 * n)) & 15) - 8);
                        for (int p = 0; p < per; ++p) part[p] += x * qs[p][32 * b + 8 * k + n];
                    }
            } else {
                for (int k = 0; k < 32; k += 4) {
                    const char4 x = *reinterpret_cast<const char4*>(cr + 32 * b + k);
                    for (int p = 0; p < per; ++p)
                        part[p] += x.x * qs[p][32 * b + k] + x.y * qs[p][32 * b + k + 1] + x.z * qs[p][32 * b + k + 2] +
                                   x.w * qs[p][32 * b + k + 3];
                }
            }
            for (int p = 0; p < per; ++p) acc[p] += part[p] * d;
        }
        const float bb = bhat[row];
        for (int p = 0; p < per; ++p)
            if (acc[p] + bb > best[p]) best[p] = acc[p] + bb, arg[p] = r;
    }
    for (int p = 0; p < per; ++p) {
        const size_t at = (size_t) (g * per + p) * hs + (size_t) t * max_chunks + c;
        score[at] = best[p];
        rstar[at] = (uint8_t) arg[p];
    }
}

// The tail of a prompt batch on the tensor cores, for KV group g: a launch block 64 tokens of the group's 3 heads
// (12 warps: 4 of 16 tokens a head) walks the value estimates 64 (chunk, reference) rows at a time; a token's weight
// of a row is exp(its score of the chunk) where the reference is its best one and the chunk is a far one it did not
// select, else 0, and the weights times the tile go in as flash attention's P V does.  The head's (top, sum, values)
// go to part [rows][per][segs][130] at segment `seg`.
template <int R>
__global__ void __launch_bounds__(kZhaWarps * 32)
zha_tail_mma_kernel(const float* __restrict__ score, const uint8_t* __restrict__ sel, const uint8_t* __restrict__ rstar,
                    const __half* __restrict__ vhat, int max_chunks, long long pos, int rows, int g, int per, int C,
                    int recent, size_t hs, int sel_heads, float* __restrict__ part, int segs, int seg, size_t q8n) {
#if ZHA_MMA
    __shared__ __align__(16) __half vs[kZhaKeys][kZhaRow];
    constexpr int CPT = kZhaKeys / R;
    const int t0 = blockIdx.x * kZhaTokens, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int gid = lane >> 2, tig = lane & 3, p = w % per, tw = t0 + 16 * (w / per);
    const bool warp_live = w < per * 4;
    const int tok[2] = {tw + gid, tw + gid + 8};
    int nf[2];
    const float* sc[2];
    const uint8_t* rs[2];
    const uint8_t* se[2];
    for (int i = 0; i < 2; ++i) {
        long long fr;
        const int t = min(tok[i], rows - 1);
        nf[i] = tok[i] < rows ? zha_far(pos + tok[i], C, recent, &fr) : 0;
        sc[i] = score + (size_t) p * hs + (size_t) t * max_chunks;
        rs[i] = rstar + (size_t) p * hs + (size_t) t * max_chunks;
        se[i] = sel + (sel_heads > 1 ? (size_t) p * hs : 0) + (size_t) t * max_chunks;
    }
    long long frl;
    const int nf_last = zha_far(pos + min(t0 + kZhaTokens, rows) - 1, C, recent, &frl);
    const __half* vg = vhat + (size_t) g * zq_cap(q8n, max_chunks) * R * kZhaDim;
    float o[16][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    // blockIdx.y: a part of the far chunks, its sums at segment seg + blockIdx.y
    const int span = ((nf_last + 1 + gridDim.y - 1) / gridDim.y + CPT - 1) / CPT * CPT;
    const int cb = blockIdx.y * span, ce = min(nf_last + 1, cb + span);
    for (int c0 = cb; c0 < ce; c0 += CPT) {
        const int nr = min(CPT, ce - c0) * R;
        for (int i = threadIdx.x; i < kZhaKeys * (kZhaDim / 8); i += blockDim.x) {
            const int r = i / (kZhaDim / 8), d = 8 * (i % (kZhaDim / 8));
            uint4 x = make_uint4(0, 0, 0, 0);
            if (r < nr) x = zsum8h(vhat, q8n, ((size_t) g * zq_cap(q8n, max_chunks) + c0) * R + r, d);
            *reinterpret_cast<uint4*>(&vs[r][d]) = x;
        }
        __syncthreads();
        if (warp_live) {
            // the weights in log units: a thread's columns 8 n + 2 tig + (0, 1) share a chunk
            float s[8][4];
            bool any = false;
            for (int n = 0; n < 8; ++n) {
                const int col = 8 * n + 2 * tig, c = c0 + col / R;
                for (int i = 0; i < 2; ++i) {
                    float v0 = -INFINITY, v1 = -INFINITY;
                    if (c >= 1 && c <= nf[i] && !se[i][c]) {
                        const int r = rs[i][c];
                        const float x = sc[i][c];
                        if (col % R == r) v0 = x;
                        if ((col + 1) % R == r) v1 = x;
                        any = true;
                    }
                    s[n][2 * i] = v0, s[n][2 * i + 1] = v1;
                }
            }
            if (__any_sync(0xffffffffu, any)) {
                for (int i = 0; i < 2; ++i) {
                    float mx = -INFINITY;
                    for (int n = 0; n < 8; ++n) mx = fmaxf(mx, fmaxf(s[n][2 * i], s[n][2 * i + 1]));
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                    const float now = fmaxf(m[i], mx), base = now == -INFINITY ? 0.f : now;
                    const float f = expf(m[i] - base);
                    m[i] = now;
                    l[i] *= f;
                    for (int n = 0; n < 16; ++n) o[n][2 * i] *= f, o[n][2 * i + 1] *= f;
                    for (int n = 0; n < 8; ++n)
                        for (int j = 2 * i; j < 2 * i + 2; ++j) {
                            s[n][j] = expf(s[n][j] - base);
                            l[i] += s[n][j];
                        }
                }
                for (int k = 0; k < 4; ++k) {
                    const uint32_t pa[4] = {zh2(s[2 * k][0], s[2 * k][1]), zh2(s[2 * k][2], s[2 * k][3]),
                                            zh2(s[2 * k + 1][0], s[2 * k + 1][1]), zh2(s[2 * k + 1][2], s[2 * k + 1][3])};
                    for (int n = 0; n < 16; n += 2) {
                        uint32_t bb[4];
                        zldm4(bb, &vs[16 * k + (lane & 7) + 8 * ((lane >> 3) & 1)][8 * n + 8 * (lane >> 4)], true);
                        zmma(o[n], pa, bb);
                        zmma(o[n + 1], pa, bb + 2);
                    }
                }
            }
        }
        __syncthreads();
    }
    if (!warp_live) return;
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        if (tok[i] >= rows) continue;
        float* pp = part + (((size_t) tok[i] * per + p) * segs + seg + blockIdx.y) * (kZhaDim + 2);
        for (int n = 0; n < 16; ++n) {
            pp[8 * n + 2 * tig] = o[n][2 * i];
            pp[8 * n + 2 * tig + 1] = o[n][2 * i + 1];
        }
        if (tig == 0) pp[kZhaDim] = m[i], pp[kZhaDim + 1] = l[i];
    }
#else
    (void) score, (void) sel, (void) rstar, (void) vhat, (void) max_chunks, (void) pos, (void) rows, (void) g, (void) per,
        (void) C, (void) recent, (void) hs, (void) sel_heads, (void) part, (void) segs, (void) seg, (void) q8n;
#endif
}

// The cache's part of a step for KV head g: each chunk the step reads (cmap >= 0) the cache holds is stamped and
// mapped (g * P + its place); the others are misses (aux: [hand, count, chunks...]).
__global__ void zha_lookup_kernel(int* __restrict__ cmap, int nch, int g, int P, int max_chunks, int now,
                                  int* __restrict__ stamp, const int* __restrict__ pmap, int* __restrict__ aux) {
    if (g < 0) g = blockIdx.y, cmap += (size_t) g * max_chunks;   // every group at once
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nch || cmap[c] < 0) return;
    const int s = pmap[(size_t) g * max_chunks + c];
    if (s >= 0) {
        stamp[(size_t) g * P + s] = now;
        cmap[c] = g * P + s;
    } else {
        int* a = aux + (size_t) g * (P + 2);
        const int i = atomicAdd(&a[1], 1);
        if (i < P) a[2 + i] = c;
    }
}

// One thread: each miss takes the next place past the hand that this step does not read, evicting its chunk.
__global__ void zha_assign_kernel(int* __restrict__ cmap, int g, int P, int max_chunks, int now, int* __restrict__ owner,
                                  int* __restrict__ stamp, int* __restrict__ pmap, int* __restrict__ aux) {
    if (g < 0) g = blockIdx.x, cmap += (size_t) g * max_chunks;   // a block a group
    int* a = aux + (size_t) g * (P + 2);
    int* own = owner + (size_t) g * P;
    int* st = stamp + (size_t) g * P;
    int* pm = pmap + (size_t) g * max_chunks;
    const int n = min(a[1], P);
    int h = a[0];
    for (int i = 0; i < n; ++i) {
        const int c = a[2 + i];
        for (int k = 0; k < P && st[h] == now; ++k) h = (h + 1) % P;
        if (own[h] >= 0) pm[own[h]] = -1;
        own[h] = c;
        pm[c] = h;
        st[h] = now;
        cmap[c] = (g * P + h) | kZhaMiss;
        h = (h + 1) % P;
    }
    a[0] = h;
}

// The same, every group at once (a block a group, P <= blockDim): the i-th miss takes the i-th place past the hand
// that this step does not read (a scan of the places in the clock's order), evicting its chunk.
__global__ void __launch_bounds__(1024) zha_assign_par_kernel(int* __restrict__ cmap, int P, int max_chunks, int now,
                                                              int* __restrict__ owner, int* __restrict__ stamp,
                                                              int* __restrict__ pmap, int* __restrict__ aux) {
    __shared__ int sums[32];
    const int g = blockIdx.x, k = threadIdx.x, lane = k & 31, w = k >> 5;
    cmap += (size_t) g * max_chunks;
    int* a = aux + (size_t) g * (P + 2);
    int* own = owner + (size_t) g * P;
    int* st = stamp + (size_t) g * P;
    int* pm = pmap + (size_t) g * max_chunks;
    const int n = min(a[1], P), hand = a[0];
    if (n == 0) return;
    const int h = (hand + k) % P;
    const bool open = k < P && st[h] != now;
    const unsigned m = __ballot_sync(0xffffffffu, open);
    if (lane == 0) sums[w] = __popc(m);
    __syncthreads();
    if (w == 0) {   // the warps' counts, an inclusive scan
        int v = lane < (int) (blockDim.x >> 5) ? sums[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int t = __shfl_up_sync(0xffffffffu, v, o);
            if (lane >= o) v += t;
        }
        sums[lane] = v;
    }
    __syncthreads();
    const int i = __popc(m & ((1u << lane) - 1)) + (w ? sums[w - 1] : 0);
    if (!open || i >= n) return;
    const int c = a[2 + i];
    if (own[h] >= 0) pm[own[h]] = -1;
    own[h] = c;
    pm[c] = h;
    st[h] = now;
    cmap[c] = (g * P + h) | kZhaMiss;
    if (i == n - 1) a[0] = (h + 1) % P;
}

// The same for 4-bit leaves (C 64), 16-byte reads: a block a miss, a thread a quarter row of codes of K and V (64 rows x 4),
// the first 64 also a row's scales; every read of a chunk in flight at once.
__global__ void __launch_bounds__(256) zha_fill_q4_kernel(ZhaKv leaves, const int* __restrict__ cmap,
                                                          const int* __restrict__ aux, int P, ZhaKv pool, int max_chunks) {
    const int g = blockIdx.y;
    cmap += (size_t) g * max_chunks;
    const int* a = aux + (size_t) g * (P + 2);
    const int misses = min(a[1], P);
    const int j = threadIdx.x >> 2, part = threadIdx.x & 3;   // row j of the chunk, its 16 bytes `part`
    for (int b = blockIdx.x; b < misses; b += gridDim.x) {
        const int c = a[2 + b], slot = cmap[c] & ~kZhaMiss;
        const size_t from = (size_t) ((long long) c * 64 + j) * leaves.row + (size_t) g * kZhaDim;   // in values
        const size_t to = ((size_t) slot * 64 + j) * kZhaDim;
        const uint4 k = *reinterpret_cast<const uint4*>(leaves.kc + from / 2 + 16 * part);
        const uint4 v = *reinterpret_cast<const uint4*>(leaves.vc + from / 2 + 16 * part);
        uint2 ks = make_uint2(0, 0), vs = ks;
        if (part == 0) {
            ks = *reinterpret_cast<const uint2*>(leaves.ks + from / 32);
            vs = *reinterpret_cast<const uint2*>(leaves.vs + from / 32);
        }
        *reinterpret_cast<uint4*>(const_cast<int8_t*>(pool.kc) + to / 2 + 16 * part) = k;
        *reinterpret_cast<uint4*>(const_cast<int8_t*>(pool.vc) + to / 2 + 16 * part) = v;
        if (part == 0) {
            *reinterpret_cast<uint2*>(const_cast<uint16_t*>(pool.ks) + to / 32) = ks;
            *reinterpret_cast<uint2*>(const_cast<uint16_t*>(pool.vs) + to / 32) = vs;
        }
    }
}

__global__ void zha_fill_int_kernel(int* __restrict__ p, size_t n, int v) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

// An answer step's maps and counts anew, every group's: cmap -1, the staged count, the list's count, the misses'.
__global__ void zha_step_reset_kernel(int* __restrict__ cmap, size_t n, int* __restrict__ count, int* __restrict__ glist,
                                      int max_chunks, int* __restrict__ aux, int P, int groups) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        cmap[i] = -1;
    if (blockIdx.x == 0 && (int) threadIdx.x < groups) {
        const int g = threadIdx.x;
        count[g] = 0;
        glist[(size_t) g * (max_chunks + 1)] = 0;
        aux[(size_t) g * (P + 2) + 1] = 0;
    }
}

// For each block of 64 tokens, the far chunks any of its tokens selected (sels selections, hs apart): a bit a
// chunk, words of 32.  A thread a word: its 32 bytes of each token's selection.
__global__ void zha_bits_kernel(const uint8_t* __restrict__ sel, int sels, size_t hs, int rows, int max_chunks,
                                uint32_t* __restrict__ bits, int words, unsigned long long* __restrict__ stats) {
    const int blk = blockIdx.x, wi = blockIdx.y * blockDim.x + threadIdx.x;
    if (wi >= words) return;
    const int t0 = blk * kZhaTokens, t1 = min(rows, t0 + kZhaTokens);
    uint4 acc0 = make_uint4(0, 0, 0, 0), acc1 = acc0;
    for (int h = 0; h < sels; ++h)
        for (int t = t0; t < t1; ++t) {
            const uint4* r = reinterpret_cast<const uint4*>(sel + h * hs + (size_t) t * max_chunks + 32 * (size_t) wi);
            const uint4 a = r[0], b = r[1];
            acc0.x |= a.x, acc0.y |= a.y, acc0.z |= a.z, acc0.w |= a.w;
            acc1.x |= b.x, acc1.y |= b.y, acc1.z |= b.z, acc1.w |= b.w;
        }
    const uint32_t v[8] = {acc0.x, acc0.y, acc0.z, acc0.w, acc1.x, acc1.y, acc1.z, acc1.w};
    uint32_t x = 0;
    for (int k = 0; k < 32; ++k)
        if ((v[k >> 2] >> (8 * (k & 3))) & 0xffu) x |= 1u << k;
    bits[(size_t) blk * words + wi] = x;
    if (stats) {   // the block's far chunks, and the pairs its tokens selected
        atomicAdd(&stats[0], (unsigned long long) __popc(x));
        unsigned long long pairs = 0;
        for (int h = 0; h < sels; ++h)
            for (int t = t0; t < t1; ++t) {
                const uint32_t* r = reinterpret_cast<const uint32_t*>(sel + h * hs + (size_t) t * max_chunks + 32 * (size_t) wi);
                for (int k = 0; k < 8; ++k) {
                    const uint32_t b = r[k];
                    pairs += ((b & 0xffu) != 0) + ((b >> 8 & 0xffu) != 0) + ((b >> 16 & 0xffu) != 0) + ((b >> 24) != 0);
                }
            }
        atomicAdd(&stats[1], pairs);
    }
}

// One token's head: chunk 0 and the selected far chunks exactly from the staging buffer, the recent keys from the
// ring (with the bias by distance), the other far chunks as one key each (their score and value estimate), and the
// sink.  Four warps take every fourth chunk, a lane 4 values; their sums meet at the end.
__global__ void __launch_bounds__(128)
zha_attend_kernel(const float* __restrict__ q, ZhaKv stage, const int* __restrict__ cmap, ZhaKv ring, long long n_ring,
                  const float* __restrict__ score, const uint8_t* __restrict__ sel, const uint8_t* __restrict__ rstar,
                  const __half* __restrict__ vhat, int max_chunks, long long pos, int n_head, int kv, int head, int R,
                  int C, int recent, int tail, const float* __restrict__ rr, const float* __restrict__ table, int extent,
                  int nrel, const float* __restrict__ sinks, float* __restrict__ out, size_t head_stride, int sel_heads,
                  float* __restrict__ part_out, int mode, int seg_base, int seg_total, int part_heads, int all,
                  size_t q8n) {
    __shared__ float rv[32], wm[4], wl[4], wo[4][kZhaDim];
    const int t = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    // the group's head blockIdx.y (or with `all`, head blockIdx.y of every group): its scores, value choices and
    // (each head's own, or the group's) selection
    head += blockIdx.y;
    const int g = head / (n_head / kv);
    if (all) {
        cmap += (size_t) g * max_chunks;
        if (sel_heads == 1) sel += (size_t) g * head_stride;
    }
    score += blockIdx.y * head_stride;
    rstar += blockIdx.y * head_stride;
    if (sel_heads > 1) sel += blockIdx.y * head_stride;
    const int seg = blockIdx.z, segs = gridDim.z;
    const long long own = pos + t;
    long long fr;
    zha_far(own, C, recent, &fr);
    const float sc = rsqrtf((float) kZhaDim);
    const float4 q4 = *reinterpret_cast<const float4*>(q + ((size_t) t * n_head + head) * kZhaDim + 4 * lane);
    if ((int) threadIdx.x < nrel) rv[threadIdx.x] = rr[((size_t) t * n_head + head) * nrel + threadIdx.x];
    __syncthreads();
    const uint8_t* mine = sel + (size_t) t * max_chunks;
    float m = -INFINITY, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    const auto take = [&](float s, const float (&v)[4]) {
        const float nm = fmaxf(m, s), f = expf(m - nm), e = expf(s - nm);
        l = l * f + e;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = acc[i] * f + e * v[i];
        m = nm;
    };
    const int last = (int) (own / C);
    const int cps = (last + segs) / segs, cb = seg * cps, ce = min(last + 1, cb + cps);   // this block's chunks
    for (int c = cb + w; c < ce; c += 4) {
        const bool recent_c = (long long) c * C >= fr;
        if (recent_c || c == 0 || mine[c]) {
            if (mode == 2) continue;   // the exact keys are another kernel's
            const long long j1 = min((long long) (c + 1) * C, own + 1);
            for (long long j = (long long) c * C; j < j1; ++j) {
                float k[4], v[4];
                if (recent_c) zload(ring, j % n_ring, g, lane, k, v);
                else zload(stage, (long long) (cmap[c] & ~kZhaMiss) * C + (j - (long long) c * C), g, lane, k, v);
                float s = warp_sum(q4.x * k[0] + q4.y * k[1] + q4.z * k[2] + q4.w * k[3]) * sc;
                const long long d = own - j;
                if (d < extent) {
                    const float* tt = table + (size_t) d * nrel;
                    float b = 0.f;
                    for (int k2 = 0; k2 < nrel; ++k2) b += rv[k2] * tt[k2];
                    s += b;
                }
                take(s, v);
            }
        } else if (tail) {
            float v[4];
            zsum4(vhat, q8n, ((size_t) g * zq_cap(q8n, max_chunks) + c) * R + rstar[(size_t) t * max_chunks + c], lane, v);
            take(score[(size_t) t * max_chunks + c], v);
        }
    }
    if (lane == 0) wm[w] = m, wl[w] = l;
#pragma unroll
    for (int i = 0; i < 4; ++i) wo[w][4 * lane + i] = acc[i];
    __syncthreads();
    if (w != 0) return;
    if (segs > 1 || mode == 2) {   // a part of the head's sums, for the join: (top, sum, values) unnormalized
        float M = -INFINITY;
        for (int k = 0; k < 4; ++k) M = fmaxf(M, wm[k]);
        float Z = 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
        for (int k = 0; k < 4; ++k) {
            if (wm[k] == -INFINITY) continue;
            const float f = expf(wm[k] - M);
            Z += wl[k] * f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o[i] += wo[k][4 * lane + i] * f;
        }
        const int ph = part_heads == n_head ? head : head % part_heads;   // a group's buffer: its heads
        float* pp = part_out + (((size_t) t * part_heads + ph) * seg_total + seg_base + seg) * (kZhaDim + 2);
#pragma unroll
        for (int i = 0; i < 4; ++i) pp[4 * lane + i] = o[i];
        if (lane == 0) pp[kZhaDim] = M, pp[kZhaDim + 1] = Z;
        return;
    }
    const float sink = sinks ? sinks[(size_t) t * n_head + head] : -INFINITY;
    float M = sink;
    for (int k = 0; k < 4; ++k) M = fmaxf(M, wm[k]);
    float Z = sinks ? expf(sink - M) : 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < 4; ++k) {
        if (wm[k] == -INFINITY) continue;
        const float f = expf(wm[k] - M);
        Z += wl[k] * f;
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] += wo[k][4 * lane + i] * f;
    }
    float* y = out + ((size_t) t * n_head + head) * kZhaDim + 4 * lane;
#pragma unroll
    for (int i = 0; i < 4; ++i) y[i] = o[i] / Z;
}

}  // namespace

void store_q4(const float* x, int8_t* codes, uint16_t* scales, int n, long long pos, int rows, long long ring,
              void* stream) {
    if (rows <= 0) return;
    if (n % 32) throw std::invalid_argument("ink::store_q4: rows of 32-value blocks");
    const size_t blocks = (size_t) rows * (n / 32);   // 8 threads a block
    store_q4_kernel<<<grid(blocks * 8, 256), 256, 0, cs(stream)>>>(x, codes, scales, n, pos, rows, ring);
}

void q4_to_q8(const int8_t* src_codes, const uint16_t* src_scales, long long from, int8_t* dst_codes,
              uint16_t* dst_scales, long long to, long long ring, int n, int rows, void* stream) {
    if (rows <= 0) return;
    const size_t blocks = (size_t) rows * (n / 32);
    q4_to_q8_kernel<<<grid(blocks, 128), 128, 0, cs(stream)>>>(src_codes, src_scales, from, dst_codes, dst_scales, to,
                                                               ring, n, rows);
}

void zha_cache_clear(ZhaCache& cache, int kv_heads, int max_chunks, void* stream) {
    if (cache.P <= 0) return;
    const size_t np = (size_t) kv_heads * cache.P, nm = (size_t) kv_heads * max_chunks;
    zha_fill_int_kernel<<<grid(np, 256), 256, 0, cs(stream)>>>(cache.owner, np, -1);
    zha_fill_int_kernel<<<grid(np, 256), 256, 0, cs(stream)>>>(cache.stamp, np, -1);
    zha_fill_int_kernel<<<grid(nm, 256), 256, 0, cs(stream)>>>(cache.pmap, nm, -1);
    zha_fill_int_kernel<<<grid((size_t) kv_heads * (cache.P + 2), 256), 256, 0, cs(stream)>>>(
        cache.aux, (size_t) kv_heads * (cache.P + 2), 0);
}

void zha_build(const float* refs, const ZhaKv& ring, long long n_ring, int c0, int c1, int C, int R, int kv_heads,
               int max_chunks, uint16_t* khat, uint16_t* vhat, float* bhat, void* stream, size_t q8n) {
    if (c1 <= c0) return;
    if (c1 > zq_cap(q8n, max_chunks)) throw std::logic_error("ink::zha_build: past the summaries' room");
    zha_build_kernel<<<dim3(c1 - c0, kv_heads), 128, 0, cs(stream)>>>(refs, ring, n_ring, c0, C, R, max_chunks,
                                                                        (__half*) khat, (__half*) vhat, bhat, q8n);
}

void zha_attend(void* blas, const Zha& z, const float* q, const ZhaKv& leaves, const ZhaKv& ring, long long n_ring,
                const uint16_t* khat, const uint16_t* vhat, const float* bhat, int built, int max_chunks, long long pos,
                int rows, int n_head, int kv_heads, const float* r, const float* table, int extent, int nrel,
                const float* sinks, const ZhaWork& wk, float* out, void* stream, ZhaCache* cache, bool keep, int sum_cap) {
    if (rows <= 0) return;
    if (z.C != kZhaKeys || z.R != 4 || !leaves.q4 || !ring.kc || ring.q4)
        throw std::logic_error("ink::zha_attend: chunks of 64, 4 references, 4-bit leaves and a Q8_0 ring");
    // a selection's chunks in shared memory: a score each (zha_pick_select_dec_kernel, zha_select_tile2/3_kernel)
    constexpr size_t kSmem = 64 * 1024 + 4096;
    constexpr int kTile = 64;         // a prompt's tokens choose their chunks together, 64 at a time
    constexpr int kSegExact = 16;     // an answer's token: the exact keys in 16 segments,
    constexpr int kSegTail = 32;      // the tail in 32
    constexpr int kFillBlocks = 4;    // the cache's misses: a few blocks a group keep the bus busy, the SMs free
    constexpr int kExactSplit = 3;    // a prompt's exact part in 3 blocks a tile (each every third chunk)
    const int cap = sum_cap > 0 ? sum_cap : max_chunks;   // the summaries' room a KV head (zq_cap())
    // the staging buffer: a head's chunks at a time (wk.heads 1), or every head's kept across a slot's calls
    const bool kept_stage = wk.heads == kv_heads;
    const int per = n_head / kv_heads;
    // a token at a time with the cache: its chunks from the VRAM it keeps, only the misses over the bus (a KV group's
    // heads choose together: one selection a group)
    const bool cached = cache && cache->P > 0 && rows == 1 && 1 + std::min(z.max_sel, std::max(built, 1)) <= cache->P;
    if (!cached && (!kept_stage || !keep)) {
        zha_fill_int_kernel<<<grid((size_t) wk.heads * max_chunks, 256), 256, 0, cs(stream)>>>(
            wk.cmap, (size_t) wk.heads * max_chunks, -1);
        if (cudaMemsetAsync(wk.count, 0, (size_t) wk.heads * sizeof(int), cs(stream)) != cudaSuccess)
            throw std::runtime_error("ink::zha_attend: memset");
    }
    if (cached) ++cache->now;
    const cublasHandle_t h = (cublasHandle_t) blas;
    const float one = 1.f, zero = 0.f;
    const int nch = std::max(built, 1);
    const size_t hs = (size_t) rows * max_chunks;   // a head's scores
    const size_t q8n = zha_q8n_make((size_t) kv_heads * cap * z.R, cap, true);   // the 4-bit summaries
    // A few tokens (an answer): every head's chunk scores first, a token's in one launch over the groups, so the
    // summaries are read once; and each head's attention in segments of its chunks, joined after.  A prompt batch: a
    // group's scores at a time, its tokens on the tensor cores.
    const bool few = rows <= 4;
    const int segs = few ? 32 : 1;
    // an answer's token through the cache with its own kernels (groups of 2 or 3 heads, selections that fit kSmem)
    const bool dec = cached && per > 1 && per <= 3 && (size_t) max_chunks * sizeof(float) <= kSmem;
    // An answer's token whose keys are all still in the ring (no far chunk, chunk 0 recent: zha_far()'s first recent
    // key 0): the exact part alone.  The path below would select nothing, fetch nothing and add an empty tail; its
    // list would hold chunk 0 only, read from the ring, so the same segments give the same sums.
    // STRATA_INK_ZHA_SHORT=0 takes the path below anyway.
    static const bool short_ok = [] { const char* e = std::getenv("STRATA_INK_ZHA_SHORT"); return !e || *e != '0'; }();
    if (dec && short_ok && pos + 1 - z.recent < z.C) {
        const int nl = built > 0 ? 1 : 0;
        if (per == 3 && nrel <= 16)
            zha_exact_dec_group_kernel<<<dim3(kv_heads, kSegExact), 128, 0, cs(stream)>>>(
                q, cache->pool, wk.cmap, wk.glist, ring, n_ring, max_chunks, pos, n_head, z.C, z.recent, r, table,
                extent, nrel, wk.S, kSegExact, nl);
        else
            zha_exact_dec_kernel<<<dim3(n_head, kSegExact), 128, 0, cs(stream)>>>(
                q, cache->pool, wk.cmap, wk.glist, ring, n_ring, max_chunks, pos, n_head, kv_heads, z.C, z.recent, r,
                table, extent, nrel, wk.S, kSegExact, nl);
        zha_join_kernel<<<dim3(rows, n_head), 32, 0, cs(stream)>>>(wk.S, kSegExact, n_head, 0, sinks, out);
        return;
    }
    if (dec && built > 0) {
        zha_score_dec_kernel<<<dim3(grid(nch, 32), kv_heads), 128, 0, cs(stream)>>>(
            q, (const int8_t*) khat, q8n, bhat, nch, max_chunks, n_head, per, wk.score, wk.rstar, hs);
    } else if (few && built > 0) {   // a thread a chunk of a group, its heads' scores at once
        zha_score_few_kernel<4><<<dim3(grid(nch, 128), kv_heads, rows), 128, 0, cs(stream)>>>(
            q, (const __half*) khat, q8n, bhat, nch, max_chunks, n_head, per, wk.score, wk.rstar, hs);
    }
    if (cached) {
        // every group at once: selections, the chunks they read, the cache's hits and misses, the attention
        const int words = max_chunks / 32;
        int sels = per;
        if (dec) {
            static bool attr = false;
            if (!attr) {
                cudaFuncSetAttribute(zha_pick_select_dec_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
                attr = true;
            }
            // a chunk the cache holds gets 1 added to its pick: an answer's tokens keep reading what VRAM has
            zha_pick_select_dec_kernel<<<kv_heads, 1024, (size_t) max_chunks * sizeof(float), cs(stream)>>>(
                wk.score, per, hs, max_chunks, pos, z.C, z.recent, z.budget, z.min_chunks, z.max_sel, wk.sel,
                cache->pmap, 1.f);
            sels = 1;
        } else if (per > 1) {
            zha_pick_kernel<<<dim3(rows, kv_heads), 128, 0, cs(stream)>>>(wk.score, per, hs, max_chunks, pos, z.C,
                                                                          z.recent, wk.score + (size_t) n_head * hs);
            zha_select_kernel<<<dim3(rows, kv_heads), 128, 0, cs(stream)>>>(
                wk.score + (size_t) n_head * hs, max_chunks, pos, z.C, z.recent, z.budget, z.min_chunks, z.max_sel,
                wk.sel, hs);
            sels = 1;
        } else {
            zha_select_kernel<<<dim3(rows, n_head), 128, 0, cs(stream)>>>(wk.score, max_chunks, pos, z.C, z.recent,
                                                                          z.budget, z.min_chunks, z.max_sel, wk.sel, hs);
        }
        zha_step_reset_kernel<<<64, 256, 0, cs(stream)>>>(wk.cmap, (size_t) kv_heads * max_chunks, wk.count, wk.glist,
                                                          max_chunks, cache->aux, cache->P, kv_heads);
        zha_union_kernel<<<dim3(grid(words, 64), kv_heads), 64, 0, cs(stream)>>>(wk.sel, sels, hs, rows, max_chunks,
                                                                                 built, wk.cmap, wk.count, 0, wk.glist);
        zha_lookup_kernel<<<dim3(grid(std::max(built, 1), 128), kv_heads), 128, 0, cs(stream)>>>(
            wk.cmap, built, -1, cache->P, max_chunks, cache->now, cache->stamp, cache->pmap, cache->aux);
        if (cache->P <= 1024)
            zha_assign_par_kernel<<<kv_heads, (cache->P + 31) / 32 * 32, 0, cs(stream)>>>(
                wk.cmap, cache->P, max_chunks, cache->now, cache->owner, cache->stamp, cache->pmap, cache->aux);
        else
            zha_assign_kernel<<<kv_heads, 1, 0, cs(stream)>>>(wk.cmap, -1, cache->P, max_chunks, cache->now,
                                                             cache->owner, cache->stamp, cache->pmap, cache->aux);
        const auto fill = [&](cudaStream_t s) {
            zha_fill_q4_kernel<<<dim3(kFillBlocks, kv_heads), 256, 0, s>>>(leaves, wk.cmap, cache->aux, cache->P,
                                                                           cache->pool, max_chunks);
        };
        if (!dec) {
            fill(cs(stream));
            zha_attend_kernel<<<dim3(rows, n_head, segs), 128, 0, cs(stream)>>>(
                q, cache->pool, wk.cmap, ring, n_ring, wk.score, wk.sel, wk.rstar, (const __half*) vhat, max_chunks,
                pos, n_head, kv_heads, 0, z.R, z.C, z.recent, 1, r, table, extent, nrel, sinks, out, hs, sels, wk.S, 1,
                0, segs, n_head, 1, q8n);
            zha_join_kernel<<<dim3(rows, n_head), 32, 0, cs(stream)>>>(wk.S, segs, n_head, 0, sinks, out);
            return;
        }
        // the misses cross the bus (the fill waits on PCIe) while the tail reads VRAM: on a stream of their own
        static cudaStream_t side = nullptr;
        static cudaEvent_t ready = nullptr, filled = nullptr;
        if (!side) {
            int lo = 0, hi = 0;   // the fill's stream first: its few blocks wait on the bus, not on the SMs
            cudaDeviceGetStreamPriorityRange(&lo, &hi);
            if (cudaStreamCreateWithPriority(&side, cudaStreamNonBlocking, hi) != cudaSuccess ||
                cudaEventCreateWithFlags(&ready, cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&filled, cudaEventDisableTiming) != cudaSuccess)
                throw std::runtime_error("ink::zha_attend: the fill's stream");
        }
        cudaEventRecord(ready, cs(stream));
        cudaStreamWaitEvent(side, ready, 0);
        fill(side);
        cudaEventRecord(filled, side);
        // the tail in segments, then (the fill done) the exact part: the selected chunks, chunk 0 and the recent keys,
        // all joined with the sinks
        constexpr int segs_all = kSegExact + kSegTail;
        zha_tail_dec_kernel<<<dim3(n_head, kSegTail), 128, 0, cs(stream)>>>(
            wk.score, wk.rstar, wk.sel, (const __half*) vhat, q8n, max_chunks, pos, n_head, kv_heads, z.R, z.C,
            z.recent, hs, sels, wk.S, kSegExact, segs_all);
        cudaStreamWaitEvent(cs(stream), filled, 0);
        if (per == 3 && nrel <= 16)   // a group's three heads in one launch
            zha_exact_dec_group_kernel<<<dim3(kv_heads, kSegExact), 128, 0, cs(stream)>>>(
                q, cache->pool, wk.cmap, wk.glist, ring, n_ring, max_chunks, pos, n_head, z.C, z.recent, r, table,
                extent, nrel, wk.S, segs_all, -1);
        else
            zha_exact_dec_kernel<<<dim3(n_head, kSegExact), 128, 0, cs(stream)>>>(
                q, cache->pool, wk.cmap, wk.glist, ring, n_ring, max_chunks, pos, n_head, kv_heads, z.C, z.recent, r,
                table, extent, nrel, wk.S, segs_all, -1);
        zha_join_kernel<<<dim3(rows, n_head), 32, 0, cs(stream)>>>(wk.S, segs_all, n_head, 0, sinks, out);
        return;
    }
    for (int g = 0; g < kv_heads; ++g) {
        float* sc = few ? wk.score + (size_t) g * per * hs : wk.score;   // the group's heads' scores, hs apart
        uint8_t* rs = few ? wk.rstar + (size_t) g * per * hs : wk.rstar;
        float* pick = few ? wk.score + (size_t) n_head * hs : wk.score + (size_t) per * hs;
        const bool fused = !few && built > 0;
        // a launch of a prompt call's 64-token blocks, cut further by chunks so it fills the card
        const int tblocks = (rows + kZhaTokens - 1) / kZhaTokens;
        const int split = std::max(1, std::min(8, (72 + tblocks - 1) / tblocks));
        // the scores' launch keeps each (head, token)'s best and zha_select_tile3_kernel picks from the scores
        const bool sel3 = fused && per > 1 && per <= 4 && (size_t) max_chunks * sizeof(float) <= kSmem;
        float* top = sel3 ? pick : nullptr;   // pick's room: per x rows floats
        if (sel3 && cudaMemsetAsync(top, 0xFF, (size_t) per * rows * sizeof(float), cs(stream)) != cudaSuccess)
            throw std::runtime_error("ink::zha_attend: memset");
        if (fused)   // the chunk scores on the tensor cores, the largest reference in registers
            zha_score_kernel<4><<<dim3(tblocks, std::min(split, std::max(1, nch / (kZhaKeys / 4)))), kZhaWarps * 32, 0,
                                  cs(stream)>>>(q, (const __half*) khat, bhat, nch, max_chunks, rows, n_head, g, per, sc,
                                                rs, hs, q8n, pos, z.C, z.recent, top);
        // the selections: the group's one (or each head's, in a group of one); a prompt's in tiles of tokens
        int sels = per;
        if (sel3) {
            static bool attr3 = false;
            if (!attr3) {
                cudaFuncSetAttribute(zha_select_tile3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
                attr3 = true;
            }
            const int tiles = (rows + kTile - 1) / kTile;
            zha_select_tile3_kernel<<<tiles, 1024, (size_t) max_chunks * sizeof(float), cs(stream)>>>(
                sc, per, hs, top, max_chunks, pos, rows, kTile, z.C, z.recent, z.budget, z.min_chunks, z.max_sel,
                wk.sel);
            sels = 1;
        } else if (per > 1) {
            zha_pick_kernel<<<rows, 128, 0, cs(stream)>>>(sc, per, hs, max_chunks, pos, z.C, z.recent, pick);
            if (few)
                zha_select_kernel<<<rows, 128, 0, cs(stream)>>>(pick, max_chunks, pos, z.C, z.recent, z.budget,
                                                                z.min_chunks, z.max_sel, wk.sel, 0);
            sels = 1;
        } else if (few) {
            zha_select_kernel<<<dim3(rows, per), 128, 0, cs(stream)>>>(sc, max_chunks, pos, z.C, z.recent, z.budget,
                                                                       z.min_chunks, z.max_sel, wk.sel, hs);
        }
        if (!few && !sel3) {
            const bool picks = sels == 1 && per > 1;
            const float* base = picks ? pick : sc;
            const int tiles = (rows + kTile - 1) / kTile;
            if ((size_t) max_chunks * sizeof(float) <= kSmem) {
                static bool attr = false;
                if (!attr) {
                    cudaFuncSetAttribute(zha_select_tile2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
                    attr = true;
                }
                zha_select_tile2_kernel<<<dim3(tiles, sels), 1024, (size_t) max_chunks * sizeof(float), cs(stream)>>>(
                    base, max_chunks, pos, rows, kTile, z.C, z.recent, z.budget, z.min_chunks, z.max_sel, wk.sel, hs,
                    picks);
            } else {
                float* comb = wk.S + (size_t) per * rows * extent + 65536;   // past the bias and the bits
                if ((size_t) (comb - wk.S) + (size_t) sels * tiles * max_chunks > wk.s_floats)
                    throw std::runtime_error("ink::zha_attend: the scratch is too small for tile selections");
                zha_select_tile_kernel<<<dim3(tiles, sels), 128, 0, cs(stream)>>>(
                    base, max_chunks, pos, rows, kTile, z.C, z.recent, z.budget, z.min_chunks, z.max_sel, wk.sel, hs,
                    comb);
            }
        }
        // the chunks they read, staged: from the leaves in RAM into VRAM of their own
        const int hk = kept_stage ? g : 0;   // the group's map
        int* cmap = wk.cmap + (size_t) hk * max_chunks;
        if (!kept_stage && g > 0) {   // the head before's places are not this one's
            zha_fill_int_kernel<<<grid((size_t) max_chunks, 256), 256, 0, cs(stream)>>>(cmap, (size_t) max_chunks, -1);
            if (cudaMemsetAsync(wk.count, 0, sizeof(int), cs(stream)) != cudaSuccess)
                throw std::runtime_error("ink::zha_attend: memset");
        }
        if (cudaMemsetAsync(wk.glist, 0, sizeof(int), cs(stream)) != cudaSuccess)
            throw std::runtime_error("ink::zha_attend: memset");
        zha_union_kernel<<<grid(max_chunks / 32, 64), 64, 0, cs(stream)>>>(wk.sel, sels, hs, rows, max_chunks, built,
                                                                           cmap, wk.count + hk, hk * max_chunks,
                                                                           wk.glist);
        zha_gather_kernel<<<std::max(built, 1), 256, 0, cs(stream)>>>(leaves, g, z.C, cmap, wk.glist, wk.stage);
        if (!few) {
            // a prompt batch: the exact keys on the tensor cores, the tail on its own, joined with the sinks.  In
            // wk.S: the group's bias by distance [per][rows][extent], the blocks' bits, the partials.
            float* bias = wk.S;
            const int words = max_chunks / 32, blocks = (rows + kZhaTokens - 1) / kZhaTokens;
            uint32_t* bits = reinterpret_cast<uint32_t*>(bias + (size_t) per * rows * extent);
            float* part = reinterpret_cast<float*>(bits + (size_t) blocks * words + 64);
            // the parts: the exact keys' at 0 .. kExactSplit - 1, the tail's after (the tensor-core tail, cut by
            // chunks)
            const int tsplit = std::max(1, std::min(split, nch / (kZhaKeys / 4))), psegs = kExactSplit + tsplit;
            if ((size_t) (part - wk.S) + (size_t) rows * per * psegs * (kZhaDim + 2) > wk.s_floats)
                throw std::runtime_error("ink::zha_attend: the scratch is too small for a prompt batch");
            for (int pp = 0; pp < per; ++pp) {   // bias[pp][t][d] = r[t][head].table[d]
                const cublasStatus_t st = cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, extent, rows, nrel, &one, table, nrel,
                                                      r + (size_t) (g * per + pp) * nrel, n_head * nrel, &zero,
                                                      bias + (size_t) pp * rows * extent, extent);
                if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("ink::zha_attend: bias product failed");
            }
            zha_bits_kernel<<<dim3(blocks, grid(words, 128)), 128, 0, cs(stream)>>>(wk.sel, sels, hs, rows, max_chunks,
                                                                                    bits, words, nullptr);
            static bool attr = false;
            if (!attr) {
                cudaFuncSetAttribute(zha_exact_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * kZhaRaw);
                attr = true;
            }
            zha_exact_kernel<<<dim3(blocks, kExactSplit), kZhaWarps * 32, 2 * kZhaRaw, cs(stream)>>>(
                q, wk.stage, cmap, ring, n_ring, wk.sel, sels, hs, max_chunks, pos, rows, n_head, g, per, z.C,
                z.recent, bias, extent, part, psegs, bits, words);
            zha_tail_mma_kernel<4><<<dim3(blocks, tsplit), kZhaWarps * 32, 0, cs(stream)>>>(
                sc, wk.sel, rs, (const __half*) vhat, max_chunks, pos, rows, g, per, z.C, z.recent, hs, sels, part,
                psegs, kExactSplit, q8n);
            zha_join_group_kernel<<<dim3(rows, per), 32, 0, cs(stream)>>>(part, psegs, per, n_head, g * per, sinks, out);
            continue;
        }
        zha_attend_kernel<<<dim3(rows, per, segs), 128, 0, cs(stream)>>>(
            q, wk.stage, cmap, ring, n_ring, sc, wk.sel, rs, (const __half*) vhat, max_chunks, pos, n_head, kv_heads,
            g * per, z.R, z.C, z.recent, 1, r, table, extent, nrel, sinks, out, hs, sels, wk.S, 1, 0, segs, n_head, 0,
            q8n);
        zha_join_kernel<<<dim3(rows, per), 32, 0, cs(stream)>>>(wk.S, segs, n_head, g * per, sinks, out);
    }
}

}  // namespace strata::poly::ink

#endif  // NVIDIA
