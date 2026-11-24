// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// src/quant/x86/dot_avx2.cpp - the Avx2 contract: ggml-cpu's dot products + bit-exact optimized AVX2 kernels for
// Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, IQ3_XXS (dual accumulators, unrolled shifts, bit-3 sign LUT, L1 prefetching,
// and 4-row shared-activation AVX2 dot_x4 kernels). Strictly AVX2+FMA+F16C only: 0 EVEX/ZMM instructions.
// (PolyStrata) Types ggml has no SIMD branch for (Q2_0) take the Generic code's bits under this contract
// (dot_q2_0_q8_0 below gives them in AVX2).
// (PolyStrata) dot_iq3_s_q8_K_words: ggml's IQ3_S bits with its grid words fetched in fewer instructions (below).
#include "tier_common.hpp"

#include "../generic.hpp"
#include "../registry.hpp"
#include "strata/quant/dot.hpp"

namespace strata::quant {
namespace avx2 {

#include "x86_helpers.inc"
#include "x86_dots.inc"

// ---- Q2_0 x Q8_0 in AVX2 (ported from the Inkling 2 engine of PR #200, src/poly/cpu_lowbit.cpp) ----

// One Q8_0 block's sum of (code - 1) * q, an exact integer, the block's 32 codes in the 8 bytes at `codes` (value
// 4i + m in byte i, bits 2m).  The activation's bytes go to "plane" order inside each 128-bit half (dword m holds
// values 4i + m, i = 0..3: one pshufb), where the codes of a plane are a byte shifted by 2m and masked: no
// interleave.  sum (c - 1) q = sum c q - sum q, both in int16 pairs (|pair| <= 6 * 127), then madd and a sum.
static inline int q2_0_block_sum(const uint8_t* STRATA_RESTRICT codes, const int8_t* STRATA_RESTRICT q) {
    const __m256i plane = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                           0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m256i qp = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i*) q), plane);
    uint64_t c8;
    std::memcpy(&c8, codes, 8);
    const __m256i cb = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(_mm_cvtsi64_si128((long long) c8)),
                                                   _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1));
    const __m256i c = _mm256_and_si256(_mm256_srlv_epi32(cb, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6)), _mm256_set1_epi8(3));
    const __m256i cq = _mm256_maddubs_epi16(c, qp);
    const __m256i sq = _mm256_maddubs_epi16(_mm256_set1_epi8(1), qp);
    return hsum_i32_8(_mm256_madd_epi16(_mm256_sub_epi16(cq, sq), _mm256_set1_epi16(1)));
}

// generic::dot_q2_0_q8_0_generic with the integer sums in AVX2: the same integers, and the float steps the generic
// code's, in its order and unfused (dot_generic.cpp is -ffp-contract=off), so the result is its bits.
void dot_q2_0_q8_0(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK2_0 == 0);
    const block_q2_0* STRATA_RESTRICT x = (const block_q2_0*) vx;
    const block_q8_0* STRATA_RESTRICT y = (const block_q8_0*) vy;
    const int nb = n / QK2_0;
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d0 = fp16_to_fp32(x[i].d);
        const int s0 = q2_0_block_sum(x[i].qs, y[2 * i].qs);
        const int s1 = q2_0_block_sum(x[i].qs + 8, y[2 * i + 1].qs);
        float sumi = 0.0f;
        float t0 = fp16_to_fp32(y[2 * i].d) * (float) s0;
        STRATA_NOFUSE(t0);
        sumi += t0;
        float t1 = fp16_to_fp32(y[2 * i + 1].d) * (float) s1;
        STRATA_NOFUSE(t1);
        sumi += t1;
        float t = d0 * sumi;
        STRATA_NOFUSE(t);
        sumf += t;
    }
    *s = sumf;
}


// ---- IQ1_S x Q8_K in AVX2 (ported from the Inkling 2 engine of PR #200, src/poly/cpu_lowbit.cpp) ----

// dot_iq1_s_q8_K (ggml's AVX2 kernel, above) with its scalar parts done in vectors, the same integers and the same
// float steps, written as it writes them:
//   * the 32 grid indices of a block, qs[k] | ((qh[k / 4] >> 3 (k % 4)) & 7) << 8, as 16-bit lanes at once (left
//     shifts by multiplies: qh << 8, << 5, << 2 by mullo, qh >> 1 by mulhi), then read out for the table loads (scalar
//     loads: AVX2's gathers took 1.25x as long on Zen 2);
//   * sumi1, the delta term: sum over the 8 groups of (bsums[2g] + bsums[2g + 1]) * (+-1) * ls, one madd of the
//     bsums, one product with the signed scales and a sum, the exact integer of the scalar loop.
void dot_iq1_s_q8_K_vidx(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_iq1_s* STRATA_RESTRICT x = (const block_iq1_s*) vx;
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;
    // a group's qh four times: 16-bit lane k of the 16 holds qh[k / 4] (one half of the block's 8 groups)
    const __m256i rep4 = _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
                                          4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7);
    const __m256i mul_lo = _mm256_setr_epi16(256, 32, 4, 0, 256, 32, 4, 0, 256, 32, 4, 0, 256, 32, 4, 0);
    const __m256i mul_hi = _mm256_setr_epi16(0, 0, 0, -32768, 0, 0, 0, -32768, 0, 0, 0, -32768, 0, 0, 0, -32768);
    const __m256i m700 = _mm256_set1_epi16(0x700);
    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {
        const int8_t* STRATA_RESTRICT q8 = y[i].qs;
        const uint16_t* STRATA_RESTRICT qh = x[i].qh;
        alignas(32) uint16_t idx[32];
        {
            const __m128i qh8 = _mm_loadu_si128((const __m128i*) qh);   // the 8 groups' qh
            const __m256i h0 = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(qh8), rep4);   // groups 0..3
            const __m256i h1 = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_srli_si128(qh8, 8)), rep4);
            const __m256i q0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) x[i].qs));
            const __m256i q1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) (x[i].qs + 16)));
            const __m256i i0 = _mm256_or_si256(
                q0, _mm256_and_si256(_mm256_or_si256(_mm256_mullo_epi16(h0, mul_lo), _mm256_mulhi_epu16(h0, mul_hi)), m700));
            const __m256i i1 = _mm256_or_si256(
                q1, _mm256_and_si256(_mm256_or_si256(_mm256_mullo_epi16(h1, mul_lo), _mm256_mulhi_epu16(h1, mul_hi)), m700));
            _mm256_store_si256((__m256i*) idx, i0);
            _mm256_store_si256((__m256i*) (idx + 16), i1);
        }
        __m256i sumi = _mm256_setzero_si256();
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 4
#endif
        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const uint16_t* id = idx + 4 * ib;
            const __m256i q1b_1 = _mm256_set_epi64x(iq1s_grid[id[3]], iq1s_grid[id[2]], iq1s_grid[id[1]], iq1s_grid[id[0]]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1s_grid[id[7]], iq1s_grid[id[6]], iq1s_grid[id[5]], iq1s_grid[id[4]]);
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*) (q8 + 32 * ib));
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32 * ib + 32));
            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
            const int16_t ls1 = 2 * ((qh[ib + 0] >> 12) & 7) + 1;
            const int16_t ls2 = 2 * ((qh[ib + 1] >> 12) & 7) + 1;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(ls2));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p1, p2));
        }
        int sumi1;
        {
            const __m256i gs = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i*) y[i].bsums), _mm256_set1_epi16(1));
            const __m256i h = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) qh));
            const __m256i ls = _mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 12), _mm256_set1_epi32(7)), 1),
                                                _mm256_set1_epi32(1));
            const __m256i sl = _mm256_sign_epi32(ls, _mm256_sub_epi32(_mm256_set1_epi32(1), _mm256_slli_epi32(_mm256_srli_epi32(h, 15), 1)));
            sumi1 = hsum_i32_8(_mm256_mullo_epi32(gs, sl));
        }
        // the float steps of dot_iq1_s_q8_K, as it writes them (so they contract as its own do under any compiler)
        const float d = y[i].d * h2f(x[i].d);
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        accum1 += d * sumi1;
    }
    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;
}

// ---- IQ3_S x Q8_K: dot_iq3_s_q8_K (ggml's AVX2 kernel, above) to the bit, its grid words fetched another way ----
//
// The same q8 sign flip ((q8 ^ s) - s), the same maddubs / madd by 2 ls + 1 lanes, the same int32 lane sums, one fmadd a block and the
// same hsum.  ggml's kernel builds the 9-bit grid indices in a vector, stores it and reads the 16 indices back one by one (each read waits
// on the store), then makes each 8-word vector from eight table loads by vpinsrd: compute-bound below the memory's speed on Zen 2.  Here
// the 16 indices of two groups are made as 16-bit lanes (the 9th bits from a table by qh byte), moved to two general registers four at a
// time (vmovq / vpextrq, no store), split by shifts, and each word read with vmovd and joined by unpacks.  (A vpgatherdd form was slower
// on Zen 2.)  STRATA_CPU_IQ3S=0 skips it (iq3s_own_on).
struct Iq3sHigh {   // qh byte -> its 8 bits as 16-bit lanes of 0 / 0x100
    alignas(16) uint16_t v[256][8];
    constexpr Iq3sHigh() : v{} {
        for (int h = 0; h < 256; ++h)
            for (int k = 0; k < 8; ++k) v[h][k] = (uint16_t) (((h >> k) & 1) << 8);
    }
};
constexpr Iq3sHigh kIq3sHigh{};

static inline __m128i iq3s_word(uint32_t i) { return _mm_cvtsi32_si128((int) iq3s_grid[i]); }
static inline __m128i iq3s_words4(uint64_t w) {   // the grid words of four 16-bit indices
    const uint32_t i0 = (uint32_t) w & 0xFFFF, i1 = (uint32_t) (w >> 16) & 0xFFFF, i2 = (uint32_t) (w >> 32) & 0xFFFF;
    const __m128i a = _mm_unpacklo_epi32(iq3s_word(i0), iq3s_word(i1));
    const __m128i b = _mm_unpacklo_epi32(iq3s_word(i2), iq3s_word((uint32_t) (w >> 48)));
    return _mm_unpacklo_epi64(a, b);
}
// The low / high 64 bits of v in a general register (x86-64), or through memory on 32-bit x86, which has no 64-bit moves.
static inline uint64_t lo64(__m128i v) {
#if defined(__x86_64__) || defined(_M_X64)
    return (uint64_t) _mm_cvtsi128_si64(v);
#else
    uint64_t w[2];
    _mm_storeu_si128((__m128i*) w, v);
    return w[0];
#endif
}
static inline uint64_t hi64(__m128i v) {
#if defined(__x86_64__) || defined(_M_X64)
    return (uint64_t) _mm_extract_epi64(v, 1);
#else
    uint64_t w[2];
    _mm_storeu_si128((__m128i*) w, v);
    return w[1];
#endif
}

void dot_iq3_s_q8_K_words(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_iq3_s* STRATA_RESTRICT x = (const block_iq3_s*) vx;
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;
    const __m256i mask1 = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                           2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i mask2 = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = h2f(x[i].d) * y[i].d;
        const uint8_t* STRATA_RESTRICT qs = x[i].qs;
        const uint8_t* STRATA_RESTRICT qh = x[i].qh;
        const uint8_t* STRATA_RESTRICT sg = x[i].signs;
        const int8_t* STRATA_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
            q8 += 64;
            const __m256i lo = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) qs));
            qs += 16;
            const __m256i hi = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_load_si128((const __m128i*) kIq3sHigh.v[qh[ib32]])),
                                                       _mm_load_si128((const __m128i*) kIq3sHigh.v[qh[ib32 + 1]]), 1);
            const __m256i idx = _mm256_or_si256(lo, hi);
            const __m128i il = _mm256_castsi256_si128(idx), ih = _mm256_extracti128_si256(idx, 1);
            const __m256i q2_1 = _mm256_inserti128_si256(_mm256_castsi128_si256(iq3s_words4(lo64(il))),
                                                         iq3s_words4(hi64(il)), 1);
            const __m256i q2_2 = _mm256_inserti128_si256(_mm256_castsi128_si256(iq3s_words4(lo64(ih))),
                                                         iq3s_words4(hi64(ih)), 1);
            uint32_t s1, s2;
            memcpy(&s1, sg, 4);
            memcpy(&s2, sg + 4, 4);
            sg += 8;
            const __m256i a1 = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) s1), mask1), mask2);
            const __m256i sm1 = _mm256_cmpeq_epi8(a1, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(sm1, q8_1), sm1);
            const __m256i a2 = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) s2), mask1), mask2);
            const __m256i sm2 = _mm256_cmpeq_epi8(a2, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(sm2, q8_2), sm2);
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32 / 2] & 0xf, ls2 = x[i].scales[ib32 / 2] >> 4;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16((short) (2 * ls1 + 1))));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16((short) (2 * ls2 + 1))));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = hsum_float_8(accumf);
}

// dot_iq3_xxs_q8_K for R activation rows at once.  A block's grid values, signs and scales are looked up once for
// all of them, and each row's terms, integer sums and float sums are the one-row function's in the same order, so
// s[r] is its bits.  The lookups were most of that function's time: 226 ns a row of 2,048 on lab-pro6000's cores.
template <int R>
void rows_iq3_xxs_q8_K(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* const* vy) {
    const block_iq3_xxs* STRATA_RESTRICT x = (const block_iq3_xxs*) vx;
    const block_q8_K* y[R];
    for (int r = 0; r < R; ++r) y[r] = (const block_q8_K*) vy[r];
    const int nb = n / QK_K;
    const uint64_t* signs64 = (const uint64_t*) keven_signs_q2xs;
    uint32_t aux32[2];
    __m256 accumf[R];
    for (int r = 0; r < R; ++r) accumf[r] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float dw = h2f(x[i].d);
        const uint8_t* STRATA_RESTRICT q3 = x[i].qs;
        const uint8_t* STRATA_RESTRICT gas = x[i].qs + QK_K / 4;
        __m256i sumi1[R], sumi2[R];
        for (int r = 0; r < R; ++r) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q2_1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            const __m256i q2_2 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            memcpy(aux32, gas, 8);
            gas += 8;
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                                   signs64[(aux32[0] >> 7) & 127], signs64[(aux32[0] >> 0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
            const uint16_t ls1 = aux32[0] >> 28;
            const uint16_t ls2 = aux32[1] >> 28;
            const __m256i sc1 = _mm256_set1_epi16(2 * ls1 + 1), sc2 = _mm256_set1_epi16(2 * ls2 + 1);
            for (int r = 0; r < R; ++r) {
                const int8_t* STRATA_RESTRICT q8 = y[r][i].qs + ib32 * 32;
                const __m256i q8s_1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) q8), s2_1);
                const __m256i q8s_2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) (q8 + 32)), s2_2);
                sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, q8s_1), sc1));
                sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, q8s_2), sc2));
            }
        }
        for (int r = 0; r < R; ++r) {
            const float d = dw * y[r][i].d;
            accumf[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[r], sumi2[r])),
                                        accumf[r]);
        }
    }
    for (int r = 0; r < R; ++r) s[r] = 0.25f * hsum_float_8(accumf[r]);
}

// dot_iq2_xxs_q8_K for R rows, as rows_iq3_xxs_q8_K is for its format.
template <int R>
void rows_iq2_xxs_q8_K(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* const* vy) {
    const block_iq2_xxs* STRATA_RESTRICT x = (const block_iq2_xxs*) vx;
    const block_q8_K* y[R];
    for (int r = 0; r < R; ++r) y[r] = (const block_q8_K*) vy[r];
    const int nb = n / QK_K;
    const uint64_t* signs64 = (const uint64_t*) keven_signs_q2xs;
    uint32_t aux32[4];
    const uint8_t* aux8 = (const uint8_t*) aux32;
    __m256 accumf[R];
    for (int r = 0; r < R; ++r) accumf[r] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float dw = h2f(x[i].d);
        const uint16_t* STRATA_RESTRICT q2 = x[i].qs;
        __m256i sumi1[R], sumi2[R];
        for (int r = 0; r < R; ++r) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            memcpy(aux32, q2, 4 * sizeof(uint32_t));
            q2 += 8;
            const __m256i q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                   signs64[(aux32[3] >> 7) & 127], signs64[(aux32[3] >> 0) & 127]);
            const uint16_t ls1 = aux32[1] >> 28;
            const uint16_t ls2 = aux32[3] >> 28;
            const __m256i sc1 = _mm256_set1_epi16(2 * ls1 + 1), sc2 = _mm256_set1_epi16(2 * ls2 + 1);
            for (int r = 0; r < R; ++r) {
                const int8_t* STRATA_RESTRICT q8 = y[r][i].qs + ib32 * 32;
                const __m256i q8s_1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) q8), s2_1);
                const __m256i q8s_2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) (q8 + 32)), s2_2);
                sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, q8s_1), sc1));
                sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, q8s_2), sc2));
            }
        }
        for (int r = 0; r < R; ++r) {
            const float d = dw * y[r][i].d;
            accumf[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[r], sumi2[r])),
                                        accumf[r]);
        }
    }
    for (int r = 0; r < R; ++r) s[r] = 0.125f * hsum_float_8(accumf[r]);
}

// dot_iq2_s_q8_K for R rows, as rows_iq3_xxs_q8_K is for its format.
template <int R>
void rows_iq2_s_q8_K(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* const* vy) {
    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};
    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    const block_iq2_s* STRATA_RESTRICT x = (const block_iq2_s*) vx;
    const block_q8_K* y[R];
    for (int r = 0; r < R; ++r) y[r] = (const block_q8_K*) vy[r];
    const int nb = n / QK_K;
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);
    const __m256i mask1 = _mm256_loadu_si256((const __m256i*) k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*) k_mask2);
    uint64_t aux64;
    __m256 accumf[R];
    for (int r = 0; r < R; ++r) accumf[r] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float dw = h2f(x[i].d);
        const uint8_t* STRATA_RESTRICT qs = x[i].qs;
        const uint8_t* STRATA_RESTRICT qh = x[i].qh;
        const uint16_t* STRATA_RESTRICT signs = (const uint16_t*) (x[i].qs + QK_K / 8);
        memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
        __m256i sumi1[R], sumi2[R];
        for (int r = 0; r < R; ++r) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32 + 0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32 + 0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32 + 0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32 + 0] << 8) & 0x300)]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32 + 1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32 + 1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32 + 1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32 + 1] << 8) & 0x300)]);
            qs += 8;
            __m256i aux256 = _mm256_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            aux256 = _mm256_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            signs += 4;
            const __m256i sc1 = _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32 + 0));
            const __m256i sc2 = _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32 + 1));
            for (int r = 0; r < R; ++r) {
                const int8_t* STRATA_RESTRICT q8 = y[r][i].qs + ib32 * 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
                const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
                const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
                sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, q8s_1), sc1));
                sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, q8s_2), sc2));
            }
        }
        for (int r = 0; r < R; ++r) {
            const float d = dw * y[r][i].d;
            accumf[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[r], sumi2[r])),
                                        accumf[r]);
        }
    }
    for (int r = 0; r < R; ++r) s[r] = 0.125f * hsum_float_8(accumf[r]);
}

// dot_q3_K_q8_K for R rows: the low and high bits and the scales unpacked once a block, every row's products, sums
// and float sums the one-row function's.
template <int R>
void rows_q3_K_q8_K(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* const* vy) {
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const block_q3_K* STRATA_RESTRICT x = (const block_q3_K*) vx;
    const block_q8_K* y[R];
    for (int r = 0; r < R; ++r) y[r] = (const block_q8_K*) vy[r];
    const int nb = n / QK_K;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);
    __m256 acc[R];
    for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
    uint32_t aux[3];
    for (int i = 0; i < nb; ++i) {
        const float dw = h2f(x[i].d);
        const uint8_t* STRATA_RESTRICT q3 = x[i].qs;
        memcpy(aux, x[i].scales, 12);
        __m128i scales128 = _mm_set_epi32(((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                                          ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                                          (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                                          (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};
        const __m256i hbits = _mm256_loadu_si256((const __m256i*) x[i].hmask);
        __m256i sumi[R];
        for (int r = 0; r < R; ++r) sumi[r] = _mm256_setzero_si256();
        int bit = 0;
        const int is = 0;   // as the one-row function has it: scales[j] is the half's, and its shuffles take 0..3
        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q3bits = _mm256_loadu_si256((const __m256i*) q3);
            q3 += 32;
            const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
            const __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;
            const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            const __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;
            const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            const __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;
            const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            const __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;
            const __m256i sc0 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 0));
            const __m256i sc1 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 1));
            const __m256i sc2 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 2));
            const __m256i sc3 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 3));
            for (int r = 0; r < R; ++r) {
                const int8_t* STRATA_RESTRICT q8 = y[r][i].qs + j * 128;
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) q8);
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 64));
                const __m256i q8_3 = _mm256_loadu_si256((const __m256i*) (q8 + 96));
                __m256i p16_0 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_0, q8_0), _mm256_maddubs_epi16(q3h_0, q8_0));
                __m256i p16_1 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_1, q8_1), _mm256_maddubs_epi16(q3h_1, q8_1));
                __m256i p16_2 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_2, q8_2), _mm256_maddubs_epi16(q3h_2, q8_2));
                __m256i p16_3 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_3, q8_3), _mm256_maddubs_epi16(q3h_3, q8_3));
                p16_0 = _mm256_madd_epi16(sc0, p16_0);
                p16_1 = _mm256_madd_epi16(sc1, p16_1);
                p16_2 = _mm256_madd_epi16(sc2, p16_2);
                p16_3 = _mm256_madd_epi16(sc3, p16_3);
                p16_0 = _mm256_add_epi32(p16_0, p16_1);
                p16_2 = _mm256_add_epi32(p16_2, p16_3);
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p16_0, p16_2));
            }
        }
        for (int r = 0; r < R; ++r) {
            const float d = y[r][i].d * dw;
            acc[r] = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }
    for (int r = 0; r < R; ++r) s[r] = hsum_float_8(acc[r]);
}

// R rows at a time, 4 at most, by the one-row kernel for a last row alone
template <void (*Two)(int, float*, const void*, const void* const*), void (*Three)(int, float*, const void*, const void* const*),
          void (*Four)(int, float*, const void*, const void* const*), void (*One)(int, float*, const void*, const void*)>
void rows_by_four(int n, int rows, float* s, const void* w, const void* const* a) {
    for (; rows >= 4; rows -= 4, s += 4, a += 4) Four(n, s, w, a);
    switch (rows) {
    case 3: Three(n, s, w, a); break;
    case 2: Two(n, s, w, a); break;
    case 1: One(n, s, w, a[0]); break;
    default: break;
    }
}

// ---- Bit-exact optimized AVX2 cold-expert dots (0 EVEX/ZMM instructions, 0 ULP diff vs x86_dots.inc) ----

void dot_q8_0_q8_0_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK8_0 == 0);
    const int nb = n / QK8_0;
    const block_q8_0* STRATA_RESTRICT x = (const block_q8_0*) vx;
    const block_q8_0* STRATA_RESTRICT y = (const block_q8_0*) vy;

    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    int ib = 0;

    for (; ib + 1 < nb; ib += 2) {
        if (ib + 4 < nb) {
            _mm_prefetch((const char*) &x[ib + 4], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[ib + 4], _MM_HINT_T0);
        }
        const __m256 d0 = _mm256_set1_ps(h2f(x[ib + 0].d) * h2f(y[ib + 0].d));
        const __m256 d1 = _mm256_set1_ps(h2f(x[ib + 1].d) * h2f(y[ib + 1].d));

        const __m256i qx0 = _mm256_loadu_si256((const __m256i*) x[ib + 0].qs);
        const __m256i qy0 = _mm256_loadu_si256((const __m256i*) y[ib + 0].qs);
        const __m256i qx1 = _mm256_loadu_si256((const __m256i*) x[ib + 1].qs);
        const __m256i qy1 = _mm256_loadu_si256((const __m256i*) y[ib + 1].qs);

        const __m256i p16_0 = _mm256_maddubs_epi16(_mm256_sign_epi8(qx0, qx0), _mm256_sign_epi8(qy0, qx0));
        const __m256i p16_1 = _mm256_maddubs_epi16(_mm256_sign_epi8(qx1, qx1), _mm256_sign_epi8(qy1, qx1));
        const __m256 q0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, p16_0));
        const __m256 q1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, p16_1));

        acc = _mm256_fmadd_ps(d0, q0, acc);
        acc = _mm256_fmadd_ps(d1, q1, acc);
    }
    for (; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(h2f(x[ib].d) * h2f(y[ib].d));
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i qy = _mm256_loadu_si256((const __m256i*) y[ib].qs);
        const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(qx, qx), _mm256_sign_epi8(qy, qx));
        acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, p16)), acc);
    }
    *s = hsum_float_8(acc);
}

void dot_q4_K_q8_K_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_q4_K* STRATA_RESTRICT x = (const block_q4_K*) vx;
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*) &x[i + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[i + 2], _MM_HINT_T0);
        }
        const float d = y[i].d * h2f(x[i].d);
        const float dmin = -y[i].d * h2f(x[i].dmin);

        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t* STRATA_RESTRICT q4 = x[i].qs;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_castsi256_si128(mins_and_scales);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();

        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
            const __m256i q8h = _mm256_loadu_si256((const __m256i*) (q8 + 32));
            q8 += 64;

            const __m256i p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l));
            const __m256i p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h));
            sumi0 = _mm256_add_epi32(sumi0, p16l);
            sumi1 = _mm256_add_epi32(sumi1, p16h);
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi0, sumi1)), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

void dot_q5_K_q8_K_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_q5_K* STRATA_RESTRICT x = (const block_q5_K*) vx;
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m16 = _mm256_set1_epi8(16);
    const __m128i mzero = _mm_setzero_si128();
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.f;

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*) &x[i + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[i + 2], _MM_HINT_T0);
        }
        const uint8_t* STRATA_RESTRICT q5 = x[i].qs;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;

        const float d = y[i].d * h2f(x[i].d);
        const float dmin = -y[i].d * h2f(x[i].dmin);

        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i sc128 = _mm256_castsi256_si128(mins_and_scales);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);
        const __m256i hbits = _mm256_loadu_si256((const __m256i*) x[i].qh);

        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();

#define Q5K_AVX2_STEP(J, QH0, QH1) do { \
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * (J) + 0)); \
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * (J) + 1)); \
            const __m256i q5bits = _mm256_loadu_si256((const __m256i*) (q5 + 32 * (J))); \
            const __m256i q5_0 = _mm256_or_si256(_mm256_and_si256(q5bits, m4), (QH0)); \
            const __m256i q5_1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4), (QH1)); \
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) (q8 + 64 * (J))); \
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) (q8 + 64 * (J) + 32)); \
            sumi0 = _mm256_add_epi32(sumi0, _mm256_madd_epi16(scale_0, _mm256_maddubs_epi16(q5_0, q8_0))); \
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(scale_1, _mm256_maddubs_epi16(q5_1, q8_1))); \
        } while (0)

        Q5K_AVX2_STEP(0, _mm256_and_si256(_mm256_slli_epi16(hbits, 4), m16), _mm256_and_si256(_mm256_slli_epi16(hbits, 3), m16));
        Q5K_AVX2_STEP(1, _mm256_and_si256(_mm256_slli_epi16(hbits, 2), m16), _mm256_and_si256(_mm256_slli_epi16(hbits, 1), m16));
        Q5K_AVX2_STEP(2, _mm256_and_si256(hbits, m16),                        _mm256_and_si256(_mm256_srli_epi16(hbits, 1), m16));
        Q5K_AVX2_STEP(3, _mm256_and_si256(_mm256_srli_epi16(hbits, 2), m16), _mm256_and_si256(_mm256_srli_epi16(hbits, 3), m16));
#undef Q5K_AVX2_STEP

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi0, sumi1)), acc);
    }

    *s = hsum_float_8(acc) + summs;
}

void dot_q6_K_q8_K_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_q6_K* STRATA_RESTRICT x = (const block_q6_K*) vx;
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m12 = _mm256_set1_epi8(12);
    const __m256i m48 = _mm256_set1_epi8(48);
    const __m256i m192 = _mm256_set1_epi8(-64);
    const __m256i m15 = _mm256_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*) &x[i + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[i + 2], _MM_HINT_T0);
        }
        const float d = y[i].d * h2f(x[i].d);
        const uint8_t* STRATA_RESTRICT q4 = x[i].ql;
        const uint8_t* STRATA_RESTRICT qh = x[i].qh;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i scales = _mm_loadu_si128((const __m128i*) x[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();
        int is = 0;

        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*) q4);
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*) (q4 + 32));
            q4 += 64;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*) qh);
            qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m12), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, m48);
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, m192), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) (q8 +  0));
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 64));
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*) (q8 + 96));
            q8 += 128;

            const __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            const __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            const __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            const __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            sumi0 = _mm256_add_epi32(sumi0, _mm256_add_epi32(_mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0),
                                                             _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1)));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(_mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2),
                                                             _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3)));
        }

        const __m256i sumi = _mm256_sub_epi32(_mm256_add_epi32(sumi0, sumi1), q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);
}

void dot_iq4_xs_q8_K_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_iq4_xs* STRATA_RESTRICT x = (const block_iq4_xs*) vx;
    const block_q8_K*   STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    // Unsigned |kvalues_iq4nl[k]| LUT + bit-3 sign mask eliminates both vpsignb(x,x) and dual 128-bit shuffles.
    static constexpr uint8_t kAbsIq4nl[16] = {127, 104, 83, 65, 49, 35, 22, 10, 1, 13, 25, 38, 53, 69, 89, 113};
    const __m256i abs256 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kAbsIq4nl));
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m256i bit3 = _mm256_set1_epi8(0x08);

    __m256 accum = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        if (ibl + 2 < nb) {
            _mm_prefetch((const char*) &x[ibl + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[ibl + 2], _MM_HINT_T0);
        }
        const uint8_t* STRATA_RESTRICT qs = x[ibl].qs;
        const int8_t*  STRATA_RESTRICT q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();

        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*) qs);
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*) (qs + 16));
            qs += 32;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*) q8);
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
            q8 += 64;

            const __m256i nib_1 = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bits_1, 4), q4bits_1), m4b);
            const __m256i nib_2 = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bits_2, 4), q4bits_2), m4b);
            const __m256i ax_1 = _mm256_shuffle_epi8(abs256, nib_1);
            const __m256i ax_2 = _mm256_shuffle_epi8(abs256, nib_2);
            const __m256i neg_1 = _mm256_cmpgt_epi8(bit3, nib_1);
            const __m256i neg_2 = _mm256_cmpgt_epi8(bit3, nib_2);
            const __m256i sy_1 = _mm256_sub_epi8(_mm256_xor_si256(q8b_1, neg_1), neg_1);
            const __m256i sy_2 = _mm256_sub_epi8(_mm256_xor_si256(q8b_2, neg_2), neg_2);

            const __m256i p16_1 = _mm256_maddubs_epi16(ax_1, sy_1);
            const __m256i p16_2 = _mm256_maddubs_epi16(ax_2, sy_2);

            const int16_t ls1 = (int16_t) (((x[ibl].scales_l[ib / 2] & 0xf) | ((sh << 4) & 0x30)) - 32);
            const int16_t ls2 = (int16_t) (((x[ibl].scales_l[ib / 2] >>  4) | ((sh << 2) & 0x30)) - 32);
            sh >>= 4;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(p16_1, _mm256_set1_epi16(ls1)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(p16_2, _mm256_set1_epi16(ls2)));
        }
        accum = _mm256_fmadd_ps(_mm256_set1_ps(h2f(x[ibl].d) * y[ibl].d),
                                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accum);
    }

    *s = hsum_float_8(accum);
}

void dot_iq3_xxs_q8_K_opt(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_iq3_xxs* STRATA_RESTRICT x = (const block_iq3_xxs*) vx;
    const block_q8_K*    STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    const uint64_t* signs64 = (const uint64_t*) keven_signs_q2xs;
    uint32_t aux32[2];

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*) &x[i + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[i + 2], _MM_HINT_T0);
        }
        const float d = h2f(x[i].d) * y[i].d;
        const uint8_t* STRATA_RESTRICT q3 = x[i].qs;
        const uint8_t* STRATA_RESTRICT gas = x[i].qs + QK_K / 4;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
            q8 += 64;
            const __m256i q2_1 = _mm256_set_epi32(
                (int) iq3xxs_grid[q3[ 7]], (int) iq3xxs_grid[q3[ 6]], (int) iq3xxs_grid[q3[ 5]], (int) iq3xxs_grid[q3[ 4]],
                (int) iq3xxs_grid[q3[ 3]], (int) iq3xxs_grid[q3[ 2]], (int) iq3xxs_grid[q3[ 1]], (int) iq3xxs_grid[q3[ 0]]);
            const __m256i q2_2 = _mm256_set_epi32(
                (int) iq3xxs_grid[q3[15]], (int) iq3xxs_grid[q3[14]], (int) iq3xxs_grid[q3[13]], (int) iq3xxs_grid[q3[12]],
                (int) iq3xxs_grid[q3[11]], (int) iq3xxs_grid[q3[10]], (int) iq3xxs_grid[q3[ 9]], (int) iq3xxs_grid[q3[ 8]]);
            q3 += 16;
            std::memcpy(aux32, gas, 8);
            gas += 8;
            const __m256i s2_1 = _mm256_set_epi64x(
                (long long) signs64[(aux32[0] >> 21) & 127], (long long) signs64[(aux32[0] >> 14) & 127],
                (long long) signs64[(aux32[0] >>  7) & 127], (long long) signs64[(aux32[0] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(
                (long long) signs64[(aux32[1] >> 21) & 127], (long long) signs64[(aux32[1] >> 14) & 127],
                (long long) signs64[(aux32[1] >>  7) & 127], (long long) signs64[(aux32[1] >>  0) & 127]);
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1, s2_1));
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2, s2_2));
            const int16_t ls1 = (int16_t) (2 * (aux32[0] >> 28) + 1);
            const int16_t ls2 = (int16_t) (2 * (aux32[1] >> 28) + 1);
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16(ls2)));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    *s = 0.25f * hsum_float_8(accumf);
}

// ---- 4-row shared-activation AVX2 kernels (zero EVEX/ZMM instructions; Alder/Raptor Lake safe) ----

void dot_q4_K_q8_K_x4_avx2(int n, float out[4], const void* STRATA_RESTRICT w_base, size_t row_bytes, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;
    const uint8_t* wb = (const uint8_t*) w_base;
    const block_q4_K* rows[4] = {
        (const block_q4_K*) (wb + 0 * row_bytes),
        (const block_q4_K*) (wb + 1 * row_bytes),
        (const block_q4_K*) (wb + 2 * row_bytes),
        (const block_q4_K*) (wb + 3 * row_bytes),
    };
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    __m128 acc_m[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            for (int r = 0; r < 4; ++r) _mm_prefetch((const char*) &rows[r][i + 2], _MM_HINT_T0);
        }
        const float yd = y[i].d;
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m256i q8v[8] = {
            _mm256_loadu_si256((const __m256i*) (y[i].qs +   0)),
            _mm256_loadu_si256((const __m256i*) (y[i].qs +  32)),
            _mm256_loadu_si256((const __m256i*) (y[i].qs +  64)),
            _mm256_loadu_si256((const __m256i*) (y[i].qs +  96)),
            _mm256_loadu_si256((const __m256i*) (y[i].qs + 128)),
            _mm256_loadu_si256((const __m256i*) (y[i].qs + 160)),
            _mm256_loadu_si256((const __m256i*) (y[i].qs + 192)),
            _mm256_loadu_si256((const __m256i*) (y[i].qs + 224)),
        };

        for (int r = 0; r < 4; ++r) {
            const block_q4_K& bx = rows[r][i];
            const float d = yd * h2f(bx.d);
            const float dmin = -yd * h2f(bx.dmin);
            uint32_t utmp[4];
            std::memcpy(utmp, bx.scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;
            const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
            const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
            acc_m[r] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[r]);

            const __m128i sc128 = _mm256_castsi256_si128(mins_and_scales);
            const __m256i scales = MM256_SET_M128I(sc128, sc128);
            __m256i sumi0 = _mm256_setzero_si256();
            __m256i sumi1 = _mm256_setzero_si256();
            const uint8_t* q4 = bx.qs;
            for (int j = 0; j < 4; ++j) {
                const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
                const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
                const __m256i q4bits = _mm256_loadu_si256((const __m256i*) (q4 + 32 * j));
                const __m256i q4l = _mm256_and_si256(q4bits, m4);
                const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                sumi0 = _mm256_add_epi32(sumi0, _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8v[2 * j + 0])));
                sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8v[2 * j + 1])));
            }
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi0, sumi1)), acc[r]);
        }
    }
    for (int r = 0; r < 4; ++r) {
        acc_m[r] = _mm_add_ps(acc_m[r], _mm_movehl_ps(acc_m[r], acc_m[r]));
        acc_m[r] = _mm_add_ss(acc_m[r], _mm_movehdup_ps(acc_m[r]));
        out[r] = hsum_float_8(acc[r]) + _mm_cvtss_f32(acc_m[r]);
    }
}

void dot_iq4_xs_q8_K_x4_avx2(int n, float out[4], const void* STRATA_RESTRICT w_base, size_t row_bytes, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;
    const uint8_t* wb = (const uint8_t*) w_base;
    const block_iq4_xs* rows[4] = {
        (const block_iq4_xs*) (wb + 0 * row_bytes),
        (const block_iq4_xs*) (wb + 1 * row_bytes),
        (const block_iq4_xs*) (wb + 2 * row_bytes),
        (const block_iq4_xs*) (wb + 3 * row_bytes),
    };
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;

    static constexpr uint8_t kAbsIq4nl[16] = {127, 104, 83, 65, 49, 35, 22, 10, 1, 13, 25, 38, 53, 69, 89, 113};
    const __m256i abs256 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kAbsIq4nl));
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m256i bit3 = _mm256_set1_epi8(0x08);

    __m256 accum[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int ibl = 0; ibl < nb; ++ibl) {
        if (ibl + 2 < nb) {
            for (int r = 0; r < 4; ++r) _mm_prefetch((const char*) &rows[r][ibl + 2], _MM_HINT_T0);
        }
        const float yd = y[ibl].d;
        const __m256i q8v[8] = {
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs +   0)),
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs +  32)),
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs +  64)),
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs +  96)),
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs + 128)),
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs + 160)),
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs + 192)),
            _mm256_loadu_si256((const __m256i*) (y[ibl].qs + 224)),
        };
        for (int r = 0; r < 4; ++r) {
            const block_iq4_xs& bx = rows[r][ibl];
            const uint8_t* qs = bx.qs;
            uint16_t sh = bx.scales_h;
            __m256i sumi1 = _mm256_setzero_si256();
            __m256i sumi2 = _mm256_setzero_si256();
            for (int ib = 0; ib < QK_K / 32; ib += 2) {
                const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*) qs);
                const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*) (qs + 16));
                qs += 32;
                const __m256i nib_1 = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bits_1, 4), q4bits_1), m4b);
                const __m256i nib_2 = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bits_2, 4), q4bits_2), m4b);
                const __m256i ax_1 = _mm256_shuffle_epi8(abs256, nib_1);
                const __m256i ax_2 = _mm256_shuffle_epi8(abs256, nib_2);
                const __m256i neg_1 = _mm256_cmpgt_epi8(bit3, nib_1);
                const __m256i neg_2 = _mm256_cmpgt_epi8(bit3, nib_2);
                const __m256i sy_1 = _mm256_sub_epi8(_mm256_xor_si256(q8v[ib + 0], neg_1), neg_1);
                const __m256i sy_2 = _mm256_sub_epi8(_mm256_xor_si256(q8v[ib + 1], neg_2), neg_2);
                const int16_t ls1 = (int16_t) (((bx.scales_l[ib / 2] & 0xf) | ((sh << 4) & 0x30)) - 32);
                const int16_t ls2 = (int16_t) (((bx.scales_l[ib / 2] >>  4) | ((sh << 2) & 0x30)) - 32);
                sh >>= 4;
                sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(_mm256_maddubs_epi16(ax_1, sy_1), _mm256_set1_epi16(ls1)));
                sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(_mm256_maddubs_epi16(ax_2, sy_2), _mm256_set1_epi16(ls2)));
            }
            accum[r] = _mm256_fmadd_ps(_mm256_set1_ps(h2f(bx.d) * yd),
                                       _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accum[r]);
        }
    }
    for (int r = 0; r < 4; ++r) out[r] = hsum_float_8(accum[r]);
}

void dot_q8_0_q8_0_x4_avx2(int n, float out[4], const void* STRATA_RESTRICT w_base, size_t row_bytes, const void* STRATA_RESTRICT vy) {
    assert(n % QK8_0 == 0);
    const int nb = n / QK8_0;
    const uint8_t* wb = (const uint8_t*) w_base;
    const block_q8_0* rows[4] = {
        (const block_q8_0*) (wb + 0 * row_bytes),
        (const block_q8_0*) (wb + 1 * row_bytes),
        (const block_q8_0*) (wb + 2 * row_bytes),
        (const block_q8_0*) (wb + 3 * row_bytes),
    };
    const block_q8_0* STRATA_RESTRICT y = (const block_q8_0*) vy;

    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};

    for (int ib = 0; ib < nb; ++ib) {
        if (ib + 4 < nb) {
            for (int r = 0; r < 4; ++r) _mm_prefetch((const char*) &rows[r][ib + 4], _MM_HINT_T0);
        }
        const float dy = h2f(y[ib].d);
        const __m256i qy = _mm256_loadu_si256((const __m256i*) y[ib].qs);
        for (int r = 0; r < 4; ++r) {
            const __m256i qx = _mm256_loadu_si256((const __m256i*) rows[r][ib].qs);
            const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(qx, qx), _mm256_sign_epi8(qy, qx));
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(h2f(rows[r][ib].d) * dy),
                                     _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, p16)), acc[r]);
        }
    }
    for (int r = 0; r < 4; ++r) out[r] = hsum_float_8(acc[r]);
}

}  // namespace avx2

DotRowsFn avx2_dot_rows(Type w) noexcept {
    using namespace avx2;
    switch (w) {
    case Type::IQ3_XXS:
        return rows_by_four<rows_iq3_xxs_q8_K<2>, rows_iq3_xxs_q8_K<3>, rows_iq3_xxs_q8_K<4>, dot_iq3_xxs_q8_K>;
    case Type::IQ2_XXS:
        return rows_by_four<rows_iq2_xxs_q8_K<2>, rows_iq2_xxs_q8_K<3>, rows_iq2_xxs_q8_K<4>, dot_iq2_xxs_q8_K>;
    case Type::IQ2_S:
        return rows_by_four<rows_iq2_s_q8_K<2>, rows_iq2_s_q8_K<3>, rows_iq2_s_q8_K<4>, dot_iq2_s_q8_K>;
    case Type::Q3_K:
        return rows_by_four<rows_q3_K_q8_K<2>, rows_q3_K_q8_K<3>, rows_q3_K_q8_K<4>, dot_q3_K_q8_K>;
    default:
        return nullptr;
    }
}

namespace {
// A type with two kernels of one ISA lists the one dot_kernel() picks first; the other stays listed so quant_bench and quant_selftest compare
// the two (Q2_0: the generic kernel, whose bits the AVX2 one gives; IQ1_S: ggml's port, the bits both give).
#define ENTRY(T, ACT, NAME)     {avx2::dot_##NAME,         Type::T, Type::ACT, Arith::Avx2, Isa::Avx2, "avx2_" #NAME}
#define ENTRY_OPT(T, ACT, NAME) {avx2::dot_##NAME##_opt,   Type::T, Type::ACT, Arith::Avx2, Isa::Avx2, "avx2_" #NAME}
#define ENTRY_X4(T, ACT, NAME)  {avx2::dot_##NAME##_x4_avx2, Type::T, Type::ACT, Arith::Avx2, Isa::Avx2, "avx2_" #NAME "_x4"}
const DotKernel kTable[] = {
    ENTRY(Q4_0, Q8_0, q4_0_q8_0),   ENTRY(Q4_1, Q8_1, q4_1_q8_1),   ENTRY(Q5_0, Q8_0, q5_0_q8_0),   ENTRY(Q5_1, Q8_1, q5_1_q8_1),
    ENTRY_OPT(Q8_0, Q8_0, q8_0_q8_0), ENTRY(IQ4_NL, Q8_0, iq4_nl_q8_0), ENTRY(MXFP4, Q8_0, mxfp4_q8_0), ENTRY(NVFP4, Q8_0, nvfp4_q8_0),
    ENTRY(Q2_K, Q8_K, q2_K_q8_K),   ENTRY(Q3_K, Q8_K, q3_K_q8_K),   ENTRY_OPT(Q4_K, Q8_K, q4_K_q8_K), ENTRY_OPT(Q5_K, Q8_K, q5_K_q8_K),
    ENTRY_OPT(Q6_K, Q8_K, q6_K_q8_K), ENTRY(IQ1_S, Q8_K, iq1_s_q8_K_vidx), ENTRY(IQ1_S, Q8_K, iq1_s_q8_K), ENTRY(IQ1_M, Q8_K, iq1_m_q8_K), ENTRY(IQ2_XXS, Q8_K, iq2_xxs_q8_K),
    ENTRY(IQ2_XS, Q8_K, iq2_xs_q8_K), ENTRY(IQ2_S, Q8_K, iq2_s_q8_K), ENTRY_OPT(IQ3_XXS, Q8_K, iq3_xxs_q8_K),
    // (PolyStrata) the own IQ3_S kernel first, so dot_kernel() takes it; ggml's port after it, for STRATA_CPU_IQ3S=0
    {avx2::dot_iq3_s_q8_K_words, Type::IQ3_S, Type::Q8_K, Arith::Avx2, Isa::Avx2, "avx2_iq3_s_q8_K_words", iq3s_own_on},
    ENTRY(IQ3_S, Q8_K, iq3_s_q8_K), ENTRY_OPT(IQ4_XS, Q8_K, iq4_xs_q8_K),
    ENTRY(Q2_0, Q8_0, q2_0_q8_0),
    {generic::dot_q2_0_q8_0_generic, Type::Q2_0, Type::Q8_0, Arith::Avx2, Isa::Scalar, "avx2_q2_0_generic"},
};
const DotX4Kernel kX4Table[] = {
    ENTRY_X4(Q8_0, Q8_0, q8_0_q8_0),
    ENTRY_X4(Q4_K, Q8_K, q4_K_q8_K),
    ENTRY_X4(IQ4_XS, Q8_K, iq4_xs_q8_K),
};
#undef ENTRY_X4
#undef ENTRY_OPT
#undef ENTRY
}  // namespace

DotTable avx2_dot_table() noexcept { return {kTable, (int) (sizeof kTable / sizeof kTable[0])}; }
DotX4Table avx2_dot_x4_table() noexcept { return {kX4Table, (int) (sizeof kX4Table / sizeof kX4Table[0])}; }

}  // namespace strata::quant
