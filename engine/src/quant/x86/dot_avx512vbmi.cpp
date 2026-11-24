// src/quant/x86/dot_avx512vbmi.cpp - the Avx2 contract on CPUs with AVX-512 VBMI as well (Zen 4 and later, Ice Lake and later):
//   IQ3_S x Q8_K with its grid words from byte lookups (vpermi2b) in place of dot_avx512.cpp's 16 scalar loads a sub-block pair.
// The integer lane sums are those of the AVX2 kernel (ggml's), and the float steps are its own (one fmadd a block into 8 lanes, then its
// hsum), so the bits are the AVX2 flavour's.
// Compiled with the AVX-512 flags of cmake/backends/quant.cmake plus VBMI; only ever called when caps() says AVX-512 F/DQ/BW/VL + VNNI + VBMI
// (Isa::Avx512Vbmi).
#include "tier_common.hpp"

#include "../generic.hpp"
#include "../registry.hpp"
#include "strata/quant/dot.hpp"

#include <cstring>

#if !defined(_MSC_VER) && (!defined(__AVX512VBMI__) || !defined(__AVX512VNNI__) || !defined(__AVX512BW__))
#error "dot_avx512vbmi.cpp needs AVX-512 BW, VNNI and VBMI"
#endif

namespace strata::quant {
namespace avx512vbmi {

#include "x86_helpers.inc"

// A block's 64 grid words come from byte lookups: byte b of the 512 words is a table of 512 bytes, four 128-byte quarters, and
// vpermi2b looks up 64 indices' low 7 bits in a quarter at once; the index's bit 7 and its 9th bit (qh) pick the quarter by masked
// blends.  The indices go in a fixed order so that unpacking the four byte planes gives the words in their order.
struct alignas(64) Iq3sPlanes {   // [byte b][quarter q][i]: byte b of iq3s_grid[128 q + i]
    uint8_t t[4][4][128];
    Iq3sPlanes() : t{} {
        for (int b = 0; b < 4; ++b)
            for (int q = 0; q < 4; ++q)
                for (int i = 0; i < 128; ++i) t[b][q][i] = (uint8_t) (iq3s_grid[128 * q + i] >> (8 * b));
    }
};

// The lookup order: unpacklo/hi_epi8 then unpacklo/hi_epi16 of the planes give, in 128-bit lane L, the words looked up at
// 16 L + 4 c + m (c the output vector, m < 4); output c is to hold words 16 c .. 16 c + 15, so position 16 L + 4 c + m looks up
// word 16 c + 4 L + m.
struct alignas(64) Iq3sOrder {
    uint8_t v[64];
    constexpr Iq3sOrder() : v{} {
        for (int L = 0; L < 4; ++L)
            for (int c = 0; c < 4; ++c)
                for (int m = 0; m < 4; ++m) v[16 * L + 4 * c + m] = (uint8_t) (16 * c + 4 * L + m);
    }
};
constexpr Iq3sOrder kIq3sOrder{};

// The signs are a mask register (the 64 sign bits of two sub-blocks as they are stored), applied as the AVX2 kernel's sign is: 0 - q8.
// maddubs and vpdpwssd by the sub-block scales take maddubs, madd and add: the 16-bit pairs fit (15 * 128 * 2), so each lane holds the
// AVX2 kernel's int32 sum, the even sub-blocks in the low half and the odd ones in the high half, added as it adds its sumi1 and sumi2.
// A Ryzen 7 7800X3D (Zen 4), one thread, an expert's gate|up rows: 446-465 us with the AVX2 kernel, 233-280 us with this one.
void dot_iq3_s_q8_K(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    // Built on the first call, not at start-up: code of this file must only run on a CPU that has VBMI (it is compiled for AVX-512,
    // and the compiler vectorizes loops), and iq3s_grid is not a constant expression for Clang, icpx or MSVC.
    static const Iq3sPlanes kIq3sPlanes{};

    const block_iq3_s* STRATA_RESTRICT x = (const block_iq3_s*) vx;
    const block_q8_K*  STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    const __m512i zero = _mm512_setzero_si512();
    const __m512i order = _mm512_load_si512((const void*) kIq3sOrder.v);
    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = h2f(x[i].d) * y[i].d;
        // the 64 indices' low bytes in lookup order, their bit 7 and their 9th bits (qh, bit 8 j + k) as masks
        const __m512i qs = _mm512_permutexvar_epi8(order, _mm512_loadu_si512((const void*) x[i].qs));
        uint64_t qh;
        std::memcpy(&qh, x[i].qh, 8);
        const __mmask64 hi = _mm512_movepi8_mask(_mm512_permutexvar_epi8(order, _mm512_movm_epi8((__mmask64) qh)));
        const __mmask64 b7 = _mm512_movepi8_mask(qs);
        __m512i P[4];
        for (int b = 0; b < 4; ++b) {
            const __m512i* t = (const __m512i*) kIq3sPlanes.t[b];
            const __m512i r0 = _mm512_permutex2var_epi8(_mm512_load_si512(t + 0), qs, _mm512_load_si512(t + 1));
            const __m512i r1 = _mm512_permutex2var_epi8(_mm512_load_si512(t + 2), qs, _mm512_load_si512(t + 3));
            const __m512i r2 = _mm512_permutex2var_epi8(_mm512_load_si512(t + 4), qs, _mm512_load_si512(t + 5));
            const __m512i r3 = _mm512_permutex2var_epi8(_mm512_load_si512(t + 6), qs, _mm512_load_si512(t + 7));
            P[b] = _mm512_mask_blend_epi8(hi, _mm512_mask_blend_epi8(b7, r0, r1), _mm512_mask_blend_epi8(b7, r2, r3));
        }
        const __m512i l01 = _mm512_unpacklo_epi8(P[0], P[1]), h01 = _mm512_unpackhi_epi8(P[0], P[1]);
        const __m512i l23 = _mm512_unpacklo_epi8(P[2], P[3]), h23 = _mm512_unpackhi_epi8(P[2], P[3]);
        const __m512i g[4] = {_mm512_unpacklo_epi16(l01, l23), _mm512_unpackhi_epi16(l01, l23),
                              _mm512_unpacklo_epi16(h01, h23), _mm512_unpackhi_epi16(h01, h23)};
        __m512i acc = zero;
        for (int p = 0; p < 4; ++p) {   // sub-blocks 2 p (low half) and 2 p + 1 (high half)
            uint64_t sg;
            std::memcpy(&sg, x[i].signs + 8 * p, 8);
            const __m512i q8 = _mm512_loadu_si512((const void*) (y[i].qs + 64 * p));
            const __m512i q8s = _mm512_mask_sub_epi8(q8, (__mmask64) sg, zero, q8);
            const uint8_t ls = x[i].scales[p];
            const __m512i sc = _mm512_mask_set1_epi16(_mm512_set1_epi16((short) (2 * (ls & 15) + 1)), 0xFFFF0000u,
                                                      (short) (2 * (ls >> 4) + 1));
            acc = _mm512_dpwssd_epi32(acc, _mm512_maddubs_epi16(g[p], q8s), sc);
        }
        const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(acc), _mm512_extracti64x4_epi64(acc, 1));
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accumf);
    }
    *s = hsum_float_8(accumf);
}

}  // namespace avx512vbmi

namespace {
const DotKernel kTable[] = {
    // (PolyStrata) STRATA_CPU_IQ3S=0 turns it off as it does the AVX2 one (iq3s_own_on): dot_avx512.cpp's kernel runs then
    {avx512vbmi::dot_iq3_s_q8_K, Type::IQ3_S, Type::Q8_K, Arith::Avx2, Isa::Avx512Vbmi, "avx512vbmi_iq3_s_q8_K", iq3s_own_on},
};
}  // namespace

DotTable avx512vbmi_dot_table() noexcept { return {kTable, (int) (sizeof kTable / sizeof kTable[0])}; }

}  // namespace strata::quant
