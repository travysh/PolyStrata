// tests/quant/iq1s_avx2_test.cpp - the AVX2 IQ1_S x Q8_K dot (x86/dot_avx2.cpp) against the kernel whose bits it must give,
// on data where the float order shows: random fp16 scales on every block, activations quantized from random floats of random size, and raw
// random activation blocks.  IQ1_S against ggml's AVX2 port.  No ggml.
// Exit 0 pass, 1 fail, 77 skip (no AVX2 on this CPU, or not x86).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "strata/quant/act.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/isa.hpp"
#include "strata/quant/types.hpp"

namespace sq = strata::quant;

namespace {
uint64_t g_s = 0x9E3779B97F4A7C15ull;
uint32_t rnd() {
    g_s ^= g_s << 13;
    g_s ^= g_s >> 7;
    g_s ^= g_s << 17;
    return (uint32_t) (g_s >> 16);
}

// a finite fp16 of either sign and exponent 2^-10 .. 2^5 (normal, so no scale is zero or denormal)
uint16_t rnd_half() { return (uint16_t) (((rnd() & 1) << 15) | ((5 + rnd() % 16) << 10) | (rnd() & 0x3FF)); }

const sq::DotKernel* named(sq::Type w, sq::Arith a, const char* name) {
    const sq::DotKernel* ks[16];
    const int n = sq::dot_kernels(w, a, ks, 16);
    for (int i = 0; i < n; ++i)
        if (!std::strcmp(ks[i]->name, name)) return ks[i];
    return nullptr;
}

int g_fail = 0;

// rows of n values: kernel `k` against `ref`, row by row, memcmp of the float
void compare(const sq::DotKernel* k, const sq::DotKernel* ref, int rows) {
    const sq::TypeInfo* ti = sq::type_info(k->weight);
    const sq::TypeInfo* ai = sq::type_info(k->act);
    const sq::ActKernel* ak = sq::act_kernel(k->act, sq::Arith::Avx2);
    const int unit = ti->blck > ai->blck ? ti->blck : ai->blck;
    long diffs = 0, done = 0;
    for (int r = 0; r < rows; ++r) {
        // lengths: 1..24 units, and the expert rows of Inkling 2 Mini (2048 and 4096 values) now and then
        const int pick = (int) (rnd() % 10);
        const int n = pick == 0 ? 4096 : pick == 1 ? 2048 : unit * (1 + (int) (rnd() % 24));
        std::vector<uint8_t> w(sq::row_bytes((int) k->weight, n));
        for (auto& b : w) b = (uint8_t) rnd();
        for (size_t o = 0; o < w.size(); o += ti->bytes) {   // both types keep their fp16 scale first
            const uint16_t h = rnd_half();
            std::memcpy(&w[o], &h, 2);
        }
        std::vector<uint8_t> a(sq::row_bytes((int) k->act, n));
        if (r % 4 != 3) {
            std::vector<float> x((size_t) n);
            for (int b = 0; b < n / ai->blck; ++b) {
                const float mag = std::ldexp(1.0f, (int) (rnd() % 24) - 12);
                for (int j = 0; j < ai->blck; ++j) x[(size_t) b * ai->blck + j] = mag * ((float) ((int) (rnd() % 20001) - 10000) / 10000.0f);
            }
            ak->fn(x.data(), a.data(), n);
        } else {   // raw blocks: random bytes, a random scale, values in [-127, 127] (Q8_K: bsums the sums of its values)
            for (auto& b : a) b = (uint8_t) rnd();
            for (size_t o = 0; o < a.size(); o += ai->bytes) {
                if (k->act == sq::Type::Q8_0) {
                    const uint16_t h = rnd_half();
                    std::memcpy(&a[o], &h, 2);
                    for (int j = 0; j < 32; ++j) a[o + 2 + j] = (uint8_t) (int8_t) ((int) (rnd() % 255) - 127);
                } else {   // block_q8_K: float d, int8 qs[256], int16 bsums[16]
                    const float d = std::ldexp((float) (rnd() % 1000 + 1) / 1000.0f, (int) (rnd() % 20) - 14);
                    std::memcpy(&a[o], &d, 4);
                    int8_t* q = (int8_t*) &a[o + 4];
                    for (int j = 0; j < 256; ++j) q[j] = (int8_t) ((int) (rnd() % 255) - 127);
                    for (int g = 0; g < 16; ++g) {
                        int s = 0;
                        for (int j = 0; j < 16; ++j) s += q[g * 16 + j];
                        const int16_t s16 = (int16_t) s;
                        std::memcpy(&a[o + 4 + 256 + 2 * g], &s16, 2);
                    }
                }
            }
        }
        float got = 0.f, want = 0.f;
        k->fn(n, &got, w.data(), a.data());
        ref->fn(n, &want, w.data(), a.data());
        diffs += std::memcmp(&got, &want, sizeof got) != 0;
        ++done;
    }
    std::printf("  %-8s %-24s vs %-24s %ld rows, %ld not the same bits\n", ti->name, k->name, ref->name, done, diffs);
    if (diffs) ++g_fail;
}
}  // namespace

// each AVX2 kernel, the kernel whose bits it gives (name, contract), and the rows compared
struct Case {
    sq::Type type;
    const char* kernel;
    const char* ref;
    sq::Arith ref_arith;
    int rows;
};
const Case kCases[] = {
    {sq::Type::IQ1_S, "avx2_iq1_s_q8_K_vidx", "avx2_iq1_s_q8_K", sq::Arith::Avx2, 40000},
};

int main() {
    if (!sq::isa_ok(sq::Isa::Avx2)) {
        std::printf("iq1s_avx2_test: SKIPPED (this CPU has no AVX2, or is not x86)\n");
        return 77;
    }
    for (const Case& c : kCases) {
        const sq::DotKernel* k = named(c.type, sq::Arith::Avx2, c.kernel);
        const sq::DotKernel* ref = named(c.type, c.ref_arith, c.ref);
        if (k == nullptr || ref == nullptr) {
            std::printf("  FAIL: %s or %s is not listed\n", c.kernel, c.ref);
            ++g_fail;
            continue;
        }
        // the engine's pick (cpu_experts.cpp: dot_kernel of the default contract) is the AVX2 kernel
        if (sq::dot_kernel(c.type, sq::Arith::Avx2) != k) {
            std::printf("  FAIL: dot_kernel() does not pick %s\n", c.kernel);
            ++g_fail;
        }
        compare(k, ref, c.rows);
    }
    std::printf("iq1s_avx2_test: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
