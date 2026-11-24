// tests/quant/quant_selftest.cpp - the layer without ggml: type table, probe, and every kernel against the reference (selfcheck.hpp);
// and the layer's own kernels (DotKernel::on set) against the ggml port of their contract, to the bit.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <vector>

#include "strata/quant/act.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/fp16.hpp"
#include "strata/quant/isa.hpp"
#include "strata/quant/selfcheck.hpp"
#include "strata/quant/types.hpp"

namespace sq = strata::quant;

static int g_fail = 0;
#define CHECK(c, ...)                                  \
    do {                                               \
        if (!(c)) {                                    \
            ++g_fail;                                  \
            std::printf("FAIL line %d: ", __LINE__);   \
            std::printf(__VA_ARGS__);                  \
            std::printf("\n");                         \
        }                                              \
    } while (0)

// Every own kernel this CPU runs against the ggml port of its contract (the last runnable kernel without `on`), to the bit: rows of
// random blocks with sane scales at the front (the formats with an own kernel: IQ3_S) at 256, 2304 (Inkling 2 Mini's gate|up rows) and
// 4096 values, times five kinds of activation: normal, tiny, one large value, every 7th large and negative (codes of +-127), and raw random
// codes; each with a -128 code planted at a set place of every 32 (the one code whose sign flip wraps).  Returns the dots compared.
static long long own_kernels_bits() {
    std::mt19937 rng(1234);
    long long all = 0;
    for (int t = 0; t < 150; ++t) {
        const sq::TypeInfo* ti = sq::type_info(t);
        if (ti == nullptr || !ti->has_dot) continue;
        const sq::Arith ar = sq::default_arith();
        const sq::DotKernel* ks[16];
        const int nk = sq::dot_kernels(ti->type, ar, ks, 16);
        const sq::DotKernel* ref = nullptr;
        for (int i = 0; i < nk; ++i)
            if (sq::isa_ok(ks[i]->isa) && ks[i]->on == nullptr) ref = ks[i];
        for (int i = 0; i < nk; ++i) {
            const sq::DotKernel* k = ks[i];
            if (k->on == nullptr || !sq::isa_ok(k->isa)) continue;
            CHECK(ref != nullptr, "%s: no ggml port of contract %s to compare with", k->name, sq::arith_name(ar));
            if (ref == nullptr) continue;
            const sq::ActKernel* ak = sq::act_kernel(ti->act, ar);
            CHECK(ak != nullptr, "%s: no activation quantizer", ti->name);
            if (ak == nullptr) continue;
            for (const int n : {256, 2304, 4096}) {
                const int rows = n == 256 ? 2000 : n == 2304 ? 4000 : 500;
                const size_t rb = sq::row_bytes(t, n), ab = sq::row_bytes((int) ti->act, n);
                std::vector<uint8_t> w((size_t) rows * rb);
                for (auto& b : w) b = (uint8_t) rng();
                std::uniform_real_distribution<float> sc(1e-4f, 3e-2f);
                for (size_t o = 0; o + ti->bytes <= w.size(); o += ti->bytes) {   // the block's fp16 scale at its front
                    const uint16_t h = sq::fp32_to_fp16(sc(rng));
                    std::memcpy(&w[o], &h, 2);
                }
                std::normal_distribution<float> nd(0.f, 1.f);
                long long differ = 0, here = 0;
                for (int kind = 0; kind < 5; ++kind) {
                    std::vector<uint8_t> a(ab);
                    if (kind < 4) {
                        std::vector<float> x((size_t) n);
                        for (float& v : x) v = nd(rng) * (kind == 1 ? 1e-4f : 1.f);
                        if (kind == 2) x[(size_t) n / 3] = 50.f;
                        if (kind == 3)
                            for (int j = 0; j < n; j += 7) x[(size_t) j] = -40.f;
                        ak->fn(x.data(), a.data(), n);
                    } else {   // raw codes: every block's bytes random, then its float scale (Q8_K: at the front)
                        for (auto& b : a) b = (uint8_t) rng();
                        const sq::TypeInfo* ai = sq::type_info(ti->act);
                        for (size_t o = 0; o + ai->bytes <= a.size(); o += ai->bytes) {
                            const float d = 0.01f;
                            std::memcpy(&a[o], &d, 4);
                        }
                    }
                    {   // a -128 code at a set place of every block (code 5 of each 32), so every row meets it under random signs
                        const sq::TypeInfo* ai = sq::type_info(ti->act);
                        const size_t qs = ti->act == sq::Type::Q8_0 ? 2 : 4;   // Q8_0: fp16 d; Q8_1: fp16 d, s; Q8_K: float d
                        for (size_t o = 0; o + ai->bytes <= a.size(); o += ai->bytes)
                            for (int j = 5; j < ai->blck; j += 32) a[o + qs + (size_t) j] = (uint8_t) 0x80;
                    }
                    for (int r = 0; r < rows; ++r, ++here) {
                        float p = 0, q = 0;
                        k->fn(n, &p, w.data() + (size_t) r * rb, a.data());
                        ref->fn(n, &q, w.data() + (size_t) r * rb, a.data());
                        differ += std::memcmp(&p, &q, sizeof p) != 0;
                    }
                }
                CHECK(differ == 0, "%s n=%d: %lld of %lld dots differ from %s's bits", k->name, n, differ, here, ref->name);
                std::printf("quant_selftest: %-24s n=%-4d %lld of %lld dots differ from %s\n", k->name, n, differ, here, ref->name);
                all += here;
            }
        }
    }
    return all;
}

int main() {
    const sq::Caps& c = sq::caps();
    std::printf("quant_selftest: best isa %s, default contract %s (avx=%d avx2=%d avx512=%d vnni=%d neon=%d)\n", sq::isa_name(sq::best_isa()),
                sq::arith_name(sq::default_arith()), c.avx, c.avx2, c.avx512, c.avx512_vnni, c.neon);

    // the table: sizes the GGUF format fixes, unknown ids answered nullptr
    CHECK(sq::row_bytes(12, 256) == 144, "q4_K row of 256 is 144 bytes");
    CHECK(sq::row_bytes(15, 512) == 584, "q8_K row of 512");
    CHECK(sq::row_bytes(8, 64) == 68, "q8_0 row of 64");
    CHECK(sq::row_bytes(42, 128) == 36, "q2_0 row of 128");
    CHECK(sq::row_bytes(12, 100) == 0, "not whole blocks");
    CHECK(sq::type_info(43) == nullptr && sq::type_info(-1) == nullptr && sq::type_info(100) == nullptr && sq::type_info(4) == nullptr, "unknown ids");
    CHECK(sq::type_info(sq::Type::IQ4_NL)->act == sq::Type::Q8_0, "iq4_nl takes q8_0");
    CHECK(sq::type_info(sq::Type::Q5_1)->act == sq::Type::Q8_1, "q5_1 takes q8_1");
    CHECK(sq::type_info(sq::Type::IQ2_S)->act == sq::Type::Q8_K, "iq2_s takes q8_K");

    // every dot kernel the layer has, that this CPU can run, against the Generic reference on exact data
    int dots = 0, acts = 0;
    for (int t = 0; t < 150; ++t) {
        const sq::TypeInfo* ti = sq::type_info(t);
        if (ti == nullptr || !ti->has_dot) continue;
        for (sq::Arith a : {sq::Arith::Generic, sq::Arith::Avx, sq::Arith::Avx2, sq::Arith::Neon}) {
            const sq::DotKernel* ks[16];
            const int n = sq::dot_kernels(ti->type, a, ks, 16);
            for (int i = 0; i < n; ++i) {
                if (!sq::isa_ok(ks[i]->isa)) continue;
                std::string why;
                const bool ok = sq::selfcheck(*ks[i], &why);
                CHECK(ok, "%s", why.c_str());
                ++dots;
            }
        }
        CHECK(sq::dot_kernel(ti->type, sq::Arith::Generic) != nullptr, "%s has no Generic dot", ti->name);
        CHECK(sq::checked_dot_kernel(ti->type, sq::Arith::Generic) != nullptr, "%s has no checked Generic kernel", ti->name);
        CHECK(sq::checked_dot_kernel(ti->type, sq::default_arith()) != nullptr, "%s has no checked kernel of the default contract", ti->name);
    }
    for (sq::Type t : {sq::Type::Q8_0, sq::Type::Q8_1, sq::Type::Q8_K})
        for (sq::Arith a : {sq::Arith::Generic, sq::Arith::Avx, sq::Arith::Avx2, sq::Arith::Neon}) {
            const sq::ActKernel* k = sq::act_kernel(t, a);
            if (k == nullptr || k->arith != a) continue;
            std::string why;
            CHECK(sq::selfcheck(*k, &why), "%s", why.c_str());
            ++acts;
        }
    for (sq::Type t : {sq::Type::Q8_0, sq::Type::Q8_1, sq::Type::Q8_K}) { CHECK(sq::checked_act_kernel(t, sq::Arith::Generic) != nullptr, "no checked Generic activation kernel"); CHECK(sq::checked_act_kernel(t, sq::default_arith()) != nullptr, "no checked activation kernel of the default contract"); }

    // the own kernels: the ggml port's bits; and STRATA_CPU_IQ3S=0 gives the port back
    const long long own = own_kernels_bits();
    {
        const sq::DotKernel* k = sq::dot_kernel(sq::Type::IQ3_S, sq::default_arith());
        const sq::DotKernel* kc = sq::checked_dot_kernel(sq::Type::IQ3_S, sq::default_arith());
        const char* e = std::getenv("STRATA_CPU_IQ3S");
        const bool off = e != nullptr && e[0] == '0';
        CHECK(k != nullptr && kc == k, "iq3_s: dot_kernel and checked_dot_kernel disagree");
        if (k != nullptr) {
            CHECK(!off || k->on == nullptr, "STRATA_CPU_IQ3S=0 still gives the own kernel %s", k->name);
            std::printf("quant_selftest: iq3_s takes %s (STRATA_CPU_IQ3S %s)\n", k->name, e ? e : "unset");
        }
    }

    // the Q4_0 requantizer: a ramp of 32 values 0..31 -> d = max/-8 with max = 31 -> d = -3.875, ids from the ggml rule
    float ramp[32];
    for (int i = 0; i < 32; ++i) ramp[i] = (float) i;
    unsigned char blk[18];
    sq::quantize_q4_0_ref(ramp, blk, 32);
    CHECK(blk[0] == 0xC0 && blk[1] == 0xC3, "q4_0 scale of the ramp is fp16 -3.875 (0xC3C0), got %02x%02x", blk[1], blk[0]);
    CHECK(blk[2] == 0x48, "q4_0 first byte of the ramp is 0x48 (x=0 -> 8, x=16 -> 4), got %02x", blk[2]);

    std::printf("quant_selftest: %d dot kernels, %d activation kernels checked, %lld own-kernel dots against ggml's port, %d failures -> %s\n", dots, acts,
                own, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
