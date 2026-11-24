// No GPU or model required: split_plan.hpp's "auto" for Inkling 2 on two cards, on made-up cards and the layer sizes
// of Inkling 2 Mini IQ4_NL (24 layers, the loop 6 to 17 three times; experts 0.534 GiB a layer for 0-19 and 1.009 for
// 20-23, dense weights about 42.5 MB a layer, the output 0.49 GB; 8 of 128 experts a run).
#include "split_plan.hpp"

#include <cstdio>

using namespace strata::poly;

namespace {

constexpr double kGiB = double(1ull << 30), kGB = 1e9;

std::vector<SplitLayer> mini(double scale = 1.0) {
    std::vector<SplitLayer> v(24);
    for (int il = 0; il < 24; ++il) {
        v[(size_t) il].dense = 42.5e6;
        v[(size_t) il].experts = scale * (il < 20 ? 0.534 : 1.009) * kGiB;
        v[(size_t) il].runs = il >= 6 && il < 18 ? 3 : 1;
    }
    return v;
}

constexpr double kHead = 0.49 * kGB, kUsed = 8.0 / 128.0;

// free VRAM less the engine's 1.5 GiB beside its layers
SplitCard card(double vram_gb, double gbs) { return SplitCard{vram_gb * kGB - 0.6 * kGB - 1.5 * kGiB, gbs}; }

// the card that runs the loop: 0 or 1, the devices' numbers
int loop_card(const SplitPlan& p) {
    const bool first = p.cut >= 18;   // the first card has the loop
    return first != p.swap ? 0 : 1;
}

}  // namespace

int main() {
    int bad = 0;
    const auto check = [&](bool ok, const char* why) {
        if (!ok) {
            std::fprintf(stderr, "FAIL %s\n", why);
            ++bad;
        }
    };
    const auto layers = mini();

    // equal cards: the plain layout and order, whatever the plain cut
    for (int plain : {6, 18, 3, 20}) {
        const SplitCard same[2] = {card(8, 448), card(8, 448)};
        const SplitPlan p = loop_split(layers, kHead, same, 6, 18, plain, kUsed);
        check(p.cut == plain && !p.swap, "two RTX 3070: the plain layout");
        const SplitCard close[2] = {card(8, 448), card(8, 448 * 1.04)};
        const SplitPlan q = loop_split(layers, kHead, close, 6, 18, plain, kUsed);
        check(q.cut == plain && !q.swap, "within 5%: the plain layout");
        const SplitCard unknown[2] = {card(16, 960), card(16, 0)};
        const SplitPlan u = loop_split(layers, kHead, unknown, 6, 18, plain, kUsed);
        check(u.cut == plain && !u.swap, "a card's bandwidth not known: the plain layout");
    }

    // RTX 5080 (960 GB/s) + RTX 4060 Ti 16 GB (288 GB/s): the loop on the 5080, in either device order
    {
        const SplitCard fast_first[2] = {card(16, 960), card(16, 288)};
        const SplitPlan p = loop_split(layers, kHead, fast_first, 6, 18, 18, kUsed);
        check(loop_card(p) == 0, "5080 + 4060 Ti: the loop on the 5080 (device 0)");
        const SplitCard fast_last[2] = {card(16, 288), card(16, 960)};
        const SplitPlan q = loop_split(layers, kHead, fast_last, 6, 18, 18, kUsed);
        check(loop_card(q) == 1, "4060 Ti + 5080: the loop on the 5080 (device 1)");
        check(q.ms < kClearlyFaster * split_ms(layers, kHead, fast_last, 18, false, kUsed),
              "4060 Ti + 5080: faster than the plain split at 18 (the loop on the 4060 Ti)");
        std::printf("5080 first: cut %d%s, %.2f ms; 5080 last: cut %d%s, %.2f ms (plain %.2f)\n", p.cut,
                    p.swap ? " swapped" : "", p.ms, q.cut, q.swap ? " swapped" : "", q.ms,
                    split_ms(layers, kHead, fast_last, 18, false, kUsed));
    }

    // a 16 GB fast card and an 8 GB slow one (RTX 4070 Ti SUPER 672 GB/s + RTX 3060 Ti 448... and a slower 8 GB)
    for (double slow : {448.0, 272.0}) {
        for (int order = 0; order < 2; ++order) {
            const SplitCard a = card(16, 672), b = card(8, slow);
            const SplitCard two[2] = {order ? b : a, order ? a : b};
            const SplitPlan p = loop_split(layers, kHead, two, 6, 18, 18, kUsed);
            check(loop_card(p) == (order ? 1 : 0), "16 GB fast + 8 GB slow: the loop on the fast card");
            std::printf("16 GB 672 + 8 GB %.0f (%s first): cut %d%s, %.2f ms\n", slow, order ? "slow" : "fast", p.cut,
                        p.swap ? " swapped" : "", p.ms);
        }
    }

    // UD-Q2_K_XL's size (about half the experts' bytes): fits two 8 GB cards; unequal ones still take the faster
    {
        const auto q2 = mini(0.36);
        const SplitCard two[2] = {card(8, 272), card(8, 448)};
        const SplitPlan p = loop_split(q2, kHead, two, 6, 18, 18, kUsed);
        check(loop_card(p) == 1, "Q2 on 8 GB 272 + 8 GB 448: the loop on the faster card");
    }

    // every choice is a legal start: outside the loop but at its first layer, the second card two layers at least
    for (double g0 : {150.0, 288.0, 448.0, 960.0, 1800.0})
        for (double g1 : {150.0, 288.0, 448.0, 960.0, 1800.0})
            for (double v0 : {6.0, 8.0, 12.0, 16.0, 24.0, 32.0})
                for (double v1 : {6.0, 8.0, 12.0, 16.0, 24.0, 32.0})
                    for (int plain : {6, 18}) {
                        const SplitCard two[2] = {card(v0, g0), card(v1, g1)};
                        const SplitPlan p = loop_split(layers, kHead, two, 6, 18, plain, kUsed);
                        check(p.cut >= 1 && p.cut <= 22 && !(p.cut > 6 && p.cut < 18), "a legal start");
                        const double plain_ms = split_ms(layers, kHead, two, plain, false, kUsed);
                        if (p.cut != plain || p.swap)
                            check(p.ms >= 0 && (plain_ms < 0 || p.ms < kClearlyFaster * plain_ms),
                                  "a change only where it is clearly faster");
                        if (g0 == g1) check(p.cut == plain && !p.swap, "equal bandwidth: the plain layout");
                    }

    // a card that cannot hold its layers' dense weights is never given them
    {
        const SplitCard two[2] = {SplitCard{0.3 * kGB, 960}, card(24, 448)};
        const SplitPlan p = loop_split(layers, kHead, two, 6, 18, 18, kUsed);
        check(p.ms >= 0, "a 0.3 GB card: a layout that fits");
        check(split_ms(layers, kHead, two, 18, false, kUsed) < 0, "0.3 GB holds no 18 layers' dense weights");
    }

    // the estimate itself: more VRAM never slower, the loop's experts cached first
    {
        const SplitCard small[2] = {card(8, 448), card(8, 448)}, big[2] = {card(12, 448), card(8, 448)};
        check(split_ms(layers, kHead, big, 18, false, kUsed) <= split_ms(layers, kHead, small, 18, false, kUsed),
              "more VRAM, not slower");
        SplitCard just{0, 448};
        for (int il = 0; il < 18; ++il) just.room += layers[(size_t) il].dense;
        for (int il = 6; il < 18; ++il) just.room += layers[(size_t) il].experts;
        // room for the loop's experts and nothing else: what stays on the CPU is layers 0-5's, a run each
        const double ms = split_card_ms(layers, 0, 18, 0, just, kUsed);
        double want_gpu = 0, want_cpu = 0;
        for (int il = 0; il < 18; ++il) {
            want_gpu += layers[(size_t) il].runs * layers[(size_t) il].dense;
            (il >= 6 ? want_gpu : want_cpu) += layers[(size_t) il].runs * layers[(size_t) il].experts * kUsed;
        }
        const double want = want_gpu / (448 * 1e6) + want_cpu / (kCpuGbs * 1e6);
        check(ms > 0.999 * want && ms < 1.001 * want, "the loop's experts cached first");
    }

    std::printf("%s\n", bad ? "split_plan_test: FAILED" : "split_plan_test: ok");
    return bad ? 1 : 0;
}
