// src/poly/split_plan.hpp - "auto" for a layer split of Inkling 2 on two cards of different speeds.
//
// A token runs Inkling 2's loop (layers loop.start to loop.end - 1) loop.count times, so in Inkling 2 Mini 36 of a
// token's 48 layer runs are the loop's, and the loop runs on one card (Engine::open()).  A card starts at a layer
// outside the loop: two layouts are worth having, the loop with the layers before it on one card, or the loop with
// the layers after it on the other.  The plain "auto" shares the layers' bytes out by free VRAM and so puts the loop
// where the bytes fall, whatever the cards' speeds.  On two cards of different memory bandwidth this one estimates a
// token's time for every place to start the second card and both card orders, and takes the fastest when it is
// clearly faster than the plain layout.  Cards within 5% of each other's bandwidth (the server's rule for equal
// cards, #1760) keep the plain layout and order.
//
// The idea of putting the loop on the faster card is cobra91's and kuraneko1's (Strata PR #6506, their two-card
// Inkling 2 layout); this is our own code for a plain layer split.
//
// The estimate: a card reads each of its layers' dense weights once a run, the share of a layer's experts a run
// routes to (expert_used_count of expert_count) once a run, and the last card the output once a token; a card's
// expert cache holds the experts of its layers that run the most first (the loop's), and the experts it has no room
// for are read on the CPU at kCpuGbs.  The cards run one after the other, so a token's time is the sum.  Pure
// arithmetic: split_plan_test.cpp tests it without a card.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace strata::poly {

struct SplitCard {
    double room = 0;   ///< bytes of VRAM for layers and the output: free less what an engine takes beside them
    double gbs = 0;    ///< memory bandwidth, GB/s; 0: not known
};

struct SplitLayer {
    double dense = 0;     ///< bytes of everything but the routed experts
    double experts = 0;   ///< bytes of the routed experts
    int runs = 1;         ///< runs a token
};

struct SplitPlan {
    int cut = 0;         ///< the second card's first layer
    bool swap = false;   ///< the second device takes the first layers
    double ms = 0;       ///< the estimate of a token, ms; < 0: a card cannot hold its dense weights
};

/// The CPU's reads of the experts that are not in VRAM, GB/s: a rough figure for two channels of DDR4/DDR5.
constexpr double kCpuGbs = 20.0;
/// The bandwidths of two cards this close are one card's (the server's #1760 rule).
constexpr double kSameGbs = 1.05;
/// A layout replaces the plain one where it is estimated at least this much faster.
constexpr double kClearlyFaster = 0.97;

/// The estimate for layers [lo, hi) on a card, with the output (head bytes) when it is the last.
inline double split_card_ms(const std::vector<SplitLayer>& layers, int lo, int hi, double head, const SplitCard& c,
                            double used) {
    double dense = head, room = c.room - head;
    for (int il = lo; il < hi; ++il) room -= layers[(size_t) il].dense;
    if (room < 0 || c.gbs <= 0) return -1;
    std::vector<int> order;
    for (int il = lo; il < hi; ++il) order.push_back(il);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return layers[(size_t) a].runs > layers[(size_t) b].runs; });
    double gpu = 0, cpu = 0;
    for (int il : order) {
        const SplitLayer& L = layers[(size_t) il];
        const double held = std::min(L.experts, std::max(room, 0.0));
        room -= held;
        dense += L.runs * L.dense;
        gpu += L.runs * held * used;
        cpu += L.runs * (L.experts - held) * used;
    }
    return (dense + gpu) / (c.gbs * 1e6) + cpu / (kCpuGbs * 1e6);
}

/// A token's estimate for the second card starting at `cut`, the devices in their order or swapped.
inline double split_ms(const std::vector<SplitLayer>& layers, double head, const SplitCard cards[2], int cut, bool swap,
                       double used) {
    const SplitCard& a = cards[swap ? 1 : 0];
    const SplitCard& b = cards[swap ? 0 : 1];
    const double ma = split_card_ms(layers, 0, cut, 0, a, used);
    const double mb = split_card_ms(layers, cut, (int) layers.size(), head, b, used);
    return ma < 0 || mb < 0 ? -1 : ma + mb;
}

/// The layout for two cards: `plain` (the plain "auto"'s cut, in the devices' order) unless the cards' bandwidths
/// differ and another cut or order is clearly faster.  No card starts inside [keep_lo, keep_hi) but at keep_lo, and
/// the second card keeps two layers at least.
inline SplitPlan loop_split(const std::vector<SplitLayer>& layers, double head, const SplitCard cards[2], int keep_lo,
                            int keep_hi, int plain, double used) {
    SplitPlan out{plain, false, split_ms(layers, head, cards, plain, false, used)};
    const double lo = std::min(cards[0].gbs, cards[1].gbs), hi = std::max(cards[0].gbs, cards[1].gbs);
    if (lo <= 0 || hi < kSameGbs * lo) return out;
    SplitPlan best = out;
    const int n = (int) layers.size();
    for (int swap = 0; swap < 2; ++swap)
        for (int cut = 1; cut <= n - 2; ++cut) {
            if (cut > keep_lo && cut < keep_hi) continue;
            const double ms = split_ms(layers, head, cards, cut, swap != 0, used);
            if (ms >= 0 && (best.ms < 0 || ms < best.ms)) best = SplitPlan{cut, swap != 0, ms};
        }
    if (out.ms < 0 || (best.ms >= 0 && best.ms < kClearlyFaster * out.ms)) return best;
    return out;
}

}  // namespace strata::poly
