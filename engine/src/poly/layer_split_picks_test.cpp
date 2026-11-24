// src/poly/layer_split_picks_test.cpp - Does a layer split write one card's greedy tokens on both of Session's paths?
// Greedy decoding with nothing that reads the logits (no --top-probs) asks the card for the most likely token
// (eval_picks(), picked()); with --top-probs the host searches the logits (logits()).  Each run opens the engine,
// reads the same prompt and writes TOKENS tokens a token at a time, one card or split, on one of the two paths.  All
// four runs must write the same ids.
//
//   poly_layer_split_picks_test MODEL.gguf [LAYER_SPLIT] [TOKENS]   (LAYER_SPLIT: as --layer-split, default "auto";
//                                                                   TOKENS: 2 to 4096, default 32)
//
// Every routed expert stays on the CPU (--no-gpu-experts), so the split's expert cache does not change its sums.  With
// STRATA_POLY_CARDS unset the split's parts run on the devices there are in turn (STRATA_POLY_CARDS = the parts), so
// one card is enough.  Needs the model file and a card, so it is run by hand.  Exit 0: the same ids; 1: they differ;
// 2: an error or a bad argument.
#include "engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using strata::poly::Engine;
using strata::poly::Options;

namespace {

int32_t argmax(const float* v, int n) {   // the lowest id of the largest score, as most_likely() picks it
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

std::vector<int32_t> write(const Options& o, const std::vector<int32_t>& prompt, int tokens, bool on_card) {
    auto e = Engine::open(o);
    const int n = (int) prompt.size();
    if (!e->read(prompt.data(), n - 1)) throw std::runtime_error("the read was given up");
    e->end_read();
    e->eval(prompt.data() + n - 1, 1, true);
    std::vector<int32_t> ids{argmax(e->logits(), e->n_vocab())};
    while ((int) ids.size() < tokens) {
        int32_t t = ids.back();
        if (on_card) e->eval_picks();
        e->eval(&t, 1, true);
        ids.push_back(on_card ? e->picked(0) : argmax(e->logits(), e->n_vocab()));
    }
    return ids;
}

std::string show(const std::vector<int32_t>& ids) {
    std::string s;
    for (size_t i = 0; i < ids.size(); ++i) s += (i ? "," : "") + std::to_string(ids[i]);
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [LAYER_SPLIT] [TOKENS]\n", argv[0]);
        return 2;
    }
    const std::string split = argc > 2 ? argv[2] : "auto";
    int tokens = 32;
    if (argc > 3) {
        char* end = nullptr;
        const long v = std::strtol(argv[3], &end, 10);
        if (end == argv[3] || *end != '\0' || v < 2 || v > 4096) {
            std::fprintf(stderr, "TOKENS is a whole number from 2 to 4096, not '%s'\n", argv[3]);
            return 2;
        }
        tokens = (int) v;
    }
    if (split.empty()) {
        std::fprintf(stderr, "LAYER_SPLIT is a split (\"auto\" or the later parts' first layers)\n");
        return 2;
    }
    if (!std::getenv("STRATA_POLY_CARDS")) {   // the parts on the devices there are, in turn
        const std::string parts = split == "auto" ? "2" : std::to_string(1 + std::count(split.begin(), split.end(), ','));
#ifdef _WIN32
        _putenv_s("STRATA_POLY_CARDS", parts == "1" ? "2" : parts.c_str());
#else
        setenv("STRATA_POLY_CARDS", parts == "1" ? "2" : parts.c_str(), 1);
#endif
    }
    Options o;
    o.model = argv[1];
    o.max_context = 4096;
    o.gpu_experts = false;
    std::vector<int32_t> prompt(64);
    for (int i = 0; i < (int) prompt.size(); ++i) prompt[(size_t) i] = 1000 + (53 * i + i / 9) % 100000;
    try {
        const std::vector<int32_t> ref = write(o, prompt, tokens, false);
        std::fprintf(stderr, "one card, logits on the host: %s\n", show(ref).c_str());
        int differ = 0;
        for (int k = 0; k < 3; ++k) {
            Options w = o;
            if (k > 0) w.layer_split = split;
            const bool on_card = k != 1;
            const std::vector<int32_t> ids = write(w, prompt, tokens, on_card);
            const bool same = ids == ref;
            const std::string where = k ? "split '" + split + "'" : "one card";
            std::fprintf(stderr, "%s, %s: %s%s\n", where.c_str(), on_card ? "picked on the card" : "logits on the host",
                         same ? "same" : "DIFFER: ", same ? "" : show(ids).c_str());
            differ += !same;
        }
        std::fprintf(stderr, "%s: %d of 3 runs differ from one card's (split '%s', STRATA_POLY_CARDS=%s, %d tokens)\n",
                     differ ? "FAIL" : "PASS", differ, split.c_str(), std::getenv("STRATA_POLY_CARDS"), tokens);
        return differ ? 1 : 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "ERROR: %s\n", ex.what());
        return 2;
    }
}
