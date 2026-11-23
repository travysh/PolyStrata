// src/poly/sampler_test.cpp - CPU-only tests for strata-poly's sampler.
//
//   clang++ -std=c++20 -Iinclude -Isrc/poly src/poly/sampler_test.cpp src/poly/grammar.cpp -o sampler_test && ./sampler_test
#include "sampler.hpp"
#include "strata/core/logit_bias.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace strata::poly {
int most_likely(const float* logits, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}
}

namespace {

int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

using strata::poly::Sampler;
using strata::poly::Sampling;

void test_greedy() {
    Sampler sampler;
    Sampling s;
    s.temperature = 0.f;
    std::mt19937_64 rng(42);
    std::vector<int32_t> history;

    std::vector<float> logits = {0.1f, 2.5f, 1.0f, -0.5f, 0.8f};
    int pick = sampler.pick(logits.data(), (int) logits.size(), s, history, rng);
    check(pick == 1, "greedy pick chooses largest logit");

    logits[3] = 10.0f;
    pick = sampler.pick(logits.data(), (int) logits.size(), s, history, rng);
    check(pick == 3, "greedy pick chooses new largest logit");
}

void test_top_k() {
    Sampler sampler;
    Sampling s;
    s.temperature = 1.0f;
    s.top_k = 2;   // only tokens 1 and 2 should ever be picked
    std::mt19937_64 rng(12345);
    std::vector<int32_t> history;

    // Tokens: 0: 0.0, 1: 5.0, 2: 4.9, 3: 1.0, 4: -2.0
    std::vector<float> logits = {0.0f, 5.0f, 4.9f, 1.0f, -2.0f};

    for (int i = 0; i < 100; ++i) {
        int pick = sampler.pick(logits.data(), (int) logits.size(), s, history, rng);
        check(pick == 1 || pick == 2, "top-k=2 restricts choices to top 2 tokens");
    }
}

void test_penalties() {
    Sampler sampler;
    Sampling s;
    s.temperature = 0.f;   // greedy
    s.penalty_last_n = 5;
    s.penalty_repeat = 2.0f;   // positive logit gets divided by 2.0
    std::mt19937_64 rng(42);

    // Initial logits: token 1 has 4.0, token 2 has 3.0
    std::vector<float> logits = {0.0f, 4.0f, 3.0f, 0.0f};
    std::vector<int32_t> history = {1};

    // With repeat penalty on token 1, logit 4.0 becomes 4.0 / 2.0 = 2.0 < 3.0 (token 2)
    int pick = sampler.pick(logits.data(), (int) logits.size(), s, history, rng);
    check(pick == 2, "penalty_repeat penalizes recent token 1 so token 2 is chosen");

    // Frequency penalty
    s.penalty_repeat = 1.0f;
    s.penalty_freq = 2.5f;
    // token 1 appears twice in history -> logit drops by 2 * 2.5 = 5.0 -> 4.0 - 5.0 = -1.0
    history = {1, 1};
    pick = sampler.pick(logits.data(), (int) logits.size(), s, history, rng);
    check(pick == 2, "penalty_freq penalizes repeated token 1 so token 2 is chosen");

    // Presence penalty
    s.penalty_freq = 0.f;
    s.penalty_present = 2.0f;
    // token 1 appears in history -> logit drops by 2.0 -> 4.0 - 2.0 = 2.0 < 3.0
    history = {1};
    pick = sampler.pick(logits.data(), (int) logits.size(), s, history, rng);
    check(pick == 2, "penalty_present penalizes present token 1 so token 2 is chosen");
}

void test_biases() {
    Sampler sampler;
    std::mt19937_64 rng(20261019);
    std::vector<int32_t> history;

    // 1. -100 excluding the greedy token
    {
        Sampling s;
        s.temperature = 0.f;   // greedy
        std::vector<float> logits = {1.0f, 5.0f, 4.0f, 0.5f};
        std::string err;
        bool ok = strata::core::parse_logit_bias("1:-100", 4, s.bias, err);
        check(ok, "parse -100 bias");
        int pick = sampler.pick(logits.data(), 4, s, history, rng);
        check(pick == 2, "-100 excludes token 1, greedy picks token 2");
    }

    // 2. +100 forcing a token
    {
        Sampling s;
        s.temperature = 0.f;   // greedy
        std::vector<float> logits = {1.0f, 5.0f, 4.0f, 0.5f};
        std::string err;
        bool ok = strata::core::parse_logit_bias("3:100", 4, s.bias, err);
        check(ok, "parse +100 bias");
        int pick = sampler.pick(logits.data(), 4, s, history, rng);
        check(pick == 3, "+100 forces token 3");
    }

    // 3. Bias plus penalties (sharing logits copy)
    {
        Sampling s;
        s.temperature = 0.f;   // greedy
        s.penalty_last_n = 5;
        s.penalty_repeat = 2.0f;
        // Token 1 has 10.0, Token 2 has 8.0
        // Bias on token 1: -3.0 -> 7.0
        // Penalty on token 2 (in history): 8.0 / 2.0 -> 4.0
        // Result: Token 1 (7.0) beats Token 2 (4.0)
        std::vector<float> logits = {0.0f, 10.0f, 8.0f, 0.0f};
        history = {2};
        std::string err;
        bool ok = strata::core::parse_logit_bias("1:-3", 4, s.bias, err);
        check(ok, "parse -3 bias");
        int pick = sampler.pick(logits.data(), 4, s, history, rng);
        check(pick == 1, "bias on token 1 and penalty on token 2 properly combined");
    }

    // 4. Bias with top-k sampling never drawing a -100 token over 1,000 seeded draws
    {
        Sampling s;
        s.temperature = 1.0f;
        s.top_k = 4;
        std::vector<float> logits = {2.0f, 10.0f, 3.0f, 1.0f};   // token 1 has highest natural logit
        std::string err;
        bool ok = strata::core::parse_logit_bias("1:-100", 4, s.bias, err);
        check(ok, "parse -100 bias for top-k test");
        history.clear();
        for (int i = 0; i < 1000; ++i) {
            int pick = sampler.pick(logits.data(), 4, s, history, rng);
            if (pick == 1) {
                check(false, "drew excluded -100 token in top-k sampling");
            }
        }
        check(true, "never drew -100 token over 1000 draws");
    }

    // 5. Pick without a bias after one with a bias giving the unbiased answer
    {
        Sampling s_biased;
        s_biased.temperature = 0.f;
        std::string err;
        strata::core::parse_logit_bias("0:100", 4, s_biased.bias, err);
        std::vector<float> logits = {1.0f, 5.0f, 4.0f, 0.5f};
        history.clear();
        int pick_biased = sampler.pick(logits.data(), 4, s_biased, history, rng);
        check(pick_biased == 0, "biased pick gave token 0");

        Sampling s_unbiased;
        s_unbiased.temperature = 0.f;   // no bias
        int pick_unbiased = sampler.pick(logits.data(), 4, s_unbiased, history, rng);
        check(pick_unbiased == 1, "unbiased pick after biased pick gave original greedy token 1");
    }
}


// Tokens 0 "a", 1 "b", 2 "x", 3 "<eos>"
llama_vocab grammar_vocab() {
    llama_vocab v;
    v.pieces = {"a", "b", "x", ""};
    v.eog = {3};
    return v;
}

std::unique_ptr<strata::poly::Grammar> grammar_of(const llama_vocab& v, const char* text) {
    std::string err;
    auto g = strata::poly::Grammar::parse(text, v, err);
    check(g != nullptr, "the test grammar parses");
    return g;
}

void test_grammar() {
    const llama_vocab v = grammar_vocab();
    std::vector<int32_t> history;
    std::mt19937_64 rng(7);

    {   // a pick the grammar accepts is the pick: no mask
        Sampler sampler;
        auto g = grammar_of(v, "root ::= \"ab\" | \"x\"");
        Sampling s;
        s.grammar = g.get();
        std::vector<float> logits = {5.f, 4.f, 3.f, 2.f};
        check(sampler.pick(logits.data(), 4, s, history, rng) == 0, "an allowed pick stands");
        Sampling plain;
        check(sampler.pick(logits.data(), 4, plain, history, rng) == 0, "and is what no grammar picks");
    }
    {   // a miss is masked and picked again
        Sampler sampler;
        auto g = grammar_of(v, "root ::= \"ab\" | \"x\"");
        Sampling s;
        s.grammar = g.get();
        std::vector<float> logits = {1.f, 5.f, 2.f, 3.f};   // b first, then the end, neither allowed yet
        check(sampler.pick(logits.data(), 4, s, history, rng) == 2, "b and the end are masked: x is next best");
        check(logits[1] == 5.f && logits[3] == 3.f, "the caller's logits are not changed");
    }
    {   // greedy gives what masking first gives, in every state
        Sampler sampler;
        std::mt19937_64 gen(99);
        std::uniform_real_distribution<float> u(-5.f, 5.f);
        for (int round = 0; round < 3; ++round) {
            auto g = grammar_of(v, "root ::= \"a\" (\"b\" | \"x\")+");
            Sampling s;
            s.grammar = g.get();
            for (int step = 0; step < 4; ++step) {
                for (int k = 0; k < 40; ++k) {
                    std::vector<float> logits = {u(gen), u(gen), u(gen), u(gen)};
                    std::vector<float> masked = logits;
                    check(g->mask(masked.data(), 4) > 0, "the grammar leaves something");
                    const int want = strata::poly::most_likely(masked.data(), 4);
                    check(sampler.pick(logits.data(), 4, s, history, rng) == want, "greedy == mask first");
                }
                static const int walk[4][4] = {{0, 1, 2, 1}, {0, 2, 2, 1}, {0, 1, 1, 2}, {0, 2, 1, 1}};
                g->accept(walk[round][step]);
            }
        }
    }
    {   // the end only when the grammar is complete, and then only that
        Sampler sampler;
        auto g = grammar_of(v, "root ::= \"x\"");
        Sampling s;
        s.grammar = g.get();
        std::vector<float> logits = {1.f, 1.f, 2.f, 9.f};
        check(sampler.pick(logits.data(), 4, s, history, rng) == 2, "the end is refused while the grammar is open");
        g->accept(2);
        logits = {9.f, 8.f, 7.f, 0.f};
        check(sampler.pick(logits.data(), 4, s, history, rng) == 3, "complete and closed: only the end is left");
    }
    {   // sampled answers stay inside the grammar
        Sampler sampler;
        auto g = grammar_of(v, "root ::= [ab]+");
        Sampling s;
        s.grammar = g.get();
        s.temperature = 1.5f;
        s.top_k = 0;
        std::vector<float> logits = {0.f, 0.f, 3.f, 2.f};   // x and the end are favoured and not allowed
        for (int i = 0; i < 200; ++i) {
            const int t = sampler.pick(logits.data(), 4, s, history, rng);
            check(t == 0 || t == 1, "a sampled pick is a or b");
        }
    }
    {   // nothing allowed at all: the end token, so that the answer stops
        llama_vocab only_x = v;
        only_x.pieces = {"y", "y", "y", ""};
        Sampler sampler;
        auto g = grammar_of(only_x, "root ::= \"x\"");
        Sampling s;
        s.grammar = g.get();
        std::vector<float> logits = {1.f, 2.f, 3.f, 0.f};
        check(sampler.pick(logits.data(), 4, s, history, rng) == 3, "a dead end gives the end token");
    }
}

}  // namespace

int main() {
    test_greedy();
    test_top_k();
    test_penalties();
    test_biases();
    test_grammar();
    std::printf("sampler_test: %d checks passed\n", checks);
    return 0;
}
