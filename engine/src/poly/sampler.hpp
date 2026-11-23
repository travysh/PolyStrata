// src/poly/sampler.hpp - sampling for strata-poly: temperature, top-k, top-p, min-p, penalties.
#pragma once

#include "grammar.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::poly {

int most_likely(const float* logits, int n);

struct Sampling {
    float temperature = 0.f;   // 0: the most likely token
    float top_p = 1.f, min_p = 0.f;
    int top_k = 20;            // 0: no limit
    int penalty_last_n = 0;
    float penalty_repeat = 1.f, penalty_freq = 0.f, penalty_present = 0.f;
    unsigned long long seed = 0;
    std::string bias_text;
    std::vector<float> bias;
    std::string grammar_text;   // the request's grammar as the engine was sent it (base64), until it is parsed
    Grammar* grammar = nullptr; // the parsed one, where the answer has got to: owned by the request, not by this
};

// llama.cpp's sampler chain in its default order: penalties, top-k, top-p, min-p, temperature, then the draw.
class Sampler {
public:
    /// pick() is most_likely() of the logits as they are: no temperature, penalty, bias or grammar.
    static bool greedy(const Sampling& s) {
        return s.temperature <= 0.f && !penalizes(s) && s.bias.empty() && !s.grammar;
    }

    int pick(const float* logits, int n_vocab, const Sampling& s, const std::vector<int32_t>& history,
             std::mt19937_64& rng) {
        bool copied = false;
        if (!s.bias.empty() || penalizes(s)) {
            pen_.assign(logits, logits + n_vocab);
            if (!s.bias.empty()) {
                const size_t n = std::min((size_t) n_vocab, s.bias.size());
                for (size_t i = 0; i < n; ++i) pen_[i] += s.bias[i];
            }
            if (penalizes(s)) penalize(s, history);
            logits = pen_.data();
            copied = true;
        }
        const int t = choose(logits, n_vocab, s, rng);
        if (!s.grammar || s.grammar->accepts(t)) return t;
        // The pick is not what the grammar allows (llama.cpp's common sampler checks it first, too: most picks are
        // fine and the mask costs a pass over the vocabulary).  Take out what is not allowed and pick again.
        if (!copied) pen_.assign(logits, logits + n_vocab);
        if (s.grammar->mask(pen_.data(), n_vocab) == 0) return s.grammar->end_token();   // a dead end
        return choose(pen_.data(), n_vocab, s, rng);
    }

private:
    // top-k, top-p, min-p, temperature and the draw, on logits the penalties have been applied to
    int choose(const float* logits, int n_vocab, const Sampling& s, std::mt19937_64& rng) {
        if (s.temperature <= 0.f) return strata::poly::most_likely(logits, n_vocab);

        // The k best, the best first and of equals the lower id.  A token is a candidate only when it beats the
        // worst of the k so far, which is the heap's first; most of a vocabulary never does.
        auto by_logit = [](const Cand& a, const Cand& b) { return a.logit != b.logit ? a.logit > b.logit : a.id < b.id; };
        const size_t k = s.top_k > 0 ? std::min((size_t) s.top_k, (size_t) n_vocab) : (size_t) n_vocab;
        cand_.clear();
        for (int i = 0; i < n_vocab; ++i) {
            if (cand_.size() == k) {
                if (!(logits[i] > cand_.front().logit)) continue;   // an equal with a higher id is no better
                std::pop_heap(cand_.begin(), cand_.end(), by_logit);
                cand_.back() = {logits[i], i};
            } else {
                cand_.push_back({logits[i], i});
            }
            std::push_heap(cand_.begin(), cand_.end(), by_logit);
        }
        std::sort_heap(cand_.begin(), cand_.end(), by_logit);

        if (s.top_p < 1.f) {
            softmax();
            double cum = 0.0;
            size_t keep = cand_.size();
            for (size_t i = 0; i < cand_.size(); ++i) {
                cum += p_[i];
                if (cum >= s.top_p) {
                    keep = i + 1;
                    break;
                }
            }
            cand_.resize(keep);
        }
        if (s.min_p > 0.f) {
            const float floor = cand_[0].logit + std::log(s.min_p);
            size_t keep = 1;
            while (keep < cand_.size() && cand_[keep].logit >= floor) ++keep;
            cand_.resize(keep);
        }
        for (Cand& c : cand_) c.logit /= s.temperature;
        softmax();
        return cand_[std::discrete_distribution<size_t>(p_.begin(), p_.end())(rng)].id;
    }

    struct Cand {
        float logit;
        int id;
    };

    static bool penalizes(const Sampling& s) {
        return s.penalty_last_n != 0 &&
               (s.penalty_repeat != 1.f || s.penalty_freq != 0.f || s.penalty_present != 0.f);
    }

    // on pen_, the logits' copy
    void penalize(const Sampling& s, const std::vector<int32_t>& history) {
        const size_t n = s.penalty_last_n < 0 ? history.size() : std::min(history.size(), (size_t) s.penalty_last_n);
        seen_.clear();
        for (size_t i = history.size() - n; i < history.size(); ++i) ++seen_[history[i]];
        for (const auto& [id, count] : seen_) {
            float& l = pen_[(size_t) id];
            l = l <= 0.f ? l * s.penalty_repeat : l / s.penalty_repeat;
            l -= (float) count * s.penalty_freq + s.penalty_present;
        }
    }

    void softmax() {
        p_.resize(cand_.size());
        float mx = cand_[0].logit;
        for (const Cand& c : cand_) mx = std::max(mx, c.logit);
        double sum = 0.0;
        for (size_t i = 0; i < cand_.size(); ++i) sum += p_[i] = std::exp((double) (cand_[i].logit - mx));
        for (double& p : p_) p /= sum;
    }

    std::vector<Cand> cand_;
    std::vector<float> pen_;
    std::vector<double> p_;
    std::unordered_map<int32_t, int> seen_;
};

}  // namespace strata::poly
