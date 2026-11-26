// src/poly/main.cpp - strata-poly: GLM 5.3 Flash from its GGUF.
//
//   strata-poly --model M.gguf --tokens 1,2,3 [--max-new N]   one prompt; prints the generated ids
//   strata-poly --model M.gguf --serve                        the line protocol serve/server.py speaks
//
// The protocol is the one `strata --serve` speaks (src/program/generate.cpp): GEN, GENI for a prompt with pictures,
// STOP and QUIT in, and RESUME, PP, T, DONE and ERR out.
#include "conversations.hpp"
#include "engine.hpp"
#include "guesses.hpp"
#include "sampler.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/core/logit_bias.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/suffix_drafter.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#undef far   // a word of this file's
#endif

#ifndef STRATA_VERSION
#define STRATA_VERSION "dev"
#endif

namespace {

using strata::poly::CacheOptions;
using strata::poly::Conversations;
using strata::poly::Engine;
using strata::poly::Guesses;
using strata::poly::KvFmt;
using strata::poly::Options;
using strata::poly::Sampler;
using strata::poly::Sampling;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// serve() saves the experts' use after its first answer and then after an answer when the last save is this long
// ago, and not only when it is told to quit: a server that is ended on Windows takes its engine with it at once.
// A save is a file of some 40 KB and a rename.
constexpr double kProfileSaveMs = 10000.0;

// The most tokens of a window whose guesses are the block's: the token the model has and three guesses, each
// from the one before.  A repeat of the conversation may fill a longer window (Options::window), and so may a
// model whose guesses come together (Engine::guesses_at_once()).
constexpr int kBlockWindow = 4;

// A window longer than that is for a copy: a repeat of the conversation whose match is this many tokens or more.
// A repeat of a few tokens is in any text, and what followed it there is seldom what follows here for long.
constexpr int kCopyMatch = 8;

// With batch slots a prompt is read in pieces: between two, a BYIELD can stop the read and the slots write for half
// the time the piece took.  A piece is a call (max_rows()) while slots write, so that they are not held up long,
// and this many calls while none does, as Strata's engine reads its chunks.
constexpr int kPiece = 8;

// After a BCOK the BGENs of the prompts it names come one after the other, each as soon as the server has the last
// one's BADM: the slots wait this long for the next one before they write a step.  16 clients of Qwen3.6 at 12 GB
// had a step of 35-50 ms between each two of them.
constexpr double kCoWaitMs = 20.0;

// A session file (SAVE, RESTORE) starts with this, moves its rows this many bytes at a time, each followed by a
// SESSION line, and a SAVE leaves this much of its disk free, the default of the other engine's.
constexpr char kSessionMagic[8] = {'P', 'O', 'L', 'Y', 'S', 'E', 'S', '1'};
constexpr size_t kSessionPiece = 16u << 20;
constexpr uint64_t kSessionFreeMib = 4096;
constexpr uint64_t kSessionFrame = sizeof kSessionMagic + 8 + 8;   // around the conversation's bytes: the magic,
                                                                   // what the file is bound to, the hash

bool parse_ids(const char* s, std::vector<int32_t>& out) {
    out.clear();
    while (*s == ' ') ++s;
    while (*s != '\0') {
        char* end = nullptr;
        const long long v = std::strtoll(s, &end, 10);
        if (end == s || v < 0 || v > INT32_MAX) return false;
        out.push_back((int32_t) v);
        s = end;
        while (*s == ' ') ++s;
        if (*s == ',') ++s;
        else if (*s != '\0') return false;
    }
    return !out.empty();
}

uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// A request's pictures: strata-vision's records (tools/vision/strata_vision.cpp), one after the other in the
// order of the prompt.
struct Pictures {
    std::vector<float> rows;
    std::vector<strata::poly::Image> at;
};

// Reads a request's pictures and finds their places: each takes the next run of `pad` in ids, a token a row.
// Those ids then become numbers below zero made from the picture, so that the conversation cache tells a picture
// from another one of the same size.  Returns what is wrong, or nothing.
std::string take_pictures(const std::string& path, int32_t pad, int n_embd, std::vector<int32_t>& ids, Pictures& out) {
    struct Record {
        int n, nx;
        size_t off;
    };
    std::vector<Record> records;
    out = Pictures{};
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    std::string err;
    for (;;) {
        int32_t hdr[5];   // 'SVE1', rows, rows of a line, lines, a row's floats
        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[1] > 1 << 20 || hdr[2] < 1 || hdr[3] < 1 ||
            (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != n_embd) {
            err = "bad embeddings file (expected strata-vision records of width " + std::to_string(n_embd) + ")";
            break;
        }
        const size_t off = out.rows.size(), count = (size_t) hdr[1] * (size_t) n_embd;
        out.rows.resize(off + count);
        if (std::fread(out.rows.data() + off, sizeof(float), count, f) != count) {
            err = "short embeddings file";
            break;
        }
        records.push_back({hdr[1], hdr[2], off});
    }
    std::fclose(f);
    if (!err.empty()) return err;

    size_t i = 0;
    for (size_t k = 0; k < records.size(); ++k) {
        const Record& r = records[k];
        while (i < ids.size() && ids[i] != pad) ++i;
        for (int j = 0; j < r.n; ++j)
            if (i + (size_t) j >= ids.size() || ids[i + (size_t) j] != pad)
                return "picture " + std::to_string(k + 1) + " has " + std::to_string(r.n) +
                       " rows and the prompt fewer places for it";
        const float* rows = out.rows.data() + r.off;
        const int32_t grid[2] = {r.n, r.nx};
        const uint64_t h = fnv1a(rows, (size_t) r.n * (size_t) n_embd * sizeof(float), fnv1a(grid, sizeof grid));
        for (int j = 0; j < r.n; ++j)
            ids[i + (size_t) j] = -1 - (int32_t) ((h + (uint64_t) j * 0x9E3779B97F4A7C15ull) >> 33);
        out.at.push_back({(int64_t) i, r.n, r.nx, rows});
        i += (size_t) r.n;
    }
    if (std::find(ids.begin() + (std::ptrdiff_t) i, ids.end(), pad) != ids.end())
        return "the prompt has more pictures than the embeddings file";
    if (!ids.empty() && ids.back() < 0) return "a prompt cannot end in a picture";
    return {};
}

// stdin on its own thread, so a STOP is seen while a request runs
class Lines {
public:
    Lines() : th_([this] { read(); }) { th_.detach(); }

    std::string pop() {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [this] { return !q_.empty(); });
        std::string s = std::move(q_.front());
        q_.pop_front();
        return s;
    }
    /// The next line when there is one.
    bool try_pop(std::string& s) {
        std::lock_guard<std::mutex> l(m_);
        if (q_.empty()) return false;
        s = std::move(q_.front());
        q_.pop_front();
        return true;
    }
    /// The next line when one comes within `ms`.
    bool pop_for(std::string& s, double ms) {
        std::unique_lock<std::mutex> l(m_);
        if (!cv_.wait_for(l, std::chrono::duration<double, std::milli>(ms), [this] { return !q_.empty(); }))
            return false;
        s = std::move(q_.front());
        q_.pop_front();
        return true;
    }
    /// The lines waiting that start with `head`, out of the queue, the others left in their order.
    std::vector<std::string> take(const char* head) {
        std::lock_guard<std::mutex> l(m_);
        std::vector<std::string> out;
        for (auto it = q_.begin(); it != q_.end();) {
            if (it->rfind(head, 0) == 0) {
                out.push_back(std::move(*it));
                it = q_.erase(it);
            } else {
                ++it;
            }
        }
        return out;
    }

    std::atomic<bool> stop{false};

private:
    void read() {
        std::string l;
        bool quit = false;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            quit = quit || l == "QUIT";
            if (l == "STOP") stop.store(true);
            else push(l);
        }
        // The server closes the input after its QUIT; without one, the server is gone (#195: the engine then
        // ended with nothing in its log)
        if (!quit)
            std::fprintf(stderr, "strata-poly: the engine's input closed without a QUIT (the server ended, or was "
                                 "ended); the engine ends\n");
        push("QUIT");
    }
    void push(std::string s) {
        { std::lock_guard<std::mutex> l(m_); q_.push_back(std::move(s)); }
        cv_.notify_one();
    }

    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::string> q_;
    std::thread th_;
};

// What the conversation cache is told on the command line.
struct CacheFlags {
    CacheOptions o;
    long long min_free_mib = 2560;   // parking leaves this much physical RAM free
    bool park_auto = false;          // --conversation-cache-mib auto: park_mib from the RAM free once loaded
};

struct Session {
    Engine& engine;
    std::vector<int32_t> eos;
    CacheFlags flags;
    Conversations<Engine> kept;   // the conversation in the engine, and the parked ones
    Sampler sampler;
    std::mt19937_64 rng{std::random_device{}()};
    const std::vector<int32_t>* follow = nullptr;   // --follow: the answer's tokens, taken in place of the picks
    const float* picked_from = nullptr;             // the logits of the last token picked
    bool reads_logits = false;                      // something reads picked_from (--top-probs)
    long long asked = 0, reused = 0;   // prompt tokens of every request, and those of them not read again
    // Guesses at the tokens to come, checked a window at a time (Engine::window).  They come from the block some
    // files carry for it (Engine::draft), and from the conversation: when the last tokens repeat an earlier stretch
    // of it, what followed there is the guess.  The policy learns from the windows so far, their hits and their
    // time, which of the two pays.  It and the repeats are the other engine's (strata/spec).
    int most;                             // the tokens of a window at most; 1: no guesses
    bool look;                            // guess from repeats
    strata::spec::SuffixDrafter repeats;
    strata::spec::DraftPolicy policy;
    Guesses guesses;                      // whether the block is asked in a round, and for how many (guesses.hpp)
    int piece_calls = kPiece;             // generate()'s pieces when it is given `between`
    std::string pinned;                   // --pinned-session: the file serve() pins before its first request
    std::vector<Engine::CoRead> co_read;  // BCO: prompts for generate()'s first read() to take along
    bool co_done = false;                 // and that read took them

    Session(Engine& e, std::vector<int32_t> eos_ids, const CacheFlags& f, int lookup)
        : engine(e), eos(std::move(eos_ids)), flags(f), kept(e, with_floor(f)),
          most(lookup > 0 || e.drafts() ? std::min<int>(e.max_window(), strata::spec::DraftPolicy::kMaxT) : 1),
          look(lookup > 0 && most > 1),
          repeats(std::max(lookup, 3), 64, (size_t) e.max_context() + 4096), policy(most) {
        vocab.pieces = e.token_texts();
        vocab.eog = eos;
    }

    // A grammar can be applied when the model's tokens have texts (byte-level BPE) and there is a token to end with.
    llama_vocab vocab;
    bool grammars() const { return !vocab.pieces.empty() && !vocab.eog.empty(); }

    static CacheOptions with_floor(const CacheFlags& f) {
        CacheOptions o = f.o;
        o.log = stderr;
        o.admit = [floor = (uint64_t) f.min_free_mib << 20](size_t bytes) {
            return strata::core::conversation_memory_admit(strata::core::available_host_bytes(), bytes, floor);
        };
        return o;
    }

    bool is_eos(int32_t t) const { return std::find(eos.begin(), eos.end(), t) != eos.end(); }

    struct Result {
        int64_t produced = 0, resume = 0, read = 0;
        int64_t gave_way = -1;   // where `between` stopped the read
        int64_t guessed = 0, guessed_right = 0;
        double prompt_ms = 0, decode_ms = 0;
        const char* finish = "length";
    };

    // Reads the prompt and generates.  on_token gets every token, the end-of-sequence one included.  The rows of
    // the prompt's pictures have ids below zero (take_pictures()); `pad` is the token they stand in for.
    // `between`, when given, reads the prompt in pieces of piece_calls calls and is asked after each but the last
    // one, with where the read is: false stops it there, as a cancel does, and the conversation holds what was read.
    template<class OnToken, class OnProgress>
    Result generate(const std::vector<int32_t>& ids, const Pictures& pictures, int32_t pad, int64_t turn_at,
                    int64_t max_new, const Sampling& s, const std::atomic<bool>& stop, OnToken on_token,
                    OnProgress on_progress, const std::function<bool(int64_t)>* between = nullptr) {
        Result r;
        const int64_t n = (int64_t) ids.size();
        Clock::time_point t0 = Clock::now(), shown = t0;   // bringing a parked conversation back is prompt time
        engine.images(pictures.at.data(), (int) pictures.at.size());
        r.resume = kept.resume(ids);
        if (const size_t freed = engine.trim_kv(n)) {
            const int had = engine.cached_experts();
            engine.mind_vram();
            std::fprintf(stderr, "strata-poly: the attention caches gave back %.2f GiB that this request and the one "
                                 "before do not reach; the expert cache %d -> %d experts\n",
                         double(freed) / double(1ull << 30), had, engine.cached_experts());
        }
        kept.turn_at(turn_at);
        asked += n;
        reused += r.resume;
        on_progress(r.resume, n, 0.0, true);
        if (s.seed != 0) rng.seed(s.seed);

        // The last token goes in alone, for the logits that come with it.  The tokens before it go in as many at
        // a time as the engine takes.  A stretch of prompt without pictures goes in several calls' worth at once,
        // which the engine reads faster than the calls (read()), and the checkpoints due inside are the engine's
        // to write.  A stretch with a picture in it goes in eval() calls, a picture whole, up to the checkpoint;
        // the text before and after that picture is read like a prompt without pictures.
        const bool pictures_any = !pictures.at.empty();
        std::vector<std::pair<int64_t, uint8_t*>> due;
        std::vector<Engine::Mark> marks;
        for (int64_t i = r.resume; i < n;) {
            if (stop.load()) {
                r.finish = "cancel";
                break;
            }
            const bool last = i == n - 1;
            const int64_t most = between ? (int64_t) piece_calls * engine.max_rows() : INT64_MAX;
            // How far the text from i runs: to the next picture's first row, or to the last token, which goes in
            // alone.  read() takes no row of a picture, so only a run of text can go through it.  A run shorter
            // than a call keeps to eval(): its pieces are the calls it would take anyway.
            int64_t picture_at = n;
            bool in_picture = false;
            for (const auto& im : pictures.at) {
                if (i >= im.at && i < im.at + im.n) in_picture = true;
                else if (im.at > i && im.at < picture_at) picture_at = im.at;
            }
            const int64_t text_run = in_picture ? 0 : std::min<int64_t>(picture_at, n - 1) - i;
            const bool text = !pictures_any || text_run >= engine.max_rows();
            const int64_t room = std::min<int64_t>({text ? text_run : n - 1 - i,
                                                    text ? engine.max_read() : engine.max_rows(), most});
            const int64_t m = last ? 1 : text ? room : engine.whole(i, kept.until_checkpoint(i, room));
            kept.before(i, last);
            if (text && !last) {
                kept.inside(i, m, due);
                marks.clear();
                for (const auto& d : due) marks.push_back({(int) d.first, d.second});
                // A long prompt is one read.  Between its layers the read says how far it is, and a request that
                // was stopped gives it up there: the conversation is then at its last checkpoint.
                const std::function<bool(int, int)> turn = [&](int done, int of) {
                    if (stop.load()) return false;
                    if (ms_since(shown) >= 500.0) {
                        on_progress(i + m * done / of, n, ms_since(t0), false);
                        shown = Clock::now();
                    }
                    return true;
                };
                // the prompts co_read holds go in the same pass as this request's first read (Strata's BCO)
                const bool with_co = !co_read.empty();
                if (!engine.read(ids.data() + i, (int) m, marks.data(), (int) marks.size(), &turn,
                                 with_co ? co_read.data() : nullptr, (int) co_read.size())) {
                    co_read.clear();
                    kept.rewind();
                    r.finish = "cancel";
                    break;
                }
                co_done = with_co;
                co_read.clear();
            } else {
                // A request with nothing to read but its last token (the server's: one that wrote alone and goes on
                // in a slot, its prompt in the conversation) reads the BCO prompts on their own first.
                if (last && !co_read.empty()) {
                    co_done = engine.read(ids.data() + i, 0, nullptr, 0, nullptr, co_read.data(), (int) co_read.size());
                    co_read.clear();
                }
                if (last) engine.end_read();   // the last token, as the answer's, with the whole expert cache
                engine.eval(ids.data() + i, (int) m, last);
            }
            kept.read(ids.data() + i, m);
            i += m;
            r.read += m;
            if (ms_since(shown) >= 500.0 || last) {
                on_progress(i, n, ms_since(t0), false);
                shown = Clock::now();
            }
            if (between && i < n - 1 && !(*between)(i)) {
                r.gave_way = i;
                r.finish = "cancel";
                break;
            }
        }
        engine.end_read();
        kept.prompt_was(n);
        r.prompt_ms = ms_since(t0);
        if (std::strcmp(r.finish, "cancel") == 0) return r;

        t0 = Clock::now();
        std::vector<int32_t> history = kept.seq();
        std::replace_if(history.begin(), history.end(), [](int32_t t) { return t < 0; }, pad);
        if (look) {
            repeats.reset();
            repeats.append(history.data(), history.size());
        }
        const auto pick = [&](const float* logits) {
            picked_from = logits;
            if (follow && r.produced < (int64_t) follow->size()) return (*follow)[(size_t) r.produced];
            return (int32_t) sampler.pick(logits, engine.n_vocab(), s, history, rng);
        };
        // Shows a token.  False when the answer ends with it.
        const auto show = [&](int32_t t) {
            on_token(t);
            ++r.produced;
            history.push_back(t);
            if (look) repeats.append(t);
            if (s.grammar) s.grammar->accept(t);
            if (is_eos(t)) r.finish = "stop";
            else if (stop.load()) r.finish = "cancel";
            else return r.produced < max_new;
            return false;
        };
        // a greedy pick that nothing else reads the logits for: a token at a time the card picks it.  Not with a
        // draft block that guesses a token at a time: Qwen3.6 with its MTP block wrote 0.4% slower after a short
        // prompt (RTX 5090, UD-Q3_K_M).
        const bool card_picks =
            Sampler::greedy(s) && !follow && !reads_logits && (!engine.drafts() || engine.guesses_at_once());
        if (max_new > 0) {
            int32_t w[strata::spec::DraftPolicy::kMaxT], d[strata::spec::DraftPolicy::kMaxT];
            const int block_window = engine.guesses_at_once() ? most : kBlockWindow;
            int32_t t = pick(engine.logits());   // picked, and not shown yet
            while (show(t)) {
                // t goes into the model now, and with it guesses at what follows when there are any that pay
                const Clock::time_point round = Clock::now();
                const bool room = engine.pos() + most <= engine.max_context();
                const int offered = look && room ? repeats.propose(most - 1, w + 1) : 0;
                const int match = repeats.last_match();
                const int widest = match >= kCopyMatch ? most : std::min(most, block_window);   // this round's window
                const int k = std::min(offered, widest - 1);
                // A repeat that fills the window, from a match of a length that has been right nine times in ten,
                // is taken as it is: the block would mostly say the same, and asking it takes time.  Otherwise the
                // block guesses, each guess from the one before, while a guess more pays (Guesses).
                const bool blocks = engine.drafts() && room && !(k == widest - 1 && policy.lookup_rate(match) >= 0.9);
                const bool ask = guesses.begin(blocks, policy);
                // the round's time, as the policy counts it: a window of `in` tokens wrote n of them
                const auto count = [&](int in, int n) {
                    return guesses.end(in, n, ms_since(round), engine.took_record());
                };
                strata::spec::DraftPolicy::Pick use = policy.choose(1, k, match);
                if (ask) {
                    int n_d = 0;
                    for (int32_t from = t; n_d < std::min(most, block_window) - 1 && guesses.wants(n_d); from = d[n_d++]) {
                        float p = 0.f;
                        const Clock::time_point asked_at = Clock::now();
                        d[n_d] = engine.draft(from, n_d, &p);
                        if (!guesses.takes(n_d, p, ms_since(asked_at))) break;
                    }
                    use = policy.choose(1 + n_d, k, match);
                }
                if (use.t == 1) {
                    if (card_picks) engine.eval_picks();
                    engine.eval(&t, 1, true);
                    kept.read(&t, 1);
                    t = card_picks ? engine.picked(0) : pick(engine.logits());
                    policy.observe(false, 1, 0, 0, count(1, 1));
                    continue;
                }
                w[0] = t;
                if (!use.lookup) std::copy(d, d + use.t - 1, w + 1);
                engine.window(w, use.t);
                // logits_of(i) is what follows w[i].  A guess that is what was picked has been through the model
                // already; the token an answer ends with is shown and never fed, as without guesses.
                int right = 0;
                bool more = true;
                t = pick(engine.logits_of(0));
                while (right < use.t - 1 && t == w[right + 1] && (more = show(t))) t = pick(engine.logits_of(++right));
                engine.keep(right + 1);
                kept.read(w, right + 1);
                r.guessed += use.t - 1;
                r.guessed_right += right + (more ? 0 : 1);
                policy.observe(use.lookup, use.t, right, match, count(use.t, right + 1));
                if (!more) break;
            }
        }
        r.decode_ms = ms_since(t0);
        return r;
    }
};

// Standard base64 (RFC 4648, with or without the = padding), or false for anything else.
bool base64_decode(const std::string& in, std::string& out) {
    static const char* const kAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    out.clear();
    uint32_t acc = 0;
    int bits = 0;
    for (const char c : in) {
        if (c == '=') break;
        const char* at = std::strchr(kAlphabet, c);
        if (c == '\0' || !at) return false;
        acc = (acc << 6) | (uint32_t) (at - kAlphabet);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((char) ((acc >> bits) & 0xFF));
        }
    }
    return true;
}

// GEN <max_new> [key=value ...] <id,id,...>, or GENI with the file of the prompt's pictures ahead of the ids and
// the key image_pad, the token whose runs in the ids are the pictures' places.  turn_at is the place in the ids
// where the prompt's last turn begins (the server knows the template's turn tokens): a checkpoint goes there.
bool parse_gen(const std::string& line, long long& max_new, Sampling& s, int32_t& pad, int64_t& turn_at,
               std::string& pictures, std::vector<int32_t>& ids) {
    const bool geni = line.rfind("GENI ", 0) == 0;
    char* p = nullptr;
    max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &p, 10);
    if (max_new < 1) return false;
    pad = -1;
    turn_at = -1;
    for (;;) {   // key=value tokens, then the file and the ids
        while (*p == ' ') ++p;
        const char* start = p;
        while (*p != '\0' && *p != ' ') ++p;
        const std::string tok(start, (size_t) (p - start));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            p = const_cast<char*>(start);
            break;
        }
        const std::string key = tok.substr(0, eq);
        const char* v = tok.c_str() + eq + 1;
        if (key == "temperature") s.temperature = std::strtof(v, nullptr);
        else if (key == "top_p") s.top_p = std::strtof(v, nullptr);
        else if (key == "top_k") s.top_k = std::atoi(v);
        else if (key == "min_p") s.min_p = std::strtof(v, nullptr);
        else if (key == "penalty_last_n") s.penalty_last_n = std::atoi(v);
        else if (key == "penalty_repeat") s.penalty_repeat = std::strtof(v, nullptr);
        else if (key == "penalty_freq") s.penalty_freq = std::strtof(v, nullptr);
        else if (key == "penalty_present") s.penalty_present = std::strtof(v, nullptr);
        else if (key == "seed") s.seed = std::strtoull(v, nullptr, 10);
        else if (key == "image_pad") pad = (int32_t) std::atoi(v);
        else if (key == "turn_at") turn_at = std::strtoll(v, nullptr, 10);
        else if (key == "logit_bias") s.bias_text = v;
        else if (key == "grammar") s.grammar_text = v;
        // other keys belong to the other engine and are skipped
    }
    pictures.clear();
    if (geni) {
        const char* end = std::strchr(p, ' ');
        if (!end || end == p || pad < 0) return false;
        pictures.assign(p, (size_t) (end - p));
        p = const_cast<char*>(end);
    }
    return parse_ids(p, ids);
}

// A batch slot (--batch): a request that goes on in a sequence of its own, slot b in sequence b + 1, written with
// the others a token each at a time (Engine::step()).  Its prompt is read in sequence 0, which the conversation
// cache knows, and copied over.
struct Slot {
    bool active = false, stop = false;
    int32_t next = 0;               // picked and not fed yet
    long long produced = 0, max_new = 0;
    Sampling s;
    std::mt19937_64 rng{std::random_device{}()};
    std::vector<int32_t> held;      // what its sequence has read: the prompt and the tokens fed since
    std::vector<float> first;       // a prompt read ahead (BCO), all of it: the logits its first token is picked from
    Clock::time_point t0;
};

// What a session file is bound to: this engine version, the model's files (each one's size, first and last MiB,
// read from where the engine has them mapped) and what decides the bytes of a row and of a state.  A change in
// the middle of a model file that keeps its size is not seen.
uint64_t session_binds(const Engine& engine) {
    strata::core::SessionIdentityBuilder b(0x504f4c5953455353ull);
    b.str("engine", STRATA_VERSION);
    for (const auto& [data, size] : engine.model_views()) {
        const size_t edge = std::min<size_t>(size, 1u << 20);
        b.u64("size", size);
        b.bytes("head", data, edge);
        b.bytes("tail", data + (size - edge), edge);
    }
    b.i64("kv", engine.kv_format());
    b.u64("row", engine.row_bytes());
    b.u64("state", engine.state_bytes());
    return b.digest();
}

// A session file's name as the server sends it and a command line gives it: UTF-8.
std::filesystem::path session_path(const std::string& name) {
    std::u8string utf8(name.size(), u8'\0');
    std::memcpy(utf8.data(), name.data(), name.size());
    return std::filesystem::path(utf8);
}

// Whether a file of `total` bytes starts as this engine's session files do: the magic and the binding, read with
// get().  Returns what is wrong with it, or nothing.
template<class Get>
std::string session_starts(uint64_t total, uint64_t binds, Get&& get) {
    char magic[sizeof kSessionMagic];
    uint64_t bound = 0;
    if (total < kSessionFrame || !get(magic, sizeof magic) || std::memcmp(magic, kSessionMagic, sizeof magic) != 0)
        return "not a strata-poly session file";
    if (!get(&bound, 8) || bound != binds) return "the file was saved with another model, engine version or --kv";
    return {};
}

// --pinned-session FILE before the first request, and PIN <path> between requests: the file's conversation becomes
// the pinned one (Conversations::pin()), a system prompt that was read once and saved, in place of one pinned
// before.  Returns what is wrong with the file, or nothing; then nothing is pinned.  `io`: it could not be read.
std::string pin_session(Session& ss, const std::string& name, bool& io) {
    namespace fs = std::filesystem;
    using Restored = Conversations<Engine>::Restored;
    const Clock::time_point t0 = Clock::now();
    io = false;
    const auto no = [&](const std::string& why) {
        ss.kept.unpin();
        return why;
    };
    std::error_code ec;
    const fs::path path = session_path(name);
    const uint64_t total = fs::file_size(path, ec);
    io = true;
    if (ec) return no("the file cannot be read: " + ec.message());
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return no("the file cannot be opened");
    io = false;
    strata::core::SessionHasher hash;
    const auto get = [&](void* p, size_t n) {
        in.read((char*) p, (std::streamsize) n);
        if ((size_t) in.gcount() != n) return false;
        hash.update(p, n);
        return true;
    };
    if (const std::string why = session_starts(total, session_binds(ss.engine), get); !why.empty()) return no(why);
    const Restored r = ss.kept.pin(total - kSessionFrame, ss.engine.max_context(), kSessionPiece, get);
    if (r == Restored::other) return no("the file is not a whole session of this model within --max-context");
    if (r == Restored::memory) return no("no memory for it");
    uint64_t want = 0;
    if (r == Restored::ok) in.read((char*) &want, 8);
    if (r != Restored::ok || in.gcount() != 8 || want != hash.digest()) return no("the file is damaged");
    std::fprintf(stderr, "strata-poly: pinned session: %lld tokens, %zu MiB of rows and states beside the conversation "
                         "cache, read in %.0f ms; a prompt that starts with them goes on from there\n",
                 (long long) ss.kept.pinned_tokens(), ss.kept.pinned_bytes() >> 20, ms_since(t0));
    return {};
}

// SAVE <path> | RESTORE <path> between requests, answered as the Strata project's engine answers them
// (docs/DETAILS.md, Session files): SAVED / RESTORED <tokens> <bytes> <ms>, SESSION <done> <total> on the way, and
// SERR <invalid|storage|memory|io> 0 <reason> for a file that is refused or fails, after which the engine goes on.
// The file: kSessionMagic, what it is bound to (session_binds()), Conversations::save()'s bytes, and a hash of all
// that.  It is written beside its name and renamed.  It is not flushed to the disk: a file that a power loss cut
// short fails its hash and is refused.  False: the engine cannot go on, and FATAL was answered.
bool session_file(Session& ss, bool save, const std::string& name, bool batch) {
    namespace fs = std::filesystem;
    using strata::core::SessionError;
    using Restored = Conversations<Engine>::Restored;
    Engine& engine = ss.engine;
    const Clock::time_point t0 = Clock::now();
    const auto refuse = [&](SessionError kind, std::string why) {
        for (char& c : why)
            if (c == '\n' || c == '\r') c = ' ';
        std::fprintf(stderr, "strata-poly: session %s %s: %s\n", save ? "save" : "restore", name.c_str(), why.c_str());
        std::printf("SERR %s 0 %s\n", strata::core::session_error_name(kind), why.c_str());
        std::fflush(stdout);
        return true;
    };
    if (name.empty()) return refuse(SessionError::invalid, "missing path");
    if (batch) return refuse(SessionError::invalid, "session files do not support --batch (parallel requests)");
    const uint64_t binds = session_binds(engine);
    const fs::path path = session_path(name);
    const uint64_t frame = kSessionFrame;
    std::error_code ec;
    strata::core::SessionHasher hash;
    uint64_t done = 0, said = 0, total = 0;
    const auto moved = [&] {
        std::printf("SESSION %llu %llu\n", (unsigned long long) (said = done), (unsigned long long) total);
        std::fflush(stdout);
    };

    if (save) {
        const uint64_t body = ss.kept.save_bytes();
        if (body == 0 || engine.pos() != (int64_t) ss.kept.seq().size())
            return refuse(SessionError::invalid, "no complete conversation to save");
        total = frame + body;
        const fs::path dir = path.has_parent_path() ? path.parent_path() : fs::path(".");
        const fs::space_info room = fs::space(dir, ec);
        if (ec) return refuse(SessionError::io, "the folder of the file: " + ec.message());
        if (room.available < total + (kSessionFreeMib << 20))
            return refuse(SessionError::storage,
                          "the file takes " + std::to_string(total >> 20) + " MiB and has to leave " +
                              std::to_string(kSessionFreeMib) + " MiB free; the disk has " +
                              std::to_string(room.available >> 20) + " MiB");
        fs::path part = path;
        part += ".part";
        std::ofstream out(part, std::ios::binary | std::ios::trunc);
        // a conversation is its owner's to read, as the other engine's session files are (mode 0600)
        fs::permissions(part, fs::perms::owner_read | fs::perms::owner_write, ec);
        const auto put = [&](const void* p, size_t n) {
            out.write((const char*) p, (std::streamsize) n);
            hash.update(p, n);
            done += n;
            if (done - said >= kSessionPiece) moved();
            return out.good();
        };
        std::string why;
        bool ok = out.is_open() && put(kSessionMagic, sizeof kSessionMagic) && put(&binds, 8), no_memory = false;
        try {
            if (ok && !ss.kept.save(kSessionPiece, put)) ok = false, no_memory = out.good();
        } catch (const std::exception& e) {
            ok = false;
            why = e.what();
        }
        const uint64_t h = hash.digest();
        if (ok) out.write((const char*) &h, 8);
        out.close();
        ok = ok && !out.fail();
        if (ok) {
            fs::rename(part, path, ec);
            if (ec) ok = false, why = ec.message();
        }
        if (!ok) {
            fs::remove(part, ec);
            if (no_memory) return refuse(SessionError::memory, "no memory to move the conversation's rows");
            const fs::space_info left = fs::space(dir, ec);
            return refuse(!ec && left.available < total ? SessionError::storage : SessionError::io,
                          why.empty() ? "the file cannot be written" : why);
        }
        done = total;
        moved();
        std::fprintf(stderr, "strata-poly: session saved: %zu tokens, %llu MiB in %.0f ms\n", ss.kept.seq().size(),
                     (unsigned long long) (total >> 20), ms_since(t0));
        std::printf("SAVED %zu %llu %.1f\n", ss.kept.seq().size(), (unsigned long long) total, ms_since(t0));
        std::fflush(stdout);
        return true;
    }

    total = fs::file_size(path, ec);
    if (ec) return refuse(SessionError::io, "the file cannot be read: " + ec.message());
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return refuse(SessionError::io, "the file cannot be opened");
    const auto get = [&](void* p, size_t n) {
        in.read((char*) p, (std::streamsize) n);
        if ((size_t) in.gcount() != n) return false;
        hash.update(p, n);
        done += n;
        if (done - said >= kSessionPiece) moved();
        return true;
    };
    uint64_t want = 0;
    if (const std::string why = session_starts(total, binds, get); !why.empty())
        return refuse(SessionError::invalid, why);
    Restored r = Restored::other;
    try {
        r = ss.kept.restore(total - frame, engine.max_context(), kSessionPiece, get);
        if (r == Restored::ok) {
            in.read((char*) &want, 8);
            if (in.gcount() != 8 || want != hash.digest()) {
                ss.kept.drop();
                engine.reset();
                r = Restored::cut;
            }
        }
    } catch (const std::exception& e) {
        // the engine's state is unknown now, as after a request that failed
        ss.kept.drop();
        try {
            engine.reset();
        } catch (const std::exception& again) {
            std::printf("FATAL %s\n", again.what());
            std::fflush(stdout);
            return false;
        }
        return refuse(SessionError::io, std::string(e.what()) + "; the engine holds no conversation now");
    }
    if (r == Restored::other)
        return refuse(SessionError::invalid, "the file is not a whole session of this model within --max-context");
    if (r == Restored::memory) return refuse(SessionError::memory, "no memory for the session's checkpoints");
    if (r == Restored::cut)
        return refuse(SessionError::invalid, "the file is damaged; the engine holds no conversation now");
    done = total;
    moved();
    std::fprintf(stderr, "strata-poly: session restored: %zu tokens, %llu MiB in %.0f ms\n", ss.kept.seq().size(),
                 (unsigned long long) (total >> 20), ms_since(t0));
    std::printf("RESTORED %zu %llu %.1f\n", ss.kept.seq().size(), (unsigned long long) total, ms_since(t0));
    std::fflush(stdout);
    return true;
}

int serve(Session& ss) {
    Engine& engine = ss.engine;
    std::vector<Slot> slots((size_t) engine.sequences() - 1);
    if (!ss.pinned.empty()) {
        // A file that cannot be used is named and the engine starts without it: the first request that brings the
        // prompt reads it then.
        bool io = false;
        if (const std::string why = pin_session(ss, ss.pinned, io); !why.empty())
            std::fprintf(stderr, "strata-poly: --pinned-session %s: %s; the engine starts without it\n",
                         ss.pinned.c_str(), why.c_str());
    }
    engine.trim();
    std::printf("INFO logit_bias=1%s context=%d expert_slots=%d expert_cache_mib=%lld pool_workers=%d conversation_cache_mib=%lld "
                "conversation_cache_slots=%d conversation_cache_min_free_mib=%lld vram_free_mib=%lld vram_elastic=1 "
                "batch_slots=%d co_read=%d sessions=1 pinned_tokens=%lld engine=" STRATA_VERSION "\n",
                ss.grammars() ? " grammar=1" : "", engine.max_context(), engine.cached_experts(), (long long) (engine.cached_gib() * 1024.0),
                engine.threads(), (long long) ss.flags.o.park_mib, ss.flags.o.park_slots, ss.flags.min_free_mib,
                (long long) engine.vram_free_mib(), (int) slots.size(), !slots.empty() && engine.co_reads() ? 1 : 0,
                (long long) ss.kept.pinned_tokens());
    std::printf("READY %d stop\n", engine.max_context());
    std::fflush(stdout);
    // READY goes to the server alone: without this line the log of an engine that waits for its first request ends
    // where the log of one that hangs before READY does (#195)
    std::fprintf(stderr, "strata-poly: ready for requests (context %d); an engine that waits for one uses no CPU\n",
                 engine.max_context());

    static Lines lines;   // its thread may still be in a read when this function returns
    Pictures pictures;    // the engine reads the last request's until the next one's are set
    // the experts' use (--expert-profile): as if it was last saved long ago, so that the first answer saves it
    Clock::time_point saved = Clock::now() - std::chrono::milliseconds((long long) kProfileSaveMs);
    const auto writing = [&] {
        return std::any_of(slots.begin(), slots.end(), [](const Slot& sl) { return sl.active; });
    };
    // A token for every slot that writes: BT <slot> <token> each, and BDONE <slot> <produced> <finish> <ms> for a
    // slot that ends with it.  False when the engine failed, which leaves it in no known state.
    std::vector<Sampler> samplers(slots.size());
    std::vector<int> seqs, kept, row_of;
    std::vector<int32_t> feed, guess, picked;
    std::vector<size_t> which;
    const auto step = [&] {
        seqs.clear();
        feed.clear();
        which.clear();
        guess.clear();
        for (size_t b = 0; b < slots.size(); ++b)
            if (slots[b].active) {
                seqs.push_back((int) b + 1);
                feed.push_back(slots[b].next);
                which.push_back(b);
            }
        // --batch-mtp: each slot's draft block guesses the token after its next one, and the step checks it
        const bool guessing = engine.steps_guesses();
        // every slot greedy with no penalty: the card picks, Sampler::pick()'s most_likely() to the token
        const bool on_card = std::all_of(which.begin(), which.end(), [&](size_t b) {
            const Sampling& s = slots[b].s;
            return Sampler::greedy(s);
        });
        try {
            if (guessing) {
                guess.resize(seqs.size());
                engine.draft_many(seqs.data(), feed.data(), (int) seqs.size(), guess.data());
                // a step of more rows reads them as a prompt batch, whose kernels round otherwise: the slots past
                // that many check no guess this step
                for (size_t i = 0, rows = seqs.size(); i < guess.size(); ++i)
                    if (guess[i] >= 0 && (int) ++rows > engine.step_rows_alone()) guess[i] = -1;
                if (on_card) engine.step_picks();
                engine.step(seqs.data(), feed.data(), (int) seqs.size(), guess.data());
            } else {
                if (on_card) engine.step_picks();
                engine.step(seqs.data(), feed.data(), (int) seqs.size());
            }
        } catch (const strata::poly::KvFull& e) {
            // The attention caches could not grow for the step, and what they hold is as it was: the step's answers
            // end here, as at the end of the context, and the engine takes the next requests.
            std::fprintf(stderr, "strata-poly: batch: %s; the step's %zu answers end here\n", e.what(), which.size());
            for (size_t b : which) {
                Slot& sl = slots[b];
                std::printf("BDONE %zu %lld length %.1f\n", b, sl.produced, ms_since(sl.t0));
                sl.active = false;
            }
            std::fflush(stdout);
            return true;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-poly: batch: %s\n", e.what());
            std::printf("ERR %s\n", e.what());
            std::fflush(stdout);
            return false;
        }
        kept.assign(which.size(), 1);
        row_of.resize(which.size());
        picked.resize(2 * which.size());
        for (size_t i = 0, row = 0; i < which.size(); ++i) {
            slots[which[i]].held.push_back(feed[i]);
            row_of[i] = (int) row;
            row += guessing && guess[i] >= 0 ? 2 : 1;
        }
        // whether a slot's answer ends with t, its next token
        const auto ends = [&](const Slot& sl, int32_t t) {
            return ss.is_eos(t) || sl.stop || sl.produced + 1 >= sl.max_new ||
                   (long long) sl.held.size() + 2 > engine.max_context();
        };
        // A slot's picks are its own (its sampler, history and generator), and a pick reads the whole vocabulary:
        // the CPU's threads take a slot each, where one thread took 1.1 ms of a 16-slot step.  A guess that is what
        // was picked has been through the model already, so the token after it is picked too, unless the answer ends
        // with the first; the token an answer ends with is shown and never fed, as without guesses.
        engine.on_threads((int) which.size(), [&](int i) {
            Slot& sl = slots[which[(size_t) i]];
            Sampler& sp = samplers[which[(size_t) i]];
            const auto pick = [&](int row) {
                return on_card ? engine.picked(row)
                               : (int32_t) sp.pick(engine.logits_of(row), engine.n_vocab(), sl.s, sl.held, sl.rng);
            };
            const int32_t t = pick(row_of[(size_t) i]);
            picked[2 * (size_t) i] = t;
            picked[2 * (size_t) i + 1] = -1;
            if (!guessing || guess[(size_t) i] != t || ends(sl, t)) return;
            sl.held.push_back(t);
            picked[2 * (size_t) i + 1] = pick(row_of[(size_t) i] + 1);
            sl.held.pop_back();
        });
        for (size_t i = 0; i < which.size(); ++i) {
            Slot& sl = slots[which[i]];
            // a token for the slot: false when its answer ends with it
            const auto emit = [&](int32_t t) {
                ++sl.produced;
                std::printf("BT %zu %d\n", which[i], (int) t);
                const char* fin = ss.is_eos(t) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                : (long long) sl.held.size() + 2 > engine.max_context() ? "length" : nullptr;
                if (fin) {
                    std::printf("BDONE %zu %lld %s %.1f\n", which[i], sl.produced, fin, ms_since(sl.t0));
                    sl.active = false;
                }
                return fin == nullptr;
            };
            int32_t t = picked[2 * i];
            if (emit(t) && picked[2 * i + 1] >= 0) {
                sl.held.push_back(t);
                kept[i] = 2;
                t = picked[2 * i + 1];
                emit(t);
            }
            sl.next = t;
        }
        if (guessing) {
            try {
                engine.keep_step(kept.data());
            } catch (const std::exception& e) {
                std::fprintf(stderr, "strata-poly: batch: %s\n", e.what());
                std::printf("ERR %s\n", e.what());
                std::fflush(stdout);
                return false;
            }
        }
        std::fflush(stdout);
        return true;
    };
    // A prompt that goes on from what a slot's sequence holds, further than from what sequence 0 holds, starts
    // from a copy of that: a request that left its slot to write alone again (with guesses), or the next turn of
    // a conversation a slot answered.
    const auto from_slot = [&](const std::vector<int32_t>& ids) {
        const auto common = [&](const std::vector<int32_t>& a) {
            size_t n = 0;
            while (n < a.size() && n < ids.size() && a[n] == ids[n]) ++n;
            return n;
        };
        size_t best = slots.size(), most = common(ss.kept.seq());
        for (size_t b = 0; b < slots.size(); ++b) {
            const std::vector<int32_t>& h = slots[b].held;
            if (!h.empty() && h.size() < ids.size() && common(h) == h.size() && h.size() > most) best = b, most = h.size();
        }
        if (best == slots.size()) return;
        engine.bind(0);
        ss.kept.leave();
        engine.copy_seq((int) best + 1, 0);
        ss.kept.adopt(slots[best].held);
    };
    // Between two pieces of a prompt read (Session::generate()): BSTOPs that came meanwhile, a BYIELD <slot> that
    // stops the read there to wait in that slot (YIELDED <slot> <tokens read>), and the slots' tokens for half the
    // time the piece took.  A read gives way only to a slot that does not write, with text alone, and with more than
    // a piece left: the rest is read the same way after it.
    int admit = -1, gives_to = -1;
    // BCO <slot> <id,id,...> (Strata's protocol; STRATA_COALESCE_PREFILL on the server): a waiting request's prompt,
    // read into that idle slot's sequence in the same pass as the next BGEN's.  All but its last token: its own BGEN
    // finds the rest in the slot (from_slot()) and reads that one.  BCOK <slot> <tokens> says the slot holds them.
    struct CoReq {
        int slot;
        std::vector<int32_t> ids;
    };
    std::vector<CoReq> co_pending;
    size_t co_due = 0;   // the BGENs still to come of the slots the last BCOK named (kCoWaitMs)
    long long ids_n = 0;
    bool text = true, failed = false;
    Clock::time_point piece_t0;
    const std::function<bool(int64_t)> between = [&](int64_t at) {
        for (const std::string& l : lines.take("BSTOP ")) {
            const int b = std::atoi(l.c_str() + 6);
            if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
        }
        int to = -1;
        for (const std::string& l : lines.take("BYIELD")) to = l.size() > 7 ? std::atoi(l.c_str() + 7) : -1;
        if (to >= 0 && to < (int) slots.size() && !slots[(size_t) to].active && (admit < 0 || to == admit) && text &&
            ids_n - at > (long long) engine.max_rows()) {
            gives_to = to;
            return false;
        }
        const double piece_ms = ms_since(piece_t0);
        const Clock::time_point t0 = Clock::now();
        while (writing() && ms_since(t0) < 0.5 * piece_ms)
            if (!step()) {
                failed = true;
                throw std::runtime_error("a batch step failed");
            }
        ss.piece_calls = writing() ? 1 : kPiece;
        piece_t0 = Clock::now();
        return true;
    };
    for (;;) {
        std::string line;
        if (!writing()) {
            line = lines.pop();
        } else if (!lines.try_pop(line) && !(co_due > 0 && lines.pop_for(line, kCoWaitMs))) {
            co_due = 0;   // (they stopped coming: the slots go on)
            if (!step()) return 1;
            continue;
        }
        if (line == "QUIT") break;
        if (line.empty()) continue;
        // BSTOP <slot>: the slot ends with its next token.  BYIELD: a long read gives way to a shorter one in the
        // other engine only; this one reads it to its end.
        if (line.rfind("BSTOP ", 0) == 0) {
            const int b = std::atoi(line.c_str() + 6);
            if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
            continue;
        }
        if (line.rfind("BYIELD", 0) == 0) continue;
        if (line.rfind("BCO ", 0) == 0) {   // kept for the next BGEN; one it cannot take says nothing
            CoReq cr;
            char* e1 = nullptr;
            cr.slot = (int) std::strtol(line.c_str() + 4, &e1, 10);
            for (const char* q = e1; q != nullptr && *q != '\0';) {
                while (*q == ' ' || *q == ',') ++q;
                if (*q == '\0') break;
                char* e2 = nullptr;
                const long long t = std::strtoll(q, &e2, 10);
                if (e2 == q || t < 0 || t >= engine.n_vocab()) {
                    cr.ids.clear();
                    break;
                }
                cr.ids.push_back((int32_t) t);
                q = e2;
            }
            if (cr.slot >= 0 && cr.slot < (int) slots.size() && cr.ids.size() >= 3) co_pending.push_back(std::move(cr));
            continue;
        }
        lines.stop.store(false);   // a STOP that came between requests is stale
        // BGEN <slot> <max_new> ... reads the prompt and picks its first token as GEN 1 does, then the request goes
        // on in that slot (BADM <slot> 1), or ends there (BADM <slot> 0)
        admit = -1;
        gives_to = -1;
        long long admit_max = 0;
        std::vector<CoReq> co_set = std::move(co_pending);   // the BCOs before this line are this line's alone
        co_pending.clear();
        if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
            co_due -= co_due > 0;
            const bool img = line[4] == 'I';
            char* e1 = nullptr;
            const long b = std::strtol(line.c_str() + (img ? 6 : 5), &e1, 10);
            char* e2 = nullptr;
            const long long mn = std::strtoll(e1, &e2, 10);
            if (b < 0 || b >= (long) slots.size() || slots[(size_t) b].active || mn < 1 || e2 == e1) {
                std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", (int) slots.size());
                std::fflush(stdout);
                continue;
            }
            admit = (int) b;
            admit_max = mn;
            line = std::string(img ? "GENI 1" : "GEN 1") + e2;
        }
        long long max_new = 0;
        Sampling s;
        std::vector<int32_t> ids;
        int32_t pad = -1;
        int64_t turn_at = -1;
        std::string file;
        // VRAM [reserve_mib], the server's POST /v1/vram: the expert cache sized again between requests, to leave
        // that much VRAM to other programs; alone, back to what the engine started with
        if (line.rfind("VRAM", 0) == 0 && (line.size() == 4 || line[4] == ' ')) {
            if (writing()) {
                std::printf("ERR VRAM: not while batch slots are writing\n");
                std::fflush(stdout);
                continue;
            }
            char* end = nullptr;
            const bool bare = line.find_first_not_of(' ', 4) == std::string::npos;
            const long long reserve = bare ? -1 : std::strtoll(line.c_str() + 4, &end, 10);
            if (!bare && (end == line.c_str() + 4 || reserve < 0 || reserve > (1LL << 24))) {
                std::printf("ERR expected: VRAM [reserve_mib]\n");
                std::fflush(stdout);
                continue;
            }
            try {
                const Clock::time_point t0 = Clock::now();
                const int before = engine.cached_experts();
                // The operator's resize is the one thing a freeze gives way to (STRATA_FREEZE_EXPERTS):
                // the cache it leaves is what every request after reads, and stays frozen again.
                engine.unfreeze_experts();
                engine.resize_cache((int) reserve);
                engine.refreeze_experts();
                const long long free_mib = (long long) engine.vram_free_mib();
                std::fprintf(stderr, "strata-poly: VRAM %d MiB kept free: the expert cache %d -> %d of %d experts "
                                     "(%.2f of %.2f GiB), %lld MiB free now, in %.0f ms\n",
                             engine.reserve_mib(), before, engine.cached_experts(), engine.full_experts(),
                             engine.cached_gib(), engine.full_gib(), free_mib, ms_since(t0));
                std::printf("VRAM reserve_mib=%d expert_slots=%d expert_slots_full=%d expert_cache_mib=%lld "
                            "expert_cache_full_mib=%lld vram_free_mib=%lld prompt_chunk=%d\n",
                            engine.reserve_mib(), engine.cached_experts(), engine.full_experts(),
                            (long long) (engine.cached_gib() * 1024.0), (long long) (engine.full_gib() * 1024.0),
                            free_mib, engine.max_rows());
            } catch (const std::exception& e) {
                std::printf("ERR VRAM: %s\n", e.what());
            }
            std::fflush(stdout);
            continue;
        }
        // A refusal is in the session protocol's own words: a bare ERR is a reply out of step there, and the server
        // ended this engine for it (issue 114).
        if (line.rfind("SAVE ", 0) == 0 || line.rfind("RESTORE ", 0) == 0) {
            const bool save = line[0] == 'S';
            if (!session_file(ss, save, line.substr(save ? 5 : 8), !slots.empty())) return 1;
            continue;
        }
        // PIN <path>: the start's --pinned-session, between requests (a server that has just saved the prompt).
        // PINNED <tokens> <bytes> <ms>, or SERR as a session file's
        if (line.rfind("PIN ", 0) == 0) {
            const Clock::time_point t0 = Clock::now();
            bool io = false;
            const std::string why = slots.empty() ? pin_session(ss, line.substr(4), io)
                                                  : "session files do not support --batch (parallel requests)";
            if (why.empty()) {
                std::printf("PINNED %lld %zu %.1f\n", (long long) ss.kept.pinned_tokens(), ss.kept.pinned_bytes(),
                            ms_since(t0));
            } else {
                std::fprintf(stderr, "strata-poly: session pin %s: %s\n", line.c_str() + 4, why.c_str());
                std::printf("SERR %s 0 %s\n", io ? "io" : "invalid", why.c_str());
            }
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("GEN ", 0) != 0 && line.rfind("GENI ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> image_pad=<id> <file> <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        if (!parse_gen(line, max_new, s, pad, turn_at, file, ids)) {
            std::printf("ERR bad request\n");
            std::fflush(stdout);
            continue;
        }
        const long long n = (long long) ids.size();
        if (n + max_new > engine.max_context()) {
            std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%d)\n", n, max_new,
                        engine.max_context());
            std::fflush(stdout);
            continue;
        }
        if (std::any_of(ids.begin(), ids.end(), [&](int32_t t) { return t >= engine.n_vocab(); })) {
            std::printf("ERR a token id is outside the vocabulary\n");
            std::fflush(stdout);
            continue;
        }
        if (!s.bias_text.empty()) {
            std::string err;
            if (!strata::core::parse_logit_bias(s.bias_text, engine.n_vocab(), s.bias, err)) {
                std::printf("ERR logit_bias: %s\n", err.c_str());
                std::fflush(stdout);
                continue;
            }
        }
        std::unique_ptr<strata::poly::Grammar> grammar;   // this request's, where its answer has got to
        if (!s.grammar_text.empty()) {
            std::string text, err;
            if (!ss.grammars()) {
                err = "this model's tokenizer is not one a grammar can be matched against";
            } else if (ss.follow) {
                err = "not with --follow";
            } else if (!slots.empty()) {
                err = "not with --batch";
            } else if (!base64_decode(s.grammar_text, text)) {
                err = "not base64";
            } else {
                grammar = strata::poly::Grammar::parse(text, ss.vocab, err);
            }
            if (!grammar) {
                std::printf("ERR grammar: %s\n", err.c_str());
                std::fflush(stdout);
                continue;
            }
            s.grammar = grammar.get();
        }
        std::string wrong;
        if (file.empty()) pictures = Pictures{};
        else if (!engine.sees()) wrong = "this model does not read pictures";
        else wrong = take_pictures(file, pad, engine.n_embd(), ids, pictures);
        if (!wrong.empty()) {
            std::printf("ERR %s\n", wrong.c_str());
            std::fflush(stdout);
            continue;
        }

        const strata::poly::Stats before = engine.stats();
        int64_t resume = 0;
        Session::Result r;
        int32_t first = -1;   // the token BGEN's read picks
        std::vector<const CoReq*> co_used;
        bool co_ok = false;   // and the BCO prompts went with it: their last tokens go in a step after its BADM
        // A BGEN whose slot holds its whole prompt, read with another's and its last token in the step after that
        // (Slot::first), reads nothing: its first token is picked from what that step left
        const bool ahead = admit >= 0 && !slots[(size_t) admit].first.empty() && slots[(size_t) admit].held == ids &&
                           pictures.at.empty();
        try {
            if (ahead) {
                r.resume = n;
                std::printf("RESUME %lld\n", n);
                if (s.seed != 0) ss.rng.seed(s.seed);
                first = (int32_t) ss.sampler.pick(slots[(size_t) admit].first.data(), engine.n_vocab(), s, ids, ss.rng);
                std::printf("T %d\n", (int) first);
                std::fflush(stdout);
                r.produced = 1;
                r.finish = ss.is_eos(first) ? "stop" : lines.stop.load() ? "cancel" : "length";
            } else {
                engine.bind(0);
                from_slot(ids);
                // The BCO prompts of idle slots go with this read when it is text and they all fit in the read with
                // it.  A slot's sequence starts again for its prompt: what it held is gone (the server gave the slot
                // to them).
                ss.co_read.clear();
                ss.co_done = false;
                int64_t co_rows = n;
                for (const CoReq& c : co_set) {
                    if (admit < 0 || !pictures.at.empty() || !engine.co_reads() || c.slot == admit ||
                        slots[(size_t) c.slot].active || co_rows + (int64_t) c.ids.size() - 1 > engine.max_read() ||
                        (int64_t) c.ids.size() + 1 > engine.max_context() ||
                        std::any_of(co_used.begin(), co_used.end(), [&](const CoReq* u) { return u->slot == c.slot; }))
                        continue;
                    engine.bind(c.slot + 1);
                    engine.images(nullptr, 0);
                    engine.reset();
                    slots[(size_t) c.slot] = Slot{};
                    co_used.push_back(&c);
                    co_rows += (int64_t) c.ids.size() - 1;
                    ss.co_read.push_back({c.slot + 1, c.ids.data(), (int) c.ids.size() - 1});
                }
                engine.bind(0);
                ids_n = n;
                text = pictures.at.empty();
                ss.piece_calls = writing() ? 1 : kPiece;
                piece_t0 = Clock::now();
                const int had = engine.cached_experts();
                if (engine.mind_vram())
                    std::fprintf(stderr, "strata-poly: the expert cache %d -> %d experts (%.2f GiB) for the VRAM free "
                                         "now, %lld MiB left\n",
                                 had, engine.cached_experts(), engine.cached_gib(), (long long) engine.vram_free_mib());
                r = ss.generate(
                    ids, pictures, pad, turn_at, max_new, s, lines.stop,
                    [&](int32_t t) {
                        first = t;
                        std::printf("T %d\n", (int) t);
                        std::fflush(stdout);
                    },
                    [&](int64_t at, int64_t total, double ms, bool first) {
                        if (first) {
                            resume = at;
                            std::printf("RESUME %lld\n", (long long) at);
                        } else {
                            std::printf("PP %lld %lld %.0f %.1f\n", (long long) at, (long long) total, ms,
                                        ms > 0.0 ? 1000.0 * (double) (at - resume) / ms : 0.0);
                        }
                        std::fflush(stdout);
                    },
                    slots.empty() ? nullptr : &between);
                co_ok = ss.co_done;
                ss.co_read.clear();
            }
            if (r.gave_way >= 0) {
                engine.copy_seq(0, gives_to + 1);
                slots[(size_t) gives_to] = Slot{};
                slots[(size_t) gives_to].held = ss.kept.seq();
                std::printf("YIELDED %d %lld\n", gives_to, (long long) r.gave_way);
            }
        } catch (const std::exception& e) {
            if (failed) return 1;   // step() said ERR: the slots' sequences are in no known state
            // the engine's state is unknown now: the conversation it was reading is lost
            ss.kept.drop();
            try { engine.reset(); } catch (const std::exception&) {}
            std::printf("ERR %s\n", e.what());
            std::fflush(stdout);
            continue;
        }
        // STRATA_REQUEST_LINES=1: one stdout line per finished request, like Strata's own engine (the same summary, to
        // the log on stderr so echo_requests picks it up): prompts, cached, fresh tokens read, and both rates.  Any
        // drafts go in the tail (the server reads them from that; DONE below is what it uses for its figures).
        const double pp = r.prompt_ms > 0 ? 1000.0 * r.read / r.prompt_ms : 0.0;
        const double tg = r.decode_ms > 0 ? 1000.0 * r.produced / r.decode_ms : 0.0;
        std::fprintf(stderr, "strata serve: prompt %lld tokens = %lld reused + %lld read in %.0f ms (%.1f tok/s), %lld generated in %.0f ms (%.1f tok/s)",
                     n, (long long) r.resume, (long long) r.read, r.prompt_ms, pp, (long long) r.produced, r.decode_ms, tg);
        if (r.guessed > 0)   // MTP drafts: the ones guessed right of the total guesses made
            std::fprintf(stderr, ", drafts accepted %lld of %lld\n", (long long) r.guessed_right, (long long) r.guessed);
        else
            std::fprintf(stderr, "\n");
        // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused>
        //      <experts from VRAM> <experts> <RAM blobs> <file blobs> <file MB> <prompt tokens read> <offloaded>
        const strata::poly::Stats& now = engine.stats();
        // the cache's share of the experts looked up while answering: a prompt batch's go through VRAM either way
        const long long hits = now.gpu_experts - before.gpu_experts;
        const long long misses = now.cpu_experts - before.cpu_experts;
        std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld 0 0 0.0 %lld 0\n", (long long) r.produced, n,
                    r.prompt_ms, r.decode_ms, r.finish, (long long) r.guessed_right, (long long) r.guessed,
                    (long long) r.resume, hits, hits + misses, (long long) r.read);
        std::fflush(stdout);
        engine.trim();
        ss.kept.release();
        if (admit >= 0) {
            Slot& sl = slots[(size_t) admit];
            const bool goes_on = std::strcmp(r.finish, "length") == 0 && first >= 0 && admit_max > 1 &&
                                 n + 2 <= engine.max_context();
            sl = Slot{};
            try {
                if (!ahead) engine.copy_seq(0, admit + 1);   // (one read ahead is in its slot's sequence already)
                sl.held = ahead ? ids : ss.kept.seq();
            } catch (const std::exception& e) {
                std::fprintf(stderr, "strata-poly: batch: %s\n", e.what());
                std::printf("ERR %s\n", e.what());
                std::fflush(stdout);
                return 1;
            }
            if (goes_on) {
                sl.active = true;
                sl.next = first;
                sl.produced = 1;
                sl.max_new = admit_max;
                sl.s = s;
                if (s.seed != 0) sl.rng.seed(s.seed + 1);
                sl.t0 = Clock::now();
            }
            std::printf("BADM %d %d\n", admit, goes_on ? 1 : 0);
            std::fflush(stdout);
        }
        // The BCO prompts read with this one: their last tokens in one step, so that each slot holds its whole prompt
        // and the logits of its first token (Slot::first).  Each on its own in its BGEN took 14 ms of Qwen3.6 at
        // 12 GB, and the slots waited for all of them.
        if (co_ok && !co_used.empty()) {
            std::vector<int> seqs;
            std::vector<int32_t> last;
            for (const CoReq* c : co_used) {
                seqs.push_back(c->slot + 1);
                last.push_back(c->ids.back());
            }
            try {
                engine.step(seqs.data(), last.data(), (int) seqs.size());
            } catch (const std::exception& e) {
                std::fprintf(stderr, "strata-poly: batch: %s\n", e.what());
                std::printf("ERR %s\n", e.what());
                std::fflush(stdout);
                return 1;
            }
            for (size_t i = 0; i < co_used.size(); ++i) {
                Slot& sl = slots[(size_t) co_used[i]->slot];
                const float* lg = engine.logits_of((int) i);
                sl.held = co_used[i]->ids;
                sl.first.assign(lg, lg + engine.n_vocab());
                std::printf("BCOK %d %zu\n", co_used[i]->slot, sl.held.size());
            }
            std::fflush(stdout);
            co_due = co_used.size();
        }
        if (ms_since(saved) >= kProfileSaveMs) {
            try {
                engine.save_profile();
            } catch (const std::exception& e) {   // the next start's cache is the worse for it, this run is not
                std::fprintf(stderr, "strata-poly: %s\n", e.what());
            }
            saved = Clock::now();
        }
    }
    std::fprintf(stderr, "strata-poly: %lld of %lld prompt tokens were not read again (%.1f%%); %zu conversations "
                         "parked in %zu MiB, %zu pushed out; %lld MiB of VRAM free at the end\n",
                 ss.reused, ss.asked, ss.asked > 0 ? 100.0 * (double) ss.reused / (double) ss.asked : 0.0,
                 ss.kept.parked(), ss.kept.parked_bytes() >> 20, ss.kept.evictions(), (long long) engine.vram_free_mib());
    return 0;
}

// The n most likely next tokens with their probabilities, to hold against another engine's on the same prompt.
void print_top(const float* logits, int n_vocab, int n) {
    std::vector<int> order((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) order[(size_t) i] = i;
    n = std::min(n, n_vocab);
    std::partial_sort(order.begin(), order.begin() + n, order.end(),
                      [logits](int a, int b) { return logits[a] > logits[b]; });
    const float top = logits[order[0]];
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum += std::exp((double) (logits[i] - top));
    std::fprintf(stderr, "top");
    for (int i = 0; i < n; ++i)
        std::fprintf(stderr, " %d:%.6g", order[(size_t) i], std::exp((double) (logits[order[(size_t) i]] - top)) / sum);
    std::fprintf(stderr, "\n");
}

int once(Session& ss, const std::vector<int32_t>& ids, const Pictures& pictures, int32_t pad, long long max_new,
         const Sampling& s, int top_probs) {
    std::atomic<bool> stop{false};
    std::string out;
    ss.reads_logits = top_probs > 0;
    const Session::Result r = ss.generate(
        ids, pictures, pad, -1, max_new, s, stop,
        [&](int32_t t) {
            if (top_probs > 0) print_top(ss.picked_from, ss.engine.n_vocab(), top_probs);
            out += (out.empty() ? "" : ",") + std::to_string(t);
        },
        [](int64_t, int64_t, double, bool) {});
    std::printf("%s\n", out.c_str());
    const strata::poly::Stats& st = ss.engine.stats();
    const double experts = (double) std::max<int64_t>(st.gpu_experts + st.cpu_experts, 1);
    std::fprintf(stderr,
                 "prompt %lld tokens %.1f tok/s, generated %lld tokens %.2f tok/s (%s); experts from VRAM %.1f%%, "
                 "CPU experts %.1f ms/token, router wait %.1f ms/token, %lld experts swapped into VRAM, "
                 "%lld experts uploaded ahead for %lld uses, %lld left out, %lld of %lld guessed tokens right\n",
                 (long long) r.read, r.prompt_ms > 0 ? 1000.0 * (double) r.read / r.prompt_ms : 0.0,
                 (long long) r.produced, r.decode_ms > 0 ? 1000.0 * (double) (r.produced - 1) / r.decode_ms : 0.0,
                 r.finish, 100.0 * (double) st.gpu_experts / experts, st.cpu_ms / (double) std::max<int64_t>(st.tokens, 1),
                 st.route_wait_ms / (double) std::max<int64_t>(st.tokens, 1), (long long) st.swapped_experts,
                 (long long) st.ahead_sent, (long long) st.ahead_used, (long long) st.tail_experts,
                 (long long) r.guessed_right, (long long) r.guessed);
    return 0;
}

// --check-batch K: K prompts, the --tokens one without its first 0, 1, ... K-1 tokens, each answered greedily on
// its own and then all of them together in their own sequences (Engine::step()).  With --check-stagger the
// sequences differ in all but the model: sequence j's prompt is the first n >> 3 (K - 1 - j) tokens of --tokens' n
// (at least 16), it is read once the others have taken j * max_new / (2K) steps, and it writes max_new - j * max_new
// / K tokens, so that the last steps are the first sequence's alone.  Returns 1 when a token differs.
int check_batch(Engine& engine, const std::vector<int32_t>& ids, int k, long long max_new, bool stagger) {
    const auto read = [&](const std::vector<int32_t>& p) {
        engine.reset();
        for (size_t i = 0; i < p.size(); i += (size_t) engine.max_rows()) {
            const int m = (int) std::min(p.size() - i, (size_t) engine.max_rows());
            engine.eval(p.data() + i, m, i + (size_t) m == p.size());
        }
        return (int32_t) strata::poly::most_likely(engine.logits(), engine.n_vocab());
    };
    std::vector<std::vector<int32_t>> want((size_t) k), got((size_t) k), prompt((size_t) k);
    std::vector<long long> wants((size_t) k, max_new), joins((size_t) k, 0);
    for (int j = 0; j < k; ++j) {
        if (!stagger) {
            prompt[(size_t) j].assign(ids.begin() + j, ids.end());
            continue;
        }
        const int shift = std::min(3 * (k - 1 - j), 63);   // (k of 23 or more would shift by 64 or more)
        const size_t n = std::min(ids.size(), std::max<size_t>(16, ids.size() >> shift));
        prompt[(size_t) j].assign(ids.begin(), ids.begin() + (std::ptrdiff_t) n);
        wants[(size_t) j] = max_new - j * max_new / k;
        joins[(size_t) j] = j * max_new / (2 * k);
    }
    std::vector<int32_t> next((size_t) k);
    for (int j = 0; j < k; ++j) {
        engine.bind(0);
        int32_t t = read(prompt[(size_t) j]);
        for (long long n = 0; n < wants[(size_t) j]; ++n) {
            want[(size_t) j].push_back(t);
            engine.eval(&t, 1, true);
            t = (int32_t) strata::poly::most_likely(engine.logits(), engine.n_vocab());
        }
    }
    // a sequence's prompt is read on its own sequence when it joins: before the first step, or between two steps of
    // the others
    double read_ms = 0;
    std::vector<bool> joined((size_t) k, false);
    const auto join = [&](long long step) {
        const Clock::time_point t0 = Clock::now();
        for (int j = 0; j < k; ++j)
            if (!joined[(size_t) j] && joins[(size_t) j] <= step) {
                engine.bind(j + 1);
                next[(size_t) j] = read(prompt[(size_t) j]);
                joined[(size_t) j] = true;
            }
        engine.bind(0);
        read_ms += ms_since(t0);
    };
    join(0);
    const Clock::time_point t1 = Clock::now();
    // --batch-mtp: each sequence guesses the token after its next one, a step checks the guesses, and a sequence
    // that has its max_new tokens leaves the steps
    const bool guessing = engine.steps_guesses();
    long long steps = 0, offered = 0, right = 0;
    std::vector<int> live, kept;
    std::vector<int32_t> feed, guess;
    long long tokens = 0;
    while (true) {
        if (stagger) join(steps);
        live.clear();
        feed.clear();
        guess.clear();
        for (int j = 0; j < k; ++j)
            if (joined[(size_t) j] && (long long) got[(size_t) j].size() < wants[(size_t) j]) {
                got[(size_t) j].push_back(next[(size_t) j]);
                live.push_back(j + 1);
                feed.push_back(next[(size_t) j]);
            }
        if (live.empty()) break;
        if (guessing) {
            guess.resize(live.size());
            engine.draft_many(live.data(), feed.data(), (int) live.size(), guess.data());
            for (size_t i = 0, rows = live.size(); i < live.size(); ++i) {
                if ((long long) got[(size_t) live[i] - 1].size() >= wants[(size_t) live[i] - 1])
                    guess[i] = -1;   // no token wanted after
                if (guess[i] >= 0 && (int) ++rows > engine.step_rows_alone()) guess[i] = -1;   // as the batch loop
                offered += guess[i] >= 0;
            }
            engine.step_picks();
            engine.step(live.data(), feed.data(), (int) live.size(), guess.data());
        } else {
            engine.step_picks();
            engine.step(live.data(), feed.data(), (int) live.size());
        }
        ++steps;
        tokens += (long long) live.size();
        kept.assign(live.size(), 1);
        for (size_t i = 0, row = 0; i < live.size(); ++i) {
            const size_t j = (size_t) live[i] - 1;
            int32_t t = engine.picked((int) row);
            const bool guessed = guessing && guess[i] >= 0;
            if (guessed && t == guess[i]) {
                got[j].push_back(t);
                kept[i] = 2;
                ++right;
                t = engine.picked((int) row + 1);
            }
            next[j] = t;
            row += guessed ? 2 : 1;
        }
        if (guessing) engine.keep_step(kept.data());
    }
    for (int j = 0; j < k; ++j) got[(size_t) j].resize((size_t) wants[(size_t) j]);
    const double step_ms = ms_since(t1);
    if (guessing)
        std::fprintf(stderr, "check-batch: --batch-mtp: %lld steps, %lld of %lld guesses right\n", steps, right,
                     offered);
    int rc = 0;
    for (int j = 0; j < k; ++j) {
        size_t same = 0;
        while (same < want[(size_t) j].size() && want[(size_t) j][same] == got[(size_t) j][same]) ++same;
        std::fprintf(stderr, "check-batch: sequence %d: the first %zu of %lld tokens as on its own\n", j + 1, same,
                     wants[(size_t) j]);
        if (same < want[(size_t) j].size()) rc = 1;
    }
    if (stagger)   // the steps' time with the prompts read between them
        std::fprintf(stderr, "check-batch: %d prompts read in %.0f ms, %lld steps of %lld tokens in all in %.0f ms\n",
                     k, read_ms, steps, tokens, step_ms);
    else
        std::fprintf(stderr, "check-batch: %d prompts read in %.0f ms, %lld steps of %d tokens in %.0f ms (%.1f tok/s "
                             "together)\n",
                     k, read_ms, max_new, k, step_ms, step_ms > 0 ? 1000.0 * (double) (max_new * k) / step_ms : 0.0);
    return rc;
}

void usage() {
#if defined(STRATA_GPU_METAL)
    std::fprintf(stderr, "Metal defaults: optimized expert overlap and compact MMQ (disable: --no-metal-optimize), "
                         "--metal-memory legacy|balanced (default balanced)\n");
#endif
    std::fprintf(stderr,
                 "usage: strata-poly --model FILE.gguf (--serve | --tokens ID,ID,... [--max-new N])\n"
                 "  --max-context N        tokens the attention caches hold (default 8192)\n"
                 "  --threads N            CPU threads for the experts that are not in VRAM (default: 70%% of the usable\n"
                 "                         CPUs; all of them in a container with a CPU quota)\n"
                 "  --vram-reserve-mib N   VRAM left free beside the expert cache, 256 at least (default: a\n"
                 "                         thirty-second of the card, 256 to 768, or a sixteenth, 512 to 1024,\n"
                 "                         when the PC's free RAM is within 4 GiB of the model's files, and then\n"
                 "                         kept whole; else half of it after the first request)\n"
                 "  --vram-pool            claim the card's VRAM once at load (what is free but for the reserve above,\n"
                 "                         which is then the margin left outside), keep it for the whole run and move\n"
                 "                         it between the expert cache and the attention caches as the context grows\n"
                 "                         and shrinks, in 2 MiB pieces; nothing is given back to the card (needs VMM)\n"
                 "  --layer-split S        the model's layers across the CUDA devices this process sees, in their\n"
                 "                         order: S is auto (by each card's free VRAM) or the first layer of each\n"
                 "                         later card, as 20 or 20,30 (default: the first card alone)\n"
                 "  --expert-gpu N         Inkling 2: CUDA device N holds the routed experts the first card has no room\n"
                 "                         for, as many as fit, and computes them for an answer's tokens; the CPU\n"
                 "                         takes the rest (default: none)\n"
                 "  --expert-gpu-reserve-mib N  VRAM that device keeps free beside them (default 512)\n"
                 "  --expert-profile FILE  expert use counts: read to choose the cached experts, written back on\n"
                 "                         exit and by --serve after an answer, every 10 s at most\n"
                 "  --no-gpu-experts       keep every routed expert on the CPU\n"
                 "  --any-node             on a PC with several NUMA nodes, let the CPU threads run on any of them\n"
                 "                         (default: on the node whose memory holds most of the experts)\n"
                 "  --window N             the most tokens that go through the model at once while an answer is\n"
                 "                         written: the one that is known and guesses at the ones after it, which\n"
                 "                         count when they turn out right (1 = no guesses; default 8, of which\n"
                 "                         the block guesses 3, and 4 for GLM 5.3 Flash)\n"
                 "  --lookup N             guess what followed an earlier repeat of the last N or more tokens\n"
                 "                         (default 3, 0 = never)\n"
                 "  --no-draft             do not run the block that guesses ahead\n"
                 "  --draft-model FILE     a file of the same model to take that block from, when --model has none;\n"
                 "                         for Gemma 4, its draft model (a gemma4-assistant file)\n"
                 "  --draft-vocab N        the block guesses among the vocabulary's first N tokens (default:\n"
                 "                         131072 for Qwen3.6, Ornith and Tiel, all of it for the others; 0 = all)\n"
                 "  --no-graphs            launch a token's kernels one by one (default: a layer's are recorded once\n"
                 "                         as a CUDA graph and replayed)\n"
                 "  --kv-whole             give the attention caches VRAM for the whole context at load (default: as\n"
                 "                         the conversation grows, the expert cache having the rest until then)\n"
                 "  --kv-resident N        attention-cache cells kept resident in VRAM, streaming the rest from host\n"
                 "                         RAM (default: 0 = all of them in VRAM; with --kv-whole, fully resident)\n"
                 "  --kv fmt               the K and V attention caches' format, opt-in: fp16 (default,\n"
                 "                         unchanged); q8_0 (ggml Q8_0 blocks: half the KV VRAM, slightly\n"
                 "                         different answers; qwen35moe's, kolibri1's, Flint's and Inkling 2's,\n"
                 "                         which takes it on CUDA and HIP, not Metal); q4_0 (ggml Q4_0\n"
                 "                         blocks: 4-bit KV cache, 47%% smaller Host/PCIe footprint with --kv-resident);\n"
                 "                         int8 / int8k (Gemma 4's:\n"
                 "                         the same Q8_0 rule with the codes and their deltas in two arrays, so a\n"
                 "                         reader takes four codes to a load; int8k keeps the keys' side\n"
                 "                         quantized and the values' in fp16); fp8 (Inkling 2's: the caches in\n"
                 "                         8 bits, about 0.7 GB less VRAM at a 32K context)\n"
#if defined(STRATA_GPU_METAL)
                 "                         Metal default: q8_0 for a compatible Qwen35MoE geometry, else fp16\n"
#endif
                 "  --expert-swaps N       while an answer is written, up to N experts it keeps using are on their way\n"
                 "                         into VRAM, in place of cached ones it uses less (default 64, 0 = the cache\n"
                 "                         stays as it was loaded)\n"
                 "  --ram-experts-mib N    hold N MiB of RAM for the experts outside VRAM, during answers too: the ones\n"
                 "                         used most lately; the rest stay in the file cache for other programs to take,\n"
                 "                         and are read back when a token needs them (default: all of them when they\n"
                 "                         fit in the RAM that is free, else what is free less 4 GiB; -1: all of them)\n"
                 "  --lock-experts         keep the experts outside VRAM in RAM, so the system cannot give it to other\n"
                 "                         programs: copied into pinned memory for the card where the RAM that is free\n"
                 "                         holds them, else their pages in the model file locked (Linux: up to ulimit -l)\n"
                 "  --prompt-batch N       prompt tokens read at a time (default 1024); about 1.4 MiB of VRAM each\n"
                 "  --stage-mib N          VRAM the experts outside the cache pass through in a prompt batch (default\n"
                 "                         128 on an NVIDIA card where a longer read widens it, else 2048; never more\n"
                 "                         than a layer's experts take); with less, more of them are copied after the\n"
                 "                         layer has routed\n"
                 "  --stage-ahead N        from N tokens, a batch copies those experts before it routes (default 1024,\n"
                 "                         0 = never)\n"
                 "  --stage-read-mib N     a prompt read of several batches grows the staging buffer to N MiB (at\n"
                 "                         most a layer's experts) with VRAM the expert cache lends it until the\n"
                 "                         answer starts (default 2048 with the 128 MiB --stage-mib default; 0 = never)\n"
                 "  --cpu-share F          under that, the share of those experts the CPU computes meanwhile, the\n"
                 "                         ones the fewest tokens chose (default: as the times say; 0 = none)\n"
                 "  --prompt-cache N       checkpoints kept of a conversation, about 150 MB of RAM each (default 6;\n"
                 "                         0 = a prompt is read from its start unless it goes on from the last token)\n"
                 "  --prompt-cache-every N a checkpoint every N prompt tokens, besides the one at a prompt's end\n"
                 "                         (default 2048, 0 = none)\n"
                 "  --conversation-cache-mib N   RAM for parked conversations, so that several can take turns\n"
                 "                         without being read again (default 0 = off; auto: 40%% of the RAM free\n"
                 "                         once loaded, above the floor below, at most 8192)\n"
                 "  --conversation-cache-slots N        parked conversations at most (default 4)\n"
                 "  --conversation-cache-min-tokens N   a conversation shorter than this is not parked (default 1024)\n"
                 "  --conversation-cache-min-free-mib N park only while this much physical RAM stays free (default\n"
                 "                         2560)\n"
                 "  --pinned-session FILE  --serve: a session file (SAVE) whose conversation stays for the whole run,\n"
                 "                         a system prompt read once: a prompt that starts with it goes on from\n"
                 "                         its end, whatever took the engine since; it is never pushed out\n"
                 "  --batch N              N answers written together, each in a sequence of its own with its own\n"
                 "                         state and attention rows (default 0 = one at a time; at most 16);\n"
                 "                         --slots N is the same\n"
                 "  --batch-mtp            with --batch and a draft block: each answer written together guesses its\n"
                 "                         next token and the step checks it, a guess right gives two tokens\n"
                 "                         (Qwen3.6's family; STRATA_BATCH_MTP=1 does the same)\n"
                 "  --check-batch K        with --tokens and --batch K or more: K prompts answered on their own and\n"
                 "                         then together, compared token by token (takes --kv-whole and\n"
                 "                         --cpu-share 0 and freezes the experts' split, so the expert cache stays as\n"
                 "                         it is)\n"
                 "  --check-stagger        with --check-batch: prompts of different lengths (the first n >> 3 (K-1-j)\n"
                 "                         tokens), read while the others write, and answers of different lengths\n"
                 "  --eos-ids ID,ID,...    tokens that end a reply (default: the GGUF's eos, eot and eom)\n"
                 "  --temperature F --top-k N --top-p F --min-p F --seed N   sampling for --tokens (default greedy)\n"
                 "  --images FILE          with --tokens: the prompt's pictures, as strata-vision writes them\n"
                 "  --image-pad ID         the token whose runs in --tokens are the pictures' places, a token a row\n"
                 "  --dump-layers          print a checksum of every layer's stages to stderr\n"
                 "  --top-probs N          with --tokens: print the N most likely ids of every step to stderr\n"
                 "  --follow ID,ID,...     with --tokens: write these tokens in place of the ones picked, for\n"
                 "                         --top-probs of the same answer under other settings (--tokens, --follow\n"
                 "                         and --eos-ids also take @FILE: the list in that file)\n"
                 "  --tail-skip N          an expert outside VRAM that every token of a window ranks at place N+1\n"
                 "                         or later among its experts is left out: faster, and the answers move a\n"
                 "                         little (default 0 = none is; setup gives Qwen3.6 6)\n"
                 "  --route-top N          a token's experts from place N+1 on are left out, in VRAM or not, where\n"
                 "                         the host routes (default 0 = none is)\n"
                 "  --attn-requant TYPE    Inkling 2: its attention's matrices quantized again to TYPE (q5_0, q4_0\n"
                 "                         or iq4_nl) at load: fewer bytes a token, and the answers move a little\n"
                 "                         (default: the file's types; setup gives an 8 GB card q5_0); off or none\n"
                 "                         says the default (the last --attn-requant counts)\n"
                 "  --loop-bf16            Inkling 2: its loop controllers' matrices in BF16, not F32;\n"
                 "                         --no-loop-bf16 says the default (F32; the last one counts)\n"
                 "  --prompt-int8          Inkling 2 on an RTX 20 (Turing): its prompt products on the int8 tensor\n"
                 "                         cores as on RTX 30 and newer, not as halves and cuBLAS (the default on\n"
                 "                         Turing): faster prompts, and the answers move more, activations int8 a\n"
                 "                         32 values; --no-prompt-int8 says the default; STRATA_PF_I8MMA=0 turns the\n"
                 "                         int8 kernel off on every card; with it, a prompt batch's experts in VRAM\n"
                 "                         go in one launch a class of rows, the same numbers in fewer launches\n"
                 "                         (STRATA_POLY_EXPERT_TILES=0: a launch a run of slots again)\n"
                 "  --skip-weight F        in place of --tail-skip: an expert outside VRAM that every token of a\n"
                 "                         window weighs under F of its experts' weight is left out (default 0 = off)\n"
                 "  --cache-bias F         qwen35moe and Inkling 2: a token's router scores (Inkling 2: its gate\n"
                 "                         plus bias) get F added for each expert in VRAM before it picks, as it\n"
                 "                         comes a few at a time; the weights stay the router's own: fewer\n"
                 "                         experts on the CPU, and the answers move (default 0 = off)\n"
                 "environment: STRATA_ROUTE_TAIL_SKIP=N  in place of --tail-skip (0 turns it off)\n"
                 "environment: STRATA_FREEZE_EXPERTS=1  hold the split of the routed experts between the card\n"
                 "                         and the CPU still, so the same request gives the same greedy answer\n"
                 "                         bit for bit; with guesses on, add --window 1 for a repeat everywhere\n");
}

}  // namespace

// Windows holds the threads of a process it takes for background work to the efficiency cores (power throttling):
// on a laptop with both kinds an answer after a long prompt came at 39 tokens/s where 58.  serve/winjob.py takes the
// engines the server starts out of it; this does the same for an engine anything else starts.
void no_power_throttling() {
#ifdef _WIN32
    for (const ULONG speed_and_timer : {0x1ul | 0x4ul, 0x1ul}) {   // Windows 10 knows the first bit only
        PROCESS_POWER_THROTTLING_STATE off{PROCESS_POWER_THROTTLING_CURRENT_VERSION, speed_and_timer, 0};
        if (SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &off, sizeof(off))) return;
    }
#endif
}

// An exception that nothing catches (on a thread other than main()'s, whose catch says what it was) ends the process
// through std::terminate, which writes nothing: the log of such an engine just stops.  The default handler aborts too.
void on_terminate() {
    std::string why = "std::terminate() without an exception";
    if (const std::exception_ptr e = std::current_exception()) {
        why = "an exception that nothing caught";
        try {
            std::rethrow_exception(e);
        } catch (const std::exception& x) {
            why += ": ";
            why += x.what();
        } catch (...) {
        }
    }
    std::fprintf(stderr, "strata-poly: %s; the engine ends\n", why.c_str());
    std::fflush(stderr);
    std::abort();
}

int main(int argc, char** argv) {
    std::set_terminate(on_terminate);
    no_power_throttling();
#if !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP)
    // The cached experts' warp kernel and 32 of them a launch (native_mmvq_slots_wide()), the HIP build's default,
    // here too unless STRATA_SLOTS_WARP says otherwise: the same bits, and on lab-pro6000 (RTX PRO 6000) Qwen3.6
    // UD-Q3_K_M wrote 380-391 tokens/s where 353-360 with every expert in VRAM, 267-269 where 232-238 held to 12 GB.
    if (!std::getenv("STRATA_SLOTS_WARP")) {
#ifdef _WIN32
        _putenv_s("STRATA_SLOTS_WARP", "1");
#else
        setenv("STRATA_SLOTS_WARP", "1", 0);
#endif
    }
#endif
    Options o;
    CacheFlags cache;
    Sampling s;
    bool want_serve = false;
    long long max_new = 32;
    int top_probs = 0, lookup = 3, image_pad = -1, check = 0;
    bool stagger = false;   // --check-stagger
    std::string images, pinned;
    std::vector<int32_t> ids, eos, follow;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-poly: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") o.model = val();
        else if (a == "--serve") want_serve = true;
#if defined(STRATA_GPU_METAL)
        else if (a == "--metal-optimize") o.metal_optimize = true;
        else if (a == "--no-metal-optimize") o.metal_optimize = false;
        else if (a == "--metal-memory") {
            const std::string policy = val();
            if (policy != "legacy" && policy != "balanced") {
                std::fprintf(stderr, "strata-poly: --metal-memory requires legacy or balanced\n");
                return 2;
            }
            o.metal_balanced = policy == "balanced";
        }
#endif
        else if (a == "--max-context") o.max_context = std::atoi(val());
        else if (a == "--threads") o.threads = std::atoi(val());
        else if (a == "--vram-reserve-mib") o.vram_reserve_mib = std::atoi(val());
        else if (a == "--vram-pool") o.vram_pool = true;
        else if (a == "--expert-profile") o.profile = val();
        else if (a == "--pinned-session") pinned = val();
        else if (a == "--no-gpu-experts") o.gpu_experts = false;
        else if (a == "--any-node") o.stay_near = false;
        else if (a == "--lock-experts") o.lock_experts = true;
        else if (a == "--no-graphs") o.graphs = false;
        else if (a == "--kv-whole") o.kv_grow = false;
        else if (a == "--kv-resident") o.kv_resident = std::atoi(val());
        else if (a == "--kv") {
            const std::string fmt = val();
            if (fmt == "fp16") o.kv_format = KvFmt::kFp16;
            else if (fmt == "q8_0") o.kv_format = KvFmt::kQ8_0;
            else if (fmt == "q4_0") o.kv_format = KvFmt::kQ4_0;
            else if (fmt == "int8") o.kv_format = KvFmt::kInt8;
            else if (fmt == "int8k") o.kv_format = KvFmt::kInt8K;
            else if (fmt == "fp8") o.kv_format = KvFmt::kFp8;
            else {
                std::fprintf(stderr, "strata-poly: --kv %s: unknown format (use fp16, q8_0, q4_0, int8, int8k or fp8)\n",
                             fmt.c_str());
                std::exit(2);
            }
        }
        else if (a == "--window") o.window = std::atoi(val());
        else if (a == "--lookup") lookup = std::atoi(val());
        else if (a == "--no-draft") o.draft = false;
        else if (a == "--draft-model") o.draft_model = val();
        else if (a == "--expert-swaps") o.expert_swaps = std::atoi(val());
        else if (a == "--ram-experts-mib") o.ram_experts_mib = std::atoll(val());
        else if (a == "--tail-skip") o.tail_skip = std::atoi(val());
        else if (a == "--route-top") o.route_top = std::atoi(val());
        else if (a == "--attn-requant") {
            // off / none: the file's types, so a flag appended to a run config can undo setup's q5_0
            std::string t = val(), low = t;
            for (char& c : low) c = (char) std::tolower((unsigned char) c);
            o.attn_requant = (low == "off" || low == "none") ? std::string() : t;
        }
        else if (a == "--loop-bf16") o.loop_bf16 = true;
        else if (a == "--no-loop-bf16") o.loop_bf16 = false;
        else if (a == "--prompt-int8") o.prompt_int8 = 1;
        else if (a == "--no-prompt-int8") o.prompt_int8 = 0;
        else if (a == "--skip-weight") o.skip_weight = (float) std::atof(val());
        else if (a == "--cache-bias") o.cache_bias = (float) std::atof(val());
        else if (a == "--layer-split") o.layer_split = val();
        else if (a == "--expert-gpu") o.expert_gpu = std::atoi(val());
        else if (a == "--expert-gpu-reserve-mib") o.expert_gpu_reserve_mib = std::atoi(val());
        else if (a == "--batch" || a == "--slots") o.batch = std::atoi(val());
        else if (a == "--batch-mtp") o.batch_mtp = true;
        else if (a == "--check-batch") check = std::atoi(val());
        else if (a == "--check-stagger") stagger = true;
        else if (a == "--draft-vocab") o.draft_vocab = std::atoi(val());
        else if (a == "--prompt-batch") o.prompt_batch = std::atoi(val());
        else if (a == "--stage-mib") o.stage_mib = std::atoi(val());
        else if (a == "--stage-ahead") o.stage_ahead = std::atoi(val());
        else if (a == "--stage-read-mib") o.stage_read_mib = std::atoi(val());
        else if (a == "--cpu-share") o.cpu_share = (float) std::atof(val());
        else if (a == "--prompt-cache") cache.o.checkpoints = std::atoi(val());
        else if (a == "--prompt-cache-every") cache.o.every = std::atoll(val());
        else if (a == "--conversation-cache-mib") {
            const std::string v = val();
            cache.park_auto = v == "auto";
            cache.o.park_mib = cache.park_auto ? 0 : std::atoll(v.c_str());
        }
        else if (a == "--conversation-cache-slots") cache.o.park_slots = std::atoi(val());
        else if (a == "--conversation-cache-min-tokens") cache.o.park_min = std::atoll(val());
        else if (a == "--conversation-cache-min-free-mib") cache.min_free_mib = std::atoll(val());
        else if (a == "--dump-layers") o.dump = true;
        else if (a == "--images") images = val();
        else if (a == "--image-pad") image_pad = std::atoi(val());
        else if (a == "--top-probs") top_probs = std::atoi(val());
        else if (a == "--max-new") max_new = std::atoll(val());
        else if (a == "--temperature") s.temperature = std::strtof(val(), nullptr);
        else if (a == "--top-k") s.top_k = std::atoi(val());
        else if (a == "--top-p") s.top_p = std::strtof(val(), nullptr);
        else if (a == "--min-p") s.min_p = std::strtof(val(), nullptr);
        else if (a == "--seed") s.seed = std::strtoull(val(), nullptr, 10);
        else if (a == "--tokens" || a == "--eos-ids" || a == "--follow") {
            // @FILE: the list is that file's text (a long prompt's list outgrows Windows' 32K command line)
            std::string v = val();
            if (v.size() > 1 && v[0] == '@') {
                std::ifstream f(v.substr(1));
                v.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
                while (!v.empty() && (v.back() == '\n' || v.back() == '\r' || v.back() == ' ')) v.pop_back();
            }
            if (!parse_ids(v.c_str(), a == "--tokens" ? ids : a == "--eos-ids" ? eos : follow)) {
                std::fprintf(stderr, "strata-poly: %s takes token ids separated by commas\n", a.c_str());
                return 2;
            }
        } else {
            if (a != "--help" && a != "-h") std::fprintf(stderr, "strata-poly: unknown option %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (o.model.empty() || (!want_serve && ids.empty()) || max_new < 1) {
        usage();
        return 2;
    }
    if (o.batch < 0 || o.batch > 16 || check < 0 || check > o.batch || (check > 0 && (int) ids.size() <= check)) {
        std::fprintf(stderr, "strata-poly: --batch takes 0 to 16, and --check-batch at most --batch prompts, each "
                             "with tokens left of --tokens\n");
        return 2;
    }
    if (const char* v = std::getenv("STRATA_BATCH_MTP"); v && std::string(v) == "1") o.batch_mtp = true;
    // --check-batch: K sequences growing together would take VRAM from the expert cache that the single runs had,
    // and an expert moved to the CPU rounds differently (issue 55).  Their attention rows get VRAM at load instead.
    // The cache moves between prompts too, --expert-swaps 0 or not: it takes a prompt's buffers between reads
    // (Engine::lend_prompt()), and a long read takes them and a wider staging buffer back.  So the split is frozen
    // for the run, and a prompt's experts go to the CPU by no share of the time it took (--cpu-share 0): with them,
    // Qwen3.6 UD-Q3_K_M at 12 GiB wrote 1 to 47 of 64 tokens as alone, poly.10 64 of 64; frozen, 64 of 64.
    if (check > 0) {
        o.kv_grow = false;
        o.freeze_experts = true;
        o.cpu_share = 0;
    }
    if (!o.kv_grow && o.kv_resident > 0) {
        std::fprintf(stderr, "strata-poly: --kv-whole takes precedence over --kv-resident (caches fully resident in VRAM)\n");
        o.kv_resident = 0;
    }
    if (o.batch > 0 && o.kv_resident > 0) {
        std::fprintf(stderr, "strata-poly: --batch takes precedence over --kv-resident (caches resident in VRAM)\n");
        o.kv_resident = 0;
    }
    // over --tail-skip, as Strata's engine reads it: =0 turns a model's default off
    if (const char* v = std::getenv("STRATA_ROUTE_TAIL_SKIP")) o.tail_skip = std::atoi(v);
    const long long most = 1LL << 24;   // MiB: 16 TiB
    if (cache.o.checkpoints < 0 || cache.o.every < 0 || cache.o.park_mib < 0 || cache.o.park_mib > most ||
        cache.o.park_slots < 0 || cache.o.park_slots > 1024 || cache.o.park_min < 0 || cache.min_free_mib < 0 ||
        cache.min_free_mib > most) {
        std::fprintf(stderr, "strata-poly: a cache option is negative or too large\n");
        return 2;
    }

    std::unique_ptr<Engine> loaded;
    try {
        const Clock::time_point t0 = Clock::now();
        loaded = Engine::open(o);
        Engine& engine = *loaded;
        std::fprintf(stderr, "strata-poly: loaded in %.1f s; %d of %d routed experts in VRAM (%.1f GiB), %d CPU threads\n",
                     ms_since(t0) / 1000.0, engine.cached_experts(), engine.total_experts(), engine.cached_gib(),
                     engine.threads());
        if (engine.ram_experts_mib() >= 0)
            std::fprintf(stderr, "strata-poly: %lld MiB of RAM for the %.1f GiB of experts outside VRAM; the others "
                                 "come from the file cache when a token needs them (--ram-experts-mib, -1 for all)\n",
                         (long long) engine.ram_experts_mib(), engine.outside_gib());
        if (o.cache_bias > 0)
            std::fprintf(stderr, "strata-poly: cache bias %.2f: a token that comes a few at a time routes with %.2f "
                                 "added to the score of each expert in VRAM; the weights stay the router's own\n",
                         o.cache_bias, o.cache_bias);
        if (o.skip_weight > 0)
            std::fprintf(stderr, "strata-poly: skip weight %.3f: an expert outside VRAM that every token of a window "
                                 "weighs under %.1f%% of its experts' weight is left out (in place of --tail-skip)\n",
                         o.skip_weight, 100.0 * o.skip_weight);
        else if (o.tail_skip > 0)
            std::fprintf(stderr, "strata-poly: tail skip %d: an expert outside VRAM that every token of a window ranks "
                                 "at place %d or later among its experts is left out; STRATA_ROUTE_TAIL_SKIP=0 turns "
                                 "it off\n",
                         o.tail_skip, o.tail_skip + 1);
        // Pages on another node than the threads are read at about two thirds of the speed.  The file cache puts a
        // page on the node of whatever read it first, so a file read back by something else lands on any.
        const bool far = engine.numa_share() > 0.0 && engine.numa_share() < 0.95;
        if (engine.numa_node() >= 0)
            std::fprintf(stderr, "strata-poly: the CPU threads stay on NUMA node %d, which has %.0f%% of the experts' "
                                 "pages%s\n", engine.numa_node(), 100.0 * engine.numa_share(),
                         far ? "; the rest is read from another node, more slowly" : "");
        else if (far)
            std::fprintf(stderr, "strata-poly: the experts' pages are on several NUMA nodes, %.0f%% on the fullest; "
                                 "the CPU reads the ones far from it more slowly\n", 100.0 * engine.numa_share());
        if (eos.empty())
            for (const char* key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                                    "tokenizer.ggml.eom_token_id"}) {
                const int64_t id = engine.meta_u(key, -1);
                if (id >= 0 && std::find(eos.begin(), eos.end(), (int32_t) id) == eos.end()) eos.push_back((int32_t) id);
            }
        // A fixed 8 GiB on a PC whose RAM the model's file already fills pushed that file out of the file cache.
        // Sized as Strata's engine sizes it (#3493); admit() still keeps min_free_mib free at every park.
        if (cache.park_auto && want_serve && cache.o.checkpoints > 0 && cache.o.park_slots > 0) {
            std::optional<uint64_t> free_b = strata::core::available_host_bytes();
#ifdef _WIN32
            // the card's experts' pages, let go to the standby list, as the RAM budget counts them (Engine::load())
            if (free_b) free_b = *free_b - std::min<uint64_t>(*free_b, engine.cached_file_bytes());
#endif
            if (free_b) free_b = *free_b - std::min<uint64_t>(*free_b, engine.host_growth_bytes());
            cache.o.park_mib = strata::core::auto_conversation_cache_mib(free_b, cache.min_free_mib);
            if (cache.o.park_mib > 0)
                std::fprintf(stderr, "strata-poly: conversation cache sized from the free RAM: %lld MiB for up to %d "
                                     "parked conversations (--conversation-cache-mib N sets it, 0 turns it off)\n",
                             (long long) cache.o.park_mib, cache.o.park_slots);
            else
                std::fprintf(stderr, "strata-poly: conversation cache off: not enough free RAM to park conversations "
                                     "(--conversation-cache-mib N asks for one anyway)\n");
        }
        Session ss(engine, eos, cache, lookup);
        ss.pinned = pinned;
        int rc = 0;
        if (want_serve) {
            rc = serve(ss);
        } else if ((long long) ids.size() + max_new > engine.max_context()) {
            throw std::runtime_error("the prompt and --max-new together exceed --max-context");
        } else if (check > 0 && check >= engine.sequences()) {
            // the engine dropped --batch (a model that runs one sequence with these options says why)
            std::fprintf(stderr, "strata-poly: --check-batch %d: the engine runs %d sequences beside the first\n", check,
                         engine.sequences() - 1);
            rc = 2;
        } else if (check > 0) {
            rc = check_batch(engine, ids, check, max_new, stagger);
        } else {
            Pictures pictures;
            if (!images.empty()) {
                if (!engine.sees()) throw std::runtime_error("this model does not read pictures");
                if (image_pad < 0) throw std::runtime_error("--images needs --image-pad");
                const std::string wrong = take_pictures(images, image_pad, engine.n_embd(), ids, pictures);
                if (!wrong.empty()) throw std::runtime_error(wrong);
            }
            if (!follow.empty()) ss.follow = &follow;
            rc = once(ss, ids, pictures, image_pad, max_new, s, top_probs);
        }
        engine.save_profile();
        // with the catch below and on_terminate(), every way out the engine takes itself leaves a last line: a log
        // that stops without one is of a process that crashed or that something else ended (#195)
        if (want_serve) std::fprintf(stderr, "strata-poly: the engine ends (exit code %d)\n", rc);
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-poly: %s\n", e.what());
        if (want_serve) {
            std::printf("ERR %s\n", e.what());
            std::fflush(stdout);
        }
        return 1;
    }
}
