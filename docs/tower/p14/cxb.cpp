// cxb.cpp - Pass 13's microbenchmark: a decode token's CPU expert work (40 MoE layers, 8 experts a token each),
// through the engine's own cpu_experts() (src/poly/cpu_experts.cpp, linked in), on the real GGUF's expert bytes.
//
//   cxb TAB GGUF MODE ROWS TOKENS POOL [THREADS]
//   MODE  mmap    rows straight from the file's mapping (page cache, 4 KiB pages): what the engine does today
//         anon4k  the pool copied into anonymous memory, MADV_NOHUGEPAGE
//         anon2m  the same, MADV_HUGEPAGE (A)
//         repack  anon2m, with Q4_K gate and up rows interleaved 8 at a time (block_q4_Kx8) and computed with
//                 ggml_gemv_q4_K_8x8_q8_K (C); down as today
//         trace   (Pass 14) cxb TAB GGUF trace TRACE REPEATS: replays the engine's recorded CPU calls (STRATA_CPU_TRACE:
//                 per call its layer, tokens and experts) from the file's mapping, REPEATS times, timed by call
//                 shape
//   ROWS  tokens a call (decode windows: 1, or 2-4 with MTP guesses); POOL experts a layer that tokens draw from
//
// Built twice, with the tower branch's cpu_experts.cpp and with PR 36's (B), by cxb.sh.
#include "cpu_experts.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

extern "C" void ggml_gemv_q4_K_8x8_q8_K(int n, float* s, size_t bs, const void* vx, const void* vy, int nr, int nc);

using namespace strata::poly;

namespace {

struct Layer {
    int gu_type, d_type;
    size_t gate_off, up_off, down_off, gu_stride, d_stride;
    std::vector<const uint8_t*> gate, up, down;   // the pool's experts
};

// From llama.cpp's ggml-cpu/repack.cpp (MIT): make_block_q4_Kx8 with an 8-byte interleave
struct BlockQ4K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct BlockQ4Kx8 { uint16_t d[8], dmin[8]; uint8_t scales[96]; uint8_t qs[1024]; };
static_assert(sizeof(BlockQ4K) == 144 && sizeof(BlockQ4Kx8) == 8 * 144, "block sizes");

BlockQ4Kx8 make_q4_Kx8(const BlockQ4K* in) {
    BlockQ4Kx8 out;
    for (int i = 0; i < 8; i++) out.d[i] = in[i].d, out.dmin[i] = in[i].dmin;
    for (int i = 0; i < 128; ++i) memcpy(&out.qs[i * 8], &in[i % 8].qs[(i / 8) * 8], 8);
    uint8_t s[8], m[8];
    for (int half = 0; half < 2; ++half)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                if (half == 0) {
                    s[j] = in[j].scales[i] & 63;
                    m[j] = in[j].scales[i + 4] & 63;
                } else {
                    s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15);
                    m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
                }
            }
            uint8_t* o = out.scales + half * 48 + i * 12;
            for (int k = 0; k < 4; ++k) {
                o[k] = (s[k] & 63) + ((s[k + 4] & 48) << 2);
                o[k + 4] = (m[k] & 63) + ((m[k + 4] & 48) << 2);
                o[k + 8] = (s[k + 4] & 15) + ((m[k + 4] & 15) << 4);
            }
        }
    return out;
}

// One expert matrix of n_rows rows, in place
void repack_q4_K(uint8_t* p, int n_rows, int n_cols) {
    const int nb = n_cols / 256;
    std::vector<BlockQ4K> src((size_t) 8 * nb);
    BlockQ4K tmp[8];
    for (int r = 0; r < n_rows; r += 8) {
        BlockQ4K* g = (BlockQ4K*) (p + (size_t) r * nb * 144);
        memcpy(src.data(), g, src.size() * 144);
        for (int x = 0; x < nb; ++x) {
            for (int i = 0; i < 8; ++i) tmp[i] = src[(size_t) x + (size_t) i * nb];
            ((BlockQ4Kx8*) g)[x] = make_q4_Kx8(tmp);
        }
    }
}

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

// cpu_experts() with gate and up from repacked Q4_K: tasks of 32 rows (4 groups of 8), each use's 32 rows of gate
// and of up in one gemv call.  Down is today's loop.
void cpu_experts_repack(Workers& workers, const CpuExpert* experts, int n, int rows, int gu_type, int d_type,
                        int n_embd, int n_ff, const float* x, float limit, float* out) {
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    const size_t gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    const size_t d_row = ggml_row_size((ggml_type) d_type, n_ff);
    const size_t a_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    const size_t h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    static std::vector<uint8_t> act, hq;
    static std::vector<float> h;
    static std::vector<int> by, run;
    act.resize((size_t) rows * a_bytes);
    hq.resize((size_t) n * h_bytes);
    h.resize((size_t) n * n_ff);
    uint8_t* actp = act.data();
    const auto act_from = traits(tg->vec_dot_type)->from_float;
    workers.run(rows, [&](int t) { act_from(x + (size_t) t * n_embd, actp + (size_t) t * a_bytes, n_embd); });
    by.resize((size_t) n);
    std::iota(by.begin(), by.end(), 0);
    std::stable_sort(by.begin(), by.end(), [&](int a, int b) { return experts[a].gate < experts[b].gate; });
    run.clear();
    for (int i = 0; i < n; ++i)
        if (i == 0 || experts[by[i]].gate != experts[by[i - 1]].gate) run.push_back(i);
    const int n_run = (int) run.size();
    run.push_back(n);
    float* hp = h.data();
    const int* byp = by.data();
    const int* runp = run.data();
    constexpr int kRows = 32;
    const int gu_tasks = n_ff / kRows;
    workers.run(n_run * gu_tasks, [&](int t) {
        const int i0 = runp[t / gu_tasks], i1 = runp[t / gu_tasks + 1];
        const CpuExpert& e = experts[byp[i0]];
        const int r0 = (t % gu_tasks) * kRows;
        float g[kRows], u[kRows];
        for (int i = i0; i < i1; ++i) {
            const int j = byp[i];
            const uint8_t* a = actp + (size_t) experts[j].row * a_bytes;
            ggml_gemv_q4_K_8x8_q8_K(n_embd, g, 0, e.gate + (size_t) r0 * gu_row, a, 1, kRows);
            ggml_gemv_q4_K_8x8_q8_K(n_embd, u, 0, e.up + (size_t) r0 * gu_row, a, 1, kRows);
            for (int k = 0; k < kRows; ++k) {
                const float gg = std::min(g[k], limit), uu = std::clamp(u[k], -limit, limit);
                hp[(size_t) j * n_ff + r0 + k] = gg / (1.f + std::exp(-gg)) * uu;
            }
        }
    });
    uint8_t* hqp = hq.data();
    const auto hq_from = traits(td->vec_dot_type)->from_float;
    workers.run(n, [&](int j) { hq_from(hp + (size_t) j * n_ff, hqp + (size_t) j * h_bytes, n_ff); });
    constexpr int kDown = 16;
    workers.run(n_embd / kDown, [&](int t) {
        static thread_local std::vector<float> one;
        one.resize((size_t) n);
        float* s = one.data();
        for (int r = t * kDown; r < (t + 1) * kDown; ++r) {
            for (int q = 0; q < n_run; ++q) {
                const uint8_t* down = experts[byp[runp[q]]].down + (size_t) r * d_row;
                for (int i = runp[q]; i < runp[q + 1]; ++i)
                    td->vec_dot(n_ff, s + byp[i], 0, down, 0, hqp + (size_t) byp[i] * h_bytes, 0, 1);
            }
            int j = 0;
            for (int row = 0; row < rows; ++row) {
                float acc = 0.f;
                for (; j < n && experts[j].row == row; ++j) acc += experts[j].weight * s[j];
                out[(size_t) row * n_embd + r] = acc;
            }
        }
    });
}

long anon_huge_kb() {
    std::ifstream f("/proc/self/smaps_rollup");
    std::string k;
    long v = 0;
    while (f >> k)
        if (k == "AnonHugePages:") {
            f >> v;
            return v;
        }
    return -1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "cxb TAB GGUF MODE ROWS TOKENS POOL [THREADS]\n");
        return 2;
    }
    const std::string mode = argv[3];
    const int rows = atoi(argv[4]), tokens = atoi(argv[5]), pool = atoi(argv[6]);
    const int threads = argc > 7 ? atoi(argv[7]) : 12;
    const int n_embd = 2048, n_ff = 512, n_used = 8, n_layers = 40, n_exp = 256;
    ggml_cpu_init();

    std::vector<Layer> L;
    {
        std::ifstream f(argv[1]);
        int l;
        Layer y;
        while (f >> l >> y.gu_type >> y.d_type >> y.gate_off >> y.up_off >> y.down_off >> y.gu_stride >> y.d_stride)
            if (l < n_layers) L.push_back(y);
    }
    const int fd = open(argv[2], O_RDONLY);
    const size_t fsize = lseek(fd, 0, SEEK_END);
    const uint8_t* file = (const uint8_t*) mmap(nullptr, fsize, PROT_READ, MAP_SHARED, fd, 0);

    if (mode == "trace") {
        struct Call { int il, m; std::vector<std::pair<int, int>> uses; double engine_us; };
        std::vector<Call> calls;
        {
            std::ifstream f(argv[4]);
            std::string line;
            while (std::getline(f, line)) {
                Call c;
                int nh, launch;
                char* p = line.data();
                c.il = (int) strtol(p, &p, 10);
                c.m = (int) strtol(p, &p, 10);
                nh = (int) strtol(p, &p, 10);
                c.engine_us = strtod(p, &p);
                launch = (int) strtol(p, &p, 10);
                (void) launch;
                for (int i = 0; i < nh; ++i) {
                    const int e = (int) strtol(p, &p, 10);
                    const int r = (int) strtol(p + 1, &p, 10);
                    c.uses.push_back({e, r});
                }
                if (c.il >= 0 && c.il < (int) L.size() && nh > 0) calls.push_back(std::move(c));
            }
        }
        const int repeats = atoi(argv[5]);
        long sink = 0;
        std::vector<char> seen(L.size() * n_exp, 0);
        for (const Call& c : calls)
            for (auto [e, r] : c.uses) {
                if (seen[(size_t) c.il * n_exp + e]++) continue;
                const Layer& y = L[c.il];
                for (const uint8_t* q : {file + y.gate_off + (size_t) e * y.gu_stride, file + y.up_off + (size_t) e * y.gu_stride})
                    for (size_t i = 0; i < y.gu_stride; i += 4096) sink += q[i];
                for (size_t i = 0; i < y.d_stride; i += 4096) sink += file[y.down_off + (size_t) e * y.d_stride + i];
            }
        Workers workers(threads);
        std::vector<float> x((size_t) 8 * n_embd), out((size_t) 8 * n_embd);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::mt19937 rng(1234);
        for (float& v : x) v = nd(rng);
        // buckets: 1 token a call; then by distinct experts, 1-8, 9-16, 17+
        const char* names[4] = {"1 token", "runs 1-8", "runs 9-16", "runs 17+"};
        double ms[4] = {0, 0, 0, 0}, eng[4] = {0, 0, 0, 0};
        long count[4] = {0, 0, 0, 0};
        uint64_t hash = 1469598103934665603ull;
        std::vector<CpuExpert> ex;
        for (int rep = 0; rep < repeats + 1; ++rep)   // the first pass is a warm-up
            for (const Call& c : calls) {
                const Layer& y = L[c.il];
                ex.clear();
                std::vector<int> es;
                for (auto [e, r] : c.uses) {
                    ex.push_back({file + y.gate_off + (size_t) e * y.gu_stride, file + y.up_off + (size_t) e * y.gu_stride,
                                  file + y.down_off + (size_t) e * y.d_stride, 1.f / n_used, r});
                    es.push_back(e);
                }
                std::sort(es.begin(), es.end());
                const int runs = (int) (std::unique(es.begin(), es.end()) - es.begin());
                const auto t0 = std::chrono::steady_clock::now();
                cpu_experts(workers, ex.data(), (int) ex.size(), c.m, y.gu_type, y.d_type, n_embd, n_ff, x.data(), INFINITY,
                            false, out.data());
                const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (rep == 0) continue;
                const int b = c.m == 1 ? 0 : runs <= 8 ? 1 : runs <= 16 ? 2 : 3;
                ms[b] += dt / repeats;
                eng[b] += c.engine_us / 1000.0 / repeats;
                count[b] += 1;
                for (int i = 0; i < c.m * n_embd; ++i) {
                    uint32_t bits;
                    memcpy(&bits, &out[(size_t) i], 4);
                    hash = (hash ^ bits) * 1099511628211ull;
                }
            }
        double tot = 0, etot = 0;
        for (int b = 0; b < 4; ++b) tot += ms[b], etot += eng[b];
        printf("trace %zu calls x %d: %.1f ms (engine recorded %.1f ms), hash %016llx |", calls.size(), repeats, tot, etot,
               (unsigned long long) hash);
        for (int b = 0; b < 4; ++b)
            printf(" %s: %ld calls %.1f ms (engine %.1f) |", names[b], count[b] / repeats, ms[b], eng[b]);
        printf("%s\n", sink == 42 ? " " : "");
        return 0;
    }

    std::mt19937 rng(1234);
    std::vector<std::vector<int>> ids(L.size());
    size_t pool_bytes = 0;
    for (size_t l = 0; l < L.size(); ++l) {
        std::vector<int> p(n_exp);
        std::iota(p.begin(), p.end(), 0);
        std::shuffle(p.begin(), p.end(), rng);
        ids[l].assign(p.begin(), p.begin() + pool);
        pool_bytes += (size_t) pool * (2 * L[l].gu_stride + L[l].d_stride);
    }

    uint8_t* anon = nullptr;
    const bool is_anon = mode != "mmap";
    if (is_anon) {
        anon = (uint8_t*) mmap(nullptr, pool_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        madvise(anon, pool_bytes, mode == "anon4k" ? MADV_NOHUGEPAGE : MADV_HUGEPAGE);
    }
    size_t at = 0;
    long sink = 0;
    for (size_t l = 0; l < L.size(); ++l) {
        Layer& y = L[l];
        for (int e : ids[l]) {
            const uint8_t* src[3] = {file + y.gate_off + (size_t) e * y.gu_stride, file + y.up_off + (size_t) e * y.gu_stride,
                                     file + y.down_off + (size_t) e * y.d_stride};
            const size_t len[3] = {y.gu_stride, y.gu_stride, y.d_stride};
            const uint8_t* dst[3];
            for (int k = 0; k < 3; ++k) {
                if (is_anon) {
                    memcpy(anon + at, src[k], len[k]);
                    if (mode == "repack" && k < 2 && y.gu_type == GGML_TYPE_Q4_K) repack_q4_K(anon + at, n_ff, n_embd);
                    dst[k] = anon + at;
                    at += len[k];
                } else {
                    for (size_t i = 0; i < len[k]; i += 4096) sink += src[k][i];   // into the page cache
                    dst[k] = src[k];
                }
            }
            y.gate.push_back(dst[0]);
            y.up.push_back(dst[1]);
            y.down.push_back(dst[2]);
        }
    }

    Workers workers(threads);
    std::vector<float> x((size_t) rows * n_embd), out((size_t) rows * n_embd), ref((size_t) rows * n_embd);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (float& v : x) v = nd(rng);
    const float limit = INFINITY;
    std::vector<CpuExpert> ex((size_t) rows * n_used);

    // Exactness of repack: each layer's first call against cpu_experts() on the same (repacked) rows can't be
    // compared, so check against the file's rows
    double max_diff = 0, max_ref = 0;
    uint64_t hash = 1469598103934665603ull;   // FNV-1a over every call's output bits: today's build and PR 36's must agree
    auto token = [&](bool check) {
        size_t bytes = 0;
        for (size_t l = 0; l < L.size(); ++l) {
            const Layer& y = L[l];
            std::vector<int> used;
            for (int r = 0; r < rows; ++r) {
                std::vector<int> pick(pool);
                std::iota(pick.begin(), pick.end(), 0);
                for (int k = 0; k < n_used; ++k) {
                    std::swap(pick[k], pick[k + rng() % (pool - k)]);
                    ex[(size_t) r * n_used + k] = {y.gate[pick[k]], y.up[pick[k]], y.down[pick[k]], 1.f / n_used, r};
                    used.push_back(pick[k]);
                }
            }
            std::sort(used.begin(), used.end());
            bytes += (size_t) (std::unique(used.begin(), used.end()) - used.begin()) * (2 * y.gu_stride + y.d_stride);
            const bool rp = mode == "repack" && y.gu_type == GGML_TYPE_Q4_K;
            if (rp)
                cpu_experts_repack(workers, ex.data(), rows * n_used, rows, y.gu_type, y.d_type, n_embd, n_ff, x.data(),
                                   limit, out.data());
            else
                cpu_experts(workers, ex.data(), rows * n_used, rows, y.gu_type, y.d_type, n_embd, n_ff, x.data(), limit,
                            false, out.data());
            for (float v : out) {
                uint32_t b;
                memcpy(&b, &v, 4);
                hash = (hash ^ b) * 1099511628211ull;
            }
            if (check && rp) {
                std::vector<CpuExpert> fx = ex;
                for (auto& c : fx) {
                    const int k = (int) (std::find(y.gate.begin(), y.gate.end(), c.gate) - y.gate.begin());
                    c.gate = file + y.gate_off + (size_t) ids[l][k] * y.gu_stride;
                    c.up = file + y.up_off + (size_t) ids[l][k] * y.gu_stride;
                }
                cpu_experts(workers, fx.data(), rows * n_used, rows, y.gu_type, y.d_type, n_embd, n_ff, x.data(), limit,
                            false, ref.data());
                for (size_t i = 0; i < out.size(); ++i) {
                    max_diff = std::max(max_diff, (double) std::fabs(out[i] - ref[i]));
                    max_ref = std::max(max_ref, (double) std::fabs(ref[i]));
                }
            }
        }
        return bytes;
    };
    if (mode == "repack") token(true);
    for (int i = 0; i < 20; ++i) token(false);   // warm-up
    double secs = 0;
    size_t bytes = 0;
    for (int i = 0; i < tokens; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        bytes += token(false);
        secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    printf("%-7s rows %d pool %d threads %d: %.2f ms/token, %.1f GB/s (%.0f MB a token)", mode.c_str(), rows, pool,
           threads, secs / tokens * 1e3, bytes / secs / 1e9, bytes / (double) tokens / 1e6);
    printf(", output hash %016llx", (unsigned long long) hash);
    if (is_anon) printf(", AnonHugePages %ld MiB of %zu", anon_huge_kb() / 1024, pool_bytes >> 20);
    if (mode == "repack") printf(", max diff %.3g (largest output %.3g)", max_diff, max_ref);
    printf("%s\n", sink == 42 ? " " : "");
    return 0;
}
