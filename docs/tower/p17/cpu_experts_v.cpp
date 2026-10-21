// Pass 17 replay variants of src/poly/cpu_experts.cpp (strata-poly/cpu-down-order 81d6dfab), down order always on.
// Every build counts, per worker, the time spent inside tasks, by phase (0 activations, 1 gate/up, 2 h, 3 down) and
// by the call shape the replay sets (cxb_shape), plus each phase's wall time on the calling thread.  Compile-time:
//   CXB_ARITH  every product reads the call's first expert's rows (the call keeps its experts and shape), so the
//              bytes come from L2/L3: an upper bound on the arithmetic
//   CXB_READS  the products replaced by 256-bit loads of the same rows, XORed into what the row writes
//   CXB_IDLE   the tasks return at once: posting and waiting only
// With none of them it is the engine's code with STRATA_CPU_DOWN_ORDER=1.
#include "cpu_experts.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <x86intrin.h>
#include <mutex>
#include <chrono>
#include <numeric>

namespace strata::poly {
namespace {

// Rows of gate and up, and of down, a task.  With fewer tasks than several a thread, the last round left threads
// idle at the end of every layer: 64 rows gave Tiel's 512-row experts 8 tasks each, and its down products 32 tasks.
constexpr int kRowsPerTask = 32;
// STRATA_CPU_DOWN_ORDER=1: a down task takes 64 rows and reads them expert by expert, each expert's rows of the task
// in one piece, where today's loop reads one row of every expert in turn.  The same products and the same sums in
// the same order, so the same bits.  On a Ryzen 5 2600X with DDR4-3200 (Cyber-Tiel-Coder 35B-A3B UD-Q4_K_XL, RTX
// 3070) the CPU experts took 7-11% less time a token and decode was 2-12% faster, also with every call one token.
constexpr int kRowsPerDownRun = 64;
constexpr int kSpinsBeforeSleep = 20000;   // about a millisecond of pause instructions

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

thread_local bool in_task = false;   // this thread is running a task of a batch

#ifndef CXB_ARITH
#define CXB_ARITH 0
#endif
#ifndef CXB_READS
#define CXB_READS 0
#endif
#ifndef CXB_IDLE
#define CXB_IDLE 0
#endif
// Tower's kernel runs on HPET (it marked the TSC unstable at boot), where a steady_clock read takes 2.5 µs.  The
// counters read the TSC (32 cycles), at its rate measured once against steady_clock.
struct Clock {
    using time_point = uint64_t;
    static time_point now() { return __rdtsc(); }
};
double tsc_per_us() {
    static const double r = [] {
        const auto w0 = std::chrono::steady_clock::now();
        const uint64_t c0 = __rdtsc();
        while (std::chrono::steady_clock::now() - w0 < std::chrono::milliseconds(200)) {}
        const uint64_t c1 = __rdtsc();
        return (c1 - c0) / std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - w0).count();
    }();
    return r;
}
inline double us_since(Clock::time_point t0) { return (__rdtsc() - t0) / tsc_per_us(); }
}  // namespace

// The replay's counters: busy[thread][shape][phase], wall[shape][phase] in µs.  The shape and phase are set by the
// caller before a run(); no run() overlaps another.
constexpr int kMaxThreads = 32, kShapes = 3, kPhases = 4;
double cxb_busy[kMaxThreads][kShapes][kPhases], cxb_wall[kShapes][kPhases];
int cxb_shape = 0, cxb_phase = 0;
double cxb_tsc_per_us() { return tsc_per_us(); }
namespace {
std::atomic<int> next_id{0};
thread_local int my_id = -1;
inline int thread_id() { return my_id >= 0 ? my_id : (my_id = next_id.fetch_add(1)); }

// The read in place of a product: every 32 bytes of the row, XORed
__attribute__((target("avx2"))) inline float read_row(const uint8_t* p, size_t n) {
    __m256i x = _mm256_setzero_si256();
    for (size_t i = 0; i + 32 <= n; i += 32) x = _mm256_xor_si256(x, _mm256_loadu_si256((const __m256i*) (p + i)));
    return (float) (_mm256_extract_epi32(x, 0) & 0xffff);
}

// One product of the variant: a row of a matrix against an activation
inline void product(const ggml_type_traits_cpu* t, int n, float* s, const uint8_t* row, size_t row_bytes,
                    const uint8_t* a) {
    if (CXB_READS) *s = read_row(row, row_bytes);
    else t->vec_dot(n, s, 0, row, 0, a, 0, 1);
}

}  // namespace

Workers::Workers(int threads) {
    for (int i = 1; i < threads; ++i) th_.emplace_back([this] { loop(); });
}

Workers::~Workers() {
    quit_.store(true);
    posted_.fetch_add(1);
    posted_.notify_all();
    for (auto& t : th_) t.join();
}

// Whether it ran a task.  next is read before it is counted up, so it stops near n however often a worker comes
// back to a batch whose tasks are taken.
bool Workers::drain(Batch& b) {
    bool ran = false;
    while (b.next.load(std::memory_order_relaxed) < b.n) {
        const int i = b.next.fetch_add(1, std::memory_order_relaxed);
        if (i >= b.n) break;
        in_task = true;
        const auto t0 = Clock::now();
        (*b.fn)(i);
        cxb_busy[thread_id()][cxb_shape][cxb_phase] += us_since(t0);
        in_task = false;
        b.left.fetch_sub(1, std::memory_order_release);
        ran = true;
    }
    return ran;
}

// A worker with nothing to do only reads posted_: a write by each of them, every turn of the loop, kept run() from
// ever seeing users_ at zero for long.  When a batch has been posted since the worker last looked, it counts itself
// in users_ and reads cur_ again before it uses it, so run() can tell when nobody holds its Batch any more.
//
// A batch stays posted after its last task is taken, until run() takes it down, and run() cannot while it has no
// CPU.  That happens with more threads than the system runs: a laptop that parks cores under load makes that of
// any count.  So a worker that found no task in what is posted leaves it alone: it pauses, then sleeps, until
// posted_ moves.  Going round the finished batch instead, the workers kept run() off the CPU for the rest of
// their time slices, every batch.
void Workers::loop() {
    int idle = 0;
    uint32_t seen = posted_.load() - 1;   // the count at which this worker last found nothing to do
    while (!quit_.load(std::memory_order_relaxed)) {
        const uint32_t now = posted_.load(std::memory_order_acquire);
        if (now != seen) {
            // cur_ is set before posted_ counts: the batch of `now` is there, or a later one, or it is done
            bool ran = false;
            if (cur_.load(std::memory_order_acquire)) {
                users_.fetch_add(1);
                Batch* b = cur_.load();
                if (b) ran = drain(*b);
                users_.fetch_sub(1);
            }
            if (ran) {
                idle = 0;
                continue;
            }
            seen = now;
        }
        if (++idle < kSpinsBeforeSleep) {
            _mm_pause();
            continue;
        }
        // run() counts posted_ up before it reads sleepers_, and wait() returns at once when posted_ has moved
        sleepers_.fetch_add(1);
        if (!quit_.load()) posted_.wait(seen);
        sleepers_.fetch_sub(1);
        idle = 0;
    }
}

void Workers::run(int n, const std::function<void(int)>& fn) {
    if (n <= 0) return;
    // One task is nothing to share out.  And a task that shares out work of its own does it alone: there is one
    // batch at a time, and a worker that waited below for the others to let go of theirs would wait for itself.
    if (n == 1 || in_task) {
        const auto t0 = Clock::now();
        for (int i = 0; i < n; ++i) fn(i);
        if (!in_task) cxb_busy[thread_id()][cxb_shape][cxb_phase] += us_since(t0);
        return;
    }
    const std::lock_guard<std::mutex> one(one_);
    Batch b;
    b.fn = &fn;
    b.n = n;
    b.left.store(n);
    cur_.store(&b);
    posted_.fetch_add(1);
    if (sleepers_.load() > 0) posted_.notify_all();
    drain(b);
    cur_.store(nullptr);   // every task is taken: a worker that comes now has nothing to find
    while (b.left.load(std::memory_order_acquire) != 0) _mm_pause();
    while (users_.load() != 0) _mm_pause();
}

bool cpu_expert_types_ok(int gu_type, int d_type, int n_embd, int n_ff, std::string& err) {
    init_once();
    for (const int type : {gu_type, d_type}) {
        const ggml_type_traits_cpu* t = traits(type);
        const ggml_type_traits_cpu* a = t && t->vec_dot ? traits(t->vec_dot_type) : nullptr;
        if (a == nullptr || a->from_float == nullptr) {
            err = "ggml-cpu has no dot product for tensor type " + std::to_string(type);
            return false;
        }
        const int n = type == gu_type ? n_embd : n_ff;
        if (n % ggml_blck_size((ggml_type) type) || n % ggml_blck_size(t->vec_dot_type)) {
            err = "a row of " + std::to_string(n) + " values is not whole blocks of tensor type " + std::to_string(type);
            return false;
        }
    }
    return true;
}

void cpu_experts(Workers& workers, const CpuExpert* experts, int n, int rows, int gu_type, int d_type, int n_embd,
                 int n_ff, const float* x, float limit, bool gelu, float* out,
                 const std::function<void()>* meanwhile) {
    init_once();
    auto timed = [&](int phase, int tasks, const std::function<void(int)>& fn) {
        cxb_phase = phase;
        const auto t0 = Clock::now();
        workers.run(tasks, fn);
        cxb_wall[cxb_shape][phase] += us_since(t0);
    };
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    const size_t gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    const size_t d_row = ggml_row_size((ggml_type) d_type, n_ff);
    const size_t a_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    const size_t h_bytes = ggml_row_size(td->vec_dot_type, n_ff);

    // The workers get these through the pointers below: a thread_local named in their code would be their own.
    static thread_local std::vector<uint8_t> act, hq;
    static thread_local std::vector<float> h;
    static thread_local std::vector<int> by, run;
    act.resize((size_t) rows * a_bytes);
    hq.resize((size_t) n * h_bytes);
    h.resize((size_t) n * n_ff);
    // ggml's quantizers for the activations are scalar: microseconds a row, which on one thread kept the others
    // waiting, here and before the down products
    uint8_t* actp = act.data();
    const auto act_from = traits(tg->vec_dot_type)->from_float;
    timed(0, rows, [&](int t) { if (CXB_IDLE) return; act_from(x + (size_t) t * n_embd, actp + (size_t) t * a_bytes, n_embd); });

    // The uses of one expert next to each other (by), and where each expert's run of them starts (run).
    by.resize((size_t) n);
    std::iota(by.begin(), by.end(), 0);
    std::stable_sort(by.begin(), by.end(), [&](int a, int b) { return experts[a].gate < experts[b].gate; });
    run.clear();
    for (int i = 0; i < n; ++i)
        if (i == 0 || experts[by[(size_t) i]].gate != experts[by[(size_t) i - 1]].gate) run.push_back(i);
    const int n_run = (int) run.size();
    run.push_back(n);

    float* hp = h.data();
    const int* byp = by.data();
    const int* runp = run.data();
    const int gu_tasks = (n_ff + kRowsPerTask - 1) / kRowsPerTask;
    const int first = meanwhile ? 1 : 0;   // task 0, so it is taken at once
    timed(1, first + n_run * gu_tasks, [&](int t) {
        if (t < first) return (*meanwhile)();
        if (CXB_IDLE) return;
        t -= first;
        const int i0 = runp[t / gu_tasks], i1 = runp[t / gu_tasks + 1];
        const CpuExpert& e = experts[CXB_ARITH ? 0 : byp[i0]];
        const int r0 = (t % gu_tasks) * kRowsPerTask, r1 = std::min(n_ff, r0 + kRowsPerTask);
        for (int r = r0; r < r1; ++r) {
            const uint8_t* gate = e.gate + (size_t) r * gu_row;
            const uint8_t* up = e.up + (size_t) r * gu_row;
            for (int i = i0; i < i1; ++i) {
                const int j = byp[i];
                const uint8_t* a = actp + (size_t) experts[j].row * a_bytes;
                float g = 0.f, u = 0.f;
                product(tg, n_embd, &g, gate, gu_row, a);
                product(tg, n_embd, &u, up, gu_row, a);
                if (gelu) {
                    hp[(size_t) j * n_ff + r] =
                        0.5f * g * (1.f + std::tanh(0.7978845608f * g * (1.f + 0.044715f * g * g))) * u;
                    continue;
                }
                g = std::min(g, limit);
                u = std::clamp(u, -limit, limit);
                hp[(size_t) j * n_ff + r] = g / (1.f + std::exp(-g)) * u;
            }
        }
    });
    uint8_t* hqp = hq.data();
    const auto hq_from = traits(td->vec_dot_type)->from_float;
    timed(2, n, [&](int j) { if (CXB_IDLE) return; hq_from(hp + (size_t) j * n_ff, hqp + (size_t) j * h_bytes, n_ff); });

    {
        const int tasks = (n_embd + kRowsPerDownRun - 1) / kRowsPerDownRun;
        timed(3, tasks, [&](int t) {
            if (CXB_IDLE) return;
            static thread_local std::vector<float> rs;   // each row's output of every use
            rs.resize((size_t) kRowsPerDownRun * n);
            const int r0 = t * kRowsPerDownRun, r1 = std::min(n_embd, r0 + kRowsPerDownRun);
            for (int u = 0; u < n_run; ++u) {
                const uint8_t* down = experts[CXB_ARITH ? 0 : byp[runp[u]]].down;
                for (int r = r0; r < r1; ++r)
                    for (int i = runp[u]; i < runp[u + 1]; ++i)
                        product(td, n_ff, rs.data() + (size_t) (r - r0) * n + byp[i], down + (size_t) r * d_row, d_row,
                                hqp + (size_t) byp[i] * h_bytes);
            }
            for (int r = r0; r < r1; ++r) {
                const float* s = rs.data() + (size_t) (r - r0) * n;
                int j = 0;
                for (int row = 0; row < rows; ++row) {
                    float acc = 0.f;
                    for (; j < n && experts[j].row == row; ++j) acc += experts[j].weight * s[j];
                    out[(size_t) row * n_embd + r] = acc;
                }
            }
        });
    }
}

}  // namespace strata::poly
