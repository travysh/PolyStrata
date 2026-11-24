// src/poly/cpu_experts.cpp - see the header.
#include "cpu_experts.hpp"

#include "strata/quant/act.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/fp16.hpp"
#include "strata/quant/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <mutex>
#include <numeric>
#include <utility>

namespace strata::poly {
namespace {

// Rows of gate and up, and of down, a task.  With fewer tasks than several a thread, the last round left threads
// idle at the end of every layer: 64 rows gave Tiel's 512-row experts 8 tasks each, and its down products 32 tasks.
constexpr int kRowsPerTask = 32;
constexpr int kRowsStep = 4;   // a gate and up task's rows are a multiple of this, from it up to kRowsPerTask
constexpr int kRowsPerDownTask = 16;
constexpr int kRowsOnCaller = 4;   // up to this many rows' activations are quantized by the caller's thread
constexpr int kSpinsBeforeSleep = 20000;   // about a millisecond of pause instructions

namespace Q = strata::quant;

// A weight type's rows: its dot product and the quantizer of the activations that product reads, from Strata's quant
// layer (src/quant) in the arithmetic this CPU's ggml-cpu build runs, so the products are the ones ggml-cpu gave.
// The layer has no rows for the float types (F32, F16, BF16): those are multiplied here, a value at a time, with the
// activations left as floats and the sum kept in a double.  That is not ggml-cpu's order of additions.
using ActFn = void (*)(const float* x, void* y, int64_t n);
using DotFn = void (*)(int n, float* s, const void* w, const void* a);

void act_floats(const float* x, void* y, int64_t n) { std::memcpy(y, x, (size_t) n * sizeof(float)); }

void dot_f32(int n, float* s, const void* w, const void* a) {
    const float* wf = (const float*) w;
    const float* af = (const float*) a;
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double) wf[i] * af[i];
    *s = (float) acc;
}

void dot_f16(int n, float* s, const void* w, const void* a) {
    const uint16_t* wh = (const uint16_t*) w;
    const float* af = (const float*) a;
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double) Q::fp16_to_fp32(wh[i]) * af[i];
    *s = (float) acc;
}

void dot_bf16(int n, float* s, const void* w, const void* a) {
    const uint16_t* wb = (const uint16_t*) w;
    const float* af = (const float*) a;
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double) Q::f32_from_bits((uint32_t) wb[i] << 16) * af[i];
    *s = (float) acc;
}

struct Rows {
    DotFn dot = nullptr;
    Q::DotRowsFn dots = nullptr;   // the same products for several activation rows at once, where there is one
    Q::DotX4Fn x4 = nullptr;       // four weight rows against one activation row, each value dot's bits, where there is one
    ActFn act = nullptr;
    int blck = 1;          // values a block of the weights
    int act_blck = 1;      // values a block of the activations
    size_t act_block = 4;  // bytes a block of the activations
    int act_type = 0;      // which activations: two types with the same one share a row of them
    size_t row(int64_t n) const { return (size_t) (n / act_blck) * act_block; }
};

Rows rows_of(int type) {
    Rows r;
    const Q::TypeInfo* t = Q::type_info(type);
    if (t == nullptr) return r;
    if (t->type == Q::Type::F32 || t->type == Q::Type::F16 || t->type == Q::Type::BF16) {
        r.dot = t->type == Q::Type::F32 ? dot_f32 : t->type == Q::Type::F16 ? dot_f16 : dot_bf16;
        r.act = act_floats;
        return r;
    }
    const Q::TypeInfo* a = t->has_dot ? Q::type_info(t->act) : nullptr;
    if (a == nullptr) return r;
    const Q::Arith arith = Q::default_arith();
    const Q::DotKernel* dot = Q::dot_kernel(t->type, arith);
    const Q::ActKernel* act = Q::act_kernel(t->act, arith);
    if (dot == nullptr || act == nullptr) return r;
    r.dot = dot->fn;
    r.dots = Q::dot_rows_kernel(t->type, arith);
    // the layer's four-row kernel of dot's own level only: one of another level (AVX-512 beside the AVX2 Q4_K the
    // table keeps) need not give dot's bits.  cpu_x4_test checks that each one taken here does.
    const Q::DotX4Kernel* x4 = Q::dot_x4_kernel(t->type, arith, dot->isa);
    if (x4 != nullptr && x4->isa == dot->isa) r.x4 = x4->fn;
    r.act = act->fn;
    r.blck = t->blck;
    r.act_blck = a->blck;
    r.act_block = a->bytes;
    r.act_type = (int) t->act;
    return r;
}

thread_local bool in_task = false;   // this thread is running a task of a batch

constexpr int kUses = 8;   // an expert's uses taken together by the gate and up products

// s[q][k] = weight row q of the nr rows at w (row_bytes apart) times activation row a[k], for k < m: four rows a call
// where the format has a four-row kernel and no several-activations kernel covers the uses, else a row a call.  Every
// value is the bits of t.dot's for that pair, whichever way it is taken.
void products(const Rows& t, int n, int nr, int m, float (*s)[kUses], const uint8_t* w, size_t row_bytes,
              const void* const* a) {
    if (nr == 4 && t.x4 && !(m > 1 && t.dots)) {
        for (int k = 0; k < m; ++k) {
            float o[4];
            t.x4(n, o, w, row_bytes, a[k]);
            for (int q = 0; q < 4; ++q) s[q][k] = o[q];
        }
        return;
    }
    for (int q = 0; q < nr; ++q) {
        const uint8_t* row = w + (size_t) q * row_bytes;
        if (m > 1 && t.dots) t.dots(n, m, s[q], row, a);
        else for (int k = 0; k < m; ++k) t.dot(n, s[q] + k, row, a[k]);
    }
}

}  // namespace

CpuDotX4 cpu_dot_x4(int type) {
    const Rows r = rows_of(type);
    CpuDotX4 out;
    if (r.x4 != nullptr) out.one = r.dot, out.four = r.x4;
    return out;
}

Workers::Workers(int threads) : busy_((size_t) std::max(threads - 1, 0)) {
    for (int i = 1; i < threads; ++i) th_.emplace_back([this, i] { loop(i - 1); });
}

Workers::~Workers() {
    quit_.store(true);
    posted_.fetch_add(1);
    posted_.notify_all();
    for (auto& t : th_) t.join();
}

// Whether it ran a task.  next is read before it is counted up, so it stops near n however often a worker comes
// back to a batch whose tasks are taken.  The tasks it ran come off left together, once it finds no more.
bool Workers::drain(Batch& b) {
    int ran = 0;
    while (b.next.load(std::memory_order_relaxed) < b.n) {
        const int i = b.next.fetch_add(1, std::memory_order_relaxed);
        if (i >= b.n) break;
        in_task = true;
        (*b.fn)(i);
        in_task = false;
        ++ran;
    }
    if (ran) b.left.fetch_sub(ran, std::memory_order_release);
    return ran > 0;
}

// A worker with nothing to do only reads posted_: a write by each of them, every turn of the loop, kept run() from
// ever seeing its Batch free for long.  When a batch has been posted since the worker last looked, it raises its
// Busy mark and reads cur_ again before it uses it, so run() can tell when nobody holds its Batch any more: the mark
// is stored and cur_ loaded in that order, and run() stores cur_ and loads the marks in that order (all seq_cst), so
// either the worker finds cur_ empty or run() finds the mark up.
//
// A batch stays posted after its last task is taken, until run() takes it down, and run() cannot while it has no
// CPU.  That happens with more threads than the system runs: a laptop that parks cores under load makes that of
// any count.  So a worker that found no task in what is posted leaves it alone: it pauses, then sleeps, until
// posted_ moves.  Going round the finished batch instead, the workers kept run() off the CPU for the rest of
// their time slices, every batch.
void Workers::loop(int i) {
    std::atomic<int>& busy = busy_[(size_t) i].on;
    int idle = 0;
    uint32_t seen = posted_.load() - 1;   // the count at which this worker last found nothing to do
    while (!quit_.load(std::memory_order_relaxed)) {
        const uint32_t now = posted_.load(std::memory_order_acquire);
        if (now != seen) {
            // cur_ is set before posted_ counts: the batch of `now` is there, or a later one, or it is done
            bool ran = false;
            if (cur_.load(std::memory_order_acquire)) {
                busy.store(1);
                Batch* b = cur_.load();
                if (b) ran = drain(*b);
                busy.store(0, std::memory_order_release);
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
        for (int i = 0; i < n; ++i) fn(i);
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
    for (Busy& w : busy_)
        while (w.on.load() != 0) _mm_pause();
}

Aside::Aside() : th_([this] { loop(); }) {}

Aside::~Aside() {
    quit_.store(true);
    posted_.fetch_add(1);
    posted_.notify_all();
    th_.join();
}

// It pauses between functions, as an idle worker does, and sleeps once that has gone on for long: the next comes a
// few hundred microseconds after the last while a model answers.
void Aside::loop() {
    uint32_t seen = 0;
    int idle = 0;
    while (true) {
        const uint32_t now = posted_.load(std::memory_order_acquire);
        if (quit_.load(std::memory_order_relaxed)) return;
        if (now != seen) {
            seen = now;
            try {
                (*fn_)();
            } catch (...) {
                failed_ = std::current_exception();
            }
            done_.store(now, std::memory_order_release);
            idle = 0;
            continue;
        }
        if (++idle < kSpinsBeforeSleep) {
            _mm_pause();
            continue;
        }
        // start() counts posted_ up before it reads sleeping_, and wait() returns at once when posted_ has moved
        sleeping_.store(true);
        posted_.wait(seen);
        sleeping_.store(false);
        idle = 0;
    }
}

void Aside::start(const std::function<void()>& fn) {
    finish();   // one function at a time: the one before is done first
    fn_ = &fn;
    posted_.fetch_add(1);
    if (sleeping_.load()) posted_.notify_one();
}

void Aside::finish() {
    const uint32_t want = posted_.load(std::memory_order_relaxed);
    while (done_.load(std::memory_order_acquire) != want) _mm_pause();
    if (failed_) std::rethrow_exception(std::exchange(failed_, nullptr));
}

bool cpu_expert_types_ok(int gu_type, int d_type, int n_embd, int n_ff, std::string& err) {
    for (const int type : {gu_type, d_type}) {
        const Rows t = rows_of(type);
        if (t.dot == nullptr) {
            err = "the CPU has no dot product for tensor type " + std::to_string(type);
            return false;
        }
        const int n = type == gu_type ? n_embd : n_ff;
        if (n % t.blck || n % t.act_blck) {
            err = "a row of " + std::to_string(n) + " values is not whole blocks of tensor type " + std::to_string(type);
            return false;
        }
    }
    return true;
}

void cpu_experts(Workers& workers, const CpuExpert* experts, int n, int rows, int gu_type, int u_type, int d_type,
                 int n_embd, int n_ff, const float* x, float limit, bool gelu, float* out,
                 const std::function<void()>* meanwhile, const uint8_t* late, const std::function<void(int)>* wait,
                 float situ_beta, float situ_lb) {
    const Rows tg = rows_of(gu_type), tu = rows_of(u_type), td = rows_of(d_type);
    const size_t gu_row = Q::row_bytes(gu_type, n_embd);
    const size_t u_row = Q::row_bytes(u_type, n_embd);
    const size_t d_row = Q::row_bytes(d_type, n_ff);
    const size_t a_bytes = tg.row(n_embd);
    const size_t au_bytes = tu.row(n_embd);
    const size_t h_bytes = td.row(n_ff);
    const bool own_act = tu.act_type != tg.act_type;   // up's dot product takes another quantization

    // The workers get these through the pointers below: a thread_local named in their code would be their own.
    static thread_local std::vector<uint8_t> act, act_u, hq;
    static thread_local std::vector<float> h;
    static thread_local std::vector<int> by, run;
    act.resize((size_t) rows * a_bytes);
    if (own_act) act_u.resize((size_t) rows * au_bytes);
    hq.resize((size_t) n * h_bytes);
    h.resize((size_t) n * n_ff);
    // Quantizing the activations takes about a microsecond a row.  A few rows are done here: sharing them out cost
    // a round of the workers, more than the rows.  Many rows on one thread kept the others waiting.
    uint8_t* actp = act.data();
    uint8_t* actup = own_act ? act_u.data() : actp;
    const ActFn act_from = tg.act, au_from = tu.act;
    const auto quantize = [&](int t) {
        act_from(x + (size_t) t * n_embd, actp + (size_t) t * a_bytes, n_embd);
        if (own_act) au_from(x + (size_t) t * n_embd, actup + (size_t) t * au_bytes, n_embd);
    };
    if (rows <= kRowsOnCaller)
        for (int t = 0; t < rows; ++t) quantize(t);
    else
        workers.run(rows, quantize);
    const size_t au_step = own_act ? au_bytes : a_bytes;

    // The uses of one expert next to each other (by), and where each expert's run of them starts (run).
    by.resize((size_t) n);
    std::iota(by.begin(), by.end(), 0);
    std::stable_sort(by.begin(), by.end(), [&](int a, int b) { return experts[a].gate < experts[b].gate; });
    run.clear();
    for (int i = 0; i < n; ++i)
        if (i == 0 || experts[by[(size_t) i]].gate != experts[by[(size_t) i - 1]].gate) run.push_back(i);
    const int n_run = (int) run.size();
    run.push_back(n);
    // the runs in the order their products are taken: the late experts' last, so their reads go on meanwhile
    static thread_local std::vector<int> order;
    order.resize((size_t) n_run);
    std::iota(order.begin(), order.end(), 0);
    if (late)
        std::stable_partition(order.begin(), order.end(), [&](int u) { return !late[by[(size_t) run[(size_t) u]]]; });

    float* hp = h.data();
    const int* byp = by.data();
    const int* runp = run.data();
    const int* orderp = order.data();
    // Rows a task: about four tasks a thread, so the last round leaves few threads idle.  At 6 GB a call has two or
    // three experts on the CPU, which at 32 rows a task were two rounds of 32 threads, the second one mostly idle.
    const int threads = std::max(workers.threads(), 1);
    int per = (n_run * n_ff) / (4 * threads);
    per = std::clamp(per - per % kRowsStep, kRowsStep, kRowsPerTask);
    const int gu_tasks = (n_ff + per - 1) / per;
    // The task that finishes an expert's run quantizes its uses' rows for the down products: a round of the workers
    // less than doing it after all of them.
    uint8_t* hqp = hq.data();
    const ActFn hq_from = td.act;
    static thread_local std::vector<std::atomic<int>> left_of;
    if (left_of.size() < (size_t) n_run) left_of = std::vector<std::atomic<int>>((size_t) n_run);
    for (int u = 0; u < n_run; ++u) left_of[(size_t) u].store(gu_tasks, std::memory_order_relaxed);
    std::atomic<int>* leftp = left_of.data();
    const int first = meanwhile ? 1 : 0;   // task 0, so it is taken at once
    workers.run(first + n_run * gu_tasks, [&](int t) {
        if (t < first) return (*meanwhile)();
        t -= first;
        const int u = orderp[t / gu_tasks], i0 = runp[u], i1 = runp[u + 1];
        if (late && late[byp[i0]]) (*wait)(byp[i0]);
        const CpuExpert& e = experts[byp[i0]];
        const int r0 = (t % gu_tasks) * per, r1 = std::min(n_ff, r0 + per);
        // Four weight rows at a time (Rows::x4, a format's four-row kernel), the last few one by one.  The expert's
        // uses kUses at a time: where the format has a kernel for several activation rows (Rows::dots), a weight row
        // is decoded once for all of them.  Each product is the bits of its own.
        for (int r = r0; r < r1;) {
            const int nr = r + 4 <= r1 ? 4 : 1;
            const uint8_t* gate = e.gate + (size_t) r * gu_row;
            const uint8_t* up = e.up + (size_t) r * u_row;
            for (int i = i0; i < i1;) {
                const int m = std::min(kUses, i1 - i);
                float gs[4][kUses], us[4][kUses];
                const void *as[kUses], *aus[kUses];
                for (int k = 0; k < m; ++k) {
                    as[k] = actp + (size_t) experts[byp[i + k]].row * a_bytes;
                    aus[k] = actup + (size_t) experts[byp[i + k]].row * au_step;
                }
                products(tg, n_embd, nr, m, gs, gate, gu_row, as);
                products(tu, n_embd, nr, m, us, up, u_row, aus);
                for (int k = 0; k < m; ++k, ++i) {
                    const int j = byp[i];
                    for (int q = 0; q < nr; ++q) {
                        float g = gs[q][k] * e.gate_scale, u = us[q][k] * e.up_scale;
                        float& hv = hp[(size_t) j * n_ff + r + q];
                        if (situ_beta > 0.f) {
                            if (situ_lb > 0.f) u = situ_lb * std::tanh(u / situ_lb);
                            hv = situ_beta * std::tanh(g / situ_beta) / (1.f + std::exp(-g)) * u;
                            continue;
                        }
                        if (gelu) {
                            hv = 0.5f * g * (1.f + std::tanh(0.7978845608f * g * (1.f + 0.044715f * g * g))) * u;
                            continue;
                        }
                        g = std::min(g, limit);
                        u = std::clamp(u, -limit, limit);
                        hv = g / (1.f + std::exp(-g)) * u;
                    }
                }
            }
            r += nr;
        }
        // the last of the run's tasks sees every other one's rows (release by each, acquire by the last)
        if (leftp[u].fetch_sub(1, std::memory_order_acq_rel) == 1)
            for (int i = i0; i < i1; ++i) {
                const int j = byp[i];
                hq_from(hp + (size_t) j * n_ff, hqp + (size_t) j * h_bytes, n_ff);
            }
    });

    const int d_tasks = (n_embd + kRowsPerDownTask - 1) / kRowsPerDownTask;
    workers.run(d_tasks, [&](int t) {
        static thread_local std::vector<float> one;   // an output value of every use, for each of up to four rows
        one.resize((size_t) 4 * n);
        const int r0 = t * kRowsPerDownTask, r1 = std::min(n_embd, r0 + kRowsPerDownTask);
        for (int r = r0; r < r1;) {
            const int nr = td.x4 && r + 4 <= r1 ? 4 : 1;   // four rows a call where the format has a four-row kernel
            for (int u = 0; u < n_run; ++u) {
                const uint8_t* down = experts[byp[runp[u]]].down + (size_t) r * d_row;
                for (int i = runp[u]; i < runp[u + 1]; ++i) {
                    const int j = byp[i];
                    const uint8_t* hj = hqp + (size_t) j * h_bytes;
                    if (nr == 4) {
                        float o[4];
                        td.x4(n_ff, o, down, d_row, hj);
                        for (int q = 0; q < 4; ++q) one[(size_t) q * n + j] = o[q];
                    } else {
                        td.dot(n_ff, one.data() + j, down, hj);
                    }
                }
            }
            // a token's sum in the order its experts were given: the order it has in a call of its own
            for (int q = 0; q < nr; ++q) {
                const float* s = one.data() + (size_t) q * n;
                int j = 0;
                for (int row = 0; row < rows; ++row) {
                    float acc = 0.f;
                    for (; j < n && experts[j].row == row; ++j) acc += experts[j].weight * s[j];
                    out[(size_t) row * n_embd + r + q] = acc;
                }
            }
            r += nr;
        }
    });
}

}  // namespace strata::poly
