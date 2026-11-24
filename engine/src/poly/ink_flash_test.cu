// src/poly/ink_flash_test.cu - ink_attend_batch() (the flash kernels: ink_flash_kernel from Ampere on,
// ink_flash75_kernel or ink_mma75_kernel on Turing) against a float64 reference of the same attention, beside the two
// paths it stands in
// for: ink_attend() (a token's scores in floats) and attend_batch() with an AttnBias (cuBLAS products, the path
// Inkling2 takes when ink_attend_batch() does not).  Inkling 2 Mini's heads: 128 values, 16 relative values; groups
// of 1 to 4 query heads, full and window layers, ring caches, a gate, calls that start past position 0.
//
// Then the same rows in one call and split over several: every output the same bits (a token's result does not depend
// on the call it is in).  On Turing it tests ink_mma75_kernel, and ink_flash75_kernel with
// STRATA_INK_MMA75=0: run it both ways there.
//
// usage: poly_ink_flash_test            (exit 0: every case within the bounds below; 77: no flash kernel here)
//                                       each case also prints a hash of the flash kernel's output bits, which two
//                                       builds of the same arithmetic must share, and fails unless the blocks of
//                                       64, 48 and 32 rows (ink_flash_tokens()'s sizes) give the same bits
//        STRATA_FAST_ATTN=1 poly_ink_flash_test   (full layers on ink_fast_kernel: twice the flash bound)
//        poly_ink_flash_test --bench [--rows N] [keys...]
//                                       (ms of one call of 1,024 rows or N, Inkling 2 Mini's shape, full and window
//                                       layers, at 2K/8K/32K keys or the ones given: the size ink_flash_tokens()
//                                       picks, then each of 64, 48 and 32 rows a block)
#include "inkling2_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace strata::poly;

constexpr int kDim = 128, kRel = 16;
// Bounds on the largest error against float64, outputs of order 1.  The flash kernels and attend_batch() round the
// queries and the weights to halves, so they sit near 1e-3; ink_attend() rounds nothing but the caches.
constexpr double kFlashErr = 4e-3, kBatchErr = 4e-3, kTokenErr = 1e-4;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, std::max<size_t>(1, v.size()) * sizeof(T)), "cudaMalloc");
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

float* zeros(size_t n) {
    float* d = nullptr;
    ck(cudaMalloc(&d, n * sizeof(float)), "cudaMalloc");
    ck(cudaMemset(d, 0, n * sizeof(float)), "cudaMemset");
    return d;
}

std::vector<float> download(const float* d, size_t n) {
    std::vector<float> v(n);
    ck(cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost), "download");
    return v;
}

uint16_t half_bits(float x) {
    const __half h = __float2half(x);
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}

float half_val(uint16_t b) {
    __half h;
    std::memcpy(&h, &b, 2);
    return __half2float(h);
}

struct Case {
    const char* name;
    int per;              // query heads on a key head
    int n_kv;
    long long pos;        // the call's first position
    int rows;
    int window;           // 0: a full layer
    int ring;             // 0: a row a position
    int extent;           // distances with a bias column
    bool gate;
};

// One call's inputs, on the host and the card.
struct Call {
    Case c;
    int n_head = 0;
    long long slots = 0;   // cache rows
    std::vector<float> q, r, sink, table, gate;
    std::vector<uint16_t> k, v;
    float *dq = nullptr, *dr = nullptr, *dsink = nullptr, *dtable = nullptr, *dgate = nullptr;
    uint16_t *dk = nullptr, *dv = nullptr;

    explicit Call(const Case& cs, unsigned seed) : c(cs) {
        n_head = c.per * c.n_kv;
        slots = c.ring > 0 ? c.ring : c.pos + c.rows;
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        const auto fill = [&](std::vector<float>& x, size_t n, float sd) {
            x.resize(n);
            for (float& e : x) e = sd * nd(rng);
        };
        // queries and keys of a spread that gives peaked and flat rows; the relative values and sinks of the
        // model's order of size
        fill(q, (size_t) c.rows * n_head * kDim, 1.5f);
        fill(r, (size_t) c.rows * n_head * kRel, 1.0f);
        fill(sink, (size_t) c.rows * n_head, 1.0f);
        fill(table, (size_t) c.extent * kRel, 0.5f);
        if (c.gate) fill(gate, (size_t) c.rows * n_head * kDim, 1.0f);
        std::vector<float> kf, vf;
        fill(kf, (size_t) slots * c.n_kv * kDim, 1.5f);
        fill(vf, (size_t) slots * c.n_kv * kDim, 1.0f);
        k.resize(kf.size());
        v.resize(vf.size());
        for (size_t i = 0; i < kf.size(); ++i) k[i] = half_bits(kf[i]), v[i] = half_bits(vf[i]);
        dq = upload(q), dr = upload(r), dsink = upload(sink), dtable = upload(table), dk = upload(k), dv = upload(v);
        if (c.gate) dgate = upload(gate);
    }
    ~Call() {
        for (void* p : {(void*) dq, (void*) dr, (void*) dsink, (void*) dtable, (void*) dgate, (void*) dk, (void*) dv})
            cudaFree(p);
    }

    InkAttn attn(float* out, float* score, size_t score_n) const {
        InkAttn a;
        a.q = dq, a.r = dr, a.sink = dsink, a.keys = dk, a.vals = dv, a.rel_bias = dtable, a.gate = dgate;
        a.out = out, a.score = score, a.score_n = score_n;
        a.pos = c.pos, a.rows = c.rows, a.n_head = n_head, a.n_kv = c.n_kv, a.dim = kDim, a.n_rel = kRel;
        a.extent = c.extent, a.window = c.window, a.ring = c.ring;
        a.scale = 1.0f / std::sqrt((float) kDim);
        return a;
    }

    // float64 of the attention the kernels make: scores q . k times the scale, plus r . the table's column of
    // the distance (below the extent), less the sink; a softmax with one more term of score 0; then the gate
    std::vector<double> reference() const {
        std::vector<double> out((size_t) c.rows * n_head * kDim, 0.0);
        const double scale = 1.0 / std::sqrt((double) kDim);
        std::vector<double> s, acc(kDim);
        for (int t = 0; t < c.rows; ++t)
            for (int h = 0; h < n_head; ++h) {
                const long long own = c.pos + t, first = c.window > 0 ? std::max(0LL, own - c.window + 1) : 0;
                const size_t th = (size_t) t * n_head + h;
                const int g = h / c.per;
                s.assign((size_t) (own - first + 1), 0.0);
                double top = 0.0;   // the term of score 0
                for (long long key = first; key <= own; ++key) {
                    const size_t at = ((size_t) (c.ring > 0 ? key % c.ring : key) * c.n_kv + g) * kDim;
                    double d = 0.0;
                    for (int i = 0; i < kDim; ++i) d += (double) q[th * kDim + i] * half_val(k[at + i]);
                    double b = 0.0;
                    if (own - key < c.extent)
                        for (int i = 0; i < kRel; ++i) b += (double) r[th * kRel + i] * table[(own - key) * kRel + i];
                    s[key - first] = d * scale + b - sink[th];
                    top = std::max(top, s[key - first]);
                }
                double sum = std::exp(0.0 - top);
                std::fill(acc.begin(), acc.end(), 0.0);
                for (long long key = first; key <= own; ++key) {
                    const double w = std::exp(s[key - first] - top);
                    sum += w;
                    const size_t at = ((size_t) (c.ring > 0 ? key % c.ring : key) * c.n_kv + g) * kDim;
                    for (int i = 0; i < kDim; ++i) acc[i] += w * half_val(v[at + i]);
                }
                for (int i = 0; i < kDim; ++i) {
                    double y = acc[i] / sum;
                    if (c.gate) y /= 1.0 + std::exp(-(double) gate[th * kDim + i]);
                    out[th * kDim + i] = y;
                }
            }
        return out;
    }
};

double max_err(const std::vector<float>& got, const std::vector<double>& want) {
    double e = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs((double) got[i] - want[i]);
        e = std::isfinite(got[i]) ? std::max(e, d) : INFINITY;
    }
    return e;
}

uint64_t fnv(const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(v.data());
    for (size_t i = 0; i < v.size() * sizeof(float); ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0.0;
    for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::fabs((double) a[i] - b[i]));
    return e;
}

constexpr size_t kScoreFloats = size_t(16) << 20;   // inkling2.cpp's

// attend_batch() as Inkling2::attention() calls it when ink_attend_batch() does not take the call
void attend_batch_path(void* blas, const Call& x, float* out, float* score, float* work, float* zero, cudaStream_t st) {
    const InkAttn a = x.attn(out, nullptr, 0);
    AttnBias bias;
    bias.r = x.dr, bias.sink = x.dsink, bias.table = x.dtable, bias.n_rel = kRel, bias.extent = x.c.extent;
    attend_batch(blas, x.dq, x.dk, x.dv, x.c.pos, x.c.rows, score, kScoreFloats, work, out, kDim, x.n_head, x.c.n_kv,
                 a.scale, st, x.c.window, x.c.ring, false, {}, {}, nullptr, nullptr, 0, zero, nullptr, &bias);
    if (x.c.gate) ink_mul_sigmoid(out, x.dgate, (size_t) x.c.rows * x.n_head * kDim, st);
}

size_t work_floats(int rows, int n_head) {   // inkling2.cpp's awork_
    const size_t heads = (size_t) rows * n_head * kDim;
    return heads + heads / 2 + 1 + 3 * (size_t) rows * n_head;
}

// A token's result must not depend on the call it is in: the case's rows in one call, then the same rows (same caches,
// same inputs) in calls of the given sizes, one after another; every output float the same bits.
int run_split_tests(cudaStream_t st) {
    struct Split {
        Case c;
        std::vector<int> sizes;
    };
    const Split splits[] = {
        // name                    per n_kv pos    rows  window ring  extent gate   call sizes
        {{"split window ring g3",   3, 8,   1900,  333,  1024,  1357, 1024,  true},  {1, 100, 232}},
        {{"split window 1K g3",     3, 8,   3072,  1024, 1024,  2048, 1024,  true},  {512, 512}},
        {{"split window 1K g3 odd", 3, 8,   3072,  1024, 1024,  2048, 1024,  true},  {333, 21, 1, 669}},
        {{"split full later g3",    3, 8,   1500,  200,  0,     0,    1024,  false}, {21, 64, 115}},
        {{"split full pos0 g3",     3, 8,   0,     700,  0,     0,    1024,  true},  {300, 17, 383}},
        {{"split window g1",        1, 8,   999,   64,   128,   192,  128,   true},  {5, 59}},
        {{"split full g2",          2, 4,   40,    130,  0,     0,    64,    false}, {17, 113}},
        {{"split full g4",          4, 2,   0,     77,   0,     0,    1024,  false}, {5, 72}},
        {{"split window g2",        2, 4,   300,   150,  128,   278,  128,   true},  {7, 30, 113}},
        {{"split window g4",        4, 2,   500,   90,   100,   190,  100,   false}, {3, 87}},
        {{"split window 20 g3",     3, 8,   64,    70,   20,    90,   20,    true},  {11, 59}},
    };
    int fails = 0;
    for (unsigned si = 0; si < sizeof(splits) / sizeof(splits[0]); ++si) {
        const Split& sp = splits[si];
        Call x(sp.c, 4321u + si);
        const size_t per_row = (size_t) x.n_head * kDim, n_out = (size_t) sp.c.rows * per_row;
        float* out = zeros(n_out);
        InkAttn a = x.attn(out, nullptr, 0);
        if (!ink_attend_batch(a, st)) {
            std::printf("%-24s flash off\n", sp.c.name);
            cudaFree(out);
            continue;
        }
        ck(cudaStreamSynchronize(st), "ink_attend_batch");
        const std::vector<float> whole = download(out, n_out);
        ck(cudaMemset(out, 0, n_out * sizeof(float)), "cudaMemset");
        // the calls last first, so a call that wrote past its rows would leave that behind
        std::vector<int> first;
        int at = 0;
        for (const int m : sp.sizes) first.push_back(at), at += m;
        bool took = true;
        for (size_t k = sp.sizes.size(); k-- > 0;) {
            const int m = sp.sizes[k];
            at = first[k];
            InkAttn b = a;
            b.q = x.dq + (size_t) at * per_row;
            b.r = x.dr + (size_t) at * x.n_head * kRel;
            b.sink = x.dsink + (size_t) at * x.n_head;
            if (x.dgate) b.gate = x.dgate + (size_t) at * per_row;
            b.out = out + (size_t) at * per_row;
            b.pos = sp.c.pos + at;
            b.rows = m;
            took = ink_attend_batch(b, st) && took;
        }
        at = first.back() + sp.sizes.back();
        ck(cudaStreamSynchronize(st), "ink_attend_batch (split)");
        const std::vector<float> parts = download(out, n_out);
        size_t differ = 0;
        for (size_t i = 0; i < n_out; ++i) differ += std::memcmp(&whole[i], &parts[i], 4) != 0;
        const bool ok = took && at == sp.c.rows && differ == 0;
        std::printf("%-24s one call fnv %016llx | %zu calls fnv %016llx | %zu floats differ  %s\n", sp.c.name,
                    (unsigned long long) fnv(whole), sp.sizes.size(), (unsigned long long) fnv(parts), differ,
                    ok ? "ok" : "FAIL");
        fails += !ok;
        cudaFree(out);
    }
    return fails;
}

int run_tests() {
    const Case cases[] = {
        // name                  per n_kv pos    rows  window ring  extent gate
        {"full pos0 g3",          3, 8,   0,     300,  0,     0,    1024,  true},
        {"full pos0 g1",          1, 4,   0,     97,   0,     0,    1024,  false},
        {"full pos0 g2",          2, 4,   0,     130,  0,     0,    64,    true},
        {"full pos0 g4",          4, 2,   0,     77,   0,     0,    1024,  false},
        {"full later g3",         3, 8,   1500,  200,  0,     0,    1024,  true},
        {"full short extent g3",  3, 8,   700,   64,   0,     0,    40,    false},
        {"window pos0 g3",        3, 8,   0,     700,  256,   0,    256,   true},
        {"window ring g3",        3, 8,   1900,  333,  1024,  1357, 1024,  true},
        {"window ring odd g3",    3, 4,   2049,  45,   100,   145,  100,   false},
        {"window ring g1",        1, 8,   999,   64,   128,   192,  128,   true},
        {"few rows g3",           3, 8,   33,    32,   0,     0,    1024,  false},
        {"window ring g2",        2, 4,   700,   150,  128,   278,  128,   true},
        {"window ring g4",        4, 2,   333,   90,   100,   190,  100,   false},
        {"window 20 g3",          3, 8,   64,    70,   20,    90,   20,    true},
        {"prompt chunk full g3",  3, 8,   1024,  1024, 0,     0,    1024,  true},
        {"prompt chunk window g3", 3, 8,  1024,  1024, 1024,  2048, 1024,  true},
    };
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    void* blas = blas_create(st);
    int fails = 0, size_calls = 0, size_same = 0;
    bool ran_flash = false;
    for (unsigned ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
        const Case& cs = cases[ci];
        Call x(cs, 1234u + ci);
        const size_t n_out = (size_t) cs.rows * x.n_head * kDim;
        const size_t score_n = std::max<size_t>(
            kScoreFloats, (size_t) 64 * ink_attend_floats(x.n_head, cs.n_kv, kDim, cs.pos + cs.rows));
        float* out = zeros(n_out);
        float* score = zeros(score_n);
        float* work = zeros(work_floats(cs.rows, x.n_head));
        float* zero = zeros((size_t) x.n_head);
        const std::vector<double> want = x.reference();

        const InkAttn a = x.attn(out, score, score_n);
        const bool flash = ink_attend_batch(a, st);
        ck(cudaStreamSynchronize(st), "ink_attend_batch");
        const std::vector<float> got_flash = download(out, n_out);
        ck(cudaMemset(out, 0, n_out * sizeof(float)), "cudaMemset");
        // every block size the kernel takes: the same bits as the size it picked
        bool same = true;
        std::string sizes;
        if (flash && ink_flash_tokens(a) > 0)   // ink_flash_kernel took it (not Turing's kernel or STRATA_FAST_ATTN's)
            for (const int t : {64, 48, 32}) {
                ck(cudaMemset(out, 0xff, n_out * sizeof(float)), "cudaMemset");   // NaNs where nothing is written
                ink_attend_batch_rows(a, st, t);
                ck(cudaStreamSynchronize(st), "ink_attend_batch");
                const uint64_t h = fnv(download(out, n_out));
                const bool eq = h == fnv(got_flash);
                same = same && eq;
                sizes += " " + std::to_string(t) + (eq ? "=" : "!");
                ++size_calls, size_same += eq;
            }
        ck(cudaMemset(out, 0, n_out * sizeof(float)), "cudaMemset");
        ink_attend(a, st);
        ck(cudaStreamSynchronize(st), "ink_attend");
        const std::vector<float> got_token = download(out, n_out);
        ck(cudaMemset(out, 0, n_out * sizeof(float)), "cudaMemset");
        attend_batch_path(blas, x, out, score, work, zero, st);
        ck(cudaStreamSynchronize(st), "attend_batch");
        const std::vector<float> got_batch = download(out, n_out);

        const double ef = flash ? max_err(got_flash, want) : 0.0, et = max_err(got_token, want),
                     eb = max_err(got_batch, want);
        const char* fv = std::getenv("STRATA_FAST_ATTN");
        const double flash_err = fv && fv[0] == '1' && cs.window <= 0 ? 2 * kFlashErr : kFlashErr;
        const bool ok = (!flash || ef <= flash_err) && et <= kTokenErr && eb <= kBatchErr && same;
        ran_flash = ran_flash || flash;
        std::printf("%-22s flash %s err %.3e | ink_attend err %.3e | attend_batch err %.3e | flash-batch %.3e | "
                    "bits %016llx picked %d, sizes%s  %s\n",
                    cs.name, flash ? "ran" : "off", ef, et, eb, flash ? max_diff(got_flash, got_batch) : 0.0,
                    (unsigned long long) fnv(got_flash), ink_flash_tokens(a), sizes.c_str(),
                    ok ? "ok" : "FAIL");
        fails += !ok;
        for (void* p : {(void*) out, (void*) score, (void*) work, (void*) zero}) cudaFree(p);
    }
    fails += run_split_tests(st);
    // FP8 caches (codes and a scale a head, ink_kv8_row() bytes a row) and Q8_0 caches (int8 codes and a half scale
    // a block of 32, ink_kvq8_row() bytes a row): no reference, the block sizes' bits only, where ink_flash_kernel
    // takes the call (ink_flash_tokens() > 0)
    for (const bool q8 : {false, true})
        for (unsigned ci = 0; ran_flash && ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
            const Case& cs = cases[ci];
            Call x(cs, 4321u + ci);
            std::mt19937 rng(99u + ci);
            const int codes = cs.n_kv * kDim;
            const size_t row = q8 ? ink_kvq8_row(cs.n_kv, kDim) : ink_kv8_row(cs.n_kv, kDim);
            std::vector<uint8_t> k8((size_t) x.slots * row), v8(k8.size());
            for (std::vector<uint8_t>* c : {&k8, &v8})
                for (long long t = 0; t < x.slots; ++t) {
                    uint8_t* r = c->data() + (size_t) t * row;
                    for (int i = 0; i < codes; ++i)   // FP8: no NaN codes (exponent never 1111); Q8_0: -127..127
                        r[i] = q8 ? (uint8_t) (int8_t) ((int) (rng() % 255) - 127) : (uint8_t) ((rng() % 255) & 0xf7);
                    for (int h = 0; h < (q8 ? codes / 32 : cs.n_kv); ++h) {
                        const float sc = 0.002f + 0.004f * (float) (rng() % 1000) / 1000.0f;
                        if (q8) {
                            const uint16_t hb = half_bits(sc * 64.0f / 127.0f);
                            std::memcpy(r + codes + h * sizeof(uint16_t), &hb, sizeof(hb));
                        } else {
                            std::memcpy(r + codes + h * sizeof(float), &sc, sizeof(float));
                        }
                    }
                }
            uint8_t* dk8 = upload(k8);
            uint8_t* dv8 = upload(v8);
            const size_t n_out = (size_t) cs.rows * x.n_head * kDim;
            float* out = zeros(n_out);
            InkAttn a = x.attn(out, nullptr, 0);
            a.keys = reinterpret_cast<const uint16_t*>(dk8), a.vals = reinterpret_cast<const uint16_t*>(dv8);
            (q8 ? a.kvq8 : a.kv8) = true;
            const int picked = ink_flash_tokens(a);
            if (picked > 0) {
                ink_attend_batch(a, st);
                ck(cudaStreamSynchronize(st), "ink_attend_batch");
                const std::vector<float> got = download(out, n_out);
                bool same = true, finite = true;
                for (const float f : got) finite = finite && std::isfinite(f);
                std::string sizes;
                for (const int t : {64, 48, 32}) {
                    ck(cudaMemset(out, 0xff, n_out * sizeof(float)), "cudaMemset");
                    ink_attend_batch_rows(a, st, t);
                    ck(cudaStreamSynchronize(st), "ink_attend_batch");
                    const bool eq = fnv(download(out, n_out)) == fnv(got);
                    same = same && eq;
                    sizes += " " + std::to_string(t) + (eq ? "=" : "!");
                    ++size_calls, size_same += eq;
                }
                const bool ok = same && finite;
                std::printf("%-22s %s caches: bits %016llx picked %d, sizes%s%s  %s\n", cs.name, q8 ? "q8_0" : "fp8 ",
                            (unsigned long long) fnv(got), picked, sizes.c_str(), finite ? "" : " NON-FINITE",
                            ok ? "ok" : "FAIL");
                fails += !ok;
            }
            for (void* p : {(void*) out, (void*) dk8, (void*) dv8}) cudaFree(p);
        }
    blas_destroy(blas);
    cudaStreamDestroy(st);
    std::printf("block sizes: %d / %d calls give the picked size's bits\n", size_same, size_calls);
    std::printf("%d failed\n", fails);
    if (!ran_flash) {
        std::printf("ink_attend_batch() takes no call on this card\n");
        return fails ? 1 : 77;
    }
    return fails ? 1 : 0;
}

// ms of one call of `rows` rows whose last position is keys - 1, Inkling 2 Mini's 24 heads on 8, window layers' 1,024
int run_bench(std::vector<long long> all_keys, int rows) {
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    void* blas = blas_create(st);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    if (all_keys.empty()) all_keys = {2048, 8192, 32768};
    for (const long long keys : all_keys)
        for (const bool swa : {false, true}) {
            const Case cs{swa ? "window" : "full", 3, 8, keys - rows, rows, swa ? 1024 : 0, swa ? 1024 + rows : 0,
                          1024, true};
            Call x(cs, 7u);
            const size_t n_out = (size_t) rows * x.n_head * kDim;
            float* out = zeros(n_out);
            float* score = zeros(kScoreFloats);
            float* work = zeros(work_floats(rows, x.n_head));
            float* zero = zeros((size_t) x.n_head);
            const InkAttn a = x.attn(out, score, kScoreFloats);
            const auto time = [&](auto&& fn) {
                fn();
                ck(cudaStreamSynchronize(st), "warm-up");
                float best = 1e30f;
                for (int rep = 0; rep < 5; ++rep) {
                    cudaEventRecord(e0, st);
                    fn();
                    cudaEventRecord(e1, st);
                    ck(cudaEventSynchronize(e1), "bench");
                    float ms = 0.0f;
                    cudaEventElapsedTime(&ms, e0, e1);
                    best = std::min(best, ms);
                }
                return best;
            };
            bool flash = true;
            const float tf = time([&] { flash = ink_attend_batch(a, st); });
            const float tb = time([&] { attend_batch_path(blas, x, out, score, work, zero, st); });
            std::printf("%-6s keys %6lld rows %d: flash %s %8.3f ms (picked %d) | attend_batch %8.3f ms\n", cs.name,
                        keys, rows, flash ? "   " : "off", flash ? tf : 0.0f, ink_flash_tokens(a), tb);
            if (flash && ink_flash_tokens(a) > 0)
                for (const int t : {64, 48, 32}) {
                    const float ms = time([&] { ink_attend_batch_rows(a, st, t); });
                    std::printf("%-6s keys %6lld rows %d: flash tokens %d %8.3f ms\n", cs.name, keys, rows, t, ms);
                }
            for (void* p : {(void*) out, (void*) score, (void*) work, (void*) zero}) cudaFree(p);
        }
    blas_destroy(blas);
    cudaStreamDestroy(st);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp p{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        std::printf("no card\n");
        return 77;
    }
    std::printf("%s (sm_%d%d)\n", p.name, p.major, p.minor);
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        std::vector<long long> keys;
        int rows = 1024;
        for (int i = 2; i < argc; ++i)
            if (std::string(argv[i]) == "--rows" && i + 1 < argc) rows = std::atoi(argv[++i]);
            else keys.push_back(std::atoll(argv[i]));
        return run_bench(keys, rows);
    }
    return run_tests();
}
