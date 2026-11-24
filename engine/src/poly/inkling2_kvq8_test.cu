// src/poly/inkling2_kvq8_test.cu - Inkling 2's Q8_0 attention caches (--kv q8_0): the stores' codes and scales
// against llama.cpp's quantize_row_q8_0 rule on the host, and every reader of them against a double attention over
// the dequantized rows on the host: ink_attend() (a decoded token), attend_batch() (a prompt batch on the cuBLAS
// products, its rows dequantized a block at a time) and ink_attend_batch() (the flash kernel, Ampere on), full and
// windowed with a ring.  attend_batch() must also give the same bits on the Q8_0 cache as on an fp16 cache of the
// dequantized halves when its workspace holds every row.
//
// usage: poly_inkling2_kvq8_test   (exit 0: codes equal, readers within their tolerances, bits equal)
#include "inkling2_kernels.hpp"
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

constexpr int kHead = 32, kKv = 8, kDim = 128, kRel = 16, kExtent = 96, kKeys = 2304, kWindow = 512;
constexpr int kRing = kWindow + 512;   // the engine's ring: the window and a batch's rows
constexpr float kEps = 1e-6f;

int failures = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, v.size() * sizeof(T)), "cudaMalloc");
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

template <typename T> std::vector<T> download(const T* d, size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

float half_f(uint16_t h) {
    __half x;
    std::memcpy(&x, &h, 2);
    return __half2float(x);
}

uint16_t f_half(float f) {
    const __half x = __float2half(f);
    uint16_t h;
    std::memcpy(&h, &x, 2);
    return h;
}

const size_t kRow = ink_kvq8_row(kKv, kDim), kCodes = (size_t) kKv * kDim;

// A cache of `cells` rows: rows of raw values through ink_clips()' head RMS (so as V is stored), into cells
// pos + t (mod ring).  Returns the normed values the store read, as the card left them.
std::vector<float> store(uint8_t* cache, const std::vector<float>& raw, long long pos, int rows, int ring) {
    float* x = upload(raw);
    InkClips c;
    c.x[2] = x;
    c.dim[2] = kDim;
    c.heads[2] = kKv;
    c.rms[2] = true;
    c.eps = kEps;
    c.pos = pos;
    c.rows = rows;
    c.cache[2] = (uint16_t*) cache;
    c.ring = ring;
    c.kvq8 = true;
    ink_clips(c, nullptr);
    ck(cudaDeviceSynchronize(), "ink_clips");
    std::vector<float> normed = download(x, raw.size());
    (void) cudaFree(x);
    return normed;
}

// llama.cpp's quantize_row_q8_0: a block's scale its largest magnitude over 127, a code round(x / scale)
void quantize_row(const float* x, int8_t* codes, uint16_t* scales) {
    for (size_t b = 0; b < kCodes / 32; ++b) {
        float a = 0.0f;
        for (int i = 0; i < 32; ++i) a = std::max(a, std::fabs(x[b * 32 + i]));
        const float d = a / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        for (int i = 0; i < 32; ++i) codes[b * 32 + i] = (int8_t) std::roundf(x[b * 32 + i] * id);
        scales[b] = f_half(d);
    }
}

// The cache's rows `cells` as the host makes them from `normed` (row t to cell (pos + t) mod ring), compared byte
// for byte; returns the dequantized floats a cell.
std::vector<float> check_store(const char* name, const uint8_t* dcache, const std::vector<float>& normed,
                               long long pos, int rows, int cells, int ring) {
    const std::vector<uint8_t> got = download(dcache, (size_t) cells * kRow);
    std::vector<uint8_t> want(got.size(), 0);
    for (int t = 0; t < rows; ++t) {
        const long long cell = ring > 0 ? (pos + t) % ring : pos + t;
        uint8_t* r = want.data() + (size_t) cell * kRow;
        quantize_row(normed.data() + (size_t) t * kCodes, (int8_t*) r, (uint16_t*) (r + kCodes));
    }
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) bad += got[i] != want[i];
    std::printf("%-34s %zu of %zu bytes differ  %s\n", name, bad, got.size(), bad ? "FAIL" : "ok");
    failures += bad != 0;
    std::vector<float> deq((size_t) cells * kCodes);
    for (int cell = 0; cell < cells; ++cell) {
        const uint8_t* r = got.data() + (size_t) cell * kRow;
        for (size_t i = 0; i < kCodes; ++i)
            deq[(size_t) cell * kCodes + i] = (float) ((const int8_t*) r)[i] * half_f(((const uint16_t*) (r + kCodes))[i / 32]);
    }
    return deq;
}

struct Case {
    std::vector<float> q, r, sink, bias;
    float scale = 1.0f / std::sqrt((float) kDim);
};

// The attention of `rows` tokens from pos in double over the dequantized cells (key cell p or p mod ring).
std::vector<float> reference(const Case& c, const std::vector<float>& k, const std::vector<float>& v, long long pos,
                             int rows, int window, int ring) {
    std::vector<float> out((size_t) rows * kHead * kDim);
    const int per = kHead / kKv;
    std::vector<double> s, acc(kDim);
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < kHead; ++h) {
            const long long p = pos + t, lo = window > 0 ? std::max(0LL, p - window + 1) : 0;
            const float* q = c.q.data() + ((size_t) t * kHead + h) * kDim;
            const float* r = c.r.data() + ((size_t) t * kHead + h) * kRel;
            s.assign((size_t) (p - lo + 1), 0.0);
            double m = 0.0;
            for (long long j = lo; j <= p; ++j) {
                const long long cell = ring > 0 ? j % ring : j;
                const float* key = k.data() + (size_t) cell * kCodes + (size_t) (h / per) * kDim;
                double d = 0.0;
                for (int i = 0; i < kDim; ++i) d += (double) q[i] * c.scale * key[i];
                if (p - j < kExtent)
                    for (int i = 0; i < kRel; ++i) d += (double) r[i] * c.bias[(size_t) (p - j) * kRel + i];
                d -= c.sink[(size_t) t * kHead + h];
                s[(size_t) (j - lo)] = d;
                m = std::max(m, d);
            }
            double sum = std::exp(-m);
            std::fill(acc.begin(), acc.end(), 0.0);
            for (long long j = lo; j <= p; ++j) {
                const long long cell = ring > 0 ? j % ring : j;
                const float* val = v.data() + (size_t) cell * kCodes + (size_t) (h / per) * kDim;
                const double w = std::exp(s[(size_t) (j - lo)] - m);
                sum += w;
                for (int i = 0; i < kDim; ++i) acc[i] += w * val[i];
            }
            for (int i = 0; i < kDim; ++i) out[((size_t) t * kHead + h) * kDim + i] = (float) (acc[i] / sum);
        }
    return out;
}

void compare(const char* name, const std::vector<float>& got, const std::vector<float>& want, double tol) {
    double se = 0.0, sw = 0.0, worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double e = (double) got[i] - want[i];
        se += e * e;
        sw += (double) want[i] * want[i];
        worst = std::max(worst, std::fabs(e));
    }
    const double rel = std::sqrt(se / std::max(sw, 1e-30));
    const bool ok = rel <= tol && std::isfinite(rel);
    std::printf("%-34s rel rms %.3e (tol %.0e), max abs %.3e  %s\n", name, rel, tol, worst, ok ? "ok" : "FAIL");
    failures += !ok;
}

struct Dev {
    float *q, *r, *sink, *bias, *out, *score, *work, *zeros;
    size_t score_n;
};

InkAttn attn_of(const Dev& d, const uint8_t* keys, const uint8_t* vals, long long pos, int rows, int window, int ring,
                const Case& c) {
    InkAttn a;
    a.q = d.q;
    a.r = d.r;
    a.sink = d.sink;
    a.keys = (const uint16_t*) keys;
    a.vals = (const uint16_t*) vals;
    a.rel_bias = d.bias;
    a.out = d.out;
    a.score = d.score;
    a.score_n = d.score_n;
    a.pos = pos;
    a.rows = rows;
    a.n_head = kHead;
    a.n_kv = kKv;
    a.dim = kDim;
    a.n_rel = kRel;
    a.extent = kExtent;
    a.window = window;
    a.ring = ring;
    a.scale = c.scale;
    a.kvq8 = true;
    return a;
}

// attend_batch() as Inkling2::attn() calls it: q8 (Kvq8 of the rows' codes and scales, work rows of workspace), or
// fp16 caches (kh, vh) where those are given
std::vector<float> batch(void* blas, const Dev& d, const uint8_t* keys, const uint8_t* vals, const uint16_t* kh,
                         const uint16_t* vh, uint16_t* work, size_t work_rows, long long pos, int rows, int window,
                         int ring, const Case& c) {
    AttnBias bias;
    bias.r = d.r;
    bias.sink = d.sink;
    bias.table = d.bias;
    bias.n_rel = kRel;
    bias.extent = kExtent;
    const bool q8 = kh == nullptr;
    const auto kv = [&](const uint8_t* p) {
        return q8 ? Kvq8{(const int8_t*) p, (const uint16_t*) (p + kCodes), kRow, kRow / 2} : Kvq8{};
    };
    const size_t halves = work_rows * kCodes;
    attend_batch(blas, d.q, kh, vh, pos, rows, d.score, d.score_n, d.work, d.out, kDim, kHead, kKv, c.scale, nullptr,
                 window, ring, false, kv(keys), kv(vals), q8 ? work : nullptr, q8 ? work + halves : nullptr,
                 q8 ? halves : 0, d.zeros, nullptr, &bias);
    ck(cudaDeviceSynchronize(), "attend_batch");
    return download(d.out, (size_t) rows * kHead * kDim);
}

}  // namespace

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.05f, 4.0f);
    // raw rows whose 32-value blocks differ in size, a few with one large value, so that the scales spread
    const auto raw_rows = [&](int rows) {
        std::vector<float> x((size_t) rows * kCodes);
        for (size_t b = 0; b < x.size() / 32; ++b) {
            const float s = ud(rng);
            for (int i = 0; i < 32; ++i) x[b * 32 + i] = s * nd(rng);
            if (b % 7 == 0) x[b * 32 + b % 32] *= 9.0f;
        }
        return x;
    };

    // full caches: every cell; window caches: a ring of kRing cells holding the last kRing positions
    uint8_t *kf, *vf, *kr, *vr;
    ck(cudaMalloc(&kf, (size_t) kKeys * kRow), "cudaMalloc");
    ck(cudaMalloc(&vf, (size_t) kKeys * kRow), "cudaMalloc");
    ck(cudaMalloc(&kr, (size_t) kRing * kRow), "cudaMalloc");
    ck(cudaMalloc(&vr, (size_t) kRing * kRow), "cudaMalloc");
    const std::vector<float> nk = store(kf, raw_rows(kKeys), 0, kKeys, 0);
    const std::vector<float> nv = store(vf, raw_rows(kKeys), 0, kKeys, 0);
    const std::vector<float> dk = check_store("store, full", kf, nk, 0, kKeys, kKeys, 0);
    const std::vector<float> dv = check_store("store, full (values)", vf, nv, 0, kKeys, kKeys, 0);
    const long long rpos = kKeys - kRing;
    const std::vector<float> nkr = store(kr, raw_rows(kRing), rpos, kRing, kRing);
    const std::vector<float> nvr = store(vr, raw_rows(kRing), rpos, kRing, kRing);
    const std::vector<float> dkr = check_store("store, ring", kr, nkr, rpos, kRing, kRing, kRing);
    const std::vector<float> dvr = check_store("store, ring (values)", vr, nvr, rpos, kRing, kRing, kRing);

    constexpr int kRows = 512;   // a prompt batch's tokens, the last kRows positions
    Case c;
    c.q.resize((size_t) kRows * kHead * kDim);
    c.r.resize((size_t) kRows * kHead * kRel);
    c.sink.resize((size_t) kRows * kHead);
    c.bias.resize((size_t) kExtent * kRel);
    for (float& x : c.q) x = 1.5f * nd(rng);
    for (float& x : c.r) x = nd(rng);
    for (float& x : c.sink) x = 0.5f * nd(rng);
    for (float& x : c.bias) x = 0.3f * nd(rng);
    Dev d;
    d.q = upload(c.q);
    d.r = upload(c.r);
    d.sink = upload(c.sink);
    d.bias = upload(c.bias);
    d.zeros = upload(std::vector<float>(kHead, 0.0f));
    ck(cudaMalloc(&d.out, c.q.size() * sizeof(float)), "cudaMalloc");
    d.score_n = std::max(size_t(16) << 20, ink_attend_floats(kHead, kKv, kDim, kKeys));
    ck(cudaMalloc(&d.score, d.score_n * sizeof(float)), "cudaMalloc");
    const size_t heads = (size_t) kRows * kHead * kDim;
    ck(cudaMalloc(&d.work, (heads + heads / 2 + 1 + 3 * (size_t) kRows * kHead) * sizeof(float)), "cudaMalloc");

    // a decoded token and a window's few: ink_attend()
    for (int rows : {1, 4}) {
        const long long pos = kKeys - rows;
        const size_t q0 = (size_t) (kRows - rows);   // the case's last `rows` tokens
        Dev dt = d;
        dt.q = d.q + q0 * kHead * kDim, dt.r = d.r + q0 * kHead * kRel, dt.sink = d.sink + q0 * kHead;
        Case ct;
        ct.q.assign(c.q.begin() + (long) (q0 * kHead * kDim), c.q.end());
        ct.r.assign(c.r.begin() + (long) (q0 * kHead * kRel), c.r.end());
        ct.sink.assign(c.sink.begin() + (long) (q0 * kHead), c.sink.end());
        ct.bias = c.bias;
        char name[64];
        ink_attend(attn_of(dt, kf, vf, pos, rows, 0, 0, ct), nullptr);
        ck(cudaDeviceSynchronize(), "ink_attend");
        std::snprintf(name, sizeof name, "ink_attend, full, %d rows", rows);
        compare(name, download(d.out, (size_t) rows * kHead * kDim), reference(ct, dk, dv, pos, rows, 0, 0), 1e-5);
        ink_attend(attn_of(dt, kr, vr, pos, rows, kWindow, kRing, ct), nullptr);
        ck(cudaDeviceSynchronize(), "ink_attend");
        std::snprintf(name, sizeof name, "ink_attend, window, %d rows", rows);
        compare(name, download(d.out, (size_t) rows * kHead * kDim),
                reference(ct, dkr, dvr, pos, rows, kWindow, kRing), 1e-5);
    }

    // a prompt batch: attend_batch() (the queries and weights as halves), and the flash kernel where it runs
    void* blas = blas_create(nullptr);
    const long long pos = kKeys - kRows;
    const std::vector<float> want_full = reference(c, dk, dv, pos, kRows, 0, 0);
    const std::vector<float> want_ring = reference(c, dkr, dvr, pos, kRows, kWindow, kRing);
    uint16_t* work;
    ck(cudaMalloc(&work, 2 * (size_t) kKeys * kCodes * sizeof(uint16_t)), "cudaMalloc");
    const std::vector<float> all = batch(blas, d, kf, vf, nullptr, nullptr, work, kKeys, pos, kRows, 0, 0, c);
    compare("attend_batch, full", all, want_full, 1e-2);
    compare("attend_batch, full, 300-row work",
            batch(blas, d, kf, vf, nullptr, nullptr, work, 300, pos, kRows, 0, 0, c), want_full, 1e-2);
    compare("attend_batch, window",
            batch(blas, d, kr, vr, nullptr, nullptr, work, kRing, pos, kRows, kWindow, kRing, c), want_ring, 1e-2);
    compare("attend_batch, window, 300-row work",
            batch(blas, d, kr, vr, nullptr, nullptr, work, 300, pos, kRows, kWindow, kRing, c), want_ring, 1e-2);
    {   // the same bits as an fp16 cache of the dequantized halves
        std::vector<uint16_t> hk(dk.size()), hv(dv.size());
        for (size_t i = 0; i < dk.size(); ++i) hk[i] = f_half(dk[i]), hv[i] = f_half(dv[i]);
        uint16_t* kh = upload(hk);
        uint16_t* vh = upload(hv);
        const std::vector<float> f16 = batch(blas, d, nullptr, nullptr, kh, vh, nullptr, 0, pos, kRows, 0, 0, c);
        const bool same = std::memcmp(f16.data(), all.data(), f16.size() * sizeof(float)) == 0;
        std::printf("%-34s %s\n", "attend_batch, q8 = fp16 of halves", same ? "same bits  ok" : "DIFF  FAIL");
        failures += !same;
        (void) cudaFree(kh);
        (void) cudaFree(vh);
    }
    if (ink_attend_batch(attn_of(d, kf, vf, pos, kRows, 0, 0, c), nullptr)) {
        ck(cudaDeviceSynchronize(), "ink_attend_batch");
        compare("ink_attend_batch, full", download(d.out, heads), want_full, 1e-2);
        ink_attend_batch(attn_of(d, kr, vr, pos, kRows, kWindow, kRing, c), nullptr);
        ck(cudaDeviceSynchronize(), "ink_attend_batch");
        compare("ink_attend_batch, window", download(d.out, heads), want_ring, 1e-2);
    } else {
        std::printf("%-34s not on this card (Ampere on)\n", "ink_attend_batch");
    }
    blas_destroy(blas);
    std::printf("%s: %d failed\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
