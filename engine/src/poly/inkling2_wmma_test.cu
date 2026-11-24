// poly_inkling2_wmma_test - ink_attend_batch() on RDNA4's matrix cores (ink_wmma_kernel) against an FP64 reference
// of Inkling 2's prompt attention (the bias by distance, the sink, the window ring, the gate), against ink_attend()'s
// score / softmax / value / join kernels, and a token's result the same to the bit in calls that start at other
// tokens.  No GGUF or model.  77: not a gfx12 card (the kernel is not taken there).
#include "inkling2_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strata::poly;

constexpr int kHeads = 24, kKv = 8, kDim = 128, kRel = 16, kExtent = 200;
int failures = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

std::vector<float> randv(size_t n, float scale, std::mt19937& g) {
    std::uniform_real_distribution<float> u(-scale, scale);
    std::vector<float> v(n);
    for (float& x : v) x = u(g);
    return v;
}

template <class T> T* up(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, v.size() * sizeof(T)), "malloc");
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

std::vector<float> down(const float* d, size_t n) {
    std::vector<float> v(n);
    ck(cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost), "download");
    return v;
}

struct Case {
    const char* name;
    int64_t ctx;   // positions before the call's first token
    int rows, window, ring;
    bool gate;
};

// The attention of token t (position ctx + t) in doubles: the keys it sees (a window's last `window` or all), the
// scale on q . k, plus R . the table's row of the distance (below the extent), the sink an extra term of the softmax.
void reference(const Case& c, const std::vector<float>& q, const std::vector<float>& r, const std::vector<float>& sink,
               const std::vector<float>& table, const std::vector<__half>& kc, const std::vector<__half>& vc,
               const std::vector<float>& gate, int t, std::vector<double>& out) {
    const double scale = 1.0 / std::sqrt((double) kDim);
    const int64_t p = c.ctx + t, lo = c.window > 0 ? std::max<int64_t>(0, p - c.window + 1) : 0;
    std::vector<double> s((size_t) (p - lo + 1));
    for (int h = 0; h < kHeads; ++h) {
        const int g = h / (kHeads / kKv);
        const float* qh = q.data() + ((size_t) t * kHeads + h) * kDim;
        const float* rh = r.data() + ((size_t) t * kHeads + h) * kRel;
        double mx = sink[(size_t) t * kHeads + h];
        for (int64_t kp = lo; kp <= p; ++kp) {
            const int64_t row = c.ring > 0 ? kp % c.ring : kp;
            const __half* k = kc.data() + ((size_t) row * kKv + g) * kDim;
            double d = 0;
            for (int i = 0; i < kDim; ++i) d += (double) qh[i] * (double) __half2float(k[i]);
            d *= scale;
            const int64_t dist = p - kp;
            if (dist < kExtent)
                for (int j = 0; j < kRel; ++j) d += (double) rh[j] * table[(size_t) dist * kRel + j];
            s[(size_t) (kp - lo)] = d;
            mx = std::max(mx, d);
        }
        double sum = std::exp(sink[(size_t) t * kHeads + h] - mx);
        for (double& x : s) sum += (x = std::exp(x - mx));
        double* o = out.data() + ((size_t) t * kHeads + h) * kDim;
        for (int i = 0; i < kDim; ++i) o[i] = 0;
        for (int64_t kp = lo; kp <= p; ++kp) {
            const int64_t row = c.ring > 0 ? kp % c.ring : kp;
            const __half* v = vc.data() + ((size_t) row * kKv + g) * kDim;
            const double w = s[(size_t) (kp - lo)] / sum;
            for (int i = 0; i < kDim; ++i) o[i] += w * (double) __half2float(v[i]);
        }
        if (c.gate)
            for (int i = 0; i < kDim; ++i)
                o[i] /= 1.0 + std::exp(-(double) gate[((size_t) t * kHeads + h) * kDim + i]);
    }
}

void run(const Case& c, std::mt19937& gen) {
    const int64_t n_pos = c.ctx + c.rows, cache_rows = c.ring > 0 ? c.ring : n_pos;
    // the cache: random values, rounded to halves as the stores leave them; a ring holds the last ring positions
    std::vector<__half> kc((size_t) cache_rows * kKv * kDim), vc(kc.size());
    for (size_t i = 0; i < kc.size(); ++i) {
        kc[i] = __float2half(std::uniform_real_distribution<float>(-1.2f, 1.2f)(gen));
        vc[i] = __float2half(std::uniform_real_distribution<float>(-1.0f, 1.0f)(gen));
    }
    const auto q = randv((size_t) c.rows * kHeads * kDim, 1.2f, gen), r = randv((size_t) c.rows * kHeads * kRel, 1.5f, gen),
               sink = randv((size_t) c.rows * kHeads, 2.0f, gen), table = randv((size_t) kExtent * kRel, 0.8f, gen),
               gate = randv((size_t) c.rows * kHeads * kDim, 3.0f, gen);
    __half *dk = up(kc), *dv = up(vc);
    float *dq = up(q), *dr = up(r), *ds = up(sink), *dt = up(table), *dg = up(gate), *o1 = nullptr, *o2 = nullptr,
          *o3 = nullptr, *score = nullptr;
    const size_t n_out = q.size();
    ck(cudaMalloc(&o1, n_out * 4), "malloc"), ck(cudaMalloc(&o2, n_out * 4), "malloc"), ck(cudaMalloc(&o3, n_out * 4), "malloc");
    const int nk = (int) (c.window > 0 ? std::min<int64_t>(c.window, n_pos) : n_pos);
    const size_t score_n = ink_attend_floats(kHeads, kKv, kDim, nk) * 8;
    ck(cudaMalloc(&score, score_n * 4), "malloc");

    InkAttn a;
    a.q = dq, a.r = dr, a.sink = ds, a.keys = (const uint16_t*) dk, a.vals = (const uint16_t*) dv, a.rel_bias = dt;
    a.gate = c.gate ? dg : nullptr, a.out = o1, a.score = score, a.score_n = score_n, a.pos = c.ctx, a.rows = c.rows;
    a.n_head = kHeads, a.n_kv = kKv, a.dim = kDim, a.n_rel = kRel, a.extent = kExtent, a.window = c.window;
    a.ring = c.ring, a.scale = 1.f / std::sqrt((float) kDim);
    if (!ink_attend_batch(a, nullptr)) throw std::logic_error("ink_attend_batch() did not take the call");
    // the same tokens from the 23rd on, in a call of their own
    const int shift = std::min(23, c.rows - 1);
    InkAttn b = a;
    const size_t off = (size_t) shift * kHeads;
    b.q = dq + off * kDim, b.r = dr + off * kRel, b.sink = ds + off, b.gate = c.gate ? dg + off * kDim : nullptr;
    b.out = o2, b.pos = c.ctx + shift, b.rows = c.rows - shift;
    if (!ink_attend_batch(b, nullptr)) throw std::logic_error("ink_attend_batch() did not take the call");
    // ink_attend()'s kernels, a token's scores written out (the few tokens' path), with the gate in the join
    InkAttn s = a;
    s.out = o3;
    ink_attend(s, nullptr);
    ck(cudaDeviceSynchronize(), "attend");
    const auto x = down(o1, n_out), y = down(o2, n_out - off * kDim), z = down(o3, n_out);

    // the reference at some tokens: the first, the last, and every 7th
    std::vector<double> ref(n_out);
    double worst = 0, worst_s = 0;
    int checked = 0;
    for (int t = 0; t < c.rows; ++t) {
        if (t != 0 && t != c.rows - 1 && t % 7 != 0) continue;
        reference(c, q, r, sink, table, kc, vc, gate, t, ref);
        ++checked;
        for (size_t i = (size_t) t * kHeads * kDim; i < (size_t) (t + 1) * kHeads * kDim; ++i) {
            worst = std::max(worst, std::fabs((double) x[i] - ref[i]));
            worst_s = std::max(worst_s, std::fabs((double) z[i] - ref[i]));
        }
    }
    const double bound = 2e-5;   // what ink_attend()'s float kernels meet
    std::printf("%-44s %d tokens checked  wmma max abs %.3e  score path %.3e  %s\n", c.name, checked, worst, worst_s,
                worst <= bound ? "ok" : "FAIL");
    if (worst > bound) ++failures;
    size_t diff = 0;
    for (size_t i = 0; i < y.size(); ++i) diff += x[off * kDim + i] != y[i];
    std::printf("%-44s %zu of %zu values differ in 2 calls  %s\n", c.name, diff, y.size(), diff ? "FAIL" : "ok");
    if (diff) ++failures;
    for (void* p : {(void*) dk, (void*) dv, (void*) dq, (void*) dr, (void*) ds, (void*) dt, (void*) dg, (void*) o1,
                    (void*) o2, (void*) o3, (void*) score})
        (void) cudaFree(p);
}
}  // namespace

int main() {
    try {
        cudaDeviceProp p{};
        ck(cudaGetDeviceProperties(&p, 0), "device");
        std::printf("device: %s (%s)\n", p.name, p.gcnArchName);
        std::mt19937 gen(1234);
        const Case cases[] = {
            {"full, 107 tokens at 0", 0, 107, 0, 0, false},
            {"full, 130 tokens after 3000, gate", 3000, 130, 0, 0, true},
            {"full, 17 tokens after 5", 5, 17, 0, 0, false},
            {"window 1024, ring 3072, 130 after 3000", 3000, 130, 1024, 3072, false},
            {"window 1024, ring 1030, wrap, 300 after 900", 900, 300, 1024, 1030, true},
            {"window 512, ring 2560, 2048 after 2056", 2056, 2048, 512, 2560, false},
            {"full, 1100 tokens at 0", 0, 1100, 0, 0, true},
        };
        for (const Case& c : cases) run(c, gen);
    } catch (const std::logic_error& e) {
        std::printf("skipped: %s (not an RDNA4 card, or STRATA_INK_FLASH=0)\n", e.what());
        return 77;
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
    std::printf(failures ? "FAILED: %d failures\n" : "PASSED\n", failures);
    return failures ? 1 : 0;
}
