// src/poly/inkling2_test.cu - Inkling 2's kernels (inkling2_kernels.hpp, inkling2.cu) against the same arithmetic on
// the host in double precision, on random inputs of Inkling 2 Mini's shapes (2,304 wide, 24 query heads on 8 key
// heads of 128, 16 relative values, a window of 1,024, 4 taps, a loop rank of 64, 128 experts with 8 a token): the
// clips and the head norms, the short convolutions (with the input-dependent taps, their state and snapshots, the
// stream's updates), the depth mixes and their norms, the loop controllers, the gates, SiTU, the router, the output's
// soft cap, the cache stores (fp16, and FP8 E4M3 where the build has it) and the attention over them, window (a ring)
// and full, a token at a time, a few, and a prompt batch (attend_batch() with its bias, and ink_attend_batch() on the
// tensor cores where the card has it).  Where a launch is documented as "the same bits" as the calls it stands for
// (the fused launches, the positions read from the card, a call split into chunks), the two are compared byte for
// byte as well.
//
// usage: poly_inkling2_test [--list] [NAME...]   (exit 0: every check passed; 1: a check failed or none ran;
//   2: no case of that NAME; 77: no card)
//   NAME: run only the cases of that name (--list lists them).  A case is a function in kCases below: add new ones there.
#include "inkling2_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strata::poly;
namespace nk = strata::kernels;

// ---- Inkling 2 Mini's shapes (Inkling-2-Mini-*.gguf, inkling2_loop.*) ----
constexpr int kEmbd = 2304, kHeads = 24, kKv = 8, kDim = 128, kRel = 16, kWindow = 1024, kTaps = 4, kRank = 64;
constexpr int kExperts = 128, kUsed = 8, kExtent = 1024;
constexpr float kEps = 1e-6f, kQkClip = 22.627416610717773f, kRelClip = 8.0f, kAlpha = 0.1f, kFloor = 8192.f;

int failures = 0, checks = 0, codegen_failures = 0;
std::mt19937 rng(20261123);

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// A stream of the test's own (native_mmvq.cu's calls refuse the legacy one).
void* stream() {
    static cudaStream_t s = [] {
        cudaStream_t t = nullptr;
        ck(cudaStreamCreateWithFlags(&t, cudaStreamNonBlocking), "stream");
        return t;
    }();
    return s;
}

void sync() { ck(cudaStreamSynchronize((cudaStream_t) stream()), "sync"); }

// A device buffer of n T's.
template <typename T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    Dev() = default;
    explicit Dev(size_t count) : n(count) {
        ck(cudaMalloc((void**) &p, std::max<size_t>(1, n) * sizeof(T)), "cudaMalloc");
        // on the legacy stream, which the test's own does not wait for: done before anything is launched on it
        ck(cudaMemset(p, 0, std::max<size_t>(1, n) * sizeof(T)), "cudaMemset");
        ck(cudaDeviceSynchronize(), "cudaMemset");
    }
    explicit Dev(const std::vector<T>& v) : Dev(v.size()) { put(v); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    Dev(Dev&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; }
    ~Dev() {
        if (p) (void) cudaFree(p);
    }
    void put(const std::vector<T>& v) {
        ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
        ck(cudaDeviceSynchronize(), "upload");
    }
    std::vector<T> get() const {
        sync();
        std::vector<T> v(n);
        ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return v;
    }
    operator T*() const { return p; }
};

std::vector<float> randv(size_t n, float sd, float mean = 0.f) {
    std::normal_distribution<float> d(mean, sd);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}

std::vector<double> dbl(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

// BF16 weights: random values rounded to BF16, as bits, and their widened floats for the reference
void bf16_of(const std::vector<float>& f, std::vector<uint16_t>& bits, std::vector<float>& wide) {
    bits.resize(f.size());
    wide.resize(f.size());
    for (size_t i = 0; i < f.size(); ++i) {
        uint32_t u;
        std::memcpy(&u, &f[i], 4);
        u += 0x7fff + ((u >> 16) & 1);
        bits[i] = (uint16_t) (u >> 16);
        const uint32_t w = uint32_t(bits[i]) << 16;
        std::memcpy(&wide[i], &w, 4);
    }
}

// ---- checks ----

// |got - ref| in units in the last place of the float nearest ref
double ulps(float got, double ref) {
    const float r = std::fabs((float) ref);
    const double unit = (double) std::nextafter(r, INFINITY) - r;
    return std::fabs((double) got - ref) / unit;
}

// max |got - ref| against tol (absolute), and the most units in the last place it is off by; a NaN or an infinity
// in got fails.
void compare(const std::string& what, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
    ++checks;
    double worst = 0, peak = 0, worst_ulp = 0;
    size_t at = 0;
    bool bad = got.size() != ref.size();
    for (size_t i = 0; i < ref.size() && i < got.size(); ++i) {
        if (!std::isfinite(got[i]) && std::isfinite(ref[i])) bad = true;
        const double e = std::fabs((double) got[i] - ref[i]);
        if (e > worst || std::isnan(e)) worst = e, at = i;
        peak = std::max(peak, std::fabs(ref[i]));
        if (ref[i] != 0) worst_ulp = std::max(worst_ulp, ulps(got[i], ref[i]));
    }
    const bool ok = !bad && worst <= tol;
    if (!ok) ++failures;
    std::printf("%-66s n=%-8zu max err %.2e (tol %.0e, |ref| max %.2g, %.1f ulp)  %s", what.c_str(), ref.size(), worst,
                tol, peak, worst_ulp, ok ? "ok" : "FAIL");
    if (!ok && at < got.size()) std::printf("  [at %zu: got %.8g ref %.8g]", at, (double) got[at], ref[at]);
    std::printf("\n");
}

// byte for byte
template <typename T>
void same_bits(const std::string& what, const std::vector<T>& a, const std::vector<T>& b) {
    ++checks;
    size_t diff = 0, first = 0;
    const size_t bytes = std::min(a.size(), b.size()) * sizeof(T);
    const auto* pa = (const uint8_t*) a.data();
    const auto* pb = (const uint8_t*) b.data();
    for (size_t i = 0; i < bytes; ++i)
        if (pa[i] != pb[i] && diff++ == 0) first = i;
    const bool ok = a.size() == b.size() && diff == 0;
    if (!ok) ++failures;
    std::printf("%-66s %zu bytes  %s", what.c_str(), bytes, ok ? "same bits" : "FAIL: bits differ");
    if (!ok) std::printf("  [%zu bytes differ, first at %zu]", diff, first);
    std::printf("\n");
}

// Byte for byte against a host reference rounded as nvcc compiles the kernel (h_conv_f32() and the like): the bits
// follow the compiler's choice of FMAs and roundings, not the arithmetic, so another nvcc or hipcc can differ here
// while every check against the double-precision reference passes.  Still a check that counts, labelled "[codegen]"
// so that such a difference is told apart from a wrong result.
template <typename T>
void codegen_bits(const std::string& what, const std::vector<T>& a, const std::vector<T>& b) {
    const int before = failures;
    same_bits("[codegen] " + what, a, b);
    codegen_failures += failures - before;
}

void skipped(const std::string& what, const char* why) { std::printf("%-66s skipped: %s\n", what.c_str(), why); }

#if defined(__GNUC__)
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
#endif
std::string fmt(const char* f, ...) {
    char b[256];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

// ---- host references ----

double h_tau(float alpha, float floor_, long long p) {
    return alpha > 0 ? 1.0 + alpha * std::log(std::max(1.0, (double) (p + 1) / floor_)) : 1.0;
}

// clip_head(): v times scale, brought down to an L2 norm of at most thr, times tau
void h_clip(const float* v, const float* sc, int dim, float thr, float alpha, float floor_, long long p, double* out) {
    double ss = 0;
    for (int i = 0; i < dim; ++i) {
        const double y = (double) v[i] * (sc ? sc[i] : 1.0);
        ss += y * y;
    }
    const double f = std::min(thr / (std::sqrt(ss) + 1e-6), 1.0) * h_tau(alpha, floor_, p);
    for (int i = 0; i < dim; ++i) out[i] = (double) v[i] * (sc ? sc[i] : 1.0) * f;
}

void h_rms(const float* v, int dim, float eps, double* out) {
    double ss = 0;
    for (int i = 0; i < dim; ++i) ss += (double) v[i] * v[i];
    const double f = 1.0 / std::sqrt(ss / dim + eps);
    for (int i = 0; i < dim; ++i) out[i] = v[i] * f;
}

// ink_conv() of `rows` tokens: out (rows * channels) and the state after the call (the last taps - 1 inputs)
void h_conv(const std::vector<float>& x, std::vector<float>& state, const std::vector<float>& w, const float* dyn,
            int channels, int heads, int taps, int rows, std::vector<double>& out) {
    const double norm = 1.0 / std::sqrt((double) ((dyn ? 2 : 1) * taps + 1));
    out.assign((size_t) rows * channels, 0.0);
    const auto past = [&](int t, int k, int c) {   // the input k tokens before token t
        const int u = t - k;
        return u >= 0 ? (double) x[(size_t) u * channels + c] : (double) state[(size_t) (taps - 1 + u) * channels + c];
    };
    for (int t = 0; t < rows; ++t)
        for (int c = 0; c < channels; ++c) {
            double acc = past(t, 0, c);
            for (int j = 0; j < taps; ++j) acc += past(t, taps - 1 - j, c) * w[(size_t) c * taps + j];
            if (dyn) {
                const float* d = dyn + ((size_t) t * heads + c / (channels / heads)) * taps;
                for (int lag = 0; lag < taps; ++lag) acc += past(t, lag, c) * d[lag];
            }
            out[(size_t) t * channels + c] = acc * norm;
        }
    std::vector<float> next(state.size());
    for (int k = 0; k < taps - 1; ++k)
        for (int c = 0; c < channels; ++c) next[(size_t) k * channels + c] = (float) past(rows, taps - 1 - k, c);
    state = next;
}

// ink_conv() in floats, rounded as nvcc compiles conv_value() (its SASS): the taps a fused multiply-add each in their
// order onto 0, the input-dependent ones after them, then (x[t] + that) rounded and times norm rounded: an add and a
// multiply, not one FMA.  A device's bits must be these.  The state is left as h_conv() leaves it.
std::vector<float> h_conv_f32(const std::vector<float>& x, const std::vector<float>& state, const std::vector<float>& w,
                              const float* dyn, int channels, int heads, int taps, int rows) {
    const float norm = 1.0f / std::sqrt((float) ((dyn ? 2 : 1) * taps + 1));   // as ink_conv() makes it on the host
    std::vector<float> out((size_t) rows * channels);
    const auto past = [&](int t, int k, int c) {
        const int u = t - k;
        return u >= 0 ? x[(size_t) u * channels + c] : state[(size_t) (taps - 1 + u) * channels + c];
    };
    for (int t = 0; t < rows; ++t)
        for (int c = 0; c < channels; ++c) {
            float acc = 0.0f;
            for (int j = 0; j < taps; ++j) acc = std::fma(past(t, taps - 1 - j, c), w[(size_t) c * taps + j], acc);
            if (dyn) {
                const float* d = dyn + ((size_t) t * heads + c / (channels / heads)) * taps;
                for (int lag = 0; lag < taps; ++lag) acc = std::fma(past(t, lag, c), d[lag], acc);
            }
            volatile float sum = past(t, 0, c) + acc;   // a rounding of its own on the host too
            out[(size_t) t * channels + c] = sum * norm;
        }
    return out;
}

// The stream's update after a convolution, as nvcc compiles it: mode 0 acc + y * a one FMA, else y
std::vector<float> h_stream_f32(const std::vector<float>& acc, const std::vector<float>& y, float a, int mode) {
    std::vector<float> out(acc.size());
    for (size_t i = 0; i < acc.size(); ++i) out[i] = mode == 0 ? std::fma(y[i], a, acc[i]) : y[i];
    return out;
}

// The state after the first `rows` tokens of a call, from the state before it
std::vector<float> h_state_after(const std::vector<float>& x, const std::vector<float>& state, int channels, int taps,
                                 int rows) {
    std::vector<double> unused;
    std::vector<float> s = state;
    h_conv(x, s, std::vector<float>((size_t) channels * taps, 0.f), nullptr, channels, 1, taps, rows, unused);
    return s;
}

// a rows of n values RMS-normed (weight w, null: none)
std::vector<double> h_rms_norm(const float* x, const float* w, int n, int rows, float eps) {
    std::vector<double> y((size_t) rows * n);
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t) r * n;
        double ss = 0;
        for (int i = 0; i < n; ++i) ss += (double) xr[i] * xr[i];
        const double f = 1.0 / std::sqrt(ss / n + eps);
        for (int i = 0; i < n; ++i) y[(size_t) r * n + i] = xr[i] * f * (w ? w[i] : 1.0);
    }
    return y;
}

// ink_mix(): states[s] rows of n, queries nq x n; out[j] rows of n
std::vector<std::vector<double>> h_mix(const std::vector<std::vector<float>>& states, const std::vector<float>& query,
                                       int nq, int n, int rows) {
    const int ns = (int) states.size();
    std::vector<std::vector<double>> out((size_t) nq, std::vector<double>((size_t) rows * n));
    for (int r = 0; r < rows; ++r) {
        std::vector<double> inv((size_t) ns);
        for (int s = 0; s < ns; ++s) {
            double ss = 0;
            for (int i = 0; i < n; ++i) ss += (double) states[s][(size_t) r * n + i] * states[s][(size_t) r * n + i];
            inv[s] = 1.0 / std::sqrt(ss / n + 1e-6);
        }
        for (int j = 0; j < nq; ++j) {
            std::vector<double> sc((size_t) ns);
            double m = -INFINITY, z = 0;
            for (int s = 0; s < ns; ++s) {
                double d = 0;
                for (int i = 0; i < n; ++i) d += (double) states[s][(size_t) r * n + i] * query[(size_t) j * n + i];
                sc[s] = d * inv[s] / std::sqrt((double) n);
                m = std::max(m, sc[s]);
            }
            for (int s = 0; s < ns; ++s) z += sc[s] = std::exp(sc[s] - m);
            for (int i = 0; i < n; ++i) {
                double acc = 0;
                for (int s = 0; s < ns; ++s) acc += sc[s] / z * inv[s] * states[s][(size_t) r * n + i];
                out[j][(size_t) r * n + i] = acc;
            }
        }
    }
    return out;
}

double h_gate(double c, double m, double g, double f) { return c - std::tanh(f) * c + std::tanh(g) * m; }

// FP8 E4M3 (OCP, the one __NV_E4M3 is): code -> value
double e4m3(uint8_t c) {
    const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    if (e == 15 && m == 7) return NAN;
    const double v = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
    return s ? -v : v;
}

// The rows of a cache as the kernels read them: fp16 halves widened, or FP8 codes times their head's scale in floats
std::vector<float> cache_rows(const std::vector<uint8_t>& bytes, bool kv8, int rows) {
    const size_t per = (size_t) kKv * kDim;
    std::vector<float> out((size_t) rows * per);
    for (int r = 0; r < rows; ++r) {
        if (kv8) {
            const uint8_t* row = bytes.data() + (size_t) r * ink_kv8_row(kKv, kDim);
            for (int h = 0; h < kKv; ++h) {
                float sc;
                std::memcpy(&sc, row + per + (size_t) h * 4, 4);
                for (int i = 0; i < kDim; ++i)
                    out[(size_t) r * per + (size_t) h * kDim + i] = (float) e4m3(row[(size_t) h * kDim + i]) * sc;
            }
        } else {
            const __half* row = (const __half*) bytes.data() + (size_t) r * per;
            for (size_t i = 0; i < per; ++i) out[(size_t) r * per + i] = __half2float(row[i]);
        }
    }
    return out;
}

size_t cache_bytes(bool kv8, int rows) {
    return (size_t) rows * (kv8 ? ink_kv8_row(kKv, kDim) : (size_t) kKv * kDim * sizeof(uint16_t));
}

// ---- the cases ----

// ink_clip(): with and without a scale and tau, the positions by value and on the card (the same bits)
void case_clip() {
    for (const int rows : {1, 7}) {
        for (const bool scaled : {false, true}) {
            const int dim = scaled ? kDim : kRel, heads = kHeads;
            const float thr = scaled ? kQkClip : kRelClip;
            std::vector<float> x((size_t) rows * heads * dim);
            for (int t = 0; t < rows; ++t) {   // heads far inside the threshold and far outside it
                const auto v = randv((size_t) heads * dim, (t % 2 ? 3.0f : 0.2f) * thr / std::sqrt((float) dim));
                std::copy(v.begin(), v.end(), x.begin() + (size_t) t * heads * dim);
            }
            const auto sc = randv((size_t) heads * dim, 0.3f, 1.0f);
            const long long pos = 20000;   // past the floor: tau > 1
            Dev<float> dx(x), dsc(sc), dy(x);
            ink_clip(dx, scaled ? dsc.p : nullptr, thr, dim, heads, rows, pos, kAlpha, kFloor, stream());
            std::vector<double> ref(x.size());
            for (int t = 0; t < rows; ++t)
                for (int h = 0; h < heads; ++h) {
                    const size_t o = ((size_t) t * heads + h) * dim;
                    h_clip(&x[o], scaled ? &sc[(size_t) h * dim] : nullptr, dim, thr, kAlpha, kFloor, pos + t, &ref[o]);
                }
            const auto got = dx.get();
            compare(fmt("ink_clip %d rows, %s", rows, scaled ? "scaled Q heads" : "relative values"), got, ref,
                    2e-6 * thr);
            std::vector<long long> at((size_t) 2 * rows);
            for (int t = 0; t < rows; ++t) at[(size_t) 2 * t] = pos + t;
            Dev<long long> dat(at);
            ink_clip(dy, scaled ? dsc.p : nullptr, thr, dim, heads, rows, 0, kAlpha, kFloor, stream(), dat);
            same_bits(fmt("ink_clip %d rows, positions on the card = by value", rows), dy.get(), got);
        }
    }
}

void case_head_rms() {
    for (const int rows : {1, 9}) {
        const auto x = randv((size_t) rows * kKv * kDim, 2.0f);
        Dev<float> dx(x);
        ink_head_rms(dx, kDim, kKv, rows, kEps, stream());
        std::vector<double> ref(x.size());
        for (size_t h = 0; h < (size_t) rows * kKv; ++h) h_rms(&x[h * kDim], kDim, kEps, &ref[h * kDim]);
        compare(fmt("ink_head_rms %d rows", rows), dx.get(), ref, 4e-6);
    }
}

// ink_clips(): Q, K, V (head RMS) and the relative values in one launch, K and V stored into their caches (fp16 and
// FP8, a ring and a full cache), against ink_clip() and ink_head_rms() and the stores one at a time (the same bits)
void case_clips() {
    for (const bool kv8 : {false, true}) {
#if defined(__HIPCC__)
        if (kv8) {
            skipped("ink_clips, FP8 caches", "no FP8 caches under HIP");
            continue;
        }
#endif
        for (const int ring : {0, 40}) {
            const int rows = 5, cells = 64;
            const long long pos = ring ? 37 : 11;   // a ring's rows go round past its end
            const auto q = randv((size_t) rows * kHeads * kDim, 0.8f), k = randv((size_t) rows * kKv * kDim, 3.0f),
                       v = randv((size_t) rows * kKv * kDim, 2.0f), r = randv((size_t) rows * kHeads * kRel, 1.5f);
            const auto qs = randv((size_t) kHeads * kDim, 0.3f, 1.0f), rs = randv((size_t) kHeads * kRel, 0.3f, 1.0f);
            Dev<float> dq(q), dk(k), dv(v), dr(r), dqs(qs), drs(rs), dq2(q), dk2(k), dv2(v), dr2(r);
            Dev<uint8_t> keys(cache_bytes(kv8, cells)), vals(cache_bytes(kv8, cells));
            Dev<uint8_t> keys2(cache_bytes(kv8, cells)), vals2(cache_bytes(kv8, cells));
            InkClips c;
            float* xs[4] = {dq, dk, dv, dr};
            const float* sc[4] = {dqs, nullptr, nullptr, drs};
            const float thr[4] = {kQkClip, kQkClip, 0.f, kRelClip}, al[4] = {kAlpha, 0.f, 0.f, kAlpha};
            const float fl[4] = {kFloor, 1.f, 1.f, kFloor};
            const int dims[4] = {kDim, kDim, kDim, kRel}, hs[4] = {kHeads, kKv, kKv, kHeads};
            for (int j = 0; j < 4; ++j) {
                c.x[j] = xs[j], c.scale[j] = sc[j], c.thr[j] = thr[j], c.alpha[j] = al[j], c.floor[j] = fl[j];
                c.dim[j] = dims[j], c.heads[j] = hs[j];
            }
            c.rms[2] = true;
            c.eps = kEps;
            c.pos = pos;
            c.rows = rows;
            c.cache[1] = (uint16_t*) keys.p;
            c.cache[2] = (uint16_t*) vals.p;
            c.ring = ring;
            c.kv8 = kv8;
            ink_clips(c, stream());
            // one at a time; the stores as ink_clips() of a pass-through clip (a threshold nothing reaches)
            ink_clip(dq2, dqs, kQkClip, kDim, kHeads, rows, pos, kAlpha, kFloor, stream());
            ink_clip(dk2, nullptr, kQkClip, kDim, kKv, rows, pos, 0.f, 1.f, stream());
            ink_head_rms(dv2, kDim, kKv, rows, kEps, stream());
            ink_clip(dr2, drs, kRelClip, kRel, kHeads, rows, pos, kAlpha, kFloor, stream());
            const std::string tag = fmt("ink_clips %s %s", kv8 ? "fp8" : "fp16", ring ? "ring" : "full");
            same_bits(tag + ": Q = ink_clip()", dq.get(), dq2.get());
            same_bits(tag + ": K = ink_clip()", dk.get(), dk2.get());
            same_bits(tag + ": V = ink_head_rms()", dv.get(), dv2.get());
            same_bits(tag + ": R = ink_clip()", dr.get(), dr2.get());
            InkClips st;
            st.x[0] = dk2, st.x[1] = dv2, st.thr[0] = st.thr[1] = 1e30f, st.dim[0] = st.dim[1] = kDim;
            st.heads[0] = st.heads[1] = kKv, st.pos = pos, st.rows = rows, st.ring = ring, st.kv8 = kv8;
            st.cache[0] = (uint16_t*) keys2.p, st.cache[1] = (uint16_t*) vals2.p;
            ink_clips(st, stream());
            same_bits(tag + ": the stores", keys.get(), keys2.get());
            same_bits(tag + ": the stores (values)", vals.get(), vals2.get());
            // the K reference, and the rows where the stores put it
            std::vector<double> kref(k.size());
            for (size_t h = 0; h < (size_t) rows * kKv; ++h)
                h_clip(&k[h * kDim], nullptr, kDim, kQkClip, 0.f, 1.f, 0, &kref[h * kDim]);
            compare(tag + ": K against the reference", dk.get(), kref, 2e-6 * kQkClip);
            const auto rowsf = cache_rows(keys.get(), kv8, cells);
            std::vector<float> stored(k.size());
            std::vector<double> want(k.size());
            for (int t = 0; t < rows; ++t) {
                const long long cell = ring ? (pos + t) % ring : pos + t;
                for (int i = 0; i < kKv * kDim; ++i) {
                    stored[(size_t) t * kKv * kDim + i] = rowsf[(size_t) cell * kKv * kDim + i];
                    want[(size_t) t * kKv * kDim + i] = kref[(size_t) t * kKv * kDim + i];
                }
            }
            if (!kv8) {   // fp16: a half's rounding of values up to ~4
                compare(tag + ": the cache's rows hold K", stored, want, 2e-3);
                continue;
            }
            // FP8: a head's codes are x / scale with scale = max|x| / 448, so a code's value is at most 448.  E4M3
            // has 3 bits of mantissa: in [2^e, 2^(e+1)) its step is 2^(e-3), at most 32 (in [256, 448]), and a
            // nearest code is off by at most half a step, 16 code units = max|x| * 16 / 448 = max|x| / 28.  So the
            // check is on x / max|x| (each head's), with a tolerance of 1/28, and 1e-4 more for K's own error (2e-6 *
            // 22.6 above, over a max|x| of about 5).
            std::vector<float> stored_n(stored.size());
            for (size_t h = 0; h < (size_t) rows * kKv; ++h) {
                double m = 0;
                for (int i = 0; i < kDim; ++i) m = std::max(m, std::fabs(want[h * kDim + i]));
                for (int i = 0; i < kDim; ++i) {
                    stored_n[h * kDim + i] = (float) (stored[h * kDim + i] / m);
                    want[h * kDim + i] /= m;
                }
            }
            compare(tag + ": the cache's rows hold K (over each head's max |K|)", stored_n, want, 1.0 / 28 + 1e-4);
        }
    }
}

// FP8 stores: each code is a nearest E4M3 value of v / scale, and the scale is the head's largest magnitude / 448
void case_kv8_store() {
#if defined(__HIPCC__)
    skipped("FP8 store", "no FP8 caches under HIP");
#else
    const int rows = 6;
    auto v = randv((size_t) rows * kKv * kDim, 1.0f);
    for (int i = 0; i < kDim; ++i) v[(size_t) 3 * kDim + i] = 0.f;            // a head of zeros (scale 1)
    for (int i = 0; i < kDim; ++i) v[(size_t) 5 * kDim + i] *= 1e-4f;         // a tiny one: subnormal codes
    Dev<float> dv(v);
    Dev<uint8_t> cache(cache_bytes(true, rows));
    InkClips st;
    st.x[0] = dv, st.thr[0] = 1e30f, st.dim[0] = kDim, st.heads[0] = kKv, st.rows = rows, st.kv8 = true;
    st.cache[0] = (uint16_t*) cache.p;
    ink_clips(st, stream());
    const auto bytes = cache.get();
    ++checks;
    int bad_scale = 0, bad_code = 0;
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < kKv; ++h) {
            const uint8_t* row = bytes.data() + (size_t) t * ink_kv8_row(kKv, kDim);
            const float* x = &v[((size_t) t * kKv + h) * kDim];
            float a = 0.f;
            for (int i = 0; i < kDim; ++i) a = std::max(a, std::fabs(x[i]));
            const float want = a > 0.f ? a / 448.0f : 1.0f;
            float sc;
            std::memcpy(&sc, row + (size_t) kKv * kDim + (size_t) h * 4, 4);
            if (sc != want) ++bad_scale;
            for (int i = 0; i < kDim; ++i) {
                const double y = (double) (x[i] * (1.0f / sc)), got = e4m3(row[(size_t) h * kDim + i]);
                double best = INFINITY;
                for (int c = 0; c < 256; ++c)
                    if (std::isfinite(e4m3((uint8_t) c))) best = std::min(best, std::fabs(e4m3((uint8_t) c) - y));
                if (!(std::fabs(got - y) <= best * (1 + 1e-12) + 1e-30)) ++bad_code;
            }
        }
    const bool ok = bad_scale == 0 && bad_code == 0;
    if (!ok) ++failures;
    std::printf("%-66s %d scales off, %d codes not a nearest value  %s\n", "FP8 store: scale = max/448, nearest codes",
                bad_scale, bad_code, ok ? "ok" : "FAIL");
#endif
}

// ink_conv(): with and without the input-dependent taps, a call of a few tokens (ink_conv_few()) and of more, the
// state carried over several calls, and a call's snapshots of its state
void case_conv() {
    const int channels = kKv * kDim;
    for (const bool dynamic : {false, true}) {
        std::vector<float> state = randv((size_t) (kTaps - 1) * channels, 1.0f);
        const auto w = randv((size_t) channels * kTaps, 0.5f);
        Dev<float> dstate(state), dw(w);
        for (const int rows : {1, 3, 8, 9, 37}) {
            const auto x = randv((size_t) rows * channels, 1.0f), dyn = randv((size_t) rows * kKv * kTaps, 0.5f);
            Dev<float> dx(x), dd(dyn), dout((size_t) rows * channels);
            const unsigned bits = rows >= 3 ? 0x5u : 0u;   // the states after tokens 0 and 2
            Dev<float> snap((size_t) 2 * (kTaps - 1) * channels);
            const std::vector<float> before = state;
            ink_conv(dx, dstate, dw, dynamic ? dd.p : nullptr, dout, channels, kKv, kTaps, rows, stream(),
                     bits ? snap.p : nullptr, (size_t) (kTaps - 1) * channels, bits);
            std::vector<double> ref;
            h_conv(x, state, w, dynamic ? dyn.data() : nullptr, channels, kKv, kTaps, rows, ref);
            const std::string tag = fmt("ink_conv %s, %d rows", dynamic ? "with taps" : "plain", rows);
            const auto got = dout.get();
            compare(tag, got, ref, 2e-6);
            codegen_bits(tag + ": = the CUDA-rounded reference",
                      got, h_conv_f32(x, before, w, dynamic ? dyn.data() : nullptr, channels, kKv, kTaps, rows));
            compare(tag + ": the state after", dstate.get(), dbl(state), 0.0);
            if (bits) {
                std::vector<float> want = h_state_after(x, before, channels, kTaps, 1);
                const std::vector<float> s2 = h_state_after(x, before, channels, kTaps, 3);
                want.insert(want.end(), s2.begin(), s2.end());
                compare(tag + ": the snapshots", snap.get(), dbl(want), 0.0);
            }
        }
    }
}

// ink_conv_few() of two jobs with the stream's updates (modes 0, 1, 2) against the reference, and ink_conv_add() (a
// token's, modes 0 to 2) against ink_conv_few() of one job: the same bits
void case_conv_acc() {
    const int channels = kEmbd;
    for (const int mode : {0, 1, 2}) {
        for (const int rows : {1, 4}) {
            const auto x = randv((size_t) rows * channels, 1.0f), acc = randv((size_t) rows * channels, 1.0f);
            const auto st = randv((size_t) (kTaps - 1) * channels, 1.0f), w = randv((size_t) channels * kTaps, 0.5f);
            Dev<float> dx(x), dacc(acc), dst(st), dw(w), dout((size_t) rows * channels), dkeep((size_t) rows * channels);
            InkConv j;
            j.x = dx, j.state = dst, j.w = dw, j.out = dout, j.acc = dacc, j.keep = dkeep, j.a = 0.75f, j.mode = mode;
            // a second job, a heads-wide one with taps (the K and V convolutions' pair)
            const int c2 = channels;
            const auto x2 = randv((size_t) rows * c2, 1.0f), dy2 = randv((size_t) rows * kKv * kTaps, 0.5f);
            const auto st2 = randv((size_t) (kTaps - 1) * c2, 1.0f);
            Dev<float> dx2(x2), dd2(dy2), dst2(st2), dout2((size_t) rows * c2);
            InkConv j2;
            j2.x = dx2, j2.state = dst2, j2.w = dw, j2.dyn = dd2, j2.out = dout2;
            ink_conv_few(j, j2, channels, kKv, kTaps, rows, stream(), 0, 0);
            std::vector<float> s = st, s2 = st2;
            std::vector<double> ref, ref2;
            h_conv(x, s, w, nullptr, channels, 1, kTaps, rows, ref);
            h_conv(x2, s2, w, dy2.data(), c2, kKv, kTaps, rows, ref2);
            std::vector<double> racc(acc.size()), rkeep(acc.size(), 0.0);
            for (size_t i = 0; i < acc.size(); ++i) {
                racc[i] = mode == 0 ? acc[i] + ref[i] * 0.75 : ref[i];
                if (mode == 1) rkeep[i] = acc[i];
            }
            const std::string tag = fmt("ink_conv_few mode %d, %d rows", mode, rows);
            const auto got = dout.get(), gacc = dacc.get(), got2 = dout2.get();
            compare(tag + ": out", got, ref, 2e-6);
            compare(tag + ": the stream", gacc, racc, 4e-6);
            const auto y32 = h_conv_f32(x, st, w, nullptr, channels, 1, kTaps, rows);
            codegen_bits(tag + ": out = the CUDA-rounded reference", got, y32);
            codegen_bits(tag + ": the stream = the CUDA-rounded reference", gacc, h_stream_f32(acc, y32, 0.75f, mode));
            if (mode == 1) compare(tag + ": the stream kept", dkeep.get(), rkeep, 0.0);
            compare(tag + ": second job (taps)", got2, ref2, 2e-6);
            codegen_bits(tag + ": second job = the CUDA-rounded reference", got2,
                      h_conv_f32(x2, st2, w, dy2.data(), c2, kKv, kTaps, rows));
            compare(tag + ": second job's state", dst2.get(), dbl(s2), 0.0);
            if (rows != 1) continue;
            Dev<float> ex(x), eacc(acc), est(st), eout((size_t) channels), ekeep((size_t) channels);
            ink_conv_add(ex, est, dw, eout, eacc, ekeep, 0.75f, mode, channels, kTaps, stream());
            same_bits(fmt("ink_conv_add mode %d = ink_conv_few", mode), eout.get(), dout.get());
            same_bits(fmt("ink_conv_add mode %d = ink_conv_few: the stream", mode), eacc.get(), dacc.get());
            same_bits(fmt("ink_conv_add mode %d = ink_conv_few: the state", mode), est.get(), dst.get());
            if (mode == 1) same_bits("ink_conv_add mode 1 = ink_conv_few: kept", ekeep.get(), dkeep.get());
        }
    }
}

// ink_moe_conv_add(): a token's experts' weighted sum (with each expert's scale and without), then its convolution
void case_moe_conv_add() {
    for (const bool scaled : {false, true}) {
        const auto x = randv((size_t) kUsed * kEmbd, 1.0f), w = randv(kUsed, 0.3f, 0.5f);
        const auto escale = randv(kExperts, 0.2f, 1.0f), acc = randv(kEmbd, 1.0f);
        const auto st = randv((size_t) (kTaps - 1) * kEmbd, 1.0f), cw = randv((size_t) kEmbd * kTaps, 0.5f);
        std::vector<int> ids(kUsed);
        for (int k = 0; k < kUsed; ++k) ids[k] = (k * 37 + 5) % kExperts;
        Dev<float> dx(x), dw(w), des(escale), dacc(acc), dst(st), dcw(cw), dfo(kEmbd), dout(kEmbd);
        Dev<int> dids(ids);
        ink_moe_conv_add(dx, dw, scaled ? des.p : nullptr, dids, kUsed, dfo, dst, dcw, dout, dacc, 1.0f, kEmbd, kTaps,
                         stream());
        std::vector<double> fo(kEmbd, 0.0);
        std::vector<float> fof(kEmbd);
        for (int i = 0; i < kEmbd; ++i) {
            for (int k = 0; k < kUsed; ++k)
                fo[i] += (double) w[k] * (scaled ? escale[ids[k]] : 1.0) * x[(size_t) k * kEmbd + i];
        }
        const auto gfo = dfo.get();
        compare(fmt("ink_moe_conv_add %s: the experts' sum", scaled ? "scaled" : "plain"), gfo, fo, 4e-6);
        std::vector<float> s = st;
        std::vector<double> ref;
        h_conv(gfo, s, cw, nullptr, kEmbd, 1, kTaps, 1, ref);   // from the card's sum: the convolution alone
        std::vector<double> racc(kEmbd);
        for (int i = 0; i < kEmbd; ++i) racc[i] = acc[i] + ref[i];
        const auto got = dout.get(), gacc = dacc.get();
        compare(fmt("ink_moe_conv_add %s: out", scaled ? "scaled" : "plain"), got, ref, 2e-6);
        compare(fmt("ink_moe_conv_add %s: the stream", scaled ? "scaled" : "plain"), gacc, racc, 4e-6);
        // nvcc's rounding: each weight times its scale, a fused multiply-add an expert onto 0, then the convolution
        std::vector<float> fo32(kEmbd);
        for (int i = 0; i < kEmbd; ++i) {
            float sum = 0.0f;
            for (int k = 0; k < kUsed; ++k)
                sum = std::fma(scaled ? w[k] * escale[ids[k]] : w[k], x[(size_t) k * kEmbd + i], sum);
            fo32[i] = 0.0f + sum;
        }
        codegen_bits(fmt("ink_moe_conv_add %s: the sum = the CUDA-rounded reference", scaled ? "scaled" : "plain"), gfo,
                  fo32);
        const auto y32 = h_conv_f32(fo32, st, cw, nullptr, kEmbd, 1, kTaps, 1);
        codegen_bits(fmt("ink_moe_conv_add %s: out = the CUDA-rounded reference", scaled ? "scaled" : "plain"), got,
                     y32);
        codegen_bits(fmt("ink_moe_conv_add %s: the stream = the CUDA-rounded reference", scaled ? "scaled" : "plain"),
                  gacc, h_stream_f32(acc, y32, 1.0f, 0));
        compare(fmt("ink_moe_conv_add %s: the state", scaled ? "scaled" : "plain"), dst.get(), dbl(s), 0.0);
    }
}

// ink_qkv(): a token's convolutions, clips, V's norm and the stores in one launch, against ink_conv_few() and
// ink_clips() (the engine's path for a few tokens: the same bits) and against the reference; fp16 and FP8, a ring
// and a full cache, the position by value and on the card
void case_qkv() {
    const int kvn = kKv * kDim;
    for (const bool kv8 : {false, true}) {
#if defined(__HIPCC__)
        if (kv8) {
            skipped("ink_qkv, FP8 caches", "no FP8 caches under HIP");
            continue;
        }
#endif
        for (const bool full : {false, true}) {
            for (const bool on_card : {false, true}) {
                const int ring = full ? 0 : kWindow + 8, cells = full ? 3000 : ring;
                const long long pos = 2100;
                const auto q = randv((size_t) kHeads * kDim, 0.8f), r = randv((size_t) kHeads * kRel, 1.5f);
                const auto kraw = randv(kvn, 2.0f), vraw = randv(kvn, 2.0f);
                const auto kst = randv((size_t) (kTaps - 1) * kvn, 2.0f), vst = randv((size_t) (kTaps - 1) * kvn, 2.0f);
                const auto kw = randv((size_t) kvn * kTaps, 0.5f), vw = randv((size_t) kvn * kTaps, 0.5f);
                const auto kd = randv((size_t) kKv * kTaps, 0.4f), vd = randv((size_t) kKv * kTaps, 0.4f);
                const auto qs = randv((size_t) kHeads * kDim, 0.3f, 1.0f), rs = randv((size_t) kHeads * kRel, 0.3f, 1.0f);
                // two of everything that is written: ink_qkv()'s and the calls'
                Dev<float> q1(q), q2(q), r1(r), r2(r), ks1(kst), ks2(kst), vs1(vst), vs2(vst), k1(kvn), k2(kvn), v1(kvn),
                    v2(kvn);
                Dev<float> dkraw(kraw), dvraw(vraw), dkw(kw), dvw(vw), dkd(kd), dvd(vd), dqs(qs), drs(rs);
                Dev<uint8_t> keys1(cache_bytes(kv8, cells)), vals1(cache_bytes(kv8, cells));
                Dev<uint8_t> keys2(cache_bytes(kv8, cells)), vals2(cache_bytes(kv8, cells));
                Dev<long long> at(std::vector<long long>{pos, 0});
                InkQkv a;
                a.q = q1, a.q_scale = dqs, a.r = r1, a.r_scale = drs, a.kraw = dkraw, a.vraw = dvraw;
                a.kstate = ks1, a.vstate = vs1, a.kw = dkw, a.vw = dvw;
                a.kdyn = full ? dkd.p : nullptr, a.vdyn = full ? dvd.p : nullptr;
                a.k = k1, a.v = v1, a.keys = (uint16_t*) keys1.p, a.vals = (uint16_t*) vals1.p;
                a.at = on_card ? at.p : nullptr, a.pos = on_card ? 0 : pos;
                a.qk_thr = kQkClip, a.rel_thr = kRelClip, a.alpha = full ? kAlpha : 0.f, a.floor = kFloor;
                a.eps = kEps, a.n_head = kHeads, a.n_kv = kKv, a.dim = kDim, a.n_rel = kRel, a.taps = kTaps;
                a.ring = ring, a.kv8 = kv8;
                ink_qkv(a, stream());
                // the engine's calls for the same (layer_head(), n > 1's branch)
                InkConv ckk, cvv;
                ckk.x = dkraw, ckk.state = ks2, ckk.w = dkw, ckk.dyn = a.kdyn, ckk.out = k2;
                cvv.x = dvraw, cvv.state = vs2, cvv.w = dvw, cvv.dyn = a.vdyn, cvv.out = v2;
                ink_conv_few(ckk, cvv, kvn, kKv, kTaps, 1, stream(), 0, 0);
                InkClips c;
                float* xs[4] = {q2, k2, v2, r2};
                const float* sc[4] = {dqs, nullptr, nullptr, drs};
                const float thr[4] = {kQkClip, kQkClip, 0.f, kRelClip}, al[4] = {a.alpha, 0.f, 0.f, a.alpha};
                const float fl[4] = {kFloor, 1.f, 1.f, kFloor};
                const int dims[4] = {kDim, kDim, kDim, kRel}, hs[4] = {kHeads, kKv, kKv, kHeads};
                for (int j = 0; j < 4; ++j) {
                    c.x[j] = xs[j], c.scale[j] = sc[j], c.thr[j] = thr[j], c.alpha[j] = al[j], c.floor[j] = fl[j];
                    c.dim[j] = dims[j], c.heads[j] = hs[j];
                }
                c.rms[2] = true, c.eps = kEps, c.pos = pos, c.rows = 1, c.ring = ring, c.kv8 = kv8;
                c.cache[1] = (uint16_t*) keys2.p, c.cache[2] = (uint16_t*) vals2.p;
                ink_clips(c, stream());
                const std::string tag = fmt("ink_qkv %s %s%s", kv8 ? "fp8" : "fp16", full ? "full+taps" : "ring",
                                            on_card ? ", position on the card" : "");
                same_bits(tag + ": Q", q1.get(), q2.get());
                same_bits(tag + ": R", r1.get(), r2.get());
                same_bits(tag + ": K", k1.get(), k2.get());
                same_bits(tag + ": V", v1.get(), v2.get());
                same_bits(tag + ": K state", ks1.get(), ks2.get());
                same_bits(tag + ": V state", vs1.get(), vs2.get());
                same_bits(tag + ": key cache", keys1.get(), keys2.get());
                same_bits(tag + ": value cache", vals1.get(), vals2.get());
                if (on_card) continue;
                // the reference: convolution, clip / norm
                std::vector<float> s = kst;
                std::vector<double> kc, vc;
                h_conv(kraw, s, kw, full ? kd.data() : nullptr, kvn, kKv, kTaps, 1, kc);
                s = vst;
                h_conv(vraw, s, vw, full ? vd.data() : nullptr, kvn, kKv, kTaps, 1, vc);
                std::vector<double> kref(kvn), vref(kvn), qref(q.size()), rref(r.size());
                std::vector<float> kcf(kc.begin(), kc.end()), vcf(vc.begin(), vc.end());
                for (int h = 0; h < kKv; ++h) {
                    h_clip(&kcf[(size_t) h * kDim], nullptr, kDim, kQkClip, 0.f, 1.f, pos, &kref[(size_t) h * kDim]);
                    h_rms(&vcf[(size_t) h * kDim], kDim, kEps, &vref[(size_t) h * kDim]);
                }
                for (int h = 0; h < kHeads; ++h) {
                    h_clip(&q[(size_t) h * kDim], &qs[(size_t) h * kDim], kDim, kQkClip, a.alpha, kFloor, pos,
                           &qref[(size_t) h * kDim]);
                    h_clip(&r[(size_t) h * kRel], &rs[(size_t) h * kRel], kRel, kRelClip, a.alpha, kFloor, pos,
                           &rref[(size_t) h * kRel]);
                }
                compare(tag + ": K against the reference", k1.get(), kref, 2e-5);
                compare(tag + ": V against the reference", v1.get(), vref, 2e-5);
                compare(tag + ": Q against the reference", q1.get(), qref, 4e-5);
                compare(tag + ": R against the reference", r1.get(), rref, 2e-5);
            }
        }
    }
}

// The depth mixes: ink_mix() of 1 to 6 states and 1 to 4 queries (a row a thread can hold, and a wider one),
// ink_mix_many() against ink_mix() of each job, ink_mix_norm() against ink_mix() + rms_norm() (+ the quantization,
// and with ffn native_add_rms_norm_q8_1()'s): the same bits
void case_mix() {
    for (const int n : {kEmbd, 4096}) {
        for (const int ns : {1, 2, 4, 6}) {
            for (const int nq : {1, 3, 4}) {
                if (n != kEmbd && (ns == 2 || nq == 3)) continue;
                const int rows = n == kEmbd ? 3 : 1;
                std::vector<std::vector<float>> st;
                std::vector<Dev<float>> dst;
                InkStates s;
                for (int k = 0; k < ns; ++k) {
                    st.push_back(randv((size_t) rows * n, 0.5f + 0.4f * k));
                    dst.emplace_back(st.back());
                    s.p[s.n++] = dst.back().p;
                }
                const auto query = randv((size_t) nq * n, 1.0f);
                Dev<float> dq(query);
                std::vector<Dev<float>> out;
                float* op[4];
                for (int j = 0; j < nq; ++j) out.emplace_back((size_t) rows * n), op[j] = out.back().p;
                ink_mix(s, dq, nq, op, n, rows, stream());
                const auto ref = h_mix(st, query, nq, n, rows);
                for (int j = 0; j < nq; ++j)
                    compare(fmt("ink_mix %d wide, %d states, query %d of %d, %d rows", n, ns, j, nq, rows),
                            out[(size_t) j].get(), ref[(size_t) j], 2e-5 * (1 + ns));
            }
        }
    }
    // ink_mix_many(): a pass's messages, one row, 12 jobs of 4 queries over 2 states
    {
        const int jobs = 12, nq = 4;
        std::vector<Dev<float>> st, qs, outs, outs2;
        InkStates s;
        for (int k = 0; k < 2; ++k) st.emplace_back(randv(kEmbd, 1.0f)), s.p[s.n++] = st.back().p;
        std::vector<const float*> qp;
        std::vector<float*> op, op2;
        for (int k = 0; k < jobs; ++k) qs.emplace_back(randv((size_t) nq * kEmbd, 1.0f)), qp.push_back(qs.back().p);
        for (int k = 0; k < jobs * nq; ++k) {
            outs.emplace_back((size_t) kEmbd), op.push_back(outs.back().p);
            outs2.emplace_back((size_t) kEmbd), op2.push_back(outs2.back().p);
        }
        ink_mix_many(s, qp.data(), nq, op.data(), jobs, kEmbd, stream());
        for (int k = 0; k < jobs; ++k) ink_mix(s, qp[(size_t) k], nq, op2.data() + (size_t) k * nq, kEmbd, 1, stream());
        std::vector<float> a, b;
        for (int k = 0; k < jobs * nq; ++k) {
            const auto x = outs[(size_t) k].get(), y = outs2[(size_t) k].get();
            a.insert(a.end(), x.begin(), x.end());
            b.insert(b.end(), y.begin(), y.end());
        }
        same_bits("ink_mix_many 12 jobs x 4 queries = ink_mix() of each", a, b);
    }
    // ink_mix_norm(): plain norm (rows 1 and 4), with the quantization, and the FFN's norm
    for (const int mode : {0, 1, 2}) {
        const int rows = mode == 0 ? 4 : 1, ns = 3;
        std::vector<Dev<float>> st;
        InkStates s;
        for (int k = 0; k < ns; ++k) st.emplace_back(randv((size_t) rows * kEmbd, 1.0f)), s.p[s.n++] = st.back().p;
        const auto q = randv((size_t) 3 * kEmbd, 1.0f), w = randv(kEmbd, 0.2f, 1.0f);
        Dev<float> dq(q), dw(w), o0((size_t) rows * kEmbd), o1((size_t) rows * kEmbd), o2((size_t) rows * kEmbd),
            y((size_t) rows * kEmbd), m0((size_t) rows * kEmbd), m1((size_t) rows * kEmbd), m2((size_t) rows * kEmbd),
            y2((size_t) rows * kEmbd);
        const size_t qb = nk::native_q8_1_bytes(kEmbd) * rows;
        Dev<uint8_t> q8(qb), q82(qb);
        float* out[3] = {o0, o1, o2};
        float* ref_out[3] = {m0, m1, m2};
        ink_mix_norm(s, dq, 3, out, kEmbd, rows, dw, y, kEps, stream(), mode ? q8.p : nullptr, mode == 2);
        ink_mix(s, dq, 3, ref_out, kEmbd, rows, stream());
        if (mode == 2) {
            nk::native_add_rms_norm_q8_1(m0, nullptr, dw, y2, q82, kEmbd, 1, kEps, sum_threads(kEmbd), stream());
        } else {
            rms_norm(m0, dw, y2, kEmbd, rows, kEps, stream());
            if (mode) nk::native_quantize_q8_1(y2, q82, kEmbd, 1, stream());
        }
        const char* what[3] = {"rms_norm()", "rms_norm() + quantize", "native_add_rms_norm_q8_1()"};
        const std::string tag = fmt("ink_mix_norm %d rows = ink_mix() + %s", rows, what[mode]);
        same_bits(tag + ": mix 0", o0.get(), m0.get());
        same_bits(tag + ": mix 1", o1.get(), m1.get());
        same_bits(tag + ": mix 2", o2.get(), m2.get());
        same_bits(tag + ": norm", y.get(), y2.get());
        if (mode) same_bits(tag + ": q8_1", q8.get(), q82.get());
        compare(fmt("ink_mix_norm %d rows: the norm against the reference", rows), y.get(),
                h_rms_norm(m0.get().data(), w.data(), kEmbd, rows, kEps), 2e-5);
    }
}

// rms_norm() and rms_norm_many() (the engine's, which Inkling's layers call), ink_norm_many_q8() against
// rms_norm_many() + native_quantize_q8_1() (the same bytes), and ink_gate_norm_q8() against ink_gate() +
// native_add_rms_norm_q8_1() (the same bytes)
void case_norms() {
    for (const int rows : {1, 5}) {
        const auto x = randv((size_t) rows * kEmbd, 3.0f), w = randv(kEmbd, 0.2f, 1.0f);
        Dev<float> dx(x), dw(w), y((size_t) rows * kEmbd);
        rms_norm(dx, dw, y, kEmbd, rows, kEps, stream());
        compare(fmt("rms_norm %d rows", rows), y.get(), h_rms_norm(x.data(), w.data(), kEmbd, rows, kEps), 1e-5);
    }
    std::vector<Dev<float>> xs, ws, ys, ys2;
    std::vector<Dev<uint8_t>> qs, qs2;
    const float* xp[3];
    const float* wp[3];
    float* yp[3];
    float* yp2[3];
    void* qp[3];
    void* qp2[3];
    std::vector<std::vector<float>> hx, hw;
    for (int j = 0; j < 3; ++j) {
        hx.push_back(randv(kEmbd, 1.0f + j)), hw.push_back(randv(kEmbd, 0.2f, 1.0f));
        xs.emplace_back(hx.back()), ws.emplace_back(hw.back()), ys.emplace_back((size_t) kEmbd),
            ys2.emplace_back((size_t) kEmbd);
        qs.emplace_back(nk::native_q8_1_bytes(kEmbd)), qs2.emplace_back(nk::native_q8_1_bytes(kEmbd));
        xp[j] = xs.back(), wp[j] = ws.back(), yp[j] = ys.back(), yp2[j] = ys2.back(), qp[j] = qs.back(),
        qp2[j] = qs2.back();
    }
    ink_norm_many_q8(xp, wp, yp, qp, 3, kEmbd, kEps, stream());
    rms_norm_many(xp, wp, yp2, 3, kEmbd, 1, kEps, stream());
    for (int j = 0; j < 3; ++j) nk::native_quantize_q8_1(yp2[j], qp2[j], kEmbd, 1, stream());
    for (int j = 0; j < 3; ++j) {
        same_bits(fmt("ink_norm_many_q8 input %d = rms_norm_many(): norm", j), ys[(size_t) j].get(),
                  ys2[(size_t) j].get());
        same_bits(fmt("ink_norm_many_q8 input %d = native_quantize_q8_1(): q8_1", j), qs[(size_t) j].get(),
                  qs2[(size_t) j].get());
        compare(fmt("rms_norm_many input %d against the reference", j), ys2[(size_t) j].get(),
                h_rms_norm(hx[(size_t) j].data(), hw[(size_t) j].data(), kEmbd, 1, kEps), 1e-5);
    }
    // ink_gate_norm_q8()
    const auto cur = randv(kEmbd, 1.0f), msg = randv(kEmbd, 1.0f), g = randv(kEmbd, 1.0f), f = randv(kEmbd, 1.0f),
               w = randv(kEmbd, 0.2f, 1.0f);
    Dev<float> dc(cur), dm(msg), dg(g), df(f), dw(w), out(kEmbd), out2(kEmbd), y(kEmbd), y2(kEmbd);
    Dev<uint8_t> q(nk::native_q8_1_bytes(kEmbd)), q2(nk::native_q8_1_bytes(kEmbd));
    ink_gate_norm_q8(dc, dm, dg, df, out, dw, y, q, kEmbd, kEps, stream());
    ink_gate(dc, dm, dg, df, out2, kEmbd, stream());
    nk::native_add_rms_norm_q8_1(out2, nullptr, dw, y2, q2, kEmbd, 1, kEps, sum_threads(kEmbd), stream());
    same_bits("ink_gate_norm_q8 = ink_gate(): the gate", out.get(), out2.get());
    same_bits("ink_gate_norm_q8 = native_add_rms_norm_q8_1(): norm", y.get(), y2.get());
    same_bits("ink_gate_norm_q8 = native_add_rms_norm_q8_1(): q8_1", q.get(), q2.get());
}

// The loop controllers: ink_cat_norm(_set)(), ink_ctrl_down() (F32: against ink_cat_norm_set() + gemv_set(), the
// same bits; BF16: the reference), ink_gate(_set)(), ink_control3_gate() (F32: against gemv_set() + ink_gate_set(),
// the same bits; BF16: the reference)
void case_controllers() {
    for (const int rows : {1, 3}) {
        const int count = 3;
        std::vector<std::vector<float>> cur, msg, down, gain, forget;
        std::vector<Dev<float>> dcur, dmsg, ddown, dgain, dforget, cat, ch, ch2, gv, fv, out, out2;
        const float* cp[3];
        const float* mp[3];
        const void* dp[3];
        float* catp[3];
        float* chp[3];
        float* ch2p[3];
        for (int j = 0; j < count; ++j) {
            cur.push_back(randv((size_t) rows * kEmbd, 1.0f + j)), msg.push_back(randv((size_t) rows * kEmbd, 0.5f));
            down.push_back(randv((size_t) kRank * 2 * kEmbd, 0.02f));
            gain.push_back(randv((size_t) kEmbd * kRank, 0.1f)), forget.push_back(randv((size_t) kEmbd * kRank, 0.1f));
            dcur.emplace_back(cur.back()), dmsg.emplace_back(msg.back()), ddown.emplace_back(down.back());
            dgain.emplace_back(gain.back()), dforget.emplace_back(forget.back());
            cat.emplace_back((size_t) rows * 2 * kEmbd), ch.emplace_back((size_t) rows * kRank),
                ch2.emplace_back((size_t) rows * kRank);
            gv.emplace_back((size_t) rows * kEmbd), fv.emplace_back((size_t) rows * kEmbd);
            out.emplace_back((size_t) rows * kEmbd), out2.emplace_back((size_t) rows * kEmbd);
            cp[j] = dcur.back(), mp[j] = dmsg.back(), dp[j] = ddown.back().p, catp[j] = cat.back(),
            chp[j] = ch.back(), ch2p[j] = ch2.back();
        }
        ink_cat_norm_set(cp, mp, catp, count, kEmbd, rows, stream());
        for (int j = 0; j < count; ++j) {
            std::vector<double> ref((size_t) rows * 2 * kEmbd);
            const auto a = h_rms_norm(cur[(size_t) j].data(), nullptr, kEmbd, rows, 1e-6f),
                       b = h_rms_norm(msg[(size_t) j].data(), nullptr, kEmbd, rows, 1e-6f);
            for (int t = 0; t < rows; ++t)
                for (int i = 0; i < kEmbd; ++i) {
                    ref[(size_t) t * 2 * kEmbd + i] = a[(size_t) t * kEmbd + i];
                    ref[(size_t) t * 2 * kEmbd + kEmbd + i] = b[(size_t) t * kEmbd + i];
                }
            compare(fmt("ink_cat_norm_set pair %d, %d rows", j, rows), cat[(size_t) j].get(), ref, 1e-5);
        }
        if (!ink_ctrl_down(cp, mp, dp, chp, count, kEmbd, kRank, rows, stream())) {
            ++failures;
            std::printf("ink_ctrl_down: refused Inkling 2 Mini's shape  FAIL\n");
            continue;
        }
        const float* wp[3] = {ddown[0], ddown[1], ddown[2]};
        const float* xp[3] = {cat[0], cat[1], cat[2]};
        const int outs[3] = {kRank, kRank, kRank};
        gemv_set(wp, xp, ch2p, outs, 3, 2 * kEmbd, rows, stream());
        for (int j = 0; j < count; ++j) {
            same_bits(fmt("ink_ctrl_down %d rows = ink_cat_norm_set() + gemv_set(), ctrl %d", rows, j),
                      ch[(size_t) j].get(), ch2[(size_t) j].get());
            const auto c = cat[(size_t) j].get();
            std::vector<double> ref((size_t) rows * kRank);
            for (int t = 0; t < rows; ++t)
                for (int o = 0; o < kRank; ++o) {
                    double s = 0;
                    for (int i = 0; i < 2 * kEmbd; ++i)
                        s += (double) down[(size_t) j][(size_t) o * 2 * kEmbd + i] * c[(size_t) t * 2 * kEmbd + i];
                    ref[(size_t) t * kRank + o] = s;
                }
            compare(fmt("ink_ctrl_down %d rows, ctrl %d against the reference", rows, j), ch[(size_t) j].get(), ref,
                    2e-5);
        }
        // the gates: gemv_set() of the gain and forget matrices, then ink_gate_set()
        {
            const float* w6[6];
            const float* x6[6];
            float* y6[6];
            int o6[6];
            for (int j = 0; j < 3; ++j) {
                w6[2 * j] = dgain[(size_t) j], w6[2 * j + 1] = dforget[(size_t) j];
                x6[2 * j] = x6[2 * j + 1] = ch[(size_t) j];
                y6[2 * j] = gv[(size_t) j], y6[2 * j + 1] = fv[(size_t) j];
                o6[2 * j] = o6[2 * j + 1] = kEmbd;
            }
            gemv_set(w6, x6, y6, o6, 6, kRank, rows, stream());
            const float* g3[3] = {gv[0], gv[1], gv[2]};
            const float* f3[3] = {fv[0], fv[1], fv[2]};
            float* o3[3] = {out2[0], out2[1], out2[2]};
            ink_gate_set(cp, mp, g3, f3, o3, 3, (size_t) rows * kEmbd, stream());
            for (int j = 0; j < 3; ++j) {
                const auto chv = ch[(size_t) j].get();
                std::vector<double> ref((size_t) rows * kEmbd);
                for (int t = 0; t < rows; ++t)
                    for (int i = 0; i < kEmbd; ++i) {
                        double g = 0, f = 0;
                        for (int k = 0; k < kRank; ++k) {
                            g += (double) gain[(size_t) j][(size_t) i * kRank + k] * chv[(size_t) t * kRank + k];
                            f += (double) forget[(size_t) j][(size_t) i * kRank + k] * chv[(size_t) t * kRank + k];
                        }
                        const size_t at = (size_t) t * kEmbd + i;
                        ref[at] = h_gate(cur[(size_t) j][at], msg[(size_t) j][at], g, f);
                    }
                compare(fmt("gemv_set + ink_gate_set %d rows, ctrl %d against the reference", rows, j),
                        out2[(size_t) j].get(), ref, 1e-5 * (2 + j));
            }
            if (rows == 1) {   // ink_control3_gate(): one launch for the six products and the gates
                const float* chs[3] = {ch[0], ch[1], ch[2]};
                const void* gw[3] = {dgain[0].p, dgain[1].p, dgain[2].p};
                const void* fw[3] = {dforget[0].p, dforget[1].p, dforget[2].p};
                float* o[3] = {out[0], out[1], out[2]};
                ink_control3_gate(cp, mp, chs, gw, fw, o, kEmbd, kRank, stream(), 3, false);
                for (int j = 0; j < 3; ++j)
                    same_bits(fmt("ink_control3_gate = gemv_set() + ink_gate_set(), ctrl %d", j), out[(size_t) j].get(),
                              out2[(size_t) j].get());
                // in place (out = cur), as a window layer's three inputs are one buffer
                Dev<float> shared(cur[0]);
                const float* c3[3] = {shared, shared, shared};
                float* o3s[3] = {shared, out[1], out[2]};
                ink_control3_gate(c3, mp, chs, gw, fw, o3s, kEmbd, kRank, stream(), 3, false);
                // each controller's gate of the one input (cur[0], as it was before controller 0 wrote over it)
                const Dev<float>* got3[3] = {&shared, &out[1], &out[2]};
                for (int j = 0; j < 3; ++j) {
                    std::vector<double> ref(kEmbd);
                    const auto chv = ch[(size_t) j].get();
                    for (int i = 0; i < kEmbd; ++i) {
                        double g = 0, f = 0;
                        for (int k = 0; k < kRank; ++k)
                            g += (double) gain[(size_t) j][(size_t) i * kRank + k] * chv[k],
                                f += (double) forget[(size_t) j][(size_t) i * kRank + k] * chv[k];
                        ref[i] = h_gate(cur[0][i], msg[(size_t) j][i], g, f);
                    }
                    compare(fmt("ink_control3_gate in place (out 0 = cur), ctrl %d", j), got3[j]->get(), ref, 2e-5);
                }
            }
        }
    }
    // BF16 matrices (--loop-bf16): ink_ctrl_down() and ink_control3_gate() of one controller
    {
        const auto cur = randv(kEmbd, 1.0f), msg = randv(kEmbd, 0.5f);
        std::vector<uint16_t> bd, bg, bf;
        std::vector<float> wd, wg, wf;
        bf16_of(randv((size_t) kRank * 2 * kEmbd, 0.02f), bd, wd);
        bf16_of(randv((size_t) kEmbd * kRank, 0.1f), bg, wg);
        bf16_of(randv((size_t) kEmbd * kRank, 0.1f), bf, wf);
        Dev<float> dc(cur), dm(msg), ch((size_t) kRank), out(kEmbd);
        Dev<uint16_t> dd(bd), dg(bg), df(bf);
        const float* cp = dc;
        const float* mp = dm;
        const void* dp = dd.p;
        float* chp = ch;
        if (!ink_ctrl_down(&cp, &mp, &dp, &chp, 1, kEmbd, kRank, 1, stream(), true)) {
            ++failures;
            std::printf("ink_ctrl_down bf16: refused  FAIL\n");
            return;
        }
        const auto a = h_rms_norm(cur.data(), nullptr, kEmbd, 1, 1e-6f), b = h_rms_norm(msg.data(), nullptr, kEmbd, 1, 1e-6f);
        std::vector<double> cref(kRank);
        for (int o = 0; o < kRank; ++o) {
            double s = 0;
            for (int i = 0; i < kEmbd; ++i)
                s += wd[(size_t) o * 2 * kEmbd + i] * a[i] + wd[(size_t) o * 2 * kEmbd + kEmbd + i] * b[i];
            cref[o] = s;
        }
        const auto chv = ch.get();
        compare("ink_ctrl_down bf16 against the reference", chv, cref, 2e-5);
        const float* chs = ch;
        const void* gw = dg.p;
        const void* fw = df.p;
        float* op = out;
        ink_control3_gate(&cp, &mp, &chs, &gw, &fw, &op, kEmbd, kRank, stream(), 1, true);
        std::vector<double> ref(kEmbd);
        for (int i = 0; i < kEmbd; ++i) {
            double g = 0, f = 0;
            for (int k = 0; k < kRank; ++k) g += wg[(size_t) i * kRank + k] * (double) chv[k], f += wf[(size_t) i * kRank + k] * (double) chv[k];
            ref[i] = h_gate(cur[i], msg[i], g, f);
        }
        compare("ink_control3_gate bf16 against the reference", out.get(), ref, 1e-5);
    }
}

// SiTU (situ_rows()): with each expert's scales and without, and its rows quantized (native_quantize_q8_1()'s bytes)
void case_situ() {
    const int n = 1152, rows = 8;
    const float beta = 4.0f, lb = 25.0f;
    for (const bool scaled : {false, true}) {
        const auto gu = randv((size_t) rows * 2 * n, 4.0f);
        const auto sg = randv(kExperts, 0.2f, 1.0f), su = randv(kExperts, 0.2f, 1.0f);
        ExpertScales es;
        std::vector<int32_t> ids(rows);
        for (int r = 0; r < rows; ++r) ids[r] = (r * 29 + 3) % kExperts;
        Dev<float> dgu(gu), dsg(sg), dsu(su), out((size_t) rows * n);
        Dev<int32_t> dids(ids);
        es.gate = dsg, es.up = dsu, es.ids = dids;
        Dev<uint8_t> q(nk::native_q8_1_bytes(n * rows)), q2(nk::native_q8_1_bytes(n * rows));
        situ_rows(dgu, out, n, rows, beta, lb, stream(), q, scaled ? &es : nullptr);
        std::vector<double> ref((size_t) rows * n);
        for (int r = 0; r < rows; ++r)
            for (int i = 0; i < n; ++i) {
                const double g = gu[(size_t) r * 2 * n + i] * (scaled ? sg[ids[r]] : 1.0);
                const double u = gu[(size_t) r * 2 * n + n + i] * (scaled ? su[ids[r]] : 1.0);
                ref[(size_t) r * n + i] = beta * std::tanh(g / beta) / (1.0 + std::exp(-g)) * lb * std::tanh(u / lb);
            }
        const auto got = out.get();
        compare(fmt("situ_rows %s", scaled ? "with expert scales" : "plain"), got, ref, 2e-5 * 25);
        nk::native_quantize_q8_1(out, q2, n * rows, 1, stream());
        same_bits(fmt("situ_rows %s: q8_1 = native_quantize_q8_1()", scaled ? "scaled" : "plain"), q.get(), q2.get());
    }
}

// The router on the card (route_rows(), Inkling2::route_card()'s call) against Inkling2::route()'s arithmetic on the
// host: sigmoid gates, the 8 largest of gate + bias, weights the bare gates over their sum, times 8.  The host's
// route() is what a prompt and a call outside a whole-pass recording use, the card's what a recorded token uses.
void case_router() {
    const int rows = 2048;
    const auto lg = randv((size_t) rows * kExperts, 2.0f), bias = randv(kExperts, 0.05f);
    Dev<float> dlg(lg), dbias(bias), dw((size_t) rows * kUsed);
    Dev<int32_t> dids((size_t) rows * kUsed);
    route_rows(dlg, kExperts, kUsed, rows, 1, 1.f, dbias, nullptr, true, 8.0f, dids, dw, stream());
    const auto ids = dids.get();
    const auto w = dw.get();
    int id_diff = 0, near_ties = 0, bits_diff = 0;
    double worst = 0;
    std::vector<double> ref(w.size());
    std::vector<float> host(w.size());
    for (int t = 0; t < rows; ++t) {
        const float* l = &lg[(size_t) t * kExperts];
        std::vector<float> probs(kExperts), pick(kExperts);   // Inkling2::route(), word for word
        for (int e = 0; e < kExperts; ++e) {
            probs[e] = 1.f / (1.f + std::exp(-l[e]));
            pick[e] = probs[e] + bias[e];
        }
        std::vector<float> by = pick;
        float sum = 0.f;
        int hid[kUsed];
        float hw[kUsed];
        for (int k = 0; k < kUsed; ++k) {
            int best = 0;
            for (int e = 1; e < kExperts; ++e)
                if (pick[e] > pick[best]) best = e;
            pick[best] = -INFINITY;
            hid[k] = best;
            hw[k] = probs[best];
            sum += hw[k];
        }
        const float denom = sum + 1e-20f;
        double dsum = 0;
        for (int k = 0; k < kUsed; ++k) dsum += 1.0 / (1.0 + std::exp(-(double) l[hid[k]]));
        for (int k = 0; k < kUsed; ++k) {
            hw[k] = hw[k] / denom * 8.0f;
            const size_t at = (size_t) t * kUsed + k;
            host[at] = hw[k];
            ref[at] = 8.0 / (1.0 + std::exp(-(double) l[hid[k]])) / dsum;
            if (ids[at] != hid[k]) {   // a near tie at the 8th place may go either way in float
                const double gap = std::fabs((double) by[ids[at]] - by[hid[k]]);
                if (gap < 1e-6) ++near_ties;
                else ++id_diff;
            } else {
                worst = std::max(worst, std::fabs((double) w[at] - ref[at]));
                if (w[at] != hw[k]) ++bits_diff;
            }
        }
    }
    ++checks;
    const bool ok = id_diff == 0 && worst <= 4e-6;
    if (!ok) ++failures;
    std::printf("%-66s %d rows: %d picks differ (%d near ties), weight max err %.2e  %s\n",
                "route_rows (sigmoid + bias, top 8, norm, x8) against the reference", rows, id_diff, near_ties, worst,
                ok ? "ok" : "FAIL");
    // not a failure: the card's weights and the host's route() need not be the same floats
    std::printf("%-66s %d of %d weights differ in their bits\n", "  route_rows against Inkling2::route() on the host",
                bits_diff, rows * kUsed);
}

void case_small() {
    {   // ink_mul_sigmoid
        const size_t n = 3 * kHeads * kDim;
        const auto x = randv(n, 1.0f), g = randv(n, 3.0f);
        Dev<float> dx(x), dg(g);
        ink_mul_sigmoid(dx, dg, n, stream());
        std::vector<double> ref(n);
        for (size_t i = 0; i < n; ++i) ref[i] = x[i] / (1.0 + std::exp(-(double) g[i]));
        compare("ink_mul_sigmoid", dx.get(), ref, 1e-6);
    }
    {   // ink_store_ring: rows past the ring's end go round
        const int n = kKv * kDim, rows = 9, ring = 16;
        const auto x = randv((size_t) rows * n, 2.0f);
        Dev<float> dx(x);
        Dev<uint16_t> cache((size_t) ring * n);
        ink_store_ring(dx, cache, n, 12, rows, ring, stream());
        const auto c = cache.get();
        std::vector<float> got((size_t) rows * n);
        std::vector<double> ref((size_t) rows * n);
        for (int t = 0; t < rows; ++t)
            for (int i = 0; i < n; ++i) {
                __half h;
                std::memcpy(&h, &c[(size_t) ((12 + t) % ring) * n + i], 2);
                got[(size_t) t * n + i] = __half2float(h);
                ref[(size_t) t * n + i] = __half2float(__float2half(x[(size_t) t * n + i]));
            }
        compare("ink_store_ring: halves at (pos + t) % ring", got, ref, 0.0);
    }
    {   // ink_softcap: the cap, and -infinity past the unpadded vocabulary
        const int n_all = 201024, n_keep = 200058, rows = 2;
        const auto x = randv((size_t) rows * n_all, 20.0f);
        Dev<float> dx(x);
        ink_softcap(dx, n_keep, n_all, rows, 30.0f, stream());
        const auto got = dx.get();
        std::vector<float> keep;
        std::vector<double> ref;
        int pads_ok = 0;
        for (int r = 0; r < rows; ++r)
            for (int i = 0; i < n_all; ++i) {
                const float g = got[(size_t) r * n_all + i];
                if (i < n_keep) keep.push_back(g), ref.push_back(30.0 * std::tanh(x[(size_t) r * n_all + i] / 30.0));
                else pads_ok += g == -INFINITY;
            }
        // the worst seen is 2.3 units in the last place; the tolerance is 1e-5 absolute (~5 ulp of a value near 30)
        compare("ink_softcap (cap 30)", keep, ref, 1e-5);
        compare("ink_softcap: the rows past the vocabulary are -inf", std::vector<float>{(float) pads_ok},
                std::vector<double>{(double) rows * (n_all - n_keep)}, 0.0);
    }
}

// ---- the attention ----

// A cache of `cells` rows (a ring, or a row a position) holding positions [0, upto): random keys and values,
// stored as the engine stores them (ink_clips(), pass-through)
struct Cache {
    bool kv8 = false;
    int ring = 0, cells = 0;
    Dev<uint8_t> keys, vals;
    std::vector<float> k, v;   // as the kernels read them, by cell
    Cache(bool fp8, int ring_, int cells_, long long upto)
        : kv8(fp8), ring(ring_), cells(cells_), keys(cache_bytes(fp8, cells_)), vals(cache_bytes(fp8, cells_)) {
        const int chunk = 256;
        for (long long p0 = std::max(0LL, ring ? upto - ring : 0LL); p0 < upto; p0 += chunk) {
            const int rows = (int) std::min<long long>(chunk, upto - p0);
            Dev<float> dk(randv((size_t) rows * kKv * kDim, 0.7f)), dv(randv((size_t) rows * kKv * kDim, 1.0f));
            InkClips st;
            st.x[0] = dk, st.x[1] = dv, st.thr[0] = st.thr[1] = 1e30f, st.dim[0] = st.dim[1] = kDim;
            st.heads[0] = st.heads[1] = kKv, st.pos = p0, st.rows = rows, st.ring = ring, st.kv8 = kv8;
            st.cache[0] = (uint16_t*) keys.p, st.cache[1] = (uint16_t*) vals.p;
            ink_clips(st, stream());
        }
        k = cache_rows(keys.get(), kv8, cells);
        v = cache_rows(vals.get(), kv8, cells);
    }
    const float* key(long long p, int g) const { return &k[((size_t) (ring ? p % ring : p) * kKv + g) * kDim]; }
    const float* val(long long p, int g) const { return &v[((size_t) (ring ? p % ring : p) * kKv + g) * kDim]; }
};

struct Inputs {
    std::vector<float> q, r, sink, gate, table;
    Dev<float> dq, dr, dsink, dgate, dtable;
    explicit Inputs(int rows)
        : q(randv((size_t) rows * kHeads * kDim, 0.7f)), r(randv((size_t) rows * kHeads * kRel, 1.2f)),
          sink(randv((size_t) rows * kHeads, 1.0f)), gate(randv((size_t) rows * kHeads * kDim, 2.0f)),
          table(randv((size_t) kExtent * kRel, 0.4f)), dq(q), dr(r), dsink(sink), dgate(gate), dtable(table) {}
};

// The attention of `rows` tokens from pos in double: scores scale q.k + r . table[d] (d < extent) - sink over the
// keys (pos + t - window, pos + t], a softmax with one more term of score 0, the values' sum, times sigmoid(gate);
// q_half: the queries rounded to halves, as the tensor cores' paths take them
std::vector<double> h_attend(const Cache& c, const Inputs& in, long long pos, int rows, int window, bool gate,
                             bool q_half) {
    const double scale = 1.0 / std::sqrt((double) kDim);
    std::vector<double> out((size_t) rows * kHeads * kDim);
    std::vector<double> s;
    for (int t = 0; t < rows; ++t) {
        const long long p = pos + t, lo = window > 0 ? std::max(0LL, p - window + 1) : 0;
        for (int h = 0; h < kHeads; ++h) {
            const int g = h / (kHeads / kKv);
            const float* q = &in.q[((size_t) t * kHeads + h) * kDim];
            const float* r = &in.r[((size_t) t * kHeads + h) * kRel];
            s.assign((size_t) (p - lo + 1), 0.0);
            double m = 0.0;   // the extra term's score
            for (long long j = lo; j <= p; ++j) {
                const float* k = c.key(j, g);
                double d = 0;
                for (int i = 0; i < kDim; ++i) d += (q_half ? (double) __half2float(__float2half(q[i])) : (double) q[i]) * k[i];
                d *= scale;
                const long long dist = p - j;
                if (dist < kExtent)
                    for (int i = 0; i < kRel; ++i) d += (double) r[i] * in.table[(size_t) dist * kRel + i];
                d -= in.sink[(size_t) t * kHeads + h];
                s[(size_t) (j - lo)] = d;
                m = std::max(m, d);
            }
            double z = std::exp(-m);
            for (double& x : s) z += x = std::exp(x - m);
            double* o = &out[((size_t) t * kHeads + h) * kDim];
            for (long long j = lo; j <= p; ++j) {
                const float* v = c.val(j, g);
                const double wgt = s[(size_t) (j - lo)] / z;
                for (int i = 0; i < kDim; ++i) o[i] += wgt * v[i];
            }
            if (gate)
                for (int i = 0; i < kDim; ++i)
                    o[i] /= 1.0 + std::exp(-(double) in.gate[((size_t) t * kHeads + h) * kDim + i]);
        }
    }
    return out;
}

InkAttn attn_of(const Cache& c, const Inputs& in, long long pos, int rows, int window, float* out, float* score,
                size_t score_n) {
    InkAttn a;
    a.q = in.dq, a.r = in.dr, a.sink = in.dsink, a.keys = (const uint16_t*) c.keys.p, a.vals = (const uint16_t*) c.vals.p;
    a.rel_bias = in.dtable, a.gate = in.dgate, a.out = out, a.score = score, a.score_n = score_n, a.pos = pos;
    a.rows = rows, a.n_head = kHeads, a.n_kv = kKv, a.dim = kDim, a.n_rel = kRel, a.extent = kExtent;
    a.window = window, a.ring = c.ring, a.scale = 1.f / std::sqrt((float) kDim), a.kv8 = c.kv8;
    return a;
}

// ink_attend() (a token and a few at a time, the decode path): window (a ring that has gone round) and full (past
// the bias table's extent), fp16 and FP8, against the reference; the positions on the card against by value, a
// call in chunks against one, and a token's output quantized (native_quantize_q8_1()'s bytes): the same bits
void case_attend() {
    for (const bool kv8 : {false, true}) {
#if defined(__HIPCC__)
        if (kv8) {
            skipped("ink_attend, FP8 caches", "no FP8 caches under HIP");
            continue;
        }
#endif
        for (const bool swa : {true, false}) {
            const long long upto = 2200;   // positions [0, upto) in the cache
            const int ring = swa ? kWindow + 32 : 0;
            const Cache c(kv8, ring, swa ? ring : (int) upto, upto);
            struct Call {
                long long pos;
                int rows;
            };
            for (const Call call : {Call{0, 1}, Call{6, 3}, Call{upto - 1, 1}, Call{upto - 7, 7}, Call{upto - 31, 31}}) {
                const int rows = call.rows, window = swa ? kWindow : 0;
                Inputs in(rows);
                const size_t per = ink_attend_floats(kHeads, kKv, kDim, upto);
                Dev<float> score(per * rows), score2(per * 2), out((size_t) rows * kHeads * kDim),
                    out2((size_t) rows * kHeads * kDim);
                InkAttn a = attn_of(c, in, call.pos, rows, window, out, score, score.n);
                ink_attend(a, stream());
                const std::string tag = fmt("ink_attend %s %s, %d rows at %lld", kv8 ? "fp8" : "fp16",
                                            swa ? "window" : "full", rows, call.pos);
                compare(tag, out.get(), h_attend(c, in, call.pos, rows, window, true, false), 2e-5);
                if (rows > 1) {   // two tokens a chunk
                    InkAttn b = attn_of(c, in, call.pos, rows, window, out2, score2, score2.n);
                    ink_attend(b, stream());
                    same_bits(tag + ": in chunks of 2 = in one", out2.get(), out.get());
                }
                if (rows <= 8) {   // a recorded launch's: the positions from the card, room for the most rows read
                    std::vector<long long> at((size_t) 2 * rows);
                    for (int t = 0; t < rows; ++t) at[(size_t) 2 * t] = call.pos + t;
                    Dev<long long> dat(at);
                    Dev<float> out3((size_t) rows * kHeads * kDim);
                    InkAttn b = attn_of(c, in, 0, rows, window, out3, score, score.n);
                    b.at = dat, b.room = (int) upto;
                    ink_attend(b, stream());
                    same_bits(tag + ": positions on the card = by value", out3.get(), out.get());
                }
                if (rows == 1) {   // and quantized for the output product
                    Dev<uint8_t> q8(nk::native_q8_1_bytes(kHeads * kDim)), q82(nk::native_q8_1_bytes(kHeads * kDim));
                    Dev<float> out3((size_t) kHeads * kDim);
                    InkAttn b = attn_of(c, in, call.pos, 1, window, out3, score, score.n);
                    b.q8 = q8;
                    ink_attend(b, stream());
                    nk::native_quantize_q8_1(out3, q82, kHeads * kDim, 1, stream());
                    same_bits(tag + ": q8_1 = native_quantize_q8_1()", q8.get(), q82.get());
                    same_bits(tag + ": with q8_1 = without", out3.get(), out.get());
                }
            }
        }
    }
}

// A prompt batch's attention: attend_batch() with its bias (the path of an fp16 cache on cards without
// ink_attend_batch(): Turing, and HIP) and ink_attend_batch() (Ampere on, fp16 and FP8), window (ring) and full,
// from the start and with context before, against the reference with the queries rounded to halves
void case_attend_batch() {
    void* blas = blas_create(stream());
    for (const bool swa : {true, false}) {
        struct Call {
            long long pos;
            int rows;
        };
        for (const Call call : {Call{0, 100}, Call{1200, 64}, Call{1500, 32}}) {
            const int rows = call.rows, window = swa ? kWindow : 0;
            const long long upto = call.pos + rows;
            const int ring = swa ? kWindow + rows : 0;
            for (const bool kv8 : {false, true}) {
                const std::string tag = fmt("%s %s, %d rows at %lld", kv8 ? "fp8" : "fp16", swa ? "window" : "full",
                                            rows, call.pos);
#if defined(__HIPCC__)
                if (kv8) continue;
#endif
                const Cache c(kv8, ring, swa ? ring : (int) upto, upto);
                Inputs in(rows);
                const auto ref = h_attend(c, in, call.pos, rows, window, true, true);
                Dev<float> out((size_t) rows * kHeads * kDim);
                InkAttn a = attn_of(c, in, call.pos, rows, window, out, nullptr, 0);
                if (ink_attend_batch(a, stream())) {
                    // the weights are rounded to halves for the values' product: a few units in 2^-11 of a value
                    compare("ink_attend_batch " + tag, out.get(), ref, 4e-3);
                } else {
                    const char* v = std::getenv("STRATA_INK_FLASH");   // read as ink_attend_batch() reads it
                    skipped("ink_attend_batch " + tag,
                            v && v[0] == '0' ? "STRATA_INK_FLASH=0" : "not on this card (from Ampere on, not HIP)");
                }
                if (kv8) continue;   // attend_batch() reads fp16 only
                const size_t score_n = size_t(16) << 20, heads = (size_t) rows * kHeads * kDim;
                Dev<float> score(score_n), work(heads + heads / 2 + 1 + 3 * (size_t) rows * kHeads), zeros(kHeads),
                    out2(heads);
                AttnBias bias;
                bias.r = in.dr, bias.sink = in.dsink, bias.table = in.dtable, bias.n_rel = kRel, bias.extent = kExtent;
                attend_batch(blas, in.dq, (const uint16_t*) c.keys.p, (const uint16_t*) c.vals.p, call.pos, rows, score,
                             score_n, work, out2, kDim, kHeads, kKv, 1.f / std::sqrt((float) kDim), stream(), window,
                             ring, false, {}, {}, nullptr, nullptr, 0, zeros, nullptr, &bias);
                ink_mul_sigmoid(out2, in.dgate, heads, stream());
                compare("attend_batch + bias " + tag, out2.get(), ref, 4e-3);
            }
        }
    }
    blas_destroy(blas);
}

// ---- the list ----

struct Case {
    const char* name;
    void (*run)();
};
const Case kCases[] = {
    {"clip", case_clip},
    {"head_rms", case_head_rms},
    {"clips", case_clips},
    {"kv8_store", case_kv8_store},
    {"conv", case_conv},
    {"conv_acc", case_conv_acc},
    {"moe_conv_add", case_moe_conv_add},
    {"qkv", case_qkv},
    {"mix", case_mix},
    {"norms", case_norms},
    {"controllers", case_controllers},
    {"situ", case_situ},
    {"router", case_router},
    {"small", case_small},
    {"attend", case_attend},
    {"attend_batch", case_attend_batch},
};

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> only;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--list") {
            for (const Case& c : kCases) std::printf("%s\n", c.name);
            return 0;
        }
        bool known = false;
        for (const Case& c : kCases) known = known || std::string(c.name) == argv[i];
        if (!known) {
            std::fprintf(stderr, "poly_inkling2_test: no case %s; the cases:", argv[i]);
            for (const Case& c : kCases) std::fprintf(stderr, " %s", c.name);
            std::fprintf(stderr, "\n");
            return 2;
        }
        only.emplace_back(argv[i]);
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
        std::printf("poly_inkling2_test: no card\n");
        return 77;
    }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
#if defined(__HIPCC__)
    std::printf("poly_inkling2_test on %s (%s)\n", prop.name, prop.gcnArchName);
#else
    std::printf("poly_inkling2_test on %s (sm_%d%d); ink_kv8_ok %d\n", prop.name, prop.major, prop.minor,
                (int) ink_kv8_ok());
#endif
    for (const Case& c : kCases) {
        bool run = only.empty();
        for (const std::string& o : only) run = run || std::string(c.name) == o;
        if (!run) continue;
        std::printf("-- %s\n", c.name);
        try {
            c.run();
            sync();
        } catch (const std::exception& e) {
            ++failures;
            std::printf("%s threw: %s  FAIL\n", c.name, e.what());
            (void) cudaGetLastError();
        }
    }
    std::printf("%d checks, %d failed", checks, failures);
    if (codegen_failures)
        std::printf(" (%d of them [codegen]: the compiler's rounding, not a wrong result)", codegen_failures);
    std::printf("\n");
    if (checks == 0) {
        std::printf("no checks ran  FAIL\n");
        return 1;
    }
    return failures ? 1 : 0;
}
