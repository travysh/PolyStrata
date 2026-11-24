// src/poly/iq3_lanes_test.cu - the cached experts' IQ3_S and IQ3_XXS slots launches with MmvqSlots::lanes (Inkling 2,
// CUDA) against the same launches without it (mmvq_slots_kernel) and against one native_mmvq() a matrix, to the bit:
// random blocks, 1 to 8 matrices a launch, row counts that leave a warp's last rows short, n_in 256 to 2304, through
// the host's list (MmvqSlots) and the card's picks (MmvqIds).  --bench also times the 8-expert launches.
//
// usage: poly_iq3_lanes_test [--bench]   (exit 0: all agree)
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

namespace k = strata::kernels;

int failures = 0;
std::mt19937 rng(1234);

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T> T* up(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, std::max<size_t>(1, v.size()) * sizeof(T)), "malloc");
    if (!v.empty()) ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

void same_bits(const std::string& what, const float* a, const float* b, size_t n) {
    std::vector<float> x(n), y(n);
    ck(cudaDeviceSynchronize(), "sync");
    ck(cudaMemcpy(x.data(), a, n * 4, cudaMemcpyDeviceToHost), "download");
    ck(cudaMemcpy(y.data(), b, n * 4, cudaMemcpyDeviceToHost), "download");
    size_t diff = 0;
    double most = 0, most_rel = 0, top = 0;   // the largest |x - y|, |x - y| / |x|, and |x|
    for (size_t i = 0; i < n; ++i) {
        diff += std::memcmp(&x[i], &y[i], 4) != 0;
        const double d = std::fabs((double) x[i] - (double) y[i]), m = std::fabs((double) x[i]);
        most = std::max(most, d), top = std::max(top, m);
        if (m > 0) most_rel = std::max(most_rel, d / m);
    }
    std::printf("%-58s %zu of %zu values differ  %s\n", what.c_str(), diff, n, diff ? "FAIL" : "ok");
    if (diff) std::printf("    max |a - b| %.3g (largest |a| %.3g), max |a - b| / |a| %.3g\n", most, top, most_rel);
    if (diff) ++failures;
}

// random bytes with a sane fp16 d at the start of each 256-value block (IQ3_S and IQ3_XXS both keep it there)
void* random_weights(int type, int n_in, int n_out) {
    const size_t blk = k::native_mmvq_weight_bytes(type, 256, 1);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(1e-3f, 1e-2f);
    std::vector<uint8_t> h(k::native_mmvq_weight_bytes(type, n_in, n_out));
    for (uint8_t& c : h) c = (uint8_t) byte(rng);
    for (size_t b = 0; b < h.size() / blk; ++b) {
        const __half d = __float2half(scale(rng));
        std::memcpy(h.data() + b * blk, &d, 2);
    }
    return up(h);
}

// n_in values of each of `rows` tokens in Q8_1 on the card
void* random_q8_1(int n_in, int rows, cudaStream_t st) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> h(size_t(n_in) * rows);
    for (float& v : h) v = nd(rng);
    float* x = up(h);
    void* q;
    ck(cudaMalloc(&q, k::native_q8_1_bytes(n_in, rows)), "malloc");
    k::native_quantize_q8_1(x, q, n_in, rows, st);
    ck(cudaStreamSynchronize(st), "quantize");
    cudaFree(x);
    return q;
}

void test_slots_lanes(int type, const char* name, int n_in, int n_out, int count) {
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    std::vector<void*> w((size_t) count);
    for (void*& p : w) p = random_weights(type, n_in, n_out);
    void* xq = random_q8_1(n_in, count, st);
    const size_t xr = k::native_q8_1_bytes(n_in), total = size_t(n_out) * count;
    float *ya, *yb, *yc, *yd;
    for (float** p : {&ya, &yb, &yc, &yd}) ck(cudaMalloc(p, total * sizeof(float)), "malloc");
    k::MmvqSlots slots{}, plain{};
    slots.lanes = true;
    for (int i = 0; i < count; ++i) {
        const void* x = (const uint8_t*) xq + i * xr;
        k::native_mmvq(type, w[(size_t) i], x, ya + size_t(i) * n_out, n_in, n_out, 1, st);
        slots.w[i] = plain.w[i] = w[(size_t) i], slots.x[i] = plain.x[i] = x;
        slots.y[i] = yb + size_t(i) * n_out, plain.y[i] = yd + size_t(i) * n_out;
    }
    // the same through the card's picks: expert i is table[ids[i]], the table backwards
    std::vector<const void*> table(w.rbegin(), w.rend());
    std::vector<int32_t> ids((size_t) count);
    for (int i = 0; i < count; ++i) ids[(size_t) i] = count - 1 - i;
    void* dtable;
    ck(cudaMalloc(&dtable, table.size() * sizeof(void*)), "malloc");
    ck(cudaMemcpy(dtable, table.data(), table.size() * sizeof(void*), cudaMemcpyHostToDevice), "table");
    int32_t* dids = up(ids);
    const k::MmvqIds picks{(const void* const*) dtable, dids, 0, xq, xr, 1, yc, size_t(n_out), true};
    const std::string what = std::string("slots lanes ") + name + " " + std::to_string(n_in) + " -> " +
                             std::to_string(n_out) + " x" + std::to_string(count);
    if (!k::native_mmvq_slots(type, slots, count, n_in, n_out, st) ||
        !k::native_mmvq_ids(type, picks, count, n_in, n_out, st) ||
        !k::native_mmvq_slots(type, plain, count, n_in, n_out, st)) {
        std::printf("%-58s not taken  FAIL\n", what.c_str());
        ++failures;
    } else {
        same_bits(what, ya, yb, total);
        same_bits(what + ", picks", ya, yc, total);
        same_bits(what + ", without lanes", yd, yb, total);
    }
    for (void* p : w) cudaFree(p);
    cudaFree(xq), cudaFree(dtable), cudaFree(dids);
    cudaFree(ya), cudaFree(yb), cudaFree(yc), cudaFree(yd);
    cudaStreamDestroy(st);
}

// the best of 5 replays of a graph of `reps` launches, a launch
template <class F> float time_us(cudaStream_t s, F&& f, int reps = 200) {
    f();
    ck(cudaStreamSynchronize(s), "warm-up");
    cudaGraph_t gr;
    cudaGraphExec_t ex;
    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed), "capture");
    for (int i = 0; i < reps; ++i) f();
    ck(cudaStreamEndCapture(s, &gr), "end capture");
    ck(cudaGraphInstantiate(&ex, gr, 0), "instantiate");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    float best = 1e30f;
    for (int i = 0; i < 6; ++i) {
        cudaEventRecord(e0, s);
        cudaGraphLaunch(ex, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (i) best = std::min(best, ms);   // the first replay uploads the graph
    }
    cudaEventDestroy(e0), cudaEventDestroy(e1);
    cudaGraphExecDestroy(ex), cudaGraphDestroy(gr);
    return best * 1e3f / reps;
}

// the cached experts' gate|up (2304 -> 2304), 8 experts a launch, as Engine::moe() launches them, with and without lanes
void bench() {
    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    const struct { int type; const char* name; } kinds[] = {{21, "IQ3_S"}, {18, "IQ3_XXS"}};
    for (const auto& kd : kinds) {
        const int n_in = 2304, n_out = 2304;
        std::vector<void*> w(8);
        for (void*& p : w) p = random_weights(kd.type, n_in, n_out);
        void* xq = random_q8_1(n_in, 8, st);
        float* y;
        ck(cudaMalloc(&y, size_t(8) * n_out * 4), "malloc");
        for (bool lanes : {false, true}) {
            k::MmvqSlots sl{};
            sl.lanes = lanes;
            for (int i = 0; i < 8; ++i)
                sl.w[i] = w[(size_t) i], sl.x[i] = (const uint8_t*) xq + i * k::native_q8_1_bytes(n_in),
                sl.y[i] = y + size_t(i) * n_out;
            const float us = time_us(st, [&] { k::native_mmvq_slots(kd.type, sl, 8, n_in, n_out, st); });
            std::printf("bench slots %-8s %d -> %d x8%-7s %8.1f us %7.1f GB/s\n", kd.name, n_in, n_out,
                        lanes ? ", lanes" : "", us, 8.0 * k::native_mmvq_weight_bytes(kd.type, n_in, n_out) / (us * 1e3));
        }
        for (void* p : w) cudaFree(p);
        cudaFree(xq), cudaFree(y);
    }
    cudaStreamDestroy(st);
}

}  // namespace

int main(int argc, char** argv) {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    // STRATA_IQ3_LANES=0 turns the lane kernel off, so every case would compare the old kernel with itself
    if (const char* v = std::getenv("STRATA_IQ3_LANES"); v && v[0] == '0') {
        std::printf("STRATA_IQ3_LANES=0 is set, so the lane kernel is off: skipped\n");
        return 77;
    }
    // unset, the lane kernel runs on sm_75 and sm_89 only (iq_kernels.cu: iq3_lanes_on()); STRATA_IQ3_LANES=1 checks it
    // on another card
    if (const char* v = std::getenv("STRATA_IQ3_LANES"); !(v && *v)) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
        if (!((major == 7 && minor == 5) || (major == 8 && minor == 9))) {
            std::printf("sm_%d%d: the lane kernel is off by default here (STRATA_IQ3_LANES=1 checks it): skipped\n",
                        major, minor);
            return 77;
        }
    }
    // merbanan's AD-IQ3_S file (IQ3_S gate and up) and UD-IQ3_XXS: a warp's rows as one run of calls, and its tails
    for (int count : {1, 8}) {
        test_slots_lanes(21, "IQ3_S", 2304, 2304, count);
        test_slots_lanes(18, "IQ3_XXS", 2304, 2304, count);
    }
    test_slots_lanes(21, "IQ3_S", 2304, 2301, 8);
    test_slots_lanes(21, "IQ3_S", 1024, 37, 3);
    test_slots_lanes(18, "IQ3_XXS", 1280, 2299, 5);
    test_slots_lanes(18, "IQ3_XXS", 256, 7, 2);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) bench();
    std::printf("%s: %d failures\n", failures ? "FAIL" : "ok", failures);
    return failures ? 1 : 0;
}
