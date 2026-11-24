// src/poly/project_mixed_test.cu - a token's attention products in one launch (Inkling 2's project(), CUDA) against
// one native_mmvq() a matrix, to the bit: native_mmvq_mixed() for Q8_0, Q4_K, Q5_K and Q6_K matrices of one width
// (the five of Unsloth's layers, which mix them, plus the sink and the two convolution taps), and native_mmvq_many()
// for the eight of a ggml-org layer, all Q8_0.  Random blocks, each matrix on a Q8_1 input of its own or shared.
// --bench also times the launches a token's products took before (one quantize and one product a matrix, or the
// five in one launch and three alone) against the one launch.
//
// usage: poly_project_mixed_test [--bench]   (exit 0: all agree; 77: no card)
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

namespace k = strata::kernels;

int failures = 0;
std::mt19937 rng(4321);

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

// random bytes with sane fp16 scales where each format keeps them: Q8_0's d at a 32-value block's start, Q4_K's and
// Q5_K's d and dmin at a 256-value block's start, Q6_K's d at its end
void* random_weights(int type, int n_in, int n_out) {
    const int qk = type == 8 ? 32 : 256;
    const size_t blk = k::native_mmvq_weight_bytes(type, qk, 1);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(1e-3f, 1e-2f);
    std::vector<uint8_t> h(k::native_mmvq_weight_bytes(type, n_in, n_out));
    for (uint8_t& c : h) c = (uint8_t) byte(rng);
    for (size_t b = 0; b < h.size() / blk; ++b) {
        uint8_t* p = h.data() + b * blk;
        const __half d = __float2half(scale(rng)), m = __float2half(scale(rng));
        if (type == 14) {
            std::memcpy(p + blk - 2, &d, 2);
        } else {
            std::memcpy(p, &d, 2);
            if (type != 8) std::memcpy(p + 2, &m, 2);
        }
    }
    return up(h);
}

void* random_q8_1(int n_in, cudaStream_t st) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> h((size_t) n_in);
    for (float& v : h) v = nd(rng);
    float* x = up(h);
    void* q;
    ck(cudaMalloc(&q, k::native_q8_1_bytes(n_in)), "malloc");
    k::native_quantize_q8_1(x, q, n_in, 1, st);
    ck(cudaStreamSynchronize(st), "quantize");
    cudaFree(x);
    return q;
}

struct Case {
    std::string name;
    int n_in;
    std::vector<int> types, n_out, input;   // matrix j: its type, rows and input (0..2)
};

struct Built {
    std::vector<void*> w;
    void* xq[3];
    std::vector<float*> ya, yb;
    k::NativeMmvqMany m{};
};

Built build(const Case& c, cudaStream_t st) {
    Built b;
    for (int i = 0; i < 3; ++i) b.xq[i] = random_q8_1(c.n_in, st);
    for (size_t j = 0; j < c.types.size(); ++j) {
        b.w.push_back(random_weights(c.types[j], c.n_in, c.n_out[j]));
        float *a, *bb;
        ck(cudaMalloc(&a, c.n_out[j] * sizeof(float)), "malloc");
        ck(cudaMalloc(&bb, c.n_out[j] * sizeof(float)), "malloc");
        ck(cudaMemset(a, 0xff, c.n_out[j] * sizeof(float)), "memset");
        ck(cudaMemset(bb, 0x7f, c.n_out[j] * sizeof(float)), "memset");
        b.ya.push_back(a), b.yb.push_back(bb);
        b.m.w[j] = b.w[j], b.m.x_q8_1[j] = b.xq[c.input[j]], b.m.y[j] = bb, b.m.n_out[j] = c.n_out[j];
    }
    b.m.count = (int) c.types.size();
    return b;
}

void release(Built& b) {
    for (void* p : b.w) cudaFree(p);
    for (void* p : b.xq) cudaFree(p);
    for (float* p : b.ya) cudaFree(p);
    for (float* p : b.yb) cudaFree(p);
}

bool one_type(const Case& c) {
    for (int t : c.types)
        if (t != c.types[0]) return false;
    return true;
}

void check(const Case& c, bool many) {
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    Built b = build(c, st);
    for (size_t j = 0; j < c.types.size(); ++j)
        k::native_mmvq(c.types[j], b.w[j], b.xq[c.input[j]], b.ya[j], c.n_in, c.n_out[j], 1, st);
    const std::string what = std::string(many ? "many  " : "mixed ") + c.name;
    bool taken;
    if (many) {
        taken = k::native_mmvq_many_fits(c.types[0], b.m, c.n_in);
        if (taken) k::native_mmvq_many(c.types[0], b.m, c.n_in, st);
    } else {
        taken = k::native_mmvq_mixed_fits(c.types.data(), b.m, c.n_in) &&
                k::native_mmvq_mixed(c.types.data(), b.m, c.n_in, st);
    }
    if (!taken) {
        std::printf("%-64s not taken  FAIL\n", what.c_str());
        ++failures;
    } else {
        ck(cudaStreamSynchronize(st), "sync");
        size_t diff = 0, total = 0;
        for (size_t j = 0; j < c.types.size(); ++j) {
            std::vector<float> x((size_t) c.n_out[j]), y((size_t) c.n_out[j]);
            ck(cudaMemcpy(x.data(), b.ya[j], x.size() * 4, cudaMemcpyDeviceToHost), "download");
            ck(cudaMemcpy(y.data(), b.yb[j], y.size() * 4, cudaMemcpyDeviceToHost), "download");
            for (size_t i = 0; i < x.size(); ++i) diff += std::memcmp(&x[i], &y[i], 4) != 0;
            total += x.size();
        }
        std::printf("%-64s %zu of %zu values differ  %s\n", what.c_str(), diff, total, diff ? "FAIL" : "ok");
        if (diff) ++failures;
    }
    release(b);
    ck(cudaStreamDestroy(st), "stream");
}

// the declines: a type it has no kernel for, a row too short for the one-row-a-block layout
void declines() {
    k::NativeMmvqMany m{};
    m.count = 2;
    for (int j = 0; j < 2; ++j) m.w[j] = m.x_q8_1[j] = (void*) 0x1000, m.y[j] = (float*) 0x1000, m.n_out[j] = 64;
    const int iq3[2] = {8, 18}, short_q4k[2] = {8, 12}, ok[2] = {8, 14};
    const bool a = k::native_mmvq_mixed_fits(iq3, m, 2304), b = k::native_mmvq_mixed_fits(short_q4k, m, 1792),
               c = k::native_mmvq_mixed_fits(ok, m, 2304);
    std::printf("%-64s %s\n", "mixed declines IQ3_XXS, Q4_K of 7 blocks; takes Q8_0 + Q6_K", !a && !b && c ? "ok" : "FAIL");
    if (a || b || !c) ++failures;
}

float time_us(const std::function<void()>& f, cudaStream_t st) {
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    for (int i = 0; i < 20; ++i) f();
    ck(cudaEventRecord(e0, st), "event");
    const int reps = 200;
    for (int i = 0; i < reps; ++i) f();
    ck(cudaEventRecord(e1, st), "event");
    ck(cudaEventSynchronize(e1), "event");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, e0, e1), "event");
    cudaEventDestroy(e0), cudaEventDestroy(e1);
    return 1000.f * ms / reps;
}

// before: what Inkling2::layer_head() launched for these (one quantize and one product a matrix where the types
// differ; the five in one launch and the others alone, each with its own quantize, where they are all Q8_0); after:
// one launch.  The inputs' float rows are quantized again before as lin() does.
void bench(const Case& c) {
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    Built b = build(c, st);
    float* xf;
    ck(cudaMalloc(&xf, c.n_in * sizeof(float)), "malloc");
    ck(cudaMemset(xf, 0, c.n_in * sizeof(float)), "memset");
    void* q;
    ck(cudaMalloc(&q, k::native_q8_1_bytes(c.n_in)), "malloc");
    const size_t n = c.types.size();
    const auto before = [&] {
        size_t j0 = 0;
        if (one_type(c)) {
            k::NativeMmvqMany five = b.m;
            five.count = 5;
            k::native_mmvq_many(c.types[0], five, c.n_in, st);
            j0 = 5;
        }
        for (size_t j = j0; j < n; ++j) {
            k::native_quantize_q8_1(xf, q, c.n_in, 1, st);
            k::native_mmvq(c.types[j], b.w[j], q, b.ya[j], c.n_in, c.n_out[j], 1, st);
        }
    };
    const auto after = [&] {
        if (one_type(c)) k::native_mmvq_many(c.types[0], b.m, c.n_in, st);
        else k::native_mmvq_mixed(c.types.data(), b.m, c.n_in, st);
    };
    const float t0 = time_us(before, st), t1 = time_us(after, st);
    std::printf("bench %-58s before %7.1f us, after %7.1f us\n", c.name.c_str(), t0, t1);
    cudaFree(xf), cudaFree(q);
    release(b);
    ck(cudaStreamDestroy(st), "stream");
}

}  // namespace

int main(int argc, char** argv) {
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd == 0) {
        std::printf("no CUDA card: skipped\n");
        return 77;
    }
    cudaDeviceProp p{};
    ck(cudaGetDeviceProperties(&p, 0), "props");
    std::printf("%s (sm_%d%d)\n", p.name, p.major, p.minor);
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    // Inkling 2 Mini: q 3072, k 1024, v 1024, rel 384, gate 3072 (inputs 0 1 2 0 0); sink 24 (0); taps 32, 32 (1, 2)
    const std::vector<int> outs8 = {3072, 1024, 1024, 384, 3072, 24, 32, 32}, in8 = {0, 1, 2, 0, 0, 0, 1, 2};
    const std::vector<int> outs5(outs8.begin(), outs8.begin() + 5), in5(in8.begin(), in8.begin() + 5);
    std::vector<Case> mixed = {
        {"Q6_K Q8_0 Q8_0 Q6_K Q6_K (UD-Q2_K_XL)", 2304, {14, 8, 8, 14, 14}, outs5, in5},
        {"Q4_K Q4_K Q5_K Q4_K Q4_K", 2304, {12, 12, 13, 12, 12}, outs5, in5},
        {"Q5_K Q5_K Q5_K Q5_K Q5_K", 2304, {13, 13, 13, 13, 13}, outs5, in5},
        {"Q8_0 Q4_K Q5_K Q6_K Q8_0 + sink, taps Q8_0", 2304, {8, 12, 13, 14, 8, 8, 8, 8}, outs8, in8},
        {"Q6_K x8, one input", 2304, {14, 14, 14, 14, 14, 14, 14, 14}, outs8, {0, 0, 0, 0, 0, 0, 0, 0}},
        {"Q8_0 Q4_K, rows 37 and 5 (short)", 2304, {8, 12}, {37, 5}, {0, 1}},
        {"Q8_0 Q6_K at 4096 inputs", 4096, {8, 14, 8}, {300, 129, 1}, {0, 1, 2}},
        {"Q8_0 alone", 2304, {8}, {2304}, {0}},
    };
    std::vector<Case> many = {
        {"Q8_0 x8 (ggml-org IQ4_NL full layer)", 2304, std::vector<int>(8, 8), outs8, in8},
        {"Q8_0 x6 (window layer: five + sink)", 2304, std::vector<int>(6, 8),
         std::vector<int>(outs8.begin(), outs8.begin() + 6), std::vector<int>(in8.begin(), in8.begin() + 6)},
    };
    for (const Case& c : mixed) check(c, false);
    for (const Case& c : many) check(c, true);
    declines();
    if (do_bench) {
        bench(mixed[0]);
        bench(mixed[1]);
        bench(many[0]);
        bench(many[1]);
    }
    std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
