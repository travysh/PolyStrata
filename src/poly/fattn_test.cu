// src/poly/fattn_test.cu - fattn_q8() (llama.cpp's flash-attention kernel, fattn.hpp) against attend_batch() on the
// same q8_0 cache, Qwen3.6's shape (heads of 256 values, 16 heads, 2 KV heads), and both against the arithmetic in
// doubles for three tokens of each batch.  Also with a scratch that holds only a few thousand keys, so that the
// keys go in many blocks and are merged.
//
// llama.cpp's kernel sums the values' products in halves (attend_batch() in floats), so it is the further from the
// doubles, by up to about 1e-2 on results up to 4 (measured on an RTX 3070).  The bounds catch a wrong mask or merge,
// which is off by tenths.
//
// usage: poly_fattn_test   (exit 0: every case within 2.5e-2 of attend_batch() and 1e-2 of the doubles)
#include "fattn.hpp"
#include "strata/poly/kernels.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

constexpr int kDim = 256, kHeads = 16, kKvHeads = 2, kWidth = kKvHeads * kDim, kMaxRows = 131072, kBatch = 1024;
constexpr size_t kScore = size_t(16) << 20;   // Qwen35's score_ (kScoreFloats)

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> download(const float* d, size_t n) {
    std::vector<float> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

struct Cache {
    std::vector<int8_t> codes;
    std::vector<uint16_t> scales;
    float at(size_t row, int i) const {
        return (float) codes[row * kWidth + i] * __half2float(__ushort_as_half(scales[row * (kWidth / 32) + i / 32]));
    }
};

Cache make(std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    Cache c;
    c.codes.resize(size_t(kMaxRows) * kWidth);
    c.scales.resize(size_t(kMaxRows) * (kWidth / 32));
    for (auto& x : c.codes) x = (int8_t) std::clamp(std::lround(40.f * nd(rng)), -127L, 127L);
    for (auto& x : c.scales) x = __half_as_ushort(__float2half(ud(rng) / 40.f));
    return c;
}

int check(cublasHandle_t blas, const Cache& K, const Cache& V, Kvq8 kq, Kvq8 vq, const std::vector<float>& q,
          const float* dq, float* score, float* awork, uint16_t* kwork, float* out, long long pos, int n,
          size_t fattn_floats) {
    const size_t cells = size_t(n) * kHeads * kDim;
    const float scale = 1.f / 16.f;
    const size_t halves = size_t(2048) * kWidth;
    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0), cudaEventCreate(&e1), cudaEventCreate(&e2), cudaEventCreate(&e3);
    cudaEventRecord(e0);
    attend_batch(blas, dq, nullptr, nullptr, pos, n, score, kScore, awork, out, kDim, kHeads, kKvHeads, scale, nullptr,
                 0, 0, false, kq, vq, kwork, kwork + halves, halves);
    cudaEventRecord(e1);
    const std::vector<float> ref = download(out, cells);
    cudaMemset(out, 0, cells * sizeof(float));
    cudaEventRecord(e2);
    if (!fattn_q8(dq, kq, vq, pos, n, out, kDim, kHeads, kKvHeads, scale, score, fattn_floats, nullptr)) {
        std::printf("FAIL pos %6lld rows %4d: fattn_q8 refused the scratch\n", pos, n);
        return 1;
    }
    cudaEventRecord(e3);
    const std::vector<float> fa = download(out, cells);
    float ms_ref = 0.f, ms_fa = 0.f;
    cudaEventElapsedTime(&ms_ref, e0, e1);
    cudaEventElapsedTime(&ms_fa, e2, e3);
    if (const cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) {
        std::printf("FAIL pos %6lld rows %4d: CUDA %s\n", pos, n, cudaGetErrorString(e));
        return 1;
    }
    float most = 0.f, big = 0.f;
    double sum = 0;
    for (size_t i = 0; i < cells; ++i) {
        const float d = std::fabs(fa[i] - ref[i]);
        most = std::max(most, std::isnan(d) ? INFINITY : d);
        sum += d;
        big = std::max(big, std::fabs(ref[i]));
    }
    // three tokens in doubles: the first, the middle and the last
    float most_fa = 0.f, most_ref = 0.f;
    for (int t : {0, n / 2, n - 1})
        for (int h = 0; h < kHeads; ++h) {
            const int g = h / (kHeads / kKvHeads);
            const float* qh = q.data() + (size_t(t) * kHeads + h) * kDim;
            const long long keys = pos + t + 1;
            std::vector<double> w(keys);
            double top = -1e300;
            for (long long r = 0; r < keys; ++r) {
                double s = 0;
                for (int i = 0; i < kDim; ++i) s += (double) qh[i] * K.at(r, g * kDim + i);
                w[r] = s * scale;
                top = std::max(top, w[r]);
            }
            double z = 0;
            for (auto& x : w) z += x = std::exp(x - top);
            for (int i = 0; i < kDim; ++i) {
                double o = 0;
                for (long long r = 0; r < keys; ++r) o += w[r] * V.at(r, g * kDim + i);
                o /= z;
                const size_t at = (size_t(t) * kHeads + h) * kDim + i;
                most_fa = std::max(most_fa, (float) std::fabs(fa[at] - o));
                most_ref = std::max(most_ref, (float) std::fabs(ref[at] - o));
            }
        }
    const bool ok = most <= 2.5e-2f && most_fa <= 1e-2f;
    std::printf("%s pos %6lld rows %4d scratch %5.1f MiB: %.2e from attend_batch (mean %.1e, results up to %.2f); "
                "against doubles: fattn %.2e, attend_batch %.2e; %.2f ms against %.2f\n",
                ok ? "ok  " : "FAIL", pos, n, fattn_floats * 4.0 / (1 << 20), most, sum / cells, big, most_fa,
                most_ref, ms_fa, ms_ref);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    if (!fattn_q8_fits(kDim, kHeads, kKvHeads)) {
        std::printf("no kernel for this card\n");
        return 1;
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q(size_t(kBatch) * kHeads * kDim);
    for (float& x : q) x = 2.f * nd(rng);   // spread scores, so the softmax has a few large weights
    const Cache K = make(rng), V = make(rng);
    int8_t *kc = upload(K.codes), *vc = upload(V.codes);
    uint16_t *ks = upload(K.scales), *vs = upload(V.scales);
    const Kvq8 kq{kc, ks}, vq{vc, vs};
    float* dq = upload(q);
    float *score, *awork, *out;
    uint16_t* kwork;
    const size_t heads = size_t(kBatch) * kHeads * kDim;
    cudaMalloc(&score, kScore * sizeof(float));
    cudaMalloc(&awork, (heads + heads / 2 + 1 + 3 * size_t(kBatch) * kHeads) * sizeof(float));
    cudaMalloc(&kwork, size_t(2) * 2048 * kWidth * sizeof(uint16_t));
    cudaMalloc(&out, heads * sizeof(float));
    cublasHandle_t blas;
    cublasCreate(&blas);
    cublasSetMathMode(blas, CUBLAS_TENSOR_OP_MATH);
    int bad = 0;
    struct Case { long long pos; int n; size_t floats; };
    // a few thousand keys a block: the batch's results (n * 4096 floats) and its metadata, then rows of 512 floats
    auto few = [](int n, int rows) { return size_t(n) * kHeads * (kDim + 4) + 64 + size_t(n) / 8 + size_t(rows) * kWidth; };
    const Case cases[] = {{0, 16, kScore},          {0, 64, kScore},          {0, kBatch, kScore},
                          {5, 37, kScore},          {1000, 517, kScore},      {4095, kBatch, kScore},
                          {30000, kBatch, kScore},  {30000, kBatch, few(kBatch, 4096)},
                          {30001, 100, few(100, 160)}, {120000, kBatch, kScore}};
    for (const Case& c : cases)
        bad += check(blas, K, V, kq, vq, q, dq, score, awork, kwork, out, c.pos, c.n, c.floats);
    cublasDestroy(blas);
    cudaFree(kc), cudaFree(vc), cudaFree(ks), cudaFree(vs), cudaFree(dq), cudaFree(score), cudaFree(awork);
    cudaFree(kwork), cudaFree(out);
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
