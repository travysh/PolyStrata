// src/poly/ctrl_fast_test.cu - ink_ctrl_down()'s one-pass kernel (STRATA_INK_FAST_CTRL=1, opt-in, other bits) against
// its default kernel and a float64 reference: ch[o] = down[o] . [cur * ia, msg * ib], ia and ib the rows' RMS scales
// (eps 1e-6), Inkling 2 Mini's shape (n_embd 2304, rank 64), 1 and 3 controllers, 1 to 4 rows, F32 and BF16 weights.
// --bench times the 3-controller token launch both ways.
//
// usage: poly_ctrl_fast_test [--bench]   (exit 0: within bounds; 77: no card)
#include "inkling2_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::poly;

namespace {

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", w, cudaGetErrorString(e));
        std::exit(1);
    }
}
constexpr int kEmbd = 2304, kRank = 64;
std::mt19937 rng(5);
int failures = 0;

std::vector<float> randf(size_t n, float sd) {
    std::normal_distribution<float> d(0.0f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
template <class T> T* up(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, v.size() * sizeof(T) + 16), "malloc");
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
float bf16_round(float x, uint16_t* b) {   // truncation to BF16 as stored; the kernels widen by a shift
    uint32_t u;
    std::memcpy(&u, &x, 4);
    *b = (uint16_t) (u >> 16);
    u &= 0xffff0000u;
    float y;
    std::memcpy(&y, &u, 4);
    return y;
}

void run_case(int count, int rows, bool bf16, bool bench) {
    std::vector<std::vector<float>> cur(count), msg(count), w(count);
    std::vector<std::vector<uint16_t>> wb(count);
    std::vector<float*> dc(count), dm(count), dy0(count), dy1(count);
    std::vector<const void*> dw(count);
    for (int j = 0; j < count; ++j) {
        cur[j] = randf((size_t) rows * kEmbd, 1.0f);
        msg[j] = randf((size_t) rows * kEmbd, 3.0f);
        w[j] = randf((size_t) kRank * 2 * kEmbd, 0.02f);
        if (bf16) {
            wb[j].resize(w[j].size());
            for (size_t i = 0; i < w[j].size(); ++i) w[j][i] = bf16_round(w[j][i], &wb[j][i]);
            dw[j] = up(wb[j]);
        } else {
            dw[j] = up(w[j]);
        }
        dc[j] = up(cur[j]), dm[j] = up(msg[j]);
        ck(cudaMalloc(&dy0[j], (size_t) rows * kRank * 4), "malloc");
        ck(cudaMalloc(&dy1[j], (size_t) rows * kRank * 4), "malloc");
    }
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    const auto launch = [&](bool fast, std::vector<float*>& y) {
        ink_set_fast_ctrl(fast);
        if (!ink_ctrl_down(dc.data(), dm.data(), dw.data(), y.data(), count, kEmbd, kRank, rows, st, bf16)) {
            std::printf("ink_ctrl_down declined\n");
            std::exit(1);
        }
    };
    launch(false, dy0);
    launch(true, dy1);
    ck(cudaStreamSynchronize(st), "sync");
    double diff = 0, peak = 0, e0 = 0, e1 = 0;
    for (int j = 0; j < count; ++j) {
        std::vector<float> a((size_t) rows * kRank), b(a.size());
        ck(cudaMemcpy(a.data(), dy0[j], a.size() * 4, cudaMemcpyDeviceToHost), "download");
        ck(cudaMemcpy(b.data(), dy1[j], b.size() * 4, cudaMemcpyDeviceToHost), "download");
        for (int t = 0; t < rows; ++t) {
            double sc = 0, sm = 0;
            for (int i = 0; i < kEmbd; ++i) {
                sc += (double) cur[j][(size_t) t * kEmbd + i] * cur[j][(size_t) t * kEmbd + i];
                sm += (double) msg[j][(size_t) t * kEmbd + i] * msg[j][(size_t) t * kEmbd + i];
            }
            const double ia = 1.0 / std::sqrt(sc / kEmbd + 1e-6), ib = 1.0 / std::sqrt(sm / kEmbd + 1e-6);
            for (int o = 0; o < kRank; ++o) {
                double r = 0;
                for (int i = 0; i < kEmbd; ++i) {
                    r += (double) w[j][(size_t) o * 2 * kEmbd + i] * cur[j][(size_t) t * kEmbd + i] * ia;
                    r += (double) w[j][(size_t) o * 2 * kEmbd + kEmbd + i] * msg[j][(size_t) t * kEmbd + i] * ib;
                }
                const size_t k = (size_t) t * kRank + o;
                peak = std::max(peak, std::fabs(r));
                diff = std::max(diff, (double) std::fabs(a[k] - b[k]));
                e0 = std::max(e0, std::fabs(a[k] - r));
                e1 = std::max(e1, std::fabs(b[k] - r));
            }
        }
    }
    const bool bad = !(e1 <= std::max(4.0 * e0, 1e-6)) || !(diff <= 1e-4 * peak);
    std::printf("%d controller(s), %d row(s), %s: |fast-default| %.2e (peak %.3f)  float64: default %.2e fast %.2e  %s\n",
                count, rows, bf16 ? "BF16" : "F32", diff, peak, e0, e1, bad ? "FAIL" : "ok");
    if (bad) ++failures;
    if (bench) {
        float t[2];
        for (int f = 0; f < 2; ++f) {
            cudaEvent_t a, b;
            ck(cudaEventCreate(&a), "event");
            ck(cudaEventCreate(&b), "event");
            for (int i = 0; i < 20; ++i) launch(f == 1, dy0);
            ck(cudaEventRecord(a, st), "event");
            for (int i = 0; i < 500; ++i) launch(f == 1, dy0);
            ck(cudaEventRecord(b, st), "event");
            ck(cudaEventSynchronize(b), "event");
            float ms;
            ck(cudaEventElapsedTime(&ms, a, b), "event");
            t[f] = 1000.f * ms / 500;
            ck(cudaEventDestroy(a), "event");
            ck(cudaEventDestroy(b), "event");
        }
        std::printf("bench %d controller(s), %d row(s), %s: default %.1f us, fast %.1f us\n", count, rows,
                    bf16 ? "BF16" : "F32", t[0], t[1]);
    }
    for (int j = 0; j < count; ++j)
        for (void* p : {(void*) dc[j], (void*) dm[j], (void*) dw[j], (void*) dy0[j], (void*) dy1[j]}) ck(cudaFree(p), "free");
    ck(cudaStreamDestroy(st), "stream");
}

}  // namespace

int main(int argc, char** argv) {
#if defined(__HIPCC__)
    std::printf("STRATA_INK_FAST_CTRL is CUDA only: skipped\n");
    return 77;
#endif
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd == 0) {
        std::printf("no CUDA card: skipped\n");
        return 77;
    }
    cudaDeviceProp p{};
    ck(cudaGetDeviceProperties(&p, 0), "props");
    std::printf("%s (sm_%d%d)\n", p.name, p.major, p.minor);
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    for (bool bf16 : {false, true})
        for (int count : {1, 3})
            for (int rows : {1, 4}) run_case(count, rows, bf16, bench && rows == 1);
    std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
