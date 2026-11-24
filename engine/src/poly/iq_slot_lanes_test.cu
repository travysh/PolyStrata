// src/poly/iq_slot_lanes_test.cu - the cached experts' IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S and IQ1_M slots launches with
// MmvqSlots::lanes (Inkling 2, CUDA: the codebook staged once for 32 rows) against the same launches without it
// (mmvq_slots_kernel) and against one native_mmvq() a matrix, to the bit: random blocks, 1 to 32 matrices a launch,
// Inkling 2 Mini's gate|up shape (2304 -> 2304, UD-Q2_K_XL's IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S layers) and shapes that
// leave a block's last rows short, through the host's list (MmvqSlots) and the card's picks (MmvqIds, up to 64).
// Each case also prints an FNV-1a hash of its outputs, the same on every card that gives the same bits.
// --bench also times the 8-expert gate|up launches with and without lanes.
//
// usage: poly_iq_slot_lanes_test [--bench]   (exit 0: all agree; 77: no card, or STRATA_IQ_SLOT_LANES=0)
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
std::mt19937 rng(2468);

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

std::vector<float> down(const float* a, size_t n) {
    std::vector<float> x(n);
    ck(cudaDeviceSynchronize(), "sync");
    ck(cudaMemcpy(x.data(), a, n * 4, cudaMemcpyDeviceToHost), "download");
    return x;
}

uint64_t fnv(const std::vector<float>& v) {
    uint64_t h = 0xcbf29ce484222325ull;
    const auto* p = (const uint8_t*) v.data();
    for (size_t i = 0; i < v.size() * 4; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

void same_bits(const std::string& what, const std::vector<float>& x, const std::vector<float>& y) {
    size_t diff = 0, finite = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        diff += std::memcmp(&x[i], &y[i], 4) != 0;
        finite += std::isfinite(y[i]);
    }
    const bool bad = diff || finite < x.size() / 2;   // mostly finite, or the blocks' scales were not sane
    std::printf("%-56s %zu of %zu differ, %zu finite  %s\n", what.c_str(), diff, x.size(), finite, bad ? "FAIL" : "ok");
    if (bad) ++failures;
}

struct Fmt {
    int type;
    const char* name;
};
const Fmt kFmts[] = {{19, "IQ1_S"}, {16, "IQ2_XXS"}, {17, "IQ2_XS"}, {22, "IQ2_S"}, {29, "IQ1_M"}};

// random bytes with a sane fp16 scale in each 256-value block (at its start; IQ1_M: the top nibbles of its last
// four 16-bit scale words)
void* random_weights(int type, int n_in, int n_out) {
    const size_t blk = k::native_mmvq_weight_bytes(type, 256, 1);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(1e-3f, 1e-2f);
    std::vector<uint8_t> h(k::native_mmvq_weight_bytes(type, n_in, n_out));
    for (uint8_t& c : h) c = (uint8_t) byte(rng);
    for (size_t b = 0; b < h.size() / blk; ++b) {
        uint8_t* p = h.data() + b * blk;
        const __half d = __float2half(scale(rng));
        if (type == 29) {
            uint16_t u;
            std::memcpy(&u, &d, 2);
            uint16_t* sc = (uint16_t*) (p + blk - 8);
            for (int i = 0; i < 4; ++i) sc[i] = (uint16_t) ((sc[i] & 0x0fff) | (((u >> (4 * i)) & 0xf) << 12));
        } else {
            std::memcpy(p, &d, 2);
        }
    }
    return up(h);
}

void* random_q8_1(int n_in, int rows, cudaStream_t st) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> h(size_t(n_in) * rows);
    for (float& v : h) v = nd(rng);
    float* x = up(h);
    void* q;
    ck(cudaMalloc(&q, k::native_q8_1_bytes(n_in) * rows), "malloc");   // (native_q8_1_bytes takes at most 8 columns)
    k::native_quantize_q8_1(x, q, n_in, rows, st);
    ck(cudaStreamSynchronize(st), "quantize");
    cudaFree(x);
    return q;
}

// `count` matrices a slots launch (at most 32); the picks launch takes `picks` matrices (count's, repeated) with
// x_every tokens' columns, as Engine::moe_card() gives it n_used experts a token
void test(const Fmt& f, int n_in, int n_out, int count, int picks = 0, int x_every = 1) {
    if (picks == 0) picks = count;
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    std::vector<void*> w((size_t) count);
    for (void*& p : w) p = random_weights(f.type, n_in, n_out);
    const int cols = std::max(count, (picks + x_every - 1) / x_every);
    void* xq = random_q8_1(n_in, cols, st);
    const size_t xr = k::native_q8_1_bytes(n_in), total = size_t(n_out) * count, ptotal = size_t(n_out) * picks;
    float *ya, *yb, *yd, *yp, *yq;
    for (float** p : {&ya, &yb, &yd}) ck(cudaMalloc(p, total * sizeof(float)), "malloc");
    for (float** p : {&yp, &yq}) ck(cudaMalloc(p, ptotal * sizeof(float)), "malloc");
    for (float* p : {ya, yb, yd, yp, yq}) ck(cudaMemset(p, 0x7f, (p == yp || p == yq ? ptotal : total) * 4), "memset");
    k::MmvqSlots slots{}, plain{};
    slots.lanes = true;
    for (int i = 0; i < count; ++i) {
        const void* x = (const uint8_t*) xq + i * xr;
        k::native_mmvq(f.type, w[(size_t) i], x, ya + size_t(i) * n_out, n_in, n_out, 1, st);
        slots.w[i] = plain.w[i] = w[(size_t) i], slots.x[i] = plain.x[i] = x;
        slots.y[i] = yb + size_t(i) * n_out, plain.y[i] = yd + size_t(i) * n_out;
    }
    // the card's picks: pick j is table[ids[j]] (the table backwards, picks past count wrap) times column j / x_every
    std::vector<const void*> table(w.rbegin(), w.rend());
    std::vector<int32_t> ids((size_t) picks);
    for (int j = 0; j < picks; ++j) ids[(size_t) j] = count - 1 - j % count;
    void* dtable;
    ck(cudaMalloc(&dtable, table.size() * sizeof(void*)), "malloc");
    ck(cudaMemcpy(dtable, table.data(), table.size() * sizeof(void*), cudaMemcpyHostToDevice), "table");
    int32_t* dids = up(ids);
    const k::MmvqIds lane_ids{(const void* const*) dtable, dids, 0, xq, xr, x_every, yp, size_t(n_out), true};
    const k::MmvqIds plain_ids{(const void* const*) dtable, dids, 0, xq, xr, x_every, yq, size_t(n_out), false};
    const std::string what = std::string(f.name) + " " + std::to_string(n_in) + " -> " + std::to_string(n_out) +
                             " x" + std::to_string(count);
    if (!k::native_mmvq_slots(f.type, slots, count, n_in, n_out, st) ||
        !k::native_mmvq_slots(f.type, plain, count, n_in, n_out, st) ||
        !k::native_mmvq_ids(f.type, lane_ids, picks, n_in, n_out, st) ||
        !k::native_mmvq_ids(f.type, plain_ids, picks, n_in, n_out, st)) {
        std::printf("%-56s not taken  FAIL\n", what.c_str());
        ++failures;
    } else {
        const std::vector<float> a = down(ya, total), b = down(yb, total), d = down(yd, total);
        const std::vector<float> p = down(yp, ptotal), q = down(yq, ptotal);
        same_bits(what + ", lanes vs native_mmvq", a, b);
        same_bits(what + ", lanes vs without lanes", d, b);
        same_bits(what + ", picks x" + std::to_string(picks) + "/" + std::to_string(x_every) + ", lanes vs without",
                  q, p);
        std::printf("hash %-51s slots %016llx picks %016llx\n", what.c_str(), (unsigned long long) fnv(b),
                    (unsigned long long) fnv(p));
    }
    for (void* p : w) cudaFree(p);
    cudaFree(xq), cudaFree(dtable), cudaFree(dids);
    for (float* p : {ya, yb, yd, yp, yq}) cudaFree(p);
    ck(cudaStreamDestroy(st), "stream");
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

// 8 experts' gate|up (2304 -> 2304) a launch, as Engine::moe() launches them, with and without lanes; more sets of
// experts than L2 holds, taken in turn
void bench() {
    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    const int n_in = 2304, n_out = 2304;
    for (const Fmt& f : kFmts) {
        const size_t one = k::native_mmvq_weight_bytes(f.type, n_in, n_out) * 8;
        const int sets = (int) std::max<size_t>(2, (size_t(64) << 20) / one + 1);
        std::vector<void*> w((size_t) sets * 8);
        for (void*& p : w) p = random_weights(f.type, n_in, n_out);
        void* xq = random_q8_1(n_in, 8, st);
        float* y;
        ck(cudaMalloc(&y, size_t(8) * n_out * 4), "malloc");
        float us[2];
        for (int lanes = 0; lanes < 2; ++lanes) {
            std::vector<k::MmvqSlots> sl((size_t) sets);
            for (int s = 0; s < sets; ++s) {
                sl[(size_t) s].lanes = lanes == 1;
                for (int i = 0; i < 8; ++i)
                    sl[(size_t) s].w[i] = w[(size_t) s * 8 + i],
                    sl[(size_t) s].x[i] = (const uint8_t*) xq + i * k::native_q8_1_bytes(n_in),
                    sl[(size_t) s].y[i] = y + size_t(i) * n_out;
            }
            int r = 0;
            us[lanes] = time_us(st, [&] { k::native_mmvq_slots(f.type, sl[(size_t) (r++ % sets)], 8, n_in, n_out, st); });
        }
        std::printf("bench %-8s 2304 -> 2304 x8: without lanes %7.1f us, with %7.1f us (us, lower is better)\n", f.name,
                    us[0], us[1]);
        for (void* p : w) cudaFree(p);
        cudaFree(xq), cudaFree(y);
    }
    cudaStreamDestroy(st);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // the lines so far survive an abort
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd == 0) {
        std::printf("no CUDA card: skipped\n");
        return 77;
    }
    // these switches keep mmvq_slots_kernel for every launch, so each case would compare it with itself
    for (const char* name : {"STRATA_IQ_SLOT_LANES", "STRATA_IQ_STAGE_GRID"})
        if (const char* v = std::getenv(name); v && v[0] == '0') {
            std::printf("%s=0 is set, so the lane kernel is off: skipped\n", name);
            return 77;
        }
    if (const char* v = std::getenv("STRATA_OLD_IQ_MMVQ"); v && v[0] && v[0] != '0') {
        std::printf("STRATA_OLD_IQ_MMVQ is set, so the lane kernel is off: skipped\n");
        return 77;
    }
    cudaDeviceProp p{};
    ck(cudaGetDeviceProperties(&p, 0), "props");
    std::printf("%s (sm_%d%d)\n", p.name, p.major, p.minor);
    // unset, the lane kernel runs on sm_75 only (iq_kernels.cu: slot_lanes_on()); STRATA_IQ_SLOT_LANES=1 checks it on
    // another card
    if (const char* v = std::getenv("STRATA_IQ_SLOT_LANES"); !(v && *v) && !(p.major == 7 && p.minor == 5)) {
        std::printf("sm_%d%d: the lane kernel is off by default here (STRATA_IQ_SLOT_LANES=1 checks it): skipped\n",
                    p.major, p.minor);
        return 77;
    }
    for (const Fmt& f : kFmts) {
        for (const int count : {1, 2, 3, 5, 7, 8, 32}) test(f, 2304, 2304, count);   // Inkling 2 Mini's gate|up
        test(f, 2304, 2304, 8, 64, 8);   // moe_card: 8 tokens' 8 experts through the picks
        test(f, 2304, 2301, 8);          // a block's last rows short
        test(f, 2304, 1152, 4);          // a mixed layer's gate or up alone
        test(f, 1280, 37, 3);            // fewer rows than one block takes
        test(f, 256, 7, 2);
    }
    if (argc > 1 && std::string(argv[1]) == "--bench") bench();
    std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
