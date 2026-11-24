// src/poly/mmq_paths_test.cu - Mmq::quantize + Mmq::mul against a double-precision product of ggml's own dequantized
// weights, for the types strata-poly's models keep matrices in: one dense matrix, and four experts of a launch with
// uneven rows, an empty one, and their products scattered to other rows (bounds and ids, the bounds mirrored on the
// host as the engine does).  The path is the card's (int8 tiles, fp16 tiles, or the BLAS path; STRATA_POLY_PROMPT_BLAS=1
// forces the BLAS path), so a run prints that path's error: run it both ways to compare them.  The activations are
// normal values with a few channels 30 times larger, and rows of different sizes, as a transformer's are.  The int8
// path is checked a second time against the same weights times the rows rounded as its quantizer rounds them, where
// only the float rounding of its sums may differ.  Then the same products with the memory Mmq::work() asks for given
// beforehand (Mmq::give(), as Engine::alloc() does): the same numbers, made in those buffers.  Where the path reads
// rows by index (Mmq::reads_by_index()), the experts' product from rows quantized once and read by index is the one
// from the same rows gathered by ids, bit for bit; and for the int8 path the experts stored in another order and found
// by their offsets (Mmq::mul_at()) give mul()'s products, bit for bit.  On an RTX 50 card NVFP4 takes the two-term FP4 products
// (nvfp4_x2.hpp) unless STRATA_PREFILL_NVFP4 names another mode: run it with STRATA_PREFILL_NVFP4=fp16 too.
// IQ1_S and MXFP4 take the int8 tiles only where an engine asks for them (gemm_iq_i8_allow(), as Inkling 2 does), so the test
// asks; where it then takes them, its product is also made on the FP16 kernel (the path it had before) and the two
// are compared.  STRATA_MMQ_TEST_BLOCKS=DIR: a type's weights are the bytes of DIR/<type name>.bin where that file is
// long enough (whole blocks cut from a model's tensor), in place of ggml's quantization of random weights.
// With STRATA_PREFILL_NVFP4=w4a8 NVFP4 takes the int8 tiles where the card has them, and is checked as the int8 types are.
//
// usage: poly_mmq_paths_test [--turing-int8]   (--turing-int8: the int8 path on Turing too, gemm_iq_i8_turing(); exit 0: every product within 2% RMS of the reference, and an int8 product within
// 1e-5 RMS of its own numbers; 77: no card).  Rows of 2,048 values for every type, then rows of 1,152 (Inkling 2's
// experts' downs) for Q8_0, IQ4_NL and NVFP4.  Where a row's product does not depend on its launch
// (Mmq::split_exact()), 100 of the dense rows multiplied as a launch of their own give the same bits.
#include "mmq.hpp"
#include "ggml.h"
#include "strata/kernels/iq_kernels.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using strata::poly::Mmq;

namespace {

bool ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("%s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    if (cudaMalloc(&d, v.size() * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

struct Err { double rms = 0, top = 0; };

// y against ref over the rows listed (each `cols` long, ld apart)
Err compare(const std::vector<float>& y, const std::vector<double>& ref, int64_t rows, int64_t cols, int64_t ld) {
    double se = 0, sr = 0, me = 0, mr = 0;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c) {
            const double e = (double) y[(size_t) (r * ld + c)] - ref[(size_t) (r * cols + c)], v = ref[(size_t) (r * cols + c)];
            se += e * e;
            sr += v * v;
            me = std::max(me, std::fabs(e));
            mr = std::max(mr, std::fabs(v));
        }
    return {std::sqrt(se / std::max(sr, 1e-300)), me / std::max(mr, 1e-300)};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--turing-int8") == 0) strata::kernels::gemm_iq_i8_turing(true);
    int cards = 0;
    if (cudaGetDeviceCount(&cards) != cudaSuccess || cards == 0) {
        std::puts("poly_mmq_paths_test: no card, skipped");
        return 77;
    }
    const char* forced = std::getenv("STRATA_POLY_PROMPT_BLAS");
    std::printf("path: %s\n", forced && forced[0] == '1' ? "BLAS (STRATA_POLY_PROMPT_BLAS=1)" : "the card's");
    // the 32-value types, the K-quants, the i-quants, MXFP4, NVFP4
    const std::vector<int> types_2048 = {2, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 39, 40};
    const std::vector<int> types_1152 = {8, 20, 40};
    const int asked[] = {19, 39};   // the types the int8 tiles take only when asked
    for (int t : asked) strata::kernels::gemm_iq_i8_allow(t, true);
    const char* blocks_dir = std::getenv("STRATA_MMQ_TEST_BLOCKS");
    constexpr int64_t kOut = 512, kTokens = 256, kExperts = 4;
    std::mt19937 rng(99);
    int bad = 0;
    for (const int64_t kIn : {(int64_t) 2048, (int64_t) 1152}) {
        const std::vector<int>& types = kIn == 2048 ? types_2048 : types_1152;
        std::printf("rows of %lld values\n", (long long) kIn);
        std::normal_distribution<float> normal(0.f, 1.f);
        // activations: a scale a row, a few outlier channels
        std::vector<float> x((size_t) (kTokens * kIn));
        std::vector<float> ch((size_t) kIn, 1.f);
        for (int64_t c = 0; c < kIn; c += 97) ch[(size_t) c] = 30.f;
        for (int64_t r = 0; r < kTokens; ++r) {
            const float s = std::exp(normal(rng));
            for (int64_t c = 0; c < kIn; ++c) x[(size_t) (r * kIn + c)] = normal(rng) * s * ch[(size_t) c];
        }
        // the experts' rows: 100, 0, 7 and the rest; row i's product goes to row perm[i]
        const int32_t bounds_h[kExperts + 1] = {0, 100, 100, 107, (int32_t) kTokens};
        std::vector<int32_t> perm((size_t) kTokens), src((size_t) kTokens);
        for (int64_t i = 0; i < kTokens; ++i) perm[(size_t) i] = (int32_t) ((i * 37 + 11) % kTokens);
        for (int64_t i = 0; i < kTokens; ++i) src[(size_t) i] = (int32_t) ((i * 7 + 3) % 61);   // a token's row read by several
        // the activations as the int8 path quantizes them (gemm_iq_i8_quantize: a block of 32 values, d_inv = 127 / max|x|,
        // code = round(x d_inv), scale = 1 / d_inv), in double: its products are sums of these times the weights, so
        // against them it is exact but for the float rounding of its sums
        std::vector<double> xi8(x.size());
        for (size_t b = 0; b < x.size(); b += 32) {
            float amax = 0.f;
            for (size_t j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b + j]));
            const float d_inv = 127.0f / amax, scale = amax == 0.f ? 0.f : 1.0f / d_inv;
            for (size_t j = 0; j < 32; ++j) xi8[b + j] = amax == 0.f ? 0.0 : (double) std::round(x[b + j] * d_inv) * scale;
        }
        float* dx = upload(x);
        int32_t* dperm = upload(perm);
        int32_t* dsrc = upload(src);
        int32_t* dbounds = upload(std::vector<int32_t>(bounds_h, bounds_h + kExperts + 1));
        if (!dx || !dperm || !dsrc || !dbounds) return 1;
        Mmq mmq;
        mmq.mirror(dbounds, bounds_h, sizeof bounds_h);
        cublasHandle_t blas = nullptr;   // the caller's, as the engine's is (Mmq::give())
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) return 1;
        void* xq = nullptr;
        float* dy = nullptr;
        if (!ok(cudaMalloc(&xq, Mmq::q8_bytes(kTokens, kIn)), "cudaMalloc") ||
            !ok(cudaMalloc(&dy, (size_t) (kTokens * kOut) * sizeof(float)), "cudaMalloc"))
            return 1;
        for (int t : types) {
            const auto ty = (ggml_type) t;
            if (!Mmq::fits(t, kOut, kIn)) {
                std::printf("%-8s no product on this card\n", ggml_type_name(ty));
                continue;
            }
            ggml_quantize_init(ty);
            const size_t mat = ggml_row_size(ty, kIn) * kOut;
            std::vector<float> wf((size_t) (kExperts * kOut * kIn)), imp((size_t) kIn);
            for (auto& v : wf) v = normal(rng) * 0.02f;
            for (auto& v : imp) v = 0.5f + std::fabs(normal(rng));
            std::vector<uint8_t> wq(mat * kExperts);
            ggml_quantize_chunk(ty, wf.data(), wq.data(), 0, kExperts * kOut, kIn, imp.data());
            if (blocks_dir) {   // a model's own blocks, where there are enough of them
                std::ifstream f(std::string(blocks_dir) + "/" + ggml_type_name(ty) + ".bin", std::ios::binary);
                std::vector<uint8_t> real(wq.size());
                if (f && f.read((char*) real.data(), (std::streamsize) real.size())) {
                    wq = real;
                    std::printf("%-8s the model's blocks\n", ggml_type_name(ty));
                }
            }
            std::vector<float> wd(wf.size());
            ggml_get_type_traits(ty)->to_float(wq.data(), wd.data(), (int64_t) wd.size());
            uint8_t* dw = upload(wq);
            if (!dw) return 1;
            // row i of a launch times its expert's matrix, in double: the rows as they are, or as the int8 path has them
            auto reference = [&](int64_t row, int e, bool as_int8, double* out) {
                const float* a = &x[(size_t) (row * kIn)];
                const double* a8 = &xi8[(size_t) (row * kIn)];
                for (int64_t o = 0; o < kOut; ++o) {
                    const float* w = &wd[(size_t) ((e * kOut + o) * kIn)];
                    double s = 0;
                    for (int64_t k = 0; k < kIn; ++k) s += (as_int8 ? a8[k] : (double) a[k]) * w[k];
                    out[o] = s;
                }
            };
            const bool int8 = Mmq::int8(t, kIn);
            std::vector<double> ref((size_t) (kTokens * kOut)), ref8(int8 ? ref.size() : 0);
            std::vector<float> y((size_t) (kTokens * kOut));
            // one dense matrix: every row, in place
            Mmq::quantize(dx, nullptr, xq, t, kIn, kIn, kTokens, nullptr);
            mmq.mul(dw, t, kOut, kIn, 0, 1, xq, nullptr, nullptr, kTokens, kTokens, dy, kOut, nullptr);
            if (!ok(cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost), "dense product")) return 1;
            for (int64_t r = 0; r < kTokens; ++r) {
                reference(r, 0, false, &ref[(size_t) (r * kOut)]);
                if (int8) reference(r, 0, true, &ref8[(size_t) (r * kOut)]);
            }
            const Err dense = compare(y, ref, kTokens, kOut, kOut);
            // rows 37 to 136 as a launch of their own
            bool apart = true;
            if (Mmq::split_exact(t, kOut)) {
                std::vector<float> y_part((size_t) (100 * kOut));
                Mmq::quantize(dx + 37 * kIn, nullptr, xq, t, kIn, kIn, 100, nullptr);
                mmq.mul(dw, t, kOut, kIn, 0, 1, xq, nullptr, nullptr, 100, 100, dy, kOut, nullptr);
                if (!ok(cudaMemcpy(y_part.data(), dy, y_part.size() * sizeof(float), cudaMemcpyDeviceToHost), "rows apart")) return 1;
                apart = std::memcmp(y_part.data(), &y[(size_t) (37 * kOut)], y_part.size() * sizeof(float)) == 0;
                std::printf("%-8s rows in a launch of their own: %s\n", ggml_type_name(ty), apart ? "same products" : "OTHER PRODUCTS");
            }
            const Err dense8 = int8 ? compare(y, ref8, kTokens, kOut, kOut) : Err{};
            // four experts: rows [bounds[e], bounds[e + 1]) of xq (row i of xq is row i of x), product of row i to row perm[i]
            cudaMemset(dy, 0, y.size() * sizeof(float));
            Mmq::quantize(dx, nullptr, xq, t, kIn, kIn, kTokens, nullptr);
            mmq.mul(dw, t, kOut, kIn, mat, kExperts, xq, dbounds, dperm, kTokens, 149, dy, kOut, nullptr);
            if (!ok(cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost), "experts' product")) return 1;
            std::vector<float> back((size_t) (kTokens * kOut));
            for (int e = 0; e < kExperts; ++e)
                for (int64_t r = bounds_h[e]; r < bounds_h[e + 1]; ++r) {
                    reference(r, e, false, &ref[(size_t) (r * kOut)]);
                    if (int8) reference(r, e, true, &ref8[(size_t) (r * kOut)]);
                    std::copy_n(&y[(size_t) perm[(size_t) r] * kOut], kOut, &back[(size_t) (r * kOut)]);
                }
            bool at_same = true;
            if (int8) {   // the same experts stored in the other order and found by their offsets (mul_at()): mul()'s bits
                std::vector<uint8_t> rev(wq.size());
                std::vector<int64_t> at((size_t) kExperts);
                for (int e = 0; e < kExperts; ++e) {
                    at[(size_t) e] = (int64_t) ((kExperts - 1 - e) * mat);
                    std::memcpy(&rev[(size_t) at[(size_t) e]], &wq[(size_t) e * mat], mat);
                }
                uint8_t* drev = upload(rev);
                int64_t* dat = upload(at);
                if (!drev || !dat) return 1;
                std::vector<float> y_at(y.size());
                cudaMemset(dy, 0, y.size() * sizeof(float));
                mmq.mul_at(drev, dat, t, kOut, kIn, kExperts, xq, dbounds, dperm, kTokens, 149, dy, kOut, nullptr);
                if (!ok(cudaMemcpy(y_at.data(), dy, y_at.size() * sizeof(float), cudaMemcpyDeviceToHost), "offsets' product"))
                    return 1;
                at_same = std::memcmp(y_at.data(), y.data(), y.size() * sizeof(float)) == 0;
                std::printf("%-8s matrices by offset: %s\n", ggml_type_name(ty), at_same ? "same products" : "OTHER PRODUCTS");
                cudaFree(drev);
                cudaFree(dat);
            }
            const Err experts = compare(back, ref, kTokens, kOut, kOut);
            const Err experts8 = int8 ? compare(back, ref8, kTokens, kOut, kOut) : Err{};
            // The int8 path against its own numbers: a weight decoded wrong anywhere shows as an error of 1e-3 or more of
            // the products' size; the float rounding of the sums is about 1e-7
            bool fine = at_same && dense.rms < 0.02 && experts.rms < 0.02 && dense8.rms < 1e-5 && experts8.rms < 1e-5 && apart;
            std::printf("%-8s dense: RMS error %.2e, largest %.2e   experts: RMS error %.2e, largest %.2e", ggml_type_name(ty),
                        dense.rms, dense.top, experts.rms, experts.top);
            if (int8)
                std::printf("   int8 numbers: RMS error %.2e / %.2e, largest %.2e / %.2e", dense8.rms, experts8.rms, dense8.top,
                            experts8.top);
            std::printf("  %s\n", fine ? "ok" : "WRONG");

            if (int8 && std::find(std::begin(asked), std::end(asked), t) != std::end(asked)) {
                // the same products on the FP16 kernel, as before the type was asked for: both against the reference, and
                // the int8 products against them
                strata::kernels::gemm_iq_i8_allow(t, false);
                std::vector<float> yh(y.size());
                if (Mmq::int8(t, kIn) || !Mmq::fits(t, kOut, kIn)) {
                    std::printf("%-8s FP16 path: not taken\n", ggml_type_name(ty));
                    fine = false;
                } else {
                    Mmq::quantize(dx, nullptr, xq, t, kIn, kIn, kTokens, nullptr);
                    mmq.mul(dw, t, kOut, kIn, mat, kExperts, xq, dbounds, dperm, kTokens, 149, dy, kOut, nullptr);
                    if (!ok(cudaMemcpy(yh.data(), dy, yh.size() * sizeof(float), cudaMemcpyDeviceToHost), "FP16 product")) return 1;
                    std::vector<float> backh((size_t) (kTokens * kOut));
                    std::vector<double> b8((size_t) (kTokens * kOut));
                    for (int64_t r = 0; r < kTokens; ++r) {
                        std::copy_n(&yh[(size_t) perm[(size_t) r] * kOut], kOut, &backh[(size_t) (r * kOut)]);
                        for (int64_t o = 0; o < kOut; ++o) b8[(size_t) (r * kOut + o)] = back[(size_t) (r * kOut + o)];
                    }
                    const Err half = compare(backh, ref, kTokens, kOut, kOut), apart = compare(backh, b8, kTokens, kOut, kOut);
                    const bool near = half.rms < 0.02 && apart.rms < 0.02;
                    std::printf("%-8s FP16 path: RMS error %.2e, largest %.2e   int8 against it: RMS %.2e, largest %.2e  %s\n",
                                ggml_type_name(ty), half.rms, half.top, apart.rms, apart.top, near ? "ok" : "WRONG");
                    fine = fine && near;
                }
                strata::kernels::gemm_iq_i8_allow(t, true);
            }

            if (Mmq::reads_by_index(t, kIn)) {
                // routed row i is token src[i]: its rows gathered and quantized a row each, against the tokens' rows
                // quantized once and read by index
                std::vector<float> y1(y.size()), y2(y.size());
                cudaMemset(dy, 0, y.size() * sizeof(float));
                Mmq::quantize(dx, dsrc, xq, t, kIn, kIn, kTokens, nullptr);
                mmq.mul(dw, t, kOut, kIn, mat, kExperts, xq, dbounds, dperm, kTokens, 149, dy, kOut, nullptr);
                if (!ok(cudaMemcpy(y1.data(), dy, y1.size() * sizeof(float), cudaMemcpyDeviceToHost), "gathered rows")) return 1;
                cudaMemset(dy, 0, y.size() * sizeof(float));
                Mmq::quantize(dx, nullptr, xq, t, kIn, kIn, 61, nullptr);
                mmq.mul(dw, t, kOut, kIn, mat, kExperts, xq, dbounds, dperm, 61, 149, dy, kOut, nullptr, dsrc);
                if (!ok(cudaMemcpy(y2.data(), dy, y2.size() * sizeof(float), cudaMemcpyDeviceToHost), "rows by index")) return 1;
                const bool same = std::memcmp(y1.data(), y2.data(), y1.size() * sizeof(float)) == 0;
                std::printf("%-8s rows read by index: %s\n", ggml_type_name(ty), same ? "same products" : "OTHER PRODUCTS");
                fine = fine && same;
            }

            // Given the memory work() asks for, the products come out the same, and in it: buffers full of 0xff that a
            // product which made its own memory would leave as they were.  (None is asked where the int8 tiles take it.)
            const Mmq::Work one = Mmq::work(t, kOut, kIn, kTokens, false, kOut);
            const Mmq::Work four = Mmq::work(t, kOut, kIn, kTokens, true, kOut);
            const size_t floats = std::max(one.floats, four.floats), halves = std::max(one.halves, four.halves);
            float* gy = nullptr;
            void* gdq = nullptr;
            if ((floats && !ok(cudaMalloc(&gy, floats * sizeof(float)), "cudaMalloc")) ||
                (halves && !ok(cudaMalloc(&gdq, halves * 2), "cudaMalloc")))
                return 1;
            if (gy) cudaMemset(gy, 0xff, floats * sizeof(float));
            if (gdq) cudaMemset(gdq, 0xff, halves * 2);
            {
                Mmq given;
                given.mirror(dbounds, bounds_h, sizeof bounds_h);
                given.give(gy, floats, gdq, halves, blas);
                std::vector<float> y2(y.size()), y3(y.size());
                Mmq::quantize(dx, nullptr, xq, t, kIn, kIn, kTokens, nullptr);
                given.mul(dw, t, kOut, kIn, 0, 1, xq, nullptr, nullptr, kTokens, kTokens, dy, kOut, nullptr);
                if (!ok(cudaMemcpy(y2.data(), dy, y2.size() * sizeof(float), cudaMemcpyDeviceToHost), "dense product")) return 1;
                cudaMemset(dy, 0, y.size() * sizeof(float));
                given.mul(dw, t, kOut, kIn, mat, kExperts, xq, dbounds, dperm, kTokens, 149, dy, kOut, nullptr);
                if (!ok(cudaMemcpy(y3.data(), dy, y3.size() * sizeof(float), cudaMemcpyDeviceToHost), "experts' product"))
                    return 1;
                // y has the experts' product of the instance that made its own memory; its dense one again, for this
                std::vector<float> dense_y((size_t) (kTokens * kOut));
                mmq.mul(dw, t, kOut, kIn, 0, 1, xq, nullptr, nullptr, kTokens, kTokens, dy, kOut, nullptr);
                if (!ok(cudaMemcpy(dense_y.data(), dy, dense_y.size() * sizeof(float), cudaMemcpyDeviceToHost), "dense product"))
                    return 1;
                uint32_t y0 = 0xffffffffu;
                uint16_t q0 = 0xffff;
                if (gy) cudaMemcpy(&y0, gy, sizeof y0, cudaMemcpyDeviceToHost);
                if (gdq) cudaMemcpy(&q0, gdq, sizeof q0, cudaMemcpyDeviceToHost);
                const bool same = std::equal(y2.begin(), y2.end(), dense_y.begin()) && std::equal(y3.begin(), y3.end(), y.begin());
                // (the BLAS path takes both for the experts: a work() that asks for none would pass the rest)
                const bool blas_path = forced && forced[0] == '1';
                const bool used = (!gy || y0 != 0xffffffffu) && (!gdq || q0 != 0xffff) && (!blas_path || (gy && gdq));
                std::printf("%-8s given %zu KiB of floats, %zu KiB of halves: %s, %s\n", ggml_type_name(ty), floats * 4 >> 10,
                            halves * 2 >> 10, same ? "same products" : "OTHER PRODUCTS", used ? "made in them" : "NOT IN THEM");
                fine = fine && same && used;
            }
            cudaFree(gy);
            cudaFree(gdq);
            bad += !fine;
            cudaFree(dw);
        }
        cublasDestroy(blas);
        cudaFree(xq);
        cudaFree(dy);
        cudaFree(dx);
        cudaFree(dperm);
        cudaFree(dsrc);
        cudaFree(dbounds);
    }
    ggml_quantize_free();
    std::printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
