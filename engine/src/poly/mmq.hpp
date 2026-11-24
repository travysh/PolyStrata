// src/poly/mmq.hpp - llama.cpp's quantized matrix products (ggml-cuda mmq.cuh, MIT) for the prompt path: the rows
// of many tokens against matrices that stay in their GGUF type.  Built for the types the model's GGUFs keep
// their matrices in: the K-quants, the I-quants and the older Q4, Q5 and Q8.
//
// Every call enqueues on the given CUDA stream.  Pointers are device pointers.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::poly {

class Mmq {
public:
    Mmq();
    ~Mmq();
    Mmq(const Mmq&) = delete;
    Mmq& operator=(const Mmq&) = delete;

    /// This build multiplies ggml_type, and the card has a tile for a matrix of [w_rows, w_cols] of it.
    static bool fits(int ggml_type, int64_t w_rows, int64_t w_cols);
    /// Bytes of `rows` rows of `cols` values as quantize() writes them.
    static size_t q8_bytes(int64_t rows, int64_t cols);
    /// quantize() keeps the rows' block sums in halves for a matrix of ggml_type (guard_half_sums()).
    static bool half_sums(int ggml_type);
    /// A row's product with a matrix of ggml_type is the same to the bit however the rows are split between
    /// launches.  Not on a card where llama.cpp splits a tile's sum between blocks by the launch's tile count
    /// (stream-k: NVIDIA from Volta, AMD CDNA): there another split adds the partial sums in another order.
    static bool split_exact(int ggml_type, int64_t w_rows);
    /// mul() takes `from` for a matrix of ggml_type with w_cols columns: rows quantized once, read by index.
    static bool reads_by_index(int ggml_type, int64_t w_cols);
    /// quantize() writes the rows for a matrix of ggml_type with w_cols columns as gemm_iq_i8_quantize does
    /// (strata/kernels/iq_kernels.hpp), so that a kernel of the caller's may write them in its place.
    static bool int8(int ggml_type, int64_t w_cols);

    /// The rows of a product with a matrix of ggml_type: row i of xq is row ids[i] of x, or row i when ids is null.
    /// x has ld floats a row, 16-byte aligned.  The rows are interleaved in xq, so a product takes the xq of one
    /// call whole.
    static void quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld,
                         int64_t rows, void* stream);

    /// ids values past a matrix's last row that mul() may read: llama.cpp's tiles load the ids of a whole tile of up
    /// to 128 rows, also where the matrix has fewer.  They are read, never used.
    static constexpr int64_t kIdsTail = 128;

    /// One launch over n matrices of [w_rows, w_cols], `stride` bytes apart from w: matrix e times the rows
    /// [bounds[e], bounds[e + 1]) of xq, which quantize() wrote with total_rows rows.  Row i's product goes to row
    /// ids[i] of dst, ld_dst floats a row; ids has kIdsTail readable values behind the rows of the last matrix.
    /// bounds (n + 1 values) and ids are on the device.  max_rows is the most rows one of the matrices has.  With
    /// one matrix, bounds and ids may both be null: it takes every row, and row i goes to row i.  With `from`
    /// (reads_by_index()), row i of the product is row from[i] of xq.  Throws std::runtime_error.
    void mul(const void* w, int ggml_type, int64_t w_rows, int64_t w_cols, size_t stride, int n, const void* xq,
             const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
             int64_t ld_dst, void* stream, const int32_t* from = nullptr);

    /// mul() over n matrices that need not lie `stride` apart: matrix e at w + at[e] bytes (n values on the device,
    /// multiples of 16), for a matrix of ggml_type with w_cols columns where int8() is true.  Each row's product is
    /// mul()'s to the bit: it does not depend on which matrices share the launch.  Throws std::runtime_error.
    void mul_at(const void* w, const int64_t* at, int ggml_type, int64_t w_rows, int64_t w_cols, int n, const void* xq,
                const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
                int64_t ld_dst, void* stream, const int32_t* from = nullptr);

    /// What a mul() with a matrix of ggml_type, [w_rows, w_cols], of up to `rows` rows takes on the card beside its
    /// arguments: floats for its products when they go to dst by ids or to rows not 16-byte aligned (ld_dst), and
    /// halves for the matrix on the BLAS path, which also takes the BLAS library.
    struct Work {
        size_t floats = 0, halves = 0;
    };
    static Work work(int ggml_type, int64_t w_rows, int64_t w_cols, int64_t rows, bool by_ids, int64_t ld_dst);
    /// The largest Work of the caller's products, from its own buffers, and its cuBLAS handle (a cublasHandle_t):
    /// given before the first mul(), so that a product allocates nothing.  Without them mul() allocates its own,
    /// outside what the caller counted.  `blas` stays the caller's to destroy; a product sets it to its stream.
    void give(float* y, size_t floats, void* dq, size_t halves, void* blas);

    /// mul()'s bounds lie in `dev`, a copy of `host` (`bytes` long) that the caller has filled for the call before
    /// it calls mul(): a product that sizes its launches on the host reads them there (mmq_own.cu's BLAS path).
    void mirror(const void* dev, const void* host, size_t bytes) {
        mirror_dev_ = dev;
        mirror_host_ = host;
        mirror_bytes_ = bytes;
    }

private:
    void* ctx_ = nullptr;   ///< llama.cpp's launch context: a small scratch pool
    const void* mirror_dev_ = nullptr;
    const void* mirror_host_ = nullptr;
    size_t mirror_bytes_ = 0;
};

}  // namespace strata::poly
