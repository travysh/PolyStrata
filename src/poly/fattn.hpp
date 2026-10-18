// src/poly/fattn.hpp - a prompt batch's attention with llama.cpp's flash-attention MMA kernel (fattn_mma.cuh, MIT)
// over a q8_0 cache: heads of 256 values, 8 heads a group (Qwen3.6 / Cyber-Tiel's full-attention layers).  The keys
// go in blocks: each block's K and V are dequantized to fp16 in the scratch, the kernel attends to them with the
// causal mask computed from the positions, and the blocks' results are merged by their softmax max and sum.
// Opt-in: STRATA_FATTN=1 (Qwen35::attn).
//
// Every call enqueues on the given CUDA stream.  Pointers are device pointers.
#pragma once

#include "strata/poly/kernels.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::poly {

/// This build has the kernel for heads of `dim` values, n_head / kv_heads heads a group, on this card.
bool fattn_q8_fits(int dim, int n_head, int kv_heads);

/// Causal attention of the n tokens from position pos to the cache rows 0 to pos + n - 1, as attend_batch()
/// (no window, no ring): q and out n_head * dim floats a token.  scratch (scratch_floats floats) holds a block's
/// results and its K and V; false, with nothing enqueued, when it cannot hold a block of n + 32 rows.
bool fattn_q8(const float* q, Kvq8 kq8, Kvq8 vq8, int64_t pos, int n, float* out, int dim, int n_head, int kv_heads,
              float scale, float* scratch, size_t scratch_floats, void* stream);

}  // namespace strata::poly
