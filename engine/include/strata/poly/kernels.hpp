// include/strata/poly/kernels.hpp - the float kernels of the forward passes.
//
// Every call enqueues on the given CUDA stream and returns; nothing here but blas_create allocates or waits.
// Pointers are device pointers.  The math follows llama.cpp's graphs (src/models/glm5-next.cpp, qwen35moe.cpp and
// gemma4.cpp) and the ggml ops they build, which are the reference the engine is checked against.
//
// A call works on `rows` tokens at consecutive positions.  A buffer of per-token values holds them row after row,
// each row as long as the one-token case describes.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::poly {

/// y[o] = sum_i w[o * n_in + i] * x[(o / rows_per_x) * n_in + i], o in [0, n_out).  With rows_per_x == n_out it is
/// a plain matrix-vector product; a smaller value gives each group of rows its own input (the per-head MLA
/// projections), so a token's x is n_in * n_out / rows_per_x long.
void gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows, void* stream);
void gemv_bf16(const uint16_t* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows,
               void* stream);
/// gemv_f32() (type 0) or gemv_bf16() (type 1) of two matrices on one input, each a plain product (rows_per_x ==
/// its n_out), in one launch with the same bits.  CUDA and HIP only.
void gemv_pair(int type, const void* w1, const void* w2, const float* x, float* y1, float* y2, int n_in, int n_out1,
               int n_out2, int rows, void* stream);
/// gemv_f32() of 1 to 6 matrices of one width (n_in), each a plain product on an input of its own, in one launch
/// with the same bits.  CUDA and HIP only.
void gemv_set(const float* const* w, const float* const* x, float* const* y, const int* n_out, int count, int n_in,
              int rows, void* stream);
/// The same for the many rows of a prompt batch, as a cuBLAS product on the stream of `blas` (blas_create), one
/// for each group of rows_per_x outputs.  It sums in another order than gemv_f32, so a row does not come out
/// to the last bit as it does there.  Throws std::runtime_error.
void gemm_f32(void* blas, const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows);
/// n BF16 values as floats, for gemm_f32.
void bf16_to_f32(const uint16_t* w, float* out, size_t n, void* stream);
/// Before a Q8_1 quantization whose blocks keep their sums in halves (llama.cpp's MMQ for Q4_K, Q5_K, Q2_K, Q4_1,
/// Q5_1 and IQ1_S rows; Mmq::half_sums()): each of the rows of x (ld floats apart, width values) whose sums or
/// values would leave a half's range is divided by a power of two, and factor[r] holds it (1 for the others).  A
/// SwiGLU's output can be that large: Flint's reaches 45,000, and a block of 32 summed past 65,504 is infinite.
/// The products of the divided rows are the true ones divided by the same power, to the bit: scale_rows() puts
/// them back.
void guard_half_sums(float* x, int64_t ld, int width, int rows, float* factor, void* stream);
/// The rows of y (ld floats apart, row r at row place[r] when place is given) multiplied by factor[r], or divided
/// by it; a row whose factor is 1 is not touched.
void scale_rows(float* y, int64_t ld, int width, int rows, const int32_t* place, const float* factor, bool divide,
                void* stream);

/// RMS norm over each of `rows` groups of n values: y = x / sqrt(mean(x^2) + eps) * w.  `w` (n values, shared by the
/// rows) may be null.  A row of y starts y_stride values after the one before it (0: n), so two calls can fill
/// the two halves of rows twice as long.
void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* stream, int y_stride = 0);
/// rms_norm() of 1 to 3 inputs of `rows` rows each, every one with its weight (null: none) and output, in one
/// launch with the same bits.
void rms_norm_many(const float* const* x, const float* const* w, float* const* y, int count, int n, int rows, float eps,
                   void* stream);
/// The threads of a block that sums over n values (rms_norm()'s among them).
int sum_threads(int n);
/// y = (x - mean) / sqrt(var + eps) * w + b.
void layer_norm(const float* x, const float* w, const float* b, float* y, int n, int rows, float eps, void* stream);

/// The hyper-connection read.  r is a token's hc * n_embd residual streams.  Writes cur (n_embd) and hcw: post[hc]
/// then comb[hc * hc] (comb[dst + hc * src]), which hc_post reads.  `mix` is hc*(2+hc)+1 floats of scratch a token.
void hc_pre(const float* r, const float* fn, const float* scale, const float* base, float* mix, float* hcw,
            float* cur, int n_embd, int rows, float norm_eps, float hc_eps, int sinkhorn_iters, void* stream);
/// The hyper-connection write, in place on r: r[dst] = x * post[dst] + sum_src r[src] * comb[dst, src].
void hc_post(const float* x, const float* hcw, float* r, int n_embd, int rows, void* stream);
/// y = mean over the hc streams, token by token.
void hc_mean(const float* r, float* y, int n_embd, int rows, void* stream);
/// DeepSeek V4's output read of the streams: y = sum_h r[h] * (sigmoid(fn[h] . r * rs * scale + base[h]) + hc_eps),
/// rs the flat RMS norm's scale.  fn: hc rows of hc * n_embd.  mix: hc + 1 floats a token of scratch.
void hc_head(const float* r, const float* fn, const float* scale, const float* base, float* mix, float* y,
             int n_embd, int rows, float norm_eps, float hc_eps, void* stream);

/// The three causal convolutions of a KDA layer and their SiLU.  proj: the q projections of every row, then the k,
/// then the v (3 * rows * d_inner); out has the same layout.  state: 3 * d_inner * (d_conv - 1) past inputs, oldest
/// first per channel, advanced here by `rows` tokens.  w*: (d_conv, d_inner).
/// With snap, the state as it is after each of the first n_snap tokens is also written there: the one after
/// token i starts i * stride floats in.  That is what a window of guessed tokens goes back to (kda_step too).
void kda_conv(const float* proj, float* state, const float* wq, const float* wk, const float* wv, float* out,
              int d_inner, int rows, void* stream, float* snap = nullptr, size_t stride = 0, int n_snap = 0);
/// The delta-rule step of every head and the gated output, token after token.  qkv: the conv outputs.  g1:
/// f_b(f_a(x)) + dt bias (d_inner); it gates the state's decay per key dimension, as in ggml's gated_delta_net op.
/// a: per head.  beta: per head, before the sigmoid.  s: n_head * head_dim * head_dim, updated.  g2: g_b(g_a(x))
/// (d_inner).  out = rms_norm(o) * o_norm * sigmoid(g2), d_inner values.  A call of 32 rows or more without snap
/// takes three launches for the same bits, and leaves its work in qkv and g1.
void kda_step(float* qkv, float* g1, const float* a, const float* beta, float* s, const float* g2,
              const float* o_norm, float* out, int head_dim, int n_head, int rows, float gate_lower_bound,
              float norm_eps, void* stream, float* snap = nullptr, size_t stride = 0, int n_snap = 0);

/// Stores the tokens' MLA latents (n values each) as fp16 from row `pos` of the layer's cache.
void store_f16(const float* x, uint16_t* cache, int n, int64_t pos, int rows, void* stream);
/// The same from the row the card holds at `at` when the launch runs; with `ring`, a row is its number % ring.
void store_f16_at(const float* x, uint16_t* cache, int n, const long long* at, int rows, int ring, void* stream);
/// --kv int8 (and int8k) keeps an attention cache's K and V as 8-bit codes with a delta a block of
/// kQ8_0Values values: that is llama.cpp's Q8_0 rule (delta = the block's largest magnitude / 127, code =
/// roundf(x / delta)) but NOT its block layout.  ggml's block_q8_0 is 34 bytes - a delta then 32 codes - so
/// half its blocks put the codes at an address that is 2 mod 4 and a lane's four codes can only be read
/// conservatively (memcpy into an int).  With the codes of a row back to back they are 4-aligned throughout
/// and a step is one load, and a row of a layer is n_kv heads of head_dim codes, head_dim being a multiple of
/// kQ8_0Values.  A reader carries two pointers instead of one.
inline constexpr int kQ8_0Values = 32;

/// One side of an attention cache in --kv int8's form (store_int8, attend_int8).
struct KvInt8 {
    const int8_t* code = nullptr;     ///< n_kv * head_dim codes a row, kQ8_0Values of them a block, back to back
    const uint16_t* delta = nullptr;  ///< n_kv * head_dim / kQ8_0Values halves a row, one a block
};

/// Turns an int8 cache back into halves, one kQ8_0Values-block a thread (out is the same shapes as the caches:
/// rows * n_kv * head_dim halves).  The fp16 kernels can then run over it - which is how a quantized cache is
/// used at its best, and what llama.cpp does on CUDA: its tensor-core flash attention converts a quantized
/// cache's K and V to f16 first ("other kernels convert these to f16"), and its Metal backend gained a dedicated
/// pass for the same thing.  It costs one O(rows) pass to get the attention back onto the fast path.
void dequant_int8_to_f16(const int8_t* code, const uint16_t* delta, uint16_t* out, int64_t n_blocks, void* stream);

/// A runtime capability probe for the int8 tensor-core path: it runs one m16n8k32 on this card and checks the
/// result against the same product computed on the host, so a kernel can be dispatched on the answer rather than
/// on an assumed architecture.  True on sm_80 and up; a backend whose int8 mma is not there (HIP without
/// mfma_i32_16x16x32_i8, a card before Ampere) answers false and the caller keeps dp4a.
bool probe_mma_layout();

/// Whether attend_int8 can take its tensor-core tier here: the m16n8k32 int8 step needs Ampere (sm_80), where the
/// fp16 tensor-core kernels need only Turing.  False under HIP, before Ampere, and on an older card's PTX.
bool int8_tensor_cores();

/// A study, not part of the engine: the same 16x8x256 product through the tensor cores and through dp4a, in ms.
void probe_mma_speed(double* mma_ms, double* dp4a_ms);

/// Stores the tokens' latents (n_kv * head_dim values a row, every kQ8_0Values of them a block) as codes and their
/// deltas from row `pos` of the layer's cache; `at` and `ring` as in store_f16_at.  Half the bytes a token takes.
void store_int8(const float* x, int8_t* code, uint16_t* delta, int n_kv, int head_dim, int64_t pos, int rows,
                void* stream);
void store_int8_at(const float* x, int8_t* code, uint16_t* delta, int n_kv, int head_dim, int rows,
                   const long long* at, int ring, void* stream);

/// Stores the tokens' rows as ggml Q8_0 (one fp16 scale a 32-value block of int8 codes) from row `pos`.
/// codes: n int8 a row; scales: n/32 fp16 a row.  n must be a multiple of 32 (QK8_0).  rs as Kvq8::rs.
void store_q8(const float* x, int8_t* codes, uint16_t* scales, int n, int64_t pos, int rows, void* stream,
              size_t rs = 0);
/// The same from the row the card holds at `at` when the launch runs; with `ring`, a row is its number % ring.
void store_q8_at(const float* x, int8_t* codes, uint16_t* scales, int n, const long long* at, int rows, int ring,
                 void* stream, size_t rs = 0);
/// store_q8_at() for rows of different sequences: row r into codes[r] and scales[r] at the row at[2 * r] (step()'s
/// pairs of position and turn).  Each row gets the bits a call of its own gives.  CUDA only.
struct RowCaches {
    static constexpr int kMax = 32;
    int8_t* codes[kMax] = {};
    uint16_t* scales[kMax] = {};
    size_t rs = 0;   ///< the codes from a cache's row to its next, as Kvq8::rs; 0: n
};
void store_q8_rows(const float* x, const RowCaches& caches, int n, const long long* at, int rows, void* stream);
/// A step's rows' fp16 caches, row r's own sequence's in rows[r] (attend_seqs() under an fp16 cache).
struct RowHalves {
    static constexpr int kMax = 32;
    uint16_t* rows[kMax] = {};
};

#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;             // scale (fp16 bits)
    uint8_t qs[16];         // 32 4-bit codes: element j in low nibble of qs[j], j + 16 in high nibble
};
#pragma pack(pop)
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

/// Stores the tokens' rows as ggml Q4_0 (one fp16 scale and 16 bytes of nibbles per 32-value block) from row `pos`.
/// blocks: (n / 32) block_q4_0 per row. n must be a multiple of 32.
void store_q4(const float* x, block_q4_0* out, int n, int64_t pos, int rows, void* stream);
void store_q4_at(const float* x, block_q4_0* out, int n, const long long* at, int rows, int ring, void* stream);

/// Dequantizes Q4_0 blocks to Q8_0 codes and scales (exact integer mapping: codes = nibble - 8, scale = scale).
/// With `at`, the destination rows start at the position held on the card when the launch runs (graph replay).
void dequant_q4_to_q8(const block_q4_0* in, int8_t* codes, uint16_t* scales, int n, int rows, void* stream,
                      const long long* at = nullptr);
/// Attention over cached latents for the tokens from position `pos`.  q: n_head * rank absorbed queries a token.
/// The first dense_rows tokens attend to every cache row up to their own.  The others each have a list in sel
/// (sel_stride apart): top_rows cache rows, then the rows of the pool their own position is still filling.
/// out: n_head * rank a token, the attention-weighted latents per head.
/// It takes `chunk` tokens at a time: their keys are copied to `keys` as floats (rank a key), and two cuBLAS
/// products per chunk do the rest, with `score` (n_head a key) between them.  Both buffers hold a chunk of tokens
/// with the most keys a token has: top_rows + kpool - 1, or the context if that is less.  Throws
/// std::runtime_error.
void dsa_attend(void* blas, const float* q, const uint16_t* cache, const int32_t* sel, int sel_stride, int top_rows,
                int kpool, int64_t pos, int rows, int dense_rows, float* keys, float* score, int chunk, float* out,
                int rank, int n_head, float scale, void* stream);
/// A cuBLAS handle that enqueues on `stream`, for dsa_attend and gemm_f32.  Throws std::runtime_error.
void* blas_create(void* stream);
void blas_destroy(void* blas);

/// The indexer's cache: a pool's key once its kpool rows are in, dim floats a pool in `pool`, and the latest rows'
/// keys and gates in `ring`, key (dim) | gate (dim) a row, the row of position p at p % ring_rows.  fp32.
/// Stores the tokens' keys and gates from position `pos`.
void idx_store(const float* key, const float* gate, float* ring, int ring_rows, int dim, int64_t pos, int rows,
               void* stream);
/// For each of the positions from `pos` that ends a pool: the pool's key from its kpool rows.  ape: (dim, kpool).
void idx_pool(const float* ring, int ring_rows, float* pool, const float* ape, int dim, int kpool, int64_t pos,
              int rows, void* stream);
/// For the tokens from position `pos`: score[p] = sum_h relu(q[h] . pool[p]) * w[h] for the pools p that are
/// complete at the token's position.  A token's scores start score_stride apart.  w is the raw indexer_proj
/// output; the 1/sqrt(dim * n_head) factor is applied here.  Throws std::invalid_argument unless dim is 128 and
/// n_head 32 or less.
void idx_score(const float* q, const float* w, const float* pool, float* score, int score_stride, int dim,
               int n_head, int kpool, int64_t pos, int rows, void* stream);
/// idx_score() for a prompt batch: the products of the queries and the pools' keys through cuBLAS, as many tokens
/// at a time as have theirs in the work_floats floats of `work`, n_head for each pool of the call's last token.
/// A score has other last bits than idx_score()'s.  Throws std::runtime_error, and std::invalid_argument when
/// work has no room for a token.
void idx_score_batch(void* blas, const float* q, const float* w, const float* pool, float* score, int score_stride,
                     int dim, int n_head, int kpool, int64_t pos, int rows, float* work, size_t work_floats,
                     void* stream);
/// For each of the tokens from position `pos`, from its scores: the n_top pools with the highest, the lower pool
/// first among equals, as their kpool rows each in that order, then the rows that are in no whole pool yet.  A
/// token's list starts sel_stride apart and takes n_top * kpool + kpool - 1 at most.  Every token has n_top
/// whole pools or more.
void idx_select(const float* score, int score_stride, int32_t* sel, int sel_stride, int n_top, int kpool,
                int64_t pos, int rows, void* stream);

/// out = silu(min(gate, limit)) * clamp(up, -limit, limit).
void swiglu_clamp(const float* gate, const float* up, float* out, int n, float limit, void* stream);
/// The global scales of NVFP4 routed experts converted from NVIDIA's checkpoints (the GGUF's ffn_gate_exps.scale
/// and ffn_up_exps.scale, one an expert), on the card.  Row r of an activation is expert ids[r]'s, or e[r]'s when
/// ids is null: its gate values are multiplied by gate[that expert] and its up values by up[it] first.
struct ExpertScales {
    const float* gate = nullptr;   ///< null: none (1)
    const float* up = nullptr;     ///< null: none (1)
    const int32_t* ids = nullptr;
    int32_t e[32] = {};
};
/// Metal's (CUDA's activations put the scales on themselves): rows that each hold n gate values and then n up values
/// into out, each row's gate and up values times its ExpertScales as above, the same floats as CUDA's activations
/// take; out may be gate_up.  Without scales (null, or neither): a copy, nothing in place.  For a Metal activation's
/// rows ahead of it (situ_rows).
void scale_expert_rows(const float* gate_up, float* out, int n, int rows, const ExpertScales* scales, void* stream);
/// swiglu_clamp for rows that each hold n gate values and then n up values; out has n values a row.
void swiglu_clamp_rows(const float* gate_up, float* out, int n, int rows, float limit, void* stream,
                       const ExpertScales* scales = nullptr);
/// Inkling 2's activation (SiTU) for the same rows: out = beta * tanh(gate / beta) * sigmoid(gate) * lb * tanh(up /
/// lb), each row's gate and up values times its expert's scales first.  Declared here because the experts' shared
/// code calls it (Engine::situ_beta_); the kernel is inkling2.cu's, and Metal has none.
void situ_rows(const float* gate_up, float* out, int n, int rows, float beta, float lb, void* stream, void* q8,
               const ExpertScales* scales = nullptr);
/// swiglu_clamp_rows, its rows written straight into xq as gemm_iq_i8_quantize writes them (n a multiple of 32),
/// the same numbers as the two calls one after the other, without the floats between them.
void swiglu_clamp_i8_rows(const float* gate_up, void* xq, int n, int rows, float limit, void* stream,
                          const ExpertScales* scales = nullptr);
/// out = gelu(gate) * up, with the tanh form of GELU that ggml's CUDA backend computes.
void geglu(const float* gate, const float* up, float* out, int n, void* stream);
/// geglu for rows that each hold n gate values and then n up values; out has n values a row.
void geglu_rows(const float* gate_up, float* out, int n, int rows, void* stream, const ExpertScales* scales = nullptr);
/// w[r] *= scale[ids[r]] for `rows` weights: the down scales of ExpertScales' experts, on their routing weights.
void expert_weights(float* w, const float* scale, const int32_t* ids, int rows, void* stream);
/// The routing of `rows` tokens from their router scores (n_expert a token, times in_scale), as an architecture's
/// route() does it on the host: probabilities by `gate` (0 softmax, 1 sigmoid, 2 sqrt(softplus), 3 sigmoid with the
/// pick on the bare scores), the n_used largest of the probabilities plus `bias` (null: none), the lower expert
/// first among equals; their weights are the probabilities, with `norm` divided by their sum (ggml's floor under
/// it), times escale[expert] (null: none), times w_scale (0: not).  ids and w: n_used a token, on the card.  Gate 3
/// is llama.cpp's SIGMOID_LOGIT_ADD, Kolibri 1's: the weights are sigmoid(scores) and the pick is scores + bias.
/// Throws std::invalid_argument past 512 experts.
void route_rows(const float* logits, int n_expert, int n_used, int rows, int gate, float in_scale, const float* bias,
                const float* escale, bool norm, float w_scale, int32_t* ids, float* w, void* stream);
/// acc[t] += sum over k < n_used of w[t * n_used + k] * x[t * n_used + k], for rows of n values: a token's
/// weighted experts added to its row.
void moe_sum(const float* x, const float* w, float* acc, int n, int n_used, int rows, void* stream);
/// acc = x * a, or acc += x * a when `add`.
void axpy(const float* x, float a, float* acc, int n, bool add, void* stream);
/// y = x for two runs of 4-byte words, n0 and n1 of them, in one launch.  Pinned host memory is the card's to read
/// and write too, and a kernel's copy to or from it has none of the wait that a copy engine's has on Windows,
/// some 25 microseconds between it and the launches beside it.
void copy2(const void* x0, void* y0, int n0, const void* x1, void* y1, int n1, void* stream);
/// An expert the card lacks, for a call routed on the card: from its slot in the pinned mirror to a staging slot.
struct FarFetch {
    const void* src;
    void* dst;
};
/// tab[e] = slots[e] for each of n_expert experts, and for an expert some pair of ids[0..rr) picks that has no slot
/// (null), the next staging slot (stage + k * stride), listed in fetch[] with its mirror slot far[e]; *count is how
/// many.  One block.  CUDA and HIP only.
void far_plan(const void* const* slots, const void* const* far, const int32_t* ids, int rr, int n_expert,
              uint8_t* stage, size_t stride, void** tab, FarFetch* fetch, int* count, void* stream);
/// uses[ids[k]] += 1 for k in [0, rr), ids below n_expert only.
void count_picks(const int32_t* ids, int rr, int n_expert, int32_t* uses, void* stream);
/// The first *count of fetch[0..most), `bytes` each (a multiple of 16), copied by the card's threads.
void far_pull(const FarFetch* fetch, const int* count, int most, size_t bytes, void* stream);
/// A doorbell in pinned memory between the card and the host for a token's routed experts outside VRAM, after Strata
/// v0.2.19's Inkling hybrid mode (engine/src/inkling2/cuda/ink_hybrid.cuh): the card rings it with the experts it
/// lacks and goes on with its own, the host computes those and answers, and the card waits for the answer only where
/// it adds their sum.  Its fields are each a cache line from the others'.
constexpr int kBellJobs = 32;
struct ExpertBell {
    volatile uint32_t ring;    ///< the card's rings so far; a new value is a new job
    uint32_t pad0[15];
    volatile uint32_t ack;     ///< the ring the host has answered
    uint32_t pad1[15];
    volatile uint32_t abort;   ///< the host gave up: every wait ends at once
    uint32_t pad2[15];
    int32_t layer, n;          ///< the job: layer and how many experts
    int32_t e[kBellJobs];      ///< their ids
    float w[kBellJobs];        ///< their router weights
};
/// One token's picks ids[0..n_used) with weights w at layer `layer`: the ones whose slots[] entry is null, at a place
/// under skip_from, go into the bell, x (n floats) into hx, and the bell rings (seq[0] counts the rings; seq[1] is
/// this one, or 0 when there is none).  One block.  CUDA only.
void bell_ring(const void* const* slots, const int32_t* ids, const float* w, int n_used, int skip_from, int layer,
               const float* x, float* hx, int n, ExpertBell* bell, uint32_t* seq, void* stream);
/// Where seq[1] is a ring: wait until the bell's ack is that ring (or abort is set), then acc += hy (n floats).
void bell_add(const ExpertBell* bell, const uint32_t* seq, const float* hy, float* acc, int n, void* stream);
/// copy2() with a third run.  CUDA and HIP only.
void copy3(const void* x0, void* y0, int n0, const void* x1, void* y1, int n1, const void* x2, void* y2, int n2,
           void* stream);
/// A piece of a gather: `bytes` from src to dst, each 16-byte aligned, as is bytes.  32 bytes, so that the kernel
/// reads a piece in whole 16-byte loads.
struct GatherPiece {
    const void* src;
    void* dst;
    unsigned long long bytes, pad;
};
/// pieces[0..n) copied, by the card's threads: src may be pinned host memory, which the card then reads over the
/// bus, and so may the piece list itself.  One launch for many pieces, where a copy call each cost the host more
/// than the card took for them.
void gather_pieces(const GatherPiece* pieces, int n, void* stream);
/// Up to 32 vectors of n values, each times its weight into its own sum, one after the other: acc[k] += x[k] * w[k]
/// for k = 0, 1, ...  Two of them may have the same sum.
struct ScaledRows {
    const float* x[32];
    float* acc[32];
    float w[32];
};
void add_scaled(const ScaledRows& rows, int count, int n, void* stream);
/// acc[row] += b for each of the rows of n values.
void add_bias(const float* b, float* acc, int n, int rows, void* stream);

// ---- attention over a key and value cache (qwen35moe, gemma4), and the gated delta net with one decay a head

/// For `rows` tokens of `heads` heads each: a head's RMS norm with the weights w (dim values, the same for every
/// head; null: none), then ggml's rotary embedding on its first n_rot values in the NeoX pairing: value i goes
/// with value i + n_rot / 2, and the pair turns by position * theta_scale^i, divided by freq[i] when freq is
/// given.  A head's input starts x_stride floats after the one before it; y has dim values a head.
/// The position is the token's place in the call counted from `pos`.  With `axes` (three numbers a token) it is
/// the token's position on one of three axes instead: pair i takes axis (i % sections) % 3, ggml's interleaved
/// rope_multi, which is how a picture's rows get a line and a column.
/// With `at`, the first token's position is not `pos` but what the card holds at `at` when the launch runs: a
/// launch recorded once can then be replayed for every position (attend() and store_f16_at() take it too).
void head_norm_rope(const float* x, int x_stride, const float* w, float* y, int dim, int heads, int n_rot,
                    float theta_scale, int64_t pos, int rows, float eps, void* stream, const float* freq = nullptr,
                    const int32_t* axes = nullptr, int sections = 1, const long long* at = nullptr);
/// attend() for a step's sequences, each its own rows, cache and position: sequence k's queries q[k] (tokens[k]
/// rows), its caches, its position at[k] on the card and the room keys[k] its call is sized for, its output
/// out[k].  score[k] holds its scores and wt[k] its weights, apart from every other sequence's.  Each sequence gets
/// the bits attend() gives it alone.  CUDA only.
struct AttnSeqs {
    static constexpr int kMax = 32;
    int n = 0;
    const float* q[kMax] = {};
    float* out[kMax] = {};
    float* score[kMax] = {};
    float* wt[kMax] = {};
    const int8_t* kc[kMax] = {};
    const uint16_t* ks[kMax] = {};
    const int8_t* vc[kMax] = {};
    const uint16_t* vs[kMax] = {};
    const long long* at[kMax] = {};
    int keys[kMax] = {}, tokens[kMax] = {};
    size_t rs = 0;   ///< every cache's codes from a row to the next, as Kvq8::rs; 0: kv_heads * dim
    bool f16 = false;   ///< kc[k] and vc[k] are fp16 caches (const half*), ks and vs unused, rs 0
};
/// Whether attend_seqs() takes these sequences: what attend() would run for each of them, a q8_0 or fp16 cache
/// with no window, no ring and no sinks, has a kernel here.  It fills in score[k] and wt[k] from `score` (score_floats).
bool attend_seqs_plan(AttnSeqs& seqs, float* score, size_t score_floats, int chunk, int dim, int n_head, int kv_heads);
void attend_seqs(const AttnSeqs& seqs, int dim, int n_head, int kv_heads, float scale, void* stream);
/// head_norm_rope() for rows of different sequences: row r turns by turns[2 * r] (step()'s pairs of position and
/// turn, from the turn of the first), the bits a call of its own gives the row.  CUDA only.
void head_norm_rope_rows(const float* x, int x_stride, const float* w, float* y, int dim, int heads, int n_rot,
                         float theta_scale, int rows, float eps, void* stream, int sections, const long long* turns);
/// YaRN as ggml's rope_yarn computes it, for head_norm_rope_yarn() (laguna's full layers).  Pair i's angle is the
/// interpolated one, freq_scale times the angle head_norm_rope() gives it, mixed with that angle itself by the
/// ramp ext_factor * (1 - clamp((i - corr_low) / max(0.001, corr_high - corr_low), 0, 1)).  mscale multiplies the
/// rotated values' cos and sin: the final magnitude, the attention factor llama.cpp passes times the
/// 1 + 0.1 * ln(1 / freq_scale) ggml adds when ext_factor is not 0.  The values past n_rot are only normed.
struct RopeYarn {
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    float corr_low = 0.0f, corr_high = 0.0f;   ///< ggml_rope_yarn_corr_dims()
    float mscale = 1.0f;
};
/// head_norm_rope() with YaRN: the same norm, NeoX pairing, positions and `at`, with no frequency factors or
/// axes.  CUDA and HIP only.
void head_norm_rope_yarn(const float* x, int x_stride, const float* w, float* y, int dim, int heads, int n_rot,
                         float theta_scale, int64_t pos, int rows, float eps, void* stream, const RopeYarn& yarn,
                         const long long* at = nullptr);
/// An attention cache stored as ggml Q8_0: a row is `n` int8 codes and `n / 32` fp16 scales with
/// `n = kv_heads * dim`, row-major (`codes[row * n + blk * 32 + j]`, `scales[row * (n / 32) + blk]`).
/// A q8 read is active wherever `codes` is not null.  The value a block holds is
/// `(float)codes[..] * __half2float(scales[..])` with one scale a 32-value block.
/// rs: the codes from a row to the next where that is not n (several caches' rows side by side in one array, as
/// Qwen3.6's layers are on CUDA), and the scales rs / 32; 0 is n.  A multiple of 32.
/// ss: the scales from a row to the next where that is not rs / 32 (a row's codes and scales side by side in one
/// array, as Inkling 2's are); 0 is rs / 32.  Only attend_batch()'s dequantized products take it.
struct Kvq8 {
    const int8_t* codes = nullptr;
    const uint16_t* scales = nullptr;
    size_t rs = 0;
    size_t ss = 0;
};

/// Causal attention for the tokens from position `pos`: token t attends the rows 0 to pos + t, or with `window`
/// only the last `window` of them.  q: n_head * dim a token.  kcache and vcache: fp16, kv_heads * dim a row; the
/// heads read a row's keys and values in turn, n_head / kv_heads heads each.  With `ring`, the sequence's row p
/// is cache row p % ring.  out: n_head * dim a token.  It takes `chunk` tokens a launch, which read the caches
/// once between them, and score is scratch of chunk * n_head floats for each row a launch's tokens attend
/// between them: pos + rows, or window - 1 + chunk if that is less.  A token comes out the same whatever tokens
/// the call has beside it.  Queries and keys are read four values at a time: dim is a multiple of 4, q lies at
/// a multiple of 16 bytes and kcache of 8, or it throws std::invalid_argument.
/// With `whole` the call's tokens see each other: each attends up to the call's last row, so a token can attend
/// window - 1 + rows of them.
/// With `at`, the tokens start at the position the card holds there; the launches are sized for `room` rows,
/// which must be no fewer than a launch's tokens attend between them, and score has room for them.  With `whole`
/// too the call is one launch: rows is no more than chunk.
/// score_floats: the floats score holds, when more than the above.  Heads of 64 values in groups of 4, or of 128 in
/// groups of 8, over the whole conversation then take a launch block each block of rows, a launch as many tokens as
/// score has room for those blocks' sums of at the longest the call can attend (attend_split_floats() for chunk of
/// them); the results differ from the other kernels' in the last bits.
/// With `kq8` or `vq8` (codes not null) that cache is read as Q8_0 rather than fp16: its codes/scale pointers
/// follow the same row and head index the fp16 cache would, and a block of 32 values dequantizes as
/// `(float)codes[..] * __half2float(scales[..])`.  The fp16 path is unchanged where they are left null.
/// With `sinks` (n_head floats on the card, flint's), each head's softmax has one more logit, its sink, which
/// weighs no value: it only takes weight from the rows.
void attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows, float* score,
            int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window = 0,
            int ring = 0, bool whole = false, const long long* at = nullptr, int room = 0, size_t score_floats = 0,
            Kvq8 kq8 = {}, Kvq8 vq8 = {}, const float* sinks = nullptr, size_t part_floats = 0,
            float* ml_out = nullptr);
/// Merges a piece's normalized attention output (piece_out) and per-(token, head) (max, sum) pairs (piece_ml)
/// into the running accumulator (acc_out, acc_ml) via exact online softmax.
void merge_attn_piece(float* acc_out, float* acc_ml, const float* piece_out, const float* piece_ml, int rows,
                      int n_head, int dim, void* stream);
/// The floats of score that attend()'s blocks of rows take for calls of up to chunk tokens attending up to max_rows
/// rows (score_floats): with that much a model's every call takes them.
size_t attend_split_floats(int chunk, int n_head, int max_rows, int dim);
/// attend() for a prompt batch: the same attention with cuBLAS products on the tensor cores, several times
/// faster on many tokens.  The products read the cache's halves where they lie, take the queries and the
/// softmax's weights rounded to halves, and sum floats.  A token's result depends on the tokens beside it, so
/// the windows of guessed tokens use attend().
/// It takes the call's tokens together and the rows they attend a block at a time: as many rows as have their
/// scores and weights, 6 bytes for each of a token's n_head heads, in the score_floats floats of `score`, which
/// must hold 2 * n_head floats.  work: rows * n_head * (dim + dim / 2 + 3) + 1 floats.  Throws
/// std::runtime_error.
/// Heads of 64 values in groups of 4 over the whole conversation take one kernel instead, which keeps the scores in
/// registers, on NVIDIA cards from Turing on; it uses neither blas, score nor work.  So do heads of 128 values in
/// groups of 8, with a window, a ring and sinks too, from Ampere on; a token's keys go in blocks of 64 counted from
/// the cache's first row, so its result is the same whatever tokens the call has beside it.
/// With `kq8` or `vq8` (codes not null) that cache is read as Q8_0 like attend(): before each block of `m` keys
/// the product reads, its rows are dequantized into the fp16 workspace `kwork` / `vwork`, and the products then
/// read that exactly as they read an fp16 cache, so the results match attend()'s q8 read.  Each workspace holds
/// work_halves halves, `work_halves / (kv_heads * dim)` rows at most: a block of keys larger than that is split,
/// taking more products.  The fp16 path (both null) is unchanged and reads the caches where they lie.
/// `sinks` as attend()'s: the sink joins each token's sum after the last block of rows.
/// `bias` (null: none): a score gets its key's bias by distance and loses its token's sink first (AttnBias), in the
/// cuBLAS products' path only, which a model with a bias takes whatever its heads.
/// A bias a score gets by its key's distance d from the query (the query's position minus the key's), for a model
/// with learned relative positions: the dot product of the query's n_rel values (r: n_rel a head, the call's
/// tokens in the engine's order) with table's column d (n_rel values a distance), nothing from `extent` on; and
/// the token's own sink, a head (sink), taken off every score.
struct AttnBias {
    const float* r = nullptr;
    const float* sink = nullptr;
    const float* table = nullptr;
    int n_rel = 0, extent = 0;
};
void attend_batch(void* blas, const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows,
                  float* score, size_t score_floats, float* work, float* out, int dim, int n_head, int kv_heads,
                  float scale, void* stream, int window = 0, int ring = 0, bool whole = false, Kvq8 kq8 = {},
                  Kvq8 vq8 = {}, uint16_t* kwork = nullptr, uint16_t* vwork = nullptr, size_t work_halves = 0,
                  const float* sinks = nullptr, float* ml_out = nullptr, const AttnBias* bias = nullptr);
/// attend() over int8 caches (store_int8): the same attention, the same order of products and sums.  `q8` is scratch
/// in the same form for this call's queries (rows * n_head * dim / kQ8_0Values blocks of codes, and as many
/// deltas).  A score is a dp4a of a lane's own run of codes - the query's against the key's, both 4-aligned - and
/// the block's two deltas scale the integer sum once: llama.cpp's vec_dot_q8_0_q8_1, whose whole point is that
/// neither side is ever turned into floats.  No transposed-value path (a decode-time call has few tokens and reads
/// the weights from the scores' buffer).  `at` (a recorded launch) and `room` as in attend().
void attend_int8(const float* q, KvInt8 q8, KvInt8 k, KvInt8 v, int64_t pos, int rows, float* score, int chunk,
                 float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window = 0, int ring = 0,
                 bool whole = false, const long long* at = nullptr, int room = 0);
/// attend_int8 with the values' side left in fp16 (--kv int8k): the scores are the int8 ones against the keys'
/// codes, the weighted sums read the fp16 cache the usual way.  `v_h` is the values' cache, `n_kv * dim`
/// halves a row, the ring's included.
void attend_int8_mixed(const float* q, KvInt8 q8, KvInt8 k, const uint16_t* v_h, int64_t pos, int rows,
                       float* score, int chunk, float* out, int dim, int n_head, int kv_heads, float scale,
                       void* stream, int window = 0, int ring = 0, bool whole = false,
                       const long long* at = nullptr, int room = 0);
/// x = cap * tanh(x / cap).
void softcap(float* x, size_t n, float cap, void* stream);
/// The first of the largest of n logits and the softmax's probability of it, summed without the logits more than
/// 20 below it.
struct Likeliest {
    int32_t token;
    float prob;
};
void likeliest(const float* logits, int n, Likeliest* out, void* stream);
/// likeliest() for `rows` rows of n scores back to back, out[r] row r's.
void likeliest_rows(const float* logits, int n, int rows, Likeliest* out, void* stream);
/// acc[r][i] *= sigmoid(g[r][i]) for `rows` groups of n values; group r's gates start r * g_stride floats into g.
/// With one_gate a group has one gate for all its values.
void mul_sigmoid(float* acc, const float* g, int n, int rows, int g_stride, bool one_gate, void* stream);
/// mul_sigmoid() with ggml's softplus in place of the sigmoid: g above 20 as it is, else log1p(exp(g)) (laguna's
/// attention output gate).  CUDA and HIP only.
void mul_softplus(float* acc, const float* g, int n, int rows, int g_stride, bool one_gate, void* stream);
/// A causal convolution of four taps over each of `channels` channels, and its SiLU.  x and out: `channels` values
/// a token.  state: 3 past inputs a channel, oldest first, advanced here by `rows` tokens.  w: (4, channels).
/// With snap, the state as it is after some of the call's tokens is also written there, for a window of guessed
/// tokens to go back to (gdn_step too): `snaps` has a bit for each token that leaves it, and the states lie in
/// the tokens' order, `stride` floats apart.
void conv4_silu(const float* x, float* state, const float* w, float* out, int channels, int rows, void* stream,
                float* snap = nullptr, size_t stride = 0, unsigned snaps = 0);
/// LFM2's gated short convolution (llama.cpp's shortconv block between its two projections).  bcx: B, C and x of a
/// token, `channels` values each, as the in_proj writes them.  out = C * conv3(B * x), `channels` values a token,
/// a causal convolution of three taps w (3, channels).  state: the 2 B * x before the call a channel, oldest
/// first, advanced here by `rows` tokens.  snap, stride, snaps: as conv4_silu's.
void shortconv(const float* bcx, float* state, const float* w, float* out, int channels, int rows, void* stream,
               float* snap = nullptr, size_t stride = 0, unsigned snaps = 0);
/// The delta-rule step of every value head and its gated output, token after token: ggml's gated_delta_net op
/// with one decay a head.  qkv: a token's conv outputs, the k_heads key heads' q, their k, then the n_head value
/// heads' v, dim values each; value head h reads key head h % k_heads.  alpha and beta: n_head a token, as their
/// projections give them.  dt, a: n_head.  The state decays by exp(softplus(alpha + dt) * a) and takes
/// sigmoid(beta) of the error.  s: n_head * dim * dim, updated.  z: n_head * dim a token.
/// out = rms_norm(o) * o_norm * silu(z), n_head * dim values a token.  qkv, alpha and beta are overwritten.
void gdn_step(float* qkv, float* alpha, float* beta, const float* dt, const float* a, float* s,
              const float* z, const float* o_norm, float* out, int dim, int n_head, int k_heads, int rows,
              float norm_eps, void* stream, float* snap = nullptr, size_t stride = 0, unsigned snaps = 0);
/// The rows of several sequences in one call, each with a recurrent state of its own: sequence k has rows
/// [first[k], first[k] + rows[k]) of the call, its state in state[k], and snap[k], when not null, takes the state as
/// it is after its first row (a token and its guess).  CUDA only.
struct SeqStates {
    static constexpr int kMax = 32;
    int n = 0;
    int first[kMax] = {}, rows[kMax] = {};
    float* state[kMax] = {};
    float* snap[kMax] = {};
};
/// conv4_silu() and gdn_step() for each sequence of `seqs`, in one launch a kernel where they launched one a
/// sequence: each sequence's values are the bits a call of its own gives.  rows: all of the call's.
void conv4_silu_seqs(const float* x, const SeqStates& seqs, const float* w, float* out, int channels, void* stream);
/// shortconv() for each sequence of `seqs` (no snapshots), one launch.
void shortconv_seqs(const float* bcx, const SeqStates& seqs, const float* w, float* out, int channels, void* stream);
void gdn_step_seqs(float* qkv, float* alpha, float* beta, const float* dt, const float* a, const SeqStates& seqs,
                   const float* z, const float* o_norm, float* out, int dim, int n_head, int k_heads, int rows,
                   float norm_eps, void* stream);
/// conv4_silu() for a block of `rows` tokens that see each other (Clover's block guesses): a token's sum adds
/// input t + d of the block times right[d - 1] for d = 1 to 3, right (3, channels).  The state is only read.
void conv4_block_silu(const float* x, const float* state, const float* w, const float* right, float* out,
                      int channels, int rows, void* stream);
/// gdn_step() for a block of up to 32 tokens that see each other (Clover's block guesses), which leaves the state
/// as it was.  Token i's output before the norm is a softmax's weights on the block's values and on the state's
/// read S q_i times the token's decay: by the dot product of the normed query and key times logit_scale[h] for
/// token j, by hist_b[h] + hist_c[h] * log(rows / 32) for the state.  The share beta is not used.  w: scratch of
/// n_head * rows * (rows + 1) floats.  Takes and leaves the rest as gdn_step() does.
void gdn_block(float* qkv, float* alpha, float* beta, const float* dt, const float* a, const float* s,
               const float* z, const float* o_norm, const float* logit_scale, const float* hist_b,
               const float* hist_c, float* out, float* w, int dim, int n_head, int k_heads, int rows,
               float norm_eps, void* stream);
/// The router scores of `rows` tokens that guess a block together (Clover), kept to few experts: each token's n_used
/// likeliest by softmax probability, those probabilities added up per expert over the rows, and the experts in order
/// of that sum while the ones before them hold less than `share` of it.  The others' scores become -inf in every
/// row.  Ties go to the lower expert.  n_expert is a multiple of 32, up to 1024.
void coreset_mask(float* logits, int n_expert, int n_used, int rows, float share, void* stream);

// ---- DeepSeek V4 (deepseek4): one key head that is its value too, a window of raw rows, rows compressed 4 or 128
// to one, and an indexer that picks among the 4-to-1 rows

/// ggml_rope_ext's turn in the normal pairing (value i with i + 1), with YaRN.  corr0 and corr1 are what
/// ggml_rope_yarn_corr_dims() gives; attn_factor is the magnitude the turn starts from.
struct Rope {
    float theta_scale = 1.f;   ///< freq_base^(-2 / n_rot)
    float freq_scale = 1.f, ext_factor = 0.f, attn_factor = 1.f;
    float corr0 = 0.f, corr1 = 0.f;
};
/// For `rows` tokens of `heads` heads of dim values: with norm, each head RMS-normed without weights; then the
/// n_rot values from `off` turned by the position and the others kept.  Token t's position is pos + t * step.
/// back: the turn the other way (ggml_rope_ext_back).  y may be x.  With `at` (step 1), token t's position is what
/// the card holds at at[2 * t] when the launch runs: a launch recorded once is replayed for every position.  The
/// kernels below take `at` the same way.
void dsv4_rope(const float* x, float* y, int dim, int heads, int off, int n_rot, const Rope& r, int64_t pos,
               int step, int rows, bool norm, float eps, bool back, void* stream, const long long* at = nullptr);
/// A compressor's rows into its rings, position p at ring row p % ring: kv as it is, and score plus
/// ape[p % ratio].  width values a row.
void dsv4_state(const float* kv, const float* score, const float* ape, float* ring_kv, float* ring_score, int ring,
                int width, int ratio, int64_t pos, int rows, void* stream, const long long* at = nullptr);
/// For each of the `rows` tokens from position pos that ends a block of ratio positions, the block's compressed row
/// into `cache` (fp16, dim values a block, at the block's index): value d is the sum over the block's rows of
/// softmax(score) * kv, RMS-normed with `norm` (dim weights), its last n_rot values turned by the block's first
/// position.  Block b's rows are the ratio from position b * ratio, with their first dim values.  With overlap a
/// row is 2 * dim values: the block's rows give their second dim, and the ratio rows before the block their first
/// dim (a row before the sequence is left out).  dim is a multiple of 32 up to 1024.
void dsv4_squeeze(const float* ring_kv, const float* ring_score, int ring, int dim, int ratio, bool overlap,
                  const float* norm, float eps, const Rope& r, int n_rot, int64_t pos, int rows, uint16_t* cache,
                  void* stream, const long long* at = nullptr);
/// The indexer's scores of the tokens from position pos: score[b] = scale * sum_h w[h] * relu(q[h] . key[b]) for
/// the compressed rows b < (pos + t + 1) / ratio.  q: n_head * dim a token; w: n_head; keys: fp16, dim a row.  A
/// token's scores start score_stride apart.  dim must be 128.  Throws std::invalid_argument, and
/// std::runtime_error when the card's shared memory has no room for a token's queries.
/// With `at`, the launch is sized for the tokens of positions below `room`.
void dsv4_idx_score(const float* q, const float* w, const uint16_t* keys, float* score, int score_stride, int dim,
                    int n_head, int ratio, int64_t pos, int rows, float scale, void* stream,
                    const long long* at = nullptr, int room = 0);
/// For each token from position pos, its compressed rows to attend, in their order: the n_top with the highest
/// scores, the lower row first among equals, or all of them when it sees n_top or fewer.  sel_stride apart.
void dsv4_select(const float* score, int score_stride, int32_t* sel, int sel_stride, int n_top, int ratio,
                 int64_t pos, int rows, void* stream, const long long* at = nullptr);
/// The attention of the tokens from position pos.  Token t (position p) attends the raw rows of the `window`
/// positions up to p (row p at p % ring of `raw`), then compressed rows of `comp`: with sel, the
/// min((p + 1) / ratio, top) of its list; without, every one below (p + 1) / ratio; with comp null, none.  The
/// keys are fp16, dim values a row, and each key is its value too.  q: n_head * dim a token; sinks: a logit a
/// head that takes part in the softmax with no value.  out: n_head * dim a token.  Throws std::runtime_error when
/// a token's keys do not fit the card's shared memory.  With `at`, the launch is sized for positions below `room`.
void dsv4_attend(const float* q, const uint16_t* raw, int ring, int window, const uint16_t* comp, int ratio,
                 const int32_t* sel, int sel_stride, int top, const float* sinks, float* out, int dim, int n_head,
                 float scale, int64_t pos, int rows, void* stream, const long long* at = nullptr, int room = 0);
/// y[g][t] = x[t][g] for `rows` tokens of `groups` groups of n values; with back, y[t][g] = x[g][t].
void permute_groups(const float* x, float* y, int n, int groups, int rows, bool back, void* stream);

}  // namespace strata::poly
