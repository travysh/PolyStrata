// src/poly/inkling2_zha.hpp - the ZHA index of Inkling 2's full layers (inkling2_zha.cu, docs/INKLING_ZHA.md).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::poly::ink {

/// ZHA (Zyphra Hierarchical Attention, arXiv 2610.42112) over a full layer's keys, one level: chunks of C keys,
/// each summarized for R reference queries; a token's attention reads key 0's chunk, its last `recent` keys and the
/// chunks its summaries rank best, and (tail) the other chunks as one estimated key each.
// q8n's flag: the summaries as 4-bit codes (two a byte, low nibble first, value (code - 8) * scale) then their fp16
// scales a 32 values; without it int8 codes then the scales.  Bits 32-61: the summaries' chunks a KV head (their
// room, which grows with a slot's context; 0: max_chunks).  Bits 0-31: the summaries' rows.
constexpr size_t kZhaQ4 = size_t(1) << 62;
inline size_t zha_q8n_make(size_t rows, int cap, bool q4) { return rows | ((size_t) cap << 32) | (q4 ? kZhaQ4 : 0); }

/// The index's shape (zha_attend() takes C 64 and R 4 only): a KV group's heads read the same chunks, a prompt's
/// tokens choose theirs 64 at a time, the summaries and the leaves are 4-bit, the ring Q8_0.
struct Zha {
    int C = 64, R = 4, recent = 1024;
    float budget = 0.16f;   ///< the share of a token's far chunks it reads exactly
    int min_chunks = 1;
    int max_sel = 256;      ///< at most this many chunks a selection
};

/// Where a full layer's keys and values are: fp16 (k16, v16) or Q8_0 (kc, ks, vc, vs: int8 codes, an fp16 scale a
/// 32-value block).  Key k of KV head g is at k * row (+ g * 128 if headed: a cache's rows hold every KV head's).
struct ZhaKv {
    const uint16_t* k16 = nullptr;
    const uint16_t* v16 = nullptr;
    const int8_t* kc = nullptr;
    const uint16_t* ks = nullptr;
    const int8_t* vc = nullptr;
    const uint16_t* vs = nullptr;
    long long row = 0;
    int headed = 1;
    int q4 = 0;   ///< kc/vc hold 4-bit codes, two a byte (low nibble first), value = (code - 8) * scale
};

/// rows rows of n values (x) as 4-bit codes and an fp16 scale a 32-value block, from row pos of codes / scales
/// (n / 2 bytes and n / 32 fp16 a row); with ring, row p goes to p % ring.
void store_q4(const float* x, int8_t* codes, uint16_t* scales, int n, long long pos, int rows, long long ring,
              void* stream);
/// rows rows of n values from 4-bit (src_codes, src_scales, from row `from`) to Q8_0 (dst, from row `to`, % ring).
void q4_to_q8(const int8_t* src_codes, const uint16_t* src_scales, long long from, int8_t* dst_codes,
              uint16_t* dst_scales, long long to, long long ring, int n, int rows, void* stream);

/// ZHA's scratch: S s_floats floats; score (heads a group + 1) * rows * max_chunks
/// floats; rstar and sel heads a group * rows * max_chunks bytes; cmap max_chunks ints, count one; stage: a KV head's
/// keys and values of max_chunks chunks (row 128, not headed), in the leaves' format.
struct ZhaWork {
    float* S = nullptr;
    size_t s_floats = 0;
    float* score = nullptr;
    uint8_t* rstar = nullptr;
    uint8_t* sel = nullptr;
    int* cmap = nullptr;    ///< [heads][max_chunks]: where each chunk is staged (heads 1, or kv_heads when kept)
    int* count = nullptr;   ///< [heads] staged chunks
    int* glist = nullptr;   ///< [max_chunks + 1]: the chunks a call stages (count first)
    int heads = 1;          ///< 1: a KV head's chunks at a time; kv_heads: every head's, kept across a slot's calls
    ZhaKv stage;
};

/// A full layer's slot's chunks kept in VRAM between an answer's tokens: a pool of P chunks a KV head (pool: the
/// leaves' format, row 128, not headed; head g's chunk s at g * P + s), which chunk each holds (owner [kv][P]), when it
/// was last read (stamp [kv][P]), where each chunk is (pmap [kv][max_chunks], -1 if not), the clock hand and the
/// misses of a step ([kv][P + 2]: hand, count, chunks).  A step of one token reads its chunks from it.
struct ZhaCache {
    ZhaKv pool;
    int P = 0;
    int* owner = nullptr;
    int* stamp = nullptr;
    int* pmap = nullptr;
    int* aux = nullptr;
    int now = 0;
};
/// Forget every chunk the cache holds (the sequence jumped: the leaves may hold other rows now).
void zha_cache_clear(ZhaCache& cache, int kv_heads, int max_chunks, void* stream);

/// The summaries of chunks [c0, c1) of a full layer's slot, from its ring (key j at j % n_ring): per KV head g and
/// reference r (refs [kv_heads][R][128]), khat = sum softmax(q_r.k / sqrt 128) k, vhat the values with the same
/// weights (fp16, [kv_heads][max_chunks][R][128]), bhat = LSE - q_r.khat / sqrt 128 ([kv_heads][max_chunks][R]).
void zha_build(const float* refs, const ZhaKv& ring, long long n_ring, int c0, int c1, int C, int R, int kv_heads,
               int max_chunks, uint16_t* khat, uint16_t* vhat, float* bhat, void* stream, size_t q8n = 0);

/// The attention of `rows` tokens from position pos (q [rows][n_head][128], already scaled by tau) over a full
/// layer's slot through its summaries (chunks [0, built)): chunk 0 and the selected far chunks from the leaves (staged
/// into wk.stage a KV head at a time), the recent keys from the ring, the bias by distance (r [rows][n_head][nrel]
/// times table [extent][nrel]) and each token's sink.  out [rows][n_head][128].
void zha_attend(void* blas, const Zha& z, const float* q, const ZhaKv& leaves, const ZhaKv& ring, long long n_ring,
                const uint16_t* khat, const uint16_t* vhat, const float* bhat, int built, int max_chunks, long long pos,
                int rows, int n_head, int kv_heads, const float* r, const float* table, int extent, int nrel,
                const float* sinks, const ZhaWork& wk, float* out, void* stream, ZhaCache* cache = nullptr,
                bool keep = false, int sum_cap = 0);   // sum_cap: the summaries' chunks a KV head (0: max_chunks)

}  // namespace strata::poly::ink
