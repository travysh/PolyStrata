// src/kernels/cuda/i8_mma.cuh - a group of quantized matrices times rows of int8 activations on the int8 tensor cores
// (gemm_iq_i8_grouped, declared in include/strata/kernels/iq_kernels.hpp), for a matrix of any size.  Included once by
// i8_mma.cu inside its anonymous namespace, after i8_expert.cuh: a weight sub-block is decoded by that file's
// i8_load / i8_pack / i8_convert (32 values as 8 words of int8, their integer scales and float base), or read by the
// loaders here and decoded by i8_pack / i8_convert, Q3_K, Q2_K, IQ1_S, MXFP4 and NVFP4 by the decoders here.  CUDA only, Ampere and newer: mma.sync
// m16n8k32 and m16n8k16 with s8 operands; Turing when asked (gemm_iq_i8_turing()), by m8n8k16.
//
// A block: WR = 64 weight rows of one matrix x up to 16 NJ of its activation rows (blockIdx.y walks a matrix's rows
// in such chunks).  K goes in stages of 256 values; a type of 32-value blocks (MXFP4's too; NVFP4: 64) may end inside the last one
// (gemm_iq_i8_takes()).  Per stage every thread decodes 8 WR / NT sub-blocks of one weight row (NT threads, 256 or
// 512: i8m_threads()) into shared memory, the stage's activations are copied beside them, and a warp multiplies 16
// weight rows with its tiles of 8 activation rows: one MMA per 32 values, or per 16 for a type with a scale per 16.
// A tile takes most of the shared memory a multiprocessor has, so a block waits alone for what it reads from the
// card's memory: a thread issues all of a stage's reads together, into registers, and the next stage's before it
// multiplies this one, so the wait is behind the products.  (Read and stored one piece after the other, the reads
// were more than half of a launch's time: the experts of Qwen3.6 UD-Q3_K_M, 256 of 64 rows each, took 20-30% longer.)
// An MMA's int32 sums become floats at once, sum * (the weight sub-block's scale) * (the activation block's scale),
// and are added to the thread's FP32 sums; a type with a minimum per sub-block adds min * (the activation block's
// scale times the sum of its codes).  A value's sum is its own, stage after stage, so it does not depend on the
// launch its row is in.
//
// The activations (gemm_iq_i8_quantize): [rows][K] int8 codes, then [rows][K / 32] float scales, then [rows][K / 32]
// floats scale * (sum of the block's codes).  A block of 32 values: scale = max|x| / 127, code = round(x / scale).
constexpr int I8M_NT = 256;
constexpr int I8M_LD = 68;   // words of a shared row of codes: 64 + 4, so the 8 rows a warp reads at once lie in 8 banks

#define STRATA_I8M_TYPES(X) X(2) X(6) X(7) X(8) X(10) X(11) X(12) X(13) X(14) X(16) X(17) X(18) X(20) X(21) X(22) X(23) X(29) X(19) X(39) X(40) X(42)
__host__ __device__ constexpr bool i8m_type(int t) { return t == 10 || t == 11 || t == 19 || t == 39 || t == 40 || i8_block_bytes(t) != 0; }
__host__ __device__ constexpr bool i8m_two(int t) { return t == 10 || t == 11 || t == 40 || i8_two_scales(t); }   // a scale per 16 values
__host__ __device__ constexpr bool i8m_min16(int t) { return t == 10; }   // and a minimum per 16 values (Q2_K)

// Bytes of a weight row's stage (256 values) as stored.  The block copies a stage's rows as they are, coalesced, into
// shared memory with cp.async while the stage before is multiplied, and decodes them from there (RAW).  Read row by row
// by each thread, their 4-byte pieces took a quarter of a launch to issue: the 32 lanes of a load are in 8 rows.  A row's
// stage keeps its place in a 16-byte piece (a Q6_K one may start 2 bytes in), so the loaders read it as they would read
// the card's memory: the copy is of the 16-byte pieces that hold it, and a row's slot has one piece more than the bytes
// (RW, the words of a slot), but for a type whose stages are whole pieces (Q8_0, Q4_K, ...: the host sees that the rows
// are too).
__host__ __device__ constexpr int i8m_raw_bytes(int t) {
    return t == 2 || t == 20 ? 144 : t == 6 ? 176 : t == 7 ? 192 : t == 8 ? 272 : t == 10 ? 84 : t == 11 ? 110 : t == 12 ? 144
         : t == 13 ? 176 : t == 14 ? 210 : t == 16 ? 66 : t == 17 ? 74 : t == 18 ? 98 : t == 21 ? 110 : t == 22 ? 82
         : t == 23 ? 136 : t == 29 ? 56 : t == 42 ? 72 : 0;
}
// The types read so: those of whole pieces, and Q6_K.  On an RTX PRO 6000 the others' experts of 2,048 values a row were
// slower so (Q3_K 617 -> 660 us, IQ3_XXS 513 -> 547, Q2_K 806 -> 820) though those of 512 were faster.
__host__ __device__ constexpr bool i8m_raw_type(int t) { return i8m_raw_bytes(t) != 0 && (i8m_raw_bytes(t) % 16 == 0 || t == 14); }

// Threads of a block: 512 for the widest tile, 16 warps on a multiprocessor where 8 left the products waiting on their
// operands, but for Q4_K and Q5_K (RTX PRO 6000, 2,048 rows: Q3_K experts 620 -> 578 us, IQ4_XS 293 -> 246, Q8_0 [2048
// -> 8192] 309 -> 297; Q4_K [2048 -> 6144] 265 -> 280, Q5_K experts 357 -> 370); 256 otherwise.
__host__ __device__ constexpr int i8m_threads(int t, int nj, int wr = 64) { return wr == 128 || (nj == 8 && t != 12 && t != 13) ? 512 : 256; }
// Weight rows of a block: 128 for IQ3_XXS in the widest tile, each activation row it reads going into twice the
// products (RTX PRO 6000, Qwen3.6's experts' gate|up, 2,048 rows: 472 -> 401 us); 64 otherwise (Q4_K [2048 -> 6144]
// 253 -> 298, Q3_K experts 579 -> 711, Q8_0 [2048 -> 2048] 82 -> 106 with 128).  The block of 128 has 512 threads,
// each warp two tiles of 16 weight rows.
__host__ __device__ constexpr int i8m_rows(int t, int nj) { return nj == 8 && t == 18 ? 128 : 64; }
// Tiles of 16 weight rows a warp multiplies: two for Q8_0, Q4_K and Q5_K in the widest tile (RTX PRO 6000, 2,048 rows:
// Q4_K [2048 -> 6144] 265 -> 253 us, LFM2.5's Q4_K experts 647 -> 632, Q8_0 [2048 -> 8192] 297 -> 287), one otherwise
// (Q3_K experts 578 -> 642, IQ3_XXS 473 -> 498, Q6_K 345 -> 363 with two).
__host__ __device__ constexpr int i8m_tiles_a(int t, int nj, int wr = 64) { return wr == 128 || (nj == 8 && (t == 8 || t == 12 || t == 13)) ? 2 : 1; }

template <int T, int NJ, int WR, bool RAW = false>
struct I8mSmem {
    static constexpr int MT = 16 * NJ;                                           // activation rows of a tile
    static constexpr int GW = !i8_gg_type(T) && i8_grid_words(T) > 4 ? (i8_grid_words(T) + 3) / 4 * 4 : 0;
    static constexpr int SG = i8_sign_type(T) ? 512 : 0;
    static constexpr int NS = i8m_two(T) ? 16 : 8;                               // scales of a weight row a stage
    static constexpr int MN = i8_min_type(T) ? 1 : 0;
    // A type with a minimum per 32 values keeps {scale, minimum} of a sub-block together (PK): those of two sub-blocks
    // are one 16-byte read
    static constexpr bool PK = MN && !i8m_two(T);
    static constexpr int LDS = PK ? 2 * NS + 4 : NS + 1, LDB = 9;               // strides: no two of a warp's reads in one bank
    static constexpr int M16 = i8m_min16(T) ? 1 : 0, LDM = 17;                   // Q2_K: [WR][LDM] minimums, one per 16 values
    // The activation rows' scales of a stage, by 32-value block: the rows in pairs, {d of the even row, d of the odd one}
    // and for a type with a minimum {scale * sum of both}, so that a thread reads the two rows of its products at once.
    // Q2_K: by 16 values, {scale * sum of the 16 codes} of the pair (S16 floats a group of 16).
    static constexpr int XQ = MN ? 4 : 2;
    static constexpr int XSTR = MT / 2 * XQ + 4;                                 // floats a block: +4 spreads the blocks over the banks
    static constexpr int S16 = MT + 4;
    static constexpr int o_sg = GW, o_w = o_sg + SG, o_x = o_w + WR * I8M_LD, o_ws = o_x + MT * I8M_LD, o_wm = o_ws + WR * LDS,
                         o_xq = (o_wm + (MN && !PK) * WR * LDB + M16 * WR * LDM + 3) / 4 * 4, o_x16 = o_xq + 8 * XSTR,
                         o_id = o_x16 + M16 * 16 * S16, o_raw = (o_id + MT + 3) / 4 * 4,
                         RW = !RAW ? 0 : i8m_raw_bytes(T) % 16 == 0 ? i8m_raw_bytes(T) / 4 : ((i8m_raw_bytes(T) + 15) / 16 + 1) * 4,
                         total = o_raw + WR * RW;
    static constexpr size_t bytes = (size_t) total * 4;
};

// (not volatile: a product has no effect but its sums, and the compiler may then move it among the others)
// Turing (sm_75) has the int8 MMA only as m8n8k16: an m16n8k32 is four of them and an m16n8k16 two, on the same
// operand registers (a0 / a1 the fragment's rows g and g + 8, a2 / a3 its next 16 values; b0 / b1 likewise).  The
// sums are integers, so they are the same as Ampere's to the bit.
__device__ __forceinline__ void i8m_mma8(int& c0, int& c1, uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a), "r"(b));
#endif
}
__device__ __forceinline__ void i8m_mma32(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    i8m_mma8(c[0], c[1], a0, b0);
    i8m_mma8(c[0], c[1], a2, b1);
    i8m_mma8(c[2], c[3], a1, b0);
    i8m_mma8(c[2], c[3], a3, b1);
#endif
}
__device__ __forceinline__ void i8m_mma16(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
#else
    i8m_mma8(c[0], c[1], a0, b0);
    i8m_mma8(c[2], c[3], a1, b0);
#endif
}
// Four 8 x 16-byte pieces of shared memory, a lane's 16 bytes at p (lanes 8 m to 8 m + 7: the rows of piece m): word
// (lane & 3) of row (lane >> 2) of each piece, which is an MMA's operand as the tensor cores take it.  One instruction
// for what took four 32-bit reads.
__device__ __forceinline__ void i8m_ldsm4(uint32_t (&r)[4], const uint32_t* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((uint32_t) __cvta_generic_to_shared(p)));
#endif
}

// cp.async: 16 bytes from the card's memory into shared memory, the first n of them read and the rest zeros, without
// registers.  i8m_cp_wait: this thread's copies have landed.
__device__ __forceinline__ void i8m_cp16n(void* s, const void* g, int n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"((uint32_t) __cvta_generic_to_shared(s)), "l"(g), "r"(n));
#endif
}
__device__ __forceinline__ void i8m_cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void i8m_cp_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
}

// N 32-bit words from p, which is 2-byte aligned: the aligned words that hold them, joined by a funnel shift (N + 1
// reads where p is not 4-byte aligned, N where it is; the last a 16-bit read, so that nothing past the run is read).
// i8_ld32 reads a word as two 16-bit halves, 2 N reads for a run.
template <int N>
__device__ __forceinline__ void i8m_ldrun(I8P p, uint32_t (&out)[N]) {
    const uint32_t sh = (uint32_t) ((uintptr_t) p & 3u) * 8u;
    I8P const b = p - (sh >> 3);
    uint32_t w[N + 1];
#pragma unroll
    for (int k = 0; k < N; ++k) w[k] = i8_ldw(b + 4 * k);
    w[N] = sh ? i8_ld16(b + 4 * N) : 0u;
#pragma unroll
    for (int k = 0; k < N; ++k) out[k] = __funnelshift_r(w[k], w[k + 1], sh);
}

// The loaders of the 2-byte aligned K-quant and Q8_0 blocks and of IQ4_XS (4-byte aligned) for this kernel, by runs of
// whole words: the words i8_load reads for the sub-block, in its order (i8_pack and i8_convert take them as they are).
// Q8_0: block ib of stage S (34 bytes: d, 32 code bytes); Q6_K: super-block S (210 bytes: ql 128, qh 64, 16 int8 scales,
// d); IQ4_XS: super-block S (136 bytes: d, scales_h, scales_l 4, 128 code bytes).
__device__ __forceinline__ void i8m_q8_load(I8P row, int S, int ib, uint32_t (&r)[9]) {
    I8P b = row + (size_t) S * 272 + 34 * ib;
    uint32_t q[8];
    i8m_ldrun<8>(b + 2, q);
#pragma unroll
    for (int k = 0; k < 8; ++k) r[k] = q[k];
    r[8] = i8_ld16(b);
}
__device__ __forceinline__ void i8m_q6k_load(I8P row, int S, int ib, uint32_t (&r)[18]) {
    I8P b = row + (size_t) S * 210;
    const int h = ib >> 2, k4 = ib & 3;
    uint32_t lo[8], hi[8];
    i8m_ldrun<8>(b + 64 * h + 32 * (k4 & 1), lo);
    i8m_ldrun<8>(b + 128 + 32 * h, hi);
#pragma unroll
    for (int k = 0; k < 8; ++k) { r[k] = lo[k]; r[8 + k] = hi[k]; }
    r[16] = i8_ld16(b + 192 + 2 * ib);
    r[17] = i8_ld16(b + 208);
}
__device__ __forceinline__ void i8m_iq4xs_load(I8P row, int S, int ib, uint32_t (&r)[6]) {
    I8P b = row + (size_t) S * 136;
#pragma unroll
    for (int k = 0; k < 4; ++k) r[k] = i8_ldw(b + 8 + 16 * ib + 4 * k);
    r[4] = i8_ldw(b);
    r[5] = i8_ldw(b + 4);
}

// Q3_K (block_q3_K, 110 bytes: hmask 32, qs 64, 12 scale bytes, d), sub-block ib (0..7) of super-block sb of the row:
// value l (0..31) = (the 2 bits of qs[32 (ib >> 2) + l] at 2 (ib & 3)) + 4 (bit ib of hmask[l]) - 4, times d and the
// 6-bit scale of its half less 32.  Scale 4 a + c has its low 4 bits in nibble (a >> 1) of scales[4 (a & 1) + c] and
// its high 2 bits at 2 a of scales[8 + c] (dq_q3_k above reads the same bits).  The words as they are read: 8 of
// qs, 8 of hmask, the 12 scale bytes, d.
constexpr int I8M_Q3K_RAW = 20;
__device__ __forceinline__ void i8m_q3k_load(I8P row, int sb, int ib, uint32_t (&r)[I8M_Q3K_RAW]) {
    I8P b = row + (size_t) sb * 110;
    uint32_t qs[8], hm[8], sc[3];
    i8m_ldrun<8>(b + 32 + 32 * (ib >> 2), qs);
    i8m_ldrun<8>(b, hm);
    i8m_ldrun<3>(b + 96, sc);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        r[k] = qs[k];
        r[8 + k] = hm[k];
    }
    r[16] = sc[0];
    r[17] = sc[1];
    r[18] = sc[2];
    r[19] = i8_ld16(b + 108);
}
__device__ __forceinline__ void i8m_q3k_convert(const uint32_t (&r)[I8M_Q3K_RAW], int ib, uint32_t (&q)[8], float& f0, float& f1) {
    const int sh = 2 * (ib & 3);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint32_t lo = (r[k] >> sh) & 0x03030303u, hi = ((r[8 + k] >> ib) & 0x01010101u) << 2;
        q[k] = (((lo | hi) | 0x80808080u) - 0x04040404u) ^ 0x80808080u;   // 0..7 less 4 in each byte, no borrow between bytes
    }
    const int a = ib >> 1, c = 2 * (ib & 1);
    const uint32_t lw = ((a & 1) ? r[17] : r[16]) >> (4 * (a >> 1));
    const int u0 = (int) (((lw >> (8 * c)) & 15u) | (((r[18] >> (8 * c + 2 * a)) & 3u) << 4));
    const int u1 = (int) (((lw >> (8 * c + 8)) & 15u) | (((r[18] >> (8 * c + 8 + 2 * a)) & 3u) << 4));
    const float d = i8_half(r[19]);
    f0 = d * (float) (u0 - 32);
    f1 = d * (float) (u1 - 32);
}
// Q2_K (block_q2_K, 84 bytes: 16 scale bytes, qs 64, d, dmin), sub-block ib (0..7) of super-block sb: value l (0..31) =
// the 2 bits of qs[32 (ib >> 2) + l] at 2 (ib & 3), times d * (scale byte 2 ib + l / 16 & 15), less dmin * (its >> 4).
// The words as they are read: 8 of qs, the sub-block's two scale bytes, d and dmin.
constexpr int I8M_Q2K_RAW = 10;
__device__ __forceinline__ void i8m_q2k_load(I8P row, int sb, int ib, uint32_t (&r)[I8M_Q2K_RAW]) {
    I8P b = row + (size_t) sb * 84;
    I8P qs = b + 16 + 32 * (ib >> 2);
#pragma unroll
    for (int k = 0; k < 8; ++k) r[k] = i8_ldw(qs + 4 * k);
    r[8] = i8_ld16(b + 2 * ib);
    r[9] = i8_ldw(b + 80);
}
__device__ __forceinline__ void i8m_q2k_convert(const uint32_t (&r)[I8M_Q2K_RAW], int ib, uint32_t (&q)[8], float& f0, float& f1,
                                                float& m0, float& m1) {
    const int sh = 2 * (ib & 3);
#pragma unroll
    for (int k = 0; k < 8; ++k) q[k] = (r[k] >> sh) & 0x03030303u;
    const float d = i8_half(r[9]), dmin = i8_half(r[9] >> 16);
    f0 = d * (float) (r[8] & 15u);
    f1 = d * (float) ((r[8] >> 8) & 15u);
    m0 = -dmin * (float) ((r[8] >> 4) & 15u);
    m1 = -dmin * (float) ((r[8] >> 12) & 15u);
}
// IQ1_S (block_iq1_s, 50 bytes: d, qs 32, qh 8 16-bit words), sub-block ib (0..7) of super-block sb: 32 values in 4
// groups of 8.  Group l's grid entry is iq1s_grid_gpu[qs[4 ib + l] | (bits 3 l .. 3 l + 2 of qh[ib]) << 8], 8 nibbles of
// 0..2 (value g - 1), and the sub-block's value is d (2 s + 1) (g - 1 +- 1/8), s = bits 12-14 of qh[ib], the 1/8 negative
// where bit 15 is set (IQ1S_DELTA).  As IQ1_M's (i8_expert.cuh) the delta goes into the int8 code: 8 (g - 1) +- 1, with a
// float scale d (2 s + 1) / 8, one a sub-block.  The words as they are read: qs's 4 bytes, qh[ib], d.
constexpr int I8M_IQ1S_RAW = 3;
__device__ __forceinline__ void i8m_iq1s_load(I8P row, int sb, int ib, uint32_t (&r)[I8M_IQ1S_RAW]) {
    I8P b = row + (size_t) sb * 50;
    r[0] = i8_ld32(b + 2 + 4 * ib);   // (2-byte aligned)
    r[1] = i8_ld16(b + 34 + 2 * ib);
    r[2] = i8_ld16(b);
}
__device__ __forceinline__ void i8m_iq1s_convert(const uint32_t (&r)[I8M_IQ1S_RAW], const uint32_t* grid, uint32_t (&q)[8], float& f0) {
    const uint32_t qh = r[1];
    // per byte 8 g + 121 (+1/8) or 8 g + 119 (-1/8), g 0..2: no carry between bytes; less 128 by the top bit's flip
    const uint32_t K = (qh & 0x8000u) ? 0x77777777u : 0x79797979u;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t e = grid[((r[0] >> (8 * l)) & 255u) | (((qh >> (3 * l)) & 7u) << 8)];
        q[2 * l] = (((e & 0x0F0F0F0Fu) << 3) + K) ^ 0x80808080u;           // values 8 l .. 8 l + 3: the low nibbles
        q[2 * l + 1] = ((((e >> 4) & 0x0F0F0F0Fu) << 3) + K) ^ 0x80808080u;  // 8 l + 4 .. 8 l + 7: the high ones
    }
    f0 = i8_half(r[2]) * 0.125f * (float) (2u * ((qh >> 12) & 7u) + 1u);
}
// MXFP4 (block_mxfp4, 17 bytes: an E8M0 exponent e, then 16 code bytes, byte j holding value j in its low nibble and
// value j + 16 in its high one), block 8 S + ib of the row.  A code is an E2M1 value doubled (kvalues_mxfp4: 0, 1, 2,
// 3, 4, 6, 8, 12 and their negatives), so a weight is exact as an int8 code times 2^(e - 128), the E8M0 scale halved,
// one float a block.  A block may start at any byte: the aligned words that hold it are read, the last only up to the
// block's end (nothing past a matrix is read), and joined.  The words as they are read: the 4 code words, then e.
constexpr int I8M_MXFP4_RAW = 5;
__device__ __forceinline__ void i8m_mxfp4_load(I8P row, int S, int ib, uint32_t (&r)[I8M_MXFP4_RAW]) {
    I8P b = row + (size_t) (8 * S + ib) * 17;
    const uint32_t o = (uint32_t) ((uintptr_t) b & 3u);
    I8P a = b - o;
    uint32_t w[5];
#pragma unroll
    for (int k = 0; k < 4; ++k) w[k] = i8_ldw(a + 4 * k);
    // bytes 16 .. o + 16 of the aligned run: the block's last o + 1
    w[4] = o == 3 ? i8_ldw(a + 16) : o == 2 ? i8_ld16(a + 16) | (i8_ld8(a + 18) << 16) : o == 1 ? i8_ld16(a + 16) : i8_ld8(a + 16);
#pragma unroll
    for (int k = 0; k < 4; ++k) r[k] = __funnelshift_rc(w[k], w[k + 1], 8u * (o + 1u));   // (o = 3: the word after, w[k + 1])
    r[4] = (w[0] >> (8u * o)) & 255u;
}
__device__ __forceinline__ void i8m_mxfp4_convert(const uint32_t (&r)[I8M_MXFP4_RAW], const uint32_t (&kv)[4], uint32_t (&q)[8], float& f0) {
#pragma unroll
    for (int k = 0; k < 4; ++k) i8_table16(r[k], kv, q[k], q[4 + k]);   // values 4 k .. 4 k + 3, and 16 more
    const uint32_t e = r[4];
    f0 = __uint_as_float(e < 2 ? 0x00200000u << e : (e - 1u) << 23);   // 2^(e - 128), as ggml's E8M0 halved (e 0 and 1 subnormal)
}
// NVFP4 (64 values a block, 36 bytes: 4 UE4M3 scales, then 32 code bytes), sub-block ib (0..7) of stage S of the row:
// half (ib & 1) of block 4 S + ib / 2, its 16-value parts 2 (ib & 1) and 2 (ib & 1) + 1.  Part p keeps its codes in
// bytes 8 p .. 8 p + 7: byte j holds value j (low nibble) and value j + 8 (high).  A code is an E2M1 value doubled
// (0, 1, 2, 3, 4, 6, 8, 12 and their negatives: kvalues_fp4), and the part's scale is its UE4M3 halved, so a weight is
// exact as int8 code times a float scale a 16 values.  The words as they are read: the half's 4 code words, the scales.
constexpr int I8M_NV4_RAW = 5;
__device__ __forceinline__ void i8m_nvfp4_load(I8P row, int S, int ib, uint32_t (&r)[I8M_NV4_RAW]) {
    I8P b = row + (size_t) (4 * S + (ib >> 1)) * 36;
#pragma unroll
    for (int k = 0; k < 4; ++k) r[k] = i8_ldw(b + 4 + 16 * (ib & 1) + 4 * k);
    r[4] = i8_ldw(b);
}
// UE4M3 halved, as iq_kernels.cu's ue4m3_to_f32_half (the decode kernels' and ggml's value; 0 and 0x7F give 0)
__device__ __forceinline__ float i8m_ue4m3_half(uint32_t x) {
    x &= 0x7Fu;
    if (x == 0 || x == 0x7F) return 0.0f;
    const uint32_t exp = (x >> 3) & 0xF, man = x & 0x7;
    return exp == 0 ? (float) man * 0.0009765625f : __uint_as_float(((exp + 119u) << 23) | (man << 20));
}
__device__ __forceinline__ void i8m_nvfp4_convert(const uint32_t (&r)[I8M_NV4_RAW], int ib, const uint32_t (&kv)[4], uint32_t (&q)[8],
                                                  float& f0, float& f1) {
    // per 16-value part: words 0 and 1 hold values 0-3 / 8-11 and 4-7 / 12-15 in their low / high nibbles
    i8_table16(r[0], kv, q[0], q[2]);
    i8_table16(r[1], kv, q[1], q[3]);
    i8_table16(r[2], kv, q[4], q[6]);
    i8_table16(r[3], kv, q[5], q[7]);
    const uint32_t sh = 16u * (uint32_t) (ib & 1);
    f0 = i8m_ue4m3_half(r[4] >> sh);
    f1 = i8m_ue4m3_half(r[4] >> (sh + 8));
}
template <int T> constexpr int I8M_RAW = T == 11 ? I8M_Q3K_RAW : T == 10 ? I8M_Q2K_RAW : T == 19 ? I8M_IQ1S_RAW : T == 39 ? I8M_MXFP4_RAW : T == 40 ? I8M_NV4_RAW : I8NR<T>;   // words a thread reads for a sub-block
// The sub-blocks of a stage a thread decodes, of its weight row (RT threads a row, 8 / RT sub-blocks each): r0, r0 + RT,
// ..., so that the threads of a row read their sub-blocks side by side; for Q3_K, Q6_K and IQ4_XS two that share most of
// what they read (Q3_K's code and high-bit words and scales, Q6_K's high bits and low bits: 0 and 2, 1 and 3, ...;
// IQ4_XS's scales), which is then read once: on an RTX 5090 their experts' products took 7-28% less time.  (Q4_K, Q5_K,
// Q8_0 and the i-quants were 2-6% slower so.)
template <int T, int RT>
__device__ __forceinline__ int i8m_sub(int r0, int part) {
    constexpr int PER = 8 / RT;
    if constexpr (T == 14 && PER == 2) return (r0 >> 1) * 4 + (r0 & 1) + 2 * part;
    if constexpr (T == 11 || T == 14 || T == 23) return PER * r0 + part;
    return r0 + RT * part;
}

// One block a row, a warp a block of 32 values at a time.
__global__ void __launch_bounds__(I8M_NT) i8m_quantize_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids, long long ld,
                                                             int K, long long rows, uint32_t* __restrict__ q, float* __restrict__ d,
                                                             float* __restrict__ ds) {
    const long long row = (long long) blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows) return;
    const float* src = x + (ids ? (long long) ids[row] : row) * ld;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nb = K / 32;
    for (int b = warp; b < nb; b += I8M_NT / 32) {
        const float v = src[b * 32 + lane];
        float amax = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d_inv = 127.0f / amax;
        const int code = amax == 0.0f ? 0 : (int) roundf(v * d_inv);
        int sum = code;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        uint32_t w = (uint32_t) (code & 255) << (8 * (lane & 3));
        w |= __shfl_xor_sync(0xffffffffu, w, 1);
        w |= __shfl_xor_sync(0xffffffffu, w, 2);
        if ((lane & 3) == 0) q[(row * K + b * 32) / 4 + (lane >> 2)] = w;
        if (lane == 0) {
            const float scale = amax == 0.0f ? 0.0f : 1.0f / d_inv;
            d[row * nb + b] = scale;
            ds[row * nb + b] = scale * (float) sum;
        }
    }
}

// The same numbers four values a thread (a float4: rows of whole float4s, 16-byte aligned), so that a block of 32 is
// 8 threads: three shuffles for its largest value and three for its codes' sum, where the kernel above takes five and
// five and two more to pack its codes, a warp a block at a time.
__global__ void __launch_bounds__(I8M_NT) i8m_quantize4_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids, long long ld,
                                                              int K, long long rows, uint32_t* __restrict__ q, float* __restrict__ d,
                                                              float* __restrict__ ds) {
    const long long row = (long long) blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows) return;
    const float4* src = (const float4*) (x + (ids ? (long long) ids[row] : row) * ld);
    const int n4 = K / 4, nb = K / 32;
    for (int i0 = 0; i0 < n4; i0 += I8M_NT) {   // (every lane of a warp goes round together: the shuffles need them all)
        const int i = i0 + threadIdx.x;
        const bool in = i < n4;
        const float4 v = in ? src[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d_inv = 127.0f / amax;
        const int c0 = amax == 0.0f ? 0 : (int) roundf(v.x * d_inv), c1 = amax == 0.0f ? 0 : (int) roundf(v.y * d_inv),
                  c2 = amax == 0.0f ? 0 : (int) roundf(v.z * d_inv), c3 = amax == 0.0f ? 0 : (int) roundf(v.w * d_inv);
        int sum = c0 + c1 + c2 + c3;
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (!in) continue;
        q[(row * K) / 4 + i] = (uint32_t) (c0 & 255) | ((uint32_t) (c1 & 255) << 8) | ((uint32_t) (c2 & 255) << 16) |
                               ((uint32_t) (c3 & 255) << 24);
        if ((threadIdx.x & 7) == 0) {
            const float scale = amax == 0.0f ? 0.0f : 1.0f / d_inv;
            d[row * nb + i / 8] = scale;
            ds[row * nb + i / 8] = scale * (float) sum;
        }
    }
}

template <int T, int NJ, int WR, bool RAW>
__global__ void __launch_bounds__(i8m_threads(T, NJ, WR)) i8m_kernel(const uint8_t* __restrict__ W, size_t mat_bytes, size_t row_bytes,
                                                                  int n_out, int K,
                                                    const uint8_t* __restrict__ X, const float* __restrict__ XD,
                                                    const float* __restrict__ XS, const int32_t* __restrict__ bounds, int ch_per_e,
                                                    const int32_t* __restrict__ ids, const int32_t* __restrict__ from,
                                                    const long long* __restrict__ at, float* __restrict__ Y, long long ldy) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750   // (a pass for an older card compiles this empty: the host asks i8m_arch_kernel first)
    using SM = I8mSmem<T, NJ, WR, RAW>;
    constexpr int MT = SM::MT, LDS = SM::LDS, LDB = SM::LDB, XQ = SM::XQ, XSTR = SM::XSTR;
    // NT threads.  A warp multiplies NA tiles of 16 weight rows (two where the activation rows are many: each piece of
    // them read out of shared memory goes into two products) with NJW tiles of 8 activation rows; RG warps side by
    // side down the weight rows, CG across the activation rows.
    constexpr int NT = i8m_threads(T, NJ, WR), NA = i8m_tiles_a(T, NJ, WR), RG = WR / (16 * NA), CG = NT / 32 / RG;
    constexpr int NJW = 2 * NJ / CG, NX = MT * 16 / NT;
    constexpr int RT = NT / WR;   // threads of a weight row: each decodes 8 / RT of its 8 sub-blocks a stage
    constexpr bool TWO = i8m_two(T), HAS_MIN = i8_min_type(T), MIN16 = i8m_min16(T);
    constexpr int LDM = SM::LDM, S16 = SM::S16;
    extern __shared__ uint4 i8m_dyn[];
    uint32_t* const smem = (uint32_t*) i8m_dyn;
    uint32_t* const s_grid = smem;
    uint32_t* const s_sg = smem + SM::o_sg;
    uint32_t* const s_w = smem + SM::o_w;                    // [WR][I8M_LD]: a stage of the weight rows' codes
    uint32_t* const s_x = smem + SM::o_x;                    // [MT][I8M_LD]: a stage of the activation rows' codes
    float* const s_ws = (float*) (smem + SM::o_ws);          // [WR][LDS]: scale of (weight row, sub-block or half of one)
    float* const s_wm = (float*) (smem + SM::o_wm);          // [WR][LDB]: minimum of (weight row, sub-block)
    float* const s_xq = (float*) (smem + SM::o_xq);          // [8][XSTR]: the activation rows' scales (I8mSmem)
    float* const s_m16 = s_wm + SM::MN * WR * LDB;           // Q2_K: [WR][LDM]: minimum of (weight row, 16 values)
    float* const s_x16 = (float*) (smem + SM::o_x16);        // Q2_K: [16][S16]: scale * sum of 16 codes, by activation row
    int* const s_id = (int*) (smem + SM::o_id);              // [MT]: the row of Y, -1 past the last
    uint32_t* const s_raw = smem + SM::o_raw;                // RAW: [WR][RW]: a stage of the weight rows as stored

    const int e = blockIdx.y / ch_per_e, ch = blockIdx.y % ch_per_e;
    const int c0 = bounds[e] + ch * MT, r1 = bounds[e + 1];
    if (c0 >= r1) return;
    const int ntok = min(MT, r1 - c0);
    const int n0 = blockIdx.x * WR;
    const int tid = threadIdx.x;
    const uint8_t* const mat = W + (at ? (size_t) at[e] : (size_t) e * mat_bytes);   // matrix e

    const uint32_t* grid_rd = i8_grid_src<T>();
    if constexpr (SM::GW > 0) {
        for (int i = tid; i < i8_grid_words(T); i += NT) s_grid[i] = grid_rd[i];
        grid_rd = s_grid;
    }
    if (i8_sign_type(T) && tid < 256) {   // entry tid: 8 sign bits (the ksigns types: 7 bits and their parity) as byte masks
        const uint32_t b = (uint32_t) tid;
        uint32_t sb = b;
        if constexpr (T != 21 && T != 22) sb = (b & 127u) | (((uint32_t) __popc(b & 127u) & 1u) << 7);
        uint32_t mlo = 0, mhi = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            mlo |= ((sb >> k) & 1u) ? (0xFFu << (8 * k)) : 0u;
            mhi |= ((sb >> (4 + k)) & 1u) ? (0xFFu << (8 * k)) : 0u;
        }
        *(uint2*) (s_sg + 2 * b) = make_uint2(mlo, mhi);
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (T == 20 || T == 23) {
        const uint32_t* t = (const uint32_t*) kvalues_iq4nl;
#pragma unroll
        for (int i = 0; i < 4; ++i) kv[i] = t[i];
    } else if constexpr (T == 2) {   // Q4_0: a nibble less 8, IQ4_NL's blocks otherwise
        kv[0] = 0xFBFAF9F8u; kv[1] = 0xFFFEFDFCu; kv[2] = 0x03020100u; kv[3] = 0x07060504u;
    } else if constexpr (T == 39 || T == 40) {   // MXFP4 (kvalues_mxfp4) and NVFP4 (kvalues_fp4): 0 1 2 3 4 6 8 12, then 0 -1 -2 -3 -4 -6 -8 -12
        kv[0] = 0x03020100u; kv[1] = 0x0C080604u; kv[2] = 0xFDFEFF00u; kv[3] = 0xF4F8FAFCu;
    }
    if (tid < MT) s_id[tid] = tid < ntok ? (ids ? ids[c0 + tid] : c0 + tid) : -1;

    // this thread's weight sub-blocks: row wr of the tile, its sub-blocks r0, r0 + RT, ... of a stage's 8
    const int wr = tid / RT, r0 = tid % RT;
    const bool w_in = n0 + wr < n_out;
    I8P const blob = I8_GPTR(mat + (size_t) n0 * row_bytes);
    const uint32_t row_off = (uint32_t) ((size_t) wr * row_bytes);
    // its tensor-core tiles: weight rows i0 + 16 a + g and 8 more, activation rows j0 + g as columns, j0 = 8 (CG n + jp)
    const int warp = tid >> 5, g = (tid & 31) >> 2, tg = tid & 3;
    const int i0 = (warp % RG) * 16 * NA, jp = warp / RG;
    const float* const sa = s_ws + (i0 + g) * LDS;
    const float* const sm = s_wm + (i0 + g) * LDB;

    float acc[NA][NJW][4];
#pragma unroll
    for (int a = 0; a < NA; ++a)
#pragma unroll
        for (int n = 0; n < NJW; ++n)
#pragma unroll
            for (int l = 0; l < 4; ++l) acc[a][n][l] = 0.0f;

    // A stage as it is read: this thread's sub-blocks of its weight row, its 16-byte pieces of the activation rows
    // and its activation scales.
    constexpr int PER = 8 / RT, NSC = (MT * 8 + NT - 1) / NT;
    uint32_t wraw[PER][I8M_RAW<T>];
    uint4 xr[NX];   // piece n: activation row (NT / 16) n + tid / 16
    float xdr[NSC], xsr[NSC];
    float x16[MIN16 ? NX : 1];   // Q2_K: the scale of the 32-value block of this thread's 16 codes of each row
    // the rows of X this thread reads: the launch's row, or the one `from` gives for it
    int xrow[NX], srow[NSC];
#pragma unroll
    for (int n = 0; n < NX; ++n) {
        const int tok = NT / 16 * n + (tid >> 4);
        xrow[n] = tok < ntok ? (from ? from[c0 + tok] : c0 + tok) : 0;
    }
#pragma unroll
    for (int c = 0; c < NSC; ++c) {
        const int i = tid + c * NT, tok = i >> 3;
        srow[c] = i < MT * 8 && tok < ntok ? (from ? from[c0 + tok] : c0 + tok) : 0;
    }
    // A row of 32-value blocks may end inside its last stage (Gemma 4 26B's down: 704 values): the sub-blocks past
    // its end are not read, and go in as zeros, codes and scales, so they add nothing.
    const auto here = [&](int S, int ib) { return S * 256 + ib * 32 < K; };
    const auto fetch = [&](int S) {
        if constexpr (RAW) {   // the pieces of each row's stage, 16 bytes at a time: consecutive threads, consecutive pieces
            constexpr int RB = i8m_raw_bytes(T), NP = SM::RW / 4;
            if constexpr (RB % 16 == 0) {
                const uint8_t* const src = mat + (size_t) n0 * row_bytes + (size_t) S * RB;
                for (int i = tid; i < WR * NP; i += NT) {
                    const int r = i / NP, c = i - r * NP;
                    const bool ok = n0 + r < n_out;
                    i8m_cp16n(s_raw + r * SM::RW + 4 * c, ok ? src + (size_t) r * row_bytes + 16 * c : W, ok ? 16 : 0);
                }
                i8m_cp_commit();
            } else {
                const uint8_t* const end = mat + (size_t) n_out * row_bytes;   // (a piece past it is read only up to it)
                for (int i = tid; i < WR * NP; i += NT) {
                    const int r = i / NP, c = i - r * NP;
                    const uint8_t* const at = mat + (size_t) (n0 + r) * row_bytes + (size_t) S * RB;
                    const uint8_t* const pc = (const uint8_t*) ((uintptr_t) at & ~(uintptr_t) 15) + 16 * c;
                    const bool ok = n0 + r < n_out && pc < at + RB;
                    i8m_cp16n(s_raw + r * SM::RW + 4 * c, ok ? pc : W, !ok ? 0 : end - pc < 16 ? (int) (end - pc) : 16);
                }
                i8m_cp_commit();
            }
        } else if (w_in) {
#pragma unroll
            for (int part = 0; part < PER; ++part) {
                const int ib = i8m_sub<T, RT>(r0, part);
                if (!here(S, ib)) continue;
                if constexpr (T == 11) i8m_q3k_load(blob + row_off, S, ib, wraw[part]);
                else if constexpr (T == 10) i8m_q2k_load(blob + row_off, S, ib, wraw[part]);
                else if constexpr (T == 8) i8m_q8_load(blob + row_off, S, ib, wraw[part]);
                else if constexpr (T == 14) i8m_q6k_load(blob + row_off, S, ib, wraw[part]);
                else if constexpr (T == 23) i8m_iq4xs_load(blob + row_off, S, ib, wraw[part]);
                else if constexpr (T == 19) i8m_iq1s_load(blob + row_off, S, ib, wraw[part]);
                else if constexpr (T == 39) i8m_mxfp4_load(blob + row_off, S, ib, wraw[part]);
                else if constexpr (T == 40) i8m_nvfp4_load(blob + row_off, S, ib, wraw[part]);
                else i8_load<T>(blob, i8_offs<T>(row_off, ib & 3), 2 * S + (ib >> 2), wraw[part]);
            }
        }
#pragma unroll
        for (int n = 0; n < NX; ++n) {   // 16 bytes of activation row (NT / 16) n + tid / 16
            const int tok = NT / 16 * n + (tid >> 4);
            xr[n] = make_uint4(0, 0, 0, 0);
            if (tok < ntok && S * 256 + (tid & 15) * 16 < K)
                xr[n] = *(const uint4*) (X + (size_t) xrow[n] * K + (size_t) S * 256 + (tid & 15) * 16);
            if constexpr (MIN16) x16[n] = tok < ntok ? XD[(size_t) xrow[n] * (K / 32) + (size_t) S * 8 + ((tid & 15) >> 1)] : 0.0f;
        }
#pragma unroll
        for (int c = 0; c < NSC; ++c) {
            const int i = tid + c * NT, tok = i >> 3;
            const size_t at = (size_t) srow[c] * (K / 32) + (size_t) S * 8 + (i & 7);
            xdr[c] = xsr[c] = 0.0f;
            if (i < MT * 8 && tok < ntok && here(S, i & 7)) {
                xdr[c] = XD[at];
                if constexpr (HAS_MIN) xsr[c] = XS[at];
            }
        }
    };
    const auto put = [&](int S) {
        if constexpr (RAW) {   // this thread's sub-blocks out of the raw stage, read as fetch() reads the card's memory
            const uintptr_t at = (uintptr_t) (mat + (size_t) (n0 + wr) * row_bytes + (size_t) S * i8m_raw_bytes(T));
            I8P const rb = I8_GPTR((const uint8_t*) (s_raw + wr * SM::RW) + (at & 15) - (size_t) S * i8m_raw_bytes(T));
#pragma unroll
            for (int part = 0; part < PER; ++part) {
                const int ib = i8m_sub<T, RT>(r0, part);
                if (!w_in) continue;
                if constexpr (T == 11) i8m_q3k_load(rb, S, ib, wraw[part]);
                else if constexpr (T == 10) i8m_q2k_load(rb, S, ib, wraw[part]);
                else if constexpr (T == 8) i8m_q8_load(rb, S, ib, wraw[part]);
                else if constexpr (T == 14) i8m_q6k_load(rb, S, ib, wraw[part]);
                else if constexpr (T == 23) i8m_iq4xs_load(rb, S, ib, wraw[part]);
                else i8_load<T>(rb, i8_offs<T>(0, ib & 3), 2 * S + (ib >> 2), wraw[part]);
            }
        }
#pragma unroll
        for (int part = 0; part < PER; ++part) {
            const int ib = i8m_sub<T, RT>(r0, part);
            uint32_t q[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            float f0 = 0.0f, f1 = 0.0f, mn = 0.0f, mn1 = 0.0f;
            if (w_in && here(S, ib)) {
                if constexpr (T == 11) {
                    i8m_q3k_convert(wraw[part], ib, q, f0, f1);
                } else if constexpr (T == 10) {
                    i8m_q2k_convert(wraw[part], ib, q, f0, f1, mn, mn1);
                } else if constexpr (T == 19) {
                    i8m_iq1s_convert(wraw[part], grid_rd, q, f0);
                } else if constexpr (T == 39) {
                    i8m_mxfp4_convert(wraw[part], kv, q, f0);
                } else if constexpr (T == 40) {
                    i8m_nvfp4_convert(wraw[part], ib, kv, q, f0, f1);
                } else {
                    uint32_t w[I8NW<T>];
                    int m0, m1;
                    float base;
                    i8_pack<T>(wraw[part], ib, w);
                    i8_convert<T>(w, grid_rd, (const uint2*) s_sg, kv, q, m0, m1, base, ib, &mn);
                    f0 = base * (float) m0;
                    f1 = base * (float) m1;
                }
            }
            uint4* dst = (uint4*) (s_w + wr * I8M_LD + ib * 8);
            dst[0] = make_uint4(q[0], q[1], q[2], q[3]);
            dst[1] = make_uint4(q[4], q[5], q[6], q[7]);
            if constexpr (TWO) {
                s_ws[wr * LDS + 2 * ib] = f0;
                s_ws[wr * LDS + 2 * ib + 1] = f1;
            } else {
                s_ws[wr * LDS + (SM::PK ? 2 * ib : ib)] = f0;
            }
            if constexpr (SM::PK) s_ws[wr * LDS + 2 * ib + 1] = mn;
            else if constexpr (HAS_MIN) s_wm[wr * LDB + ib] = mn;
            if constexpr (MIN16) {
                s_m16[wr * LDM + 2 * ib] = mn;
                s_m16[wr * LDM + 2 * ib + 1] = mn1;
            }
        }
#pragma unroll
        for (int n = 0; n < NX; ++n) {
            *(uint4*) (s_x + (NT / 16 * n + (tid >> 4)) * I8M_LD + (tid & 15) * 4) = xr[n];
            if constexpr (MIN16) {   // the sum of the thread's 16 codes, times their block's scale
                const int sum = __dp4a((int) xr[n].w, 0x01010101, __dp4a((int) xr[n].z, 0x01010101,
                                       __dp4a((int) xr[n].y, 0x01010101, __dp4a((int) xr[n].x, 0x01010101, 0))));
                s_x16[(tid & 15) * S16 + NT / 16 * n + (tid >> 4)] = x16[n] * (float) sum;
            }
        }
#pragma unroll
        for (int c = 0; c < NSC; ++c) {
            const int i = tid + c * NT;
            if (i < MT * 8) {
                const int tok = i >> 3;
                float* const p = s_xq + (i & 7) * XSTR + (tok >> 1) * XQ + (tok & 1);
                p[0] = xdr[c];
                if constexpr (HAS_MIN) p[2] = xsr[c];
            }
        }
    };

    // The products of a stage.  A warp's operands come out of shared memory by ldmatrix, four 8 x 16-byte pieces an
    // instruction: its two tiles of 16 weight rows (NA = 2) for two 32-value blocks, and an 8-row tile of activations
    // for the same two, which then goes into the products of both weight tiles; a thread's two activation rows' scales
    // are one 8- or 16-byte read.  Each sum takes its terms in the order of the blocks, so a product does not depend
    // on how its launch is cut.
    const int lane = tid & 31;
    const uint32_t* const a_row = s_w + (i0 + ((lane >> 3) & 1) * 8 + (lane & 7)) * I8M_LD + (lane >> 4) * 4;
    const uint32_t* const b_row = s_x + (8 * jp + (lane & 7)) * I8M_LD + (lane >> 4) * 8 + ((lane >> 3) & 1) * 4;
    const auto scales = [&](int kb, int j0, float& x0, float& x1, float& s0, float& s1) {
        const float* const p = s_xq + kb * XSTR + (j0 / 2 + tg) * XQ;
        if constexpr (HAS_MIN) {
            const float4 v = *(const float4*) p;
            x0 = v.x; x1 = v.y; s0 = v.z; s1 = v.w;
        } else {
            const float2 v = *(const float2*) p;
            x0 = v.x; x1 = v.y; s0 = s1 = 0.0f;
        }
    };
    const auto multiply = [&]() {
        if constexpr (TWO) {
#pragma unroll
            for (int kq = 0; kq < 16; kq += 4) {
                uint32_t A[2][NA][4];   // [kq / 2 pair][tile]: {a0, a1} of kq, kq + 1 (or kq + 2, kq + 3)
                float w0[4][NA], w1[4][NA], m0[4][NA], m1[4][NA];
#pragma unroll
                for (int a = 0; a < NA; ++a) {
                    i8m_ldsm4(A[0][a], a_row + 16 * a * I8M_LD + kq * 4);
                    i8m_ldsm4(A[1][a], a_row + 16 * a * I8M_LD + (kq + 2) * 4);
#pragma unroll
                    for (int h = 0; h < 4; ++h) {
                        w0[h][a] = sa[16 * a * LDS + kq + h];
                        w1[h][a] = sa[(16 * a + 8) * LDS + kq + h];
                        m0[h][a] = m1[h][a] = 0.0f;
                        if constexpr (MIN16) {
                            m0[h][a] = s_m16[(i0 + 16 * a + g) * LDM + kq + h];
                            m1[h][a] = s_m16[(i0 + 16 * a + g + 8) * LDM + kq + h];
                        }
                    }
                }
#pragma unroll
                for (int n = 0; n < NJW; ++n) {
                    const int j0 = 8 * (CG * n + jp);
                    if (j0 >= ntok) break;
                    uint32_t B[4];   // b0 of kq .. kq + 3
                    i8m_ldsm4(B, b_row + 8 * CG * n * I8M_LD + kq * 4);
                    float x0[2], x1[2], s0, s1;
                    scales(kq >> 1, j0, x0[0], x1[0], s0, s1);
                    scales((kq >> 1) + 1, j0, x0[1], x1[1], s0, s1);
#pragma unroll
                    for (int h = 0; h < 4; ++h) {
                        float2 sm16 = make_float2(0.0f, 0.0f);
                        if constexpr (MIN16) sm16 = *(const float2*) (s_x16 + (kq + h) * S16 + j0 + 2 * tg);
#pragma unroll
                        for (int a = 0; a < NA; ++a) {
                            int c[4] = {0, 0, 0, 0};
                            i8m_mma16(c, A[h >> 1][a][2 * (h & 1)], A[h >> 1][a][2 * (h & 1) + 1], B[h]);
                            const float xa = x0[h >> 1], xb = x1[h >> 1];
                            acc[a][n][0] = fmaf((float) c[0], w0[h][a] * xa, acc[a][n][0]);
                            acc[a][n][1] = fmaf((float) c[1], w0[h][a] * xb, acc[a][n][1]);
                            acc[a][n][2] = fmaf((float) c[2], w1[h][a] * xa, acc[a][n][2]);
                            acc[a][n][3] = fmaf((float) c[3], w1[h][a] * xb, acc[a][n][3]);
                            if constexpr (MIN16) {
                                acc[a][n][0] = fmaf(m0[h][a], sm16.x, acc[a][n][0]);
                                acc[a][n][1] = fmaf(m0[h][a], sm16.y, acc[a][n][1]);
                                acc[a][n][2] = fmaf(m1[h][a], sm16.x, acc[a][n][2]);
                                acc[a][n][3] = fmaf(m1[h][a], sm16.y, acc[a][n][3]);
                            }
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (int kb = 0; kb < 8; kb += 2) {
                uint32_t A[2][NA][4];   // [block kb + h][tile]: a0 .. a3
                float w0[2][NA], w1[2][NA], m0[2][NA], m1[2][NA];
                if constexpr (SM::PK) {
#pragma unroll
                    for (int a = 0; a < NA; ++a) {
                        const float4 v0 = *(const float4*) (sa + 16 * a * LDS + 2 * kb);
                        const float4 v1 = *(const float4*) (sa + (16 * a + 8) * LDS + 2 * kb);
                        w0[0][a] = v0.x; m0[0][a] = v0.y; w0[1][a] = v0.z; m0[1][a] = v0.w;
                        w1[0][a] = v1.x; m1[0][a] = v1.y; w1[1][a] = v1.z; m1[1][a] = v1.w;
                    }
                }
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int a = 0; a < NA; ++a) {
                        i8m_ldsm4(A[h][a], a_row + 16 * a * I8M_LD + (kb + h) * 8);
                        if constexpr (!SM::PK) {
                            m0[h][a] = m1[h][a] = 0.0f;
                            w0[h][a] = sa[16 * a * LDS + kb + h];
                            w1[h][a] = sa[(16 * a + 8) * LDS + kb + h];
                            if constexpr (HAS_MIN) {
                                m0[h][a] = sm[16 * a * LDB + kb + h];
                                m1[h][a] = sm[(16 * a + 8) * LDB + kb + h];
                            }
                        }
                    }
#pragma unroll
                for (int n = 0; n < NJW; ++n) {
                    const int j0 = 8 * (CG * n + jp);
                    if (j0 >= ntok) break;
                    uint32_t B[4];   // b0, b1 of block kb, then of kb + 1
                    i8m_ldsm4(B, b_row + 8 * CG * n * I8M_LD + kb * 8);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float x0, x1, s0, s1;
                        scales(kb + h, j0, x0, x1, s0, s1);
#pragma unroll
                        for (int a = 0; a < NA; ++a) {
                            int c[4] = {0, 0, 0, 0};
                            i8m_mma32(c, A[h][a][0], A[h][a][1], A[h][a][2], A[h][a][3], B[2 * h], B[2 * h + 1]);
                            acc[a][n][0] = fmaf((float) c[0], w0[h][a] * x0, acc[a][n][0]);
                            acc[a][n][1] = fmaf((float) c[1], w0[h][a] * x1, acc[a][n][1]);
                            acc[a][n][2] = fmaf((float) c[2], w1[h][a] * x0, acc[a][n][2]);
                            acc[a][n][3] = fmaf((float) c[3], w1[h][a] * x1, acc[a][n][3]);
                            if constexpr (HAS_MIN) {
                                acc[a][n][0] = fmaf(m0[h][a], s0, acc[a][n][0]);
                                acc[a][n][1] = fmaf(m0[h][a], s1, acc[a][n][1]);
                                acc[a][n][2] = fmaf(m1[h][a], s0, acc[a][n][2]);
                                acc[a][n][3] = fmaf(m1[h][a], s1, acc[a][n][3]);
                            }
                        }
                    }
                }
            }
        }
    };

    const int stages = (K + 255) / 256;
    fetch(0);
    for (int S = 0; S < stages; ++S) {
        if constexpr (RAW) i8m_cp_wait();   // this thread's pieces of the raw stage (fetch() after put() reads it: one copy)
        __syncthreads();   // the stage before is multiplied (and the tables are written)
        put(S);
        __syncthreads();
        if (S + 1 < stages) fetch(S + 1);   // read while this stage is multiplied
        multiply();
    }

    // a thread holds weight rows i0 + 16 a + g and 8 more for activation rows j0 + 2 tg and j0 + 2 tg + 1
#pragma unroll
    for (int n = 0; n < NJW; ++n) {
        const int j0 = 8 * (CG * n + jp);
        if (j0 >= ntok) break;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            const int id = s_id[j0 + 2 * tg + l];
            if (id < 0) continue;
            float* out = Y + (size_t) id * (size_t) ldy;
#pragma unroll
            for (int a = 0; a < NA; ++a) {
                const int ia = n0 + i0 + 16 * a + g;
                if (ia < n_out) out[ia] = acc[a][n][l];
                if (ia + 8 < n_out) out[ia + 8] = acc[a][n][2 + l];
            }
        }
    }
#endif
}

// the architecture the code running on this card was compiled for (a build may hold no pass for the card's own)
__global__ void i8m_arch_kernel(int* out) {
#if defined(__CUDA_ARCH__)
    *out = __CUDA_ARCH__;
#else
    *out = 0;
#endif
}

struct I8mCard {
    bool read = false, on = false;
    int arch = 0;                       // the architecture of the code it runs (i8m_arch_kernel)
    size_t shared = 0, shared_sm = 0;   // a block's most, a multiprocessor's
};

// gemm_iq_i8_turing(): the kernel also on Turing's code (sm_75), whose products a model asks for
std::atomic<bool> i8m_turing{false};

// the current card: int8 tensor cores in the code it runs (Ampere's, or Turing's when asked), and the shared memory
// it gives a block
I8mCard i8m_card() {
    static I8mCard known[64];
    static std::mutex mu;
    static const bool off = [] { const char* v = std::getenv("STRATA_PF_I8MMA"); return v != nullptr && v[0] == '0'; }();
    int dev = 0;
    if (off || cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return I8mCard{};
    std::lock_guard<std::mutex> lock(mu);
    if (!known[dev].read) {
        I8mCard c;
        c.read = true;
        int v = 0, *d_arch = nullptr, arch = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) == cudaSuccess) c.shared = (size_t) v;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev) == cudaSuccess) c.shared_sm = (size_t) v;
        if (cudaMalloc(&d_arch, sizeof(int)) == cudaSuccess) {
            i8m_arch_kernel<<<1, 1>>>(d_arch);
            if (cudaMemcpy(&arch, d_arch, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) arch = 0;
            cudaFree(d_arch);
        }
        (void) cudaGetLastError();
        c.arch = arch;
        known[dev] = c;
    }
    I8mCard c = known[dev];
    c.on = c.arch >= 800 || (c.arch >= 750 && i8m_turing.load(std::memory_order_relaxed));
    return c;
}

template <int T, int NJ, int WR, bool RAW>
bool i8m_launch(int dev, dim3 grid, cudaStream_t s, const uint8_t* W, size_t mat_bytes, size_t row_bytes, int n_out, int K,
                const uint8_t* X, const float* XD, const float* XS, const int32_t* bounds, int ch, const int32_t* ids,
                const int32_t* from, const long long* at, float* Y, long long ldy) {
    constexpr size_t bytes = I8mSmem<T, NJ, WR, RAW>::bytes;
    static bool attr[64];   // the attribute belongs to a card (see gemm_iq_f16_grouped)
    if (!attr[dev]) {
        cudaFuncSetAttribute(i8m_kernel<T, NJ, WR, RAW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes);
        attr[dev] = true;
    }
    i8m_kernel<T, NJ, WR, RAW><<<grid, i8m_threads(T, NJ, WR), bytes, s>>>(W, mat_bytes, row_bytes, n_out, K, X, XD, XS, bounds, ch,
                                                                       ids, from, at, Y, ldy);
    return cudaGetLastError() == cudaSuccess;
}

// 64 weight rows a block: 128 measured slower on an RTX PRO 6000 (a dense Q8_0 [2048 -> 8192] times 2,048 rows 0.61 ms
// against 0.46, the experts no faster), so only 64 is built
#define STRATA_I8M_TILES(X) X(1, 64) X(2, 64) X(4, 64) X(8, 64) X(8, 128)

template <int T>
size_t i8m_tile_bytes(int nj, int wr, bool raw) {
#define STRATA_I8M_SIZE(NJ_, WR_) \
    if (nj == NJ_ && wr == WR_) return raw ? I8mSmem<T, NJ_, WR_, true>::bytes : I8mSmem<T, NJ_, WR_>::bytes;
    STRATA_I8M_TILES(STRATA_I8M_SIZE)
#undef STRATA_I8M_SIZE
    return (size_t) -1;
}

// the tile as wide in activation rows as the launch's rows need, of what the card has the shared memory for
template <int T>
bool i8m_launch_type(const I8mCard& card, int dev, cudaStream_t s, const uint8_t* W, size_t mat_bytes, size_t row_bytes, int n_out, int K,
                     const uint8_t* X, const float* XD, const float* XS, const int32_t* bounds, int n_mats, int max_rows,
                     const int32_t* ids, const int32_t* from, const long long* at, float* Y, long long ldy) {
    int nj = max_rows <= 16 ? 1 : max_rows <= 32 ? 2 : max_rows <= 64 ? 4 : 8;
    const int wr = i8m_rows(T, nj) == 128 && n_out >= 128 && i8m_tile_bytes<T>(8, 128, false) <= card.shared ? 128 : 64;
    // The raw stage: whole stages, no piece before the matrices' start, rows of whole pieces for a type read so; and as
    // many blocks a multiprocessor as without it (its shared memory took Qwen3.6's experts, 64 rows of activations, from
    // two blocks to one: 6% slower).  (The blocks a multiprocessor by shared memory, 1 KB of it the system's a block.)
    static const bool raw_on = [] { const char* v = std::getenv("STRATA_I8M_RAW"); return v == nullptr || v[0] != '0'; }();
    const auto per_sm = [&](size_t b) { return card.shared_sm / (b + 1024); };
    // (Turing has no cp.async to copy it with)
    const bool raw = raw_on && card.arch >= 800 && i8m_raw_type(T) && K % 256 == 0 && (uintptr_t) W % 16 == 0 &&
                     (i8m_raw_bytes(T) % 16 != 0 || (row_bytes % 16 == 0 && mat_bytes % 16 == 0)) &&
                     i8m_tile_bytes<T>(nj, wr, true) <= card.shared &&
                     per_sm(i8m_tile_bytes<T>(nj, wr, true)) >= per_sm(i8m_tile_bytes<T>(nj, wr, false));
    while (nj > 1 && i8m_tile_bytes<T>(nj, wr, raw) > card.shared) nj /= 2;
    if (i8m_tile_bytes<T>(nj, wr, raw) > card.shared) return false;
    const long long ch = (max_rows + 16 * nj - 1) / (16 * nj), gy = (long long) n_mats * ch;
    if (gy > 65535) return false;
    const dim3 grid((unsigned) ((n_out + wr - 1) / wr), (unsigned) gy);
#define STRATA_I8M_GO(NJ_, WR_) \
    if (nj == NJ_ && wr == WR_) \
        return raw ? i8m_launch<T, NJ_, WR_, i8m_raw_type(T)>(dev, grid, s, W, mat_bytes, row_bytes, n_out, K, X, XD, XS, \
                                                                     bounds, (int) ch, ids, from, at, Y, ldy) \
                   : i8m_launch<T, NJ_, WR_, false>(dev, grid, s, W, mat_bytes, row_bytes, n_out, K, X, XD, XS, bounds, (int) ch, \
                                                    ids, from, at, Y, ldy);
    STRATA_I8M_TILES(STRATA_I8M_GO)
#undef STRATA_I8M_GO
    return false;
}
