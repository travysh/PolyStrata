// src/poly/inkling2.hpp - Thinking Machines' Inkling 2 with a looped core (llama.cpp architecture "inkling2_loop"):
// Inkling 2 Mini.  Unsloth's top-level files (architecture "inkling2") are the same weights under other names; the
// engine reads them through tensor_alias(), meta_alias() and msg_query() (inkling2.cpp).
//
// 24 layers of weights make 48 layer applications: the layers before the core run once, the core's run
// loop.count times, each pass from the output of the layers before it, and the layers after it once.  A layer of a
// later pass mixes messages from the pass before into its inputs through small gated controllers.  Every application
// has attention rows and convolution state of its own (a "slot"); the routed experts are the layer's, whatever the
// pass.
//
// A layer: attention with no rotary embedding (a learned bias by distance and a sink term in the softmax), short
// causal convolutions over K, V, the attention's output and the FFN's, and routed experts with the SiTU activation.
// Every block_size layers start a new residual stream; a layer's inputs are a learned mix of the streams so far.
// Most layers attend a window of the last tokens, a few all of them.
#pragma once

#include "engine.hpp"
#include "inkling2_kernels.hpp"
#include "inkling2_zha.hpp"

#include <array>
#include <functional>
#include <utility>
#include <vector>

namespace strata::poly {

class Inkling2 final : public Engine {
public:
    ~Inkling2() override;
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;
    // pictures (inkling2_pictures.cpp): a picture's rows go in a call of their own, and skip the token norm
    bool sees() const override { return true; }
    int64_t whole(int64_t at, int64_t m) const override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    const Names* tensor_alias(const char* name) const override;
    const Names* meta_alias(const char* key) const override;
    void begin_pass() override;
    size_t recurrent_floats() const override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    size_t extra_bytes() const override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    void alloc_caches() override;
    void assign_scratch() override;
    // A call of a few tokens keeps a scratch of its own, and a prompt batch's is lent to the expert cache between
    // prompts (Engine::use_scratch()).  Not with the drafter, whose features are a whole call's rows, nor on ROCm on
    // Linux, nor where nothing would be lent: the experts frozen, or every one of them in VRAM (inkling2.cpp).
    bool scratch_layouts() const override;
    bool scores_apart() const;   ///< the attention's scores outside the scratch, for its two layouts to share
    bool every_expert_fits() const;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    int next_layer(int il) override;
    // a layer is some 20 launches: the card has 4 layers' work while the host launches the other 44
    int whole_first_layers() const override { return 4; }
    // its kernels in inkling2.cu wait for the launch before them in pdl_wait()
    bool whole_pdl() const override { return true; }
    int head_key(int il, int n) const override;
    int tail_key(int il, int n) const override;
    // step()'s rows in one head: the products, mixes, norms, controllers and router for all of them, and what reads or
    // writes a sequence's convolution states, attention caches or position a row at a time, the row's own sequence's
    // (run_slot()).  The tail's convolution of the experts' sum is a row's own state too.
    bool batch_heads() const override { return true; }
    bool tail_by_rows() const override { return true; }
    bool reads_by_layer() const override { return false; }
    // A call's residual streams and the core's bank from one layer run to the next: read() takes the calls through
    // the runs a group at a time.  Not with the drafter, whose taps of three runs are a call's too.
    int carry_rows() const override { return ds_file_ ? 0 : kBlocks + 2; }
    void carry_at(float* at) override;
    float* carried(int i) override;
    void save_run(int il, int64_t pos, uint8_t* dst) override;
    void output_rows(int first, int rows, float* h) override;
    void finish_logits(float* logits, int rows) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    void route_cached(const Layer& L, int il, const float* logits, int32_t token, int* ids, float* w,
                      float* tmp) override;
    void slots_moved() override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;

private:
    static constexpr int kCtrl = 4;     ///< a looped layer's controllers: Q, K, V and the FFN
    static constexpr int kBlocks = 4;   ///< the most residual streams before the current one

    /// A loop controller: [normed input | normed message] down to the gate's rank, and its gain and forget gates.
    struct Ctrl {
        Dev down, gain, forget;
    };
    /// A layer's weights beside the ones every architecture has (Layer).
    struct Block {
        bool swa = false;   ///< attends the window; else every token so far
        int extent = 0;     ///< the distances its bias table has
        Dev wq, wk, wv, wo, gate, rel, sink;
        Dev q_clip, rel_bias, rel_clip;
        Dev k_conv, v_conv, a_conv, f_conv;
        Dev attn_norm;                       ///< a window layer's
        Dev q_norm, k_norm, v_norm;          ///< a full layer's
        Dev k_dyn, v_dyn;                    ///< a full layer's convolution taps by the input
        Dev attn_query, ffn_query;           ///< the mixes over the residual streams
        Dev ffn_norm;
        // a core layer's
        Dev msg_query;
        Ctrl ctrl[kCtrl];
        // routed experts: the bias their pick adds to the gates
        std::vector<float> probs_b;
        Dev probs_b_dev;
    };
    /// What one application of a layer keeps of the sequence.
    struct Slot {
        int il = 0;
        uint16_t* keys = nullptr;   ///< fp16, n_kv * head_dim a row: a row a position, or a window layer's ring
        uint16_t* vals = nullptr;
        float* past = nullptr;      ///< the convolutions' last inputs: K, V, the attention's output, the FFN's
    };

    int slot_of(int il, int pass) const {
        return pass == 0 ? il : n_layer_ + (pass - 1) * (core_end_ - core_start_) + (il - core_start_);
    }
    /// The pass layer il's application under way is of: the core's pass_, and 0 outside the core.
    int pass_of(int il) const { return il >= core_start_ && il < core_end_ ? pass_ : 0; }
    size_t past_floats() const { return (size_t) (taps_ - 1) * (2 * (size_t) n_kv_ * head_dim_ + 2 * (size_t) n_embd_); }
    float* past_k(const Slot& s) const { return s.past; }
    float* past_v(const Slot& s) const { return s.past + (size_t) (taps_ - 1) * n_kv_ * head_dim_; }
    float* past_a(const Slot& s) const { return s.past + (size_t) (taps_ - 1) * 2 * n_kv_ * head_dim_; }
    float* past_f(const Slot& s) const { return past_a(s) + (size_t) (taps_ - 1) * n_embd_; }
    /// Where a window's first snapshot of the state at `past` goes (snap_at_ at its place in state_), or null.
    float* snap_of(const float* past) const { return snap_bits_ ? snap_at_ + (past - state_) : nullptr; }
    InkStates streams_of(int blocks, size_t at) const;
    /// The slot of layer il's application under way: row i's own sequence's among step()'s rows (rows_of_), else the
    /// bound sequence's.
    Slot run_slot(int il, int i) const;
    /// Launches on side_ from here run beside the stream's, after what the stream has so far.
    void fork();
    /// The stream waits for what side_ has.
    void join();
    void control(const float* cur, const float* msg, const Ctrl& c, float* out, int n);
    bool control3(const float* const* cur, float* const* msg, const Ctrl* c, float* const* out, int n);
    bool project(const Block& B, const float* const* x, int n, bool quantized = false, bool* extras = nullptr);
    bool control_ffn(const float* msg, const Ctrl& c, const Block& B, int row = 0);
    void guess_block(int32_t token) override;
    bool own_blocks() const override { return ds_file_ != nullptr; }
    // only where moe_card() and layer_tail() are recorded together (whole_rec_), and not with the doorbell, whose
    // experts' sum comes in after the card's
    bool moe_sum_later(int n) const override { return n == 1 && whole_rec_ && !bell_; }
    bool bells() const override { return true; }
#if defined(STRATA_USE_HIP)
    // the shared expert is one memset (moe_shared()), and a graph launch a layer for it cost an RX 9070 XT's host
    // more than the memset: with experts on the CPU (no doorbell on HIP) a token's layers are launched one by one
    bool shared_recorded() const override { return false; }
#endif
    // routed on the card in that recording: the host reads neither
    bool router_rows_to_host(int n) const override { return !(n == 1 && whole_rec_); }
#if !defined(STRATA_GPU_METAL)
    // Both GGUFs' routers are F32 of BF16 values: as BF16 a token reads half of their bytes, the same scores
    // (gemv_bf16() is gemv_f32()'s sums, a prompt batch's gemm_f32() takes them widened).  Idea: merbanan's Strata
    // branch inkling2-turing.
    bool router_bf16() const override { return true; }
#endif
    void copy_rows(const float* from, float* to, int n);
    void tail_rows(int il, int n);
    /// The application of layer il in pass `pass`, in the order a token runs them.
    int visit_of(int il, int pass) const {
        const int core = core_end_ - core_start_;
        return il < core_start_ ? il : il < core_end_ ? core_start_ + pass * core + il - core_start_
                                                       : il + (n_pass_ - 1) * core;
    }
    Dev msg_query(int il);
    void ds_open();
    void ds_load();
    void ds_commit(int n);
    void ds_fold();
    void ds_block(int nb);
    DsScreen ds_screen() const;
    template <class F> void window_runs(int64_t pos, F&& f) const;

    std::vector<Block> blocks_;   ///< one a layer
    std::vector<Slot> slots_;     ///< one an application of a layer
    std::vector<bool> swa_;       ///< a layer attends the window
    Dev tok_norm_, out_query_;
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0, n_rel_ = 0, taps_ = 0, blk_ = 0;
    int window_ = 0, ring_ = 0, keep_ = 0;
    int core_start_ = 0, core_end_ = 0, n_pass_ = 1, rank_ = 0;
    int n_unpadded_ = 0;
    float qk_clip_ = 0, rel_clip_ = 0, log_alpha_ = 0, log_floor_ = 1, softcap_ = 0;
    float w_scale_ = 0;
    float* cache_b_ = nullptr;   ///< --cache-bias: each layer's pick bias for the card (slots_moved())
    void route_by(const float* logits, const float* bias, int* ids, float* w, float* probs) const;
    void cached_bias(const Layer& L, int il, float* b) const;
    uint32_t attn_type_ = 0;   ///< --attn-requant's type for the attention's matrices; 0: the file's
    bool unsloth_ = false;     ///< an "inkling2" file: block_count counts applications, the loop's names are Unsloth's
    bool ctrl_bf16_ = false;   ///< --loop-bf16: the loop controllers' matrices in BF16
    bool kv8_ = false;         ///< the attention caches in FP8 E4M3 (load_geometry()); else fp16
    bool kvq8_ = false;        ///< the attention caches in Q8_0 (load_geometry(), --kv q8_0)
    bool int8_dot_ = false;    ///< STRATA_KV_INT8: long rows' scores by __dp4a on the Q8_0 caches (InkAttn::int8_dot)
    size_t kv_row() const {
        return kv8_    ? ink_kv8_row(n_kv_, head_dim_)
               : kvq8_ ? ink_kvq8_row(n_kv_, head_dim_)
                       : (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    }
    int pass_ = 0;   ///< the core's pass the call under way is in
    // The DSpark drafter (--draft-model, a file of architecture dflash): three dense layers over a block of the
    // anchor token and mask rows, reading keys and values made from four of the target's features of every position
    // it ran (the inputs of three layer runs and the final normed state), then a recurrent Markov head over the
    // target's output matrix.  guess_block() runs it; the target checks every guess (window()).
    struct DsLayer {
        Dev wq, wk, wv, wo, k_ctx, v_ctx, gate, up, down;
        const float *attn_norm = nullptr, *ffn_norm = nullptr, *dflare = nullptr, *q_norm = nullptr,
                    *k_norm = nullptr, *k_ctx_norm = nullptr;
        bool swa = false;
        uint16_t *keys = nullptr, *vals = nullptr;   ///< fp16, n_kv * head_dim a cell, position p at cell p % cap
        long long* tags = nullptr;   ///< a window layer's: the position each cell holds (-1: none)
        int64_t cap = 0;
    };
    static constexpr int kDsRing = 16;   ///< positions whose Markov-head context ds_ring_ keeps
    static constexpr int kDsRows = 8;    ///< a block's rows at most
    std::unique_ptr<Shards> ds_file_;
    std::vector<DsLayer> ds_;
    int ds_block_ = 0, ds_window_ = 0, ds_ff_ = 0, ds_rank_ = 0, ds_visit_[3] = {};
    float ds_theta_ = 0, ds_eps_ = 0, ds_cap_ = 0;
    float *ds_alpha_ = nullptr, *ds_mask_ = nullptr, *ds_norm_ = nullptr, *ds_mnorm_ = nullptr, *ds_jb_ = nullptr,
          *ds_ob_ = nullptr, *ds_wt_ = nullptr, *ds_fold_b_ = nullptr, *ds_ring_ = nullptr, *h_ds_row_ = nullptr;
    uint16_t* ds_w1_ = nullptr;   ///< the Markov head's token rows, BF16 [vocabulary][rank]
    Dev ds_wsp_, ds_wh_, ds_wf_, ds_fold_;
    // the screen of the pick (ds_screen()): 2-bit copies of the output matrix's rows and the fold's, where the output
    // matrix is Q8_0; null: every guess reads both whole
    void *ds_head_sc_ = nullptr, *ds_fold_sc_ = nullptr;
    float* ds_fold_bp_ = nullptr;
    // a block's first position and the ones after it (at[2 t]), then its anchor token (at[2 kDsRows]): read on the
    // card, so that a block's launches are recorded once for its length (ds_graph_)
    long long *ds_at_ = nullptr, *h_ds_at_ = nullptr;
    void* ds_graph_[kDsRows + 1] = {};
    int32_t* ds_prev_ = nullptr;
    Likeliest* ds_pick_ = nullptr;
    float *ds_tap_ = nullptr, *ds_mix_ = nullptr, *ds_k_ = nullptr, *ds_v_ = nullptr, *ds_x_ = nullptr,
          *ds_y_ = nullptr, *ds_q_ = nullptr, *ds_att_ = nullptr, *ds_g_ = nullptr, *ds_u_ = nullptr, *ds_h_ = nullptr,
          *ds_base_ = nullptr, *ds_z_ = nullptr, *ds_fy_ = nullptr, *ds_part_ = nullptr, *ds_ctxh_ = nullptr,
          *ds_ctxf_ = nullptr, *ds_in_ = nullptr, *ds_sp_ = nullptr, *ds_state_ = nullptr, *ds_lat_ = nullptr,
          *ds_frc_ = nullptr, *ds_approx_ = nullptr, *ds_hscr_ = nullptr;
    void *ds_yq_ = nullptr, *ds_hq_ = nullptr, *ds_latq_ = nullptr;
    // STRATA_INK_ZHA=REFS: the full layers' attention through a ZHA index (arXiv 2610.42112): their keys and values
    // in pinned RAM as 4-bit leaves, chunk summaries in VRAM, a ring of the recent rows; each token reads the chunks
    // the summaries pick exactly and the others by their summaries (inkling2_zha.cu).  The rest of the cache: none.
    struct ZSlot {
        int8_t *kc = nullptr, *vc = nullptr;      ///< the leaves: codes, 4 bits a value, a row a position
        uint16_t *ksc = nullptr, *vsc = nullptr;  ///< their fp16 scales, a 32 values
        int8_t *rkc = nullptr, *rvc = nullptr;    ///< the ring of the last zring_ rows, Q8_0 codes, in VRAM
        uint16_t *rks = nullptr, *rvs = nullptr;
        float* zrefs = nullptr;                   ///< the reference queries [kv][R][128]
        uint16_t *zk = nullptr, *zv = nullptr;    ///< the chunks' summaries
        float* zb = nullptr;
        int zbuilt = 0;                           ///< the chunks summarized
        int zcap = 0;                             ///< the summaries' room, chunks a KV head (it grows: zha_recap())
        int64_t zpos = -1;                        ///< the position after the last call's rows (-1: rebuild)
        ink::ZhaCache zcache;                     ///< an answer's chunks in VRAM
    };
    std::vector<ZSlot> zs_;   ///< a slot's, by its index
    ink::Zha zha_;
    bool zha_on_ = false;
    int zha_pool_ = 0, zha_max_chunks_ = 0;
    int64_t zring_ = 0;
    std::vector<float> zha_refs_;
    std::vector<std::pair<int, int>> zha_ids_;
    ink::ZhaWork zw_;
    const void* zw_owner_ = nullptr;
    int64_t zw_next_ = -1;
    bool zha_full(const Block& B) const { return zha_on_ && !B.swa; }
    /// a position's leaf bytes in one slot, keys and values: 4-bit codes and fp16 scales a 32 values (1,152 B)
    size_t zha_leaf_row() const { return 2 * ((size_t) n_kv_ * head_dim_ / 2 + (size_t) n_kv_ * head_dim_ / 32 * sizeof(uint16_t)); }
    /// a slot's leaf arrays with their bytes a position, in the order a parked conversation keeps them
    std::array<std::pair<uint8_t*, size_t>, 4> zha_rows(ZSlot& z) const {
        const size_t c = (size_t) n_kv_ * head_dim_ / 2, sc = (size_t) n_kv_ * head_dim_ / 32 * sizeof(uint16_t);
        return {{{(uint8_t*) z.kc, c}, {(uint8_t*) z.ksc, sc}, {(uint8_t*) z.vc, c}, {(uint8_t*) z.vsc, sc}}};
    }
    size_t zha_q8n(int cap) const {
        return ink::zha_q8n_make((size_t) n_kv_ * cap * zha_.R, cap, true);   // 4-bit summaries
    }
    bool zgrow_ = false;   ///< the summaries grow with the context
    int zcap0_ = 0;        ///< a slot's first room
    void zha_recap(ZSlot& S, int cap);
    void zha_room(ZSlot& S, int need);
    int zha_pool_for(int chunks) const;
    void zha_repool(ZSlot& S, int P);
    void zha_room_all(int64_t end);
    bool zread_full_ = false;   ///< the read under way is max_read()'s: the prompt goes on, the room is whole
    void zha_load();
    void zha_alloc(int slot, int il, int pass);
    // The leaves' RAM reserved for the whole context at the start and pinned a piece at a time as the context
    // reaches it, let go again when a far shorter one follows (zha_pin()); STRATA_INK_ZHA_PIN_ALL=1 pins it all at the
    // start.  Where the card reads registered memory at the host's addresses (Linux) and the context is over a piece.
    static constexpr int64_t kZhaPiece = 65536;
    bool zlazy_ = false;
    int64_t zpinned_ = 0;   ///< the positions pinned
    void zha_pin(int64_t end, bool trim);
    uint64_t host_growth_bytes() const override {
        return zlazy_ ? (uint64_t) (opt_.max_context - zpinned_) * card_row_bytes() : 0;
    }
    void zha_scratch();
    void zha_attention(int il, int n);
    // the prompt buffers lent (with the driver's virtual memory): score, rstar, sel, stage kc, vc, ks, vs, S
    Mapped zmap_[8];          // 0-6: zha_scratch()'s p[], 7: S
    size_t zfull_[8] = {};
    bool zlend_ = false, zwide_ = false;
    int zkeep_now_ = 1;   ///< the heads the prompt staging is mapped for now
    size_t zcut_ = 0;
    ink::ZhaWork zwd_;   ///< an answer's token's (rows <= 4) when the prompt's are lent
    static constexpr int64_t zkeep_from_ = 131072;   ///< KEEP (every KV head's staging kept) for prompts from here on
    void zha_widen(bool keep);
    void zha_narrow();
    void prompt_begins() override {
        zha_pin(std::max<int64_t>(pos_, read_end_), true);
        if (zgrow_) zha_room_all(std::max<int64_t>(pos_, read_end_));   // before the read takes the VRAM
        if (zlend_) zha_widen(std::max<int64_t>(pos_, read_end_) >= zkeep_from_);
    }
    void prompt_ends() override { if (zlend_) zha_narrow(); }
    void head_part(Layer& L, int il, int n, int part);
    void* grab(const std::function<void()>& work);
    std::vector<void*> zgraph_;   ///< a full slot's two recorded parts of its head (an answer's token)

    void* side_ = nullptr;                             ///< fork()'s stream
    void *side_go_ = nullptr, *side_done_ = nullptr;   ///< its fork and its join

    // device scratch, a row a token
    float* bl_[kBlocks] = {};   ///< the residual streams the blocks started from
    float *bank_ = nullptr, *bank_out_ = nullptr;
    float* home_[kBlocks + 2] = {};   ///< bl_, bank_ and bank_out_ in the scratch, where carry_at(nullptr) puts them
    void save_slots(int il, int pass, int64_t pos, uint8_t* dst);
    float* in_[3] = {};
    float* xin_[3] = {};
    float* msg_[kCtrl] = {};
    std::vector<float*> msgs_;   ///< a later pass's messages of every core layer, kCtrl a layer, a token's
    float *cat_ = nullptr, *ch_ = nullptr, *gain_ = nullptr, *forget_ = nullptr, *mixf_ = nullptr;
    float *cat3_[3] = {}, *ch3_[3] = {}, *gain3_[3] = {}, *forget3_[3] = {};   ///< control3()'s, kStageRows rows
    void* xq3_[3] = {};   ///< project()'s quantized inputs, a token's
    void* xqo_ = nullptr;   ///< a token's attention output quantized for the output product
    float *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *vraw_ = nullptr, *v_ = nullptr, *rr_ = nullptr,
          *sink_ = nullptr, *gate_ = nullptr, *kdyn_ = nullptr, *vdyn_ = nullptr, *att_ = nullptr, *aoraw_ = nullptr,
          *fc_ = nullptr, *score_ = nullptr, *zeros_ = nullptr, *awork_ = nullptr;
    size_t score_n_ = 0;
    static constexpr int64_t kKvq8WorkRows = 4096;   ///< the rows of a Q8_0 cache attend_batch() dequantizes at once
    float* kvw_ = nullptr;                           ///< their halves, keys then values (kvq8_)
    size_t kvw_halves_ = 0;
    float* scores_ = nullptr;   ///< score_ where it is apart from the scratch (scores_apart())
};

}  // namespace strata::poly
