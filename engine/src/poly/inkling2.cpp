// src/poly/inkling2.cpp - the layers of Inkling 2 with a looped core.  The reference for the math is llama.cpp's
// inkling2 graph (src/models/inkling2.cpp and inkling2-loop.cpp).
#include "inkling2.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#if !defined(STRATA_GPU_METAL)
#include "strata/kernels/iq_kernels.hpp"
#endif

#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#ifndef _WIN32
#include <sys/mman.h>
#endif

namespace strata::poly {
namespace {

constexpr size_t kScoreFloats = size_t(16) << 20;   // the attention's scores of the tokens it takes at a time

double env_num(const char* key, double fallback) {
    const char* v = std::getenv(key);
    return v && *v ? std::atof(v) : fallback;
}

// STRATA_INK_ZHA's value: the ZHA index's reference file ("" when unset).
std::string zha_file() {
    const char* v = std::getenv("STRATA_INK_ZHA");
    return v ? v : "";
}
const char* const kCtrlName[] = {"q", "k", "v", "ffn"};

// Unsloth's top-level files (architecture "inkling2": unsloth/Inkling-2-Mini-GGUF outside its Mainline_Support folder)
// name the loop's parts otherwise.  Byte for byte against the inkling2_loop files of the same weights: attn = q,
// mlp = ffn, and the loop's keys are num_loops, loop.core_start and loop.core_end.
const Names kUnslothTensors[] = {
    {"loop_q_ctrl_down.weight", {"loop_attn_down.weight", nullptr}},
    {"loop_q_ctrl_gain.weight", {"loop_attn_gain.weight", nullptr}},
    {"loop_q_ctrl_forget.weight", {"loop_attn_forget.weight", nullptr}},
    {"loop_k_ctrl_down.weight", {"loop_k_down.weight", nullptr}},
    {"loop_k_ctrl_gain.weight", {"loop_k_gain.weight", nullptr}},
    {"loop_k_ctrl_forget.weight", {"loop_k_forget.weight", nullptr}},
    {"loop_v_ctrl_down.weight", {"loop_v_down.weight", nullptr}},
    {"loop_v_ctrl_gain.weight", {"loop_v_gain.weight", nullptr}},
    {"loop_v_ctrl_forget.weight", {"loop_v_forget.weight", nullptr}},
    {"loop_ffn_ctrl_down.weight", {"loop_mlp_down.weight", nullptr}},
    {"loop_ffn_ctrl_gain.weight", {"loop_mlp_gain.weight", nullptr}},
    {"loop_ffn_ctrl_forget.weight", {"loop_mlp_forget.weight", nullptr}},
};
const Names kUnslothKeys[] = {
    {"loop.start", {"loop.core_start", nullptr}},
    {"loop.end", {"loop.core_end", nullptr}},
    {"loop.count", {"num_loops", nullptr}},
};

}  // namespace

const Names* Inkling2::tensor_alias(const char* name) const {
    if (unsloth_)
        for (const Names& n : kUnslothTensors)
            if (std::strcmp(n.name, name) == 0) return &n;
    return nullptr;
}

const Names* Inkling2::meta_alias(const char* key) const {
    if (unsloth_)
        for (const Names& n : kUnslothKeys)
            if (std::strcmp(n.name, key) == 0) return &n;
    return nullptr;
}

// A core layer's message queries, [n_embd, 4] for Q, K, V and the FFN.  An inkling2_loop file has them as the layer's
// loop_msg_query; an Unsloth file has two tensors of the model, [n_embd, 2 * core] each: rows 2i and 2i + 1 of
// loop_res_query are core layer start + i's Q and FFN queries, rows 2i and 2i + 1 of loop_kv_res_query its K and V
// ones.  The four rows go to the dense arena as upload() puts a tensor there, counted on the first pass of the load.
Dev Inkling2::msg_query(int il) {
    const int64_t e = n_embd_;
    if (!unsloth_) return upload(tensor(il, "loop_msg_query.weight"), blk(il, "loop_msg_query.weight"), e, kCtrl, kF32);
    const TensorInfo* rq = gg_->find("loop_res_query.weight");
    const TensorInfo* kq = gg_->find("loop_kv_res_query.weight");
    const int64_t rows = 2 * (int64_t) (core_end_ - core_start_);
    for (const TensorInfo* t : {rq, kq})
        if (!t || t->type != kTypeF32 || t->shape.size() != 2 || (int64_t) t->shape[0] != e ||
            (int64_t) t->shape[1] != rows)
            throw std::runtime_error(std::string(t ? t->name + " is " + shape_of(*t) : "loop_res_query.weight or "
                                                 "loop_kv_res_query.weight is missing") +
                                     ", expected F32 [" + std::to_string(e) + ", " + std::to_string(rows) + "]");
    const size_t bytes = (size_t) kCtrl * (size_t) e * sizeof(float);
    const Dev d{dense_ ? dense_ + dense_at_ : nullptr, kTypeF32, (int) e, kCtrl};
    dense_at_ += pad(bytes) + kTail;
    if (!dense_) return d;
    if (dense_at_ > dense_bytes_) throw std::logic_error("the dense arena is too small for " + blk(il, "loop_msg_query"));
    const float* r = (const float*) home(*rq).data(*rq);
    const float* k = (const float*) home(*kq).data(*kq);
    const size_t i = (size_t) (il - core_start_), row = (size_t) e;
    const float* from[kCtrl] = {r + 2 * i * row, k + 2 * i * row, k + (2 * i + 1) * row, r + (2 * i + 1) * row};
    std::vector<float> q((size_t) kCtrl * row);
    for (int c = 0; c < kCtrl; ++c) std::memcpy(q.data() + (size_t) c * row, from[c], row * sizeof(float));
    h2d(d.p, q.data(), bytes);
    return d;
}

void Inkling2::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    tok_norm_ = out_query_ = Dev{};
}

void Inkling2::load_geometry() {
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    // The file has no block that guesses ahead; --draft-model's DSpark drafter guesses blocks (guess_block()).  A
    // window's tokens leave their convolutions' states for keep() to go back to (snap_of()).  Without the drafter a
    // token at a time unless --window asks for more: guesses from repeats of the conversation wrote 1.8% slower after
    // a short prompt that had few (RTX 5090, UD-Q4_K_XL, --window 4).  With it a window is its block.
    opt_.draft = false;
    // --batch: sequences beside the first, a token of each in one step() (batch_heads()), whose rows the _rows kernels
    // take kInkRows at a time.  Not with the drafter, whose caches and block are one sequence's.
    // At most 16 rows a step: under kStageRows, so a step's products are the few-token kernels that give each row the
    // bits it gets alone (32 rows would take the prompt batch's), and as many as were checked (--check-batch).
    need(opt_.batch <= 15, "Inkling 2 takes up to 15 sequences beside the first (--batch 1 to 15)");
    // A --batch this build or these options cannot run is said and dropped, and the engine answers one request at a
    // time: the server passes on a "parallel" the engine cannot use and expects it to adjust (parallel_args() in
    // serve/server_parts/engines.py).  Refused, it left a config with "parallel" 4 and --kv q8_0 with no server.
    const auto one_sequence = [this](bool ok, const char* where) {
        if (ok || opt_.batch == 0) return;
        std::fprintf(stderr, "strata-poly: Inkling 2 runs one sequence %s: --batch %d is not used\n", where, opt_.batch);
        opt_.batch = 0;
    };
    static_assert(16 <= kInkRows && 16 < kStageRows, "a step's rows fit the _rows kernels and the few-token paths");
#if defined(STRATA_GPU_METAL)
    one_sequence(false, "on Metal");
#endif
#if defined(STRATA_USE_HIP)
    // the _rows kernels are the fused q/k/v and clips kernels HIP does not run (inkling2.cu: clips_apart())
    one_sequence(false, "on AMD");
#endif
    one_sequence(opt_.draft_model.empty(), "with --draft-model");
    // On several cards (--layer-split) a card runs a run of the layers, and the loop's (core_start_ to core_end_ - 1)
    // are on one card: next_layer() runs them again for a later pass.  Engine::open() starts no card inside them.
    // The drafter stays on one card: its features are three layer runs' inputs, which may be another card's.
    need(!split_ || opt_.draft_model.empty(), "Inkling 2's --draft-model runs on one card (no --layer-split)");
    // --batch with --layer-split has not been run
    one_sequence(!split_, "on several cards (--layer-split)");
    need(opt_.kv_format == KvFmt::kFp16 || opt_.kv_format == KvFmt::kFp8 || opt_.kv_format == KvFmt::kQ8_0,
         "Inkling 2's attention caches are fp16, fp8 or q8_0");
    // --batch's _rows kernels read fp16 and FP8 caches only
    one_sequence(opt_.kv_format != KvFmt::kQ8_0, "with --kv q8_0");
    unsloth_ = gg_->get("general.architecture")->s == "inkling2";
    load_shape();
    if (unsloth_) {
        // block_count counts the applications (48), the layers before the core, the core's passes and the layers
        // after it; the file has tensors for the 24 layers of weights
        const int passes = shape_i("loop.count"), core = shape_i("loop.end") - shape_i("loop.start");
        need(passes >= 2 && core > 0 && n_layer_ > (passes - 1) * core, "bad loop");
        n_layer_ -= (passes - 1) * core;
        need(tensor(n_layer_ - 1, "ffn_norm.weight") && !tensor(n_layer_, "ffn_norm.weight"),
             "an inkling2 file's layers are not block_count less the core's later passes");
    }
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto first_u = [&](const char* key) {   // a number, or a list's first where every layer has the same
        const MetaValue* v = find(key);
        need(v != nullptr, key);
        if (v->type != MetaType::ARRAY) return (int) v->u;
        need((int) v->items.size() >= n_layer_, key);
        for (int il = 1; il < n_layer_; ++il) need(v->items[(size_t) il].u == v->items[0].u, key);
        return (int) v->items[0].u;
    };
    n_head_ = first_u("attention.head_count");
    n_kv_ = first_u("attention.head_count_kv");
    head_dim_ = shape_i("attention.key_length");
    need(n_head_ > 0 && n_kv_ > 0 && n_head_ % n_kv_ == 0 && n_head_ <= 256, "bad attention heads");
    need(head_dim_ > 0 && head_dim_ <= 128 && shape_i("attention.value_length") == head_dim_, "bad head size");
    n_rel_ = shape_i("attention.relative_length");
    need(n_rel_ > 0 && n_rel_ <= 32, "bad attention.relative_length");
    need(n_head_ / n_kv_ <= 4 && (head_dim_ == 64 || head_dim_ == 128), "attention heads that ink_attend() does not take");
    qk_clip_ = shape_f("attention.qk_clip");
    rel_clip_ = shape_f("attention.relative_clip");
    log_floor_ = (float) shape_i("attention.log_scale_floor");
    log_alpha_ = shape_f("attention.log_scale_alpha");
    need(qk_clip_ > 0 && rel_clip_ > 0 && log_floor_ >= 1 && log_alpha_ >= 0, "bad clipping or scaling");
    window_ = shape_i("attention.sliding_window");
    need(window_ > 0, "bad attention.sliding_window");
    taps_ = shape_i("shortconv.l_cache");
    need(taps_ >= 2 && taps_ <= 8, "shortconv.l_cache is not 2 to 8");
    blk_ = shape_i("attn_res.block_size");
    need(blk_ > 0 && (n_layer_ - 1) / blk_ + 1 <= kBlocks, "bad attn_res.block_size");
    const MetaValue* scale = find("expert_weights_scale");
    w_scale_ = scale ? (float) scale->num() : 0.f;
    situ_beta_ = shape_f("activation.situ_beta");
    situ_lb_ = shape_f("activation.situ_linear_beta");
    need(situ_beta_ > 0 && situ_lb_ >= 0, "bad activation.situ_beta");
    const MetaValue* cap = find("final_logit_softcapping");
    softcap_ = cap ? (float) cap->num() : 0.f;
    const MetaValue* unpadded = find("vocab_size_unpadded");
    n_unpadded_ = unpadded ? (int) unpadded->u : n_vocab_;
    need(n_unpadded_ > 0 && n_unpadded_ <= n_vocab_, "bad vocab_size_unpadded");
    core_start_ = shape_i("loop.start");
    core_end_ = shape_i("loop.end");
    n_pass_ = shape_i("loop.count");
    rank_ = shape_i("loop.gate_rank");
    const MetaValue* varying = find("loop.varying");
    need(!varying || varying->u == 0, "a loop of a varying number of passes is not supported");
    need(n_pass_ >= 2 && n_pass_ <= 8 && rank_ > 0 && core_start_ > 0 && core_start_ < core_end_ &&
             core_end_ <= n_layer_ && core_start_ % blk_ == 0 && core_end_ % blk_ == 0,
         "bad loop");
    need((core_end_ - core_start_) / blk_ - 1 <= 1, "a loop core of more than two blocks is not supported");
    const auto inside = [&](int il) { return il > core_start_ && il < core_end_; };
    need(!inside(lo_) && !(end_ >= 0 && inside(end_)), "a card of a split starts inside the loop");
    // every expert's scale goes on its joined gate and up product, whatever the experts' type
    joined_scales_ = true;
    // the cached experts' IQ3_S / IQ3_XXS and IQ1_S / IQ2_* products through the kernels that keep all lanes busy (CUDA;
    // the same bits)
    lane_slots_ = true;
#if !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP)
    // In a prompt's read of several calls, a call's row tables go to the card by a kernel, not behind the experts'
    // copies (kernel_tables_).  The same tables reach the same buffers, so the results are the same.  Measured on
    // CUDA only; STRATA_INK_KERNEL_TABLES=0 turns it off.
    {
        const char* v = std::getenv("STRATA_INK_KERNEL_TABLES");
        kernel_tables_ = !(v && std::string(v) == "0");
    }
#endif
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
    // --attn-requant: the attention's six matrices quantized again to that type at load (kGuessMat), fewer bytes a
    // token than their Q8_0.  --loop-bf16: the loop controllers' matrices as BF16, half the bytes of their F32.
    const std::string& a = opt_.attn_requant;
    attn_type_ = a == "q4_0" ? 2 : a == "q5_0" ? 6 : a == "iq4_nl" ? 20 : 0;
    if (attn_type_) guess_type_ = attn_type_;
    ctrl_bf16_ = opt_.loop_bf16;
#if !defined(STRATA_GPU_METAL)
    // Turing's prompt products on the int8 kernel that Ampere and newer take by default (i8_mma.cuh), only with
    // --prompt-int8: on Turing it gave the same most likely token as the halves at 560 of 576 steps (97.2%), under
    // the 97.8% a default change needs, so the halves and cuBLAS stay Turing's default (STRATA_PF_I8MMA=0 turns the
    // kernel off on every card)
    kernels::gemm_iq_i8_turing(opt_.prompt_int8 == 1);
    // STRATA_PREFILL_IQ1S=int8: IQ1_S (UD-Q2_K_XL's gate|up in 6 layers) on the int8 prompt kernel, as the file's
    // other types are on Ampere and newer (i8_mma.cuh: activations int8 a 32 values).  Opt-in: on a 302-token prompt
    // it cost more KL a matrix than those types do.  Unset or =fp16: the FP16 kernel, as before.  Other models never
    // ask, so their IQ1_S products stay on the FP16 kernel.
    {
        const char* v = std::getenv("STRATA_PREFILL_IQ1S");
        kernels::gemm_iq_i8_allow(19, v && std::string(v) == "int8");
    }
    // MXFP4 (UD-Q2_K_XL's down in one layer) on the int8 prompt kernel, as the file's other types already are on
    // Ampere and newer (i8_mma.cuh: activations int8 a 32 values); STRATA_PREFILL_MXFP4=fp16 keeps it on the FP16
    // kernel.  Other models never ask, so their MXFP4 products stay on the FP16 kernel.
    {
        const char* v = std::getenv("STRATA_PREFILL_MXFP4");
        kernels::gemm_iq_i8_allow(39, !(v && std::string(v) == "fp16"));
    }
    // On Turing, int8 products on (--prompt-int8): a prompt batch's cached experts in a launch a class of rows
    // (Engine::expert_tiles_), the same numbers in fewer launches; STRATA_POLY_EXPERT_TILES=0: a launch a run of slots
    // again.  Not from Ampere on, where the runs' launches were not measured slower.
    if (const char* v = std::getenv("STRATA_POLY_EXPERT_TILES"); !(v && v[0] == '0') && opt_.prompt_int8 == 1) {
        int dev = 0, major = 0, minor = 0;
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess)
            expert_tiles_ = major == 7 && minor >= 5;
    }
#endif
    // --kv fp8: the attention caches in FP8 E4M3 with a scale a row's head, about half the bytes of fp16, as Strata
    // v0.2.19's Inkling engine has them on small cards (ink_cuda_backend.cu): at 32K tokens the eight full-attention
    // runs' caches are 1 GiB in fp16, VRAM the expert cache does without.  Only where the flash kernel reads prompts
    // (attend_batch() reads fp16 only).
    kv8_ = opt_.kv_format == KvFmt::kFp8;
    need(!kv8_ || (ink_kv8_ok() && head_dim_ == 128 && n_rel_ == 16),
         "--kv fp8 needs ink_attend_batch(): an NVIDIA card from Ampere on, STRATA_INK_FLASH not 0");
    // --kv q8_0: the caches in Q8_0, a half scale for each block of 32 values (ink_kvq8_row()): 1,088 bytes a row
    // for Inkling 2 Mini's 8 heads of 128, where fp16 takes 2,048.  Unlike fp8 it needs no FP8 conversions, so it
    // works on every card, Turing and HIP included: the token kernels and the flash kernel turn the codes into
    // floats as they read them, and attend_batch() dequantizes a block of rows at a time into kvw_ (halves) for its
    // products.  Opt-in, off by default: the codes round the keys and values.
    kvq8_ = opt_.kv_format == KvFmt::kQ8_0;
    need(!kvq8_ || head_dim_ % 32 == 0, "--kv q8_0 needs heads of a multiple of 32 values");
#if defined(STRATA_GPU_METAL)
    need(!kvq8_, "--kv q8_0 has no Metal kernels for Inkling 2 yet");
#endif
    // STRATA_KV_INT8=1 (the name of Kolibri's opt-in and of our first engine's), with --kv q8_0: a decoded token (a
    // call of one row, or a window's guesses; never a prompt's rows) of a full layer that reads 32,768 keys or more
    // takes its scores by __dp4a, its queries quantized to Q8_0 codes too (InkAttn::int8_dot).  Opt-in: other bits.
    if (const char* e = std::getenv("STRATA_KV_INT8"); e && std::atoi(e) != 0) {
        int8_dot_ = kvq8_;
        std::fprintf(stderr, "strata-poly: STRATA_KV_INT8: %s\n",
                     int8_dot_ ? "scores of tokens past 32K keys by __dp4a on the q8_0 cache"
                               : "unchanged (the cache is not q8_0)");
    }

    const MetaValue* pattern = find("attention.sliding_window_pattern");
    need(pattern && pattern->type == MetaType::ARRAY && (int) pattern->items.size() >= n_layer_,
         "attention.sliding_window_pattern has no value for every layer");
    slots_.assign((size_t) n_layer_ + (size_t) (n_pass_ - 1) * (core_end_ - core_start_), Slot{});
    for (int il = 0; il < n_layer_; ++il)
        for (int p = 0; p < (il >= core_start_ && il < core_end_ ? n_pass_ : 1); ++p)
            slots_[(size_t) slot_of(il, p)].il = il;
    swa_.assign((size_t) n_layer_, false);
    // an Unsloth file lists the applications (48 for Inkling 2 Mini), so a layer's value is its first application's;
    // index il there would be right only while the pattern's period divides the loop's length
    for (int il = 0; il < n_layer_; ++il) {
        const size_t v = unsloth_ ? (size_t) visit_of(il, 0) : (size_t) il;
        need(v < pattern->items.size(), "attention.sliding_window_pattern has no value for every layer");
        swa_[(size_t) il] = pattern->items[v].u != 0;
    }
    // and every other application must be its layer's
    if (unsloth_)
        for (int il = 0; il < n_layer_; ++il)
            for (int p = 0; p < (il >= core_start_ && il < core_end_ ? n_pass_ : 1); ++p) {
                const size_t v = (size_t) visit_of(il, p);
                need(v < pattern->items.size() && (pattern->items[v].u != 0) == swa_[(size_t) il],
                     "attention.sliding_window_pattern differs between a layer's applications");
            }
    if (!opt_.draft_model.empty()) ds_open();
    if (opt_.window == 0) opt_.window = ds_file_ ? ds_block_ : 1;
    zha_load();
    // The index keeps one sequence's leaves, ring and summaries, and has not been run beside the drafter.
    need(!zha_on_ || !ds_file_, "the ZHA index (STRATA_INK_ZHA) runs without --draft-model");
    one_sequence(!zha_on_, "with the ZHA index (STRATA_INK_ZHA)");
}

// --draft-model's DSpark drafter: its shape, checked against what guess_block() runs.
void Inkling2::ds_open() {
    ds_file_ = std::make_unique<Shards>(opt_.draft_model);
    const auto need = [&](bool ok, const std::string& what) {
        if (!ok) throw std::runtime_error(opt_.draft_model + ": not a DSpark drafter of this model (" + what + ")");
    };
    const MetaValue* arch = ds_file_->get("general.architecture");
    need(arch && arch->s == "dflash" && ds_file_->find("markov_w1.weight"), "general.architecture dflash, markov_w1");
    const auto key = [&](const char* k) {
        const MetaValue* v = ds_file_->get(std::string("dflash.") + k);
        need(v != nullptr, std::string("dflash.") + k);
        return v;
    };
    ds_block_ = (int) key("block_size")->u;
    ds_window_ = (int) key("attention.sliding_window")->u;
    ds_ff_ = (int) key("feed_forward_length")->u;
    ds_rank_ = (int) key("dspark.markov_rank")->u;
    ds_theta_ = (float) key("rope.freq_base")->num();
    ds_eps_ = (float) key("attention.layer_norm_rms_epsilon")->num();
    ds_cap_ = (float) key("final_logit_softcapping")->num();
    const int layers = (int) key("block_count")->u;
    const MetaValue* swa = key("attention.sliding_window_pattern");
    const MetaValue* targets = key("target_layers");
    const int visits = n_layer_ + (n_pass_ - 1) * (core_end_ - core_start_);
    need((int) key("embedding_length")->u == n_embd_ && (int) key("attention.head_count")->u == n_head_ &&
             (int) key("attention.head_count_kv")->u == n_kv_ && (int) key("attention.key_length")->u == head_dim_ &&
             head_dim_ == 128 && (int) key("vocab_size")->u == n_vocab_,
         "the target's width, heads and vocabulary");
    need(ds_block_ >= 2 && ds_block_ <= kDsRows && ds_rank_ % 32 == 0 && ds_rank_ <= 1024 && ds_ff_ > 0 &&
             ds_window_ > 0 && n_head_ % n_kv_ == 0 && n_head_ / n_kv_ * ds_block_ <= 24,
         "a block of 2 to 8, a Markov rank of a multiple of 32, 24 query rows a key head at most");
    need(!key("sample_from_anchor")->u && !key("attention.causal")->u, "drafts from the mask rows, a block's rows "
                                                                       "seeing each other");
    need(swa->type == MetaType::ARRAY && (int) swa->items.size() == layers && targets->type == MetaType::ARRAY &&
             targets->items.size() == 4 && (int) targets->items[3].u == visits,
         "a window pattern a layer, three layer runs and the final state as its features");
    for (int k = 0; k < 3; ++k) ds_visit_[k] = (int) targets->items[(size_t) k].u;
    ds_.assign((size_t) layers, DsLayer{});
    for (int l = 0; l < layers; ++l) ds_[(size_t) l].swa = swa->items[(size_t) l].u != 0;
}

// The drafter's weights and caches to VRAM, once (before the expert cache takes the rest).
void Inkling2::ds_load() {
    const Shards& f = *ds_file_;
    const auto find = [&](const std::string& name, uint32_t type, uint64_t n_in, uint64_t n_out) {
        const TensorInfo* t = f.find(name);
        if (!t || t->type != type || t->shape.empty() || t->shape[0] != n_in || t->elements() != n_in * n_out)
            throw std::runtime_error(opt_.draft_model + ": " + name + (t ? " is " + shape_of(*t) : " is missing"));
        return t;
    };
    const auto to_dev = [&](const TensorInfo* t) {
        const size_t bytes = f.payload(*t);
        void* d = dmalloc(bytes, "the drafter");
        ck(cudaMemcpy(d, f.data(*t), bytes, cudaMemcpyHostToDevice), "the drafter");
        return d;
    };
    const auto f32 = [&](const std::string& name, uint64_t n) { return (float*) to_dev(find(name, kTypeF32, n, 1)); };
    const auto mat = [&](const std::string& name, uint64_t n_in, uint64_t n_out) {
        const TensorInfo* t = f.find(name);
        const uint32_t type = t ? t->type : 0;
        find(name, type == kTypeBF16 ? kTypeBF16 : 8u, n_in, n_out);   // BF16 or Q8_0
        return Dev{to_dev(t), type, (int) n_in, (int) n_out};
    };
    const uint64_t E = (uint64_t) n_embd_, hq = (uint64_t) n_head_ * head_dim_, kv = (uint64_t) n_kv_ * head_dim_,
                   ff = (uint64_t) ds_ff_, R = (uint64_t) ds_rank_, V = (uint64_t) n_vocab_;
    const size_t cell = kv * sizeof(uint16_t);
    for (size_t l = 0; l < ds_.size(); ++l) {
        DsLayer& L = ds_[l];
        const std::string b = "blk." + std::to_string(l) + ".";
        L.wq = mat(b + "attn_q.weight", E, hq);
        L.wk = mat(b + "attn_k.weight", E, kv);
        L.wv = mat(b + "attn_v.weight", E, kv);
        L.wo = mat(b + "attn_output.weight", hq, E);
        L.k_ctx = mat(b + "attn_k_ctx.weight", E, kv);
        L.v_ctx = mat(b + "attn_v_ctx.weight", E, kv);
        L.gate = mat(b + "ffn_gate.weight", E, ff);
        L.up = mat(b + "ffn_up.weight", E, ff);
        L.down = mat(b + "ffn_down.weight", ff, E);
        L.attn_norm = f32(b + "attn_norm.weight", E);
        L.ffn_norm = f32(b + "ffn_norm.weight", E);
        L.dflare = f32(b + "dflare_norm.weight", E);
        L.q_norm = f32(b + "attn_q_norm.weight", (uint64_t) head_dim_);
        L.k_norm = f32(b + "attn_k_norm.weight", (uint64_t) head_dim_);
        L.k_ctx_norm = f32(b + "attn_k_ctx_norm.weight", (uint64_t) head_dim_);
        // a window layer's ring holds its window, a block and room past it; a full layer every position
        L.cap = L.swa ? (int64_t) ds_window_ + 64 : (int64_t) opt_.max_context + 64;
        L.keys = (uint16_t*) dmalloc((size_t) L.cap * cell, "the drafter's caches");
        L.vals = (uint16_t*) dmalloc((size_t) L.cap * cell, "the drafter's caches");
        if (L.swa) {   // a conversation taken up again from an earlier place finds later positions in the ring
            L.tags = (long long*) dmalloc((size_t) L.cap * sizeof(long long), "the drafter's caches");
            ck(cudaMemset(L.tags, 0xff, (size_t) L.cap * sizeof(long long)), "the drafter's caches");
        }
    }
    {   // each mix's weights over the four features, a softmax (in double) of the file's
        const TensorInfo* t = find("dspark_fuse.weight", kTypeF32, 4, 4);
        f.payload(*t);
        const float* raw = (const float*) f.data(*t);
        float al[16];
        for (int j = 0; j < 4; ++j) {
            double m = raw[j * 4], z = 0;
            for (int k = 1; k < 4; ++k) m = std::max(m, (double) raw[j * 4 + k]);
            for (int k = 0; k < 4; ++k) z += std::exp((double) raw[j * 4 + k] - m);
            for (int k = 0; k < 4; ++k) al[j * 4 + k] = (float) (std::exp((double) raw[j * 4 + k] - m) / z);
        }
        ds_alpha_ = (float*) dmalloc(sizeof(al), "the drafter");
        ck(cudaMemcpy(ds_alpha_, al, sizeof(al), cudaMemcpyHostToDevice), "the drafter");
    }
    ds_mask_ = f32("mask_embd.weight", E);
    ds_norm_ = f32("output_norm.weight", E);
    ds_mnorm_ = f32("dspark_norm.weight", E);
    ds_jb_ = f32("joint_proj.bias", 3 * R);
    ds_ob_ = f32("dspark_hidden_out.bias", E);
    ds_w1_ = (uint16_t*) to_dev(find("markov_w1.weight", kTypeBF16, R, V));
    {   // the recurrence's projection by its inputs: [state ; previous token's row] | hidden | context
        const uint64_t in = 2 * R + 2 * E;
        const TensorInfo* t = find("joint_proj.weight", kTypeBF16, in, 3 * R);
        f.payload(*t);
        const uint16_t* w = (const uint16_t*) f.data(*t);
        std::vector<uint16_t> sp(3 * R * 2 * R), h(3 * R * E), c(3 * R * E);
        for (uint64_t o = 0; o < 3 * R; ++o) {
            std::memcpy(&sp[o * 2 * R], w + o * in, 2 * R * 2);
            std::memcpy(&h[o * E], w + o * in + 2 * R, E * 2);
            std::memcpy(&c[o * E], w + o * in + 2 * R + E, E * 2);
        }
        const auto up = [&](const std::vector<uint16_t>& v, uint64_t n_in) {
            void* d = dmalloc(v.size() * 2, "the drafter");
            ck(cudaMemcpy(d, v.data(), v.size() * 2, cudaMemcpyHostToDevice), "the drafter");
            return Dev{d, kTypeBF16, (int) n_in, (int) (3 * R)};
        };
        ds_wsp_ = up(sp, 2 * R);
        ds_wh_ = up(h, E);
        ds_wf_ = up(c, E);
    }
    {   // hidden_out's columns as rows (fp32), for its fold into the output matrix (ds_fold())
        const TensorInfo* t = find("dspark_hidden_out.weight", kTypeBF16, R, E);
        f.payload(*t);
        const uint16_t* w = (const uint16_t*) f.data(*t);
        std::vector<float> wt(R * E);
        for (uint64_t i = 0; i < E; ++i)
            for (uint64_t r = 0; r < R; ++r) {
                const uint32_t bits = uint32_t(w[i * R + r]) << 16;
                std::memcpy(&wt[r * E + i], &bits, 4);
            }
        ds_wt_ = (float*) dmalloc(wt.size() * 4, "the drafter");
        ck(cudaMemcpy(ds_wt_, wt.data(), wt.size() * 4, cudaMemcpyHostToDevice), "the drafter");
    }
    ds_fold_ = Dev{dmalloc(V * (R / 32) * 34, "the drafter"), 8u, (int) R, (int) V};
    ds_fold_b_ = (float*) dmalloc(V * 4, "the drafter");
    ds_ring_ = (float*) dmalloc((size_t) kDsRing * E * 4, "the drafter");
    ds_prev_ = (int32_t*) dmalloc(64, "the drafter");
    ds_pick_ = (Likeliest*) dmalloc((size_t) ds_block_ * sizeof(Likeliest), "the drafter");
    ds_at_ = (long long*) dmalloc((2 * kDsRows + 1) * sizeof(long long), "the drafter");
    h_ds_row_ = (float*) hmalloc(E * 4, "pinned memory");
    h_ds_at_ = (long long*) hmalloc((2 * kDsRows + 1) * sizeof(long long), "pinned memory");
    ds_fold();
    if (output_.type == 8u && ds_screens(n_embd_, ds_rank_)) {   // the pick's screen (ds_screen())
        const int64_t n = n_unpadded_;
        ds_head_sc_ = dmalloc(ds_screen_bytes(n_embd_, n), "the drafter");
        ds_fold_sc_ = dmalloc(ds_screen_bytes(ds_rank_, n), "the drafter");
        ds_fold_bp_ = (float*) dmalloc((size_t) ds_screen_rows(n) * 4, "the drafter");
        ds_screen_pack(output_.p, n_embd_, n, ds_head_sc_, st_);
        ds_screen_pack(ds_fold_.p, ds_rank_, n, ds_fold_sc_, st_);
        ds_screen_order(ds_fold_b_, n, ds_fold_bp_, st_);
        ck(cudaStreamSynchronize(cs(st_)), "the drafter's screen");
    }
}

// The fold of the Markov head's output into the output matrix, made as the drafter loads: fold[v] = output[v] x
// hidden_out (a Q8_0 row of rank values a token) and fold_b[v] = output[v] . hidden_out's bias, so that a draft's
// scores are output . h plus fold . latent plus fold_b.  Its scratch is its own: the engine's is not there yet.
void Inkling2::ds_fold() {
    const int R = ds_rank_;
    const size_t E = (size_t) n_embd_, V = (size_t) n_vocab_, row = kernels::native_q8_1_bytes(n_embd_);
    float* ft = nullptr;
    uint8_t* xq = nullptr;
    ck(cudaMalloc((void**) &ft, 32 * V * sizeof(float)), "the drafter's fold");
    if (cudaMalloc((void**) &xq, (size_t) (R + 1) * row) != cudaSuccess) {
        cudaFree(ft);
        ck(cudaErrorMemoryAllocation, "the drafter's fold");
    }
    // as lin() quantizes them: a row's blocks are the same alone or among others
    kernels::native_quantize_q8_1(ds_wt_, xq, n_embd_ * R, 1, st_);
    kernels::native_quantize_q8_1(ds_ob_, xq + R * row, n_embd_, 1, st_);
    for (int b = 0; b < R / 32; ++b) {
        for (int c = 0; c < 4; ++c)
            lin(output_, ds_wt_ + (size_t) (b * 32 + c * 8) * E, ft + (size_t) c * 8 * V, 8,
                xq + (b * 32 + c * 8) * row);
        ds_fold_q8(ft, n_vocab_, R / 32, b, ds_fold_.p, st_);
    }
    lin(output_, ds_ob_, ds_fold_b_, 1, xq + R * row);
    const cudaError_t e = cudaStreamSynchronize(cs(st_));
    cudaFree(ft);
    cudaFree(xq);
    ck(e, "the drafter's fold");
}

DsScreen Inkling2::ds_screen() const {
    DsScreen h;
    h.head = output_.p;
    h.fold = ds_fold_.p;
    h.fold_b = ds_fold_b_;
    h.fold_bp = ds_fold_bp_;
    h.head_sc = ds_head_sc_;
    h.fold_sc = ds_fold_sc_;
    h.n_rows = n_unpadded_;
    h.cap = ds_cap_;
    return h;
}

// A call's rows' features for the drafter: their final normed state (the other three were copied as the layers ran),
// mixed into each drafter layer's keys and values at the rows' positions and the Markov head's context.
void Inkling2::ds_commit(int n) {
    const size_t E = (size_t) n_embd_;
    output_rows(0, n, ds_tap_ + 3 * (size_t) rows_ * E);
    const DsPos p{recording_ ? at_ : nullptr, (long long) pos_};
    DsMix m;
    m.tap = ds_tap_;
    m.tap_stride = (size_t) rows_ * E;
    m.alpha = ds_alpha_;
    for (int l = 0; l < 3; ++l) m.norm[l] = ds_[(size_t) l].dflare;
    m.norm[3] = ds_mnorm_;
    m.out = ds_mix_;
    m.ring = ds_ring_;
    m.ring_rows = kDsRing;
    m.pos = p;
    ds_mix(m, n_embd_, n, ds_eps_, st_);
    for (size_t l = 0; l < ds_.size(); ++l) {
        DsLayer& L = ds_[l];
        lin(L.k_ctx, ds_mix_ + l * (size_t) n * E, ds_k_, n);
        lin(L.v_ctx, ds_mix_ + l * (size_t) n * E, ds_v_, n);
        ds_rope(ds_k_, n_kv_, n, L.k_ctx_norm, ds_eps_, ds_theta_, p, L.keys, L.cap, ds_v_, L.vals, st_, L.tags);
    }
}

// The drafter's block after `token` (the token at pos(), not yet run): its guesses of the tokens after it, with their
// probabilities, into h_block_.  A block's launches read its position and anchor on the card (ds_at_), so they are
// the same for every block of its length: recorded once, then replayed with one call.  Launched one by one, a
// block's 123 kernels spanned 1.55 ms of an RTX 5090's time, 1.33 ms of it busy.
void Inkling2::guess_block(int32_t token) {
    block_n_ = 0;
    const int nb = (int) std::min<int64_t>({(int64_t) ds_block_, (int64_t) opt_.window, opt_.max_context - pos_});
    if (!ds_file_ || nb < 2 || pos_ < 1) return;
    input(token, pos_, h_ds_row_);
    for (int t = 0; t < kDsRows; ++t) h_ds_at_[2 * t] = pos_ + t;
    h_ds_at_[2 * kDsRows] = token;
    if (opt_.graphs && !prompt_scratch_) {   // the recording holds the few tokens' scratch
        void*& g = ds_graph_[nb];
        if (!g) {
            cudaGraph_t graph = nullptr;
            ck(cudaStreamBeginCapture(cs(st_), cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
            try {
                ds_block(nb);
            } catch (...) {
                if (cudaStreamEndCapture(cs(st_), &graph) == cudaSuccess && graph) cudaGraphDestroy(graph);
                cudaGetLastError();
                throw;
            }
            ck(cudaStreamEndCapture(cs(st_), &graph), "cudaStreamEndCapture");
            cudaGraphExec_t exec = nullptr;
            const cudaError_t made = cudaGraphInstantiateWithFlags(&exec, graph, 0);
            cudaGraphDestroy(graph);
            ck(made, "cudaGraphInstantiate");
            graphs_.push_back(exec);
            g = exec;
            recorded_ = true;
        }
        ck(cudaGraphLaunch((cudaGraphExec_t) g, cs(st_)), "the drafter");
    } else {
        ds_block(nb);
    }
    ck(cudaStreamSynchronize(cs(st_)), "the drafter");
    block_n_ = nb - 1;
}

// guess_block()'s launches for a block of nb rows: the anchor's normed embedding and nb - 1 mask rows through the
// layers, then the Markov head on the mask rows' final rows, a guess at a time from the one before.  The weights read
// are the drafter's, and the output matrix and the fold, once for the guesses (or their 2-bit copies, ds_screen()).
void Inkling2::ds_block(int nb) {
    const size_t E = (size_t) n_embd_, V = (size_t) n_vocab_, R = (size_t) ds_rank_;
    const int nd = nb - 1;
    ck(cudaMemcpyAsync(ds_at_, h_ds_at_, (2 * kDsRows + 1) * sizeof(long long), cudaMemcpyHostToDevice, cs(st_)),
       "the drafter");
    ck(cudaMemcpyAsync(ds_y_, h_ds_row_, E * 4, cudaMemcpyHostToDevice, cs(st_)), "the drafter");
    rms_norm(ds_y_, fp(tok_norm_), ds_x_, n_embd_, 1, rms_eps_, st_);
    const DsPos p{ds_at_, 0};
    ds_block_in(ds_x_, ds_mask_, n_embd_, nb, p, ds_at_ + 2 * kDsRows, ds_state_, ds_rank_, ds_prev_, ds_ring_,
                kDsRing, ds_frc_, st_);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    for (DsLayer& L : ds_) {
        rms_norm(ds_x_, L.attn_norm, ds_y_, n_embd_, nb, ds_eps_, st_);
        kernels::native_quantize_q8_1(ds_y_, ds_yq_, n_embd_ * nb, 1, st_);
        lin(L.wq, ds_y_, ds_q_, nb, ds_yq_);
        lin_pair(L.wk, L.wv, ds_y_, ds_k_, ds_v_, nb, ds_yq_);
        ds_rope(ds_q_, n_head_, nb, L.q_norm, ds_eps_, ds_theta_, p, nullptr, 0, nullptr, nullptr, st_);
        ds_rope(ds_k_, n_kv_, nb, L.k_norm, ds_eps_, ds_theta_, p, L.keys, L.cap, ds_v_, L.vals, st_, L.tags);
        // a window layer's context is the window before the block, a full one's every position; and the block
        ds_attend(ds_q_, L.keys, L.vals, L.tags, L.cap, p, L.swa ? ds_window_ : 0, nb, n_head_, n_kv_, scale,
                  ds_part_, ds_att_, st_);
        lin(L.wo, ds_att_, ds_y_, nb);
        axpy(ds_y_, 1.f, ds_x_, nb * n_embd_, true, st_);
        rms_norm(ds_x_, L.ffn_norm, ds_y_, n_embd_, nb, ds_eps_, st_);
        kernels::native_quantize_q8_1(ds_y_, ds_yq_, n_embd_ * nb, 1, st_);
        lin_pair(L.gate, L.up, ds_y_, ds_g_, ds_u_, nb, ds_yq_);
        ds_swiglu(ds_g_, ds_u_, (size_t) nb * ds_ff_, st_);
        lin(L.down, ds_g_, ds_y_, nb);
        axpy(ds_y_, 1.f, ds_x_, nb * n_embd_, true, st_);
    }
    rms_norm(ds_x_ + E, ds_norm_, ds_h_, n_embd_, nd, ds_eps_, st_);
    const bool screen = ds_head_sc_ != nullptr;
    if (screen) {
        kernels::native_quantize_q8_1(ds_h_, ds_hq_, n_embd_ * nd, 1, st_);
        ds_head_base(ds_screen(), ds_hq_, nd, ds_approx_, st_);
    } else {
        lin(output_, ds_h_, ds_base_, nd);
    }
    lin(ds_wh_, ds_h_, ds_ctxh_, nd);
    lin(ds_wf_, ds_frc_, ds_ctxf_, 1);
    for (int i = 0; i < nd; ++i) {
        ds_markov_in(ds_state_, ds_w1_, ds_prev_, ds_in_, ds_rank_, st_);
        lin(ds_wsp_, ds_in_, ds_sp_, 1);
        ds_markov_step(ds_ctxh_ + (size_t) i * 3 * R, ds_ctxf_, ds_jb_, ds_sp_, ds_state_, ds_lat_, ds_rank_, st_);
        if (screen) {
            kernels::native_quantize_q8_1(ds_lat_, ds_latq_, ds_rank_, 1, st_);
            ds_head_pick(ds_screen(), ds_hq_, ds_latq_, ds_approx_, i, ds_hscr_, ds_prev_, ds_pick_ + i, st_);
        } else {
            lin(ds_fold_, ds_lat_, ds_fy_, 1);
            ds_pick(ds_base_ + (size_t) i * V, ds_fy_, ds_fold_b_, n_unpadded_, ds_cap_, ds_z_, ds_pick_ + i,
                    ds_prev_, st_);
        }
    }
    ck(cudaMemcpyAsync(h_block_, ds_pick_, (size_t) nd * sizeof(Likeliest), cudaMemcpyDeviceToHost, cs(st_)),
       "the drafter");
}

void Inkling2::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    auto att = [&](const char* n, int64_t a, int64_t b) {
        return upload(tensor(il, n), blk(il, n), a, b, attn_type_ ? kGuessMat : kAnyMat);
    };
    const int64_t e = n_embd_, hq = (int64_t) n_head_ * head_dim_, kv = (int64_t) n_kv_ * head_dim_;

    if (il == lo_) {   // beside the card's first layer: the input's norm on the first card, the output's mix on the last
        if (lo_ == 0) tok_norm_ = upload(gg_->find("token_embd_norm.weight"), "token_embd_norm.weight", e, 1, kF32);
        if (hi_ == n_layer_)
            out_query_ = upload(gg_->find("output_res_query.weight"), "output_res_query.weight", e, 1, kF32);
    }
    B.swa = swa_[(size_t) il];
    // the launches read the position on the card when they are recorded (head_key(), tail_key()); a full layer
    // through the ZHA index decides on the host what it launches (chunks built, the ring, a jump): never recorded
    L.fixed = !zha_full(B);
    B.wq = att("attn_q.weight", e, hq);
    B.wk = att("attn_k.weight", e, kv);
    B.wv = att("attn_v.weight", e, kv);
    B.wo = att("attn_output.weight", hq, e);
    B.gate = att("attn_gate.weight", e, hq);
    B.rel = att("attn_rel.weight", e, (int64_t) n_rel_ * n_head_);
    B.sink = mat("attn_sink_proj.weight", e, n_head_);
    B.q_clip = f32("attn_q_clip.weight", head_dim_, n_head_);
    B.rel_clip = f32("attn_rel_clip.weight", n_rel_, n_head_);
    const TensorInfo* bias = tensor(il, "attn_rel_bias.weight");
    if (!bias || bias->shape.size() != 2 || (int64_t) bias->shape[0] != n_rel_)
        throw std::runtime_error(blk(il, "attn_rel_bias.weight") + " is " + (bias ? shape_of(*bias) : "missing"));
    B.extent = (int) bias->shape[1];
    if (B.swa && B.extent != window_)
        throw std::runtime_error(blk(il, "attn_rel_bias.weight") + " does not have a column a position of the window");
    B.rel_bias = f32("attn_rel_bias.weight", n_rel_, B.extent);
    B.k_conv = f32("attn_k_conv.weight", taps_, kv);
    B.v_conv = f32("attn_v_conv.weight", taps_, kv);
    B.a_conv = f32("attn_post_conv.weight", taps_, e);
    B.f_conv = f32("ffn_post_conv.weight", taps_, e);
    if (B.swa) {
        B.attn_norm = f32("attn_norm.weight", e);
    } else {
        B.q_norm = f32("attn_q_inp_norm.weight", e);
        B.k_norm = f32("attn_k_inp_norm.weight", e);
        B.v_norm = f32("attn_v_inp_norm.weight", e);
        B.k_dyn = mat("attn_k_conv_proj.weight", e, (int64_t) taps_ * n_kv_);
        B.v_dyn = mat("attn_v_conv_proj.weight", e, (int64_t) taps_ * n_kv_);
    }
    if (il > 0) B.attn_query = f32("attn_res_query.weight", e, B.swa ? 1 : 3);
    B.ffn_query = f32("ffn_res_query.weight", e);
    B.ffn_norm = f32("ffn_norm.weight", e);
    if (il >= core_start_ && il < core_end_) {
        B.msg_query = msg_query(il);
        // a `fix` takes upload() past the file's F32 to the matrices' BF16
        auto ctl = [&](const std::string& n, int64_t a, int64_t b) {
            return ctrl_bf16_ ? upload(tensor(il, n.c_str()), blk(il, n.c_str()), a, b, kFloatMat,
                                       [](float v) { return v; })
                              : f32(n.c_str(), a, b);
        };
        for (int c = 0; c < kCtrl; ++c) {
            const std::string pre = std::string("loop_") + kCtrlName[c] + "_ctrl_";
            B.ctrl[c].down = ctl(pre + "down.weight", 2 * e, rank_);
            B.ctrl[c].gain = ctl(pre + "gain.weight", rank_, e);
            B.ctrl[c].forget = ctl(pre + "forget.weight", rank_, e);
        }
    }
    load_moe(il, false);
    B.probs_b.assign((size_t) n_expert_, 0.f);
    const TensorInfo* b = tensor(il, "exp_probs_b.bias");
    if (b) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(blk(il, "exp_probs_b.bias") + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        home(*b).payload(*b);
        std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
        B.probs_b_dev = upload(b, blk(il, "exp_probs_b.bias"), n_expert_, 1, kF32);
    }
}

size_t Inkling2::recurrent_floats() const { return slots_.size() * past_floats(); }

// ROCm on Linux gives VRAM back only with its addresses (Engine::vmm_unmap()), and a prompt's lent buffers crashed
// strata-poly there (an RX 9070 XT, Inkling 2 Mini IQ4_NL with some experts in RAM): the prompt batch's scratch stays
// out of them, in one layout as before.  With the experts frozen (STRATA_FREEZE_EXPERTS, --check-batch) lend_prompt()
// lends nothing, and the small layout would only take VRAM from the expert cache for the whole run.
bool Inkling2::scores_apart() const {
#ifdef STRATA_VMM_RELEASE_WITH_ADDRESSES
    return false;
#else
    return !ds_file_ && !freeze_experts_asked_ && mmq_ && vmm_ && !split_ && rows_ >= kStageRows;
#endif
}

// A card that holds every expert lends nothing either (lend_prompt() with all_cached()).  The cache is not sized yet
// when the engine asks, so this is from the free VRAM: every expert's bytes, the prompt batch's scratch (scratch_at_,
// just counted) and 2 GiB for the reserve, the prompt's other buffers and the staging buffer.  Where it is wrong the
// card either has the small layout it does not use (a few MiB) or no layouts, as main.
bool Inkling2::every_expert_fits() const {
    size_t all = 0;
    for (const Layer& L : layers_) {
        if (!L.moe) continue;
        if (!kernels::native_mmvq_supported((int) L.gu_type) || !kernels::native_mmvq_supported((int) L.u_type) ||
            !kernels::native_mmvq_supported((int) L.d_type))
            return false;   // fill_expert_cache() keeps such a layer's experts in RAM
        all += (size_t) n_expert_ * (L.gu2() + L.d_bytes);
    }
    const size_t need = all + scratch_at_ * sizeof(float) + (size_t(2) << 30);
    return (size_t) vram_free_mib() << 20 >= need;
}

bool Inkling2::scratch_layouts() const { return scores_apart() && !every_expert_fits(); }

int Inkling2::widest() const { return std::max(n_head_ * head_dim_, 2 * n_embd_); }

// Every application of a layer has caches of its own: a full layer's keep a row a position, a window layer's a
// ring of the rows a call can read.  The convolutions' past inputs are in state_.
void Inkling2::alloc_caches() {
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    const size_t row = kv_row();
    float* st = state_;
    if (zha_on_ && zs_.size() != slots_.size()) zs_.assign(slots_.size(), ZSlot{});
    for (Slot& s : slots_) {
        if (!runs(s.il)) continue;
        s.past = st;
        st += past_floats();
        if (blocks_[(size_t) s.il].swa) {
            s.keys = (uint16_t*) seq_alloc((size_t) ring_ * row, "the attention cache");
            s.vals = (uint16_t*) seq_alloc((size_t) ring_ * row, "the attention cache");
        } else if (zha_on_) {   // the index's: leaves in RAM, the ring and the summaries in VRAM
            const int i = (int) (&s - slots_.data()), core = core_end_ - core_start_;
            s.keys = s.vals = nullptr;
            zha_alloc(i, s.il, i < n_layer_ ? 0 : 1 + (i - n_layer_) / core);
        } else {
            s.keys = (uint16_t*) kv_alloc(row, 1);
            s.vals = (uint16_t*) kv_alloc(row, 1);
        }
    }
}

// Per-token buffers hold buf_rows_ tokens, row after row.
void Inkling2::assign_scratch() {
    const size_t n = (size_t) buf_rows_, e = n * n_embd_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    for (float*& b : bl_) b = fbuf(e);
    bank_ = fbuf(e);
    bank_out_ = fbuf(e);
    std::copy(bl_, bl_ + kBlocks, home_);
    home_[kBlocks] = bank_;
    home_[kBlocks + 1] = bank_out_;
    for (float*& b : in_) b = fbuf(e);
    for (float*& b : xin_) b = fbuf(e);
    for (float*& b : msg_) b = fbuf(e);
    msgs_.assign((size_t) (core_end_ - core_start_) * kCtrl, nullptr);
    for (float*& b : msgs_) b = fbuf((size_t) n_embd_);
    cat_ = fbuf(2 * e);
    ch_ = fbuf(n * rank_);
    gain_ = fbuf(e);
    forget_ = fbuf(e);
    const size_t few = (size_t) std::min(buf_rows_, kStageRows);
    for (int j = 0; j < 3; ++j) {
        cat3_[j] = fbuf(2 * few * n_embd_);
        ch3_[j] = fbuf(few * rank_);
        gain3_[j] = fbuf(few * n_embd_);
        forget3_[j] = fbuf(few * n_embd_);
        xq3_[j] = fbuf((few * kernels::native_q8_1_bytes(n_embd_) + sizeof(float) - 1) / sizeof(float));
    }
    mixf_ = fbuf(e);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    vraw_ = fbuf(kv);
    v_ = fbuf(kv);
    rr_ = fbuf(n * n_head_ * n_rel_);
    sink_ = fbuf(n * n_head_);
    gate_ = fbuf(heads);
    kdyn_ = fbuf(n * taps_ * n_kv_);
    vdyn_ = fbuf(n * taps_ * n_kv_);
    att_ = fbuf(heads);
    aoraw_ = fbuf(e);
    fc_ = fbuf(e);
    // a token's scores of the whole context, and what ink_attend() keeps beside them
    // under the index only the window layers' ring goes through ink_attend(): its keys, not the context's (the
    // same for both layouts, which share the scores where they are apart)
    const int64_t att_keys = zha_on_ ? std::min<int64_t>(opt_.max_context, (int64_t) window_ + rows_) : opt_.max_context;
    score_n_ = std::max(kScoreFloats, ink_attend_floats(n_head_, n_kv_, head_dim_, att_keys));
    if (scores_apart()) {   // sized for the context, not for a call's tokens: 64 MiB that both layouts share
        if (!scores_) scores_ = (float*) dmalloc(score_n_ * sizeof(float), "the attention's scores");
        score_ = scores_;
    } else {
        score_ = fbuf(score_n_);
    }
    zeros_ = fbuf((size_t) n_head_);
    xqo_ = fbuf((kernels::native_q8_1_bytes(n_head_ * head_dim_) + sizeof(float) - 1) / sizeof(float));
    // a prompt batch's attention; none in the few tokens' layout, where the batch layout's may be lent
    awork_ = buf_rows_ >= kStageRows ? fbuf(heads + heads / 2 + 1 + 3 * n * n_head_) : nullptr;
    kvw_ = nullptr;
    if (buf_rows_ >= kStageRows && kvq8_) {
        // attend_batch()'s fp16 rows of a Q8_0 cache, keys then values: a block of kKvq8WorkRows rows a product
        // (16 MiB for Inkling 2 Mini), where ink_attend_batch() does not take the batch
        kvw_halves_ = (size_t) std::min<int64_t>(opt_.max_context, kKvq8WorkRows) * n_kv_ * head_dim_;
        kvw_ = fbuf(kvw_halves_);   // 2 * kvw_halves_ halves
    }
    if (zha_on_) zha_scratch();
    if (ds_file_) {   // the drafter's: a call's features and mixes, and a block's rows
        const size_t E = (size_t) n_embd_, hq = (size_t) n_head_ * head_dim_, kv = (size_t) n_kv_ * head_dim_,
                     B = (size_t) ds_block_, V = (size_t) n_vocab_, R = (size_t) ds_rank_;
        ds_tap_ = fbuf(4 * n * E);
        ds_mix_ = fbuf(3 * n * E);
        ds_k_ = fbuf(std::max(n, B) * kv);
        ds_v_ = fbuf(std::max(n, B) * kv);
        ds_x_ = fbuf(B * E);
        ds_y_ = fbuf(B * E);
        ds_q_ = fbuf(B * hq);
        ds_att_ = fbuf(B * hq);
        ds_g_ = fbuf(B * ds_ff_);
        ds_u_ = fbuf(B * ds_ff_);
        ds_h_ = fbuf(B * E);
        ds_base_ = fbuf(B * V);
        ds_z_ = fbuf(V);
        ds_fy_ = fbuf(V);
        ds_part_ = fbuf(ds_attend_floats(n_kv_));
        ds_ctxh_ = fbuf(B * 3 * R);
        ds_ctxf_ = fbuf(3 * R);
        ds_in_ = fbuf(2 * R);
        ds_sp_ = fbuf(3 * R);
        ds_state_ = fbuf(R);
        ds_lat_ = fbuf(R);
        ds_frc_ = fbuf(E);
        const auto bytes = [&](size_t n) { return (void*) fbuf((n + sizeof(float) - 1) / sizeof(float)); };
        ds_yq_ = bytes(B * kernels::native_q8_1_bytes(n_embd_));
        ds_hq_ = bytes(B * kernels::native_q8_1_bytes(n_embd_));
        ds_latq_ = bytes(kernels::native_q8_1_bytes(ds_rank_));
        ds_approx_ = fbuf(B * (size_t) ds_screen_rows(n_unpadded_));
        ds_hscr_ = (float*) bytes(ds_screen_scratch_bytes(n_unpadded_));
        if (!ds_ring_ && scratch_) ds_load();
    }
    if (!side_) {
        cudaStream_t s = nullptr;
        cudaEvent_t go = nullptr, done = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate");
        ck(cudaEventCreateWithFlags(&go, cudaEventDisableTiming), "cudaEventCreate");
        ck(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "cudaEventCreate");
        side_ = s;
        side_go_ = go;
        side_done_ = done;
    }
}

Inkling2::~Inkling2() {
    if (side_) cudaStreamSynchronize(cs(side_));
    for (ZSlot& z : zs_) {   // the summaries and the leaves are the slots' own
        for (void* p : {(void*) z.zk, (void*) z.zv, (void*) z.zb})
            if (p) cudaFree(p);
        if (zgrow_)   // zha_repool()'s
            for (void* p : {(void*) z.zcache.pool.kc, (void*) z.zcache.pool.vc, (void*) z.zcache.pool.ks,
                            (void*) z.zcache.pool.vs, (void*) z.zcache.owner, (void*) z.zcache.stamp,
                            (void*) z.zcache.aux})
                if (p) cudaFree(p);
        for (auto [p, b] : zha_rows(z)) {
            if (!p) continue;
            if (!zlazy_) {
                cudaFreeHost(p);
                continue;
            }
#ifndef _WIN32
            for (int64_t at = 0; at < zpinned_; at += kZhaPiece) cudaHostUnregister(p + (size_t) at * b);
            munmap(p, (size_t) opt_.max_context * b);
#endif
        }
    }
    for (void* e : {side_go_, side_done_})
        if (e) cudaEventDestroy((cudaEvent_t) e);
    if (side_) cudaStreamDestroy(cs(side_));
}

void Inkling2::fork() {
    ck(cudaEventRecord((cudaEvent_t) side_go_, cs(st_)), "branch");
    ck(cudaStreamWaitEvent(cs(side_), (cudaEvent_t) side_go_, 0), "branch");
}

void Inkling2::join() {
    ck(cudaEventRecord((cudaEvent_t) side_done_, cs(side_)), "branch");
    ck(cudaStreamWaitEvent(cs(st_), (cudaEvent_t) side_done_, 0), "branch");
}

void Inkling2::copy_rows(const float* from, float* to, int n) {
#if defined(STRATA_GPU_METAL)
    // a kernel's copy: Metal's copies are a blit encoder's, which ends the compute encoder and fences the next
    copy2(from, to, n * n_embd_, nullptr, nullptr, 0, st_);
#else
    ck(cudaMemcpyAsync(to, from, (size_t) n * n_embd_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)), "rows");
#endif
}

// The first `blocks` residual streams and the current one, from float `at` of each.
InkStates Inkling2::streams_of(int blocks, size_t at) const {
    InkStates s;
    for (int i = 0; i < blocks; ++i) s.p[s.n++] = bl_[i] + at;
    s.p[s.n++] = r_ + at;
    return s;
}

Inkling2::Slot Inkling2::run_slot(int il, int i) const {
    const Slot& s = slots_[(size_t) slot_of(il, pass_of(il))];
    if (!rows_of_) return s;
    const int q = rows_of_[i].seq;
    Slot r = s;
    r.keys = (uint16_t*) seq_buf(q, s.keys);
    r.vals = (uint16_t*) seq_buf(q, s.vals);
    r.past = seq_state(q, s.past);
    return r;
}

// A loop controller: out = cur - tanh(forget) * cur + tanh(gain) * msg, the two gates from cur and msg.  out may
// be cur.
void Inkling2::control(const float* cur, const float* msg, const Ctrl& c, float* out, int n) {
    ink_cat_norm(cur, msg, cat_, n_embd_, n, st_);
    lin(c.down, cat_, ch_, n);
    if (n < kStageRows && c.gain.type == kTypeF32 && c.forget.type == kTypeF32 && c.gain.scale == 1.0f &&
        c.forget.scale == 1.0f) {
        const float* w[2] = {fp(c.gain), fp(c.forget)};
        const float* x[2] = {ch_, ch_};
        float* y[2] = {gain_, forget_};
        const int outs[2] = {n_embd_, n_embd_};
        gemv_set(w, x, y, outs, 2, rank_, n, st_);
    } else {
        lin_pair(c.gain, c.forget, ch_, gain_, forget_, n, nullptr);
    }
    ink_gate(cur, msg, gain_, forget_, out, (size_t) n * n_embd_, st_);
}

// A later pass's FFN controller for one token (control() on mixf_), its gate in the launch of the experts' input
// norm and quantization (native_add_rms_norm_q8_1()'s): the same bytes.  False, with nothing launched, where
// control() and that norm do it.  `row`: the token's row of mixf_, msg, xn_ and xq_ (step()'s rows, one at a time).
bool Inkling2::control_ffn(const float* msg, const Ctrl& c, const Block& B, int row) {
    float* const mixf = mixf_ + (size_t) row * n_embd_;
    float* const xn = xn_ + (size_t) row * n_embd_;
    void* const xq = (uint8_t*) xq_ + (size_t) row * kernels::native_q8_1_bytes(n_embd_);
    msg += (size_t) row * n_embd_;
    if (c.down.type == kTypeBF16) {   // the down product, then the gate (ink_control3_gate()'s of one), then the norm
        if (c.gain.type != kTypeBF16 || c.forget.type != kTypeBF16 || c.down.scale != 1.0f || c.gain.scale != 1.0f ||
            c.forget.scale != 1.0f || rank_ >= 128)
            return false;
        const float* cur = mixf;
        const void* down = c.down.p;
        float* ch = ch_;
        if (!ink_ctrl_down(&cur, &msg, &down, &ch, 1, n_embd_, rank_, 1, st_, true)) return false;
        const float* chs = ch_;
        const void* g = c.gain.p;
        const void* f = c.forget.p;
        float* out = mixf;
        ink_control3_gate(&cur, &msg, &chs, &g, &f, &out, n_embd_, rank_, st_, 1, true);
        kernels::native_add_rms_norm_q8_1(mixf, nullptr, fp(B.ffn_norm), xn, xq, n_embd_, 1, rms_eps_,
                                          sum_threads(n_embd_), st_);
        return true;
    }
    if (c.gain.type != kTypeF32 || c.forget.type != kTypeF32 || c.gain.scale != 1.0f || c.forget.scale != 1.0f ||
        sum_threads(n_embd_) != 1024)
        return false;
    const float* cur = mixf;
    const void* down = fp(c.down);
    float* ch = ch_;
    if (c.down.type != kTypeF32 || c.down.scale != 1.0f ||
        !ink_ctrl_down(&cur, &msg, &down, &ch, 1, n_embd_, rank_, 1, st_)) {   // the norms and lin()'s gemv_kernel
        ink_cat_norm(mixf, msg, cat_, n_embd_, 1, st_);
        lin(c.down, cat_, ch_, 1);
    }
    const float* w[2] = {fp(c.gain), fp(c.forget)};
    const float* x[2] = {ch_, ch_};
    float* y[2] = {gain_, forget_};
    const int outs[2] = {n_embd_, n_embd_};
    gemv_set(w, x, y, outs, 2, rank_, 1, st_);
    ink_gate_norm_q8(mixf, msg, gain_, forget_, mixf, fp(B.ffn_norm), xn, xq, n_embd_, rms_eps_, st_);
    return true;
}

// A token's Q, K, V, relative positions' and gate's products: each distinct input quantized once, the three in one
// launch, and the five Q8_0 matrices in one launch, every output the bits lin() gives it.  A window layer's three
// inputs are one buffer until a later pass's controllers make them differ.  False, with nothing launched, where
// lin() does them: more tokens, or matrices native_mmvq_many() does not take.
//
// A token's on CUDA: the five of several types (Unsloth's files mix Q8_0 with K-quants in a layer) in one launch too,
// native_mmvq_mixed()'s, each output its own launch's bits.  extras (where the caller has no other launch for them):
// the sink's and a full layer's two convolution taps' products join the launch, quantized matrices of n_embd_ inputs
// (the ggml-org files' Q8_0); *extras says whether they did, else the caller launches them.
bool Inkling2::project(const Block& B, const float* const* x, int n, bool quantized, bool* extras) {
    if (extras) *extras = false;
    if (n >= kStageRows) return false;
    const Dev* w[5] = {&B.wq, &B.wk, &B.wv, &B.rel, &B.gate};
    float* y[5] = {q_, kraw_, vraw_, rr_, gate_};
    const int from[5] = {0, 1, 2, 0, 0};
#if defined(STRATA_GPU_METAL)
    // On Metal the five need not be of one type: a launch for each type among them (below)
    constexpr bool kOneType = false;
#else
    constexpr bool kOneType = true;
#endif
#if !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP)
    // one launch of several types: a token's, on CUDA (native_mmvq_mixed() takes no other; HIP keeps its launches)
    static const bool mixed_on = [] {
        const char* v = std::getenv("STRATA_INK_PROJECT_MIXED");
        return v == nullptr || v[0] != '0';
    }();
    const bool mixed = n == 1 && mixed_on;
#else
    constexpr bool mixed = false;
#endif
    for (const Dev* d : w)
        if ((kOneType && !mixed && d->type != w[0]->type) || d->n_in != n_embd_ || d->scale != 1.0f) return false;
    const float* xs[3];
    void* qs[3];
    int nx = 0, slot[3];
    for (int j = 0; j < 3; ++j) {
        int k = 0;
        while (k < nx && xs[k] != x[j]) ++k;
        if (k == nx) {
            xs[nx] = x[j];
            qs[nx] = xq3_[nx];
            ++nx;
        }
        slot[j] = k;
    }
#if defined(STRATA_GPU_METAL)
    // A token's: native_mmvq_many() of the matrices of each type among the five (Unsloth's files mix Q8_0 with K-quants
    // and i-quants in a layer), on the inputs quantized once; a type it does not take, lin() of each.
    if (n == 1) {
        if (!quantized) kernels::native_quantize_q8_1_many(xs, qs, nx, n_embd_, st_);
        bool done[5] = {};
        for (int i = 0; i < 5; ++i) {
            if (done[i]) continue;
            kernels::NativeMmvqMany m{};
            int at[5], c = 0;
            for (int j = i; j < 5; ++j) {
                if (done[j] || w[j]->type != w[i]->type) continue;
                m.w[c] = w[j]->p;
                m.x_q8_1[c] = qs[slot[from[j]]];
                m.y[c] = y[j];
                m.n_out[c] = w[j]->n_out;
                at[c++] = j;
                done[j] = true;
            }
            m.count = c;
            if (w[i]->type != kTypeF32 && w[i]->type != kTypeBF16 &&
                kernels::native_mmvq_many_fits((int) w[i]->type, m, n_embd_)) {
                kernels::native_mmvq_many((int) w[i]->type, m, n_embd_, st_);
                continue;
            }
            for (int k = 0; k < c; ++k) lin(*w[at[k]], x[from[at[k]]], y[at[k]], n, qs[slot[from[at[k]]]]);
        }
        return true;
    }
#endif
    if (n > 1 || !kOneType) {   // a window's: each input quantized once, in one launch, as lin() quantizes it
        if (!quantized) kernels::native_quantize_q8_1_many(xs, qs, nx, n_embd_ * n, st_);
        bool done[5] = {};   // two products of one input a launch (lin_pair())
        for (int i = 0; i < 5; ++i) {
            if (done[i]) continue;
            int j = i + 1;
            while (j < 5 && (done[j] || slot[from[j]] != slot[from[i]])) ++j;
            if (j < 5) lin_pair(*w[i], *w[j], x[from[i]], y[i], y[j], n, qs[slot[from[i]]]);
            else lin(*w[i], x[from[i]], y[i], n, qs[slot[from[i]]]);
            done[i] = true;
            if (j < 5) done[j] = true;
        }
        return true;
    }
#if !defined(STRATA_GPU_METAL)
    // the five, then the sink's (input 0) and a full layer's taps (inputs 1 and 2) where they are to join
    const Dev* all[8] = {w[0], w[1], w[2], w[3], w[4]};
    float* ys[8] = {y[0], y[1], y[2], y[3], y[4]};
    int src[8] = {from[0], from[1], from[2], from[3], from[4]};
    int count = 5;
    const auto plain = [&](const Dev& d) {
        return d.p && d.type != kTypeF32 && d.type != kTypeBF16 && d.n_in == n_embd_ && d.scale == 1.0f;
    };
    if (extras && mixed && plain(B.sink) && (B.swa || (plain(B.k_dyn) && plain(B.v_dyn)))) {
        all[count] = &B.sink, ys[count] = sink_, src[count++] = 0;
        if (!B.swa) {
            all[count] = &B.k_dyn, ys[count] = kdyn_, src[count++] = 1;
            all[count] = &B.v_dyn, ys[count] = vdyn_, src[count++] = 2;
        }
    }
    // the most that one launch takes: all of them, else the five alone
    const int tries[2] = {count, 5};
    for (int t = 0; t < (count > 5 ? 2 : 1); ++t) {
        const int c = tries[t];
        kernels::NativeMmvqMany m{};
        int types[8];
        bool one = true;
        for (int i = 0; i < c; ++i) {
            m.w[i] = all[i]->p;
            m.x_q8_1[i] = qs[slot[src[i]]];
            m.y[i] = ys[i];
            m.n_out[i] = all[i]->n_out;
            types[i] = (int) all[i]->type;
            one = one && types[i] == types[0];
        }
        m.count = c;
        const bool many = one && kernels::native_mmvq_many_fits(types[0], m, n_embd_);
        if (!many && !(mixed && kernels::native_mmvq_mixed_fits(types, m, n_embd_))) continue;
        if (!quantized) kernels::native_quantize_q8_1_many(xs, qs, nx, n_embd_, st_);
        if (many) kernels::native_mmvq_many(types[0], m, n_embd_, st_);
        else kernels::native_mmvq_mixed(types, m, n_embd_, st_);
        if (extras) *extras = c > 5;
        return true;
    }
#endif
    return false;   // lin() quantizes its own (on Metal not reached: a window's block above takes every call)
}

// The three controllers of a later pass's attention inputs for a few tokens, each step one launch for the three
// (ink_cat_norm_set(), gemv_set(), ink_gate_set()), every output the bits control() gives it.  A window layer's
// three cur are one buffer, and out[0] is that buffer too: ink_gate_set() reads every cur before it writes.  False,
// with nothing launched, where control() does it: a prompt batch, or matrices gemv_set() does not take.  Of step()'s
// rows (each a token of its own sequence) the BF16 controllers' gates are a token's launch a row.
bool Inkling2::control3(const float* const* cur, float* const* msg, const Ctrl* c, float* const* out, int n) {
    if (n >= kStageRows) return false;
    const bool bf = c[0].down.type == kTypeBF16;   // --loop-bf16: a token's only, control() the others
    const bool per_row = bf && rows_of_ && n > 1;
    if (bf && !((n == 1 || per_row) && rank_ < 128)) return false;
    for (int j = 0; j < 3; ++j)
        for (const Dev* d : {&c[j].down, &c[j].gain, &c[j].forget})
            if (d->type != (bf ? kTypeBF16 : kTypeF32) || d->scale != 1.0f) return false;
    const float* m[3] = {msg[0], msg[1], msg[2]};
    const void* down[3];
    const float* w[6];
    const float* x[6];
    float* y[6];
    int outs[6];
    for (int j = 0; j < 3; ++j) {
        down[j] = c[j].down.p;
        w[j] = fp(c[j].down);
        x[j] = cat3_[j];
        y[j] = ch3_[j];
        outs[j] = rank_;
    }
    if (!ink_ctrl_down(cur, m, down, y, 3, n_embd_, rank_, n, st_, bf)) {   // the norms and the down products in one launch
        if (bf) return false;
        ink_cat_norm_set(cur, m, cat3_, 3, n_embd_, n, st_);
        gemv_set(w, x, y, outs, 3, 2 * n_embd_, n, st_);
    }
    if ((n == 1 || per_row) && rank_ < 128) {   // the gain and forget products and the gates in one launch
        const void* gw[3] = {c[0].gain.p, c[1].gain.p, c[2].gain.p};
        const void* fw[3] = {c[0].forget.p, c[1].forget.p, c[2].forget.p};
        for (int i = 0; i < n; ++i) {   // a row at a time
            const size_t e = (size_t) i * n_embd_, r = (size_t) i * rank_;
            const float* ci[3] = {cur[0] + e, cur[1] + e, cur[2] + e};
            const float* mi[3] = {m[0] + e, m[1] + e, m[2] + e};
            const float* ch[3] = {ch3_[0] + r, ch3_[1] + r, ch3_[2] + r};
            float* oi[3] = {out[0] + e, out[1] + e, out[2] + e};
            ink_control3_gate(ci, mi, ch, gw, fw, oi, n_embd_, rank_, st_, 3, bf);
        }
        return true;
    }
    for (int j = 0; j < 3; ++j) {
        w[2 * j] = fp(c[j].gain);
        w[2 * j + 1] = fp(c[j].forget);
        x[2 * j] = x[2 * j + 1] = ch3_[j];
        y[2 * j] = gain3_[j];
        y[2 * j + 1] = forget3_[j];
        outs[2 * j] = outs[2 * j + 1] = n_embd_;
    }
    gemv_set(w, x, y, outs, 6, rank_, n, st_);
    const float* g[3] = {gain3_[0], gain3_[1], gain3_[2]};
    const float* f[3] = {forget3_[0], forget3_[1], forget3_[2]};
    ink_gate_set(cur, m, g, f, out, 3, (size_t) n * n_embd_, st_);
    return true;
}

// A layer from its input to its router's copy: the attention, the block's residual, and the experts' input.
void Inkling2::layer_head(Layer& L, int il, int n) {
    // A call starts at its card's first layer.  A card that starts at the loop's first layer comes back to it for
    // each later pass, so it keeps the pass there: next_layer() leaves it 0 after the last pass.
    if (il == lo_ && il != core_start_) pass_ = 0;
    // A full layer under the index, a token: its launches before the index's and after them as two recordings of
    // its own (the engine records only a fixed layer's head, and this one decides on the host what the index runs).
    // Not within a prompt's read: its calls' streams are then in read_groups()' rows (carry_at()), which it gives
    // back after the read, and a recording made with them would write there at its replays.
    const bool home = bl_[0] == home_[0] && bank_ == home_[kBlocks] && bank_out_ == home_[kBlocks + 1];
    if (zha_full(blocks_[(size_t) il]) && n == 1 && !rows_of_ && opt_.graphs && opt_.window <= 1 && !recording_ &&
        home && !prompt_scratch_) {
        if (zgraph_.empty()) zgraph_.assign(slots_.size() * 2, nullptr);
        const size_t at = (size_t) slot_of(il, pass_of(il)) * 2;
        if (!zgraph_[at]) zgraph_[at] = grab([&] { head_part(L, il, n, 1); });
        ck(cudaGraphLaunch((cudaGraphExec_t) zgraph_[at], cs(st_)), "layer graph");
        zha_attention(il, n);
        ink_mul_sigmoid(att_, gate_, (size_t) n_head_ * head_dim_, st_);
        if (!zgraph_[at + 1]) zgraph_[at + 1] = grab([&] { head_part(L, il, n, 2); });
        ck(cudaGraphLaunch((cudaGraphExec_t) zgraph_[at + 1], cs(st_)), "layer graph");
        return;
    }
    head_part(L, il, n, 3);
}

// Engine::record()'s capture (a template of engine.cpp's): what `work` hands to the stream, as a graph.
void* Inkling2::grab(const std::function<void()>& work) {
    cudaGraph_t g = nullptr;
    ck(cudaStreamBeginCapture(cs(st_), cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
    recording_ = recorded_ = true;
    try {
        work();
    } catch (...) {
        recording_ = false;
        if (cudaStreamEndCapture(cs(st_), &g) == cudaSuccess && g) cudaGraphDestroy(g);
        cudaGetLastError();
        throw;
    }
    recording_ = false;
    ck(cudaStreamEndCapture(cs(st_), &g), "cudaStreamEndCapture");
    cudaGraphExec_t exec = nullptr;
    const cudaError_t made = cudaGraphInstantiateWithFlags(&exec, g, 0);
    cudaGraphDestroy(g);
    ck(made, "cudaGraphInstantiate");
    graphs_.push_back(exec);
    return exec;
}

// A layer's head: part 1 its launches up to the keys and values, 2 the ones after the attention, 3 all of them.
void Inkling2::head_part(Layer& L, int il, int n, int part) {
    Block& B = blocks_[(size_t) il];
    if ((part & 1) && ds_file_) {   // the drafter's features: the inputs of three layer runs
        const int visit = visit_of(il, pass_of(il));
        for (int k = 0; k < 3; ++k)
            if (visit == ds_visit_[k]) copy_rows(r_, ds_tap_ + (size_t) k * rows_ * n_embd_, n);
    }
    Slot& S = slots_[(size_t) slot_of(il, pass_of(il))];
    const size_t ne = (size_t) n * n_embd_;
    const int kv = n_kv_ * head_dim_, hq = n_head_ * head_dim_;
    // step()'s rows (rows_of_), each a token of its own sequence: what reads or writes a sequence's states, caches or
    // position goes a row at a time, each with a token's launches and its own sequence's slot (run_slot())
    const bool per_row = rows_of_ != nullptr;
    const int each = per_row ? n : 1;
    // and their launches of a row each in one launch for all of them (the _rows kernels: a row's bits are its own
    // launch's); not on Metal, whose kernels have none (load_geometry() refuses --batch there)
#if defined(STRATA_GPU_METAL)
    const bool rows_launch = false;
#else
    const bool rows_launch = per_row && n > 1;
#endif

    // the tokens' embeddings, normed (a picture's rows are not: whole(); step()'s rows have none)
    if ((part & 1) && il == 0 && (per_row || !image_at(pos_))) {
        rms_norm(r_, fp(tok_norm_), mixf_, n_embd_, n, rms_eps_, st_);
        copy_rows(mixf_, r_, n);
    }
    // A later pass of the core starts again from the stream the core's first block started from, with the pass
    // before it as its bank: that pass's later block (if the core has two) and its output.
    const int core_block = core_start_ / blk_;
    const int n_bank = (core_end_ - core_start_) / blk_ - 1;
    if ((part & 1) && il == core_start_ && pass_ > 0) {
        if (n_bank > 0) copy_rows(bl_[core_block + 1], bank_, n);
        copy_rows(r_, bank_out_, n);
        copy_rows(bl_[core_block], r_, n);
        const int core = core_end_ - core_start_;
        if (n == 1 && core <= 12) {   // the pass's messages of every core layer, in one launch
            InkStates bank;
            if (n_bank > 0) bank.p[bank.n++] = bank_;
            bank.p[bank.n++] = bank_out_;
            const float* q[12];
            for (int k = 0; k < core; ++k) q[k] = fp(blocks_[(size_t) (core_start_ + k)].msg_query);
            ink_mix_many(bank, q, kCtrl, msgs_.data(), core, n_embd_, st_);
        }
    }

    // the attention's inputs: a mix of the streams so far, one for a window layer and one each for Q, K and V for
    // a full one
    const float* in[3] = {r_, r_, r_};
    const bool looped = pass_of(il) > 0;   // a later pass of the core: its inputs take the pass before's messages
    // a window layer outside a later pass has one input, normed in the mix's launch
    const bool mix_normed = il > 0 && B.swa && !looped && sum_threads(n_embd_) == 1024;
    if ((part & 1) && il > 0) {
        float* out[3] = {in_[0], in_[1], in_[2]};
        ink_mix_norm(streams_of((il - 1) / blk_ + 1, 0), fp(B.attn_query), B.swa ? 1 : 3, out, n_embd_, n,
                     mix_normed ? fp(B.attn_norm) : nullptr, mix_normed ? xin_[0] : nullptr, rms_eps_, st_,
                     mix_normed && n == 1 ? xq3_[0] : nullptr);
        in[0] = in_[0];
        in[1] = B.swa ? in_[0] : in_[1];
        in[2] = B.swa ? in_[0] : in_[2];
    }
    float* const* msg = msg_;   // the pass's messages for this layer
    if (looped) {
        if (n == 1 && core_end_ - core_start_ <= 12) {
            msg = msgs_.data() + (size_t) (il - core_start_) * kCtrl;
        } else if (part & 1) {
            InkStates bank;
            if (n_bank > 0) bank.p[bank.n++] = bank_;
            bank.p[bank.n++] = bank_out_;
            ink_mix(bank, fp(B.msg_query), kCtrl, msg_, n_embd_, n, st_);
        }
        // Q's last: a window layer's three inputs are one buffer until the messages make them differ
        if ((part & 1) && !control3(in, msg, B.ctrl, in_, n))
            for (int j = 2; j >= 0; --j) control(in[j], msg[j], B.ctrl[j], in_[j], n);
        for (int j = 0; j < 3; ++j) in[j] = in_[j];
    }
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens' positions from what at_ holds
    if (part & 1) {
    const float* x[3];
    bool quantized = false;   // project()'s inputs quantized by the norms' launch
    if (B.swa && !looped) {
        if (!mix_normed) rms_norm(in[0], fp(B.attn_norm), xin_[0], n_embd_, n, rms_eps_, st_);
        x[0] = x[1] = x[2] = xin_[0];
        quantized = mix_normed && n == 1;
    } else {   // three inputs, normed in one launch
        const float* w[3] = {fp(B.attn_norm), fp(B.attn_norm), fp(B.attn_norm)};
        if (!B.swa) {
            w[0] = fp(B.q_norm);
            w[1] = fp(B.k_norm);
            w[2] = fp(B.v_norm);
        }
        if (n == 1) {
            ink_norm_many_q8(in, w, xin_, xq3_, 3, n_embd_, rms_eps_, st_);
            quantized = true;
        } else {
            rms_norm_many(in, w, xin_, 3, n_embd_, n, rms_eps_, st_);
        }
        for (int j = 0; j < 3; ++j) x[j] = xin_[j];
    }
    dbg("attn_in", il, xin_[0], n_embd_);

    // A token's BF16 products (the sink's, a full layer's convolution taps) are a few microseconds each of a block
    // an output, the card's memory idle: in a whole pass's recording they run beside the Q8_0 ones, a branch off
    // the stream that it joins after them.  gemv_bf16() is lin()'s launch for them.  Not on Metal: a recording there
    // is the capturing stream's alone, so the branch's launches ran once as it was made and never in its replays (the
    // sink and the taps stayed the recording token's), and each event committed a command buffer.
    const auto plain_bf16 = [](const Dev& d) { return !d.p || (d.type == kTypeBF16 && d.scale == 1.0f); };
#if defined(STRATA_GPU_METAL)
    constexpr bool kBranch = false;
#else
    constexpr bool kBranch = true;
#endif
    const bool branch = kBranch && n == 1 && whole_rec_ && plain_bf16(B.sink) && plain_bf16(B.k_dyn) &&
                        plain_bf16(B.v_dyn);
    if (branch) {
        fork();
        gemv_bf16((const uint16_t*) B.sink.p, x[0], sink_, B.sink.n_in, B.sink.n_out, B.sink.n_out, 1, side_);
        if (!B.swa) {
            gemv_bf16((const uint16_t*) B.k_dyn.p, x[1], kdyn_, B.k_dyn.n_in, B.k_dyn.n_out, B.k_dyn.n_out, 1, side_);
            gemv_bf16((const uint16_t*) B.v_dyn.p, x[2], vdyn_, B.v_dyn.n_in, B.v_dyn.n_out, B.v_dyn.n_out, 1, side_);
        }
    }
    bool extras = false;   // the sink's and the taps' products in project()'s launch
    if (!project(B, x, n, quantized, branch ? nullptr : &extras)) {
        lin(B.wq, x[0], q_, n);
        lin(B.wk, x[1], kraw_, n);
        lin(B.wv, x[2], vraw_, n);
        lin(B.rel, x[0], rr_, n);
        lin(B.gate, x[0], gate_, n);
    }
    if (branch) join();
    else if (!extras) lin(B.sink, x[0], sink_, n);
    const float *kd = nullptr, *vd = nullptr;
    if (!B.swa) {
        if (!branch && !extras) {
            lin(B.k_dyn, x[1], kdyn_, n);
            lin(B.v_dyn, x[2], vdyn_, n);
        }
        kd = kdyn_;
        vd = vdyn_;
    }
    const float alpha = B.swa ? 0.f : log_alpha_;
    // one launch for all of it, the same bits (step()'s rows: a launch a row; load_geometry() takes no other heads)
    if ((n == 1 || per_row) && head_dim_ <= 128 && n_rel_ <= 128) {
        for (int i = 0; i < each; ++i) {
            const Slot s = run_slot(il, i);
            const size_t tq = (size_t) i * hq, tk = (size_t) i * kv, tr = (size_t) i * n_head_ * n_rel_,
                         td = (size_t) i * taps_ * n_kv_;
            InkQkv c;
            c.q = q_ + tq;
            c.q_scale = fp(B.q_clip);
            c.r = rr_ + tr;
            c.r_scale = fp(B.rel_clip);
            c.kraw = kraw_ + tk;
            c.vraw = vraw_ + tk;
            c.kstate = past_k(s);
            c.vstate = past_v(s);
            c.kw = fp(B.k_conv);
            c.vw = fp(B.v_conv);
            c.kdyn = kd ? kd + td : nullptr;
            c.vdyn = vd ? vd + td : nullptr;
            c.k = k_ + tk;
            c.v = v_ + tk;
            c.keys = s.keys;
            c.vals = s.vals;
            c.at = at ? at + 2 * i : nullptr;
            c.pos = per_row ? rows_of_[i].pos : pos_;
            c.qk_thr = qk_clip_;
            c.rel_thr = rel_clip_;
            c.alpha = alpha;
            c.floor = log_floor_;
            c.eps = rms_eps_;
            c.n_head = n_head_;
            c.n_kv = n_kv_;
            c.dim = head_dim_;
            c.n_rel = n_rel_;
            c.taps = taps_;
            c.ring = B.swa ? ring_ : 0;
            c.kv8 = kv8_;
            c.kvq8 = kvq8_;
#if !defined(STRATA_GPU_METAL)
            if (rows_launch) {   // all of step()'s rows in one launch, from row 0's, positions from at_
                InkQkvRows r;
                for (int j = 0; j < n; ++j) {
                    const Slot sj = run_slot(il, j);
                    r.kstate[j] = past_k(sj);
                    r.vstate[j] = past_v(sj);
                    r.keys[j] = sj.keys;
                    r.vals[j] = sj.vals;
                }
                c.at = at_;
                ink_qkv_rows(c, r, n, st_);
                break;
            }
#endif
            ink_qkv(c, st_);
        }
    } else {
        if (n <= kConvFew) {   // a window's K and V convolutions in one launch
            InkConv ck, cv;
            ck.x = kraw_, ck.state = past_k(S), ck.w = fp(B.k_conv), ck.dyn = kd, ck.out = k_;
            cv.x = vraw_, cv.state = past_v(S), cv.w = fp(B.v_conv), cv.dyn = vd, cv.out = v_;
            if (snap_bits_) ck.snap = snap_of(past_k(S)), cv.snap = snap_of(past_v(S));
            ink_conv_few(ck, cv, kv, n_kv_, taps_, n, st_, rec_n_, snap_bits_);
        } else {
            ink_conv(kraw_, past_k(S), fp(B.k_conv), kd, k_, kv, n_kv_, taps_, n, st_, snap_of(past_k(S)), rec_n_,
                     snap_bits_);
            ink_conv(vraw_, past_v(S), fp(B.v_conv), vd, v_, kv, n_kv_, taps_, n, st_, snap_of(past_v(S)), rec_n_,
                     snap_bits_);
        }
        InkClips c;   // q, k, v, rr: one launch
        float* xs[4] = {q_, k_, v_, rr_};
        const float* sc[4] = {fp(B.q_clip), nullptr, nullptr, fp(B.rel_clip)};
        const float thr[4] = {qk_clip_, qk_clip_, 0.f, rel_clip_}, al[4] = {alpha, 0.f, 0.f, alpha};
        const float fl[4] = {log_floor_, 1.f, 1.f, log_floor_};
        const int dims[4] = {head_dim_, head_dim_, head_dim_, n_rel_}, hs[4] = {n_head_, n_kv_, n_kv_, n_head_};
        for (int j = 0; j < 4; ++j) {
            c.x[j] = xs[j];
            c.scale[j] = sc[j];
            c.thr[j] = thr[j];
            c.alpha[j] = al[j];
            c.floor[j] = fl[j];
            c.dim[j] = dims[j];
            c.heads[j] = hs[j];
        }
        c.rms[2] = true;
        c.eps = rms_eps_;
        c.pos = pos_;
        c.at = at;
        c.rows = n;
        c.cache[1] = S.keys;   // and the keys and values to their caches
        c.cache[2] = S.vals;
        c.ring = B.swa ? ring_ : 0;
        c.kv8 = kv8_;
        c.kvq8 = kvq8_;
        ink_clips(c, st_);
    }
    }   // part 1
    // a full layer under the index: its keys and values to the leaves and the ring, its attention through the
    // summaries, then the output gate
    const bool through_index = zha_full(B);
    if (part == 3) {
    if (through_index) {
        zha_attention(il, n);
        ink_mul_sigmoid(att_, gate_, (size_t) n * hq, st_);
    }
    InkAttn a;
    a.q = q_;
    a.r = rr_;
    a.sink = sink_;
    a.keys = S.keys;
    a.vals = S.vals;
    a.rel_bias = fp(B.rel_bias);
    a.out = att_;
    a.score = score_;
    a.score_n = score_n_;
    a.pos = pos_;
    a.at = at;
    a.room = at ? room(n) : 0;
    a.rows = n;
    a.n_head = n_head_;
    a.n_kv = n_kv_;
    a.dim = head_dim_;
    a.n_rel = n_rel_;
    a.extent = B.extent;
    a.window = B.swa ? window_ : 0;
    a.ring = B.swa ? ring_ : 0;
    a.scale = 1.f / std::sqrt((float) head_dim_);
    a.gate = gate_;   // the output gate in the join's launch, or the flash kernel's
    a.kv8 = kv8_;
    a.kvq8 = kvq8_;
    // STRATA_KV_INT8 for decode only: a token alone or a window's (guessed tokens checked), not a prompt's rows,
    // which a chunk of fewer than kStageRows also brings here (a short turn, the end of a long prompt)
    a.int8_dot = int8_dot_ && (n == 1 || windowed_);
    if (through_index) {   // done above
    } else if (rows_launch) {   // step()'s rows: each over its own sequence's caches, at its position, as it is alone
#if !defined(STRATA_GPU_METAL)
        InkAttnRows c;
        InkAttn b = a;
        b.at = at_;
        b.room = 0;
        const int64_t keep = pos_;
        for (int i = 0; i < n; ++i) {
            const Slot s = run_slot(il, i);
            c.keys[i] = s.keys;
            c.vals[i] = s.vals;
            // the rows its launch alone is sized for: room()'s steps when recorded, else its position and itself
            pos_ = rows_of_[i].pos;
            const int64_t last = at ? room(1) : pos_ + 1;
            c.parts[i] = ink_attend_parts(n_head_, n_kv_, head_dim_, b.window, last);
            b.room = (int) std::max<int64_t>(b.room, last);
        }
        pos_ = keep;
        ink_attend_rows(b, c, st_);
#endif
    } else if (per_row) {   // the same a launch a row
        const int64_t keep = pos_;
        for (int i = 0; i < n; ++i) {
            const Slot s = run_slot(il, i);
            InkAttn b = a;
            b.q = q_ + (size_t) i * hq;
            b.r = rr_ + (size_t) i * n_head_ * n_rel_;
            b.sink = sink_ + (size_t) i * n_head_;
            b.keys = s.keys;
            b.vals = s.vals;
            b.out = att_ + (size_t) i * hq;
            b.gate = gate_ + (size_t) i * hq;
            b.pos = pos_ = rows_of_[i].pos;
            b.at = at ? at + 2 * i : nullptr;
            b.room = at ? room(1) : 0;
            b.rows = 1;
            if (n == 1) b.q8 = xqo_;   // and the output quantized for wo
            ink_attend(b, st_);
        }
        pos_ = keep;
    } else if (n >= kStageRows && ink_attend_batch(a, st_)) {
        // a prompt batch on the tensor cores with no score written out: attend_batch()'s path below wrote 1.6e11
        // scores of a 30,954-token read and read them back, 8.9 of its 16.8 s of attention on the RTX 4070 Laptop
    } else if (n >= kStageRows) {
        // a prompt batch: the engine's products on the tensor cores, which round the queries and the softmax's
        // weights to halves; the sink's own term of the softmax has score 0 (zeros_)
        if (kv8_) throw std::logic_error("Inkling 2: FP8 caches need ink_attend_batch()");
        AttnBias bias;
        bias.r = rr_;
        bias.sink = sink_;
        bias.table = fp(B.rel_bias);
        bias.n_rel = n_rel_;
        bias.extent = B.extent;
        ck(cudaMemsetAsync(zeros_, 0, (size_t) n_head_ * sizeof(float), cs(st_)), "attention");
        // a Q8_0 cache: a row's codes, then its scales (Kvq8::ss), the products reading rows dequantized into kvw_
        const size_t row = kv_row(), codes = (size_t) n_kv_ * head_dim_;
        const auto q8 = [&](const uint16_t* c) {
            return kvq8_ ? Kvq8{(const int8_t*) c, (const uint16_t*) ((const uint8_t*) c + codes), row, row / 2}
                         : Kvq8{};
        };
        uint16_t* kw = kvq8_ ? (uint16_t*) kvw_ : nullptr;
        attend_batch(blas_, q_, kvq8_ ? nullptr : S.keys, kvq8_ ? nullptr : S.vals, pos_, n, score_, score_n_, awork_,
                     att_, head_dim_, n_head_, n_kv_, a.scale, st_, a.window, a.ring, false, q8(S.keys), q8(S.vals),
                     kw, kw ? kw + kvw_halves_ : nullptr, kvw_halves_, zeros_, nullptr, &bias);
        ink_mul_sigmoid(att_, gate_, (size_t) n * hq, st_);
    } else {
        if (n == 1) a.q8 = xqo_;   // and the output quantized for wo
        ink_attend(a, st_);
    }
    }   // part 3
    if (!(part & 2)) return;
    lin(B.wo, att_, aoraw_, n, n == 1 && n < kStageRows && !through_index ? xqo_ : nullptr);
    // the first layer of a block starts a new stream from the attention's output; the one it had joins the mixes
    if (n == 1 || per_row) {   // the convolution and the stream's update in one launch (a launch a row of step()'s)
        const int mode = il % blk_ != 0 ? 0 : il == core_start_ && looped ? 2 : 1;
        float* const keep = mode == 1 ? bl_[il / blk_] : nullptr;
#if !defined(STRATA_GPU_METAL)
        if (rows_launch) {
            InkStateRows st;
            for (int i = 0; i < n; ++i) st.p[i] = past_a(run_slot(il, i));
            ink_conv_add_rows(aoraw_, st, fp(B.a_conv), ao_, r_, keep, 1.f, mode, n_embd_, taps_, n, st_);
        }
#endif
        for (int i = 0; i < each && !rows_launch; ++i) {
            const size_t t = (size_t) i * n_embd_;
            ink_conv_add(aoraw_ + t, past_a(run_slot(il, i)), fp(B.a_conv), ao_ + t, r_ + t, keep ? keep + t : nullptr,
                         1.f, mode, n_embd_, taps_, st_);
        }
        dbg("attn", il, ao_, n_embd_);
    } else {
        if (n <= kConvFew) {   // the convolution and the stream's update in one launch, as for a token
            InkConv c;
            c.x = aoraw_, c.state = past_a(S), c.w = fp(B.a_conv), c.out = ao_, c.acc = r_;
            if (snap_bits_) c.snap = snap_of(past_a(S));
            c.mode = il % blk_ != 0 ? 0 : il == core_start_ && looped ? 2 : 1;
            if (c.mode == 1) c.keep = bl_[il / blk_];
            ink_conv_few(c, InkConv{}, n_embd_, 1, taps_, n, st_, rec_n_, snap_bits_);
            dbg("attn", il, ao_, n_embd_);
        } else {
            ink_conv(aoraw_, past_a(S), fp(B.a_conv), nullptr, ao_, n_embd_, 1, taps_, n, st_, snap_of(past_a(S)),
                     rec_n_, snap_bits_);
            dbg("attn", il, ao_, n_embd_);
            if (il % blk_ == 0) {
                if (!(il == core_start_ && looped)) copy_rows(r_, bl_[il / blk_], n);
                copy_rows(ao_, r_, n);
            } else {
                axpy(ao_, 1.f, r_, (int) ne, true, st_);
            }
        }
    }

    // the experts' input
    float* mixed[1] = {mixf_};
    if (n == 1 && !looped && sum_threads(n_embd_) == 1024) {   // the mix, add_rms_norm_q8_1's norm and quantization
        ink_mix_norm(streams_of(il / blk_ + 1, 0), fp(B.ffn_query), 1, mixed, n_embd_, n, fp(B.ffn_norm), xn_,
                     rms_eps_, st_, xq_, true);
    } else {
        ink_mix(streams_of(il / blk_ + 1, 0), fp(B.ffn_query), 1, mixed, n_embd_, n, st_);
        // a token's controller (step()'s rows: a row at a time; each takes it or none does)
        bool ffn_done = looped && (n == 1 || per_row) && control_ffn(msg[3], B.ctrl[3], B, 0);
        for (int i = 1; i < n && ffn_done; ++i) control_ffn(msg[3], B.ctrl[3], B, i);
        if (!ffn_done) {
            if (looped) control(mixf_, msg[3], B.ctrl[3], mixf_, n);
            kernels::native_add_rms_norm_q8_1(mixf_, nullptr, fp(B.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_,
                                              sum_threads(n_embd_), st_);
        }
    }
    dbg("ffn_in", il, xn_, n_embd_);
    if (!batched(L, n)) moe_router(L, n);
}

// The experts' sum through its convolution, onto the stream; after the last layer, the drafter's features.
void Inkling2::layer_tail(Layer&, int il, int n) {
    tail_rows(il, n);
    if (ds_file_ && il == n_layer_ - 1) ds_commit(n);
}

void Inkling2::tail_rows(int il, int n) {
    const Slot& S = slots_[(size_t) slot_of(il, pass_of(il))];
    if (n == 1 && std::exchange(moe_left_, false)) {   // moe_card() left the experts' sum here
        ink_moe_conv_add(d_, d_wts_, moe_left_scale_, d_ids_, n_used_, fo_, past_f(run_slot(il, 0)),
                         fp(blocks_[(size_t) il].f_conv), fc_, r_, 1.f, n_embd_, taps_, st_);
        return;
    }
    if (n == 1 || rows_of_) {   // step()'s rows: each with its own sequence's state, in one launch
#if !defined(STRATA_GPU_METAL)
        if (rows_of_ && n > 1) {
            InkStateRows st;
            for (int i = 0; i < n; ++i) st.p[i] = past_f(run_slot(il, i));
            ink_conv_add_rows(fo_, st, fp(blocks_[(size_t) il].f_conv), fc_, r_, nullptr, 1.f, 0, n_embd_, taps_, n,
                              st_);
            return;
        }
#endif
        for (int i = 0; i < (rows_of_ ? n : 1); ++i) {
            const size_t t = (size_t) i * n_embd_;
            ink_conv_add(fo_ + t, past_f(run_slot(il, i)), fp(blocks_[(size_t) il].f_conv), fc_ + t, r_ + t, nullptr,
                         1.f, 0, n_embd_, taps_, st_);
        }
        return;
    }
    if (n <= kConvFew) {   // the convolution and the stream's update in one launch, as for a token
        InkConv c;
        c.x = fo_, c.state = past_f(S), c.w = fp(blocks_[(size_t) il].f_conv), c.out = fc_, c.acc = r_;
        if (snap_bits_) c.snap = snap_of(past_f(S));
        ink_conv_few(c, InkConv{}, n_embd_, 1, taps_, n, st_, rec_n_, snap_bits_);
        return;
    }
    ink_conv(fo_, past_f(S), fp(blocks_[(size_t) il].f_conv), nullptr, fc_, n_embd_, 1, taps_, n, st_,
             snap_of(past_f(S)), rec_n_, snap_bits_);
    axpy(fc_, 1.f, r_, n * n_embd_, true, st_);
}

// After the core's last layer, its first again, until the passes are done.  Called for every layer a call runs,
// replayed or not, so the pass is right for the recordings' keys.
int Inkling2::next_layer(int il) {
    if (il == core_end_ - 1) {
        if (pass_ + 1 < n_pass_) {
            ++pass_;
            return core_start_;
        }
        pass_ = 0;
    }
    return il + 1;
}

// A layer's recordings: one a pass of the core, and for a full layer one a size of the rows it reads.
int Inkling2::head_key(int il, int n) const {
    return pass_of(il) * 32 + (blocks_[(size_t) il].swa ? 0 : room_key(n));
}

// A tail's recordings: one a pass of the core and a sequence, whose state is in them (the heads' are the sequence's
// own: Engine::Seq::heads).
int Inkling2::tail_key(int il, int) const { return pass_of(il) + n_pass_ * bound(); }

void Inkling2::output_rows(int first, int rows, float* h) {
    const size_t at = (size_t) first * n_embd_;
    float* mixed[1] = {mixf_ + at};
    ink_mix(streams_of((n_layer_ - 1) / blk_ + 1, at), fp(out_query_), 1, mixed, n_embd_, rows, st_);
    rms_norm(mixf_ + at, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// The scores' soft cap, and the output matrix's rows past the vocabulary out of every pick.
void Inkling2::finish_logits(float* logits, int rows) {
    if (softcap_ > 0.f || n_unpadded_ < n_vocab_)
        ink_softcap(logits, n_unpadded_, n_vocab_, rows, softcap_ > 0.f ? softcap_ : INFINITY, st_);
}

// What llama.cpp's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias, and weights
// from the bare gates, divided by their sum and scaled.
void Inkling2::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    route_by(logits, blocks_[(size_t) il].probs_b.data(), ids, w, probs);
}

// --cache-bias: F onto the pick of each expert in VRAM.  The pick is the gate plus the layer's bias, so the experts in
// VRAM keep their order among themselves, as do the others, and only a close call between the two goes the other way;
// a looped layer's passes pick among the same experts in VRAM.  The weights stay the bare gates'.
void Inkling2::cached_bias(const Layer& L, int il, float* b) const {
    const std::vector<float>& pb = blocks_[(size_t) il].probs_b;
    for (int e = 0; e < n_expert_; ++e) b[e] = pb[(size_t) e] + (L.slot[(size_t) e] ? opt_.cache_bias : 0.f);
}

void Inkling2::route_cached(const Layer& L, int il, const float* logits, int32_t, int* ids, float* w, float* probs) {
    cached_bias(L, il, bias_rl_.data());
    route_by(logits, bias_rl_.data(), ids, w, probs);
}

// route_card()'s bias for a few tokens, every layer's, from the experts card_moe() has just put on the card
void Inkling2::slots_moved() {
    if (!(opt_.cache_bias > 0.f)) return;
    const size_t ne = (size_t) n_expert_;
    std::vector<float> b((size_t) n_layer_ * ne, 0.f);
    for (int il = 0; il < n_layer_; ++il)
        if (layers_[(size_t) il].moe) cached_bias(layers_[(size_t) il], il, b.data() + (size_t) il * ne);
    if (!cache_b_) cache_b_ = (float*) dmalloc(b.size() * sizeof(float), "scratch");
    ck(cudaMemcpy(cache_b_, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice), "cache bias");
}

void Inkling2::route_by(const float* logits, const float* bias, int* ids, float* w, float* probs) const {
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[e];
    }
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = probs[best];
        sum += w[k];
    }
    const float denom = sum + 1e-20f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

bool Inkling2::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    const float* bias = B.probs_b_dev.p ? fp(B.probs_b_dev) : nullptr;
    // a few tokens (moe_card()) with --cache-bias pick by slots_moved()'s bias; a prompt batch as before
    if (cache_b_ && n < kStageRows) bias = cache_b_ + (size_t) il * n_expert_;
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, bias, nullptr, true, w_scale_, ids, w, st_);
    return true;
}

// No shared expert: the routed experts' sum starts from nothing (where layer_tail() sums them, it starts there).
void Inkling2::moe_shared(const Layer&, int, int n) {
    if (moe_sum_later(n)) return;
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
}

// ---- a sequence's state and rows ----

// The last keep_ positions before `pos` as runs of a ring's rows: f(the run's place among the keep_, its first
// ring row, its rows).  A position before the sequence's start has no run.
template <class F>
void Inkling2::window_runs(int64_t pos, F&& f) const {
    for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
        const int64_t at = p % ring_, m = std::min<int64_t>(pos - p, ring_ - at);
        f((size_t) (p - (pos - keep_)), (size_t) at, (size_t) m);
        p += m;
    }
}

// What the window slots have of the sequence: the rows the next token attends, keys then values.
size_t Inkling2::extra_bytes() const {
    const size_t row = kv_row();
    size_t n = 0;
    for (const Slot& s : slots_) n += blocks_[(size_t) s.il].swa ? 1 : 0;
    return n * 2 * (size_t) keep_ * row;
}

// On a layer split the next card goes on from these and r_: what crosses is the stream, the four blocks' starts the
// mixes read and the loop's bank (64.5 KB a token at Inkling 2 Mini's width).
float* Inkling2::carried(int i) { return i < kBlocks ? bl_[i] : i == kBlocks ? bank_ : bank_out_; }

void Inkling2::carry_at(float* at) {
    const auto place = [&](int i) { return at ? at + (size_t) i * rows_ * n_embd_ : home_[i]; };
    for (int i = 0; i < kBlocks; ++i) bl_[i] = place(i);
    bank_ = place(kBlocks);
    bank_out_ = place(kBlocks + 1);
}

void Inkling2::save_layer(int il, int64_t pos, uint8_t* dst) { save_slots(il, -1, pos, dst); }

// read()'s groups take a slot of the layer at a time: the run under way's
void Inkling2::save_run(int il, int64_t pos, uint8_t* dst) { save_slots(il, pass_of(il), pos, dst); }

// A layer's slots (pass < 0), or its slot of that pass: each one's convolution inputs at their place in the state,
// and a window slot's rows behind the state, after the window slots before it.
void Inkling2::save_slots(int il, int pass, int64_t pos, uint8_t* dst) {
    const size_t row = kv_row(), whole = 2 * (size_t) keep_ * row;
    const Slot* only = pass < 0 ? nullptr : &slots_[(size_t) slot_of(il, pass)];
    size_t rings = 0;
    for (const Slot& s : slots_) {
        const bool swa = blocks_[(size_t) s.il].swa;
        if (s.il == il && s.past && (!only || &s == only)) {
            ck(cudaMemcpyAsync(dst + (size_t) (s.past - state_) * sizeof(float), s.past,
                               past_floats() * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
               "save state");
            if (swa) {
                uint8_t* out = dst + state_n_ * sizeof(float) + rings * whole;
                for (const uint16_t* cache : {s.keys, s.vals}) {
                    std::memset(out, 0, (size_t) keep_ * row);   // the places of positions before the start
                    window_runs(pos, [&](size_t place, size_t at, size_t m) {
                        ck(cudaMemcpyAsync(out + place * row, (const uint8_t*) cache + at * row, m * row,
                                           cudaMemcpyDeviceToHost, cs(st_)),
                           "save state");
                    });
                    out += (size_t) keep_ * row;
                }
            }
        }
        rings += swa ? 1 : 0;
    }
}

void Inkling2::load_extra(const uint8_t* src, int64_t pos) {
    const size_t row = kv_row();
    for (const Slot& s : slots_) {
        if (!blocks_[(size_t) s.il].swa) continue;
        for (uint16_t* cache : {s.keys, s.vals}) {
            if (cache)
                window_runs(pos, [&](size_t place, size_t at, size_t m) {
                    ck(cudaMemcpy((uint8_t*) cache + at * row, src + place * row, m * row, cudaMemcpyHostToDevice),
                       "load state");
                });
            src += (size_t) keep_ * row;
        }
    }
}

// The full slots' rows of the whole context, keys then values a slot.
// With the index, a full slot's rows are its leaves (pinned RAM): a parked conversation keeps them, as it keeps the
// card's rows without the index, since the leaves are by position and the next conversation writes over them.
size_t Inkling2::card_row_bytes() const {
    size_t n = 0;
    for (const Slot& s : slots_) n += blocks_[(size_t) s.il].swa ? 0 : 1;
    return n * (zha_on_ ? zha_leaf_row() : 2 * kv_row());
}

void Inkling2::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    if (zha_on_) {   // the leaves, a slot's key codes, key scales, value codes, value scales (zha_rows())
        for (ZSlot& z : zs_)
            for (auto [p, b] : zha_rows(z)) {
                if (!z.kc) break;
                std::memcpy(dst, p + (size_t) first * b, (size_t) n * b);
                dst += (size_t) n * b;
            }
        return;
    }
    const size_t row = kv_row();
    for (const Slot& s : slots_) {
        if (blocks_[(size_t) s.il].swa || !s.keys || n == 0) continue;
        for (const uint16_t* cache : {s.keys, s.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + (size_t) first * row, (size_t) n * row,
                          cudaMemcpyDeviceToHost),
               "save rows");
            dst += (size_t) n * row;
        }
    }
}

void Inkling2::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const uint8_t* at = src;   // the slots one after the other, as save_card_rows() wrote them
    for (ZSlot& z : zs_) {   // the index's ring and summaries again from its leaves, the chunks from `first` on
        z.zpos = -1;
        z.zbuilt = std::min(z.zbuilt, (int) (first / zha_.C));
        if (!z.kc) continue;
        for (auto [p, b] : zha_rows(z)) {
            std::memcpy(p + (size_t) first * b, at + (size_t) (first - from) * b, (size_t) n * b);
            at += (size_t) count * b;
        }
    }
    if (zha_on_) return;
    const size_t row = kv_row();
    for (const Slot& s : slots_) {
        if (blocks_[(size_t) s.il].swa || !s.keys || n == 0) continue;
        for (uint16_t* cache : {s.keys, s.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + (size_t) first * row, src + (size_t) (first - from) * row,
                          (size_t) n * row, cudaMemcpyHostToDevice),
               "load rows");
            src += (size_t) count * row;
        }
    }
}


// ---- the ZHA index over the full layers' keys --------------------------------------------------------------------

// STRATA_INK_ZHA=FILE: the ZHA index over the full layers (docs/INKLING_ZHA.md).  The file: int32 'ZHAR', C, R,
// slots, then each slot's int32 layer, pass and [kv][R][128] floats (its reference queries).  Its knobs, defaulting
// to what reached 1M tokens on an 8 GB card: STRATA_INK_ZHA_BUDGET (0.16: the share of a token's far chunks it reads
// exactly), STRATA_INK_ZHA_MAX (256: at most that many), STRATA_INK_ZHA_POOL (600: an answer's chunks kept in VRAM a
// KV head, 0 none).
void Inkling2::zha_load() {
    const std::string path = zha_file();
    if (path.empty()) return;
#if defined(STRATA_GPU_METAL) || defined(STRATA_USE_HIP)
    throw std::runtime_error("STRATA_INK_ZHA: the ZHA index runs on NVIDIA cards only");
#else
    const auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(what);
    };
    {
        int dev = 0, major = 0, minor = 0;
        const bool known = cudaGetDevice(&dev) == cudaSuccess &&
                           cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                           cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess;
        need(known && major * 10 + minor >= 75, "STRATA_INK_ZHA: the ZHA index needs an NVIDIA card from Turing on");
    }
    need(opt_.layer_split.empty() && opt_.expert_gpu < 0, "STRATA_INK_ZHA: the ZHA index runs on one card");
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("STRATA_INK_ZHA: cannot open " + path);
    int32_t h[4];
    if (std::fread(h, sizeof h, 1, f) != 1 || h[0] != 0x5A484152) {
        std::fclose(f);
        throw std::runtime_error("STRATA_INK_ZHA: not a ZHAR file");
    }
    zha_.C = h[1];
    zha_.R = h[2];
    if (head_dim_ != 128 || zha_.R != 4 || zha_.C != 64) {
        std::fclose(f);
        throw std::runtime_error("STRATA_INK_ZHA: heads of 128, chunks of 64 and 4 references");
    }
    const size_t per = (size_t) n_kv_ * zha_.R * head_dim_;
    zha_refs_.resize(per * (size_t) h[3]);
    for (int k = 0; k < h[3]; ++k) {
        int32_t id[2];
        if (std::fread(id, sizeof id, 1, f) != 1 || std::fread(zha_refs_.data() + k * per, sizeof(float), per, f) != per) {
            std::fclose(f);
            throw std::runtime_error("STRATA_INK_ZHA: the file ends early");
        }
        zha_ids_.push_back({id[0], id[1]});
    }
    std::fclose(f);
    zha_.budget = (float) env_num("STRATA_INK_ZHA_BUDGET", 0.16);
    zha_.max_sel = (int) env_num("STRATA_INK_ZHA_MAX", 256);
    zha_pool_ = (int) env_num("STRATA_INK_ZHA_POOL", 600);
    need(zha_.budget > 0.f && zha_.budget <= 1.f && zha_.max_sel >= 1 && zha_pool_ >= 0,
         "STRATA_INK_ZHA: _BUDGET in (0, 1], _MAX 1 or more, _POOL 0 or more");
    zring_ = (int64_t) zha_.recent + 2 * zha_.C + std::max(opt_.prompt_batch, 64) + 64;
    zha_max_chunks_ = (int) ((opt_.max_context + zha_.C - 1) / zha_.C + 31) / 32 * 32;   // words of 32 chunks
    zha_on_ = true;
#ifndef _WIN32
    int same = 0;
    cudaDeviceGetAttribute(&same, cudaDevAttrCanUseHostPointerForRegisteredMem, device_);
    zlazy_ = same && opt_.max_context > kZhaPiece && env_num("STRATA_INK_ZHA_PIN_ALL", 0) == 0;
#endif
    std::fprintf(stderr, "strata-poly: ZHA index: chunks of %d, %d references, budget %.3f (at most %d), a pool of %d "
                 "chunks a KV head, its RAM pinned %s\n", zha_.C, zha_.R, zha_.budget, zha_.max_sel, zha_pool_,
                 zlazy_ ? "as the context grows" : "at the start");
#endif
}

// The leaves pinned up to position `end`, kZhaPiece positions a registration, at the host's addresses (the card reads
// them there); with `trim`, a context far shorter than what is pinned gives the pieces past it back to the system.
void Inkling2::zha_pin(int64_t end, bool trim) {
#ifndef _WIN32
    if (!zlazy_) return;
    const int64_t want = std::min<int64_t>(opt_.max_context, (end + kZhaPiece - 1) / kZhaPiece * kZhaPiece);
    const auto piece = [&](int64_t at) { return (size_t) std::min(kZhaPiece, opt_.max_context - at); };
    if (trim && zpinned_ > std::max(kZhaPiece, 4 * want)) {
        const int64_t keep = std::max(kZhaPiece, want);
        ck(cudaStreamSynchronize(cs(st_)), "the index's leaves");
        for (ZSlot& z : zs_)
            for (auto [p, b] : zha_rows(z)) {
                if (!p) break;
                for (int64_t at = keep; at < zpinned_; at += kZhaPiece) {
                    ck(cudaHostUnregister(p + (size_t) at * b), "the index's leaves");
                    madvise(p + (size_t) at * b, piece(at) * b, MADV_DONTNEED);
                }
            }
        zpinned_ = keep;
    }
    if (want <= zpinned_) return;
    for (ZSlot& z : zs_)
        for (auto [p, b] : zha_rows(z)) {
            if (!p) break;
            for (int64_t at = zpinned_; at < want; at += kZhaPiece)
                ck(cudaHostRegister(p + (size_t) at * b, piece(at) * b, cudaHostRegisterPortable | cudaHostRegisterMapped),
                   "the index's leaves (pinned RAM)");
        }
    zpinned_ = want;
#else
    (void) end, (void) trim;
#endif
}

// A full slot's index: its leaves in pinned RAM (every position's keys and values), the ring of its recent rows,
// its reference queries, its summaries of the whole context and its answer's chunk cache in VRAM.
void Inkling2::zha_alloc(int slot, int il, int pass) {
    ZSlot& S = zs_[(size_t) slot];
    if (S.kc) return;
    const size_t codes = (size_t) n_kv_ * head_dim_, scales = codes / 32 * sizeof(uint16_t);
    const size_t rows = (size_t) opt_.max_context;
    const auto host = [&](size_t bytes) {
        void* p = nullptr;
#ifndef _WIN32
        if (zlazy_) {   // the addresses only: zha_pin() pins it
            p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (p == MAP_FAILED) throw std::runtime_error("the index's leaves: no address space for them");
            return p;
        }
#endif
        ck(cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable), "the index's leaves (pinned RAM)");
        return p;
    };
    S.kc = (int8_t*) host(rows * codes / 2);   // 4-bit
    S.ksc = (uint16_t*) host(rows * scales);
    S.vc = (int8_t*) host(rows * codes / 2);
    S.vsc = (uint16_t*) host(rows * scales);
    S.rkc = (int8_t*) dmalloc((size_t) zring_ * codes, "the index's ring");
    S.rks = (uint16_t*) dmalloc((size_t) zring_ * scales, "the index's ring");
    S.rvc = (int8_t*) dmalloc((size_t) zring_ * codes, "the index's ring");
    S.rvs = (uint16_t*) dmalloc((size_t) zring_ * scales, "the index's ring");
    if (zcap0_ == 0) {   // the summaries' room grows with the context where VRAM can come from the expert cache
        zgrow_ = vmm_ && !pool_on_ && !freeze_experts_;
        zcap0_ = zgrow_ ? std::min(zha_max_chunks_, 1024) : zha_max_chunks_;   // 64K tokens first
    }
    if (zha_pool_ > 0 && zgrow_) {   // an answer's chunk cache grows with the context too (zha_repool())
        S.zcache.pmap = (int*) dmalloc((size_t) n_kv_ * zha_max_chunks_ * sizeof(int), "the index's chunk cache");
        zha_repool(S, zha_pool_for(0));
    } else if (zha_pool_ > 0) {
        ink::ZhaCache& Z = S.zcache;
        const size_t keys = (size_t) n_kv_ * zha_pool_ * zha_.C, per_key = head_dim_ / 2;
        Z.P = zha_pool_;
        Z.pool.row = head_dim_;
        Z.pool.headed = 0;
        Z.pool.q4 = 1;
        Z.pool.kc = (int8_t*) dmalloc(keys * per_key, "the index's chunk cache");
        Z.pool.vc = (int8_t*) dmalloc(keys * per_key, "the index's chunk cache");
        Z.pool.ks = (uint16_t*) dmalloc(keys * head_dim_ / 32 * sizeof(uint16_t), "the index's chunk cache");
        Z.pool.vs = (uint16_t*) dmalloc(keys * head_dim_ / 32 * sizeof(uint16_t), "the index's chunk cache");
        Z.owner = (int*) dmalloc((size_t) n_kv_ * Z.P * sizeof(int), "the index's chunk cache");
        Z.stamp = (int*) dmalloc((size_t) n_kv_ * Z.P * sizeof(int), "the index's chunk cache");
        Z.pmap = (int*) dmalloc((size_t) n_kv_ * zha_max_chunks_ * sizeof(int), "the index's chunk cache");
        Z.aux = (int*) dmalloc((size_t) n_kv_ * (Z.P + 2) * sizeof(int), "the index's chunk cache");
        ink::zha_cache_clear(Z, n_kv_, zha_max_chunks_, st_);
    }
    size_t k = 0;
    while (k < zha_ids_.size() && zha_ids_[k] != std::make_pair(il, pass)) ++k;
    if (k == zha_ids_.size())
        throw std::runtime_error("STRATA_INK_ZHA: no references for layer " + std::to_string(il) + " pass " +
                                 std::to_string(pass));
    const size_t refs = (size_t) n_kv_ * zha_.R * head_dim_;
    S.zrefs = (float*) dmalloc(refs * sizeof(float), "the index");
    ck(cudaMemcpy(S.zrefs, zha_refs_.data() + k * refs, refs * sizeof(float), cudaMemcpyHostToDevice), "the index");
    S.zcap = 0;
    S.zbuilt = 0;
    S.zpos = -1;
    zha_recap(S, zcap0_);
}

// The slot's summaries in a room of cap chunks a KV head: the built ones copied over (each KV head's rows, then, as
// q8n lays them out, the scales after every head's codes), the old room freed.  VRAM the card lacks for the new one
// comes from the expert cache; a smaller room's leftover goes back to it.
void Inkling2::zha_recap(ZSlot& S, int cap) {
    if (cap == S.zcap) return;
    const size_t R = (size_t) zha_.R, kv = (size_t) n_kv_;
    const size_t code = head_dim_ / 2, scale = head_dim_ / 32 * sizeof(uint16_t);   // 4-bit codes, fp16 scales
    const size_t rows_new = kv * (size_t) cap * R, rows_old = kv * (size_t) S.zcap * R;
    const size_t bytes_new = 2 * rows_new * (code + scale) + rows_new * sizeof(float);
    const size_t bytes_old = 2 * rows_old * (code + scale) + rows_old * sizeof(float);
    ck(cudaStreamSynchronize(cs(st_)), "the index's summaries");
    if (S.zk && exp_ && bytes_new > bytes_old) {
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        const size_t quarter = (size_t(std::max(reserve_mib_, 0)) << 20) / 4;
        if (free_b < bytes_new + quarter) {
            shrink_cache(bytes_new + quarter - free_b);
            held_is_budget();
        }
    }
    void* p[3] = {};
    const size_t sizes[3] = {rows_new * (code + scale), rows_new * (code + scale), rows_new * sizeof(float)};
    for (int i = 0; i < 3; ++i)
        if (cudaMalloc(&p[i], std::max<size_t>(sizes[i], 256)) != cudaSuccess) {
            cudaGetLastError();
            for (int k = 0; k < i; ++k) cudaFree(p[k]);
            throw std::runtime_error("no VRAM for the index's summaries (" + std::to_string(cap) + " chunks a head)");
        }
    const int keep = std::min(S.zbuilt, std::min(cap, S.zcap));
    if (S.zk && keep > 0) {
        const size_t w = (size_t) keep * R;   // rows a KV head to copy
        uint16_t* const olds[2] = {S.zk, S.zv};
        for (int i = 0; i < 2; ++i) {
            uint8_t* d = (uint8_t*) p[i];
            const uint8_t* s = (const uint8_t*) olds[i];
            ck(cudaMemcpy2DAsync(d, (size_t) cap * R * code, s, (size_t) S.zcap * R * code, w * code, kv,
                                 cudaMemcpyDeviceToDevice, cs(st_)), "the index's summaries");
            ck(cudaMemcpy2DAsync(d + rows_new * code, (size_t) cap * R * scale, s + rows_old * code,
                                 (size_t) S.zcap * R * scale, w * scale, kv, cudaMemcpyDeviceToDevice, cs(st_)),
               "the index's summaries");
        }
        ck(cudaMemcpy2DAsync(p[2], (size_t) cap * R * sizeof(float), S.zb, (size_t) S.zcap * R * sizeof(float),
                             w * sizeof(float), kv, cudaMemcpyDeviceToDevice, cs(st_)), "the index's summaries");
        ck(cudaStreamSynchronize(cs(st_)), "the index's summaries");
    }
    for (void* o : {(void*) S.zk, (void*) S.zv, (void*) S.zb})
        if (o) cudaFree(o);
    const bool had = S.zk != nullptr;
    S.zk = (uint16_t*) p[0], S.zv = (uint16_t*) p[1], S.zb = (float*) p[2];
    S.zbuilt = keep;
    S.zcap = cap;
    if (had && exp_ && bytes_old > bytes_new + kArenaPiece) grow_cache(bytes_old - bytes_new, true);
}

// The chunk cache an answer over `chunks` chunks needs: all of them and chunk 0 (zha_attend() reads through the cache
// only when a selection fits), 64 at least and twice the room at a time, at most STRATA_INK_ZHA_POOL.
int Inkling2::zha_pool_for(int chunks) const {
    int P = std::min(zha_pool_, 64);
    while (P < chunks + 1 && P < zha_pool_) P = std::min(zha_pool_, 2 * P);
    return P;
}

// A slot's chunk cache at P chunks a KV head, empty: the old one's VRAM back first, then from the expert cache what the
// card has not free, as zha_recap() does; a smaller one gives the difference back to the experts.
void Inkling2::zha_repool(ZSlot& S, int P) {
    ink::ZhaCache& Z = S.zcache;
    if (P == Z.P) return;
    const size_t keys = (size_t) n_kv_ * P * zha_.C, code = keys * head_dim_ / 2, scale = keys * head_dim_ / 32 * 2;
    const auto bytes = [&](int p) {
        return p ? (size_t) n_kv_ * p * zha_.C * (head_dim_ + head_dim_ / 8) + (size_t) n_kv_ * (3 * p + 2) * 4 : 0;
    };
    const size_t b_new = bytes(P), b_old = bytes(Z.P);
    ck(cudaStreamSynchronize(cs(st_)), "the index's chunk cache");
    for (void* o : {(void*) Z.pool.kc, (void*) Z.pool.vc, (void*) Z.pool.ks, (void*) Z.pool.vs, (void*) Z.owner,
                    (void*) Z.stamp, (void*) Z.aux})
        if (o) cudaFree(o);
    if (exp_ && b_new > b_old) {
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        const size_t quarter = (size_t(std::max(reserve_mib_, 0)) << 20) / 4;
        if (free_b < b_new + quarter) {
            shrink_cache(b_new + quarter - free_b);
            held_is_budget();
        }
    }
    void* p[7] = {};
    const size_t sizes[7] = {code, code, scale, scale, (size_t) n_kv_ * P * 4, (size_t) n_kv_ * P * 4,
                             (size_t) n_kv_ * (P + 2) * 4};
    for (int i = 0; i < 7; ++i)
        if (cudaMalloc(&p[i], sizes[i]) != cudaSuccess) {
            cudaGetLastError();
            for (int k = 0; k < i; ++k) cudaFree(p[k]);
            Z.P = 0, Z.pool.kc = Z.pool.vc = nullptr, Z.pool.ks = Z.pool.vs = nullptr, Z.owner = Z.stamp = Z.aux = nullptr;
            throw std::runtime_error("no VRAM for the index's chunk cache (" + std::to_string(P) + " chunks a head)");
        }
    Z.P = P;
    Z.pool.row = head_dim_;
    Z.pool.headed = 0;
    Z.pool.q4 = 1;
    Z.pool.kc = (int8_t*) p[0], Z.pool.vc = (int8_t*) p[1];
    Z.pool.ks = (uint16_t*) p[2], Z.pool.vs = (uint16_t*) p[3];
    Z.owner = (int*) p[4], Z.stamp = (int*) p[5], Z.aux = (int*) p[6];
    ink::zha_cache_clear(Z, n_kv_, zha_max_chunks_, st_);
    if (exp_ && b_old > b_new + kArenaPiece) grow_cache(b_old - b_new, true);
}

// A prompt's read starts (prompt_begins()): every slot's room for the chunks up to the read's end, now, while the
// expert cache still has VRAM to give: inside a long prompt the cache has lent all of it to the read's buffers.  A
// read of max_read() rows is a prompt that goes on past it (its later reads would find no VRAM to grow into): the
// whole room at once.
void Inkling2::zha_room_all(int64_t end) {
    int need = (int) ((end + zha_.C - 1) / zha_.C) + 1;
    zread_full_ = end - pos_ >= read_rows_;
    if (zread_full_) need = zha_max_chunks_;
    for (ZSlot& S : zs_)
        if (S.zk) zha_room(S, need);
}

// Room for need chunks: twice the room until it holds them (at most max_chunks); the chunk cache as zha_pool_for().
void Inkling2::zha_room(ZSlot& S, int need) {
    if (zgrow_ && zha_pool_ > 0 && S.zcache.P < zha_pool_for(need)) zha_repool(S, zha_pool_for(need));
    if (!zgrow_ || need <= S.zcap) return;
    int cap = S.zcap;
    while (cap < need && cap < zha_max_chunks_) cap = std::min(zha_max_chunks_, cap * 2);
    zha_recap(S, cap);
}

// The index's work buffers: a prompt call's chunk scores, choices and selections, its staged chunks, an answer's
// parts.  With the driver's virtual memory they are lent: mapped while a prompt is read, the expert cache's between.
void Inkling2::zha_scratch() {
    if (zw_.S) return;
    const int per = n_head_ / n_kv_;
    const size_t sub = (size_t) rows_, extent = (size_t) std::max(window_, 1024);
    zlend_ = vmm_;
    const size_t need = (size_t) per * sub * extent + 65536 + (sub / 64 + 1) * (size_t) zha_max_chunks_ +
                        sub * per * 9 * (head_dim_ + 2) + (size_t) n_head_ * 128 * (head_dim_ + 2);
    zw_.s_floats = std::max<size_t>(size_t(4) << 20, need + (size_t(1) << 20));
    zfull_[7] = zw_.s_floats * sizeof(float);
    if (zlend_) {   // lent with the other prompt buffers
        vmm_reserve(zmap_[7], zfull_[7]);
        zw_.S = (float*) zmap_[7].base;
    } else {
        zw_.S = (float*) dmalloc(zfull_[7], "the index");
    }
    const size_t hs = sub * zha_max_chunks_;
    // lent, every KV head's staged chunks are kept across a slot's prompt calls (KEEP, from zkeep_from_ on): a chunk
    // crosses the bus once a read's group, not once a call
    zw_.heads = zlend_ ? n_kv_ : 1;
    zw_.cmap = (int*) dmalloc((size_t) n_kv_ * zha_max_chunks_ * sizeof(int), "the index");
    zw_.count = (int*) dmalloc((size_t) n_kv_ * sizeof(int), "the index");
    zw_.glist = (int*) dmalloc((size_t) n_kv_ * (zha_max_chunks_ + 1) * sizeof(int), "the index");
    const size_t keys = (size_t) zw_.heads * zha_max_chunks_ * zha_.C, per_key = head_dim_ / 2;
    zw_.stage.row = head_dim_;
    zw_.stage.headed = 0;
    zw_.stage.q4 = 1;
    zfull_[0] = (size_t) (per + 1) * hs * sizeof(float);
    zfull_[1] = zfull_[2] = (size_t) per * hs;
    zfull_[3] = zfull_[4] = keys * per_key;
    zfull_[5] = zfull_[6] = keys * head_dim_ / 32 * sizeof(uint16_t);
    void* p[7];
    for (int i = 0; i < 7; ++i) {
        if (zlend_) {   // addresses now, VRAM when a prompt starts
            vmm_reserve(zmap_[i], zfull_[i]);
            p[i] = (void*) zmap_[i].base;
        } else {
            p[i] = dmalloc(zfull_[i], "the index");
        }
    }
    zw_.score = (float*) p[0];
    zw_.rstar = (uint8_t*) p[1];
    zw_.sel = (uint8_t*) p[2];
    zw_.stage.kc = (int8_t*) p[3];
    zw_.stage.vc = (int8_t*) p[4];
    zw_.stage.ks = (uint16_t*) p[5];
    zw_.stage.vs = (uint16_t*) p[6];
    if (zlend_) {   // an answer's token's own: rows of 4 at most, the same maps and lists
        const size_t hd = (size_t) 4 * zha_max_chunks_;
        zwd_ = zw_;
        zwd_.s_floats = size_t(2) << 20;
        zwd_.S = (float*) dmalloc(zwd_.s_floats * sizeof(float), "the index");
        zwd_.score = (float*) dmalloc((size_t) (n_head_ + n_kv_) * hd * sizeof(float), "the index");
        zwd_.rstar = (uint8_t*) dmalloc((size_t) n_head_ * hd, "the index");
        zwd_.sel = (uint8_t*) dmalloc((size_t) n_head_ * hd, "the index");
        zwd_.heads = 1;
    }
}

// The prompt buffers mapped, the expert cache giving up what the card has not free; with `keep` the staging for every
// KV head (KEEP: a deep prompt's chunks cross the bus once a read's group), else for one.
void Inkling2::zha_widen(bool keep) {
    const int heads = keep && zw_.heads == n_kv_ ? n_kv_ : 1;
    size_t want[8] = {};
    const int nm = zmap_[7].base ? 8 : 7;   // S where it is lent
    for (int i = 0; i < nm; ++i)
        want[i] = i >= 3 && i < 7 && zw_.heads > 1 ? zfull_[i] / zw_.heads * heads : zfull_[i];
    bool enough = zwide_;
    for (int i = 0; i < nm && enough; ++i) enough = zmap_[i].mapped >= want[i];
    if (enough) {
        zkeep_now_ = std::max(zkeep_now_, heads);
        return;
    }
    ck(cudaStreamSynchronize(cs(st_)), "the index's prompt buffers");
    const size_t grain = pool_on_ ? pg_ : grain_;
    const auto whole = [&](size_t x) { return (x + grain - 1) / grain * grain; };
    size_t more = 0;
    for (int i = 0; i < nm; ++i) more += whole(want[i]) > zmap_[i].mapped ? whole(want[i]) - zmap_[i].mapped : 0;
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    if (pool_on_) free_b = vpool_bytes();
    const size_t quarter = pool_on_ ? 0 : (size_t(std::max(reserve_mib_, 0)) << 20) / 4;
    if (exp_ && free_b < more + quarter) {
        const size_t had = arena_.mapped;
        shrink_cache(more + quarter - free_b);
        held_is_budget();
        zcut_ += had - arena_.mapped;
    }
    int mapped_heads = heads;
    try {
        for (int i = 0; i < nm; ++i) vmm_map(zmap_[i], want[i], SIZE_MAX, "the index's prompt buffers");
    } catch (const std::runtime_error&) {
        if (heads == 1) throw;
        // no VRAM for every head's staging: one head's, which an earlier prompt or the first try may have mapped
        mapped_heads = 1;
        for (int i = 0; i < nm; ++i)
            vmm_map(zmap_[i], i >= 3 && i < 7 ? zfull_[i] / zw_.heads : zfull_[i], SIZE_MAX,
                    "the index's prompt buffers");
    }
    zwide_ = true;
    zkeep_now_ = mapped_heads;
    zw_owner_ = nullptr;   // nothing staged survives the buffers' time away
}

// The prompt is read: the expert cache takes the buffers' VRAM back.
void Inkling2::zha_narrow() {
    if (!zwide_) return;
    ck(cudaStreamSynchronize(cs(st_)), "the index's prompt buffers");
    for (int i = 0; i < (zmap_[7].base ? 8 : 7); ++i) vmm_unmap(zmap_[i], 0);
    zwide_ = false;
    zkeep_now_ = 1;
    zw_owner_ = nullptr;
    if (exp_ && zcut_ > 0) grow_cache(zcut_, true);
    zcut_ = 0;
}


// A full layer's call under the index: its rows' keys and values (k_, v_) into the leaves and the ring, the chunks
// they complete summarized, the attention into att_ (before the output gate).
void Inkling2::zha_attention(int il, int n) {
    const Block& B = blocks_[(size_t) il];
    ZSlot& S = zs_[(size_t) slot_of(il, pass_of(il))];
    const int hk = n_kv_ * head_dim_, C = zha_.C;
    zha_pin(pos_ + n, false);
    ink::store_q4(k_, S.kc, S.ksc, hk, pos_, n, 0, st_);   // the leaves: 4-bit
    ink::store_q4(v_, S.vc, S.vsc, hk, pos_, n, 0, st_);
    {   // and the recent rows' ring in VRAM
        const int64_t first = pos_ % zring_;
        const int head = (int) std::min<int64_t>(n, zring_ - first);
        store_q8(k_, S.rkc, S.rks, hk, first, head, st_);
        store_q8(v_, S.rvc, S.rvs, hk, first, head, st_);
        if (head < n) {
            store_q8(k_ + (size_t) head * hk, S.rkc, S.rks, hk, 0, n - head, st_);
            store_q8(v_ + (size_t) head * hk, S.rvc, S.rvs, hk, 0, n - head, st_);
        }
    }
    ink::ZhaKv leaves, ring;
    leaves.kc = S.kc, leaves.ks = S.ksc, leaves.vc = S.vc, leaves.vs = S.vsc;
    leaves.q4 = 1;
    ring.kc = S.rkc, ring.ks = S.rks, ring.vc = S.rvc, ring.vs = S.rvs;
    leaves.row = ring.row = (long long) hk;
    const int64_t n_ring = zring_;
    // A jump in the sequence (a new prompt, a restored state, guesses taken back): the ring's rows before the call
    // again from the leaves, and the summaries of the chunks before it from the leaves.
    if (pos_ != S.zpos) {
        const int64_t refill = std::max<int64_t>(0, pos_ - (zring_ - rows_));
        if (refill < pos_) {   // 4-bit leaves to the Q8_0 ring
            ink::q4_to_q8(S.kc, S.ksc, refill, S.rkc, S.rks, refill, zring_, hk, (int) (pos_ - refill), st_);
            ink::q4_to_q8(S.vc, S.vsc, refill, S.rvc, S.rvs, refill, zring_, hk, (int) (pos_ - refill), st_);
        }
        S.zbuilt = std::min(S.zbuilt, (int) (pos_ / C));
        ink::zha_cache_clear(S.zcache, n_kv_, zha_max_chunks_, st_);
        // the read under way's end too: its start (prompt_begins()) gave the room for all of it
        const int want = (int) (std::max<int64_t>(pos_ + n, read_end_) / C) + 1;
        if (zgrow_ && !zread_full_ && S.zcap > zcap0_ && 4 * want <= S.zcap) {   // a short context again
            int cap = zcap0_;
            while (cap < want) cap *= 2;
            zha_recap(S, std::min(cap, zha_max_chunks_));
        }
        if (zgrow_ && zha_pool_ > 0 && 4 * zha_pool_for(want) <= S.zcache.P) zha_repool(S, zha_pool_for(want));
        zha_room(S, (int) (pos_ / C));
        ink::zha_build(S.zrefs, leaves, (int64_t) opt_.max_context + 1, S.zbuilt, (int) (pos_ / C), C, zha_.R, n_kv_,
                       zha_max_chunks_, S.zk, S.zv, S.zb, st_, zha_q8n(S.zcap));
        S.zbuilt = (int) (pos_ / C);
    }
    const int done = (int) ((pos_ + n) / C);   // the chunks the call completes: from the ring
    zha_room(S, done);
    ink::zha_build(S.zrefs, ring, n_ring, S.zbuilt, done, C, zha_.R, n_kv_, zha_max_chunks_, S.zk, S.zv, S.zb, st_,
                   zha_q8n(S.zcap));
    S.zbuilt = std::max(S.zbuilt, done);
    S.zpos = pos_ + n;
    // an answer's token through the chunk cache takes its own small buffers; anything else the prompt's (a
    // prompt's read maps them at its start: take_prompt())
    const bool small = zlend_ && n == 1 && zha_pool_ > 0;
    if (zlend_ && !small && !zwide_) zha_widen(false);
    ink::ZhaWork wkp = zw_;
    if (zlend_) wkp.heads = zkeep_now_;   // KEEP where the staging is every head's
    const ink::ZhaWork& wk = small ? zwd_ : wkp;
    // a prompt call that goes on from this slot's last one keeps what it staged
    const bool keep = n >= kStageRows && zw_owner_ == &S && zw_next_ == pos_;
    ink::zha_attend(blas_, zha_, q_, leaves, ring, n_ring, S.zk, S.zv, S.zb, S.zbuilt, zha_max_chunks_, pos_, n, n_head_,
                    n_kv_, rr_, fp(B.rel_bias), B.extent, n_rel_, sink_, wk, att_, st_, &S.zcache, keep, S.zcap);
    zw_owner_ = &S;
    zw_next_ = n >= kStageRows ? pos_ + n : -1;
}

}  // namespace strata::poly
