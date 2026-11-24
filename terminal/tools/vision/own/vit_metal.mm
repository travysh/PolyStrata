// tools/vision/own/vit_metal.mm - see vit_gpu.h: the vision towers on a Mac's GPU (Metal), the Qwen family's, Gemma 4's and MiniCPM-V 4.7's: the
// counterpart of vit_gpu.cu (CUDA / HIP) and vit_sycl.cpp (Intel, Qwen only).  The same host flow and the same numbers: weights stay as the file
// stores them (BF16 or F16; F32 matrices become BF16, Q8_0 becomes F16), float32 activations, each matrix product with a 16-bit input of the
// weight's type and float32 accumulation, q / k / v in F16 (Gemma 4: q and k float32 where the float32 tensor-op product passes its check), the
// scores in F16 (Gemma 4 and MiniCPM-V: float32, the probabilities F16), the same chunk rules.  Its own small Metal host (one device, one queue, shared buffers, one command buffer per layer); the kernels
// (vit_metal.metal, and the tensor-op products in vit_metal_mpp.metal) are embedded in the program and compiled when the reader starts.
//
// The products: the Metal 4 tensor ops (MetalPerformancePrimitives matmul2d) where they compile and a self-test against the plain simdgroup kernel
// passes, decided per product at load and logged once; else the plain kernel.  STRATA_VIT_NO_MPP=1 (a diagnostic) asks for the plain kernel everywhere.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "vit_f16.h"
#include "vit_gpu.h"
#include "vit_metal_src.h"     // generated at build time: kVitMetalSrc, kVitMetalMppSrc

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>

#include <fcntl.h>
#include <mach/mach.h>
#include <malloc/malloc.h>
#include <sys/mman.h>
#include <unistd.h>

namespace strata_vit {

namespace {

// K_HF: a BF16 matrix held as F16 with its inputs written as F16 (clamped to +-65,504), the default for BF16 matrices as on CUDA / HIP (vit_gpu.cu, 11-14):
// BF16's 7-bit mantissa on the inputs left the Qwen3.8 reader at cosine 0.998058 against the float32 reference on code4217.png (M5 Max, 11-16), F16's 10 bits
// do not.  A matrix with a weight beyond F16's range stays K_BF16.
enum Kind { K_F32 = 0, K_F16 = 1, K_BF16 = 2, K_HF = 3 };

struct DevW {            // a weight matrix [rows = out][cols = in] on the device
    id<MTLBuffer> b = nil;
    size_t off = 0;      // where it starts in b (the mapped mmproj file's buffer: its offset in the file)
    Kind kind = K_BF16;
    int rows = 0, cols = 0;
};

struct DevLayer {
    id<MTLBuffer> ln1_w = nil, ln1_b = nil, ln2_w = nil, ln2_b = nil, qkv_b = nil, o_b = nil, up_b = nil, down_b = nil;
    DevW qkv, o, up, down;
};

struct GemmaLayer {      // gemma4v: RMS norm weights, the per-head q / k norms, no biases; q, k, v and gate, up as their own matrices
    id<MTLBuffer> ln1 = nil, ln2 = nil, attn_post = nil, ffn_post = nil, q_norm = nil, k_norm = nil;
    DevW q, k, v, o, gate, up, down;
};

// a buffer and a byte offset into it
struct BO {
    id<MTLBuffer> b;
    size_t off;
    BO(id<MTLBuffer> buf, size_t o = 0) : b(buf), off(o) {}
};

// the argument structs of vit_metal.metal (same layout)
struct PatchPostArgs { int32_t G, pw, ph, nbx, D; };
struct NormArgs { int32_t D, kind; float eps; };
struct RopeArgs { int32_t row0, nbx, heads, hd; float theta_scale, qscale; };
struct SoftmaxArgs { int32_t n, ld; };
struct EltArgs { uint64_t n; int32_t kind; int32_t D; };
struct GemmArgs { uint32_t rows, in, out, ldx, ldw, ldy; uint64_t sx, sw, sy; };
struct RmsArgs { int32_t D, kind, has_w; float eps; };
struct GemmaPosArgs { int32_t T, pw, D; };
struct GemmaQkvArgs { int32_t row0, pw, heads, hd; float theta_scale, eps; };
struct GemmaPoolArgs { int32_t pw, pool, ox, t0, D, has_std, kind; float sq, eps; };
struct SoftmaxF32Args { int32_t n, ld_in, ld_out; };
struct MiniPosArgs { int32_t T, pw, ph, D; };
struct MiniQkvArgs { int32_t row0, D; float qscale; };
struct WindowArgs { int32_t pw, heads, hd; };
struct Gather4Args { int32_t gw, D; };

uint16_t host_f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
uint16_t host_f32_to_f16(float f) {
    const __fp16 h = (__fp16) f;      // round to nearest even (clang on arm64)
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
// STRATA_VIT_MEM=1 (a diagnostic, stderr only): the process's physical footprint (what the gate reads with `footprint`) at a point of the load
void footprint_note(const char* where) {
    static const bool on = std::getenv("STRATA_VIT_MEM") != nullptr;
    if (!on) return;
    task_vm_info_data_t vi;
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &vi, &n) == KERN_SUCCESS)
        std::fprintf(stderr, "vit-mem: %-34s footprint %.1f MiB\n", where, (double) vi.phys_footprint / 1048576.0);
}

float host_f16_to_f32(uint16_t u) {
    __fp16 h;
    std::memcpy(&h, &u, 2);
    return (float) h;
}

}  // namespace

struct VitGpuImpl {
    VitGeometry g;
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> lib = nil, lib_mpp = nil;
    std::map<std::string, id<MTLComputePipelineState>> pipes;
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    std::string fail;          // the first dispatch problem (a missing pipeline, a group too large for a pipeline): encode reports it
    size_t bytes = 0, work_bytes = 0;

    id<MTLBuffer> patch_w = nil, patch_b = nil, pos = nil, post_w = nil, post_b = nil, mm0_b = nil, mm2_b = nil;
    DevW mm0, mm2;
    std::vector<DevLayer> layers;
    std::vector<GemmaLayer> glayers;                   // gemma4v
    id<MTLBuffer> std_bias = nil, std_scale = nil;     // gemma4v, when the file has them
    DevW proj;                                         // gemma4v: mm.input_projection
    bool gemma() const { return g.projector == "gemma4v"; }
    // minicpmv4_7: the layers are `layers` (q, k, v joined into qkv, their biases into qkv_b); wm the window merger's attention (ln1, qkv, o)
    DevLayer wm;
    id<MTLBuffer> ds_ln_w = nil, ds_ln_b = nil, ds_up_b = nil, ds_down_b = nil, mmn_w = nil, mmn_b = nil, mm_up_b = nil, mm_down_b = nil;
    DevW ds_up, ds_down, mm_up, mm_down;
    // the attention's scores in float32 and its probabilities in F16 (d_P): Gemma 4 (unscaled, large scores) and MiniCPM-V (as vit_gpu.cu); the Qwen
    // tower keeps F16 scores
    bool f32_scores() const { return gemma() || g.minicpmv(); }

    // the product kernels chosen at load (the self-test): tensor op or plain, per product
    bool mpp_bf16 = false, mpp_f16 = false, mpp_s = false, mpp_pv = false;

    int cap_patches = 0;
    // work buffers (as vit_gpu.cu): d_x residual [N][D]; d_ctx context [N][D] (also the patch matrix, and the projector's output at the end); d_y one
    // chunk's product; d_a a chunk's 16-bit GEMM input; d_f16 the activated FFN / merger hidden of a chunk; q, k, v [N][D] F16; d_S the scores of a chunk
    id<MTLBuffer> d_x = nil, d_ctx = nil, d_y = nil, d_a = nil, d_f16 = nil, d_q16 = nil, d_k16 = nil, d_v16 = nil, d_S = nil, d_P = nil;
    id<MTLBuffer> d_qf = nil, d_kf = nil;     // gemma_qk32: q and k float32 [N][D]
    bool gemma_qk32 = false;                  // Gemma 4 with the float32 tensor-op product: q and k float32 for its unscaled scores
    size_t sp_elems = 0, y_elems = 0, f16_elems = 0;     // the scores of a chunk, the floats d_y holds, the 16-bit values d_f16 holds
    int kRows = 512;
    // scores per attention chunk, 32M (64 MB of F16).  With 8M a 2,256-token picture (9,024 patches) ran 58 queries a chunk and its P V products 32 groups
    // on the GPU: Qwen3.6 on the M5 Ultra 972 ms at 8M, 808 at 32M, 778 at 64M, 856 at 256M (1,900 tokens: 711, 585, 590, 594; 1,024 tokens 154, 140)
    size_t kScoreElems = (size_t) 32 << 20;
    size_t kScoreElemsLarge = (size_t) 256 << 20;    // above 16,384 patches (512 MB of F16; the SYCL tower's choice; vit_gpu.cu takes 512M on CUDA / HIP)
    // Gemma 4's float32 scores: 64M (256 MB of scores, 128 MB of F16 probabilities).  A chunk of P V products is the queries of the chunk by a head's
    // width (72), so a small chunk leaves the GPU idle; each 16M more is 96 MB more of the reader's peak.  Exact.png (8,928 patches) on the M5 Ultra, the
    // whole picture / peak footprint: 32M 1,004 ms / 0.79 GiB, 48M 944 / 0.90, 64M 920 / 0.98, 128M 880 / 1.37; poly.10's reader 965 ms / 1.80 GiB.
    // 64M keeps a few per cent of speed over poly.10's reader (48M is the smallest chunk still faster) at about half its memory.
    // MiniCPM-V reads a tile of about a slice (1,024 patches, 17M scores) at a time: the 32M chunk of the F16 scores holds it.
    size_t kScoreElemsF32 = (size_t) 64 << 20;
    bool scores_env = false;
    int n_pad = 0, q_align = 0;
    int pad_for(size_t n) const { return n_pad ? n_pad : n > 16384 ? 64 : 8; }
    int align_for(size_t n) const { return q_align ? q_align : n > 16384 ? 64 : 1; }
    size_t padded(size_t n) const { const size_t p = (size_t) pad_for(n); return (n + p - 1) / p * p; }
    bool round_out = false;
    // BF16 matrices: BF16_TO_F16 (K_HF) by default; STRATA_VIT_BF16_AS_F16=0 (a diagnostic) keeps them BF16 as stored.  The BF16-rounded kind (1) of the
    // CUDA / HIP tower has no Metal kernel: 1 reads as 0 here.
    int bf16_mode = BF16_TO_F16;

    // the mmproj file mapped read-only and wrapped as one no-copy buffer: an F16 matrix is read where it lies in the file.  Its pages are the
    // file's, which macOS can drop and read again, not memory of the process (a copy was the file's size more of the footprint)
    void* map_base = nullptr;
    size_t map_len = 0;
    id<MTLBuffer> d_file = nil;
    uint64_t file_data = 0;      // the file offset of its tensor data
    void map_file(const Gguf& gg) {
        const int fd = ::open(gg.path().c_str(), O_RDONLY);
        if (fd < 0) return;
        const size_t page = (size_t) getpagesize();
        const size_t len = ((size_t) gg.file_size() + page - 1) / page * page;
        void* p = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) return;
        d_file = [dev newBufferWithBytesNoCopy:p length:len options:MTLResourceStorageModeShared deallocator:nil];
        if (!d_file) { munmap(p, len); return; }   // copies then, as before
        map_base = p;
        map_len = len;
        file_data = gg.data_start();
    }
    ~VitGpuImpl() {
        finish();
        d_file = nil;
        if (map_base) munmap(map_base, map_len);
    }

    id<MTLBuffer> buf(size_t n_bytes, bool weight, std::string& err) {
        id<MTLBuffer> b = [dev newBufferWithLength:std::max<size_t>(n_bytes, 16) options:MTLResourceStorageModeShared];
        if (!b) { err = "GPU memory: no buffer of " + std::to_string(n_bytes) + " bytes"; return nil; }
        bytes += n_bytes;
        if (!weight) work_bytes += n_bytes;
        return b;
    }
    id<MTLBuffer> buf_from(const void* src, size_t n_bytes, std::string& err) {
        id<MTLBuffer> b = buf(n_bytes, true, err);
        if (b) std::memcpy([b contents], src, n_bytes);
        return b;
    }
    // vectors one after the other in one buffer (MiniCPM-V's q, k, v biases: one bias of the joined product)
    bool upload_f32(__strong id<MTLBuffer>& dst, const Gguf& gg, const std::vector<std::string>& names, std::string& err) {
        std::vector<const GgufTensorInfo*> ts;
        size_t n = 0;
        for (const std::string& name : names) {
            const GgufTensorInfo* t = gg.tensor(name);
            if (!t) { err = "the mmproj file lacks the tensor " + name; return false; }
            ts.push_back(t);
            n += (size_t) t->nelem();
        }
        dst = buf(n * 4, true, err);     // read in the buffer itself (as upload_matrix)
        if (!dst) return false;
        float* h = (float*) [dst contents];
        for (const GgufTensorInfo* t : ts) {
            if (!gg.read_f32(*t, h, err)) return false;
            h += t->nelem();
        }
        return true;
    }
    bool upload_f32(__strong id<MTLBuffer>& dst, const Gguf& gg, const std::string& name, std::string& err) {
        return upload_f32(dst, gg, std::vector<std::string>{name}, err);
    }
    bool upload_matrix(DevW& w, const Gguf& gg, const std::string& name, std::string& err) {
        return upload_matrix(w, gg, std::vector<std::string>{name}, err);
    }
    // matrices of one width, their rows one after the other in one matrix (MiniCPM-V's q, k, v: one product).  Their types may
    // differ (#154: a Q4_0 mmproj made by hand had v in another type than q): a mix goes through float32 like a quantized matrix.
    bool upload_matrix(DevW& w, const Gguf& gg, const std::vector<std::string>& names, std::string& err) {
        std::vector<const GgufTensorInfo*> ts;
        bool same = true, wide = false;   // wide: a matrix in F32 or BF16, whose range F16 may not hold
        for (const std::string& name : names) {
            const GgufTensorInfo* t = gg.tensor(name);
            if (!t || t->ne.size() != 2) { err = "the mmproj file lacks the matrix " + name; return false; }
            if (!ts.empty() && t->ne[0] != ts[0]->ne[0]) { err = "the matrices " + names[0] + " and " + name + " differ in width"; return false; }
            same = same && (ts.empty() || t->type == ts[0]->type);
            wide = wide || t->type == GT_F32 || t->type == GT_BF16;
            ts.push_back(t);
        }
        const GgufTensorInfo* t = ts[0];
        w.cols = (int) t->ne[0];
        w.rows = 0;
        size_t n = 0;
        for (const GgufTensorInfo* ti : ts) { w.rows += (int) ti->ne[1]; n += (size_t) ti->nelem(); }
        if (d_file && ts.size() == 1 && t->type == GT_F16) {     // read in place, in the mapped file
            w.kind = K_F16;
            w.b = d_file;
            w.off = (size_t) (file_data + t->offset);
            return true;
        }
        // written in the buffer itself: a staging copy kept a large matrix's size twice in the footprint (MiniCPM-V's 151 MB merger matrix)
        w.b = buf(n * 2, true, err);
        if (!w.b) return false;
        uint16_t* raw = (uint16_t*) [w.b contents];
        if (same && (t->type == GT_F16 || t->type == GT_BF16)) {   // as stored; a BF16 matrix as F16 (K_HF) unless a weight is beyond F16's range
            size_t at = 0;
            for (const GgufTensorInfo* ti : ts) {
                if (!gg.read_raw(*ti, raw + at, err)) return false;
                at += (size_t) ti->nelem();
            }
            w.kind = t->type == GT_F16 ? K_F16 : K_BF16;
            if (w.kind == K_BF16 && bf16_mode == BF16_TO_F16 && bf16_matrix_fits_f16(bf16_count_beyond_f16(raw, (long long) n))) {
                w.kind = K_HF;
                for (size_t i = 0; i < n; ++i) {           // BF16 -> float32 (exact) -> F16, round to nearest even
                    const uint32_t u = (uint32_t) raw[i] << 16;
                    float f;
                    std::memcpy(&f, &u, 4);
                    raw[i] = host_f32_to_f16(f);
                }
            }
        } else {                                                // F32 -> BF16, quantized -> F16; a mix with F32 or BF16 -> BF16
            std::vector<float> f(n);
            size_t at = 0;
            for (const GgufTensorInfo* ti : ts) {
                if (!gg.read_f32(*ti, f.data() + at, err)) return false;
                at += (size_t) ti->nelem();
            }
            w.kind = wide ? K_BF16 : K_F16;
            for (size_t i = 0; i < n; ++i) raw[i] = wide ? host_f32_to_bf16(f[i]) : host_f32_to_f16(f[i]);
        }
        return true;
    }

    // ---- dispatching ----
    id<MTLComputePipelineState> pipe(const std::string& fn) {
        auto it = pipes.find(fn);
        if (it != pipes.end()) return it->second;
        id<MTLLibrary> L = fn.rfind("vm_", 0) == 0 ? lib_mpp : lib;
        id<MTLFunction> f = L ? [L newFunctionWithName:[NSString stringWithUTF8String:fn.c_str()]] : nil;
        NSError* e = nil;
        id<MTLComputePipelineState> p = f ? [dev newComputePipelineStateWithFunction:f error:&e] : nil;
        pipes[fn] = p;
        return p;
    }
    id<MTLComputeCommandEncoder> encoder() {
        if (!cb) cb = [queue commandBuffer];
        if (!enc) enc = [cb computeCommandEncoder];      // serial dispatch: each dispatch sees the previous one's writes
        return enc;
    }
    void go(const std::string& fn, std::initializer_list<BO> bufs, const void* args, size_t args_bytes, MTLSize grid, NSUInteger threads) {
        if (!fail.empty()) return;
        if (grid.width == 0 || grid.height == 0 || grid.depth == 0) return;
        id<MTLComputePipelineState> p = pipe(fn);
        if (!p) { fail = "no pipeline " + fn; return; }
        // a group larger than the pipeline allows is NOT run by Metal, and no error is raised (the engine's #2323): so check every dispatch
        if ([p maxTotalThreadsPerThreadgroup] < threads) {
            fail = fn + " needs " + std::to_string(threads) + " threads a group, this GPU allows " + std::to_string((size_t) [p maxTotalThreadsPerThreadgroup]);
            return;
        }
        id<MTLComputeCommandEncoder> e = encoder();
        [e setComputePipelineState:p];
        NSUInteger i = 0;
        for (const BO& b : bufs) [e setBuffer:b.b offset:b.off atIndex:i++];
        [e setBytes:args length:args_bytes atIndex:15];
        [e dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        // STRATA_VIT_MEM=1: the footprint after the first run of each kernel (the GPU waited for), to see which one takes memory on first use
        static const bool mem = std::getenv("STRATA_VIT_MEM") != nullptr;
        if (mem && seen.insert(fn).second) {
            (void) finish();
            footprint_note(("first run of " + fn).c_str());
        }
    }
    std::set<std::string> seen;
    // commit what is encoded as its own command buffer, without waiting (one per layer: no long-running buffer)
    void split() {
        if (enc) { [enc endEncoding]; enc = nil; }
        if (cb) { [cb commit]; last = cb; cb = nil; }
    }
    id<MTLCommandBuffer> last = nil;
    // commit and wait for everything encoded so far; false (with the reason) on a GPU error
    bool finish(std::string* err = nullptr) {
        split();
        if (last) {
            [last waitUntilCompleted];
            const bool ok = [last status] == MTLCommandBufferStatusCompleted;
            if (!ok && err) {
                NSError* e = [last error];
                *err = std::string("the GPU work failed: ") + (e ? [[e localizedDescription] UTF8String] : "unknown");
            }
            last = nil;
            return ok;
        }
        return true;
    }

    // Y[z][rows][out] = X[z][rows][in] * W[z](out, in), float32 accumulation; `wn`: W is [in][out]; the product's kernel by name (plain) or tensor op
    void gemm(const char* plain, const char* mpp, BO X, BO W, BO Y, int rows, int in, int out, int ldx, int ldw, int ldy, size_t sx, size_t sw, size_t sy,
              int batch, int tm = 64) {
        const GemmArgs a{(uint32_t) rows, (uint32_t) in, (uint32_t) out, (uint32_t) ldx, (uint32_t) ldw, (uint32_t) ldy, sx, sw, sy};
        const bool use = mpp != nullptr;
        const int TM = use ? tm : 64;
        go(use ? mpp : plain, {X, W, Y}, &a, sizeof a, MTLSizeMake((out + 63) / 64, (rows + TM - 1) / TM, batch), 128);
    }
    // y[M][w.rows] (float32) = A[M][K] (w's kind) * w^T
    void wgemm(const DevW& w, BO A, int M, BO y) {
        const bool f16 = w.kind == K_F16 || w.kind == K_HF;     // K_HF: F16 weights and F16 inputs, the F16 products
        const char* plain = f16 ? "vk_gemm_f16" : "vk_gemm_bf16";
        const char* mpp = nullptr;
        if (f16 && mpp_f16) mpp = M >= 128 ? "vm_gemm_f16_128" : "vm_gemm_f16_64";
        if (w.kind == K_BF16 && mpp_bf16) mpp = M >= 128 ? "vm_gemm_bf16_128" : "vm_gemm_bf16_64";
        gemm(plain, mpp, A, BO(w.b, w.off), y, M, w.cols, w.rows, w.cols, w.cols, w.rows, 0, 0, 0, 1, M >= 128 ? 128 : 64);
        if (round_out) {
            const EltArgs e{(uint64_t) M * w.rows, (int32_t) w.kind, 0};     // (K_HF rounds to BF16 here, as vit_gpu.cu's k_round does)
            go("vk_round", {y}, &e, sizeof e, MTLSizeMake(blocks_for(e.n), 1, 1), 256);
        }
    }
    static NSUInteger blocks_for(size_t n, int per = 256) { return (NSUInteger) std::min<size_t>((n + per - 1) / per, 65535); }

    bool size_buffers(int patches, std::string& err) {
        if (patches <= cap_patches) return true;
        if (!finish(&err)) return false;
        bytes -= work_bytes;
        work_bytes = 0;
        d_x = d_ctx = d_y = d_a = d_f16 = d_q16 = d_k16 = d_v16 = d_S = d_P = d_qf = d_kf = nil;
        const size_t N = (size_t) patches, D = (size_t) g.width, F = (size_t) std::max(g.ffn, g.merger_hidden);
        const size_t K0 = (size_t) 3 * g.patch * g.patch;
        const size_t Np = padded(N);
        const size_t budget = scores_env ? kScoreElems : gemma() ? kScoreElemsF32 : N > 16384 ? kScoreElemsLarge : kScoreElems;
        sp_elems = std::min<size_t>((size_t) g.heads * N * Np, std::max<size_t>(budget, (size_t) g.heads * Np * 64));
        const size_t R = std::min<size_t>(N, (size_t) kRows);
        const size_t ctx_elems = std::max({N * D, N * K0, (N / 4) * (size_t) g.out_dim});
        y_elems = R * std::max<size_t>(3 * D, gemma() ? 2 * F : F);     // Gemma 4: gate and up side by side
        f16_elems = R * F;
        d_x = buf(N * D * 4, false, err);
        d_ctx = d_x ? buf(ctx_elems * 4, false, err) : nil;
        d_y = d_ctx ? buf(y_elems * 4, false, err) : nil;
        d_a = d_y ? buf(R * D * 2, false, err) : nil;
        d_f16 = d_a ? buf(f16_elems * 2, false, err) : nil;
        d_q16 = d_f16 ? buf(N * D * 2, false, err) : nil;
        d_k16 = d_q16 ? buf(N * D * 2, false, err) : nil;
        d_v16 = d_k16 ? buf(N * D * 2, false, err) : nil;
        if (!d_v16) { cap_patches = 0; return false; }
        // f32_scores(): the scores in float32 (d_S) and their F16 probabilities (d_P); else F16 scores in d_S, the probabilities in place
        const size_t sb = f32_scores() ? 4 : 2;
        d_S = buf(sp_elems * sb, false, err);
        if (!d_S && budget != kScoreElems) {      // no room for the large chunk: the default chunk instead, slower, never an error
            err.clear();
            sp_elems = std::min<size_t>((size_t) g.heads * N * Np, std::max<size_t>(kScoreElems, (size_t) g.heads * Np * 64));
            d_S = buf(sp_elems * sb, false, err);
        }
        if (!d_S) { cap_patches = 0; return false; }
        if (f32_scores()) {
            d_P = buf(sp_elems * 2, false, err);
            if (!d_P) { cap_patches = 0; return false; }
        }
        if (gemma_qk32) {
            d_qf = buf(N * D * 4, false, err);
            d_kf = d_qf ? buf(N * D * 4, false, err) : nil;
            if (!d_kf) { cap_patches = 0; return false; }
        }
        cap_patches = patches;
        return true;
    }

    // softmax(q k^T) v of every head over n tokens into d_ctx (float32), q, k, v F16 [n][D] in d_q16, d_k16, d_v16 (any scale already in q): the scores
    // float32 (16-bit inputs, float32 sums), the softmax float32, the probabilities F16 (d_P) for the product with v, float32 sums.  Gemma 4 (no scale:
    // its scores are large and an F16 score loses the softmax's precision) and MiniCPM-V, as vit_gpu.cu's attend() keeps them.
    void attend_f32(int n) {
        const int D = g.width, H = g.heads, HD = g.head_dim, Np = (int) padded((size_t) n);
        int QC = (int) std::max<size_t>(1, std::min<size_t>((size_t) n, sp_elems / ((size_t) H * Np)));
        const int qa = align_for((size_t) n);
        if (qa > 1 && QC < n && QC >= qa) QC -= QC % qa;
        for (int q0 = 0; q0 < n; q0 += QC) {
            const int qc = std::min(QC, n - q0);
            if (gemma_qk32)     // q and k float32: the float32 tensor-op product
                gemm("vk_gemm_f32", "vm_gemm_f32_64", BO(d_qf, (size_t) q0 * D * 4), BO(d_kf), BO(d_S), qc, HD, n, D, D, Np, (size_t) HD, (size_t) HD,
                     (size_t) qc * Np, H);
            else
                gemm("vk_gemm_f16", mpp_f16 ? "vm_gemm_f16_64" : nullptr, BO(d_q16, (size_t) q0 * D * 2), BO(d_k16), BO(d_S), qc, HD, n, D, D, Np,
                     (size_t) HD, (size_t) HD, (size_t) qc * Np, H);
            const SoftmaxF32Args sa{n, Np, Np};
            go("vk_softmax_f32_to16", {BO(d_S), BO(d_P)}, &sa, sizeof sa, MTLSizeMake((size_t) H * qc, 1, 1), 256);
            gemm("vk_gemm_pv", mpp_pv ? "vm_gemm_pv" : nullptr, BO(d_P), BO(d_v16), BO(d_ctx, (size_t) q0 * D * 4), qc, n, HD, Np, D, D, (size_t) qc * Np,
                 (size_t) HD, (size_t) HD, H);
        }
    }

    // Gemma 4's weights (as load() reads the Qwen ones): the patch embedding float32 (no bias), the position tables, the projection, the layers
    bool load_gemma(const Gguf& gg, std::string& err) {
        if (!upload_f32(patch_w, gg, "v.patch_embd.weight", err) || !upload_f32(pos, gg, "v.position_embd.weight", err) ||
            !upload_matrix(proj, gg, "mm.input_projection.weight", err))
            return false;
        if (g.std_norm && (!upload_f32(std_bias, gg, "v.std_bias", err) || !upload_f32(std_scale, gg, "v.std_scale", err))) return false;
        glayers.assign((size_t) g.layers, GemmaLayer{});
        for (int il = 0; il < g.layers; ++il) {
            GemmaLayer& L = glayers[(size_t) il];
            const std::string p = "v.blk." + std::to_string(il) + ".";
            if (!upload_f32(L.ln1, gg, p + "ln1.weight", err) || !upload_f32(L.ln2, gg, p + "ln2.weight", err) ||
                !upload_f32(L.attn_post, gg, p + "attn_post_norm.weight", err) || !upload_f32(L.ffn_post, gg, p + "ffn_post_norm.weight", err) ||
                !upload_f32(L.q_norm, gg, p + "attn_q_norm.weight", err) || !upload_f32(L.k_norm, gg, p + "attn_k_norm.weight", err) ||
                !upload_matrix(L.q, gg, p + "attn_q.weight", err) || !upload_matrix(L.k, gg, p + "attn_k.weight", err) ||
                !upload_matrix(L.v, gg, p + "attn_v.weight", err) || !upload_matrix(L.o, gg, p + "attn_out.weight", err) ||
                !upload_matrix(L.gate, gg, p + "ffn_gate.weight", err) || !upload_matrix(L.up, gg, p + "ffn_up.weight", err) ||
                !upload_matrix(L.down, gg, p + "ffn_down.weight", err))
                return false;
            // the q, k, v products share one input (the normed row in the kind of the first): the three must be of one kind, and gate and up too
            if (L.q.kind != L.k.kind || L.q.kind != L.v.kind || L.gate.kind != L.up.kind) { err = "a gemma4v layer with matrices of mixed types, which this reader does not read"; return false; }
        }
        return true;
    }

    // Gemma 4's tower, the Metal counterpart of VitCpu::encode_gemma4 (and vit_gpu.cu's encode_gemma4): the pixels as 2 v - 1 in reading order, the x and y
    // position tables, per layer: RMS norm, the q / k / v products, the per-head norms and the rotary (F16), the attention without a scale (attend_f32:
    // float32 scores), the output, its post norm, the residual, the same around a quick-GELU gated FFN; then the average of each pool x pool
    // square times sqrt(width), the standardisation, an RMS norm and the projection.  The grid is reported as one line (nx = rows, ny = 1).
    bool encode_gemma(const PreparedImage& img, std::vector<float>& rows, int& nx, int& ny, std::string& err) {
        if (glayers.empty()) { err = "the tower is not loaded"; return false; }
        const int P = g.patch, D = g.width, H = g.heads, HD = g.head_dim, F = g.ffn, K = g.pool, T = g.pos_grid;
        if (img.w <= 0 || img.h <= 0 || img.w % (P * K) != 0 || img.h % (P * K) != 0) { err = "the picture size is not a multiple of patch * pool"; return false; }
        if (H > 32 || HD > 128 || HD % 4 != 0 || D > 4096) { err = "this tower's heads or width are not handled by the GPU kernels"; return false; }
        const int pw = img.w / P, ph = img.h / P, N = pw * ph;
        if (pw > T || ph > T) { err = "the picture has more patches a side than the position tables"; return false; }
        if (!size_buffers(N, err)) return false;
        fail.clear();
        // STRATA_VIT_PROF=1 (a diagnostic, stderr only, off by default): the time of each stage, the GPU waited for after every stage
        static const bool prof = std::getenv("STRATA_VIT_PROF") != nullptr;
        double pt[5] = {0, 0, 0, 0, 0};
        auto pclock = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        double pmark = prof ? pclock() : 0.0;
        auto stage = [&](int i) {
            if (!prof) return;
            (void) finish();
            const double now = pclock();
            pt[i] += now - pmark;
            pmark = now;
        };
        const int K0 = 3 * P * P;
        {   // the patch matrix [N][K0] in reading order, written by the host into the context buffer (the GPU is idle here)
            float* pm = (float*) [d_ctx contents];
            const size_t plane = (size_t) img.w * img.h;
            for (int t = 0; t < N; ++t) {
                const int py = t / pw, px = t % pw;
                float* a = pm + (size_t) t * K0;
                for (int c = 0; c < 3; ++c)
                    for (int ky = 0; ky < P; ++ky) {
                        const float* src = &img.chw[c * plane + (size_t) (py * P + ky) * img.w + (size_t) px * P];
                        float* dst = a + (size_t) c * P * P + (size_t) ky * P;
                        for (int kx = 0; kx < P; ++kx) dst[kx] = src[kx] * 2.0f - 1.0f;
                    }
            }
        }
        {
            const GemmArgs a{(uint32_t) N, (uint32_t) K0, (uint32_t) D, (uint32_t) K0, (uint32_t) K0, (uint32_t) D, 0, 0, 0};
            go("vk_gemm_seq_f32", {BO(d_ctx), BO(patch_w), BO(d_x)}, &a, sizeof a, MTLSizeMake((D + 63) / 64, N, 1), 64);
        }
        {
            const GemmaPosArgs pa{T, pw, D};
            go("vk_gemma_pos", {BO(d_x), BO(pos)}, &pa, sizeof pa, MTLSizeMake(N, 1, 1), 256);
        }
        stage(0);
        const float eps = g.eps;
        const float theta_scale = std::pow(g.rope_theta, -2.0f / (float) (HD / 2));
        const int R = kRows;
        for (const GemmaLayer& L : glayers) {
            for (int r0 = 0; r0 < N; r0 += R) {
                const int rn = std::min(R, N - r0);
                const RmsArgs na{D, (int32_t) L.q.kind, 1, eps};
                go("vk_rms_cast", {BO(d_x, (size_t) r0 * D * 4), BO(L.ln1), BO(d_a)}, &na, sizeof na, MTLSizeMake(rn, 1, 1), 256);
                wgemm(L.q, BO(d_a), rn, BO(d_y));
                wgemm(L.k, BO(d_a), rn, BO(d_y, (size_t) rn * D * 4));
                wgemm(L.v, BO(d_a), rn, BO(d_y, (size_t) 2 * rn * D * 4));
                const GemmaQkvArgs qa{r0, pw, H, HD, theta_scale, eps};
                go(gemma_qk32 ? "vk_gemma_qkv32" : "vk_gemma_qkv", {BO(d_y), BO(d_y, (size_t) rn * D * 4), BO(d_y, (size_t) 2 * rn * D * 4), BO(L.q_norm),
                   BO(L.k_norm), BO(gemma_qk32 ? d_qf : d_q16), BO(gemma_qk32 ? d_kf : d_k16), BO(d_v16)}, &qa, sizeof qa, MTLSizeMake(rn, 1, 1), 256);
            }
            stage(1);
            attend_f32(N);      // no scale: q is as the norm and the rotary leave it
            stage(2);
            for (int r0 = 0; r0 < N; r0 += R) {
                const int rn = std::min(R, N - r0);
                const EltArgs ca{(uint64_t) rn * D, (int32_t) L.o.kind, D};
                go("vk_cast", {BO(d_ctx, (size_t) r0 * D * 4), BO(d_a)}, &ca, sizeof ca, MTLSizeMake(VitGpuImpl::blocks_for(ca.n), 1, 1), 256);
                wgemm(L.o, BO(d_a), rn, BO(d_y));
                const RmsArgs pa{D, 0, 1, eps};
                go("vk_rms_add", {BO(d_x, (size_t) r0 * D * 4), BO(d_y), BO(L.attn_post)}, &pa, sizeof pa, MTLSizeMake(rn, 1, 1), 256);
                const RmsArgs fa{D, (int32_t) L.gate.kind, 1, eps};
                go("vk_rms_cast", {BO(d_x, (size_t) r0 * D * 4), BO(L.ln2), BO(d_a)}, &fa, sizeof fa, MTLSizeMake(rn, 1, 1), 256);
                wgemm(L.gate, BO(d_a), rn, BO(d_y));
                wgemm(L.up, BO(d_a), rn, BO(d_y, (size_t) rn * F * 4));
                const EltArgs ga{(uint64_t) rn * F, (int32_t) L.down.kind, F};
                go("vk_geglu_quick_cast", {BO(d_y), BO(d_y, (size_t) rn * F * 4), BO(d_f16)}, &ga, sizeof ga, MTLSizeMake(VitGpuImpl::blocks_for(ga.n), 1, 1), 256);
                wgemm(L.down, BO(d_f16), rn, BO(d_y));
                go("vk_rms_add", {BO(d_x, (size_t) r0 * D * 4), BO(d_y), BO(L.ffn_post)}, &pa, sizeof pa, MTLSizeMake(rn, 1, 1), 256);
            }
            stage(3);
            split();      // each layer is its own command buffer, as the Qwen tower's
        }
        const int ox = pw / K, oy = ph / K, M = ox * oy;
        const float sq = std::sqrt((float) D);
        for (int r0 = 0; r0 < M; r0 += R) {
            const int rn = std::min(R, M - r0);
            const GemmaPoolArgs pa{pw, K, ox, r0, D, std_bias ? 1 : 0, (int32_t) proj.kind, sq, eps};
            go("vk_gemma_pool", {BO(d_x), BO(std_bias ? std_bias : d_x), BO(std_scale ? std_scale : d_x), BO(d_a)}, &pa, sizeof pa, MTLSizeMake(rn, 1, 1), 256);
            wgemm(proj, BO(d_a), rn, BO(d_ctx, (size_t) r0 * g.out_dim * 4));
        }
        std::string ferr;
        const bool ok = finish(&ferr);
        if (!fail.empty()) { err = fail; fail.clear(); return false; }
        if (!ok) { err = ferr; return false; }
        stage(4);
        if (prof)
            std::fprintf(stderr, "vit-prof gemma4 N=%d: patches %.2f, norm+qkv %.2f, attention %.2f, o-proj+ffn %.2f, pool+projection %.2f ms\n", N, pt[0], pt[1],
                         pt[2], pt[3], pt[4]);
        rows.resize((size_t) M * g.out_dim);
        std::memcpy(rows.data(), [d_ctx contents], rows.size() * 4);
        malloc_zone_pressure_relief(nullptr, 0);
        nx = M;
        ny = 1;
        return true;
    }

    // MiniCPM-V 4.7's weights (vit_gpu.cu's load_minicpmv): q, k, v joined into one matrix and their biases into one vector, as the CUDA tower has them
    bool load_minicpm(const Gguf& gg, std::string& err) {
        auto attn = [&](const std::string& p, DevLayer& L) {
            return upload_f32(L.ln1_w, gg, p + "ln1.weight", err) && upload_f32(L.ln1_b, gg, p + "ln1.bias", err) &&
                   upload_matrix(L.qkv, gg, std::vector<std::string>{p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"}, err) &&
                   upload_f32(L.qkv_b, gg, std::vector<std::string>{p + "attn_q.bias", p + "attn_k.bias", p + "attn_v.bias"}, err) &&
                   upload_matrix(L.o, gg, p + "attn_out.weight", err) && upload_f32(L.o_b, gg, p + "attn_out.bias", err);
        };
        if (!upload_f32(patch_w, gg, "v.patch_embd.weight", err) || !upload_f32(patch_b, gg, "v.patch_embd.bias", err) ||
            !upload_f32(pos, gg, "v.position_embd.weight", err) || !upload_f32(post_w, gg, "v.post_ln.weight", err) ||
            !upload_f32(post_b, gg, "v.post_ln.bias", err) || !attn("v.vit_merger.", wm) || !upload_f32(ds_ln_w, gg, "v.vit_merger.ds_ln.weight", err) ||
            !upload_f32(ds_ln_b, gg, "v.vit_merger.ds_ln.bias", err) || !upload_matrix(ds_up, gg, "v.vit_merger.ds_ffn_up.weight", err) ||
            !upload_f32(ds_up_b, gg, "v.vit_merger.ds_ffn_up.bias", err) || !upload_matrix(ds_down, gg, "v.vit_merger.ds_ffn_down.weight", err) ||
            !upload_f32(ds_down_b, gg, "v.vit_merger.ds_ffn_down.bias", err) || !upload_f32(mmn_w, gg, "mm.input_norm.weight", err) ||
            !upload_f32(mmn_b, gg, "mm.input_norm.bias", err) || !upload_matrix(mm_up, gg, "mm.up.weight", err) || !upload_f32(mm_up_b, gg, "mm.up.bias", err) ||
            !upload_matrix(mm_down, gg, "mm.down.weight", err) || !upload_f32(mm_down_b, gg, "mm.down.bias", err))
            return false;
        layers.assign((size_t) g.layers, DevLayer{});
        for (int il = 0; il < g.layers; ++il) {
            DevLayer& L = layers[(size_t) il];
            const std::string p = "v.blk." + std::to_string(il) + ".";
            if (!attn(p, L) || !upload_f32(L.ln2_w, gg, p + "ln2.weight", err) || !upload_f32(L.ln2_b, gg, p + "ln2.bias", err) ||
                !upload_matrix(L.up, gg, p + "ffn_up.weight", err) || !upload_f32(L.up_b, gg, p + "ffn_up.bias", err) ||
                !upload_matrix(L.down, gg, p + "ffn_down.weight", err) || !upload_f32(L.down_b, gg, p + "ffn_down.bias", err))
                return false;
        }
        return true;
    }

    // MiniCPM-V 4.7's tower on one tile, the Metal counterpart of VitCpu::encode_minicpmv (and vit_gpu.cu's encode_minicpmv): the products with 16-bit
    // inputs of the weight's kind and float32 sums, the attention with float32 scores and F16 probabilities, the window merger's 4-patch attention by
    // vk_window_attn.  d_ctx holds in turn the patch matrix, the attention context, the downsample's and the final merger's gathered rows (4 D floats a
    // row: N / 4 and N / 16 rows), and the output rows (over the gathered rows the last chunk no longer reads); d_y the post norm's float32 rows before
    // the final gather.
    bool encode_minicpm(const PreparedImage& img, std::vector<float>& rows, int& nx, int& ny, std::string& err) {
        if (layers.empty()) { err = "the tower is not loaded"; return false; }
        const int P = g.patch, D = g.width, H = g.heads, HD = g.head_dim, F = g.ffn, T = g.pos_grid, D4 = 4 * D;
        if (img.w <= 0 || img.h <= 0 || img.w % (P * 4) != 0 || img.h % (P * 4) != 0) { err = "the tile size is not a multiple of patch * 4"; return false; }
        if (H > 32 || HD % 4 != 0) { err = "this tower's heads are not handled by the GPU kernels"; return false; }
        const int pw = img.w / P, ph = img.h / P, N = pw * ph, hw = pw / 2, hh = ph / 2, N2 = hw * hh, M = (hw / 2) * (hh / 2);
        if (!size_buffers(N, err)) return false;
        fail.clear();
        const int R = std::min(N, kRows);
        // the wide products (the downsample's up to ds_ffn, the merger's up) and their 4 D-wide norms run in chunks that fit d_y, d_f16 and d_a
        const int RC = (int) std::max<size_t>(1, std::min({(size_t) R / 4, y_elems / (size_t) std::max(g.ds_ffn, g.merger_hidden), f16_elems / (size_t) g.ds_ffn}));
        if ((size_t) N2 * D > y_elems) { err = "the tile is larger than the work buffers"; return false; }
        const int K0 = 3 * P * P;
        {   // the patch matrix [N][K0] in reading order, written by the host into the context buffer (the GPU is idle here)
            float* pm = (float*) [d_ctx contents];
            const size_t plane = (size_t) img.w * img.h;
            for (int t = 0; t < N; ++t) {
                const int py = t / pw, px = t % pw;
                float* a = pm + (size_t) t * K0;
                for (int c = 0; c < 3; ++c)
                    for (int ky = 0; ky < P; ++ky)
                        std::memcpy(a + (size_t) c * P * P + (size_t) ky * P, &img.chw[c * plane + (size_t) (py * P + ky) * img.w + (size_t) px * P], (size_t) P * sizeof(float));
            }
        }
        {
            const GemmArgs a{(uint32_t) N, (uint32_t) K0, (uint32_t) D, (uint32_t) K0, (uint32_t) K0, (uint32_t) D, 0, 0, 0};
            go("vk_gemm_seq_f32", {BO(d_ctx), BO(patch_w), BO(d_x)}, &a, sizeof a, MTLSizeMake((D + 63) / 64, N, 1), 64);
            const MiniPosArgs pa{T, pw, ph, D};
            go("vk_mini_pos", {BO(d_x), BO(patch_b), BO(pos)}, &pa, sizeof pa, MTLSizeMake(N, 1, 1), 256);
        }
        const float qscale = 1.0f / std::sqrt((float) HD);
        auto project = [&](const DevLayer& L, int n) {      // LN1, the joined q / k / v product, its bias, q scaled, into F16
            for (int r0 = 0; r0 < n; r0 += R) {
                const int rn = std::min(R, n - r0);
                const NormArgs na{D, (int32_t) L.qkv.kind, g.eps};
                go("vk_layernorm", {BO(d_x, (size_t) r0 * D * 4), BO(L.ln1_w), BO(L.ln1_b), BO(d_a)}, &na, sizeof na, MTLSizeMake(rn, 1, 1), 256);
                wgemm(L.qkv, BO(d_a), rn, BO(d_y));
                const MiniQkvArgs qa{r0, D, qscale};
                go("vk_mini_qkv", {BO(d_y), BO(L.qkv_b), BO(d_q16), BO(d_k16), BO(d_v16)}, &qa, sizeof qa, MTLSizeMake(rn, 1, 1), 256);
            }
        };
        auto out_proj = [&](const DevLayer& L, int n) {     // x += o(ctx)
            for (int r0 = 0; r0 < n; r0 += R) {
                const int rn = std::min(R, n - r0);
                const EltArgs ca{(uint64_t) rn * D, (int32_t) L.o.kind, D};
                go("vk_cast", {BO(d_ctx, (size_t) r0 * D * 4), BO(d_a)}, &ca, sizeof ca, MTLSizeMake(blocks_for(ca.n), 1, 1), 256);
                wgemm(L.o, BO(d_a), rn, BO(d_y));
                const EltArgs ra{0, 0, D};
                go("vk_add_bias_res", {BO(d_x, (size_t) r0 * D * 4), BO(d_y), BO(L.o_b)}, &ra, sizeof ra, MTLSizeMake(rn, 1, 1), 256);
            }
        };
        auto layer = [&](const DevLayer& L, int n) {
            project(L, n);
            attend_f32(n);
            out_proj(L, n);
            for (int r0 = 0; r0 < n; r0 += R) {      // x += down(gelu(up(LN2 x)))
                const int rn = std::min(R, n - r0);
                const NormArgs na{D, (int32_t) L.up.kind, g.eps};
                go("vk_layernorm", {BO(d_x, (size_t) r0 * D * 4), BO(L.ln2_w), BO(L.ln2_b), BO(d_a)}, &na, sizeof na, MTLSizeMake(rn, 1, 1), 256);
                wgemm(L.up, BO(d_a), rn, BO(d_y));
                const EltArgs ga{0, (int32_t) L.down.kind, F};
                go("vk_bias_gelu_cast", {BO(d_y), BO(L.up_b), BO(d_f16)}, &ga, sizeof ga, MTLSizeMake(rn, 1, 1), 256);
                wgemm(L.down, BO(d_f16), rn, BO(d_y));
                const EltArgs ra{0, 0, D};
                go("vk_add_bias_res", {BO(d_x, (size_t) r0 * D * 4), BO(d_y), BO(L.down_b)}, &ra, sizeof ra, MTLSizeMake(rn, 1, 1), 256);
            }
            split();      // each layer is its own command buffer, as the Qwen tower's
        };
        // the four rows of each 2x2 window side by side into d_ctx, through LN (4 D wide), up, the activation and down: the downsample (tanh GELU, then
        // the four rows' mean added) or the final merger (erf GELU, into the output rows)
        auto merge4 = [&](BO src, int gw, int m, id<MTLBuffer> lw, id<MTLBuffer> lb, const DevW& wu, id<MTLBuffer> bu, const DevW& wd, id<MTLBuffer> bd, bool last) {
            const Gather4Args ga{gw, D};
            go("vk_gather4", {src, BO(d_ctx)}, &ga, sizeof ga, MTLSizeMake(m, 1, 1), 256);
            for (int r0 = 0; r0 < m; r0 += RC) {
                const int rn = std::min(RC, m - r0);
                const NormArgs na{D4, (int32_t) wu.kind, g.eps};
                go("vk_layernorm", {BO(d_ctx, (size_t) r0 * D4 * 4), BO(lw), BO(lb), BO(d_a)}, &na, sizeof na, MTLSizeMake(rn, 1, 1), 256);
                wgemm(wu, BO(d_a), rn, BO(d_y));
                const EltArgs ea{0, (int32_t) wd.kind, wu.rows};
                go(last ? "vk_bias_gelu_erf_cast" : "vk_bias_gelu_cast", {BO(d_y), BO(bu), BO(d_f16)}, &ea, sizeof ea, MTLSizeMake(rn, 1, 1), 256);
                if (last) {
                    wgemm(wd, BO(d_f16), rn, BO(d_ctx, (size_t) r0 * g.out_dim * 4));
                    const EltArgs ba{0, 0, g.out_dim};
                    go("vk_add_bias", {BO(d_ctx, (size_t) r0 * g.out_dim * 4), BO(bd)}, &ba, sizeof ba, MTLSizeMake(rn, 1, 1), 256);
                } else {
                    wgemm(wd, BO(d_f16), rn, BO(d_y));
                    const EltArgs oa{0, 0, D};
                    go("vk_ds_out", {BO(d_x, (size_t) r0 * D * 4), BO(d_y), BO(bd), BO(d_ctx, (size_t) r0 * D4 * 4)}, &oa, sizeof oa, MTLSizeMake(rn, 1, 1), 256);
                }
            }
        };

        for (int il = 0; il <= g.insert_layer; ++il) layer(layers[(size_t) il], N);
        // the window merger: attention inside each 2x2 window, its residual; then the downsample into the first N2 rows of x
        project(wm, N);
        {
            const WindowArgs wa{pw, H, HD};
            go("vk_window_attn", {BO(d_q16), BO(d_k16), BO(d_v16), BO(d_ctx)}, &wa, sizeof wa, MTLSizeMake(N2, 1, 1), 32 * H);
        }
        out_proj(wm, N);
        merge4(BO(d_x), pw, N2, ds_ln_w, ds_ln_b, ds_up, ds_up_b, ds_down, ds_down_b, false);
        split();
        for (int il = g.insert_layer + 1; il < g.layers; ++il) layer(layers[(size_t) il], N2);
        // the post norm (float32, into d_y), then the final 2x2 merger into the output rows
        {
            const NormArgs na{D, (int32_t) K_F32, g.eps};
            go("vk_layernorm", {BO(d_x), BO(post_w), BO(post_b), BO(d_y)}, &na, sizeof na, MTLSizeMake(N2, 1, 1), 256);
        }
        merge4(BO(d_y), hw, M, mmn_w, mmn_b, mm_up, mm_up_b, mm_down, mm_down_b, true);
        std::string ferr;
        const bool ok = finish(&ferr);
        if (!fail.empty()) { err = fail; fail.clear(); return false; }
        if (!ok) { err = ferr; return false; }
        rows.resize((size_t) M * g.out_dim);
        std::memcpy(rows.data(), [d_ctx contents], rows.size() * 4);
        malloc_zone_pressure_relief(nullptr, 0);
        nx = hw / 2;
        ny = hh / 2;
        return true;
    }

    // The tensor-op products against the plain kernel on small odd sizes (a batch of 2, strides wider than the rows): true when every value agrees within
    // float32 rounding of a different summation order.  One check per product kind.  ti 0: float32 inputs, held to float32 products (a tensor op that
    // rounds its inputs, to TF32 or F16, fails it).
    bool mpp_check(const char* plain, const char* mpp, int ti /* 0 float32, 1 F16, 2 BF16 */, bool wn, bool out_half, std::string& why) {
        const int M = 70, N = 90, K = 100, LDX = 104, LDW = wn ? 136 : 104, LDY = 96, B = 2;
        const size_t sx = (size_t) M * LDX, sw = (size_t) (wn ? K : N) * LDW, sy = (size_t) M * LDY, es = ti == 0 ? 4 : 2;
        std::string e;
        id<MTLBuffer> bx = buf(sx * B * es, false, e), bw = buf(sw * B * es, false, e), y0 = buf(sy * B * 4, false, e), y1 = buf(sy * B * 4, false, e);
        if (!bx || !bw || !y0 || !y1) { why = "no memory for the self-test"; return false; }
        uint32_t seed = 12345u;
        auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (float) ((seed >> 8) & 0xffffu) / 32768.0f - 1.0f; };
        auto put = [&](id<MTLBuffer> b, size_t i) {
            const float f = ti == 0 ? rnd() * 1.000123f : rnd();       // float32 inputs: values a 16-bit type cannot hold
            if (ti == 0) ((float*) [b contents])[i] = f;
            else ((uint16_t*) [b contents])[i] = ti == 1 ? host_f32_to_f16(f) : host_f32_to_bf16(f);
        };
        for (size_t i = 0; i < sx * B; ++i) put(bx, i);
        for (size_t i = 0; i < sw * B; ++i) put(bw, i);
        std::memset([y0 contents], 0, sy * B * 4);
        std::memset([y1 contents], 0, sy * B * 4);
        // the expected result on the host (double sums of the inputs), written in the output's type into y0.  Not the plain GPU kernel: its first
        // run costs the process ~136 MB of GPU-owned memory (D5), which a Mac with the tensor ops would otherwise never pay
        (void) plain;
        {
            auto in = [&](id<MTLBuffer> b, size_t i) {
                if (ti == 0) return (double) ((const float*) [b contents])[i];
                const uint16_t u = ((const uint16_t*) [b contents])[i];
                if (ti == 1) return (double) host_f16_to_f32(u);
                const uint32_t v = (uint32_t) u << 16;
                float f;
                std::memcpy(&f, &v, 4);
                return (double) f;
            };
            for (int z = 0; z < B; ++z)
                for (int r = 0; r < M; ++r)
                    for (int c = 0; c < N; ++c) {
                        double s = 0.0;
                        for (int k = 0; k < K; ++k) {
                            const double x = in(bx, (size_t) z * sx + (size_t) r * LDX + k);
                            const double w = wn ? in(bw, (size_t) z * sw + (size_t) k * LDW + c) : in(bw, (size_t) z * sw + (size_t) c * LDW + k);
                            s += x * w;
                        }
                        const size_t i = (size_t) z * sy + (size_t) r * LDY + c;
                        if (out_half) ((uint16_t*) [y0 contents])[i] = host_f32_to_f16((float) s);
                        else ((float*) [y0 contents])[i] = (float) s;
                    }
        }
        gemm(plain, mpp, BO(bx), BO(bw), BO(y1), M, K, N, LDX, LDW, LDY, sx, sw, sy, B, 64);
        std::string ferr;
        const bool ran = fail.empty() && finish(&ferr);
        bytes -= (sx * B * es + sw * B * es + sy * B * 8);
        work_bytes -= (sx * B * es + sw * B * es + sy * B * 8);
        if (!ran) { why = fail.empty() ? ferr : fail; fail.clear(); return false; }
        double worst = 0.0;
        for (int z = 0; z < B; ++z)
            for (int r = 0; r < M; ++r)
                for (int c = 0; c < N; ++c) {
                    const size_t i = (size_t) z * sy + (size_t) r * LDY + c;
                    float a, b;
                    if (out_half) { a = host_f16_to_f32(((uint16_t*) [y0 contents])[i]); b = host_f16_to_f32(((uint16_t*) [y1 contents])[i]); }
                    else { a = ((float*) [y0 contents])[i]; b = ((float*) [y1 contents])[i]; }
                    const double d = std::fabs((double) a - (double) b);
                    worst = std::max(worst, d);
                    const double tol = out_half ? 1e-2 + 2e-3 * std::fabs((double) a) : ti == 0 ? 1e-4 + 1e-5 * std::fabs((double) a) : 1e-3 + 1e-4 * std::fabs((double) a);
                    if (!(d <= tol)) {
                        char m[160];
                        std::snprintf(m, sizeof m, "differs from the exact product (%g against %g at %d,%d,%d)", (double) b, (double) a, z, r, c);
                        why = m;
                        return false;
                    }
                }
        return true;
    }
};

VitGpu::VitGpu() : impl_(new VitGpuImpl()) {}
VitGpu::~VitGpu() = default;
size_t VitGpu::device_bytes() const { return impl_->bytes; }

bool VitGpu::available(std::string& why) {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        if (!d) { why = "no Metal GPU"; return false; }
        return true;
    }
}

std::string VitGpu::arch_name() { return ""; }      // (only the HIP reader checks its build's architectures)

bool VitGpu::load(const Gguf& gg, const VitGeometry& g, int device, int max_patches, std::string& err) {
    @autoreleasepool {
        VitGpuImpl& m = *impl_;
        (void) device;      // one GPU on a Mac
        std::string why;
        // the Qwen family, Gemma 4 and MiniCPM-V 4.7 (vit_geometry_qwen refuses the other two, which this tower reads as vit_gpu.cu does)
        if (!vit_geometry_supported(g, why) || (g.projector != "gemma4v" && !g.minicpmv() && !vit_geometry_qwen(g, why))) {
            err = "this tower is not supported: " + why;
            return false;
        }
        m.g = g;
        footprint_note("start of the load");
        m.dev = MTLCreateSystemDefaultDevice();
        if (!m.dev) { err = "no Metal GPU"; return false; }
        m.queue = [m.dev newCommandQueue];
        footprint_note("device and queue");
        // the kernels: safe maths (no fast-math), compiled from the embedded source
        {
            MTLCompileOptions* o = [MTLCompileOptions new];
            if (@available(macOS 15.0, *)) o.mathMode = MTLMathModeSafe;
            else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                o.fastMathEnabled = NO;     // macOS before 15: the only switch there (fast maths is the default)
#pragma clang diagnostic pop
            }
            NSError* e = nil;
            m.lib = [m.dev newLibraryWithSource:[NSString stringWithUTF8String:kVitMetalSrc] options:o error:&e];
            if (!m.lib) { err = std::string("the Metal kernels do not compile: ") + (e ? [[e localizedDescription] UTF8String] : "unknown"); return false; }
        }
        footprint_note("plain kernels compiled");
        if (std::getenv("STRATA_VIT_MEM")) {     // diagnostic: does any first dispatch cost the ~250 MB, or only the product kernels' (D5)?
            std::string e;
            id<MTLBuffer> t = m.buf(4096, false, e), tb = m.buf(4096, false, e);
            const EltArgs a{0, 0, 16};
            m.go("vk_add_bias", {BO(t), BO(tb)}, &a, sizeof a, MTLSizeMake(4, 1, 1), 32);
            (void) m.finish();
            footprint_note("after one tiny vk_add_bias");
            m.bytes -= 8192; m.work_bytes -= 8192;
        }
        // the tensor-op products: Metal 4 only, then per product a self-test against the plain kernel
        std::string mpp_why;
        const bool no_mpp = std::getenv("STRATA_VIT_NO_MPP") != nullptr;
        bool metal4 = false;
        if (@available(macOS 26.0, *)) metal4 = [m.dev supportsFamily:MTLGPUFamilyMetal4];
        if (no_mpp) mpp_why = "STRATA_VIT_NO_MPP";
        else if (!metal4) mpp_why = "no Metal 4 on this GPU or macOS";
        else {
            MTLCompileOptions* o = [MTLCompileOptions new];
            if (@available(macOS 15.0, *)) o.mathMode = MTLMathModeSafe;
            NSError* e = nil;
            m.lib_mpp = [m.dev newLibraryWithSource:[NSString stringWithUTF8String:kVitMetalMppSrc] options:o error:&e];
            if (!m.lib_mpp) mpp_why = std::string("the tensor-op kernels do not compile (") + (e ? std::string([[e localizedDescription] UTF8String]).substr(0, 160) : "unknown") + ")";
        }
        std::string w_bf, w_f16, w_s, w_pv, w_f32;
        if (m.lib_mpp) {
            m.mpp_bf16 = m.mpp_check("vk_gemm_bf16", "vm_gemm_bf16_64", 2, false, false, w_bf);
            m.mpp_f16 = m.mpp_check("vk_gemm_f16", "vm_gemm_f16_64", 1, false, false, w_f16);
            m.mpp_s = m.mpp_check("vk_gemm_s", "vm_gemm_s", 1, false, true, w_s);
            m.mpp_pv = m.mpp_check("vk_gemm_pv", "vm_gemm_pv", 1, true, false, w_pv);
            if (m.gemma()) m.gemma_qk32 = m.mpp_check("vk_gemm_f32", "vm_gemm_f32_64", 0, false, false, w_f32);
        }
        if (m.gemma() && std::getenv("STRATA_VIT_QK16")) { m.gemma_qk32 = false; w_f32 = "STRATA_VIT_QK16"; }     // a diagnostic: Gemma 4's q and k in F16
        if (m.gemma())
            std::fprintf(stderr, "strata-vision: Gemma 4's q and k %s\n", m.gemma_qk32 ? "float32 (float32 tensor-op scores)" : ("F16 (" + (w_f32.empty() ? mpp_why : w_f32) + ")").c_str());
        // diagnostics: the plain kernel for one product only (STRATA_VIT_NO_MPP_W: the weight products, _S: the scores, _PV: the context)
        if (std::getenv("STRATA_VIT_NO_MPP_W")) { m.mpp_bf16 = m.mpp_f16 = false; w_bf = w_f16 = "STRATA_VIT_NO_MPP_W"; }
        if (std::getenv("STRATA_VIT_NO_MPP_S")) { m.mpp_s = false; w_s = "STRATA_VIT_NO_MPP_S"; }
        if (std::getenv("STRATA_VIT_NO_MPP_PV")) { m.mpp_pv = false; w_pv = "STRATA_VIT_NO_MPP_PV"; }
        auto say = [&](bool on, const std::string& w) { return on ? std::string("tensor op") : "plain" + (w.empty() ? (mpp_why.empty() ? std::string() : " (" + mpp_why + ")") : " (" + w + ")"); };
        std::fprintf(stderr, "strata-vision: Metal tower on %s; products: BF16 weights %s, F16 weights %s, attention S %s, P*V %s; patch matrix float32, one output a thread\n",
                     [[m.dev name] UTF8String], say(m.mpp_bf16, w_bf).c_str(), say(m.mpp_f16, w_f16).c_str(), say(m.mpp_s, w_s).c_str(), say(m.mpp_pv, w_pv).c_str());
        m.bytes = m.work_bytes = 0;
        footprint_note("kernels compiled, self-test done");
        // the patch embedding: the two temporal taps summed, kept in float32 (a small matrix)
        m.bf16_mode = vit_gpu_bf16_mode(std::getenv("STRATA_VIT_BF16_AS_F16"), BF16_TO_F16) == BF16_TO_F16 ? BF16_TO_F16 : BF16_STORED;     // before the matrices are uploaded
        if (const char* e = std::getenv("STRATA_VIT_REF_ROUND")) m.round_out = std::atoi(e) != 0;
        if (const char* e = std::getenv("STRATA_VIT_ROWS")) { const int v = std::atoi(e); if (v >= 16) m.kRows = v; }
        if (const char* e = std::getenv("STRATA_VIT_SCORES")) { const int v = std::atoi(e); if (v >= 1) { m.kScoreElems = (size_t) v << 20; m.scores_env = true; } }
        if (const char* e = std::getenv("STRATA_VIT_SCORES_LARGE")) { const long v = std::atol(e); if (v >= 1) m.kScoreElemsLarge = (size_t) v << 20; }
        if (const char* e = std::getenv("STRATA_VIT_NPAD")) { const int v = std::atoi(e); if (v >= 8 && v % 8 == 0) m.n_pad = v; }
        if (const char* e = std::getenv("STRATA_VIT_QALIGN")) { const int v = std::atoi(e); if (v >= 1) m.q_align = v; }
        m.map_file(gg);
        if (m.gemma() || g.minicpmv()) {
            if (!(m.gemma() ? m.load_gemma(gg, err) : m.load_minicpm(gg, err))) return false;
            footprint_note("weights uploaded");
            const bool okg = m.size_buffers(std::max(max_patches, 16), err);
            malloc_zone_pressure_relief(nullptr, 0);
            return okg;
        }
        {
            const GgufTensorInfo* t0 = gg.tensor("v.patch_embd.weight");
            if (!t0) { err = "the mmproj file lacks v.patch_embd.weight"; return false; }
            std::vector<float> w0((size_t) t0->nelem()), w1;
            if (!gg.read_f32(*t0, w0.data(), err)) return false;
            if (g.patch_taps == 2) {
                const GgufTensorInfo* t1 = gg.tensor("v.patch_embd.weight.1");
                w1.resize(w0.size());
                if (!t1 || !gg.read_f32(*t1, w1.data(), err)) { if (err.empty()) err = "the mmproj file lacks v.patch_embd.weight.1"; return false; }
                for (size_t i = 0; i < w0.size(); ++i) w0[i] += w1[i];
            }
            m.patch_w = m.buf_from(w0.data(), w0.size() * 4, err);
            if (!m.patch_w) return false;
        }
        if (!m.upload_f32(m.patch_b, gg, "v.patch_embd.bias", err) || !m.upload_f32(m.pos, gg, "v.position_embd.weight", err) ||
            !m.upload_f32(m.post_w, gg, "v.post_ln.weight", err) || !m.upload_f32(m.post_b, gg, "v.post_ln.bias", err) ||
            !m.upload_matrix(m.mm0, gg, "mm.0.weight", err) || !m.upload_f32(m.mm0_b, gg, "mm.0.bias", err) ||
            !m.upload_matrix(m.mm2, gg, "mm.2.weight", err) || !m.upload_f32(m.mm2_b, gg, "mm.2.bias", err))
            return false;
        m.layers.assign((size_t) g.layers, DevLayer{});
        for (int il = 0; il < g.layers; ++il) {
            DevLayer& L = m.layers[(size_t) il];
            const std::string p = "v.blk." + std::to_string(il) + ".";
            if (!m.upload_f32(L.ln1_w, gg, p + "ln1.weight", err) || !m.upload_f32(L.ln1_b, gg, p + "ln1.bias", err) ||
                !m.upload_f32(L.ln2_w, gg, p + "ln2.weight", err) || !m.upload_f32(L.ln2_b, gg, p + "ln2.bias", err) ||
                !m.upload_matrix(L.qkv, gg, p + "attn_qkv.weight", err) || !m.upload_f32(L.qkv_b, gg, p + "attn_qkv.bias", err) ||
                !m.upload_matrix(L.o, gg, p + "attn_out.weight", err) || !m.upload_f32(L.o_b, gg, p + "attn_out.bias", err) ||
                !m.upload_matrix(L.up, gg, p + "ffn_up.weight", err) || !m.upload_f32(L.up_b, gg, p + "ffn_up.bias", err) ||
                !m.upload_matrix(L.down, gg, p + "ffn_down.weight", err) || !m.upload_f32(L.down_b, gg, p + "ffn_down.bias", err))
                return false;
        }
        footprint_note("weights uploaded");
        const bool ok = m.size_buffers(std::max(max_patches, 16), err);
        footprint_note("work buffers");
        // the host copies of the weights (one matrix at a time, up to ~10 MB each) are freed by now, but macOS's allocator keeps freed large blocks in its
        // cache, where they count in the reader's footprint (87 MB at READY, RESULTS-OWN-3V.md D5): hand them back
        malloc_zone_pressure_relief(nullptr, 0);
        return ok;
    }
}

bool VitGpu::encode(const PreparedImage& img, std::vector<float>& rows, int& nx, int& ny, std::string& err) {
    @autoreleasepool {
        VitGpuImpl& m = *impl_;
        const VitGeometry& g = m.g;
        if (m.gemma()) return m.encode_gemma(img, rows, nx, ny, err);
        if (g.minicpmv()) return m.encode_minicpm(img, rows, nx, ny, err);
        if (m.layers.empty()) { err = "the tower is not loaded"; return false; }
        if (img.w <= 0 || img.h <= 0 || img.w % g.align() != 0 || img.h % g.align() != 0) { err = "the picture size is not a multiple of patch * merge"; return false; }
        if (g.head_dim > 96 || g.head_dim % 4 != 0) { err = "this head size is not handled by the GPU attention kernel"; return false; }
        const int P = g.patch, D = g.width, H = g.heads, HD = g.head_dim, F = g.ffn;
        const int pw = img.w / P, ph = img.h / P, N = pw * ph, nbx = pw / 2;
        nx = pw / 2;
        ny = ph / 2;
        if (!m.size_buffers(N, err)) return false;
        m.fail.clear();
        // STRATA_VIT_PROF=1 (a diagnostic, stderr only, off by default): the time of each stage, the GPU waited for after every stage
        static const bool prof = std::getenv("STRATA_VIT_PROF") != nullptr;
        double pt[10] = {0};
        auto pclock = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        double pmark = prof ? pclock() : 0.0;
        auto stage = [&](int i) {
            if (!prof) return;
            (void) m.finish();
            const double now = pclock();
            pt[i] += now - pmark;
            pmark = now;
        };
        const int K0 = 3 * P * P;
        {   // the patch matrix [N][K0], channel-major per patch, tokens in 2x2-merge order, written by the host into the context buffer (the GPU is idle here)
            float* pm = (float*) [m.d_ctx contents];
            const size_t plane = (size_t) img.w * img.h;
            for (int t = 0; t < N; ++t) {
                const int b = t / 4, r = t % 4, py = 2 * (b / nbx) + r / 2, px = 2 * (b % nbx) + r % 2;
                float* a = pm + (size_t) t * K0;
                for (int c = 0; c < 3; ++c)
                    for (int ky = 0; ky < P; ++ky)
                        std::memcpy(a + (size_t) c * P * P + (size_t) ky * P, &img.chw[c * plane + (size_t) (py * P + ky) * img.w + (size_t) px * P], (size_t) P * sizeof(float));
            }
        }
        stage(0);
        {
            const GemmArgs a{(uint32_t) N, (uint32_t) K0, (uint32_t) D, (uint32_t) K0, (uint32_t) K0, (uint32_t) D, 0, 0, 0};
            m.go("vk_gemm_seq_f32", {BO(m.d_ctx), BO(m.patch_w), BO(m.d_x)}, &a, sizeof a, MTLSizeMake((D + 63) / 64, N, 1), 64);
        }
        {
            const PatchPostArgs a{g.pos_grid, pw, ph, nbx, D};
            m.go("vk_patch_post", {BO(m.d_x), BO(m.patch_b), BO(m.pos)}, &a, sizeof a, MTLSizeMake(N, 1, 1), 256);
        }
        stage(1);
        const float theta_scale = std::pow(g.rope_theta, -2.0f / (float) (HD / 2));
        const float kq_scale = 1.0f / std::sqrt((float) HD);
        const int R = m.kRows;
        for (const DevLayer& L : m.layers) {
            for (int r0 = 0; r0 < N; r0 += R) {
                const int rows_c = std::min(R, N - r0);
                const NormArgs na{D, (int32_t) L.qkv.kind, g.eps};
                m.go("vk_layernorm", {BO(m.d_x, (size_t) r0 * D * 4), BO(L.ln1_w), BO(L.ln1_b), BO(m.d_a)}, &na, sizeof na, MTLSizeMake(rows_c, 1, 1), 256);
                m.wgemm(L.qkv, BO(m.d_a), rows_c, BO(m.d_y));
                const RopeArgs ra{r0, nbx, H, HD, theta_scale, kq_scale};
                m.go("vk_qkv_rope", {BO(m.d_y), BO(L.qkv_b), BO(m.d_q16), BO(m.d_k16), BO(m.d_v16)}, &ra, sizeof ra, MTLSizeMake(rows_c, 1, 1), 256);
            }
            stage(2);
            const int Np = (int) m.padded((size_t) N);
            int QC = (int) std::max<size_t>(1, std::min<size_t>((size_t) N, m.sp_elems / ((size_t) H * Np)));
            const int qa = m.align_for((size_t) N);
            if (qa > 1 && QC < N && QC >= qa) QC -= QC % qa;
            for (int q0 = 0; q0 < N; q0 += QC) {
                const int qc = std::min(QC, N - q0);
                // S [head][query][key] = q k^T (q carries the scale), F16
                m.gemm("vk_gemm_s", m.mpp_s ? "vm_gemm_s" : nullptr, BO(m.d_q16, (size_t) q0 * D * 2), BO(m.d_k16), BO(m.d_S), qc, HD, N, D, D, Np,
                       (size_t) HD, (size_t) HD, (size_t) qc * Np, H);
                stage(8);
                const SoftmaxArgs sa{N, Np};
                const char* sm = nullptr;
                NSUInteger st = 256;
                if (N <= 128) { sm = "vk_softmax_rows_2"; st = 64; }
                else if (N <= 512) { sm = "vk_softmax_rows_8"; st = 64; }
                else if (N <= 1024) { sm = "vk_softmax_rows_8"; st = 128; }
                else if (N <= 4096) { sm = "vk_softmax_rows_16"; st = 256; }
                else if (N <= 16384) { sm = "vk_softmax_rows_64"; st = 256; }
                if (sm && [m.pipe(sm) maxTotalThreadsPerThreadgroup] < st) sm = nullptr;     // registers too few for that group here: the long-row kernel
                if (sm) m.go(sm, {BO(m.d_S)}, &sa, sizeof sa, MTLSizeMake((size_t) H * qc, 1, 1), st);
                else m.go("vk_softmax_rows_long", {BO(m.d_S)}, &sa, sizeof sa, MTLSizeMake((size_t) H * qc, 1, 1), 256);
                stage(9);
                // context [query][head * HD + d] = P V, float32
                m.gemm("vk_gemm_pv", m.mpp_pv ? "vm_gemm_pv" : nullptr, BO(m.d_S), BO(m.d_v16), BO(m.d_ctx, (size_t) q0 * D * 4), qc, N, HD, Np, D, D,
                       (size_t) qc * Np, (size_t) HD, (size_t) HD, H);
            }
            stage(3);
            for (int r0 = 0; r0 < N; r0 += R) {
                const int rows_c = std::min(R, N - r0);
                const EltArgs ca{(uint64_t) rows_c * D, (int32_t) L.o.kind, D};
                m.go("vk_cast", {BO(m.d_ctx, (size_t) r0 * D * 4), BO(m.d_a)}, &ca, sizeof ca, MTLSizeMake(VitGpuImpl::blocks_for(ca.n), 1, 1), 256);
                m.wgemm(L.o, BO(m.d_a), rows_c, BO(m.d_y));
                const EltArgs ra{0, 0, D};
                m.go("vk_add_bias_res", {BO(m.d_x, (size_t) r0 * D * 4), BO(m.d_y), BO(L.o_b)}, &ra, sizeof ra, MTLSizeMake(rows_c, 1, 1), 256);
            }
            stage(4);
            for (int r0 = 0; r0 < N; r0 += R) {
                const int rows_c = std::min(R, N - r0);
                const NormArgs na{D, (int32_t) L.up.kind, g.eps};
                m.go("vk_layernorm", {BO(m.d_x, (size_t) r0 * D * 4), BO(L.ln2_w), BO(L.ln2_b), BO(m.d_a)}, &na, sizeof na, MTLSizeMake(rows_c, 1, 1), 256);
                m.wgemm(L.up, BO(m.d_a), rows_c, BO(m.d_y));
                const EltArgs ga{0, (int32_t) L.down.kind, F};
                m.go("vk_bias_gelu_cast", {BO(m.d_y), BO(L.up_b), BO(m.d_f16)}, &ga, sizeof ga, MTLSizeMake(rows_c, 1, 1), 256);
                m.wgemm(L.down, BO(m.d_f16), rows_c, BO(m.d_y));
                const EltArgs ra{0, 0, D};
                m.go("vk_add_bias_res", {BO(m.d_x, (size_t) r0 * D * 4), BO(m.d_y), BO(L.down_b)}, &ra, sizeof ra, MTLSizeMake(rows_c, 1, 1), 256);
            }
            stage(5);
            m.split();      // each layer is its own command buffer: none runs long enough for the system to cut it short
        }
        // post norm (into the merger's input kind), then the merger on 4-patch rows, in chunks; the output rows go to the context buffer
        const int M = N / 4, RM = std::max(1, R / 4);
        for (int r0 = 0; r0 < M; r0 += RM) {
            const int rows_c = std::min(RM, M - r0);
            const NormArgs na{D, (int32_t) m.mm0.kind, g.eps};
            m.go("vk_layernorm", {BO(m.d_x, (size_t) r0 * 4 * D * 4), BO(m.post_w), BO(m.post_b), BO(m.d_a)}, &na, sizeof na, MTLSizeMake((size_t) rows_c * 4, 1, 1), 256);
            m.wgemm(m.mm0, BO(m.d_a), rows_c, BO(m.d_y));
            const EltArgs ga{0, (int32_t) m.mm2.kind, g.merger_hidden};
            m.go("vk_bias_gelu_cast", {BO(m.d_y), BO(m.mm0_b), BO(m.d_f16)}, &ga, sizeof ga, MTLSizeMake(rows_c, 1, 1), 256);
            m.wgemm(m.mm2, BO(m.d_f16), rows_c, BO(m.d_ctx, (size_t) r0 * g.out_dim * 4));
            const EltArgs ba{0, 0, g.out_dim};
            m.go("vk_add_bias", {BO(m.d_ctx, (size_t) r0 * g.out_dim * 4), BO(m.mm2_b)}, &ba, sizeof ba, MTLSizeMake(rows_c, 1, 1), 256);
        }
        std::string ferr;
        const bool ok = m.finish(&ferr);
        if (!m.fail.empty()) { err = m.fail; return false; }
        if (!ok) { err = ferr; return false; }
        stage(6);
        rows.resize((size_t) M * g.out_dim);
        std::memcpy(rows.data(), [m.d_ctx contents], rows.size() * 4);
        footprint_note("after a picture");
        malloc_zone_pressure_relief(nullptr, 0);     // the picture's decode and preparation buffers, freed by the caller before this one: see load()
        stage(7);
        if (prof)
            std::fprintf(stderr, "vit-prof N=%d: patch-matrix %.2f, patch-embed %.2f, ln+qkv %.2f, attention %.2f (S product %.2f, softmax %.2f), o-proj %.2f, ffn %.2f, merger %.2f, copy-out %.2f ms\n",
                         N, pt[0], pt[1], pt[2], pt[3] + pt[8] + pt[9], pt[8], pt[9], pt[4], pt[5], pt[6], pt[7]);
        return true;
    }
}

}  // namespace strata_vit
