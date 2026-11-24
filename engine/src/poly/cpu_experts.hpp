// src/poly/cpu_experts.hpp - routed experts computed on the CPU from their GGUF bytes.
//
// The dot products and the activation quantizers are Strata's quant layer's (strata/quant), which gives the bits of
// ggml-cpu's for the same rows, so an expert here still computes what llama.cpp's CPU backend computes for the same
// tensor.  (Float tensors are the exception: cpu_experts.cpp.)  The workers are this file's own: the pool
// in strata/kernels/cpu/pool.hpp is built around the other model's expert blob.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::poly {

/// Runs fn(0..n-1) on the workers and the calling thread, and returns when all of them are done.  A call from
/// inside fn does its work on the thread it is on.
class Workers {
public:
    explicit Workers(int threads);
    ~Workers();
    Workers(const Workers&) = delete;
    Workers& operator=(const Workers&) = delete;

    void run(int n, const std::function<void(int)>& fn);
    int threads() const { return (int) th_.size() + 1; }

private:
    // Every thread takes tasks from next and counts what it ran down from left once, when it finds no more: with a
    // count down a task, and both counts in one cache line, an empty batch of 32 tasks took 11 us of 32 threads.
    struct Batch {
        const std::function<void(int)>* fn;
        int n;
        alignas(64) std::atomic<int> next{0};
        alignas(64) std::atomic<int> left{0};
    };
    // A worker's mark that it may hold a Batch pointer, on a line of its own: run() waits for every one of them to
    // be down before its Batch goes.  One count of them all was two writes a worker to the same line, every batch.
    struct alignas(64) Busy {
        std::atomic<int> on{0};
    };
    void loop(int i);
    static bool drain(Batch& b);

    std::vector<std::thread> th_;
    std::vector<Busy> busy_;                     ///< a worker's Busy
    alignas(64) std::atomic<Batch*> cur_{nullptr};   ///< every idle worker reads it in a loop
    alignas(64) std::atomic<int> sleepers_{0};
    std::mutex one_;                             ///< one batch at a time: on several cards, each card's thread posts
    alignas(64) std::atomic<uint32_t> posted_{0};   ///< counts the batches: what a sleeping worker waits on
    std::atomic<bool> quit_{false};
};

/// A thread of its own that runs one function at a time beside the caller: start() hands it over, finish() returns
/// once it has run and throws what it threw.  Workers::run() returns when every task of a batch is done, so work
/// handed to one of its tasks held the whole batch up while it ran.
class Aside {
public:
    Aside();
    ~Aside();
    Aside(const Aside&) = delete;
    Aside& operator=(const Aside&) = delete;

    void start(const std::function<void()>& fn);
    void finish();

private:
    void loop();

    const std::function<void()>* fn_ = nullptr;
    std::exception_ptr failed_;
    alignas(64) std::atomic<uint32_t> posted_{0};
    alignas(64) std::atomic<uint32_t> done_{0};
    std::atomic<bool> sleeping_{false}, quit_{false};
    std::thread th_;   ///< last: the constructor starts it, and loop() reads the members above
};

/// One routed expert of a token: its three matrices in the GGUF, its router weight, and which of the call's tokens
/// it is for.
struct CpuExpert {
    const uint8_t* gate;
    const uint8_t* up;
    const uint8_t* down;
    float weight;
    int row = 0;
    float gate_scale = 1.f, up_scale = 1.f;   ///< NVFP4's global scales (ExpertScales), on the products
};

/// Checks that ggml-cpu has a dot product for both types.  Call once per layer format at load.
bool cpu_expert_types_ok(int gu_type, int d_type, int n_embd, int n_ff, std::string& err);

/// The product cpu_experts() takes four weight rows at a time with for weights of `type`, one activation row against
/// rows w, w + row_bytes, ... (`four`), and the one-row product whose bits each of its values is (`one`); both null where
/// it takes one row at a time.  For cpu_x4_test.
struct CpuDotX4 {
    void (*one)(int n, float* s, const void* w, const void* a) = nullptr;
    void (*four)(int n, float out[4], const void* w, size_t row_bytes, const void* a) = nullptr;
};
CpuDotX4 cpu_dot_x4(int type);


/// For each of `rows` tokens, with n_embd values of x and of out each:
/// out = sum over the token's experts of weight * down(silu(min(gate x, limit)) * clamp(up x, -limit, limit)),
/// or with `gelu` of weight * down(gelu(gate x) * up x), in the tanh form of GELU the GPU experts compute, or with
/// situ_beta above 0 of weight * down(beta * tanh(gate x / beta) * sigmoid(gate x) * lb * tanh(up x / lb)) (lb 0: up x).
/// The n experts come token after token, a token's together.  An expert that several of the tokens use is read
/// once for all of them: reading it is most of the work.  A token's result is the one it has in a call of its own.
/// `meanwhile`, when given, is run once by one of the threads, beside the first products.  It must not throw.
/// u_type is up's type: gate's in every file but the few that quantize the two apart.
/// `late`, when given, marks the experts whose weights are still being read: their products come after the others',
/// and `wait(j)` is called before experts[j]'s are read.  The sums are as without them.
void cpu_experts(Workers& workers, const CpuExpert* experts, int n, int rows, int gu_type, int u_type, int d_type,
                 int n_embd, int n_ff, const float* x, float limit, bool gelu, float* out,
                 const std::function<void()>* meanwhile = nullptr, const uint8_t* late = nullptr,
                 const std::function<void(int)>* wait = nullptr, float situ_beta = 0.f, float situ_lb = 0.f);

}  // namespace strata::poly
