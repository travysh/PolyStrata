// metal/src/q36/stubs_0220.cpp - the Strata project's 0.2.20: peer_experts.cpp (in the host library) rotates, quantizes and
// scatters a second card's expert rows with kernels the Metal kernel library has as stubs (stubs.cpp, stubs_gyro.cpp) or as
// host code (setup_host.cpp).  strata-q36 links the host library without that kernel library, so it gets its own stopping
// stubs here: a Mac has one GPU, so PeerExperts never opens and none of them is reached.

#include <cstdint>

namespace strata::metal_engine { void missing(const char* name); }

namespace strata::kernels {

void hadamard_rows(const float*, float*, int64_t, int64_t, int, const float*, void*) {
    strata::metal_engine::missing("strata::kernels::hadamard_rows");
}
const float* hadamard_device_signs(const float*, int) {
    strata::metal_engine::missing("strata::kernels::hadamard_device_signs");
    return nullptr;
}
void scatter_rows_f32(const float*, float*, const int32_t*, int64_t, int64_t, void*) {
    strata::metal_engine::missing("strata::kernels::scatter_rows_f32");
}
void scatter_rows_and_signal_f32(const float*, float*, const int32_t*, int64_t, int64_t, uint32_t*, uint32_t, int*, void*) {
    strata::metal_engine::missing("strata::kernels::scatter_rows_and_signal_f32");
}
void signal_stream_done(uint32_t*, uint32_t, void*) {
    strata::metal_engine::missing("strata::kernels::signal_stream_done");
}
void quantize_q8_1_rows(const float*, int64_t, int64_t, void*, void*) {
    strata::metal_engine::missing("strata::kernels::quantize_q8_1_rows");
}
uint64_t moe_hit_grouped_scratch_bytes(int64_t, int64_t, int64_t) {
    strata::metal_engine::missing("strata::kernels::moe_hit_grouped_scratch_bytes");
    return 0;
}

}  // namespace strata::kernels
