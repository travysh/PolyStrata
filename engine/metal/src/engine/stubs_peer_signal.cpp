// metal/src/engine/stubs_peer_signal.cpp - the second card's completion signal of peer_experts.cpp (PR 272): the
// fused scatter that signals when its rows are written, and the stream's done flag.  A Mac has one GPU, so PeerExperts
// never opens and neither is reached; each stops the engine with its name, like stubs.cpp.  A file of its own: the
// linker pulls it for these two alone (stubs_current.cpp also has stubs_0145.cpp's entry points).

#include <cstdint>

namespace strata::metal_engine { void missing(const char* name); }

namespace strata::kernels {

void scatter_rows_and_signal_f32(const float*, float*, const int32_t*, int64_t, int64_t, uint32_t*, uint32_t, int*, void*) {
    strata::metal_engine::missing("strata::kernels::scatter_rows_and_signal_f32");
}
void signal_stream_done(uint32_t*, uint32_t, void*) {
    strata::metal_engine::missing("strata::kernels::signal_stream_done");
}

}  // namespace strata::kernels
