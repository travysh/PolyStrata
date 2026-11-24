// src/quant/registry.hpp - the per-tier kernel tables the dispatcher (dispatch.cpp) walks.  One table function per source file that holds kernels.
#pragma once

#include "strata/quant/act.hpp"
#include "strata/quant/dot.hpp"

namespace strata::quant {

struct DotTable {
    const DotKernel* v;
    int n;
};
struct DotX4Table {
    const DotX4Kernel* v;
    int n;
};
struct ActTable {
    const ActKernel* v;
    int n;
};

DotTable generic_dot_table() noexcept;   // dot_generic.cpp
ActTable generic_act_table() noexcept;   // act_ref.cpp

/// DotKernel::on of the layer's own IQ3_S kernels (dispatch.cpp): false with STRATA_CPU_IQ3S=0 in the environment (only the exact value 0),
/// which keeps the ggml ports (the same bits, slower).  Read once.
bool iq3s_own_on() noexcept;

#ifdef STRATA_QUANT_X86_TIERS
DotTable avx_dot_table() noexcept;       // x86/dot_avx.cpp      (Arith::Avx, Isa::Avx1)
DotTable avx2_dot_table() noexcept;      // x86/dot_avx2.cpp     (Arith::Avx2, Isa::Avx2)
DotX4Table avx2_dot_x4_table() noexcept; // x86/dot_avx2.cpp     (4-row shared-activation AVX2 kernels)
ActTable avx_act_table() noexcept;       // x86/act_avx.cpp
ActTable avx2_act_table() noexcept;      // x86/act_avx2.cpp
ActTable avx512_act_table() noexcept;    // x86/act_avx512.cpp   (Arith::Avx2 bits in 512-bit lanes, Isa::Avx512)
DotTable avx512_dot_table() noexcept;    // x86/dot_avx512.cpp   (Arith::Avx2 bits in 512-bit lanes, Isa::Avx512)
DotTable avx512vbmi_dot_table() noexcept;   // x86/dot_avx512vbmi.cpp (Arith::Avx2 bits with VBMI byte lookups, Isa::Avx512Vbmi)
DotTable avxvnni_dot_table() noexcept;   // x86/dot_avxvnni.cpp  (Arith::Avx2 bits with AVX-VNNI, Isa::AvxVnni), with STRATA_QUANT_AVXVNNI
DotRowsFn avx2_dot_rows(Type w) noexcept;   // x86/dot_avx2.cpp: dot_rows_kernel() of the Avx2 contract, or nullptr
DotX4Table avx512_dot_x4_table() noexcept; // x86/dot_avx512.cpp (4-row shared-activation VPDPBUSD/VPDPWSSD kernels)
#endif

#ifdef STRATA_QUANT_ARM_TIERS
DotTable neon_dot_table() noexcept;      // arm/dot_neon.cpp     (Arith::Neon, Isa::Neon)
ActTable neon_act_table() noexcept;      // arm/act_neon.cpp
#endif

}  // namespace strata::quant
