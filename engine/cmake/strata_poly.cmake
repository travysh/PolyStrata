# PolyStrata's own targets: strata-poly, the engine for the models read from their GGUF files (GLM 5.3 Flash,
# Qwen3.6, Gemma 4, LFM2.5, DeepSeek V4 Flash, Flint, Kolibri 1, ...), and its tests.  CMakeLists.txt calls
# strata_poly() after Strata's own targets exist; it uses their libraries (strata_kernels, strata_spec, the GPU
# runtime and BLAS targets).
#
# strata-poly needs no ggml and no llama.cpp: a prompt's quantized products are Strata's own kernels
# (src/poly/mmq_own.cu: int8 and fp16 tiles where the card has them, else a dequantized matrix and the BLAS library,
# which is the HIP and Turing path), the CPU experts' rows its quant layer (strata_quant).  -DSTRATA_POLY_MMQ=ON
# builds the products llama.cpp's way instead (its MMQ kernels, its flash-attention kernel and the GDN adapter that
# needs its headers), from a llama.cpp checkout given as -DSTRATA_GGML_DIR: an override kept for two releases.

macro(strata_poly)
if(STRATA_BUILD_TESTS)
  add_executable(glm_conversations_test src/poly/conversations_test.cpp src/poly/spill.cpp)
  add_test(NAME glm_conversations_test COMMAND glm_conversations_test)
  add_executable(poly_guesses_test src/poly/guesses_test.cpp src/spec/draft_policy.cpp)
  target_include_directories(poly_guesses_test PRIVATE include)
  add_test(NAME poly_guesses_test COMMAND poly_guesses_test)
  add_executable(poly_sampler_test src/poly/sampler_test.cpp src/poly/grammar.cpp)
  target_include_directories(poly_sampler_test PRIVATE include src/poly)
  add_test(NAME poly_sampler_test COMMAND poly_sampler_test)
  add_executable(poly_grammar_test src/poly/grammar_test.cpp src/poly/grammar.cpp)
  target_include_directories(poly_grammar_test PRIVATE include src/poly)
  add_test(NAME poly_grammar_test COMMAND poly_grammar_test)

endif()

if(STRATA_BUILD_POLY)
  # Clang turns ggml-cpu's i-quant grid lookups (_mm256_set_epi32 of eight table loads, IQ2_*/IQ3_*) into
  # vpgatherdd, which Intel's Gather Data Sampling microcode mitigation (2023) made slow on Skylake to Tiger Lake.
  # -mno-gather keeps them scalar loads: the same values, so the same products.  On an i9-9980HK (Windows, ROCm's
  # clang, one thread) a 2048-value IQ2_S row went from 500 to 183 ns and an IQ3_XXS row from 978 to 269 ns;
  # IQ4_XS, Q3_K and Q8_0 rows did not move.  On by default only where a GNU-style clang compiles ggml-cpu for
  # x86-64 on an Intel CPU (the Windows HIP build on one: the engine is compiled on the PC it runs on); it was not
  # measured on an AMD CPU or on an Intel one newer than Tiger Lake.  -DSTRATA_CPU_NO_GATHER=ON / OFF sets it.
  set(_strata_no_gather_default OFF)
  if(CMAKE_C_COMPILER_ID STREQUAL "Clang" AND NOT CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC" AND
     CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64|amd64)$" AND NOT STRATA_ISA_FLOOR)
    cmake_host_system_information(RESULT _strata_host_cpu QUERY PROCESSOR_DESCRIPTION)
    include(CheckCCompilerFlag)
    check_c_compiler_flag(-mno-gather STRATA_HAVE_MNO_GATHER)
    if(STRATA_HAVE_MNO_GATHER AND _strata_host_cpu MATCHES "GenuineIntel")
      set(_strata_no_gather_default ON)
    endif()
  endif()
  option(STRATA_CPU_NO_GATHER "Compile ggml-cpu with -mno-gather (scalar table loads in place of gathers)"
         ${_strata_no_gather_default})
  if(STRATA_CPU_NO_GATHER AND TARGET ggml-cpu)
    message(STATUS "ggml-cpu: -mno-gather (STRATA_CPU_NO_GATHER)")
    target_compile_options(ggml-cpu PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-mno-gather>)
  endif()
endif()

# The opt-in lifecycle is a host-only test, including on non-CUDA builds.
if(STRATA_BUILD_TESTS)
  add_executable(poly_hip_value_split_policy_test src/poly/hip_value_split_policy_test.cpp)
  add_test(NAME poly_hip_value_split_policy_test COMMAND poly_hip_value_split_policy_test)
  add_executable(poly_gdn_mma_control_test src/poly/gdn_mma_control_test.cpp)
  add_test(NAME poly_gdn_mma_control_test COMMAND poly_gdn_mma_control_test)
  add_executable(poly_q35_scratch_arena_test src/poly/q35_scratch_arena_test.cpp)
  add_test(NAME poly_q35_scratch_arena_test COMMAND poly_q35_scratch_arena_test)
  add_executable(poly_stream_stage_test src/poly/stage_copy_test.cpp)
  add_test(NAME poly_stream_stage_test COMMAND poly_stream_stage_test)
endif()

# Shared prepared-input GDN recurrence on the tensor cores (gdn_mma_tiles.cuh has the tiles it takes from llama.cpp's
# mma.cuh). Either engine may link this library; the Poly Qwen35 adapter is the one caller so far. Keep FP32/BF16
# hi+lo math separate from the MMQ targets' -use_fast_math compilation options.
if(STRATA_BUILD_POLY AND STRATA_NATIVE_EXPERTS AND STRATA_ENABLE_CUDA)
  add_library(strata_gdn_mma STATIC src/kernels/cuda/gdn_mma.cu)
  target_include_directories(strata_gdn_mma PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(strata_gdn_mma PRIVATE ${_strata_gpu_runtime_target})
  if(STRATA_BUILD_TESTS)
    add_executable(poly_gdn_mma_policy_test src/kernels/gdn_mma_policy_test.cpp)
    target_include_directories(poly_gdn_mma_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME poly_gdn_mma_policy_test COMMAND poly_gdn_mma_policy_test)
    add_executable(gdn_mma_test src/kernels/gdn_mma_test.cu)
    target_link_libraries(gdn_mma_test PRIVATE strata_gdn_mma ${_strata_gpu_runtime_target})
    add_test(NAME gdn_mma_test COMMAND gdn_mma_test)
    set_tests_properties(gdn_mma_test PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()

# ---- strata-poly: GLM 5.3 Flash (llama.cpp's glm5-next), read from its GGUF ----
#
# A second engine beside `strata`.  It shares the quantized matvec kernels, the grouped prompt kernel, the quant
# layer and the GGUF reader.  On HIP it is the same CUDA-shaped code through the hip_compat headers (the wave32
# RDNA backend; gfx906's wave64 compat layer is not supported here).
if(STRATA_BUILD_POLY AND STRATA_NATIVE_EXPERTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND NOT STRATA_HIP_GFX906)
  if(STRATA_POLY_MMQ)
  # The prompt path's quantized products: llama.cpp's MMQ kernels (ggml-cuda, MIT) for the types the model's
  # GGUFs keep their matrices in (the list in src/poly/mmq.cu), compiled from the given checkout.  A library of its
  # own: `strata` has no MMQ.
  set(_strata_gg ${_strata_ggml_root}/ggml)
  set(_strata_ggml_cuda ${_strata_gg}/src/ggml-cuda)
  # Navi 12 / Navi 14 (gfx1011, gfx1012) have the signed dot4 that llama.cpp's ggml_cuda_dp4a leaves to RDNA1's SDWA
  # sequence (its vendors/hip.h counts gfx1012 as RDNA1); include/strata/hip_compat/intrinsics.hpp has the
  # measurement.  A HIP build for either has the checkout's common.cuh take __builtin_amdgcn_sdot4 there as well: the
  # same products.  The checkout is downloaded, not part of this tree, so it is patched here, once, by its exact line;
  # STRATA_GFX1012_SDWA_DOT=ON (cmake/hip_backend.cmake) leaves it as it is.
  if(STRATA_ENABLE_HIP AND CMAKE_HIP_ARCHITECTURES MATCHES "gfx101[12]" AND NOT STRATA_GFX1012_SDWA_DOT)
    set(_strata_cc ${_strata_ggml_cuda}/common.cuh)
    file(READ ${_strata_cc} _strata_cc_text)
    set(_strata_cc_old "#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__)\n    c = __builtin_amdgcn_sdot4")
    set(_strata_cc_new "#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__) || defined(__gfx1011__) || defined(__gfx1012__)\n    c = __builtin_amdgcn_sdot4")
    string(FIND "${_strata_cc_text}" "${_strata_cc_old}" _strata_cc_at)
    if(_strata_cc_at GREATER -1)
      string(REPLACE "${_strata_cc_old}" "${_strata_cc_new}" _strata_cc_text "${_strata_cc_text}")
      file(WRITE ${_strata_cc} "${_strata_cc_text}")
      message(STATUS "gfx1011/gfx1012: ${_strata_cc}'s ggml_cuda_dp4a takes the signed dot4")
    endif()
  endif()
  set(_strata_poly_mmq_instances "")
  foreach(_t q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k iq1_s iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl
             iq4_xs)
    list(APPEND _strata_poly_mmq_instances ${_strata_ggml_cuda}/template-instances/mmq-instance-${_t}.cu)
  endforeach()
  # MXFP4 and NVFP4: on RTX 50 llama.cpp's tiles use the FP4 MMA with block scales, which exists only in the
  # arch-specific target, so on CUDA their instances alone are built with 12x -> 12xa, as Strata's W4A4 unit is.  A
  # Blackwell other than 12.0 has no image of them, and Mmq::fits() leaves MXFP4 out there.  NVFP4 takes this
  # instance only on a 12.0 card with STRATA_PREFILL_NVFP4=w4a4; otherwise its tiles are mmq_nvfp4.cu's int8 ones,
  # built for the architectures as given.  On HIP there is no NVFP4 prompt product.
  set(_strata_poly_mxfp4 ${_strata_ggml_cuda}/template-instances/mmq-instance-mxfp4.cu)
  if(STRATA_ENABLE_HIP)
    list(APPEND _strata_poly_mmq_instances ${_strata_poly_mxfp4})
  else()
    list(APPEND _strata_poly_mxfp4 ${_strata_ggml_cuda}/template-instances/mmq-instance-nvfp4.cu)
    list(APPEND _strata_poly_mmq_instances src/poly/mmq_nvfp4.cu)
    set(_strata_poly_fp4_archs ${CMAKE_CUDA_ARCHITECTURES})
    list(TRANSFORM _strata_poly_fp4_archs REPLACE "^(12[0-9])$" "\\1a")
    list(TRANSFORM _strata_poly_fp4_archs REPLACE "^(12[0-9])-(real|virtual)$" "\\1a-\\2")
    add_library(strata_poly_mmq_mxfp4 OBJECT ${_strata_poly_mxfp4})
    set_target_properties(strata_poly_mmq_mxfp4 PROPERTIES CUDA_ARCHITECTURES "${_strata_poly_fp4_archs}")
    target_include_directories(strata_poly_mmq_mxfp4 PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_ggml_cuda})
    target_compile_options(strata_poly_mmq_mxfp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_poly_mmq_mxfp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    list(APPEND _strata_poly_mmq_instances $<TARGET_OBJECTS:strata_poly_mmq_mxfp4>)
  endif()
  if(STRATA_ENABLE_HIP)
    # CMake does not infer HIP from the .cu suffixes outside src/ (and drops a .cu source it has no language
    # for), and ggml's MMQ sources take its native HIP path (vendors/hip.h) instead of the hip_compat headers.
    set_source_files_properties(${_strata_ggml_cuda}/quantize.cu ${_strata_poly_mmq_instances} PROPERTIES LANGUAGE HIP)
  endif()
  add_library(strata_poly_mmq STATIC src/poly/mmq.cu src/poly/fattn.cu src/poly/ggml_cuda_host.cu ${_strata_ggml_cuda}/quantize.cu
              ${_strata_poly_mmq_instances})
  target_include_directories(strata_poly_mmq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_ggml_cuda}
                                                    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(strata_poly_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
  if(MSVC)
    target_compile_options(strata_poly_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
  endif()
  if(STRATA_ENABLE_HIP)
    target_compile_definitions(strata_poly_mmq PRIVATE GGML_USE_HIP=1)
    target_link_libraries(strata_poly_mmq PUBLIC ggml-base hip::host)
  else()
    target_compile_definitions(strata_poly_mmq PRIVATE STRATA_POLY_NVFP4=1)
    target_link_libraries(strata_poly_mmq PUBLIC ggml-base ${_strata_gpu_runtime_target})
  endif()

  else()
    # Strata's own kernels (mmq_own.cu); fattn.cu keeps its decode kernels and leaves llama.cpp's prompt kernel out
    add_library(strata_poly_mmq STATIC src/poly/mmq_own.cu src/poly/fattn.cu)
    target_include_directories(strata_poly_mmq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
    target_compile_definitions(strata_poly_mmq PRIVATE STRATA_POLY_OWN=1)
    target_link_libraries(strata_poly_mmq PUBLIC strata_kernels ${_strata_gpu_runtime_target})
    if(STRATA_ENABLE_HIP)
      target_compile_definitions(strata_poly_mmq PRIVATE GGML_USE_HIP=1)   # fattn.cu: no kernel of either kind on HIP
    else()
      # STRATA_PREFILL_NVFP4=w4a4x2's products (mmq_nvfp4_x2.cu): the block-scaled FP4 MMA exists only in the
      # arch-specific target, so this unit alone is built with 12x -> 12xa.  For another target, or a CUDA before 12.8,
      # its kernels are empty and Mmq never takes them.
      set(_strata_poly_fp4_archs ${CMAKE_CUDA_ARCHITECTURES})
      list(TRANSFORM _strata_poly_fp4_archs REPLACE "^(12[0-9])$" "\\1a")
      list(TRANSFORM _strata_poly_fp4_archs REPLACE "^(12[0-9])-(real|virtual)$" "\\1a-\\2")
      add_library(strata_poly_nvfp4_x2 OBJECT src/poly/mmq_nvfp4_x2.cu)
      set_target_properties(strata_poly_nvfp4_x2 PROPERTIES CUDA_ARCHITECTURES "${_strata_poly_fp4_archs}")
      target_sources(strata_poly_mmq PRIVATE $<TARGET_OBJECTS:strata_poly_nvfp4_x2>)
      target_compile_definitions(strata_poly_mmq PRIVATE STRATA_POLY_X2=1)
    endif()
  endif()

  if(STRATA_ENABLE_CUDA AND TARGET strata_gdn_mma)
    target_compile_definitions(strata_poly_mmq PUBLIC STRATA_POLY_GDN_MMA=1)
    target_link_libraries(strata_poly_mmq PUBLIC strata_gdn_mma)

    add_executable(poly_gdn_mma_test src/poly/gdn_mma_test.cu src/poly/kernels.cu)
    target_include_directories(poly_gdn_mma_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(poly_gdn_mma_test PRIVATE strata_poly_mmq ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    add_test(NAME poly_gdn_mma_test COMMAND poly_gdn_mma_test)
    set_tests_properties(poly_gdn_mma_test PROPERTIES SKIP_RETURN_CODE 77)
    # Optional model/state check; run by hand with an MTP GGUF and frozen prompt IDs.
    add_executable(poly_gdn_mma_state_test src/poly/gdn_mma_state_test.cpp src/poly/engine.cpp src/poly/expert_card.cpp src/poly/glm5next.cpp
                   src/poly/qwen35.cpp src/poly/gemma4.cpp src/poly/deepseek4.cpp src/poly/lfm2moe.cpp
                   src/poly/flint.cpp src/poly/kolibri1.cpp src/poly/laguna.cpp src/poly/inkling2.cpp src/poly/inkling2_pictures.cpp src/poly/inkling2.cu src/poly/cpu_experts.cpp src/poly/kernels.cu
                   src/poly/kernels_dsv4.cu src/core/conversation_memory.cpp)
    target_include_directories(poly_gdn_mma_state_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
    target_link_libraries(poly_gdn_mma_state_test PRIVATE strata_poly_mmq strata_kernels strata_spec strata_quant
                         ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target} CUDA::cuda_driver Threads::Threads)
  endif()

  add_executable(strata-poly src/poly/main.cpp src/poly/grammar.cpp src/poly/engine.cpp src/poly/expert_card.cpp src/poly/glm5next.cpp src/poly/qwen35.cpp
                            src/poly/gemma4.cpp src/poly/deepseek4.cpp src/poly/lfm2moe.cpp src/poly/flint.cpp
                            src/poly/kolibri1.cpp src/poly/laguna.cpp src/poly/inkling2.cpp src/poly/inkling2_pictures.cpp src/poly/inkling2.cu src/poly/cpu_experts.cpp src/poly/kernels.cu src/poly/kernels_dsv4.cu
                            src/poly/spill.cpp src/core/conversation_memory.cpp src/core/conversation_file.cpp)
  target_include_directories(strata-poly PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  # the driver's virtual memory calls: the attention caches grow with the conversation; on HIP they are HIP's
  # memory management calls, which the HIP runtime itself provides (hip_compat/cuda.h)
  if(STRATA_ENABLE_HIP)
    set(_strata_poly_driver "")
  else()
    set(_strata_poly_driver CUDA::cuda_driver)
  endif()
  target_link_libraries(strata-poly PRIVATE strata_poly_mmq strata_kernels strata_spec strata_quant
                                           ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                           ${_strata_poly_driver}
                                           Threads::Threads)
  # needs the model file, so it is run by hand: poly_expert_parity MODEL.gguf
  add_executable(poly_expert_parity src/poly/expert_parity.cpp src/poly/cpu_experts.cpp src/poly/kernels.cu)
  target_include_directories(poly_expert_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                       ${_strata_gpu_include_directories})
  target_link_libraries(poly_expert_parity PRIVATE strata_poly_mmq strata_kernels strata_quant
                                                  ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                  Threads::Threads)
  # a read with the draft block repeats bit for bit; needs the model file, so it is run by hand:
  # poly_draft_race_test MODEL.gguf [RUNS] [LAYER_SPLIT]
  add_executable(poly_draft_race_test src/poly/draft_race_test.cpp src/poly/engine.cpp src/poly/expert_card.cpp src/poly/glm5next.cpp
                                      src/poly/qwen35.cpp src/poly/gemma4.cpp src/poly/deepseek4.cpp
                                      src/poly/lfm2moe.cpp src/poly/flint.cpp src/poly/kolibri1.cpp src/poly/laguna.cpp src/poly/inkling2.cpp src/poly/inkling2_pictures.cpp src/poly/inkling2.cu
                                      src/poly/cpu_experts.cpp src/poly/kernels.cu src/poly/kernels_dsv4.cu
                                      src/core/conversation_memory.cpp)
  target_include_directories(poly_draft_race_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                         ${_strata_gpu_include_directories})
  target_link_libraries(poly_draft_race_test PRIVATE strata_poly_mmq strata_kernels strata_spec strata_quant
                                                    ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                    ${_strata_poly_driver} Threads::Threads)
  # a prompt batch's routing on the card against the hosts' routing; needs a card, so it is run by hand
  add_executable(poly_route_test src/poly/route_test.cu src/poly/kernels.cu)
  target_include_directories(poly_route_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                    ${_strata_gpu_include_directories})
  target_link_libraries(poly_route_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # the fused residual sum, RMS norm and Q8_1 quantization against the three launches; needs a card, run by hand
  add_executable(poly_norm_q8_test src/poly/norm_q8_test.cu src/poly/kernels.cu)
  target_include_directories(poly_norm_q8_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                      ${_strata_gpu_include_directories})
  target_link_libraries(poly_norm_q8_test PRIVATE strata_kernels ${_strata_gpu_blas_target}
                                                  ${_strata_gpu_runtime_target})
  # a prompt batch's products with few outputs against the same sums on the host; needs a card, run by hand
  add_executable(poly_gemm_small_test src/poly/gemm_small_test.cu src/poly/kernels.cu)
  target_include_directories(poly_gemm_small_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                         ${_strata_gpu_include_directories})
  target_link_libraries(poly_gemm_small_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # attend()'s blocks of rows against its other kernels and itself; needs a card, so it is run by hand
  add_executable(poly_attend_test src/poly/attend_test.cu src/poly/kernels.cu)
  target_include_directories(poly_attend_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                     ${_strata_gpu_include_directories})
  target_link_libraries(poly_attend_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # DeepSeek V4's hyper-connection read for a batch against the rows one at a time; needs a card, run by hand
  add_executable(poly_hc_test src/poly/hc_test.cu src/poly/kernels.cu)
  target_include_directories(poly_hc_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  target_link_libraries(poly_hc_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # a streamed cache's attention (--kv-resident): pieces merged by their (max, sum) against one call; needs a card
  add_executable(poly_merge_attn_test src/poly/merge_attn_test.cu src/poly/kernels.cu)
  target_include_directories(poly_merge_attn_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                          ${_strata_gpu_include_directories})
  target_link_libraries(poly_merge_attn_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  add_executable(poly_hip_value_split_test src/poly/hip_value_split_test.cu src/poly/kernels.cu)
  target_include_directories(poly_hip_value_split_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${_strata_gpu_include_directories})
  target_link_libraries(poly_hip_value_split_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  add_test(NAME poly_hip_value_split_test COMMAND poly_hip_value_split_test)
  set_tests_properties(poly_hip_value_split_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 240)
  # STRATA_HIP_SPLIT_ATTEND (heads of 256 over a q8_0 cache): against an FP64 reference and the old kernels, the position on the card, the
  # fallback and a graph replay; needs a card with 32-lane waves (77 = skipped)
  add_executable(poly_hip_split_attend_test src/poly/hip_split_attend_test.cu src/poly/kernels.cu)
  target_include_directories(poly_hip_split_attend_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${_strata_gpu_include_directories})
  target_link_libraries(poly_hip_split_attend_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  add_test(NAME poly_hip_split_attend_test COMMAND poly_hip_split_attend_test)
  set_tests_properties(poly_hip_split_attend_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
  # attend_batch() against attend() with Flint's heads, sinks and window ring; needs a card
  add_executable(poly_sink_attend_test src/poly/sink_attend_test.cu src/poly/kernels.cu)
  target_include_directories(poly_sink_attend_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                          ${_strata_gpu_include_directories})
  target_link_libraries(poly_sink_attend_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # STRATA_HIP_PREFILL_FP32 (attend_batch() on FP32 inputs over a q8_0 cache): against an FP64 reference and the default path, and the
  # fallbacks; needs a card (77 = skipped)
  add_executable(poly_hip_prefill_fp32_test src/poly/hip_prefill_fp32_test.cu src/poly/kernels.cu)
  target_include_directories(poly_hip_prefill_fp32_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${_strata_gpu_include_directories})
  target_link_libraries(poly_hip_prefill_fp32_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  add_test(NAME poly_hip_prefill_fp32_test COMMAND poly_hip_prefill_fp32_test)
  set_tests_properties(poly_hip_prefill_fp32_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
  # fattn_q8() and fattn_f16() (llama.cpp's flash-attention kernel) against attend_batch(), and with `dec` fdec_q8()
  # against attend(); needs a card, run by hand
  if(STRATA_POLY_MMQ)
    add_executable(poly_fattn_test src/poly/fattn_test.cu src/poly/kernels.cu)
    target_include_directories(poly_fattn_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/poly
                                                      ${_strata_gpu_include_directories})
    target_link_libraries(poly_fattn_test PRIVATE strata_poly_mmq ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  endif()
  # STRATA_POLY_STAGED_TILES: the staged experts in size classes, each a launch of its own tile width, against one
  # launch, bit for bit; needs a card
  add_executable(poly_mmq_tiles_test src/poly/mmq_tiles_test.cu)
  target_include_directories(poly_mmq_tiles_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/poly
                                                        ${_strata_gpu_include_directories})
  target_link_libraries(poly_mmq_tiles_test PRIVATE strata_poly_mmq ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  add_test(NAME poly_mmq_tiles_test COMMAND poly_mmq_tiles_test)
  set_tests_properties(poly_mmq_tiles_test PROPERTIES SKIP_RETURN_CODE 77)
  # the int8 product of a matrix whose rows end inside a stage (32-value blocks) against the same rows padded with
  # zero blocks, bit for bit; needs a card
  if(NOT STRATA_POLY_MMQ)
    add_executable(poly_mmq_k32_test src/poly/mmq_k32_test.cu)
    target_include_directories(poly_mmq_k32_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/poly
                                                        ${_strata_gpu_include_directories})
    target_link_libraries(poly_mmq_k32_test PRIVATE strata_poly_mmq ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    add_test(NAME poly_mmq_k32_test COMMAND poly_mmq_k32_test)
    set_tests_properties(poly_mmq_k32_test PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  # a prompt product of each path (the card's, and the BLAS path with STRATA_POLY_PROMPT_BLAS=1) against a double-precision
  # product of ggml's dequantized weights; needs a card
  if(TARGET ggml-base)
    add_executable(poly_mmq_paths_test src/poly/mmq_paths_test.cu)
    target_include_directories(poly_mmq_paths_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/poly
                                                          ${_strata_gpu_include_directories} ${_strata_ggml_root}/ggml/include)
    target_link_libraries(poly_mmq_paths_test PRIVATE strata_poly_mmq ggml-base ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    add_test(NAME poly_mmq_paths_test COMMAND poly_mmq_paths_test)
    add_test(NAME poly_mmq_paths_test_blas COMMAND poly_mmq_paths_test)
    add_test(NAME poly_mmq_paths_test_x2 COMMAND poly_mmq_paths_test)
    set_tests_properties(poly_mmq_paths_test poly_mmq_paths_test_blas poly_mmq_paths_test_x2 PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(poly_mmq_paths_test_blas PROPERTIES ENVIRONMENT "STRATA_POLY_PROMPT_BLAS=1")
    set_tests_properties(poly_mmq_paths_test_x2 PROPERTIES ENVIRONMENT "STRATA_PREFILL_NVFP4=w4a4x2")
  endif()
  # LFM2's short convolution against the same arithmetic on the host; needs a card, so it is run by hand
  add_executable(poly_shortconv_test src/poly/shortconv_test.cu src/poly/kernels.cu)
  target_include_directories(poly_shortconv_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                        ${_strata_gpu_include_directories})
  target_link_libraries(poly_shortconv_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # DeepSeek V4's kernels against the same arithmetic on the host; needs a card, so it is run by hand
  add_executable(poly_dsv4_test src/poly/dsv4_test.cu src/poly/kernels_dsv4.cu)
  target_include_directories(poly_dsv4_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                   ${_strata_gpu_include_directories})
  target_link_libraries(poly_dsv4_test PRIVATE ${_strata_gpu_runtime_target})
  # the pool of the CPU experts, without a model
  add_executable(poly_workers_test src/poly/workers_test.cpp src/poly/cpu_experts.cpp)
  target_link_libraries(poly_workers_test PRIVATE strata_quant Threads::Threads)
  add_test(NAME poly_workers_test COMMAND poly_workers_test)
  # the card's MXFP4 product against ggml's dequantizer on random blocks, without a model; needs a card, run by hand
  if(TARGET ggml-base)
    add_executable(poly_mxfp4_test src/poly/mxfp4_test.cu)
    target_include_directories(poly_mxfp4_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_ggml_root}/ggml/include
                                                      ${_strata_gpu_include_directories})
    target_link_libraries(poly_mxfp4_test PRIVATE strata_kernels ggml-base ${_strata_gpu_runtime_target})
  endif()
  # the Q8_0 KV cache kernels (store_q8 / store_q8_at) against a host reference (GPU, synthetic, no model)
  if(STRATA_BUILD_TESTS)
    add_executable(poly_kv_q8_test src/poly/kv_q8_test.cpp src/poly/kernels.cu)
    target_include_directories(poly_kv_q8_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                       ${_strata_gpu_include_directories})
    target_link_libraries(poly_kv_q8_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                  Threads::Threads)
    add_test(NAME poly_kv_q8_test COMMAND poly_kv_q8_test)
  endif()
  # --kv int8/int8k's kernels (store_int8 / attend_int8) against ggml's own quantizer and against attend() over the
  # same numbers in fp16.  The reference is ggml-cpu's quantize_row_q8_0_ref, hence the ggml links; under HIP the
  # mma tier's probes are behind STRATA_USE_HIP (the dp4a tier is what runs there anyway).
  if(STRATA_BUILD_TESTS AND TARGET ggml-cpu)
    add_executable(poly_kv_int8_test src/poly/kv_int8_test.cpp src/poly/kernels.cu)
    target_include_directories(poly_kv_int8_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                      ${_strata_gpu_include_directories} ${_strata_ggml_root}/ggml/include)
    target_link_libraries(poly_kv_int8_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                  ggml-cpu ggml-base Threads::Threads)
    add_test(NAME poly_kv_int8_test COMMAND poly_kv_int8_test)
  endif()
endif()
endmacro()
