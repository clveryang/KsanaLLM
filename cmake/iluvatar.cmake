# Copyright 2026 Tencent Inc.  All rights reserved.
#
# Iluvatar Corex backend for KsanaLLM.
# Strategy:
#   - Do NOT call enable_language(CUDA). The platform's nvcc is a dummy script.
#   - Compile all .cu files as CXX (LANGUAGE CXX). clang++ with -x ivcore
#     picks them up natively via file extension.
#   - Kernel implementations come from libixformer.so (pre-built by Iluvatar),
#     not from 3rdparty/LLM_kernels.
# ==============================================================================

message(STATUS "==== Configuring Iluvatar Corex backend ====")
message(STATUS "IXCOREX_ROOT: ${IXCOREX_ROOT}")
message(STATUS "Target GPU arch: ${IX_GPU_ARCH}")

# ----- compilers & flags ------------------------------------------------------
# (CMAKE_C_COMPILER / CMAKE_CXX_COMPILER already set before project() in top-level)
set(IX_CUDA_FLAGS "-x ivcore --cuda-gpu-arch=${IX_GPU_ARCH}")

# Force-include Iluvatar compat shim to define __CUDA_ALIGN__ etc.
# before any 3rdparty/LLM_kernels header uses it.
# Suppress clang-strict errors that GCC/NVCC allow as extensions
add_compile_options(
    -include ${CMAKE_SOURCE_DIR}/src/ksana_llm/kernels/iluvatar/compat_shim.h
    -Wno-c++11-narrowing
    -Wno-void-pointer-arithmetic
    -fms-extensions
    -Wno-microsoft-cast          # void* member-function-ptr casts
    -ferror-limit=0              # don't stop at 20 errors
    -Wno-non-pod-varargs         # std::string through variadic
)

# ----- Iluvatar CUDA-like runtime ---------------------------------------------
set(CUDA_TOOLKIT_ROOT_DIR "${IXCOREX_ROOT}" CACHE PATH "")
set(CUDAToolkit_ROOT      "${IXCOREX_ROOT}" CACHE PATH "")
set(CUDA_INCLUDE_DIRS     "${IXCOREX_ROOT}/include")
set(CUDA_CUDART_LIBRARY   "${IXCOREX_ROOT}/lib64/libcudart.so")
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${IXCOREX_ROOT}/lib64)

# ----- NCCL (ixccl is drop-in on Iluvatar) ------------------------------------
set(NCCL_INCLUDE_DIRS "${IXCOREX_ROOT}/include" CACHE PATH "")
set(NCCL_LIBRARIES    "${IXCOREX_ROOT}/lib64/libnccl.so" CACHE PATH "")

# ----- ixinfer (the actual kernel lib, corex SDK native) ----------------------
# ixinfer is the low-level inference SDK shipped with Corex. Headers at
# ${IXCOREX_ROOT}/include/ixinfer.h, libs at ${IXCOREX_ROOT}/lib64/libcuinfer*.so.
# Unlike ixformer (Python-dist-packages), ixinfer is part of the stable SDK.
set(IXINFER_LIBS cuinfer cuinfer_attn cuinfer_blas cuinfer_misc)
message(STATUS "IXINFER_LIBS    : ${IXINFER_LIBS}")
# ixinfer.h is already reachable via ${IXCOREX_ROOT}/include (set above).

# ----- misc -------------------------------------------------------------------
# Llama-style models only — dense, no MoE, no quant, no FP8.
set(ENABLE_FP8 OFF)
set(ENABLE_FLASH_MLA OFF)
set(ENABLE_DEEPSEEK_DEEPGEMM OFF)
add_definitions(-DSINGLE_CARD_ILUVATAR -DILUVATAR_MINIMAL)

# Allow undefined symbols in shared lib — nvidia layer vtables reference kernel functions
# that don't exist on Iluvatar; they won't be called for Llama models.
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--unresolved-symbols=ignore-all" CACHE STRING "" FORCE)
