# Copyright 2026 Tencent Inc.  All rights reserved.
#
# Iluvatar utils target.
# 源码直接复用 ../nvidia/ 下的几个文件：cuda_utils.cu / cuda_fp8_utils.cu /
# string_utils.cpp / kllm_exception.cpp / assert.cpp / ../common.cc。
# IXCC `-x ivcore` 可以直接吃 .cu，不需要 nvcc。
# 注意：去掉 nvidia 链接的 -lnvidia-ml（Iluvatar 上没有这个库，
# 也不会被 embedding 这条路径调用）。

set(NV_UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../nvidia)

set(ILUVATAR_UTILS_SRCS
    ${NV_UTILS_DIR}/cuda_utils.cu
    ${NV_UTILS_DIR}/string_utils.cpp
    ${NV_UTILS_DIR}/kllm_exception.cpp
    ${NV_UTILS_DIR}/assert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common.cc
)
# cuda_fp8_utils.cu 在 Iluvatar 上不需要（ENABLE_FP8 OFF），且引用 vec_dtypes.cuh
# 里大量 stock-clang 不支持的 PTX 内联，先排除。

# 把 .cu 也当 C++ 编（顶层项目没 enable_language(CUDA)），并显式追加 `-x ivcore`，
# 让 IXCC 走 ivcore 后端而不是 stock clang 的 -x cuda 路径。
set_source_files_properties(${ILUVATAR_UTILS_SRCS} PROPERTIES LANGUAGE CXX)
set_source_files_properties(
    ${NV_UTILS_DIR}/cuda_utils.cu
    PROPERTIES COMPILE_OPTIONS "-x;ivcore;--cuda-gpu-arch=${IX_GPU_ARCH};-std=c++17")

# Target 名以 iluvatar 命名（产物 libllm_kernels_iluvatar_utils.a）；
# 命名空间仍是 llm_kernels::utils（源码保持不动），仅链接器层面区分硬件后端。
add_library(llm_kernels_iluvatar_utils STATIC ${ILUVATAR_UTILS_SRCS})
target_compile_definitions(llm_kernels_iluvatar_utils PRIVATE FLASHINFER_USE_FP8=1)
# `csrc/...` 形式的 include 需要 3rdparty/LLM_kernels 作为搜索根。
# corex include 给 .cpp 文件用（.cu 走 -x ivcore 自带头路径，但 .cpp 没有）。
target_include_directories(llm_kernels_iluvatar_utils PUBLIC
    ${CMAKE_SOURCE_DIR}/3rdparty/LLM_kernels
    /usr/local/corex/include)
target_link_libraries(llm_kernels_iluvatar_utils PUBLIC -lcublas -lcudart -lcurand -lcuda)
set_property(TARGET llm_kernels_iluvatar_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
