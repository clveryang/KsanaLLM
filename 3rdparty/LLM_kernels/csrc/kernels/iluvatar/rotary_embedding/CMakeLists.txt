# Copyright 2026 Tencent Inc.  All rights reserved.
#
# Iluvatar rotary_embedding kernel.
# 与 nvidia/rotary_embedding/rotary_embedding.cu 内容完全一致——纯 CUDA、无 PTX 内联，
# IXCC `-x ivcore` 可直接吃。命名空间保持 llm_kernels::nvidia 不变（ABI 契约）。
# Target 名 llm_kernels_iluvatar_kernel_rotary_embedding，库产物
# libllm_kernels_iluvatar_kernel_rotary_embedding.a，与硬件后端 1:1 命名。

set(ROTARY_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/rotary_embedding.cu)

# 顶层项目没 enable_language(CUDA)，把 .cu 当 C++ 编，并显式追加 `-x ivcore` +
# `-std=c++17`（rotary_embedding.cu 用到 std::min/std::max，需要 C++17 模式）。
set_source_files_properties(${ROTARY_SRCS} PROPERTIES LANGUAGE CXX
    COMPILE_OPTIONS "-x;ivcore;--cuda-gpu-arch=${IX_GPU_ARCH};-std=c++17")

add_library(llm_kernels_iluvatar_kernel_rotary_embedding STATIC ${ROTARY_SRCS})
target_include_directories(llm_kernels_iluvatar_kernel_rotary_embedding PUBLIC
    ${CMAKE_SOURCE_DIR}/3rdparty/LLM_kernels
    /usr/local/corex/include)
set_property(TARGET llm_kernels_iluvatar_kernel_rotary_embedding PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(llm_kernels_iluvatar_kernel_rotary_embedding PUBLIC
    llm_kernels_iluvatar_utils -lcublas -lcudart)
