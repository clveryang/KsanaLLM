# Copyright 2026 Tencent Inc.  All rights reserved.
#
# Iluvatar embedding kernel.
# 该 kernel 是纯 CUDA（__global__ + __shfl_sync + float4/uint4），不依赖 cuDNN
# 或 PTX 内联，IXCC `-x ivcore` 可直接吃。
# 内容与 ../../nvidia/embedding/embedding.cu 几乎一致，仅做编译器兼容性微调
# （std::min 类型推导）。命名空间 llm_kernels::nvidia 不变。

set(EMBEDDING_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/embedding.cu)

# 把 .cu 当 C++ 编（项目没 enable_language(CUDA)），并显式追加 `-x ivcore`，
# 让 IXCC 走 ivcore 后端而不是 stock clang 的 -x cuda 路径。
set_source_files_properties(${EMBEDDING_SRCS} PROPERTIES LANGUAGE CXX
    COMPILE_OPTIONS "-x;ivcore;--cuda-gpu-arch=${IX_GPU_ARCH};-std=c++17")

# Target 名以 iluvatar 命名（产物 libllm_kernels_iluvatar_kernel_embedding.a）；
# 命名空间仍是 llm_kernels::nvidia（源码层契约不变），仅链接器层面区分硬件后端。
add_library(llm_kernels_iluvatar_kernel_embedding STATIC ${EMBEDDING_SRCS})
target_include_directories(llm_kernels_iluvatar_kernel_embedding PUBLIC
    ${CMAKE_SOURCE_DIR}/3rdparty/LLM_kernels
    /usr/local/corex/include)
set_property(TARGET llm_kernels_iluvatar_kernel_embedding PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(llm_kernels_iluvatar_kernel_embedding PUBLIC
    llm_kernels_iluvatar_utils -lcublas -lcudart -lcublasLt)
