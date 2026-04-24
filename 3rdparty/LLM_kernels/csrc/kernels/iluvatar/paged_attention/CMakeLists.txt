# Copyright 2026 Tencent Inc.  All rights reserved.
#
# Iluvatar paged_attention kernel.
#
# 与 nvidia/paged_attention/ 几乎一致：
#   - cache_copy.cu / cache_copy_flash_attn_layout.cu —— K/V 写入 paged cache
#   - paged_attention.cu —— vLLM v1 风格的 decode kernel（per-seq pointer array
#     + cache_offsets，**不**依赖 block_table）
#   - paged_attention_scalar_*_cache_auto.cu —— 显式实例化（仅 fp16/bf16/fp32
#     的 kAuto；fp8、MLA、e4m3、e5m2 文件已物理删除以避免引入 vec_dtypes.cuh
#     里 IXCC 不支持的 PTX 内联）。
#
# 内含的 __shfl_xor_sync / __shfl_sync 都是 IXCC 原生支持的标准 CUDA intrinsic。
# 命名空间保持 llm_kernels::nvidia 不变（ABI 契约）。
# Target 名 llm_kernels_iluvatar_kernel_paged_attention，与硬件后端 1:1 命名。

file(GLOB PAGED_ATTENTION_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# 顶层项目没 enable_language(CUDA)，把 .cu 当 C++ 编，并显式追加 `-x ivcore` +
# `-std=c++17`（用到 std::min/std::max）。
set_source_files_properties(${PAGED_ATTENTION_SRCS} PROPERTIES LANGUAGE CXX
    COMPILE_OPTIONS "-x;ivcore;--cuda-gpu-arch=${IX_GPU_ARCH};-std=c++17")

add_library(llm_kernels_iluvatar_kernel_paged_attention STATIC ${PAGED_ATTENTION_SRCS})
target_include_directories(llm_kernels_iluvatar_kernel_paged_attention PUBLIC
    ${CMAKE_SOURCE_DIR}/3rdparty/LLM_kernels
    /usr/local/corex/include)
set_property(TARGET llm_kernels_iluvatar_kernel_paged_attention PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(llm_kernels_iluvatar_kernel_paged_attention PUBLIC
    llm_kernels_iluvatar_utils -lcublas -lcudart)
