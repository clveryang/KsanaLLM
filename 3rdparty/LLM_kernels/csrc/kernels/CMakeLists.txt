# Copyright 2024 Tencent Inc.  All rights reserved.

if(WITH_CUDA)
  add_definitions(-DENABLE_FLASHINFER=ON)
  add_subdirectory(nvidia)
endif()

if(WITH_ACL)
  add_subdirectory(ascend)
  set(llm_kernels_ascend_libs ${llm_kernels_ascend_libs} PARENT_SCOPE)
endif()

# Iluvatar: 不编译整个 nvidia/ 树（依赖 cuDNN/cutlass/flashinfer 的子目录无法用 IXCC 直接编译），
# 只挑出对 IXCC `-x ivcore` 友好的纯 CUDA kernel 子目录，按算子优先级逐步接入。
# 命名空间保持 llm_kernels::nvidia，调用方零修改。
if(WITH_ILUVATAR)
  add_subdirectory(iluvatar)
endif()
