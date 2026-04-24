# Copyright 2024 Tencent Inc.  All rights reserved.

if(WITH_CUDA)
  add_subdirectory(nvidia)
endif()

if(WITH_ACL)
  add_subdirectory(ascend)
endif()

if(WITH_TOPS)
  add_subdirectory(zixiao)
endif()

# Iluvatar: utils 直接复用 nvidia/ 下的源码（cuda_utils / string_utils / assert /
# kllm_exception / cuda_fp8_utils），它们都是纯 C++/CUDA，没有 cuDNN 依赖，IXCC 可编。
if(WITH_ILUVATAR)
  add_subdirectory(iluvatar)
endif()
