# Copyright 2026 Tencent Inc.  All rights reserved.
#
# Iluvatar kernels —— 解耦后的真实 GPU 算子接入点。
#
# 策略：按算子优先级（embedding -> layernorm -> rotary -> paged_attention ->
# activation -> samplers）逐个把 nvidia/ 下纯 CUDA 的算子挑出来用 IXCC 编译。
# 不直接 add_subdirectory(../nvidia/<x>) —— 那些 CMakeLists.txt 用了 file(GLOB)
# 把测试也带进来，且部分链接 -lnvidia-ml；这里写独立的薄 CMakeLists，只编必要源文件。
#
# 命名空间保持 llm_kernels::nvidia，调用方零修改。

add_subdirectory(embedding)
add_subdirectory(rotary_embedding)
add_subdirectory(paged_attention)

# TODO(iluvatar): 按优先级依次接入
#   add_subdirectory(layernorm)
#   add_subdirectory(activation)
#   add_subdirectory(samplers)
