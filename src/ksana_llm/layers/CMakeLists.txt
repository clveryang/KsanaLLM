# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE layers_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/cpu/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/*.cpp)
list(FILTER layers_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER layers_SRCS EXCLUDE REGEX ".*/nvidia/.*.cpp")
list(FILTER layers_SRCS EXCLUDE REGEX ".*/ascend/.*.cpp")
list(FILTER layers_SRCS EXCLUDE REGEX ".*/zixiao/.*.cpp")
if(NOT WITH_CUDA)
  list(FILTER layers_SRCS EXCLUDE REGEX ".*/flashinfer_.*\\.cpp")
endif()

message(STATUS "layers_SRCS: ${layers_SRCS}")

set(layers_nvidia_SRCS "")
set(layers_nvidia_LIBS "")

if(WITH_CUDA)
  file(GLOB_RECURSE layers_nvidia_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/nvidia/*.cpp)
  list(APPEND layers_nvidia_LIBS ${NCCL_LIBRARIES} -lcudart -lcublas -lcublasLt)
elseif(WITH_ILUVATAR)
  # For Iluvatar: reuse most nvidia layer cpps, but exclude the ones that
  # depend on NV-only kernels (cutlass / machete / marlin / fp8 / NVLS /
  # flash-MLA / sparse-MLA) or are taken over by layers/iluvatar/
  # implementations using the llm_kernels::iluvatar namespace.
  # Excluding here keeps the nvidia/*.cpp source files completely untouched.
  file(GLOB_RECURSE layers_nvidia_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/nvidia/*.cpp)
  set(_iluvatar_excluded_nvidia_layers
    # taken over by layers/iluvatar/ implementations
    "attention_layer" "paged_attention_layer" "flash_attention_layer"
    # MLA / sparse MLA: NV-only kernels (cutlass / flash-mla / sparse indexer)
    "paged_mla_attention_layer" "flash_mla_attention_layer"
    "paged_sparse_mla_attention_layer" "flash_sparse_mla_attention_layer"
    "paged_sparse_mla_indexer_layer" "flash_sparse_mla_indexer_layer"
    # NV-only collectives (NVLS / custom all-reduce)
    "all_reduce_residual_add_norm_layer" "custom_all_reduce_sum_layer"
    # NV-only matmul / MoE backends (cutlass / machete / marlin / fp8)
    "batched_matmul_layer" "blockwise_matmul_layer"
    "cutlass_matmul_layer" "cutlass_moe_layer"
    "fp8_matmul_layer" "fp8_moe_layer"
    "machete_matmul_layer" "marlin_matmul_layer" "marlin_moe_layer"
    "moe_layer" "finegrained_mixed_dtype_gemm_layer"
  )
  string(REPLACE ";" "|" _iluvatar_excluded_re "${_iluvatar_excluded_nvidia_layers}")
  list(FILTER layers_nvidia_SRCS EXCLUDE REGEX ".*/nvidia/(${_iluvatar_excluded_re})\\.cpp$")
  list(APPEND layers_nvidia_LIBS ${NCCL_LIBRARIES} -lcudart -lcublas -lcublasLt)
endif()

# ---- Iluvatar layers (separate from nvidia, no cross-contamination) ----
set(layers_iluvatar_SRCS "")
set(layers_iluvatar_LIBS "")
if(WITH_ILUVATAR)
  file(GLOB_RECURSE layers_iluvatar_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/iluvatar/*.cpp)
  list(APPEND layers_iluvatar_LIBS ${NCCL_LIBRARIES} -lcudart -lcublas -lcublasLt ${IXINFER_LIBS})
endif()

set(layers_ascend_SRCS "")
set(layers_ascend_LIBS "")

if(WITH_ACL)
  file(GLOB_RECURSE layers_ascend_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/ascend/*.cpp)
  list(APPEND layers_ascend_LIBS ${ACL_SHARED_LIBS} llm_kernels_ascend_utils_common
    atb_attention)
endif()

set(layers_zixiao_SRCS "")
set(layers_zixiao_LIBS "")

if(WITH_TOPS)
  file(GLOB_RECURSE layers_zixiao_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/zixiao/*.cpp)
  list(APPEND layers_zixiao_LIBS ${TOPS_SHARED_LIBS})
endif()

add_library(layers STATIC ${layers_SRCS} ${layers_nvidia_SRCS} ${layers_ascend_SRCS} ${layers_zixiao_SRCS} ${layers_iluvatar_SRCS})
target_link_libraries(layers PUBLIC
  kernels ${layers_nvidia_LIBS} ${layers_ascend_LIBS} ${layers_iluvatar_LIBS} layer_progress_tracker data_hub
)

# for test
set(layers_test_nvidia_SRCS "")
set(layers_test_ascend_SRCS "")
set(layers_test_zixiao_SRCS "")

if(WITH_CUDA)
  file(GLOB_RECURSE layers_test_nvidia_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/nvidia/*test.cpp)
endif()

if(WITH_ACL)
  file(GLOB_RECURSE layers_test_ascend_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/ascend/*test.cpp)
endif()

file(GLOB_RECURSE layers_test_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/*test.cpp
  ${layers_test_nvidia_SRCS} ${layers_test_ascend_SRCS})
message(STATUS "layers_test_SRCS: ${layers_test_SRCS}")
cpp_test(layers_test SRCS ${layers_test_SRCS} DEPS layers runtime utils)
