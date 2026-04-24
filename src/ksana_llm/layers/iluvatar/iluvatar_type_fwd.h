/* Iluvatar type forward declarations.
 *
 * 原则:
 *  - iluvatar 自己有移植版的 header (rotary_embedding) 走 csrc/kernels/iluvatar/...
 *  - iluvatar 暂未移植、仅做 type-only 共享声明的 header (GPUGemmAlgoHelper /
 *    Activation tag 等) 继续从 csrc/kernels/nvidia/... 取, 不会引入 nvidia .cu
 *    链接依赖, 因为 iluvatar 不编 kernels/nvidia/*.cpp.
 * ===========================================================================*/
#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>
#include <type_traits>
#include <cmath>

#include "csrc/utils/quant_type.h"
#include "csrc/kernels/iluvatar/rotary_embedding/rotary_embedding.h"
// type-only headers (no .cu link dependency):
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_algo_map.h"
#include "csrc/kernels/nvidia/activation/activation.h"
