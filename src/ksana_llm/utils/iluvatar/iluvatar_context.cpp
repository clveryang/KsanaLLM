/* Iluvatar-specific Context initialization.
 * Simplified version of nvidia_context.cpp:
 * - Skip P2P access (single GPU for MVP)
 * - Skip NVLS multicast
 * - Skip NCCL init for TP=1
 * - Use simpler memory pool init (no cudaMallocAsync)
 * ===========================================================================*/
#ifdef ENABLE_ILUVATAR

#include "ksana_llm/utils/iluvatar/iluvatar_context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

// --- Methods that nvidia_context.cpp normally provides ---

template <>
void NvidiaContextExtension<DEVICE_TYPE_NVIDIA>::InitGpuMemoryPool(const int worker_id) {
  // Iluvatar: skip cudaMallocAsync-based memory pool (may not be supported)
  KLLM_LOG_INFO << "Iluvatar: skip GPU memory pool for worker " << worker_id;
}

template <>
void NvidiaContextExtension<DEVICE_TYPE_NVIDIA>::InitCublasHandle(const int worker_id) {
  KLLM_LOG_INFO << "Iluvatar: init cublas on worker " << worker_id;
  CUDA_CHECK(cublasCreate(&cublas_handles_[worker_id]));
  CUDA_CHECK(cublasLtCreate(&cublaslt_handles_[worker_id]));
  base_ptr_->is_gemm_fp8_supported_ = false;  // Iluvatar MR-V100 no FP8
  CUDA_CHECK(cublasSetStream(cublas_handles_[worker_id], base_ptr_->compute_streams_[worker_id].Get()));
}

template <>
void NvidiaContextExtension<DEVICE_TYPE_NVIDIA>::InitNcclParam() {
  // Iluvatar: minimal NCCL init for TP>1 (not needed for single GPU)
  const size_t world_size = base_ptr_->tensor_parallel_size_;
  reduce_signals_.resize(world_size);
  reduce_inputs_.resize(world_size);
  trt_reduce_buffers_.resize(3 * world_size);
  trt_reduce_flags_.resize(world_size);
  trt_reduce_workspaces_.resize(world_size);
  nccl_params_.resize(world_size);
  // Skip actual ncclCommInitRank for single GPU
  init_done_.store(true, std::memory_order_relaxed);
}

template <>
bool NvidiaContextExtension<DEVICE_TYPE_NVIDIA>::EnableGpuP2PAccess() {
  // Iluvatar single GPU: no P2P needed
  return false;
}

template <>
void NvidiaContextExtension<DEVICE_TYPE_NVIDIA>::Destroy() {
  for (size_t i = 0; i < cublas_handles_.size(); ++i) {
    if (cublas_handles_[i]) cublasDestroy(cublas_handles_[i]);
    if (cublaslt_handles_[i]) cublasLtDestroy(cublaslt_handles_[i]);
  }
}

template <>
void NvidiaContextExtension<DEVICE_TYPE_NVIDIA>::Initialize() {
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    KLLM_THROW("No GPUs detected.");
  }
  CUDA_CHECK(cudaDriverGetVersion(&base_ptr_->driver_version_));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  sm_ = prop.major * 10 + prop.minor;
  int cuda_ver_tmp;
  CUDA_CHECK(cudaRuntimeGetVersion(&cuda_ver_tmp));
  cuda_ver_ = static_cast<uint32_t>(cuda_ver_tmp);
  KLLM_LOG_INFO << fmt::format("Iluvatar: SM={}, CUDA version={}", sm_, cuda_ver_);

  memory_pool_.resize(base_ptr_->tensor_parallel_size_);
  cublas_handles_.resize(base_ptr_->tensor_parallel_size_);
  cublaslt_handles_.resize(base_ptr_->tensor_parallel_size_);

  // Single GPU: skip P2P
  is_p2p_enable_ = false;
  is_full_nvlink_ = false;
  is_multicast_enable_ = false;

  // Init cublas per worker (no threads — Iluvatar needs sequential init)
  for (size_t worker_id = 0; worker_id < base_ptr_->tensor_parallel_size_; ++worker_id) {
    KLLM_LOG_INFO << "Iluvatar: init worker " << worker_id;
    CUDA_CHECK(cudaSetDevice(worker_id));
    // Skip memory pool (Iluvatar may not support cudaMallocAsync)
    InitCublasHandle(worker_id);
    KLLM_LOG_INFO << "Iluvatar: worker " << worker_id << " cublas OK";
  }

  KLLM_LOG_INFO << fmt::format("Iluvatar context: p2p={}, nvlink={}, multicast={}",
                               is_p2p_enable_, is_full_nvlink_, is_multicast_enable_);

  // Skip NCCL for single GPU — but still init the vectors that other code reads
  const size_t world_size = base_ptr_->tensor_parallel_size_;
  reduce_signals_.resize(world_size, nullptr);
  reduce_inputs_.resize(world_size, nullptr);
  trt_reduce_buffers_.resize(3 * world_size, nullptr);
  trt_reduce_flags_.resize(world_size, nullptr);
  trt_reduce_workspaces_.resize(world_size, nullptr);
  nccl_params_.resize(world_size);
  if (world_size > 1) {
    std::thread([this]() { InitNcclParam(); }).detach();
  } else {
    init_done_.store(true, std::memory_order_relaxed);
    KLLM_LOG_INFO << "Iluvatar: single GPU, skip NCCL init";
  }

  CUDA_CHECK(cudaSetDevice(base_ptr_->defalt_device_id_));
  KLLM_LOG_INFO << "Iluvatar: Context initialized OK";
}

// Provide InitializeExtension for the ContextT template
template <>
void ContextT<DEVICE_TYPE_NVIDIA>::InitializeExtension() {
  ext = new NvidiaContextExtension<DEVICE_TYPE_NVIDIA>(this);
  ext->Initialize();
}

template <>
void ContextT<DEVICE_TYPE_NVIDIA>::DestroyExtension() {
  ext->Destroy();
  delete ext;
}

}  // namespace ksana_llm
#endif
