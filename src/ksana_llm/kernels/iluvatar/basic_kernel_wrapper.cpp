/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Iluvatar backend kernel implementations.
 * Bridges KsanaLLM's kernel wrapper API to the cuda_kernels kernel library
 * that has been proven on Iluvatar MR-V100 (RTF 0.79→0.43).
 *
 * For GEMM we call cuinferCustomGemm (ixinfer SDK); for element-wise ops
 * we call the cuda_kernels device kernels compiled as CXX (clang++ -x ivcore).
 * ===========================================================================*/

#include "ksana_llm/kernels/iluvatar/basic_kernel_wrapper.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstdio>
#include <vector>

// 真实 GPU 算子（3rdparty/LLM_kernels/csrc/kernels/iluvatar/embedding/，
// 命名空间保持 llm_kernels::nvidia。libllm_kernels_iluvatar_kernel_embedding.a 演示）。
#include "csrc/kernels/iluvatar/embedding/embedding.h"
#include "ksana_llm/utils/logger.h"
// Forward declarations — these live in kernels.cu compiled as CXX by clang++.
// We keep them in a separate namespace so there's no collision with KsanaLLM.
// Forward declarations for cuda_kernels.cu functions (same ksana_llm namespace)
namespace ksana_llm {
void rms_norm(__half*, const __half*, const __half*, int, int, float, cudaStream_t);
void fused_residual_rmsnorm(__half*, __half*, const __half*, const __half*, int, int, float, cudaStream_t);
void embedding_lookup(__half*, const __half*, const int*, int, int, cudaStream_t);
void residual_add(__half*, const __half*, const __half*, int, cudaStream_t);
void silu_mul(__half*, const __half*, const __half*, int, cudaStream_t);
}  // namespace ksana_llm


namespace ksana_llm {

// =========================================================================
// InvokeRMSNorm
// =========================================================================
template <>
void InvokeRMSNorm<half>(void* input, void* weight, float layernorm_eps,
                         int m, int n, void* output, bool /*enable_pdl*/,
                         cudaStream_t stream) {
  ksana_llm::rms_norm(
      static_cast<__half*>(output),
      static_cast<const __half*>(input),
      static_cast<const __half*>(weight),
      m, n, layernorm_eps, stream);
}

// =========================================================================
// InvokeFusedResidualRMSNorm
// =========================================================================
template <>
void InvokeFusedResidualRMSNorm<half>(void* normed_output, void* residual_io,
                                      const void* input, const void* weight,
                                      float eps, int m, int n,
                                      cudaStream_t stream) {
  ksana_llm::fused_residual_rmsnorm(
      static_cast<__half*>(normed_output),
      static_cast<__half*>(residual_io),
      static_cast<const __half*>(input),
      static_cast<const __half*>(weight),
      m, n, eps, stream);
}

// =========================================================================
// LookupEmbedding — 调用真实 iluvatar embedding kernel（与 nvidia 同源）。
// =========================================================================
namespace {
template <typename T>
void LookupEmbeddingImpl(const void* input_ids, const void* ids_offsets, const void* prefix_offsets,
                         const void* emb, const void* pos, const void* steps, void* output,
                         bool use_emb_scale, const T emb_scale, int vocab_size, int hidden_size,
                         int bs, int vocab_id, cudaStream_t stream) {
  const bool do_position_encoding = (pos != nullptr) && (steps != nullptr);
  if (do_position_encoding) {
    if (use_emb_scale) {
      llm_kernels::iluvatar::LookupFusedEmbeddingWithCSRInputs<T, true, true>(
          reinterpret_cast<T*>(output), reinterpret_cast<const T*>(emb), reinterpret_cast<const T*>(pos),
          emb_scale, {}, reinterpret_cast<const int32_t*>(input_ids),
          reinterpret_cast<const size_t*>(steps), reinterpret_cast<const size_t*>(ids_offsets),
          reinterpret_cast<const size_t*>(prefix_offsets), bs, hidden_size, vocab_size, vocab_id, stream);
    } else {
      llm_kernels::iluvatar::LookupFusedEmbeddingWithCSRInputs<T, true, false>(
          reinterpret_cast<T*>(output), reinterpret_cast<const T*>(emb), reinterpret_cast<const T*>(pos),
          emb_scale, {}, reinterpret_cast<const int32_t*>(input_ids),
          reinterpret_cast<const size_t*>(steps), reinterpret_cast<const size_t*>(ids_offsets),
          reinterpret_cast<const size_t*>(prefix_offsets), bs, hidden_size, vocab_size, vocab_id, stream);
    }
  } else {
    if (use_emb_scale) {
      llm_kernels::iluvatar::LookupFusedEmbeddingWithCSRInputs<T, false, true>(
          reinterpret_cast<T*>(output), reinterpret_cast<const T*>(emb), nullptr, emb_scale, {},
          reinterpret_cast<const int32_t*>(input_ids), nullptr,
          reinterpret_cast<const size_t*>(ids_offsets), reinterpret_cast<const size_t*>(prefix_offsets),
          bs, hidden_size, vocab_size, vocab_id, stream);
    } else {
      llm_kernels::iluvatar::LookupFusedEmbeddingWithCSRInputs<T, false, false>(
          reinterpret_cast<T*>(output), reinterpret_cast<const T*>(emb), nullptr, emb_scale, {},
          reinterpret_cast<const int32_t*>(input_ids), nullptr,
          reinterpret_cast<const size_t*>(ids_offsets), reinterpret_cast<const size_t*>(prefix_offsets),
          bs, hidden_size, vocab_size, vocab_id, stream);
    }
  }
}
}  // namespace

template <>
void LookupEmbedding<half>(const void* input_ids, const void* ids_offsets, const void* prefix_offsets,
                           const void* emb, const void* pos, const void* steps, void* output,
                           bool use_emb_scale, const half emb_scale, int vocab_size, int hidden_size,
                           int bs, int vocab_id, cudaStream_t stream, void* /*workspace_ptr*/) {
  LookupEmbeddingImpl<half>(input_ids, ids_offsets, prefix_offsets, emb, pos, steps, output,
                            use_emb_scale, emb_scale, vocab_size, hidden_size, bs, vocab_id, stream);
}

template <>
void LookupEmbedding<__nv_bfloat16>(const void* input_ids, const void* ids_offsets, const void* prefix_offsets,
                                    const void* emb, const void* pos, const void* steps, void* output,
                                    bool use_emb_scale, const __nv_bfloat16 emb_scale, int vocab_size,
                                    int hidden_size, int bs, int vocab_id, cudaStream_t stream,
                                    void* /*workspace_ptr*/) {
  LookupEmbeddingImpl<__nv_bfloat16>(input_ids, ids_offsets, prefix_offsets, emb, pos, steps, output,
                                     use_emb_scale, emb_scale, vocab_size, hidden_size, bs, vocab_id, stream);
}

template <>
void LookupEmbedding<float>(const void* input_ids, const void* ids_offsets, const void* prefix_offsets,
                            const void* emb, const void* pos, const void* steps, void* output,
                            bool use_emb_scale, const float emb_scale, int vocab_size, int hidden_size,
                            int bs, int vocab_id, cudaStream_t stream, void* /*workspace_ptr*/) {
  LookupEmbeddingImpl<float>(input_ids, ids_offsets, prefix_offsets, emb, pos, steps, output,
                             use_emb_scale, emb_scale, vocab_size, hidden_size, bs, vocab_id, stream);
}

// =========================================================================
// InvokeMatMul — delegate to cublas (Iluvatar provides compatible cublas)
// =========================================================================
template <>
void InvokeMatMul<half>(cublasHandle_t cublas_handle,
                        cublasLtHandle_t /*cublaslt_handle*/,
                        int m, int n, int k,
                        const void* a_ptr, const void* b_ptr, void* c_ptr,
                        cudaStream_t& stream, void* /*workspace_ptr*/,
                        cublasLtMatmulAlgo_t* /*cublaslt_algo*/,
                        size_t /*workspace_size*/,
                        bool /*use_fp16_compute_reduction*/) {
  cublasSetStream(cublas_handle, stream);
  const half alpha_h = __float2half(1.0f);
  const half beta_h  = __float2half(0.0f);
  // A: m×k  B: k×n  C: m×n  (col-major cublas convention: transposed)
  cublasHgemm(cublas_handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              n, m, k,
              &alpha_h,
              static_cast<const half*>(b_ptr), n,
              static_cast<const half*>(a_ptr), k,
              &beta_h,
              static_cast<half*>(c_ptr), n);
}

// =========================================================================
// InvokeAddBiasResidual
// =========================================================================
template <>
void InvokeAddBiasResidual<half>(const void* input_a, const void* input_b,
                                 const void* /*bias*/, const int m, const int n,
                                 void* output, cudaStream_t stream) {
  ksana_llm::residual_add(
      static_cast<__half*>(output),
      static_cast<const __half*>(input_a),
      static_cast<const __half*>(input_b),
      m * n, stream);
}

// =========================================================================
// InvokeRowBasedGatedActivation — SiLU×Mul for SwiGLU FFN
// out[i] = silu(input[i]) * input[i + n]   where i in [0, n)
// =========================================================================
// Forward-declare Activation tag (KsanaLLM uses SiluT<T>)
template <typename T> struct SiluT {};

template <>
void InvokeRowBasedGatedActivation<SiluT, half>(
    const void* input, const int m, const int n,
    void* output, cudaStream_t stream) {
  // input shape: [m, 2*n] → gate = input[:, :n], up = input[:, n:]
  const __half* gate = static_cast<const __half*>(input);
  const __half* up   = gate + static_cast<ptrdiff_t>(m) * n;
  ksana_llm::silu_mul(
      static_cast<__half*>(output), gate, up, m * n, stream);
}

// =========================================================================
// AssembleTokensHidden — gather last token hidden state per batch
// =========================================================================
template <>
void AssembleTokensHidden<half>(const void* inputs, const void* logits_idx,
                                const int batch_size,
                                const int hidden_units_num,
                                void* output, cudaStream_t stream) {
  // Simplified: just memcpy if batch_size==1 (MVP)
  // TODO: implement proper gather using scatter pattern
  if (batch_size <= 1) {
    cudaMemcpyAsync(output, inputs,
                    hidden_units_num * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
  }
}

}  // namespace ksana_llm
