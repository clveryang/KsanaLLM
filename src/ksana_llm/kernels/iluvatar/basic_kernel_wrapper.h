/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Iluvatar backend — basic kernel wrappers.
 * Implements the same public API as nvidia/basic_kernel_wrapper.h but backed
 * by triton_qwen2 kernels + cuinferCustomGemm (ixinfer SDK).
 * ===========================================================================*/
#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <optional>

#include "csrc/kernels/iluvatar/rotary_embedding/rotary_embedding.h"
#include "csrc/utils/quant_type.h"

namespace ksana_llm {

// ------- RMSNorm ---------------------------------------------------------
// Matches nvidia InvokeRMSNorm signature.
// Uses triton_qwen2::kernels::rms_norm internally.
template <typename T>
void InvokeRMSNorm(void* input, void* weight, float layernorm_eps, int m, int n,
                   void* output, bool enable_pdl, cudaStream_t stream);

// ------- Fused Residual + RMSNorm ----------------------------------------
template <typename T>
void InvokeFusedResidualRMSNorm(void* normed_output, void* residual_io,
                                const void* input, const void* weight,
                                float eps, int m, int n, cudaStream_t stream);

// ------- Embedding -------------------------------------------------------
template <typename T>
void LookupEmbedding(const void* input_ids, const void* ids_offsets,
                     const void* prefix_offsets, const void* emb,
                     const void* pos, const void* steps, void* output,
                     bool use_emb_scale, const T emb_scale,
                     int vocab_size, int hidden_size, int bs, int vocab_id,
                     cudaStream_t stream, void* workspace_ptr = nullptr);

// ------- MatMul (cublas) -------------------------------------------------
template <typename T>
void InvokeMatMul(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                  int m, int n, int k,
                  const void* a_ptr, const void* b_ptr, void* c_ptr,
                  cudaStream_t& stream, void* workspace_ptr,
                  cublasLtMatmulAlgo_t* cublaslt_algo,
                  size_t workspace_size = 0,
                  bool use_fp16_compute_reduction = false);

// ------- Residual Add ----------------------------------------------------
template <typename T>
void InvokeAddBiasResidual(const void* input_a, const void* input_b,
                           const void* bias, const int m, const int n,
                           void* output, cudaStream_t stream);

// ------- SiLU × Mul (gated activation) -----------------------------------
// out = silu(input[:, :n]) * input[:, n:]
template <template <typename T2> class Activation, typename T>
void InvokeRowBasedGatedActivation(const void* input, const int m, const int n,
                                   void* output, cudaStream_t stream);

// ------- AssembleTokensHidden (gather for sampling) ----------------------
template <typename T>
void AssembleTokensHidden(const void* inputs, const void* logits_idx,
                          const int batch_size, const int hidden_units_num,
                          void* output, cudaStream_t stream);

// ------- Attention (paged decode + varlen prefill) -----------------------
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokePagedAttention(
    void* output_ptr, void* query_ptr, void** key_cache_ptrs, void** value_cache_ptrs,
    void* context_lens_ptr, int max_context_len, cudaStream_t stream, void* cache_offsets_ptr,
    int seqs_num, int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size,
    float k_scale, float v_scale, int batch, void* rotary_embedding_pos,
    void* rotary_embedding_mask, int total_tokens,
    std::optional<llm_kernels::iluvatar::RotaryEmbeddingCuda>& rotary_embedding_cuda,
    void* workspace_ptr, float layernorm_eps, bool use_qk_norm, void* q_norm_weight,
    void* k_norm_weight, size_t work_size, int rank, const std::optional<void*>& alibi_slopes,
    void* qkv_workspace, void* flashinfer_extra_workspace, void* page_locked_workspace,
    void* k_cache_ptr, void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num,
    int max_blocks_per_seq, bool enable_qk_pre_norm_before_rotary_pos, bool no_rope,
    bool attn_temperature_tuning, float attn_scale, size_t floor_scale,
    bool enable_blocked_multi_token_forwarding_kv, bool is_first_layer_on_node,
    bool use_flashinfer_for_decode, void* flashinfer_prefill_helper);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void AttenVarlen(void* qkv_ptr, void* rotary_embedding_pos, void* rotary_embedding_mask, void* out, void* seqlen,
                 std::optional<llm_kernels::iluvatar::RotaryEmbeddingCuda>& rotary_embedding_cuda, int total_tokens,
                 int max_tokens, int batch, int num_heads, int num_kv_heads, int head_size, int stride_size,
                 float k_scale, float v_scale, size_t tensor_para_size, bool is_causal, int rank, int block_size,
                 void** k_list, void** v_list, void* prefix_offsets, void* block_offsets,
                 const std::optional<void*>& alibi_slopes, int layer_index, void* flexible_rotary_embedding_pos_ptr,
                 void* flexible_rotary_embedding_mask_ptr, void* dst_flexible_kv_cache_ptr,
                 void* src_flexible_kv_cache_ptr, void* dst_flexible_token_idx_ptr, void* src_flexible_token_idx_ptr,
                 void* flexible_offset_uint64_ptr, int flexible_len, float layernorm_eps, bool use_qk_norm,
                 void* q_norm_weight, void* k_norm_weight, bool use_cache, cudaStream_t stream, void* k_cache_ptr,
                 void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq,
                 size_t* without_prefix_offsets, int max_forwarding_tokens, bool enable_qk_pre_norm_before_rotary_pos,
                 bool no_rope, bool attn_temperature_tuning, float attn_scale, size_t floor_scale,
                 bool enable_blocked_multi_token_forwarding_kv, bool use_flashinfer_for_decode);

}  // namespace ksana_llm
