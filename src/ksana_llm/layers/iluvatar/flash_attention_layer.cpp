/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * Iluvatar 版 FlashAttentionLayer.
 * 与 layers/nvidia/flash_attention_layer.cpp 对齐，差别：
 *   - 走 ksana_llm/kernels/iluvatar/attention_kernel_wrapper.h 的 AttenVarlen
 *     模板（参数 std::optional<iluvatar::RotaryEmbeddingCuda>&）
 *   - iluvatar 不支持 FlashInfer
 * ===========================================================================*/

#include "ksana_llm/layers/flash_attention_layer.h"
#include "ksana_llm/kernels/iluvatar/basic_kernel_wrapper.h"
#include "ksana_llm/runtime/layer_progress_tracker.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

Status FlashAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                 std::shared_ptr<Context> context, int rank) {
  enable_blocked_multi_token_forwarding_kv_ =
      runtime_config.attn_backend_config.enable_blocked_multi_token_forwarding_kv;
  return AttentionLayer::Init(parameters, runtime_config, context, rank);
}

Status FlashAttentionLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_DTYPE_AND_KVTYPE(inter_data_type_, kv_cache_dtype_, ForwardT, input_tensors, output_tensors);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashAttentionLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  int max_tokens = input_tensors[14].shape[8];
  int batch_size = input_tensors[14].shape[7];
  int layer_block_num = input_tensors[14].shape[2];
  int total_tokens = input_tensors[0].shape[0] - input_tensors[14].shape[3];
  bool use_cache = input_tensors[17].GetPtr<bool>()[0];

  void** k_list = (input_tensors[2].GetPtr<void*>()) + this->layer_index_ * layer_block_num * 2;
  void** v_list = k_list + layer_block_num;

  int64_t kv_cache_block_num = 0;
  void** layer_kv_cache_ptr = nullptr;
  void* k_cache_ptr = nullptr;
  void* v_cache_ptr = nullptr;
  int32_t* block_table_ptr = nullptr;
  int max_blocks_per_seq = 0;
  size_t* input_without_prefix_offset = nullptr;
  int max_forwarding_tokens = 0;

  if (enable_blocked_multi_token_forwarding_kv_) {
    kv_cache_block_num = *(input_tensors[18].GetPtr<int64_t>());
    layer_kv_cache_ptr = input_tensors[18].GetPtr<void*>() + 1;
    k_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2];
    v_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2 + 1];
    block_table_ptr = input_tensors[19].GetPtr<int32_t>();
    max_blocks_per_seq = input_tensors[19].shape[1];
    input_without_prefix_offset = input_tensors[20].GetPtr<size_t>();
    max_forwarding_tokens = input_tensors[14].shape[6];
  }

  AttenVarlen<SCALAR_T, CACHE_T, KV_DTYPE>(
      input_tensors[0].GetPtr<void>(), input_tensors[5].GetPtr<void>(), input_tensors[6].GetPtr<void>(),
      output_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), this->rotary_embedding_cuda_, total_tokens,
      max_tokens, batch_size, this->num_heads_, this->num_kv_heads_, this->head_size_, this->stride_size_,
      this->k_scale_, this->v_scale_, this->tensor_para_size_, this->is_causal_, this->rank_, this->block_token_num_,
      k_list, v_list, input_tensors[3].GetPtr<void>(), input_tensors[4].GetPtr<void>(), this->alibi_slopes_,
      this->layer_index_, input_tensors[7].GetPtr<void>(), input_tensors[8].GetPtr<void>(),
      input_tensors[9].GetPtr<void>(), input_tensors[10].GetPtr<void>(), input_tensors[11].GetPtr<void>(),
      input_tensors[12].GetPtr<void>(), input_tensors[13].GetPtr<void>(), input_tensors[9].shape[0],
      this->layernorm_eps_, this->use_qk_norm_, input_tensors[15].GetPtr<void>(), input_tensors[16].GetPtr<void>(),
      use_cache, this->context_->GetComputeStreams()[this->rank_].Get(), k_cache_ptr, v_cache_ptr, block_table_ptr,
      kv_cache_block_num, max_blocks_per_seq, input_without_prefix_offset, max_forwarding_tokens,
      this->enable_qk_pre_norm_before_rotary_pos_, this->no_rope_, this->attn_temperature_tuning_, this->attn_scale_,
      this->floor_scale_, this->enable_blocked_multi_token_forwarding_kv_, /*use_flashinfer_for_decode=*/false);

  Singleton<LayerProgressTracker>::GetInstance()->RecordLayerProgress(
      this->rank_, this->layer_index_, this->context_->GetComputeStreams()[this->rank_]);

  output_tensors[0].shape[0] = total_tokens;
  output_tensors[0].shape[1] = this->num_heads_ * this->head_size_;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

}  // namespace ksana_llm
