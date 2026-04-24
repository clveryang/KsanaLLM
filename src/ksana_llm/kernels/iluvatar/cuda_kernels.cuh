#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace ksana_llm {


// ============================================================
// Fused RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * weight
// ============================================================
void rms_norm(
    __half* output,
    const __half* input,
    const __half* weight,
    int seq_len,
    int hidden_size,
    float eps,
    cudaStream_t stream
);

// ============================================================
// RoPE: Apply rotary positional embedding in-place
// ============================================================
void apply_rope(
    __half* q,
    __half* k,
    const float* cos,
    const float* sin,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int past_len,
    cudaStream_t stream
);

// Graph-compatible RoPE: reads past_len from GPU pointer (seq_len=1)
void apply_rope_graph(
    __half* q,
    __half* k,
    const float* cos,
    const float* sin,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    const int* past_len_gpu,
    cudaStream_t stream
);

// ============================================================
// Fused SiLU Gate MLP: output = silu(gate) * up
// ============================================================
void silu_mul(
    __half* output,
    const __half* gate,
    const __half* up,
    int count,
    cudaStream_t stream
);

// ============================================================
// Element-wise multiply: output = a * b
// ============================================================
void elementwise_mul(
    __half* output,
    const __half* a,
    const __half* b,
    int count,
    cudaStream_t stream
);

// ============================================================
// Residual add: output = x + residual (in-place on output)
// ============================================================
void residual_add(
    __half* output,
    const __half* residual,
    int count,
    cudaStream_t stream
);

// ============================================================
// Phase 3: Fused residual add + RMSNorm
// normed_output = rmsnorm(residual_io + input, weight)
// residual_io is updated in-place: residual_io += input
// ============================================================
void fused_residual_rmsnorm(
    __half* normed_output,
    __half* residual_io,
    const __half* input,
    const __half* weight,
    int seq_len,
    int hidden_size,
    float eps,
    cudaStream_t stream
);

// ============================================================
// Embedding lookup: output[i] = table[ids[i]]
// ============================================================
void embedding_lookup(
    __half* output,
    const __half* table,
    const int* ids,
    int seq_len,
    int hidden_size,
    cudaStream_t stream
);

// ============================================================
// Add bias: output[i] += bias[i % bias_size]
// ============================================================
void add_bias(
    __half* output,
    const __half* bias,
    int count,
    int bias_size,
    cudaStream_t stream
);

// ============================================================
// fp16 -> fp32 conversion for sampling
// ============================================================
void half_to_fp32(
    float* output,
    const __half* input,
    int count,
    cudaStream_t stream
);

// ============================================================
// GQA Attention (fallback)
// ============================================================
void gqa_attention(
    __half* output,
    const __half* q,
    const __half* k_cache,
    const __half* v_cache,
    float* attn_scores,
    int seq_len,
    int past_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    cudaStream_t stream
);

// ============================================================
// Scatter KV from SHD to BHSD cache layout
// ============================================================
void scatter_kv_to_cache(
    __half* dst_cache,
    const __half* src,
    int seq_len,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int start_pos,
    cudaStream_t stream
);

// Graph-compatible scatter KV: reads start_pos from GPU pointer (seq_len=1)
void fused_rope_scatter(
    __half* q,
    const __half* k, const __half* v,
    __half* k_cache, __half* v_cache,
    const float* cos_tab, const float* sin_tab,
    int num_q_heads, int num_kv_heads,
    int head_dim, int max_seq_len,
    const int* past_len_gpu,
    cudaStream_t stream
);

void scatter_kv_to_cache_graph(
    __half* dst_cache,
    const __half* src,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    const int* start_pos_gpu,
    cudaStream_t stream
);

// ============================================================
// GPU-side sampling: temperature + top-k + top-p + categorical
// ============================================================
void gpu_sample(
    int* output_token,
    const __half* logits_fp16,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    const float* rand_val,
    cudaStream_t stream
);

// ============================================================
// Phase 4: GPU-side repetition penalty
// ============================================================
void apply_rep_penalty(
    __half* logits,
    const int* generated_ids,
    int num_ids,
    int vocab_size,
    float rep_penalty,
    cudaStream_t stream
);

// ============================================================
// CUDA Graph utility kernels
// ============================================================
void set_gpu_value(int* ptr, int value, cudaStream_t stream);
void update_cache_seqlens(int* seqlens, const int* position_counter, cudaStream_t stream);
void increment_position(int* counter, cudaStream_t stream);

// ============================================================
// Phase 6: FP32 repetition penalty
// ============================================================
void apply_rep_penalty_fp32(
    float* logits,
    const int* generated_ids,
    int num_ids,
    int vocab_size,
    float rep_penalty,
    cudaStream_t stream
);

// ============================================================
// Phase 6: FP32 GPU sampling
// ============================================================
void gpu_sample_fp32(
    int* output_token,
    const float* logits_fp32,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    const float* rand_val,
    cudaStream_t stream
);

// ============================================================
// Phase 2: Split QKV buffer [seq_len, QD+2*KVD] → Q,K,V
// ============================================================
void split_qkv(
    __half* q, __half* k, __half* v,
    const __half* qkv,
    int seq_len, int q_dim, int kv_dim,
    cudaStream_t stream
);

// ============================================================
// Phase 2: Fused split gate_up + SiLU * mul for MLP
// gate_up [seq_len, 2*I] → output [seq_len, I]
// output[s*I+d] = silu(gate_up[s*2I+d]) * gate_up[s*2I+I+d]
// ============================================================
void split_silu_mul(
    __half* output,
    const __half* gate_up,
    int seq_len, int intermediate_size,
    cudaStream_t stream
);

// ============================================================
// Phase 3: Device-to-device single int copy
// ============================================================
void copy_int_d2d(int* dst, const int* src, cudaStream_t stream);


}  // namespace ksana_llm
