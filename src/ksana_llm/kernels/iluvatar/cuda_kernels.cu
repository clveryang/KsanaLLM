#include <algorithm>
#include "cuda_kernels.cuh"
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <vector>

#include <cuda_bf16.h>
#include <ixinfer.h>

namespace ksana_llm {

namespace {
inline cuinferHandle_t GetCuinferHandle() {
  static cuinferHandle_t handle = []() {
    cuinferHandle_t h;
    cuinferStatus_t s = cuinferCreate(&h);
    if (s != CUINFER_STATUS_SUCCESS) {
      fprintf(stderr, "[ksana-iluvatar] cuinferCreate failed: %s\n",
              cuinferGetErrorString(s));
      std::abort();
    }
    return h;
  }();
  return handle;
}

__global__ void split_qkv_bf16_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    int N, int nq, int nkv, int hd) {
  int t = blockIdx.x;
  int row_stride = (nq + 2 * nkv) * hd;
  int q_size  = nq  * hd;
  int kv_size = nkv * hd;
  for (int idx = threadIdx.x; idx < row_stride; idx += blockDim.x) {
    __nv_bfloat16 val = qkv[t * row_stride + idx];
    if (idx < q_size) {
      q[t * q_size + idx] = val;
    } else if (idx < q_size + kv_size) {
      k[t * kv_size + (idx - q_size)] = val;
    } else {
      v[t * kv_size + (idx - q_size - kv_size)] = val;
    }
  }
}
}  // namespace

// ============================================================
// FP16 kernels (used by basic_kernel_wrapper.cpp)
// ============================================================

__global__ void rms_norm_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    int hidden_size, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    const __half* x = input + row * hidden_size;
    __half* y = output + row * hidden_size;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x + 31) / 32) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    if (threadIdx.x == 0) smem[0] = rsqrtf(sum_sq / hidden_size + eps);
    __syncthreads();
    float inv_rms = smem[0];
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(x[i]) * inv_rms;
        float w = __half2float(weight[i]);
        y[i] = __float2half(val * w);
    }
}

void rms_norm(__half* output, const __half* input, const __half* weight,
              int seq_len, int hidden_size, float eps, cudaStream_t stream) {
    int threads = std::min(1024, (hidden_size + 31) / 32 * 32);
    int smem = (threads / 32 + 1) * sizeof(float);
    rms_norm_kernel<<<seq_len, threads, smem, stream>>>(output, input, weight, hidden_size, eps);
}

__global__ void fused_residual_rmsnorm_kernel(
    __half* __restrict__ normed_output,
    __half* __restrict__ residual_io,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    int hidden_size, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    __half* res_row = residual_io + row * hidden_size;
    const __half* inp_row = input + row * hidden_size;
    __half* out_row = normed_output + row * hidden_size;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float r = __half2float(res_row[i]);
        float x = __half2float(inp_row[i]);
        float val = r + x;
        res_row[i] = __float2half(val);
        sum_sq += val * val;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x + 31) / 32) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    if (threadIdx.x == 0) smem[0] = rsqrtf(sum_sq / hidden_size + eps);
    __syncthreads();
    float inv_rms = smem[0];
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(res_row[i]) * inv_rms;
        float w = __half2float(weight[i]);
        out_row[i] = __float2half(val * w);
    }
}

void fused_residual_rmsnorm(__half* normed_output, __half* residual_io,
                            const __half* input, const __half* weight,
                            int seq_len, int hidden_size, float eps, cudaStream_t stream) {
    int threads = std::min(1024, (hidden_size + 31) / 32 * 32);
    int smem = (threads / 32 + 1) * sizeof(float);
    fused_residual_rmsnorm_kernel<<<seq_len, threads, smem, stream>>>(
        normed_output, residual_io, input, weight, hidden_size, eps);
}

__global__ void silu_mul_kernel(
    __half* __restrict__ output, const __half* __restrict__ gate,
    const __half* __restrict__ up, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float silu_g = g / (1.0f + expf(-g));
    output[idx] = __float2half(silu_g * u);
}

void silu_mul(__half* output, const __half* gate, const __half* up,
              int count, cudaStream_t stream) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads, 0, stream>>>(output, gate, up, count);
}

__global__ void residual_add_kernel(
    __half* __restrict__ output, const __half* __restrict__ residual, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    float a = __half2float(output[idx]);
    float b = __half2float(residual[idx]);
    output[idx] = __float2half(a + b);
}

void residual_add(__half* output, const __half* residual, int count, cudaStream_t stream) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads, 0, stream>>>(output, residual, count);
}

__global__ void embedding_kernel(
    __half* __restrict__ output, const __half* __restrict__ table,
    const int* __restrict__ ids, int seq_len, int hidden_size) {
    int s = blockIdx.x;
    int token_id = ids[s];
    const __half* row = table + token_id * hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        output[s * hidden_size + i] = row[i];
    }
}

void embedding_lookup(__half* output, const __half* table, const int* ids,
                      int seq_len, int hidden_size, cudaStream_t stream) {
    embedding_kernel<<<seq_len, 256, 0, stream>>>(output, table, ids, seq_len, hidden_size);
}

}  // namespace ksana_llm

// ============================================================
// BFloat16 kernels
// ============================================================
namespace ksana_llm {

__global__ void rms_norm_bf16_kernel(
    __nv_bfloat16* __restrict__ output, const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight, int hidden_size, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    const __nv_bfloat16* x = input + row * hidden_size;
    __nv_bfloat16* y = output + row * hidden_size;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(x[i]);
        sum_sq += val * val;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    if (lane_id == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x + 31) / 32) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    if (threadIdx.x == 0) smem[0] = rsqrtf(sum_sq / hidden_size + eps);
    __syncthreads();
    float inv_rms = smem[0];
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(x[i]) * inv_rms;
        float w = __bfloat162float(weight[i]);
        y[i] = __float2bfloat16(val * w);
    }
}

void rms_norm_bf16(__nv_bfloat16* output, const __nv_bfloat16* input, const __nv_bfloat16* weight,
                   int seq_len, int hidden_size, float eps, cudaStream_t stream) {
    int threads = std::min(1024, (hidden_size + 31) / 32 * 32);
    int smem = (threads / 32 + 1) * sizeof(float);
    rms_norm_bf16_kernel<<<seq_len, threads, smem, stream>>>(output, input, weight, hidden_size, eps);
}

__global__ void silu_mul_rowbased_bf16_kernel(__nv_bfloat16* __restrict__ out,
                                              const __nv_bfloat16* __restrict__ input,
                                              int m, int hidden) {
    int row = blockIdx.x;
    if (row >= m) return;
    const int row_stride = 2 * hidden;
    const __nv_bfloat16* val_row  = input + (size_t)row * row_stride;
    const __nv_bfloat16* gate_row = val_row + hidden;
    __nv_bfloat16* out_row = out + (size_t)row * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = __bfloat162float(val_row[j]);
        float g = __bfloat162float(gate_row[j]);
        float silu_v = v / (1.0f + expf(-v));
        out_row[j] = __float2bfloat16(silu_v * g);
    }
}

void silu_mul_rowbased_bf16(__nv_bfloat16* output, const __nv_bfloat16* input,
                            int m, int hidden, cudaStream_t stream) {
    if (m <= 0 || hidden <= 0) return;
    int threads = (hidden < 256) ? hidden : 256;
    silu_mul_rowbased_bf16_kernel<<<m, threads, 0, stream>>>(output, input, m, hidden);
}

__global__ void silu_mul_rowbased_fp16_kernel(__half* __restrict__ out,
                                              const __half* __restrict__ input,
                                              int m, int hidden) {
    int row = blockIdx.x;
    if (row >= m) return;
    const int row_stride = 2 * hidden;
    const __half* val_row  = input + (size_t)row * row_stride;
    const __half* gate_row = val_row + hidden;
    __half* out_row = out + (size_t)row * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = __half2float(val_row[j]);
        float g = __half2float(gate_row[j]);
        float silu_v = v / (1.0f + expf(-v));
        out_row[j] = __float2half(silu_v * g);
    }
}

void silu_mul_rowbased_fp16(__half* output, const __half* input,
                            int m, int hidden, cudaStream_t stream) {
    if (m <= 0 || hidden <= 0) return;
    int threads = (hidden < 256) ? hidden : 256;
    silu_mul_rowbased_fp16_kernel<<<m, threads, 0, stream>>>(output, input, m, hidden);
}

__global__ void silu_mul_2input_bf16_kernel(__nv_bfloat16* __restrict__ out,
                                            const __nv_bfloat16* __restrict__ gate,
                                            const __nv_bfloat16* __restrict__ up,
                                            int m, int hidden) {
    int row = blockIdx.x;
    if (row >= m) return;
    size_t off = (size_t)row * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float g = __bfloat162float(gate[off + j]);
        float u = __bfloat162float(up[off + j]);
        float silu_g = g / (1.0f + expf(-g));
        out[off + j] = __float2bfloat16(silu_g * u);
    }
}

void silu_mul_2input_bf16(__nv_bfloat16* output, const __nv_bfloat16* gate, const __nv_bfloat16* up,
                          int m, int hidden, cudaStream_t stream) {
    if (m <= 0 || hidden <= 0) return;
    int threads = (hidden < 256) ? hidden : 256;
    silu_mul_2input_bf16_kernel<<<m, threads, 0, stream>>>(output, gate, up, m, hidden);
}

__global__ void silu_mul_2input_fp16_kernel(__half* __restrict__ out,
                                            const __half* __restrict__ gate,
                                            const __half* __restrict__ up,
                                            int m, int hidden) {
    int row = blockIdx.x;
    if (row >= m) return;
    size_t off = (size_t)row * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float g = __half2float(gate[off + j]);
        float u = __half2float(up[off + j]);
        float silu_g = g / (1.0f + expf(-g));
        out[off + j] = __float2half(silu_g * u);
    }
}

void silu_mul_2input_fp16(__half* output, const __half* gate, const __half* up,
                          int m, int hidden, cudaStream_t stream) {
    if (m <= 0 || hidden <= 0) return;
    int threads = (hidden < 256) ? hidden : 256;
    silu_mul_2input_fp16_kernel<<<m, threads, 0, stream>>>(output, gate, up, m, hidden);
}

template <typename T>
__global__ void assemble_tokens_hidden_kernel(T* __restrict__ out, const T* __restrict__ in,
                                              const size_t* __restrict__ idx,
                                              int n_acc, int hidden) {
    int i = blockIdx.x;
    if (i >= n_acc) return;
    size_t in_off = idx[i] * (size_t)hidden;
    size_t out_off = (size_t)i * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        out[out_off + j] = in[in_off + j];
    }
}

void assemble_tokens_hidden_bf16(__nv_bfloat16* out, const __nv_bfloat16* in,
                                 const size_t* idx, int n_acc, int hidden, cudaStream_t s) {
    if (n_acc <= 0 || hidden <= 0) return;
    int t = (hidden < 256) ? hidden : 256;
    assemble_tokens_hidden_kernel<__nv_bfloat16><<<n_acc, t, 0, s>>>(out, in, idx, n_acc, hidden);
}
void assemble_tokens_hidden_fp16(__half* out, const __half* in,
                                 const size_t* idx, int n_acc, int hidden, cudaStream_t s) {
    if (n_acc <= 0 || hidden <= 0) return;
    int t = (hidden < 256) ? hidden : 256;
    assemble_tokens_hidden_kernel<__half><<<n_acc, t, 0, s>>>(out, in, idx, n_acc, hidden);
}
void assemble_tokens_hidden_fp32(float* out, const float* in,
                                 const size_t* idx, int n_acc, int hidden, cudaStream_t s) {
    if (n_acc <= 0 || hidden <= 0) return;
    int t = (hidden < 256) ? hidden : 256;
    assemble_tokens_hidden_kernel<float><<<n_acc, t, 0, s>>>(out, in, idx, n_acc, hidden);
}

__global__ void embedding_bf16_kernel(__nv_bfloat16* output, const __nv_bfloat16* table,
                                      const int* ids, int hidden_size) {
    int token = blockIdx.x;
    int id = ids[token];
    const __nv_bfloat16* src = table + id * hidden_size;
    __nv_bfloat16* dst = output + token * hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        dst[i] = src[i];
}

void embedding_lookup_bf16(__nv_bfloat16* output, const __nv_bfloat16* table,
                           const int* ids, int num_tokens, int hidden_size, cudaStream_t stream) {
    int threads = std::min(256, hidden_size);
    embedding_bf16_kernel<<<num_tokens, threads, 0, stream>>>(output, table, ids, hidden_size);
}

__global__ void residual_add_bf16_kernel(__nv_bfloat16* out, const __nv_bfloat16* a,
                                          const __nv_bfloat16* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

void residual_add_bf16(__nv_bfloat16* output, const __nv_bfloat16* a, const __nv_bfloat16* b,
                       int count, cudaStream_t stream) {
    int threads = 256, blocks = (count + threads - 1) / threads;
    residual_add_bf16_kernel<<<blocks, threads, 0, stream>>>(output, a, b, count);
}

__global__ void fused_residual_rmsnorm_bf16_kernel(
    __nv_bfloat16* normed, __nv_bfloat16* residual,
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    int hidden_size, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    __nv_bfloat16* res = residual + row * hidden_size;
    const __nv_bfloat16* inp = input + row * hidden_size;
    __nv_bfloat16* out = normed + row * hidden_size;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float r = __bfloat162float(res[i]);
        float x = __bfloat162float(inp[i]);
        float val = r + x;
        res[i] = __float2bfloat16(val);
        sum_sq += val * val;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    if (lane_id == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x + 31) / 32) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    if (threadIdx.x == 0) smem[0] = rsqrtf(sum_sq / hidden_size + eps);
    __syncthreads();
    float inv_rms = smem[0];
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(res[i]) * inv_rms;
        float w = __bfloat162float(weight[i]);
        out[i] = __float2bfloat16(val * w);
    }
}

void fused_residual_rmsnorm_bf16(__nv_bfloat16* normed, __nv_bfloat16* residual,
                                  const __nv_bfloat16* input, const __nv_bfloat16* weight,
                                  int seq_len, int hidden_size, float eps, cudaStream_t stream) {
    int threads = std::min(1024, (hidden_size + 31) / 32 * 32);
    int smem = (threads / 32 + 1) * sizeof(float);
    fused_residual_rmsnorm_bf16_kernel<<<seq_len, threads, smem, stream>>>(normed, residual, input, weight, hidden_size, eps);
}

}  // namespace ksana_llm

// ============================================================
// ArgMax via ixinfer cuinferTopKBatchV2
// ============================================================
namespace ksana_llm {

struct ArgMaxWorkspace {
  void* ptr = nullptr;
  size_t size = 0;
  void ensure(size_t need) {
    if (need <= size) return;
    if (ptr) cudaFree(ptr);
    cudaMalloc(&ptr, need);
    size = need;
  }
};

static ArgMaxWorkspace& GetArgMaxWorkspace() {
  static ArgMaxWorkspace ws;
  return ws;
}

static void argmax_via_topk(const void* logits, int batch_size, int vocab_size,
                            unsigned int* output, cudaStream_t stream,
                            cuinferDataType_t dtype) {
  size_t ws_size = 0;
  cuinferGetTopKBatchV2Workspace(
      1, batch_size, vocab_size, 1, 1, true,
      CUINFER_TOPK_RESULT_ORDER_ARBITARY, 0,
      false, true, dtype, CUINFER_TOPK_STRATEGY_FASTER, &ws_size);
  GetArgMaxWorkspace().ensure(ws_size);
  cuinferHandle_t handle = GetCuinferHandle();
  cuinferSetStream(handle, stream);
  cuinferTopKBatchV2(
      handle, logits, 1, batch_size, vocab_size, 1, 1, true,
      CUINFER_TOPK_RESULT_ORDER_ARBITARY, 0,
      nullptr, reinterpret_cast<int*>(output),
      dtype, CUINFER_TOPK_STRATEGY_FASTER, GetArgMaxWorkspace().ptr);
}

void argmax_gpu_fp32(const float* logits, int batch_size, int vocab_size, unsigned int* output, cudaStream_t stream) {
  argmax_via_topk(logits, batch_size, vocab_size, output, stream, CUINFER_DATA_FLOAT);
}
void argmax_gpu_fp16(const __half* logits, int batch_size, int vocab_size, unsigned int* output, cudaStream_t stream) {
  argmax_via_topk(logits, batch_size, vocab_size, output, stream, CUINFER_DATA_HALF);
}
void argmax_gpu_bf16(const __nv_bfloat16* logits, int batch_size, int vocab_size, unsigned int* output, cudaStream_t stream) {
  argmax_via_topk(logits, batch_size, vocab_size, output, stream, CUINFER_DATA_BFLOAT16);
}
void argmax_gpu(const void* logits, int batch_size, int vocab_size, unsigned int* output, cudaStream_t stream) {
  argmax_gpu_fp32((const float*)logits, batch_size, vocab_size, output, stream);
}

}  // namespace ksana_llm

// ============================================================
// BF16 prefill attention (cuinfer FMHA + naive fallback)
// ============================================================
namespace ksana_llm {

__global__ void gqa_prefill_bf16_k(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ qkv,
    int total_tokens, int nq, int nkv, int hd, int row_stride, float sc) {
  const int qi = blockIdx.x;
  const int hi = blockIdx.y;
  if (qi >= total_tokens || hi >= nq) return;
  const int ki = hi / (nq / nkv);
  const int kv_off = nq * hd;
  const int v_off  = (nq + nkv) * hd;
  if (threadIdx.x != 0) return;
  extern __shared__ float scores[];
  float m = -1e30f;
  for (int t = 0; t <= qi; ++t) {
    float s = 0.0f;
    for (int d = 0; d < hd; ++d) {
      s += __bfloat162float(qkv[qi * row_stride + hi * hd + d]) *
           __bfloat162float(qkv[t * row_stride + kv_off + ki * hd + d]);
    }
    s *= sc;
    scores[t] = s;
    if (s > m) m = s;
  }
  float denom = 0.0f;
  for (int t = 0; t <= qi; ++t) {
    scores[t] = expf(scores[t] - m);
    denom += scores[t];
  }
  float inv_denom = 1.0f / (denom + 1e-8f);
  for (int d = 0; d < hd; ++d) {
    float v = 0.0f;
    for (int t = 0; t <= qi; ++t) {
      v += scores[t] * inv_denom *
           __bfloat162float(qkv[t * row_stride + v_off + ki * hd + d]);
    }
    out[qi * nq * hd + hi * hd + d] = __float2bfloat16(v);
  }
}

void gqa_prefill_bf16(__nv_bfloat16* out, const __nv_bfloat16* qkv,
                      int total_tokens, int num_q_heads, int num_kv_heads,
                      int head_dim, float scale, cudaStream_t stream) {
  if (total_tokens <= 0 || num_q_heads <= 0) return;

  if (head_dim % 32 != 0 || head_dim > 512 || num_q_heads % num_kv_heads != 0) {
    int row_stride = (num_q_heads + 2 * num_kv_heads) * head_dim;
    size_t smem_bytes = total_tokens * sizeof(float);
    dim3 grid(total_tokens, num_q_heads);
    gqa_prefill_bf16_k<<<grid, 32, smem_bytes, stream>>>(
        out, qkv, total_tokens, num_q_heads, num_kv_heads, head_dim, row_stride, scale);
    return;
  }

  size_t q_bytes  = size_t(total_tokens) * num_q_heads  * head_dim * sizeof(__nv_bfloat16);
  size_t kv_bytes = size_t(total_tokens) * num_kv_heads * head_dim * sizeof(__nv_bfloat16);
  __nv_bfloat16 *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
  int *d_cu_seq = nullptr;
  cudaMallocAsync(&d_q, q_bytes,  stream);
  cudaMallocAsync(&d_k, kv_bytes, stream);
  cudaMallocAsync(&d_v, kv_bytes, stream);
  cudaMallocAsync(&d_cu_seq, 2 * sizeof(int), stream);

  int h_cu_seq[2] = {0, total_tokens};
  cudaMemcpyAsync(d_cu_seq, h_cu_seq, 2 * sizeof(int), cudaMemcpyHostToDevice, stream);

  split_qkv_bf16_kernel<<<total_tokens, 256, 0, stream>>>(
      qkv, d_q, d_k, d_v, total_tokens, num_q_heads, num_kv_heads, head_dim);

  cuinferHandle_t handle = GetCuinferHandle();
  cuinferSetStream(handle, stream);

  cuinferFlashAttnConfigInfo info;
  info.layout       = CUINFER_FATTN_BSHD;
  info.isCausal     = true;
  info.scaling      = scale;
  info.qoSeqArray   = d_cu_seq;
  info.kvSeqArray   = d_cu_seq;
  info.kvHeadNum    = num_kv_heads;
  info.kvSeqStart   = 0;
  info.kvSeqEnd     = 0;
  info.isAlibi      = false;
  info.alibiMode    = CUINFER_FATTN_ALIBI_MODE_NONE;
  info.slopeM       = nullptr;
  info.qStride      = num_q_heads  * head_dim;
  info.kStride      = num_kv_heads * head_dim;
  info.vStride      = num_kv_heads * head_dim;
  info.isPersistent = false;

  cuinferTensorDescriptor_t qDesc, kDesc, vDesc, maskDesc, oDesc;
  cuinferCreateTensorDescriptor(&qDesc);
  cuinferCreateTensorDescriptor(&kDesc);
  cuinferCreateTensorDescriptor(&vDesc);
  cuinferCreateTensorDescriptor(&maskDesc);
  cuinferCreateTensorDescriptor(&oDesc);
  cuinferSetTensor4dDescriptor(qDesc, CUINFER_TENSOR_NCHW, CUINFER_DATA_BFLOAT16,
                               1, total_tokens, num_q_heads,  head_dim);
  cuinferSetTensor4dDescriptor(kDesc, CUINFER_TENSOR_NCHW, CUINFER_DATA_BFLOAT16,
                               1, total_tokens, num_kv_heads, head_dim);
  cuinferSetTensor4dDescriptor(vDesc, CUINFER_TENSOR_NCHW, CUINFER_DATA_BFLOAT16,
                               1, total_tokens, num_kv_heads, head_dim);
  cuinferSetTensor4dDescriptor(oDesc, CUINFER_TENSOR_NCHW, CUINFER_DATA_BFLOAT16,
                               1, total_tokens, num_q_heads,  head_dim);

  cuinferStatus_t st = cuinferFMHAForwardEx(
      handle, info, qDesc, d_q, kDesc, d_k, vDesc, d_v,
      maskDesc, nullptr, oDesc, out);
  if (st != CUINFER_STATUS_SUCCESS) {
    fprintf(stderr, "[ksana-iluvatar] cuinferFMHAForwardEx failed: %s "
                    "(T=%d, nq=%d, nkv=%d, hd=%d) — falling back to naive.\n",
            cuinferGetErrorString(st), total_tokens, num_q_heads, num_kv_heads, head_dim);
    int row_stride = (num_q_heads + 2 * num_kv_heads) * head_dim;
    size_t smem_bytes = total_tokens * sizeof(float);
    dim3 grid(total_tokens, num_q_heads);
    gqa_prefill_bf16_k<<<grid, 32, smem_bytes, stream>>>(
        out, qkv, total_tokens, num_q_heads, num_kv_heads, head_dim, row_stride, scale);
  }

  cuinferDestroyTensorDescriptor(qDesc);
  cuinferDestroyTensorDescriptor(kDesc);
  cuinferDestroyTensorDescriptor(vDesc);
  cuinferDestroyTensorDescriptor(maskDesc);
  cuinferDestroyTensorDescriptor(oDesc);

  cudaFreeAsync(d_q, stream);
  cudaFreeAsync(d_k, stream);
  cudaFreeAsync(d_v, stream);
  cudaFreeAsync(d_cu_seq, stream);
}

}  // namespace ksana_llm
