/* Iluvatar runtime stubs for KsanaLLM.
 * Provides stub/real implementations for functions normally in 3rdparty/LLM_kernels
 * and kernels/nvidia/. Organized as clean single-pass namespace blocks.
 * ===========================================================================*/

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <type_traits>
#include <optional>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <curand_kernel.h>
#include <torch/torch.h>

#include "ksana_llm/layers/iluvatar/iluvatar_type_fwd.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/device_utils.h"

// Layer headers for vtable stubs
#include "ksana_llm/layers/batched_matmul_layer.h"
#include "ksana_llm/layers/moe_layer.h"
#include "ksana_llm/layers/all_reduce_residual_add_norm_layer.h"
#include "ksana_llm/layers/custom_all_reduce_sum_layer.h"
#include "ksana_llm/layers/flash_mla_attention_layer.h"
#include "ksana_llm/layers/paged_mla_attention_layer.h"
#include "ksana_llm/layers/flash_sparse_mla_attention_layer.h"
#include "ksana_llm/layers/paged_sparse_mla_attention_layer.h"
#include "ksana_llm/layers/flash_sparse_mla_indexer_layer.h"
#include "ksana_llm/layers/paged_sparse_mla_indexer_layer.h"

// 依赖真实 iluvatar decode kernel（下面的 InvokePagedAttention<__half,__half,kAuto> 走这条路径）。
#include "ksana_llm/kernels/iluvatar/basic_kernel_wrapper.h"
// libllm_kernels_iluvatar_kernel_paged_attention.a 提供的 vLLM-style paged decode：
#include "csrc/kernels/iluvatar/paged_attention/paged_attention.h"
#include "csrc/kernels/iluvatar/paged_attention/cache_copy.h"
#include "ksana_llm/utils/logger.h"
// ===================================================================
// Section 1: llm_kernels::utils stubs
// ===================================================================
namespace llm_kernels { namespace utils {
void ParseNpyIntro(FILE*&, unsigned int&, unsigned int&) {}
void ParseNpyHeader(FILE*&, unsigned int, std::vector<unsigned long>&) {}
unsigned int GetNvLinkVersion(unsigned int, unsigned int) { return 0; }
std::string fmtstr(const char* fmt, ...) { return fmt ? fmt : ""; }
}}

// ===================================================================
// Section 2: llm_kernels::nvidia stubs (cast, alibi, argmax, identity, etc.)
// ===================================================================
namespace llm_kernels { namespace nvidia {

void FloatToHalf(const float*, unsigned long, __half*, cudaStream_t) {}
void HalfToFloat(const __half*, unsigned long, float*, cudaStream_t, unsigned long, unsigned long) {}
void BFloat16ToHalf(void*, unsigned long, cudaStream_t) {}
void HalfToBFloat16(void*, unsigned long, cudaStream_t) {}
void BFloat16ToFloat(const __nv_bfloat16*, unsigned long, float*, cudaStream_t, unsigned long, unsigned long) {}
void FloatToBFloat16(const float*, unsigned long, __nv_bfloat16*, cudaStream_t) {}

void GetAlibiSlopesCuda(float*, int, cudaStream_t&) {}
void InvokeWarpArgMaxReduce(const float*, int, int, unsigned int*, cudaStream_t) {}

template <typename T> void InvokeLocalArgMaxReduce(const T*, int, int, int, int, float*, cudaStream_t) {}
template void InvokeLocalArgMaxReduce<__half>(const __half*, int, int, int, int, float*, cudaStream_t);
template void InvokeLocalArgMaxReduce<__nv_bfloat16>(const __nv_bfloat16*, int, int, int, int, float*, cudaStream_t);
template void InvokeLocalArgMaxReduce<float>(const float*, int, int, int, int, float*, cudaStream_t);

template <typename T> void InitIdentityMatrixAdaptive(T*, unsigned long, unsigned long, cudaStream_t) {}
template void InitIdentityMatrixAdaptive<__half>(__half*, unsigned long, unsigned long, cudaStream_t);
template void InitIdentityMatrixAdaptive<__nv_bfloat16>(__nv_bfloat16*, unsigned long, unsigned long, cudaStream_t);
template void InitIdentityMatrixAdaptive<float>(float*, unsigned long, unsigned long, cudaStream_t);

template <typename T> void InvokeCopyElements(T**, T*, unsigned long, cudaStream_t&) {}
template void InvokeCopyElements<float>(float**, float*, unsigned long, cudaStream_t&);

int get_permutation_map(int) { return 0; }
struct FlashMlaWorkspaceMap {};
int GetNumSmParts(FlashMlaWorkspaceMap&, int, int, int) { return 0; }
void InvokeGetMlaMetadata(int*, FlashMlaWorkspaceMap&, int, cudaStream_t) {}
void GetAttnImplMeta(int, int, int, bool, bool) {}
void InvokeGetSparseMlaMetadata(int*, int, int, int, int, bool, int, cudaStream_t, int*, int*) {}

}}  // namespace llm_kernels::nvidia

// ===================================================================
// Section 3: tensorrt_llm::kernels stubs (sampling)
// ===================================================================
namespace tensorrt_llm { namespace kernels {
struct FinishedState { static constexpr int empty() { return 0; } };

template <typename T>
void invoke_batch_topk_sampling(void*, size_t&, const T*, int**, int*,
    const FinishedState*, FinishedState*, float*, float*,
    curandState_t*, int, const int*, float, const float*,
    int, const int*, const int*, cudaStream_t, int, int,
    const bool*, bool, bool) {}
template void invoke_batch_topk_sampling<float>(void*, size_t&, const float*, int**, int*,
    const FinishedState*, FinishedState*, float*, float*,
    curandState_t*, int, const int*, float, const float*,
    int, const int*, const int*, cudaStream_t, int, int,
    const bool*, bool, bool);

void InvokeCurandBatchInitialize(curandState_t*, const int*, size_t, const uint64_t*, cudaStream_t) {}

template <typename T>
void InvokeAddBiasSoftMax(T*, T**, T*, const T*, const int*,
    const FinishedState*, const int*, int, int, int, int, int, bool, bool, cudaStream_t) {}
template void InvokeAddBiasSoftMax<float>(float*, float**, float*, const float*, const int*,
    const FinishedState*, const int*, int, int, int, int, int, bool, bool, cudaStream_t);
}}  // namespace tensorrt_llm::kernels

// ===================================================================
// Section 4: ksana_llm — ALL kernel functions in ONE namespace block
// ===================================================================
namespace ksana_llm {

// --- Forward declarations for cuda_kernels.cu functions ---
void rms_norm(__half*, const __half*, const __half*, int, int, float, cudaStream_t);
void silu_mul(__half*, const __half*, const __half*, int, cudaStream_t);
void embedding_lookup(__half*, const __half*, const int*, int, int, cudaStream_t);
void residual_add(__half*, const __half*, const __half*, int, cudaStream_t);
void fused_residual_rmsnorm(__half*, __half*, const __half*, const __half*, int, int, float, cudaStream_t);
void gqa_attention(__half*, const __half*, const __half*, const __half*, float*, int, int, int, int, int, float, cudaStream_t);
void argmax_gpu(const void*, int, int, unsigned int*, cudaStream_t);
// bf16 variants
void rms_norm_bf16(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, int, float, cudaStream_t);
void silu_mul_bf16(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, cudaStream_t);
// 行内 row-based gated activation: input shape=[m, 2*hidden]，每行前 hidden 个是 val，后 hidden 是 gate
// 输出: out[i, j] = silu(input[i, j]) * input[i, hidden + j]
void silu_mul_rowbased_bf16(__nv_bfloat16*, const __nv_bfloat16*, int, int, cudaStream_t);
void silu_mul_rowbased_fp16(__half*, const __half*, int, int, cudaStream_t);
void embedding_lookup_bf16(__nv_bfloat16*, const __nv_bfloat16*, const int*, int, int, cudaStream_t);
void residual_add_bf16(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, cudaStream_t);
void fused_residual_rmsnorm_bf16(__nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, int, float, cudaStream_t);
void gqa_attention_bf16(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, int, int, int, int, float, cudaStream_t);
// 真实的 BF16 GQA causal prefill（batch=1，输入是 [total_tokens, (nq+2*nkv)*hd] qkv，
// 写出 [total_tokens, nq, hd]）。参见 cuda_kernels.cu 实现。
void gqa_prefill_bf16(__nv_bfloat16* out, const __nv_bfloat16* qkv,
                      int total_tokens, int num_q_heads, int num_kv_heads,
                      int head_dim, float scale, cudaStream_t stream);
// ---- InvokeMatMul: REAL cublas ----
template <typename T>
void InvokeMatMul(cublasHandle_t h, cublasLtHandle_t, int m, int n, int k,
                  const void* a, const void* b, void* c,
                  cudaStream_t& s, void*, cublasLtMatmulAlgo_t*, size_t, bool) {
  cublasSetStream(h, s);
  float alpha = 1.0f, beta = 0.0f;
  cudaDataType_t dt;
  if constexpr (std::is_same_v<T, __nv_bfloat16>) dt = CUDA_R_16BF;
  else if constexpr (std::is_same_v<T, __half>) dt = CUDA_R_16F;
  else dt = CUDA_R_32F;
  cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, dt, n, a, dt, k, &beta, c, dt, n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
}
template void InvokeMatMul<__half>(cublasHandle_t, cublasLtHandle_t, int, int, int, const void*, const void*, void*, cudaStream_t&, void*, cublasLtMatmulAlgo_t*, size_t, bool);
template void InvokeMatMul<__nv_bfloat16>(cublasHandle_t, cublasLtHandle_t, int, int, int, const void*, const void*, void*, cudaStream_t&, void*, cublasLtMatmulAlgo_t*, size_t, bool);
template void InvokeMatMul<float>(cublasHandle_t, cublasLtHandle_t, int, int, int, const void*, const void*, void*, cudaStream_t&, void*, cublasLtMatmulAlgo_t*, size_t, bool);

// ---- InvokeRMSNorm: REAL for half + bf16, fallback for float ----
template <typename T>
void InvokeRMSNorm(void* in, void* w, float eps, int m, int n, void* out, bool, cudaStream_t s) {
  if (m * n > 0) cudaMemcpyAsync(out, in, m * n * sizeof(T), cudaMemcpyDeviceToDevice, s);
}
// InvokeRMSNorm<__half> 在 kernels/iluvatar/basic_kernel_wrapper.cpp 已提供真实实现，避免 multiple definition。
template <> void InvokeRMSNorm<__nv_bfloat16>(void* in, void* w, float eps, int m, int n, void* out, bool, cudaStream_t s) {
  rms_norm_bf16((__nv_bfloat16*)out, (const __nv_bfloat16*)in, (const __nv_bfloat16*)w, m, n, eps, s);
}
template void InvokeRMSNorm<float>(void*, void*, float, int, int, void*, bool, cudaStream_t);

// ---- InvokeLayerNorm: stub ----
template <typename T> void InvokeLayerNorm(const void*, const void*, const void*, float, int, int, void*, cudaStream_t) {}
template void InvokeLayerNorm<__half>(const void*, const void*, const void*, float, int, int, void*, cudaStream_t);
template void InvokeLayerNorm<__nv_bfloat16>(const void*, const void*, const void*, float, int, int, void*, cudaStream_t);
template void InvokeLayerNorm<float>(const void*, const void*, const void*, float, int, int, void*, cudaStream_t);

// LookupEmbedding 已迁移到 kernels/iluvatar/basic_kernel_wrapper.cpp，调用真实 libllm_kernels_iluvatar_kernel_embedding.a。这里不再定义，避免符号冲突。

// ---- InvokeAddBiasResidual: REAL for half + bf16 ----
template <typename T>
void InvokeAddBiasResidual(const void* a, const void* b, const void*, int m, int n, void* out, cudaStream_t s) {
  if (m * n > 0) cudaMemcpyAsync(out, a, m * n * sizeof(T), cudaMemcpyDeviceToDevice, s);
}
// InvokeAddBiasResidual<__half> 在 kernels/iluvatar/basic_kernel_wrapper.cpp 已提供真实实现，避免 multiple definition。
template <> void InvokeAddBiasResidual<__nv_bfloat16>(const void* a, const void* b, const void*, int m, int n, void* out, cudaStream_t s) {
  residual_add_bf16((__nv_bfloat16*)out, (const __nv_bfloat16*)a, (const __nv_bfloat16*)b, m * n, s);
}
template void InvokeAddBiasResidual<float>(const void*, const void*, const void*, int, int, void*, cudaStream_t);

// ---- InvokeFusedAddRmsNorm: REAL for half + bf16 ----
template <typename T>
void InvokeFusedAddRmsNorm(void* res, void* in, void* w, double eps, int m, int n, bool, cudaStream_t s) {
  if (m * n > 0) cudaMemcpyAsync(res, in, m * n * sizeof(T), cudaMemcpyDeviceToDevice, s);
}
// 关键 bug 修复：原本 normed/residual 都传 res，导致 residual 没真正 in-place 累加，
// 而是被 normed 输出覆盖。AddNormLayer 语义是：
//   step 1: residual[i] += input[i]  (in-place 累加到 residual_buffer)
//   step 2: input[i]    = norm(residual) * weight
// 调用方约定：input_tensors[0]=hidden(=normed output), input_tensors[1]=residual。
// 所以 stub 接收的是 res=hidden, in=residual，正确映射应是：
//   normed = hidden (= res)
//   residual = residual_buffer (= in)
//   input  = hidden 的旧值 = res （会被 normed 覆盖，但我们这里读 res 算 sum 后再写回 normed=res）
//
// fused_residual_rmsnorm kernel 实际语义：res[i]+=inp[i]; out[i]=norm(res)*w
// 因此把 residual 参数传 in，input 参数传 res（旧值），output 参数传 res（覆盖）。
// 但因为 kernel 在循环内同时读 res 和 inp，再回写 res — 把 res(新)/inp 调换无问题。
template <> void InvokeFusedAddRmsNorm<__half>(void* res, void* in, void* w, double eps, int m, int n, bool, cudaStream_t s) {
  fused_residual_rmsnorm((__half*)res, (__half*)in, (const __half*)res, (const __half*)w, m, n, (float)eps, s);
}
template <> void InvokeFusedAddRmsNorm<__nv_bfloat16>(void* res, void* in, void* w, double eps, int m, int n, bool, cudaStream_t s) {
  fused_residual_rmsnorm_bf16((__nv_bfloat16*)res, (__nv_bfloat16*)in, (const __nv_bfloat16*)res, (const __nv_bfloat16*)w, m, n, (float)eps, s);
}
template void InvokeFusedAddRmsNorm<float>(void*, void*, void*, double, int, int, bool, cudaStream_t);

// ---- InvokeMulThenAdd: stub ----
template <typename T> void InvokeMulThenAdd(const void*, const void*, T, T, int, int, void*, cudaStream_t) {}
template void InvokeMulThenAdd<__half>(const void*, const void*, __half, __half, int, int, void*, cudaStream_t);
template void InvokeMulThenAdd<__nv_bfloat16>(const void*, const void*, __nv_bfloat16, __nv_bfloat16, int, int, void*, cudaStream_t);
template void InvokeMulThenAdd<float>(const void*, const void*, float, float, int, int, void*, cudaStream_t);

// ---- AssembleTokensHidden: stub ----
// 修复: AssembleTokensHidden 之前是空 stub，导致 lm_head 读到旧 buffer，
// 触发 generate 死循环（ální featured / stands featured / yn pers …）。
// 语义: 从 input [N, hidden] 按 accepted_tokens_idx[i] 取第 i 个 batch 的对应 token
// 写到 output [accepted_tokens_size, hidden]。
// 真实 kernel 在 cuda_kernels.cu 里：assemble_tokens_hidden_bf16 / _fp16 / _fp32。
void assemble_tokens_hidden_bf16(__nv_bfloat16*, const __nv_bfloat16*, const size_t*, int, int, cudaStream_t);
void assemble_tokens_hidden_fp16(__half*, const __half*, const size_t*, int, int, cudaStream_t);
void assemble_tokens_hidden_fp32(float*, const float*, const size_t*, int, int, cudaStream_t);
template <typename T>
void AssembleTokensHidden(const void* input, const void* accepted_tokens_idx,
                          int accepted_tokens_size, int hidden_units_num,
                          void* output, cudaStream_t& stream);
template <> void AssembleTokensHidden<__nv_bfloat16>(const void* input, const void* accepted_tokens_idx,
                                                     int accepted_tokens_size, int hidden_units_num,
                                                     void* output, cudaStream_t& stream) {
  if (accepted_tokens_size <= 0 || hidden_units_num <= 0) return;
  assemble_tokens_hidden_bf16(reinterpret_cast<__nv_bfloat16*>(output),
                              reinterpret_cast<const __nv_bfloat16*>(input),
                              reinterpret_cast<const size_t*>(accepted_tokens_idx),
                              accepted_tokens_size, hidden_units_num, stream);
}
template <> void AssembleTokensHidden<__half>(const void* input, const void* accepted_tokens_idx,
                                              int accepted_tokens_size, int hidden_units_num,
                                              void* output, cudaStream_t& stream) {
  if (accepted_tokens_size <= 0 || hidden_units_num <= 0) return;
  assemble_tokens_hidden_fp16(reinterpret_cast<__half*>(output),
                              reinterpret_cast<const __half*>(input),
                              reinterpret_cast<const size_t*>(accepted_tokens_idx),
                              accepted_tokens_size, hidden_units_num, stream);
}
template <> void AssembleTokensHidden<float>(const void* input, const void* accepted_tokens_idx,
                                             int accepted_tokens_size, int hidden_units_num,
                                             void* output, cudaStream_t& stream) {
  if (accepted_tokens_size <= 0 || hidden_units_num <= 0) return;
  assemble_tokens_hidden_fp32(reinterpret_cast<float*>(output),
                              reinterpret_cast<const float*>(input),
                              reinterpret_cast<const size_t*>(accepted_tokens_idx),
                              accepted_tokens_size, hidden_units_num, stream);
}
template void AssembleTokensHidden<__half>(const void*, const void*, int, int, void*, cudaStream_t&);
template void AssembleTokensHidden<__nv_bfloat16>(const void*, const void*, int, int, void*, cudaStream_t&);
template void AssembleTokensHidden<float>(const void*, const void*, int, int, void*, cudaStream_t&);

// ---- DataToFloat, InvokeSplit, InvokePermute, Concat, CalcLogprobs: stubs ----
template <typename T> void DataToFloat(const void*, int, size_t, size_t, void*, cudaStream_t&) {}
template void DataToFloat<__half>(const void*, int, size_t, size_t, void*, cudaStream_t&);
template void DataToFloat<__nv_bfloat16>(const void*, int, size_t, size_t, void*, cudaStream_t&);
template void DataToFloat<float>(const void*, int, size_t, size_t, void*, cudaStream_t&);

template <typename T> void InvokeSplit(const T*, const std::vector<T*>&, std::vector<int>&, int, int, int, cudaStream_t&) {}
template void InvokeSplit<__half>(const __half*, const std::vector<__half*>&, std::vector<int>&, int, int, int, cudaStream_t&);
template void InvokeSplit<__nv_bfloat16>(const __nv_bfloat16*, const std::vector<__nv_bfloat16*>&, std::vector<int>&, int, int, int, cudaStream_t&);
template void InvokeSplit<float>(const float*, const std::vector<float*>&, std::vector<int>&, int, int, int, cudaStream_t&);

template <typename T> void InvokePermute(void*, void*, std::vector<size_t>, std::vector<size_t>, cudaStream_t&) {}
template void InvokePermute<__half>(void*, void*, std::vector<size_t>, std::vector<size_t>, cudaStream_t&);
template void InvokePermute<__nv_bfloat16>(void*, void*, std::vector<size_t>, std::vector<size_t>, cudaStream_t&);
template void InvokePermute<float>(void*, void*, std::vector<size_t>, std::vector<size_t>, cudaStream_t&);

template <typename T> void Concat(const void*, const void*, size_t, size_t, size_t, size_t, void*, cudaStream_t&) {}
template void Concat<__half>(const void*, const void*, size_t, size_t, size_t, size_t, void*, cudaStream_t&);
template void Concat<__nv_bfloat16>(const void*, const void*, size_t, size_t, size_t, size_t, void*, cudaStream_t&);
template void Concat<float>(const void*, const void*, size_t, size_t, size_t, size_t, void*, cudaStream_t&);

void CalcLogprobs(float*, float*, int, int, int, float*, long*) {}

// ---- ArgMax: REAL via cuda_kernels.cu (type-dispatched per element type) ----
void argmax_gpu_fp32(const float*, int, int, unsigned int*, cudaStream_t);
void argmax_gpu_fp16(const __half*, int, int, unsigned int*, cudaStream_t);
void argmax_gpu_bf16(const __nv_bfloat16*, int, int, unsigned int*, cudaStream_t);

template <typename T>
Status ArgMax(const T* logits, int batch_size, int vocab_size, unsigned int* output, StreamT<DEVICE_TYPE_NVIDIA>& stream, void*);

template <>
Status ArgMax<float>(const float* logits, int batch_size, int vocab_size, unsigned int* output, StreamT<DEVICE_TYPE_NVIDIA>& stream, void*) {
  argmax_gpu_fp32(logits, batch_size, vocab_size, output, stream.Get());
  return Status();
}
template <>
Status ArgMax<__half>(const __half* logits, int batch_size, int vocab_size, unsigned int* output, StreamT<DEVICE_TYPE_NVIDIA>& stream, void*) {
  argmax_gpu_fp16(logits, batch_size, vocab_size, output, stream.Get());
  return Status();
}
template <>
Status ArgMax<__nv_bfloat16>(const __nv_bfloat16* logits, int batch_size, int vocab_size, unsigned int* output, StreamT<DEVICE_TYPE_NVIDIA>& stream, void*) {
  argmax_gpu_bf16(logits, batch_size, vocab_size, output, stream.Get());
  return Status();
}

// ---- Remaining stubs ----
template <typename T> void InvokeExtractSubMatrix(const T*, T*, size_t, size_t, size_t, cudaStream_t&) {}
template void InvokeExtractSubMatrix<__half>(const __half*, __half*, size_t, size_t, size_t, cudaStream_t&);
template void InvokeExtractSubMatrix<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t, size_t, size_t, cudaStream_t&);
template void InvokeExtractSubMatrix<float>(const float*, float*, size_t, size_t, size_t, cudaStream_t&);

template <typename T> void InvokeSigmoidActivation(void*, size_t, float, cudaStream_t&) {}
template void InvokeSigmoidActivation<__half>(void*, size_t, float, cudaStream_t&);
template void InvokeSigmoidActivation<__nv_bfloat16>(void*, size_t, float, cudaStream_t&);
template void InvokeSigmoidActivation<float>(void*, size_t, float, cudaStream_t&);

template <typename T> void InvokeMarlinPermuteScales(cudaStream_t, const void*, void*, size_t, size_t, long) {}
template void InvokeMarlinPermuteScales<__half>(cudaStream_t, const void*, void*, size_t, size_t, long);
template void InvokeMarlinPermuteScales<__nv_bfloat16>(cudaStream_t, const void*, void*, size_t, size_t, long);
template void InvokeMarlinPermuteScales<float>(cudaStream_t, const void*, void*, size_t, size_t, long);

// 关键修复：原本是空 stub，导致 lm_head（m=1 路径）从来没真正做 GEMM，
// 输出全是 garbage，sample 出来的 token 完全无意义（hidden state 完美匹配 HF
// 但 logits[0..7] 全是 ~0 而 argmax 落在 buffer 中残留的某个大值）。
// 这里转发到 cublasGemmStridedBatchedEx，与 nvidia 同语义。
template <typename T>
void InvokeStridedBatchedMatMul(cublasHandle_t cublas_handle, cublasLtHandle_t /*cublaslt_handle*/,
                                cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                const void* a_ptr, int lda, long stride_a,
                                const void* b_ptr, int ldb, long stride_b,
                                void* c_ptr, int ldc, long stride_c,
                                int batch_count, float alpha, float beta) {
  cudaDataType_t dt;
  if constexpr (std::is_same_v<T, __nv_bfloat16>) dt = CUDA_R_16BF;
  else if constexpr (std::is_same_v<T, __half>) dt = CUDA_R_16F;
  else dt = CUDA_R_32F;
  // 与 nvidia 实现一致：交换 (a,b) -> (b,a) 以适配 column-major cuBLAS。
  // ksana 调用方按 row-major 思维：A[m,k] * B[k,n] = C[m,n]，CUBLAS_OP_N 表示不转置。
  // 在 column-major 下把 C 看成 [n,m]，即 B^T * A^T = (A*B)^T，所以传给 cublas 的
  // first matrix 是 B（lda 路径），second 是 A。维度参数也交换 (m<->n)。
  cublasGemmStridedBatchedEx(cublas_handle, transb, transa, n, m, k, &alpha,
                             b_ptr, dt, ldb, stride_b,
                             a_ptr, dt, lda, stride_a,
                             &beta, c_ptr, dt, ldc, stride_c,
                             batch_count, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
}
template void InvokeStridedBatchedMatMul<__half>(cublasHandle_t, cublasLtHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, int, long, const void*, int, long, void*, int, long, int, float, float);
template void InvokeStridedBatchedMatMul<__nv_bfloat16>(cublasHandle_t, cublasLtHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, int, long, const void*, int, long, void*, int, long, int, float, float);
template void InvokeStridedBatchedMatMul<float>(cublasHandle_t, cublasLtHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, int, long, const void*, int, long, void*, int, long, int, float, float);

template <typename T> void InvokeApplyTokenBitmaskInplace(void*, const void*, const void*, int, int, int, int, cudaStream_t) {}
template void InvokeApplyTokenBitmaskInplace<float>(void*, const void*, const void*, int, int, int, int, cudaStream_t);

template <typename T> void InvokeMul(void*, void*, void*, int, int, int, int, int) {}
template void InvokeMul<__half>(void*, void*, void*, int, int, int, int, int);
template void InvokeMul<__nv_bfloat16>(void*, void*, void*, int, int, int, int, int);
template void InvokeMul<float>(void*, void*, void*, int, int, int, int, int);

void InvokeProcessKvList(void** kv_list, size_t layer_num, size_t block_num, size_t block_size, cudaStream_t stream) {
  // 关键修复：原本是空 stub，导致除 layer 0 之外所有层的 K/V 指针都是 0/未初始化，
  // 这是 BF16 decode 走 CachePosCopy + 我们自写的 gqa_paged_decode_bf16 之后输出 948,6552
  // 死循环的根本原因。Iluvatar 的 ProcessKvList 实现来自
  // 3rdparty/LLM_kernels/csrc/kernels/iluvatar/paged_attention/cache_copy.cu，
  // 其语义与 nvidia 同名 kernel 一致：
  //   for layer_i in [1, layer_num): kv_list[layer_i*block_num + b] =
  //     (char*)kv_list[b] + layer_i * (block_size / layer_num)
  // 这里 block_size 是单个 KV cache block（K+V 全 layer 总字节数），
  // block_num = total_block_num * 2（K 段 + V 段），
  // 因此 block_size / layer_num 正好是单层 K+V 字节数。
  if (layer_num <= 1 || block_num == 0) return;
  llm_kernels::iluvatar::ProcessKvList(kv_list, layer_num, block_num, block_size, stream);
}
void InvokeSetTorchStream(cudaStream_t&, int) {}
void InvokeMarlinAwqRepack(const void*, void*, long, long, long, int, cudaStream_t) {}
void InvokeMarlinGptqRepack(const void*, const void*, void*, long, long, long, long, bool, int, cudaStream_t) {}
void InvokeCalculateChecksum(void**, size_t*, int, size_t, cudaStream_t) {}
void InvokeMul(float*, float*, float*, int, int) {}
void CalcInputLogprobs(float*, float*, int, int, std::vector<std::vector<std::pair<int, float>>>&, int) {}

template <typename T> c10::ScalarType GetTorchDataType();
template <> c10::ScalarType GetTorchDataType<__half>() { return c10::ScalarType::Half; }
template <> c10::ScalarType GetTorchDataType<__nv_bfloat16>() { return c10::ScalarType::BFloat16; }
template <> c10::ScalarType GetTorchDataType<float>() { return c10::ScalarType::Float; }

// ---- Activation: REAL SiluMul for half + bf16, stubs for others ----
extern void silu_mul_2input_bf16(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, int, cudaStream_t);
extern void silu_mul_2input_fp16(__half*, const __half*, const __half*, int, int, cudaStream_t);

// 2-input gated activation 接口顺序（与 nvidia kernel_wrapper.cpp 保持一致）：
//   InvokeGatedActivation(gate, gate_bias, up, up_bias, m, n, out, stream)
// 之前是 NO-OP stub，导致 TinyLlama-1.1B 这类没 fuse gate_up_proj 的模型 MLP 输出
// 全是缓冲区残留 → 整个生成循环 (>= rel >= rel ...).
template <template<typename> class Activation, typename T>
void InvokeGatedActivation(const void* gate, const void* gate_bias, const void* up, const void* up_bias,
                           int m, int n, void* out, cudaStream_t s) {}

template <> void InvokeGatedActivation<llm_kernels::nvidia::SiluActivation, __nv_bfloat16>(
    const void* gate, const void* /*gate_bias*/, const void* up, const void* /*up_bias*/,
    int m, int n, void* out, cudaStream_t s) {
  silu_mul_2input_bf16(reinterpret_cast<__nv_bfloat16*>(out),
                       reinterpret_cast<const __nv_bfloat16*>(gate),
                       reinterpret_cast<const __nv_bfloat16*>(up),
                       m, n, s);
}

template <> void InvokeGatedActivation<llm_kernels::nvidia::SiluActivation, __half>(
    const void* gate, const void* /*gate_bias*/, const void* up, const void* /*up_bias*/,
    int m, int n, void* out, cudaStream_t s) {
  silu_mul_2input_fp16(reinterpret_cast<__half*>(out),
                       reinterpret_cast<const __half*>(gate),
                       reinterpret_cast<const __half*>(up),
                       m, n, s);
}

template <template<typename> class Activation, typename T>
void InvokeRowBasedGatedActivation(const void* input, int m, int n, void* output, cudaStream_t s) {
  if (m * n > 0) cudaMemsetAsync(output, 0, m * n * sizeof(T), s);
}

// 注意：nvidia 端的 layout —— input shape=[m, n]（n 是 fused 后的 2*hidden），
// 每行前 n/2 是 val，后 n/2 是 gate；输出 shape=[m, n/2] = silu(val) * gate。
// 之前的实现假设 input 是 [2, m, hidden]（gate 全部在前 m*hidden、up 全部在后 m*hidden），
// 完全错位，导致 MLP 输出乱码 → 整个生成死循环。
template <> void InvokeRowBasedGatedActivation<llm_kernels::nvidia::SiluActivation, __half>(
    const void* input, int m, int n, void* output, cudaStream_t s) {
  const int hidden = n / 2;
  silu_mul_rowbased_fp16(reinterpret_cast<__half*>(output),
                         reinterpret_cast<const __half*>(input),
                         m, hidden, s);
}
template <> void InvokeRowBasedGatedActivation<llm_kernels::nvidia::SiluActivation, __nv_bfloat16>(
    const void* input, int m, int n, void* output, cudaStream_t s) {
  const int hidden = n / 2;
  silu_mul_rowbased_bf16(reinterpret_cast<__nv_bfloat16*>(output),
                         reinterpret_cast<const __nv_bfloat16*>(input),
                         m, hidden, s);
}

#define INST_ACT(Act, T) \
  template void InvokeGatedActivation<Act, T>(const void*, const void*, const void*, const void*, int, int, void*, cudaStream_t); \
  template void InvokeRowBasedGatedActivation<Act, T>(const void*, int, int, void*, cudaStream_t);
// SiluActivation Half/BF16: 已用模板特化提供真实实现，这里不再 explicit instantiate
// (避免与特化冲突)。InvokeRowBasedGatedActivation 已特化 Half/BF16 (Silu)。
INST_ACT(llm_kernels::nvidia::SiluActivation, float)
INST_ACT(llm_kernels::nvidia::GeluActivation, __half)
INST_ACT(llm_kernels::nvidia::GeluActivation, __nv_bfloat16)
INST_ACT(llm_kernels::nvidia::GeluActivation, float)
INST_ACT(llm_kernels::nvidia::ReluActivation, __half)
INST_ACT(llm_kernels::nvidia::ReluActivation, __nv_bfloat16)
INST_ACT(llm_kernels::nvidia::ReluActivation, float)
#undef INST_ACT

// ---- AttenVarlen: prefill-only path（KsanaLLM 的 FlashAttentionLayer 入口）.
// 参数顺序与 src/ksana_llm/kernels/nvidia/attention_kernel_wrapper.cpp 完全一致；
// 这里只用我们 BF16/FP16 prefill 实际需要的：qkv/out/seqlen/ropt/total_tokens/batch/
// num_heads/num_kv_heads/head_size/stride_size/k_scale/v_scale/block_size/k_list/v_list/
// prefix_offsets/block_offsets/use_cache/stream/no_rope。
//
// Prefill 流程（与 nvidia 同分支：!enable_blocked_multi_token_forwarding_kv &&
// !use_flashinfer_for_decode）:
//   1) 可选 RoPE on Q/K（直接在 qkv_ptr 上原地改）
//   2) 把 K/V 用 vLLM-v1 layout 写到 paged cache（CacheCopy<...>）
//   3) BF16 causal GQA 直算（gqa_prefill_bf16；FP16 走老 gqa_attention）
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void AttenVarlen(void* qkv_ptr, void* rotary_embedding_pos, void* rotary_embedding_mask, void* out,
    void* seqlen,
    std::optional<llm_kernels::iluvatar::RotaryEmbeddingCuda>& rotary_embedding_cuda,
    int total_tokens, int /*max_tokens*/, int batch, int num_heads, int num_kv_heads,
    int head_size, int stride_size, float k_scale, float v_scale, size_t /*tp*/, bool /*is_causal*/, int /*rank*/, int block_size,
    void** k_list, void** v_list, void* prefix_offsets, void* block_offsets, const std::optional<void*>& /*alibi*/, int /*layer_index*/,
    void*, void*, void*, void*, void*, void*, void* flexible_offset_uint64_ptr, int /*flex_len*/, float /*ln_eps*/, bool /*use_qk_norm*/, void*, void*,
    bool use_cache, cudaStream_t stream, void* k_cache_ptr, void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq, size_t* /*without_prefix_offsets*/, int /*max_forwarding_tokens*/, bool /*qk_pre_norm*/, bool no_rope, bool /*attn_temperature_tuning*/, float /*attn_scale*/, size_t /*floor_scale*/, bool enable_blocked_multi_token_forwarding_kv, bool /*use_flashinfer_for_decode*/) {
  if (total_tokens <= 0 || num_heads <= 0 || head_size <= 0) return;

  // ---- (1) RoPE: 与 decode 路径同款，需要传 q_ptr / k_ptr 偏移到 qkv 起点的非连续位置 ----
  // qkv 在内存上是 [total_tokens, (nq+2*nkv)*hd]，row stride = (nq+2*nkv)*hd
  // RotaryEmbeddingCuda::SetInput 把 q/k 起始指针 + total_tokens 喂进去，
  // kernel 内部按非连续 stride（其实它要求 stride_size = (nq+2*nkv)*hd，已通过 InitRopeCuda 注入）来跑。
  // RoPE: 之前 InvokeProcessKvList 是空 stub，导致 layer>0 的 KV 全错，
  // 当时只能临时关 RoPE 缩小变量空间。现在 ProcessKvList 已修复，重新打开 RoPE。
  if (!no_rope && rotary_embedding_cuda.has_value()) {
    void* q_base = qkv_ptr;
    void* k_base = static_cast<char*>(qkv_ptr) + (size_t)num_heads * head_size * sizeof(SCALAR_T);
    rotary_embedding_cuda->SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos),
                                    reinterpret_cast<int64_t*>(rotary_embedding_mask),
                                    q_base, k_base, total_tokens, stream);
    rotary_embedding_cuda->Forward<SCALAR_T>();
  }

  // ---- (2) 写 K/V 到 paged cache（CacheCopy 与 nvidia 同款 vLLM-v1 layout） ----
  // K/V 起点同样是 qkv_ptr 内的非连续段；CacheCopy 内部按 stride_size 寻址行。
  if (use_cache) {
    SCALAR_T* k_src = reinterpret_cast<SCALAR_T*>(
        static_cast<char*>(qkv_ptr) + (size_t)num_heads * head_size * sizeof(SCALAR_T));
    SCALAR_T* v_src = reinterpret_cast<SCALAR_T*>(
        static_cast<char*>(qkv_ptr) + (size_t)(num_heads + num_kv_heads) * head_size * sizeof(SCALAR_T));
    if constexpr (std::is_same_v<SCALAR_T, __nv_bfloat16> &&
                  KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
      llm_kernels::iluvatar::CacheCopy<__nv_bfloat16, __nv_bfloat16,
                                     llm_kernels::utils::KVCacheType::kAuto>(
          k_src, v_src, k_list, v_list,
          reinterpret_cast<size_t*>(seqlen),
          reinterpret_cast<size_t*>(flexible_offset_uint64_ptr),
          reinterpret_cast<int*>(block_offsets),
          block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
          k_scale, v_scale, stream);
    } else if constexpr (std::is_same_v<SCALAR_T, __half> &&
                         KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
      // FP16 cache copy 用 uint16_t bit-pattern 路径（与 PagedAttentionOp 同步）
      llm_kernels::iluvatar::CacheCopy<uint16_t, uint16_t,
                                     llm_kernels::utils::KVCacheType::kAuto>(
          reinterpret_cast<uint16_t*>(k_src), reinterpret_cast<uint16_t*>(v_src),
          k_list, v_list,
          reinterpret_cast<size_t*>(seqlen),
          reinterpret_cast<size_t*>(flexible_offset_uint64_ptr),
          reinterpret_cast<int*>(block_offsets),
          block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
          k_scale, v_scale, stream);
    }
    // 其他 SCALAR_T / KV_DTYPE 组合（fp8/float）TinyLlama 不走，跳过。
  }

  // ---- (3) prefill attention ----
  float scale = 1.0f / sqrtf((float)head_size);
  if constexpr (std::is_same_v<SCALAR_T, __nv_bfloat16>) {
    gqa_prefill_bf16(reinterpret_cast<__nv_bfloat16*>(out),
                     reinterpret_cast<const __nv_bfloat16*>(qkv_ptr),
                     total_tokens, num_heads, num_kv_heads, head_size, scale, stream);
  } else if constexpr (std::is_same_v<SCALAR_T, __half>) {
    // 旧 gqa_attention 假设 K/V 已 contiguous，先用兜底（后续再补 fp16 prefill）
    const __half* qkv = (const __half*)qkv_ptr;
    gqa_attention((__half*)out, qkv, qkv + num_heads * head_size,
        qkv + (num_heads + num_kv_heads) * head_size,
        nullptr, 1, 0, num_heads, num_kv_heads, head_size, scale, stream);
  } else {
    cudaMemsetAsync(out, 0, (size_t)total_tokens * num_heads * head_size * sizeof(SCALAR_T), stream);
  }
}

#define INST_AV(S, C, KV) \
  template void AttenVarlen<S, C, KV>(void*, void*, void*, void*, void*, \
    std::optional<llm_kernels::iluvatar::RotaryEmbeddingCuda>&, \
    int, int, int, int, int, int, int, float, float, size_t, bool, int, int, \
    void**, void**, void*, void*, const std::optional<void*>&, int, \
    void*, void*, void*, void*, void*, void*, void*, int, float, bool, void*, void*, \
    bool, cudaStream_t, void*, void*, int32_t*, int64_t, int, size_t*, int, bool, bool, bool, float, size_t, bool, bool);
INST_AV(__half, __half, llm_kernels::utils::KVCacheType::kAuto)
INST_AV(__half, unsigned char, llm_kernels::utils::KVCacheType::kFp8E4M3)
INST_AV(__half, unsigned char, llm_kernels::utils::KVCacheType::kFp8E5M2)
INST_AV(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto)
INST_AV(__nv_bfloat16, unsigned char, llm_kernels::utils::KVCacheType::kFp8E4M3)
INST_AV(__nv_bfloat16, unsigned char, llm_kernels::utils::KVCacheType::kFp8E5M2)
INST_AV(float, float, llm_kernels::utils::KVCacheType::kAuto)
INST_AV(float, unsigned char, llm_kernels::utils::KVCacheType::kFp8E4M3)
INST_AV(float, unsigned char, llm_kernels::utils::KVCacheType::kFp8E5M2)
#undef INST_AV

// ---- PagedAttentionOp, InvokePagedAttention, FlashinferBatch ----
// generic stub（其余 fp8/bf16/fp32 路径模型不走，保留 stub 防 link error）。
template <typename S, typename C, llm_kernels::utils::KVCacheType K>
void PagedAttentionOp(int, int, int, int, int, float, float, void*, void*, void*, void*, void*, void*, int, int, cudaStream_t&, void*, size_t, const float*) {}
#define INST_PA(S, C, KV) template void PagedAttentionOp<S, C, KV>(int, int, int, int, int, float, float, void*, void*, void*, void*, void*, void*, int, int, cudaStream_t&, void*, size_t, const float*);
// fp16/bf16 + KVCacheType::kAuto 走真实 PagedAttentionCuda，下面显式特化，所以从 INST_PA 里去掉。
INST_PA(__half, unsigned char, llm_kernels::utils::KVCacheType::kFp8E4M3) INST_PA(__half, unsigned char, llm_kernels::utils::KVCacheType::kFp8E5M2)
INST_PA(__nv_bfloat16, unsigned char, llm_kernels::utils::KVCacheType::kFp8E4M3) INST_PA(__nv_bfloat16, unsigned char, llm_kernels::utils::KVCacheType::kFp8E5M2)
INST_PA(float, float, llm_kernels::utils::KVCacheType::kAuto) INST_PA(float, unsigned char, llm_kernels::utils::KVCacheType::kFp8E4M3) INST_PA(float, unsigned char, llm_kernels::utils::KVCacheType::kFp8E5M2)
#undef INST_PA

// ===== PagedAttentionOp<__half, __half, kAuto> —— vLLM v1 风格 decode 实例化 ======
// 把 PagedAttentionCuda<uint16_t, uint16_t, kAuto> 包装成 KsanaLLM 的 ABI；
// 来源：src/ksana_llm/kernels/nvidia/attention_kernel_wrapper.cpp 的 PAGED_ATTENTION 宏展开。
template <>
void PagedAttentionOp<__half, __half, llm_kernels::utils::KVCacheType::kAuto>(
    int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size,
    float k_scale, float v_scale, void* out, void* q_tensor_ptr,
    void* key_cache_ptrs, void* value_cache_ptrs, void* cache_offsets_ptr,
    void* context_lens_ptr, int max_context_len, int num_seqs, cudaStream_t& stream,
    void* workspace, size_t work_size, const float* alibi_slopes_ptr) {
  llm_kernels::iluvatar::PagedAttentionCuda<uint16_t, uint16_t, llm_kernels::utils::KVCacheType::kAuto> op;
  op.SetConfig(num_kv_heads, num_heads, head_size, block_size, stride_size, k_scale, v_scale);
  op.SetInput(reinterpret_cast<uint16_t*>(out),
              reinterpret_cast<const uint16_t*>(q_tensor_ptr),
              reinterpret_cast<uint16_t**>(key_cache_ptrs),
              reinterpret_cast<uint16_t**>(value_cache_ptrs),
              reinterpret_cast<const int*>(cache_offsets_ptr),
              reinterpret_cast<const int*>(context_lens_ptr),
              max_context_len, num_seqs, stream, workspace, work_size, alibi_slopes_ptr);
  op.Forward();
}

// ===== PagedAttentionOp<__nv_bfloat16, __nv_bfloat16, kAuto> —— bf16 同款实例化 =====
// TinyLlama 默认 KV cache dtype 自动选 bf16，所以这条特化必须存在。
template <>
void PagedAttentionOp<__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto>(
    int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size,
    float k_scale, float v_scale, void* out, void* q_tensor_ptr,
    void* key_cache_ptrs, void* value_cache_ptrs, void* cache_offsets_ptr,
    void* context_lens_ptr, int max_context_len, int num_seqs, cudaStream_t& stream,
    void* workspace, size_t work_size, const float* alibi_slopes_ptr) {
  llm_kernels::iluvatar::PagedAttentionCuda<__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto> op;
  op.SetConfig(num_kv_heads, num_heads, head_size, block_size, stride_size, k_scale, v_scale);
  op.SetInput(reinterpret_cast<__nv_bfloat16*>(out),
              reinterpret_cast<const __nv_bfloat16*>(q_tensor_ptr),
              reinterpret_cast<__nv_bfloat16**>(key_cache_ptrs),
              reinterpret_cast<__nv_bfloat16**>(value_cache_ptrs),
              reinterpret_cast<const int*>(cache_offsets_ptr),
              reinterpret_cast<const int*>(context_lens_ptr),
              max_context_len, num_seqs, stream, workspace, work_size, alibi_slopes_ptr);
  op.Forward();
}

template <typename S, typename C, llm_kernels::utils::KVCacheType K>
void InvokePagedAttention(void*, void*, void**, void**, void*, int, cudaStream_t, void*, int, int, int, int, int, int, float, float, int, void*, void*, int,
    std::optional<llm_kernels::iluvatar::RotaryEmbeddingCuda>&, void*, float, bool, void*, void*, size_t, int, const std::optional<void*>&, void*, void*, void*, void*, void*, int32_t*, int64_t, int, bool, bool, bool, float, size_t, bool, bool, bool, void*) {}
#define INST_IPA(S, C, KV) template void InvokePagedAttention<S, C, KV>(void*, void*, void**, void**, void*, int, cudaStream_t, void*, int, int, int, int, int, int, float, float, int, void*, void*, int, std::optional<llm_kernels::iluvatar::RotaryEmbeddingCuda>&, void*, float, bool, void*, void*, size_t, int, const std::optional<void*>&, void*, void*, void*, void*, void*, int32_t*, int64_t, int, bool, bool, bool, float, size_t, bool, bool, bool, void*);
// __half + kAuto 和 __nv_bfloat16 + kAuto 走真实 PagedAttentionCuda（下面显式特化）。
// 剩下两个 fp8 KV cache 变体 TinyLlama 不走，继续用 generic stub。
INST_IPA(__half, unsigned char, llm_kernels::utils::KVCacheType::kFp8E4M3)
INST_IPA(__half, unsigned char, llm_kernels::utils::KVCacheType::kFp8E5M2)
INST_IPA(__nv_bfloat16, unsigned char, llm_kernels::utils::KVCacheType::kFp8E4M3) INST_IPA(__nv_bfloat16, unsigned char, llm_kernels::utils::KVCacheType::kFp8E5M2)
INST_IPA(float, float, llm_kernels::utils::KVCacheType::kAuto) INST_IPA(float, unsigned char, llm_kernels::utils::KVCacheType::kFp8E4M3) INST_IPA(float, unsigned char, llm_kernels::utils::KVCacheType::kFp8E5M2)
#undef INST_IPA

// =========================================================================
// InvokePagedAttention<__half, __half, kAuto> —— fp16 decode 真实路径。
//
// 与 src/ksana_llm/kernels/nvidia/attention_kernel_wrapper.cpp 中
// `InvokePagedAttention` 的 `enable_blocked_multi_token_forwarding_kv == false`
// && `use_flashinfer_for_decode == false` 分支严格对齐：
//   1) torch::from_blob split QKV
//   2) RoPE on Q/K（复用已接好的 RotaryEmbeddingCuda）
//   3) CachePosCopy 把 K/V 写到 vLLM-v1 layout 的 per-seq paged cache
//      (key_cache_ptrs/value_cache_ptrs 是 num_seqs 个 device 指针组成的数组，
//       block_offsets 即 cache_offsets_ptr，按 block_size 寻址)
//   4) PagedAttentionCuda<uint16_t,uint16_t,kAuto>::Forward 跑 decode attention
//
// 注意：此分支**不**用 block_table_ptr（那是 flash-attn 的 layout）；
// 相应地 yaml 必须保持 enable_blocked_multi_token_forwarding_kv=false（默认值）。
// =========================================================================
template <>
void InvokePagedAttention<__half, __half, llm_kernels::utils::KVCacheType::kAuto>(
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
    bool use_flashinfer_for_decode, void* flashinfer_prefill_helper) {
  using SCALAR_T = __half;
  using CACHE_T = __half;
  constexpr auto KV_DTYPE = llm_kernels::utils::KVCacheType::kAuto;

  // ---- split QKV  (query_ptr [total_tokens, (num_heads + 2*num_kv_heads) * head_size]) ----
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kHalf);
  torch::Tensor qkv_tensor =
      torch::from_blob(query_ptr, {total_tokens, (num_heads + num_kv_heads * 2) * head_size}, options);
  auto tt = qkv_tensor.split({num_heads * head_size, num_kv_heads * head_size, num_kv_heads * head_size}, -1);
  void* q_tensor_ptr = tt[0].data_ptr();
  void* k_tensor_ptr = tt[1].data_ptr();
  void* v_tensor_ptr = tt[2].data_ptr();

  // ---- RoPE on Q/K ----
  if (!no_rope && rotary_embedding_cuda.has_value()) {
    rotary_embedding_cuda->SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos),
                                    reinterpret_cast<int64_t*>(rotary_embedding_mask),
                                    q_tensor_ptr, k_tensor_ptr, total_tokens, stream);
    rotary_embedding_cuda->Forward<SCALAR_T>();
  }

  // ---- scatter K/V 到 vLLM-v1 layout 的 per-seq paged cache ----
  // 与 nvidia 同名分支一致：CachePosCopy（不是 FlashAttn layout）。
  // SCALAR_T 用 uint16_t，因为 dtype_float16.cuh 里的 paged_attention 整套都按
  // uint16_t 处理 fp16 bit pattern。
  llm_kernels::iluvatar::CachePosCopy<uint16_t, uint16_t, KV_DTYPE>(
      reinterpret_cast<uint16_t*>(k_tensor_ptr),
      reinterpret_cast<uint16_t*>(v_tensor_ptr),
      key_cache_ptrs, value_cache_ptrs,
      reinterpret_cast<int*>(context_lens_ptr),
      reinterpret_cast<int*>(cache_offsets_ptr),
      block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
      k_scale, v_scale, stream);

  // ---- decode attention（PagedAttentionCuda）----
  const float* alibi_slopes_ptr =
      reinterpret_cast<const float*>(alibi_slopes.has_value() ? alibi_slopes.value() : nullptr);
  PagedAttentionOp<SCALAR_T, CACHE_T, KV_DTYPE>(
      num_heads, head_size, num_kv_heads, stride_size, block_size, k_scale, v_scale,
      output_ptr, q_tensor_ptr, key_cache_ptrs, value_cache_ptrs,
      cache_offsets_ptr, context_lens_ptr, max_context_len, seqs_num, stream,
      workspace_ptr, work_size, alibi_slopes_ptr);
}

// ===== InvokePagedAttention<__nv_bfloat16, __nv_bfloat16, kAuto> —— bf16 同款 =====
// TinyLlama 默认 KV cache 走 bf16，所以模型实际 dispatch 到这里，**不是**上面 fp16 那个。
template <>
void InvokePagedAttention<__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto>(
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
    bool use_flashinfer_for_decode, void* flashinfer_prefill_helper) {
  using SCALAR_T = __nv_bfloat16;
  using CACHE_T = __nv_bfloat16;
  constexpr auto KV_DTYPE = llm_kernels::utils::KVCacheType::kAuto;

  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kBFloat16);
  torch::Tensor qkv_tensor =
      torch::from_blob(query_ptr, {total_tokens, (num_heads + num_kv_heads * 2) * head_size}, options);
  auto tt = qkv_tensor.split({num_heads * head_size, num_kv_heads * head_size, num_kv_heads * head_size}, -1);
  void* q_tensor_ptr = tt[0].data_ptr();
  void* k_tensor_ptr = tt[1].data_ptr();
  void* v_tensor_ptr = tt[2].data_ptr();

  // BF16 decode 端 RoPE：之前因 KV cache 跨 layer 错位导致输出全乱，临时关掉过；
  // ProcessKvList 修复后重新打开，否则没有位置信息会导致死循环输出。
  if (!no_rope && rotary_embedding_cuda.has_value()) {
    rotary_embedding_cuda->SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos),
                                    reinterpret_cast<int64_t*>(rotary_embedding_mask),
                                    q_tensor_ptr, k_tensor_ptr, total_tokens, stream);
    rotary_embedding_cuda->Forward<SCALAR_T>();
  }

  llm_kernels::iluvatar::CachePosCopy<__nv_bfloat16, __nv_bfloat16, KV_DTYPE>(
      reinterpret_cast<__nv_bfloat16*>(k_tensor_ptr),
      reinterpret_cast<__nv_bfloat16*>(v_tensor_ptr),
      key_cache_ptrs, value_cache_ptrs,
      reinterpret_cast<int*>(context_lens_ptr),
      reinterpret_cast<int*>(cache_offsets_ptr),
      block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
      k_scale, v_scale, stream);

  // ----- BF16 paged decode 走 3rdparty PagedAttentionCuda<bf16,bf16,kAuto> -----
  // 这是 vLLM-v1 风格的 paged attention，与上面 fp16 path 完全对称。曾经因为 bringup
  // 期间 nvidia 模板编译问题临时绕路到 naive gqa_paged_decode_bf16，并且为了从 device
  // 读 ctx_len 还插了 cudaStreamSynchronize；现在 PagedAttentionOp 路径已验证可用且
  // 比 naive 快 ~2.3x（786 → 342 ms / req on TinyLlama），裸路径已废弃。
  // 旧开关 `KSANA_USE_NVIDIA_DECODE` 已无意义并被移除。
  const float* alibi_slopes_ptr =
      reinterpret_cast<const float*>(alibi_slopes.has_value() ? alibi_slopes.value() : nullptr);
  PagedAttentionOp<SCALAR_T, CACHE_T, KV_DTYPE>(
      num_heads, head_size, num_kv_heads, stride_size, block_size, k_scale, v_scale,
      output_ptr, q_tensor_ptr, key_cache_ptrs, value_cache_ptrs,
      cache_offsets_ptr, context_lens_ptr, max_context_len, seqs_num, stream,
      workspace_ptr, work_size, alibi_slopes_ptr);
}


template <typename S, llm_kernels::utils::KVCacheType K, typename I>
void FlashinferBatchPrefillPagedAttentionOp(int, int, int, int, void*, void*, void*, void*, I*, void*, int, int, float*, bool, float, void*, size_t, void*, void*, bool, cudaStream_t&, void*) {}
template void FlashinferBatchPrefillPagedAttentionOp<__nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto, int>(int, int, int, int, void*, void*, void*, void*, int*, void*, int, int, float*, bool, float, void*, size_t, void*, void*, bool, cudaStream_t&, void*);
template void FlashinferBatchPrefillPagedAttentionOp<__half, llm_kernels::utils::KVCacheType::kAuto, int>(int, int, int, int, void*, void*, void*, void*, int*, void*, int, int, float*, bool, float, void*, size_t, void*, void*, bool, cudaStream_t&, void*);


}  // namespace ksana_llm




// ===================================================================
// Section 7: RotaryEmbeddingCuda 已迁移到 libllm_kernels_iluvatar_kernel_rotary_embedding.a。
//             这里不再提供 stub，避免 multiple definition。
// ===================================================================
