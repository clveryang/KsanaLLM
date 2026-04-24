

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

namespace ksana_llm {

// === BatchedMatMulLayer ===
Status BatchedMatMulLayer::Init(const std::vector<std::any>&, const RuntimeConfig&, std::shared_ptr<Context>, int) { return Status(); }
size_t BatchedMatMulLayer::GetWorkspaceSize() { return 0; }
Status BatchedMatMulLayer::Forward(const std::vector<Tensor>&, std::vector<Tensor>&) { return Status(RET_UNDEFINED_REFERENCE, "N/A"); }

// === MoeLayer ===
Status MoeLayer::Init(const std::vector<std::any>&, const RuntimeConfig&, std::shared_ptr<Context>, int) { return Status(); }
size_t MoeLayer::GetWorkspaceSize() { return 0; }
Status MoeLayer::SetWorkspaceBuffer(const std::shared_ptr<Tensor>&) { return Status(); }
Status MoeLayer::Preprocess(const ModelConfig&, const RuntimeConfig&) { return Status(); }
Status MoeLayer::Forward(const std::vector<Tensor>&, std::vector<Tensor>&) { return Status(RET_UNDEFINED_REFERENCE, "N/A"); }

// === AllReduceResidualAddNormLayer ===
Status AllReduceResidualAddNormLayer::Init(const std::vector<std::any>&, const RuntimeConfig&, std::shared_ptr<Context>, int) { return Status(); }
Status AllReduceResidualAddNormLayer::Forward(const std::vector<Tensor>&, std::vector<Tensor>&) { return Status(RET_UNDEFINED_REFERENCE, "N/A"); }

// === CustomAllReduceSumLayer ===
Status CustomAllReduceSumLayer::Init(const std::vector<std::any>&, const RuntimeConfig&, std::shared_ptr<Context>, int) { return Status(); }
Status CustomAllReduceSumLayer::Forward(const std::vector<Tensor>&, std::vector<Tensor>&) { return Status(); }
void CustomAllReduceSumLayer::Clear() {}
void CustomAllReduceSumLayer::ResetInputBuffer(void*) {}
void CustomAllReduceSumLayer::ResetSignalBuffer(void*, size_t) {}

// === FlashMlaAttentionLayer ===
Status FlashMlaAttentionLayer::Init(const std::vector<std::any>&, const RuntimeConfig&, std::shared_ptr<Context>, int) { return Status(); }
size_t FlashMlaAttentionLayer::GetWorkspaceSize() { return 0; }
Status FlashMlaAttentionLayer::Forward(const std::vector<Tensor>&, std::vector<Tensor>&) { return Status(RET_UNDEFINED_REFERENCE, "N/A"); }

// === PagedMlaAttentionLayer ===
Status PagedMlaAttentionLayer::Init(const std::vector<std::any>&, const RuntimeConfig&, std::shared_ptr<Context>, int) { return Status(); }
Status PagedMlaAttentionLayer::Forward(const std::vector<Tensor>&, std::vector<Tensor>&) { return Status(RET_UNDEFINED_REFERENCE, "N/A"); }

// === FlashSparseMlaAttentionLayer ===
Status FlashSparseMlaAttentionLayer::Init(const std::vector<std::any>&, const RuntimeConfig&, std::shared_ptr<Context>, int) { return Status(); }
Status FlashSparseMlaAttentionLayer::Forward(const std::vector<Tensor>&, std::vector<Tensor>&) { return Status(RET_UNDEFINED_REFERENCE, "N/A"); }

// === PagedSparseMlaAttentionLayer ===
Status PagedSparseMlaAttentionLayer::Init(const std::vector<std::any>&, const RuntimeConfig&, std::shared_ptr<Context>, int) { return Status(); }
Status PagedSparseMlaAttentionLayer::Forward(const std::vector<Tensor>&, std::vector<Tensor>&) { return Status(RET_UNDEFINED_REFERENCE, "N/A"); }

// === FlashSparseMlaIndexerLayer ===
Status FlashSparseMlaIndexerLayer::Init(const std::vector<std::any>&, const RuntimeConfig&, std::shared_ptr<Context>, int) { return Status(); }
size_t FlashSparseMlaIndexerLayer::GetWorkspaceSize() { return 0; }
Status FlashSparseMlaIndexerLayer::Forward(const std::vector<Tensor>&, std::vector<Tensor>&) { return Status(RET_UNDEFINED_REFERENCE, "N/A"); }

// === PagedSparseMlaIndexerLayer ===
Status PagedSparseMlaIndexerLayer::Init(const std::vector<std::any>&, const RuntimeConfig&, std::shared_ptr<Context>, int) { return Status(); }
size_t PagedSparseMlaIndexerLayer::GetWorkspaceSize() { return 0; }
Status PagedSparseMlaIndexerLayer::Forward(const std::vector<Tensor>&, std::vector<Tensor>&) { return Status(RET_UNDEFINED_REFERENCE, "N/A"); }

}  // namespace ksana_llm



// Provide method bodies for llm_kernels::utils::KllmException (typeinfo emitted here)

#include "csrc/utils/nvidia/kllm_exception.h"
namespace llm_kernels { namespace utils {
KllmException::KllmException(char const* file, std::size_t line, std::string const& msg)
    : std::runtime_error(msg), mNbFrames(0) {}
KllmException::~KllmException() noexcept = default;
std::string KllmException::getTrace() const { return ""; }
std::string KllmException::demangle(char const* name) { return name ? name : ""; }
}}

// Stub for llm_kernels::nvidia::GPUGemmAlgoHelper (Iluvatar uses default cublas, no algo tuning)

#include "csrc/kernels/nvidia/gemm_wrapper/gemm_algo_map.h"
namespace llm_kernels { namespace nvidia {
bool GPUGemmAlgoHelper::SaveToYaml(const std::string&) { return false; }
bool GPUGemmAlgoHelper::LoadFromYaml(const std::string&) { return false; }
bool GPUGemmAlgoHelper::AddGemmAlgo(const uint32_t, const uint32_t, GemmAlgoFingerprint, GemmAlgoInfo) { return false; }
const GemmAlgoInfo GPUGemmAlgoHelper::GetGemmAlgo(const uint32_t, const uint32_t, const uint64_t, const uint64_t,
    const uint64_t, const uint64_t, const cudaDataType_t, const cudaDataType_t, const cudaDataType_t,
    const cudaDataType_t, const cublasOperation_t, const cublasOperation_t) { return {}; }
bool GPUGemmAlgoHelper::IsGemmAlgoExist(const uint32_t, const uint32_t, const uint64_t, const uint64_t,
    const uint64_t, const uint64_t, const cudaDataType_t, const cudaDataType_t, const cudaDataType_t,
    const cudaDataType_t, const cublasOperation_t, const cublasOperation_t) { return false; }
std::unordered_map<GemmAlgoFingerprint, GemmAlgoInfo, GemmFingerprintHasher>& GPUGemmAlgoHelper::GetOrCreateAlgoMap(
    const uint32_t sm, const uint32_t cuda_ver) {
  static std::unordered_map<GemmAlgoFingerprint, GemmAlgoInfo, GemmFingerprintHasher> empty;
  return empty;
}
}}
