/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/grammar_backend.h"
#include "ksana_llm/utils/device_types.h"

#include "ksana_llm/utils/nvidia/grammar_backend_nvidia.h"

namespace ksana_llm {

std::unique_ptr<GrammarBackend> GrammarBackend::Create(const std::vector<std::string>& vocab, int vocab_size,
                                                       const std::vector<int>& stop_token_ids, int vocab_type,
                                                       bool add_prefix_space) {
  if (ACTIVE_DEVICE_TYPE == DEVICE_TYPE_NVIDIA) {
    return std::make_unique<GrammarBackendNvidia>(vocab, vocab_size, stop_token_ids, vocab_type, add_prefix_space);
  } else {
    KLLM_LOG_WARNING << "GrammarBackend is not supported on Ascend platforms, returning nullptr";
    return nullptr;
  }
}

}  // namespace ksana_llm
