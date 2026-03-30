/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/nvidia/grammar_backend_nvidia.h"
#include "ksana_llm/utils/grammar_matcher.h"

namespace ksana_llm {

constexpr int kDefaultMaxThreads = 8;
constexpr bool kDefaultCacheEnabled = true;
constexpr int kDefaultMaxMemoryBytes = -1;  // unlimited

GrammarBackendNvidia::GrammarBackendNvidia(const std::vector<std::string>& vocab, int vocab_size,
                                           const std::vector<int>& stop_token_ids, int vocab_type,
                                           bool add_prefix_space) {
  KLLM_LOG_INFO << "GrammarBackendNvidia: vocab_type=" << vocab_type
                << ", add_prefix_space=" << (add_prefix_space ? "true" : "false");

  std::vector<int32_t> stop_tokens_int32(stop_token_ids.begin(), stop_token_ids.end());
  xgrammar::VocabType xgrammar_vocab_type = static_cast<xgrammar::VocabType>(vocab_type);

  tokenizer_info_ = std::make_unique<xgrammar::TokenizerInfo>(vocab, xgrammar_vocab_type, vocab_size, stop_tokens_int32,
                                                              add_prefix_space);

  compiler_ = std::make_unique<xgrammar::GrammarCompiler>(*tokenizer_info_, kDefaultMaxThreads, kDefaultCacheEnabled,
                                                          kDefaultMaxMemoryBytes);

  initialized_ = true;
}

GrammarBackendNvidia::~GrammarBackendNvidia() {}

std::shared_ptr<CompiledGrammar> GrammarBackendNvidia::CompileJSONSchema(const std::string& schema) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto compiled_grammar = std::make_shared<CompiledGrammar>(
      compiler_->CompileJSONSchema(schema, true /* any_whitespace */, std::nullopt /* indent */,
                                   std::nullopt /* separators */, true /* strict_mode */));

  KLLM_LOG_DEBUG << "JSON schema compiled, memory usage: " << compiled_grammar->MemorySizeBytes() << " bytes";
  return compiled_grammar;
}

std::shared_ptr<GrammarMatcherWrapper> GrammarBackendNvidia::CreateMatcher(std::shared_ptr<CompiledGrammar> grammar) {
  return GrammarMatcherWrapper::Create(grammar);
}

const xgrammar::TokenizerInfo& GrammarBackendNvidia::GetTokenizerInfo() const { return *tokenizer_info_; }

}  // namespace ksana_llm
