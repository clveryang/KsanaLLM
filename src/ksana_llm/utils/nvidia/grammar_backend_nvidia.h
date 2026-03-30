/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ksana_llm/utils/grammar_backend.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"
#include "xgrammar/xgrammar.h"

namespace ksana_llm {

class GrammarBackendNvidia : public GrammarBackend {
 public:
  // vocab_type: 0 = RAW, 1 = BYTE_FALLBACK, 2 = BYTE_LEVEL (detected by Python side)
  GrammarBackendNvidia(const std::vector<std::string>& vocab, int vocab_size, const std::vector<int>& stop_token_ids,
                       int vocab_type = 0, bool add_prefix_space = false);
  ~GrammarBackendNvidia() override;

  std::shared_ptr<CompiledGrammar> CompileJSONSchema(const std::string& schema) override;
  std::shared_ptr<GrammarMatcherWrapper> CreateMatcher(std::shared_ptr<CompiledGrammar> grammar) override;
  const xgrammar::TokenizerInfo& GetTokenizerInfo() const override;
  bool IsInitialized() const override { return initialized_; }

 private:
  std::unique_ptr<xgrammar::TokenizerInfo> tokenizer_info_;
  std::unique_ptr<xgrammar::GrammarCompiler> compiler_;
  bool initialized_ = false;
};

}  // namespace ksana_llm
