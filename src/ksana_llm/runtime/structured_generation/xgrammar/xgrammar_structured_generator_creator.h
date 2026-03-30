/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/structured_generation/structured_generator_factory.h"
#include "ksana_llm/runtime/structured_generation/xgrammar/xgrammar_structured_generator.h"
#include "ksana_llm/utils/grammar_backend.h"

namespace ksana_llm {

/*!
 * \\brief Creator for grammar constraint generators.
 */
class GrammarGeneratorCreator : public GeneratorCreator {
 public:
  GrammarGeneratorCreator(const std::vector<std::string>& vocab, int vocab_size, const std::vector<int>& stop_token_ids,
                          int vocab_type = 0, bool add_prefix_space = false) {
    try {
      grammar_backend_ = GrammarBackend::Create(vocab, vocab_size, stop_token_ids, vocab_type, add_prefix_space);
    } catch (const std::exception& e) {
      throw std::runtime_error("Grammar backend creation failed: " + std::string(e.what()));
    }
  }

  std::shared_ptr<StructuredGeneratorInterface> CreateGenerator(const StructuredGeneratorConfig& config) override {
    if (!grammar_backend_) {
      throw std::runtime_error("Grammar backend is not initialized");
    }

    // Compile the grammar from the constraint specification
    auto compiled_grammar = grammar_backend_->CompileJSONSchema(config.constraint_spec);
    if (!compiled_grammar) {
      throw std::runtime_error("Failed to compile grammar from constraint specification");
    }

    return std::make_shared<GrammarStructuredGenerator>(compiled_grammar);
  }

  StructuredConstraintType GetConstraintType() const override { return StructuredConstraintType::JSON; }

 private:
  std::shared_ptr<GrammarBackend> grammar_backend_;
};

}  // namespace ksana_llm
