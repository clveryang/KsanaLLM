/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include "ksana_llm/utils/status.h"

namespace py = pybind11;

namespace ksana_llm {

// Wraps the tokenizer for internal various usage
class Tokenizer {
 public:
  // Initialize the tokenizer from the given tokenizer_path.
  Status InitTokenizer(const std::string& tokenizer_path);

  // Destroy the tokenizer.
  void DestroyTokenizer();

  // Decode the given input token ids into string
  Status Decode(std::vector<int>& input_tokens, std::string& output, bool skip_special_tokens = true);

  // Encode the given prompt into token ids
  Status Encode(const std::string& prompt, std::vector<int>& input_tokens, bool add_special_tokens = true);

  // Extract vocabulary information and detect tokenizer metadata for grammar-guided generation.
  //
  // vocab_type: detected tokenizer encoding type, maps to xgrammar::VocabType:
  //   0 = RAW           — tokens are raw strings, no special encoding
  //   1 = BYTE_FALLBACK — SentencePiece with byte fallback (e.g. <0x1B> for \x1b)
  //   2 = BYTE_LEVEL    — GPT-2 style bytes_to_unicode mapping (e.g. Ġ for space)
  //
  // add_prefix_space: whether the tokenizer prepends a space before text during tokenization.
  //   Detected from tokenizer attribute or XGrammar metadata (for fast tokenizers).
  Status GetVocabInfo(std::vector<std::string>& vocab, int& vocab_size, std::vector<int>& stop_token_ids,
                      int& vocab_type, bool& add_prefix_space);

 public:
  py::object tokenizer_;
};

}  // namespace ksana_llm
