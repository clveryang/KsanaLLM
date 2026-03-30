/* Copyright 2025 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include "ksana_llm/utils/tokenizer.h"

#include "gflags/gflags.h"
#include "test.h"

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

TEST(TokenizerTest, WrongTokenizerPath) {
  Status status = Singleton<Tokenizer>::GetInstance()->InitTokenizer("wrong_path");
  EXPECT_EQ(status.GetCode(), RET_INVALID_ARGUMENT);
}

TEST(TokenizerTest, TokenizeTest) {
  Singleton<Tokenizer>::GetInstance()->InitTokenizer("/model/llama-hf/7B");
  std::string prompt = "Hello. What's your name?";
  std::vector<int> token_list;
  std::vector<int> target_token_list = {1, 15043, 29889, 1724, 29915, 29879, 596, 1024, 29973};
  Singleton<Tokenizer>::GetInstance()->Encode(prompt, token_list, true);
  EXPECT_EQ(token_list.size(), target_token_list.size());
  for (size_t i = 0; i < token_list.size(); ++i) {
    EXPECT_EQ(token_list[i], target_token_list[i]);
  }

  std::string output_prompt = "";
  std::string target_prompt = "Hello. What's your name? My name is David.";
  token_list.emplace_back(1619);
  token_list.emplace_back(1024);
  token_list.emplace_back(338);
  token_list.emplace_back(4699);
  token_list.emplace_back(29889);
  Singleton<Tokenizer>::GetInstance()->Decode(token_list, output_prompt, true);
  EXPECT_EQ(target_prompt, output_prompt);

  Singleton<Tokenizer>::GetInstance()->DestroyTokenizer();
}

TEST(TokenizerTest, GetVocabInfoTest) {
  // Initialize tokenizer first
  Singleton<Tokenizer>::GetInstance()->InitTokenizer("/model/llama-hf/7B");

  std::vector<std::string> vocab;
  int vocab_size = 32000;
  std::vector<int> stop_token_ids;
  int vocab_type = 0;
  bool add_prefix_space = false;

  Status status = Singleton<Tokenizer>::GetInstance()->GetVocabInfo(vocab, vocab_size, stop_token_ids, vocab_type,
                                                                    add_prefix_space);
  EXPECT_TRUE(status.OK());

  // Verify basic functionality
  EXPECT_GE(vocab_size, 32000);                              // Should be at least the input size
  EXPECT_EQ(vocab.size(), static_cast<size_t>(vocab_size));  // vocab vector should match vocab_size
  EXPECT_GT(stop_token_ids.size(), 0);                       // Should have at least one stop token (EOS)

  // Verify some tokens exist
  EXPECT_FALSE(vocab[1].empty());  // Token ID 1 should exist (BOS token for LLaMA)

  // Clean up
  Singleton<Tokenizer>::GetInstance()->DestroyTokenizer();
}

TEST(TokenizerTest, GetVocabInfoErrorHandlingTest) {
  // Test error handling when tokenizer is not properly initialized
  // This should trigger the catch block in GetVocabInfo

  // Ensure tokenizer is destroyed/uninitialized
  Singleton<Tokenizer>::GetInstance()->DestroyTokenizer();

  std::vector<std::string> vocab;
  int vocab_size = 32000;
  std::vector<int> stop_token_ids;
  int vocab_type = 0;
  bool add_prefix_space = false;

  // Call GetVocabInfo without proper initialization
  // This should trigger the exception handling in GetVocabInfo
  Status status = Singleton<Tokenizer>::GetInstance()->GetVocabInfo(vocab, vocab_size, stop_token_ids, vocab_type,
                                                                    add_prefix_space);

  // Verify that the error is properly handled
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RET_INVALID_ARGUMENT);
  EXPECT_EQ(status.GetMessage(), "Failed to extract tokenizer information");
}

// Tests for non-fast tokenizer vocab_type detection using Python mock objects.
// These tests verify the detection paths in GetVocabInfo for non-PreTrainedTokenizerFast:
//   1. byte_encoder attribute → BYTE_LEVEL(2)
//   2. sp_model attribute + "<0x0A>" in vocab → BYTE_FALLBACK(1)
//   3. no matching attributes → RAW(0)

TEST(TokenizerTest, DetectMetadata_ByteLevel_MockTokenizer) {
  // Mock a non-fast tokenizer with byte_encoder and get_vocab (like Kimi K2's TikTokenTokenizer)
  pybind11::gil_scoped_acquire acquire;

  pybind11::module types = pybind11::module::import("types");
  pybind11::object mock_tokenizer = types.attr("SimpleNamespace")();
  mock_tokenizer.attr("byte_encoder") = pybind11::dict();
  // get_vocab returns a minimal vocab dict
  pybind11::dict vocab_dict;
  vocab_dict["hello"] = 0;
  vocab_dict["world"] = 1;
  mock_tokenizer.attr("get_vocab") = pybind11::cpp_function([vocab_dict]() { return vocab_dict; });
  // eos_token_id
  mock_tokenizer.attr("eos_token_id") = 1;

  auto tokenizer = Singleton<Tokenizer>::GetInstance();
  tokenizer->tokenizer_ = mock_tokenizer;

  std::vector<std::string> vocab;
  int vocab_size = 2;
  std::vector<int> stop_token_ids;
  int vocab_type = 0;
  bool add_prefix_space = false;

  Status status = tokenizer->GetVocabInfo(vocab, vocab_size, stop_token_ids, vocab_type, add_prefix_space);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(vocab_type, 2) << "Expected BYTE_LEVEL(2) for tokenizer with byte_encoder attribute";
  EXPECT_FALSE(add_prefix_space);

  tokenizer->tokenizer_ = pybind11::none();
}

TEST(TokenizerTest, DetectMetadata_ByteFallback_MockTokenizer) {
  // Mock a non-fast tokenizer with sp_model and byte fallback vocab (like LlamaTokenizer)
  pybind11::gil_scoped_acquire acquire;

  pybind11::module types = pybind11::module::import("types");
  pybind11::object mock_tokenizer = types.attr("SimpleNamespace")();
  mock_tokenizer.attr("sp_model") = pybind11::none();
  // get_vocab with <0x0A> byte fallback token
  pybind11::dict vocab_dict;
  vocab_dict["<unk>"] = 0;
  vocab_dict["<s>"] = 1;
  vocab_dict["</s>"] = 2;
  vocab_dict["<0x0A>"] = 3;
  vocab_dict["hello"] = 4;
  mock_tokenizer.attr("get_vocab") = pybind11::cpp_function([vocab_dict]() { return vocab_dict; });
  mock_tokenizer.attr("eos_token_id") = 2;

  auto tokenizer = Singleton<Tokenizer>::GetInstance();
  tokenizer->tokenizer_ = mock_tokenizer;

  std::vector<std::string> vocab;
  int vocab_size = 5;
  std::vector<int> stop_token_ids;
  int vocab_type = 0;
  bool add_prefix_space = false;

  Status status = tokenizer->GetVocabInfo(vocab, vocab_size, stop_token_ids, vocab_type, add_prefix_space);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(vocab_type, 1) << "Expected BYTE_FALLBACK(1) for tokenizer with sp_model + <0x0A> in vocab";
  EXPECT_TRUE(add_prefix_space);

  tokenizer->tokenizer_ = pybind11::none();
}

TEST(TokenizerTest, DetectMetadata_SpModelNoByteFallback_MockTokenizer) {
  // Mock a non-fast tokenizer with sp_model but no byte fallback tokens (like ChatGLM3)
  pybind11::gil_scoped_acquire acquire;

  pybind11::module types = pybind11::module::import("types");
  pybind11::object mock_tokenizer = types.attr("SimpleNamespace")();
  mock_tokenizer.attr("sp_model") = pybind11::none();
  // get_vocab without <0x0A>
  pybind11::dict vocab_dict;
  vocab_dict["<unk>"] = 0;
  vocab_dict["hello"] = 1;
  vocab_dict["world"] = 2;
  mock_tokenizer.attr("get_vocab") = pybind11::cpp_function([vocab_dict]() { return vocab_dict; });
  mock_tokenizer.attr("eos_token_id") = 0;

  auto tokenizer = Singleton<Tokenizer>::GetInstance();
  tokenizer->tokenizer_ = mock_tokenizer;

  std::vector<std::string> vocab;
  int vocab_size = 3;
  std::vector<int> stop_token_ids;
  int vocab_type = 0;
  bool add_prefix_space = false;

  Status status = tokenizer->GetVocabInfo(vocab, vocab_size, stop_token_ids, vocab_type, add_prefix_space);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(vocab_type, 0) << "Expected RAW(0) for sp_model tokenizer without byte fallback tokens";
  EXPECT_FALSE(add_prefix_space);

  tokenizer->tokenizer_ = pybind11::none();
}

TEST(TokenizerTest, DetectMetadata_Raw_MockTokenizer) {
  // Mock a non-fast tokenizer with no byte_encoder and no sp_model → RAW
  pybind11::gil_scoped_acquire acquire;

  pybind11::module types = pybind11::module::import("types");
  pybind11::object mock_tokenizer = types.attr("SimpleNamespace")();
  // get_vocab only
  pybind11::dict vocab_dict;
  vocab_dict["hello"] = 0;
  vocab_dict["world"] = 1;
  mock_tokenizer.attr("get_vocab") = pybind11::cpp_function([vocab_dict]() { return vocab_dict; });
  mock_tokenizer.attr("eos_token_id") = 1;

  auto tokenizer = Singleton<Tokenizer>::GetInstance();
  tokenizer->tokenizer_ = mock_tokenizer;

  std::vector<std::string> vocab;
  int vocab_size = 2;
  std::vector<int> stop_token_ids;
  int vocab_type = 0;
  bool add_prefix_space = false;

  Status status = tokenizer->GetVocabInfo(vocab, vocab_size, stop_token_ids, vocab_type, add_prefix_space);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(vocab_type, 0) << "Expected RAW(0) for tokenizer without byte_encoder or sp_model";
  EXPECT_FALSE(add_prefix_space);

  tokenizer->tokenizer_ = pybind11::none();
}

}  // namespace ksana_llm
