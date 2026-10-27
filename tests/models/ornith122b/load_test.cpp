// SPDX-License-Identifier: MIT
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/models/ornith122b/config.hpp"
#include "src/models/ornith122b/weights.hpp"

int main(int argc, char* argv[]) {
  std::string model_path =
      "D:\\AI_Stack\\Models\\Harsha001\\Ornith-1.5-122B-ROCmFPX\\"
      "Ornith-1.5-122B-ROCmFP4-Q4_0_STRIX-00001-of-00002.gguf";
  if (argc > 1) {
    model_path = argv[1];
  }

  std::cout << "Loading model: " << model_path << "\n";
  const auto t0 = std::chrono::steady_clock::now();

  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::cerr << "Failed to open GGUF: " << error << "\n";
    return 1;
  }

  const auto t_mapped = std::chrono::steady_clock::now();
  const double map_s =
      std::chrono::duration<double>(t_mapped - t0).count();
  std::cout << "Mapped GGUF in " << map_s << " s.\n";
  std::cout << "  Version: " << reader->GetVersion() << "\n";
  std::cout << "  Tensors: " << reader->GetTensorCount() << "\n";
  std::cout << "  Mapped regions (shards): "
            << reader->GetMappedRegions().size() << "\n";

  assert(reader->GetTensorCount() == 879);
  assert(reader->GetMappedRegions().size() == 2);

  std::cout << "Parsing Config...\n";
  auto config = gufo::models::ornith122b::Config::FromGguf(*reader, &error);
  if (!config) {
    std::cerr << "Failed to parse Config: " << error << "\n";
    return 1;
  }

  std::cout << "Config parsed successfully:\n";
  std::cout << "  Layers: " << config->num_layers << "\n";
  std::cout << "  Hidden size: " << config->hidden_size << "\n";
  std::cout << "  Vocab size: " << config->vocab_size << "\n";
  std::cout << "  Full attention interval: " << config->full_attention_interval << "\n";
  std::cout << "  SSM value dim: " << config->SsmValueDim() << "\n";
  std::cout << "  SSM conv channels: " << config->SsmConvChannels() << "\n";
  std::cout << "  Attention Q dim: " << config->AttentionQDim() << "\n";
  std::cout << "  Attention KV dim: " << config->AttentionKvDim() << "\n";
  std::cout << "  Routed experts: " << config->num_experts << " (used: " << config->num_experts_used << ")\n";
  std::cout << "  Expert FF: " << config->expert_ff << "\n";
  std::cout << "  Shared expert FF: " << config->shared_expert_ff << "\n";

  assert(config->num_layers == 48);
  assert(config->hidden_size == 3072);
  assert(config->vocab_size == 248320);
  assert(config->full_attention_interval == 4);
  assert(config->num_experts == 256);
  assert(config->num_experts_used == 8);

  std::cout << "Binding Weights...\n";
  auto weights = gufo::models::ornith122b::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    std::cerr << "Failed to bind Weights: " << error << "\n";
    return 1;
  }

  const auto t_bound = std::chrono::steady_clock::now();
  const double bind_s =
      std::chrono::duration<double>(t_bound - t_mapped).count();
  std::cout << "Bound all " << weights->layers.size() << " layers in "
            << bind_s << " s.\n";

  // Validate global weights
  assert(!weights->token_embd.empty());
  assert(!weights->output.empty());
  assert(!weights->output_norm.empty());
  std::cout << "  token_embd format: " << gufo::core::ToString(weights->token_embd.type) << "\n";
  std::cout << "  output format: " << gufo::core::ToString(weights->output.type) << "\n";
  std::cout << "  output_norm format: " << gufo::core::ToString(weights->output_norm.type) << "\n";

  // Validate layer weights
  std::size_t linear_count = 0;
  std::size_t full_count = 0;
  for (std::uint32_t i = 0; i < weights->layers.size(); ++i) {
    const auto& l = weights->layers[i];
    assert(!l.attn_norm.empty());
    assert(!l.post_attention_norm.empty());
    assert(!l.router.empty());
    assert(!l.ffn_gate_exps.empty());
    assert(!l.ffn_up_exps.empty());
    assert(!l.ffn_down_exps.empty());
    assert(!l.shexp_gate.empty());
    assert(!l.shexp_up.empty());
    assert(!l.shexp_down.empty());
    assert(l.ffn_gate_exps.experts == 256);

    if (l.is_full_attention) {
      ++full_count;
      assert(!l.attn_q.empty());
      assert(!l.attn_k.empty());
      assert(!l.attn_v.empty());
      assert(!l.attn_output.empty());
      assert(!l.attn_q_norm.empty());
      assert(!l.attn_k_norm.empty());
    } else {
      ++linear_count;
      assert(!l.attn_qkv.empty());
      assert(!l.attn_gate.empty());
      assert(!l.ssm_conv1d.empty());
      assert(!l.ssm_alpha.empty());
      assert(!l.ssm_beta.empty());
      assert(!l.ssm_dt.empty());
      assert(!l.ssm_a.empty());
      assert(!l.ssm_norm.empty());
      assert(!l.ssm_out.empty());
    }
  }

  std::cout << "Verified " << weights->layers.size() << " layers: "
            << linear_count << " linear SSM layers, "
            << full_count << " full attention layers.\n";
  std::cout << "All assertions passed successfully! Ornith 1.5 122B ROCmFP4 is ready!\n";
  return 0;
}
