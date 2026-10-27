// SPDX-License-Identifier: MIT
#ifndef GUFO_MODELS_ORNITH122B_CONFIG_HPP_
#define GUFO_MODELS_ORNITH122B_CONFIG_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::ornith122b {

/// Architecture parameters of an Ornith 1.5 122B (`qwen35moe`) GGUF artifact.
struct Config {
  std::uint32_t num_layers{48};      ///< Trunk layers (48 layers).
  std::uint32_t hidden_size{3072};   ///< n_embd, 3072.
  std::uint32_t vocab_size{248320};  ///< 248320.
  std::uint32_t context_length{262144}; ///< 256K.
  float rms_eps{1e-6F};

  // Full (gated GQA) attention on every full_attention_interval-th layer.
  std::uint32_t full_attention_interval{4};  ///< 4 (every 4th layer: 3, 7, 11, ...)
  std::uint32_t num_heads{32};               ///< 32
  std::uint32_t num_kv_heads{2};            ///< 2 (GQA)
  std::uint32_t head_dim{256};               ///< 256
  std::uint32_t rotary_dim{64};              ///< 64
  float rope_theta{10000000.0F};             ///< 1e7
  std::array<std::uint32_t, 4> rope_sections{11, 11, 10, 0};

  // Gated DeltaNet linear attention (ssm).
  std::uint32_t ssm_conv_kernel{4};   ///< 4
  std::uint32_t ssm_head_dim{128};    ///< 128 (state_size; key and value dim)
  std::uint32_t ssm_num_k_heads{16};  ///< 16 (group_count)
  std::uint32_t ssm_num_v_heads{64};  ///< 64 (time_step_rank)
  std::uint32_t ssm_inner_size{8192}; ///< 8192

  // Mixture of experts with one always-on shared expert.
  std::uint32_t num_experts{256};      ///< 256
  std::uint32_t num_experts_used{8};   ///< 8
  std::uint32_t expert_ff{1024};       ///< 1024
  std::uint32_t shared_expert_ff{1024};///< 1024

  // Speculative MTP block (`nextn`).
  std::uint32_t nextn_layers{0};

  [[nodiscard]] std::uint32_t SsmKeyDim() const noexcept {
    return ssm_num_k_heads * ssm_head_dim;
  }
  [[nodiscard]] std::uint32_t SsmValueDim() const noexcept {
    return ssm_num_v_heads * ssm_head_dim;
  }
  [[nodiscard]] std::uint32_t SsmConvChannels() const noexcept {
    return 2 * SsmKeyDim() + SsmValueDim();
  }
  [[nodiscard]] std::uint32_t AttentionQDim() const noexcept {
    return num_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t AttentionKvDim() const noexcept {
    return num_kv_heads * head_dim;
  }
  [[nodiscard]] bool IsFullAttentionLayer(std::uint32_t layer) const noexcept {
    return full_attention_interval > 0 &&
           (layer % full_attention_interval == (full_attention_interval - 1));
  }

  /// Parses and validates Ornith 1.5 122B hyperparameters from `gguf`.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& gguf, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::ornith122b

#endif  // GUFO_MODELS_ORNITH122B_CONFIG_HPP_
