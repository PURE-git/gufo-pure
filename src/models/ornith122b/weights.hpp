// SPDX-License-Identifier: MIT
#ifndef GUFO_MODELS_ORNITH122B_WEIGHTS_HPP_
#define GUFO_MODELS_ORNITH122B_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ornith122b/config.hpp"

namespace gufo::models::ornith122b {

struct TensorRef {
  const void* data{nullptr};
  std::uint64_t cols{0};
  std::uint64_t rows{0};
  std::uint64_t experts{1};
  core::GgmlType type{core::GgmlType::kF32};
  std::size_t size_bytes{0};
};

struct LayerWeights {
  bool is_full_attention{false};

  // Pre-attention normalization
  TensorRef attn_norm;

  // Full self-attention (every 4th layer: 3, 7, 11, ...)
  TensorRef attn_q;
  TensorRef attn_k;
  TensorRef attn_v;
  TensorRef attn_q_norm;
  TensorRef attn_k_norm;
  TensorRef attn_output;

  // Linear attention (GatedDeltaNet SSM)
  TensorRef attn_qkv;
  TensorRef attn_gate;
  TensorRef ssm_conv1d;
  TensorRef ssm_a;
  TensorRef ssm_alpha;
  TensorRef ssm_beta;
  TensorRef ssm_dt;
  TensorRef ssm_norm;
  TensorRef ssm_out;

  // Post-attention normalization
  TensorRef post_attention_norm;

  // Mixture of Experts (256 routed experts + 1 shared expert)
  TensorRef router;           ///< ffn_gate_inp.weight
  TensorRef ffn_gate_exps;    ///< ffn_gate_exps.weight (ROCmFP4)
  TensorRef ffn_up_exps;      ///< ffn_up_exps.weight (ROCmFP4)
  TensorRef ffn_down_exps;    ///< ffn_down_exps.weight (ROCmFP4)

  TensorRef shexp_gate_inp;   ///< ffn_gate_inp_shexp.weight
  TensorRef shexp_gate;       ///< ffn_gate_shexp.weight (ROCmFP4)
  TensorRef shexp_up;         ///< ffn_up_shexp.weight (ROCmFP4)
  TensorRef shexp_down;       ///< ffn_down_shexp.weight (ROCmFP4)
};

struct Weights {
  Config config;

  TensorRef token_embd;
  TensorRef output_norm;
  TensorRef output;

  std::vector<LayerWeights> layers;

  [[nodiscard]] static std::optional<Weights> Load(
      const core::GgufReader& gguf, const Config& config,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::ornith122b

#endif  // GUFO_MODELS_ORNITH122B_WEIGHTS_HPP_
