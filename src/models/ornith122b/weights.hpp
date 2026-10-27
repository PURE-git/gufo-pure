// SPDX-License-Identifier: MIT
#ifndef GUFO_MODELS_ORNITH122B_WEIGHTS_HPP_
#define GUFO_MODELS_ORNITH122B_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ornith122b/config.hpp"

namespace gufo::models::ornith122b {

/// Non-owning view of one GGUF tensor. `rows` x `cols` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole matrices.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::uint64_t file_offset{0};  ///< Byte offset inside the owning shard.
  std::uint32_t shard{0};        ///< Mapped region index in the reader.
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    return cols * rows * experts;
  }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
  /// Base address of one stacked expert matrix.
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
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
  TensorRef ssm_alpha;
  TensorRef ssm_beta;
  TensorRef ssm_dt;
  TensorRef ssm_a;
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

struct ModelWeights {
  Config config;

  TensorRef token_embd;
  TensorRef output_norm;
  TensorRef output;

  std::vector<LayerWeights> layers;

  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::ornith122b

#endif  // GUFO_MODELS_ORNITH122B_WEIGHTS_HPP_
