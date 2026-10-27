// SPDX-License-Identifier: MIT
#include "src/models/ornith122b/config.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <utility>
#include <variant>

namespace gufo::models::ornith122b {
namespace {

constexpr std::string_view kArchitecture = "qwen35moe";

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  std::uint32_t U32(std::string_view key, bool required = true,
                    std::uint32_t fallback = 0) {
    const auto value = gguf.GetMetadataUint32(key);
    if (value.has_value()) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF integer " + std::string(key));
    }
    return fallback;
  }

  float F32(std::string_view key, float fallback) {
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value || !std::isfinite(*value))
      Fail("missing or invalid GGUF float " + std::string(key));
    return value.value_or(fallback);
  }

  std::vector<std::uint64_t> U64Array(std::string_view key,
                                      bool required = true) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      if (required)
        Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *s) {
        if (v < 0) {
          Fail("negative value in unsigned array " + std::string(key));
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("unexpected value kind for array " + std::string(key));
    }
    return out;
  }
};

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  const auto arch = gguf.GetMetadataString("general.architecture").value_or("");
  if (arch != kArchitecture && arch != "qwen35_moe") {
    r.Fail("GGUF architecture is not qwen35moe");
    return std::nullopt;
  }

  Config c;
  const std::string p = std::string(arch) + ".";
  auto key = [&](std::string_view suffix) {
    return p + std::string(suffix);
  };

  const auto total_blocks = r.U32(key("block_count"));
  c.nextn_layers = r.U32(key("nextn_predict_layers"), false, 0);
  if (c.nextn_layers >= total_blocks) {
    r.Fail("nextn_predict_layers must be below block_count");
    return std::nullopt;
  }
  c.num_layers = total_blocks - c.nextn_layers;
  c.hidden_size = r.U32(key("embedding_length"));
  c.context_length = r.U32(key("context_length"), false, 262144);
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"), 1e-6F);

  c.full_attention_interval = r.U32(key("full_attention_interval"), false, 4);
  c.num_heads = r.U32(key("attention.head_count"));
  c.num_kv_heads = r.U32(key("attention.head_count_kv"));
  c.head_dim = r.U32(key("attention.key_length"));
  c.rotary_dim = r.U32(key("rope.dimension_count"), false, 64);
  c.rope_theta = r.F32(key("rope.freq_base"), 10000000.0F);

  const auto sections = r.U64Array(key("rope.dimension_sections"), false);
  if (sections.size() == 4) {
    for (std::size_t i = 0; i < 4; ++i) {
      c.rope_sections[i] = static_cast<std::uint32_t>(sections[i]);
    }
  }

  c.ssm_conv_kernel = r.U32(key("ssm.conv_kernel"), false, 4);
  c.ssm_head_dim = r.U32(key("ssm.state_size"), false, 128);
  c.ssm_num_k_heads = r.U32(key("ssm.group_count"), false, 16);
  c.ssm_num_v_heads = r.U32(key("ssm.time_step_rank"), false, 64);
  c.ssm_inner_size = r.U32(key("ssm.inner_size"), false, 8192);

  c.num_experts = r.U32(key("expert_count"), false, 256);
  c.num_experts_used = r.U32(key("expert_used_count"), false, 8);
  c.expert_ff = r.U32(key("expert_feed_forward_length"), false, 1024);
  c.shared_expert_ff = r.U32(key("expert_shared_feed_forward_length"), false, 1024);

  if (!r.ok) {
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::ornith122b
