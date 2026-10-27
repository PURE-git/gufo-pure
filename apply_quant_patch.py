with open('src/core/quant/ggml_dequant.hpp', 'r', encoding='utf-8') as f:
    content = f.read()

target1 = """struct block_iq3_s {
  std::uint16_t d;
  std::uint8_t qs[64];
  std::uint8_t qh[8];
  std::uint8_t signs[32];
  std::uint8_t scales[4];
};
#pragma pack(pop)"""

replacement1 = """struct block_iq3_s {
  std::uint16_t d;
  std::uint8_t qs[64];
  std::uint8_t qh[8];
  std::uint8_t signs[32];
  std::uint8_t scales[4];
};

struct block_rocmfp4 {
  std::uint16_t d;
  std::uint8_t qs[16];
};

struct block_rocmfp4_fast {
  std::uint8_t d;
  std::uint8_t qs[16];
};
#pragma pack(pop)"""

target2 = 'static_assert(sizeof(block_iq3_s) == 110, "block_iq3_s must be 110 bytes");'

replacement2 = """static_assert(sizeof(block_iq3_s) == 110, "block_iq3_s must be 110 bytes");
static_assert(sizeof(block_rocmfp4) == 18, "block_rocmfp4 must be 18 bytes");
static_assert(sizeof(block_rocmfp4_fast) == 17, "block_rocmfp4_fast must be 17 bytes");"""

target3 = """    case core::GgmlType::kQ8_0:
    case core::GgmlType::kIQ4_NL:
      return 32;"""

replacement3 = """    case core::GgmlType::kQ8_0:
    case core::GgmlType::kIQ4_NL:
    case core::GgmlType::kQ4_0_ROCMFP4:
    case core::GgmlType::kQ4_0_ROCMFP4_FAST:
      return 32;"""

assert target1 in content, "target1 not found"
assert target2 in content, "target2 not found"
assert target3 in content, "target3 not found"

content = content.replace(target1, replacement1).replace(target2, replacement2).replace(target3, replacement3)

with open('src/core/quant/ggml_dequant.hpp', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)
print("Successfully patched src/core/quant/ggml_dequant.hpp")

with open('src/core/quant/ggml_dequant.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

target = """    case core::GgmlType::kIQ3_S:
      block_bytes = sizeof(block_iq3_s);
      break;"""

replacement = """    case core::GgmlType::kIQ3_S:
      block_bytes = sizeof(block_iq3_s);
      break;
    case core::GgmlType::kQ4_0_ROCMFP4:
      block_bytes = sizeof(block_rocmfp4);
      break;
    case core::GgmlType::kQ4_0_ROCMFP4_FAST:
      block_bytes = sizeof(block_rocmfp4_fast);
      break;"""

assert target in content, "target not found"
content = content.replace(target, replacement)

with open('src/core/quant/ggml_dequant.cpp', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)
print("Successfully patched src/core/quant/ggml_dequant.cpp")
