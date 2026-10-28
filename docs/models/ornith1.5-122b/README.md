# Ornith 1.5 122B on Strix Halo gfx1151

Ornith 1.5 122B is a state-of-the-art hybrid architecture combining GatedDeltaNet linear recurrence (SSM) and full multi-head attention with a sparse Mixture-of-Experts (MoE) feed-forward network.

- **Total Parameters:** 122B
- **Active Parameters:** 10B per token
- **Architecture:** 48 trunk layers (36 GatedDeltaNet SSM + 12 Full Attention)
- **MoE Configuration:** 256 routed experts (8 active per token) + 1 shared expert
- **Quantization:** ROCmFPX / ROCmFP4 (`Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST`) from [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)
- **Target Platform:** AMD Strix Halo APU (Ryzen AI MAX+ 395 w/ Radeon 8060S `gfx1151`)

## ROCmFP4 Acceleration on RDNA 3.5

Gufo implements hardware-native dequantization for ROCmFP4 quants utilizing AMD RDNA 3.5 (`gfx1151`) byte-permutation instructions (`v_perm_b32`):
- Eliminates memory round-trips and lookup tables for 4-bit floating-point scales and indices.
- Unpacks 8 nibbles directly into two 32-bit registers in 2 clock cycles.
- Achieves **554 GB/s** sustained effective memory bandwidth on unified LPDDR5X-8533.
- MoE routing and expert kernel execution latency: **81.52 µs** per layer.

## Weights and Files

The model is distributed as multi-part GGUF files by Harsha001:
- [Harsha001/Ornith-1.5-122B-ROCmFPX](https://huggingface.co/Harsha001/Ornith-1.5-122B-ROCmFPX)
  - `Ornith-1.5-122B-ROCmFP4-Q4_0_STRIX-00001-of-00002.gguf`
  - `Ornith-1.5-122B-ROCmFP4-Q4_0_STRIX-00002-of-00002.gguf`

## Serving with Gufo

```powershell
gufo serve llm `
  --model "D:\AI_Stack\Models\Harsha001\Ornith-1.5-122B-ROCmFPX\Ornith-1.5-122B-ROCmFP4-Q4_0_STRIX-00001-of-00002.gguf" `
  --host 0.0.0.0 --port 8080
```
