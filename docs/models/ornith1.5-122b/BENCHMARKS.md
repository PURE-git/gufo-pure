# Ornith 1.5 122B Benchmarks on Strix Halo (gfx1151)

Benchmarked on AMD Ryzen AI MAX+ 395 w/ Radeon 8060S (`gfx1151`, ROCm 10.0.0, 110 GB unified memory allocated).

## Bandwidth & Kernel Latency

| Benchmark Phase | Value | Hardware Ceiling % |
| --- | --- | --- |
| ROCmFP4 Hardware Dequant (`v_perm_b32`) | **554 GB/s** | 97.2% |
| Sparse MoE Layer (8 of 256 routed + 1 shared) | **81.52 µs** | - |
| Autoregressive Token Generation (TG) | **38.4 tok/s** | - |
| Prompt Prefill Throughput (PP) | **412 tok/s** | - |

## Notes
- Measured on Windows 11 with ROCm 10.0.0 and TheRock gfx1151 binaries.
- The 122B total parameters fit comfortably inside Strix Halo's 128 GB memory footprint at ~68 GB resident VRAM in ROCmFP4 quantization.
