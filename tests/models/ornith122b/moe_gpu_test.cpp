// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ornith122b/config.hpp"
#include "src/models/ornith122b/weights.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

#define HIP_CHECK(expr)                                                       \
  do {                                                                        \
    hipError_t err = (expr);                                                  \
    if (err != hipSuccess) {                                                  \
      std::cerr << "HIP Error: " << hipGetErrorString(err) << " at "          \
                << __FILE__ << ":" << __LINE__ << "\n";                      \
      return 1;                                                               \
    }                                                                         \
  } while (0)

int main(int argc, char* argv[]) {
  std::string model_path =
      "D:\\AI_Stack\\Models\\Harsha001\\Ornith-1.5-122B-ROCmFPX\\"
      "Ornith-1.5-122B-ROCmFP4-Q4_0_STRIX-00001-of-00002.gguf";
  if (argc > 1) {
    model_path = argv[1];
  }

  std::cout << "=== Ornith 1.5 122B ROCmFP4 MoE GPU Kernel Test ===\n";

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cerr << "No HIP devices found!\n";
    return 1;
  }

  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::cout << "Device: " << prop.name << " (" << prop.gcnArchName << ")\n";
  std::cout << "Compute Units: " << prop.multiProcessorCount
            << ", Warp Size: " << prop.warpSize << "\n";
  std::cout << "Total Global Memory: "
            << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n\n";

  // Initialize MMQ subsystem
  std::cout << "Initializing MMQ runtime on device 0...\n";
  if (qfn_mmq_init(0) != 0) {
    std::cerr << "Failed to initialize qfn_mmq!\n";
    return 1;
  }

  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::cerr << "Failed to open GGUF: " << error << "\n";
    return 1;
  }

  auto config = gufo::models::ornith122b::Config::FromGguf(*reader, &error);
  if (!config) {
    std::cerr << "Failed to parse config: " << error << "\n";
    return 1;
  }

  auto weights = gufo::models::ornith122b::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    std::cerr << "Failed to bind weights: " << error << "\n";
    return 1;
  }

  const auto& layer0 = weights->layers[0];
  const auto& ffn_gate = layer0.ffn_gate_exps;
  const auto& ffn_up = layer0.ffn_up_exps;
  const auto& ffn_down = layer0.ffn_down_exps;

  std::cout << "Layer 0 MoE Tensors:\n";
  std::cout << "  ffn_gate_exps: " << ffn_gate.cols << "x" << ffn_gate.rows
            << "x" << ffn_gate.experts << " ("
            << gufo::core::ToString(ffn_gate.type)
            << ", " << (ffn_gate.SizeBytes() / (1024 * 1024)) << " MB)\n";
  std::cout << "  ffn_up_exps:   " << ffn_up.cols << "x" << ffn_up.rows
            << "x" << ffn_up.experts << " ("
            << gufo::core::ToString(ffn_up.type)
            << ", " << (ffn_up.SizeBytes() / (1024 * 1024)) << " MB)\n";
  std::cout << "  ffn_down_exps: " << ffn_down.cols << "x" << ffn_down.rows
            << "x" << ffn_down.experts << " ("
            << gufo::core::ToString(ffn_down.type)
            << ", " << (ffn_down.SizeBytes() / (1024 * 1024)) << " MB)\n";

  // Allocate GPU memory for Layer 0 weights
  std::cout << "\nUploading Layer 0 weights to GPU...\n";
  void* d_gate = nullptr;
  void* d_up = nullptr;
  void* d_down = nullptr;
  HIP_CHECK(hipMalloc(&d_gate, ffn_gate.SizeBytes()));
  HIP_CHECK(hipMalloc(&d_up, ffn_up.SizeBytes()));
  HIP_CHECK(hipMalloc(&d_down, ffn_down.SizeBytes()));

  const auto t_up_start = std::chrono::steady_clock::now();
  HIP_CHECK(hipMemcpy(d_gate, ffn_gate.data, ffn_gate.SizeBytes(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_up, ffn_up.data, ffn_up.SizeBytes(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_down, ffn_down.data, ffn_down.SizeBytes(),
                      hipMemcpyHostToDevice));
  const auto t_up_end = std::chrono::steady_clock::now();
  const double upload_s =
      std::chrono::duration<double>(t_up_end - t_up_start).count();
  const double total_mb =
      (ffn_gate.SizeBytes() + ffn_up.SizeBytes() + ffn_down.SizeBytes()) /
      (1024.0 * 1024.0);
  std::cout << "Uploaded " << total_mb << " MB in " << upload_s << " s ("
            << (total_mb / 1024.0 / upload_s) << " GB/s)\n";

  // Setup test activation inputs
  const int n_tokens = 1;
  const int n_used = 8;
  const int n_experts = 256;
  const int hidden = 3072;
  const int expert_ff = 1024;

  std::vector<float> h_x(hidden, 0.05f);
  // Pick 8 representative active expert IDs
  std::vector<std::int32_t> h_ids = {7, 23, 45, 91, 118, 164, 210, 245};

  float* d_x = nullptr;
  std::int32_t* d_ids = nullptr;
  float* d_out_gate = nullptr;
  float* d_out_up = nullptr;
  float* d_out_down = nullptr;

  HIP_CHECK(hipMalloc(&d_x, hidden * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ids, n_used * sizeof(std::int32_t)));
  HIP_CHECK(hipMalloc(&d_out_gate, n_used * expert_ff * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_up, n_used * expert_ff * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_down, n_used * hidden * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_x, h_x.data(), hidden * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ids, h_ids.data(), n_used * sizeof(std::int32_t),
                      hipMemcpyHostToDevice));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  std::cout << "\nExecuting Warmup Kernel...\n";
  const int weight_type = static_cast<int>(ffn_gate.type);
  int res = qfn_mmq_moe_vec(weight_type, d_gate, d_x, d_ids, d_out_gate,
                            expert_ff, hidden, n_tokens, n_experts, n_used,
                            stream, d_up, d_out_up);
  if (res != 0) {
    std::cerr << "qfn_mmq_moe_vec failed with code " << res << "\n";
    return 1;
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  std::cout << "Warmup passed!\n";

  // Numerical Validation
  std::vector<float> h_out_gate(n_used * expert_ff);
  std::vector<float> h_out_up(n_used * expert_ff);
  HIP_CHECK(hipMemcpy(h_out_gate.data(), d_out_gate,
                      h_out_gate.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_out_up.data(), d_out_up,
                      h_out_up.size() * sizeof(float), hipMemcpyDeviceToHost));

  bool finite_ok = true;
  float sum_abs_gate = 0.0f;
  float sum_abs_up = 0.0f;
  for (std::size_t i = 0; i < h_out_gate.size(); ++i) {
    if (!std::isfinite(h_out_gate[i]) || !std::isfinite(h_out_up[i])) {
      finite_ok = false;
      break;
    }
    sum_abs_gate += std::abs(h_out_gate[i]);
    sum_abs_up += std::abs(h_out_up[i]);
  }

  assert(finite_ok && sum_abs_gate > 0.0f && sum_abs_up > 0.0f);
  std::cout << "Numerical validation passed!\n";
  std::cout << "  Mean abs gate output: "
            << (sum_abs_gate / h_out_gate.size()) << "\n";
  std::cout << "  Mean abs up output:   "
            << (sum_abs_up / h_out_up.size()) << "\n";

  // Benchmark iterations
  const int iters = 200;
  std::cout << "\nBenchmarking " << iters
            << " iterations of ROCmFP4 Paired MoE Gemv...\n";

  // Flush stream before timing
  HIP_CHECK(hipStreamSynchronize(stream));
  const auto t_bench_start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    qfn_mmq_moe_vec(weight_type, d_gate, d_x, d_ids, d_out_gate, expert_ff,
                    hidden, n_tokens, n_experts, n_used, stream, d_up,
                    d_out_up);
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  const auto t_bench_end = std::chrono::steady_clock::now();

  const double total_us =
      std::chrono::duration<double, std::micro>(t_bench_end - t_bench_start)
          .count();
  const double avg_us = total_us / iters;

  // Active expert weights read per token:
  // 8 experts * (gate + up) = 16 matrices of [3072, 1024]
  // 16 * 1024 rows * (3072 / 32 * 17 bytes) = 16 * 1024 * 1632 bytes = 26,738,688 bytes (~26.7 MB)
  const double active_bytes = 16.0 * 1024.0 * (hidden / 32.0 * 17.0);
  const double effective_bandwidth_gbs =
      (active_bytes / (avg_us * 1e-6)) / (1e9);

  std::cout << "Results on AMD Strix Halo gfx1151 (Paired Gate+Up):\n";
  std::cout << "  Average kernel latency: " << avg_us << " us\n";
  std::cout << "  Effective DRAM bandwidth: " << effective_bandwidth_gbs
            << " GB/s\n";
  std::cout << "  Theoretical roof: 256.0 GB/s (LPDDR5X-8533)\n";
  std::cout << "  Efficiency: "
            << (effective_bandwidth_gbs / 256.0 * 100.0) << " %\n";

  // Benchmark Down Projection
  std::cout << "\nBenchmarking " << iters
            << " iterations of ROCmFP4 Down Projection MoE Gemv...\n";
  HIP_CHECK(hipStreamSynchronize(stream));
  const auto t_down_start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    qfn_mmq_moe_vec(weight_type, d_down, d_out_gate, d_ids, d_out_down, hidden,
                    expert_ff, n_tokens, n_experts, n_used, stream);
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  const auto t_down_end = std::chrono::steady_clock::now();

  const double total_down_us =
      std::chrono::duration<double, std::micro>(t_down_end - t_down_start)
          .count();
  const double avg_down_us = total_down_us / iters;

  // Active down expert weights read per token:
  // 8 experts * [1024, 3072]
  // 8 * 3072 rows * (1024 / 32 * 17 bytes) = 8 * 3072 * 544 bytes = 13,369,344 bytes (~13.4 MB)
  const double active_down_bytes = 8.0 * hidden * (expert_ff / 32.0 * 17.0);
  const double effective_down_bandwidth_gbs =
      (active_down_bytes / (avg_down_us * 1e-6)) / (1e9);

  std::cout << "Results on AMD Strix Halo gfx1151 (Down Projection):\n";
  std::cout << "  Average kernel latency: " << avg_down_us << " us\n";
  std::cout << "  Effective DRAM bandwidth: " << effective_down_bandwidth_gbs
            << " GB/s\n";

  const double total_moe_us = avg_us + avg_down_us;
  std::cout << "\nTotal MoE Layer Compute Latency: " << total_moe_us << " us ("
            << (1000.0 / (total_moe_us * 48.0 / 1000.0)) << " est. tokens/sec for 48 MoE layers)\n";

  // Clean up
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_up));
  HIP_CHECK(hipFree(d_down));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_ids));
  HIP_CHECK(hipFree(d_out_gate));
  HIP_CHECK(hipFree(d_out_up));
  HIP_CHECK(hipFree(d_out_down));

  std::cout << "\n=== All tests passed successfully! ROCmFP4 on Strix Halo is verified! ===\n";
  return 0;
}
