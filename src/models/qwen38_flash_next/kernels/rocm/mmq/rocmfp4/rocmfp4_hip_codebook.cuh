#pragma once

#if __has_include("rocmfp4_hip_scale.cuh")
#include "rocmfp4_hip_scale.cuh"
#elif __has_include("rocmfp4/rocmfp4_hip_scale.cuh")
#include "rocmfp4/rocmfp4_hip_scale.cuh"
#endif

#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>

#ifndef GGML_ROCMFP4_UNALIGNED_QS_DWORD_LOAD
#define GGML_ROCMFP4_UNALIGNED_QS_DWORD_LOAD 1
#endif

static __device__ __forceinline__ int rocmfp4_get_qs_i32(const void * x, const int & i32) {
#if (defined(__AMDGCN__) || defined(__HIP__) || defined(ENGINE_ENABLE_HIP) || defined(GGML_USE_HIP)) && GGML_ROCMFP4_UNALIGNED_QS_DWORD_LOAD
    return *((const int *) ((const uint8_t *) x + 4*i32));
#else
    const uint8_t * x8 = (const uint8_t *) x;

    int x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    return x32;
#endif
}

// Codebook table for ROCmFP4:
// 16 entries mapping 4-bit indices to signed 8-bit integers:
// 0..7: 0, 1, 2, 3, 4, 6, 8, 10
// 8..15: 0, -1, -2, -3, -4, -6, -8, -10
static __device__ __constant__ const int8_t kvalues_rocmfp4_hip[16] = {
    0, 1, 2, 3, 4, 6, 8, 10,
    0, -1, -2, -3, -4, -6, -8, -10
};

// Vectorized codebook dequantization for AMD GCN / CDNA (v_perm_b32):
// Decodes 8 packed 4-bit nibbles (one uint32_t) into two int32_t values (int2),
// where each int32_t holds 4 unpacked signed int8 codebook values.
static __device__ __forceinline__ int2 rocmfp4_get_int_from_codebook_16(const int & q4, const int8_t * fallback_table) {
#if defined(__AMDGCN__) || defined(__HIP__) || defined(ENGINE_ENABLE_HIP) || defined(GGML_USE_HIP)
    constexpr uint32_t values0 = 0x03020100u;
    constexpr uint32_t values1 = 0x0a080604u;
    constexpr uint32_t values2 = 0xfdfeff00u;
    constexpr uint32_t values3 = 0xf6f8fafcu;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;

    const uint32_t v_even_low  = __builtin_amdgcn_perm(values1, values0, q_even & 0x07070707u);
    const uint32_t v_odd_low   = __builtin_amdgcn_perm(values1, values0, q_odd  & 0x07070707u);
    const uint32_t v_even_high = __builtin_amdgcn_perm(values3, values2, q_even & 0x07070707u);
    const uint32_t v_odd_high  = __builtin_amdgcn_perm(values3, values2, q_odd  & 0x07070707u);

    const uint32_t mask_even = 0x03020100u | ((q_even & 0x08080808u) >> 1);
    const uint32_t mask_odd  = 0x03020100u | ((q_odd  & 0x08080808u) >> 1);

    return make_int2(
        __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even),
        __builtin_amdgcn_perm(v_odd_high,  v_odd_low,  mask_odd));
#else
    const int8_t * t = fallback_table ? fallback_table : kvalues_rocmfp4_hip;
    const uint8_t * p = (const uint8_t *)&q4;
    int8_t v_even[4] = { t[p[0] & 0x0F], t[p[1] & 0x0F], t[p[2] & 0x0F], t[p[3] & 0x0F] };
    int8_t v_odd[4]  = { t[p[0] >> 4],   t[p[1] >> 4],   t[p[2] >> 4],   t[p[3] >> 4] };
    int r_even, r_odd;
    memcpy(&r_even, v_even, 4);
    memcpy(&r_odd,  v_odd,  4);
    return make_int2(r_even, r_odd);
#endif
}

// Variant for call sites that already selected either the low or high nibble
// stream and only need one DP4A operand. This avoids the extra odd/even table
// expansion work in ROCmFP4 FlashAttention K/V decode.
static __device__ __forceinline__ int rocmfp4_get_low_int_from_codebook_16(const int & q4, const int8_t * fallback_table) {
#if defined(__AMDGCN__) || defined(__HIP__) || defined(ENGINE_ENABLE_HIP) || defined(GGML_USE_HIP)
    constexpr uint32_t values0 = 0x03020100u;
    constexpr uint32_t values1 = 0x0a080604u;
    constexpr uint32_t values2 = 0xfdfeff00u;
    constexpr uint32_t values3 = 0xf6f8fafcu;

    const uint32_t q = q4;

    const uint32_t v_low  = __builtin_amdgcn_perm(values1, values0, q & 0x07070707u);
    const uint32_t v_high = __builtin_amdgcn_perm(values3, values2, q & 0x07070707u);
    const uint32_t mask   = 0x03020100u | ((q & 0x08080808u) >> 1);

    return __builtin_amdgcn_perm(v_high, v_low, mask);
#else
    const int8_t * t = fallback_table ? fallback_table : kvalues_rocmfp4_hip;
    const uint8_t * p = (const uint8_t *)&q4;
    int8_t v[4] = { t[p[0] & 0x0F], t[p[1] & 0x0F], t[p[2] & 0x0F], t[p[3] & 0x0F] };
    int r;
    memcpy(&r, v, 4);
    return r;
#endif
}
