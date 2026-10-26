// Shared helpers for DecodeKernels: type conversion, vector types, reductions.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace dk {

constexpr int kWarpSize = 32;

// ---------------------------------------------------------------------------
// fp16 / bf16 <-> fp32 conversion
// ---------------------------------------------------------------------------
template <typename T>
struct TypeConv;

template <>
struct TypeConv<__half> {
  static __device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
  static __device__ __forceinline__ __half from_float(float x) { return __float2half_rn(x); }
};

template <>
struct TypeConv<__nv_bfloat16> {
  static __device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
  static __device__ __forceinline__ __nv_bfloat16 from_float(float x) { return __float2bfloat16_rn(x); }
};

template <typename T>
__device__ __forceinline__ float to_float(T x) {
  return TypeConv<T>::to_float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x) {
  return TypeConv<T>::from_float(x);
}

// N contiguous elements loaded/stored as one aligned vector transaction
// (e.g. 8 x fp16 = one 16-byte load).
template <typename T, int N>
struct alignas(sizeof(T) * N) VecN {
  T val[N];
};

// ---------------------------------------------------------------------------
// Warp- and block-level reductions
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
  }
  return v;
}

// Block-wide sum. `scratch` must hold at least 32 floats of shared memory.
// Every thread of the block must call this; the result is returned to all.
__device__ __forceinline__ float block_reduce_sum(float v, float* scratch) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
  v = warp_reduce_sum(v);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  v = (threadIdx.x < num_warps) ? scratch[threadIdx.x] : 0.0f;
  if (warp == 0) v = warp_reduce_sum(v);
  if (threadIdx.x == 0) scratch[0] = v;
  __syncthreads();
  v = scratch[0];
  __syncthreads();  // scratch may be reused right after return
  return v;
}

// Block-wide max with the same contract as block_reduce_sum.
__device__ __forceinline__ float block_reduce_max(float v, float* scratch) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
  v = warp_reduce_max(v);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  v = (threadIdx.x < num_warps) ? scratch[threadIdx.x] : -INFINITY;
  if (warp == 0) v = warp_reduce_max(v);
  if (threadIdx.x == 0) scratch[0] = v;
  __syncthreads();
  v = scratch[0];
  __syncthreads();
  return v;
}

__host__ __device__ __forceinline__ int ceil_div(int a, int b) { return (a + b - 1) / b; }

inline bool is_aligned(const void* p, size_t bytes) {
  return (reinterpret_cast<uintptr_t>(p) % bytes) == 0;
}

}  // namespace dk
