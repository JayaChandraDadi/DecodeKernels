// Memory-bound decoder-layer kernels, vLLM-compatible semantics:
//
//   rms_norm(out, x, w, eps):           out = x / sqrt(mean(x^2) + eps) * w
//   fused_add_rms_norm(x, res, w, eps): res = x + res;  x = rms_norm(res) * w
//                                       (both in place, one read of each input)
//   silu_and_mul(out, x):               out = silu(x[:, :d]) * x[:, d:]
//
// One thread block per token row. Rows whose width is a multiple of 8 (and
// 16-byte aligned) use 16-byte vector loads/stores; other shapes fall back to
// a scalar path. All math is in fp32.
#pragma once

#include "common.cuh"

namespace dk {

constexpr int kVec = 8;  // 8 x 16-bit elements = 16 bytes

namespace detail {

template <typename T, bool VEC>
__global__ void rms_norm_kernel(T* __restrict__ out, const T* __restrict__ in, const T* __restrict__ weight, int H,
                                int64_t in_stride, int64_t out_stride, float eps) {
  __shared__ float scratch[32];
  const T* x = in + blockIdx.x * in_stride;
  T* y = out + blockIdx.x * out_stride;

  float ss = 0.0f;
  if constexpr (VEC) {
    using Vec = VecN<T, kVec>;
    for (int i = threadIdx.x * kVec; i < H; i += blockDim.x * kVec) {
      const Vec v = *reinterpret_cast<const Vec*>(x + i);
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const float f = to_float(v.val[j]);
        ss = fmaf(f, f, ss);
      }
    }
  } else {
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
      const float f = to_float(x[i]);
      ss = fmaf(f, f, ss);
    }
  }
  ss = block_reduce_sum(ss, scratch);
  const float inv_rms = rsqrtf(ss / static_cast<float>(H) + eps);

  if constexpr (VEC) {
    using Vec = VecN<T, kVec>;
    for (int i = threadIdx.x * kVec; i < H; i += blockDim.x * kVec) {
      const Vec v = *reinterpret_cast<const Vec*>(x + i);
      const Vec w = *reinterpret_cast<const Vec*>(weight + i);
      Vec o;
#pragma unroll
      for (int j = 0; j < kVec; ++j) o.val[j] = from_float<T>(to_float(v.val[j]) * inv_rms * to_float(w.val[j]));
      *reinterpret_cast<Vec*>(y + i) = o;
    }
  } else {
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
      y[i] = from_float<T>(to_float(x[i]) * inv_rms * to_float(weight[i]));
    }
  }
}

template <typename T, bool VEC>
__global__ void fused_add_rms_norm_kernel(T* __restrict__ input, T* __restrict__ residual,
                                          const T* __restrict__ weight, int H, int64_t in_stride, int64_t res_stride,
                                          float eps) {
  __shared__ float scratch[32];
  T* x = input + blockIdx.x * in_stride;
  T* r = residual + blockIdx.x * res_stride;

  // Pass 1: r = x + r (rounded to T, as the next layer will read it) and the
  // sum of squares of the rounded values.
  float ss = 0.0f;
  if constexpr (VEC) {
    using Vec = VecN<T, kVec>;
    for (int i = threadIdx.x * kVec; i < H; i += blockDim.x * kVec) {
      const Vec xv = *reinterpret_cast<const Vec*>(x + i);
      Vec rv = *reinterpret_cast<const Vec*>(r + i);
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        rv.val[j] = from_float<T>(to_float(xv.val[j]) + to_float(rv.val[j]));
        const float f = to_float(rv.val[j]);
        ss = fmaf(f, f, ss);
      }
      *reinterpret_cast<Vec*>(r + i) = rv;
    }
  } else {
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
      const T z = from_float<T>(to_float(x[i]) + to_float(r[i]));
      r[i] = z;
      const float f = to_float(z);
      ss = fmaf(f, f, ss);
    }
  }
  ss = block_reduce_sum(ss, scratch);
  const float inv_rms = rsqrtf(ss / static_cast<float>(H) + eps);

  // Pass 2: each thread re-reads exactly the elements it wrote above, so no
  // extra synchronisation is needed.
  if constexpr (VEC) {
    using Vec = VecN<T, kVec>;
    for (int i = threadIdx.x * kVec; i < H; i += blockDim.x * kVec) {
      const Vec rv = *reinterpret_cast<const Vec*>(r + i);
      const Vec w = *reinterpret_cast<const Vec*>(weight + i);
      Vec o;
#pragma unroll
      for (int j = 0; j < kVec; ++j) o.val[j] = from_float<T>(to_float(rv.val[j]) * inv_rms * to_float(w.val[j]));
      *reinterpret_cast<Vec*>(x + i) = o;
    }
  } else {
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
      x[i] = from_float<T>(to_float(r[i]) * inv_rms * to_float(weight[i]));
    }
  }
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

template <typename T, bool VEC>
__global__ void silu_and_mul_kernel(T* __restrict__ out, const T* __restrict__ in, int d, int64_t in_stride,
                                    int64_t out_stride) {
  const T* gate = in + blockIdx.x * in_stride;
  const T* up = gate + d;
  T* y = out + blockIdx.x * out_stride;
  if constexpr (VEC) {
    using Vec = VecN<T, kVec>;
    for (int i = threadIdx.x * kVec; i < d; i += blockDim.x * kVec) {
      const Vec g = *reinterpret_cast<const Vec*>(gate + i);
      const Vec u = *reinterpret_cast<const Vec*>(up + i);
      Vec o;
#pragma unroll
      for (int j = 0; j < kVec; ++j) o.val[j] = from_float<T>(silu(to_float(g.val[j])) * to_float(u.val[j]));
      *reinterpret_cast<Vec*>(y + i) = o;
    }
  } else {
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
      y[i] = from_float<T>(silu(to_float(gate[i])) * to_float(up[i]));
    }
  }
}

// Threads per row: enough to cover the row once with vectors, 32..1024.
inline int row_threads(int width, bool vec) {
  int t = vec ? (width + kVec - 1) / kVec : width;
  t = ((t + kWarpSize - 1) / kWarpSize) * kWarpSize;
  return t < kWarpSize ? kWarpSize : (t > 1024 ? 1024 : t);
}

}  // namespace detail

template <typename T>
cudaError_t launch_rms_norm(T* out, const T* in, const T* weight, int rows, int H, int64_t in_stride,
                            int64_t out_stride, float eps, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  const bool vec = H % kVec == 0 && in_stride % kVec == 0 && out_stride % kVec == 0 && is_aligned(in, 16) &&
                   is_aligned(out, 16) && is_aligned(weight, 16);
  const int threads = detail::row_threads(H, vec);
  if (vec) {
    detail::rms_norm_kernel<T, true><<<rows, threads, 0, stream>>>(out, in, weight, H, in_stride, out_stride, eps);
  } else {
    detail::rms_norm_kernel<T, false><<<rows, threads, 0, stream>>>(out, in, weight, H, in_stride, out_stride, eps);
  }
  return cudaGetLastError();
}

template <typename T>
cudaError_t launch_fused_add_rms_norm(T* input, T* residual, const T* weight, int rows, int H, int64_t in_stride,
                                      int64_t res_stride, float eps, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  const bool vec = H % kVec == 0 && in_stride % kVec == 0 && res_stride % kVec == 0 && is_aligned(input, 16) &&
                   is_aligned(residual, 16) && is_aligned(weight, 16);
  const int threads = detail::row_threads(H, vec);
  if (vec) {
    detail::fused_add_rms_norm_kernel<T, true>
        <<<rows, threads, 0, stream>>>(input, residual, weight, H, in_stride, res_stride, eps);
  } else {
    detail::fused_add_rms_norm_kernel<T, false>
        <<<rows, threads, 0, stream>>>(input, residual, weight, H, in_stride, res_stride, eps);
  }
  return cudaGetLastError();
}

template <typename T>
cudaError_t launch_silu_and_mul(T* out, const T* in, int rows, int d, int64_t in_stride, int64_t out_stride,
                                cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  const bool vec = d % kVec == 0 && in_stride % kVec == 0 && out_stride % kVec == 0 && is_aligned(in, 16) &&
                   is_aligned(out, 16);
  const int threads = detail::row_threads(d, vec);
  if (vec) {
    detail::silu_and_mul_kernel<T, true><<<rows, threads, 0, stream>>>(out, in, d, in_stride, out_stride);
  } else {
    detail::silu_and_mul_kernel<T, false><<<rows, threads, 0, stream>>>(out, in, d, in_stride, out_stride);
  }
  return cudaGetLastError();
}

}  // namespace dk
