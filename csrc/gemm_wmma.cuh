// Tensor-core GEMM for linear layers: C[M,N] = A[M,K] * B[N,K]^T (+ bias[N]).
//
// A holds activations and B holds weights in PyTorch's nn.Linear layout
// ([out_features, in_features], K contiguous), so this is the "TN" GEMM that
// every projection in an LLM runs.
//
// Design:
// * Block tile BM x BN x BK with a WARPS_M x WARPS_N grid of warps; each warp
//   owns a (BM/WARPS_M) x (BN/WARPS_N) sub-tile made of 16x16x16 WMMA
//   fragments accumulated in fp32 on tensor cores.
// * Two shared-memory stages filled with cp.async (16-byte copies, zero-fill
//   for out-of-range rows/cols): the next K-tile streams in while the current
//   one is multiplied.
// * Shared-memory rows are padded by 8 elements to break bank conflicts on
//   the fragment loads.
// * Two configurations: 128x128 tiles (8 warps) for prefill-sized M, and a
//   32x128 "skinny" tile (4 warps) for decode, where M is the batch size and
//   a 128-row tile would waste most of its work on padding.
// * Epilogue stages each fp32 fragment through shared memory, adds the
//   optional bias and converts to fp16/bf16 with bounds checks.
//
// Requirements: K, lda and ldb multiples of 8; A and B 16-byte aligned.
// Needs sm_80+ (cp.async, bf16 WMMA).
#pragma once

#include <mma.h>

#include "common.cuh"

namespace dk {

namespace detail {

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  const int src_bytes = pred ? 16 : 0;  // 0 => fill the 16 bytes with zeros
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem_src), "r"(src_bytes));
#else
  if (pred) {
    *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
  } else {
    *reinterpret_cast<uint4*>(smem_dst) = make_uint4(0, 0, 0, 0);
  }
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

constexpr int kBK = 32;
constexpr int kPad = 8;
constexpr int kLds = kBK + kPad;  // shared-memory row stride in elements

template <int BM, int BN>
__host__ __device__ constexpr size_t gemm_smem_bytes() {
  return 2ull * (BM + BN) * kLds * 2ull;  // 2 stages, 2-byte elements
}

template <typename T, int BM, int BN, int WARPS_M, int WARPS_N>
__global__ void __launch_bounds__(WARPS_M* WARPS_N* kWarpSize)
    gemm_tn_wmma_kernel(const T* __restrict__ A, const T* __restrict__ B, const T* __restrict__ bias,
                        T* __restrict__ C, int M, int N, int K, int lda, int ldb, int ldc) {
  using namespace nvcuda;
  constexpr int kThreads = WARPS_M * WARPS_N * kWarpSize;
  constexpr int WM = BM / WARPS_M;
  constexpr int WN = BN / WARPS_N;
  constexpr int FM = WM / 16;
  constexpr int FN = WN / 16;
  static_assert(WM % 16 == 0 && WN % 16 == 0, "warp tile must be a multiple of 16");
  static_assert(WARPS_M * WARPS_N * 256 * sizeof(float) <= gemm_smem_bytes<BM, BN>(),
                "epilogue staging (one 16x16 fp32 tile per warp) must fit in the main-loop shared memory");

  extern __shared__ __align__(128) unsigned char smem_raw[];
  T* As = reinterpret_cast<T*>(smem_raw);  // [2][BM][kLds]
  T* Bs = As + 2 * BM * kLds;              // [2][BN][kLds]

  const int tid = threadIdx.x;
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int warp_m = warp / WARPS_N;
  const int warp_n = warp % WARPS_N;
  const int block_m = blockIdx.y * BM;
  const int block_n = blockIdx.x * BN;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i) {
#pragma unroll
    for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
  }

  // Copy one BMxBK tile of A and one BNxBK tile of B into `stage`, 16 bytes
  // (8 elements) per cp.async.
  auto load_tile = [&](int stage, int k0) {
    constexpr int kChunksPerRow = kBK / 8;
    T* as = As + stage * BM * kLds;
    T* bs = Bs + stage * BN * kLds;
#pragma unroll
    for (int c = tid; c < BM * kChunksPerRow; c += kThreads) {
      const int row = c / kChunksPerRow;
      const int col = (c % kChunksPerRow) * 8;
      const int gr = block_m + row;
      const int gc = k0 + col;
      const bool pred = gr < M && gc < K;
      cp_async_16(as + row * kLds + col, pred ? A + static_cast<int64_t>(gr) * lda + gc : A, pred);
    }
#pragma unroll
    for (int c = tid; c < BN * kChunksPerRow; c += kThreads) {
      const int row = c / kChunksPerRow;
      const int col = (c % kChunksPerRow) * 8;
      const int gr = block_n + row;
      const int gc = k0 + col;
      const bool pred = gr < N && gc < K;
      cp_async_16(bs + row * kLds + col, pred ? B + static_cast<int64_t>(gr) * ldb + gc : B, pred);
    }
  };

  const int k_tiles = (K + kBK - 1) / kBK;
  load_tile(0, 0);
  cp_async_commit();

  for (int kt = 0; kt < k_tiles; ++kt) {
    const int stage = kt & 1;
    if (kt + 1 < k_tiles) {
      load_tile(stage ^ 1, (kt + 1) * kBK);  // prefetch next tile
      cp_async_commit();
      cp_async_wait<1>();  // current tile has landed, next one may be in flight
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    const T* as = As + stage * BM * kLds;
    const T* bs = Bs + stage * BN * kLds;
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag[FM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b_frag[FN];
#pragma unroll
      for (int i = 0; i < FM; ++i) wmma::load_matrix_sync(a_frag[i], as + (warp_m * WM + i * 16) * kLds + kk, kLds);
#pragma unroll
      for (int j = 0; j < FN; ++j) wmma::load_matrix_sync(b_frag[j], bs + (warp_n * WN + j * 16) * kLds + kk, kLds);
#pragma unroll
      for (int i = 0; i < FM; ++i) {
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
      }
    }
    __syncthreads();  // stage is free to be overwritten by the next prefetch
  }

  // Epilogue: stage each 16x16 fp32 fragment in this warp's slice of shared
  // memory, then write it out with bias, conversion and bounds checks.
  float* stage_f = reinterpret_cast<float*>(smem_raw) + warp * 256;
#pragma unroll
  for (int i = 0; i < FM; ++i) {
#pragma unroll
    for (int j = 0; j < FN; ++j) {
      wmma::store_matrix_sync(stage_f, acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      const int row0 = block_m + warp_m * WM + i * 16;
      const int col0 = block_n + warp_n * WN + j * 16;
#pragma unroll
      for (int e = lane; e < 256; e += kWarpSize) {
        const int r = e / 16;
        const int c = e % 16;
        const int gr = row0 + r;
        const int gc = col0 + c;
        if (gr < M && gc < N) {
          float v = stage_f[e];
          if (bias != nullptr) v += to_float(bias[gc]);
          C[static_cast<int64_t>(gr) * ldc + gc] = from_float<T>(v);
        }
      }
      __syncwarp();
    }
  }
}

template <typename T, int BM, int BN, int WARPS_M, int WARPS_N>
cudaError_t launch_gemm(const T* A, const T* B, const T* bias, T* C, int M, int N, int K, int lda, int ldb, int ldc,
                        cudaStream_t stream) {
  constexpr size_t smem = gemm_smem_bytes<BM, BN>();
  auto kernel = gemm_tn_wmma_kernel<T, BM, BN, WARPS_M, WARPS_N>;
  if (smem > 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    if (e != cudaSuccess) return e;
  }
  const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  const dim3 block(WARPS_M * WARPS_N * kWarpSize);
  kernel<<<grid, block, smem, stream>>>(A, B, bias, C, M, N, K, lda, ldb, ldc);
  return cudaGetLastError();
}

}  // namespace detail

// Picks the tile configuration from M (tokens in the batch).
template <typename T>
cudaError_t launch_gemm_tn(const T* A, const T* B, const T* bias, T* C, int M, int N, int K, int lda, int ldb, int ldc,
                           cudaStream_t stream) {
  if (M == 0 || N == 0) return cudaSuccess;
  if (K % 8 != 0 || lda % 8 != 0 || ldb % 8 != 0) return cudaErrorInvalidValue;
  if (!is_aligned(A, 16) || !is_aligned(B, 16)) return cudaErrorMisalignedAddress;
  if (M <= 64) {
    return detail::launch_gemm<T, 32, 128, 1, 4>(A, B, bias, C, M, N, K, lda, ldb, ldc, stream);
  }
  return detail::launch_gemm<T, 128, 128, 4, 2>(A, B, bias, C, M, N, K, lda, ldb, ldc, stream);
}

}  // namespace dk
