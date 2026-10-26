// Paged-attention decode with flash-decoding (split-K over the sequence).
//
// One query token per sequence attends to its keys/values, which live in a
// paged KV cache addressed through a block table (vLLM / FlashInfer style).
//
//   q            [num_seqs, num_q_heads, head_dim]
//   k/v cache    [num_blocks, block_size, num_kv_heads, head_dim]
//                (any strides for the first three dims; head_dim contiguous,
//                 so vLLM's interleaved K|V layout and FlashInfer NHD/HND
//                 views all work without a copy)
//   block_tables [num_seqs, max_blocks_per_seq] int32
//   seq_lens     [num_seqs] int32   (tokens already in the cache, incl. the
//                                    current one)
//
// Design:
// * GQA-aware: a thread block owns one KV head and computes every query head
//   in its group, so each K/V row is read from HBM once per group instead of
//   once per query head.
// * Short sequences (max_seq_len <= kPartitionSize) run a single pass that
//   writes the output directly.
// * Long sequences are split into kPartitionSize-token partitions that run in
//   parallel (flash-decoding). Each partition stores its normalised partial
//   output plus (max logit, exp-sum); a reduce kernel merges partitions with
//   the log-sum-exp rule, so small batches still fill every SM.
// * Each warp reads one full K/V row per step with vectorised, coalesced
//   loads (head_dim / 32 elements per lane) and reduces the dot product with
//   warp shuffles.
#pragma once

#include "common.cuh"

namespace dk {

constexpr int kPartitionSize = 512;  // tokens per split-K partition
constexpr int kPANumThreads = 128;
constexpr int kPANumWarps = kPANumThreads / kWarpSize;
constexpr int kMaxGroup = 8;  // max query heads per KV head (GQA ratio)

struct PagedAttnParams {
  void* out;
  const void* q;
  const void* k_cache;
  const void* v_cache;
  const int* block_tables;
  const int* seq_lens;
  // split-K workspace (only used when max_seq_len > kPartitionSize)
  float* tmp_out;     // [num_seqs, num_q_heads, max_partitions, head_dim]
  float* max_logits;  // [num_seqs, num_q_heads, max_partitions]
  float* exp_sums;    // [num_seqs, num_q_heads, max_partitions]
  // strides in elements
  int64_t q_stride_seq, q_stride_head;
  int64_t out_stride_seq, out_stride_head;
  int64_t k_stride_block, k_stride_token, k_stride_head;
  int64_t v_stride_block, v_stride_token, v_stride_head;
  int64_t bt_stride;
  int num_seqs, num_q_heads, num_kv_heads, block_size, max_partitions;
  float scale;
};

namespace detail {

template <typename T, int HEAD_DIM>
__device__ __forceinline__ const T* kv_row_ptr(const void* cache, const PagedAttnParams& p, int64_t stride_block,
                                               int64_t stride_token, int64_t stride_head, int seq, int kv_head,
                                               int token) {
  const int logical_block = token / p.block_size;
  const int offset = token - logical_block * p.block_size;
  const int physical_block = p.block_tables[static_cast<int64_t>(seq) * p.bt_stride + logical_block];
  return static_cast<const T*>(cache) + physical_block * stride_block + offset * stride_token +
         kv_head * stride_head;
}

// Phase 1: grid = (num_kv_heads, num_seqs, SPLIT ? max_partitions : 1)
template <typename T, int HEAD_DIM, bool SPLIT>
__global__ void __launch_bounds__(kPANumThreads) paged_attention_kernel(const PagedAttnParams p) {
  static_assert(HEAD_DIM % kWarpSize == 0, "head_dim must be a multiple of 32");
  constexpr int V = HEAD_DIM / kWarpSize;  // elements per lane per K/V row
  using Vec = VecN<T, V>;

  const int kv_head = blockIdx.x;
  const int seq = blockIdx.y;
  const int part = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid % kWarpSize;
  const int warp = tid / kWarpSize;
  const int group = p.num_q_heads / p.num_kv_heads;

  const int seq_len = p.seq_lens[seq];
  const int start = SPLIT ? part * kPartitionSize : 0;

  if (start >= seq_len) {
    // Nothing to attend to. In the single-pass path this only happens for
    // empty (padding) sequences: write zeros. Split partitions past the end
    // are skipped by the reduce kernel.
    if (!SPLIT) {
      T* out = static_cast<T*>(p.out);
      for (int idx = tid; idx < group * HEAD_DIM; idx += kPANumThreads) {
        const int g = idx / HEAD_DIM;
        const int d = idx - g * HEAD_DIM;
        const int q_head = kv_head * group + g;
        out[seq * p.out_stride_seq + q_head * p.out_stride_head + d] = from_float<T>(0.0f);
      }
    }
    return;
  }
  // The single-pass path is only launched when every sequence fits in one
  // partition; clamping keeps memory access in bounds even if a caller
  // passes a too-small max_seq_len.
  const int end = min(start + kPartitionSize, seq_len);
  const int n = end - start;

  extern __shared__ float smem[];
  float* q_s = smem;                       // [group][HEAD_DIM], pre-scaled
  float* logits = q_s + group * HEAD_DIM;  // [group][kPartitionSize]
  float* red = logits;                     // reused after PV: [warps][group][HEAD_DIM]
  __shared__ float s_max[kMaxGroup];
  __shared__ float s_sum[kMaxGroup];

  // Stage the group's query vectors in shared memory (fp32, scaled).
  const T* q = static_cast<const T*>(p.q);
  for (int idx = tid; idx < group * HEAD_DIM; idx += kPANumThreads) {
    const int g = idx / HEAD_DIM;
    const int d = idx - g * HEAD_DIM;
    const int q_head = kv_head * group + g;
    q_s[idx] = to_float(q[seq * p.q_stride_seq + q_head * p.q_stride_head + d]) * p.scale;
  }
  __syncthreads();

  // QK^T: each warp takes tokens start+warp, start+warp+4, ...; lanes split
  // the head dimension and finish the dot product with a shuffle reduction.
  for (int t = start + warp; t < end; t += kPANumWarps) {
    const T* k_ptr = kv_row_ptr<T, HEAD_DIM>(p.k_cache, p, p.k_stride_block, p.k_stride_token, p.k_stride_head,
                                             seq, kv_head, t);
    const Vec kv = *reinterpret_cast<const Vec*>(k_ptr + lane * V);
    float kf[V];
#pragma unroll
    for (int j = 0; j < V; ++j) kf[j] = to_float(kv.val[j]);

    for (int g = 0; g < group; ++g) {
      const float* qg = q_s + g * HEAD_DIM + lane * V;
      float s = 0.0f;
#pragma unroll
      for (int j = 0; j < V; ++j) s = fmaf(qg[j], kf[j], s);
      s = warp_reduce_sum(s);
      if (lane == 0) logits[g * kPartitionSize + (t - start)] = s;
    }
  }
  __syncthreads();

  // Softmax inside the partition: one warp per query head of the group.
  for (int g = warp; g < group; g += kPANumWarps) {
    float* lg = logits + g * kPartitionSize;
    float m = -INFINITY;
    for (int i = lane; i < n; i += kWarpSize) m = fmaxf(m, lg[i]);
    m = warp_reduce_max(m);
    float sum = 0.0f;
    for (int i = lane; i < n; i += kWarpSize) {
      const float e = __expf(lg[i] - m);
      lg[i] = e;
      sum += e;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
      s_max[g] = m;
      s_sum[g] = sum;
    }
  }
  __syncthreads();

  // P.V: same token striding as QK^T; per-lane accumulators for every head
  // in the group stay in registers.
  float acc[kMaxGroup][V];
#pragma unroll
  for (int g = 0; g < kMaxGroup; ++g) {
#pragma unroll
    for (int j = 0; j < V; ++j) acc[g][j] = 0.0f;
  }
  for (int t = start + warp; t < end; t += kPANumWarps) {
    const T* v_ptr = kv_row_ptr<T, HEAD_DIM>(p.v_cache, p, p.v_stride_block, p.v_stride_token, p.v_stride_head,
                                             seq, kv_head, t);
    const Vec vv = *reinterpret_cast<const Vec*>(v_ptr + lane * V);
    float vf[V];
#pragma unroll
    for (int j = 0; j < V; ++j) vf[j] = to_float(vv.val[j]);
#pragma unroll
    for (int g = 0; g < kMaxGroup; ++g) {
      if (g < group) {
        const float w = logits[g * kPartitionSize + (t - start)];
#pragma unroll
        for (int j = 0; j < V; ++j) acc[g][j] = fmaf(w, vf[j], acc[g][j]);
      }
    }
  }
  __syncthreads();  // all warps are done reading logits; reuse it as `red`

#pragma unroll
  for (int g = 0; g < kMaxGroup; ++g) {
    if (g < group) {
#pragma unroll
      for (int j = 0; j < V; ++j) red[(warp * group + g) * HEAD_DIM + lane * V + j] = acc[g][j];
    }
  }
  __syncthreads();

  // Cross-warp reduction and write-out.
  for (int idx = tid; idx < group * HEAD_DIM; idx += kPANumThreads) {
    const int g = idx / HEAD_DIM;
    const int d = idx - g * HEAD_DIM;
    float o = 0.0f;
#pragma unroll
    for (int w = 0; w < kPANumWarps; ++w) o += red[(w * group + g) * HEAD_DIM + d];
    o /= s_sum[g];
    const int q_head = kv_head * group + g;
    if (SPLIT) {
      const int64_t row = (static_cast<int64_t>(seq) * p.num_q_heads + q_head) * p.max_partitions + part;
      p.tmp_out[row * HEAD_DIM + d] = o;
      if (d == 0) {
        p.max_logits[row] = s_max[g];
        p.exp_sums[row] = s_sum[g];
      }
    } else {
      static_cast<T*>(p.out)[seq * p.out_stride_seq + q_head * p.out_stride_head + d] = from_float<T>(o);
    }
  }
}

// Phase 2 (split-K only): grid = (num_q_heads, num_seqs). Merges partitions:
//   M = max_p m_p,  w_p = s_p * exp(m_p - M),  out = sum_p w_p o_p / sum_p w_p
template <typename T, int HEAD_DIM>
__global__ void __launch_bounds__(kPANumThreads) paged_attention_reduce_kernel(const PagedAttnParams p) {
  const int q_head = blockIdx.x;
  const int seq = blockIdx.y;
  const int tid = threadIdx.x;
  T* out = static_cast<T*>(p.out) + seq * p.out_stride_seq + q_head * p.out_stride_head;

  const int seq_len = p.seq_lens[seq];
  if (seq_len <= 0) {
    for (int d = tid; d < HEAD_DIM; d += kPANumThreads) out[d] = from_float<T>(0.0f);
    return;
  }
  const int num_parts = min(ceil_div(seq_len, kPartitionSize), p.max_partitions);

  extern __shared__ float weights[];  // [max_partitions]
  __shared__ float scratch[32];

  const int64_t row0 = (static_cast<int64_t>(seq) * p.num_q_heads + q_head) * p.max_partitions;
  const float* ml = p.max_logits + row0;
  const float* es = p.exp_sums + row0;

  float m = -INFINITY;
  for (int i = tid; i < num_parts; i += kPANumThreads) m = fmaxf(m, ml[i]);
  const float global_max = block_reduce_max(m, scratch);

  float total = 0.0f;
  for (int i = tid; i < num_parts; i += kPANumThreads) {
    const float w = es[i] * __expf(ml[i] - global_max);
    weights[i] = w;
    total += w;
  }
  total = block_reduce_sum(total, scratch);  // contains __syncthreads, so weights are visible
  const float inv_total = 1.0f / total;

  const float* tmp = p.tmp_out + row0 * HEAD_DIM;
  for (int d = tid; d < HEAD_DIM; d += kPANumThreads) {
    float o = 0.0f;
    for (int i = 0; i < num_parts; ++i) o = fmaf(weights[i], tmp[i * HEAD_DIM + d], o);
    out[d] = from_float<T>(o * inv_total);
  }
}

template <typename T, int HEAD_DIM>
cudaError_t launch(const PagedAttnParams& p, bool split, cudaStream_t stream) {
  const int group = p.num_q_heads / p.num_kv_heads;
  const size_t logits_floats = static_cast<size_t>(group) * kPartitionSize;
  const size_t red_floats = static_cast<size_t>(kPANumWarps) * group * HEAD_DIM;
  const size_t smem = (static_cast<size_t>(group) * HEAD_DIM + (logits_floats > red_floats ? logits_floats : red_floats)) *
                      sizeof(float);
  const dim3 block(kPANumThreads);

  if (!split) {
    auto kernel = paged_attention_kernel<T, HEAD_DIM, false>;
    if (smem > 48 * 1024) {
      cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
      if (e != cudaSuccess) return e;
    }
    const dim3 grid(p.num_kv_heads, p.num_seqs, 1);
    kernel<<<grid, block, smem, stream>>>(p);
    return cudaGetLastError();
  }

  auto kernel = paged_attention_kernel<T, HEAD_DIM, true>;
  if (smem > 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    if (e != cudaSuccess) return e;
  }
  const dim3 grid(p.num_kv_heads, p.num_seqs, p.max_partitions);
  kernel<<<grid, block, smem, stream>>>(p);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return e;

  const dim3 reduce_grid(p.num_q_heads, p.num_seqs);
  const size_t reduce_smem = static_cast<size_t>(p.max_partitions) * sizeof(float);
  paged_attention_reduce_kernel<T, HEAD_DIM><<<reduce_grid, block, reduce_smem, stream>>>(p);
  return cudaGetLastError();
}

}  // namespace detail

// max_seq_len must be an upper bound on seq_lens; it selects the single-pass
// or split-K path. p.max_partitions must be >= ceil(max_seq_len / 512) when
// split-K is used.
template <typename T>
cudaError_t launch_paged_attention_decode(const PagedAttnParams& p, int head_dim, int max_seq_len,
                                          cudaStream_t stream) {
  if (p.num_seqs == 0) return cudaSuccess;
  if (p.num_kv_heads <= 0 || p.num_q_heads % p.num_kv_heads != 0) return cudaErrorInvalidValue;
  if (p.num_q_heads / p.num_kv_heads > kMaxGroup) return cudaErrorInvalidValue;
  const bool split = max_seq_len > kPartitionSize;
  if (split && (p.tmp_out == nullptr || p.max_logits == nullptr || p.exp_sums == nullptr ||
                p.max_partitions < ceil_div(max_seq_len, kPartitionSize))) {
    return cudaErrorInvalidValue;
  }
  switch (head_dim) {
    case 64:
      return detail::launch<T, 64>(p, split, stream);
    case 128:
      return detail::launch<T, 128>(p, split, stream);
    case 256:
      return detail::launch<T, 256>(p, split, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace dk
