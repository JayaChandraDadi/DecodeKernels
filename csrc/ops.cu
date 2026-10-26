// PyTorch entry points: argument checks, dtype dispatch and launches on the
// current CUDA stream. Signatures of rms_norm / fused_add_rms_norm /
// silu_and_mul match vLLM's custom ops so they can be swapped in directly.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <algorithm>
#include <optional>

#include "gemm_wmma.cuh"
#include "norm_act.cuh"
#include "paged_attention.cuh"

namespace {

#define DK_DISPATCH_HALF_TYPES(DTYPE, NAME, ...)                                  \
  [&] {                                                                           \
    if ((DTYPE) == at::ScalarType::Half) {                                        \
      using scalar_t = __half;                                                    \
      return __VA_ARGS__();                                                       \
    } else if ((DTYPE) == at::ScalarType::BFloat16) {                             \
      using scalar_t = __nv_bfloat16;                                             \
      return __VA_ARGS__();                                                       \
    }                                                                             \
    TORCH_CHECK(false, NAME, ": expected float16 or bfloat16, got ", (DTYPE)); \
  }()

void check_cuda_status(cudaError_t err, const char* name) {
  TORCH_CHECK(err == cudaSuccess, name, " failed: ", cudaGetErrorString(err));
}

void check_same(const torch::Tensor& a, const torch::Tensor& b, const char* what) {
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), what, ": dtype mismatch (", a.scalar_type(), " vs ",
              b.scalar_type(), ")");
  TORCH_CHECK(a.device() == b.device(), what, ": tensors must be on the same device");
}

}  // namespace

// ---------------------------------------------------------------------------
// Paged attention (decode)
// ---------------------------------------------------------------------------
void paged_attention_decode(torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& k_cache,
                            const torch::Tensor& v_cache, const torch::Tensor& block_tables,
                            const torch::Tensor& seq_lens, double scale, int64_t max_seq_len, torch::Tensor& tmp_out,
                            torch::Tensor& max_logits, torch::Tensor& exp_sums) {
  TORCH_CHECK(q.is_cuda(), "paged_attention_decode: q must be a CUDA tensor");
  TORCH_CHECK(q.dim() == 3 && out.dim() == 3, "q and out must be [num_seqs, num_q_heads, head_dim]");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
              "k_cache and v_cache must be [num_blocks, block_size, num_kv_heads, head_dim]");
  check_same(q, out, "q/out");
  check_same(q, k_cache, "q/k_cache");
  check_same(q, v_cache, "q/v_cache");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have the same shape as q");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache shapes differ");

  const int64_t num_seqs = q.size(0);
  const int64_t num_q_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t block_size = k_cache.size(1);
  const int64_t num_kv_heads = k_cache.size(2);
  TORCH_CHECK(k_cache.size(3) == head_dim, "cache head_dim does not match q");
  TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256, "head_dim must be 64, 128 or 256, got ",
              head_dim);
  TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be a multiple of num_kv_heads");
  TORCH_CHECK(num_q_heads / num_kv_heads <= dk::kMaxGroup, "GQA group size above ", dk::kMaxGroup,
              " is not supported");
  TORCH_CHECK(q.stride(2) == 1 && out.stride(2) == 1 && k_cache.stride(3) == 1 && v_cache.stride(3) == 1,
              "head_dim must be the contiguous (innermost) dimension");

  // Each lane reads head_dim / 32 contiguous elements with one vector load.
  const int64_t vec = head_dim / 32;
  for (int d = 0; d < 3; ++d) {
    TORCH_CHECK(k_cache.stride(d) % vec == 0 && v_cache.stride(d) % vec == 0,
                "KV-cache strides must be multiples of head_dim / 32 for vectorised loads");
  }
  TORCH_CHECK(dk::is_aligned(k_cache.data_ptr(), vec * 2) && dk::is_aligned(v_cache.data_ptr(), vec * 2),
              "KV-cache pointers must be aligned to head_dim / 16 bytes");

  TORCH_CHECK(block_tables.scalar_type() == at::kInt && block_tables.dim() == 2 && block_tables.stride(1) == 1,
              "block_tables must be a 2-D int32 tensor with a contiguous last dimension");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.dim() == 1 && seq_lens.is_contiguous(),
              "seq_lens must be a contiguous 1-D int32 tensor");
  TORCH_CHECK(block_tables.size(0) >= num_seqs && seq_lens.size(0) >= num_seqs,
              "block_tables and seq_lens must have at least num_seqs rows");
  TORCH_CHECK(max_seq_len >= 0, "max_seq_len must be non-negative");

  const bool split = max_seq_len > dk::kPartitionSize;
  const int64_t needed_parts = (max_seq_len + dk::kPartitionSize - 1) / dk::kPartitionSize;
  int max_partitions = 1;
  if (split) {
    TORCH_CHECK(tmp_out.scalar_type() == at::kFloat && max_logits.scalar_type() == at::kFloat &&
                    exp_sums.scalar_type() == at::kFloat,
                "split-K workspace tensors must be float32");
    TORCH_CHECK(tmp_out.is_contiguous() && max_logits.is_contiguous() && exp_sums.is_contiguous(),
                "split-K workspace tensors must be contiguous");
    TORCH_CHECK(tmp_out.dim() == 4 && tmp_out.size(0) == num_seqs && tmp_out.size(1) == num_q_heads &&
                    tmp_out.size(2) >= needed_parts && tmp_out.size(3) == head_dim,
                "tmp_out must be [num_seqs, num_q_heads, >= ceil(max_seq_len / 512), head_dim]");
    TORCH_CHECK(max_logits.sizes() == exp_sums.sizes() && max_logits.dim() == 3 &&
                    max_logits.size(0) == num_seqs && max_logits.size(1) == num_q_heads &&
                    max_logits.size(2) == tmp_out.size(2),
                "max_logits / exp_sums must be [num_seqs, num_q_heads, max_partitions]");
    max_partitions = static_cast<int>(tmp_out.size(2));
  }
  if (num_seqs == 0) return;

  dk::PagedAttnParams p{};
  p.out = out.data_ptr();
  p.q = q.data_ptr();
  p.k_cache = k_cache.data_ptr();
  p.v_cache = v_cache.data_ptr();
  p.block_tables = block_tables.data_ptr<int>();
  p.seq_lens = seq_lens.data_ptr<int>();
  p.tmp_out = split ? tmp_out.data_ptr<float>() : nullptr;
  p.max_logits = split ? max_logits.data_ptr<float>() : nullptr;
  p.exp_sums = split ? exp_sums.data_ptr<float>() : nullptr;
  p.q_stride_seq = q.stride(0);
  p.q_stride_head = q.stride(1);
  p.out_stride_seq = out.stride(0);
  p.out_stride_head = out.stride(1);
  p.k_stride_block = k_cache.stride(0);
  p.k_stride_token = k_cache.stride(1);
  p.k_stride_head = k_cache.stride(2);
  p.v_stride_block = v_cache.stride(0);
  p.v_stride_token = v_cache.stride(1);
  p.v_stride_head = v_cache.stride(2);
  p.bt_stride = block_tables.stride(0);
  p.num_seqs = static_cast<int>(num_seqs);
  p.num_q_heads = static_cast<int>(num_q_heads);
  p.num_kv_heads = static_cast<int>(num_kv_heads);
  p.block_size = static_cast<int>(block_size);
  p.max_partitions = max_partitions;
  p.scale = static_cast<float>(scale);

  const at::cuda::OptionalCUDAGuard guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DK_DISPATCH_HALF_TYPES(q.scalar_type(), "paged_attention_decode", [&] {
    check_cuda_status(dk::launch_paged_attention_decode<scalar_t>(p, static_cast<int>(head_dim),
                                                                  static_cast<int>(max_seq_len), stream),
                      "paged_attention_decode");
  });
}

// ---------------------------------------------------------------------------
// GEMM: c = a @ b^T (+ bias)
// ---------------------------------------------------------------------------
void gemm_tn(torch::Tensor& c, const torch::Tensor& a, const torch::Tensor& b,
             const std::optional<torch::Tensor>& bias) {
  TORCH_CHECK(a.is_cuda(), "gemm_tn: tensors must be on CUDA");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "gemm_tn: a [M,K], b [N,K], c [M,N]");
  check_same(a, b, "a/b");
  check_same(a, c, "a/c");
  const int64_t M = a.size(0), K = a.size(1), N = b.size(0);
  TORCH_CHECK(b.size(1) == K, "gemm_tn: K mismatch");
  TORCH_CHECK(c.size(0) == M && c.size(1) == N, "gemm_tn: c must be [M, N]");
  TORCH_CHECK(a.stride(1) == 1 && b.stride(1) == 1 && c.stride(1) == 1, "gemm_tn: last dims must be contiguous");
  TORCH_CHECK(K % 8 == 0 && a.stride(0) % 8 == 0 && b.stride(0) % 8 == 0,
              "gemm_tn: K and row strides of a/b must be multiples of 8");
  TORCH_CHECK(dk::is_aligned(a.data_ptr(), 16) && dk::is_aligned(b.data_ptr(), 16),
              "gemm_tn: a and b must be 16-byte aligned");
  const void* bias_ptr = nullptr;
  if (bias.has_value()) {
    check_same(a, *bias, "a/bias");
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == N && bias->is_contiguous(), "gemm_tn: bias must be [N]");
    bias_ptr = bias->data_ptr();
  }

  const at::cuda::OptionalCUDAGuard guard(a.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DK_DISPATCH_HALF_TYPES(a.scalar_type(), "gemm_tn", [&] {
    check_cuda_status(dk::launch_gemm_tn<scalar_t>(static_cast<const scalar_t*>(a.data_ptr()),
                                                   static_cast<const scalar_t*>(b.data_ptr()),
                                                   static_cast<const scalar_t*>(bias_ptr),
                                                   static_cast<scalar_t*>(c.data_ptr()), static_cast<int>(M),
                                                   static_cast<int>(N), static_cast<int>(K),
                                                   static_cast<int>(a.stride(0)), static_cast<int>(b.stride(0)),
                                                   static_cast<int>(c.stride(0)), stream),
                      "gemm_tn");
  });
}

// ---------------------------------------------------------------------------
// RMSNorm, fused residual-add + RMSNorm, SiLU-and-mul (vLLM signatures)
// ---------------------------------------------------------------------------
namespace {
// View a [..., H] tensor as rows of H with a uniform row stride.
int64_t row_stride_of(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.stride(-1) == 1, name, ": last dimension must be contiguous");
  if (t.dim() == 1) return t.size(0);
  // All leading dims must collapse to a single row stride.
  for (int64_t d = t.dim() - 2; d > 0; --d) {
    TORCH_CHECK(t.stride(d - 1) == t.stride(d) * t.size(d), name, ": leading dimensions must be collapsible");
  }
  return t.stride(-2);
}
}  // namespace

void rms_norm(torch::Tensor& out, const torch::Tensor& input, const torch::Tensor& weight, double eps) {
  TORCH_CHECK(input.is_cuda(), "rms_norm: input must be on CUDA");
  check_same(input, out, "input/out");
  check_same(input, weight, "input/weight");
  TORCH_CHECK(out.sizes() == input.sizes(), "rms_norm: out must match input shape");
  const int64_t H = input.size(-1);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == H && weight.is_contiguous(), "rms_norm: weight must be [H]");
  const int64_t rows = input.numel() / std::max<int64_t>(H, 1);
  const int64_t in_stride = row_stride_of(input, "rms_norm(input)");
  const int64_t out_stride = row_stride_of(out, "rms_norm(out)");

  const at::cuda::OptionalCUDAGuard guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DK_DISPATCH_HALF_TYPES(input.scalar_type(), "rms_norm", [&] {
    check_cuda_status(dk::launch_rms_norm<scalar_t>(static_cast<scalar_t*>(out.data_ptr()),
                                                    static_cast<const scalar_t*>(input.data_ptr()),
                                                    static_cast<const scalar_t*>(weight.data_ptr()),
                                                    static_cast<int>(rows), static_cast<int>(H), in_stride,
                                                    out_stride, static_cast<float>(eps), stream),
                      "rms_norm");
  });
}

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual, const torch::Tensor& weight, double eps) {
  TORCH_CHECK(input.is_cuda(), "fused_add_rms_norm: input must be on CUDA");
  check_same(input, residual, "input/residual");
  check_same(input, weight, "input/weight");
  TORCH_CHECK(residual.sizes() == input.sizes(), "fused_add_rms_norm: residual must match input shape");
  const int64_t H = input.size(-1);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == H && weight.is_contiguous(),
              "fused_add_rms_norm: weight must be [H]");
  const int64_t rows = input.numel() / std::max<int64_t>(H, 1);
  const int64_t in_stride = row_stride_of(input, "fused_add_rms_norm(input)");
  const int64_t res_stride = row_stride_of(residual, "fused_add_rms_norm(residual)");

  const at::cuda::OptionalCUDAGuard guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DK_DISPATCH_HALF_TYPES(input.scalar_type(), "fused_add_rms_norm", [&] {
    check_cuda_status(dk::launch_fused_add_rms_norm<scalar_t>(
                          static_cast<scalar_t*>(input.data_ptr()), static_cast<scalar_t*>(residual.data_ptr()),
                          static_cast<const scalar_t*>(weight.data_ptr()), static_cast<int>(rows),
                          static_cast<int>(H), in_stride, res_stride, static_cast<float>(eps), stream),
                      "fused_add_rms_norm");
  });
}

void silu_and_mul(torch::Tensor& out, const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "silu_and_mul: input must be on CUDA");
  check_same(input, out, "input/out");
  const int64_t d = input.size(-1) / 2;
  TORCH_CHECK(input.size(-1) == 2 * d, "silu_and_mul: last dimension must be even");
  TORCH_CHECK(out.size(-1) == d && out.numel() * 2 == input.numel(), "silu_and_mul: out must be [..., d]");
  const int64_t rows = d == 0 ? 0 : input.numel() / (2 * d);
  const int64_t in_stride = row_stride_of(input, "silu_and_mul(input)");
  const int64_t out_stride = row_stride_of(out, "silu_and_mul(out)");

  const at::cuda::OptionalCUDAGuard guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DK_DISPATCH_HALF_TYPES(input.scalar_type(), "silu_and_mul", [&] {
    check_cuda_status(dk::launch_silu_and_mul<scalar_t>(static_cast<scalar_t*>(out.data_ptr()),
                                                        static_cast<const scalar_t*>(input.data_ptr()),
                                                        static_cast<int>(rows), static_cast<int>(d), in_stride,
                                                        out_stride, stream),
                      "silu_and_mul");
  });
}

// ---------------------------------------------------------------------------
// Registration: torch.ops.decode_kernels.*
// ---------------------------------------------------------------------------
TORCH_LIBRARY(decode_kernels, m) {
  m.def(
      "paged_attention_decode(Tensor! out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, "
      "Tensor seq_lens, float scale, int max_seq_len, Tensor! tmp_out, Tensor! max_logits, Tensor! exp_sums) -> ()");
  m.def("gemm_tn(Tensor! c, Tensor a, Tensor b, Tensor? bias) -> ()");
  m.def("rms_norm(Tensor! out, Tensor input, Tensor weight, float eps) -> ()");
  m.def("fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float eps) -> ()");
  m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
}

TORCH_LIBRARY_IMPL(decode_kernels, CUDA, m) {
  m.impl("paged_attention_decode", &paged_attention_decode);
  m.impl("gemm_tn", &gemm_tn);
  m.impl("rms_norm", &rms_norm);
  m.impl("fused_add_rms_norm", &fused_add_rms_norm);
  m.impl("silu_and_mul", &silu_and_mul);
}
