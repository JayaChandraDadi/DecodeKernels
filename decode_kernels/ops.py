"""Python wrappers around torch.ops.decode_kernels.*.

The wrappers allocate outputs and the split-K workspace; the raw ops are
mutating (``Tensor!`` arguments) so they can also be called on preallocated
buffers, e.g. inside CUDA graphs.
"""

from __future__ import annotations

import math

import torch

PARTITION_SIZE = 512  # must match dk::kPartitionSize in csrc/paged_attention.cuh
MAX_GQA_GROUP = 8  # must match dk::kMaxGroup
SUPPORTED_HEAD_DIMS = (64, 128, 256)

try:
    from . import _C  # noqa: F401  (registers torch.ops.decode_kernels.*)

    _AVAILABLE = True
    _IMPORT_ERROR: Exception | None = None
except ImportError as e:  # extension not built (e.g. CPU-only machine)
    _AVAILABLE = False
    _IMPORT_ERROR = e


def is_available() -> bool:
    """True when the CUDA extension is built and a GPU is visible."""
    return _AVAILABLE and torch.cuda.is_available()


def _require() -> None:
    if not _AVAILABLE:
        raise RuntimeError(
            "decode_kernels._C is not built. Run `pip install -e . --no-build-isolation` on a machine "
            f"with CUDA. Original error: {_IMPORT_ERROR}"
        )


if _AVAILABLE:
    # Fake (meta) implementations let torch.compile trace through the ops.
    @torch.library.register_fake("decode_kernels::paged_attention_decode")
    def _(out, q, k_cache, v_cache, block_tables, seq_lens, scale, max_seq_len, tmp_out, max_logits, exp_sums):
        return None

    @torch.library.register_fake("decode_kernels::gemm_tn")
    def _(c, a, b, bias):
        return None

    @torch.library.register_fake("decode_kernels::rms_norm")
    def _(out, input, weight, eps):
        return None

    @torch.library.register_fake("decode_kernels::fused_add_rms_norm")
    def _(input, residual, weight, eps):
        return None

    @torch.library.register_fake("decode_kernels::silu_and_mul")
    def _(out, input):
        return None


# ---------------------------------------------------------------------------
# Paged attention
# ---------------------------------------------------------------------------
def num_partitions(max_seq_len: int) -> int:
    return max(1, math.ceil(max_seq_len / PARTITION_SIZE))


def alloc_workspace(num_seqs: int, num_q_heads: int, head_dim: int, max_seq_len: int, device) -> tuple:
    """Split-K buffers for sequences longer than PARTITION_SIZE (empty otherwise)."""
    if max_seq_len <= PARTITION_SIZE:
        empty = torch.empty(0, dtype=torch.float32, device=device)
        return empty, empty, empty
    parts = num_partitions(max_seq_len)
    tmp_out = torch.empty(num_seqs, num_q_heads, parts, head_dim, dtype=torch.float32, device=device)
    max_logits = torch.empty(num_seqs, num_q_heads, parts, dtype=torch.float32, device=device)
    exp_sums = torch.empty_like(max_logits)
    return tmp_out, max_logits, exp_sums


def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
    max_seq_len: int | None = None,
    out: torch.Tensor | None = None,
    workspace: tuple | None = None,
) -> torch.Tensor:
    """Single-token decode attention over a paged KV cache.

    q: [num_seqs, num_q_heads, head_dim]; k_cache / v_cache:
    [num_blocks, block_size, num_kv_heads, head_dim] (any strides on the first
    three dims); block_tables: [num_seqs, max_blocks] int32; seq_lens:
    [num_seqs] int32. ``max_seq_len`` must be >= every entry of seq_lens; pass
    it when known to avoid a device-to-host sync.
    """
    _require()
    num_seqs, num_q_heads, head_dim = q.shape
    if scale is None:
        scale = head_dim**-0.5
    if max_seq_len is None:
        max_seq_len = int(seq_lens[:num_seqs].max().item()) if num_seqs > 0 else 0
    if out is None:
        out = torch.empty_like(q)
    if workspace is None:
        workspace = alloc_workspace(num_seqs, num_q_heads, head_dim, max_seq_len, q.device)
    tmp_out, max_logits, exp_sums = workspace
    torch.ops.decode_kernels.paged_attention_decode(
        out, q, k_cache, v_cache, block_tables, seq_lens, float(scale), int(max_seq_len), tmp_out, max_logits, exp_sums
    )
    return out


# ---------------------------------------------------------------------------
# GEMM
# ---------------------------------------------------------------------------
def gemm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None, out: torch.Tensor | None = None):
    """``a @ b.T + bias`` (nn.Linear semantics) on tensor cores. a: [..., K], b: [N, K]."""
    _require()
    lead = a.shape[:-1]
    a2 = a.reshape(-1, a.shape[-1])
    if a2.stride(-1) != 1:
        a2 = a2.contiguous()
    if out is None:
        out = torch.empty(a2.shape[0], b.shape[0], dtype=a.dtype, device=a.device)
    torch.ops.decode_kernels.gemm_tn(out, a2, b, bias)
    return out.view(*lead, b.shape[0])


# ---------------------------------------------------------------------------
# Norm / activation (vLLM-compatible)
# ---------------------------------------------------------------------------
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, out: torch.Tensor | None = None):
    _require()
    if out is None:
        out = torch.empty_like(x)
    torch.ops.decode_kernels.rms_norm(out, x, weight, float(eps))
    return out


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """In place: residual += x; x = rms_norm(residual) * weight. Returns (x, residual)."""
    _require()
    torch.ops.decode_kernels.fused_add_rms_norm(x, residual, weight, float(eps))
    return x, residual


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    """silu(x[..., :d]) * x[..., d:] with d = x.shape[-1] // 2."""
    _require()
    d = x.shape[-1] // 2
    if out is None:
        out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
    torch.ops.decode_kernels.silu_and_mul(out, x)
    return out
