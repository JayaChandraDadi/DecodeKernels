"""Pure-PyTorch reference implementations (CPU or GPU).

These define the expected results for every kernel. They are written for
clarity, not speed: tests compare the CUDA kernels against them, and
``split_k_attention_reference`` re-implements the exact partition/merge math
of the split-K kernel so the algorithm can be checked on a CPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gather_kv(cache: torch.Tensor, block_table_row: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Token rows [seq_len, num_kv_heads, head_dim] of one sequence from a paged cache."""
    block_size = cache.shape[1]
    idx = torch.arange(seq_len, device=cache.device)
    blocks = block_table_row[idx // block_size].long()
    return cache[blocks, idx % block_size]


def _per_seq_kv(k_cache, v_cache, block_tables, s, seq_len, group):
    k = gather_kv(k_cache, block_tables[s], seq_len).float().repeat_interleave(group, dim=1)
    v = gather_kv(v_cache, block_tables[s], seq_len).float().repeat_interleave(group, dim=1)
    return k, v  # [L, num_q_heads, head_dim]


def paged_attention_reference(q, k_cache, v_cache, block_tables, seq_lens, scale=None) -> torch.Tensor:
    """Dense softmax attention of each query token over its whole sequence."""
    num_seqs, num_q_heads, head_dim = q.shape
    group = num_q_heads // k_cache.shape[2]
    scale = head_dim**-0.5 if scale is None else scale
    out = torch.zeros(num_seqs, num_q_heads, head_dim, dtype=torch.float32, device=q.device)
    for s in range(num_seqs):
        seq_len = int(seq_lens[s])
        if seq_len == 0:
            continue
        k, v = _per_seq_kv(k_cache, v_cache, block_tables, s, seq_len, group)
        scores = torch.einsum("hd,lhd->hl", q[s].float() * scale, k)
        probs = torch.softmax(scores, dim=-1)
        out[s] = torch.einsum("hl,lhd->hd", probs, v)
    return out.to(q.dtype)


def split_k_attention_reference(
    q, k_cache, v_cache, block_tables, seq_lens, scale=None, partition_size: int = 512
) -> torch.Tensor:
    """Flash-decoding math: per-partition (max, exp-sum, normalised output), then
    a log-sum-exp merge. Returns float32 so it can be compared with the dense
    reference without rounding noise."""
    num_seqs, num_q_heads, head_dim = q.shape
    group = num_q_heads // k_cache.shape[2]
    scale = head_dim**-0.5 if scale is None else scale
    out = torch.zeros(num_seqs, num_q_heads, head_dim, dtype=torch.float32, device=q.device)
    for s in range(num_seqs):
        seq_len = int(seq_lens[s])
        if seq_len == 0:
            continue
        k, v = _per_seq_kv(k_cache, v_cache, block_tables, s, seq_len, group)
        qs = q[s].float() * scale
        maxes, sums, outs = [], [], []
        for start in range(0, seq_len, partition_size):
            end = min(start + partition_size, seq_len)
            logits = torch.einsum("hd,lhd->hl", qs, k[start:end])  # [H, n]
            m = logits.max(dim=-1).values  # [H]
            e = torch.exp(logits - m[:, None])
            ssum = e.sum(dim=-1)  # [H]
            o = torch.einsum("hl,lhd->hd", e, v[start:end]) / ssum[:, None]
            maxes.append(m)
            sums.append(ssum)
            outs.append(o)
        m = torch.stack(maxes)  # [P, H]
        ssum = torch.stack(sums)  # [P, H]
        o = torch.stack(outs)  # [P, H, D]
        global_max = m.max(dim=0).values
        w = ssum * torch.exp(m - global_max)  # [P, H]
        out[s] = (w[:, :, None] * o).sum(dim=0) / w.sum(dim=0)[:, None]
    return out


def gemm_reference(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    out = a.float() @ b.float().t()
    if bias is not None:
        out = out + bias.float()
    return out.to(a.dtype)


def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv * weight.float()).to(x.dtype)


def fused_add_rms_norm_reference(x, residual, weight, eps: float = 1e-6):
    """Returns (normed, new_residual); the residual is rounded to the input
    dtype before the norm, matching vLLM and the kernel."""
    new_residual = (x.float() + residual.float()).to(x.dtype)
    return rms_norm_reference(new_residual, weight, eps), new_residual


def silu_and_mul_reference(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return (F.silu(x[..., :d].float()) * x[..., d:].float()).to(x.dtype)
