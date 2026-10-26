"""Helpers that build paged KV caches for tests and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PagedKV:
    k_cache: torch.Tensor  # [num_blocks, block_size, num_kv_heads, head_dim] (possibly a strided view)
    v_cache: torch.Tensor
    block_tables: torch.Tensor  # [num_seqs, max_blocks] int32
    seq_lens: torch.Tensor  # [num_seqs] int32
    block_size: int

    @property
    def max_seq_len(self) -> int:
        return int(self.seq_lens.max().item()) if self.seq_lens.numel() else 0


def make_paged_kv(
    seq_lens: list[int],
    num_kv_heads: int,
    head_dim: int,
    block_size: int = 16,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cpu",
    layout: str = "nhd",
    extra_blocks: int = 3,
    seed: int = 0,
) -> PagedKV:
    """Random paged KV cache with shuffled physical blocks.

    layout:
      * ``nhd``  -- contiguous [num_blocks, block_size, num_kv_heads, head_dim]
                    (FlashInfer NHD).
      * ``hnd``  -- stored as [num_blocks, num_kv_heads, block_size, head_dim]
                    and viewed as NHD (FlashInfer HND).
      * ``vllm`` -- vLLM's interleaved cache [num_blocks, num_kv_heads,
                    block_size, 2 * head_dim] viewed as two strided NHD tensors,
                    exactly as FlashAttentionImpl.forward does.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    blocks_per_seq = [(n + block_size - 1) // block_size for n in seq_lens]
    num_blocks = sum(blocks_per_seq) + extra_blocks
    shape_args = dict(dtype=dtype)

    if layout == "nhd":
        k = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, generator=gen, **shape_args)
        v = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, generator=gen, **shape_args)
        k, v = k.to(device), v.to(device)
    elif layout == "hnd":
        k = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, generator=gen, **shape_args)
        v = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, generator=gen, **shape_args)
        k, v = k.to(device).transpose(1, 2), v.to(device).transpose(1, 2)
    elif layout == "vllm":
        kv = torch.randn(num_blocks, num_kv_heads, block_size, 2 * head_dim, generator=gen, **shape_args).to(device)
        k, v = kv.transpose(1, 2).split(head_dim, dim=-1)
    else:
        raise ValueError(f"unknown layout {layout!r}")

    perm = torch.randperm(num_blocks, generator=gen)
    max_blocks = max(blocks_per_seq) if blocks_per_seq else 1
    block_tables = torch.zeros(len(seq_lens), max(max_blocks, 1), dtype=torch.int32)
    cursor = 0
    for s, nb in enumerate(blocks_per_seq):
        block_tables[s, :nb] = perm[cursor : cursor + nb].to(torch.int32)
        cursor += nb

    return PagedKV(
        k_cache=k,
        v_cache=v,
        block_tables=block_tables.to(device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        block_size=block_size,
    )
