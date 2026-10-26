"""CPU tests for the reference implementations and the split-K algorithm.

These run anywhere (no GPU) and check the math the CUDA kernels implement:
the partition / log-sum-exp merge must equal dense attention, paged gathering
must equal contiguous attention, and strided cache layouts must agree.
"""

import pytest
import torch
import torch.nn.functional as F

from decode_kernels.reference import (
    fused_add_rms_norm_reference,
    gemm_reference,
    paged_attention_reference,
    rms_norm_reference,
    silu_and_mul_reference,
    split_k_attention_reference,
)
from decode_kernels.testing import make_paged_kv

torch.manual_seed(0)


def _queries(num_seqs, num_q_heads, head_dim, dtype=torch.float32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_seqs, num_q_heads, head_dim, generator=g, dtype=dtype)


@pytest.mark.parametrize("partition_size", [64, 512])
@pytest.mark.parametrize("group", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_split_k_merge_matches_dense(partition_size, group, head_dim):
    seq_lens = [1, 7, 63, 64, 65, 511, 512, 513, 1300]
    num_kv_heads = 2
    kv = make_paged_kv(seq_lens, num_kv_heads, head_dim, block_size=16, dtype=torch.float32)
    q = _queries(len(seq_lens), num_kv_heads * group, head_dim)
    dense = paged_attention_reference(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens)
    split = split_k_attention_reference(
        q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens, partition_size=partition_size
    )
    torch.testing.assert_close(split, dense, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("block_size", [1, 16, 32])
def test_paged_gather_matches_contiguous_attention(block_size):
    """Scattered physical blocks must give the same result as SDPA over the
    contiguous K/V of each sequence."""
    seq_lens = [5, 33, 100]
    num_kv_heads, group, head_dim = 2, 4, 64
    kv = make_paged_kv(seq_lens, num_kv_heads, head_dim, block_size=block_size, dtype=torch.float32)
    q = _queries(len(seq_lens), num_kv_heads * group, head_dim)
    out = paged_attention_reference(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens)

    for s, L in enumerate(seq_lens):
        idx = torch.arange(L)
        blocks = kv.block_tables[s, idx // block_size].long()
        k = kv.k_cache[blocks, idx % block_size].transpose(0, 1)  # [Hkv, L, D]
        v = kv.v_cache[blocks, idx % block_size].transpose(0, 1)
        k = k.repeat_interleave(group, dim=0)
        v = v.repeat_interleave(group, dim=0)
        expected = F.scaled_dot_product_attention(q[s][:, None, :], k, v)[:, 0, :]  # [Hq, D]
        torch.testing.assert_close(out[s], expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("layout", ["hnd", "vllm"])
def test_strided_layouts_match_nhd(layout):
    """vLLM's interleaved K|V cache and FlashInfer's HND layout, viewed as
    strided NHD tensors, must give the same attention as a contiguous copy."""
    seq_lens = [3, 40, 129]
    kv = make_paged_kv(seq_lens, 2, 64, block_size=16, dtype=torch.float32, layout=layout)
    assert not kv.k_cache.is_contiguous()
    assert kv.k_cache.stride(-1) == 1
    q = _queries(len(seq_lens), 8, 64)
    strided = paged_attention_reference(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens)
    contiguous = paged_attention_reference(
        q, kv.k_cache.contiguous(), kv.v_cache.contiguous(), kv.block_tables, kv.seq_lens
    )
    torch.testing.assert_close(strided, contiguous)


def test_vllm_layout_views_point_into_one_buffer():
    kv = make_paged_kv([20], 2, 64, block_size=16, dtype=torch.float32, layout="vllm")
    # K and V share storage, V starts head_dim elements after K.
    assert kv.k_cache.untyped_storage().data_ptr() == kv.v_cache.untyped_storage().data_ptr()
    assert kv.v_cache.data_ptr() - kv.k_cache.data_ptr() == 64 * kv.k_cache.element_size()


def test_empty_sequence_gives_zeros():
    kv = make_paged_kv([0, 10], 1, 64, dtype=torch.float32)
    q = _queries(2, 1, 64)
    out = paged_attention_reference(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens)
    assert torch.count_nonzero(out[0]) == 0
    split = split_k_attention_reference(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens, partition_size=4)
    torch.testing.assert_close(split, out)


def test_gemm_reference_matches_linear():
    a = torch.randn(7, 64)
    b = torch.randn(48, 64)
    bias = torch.randn(48)
    torch.testing.assert_close(gemm_reference(a, b, bias), F.linear(a, b, bias))


def test_rms_norm_reference():
    x = torch.randn(4, 256)
    w = torch.randn(256)
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(rms_norm_reference(x, w), expected)


def test_fused_add_rms_norm_reference_rounds_residual():
    x = torch.randn(4, 256, dtype=torch.float16)
    r = torch.randn(4, 256, dtype=torch.float16)
    w = torch.randn(256, dtype=torch.float16)
    normed, new_res = fused_add_rms_norm_reference(x, r, w)
    assert new_res.dtype == torch.float16
    torch.testing.assert_close(new_res, (x.float() + r.float()).half())
    torch.testing.assert_close(normed, rms_norm_reference(new_res, w))


def test_silu_and_mul_reference():
    x = torch.randn(3, 2 * 40)
    torch.testing.assert_close(silu_and_mul_reference(x), F.silu(x[:, :40]) * x[:, 40:])
