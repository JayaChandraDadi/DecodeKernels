"""GPU tests: every CUDA kernel against the PyTorch reference.

Skipped automatically when no GPU is visible or the extension is not built.
"""

import pytest
import torch

import decode_kernels as dk
from decode_kernels.reference import (
    fused_add_rms_norm_reference,
    gemm_reference,
    paged_attention_reference,
    rms_norm_reference,
    silu_and_mul_reference,
)
from decode_kernels.testing import make_paged_kv

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not dk.is_available(), reason="needs a CUDA GPU and the built extension"),
]

DTYPES = [torch.float16, torch.bfloat16]
ATTN_TOL = {torch.float16: dict(atol=3e-3, rtol=1e-2), torch.bfloat16: dict(atol=2e-2, rtol=2e-2)}


def _q(num_seqs, num_q_heads, head_dim, dtype, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(num_seqs, num_q_heads, head_dim, generator=g, device="cuda", dtype=dtype)


# ---------------------------------------------------------------------------
# Paged attention
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("group", [1, 4, 8])
@pytest.mark.parametrize("block_size", [16, 32])
def test_paged_attention_short(dtype, head_dim, group, block_size):
    """Single-pass path: every sequence fits in one partition."""
    seq_lens = [1, 2, 17, 100, 255, 512]
    num_kv_heads = 2
    kv = make_paged_kv(seq_lens, num_kv_heads, head_dim, block_size, dtype, "cuda")
    q = _q(len(seq_lens), num_kv_heads * group, head_dim, dtype)
    out = dk.paged_attention_decode(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens)
    ref = paged_attention_reference(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens)
    torch.testing.assert_close(out, ref, **ATTN_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("group", [1, 8])
def test_paged_attention_split_k(dtype, head_dim, group):
    """Split-K path, including partition boundaries and sequences much
    shorter than the batch maximum."""
    seq_lens = [1, 511, 512, 513, 1024, 1025, 3000, 8192]
    num_kv_heads = 2
    kv = make_paged_kv(seq_lens, num_kv_heads, head_dim, 16, dtype, "cuda")
    q = _q(len(seq_lens), num_kv_heads * group, head_dim, dtype)
    out = dk.paged_attention_decode(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens)
    ref = paged_attention_reference(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens)
    torch.testing.assert_close(out, ref, **ATTN_TOL[dtype])


@pytest.mark.parametrize("layout", ["hnd", "vllm"])
@pytest.mark.parametrize("seq_lens", [[5, 300], [700, 2000]])
def test_paged_attention_strided_layouts(layout, seq_lens):
    """vLLM's interleaved K|V cache and FlashInfer HND views, read in place."""
    dtype = torch.float16
    kv = make_paged_kv(seq_lens, 2, 128, 16, dtype, "cuda", layout=layout)
    assert not kv.k_cache.is_contiguous()
    q = _q(len(seq_lens), 8, 128, dtype)
    out = dk.paged_attention_decode(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens)
    ref = paged_attention_reference(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens)
    torch.testing.assert_close(out, ref, **ATTN_TOL[dtype])


@pytest.mark.parametrize("max_seq_len", [100, 2000])
def test_paged_attention_empty_sequences(max_seq_len):
    """Padding rows (seq_len 0) must produce zeros, not NaNs."""
    seq_lens = [0, 50, 0, max_seq_len]
    kv = make_paged_kv(seq_lens, 2, 64, 16, torch.float16, "cuda")
    q = _q(len(seq_lens), 4, 64, torch.float16)
    out = dk.paged_attention_decode(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[0]) == 0 and torch.count_nonzero(out[2]) == 0
    ref = paged_attention_reference(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens)
    torch.testing.assert_close(out, ref, **ATTN_TOL[torch.float16])


def test_paged_attention_custom_scale_and_output_buffer():
    kv = make_paged_kv([64, 900], 1, 128, 16, torch.bfloat16, "cuda")
    q = _q(2, 4, 128, torch.bfloat16)
    out = torch.full_like(q, float("nan"))
    dk.paged_attention_decode(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens, scale=0.05, out=out)
    ref = paged_attention_reference(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens, scale=0.05)
    torch.testing.assert_close(out, ref, **ATTN_TOL[torch.bfloat16])


def test_paged_attention_cuda_graph_replay():
    """Preallocated buffers make the op capturable; replay must track new
    cache contents and lengths (within the captured max_seq_len)."""
    dtype = torch.float16
    seq_lens = [300, 1500, 40]
    max_seq_len = 2048
    kv = make_paged_kv([max_seq_len] * 3, 2, 128, 16, dtype, "cuda")
    kv.seq_lens.copy_(torch.tensor(seq_lens, dtype=torch.int32))
    q = _q(3, 8, 128, dtype)
    out = torch.empty_like(q)
    ws = dk.alloc_workspace(3, 8, 128, max_seq_len, "cuda")

    def step():
        dk.paged_attention_decode(
            q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens, max_seq_len=max_seq_len, out=out, workspace=ws
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()  # warm up outside the graph
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    for new_lens in ([300, 1500, 40], [2048, 1, 777]):
        kv.seq_lens.copy_(torch.tensor(new_lens, dtype=torch.int32))
        q.normal_()
        graph.replay()
        torch.cuda.synchronize()
        ref = paged_attention_reference(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens)
        torch.testing.assert_close(out, ref, **ATTN_TOL[dtype])


def test_paged_attention_rejects_bad_inputs():
    kv = make_paged_kv([10], 2, 128, 16, torch.float16, "cuda")
    q = _q(1, 32, 128, torch.float16)  # group 16 > 8
    with pytest.raises(RuntimeError, match="GQA group"):
        dk.paged_attention_decode(q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens)
    kv96 = make_paged_kv([10], 2, 96, 16, torch.float16, "cuda")
    with pytest.raises(RuntimeError, match="head_dim"):
        dk.paged_attention_decode(_q(1, 2, 96, torch.float16), kv96.k_cache, kv96.v_cache, kv96.block_tables,
                                  kv96.seq_lens)


# ---------------------------------------------------------------------------
# GEMM
# ---------------------------------------------------------------------------
GEMM_SHAPES = [
    (1, 4096, 4096),  # decode, batch 1
    (7, 1000, 520),  # ragged everything
    (64, 6144, 4096),  # largest skinny-tile M
    (65, 512, 256),  # first M on the 128x128 path
    (128, 4096, 4096),
    (333, 1024, 1032),
    (512, 11008, 4096),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M,N,K", GEMM_SHAPES)
@pytest.mark.parametrize("with_bias", [False, True])
def test_gemm(dtype, M, N, K, with_bias):
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(M, K, generator=g, device="cuda", dtype=dtype)
    b = torch.randn(N, K, generator=g, device="cuda", dtype=dtype) / K**0.5
    bias = torch.randn(N, generator=g, device="cuda", dtype=dtype) if with_bias else None
    out = dk.gemm(a, b, bias)
    ref = gemm_reference(a, b, bias)
    tol = dict(atol=2e-2, rtol=2e-2) if dtype == torch.float16 else dict(atol=6e-2, rtol=4e-2)
    torch.testing.assert_close(out, ref, **tol)


def test_gemm_batched_input_and_strided_rows():
    a = torch.randn(2, 3, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(80, 256, device="cuda", dtype=torch.float16) / 16
    out = dk.gemm(a, b)
    assert out.shape == (2, 3, 80)
    torch.testing.assert_close(out, gemm_reference(a.reshape(-1, 256), b).view(2, 3, 80), atol=2e-2, rtol=2e-2)
    # Row-strided A (a column slice of a wider buffer).
    wide = torch.randn(5, 512, device="cuda", dtype=torch.float16)
    a2 = wide[:, :256]
    torch.testing.assert_close(dk.gemm(a2, b), gemm_reference(a2, b), atol=2e-2, rtol=2e-2)


def test_gemm_rejects_unaligned_k():
    a = torch.randn(4, 100, device="cuda", dtype=torch.float16)
    b = torch.randn(8, 100, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="multiples of 8"):
        dk.gemm(a, b)


# ---------------------------------------------------------------------------
# RMSNorm / fused add / SiLU-and-mul
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows,H", [(1, 4096), (33, 5120), (7, 8), (4, 100), (128, 8192)])
def test_rms_norm(dtype, rows, H):
    x = torch.randn(rows, H, device="cuda", dtype=dtype)
    w = torch.randn(H, device="cuda", dtype=dtype)
    torch.testing.assert_close(dk.rms_norm(x, w, 1e-6), rms_norm_reference(x, w, 1e-6), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows,H", [(1, 4096), (17, 2048), (5, 100)])
def test_fused_add_rms_norm(dtype, rows, H):
    x = torch.randn(rows, H, device="cuda", dtype=dtype)
    r = torch.randn(rows, H, device="cuda", dtype=dtype)
    w = torch.randn(H, device="cuda", dtype=dtype)
    ref_out, ref_res = fused_add_rms_norm_reference(x, r, w, 1e-5)
    out, res = dk.fused_add_rms_norm(x.clone(), r.clone(), w, 1e-5)
    torch.testing.assert_close(res, ref_res, atol=0, rtol=0)  # residual add is exact up to rounding
    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)


def test_rms_norm_3d_input():
    x = torch.randn(2, 5, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(dk.rms_norm(x, w), rms_norm_reference(x, w), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows,d", [(1, 14336), (64, 11008), (3, 100), (9, 8)])
def test_silu_and_mul(dtype, rows, d):
    x = torch.randn(rows, 2 * d, device="cuda", dtype=dtype)
    torch.testing.assert_close(dk.silu_and_mul(x), silu_and_mul_reference(x), atol=2e-2, rtol=2e-2)


def test_ops_work_under_torch_compile():
    """Fake implementations let torch.compile trace the mutating ops."""
    w = torch.randn(256, device="cuda", dtype=torch.float16)

    @torch.compile(fullgraph=True)
    def f(x):
        out = torch.empty_like(x)
        torch.ops.decode_kernels.rms_norm(out, x, w, 1e-6)
        return out * 2

    x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(f(x), rms_norm_reference(x, w) * 2, atol=4e-2, rtol=2e-2)
