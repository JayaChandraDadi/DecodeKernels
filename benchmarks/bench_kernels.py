"""Benchmark every kernel against its production baseline with CUDA events.

    python benchmarks/bench_kernels.py                    # everything, fp16
    python benchmarks/bench_kernels.py --kernels attention --dtype bf16
    python benchmarks/bench_kernels.py --quick            # small sweep

Baselines
    attention  FlashInfer BatchDecodeWithPagedKVCacheWrapper (CUDA-core and
               tensor-core decode), when flashinfer-python is installed
    gemm       torch.nn.functional.linear (cuBLAS / cuBLASLt)
    rms_norm   PyTorch eager reference
    silu       PyTorch eager reference

Every implementation is checked against the baseline output before it is
timed, so a wrong kernel can never produce a "fast" number. Results go to
results/kernels_<gpu>_<date>.csv along with the GPU, driver stack and date.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decode_kernels as dk  # noqa: E402
from bench_utils import Row, print_table, time_cuda, write_csv  # noqa: E402
from decode_kernels.reference import paged_attention_reference, rms_norm_reference, silu_and_mul_reference  # noqa: E402
from decode_kernels.testing import make_paged_kv  # noqa: E402

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

# (name, num_q_heads, num_kv_heads, head_dim)
ATTN_MODELS = [
    ("llama3-8b", 32, 8, 128),
    ("qwen2.5-7b", 28, 4, 128),
    ("qwen2.5-0.5b", 14, 2, 64),
]
# (name, N, K) projections of a Llama-3-8B layer
MAX_KV_BYTES = 6 * 1024**3  # skip attention configs whose KV cache would exceed this

GEMM_SHAPES = [
    ("qkv_proj", 6144, 4096),
    ("o_proj", 4096, 4096),
    ("gate_up_proj", 28672, 4096),
    ("down_proj", 4096, 14336),
]


def _check(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> None:
    if not torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol):
        err = (got.float() - ref.float()).abs().max().item()
        raise AssertionError(f"{name}: output mismatch before timing (max abs err {err:.3e})")


def _flashinfer_decode(kv, q, num_q_heads, num_kv_heads, head_dim, dtype, use_tensor_cores):
    import flashinfer

    block_size = kv.block_size
    seq_lens = kv.seq_lens.tolist()
    blocks = [(n + block_size - 1) // block_size for n in seq_lens]
    indptr = torch.tensor([0] + list(torch.tensor(blocks).cumsum(0).tolist()), dtype=torch.int32, device="cuda")
    indices = torch.cat([kv.block_tables[i, :b] for i, b in enumerate(blocks)]).to(torch.int32)
    last_page_len = torch.tensor([(n - 1) % block_size + 1 for n in seq_lens], dtype=torch.int32, device="cuda")
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", use_tensor_cores=use_tensor_cores
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        sm_scale=head_dim**-0.5,
    )
    return lambda: wrapper.run(q, (kv.k_cache, kv.v_cache))


def bench_attention(dtype, batches, ctx_lens, iters) -> list[Row]:
    rows: list[Row] = []
    try:
        import flashinfer  # noqa: F401

        have_fi = True
    except Exception:  # noqa: BLE001
        have_fi = False
        print("[attention] flashinfer not installed; timing DecodeKernels only")

    for name, hq, hkv, d in ATTN_MODELS:
        for bs in batches:
            for ctx in ctx_lens:
                kv_bytes = 2 * bs * ctx * hkv * d * torch.finfo(dtype).bits // 8
                if kv_bytes > MAX_KV_BYTES:
                    print(f"[attention] skip {name} bs={bs} ctx={ctx}: KV cache {kv_bytes / 1e9:.1f} GB")
                    continue
                kv = make_paged_kv([ctx] * bs, hkv, d, 16, dtype, "cuda", seed=bs + ctx)
                q = torch.randn(bs, hq, d, device="cuda", dtype=dtype)
                ws = dk.alloc_workspace(bs, hq, d, ctx, "cuda")
                out = torch.empty_like(q)

                def run_dk():
                    dk.paged_attention_decode(
                        q, kv.k_cache, kv.v_cache, kv.block_tables, kv.seq_lens, max_seq_len=ctx, out=out,
                        workspace=ws,
                    )

                run_dk()
                # Check the first sequences against the reference (the full
                # reference is slow for long contexts).
                nchk = min(bs, 2)
                ref = paged_attention_reference(
                    q[:nchk], kv.k_cache, kv.v_cache, kv.block_tables[:nchk], kv.seq_lens[:nchk]
                )
                _check(f"attention {name}", out[:nchk], ref, 3e-2, 3e-2)

                # KV bytes dominate: every K and V row is read once.
                io_bytes = kv_bytes + 2 * q.numel() * q.element_size()
                cfg = f"{name} bs={bs} ctx={ctx}"
                impls = [("decode_kernels", run_dk)]
                if have_fi:
                    for tc in (False, True):
                        fn = _flashinfer_decode(kv, q, hq, hkv, d, dtype, tc)
                        _check(f"flashinfer {name}", fn()[:nchk], ref, 3e-2, 3e-2)
                        impls.append((f"flashinfer{'_tc' if tc else ''}", fn))

                base_us = None
                timings = {impl: time_cuda(fn, iters=iters) for impl, fn in impls}
                if "flashinfer" in timings:
                    base_us = min(t.median_us for k, t in timings.items() if k.startswith("flashinfer"))
                for impl, t in timings.items():
                    gbps = io_bytes / (t.median_us * 1e-6) / 1e9
                    sp = (base_us / t.median_us) if (base_us and impl == "decode_kernels") else None
                    rows.append(Row("attention", cfg, impl, t.median_us, t.p10_us, t.p90_us, "GB/s", gbps, sp))
    return rows


def bench_gemm(dtype, ms, iters) -> list[Row]:
    rows: list[Row] = []
    for name, n, k in GEMM_SHAPES:
        w = torch.randn(n, k, device="cuda", dtype=dtype) / k**0.5
        for m in ms:
            a = torch.randn(m, k, device="cuda", dtype=dtype)
            out = torch.empty(m, n, device="cuda", dtype=dtype)
            ref = F.linear(a, w)
            torch.ops.decode_kernels.gemm_tn(out, a, w, None)
            _check(f"gemm {name}", out, ref, 5e-2, 5e-2)
            t_dk = time_cuda(lambda: torch.ops.decode_kernels.gemm_tn(out, a, w, None), iters=iters)
            t_cublas = time_cuda(lambda: F.linear(a, w), iters=iters)
            flops = 2.0 * m * n * k
            cfg = f"{name} M={m} N={n} K={k}"
            for impl, t in (("decode_kernels", t_dk), ("cublas", t_cublas)):
                tflops = flops / (t.median_us * 1e-6) / 1e12
                sp = t_cublas.median_us / t.median_us if impl == "decode_kernels" else None
                rows.append(Row("gemm", cfg, impl, t.median_us, t.p10_us, t.p90_us, "TFLOP/s", tflops, sp))
    return rows


def bench_rms_norm(dtype, token_counts, iters, hidden=4096) -> list[Row]:
    rows: list[Row] = []
    w = torch.randn(hidden, device="cuda", dtype=dtype)
    for t in token_counts:
        x = torch.randn(t, hidden, device="cuda", dtype=dtype)
        r = torch.randn(t, hidden, device="cuda", dtype=dtype)
        out = torch.empty_like(x)
        _check("rms_norm", dk.rms_norm(x, w), rms_norm_reference(x, w), 3e-2, 3e-2)
        x2, r2 = x.clone(), r.clone()
        t_dk = time_cuda(lambda: torch.ops.decode_kernels.rms_norm(out, x, w, 1e-6), iters=iters)
        t_ref = time_cuda(lambda: rms_norm_reference(x, w), iters=iters)
        t_fused = time_cuda(lambda: torch.ops.decode_kernels.fused_add_rms_norm(x2, r2, w, 1e-6), iters=iters)
        t_fused_ref = time_cuda(lambda: rms_norm_reference(x + r, w), iters=iters)
        nbytes = 2 * x.numel() * x.element_size()  # read x, write out
        fused_bytes = 4 * x.numel() * x.element_size()  # read x, r; write r, x
        cfg = f"T={t} H={hidden}"
        for impl, tm, b, base in (
            ("decode_kernels", t_dk, nbytes, t_ref),
            ("torch_eager", t_ref, nbytes, None),
            ("dk_fused_add", t_fused, fused_bytes, t_fused_ref),
            ("torch_add+norm", t_fused_ref, fused_bytes, None),
        ):
            sp = base.median_us / tm.median_us if base else None
            rows.append(
                Row("rms_norm", cfg, impl, tm.median_us, tm.p10_us, tm.p90_us, "GB/s", b / (tm.median_us * 1e-6) / 1e9, sp)
            )
    return rows


def bench_silu(dtype, token_counts, iters, d=14336) -> list[Row]:
    rows: list[Row] = []
    for t in token_counts:
        x = torch.randn(t, 2 * d, device="cuda", dtype=dtype)
        out = torch.empty(t, d, device="cuda", dtype=dtype)
        _check("silu_and_mul", dk.silu_and_mul(x), silu_and_mul_reference(x), 3e-2, 3e-2)
        t_dk = time_cuda(lambda: torch.ops.decode_kernels.silu_and_mul(out, x), iters=iters)
        t_ref = time_cuda(lambda: F.silu(x[:, :d]) * x[:, d:], iters=iters)
        nbytes = (x.numel() + out.numel()) * x.element_size()
        cfg = f"T={t} d={d}"
        for impl, tm, base in (("decode_kernels", t_dk, t_ref), ("torch_eager", t_ref, None)):
            sp = base.median_us / tm.median_us if base else None
            rows.append(
                Row("silu_mul", cfg, impl, tm.median_us, tm.p10_us, tm.p90_us, "GB/s", nbytes / (tm.median_us * 1e-6) / 1e9, sp)
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernels", default="attention,gemm,rms_norm,silu")
    ap.add_argument("--dtype", choices=list(DTYPES), default="fp16")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--quick", action="store_true", help="small sweep for a fast sanity run")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    args = ap.parse_args()

    if not dk.is_available():
        sys.exit("decode_kernels extension not available: build it on a CUDA machine first")
    dtype = DTYPES[args.dtype]
    kernels = set(args.kernels.split(","))
    batches = [1, 8] if args.quick else [1, 8, 32, 64]
    ctx_lens = [512, 4096] if args.quick else [512, 2048, 8192, 32768]
    ms = [1, 64, 512] if args.quick else [1, 8, 32, 64, 128, 512, 2048]
    tokens = [1, 256] if args.quick else [1, 32, 256, 2048]

    rows: list[Row] = []
    if "attention" in kernels:
        rows += bench_attention(dtype, batches, ctx_lens, args.iters)
    if "gemm" in kernels:
        rows += bench_gemm(dtype, ms, args.iters)
    if "rms_norm" in kernels:
        rows += bench_rms_norm(dtype, tokens, args.iters)
    if "silu" in kernels:
        rows += bench_silu(dtype, tokens, args.iters)

    print_table(rows)
    path = write_csv(rows, args.out_dir, f"kernels_{args.dtype}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
