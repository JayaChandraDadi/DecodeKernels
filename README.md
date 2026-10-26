# DecodeKernels

CUDA C++ kernels for the decode phase of LLM inference, packaged as PyTorch
custom ops and wired into vLLM through a plugin:

| Kernel | What it does | Baseline it is measured against |
|---|---|---|
| `paged_attention_decode` | One query token per sequence attending over a paged KV cache, with flash-decoding split-K for long contexts | FlashInfer `BatchDecodeWithPagedKVCacheWrapper` |
| `gemm_tn` | `C = A·Bᵀ (+ bias)` on tensor cores (WMMA), the shape of every `nn.Linear` | cuBLAS via `torch.nn.functional.linear` |
| `rms_norm`, `fused_add_rms_norm` | RMSNorm and residual-add + RMSNorm in one pass | PyTorch eager |
| `silu_and_mul` | SwiGLU activation `silu(x[:, :d]) * x[:, d:]` | PyTorch eager |

All kernels support fp16 and bf16 and target sm_80+ (A100, A10/L4/L40/RTX 30–40,
H100). The norm and activation ops use vLLM's exact signatures, and the
attention op reads vLLM's KV cache in place.

## Status

* All kernels compile for sm_80, sm_89 and sm_90 with no register spills
  (`tools/compile_check.sh`, also run in CI).
* The reference implementations and the split-K merge math are tested on CPU
  (`tests/test_reference.py`, run in CI).
* GPU correctness tests, kernel benchmarks, Nsight profiles and the vLLM
  parity/throughput runs need an NVIDIA GPU. Their outputs go to `results/`;
  **no performance numbers are claimed in this README until they are measured
  there.**

## Layout

```
csrc/
  common.cuh            fp16/bf16 conversion, vector types, warp/block reductions
  paged_attention.cuh   paged-attention decode (single pass + split-K + reduce)
  gemm_wmma.cuh         tensor-core GEMM with cp.async double buffering
  norm_act.cuh          rms_norm, fused_add_rms_norm, silu_and_mul
  ops.cu                PyTorch bindings: checks, dtype dispatch, TORCH_LIBRARY
  module.cpp            Python module entry point
decode_kernels/
  ops.py                Python wrappers, workspace allocation, torch.compile fakes
  reference.py          PyTorch reference implementations (CPU or GPU)
  testing.py            paged KV-cache builders (NHD, HND and vLLM layouts)
  vllm_plugin.py        vLLM general-plugin entry point
  vllm_patch.py         swaps the kernels into vLLM (with strict fallbacks)
tests/                  CPU reference tests + GPU kernel tests
benchmarks/             CUDA-event benchmarks, Nsight script, vLLM parity and e2e
tools/                  GPU-less compile check with ptxas stats
```

## Kernel design

### Paged-attention decode

Inputs follow vLLM / FlashInfer: `q [num_seqs, Hq, D]`, K and V caches
`[num_blocks, block_size, Hkv, D]` with **arbitrary strides on the first three
dims** (so vLLM's interleaved `K|V` cache and FlashInfer's NHD/HND layouts are
read without a copy), an int32 block table and int32 sequence lengths.

* **GQA-aware blocks.** A thread block owns one KV head and computes every
  query head in its group (up to 8), so each K/V row is read from HBM once per
  group instead of once per query head. Decode attention is memory-bound, so
  this is the main lever.
* **Coalesced, vectorised loads.** Each warp reads one full K or V row per
  step: lane *i* loads `D/32` contiguous elements in a single vector load (8
  bytes for D=128), and the dot product finishes with a `__shfl_xor_sync`
  reduction.
* **Flash-decoding split-K.** Sequences longer than 512 tokens are split into
  512-token partitions processed by separate blocks. Each partition writes its
  normalised partial output plus its max logit *mᵢ* and exp-sum *sᵢ*; a reduce
  kernel merges them with `M = max mᵢ`, `wᵢ = sᵢ·exp(mᵢ − M)`,
  `out = Σ wᵢ·oᵢ / Σ wᵢ`. At batch 1 with a long context, this turns a handful
  of blocks into hundreds, so all SMs share the KV read.
* **Single pass for short batches.** When every sequence fits in one partition,
  the kernel writes the output directly and the reduce launch is skipped.
* **Graph-safe.** Workspaces can be preallocated, and lengths are read on the
  device, so a captured CUDA graph tracks changing sequence lengths (up to the
  captured `max_seq_len`).

`decode_kernels/reference.py::split_k_attention_reference` re-implements the
partition/merge math in PyTorch; `tests/test_reference.py` checks it against
dense attention on CPU, including partition boundaries (511/512/513 tokens),
empty sequences and strided layouts.

### Tensor-core GEMM (`C = A·Bᵀ`)

* WMMA 16×16×16 fragments with fp32 accumulation.
* **Two tile shapes:** 128×128 (8 warps, 4×2 warp grid, 32×64 per warp) for
  prefill-sized M, and a 32×128 *skinny* tile (4 warps) for M ≤ 64, because in
  decode M is the batch size and a 128-row tile would spend most of its work on
  padding.
* **cp.async double buffering:** the next 32-wide K tile streams into shared
  memory (16-byte copies, zero-fill at the edges) while the current tile is
  multiplied.
* Shared-memory rows are padded by 8 elements to avoid bank conflicts on the
  fragment loads.
* The epilogue stages each fp32 fragment through shared memory, adds the
  optional bias and converts with bounds checks, so any M and N work
  (K must be a multiple of 8).

### RMSNorm, fused add + RMSNorm, SiLU-and-mul

One block per token row with 16-byte vector loads (scalar fallback for odd
widths), fp32 math and a warp-shuffle + shared-memory block reduction.
`fused_add_rms_norm` writes the updated residual and the normalised output in
a single kernel (vLLM semantics: the residual is rounded to the model dtype
before the norm).

### Register usage (ptxas, sm_80)

From `tools/compile_check.sh`; no kernel spills registers.

| Kernel | Registers / thread |
|---|---|
| `paged_attention_kernel<half, 128, split>` | 54 |
| `paged_attention_kernel<half, 256, split>` | 88 |
| `paged_attention_reduce_kernel` | 32 |
| `gemm_tn_wmma_kernel` 128×128 | 123–124 |
| `gemm_tn_wmma_kernel` 32×128 (skinny) | 96 |
| `rms_norm_kernel` (vectorised) | 30 |
| `silu_and_mul_kernel` (vectorised) | 29 |

## Running it on a GPU

Any sm_80+ GPU works; a rented L4, A10G or A100 for a few hours is enough for
everything below.

```bash
git clone https://github.com/JayaChandraDadi/DecodeKernels && cd DecodeKernels
pip install torch          # CUDA build matching the machine's driver
pip install -e ".[test,bench]" --no-build-isolation
# TORCH_CUDA_ARCH_LIST="8.9" ... to build for one GPU only (faster)

pytest -q                                   # CPU reference tests + GPU kernel tests
python benchmarks/bench_kernels.py --quick  # sanity sweep
python benchmarks/bench_kernels.py          # full sweep -> results/kernels_*.csv
bash benchmarks/profile_ncu.sh              # Nsight Compute + Nsight Systems reports
```

Each benchmark first checks the kernel's output against the baseline, then
times it with CUDA events (median of 100 runs, L2 flushed before each run) and
reports GB/s (memory-bound kernels) or TFLOP/s (GEMM) next to the baseline.

## vLLM integration

Installing the package registers a vLLM general plugin that does nothing until
it is switched on:

```bash
pip install -e ".[vllm]" --no-build-isolation
DECODE_KERNELS_VLLM=1 python -c "
from vllm import LLM, SamplingParams
llm = LLM('Qwen/Qwen2.5-0.5B-Instruct', enforce_eager=True)
print(llm.generate(['Hello'], SamplingParams(max_tokens=32))[0].outputs[0].text)"
```

`decode_kernels/vllm_patch.py` (written against vLLM 0.32) swaps in:

* **Attention:** in `FlashAttentionImpl.forward`, pure-decode batches (one
  query token per request) call `paged_attention_decode` on vLLM's own KV cache
  and block table. Anything else (prefill or mixed batches, FP8 KV cache,
  sliding window, ALiBi, soft-capping, sinks, context parallelism, CUDA-graph
  capture) falls back to FlashAttention.
* **RMSNorm / fused add + RMSNorm** in `RMSNorm.forward_cuda`.
* **SiLU-and-mul** in `SiluAndMul.forward_cuda`.

`DECODE_KERNELS_OPS=attention,rms_norm,silu_and_mul` selects which to swap.
`vllm_patch.stats()` counts how often each patched op used the kernel or fell
back, so a silent fallback cannot pass for a result.

Two scripts compare vLLM with and without the kernels, each mode in its own
process:

```bash
python benchmarks/vllm_parity.py --model Qwen/Qwen2.5-0.5B-Instruct   # greedy outputs, token by token
python benchmarks/vllm_e2e.py    --model Qwen/Qwen2.5-0.5B-Instruct   # decode tokens/s at batch 1, 8, 32
```

The parity script reports exact-match rate, the first differing position and
log-prob deltas over the matching prefix, and includes prompts longer than 512
tokens so the split-K path runs. Bitwise equality is not expected: the kernels
reduce in a different order than FlashAttention, so greedy outputs can diverge
late in a sequence.

## Limitations

* Head dims 64, 128 and 256; GQA groups up to 8; fp16/bf16 KV cache (no FP8).
* The GEMM uses WMMA, not Hopper `wgmma`/TMA, so cuBLAS should win on H100 for
  large shapes; the interesting comparisons are the skinny decode shapes.
* The vLLM patch targets vLLM 0.32's FlashAttention backend and runs attention
  only outside full CUDA graphs (`enforce_eager=True` or piecewise graphs).

## License

MIT
