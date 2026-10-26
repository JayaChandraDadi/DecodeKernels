"""Swap DecodeKernels into vLLM.

Written against vLLM 0.32 (FlashAttention backend, v1 engine). Three
substitutions, each with a strict eligibility check that falls back to the
original vLLM code path for anything it does not handle:

* ``FlashAttentionImpl.forward`` -- pure-decode batches (one query token per
  request) run ``paged_attention_decode`` directly on vLLM's KV cache. vLLM
  stores K and V interleaved as [num_blocks, num_kv_heads, block_size,
  2 * head_dim]; the kernel reads the strided K and V views in place, so no
  copy is made. vLLM writes the new token's K/V into the cache before
  ``forward`` runs (``do_kv_cache_update``), so ``seq_lens`` already includes it.
  Prefill and mixed batches, quantized KV caches, sliding windows, ALiBi,
  soft-capping, attention sinks, context parallelism and CUDA-graph capture
  all go to FlashAttention.
* ``RMSNorm.forward_cuda`` -- rms_norm / fused_add_rms_norm.
* ``SiluAndMul.forward_cuda`` -- silu_and_mul.

Custom-op forward_cuda methods only run when vLLM uses its custom ops (eager
mode, or ``custom_ops`` enabled in the compilation config); attention runs
eagerly under the default piecewise CUDA graphs. The benchmark scripts use
``enforce_eager=True`` so both runs execute the same code paths.
"""

from __future__ import annotations

import logging

import torch

from . import ops as dk_ops

logger = logging.getLogger("decode_kernels.vllm")

_STATS = {
    "attention_kernel_calls": 0,
    "attention_fallback_calls": 0,
    "rms_norm_kernel_calls": 0,
    "rms_norm_fallback_calls": 0,
    "silu_and_mul_kernel_calls": 0,
    "silu_and_mul_fallback_calls": 0,
}
_APPLIED: set[str] = set()
_HALF_TYPES = (torch.float16, torch.bfloat16)


def stats() -> dict[str, int]:
    """How often each patched op used the DecodeKernels path vs fell back."""
    return dict(_STATS)


def reset_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
def _attention_eligible(impl, query, kv_cache, attn_metadata, output, output_scale, output_block_scale) -> bool:
    from vllm.v1.attention.backend import AttentionType

    if attn_metadata is None or output_scale is not None or output_block_scale is not None:
        return False
    if torch.cuda.is_current_stream_capturing():
        # The kernel's single-pass / split-K choice depends on the host-side
        # max_seq_len, which a captured graph would freeze. Leave captured
        # graphs to FlashAttention.
        return False
    if impl.attn_type != AttentionType.DECODER or attn_metadata.max_query_len != 1:
        return False
    if getattr(impl, "dcp_world_size", 1) > 1:
        return False
    if str(impl.kv_cache_dtype) not in ("auto", "float16", "bfloat16") or kv_cache.dtype != query.dtype:
        return False
    if query.dtype not in _HALF_TYPES:
        return False
    if impl.alibi_slopes is not None or impl.logits_soft_cap:
        return False
    if impl.sliding_window not in (None, (-1, -1)):
        return False
    if getattr(impl, "sinks", None) is not None:
        return False
    if impl.head_size not in dk_ops.SUPPORTED_HEAD_DIMS:
        return False
    if impl.num_heads % impl.num_kv_heads or impl.num_heads // impl.num_kv_heads > dk_ops.MAX_GQA_GROUP:
        return False
    if getattr(attn_metadata, "causal", True) is not True:
        return False
    if getattr(attn_metadata, "mm_prefix_query_range_tensor", None) is not None:
        return False
    if getattr(attn_metadata, "rswa_prefix_lens", None) is not None:
        return False
    n = attn_metadata.num_actual_tokens
    if attn_metadata.seq_lens.shape[0] < n or attn_metadata.block_table.shape[0] < n:
        return False
    if not output.is_contiguous():
        return False
    return True


def _patch_attention() -> None:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    original = FlashAttentionImpl.forward

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if _attention_eligible(self, query, kv_cache, attn_metadata, output, output_scale, output_block_scale):
            n = attn_metadata.num_actual_tokens
            # (B, H, N, 2*D) -> two strided (B, N, H, D) views, as vLLM does.
            key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
            q = query[:n].reshape(n, self.num_heads, self.head_size)
            out = output[:n].view(n, self.num_heads, self.head_size)
            dk_ops.paged_attention_decode(
                q,
                key_cache,
                value_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                scale=self.scale,
                max_seq_len=attn_metadata.max_seq_len,
                out=out,
            )
            _STATS["attention_kernel_calls"] += 1
            return output
        _STATS["attention_fallback_calls"] += 1
        return original(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)

    forward.__wrapped__ = original  # type: ignore[attr-defined]
    FlashAttentionImpl.forward = forward


# ---------------------------------------------------------------------------
# RMSNorm and SiluAndMul
# ---------------------------------------------------------------------------
def _patch_rms_norm() -> None:
    import vllm.envs as envs
    from vllm.model_executor.layers.layernorm import RMSNorm

    original = RMSNorm.forward_cuda

    def forward_cuda(self, x, residual=None):
        eligible = (
            not envs.VLLM_BATCH_INVARIANT
            and not torch.compiler.is_compiling()
            and self.variance_size_override is None
            and self.has_weight
            and x.dtype in _HALF_TYPES
            and self.weight.dtype == x.dtype
            and x.shape[-1] == self.hidden_size
            and x.stride(-1) == 1
            and (residual is None or (residual.dtype == x.dtype and residual.shape == x.shape))
        )
        if not eligible:
            _STATS["rms_norm_fallback_calls"] += 1
            return original(self, x, residual)
        _STATS["rms_norm_kernel_calls"] += 1
        if residual is None:
            return dk_ops.rms_norm(x, self.weight.data, self.variance_epsilon)
        return dk_ops.fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)

    forward_cuda.__wrapped__ = original  # type: ignore[attr-defined]
    RMSNorm.forward_cuda = forward_cuda


def _patch_silu_and_mul() -> None:
    from vllm.model_executor.layers.activation import SiluAndMul

    original = SiluAndMul.forward_cuda

    def forward_cuda(self, x):
        if x.dtype in _HALF_TYPES and not torch.compiler.is_compiling() and x.stride(-1) == 1:
            _STATS["silu_and_mul_kernel_calls"] += 1
            return dk_ops.silu_and_mul(x)
        _STATS["silu_and_mul_fallback_calls"] += 1
        return original(self, x)

    forward_cuda.__wrapped__ = original  # type: ignore[attr-defined]
    SiluAndMul.forward_cuda = forward_cuda


_PATCHERS = {
    "attention": _patch_attention,
    "rms_norm": _patch_rms_norm,
    "silu_and_mul": _patch_silu_and_mul,
}


def apply(ops: list[str] | None = None) -> None:
    """Install the patches (idempotent)."""
    if not dk_ops.is_available():
        logger.warning("decode_kernels: CUDA extension unavailable; vLLM left unpatched")
        return
    for name in ops or list(_PATCHERS):
        if name not in _PATCHERS:
            raise ValueError(f"unknown op {name!r}; choose from {sorted(_PATCHERS)}")
        if name in _APPLIED:
            continue
        _PATCHERS[name]()
        _APPLIED.add(name)
    logger.info("decode_kernels: patched vLLM ops %s", sorted(_APPLIED))
