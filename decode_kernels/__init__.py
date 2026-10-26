"""DecodeKernels: CUDA kernels for LLM decoding.

    import decode_kernels as dk
    out = dk.paged_attention_decode(q, k_cache, v_cache, block_tables, seq_lens)
"""

from .ops import (
    PARTITION_SIZE,
    alloc_workspace,
    fused_add_rms_norm,
    gemm,
    is_available,
    paged_attention_decode,
    rms_norm,
    silu_and_mul,
)

__all__ = [
    "PARTITION_SIZE",
    "alloc_workspace",
    "fused_add_rms_norm",
    "gemm",
    "is_available",
    "paged_attention_decode",
    "rms_norm",
    "silu_and_mul",
]
__version__ = "0.1.0"
