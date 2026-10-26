"""vLLM general plugin entry point (see pyproject.toml).

vLLM imports this in every process it starts (API server, engine core,
workers). It is a no-op unless DECODE_KERNELS_VLLM=1, so installing the
package never changes vLLM's behaviour by itself.

    DECODE_KERNELS_VLLM=1                       enable
    DECODE_KERNELS_OPS=attention,rms_norm,silu_and_mul   choose which ops to swap (default: all)
"""

import os


def register() -> None:
    if os.environ.get("DECODE_KERNELS_VLLM", "0") != "1":
        return
    from .vllm_patch import apply

    ops = os.environ.get("DECODE_KERNELS_OPS", "attention,rms_norm,silu_and_mul")
    apply([o.strip() for o in ops.split(",") if o.strip()])
