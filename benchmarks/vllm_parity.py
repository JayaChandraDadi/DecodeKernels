"""Output parity: vLLM with its default kernels vs vLLM with DecodeKernels.

    python benchmarks/vllm_parity.py --model Qwen/Qwen2.5-0.5B-Instruct

Runs the same greedy generations twice in separate processes (baseline, then
DECODE_KERNELS_VLLM=1) and compares them token by token:

* exact-match rate of the generated token sequences,
* position of the first differing token for prompts that diverge,
* mean / max |delta log-prob| of the chosen tokens over the shared prefix,
* how many attention / norm / activation calls actually took the
  DecodeKernels path (so a silent fallback cannot look like a pass).

fp16/bf16 kernels with different reduction orders are not bit-identical, so
a late divergence on an otherwise matching greedy sequence is expected; the
log-prob deltas show how close the two runs are before that point. Prompts
include long ones (> 512 tokens) so the split-K path is exercised.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROMPTS = [
    "Explain how a paged KV cache works in an LLM inference server.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "What are the main differences between TCP and UDP?",
    "Summarize the causes of the French Revolution in three sentences.",
    "Translate to French: The weather is lovely today and we are going to the park.",
    "List five prime numbers greater than 100 and explain how you checked them.",
    "Describe the water cycle to a ten-year-old.",
    "What does a GPU tensor core do, and why does it speed up matrix multiplication?",
]

LONG_PASSAGE = (
    "Large language model inference has two phases. During prefill the model processes the whole prompt at once, "
    "which is compute bound. During decode it generates one token at a time, re-reading the weights and the key-value "
    "cache for every step, which is memory-bandwidth bound. Serving systems therefore batch many requests together, "
    "store the cache in fixed-size blocks to avoid fragmentation, and use specialised attention kernels. "
)


def build_prompts(n: int) -> list[str]:
    prompts = []
    for i in range(n):
        if i % 4 == 3:
            # ~800-2000 tokens: forces the split-K path during decode.
            reps = 8 + (i % 3) * 6
            prompts.append(LONG_PASSAGE * reps + "\nQuestion: what are the two phases, and which one is memory bound?")
        else:
            prompts.append(PROMPTS[i % len(PROMPTS)])
    return prompts


def run_child(args) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        enforce_eager=True,
        seed=0,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=1)
    outputs = llm.generate(build_prompts(args.num_prompts), params)
    records = []
    for o in outputs:
        c = o.outputs[0]
        lps = [lp[tok].logprob for tok, lp in zip(c.token_ids, c.logprobs)] if c.logprobs else []
        records.append({"prompt_tokens": len(o.prompt_token_ids), "token_ids": list(c.token_ids), "logprobs": lps})

    stats = {}
    if os.environ.get("DECODE_KERNELS_VLLM") == "1":
        from decode_kernels import vllm_patch

        stats = vllm_patch.stats()
    Path(args.child_out).write_text(json.dumps({"records": records, "stats": stats}))


def compare(base: dict, dk: dict) -> dict:
    exact, first_diffs, deltas = 0, [], []
    for b, d in zip(base["records"], dk["records"]):
        bt, dt = b["token_ids"], d["token_ids"]
        n = min(len(bt), len(dt))
        diff_at = next((i for i in range(n) if bt[i] != dt[i]), None)
        if diff_at is None and len(bt) == len(dt):
            exact += 1
        else:
            first_diffs.append(diff_at if diff_at is not None else n)
        upto = diff_at if diff_at is not None else n
        deltas += [abs(x - y) for x, y in zip(b["logprobs"][:upto], d["logprobs"][:upto])]
    total = len(base["records"])
    return {
        "prompts": total,
        "exact_match": exact,
        "exact_match_rate": exact / total if total else 0.0,
        "first_divergence_positions": first_diffs,
        "mean_abs_logprob_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "max_abs_logprob_delta": max(deltas) if deltas else 0.0,
        "decode_kernels_stats": dk["stats"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ap.add_argument("--ops", default="attention,rms_norm,silu_and_mul")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--child", choices=["baseline", "decode_kernels"], help=argparse.SUPPRESS)
    ap.add_argument("--child-out", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        run_child(args)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for mode in ("baseline", "decode_kernels"):
        out = Path(args.out_dir) / f".parity_{mode}.json"
        env = dict(os.environ)
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # keep the engine in-process so stats are readable
        env["DECODE_KERNELS_VLLM"] = "1" if mode == "decode_kernels" else "0"
        env["DECODE_KERNELS_OPS"] = args.ops
        cmd = [sys.executable, __file__, "--child", mode, "--child-out", str(out)] + [
            a for a in sys.argv[1:] if not a.startswith("--child")
        ]
        print(f"== running {mode}")
        subprocess.run(cmd, env=env, check=True)
        results[mode] = json.loads(out.read_text())
        out.unlink()

    summary = compare(results["baseline"], results["decode_kernels"])
    summary.update({"model": args.model, "dtype": args.dtype, "max_tokens": args.max_tokens, "ops": args.ops})
    path = Path(args.out_dir) / f"vllm_parity_{args.model.split('/')[-1]}_{args.dtype}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["decode_kernels_stats"].get("attention_kernel_calls", 0) == 0 and "attention" in args.ops:
        print("WARNING: the DecodeKernels attention path was never taken; check the eligibility conditions")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
