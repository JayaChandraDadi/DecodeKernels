"""End-to-end decode throughput in vLLM, baseline vs DecodeKernels.

    python benchmarks/vllm_e2e.py --model Qwen/Qwen2.5-0.5B-Instruct

For each batch size, generates a fixed number of tokens from fixed-length
prompts (ignore_eos, greedy) and reports output tokens/s and time per output
token. Each mode runs in its own process; both use enforce_eager=True so the
only difference is which kernels run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_child(args) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        enforce_eager=True,
        seed=0,
        max_model_len=args.input_len + args.output_len + 16,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True)
    rows = []
    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        prompts = [
            TokensPrompt(prompt_token_ids=[(1000 + 37 * i + j) % 20000 + 100 for j in range(args.input_len)])
            for i in range(bs)
        ]
        llm.generate(prompts[: min(bs, 2)], SamplingParams(max_tokens=8, ignore_eos=True))  # warm-up
        best = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            outs = llm.generate(prompts, params, use_tqdm=False)
            elapsed = time.perf_counter() - t0
            best = elapsed if best is None else min(best, elapsed)
        out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        rows.append(
            {
                "batch_size": bs,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "elapsed_s": best,
                "output_tok_per_s": out_tokens / best,
                "ms_per_output_token": 1000.0 * best / args.output_len,
            }
        )
    Path(args.child_out).write_text(json.dumps(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-sizes", default="1,8,32")
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
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
        out = Path(args.out_dir) / f".e2e_{mode}.json"
        env = dict(os.environ)
        env["DECODE_KERNELS_VLLM"] = "1" if mode == "decode_kernels" else "0"
        env["DECODE_KERNELS_OPS"] = args.ops
        cmd = [sys.executable, __file__, "--child", mode, "--child-out", str(out)] + [
            a for a in sys.argv[1:] if not a.startswith("--child")
        ]
        print(f"== running {mode}")
        subprocess.run(cmd, env=env, check=True)
        results[mode] = json.loads(out.read_text())
        out.unlink()

    path = Path(args.out_dir) / f"vllm_e2e_{args.model.split('/')[-1]}_{args.dtype}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "input_len", "output_len", "mode", "output_tok_per_s", "ms_per_output_token", "speedup"])
        print(f"{'batch':>6} {'baseline tok/s':>15} {'dk tok/s':>10} {'speedup':>8}")
        for b, d in zip(results["baseline"], results["decode_kernels"]):
            speedup = d["output_tok_per_s"] / b["output_tok_per_s"]
            for mode, r in (("baseline", b), ("decode_kernels", d)):
                writer.writerow(
                    [r["batch_size"], r["input_len"], r["output_len"], mode, f"{r['output_tok_per_s']:.1f}",
                     f"{r['ms_per_output_token']:.3f}", f"{speedup:.3f}" if mode == "decode_kernels" else ""]
                )
            print(f"{b['batch_size']:>6} {b['output_tok_per_s']:>15.1f} {d['output_tok_per_s']:>10.1f} {speedup:>7.2f}x")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
