"""Timing helpers shared by the benchmark scripts."""

from __future__ import annotations

import csv
import datetime as _dt
import os
import platform
import statistics
from dataclasses import asdict, dataclass

import torch

_L2_FLUSH: torch.Tensor | None = None


def flush_l2() -> None:
    """Overwrite a 256 MB buffer so every timed call starts with a cold L2,
    which is what a decode step sees after the rest of the layer ran."""
    global _L2_FLUSH
    if _L2_FLUSH is None:
        _L2_FLUSH = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    _L2_FLUSH.zero_()


@dataclass
class Timing:
    median_us: float
    p10_us: float
    p90_us: float


def time_cuda(fn, warmup: int = 10, iters: int = 100, cold_l2: bool = True) -> Timing:
    """Per-call GPU time from CUDA events (median and 10th/90th percentiles)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if cold_l2:
            flush_l2()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    us = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return Timing(
        median_us=statistics.median(us),
        p10_us=us[int(0.1 * (len(us) - 1))],
        p90_us=us[int(0.9 * (len(us) - 1))],
    )


@dataclass
class Row:
    kernel: str
    config: str
    impl: str
    median_us: float
    p10_us: float
    p90_us: float
    metric: str
    value: float
    speedup_vs_baseline: float | None = None


def environment() -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "sm": f"{props.major}.{props.minor}",
        "num_sms": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
    }


def write_csv(rows: list[Row], out_dir: str, prefix: str) -> str:
    env = environment()
    os.makedirs(out_dir, exist_ok=True)
    gpu = env["gpu"].replace(" ", "_").replace("/", "_")
    path = os.path.join(out_dir, f"{prefix}_{gpu}_{env['date'][:10]}.csv")
    with open(path, "w", newline="") as f:
        f.write("# " + ", ".join(f"{k}={v}" for k, v in env.items()) + "\n")
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    return path


def print_table(rows: list[Row]) -> None:
    header = f"{'kernel':<10} {'config':<38} {'impl':<18} {'median_us':>10} {'metric':>10} {'value':>9} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        sp = f"{r.speedup_vs_baseline:.2f}x" if r.speedup_vs_baseline else ""
        print(
            f"{r.kernel:<10} {r.config:<38} {r.impl:<18} {r.median_us:>10.1f} {r.metric:>10} {r.value:>9.1f} {sp:>8}"
        )
