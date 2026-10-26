#!/usr/bin/env bash
# Profile the kernels with Nsight Compute (per-kernel metrics) and Nsight
# Systems (timeline). Reports land in results/profiles/.
#
#   bash benchmarks/profile_ncu.sh            # all kernels
#   KERNEL=paged_attention bash benchmarks/profile_ncu.sh
#
# Open the .ncu-rep in Nsight Compute and look at: Memory Workload Analysis
# (DRAM throughput vs peak, L1/L2 hit rates, shared-memory bank conflicts),
# Warp State Statistics (stall reasons), Occupancy, and Source Counters (the
# kernels are built with -lineinfo, so stalls map back to CUDA source lines).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/profiles

KERNEL="${KERNEL:-paged_attention|gemm_tn|rms_norm|silu_and_mul}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# Full metric set for a few launches of each matching kernel (skip warm-up).
ncu --set full \
    --kernel-name "regex:${KERNEL}" \
    --launch-skip 5 --launch-count 6 \
    --import-source yes \
    -o "results/profiles/ncu_${STAMP}" \
    python benchmarks/bench_kernels.py --quick --iters 5

# Timeline of one quick benchmark pass.
nsys profile --trace cuda,nvtx --force-overwrite true \
    -o "results/profiles/nsys_${STAMP}" \
    python benchmarks/bench_kernels.py --quick --iters 5

echo "reports in results/profiles/ (ncu_${STAMP}.ncu-rep, nsys_${STAMP}.nsys-rep)"
