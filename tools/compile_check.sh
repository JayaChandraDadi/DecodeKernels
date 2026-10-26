#!/usr/bin/env bash
# Compile every kernel for sm_80 / sm_89 / sm_90 without PyTorch or a GPU and
# print per-kernel register, shared-memory and spill counts from ptxas.
#
#   bash tools/compile_check.sh               # uses nvcc from PATH
#   NVCC=/path/to/nvcc bash tools/compile_check.sh
#
# Without a CUDA toolkit:  pip install nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cuda-cccl
# then NVCC="$(python -c 'import nvidia, os; print(os.path.join(nvidia.__path__[0], "cu13/bin/nvcc"))')"
set -euo pipefail
cd "$(dirname "$0")/.."
NVCC="${NVCC:-nvcc}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

status=0
for arch in 80 89 90; do
  echo "== sm_${arch}"
  if ! "$NVCC" -std=c++17 -O3 -lineinfo -arch="sm_${arch}" -Xptxas -v \
      -c tools/compile_check.cu -o "$OUT/check_${arch}.o" 2> "$OUT/ptxas_${arch}.log"; then
    cat "$OUT/ptxas_${arch}.log"
    status=1
    continue
  fi
  # One line per kernel: name, registers, spills.
  python3 - "$OUT/ptxas_${arch}.log" <<'PY'
import re, subprocess, sys
text = open(sys.argv[1]).read()
blocks = re.split(r"Compiling entry function '([^']+)'", text)[1:]
for name, body in zip(blocks[::2], blocks[1::2]):
    try:
        pretty = subprocess.run(["c++filt", name], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        pretty = name
    pretty = re.sub(r"\(.*", "", pretty).replace("dk::detail::", "")
    regs = re.search(r"Used (\d+) registers", body)
    spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", body)
    print(f"  {pretty:<70} regs={regs.group(1) if regs else '?':>4}  spills={spill.group(1) if spill else '?'}/{spill.group(2) if spill else '?'}")
    if spill and (int(spill.group(1)) or int(spill.group(2))):
        print("    WARNING: register spills")
PY
done
exit $status
