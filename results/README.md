# Results

Benchmark CSVs, vLLM parity/throughput reports and Nsight profiles are written
here by the scripts in `benchmarks/`. Every CSV starts with a comment line
recording the GPU, compute capability, PyTorch and CUDA versions and the date.

Numbers are only committed after they have been measured on real hardware;
nothing in this directory is hand-written.

| File | Produced by |
|---|---|
| `kernels_<dtype>_<gpu>_<date>.csv` | `python benchmarks/bench_kernels.py` |
| `vllm_parity_<model>_<dtype>.json` | `python benchmarks/vllm_parity.py` |
| `vllm_e2e_<model>_<dtype>.csv` | `python benchmarks/vllm_e2e.py` |
| `profiles/*.ncu-rep`, `profiles/*.nsys-rep` | `bash benchmarks/profile_ncu.sh` (large; usually not committed) |
