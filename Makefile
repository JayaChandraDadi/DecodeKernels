.PHONY: install test test-cpu bench bench-quick profile parity e2e compile-check

install:            ## build the CUDA extension in place
	pip install -e ".[test]" --no-build-isolation

test:               ## all tests (GPU tests skip without a GPU)
	pytest -q

test-cpu:           ## reference + split-K math tests only
	pytest -q tests/test_reference.py

bench:              ## full kernel sweep -> results/
	python benchmarks/bench_kernels.py

bench-quick:
	python benchmarks/bench_kernels.py --quick

profile:            ## Nsight Compute + Nsight Systems reports
	bash benchmarks/profile_ncu.sh

parity:             ## vLLM output parity, baseline vs DecodeKernels
	python benchmarks/vllm_parity.py

e2e:                ## vLLM decode throughput, baseline vs DecodeKernels
	python benchmarks/vllm_e2e.py

compile-check:      ## compile all kernels for sm_80/89/90 without a GPU
	bash tools/compile_check.sh
