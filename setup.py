"""Build the CUDA extension.

    pip install -e . --no-build-isolation

Target GPUs default to Ampere, Ada and Hopper; override with e.g.
    TORCH_CUDA_ARCH_LIST="8.9" pip install -e . --no-build-isolation
"""

import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0")

setup(
    packages=find_packages(include=["decode_kernels", "decode_kernels.*"]),
    ext_modules=[
        CUDAExtension(
            name="decode_kernels._C",
            sources=["csrc/module.cpp", "csrc/ops.cu"],
            # No -std flag: PyTorch's build helper picks the C++ standard its
            # headers need (C++17 for older releases, C++20 for 2.10+).
            extra_compile_args={
                "cxx": ["-O3"],
                # -lineinfo keeps source-line mapping for Nsight Compute.
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
