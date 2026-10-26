// Python module entry point. The ops themselves are registered with the
// PyTorch dispatcher in ops.cu (TORCH_LIBRARY); importing decode_kernels._C
// loads this shared library and makes torch.ops.decode_kernels.* available.
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "DecodeKernels: CUDA kernels for LLM decoding (registered under torch.ops.decode_kernels)";
}
