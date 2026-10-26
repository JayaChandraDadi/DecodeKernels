// Compiles every kernel template for fp16 and bf16 without PyTorch, so the
// CUDA code can be checked (and its register / shared-memory usage reported
// with -Xptxas -v) on machines without a GPU. Used by tools/compile_check.sh
// and CI.
#include "../csrc/gemm_wmma.cuh"
#include "../csrc/norm_act.cuh"
#include "../csrc/paged_attention.cuh"

namespace dk {

template cudaError_t launch_paged_attention_decode<__half>(const PagedAttnParams&, int, int, cudaStream_t);
template cudaError_t launch_paged_attention_decode<__nv_bfloat16>(const PagedAttnParams&, int, int, cudaStream_t);

template cudaError_t launch_gemm_tn<__half>(const __half*, const __half*, const __half*, __half*, int, int, int, int,
                                            int, int, cudaStream_t);
template cudaError_t launch_gemm_tn<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*,
                                                   const __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, int, int,
                                                   cudaStream_t);

template cudaError_t launch_rms_norm<__half>(__half*, const __half*, const __half*, int, int, int64_t, int64_t, float,
                                             cudaStream_t);
template cudaError_t launch_rms_norm<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int,
                                                    int, int64_t, int64_t, float, cudaStream_t);

template cudaError_t launch_fused_add_rms_norm<__half>(__half*, __half*, const __half*, int, int, int64_t, int64_t,
                                                       float, cudaStream_t);
template cudaError_t launch_fused_add_rms_norm<__nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
                                                              int, int, int64_t, int64_t, float, cudaStream_t);

template cudaError_t launch_silu_and_mul<__half>(__half*, const __half*, int, int, int64_t, int64_t, cudaStream_t);
template cudaError_t launch_silu_and_mul<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, int, int, int64_t,
                                                        int64_t, cudaStream_t);

}  // namespace dk
