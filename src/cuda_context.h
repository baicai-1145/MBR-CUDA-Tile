#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace cudasep {

extern bool g_quantize_fp16;

class CudaContext {
public:
    static CudaContext& instance();
    cublasHandle_t cublas() { return cublas_; }
    cudaStream_t stream() { return stream_; }

private:
    CudaContext();
    ~CudaContext();
    cublasHandle_t cublas_;
    cudaStream_t stream_;
};

}  // namespace cudasep
