#pragma once

#include <cuda_runtime.h>

namespace cudasep {

extern bool g_quantize_fp16;
extern bool g_quantize_bf16;

class CudaContext {
public:
    static CudaContext& instance();
    cudaStream_t stream() { return stream_; }

private:
    CudaContext();
    ~CudaContext();
    cudaStream_t stream_;
};

}  // namespace cudasep
