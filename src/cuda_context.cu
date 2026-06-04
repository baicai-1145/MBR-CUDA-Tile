#include "cuda_context.h"

#include "tensor.h"

#include <stdexcept>
#include <string>

namespace cudasep {

bool g_quantize_fp16 = false;
bool g_quantize_bf16 = false;

CudaContext::CudaContext() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

CudaContext::~CudaContext() {
    cudaStreamDestroy(stream_);
}

CudaContext& CudaContext::instance() {
    static CudaContext ctx;
    return ctx;
}

}  // namespace cudasep
