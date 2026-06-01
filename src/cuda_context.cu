#include "cuda_context.h"

#include "tensor.h"

#include <stdexcept>
#include <string>

namespace cudasep {

bool g_quantize_fp16 = false;
bool g_quantize_bf16 = false;

CudaContext::CudaContext() {
    cublasStatus_t status = cublasCreate(&cublas_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasCreate failed with status " + std::to_string((int)status));
    }
    CUDA_CHECK(cudaStreamCreate(&stream_));
    cublasSetStream(cublas_, stream_);
    cublasSetMathMode(cublas_, CUBLAS_TF32_TENSOR_OP_MATH);
}

CudaContext::~CudaContext() {
    cublasDestroy(cublas_);
    cudaStreamDestroy(stream_);
}

CudaContext& CudaContext::instance() {
    static CudaContext ctx;
    return ctx;
}

}  // namespace cudasep
