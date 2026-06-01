#pragma once
#include "tensor.h"
#include <cublas_v2.h>

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

namespace ops {

Tensor tanh_act(const Tensor& x);
Tensor complex_mul(const Tensor& a, const Tensor& b);

Tensor rms_norm(const Tensor& x, const Tensor& gamma, float scale);

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias);
Tensor linear_no_bias(const Tensor& x, const Tensor& weight);
Tensor linear_gelu(const Tensor& x, const Tensor& weight, const Tensor& bias);
Tensor linear_sigmoid(const Tensor& x, const Tensor& weight, const Tensor& bias);

Tensor scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                     float scale = 0.0f, float dropout = 0.0f);

Tensor stft(const Tensor& signal, int n_fft, int hop_length, int win_length,
            const Tensor& window, bool center = true, bool normalized = false);
Tensor istft(const Tensor& complex_spec, int n_fft, int hop_length, int win_length,
             const Tensor& window, int64_t length = -1, bool center = true, bool normalized = false);
Tensor hann_window(int size);

void apply_rotary_emb(Tensor& q, Tensor& k, const Tensor& cos_freqs, const Tensor& sin_freqs);

void scatter_add(Tensor& dest, int dim, const Tensor& indices, const Tensor& src);

Tensor glu(const Tensor& x, int dim = -1);
Tensor index_fill(const Tensor& x, int dim, int64_t index, float value);

void overlap_add(Tensor& dest, const Tensor& src, const Tensor& window, int64_t offset);
void weight_accumulate(Tensor& weight_sum, const Tensor& window, int64_t offset);
void normalize_by_weights(Tensor& data, const Tensor& weight_sum);

} // namespace ops
} // namespace cudasep
