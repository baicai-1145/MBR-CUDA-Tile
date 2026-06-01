#include "mbr_cuda_tile.h"

#include "cuda_tile.h"
#include "cuda_context.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudasep::mbr_tile {
namespace {

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kTile = 256;
constexpr int kRmsTile = 1024;
constexpr int kSoftmaxTile = 2048;
using I64Tile = ct::tile<long long, ct::shape<kTile>>;
using F32Tile = ct::tile<float, ct::shape<kTile>>;
using RmsI64Tile = ct::tile<long long, ct::shape<kRmsTile>>;
using RmsF32Tile = ct::tile<float, ct::shape<kRmsTile>>;
using SoftmaxI64Tile = ct::tile<long long, ct::shape<kSoftmaxTile>>;
using SoftmaxF16Tile = ct::tile<__half, ct::shape<kSoftmaxTile>>;

static inline int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

void check_cublas(cublasStatus_t status, const char* op) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(op) + " failed with cuBLAS status " +
                                 std::to_string((int)status));
    }
}

__tile_global__ void split_qkv_heads_rotary_kernel(const float* __restrict__ qkv,
                                                   const float* __restrict__ cos_f,
                                                   const float* __restrict__ sin_f,
                                                   float* __restrict__ q,
                                                   float* __restrict__ k,
                                                   float* __restrict__ v,
                                                   long long total,
                                                   int heads,
                                                   int n_tokens,
                                                   int dim_head) {
    qkv = ct::assume_aligned(qkv, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    int half_dim = dim_head / 2;
    auto i = idx % half_dim;
    auto n = (idx / half_dim) % n_tokens;
    auto h = (idx / ((long long)half_dim * n_tokens)) % heads;
    auto b = idx / ((long long)half_dim * n_tokens * heads);
    auto pair_d = i * 2;

    auto qkv_base = (b * n_tokens + n) * (3LL * heads * dim_head);
    auto head_offset = h * dim_head + pair_d;
    auto out_base = ((b * heads + h) * n_tokens + n) * dim_head;

    auto c = ct::load_masked(cos_f + n * half_dim + i, in_bounds);
    auto s = ct::load_masked(sin_f + n * half_dim + i, in_bounds);

    auto q0 = ct::load_masked(qkv + qkv_base + head_offset, in_bounds);
    auto q1 = ct::load_masked(qkv + qkv_base + head_offset + 1, in_bounds);
    auto k0 = ct::load_masked(qkv + qkv_base + (long long)heads * dim_head + head_offset, in_bounds);
    auto k1 = ct::load_masked(qkv + qkv_base + (long long)heads * dim_head + head_offset + 1, in_bounds);
    auto v0 = ct::load_masked(qkv + qkv_base + 2LL * heads * dim_head + head_offset, in_bounds);
    auto v1 = ct::load_masked(qkv + qkv_base + 2LL * heads * dim_head + head_offset + 1, in_bounds);

    ct::store_masked(q + out_base + pair_d, q0 * c - q1 * s, in_bounds);
    ct::store_masked(q + out_base + pair_d + 1, q0 * s + q1 * c, in_bounds);
    ct::store_masked(k + out_base + pair_d, k0 * c - k1 * s, in_bounds);
    ct::store_masked(k + out_base + pair_d + 1, k0 * s + k1 * c, in_bounds);
    ct::store_masked(v + out_base + pair_d, v0, in_bounds);
    ct::store_masked(v + out_base + pair_d + 1, v1, in_bounds);
}

__tile_global__ void gather_freqs_fold_complex_kernel(const float* __restrict__ stft,
                                                      const int64_t* __restrict__ freq_indices,
                                                      float* __restrict__ out,
                                                      long long total,
                                                      int total_freq,
                                                      int total_band_freqs,
                                                      int frames) {
    stft = ct::assume_aligned(stft, 16_ic);
    freq_indices = ct::assume_aligned(freq_indices, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    auto complex_part = idx % 2;
    auto band_f = (idx / 2) % total_band_freqs;
    auto t = (idx / (2LL * total_band_freqs)) % frames;
    auto b = idx / (2LL * total_band_freqs * frames);
    auto freq = ct::load_masked(freq_indices + band_f, in_bounds);
    auto src_idx = ((b * (long long)total_freq + freq) * frames + t) * 2 + complex_part;
    auto values = ct::load_masked(stft + src_idx, in_bounds);
    ct::store_masked(out + idx, values, in_bounds);
}

__tile_global__ void apply_gates_and_merge_heads_kernel(const float* __restrict__ attn,
                                                        const float* __restrict__ gates,
                                                        float* __restrict__ merged,
                                                        long long total,
                                                        int heads,
                                                        int n_tokens,
                                                        int dim_head) {
    attn = ct::assume_aligned(attn, 16_ic);
    gates = ct::assume_aligned(gates, 16_ic);
    merged = ct::assume_aligned(merged, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    auto d = idx % dim_head;
    auto n = (idx / dim_head) % n_tokens;
    auto h = (idx / ((long long)dim_head * n_tokens)) % heads;
    auto b = idx / ((long long)dim_head * n_tokens * heads);

    auto gate_idx = (b * n_tokens + n) * heads + h;
    auto merged_idx = (b * n_tokens + n) * ((long long)heads * dim_head) + h * dim_head + d;

    auto values = ct::load_masked(attn + idx, in_bounds) *
                  ct::load_masked(gates + gate_idx, in_bounds);
    ct::store_masked(merged + merged_idx, values, in_bounds);
}

__tile_global__ void rms_norm_kernel(const float* __restrict__ x,
                                     const float* __restrict__ gamma,
                                     float* __restrict__ out,
                                     int dim,
                                     float scale) {
    x = ct::assume_aligned(x, 16_ic);
    gamma = ct::assume_aligned(gamma, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    long long row = (long long)ct::bid().x;
    RmsI64Tile d = ct::iota<RmsI64Tile>();
    auto in_bounds = d < dim;
    auto row_offset = row * dim;

    auto values = ct::load_masked(x + row_offset + d, in_bounds);
    auto zeros = values * 0.0f;
    auto sum_sq = ct::sum<0>(ct::select(in_bounds, values * values, zeros));
    auto eps = sum_sq * 0.0f + 1.0e-12f;
    auto inv_norm = ct::rsqrt(sum_sq + eps);
    auto gamma_values = ct::load_masked(gamma + d, in_bounds);

    ct::store_masked(out + row_offset + d, values * inv_norm * gamma_values * scale, in_bounds);
}

__tile_global__ void scale_softmax_kernel(float* __restrict__ data,
                                          int cols,
                                          float scale) {
    data = ct::assume_aligned(data, 16_ic);

    long long row = (long long)ct::bid().x;
    SoftmaxI64Tile col = ct::iota<SoftmaxI64Tile>();
    auto in_bounds = col < cols;
    auto row_offset = row * cols;

    auto values = ct::load_masked(data + row_offset + col, in_bounds);
    auto zeros = values * 0.0f;
    auto scaled = ct::select(in_bounds, values * scale, zeros - 3.402823466e38f);
    auto row_max = ct::reduce_max<0>(scaled);
    auto exp_values = ct::select(in_bounds, ct::exp(scaled - row_max), zeros);
    auto denom = ct::sum<0>(exp_values);

    ct::store_masked(data + row_offset + col, exp_values / denom, in_bounds);
}

__tile_global__ void scale_softmax_to_half_kernel(const float* __restrict__ data,
                                                  __half* __restrict__ out,
                                                  int cols,
                                                  float scale) {
    data = ct::assume_aligned(data, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    long long row = (long long)ct::bid().x;
    SoftmaxI64Tile col = ct::iota<SoftmaxI64Tile>();
    auto in_bounds = col < cols;
    auto row_offset = row * cols;

    auto values = ct::load_masked(data + row_offset + col, in_bounds);
    auto zeros = values * 0.0f;
    auto scaled = ct::select(in_bounds, values * scale, zeros - 3.402823466e38f);
    auto row_max = ct::reduce_max<0>(scaled);
    auto exp_values = ct::select(in_bounds, ct::exp(scaled - row_max), zeros);
    auto denom = ct::sum<0>(exp_values);

    SoftmaxF16Tile half_values(exp_values / denom);
    ct::store_masked(out + row_offset + col, half_values, in_bounds);
}

__tile_global__ void add_bias_kernel(float* __restrict__ out,
                                     const float* __restrict__ bias,
                                     long long total,
                                     int out_features) {
    out = ct::assume_aligned(out, 16_ic);
    bias = ct::assume_aligned(bias, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(out + idx, in_bounds) +
                  ct::load_masked(bias + (idx % out_features), in_bounds);
    ct::store_masked(out + idx, values, in_bounds);
}

__tile_global__ void add_bias_sigmoid_kernel(float* __restrict__ out,
                                             const float* __restrict__ bias,
                                             long long total,
                                             int out_features) {
    out = ct::assume_aligned(out, 16_ic);
    bias = ct::assume_aligned(bias, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(out + idx, in_bounds) +
                  ct::load_masked(bias + (idx % out_features), in_bounds);
    auto sigmoid = 1.0f / (1.0f + ct::exp(-values));
    ct::store_masked(out + idx, sigmoid, in_bounds);
}

__tile_global__ void add_bias_gelu_kernel(float* __restrict__ out,
                                          const float* __restrict__ bias,
                                          long long total,
                                          int out_features) {
    out = ct::assume_aligned(out, 16_ic);
    bias = ct::assume_aligned(bias, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto x = ct::load_masked(out + idx, in_bounds) +
             ct::load_masked(bias + (idx % out_features), in_bounds);

    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto sign = ct::select(x < zero, zero - one, one);
    auto ax = ct::abs(x);
    auto t = one / (one + 0.3275911f * ax);
    auto poly = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t -
                  0.284496736f) * t + 0.254829592f) * t;
    auto erf_approx = sign * (one - poly * ct::exp(-(ax * ax)));
    auto gelu = 0.5f * x * (one + erf_approx);
    ct::store_masked(out + idx, gelu, in_bounds);
}

__tile_global__ void apply_mask_and_scatter_kernel(const float* __restrict__ stft,
                                                   const float* __restrict__ mask0,
                                                   const float* __restrict__ mask1,
                                                   const int64_t* __restrict__ freq_indices,
                                                   const float* __restrict__ bands_per_freq,
                                                   float* __restrict__ out,
                                                   long long total,
                                                   int num_stems,
                                                   int total_band_freqs,
                                                   int frames,
                                                   int total_freq,
                                                   int audio_channels,
                                                   int freq_bins) {
    stft = ct::assume_aligned(stft, 16_ic);
    mask0 = ct::assume_aligned(mask0, 16_ic);
    mask1 = ct::assume_aligned(mask1, 16_ic);
    freq_indices = ct::assume_aligned(freq_indices, 16_ic);
    bands_per_freq = ct::assume_aligned(bands_per_freq, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    auto complex_part = idx % 2;
    auto tmp_t = idx / 2;
    auto t = tmp_t % frames;
    auto tmp_band = tmp_t / frames;
    auto band_f = tmp_band % total_band_freqs;
    auto tmp_stem = tmp_band / total_band_freqs;
    auto stem = tmp_stem % num_stems;
    auto batch_idx = tmp_stem / num_stems;

    auto freq = ct::load_masked(freq_indices + band_f, in_bounds);
    auto denom = ct::load_masked(bands_per_freq + freq, in_bounds);
    auto eps = denom * 0.0f + 1.0e-8f;
    denom = ct::select(denom > eps, denom, eps);

    auto stft_base = ((batch_idx * total_freq + freq) * frames + t) * 2;
    auto mask_base = (batch_idx * frames + t) * (2LL * total_band_freqs) + band_f * 2;

    auto stft_r = ct::load_masked(stft + stft_base, in_bounds);
    auto stft_i = ct::load_masked(stft + stft_base + 1, in_bounds);
    auto stem0 = in_bounds && (stem == 0);
    auto stem1 = in_bounds && (stem == 1);
    auto mask0_r = ct::load_masked(mask0 + mask_base, stem0);
    auto mask0_i = ct::load_masked(mask0 + mask_base + 1, stem0);
    auto mask1_r = ct::load_masked(mask1 + mask_base, stem1);
    auto mask1_i = ct::load_masked(mask1 + mask_base + 1, stem1);
    auto mask_r = ct::select(stem == 0, mask0_r, mask1_r);
    auto mask_i = ct::select(stem == 0, mask0_i, mask1_i);

    auto value_r = (stft_r * mask_r - stft_i * mask_i) / denom;
    auto value_i = (stft_r * mask_i + stft_i * mask_r) / denom;
    auto value = ct::select(complex_part == 0, value_r, value_i);

    auto freq_bin = freq / audio_channels;
    auto channel = freq % audio_channels;
    auto outer = ((batch_idx * num_stems + stem) * audio_channels + channel);
    auto out_idx = ((outer * freq_bins + freq_bin) * frames + t) * 2 + complex_part;
    ct::atomic_add_masked<ct::memory_order::relaxed>(out + out_idx, value, in_bounds);
}

__tile_global__ void zero_dc_kernel(float* __restrict__ data,
                                    long long total,
                                    int freq_bins,
                                    int frames) {
    data = ct::assume_aligned(data, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    int inner_size = frames * 2;
    auto outer = idx / inner_size;
    auto inner = idx % inner_size;
    auto linear = outer * (long long)freq_bins * inner_size + inner;
    auto zero = ct::load_masked(data + linear, in_bounds) * 0.0f;
    ct::store_masked(data + linear, zero, in_bounds);
}

__tile_global__ void tanh_kernel(const float* __restrict__ x,
                                 float* __restrict__ out,
                                 long long total) {
    x = ct::assume_aligned(x, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = tanh(ct::load_masked(x + idx, in_bounds));
    ct::store_masked(out + idx, values, in_bounds);
}

__tile_global__ void glu_last_dim_kernel(const float* __restrict__ x,
                                         float* __restrict__ out,
                                         long long total,
                                         int half_dim) {
    x = ct::assume_aligned(x, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    auto d = idx % half_dim;
    auto row = idx / half_dim;
    auto base = row * (2LL * half_dim);
    auto first = ct::load_masked(x + base + d, in_bounds);
    auto second = ct::load_masked(x + base + half_dim + d, in_bounds);
    auto gate = 1.0f / (1.0f + exp(-second));
    ct::store_masked(out + idx, first * gate, in_bounds);
}

}  // namespace

void split_qkv_heads_rotary(const Tensor& qkv, int heads, int dim_head,
                            const Tensor& cos_freqs, const Tensor& sin_freqs,
                            Tensor& q, Tensor& k, Tensor& v) {
    if ((dim_head % 2) != 0) {
        throw std::runtime_error("mbr_tile::split_qkv_heads_rotary: dim_head must be even");
    }

    Tensor qkv_work = (qkv.dtype() == DType::Float32) ? qkv.contiguous() : qkv.to_f32().contiguous();
    Tensor cos_work = (cos_freqs.dtype() == DType::Float32) ? cos_freqs.contiguous() : cos_freqs.to_f32().contiguous();
    Tensor sin_work = (sin_freqs.dtype() == DType::Float32) ? sin_freqs.contiguous() : sin_freqs.to_f32().contiguous();

    int B = (int)qkv_work.size(0);
    int N = (int)qkv_work.size(1);
    q = Tensor::empty({(int64_t)B, (int64_t)heads, (int64_t)N, (int64_t)dim_head}, DType::Float32);
    k = Tensor::empty({(int64_t)B, (int64_t)heads, (int64_t)N, (int64_t)dim_head}, DType::Float32);
    v = Tensor::empty({(int64_t)B, (int64_t)heads, (int64_t)N, (int64_t)dim_head}, DType::Float32);

    long long total = (long long)B * heads * N * (dim_head / 2);
    split_qkv_heads_rotary_kernel<<<(int)ceildiv(total, kTile), 1>>>(
        qkv_work.data_f32(), cos_work.data_f32(), sin_work.data_f32(),
        q.data_f32(), k.data_f32(), v.data_f32(),
        total, heads, N, dim_head);
    CUDA_CHECK(cudaGetLastError());
}

Tensor gather_freqs_fold_complex(const Tensor& stft_repr, const Tensor& freq_indices) {
    Tensor stft_work = (stft_repr.dtype() == DType::Float32) ? stft_repr.contiguous() : stft_repr.to_f32().contiguous();
    if (stft_work.ndim() != 4 || stft_work.size(3) != 2) {
        throw std::runtime_error("mbr_tile::gather_freqs_fold_complex: expected [B, F, T, 2]");
    }

    int64_t batch = stft_work.size(0);
    int64_t total_freq = stft_work.size(1);
    int64_t frames = stft_work.size(2);
    int64_t total_band_freqs = freq_indices.numel();
    Tensor out = Tensor::empty({batch, frames, total_band_freqs * 2}, DType::Float32);

    long long total = out.numel();
    gather_freqs_fold_complex_kernel<<<(int)ceildiv(total, kTile), 1>>>(
        stft_work.data_f32(), freq_indices.data_i64(), out.data_f32(),
        total, (int)total_freq, (int)total_band_freqs, (int)frames);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor apply_gates_and_merge_heads(const Tensor& attn, const Tensor& gates,
                                   int heads, int dim_head) {
    int B = (int)attn.size(0);
    int N = (int)attn.size(2);
    Tensor merged = Tensor::empty({(int64_t)B, (int64_t)N, (int64_t)heads * dim_head}, DType::Float32);

    long long total = (long long)B * heads * N * dim_head;
    apply_gates_and_merge_heads_kernel<<<(int)ceildiv(total, kTile), 1>>>(
        attn.data_f32(), gates.data_f32(), merged.data_f32(),
        total, heads, N, dim_head);
    CUDA_CHECK(cudaGetLastError());
    return merged;
}

Tensor rms_norm(const Tensor& x, const Tensor& gamma, float scale) {
    Tensor xf = (x.dtype() == DType::Float32) ? x.contiguous() : x.to_f32().contiguous();
    Tensor gf = (gamma.dtype() == DType::Float32) ? gamma.contiguous() : gamma.to_f32().contiguous();
    int dim = (int)xf.size(xf.ndim() - 1);
    if (dim > kRmsTile) {
        throw std::runtime_error("mbr_tile::rms_norm: last dimension exceeds tile size");
    }

    Tensor out = Tensor::empty(xf.shape(), DType::Float32);
    long long rows = xf.numel() / dim;
    rms_norm_kernel<<<(int)rows, 1>>>(xf.data_f32(), gf.data_f32(), out.data_f32(), dim, scale);
    CUDA_CHECK(cudaGetLastError());
    return (x.dtype() == DType::Float16) ? out.to_f16() : out;
}

Tensor scaled_dot_product_attention(const Tensor& q,
                                    const Tensor& k,
                                    const Tensor& v,
                                    float scale) {
    Tensor qf = (q.dtype() == DType::Float32) ? q.contiguous() : q.to_f32().contiguous();
    Tensor kf = (k.dtype() == DType::Float32) ? k.contiguous() : k.to_f32().contiguous();
    Tensor vf = (v.dtype() == DType::Float32) ? v.contiguous() : v.to_f32().contiguous();

    int64_t B = qf.size(0);
    int64_t H = qf.size(1);
    int64_t N = qf.size(2);
    int64_t D = qf.size(3);
    int64_t N_k = kf.size(2);
    if (N_k > kSoftmaxTile) {
        throw std::runtime_error("mbr_tile::scaled_dot_product_attention: sequence exceeds tile size");
    }
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt((float)D);
    }
    bool use_fp16_value_gemm = g_quantize_fp16 && D <= 128;

    int64_t BH = B * H;
    Tensor scores = Tensor::empty({BH, N, N_k}, DType::Float32);
    Tensor scores_half;
    Tensor vh;
    if (use_fp16_value_gemm) {
        scores_half = Tensor::empty({BH, N, N_k}, DType::Float16);
        vh = vf.to_f16().contiguous();
    }
    Tensor out = Tensor::empty({BH, N, D}, DType::Float32);

    float alpha = 1.0f;
    float beta = 0.0f;
    check_cublas(cublasGemmStridedBatchedEx(CudaContext::instance().cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)N_k, (int)N, (int)D,
        &alpha,
        kf.data_f32(), CUDA_R_32F, (int)D, (long long)(N_k * D),
        qf.data_f32(), CUDA_R_32F, (int)D, (long long)(N * D),
        &beta,
        scores.data_f32(), CUDA_R_32F, (int)N_k, (long long)(N * N_k),
        (int)BH,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP), "attention QK GEMM");

    if (use_fp16_value_gemm) {
        scale_softmax_to_half_kernel<<<(int)(BH * N), 1>>>(
            scores.data_f32(), scores_half.data_f16(), (int)N_k, scale);
    } else {
        scale_softmax_kernel<<<(int)(BH * N), 1>>>(scores.data_f32(), (int)N_k, scale);
    }
    CUDA_CHECK(cudaGetLastError());

    if (use_fp16_value_gemm) {
        check_cublas(cublasGemmStridedBatchedEx(CudaContext::instance().cublas(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            (int)D, (int)N, (int)N_k,
            &alpha,
            vh.data_ptr(), CUDA_R_16F, (int)D, (long long)(N_k * D),
            scores_half.data_ptr(), CUDA_R_16F, (int)N_k, (long long)(N * N_k),
            &beta,
            out.data_f32(), CUDA_R_32F, (int)D, (long long)(N * D),
            (int)BH,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP), "attention AV FP16 GEMM");
    } else {
        check_cublas(cublasGemmStridedBatchedEx(CudaContext::instance().cublas(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            (int)D, (int)N, (int)N_k,
            &alpha,
            vf.data_f32(), CUDA_R_32F, (int)D, (long long)(N_k * D),
            scores.data_f32(), CUDA_R_32F, (int)N_k, (long long)(N * N_k),
            &beta,
            out.data_f32(), CUDA_R_32F, (int)D, (long long)(N * D),
            (int)BH,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP), "attention AV FP32 GEMM");
    }

    Tensor result = out.reshape({B, H, N, D});
    return (q.dtype() == DType::Float16) ? result.to_f16() : result;
}

Tensor linear_gemm_f32_output(const Tensor& x,
                              const Tensor& weight,
                              int64_t& total_batch,
                              int64_t& out_features,
                              std::vector<int64_t>& out_shape) {
    int64_t in_features = weight.size(1);
    out_features = weight.size(0);

    if (weight.dtype() == DType::Float16) {
        Tensor xh = (x.dtype() == DType::Float16) ? x.contiguous() : x.to_f16().contiguous();
        Tensor wh = weight.contiguous();
        total_batch = xh.numel() / in_features;
        out_shape = xh.shape();
        out_shape.back() = out_features;

        Tensor out = Tensor::empty({total_batch, out_features}, DType::Float32);
        float alpha = 1.0f;
        float beta = 0.0f;
        check_cublas(cublasGemmEx(CudaContext::instance().cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            (int)out_features, (int)total_batch, (int)in_features,
            &alpha,
            wh.data_ptr(), CUDA_R_16F, (int)in_features,
            xh.data_ptr(), CUDA_R_16F, (int)in_features,
            &beta,
            out.data_f32(), CUDA_R_32F, (int)out_features,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP), "linear FP16 GEMM");
        return out;
    }

    Tensor xf = (x.dtype() == DType::Float32) ? x.contiguous() : x.to_f32().contiguous();
    Tensor wf = (weight.dtype() == DType::Float32) ? weight.contiguous() : weight.to_f32().contiguous();
    total_batch = xf.numel() / in_features;
    out_shape = xf.shape();
    out_shape.back() = out_features;

    Tensor out = Tensor::empty({total_batch, out_features}, DType::Float32);
    float alpha = 1.0f;
    float beta = 0.0f;
    check_cublas(cublasGemmEx(CudaContext::instance().cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)out_features, (int)total_batch, (int)in_features,
        &alpha,
        wf.data_f32(), CUDA_R_32F, (int)in_features,
        xf.data_f32(), CUDA_R_32F, (int)in_features,
        &beta,
        out.data_f32(), CUDA_R_32F, (int)out_features,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP), "linear FP32 GEMM");
    return out;
}

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    int64_t total_batch = 0;
    int64_t out_features = 0;
    std::vector<int64_t> out_shape;
    Tensor out = linear_gemm_f32_output(x, weight, total_batch, out_features, out_shape);
    Tensor bf = (bias.dtype() == DType::Float32) ? bias.contiguous() : bias.to_f32().contiguous();

    long long total = total_batch * out_features;
    add_bias_kernel<<<(int)ceildiv(total, kTile), 1>>>(
        out.data_f32(), bf.data_f32(), total, (int)out_features);
    CUDA_CHECK(cudaGetLastError());

    Tensor result = out.reshape(out_shape);
    return (weight.dtype() == DType::Float16) ? result :
           ((x.dtype() == DType::Float16) ? result.to_f16() : result);
}

Tensor linear_gelu(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    int64_t total_batch = 0;
    int64_t out_features = 0;
    std::vector<int64_t> out_shape;
    Tensor out = linear_gemm_f32_output(x, weight, total_batch, out_features, out_shape);
    Tensor bf = (bias.dtype() == DType::Float32) ? bias.contiguous() : bias.to_f32().contiguous();

    long long total = total_batch * out_features;
    add_bias_gelu_kernel<<<(int)ceildiv(total, kTile), 1>>>(
        out.data_f32(), bf.data_f32(), total, (int)out_features);
    CUDA_CHECK(cudaGetLastError());

    Tensor result = out.reshape(out_shape);
    return (x.dtype() == DType::Float16) ? result.to_f16() : result;
}

Tensor linear_no_bias(const Tensor& x, const Tensor& weight) {
    int64_t total_batch = 0;
    int64_t out_features = 0;
    std::vector<int64_t> out_shape;
    Tensor out = linear_gemm_f32_output(x, weight, total_batch, out_features, out_shape);
    Tensor result = out.reshape(out_shape);
    return (weight.dtype() == DType::Float16) ? result :
           ((x.dtype() == DType::Float16) ? result.to_f16() : result);
}

Tensor linear_sigmoid(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    int64_t total_batch = 0;
    int64_t out_features = 0;
    std::vector<int64_t> out_shape;
    Tensor out = linear_gemm_f32_output(x, weight, total_batch, out_features, out_shape);
    Tensor bf = (bias.dtype() == DType::Float32) ? bias.contiguous() : bias.to_f32().contiguous();

    long long total = total_batch * out_features;
    add_bias_sigmoid_kernel<<<(int)ceildiv(total, kTile), 1>>>(
        out.data_f32(), bf.data_f32(), total, (int)out_features);
    CUDA_CHECK(cudaGetLastError());

    Tensor result = out.reshape(out_shape);
    return (x.dtype() == DType::Float16) ? result.to_f16() : result;
}

Tensor apply_mask_and_scatter(const Tensor& stft_repr,
                              const std::vector<Tensor>& stem_masks,
                              const Tensor& freq_indices,
                              const Tensor& bands_per_freq,
                              int64_t batch,
                              int64_t num_stems,
                              int64_t total_freq,
                              int64_t total_band_freqs,
                              int64_t frames,
                              int64_t audio_channels) {
    if ((int64_t)stem_masks.size() != num_stems) {
        throw std::runtime_error("mbr_tile::apply_mask_and_scatter: stem mask count mismatch");
    }
    if (num_stems < 1 || num_stems > 2) {
        throw std::runtime_error("mbr_tile::apply_mask_and_scatter: expected one or two stems");
    }

    std::vector<Tensor> mask_work;
    mask_work.reserve(stem_masks.size());
    for (size_t i = 0; i < stem_masks.size(); ++i) {
        Tensor mask = (stem_masks[i].dtype() == DType::Float32) ? stem_masks[i].contiguous()
                                                                : stem_masks[i].to_f32().contiguous();
        if (mask.ndim() != 3 || mask.size(0) != batch || mask.size(1) != frames ||
            mask.size(2) != total_band_freqs * 2) {
            throw std::runtime_error("mbr_tile::apply_mask_and_scatter: expected [B, T, total_band_freqs * 2]");
        }
        mask_work.push_back(mask);
    }
    const float* mask0 = mask_work[0].data_f32();
    const float* mask1 = (num_stems == 2) ? mask_work[1].data_f32() : mask_work[0].data_f32();

    if (audio_channels < 1 || total_freq % audio_channels != 0) {
        throw std::runtime_error("mbr_tile::apply_mask_and_scatter: invalid audio channel count");
    }

    int64_t freq_bins = total_freq / audio_channels;
    Tensor out = Tensor::zeros({batch * num_stems * audio_channels, freq_bins, frames, 2});
    long long total = (long long)batch * num_stems * total_band_freqs * frames * 2;
    apply_mask_and_scatter_kernel<<<(int)ceildiv(total, kTile), 1>>>(
        stft_repr.data_f32(), mask0, mask1, freq_indices.data_i64(),
        bands_per_freq.data_f32(), out.data_f32(), total, (int)num_stems,
        (int)total_band_freqs, (int)frames, (int)total_freq,
        (int)audio_channels, (int)freq_bins);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

void zero_dc(Tensor& complex_spec) {
    long long outer = complex_spec.numel() / (complex_spec.size(1) * complex_spec.size(2) * 2);
    long long total = outer * complex_spec.size(2) * 2;
    zero_dc_kernel<<<(int)ceildiv(total, kTile), 1>>>(
        complex_spec.data_f32(), total, (int)complex_spec.size(1), (int)complex_spec.size(2));
    CUDA_CHECK(cudaGetLastError());
}

Tensor tanh_act(const Tensor& x) {
    Tensor xf = (x.dtype() == DType::Float32) ? x.contiguous() : x.to_f32().contiguous();
    Tensor out = Tensor::empty(xf.shape(), DType::Float32);
    long long total = xf.numel();
    tanh_kernel<<<(int)ceildiv(total, kTile), 1>>>(xf.data_f32(), out.data_f32(), total);
    CUDA_CHECK(cudaGetLastError());
    return (x.dtype() == DType::Float16) ? out.to_f16() : out;
}

Tensor glu_last_dim(const Tensor& x) {
    Tensor xf = (x.dtype() == DType::Float32) ? x.contiguous() : x.to_f32().contiguous();
    int ndim = xf.ndim();
    int64_t full_dim = xf.size(ndim - 1);
    if ((full_dim % 2) != 0) {
        throw std::runtime_error("glu_last_dim: last dimension must be even");
    }

    std::vector<int64_t> out_shape = xf.shape();
    out_shape.back() = full_dim / 2;

    Tensor out = Tensor::empty(out_shape, DType::Float32);
    long long total = out.numel();
    glu_last_dim_kernel<<<(int)ceildiv(total, kTile), 1>>>(
        xf.data_f32(), out.data_f32(), total, (int)(full_dim / 2));
    CUDA_CHECK(cudaGetLastError());
    return (x.dtype() == DType::Float16) ? out.to_f16() : out;
}

}  // namespace cudasep::mbr_tile
