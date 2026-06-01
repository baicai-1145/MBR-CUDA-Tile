#pragma once

#include "tensor.h"
#include <vector>

namespace cudasep::mbr_tile {

void split_qkv_heads_rotary(const Tensor& qkv, int heads, int dim_head,
                            const Tensor& cos_freqs, const Tensor& sin_freqs,
                            Tensor& q, Tensor& k, Tensor& v);

Tensor gather_freqs_fold_complex(const Tensor& stft_repr, const Tensor& freq_indices);

Tensor apply_gates_and_merge_heads(const Tensor& attn, const Tensor& gates,
                                   int heads, int dim_head);

Tensor rms_norm(const Tensor& x, const Tensor& gamma, float scale);

Tensor scaled_dot_product_attention(const Tensor& q,
                                    const Tensor& k,
                                    const Tensor& v,
                                    float scale);

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias);

Tensor linear_gelu(const Tensor& x, const Tensor& weight, const Tensor& bias);

Tensor linear_no_bias(const Tensor& x, const Tensor& weight);

Tensor linear_sigmoid(const Tensor& x, const Tensor& weight, const Tensor& bias);

Tensor apply_mask_and_scatter(const Tensor& stft_repr,
                              const std::vector<Tensor>& stem_masks,
                              const Tensor& freq_indices,
                              const Tensor& bands_per_freq,
                              int64_t batch,
                              int64_t num_stems,
                              int64_t total_freq,
                              int64_t total_band_freqs,
                              int64_t frames,
                              int64_t audio_channels);

Tensor tanh_act(const Tensor& x);

Tensor glu_last_dim(const Tensor& x);

void zero_dc(Tensor& complex_spec);

}  // namespace cudasep::mbr_tile
