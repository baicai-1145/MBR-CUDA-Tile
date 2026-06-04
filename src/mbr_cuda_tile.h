#pragma once

#include "tensor.h"
#include <vector>

namespace cudasep::mbr_tile {

void split_qkv_heads_rotary(const Tensor& qkv, int heads, int dim_head,
                            const Tensor& cos_freqs, const Tensor& sin_freqs,
                            Tensor& q, Tensor& k, Tensor& v);

Tensor gather_freqs_fold_complex(const Tensor& stft_repr, const Tensor& freq_indices);

bool residual_bf16_enabled();

bool bias_bf16_enabled();

bool linear_bkn_long_enabled();

bool linear_bkn_ffn_long_enabled();

Tensor apply_gates_and_merge_heads(const Tensor& attn, const Tensor& gates,
                                   int heads, int dim_head);

Tensor rms_norm(const Tensor& x, const Tensor& gamma, float scale);

Tensor scaled_dot_product_attention(const Tensor& q,
                                    const Tensor& k,
                                    const Tensor& v,
                                    float scale);

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias);

Tensor linear_bkn(const Tensor& x, const Tensor& weight, const Tensor& weight_bkn,
                  const Tensor& bias);

Tensor linear_gelu(const Tensor& x, const Tensor& weight, const Tensor& bias);

Tensor linear_gelu_bkn(const Tensor& x, const Tensor& weight, const Tensor& weight_bkn,
                       const Tensor& bias);

bool try_feedforward_fused(const Tensor& x,
                           const Tensor& linear1_w,
                           const Tensor& linear1_b,
                           const Tensor& linear2_w,
                           const Tensor& linear2_b,
                           Tensor& out);

void launch_ffn12_fused256_cutile(int gelu_mode,
                                  bool full_bf16,
                                  bool split2_output,
                                  const Tensor& x,
                                  const Tensor& linear1_w,
                                  const Tensor& linear1_b,
                                  const Tensor& linear2_w,
                                  const Tensor& linear2_b,
                                  Tensor& out);

void launch_time_attention1301_split_tail_cutile(const Tensor& q,
                                                 const Tensor& k,
                                                 const Tensor& v,
                                                 Tensor& out,
                                                 int64_t bh,
                                                 float scale,
                                                 bool use_k32,
                                                 bool use_tail_q32,
                                                 bool use_exp2);

void launch_time_attention1301_full_cutile(const Tensor& q,
                                           const Tensor& k,
                                           const Tensor& v,
                                           Tensor& out,
                                           int64_t bh,
                                           float scale,
                                           int qrows,
                                           int ktile);

Tensor linear_no_bias(const Tensor& x, const Tensor& weight);

Tensor linear_no_bias_bkn(const Tensor& x, const Tensor& weight, const Tensor& weight_bkn);

Tensor linear_no_bias_bf16_output(const Tensor& x, const Tensor& weight);

void linear_qkv_rotary_bf16_output(const Tensor& x, const Tensor& weight,
                                   int heads, int dim_head,
                                   const Tensor& cos_freqs, const Tensor& sin_freqs,
                                   Tensor& q, Tensor& k, Tensor& v);

void linear_qkv_rotary_bf16_output_bkn(const Tensor& x,
                                       const Tensor& weight,
                                       const Tensor& weight_bkn,
                                       int heads,
                                       int dim_head,
                                       const Tensor& cos_freqs,
                                       const Tensor& sin_freqs,
                                       Tensor& q,
                                       Tensor& k,
                                       Tensor& v);

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
