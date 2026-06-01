// model_mel_band_roformer.cu - Complete MelBandRoformer inference implementation.
//
// This implements the MelBandRoformer model for music source separation,
// mirroring the Python forward pass exactly.

#include "model_mel_band_roformer.h"
#include "mbr_cuda_tile.h"
#include "stft_cuda_tile.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace cudasep {

// ============================================================================
// MBRConfig
// ============================================================================

MBRConfig MBRConfig::from_json(const JsonValue& j) {
    MBRConfig c;
    c.dim                    = j.get_int("dim", 384);
    c.depth                  = j.get_int("depth", 6);
    c.stereo                 = j.get_bool("stereo", true);
    c.num_stems              = j.get_int("num_stems", 1);
    c.time_transformer_depth = j.get_int("time_transformer_depth", 1);
    c.freq_transformer_depth = j.get_int("freq_transformer_depth", 1);
    c.num_bands              = j.get_int("num_bands", 60);
    c.dim_head               = j.get_int("dim_head", 64);
    c.heads                  = j.get_int("heads", 8);
    c.mask_estimator_depth   = j.get_int("mask_estimator_depth", 2);
    c.mlp_expansion_factor   = j.get_int("mlp_expansion_factor", 4);
    c.zero_dc                = j.get_bool("zero_dc", true);
    c.sample_rate            = j.get_int("sample_rate", 44100);
    c.stft_n_fft             = j.get_int("stft_n_fft", 2048);
    c.stft_hop_length        = j.get_int("stft_hop_length", 441);
    c.stft_win_length        = j.get_int("stft_win_length", 2048);
    c.stft_normalized        = j.get_bool("stft_normalized", false);
    c.match_input_audio_length = j.get_bool("match_input_audio_length", false);
    c.dim_freqs_in           = j.get_int("dim_freqs_in", 1025);
    c.skip_connection        = j.get_bool("skip_connection", false);
    return c;
}

// ============================================================================
// load
// ============================================================================

void MelBandRoformer::load(const ModelWeights& weights) {
    // 1. Parse config
    cfg_ = MBRConfig::from_json(weights.config());

    // 2. Load precomputed mel band data (from converter)
    if (!weights.has("__precomputed__.freq_indices")) {
        throw std::runtime_error("MelBandRoformer .csm must include precomputed mel band data");
    }

    // Use precomputed mel bands from the .csm file.
    freq_indices_gpu_ = weights.get("__precomputed__.freq_indices");
    Tensor num_freqs_t = weights.get("__precomputed__.num_freqs_per_band");
    Tensor num_bands_f = weights.get("__precomputed__.num_bands_per_freq");

    int num_bands = cfg_.num_bands;
    int audio_channels = cfg_.stereo ? 2 : 1;
    int num_fft_bins = cfg_.stft_n_fft / 2 + 1;

    std::vector<int64_t> nfpb_cpu(num_bands);
    num_freqs_t.copy_to_cpu(nfpb_cpu.data(), num_bands * sizeof(int64_t));
    num_freqs_per_band_ = std::move(nfpb_cpu);

    int total_freq = num_fft_bins * audio_channels;
    std::vector<int64_t> nbpf_cpu(num_fft_bins);
    num_bands_f.copy_to_cpu(nbpf_cpu.data(), num_fft_bins * sizeof(int64_t));

    num_bands_per_freq_.resize(total_freq);
    if (cfg_.stereo) {
        for (int f = 0; f < num_fft_bins; f++) {
            num_bands_per_freq_[f * 2] = nbpf_cpu[f];
            num_bands_per_freq_[f * 2 + 1] = nbpf_cpu[f];
        }
    } else {
        num_bands_per_freq_ = std::move(nbpf_cpu);
    }

    band_freq_dims_.resize(num_bands);
    for (int b = 0; b < num_bands; b++) {
        band_freq_dims_[b] = 2 * num_freqs_per_band_[b] * audio_channels;
    }

    int64_t n_idx = freq_indices_gpu_.numel();
    freq_indices_.resize(n_idx);
    freq_indices_gpu_.copy_to_cpu(freq_indices_.data(), n_idx * sizeof(int64_t));

    std::vector<float> nbpf_float(num_bands_per_freq_.begin(), num_bands_per_freq_.end());
    num_bands_per_freq_gpu_ = Tensor::from_cpu_f32(nbpf_float.data(), {(int64_t)total_freq});

    std::cout << "[MelBandRoformer] Using precomputed mel bands from .csm file" << std::endl;

    // 3. Create STFT window
    stft_window_ = stft_tile::hann_window(cfg_.stft_win_length);

    int dim = cfg_.dim;

    // 4. Load band_split weights
    band_split_layers_.resize(num_bands);
    for (int b = 0; b < num_bands; b++) {
        std::string prefix = "band_split.to_features." + std::to_string(b);
        band_split_layers_[b].norm_gamma = weights.get(prefix + ".0.gamma");
        band_split_layers_[b].linear_w   = weights.get(prefix + ".1.weight");
        band_split_layers_[b].linear_b   = weights.get(prefix + ".1.bias");
    }

    // 5. Load transformer weights
    // Model has cfg_.depth depth blocks, each with time and freq transformers.
    // No linear_transformer in standard configs (linear_transformer_depth=0).
    // Weight naming:
    //   layers.{d}.0 = time_transformer
    //   layers.{d}.1 = freq_transformer
    // Each transformer has:
    //   layers.{l}.0 = Attention
    //   layers.{l}.1 = FeedForward
    //   norm.gamma   = final RMSNorm

    depth_blocks_.resize(cfg_.depth);
    for (int d = 0; d < cfg_.depth; d++) {
        auto load_transformer = [&](const std::string& tprefix, int depth,
                                    TransformerLayerWeights& tw) {
            tw.attn_layers.resize(depth);
            tw.ff_layers.resize(depth);
            for (int l = 0; l < depth; l++) {
                std::string lprefix = tprefix + ".layers." + std::to_string(l);
                // Attention: lprefix.0.*
                std::string aprefix = lprefix + ".0";
                tw.attn_layers[l].norm_gamma = weights.get(aprefix + ".norm.gamma");
                tw.attn_layers[l].to_qkv_w   = weights.get(aprefix + ".to_qkv.weight");
                tw.attn_layers[l].to_gates_w  = weights.get(aprefix + ".to_gates.weight");
                tw.attn_layers[l].to_gates_b  = weights.get(aprefix + ".to_gates.bias");
                tw.attn_layers[l].to_out_w    = weights.get(aprefix + ".to_out.0.weight");
                // FeedForward: lprefix.1.net.*
                std::string fprefix = lprefix + ".1";
                tw.ff_layers[l].norm_gamma = weights.get(fprefix + ".net.0.gamma");
                tw.ff_layers[l].linear1_w  = weights.get(fprefix + ".net.1.weight");
                tw.ff_layers[l].linear1_b  = weights.get(fprefix + ".net.1.bias");
                tw.ff_layers[l].linear2_w  = weights.get(fprefix + ".net.4.weight");
                tw.ff_layers[l].linear2_b  = weights.get(fprefix + ".net.4.bias");
            }
            tw.final_norm_gamma = weights.get(tprefix + ".norm.gamma");
        };

        std::string dprefix = "layers." + std::to_string(d);
        load_transformer(dprefix + ".0", cfg_.time_transformer_depth,
                         depth_blocks_[d].time_transformer);
        load_transformer(dprefix + ".1", cfg_.freq_transformer_depth,
                         depth_blocks_[d].freq_transformer);
    }

    // 6. Load mask estimator weights
    // MaskEstimator for each stem. Each has num_bands band MLPs.
    // MLP structure depends on mask_estimator_depth:
    //   depth=1: dims=(dim, dim_hidden, dim_in*2) -> 2 linears at Sequential indices 0, 2
    //   depth=2: dims=(dim, dim_hidden, dim_hidden, dim_in*2) -> 3 linears at indices 0, 2, 4
    //   depth=k: (k+1) linears at indices 0, 2, 4, ..., 2k
    // Outer MaskEstimator Sequential: [MLP(index 0), GLU(index 1)]
    // So weight prefix: mask_estimators.{s}.to_freqs.{b}.0.{2*i}.weight/bias

    int mlp_depth = cfg_.mask_estimator_depth;
    int num_mlp_linears = mlp_depth + 1;  // depth hidden layers + 1 output layer

    mask_estimators_.resize(cfg_.num_stems);
    for (int s = 0; s < cfg_.num_stems; s++) {
        mask_estimators_[s].band_mlps.resize(num_bands);
        for (int b = 0; b < num_bands; b++) {
            auto& mlp = mask_estimators_[s].band_mlps[b];
            mlp.linear_w.resize(num_mlp_linears);
            mlp.linear_b.resize(num_mlp_linears);
            for (int i = 0; i < num_mlp_linears; i++) {
                std::string prefix = "mask_estimators." + std::to_string(s) +
                                     ".to_freqs." + std::to_string(b) +
                                     ".0." + std::to_string(i * 2);
                mlp.linear_w[i] = weights.get(prefix + ".weight");
                mlp.linear_b[i] = weights.get(prefix + ".bias");
            }
        }
    }

    std::cout << "[MelBandRoformer] Model loaded: dim=" << dim
              << " depth=" << cfg_.depth
              << " bands=" << num_bands
              << " heads=" << cfg_.heads
              << " stems=" << cfg_.num_stems
              << " stereo=" << cfg_.stereo
              << " freq_indices=" << freq_indices_.size()
              << std::endl;
}

// ============================================================================
// compute_rotary_cos_sin
// ============================================================================

Tensor MelBandRoformer::compute_rotary_cos_sin(int seq_len, int dim) {
    // Compute rotary embedding frequencies:
    // freqs_base = 1.0 / (10000 ^ (arange(0, dim, 2) / dim))
    // theta = outer(arange(seq_len), freqs_base) -> [seq_len, dim/2]
    // Returns stacked [2, seq_len, dim/2]: row 0 = cos(theta), row 1 = sin(theta)

    int half_dim = dim / 2;
    float theta_base = 10000.0f;

    // Compute on CPU then upload
    std::vector<float> cos_data(seq_len * half_dim);
    std::vector<float> sin_data(seq_len * half_dim);

    for (int n = 0; n < seq_len; n++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / std::pow(theta_base, (float)(2 * i) / (float)dim);
            float angle = (float)n * freq;
            cos_data[n * half_dim + i] = std::cos(angle);
            sin_data[n * half_dim + i] = std::sin(angle);
        }
    }

    Tensor cos_t = Tensor::from_cpu_f32(cos_data.data(), {(int64_t)seq_len, (int64_t)half_dim});
    Tensor sin_t = Tensor::from_cpu_f32(sin_data.data(), {(int64_t)seq_len, (int64_t)half_dim});

    return Tensor::stack({cos_t, sin_t}, 0);  // [2, seq_len, dim/2]
}

// ============================================================================
// apply_attention
// ============================================================================

Tensor MelBandRoformer::apply_attention(const Tensor& x, const AttentionWeights& w,
                                          const Tensor& cos_freqs, const Tensor& sin_freqs) {
    // x: [B, N, dim]
    int dim = cfg_.dim;
    int heads = cfg_.heads;
    int dim_head = cfg_.dim_head;
    float scale = std::sqrt((float)dim);

    // 1. RMS Norm
    Tensor normed = mbr_tile::rms_norm(x, w.norm_gamma, scale);

    // 2. QKV projection: [B, N, 3*dim_inner]
    Tensor qkv = mbr_tile::linear_no_bias(normed, w.to_qkv_w);

    // 3-4. Split q, k, v and apply rotary embedding to q/k.
    // qkv: [B, N, 3*heads*dim_head] -> q/k/v: [B, heads, N, dim_head]
    int64_t B = qkv.size(0);
    int64_t N = qkv.size(1);
    Tensor q;
    Tensor k;
    Tensor v;
    mbr_tile::split_qkv_heads_rotary(qkv, heads, dim_head, cos_freqs, sin_freqs, q, k, v);

    // 5. Scaled dot-product attention
    float attn_scale = 1.0f / std::sqrt((float)dim_head);
    Tensor out = mbr_tile::scaled_dot_product_attention(q, k, v, attn_scale);
    // out: [B, H, N, D]

    // 6. Gating
    // gates = sigmoid(linear(normed, to_gates_w, to_gates_b)) -> [B, N, heads]
    Tensor gates = mbr_tile::linear_sigmoid(normed, w.to_gates_w, w.to_gates_b);

    // 7-8. Apply gates and merge heads directly to [B, N, dim_inner]
    out = mbr_tile::apply_gates_and_merge_heads(out, gates, heads, dim_head);

    // 9. Output projection
    out = mbr_tile::linear_no_bias(out, w.to_out_w);  // [B, N, dim]

    return out;
}

// ============================================================================
// apply_feedforward
// ============================================================================

Tensor MelBandRoformer::apply_feedforward(const Tensor& x, const FeedForwardWeights& w) {
    // x: [B, N, dim]
    float scale = std::sqrt((float)cfg_.dim);

    // 1. RMS Norm
    Tensor h = mbr_tile::rms_norm(x, w.norm_gamma, scale);

    // 2. Fused Linear1 + GELU: [B, N, dim] -> [B, N, dim*4]
    h = mbr_tile::linear_gelu(h, w.linear1_w, w.linear1_b);

    // 3. Linear2: [B, N, dim*4] -> [B, N, dim]
    h = mbr_tile::linear(h, w.linear2_w, w.linear2_b);

    return h;
}

// ============================================================================
// apply_transformer
// ============================================================================

Tensor MelBandRoformer::apply_transformer(const Tensor& x,
                                            const TransformerLayerWeights& w,
                                            const Tensor& cos_freqs,
                                            const Tensor& sin_freqs) {
    // x: [B, N, dim]
    Tensor out = x;
    float scale = std::sqrt((float)cfg_.dim);

    int depth = (int)w.attn_layers.size();
    for (int i = 0; i < depth; i++) {
        // Attention + residual
        Tensor attn_out = apply_attention(out, w.attn_layers[i], cos_freqs, sin_freqs);
        out.add_(attn_out);

        // FeedForward + residual
        Tensor ff_out = apply_feedforward(out, w.ff_layers[i]);
        out.add_(ff_out);
    }

    // Final RMS norm
    out = mbr_tile::rms_norm(out, w.final_norm_gamma, scale);

    return out;
}

// ============================================================================
// apply_band_split
// ============================================================================

Tensor MelBandRoformer::apply_band_split(const Tensor& x) {
    // x: [B, T, total_freq_complex]
    // Split along last dim according to band_freq_dims_, apply per-band RMSNorm + Linear
    // Output: [B, T, num_bands, dim]

    int64_t B = x.size(0);
    int64_t T = x.size(1);
    int num_bands = cfg_.num_bands;
    float scale_rms = std::sqrt((float)band_freq_dims_[0]);  // will recompute per band

    // Split x along last dimension by band_freq_dims_
    std::vector<int64_t> split_sizes(band_freq_dims_.begin(), band_freq_dims_.end());
    std::vector<Tensor> band_splits = x.split(2, split_sizes);

    // Process each band and collect results
    std::vector<Tensor> band_outputs;
    band_outputs.reserve(num_bands);

    for (int b = 0; b < num_bands; b++) {
        // band_splits[b]: [B, T, band_freq_dim]
        float band_scale = std::sqrt((float)band_freq_dims_[b]);
        Tensor band_normed = mbr_tile::rms_norm(band_splits[b], band_split_layers_[b].norm_gamma,
                                                band_scale);
        Tensor band_out = mbr_tile::linear(band_normed, band_split_layers_[b].linear_w,
                                           band_split_layers_[b].linear_b);
        // band_out: [B, T, dim]
        band_outputs.push_back(band_out);
    }

    // Stack along new dim: [B, T, num_bands, dim]
    Tensor result = Tensor::stack(band_outputs, 2);
    return result;
}

// ============================================================================
// apply_mask_estimator
// ============================================================================

Tensor MelBandRoformer::apply_mask_estimator(const Tensor& x,
                                              const MaskEstimatorWeights& w) {
    // x: [B, T, num_bands, dim]
    // For each band: MLP layers -> Tanh (between linears) -> GLU at end
    // Output: [B, T, total_freq_complex]

    int64_t B = x.size(0);
    int64_t T = x.size(1);
    int num_bands = cfg_.num_bands;

    // Materialize [band, batch, time, dim] once so each per-band view stays contiguous.
    Tensor x_by_band = x.permute({2, 0, 1, 3}).contiguous();

    std::vector<Tensor> band_outputs;
    band_outputs.reserve(num_bands);

    for (int b = 0; b < num_bands; b++) {
        // Extract band features from the prepacked band-major buffer: [B, T, dim]
        Tensor band_x = x_by_band.slice(0, b, b + 1).squeeze(0);

        const auto& mlp = w.band_mlps[b];
        int num_linears = (int)mlp.linear_w.size();

        Tensor h = band_x;
        for (int i = 0; i < num_linears; i++) {
            h = mbr_tile::linear(h, mlp.linear_w[i], mlp.linear_b[i]);
            // Apply Tanh activation between linear layers (not after the last one)
            if (i < num_linears - 1) {
                h = mbr_tile::tanh_act(h);
            }
        }

        // Apply GLU: splits last dim in half, applies sigmoid gate
        // h: [B, T, band_freq_dim * 2] -> [B, T, band_freq_dim]
        h = mbr_tile::glu_last_dim(h);

        band_outputs.push_back(h);
    }

    // Concatenate all bands along last dim
    Tensor result = Tensor::cat(band_outputs, -1);
    return result;  // [B, T, total_freq_complex]
}

// ============================================================================
// forward
// ============================================================================

Tensor MelBandRoformer::forward(const Tensor& audio) {
    // audio: [B, channels, samples] or [B, samples]
    int audio_channels = cfg_.stereo ? 2 : 1;
    int num_stems = cfg_.num_stems;
    int num_bands = cfg_.num_bands;
    int dim = cfg_.dim;

    // ---- 1. Prepare audio ----
    Tensor raw_audio = audio;
    if (raw_audio.ndim() == 2) {
        // [B, T] -> [B, 1, T]
        raw_audio = raw_audio.unsqueeze(1);
    }

    int64_t batch = raw_audio.size(0);
    int64_t channels = raw_audio.size(1);
    int64_t raw_audio_length = raw_audio.size(2);
    int64_t istft_length = cfg_.match_input_audio_length ? raw_audio_length : -1;

    // ---- 2. STFT ----
    // Reshape for STFT: [B, channels, T] -> [B*channels, T]
    Tensor audio_flat = raw_audio.reshape({batch * channels, raw_audio_length});

    // STFT: [B*channels, F, T_stft, 2]
    Tensor stft_repr = stft_tile::stft(audio_flat, cfg_.stft_n_fft, cfg_.stft_hop_length,
                                       cfg_.stft_win_length, stft_window_, true,
                                       cfg_.stft_normalized);

    int64_t F = stft_repr.size(1);   // num frequency bins (n_fft/2 + 1)
    int64_t T = stft_repr.size(2);   // num time frames

    // ---- 3. Reshape back to [B, channels, F, T, 2] ----
    stft_repr = stft_repr.reshape({batch, channels, F, T, 2});

    // ---- 4. Merge stereo into frequency ----
    // Python: rearrange(stft_repr, 'b s f t c -> b (f s) t c')
    // This interleaves: for freq f, put channel 0 then channel 1
    // Result: [B, F*channels, T, 2] where dim1 = (f0_ch0, f0_ch1, f1_ch0, f1_ch1, ...)
    if (channels > 1) {
        // [B, S, F, T, 2] -> [B, F, S, T, 2] -> [B, F*S, T, 2]
        stft_repr = stft_repr.permute({0, 2, 1, 3, 4}).contiguous();
        // Now [B, F, S, T, 2], reshape to [B, F*S, T, 2]
        stft_repr = stft_repr.reshape({batch, F * channels, T, 2});
    }

    int64_t total_freq = F * channels;

    // ---- 5-6. Gather frequency bands and fold complex into frequency dimension ----
    Tensor x = mbr_tile::gather_freqs_fold_complex(stft_repr, freq_indices_gpu_);
    int64_t total_band_freqs = x.size(2) / 2;
    // x: [B, T, total_band_freqs * 2]

    // ---- 7. Band split ----
    x = apply_band_split(x);
    // x: [B, T, num_bands, dim]

    // ---- 8. Transformer layers (depth iterations) ----
    // Store for skip connections
    std::vector<Tensor> skip_store;
    if (cfg_.skip_connection) {
        skip_store.resize(cfg_.depth);
    }

    for (int d = 0; d < cfg_.depth; d++) {
        // Skip connections: sum all previous
        if (cfg_.skip_connection) {
            for (int j = 0; j < d; j++) {
                x = x + skip_store[j];
            }
        }

        // Time attention: [B, T, F, D] -> [B, F, T, D] -> [B*F, T, D]
        // Python: x = rearrange(x, 'b t f d -> b f t d')
        x = x.permute({0, 2, 1, 3}).contiguous();  // [B, num_bands, T, dim]
        int64_t BF = batch * num_bands;
        x = x.reshape({BF, T, (int64_t)dim});      // [B*num_bands, T, dim]

        // Compute rotary embeddings for time dimension (cached)
        auto& time_cached = rotary_cache_[(int)T];
        if (time_cached.first.numel() == 0) {
            Tensor time_rot = compute_rotary_cos_sin((int)T, cfg_.dim_head);
            time_cached.first = time_rot.slice(0, 0, 1).squeeze(0).contiguous();
            time_cached.second = time_rot.slice(0, 1, 2).squeeze(0).contiguous();
        }
        const Tensor& time_cos = time_cached.first;
        const Tensor& time_sin = time_cached.second;

        x = apply_transformer(x, depth_blocks_[d].time_transformer, time_cos, time_sin);

        // Reshape back: [B*F, T, D] -> [B, F, T, D] -> [B, T, F, D]
        x = x.reshape({batch, (int64_t)num_bands, T, (int64_t)dim});
        x = x.permute({0, 2, 1, 3}).contiguous();  // [B, T, num_bands, dim]

        // Freq attention: [B, T, F, D] -> [B*T, F, D]
        int64_t BT = batch * T;
        x = x.reshape({BT, (int64_t)num_bands, (int64_t)dim});

        // Compute rotary embeddings for freq dimension (cached)
        auto& freq_cached = rotary_cache_[num_bands];
        if (freq_cached.first.numel() == 0) {
            Tensor freq_rot = compute_rotary_cos_sin(num_bands, cfg_.dim_head);
            freq_cached.first = freq_rot.slice(0, 0, 1).squeeze(0).contiguous();
            freq_cached.second = freq_rot.slice(0, 1, 2).squeeze(0).contiguous();
        }
        const Tensor& freq_cos = freq_cached.first;
        const Tensor& freq_sin = freq_cached.second;

        x = apply_transformer(x, depth_blocks_[d].freq_transformer, freq_cos, freq_sin);

        // Reshape back: [B*T, F, D] -> [B, T, F, D]
        x = x.reshape({batch, T, (int64_t)num_bands, (int64_t)dim});

        // Store for skip connections
        if (cfg_.skip_connection) {
            skip_store[d] = x;
        }
    }

    // ---- 9. Mask estimation ----
    // x: [B, T, num_bands, dim]
    // Apply each mask estimator and stack
    std::vector<Tensor> stem_masks;
    stem_masks.reserve(num_stems);
    for (int s = 0; s < num_stems; s++) {
        Tensor mask = apply_mask_estimator(x, mask_estimators_[s]);
        // mask: [B, T, total_freq_complex]
        stem_masks.push_back(mask);
    }

    // ---- 10. Apply masks and scatter overlapping bands ----
    Tensor stft_result = mbr_tile::apply_mask_and_scatter(
        stft_repr, stem_masks, freq_indices_gpu_, num_bands_per_freq_gpu_,
        batch, num_stems, total_freq, total_band_freqs, T, audio_channels);
    // stft_result: [(B * num_stems * audio_channels), F, T, 2]

    // Zero DC component if requested
    if (cfg_.zero_dc) {
        mbr_tile::zero_dc(stft_result);
    }

    // ---- 12. iSTFT ----
    Tensor recon_audio = stft_tile::istft(stft_result, cfg_.stft_n_fft, cfg_.stft_hop_length,
                                          cfg_.stft_win_length, stft_window_, istft_length,
                                          true, cfg_.stft_normalized);

    int64_t out_length = recon_audio.size(1);

    // ---- 13. Reshape output ----
    // Python: rearrange(recon_audio, '(b n s) t -> b n s t', b=batch, s=audio_channels, n=num_stems)
    recon_audio = recon_audio.reshape({batch, (int64_t)num_stems,
                                       (int64_t)audio_channels, out_length});

    if (num_stems == 1) {
        // Python: rearrange(recon_audio, 'b 1 s t -> b s t')
        recon_audio = recon_audio.squeeze(1);  // [B, channels, T]
    }

    return recon_audio;
}

} // namespace cudasep
