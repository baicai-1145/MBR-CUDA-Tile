// model_mel_band_roformer.cu - Complete MelBandRoformer inference implementation.
//
// This implements the MelBandRoformer model for music source separation,
// mirroring the Python forward pass exactly.

#include "model_mel_band_roformer.h"
#include "mbr_cuda_tile.h"
#include "stft_cuda_tile.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace cudasep {
namespace {

Tensor match_dtype_for_add(const Tensor& value, DType dtype) {
    return (value.dtype() == dtype) ? value : value.to_dtype(dtype);
}

Tensor maybe_residual_bf16(const Tensor& value) {
    return (mbr_tile::residual_bf16_enabled() && value.dtype() != DType::BFloat16)
        ? value.to_bf16()
        : value;
}

Tensor maybe_bf16_bias(const Tensor& value) {
    return (mbr_tile::bias_bf16_enabled() && value.dtype() == DType::Float32)
        ? value.to_bf16()
        : value;
}

Tensor maybe_bf16_norm_gamma(const Tensor& value) {
    return (mbr_tile::norm_gamma_bf16_enabled() && value.dtype() == DType::Float32)
        ? value.to_bf16()
        : value;
}

bool local_env_flag_enabled(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') return false;
    std::string value(raw);
    return value != "0" && value != "false" && value != "FALSE" &&
           value != "off" && value != "OFF";
}

bool debug_compare_time_qkv_fused_rotary_enabled() {
    static bool enabled = local_env_flag_enabled(
        "CUDASEP_DEBUG_COMPARE_TIME_QKV_FUSED_ROTARY");
    return enabled;
}

int debug_compare_time_qkv_fused_rotary_limit() {
    static int limit = []() {
        const char* raw = std::getenv("CUDASEP_DEBUG_COMPARE_TIME_QKV_FUSED_ROTARY_LIMIT");
        if (!raw || raw[0] == '\0') return 1;
        int parsed = std::atoi(raw);
        return parsed > 0 ? parsed : 0;
    }();
    return limit;
}

struct TensorDiffStats {
    double max_abs = 0.0;
    double rms = 0.0;
    double rel_rms = 0.0;
    int64_t max_idx = -1;
    float ref_at_max = 0.0f;
    float test_at_max = 0.0f;
};

TensorDiffStats tensor_diff_stats(const Tensor& ref, const Tensor& test) {
    if (ref.shape() != test.shape()) {
        throw std::runtime_error("tensor diff: shape mismatch");
    }
    std::vector<float> ref_cpu = ref.to_cpu_f32();
    std::vector<float> test_cpu = test.to_cpu_f32();
    TensorDiffStats stats;
    double sum_sq = 0.0;
    double ref_sum_sq = 0.0;
    for (int64_t i = 0; i < ref.numel(); ++i) {
        double diff = (double)test_cpu[i] - (double)ref_cpu[i];
        double abs_diff = std::fabs(diff);
        sum_sq += diff * diff;
        ref_sum_sq += (double)ref_cpu[i] * (double)ref_cpu[i];
        if (abs_diff > stats.max_abs) {
            stats.max_abs = abs_diff;
            stats.max_idx = i;
            stats.ref_at_max = ref_cpu[i];
            stats.test_at_max = test_cpu[i];
        }
    }
    if (ref.numel() > 0) {
        stats.rms = std::sqrt(sum_sq / (double)ref.numel());
        stats.rel_rms = std::sqrt(sum_sq / std::max(ref_sum_sq, 1e-30));
    }
    return stats;
}

void print_time_qkv_fused_rotary_diff(int call_idx,
                                      const char* name,
                                      const Tensor& ref,
                                      const Tensor& test) {
    TensorDiffStats stats = tensor_diff_stats(ref, test);
    std::cerr << "[debug time-qkv fused-rotary] call=" << call_idx
              << " " << name
              << " max_abs=" << stats.max_abs
              << " rms=" << stats.rms
              << " rel_rms=" << stats.rel_rms
              << " max_idx=" << stats.max_idx;
    if (stats.max_idx >= 0 && ref.ndim() == 4) {
        int64_t d = ref.size(3);
        int64_t h = ref.size(2);
        int64_t n = ref.size(1);
        int64_t idx = stats.max_idx;
        int64_t dim = idx % d;
        idx /= d;
        int64_t head = idx % h;
        idx /= h;
        int64_t token = idx % n;
        int64_t batch = idx / n;
        std::cerr << " at=[b" << batch << ",n" << token
                  << ",h" << head << ",d" << dim << "]";
    }
    std::cerr << " ref=" << stats.ref_at_max
              << " test=" << stats.test_at_max << std::endl;
}

struct TimeQkvProfileStats {
    double split_producer_ms = 0.0;
    double split_rotary_ms = 0.0;
    double bkn_fused_ms = 0.0;
    double pair_fused_ms = 0.0;
    int split_producer_calls = 0;
    int split_rotary_calls = 0;
    int bkn_fused_calls = 0;
    int pair_fused_calls = 0;
};

TimeQkvProfileStats& time_qkv_profile_stats() {
    static TimeQkvProfileStats stats;
    return stats;
}

void print_time_qkv_profile_stats() {
    const auto& s = time_qkv_profile_stats();
    auto print = [](const char* name, double total_ms, int calls) {
        if (calls <= 0) return;
        std::cerr << "[profile time-qkv] " << name
                  << " calls=" << calls
                  << " total_ms=" << total_ms
                  << " avg_ms=" << (total_ms / (double)calls)
                  << std::endl;
    };
    print("split_producer", s.split_producer_ms, s.split_producer_calls);
    print("split_rotary", s.split_rotary_ms, s.split_rotary_calls);
    print("bkn_fused_producer", s.bkn_fused_ms, s.bkn_fused_calls);
    print("pair_fused_producer", s.pair_fused_ms, s.pair_fused_calls);
}

bool profile_time_qkv_enabled() {
    static bool enabled = []() {
        bool on = local_env_flag_enabled("CUDASEP_PROFILE_TIME_QKV_PRODUCER");
        if (on) std::atexit(print_time_qkv_profile_stats);
        return on;
    }();
    return enabled;
}

enum class TimeQkvProfileSegment {
    SplitProducer,
    SplitRotary,
    BknFused,
    PairFused,
};

void add_time_qkv_profile(TimeQkvProfileSegment segment, float ms) {
    auto& s = time_qkv_profile_stats();
    switch (segment) {
        case TimeQkvProfileSegment::SplitProducer:
            s.split_producer_ms += ms;
            ++s.split_producer_calls;
            break;
        case TimeQkvProfileSegment::SplitRotary:
            s.split_rotary_ms += ms;
            ++s.split_rotary_calls;
            break;
        case TimeQkvProfileSegment::BknFused:
            s.bkn_fused_ms += ms;
            ++s.bkn_fused_calls;
            break;
        case TimeQkvProfileSegment::PairFused:
            s.pair_fused_ms += ms;
            ++s.pair_fused_calls;
            break;
    }
}

template <typename Fn>
void run_time_qkv_profiled(TimeQkvProfileSegment segment, Fn&& fn) {
    if (!profile_time_qkv_enabled()) {
        std::forward<Fn>(fn)();
        return;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    std::forward<Fn>(fn)();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    add_time_qkv_profile(segment, elapsed_ms);
}

}  // namespace

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
    if (mbr_tile::bands_per_freq_bf16_enabled()) {
        num_bands_per_freq_gpu_ = num_bands_per_freq_gpu_.to_bf16();
    }

    std::cout << "[MelBandRoformer] Using precomputed mel bands from .csm file" << std::endl;

    // 3. Create STFT window
    stft_window_ = stft_tile::hann_window(cfg_.stft_win_length);

    int dim = cfg_.dim;

    // 4. Load band_split weights
    band_split_layers_.resize(num_bands);
    for (int b = 0; b < num_bands; b++) {
        std::string prefix = "band_split.to_features." + std::to_string(b);
        band_split_layers_[b].norm_gamma = maybe_bf16_norm_gamma(weights.get(prefix + ".0.gamma"));
        band_split_layers_[b].linear_w   = weights.get(prefix + ".1.weight");
        band_split_layers_[b].linear_b   = maybe_bf16_bias(weights.get(prefix + ".1.bias"));
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
                tw.attn_layers[l].norm_gamma =
                    maybe_bf16_norm_gamma(weights.get(aprefix + ".norm.gamma"));
                tw.attn_layers[l].to_qkv_w   = weights.get(aprefix + ".to_qkv.weight");
                if (mbr_tile::linear_bkn_long_enabled()) {
                    tw.attn_layers[l].to_qkv_w_bkn =
                        tw.attn_layers[l].to_qkv_w.transpose(0, 1).contiguous();
                }
                tw.attn_layers[l].to_gates_w  = weights.get(aprefix + ".to_gates.weight");
                tw.attn_layers[l].to_gates_b  = maybe_bf16_bias(weights.get(aprefix + ".to_gates.bias"));
                tw.attn_layers[l].to_out_w    = weights.get(aprefix + ".to_out.0.weight");
                // FeedForward: lprefix.1.net.*
                std::string fprefix = lprefix + ".1";
                tw.ff_layers[l].norm_gamma =
                    maybe_bf16_norm_gamma(weights.get(fprefix + ".net.0.gamma"));
                tw.ff_layers[l].linear1_w  = weights.get(fprefix + ".net.1.weight");
                if (mbr_tile::linear_bkn_ffn_long_enabled()) {
                    tw.ff_layers[l].linear1_w_bkn =
                        tw.ff_layers[l].linear1_w.transpose(0, 1).contiguous();
                }
                tw.ff_layers[l].linear1_b  = maybe_bf16_bias(weights.get(fprefix + ".net.1.bias"));
                tw.ff_layers[l].linear2_w  = weights.get(fprefix + ".net.4.weight");
                if (mbr_tile::linear_bkn_ffn_long_enabled()) {
                    tw.ff_layers[l].linear2_w_bkn =
                        tw.ff_layers[l].linear2_w.transpose(0, 1).contiguous();
                }
                tw.ff_layers[l].linear2_b  = maybe_bf16_bias(weights.get(fprefix + ".net.4.bias"));
            }
            tw.final_norm_gamma =
                maybe_bf16_norm_gamma(weights.get(tprefix + ".norm.gamma"));
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
                mlp.linear_b[i] = maybe_bf16_bias(weights.get(prefix + ".bias"));
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
    if (mbr_tile::rotary_freqs_bf16_enabled()) {
        cos_t = cos_t.to_bf16();
        sin_t = sin_t.to_bf16();
    }

    return Tensor::stack({cos_t, sin_t}, 0);  // [2, seq_len, dim/2]
}

// ============================================================================
// apply_attention
// ============================================================================

Tensor MelBandRoformer::apply_attention(const Tensor& x, const AttentionWeights& w,
                                          const Tensor& cos_freqs, const Tensor& sin_freqs,
                                          const Tensor* residual,
                                          bool* used_fused_residual) {
    float scale = std::sqrt((float)cfg_.dim);
    Tensor normed = mbr_tile::rms_norm(x, w.norm_gamma, scale);
    return apply_attention_normed(
        normed, w, cos_freqs, sin_freqs, residual, used_fused_residual);
}

Tensor MelBandRoformer::apply_attention_normed(const Tensor& normed, const AttentionWeights& w,
                                                const Tensor& cos_freqs, const Tensor& sin_freqs,
                                                const Tensor* residual,
                                                bool* used_fused_residual) {
    if (used_fused_residual) {
        *used_fused_residual = false;
    }
    // normed: [B, N, dim]
    int heads = cfg_.heads;
    int dim_head = cfg_.dim_head;

    // 2-4. QKV projection, split heads, and apply rotary embedding to q/k.
    float attn_scale = 1.0f / std::sqrt((float)dim_head);
    Tensor out;
    bool used_split_contig_time = false;
    if (mbr_tile::time_qkv_split_contig_path_enabled() &&
        !w.to_qkv_w_bkn.is_empty() &&
        normed.dtype() == DType::BFloat16 &&
        w.to_qkv_w.dtype() == DType::BFloat16 &&
        w.to_qkv_w_bkn.dtype() == DType::BFloat16 &&
        normed.ndim() == 3 && normed.size(1) == 1301 &&
        heads == 8 && dim_head == 64) {
        Tensor q_sc;
        Tensor k_sc;
        Tensor v_sc;
        static int debug_time_qkv_fused_rotary_reports = 0;
        bool debug_compare_fused =
            debug_compare_time_qkv_fused_rotary_enabled() &&
            debug_time_qkv_fused_rotary_reports <
                debug_compare_time_qkv_fused_rotary_limit();
        bool use_pair_rotary = mbr_tile::time_qkv_pair_rotary_producer_enabled();
        bool use_fused_rotary = mbr_tile::time_qkv_fused_rotary_producer_enabled();
        bool use_qrot_attention =
            mbr_tile::time_q_rotary_in_attention_enabled() &&
            !use_pair_rotary && !use_fused_rotary && !debug_compare_fused;
        if (debug_compare_fused) {
            int debug_call = debug_time_qkv_fused_rotary_reports;
            Tensor q_ref;
            Tensor k_ref;
            Tensor v_ref;
            Tensor q_test;
            Tensor k_test;
            Tensor v_test;
            mbr_tile::linear_qkv_bkn_split_contig_time(
                normed, w.to_qkv_w, w.to_qkv_w_bkn, q_ref, k_ref, v_ref);
            mbr_tile::apply_rotary_time_split_contig_inplace(
                q_ref, k_ref, cos_freqs, sin_freqs);
            if (use_pair_rotary) {
                mbr_tile::linear_qkv_pair_rotary_split_contig_time(
                    normed, w.to_qkv_w, cos_freqs, sin_freqs,
                    q_test, k_test, v_test);
            } else {
                mbr_tile::linear_qkv_bkn_rotary_split_contig_time(
                    normed, w.to_qkv_w, w.to_qkv_w_bkn, cos_freqs, sin_freqs,
                    q_test, k_test, v_test);
            }
            print_time_qkv_fused_rotary_diff(debug_call, "q", q_ref, q_test);
            print_time_qkv_fused_rotary_diff(debug_call, "k", k_ref, k_test);
            print_time_qkv_fused_rotary_diff(debug_call, "v", v_ref, v_test);
            ++debug_time_qkv_fused_rotary_reports;
            if (use_pair_rotary || use_fused_rotary) {
                q_sc = q_test;
                k_sc = k_test;
                v_sc = v_test;
            } else {
                q_sc = q_ref;
                k_sc = k_ref;
                v_sc = v_ref;
            }
        } else if (use_pair_rotary) {
            run_time_qkv_profiled(TimeQkvProfileSegment::PairFused, [&]() {
                mbr_tile::linear_qkv_pair_rotary_split_contig_time(
                    normed, w.to_qkv_w, cos_freqs, sin_freqs,
                    q_sc, k_sc, v_sc);
            });
        } else if (use_fused_rotary) {
            run_time_qkv_profiled(TimeQkvProfileSegment::BknFused, [&]() {
                mbr_tile::linear_qkv_bkn_rotary_split_contig_time(
                    normed, w.to_qkv_w, w.to_qkv_w_bkn, cos_freqs, sin_freqs,
                    q_sc, k_sc, v_sc);
            });
        } else {
            run_time_qkv_profiled(TimeQkvProfileSegment::SplitProducer, [&]() {
                mbr_tile::linear_qkv_bkn_split_contig_time(
                    normed, w.to_qkv_w, w.to_qkv_w_bkn, q_sc, k_sc, v_sc);
            });
            run_time_qkv_profiled(TimeQkvProfileSegment::SplitRotary, [&]() {
                if (use_qrot_attention) {
                    mbr_tile::apply_rotary_time_split_contig_k_only_inplace(
                        k_sc, cos_freqs, sin_freqs);
                } else {
                    mbr_tile::apply_rotary_time_split_contig_inplace(
                        q_sc, k_sc, cos_freqs, sin_freqs);
                }
            });
        }
        if (use_qrot_attention) {
            out = mbr_tile::scaled_dot_product_attention_time_split_contig_q_rotary(
                q_sc, k_sc, v_sc, cos_freqs, sin_freqs, attn_scale);
        } else {
            out = mbr_tile::scaled_dot_product_attention_time_split_contig(
                q_sc, k_sc, v_sc, attn_scale);
        }
        used_split_contig_time = true;
    }

    if (!used_split_contig_time) {
        Tensor q;
        Tensor k;
        Tensor v;
        if (!w.to_qkv_w_bkn.is_empty()) {
            mbr_tile::linear_qkv_rotary_bf16_output_bkn(
                normed, w.to_qkv_w, w.to_qkv_w_bkn, heads, dim_head,
                cos_freqs, sin_freqs, q, k, v);
        } else {
            mbr_tile::linear_qkv_rotary_bf16_output(
                normed, w.to_qkv_w, heads, dim_head, cos_freqs, sin_freqs, q, k, v);
        }

        // 5. Scaled dot-product attention
        out = mbr_tile::scaled_dot_product_attention(q, k, v, attn_scale);
    }
    // out: [B, H, N, D]

    // 6. Gating
    // gates = sigmoid(linear(normed, to_gates_w, to_gates_b)) -> [B, N, heads]
    Tensor gates = mbr_tile::linear_sigmoid(normed, w.to_gates_w, w.to_gates_b);

    // 7-8. Apply gates and merge heads directly to [B, N, dim_inner]
    out = mbr_tile::apply_gates_and_merge_heads(out, gates, heads, dim_head);

    // 9. Output projection
    if (residual && used_fused_residual &&
        mbr_tile::try_linear_no_bias_residual_bf16_output(
            out, w.to_out_w, *residual, out)) {
        *used_fused_residual = true;
        return out;
    }
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

    Tensor fused;
    if (w.linear1_w_bkn.is_empty() && w.linear2_w_bkn.is_empty() &&
        mbr_tile::try_feedforward_fused(
            h, w.linear1_w, w.linear1_b, w.linear2_w, w.linear2_b, fused)) {
        return fused;
    }

    // 2. Fused Linear1 + GELU: [B, N, dim] -> [B, N, dim*4]
    h = !w.linear1_w_bkn.is_empty()
        ? mbr_tile::linear_gelu_bkn(h, w.linear1_w, w.linear1_w_bkn, w.linear1_b)
        : mbr_tile::linear_gelu(h, w.linear1_w, w.linear1_b);

    // 3. Linear2: [B, N, dim*4] -> [B, N, dim]
    h = !w.linear2_w_bkn.is_empty()
        ? mbr_tile::linear_bkn(h, w.linear2_w, w.linear2_w_bkn, w.linear2_b)
        : mbr_tile::linear(h, w.linear2_w, w.linear2_b);

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
    Tensor out = maybe_residual_bf16(x);
    float scale = std::sqrt((float)cfg_.dim);

    int depth = (int)w.attn_layers.size();
    Tensor pending_attn_normed;
    bool has_pending_attn_normed = false;
    Tensor pending_final_normed;
    bool has_pending_final_normed = false;
    for (int i = 0; i < depth; i++) {
        // Attention + residual
        bool used_fused_attn_residual = false;
        Tensor attn_out;
        if (has_pending_attn_normed) {
            attn_out = apply_attention_normed(pending_attn_normed, w.attn_layers[i],
                                              cos_freqs, sin_freqs,
                                              &out, &used_fused_attn_residual);
            has_pending_attn_normed = false;
        } else {
            attn_out = apply_attention(out, w.attn_layers[i], cos_freqs, sin_freqs,
                                       &out, &used_fused_attn_residual);
        }
        if (used_fused_attn_residual) {
            out = maybe_residual_bf16(attn_out);
        } else {
            attn_out = match_dtype_for_add(attn_out, out.dtype());
            out.add_(attn_out);
            out = maybe_residual_bf16(out);
        }

        // FeedForward + residual
        Tensor ff_residual_out;
        bool used_fused_ff_residual = false;
        if (w.ff_layers[i].linear1_w_bkn.is_empty() &&
            w.ff_layers[i].linear2_w_bkn.is_empty()) {
            Tensor ff_normed = mbr_tile::rms_norm(out, w.ff_layers[i].norm_gamma, scale);
            used_fused_ff_residual = mbr_tile::try_feedforward_fused_residual(
                ff_normed,
                w.ff_layers[i].linear1_w,
                w.ff_layers[i].linear1_b,
                w.ff_layers[i].linear2_w,
                w.ff_layers[i].linear2_b,
                out,
                ff_residual_out);
        }
        if (used_fused_ff_residual) {
            out = maybe_residual_bf16(ff_residual_out);
        } else {
            Tensor ff_out = apply_feedforward(out, w.ff_layers[i]);
            ff_out = match_dtype_for_add(ff_out, out.dtype());
            Tensor residual_sum;
            Tensor next_normed;
            const Tensor& next_gamma =
                (i + 1 < depth) ? w.attn_layers[i + 1].norm_gamma : w.final_norm_gamma;
            if (mbr_tile::try_residual_add_rms_norm(
                    out, ff_out, next_gamma, scale, residual_sum, next_normed)) {
                out = maybe_residual_bf16(residual_sum);
                if (i + 1 < depth) {
                    pending_attn_normed = next_normed;
                    has_pending_attn_normed = true;
                } else {
                    pending_final_normed = next_normed;
                    has_pending_final_normed = true;
                }
            } else {
                out.add_(ff_out);
                out = maybe_residual_bf16(out);
            }
        }
    }

    // Final RMS norm
    out = has_pending_final_normed
        ? pending_final_normed
        : mbr_tile::rms_norm(out, w.final_norm_gamma, scale);
    out = maybe_residual_bf16(out);

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
        bool used_fused_glu = false;
        for (int i = 0; i < num_linears; i++) {
            if (i == num_linears - 1) {
                Tensor fused_glu;
                if (mbr_tile::try_linear_glu_last_dim_bf16_output(
                        h, mlp.linear_w[i], mlp.linear_b[i], fused_glu)) {
                    std::vector<int64_t> fused_shape = h.shape();
                    fused_shape.back() = mlp.linear_w[i].size(0) / 2;
                    h = fused_glu.reshape(fused_shape);
                    used_fused_glu = true;
                    break;
                }
            }
            h = mbr_tile::linear(h, mlp.linear_w[i], mlp.linear_b[i]);
            // Apply Tanh activation between linear layers (not after the last one)
            if (i < num_linears - 1) {
                h = mbr_tile::tanh_act(h);
            }
        }

        // Apply GLU: splits last dim in half, applies sigmoid gate
        // h: [B, T, band_freq_dim * 2] -> [B, T, band_freq_dim]
        if (!used_fused_glu) {
            h = mbr_tile::glu_last_dim(h);
        }

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

    mbr_tile::set_time_attention_context_depth(-1);
    for (int d = 0; d < cfg_.depth; d++) {
        // Skip connections: sum all previous
        if (cfg_.skip_connection) {
            for (int j = 0; j < d; j++) {
                x = x + match_dtype_for_add(skip_store[j], x.dtype());
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

        mbr_tile::set_time_attention_context_depth(d);
        x = apply_transformer(x, depth_blocks_[d].time_transformer, time_cos, time_sin);
        mbr_tile::set_time_attention_context_depth(-1);

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
            skip_store[d] = maybe_residual_bf16(x);
        }
    }
    mbr_tile::set_time_attention_context_depth(-1);

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
