#include "mbr_cuda_tile.h"

#include "cuda_tile.h"
#include "cuda_context.h"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef CUDASEP_ENABLE_FREQ_ATTENTION60_FUSED
#define CUDASEP_ENABLE_FREQ_ATTENTION60_FUSED 1
#endif

#ifndef CUDASEP_ENABLE_QKV_BF16_OUTPUT
#define CUDASEP_ENABLE_QKV_BF16_OUTPUT 1
#endif

namespace cudasep::mbr_tile {

bool try_residual_add_bf16_cutile(const Tensor& x, const Tensor& residual, Tensor& out);

namespace {

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kTile = 256;
constexpr int kTimeRotaryTileDefault = 128;
constexpr int kRmsTile = 1024;
constexpr int kRmsD256 = 256;
constexpr int kSmallSoftmaxTile = 64;
constexpr int kSoftmaxTile = 2048;
constexpr int kFreqAttnN = 60;
constexpr int kFreqAttnPadN = 64;
constexpr int kFreqAttnD = 64;
constexpr int kFreqAttnCutileQRows = 16;
constexpr int kQkvFusedHeads = 8;
constexpr int kTimeAttnN = 1301;
constexpr int kTimeAttnD = 64;
constexpr int kTimeAttnCutileQRows16 = 16;
constexpr int kTimeAttnCutileQRows32 = 32;
constexpr int kTimeAttnCutileQRows64 = 64;
constexpr int kTimeAttnCutileQRows128 = 128;
constexpr int kTimeAttnCutileKTile32 = 32;
constexpr int kTimeAttnCutileKTile64 = 64;
constexpr int kTimeAttnCutileKTile128 = 128;
constexpr float kTimeAttnScale = 0.125f;
constexpr int kGateMergeTokenD64Tile = kQkvFusedHeads * kTimeAttnD;
constexpr int kLinearCutileStaticM = 78048;
constexpr int kLinearCutileStaticM64 = 78016;
constexpr int kLinearCutileExpectedM = 78060;
constexpr int kLinearCutileSmallExpectedM = 1301;
constexpr int kLinearCutileSmallPaddedM32 = 1312;
constexpr int kLinearCutileSmallPaddedM64 = 1344;
constexpr int kLinearCutileTileM = 32;
constexpr int kLinearCutileTileN = 64;
constexpr int kLinearCutileTileK = 32;
constexpr int kGeluErf = 0;
constexpr int kGeluHard = 1;
constexpr int kGeluQuick = 2;
constexpr int kGeluTanh = 3;
constexpr int kGeluErfPoly5L25 = 4;
constexpr int kGeluErfPoly7L25 = 5;
constexpr int kGeluErfPoly9L30 = 6;
constexpr int kGeluErfPoly9TinyBlendL30 = 7;
constexpr int kGeluErfOdd5L175 = 8;
using I64Tile = ct::tile<long long, ct::shape<kTile>>;
using F32Tile = ct::tile<float, ct::shape<kTile>>;
using RmsI64Tile = ct::tile<long long, ct::shape<kRmsTile>>;
using RmsF32Tile = ct::tile<float, ct::shape<kRmsTile>>;
using RmsD256I64Tile = ct::tile<long long, ct::shape<kRmsD256>>;
using RmsD256F32Tile = ct::tile<float, ct::shape<kRmsD256>>;
using SmallSoftmaxI64Tile = ct::tile<long long, ct::shape<kSmallSoftmaxTile>>;
using SmallSoftmaxF16Tile = ct::tile<__half, ct::shape<kSmallSoftmaxTile>>;
using SmallSoftmaxBF16Tile = ct::tile<__nv_bfloat16, ct::shape<kSmallSoftmaxTile>>;
using SoftmaxI64Tile = ct::tile<long long, ct::shape<kSoftmaxTile>>;
using SoftmaxF16Tile = ct::tile<__half, ct::shape<kSoftmaxTile>>;
using SoftmaxBF16Tile = ct::tile<__nv_bfloat16, ct::shape<kSoftmaxTile>>;

thread_local int g_time_attention_context_chunk = -1;
thread_local int g_time_attention_context_depth = -1;

static inline int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

template <typename T>
static __tile__ auto bf16_round(T value) {
    return ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(value));
}

template <bool FullBF16, typename T>
static __tile__ auto bf16_round_if(T value) {
    if constexpr (FullBF16) {
        return bf16_round(value);
    }
    return value;
}

bool env_flag_enabled(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) return false;
    std::string value(raw);
    return !(value.empty() || value == "0" || value == "false" || value == "FALSE" ||
             value == "off" || value == "OFF");
}

int env_nonnegative_int_or(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') return fallback;
    int parsed = std::atoi(raw);
    return parsed >= 0 ? parsed : fallback;
}

bool custom_bf16_path_enabled() {
    return g_quantize_bf16 && !env_flag_enabled("CUDASEP_DISABLE_BF16_CUSTOM_PATH");
}

[[noreturn]] void unsupported_gemm_path(const char* op) {
    throw std::runtime_error(std::string(op) +
                             ": unsupported GEMM path for the current fixed BF16 build");
}

bool all_bf16_experiments_enabled() {
    static int enabled = env_flag_enabled("CUDASEP_ALL_BF16_EXPERIMENTS") ? 1 : 0;
    return g_quantize_bf16 && (custom_bf16_path_enabled() || enabled != 0);
}

bool remaining_bf16_experiments_enabled() {
    static int enabled = env_flag_enabled("CUDASEP_REMAINING_BF16_EXPERIMENTS") ? 1 : 0;
    return g_quantize_bf16 && (custom_bf16_path_enabled() || enabled != 0);
}

bool bf16_experiment_enabled(const char* name) {
    return g_quantize_bf16 && (all_bf16_experiments_enabled() || env_flag_enabled(name));
}

bool remaining_bf16_experiment_enabled(const char* name) {
    return bf16_experiment_enabled(name) || remaining_bf16_experiments_enabled();
}

bool gate_sigmoid_bf16_enabled() {
    return bf16_experiment_enabled("CUDASEP_GATE_SIGMOID_BF16");
}

bool gate_merge_bf16_enabled() {
    return bf16_experiment_enabled("CUDASEP_GATE_MERGE_BF16");
}

bool rms_norm_bf16_enabled() {
    return bf16_experiment_enabled("CUDASEP_RMS_NORM_BF16");
}

bool rms_norm_d256_cutile_fixed_enabled() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_RMS_NORM_D256_CUTILE_FIXED");
}

int rms_norm_d256_rows_per_cta() {
    static int rows_per_cta = []() {
        const char* raw = std::getenv("CUDASEP_RMS_NORM_D256_ROWS_PER_CTA");
        if (raw == nullptr) return 1;
        int parsed = std::atoi(raw);
        if (parsed == 2 || parsed == 4) return parsed;
        return 1;
    }();
    return rows_per_cta;
}

bool residual_rms_norm_fused_enabled() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_RESIDUAL_RMS_NORM_FUSED");
}

int residual_rms_norm_d256_rows_per_cta() {
    static int rows_per_cta = []() {
        const char* raw = std::getenv("CUDASEP_RESIDUAL_RMS_NORM_D256_ROWS_PER_CTA");
        if (raw == nullptr) return 1;
        int parsed = std::atoi(raw);
        if (parsed == 2 || parsed == 4) return parsed;
        return 1;
    }();
    return rows_per_cta;
}

bool linear_bf16_output_enabled() {
    return bf16_experiment_enabled("CUDASEP_LINEAR_BF16_OUTPUT");
}

bool attention_av_bf16_output_enabled() {
    return bf16_experiment_enabled("CUDASEP_ATTENTION_AV_BF16_OUTPUT");
}

bool full_bf16_arith_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_FULL_BF16_ARITH_EXPERIMENTS");
}

bool attention_qk_bf16_scores_enabled() {
    return bf16_experiment_enabled("CUDASEP_ATTENTION_QK_BF16_SCORES") ||
           full_bf16_arith_enabled();
}

bool attention_qk_bf16_accum_enabled() {
    return g_quantize_bf16 && (env_flag_enabled("CUDASEP_ATTENTION_QK_BF16_ACCUM") ||
                               (full_bf16_arith_enabled() &&
                                !env_flag_enabled("CUDASEP_FULL_BF16_ARITH_KEEP_QK_ACCUM_FP32")));
}

bool freq_attention60_cutile_padded_enabled() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_FREQ_ATTENTION60_CUTILE_PADDED");
}

bool freq_split_skip_qk_pad_zero_enabled() {
    return freq_attention60_cutile_padded_enabled() &&
           env_flag_enabled("CUDASEP_ENABLE_FREQ_SPLIT_SKIP_QK_PAD_ZERO") &&
           !env_flag_enabled("CUDASEP_DISABLE_FREQ_SPLIT_SKIP_QK_PAD_ZERO");
}

bool freq_split_skip_v_pad_zero_enabled() {
    return freq_split_skip_qk_pad_zero_enabled() &&
           env_flag_enabled("CUDASEP_ENABLE_FREQ_SPLIT_SKIP_V_PAD_ZERO") &&
           !env_flag_enabled("CUDASEP_DISABLE_FREQ_SPLIT_SKIP_V_PAD_ZERO");
}

bool freq_attention60_v32_enabled() {
    return freq_attention60_cutile_padded_enabled() &&
           env_flag_enabled("CUDASEP_ENABLE_FREQ_ATTENTION60_V32") &&
           !env_flag_enabled("CUDASEP_DISABLE_FREQ_ATTENTION60_V32");
}

bool time_attention_fused_fragscale_enabled() {
    return g_quantize_bf16 &&
           (custom_bf16_path_enabled() || env_flag_enabled("CUDASEP_TIME_ATTENTION_FUSED_FRAGSCALE"));
}

bool time_attention_cutile_q32_enabled() {
    return g_quantize_bf16 &&
           env_flag_enabled("CUDASEP_ENABLE_TIME_ATTENTION_CUTILE_Q32") &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_ATTENTION_CUTILE_Q32");
}

bool time_attention_cutile_q16_enabled() {
    return g_quantize_bf16 &&
           env_flag_enabled("CUDASEP_ENABLE_TIME_ATTENTION_CUTILE_Q16") &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_ATTENTION_CUTILE_Q16");
}

bool time_attention_cutile_q64_enabled() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_ATTENTION_CUTILE_Q64");
}

bool time_attention_cutile_q128_enabled() {
    return g_quantize_bf16 &&
           env_flag_enabled("CUDASEP_ENABLE_TIME_ATTENTION_CUTILE_Q128") &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_ATTENTION_CUTILE_Q128");
}

bool time_attention_cutile_k32_enabled() {
    return g_quantize_bf16 &&
           env_flag_enabled("CUDASEP_ENABLE_TIME_ATTENTION_CUTILE_K32") &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_ATTENTION_CUTILE_K32");
}

bool time_attention_cutile_split_tail_k32_enabled() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_ATTENTION_CUTILE_K32") &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_ATTENTION_CUTILE_SPLIT_TAIL_K32");
}

bool time_attention_cutile_split_tail_q32_enabled() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_ATTENTION_CUTILE_SPLIT_TAIL_Q32");
}

bool time_attention_cutile_k128_enabled() {
    return g_quantize_bf16 &&
           env_flag_enabled("CUDASEP_ENABLE_TIME_ATTENTION_CUTILE_K128") &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_ATTENTION_CUTILE_K128");
}

bool time_attention_cutile_split_tail_enabled() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_ATTENTION_CUTILE_SPLIT_TAIL");
}

bool time_attention_cutile_exp2_enabled() {
    return g_quantize_bf16 && !env_flag_enabled("CUDASEP_DISABLE_TIME_ATTENTION_EXP2");
}

bool time_attention_cutile_skip_keytail_enabled() {
    return g_quantize_bf16 &&
           env_flag_enabled("CUDASEP_ENABLE_TIME_ATTENTION_SKIP_KEYTAIL") &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_ATTENTION_SKIP_KEYTAIL");
}

bool time_attention_approx_softmax_enabled_impl() {
    return g_quantize_bf16 &&
           env_flag_enabled("CUDASEP_ENABLE_TIME_ATTENTION_APPROX_SOFTMAX") &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_ATTENTION_APPROX_SOFTMAX");
}

bool time_attention_approx_softmax_enabled_for_call() {
    if (!time_attention_approx_softmax_enabled_impl()) return false;
    static int start_call = env_nonnegative_int_or(
        "CUDASEP_TIME_ATTENTION_APPROX_SOFTMAX_START_CALL", 0);
    static int max_calls = env_nonnegative_int_or(
        "CUDASEP_TIME_ATTENTION_APPROX_SOFTMAX_MAX_CALLS", -1);
    static int target_chunk = env_nonnegative_int_or(
        "CUDASEP_TIME_ATTENTION_APPROX_SOFTMAX_CHUNK", -1);
    static int target_depth = env_nonnegative_int_or(
        "CUDASEP_TIME_ATTENTION_APPROX_SOFTMAX_DEPTH", -1);
    static int calls = 0;
    int call = calls++;
    if (target_chunk >= 0 && g_time_attention_context_chunk != target_chunk) {
        return false;
    }
    if (target_depth >= 0 && g_time_attention_context_depth != target_depth) {
        return false;
    }
    if (call < start_call) return false;
    if (max_calls < 0) return true;
    return call < start_call + max_calls;
}

bool time_attention_stats_enabled_for_current_context_impl() {
    if (!env_flag_enabled("CUDASEP_ENABLE_TIME_ATTENTION_STATS")) return false;
    static int target_chunk = env_nonnegative_int_or(
        "CUDASEP_TIME_ATTENTION_STATS_CHUNK", -1);
    static int target_depth = env_nonnegative_int_or(
        "CUDASEP_TIME_ATTENTION_STATS_DEPTH", -1);
    if (target_chunk >= 0 && g_time_attention_context_chunk != target_chunk) {
        return false;
    }
    if (target_depth >= 0 && g_time_attention_context_depth != target_depth) {
        return false;
    }
    return true;
}

bool mask_scatter_bf16_enabled() {
    return bf16_experiment_enabled("CUDASEP_MASK_SCATTER_BF16");
}

bool glu_bf16_output_enabled() {
    return bf16_experiment_enabled("CUDASEP_GLU_BF16_OUTPUT");
}

bool tanh_bf16_input_enabled() {
    return remaining_bf16_experiment_enabled("CUDASEP_TANH_BF16_INPUT");
}

bool glu_bf16_input_enabled() {
    return remaining_bf16_experiment_enabled("CUDASEP_GLU_BF16_INPUT");
}

bool linear_glu_last_dim_fused_enabled() {
    return g_quantize_bf16 &&
           env_flag_enabled("CUDASEP_ENABLE_LINEAR_GLU_LAST_DIM_FUSED") &&
           !env_flag_enabled("CUDASEP_DISABLE_LINEAR_GLU_LAST_DIM_FUSED");
}

bool linear_glu_last_dim_fused_tk32_enabled() {
    return linear_glu_last_dim_fused_enabled() &&
           env_flag_enabled("CUDASEP_ENABLE_LINEAR_GLU_LAST_DIM_FUSED_TK32");
}

bool linear_direct_bf16_output_enabled() {
    return remaining_bf16_experiment_enabled("CUDASEP_LINEAR_DIRECT_BF16_OUTPUT");
}

bool gather_bf16_output_enabled() {
    return remaining_bf16_experiment_enabled("CUDASEP_GATHER_BF16_OUTPUT");
}

bool residual_bf16_enabled_impl() {
    return remaining_bf16_experiment_enabled("CUDASEP_RESIDUAL_BF16");
}

bool bias_bf16_enabled_impl() {
    return remaining_bf16_experiment_enabled("CUDASEP_BIAS_BF16");
}

bool norm_gamma_bf16_enabled_impl() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_NORM_GAMMA_BF16");
}

bool bands_per_freq_bf16_enabled_impl() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_BANDS_PER_FREQ_BF16");
}

bool rotary_freqs_bf16_enabled_impl() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_ROTARY_FREQS_BF16");
}

bool linear_cutile_static_bf16_output_enabled() {
    return g_quantize_bf16 && !env_flag_enabled("CUDASEP_DISABLE_LINEAR_CUTILE_STATIC_BF16");
}

bool linear_bkn_long_path_enabled() {
    return g_quantize_bf16 && !env_flag_enabled("CUDASEP_DISABLE_LINEAR_BKN_LONG");
}

bool linear_bkn_qkv_tk16_enabled() {
    return g_quantize_bf16 && !env_flag_enabled("CUDASEP_DISABLE_LINEAR_BKN_QKV_TK16");
}

bool linear_bkn_qkv_m64_tn128_enabled() {
    return linear_bkn_qkv_tk16_enabled() &&
           !env_flag_enabled("CUDASEP_DISABLE_LINEAR_BKN_QKV_M64_TN128");
}

bool linear_bkn_qkv_m64_tn128_lat1_enabled() {
    return linear_bkn_qkv_m64_tn128_enabled() &&
           !env_flag_enabled("CUDASEP_DISABLE_LINEAR_BKN_QKV_M64_TN128_LAT1");
}

bool time_qkv_bkn_m64_tn128_enabled() {
    return g_quantize_bf16 && !env_flag_enabled("CUDASEP_DISABLE_TIME_QKV_BKN_M64_TN128");
}

bool linear_bkn_ffn_long_path_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_LINEAR_BKN_FFN_LONG");
}

bool linear_padded_bias_no_round_enabled() {
    return g_quantize_bf16 &&
           env_flag_enabled("CUDASEP_ENABLE_LINEAR_PADDED_BIAS_NO_ROUND") &&
           !env_flag_enabled("CUDASEP_DISABLE_LINEAR_PADDED_BIAS_NO_ROUND");
}

bool time_qkv_split_contig_path_enabled_impl() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_QKV_SPLIT_CONTIG_PATH");
}

bool time_qkv_fused_rotary_producer_enabled_impl() {
    return time_qkv_split_contig_path_enabled_impl() &&
           env_flag_enabled("CUDASEP_ENABLE_TIME_QKV_FUSED_ROTARY_PRODUCER") &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_QKV_FUSED_ROTARY_PRODUCER");
}

bool time_qkv_pair_rotary_producer_enabled_impl() {
    return time_qkv_split_contig_path_enabled_impl() &&
           env_flag_enabled("CUDASEP_ENABLE_TIME_QKV_PAIR_ROTARY_PRODUCER") &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_QKV_PAIR_ROTARY_PRODUCER");
}

bool time_rotary_row_tile_enabled_impl() {
    return g_quantize_bf16 &&
           env_flag_enabled("CUDASEP_ENABLE_TIME_ROTARY_ROW_TILE") &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_ROTARY_ROW_TILE");
}

int time_rotary_tile_size() {
    static int tile_size = []() {
        const char* raw = std::getenv("CUDASEP_TIME_ROTARY_TILE");
        if (raw == nullptr) return kTimeRotaryTileDefault;
        int parsed = std::atoi(raw);
        if (parsed == 128 || parsed == kTile || parsed == 512 || parsed == 1024) {
            return parsed;
        }
        return kTimeRotaryTileDefault;
    }();
    return tile_size;
}

bool time_q_rotary_in_attention_enabled_impl() {
    return time_qkv_split_contig_path_enabled_impl() &&
           env_flag_enabled("CUDASEP_ENABLE_TIME_Q_ROTARY_IN_ATTENTION") &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_Q_ROTARY_IN_ATTENTION");
}

bool time_q_rotary_in_attention_q32_enabled_impl() {
    return time_q_rotary_in_attention_enabled_impl() &&
           env_flag_enabled("CUDASEP_ENABLE_TIME_Q_ROTARY_IN_ATTENTION_Q32") &&
           !env_flag_enabled("CUDASEP_DISABLE_TIME_Q_ROTARY_IN_ATTENTION_Q32");
}

bool linear_cutile_static_small_bf16_output_enabled() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_LINEAR_CUTILE_STATIC_SMALL_BF16");
}

bool linear_cutile_gate_sigmoid_bf16_output_enabled() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_LINEAR_CUTILE_GATE_SIGMOID_BF16");
}

bool linear_hard_gelu_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_LINEAR_HARD_GELU");
}

bool linear_quick_gelu_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_LINEAR_QUICK_GELU");
}

bool linear_tanh_gelu_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_LINEAR_TANH_GELU");
}

bool linear_gelu_split_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_LINEAR_GELU_SPLIT");
}

bool linear_ffn1_m16n128_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_LINEAR_FFN1_M16N128");
}

bool linear_ffn1_m64n32_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_LINEAR_FFN1_M64N32");
}

bool linear_ffn1_m32n128_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_LINEAR_FFN1_M32N128");
}

bool linear_ffn2_m32n128_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_LINEAR_FFN2_M32N128");
}

bool linear_qkv_m32n128_enabled() {
    return g_quantize_bf16 && !env_flag_enabled("CUDASEP_DISABLE_LINEAR_QKV_M32N128");
}

bool linear_qkv_m32n256_enabled() {
    return g_quantize_bf16 && !env_flag_enabled("CUDASEP_DISABLE_LINEAR_QKV_M32N256");
}

bool linear_attn_out_m32n128_enabled() {
    return g_quantize_bf16 && !env_flag_enabled("CUDASEP_DISABLE_LINEAR_ATTN_OUT_M32N128");
}

bool attention_out_residual_enabled() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_ATTENTION_OUT_RESIDUAL");
}

int attention_out_residual_tile_n() {
    static int tile_n = []() {
        const char* raw = std::getenv("CUDASEP_ATTENTION_OUT_RESIDUAL_TILE_N");
        if (raw == nullptr) return 128;
        int parsed = std::atoi(raw);
        return (parsed == 64) ? 64 : 128;
    }();
    return tile_n;
}

bool ffn12_fused_cutile_enabled() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_FFN12_FUSED_CUTILE");
}

bool ffn12_fused_hard_gelu_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_FFN12_FUSED_HARD_GELU");
}

bool ffn12_fused_quick_gelu_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_FFN12_FUSED_QUICK_GELU");
}

bool ffn12_fused_tanh_gelu_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_FFN12_FUSED_TANH_GELU");
}

bool ffn12_fused_poly5_gelu_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_FFN12_FUSED_POLY5_GELU");
}

bool ffn12_fused_poly7_gelu_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_FFN12_FUSED_POLY7_GELU");
}

bool ffn12_fused_odd5_gelu_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_FFN12_FUSED_ODD5_GELU");
}

bool ffn12_fused_tinyblend_gelu_enabled() {
    return g_quantize_bf16 && env_flag_enabled("CUDASEP_ENABLE_FFN12_FUSED_TINYBLEND_GELU");
}

bool ffn12_fused_poly9_gelu_enabled() {
    if (!g_quantize_bf16 || env_flag_enabled("CUDASEP_DISABLE_FFN12_FUSED_POLY9_GELU")) {
        return false;
    }
    if (env_flag_enabled("CUDASEP_ENABLE_FFN12_FUSED_POLY9_GELU")) {
        return true;
    }
    return !ffn12_fused_poly7_gelu_enabled() &&
           !ffn12_fused_odd5_gelu_enabled() &&
           !ffn12_fused_poly5_gelu_enabled() &&
           !ffn12_fused_tinyblend_gelu_enabled() &&
           !ffn12_fused_tanh_gelu_enabled() &&
           !ffn12_fused_quick_gelu_enabled() &&
           !ffn12_fused_hard_gelu_enabled() &&
           !linear_tanh_gelu_enabled() &&
           !linear_quick_gelu_enabled() &&
           !linear_hard_gelu_enabled();
}

bool ffn12_fused_split2_output_enabled() {
    return g_quantize_bf16 &&
           !env_flag_enabled("CUDASEP_DISABLE_FFN12_FUSED_SPLIT2_OUTPUT");
}

bool ffn12_fused_split2_pairh32_enabled() {
    return g_quantize_bf16 &&
           ffn12_fused_split2_output_enabled() &&
           !env_flag_enabled("CUDASEP_DISABLE_FFN12_FUSED_SPLIT2_PAIRH32");
}

bool ffn12_fused_pairh32_tk64_enabled() {
    return ffn12_fused_split2_pairh32_enabled() &&
           !env_flag_enabled("CUDASEP_DISABLE_FFN12_FUSED_PAIRH32_TK64");
}

bool ffn12_fused_residual_enabled() {
    return ffn12_fused_cutile_enabled() &&
           !env_flag_enabled("CUDASEP_DISABLE_FFN12_FUSED_RESIDUAL");
}

bool ffn12_residual_two_kernel_enabled() {
    return ffn12_fused_residual_enabled() &&
           !env_flag_enabled("CUDASEP_DISABLE_FFN12_RESIDUAL_TWO_KERNEL");
}

bool split_qkv_time_cutile_fixed_enabled() {
    return g_quantize_bf16 && !env_flag_enabled("CUDASEP_DISABLE_SPLIT_QKV_TIME_CUTILE_FIXED");
}

bool qkv_freq_rotary_cutile_fused_enabled() {
    return g_quantize_bf16 &&
           env_flag_enabled("CUDASEP_ENABLE_QKV_FREQ_ROTARY_CUTILE_FUSED") &&
           !env_flag_enabled("CUDASEP_DISABLE_QKV_FREQ_ROTARY_CUTILE_FUSED");
}

bool qkv_freq_rotary_bkn_cutile_fused_enabled() {
    return linear_bkn_long_path_enabled() &&
           env_flag_enabled("CUDASEP_ENABLE_QKV_FREQ_ROTARY_BKN_CUTILE_FUSED") &&
           !env_flag_enabled("CUDASEP_DISABLE_QKV_FREQ_ROTARY_BKN_CUTILE_FUSED");
}

int qkv_freq_rotary_bkn_cutile_tile_n() {
    static int tile_n = []() {
        const char* raw = std::getenv("CUDASEP_QKV_FREQ_ROTARY_BKN_CUTILE_TILE_N");
        if (raw == nullptr) return 256;
        int parsed = std::atoi(raw);
        if (parsed == 64 || parsed == 128) return parsed;
        return 256;
    }();
    return tile_n;
}

bool gate_merge_token_d64_enabled() {
    return g_quantize_bf16 && !env_flag_enabled("CUDASEP_DISABLE_GATE_MERGE_TOKEN_D64");
}

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_erf_approx(TileT x) {
    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto sign = ct::select(x < zero, zero - one, one);
    auto ax = ct::select(x < zero, zero - x, x);
    auto t = one / (one + 0.3275911f * ax);
    t = bf16_round_if<FullBF16>(t);
    auto poly = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t -
                  0.284496736f) *
                     t +
                 0.254829592f) *
                t;
    poly = bf16_round_if<FullBF16>(poly);
    auto erf_approx = sign * (one - poly * ct::exp(zero - ax * ax));
    erf_approx = bf16_round_if<FullBF16>(erf_approx);
    auto gelu = 0.5f * x * (one + erf_approx);
    return bf16_round_if<FullBF16>(gelu);
}

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_hard_approx(TileT x) {
    auto zero = x * 0.0f;
    auto gate = ct::min(ct::max(0.5f + 0.2f * x, zero), zero + 1.0f);
    gate = bf16_round_if<FullBF16>(gate);
    auto gelu = x * gate;
    return bf16_round_if<FullBF16>(gelu);
}

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_quick_approx(TileT x) {
    auto sigmoid = 1.0f / (1.0f + ct::exp(-1.702f * x));
    sigmoid = bf16_round_if<FullBF16>(sigmoid);
    auto gelu = x * sigmoid;
    return bf16_round_if<FullBF16>(gelu);
}

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_tanh_approx(TileT x) {
    auto x2 = x * x;
    x2 = bf16_round_if<FullBF16>(x2);
    auto cubic = x2 * x;
    cubic = bf16_round_if<FullBF16>(cubic);
    auto inner = 0.7978845608f * (x + 0.044715f * cubic);
    inner = bf16_round_if<FullBF16>(inner);
    auto gate = 0.5f * (1.0f + tanh(inner));
    gate = bf16_round_if<FullBF16>(gate);
    auto gelu = x * gate;
    return bf16_round_if<FullBF16>(gelu);
}

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_erf_poly5_l25(TileT x) {
    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto ax = ct::select(x < zero, zero - x, x);
    auto z = ax * ax;
    z = bf16_round_if<FullBF16>(z);
    auto p = (((0.000677416775f * z - 0.0121774335f) * z +
               0.0889425898f) * z - 0.361254819f) * z +
             1.12684393f;
    p = bf16_round_if<FullBF16>(p);
    auto erf_abs = ct::min(ct::max(ax * p, zero), one);
    erf_abs = bf16_round_if<FullBF16>(erf_abs);
    auto erf_approx = ct::select(x < zero, zero - erf_abs, erf_abs);
    auto gelu = 0.5f * x * (one + erf_approx);
    return bf16_round_if<FullBF16>(gelu);
}

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_erf_poly7_l25(TileT x) {
    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto ax = ct::select(x < zero, zero - x, x);
    auto z = ax * ax;
    z = bf16_round_if<FullBF16>(z);
    auto p = ((((((0.0000119948033f * z - 0.000310497426f) * z +
                  0.00352976049f) * z - 0.0238667561f) * z +
                0.110178845f) * z - 0.37522094f) * z +
              1.12832882f);
    p = bf16_round_if<FullBF16>(p);
    auto erf_abs = ct::min(ct::max(ax * p, zero), one);
    erf_abs = bf16_round_if<FullBF16>(erf_abs);
    auto erf_approx = ct::select(x < zero, zero - erf_abs, erf_abs);
    auto gelu = 0.5f * x * (one + erf_approx);
    return bf16_round_if<FullBF16>(gelu);
}

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_erf_poly9_l30(TileT x) {
    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto ax = ct::select(x < zero, zero - x, x);
    auto z = ax * ax;
    z = bf16_round_if<FullBF16>(z);
    auto p = ((((((((0.00000005422539767f * z - 0.000002440964777f) * z +
                    0.00004855766724f) * z - 0.0005709642654f) * z +
                  0.004507274577f) * z - 0.02579950512f) * z +
                0.1120213868f) * z - 0.3758834075f) * z +
              1.128367753f);
    p = bf16_round_if<FullBF16>(p);
    auto erf_abs = ct::min(ct::max(ax * p, zero), one);
    erf_abs = bf16_round_if<FullBF16>(erf_abs);
    auto erf_approx = ct::select(x < zero, zero - erf_abs, erf_abs);
    auto gelu = 0.5f * x * (one + erf_approx);
    return bf16_round_if<FullBF16>(gelu);
}

template <bool FullBF16, typename TileT>
static __tile__ auto gelu_erf_odd5_l175(TileT x) {
    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto ax = ct::select(x < zero, zero - x, x);
    auto z = ax * ax;
    z = bf16_round_if<FullBF16>(z);
    auto p = (0.04752145079070458f * z - 0.32203058651122096f) * z +
             1.1212825366624732f;
    p = bf16_round_if<FullBF16>(p);
    auto erf_abs = ct::min(ct::max(ax * p, zero), one);
    erf_abs = bf16_round_if<FullBF16>(erf_abs);
    auto erf_approx = ct::select(x < zero, zero - erf_abs, erf_abs);
    auto gelu = 0.5f * x * (one + erf_approx);
    return bf16_round_if<FullBF16>(gelu);
}

template <int GeluMode, bool FullBF16, typename TileT>
static __tile__ auto gelu_selected(TileT x) {
    if constexpr (GeluMode == kGeluErfPoly9L30) {
        return gelu_erf_poly9_l30<FullBF16>(x);
    } else if constexpr (GeluMode == kGeluErfOdd5L175) {
        return gelu_erf_odd5_l175<FullBF16>(x);
    } else if constexpr (GeluMode == kGeluErfPoly7L25) {
        return gelu_erf_poly7_l25<FullBF16>(x);
    } else if constexpr (GeluMode == kGeluErfPoly5L25) {
        return gelu_erf_poly5_l25<FullBF16>(x);
    } else if constexpr (GeluMode == kGeluTanh) {
        return gelu_tanh_approx<FullBF16>(x);
    } else if constexpr (GeluMode == kGeluQuick) {
        return gelu_quick_approx<FullBF16>(x);
    } else if constexpr (GeluMode == kGeluHard) {
        return gelu_hard_approx<FullBF16>(x);
    } else {
        static_assert(GeluMode == kGeluErf);
        return gelu_erf_approx<FullBF16>(x);
    }
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          bool AddBias,
          bool ApplyGelu,
          int GeluMode,
          bool FullBF16,
          bool AddResidual = false>
__tile_global__ void linear_cutile_static_full_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_nt,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ c,
    const __nv_bfloat16* __restrict__ residual) {
    static_assert(!ApplyGelu || AddBias);
    static_assert(M % TM == 0);
    static_assert(N % TN == 0);
    static_assert(K % TK == 0);
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    if constexpr (AddBias) {
        bias = ct::assume_aligned(bias, 16_ic);
    }
    c = ct::assume_aligned(c, 16_ic);
    if constexpr (AddResidual) {
        residual = ct::assume_aligned(residual, 16_ic);
    }

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_nt, ct::shape<K, N>{}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
    }

    auto value = acc;
    if constexpr (AddBias) {
        value = bf16_round(value);
        I64OutTile local = ct::iota<I64OutTile>();
        auto cols = static_cast<long long>(tile_n) * TN + (local % TN);
        auto bias_values = ct::element_cast<float>(ct::load(bias + cols));
        value = value + bias_values;
        value = bf16_round_if<FullBF16>(value);
    }
    if constexpr (ApplyGelu) {
        value = gelu_selected<GeluMode, FullBF16>(value);
    }
    auto out_value = ct::element_cast<__nv_bfloat16>(value);
    if constexpr (AddResidual) {
        auto residual_view = ct::partition_view{
            ct::tensor_span{residual, ct::shape<M, N>{}},
            ct::shape<TM, TN>{}
        };
        out_value = out_value + residual_view.load(tile_m, tile_n);
    }
    c_view.store(out_value, tile_m, tile_n);
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          bool AddBias,
          bool ApplyGelu,
          int GeluMode,
          bool FullBF16,
          int LoadLatency = 0>
__tile_global__ void linear_cutile_static_full_bkn_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ c) {
    static_assert(!ApplyGelu || AddBias);
    static_assert(M % TM == 0);
    static_assert(N % TN == 0);
    static_assert(K % TK == 0);
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using BTile = ct::tile<__nv_bfloat16, ct::shape<TK, TN>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    if constexpr (AddBias) {
        bias = ct::assume_aligned(bias, 16_ic);
    }
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        if constexpr (LoadLatency > 0) {
            ATile a_tile;
            BTile b_tile;
            [[ cutile::hint(0, latency=LoadLatency) ]]
            a_tile = a_view.load(tile_m, kk);
            [[ cutile::hint(0, latency=LoadLatency) ]]
            b_tile = b_view.load(kk, tile_n);
            acc = ct::mma(a_tile, b_tile, acc);
        } else {
            acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
        }
    }

    auto value = acc;
    if constexpr (AddBias) {
        value = bf16_round(value);
        I64OutTile local = ct::iota<I64OutTile>();
        auto cols = static_cast<long long>(tile_n) * TN + (local % TN);
        auto bias_values = ct::element_cast<float>(ct::load(bias + cols));
        value = value + bias_values;
        value = bf16_round_if<FullBF16>(value);
    }
    if constexpr (ApplyGelu) {
        value = gelu_selected<GeluMode, FullBF16>(value);
    }
    auto out_value = ct::element_cast<__nv_bfloat16>(value);
    if constexpr (LoadLatency > 0) {
        [[ cutile::hint(0, latency=LoadLatency) ]]
        c_view.store(out_value, tile_m, tile_n);
    } else {
        c_view.store(out_value, tile_m, tile_n);
    }
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int TailStart,
          int LoadLatency = 0>
__tile_global__ void linear_cutile_static_bkn_masked_tail_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ c) {
    static_assert(TailStart % TM == 0);
    static_assert(TailStart < M);
    static_assert(N % TN == 0);
    static_assert(K % TK == 0);
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using BTile = ct::tile<__nv_bfloat16, ct::shape<TK, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tail_tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    int tile_m = TailStart / TM + tail_tile_m;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        if constexpr (LoadLatency > 0) {
            ATile a_tile;
            BTile b_tile;
            [[ cutile::hint(0, latency=LoadLatency) ]]
            a_tile = a_view.load_masked(tile_m, kk);
            [[ cutile::hint(0, latency=LoadLatency) ]]
            b_tile = b_view.load(kk, tile_n);
            acc = ct::mma(a_tile, b_tile, acc);
        } else {
            acc = ct::mma(a_view.load_masked(tile_m, kk), b_view.load(kk, tile_n), acc);
        }
    }

    c_view.store_masked(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
}

template <int N,
          int K,
          bool AddBias,
          bool ApplyGelu,
          int GeluMode,
          bool FullBF16,
          bool AddResidual = false>
__tile_global__ void linear_cutile_tail_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_nt,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ c,
    const __nv_bfloat16* __restrict__ residual,
    int m,
    int row_start) {
    static_assert(!ApplyGelu || AddBias);

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    if constexpr (AddBias) {
        bias = ct::assume_aligned(bias, 16_ic);
    }
    c = ct::assume_aligned(c, 16_ic);
    if constexpr (AddResidual) {
        residual = ct::assume_aligned(residual, 16_ic);
    }

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    long long total = (long long)(m - row_start) * N;
    auto in_bounds = idx < total;
    auto row = row_start + idx / N;
    auto col = idx % N;
    F32Tile acc = ct::element_cast<float>(idx * 0LL);
#pragma unroll
    for (int kk = 0; kk < K; ++kk) {
        auto av = ct::element_cast<float>(
            ct::load_masked(a + row * K + kk, in_bounds));
        auto bv = ct::element_cast<float>(
            ct::load_masked(b_nt + col * K + kk, in_bounds));
        acc = acc + av * bv;
    }
    if constexpr (AddBias) {
        acc = bf16_round(acc);
        auto bias_values = ct::element_cast<float>(ct::load_masked(bias + col, in_bounds));
        acc = acc + bias_values;
        acc = bf16_round_if<FullBF16>(acc);
    }
    if constexpr (ApplyGelu) {
        acc = gelu_selected<GeluMode, FullBF16>(acc);
    }
    auto out_value = ct::element_cast<__nv_bfloat16>(acc);
    if constexpr (AddResidual) {
        out_value = out_value + ct::load_masked(residual + row * N + col, in_bounds);
    }
    ct::store_masked(c + row * N + col, out_value, in_bounds);
}

template <int TM, int TN, int TK, int M, int N, int K, int LoadLatency = 0>
__tile_global__ void qkv_bkn_split_contig_static_full_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v) {
    static_assert(M % TM == 0);
    static_assert(N == 3 * kQkvFusedHeads * kTimeAttnD);
    static_assert((N / 3) % TN == 0);
    static_assert(K % TK == 0);
    constexpr int kComponentTiles = (N / 3) / TN;
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using BTile = ct::tile<__nv_bfloat16, ct::shape<TK, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k_out = ct::assume_aligned(k_out, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };
    auto q_view = ct::partition_view{
        ct::tensor_span{q, ct::shape<M, N / 3>{}},
        ct::shape<TM, TN>{}
    };
    auto k_view = ct::partition_view{
        ct::tensor_span{k_out, ct::shape<M, N / 3>{}},
        ct::shape<TM, TN>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v, ct::shape<M, N / 3>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        if constexpr (LoadLatency > 0) {
            ATile a_tile;
            BTile b_tile;
            [[ cutile::hint(0, latency=LoadLatency) ]]
            a_tile = a_view.load(tile_m, kk);
            [[ cutile::hint(0, latency=LoadLatency) ]]
            b_tile = b_view.load(kk, tile_n);
            acc = ct::mma(a_tile, b_tile, acc);
        } else {
            acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
        }
    }

    auto out = ct::element_cast<__nv_bfloat16>(acc);
    if (tile_n < kComponentTiles) {
        if constexpr (LoadLatency > 0) {
            [[ cutile::hint(0, latency=LoadLatency) ]]
            q_view.store(out, tile_m, tile_n);
        } else {
            q_view.store(out, tile_m, tile_n);
        }
    } else if (tile_n < 2 * kComponentTiles) {
        if constexpr (LoadLatency > 0) {
            [[ cutile::hint(0, latency=LoadLatency) ]]
            k_view.store(out, tile_m, tile_n - kComponentTiles);
        } else {
            k_view.store(out, tile_m, tile_n - kComponentTiles);
        }
    } else {
        if constexpr (LoadLatency > 0) {
            [[ cutile::hint(0, latency=LoadLatency) ]]
            v_view.store(out, tile_m, tile_n - 2 * kComponentTiles);
        } else {
            v_view.store(out, tile_m, tile_n - 2 * kComponentTiles);
        }
    }
}

template <int N, int K>
__tile_global__ void qkv_bkn_split_contig_tail_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v,
    int m,
    int row_start) {
    static_assert(N == 3 * kQkvFusedHeads * kTimeAttnD);

    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k_out = ct::assume_aligned(k_out, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    long long total = (long long)(m - row_start) * N;
    auto in_bounds = idx < total;
    auto row = row_start + idx / N;
    auto col = idx % N;
    F32Tile acc = ct::element_cast<float>(idx * 0LL);
#pragma unroll
    for (int kk = 0; kk < K; ++kk) {
        auto av = ct::element_cast<float>(
            ct::load_masked(a + row * K + kk, in_bounds));
        auto bv = ct::element_cast<float>(
            ct::load_masked(b + col * K + kk, in_bounds));
        acc = acc + av * bv;
    }

    auto part = col / (kQkvFusedHeads * kTimeAttnD);
    auto head_col = col % (kQkvFusedHeads * kTimeAttnD);
    auto batch = row / kTimeAttnN;
    auto token = row - batch * kTimeAttnN;
    auto out_offset =
        (batch * kTimeAttnN + token) * (kQkvFusedHeads * kTimeAttnD) + head_col;
    auto out = ct::element_cast<__nv_bfloat16>(acc);
    ct::store_masked(q + out_offset, out, in_bounds && (part == 0LL));
    ct::store_masked(k_out + out_offset, out, in_bounds && (part == 1LL));
    ct::store_masked(v + out_offset, out, in_bounds && (part == 2LL));
}

template <int TM, int TN, int TK, int M, int N, int K, int TailStart, int LoadLatency = 0>
__tile_global__ void qkv_bkn_split_contig_masked_tail_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v) {
    static_assert(TailStart % TM == 0);
    static_assert(TailStart < M);
    static_assert(N == 3 * kQkvFusedHeads * kTimeAttnD);
    static_assert((N / 3) % TN == 0);
    static_assert(K % TK == 0);
    constexpr int kComponentTiles = (N / 3) / TN;
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using BTile = ct::tile<__nv_bfloat16, ct::shape<TK, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k_out = ct::assume_aligned(k_out, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };
    auto q_view = ct::partition_view{
        ct::tensor_span{q, ct::shape<M, N / 3>{}},
        ct::shape<TM, TN>{}
    };
    auto k_view = ct::partition_view{
        ct::tensor_span{k_out, ct::shape<M, N / 3>{}},
        ct::shape<TM, TN>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v, ct::shape<M, N / 3>{}},
        ct::shape<TM, TN>{}
    };

    auto [tail_tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    int tile_m = TailStart / TM + tail_tile_m;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        if constexpr (LoadLatency > 0) {
            ATile a_tile;
            BTile b_tile;
            [[ cutile::hint(0, latency=LoadLatency) ]]
            a_tile = a_view.load_masked(tile_m, kk);
            [[ cutile::hint(0, latency=LoadLatency) ]]
            b_tile = b_view.load(kk, tile_n);
            acc = ct::mma(a_tile, b_tile, acc);
        } else {
            acc = ct::mma(a_view.load_masked(tile_m, kk), b_view.load(kk, tile_n), acc);
        }
    }

    auto out = ct::element_cast<__nv_bfloat16>(acc);
    if (tile_n < kComponentTiles) {
        q_view.store_masked(out, tile_m, tile_n);
    } else if (tile_n < 2 * kComponentTiles) {
        k_view.store_masked(out, tile_m, tile_n - kComponentTiles);
    } else {
        v_view.store_masked(out, tile_m, tile_n - 2 * kComponentTiles);
    }
}

template <int TM, int TN, int TK, int M, int N, int K, typename TrigT = float>
__tile_global__ void qkv_bkn_rotary_split_contig_static_full_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v,
    bool full_bf16) {
    static_assert(M % TM == 0);
    static_assert(N == 3 * kQkvFusedHeads * kTimeAttnD);
    static_assert((N / 3) % TN == 0);
    static_assert(TN == 4 * kTimeAttnD);
    static_assert(K % TK == 0);
    constexpr int kHalfDim = kTimeAttnD / 2;
    constexpr int kHeadsPerTile = TN / kTimeAttnD;
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using Acc4Tile = ct::tile<float, ct::shape<TM, kHeadsPerTile, kHalfDim, 2>>;
    using PairTile = ct::tile<float, ct::shape<TM, kHeadsPerTile, kHalfDim, 1>>;
    using I64PairTile = ct::tile<long long, ct::shape<TM, kHeadsPerTile, kHalfDim, 1>>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k_out = ct::assume_aligned(k_out, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_kn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };
    auto q_view = ct::partition_view{
        ct::tensor_span{q, ct::shape<M, N / 3>{}},
        ct::shape<TM, TN>{}
    };
    auto k_view = ct::partition_view{
        ct::tensor_span{k_out, ct::shape<M, N / 3>{}},
        ct::shape<TM, TN>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v, ct::shape<M, N / 3>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
    }

    if (tile_n < 4) {
        Acc4Tile acc4 = ct::reshape(acc, ct::shape<TM, kHeadsPerTile, kHalfDim, 2>{});
        PairTile even = ct::extract(acc4, ct::shape<TM, kHeadsPerTile, kHalfDim, 1>{},
                                    0, 0, 0, 0);
        PairTile odd = ct::extract(acc4, ct::shape<TM, kHeadsPerTile, kHalfDim, 1>{},
                                   0, 0, 0, 1);
        even = bf16_round(even);
        odd = bf16_round(odd);

        I64PairTile local = ct::iota<I64PairTile>();
        auto rows = static_cast<long long>(tile_m) * TM +
                    local / (kHeadsPerTile * kHalfDim);
        auto pair = local % kHalfDim;
        auto token = rows % kTimeAttnN;
        PairTile c = ct::element_cast<float>(
            ct::load(cos_f + token * kHalfDim + pair));
        PairTile s = ct::element_cast<float>(
            ct::load(sin_f + token * kHalfDim + pair));
        c = ct::select(full_bf16, bf16_round(c), c);
        s = ct::select(full_bf16, bf16_round(s), s);

        auto rot_even = even * c - odd * s;
        auto rot_odd = even * s + odd * c;
        rot_even = ct::select(full_bf16, bf16_round(rot_even), rot_even);
        rot_odd = ct::select(full_bf16, bf16_round(rot_odd), rot_odd);
        auto rotated = ct::reshape(ct::cat<3>(rot_even, rot_odd),
                                   ct::shape<TM, TN>{});
        auto out = ct::element_cast<__nv_bfloat16>(rotated);
        if (tile_n < 2) {
            q_view.store(out, tile_m, tile_n);
        } else {
            k_view.store(out, tile_m, tile_n - 2);
        }
    } else {
        v_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n - 4);
    }
}

template <int N, int K, typename TrigT = float>
__tile_global__ void qkv_bkn_rotary_split_contig_tail_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v,
    int m,
    int row_start,
    bool full_bf16) {
    static_assert(N == 3 * kQkvFusedHeads * kTimeAttnD);
    constexpr int kHeadStride = kQkvFusedHeads * kTimeAttnD;
    constexpr int kHalfDim = kTimeAttnD / 2;

    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k_out = ct::assume_aligned(k_out, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    long long total = (long long)(m - row_start) * N;
    auto in_bounds = idx < total;
    auto row = row_start + idx / N;
    auto col = idx % N;
    auto part = col / kHeadStride;
    auto head_col = col % kHeadStride;
    auto dim = head_col % kTimeAttnD;
    auto pair_delta =
        ct::select((dim % 2LL) == 0LL, dim * 0LL + 1LL, dim * 0LL - 1LL);
    auto pair_col = col + pair_delta;
    auto needs_rot = in_bounds && (part < 2LL);

    F32Tile acc = ct::element_cast<float>(idx * 0LL);
    F32Tile pair_acc = ct::element_cast<float>(idx * 0LL);
#pragma unroll
    for (int kk = 0; kk < K; ++kk) {
        auto av = ct::element_cast<float>(
            ct::load_masked(a + row * K + kk, in_bounds));
        auto bv = ct::element_cast<float>(
            ct::load_masked(b + col * K + kk, in_bounds));
        auto pv = ct::element_cast<float>(
            ct::load_masked(b + pair_col * K + kk, needs_rot));
        acc = acc + av * bv;
        pair_acc = pair_acc + av * pv;
    }

    acc = bf16_round(acc);
    pair_acc = bf16_round(pair_acc);
    auto batch = row / kTimeAttnN;
    auto token = row - batch * kTimeAttnN;
    auto pair = dim / 2LL;
    auto c = ct::element_cast<float>(
        ct::load_masked(cos_f + token * kHalfDim + pair, needs_rot));
    auto s = ct::element_cast<float>(
        ct::load_masked(sin_f + token * kHalfDim + pair, needs_rot));
    c = ct::select(full_bf16, bf16_round(c), c);
    s = ct::select(full_bf16, bf16_round(s), s);
    auto even_out = acc * c - pair_acc * s;
    auto odd_out = pair_acc * s + acc * c;
    auto rot = ct::select((dim % 2LL) == 0LL, even_out, odd_out);
    rot = ct::select(full_bf16, bf16_round(rot), rot);
    auto value = ct::select(part < 2LL, rot, acc);

    auto out_offset =
        (batch * kTimeAttnN + token) * kHeadStride + head_col;
    auto out = ct::element_cast<__nv_bfloat16>(value);
    ct::store_masked(q + out_offset, out, in_bounds && (part == 0LL));
    ct::store_masked(k_out + out_offset, out, in_bounds && (part == 1LL));
    ct::store_masked(v + out_offset, out, in_bounds && (part == 2LL));
}

template <int TM, int TPairs, int TK, int M, int K, typename TrigT = float>
__tile_global__ void qkv_time_rotary_pair_split_contig_static_full_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v,
    bool full_bf16) {
    static_assert(M % TM == 0);
    static_assert(K % TK == 0);
    static_assert(TPairs == kTimeAttnD / 2);
    using PairTile = ct::tile<float, ct::shape<TM, TPairs>>;
    using I64PairTile = ct::tile<long long, ct::shape<TM, TPairs>>;
    using WShape = ct::shape<K, TPairs>;
    using WStrides = ct::shape<1, 2 * K>;
    using WLayout = ct::layout_strided<WStrides>;
    using WMapping = typename WLayout::template mapping<WShape>;

    x = ct::assume_aligned(x, 16_ic);
    weight = ct::assume_aligned(weight, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k_out = ct::assume_aligned(k_out, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    auto [tile_m, group_raw, tile_z] = ct::bid();
    (void)tile_z;
    int group = static_cast<int>(group_raw);
    int part = group / kQkvFusedHeads;
    int h = group - part * kQkvFusedHeads;
    int feature_base = part * kQkvFusedHeads * kTimeAttnD + h * kTimeAttnD;

    auto x_view = ct::partition_view{
        ct::tensor_span{x, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto w_even_view = ct::partition_view{
        ct::tensor_span{weight + static_cast<std::size_t>(feature_base) * K,
                        WMapping{WShape{}, WStrides{}}},
        ct::shape<TK, TPairs>{}
    };
    auto w_odd_view = ct::partition_view{
        ct::tensor_span{weight + static_cast<std::size_t>(feature_base + 1) * K,
                        WMapping{WShape{}, WStrides{}}},
        ct::shape<TK, TPairs>{}
    };

    auto even = ct::full<PairTile>(0.0f);
    auto odd = ct::full<PairTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        auto x_tile = x_view.load(tile_m, kk);
        even = ct::mma(x_tile, w_even_view.load(kk, 0), even);
        odd = ct::mma(x_tile, w_odd_view.load(kk, 0), odd);
    }

    even = bf16_round(even);
    odd = bf16_round(odd);

    I64PairTile local = ct::iota<I64PairTile>();
    auto rows = static_cast<long long>(tile_m) * TM + local / TPairs;
    auto pair = local % TPairs;
    auto token = rows % kTimeAttnN;
    auto batch = rows / kTimeAttnN;
    auto d_even = pair * 2;
    auto out_base =
        ((batch * kTimeAttnN + token) * kQkvFusedHeads + h) * kTimeAttnD;

    __nv_bfloat16* out_ptr = q;
    if (part == 1) {
        out_ptr = k_out;
    } else if (part == 2) {
        out_ptr = v;
    }

    if (part == 2) {
        ct::store_masked(out_ptr + out_base + d_even,
                         ct::element_cast<__nv_bfloat16>(even),
                         rows < M);
        ct::store_masked(out_ptr + out_base + d_even + 1,
                         ct::element_cast<__nv_bfloat16>(odd),
                         rows < M);
        return;
    }

    auto c = ct::element_cast<float>(ct::load(cos_f + token * TPairs + pair));
    auto s = ct::element_cast<float>(ct::load(sin_f + token * TPairs + pair));
    c = ct::select(full_bf16, bf16_round(c), c);
    s = ct::select(full_bf16, bf16_round(s), s);
    auto rot_even = even * c - odd * s;
    auto rot_odd = even * s + odd * c;
    rot_even = ct::select(full_bf16, bf16_round(rot_even), rot_even);
    rot_odd = ct::select(full_bf16, bf16_round(rot_odd), rot_odd);

    ct::store_masked(out_ptr + out_base + d_even,
                     ct::element_cast<__nv_bfloat16>(rot_even),
                     rows < M);
    ct::store_masked(out_ptr + out_base + d_even + 1,
                     ct::element_cast<__nv_bfloat16>(rot_odd),
                     rows < M);
}

template <int TM, int TN, int TK, int M, int N, int K, bool AddBias, bool ApplyGelu, int GeluMode, bool FullBF16>
__tile_global__ void linear_cutile_static_masked_m_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_nt,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ c) {
    static_assert(!ApplyGelu || AddBias);
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    if constexpr (AddBias) {
        bias = ct::assume_aligned(bias, 16_ic);
    }
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_nt, ct::shape<K, N>{}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    bool full_m_tile = tile_m < M / TM;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        auto b_tile = b_view.load(kk, tile_n);
        if (full_m_tile) {
            acc = ct::mma(a_view.load(tile_m, kk), b_tile, acc);
        } else {
            acc = ct::mma(a_view.load_masked(tile_m, kk), b_tile, acc);
        }
    }

    auto value = acc;
    if constexpr (AddBias) {
        value = bf16_round(value);
        I64OutTile local = ct::iota<I64OutTile>();
        auto cols = static_cast<long long>(tile_n) * TN + (local % TN);
        auto bias_values = ct::element_cast<float>(ct::load(bias + cols));
        value = value + bias_values;
        value = bf16_round_if<FullBF16>(value);
    }
    if constexpr (ApplyGelu) {
        value = gelu_selected<GeluMode, FullBF16>(value);
    }

    if (full_m_tile) {
        c_view.store(ct::element_cast<__nv_bfloat16>(value), tile_m, tile_n);
    } else {
        c_view.store_masked(ct::element_cast<__nv_bfloat16>(value), tile_m, tile_n);
    }
}

template <int TM,
          int TN,
          int TK,
          int MPad,
          int MActual,
          int N,
          int K,
          bool AddBias,
          bool ApplyGelu,
          int GeluMode,
          bool FullBF16,
          bool RoundBeforeBias = true>
__tile_global__ void linear_cutile_static_padded_m_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_nt,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ c) {
    static_assert(!ApplyGelu || AddBias);
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using I64ATile = ct::tile<long long, ct::shape<TM, TK>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    if constexpr (AddBias) {
        bias = ct::assume_aligned(bias, 16_ic);
    }
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<MPad, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_nt, ct::shape<K, N>{}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<MPad, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    bool full_m_tile = tile_m < MActual / TM;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        auto b_tile = b_view.load(kk, tile_n);
        if (full_m_tile) {
            acc = ct::mma(a_view.load(tile_m, kk), b_tile, acc);
        } else {
            I64ATile local = ct::iota<I64ATile>();
            auto rows = static_cast<long long>(tile_m) * TM + local / TK;
            auto k_cols = static_cast<long long>(kk) * TK + local % TK;
            auto valid = rows < MActual;
            auto a_tile = ct::load_masked(a + rows * K + k_cols, valid);
            acc = ct::mma(a_tile, b_tile, acc);
        }
    }

    auto value = acc;
    if constexpr (AddBias) {
        if constexpr (RoundBeforeBias) {
            value = bf16_round(value);
        }
        I64OutTile local = ct::iota<I64OutTile>();
        auto cols = static_cast<long long>(tile_n) * TN + (local % TN);
        auto bias_values = ct::element_cast<float>(ct::load(bias + cols));
        value = value + bias_values;
        value = bf16_round_if<FullBF16>(value);
    }
    if constexpr (ApplyGelu) {
        value = gelu_selected<GeluMode, FullBF16>(value);
    }

    auto out = ct::element_cast<__nv_bfloat16>(value);
    if (full_m_tile) {
        c_view.store(out, tile_m, tile_n);
    } else {
        I64OutTile local = ct::iota<I64OutTile>();
        auto rows = static_cast<long long>(tile_m) * TM + local / TN;
        auto cols = static_cast<long long>(tile_n) * TN + (local % TN);
        auto valid = rows < MActual;
        ct::store_masked(c + rows * N + cols, out, valid);
    }
}

template <int TM, int TN, int TK, int MActual, int N>
__tile_global__ void linear_cutile_small_n256_dynamic_k_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_nt,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ c,
    int k,
    bool full_bf16) {
    static_assert(N == 256);
    static_assert(N % TN == 0);
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using I64ATile = ct::tile<long long, ct::shape<TM, TK>>;
    using I64BTile = ct::tile<long long, ct::shape<TK, TN>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    bias = ct::assume_aligned(bias, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    I64ATile a_local = ct::iota<I64ATile>();
    auto a_rows = static_cast<long long>(tile_m) * TM + a_local / TK;
    I64BTile b_local = ct::iota<I64BTile>();
    auto b_cols = static_cast<long long>(tile_n) * TN + b_local % TN;

    auto acc = ct::full<AccTile>(0.0f);
    int k_tiles = (k + TK - 1) / TK;
    for (int kk = 0; kk < k_tiles; ++kk) {
        auto a_cols = static_cast<long long>(kk) * TK + a_local % TK;
        auto b_rows = static_cast<long long>(kk) * TK + b_local / TN;
        auto a_valid = (a_rows < MActual) & (a_cols < k);
        auto b_valid = b_rows < k;
        auto a_tile = ct::load_masked(a + a_rows * static_cast<long long>(k) + a_cols,
                                      a_valid);
        auto b_tile = ct::load_masked(b_nt + b_cols * static_cast<long long>(k) + b_rows,
                                      b_valid);
        acc = ct::mma(a_tile, b_tile, acc);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(tile_m) * TM + out_local / TN;
    auto out_cols = static_cast<long long>(tile_n) * TN + out_local % TN;
    auto value = bf16_round(acc);
    value = value + ct::element_cast<float>(ct::load(bias + out_cols));
    value = ct::select(full_bf16, bf16_round(value), value);
    ct::store_masked(c + out_rows * N + out_cols,
                     ct::element_cast<__nv_bfloat16>(value),
                     out_rows < MActual);
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          bool AddBias,
          bool ApplySigmoid>
__tile_global__ void linear_cutile_static_masked_mn_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_nt,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ c,
    bool full_bf16) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    if constexpr (AddBias) {
        bias = ct::assume_aligned(bias, 16_ic);
    }
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_nt, ct::shape<K, N>{}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, N>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    bool full_m_tile = tile_m < M / TM;
    bool full_n_tile = false;
    if constexpr (N >= TN) {
        full_n_tile = tile_n < N / TN;
    }
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        auto a_tile = full_m_tile ? a_view.load(tile_m, kk)
                                  : a_view.load_masked(tile_m, kk);
        auto b_tile = full_n_tile ? b_view.load(kk, tile_n)
                                  : b_view.load_masked(kk, tile_n);
        acc = ct::mma(a_tile, b_tile, acc);
    }

    auto value = acc;
    I64OutTile local = ct::iota<I64OutTile>();
    auto cols = static_cast<long long>(tile_n) * TN + (local % TN);
    if constexpr (AddBias) {
        value = bf16_round(value);
        auto bias_values = ct::element_cast<float>(
            ct::load_masked(bias + cols, cols < N));
        value = value + bias_values;
        value = ct::select(full_bf16, bf16_round(value), value);
    }
    if constexpr (ApplySigmoid) {
        value = 1.0f / (1.0f + ct::exp(-value));
        value = ct::select(full_bf16, bf16_round(value), value);
    }

    auto out = ct::element_cast<__nv_bfloat16>(value);
    if (full_m_tile && full_n_tile) {
        c_view.store(out, tile_m, tile_n);
    } else {
        c_view.store_masked(out, tile_m, tile_n);
    }
}

template <int TM, int TNOut, int TK, int M, int NOut, int K>
__tile_global__ void linear_glu_last_dim_static_masked_mn_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_nt,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ c,
    bool full_bf16) {
    static_assert(NOut <= TNOut);
    using AccTile = ct::tile<float, ct::shape<TM, TNOut>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, TNOut>>;

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    bias = ct::assume_aligned(bias, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_first_view = ct::partition_view{
        ct::tensor_span{b_nt, ct::shape<K, NOut>{}, ct::layout_left{}},
        ct::shape<TK, TNOut>{}
    };
    auto b_gate_view = ct::partition_view{
        ct::tensor_span{b_nt + static_cast<std::size_t>(NOut) * K,
                        ct::shape<K, NOut>{},
                        ct::layout_left{}},
        ct::shape<TK, TNOut>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<M, NOut>{}},
        ct::shape<TM, TNOut>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    bool full_m_tile = tile_m < M / TM;
    bool full_n_tile = false;
    if constexpr (NOut >= TNOut) {
        full_n_tile = tile_n < NOut / TNOut;
    }

    auto first = ct::full<AccTile>(0.0f);
    auto gate = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        auto a_tile = full_m_tile ? a_view.load(tile_m, kk)
                                  : a_view.load_masked(tile_m, kk);
        auto first_w = full_n_tile ? b_first_view.load(kk, tile_n)
                                   : b_first_view.load_masked(kk, tile_n);
        auto gate_w = full_n_tile ? b_gate_view.load(kk, tile_n)
                                  : b_gate_view.load_masked(kk, tile_n);
        first = ct::mma(a_tile, first_w, first);
        gate = ct::mma(a_tile, gate_w, gate);
    }

    I64OutTile local = ct::iota<I64OutTile>();
    auto cols = static_cast<long long>(tile_n) * TNOut + (local % TNOut);
    first = bf16_round(first);
    gate = bf16_round(gate);
    auto first_bias = ct::element_cast<float>(ct::load_masked(bias + cols, cols < NOut));
    auto gate_bias = ct::element_cast<float>(ct::load_masked(bias + NOut + cols, cols < NOut));
    first = first + first_bias;
    gate = gate + gate_bias;
    first = ct::select(full_bf16, bf16_round(first), first);
    gate = ct::select(full_bf16, bf16_round(gate), gate);
    gate = 1.0f / (1.0f + ct::exp(-gate));
    gate = ct::select(full_bf16, bf16_round(gate), gate);
    auto value = first * gate;
    value = ct::select(full_bf16, bf16_round(value), value);
    auto out = ct::element_cast<__nv_bfloat16>(value);

    if (full_m_tile && full_n_tile) {
        c_view.store(out, tile_m, tile_n);
    } else {
        c_view.store_masked(out, tile_m, tile_n);
    }
}

template <int TM, int TPairs, int TK, int M, int K, typename TrigT = float>
__tile_global__ void qkv_freq60_rotary_pad64_cutile_static_full_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    bool full_bf16) {
    static_assert(M % TM == 0);
    static_assert(K % TK == 0);
    static_assert(TPairs == kFreqAttnD / 2);
    using PairTile = ct::tile<float, ct::shape<TM, TPairs>>;
    using I64PairTile = ct::tile<long long, ct::shape<TM, TPairs>>;
    using WShape = ct::shape<K, TPairs>;
    using WStrides = ct::shape<1, 2 * K>;
    using WLayout = ct::layout_strided<WStrides>;
    using WMapping = typename WLayout::template mapping<WShape>;

    x = ct::assume_aligned(x, 16_ic);
    weight = ct::assume_aligned(weight, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    auto [tile_m, group_raw, tile_z] = ct::bid();
    (void)tile_z;
    int group = static_cast<int>(group_raw);
    int part = group / kQkvFusedHeads;
    int h = group - part * kQkvFusedHeads;
    int feature_base = part * kQkvFusedHeads * kFreqAttnD + h * kFreqAttnD;

    auto x_view = ct::partition_view{
        ct::tensor_span{x, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto w_even_view = ct::partition_view{
        ct::tensor_span{weight + static_cast<std::size_t>(feature_base) * K,
                        WMapping{WShape{}, WStrides{}}},
        ct::shape<TK, TPairs>{}
    };
    auto w_odd_view = ct::partition_view{
        ct::tensor_span{weight + static_cast<std::size_t>(feature_base + 1) * K,
                        WMapping{WShape{}, WStrides{}}},
        ct::shape<TK, TPairs>{}
    };

    auto even = ct::full<PairTile>(0.0f);
    auto odd = ct::full<PairTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        auto x_tile = x_view.load(tile_m, kk);
        even = ct::mma(x_tile, w_even_view.load(kk, 0), even);
        odd = ct::mma(x_tile, w_odd_view.load(kk, 0), odd);
    }

    even = bf16_round(even);
    odd = bf16_round(odd);

    I64PairTile local = ct::iota<I64PairTile>();
    auto rows = static_cast<long long>(tile_m) * TM + local / TPairs;
    auto pair = local % TPairs;
    auto b = rows / kFreqAttnN;
    auto n = rows % kFreqAttnN;
    auto d_even = pair * 2;
    auto out_base = ((b * kQkvFusedHeads + h) * kFreqAttnPadN + n) * kFreqAttnD;

    __nv_bfloat16* out_ptr = q;
    if (part == 1) {
        out_ptr = k;
    } else if (part == 2) {
        out_ptr = v;
    }

    if (part == 2) {
        ct::store_masked(out_ptr + out_base + d_even,
                         ct::element_cast<__nv_bfloat16>(even),
                         rows < M);
        ct::store_masked(out_ptr + out_base + d_even + 1,
                         ct::element_cast<__nv_bfloat16>(odd),
                         rows < M);
        return;
    }

    auto c = ct::element_cast<float>(ct::load(cos_f + n * TPairs + pair));
    auto s = ct::element_cast<float>(ct::load(sin_f + n * TPairs + pair));
    c = ct::select(full_bf16, bf16_round(c), c);
    s = ct::select(full_bf16, bf16_round(s), s);
    auto rot_even = even * c - odd * s;
    auto rot_odd = even * s + odd * c;
    rot_even = ct::select(full_bf16, bf16_round(rot_even), rot_even);
    rot_odd = ct::select(full_bf16, bf16_round(rot_odd), rot_odd);

    ct::store_masked(out_ptr + out_base + d_even,
                     ct::element_cast<__nv_bfloat16>(rot_even),
                     rows < M);
    ct::store_masked(out_ptr + out_base + d_even + 1,
                     ct::element_cast<__nv_bfloat16>(rot_odd),
                     rows < M);
}

template <int K, typename TrigT = float>
__tile_global__ void qkv_freq60_rotary_pad64_cutile_tail_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    int total_batch,
    int row_start,
    bool full_bf16) {
    x = ct::assume_aligned(x, 16_ic);
    weight = ct::assume_aligned(weight, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    int group = static_cast<int>(ct::bid().y);
    int part = group / kQkvFusedHeads;
    int h = group - part * kQkvFusedHeads;
    int feature_base = part * kQkvFusedHeads * kFreqAttnD + h * kFreqAttnD;

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    long long total = (long long)(total_batch - row_start) * (kFreqAttnD / 2);
    auto in_bounds = idx < total;
    auto pair = idx % (kFreqAttnD / 2);
    auto rows = row_start + idx / (kFreqAttnD / 2);
    auto b = rows / kFreqAttnN;
    auto n = rows % kFreqAttnN;
    auto d_even = pair * 2;

    F32Tile even = ct::element_cast<float>(idx * 0LL);
    F32Tile odd = ct::element_cast<float>(idx * 0LL);
#pragma unroll
    for (int kk = 0; kk < K; ++kk) {
        auto xv = ct::element_cast<float>(
            ct::load_masked(x + rows * K + kk, in_bounds));
        auto w_even = ct::element_cast<float>(
            ct::load_masked(weight + (feature_base + d_even) * K + kk, in_bounds));
        auto w_odd = ct::element_cast<float>(
            ct::load_masked(weight + (feature_base + d_even + 1) * K + kk, in_bounds));
        even = even + xv * w_even;
        odd = odd + xv * w_odd;
    }

    even = bf16_round(even);
    odd = bf16_round(odd);

    __nv_bfloat16* out_ptr = q;
    if (part == 1) {
        out_ptr = k;
    } else if (part == 2) {
        out_ptr = v;
    }
    auto out_base = ((b * kQkvFusedHeads + h) * kFreqAttnPadN + n) * kFreqAttnD;

    if (part == 2) {
        ct::store_masked(out_ptr + out_base + d_even,
                         ct::element_cast<__nv_bfloat16>(even),
                         in_bounds);
        ct::store_masked(out_ptr + out_base + d_even + 1,
                         ct::element_cast<__nv_bfloat16>(odd),
                         in_bounds);
        return;
    }

    auto c = ct::element_cast<float>(
        ct::load_masked(cos_f + n * (kFreqAttnD / 2) + pair, in_bounds));
    auto s = ct::element_cast<float>(
        ct::load_masked(sin_f + n * (kFreqAttnD / 2) + pair, in_bounds));
    c = ct::select(full_bf16, bf16_round(c), c);
    s = ct::select(full_bf16, bf16_round(s), s);
    auto rot_even = even * c - odd * s;
    auto rot_odd = even * s + odd * c;
    rot_even = ct::select(full_bf16, bf16_round(rot_even), rot_even);
    rot_odd = ct::select(full_bf16, bf16_round(rot_odd), rot_odd);
    ct::store_masked(out_ptr + out_base + d_even,
                     ct::element_cast<__nv_bfloat16>(rot_even),
                     in_bounds);
    ct::store_masked(out_ptr + out_base + d_even + 1,
                     ct::element_cast<__nv_bfloat16>(rot_odd),
                     in_bounds);
}

template <int TM, int TN, int TK, int M, int N, int K, typename TrigT = float>
__tile_global__ void qkv_freq60_rotary_pad64_bkn_cutile_static_full_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight_bkn,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    bool full_bf16) {
    static_assert(M % TM == 0);
    static_assert(N == 3 * kQkvFusedHeads * kFreqAttnD);
    static_assert((N / 3) % TN == 0);
    static_assert(TN % kFreqAttnD == 0);
    static_assert(K % TK == 0);
    constexpr int kHalfDim = kFreqAttnD / 2;
    constexpr int kHeadsPerTile = TN / kFreqAttnD;
    static_assert(kHeadsPerTile > 0);
    static_assert(kQkvFusedHeads % kHeadsPerTile == 0);
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using Acc4Tile = ct::tile<float, ct::shape<TM, kHeadsPerTile, kHalfDim, 2>>;
    using PairTile = ct::tile<float, ct::shape<TM, kHeadsPerTile, kHalfDim, 1>>;
    using I64PairTile = ct::tile<long long, ct::shape<TM, kHeadsPerTile, kHalfDim, 1>>;

    x = ct::assume_aligned(x, 16_ic);
    weight_bkn = ct::assume_aligned(weight_bkn, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    auto x_view = ct::partition_view{
        ct::tensor_span{x, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };
    auto w_view = ct::partition_view{
        ct::tensor_span{weight_bkn, ct::shape<K, N>{}},
        ct::shape<TK, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        acc = ct::mma(x_view.load(tile_m, kk), w_view.load(kk, tile_n), acc);
    }

    int tile_n_i = static_cast<int>(tile_n);
    int part = tile_n_i / (kQkvFusedHeads / kHeadsPerTile);
    int head_base = (tile_n_i - part * (kQkvFusedHeads / kHeadsPerTile)) *
                    kHeadsPerTile;

    Acc4Tile acc4 = ct::reshape(acc, ct::shape<TM, kHeadsPerTile, kHalfDim, 2>{});
    PairTile even = ct::extract(acc4, ct::shape<TM, kHeadsPerTile, kHalfDim, 1>{},
                                0, 0, 0, 0);
    PairTile odd = ct::extract(acc4, ct::shape<TM, kHeadsPerTile, kHalfDim, 1>{},
                               0, 0, 0, 1);
    even = bf16_round(even);
    odd = bf16_round(odd);

    I64PairTile local = ct::iota<I64PairTile>();
    auto rows = static_cast<long long>(tile_m) * TM +
                local / (kHeadsPerTile * kHalfDim);
    auto local_head = (local / kHalfDim) % kHeadsPerTile;
    auto pair = local % kHalfDim;
    auto b = rows / kFreqAttnN;
    auto n = rows - b * kFreqAttnN;
    auto h = local_head + head_base;
    auto d_even = pair * 2;
    auto out_base = ((b * kQkvFusedHeads + h) * kFreqAttnPadN + n) * kFreqAttnD;

    __nv_bfloat16* out_ptr = q;
    if (part == 1) {
        out_ptr = k;
    } else if (part == 2) {
        out_ptr = v;
    }

    if (part == 2) {
        ct::store_masked(out_ptr + out_base + d_even,
                         ct::element_cast<__nv_bfloat16>(even),
                         rows < M);
        ct::store_masked(out_ptr + out_base + d_even + 1,
                         ct::element_cast<__nv_bfloat16>(odd),
                         rows < M);
        return;
    }

    auto c = ct::element_cast<float>(ct::load(cos_f + n * kHalfDim + pair));
    auto s = ct::element_cast<float>(ct::load(sin_f + n * kHalfDim + pair));
    c = ct::select(full_bf16, bf16_round(c), c);
    s = ct::select(full_bf16, bf16_round(s), s);
    auto rot_even = even * c - odd * s;
    auto rot_odd = even * s + odd * c;
    rot_even = ct::select(full_bf16, bf16_round(rot_even), rot_even);
    rot_odd = ct::select(full_bf16, bf16_round(rot_odd), rot_odd);

    ct::store_masked(out_ptr + out_base + d_even,
                     ct::element_cast<__nv_bfloat16>(rot_even),
                     rows < M);
    ct::store_masked(out_ptr + out_base + d_even + 1,
                     ct::element_cast<__nv_bfloat16>(rot_odd),
                     rows < M);
}

template <int K, int N, typename TrigT = float>
__tile_global__ void qkv_freq60_rotary_pad64_bkn_cutile_tail_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight_bkn,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    int total_batch,
    int row_start,
    bool full_bf16) {
    static_assert(N == 3 * kQkvFusedHeads * kFreqAttnD);
    constexpr int kHalfDim = kFreqAttnD / 2;

    x = ct::assume_aligned(x, 16_ic);
    weight_bkn = ct::assume_aligned(weight_bkn, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    int group = static_cast<int>(ct::bid().y);
    int part = group / kQkvFusedHeads;
    int h = group - part * kQkvFusedHeads;
    int feature_base = part * kQkvFusedHeads * kFreqAttnD + h * kFreqAttnD;

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    long long total = (long long)(total_batch - row_start) * kHalfDim;
    auto in_bounds = idx < total;
    auto pair = idx % kHalfDim;
    auto rows = row_start + idx / kHalfDim;
    auto b = rows / kFreqAttnN;
    auto n = rows - b * kFreqAttnN;
    auto d_even = pair * 2;

    F32Tile even = ct::element_cast<float>(idx * 0LL);
    F32Tile odd = ct::element_cast<float>(idx * 0LL);
#pragma unroll
    for (int kk = 0; kk < K; ++kk) {
        auto xv = ct::element_cast<float>(
            ct::load_masked(x + rows * K + kk, in_bounds));
        auto w_even = ct::element_cast<float>(
            ct::load_masked(weight_bkn + kk * N + feature_base + d_even, in_bounds));
        auto w_odd = ct::element_cast<float>(
            ct::load_masked(weight_bkn + kk * N + feature_base + d_even + 1, in_bounds));
        even = even + xv * w_even;
        odd = odd + xv * w_odd;
    }

    even = bf16_round(even);
    odd = bf16_round(odd);

    __nv_bfloat16* out_ptr = q;
    if (part == 1) {
        out_ptr = k;
    } else if (part == 2) {
        out_ptr = v;
    }
    auto out_base = ((b * kQkvFusedHeads + h) * kFreqAttnPadN + n) * kFreqAttnD;

    if (part == 2) {
        ct::store_masked(out_ptr + out_base + d_even,
                         ct::element_cast<__nv_bfloat16>(even),
                         in_bounds);
        ct::store_masked(out_ptr + out_base + d_even + 1,
                         ct::element_cast<__nv_bfloat16>(odd),
                         in_bounds);
        return;
    }

    auto c = ct::element_cast<float>(
        ct::load_masked(cos_f + n * kHalfDim + pair, in_bounds));
    auto s = ct::element_cast<float>(
        ct::load_masked(sin_f + n * kHalfDim + pair, in_bounds));
    c = ct::select(full_bf16, bf16_round(c), c);
    s = ct::select(full_bf16, bf16_round(s), s);
    auto rot_even = even * c - odd * s;
    auto rot_odd = even * s + odd * c;
    rot_even = ct::select(full_bf16, bf16_round(rot_even), rot_even);
    rot_odd = ct::select(full_bf16, bf16_round(rot_odd), rot_odd);
    ct::store_masked(out_ptr + out_base + d_even,
                     ct::element_cast<__nv_bfloat16>(rot_even),
                     in_bounds);
    ct::store_masked(out_ptr + out_base + d_even + 1,
                     ct::element_cast<__nv_bfloat16>(rot_odd),
                     in_bounds);
}

__tile_global__ void qkv_freq60_pad_rows_zero_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    long long total,
    int batches) {
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto d = idx % kFreqAttnD;
    auto pad_row = (idx / kFreqAttnD) % (kFreqAttnPadN - kFreqAttnN);
    auto h = (idx / ((long long)kFreqAttnD * (kFreqAttnPadN - kFreqAttnN))) %
             kQkvFusedHeads;
    auto b = idx / ((long long)kFreqAttnD * (kFreqAttnPadN - kFreqAttnN) *
                    kQkvFusedHeads);
    auto n = pad_row + kFreqAttnN;
    auto out_idx = ((b * kQkvFusedHeads + h) * kFreqAttnPadN + n) * kFreqAttnD + d;
    auto zero = ct::element_cast<__nv_bfloat16>(ct::element_cast<float>(idx * 0LL));
    auto valid = in_bounds && (b < batches);
    ct::store_masked(q + out_idx, zero, valid);
    ct::store_masked(k + out_idx, zero, valid);
    ct::store_masked(v + out_idx, zero, valid);
}

bool linear_cutile_static_qkv_candidate(int64_t total_batch,
                                        int64_t out_features,
                                        int64_t in_features) {
    return linear_cutile_static_bf16_output_enabled() &&
           total_batch == kLinearCutileExpectedM &&
           out_features == 1536 &&
           in_features == 256;
}

bool linear_cutile_static_attn_out_candidate(int64_t total_batch,
                                             int64_t out_features,
                                             int64_t in_features) {
    return linear_cutile_static_bf16_output_enabled() &&
           total_batch == kLinearCutileExpectedM &&
           out_features == 256 &&
           in_features == 512;
}

bool linear_cutile_static_ffn2_candidate(int64_t total_batch,
                                         int64_t out_features,
                                         int64_t in_features) {
    return linear_cutile_static_bf16_output_enabled() &&
           total_batch == kLinearCutileExpectedM &&
           out_features == 256 &&
           in_features == 1024;
}

bool linear_cutile_static_ffn1_candidate(int64_t total_batch,
                                         int64_t out_features,
                                         int64_t in_features) {
    return linear_cutile_static_bf16_output_enabled() &&
           total_batch == kLinearCutileExpectedM &&
           out_features == 1024 &&
           in_features == 256;
}

bool linear_cutile_static_small_bias_candidate(int64_t total_batch,
                                               int64_t out_features,
                                               int64_t in_features) {
    if (!linear_cutile_static_small_bf16_output_enabled() ||
        total_batch != kLinearCutileSmallExpectedM) {
        return false;
    }
    if (out_features == 1024 && (in_features == 256 || in_features == 1024)) {
        return true;
    }
    if (out_features == 256 &&
        in_features > 0 &&
        in_features <= 1024) {
        return true;
    }
    if (out_features == 128 && in_features == 1024) {
        return true;
    }
    if (out_features == 48 && in_features == 1024) {
        return true;
    }
    if (in_features == 1024 &&
        (out_features == 56 || out_features == 72 || out_features == 80 ||
         out_features == 88 || out_features == 104 || out_features == 120 ||
         out_features == 136 || out_features == 152 || out_features == 160 ||
         out_features == 176 || out_features == 192 || out_features == 208 ||
         out_features == 224 || out_features == 232 || out_features == 248 ||
         out_features == 264 || out_features == 288 || out_features == 312 ||
         out_features == 328 || out_features == 352 || out_features == 376 ||
         out_features == 400 || out_features == 432 || out_features == 456 ||
         out_features == 488 || out_features == 528 || out_features == 568 ||
         out_features == 608 || out_features == 640 || out_features == 688 ||
         out_features == 744 || out_features == 792 || out_features == 840 ||
         out_features == 904 || out_features == 976 || out_features == 1040)) {
        return true;
    }
    return false;
}

template <int GeluMode,
          bool FullBF16,
          int StaticM,
          int TM,
          int TN,
          int TK,
          int N,
          int K,
          bool AddBias,
          bool ApplyGelu,
          bool AddResidual = false>
void launch_linear_cutile_static_bf16_tiled_selected(const Tensor& x,
                                                     const Tensor& weight,
                                                     const Tensor* bias,
                                                     Tensor& out,
                                                     int total_batch,
                                                     const Tensor* residual = nullptr) {
    dim3 full_grid(StaticM / TM,
                   N / TN,
                   1);
    linear_cutile_static_full_bf16_kernel<TM,
                                          TN,
                                          TK,
                                          StaticM,
                                          N,
                                          K,
                                          AddBias,
                                          ApplyGelu,
                                          GeluMode,
                                          FullBF16,
                                          AddResidual><<<full_grid, 1>>>(
        x.data_bf16(),
        weight.data_bf16(),
        AddBias ? bias->data_bf16() : nullptr,
        out.data_bf16(),
        AddResidual ? residual->data_bf16() : nullptr);

    int tail_rows = total_batch - StaticM;
    if (tail_rows > 0) {
        int tail_total = tail_rows * N;
        linear_cutile_tail_bf16_kernel<N,
                                       K,
                                       AddBias,
                                       ApplyGelu,
                                       GeluMode,
                                       FullBF16,
                                       AddResidual>
            <<<(int)ceildiv(tail_total, kTile), 1>>>(
                x.data_bf16(),
                weight.data_bf16(),
                AddBias ? bias->data_bf16() : nullptr,
                out.data_bf16(),
                AddResidual ? residual->data_bf16() : nullptr,
                total_batch,
                StaticM);
    }
}

template <int GeluMode,
          bool FullBF16,
          int StaticM,
          int TM,
          int TN,
          int TK,
          int N,
          int K,
          bool AddBias,
          bool ApplyGelu,
          int LoadLatency = 0>
void launch_linear_cutile_static_bkn_bf16_tiled_selected(const Tensor& x,
                                                         const Tensor& weight,
                                                         const Tensor& weight_bkn,
                                                         const Tensor* bias,
                                                         Tensor& out,
                                                         int total_batch) {
    dim3 full_grid(StaticM / TM,
                   N / TN,
                   1);
    linear_cutile_static_full_bkn_bf16_kernel<TM,
                                              TN,
                                              TK,
                                              StaticM,
                                              N,
                                              K,
                                              AddBias,
                                              ApplyGelu,
                                              GeluMode,
                                              FullBF16,
                                              LoadLatency><<<full_grid, 1>>>(
        x.data_bf16(),
        weight_bkn.data_bf16(),
        AddBias ? bias->data_bf16() : nullptr,
        out.data_bf16());

    int tail_rows = total_batch - StaticM;
    if (tail_rows > 0) {
        if constexpr (StaticM == kLinearCutileStaticM64 &&
                      TM == 64 &&
                      TN == 128 &&
                      TK == 16 &&
                      N == 1536 &&
                      K == 256 &&
                      !AddBias &&
                      !ApplyGelu) {
            if (total_batch == kLinearCutileExpectedM) {
                dim3 tail_grid((unsigned int)ceildiv(tail_rows, TM),
                               N / TN,
                               1);
                linear_cutile_static_bkn_masked_tail_bf16_kernel<TM,
                                                                 TN,
                                                                 TK,
                                                                 kLinearCutileExpectedM,
                                                                 N,
                                                                 K,
                                                                 StaticM,
                                                                 LoadLatency>
                    <<<tail_grid, 1>>>(
                        x.data_bf16(),
                        weight_bkn.data_bf16(),
                        out.data_bf16());
                return;
            }
        }
        int tail_total = tail_rows * N;
        linear_cutile_tail_bf16_kernel<N, K, AddBias, ApplyGelu, GeluMode, FullBF16>
            <<<(int)ceildiv(tail_total, kTile), 1>>>(
                x.data_bf16(),
                weight.data_bf16(),
                AddBias ? bias->data_bf16() : nullptr,
                out.data_bf16(),
                nullptr,
                total_batch,
                StaticM);
    }
}

void launch_linear_cutile_static_attn_out_residual_bf16(const Tensor& x,
                                                        const Tensor& weight,
                                                        const Tensor& residual,
                                                        Tensor& out,
                                                        int total_batch) {
    if (attention_out_residual_tile_n() == 64) {
        launch_linear_cutile_static_bf16_tiled_selected<kGeluErf,
                                                        false,
                                                        kLinearCutileStaticM,
                                                        32,
                                                        64,
                                                        kLinearCutileTileK,
                                                        256,
                                                        512,
                                                        false,
                                                        false,
                                                        true>(
            x, weight, nullptr, out, total_batch, &residual);
    } else {
        launch_linear_cutile_static_bf16_tiled_selected<kGeluErf,
                                                        false,
                                                        kLinearCutileStaticM,
                                                        32,
                                                        128,
                                                        kLinearCutileTileK,
                                                        256,
                                                        512,
                                                        false,
                                                        false,
                                                        true>(
            x, weight, nullptr, out, total_batch, &residual);
    }
}

template <bool FullBF16,
          int StaticM,
          int TM,
          int TN,
          int TK,
          int N,
          int K,
          bool AddBias,
          bool ApplyGelu>
void launch_linear_cutile_static_bf16_tiled_full(const Tensor& x,
                                                 const Tensor& weight,
                                                 const Tensor* bias,
                                                 Tensor& out,
                                                 int total_batch) {
    if constexpr (ApplyGelu) {
        if (linear_tanh_gelu_enabled()) {
            launch_linear_cutile_static_bf16_tiled_selected<kGeluTanh,
                                                            FullBF16,
                                                            StaticM,
                                                            TM,
                                                            TN,
                                                            TK,
                                                            N,
                                                            K,
                                                            AddBias,
                                                            ApplyGelu>(
                x, weight, bias, out, total_batch);
            return;
        }
        if (linear_quick_gelu_enabled()) {
            launch_linear_cutile_static_bf16_tiled_selected<kGeluQuick,
                                                            FullBF16,
                                                            StaticM,
                                                            TM,
                                                            TN,
                                                            TK,
                                                            N,
                                                            K,
                                                            AddBias,
                                                            ApplyGelu>(
                x, weight, bias, out, total_batch);
            return;
        }
        if (linear_hard_gelu_enabled()) {
            launch_linear_cutile_static_bf16_tiled_selected<kGeluHard,
                                                            FullBF16,
                                                            StaticM,
                                                            TM,
                                                            TN,
                                                            TK,
                                                            N,
                                                            K,
                                                            AddBias,
                                                            ApplyGelu>(
                x, weight, bias, out, total_batch);
            return;
        }
    }
    launch_linear_cutile_static_bf16_tiled_selected<kGeluErf,
                                                    FullBF16,
                                                    StaticM,
                                                    TM,
                                                    TN,
                                                    TK,
                                                    N,
                                                    K,
                                                    AddBias,
                                                    ApplyGelu>(
        x, weight, bias, out, total_batch);
}

template <bool FullBF16,
          int StaticM,
          int TM,
          int TN,
          int TK,
          int N,
          int K,
          bool AddBias,
          bool ApplyGelu,
          int LoadLatency = 0>
void launch_linear_cutile_static_bkn_bf16_tiled_full(const Tensor& x,
                                                     const Tensor& weight,
                                                     const Tensor& weight_bkn,
                                                     const Tensor* bias,
                                                     Tensor& out,
                                                     int total_batch) {
    if constexpr (ApplyGelu) {
        if (linear_tanh_gelu_enabled()) {
            launch_linear_cutile_static_bkn_bf16_tiled_selected<kGeluTanh,
                                                                FullBF16,
                                                                StaticM,
                                                                TM,
                                                                TN,
                                                                TK,
                                                                N,
                                                                K,
                                                                AddBias,
                                                                ApplyGelu,
                                                                LoadLatency>(
                x, weight, weight_bkn, bias, out, total_batch);
            return;
        }
        if (linear_quick_gelu_enabled()) {
            launch_linear_cutile_static_bkn_bf16_tiled_selected<kGeluQuick,
                                                                FullBF16,
                                                                StaticM,
                                                                TM,
                                                                TN,
                                                                TK,
                                                                N,
                                                                K,
                                                                AddBias,
                                                                ApplyGelu,
                                                                LoadLatency>(
                x, weight, weight_bkn, bias, out, total_batch);
            return;
        }
        if (linear_hard_gelu_enabled()) {
            launch_linear_cutile_static_bkn_bf16_tiled_selected<kGeluHard,
                                                                FullBF16,
                                                                StaticM,
                                                                TM,
                                                                TN,
                                                                TK,
                                                                N,
                                                                K,
                                                                AddBias,
                                                                ApplyGelu,
                                                                LoadLatency>(
                x, weight, weight_bkn, bias, out, total_batch);
            return;
        }
    }
    launch_linear_cutile_static_bkn_bf16_tiled_selected<kGeluErf,
                                                        FullBF16,
                                                        StaticM,
                                                        TM,
                                                        TN,
                                                        TK,
                                                        N,
                                                        K,
                                                        AddBias,
                                                        ApplyGelu,
                                                        LoadLatency>(
        x, weight, weight_bkn, bias, out, total_batch);
}

template <int StaticM, int TM, int TN, int TK, int N, int K, bool AddBias, bool ApplyGelu>
void launch_linear_cutile_static_bf16_tiled(const Tensor& x,
                                            const Tensor& weight,
                                            const Tensor* bias,
                                            Tensor& out,
                                            int total_batch,
                                            bool full_bf16) {
    if (full_bf16) {
        launch_linear_cutile_static_bf16_tiled_full<true,
                                                    StaticM,
                                                    TM,
                                                    TN,
                                                    TK,
                                                    N,
                                                    K,
                                                    AddBias,
                                                    ApplyGelu>(
            x, weight, bias, out, total_batch);
        return;
    }
    launch_linear_cutile_static_bf16_tiled_full<false,
                                                StaticM,
                                                TM,
                                                TN,
                                                TK,
                                                N,
                                                K,
                                                AddBias,
                                                ApplyGelu>(
        x, weight, bias, out, total_batch);
}

template <int StaticM,
          int TM,
          int TN,
          int TK,
          int N,
          int K,
          bool AddBias,
          bool ApplyGelu,
          int LoadLatency = 0>
void launch_linear_cutile_static_bkn_bf16_tiled(const Tensor& x,
                                                const Tensor& weight,
                                                const Tensor& weight_bkn,
                                                const Tensor* bias,
                                                Tensor& out,
                                                int total_batch,
                                                bool full_bf16) {
    if (full_bf16) {
        launch_linear_cutile_static_bkn_bf16_tiled_full<true,
                                                        StaticM,
                                                        TM,
                                                        TN,
                                                        TK,
                                                        N,
                                                        K,
                                                        AddBias,
                                                        ApplyGelu,
                                                        LoadLatency>(
            x, weight, weight_bkn, bias, out, total_batch);
        return;
    }
    launch_linear_cutile_static_bkn_bf16_tiled_full<false,
                                                    StaticM,
                                                    TM,
                                                    TN,
                                                    TK,
                                                    N,
                                                    K,
                                                    AddBias,
                                                    ApplyGelu,
                                                    LoadLatency>(
        x, weight, weight_bkn, bias, out, total_batch);
}

template <int N, int K, bool AddBias, bool ApplyGelu>
void launch_linear_cutile_static_bf16(const Tensor& x,
                                      const Tensor& weight,
                                      const Tensor* bias,
                                      Tensor& out,
                                      int total_batch,
                                      bool full_bf16) {
    launch_linear_cutile_static_bf16_tiled<kLinearCutileStaticM,
                                           kLinearCutileTileM,
                                           kLinearCutileTileN,
                                           kLinearCutileTileK,
                                           N,
                                           K,
                                           AddBias,
                                           ApplyGelu>(
        x, weight, bias, out, total_batch, full_bf16);
}

template <int GeluMode,
          bool FullBF16,
          int StaticM,
          int TM,
          int TN,
          int TK,
          int N,
          int K,
          bool AddBias,
          bool ApplyGelu>
void launch_linear_cutile_static_masked_m_bf16_tiled_selected(const Tensor& x,
                                                              const Tensor& weight,
                                                              const Tensor* bias,
                                                              Tensor& out) {
    dim3 grid((unsigned int)ceildiv(StaticM, TM),
              N / TN,
              1);
    linear_cutile_static_masked_m_bf16_kernel<TM,
                                              TN,
                                              TK,
                                              StaticM,
                                              N,
                                              K,
                                              AddBias,
                                              ApplyGelu,
                                              GeluMode,
                                              FullBF16><<<grid, 1>>>(
        x.data_bf16(),
        weight.data_bf16(),
        AddBias ? bias->data_bf16() : nullptr,
        out.data_bf16());
}

template <bool FullBF16,
          int StaticM,
          int TM,
          int TN,
          int TK,
          int N,
          int K,
          bool AddBias,
          bool ApplyGelu>
void launch_linear_cutile_static_masked_m_bf16_tiled_full(const Tensor& x,
                                                          const Tensor& weight,
                                                          const Tensor* bias,
                                                          Tensor& out) {
    if constexpr (ApplyGelu) {
        if (linear_tanh_gelu_enabled()) {
            launch_linear_cutile_static_masked_m_bf16_tiled_selected<kGeluTanh,
                                                                     FullBF16,
                                                                     StaticM,
                                                                     TM,
                                                                     TN,
                                                                     TK,
                                                                     N,
                                                                     K,
                                                                     AddBias,
                                                                     ApplyGelu>(
                x, weight, bias, out);
            return;
        }
        if (linear_quick_gelu_enabled()) {
            launch_linear_cutile_static_masked_m_bf16_tiled_selected<kGeluQuick,
                                                                     FullBF16,
                                                                     StaticM,
                                                                     TM,
                                                                     TN,
                                                                     TK,
                                                                     N,
                                                                     K,
                                                                     AddBias,
                                                                     ApplyGelu>(
                x, weight, bias, out);
            return;
        }
        if (linear_hard_gelu_enabled()) {
            launch_linear_cutile_static_masked_m_bf16_tiled_selected<kGeluHard,
                                                                     FullBF16,
                                                                     StaticM,
                                                                     TM,
                                                                     TN,
                                                                     TK,
                                                                     N,
                                                                     K,
                                                                     AddBias,
                                                                     ApplyGelu>(
                x, weight, bias, out);
            return;
        }
    }
    launch_linear_cutile_static_masked_m_bf16_tiled_selected<kGeluErf,
                                                             FullBF16,
                                                             StaticM,
                                                             TM,
                                                             TN,
                                                             TK,
                                                             N,
                                                             K,
                                                             AddBias,
                                                             ApplyGelu>(
        x, weight, bias, out);
}

template <int StaticM, int TM, int TN, int TK, int N, int K, bool AddBias, bool ApplyGelu>
void launch_linear_cutile_static_masked_m_bf16_tiled(const Tensor& x,
                                                     const Tensor& weight,
                                                     const Tensor* bias,
                                                     Tensor& out,
                                                     bool full_bf16) {
    if (full_bf16) {
        launch_linear_cutile_static_masked_m_bf16_tiled_full<true,
                                                             StaticM,
                                                             TM,
                                                             TN,
                                                             TK,
                                                             N,
                                                             K,
                                                             AddBias,
                                                             ApplyGelu>(
            x, weight, bias, out);
        return;
    }
    launch_linear_cutile_static_masked_m_bf16_tiled_full<false,
                                                         StaticM,
                                                         TM,
                                                         TN,
                                                         TK,
                                                         N,
                                                         K,
                                                         AddBias,
                                                         ApplyGelu>(
        x, weight, bias, out);
}

template <int GeluMode,
          bool FullBF16,
          int PaddedM,
          int ActualM,
          int TM,
          int TN,
          int TK,
          int N,
          int K,
          bool AddBias,
          bool ApplyGelu>
void launch_linear_cutile_static_padded_m_bf16_tiled_selected(const Tensor& x,
                                                              const Tensor& weight,
                                                              const Tensor* bias,
                                                              Tensor& out) {
    dim3 grid(PaddedM / TM,
              N / TN,
              1);
    if constexpr (AddBias && !ApplyGelu && !FullBF16) {
        if (linear_padded_bias_no_round_enabled()) {
            linear_cutile_static_padded_m_bf16_kernel<TM,
                                                      TN,
                                                      TK,
                                                      PaddedM,
                                                      ActualM,
                                                      N,
                                                      K,
                                                      AddBias,
                                                      ApplyGelu,
                                                      GeluMode,
                                                      FullBF16,
                                                      false><<<grid, 1>>>(
                x.data_bf16(),
                weight.data_bf16(),
                bias->data_bf16(),
                out.data_bf16());
            return;
        }
    }
    linear_cutile_static_padded_m_bf16_kernel<TM,
                                              TN,
                                              TK,
                                              PaddedM,
                                              ActualM,
                                              N,
                                              K,
                                              AddBias,
                                              ApplyGelu,
                                              GeluMode,
                                              FullBF16,
                                              true><<<grid, 1>>>(
        x.data_bf16(),
        weight.data_bf16(),
        AddBias ? bias->data_bf16() : nullptr,
        out.data_bf16());
}

template <bool FullBF16,
          int PaddedM,
          int ActualM,
          int TM,
          int TN,
          int TK,
          int N,
          int K,
          bool AddBias,
          bool ApplyGelu>
void launch_linear_cutile_static_padded_m_bf16_tiled_full(const Tensor& x,
                                                          const Tensor& weight,
                                                          const Tensor* bias,
                                                          Tensor& out) {
    if constexpr (ApplyGelu) {
        if (linear_tanh_gelu_enabled()) {
            launch_linear_cutile_static_padded_m_bf16_tiled_selected<kGeluTanh,
                                                                     FullBF16,
                                                                     PaddedM,
                                                                     ActualM,
                                                                     TM,
                                                                     TN,
                                                                     TK,
                                                                     N,
                                                                     K,
                                                                     AddBias,
                                                                     ApplyGelu>(
                x, weight, bias, out);
            return;
        }
        if (linear_quick_gelu_enabled()) {
            launch_linear_cutile_static_padded_m_bf16_tiled_selected<kGeluQuick,
                                                                     FullBF16,
                                                                     PaddedM,
                                                                     ActualM,
                                                                     TM,
                                                                     TN,
                                                                     TK,
                                                                     N,
                                                                     K,
                                                                     AddBias,
                                                                     ApplyGelu>(
                x, weight, bias, out);
            return;
        }
        if (linear_hard_gelu_enabled()) {
            launch_linear_cutile_static_padded_m_bf16_tiled_selected<kGeluHard,
                                                                     FullBF16,
                                                                     PaddedM,
                                                                     ActualM,
                                                                     TM,
                                                                     TN,
                                                                     TK,
                                                                     N,
                                                                     K,
                                                                     AddBias,
                                                                     ApplyGelu>(
                x, weight, bias, out);
            return;
        }
    }
    launch_linear_cutile_static_padded_m_bf16_tiled_selected<kGeluErf,
                                                             FullBF16,
                                                             PaddedM,
                                                             ActualM,
                                                             TM,
                                                             TN,
                                                             TK,
                                                             N,
                                                             K,
                                                             AddBias,
                                                             ApplyGelu>(
        x, weight, bias, out);
}

template <int PaddedM,
          int ActualM,
          int TM,
          int TN,
          int TK,
          int N,
          int K,
          bool AddBias,
          bool ApplyGelu>
void launch_linear_cutile_static_padded_m_bf16_tiled(const Tensor& x,
                                                     const Tensor& weight,
                                                     const Tensor* bias,
                                                     Tensor& out,
                                                     bool full_bf16) {
    if (full_bf16) {
        launch_linear_cutile_static_padded_m_bf16_tiled_full<true,
                                                             PaddedM,
                                                             ActualM,
                                                             TM,
                                                             TN,
                                                             TK,
                                                             N,
                                                             K,
                                                             AddBias,
                                                             ApplyGelu>(
            x, weight, bias, out);
        return;
    }
    launch_linear_cutile_static_padded_m_bf16_tiled_full<false,
                                                         PaddedM,
                                                         ActualM,
                                                         TM,
                                                         TN,
                                                         TK,
                                                         N,
                                                         K,
                                                         AddBias,
                                                         ApplyGelu>(
        x, weight, bias, out);
}

template <int StaticM,
          int TM,
          int TN,
          int TK,
          int N,
          int K,
          bool AddBias,
          bool ApplySigmoid>
void launch_linear_cutile_static_masked_mn_bf16_tiled(const Tensor& x,
                                                      const Tensor& weight,
                                                      const Tensor* bias,
                                                      Tensor& out,
                                                      bool full_bf16) {
    dim3 grid((unsigned int)ceildiv(StaticM, TM),
              (unsigned int)ceildiv(N, TN),
              1);
    linear_cutile_static_masked_mn_bf16_kernel<TM,
                                               TN,
                                               TK,
                                               StaticM,
                                               N,
                                               K,
                                               AddBias,
                                               ApplySigmoid><<<grid, 1>>>(
        x.data_bf16(),
        weight.data_bf16(),
        AddBias ? bias->data_bf16() : nullptr,
        out.data_bf16(),
        full_bf16);
}

template <int StaticM, int TM, int TNOut, int TK, int NOut, int K>
void launch_linear_glu_last_dim_static_masked_mn_bf16_tiled(const Tensor& x,
                                                            const Tensor& weight,
                                                            const Tensor& bias,
                                                            Tensor& out,
                                                            bool full_bf16) {
    dim3 grid((unsigned int)ceildiv(StaticM, TM),
              (unsigned int)ceildiv(NOut, TNOut),
              1);
    linear_glu_last_dim_static_masked_mn_bf16_kernel<TM,
                                                     TNOut,
                                                     TK,
                                                     StaticM,
                                                     NOut,
                                                     K><<<grid, 1>>>(
        x.data_bf16(),
        weight.data_bf16(),
        bias.data_bf16(),
        out.data_bf16(),
        full_bf16);
}

template <int N>
void launch_linear_cutile_small_k1024_masked_n_bf16(const Tensor& x,
                                                    const Tensor& weight,
                                                    const Tensor& bias,
                                                    Tensor& out,
                                                    bool full_bf16) {
    launch_linear_cutile_static_masked_mn_bf16_tiled<kLinearCutileSmallExpectedM,
                                                     32,
                                                     64,
                                                     32,
                                                     N,
                                                     1024,
                                                     true,
                                                     false>(
        x, weight, &bias, out, full_bf16);
}

void launch_linear_cutile_small_n256_dynamic_k_bf16(const Tensor& x,
                                                    const Tensor& weight,
                                                    const Tensor& bias,
                                                    Tensor& out,
                                                    int in_features,
                                                    bool full_bf16) {
    dim3 grid((unsigned int)ceildiv(kLinearCutileSmallExpectedM, 64),
              256 / 64,
              1);
    linear_cutile_small_n256_dynamic_k_bf16_kernel<64,
                                                   64,
                                                   16,
                                                   kLinearCutileSmallExpectedM,
                                                   256>
        <<<grid, 1>>>(
            x.data_bf16(),
            weight.data_bf16(),
            bias.data_bf16(),
            out.data_bf16(),
            in_features,
            full_bf16);
}

bool try_linear_cutile_static_bf16_output(const Tensor& x,
                                          const Tensor& weight,
                                          int64_t total_batch,
                                          int64_t out_features,
                                          int64_t in_features,
                                          Tensor& out) {
    if (!linear_cutile_static_bf16_output_enabled()) return false;
    if (x.dtype() != DType::BFloat16 || weight.dtype() != DType::BFloat16) return false;
    if (total_batch != kLinearCutileExpectedM) return false;
    if (linear_cutile_static_qkv_candidate(total_batch, out_features, in_features)) {
        out = Tensor::empty({total_batch, out_features}, DType::BFloat16);
        if (linear_qkv_m32n256_enabled()) {
            launch_linear_cutile_static_bf16_tiled<kLinearCutileStaticM,
                                                   32,
                                                   256,
                                                   kLinearCutileTileK,
                                                   1536,
                                                   256,
                                                   false,
                                                   false>(
                x, weight, nullptr, out, (int)total_batch, false);
        } else if (linear_qkv_m32n128_enabled()) {
            launch_linear_cutile_static_bf16_tiled<kLinearCutileStaticM,
                                                   32,
                                                   128,
                                                   kLinearCutileTileK,
                                                   1536,
                                                   256,
                                                   false,
                                                   false>(
                x, weight, nullptr, out, (int)total_batch, false);
        } else {
            launch_linear_cutile_static_bf16<1536, 256, false, false>(
                x, weight, nullptr, out, (int)total_batch, false);
        }
    } else if (linear_cutile_static_attn_out_candidate(total_batch, out_features, in_features)) {
        out = Tensor::empty({total_batch, out_features}, DType::BFloat16);
        if (linear_attn_out_m32n128_enabled()) {
            launch_linear_cutile_static_bf16_tiled<kLinearCutileStaticM,
                                                   32,
                                                   128,
                                                   kLinearCutileTileK,
                                                   256,
                                                   512,
                                                   false,
                                                   false>(
                x, weight, nullptr, out, (int)total_batch, false);
        } else {
            launch_linear_cutile_static_bf16<256, 512, false, false>(
                x, weight, nullptr, out, (int)total_batch, false);
        }
    } else {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool try_linear_cutile_static_small_bf16_bias_output(const Tensor& x,
                                                     const Tensor& weight,
                                                     const Tensor& bias,
                                                     int64_t total_batch,
                                                     int64_t out_features,
                                                     int64_t in_features,
                                                     Tensor& out) {
    if (!linear_cutile_static_small_bias_candidate(total_batch, out_features, in_features)) {
        return false;
    }
    if (x.dtype() != DType::BFloat16 || weight.dtype() != DType::BFloat16 ||
        bias.dtype() != DType::BFloat16) {
        return false;
    }

    out = Tensor::empty({total_batch, out_features}, DType::BFloat16);
    bool full_bf16 = full_bf16_arith_enabled();
    if (out_features == 1024 && in_features == 256) {
        launch_linear_cutile_static_padded_m_bf16_tiled<kLinearCutileSmallPaddedM32,
                                                        kLinearCutileSmallExpectedM,
                                                        32,
                                                        64,
                                                        64,
                                                        1024,
                                                        256,
                                                        true,
                                                        false>(
            x, weight, &bias, out, full_bf16);
    } else if (out_features == 1024 && in_features == 1024) {
        launch_linear_cutile_static_padded_m_bf16_tiled<kLinearCutileSmallPaddedM32,
                                                        kLinearCutileSmallExpectedM,
                                                        32,
                                                        64,
                                                        64,
                                                        1024,
                                                        1024,
                                                        true,
                                                        false>(
            x, weight, &bias, out, full_bf16);
    } else if (out_features == 256 && in_features == 16) {
        launch_linear_cutile_static_padded_m_bf16_tiled<kLinearCutileSmallPaddedM64,
                                                        kLinearCutileSmallExpectedM,
                                                        64,
                                                        64,
                                                        16,
                                                        256,
                                                        16,
                                                        true,
                                                        false>(
            x, weight, &bias, out, full_bf16);
    } else if (out_features == 256 && in_features == 32) {
        launch_linear_cutile_static_padded_m_bf16_tiled<kLinearCutileSmallPaddedM64,
                                                        kLinearCutileSmallExpectedM,
                                                        64,
                                                        64,
                                                        16,
                                                        256,
                                                        32,
                                                        true,
                                                        false>(
            x, weight, &bias, out, full_bf16);
    } else if (out_features == 256 && in_features == 64) {
        launch_linear_cutile_static_padded_m_bf16_tiled<kLinearCutileSmallPaddedM64,
                                                        kLinearCutileSmallExpectedM,
                                                        64,
                                                        64,
                                                        16,
                                                        256,
                                                        64,
                                                        true,
                                                        false>(
            x, weight, &bias, out, full_bf16);
    } else if (out_features == 256 && in_features > 0 && in_features <= 1024) {
        launch_linear_cutile_small_n256_dynamic_k_bf16(
            x, weight, bias, out, (int)in_features, full_bf16);
    } else if (out_features == 128 && in_features == 1024) {
        launch_linear_cutile_static_masked_m_bf16_tiled<kLinearCutileSmallExpectedM,
                                                        32,
                                                        64,
                                                        64,
                                                        128,
                                                        1024,
                                                        true,
                                                        false>(
            x, weight, &bias, out, full_bf16);
    } else if (out_features == 48 && in_features == 1024) {
        launch_linear_cutile_static_masked_mn_bf16_tiled<kLinearCutileSmallExpectedM,
                                                         32,
                                                         64,
                                                         64,
                                                         48,
                                                         1024,
                                                         true,
                                                         false>(
            x, weight, &bias, out, full_bf16);
    } else if (in_features == 1024) {
        switch ((int)out_features) {
            case 56:
                launch_linear_cutile_small_k1024_masked_n_bf16<56>(
                    x, weight, bias, out, full_bf16);
                break;
            case 72:
                launch_linear_cutile_small_k1024_masked_n_bf16<72>(
                    x, weight, bias, out, full_bf16);
                break;
            case 80:
                launch_linear_cutile_small_k1024_masked_n_bf16<80>(
                    x, weight, bias, out, full_bf16);
                break;
            case 88:
                launch_linear_cutile_small_k1024_masked_n_bf16<88>(
                    x, weight, bias, out, full_bf16);
                break;
            case 104:
                launch_linear_cutile_small_k1024_masked_n_bf16<104>(
                    x, weight, bias, out, full_bf16);
                break;
            case 120:
                launch_linear_cutile_small_k1024_masked_n_bf16<120>(
                    x, weight, bias, out, full_bf16);
                break;
            case 136:
                launch_linear_cutile_small_k1024_masked_n_bf16<136>(
                    x, weight, bias, out, full_bf16);
                break;
            case 152:
                launch_linear_cutile_small_k1024_masked_n_bf16<152>(
                    x, weight, bias, out, full_bf16);
                break;
            case 160:
                launch_linear_cutile_small_k1024_masked_n_bf16<160>(
                    x, weight, bias, out, full_bf16);
                break;
            case 176:
                launch_linear_cutile_small_k1024_masked_n_bf16<176>(
                    x, weight, bias, out, full_bf16);
                break;
            case 192:
                launch_linear_cutile_small_k1024_masked_n_bf16<192>(
                    x, weight, bias, out, full_bf16);
                break;
            case 208:
                launch_linear_cutile_small_k1024_masked_n_bf16<208>(
                    x, weight, bias, out, full_bf16);
                break;
            case 224:
                launch_linear_cutile_small_k1024_masked_n_bf16<224>(
                    x, weight, bias, out, full_bf16);
                break;
            case 232:
                launch_linear_cutile_small_k1024_masked_n_bf16<232>(
                    x, weight, bias, out, full_bf16);
                break;
            case 248:
                launch_linear_cutile_small_k1024_masked_n_bf16<248>(
                    x, weight, bias, out, full_bf16);
                break;
            case 264:
                launch_linear_cutile_small_k1024_masked_n_bf16<264>(
                    x, weight, bias, out, full_bf16);
                break;
            case 288:
                launch_linear_cutile_small_k1024_masked_n_bf16<288>(
                    x, weight, bias, out, full_bf16);
                break;
            case 312:
                launch_linear_cutile_small_k1024_masked_n_bf16<312>(
                    x, weight, bias, out, full_bf16);
                break;
            case 328:
                launch_linear_cutile_small_k1024_masked_n_bf16<328>(
                    x, weight, bias, out, full_bf16);
                break;
            case 352:
                launch_linear_cutile_small_k1024_masked_n_bf16<352>(
                    x, weight, bias, out, full_bf16);
                break;
            case 376:
                launch_linear_cutile_small_k1024_masked_n_bf16<376>(
                    x, weight, bias, out, full_bf16);
                break;
            case 400:
                launch_linear_cutile_small_k1024_masked_n_bf16<400>(
                    x, weight, bias, out, full_bf16);
                break;
            case 432:
                launch_linear_cutile_small_k1024_masked_n_bf16<432>(
                    x, weight, bias, out, full_bf16);
                break;
            case 456:
                launch_linear_cutile_small_k1024_masked_n_bf16<456>(
                    x, weight, bias, out, full_bf16);
                break;
            case 488:
                launch_linear_cutile_small_k1024_masked_n_bf16<488>(
                    x, weight, bias, out, full_bf16);
                break;
            case 528:
                launch_linear_cutile_small_k1024_masked_n_bf16<528>(
                    x, weight, bias, out, full_bf16);
                break;
            case 568:
                launch_linear_cutile_small_k1024_masked_n_bf16<568>(
                    x, weight, bias, out, full_bf16);
                break;
            case 608:
                launch_linear_cutile_small_k1024_masked_n_bf16<608>(
                    x, weight, bias, out, full_bf16);
                break;
            case 640:
                launch_linear_cutile_small_k1024_masked_n_bf16<640>(
                    x, weight, bias, out, full_bf16);
                break;
            case 688:
                launch_linear_cutile_small_k1024_masked_n_bf16<688>(
                    x, weight, bias, out, full_bf16);
                break;
            case 744:
                launch_linear_cutile_small_k1024_masked_n_bf16<744>(
                    x, weight, bias, out, full_bf16);
                break;
            case 792:
                launch_linear_cutile_small_k1024_masked_n_bf16<792>(
                    x, weight, bias, out, full_bf16);
                break;
            case 840:
                launch_linear_cutile_small_k1024_masked_n_bf16<840>(
                    x, weight, bias, out, full_bf16);
                break;
            case 904:
                launch_linear_cutile_small_k1024_masked_n_bf16<904>(
                    x, weight, bias, out, full_bf16);
                break;
            case 976:
                launch_linear_cutile_small_k1024_masked_n_bf16<976>(
                    x, weight, bias, out, full_bf16);
                break;
            case 1040:
                launch_linear_cutile_small_k1024_masked_n_bf16<1040>(
                    x, weight, bias, out, full_bf16);
                break;
            default:
                return false;
        }
    } else {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool try_linear_glu_last_dim_bf16_output_impl(const Tensor& x,
                                              const Tensor& weight,
                                              const Tensor& bias,
                                              Tensor& out) {
    if (!linear_glu_last_dim_fused_enabled()) return false;
    if (x.dtype() != DType::BFloat16 || weight.dtype() != DType::BFloat16 ||
        bias.dtype() != DType::BFloat16) {
        return false;
    }
    if (x.ndim() == 0 || weight.ndim() != 2 || bias.ndim() != 1) return false;

    int64_t in_features = weight.size(1);
    int64_t out_features = weight.size(0);
    if ((out_features % 2) != 0 || bias.size(0) != out_features) return false;

    int64_t total_batch = x.numel() / in_features;
    int64_t out_half = out_features / 2;
    if (total_batch != kLinearCutileSmallExpectedM || in_features != 1024 ||
        out_half != 24) {
        return false;
    }

    Tensor xb = x.contiguous();
    Tensor wb = weight.contiguous();
    Tensor bb = bias.contiguous();
    out = Tensor::empty({total_batch, out_half}, DType::BFloat16);
    if (linear_glu_last_dim_fused_tk32_enabled()) {
        launch_linear_glu_last_dim_static_masked_mn_bf16_tiled<
            kLinearCutileSmallExpectedM,
            32,
            32,
            32,
            24,
            1024>(xb, wb, bb, out, full_bf16_arith_enabled());
    } else {
        launch_linear_glu_last_dim_static_masked_mn_bf16_tiled<
            kLinearCutileSmallExpectedM,
            32,
            32,
            64,
            24,
            1024>(xb, wb, bb, out, full_bf16_arith_enabled());
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool try_linear_cutile_static_bf16_bias_output(const Tensor& x,
                                               const Tensor& weight,
                                               const Tensor& bias,
                                               int64_t total_batch,
                                               int64_t out_features,
                                               int64_t in_features,
                                               Tensor& out) {
    if (!linear_cutile_static_bf16_output_enabled()) return false;
    if (x.dtype() != DType::BFloat16 || weight.dtype() != DType::BFloat16 ||
        bias.dtype() != DType::BFloat16) {
        return false;
    }
    if (total_batch != kLinearCutileExpectedM) return false;
    if (!linear_cutile_static_ffn2_candidate(total_batch, out_features, in_features)) {
        return false;
    }
    out = Tensor::empty({total_batch, out_features}, DType::BFloat16);
    if (linear_ffn2_m32n128_enabled()) {
        launch_linear_cutile_static_bf16_tiled<kLinearCutileStaticM,
                                               32,
                                               128,
                                               kLinearCutileTileK,
                                               256,
                                               1024,
                                               true,
                                               false>(
            x, weight, &bias, out, (int)total_batch, full_bf16_arith_enabled());
    } else {
        launch_linear_cutile_static_bf16<256, 1024, true, false>(
            x, weight, &bias, out, (int)total_batch, full_bf16_arith_enabled());
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool try_linear_cutile_static_bf16_bias_gelu_output(const Tensor& x,
                                                    const Tensor& weight,
                                                    const Tensor& bias,
                                                    int64_t total_batch,
                                                    int64_t out_features,
                                                    int64_t in_features,
                                                    Tensor& out) {
    if (!linear_cutile_static_bf16_output_enabled()) return false;
    if (x.dtype() != DType::BFloat16 || weight.dtype() != DType::BFloat16 ||
        bias.dtype() != DType::BFloat16) {
        return false;
    }
    if (total_batch != kLinearCutileExpectedM) return false;
    if (!linear_cutile_static_ffn1_candidate(total_batch, out_features, in_features)) {
        return false;
    }
    out = Tensor::empty({total_batch, out_features}, DType::BFloat16);
    if (linear_ffn1_m32n128_enabled()) {
        launch_linear_cutile_static_bf16_tiled<kLinearCutileStaticM,
                                               32,
                                               128,
                                               kLinearCutileTileK,
                                               1024,
                                               256,
                                               true,
                                               true>(
            x, weight, &bias, out, (int)total_batch, full_bf16_arith_enabled());
    } else if (linear_ffn1_m64n32_enabled()) {
        launch_linear_cutile_static_bf16_tiled<kLinearCutileStaticM64,
                                               64,
                                               32,
                                               kLinearCutileTileK,
                                               1024,
                                               256,
                                               true,
                                               true>(
            x, weight, &bias, out, (int)total_batch, full_bf16_arith_enabled());
    } else if (linear_ffn1_m16n128_enabled()) {
        launch_linear_cutile_static_bf16_tiled<kLinearCutileStaticM,
                                               16,
                                               128,
                                               kLinearCutileTileK,
                                               1024,
                                               256,
                                               true,
                                               true>(
            x, weight, &bias, out, (int)total_batch, full_bf16_arith_enabled());
    } else {
        launch_linear_cutile_static_bf16<1024, 256, true, true>(
            x, weight, &bias, out, (int)total_batch, full_bf16_arith_enabled());
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool try_linear_cutile_static_bkn_bf16_output(const Tensor& x,
                                              const Tensor& weight,
                                              const Tensor& weight_bkn,
                                              const Tensor* bias,
                                              int64_t total_batch,
                                              int64_t out_features,
                                              int64_t in_features,
                                              bool add_bias,
                                              bool apply_gelu,
                                              Tensor& out) {
    if (!linear_bkn_long_path_enabled() && !linear_bkn_ffn_long_path_enabled()) return false;
    if (x.dtype() != DType::BFloat16 || weight.dtype() != DType::BFloat16 ||
        weight_bkn.dtype() != DType::BFloat16) {
        return false;
    }
    if (add_bias && (!bias || bias->dtype() != DType::BFloat16)) return false;
    if (weight_bkn.ndim() != 2 ||
        weight_bkn.size(0) != in_features ||
        weight_bkn.size(1) != out_features) {
        return false;
    }
    if (total_batch != kLinearCutileExpectedM) return false;

    out = Tensor::empty({total_batch, out_features}, DType::BFloat16);
    if (linear_bkn_long_path_enabled() &&
        !add_bias && !apply_gelu &&
        linear_cutile_static_qkv_candidate(total_batch, out_features, in_features)) {
        if (linear_bkn_qkv_tk16_enabled()) {
            if (linear_bkn_qkv_m64_tn128_enabled()) {
                if (linear_bkn_qkv_m64_tn128_lat1_enabled()) {
                    launch_linear_cutile_static_bkn_bf16_tiled<kLinearCutileStaticM64,
                                                               64,
                                                               128,
                                                               16,
                                                               1536,
                                                               256,
                                                               false,
                                                               false,
                                                               1>(
                        x, weight, weight_bkn, nullptr, out, (int)total_batch, false);
                } else {
                    launch_linear_cutile_static_bkn_bf16_tiled<kLinearCutileStaticM64,
                                                               64,
                                                               128,
                                                               16,
                                                               1536,
                                                               256,
                                                               false,
                                                               false,
                                                               2>(
                        x, weight, weight_bkn, nullptr, out, (int)total_batch, false);
                }
            } else {
                launch_linear_cutile_static_bkn_bf16_tiled<kLinearCutileStaticM,
                                                           32,
                                                           256,
                                                           16,
                                                           1536,
                                                           256,
                                                           false,
                                                           false,
                                                           2>(
                    x, weight, weight_bkn, nullptr, out, (int)total_batch, false);
            }
        } else {
            launch_linear_cutile_static_bkn_bf16_tiled<kLinearCutileStaticM,
                                                       32,
                                                       256,
                                                       kLinearCutileTileK,
                                                       1536,
                                                       256,
                                                       false,
                                                       false>(
                x, weight, weight_bkn, nullptr, out, (int)total_batch, false);
        }
    } else if (linear_bkn_ffn_long_path_enabled() &&
               add_bias && apply_gelu &&
               linear_cutile_static_ffn1_candidate(total_batch, out_features, in_features)) {
        launch_linear_cutile_static_bkn_bf16_tiled<kLinearCutileStaticM,
                                                   kLinearCutileTileM,
                                                   kLinearCutileTileN,
                                                   kLinearCutileTileK,
                                                   1024,
                                                   256,
                                                   true,
                                                   true>(
            x, weight, weight_bkn, bias, out, (int)total_batch, full_bf16_arith_enabled());
    } else if (linear_bkn_ffn_long_path_enabled() &&
               add_bias && !apply_gelu &&
               linear_cutile_static_ffn2_candidate(total_batch, out_features, in_features)) {
        launch_linear_cutile_static_bkn_bf16_tiled<kLinearCutileStaticM,
                                                   kLinearCutileTileM,
                                                   kLinearCutileTileN,
                                                   kLinearCutileTileK,
                                                   256,
                                                   1024,
                                                   true,
                                                   false>(
            x, weight, weight_bkn, bias, out, (int)total_batch, full_bf16_arith_enabled());
    } else {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <int GeluMode, bool FullBF16>
__tile_global__ void gelu_bf16_inplace_kernel(__nv_bfloat16* __restrict__ out,
                                              long long total) {
    out = ct::assume_aligned(out, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto x = ct::element_cast<float>(ct::load_masked(out + idx, in_bounds));
    auto gelu = gelu_selected<GeluMode, FullBF16>(x);
    ct::store_masked(out + idx, ct::element_cast<__nv_bfloat16>(gelu), in_bounds);
}

bool try_linear_cutile_static_bf16_bias_gelu_split_output(const Tensor& x,
                                                          const Tensor& weight,
                                                          const Tensor& bias,
                                                          int64_t total_batch,
                                                          int64_t out_features,
                                                          int64_t in_features,
                                                          Tensor& out) {
    if (!linear_gelu_split_enabled()) return false;
    if (!linear_cutile_static_bf16_output_enabled()) return false;
    if (x.dtype() != DType::BFloat16 || weight.dtype() != DType::BFloat16 ||
        bias.dtype() != DType::BFloat16) {
        return false;
    }
    if (total_batch != kLinearCutileExpectedM) return false;
    if (!linear_cutile_static_ffn1_candidate(total_batch, out_features, in_features)) {
        return false;
    }

    out = Tensor::empty({total_batch, out_features}, DType::BFloat16);
    launch_linear_cutile_static_bf16<1024, 256, true, false>(
        x, weight, &bias, out, (int)total_batch, full_bf16_arith_enabled());
    CUDA_CHECK(cudaGetLastError());

    long long total = total_batch * out_features;
    bool full_bf16 = full_bf16_arith_enabled();
    if (linear_tanh_gelu_enabled()) {
        if (full_bf16) {
            gelu_bf16_inplace_kernel<kGeluTanh, true><<<(int)ceildiv(total, kTile), 1>>>(
                out.data_bf16(), total);
        } else {
            gelu_bf16_inplace_kernel<kGeluTanh, false><<<(int)ceildiv(total, kTile), 1>>>(
                out.data_bf16(), total);
        }
    } else if (linear_quick_gelu_enabled()) {
        if (full_bf16) {
            gelu_bf16_inplace_kernel<kGeluQuick, true><<<(int)ceildiv(total, kTile), 1>>>(
                out.data_bf16(), total);
        } else {
            gelu_bf16_inplace_kernel<kGeluQuick, false><<<(int)ceildiv(total, kTile), 1>>>(
                out.data_bf16(), total);
        }
    } else if (linear_hard_gelu_enabled()) {
        if (full_bf16) {
            gelu_bf16_inplace_kernel<kGeluHard, true><<<(int)ceildiv(total, kTile), 1>>>(
                out.data_bf16(), total);
        } else {
            gelu_bf16_inplace_kernel<kGeluHard, false><<<(int)ceildiv(total, kTile), 1>>>(
                out.data_bf16(), total);
        }
    } else {
        if (full_bf16) {
            gelu_bf16_inplace_kernel<kGeluErf, true><<<(int)ceildiv(total, kTile), 1>>>(
                out.data_bf16(), total);
        } else {
            gelu_bf16_inplace_kernel<kGeluErf, false><<<(int)ceildiv(total, kTile), 1>>>(
                out.data_bf16(), total);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool try_linear_cutile_static_gate_sigmoid_bf16_output(const Tensor& x,
                                                       const Tensor& weight,
                                                       const Tensor& bias,
                                                       int64_t total_batch,
                                                       int64_t out_features,
                                                       int64_t in_features,
                                                       Tensor& out) {
    if (!linear_cutile_gate_sigmoid_bf16_output_enabled()) return false;
    if (x.dtype() != DType::BFloat16 || weight.dtype() != DType::BFloat16 ||
        bias.dtype() != DType::BFloat16) {
        return false;
    }
    if (total_batch != kLinearCutileExpectedM || out_features != 8 || in_features != 256) {
        return false;
    }

    out = Tensor::empty({total_batch, out_features}, DType::BFloat16);
    launch_linear_cutile_static_masked_mn_bf16_tiled<kLinearCutileExpectedM,
                                                     32,
                                                     16,
                                                     32,
                                                     8,
                                                     256,
                                                     true,
                                                     true>(
        x, weight, &bias, out, full_bf16_arith_enabled());
    CUDA_CHECK(cudaGetLastError());
    return true;
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

__tile_global__ void split_qkv_heads_rotary_bf16_kernel(const float* __restrict__ qkv,
                                                        const float* __restrict__ cos_f,
                                                        const float* __restrict__ sin_f,
                                                        __nv_bfloat16* __restrict__ q,
                                                        __nv_bfloat16* __restrict__ k,
                                                        __nv_bfloat16* __restrict__ v,
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

    ct::store_masked(q + out_base + pair_d,
                     ct::element_cast<__nv_bfloat16>(q0 * c - q1 * s), in_bounds);
    ct::store_masked(q + out_base + pair_d + 1,
                     ct::element_cast<__nv_bfloat16>(q0 * s + q1 * c), in_bounds);
    ct::store_masked(k + out_base + pair_d,
                     ct::element_cast<__nv_bfloat16>(k0 * c - k1 * s), in_bounds);
    ct::store_masked(k + out_base + pair_d + 1,
                     ct::element_cast<__nv_bfloat16>(k0 * s + k1 * c), in_bounds);
    ct::store_masked(v + out_base + pair_d, ct::element_cast<__nv_bfloat16>(v0), in_bounds);
    ct::store_masked(v + out_base + pair_d + 1, ct::element_cast<__nv_bfloat16>(v1), in_bounds);
}

template <typename TrigT = float>
__tile_global__ void split_qkv_heads_rotary_qkv_bf16_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    long long total,
    int heads,
    int n_tokens,
    int dim_head,
    bool full_bf16) {
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

    auto c = ct::element_cast<float>(
        ct::load_masked(cos_f + n * half_dim + i, in_bounds));
    auto s = ct::element_cast<float>(
        ct::load_masked(sin_f + n * half_dim + i, in_bounds));
    c = ct::select(full_bf16, bf16_round(c), c);
    s = ct::select(full_bf16, bf16_round(s), s);

    auto q0 = ct::element_cast<float>(ct::load_masked(qkv + qkv_base + head_offset, in_bounds));
    auto q1 = ct::element_cast<float>(ct::load_masked(qkv + qkv_base + head_offset + 1, in_bounds));
    auto k0 = ct::element_cast<float>(
        ct::load_masked(qkv + qkv_base + (long long)heads * dim_head + head_offset, in_bounds));
    auto k1 = ct::element_cast<float>(
        ct::load_masked(qkv + qkv_base + (long long)heads * dim_head + head_offset + 1, in_bounds));
    auto v0 = ct::load_masked(qkv + qkv_base + 2LL * heads * dim_head + head_offset, in_bounds);
    auto v1 = ct::load_masked(qkv + qkv_base + 2LL * heads * dim_head + head_offset + 1, in_bounds);
    q0 = ct::select(full_bf16, bf16_round(q0), q0);
    q1 = ct::select(full_bf16, bf16_round(q1), q1);
    k0 = ct::select(full_bf16, bf16_round(k0), k0);
    k1 = ct::select(full_bf16, bf16_round(k1), k1);

    auto q_rot0 = q0 * c - q1 * s;
    auto q_rot1 = q0 * s + q1 * c;
    auto k_rot0 = k0 * c - k1 * s;
    auto k_rot1 = k0 * s + k1 * c;
    q_rot0 = ct::select(full_bf16, bf16_round(q_rot0), q_rot0);
    q_rot1 = ct::select(full_bf16, bf16_round(q_rot1), q_rot1);
    k_rot0 = ct::select(full_bf16, bf16_round(k_rot0), k_rot0);
    k_rot1 = ct::select(full_bf16, bf16_round(k_rot1), k_rot1);

    ct::store_masked(q + out_base + pair_d, ct::element_cast<__nv_bfloat16>(q_rot0), in_bounds);
    ct::store_masked(q + out_base + pair_d + 1, ct::element_cast<__nv_bfloat16>(q_rot1), in_bounds);
    ct::store_masked(k + out_base + pair_d, ct::element_cast<__nv_bfloat16>(k_rot0), in_bounds);
    ct::store_masked(k + out_base + pair_d + 1, ct::element_cast<__nv_bfloat16>(k_rot1), in_bounds);
    ct::store_masked(v + out_base + pair_d, v0, in_bounds);
    ct::store_masked(v + out_base + pair_d + 1, v1, in_bounds);
}

template <typename TrigT = float>
__tile_global__ void split_qkv_heads_rotary_qkv_bf16_time1301_d64_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    long long total,
    bool full_bf16) {
    qkv = ct::assume_aligned(qkv, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    constexpr int kHalfDim = kTimeAttnD / 2;
    constexpr long long kTokenStride = 3LL * kQkvFusedHeads * kTimeAttnD;
    constexpr long long kHeadPlane = (long long)kHalfDim * kTimeAttnN;
    constexpr long long kBatchPlane = kHeadPlane * kQkvFusedHeads;

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    auto i = idx % kHalfDim;
    auto n = (idx / kHalfDim) % kTimeAttnN;
    auto h = (idx / kHeadPlane) % kQkvFusedHeads;
    auto b = idx / kBatchPlane;
    auto pair_d = i * 2;

    auto qkv_base = (b * kTimeAttnN + n) * kTokenStride;
    auto head_offset = h * kTimeAttnD + pair_d;
    auto out_base = ((b * kQkvFusedHeads + h) * kTimeAttnN + n) * kTimeAttnD;

    auto c = ct::element_cast<float>(
        ct::load_masked(cos_f + n * kHalfDim + i, in_bounds));
    auto s = ct::element_cast<float>(
        ct::load_masked(sin_f + n * kHalfDim + i, in_bounds));
    c = ct::select(full_bf16, bf16_round(c), c);
    s = ct::select(full_bf16, bf16_round(s), s);

    auto q0 = ct::element_cast<float>(ct::load_masked(qkv + qkv_base + head_offset, in_bounds));
    auto q1 = ct::element_cast<float>(ct::load_masked(qkv + qkv_base + head_offset + 1, in_bounds));
    auto k0 = ct::element_cast<float>(
        ct::load_masked(qkv + qkv_base + (long long)kQkvFusedHeads * kTimeAttnD + head_offset,
                        in_bounds));
    auto k1 = ct::element_cast<float>(
        ct::load_masked(qkv + qkv_base + (long long)kQkvFusedHeads * kTimeAttnD + head_offset + 1,
                        in_bounds));
    auto v0 = ct::load_masked(
        qkv + qkv_base + 2LL * kQkvFusedHeads * kTimeAttnD + head_offset, in_bounds);
    auto v1 = ct::load_masked(
        qkv + qkv_base + 2LL * kQkvFusedHeads * kTimeAttnD + head_offset + 1, in_bounds);
    q0 = ct::select(full_bf16, bf16_round(q0), q0);
    q1 = ct::select(full_bf16, bf16_round(q1), q1);
    k0 = ct::select(full_bf16, bf16_round(k0), k0);
    k1 = ct::select(full_bf16, bf16_round(k1), k1);

    auto q_rot0 = q0 * c - q1 * s;
    auto q_rot1 = q0 * s + q1 * c;
    auto k_rot0 = k0 * c - k1 * s;
    auto k_rot1 = k0 * s + k1 * c;
    q_rot0 = ct::select(full_bf16, bf16_round(q_rot0), q_rot0);
    q_rot1 = ct::select(full_bf16, bf16_round(q_rot1), q_rot1);
    k_rot0 = ct::select(full_bf16, bf16_round(k_rot0), k_rot0);
    k_rot1 = ct::select(full_bf16, bf16_round(k_rot1), k_rot1);

    ct::store_masked(q + out_base + pair_d, ct::element_cast<__nv_bfloat16>(q_rot0), in_bounds);
    ct::store_masked(q + out_base + pair_d + 1, ct::element_cast<__nv_bfloat16>(q_rot1), in_bounds);
    ct::store_masked(k + out_base + pair_d, ct::element_cast<__nv_bfloat16>(k_rot0), in_bounds);
    ct::store_masked(k + out_base + pair_d + 1, ct::element_cast<__nv_bfloat16>(k_rot1), in_bounds);
    ct::store_masked(v + out_base + pair_d, v0, in_bounds);
    ct::store_masked(v + out_base + pair_d + 1, v1, in_bounds);
}

template <int TileSize = kTile, typename TrigT = float>
__tile_global__ void rotary_time_split_contig_inplace_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    long long total,
    bool full_bf16) {
    static_assert(TileSize == 128 || TileSize == kTile ||
                  TileSize == 512 || TileSize == 1024);
    using I64RotaryTile = ct::tile<long long, ct::shape<TileSize>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);

    constexpr int kHalfDim = kTimeAttnD / 2;
    constexpr long long kTokenPairs = (long long)kQkvFusedHeads * kHalfDim;
    constexpr long long kBatchPairs = (long long)kTimeAttnN * kTokenPairs;

    I64RotaryTile idx =
        (long long)ct::bid().x * TileSize + ct::iota<I64RotaryTile>();
    auto in_bounds = idx < total;

    auto i = idx % kHalfDim;
    auto h = (idx / kHalfDim) % kQkvFusedHeads;
    auto n = (idx / kTokenPairs) % kTimeAttnN;
    auto b = idx / kBatchPairs;
    auto pair_d = i * 2;
    auto offset =
        ((b * kTimeAttnN + n) * kQkvFusedHeads + h) * kTimeAttnD + pair_d;

    auto c = ct::element_cast<float>(
        ct::load_masked(cos_f + n * kHalfDim + i, in_bounds));
    auto s = ct::element_cast<float>(
        ct::load_masked(sin_f + n * kHalfDim + i, in_bounds));
    c = ct::select(full_bf16, bf16_round(c), c);
    s = ct::select(full_bf16, bf16_round(s), s);

    auto q0 = ct::element_cast<float>(ct::load_masked(q + offset, in_bounds));
    auto q1 = ct::element_cast<float>(ct::load_masked(q + offset + 1, in_bounds));
    auto k0 = ct::element_cast<float>(ct::load_masked(k + offset, in_bounds));
    auto k1 = ct::element_cast<float>(ct::load_masked(k + offset + 1, in_bounds));
    q0 = ct::select(full_bf16, bf16_round(q0), q0);
    q1 = ct::select(full_bf16, bf16_round(q1), q1);
    k0 = ct::select(full_bf16, bf16_round(k0), k0);
    k1 = ct::select(full_bf16, bf16_round(k1), k1);

    auto q_rot0 = q0 * c - q1 * s;
    auto q_rot1 = q0 * s + q1 * c;
    auto k_rot0 = k0 * c - k1 * s;
    auto k_rot1 = k0 * s + k1 * c;
    q_rot0 = ct::select(full_bf16, bf16_round(q_rot0), q_rot0);
    q_rot1 = ct::select(full_bf16, bf16_round(q_rot1), q_rot1);
    k_rot0 = ct::select(full_bf16, bf16_round(k_rot0), k_rot0);
    k_rot1 = ct::select(full_bf16, bf16_round(k_rot1), k_rot1);

    ct::store_masked(q + offset, ct::element_cast<__nv_bfloat16>(q_rot0), in_bounds);
    ct::store_masked(q + offset + 1, ct::element_cast<__nv_bfloat16>(q_rot1), in_bounds);
    ct::store_masked(k + offset, ct::element_cast<__nv_bfloat16>(k_rot0), in_bounds);
    ct::store_masked(k + offset + 1, ct::element_cast<__nv_bfloat16>(k_rot1), in_bounds);
}

template <typename TrigT = float>
__tile_global__ void rotary_time_split_contig_k_only_inplace_kernel(
    __nv_bfloat16* __restrict__ k,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    long long total,
    bool full_bf16) {
    k = ct::assume_aligned(k, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);

    constexpr int kHalfDim = kTimeAttnD / 2;
    constexpr long long kTokenPairs = (long long)kQkvFusedHeads * kHalfDim;
    constexpr long long kBatchPairs = (long long)kTimeAttnN * kTokenPairs;

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    auto i = idx % kHalfDim;
    auto h = (idx / kHalfDim) % kQkvFusedHeads;
    auto n = (idx / kTokenPairs) % kTimeAttnN;
    auto b = idx / kBatchPairs;
    auto pair_d = i * 2;
    auto offset =
        ((b * kTimeAttnN + n) * kQkvFusedHeads + h) * kTimeAttnD + pair_d;

    auto c = ct::element_cast<float>(
        ct::load_masked(cos_f + n * kHalfDim + i, in_bounds));
    auto s = ct::element_cast<float>(
        ct::load_masked(sin_f + n * kHalfDim + i, in_bounds));
    c = ct::select(full_bf16, bf16_round(c), c);
    s = ct::select(full_bf16, bf16_round(s), s);

    auto k0 = ct::element_cast<float>(ct::load_masked(k + offset, in_bounds));
    auto k1 = ct::element_cast<float>(ct::load_masked(k + offset + 1, in_bounds));
    k0 = ct::select(full_bf16, bf16_round(k0), k0);
    k1 = ct::select(full_bf16, bf16_round(k1), k1);

    auto k_rot0 = k0 * c - k1 * s;
    auto k_rot1 = k0 * s + k1 * c;
    k_rot0 = ct::select(full_bf16, bf16_round(k_rot0), k_rot0);
    k_rot1 = ct::select(full_bf16, bf16_round(k_rot1), k_rot1);

    ct::store_masked(k + offset, ct::element_cast<__nv_bfloat16>(k_rot0), in_bounds);
    ct::store_masked(k + offset + 1, ct::element_cast<__nv_bfloat16>(k_rot1), in_bounds);
}

template <typename TrigT = float>
__tile_global__ void rotary_time_split_contig_row_tile_inplace_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    long long total_rows,
    bool full_bf16) {
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);

    constexpr int kHalfDim = kTimeAttnD / 2;
    using PairI64Tile = ct::tile<long long, ct::shape<kQkvFusedHeads, kHalfDim>>;
    using CosI64Tile = ct::tile<long long, ct::shape<1, kHalfDim>>;

    long long row = static_cast<long long>(ct::bid().x);
    auto row_valid = row < total_rows;
    auto n = row % kTimeAttnN;
    PairI64Tile local = ct::iota<PairI64Tile>();
    auto h = local / kHalfDim;
    auto i = local % kHalfDim;
    auto pair_d = i * 2;
    auto offset = row * (long long)kQkvFusedHeads * kTimeAttnD +
                  h * kTimeAttnD + pair_d;

    CosI64Tile cos_i = ct::iota<CosI64Tile>();
    auto c = ct::broadcast(ct::element_cast<float>(ct::load(cos_f + n * kHalfDim + cos_i)),
                           ct::shape<kQkvFusedHeads, kHalfDim>{});
    auto s = ct::broadcast(ct::element_cast<float>(ct::load(sin_f + n * kHalfDim + cos_i)),
                           ct::shape<kQkvFusedHeads, kHalfDim>{});
    c = ct::select(full_bf16, bf16_round(c), c);
    s = ct::select(full_bf16, bf16_round(s), s);

    auto q0 = ct::element_cast<float>(ct::load(q + offset));
    auto q1 = ct::element_cast<float>(ct::load(q + offset + 1));
    auto k0 = ct::element_cast<float>(ct::load(k + offset));
    auto k1 = ct::element_cast<float>(ct::load(k + offset + 1));
    q0 = ct::select(full_bf16, bf16_round(q0), q0);
    q1 = ct::select(full_bf16, bf16_round(q1), q1);
    k0 = ct::select(full_bf16, bf16_round(k0), k0);
    k1 = ct::select(full_bf16, bf16_round(k1), k1);

    auto q_rot0 = q0 * c - q1 * s;
    auto q_rot1 = q0 * s + q1 * c;
    auto k_rot0 = k0 * c - k1 * s;
    auto k_rot1 = k0 * s + k1 * c;
    q_rot0 = ct::select(full_bf16, bf16_round(q_rot0), q_rot0);
    q_rot1 = ct::select(full_bf16, bf16_round(q_rot1), q_rot1);
    k_rot0 = ct::select(full_bf16, bf16_round(k_rot0), k_rot0);
    k_rot1 = ct::select(full_bf16, bf16_round(k_rot1), k_rot1);

    ct::store_masked(q + offset, ct::element_cast<__nv_bfloat16>(q_rot0), row_valid);
    ct::store_masked(q + offset + 1, ct::element_cast<__nv_bfloat16>(q_rot1), row_valid);
    ct::store_masked(k + offset, ct::element_cast<__nv_bfloat16>(k_rot0), row_valid);
    ct::store_masked(k + offset + 1, ct::element_cast<__nv_bfloat16>(k_rot1), row_valid);
}

template <typename TrigT = float>
__tile_global__ void rotary_time_split_contig_tail_inplace_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    long long total_rows,
    long long row_start,
    bool full_bf16) {
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);

    constexpr int kHalfDim = kTimeAttnD / 2;
    constexpr long long kTokenPairs = (long long)kQkvFusedHeads * kHalfDim;
    long long tail_rows = total_rows - row_start;
    long long total = tail_rows * kTokenPairs;

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    auto pair_idx = idx % kTokenPairs;
    auto row = row_start + idx / kTokenPairs;
    auto i = pair_idx % kHalfDim;
    auto h = pair_idx / kHalfDim;
    auto n = row % kTimeAttnN;
    auto b = row / kTimeAttnN;
    auto pair_d = i * 2;
    auto offset =
        ((b * kTimeAttnN + n) * kQkvFusedHeads + h) * kTimeAttnD + pair_d;

    auto c = ct::element_cast<float>(
        ct::load_masked(cos_f + n * kHalfDim + i, in_bounds));
    auto s = ct::element_cast<float>(
        ct::load_masked(sin_f + n * kHalfDim + i, in_bounds));
    c = ct::select(full_bf16, bf16_round(c), c);
    s = ct::select(full_bf16, bf16_round(s), s);

    auto q0 = ct::element_cast<float>(ct::load_masked(q + offset, in_bounds));
    auto q1 = ct::element_cast<float>(ct::load_masked(q + offset + 1, in_bounds));
    auto k0 = ct::element_cast<float>(ct::load_masked(k + offset, in_bounds));
    auto k1 = ct::element_cast<float>(ct::load_masked(k + offset + 1, in_bounds));
    q0 = ct::select(full_bf16, bf16_round(q0), q0);
    q1 = ct::select(full_bf16, bf16_round(q1), q1);
    k0 = ct::select(full_bf16, bf16_round(k0), k0);
    k1 = ct::select(full_bf16, bf16_round(k1), k1);

    auto q_rot0 = q0 * c - q1 * s;
    auto q_rot1 = q0 * s + q1 * c;
    auto k_rot0 = k0 * c - k1 * s;
    auto k_rot1 = k0 * s + k1 * c;
    q_rot0 = ct::select(full_bf16, bf16_round(q_rot0), q_rot0);
    q_rot1 = ct::select(full_bf16, bf16_round(q_rot1), q_rot1);
    k_rot0 = ct::select(full_bf16, bf16_round(k_rot0), k_rot0);
    k_rot1 = ct::select(full_bf16, bf16_round(k_rot1), k_rot1);

    ct::store_masked(q + offset, ct::element_cast<__nv_bfloat16>(q_rot0), in_bounds);
    ct::store_masked(q + offset + 1, ct::element_cast<__nv_bfloat16>(q_rot1), in_bounds);
    ct::store_masked(k + offset, ct::element_cast<__nv_bfloat16>(k_rot0), in_bounds);
    ct::store_masked(k + offset + 1, ct::element_cast<__nv_bfloat16>(k_rot1), in_bounds);
}

template <typename TrigT = float>
__tile_global__ void split_qkv_heads_rotary_qkv_bf16_freq60_pad64_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    long long total,
    int heads,
    bool full_bf16) {
    qkv = ct::assume_aligned(qkv, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    constexpr int half_dim = kFreqAttnD / 2;
    auto i = idx % half_dim;
    auto n_pad = (idx / half_dim) % kFreqAttnPadN;
    auto h = (idx / ((long long)half_dim * kFreqAttnPadN)) % heads;
    auto b = idx / ((long long)half_dim * kFreqAttnPadN * heads);
    auto valid = in_bounds && (n_pad < kFreqAttnN);
    auto safe_n = ct::select(valid, n_pad, n_pad * 0LL);
    auto pair_d = i * 2;

    auto qkv_base = (b * kFreqAttnN + safe_n) * (3LL * heads * kFreqAttnD);
    auto head_offset = h * kFreqAttnD + pair_d;
    auto out_base = ((b * heads + h) * kFreqAttnPadN + n_pad) * kFreqAttnD;

    auto c = ct::element_cast<float>(
        ct::load_masked(cos_f + safe_n * half_dim + i, valid));
    auto s = ct::element_cast<float>(
        ct::load_masked(sin_f + safe_n * half_dim + i, valid));
    c = ct::select(full_bf16, bf16_round(c), c);
    s = ct::select(full_bf16, bf16_round(s), s);

    auto q0 = ct::element_cast<float>(ct::load_masked(qkv + qkv_base + head_offset, valid));
    auto q1 = ct::element_cast<float>(ct::load_masked(qkv + qkv_base + head_offset + 1, valid));
    auto k0 = ct::element_cast<float>(
        ct::load_masked(qkv + qkv_base + (long long)heads * kFreqAttnD + head_offset, valid));
    auto k1 = ct::element_cast<float>(
        ct::load_masked(qkv + qkv_base + (long long)heads * kFreqAttnD + head_offset + 1, valid));
    auto v0 = ct::load_masked(qkv + qkv_base + 2LL * heads * kFreqAttnD + head_offset, valid);
    auto v1 = ct::load_masked(qkv + qkv_base + 2LL * heads * kFreqAttnD + head_offset + 1, valid);
    q0 = ct::select(full_bf16, bf16_round(q0), q0);
    q1 = ct::select(full_bf16, bf16_round(q1), q1);
    k0 = ct::select(full_bf16, bf16_round(k0), k0);
    k1 = ct::select(full_bf16, bf16_round(k1), k1);

    auto q_rot0 = q0 * c - q1 * s;
    auto q_rot1 = q0 * s + q1 * c;
    auto k_rot0 = k0 * c - k1 * s;
    auto k_rot1 = k0 * s + k1 * c;
    q_rot0 = ct::select(full_bf16, bf16_round(q_rot0), q_rot0);
    q_rot1 = ct::select(full_bf16, bf16_round(q_rot1), q_rot1);
    k_rot0 = ct::select(full_bf16, bf16_round(k_rot0), k_rot0);
    k_rot1 = ct::select(full_bf16, bf16_round(k_rot1), k_rot1);

    auto zero_f = q_rot0 * 0.0f;
    auto zero_b = ct::element_cast<__nv_bfloat16>(zero_f);
    ct::store_masked(q + out_base + pair_d,
                     ct::element_cast<__nv_bfloat16>(ct::select(valid, q_rot0, zero_f)),
                     in_bounds);
    ct::store_masked(q + out_base + pair_d + 1,
                     ct::element_cast<__nv_bfloat16>(ct::select(valid, q_rot1, zero_f)),
                     in_bounds);
    ct::store_masked(k + out_base + pair_d,
                     ct::element_cast<__nv_bfloat16>(ct::select(valid, k_rot0, zero_f)),
                     in_bounds);
    ct::store_masked(k + out_base + pair_d + 1,
                     ct::element_cast<__nv_bfloat16>(ct::select(valid, k_rot1, zero_f)),
                     in_bounds);
    ct::store_masked(v + out_base + pair_d, ct::select(valid, v0, zero_b), in_bounds);
    ct::store_masked(v + out_base + pair_d + 1, ct::select(valid, v1, zero_b), in_bounds);
}

template <typename TrigT = float>
__tile_global__ void split_qkv_heads_rotary_qkv_bf16_freq60_to_pad64_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const TrigT* __restrict__ cos_f,
    const TrigT* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    long long total,
    int heads,
    bool full_bf16) {
    qkv = ct::assume_aligned(qkv, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    constexpr int half_dim = kFreqAttnD / 2;
    auto i = idx % half_dim;
    auto n = (idx / half_dim) % kFreqAttnN;
    auto h = (idx / ((long long)half_dim * kFreqAttnN)) % heads;
    auto b = idx / ((long long)half_dim * kFreqAttnN * heads);
    auto pair_d = i * 2;

    auto qkv_base = (b * kFreqAttnN + n) * (3LL * heads * kFreqAttnD);
    auto head_offset = h * kFreqAttnD + pair_d;
    auto out_base = ((b * heads + h) * kFreqAttnPadN + n) * kFreqAttnD;

    auto c = ct::element_cast<float>(
        ct::load_masked(cos_f + n * half_dim + i, in_bounds));
    auto s = ct::element_cast<float>(
        ct::load_masked(sin_f + n * half_dim + i, in_bounds));
    c = ct::select(full_bf16, bf16_round(c), c);
    s = ct::select(full_bf16, bf16_round(s), s);

    auto q0 = ct::element_cast<float>(ct::load_masked(qkv + qkv_base + head_offset, in_bounds));
    auto q1 = ct::element_cast<float>(ct::load_masked(qkv + qkv_base + head_offset + 1, in_bounds));
    auto k0 = ct::element_cast<float>(
        ct::load_masked(qkv + qkv_base + (long long)heads * kFreqAttnD + head_offset, in_bounds));
    auto k1 = ct::element_cast<float>(
        ct::load_masked(qkv + qkv_base + (long long)heads * kFreqAttnD + head_offset + 1, in_bounds));
    auto v0 = ct::load_masked(qkv + qkv_base + 2LL * heads * kFreqAttnD + head_offset, in_bounds);
    auto v1 = ct::load_masked(qkv + qkv_base + 2LL * heads * kFreqAttnD + head_offset + 1, in_bounds);
    q0 = ct::select(full_bf16, bf16_round(q0), q0);
    q1 = ct::select(full_bf16, bf16_round(q1), q1);
    k0 = ct::select(full_bf16, bf16_round(k0), k0);
    k1 = ct::select(full_bf16, bf16_round(k1), k1);

    auto q_rot0 = q0 * c - q1 * s;
    auto q_rot1 = q0 * s + q1 * c;
    auto k_rot0 = k0 * c - k1 * s;
    auto k_rot1 = k0 * s + k1 * c;
    q_rot0 = ct::select(full_bf16, bf16_round(q_rot0), q_rot0);
    q_rot1 = ct::select(full_bf16, bf16_round(q_rot1), q_rot1);
    k_rot0 = ct::select(full_bf16, bf16_round(k_rot0), k_rot0);
    k_rot1 = ct::select(full_bf16, bf16_round(k_rot1), k_rot1);

    ct::store_masked(q + out_base + pair_d,
                     ct::element_cast<__nv_bfloat16>(q_rot0),
                     in_bounds);
    ct::store_masked(q + out_base + pair_d + 1,
                     ct::element_cast<__nv_bfloat16>(q_rot1),
                     in_bounds);
    ct::store_masked(k + out_base + pair_d,
                     ct::element_cast<__nv_bfloat16>(k_rot0),
                     in_bounds);
    ct::store_masked(k + out_base + pair_d + 1,
                     ct::element_cast<__nv_bfloat16>(k_rot1),
                     in_bounds);
    ct::store_masked(v + out_base + pair_d, v0, in_bounds);
    ct::store_masked(v + out_base + pair_d + 1, v1, in_bounds);
}

__tile_global__ void freq60_v_pad_rows_zero_kernel(__nv_bfloat16* __restrict__ v,
                                                   long long total,
                                                   int heads) {
    v = ct::assume_aligned(v, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto d = idx % kFreqAttnD;
    auto pad_row = (idx / kFreqAttnD) % (kFreqAttnPadN - kFreqAttnN);
    auto h = (idx / ((long long)kFreqAttnD * (kFreqAttnPadN - kFreqAttnN))) % heads;
    auto b = idx / ((long long)kFreqAttnD * (kFreqAttnPadN - kFreqAttnN) * heads);
    auto n = pad_row + kFreqAttnN;
    auto out_idx = ((b * heads + h) * kFreqAttnPadN + n) * kFreqAttnD + d;
    auto zero = ct::element_cast<__nv_bfloat16>(ct::element_cast<float>(idx * 0LL));
    ct::store_masked(v + out_idx, zero, in_bounds);
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

__tile_global__ void gather_freqs_fold_complex_to_bf16_kernel(
    const float* __restrict__ stft,
    const int64_t* __restrict__ freq_indices,
    __nv_bfloat16* __restrict__ out,
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
    ct::store_masked(out + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
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

__tile_global__ void apply_gates_and_merge_heads_gate_bf16_kernel(
    const float* __restrict__ attn,
    const __nv_bfloat16* __restrict__ gates,
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

    auto gate_values = ct::element_cast<float>(ct::load_masked(gates + gate_idx, in_bounds));
    auto values = ct::load_masked(attn + idx, in_bounds) * gate_values;
    ct::store_masked(merged + merged_idx, values, in_bounds);
}

__tile_global__ void apply_gates_and_merge_heads_to_bf16_kernel(
    const float* __restrict__ attn,
    const float* __restrict__ gates,
    __nv_bfloat16* __restrict__ merged,
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
    ct::store_masked(merged + merged_idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

__tile_global__ void apply_gates_and_merge_heads_bf16_gate_to_bf16_kernel(
    const float* __restrict__ attn,
    const __nv_bfloat16* __restrict__ gates,
    __nv_bfloat16* __restrict__ merged,
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

    auto gate_values = ct::element_cast<float>(ct::load_masked(gates + gate_idx, in_bounds));
    auto values = ct::load_masked(attn + idx, in_bounds) * gate_values;
    ct::store_masked(merged + merged_idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

template <typename AttnT, typename GateT, typename OutT>
__tile_global__ void apply_gates_and_merge_heads_typed_kernel(
    const AttnT* __restrict__ attn,
    const GateT* __restrict__ gates,
    OutT* __restrict__ merged,
    long long total,
    int heads,
    int n_tokens,
    int dim_head,
    bool full_bf16) {
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

    F32Tile attn_values = ct::element_cast<float>(ct::load_masked(attn + idx, in_bounds));
    F32Tile gate_values = ct::element_cast<float>(ct::load_masked(gates + gate_idx, in_bounds));
    attn_values = ct::select(full_bf16, bf16_round(attn_values), attn_values);
    gate_values = ct::select(full_bf16, bf16_round(gate_values), gate_values);
    auto values = attn_values * gate_values;
    values = ct::select(full_bf16, bf16_round(values), values);
    if constexpr (std::is_same_v<OutT, __nv_bfloat16>) {
        ct::store_masked(merged + merged_idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
    } else {
        ct::store_masked(merged + merged_idx, values, in_bounds);
    }
}

__tile_global__ void apply_gates_and_merge_heads_bf16_token_d64_cutile_kernel(
    const __nv_bfloat16* __restrict__ attn,
    const __nv_bfloat16* __restrict__ gates,
    __nv_bfloat16* __restrict__ merged,
    int tokens,
    int n_tokens,
    bool full_bf16) {
    using GateI64Tile = ct::tile<long long, ct::shape<kGateMergeTokenD64Tile>>;
    using GateF32Tile = ct::tile<float, ct::shape<kGateMergeTokenD64Tile>>;

    attn = ct::assume_aligned(attn, 16_ic);
    gates = ct::assume_aligned(gates, 16_ic);
    merged = ct::assume_aligned(merged, 16_ic);

    int token = static_cast<int>(ct::bid().x);
    auto e = ct::iota<GateI64Tile>();
    auto in_bounds = token < tokens;

    int n = token % n_tokens;
    int b = token / n_tokens;
    auto h = e / kTimeAttnD;
    auto d = e % kTimeAttnD;

    auto src_idx = ((static_cast<long long>(b) * kQkvFusedHeads + h) * n_tokens + n) *
                   kTimeAttnD + d;
    auto gate_idx = static_cast<long long>(token) * kQkvFusedHeads + h;
    auto dst_idx = static_cast<long long>(token) * kGateMergeTokenD64Tile + e;

    GateF32Tile attn_values =
        ct::element_cast<float>(ct::load_masked(attn + src_idx, in_bounds));
    GateF32Tile gate_values =
        ct::element_cast<float>(ct::load_masked(gates + gate_idx, in_bounds));
    attn_values = ct::select(full_bf16, bf16_round(attn_values), attn_values);
    gate_values = ct::select(full_bf16, bf16_round(gate_values), gate_values);
    auto values = attn_values * gate_values;
    values = ct::select(full_bf16, bf16_round(values), values);
    ct::store_masked(merged + dst_idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
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

template <typename InT, typename GammaT>
__tile_global__ void rms_norm_to_bf16_kernel(const InT* __restrict__ x,
                                             const GammaT* __restrict__ gamma,
                                             __nv_bfloat16* __restrict__ out,
                                             int dim,
                                             float scale,
                                             bool full_bf16) {
    x = ct::assume_aligned(x, 16_ic);
    gamma = ct::assume_aligned(gamma, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    long long row = (long long)ct::bid().x;
    RmsI64Tile d = ct::iota<RmsI64Tile>();
    auto in_bounds = d < dim;
    auto row_offset = row * dim;

    RmsF32Tile values = ct::element_cast<float>(ct::load_masked(x + row_offset + d, in_bounds));
    values = ct::select(full_bf16, bf16_round(values), values);
    auto zeros = values * 0.0f;
    auto sum_sq = ct::sum<0>(ct::select(in_bounds, values * values, zeros));
    sum_sq = ct::select(full_bf16, bf16_round(sum_sq), sum_sq);
    auto eps = sum_sq * 0.0f + 1.0e-12f;
    auto inv_norm = ct::rsqrt(sum_sq + eps);
    RmsF32Tile gamma_values =
        ct::element_cast<float>(ct::load_masked(gamma + d, in_bounds));
    inv_norm = ct::select(full_bf16, bf16_round(inv_norm), inv_norm);
    gamma_values = ct::select(full_bf16, bf16_round(gamma_values), gamma_values);
    auto normalized = values * inv_norm * gamma_values * scale;
    normalized = ct::select(full_bf16, bf16_round(normalized), normalized);

    ct::store_masked(out + row_offset + d, ct::element_cast<__nv_bfloat16>(normalized), in_bounds);
}

template <typename InT, typename GammaT, bool FullBF16>
__tile_global__ void rms_norm_d256_to_bf16_kernel(const InT* __restrict__ x,
                                                  const GammaT* __restrict__ gamma,
                                                  __nv_bfloat16* __restrict__ out,
                                                  float scale) {
    x = ct::assume_aligned(x, 16_ic);
    gamma = ct::assume_aligned(gamma, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    long long row = (long long)ct::bid().x;
    RmsD256I64Tile d = ct::iota<RmsD256I64Tile>();
    auto row_offset = row * kRmsD256;

    RmsD256F32Tile values = ct::element_cast<float>(ct::load(x + row_offset + d));
    values = bf16_round_if<FullBF16>(values);
    auto sum_sq = ct::sum<0>(values * values);
    sum_sq = bf16_round_if<FullBF16>(sum_sq);
    auto eps = sum_sq * 0.0f + 1.0e-12f;
    auto inv_norm = ct::rsqrt(sum_sq + eps);
    inv_norm = bf16_round_if<FullBF16>(inv_norm);
    RmsD256F32Tile gamma_values =
        ct::element_cast<float>(ct::load(gamma + d));
    gamma_values = bf16_round_if<FullBF16>(gamma_values);
    auto normalized = values * inv_norm * gamma_values * scale;
    normalized = bf16_round_if<FullBF16>(normalized);

    ct::store(out + row_offset + d, ct::element_cast<__nv_bfloat16>(normalized));
}

template <typename GammaT, bool FullBF16>
__tile_global__ void residual_add_rms_norm_d256_to_bf16_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ update,
    const GammaT* __restrict__ gamma,
    __nv_bfloat16* __restrict__ residual_out,
    __nv_bfloat16* __restrict__ normed,
    float scale) {
    residual = ct::assume_aligned(residual, 16_ic);
    update = ct::assume_aligned(update, 16_ic);
    gamma = ct::assume_aligned(gamma, 16_ic);
    residual_out = ct::assume_aligned(residual_out, 16_ic);
    normed = ct::assume_aligned(normed, 16_ic);

    long long row = (long long)ct::bid().x;
    RmsD256I64Tile d = ct::iota<RmsD256I64Tile>();
    auto row_offset = row * kRmsD256;

    RmsD256F32Tile values =
        ct::element_cast<float>(ct::load(residual + row_offset + d)) +
        ct::element_cast<float>(ct::load(update + row_offset + d));
    values = bf16_round(values);
    ct::store(residual_out + row_offset + d, ct::element_cast<__nv_bfloat16>(values));

    values = bf16_round_if<FullBF16>(values);
    auto sum_sq = ct::sum<0>(values * values);
    sum_sq = bf16_round_if<FullBF16>(sum_sq);
    auto eps = sum_sq * 0.0f + 1.0e-12f;
    auto inv_norm = ct::rsqrt(sum_sq + eps);
    inv_norm = bf16_round_if<FullBF16>(inv_norm);
    RmsD256F32Tile gamma_values =
        ct::element_cast<float>(ct::load(gamma + d));
    gamma_values = bf16_round_if<FullBF16>(gamma_values);
    auto normalized = values * inv_norm * gamma_values * scale;
    normalized = bf16_round_if<FullBF16>(normalized);

    ct::store(normed + row_offset + d, ct::element_cast<__nv_bfloat16>(normalized));
}

template <typename GammaT, bool FullBF16, int RowsPerCta>
__tile_global__ void residual_add_rms_norm_d256_rows_to_bf16_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ update,
    const GammaT* __restrict__ gamma,
    __nv_bfloat16* __restrict__ residual_out,
    __nv_bfloat16* __restrict__ normed,
    long long total_rows,
    float scale) {
    static_assert(RowsPerCta == 2 || RowsPerCta == 4);
    using RowsI64Tile = ct::tile<long long, ct::shape<RowsPerCta, kRmsD256>>;
    using RowsF32Tile = ct::tile<float, ct::shape<RowsPerCta, kRmsD256>>;

    residual = ct::assume_aligned(residual, 16_ic);
    update = ct::assume_aligned(update, 16_ic);
    gamma = ct::assume_aligned(gamma, 16_ic);
    residual_out = ct::assume_aligned(residual_out, 16_ic);
    normed = ct::assume_aligned(normed, 16_ic);

    RowsI64Tile local = ct::iota<RowsI64Tile>();
    auto row = static_cast<long long>(ct::bid().x) * RowsPerCta +
               local / kRmsD256;
    auto d = local % kRmsD256;
    auto valid = row < total_rows;

    RowsF32Tile values =
        ct::element_cast<float>(ct::load_masked(residual + row * kRmsD256 + d, valid)) +
        ct::element_cast<float>(ct::load_masked(update + row * kRmsD256 + d, valid));
    values = bf16_round(values);
    ct::store_masked(residual_out + row * kRmsD256 + d,
                     ct::element_cast<__nv_bfloat16>(values),
                     valid);

    values = bf16_round_if<FullBF16>(values);
    auto zeros = values * 0.0f;
    auto sum_sq = ct::sum<1>(ct::select(valid, values * values, zeros));
    sum_sq = bf16_round_if<FullBF16>(sum_sq);
    auto eps = sum_sq * 0.0f + 1.0e-12f;
    auto inv_norm = ct::rsqrt(sum_sq + eps);
    inv_norm = bf16_round_if<FullBF16>(inv_norm);
    RowsF32Tile gamma_values =
        ct::element_cast<float>(ct::load(gamma + d));
    gamma_values = bf16_round_if<FullBF16>(gamma_values);
    auto normalized = values * inv_norm * gamma_values * scale;
    normalized = bf16_round_if<FullBF16>(normalized);

    ct::store_masked(normed + row * kRmsD256 + d,
                     ct::element_cast<__nv_bfloat16>(normalized),
                     valid);
}

template <typename InT, typename GammaT, bool FullBF16, int RowsPerCta>
__tile_global__ void rms_norm_d256_rows_to_bf16_kernel(const InT* __restrict__ x,
                                                       const GammaT* __restrict__ gamma,
                                                       __nv_bfloat16* __restrict__ out,
                                                       long long total_rows,
                                                       float scale) {
    static_assert(RowsPerCta == 2 || RowsPerCta == 4);
    using RowsI64Tile = ct::tile<long long, ct::shape<RowsPerCta, kRmsD256>>;
    using RowsF32Tile = ct::tile<float, ct::shape<RowsPerCta, kRmsD256>>;

    x = ct::assume_aligned(x, 16_ic);
    gamma = ct::assume_aligned(gamma, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    RowsI64Tile local = ct::iota<RowsI64Tile>();
    auto row = static_cast<long long>(ct::bid().x) * RowsPerCta +
               local / kRmsD256;
    auto d = local % kRmsD256;
    auto valid = row < total_rows;

    RowsF32Tile values =
        ct::element_cast<float>(ct::load_masked(x + row * kRmsD256 + d, valid));
    values = bf16_round_if<FullBF16>(values);
    auto zeros = values * 0.0f;
    auto sum_sq = ct::sum<1>(ct::select(valid, values * values, zeros));
    sum_sq = bf16_round_if<FullBF16>(sum_sq);
    auto eps = sum_sq * 0.0f + 1.0e-12f;
    auto inv_norm = ct::rsqrt(sum_sq + eps);
    inv_norm = bf16_round_if<FullBF16>(inv_norm);
    RowsF32Tile gamma_values =
        ct::element_cast<float>(ct::load(gamma + d));
    gamma_values = bf16_round_if<FullBF16>(gamma_values);
    auto normalized = values * inv_norm * gamma_values * scale;
    normalized = bf16_round_if<FullBF16>(normalized);

    ct::store_masked(out + row * kRmsD256 + d,
                     ct::element_cast<__nv_bfloat16>(normalized),
                     valid);
}

template <typename InT, typename GammaT>
void launch_rms_norm_d256_to_bf16_typed(const InT* x,
                                        const GammaT* gamma,
                                        __nv_bfloat16* out,
                                        long long rows,
                                        float scale,
                                        bool full_bf16,
                                        int rows_per_cta) {
    if (rows_per_cta == 4) {
        if (full_bf16) {
            rms_norm_d256_rows_to_bf16_kernel<InT, GammaT, true, 4>
                <<<(int)ceildiv(rows, 4), 1>>>(x, gamma, out, rows, scale);
        } else {
            rms_norm_d256_rows_to_bf16_kernel<InT, GammaT, false, 4>
                <<<(int)ceildiv(rows, 4), 1>>>(x, gamma, out, rows, scale);
        }
    } else if (rows_per_cta == 2) {
        if (full_bf16) {
            rms_norm_d256_rows_to_bf16_kernel<InT, GammaT, true, 2>
                <<<(int)ceildiv(rows, 2), 1>>>(x, gamma, out, rows, scale);
        } else {
            rms_norm_d256_rows_to_bf16_kernel<InT, GammaT, false, 2>
                <<<(int)ceildiv(rows, 2), 1>>>(x, gamma, out, rows, scale);
        }
    } else if (full_bf16) {
        rms_norm_d256_to_bf16_kernel<InT, GammaT, true>
            <<<(int)rows, 1>>>(x, gamma, out, scale);
    } else {
        rms_norm_d256_to_bf16_kernel<InT, GammaT, false>
            <<<(int)rows, 1>>>(x, gamma, out, scale);
    }
}

template <typename GammaT>
void launch_residual_add_rms_norm_d256_to_bf16_typed(
    const __nv_bfloat16* residual,
    const __nv_bfloat16* update,
    const GammaT* gamma,
    __nv_bfloat16* residual_out,
    __nv_bfloat16* normed,
    long long rows,
    float scale,
    bool full_bf16,
    int rows_per_cta) {
    if (rows_per_cta == 4) {
        if (full_bf16) {
            residual_add_rms_norm_d256_rows_to_bf16_kernel<GammaT, true, 4>
                <<<(int)ceildiv(rows, 4), 1>>>(
                    residual, update, gamma, residual_out, normed, rows, scale);
        } else {
            residual_add_rms_norm_d256_rows_to_bf16_kernel<GammaT, false, 4>
                <<<(int)ceildiv(rows, 4), 1>>>(
                    residual, update, gamma, residual_out, normed, rows, scale);
        }
    } else if (rows_per_cta == 2) {
        if (full_bf16) {
            residual_add_rms_norm_d256_rows_to_bf16_kernel<GammaT, true, 2>
                <<<(int)ceildiv(rows, 2), 1>>>(
                    residual, update, gamma, residual_out, normed, rows, scale);
        } else {
            residual_add_rms_norm_d256_rows_to_bf16_kernel<GammaT, false, 2>
                <<<(int)ceildiv(rows, 2), 1>>>(
                    residual, update, gamma, residual_out, normed, rows, scale);
        }
    } else if (full_bf16) {
        residual_add_rms_norm_d256_to_bf16_kernel<GammaT, true>
            <<<(int)rows, 1>>>(residual, update, gamma, residual_out, normed, scale);
    } else {
        residual_add_rms_norm_d256_to_bf16_kernel<GammaT, false>
            <<<(int)rows, 1>>>(residual, update, gamma, residual_out, normed, scale);
    }
}

template <typename InT, typename GammaT>
void launch_rms_norm_to_bf16_typed(const InT* x,
                                   const GammaT* gamma,
                                   __nv_bfloat16* out,
                                   long long rows,
                                   int dim,
                                   float scale,
                                   bool full_bf16) {
    rms_norm_to_bf16_kernel<<<(int)rows, 1>>>(
        x, gamma, out, dim, scale, full_bf16);
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

__tile_global__ void scale_softmax_to_bf16_kernel(const float* __restrict__ data,
                                                  __nv_bfloat16* __restrict__ out,
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

    SoftmaxBF16Tile bf16_values(exp_values / denom);
    ct::store_masked(out + row_offset + col, bf16_values, in_bounds);
}

__tile_global__ void scale_softmax_small_kernel(float* __restrict__ data,
                                                int cols,
                                                float scale) {
    data = ct::assume_aligned(data, 16_ic);

    long long row = (long long)ct::bid().x;
    SmallSoftmaxI64Tile col = ct::iota<SmallSoftmaxI64Tile>();
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

__tile_global__ void scale_softmax_small_to_half_kernel(const float* __restrict__ data,
                                                        __half* __restrict__ out,
                                                        int cols,
                                                        float scale) {
    data = ct::assume_aligned(data, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    long long row = (long long)ct::bid().x;
    SmallSoftmaxI64Tile col = ct::iota<SmallSoftmaxI64Tile>();
    auto in_bounds = col < cols;
    auto row_offset = row * cols;

    auto values = ct::load_masked(data + row_offset + col, in_bounds);
    auto zeros = values * 0.0f;
    auto scaled = ct::select(in_bounds, values * scale, zeros - 3.402823466e38f);
    auto row_max = ct::reduce_max<0>(scaled);
    auto exp_values = ct::select(in_bounds, ct::exp(scaled - row_max), zeros);
    auto denom = ct::sum<0>(exp_values);

    SmallSoftmaxF16Tile half_values(exp_values / denom);
    ct::store_masked(out + row_offset + col, half_values, in_bounds);
}

__tile_global__ void scale_softmax_small_to_bf16_kernel(const float* __restrict__ data,
                                                        __nv_bfloat16* __restrict__ out,
                                                        int cols,
                                                        float scale) {
    data = ct::assume_aligned(data, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    long long row = (long long)ct::bid().x;
    SmallSoftmaxI64Tile col = ct::iota<SmallSoftmaxI64Tile>();
    auto in_bounds = col < cols;
    auto row_offset = row * cols;

    auto values = ct::load_masked(data + row_offset + col, in_bounds);
    auto zeros = values * 0.0f;
    auto scaled = ct::select(in_bounds, values * scale, zeros - 3.402823466e38f);
    auto row_max = ct::reduce_max<0>(scaled);
    auto exp_values = ct::select(in_bounds, ct::exp(scaled - row_max), zeros);
    auto denom = ct::sum<0>(exp_values);

    SmallSoftmaxBF16Tile bf16_values(exp_values / denom);
    ct::store_masked(out + row_offset + col, bf16_values, in_bounds);
}

__tile_global__ void scale_softmax_bf16_to_bf16_kernel(const __nv_bfloat16* __restrict__ data,
                                                       __nv_bfloat16* __restrict__ out,
                                                       int cols,
                                                       float scale,
                                                       bool full_bf16) {
    data = ct::assume_aligned(data, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    long long row = (long long)ct::bid().x;
    SoftmaxI64Tile col = ct::iota<SoftmaxI64Tile>();
    auto in_bounds = col < cols;
    auto row_offset = row * cols;

    auto values = ct::element_cast<float>(ct::load_masked(data + row_offset + col, in_bounds));
    values = ct::select(full_bf16, bf16_round(values), values);
    auto zeros = values * 0.0f;
    auto scaled = ct::select(in_bounds, values * scale, zeros - 3.402823466e38f);
    scaled = ct::select(full_bf16, bf16_round(scaled), scaled);
    auto row_max = ct::reduce_max<0>(scaled);
    row_max = ct::select(full_bf16, bf16_round(row_max), row_max);
    auto exp_values = ct::select(in_bounds, ct::exp(scaled - row_max), zeros);
    exp_values = ct::select(full_bf16, bf16_round(exp_values), exp_values);
    auto denom = ct::sum<0>(exp_values);
    denom = ct::select(full_bf16, bf16_round(denom), denom);
    auto probs = exp_values / denom;
    probs = ct::select(full_bf16, bf16_round(probs), probs);

    SoftmaxBF16Tile bf16_values(probs);
    ct::store_masked(out + row_offset + col, bf16_values, in_bounds);
}

__tile_global__ void scale_softmax_small_bf16_to_bf16_kernel(
    const __nv_bfloat16* __restrict__ data,
    __nv_bfloat16* __restrict__ out,
    int cols,
    float scale,
    bool full_bf16) {
    data = ct::assume_aligned(data, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    long long row = (long long)ct::bid().x;
    SmallSoftmaxI64Tile col = ct::iota<SmallSoftmaxI64Tile>();
    auto in_bounds = col < cols;
    auto row_offset = row * cols;

    auto values = ct::element_cast<float>(ct::load_masked(data + row_offset + col, in_bounds));
    values = ct::select(full_bf16, bf16_round(values), values);
    auto zeros = values * 0.0f;
    auto scaled = ct::select(in_bounds, values * scale, zeros - 3.402823466e38f);
    scaled = ct::select(full_bf16, bf16_round(scaled), scaled);
    auto row_max = ct::reduce_max<0>(scaled);
    row_max = ct::select(full_bf16, bf16_round(row_max), row_max);
    auto exp_values = ct::select(in_bounds, ct::exp(scaled - row_max), zeros);
    exp_values = ct::select(full_bf16, bf16_round(exp_values), exp_values);
    auto denom = ct::sum<0>(exp_values);
    denom = ct::select(full_bf16, bf16_round(denom), denom);
    auto probs = exp_values / denom;
    probs = ct::select(full_bf16, bf16_round(probs), probs);

    SmallSoftmaxBF16Tile bf16_values(probs);
    ct::store_masked(out + row_offset + col, bf16_values, in_bounds);
}

__tile_global__ void attention_qk_bf16_accum_kernel(const __nv_bfloat16* __restrict__ q,
                                                    const __nv_bfloat16* __restrict__ k,
                                                    __nv_bfloat16* __restrict__ scores,
                                                    long long total,
                                                    int N,
                                                    int N_k,
                                                    int D) {
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    scores = ct::assume_aligned(scores, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto nk = idx % N_k;
    auto n = (idx / N_k) % N;
    auto bh = idx / ((long long)N * N_k);

    F32Tile acc = ct::element_cast<float>(idx * 0LL);
    for (int d = 0; d < D; ++d) {
        F32Tile qv = ct::element_cast<float>(
            ct::load_masked(q + (bh * (long long)N + n) * D + d, in_bounds));
        F32Tile kv = ct::element_cast<float>(
            ct::load_masked(k + (bh * (long long)N_k + nk) * D + d, in_bounds));
        acc = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(acc + qv * kv));
    }
    ct::store_masked(scores + idx, ct::element_cast<__nv_bfloat16>(acc), in_bounds);
}

template <int QRows, bool ConstNegInf = false>
__tile_global__ void freq_attention60_cutile_padded_out60_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    using ScoreTile = ct::tile<float, ct::shape<QRows, kFreqAttnPadN>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kFreqAttnD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, kFreqAttnPadN>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kFreqAttnD>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kFreqAttnPadN * kFreqAttnD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kFreqAttnPadN * kFreqAttnD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kFreqAttnPadN * kFreqAttnD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kFreqAttnN * kFreqAttnD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kFreqAttnPadN, kFreqAttnD>{}},
        ct::shape<QRows, kFreqAttnD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kFreqAttnD, kFreqAttnPadN>{}, ct::layout_left{}},
        ct::shape<kFreqAttnD, kFreqAttnPadN>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kFreqAttnPadN, kFreqAttnD>{}},
        ct::shape<kFreqAttnPadN, kFreqAttnD>{}
    };

    auto scores = ct::mma(q_view.load(q_block, 0), k_t_view.load(0, 0),
                          ct::full<ScoreTile>(0.0f));
    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows +
                      score_local / kFreqAttnPadN;
    auto score_cols = score_local % kFreqAttnPadN;
    auto score_valid = (score_rows < kFreqAttnN) && (score_cols < kFreqAttnN);
    auto neg_inf = [&]() {
        if constexpr (ConstNegInf) {
            return ct::full<ScoreTile>(-3.402823466e38f);
        } else {
            return scores * 0.0f - 3.402823466e38f;
        }
    }();
    scores = ct::select(score_valid, scores * scale, neg_inf);

    auto row_max = ct::reduce_max<1>(scores);
    auto probs_f32 = ct::select(score_valid, ct::exp(scores - row_max), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto denom = ct::sum<1>(probs_f32);

    auto out_acc = ct::mma(probs_bf16, v_view.load(0, 0), ct::full<OutTile>(0.0f));
    out_acc = out_acc / denom;

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows +
                    out_local / kFreqAttnD;
    auto out_cols = out_local % kFreqAttnD;
    auto out_valid = out_rows < kFreqAttnN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kFreqAttnD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows, int VCols, bool ConstNegInf = false>
__tile_global__ void freq_attention60_cutile_padded_out60_vsplit_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int kVCols = VCols;
    static_assert(kFreqAttnD % kVCols == 0);
    using ScoreTile = ct::tile<float, ct::shape<QRows, kFreqAttnPadN>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kVCols>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, kFreqAttnPadN>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kVCols>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kFreqAttnPadN * kFreqAttnD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kFreqAttnPadN * kFreqAttnD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kFreqAttnPadN * kFreqAttnD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kFreqAttnN * kFreqAttnD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kFreqAttnPadN, kFreqAttnD>{}},
        ct::shape<QRows, kFreqAttnD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kFreqAttnD, kFreqAttnPadN>{}, ct::layout_left{}},
        ct::shape<kFreqAttnD, kFreqAttnPadN>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kFreqAttnPadN, kFreqAttnD>{}},
        ct::shape<kFreqAttnPadN, kVCols>{}
    };

    auto scores = ct::mma(q_view.load(q_block, 0), k_t_view.load(0, 0),
                          ct::full<ScoreTile>(0.0f));
    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows +
                      score_local / kFreqAttnPadN;
    auto score_cols = score_local % kFreqAttnPadN;
    auto score_valid = (score_rows < kFreqAttnN) && (score_cols < kFreqAttnN);
    auto neg_inf = [&]() {
        if constexpr (ConstNegInf) {
            return ct::full<ScoreTile>(-3.402823466e38f);
        } else {
            return scores * 0.0f - 3.402823466e38f;
        }
    }();
    scores = ct::select(score_valid, scores * scale, neg_inf);

    auto row_max = ct::reduce_max<1>(scores);
    auto probs_f32 = ct::select(score_valid, ct::exp(scores - row_max),
                                scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto denom = ct::sum<1>(probs_f32);

#pragma unroll
    for (int d_part = 0; d_part < kFreqAttnD / kVCols; ++d_part) {
        auto out_acc = ct::mma(probs_bf16, v_view.load(0, d_part),
                               ct::full<OutTile>(0.0f));
        out_acc = out_acc / denom;

        I64OutTile out_local = ct::iota<I64OutTile>();
        auto out_rows = static_cast<long long>(q_block) * QRows +
                        out_local / kVCols;
        auto out_cols = d_part * kVCols + out_local % kVCols;
        auto out_valid = out_rows < kFreqAttnN;
        auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
        ct::store_masked(out_batch + safe_rows * kFreqAttnD + out_cols,
                         ct::element_cast<__nv_bfloat16>(out_acc),
                         out_valid);
    }
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

__tile_global__ void add_bias_to_bf16_kernel(const float* __restrict__ out,
                                             const float* __restrict__ bias,
                                             __nv_bfloat16* __restrict__ dst,
                                             long long total,
                                             int out_features) {
    out = ct::assume_aligned(out, 16_ic);
    bias = ct::assume_aligned(bias, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(out + idx, in_bounds) +
                  ct::load_masked(bias + (idx % out_features), in_bounds);
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
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

__tile_global__ void add_bias_sigmoid_to_bf16_kernel(const float* __restrict__ out,
                                                     const float* __restrict__ bias,
                                                     __nv_bfloat16* __restrict__ dst,
                                                     long long total,
                                                     int out_features) {
    out = ct::assume_aligned(out, 16_ic);
    bias = ct::assume_aligned(bias, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(out + idx, in_bounds) +
                  ct::load_masked(bias + (idx % out_features), in_bounds);
    auto sigmoid = 1.0f / (1.0f + ct::exp(-values));
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(sigmoid), in_bounds);
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

__tile_global__ void add_bias_gelu_to_bf16_kernel(const float* __restrict__ out,
                                                  const float* __restrict__ bias,
                                                  __nv_bfloat16* __restrict__ dst,
                                                  long long total,
                                                  int out_features) {
    out = ct::assume_aligned(out, 16_ic);
    bias = ct::assume_aligned(bias, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

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
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(gelu), in_bounds);
}

template <typename BandsT = float>
__tile_global__ void apply_mask_and_scatter_kernel(const float* __restrict__ stft,
                                                   const float* __restrict__ mask0,
                                                   const float* __restrict__ mask1,
                                                   const int64_t* __restrict__ freq_indices,
                                                   const BandsT* __restrict__ bands_per_freq,
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
    auto denom = ct::element_cast<float>(
        ct::load_masked(bands_per_freq + freq, in_bounds));
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

template <typename BandsT = float>
__tile_global__ void apply_mask_and_scatter_bf16_kernel(
    const float* __restrict__ stft,
    const __nv_bfloat16* __restrict__ mask0,
    const __nv_bfloat16* __restrict__ mask1,
    const int64_t* __restrict__ freq_indices,
    const BandsT* __restrict__ bands_per_freq,
    float* __restrict__ out,
    long long total,
    int num_stems,
    int total_band_freqs,
    int frames,
    int total_freq,
    int audio_channels,
    int freq_bins,
    bool full_bf16) {
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
    auto denom = ct::element_cast<float>(
        ct::load_masked(bands_per_freq + freq, in_bounds));
    auto eps = denom * 0.0f + 1.0e-8f;
    denom = ct::select(denom > eps, denom, eps);

    auto stft_base = ((batch_idx * total_freq + freq) * frames + t) * 2;
    auto mask_base = (batch_idx * frames + t) * (2LL * total_band_freqs) + band_f * 2;

    auto stft_r = ct::load_masked(stft + stft_base, in_bounds);
    auto stft_i = ct::load_masked(stft + stft_base + 1, in_bounds);
    auto stem0 = in_bounds && (stem == 0);
    auto stem1 = in_bounds && (stem == 1);
    F32Tile mask0_r = ct::element_cast<float>(ct::load_masked(mask0 + mask_base, stem0));
    F32Tile mask0_i = ct::element_cast<float>(ct::load_masked(mask0 + mask_base + 1, stem0));
    F32Tile mask1_r = ct::element_cast<float>(ct::load_masked(mask1 + mask_base, stem1));
    F32Tile mask1_i = ct::element_cast<float>(ct::load_masked(mask1 + mask_base + 1, stem1));
    auto mask_r = ct::select(stem == 0, mask0_r, mask1_r);
    auto mask_i = ct::select(stem == 0, mask0_i, mask1_i);
    mask_r = ct::select(full_bf16, bf16_round(mask_r), mask_r);
    mask_i = ct::select(full_bf16, bf16_round(mask_i), mask_i);

    auto value_r = (stft_r * mask_r - stft_i * mask_i) / denom;
    auto value_i = (stft_r * mask_i + stft_i * mask_r) / denom;
    value_r = ct::select(full_bf16, bf16_round(value_r), value_r);
    value_i = ct::select(full_bf16, bf16_round(value_i), value_i);
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

__tile_global__ void tanh_to_bf16_kernel(const float* __restrict__ x,
                                         __nv_bfloat16* __restrict__ out,
                                         long long total) {
    x = ct::assume_aligned(x, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = tanh(ct::load_masked(x + idx, in_bounds));
    ct::store_masked(out + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

__tile_global__ void tanh_bf16_to_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                              __nv_bfloat16* __restrict__ out,
                                              long long total,
                                              bool full_bf16) {
    x = ct::assume_aligned(x, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    F32Tile values = ct::element_cast<float>(ct::load_masked(x + idx, in_bounds));
    values = ct::select(full_bf16, bf16_round(values), values);
    values = tanh(values);
    values = ct::select(full_bf16, bf16_round(values), values);
    ct::store_masked(out + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
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

__tile_global__ void glu_last_dim_to_bf16_kernel(const float* __restrict__ x,
                                                 __nv_bfloat16* __restrict__ out,
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
    auto values = first * gate;
    ct::store_masked(out + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

__tile_global__ void glu_last_dim_bf16_to_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                                      __nv_bfloat16* __restrict__ out,
                                                      long long total,
                                                      int half_dim,
                                                      bool full_bf16) {
    x = ct::assume_aligned(x, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    auto d = idx % half_dim;
    auto row = idx / half_dim;
    auto base = row * (2LL * half_dim);
    F32Tile first = ct::element_cast<float>(ct::load_masked(x + base + d, in_bounds));
    F32Tile second = ct::element_cast<float>(ct::load_masked(x + base + half_dim + d, in_bounds));
    first = ct::select(full_bf16, bf16_round(first), first);
    second = ct::select(full_bf16, bf16_round(second), second);
    auto gate = 1.0f / (1.0f + exp(-second));
    gate = ct::select(full_bf16, bf16_round(gate), gate);
    auto values = first * gate;
    values = ct::select(full_bf16, bf16_round(values), values);
    ct::store_masked(out + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

template <typename T>
const T* tensor_data_as(const Tensor& t) {
    return static_cast<const T*>(t.data_ptr());
}

template <int TileSize, typename TrigT>
void launch_rotary_time_split_contig_inplace_typed(Tensor& q,
                                                   Tensor& k,
                                                   const Tensor& cos_work,
                                                   const Tensor& sin_work,
                                                   long long total,
                                                   bool full_bf16) {
    rotary_time_split_contig_inplace_kernel<TileSize, TrigT>
        <<<(int)ceildiv(total, TileSize), 1>>>(
            q.data_bf16(), k.data_bf16(),
            tensor_data_as<TrigT>(cos_work), tensor_data_as<TrigT>(sin_work),
            total, full_bf16);
}

template <typename TrigT>
void launch_rotary_time_split_contig_inplace_by_tile(Tensor& q,
                                                     Tensor& k,
                                                     const Tensor& cos_work,
                                                     const Tensor& sin_work,
                                                     long long total,
                                                     bool full_bf16) {
    int rotary_tile = time_rotary_tile_size();
    if (rotary_tile == 128) {
        launch_rotary_time_split_contig_inplace_typed<128, TrigT>(
            q, k, cos_work, sin_work, total, full_bf16);
    } else if (rotary_tile == 512) {
        launch_rotary_time_split_contig_inplace_typed<512, TrigT>(
            q, k, cos_work, sin_work, total, full_bf16);
    } else if (rotary_tile == 1024) {
        launch_rotary_time_split_contig_inplace_typed<1024, TrigT>(
            q, k, cos_work, sin_work, total, full_bf16);
    } else {
        launch_rotary_time_split_contig_inplace_typed<kTile, TrigT>(
            q, k, cos_work, sin_work, total, full_bf16);
    }
}

}  // namespace

void set_time_attention_context_chunk(int chunk_index) {
    g_time_attention_context_chunk = chunk_index;
}

void set_time_attention_context_depth(int depth_index) {
    g_time_attention_context_depth = depth_index;
}

bool time_attention_stats_enabled_for_current_context() {
    return time_attention_stats_enabled_for_current_context_impl();
}

bool residual_bf16_enabled() {
    return residual_bf16_enabled_impl();
}

bool bias_bf16_enabled() {
    return bias_bf16_enabled_impl();
}

bool norm_gamma_bf16_enabled() {
    return norm_gamma_bf16_enabled_impl();
}

bool bands_per_freq_bf16_enabled() {
    return bands_per_freq_bf16_enabled_impl();
}

bool rotary_freqs_bf16_enabled() {
    return rotary_freqs_bf16_enabled_impl();
}

bool try_linear_glu_last_dim_bf16_output(const Tensor& x,
                                         const Tensor& weight,
                                         const Tensor& bias,
                                         Tensor& out) {
    return try_linear_glu_last_dim_bf16_output_impl(x, weight, bias, out);
}

bool linear_bkn_long_enabled() {
    return linear_bkn_long_path_enabled();
}

bool linear_bkn_ffn_long_enabled() {
    return linear_bkn_ffn_long_path_enabled();
}

bool time_qkv_split_contig_path_enabled() {
    return time_qkv_split_contig_path_enabled_impl();
}

bool time_qkv_fused_rotary_producer_enabled() {
    return time_qkv_fused_rotary_producer_enabled_impl();
}

bool time_qkv_pair_rotary_producer_enabled() {
    return time_qkv_pair_rotary_producer_enabled_impl();
}

bool time_q_rotary_in_attention_enabled() {
    return time_q_rotary_in_attention_enabled_impl();
}

void split_qkv_heads_rotary(const Tensor& qkv, int heads, int dim_head,
                            const Tensor& cos_freqs, const Tensor& sin_freqs,
                            Tensor& q, Tensor& k, Tensor& v) {
    if ((dim_head % 2) != 0) {
        throw std::runtime_error("mbr_tile::split_qkv_heads_rotary: dim_head must be even");
    }

    int B = (int)qkv.size(0);
    int N = (int)qkv.size(1);
    bool output_bf16 = g_quantize_bf16 && dim_head <= 128;
    bool freq_pad64 = output_bf16 &&
                      qkv.dtype() == DType::BFloat16 &&
                      freq_attention60_cutile_padded_enabled() &&
                      N == kFreqAttnN &&
                      dim_head == kFreqAttnD;
    int out_N = freq_pad64 ? kFreqAttnPadN : N;
    DType out_dtype = output_bf16 ? DType::BFloat16 : DType::Float32;
    q = Tensor::empty({(int64_t)B, (int64_t)heads, (int64_t)out_N, (int64_t)dim_head}, out_dtype);
    k = Tensor::empty({(int64_t)B, (int64_t)heads, (int64_t)out_N, (int64_t)dim_head}, out_dtype);
    v = Tensor::empty({(int64_t)B, (int64_t)heads, (int64_t)out_N, (int64_t)dim_head}, out_dtype);

    long long total = (long long)B * heads * N * (dim_head / 2);
    if (output_bf16) {
        if (qkv.dtype() == DType::BFloat16) {
            Tensor qkv_work = qkv.contiguous();
            if (freq_pad64) {
                bool trig_bf16 = cos_freqs.dtype() == DType::BFloat16 &&
                                  sin_freqs.dtype() == DType::BFloat16;
                if (freq_split_skip_qk_pad_zero_enabled()) {
                    long long total_nonpad =
                        (long long)B * heads * kFreqAttnN * (kFreqAttnD / 2);
                    if (trig_bf16) {
                        Tensor cos_work = cos_freqs.contiguous();
                        Tensor sin_work = sin_freqs.contiguous();
                        split_qkv_heads_rotary_qkv_bf16_freq60_to_pad64_kernel<__nv_bfloat16>
                            <<<(int)ceildiv(total_nonpad, kTile), 1>>>(
                                qkv_work.data_bf16(),
                                cos_work.data_bf16(), sin_work.data_bf16(),
                                q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                total_nonpad, heads, full_bf16_arith_enabled());
                    } else {
                        Tensor cos_work = (cos_freqs.dtype() == DType::Float32)
                            ? cos_freqs.contiguous()
                            : cos_freqs.to_f32().contiguous();
                        Tensor sin_work = (sin_freqs.dtype() == DType::Float32)
                            ? sin_freqs.contiguous()
                            : sin_freqs.to_f32().contiguous();
                        split_qkv_heads_rotary_qkv_bf16_freq60_to_pad64_kernel<float>
                            <<<(int)ceildiv(total_nonpad, kTile), 1>>>(
                                qkv_work.data_bf16(),
                                cos_work.data_f32(), sin_work.data_f32(),
                                q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                total_nonpad, heads, full_bf16_arith_enabled());
                    }
                    if (!freq_split_skip_v_pad_zero_enabled()) {
                        long long v_pad_total =
                            (long long)B * heads *
                            (kFreqAttnPadN - kFreqAttnN) * kFreqAttnD;
                        freq60_v_pad_rows_zero_kernel<<<(int)ceildiv(v_pad_total, kTile), 1>>>(
                            v.data_bf16(), v_pad_total, heads);
                    }
                } else {
                    long long total_pad =
                        (long long)B * heads * kFreqAttnPadN * (kFreqAttnD / 2);
                    if (trig_bf16) {
                        Tensor cos_work = cos_freqs.contiguous();
                        Tensor sin_work = sin_freqs.contiguous();
                        split_qkv_heads_rotary_qkv_bf16_freq60_pad64_kernel<__nv_bfloat16>
                            <<<(int)ceildiv(total_pad, kTile), 1>>>(
                                qkv_work.data_bf16(),
                                cos_work.data_bf16(), sin_work.data_bf16(),
                                q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                total_pad, heads, full_bf16_arith_enabled());
                    } else {
                        Tensor cos_work = (cos_freqs.dtype() == DType::Float32)
                            ? cos_freqs.contiguous()
                            : cos_freqs.to_f32().contiguous();
                        Tensor sin_work = (sin_freqs.dtype() == DType::Float32)
                            ? sin_freqs.contiguous()
                            : sin_freqs.to_f32().contiguous();
                        split_qkv_heads_rotary_qkv_bf16_freq60_pad64_kernel<float>
                            <<<(int)ceildiv(total_pad, kTile), 1>>>(
                                qkv_work.data_bf16(),
                                cos_work.data_f32(), sin_work.data_f32(),
                                q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                total_pad, heads, full_bf16_arith_enabled());
                    }
                }
            } else if (split_qkv_time_cutile_fixed_enabled() &&
                       N == kTimeAttnN &&
                       heads == kQkvFusedHeads &&
                       dim_head == kTimeAttnD) {
                bool trig_bf16 = cos_freqs.dtype() == DType::BFloat16 &&
                                  sin_freqs.dtype() == DType::BFloat16;
                if (trig_bf16) {
                    Tensor cos_work = cos_freqs.contiguous();
                    Tensor sin_work = sin_freqs.contiguous();
                    split_qkv_heads_rotary_qkv_bf16_time1301_d64_kernel<__nv_bfloat16>
                        <<<(int)ceildiv(total, kTile), 1>>>(
                            qkv_work.data_bf16(), cos_work.data_bf16(), sin_work.data_bf16(),
                            q.data_bf16(), k.data_bf16(), v.data_bf16(),
                            total, full_bf16_arith_enabled());
                } else {
                    Tensor cos_work = (cos_freqs.dtype() == DType::Float32) ? cos_freqs.contiguous() : cos_freqs.to_f32().contiguous();
                    Tensor sin_work = (sin_freqs.dtype() == DType::Float32) ? sin_freqs.contiguous() : sin_freqs.to_f32().contiguous();
                    split_qkv_heads_rotary_qkv_bf16_time1301_d64_kernel<float>
                        <<<(int)ceildiv(total, kTile), 1>>>(
                            qkv_work.data_bf16(), cos_work.data_f32(), sin_work.data_f32(),
                            q.data_bf16(), k.data_bf16(), v.data_bf16(),
                            total, full_bf16_arith_enabled());
                }
            } else {
                bool trig_bf16 = cos_freqs.dtype() == DType::BFloat16 &&
                                  sin_freqs.dtype() == DType::BFloat16;
                if (trig_bf16) {
                    Tensor cos_work = cos_freqs.contiguous();
                    Tensor sin_work = sin_freqs.contiguous();
                    split_qkv_heads_rotary_qkv_bf16_kernel<__nv_bfloat16>
                        <<<(int)ceildiv(total, kTile), 1>>>(
                            qkv_work.data_bf16(), cos_work.data_bf16(), sin_work.data_bf16(),
                            q.data_bf16(), k.data_bf16(), v.data_bf16(),
                            total, heads, N, dim_head, full_bf16_arith_enabled());
                } else {
                    Tensor cos_work = (cos_freqs.dtype() == DType::Float32)
                        ? cos_freqs.contiguous()
                        : cos_freqs.to_f32().contiguous();
                    Tensor sin_work = (sin_freqs.dtype() == DType::Float32)
                        ? sin_freqs.contiguous()
                        : sin_freqs.to_f32().contiguous();
                    split_qkv_heads_rotary_qkv_bf16_kernel<float>
                        <<<(int)ceildiv(total, kTile), 1>>>(
                            qkv_work.data_bf16(), cos_work.data_f32(), sin_work.data_f32(),
                            q.data_bf16(), k.data_bf16(), v.data_bf16(),
                            total, heads, N, dim_head, full_bf16_arith_enabled());
                }
            }
        } else {
            Tensor cos_work = (cos_freqs.dtype() == DType::Float32) ? cos_freqs.contiguous() : cos_freqs.to_f32().contiguous();
            Tensor sin_work = (sin_freqs.dtype() == DType::Float32) ? sin_freqs.contiguous() : sin_freqs.to_f32().contiguous();
            Tensor qkv_work = (qkv.dtype() == DType::Float32) ? qkv.contiguous() : qkv.to_f32().contiguous();
            split_qkv_heads_rotary_bf16_kernel<<<(int)ceildiv(total, kTile), 1>>>(
                qkv_work.data_f32(), cos_work.data_f32(), sin_work.data_f32(),
                q.data_bf16(), k.data_bf16(), v.data_bf16(),
                total, heads, N, dim_head);
        }
    } else {
        Tensor cos_work = (cos_freqs.dtype() == DType::Float32) ? cos_freqs.contiguous() : cos_freqs.to_f32().contiguous();
        Tensor sin_work = (sin_freqs.dtype() == DType::Float32) ? sin_freqs.contiguous() : sin_freqs.to_f32().contiguous();
        Tensor qkv_work = (qkv.dtype() == DType::Float32) ? qkv.contiguous() : qkv.to_f32().contiguous();
        split_qkv_heads_rotary_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            qkv_work.data_f32(), cos_work.data_f32(), sin_work.data_f32(),
            q.data_f32(), k.data_f32(), v.data_f32(),
            total, heads, N, dim_head);
    }
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
    bool out_bf16 = gather_bf16_output_enabled();
    Tensor out = Tensor::empty({batch, frames, total_band_freqs * 2},
                               out_bf16 ? DType::BFloat16 : DType::Float32);

    long long total = out.numel();
    if (out_bf16) {
        gather_freqs_fold_complex_to_bf16_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            stft_work.data_f32(), freq_indices.data_i64(), out.data_bf16(),
            total, (int)total_freq, (int)total_band_freqs, (int)frames);
    } else {
        gather_freqs_fold_complex_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            stft_work.data_f32(), freq_indices.data_i64(), out.data_f32(),
            total, (int)total_freq, (int)total_band_freqs, (int)frames);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor apply_gates_and_merge_heads(const Tensor& attn, const Tensor& gates,
                                   int heads, int dim_head) {
    int B = (int)attn.size(0);
    int N = (int)attn.size(2);
    bool attn_bf16 = attn.dtype() == DType::BFloat16;
    bool gate_bf16 = gates.dtype() == DType::BFloat16;
    bool merge_bf16 = gate_merge_bf16_enabled();
    DType out_dtype = merge_bf16 ? DType::BFloat16 : DType::Float32;
    Tensor merged = Tensor::empty({(int64_t)B, (int64_t)N, (int64_t)heads * dim_head}, out_dtype);

    long long total = (long long)B * heads * N * dim_head;
    if (attn_bf16 && gate_bf16 && merge_bf16) {
        if (gate_merge_token_d64_enabled() && heads == 8 && dim_head == 64) {
            apply_gates_and_merge_heads_bf16_token_d64_cutile_kernel<<<B * N, 1>>>(
                attn.data_bf16(), gates.data_bf16(), merged.data_bf16(),
                B * N, N, full_bf16_arith_enabled());
        } else {
            apply_gates_and_merge_heads_typed_kernel<<<(int)ceildiv(total, kTile), 1>>>(
                attn.data_bf16(), gates.data_bf16(), merged.data_bf16(),
                total, heads, N, dim_head, full_bf16_arith_enabled());
        }
    } else if (attn_bf16 && gate_bf16) {
        apply_gates_and_merge_heads_typed_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            attn.data_bf16(), gates.data_bf16(), merged.data_f32(),
            total, heads, N, dim_head, full_bf16_arith_enabled());
    } else if (attn_bf16 && merge_bf16) {
        apply_gates_and_merge_heads_typed_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            attn.data_bf16(), gates.data_f32(), merged.data_bf16(),
            total, heads, N, dim_head, full_bf16_arith_enabled());
    } else if (attn_bf16) {
        apply_gates_and_merge_heads_typed_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            attn.data_bf16(), gates.data_f32(), merged.data_f32(),
            total, heads, N, dim_head, full_bf16_arith_enabled());
    } else if (merge_bf16 && gate_bf16) {
        apply_gates_and_merge_heads_bf16_gate_to_bf16_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            attn.data_f32(), gates.data_bf16(), merged.data_bf16(),
            total, heads, N, dim_head);
    } else if (merge_bf16) {
        apply_gates_and_merge_heads_to_bf16_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            attn.data_f32(), gates.data_f32(), merged.data_bf16(),
            total, heads, N, dim_head);
    } else if (gate_bf16) {
        apply_gates_and_merge_heads_gate_bf16_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            attn.data_f32(), gates.data_bf16(), merged.data_f32(),
            total, heads, N, dim_head);
    } else {
        apply_gates_and_merge_heads_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            attn.data_f32(), gates.data_f32(), merged.data_f32(),
            total, heads, N, dim_head);
    }
    CUDA_CHECK(cudaGetLastError());
    return merged;
}

Tensor rms_norm(const Tensor& x, const Tensor& gamma, float scale) {
    Tensor gf = (gamma.dtype() == DType::Float32 || gamma.dtype() == DType::BFloat16)
        ? gamma.contiguous()
        : gamma.to_f32().contiguous();
    int dim = (int)x.size(x.ndim() - 1);
    if (dim > kRmsTile) {
        throw std::runtime_error("mbr_tile::rms_norm: last dimension exceeds tile size");
    }

    if (rms_norm_bf16_enabled()) {
        Tensor xc = x.contiguous();
        Tensor out = Tensor::empty(xc.shape(), DType::BFloat16);
        long long rows = xc.numel() / dim;
        if (dim == kRmsD256 && rms_norm_d256_cutile_fixed_enabled()) {
            bool full_bf16 = full_bf16_arith_enabled();
            int rows_per_cta = rms_norm_d256_rows_per_cta();
            if (xc.dtype() == DType::BFloat16) {
                if (gf.dtype() == DType::BFloat16) {
                    launch_rms_norm_d256_to_bf16_typed(
                        xc.data_bf16(), gf.data_bf16(), out.data_bf16(),
                        rows, scale, full_bf16, rows_per_cta);
                } else {
                    launch_rms_norm_d256_to_bf16_typed(
                        xc.data_bf16(), gf.data_f32(), out.data_bf16(),
                        rows, scale, full_bf16, rows_per_cta);
                }
            } else {
                Tensor xf = (xc.dtype() == DType::Float32) ? xc : xc.to_f32().contiguous();
                if (gf.dtype() == DType::BFloat16) {
                    launch_rms_norm_d256_to_bf16_typed(
                        xf.data_f32(), gf.data_bf16(), out.data_bf16(),
                        rows, scale, full_bf16, rows_per_cta);
                } else {
                    launch_rms_norm_d256_to_bf16_typed(
                        xf.data_f32(), gf.data_f32(), out.data_bf16(),
                        rows, scale, full_bf16, rows_per_cta);
                }
            }
            CUDA_CHECK(cudaGetLastError());
            return out;
        }
        if (xc.dtype() == DType::BFloat16) {
            if (gf.dtype() == DType::BFloat16) {
                launch_rms_norm_to_bf16_typed(
                    xc.data_bf16(), gf.data_bf16(), out.data_bf16(),
                    rows, dim, scale, full_bf16_arith_enabled());
            } else {
                launch_rms_norm_to_bf16_typed(
                    xc.data_bf16(), gf.data_f32(), out.data_bf16(),
                    rows, dim, scale, full_bf16_arith_enabled());
            }
        } else {
            Tensor xf = (xc.dtype() == DType::Float32) ? xc : xc.to_f32().contiguous();
            if (gf.dtype() == DType::BFloat16) {
                launch_rms_norm_to_bf16_typed(
                    xf.data_f32(), gf.data_bf16(), out.data_bf16(),
                    rows, dim, scale, full_bf16_arith_enabled());
            } else {
                launch_rms_norm_to_bf16_typed(
                    xf.data_f32(), gf.data_f32(), out.data_bf16(),
                    rows, dim, scale, full_bf16_arith_enabled());
            }
        }
        CUDA_CHECK(cudaGetLastError());
        return out;
    }

    Tensor xf = (x.dtype() == DType::Float32) ? x.contiguous() : x.to_f32().contiguous();
    if (gf.dtype() != DType::Float32) {
        gf = gf.to_f32().contiguous();
    }
    Tensor out = Tensor::empty(xf.shape(), DType::Float32);
    long long rows = xf.numel() / dim;
    rms_norm_kernel<<<(int)rows, 1>>>(xf.data_f32(), gf.data_f32(), out.data_f32(), dim, scale);
    CUDA_CHECK(cudaGetLastError());
    return (x.dtype() == DType::Float16) ? out.to_f16() : out;
}

bool try_residual_add_rms_norm(const Tensor& residual,
                               const Tensor& update,
                               const Tensor& gamma,
                               float scale,
                               Tensor& residual_out,
                               Tensor& normed) {
    if (!residual_rms_norm_fused_enabled()) return false;
    if (!rms_norm_bf16_enabled() || !rms_norm_d256_cutile_fixed_enabled()) return false;
    if (rms_norm_d256_rows_per_cta() != 1) return false;
    if (residual.dtype() != DType::BFloat16 || update.dtype() != DType::BFloat16) {
        return false;
    }
    if (gamma.dtype() != DType::Float32 && gamma.dtype() != DType::BFloat16) {
        return false;
    }
    if (residual.shape() != update.shape() || residual.ndim() == 0 ||
        residual.size(residual.ndim() - 1) != kRmsD256 ||
        residual.numel() % kRmsD256 != 0) {
        return false;
    }
    if (!residual.is_contiguous() || !update.is_contiguous()) {
        return false;
    }

    Tensor gf = gamma.contiguous();
    long long rows = residual.numel() / kRmsD256;
    int rows_per_cta = residual_rms_norm_d256_rows_per_cta();
    residual_out = Tensor::empty(residual.shape(), DType::BFloat16);
    normed = Tensor::empty(residual.shape(), DType::BFloat16);
    if (gf.dtype() == DType::BFloat16) {
        launch_residual_add_rms_norm_d256_to_bf16_typed(
            residual.data_bf16(), update.data_bf16(), gf.data_bf16(),
            residual_out.data_bf16(), normed.data_bf16(), rows, scale,
            full_bf16_arith_enabled(), rows_per_cta);
    } else {
        launch_residual_add_rms_norm_d256_to_bf16_typed(
            residual.data_bf16(), update.data_bf16(), gf.data_f32(),
            residual_out.data_bf16(), normed.data_bf16(), rows, scale,
            full_bf16_arith_enabled(), rows_per_cta);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

Tensor scaled_dot_product_attention_time_split_contig(const Tensor& q,
                                                      const Tensor& k,
                                                      const Tensor& v,
                                                      float scale) {
    if (q.dtype() != DType::BFloat16 || k.dtype() != DType::BFloat16 ||
        v.dtype() != DType::BFloat16) {
        throw std::runtime_error(
            "mbr_tile::scaled_dot_product_attention_time_split_contig: expected BF16 q/k/v");
    }
    if (q.ndim() != 4 || q.size(1) != kTimeAttnN ||
        q.size(2) != kQkvFusedHeads || q.size(3) != kTimeAttnD ||
        k.shape() != q.shape() || v.shape() != q.shape()) {
        throw std::runtime_error(
            "mbr_tile::scaled_dot_product_attention_time_split_contig: expected q/k/v [B,1301,8,64]");
    }
    if (scale == 0.0f) {
        scale = kTimeAttnScale;
    }
    if (std::fabs(scale - kTimeAttnScale) >= 1.0e-7f) {
        throw std::runtime_error(
            "mbr_tile::scaled_dot_product_attention_time_split_contig: unsupported scale");
    }

    int64_t batches = q.size(0);
    int64_t bh = batches * kQkvFusedHeads;
    if (batches <= 0 || bh > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "mbr_tile::scaled_dot_product_attention_time_split_contig: invalid batch count");
    }
    Tensor qb = q.is_contiguous() ? q : q.contiguous();
    Tensor kb = k.is_contiguous() ? k : k.contiguous();
    Tensor vb = v.is_contiguous() ? v : v.contiguous();
    Tensor out = Tensor::empty({bh, kTimeAttnN, kTimeAttnD}, DType::BFloat16);

    bool use_exp2 = time_attention_cutile_exp2_enabled();
    bool skip_keytail = time_attention_cutile_skip_keytail_enabled();
    bool approx_softmax = time_attention_approx_softmax_enabled_for_call();
    launch_time_attention1301_split_contig_main_cutile(
        qb, kb, vb, out, scale, use_exp2, skip_keytail, approx_softmax);
    launch_time_attention1301_split_contig_tail_cutile(
        qb, kb, vb, out, scale,
        time_attention_cutile_split_tail_q32_enabled(),
        use_exp2);
    return out.reshape({batches, kQkvFusedHeads, kTimeAttnN, kTimeAttnD});
}

Tensor scaled_dot_product_attention_time_split_contig_q_rotary(const Tensor& q,
                                                               const Tensor& k,
                                                               const Tensor& v,
                                                               const Tensor& cos_freqs,
                                                               const Tensor& sin_freqs,
                                                               float scale) {
    if (q.dtype() != DType::BFloat16 || k.dtype() != DType::BFloat16 ||
        v.dtype() != DType::BFloat16) {
        throw std::runtime_error(
            "mbr_tile::scaled_dot_product_attention_time_split_contig_q_rotary: expected BF16 q/k/v");
    }
    if (q.ndim() != 4 || q.size(1) != kTimeAttnN ||
        q.size(2) != kQkvFusedHeads || q.size(3) != kTimeAttnD ||
        k.shape() != q.shape() || v.shape() != q.shape()) {
        throw std::runtime_error(
            "mbr_tile::scaled_dot_product_attention_time_split_contig_q_rotary: expected q/k/v [B,1301,8,64]");
    }
    if (cos_freqs.ndim() != 2 || sin_freqs.ndim() != 2 ||
        cos_freqs.size(0) != kTimeAttnN || sin_freqs.size(0) != kTimeAttnN ||
        cos_freqs.size(1) != kTimeAttnD / 2 || sin_freqs.size(1) != kTimeAttnD / 2) {
        throw std::runtime_error(
            "mbr_tile::scaled_dot_product_attention_time_split_contig_q_rotary: expected cos/sin [1301,32]");
    }
    if (scale == 0.0f) {
        scale = kTimeAttnScale;
    }
    if (std::fabs(scale - kTimeAttnScale) >= 1.0e-7f) {
        throw std::runtime_error(
            "mbr_tile::scaled_dot_product_attention_time_split_contig_q_rotary: unsupported scale");
    }

    int64_t batches = q.size(0);
    int64_t bh = batches * kQkvFusedHeads;
    if (batches <= 0 || bh > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "mbr_tile::scaled_dot_product_attention_time_split_contig_q_rotary: invalid batch count");
    }
    Tensor qb = q.is_contiguous() ? q : q.contiguous();
    Tensor kb = k.is_contiguous() ? k : k.contiguous();
    Tensor vb = v.is_contiguous() ? v : v.contiguous();
    bool trig_bf16 = cos_freqs.dtype() == DType::BFloat16 &&
                     sin_freqs.dtype() == DType::BFloat16;
    Tensor cos_work = trig_bf16 ? cos_freqs.contiguous()
                                : ((cos_freqs.dtype() == DType::Float32)
                                       ? cos_freqs.contiguous()
                                       : cos_freqs.to_f32().contiguous());
    Tensor sin_work = trig_bf16 ? sin_freqs.contiguous()
                                : ((sin_freqs.dtype() == DType::Float32)
                                       ? sin_freqs.contiguous()
                                       : sin_freqs.to_f32().contiguous());
    Tensor out = Tensor::empty({bh, kTimeAttnN, kTimeAttnD}, DType::BFloat16);

    bool use_exp2 = time_attention_cutile_exp2_enabled();
    bool full_bf16 = full_bf16_arith_enabled();
    bool use_q32 = time_q_rotary_in_attention_q32_enabled_impl();
    bool skip_keytail = time_attention_cutile_skip_keytail_enabled();
    launch_time_attention1301_split_contig_qrot_main_cutile(
        qb, kb, vb, cos_work, sin_work, out, scale, full_bf16, use_q32, use_exp2,
        skip_keytail);
    launch_time_attention1301_split_contig_qrot_tail_cutile(
        qb, kb, vb, cos_work, sin_work, out, scale, full_bf16,
        time_attention_cutile_split_tail_q32_enabled(),
        use_exp2);
    return out.reshape({batches, kQkvFusedHeads, kTimeAttnN, kTimeAttnD});
}

Tensor scaled_dot_product_attention(const Tensor& q,
                                    const Tensor& k,
                                    const Tensor& v,
                                    float scale) {
    int64_t B = q.size(0);
    int64_t H = q.size(1);
    int64_t N = q.size(2);
    int64_t D = q.size(3);
    int64_t N_k = k.size(2);
    if (N_k > kSoftmaxTile) {
        throw std::runtime_error("mbr_tile::scaled_dot_product_attention: sequence exceeds tile size");
    }
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt((float)D);
    }

    bool use_fp16_value_gemm = g_quantize_fp16 && D <= 128;
    bool use_bf16_value_gemm = g_quantize_bf16 && D <= 128;

    if (freq_attention60_cutile_padded_enabled() &&
        q.dtype() == DType::BFloat16 && k.dtype() == DType::BFloat16 &&
        v.dtype() == DType::BFloat16 &&
        N == kFreqAttnPadN && N_k == kFreqAttnPadN && D == kFreqAttnD) {
        int64_t BH = B * H;
        if (BH <= 0 || BH > std::numeric_limits<int>::max()) {
            throw std::runtime_error("mbr_tile::scaled_dot_product_attention: invalid padded freq-attention batch");
        }
        Tensor qb = q.contiguous();
        Tensor kb = k.contiguous();
        Tensor vb = v.contiguous();
        Tensor out = Tensor::empty({BH, kFreqAttnN, kFreqAttnD}, DType::BFloat16);
        dim3 grid((unsigned int)ceildiv(kFreqAttnN, kFreqAttnCutileQRows),
                  (unsigned int)BH);
        bool use_v32 = freq_attention60_v32_enabled();
        if (use_v32 && freq_split_skip_qk_pad_zero_enabled()) {
            freq_attention60_cutile_padded_out60_vsplit_kernel<
                kFreqAttnCutileQRows, 32, true>
                <<<grid, 1>>>(qb.data_bf16(), kb.data_bf16(), vb.data_bf16(),
                              out.data_bf16(), scale);
        } else if (use_v32) {
            freq_attention60_cutile_padded_out60_vsplit_kernel<
                kFreqAttnCutileQRows, 32>
                <<<grid, 1>>>(qb.data_bf16(), kb.data_bf16(), vb.data_bf16(),
                              out.data_bf16(), scale);
        } else if (freq_split_skip_qk_pad_zero_enabled()) {
            freq_attention60_cutile_padded_out60_kernel<kFreqAttnCutileQRows, true>
                <<<grid, 1>>>(qb.data_bf16(), kb.data_bf16(), vb.data_bf16(),
                              out.data_bf16(), scale);
        } else {
            freq_attention60_cutile_padded_out60_kernel<kFreqAttnCutileQRows>
                <<<grid, 1>>>(qb.data_bf16(), kb.data_bf16(), vb.data_bf16(),
                              out.data_bf16(), scale);
        }
        CUDA_CHECK(cudaGetLastError());
        Tensor result = out.reshape({B, H, kFreqAttnN, kFreqAttnD});
        return (q.dtype() == DType::Float16) ? result.to_f16() : result;
    }

    if (time_attention_fused_fragscale_enabled() &&
        use_bf16_value_gemm &&
        N == kTimeAttnN && N_k == kTimeAttnN && D == kTimeAttnD &&
        std::fabs(scale - kTimeAttnScale) < 1.0e-7f) {
        int64_t BH = B * H;
        if (BH <= 0 || BH > std::numeric_limits<int>::max()) {
            throw std::runtime_error("mbr_tile::scaled_dot_product_attention: invalid time-attention batch");
        }
        Tensor qb = (q.dtype() == DType::BFloat16) ? q.contiguous() : q.to_bf16().contiguous();
        Tensor kb = (k.dtype() == DType::BFloat16) ? k.contiguous() : k.to_bf16().contiguous();
        Tensor vb = (v.dtype() == DType::BFloat16) ? v.contiguous() : v.to_bf16().contiguous();
        Tensor out = Tensor::empty({BH, N, D}, DType::BFloat16);
        if (time_attention_cutile_split_tail_enabled()) {
            launch_time_attention1301_split_tail_cutile(
                qb, kb, vb, out, BH, scale,
                time_attention_cutile_split_tail_k32_enabled(),
                time_attention_cutile_split_tail_q32_enabled(),
                time_attention_cutile_exp2_enabled());
        } else if (time_attention_cutile_k128_enabled()) {
            launch_time_attention1301_full_cutile(
                qb, kb, vb, out, BH, scale, kTimeAttnCutileQRows64, kTimeAttnCutileKTile128);
        } else if (time_attention_cutile_k32_enabled()) {
            launch_time_attention1301_full_cutile(
                qb, kb, vb, out, BH, scale, kTimeAttnCutileQRows64, kTimeAttnCutileKTile32);
        } else if (time_attention_cutile_q128_enabled()) {
            launch_time_attention1301_full_cutile(
                qb, kb, vb, out, BH, scale, kTimeAttnCutileQRows128, kTimeAttnCutileKTile64);
        } else if (time_attention_cutile_q32_enabled()) {
            launch_time_attention1301_full_cutile(
                qb, kb, vb, out, BH, scale, kTimeAttnCutileQRows32, kTimeAttnCutileKTile64);
        } else if (time_attention_cutile_q16_enabled()) {
            launch_time_attention1301_full_cutile(
                qb, kb, vb, out, BH, scale, kTimeAttnCutileQRows16, kTimeAttnCutileKTile64);
        } else if (time_attention_cutile_q64_enabled()) {
            launch_time_attention1301_full_cutile(
                qb, kb, vb, out, BH, scale, kTimeAttnCutileQRows64, kTimeAttnCutileKTile64);
        } else {
            unsupported_gemm_path("mbr_tile::scaled_dot_product_attention time CUDA Tile");
        }
        Tensor result = out.reshape({B, H, N, D});
        return (q.dtype() == DType::Float16) ? result.to_f16() : result;
    }

    int64_t BH = B * H;
    bool qk_bf16_scores = use_bf16_value_gemm && attention_qk_bf16_scores_enabled();
    bool qk_bf16_accum = qk_bf16_scores && attention_qk_bf16_accum_enabled();
    Tensor scores = Tensor::empty({BH, N, N_k}, qk_bf16_scores ? DType::BFloat16 : DType::Float32);
    Tensor scores_lowp;
    Tensor q_lowp;
    Tensor k_lowp;
    Tensor v_lowp;
    if (use_fp16_value_gemm) {
        Tensor vf = (v.dtype() == DType::Float32) ? v.contiguous() : v.to_f32().contiguous();
        scores_lowp = Tensor::empty({BH, N, N_k}, DType::Float16);
        v_lowp = vf.to_f16().contiguous();
    } else if (use_bf16_value_gemm) {
        q_lowp = (q.dtype() == DType::BFloat16) ? q.contiguous() : q.to_bf16().contiguous();
        k_lowp = (k.dtype() == DType::BFloat16) ? k.contiguous() : k.to_bf16().contiguous();
        scores_lowp = Tensor::empty({BH, N, N_k}, DType::BFloat16);
        v_lowp = (v.dtype() == DType::BFloat16) ? v.contiguous() : v.to_bf16().contiguous();
    }
    bool av_out_bf16 = use_bf16_value_gemm && attention_av_bf16_output_enabled();
    Tensor out = Tensor::empty({BH, N, D}, av_out_bf16 ? DType::BFloat16 : DType::Float32);

    if (qk_bf16_accum) {
        attention_qk_bf16_accum_kernel<<<(int)ceildiv(BH * N * N_k, kTile), 1>>>(
            q_lowp.data_bf16(), k_lowp.data_bf16(), scores.data_bf16(),
            BH * N * N_k, (int)N, (int)N_k, (int)D);
        CUDA_CHECK(cudaGetLastError());
    } else if (use_bf16_value_gemm) {
        unsupported_gemm_path("mbr_tile::scaled_dot_product_attention QK BF16");
    } else {
        unsupported_gemm_path("mbr_tile::scaled_dot_product_attention QK FP32/FP16");
    }

    bool use_small_softmax = N_k <= kSmallSoftmaxTile;
    if (use_fp16_value_gemm) {
        if (use_small_softmax) {
            scale_softmax_small_to_half_kernel<<<(int)(BH * N), 1>>>(
                scores.data_f32(), scores_lowp.data_f16(), (int)N_k, scale);
        } else {
            scale_softmax_to_half_kernel<<<(int)(BH * N), 1>>>(
                scores.data_f32(), scores_lowp.data_f16(), (int)N_k, scale);
        }
    } else if (use_bf16_value_gemm) {
        if (qk_bf16_scores) {
            if (use_small_softmax) {
                scale_softmax_small_bf16_to_bf16_kernel<<<(int)(BH * N), 1>>>(
                    scores.data_bf16(), scores_lowp.data_bf16(), (int)N_k, scale,
                    full_bf16_arith_enabled());
            } else {
                scale_softmax_bf16_to_bf16_kernel<<<(int)(BH * N), 1>>>(
                    scores.data_bf16(), scores_lowp.data_bf16(), (int)N_k, scale,
                    full_bf16_arith_enabled());
            }
        } else if (use_small_softmax) {
            scale_softmax_small_to_bf16_kernel<<<(int)(BH * N), 1>>>(
                scores.data_f32(), scores_lowp.data_bf16(), (int)N_k, scale);
        } else {
            scale_softmax_to_bf16_kernel<<<(int)(BH * N), 1>>>(
                scores.data_f32(), scores_lowp.data_bf16(), (int)N_k, scale);
        }
    } else {
        if (use_small_softmax) {
            scale_softmax_small_kernel<<<(int)(BH * N), 1>>>(scores.data_f32(), (int)N_k, scale);
        } else {
            scale_softmax_kernel<<<(int)(BH * N), 1>>>(scores.data_f32(), (int)N_k, scale);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    if (use_fp16_value_gemm) {
        unsupported_gemm_path("mbr_tile::scaled_dot_product_attention AV FP16");
    } else if (use_bf16_value_gemm) {
        unsupported_gemm_path("mbr_tile::scaled_dot_product_attention AV BF16");
    } else {
        unsupported_gemm_path("mbr_tile::scaled_dot_product_attention AV FP32");
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
    total_batch = x.numel() / in_features;
    out_shape = x.shape();
    out_shape.back() = out_features;
    unsupported_gemm_path("mbr_tile::linear_gemm_f32_output");
}

Tensor linear_gemm_bf16_output(const Tensor& x,
                               const Tensor& weight,
                               int64_t& total_batch,
                               int64_t& out_features,
                               std::vector<int64_t>& out_shape) {
    if (weight.dtype() != DType::BFloat16) {
        return linear_gemm_f32_output(x, weight, total_batch, out_features, out_shape).to_bf16();
    }

    int64_t in_features = weight.size(1);
    out_features = weight.size(0);
    Tensor xb = (x.dtype() == DType::BFloat16) ? x.contiguous() : x.to_bf16().contiguous();
    Tensor wb = weight.contiguous();
    total_batch = xb.numel() / in_features;
    out_shape = xb.shape();
    out_shape.back() = out_features;

    Tensor tile_out;
    if (try_linear_cutile_static_bf16_output(
            xb, wb, total_batch, out_features, in_features, tile_out)) {
        return tile_out;
    }
    unsupported_gemm_path("mbr_tile::linear_gemm_bf16_output");
}

bool try_feedforward_fused_impl(const Tensor& x,
                                const Tensor& linear1_w,
                                const Tensor& linear1_b,
                                const Tensor& linear2_w,
                                const Tensor& linear2_b,
                                const Tensor* residual,
                                Tensor& out) {
    if (!ffn12_fused_cutile_enabled()) return false;
    if (residual && !ffn12_fused_residual_enabled()) return false;
    if (residual &&
        (!(ffn12_fused_poly9_gelu_enabled() || ffn12_fused_odd5_gelu_enabled()) ||
         full_bf16_arith_enabled() ||
         !ffn12_fused_split2_output_enabled() ||
         !ffn12_fused_split2_pairh32_enabled() ||
         !ffn12_fused_pairh32_tk64_enabled())) {
        return false;
    }
    if (linear_gelu_split_enabled()) {
        return false;
    }
    if (linear1_w.dtype() != DType::BFloat16 || linear2_w.dtype() != DType::BFloat16) {
        return false;
    }
    if (linear1_w.ndim() != 2 || linear2_w.ndim() != 2 ||
        linear1_w.size(0) != 1024 || linear1_w.size(1) != 256 ||
        linear2_w.size(0) != 256 || linear2_w.size(1) != 1024) {
        return false;
    }
    if (x.numel() % 256 != 0) return false;
    int64_t total_batch = x.numel() / 256;
    if (total_batch != kLinearCutileExpectedM) return false;
    if (residual) {
        if (residual->dtype() != DType::BFloat16 ||
            residual->shape() != x.shape() ||
            residual->numel() != x.numel()) {
            return false;
        }
    }

    Tensor xb = (x.dtype() == DType::BFloat16) ? x.contiguous() : x.to_bf16().contiguous();
    Tensor w1 = linear1_w.contiguous();
    Tensor w2 = linear2_w.contiguous();
    Tensor b1 = (linear1_b.dtype() == DType::BFloat16)
        ? linear1_b.contiguous()
        : linear1_b.to_bf16().contiguous();
    Tensor b2 = (linear2_b.dtype() == DType::BFloat16)
        ? linear2_b.contiguous()
        : linear2_b.to_bf16().contiguous();
    Tensor residual_flat;
    if (residual) {
        Tensor residual_work = residual->contiguous();
        residual_flat = residual_work.reshape({total_batch, 256});
    }

    std::vector<int64_t> out_shape = xb.shape();
    out_shape.back() = 256;
    Tensor out_flat = Tensor::empty({total_batch, 256}, DType::BFloat16);
    int gelu_mode = kGeluErf;
    if (ffn12_fused_tinyblend_gelu_enabled()) {
        gelu_mode = kGeluErfPoly9TinyBlendL30;
    } else if (ffn12_fused_odd5_gelu_enabled()) {
        gelu_mode = kGeluErfOdd5L175;
    } else if (ffn12_fused_poly9_gelu_enabled()) {
        gelu_mode = kGeluErfPoly9L30;
    } else if (ffn12_fused_poly7_gelu_enabled()) {
        gelu_mode = kGeluErfPoly7L25;
    } else if (ffn12_fused_poly5_gelu_enabled()) {
        gelu_mode = kGeluErfPoly5L25;
    } else if (ffn12_fused_tanh_gelu_enabled() || linear_tanh_gelu_enabled()) {
        gelu_mode = kGeluTanh;
    } else if (ffn12_fused_quick_gelu_enabled() || linear_quick_gelu_enabled()) {
        gelu_mode = kGeluQuick;
    } else if (ffn12_fused_hard_gelu_enabled() || linear_hard_gelu_enabled()) {
        gelu_mode = kGeluHard;
    }
    if (residual && ffn12_residual_two_kernel_enabled()) {
        Tensor ff_out_flat = Tensor::empty({total_batch, 256}, DType::BFloat16);
        launch_ffn12_fused256_cutile(gelu_mode,
                                     full_bf16_arith_enabled(),
                                     ffn12_fused_split2_output_enabled(),
                                     ffn12_fused_split2_pairh32_enabled(),
                                     ffn12_fused_pairh32_tk64_enabled(),
                                     xb,
                                     w1,
                                     b1,
                                     w2,
                                     b2,
                                     ff_out_flat);
        CUDA_CHECK(cudaGetLastError());
        if (!try_residual_add_bf16_cutile(ff_out_flat, residual_flat, out_flat)) {
            return false;
        }
    } else if (residual) {
        launch_ffn12_fused256_residual_cutile(gelu_mode,
                                              full_bf16_arith_enabled(),
                                              ffn12_fused_split2_output_enabled(),
                                              ffn12_fused_split2_pairh32_enabled(),
                                              ffn12_fused_pairh32_tk64_enabled(),
                                              xb,
                                              w1,
                                              b1,
                                              w2,
                                              b2,
                                              residual_flat,
                                              out_flat);
    } else {
        launch_ffn12_fused256_cutile(gelu_mode,
                                     full_bf16_arith_enabled(),
                                     ffn12_fused_split2_output_enabled(),
                                     ffn12_fused_split2_pairh32_enabled(),
                                     ffn12_fused_pairh32_tk64_enabled(),
                                     xb,
                                     w1,
                                     b1,
                                     w2,
                                     b2,
                                     out_flat);
    }
    CUDA_CHECK(cudaGetLastError());
    out = out_flat.reshape(out_shape);
    return true;
}

bool try_feedforward_fused(const Tensor& x,
                           const Tensor& linear1_w,
                           const Tensor& linear1_b,
                           const Tensor& linear2_w,
                           const Tensor& linear2_b,
                           Tensor& out) {
    return try_feedforward_fused_impl(
        x, linear1_w, linear1_b, linear2_w, linear2_b, nullptr, out);
}

bool try_feedforward_fused_residual(const Tensor& x,
                                    const Tensor& linear1_w,
                                    const Tensor& linear1_b,
                                    const Tensor& linear2_w,
                                    const Tensor& linear2_b,
                                    const Tensor& residual,
                                    Tensor& out) {
    return try_feedforward_fused_impl(
        x, linear1_w, linear1_b, linear2_w, linear2_b, &residual, out);
}

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    int64_t total_batch = 0;
    int64_t out_features = 0;
    std::vector<int64_t> out_shape;
    bool direct_bf16 = linear_direct_bf16_output_enabled() && weight.dtype() == DType::BFloat16;
    if (direct_bf16) {
        int64_t in_features = weight.size(1);
        out_features = weight.size(0);
        Tensor xb = (x.dtype() == DType::BFloat16) ? x.contiguous() : x.to_bf16().contiguous();
        Tensor wb = weight.contiguous();
        Tensor bb = (bias.dtype() == DType::BFloat16) ? bias.contiguous() : bias.to_bf16().contiguous();
        total_batch = xb.numel() / in_features;
        out_shape = xb.shape();
        out_shape.back() = out_features;

        Tensor out;
        if (try_linear_cutile_static_bf16_bias_output(
                xb, wb, bb, total_batch, out_features, in_features, out)) {
            return out.reshape(out_shape);
        }
        if (try_linear_cutile_static_small_bf16_bias_output(
                xb, wb, bb, total_batch, out_features, in_features, out)) {
            return out.reshape(out_shape);
        }
        unsupported_gemm_path("mbr_tile::linear");
    }

    Tensor out = linear_gemm_f32_output(x, weight, total_batch, out_features, out_shape);
    Tensor bf = (bias.dtype() == DType::Float32) ? bias.contiguous() : bias.to_f32().contiguous();

    long long total = total_batch * out_features;
    if (linear_bf16_output_enabled() && weight.dtype() == DType::BFloat16) {
        Tensor out_bf16 = Tensor::empty({total_batch, out_features}, DType::BFloat16);
        add_bias_to_bf16_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            out.data_f32(), bf.data_f32(), out_bf16.data_bf16(), total, (int)out_features);
        CUDA_CHECK(cudaGetLastError());
        return out_bf16.reshape(out_shape);
    }

    add_bias_kernel<<<(int)ceildiv(total, kTile), 1>>>(
        out.data_f32(), bf.data_f32(), total, (int)out_features);
    CUDA_CHECK(cudaGetLastError());

    Tensor result = out.reshape(out_shape);
    return (weight.dtype() == DType::Float16 || weight.dtype() == DType::BFloat16) ? result :
           ((x.dtype() == DType::Float16) ? result.to_f16() : result);
}

Tensor linear_bkn(const Tensor& x,
                  const Tensor& weight,
                  const Tensor& weight_bkn,
                  const Tensor& bias) {
    if (linear_direct_bf16_output_enabled() &&
        weight.dtype() == DType::BFloat16 &&
        weight_bkn.dtype() == DType::BFloat16) {
        int64_t in_features = weight.size(1);
        int64_t out_features = weight.size(0);
        Tensor xb = (x.dtype() == DType::BFloat16) ? x.contiguous() : x.to_bf16().contiguous();
        Tensor wb = weight.contiguous();
        Tensor wbkn = weight_bkn.contiguous();
        Tensor bb = (bias.dtype() == DType::BFloat16) ? bias.contiguous() : bias.to_bf16().contiguous();
        int64_t total_batch = xb.numel() / in_features;
        std::vector<int64_t> out_shape = xb.shape();
        out_shape.back() = out_features;

        Tensor out;
        if (try_linear_cutile_static_bkn_bf16_output(
                xb, wb, wbkn, &bb, total_batch, out_features, in_features,
                true, false, out)) {
            return out.reshape(out_shape);
        }
    }
    return linear(x, weight, bias);
}

Tensor linear_gelu(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    int64_t total_batch = 0;
    int64_t out_features = 0;
    std::vector<int64_t> out_shape;
    bool direct_bf16 = linear_direct_bf16_output_enabled() && weight.dtype() == DType::BFloat16;
    if (direct_bf16) {
        int64_t in_features = weight.size(1);
        out_features = weight.size(0);
        Tensor xb = (x.dtype() == DType::BFloat16) ? x.contiguous() : x.to_bf16().contiguous();
        Tensor wb = weight.contiguous();
        Tensor bb = (bias.dtype() == DType::BFloat16) ? bias.contiguous() : bias.to_bf16().contiguous();
        total_batch = xb.numel() / in_features;
        out_shape = xb.shape();
        out_shape.back() = out_features;

        Tensor out;
        if (try_linear_cutile_static_bf16_bias_gelu_split_output(
                xb, wb, bb, total_batch, out_features, in_features, out)) {
            return out.reshape(out_shape);
        }
        if (try_linear_cutile_static_bf16_bias_gelu_output(
                xb, wb, bb, total_batch, out_features, in_features, out)) {
            return out.reshape(out_shape);
        }
        unsupported_gemm_path("mbr_tile::linear_gelu");
    }

    Tensor out = linear_gemm_f32_output(x, weight, total_batch, out_features, out_shape);
    Tensor bf = (bias.dtype() == DType::Float32) ? bias.contiguous() : bias.to_f32().contiguous();

    long long total = total_batch * out_features;
    if (weight.dtype() == DType::BFloat16) {
        Tensor out_bf16 = Tensor::empty({total_batch, out_features}, DType::BFloat16);
        add_bias_gelu_to_bf16_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            out.data_f32(), bf.data_f32(), out_bf16.data_bf16(), total, (int)out_features);
        CUDA_CHECK(cudaGetLastError());
        return out_bf16.reshape(out_shape);
    } else {
        add_bias_gelu_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            out.data_f32(), bf.data_f32(), total, (int)out_features);
        CUDA_CHECK(cudaGetLastError());
    }

    Tensor result = out.reshape(out_shape);
    return (x.dtype() == DType::Float16) ? result.to_f16() : result;
}

Tensor linear_gelu_bkn(const Tensor& x,
                       const Tensor& weight,
                       const Tensor& weight_bkn,
                       const Tensor& bias) {
    if (linear_direct_bf16_output_enabled() &&
        weight.dtype() == DType::BFloat16 &&
        weight_bkn.dtype() == DType::BFloat16) {
        int64_t in_features = weight.size(1);
        int64_t out_features = weight.size(0);
        Tensor xb = (x.dtype() == DType::BFloat16) ? x.contiguous() : x.to_bf16().contiguous();
        Tensor wb = weight.contiguous();
        Tensor wbkn = weight_bkn.contiguous();
        Tensor bb = (bias.dtype() == DType::BFloat16) ? bias.contiguous() : bias.to_bf16().contiguous();
        int64_t total_batch = xb.numel() / in_features;
        std::vector<int64_t> out_shape = xb.shape();
        out_shape.back() = out_features;

        Tensor out;
        if (try_linear_cutile_static_bkn_bf16_output(
                xb, wb, wbkn, &bb, total_batch, out_features, in_features,
                true, true, out)) {
            return out.reshape(out_shape);
        }
    }
    return linear_gelu(x, weight, bias);
}

Tensor linear_no_bias(const Tensor& x, const Tensor& weight) {
    int64_t total_batch = 0;
    int64_t out_features = 0;
    std::vector<int64_t> out_shape;
    if (linear_direct_bf16_output_enabled() && weight.dtype() == DType::BFloat16) {
        Tensor out = linear_gemm_bf16_output(x, weight, total_batch, out_features, out_shape);
        return out.reshape(out_shape);
    }
    if (linear_bf16_output_enabled() && weight.dtype() == DType::BFloat16) {
        return linear_no_bias_bf16_output(x, weight);
    }
    Tensor out = linear_gemm_f32_output(x, weight, total_batch, out_features, out_shape);
    Tensor result = out.reshape(out_shape);
    return (weight.dtype() == DType::Float16 || weight.dtype() == DType::BFloat16) ? result :
           ((x.dtype() == DType::Float16) ? result.to_f16() : result);
}

Tensor linear_no_bias_bkn(const Tensor& x, const Tensor& weight, const Tensor& weight_bkn) {
#if CUDASEP_ENABLE_QKV_BF16_OUTPUT
    if (weight.dtype() == DType::BFloat16 && weight_bkn.dtype() == DType::BFloat16) {
        int64_t in_features = weight.size(1);
        int64_t out_features = weight.size(0);
        Tensor xb = (x.dtype() == DType::BFloat16) ? x.contiguous() : x.to_bf16().contiguous();
        Tensor wb = weight.contiguous();
        Tensor wbkn = weight_bkn.contiguous();
        int64_t total_batch = xb.numel() / in_features;
        std::vector<int64_t> out_shape = xb.shape();
        out_shape.back() = out_features;

        Tensor out;
        if (try_linear_cutile_static_bkn_bf16_output(
                xb, wb, wbkn, nullptr, total_batch, out_features, in_features,
                false, false, out)) {
            return out.reshape(out_shape);
        }
    }
#endif
    return linear_no_bias_bf16_output(x, weight);
}

Tensor linear_no_bias_bf16_output(const Tensor& x, const Tensor& weight) {
#if CUDASEP_ENABLE_QKV_BF16_OUTPUT
    if (weight.dtype() != DType::BFloat16) {
        return linear_no_bias(x, weight);
    }

    int64_t in_features = weight.size(1);
    int64_t out_features = weight.size(0);
    Tensor xb = (x.dtype() == DType::BFloat16) ? x.contiguous() : x.to_bf16().contiguous();
    Tensor wb = weight.contiguous();
    int64_t total_batch = xb.numel() / in_features;
    std::vector<int64_t> out_shape = xb.shape();
    out_shape.back() = out_features;

    Tensor tile_out;
    if (try_linear_cutile_static_bf16_output(
            xb, wb, total_batch, out_features, in_features, tile_out)) {
        return tile_out.reshape(out_shape);
    }
    unsupported_gemm_path("mbr_tile::linear_no_bias_bf16_output");
#else
    return linear_no_bias(x, weight);
#endif
}

bool try_linear_no_bias_residual_bf16_output(const Tensor& x,
                                             const Tensor& weight,
                                             const Tensor& residual,
                                             Tensor& out) {
#if CUDASEP_ENABLE_QKV_BF16_OUTPUT
    if (!attention_out_residual_enabled()) return false;
    if (x.dtype() != DType::BFloat16 || weight.dtype() != DType::BFloat16 ||
        residual.dtype() != DType::BFloat16) {
        return false;
    }
    if (x.ndim() == 0 || weight.ndim() != 2) return false;

    int64_t in_features = weight.size(1);
    int64_t out_features = weight.size(0);
    if (x.size(x.ndim() - 1) != in_features) return false;
    if (x.numel() % in_features != 0) return false;

    int64_t total_batch = x.numel() / in_features;
    if (!linear_cutile_static_attn_out_candidate(
            total_batch, out_features, in_features)) {
        return false;
    }

    std::vector<int64_t> out_shape = x.shape();
    out_shape.back() = out_features;
    if (residual.shape() != out_shape || !residual.is_contiguous()) {
        return false;
    }

    Tensor xb = x.is_contiguous() ? x : x.contiguous();
    Tensor wb = weight.is_contiguous() ? weight : weight.contiguous();
    Tensor flat = Tensor::empty({total_batch, out_features}, DType::BFloat16);
    launch_linear_cutile_static_attn_out_residual_bf16(
        xb, wb, residual, flat, (int)total_batch);
    CUDA_CHECK(cudaGetLastError());
    out = flat.reshape(out_shape);
    return true;
#else
    (void)x;
    (void)weight;
    (void)residual;
    (void)out;
    return false;
#endif
}

void linear_qkv_bkn_split_contig_time(const Tensor& x,
                                      const Tensor& weight,
                                      const Tensor& weight_bkn,
                                      Tensor& q,
                                      Tensor& k,
                                      Tensor& v) {
    if (x.dtype() != DType::BFloat16 || weight.dtype() != DType::BFloat16 ||
        weight_bkn.dtype() != DType::BFloat16) {
        throw std::runtime_error("time QKV split-contig CUDA Tile: expected BF16 tensors");
    }
    if (x.ndim() != 3 || x.size(1) != kTimeAttnN || x.size(2) != 256) {
        throw std::runtime_error(
            "time QKV split-contig CUDA Tile: expected x [B,1301,256]");
    }
    constexpr int kOutFeatures = 3 * kQkvFusedHeads * kTimeAttnD;
    constexpr int kInFeatures = 256;
    if (weight.ndim() != 2 || weight.size(0) != kOutFeatures ||
        weight.size(1) != kInFeatures) {
        throw std::runtime_error(
            "time QKV split-contig CUDA Tile: expected weight [1536,256]");
    }
    if (weight_bkn.ndim() != 2 || weight_bkn.size(0) != kInFeatures ||
        weight_bkn.size(1) != kOutFeatures) {
        throw std::runtime_error(
            "time QKV split-contig CUDA Tile: expected weight_bkn [256,1536]");
    }

    int64_t batches = x.size(0);
    int64_t total_batch = batches * kTimeAttnN;
    if (total_batch != kLinearCutileExpectedM) {
        throw std::runtime_error(
            "time QKV split-contig CUDA Tile: expected B*1301 == 78060");
    }
    if (batches <= 0 || batches > std::numeric_limits<int>::max()) {
        throw std::runtime_error("time QKV split-contig CUDA Tile: invalid batch count");
    }

    Tensor x_work = x.contiguous();
    Tensor weight_work = weight.contiguous();
    Tensor weight_bkn_work = weight_bkn.contiguous();
    q = Tensor::empty({batches, kTimeAttnN, kQkvFusedHeads, kTimeAttnD},
                      DType::BFloat16);
    k = Tensor::empty({batches, kTimeAttnN, kQkvFusedHeads, kTimeAttnD},
                      DType::BFloat16);
    v = Tensor::empty({batches, kTimeAttnN, kQkvFusedHeads, kTimeAttnD},
                      DType::BFloat16);

    bool use_m64_tn128 = time_qkv_bkn_m64_tn128_enabled();
    int static_rows = kLinearCutileStaticM;
    if (use_m64_tn128) {
        static_rows = kLinearCutileStaticM64;
        dim3 full_grid(kLinearCutileStaticM64 / 64, kOutFeatures / 128, 1);
        qkv_bkn_split_contig_static_full_kernel<64,
                                                128,
                                                16,
                                                kLinearCutileStaticM64,
                                                kOutFeatures,
                                                kInFeatures,
                                                2>
            <<<full_grid, 1>>>(
                x_work.data_bf16(), weight_bkn_work.data_bf16(),
                q.data_bf16(), k.data_bf16(), v.data_bf16());
    } else {
        dim3 full_grid(kLinearCutileStaticM / 32, kOutFeatures / 256, 1);
        qkv_bkn_split_contig_static_full_kernel<32,
                                                256,
                                                16,
                                                kLinearCutileStaticM,
                                                kOutFeatures,
                                                kInFeatures,
                                                2>
            <<<full_grid, 1>>>(
                x_work.data_bf16(), weight_bkn_work.data_bf16(),
                q.data_bf16(), k.data_bf16(), v.data_bf16());
    }

    int tail_rows = (int)(total_batch - static_rows);
    if (tail_rows > 0) {
        if (use_m64_tn128) {
            dim3 tail_grid((unsigned int)ceildiv(tail_rows, 64),
                           kOutFeatures / 128,
                           1);
            qkv_bkn_split_contig_masked_tail_kernel<64,
                                                    128,
                                                    16,
                                                    kLinearCutileExpectedM,
                                                    kOutFeatures,
                                                    kInFeatures,
                                                    kLinearCutileStaticM64,
                                                    2>
                <<<tail_grid, 1>>>(
                    x_work.data_bf16(), weight_bkn_work.data_bf16(),
                    q.data_bf16(), k.data_bf16(), v.data_bf16());
        } else {
            int tail_total = tail_rows * kOutFeatures;
            qkv_bkn_split_contig_tail_kernel<kOutFeatures, kInFeatures>
                <<<(int)ceildiv(tail_total, kTile), 1>>>(
                    x_work.data_bf16(), weight_work.data_bf16(),
                    q.data_bf16(), k.data_bf16(), v.data_bf16(),
                    (int)total_batch, static_rows);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void linear_qkv_bkn_rotary_split_contig_time(const Tensor& x,
                                             const Tensor& weight,
                                             const Tensor& weight_bkn,
                                             const Tensor& cos_freqs,
                                             const Tensor& sin_freqs,
                                             Tensor& q,
                                             Tensor& k,
                                             Tensor& v) {
    if (x.dtype() != DType::BFloat16 || weight.dtype() != DType::BFloat16 ||
        weight_bkn.dtype() != DType::BFloat16) {
        throw std::runtime_error(
            "time QKV+rotary split-contig CUDA Tile: expected BF16 tensors");
    }
    if (x.ndim() != 3 || x.size(1) != kTimeAttnN || x.size(2) != 256) {
        throw std::runtime_error(
            "time QKV+rotary split-contig CUDA Tile: expected x [B,1301,256]");
    }
    constexpr int kOutFeatures = 3 * kQkvFusedHeads * kTimeAttnD;
    constexpr int kInFeatures = 256;
    if (weight.ndim() != 2 || weight.size(0) != kOutFeatures ||
        weight.size(1) != kInFeatures) {
        throw std::runtime_error(
            "time QKV+rotary split-contig CUDA Tile: expected weight [1536,256]");
    }
    if (weight_bkn.ndim() != 2 || weight_bkn.size(0) != kInFeatures ||
        weight_bkn.size(1) != kOutFeatures) {
        throw std::runtime_error(
            "time QKV+rotary split-contig CUDA Tile: expected weight_bkn [256,1536]");
    }
    if (cos_freqs.ndim() != 2 || sin_freqs.ndim() != 2 ||
        cos_freqs.size(0) != kTimeAttnN || sin_freqs.size(0) != kTimeAttnN ||
        cos_freqs.size(1) != kTimeAttnD / 2 || sin_freqs.size(1) != kTimeAttnD / 2) {
        throw std::runtime_error(
            "time QKV+rotary split-contig CUDA Tile: expected cos/sin [1301,32]");
    }

    int64_t batches = x.size(0);
    int64_t total_batch = batches * kTimeAttnN;
    if (total_batch != kLinearCutileExpectedM) {
        throw std::runtime_error(
            "time QKV+rotary split-contig CUDA Tile: expected B*1301 == 78060");
    }
    if (batches <= 0 || batches > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "time QKV+rotary split-contig CUDA Tile: invalid batch count");
    }

    Tensor x_work = x.contiguous();
    Tensor weight_work = weight.contiguous();
    Tensor weight_bkn_work = weight_bkn.contiguous();
    bool trig_bf16 = cos_freqs.dtype() == DType::BFloat16 &&
                     sin_freqs.dtype() == DType::BFloat16;
    Tensor cos_work = trig_bf16 ? cos_freqs.contiguous()
                                : ((cos_freqs.dtype() == DType::Float32)
                                       ? cos_freqs.contiguous()
                                       : cos_freqs.to_f32().contiguous());
    Tensor sin_work = trig_bf16 ? sin_freqs.contiguous()
                                : ((sin_freqs.dtype() == DType::Float32)
                                       ? sin_freqs.contiguous()
                                       : sin_freqs.to_f32().contiguous());
    q = Tensor::empty({batches, kTimeAttnN, kQkvFusedHeads, kTimeAttnD},
                      DType::BFloat16);
    k = Tensor::empty({batches, kTimeAttnN, kQkvFusedHeads, kTimeAttnD},
                      DType::BFloat16);
    v = Tensor::empty({batches, kTimeAttnN, kQkvFusedHeads, kTimeAttnD},
                      DType::BFloat16);

    dim3 full_grid(kLinearCutileStaticM / 32, kOutFeatures / 256, 1);
    if (trig_bf16) {
        qkv_bkn_rotary_split_contig_static_full_kernel<32,
                                                       256,
                                                       16,
                                                       kLinearCutileStaticM,
                                                       kOutFeatures,
                                                       kInFeatures,
                                                       __nv_bfloat16>
            <<<full_grid, 1>>>(
                x_work.data_bf16(), weight_bkn_work.data_bf16(),
                cos_work.data_bf16(), sin_work.data_bf16(),
                q.data_bf16(), k.data_bf16(), v.data_bf16(),
                full_bf16_arith_enabled());
    } else {
        qkv_bkn_rotary_split_contig_static_full_kernel<32,
                                                       256,
                                                       16,
                                                       kLinearCutileStaticM,
                                                       kOutFeatures,
                                                       kInFeatures,
                                                       float>
            <<<full_grid, 1>>>(
                x_work.data_bf16(), weight_bkn_work.data_bf16(),
                cos_work.data_f32(), sin_work.data_f32(),
                q.data_bf16(), k.data_bf16(), v.data_bf16(),
                full_bf16_arith_enabled());
    }

    int tail_rows = (int)(total_batch - kLinearCutileStaticM);
    if (tail_rows > 0) {
        int tail_total = tail_rows * kOutFeatures;
        qkv_bkn_split_contig_tail_kernel<kOutFeatures, kInFeatures>
            <<<(int)ceildiv(tail_total, kTile), 1>>>(
                x_work.data_bf16(), weight_work.data_bf16(),
                q.data_bf16(), k.data_bf16(), v.data_bf16(),
                (int)total_batch, kLinearCutileStaticM);
        long long tail_pairs =
            (long long)tail_rows * kQkvFusedHeads * (kTimeAttnD / 2);
        if (trig_bf16) {
            rotary_time_split_contig_tail_inplace_kernel<__nv_bfloat16>
                <<<(int)ceildiv(tail_pairs, kTile), 1>>>(
                    q.data_bf16(), k.data_bf16(),
                    cos_work.data_bf16(), sin_work.data_bf16(),
                    total_batch, kLinearCutileStaticM,
                    full_bf16_arith_enabled());
        } else {
            rotary_time_split_contig_tail_inplace_kernel<float>
                <<<(int)ceildiv(tail_pairs, kTile), 1>>>(
                    q.data_bf16(), k.data_bf16(),
                    cos_work.data_f32(), sin_work.data_f32(),
                    total_batch, kLinearCutileStaticM,
                    full_bf16_arith_enabled());
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void linear_qkv_pair_rotary_split_contig_time(const Tensor& x,
                                             const Tensor& weight,
                                             const Tensor& cos_freqs,
                                             const Tensor& sin_freqs,
                                             Tensor& q,
                                             Tensor& k,
                                             Tensor& v) {
    if (x.dtype() != DType::BFloat16 || weight.dtype() != DType::BFloat16) {
        throw std::runtime_error(
            "time QKV+rotary pair split-contig CUDA Tile: expected BF16 tensors");
    }
    if (x.ndim() != 3 || x.size(1) != kTimeAttnN || x.size(2) != 256) {
        throw std::runtime_error(
            "time QKV+rotary pair split-contig CUDA Tile: expected x [B,1301,256]");
    }
    constexpr int kOutFeatures = 3 * kQkvFusedHeads * kTimeAttnD;
    constexpr int kInFeatures = 256;
    if (weight.ndim() != 2 || weight.size(0) != kOutFeatures ||
        weight.size(1) != kInFeatures) {
        throw std::runtime_error(
            "time QKV+rotary pair split-contig CUDA Tile: expected weight [1536,256]");
    }
    if (cos_freqs.ndim() != 2 || sin_freqs.ndim() != 2 ||
        cos_freqs.size(0) != kTimeAttnN || sin_freqs.size(0) != kTimeAttnN ||
        cos_freqs.size(1) != kTimeAttnD / 2 || sin_freqs.size(1) != kTimeAttnD / 2) {
        throw std::runtime_error(
            "time QKV+rotary pair split-contig CUDA Tile: expected cos/sin [1301,32]");
    }

    int64_t batches = x.size(0);
    int64_t total_batch = batches * kTimeAttnN;
    if (total_batch != kLinearCutileExpectedM) {
        throw std::runtime_error(
            "time QKV+rotary pair split-contig CUDA Tile: expected B*1301 == 78060");
    }
    if (batches <= 0 || batches > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "time QKV+rotary pair split-contig CUDA Tile: invalid batch count");
    }

    Tensor x_work = x.contiguous();
    Tensor weight_work = weight.contiguous();
    bool trig_bf16 = cos_freqs.dtype() == DType::BFloat16 &&
                     sin_freqs.dtype() == DType::BFloat16;
    Tensor cos_work = trig_bf16 ? cos_freqs.contiguous()
                                : ((cos_freqs.dtype() == DType::Float32)
                                       ? cos_freqs.contiguous()
                                       : cos_freqs.to_f32().contiguous());
    Tensor sin_work = trig_bf16 ? sin_freqs.contiguous()
                                : ((sin_freqs.dtype() == DType::Float32)
                                       ? sin_freqs.contiguous()
                                       : sin_freqs.to_f32().contiguous());
    q = Tensor::empty({batches, kTimeAttnN, kQkvFusedHeads, kTimeAttnD},
                      DType::BFloat16);
    k = Tensor::empty({batches, kTimeAttnN, kQkvFusedHeads, kTimeAttnD},
                      DType::BFloat16);
    v = Tensor::empty({batches, kTimeAttnN, kQkvFusedHeads, kTimeAttnD},
                      DType::BFloat16);

    dim3 full_grid(kLinearCutileStaticM / 32, 3 * kQkvFusedHeads, 1);
    if (trig_bf16) {
        qkv_time_rotary_pair_split_contig_static_full_kernel<32,
                                                             kTimeAttnD / 2,
                                                             16,
                                                             kLinearCutileStaticM,
                                                             kInFeatures,
                                                             __nv_bfloat16>
            <<<full_grid, 1>>>(
                x_work.data_bf16(), weight_work.data_bf16(),
                cos_work.data_bf16(), sin_work.data_bf16(),
                q.data_bf16(), k.data_bf16(), v.data_bf16(),
                full_bf16_arith_enabled());
    } else {
        qkv_time_rotary_pair_split_contig_static_full_kernel<32,
                                                             kTimeAttnD / 2,
                                                             16,
                                                             kLinearCutileStaticM,
                                                             kInFeatures,
                                                             float>
            <<<full_grid, 1>>>(
                x_work.data_bf16(), weight_work.data_bf16(),
                cos_work.data_f32(), sin_work.data_f32(),
                q.data_bf16(), k.data_bf16(), v.data_bf16(),
                full_bf16_arith_enabled());
    }

    int tail_rows = (int)(total_batch - kLinearCutileStaticM);
    if (tail_rows > 0) {
        int tail_total = tail_rows * kOutFeatures;
        qkv_bkn_split_contig_tail_kernel<kOutFeatures, kInFeatures>
            <<<(int)ceildiv(tail_total, kTile), 1>>>(
                x_work.data_bf16(), weight_work.data_bf16(),
                q.data_bf16(), k.data_bf16(), v.data_bf16(),
                (int)total_batch, kLinearCutileStaticM);
        long long tail_pairs =
            (long long)tail_rows * kQkvFusedHeads * (kTimeAttnD / 2);
        if (trig_bf16) {
            rotary_time_split_contig_tail_inplace_kernel<__nv_bfloat16>
                <<<(int)ceildiv(tail_pairs, kTile), 1>>>(
                    q.data_bf16(), k.data_bf16(),
                    cos_work.data_bf16(), sin_work.data_bf16(),
                    total_batch, kLinearCutileStaticM, full_bf16_arith_enabled());
        } else {
            rotary_time_split_contig_tail_inplace_kernel<float>
                <<<(int)ceildiv(tail_pairs, kTile), 1>>>(
                    q.data_bf16(), k.data_bf16(),
                    cos_work.data_f32(), sin_work.data_f32(),
                    total_batch, kLinearCutileStaticM, full_bf16_arith_enabled());
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void apply_rotary_time_split_contig_inplace(Tensor& q,
                                            Tensor& k,
                                            const Tensor& cos_freqs,
                                            const Tensor& sin_freqs) {
    if (q.dtype() != DType::BFloat16 || k.dtype() != DType::BFloat16) {
        throw std::runtime_error("time split-contig rotary CUDA Tile: expected BF16 q/k");
    }
    if (q.ndim() != 4 || q.size(1) != kTimeAttnN ||
        q.size(2) != kQkvFusedHeads || q.size(3) != kTimeAttnD ||
        k.shape() != q.shape()) {
        throw std::runtime_error(
            "time split-contig rotary CUDA Tile: expected q/k [B,1301,8,64]");
    }
    if (!q.is_contiguous() || !k.is_contiguous()) {
        throw std::runtime_error(
            "time split-contig rotary CUDA Tile: expected contiguous q/k");
    }
    if (cos_freqs.ndim() != 2 || sin_freqs.ndim() != 2 ||
        cos_freqs.size(0) != kTimeAttnN || sin_freqs.size(0) != kTimeAttnN ||
        cos_freqs.size(1) != kTimeAttnD / 2 || sin_freqs.size(1) != kTimeAttnD / 2) {
        throw std::runtime_error(
            "time split-contig rotary CUDA Tile: expected cos/sin [1301,32]");
    }

    int64_t batches = q.size(0);
    if (batches * kTimeAttnN != kLinearCutileExpectedM) {
        throw std::runtime_error(
            "time split-contig rotary CUDA Tile: expected B*1301 == 78060");
    }
    long long total = batches * (long long)kTimeAttnN *
                      kQkvFusedHeads * (kTimeAttnD / 2);
    if (time_rotary_row_tile_enabled_impl()) {
        long long total_rows = batches * (long long)kTimeAttnN;
        bool trig_bf16 = cos_freqs.dtype() == DType::BFloat16 &&
                          sin_freqs.dtype() == DType::BFloat16;
        if (trig_bf16) {
            Tensor cos_work = cos_freqs.contiguous();
            Tensor sin_work = sin_freqs.contiguous();
            rotary_time_split_contig_row_tile_inplace_kernel<__nv_bfloat16>
                <<<(int)total_rows, 1>>>(
                    q.data_bf16(), k.data_bf16(),
                    cos_work.data_bf16(), sin_work.data_bf16(),
                    total_rows, full_bf16_arith_enabled());
        } else {
            Tensor cos_work = (cos_freqs.dtype() == DType::Float32)
                ? cos_freqs.contiguous()
                : cos_freqs.to_f32().contiguous();
            Tensor sin_work = (sin_freqs.dtype() == DType::Float32)
                ? sin_freqs.contiguous()
                : sin_freqs.to_f32().contiguous();
            rotary_time_split_contig_row_tile_inplace_kernel<float>
                <<<(int)total_rows, 1>>>(
                    q.data_bf16(), k.data_bf16(),
                    cos_work.data_f32(), sin_work.data_f32(),
                    total_rows, full_bf16_arith_enabled());
        }
    } else {
        bool trig_bf16 = cos_freqs.dtype() == DType::BFloat16 &&
                          sin_freqs.dtype() == DType::BFloat16;
        if (trig_bf16) {
            Tensor cos_work = cos_freqs.contiguous();
            Tensor sin_work = sin_freqs.contiguous();
            launch_rotary_time_split_contig_inplace_by_tile<__nv_bfloat16>(
                q, k, cos_work, sin_work, total, full_bf16_arith_enabled());
        } else {
            Tensor cos_work = (cos_freqs.dtype() == DType::Float32)
                ? cos_freqs.contiguous()
                : cos_freqs.to_f32().contiguous();
            Tensor sin_work = (sin_freqs.dtype() == DType::Float32)
                ? sin_freqs.contiguous()
                : sin_freqs.to_f32().contiguous();
            launch_rotary_time_split_contig_inplace_by_tile<float>(
                q, k, cos_work, sin_work, total, full_bf16_arith_enabled());
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void apply_rotary_time_split_contig_k_only_inplace(Tensor& k,
                                                   const Tensor& cos_freqs,
                                                   const Tensor& sin_freqs) {
    if (k.dtype() != DType::BFloat16) {
        throw std::runtime_error("time split-contig K-only rotary CUDA Tile: expected BF16 k");
    }
    if (k.ndim() != 4 || k.size(1) != kTimeAttnN ||
        k.size(2) != kQkvFusedHeads || k.size(3) != kTimeAttnD) {
        throw std::runtime_error(
            "time split-contig K-only rotary CUDA Tile: expected k [B,1301,8,64]");
    }
    if (!k.is_contiguous()) {
        throw std::runtime_error(
            "time split-contig K-only rotary CUDA Tile: expected contiguous k");
    }
    if (cos_freqs.ndim() != 2 || sin_freqs.ndim() != 2 ||
        cos_freqs.size(0) != kTimeAttnN || sin_freqs.size(0) != kTimeAttnN ||
        cos_freqs.size(1) != kTimeAttnD / 2 || sin_freqs.size(1) != kTimeAttnD / 2) {
        throw std::runtime_error(
            "time split-contig K-only rotary CUDA Tile: expected cos/sin [1301,32]");
    }

    int64_t batches = k.size(0);
    if (batches * kTimeAttnN != kLinearCutileExpectedM) {
        throw std::runtime_error(
            "time split-contig K-only rotary CUDA Tile: expected B*1301 == 78060");
    }
    long long total = batches * (long long)kTimeAttnN *
                      kQkvFusedHeads * (kTimeAttnD / 2);
    if (cos_freqs.dtype() == DType::BFloat16 &&
        sin_freqs.dtype() == DType::BFloat16) {
        Tensor cos_work = cos_freqs.contiguous();
        Tensor sin_work = sin_freqs.contiguous();
        rotary_time_split_contig_k_only_inplace_kernel<__nv_bfloat16>
            <<<(int)ceildiv(total, kTile), 1>>>(
                k.data_bf16(),
                cos_work.data_bf16(), sin_work.data_bf16(),
                total, full_bf16_arith_enabled());
    } else {
        Tensor cos_work = (cos_freqs.dtype() == DType::Float32)
            ? cos_freqs.contiguous()
            : cos_freqs.to_f32().contiguous();
        Tensor sin_work = (sin_freqs.dtype() == DType::Float32)
            ? sin_freqs.contiguous()
            : sin_freqs.to_f32().contiguous();
        rotary_time_split_contig_k_only_inplace_kernel<float>
            <<<(int)ceildiv(total, kTile), 1>>>(
                k.data_bf16(),
                cos_work.data_f32(), sin_work.data_f32(),
                total, full_bf16_arith_enabled());
    }
    CUDA_CHECK(cudaGetLastError());
}

bool try_linear_qkv_freq_rotary_cutile_fused(const Tensor& x,
                                             const Tensor& weight,
                                             int heads,
                                             int dim_head,
                                             const Tensor& cos_freqs,
                                             const Tensor& sin_freqs,
                                             Tensor& q,
                                             Tensor& k,
                                             Tensor& v) {
    if (!qkv_freq_rotary_cutile_fused_enabled()) return false;
    if (x.dtype() != DType::BFloat16 || weight.dtype() != DType::BFloat16) return false;
    if (x.ndim() != 3 || weight.ndim() != 2) return false;
    if (heads != kQkvFusedHeads || dim_head != kFreqAttnD) return false;
    if (x.size(1) != kFreqAttnN || x.size(2) != 256) return false;
    if (weight.size(0) != 3LL * kQkvFusedHeads * kFreqAttnD || weight.size(1) != 256) {
        return false;
    }

    int64_t batches = x.size(0);
    int64_t total_batch = batches * kFreqAttnN;
    if (total_batch != kLinearCutileExpectedM) return false;
    if (batches <= 0 || batches > std::numeric_limits<int>::max()) return false;

    Tensor x_work = x.contiguous();
    Tensor weight_work = weight.contiguous();
    bool trig_bf16 = cos_freqs.dtype() == DType::BFloat16 &&
                     sin_freqs.dtype() == DType::BFloat16;
    Tensor cos_work = trig_bf16 ? cos_freqs.contiguous()
                                : ((cos_freqs.dtype() == DType::Float32)
                                       ? cos_freqs.contiguous()
                                       : cos_freqs.to_f32().contiguous());
    Tensor sin_work = trig_bf16 ? sin_freqs.contiguous()
                                : ((sin_freqs.dtype() == DType::Float32)
                                       ? sin_freqs.contiguous()
                                       : sin_freqs.to_f32().contiguous());

    q = Tensor::empty({batches, heads, kFreqAttnPadN, dim_head}, DType::BFloat16);
    k = Tensor::empty({batches, heads, kFreqAttnPadN, dim_head}, DType::BFloat16);
    v = Tensor::empty({batches, heads, kFreqAttnPadN, dim_head}, DType::BFloat16);

    constexpr int kInFeatures = 256;
    constexpr int kPairs = kFreqAttnD / 2;
    constexpr int kQkvFreqFusedTileM = 16;
    dim3 full_grid(kLinearCutileStaticM / kQkvFreqFusedTileM,
                   3 * kQkvFusedHeads,
                   1);
    if (trig_bf16) {
        qkv_freq60_rotary_pad64_cutile_static_full_kernel<kQkvFreqFusedTileM,
                                                          kPairs,
                                                          kLinearCutileTileK,
                                                          kLinearCutileStaticM,
                                                          kInFeatures,
                                                          __nv_bfloat16>
            <<<full_grid, 1>>>(
                x_work.data_bf16(), weight_work.data_bf16(),
                cos_work.data_bf16(), sin_work.data_bf16(),
                q.data_bf16(), k.data_bf16(), v.data_bf16(),
                full_bf16_arith_enabled());
    } else {
        qkv_freq60_rotary_pad64_cutile_static_full_kernel<kQkvFreqFusedTileM,
                                                          kPairs,
                                                          kLinearCutileTileK,
                                                          kLinearCutileStaticM,
                                                          kInFeatures,
                                                          float>
            <<<full_grid, 1>>>(
                x_work.data_bf16(), weight_work.data_bf16(),
                cos_work.data_f32(), sin_work.data_f32(),
                q.data_bf16(), k.data_bf16(), v.data_bf16(),
                full_bf16_arith_enabled());
    }

    int tail_rows = (int)(total_batch - kLinearCutileStaticM);
    if (tail_rows > 0) {
        dim3 tail_grid((unsigned int)ceildiv((long long)tail_rows * kPairs, kTile),
                       3 * kQkvFusedHeads,
                       1);
        if (trig_bf16) {
            qkv_freq60_rotary_pad64_cutile_tail_kernel<kInFeatures, __nv_bfloat16>
                <<<tail_grid, 1>>>(
                    x_work.data_bf16(), weight_work.data_bf16(),
                    cos_work.data_bf16(), sin_work.data_bf16(),
                    q.data_bf16(), k.data_bf16(), v.data_bf16(),
                    (int)total_batch, kLinearCutileStaticM,
                    full_bf16_arith_enabled());
        } else {
            qkv_freq60_rotary_pad64_cutile_tail_kernel<kInFeatures, float>
                <<<tail_grid, 1>>>(
                    x_work.data_bf16(), weight_work.data_bf16(),
                    cos_work.data_f32(), sin_work.data_f32(),
                    q.data_bf16(), k.data_bf16(), v.data_bf16(),
                    (int)total_batch, kLinearCutileStaticM,
                    full_bf16_arith_enabled());
        }
    }

    long long pad_total = batches * (long long)heads *
                          (kFreqAttnPadN - kFreqAttnN) * dim_head;
    qkv_freq60_pad_rows_zero_kernel<<<(int)ceildiv(pad_total, kTile), 1>>>(
        q.data_bf16(), k.data_bf16(), v.data_bf16(), pad_total, (int)batches);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool try_linear_qkv_freq_rotary_bkn_cutile_fused(const Tensor& x,
                                                 const Tensor& weight,
                                                 const Tensor& weight_bkn,
                                                 int heads,
                                                 int dim_head,
                                                 const Tensor& cos_freqs,
                                                 const Tensor& sin_freqs,
                                                 Tensor& q,
                                                 Tensor& k,
                                                 Tensor& v) {
    if (!qkv_freq_rotary_bkn_cutile_fused_enabled()) return false;
    if (x.dtype() != DType::BFloat16 || weight.dtype() != DType::BFloat16 ||
        weight_bkn.dtype() != DType::BFloat16) {
        return false;
    }
    if (x.ndim() != 3 || weight.ndim() != 2 || weight_bkn.ndim() != 2) return false;
    if (heads != kQkvFusedHeads || dim_head != kFreqAttnD) return false;
    if (x.size(1) != kFreqAttnN || x.size(2) != 256) return false;

    constexpr int kOutFeatures = 3 * kQkvFusedHeads * kFreqAttnD;
    constexpr int kInFeatures = 256;
    if (weight.size(0) != kOutFeatures || weight.size(1) != kInFeatures) {
        return false;
    }
    if (weight_bkn.size(0) != kInFeatures || weight_bkn.size(1) != kOutFeatures) {
        return false;
    }

    int64_t batches = x.size(0);
    int64_t total_batch = batches * kFreqAttnN;
    if (total_batch != kLinearCutileExpectedM) return false;
    if (batches <= 0 || batches > std::numeric_limits<int>::max()) return false;

    Tensor x_work = x.contiguous();
    Tensor weight_bkn_work = weight_bkn.contiguous();
    bool trig_bf16 = cos_freqs.dtype() == DType::BFloat16 &&
                     sin_freqs.dtype() == DType::BFloat16;
    Tensor cos_work = trig_bf16 ? cos_freqs.contiguous()
                                : ((cos_freqs.dtype() == DType::Float32)
                                       ? cos_freqs.contiguous()
                                       : cos_freqs.to_f32().contiguous());
    Tensor sin_work = trig_bf16 ? sin_freqs.contiguous()
                                : ((sin_freqs.dtype() == DType::Float32)
                                       ? sin_freqs.contiguous()
                                       : sin_freqs.to_f32().contiguous());

    q = Tensor::empty({batches, heads, kFreqAttnPadN, dim_head}, DType::BFloat16);
    k = Tensor::empty({batches, heads, kFreqAttnPadN, dim_head}, DType::BFloat16);
    v = Tensor::empty({batches, heads, kFreqAttnPadN, dim_head}, DType::BFloat16);

    int tile_n = qkv_freq_rotary_bkn_cutile_tile_n();
    if (tile_n == 64) {
        dim3 full_grid(kLinearCutileStaticM / 32,
                       kOutFeatures / 64,
                       1);
        if (trig_bf16) {
            qkv_freq60_rotary_pad64_bkn_cutile_static_full_kernel<32,
                                                                  64,
                                                                  16,
                                                                  kLinearCutileStaticM,
                                                                  kOutFeatures,
                                                                  kInFeatures,
                                                                  __nv_bfloat16>
                <<<full_grid, 1>>>(
                    x_work.data_bf16(), weight_bkn_work.data_bf16(),
                    cos_work.data_bf16(), sin_work.data_bf16(),
                    q.data_bf16(), k.data_bf16(), v.data_bf16(),
                    full_bf16_arith_enabled());
        } else {
            qkv_freq60_rotary_pad64_bkn_cutile_static_full_kernel<32,
                                                                  64,
                                                                  16,
                                                                  kLinearCutileStaticM,
                                                                  kOutFeatures,
                                                                  kInFeatures,
                                                                  float>
                <<<full_grid, 1>>>(
                    x_work.data_bf16(), weight_bkn_work.data_bf16(),
                    cos_work.data_f32(), sin_work.data_f32(),
                    q.data_bf16(), k.data_bf16(), v.data_bf16(),
                    full_bf16_arith_enabled());
        }
    } else if (tile_n == 128) {
        dim3 full_grid(kLinearCutileStaticM / 32,
                       kOutFeatures / 128,
                       1);
        if (trig_bf16) {
            qkv_freq60_rotary_pad64_bkn_cutile_static_full_kernel<32,
                                                                  128,
                                                                  16,
                                                                  kLinearCutileStaticM,
                                                                  kOutFeatures,
                                                                  kInFeatures,
                                                                  __nv_bfloat16>
                <<<full_grid, 1>>>(
                    x_work.data_bf16(), weight_bkn_work.data_bf16(),
                    cos_work.data_bf16(), sin_work.data_bf16(),
                    q.data_bf16(), k.data_bf16(), v.data_bf16(),
                    full_bf16_arith_enabled());
        } else {
            qkv_freq60_rotary_pad64_bkn_cutile_static_full_kernel<32,
                                                                  128,
                                                                  16,
                                                                  kLinearCutileStaticM,
                                                                  kOutFeatures,
                                                                  kInFeatures,
                                                                  float>
                <<<full_grid, 1>>>(
                    x_work.data_bf16(), weight_bkn_work.data_bf16(),
                    cos_work.data_f32(), sin_work.data_f32(),
                    q.data_bf16(), k.data_bf16(), v.data_bf16(),
                    full_bf16_arith_enabled());
        }
    } else {
        dim3 full_grid(kLinearCutileStaticM / 32,
                       kOutFeatures / 256,
                       1);
        if (trig_bf16) {
            qkv_freq60_rotary_pad64_bkn_cutile_static_full_kernel<32,
                                                                  256,
                                                                  16,
                                                                  kLinearCutileStaticM,
                                                                  kOutFeatures,
                                                                  kInFeatures,
                                                                  __nv_bfloat16>
                <<<full_grid, 1>>>(
                    x_work.data_bf16(), weight_bkn_work.data_bf16(),
                    cos_work.data_bf16(), sin_work.data_bf16(),
                    q.data_bf16(), k.data_bf16(), v.data_bf16(),
                    full_bf16_arith_enabled());
        } else {
            qkv_freq60_rotary_pad64_bkn_cutile_static_full_kernel<32,
                                                                  256,
                                                                  16,
                                                                  kLinearCutileStaticM,
                                                                  kOutFeatures,
                                                                  kInFeatures,
                                                                  float>
                <<<full_grid, 1>>>(
                    x_work.data_bf16(), weight_bkn_work.data_bf16(),
                    cos_work.data_f32(), sin_work.data_f32(),
                    q.data_bf16(), k.data_bf16(), v.data_bf16(),
                    full_bf16_arith_enabled());
        }
    }

    int tail_rows = (int)(total_batch - kLinearCutileStaticM);
    if (tail_rows > 0) {
        dim3 tail_grid((unsigned int)ceildiv((long long)tail_rows * (kFreqAttnD / 2), kTile),
                       3 * kQkvFusedHeads,
                       1);
        if (trig_bf16) {
            qkv_freq60_rotary_pad64_bkn_cutile_tail_kernel<kInFeatures,
                                                           kOutFeatures,
                                                           __nv_bfloat16>
                <<<tail_grid, 1>>>(
                    x_work.data_bf16(), weight_bkn_work.data_bf16(),
                    cos_work.data_bf16(), sin_work.data_bf16(),
                    q.data_bf16(), k.data_bf16(), v.data_bf16(),
                    (int)total_batch, kLinearCutileStaticM,
                    full_bf16_arith_enabled());
        } else {
            qkv_freq60_rotary_pad64_bkn_cutile_tail_kernel<kInFeatures,
                                                           kOutFeatures,
                                                           float>
                <<<tail_grid, 1>>>(
                    x_work.data_bf16(), weight_bkn_work.data_bf16(),
                    cos_work.data_f32(), sin_work.data_f32(),
                    q.data_bf16(), k.data_bf16(), v.data_bf16(),
                    (int)total_batch, kLinearCutileStaticM,
                    full_bf16_arith_enabled());
        }
    }

    long long pad_total = batches * (long long)heads *
                          (kFreqAttnPadN - kFreqAttnN) * dim_head;
    qkv_freq60_pad_rows_zero_kernel<<<(int)ceildiv(pad_total, kTile), 1>>>(
        q.data_bf16(), k.data_bf16(), v.data_bf16(), pad_total, (int)batches);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

void linear_qkv_rotary_bf16_output(const Tensor& x, const Tensor& weight,
                                   int heads, int dim_head,
                                   const Tensor& cos_freqs, const Tensor& sin_freqs,
                                   Tensor& q, Tensor& k, Tensor& v) {
    if (try_linear_qkv_freq_rotary_cutile_fused(
            x, weight, heads, dim_head, cos_freqs, sin_freqs, q, k, v)) {
        return;
    }
    Tensor qkv = linear_no_bias_bf16_output(x, weight);
    split_qkv_heads_rotary(qkv, heads, dim_head, cos_freqs, sin_freqs, q, k, v);
}

void linear_qkv_rotary_bf16_output_bkn(const Tensor& x,
                                       const Tensor& weight,
                                       const Tensor& weight_bkn,
                                       int heads,
                                       int dim_head,
                                       const Tensor& cos_freqs,
                                       const Tensor& sin_freqs,
                                       Tensor& q,
                                       Tensor& k,
                                       Tensor& v) {
    if (try_linear_qkv_freq_rotary_bkn_cutile_fused(
            x, weight, weight_bkn, heads, dim_head, cos_freqs, sin_freqs, q, k, v)) {
        return;
    }
    if (!linear_bkn_long_path_enabled()) {
        linear_qkv_rotary_bf16_output(
            x, weight, heads, dim_head, cos_freqs, sin_freqs, q, k, v);
        return;
    }
    Tensor qkv = linear_no_bias_bkn(x, weight, weight_bkn);
    split_qkv_heads_rotary(qkv, heads, dim_head, cos_freqs, sin_freqs, q, k, v);
}

Tensor linear_sigmoid(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    int64_t total_batch = 0;
    int64_t out_features = 0;
    std::vector<int64_t> out_shape;
    bool direct_bf16 = linear_direct_bf16_output_enabled() && weight.dtype() == DType::BFloat16 &&
                       gate_sigmoid_bf16_enabled();
    if (direct_bf16) {
        int64_t in_features = weight.size(1);
        out_features = weight.size(0);
        Tensor xb = (x.dtype() == DType::BFloat16) ? x.contiguous() : x.to_bf16().contiguous();
        Tensor wb = weight.contiguous();
        Tensor bb = (bias.dtype() == DType::BFloat16) ? bias.contiguous() : bias.to_bf16().contiguous();
        total_batch = xb.numel() / in_features;
        out_shape = xb.shape();
        out_shape.back() = out_features;

        Tensor out;
        if (try_linear_cutile_static_gate_sigmoid_bf16_output(
                xb, wb, bb, total_batch, out_features, in_features, out)) {
            return out.reshape(out_shape);
        }
        unsupported_gemm_path("mbr_tile::linear_sigmoid");
    }

    Tensor out = linear_gemm_f32_output(x, weight, total_batch, out_features, out_shape);
    Tensor bf = (bias.dtype() == DType::Float32) ? bias.contiguous() : bias.to_f32().contiguous();

    long long total = total_batch * out_features;
    if (gate_sigmoid_bf16_enabled()) {
        Tensor out_bf16 = Tensor::empty({total_batch, out_features}, DType::BFloat16);
        add_bias_sigmoid_to_bf16_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            out.data_f32(), bf.data_f32(), out_bf16.data_bf16(), total, (int)out_features);
        CUDA_CHECK(cudaGetLastError());
        return out_bf16.reshape(out_shape);
    } else {
        add_bias_sigmoid_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            out.data_f32(), bf.data_f32(), total, (int)out_features);
        CUDA_CHECK(cudaGetLastError());
    }

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

    bool use_bf16_masks = mask_scatter_bf16_enabled();
    std::vector<Tensor> mask_work;
    mask_work.reserve(stem_masks.size());
    for (size_t i = 0; i < stem_masks.size(); ++i) {
        Tensor mask = use_bf16_masks
            ? ((stem_masks[i].dtype() == DType::BFloat16) ? stem_masks[i].contiguous()
                                                          : stem_masks[i].to_bf16().contiguous())
            : ((stem_masks[i].dtype() == DType::Float32) ? stem_masks[i].contiguous()
                                                         : stem_masks[i].to_f32().contiguous());
        if (mask.ndim() != 3 || mask.size(0) != batch || mask.size(1) != frames ||
            mask.size(2) != total_band_freqs * 2) {
            throw std::runtime_error("mbr_tile::apply_mask_and_scatter: expected [B, T, total_band_freqs * 2]");
        }
        mask_work.push_back(mask);
    }

    if (audio_channels < 1 || total_freq % audio_channels != 0) {
        throw std::runtime_error("mbr_tile::apply_mask_and_scatter: invalid audio channel count");
    }
    if (bands_per_freq.dtype() != DType::Float32 &&
        bands_per_freq.dtype() != DType::BFloat16) {
        throw std::runtime_error("mbr_tile::apply_mask_and_scatter: expected FP32/BF16 bands_per_freq");
    }
    Tensor bands_work = bands_per_freq.contiguous();
    bool bands_bf16 = bands_work.dtype() == DType::BFloat16;

    int64_t freq_bins = total_freq / audio_channels;
    Tensor out = Tensor::zeros({batch * num_stems * audio_channels, freq_bins, frames, 2});
    long long total = (long long)batch * num_stems * total_band_freqs * frames * 2;
    if (use_bf16_masks) {
        const __nv_bfloat16* mask0 = mask_work[0].data_bf16();
        const __nv_bfloat16* mask1 = (num_stems == 2) ? mask_work[1].data_bf16() : mask_work[0].data_bf16();
        if (bands_bf16) {
            apply_mask_and_scatter_bf16_kernel<__nv_bfloat16>
                <<<(int)ceildiv(total, kTile), 1>>>(
                    stft_repr.data_f32(), mask0, mask1, freq_indices.data_i64(),
                    bands_work.data_bf16(), out.data_f32(), total, (int)num_stems,
                    (int)total_band_freqs, (int)frames, (int)total_freq,
                    (int)audio_channels, (int)freq_bins, full_bf16_arith_enabled());
        } else {
            apply_mask_and_scatter_bf16_kernel<float>
                <<<(int)ceildiv(total, kTile), 1>>>(
                    stft_repr.data_f32(), mask0, mask1, freq_indices.data_i64(),
                    bands_work.data_f32(), out.data_f32(), total, (int)num_stems,
                    (int)total_band_freqs, (int)frames, (int)total_freq,
                    (int)audio_channels, (int)freq_bins, full_bf16_arith_enabled());
        }
    } else {
        const float* mask0 = mask_work[0].data_f32();
        const float* mask1 = (num_stems == 2) ? mask_work[1].data_f32() : mask_work[0].data_f32();
        if (bands_bf16) {
            apply_mask_and_scatter_kernel<__nv_bfloat16>
                <<<(int)ceildiv(total, kTile), 1>>>(
                    stft_repr.data_f32(), mask0, mask1, freq_indices.data_i64(),
                    bands_work.data_bf16(), out.data_f32(), total, (int)num_stems,
                    (int)total_band_freqs, (int)frames, (int)total_freq,
                    (int)audio_channels, (int)freq_bins);
        } else {
            apply_mask_and_scatter_kernel<float>
                <<<(int)ceildiv(total, kTile), 1>>>(
                    stft_repr.data_f32(), mask0, mask1, freq_indices.data_i64(),
                    bands_work.data_f32(), out.data_f32(), total, (int)num_stems,
                    (int)total_band_freqs, (int)frames, (int)total_freq,
                    (int)audio_channels, (int)freq_bins);
        }
    }
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
    if (g_quantize_bf16 && tanh_bf16_input_enabled() && x.dtype() == DType::BFloat16) {
        Tensor xc = x.contiguous();
        Tensor out = Tensor::empty(xc.shape(), DType::BFloat16);
        tanh_bf16_to_bf16_kernel<<<(int)ceildiv(xc.numel(), kTile), 1>>>(
            xc.data_bf16(), out.data_bf16(), xc.numel(), full_bf16_arith_enabled());
        CUDA_CHECK(cudaGetLastError());
        return out;
    }

    Tensor xf = (x.dtype() == DType::Float32) ? x.contiguous() : x.to_f32().contiguous();
    long long total = xf.numel();
    if (g_quantize_bf16) {
        Tensor out = Tensor::empty(xf.shape(), DType::BFloat16);
        tanh_to_bf16_kernel<<<(int)ceildiv(total, kTile), 1>>>(xf.data_f32(), out.data_bf16(), total);
        CUDA_CHECK(cudaGetLastError());
        return out;
    } else {
        Tensor out = Tensor::empty(xf.shape(), DType::Float32);
        tanh_kernel<<<(int)ceildiv(total, kTile), 1>>>(xf.data_f32(), out.data_f32(), total);
        CUDA_CHECK(cudaGetLastError());
        return (x.dtype() == DType::Float16) ? out.to_f16() : out;
    }
}

Tensor glu_last_dim(const Tensor& x) {
    if (glu_bf16_input_enabled() && glu_bf16_output_enabled() && x.dtype() == DType::BFloat16) {
        Tensor xb = x.contiguous();
        int ndim = xb.ndim();
        int64_t full_dim = xb.size(ndim - 1);
        if ((full_dim % 2) != 0) {
            throw std::runtime_error("glu_last_dim: last dimension must be even");
        }
        std::vector<int64_t> out_shape = xb.shape();
        out_shape.back() = full_dim / 2;
        Tensor out = Tensor::empty(out_shape, DType::BFloat16);
        long long total = out.numel();
        glu_last_dim_bf16_to_bf16_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            xb.data_bf16(), out.data_bf16(), total, (int)(full_dim / 2),
            full_bf16_arith_enabled());
        CUDA_CHECK(cudaGetLastError());
        return out;
    }

    Tensor xf = (x.dtype() == DType::Float32) ? x.contiguous() : x.to_f32().contiguous();
    int ndim = xf.ndim();
    int64_t full_dim = xf.size(ndim - 1);
    if ((full_dim % 2) != 0) {
        throw std::runtime_error("glu_last_dim: last dimension must be even");
    }

    std::vector<int64_t> out_shape = xf.shape();
    out_shape.back() = full_dim / 2;

    bool out_bf16 = glu_bf16_output_enabled();
    Tensor out = Tensor::empty(out_shape, out_bf16 ? DType::BFloat16 : DType::Float32);
    long long total = out.numel();
    if (out_bf16) {
        glu_last_dim_to_bf16_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            xf.data_f32(), out.data_bf16(), total, (int)(full_dim / 2));
    } else {
        glu_last_dim_kernel<<<(int)ceildiv(total, kTile), 1>>>(
            xf.data_f32(), out.data_f32(), total, (int)(full_dim / 2));
    }
    CUDA_CHECK(cudaGetLastError());
    return (x.dtype() == DType::Float16) ? out.to_f16() : out;
}

}  // namespace cudasep::mbr_tile
