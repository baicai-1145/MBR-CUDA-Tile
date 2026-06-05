#include "cuda_tile.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ct = cuda::tiles;
using namespace ct::literals;

#define CUDA_CHECK(call)                                                             \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            throw std::runtime_error(std::string(#call) + " failed: " +             \
                                     cudaGetErrorString(err__));                    \
        }                                                                           \
    } while (0)

constexpr int kN = 1301;
constexpr int kNPad = 1344;
constexpr int kNMain = 1280;
constexpr int kD = 64;
constexpr int kBH = 480;
constexpr int kHeads = 8;
constexpr int kBatches = kBH / kHeads;
constexpr int kQkvStride = 3 * kHeads * kD;
constexpr int kKTile = 64;
constexpr int kInitTile = 256;
constexpr float kLog2E = 1.44269504088896340736f;

struct Options {
    int warmup = 1;
    int iters = 3;
    bool validate = false;
    bool compare_baseline = false;
    bool describe = false;
    std::string variant = "all";
};

int parse_int_arg(const char* name, const char* value) {
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > 1000000L) {
        throw std::runtime_error(std::string("invalid value for ") + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--warmup") == 0) {
            opts.warmup = parse_int_arg(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            opts.iters = parse_int_arg(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--variant") == 0) {
            opts.variant = need_value(argv[i]);
        } else if (std::strcmp(argv[i], "--compare-baseline") == 0) {
            opts.compare_baseline = true;
        } else if (std::strcmp(argv[i], "--validate") == 0) {
            opts.validate = true;
        } else if (std::strcmp(argv[i], "--describe") == 0) {
            opts.describe = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: bench_bf16_time_attention_cutile [options]\n"
                "  --warmup N  warmup launches, default 1\n"
                "  --iters N   measured launches, default 3\n"
                "  --variant NAME  run one focused variant, default all\n"
                "                  main1280_q32k32_exp2,\n"
                "                  main1280_q64k16_exp2,\n"
                "                  main1280_q64k32_exp2,\n"
                "                  main1280_q64k64_exp2,\n"
                "                  main1280_q128k32_exp2,\n"
                "                  main1280_q64k32_exp2_sum_bf16,\n"
                "                  main1280_q64k32_exp2_score_bf16,\n"
                "                  main1280_q64k32_exp2_prescale_q,\n"
                "                  main1280_q64k32_exp2_split_d32,\n"
                "                  main1280_q64k32_exp2_qkv_direct_rotary,\n"
                "                  main1280_q64k32_exp2_split_contig_input,\n"
                "                  main1280_q64k32_exp2_split_contig_kt_input,\n"
                "                  main1280_q64k32_exp2_split_contig_kt_input_no_keytail,\n"
                "                  main1280_q64k32_exp2_split_contig_tail_idx32,\n"
                "                  main1280_q64k32_exp2_split_contig_tail_col_broadcast,\n"
                "                  main1280_q64k32_exp2_split_contig_tail_col_broadcast_out_acc_bf16,\n"
                "                  main1280_q64k32_exp2_split_contig_alpha_one,\n"
                "                  main1280_q64k32_exp2_split_contig_prob_linear,\n"
                "                  main1280_q64k32_exp2_split_contig_prob_linear_noclamp,\n"
                "                  main1280_q64k32_exp2_split_contig_prob_poly2,\n"
                "                  main1280_q64k32_exp2_split_contig_prob_poly4,\n"
                "                  main1280_q64k32_exp2_split_contig_prob_poly2_noclamp,\n"
                "                  main1280_q64k32_exp2_split_contig_prob_poly3_noclamp,\n"
                "                  main1280_q64k32_exp2_split_contig_prob_poly3_noclamp_alpha_poly3_clamp,\n"
                "                  main1280_q64k32_exp2_split_contig_prob_poly3_noclamp_alpha_poly3_clamp_final_rcp,\n"
                "                  main1280_q64k32_exp2_split_contig_prob_poly3_outclamp_alpha_outclamp,\n"
                "                  main1280_q64k32_exp2_split_contig_prob_poly3_noclamp_alpha_linear_clamp,\n"
                "                  main1280_q64k32_exp2_split_contig_prob_rational4_alpha_rational4,\n"
                "                  main1280_q64k32_exp2_split_contig_prob_poly4_noclamp,\n"
                "                  main1280_q64k32_exp2_split_contig_prob_poly4_noclamp_bias,\n"
                "                  main1280_q64k32_exp2_split_contig_prob_poly4_noclamp_sum_bf16,\n"
                "                  main1280_q64k32_exp2_split_contig_tail_prob_mask_only,\n"
                "                  main1280_q64k32_exp2_split_contig_tail16_8,\n"
                "                  main1280_q64k32_exp2_split_contig_tail_helper,\n"
                "                  main1280_q64k32_exp2_split_contig_seg2,\n"
                "                  main1280_q64k32_exp2_split_contig_seg2_no_keytail,\n"
                "                  main1280_q64k32_exp2_split_contig_tail_first,\n"
                "                  main1280_q64k32_exp2_split_contig_tail_first_padded_tail_load,\n"
                "                  main1280_q64k32_exp2_split_contig_padded_tail_load,\n"
                "                  main1280_q64k32_exp2_split_contig_tail_col_broadcast_padded_tail_load,\n"
                "                  main1280_q64k32_exp2_split_contig_two_pass_state,\n"
                "                  main1280_q64k32_exp2_split_contig_score_av_lb,\n"
                "                  main1280_q64k32_exp2_split_contig_score_av_lb_no_keytail,\n"
                "                  main1280_q64k32_exp2_split_contig_tile_local_softmax_lb,\n"
                "                  main1280_q64k32_exp2_split_contig_tile_local_softmax_lb_no_keytail,\n"
                "                  main1280_q64k32_exp2_split_contig_final_rcp,\n"
                "                  main1280_q64k32_exp2_split_contig_final_rcp_no_keytail,\n"
                "                  main1280_q64k32_exp2_split_contig_lat2,\n"
                "                  main1280_q64k32_exp2_split_contig_lat2_no_keytail,\n"
                "                  main1280_q64k32_exp2_split_contig_q_lat2,\n"
                "                  main1280_q64k32_exp2_split_contig_q_lat2_no_keytail,\n"
                "                  main1280_q64k32_exp2_split_contig_k_lat2,\n"
                "                  main1280_q64k32_exp2_split_contig_k_lat2_no_keytail,\n"
                "                  main1280_q64k32_exp2_split_contig_v_lat2,\n"
                "                  main1280_q64k32_exp2_split_contig_v_lat2_no_keytail,\n"
                "                  main1280_q64k32_exp2_split_contig_kv_lat2,\n"
                "                  main1280_q64k32_exp2_split_contig_kv_lat2_no_keytail,\n"
                "                  main1280_q64k32_exp2_split_contig_row_l_bf16,\n"
                "                  main1280_q64k32_exp2_split_contig_row_l_bf16_no_keytail,\n"
                "                  main1280_q64k32_exp2_split_contig_row_state_bf16,\n"
                "                  main1280_q64k32_exp2_split_contig_row_state_bf16_no_keytail,\n"
                "                  main1280_q64k32_exp2_split_contig_first_init,\n"
                "                  main1280_q64k32_exp2_split_contig_first_init_tail_col_broadcast_padded_tail_load,\n"
                "                  main1280_q64k32_exp2_split_contig_first_init_no_keytail,\n"
                "                  main1280_q64k32_exp2_split_contig_out_acc_bf16,\n"
                "                  main1280_q64k32_exp2_split_contig_out_acc_bf16_no_keytail,\n"
                "                  main1280_q64k32_exp2_split_contig_gated_store,\n"
                "                  main1280_q64k32_exp2_split_contig_input_no_keytail,\n"
                "                  split_tail_q64k32_tail32k32_exp2\n"
                "  --compare-baseline  compare focused output against q64/k32 exp2;\n"
                "                      gated_store compares against attention+gate-merge\n"
                "  --validate  compare several BH0 rows against CPU reference\n"
                "  --describe  print CUDA runtime resource/occupancy diagnostics\n");
            std::exit(0);
        } else {
            throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
        }
    }
    return opts;
}

int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

template <bool UseExp2, typename TileT>
static __tile__ auto softmax_exp(TileT x) {
    if constexpr (UseExp2) {
        return ct::exp2(x * kLog2E);
    }
    return ct::exp(x);
}

enum ProbMode : int {
    kProbExp = 0,
    kProbLinear = 1,
    kProbPoly4 = 2,
    kProbPoly2 = 3,
    kProbPoly4NoClamp = 4,
    kProbPoly2NoClamp = 5,
    kProbLinearNoClamp = 6,
    kProbPoly4NoClampBias = 7,
    kProbPoly3NoClamp = 8,
    kProbPoly3OutputClamp = 9,
    kProbRational4 = 10,
};

enum AlphaMode : int {
    kAlphaExact = 0,
    kAlphaProbClamp = 1,
    kAlphaLinearClamp = 2,
};

template <bool UseExp2, int Prob, typename TileT>
static __tile__ auto softmax_prob(TileT x) {
    if constexpr (Prob == kProbLinearNoClamp) {
        return x * 0.25f + 1.0f;
    } else if constexpr (Prob == kProbPoly3NoClamp) {
        auto t = x * 0.333333343f + 1.0f;
        auto t2 = t * t;
        return t2 * t;
    } else if constexpr (Prob == kProbPoly3OutputClamp) {
        auto zero = x * 0.0f;
        auto t = x * 0.333333343f + 1.0f;
        auto p = t * t * t;
        return ct::select(p > zero, p, zero);
    } else if constexpr (Prob == kProbRational4) {
        auto t = 1.0f / (1.0f - x * 0.25f);
        auto t2 = t * t;
        return t2 * t2;
    } else if constexpr (Prob == kProbPoly2NoClamp) {
        auto t = x * 0.5f + 1.0f;
        return t * t;
    } else if constexpr (Prob == kProbPoly4NoClamp) {
        auto t = x * 0.25f + 1.0f;
        auto t2 = t * t;
        return t2 * t2;
    } else if constexpr (Prob == kProbPoly2) {
        auto zero = x * 0.0f;
        auto t = x * 0.5f + 1.0f;
        t = ct::select(t > zero, t, zero);
        return t * t;
    } else if constexpr (Prob == kProbPoly4) {
        auto zero = x * 0.0f;
        auto t = x * 0.25f + 1.0f;
        t = ct::select(t > zero, t, zero);
        auto t2 = t * t;
        return t2 * t2;
    } else if constexpr (Prob == kProbLinear) {
        auto zero = x * 0.0f;
        auto y = x * 0.25f + 1.0f;
        return ct::select(y > zero, y, zero);
    }
    return softmax_exp<UseExp2>(x);
}

template <bool UseExp2, int Prob, typename TileT>
static __tile__ auto softmax_alpha_approx(TileT x) {
    auto zero = x * 0.0f;
    if constexpr (Prob == kProbPoly3NoClamp) {
        auto t = x * 0.333333343f + 1.0f;
        t = ct::select(t > zero, t, zero);
        auto t2 = t * t;
        return t2 * t;
    } else if constexpr (Prob == kProbRational4) {
        auto t = 1.0f / (1.0f - x * 0.25f);
        auto t2 = t * t;
        return t2 * t2;
    } else if constexpr (Prob == kProbPoly4NoClamp ||
                         Prob == kProbPoly4NoClampBias) {
        auto t = x * 0.25f + 1.0f;
        t = ct::select(t > zero, t, zero);
        auto t2 = t * t;
        return t2 * t2;
    } else if constexpr (Prob == kProbPoly2NoClamp) {
        auto t = x * 0.5f + 1.0f;
        t = ct::select(t > zero, t, zero);
        return t * t;
    } else if constexpr (Prob == kProbLinearNoClamp) {
        auto y = x * 0.25f + 1.0f;
        return ct::select(y > zero, y, zero);
    }
    return softmax_prob<UseExp2, Prob>(x);
}

template <bool UseExp2, int Prob, int Alpha, typename TileT>
static __tile__ auto softmax_alpha(TileT x) {
    if constexpr (Alpha == kAlphaProbClamp) {
        return softmax_alpha_approx<UseExp2, Prob>(x);
    } else if constexpr (Alpha == kAlphaLinearClamp) {
        auto zero = x * 0.0f;
        auto y = x + 1.0f;
        return ct::select(y > zero, y, zero);
    }
    return softmax_exp<UseExp2>(x);
}

template <bool UseExp2,
          int Prob,
          typename ScoreTile,
          typename RowTile>
static __tile__ auto softmax_probs_from_scores(ScoreTile scores, RowTile new_m) {
    if constexpr (Prob == kProbPoly4NoClampBias) {
        auto t = scores * 0.25f + (1.0f - new_m * 0.25f);
        auto t2 = t * t;
        return t2 * t2;
    }
    return softmax_prob<UseExp2, Prob>(scores - new_m);
}

template <bool ProbSumBf16, typename ProbF32Tile, typename ProbBf16Tile>
static __tile__ auto softmax_tile_sum(ProbF32Tile probs_f32,
                                      ProbBf16Tile probs_bf16) {
    if constexpr (ProbSumBf16) {
        return ct::sum<1>(ct::element_cast<float>(probs_bf16));
    }
    return ct::sum<1>(probs_f32);
}

template <typename RowTile, typename OutTile>
struct TailMergeState {
    RowTile row_l;
    OutTile out_acc;
};

float percentile(std::vector<float> values, float q) {
    std::sort(values.begin(), values.end());
    float pos = q * static_cast<float>(values.size() - 1);
    int lo = static_cast<int>(pos);
    int hi = std::min(lo + 1, static_cast<int>(values.size() - 1));
    float t = pos - static_cast<float>(lo);
    return values[lo] * (1.0f - t) + values[hi] * t;
}

__tile_global__ void fill_bf16_kernel(__nv_bfloat16* __restrict__ dst, long long total) {
    using I64Tile = ct::tile<long long, ct::shape<kInitTile>>;
    using F32Tile = ct::tile<float, ct::shape<kInitTile>>;
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    F32Tile values =
        0.125f + ct::element_cast<float>((idx * 17LL) & 1023LL) * 0.000244140625f;
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

__tile_global__ void fill_gate_bf16_kernel(__nv_bfloat16* __restrict__ dst,
                                           long long total) {
    using I64Tile = ct::tile<long long, ct::shape<kInitTile>>;
    using F32Tile = ct::tile<float, ct::shape<kInitTile>>;
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    F32Tile values =
        0.25f + ct::element_cast<float>((idx * 13LL) & 255LL) * 0.001953125f;
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

__tile_global__ void transpose_k_nd_to_dn_kernel(const __nv_bfloat16* __restrict__ src,
                                                 __nv_bfloat16* __restrict__ dst,
                                                 long long total) {
    using I64Tile = ct::tile<long long, ct::shape<256>>;
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    I64Tile idx = (long long)ct::bid().x * 256 + ct::iota<I64Tile>();
    auto valid = idx < total;
    auto d = idx % kD;
    auto n = (idx / kD) % kN;
    auto bh = idx / ((long long)kN * kD);
    auto dst_idx = (bh * kD + d) * kN + n;
    auto values = ct::load_masked(src + idx, valid);
    ct::store_masked(dst + dst_idx, values, valid);
}

__tile_global__ void fill_rotary_identity_kernel(float* __restrict__ cos_f,
                                                 float* __restrict__ sin_f,
                                                 long long total) {
    using I64Tile = ct::tile<long long, ct::shape<256>>;
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);

    I64Tile idx = (long long)ct::bid().x * 256 + ct::iota<I64Tile>();
    auto valid = idx < total;
    auto one = ct::element_cast<float>(idx * 0LL) + 1.0f;
    auto zero = one * 0.0f;
    ct::store_masked(cos_f + idx, one, valid);
    ct::store_masked(sin_f + idx, zero, valid);
}

__tile_global__ void pack_time_qkv_kernel(const __nv_bfloat16* __restrict__ q,
                                          const __nv_bfloat16* __restrict__ k,
                                          const __nv_bfloat16* __restrict__ v,
                                          __nv_bfloat16* __restrict__ qkv,
                                          long long total) {
    using I64Tile = ct::tile<long long, ct::shape<256>>;
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    qkv = ct::assume_aligned(qkv, 16_ic);

    I64Tile idx = (long long)ct::bid().x * 256 + ct::iota<I64Tile>();
    auto valid = idx < total;
    auto d = idx % kD;
    auto n = (idx / kD) % kN;
    auto bh = idx / ((long long)kN * kD);
    auto b = bh / kHeads;
    auto h = bh - b * kHeads;
    auto head_offset = h * kD + d;
    auto qkv_base = (b * kN + n) * kQkvStride;

    auto qv = ct::load_masked(q + idx, valid);
    auto kv = ct::load_masked(k + idx, valid);
    auto vv = ct::load_masked(v + idx, valid);
    ct::store_masked(qkv + qkv_base + head_offset, qv, valid);
    ct::store_masked(qkv + qkv_base + kHeads * kD + head_offset, kv, valid);
    ct::store_masked(qkv + qkv_base + 2LL * kHeads * kD + head_offset, vv, valid);
}

__tile_global__ void pack_time_split_contig_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ q_split,
    __nv_bfloat16* __restrict__ k_split,
    __nv_bfloat16* __restrict__ v_split,
    long long total) {
    using I64Tile = ct::tile<long long, ct::shape<256>>;
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    q_split = ct::assume_aligned(q_split, 16_ic);
    k_split = ct::assume_aligned(k_split, 16_ic);
    v_split = ct::assume_aligned(v_split, 16_ic);

    I64Tile idx = (long long)ct::bid().x * 256 + ct::iota<I64Tile>();
    auto valid = idx < total;
    auto d = idx % kD;
    auto n = (idx / kD) % kN;
    auto bh = idx / ((long long)kN * kD);
    auto b = bh / kHeads;
    auto h = bh - b * kHeads;
    auto dst = (b * kN + n) * (kHeads * kD) + h * kD + d;

    auto qv = ct::load_masked(q + idx, valid);
    auto kv = ct::load_masked(k + idx, valid);
    auto vv = ct::load_masked(v + idx, valid);
    ct::store_masked(q_split + dst, qv, valid);
    ct::store_masked(k_split + dst, kv, valid);
    ct::store_masked(v_split + dst, vv, valid);
}

__tile_global__ void pack_time_split_contig_k_transposed_kernel(
    const __nv_bfloat16* __restrict__ k_split,
    __nv_bfloat16* __restrict__ k_t,
    long long total) {
    using I64Tile = ct::tile<long long, ct::shape<256>>;
    k_split = ct::assume_aligned(k_split, 16_ic);
    k_t = ct::assume_aligned(k_t, 16_ic);

    I64Tile idx = (long long)ct::bid().x * 256 + ct::iota<I64Tile>();
    auto valid = idx < total;
    auto d = idx % kD;
    auto h = (idx / kD) % kHeads;
    auto n = (idx / ((long long)kD * kHeads)) % kN;
    auto b = idx / ((long long)kN * kHeads * kD);
    auto dst = ((b * kHeads + h) * kD + d) * kN + n;
    auto values = ct::load_masked(k_split + idx, valid);
    ct::store_masked(k_t + dst, values, valid);
}

__tile_global__ void pack_time_split_contig_padded_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ q_split,
    __nv_bfloat16* __restrict__ k_split,
    __nv_bfloat16* __restrict__ v_split,
    long long total) {
    using I64Tile = ct::tile<long long, ct::shape<256>>;
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    q_split = ct::assume_aligned(q_split, 16_ic);
    k_split = ct::assume_aligned(k_split, 16_ic);
    v_split = ct::assume_aligned(v_split, 16_ic);

    I64Tile idx = (long long)ct::bid().x * 256 + ct::iota<I64Tile>();
    auto valid = idx < total;
    auto d = idx % kD;
    auto h = (idx / kD) % kHeads;
    auto n = (idx / ((long long)kD * kHeads)) % kNPad;
    auto b = idx / ((long long)kNPad * kHeads * kD);
    auto src = ((b * kHeads + h) * kN + n) * kD + d;
    auto dst = (b * kNPad + n) * (kHeads * kD) + h * kD + d;
    auto src_valid = valid && (n < kN);

    auto qv = ct::load_masked(q + src, src_valid);
    auto kv = ct::load_masked(k + src, src_valid);
    auto vv = ct::load_masked(v + src, src_valid);
    auto zero = ct::element_cast<__nv_bfloat16>(ct::element_cast<float>(idx * 0LL));
    qv = ct::select(src_valid, qv, zero);
    kv = ct::select(src_valid, kv, zero);
    vv = ct::select(src_valid, vv, zero);
    ct::store_masked(q_split + dst, qv, valid);
    ct::store_masked(k_split + dst, kv, valid);
    ct::store_masked(v_split + dst, vv, valid);
}

__tile_global__ void gate_merge_time_main1280_token_d64_kernel(
    const __nv_bfloat16* __restrict__ attn,
    const __nv_bfloat16* __restrict__ gates,
    __nv_bfloat16* __restrict__ merged) {
    using I64Tile = ct::tile<long long, ct::shape<kHeads * kD>>;
    using F32Tile = ct::tile<float, ct::shape<kHeads * kD>>;

    attn = ct::assume_aligned(attn, 16_ic);
    gates = ct::assume_aligned(gates, 16_ic);
    merged = ct::assume_aligned(merged, 16_ic);

    int token = static_cast<int>(ct::bid().x);
    auto e = ct::iota<I64Tile>();
    int n = token % kNMain;
    int b = token / kNMain;
    auto h = e / kD;
    auto d = e % kD;

    auto src_idx = ((static_cast<long long>(b) * kHeads + h) * kN + n) * kD + d;
    auto gate_idx = (static_cast<long long>(b) * kN + n) * kHeads + h;
    auto dst_idx = static_cast<long long>(token) * kHeads * kD + e;

    F32Tile attn_values = ct::element_cast<float>(ct::load(attn + src_idx));
    F32Tile gate_values = ct::element_cast<float>(ct::load(gates + gate_idx));
    ct::store(merged + dst_idx,
              ct::element_cast<__nv_bfloat16>(attn_values * gate_values));
}

template <int QRows>
__tile_global__ void time_attention1301_cutile_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    using ScoreTile = ct::tile<float, ct::shape<QRows, kKTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, kKTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kNPad * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kNPad, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kNPad>{}, ct::layout_left{}},
        ct::shape<kD, kKTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kNPad, kD>{}},
        ct::shape<kKTile, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / kKTile;
    auto score_cols_local = score_local % kKTile;

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{kNPad / kKTile})) {
        auto key_cols = static_cast<long long>(kt) * kKTile + score_cols_local;
        auto valid = (score_rows < kN) && (key_cols < kN);
        ScoreTile scores = ct::mma(q_tile, k_t_view.load(0, kt),
                                   ct::full<ScoreTile>(0.0f));
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores * scale, neg_inf);

        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = ct::exp(row_m - new_m);
        auto probs_f32 = ct::select(valid, ct::exp(scores - new_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(ct::element_cast<float>(probs_bf16));

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16, v_view.load(kt, 0), ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    out_acc = out_acc / row_l;
    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_valid = out_rows < kN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows, int KTile>
__tile_global__ void time_attention1301_cutile_masked_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int KTiles = (kN + KTile - 1) / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };

    auto q_tile = q_view.load_masked(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / KTile;
    auto score_cols_local = score_local % KTile;

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{KTiles})) {
        auto key_cols = static_cast<long long>(kt) * KTile + score_cols_local;
        auto valid = (score_rows < kN) && (key_cols < kN);
        ScoreTile scores = ct::mma(q_tile, k_t_view.load_masked(0, kt),
                                   ct::full<ScoreTile>(0.0f));
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores * scale, neg_inf);

        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = ct::exp(row_m - new_m);
        auto probs_f32 = ct::select(valid, ct::exp(scores - new_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(ct::element_cast<float>(probs_bf16));

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16, v_view.load_masked(kt, 0), ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    out_acc = out_acc / row_l;
    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_valid = out_rows < kN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows, int KTile>
__tile_global__ void time_attention1301_cutile_masked_score_av_lb_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int KTiles = (kN + KTile - 1) / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };

    auto q_tile = q_view.load_masked(q_block, 0);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / KTile;
    auto score_cols_local = score_local % KTile;

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{KTiles})) {
        auto key_cols = static_cast<long long>(kt) * KTile + score_cols_local;
        auto valid = (score_rows < kN) && (key_cols < kN);
        ScoreTile scores = ct::mma(q_tile, k_t_view.load_masked(0, kt),
                                   ct::full<ScoreTile>(0.0f));
        auto score_values = ct::select(valid, scores * scale, scores * 0.0f);
        auto score_bf16 = ct::element_cast<__nv_bfloat16>(score_values);
        out_acc = out_acc + ct::mma(score_bf16,
                                    v_view.load_masked(kt, 0),
                                    ct::full<OutTile>(0.0f));
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_valid = out_rows < kN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows>
__tile_global__ void time_attention1301_main1280_av_const_kernel(
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out) {
    constexpr int KTile = 64;
    constexpr int FullKTiles = kNMain / KTile;
    using ProbTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ProbTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(ct::full<ProbTile>(0.125f));
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        out_acc = out_acc +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
    }

    I64ProbTile prob_local = ct::iota<I64ProbTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + prob_local % KTile;
    auto valid = key_cols < kN;
    auto tail_probs = ct::element_cast<__nv_bfloat16>(
        ct::select(valid, ct::full<ProbTile>(0.125f), ct::full<ProbTile>(0.0f)));
    out_acc = out_acc +
              ct::mma(tail_probs,
                      v_view.load_masked(FullKTiles, 0),
                      ct::full<OutTile>(0.0f));

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc), q_block, 0);
}

template <int QRows>
__tile_global__ void time_attention1301_main1280_qk_only_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int KTile = 64;
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    ScoreTile score_acc = ct::full<ScoreTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        score_acc = score_acc +
                    ct::mma(q_tile,
                            k_t_view.load(0, kt),
                            ct::full<ScoreTile>(0.0f)) * scale;
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto tail_scores = ct::mma(q_tile,
                               k_t_view.load_masked(0, FullKTiles),
                               ct::full<ScoreTile>(0.0f)) * scale;
    score_acc = score_acc + ct::select(valid, tail_scores, tail_scores * 0.0f);

    out_view.store(ct::element_cast<__nv_bfloat16>(score_acc), q_block, 0);
}

template <int QRows>
__tile_global__ void time_attention1301_main1280_qk_only_kt_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_t,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int KTile = 64;
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    q = ct::assume_aligned(q, 16_ic);
    k_t = ct::assume_aligned(k_t, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_t_batch =
        k_t + static_cast<std::size_t>(bh) * kD * kN;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_t_batch, ct::shape<kD, kN>{}},
        ct::shape<kD, KTile>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    ScoreTile score_acc = ct::full<ScoreTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        score_acc = score_acc +
                    ct::mma(q_tile,
                            k_t_view.load(0, kt),
                            ct::full<ScoreTile>(0.0f)) * scale;
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto tail_scores = ct::mma(q_tile,
                               k_t_view.load_masked(0, FullKTiles),
                               ct::full<ScoreTile>(0.0f)) * scale;
    score_acc = score_acc + ct::select(valid, tail_scores, tail_scores * 0.0f);

    out_view.store(ct::element_cast<__nv_bfloat16>(score_acc), q_block, 0);
}

template <int QRows>
__tile_global__ void time_attention1301_main1280_qk_store_p_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ p,
    float scale) {
    constexpr int KTile = 64;
    constexpr int QBlocks = kNMain / QRows;
    constexpr int FullKTiles = kNMain / KTile;
    constexpr int KTiles = (kN + KTile - 1) / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    p = ct::assume_aligned(p, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* p_block =
        p + ((static_cast<std::size_t>(bh) * QBlocks + q_block) *
             KTiles * QRows * KTile);

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    I64ScoreTile local = ct::iota<I64ScoreTile>();
    auto all_valid = local >= 0LL;

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        ct::store_masked(p_block + static_cast<std::size_t>(kt) * QRows * KTile + local,
                         ct::element_cast<__nv_bfloat16>(scores),
                         all_valid);
    }

    auto key_cols = static_cast<long long>(FullKTiles) * KTile + local % KTile;
    auto valid = key_cols < kN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f));
    auto p_values = ct::select(valid, scores * scale, scores * 0.0f);
    ct::store_masked(p_block + static_cast<std::size_t>(FullKTiles) * QRows * KTile + local,
                     ct::element_cast<__nv_bfloat16>(p_values),
                     all_valid);
}

template <int QRows>
__tile_global__ void time_attention1301_main1280_av_load_p_kernel(
    const __nv_bfloat16* __restrict__ p,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out) {
    constexpr int KTile = 64;
    constexpr int QBlocks = kNMain / QRows;
    constexpr int FullKTiles = kNMain / KTile;
    constexpr int KTiles = (kN + KTile - 1) / KTile;
    using ProbTile = ct::tile<__nv_bfloat16, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ProbTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    p = ct::assume_aligned(p, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* p_block =
        p + ((static_cast<std::size_t>(bh) * QBlocks + q_block) *
             KTiles * QRows * KTile);
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    I64ProbTile local = ct::iota<I64ProbTile>();
    auto all_valid = local >= 0LL;
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        ProbTile probs = ct::load_masked(
            p_block + static_cast<std::size_t>(kt) * QRows * KTile + local,
            all_valid);
        out_acc = out_acc +
                  ct::mma(probs,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
    }

    ProbTile probs = ct::load_masked(
        p_block + static_cast<std::size_t>(FullKTiles) * QRows * KTile + local,
        all_valid);
    out_acc = out_acc +
              ct::mma(probs,
                      v_view.load_masked(FullKTiles, 0),
                      ct::full<OutTile>(0.0f));

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc), q_block, 0);
}

__tile_global__ void time_attention1301_q64k64_main1280_score_av_lb_split_d32_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int QRows = 64;
    constexpr int KTile = 64;
    constexpr int DTile = 32;
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, DTile>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, DTile>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, DTile>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    OutTile out_acc0 = ct::full<OutTile>(0.0f);
    OutTile out_acc1 = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(scores);
        out_acc0 = out_acc0 +
                   ct::mma(probs_bf16,
                           v_view.load(kt, 0),
                           ct::full<OutTile>(0.0f));
        out_acc1 = out_acc1 +
                   ct::mma(probs_bf16,
                           v_view.load(kt, 1),
                           ct::full<OutTile>(0.0f));
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f));
    auto score_values = ct::select(valid, scores * scale, scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(score_values);
    out_acc0 = out_acc0 +
               ct::mma(probs_bf16,
                       v_view.load_masked(FullKTiles, 0),
                       ct::full<OutTile>(0.0f));
    out_acc1 = out_acc1 +
               ct::mma(probs_bf16,
                       v_view.load_masked(FullKTiles, 1),
                       ct::full<OutTile>(0.0f));

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc0), q_block, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc1), q_block, 1);
}

__tile_global__ void time_attention1301_q64k64_main1280_score_av_lb_prescale_q_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int QRows = 64;
    constexpr int KTile = 64;
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = ct::element_cast<__nv_bfloat16>(
        ct::element_cast<float>(q_view.load(q_block, 0)) * scale);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f));
        out_acc = out_acc +
                  ct::mma(ct::element_cast<__nv_bfloat16>(scores),
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f));
    auto score_values = ct::select(valid, scores, scores * 0.0f);
    out_acc = out_acc +
              ct::mma(ct::element_cast<__nv_bfloat16>(score_values),
                      v_view.load_masked(FullKTiles, 0),
                      ct::full<OutTile>(0.0f));

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc), q_block, 0);
}

template <int QRows, int KTile = 64, bool SumF32 = false, bool UseExp2 = false,
          bool ScoreBf16 = false, bool PrescaleQ = false>
__tile_global__ void time_attention1301_main1280_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    if constexpr (PrescaleQ) {
        q_tile = ct::element_cast<__nv_bfloat16>(
            ct::element_cast<float>(q_tile) * scale);
    }
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f));
        if constexpr (!PrescaleQ) {
            scores = scores * scale;
        }
        if constexpr (ScoreBf16) {
            scores = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(scores));
        }
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);
        if constexpr (!SumF32) {
            tile_l = ct::sum<1>(ct::element_cast<float>(probs_bf16));
        }

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f));
    if constexpr (!PrescaleQ) {
        scores = scores * scale;
    }
    if constexpr (ScoreBf16) {
        scores = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(scores));
    }
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(valid, scores, neg_inf);
    auto tile_m = ct::reduce_max<1>(scores);
    auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
    auto alpha = softmax_exp<UseExp2>(row_m - new_m);
    auto probs_f32 = ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto tile_l = ct::sum<1>(probs_f32);
    if constexpr (!SumF32) {
        tile_l = ct::sum<1>(ct::element_cast<float>(probs_bf16));
    }
    out_acc = out_acc * alpha +
              ct::mma(probs_bf16,
                      v_view.load_masked(FullKTiles, 0),
                      ct::full<OutTile>(0.0f));
    row_l = row_l * alpha + tile_l;

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

template <int QRows,
          int KTile,
          bool UseExp2 = true,
          bool IncludeKeyTail = true,
          bool TailIdx32 = false,
          bool UseFinalReciprocal = false,
          bool TailColBroadcast = false,
          bool RoundOutAcc = false,
          bool SkipAlphaExp = false,
          int Prob = kProbExp,
          bool ProbSumBf16 = false,
          int Alpha = kAlphaExact>
__tile_global__ void time_attention1301_main1280_split_contig_input_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64ColTile = ct::tile<long long, ct::shape<1, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kN, kD>;
    using DNShape = ct::shape<kD, kN>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kN * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = ct::full<RowTile>(1.0f);
        if constexpr (!SkipAlphaExp) {
            alpha = softmax_alpha<UseExp2, Prob, Alpha>(row_m - new_m);
        }
        auto probs_f32 =
            softmax_probs_from_scores<UseExp2, Prob>(scores, new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = softmax_tile_sum<ProbSumBf16>(probs_f32, probs_bf16);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        if constexpr (RoundOutAcc) {
            out_acc = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(out_acc));
        }
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    if constexpr (IncludeKeyTail) {
        if constexpr (TailColBroadcast) {
            I64ColTile col_local = ct::iota<I64ColTile>();
            auto key_cols = static_cast<long long>(FullKTiles) * KTile + col_local;
            auto valid = key_cols < kN;
            auto scores = ct::mma(q_tile,
                                  k_t_view.load_masked(0, FullKTiles),
                                  ct::full<ScoreTile>(0.0f)) * scale;
            auto neg_inf = scores * 0.0f - 3.402823466e38f;
            scores = ct::select(valid, scores, neg_inf);
            auto tile_m = ct::reduce_max<1>(scores);
            auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
            auto alpha = ct::full<RowTile>(1.0f);
            if constexpr (!SkipAlphaExp) {
                alpha = softmax_alpha<UseExp2, Prob, Alpha>(row_m - new_m);
            }
            auto probs_f32 =
                ct::select(valid,
                           softmax_probs_from_scores<UseExp2, Prob>(scores, new_m),
                           scores * 0.0f);
            auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
            auto tile_l = softmax_tile_sum<ProbSumBf16>(probs_f32, probs_bf16);
            out_acc = out_acc * alpha +
                      ct::mma(probs_bf16,
                              v_view.load_masked(FullKTiles, 0),
                              ct::full<OutTile>(0.0f));
            if constexpr (RoundOutAcc) {
                out_acc = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(out_acc));
            }
            row_l = row_l * alpha + tile_l;
        } else if constexpr (TailIdx32) {
            using I32ScoreTile = ct::tile<int, ct::shape<QRows, KTile>>;
            I32ScoreTile score_local = ct::iota<I32ScoreTile>();
            auto key_cols = FullKTiles * KTile + score_local % KTile;
            auto valid = key_cols < kN;
            auto scores = ct::mma(q_tile,
                                  k_t_view.load_masked(0, FullKTiles),
                                  ct::full<ScoreTile>(0.0f)) * scale;
            auto neg_inf = scores * 0.0f - 3.402823466e38f;
            scores = ct::select(valid, scores, neg_inf);
            auto tile_m = ct::reduce_max<1>(scores);
            auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
            auto alpha = ct::full<RowTile>(1.0f);
            if constexpr (!SkipAlphaExp) {
                alpha = softmax_alpha<UseExp2, Prob, Alpha>(row_m - new_m);
            }
            auto probs_f32 =
                ct::select(valid,
                           softmax_probs_from_scores<UseExp2, Prob>(scores, new_m),
                           scores * 0.0f);
            auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
            auto tile_l = softmax_tile_sum<ProbSumBf16>(probs_f32, probs_bf16);
            out_acc = out_acc * alpha +
                      ct::mma(probs_bf16,
                              v_view.load_masked(FullKTiles, 0),
                              ct::full<OutTile>(0.0f));
            if constexpr (RoundOutAcc) {
                out_acc = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(out_acc));
            }
            row_l = row_l * alpha + tile_l;
        } else {
            I64ScoreTile score_local = ct::iota<I64ScoreTile>();
            auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
            auto valid = key_cols < kN;
            auto scores = ct::mma(q_tile,
                                  k_t_view.load_masked(0, FullKTiles),
                                  ct::full<ScoreTile>(0.0f)) * scale;
            auto neg_inf = scores * 0.0f - 3.402823466e38f;
            scores = ct::select(valid, scores, neg_inf);
            auto tile_m = ct::reduce_max<1>(scores);
            auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
            auto alpha = ct::full<RowTile>(1.0f);
            if constexpr (!SkipAlphaExp) {
                alpha = softmax_alpha<UseExp2, Prob, Alpha>(row_m - new_m);
            }
            auto probs_f32 =
                ct::select(valid,
                           softmax_probs_from_scores<UseExp2, Prob>(scores, new_m),
                           scores * 0.0f);
            auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
            auto tile_l = softmax_tile_sum<ProbSumBf16>(probs_f32, probs_bf16);
            out_acc = out_acc * alpha +
                      ct::mma(probs_bf16,
                              v_view.load_masked(FullKTiles, 0),
                              ct::full<OutTile>(0.0f));
            if constexpr (RoundOutAcc) {
                out_acc = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(out_acc));
            }
            row_l = row_l * alpha + tile_l;
        }
    }

    if constexpr (UseFinalReciprocal) {
        auto inv_l = 1.0f / row_l;
        out_view.store(ct::element_cast<__nv_bfloat16>(out_acc * inv_l), q_block, 0);
    } else {
        out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
    }
}

template <int QRows,
          int KTile,
          bool UseExp2,
          typename QTile,
          typename KTView,
          typename VView,
          typename RowTile,
          typename OutTile>
static __tile__ TailMergeState<RowTile, OutTile> merge_keytail_helper(
    QTile q_tile,
    KTView k_t_view,
    VView v_view,
    RowTile row_m,
    RowTile row_l,
    OutTile out_acc,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f)) * scale;
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(valid, scores, neg_inf);
    auto tile_m = ct::reduce_max<1>(scores);
    auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
    auto alpha = softmax_exp<UseExp2>(row_m - new_m);
    auto probs_f32 =
        ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto tile_l = ct::sum<1>(probs_f32);
    out_acc = out_acc * alpha +
              ct::mma(probs_bf16,
                      v_view.load_masked(FullKTiles, 0),
                      ct::full<OutTile>(0.0f));
    row_l = row_l * alpha + tile_l;
    return TailMergeState<RowTile, OutTile>{row_l, out_acc};
}

template <int QRows, int KTile, bool UseExp2 = true>
__tile_global__ void time_attention1301_main1280_split_contig_tail_helper_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kN, kD>;
    using DNShape = ct::shape<kD, kN>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kN * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    auto state = merge_keytail_helper<QRows, KTile, UseExp2>(
        q_tile, k_t_view, v_view, row_m, row_l, out_acc, scale);
    out_view.store(ct::element_cast<__nv_bfloat16>(state.out_acc / state.row_l),
                   q_block, 0);
}

template <int QRows, int KTile, bool UseExp2 = true>
__tile_global__ void
time_attention1301_main1280_split_contig_tail_prob_mask_only_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kN, kD>;
    using DNShape = ct::shape<kD, kN>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kN * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f)) * scale;
    auto tile_m = ct::reduce_max<1>(scores);
    auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
    auto alpha = softmax_exp<UseExp2>(row_m - new_m);
    auto probs_f32 =
        ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto tile_l = ct::sum<1>(probs_f32);
    out_acc = out_acc * alpha +
              ct::mma(probs_bf16,
                      v_view.load_masked(FullKTiles, 0),
                      ct::full<OutTile>(0.0f));
    row_l = row_l * alpha + tile_l;

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

template <int QRows, int KTile, bool UseExp2 = true>
__tile_global__ void time_attention1301_main1280_split_contig_tail16_8_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    static_assert(KTile == 32);
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using ScoreTile16 = ct::tile<float, ct::shape<QRows, 16>>;
    using ScoreTile8 = ct::tile<float, ct::shape<QRows, 8>>;
    using I64ColTile8 = ct::tile<long long, ct::shape<1, 8>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kN, kD>;
    using DNShape = ct::shape<kD, kN>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kN * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto k_t_view16 = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, 16>{}
    };
    auto v_view16 = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<16, kD>{}
    };
    auto k_t_view8 = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, 8>{}
    };
    auto v_view8 = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<8, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    {
        auto scores = ct::mma(q_tile,
                              k_t_view16.load(0, kNMain / 16),
                              ct::full<ScoreTile16>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);
        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view16.load(kNMain / 16, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    {
        I64ColTile8 col_local = ct::iota<I64ColTile8>();
        auto key_cols = static_cast<long long>(kNMain + 16) + col_local;
        auto valid = key_cols < kN;
        auto scores = ct::mma(q_tile,
                              k_t_view8.load_masked(0, (kNMain + 16) / 8),
                              ct::full<ScoreTile8>(0.0f)) * scale;
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores, neg_inf);
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 =
            ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);
        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view8.load_masked((kNMain + 16) / 8, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
    }

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

template <int QRows,
          int KTile,
          bool UseExp2 = true,
          bool IncludeKeyTail = true>
__tile_global__ void time_attention1301_main1280_split_contig_gated_store_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ gates,
    __nv_bfloat16* __restrict__ merged,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using GateTile = ct::tile<float, ct::shape<QRows, 1>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kN, kD>;
    using DNShape = ct::shape<kD, kN>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    gates = ct::assume_aligned(gates, 16_ic);
    merged = ct::assume_aligned(merged, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kN * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    const __nv_bfloat16* gate_batch =
        gates + static_cast<std::size_t>(b) * kN * kHeads + h;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    if constexpr (IncludeKeyTail) {
        I64ScoreTile score_local = ct::iota<I64ScoreTile>();
        auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
        auto valid = key_cols < kN;
        auto scores = ct::mma(q_tile,
                              k_t_view.load_masked(0, FullKTiles),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores, neg_inf);
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 =
            ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);
        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load_masked(FullKTiles, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
    }

    using GateIndexTile = ct::tile<long long, ct::shape<QRows, 1>>;
    GateIndexTile gate_local = ct::iota<GateIndexTile>();
    auto gate_rows = static_cast<long long>(q_block) * QRows + gate_local;
    GateTile gate_values =
        ct::element_cast<float>(ct::load(gate_batch + gate_rows * kHeads));
    auto attn_values =
        ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(out_acc / row_l));
    auto gated = attn_values * gate_values;
    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_idx = (static_cast<long long>(b) * kNMain + out_rows) * (kHeads * kD) +
                   static_cast<long long>(h) * kD + out_cols;
    ct::store(merged + out_idx, ct::element_cast<__nv_bfloat16>(gated));
}

template <int QRows, int KTile, bool UseExp2 = true, bool IncludeKeyTail = true>
__tile_global__ void time_attention1301_main1280_split_contig_kt_input_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_t,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kN, kD>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;

    q = ct::assume_aligned(q, 16_ic);
    k_t = ct::assume_aligned(k_t, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t split_base =
        (static_cast<std::size_t>(b) * kN * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + split_base;
    const __nv_bfloat16* v_batch = v + split_base;
    const __nv_bfloat16* k_t_batch =
        k_t + static_cast<std::size_t>(bh) * kD * kN;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_t_batch, ct::shape<kD, kN>{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    if constexpr (IncludeKeyTail) {
        I64ScoreTile score_local = ct::iota<I64ScoreTile>();
        auto key_cols =
            static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
        auto valid = key_cols < kN;
        auto scores = ct::mma(q_tile,
                              k_t_view.load_masked(0, FullKTiles),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores, neg_inf);
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 =
            ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);
        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load_masked(FullKTiles, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
    }

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

template <int QRows, int KTile, bool UseExp2 = true, bool IncludeKeyTail = true>
__tile_global__ void time_attention1301_main1280_split_contig_seg2_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    static_assert(FullKTiles % 2 == 0);
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kN, kD>;
    using DNShape = ct::shape<kD, kN>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kN * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto seg : ct::irange(std::size_t{0}, std::size_t{FullKTiles / 2})) {
        auto kt0 = seg * 2;
        auto kt1 = kt0 + 1;

        auto scores0 = ct::mma(q_tile,
                               k_t_view.load(0, kt0),
                               ct::full<ScoreTile>(0.0f)) * scale;
        RowTile local_m = ct::reduce_max<1>(scores0);
        auto probs0_f32 = softmax_exp<UseExp2>(scores0 - local_m);
        auto probs0_bf16 = ct::element_cast<__nv_bfloat16>(probs0_f32);
        RowTile local_l = ct::sum<1>(probs0_f32);
        OutTile local_acc = ct::mma(probs0_bf16,
                                    v_view.load(kt0, 0),
                                    ct::full<OutTile>(0.0f));

        auto scores1 = ct::mma(q_tile,
                               k_t_view.load(0, kt1),
                               ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores1);
        auto new_local_m = ct::select(local_m > tile_m, local_m, tile_m);
        auto local_alpha = softmax_exp<UseExp2>(local_m - new_local_m);
        auto probs1_f32 = softmax_exp<UseExp2>(scores1 - new_local_m);
        auto probs1_bf16 = ct::element_cast<__nv_bfloat16>(probs1_f32);
        auto tile_l = ct::sum<1>(probs1_f32);

        local_acc = local_acc * local_alpha +
                    ct::mma(probs1_bf16,
                            v_view.load(kt1, 0),
                            ct::full<OutTile>(0.0f));
        local_l = local_l * local_alpha + tile_l;
        local_m = new_local_m;

        auto new_m = ct::select(row_m > local_m, row_m, local_m);
        auto global_alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto local_beta = softmax_exp<UseExp2>(local_m - new_m);
        out_acc = out_acc * global_alpha + local_acc * local_beta;
        row_l = row_l * global_alpha + local_l * local_beta;
        row_m = new_m;
    }

    if constexpr (IncludeKeyTail) {
        I64ScoreTile score_local = ct::iota<I64ScoreTile>();
        auto key_cols =
            static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
        auto valid = key_cols < kN;
        auto scores = ct::mma(q_tile,
                              k_t_view.load_masked(0, FullKTiles),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores, neg_inf);
        auto local_m = ct::reduce_max<1>(scores);
        auto probs_f32 =
            ct::select(valid, softmax_exp<UseExp2>(scores - local_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto local_l = ct::sum<1>(probs_f32);
        auto local_acc = ct::mma(probs_bf16,
                                 v_view.load_masked(FullKTiles, 0),
                                 ct::full<OutTile>(0.0f));

        auto new_m = ct::select(row_m > local_m, row_m, local_m);
        auto global_alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto local_beta = softmax_exp<UseExp2>(local_m - new_m);
        out_acc = out_acc * global_alpha + local_acc * local_beta;
        row_l = row_l * global_alpha + local_l * local_beta;
    }

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

template <int QRows,
          int KTile,
          bool IncludeKeyTail = true,
          bool TailColBroadcastPaddedTailLoad = false>
__tile_global__ void time_attention1301_main1280_split_contig_first_init_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    constexpr int NLayout =
        TailColBroadcastPaddedTailLoad ? kNPad : kN;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64ColTile = ct::tile<long long, ct::shape<1, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<NLayout, kD>;
    using DNShape = ct::shape<kD, NLayout>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * NLayout * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    auto scores0 = ct::mma(q_tile,
                           k_t_view.load(0, 0),
                           ct::full<ScoreTile>(0.0f)) * scale;
    RowTile row_m = ct::reduce_max<1>(scores0);
    auto probs0_f32 = softmax_exp<true>(scores0 - row_m);
    auto probs0_bf16 = ct::element_cast<__nv_bfloat16>(probs0_f32);
    RowTile row_l = ct::sum<1>(probs0_f32);
    OutTile out_acc = ct::mma(probs0_bf16,
                              v_view.load(0, 0),
                              ct::full<OutTile>(0.0f));

    for (auto kt : ct::irange(std::size_t{1}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<true>(row_m - new_m);
        auto probs_f32 = softmax_exp<true>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    if constexpr (IncludeKeyTail) {
        if constexpr (TailColBroadcastPaddedTailLoad) {
            I64ColTile col_local = ct::iota<I64ColTile>();
            auto key_cols =
                static_cast<long long>(FullKTiles) * KTile + col_local;
            auto valid = key_cols < kN;
            auto scores = ct::mma(q_tile,
                                  k_t_view.load(0, FullKTiles),
                                  ct::full<ScoreTile>(0.0f)) * scale;
            auto neg_inf = scores * 0.0f - 3.402823466e38f;
            scores = ct::select(valid, scores, neg_inf);
            auto tile_m = ct::reduce_max<1>(scores);
            auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
            auto alpha = softmax_exp<true>(row_m - new_m);
            auto probs_f32 =
                ct::select(valid, softmax_exp<true>(scores - new_m),
                           scores * 0.0f);
            auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
            auto tile_l = ct::sum<1>(probs_f32);
            out_acc = out_acc * alpha +
                      ct::mma(probs_bf16,
                              v_view.load(FullKTiles, 0),
                              ct::full<OutTile>(0.0f));
            row_l = row_l * alpha + tile_l;
        } else {
            I64ScoreTile score_local = ct::iota<I64ScoreTile>();
            auto key_cols =
                static_cast<long long>(FullKTiles) * KTile +
                score_local % KTile;
            auto valid = key_cols < kN;
            auto scores = ct::mma(q_tile,
                                  k_t_view.load_masked(0, FullKTiles),
                                  ct::full<ScoreTile>(0.0f)) * scale;
            auto neg_inf = scores * 0.0f - 3.402823466e38f;
            scores = ct::select(valid, scores, neg_inf);
            auto tile_m = ct::reduce_max<1>(scores);
            auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
            auto alpha = softmax_exp<true>(row_m - new_m);
            auto probs_f32 =
                ct::select(valid, softmax_exp<true>(scores - new_m),
                           scores * 0.0f);
            auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
            auto tile_l = ct::sum<1>(probs_f32);
            out_acc = out_acc * alpha +
                      ct::mma(probs_bf16,
                              v_view.load_masked(FullKTiles, 0),
                              ct::full<OutTile>(0.0f));
            row_l = row_l * alpha + tile_l;
        }
    }

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

template <int QRows,
          int KTile,
          bool IncludeKeyTail = true,
          bool RoundRowM = false>
__tile_global__ void time_attention1301_main1280_split_contig_row_state_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kN, kD>;
    using DNShape = ct::shape<kD, kN>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kN * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        if constexpr (RoundRowM) {
            new_m = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(new_m));
        }
        auto alpha = softmax_exp<true>(row_m - new_m);
        alpha = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(alpha));
        auto probs_f32 = softmax_exp<true>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(ct::element_cast<float>(probs_bf16));

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_l = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(row_l));
        row_m = new_m;
    }

    if constexpr (IncludeKeyTail) {
        I64ScoreTile score_local = ct::iota<I64ScoreTile>();
        auto key_cols =
            static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
        auto valid = key_cols < kN;
        auto scores = ct::mma(q_tile,
                              k_t_view.load_masked(0, FullKTiles),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores, neg_inf);
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        if constexpr (RoundRowM) {
            new_m = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(new_m));
        }
        auto alpha = softmax_exp<true>(row_m - new_m);
        alpha = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(alpha));
        auto probs_f32 =
            ct::select(valid, softmax_exp<true>(scores - new_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(ct::element_cast<float>(probs_bf16));
        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load_masked(FullKTiles, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_l = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(row_l));
    }

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

template <int QRows,
          int KTile,
          bool IncludeKeyTail = true,
          int QLoadLatency = 2,
          int KLoadLatency = 2,
          int VLoadLatency = 2>
__tile_global__ void time_attention1301_main1280_split_contig_input_lat_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kN, kD>;
    using DNShape = ct::shape<kD, kN>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kN * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    decltype(q_view.load(q_block, 0)) q_tile;
    if constexpr (QLoadLatency > 0) {
        [[cutile::hint(0, latency=QLoadLatency)]]
        q_tile = q_view.load(q_block, 0);
    } else {
        q_tile = q_view.load(q_block, 0);
    }
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        decltype(k_t_view.load(0, kt)) k_tile;
        decltype(v_view.load(kt, 0)) v_tile;
        if constexpr (KLoadLatency > 0) {
            [[cutile::hint(0, latency=KLoadLatency)]]
            k_tile = k_t_view.load(0, kt);
        } else {
            k_tile = k_t_view.load(0, kt);
        }
        if constexpr (VLoadLatency > 0) {
            [[cutile::hint(0, latency=VLoadLatency)]]
            v_tile = v_view.load(kt, 0);
        } else {
            v_tile = v_view.load(kt, 0);
        }
        auto scores = ct::mma(q_tile,
                              k_tile,
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<true>(row_m - new_m);
        auto probs_f32 = softmax_exp<true>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_tile,
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    if constexpr (IncludeKeyTail) {
        I64ScoreTile score_local = ct::iota<I64ScoreTile>();
        auto key_cols =
            static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
        auto valid = key_cols < kN;
        decltype(k_t_view.load_masked(0, FullKTiles)) k_tail;
        decltype(v_view.load_masked(FullKTiles, 0)) v_tail;
        if constexpr (KLoadLatency > 0) {
            [[cutile::hint(0, latency=KLoadLatency)]]
            k_tail = k_t_view.load_masked(0, FullKTiles);
        } else {
            k_tail = k_t_view.load_masked(0, FullKTiles);
        }
        if constexpr (VLoadLatency > 0) {
            [[cutile::hint(0, latency=VLoadLatency)]]
            v_tail = v_view.load_masked(FullKTiles, 0);
        } else {
            v_tail = v_view.load_masked(FullKTiles, 0);
        }
        auto scores = ct::mma(q_tile,
                              k_tail,
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores, neg_inf);
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<true>(row_m - new_m);
        auto probs_f32 =
            ct::select(valid, softmax_exp<true>(scores - new_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);
        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_tail,
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
    }

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

template <int QRows, int KTile, bool IncludeKeyTail = true>
__tile_global__ void time_attention1301_main1280_split_contig_score_av_lb_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using NDShape = ct::shape<kN, kD>;
    using DNShape = ct::shape<kD, kN>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kN * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        out_acc = out_acc +
                  ct::mma(ct::element_cast<__nv_bfloat16>(scores),
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
    }

    if constexpr (IncludeKeyTail) {
        I64ScoreTile score_local = ct::iota<I64ScoreTile>();
        auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
        auto valid = key_cols < kN;
        auto scores = ct::mma(q_tile,
                              k_t_view.load_masked(0, FullKTiles),
                              ct::full<ScoreTile>(0.0f));
        auto score_values = ct::select(valid, scores * scale, scores * 0.0f);
        out_acc = out_acc +
                  ct::mma(ct::element_cast<__nv_bfloat16>(score_values),
                          v_view.load_masked(FullKTiles, 0),
                          ct::full<OutTile>(0.0f));
    }

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc), q_block, 0);
}

template <int QRows, int KTile, bool IncludeKeyTail = true>
__tile_global__ void time_attention1301_main1280_split_contig_tile_local_softmax_lb_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kN, kD>;
    using DNShape = ct::shape<kD, kN>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kN * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    OutTile out_acc = ct::full<OutTile>(0.0f);
    RowTile denom_acc = ct::full<RowTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto probs_f32 = softmax_exp<true>(scores - tile_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        out_acc = out_acc +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        denom_acc = denom_acc + ct::sum<1>(probs_f32);
    }

    if constexpr (IncludeKeyTail) {
        I64ScoreTile score_local = ct::iota<I64ScoreTile>();
        auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
        auto valid = key_cols < kN;
        auto scores = ct::mma(q_tile,
                              k_t_view.load_masked(0, FullKTiles),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores, neg_inf);
        auto tile_m = ct::reduce_max<1>(scores);
        auto probs_f32 =
            ct::select(valid, softmax_exp<true>(scores - tile_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        out_acc = out_acc +
                  ct::mma(probs_bf16,
                          v_view.load_masked(FullKTiles, 0),
                          ct::full<OutTile>(0.0f));
        denom_acc = denom_acc + ct::sum<1>(probs_f32);
    }

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / denom_acc), q_block, 0);
}

template <int QRows, int KTile, bool UseExp2 = true>
__tile_global__ void time_attention1301_main1280_split_contig_tail_first_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kN, kD>;
    using DNShape = ct::shape<kD, kN>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kN * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    {
        I64ScoreTile score_local = ct::iota<I64ScoreTile>();
        auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
        auto valid = key_cols < kN;
        auto scores = ct::mma(q_tile,
                              k_t_view.load_masked(0, FullKTiles),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores, neg_inf);
        row_m = ct::reduce_max<1>(scores);
        auto probs_f32 =
            ct::select(valid, softmax_exp<UseExp2>(scores - row_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        row_l = ct::sum<1>(probs_f32);
        out_acc = ct::mma(probs_bf16,
                          v_view.load_masked(FullKTiles, 0),
                          ct::full<OutTile>(0.0f));
    }

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

template <int QRows, int KTile, bool UseExp2 = true>
__tile_global__ void
time_attention1301_main1280_split_contig_tail_first_padded_tail_load_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kNPad, kD>;
    using DNShape = ct::shape<kD, kNPad>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kNPad * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    {
        I64ScoreTile score_local = ct::iota<I64ScoreTile>();
        auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
        auto valid = key_cols < kN;
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, FullKTiles),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores, neg_inf);
        row_m = ct::reduce_max<1>(scores);
        auto probs_f32 =
            ct::select(valid, softmax_exp<UseExp2>(scores - row_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        row_l = ct::sum<1>(probs_f32);
        out_acc = ct::mma(probs_bf16,
                          v_view.load(FullKTiles, 0),
                          ct::full<OutTile>(0.0f));
    }

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

template <int QRows,
          int KTile,
          bool UseExp2 = true,
          bool TailColBroadcast = false>
__tile_global__ void time_attention1301_main1280_split_contig_padded_tail_load_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64ColTile = ct::tile<long long, ct::shape<1, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kNPad, kD>;
    using DNShape = ct::shape<kD, kNPad>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kNPad * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    if constexpr (TailColBroadcast) {
        I64ColTile col_local = ct::iota<I64ColTile>();
        auto key_cols =
            static_cast<long long>(FullKTiles) * KTile + col_local;
        auto valid = key_cols < kN;
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, FullKTiles),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores, neg_inf);
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 =
            ct::select(valid, softmax_exp<UseExp2>(scores - new_m),
                       scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);
        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(FullKTiles, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
    } else {
        I64ScoreTile score_local = ct::iota<I64ScoreTile>();
        auto key_cols =
            static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
        auto valid = key_cols < kN;
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, FullKTiles),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores, neg_inf);
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 =
            ct::select(valid, softmax_exp<UseExp2>(scores - new_m),
                       scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);
        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(FullKTiles, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
    }

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

template <int QRows, int KTile, bool UseExp2 = true>
__tile_global__ void time_attention1301_main1280_split_contig_state1280_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    float* __restrict__ state_acc,
    float* __restrict__ state_m,
    float* __restrict__ state_l,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kN, kD>;
    using DNShape = ct::shape<kD, kN>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    state_acc = ct::assume_aligned(state_acc, 16_ic);
    state_m = ct::assume_aligned(state_m, 16_ic);
    state_l = ct::assume_aligned(state_l, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kN * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    float* state_acc_batch =
        state_acc + static_cast<std::size_t>(bh) * kNMain * kD;
    float* state_m_batch =
        state_m + static_cast<std::size_t>(bh) * kNMain;
    float* state_l_batch =
        state_l + static_cast<std::size_t>(bh) * kNMain;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto state_acc_view = ct::partition_view{
        ct::tensor_span{state_acc_batch, ct::shape<kNMain, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto state_m_view = ct::partition_view{
        ct::tensor_span{state_m_batch, ct::shape<kNMain, 1>{}},
        ct::shape<QRows, 1>{}
    };
    auto state_l_view = ct::partition_view{
        ct::tensor_span{state_l_batch, ct::shape<kNMain, 1>{}},
        ct::shape<QRows, 1>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    state_acc_view.store(out_acc, q_block, 0);
    state_m_view.store(row_m, q_block, 0);
    state_l_view.store(row_l, q_block, 0);
}

template <int QRows, int KTile, bool UseExp2 = true>
__tile_global__ void time_attention1301_main1280_split_contig_tail_finalize_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state_acc,
    const float* __restrict__ state_m,
    const float* __restrict__ state_l,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;
    using NDShape = ct::shape<kN, kD>;
    using DNShape = ct::shape<kD, kN>;
    using NDStrides = ct::shape<kHeads * kD, 1>;
    using DNStrides = ct::shape<1, kHeads * kD>;
    using NDLayout = ct::layout_strided<NDStrides>;
    using DNLayout = ct::layout_strided<DNStrides>;
    using NDMapping = typename NDLayout::template mapping<NDShape>;
    using DNMapping = typename DNLayout::template mapping<DNShape>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    state_acc = ct::assume_aligned(state_acc, 16_ic);
    state_m = ct::assume_aligned(state_m, 16_ic);
    state_l = ct::assume_aligned(state_l, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kN * kHeads + h) * kD;
    const __nv_bfloat16* q_batch = q + base;
    const __nv_bfloat16* k_batch = k + base;
    const __nv_bfloat16* v_batch = v + base;
    const float* state_acc_batch =
        state_acc + static_cast<std::size_t>(bh) * kNMain * kD;
    const float* state_m_batch =
        state_m + static_cast<std::size_t>(bh) * kNMain;
    const float* state_l_batch =
        state_l + static_cast<std::size_t>(bh) * kNMain;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, DNMapping{DNShape{}, DNStrides{}}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, NDMapping{NDShape{}, NDStrides{}}},
        ct::shape<KTile, kD>{}
    };
    auto state_acc_view = ct::partition_view{
        ct::tensor_span{state_acc_batch, ct::shape<kNMain, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto state_m_view = ct::partition_view{
        ct::tensor_span{state_m_batch, ct::shape<kNMain, 1>{}},
        ct::shape<QRows, 1>{}
    };
    auto state_l_view = ct::partition_view{
        ct::tensor_span{state_l_batch, ct::shape<kNMain, 1>{}},
        ct::shape<QRows, 1>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    OutTile out_acc = state_acc_view.load(q_block, 0);
    RowTile row_m = state_m_view.load(q_block, 0);
    RowTile row_l = state_l_view.load(q_block, 0);

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f)) * scale;
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(valid, scores, neg_inf);
    auto tile_m = ct::reduce_max<1>(scores);
    auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
    auto alpha = softmax_exp<UseExp2>(row_m - new_m);
    auto probs_f32 =
        ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto tile_l = ct::sum<1>(probs_f32);
    out_acc = out_acc * alpha +
              ct::mma(probs_bf16,
                      v_view.load_masked(FullKTiles, 0),
                      ct::full<OutTile>(0.0f));
    row_l = row_l * alpha + tile_l;

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

__tile_global__ void time_attention1301_main1280_q64k32_exp2_split_d32_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int QRows = 64;
    constexpr int KTile = 32;
    constexpr int DTile = 32;
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, DTile>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, DTile>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, DTile>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc0 = ct::full<OutTile>(0.0f);
    OutTile out_acc1 = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<true>(row_m - new_m);
        auto probs_f32 = softmax_exp<true>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc0 = out_acc0 * alpha +
                   ct::mma(probs_bf16,
                           v_view.load(kt, 0),
                           ct::full<OutTile>(0.0f));
        out_acc1 = out_acc1 * alpha +
                   ct::mma(probs_bf16,
                           v_view.load(kt, 1),
                           ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f)) * scale;
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(valid, scores, neg_inf);
    auto tile_m = ct::reduce_max<1>(scores);
    auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
    auto alpha = softmax_exp<true>(row_m - new_m);
    auto probs_f32 = ct::select(valid, softmax_exp<true>(scores - new_m), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto tile_l = ct::sum<1>(probs_f32);
    out_acc0 = out_acc0 * alpha +
               ct::mma(probs_bf16,
                       v_view.load_masked(FullKTiles, 0),
                       ct::full<OutTile>(0.0f));
    out_acc1 = out_acc1 * alpha +
               ct::mma(probs_bf16,
                       v_view.load_masked(FullKTiles, 1),
                       ct::full<OutTile>(0.0f));
    row_l = row_l * alpha + tile_l;

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc0 / row_l), q_block, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc1 / row_l), q_block, 1);
}

template <int QRows, int KTile>
__tile_global__ void time_attention1301_main1280_direct_ptr_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using QIndexTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using KIndexTile = ct::tile<long long, ct::shape<kD, KTile>>;
    using VIndexTile = ct::tile<long long, ct::shape<KTile, kD>>;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const std::size_t batch_offset = static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* q_batch = q + batch_offset;
    const __nv_bfloat16* k_batch = k + batch_offset;
    const __nv_bfloat16* v_batch = v + batch_offset;
    __nv_bfloat16* out_batch = out + batch_offset;

    QIndexTile q_local = ct::iota<QIndexTile>();
    auto q_rows = static_cast<long long>(q_block) * QRows + q_local / kD;
    auto q_cols = q_local % kD;
    auto q_tile = ct::load(q_batch + q_rows * kD + q_cols);

    KIndexTile k_local = ct::iota<KIndexTile>();
    auto k_d = k_local / KTile;
    auto k_cols_local = k_local % KTile;

    VIndexTile v_local = ct::iota<VIndexTile>();
    auto v_rows_local = v_local / kD;
    auto v_cols = v_local % kD;

    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto k_cols = static_cast<long long>(kt) * KTile + k_cols_local;
        auto k_tile = ct::load(k_batch + k_cols * kD + k_d);

        auto scores = ct::mma(q_tile,
                              k_tile,
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = ct::exp(row_m - new_m);
        auto probs_f32 = ct::exp(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        auto v_rows = static_cast<long long>(kt) * KTile + v_rows_local;
        auto v_tile = ct::load(v_batch + v_rows * kD + v_cols);
        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_tile,
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto score_valid = score_key_cols < kN;
    auto tail_k_cols = static_cast<long long>(FullKTiles) * KTile + k_cols_local;
    auto tail_k_valid = tail_k_cols < kN;
    auto tail_k_safe_cols = ct::select(tail_k_valid, tail_k_cols, tail_k_cols * 0LL);
    auto k_tile = ct::load_masked(k_batch + tail_k_safe_cols * kD + k_d, tail_k_valid);
    auto scores = ct::mma(q_tile,
                          k_tile,
                          ct::full<ScoreTile>(0.0f)) * scale;
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(score_valid, scores, neg_inf);
    auto tile_m = ct::reduce_max<1>(scores);
    auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
    auto alpha = ct::exp(row_m - new_m);
    auto probs_f32 = ct::select(score_valid, ct::exp(scores - new_m), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto tile_l = ct::sum<1>(probs_f32);

    auto tail_v_rows = static_cast<long long>(FullKTiles) * KTile + v_rows_local;
    auto tail_v_valid = tail_v_rows < kN;
    auto tail_v_safe_rows = ct::select(tail_v_valid, tail_v_rows, tail_v_rows * 0LL);
    auto v_tile = ct::load_masked(v_batch + tail_v_safe_rows * kD + v_cols, tail_v_valid);
    out_acc = out_acc * alpha +
              ct::mma(probs_bf16,
                      v_tile,
                      ct::full<OutTile>(0.0f));
    row_l = row_l * alpha + tile_l;

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    ct::store_masked(out_batch + out_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc / row_l),
                     out_rows < kN);
}

template <int QRows, int KTile, bool UseExp2 = true>
__tile_global__ void time_attention1301_main1280_qkv_direct_rotary_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const float* __restrict__ cos_f,
    const float* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using QIndexTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using KIndexTile = ct::tile<long long, ct::shape<kD, KTile>>;
    using VIndexTile = ct::tile<long long, ct::shape<KTile, kD>>;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    qkv = ct::assume_aligned(qkv, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_raw, tile_z] = ct::bid();
    (void)tile_z;
    int bh = static_cast<int>(bh_raw);
    int b = bh / kHeads;
    int h = bh - b * kHeads;
    const long long head_base = static_cast<long long>(h) * kD;
    const long long k_part_base = kHeads * kD + head_base;
    const long long v_part_base = 2LL * kHeads * kD + head_base;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    QIndexTile q_local = ct::iota<QIndexTile>();
    auto q_rows = static_cast<long long>(q_block) * QRows + q_local / kD;
    auto q_cols = q_local % kD;
    auto q_pair = q_cols / 2LL;
    auto q_even_d = q_pair * 2LL;
    auto q_is_odd = (q_cols & 1LL) != 0LL;
    auto q_base = (static_cast<long long>(b) * kN + q_rows) * kQkvStride;
    auto q_even = ct::element_cast<float>(
        ct::load(qkv + q_base + head_base + q_even_d));
    auto q_odd = ct::element_cast<float>(
        ct::load(qkv + q_base + head_base + q_even_d + 1LL));
    auto q_c = ct::load(cos_f + q_rows * (kD / 2) + q_pair);
    auto q_s = ct::load(sin_f + q_rows * (kD / 2) + q_pair);
    auto q_rot_even = q_even * q_c - q_odd * q_s;
    auto q_rot_odd = q_even * q_s + q_odd * q_c;
    auto q_tile = ct::element_cast<__nv_bfloat16>(
        ct::select(q_is_odd, q_rot_odd, q_rot_even));

    KIndexTile k_local = ct::iota<KIndexTile>();
    auto k_d = k_local / KTile;
    auto k_cols_local = k_local % KTile;
    auto k_pair = k_d / 2LL;
    auto k_even_d = k_pair * 2LL;
    auto k_is_odd = (k_d & 1LL) != 0LL;

    VIndexTile v_local = ct::iota<VIndexTile>();
    auto v_rows_local = v_local / kD;
    auto v_cols = v_local % kD;

    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto key_cols = static_cast<long long>(kt) * KTile + k_cols_local;
        auto k_base = (static_cast<long long>(b) * kN + key_cols) * kQkvStride;
        auto k_even = ct::element_cast<float>(
            ct::load(qkv + k_base + k_part_base + k_even_d));
        auto k_odd = ct::element_cast<float>(
            ct::load(qkv + k_base + k_part_base + k_even_d + 1LL));
        auto k_c = ct::load(cos_f + key_cols * (kD / 2) + k_pair);
        auto k_s = ct::load(sin_f + key_cols * (kD / 2) + k_pair);
        auto k_rot_even = k_even * k_c - k_odd * k_s;
        auto k_rot_odd = k_even * k_s + k_odd * k_c;
        auto k_tile = ct::element_cast<__nv_bfloat16>(
            ct::select(k_is_odd, k_rot_odd, k_rot_even));

        auto scores = ct::mma(q_tile, k_tile, ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        auto v_rows = static_cast<long long>(kt) * KTile + v_rows_local;
        auto v_base = (static_cast<long long>(b) * kN + v_rows) * kQkvStride;
        auto v_tile = ct::load(qkv + v_base + v_part_base + v_cols);
        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16, v_tile, ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto score_valid = score_key_cols < kN;
    auto tail_k_cols = static_cast<long long>(FullKTiles) * KTile + k_cols_local;
    auto tail_k_valid = tail_k_cols < kN;
    auto tail_k_safe_cols = ct::select(tail_k_valid, tail_k_cols, tail_k_cols * 0LL);
    auto tail_k_base = (static_cast<long long>(b) * kN + tail_k_safe_cols) * kQkvStride;
    auto k_even = ct::element_cast<float>(
        ct::load_masked(qkv + tail_k_base + k_part_base + k_even_d, tail_k_valid));
    auto k_odd = ct::element_cast<float>(
        ct::load_masked(qkv + tail_k_base + k_part_base + k_even_d + 1LL, tail_k_valid));
    auto k_c = ct::load_masked(cos_f + tail_k_safe_cols * (kD / 2) + k_pair, tail_k_valid);
    auto k_s = ct::load_masked(sin_f + tail_k_safe_cols * (kD / 2) + k_pair, tail_k_valid);
    auto k_rot_even = k_even * k_c - k_odd * k_s;
    auto k_rot_odd = k_even * k_s + k_odd * k_c;
    auto k_tile = ct::element_cast<__nv_bfloat16>(
        ct::select(k_is_odd, k_rot_odd, k_rot_even));

    auto scores = ct::mma(q_tile, k_tile, ct::full<ScoreTile>(0.0f)) * scale;
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(score_valid, scores, neg_inf);
    auto tile_m = ct::reduce_max<1>(scores);
    auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
    auto alpha = softmax_exp<UseExp2>(row_m - new_m);
    auto probs_f32 = ct::select(score_valid, softmax_exp<UseExp2>(scores - new_m),
                                scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto tile_l = ct::sum<1>(probs_f32);

    auto tail_v_rows = static_cast<long long>(FullKTiles) * KTile + v_rows_local;
    auto tail_v_valid = tail_v_rows < kN;
    auto tail_v_safe_rows = ct::select(tail_v_valid, tail_v_rows, tail_v_rows * 0LL);
    auto tail_v_base = (static_cast<long long>(b) * kN + tail_v_safe_rows) * kQkvStride;
    auto v_tile = ct::load_masked(qkv + tail_v_base + v_part_base + v_cols,
                                  tail_v_valid);
    out_acc = out_acc * alpha +
              ct::mma(probs_bf16, v_tile, ct::full<OutTile>(0.0f));
    row_l = row_l * alpha + tile_l;

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    ct::store_masked(out_batch + out_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc / row_l),
                     out_rows < kN);
}

template <int QRows, int KTile, int QBlockOffset, bool UseExp2 = false>
__tile_global__ void time_attention1301_tail_offset_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int KTiles = (kN + KTile - 1) / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block_local, bh, tile_z] = ct::bid();
    (void)tile_z;
    auto q_block = q_block_local + QBlockOffset;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };

    auto q_tile = q_view.load_masked(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / KTile;
    auto score_cols_local = score_local % KTile;

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{KTiles})) {
        auto key_cols = static_cast<long long>(kt) * KTile + score_cols_local;
        auto valid = (score_rows < kN) && (key_cols < kN);
        auto scores = ct::mma(q_tile,
                              k_t_view.load_masked(0, kt),
                              ct::full<ScoreTile>(0.0f));
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores * scale, neg_inf);

        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load_masked(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    out_acc = out_acc / row_l;
    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_valid = out_rows < kN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows>
__tile_global__ void time_attention1301_main1280_score_av_lb_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int KTile = 64;
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        out_acc = out_acc +
                  ct::mma(ct::element_cast<__nv_bfloat16>(scores),
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f));
    auto score_values = ct::select(valid, scores * scale, scores * 0.0f);
    out_acc = out_acc +
              ct::mma(ct::element_cast<__nv_bfloat16>(score_values),
                      v_view.load_masked(FullKTiles, 0),
                      ct::full<OutTile>(0.0f));

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc), q_block, 0);
}

float deterministic_bf16_value(size_t idx) {
    float v = 0.125f + static_cast<float>((idx * 17ULL) & 1023ULL) * 0.000244140625f;
    return __bfloat162float(__float2bfloat16(v));
}

template <int KTile>
void validate_rows(const std::vector<__nv_bfloat16>& out, float scale) {
    int rows[] = {0, 37, kN - 1};
    double max_abs = 0.0;
    double sum_sq = 0.0;
    long long count = 0;
    for (int row : rows) {
        std::vector<float> acc(kD, 0.0f);
        float row_m = -3.402823466e38f;
        float row_l = 0.0f;
        for (int kt = 0; kt < ceildiv(kN, KTile) * KTile; kt += KTile) {
            float tile_m = -3.402823466e38f;
            float scores[KTile];
            for (int j = 0; j < KTile; ++j) {
                int col = kt + j;
                float score = -3.402823466e38f;
                if (col < kN) {
                    float dot = 0.0f;
                    for (int d = 0; d < kD; ++d) {
                        float qv = deterministic_bf16_value(static_cast<size_t>(row) * kD + d);
                        float kv = deterministic_bf16_value(static_cast<size_t>(col) * kD + d);
                        dot += qv * kv;
                    }
                    score = dot * scale;
                    tile_m = std::max(tile_m, score);
                }
                scores[j] = score;
            }
            float new_m = std::max(row_m, tile_m);
            float alpha = std::exp(row_m - new_m);
            for (int d = 0; d < kD; ++d) {
                acc[d] *= alpha;
            }
            float tile_l = 0.0f;
            for (int j = 0; j < KTile; ++j) {
                int col = kt + j;
                if (col >= kN) continue;
                float p = std::exp(scores[j] - new_m);
                p = __bfloat162float(__float2bfloat16(p));
                tile_l += p;
                for (int d = 0; d < kD; ++d) {
                    float vv = deterministic_bf16_value(static_cast<size_t>(col) * kD + d);
                    acc[d] += p * vv;
                }
            }
            row_l = row_l * alpha + tile_l;
            row_m = new_m;
        }
        for (int d = 0; d < kD; ++d) {
            float ref = __bfloat162float(__float2bfloat16(acc[d] / row_l));
            float got = __bfloat162float(out[static_cast<size_t>(row) * kD + d]);
            double diff = static_cast<double>(got) - static_cast<double>(ref);
            max_abs = std::max(max_abs, std::abs(diff));
            sum_sq += diff * diff;
            ++count;
        }
    }
    std::printf("validate BH0 rows=3 max_abs=%.9g rms=%.9g\n",
                max_abs, std::sqrt(sum_sq / static_cast<double>(count)));
}

template <int QRows>
void run_padded_variant(const Options& opts,
                        const __nv_bfloat16* d_q,
                        const __nv_bfloat16* d_k,
                        const __nv_bfloat16* d_v,
                        __nv_bfloat16* d_out,
                        float scale,
                        bool run_validation) {
    dim3 grid(ceildiv(kN, QRows), kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_cutile_kernel<QRows><<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_cutile_kernel<QRows><<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    if (run_validation) {
        std::vector<__nv_bfloat16> out_bh0(static_cast<size_t>(kN) * kD);
        CUDA_CHECK(cudaMemcpy(out_bh0.data(), d_out, out_bh0.size() * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        validate_rows<kKTile>(out_bh0, scale);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kN * kN * kD;
    double padded_flops = 4.0 * static_cast<double>(kBH) * kN * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double padded_tflops = padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("padded qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s padded_math=%.2f TF/s checksum=%.6f\n",
                QRows, kKTile, best_ms, median_ms, real_tflops, padded_tflops, checksum);
}

template <int QRows, int KTile>
void run_masked_variant(const Options& opts,
                        const __nv_bfloat16* d_q,
                        const __nv_bfloat16* d_k,
                        const __nv_bfloat16* d_v,
                        __nv_bfloat16* d_out,
                        float scale,
                        bool run_validation) {
    dim3 grid(ceildiv(kN, QRows), kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_cutile_masked_kernel<QRows, KTile>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_cutile_masked_kernel<QRows, KTile>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    if (run_validation) {
        std::vector<__nv_bfloat16> out_bh0(static_cast<size_t>(kN) * kD);
        CUDA_CHECK(cudaMemcpy(out_bh0.data(), d_out, out_bh0.size() * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        validate_rows<KTile>(out_bh0, scale);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kN * kN * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kN * (ceildiv(kN, KTile) * KTile) * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("masked qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s checksum=%.6f\n",
                QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops, checksum);
}

template <int QRows, int KTile>
void run_score_av_lower_bound_variant(const Options& opts,
                                      const __nv_bfloat16* d_q,
                                      const __nv_bfloat16* d_k,
                                      const __nv_bfloat16* d_v,
                                      __nv_bfloat16* d_out,
                                      float scale) {
    dim3 grid(ceildiv(kN, QRows), kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_cutile_masked_score_av_lb_kernel<QRows, KTile>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_cutile_masked_score_av_lb_kernel<QRows, KTile>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kN * kN * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kN * (ceildiv(kN, KTile) * KTile) * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("score_av_lb qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s checksum=%.6f\n",
                QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops, checksum);
}

template <int QRows, bool ScoreAvLowerBound, bool SumF32 = false, bool UseExp2 = false,
          int KTile = 64, bool ScoreBf16 = false, bool PrescaleQ = false>
void run_main1280_variant(const Options& opts,
                          const __nv_bfloat16* d_q,
                          const __nv_bfloat16* d_k,
                          const __nv_bfloat16* d_v,
                          __nv_bfloat16* d_out,
                          float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        if constexpr (ScoreAvLowerBound) {
            time_attention1301_main1280_score_av_lb_kernel<QRows>
                <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        } else {
            time_attention1301_main1280_kernel<QRows, KTile, SumF32, UseExp2,
                                               ScoreBf16, PrescaleQ>
                <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        if constexpr (ScoreAvLowerBound) {
            time_attention1301_main1280_score_av_lb_kernel<QRows>
                <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        } else {
            time_attention1301_main1280_kernel<QRows, KTile, SumF32, UseExp2,
                                               ScoreBf16, PrescaleQ>
                <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops = 4.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    const char* name = "main1280";
    if constexpr (ScoreAvLowerBound) {
        name = "main1280_score_av_lb";
    } else if constexpr (PrescaleQ && UseExp2 && SumF32) {
        name = "main1280_sum_f32_exp2_prescale_q";
    } else if constexpr (PrescaleQ && SumF32) {
        name = "main1280_sum_f32_prescale_q";
    } else if constexpr (PrescaleQ) {
        name = "main1280_prescale_q";
    } else if constexpr (ScoreBf16 && UseExp2 && SumF32) {
        name = "main1280_sum_f32_exp2_score_bf16";
    } else if constexpr (ScoreBf16 && SumF32) {
        name = "main1280_sum_f32_score_bf16";
    } else if constexpr (ScoreBf16) {
        name = "main1280_score_bf16";
    } else if constexpr (UseExp2 && SumF32) {
        name = "main1280_sum_f32_exp2";
    } else if constexpr (UseExp2) {
        name = "main1280_exp2_sum_bf16";
    } else if constexpr (SumF32) {
        name = "main1280_sum_f32";
    }
    std::printf("%s qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                name, QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

template <int QRows,
          int KTile,
          bool IncludeKeyTail = true,
          bool TailIdx32 = false,
          bool UseFinalReciprocal = false,
          bool TailColBroadcast = false,
          bool RoundOutAcc = false,
          bool SkipAlphaExp = false,
          int Prob = kProbExp,
          bool ProbSumBf16 = false,
          int Alpha = kAlphaExact>
void run_main1280_split_contig_input_variant(const Options& opts,
                                             const __nv_bfloat16* d_q,
                                             const __nv_bfloat16* d_k,
                                             const __nv_bfloat16* d_v,
                                             __nv_bfloat16* d_out,
                                             float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_input_kernel
            <QRows, KTile, true, IncludeKeyTail, TailIdx32, UseFinalReciprocal,
             TailColBroadcast, RoundOutAcc, SkipAlphaExp, Prob, ProbSumBf16,
             Alpha>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_input_kernel
            <QRows, KTile, true, IncludeKeyTail, TailIdx32, UseFinalReciprocal,
             TailColBroadcast, RoundOutAcc, SkipAlphaExp, Prob, ProbSumBf16,
             Alpha>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    constexpr int EffectiveK = IncludeKeyTail ? kN : kNMain;
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * EffectiveK * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain *
        (IncludeKeyTail ? kNPad : kNMain) * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    const char* name = "main1280_split_contig_input_exp2";
    if constexpr (UseFinalReciprocal && Alpha == kAlphaProbClamp &&
                  Prob == kProbPoly3NoClamp) {
        name = "main1280_split_contig_input_exp2_prob_poly3_noclamp_alpha_poly3_clamp_final_rcp";
    } else if constexpr (Alpha == kAlphaProbClamp && Prob == kProbRational4) {
        name = "main1280_split_contig_input_exp2_prob_rational4_alpha_rational4";
    } else if constexpr (Alpha == kAlphaLinearClamp && Prob == kProbPoly3NoClamp) {
        name = "main1280_split_contig_input_exp2_prob_poly3_noclamp_alpha_linear_clamp";
    } else if constexpr (Alpha == kAlphaProbClamp && Prob == kProbPoly3NoClamp) {
        name = "main1280_split_contig_input_exp2_prob_poly3_noclamp_alpha_poly3_clamp";
    } else if constexpr (Alpha == kAlphaProbClamp && Prob == kProbPoly3OutputClamp) {
        name = "main1280_split_contig_input_exp2_prob_poly3_outclamp_alpha_outclamp";
    } else if constexpr (Prob == kProbPoly3NoClamp) {
        name = "main1280_split_contig_input_exp2_prob_poly3_noclamp";
    } else if constexpr (ProbSumBf16 && Prob == kProbPoly4NoClamp) {
        name = "main1280_split_contig_input_exp2_prob_poly4_noclamp_sum_bf16";
    } else if constexpr (Prob == kProbPoly4NoClampBias) {
        name = "main1280_split_contig_input_exp2_prob_poly4_noclamp_bias";
    } else if constexpr (Prob == kProbLinearNoClamp) {
        name = "main1280_split_contig_input_exp2_prob_linear_noclamp";
    } else if constexpr (Prob == kProbPoly2NoClamp) {
        name = "main1280_split_contig_input_exp2_prob_poly2_noclamp";
    } else if constexpr (Prob == kProbPoly4NoClamp) {
        name = "main1280_split_contig_input_exp2_prob_poly4_noclamp";
    } else if constexpr (Prob == kProbPoly2) {
        name = "main1280_split_contig_input_exp2_prob_poly2";
    } else if constexpr (Prob == kProbPoly4) {
        name = "main1280_split_contig_input_exp2_prob_poly4";
    } else if constexpr (Prob == kProbLinear) {
        name = "main1280_split_contig_input_exp2_prob_linear";
    } else if constexpr (SkipAlphaExp) {
        name = "main1280_split_contig_input_exp2_alpha_one";
    } else if constexpr (UseFinalReciprocal && !IncludeKeyTail) {
        name = "main1280_split_contig_input_exp2_final_rcp_no_keytail";
    } else if constexpr (UseFinalReciprocal) {
        name = "main1280_split_contig_input_exp2_final_rcp";
    } else if constexpr (TailColBroadcast && RoundOutAcc) {
        name = "main1280_split_contig_input_exp2_tail_col_broadcast_out_acc_bf16";
    } else if constexpr (RoundOutAcc && !IncludeKeyTail) {
        name = "main1280_split_contig_input_exp2_out_acc_bf16_no_keytail";
    } else if constexpr (RoundOutAcc) {
        name = "main1280_split_contig_input_exp2_out_acc_bf16";
    } else if constexpr (TailColBroadcast) {
        name = "main1280_split_contig_input_exp2_tail_col_broadcast";
    } else if constexpr (!IncludeKeyTail) {
        name = "main1280_split_contig_input_exp2_no_keytail";
    } else if constexpr (TailIdx32) {
        name = "main1280_split_contig_input_exp2_tail_idx32";
    }
    std::printf("%s qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                name, QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, __bfloat162float(checksum_bf16));
}

template <int QRows, int KTile>
void run_main1280_split_contig_tail_prob_mask_only_variant(
    const Options& opts,
    const __nv_bfloat16* d_q,
    const __nv_bfloat16* d_k,
    const __nv_bfloat16* d_v,
    __nv_bfloat16* d_out,
    float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_tail_prob_mask_only_kernel
            <QRows, KTile, true>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_tail_prob_mask_only_kernel
            <QRows, KTile, true>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_split_contig_input_exp2_tail_prob_mask_only qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, __bfloat162float(checksum_bf16));
}

template <int QRows, int KTile>
void run_main1280_split_contig_tail16_8_variant(const Options& opts,
                                                const __nv_bfloat16* d_q,
                                                const __nv_bfloat16* d_k,
                                                const __nv_bfloat16* d_v,
                                                __nv_bfloat16* d_out,
                                                float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_tail16_8_kernel<QRows, KTile, true>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_tail16_8_kernel<QRows, KTile, true>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain * (kNMain + 24) * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_split_contig_input_exp2_tail16_8 qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, __bfloat162float(checksum_bf16));
}

template <int QRows, int KTile>
void run_main1280_split_contig_tail_helper_variant(
    const Options& opts,
    const __nv_bfloat16* d_q,
    const __nv_bfloat16* d_k,
    const __nv_bfloat16* d_v,
    __nv_bfloat16* d_out,
    float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_tail_helper_kernel<QRows, KTile, true>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_tail_helper_kernel<QRows, KTile, true>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_split_contig_input_exp2_tail_helper qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, __bfloat162float(checksum_bf16));
}

template <int QRows, int KTile, bool IncludeKeyTail = true>
void run_main1280_split_contig_gated_store_variant(const Options& opts,
                                                   const __nv_bfloat16* d_q,
                                                   const __nv_bfloat16* d_k,
                                                   const __nv_bfloat16* d_v,
                                                   const __nv_bfloat16* d_gates,
                                                   __nv_bfloat16* d_attn_tmp,
                                                   __nv_bfloat16* d_merged,
                                                   __nv_bfloat16* d_ref_merged,
                                                   float scale) {
    dim3 attn_grid(kNMain / QRows, kBH);
    dim3 merge_grid(kBatches * kNMain);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_input_kernel
            <QRows, KTile, true, IncludeKeyTail, false, false, false, false>
            <<<attn_grid, 1>>>(d_q, d_k, d_v, d_attn_tmp, scale);
        gate_merge_time_main1280_token_d64_kernel
            <<<merge_grid, 1>>>(d_attn_tmp, d_gates, d_merged);
        time_attention1301_main1280_split_contig_gated_store_kernel
            <QRows, KTile, true, IncludeKeyTail>
            <<<attn_grid, 1>>>(d_q, d_k, d_v, d_gates, d_merged, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> baseline_times_ms;
    std::vector<float> fused_times_ms;
    baseline_times_ms.reserve(opts.iters);
    fused_times_ms.reserve(opts.iters);

    __nv_bfloat16* baseline_merged = d_ref_merged ? d_ref_merged : d_merged;
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_input_kernel
            <QRows, KTile, true, IncludeKeyTail, false, false, false, false>
            <<<attn_grid, 1>>>(d_q, d_k, d_v, d_attn_tmp, scale);
        gate_merge_time_main1280_token_d64_kernel
            <<<merge_grid, 1>>>(d_attn_tmp, d_gates, baseline_merged);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        baseline_times_ms.push_back(ms);
    }

    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_gated_store_kernel
            <QRows, KTile, true, IncludeKeyTail>
            <<<attn_grid, 1>>>(d_q, d_k, d_v, d_gates, d_merged, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        fused_times_ms.push_back(ms);
    }

    if (d_ref_merged) {
        time_attention1301_main1280_split_contig_input_kernel
            <QRows, KTile, true, IncludeKeyTail, false, false, false, false>
            <<<attn_grid, 1>>>(d_q, d_k, d_v, d_attn_tmp, scale);
        gate_merge_time_main1280_token_d64_kernel
            <<<merge_grid, 1>>>(d_attn_tmp, d_gates, d_ref_merged);
        time_attention1301_main1280_split_contig_gated_store_kernel
            <QRows, KTile, true, IncludeKeyTail>
            <<<attn_grid, 1>>>(d_q, d_k, d_v, d_gates, d_merged, scale);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_merged, sizeof(checksum_bf16),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float baseline_best_ms =
        *std::min_element(baseline_times_ms.begin(), baseline_times_ms.end());
    float baseline_median_ms = percentile(baseline_times_ms, 0.5f);
    float fused_best_ms = *std::min_element(fused_times_ms.begin(), fused_times_ms.end());
    float fused_median_ms = percentile(fused_times_ms, 0.5f);
    constexpr int EffectiveK = IncludeKeyTail ? kN : kNMain;
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * EffectiveK * kD;
    double fused_tflops = real_flops / (static_cast<double>(fused_best_ms) * 1.0e-3) / 1.0e12;
    double baseline_tflops =
        real_flops / (static_cast<double>(baseline_best_ms) * 1.0e-3) / 1.0e12;
    double saved_gib = 2.0 * static_cast<double>(kBH) * kNMain * kD *
                       sizeof(__nv_bfloat16) / (1024.0 * 1024.0 * 1024.0);
    double speedup = baseline_best_ms / fused_best_ms;
    std::printf("main1280_split_contig_gated_store qrows=%d ktile=%d baseline_pair=%.3f ms median=%.3f ms %.2f TF/s fused=%.3f ms median=%.3f ms %.2f TF/s roof70=%.1f%% speedup=%.3fx saved_attn_boundary=%.3f GiB checksum=%.6f\n",
                QRows, KTile, baseline_best_ms, baseline_median_ms, baseline_tflops,
                fused_best_ms, fused_median_ms, fused_tflops,
                fused_tflops / 70.0 * 100.0, speedup, saved_gib,
                __bfloat162float(checksum_bf16));
}

template <int QRows, int KTile, bool IncludeKeyTail = true>
void run_main1280_split_contig_kt_input_variant(const Options& opts,
                                                const __nv_bfloat16* d_q,
                                                const __nv_bfloat16* d_k_t,
                                                const __nv_bfloat16* d_v,
                                                __nv_bfloat16* d_out,
                                                float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_kt_input_kernel
            <QRows, KTile, true, IncludeKeyTail>
            <<<grid, 1>>>(d_q, d_k_t, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_kt_input_kernel
            <QRows, KTile, true, IncludeKeyTail>
            <<<grid, 1>>>(d_q, d_k_t, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    constexpr int EffectiveK = IncludeKeyTail ? kN : kNMain;
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * EffectiveK * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain *
        (IncludeKeyTail ? kNPad : kNMain) * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    const char* name = "main1280_split_contig_kt_input_exp2";
    if constexpr (!IncludeKeyTail) {
        name = "main1280_split_contig_kt_input_exp2_no_keytail";
    }
    std::printf("%s qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                name, QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, __bfloat162float(checksum_bf16));
}

template <int QRows, int KTile, bool IncludeKeyTail = true>
void run_main1280_split_contig_seg2_variant(const Options& opts,
                                            const __nv_bfloat16* d_q,
                                            const __nv_bfloat16* d_k,
                                            const __nv_bfloat16* d_v,
                                            __nv_bfloat16* d_out,
                                            float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_seg2_kernel
            <QRows, KTile, true, IncludeKeyTail>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_seg2_kernel
            <QRows, KTile, true, IncludeKeyTail>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    constexpr int EffectiveK = IncludeKeyTail ? kN : kNMain;
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * EffectiveK * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain *
        (IncludeKeyTail ? kNPad : kNMain) * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    const char* name = "main1280_split_contig_seg2_exp2";
    if constexpr (!IncludeKeyTail) {
        name = "main1280_split_contig_seg2_exp2_no_keytail";
    }
    std::printf("%s qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                name, QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, __bfloat162float(checksum_bf16));
}

template <int QRows,
          int KTile,
          bool IncludeKeyTail = true,
          int QLoadLatency = 2,
          int KLoadLatency = 2,
          int VLoadLatency = 2>
void run_main1280_split_contig_input_lat_variant(const Options& opts,
                                                const __nv_bfloat16* d_q,
                                                const __nv_bfloat16* d_k,
                                                const __nv_bfloat16* d_v,
                                                __nv_bfloat16* d_out,
                                                float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_input_lat_kernel
            <QRows, KTile, IncludeKeyTail, QLoadLatency, KLoadLatency, VLoadLatency>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_input_lat_kernel
            <QRows, KTile, IncludeKeyTail, QLoadLatency, KLoadLatency, VLoadLatency>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    constexpr int EffectiveK = IncludeKeyTail ? kN : kNMain;
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * EffectiveK * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain *
        (IncludeKeyTail ? kNPad : kNMain) * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    const char* name = "main1280_split_contig_input_exp2_lat2";
    if constexpr (QLoadLatency == 2 && KLoadLatency == 0 && VLoadLatency == 0) {
        name = "main1280_split_contig_input_exp2_q_lat2";
    } else if constexpr (QLoadLatency == 0 && KLoadLatency == 2 && VLoadLatency == 0) {
        name = "main1280_split_contig_input_exp2_k_lat2";
    } else if constexpr (QLoadLatency == 0 && KLoadLatency == 0 && VLoadLatency == 2) {
        name = "main1280_split_contig_input_exp2_v_lat2";
    } else if constexpr (QLoadLatency == 0 && KLoadLatency == 2 && VLoadLatency == 2) {
        name = "main1280_split_contig_input_exp2_kv_lat2";
    }
    if constexpr (!IncludeKeyTail &&
                  QLoadLatency == 2 && KLoadLatency == 2 && VLoadLatency == 2) {
        name = "main1280_split_contig_input_exp2_lat2_no_keytail";
    } else if constexpr (!IncludeKeyTail &&
                         QLoadLatency == 2 && KLoadLatency == 0 &&
                         VLoadLatency == 0) {
        name = "main1280_split_contig_input_exp2_q_lat2_no_keytail";
    } else if constexpr (!IncludeKeyTail &&
                         QLoadLatency == 0 && KLoadLatency == 2 &&
                         VLoadLatency == 0) {
        name = "main1280_split_contig_input_exp2_k_lat2_no_keytail";
    } else if constexpr (!IncludeKeyTail &&
                         QLoadLatency == 0 && KLoadLatency == 0 &&
                         VLoadLatency == 2) {
        name = "main1280_split_contig_input_exp2_v_lat2_no_keytail";
    } else if constexpr (!IncludeKeyTail &&
                         QLoadLatency == 0 && KLoadLatency == 2 &&
                         VLoadLatency == 2) {
        name = "main1280_split_contig_input_exp2_kv_lat2_no_keytail";
    }
    std::printf("%s qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                name, QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, __bfloat162float(checksum_bf16));
}

template <int QRows,
          int KTile,
          bool IncludeKeyTail = true,
          bool TailColBroadcastPaddedTailLoad = false>
void run_main1280_split_contig_first_init_variant(const Options& opts,
                                                 const __nv_bfloat16* d_q,
                                                 const __nv_bfloat16* d_k,
                                                 const __nv_bfloat16* d_v,
                                                 __nv_bfloat16* d_out,
                                                 float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_first_init_kernel
            <QRows, KTile, IncludeKeyTail, TailColBroadcastPaddedTailLoad>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_first_init_kernel
            <QRows, KTile, IncludeKeyTail, TailColBroadcastPaddedTailLoad>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    constexpr int EffectiveK = IncludeKeyTail ? kN : kNMain;
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * EffectiveK * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain *
        (IncludeKeyTail ? kNPad : kNMain) * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    const char* name = "main1280_split_contig_input_exp2_first_init";
    if constexpr (TailColBroadcastPaddedTailLoad) {
        name =
            "main1280_split_contig_input_exp2_first_init_tail_col_broadcast_padded_tail_load";
    } else if constexpr (!IncludeKeyTail) {
        name = "main1280_split_contig_input_exp2_first_init_no_keytail";
    }
    std::printf("%s qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                name, QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, __bfloat162float(checksum_bf16));
}

template <int QRows,
          int KTile,
          bool IncludeKeyTail = true,
          bool RoundRowM = false>
void run_main1280_split_contig_row_state_bf16_variant(
    const Options& opts,
    const __nv_bfloat16* d_q,
    const __nv_bfloat16* d_k,
    const __nv_bfloat16* d_v,
    __nv_bfloat16* d_out,
    float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_row_state_bf16_kernel
            <QRows, KTile, IncludeKeyTail, RoundRowM>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_row_state_bf16_kernel
            <QRows, KTile, IncludeKeyTail, RoundRowM>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    constexpr int EffectiveK = IncludeKeyTail ? kN : kNMain;
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * EffectiveK * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain *
        (IncludeKeyTail ? kNPad : kNMain) * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    const char* name = "main1280_split_contig_input_exp2_row_l_bf16";
    if constexpr (RoundRowM && !IncludeKeyTail) {
        name = "main1280_split_contig_input_exp2_row_state_bf16_no_keytail";
    } else if constexpr (RoundRowM) {
        name = "main1280_split_contig_input_exp2_row_state_bf16";
    } else if constexpr (!IncludeKeyTail) {
        name = "main1280_split_contig_input_exp2_row_l_bf16_no_keytail";
    }
    std::printf("%s qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                name, QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, __bfloat162float(checksum_bf16));
}

template <int QRows, int KTile, bool IncludeKeyTail = true>
void run_main1280_split_contig_score_av_lb_variant(const Options& opts,
                                                   const __nv_bfloat16* d_q,
                                                   const __nv_bfloat16* d_k,
                                                   const __nv_bfloat16* d_v,
                                                   __nv_bfloat16* d_out,
                                                   float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_score_av_lb_kernel
            <QRows, KTile, IncludeKeyTail>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_score_av_lb_kernel
            <QRows, KTile, IncludeKeyTail>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    constexpr int EffectiveK = IncludeKeyTail ? kN : kNMain;
    constexpr int LogicalK = IncludeKeyTail
        ? ((kN + KTile - 1) / KTile) * KTile
        : kNMain;
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * EffectiveK * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain * LogicalK * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    const char* name = IncludeKeyTail
        ? "main1280_split_contig_score_av_lb"
        : "main1280_split_contig_score_av_lb_no_keytail";
    std::printf("%s qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                name, QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, __bfloat162float(checksum_bf16));
}

template <int QRows, int KTile, bool IncludeKeyTail = true>
void run_main1280_split_contig_tile_local_softmax_lb_variant(const Options& opts,
                                                            const __nv_bfloat16* d_q,
                                                            const __nv_bfloat16* d_k,
                                                            const __nv_bfloat16* d_v,
                                                            __nv_bfloat16* d_out,
                                                            float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_tile_local_softmax_lb_kernel
            <QRows, KTile, IncludeKeyTail>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_tile_local_softmax_lb_kernel
            <QRows, KTile, IncludeKeyTail>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    constexpr int EffectiveK = IncludeKeyTail ? kN : kNMain;
    constexpr int LogicalK = IncludeKeyTail
        ? ((kN + KTile - 1) / KTile) * KTile
        : kNMain;
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * EffectiveK * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain * LogicalK * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    const char* name = IncludeKeyTail
        ? "main1280_split_contig_tile_local_softmax_lb"
        : "main1280_split_contig_tile_local_softmax_lb_no_keytail";
    std::printf("%s qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                name, QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, __bfloat162float(checksum_bf16));
}

template <int QRows, int KTile>
void run_main1280_split_contig_tail_first_variant(const Options& opts,
                                                  const __nv_bfloat16* d_q,
                                                  const __nv_bfloat16* d_k,
                                                  const __nv_bfloat16* d_v,
                                                  __nv_bfloat16* d_out,
                                                  float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_tail_first_kernel
            <QRows, KTile, true>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_tail_first_kernel
            <QRows, KTile, true>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_split_contig_tail_first_exp2 qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, __bfloat162float(checksum_bf16));
}

template <int QRows, int KTile>
void run_main1280_split_contig_tail_first_padded_tail_load_variant(
    const Options& opts,
    const __nv_bfloat16* d_q,
    const __nv_bfloat16* d_k,
    const __nv_bfloat16* d_v,
    __nv_bfloat16* d_out,
    float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_tail_first_padded_tail_load_kernel
            <QRows, KTile, true>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_tail_first_padded_tail_load_kernel
            <QRows, KTile, true>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops =
        real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_split_contig_tail_first_padded_tail_load_exp2 qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                QRows, KTile, best_ms, median_ms, real_tflops,
                logical_tflops, real_tflops / 70.0 * 100.0,
                __bfloat162float(checksum_bf16));
}

template <int QRows, int KTile, bool TailColBroadcast = false>
void run_main1280_split_contig_padded_tail_load_variant(const Options& opts,
                                                        const __nv_bfloat16* d_q,
                                                        const __nv_bfloat16* d_k,
                                                        const __nv_bfloat16* d_v,
                                                        __nv_bfloat16* d_out,
                                                        float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_padded_tail_load_kernel
            <QRows, KTile, true, TailColBroadcast>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_padded_tail_load_kernel
            <QRows, KTile, true, TailColBroadcast>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    const char* name = TailColBroadcast
        ? "main1280_split_contig_tail_col_broadcast_padded_tail_load_exp2"
        : "main1280_split_contig_padded_tail_load_exp2";
    std::printf("%s qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                name, QRows, KTile, best_ms, median_ms, real_tflops,
                logical_tflops, real_tflops / 70.0 * 100.0,
                __bfloat162float(checksum_bf16));
}

template <int QRows, int KTile>
void run_main1280_split_contig_two_pass_state_variant(const Options& opts,
                                                      const __nv_bfloat16* d_q,
                                                      const __nv_bfloat16* d_k,
                                                      const __nv_bfloat16* d_v,
                                                      float* d_state_acc,
                                                      float* d_state_m,
                                                      float* d_state_l,
                                                      __nv_bfloat16* d_out,
                                                      float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_split_contig_state1280_kernel
            <QRows, KTile, true>
            <<<grid, 1>>>(d_q, d_k, d_v, d_state_acc, d_state_m, d_state_l,
                          scale);
        time_attention1301_main1280_split_contig_tail_finalize_kernel
            <QRows, KTile, true>
            <<<grid, 1>>>(d_q, d_k, d_v, d_state_acc, d_state_m, d_state_l,
                          d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_split_contig_state1280_kernel
            <QRows, KTile, true>
            <<<grid, 1>>>(d_q, d_k, d_v, d_state_acc, d_state_m, d_state_l,
                          scale);
        time_attention1301_main1280_split_contig_tail_finalize_kernel
            <QRows, KTile, true>
            <<<grid, 1>>>(d_q, d_k, d_v, d_state_acc, d_state_m, d_state_l,
                          d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_split_contig_two_pass_state_exp2 qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, __bfloat162float(checksum_bf16));
}

void run_main1280_q64k32_exp2_split_d32_variant(const Options& opts,
                                                const __nv_bfloat16* d_q,
                                                const __nv_bfloat16* d_k,
                                                const __nv_bfloat16* d_v,
                                                __nv_bfloat16* d_out,
                                                float scale) {
    dim3 grid(kNMain / 64, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_q64k32_exp2_split_d32_kernel
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_q64k32_exp2_split_d32_kernel
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops = 4.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_sum_f32_exp2_split_d32 qrows=64 ktile=32 best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

template <int MainKTile, int TailQRows = 64, int TailKTile = 64, bool UseExp2 = false>
void run_split_tail_pair_variant(const Options& opts,
                                 const __nv_bfloat16* d_q,
                                 const __nv_bfloat16* d_k,
                                 const __nv_bfloat16* d_v,
                                 __nv_bfloat16* d_out,
                                 float scale) {
    constexpr int QRows = 64;
    static_assert(kNMain % TailQRows == 0);
    dim3 grid_main(kNMain / QRows, kBH);
    dim3 grid_tail(ceildiv(kN - kNMain, TailQRows), kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_kernel<QRows, MainKTile, true, UseExp2>
            <<<grid_main, 1>>>(d_q, d_k, d_v, d_out, scale);
        time_attention1301_tail_offset_kernel<TailQRows, TailKTile, kNMain / TailQRows, UseExp2>
            <<<grid_tail, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_kernel<QRows, MainKTile, true, UseExp2>
            <<<grid_main, 1>>>(d_q, d_k, d_v, d_out, scale);
        time_attention1301_tail_offset_kernel<TailQRows, TailKTile, kNMain / TailQRows, UseExp2>
            <<<grid_tail, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum0_bf16{};
    __nv_bfloat16 checksum_tail_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum0_bf16, d_out, sizeof(checksum0_bf16),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&checksum_tail_bf16, d_out + static_cast<std::size_t>(kNMain) * kD,
                          sizeof(checksum_tail_bf16), cudaMemcpyDeviceToHost));
    float checksum0 = __bfloat162float(checksum0_bf16);
    float checksum_tail = __bfloat162float(checksum_tail_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kN * kN * kD;
    double logical_main_keys = static_cast<double>(kNMain + MainKTile);
    double logical_tail_qrows =
        static_cast<double>(ceildiv(kN - kNMain, TailQRows) * TailQRows);
    double logical_tail_keys = static_cast<double>(ceildiv(kN, TailKTile) * TailKTile);
    double logical_flops =
        4.0 * static_cast<double>(kBH) * kNMain * logical_main_keys * kD +
        4.0 * static_cast<double>(kBH) * logical_tail_qrows * logical_tail_keys * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops = logical_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("split_tail_pair_sum_f32%s main_ktile=%d tail_qrows=%d tail_ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum0=%.6f checksum_tail=%.6f\n",
                UseExp2 ? "_exp2" : "", MainKTile, TailQRows, TailKTile,
                best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum0, checksum_tail);
}

template <int QRows, int KTile, bool UseExp2 = false>
void run_tail_only_variant(const Options& opts,
                           const __nv_bfloat16* d_q,
                           const __nv_bfloat16* d_k,
                           const __nv_bfloat16* d_v,
                           __nv_bfloat16* d_out,
                           float scale) {
    static_assert(kNMain % QRows == 0);
    constexpr int QBlockOffset = kNMain / QRows;
    constexpr int TailRows = kN - kNMain;
    dim3 grid(ceildiv(TailRows, QRows), kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_tail_offset_kernel<QRows, KTile, QBlockOffset, UseExp2>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_tail_offset_kernel<QRows, KTile, QBlockOffset, UseExp2>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out + static_cast<std::size_t>(kNMain) * kD,
                          sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * TailRows * kN * kD;
    double logical_qrows = static_cast<double>(ceildiv(TailRows, QRows) * QRows);
    double logical_keys = static_cast<double>(ceildiv(kN, KTile) * KTile);
    double logical_flops = 4.0 * static_cast<double>(kBH) * logical_qrows * logical_keys * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops = logical_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("tail_only%s qrows=%d ktile=%d launches=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                UseExp2 ? "_exp2" : "", QRows, KTile, ceildiv(TailRows, QRows),
                best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

template <int QRows, int KTile>
void run_main1280_direct_ptr_variant(const Options& opts,
                                     const __nv_bfloat16* d_q,
                                     const __nv_bfloat16* d_k,
                                     const __nv_bfloat16* d_v,
                                     __nv_bfloat16* d_out,
                                     float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_direct_ptr_kernel<QRows, KTile>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_direct_ptr_kernel<QRows, KTile>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16),
                          cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain * (kNMain + KTile) * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_direct_ptr_sum_f32 qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

template <int QRows, int KTile>
void run_main1280_qkv_direct_rotary_variant(const Options& opts,
                                            const __nv_bfloat16* d_qkv,
                                            const float* d_cos,
                                            const float* d_sin,
                                            __nv_bfloat16* d_out,
                                            float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_qkv_direct_rotary_kernel<QRows, KTile, true>
            <<<grid, 1>>>(d_qkv, d_cos, d_sin, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_qkv_direct_rotary_kernel<QRows, KTile, true>
            <<<grid, 1>>>(d_qkv, d_cos, d_sin, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16),
                          cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain * (kNMain + KTile) * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_qkv_direct_rotary_exp2 qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

template <int QRows>
void run_main1280_qk_only_variant(const Options& opts,
                                  const __nv_bfloat16* d_q,
                                  const __nv_bfloat16* d_k,
                                  __nv_bfloat16* d_out,
                                  float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_qk_only_kernel<QRows>
            <<<grid, 1>>>(d_q, d_k, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_qk_only_kernel<QRows>
            <<<grid, 1>>>(d_q, d_k, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 2.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops = 2.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_qk_only qrows=%d ktile=64 best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                QRows, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

template <int QRows>
void run_main1280_qk_only_kt_variant(const Options& opts,
                                     const __nv_bfloat16* d_q,
                                     const __nv_bfloat16* d_k_t,
                                     __nv_bfloat16* d_out,
                                     float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_qk_only_kt_kernel<QRows>
            <<<grid, 1>>>(d_q, d_k_t, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_qk_only_kt_kernel<QRows>
            <<<grid, 1>>>(d_q, d_k_t, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 2.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops = 2.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_qk_only_kt qrows=%d ktile=64 best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                QRows, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

template <int QRows>
void run_main1280_av_const_variant(const Options& opts,
                                   const __nv_bfloat16* d_v,
                                   __nv_bfloat16* d_out) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_av_const_kernel<QRows>
            <<<grid, 1>>>(d_v, d_out);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_av_const_kernel<QRows>
            <<<grid, 1>>>(d_v, d_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 2.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops = 2.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_av_const qrows=%d ktile=64 best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                QRows, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

template <int QRows>
void run_main1280_split_global_variant(const Options& opts,
                                       const __nv_bfloat16* d_q,
                                       const __nv_bfloat16* d_k,
                                       const __nv_bfloat16* d_v,
                                       __nv_bfloat16* d_p,
                                       __nv_bfloat16* d_out,
                                       float scale) {
    constexpr int QBlocks = kNMain / QRows;
    constexpr int KTiles = (kN + kKTile - 1) / kKTile;
    dim3 grid(QBlocks, kBH);

    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_qk_store_p_kernel<QRows>
            <<<grid, 1>>>(d_q, d_k, d_p, scale);
        time_attention1301_main1280_av_load_p_kernel<QRows>
            <<<grid, 1>>>(d_p, d_v, d_out);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> qk_times_ms;
    std::vector<float> av_times_ms;
    std::vector<float> pair_times_ms;
    qk_times_ms.reserve(opts.iters);
    av_times_ms.reserve(opts.iters);
    pair_times_ms.reserve(opts.iters);

    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_qk_store_p_kernel<QRows>
            <<<grid, 1>>>(d_q, d_k, d_p, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        qk_times_ms.push_back(ms);
    }

    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_av_load_p_kernel<QRows>
            <<<grid, 1>>>(d_p, d_v, d_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        av_times_ms.push_back(ms);
    }

    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_qk_store_p_kernel<QRows>
            <<<grid, 1>>>(d_q, d_k, d_p, scale);
        time_attention1301_main1280_av_load_p_kernel<QRows>
            <<<grid, 1>>>(d_p, d_v, d_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        pair_times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float qk_best_ms = *std::min_element(qk_times_ms.begin(), qk_times_ms.end());
    float av_best_ms = *std::min_element(av_times_ms.begin(), av_times_ms.end());
    float pair_best_ms = *std::min_element(pair_times_ms.begin(), pair_times_ms.end());
    float pair_median_ms = percentile(pair_times_ms, 0.5f);
    double half_flops = 2.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double pair_flops = 2.0 * half_flops;
    double qk_tflops = half_flops / (static_cast<double>(qk_best_ms) * 1.0e-3) / 1.0e12;
    double av_tflops = half_flops / (static_cast<double>(av_best_ms) * 1.0e-3) / 1.0e12;
    double pair_tflops = pair_flops / (static_cast<double>(pair_best_ms) * 1.0e-3) / 1.0e12;
    double p_elems = static_cast<double>(kBH) * QBlocks * KTiles * QRows * kKTile;
    double p_roundtrip_gb = p_elems * sizeof(__nv_bfloat16) * 2.0 / 1.0e9;
    double p_roundtrip_gbs =
        p_roundtrip_gb / (static_cast<double>(pair_best_ms) * 1.0e-3);
    std::printf("main1280_split_global qrows=%d ktile=64 qk=%.3f ms %.2f TF/s av=%.3f ms %.2f TF/s pair=%.3f ms median=%.3f ms pair_math=%.2f TF/s roof70=%.1f%% p_roundtrip=%.2f GB %.1f GB/s checksum=%.6f\n",
                QRows, qk_best_ms, qk_tflops, av_best_ms, av_tflops,
                pair_best_ms, pair_median_ms, pair_tflops,
                pair_tflops / 70.0 * 100.0, p_roundtrip_gb,
                p_roundtrip_gbs, checksum);
}

void run_main1280_split_d32_lower_bound_variant(const Options& opts,
                                                const __nv_bfloat16* d_q,
                                                const __nv_bfloat16* d_k,
                                                const __nv_bfloat16* d_v,
                                                __nv_bfloat16* d_out,
                                                float scale) {
    dim3 grid(kNMain / 64, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_q64k64_main1280_score_av_lb_split_d32_kernel
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_q64k64_main1280_score_av_lb_split_d32_kernel
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops = 4.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_score_av_lb_split_d32 qrows=64 ktile=64 best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

void run_main1280_prescale_q_lower_bound_variant(const Options& opts,
                                                 const __nv_bfloat16* d_q,
                                                 const __nv_bfloat16* d_k,
                                                 const __nv_bfloat16* d_v,
                                                 __nv_bfloat16* d_out,
                                                 float scale) {
    dim3 grid(kNMain / 64, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_q64k64_main1280_score_av_lb_prescale_q_kernel
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_q64k64_main1280_score_av_lb_prescale_q_kernel
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops = 4.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_score_av_lb_prescale_q qrows=64 ktile=64 best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

void run_focused_variant(const Options& opts,
                         const __nv_bfloat16* d_q,
                         const __nv_bfloat16* d_k,
                         const __nv_bfloat16* d_v,
                         __nv_bfloat16* d_out,
                         float scale) {
    if (opts.variant == "main1280_q32k32_exp2") {
        run_main1280_variant<32, false, true, true, 32>(
            opts, d_q, d_k, d_v, d_out, scale);
        return;
    }
    if (opts.variant == "main1280_q64k16_exp2") {
        run_main1280_variant<64, false, true, true, 16>(
            opts, d_q, d_k, d_v, d_out, scale);
        return;
    }
    if (opts.variant == "main1280_q64k32_exp2" ||
        opts.variant == "main1280_sum_f32_exp2") {
        run_main1280_variant<64, false, true, true, 32>(
            opts, d_q, d_k, d_v, d_out, scale);
        return;
    }
    if (opts.variant == "main1280_q64k64_exp2") {
        run_main1280_variant<64, false, true, true>(
            opts, d_q, d_k, d_v, d_out, scale);
        return;
    }
    if (opts.variant == "main1280_q128k32_exp2") {
        run_main1280_variant<128, false, true, true, 32>(
            opts, d_q, d_k, d_v, d_out, scale);
        return;
    }
    if (opts.variant == "main1280_q64k32_exp2_sum_bf16" ||
        opts.variant == "main1280_exp2_sum_bf16") {
        run_main1280_variant<64, false, false, true, 32>(
            opts, d_q, d_k, d_v, d_out, scale);
        return;
    }
    if (opts.variant == "main1280_q64k32_exp2_score_bf16" ||
        opts.variant == "main1280_sum_f32_exp2_score_bf16") {
        run_main1280_variant<64, false, true, true, 32, true>(
            opts, d_q, d_k, d_v, d_out, scale);
        return;
    }
    if (opts.variant == "main1280_q64k32_exp2_prescale_q" ||
        opts.variant == "main1280_sum_f32_exp2_prescale_q") {
        run_main1280_variant<64, false, true, true, 32, false, true>(
            opts, d_q, d_k, d_v, d_out, scale);
        return;
    }
    if (opts.variant == "main1280_q64k32_exp2_split_d32" ||
        opts.variant == "main1280_sum_f32_exp2_split_d32") {
        run_main1280_q64k32_exp2_split_d32_variant(
            opts, d_q, d_k, d_v, d_out, scale);
        return;
    }
    if (opts.variant == "split_tail_q64k32_tail32k32_exp2") {
        run_split_tail_pair_variant<32, 32, 32, true>(
            opts, d_q, d_k, d_v, d_out, scale);
        return;
    }
    throw std::runtime_error("unknown --variant: " + opts.variant);
}

void launch_main1280_q64k32_exp2_once(const __nv_bfloat16* d_q,
                                      const __nv_bfloat16* d_k,
                                      const __nv_bfloat16* d_v,
                                      __nv_bfloat16* d_out,
                                      float scale) {
    dim3 grid(kNMain / 64, kBH);
    time_attention1301_main1280_kernel<64, 32, true, true>
        <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void compare_outputs_to_baseline(const __nv_bfloat16* d_ref,
                                 const __nv_bfloat16* d_out,
                                 std::size_t elems) {
    std::vector<__nv_bfloat16> ref(elems);
    std::vector<__nv_bfloat16> out(elems);
    CUDA_CHECK(cudaMemcpy(ref.data(), d_ref, elems * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, elems * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));

    double max_abs = 0.0;
    double sum_sq = 0.0;
    double ref_sum_sq = 0.0;
    for (std::size_t i = 0; i < elems; ++i) {
        double ref_v = static_cast<double>(__bfloat162float(ref[i]));
        double out_v = static_cast<double>(__bfloat162float(out[i]));
        double diff = out_v - ref_v;
        double abs_diff = std::abs(diff);
        max_abs = std::max(max_abs, abs_diff);
        sum_sq += diff * diff;
        ref_sum_sq += ref_v * ref_v;
    }

    double rms = std::sqrt(sum_sq / static_cast<double>(elems));
    double ref_rms = std::sqrt(ref_sum_sq / static_cast<double>(elems));
    double rel_rms = ref_rms > 0.0 ? rms / ref_rms : 0.0;
    std::printf("compare_vs_main1280_q64k32_exp2 elems=%zu max_abs=%.9g rms=%.9g ref_rms=%.9g rel_rms=%.9g\n",
                elems, max_abs, rms, ref_rms, rel_rms);
}

bool should_describe(const Options& opts, const char* name) {
    return opts.variant == "all" || opts.variant == name;
}

template <typename Kernel>
void describe_kernel_entry(const Options& opts,
                           const char* filter_name,
                           const char* print_name,
                           Kernel kernel,
                           dim3 grid) {
    if (!should_describe(opts, filter_name)) return;

    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                                             kernel,
                                                             1,
                                                             0));

    cudaDeviceProp prop{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int shared_limit = attr.sharedSizeBytes > 0
        ? static_cast<int>(prop.sharedMemPerMultiprocessor / attr.sharedSizeBytes)
        : prop.maxBlocksPerMultiProcessor;
    double sm_waves = static_cast<double>(grid.x) *
                      static_cast<double>(grid.y) *
                      static_cast<double>(std::max(1u, grid.z)) /
                      static_cast<double>(prop.multiProcessorCount);

    std::printf(
        "%-58s grid=(%u,%u,%u) waves/SM=%.1f attr_regs=%d "
        "static_shared=%zuB shared_limit=%d occupancy_active_cta_per_sm=%d "
        "max_threads_per_block=%d local=%zuB const=%zuB ptx=%d binary=%d\n",
        print_name,
        grid.x,
        grid.y,
        grid.z,
        sm_waves,
        attr.numRegs,
        attr.sharedSizeBytes,
        shared_limit,
        active_blocks,
        attr.maxThreadsPerBlock,
        attr.localSizeBytes,
        attr.constSizeBytes,
        attr.ptxVersion,
        attr.binaryVersion);
}

template <typename Kernel>
void describe_kernel(const Options& opts, const char* name, Kernel kernel, dim3 grid) {
    describe_kernel_entry(opts, name, name, kernel, grid);
}

void describe_all(const Options& opts, const cudaDeviceProp& prop) {
    std::printf("Device resources: shared_per_sm=%zuB regs_per_sm=%d max_blocks_per_sm=%d\n",
                prop.sharedMemPerMultiprocessor,
                prop.regsPerMultiprocessor,
                prop.maxBlocksPerMultiProcessor);

    describe_kernel(opts,
                    "main1280_q64k32_exp2",
                    time_attention1301_main1280_kernel<64, 32, true, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q32k32_exp2",
                    time_attention1301_main1280_kernel<32, 32, true, true>,
                    dim3(kNMain / 32, kBH));
    describe_kernel(opts,
                    "main1280_q64k16_exp2",
                    time_attention1301_main1280_kernel<64, 16, true, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k64_exp2",
                    time_attention1301_main1280_kernel<64, 64, true, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q128k32_exp2",
                    time_attention1301_main1280_kernel<128, 32, true, true>,
                    dim3(kNMain / 128, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_input",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_input_no_keytail",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, false, false, false, false, false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_tail_idx32",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, true, false, false, false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_tail_col_broadcast",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, true, false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_tail_col_broadcast_out_acc_bf16",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, true, true, false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_alpha_one",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, false, true,
                         false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_prob_linear",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, false, false,
                         kProbLinear>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_prob_linear_noclamp",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, false, false,
                         kProbLinearNoClamp>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_prob_poly2",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, false, false,
                         kProbPoly2>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_prob_poly4",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, false, false,
                         kProbPoly4>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_prob_poly2_noclamp",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, false, false,
                         kProbPoly2NoClamp>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_prob_poly3_noclamp",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, false, false,
                         kProbPoly3NoClamp>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_prob_poly3_noclamp_alpha_poly3_clamp",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, false, false,
                         kProbPoly3NoClamp, false, kAlphaProbClamp>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_prob_poly3_noclamp_alpha_poly3_clamp_final_rcp",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, true, false, false, false,
                         kProbPoly3NoClamp, false, kAlphaProbClamp>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_prob_poly3_outclamp_alpha_outclamp",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, false, false,
                         kProbPoly3OutputClamp, false, kAlphaProbClamp>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_prob_poly3_noclamp_alpha_linear_clamp",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, false, false,
                         kProbPoly3NoClamp, false, kAlphaLinearClamp>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_prob_rational4_alpha_rational4",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, false, false,
                         kProbRational4, false, kAlphaProbClamp>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_prob_poly4_noclamp",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, false, false,
                         kProbPoly4NoClamp>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_prob_poly4_noclamp_sum_bf16",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, false, false,
                         kProbPoly4NoClamp, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_prob_poly4_noclamp_bias",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, false, false,
                         kProbPoly4NoClampBias>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_tail_prob_mask_only",
                    time_attention1301_main1280_split_contig_tail_prob_mask_only_kernel
                        <64, 32, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_tail16_8",
                    time_attention1301_main1280_split_contig_tail16_8_kernel
                        <64, 32, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_tail_helper",
                    time_attention1301_main1280_split_contig_tail_helper_kernel
                        <64, 32, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_seg2",
                    time_attention1301_main1280_split_contig_seg2_kernel
                        <64, 32, true, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_seg2_no_keytail",
                    time_attention1301_main1280_split_contig_seg2_kernel
                        <64, 32, true, false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_tail_first",
                    time_attention1301_main1280_split_contig_tail_first_kernel
                        <64, 32, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_tail_first_padded_tail_load",
                    time_attention1301_main1280_split_contig_tail_first_padded_tail_load_kernel
                        <64, 32, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_padded_tail_load",
                    time_attention1301_main1280_split_contig_padded_tail_load_kernel
                        <64, 32, true, false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_tail_col_broadcast_padded_tail_load",
                    time_attention1301_main1280_split_contig_padded_tail_load_kernel
                        <64, 32, true, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_out_acc_bf16",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, false, false, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_out_acc_bf16_no_keytail",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, false, false, false, false, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_final_rcp",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, true, false, true, false, false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_final_rcp_no_keytail",
                    time_attention1301_main1280_split_contig_input_kernel
                        <64, 32, true, false, false, true, false, false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_first_init",
                    time_attention1301_main1280_split_contig_first_init_kernel
                        <64, 32, true, false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_first_init_tail_col_broadcast_padded_tail_load",
                    time_attention1301_main1280_split_contig_first_init_kernel
                        <64, 32, true, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_first_init_no_keytail",
                    time_attention1301_main1280_split_contig_first_init_kernel
                        <64, 32, false, false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_row_l_bf16",
                    time_attention1301_main1280_split_contig_row_state_bf16_kernel
                        <64, 32, true, false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_row_l_bf16_no_keytail",
                    time_attention1301_main1280_split_contig_row_state_bf16_kernel
                        <64, 32, false, false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_row_state_bf16",
                    time_attention1301_main1280_split_contig_row_state_bf16_kernel
                        <64, 32, true, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_row_state_bf16_no_keytail",
                    time_attention1301_main1280_split_contig_row_state_bf16_kernel
                        <64, 32, false, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_score_av_lb",
                    time_attention1301_main1280_split_contig_score_av_lb_kernel
                        <64, 32, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_score_av_lb_no_keytail",
                    time_attention1301_main1280_split_contig_score_av_lb_kernel
                        <64, 32, false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_tile_local_softmax_lb",
                    time_attention1301_main1280_split_contig_tile_local_softmax_lb_kernel
                        <64, 32, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_tile_local_softmax_lb_no_keytail",
                    time_attention1301_main1280_split_contig_tile_local_softmax_lb_kernel
                        <64, 32, false>,
                    dim3(kNMain / 64, kBH));
    describe_kernel_entry(opts,
                          "main1280_q64k32_exp2_split_contig_two_pass_state",
                          "main1280_q64k32_exp2_split_contig_two_pass_state1280",
                          time_attention1301_main1280_split_contig_state1280_kernel
                              <64, 32, true>,
                          dim3(kNMain / 64, kBH));
    describe_kernel_entry(opts,
                          "main1280_q64k32_exp2_split_contig_two_pass_state",
                          "main1280_q64k32_exp2_split_contig_two_pass_tail_finalize",
                          time_attention1301_main1280_split_contig_tail_finalize_kernel
                              <64, 32, true>,
                          dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "main1280_q64k32_exp2_split_contig_gated_store",
                    time_attention1301_main1280_split_contig_gated_store_kernel
                        <64, 32, true, true>,
                    dim3(kNMain / 64, kBH));
    describe_kernel(opts,
                    "gate_merge_time_main1280_token_d64",
                    gate_merge_time_main1280_token_d64_kernel,
                    dim3(kBatches * kNMain));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        if (opts.compare_baseline && opts.variant == "all") {
            throw std::runtime_error("--compare-baseline requires --variant NAME");
        }

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::printf("GPU: %s, SM %d.%d, SMs %d\n",
                    prop.name, prop.major, prop.minor, prop.multiProcessorCount);
        std::printf("Shape: BH=%d N=%d NPad=%d D=%d, CUDA Tile streaming QK-softmax-AV\n",
                    kBH, kN, kNPad, kD);
        if (opts.describe) {
            describe_all(opts, prop);
            return 0;
        }

        size_t padded_elems = static_cast<size_t>(kBH) * kNPad * kD;
        size_t unpadded_elems = static_cast<size_t>(kBH) * kN * kD;
        size_t qkv_elems = static_cast<size_t>(kBatches) * kN * kQkvStride;
        size_t split_padded_elems =
            static_cast<size_t>(kBatches) * kNPad * kHeads * kD;
        size_t rotary_elems = static_cast<size_t>(kN) * (kD / 2);
        size_t out_elems = static_cast<size_t>(kBH) * kN * kD;
        size_t gate_elems = static_cast<size_t>(kBatches) * kN * kHeads;
        size_t merged_main_elems =
            static_cast<size_t>(kBatches) * kNMain * kHeads * kD;
        size_t state_acc_elems = static_cast<size_t>(kBH) * kNMain * kD;
        size_t state_row_elems = static_cast<size_t>(kBH) * kNMain;
        constexpr int split_qrows = 64;
        constexpr int split_qblocks = kNMain / split_qrows;
        constexpr int split_ktiles = (kN + kKTile - 1) / kKTile;
        size_t split_p_elems =
            static_cast<size_t>(kBH) * split_qblocks * split_ktiles * split_qrows * kKTile;
        __nv_bfloat16* d_q = nullptr;
        __nv_bfloat16* d_k = nullptr;
        __nv_bfloat16* d_v = nullptr;
        __nv_bfloat16* d_q_masked = nullptr;
        __nv_bfloat16* d_k_masked = nullptr;
        __nv_bfloat16* d_k_t_masked = nullptr;
        __nv_bfloat16* d_v_masked = nullptr;
        __nv_bfloat16* d_qkv_masked = nullptr;
        __nv_bfloat16* d_q_split_contig = nullptr;
        __nv_bfloat16* d_k_split_contig = nullptr;
        __nv_bfloat16* d_k_t_split_contig = nullptr;
        __nv_bfloat16* d_v_split_contig = nullptr;
        __nv_bfloat16* d_q_split_padded = nullptr;
        __nv_bfloat16* d_k_split_padded = nullptr;
        __nv_bfloat16* d_v_split_padded = nullptr;
        __nv_bfloat16* d_gates = nullptr;
        __nv_bfloat16* d_merged_main = nullptr;
        __nv_bfloat16* d_ref_merged_main = nullptr;
        float* d_cos = nullptr;
        float* d_sin = nullptr;
        float* d_state_acc = nullptr;
        float* d_state_m = nullptr;
        float* d_state_l = nullptr;
        __nv_bfloat16* d_out = nullptr;
        __nv_bfloat16* d_ref = nullptr;
        __nv_bfloat16* d_p_split_q64 = nullptr;
        CUDA_CHECK(cudaMalloc(&d_q, padded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_k, padded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_v, padded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_q_masked, unpadded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_k_masked, unpadded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_k_t_masked, unpadded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_v_masked, unpadded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_qkv_masked, qkv_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_q_split_contig, unpadded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_k_split_contig, unpadded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_k_t_split_contig, unpadded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_v_split_contig, unpadded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_q_split_padded, split_padded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_k_split_padded, split_padded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_v_split_padded, split_padded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_gates, gate_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_merged_main, merged_main_elems * sizeof(__nv_bfloat16)));
        if (opts.compare_baseline) {
            CUDA_CHECK(cudaMalloc(&d_ref_merged_main,
                                  merged_main_elems * sizeof(__nv_bfloat16)));
        }
        CUDA_CHECK(cudaMalloc(&d_cos, rotary_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sin, rotary_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_state_acc, state_acc_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_state_m, state_row_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_state_l, state_row_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(__nv_bfloat16)));
        if (opts.compare_baseline) {
            CUDA_CHECK(cudaMalloc(&d_ref, out_elems * sizeof(__nv_bfloat16)));
        }
        CUDA_CHECK(cudaMalloc(&d_p_split_q64, split_p_elems * sizeof(__nv_bfloat16)));

        int fill_blocks = static_cast<int>((padded_elems + kInitTile - 1) / kInitTile);
        fill_bf16_kernel<<<fill_blocks, 1>>>(d_q, static_cast<long long>(padded_elems));
        fill_bf16_kernel<<<fill_blocks, 1>>>(d_k, static_cast<long long>(padded_elems));
        fill_bf16_kernel<<<fill_blocks, 1>>>(d_v, static_cast<long long>(padded_elems));
        int fill_masked_blocks =
            static_cast<int>((unpadded_elems + kInitTile - 1) / kInitTile);
        fill_bf16_kernel<<<fill_masked_blocks, 1>>>(
            d_q_masked, static_cast<long long>(unpadded_elems));
        fill_bf16_kernel<<<fill_masked_blocks, 1>>>(
            d_k_masked, static_cast<long long>(unpadded_elems));
        fill_bf16_kernel<<<fill_masked_blocks, 1>>>(
            d_v_masked, static_cast<long long>(unpadded_elems));
        int fill_gate_blocks =
            static_cast<int>((gate_elems + kInitTile - 1) / kInitTile);
        fill_gate_bf16_kernel<<<fill_gate_blocks, 1>>>(
            d_gates, static_cast<long long>(gate_elems));
        transpose_k_nd_to_dn_kernel<<<fill_masked_blocks, 1>>>(
            d_k_masked, d_k_t_masked, static_cast<long long>(unpadded_elems));
        pack_time_qkv_kernel<<<fill_masked_blocks, 1>>>(
            d_q_masked, d_k_masked, d_v_masked, d_qkv_masked,
            static_cast<long long>(unpadded_elems));
        pack_time_split_contig_kernel<<<fill_masked_blocks, 1>>>(
            d_q_masked, d_k_masked, d_v_masked,
            d_q_split_contig, d_k_split_contig, d_v_split_contig,
            static_cast<long long>(unpadded_elems));
        pack_time_split_contig_k_transposed_kernel<<<fill_masked_blocks, 1>>>(
            d_k_split_contig, d_k_t_split_contig,
            static_cast<long long>(unpadded_elems));
        int fill_split_padded_blocks =
            static_cast<int>((split_padded_elems + 255) / 256);
        pack_time_split_contig_padded_kernel<<<fill_split_padded_blocks, 1>>>(
            d_q_masked, d_k_masked, d_v_masked,
            d_q_split_padded, d_k_split_padded, d_v_split_padded,
            static_cast<long long>(split_padded_elems));
        int fill_rotary_blocks =
            static_cast<int>((rotary_elems + 255) / 256);
        fill_rotary_identity_kernel<<<fill_rotary_blocks, 1>>>(
            d_cos, d_sin, static_cast<long long>(rotary_elems));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float scale = 1.0f / std::sqrt(static_cast<float>(kD));
        if (opts.variant != "all") {
            bool gated_store_variant =
                opts.variant == "main1280_q64k32_exp2_split_contig_gated_store";
            if (opts.compare_baseline && !gated_store_variant) {
                launch_main1280_q64k32_exp2_once(
                    d_q_masked, d_k_masked, d_v_masked, d_ref, scale);
            }
            if (opts.variant == "main1280_q64k32_exp2_qkv_direct_rotary") {
                run_main1280_qkv_direct_rotary_variant<64, 32>(
                    opts, d_qkv_masked, d_cos, d_sin, d_out, scale);
            } else if (opts.variant == "main1280_q64k32_exp2_split_contig_input") {
                run_main1280_split_contig_input_variant<64, 32>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_kt_input") {
                run_main1280_split_contig_kt_input_variant<64, 32>(
                    opts, d_q_split_contig, d_k_t_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_kt_input_no_keytail") {
                run_main1280_split_contig_kt_input_variant<64, 32, false>(
                    opts, d_q_split_contig, d_k_t_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_tail_idx32") {
                run_main1280_split_contig_input_variant<64, 32, true, true>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_tail_col_broadcast") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, true>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_tail_col_broadcast_out_acc_bf16") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, true, true>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_alpha_one") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, false, false, true>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_prob_linear") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, false, false, false,
                                                        kProbLinear>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_prob_linear_noclamp") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, false, false, false,
                                                        kProbLinearNoClamp>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_prob_poly2") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, false, false, false,
                                                        kProbPoly2>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_prob_poly4") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, false, false, false,
                                                        kProbPoly4>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_prob_poly2_noclamp") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, false, false, false,
                                                        kProbPoly2NoClamp>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_prob_poly3_noclamp") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, false, false, false,
                                                        kProbPoly3NoClamp>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_prob_poly3_noclamp_alpha_poly3_clamp") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, false, false, false,
                                                        kProbPoly3NoClamp, false,
                                                        kAlphaProbClamp>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_prob_poly3_noclamp_alpha_poly3_clamp_final_rcp") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        true, false, false, false,
                                                        kProbPoly3NoClamp, false,
                                                        kAlphaProbClamp>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_prob_poly3_outclamp_alpha_outclamp") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, false, false, false,
                                                        kProbPoly3OutputClamp, false,
                                                        kAlphaProbClamp>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_prob_poly3_noclamp_alpha_linear_clamp") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, false, false, false,
                                                        kProbPoly3NoClamp, false,
                                                        kAlphaLinearClamp>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_prob_rational4_alpha_rational4") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, false, false, false,
                                                        kProbRational4, false,
                                                        kAlphaProbClamp>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_prob_poly4_noclamp") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, false, false, false,
                                                        kProbPoly4NoClamp>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_prob_poly4_noclamp_sum_bf16") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, false, false, false,
                                                        kProbPoly4NoClamp, true>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_prob_poly4_noclamp_bias") {
                run_main1280_split_contig_input_variant<64, 32, true, false,
                                                        false, false, false, false,
                                                        kProbPoly4NoClampBias>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_tail_prob_mask_only") {
                run_main1280_split_contig_tail_prob_mask_only_variant<64, 32>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_tail16_8") {
                run_main1280_split_contig_tail16_8_variant<64, 32>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_tail_helper") {
                run_main1280_split_contig_tail_helper_variant<64, 32>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_seg2") {
                run_main1280_split_contig_seg2_variant<64, 32>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_seg2_no_keytail") {
                run_main1280_split_contig_seg2_variant<64, 32, false>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_tail_first") {
                run_main1280_split_contig_tail_first_variant<64, 32>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_tail_first_padded_tail_load") {
                run_main1280_split_contig_tail_first_padded_tail_load_variant<64, 32>(
                    opts, d_q_split_padded, d_k_split_padded, d_v_split_padded,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_padded_tail_load") {
                run_main1280_split_contig_padded_tail_load_variant<64, 32>(
                    opts, d_q_split_padded, d_k_split_padded, d_v_split_padded,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_tail_col_broadcast_padded_tail_load") {
                run_main1280_split_contig_padded_tail_load_variant<64, 32, true>(
                    opts, d_q_split_padded, d_k_split_padded, d_v_split_padded,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_two_pass_state") {
                run_main1280_split_contig_two_pass_state_variant<64, 32>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_state_acc, d_state_m, d_state_l, d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_score_av_lb") {
                run_main1280_split_contig_score_av_lb_variant<64, 32>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_score_av_lb_no_keytail") {
                run_main1280_split_contig_score_av_lb_variant<64, 32, false>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_tile_local_softmax_lb") {
                run_main1280_split_contig_tile_local_softmax_lb_variant<64, 32>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_tile_local_softmax_lb_no_keytail") {
                run_main1280_split_contig_tile_local_softmax_lb_variant<64, 32, false>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_final_rcp") {
                run_main1280_split_contig_input_variant<64, 32, true, false, true>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_final_rcp_no_keytail") {
                run_main1280_split_contig_input_variant<64, 32, false, false, true>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_lat2") {
                run_main1280_split_contig_input_lat_variant<64, 32>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_lat2_no_keytail") {
                run_main1280_split_contig_input_lat_variant<64, 32, false>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_q_lat2") {
                run_main1280_split_contig_input_lat_variant<64, 32, true, 2, 0, 0>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_q_lat2_no_keytail") {
                run_main1280_split_contig_input_lat_variant<64, 32, false, 2, 0, 0>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_k_lat2") {
                run_main1280_split_contig_input_lat_variant<64, 32, true, 0, 2, 0>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_k_lat2_no_keytail") {
                run_main1280_split_contig_input_lat_variant<64, 32, false, 0, 2, 0>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_v_lat2") {
                run_main1280_split_contig_input_lat_variant<64, 32, true, 0, 0, 2>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_v_lat2_no_keytail") {
                run_main1280_split_contig_input_lat_variant<64, 32, false, 0, 0, 2>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_kv_lat2") {
                run_main1280_split_contig_input_lat_variant<64, 32, true, 0, 2, 2>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_kv_lat2_no_keytail") {
                run_main1280_split_contig_input_lat_variant<64, 32, false, 0, 2, 2>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_row_l_bf16") {
                run_main1280_split_contig_row_state_bf16_variant<64, 32>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_row_l_bf16_no_keytail") {
                run_main1280_split_contig_row_state_bf16_variant<64, 32, false>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_row_state_bf16") {
                run_main1280_split_contig_row_state_bf16_variant<64, 32, true, true>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_row_state_bf16_no_keytail") {
                run_main1280_split_contig_row_state_bf16_variant<64, 32, false, true>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_first_init") {
                run_main1280_split_contig_first_init_variant<64, 32>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_first_init_tail_col_broadcast_padded_tail_load") {
                run_main1280_split_contig_first_init_variant<64, 32, true, true>(
                    opts, d_q_split_padded, d_k_split_padded, d_v_split_padded,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_first_init_no_keytail") {
                run_main1280_split_contig_first_init_variant<64, 32, false>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_out_acc_bf16") {
                run_main1280_split_contig_input_variant<64, 32, true, false, false,
                                                        false, true>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_out_acc_bf16_no_keytail") {
                run_main1280_split_contig_input_variant<64, 32, false, false, false,
                                                        false, true>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_gated_store") {
                run_main1280_split_contig_gated_store_variant<64, 32>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_gates, d_out, d_merged_main, d_ref_merged_main, scale);
            } else if (opts.variant ==
                       "main1280_q64k32_exp2_split_contig_input_no_keytail") {
                run_main1280_split_contig_input_variant<64, 32, false>(
                    opts, d_q_split_contig, d_k_split_contig, d_v_split_contig,
                    d_out, scale);
            } else {
                run_focused_variant(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
            }
            if (opts.compare_baseline && !gated_store_variant) {
                compare_outputs_to_baseline(d_ref, d_out, out_elems);
            } else if (opts.compare_baseline && gated_store_variant) {
                compare_outputs_to_baseline(d_ref_merged_main, d_merged_main,
                                            merged_main_elems);
            }
        } else {
        run_padded_variant<8>(opts, d_q, d_k, d_v, d_out, scale, false);
        run_padded_variant<16>(opts, d_q, d_k, d_v, d_out, scale, opts.validate);
        run_padded_variant<32>(opts, d_q, d_k, d_v, d_out, scale, false);
        run_padded_variant<64>(opts, d_q, d_k, d_v, d_out, scale, false);
        run_padded_variant<128>(opts, d_q, d_k, d_v, d_out, scale, false);
        run_masked_variant<16, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale, false);
        run_masked_variant<16, 64>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale, false);
        run_masked_variant<32, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale, false);
        run_masked_variant<32, 64>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale, opts.validate);
        run_masked_variant<64, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale, opts.validate);
        run_masked_variant<64, 64>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale, false);
        run_masked_variant<128, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale, false);
        run_score_av_lower_bound_variant<64, 32>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_score_av_lower_bound_variant<64, 64>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_qk_only_variant<32>(opts, d_q_masked, d_k_masked, d_out, scale);
        run_main1280_qk_only_variant<64>(opts, d_q_masked, d_k_masked, d_out, scale);
        run_main1280_qk_only_kt_variant<64>(opts, d_q_masked, d_k_t_masked, d_out, scale);
        run_main1280_qk_only_variant<128>(opts, d_q_masked, d_k_masked, d_out, scale);
        run_main1280_av_const_variant<32>(opts, d_v_masked, d_out);
        run_main1280_av_const_variant<64>(opts, d_v_masked, d_out);
        run_main1280_av_const_variant<128>(opts, d_v_masked, d_out);
        run_main1280_variant<32, false>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<32, false, true>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<32, false, true, false, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<32, false, true, true, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<32, true>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, false, 16>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, true, 16>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, false, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, true, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, true, 32, true>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, true, 32, false, true>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_direct_ptr_variant<64, 32>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, false, 128>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_tail_only_variant<64, 64>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_tail_only_variant<32, 64>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_tail_only_variant<16, 64>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_tail_only_variant<32, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_tail_only_variant<64, 64, true>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_tail_only_variant<32, 32, true>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_split_tail_pair_variant<64>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_split_tail_pair_variant<32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_split_tail_pair_variant<32, 32, 64>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_split_tail_pair_variant<32, 32, 32>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_split_tail_pair_variant<32, 64, 64, true>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_split_tail_pair_variant<32, 32, 32, true>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, false, 64, true>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, false, 32, true>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, true>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_q64k32_exp2_split_d32_variant(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, true>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_prescale_q_lower_bound_variant(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_split_d32_lower_bound_variant(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_split_global_variant<64>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_p_split_q64, d_out, scale);
        run_main1280_variant<128, false>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<128, false, true>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<128, false, true, false, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<128, false, true, true, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<128, true>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        }
        CUDA_CHECK(cudaFree(d_q));
        CUDA_CHECK(cudaFree(d_k));
        CUDA_CHECK(cudaFree(d_v));
        CUDA_CHECK(cudaFree(d_q_masked));
        CUDA_CHECK(cudaFree(d_k_masked));
        CUDA_CHECK(cudaFree(d_k_t_masked));
        CUDA_CHECK(cudaFree(d_v_masked));
        CUDA_CHECK(cudaFree(d_qkv_masked));
        CUDA_CHECK(cudaFree(d_q_split_contig));
        CUDA_CHECK(cudaFree(d_k_split_contig));
        CUDA_CHECK(cudaFree(d_k_t_split_contig));
        CUDA_CHECK(cudaFree(d_v_split_contig));
        CUDA_CHECK(cudaFree(d_q_split_padded));
        CUDA_CHECK(cudaFree(d_k_split_padded));
        CUDA_CHECK(cudaFree(d_v_split_padded));
        CUDA_CHECK(cudaFree(d_gates));
        CUDA_CHECK(cudaFree(d_merged_main));
        if (d_ref_merged_main) {
            CUDA_CHECK(cudaFree(d_ref_merged_main));
        }
        CUDA_CHECK(cudaFree(d_cos));
        CUDA_CHECK(cudaFree(d_sin));
        CUDA_CHECK(cudaFree(d_state_acc));
        CUDA_CHECK(cudaFree(d_state_m));
        CUDA_CHECK(cudaFree(d_state_l));
        CUDA_CHECK(cudaFree(d_out));
        if (d_ref) {
            CUDA_CHECK(cudaFree(d_ref));
        }
        CUDA_CHECK(cudaFree(d_p_split_q64));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
