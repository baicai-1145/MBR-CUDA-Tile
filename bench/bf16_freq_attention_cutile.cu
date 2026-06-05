#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                             \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            throw std::runtime_error(std::string(#call) + " failed: " +             \
                                     cudaGetErrorString(err__));                    \
        }                                                                           \
    } while (0)

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kN = 60;
constexpr int kNPad = 64;
constexpr int kD = 64;
constexpr int kBH = 10408;
constexpr int kHeads = 8;
constexpr int kBatches = kBH / kHeads;
constexpr int kQkvFeatures = 3 * kHeads * kD;
constexpr int kInitTile = 256;
constexpr float kScale = 0.125f;
constexpr double kA10gDenseBf16Tflops = 70.0;
static_assert(kBH % kHeads == 0);

using I64InitTile = ct::tile<long long, ct::shape<kInitTile>>;
using F32InitTile = ct::tile<float, ct::shape<kInitTile>>;
using U32InitTile = ct::tile<unsigned int, ct::shape<kInitTile>>;

struct Options {
    int warmup = 1;
    int iters = 5;
    std::string variant = "all";
    bool validate = false;
    bool compare_baseline = false;
    bool describe = false;
};

int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

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
                "Usage: bench_bf16_freq_attention_cutile [options]\n"
                "  --variant NAME  all, q8, q16, q32, q64, q128, q8m, q16m, q32m,\n"
                "                  q64m, q128m, q8p, q16p, q32p, q64p, q128p,\n"
                "                  q8s, q16s, q16s_exp2, q16sb, q16sc,\n"
                "                  q16sc_exp2, q16s_poly3_noclamp,\n"
                "                  q16s_poly3_clamp, q16s_poly2_l4,\n"
                "                  q16s_nodenom, q16s_poly3_clamp_nodenom,\n"
                "                  q16s_bh2, q16s_bh4, q16s_v32, q16s_v16,\n"
                "                  q32s, q64s, q128s,\n"
                "                  q16qkv_pipe, q16qkv_pipe_u32,\n"
                "                  qkv_rot_split, qkv_rot_split_vu32,\n"
                "                  q16qkv_rot_pipe, q16qkv_rot_pipe_vu32,\n"
                "                  qkv_rot_compact_split,\n"
                "                  qkv_rot_compact_split_vu32,\n"
                "                  q16qkv_rot_compact_pipe,\n"
                "                  q16qkv_rot_compact_pipe_vu32,\n"
                "                  qkv_rot_validpad_split,\n"
                "                  qkv_rot_validpad_split_vu32,\n"
                "                  q16qkv_rot_validpad_pipe,\n"
                "                  q16qkv_rot_validpad_pipe_vu32,\n"
                "                  q16qkv_rot_direct, q32qkv_rot_direct,\n"
                "                  q64qkv_rot_direct,\n"
                "                  q16qkv_direct, q32qkv_direct, q64qkv_direct,\n"
                "                  default all\n"
                "  --warmup N      warmup launches, default 1\n"
                "  --iters N       measured launches, default 5\n"
                "  --compare-baseline  compare source-like output against qrows baseline\n"
                "  --validate      compare BH=0 output against CPU reference\n"
                "  --describe      print CUDA runtime resource/occupancy diagnostics\n");
            std::exit(0);
        } else {
            throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
        }
    }
    return opts;
}

float percentile(std::vector<float> values, float q) {
    std::sort(values.begin(), values.end());
    float pos = q * static_cast<float>(values.size() - 1);
    int lo = static_cast<int>(pos);
    int hi = std::min(lo + 1, static_cast<int>(values.size() - 1));
    float t = pos - static_cast<float>(lo);
    return values[lo] * (1.0f - t) + values[hi] * t;
}

enum FreqProbMode : int {
    kFreqProbExp = 0,
    kFreqProbPoly3NoClamp = 1,
    kFreqProbPoly3Clamp = 2,
    kFreqProbPoly2L4 = 3,
};

template <int Prob, bool UseExp2, typename TileT>
static __tile__ auto freq_softmax_prob(TileT shifted) {
    if constexpr (Prob == kFreqProbPoly3NoClamp) {
        auto t = shifted * 0.333333343f + 1.0f;
        auto t2 = t * t;
        return t2 * t;
    } else if constexpr (Prob == kFreqProbPoly3Clamp) {
        auto zero = shifted * 0.0f;
        auto t = shifted * 0.333333343f + 1.0f;
        t = ct::select(t > zero, t, zero);
        auto t2 = t * t;
        return t2 * t;
    } else if constexpr (Prob == kFreqProbPoly2L4) {
        auto zero = shifted * 0.0f;
        auto t = shifted * 0.25f + 1.0f;
        t = ct::select(t > zero, t, zero);
        return t * t;
    } else if constexpr (UseExp2) {
        return ct::exp2(shifted * 1.4426950408889634f);
    } else {
        return ct::exp(shifted);
    }
}

__tile_global__ void fill_bf16_kernel(__nv_bfloat16* __restrict__ dst, long long total) {
    dst = ct::assume_aligned(dst, 16_ic);
    I64InitTile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64InitTile>();
    auto in_bounds = idx < total;
    F32InitTile values = 0.125f +
        ct::element_cast<float>((idx * 17LL) & 1023LL) * 0.000244140625f;
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

__tile_global__ void fill_trig_bf16_kernel(__nv_bfloat16* __restrict__ cos_f,
                                           __nv_bfloat16* __restrict__ sin_f,
                                           long long total) {
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);

    I64InitTile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64InitTile>();
    auto in_bounds = idx < total;
    F32InitTile phase0 = ct::element_cast<float>((idx * 5LL) & 31LL) * 0.00390625f;
    F32InitTile phase1 = ct::element_cast<float>((idx * 7LL) & 31LL) * 0.00390625f;
    auto c = 0.8125f + phase0;
    auto s = 0.125f + phase1;
    ct::store_masked(cos_f + idx, ct::element_cast<__nv_bfloat16>(c), in_bounds);
    ct::store_masked(sin_f + idx, ct::element_cast<__nv_bfloat16>(s), in_bounds);
}

__tile_global__ void pad3_freq60_to64_kernel(const __nv_bfloat16* __restrict__ q,
                                             const __nv_bfloat16* __restrict__ k,
                                             const __nv_bfloat16* __restrict__ v,
                                             __nv_bfloat16* __restrict__ q_pad,
                                             __nv_bfloat16* __restrict__ k_pad,
                                             __nv_bfloat16* __restrict__ v_pad,
                                             long long total) {
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    q_pad = ct::assume_aligned(q_pad, 16_ic);
    k_pad = ct::assume_aligned(k_pad, 16_ic);
    v_pad = ct::assume_aligned(v_pad, 16_ic);

    I64InitTile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64InitTile>();
    auto in_bounds = idx < total;
    auto d = idx % kD;
    auto row = (idx / kD) % kNPad;
    auto bh = idx / (kNPad * kD);
    auto valid = in_bounds && (row < kN);
    auto safe_row = ct::select(valid, row, row * 0LL);
    auto src = bh * (long long)kN * kD + safe_row * kD + d;
    auto zero = ct::element_cast<__nv_bfloat16>(ct::element_cast<float>(idx * 0LL));
    auto qv = ct::load_masked(q + src, valid, zero);
    auto kv = ct::load_masked(k + src, valid, zero);
    auto vv = ct::load_masked(v + src, valid, zero);
    ct::store_masked(q_pad + idx, qv, in_bounds);
    ct::store_masked(k_pad + idx, kv, in_bounds);
    ct::store_masked(v_pad + idx, vv, in_bounds);
}

__tile_global__ void split_packed_qkv_identity_freq60_pad64_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    long long total) {
    qkv = ct::assume_aligned(qkv, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    constexpr int half_dim = kD / 2;
    I64InitTile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64InitTile>();
    auto in_bounds = idx < total;
    auto pair = idx % half_dim;
    auto n_pad = (idx / half_dim) % kNPad;
    auto h = (idx / ((long long)half_dim * kNPad)) % kHeads;
    auto b = idx / ((long long)half_dim * kNPad * kHeads);
    auto valid = in_bounds && (n_pad < kN);
    auto safe_n = ct::select(valid, n_pad, n_pad * 0LL);
    auto pair_d = pair * 2;

    auto qkv_base = (b * kN + safe_n) * (long long)kQkvFeatures + h * kD + pair_d;
    auto out_base = ((b * kHeads + h) * kNPad + n_pad) * kD + pair_d;
    auto zero = ct::element_cast<__nv_bfloat16>(ct::element_cast<float>(idx * 0LL));

    auto q0 = ct::load_masked(qkv + qkv_base, valid, zero);
    auto q1 = ct::load_masked(qkv + qkv_base + 1, valid, zero);
    auto k0 = ct::load_masked(qkv + qkv_base + kHeads * kD, valid, zero);
    auto k1 = ct::load_masked(qkv + qkv_base + kHeads * kD + 1, valid, zero);
    auto v0 = ct::load_masked(qkv + qkv_base + 2LL * kHeads * kD, valid, zero);
    auto v1 = ct::load_masked(qkv + qkv_base + 2LL * kHeads * kD + 1, valid, zero);

    ct::store_masked(q + out_base, q0, in_bounds);
    ct::store_masked(q + out_base + 1, q1, in_bounds);
    ct::store_masked(k + out_base, k0, in_bounds);
    ct::store_masked(k + out_base + 1, k1, in_bounds);
    ct::store_masked(v + out_base, v0, in_bounds);
    ct::store_masked(v + out_base + 1, v1, in_bounds);
}

__tile_global__ void split_packed_qkv_identity_freq60_pad64_u32_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    long long total) {
    auto qkv_u32 = ct::assume_aligned(
        reinterpret_cast<const unsigned int*>(qkv), 16_ic);
    auto q_u32 = ct::assume_aligned(reinterpret_cast<unsigned int*>(q), 16_ic);
    auto k_u32 = ct::assume_aligned(reinterpret_cast<unsigned int*>(k), 16_ic);
    auto v_u32 = ct::assume_aligned(reinterpret_cast<unsigned int*>(v), 16_ic);

    constexpr int half_dim = kD / 2;
    I64InitTile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64InitTile>();
    auto in_bounds = idx < total;
    auto pair = idx % half_dim;
    auto n_pad = (idx / half_dim) % kNPad;
    auto h = (idx / ((long long)half_dim * kNPad)) % kHeads;
    auto b = idx / ((long long)half_dim * kNPad * kHeads);
    auto valid = in_bounds && (n_pad < kN);
    auto safe_n = ct::select(valid, n_pad, n_pad * 0LL);

    auto qkv_pair_base = ((b * kN + safe_n) * (long long)kQkvFeatures +
                          h * kD) / 2 + pair;
    auto out_pair_base = (((b * kHeads + h) * kNPad + n_pad) * kD) / 2 + pair;
    U32InitTile zero = ct::element_cast<unsigned int>(idx * 0LL);

    auto q_pair = ct::load_masked(qkv_u32 + qkv_pair_base, valid, zero);
    auto k_pair = ct::load_masked(qkv_u32 + qkv_pair_base + (kHeads * kD) / 2,
                                  valid, zero);
    auto v_pair = ct::load_masked(qkv_u32 + qkv_pair_base + kHeads * kD,
                                  valid, zero);

    ct::store_masked(q_u32 + out_pair_base, q_pair, in_bounds);
    ct::store_masked(k_u32 + out_pair_base, k_pair, in_bounds);
    ct::store_masked(v_u32 + out_pair_base, v_pair, in_bounds);
}

template <bool VU32 = false>
__tile_global__ void split_packed_qkv_rotary_freq60_pad64_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ cos_f,
    const __nv_bfloat16* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    long long total) {
    qkv = ct::assume_aligned(qkv, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    const unsigned int* qkv_u32 = nullptr;
    unsigned int* v_u32 = nullptr;
    if constexpr (VU32) {
        qkv_u32 = ct::assume_aligned(
            reinterpret_cast<const unsigned int*>(qkv), 16_ic);
        v_u32 = ct::assume_aligned(reinterpret_cast<unsigned int*>(v), 16_ic);
    }

    constexpr int half_dim = kD / 2;
    I64InitTile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64InitTile>();
    auto in_bounds = idx < total;
    auto pair = idx % half_dim;
    auto n_pad = (idx / half_dim) % kNPad;
    auto h = (idx / ((long long)half_dim * kNPad)) % kHeads;
    auto b = idx / ((long long)half_dim * kNPad * kHeads);
    auto valid = in_bounds && (n_pad < kN);
    auto safe_n = ct::select(valid, n_pad, n_pad * 0LL);
    auto pair_d = pair * 2;

    auto qkv_base = (b * kN + safe_n) * (long long)kQkvFeatures + h * kD + pair_d;
    auto out_base = ((b * kHeads + h) * kNPad + n_pad) * kD + pair_d;
    auto zero_f = ct::element_cast<float>(idx * 0LL);
    auto zero_b = ct::element_cast<__nv_bfloat16>(zero_f);

    auto c = ct::element_cast<float>(ct::load_masked(
        cos_f + safe_n * half_dim + pair, valid));
    auto s = ct::element_cast<float>(ct::load_masked(
        sin_f + safe_n * half_dim + pair, valid));

    auto q0 = ct::element_cast<float>(ct::load_masked(qkv + qkv_base, valid));
    auto q1 = ct::element_cast<float>(ct::load_masked(qkv + qkv_base + 1, valid));
    auto k0 = ct::element_cast<float>(
        ct::load_masked(qkv + qkv_base + kHeads * kD, valid));
    auto k1 = ct::element_cast<float>(
        ct::load_masked(qkv + qkv_base + kHeads * kD + 1, valid));

    auto q_rot0 = q0 * c - q1 * s;
    auto q_rot1 = q0 * s + q1 * c;
    auto k_rot0 = k0 * c - k1 * s;
    auto k_rot1 = k0 * s + k1 * c;

    ct::store_masked(q + out_base,
                     ct::element_cast<__nv_bfloat16>(ct::select(valid, q_rot0, zero_f)),
                     in_bounds);
    ct::store_masked(q + out_base + 1,
                     ct::element_cast<__nv_bfloat16>(ct::select(valid, q_rot1, zero_f)),
                     in_bounds);
    ct::store_masked(k + out_base,
                     ct::element_cast<__nv_bfloat16>(ct::select(valid, k_rot0, zero_f)),
                     in_bounds);
    ct::store_masked(k + out_base + 1,
                     ct::element_cast<__nv_bfloat16>(ct::select(valid, k_rot1, zero_f)),
                     in_bounds);

    if constexpr (VU32) {
        auto qkv_pair_base = (qkv_base + 2LL * kHeads * kD) / 2;
        auto out_pair_base = out_base / 2;
        U32InitTile zero_u32 = ct::element_cast<unsigned int>(idx * 0LL);
        auto v_pair = ct::load_masked(qkv_u32 + qkv_pair_base, valid, zero_u32);
        ct::store_masked(v_u32 + out_pair_base, v_pair, in_bounds);
    } else {
        auto v0 = ct::load_masked(qkv + qkv_base + 2LL * kHeads * kD, valid, zero_b);
        auto v1 = ct::load_masked(qkv + qkv_base + 2LL * kHeads * kD + 1, valid, zero_b);
        ct::store_masked(v + out_base, v0, in_bounds);
        ct::store_masked(v + out_base + 1, v1, in_bounds);
    }
}

template <bool VU32 = false>
__tile_global__ void split_packed_qkv_rotary_freq60_compact_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ cos_f,
    const __nv_bfloat16* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    long long total) {
    qkv = ct::assume_aligned(qkv, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    const unsigned int* qkv_u32 = nullptr;
    unsigned int* v_u32 = nullptr;
    if constexpr (VU32) {
        qkv_u32 = ct::assume_aligned(
            reinterpret_cast<const unsigned int*>(qkv), 16_ic);
        v_u32 = ct::assume_aligned(reinterpret_cast<unsigned int*>(v), 16_ic);
    }

    constexpr int half_dim = kD / 2;
    I64InitTile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64InitTile>();
    auto in_bounds = idx < total;
    auto pair = idx % half_dim;
    auto n = (idx / half_dim) % kN;
    auto h = (idx / ((long long)half_dim * kN)) % kHeads;
    auto b = idx / ((long long)half_dim * kN * kHeads);
    auto pair_d = pair * 2;

    auto qkv_base = (b * kN + n) * (long long)kQkvFeatures + h * kD + pair_d;
    auto out_base = ((b * kHeads + h) * kN + n) * kD + pair_d;
    auto zero_f = ct::element_cast<float>(idx * 0LL);
    auto zero_b = ct::element_cast<__nv_bfloat16>(zero_f);

    auto c = ct::element_cast<float>(ct::load_masked(
        cos_f + n * half_dim + pair, in_bounds));
    auto s = ct::element_cast<float>(ct::load_masked(
        sin_f + n * half_dim + pair, in_bounds));

    auto q0 = ct::element_cast<float>(ct::load_masked(qkv + qkv_base, in_bounds));
    auto q1 = ct::element_cast<float>(ct::load_masked(qkv + qkv_base + 1, in_bounds));
    auto k0 = ct::element_cast<float>(
        ct::load_masked(qkv + qkv_base + kHeads * kD, in_bounds));
    auto k1 = ct::element_cast<float>(
        ct::load_masked(qkv + qkv_base + kHeads * kD + 1, in_bounds));

    auto q_rot0 = q0 * c - q1 * s;
    auto q_rot1 = q0 * s + q1 * c;
    auto k_rot0 = k0 * c - k1 * s;
    auto k_rot1 = k0 * s + k1 * c;

    ct::store_masked(q + out_base,
                     ct::element_cast<__nv_bfloat16>(ct::select(in_bounds, q_rot0, zero_f)),
                     in_bounds);
    ct::store_masked(q + out_base + 1,
                     ct::element_cast<__nv_bfloat16>(ct::select(in_bounds, q_rot1, zero_f)),
                     in_bounds);
    ct::store_masked(k + out_base,
                     ct::element_cast<__nv_bfloat16>(ct::select(in_bounds, k_rot0, zero_f)),
                     in_bounds);
    ct::store_masked(k + out_base + 1,
                     ct::element_cast<__nv_bfloat16>(ct::select(in_bounds, k_rot1, zero_f)),
                     in_bounds);

    if constexpr (VU32) {
        auto qkv_pair_base = (qkv_base + 2LL * kHeads * kD) / 2;
        auto out_pair_base = out_base / 2;
        U32InitTile zero_u32 = ct::element_cast<unsigned int>(idx * 0LL);
        auto v_pair = ct::load_masked(qkv_u32 + qkv_pair_base, in_bounds, zero_u32);
        ct::store_masked(v_u32 + out_pair_base, v_pair, in_bounds);
    } else {
        auto v0 = ct::load_masked(qkv + qkv_base + 2LL * kHeads * kD,
                                  in_bounds, zero_b);
        auto v1 = ct::load_masked(qkv + qkv_base + 2LL * kHeads * kD + 1,
                                  in_bounds, zero_b);
        ct::store_masked(v + out_base, v0, in_bounds);
        ct::store_masked(v + out_base + 1, v1, in_bounds);
    }
}

template <bool VU32 = false>
__tile_global__ void split_packed_qkv_rotary_freq60_validpad_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ cos_f,
    const __nv_bfloat16* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    long long total) {
    qkv = ct::assume_aligned(qkv, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    const unsigned int* qkv_u32 = nullptr;
    unsigned int* v_u32 = nullptr;
    if constexpr (VU32) {
        qkv_u32 = ct::assume_aligned(
            reinterpret_cast<const unsigned int*>(qkv), 16_ic);
        v_u32 = ct::assume_aligned(reinterpret_cast<unsigned int*>(v), 16_ic);
    }

    constexpr int half_dim = kD / 2;
    I64InitTile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64InitTile>();
    auto in_bounds = idx < total;
    auto pair = idx % half_dim;
    auto n = (idx / half_dim) % kN;
    auto h = (idx / ((long long)half_dim * kN)) % kHeads;
    auto b = idx / ((long long)half_dim * kN * kHeads);
    auto pair_d = pair * 2;

    auto qkv_base = (b * kN + n) * (long long)kQkvFeatures + h * kD + pair_d;
    auto out_base = ((b * kHeads + h) * kNPad + n) * kD + pair_d;
    auto zero_f = ct::element_cast<float>(idx * 0LL);
    auto zero_b = ct::element_cast<__nv_bfloat16>(zero_f);

    auto c = ct::element_cast<float>(ct::load_masked(
        cos_f + n * half_dim + pair, in_bounds));
    auto s = ct::element_cast<float>(ct::load_masked(
        sin_f + n * half_dim + pair, in_bounds));

    auto q0 = ct::element_cast<float>(ct::load_masked(qkv + qkv_base, in_bounds));
    auto q1 = ct::element_cast<float>(ct::load_masked(qkv + qkv_base + 1, in_bounds));
    auto k0 = ct::element_cast<float>(
        ct::load_masked(qkv + qkv_base + kHeads * kD, in_bounds));
    auto k1 = ct::element_cast<float>(
        ct::load_masked(qkv + qkv_base + kHeads * kD + 1, in_bounds));

    auto q_rot0 = q0 * c - q1 * s;
    auto q_rot1 = q0 * s + q1 * c;
    auto k_rot0 = k0 * c - k1 * s;
    auto k_rot1 = k0 * s + k1 * c;

    ct::store_masked(q + out_base,
                     ct::element_cast<__nv_bfloat16>(ct::select(in_bounds, q_rot0, zero_f)),
                     in_bounds);
    ct::store_masked(q + out_base + 1,
                     ct::element_cast<__nv_bfloat16>(ct::select(in_bounds, q_rot1, zero_f)),
                     in_bounds);
    ct::store_masked(k + out_base,
                     ct::element_cast<__nv_bfloat16>(ct::select(in_bounds, k_rot0, zero_f)),
                     in_bounds);
    ct::store_masked(k + out_base + 1,
                     ct::element_cast<__nv_bfloat16>(ct::select(in_bounds, k_rot1, zero_f)),
                     in_bounds);

    if constexpr (VU32) {
        auto qkv_pair_base = (qkv_base + 2LL * kHeads * kD) / 2;
        auto out_pair_base = out_base / 2;
        U32InitTile zero_u32 = ct::element_cast<unsigned int>(idx * 0LL);
        auto v_pair = ct::load_masked(qkv_u32 + qkv_pair_base, in_bounds, zero_u32);
        ct::store_masked(v_u32 + out_pair_base, v_pair, in_bounds);
    } else {
        auto v0 = ct::load_masked(qkv + qkv_base + 2LL * kHeads * kD,
                                  in_bounds, zero_b);
        auto v1 = ct::load_masked(qkv + qkv_base + 2LL * kHeads * kD + 1,
                                  in_bounds, zero_b);
        ct::store_masked(v + out_base, v0, in_bounds);
        ct::store_masked(v + out_base + 1, v1, in_bounds);
    }
}

template <int QRows>
__tile_global__ void freq_attention60_cutile_kernel(const __nv_bfloat16* __restrict__ q,
                                                    const __nv_bfloat16* __restrict__ k,
                                                    const __nv_bfloat16* __restrict__ v,
                                                    __nv_bfloat16* __restrict__ out) {
    using ScoreTile = ct::tile<float, ct::shape<QRows, kNPad>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, kNPad>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch = q + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* k_batch = k + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* v_batch = v + static_cast<std::size_t>(bh) * kNPad * kD;
    __nv_bfloat16* out_batch = out + static_cast<std::size_t>(bh) * kNPad * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kNPad, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kNPad>{}, ct::layout_left{}},
        ct::shape<kD, kNPad>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kNPad, kD>{}},
        ct::shape<kNPad, kD>{}
    };

    auto scores = ct::mma(q_view.load(q_block, 0), k_t_view.load(0, 0),
                          ct::full<ScoreTile>(0.0f));
    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / kNPad;
    auto score_cols = score_local % kNPad;
    auto score_valid = (score_rows < kN) && (score_cols < kN);
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(score_valid, scores * kScale, neg_inf);

    auto row_max = ct::reduce_max<1>(scores);
    auto probs_f32 = ct::select(score_valid, ct::exp(scores - row_max), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto denom = ct::sum<1>(ct::element_cast<float>(probs_bf16));

    auto out_acc = ct::mma(probs_bf16, v_view.load(0, 0), ct::full<OutTile>(0.0f));
    out_acc = out_acc / denom;

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_valid = out_rows < kN;
    ct::store_masked(out_batch + out_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows, bool UseExp2 = false>
__tile_global__ void freq_attention60_packed_qkv_identity_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ out) {
    using QTile = ct::tile<__nv_bfloat16, ct::shape<QRows, kD>>;
    using KTile = ct::tile<__nv_bfloat16, ct::shape<kD, kNPad>>;
    using VTile = ct::tile<__nv_bfloat16, ct::shape<kNPad, kD>>;
    using ScoreTile = ct::tile<float, ct::shape<QRows, kNPad>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64QTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using I64KTile = ct::tile<long long, ct::shape<kD, kNPad>>;
    using I64VTile = ct::tile<long long, ct::shape<kNPad, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, kNPad>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;

    qkv = ct::assume_aligned(qkv, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    int bh_i = static_cast<int>(bh);
    int b = bh_i / kHeads;
    int h = bh_i - b * kHeads;
    __nv_bfloat16* out_batch = out + static_cast<std::size_t>(bh_i) * kN * kD;

    I64QTile q_local = ct::iota<I64QTile>();
    auto q_rows = static_cast<long long>(q_block) * QRows + q_local / kD;
    auto q_cols = q_local % kD;
    auto q_valid = q_rows < kN;
    auto safe_q_rows = ct::select(q_valid, q_rows, q_rows * 0LL);
    auto q_offsets = ((long long)b * kN + safe_q_rows) * kQkvFeatures +
                     h * kD + q_cols;
    QTile q_tile = ct::load_masked(qkv + q_offsets, q_valid);

    I64KTile k_local = ct::iota<I64KTile>();
    auto k_dim = k_local / kNPad;
    auto k_col = k_local % kNPad;
    auto k_valid = k_col < kN;
    auto safe_k_col = ct::select(k_valid, k_col, k_col * 0LL);
    auto k_offsets = ((long long)b * kN + safe_k_col) * kQkvFeatures +
                     (long long)kHeads * kD + h * kD + k_dim;
    KTile k_tile = ct::load_masked(qkv + k_offsets, k_valid);

    I64VTile v_local = ct::iota<I64VTile>();
    auto v_row = v_local / kD;
    auto v_col = v_local % kD;
    auto v_valid = v_row < kN;
    auto safe_v_row = ct::select(v_valid, v_row, v_row * 0LL);
    auto v_offsets = ((long long)b * kN + safe_v_row) * kQkvFeatures +
                     2LL * kHeads * kD + h * kD + v_col;
    VTile v_tile = ct::load_masked(qkv + v_offsets, v_valid);

    auto scores = ct::mma(q_tile, k_tile, ct::full<ScoreTile>(0.0f));
    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / kNPad;
    auto score_cols = score_local % kNPad;
    auto score_valid = (score_rows < kN) && (score_cols < kN);
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(score_valid, scores * kScale, neg_inf);

    auto row_max = ct::reduce_max<1>(scores);
    auto shifted = scores - row_max;
    auto probs_f32 = [&]() {
        if constexpr (UseExp2) {
            return ct::select(score_valid, ct::exp2(shifted * 1.4426950408889634f),
                              scores * 0.0f);
        } else {
            return ct::select(score_valid, ct::exp(shifted), scores * 0.0f);
        }
    }();
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto denom = ct::sum<1>(probs_f32);

    auto out_acc = ct::mma(probs_bf16, v_tile, ct::full<OutTile>(0.0f));
    out_acc = out_acc / denom;

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_valid = out_rows < kN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows, bool UseExp2 = false>
__tile_global__ void freq_attention60_packed_qkv_rotary_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ cos_f,
    const __nv_bfloat16* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ out) {
    using QTile = ct::tile<__nv_bfloat16, ct::shape<QRows, kD>>;
    using KTile = ct::tile<__nv_bfloat16, ct::shape<kD, kNPad>>;
    using VTile = ct::tile<__nv_bfloat16, ct::shape<kNPad, kD>>;
    using ScoreTile = ct::tile<float, ct::shape<QRows, kNPad>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64QTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using I64KTile = ct::tile<long long, ct::shape<kD, kNPad>>;
    using I64VTile = ct::tile<long long, ct::shape<kNPad, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, kNPad>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;

    qkv = ct::assume_aligned(qkv, 16_ic);
    cos_f = ct::assume_aligned(cos_f, 16_ic);
    sin_f = ct::assume_aligned(sin_f, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    int bh_i = static_cast<int>(bh);
    int b = bh_i / kHeads;
    int h = bh_i - b * kHeads;
    __nv_bfloat16* out_batch = out + static_cast<std::size_t>(bh_i) * kN * kD;

    I64QTile q_local = ct::iota<I64QTile>();
    auto q_rows = static_cast<long long>(q_block) * QRows + q_local / kD;
    auto q_cols = q_local % kD;
    auto q_pair = q_cols / 2;
    auto q_even_col = q_pair * 2;
    auto q_odd_col = q_even_col + 1;
    auto q_is_even = (q_cols & 1LL) == 0LL;
    auto q_valid = q_rows < kN;
    auto safe_q_rows = ct::select(q_valid, q_rows, q_rows * 0LL);
    auto q_base = ((long long)b * kN + safe_q_rows) * kQkvFeatures + h * kD;
    auto q_c = ct::element_cast<float>(ct::load_masked(
        cos_f + safe_q_rows * (kD / 2) + q_pair, q_valid));
    auto q_s = ct::element_cast<float>(ct::load_masked(
        sin_f + safe_q_rows * (kD / 2) + q_pair, q_valid));
    auto q0 = ct::element_cast<float>(
        ct::load_masked(qkv + q_base + q_even_col, q_valid));
    auto q1 = ct::element_cast<float>(
        ct::load_masked(qkv + q_base + q_odd_col, q_valid));
    auto q_rot_even = q0 * q_c - q1 * q_s;
    auto q_rot_odd = q0 * q_s + q1 * q_c;
    QTile q_tile = ct::element_cast<__nv_bfloat16>(
        ct::select(q_is_even, q_rot_even, q_rot_odd));

    I64KTile k_local = ct::iota<I64KTile>();
    auto k_dim = k_local / kNPad;
    auto k_col = k_local % kNPad;
    auto k_pair = k_dim / 2;
    auto k_even_dim = k_pair * 2;
    auto k_odd_dim = k_even_dim + 1;
    auto k_is_even = (k_dim & 1LL) == 0LL;
    auto k_valid = k_col < kN;
    auto safe_k_col = ct::select(k_valid, k_col, k_col * 0LL);
    auto k_base = ((long long)b * kN + safe_k_col) * kQkvFeatures +
                  (long long)kHeads * kD + h * kD;
    auto k_c = ct::element_cast<float>(ct::load_masked(
        cos_f + safe_k_col * (kD / 2) + k_pair, k_valid));
    auto k_s = ct::element_cast<float>(ct::load_masked(
        sin_f + safe_k_col * (kD / 2) + k_pair, k_valid));
    auto k0 = ct::element_cast<float>(
        ct::load_masked(qkv + k_base + k_even_dim, k_valid));
    auto k1 = ct::element_cast<float>(
        ct::load_masked(qkv + k_base + k_odd_dim, k_valid));
    auto k_rot_even = k0 * k_c - k1 * k_s;
    auto k_rot_odd = k0 * k_s + k1 * k_c;
    KTile k_tile = ct::element_cast<__nv_bfloat16>(
        ct::select(k_is_even, k_rot_even, k_rot_odd));

    I64VTile v_local = ct::iota<I64VTile>();
    auto v_row = v_local / kD;
    auto v_col = v_local % kD;
    auto v_valid = v_row < kN;
    auto safe_v_row = ct::select(v_valid, v_row, v_row * 0LL);
    auto v_offsets = ((long long)b * kN + safe_v_row) * kQkvFeatures +
                     2LL * kHeads * kD + h * kD + v_col;
    VTile v_tile = ct::load_masked(qkv + v_offsets, v_valid);

    auto scores = ct::mma(q_tile, k_tile, ct::full<ScoreTile>(0.0f));
    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / kNPad;
    auto score_cols = score_local % kNPad;
    auto score_valid = (score_rows < kN) && (score_cols < kN);
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(score_valid, scores * kScale, neg_inf);

    auto row_max = ct::reduce_max<1>(scores);
    auto shifted = scores - row_max;
    auto probs_f32 = [&]() {
        if constexpr (UseExp2) {
            return ct::select(score_valid, ct::exp2(shifted * 1.4426950408889634f),
                              scores * 0.0f);
        } else {
            return ct::select(score_valid, ct::exp(shifted), scores * 0.0f);
        }
    }();
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto denom = ct::sum<1>(probs_f32);

    auto out_acc = ct::mma(probs_bf16, v_tile, ct::full<OutTile>(0.0f));
    out_acc = out_acc / denom;

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_valid = out_rows < kN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows, bool SumBF16Denom = true, bool ConstNegInf = false,
          bool UseExp2 = false, int Prob = kFreqProbExp, bool Normalize = true>
static __tile__ void freq_attention60_cutile_padded_out60_one_bh(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    long long q_block,
    long long bh) {
    using ScoreTile = ct::tile<float, ct::shape<QRows, kNPad>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, kNPad>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;

    const __nv_bfloat16* q_batch = q + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* k_batch = k + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* v_batch = v + static_cast<std::size_t>(bh) * kNPad * kD;
    __nv_bfloat16* out_batch = out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kNPad, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kNPad>{}, ct::layout_left{}},
        ct::shape<kD, kNPad>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kNPad, kD>{}},
        ct::shape<kNPad, kD>{}
    };

    auto scores = ct::mma(q_view.load(q_block, 0), k_t_view.load(0, 0),
                          ct::full<ScoreTile>(0.0f));
    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = q_block * QRows + score_local / kNPad;
    auto score_cols = score_local % kNPad;
    auto score_valid = (score_rows < kN) && (score_cols < kN);
    auto neg_inf = [&]() {
        if constexpr (ConstNegInf) {
            return ct::full<ScoreTile>(-3.402823466e38f);
        } else {
            return scores * 0.0f - 3.402823466e38f;
        }
    }();
    scores = ct::select(score_valid, scores * kScale, neg_inf);

    auto row_max = ct::reduce_max<1>(scores);
    auto shifted = scores - row_max;
    auto probs_f32 = [&]() {
        return ct::select(score_valid,
                          freq_softmax_prob<Prob, UseExp2>(shifted),
                          scores * 0.0f);
    }();
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);

    auto out_acc = ct::mma(probs_bf16, v_view.load(0, 0), ct::full<OutTile>(0.0f));
    if constexpr (Normalize) {
        auto denom = [&]() {
            if constexpr (SumBF16Denom) {
                return ct::sum<1>(ct::element_cast<float>(probs_bf16));
            } else {
                return ct::sum<1>(probs_f32);
            }
        }();
        out_acc = out_acc / denom;
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = q_block * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_valid = out_rows < kN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows, int BHPack, bool SumBF16Denom = true,
          bool ConstNegInf = false, bool UseExp2 = false,
          int Prob = kFreqProbExp, bool Normalize = true>
__tile_global__ void freq_attention60_cutile_padded_out60_bhpack_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out) {
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh_pack, tile_z] = ct::bid();
    (void)tile_z;
    long long first_bh = static_cast<long long>(bh_pack) * BHPack;

#pragma unroll
    for (int i = 0; i < BHPack; ++i) {
        long long bh = first_bh + i;
        if (bh < kBH) {
            freq_attention60_cutile_padded_out60_one_bh<
                QRows, SumBF16Denom, ConstNegInf, UseExp2, Prob, Normalize>(
                    q, k, v, out, static_cast<long long>(q_block), bh);
        }
    }
}

template <int QRows, bool SumBF16Denom = true, bool ConstNegInf = false,
          bool UseExp2 = false, int Prob = kFreqProbExp, bool Normalize = true>
__tile_global__ void freq_attention60_cutile_padded_out60_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out) {
    using ScoreTile = ct::tile<float, ct::shape<QRows, kNPad>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, kNPad>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch = q + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* k_batch = k + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* v_batch = v + static_cast<std::size_t>(bh) * kNPad * kD;
    __nv_bfloat16* out_batch = out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kNPad, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kNPad>{}, ct::layout_left{}},
        ct::shape<kD, kNPad>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kNPad, kD>{}},
        ct::shape<kNPad, kD>{}
    };

    auto scores = ct::mma(q_view.load(q_block, 0), k_t_view.load(0, 0),
                          ct::full<ScoreTile>(0.0f));
    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / kNPad;
    auto score_cols = score_local % kNPad;
    auto score_valid = (score_rows < kN) && (score_cols < kN);
    auto neg_inf = [&]() {
        if constexpr (ConstNegInf) {
            return ct::full<ScoreTile>(-3.402823466e38f);
        } else {
            return scores * 0.0f - 3.402823466e38f;
        }
    }();
    scores = ct::select(score_valid, scores * kScale, neg_inf);

    auto row_max = ct::reduce_max<1>(scores);
    auto shifted = scores - row_max;
    auto probs_f32 = [&]() {
        return ct::select(score_valid,
                          freq_softmax_prob<Prob, UseExp2>(shifted),
                          scores * 0.0f);
    }();
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto out_acc = ct::mma(probs_bf16, v_view.load(0, 0), ct::full<OutTile>(0.0f));
    if constexpr (Normalize) {
        auto denom = [&]() {
            if constexpr (SumBF16Denom) {
                return ct::sum<1>(ct::element_cast<float>(probs_bf16));
            } else {
                return ct::sum<1>(probs_f32);
            }
        }();
        out_acc = out_acc / denom;
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

template <int QRows, int VCols, bool SumBF16Denom = true,
          bool ConstNegInf = false, bool UseExp2 = false,
          int Prob = kFreqProbExp, bool Normalize = true>
__tile_global__ void freq_attention60_cutile_padded_out60_vsplit_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out) {
    constexpr int kVCols = VCols;
    static_assert(kD % kVCols == 0);
    using ScoreTile = ct::tile<float, ct::shape<QRows, kNPad>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kVCols>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, kNPad>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kVCols>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch = q + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* k_batch = k + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* v_batch = v + static_cast<std::size_t>(bh) * kNPad * kD;
    __nv_bfloat16* out_batch = out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kNPad, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kNPad>{}, ct::layout_left{}},
        ct::shape<kD, kNPad>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kNPad, kD>{}},
        ct::shape<kNPad, kVCols>{}
    };

    auto scores = ct::mma(q_view.load(q_block, 0), k_t_view.load(0, 0),
                          ct::full<ScoreTile>(0.0f));
    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / kNPad;
    auto score_cols = score_local % kNPad;
    auto score_valid = (score_rows < kN) && (score_cols < kN);
    auto neg_inf = [&]() {
        if constexpr (ConstNegInf) {
            return ct::full<ScoreTile>(-3.402823466e38f);
        } else {
            return scores * 0.0f - 3.402823466e38f;
        }
    }();
    scores = ct::select(score_valid, scores * kScale, neg_inf);

    auto row_max = ct::reduce_max<1>(scores);
    auto shifted = scores - row_max;
    auto probs_f32 = [&]() {
        return ct::select(score_valid,
                          freq_softmax_prob<Prob, UseExp2>(shifted),
                          scores * 0.0f);
    }();
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto denom = [&]() {
        if constexpr (SumBF16Denom) {
            return ct::sum<1>(ct::element_cast<float>(probs_bf16));
        } else {
            return ct::sum<1>(probs_f32);
        }
    }();

#pragma unroll
    for (int d_part = 0; d_part < kD / kVCols; ++d_part) {
        auto out_acc = ct::mma(probs_bf16, v_view.load(0, d_part),
                               ct::full<OutTile>(0.0f));
        if constexpr (Normalize) {
            out_acc = out_acc / denom;
        }

        I64OutTile out_local = ct::iota<I64OutTile>();
        auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kVCols;
        auto out_cols = d_part * kVCols + out_local % kVCols;
        auto out_valid = out_rows < kN;
        auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
        ct::store_masked(out_batch + safe_rows * kD + out_cols,
                         ct::element_cast<__nv_bfloat16>(out_acc),
                         out_valid);
    }
}

template <int QRows>
__tile_global__ void freq_attention60_cutile_masked_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out) {
    using QTile = ct::tile<__nv_bfloat16, ct::shape<QRows, kD>>;
    using KTile = ct::tile<__nv_bfloat16, ct::shape<kD, kNPad>>;
    using VTile = ct::tile<__nv_bfloat16, ct::shape<kNPad, kD>>;
    using ScoreTile = ct::tile<float, ct::shape<QRows, kNPad>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64QTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using I64KTile = ct::tile<long long, ct::shape<kD, kNPad>>;
    using I64VTile = ct::tile<long long, ct::shape<kNPad, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, kNPad>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch = q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch = k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch = v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch = out + static_cast<std::size_t>(bh) * kN * kD;

    I64QTile q_local = ct::iota<I64QTile>();
    auto q_rows = static_cast<long long>(q_block) * QRows + q_local / kD;
    auto q_cols = q_local % kD;
    auto q_valid = q_rows < kN;
    auto q_safe_rows = ct::select(q_valid, q_rows, q_rows * 0LL);
    QTile q_tile = ct::load_masked(q_batch + q_safe_rows * kD + q_cols, q_valid);

    I64KTile k_local = ct::iota<I64KTile>();
    auto k_dim = k_local / kNPad;
    auto k_col = k_local % kNPad;
    auto k_valid = k_col < kN;
    auto k_safe_col = ct::select(k_valid, k_col, k_col * 0LL);
    KTile k_tile = ct::load_masked(k_batch + k_safe_col * kD + k_dim, k_valid);

    I64VTile v_local = ct::iota<I64VTile>();
    auto v_row = v_local / kD;
    auto v_col = v_local % kD;
    auto v_valid = v_row < kN;
    auto v_safe_row = ct::select(v_valid, v_row, v_row * 0LL);
    VTile v_tile = ct::load_masked(v_batch + v_safe_row * kD + v_col, v_valid);

    auto scores = ct::mma(q_tile, k_tile, ct::full<ScoreTile>(0.0f));
    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / kNPad;
    auto score_cols = score_local % kNPad;
    auto score_valid = (score_rows < kN) && (score_cols < kN);
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(score_valid, scores * kScale, neg_inf);

    auto row_max = ct::reduce_max<1>(scores);
    auto probs_f32 = ct::select(score_valid, ct::exp(scores - row_max), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto denom = ct::sum<1>(ct::element_cast<float>(probs_bf16));

    auto out_acc = ct::mma(probs_bf16, v_tile, ct::full<OutTile>(0.0f));
    out_acc = out_acc / denom;

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_valid = out_rows < kN;
    auto out_safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + out_safe_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

void init_bf16(__nv_bfloat16* ptr, size_t elems) {
    fill_bf16_kernel<<<ceildiv(static_cast<int>(elems), kInitTile), 1>>>(
        ptr, static_cast<long long>(elems));
    CUDA_CHECK(cudaGetLastError());
}

void init_trig_bf16(__nv_bfloat16* cos_f, __nv_bfloat16* sin_f, size_t elems) {
    fill_trig_bf16_kernel<<<ceildiv(static_cast<int>(elems), kInitTile), 1>>>(
        cos_f, sin_f, static_cast<long long>(elems));
    CUDA_CHECK(cudaGetLastError());
}

template <int QRows>
void launch_cutile(const __nv_bfloat16* q,
                   const __nv_bfloat16* k,
                   const __nv_bfloat16* v,
                   __nv_bfloat16* out) {
    dim3 grid(ceildiv(kN, QRows), kBH, 1);
    freq_attention60_cutile_kernel<QRows><<<grid, 1>>>(q, k, v, out);
}

template <int QRows>
void launch_cutile_masked(const __nv_bfloat16* q,
                          const __nv_bfloat16* k,
                          const __nv_bfloat16* v,
                          __nv_bfloat16* out) {
    dim3 grid(ceildiv(kN, QRows), kBH, 1);
    freq_attention60_cutile_masked_kernel<QRows><<<grid, 1>>>(q, k, v, out);
}

template <int QRows, bool SumBF16Denom = true, bool ConstNegInf = false,
          bool UseExp2 = false, int Prob = kFreqProbExp, bool Normalize = true>
void launch_cutile_padded_out60(const __nv_bfloat16* q,
                                const __nv_bfloat16* k,
                                const __nv_bfloat16* v,
                                __nv_bfloat16* out) {
    dim3 grid(ceildiv(kN, QRows), kBH, 1);
    freq_attention60_cutile_padded_out60_kernel<QRows, SumBF16Denom, ConstNegInf,
                                                UseExp2, Prob, Normalize>
        <<<grid, 1>>>(q, k, v, out);
}

template <int QRows, int BHPack, bool SumBF16Denom = true,
          bool ConstNegInf = false, bool UseExp2 = false,
          int Prob = kFreqProbExp, bool Normalize = true>
void launch_cutile_padded_out60_bhpack(const __nv_bfloat16* q,
                                       const __nv_bfloat16* k,
                                       const __nv_bfloat16* v,
                                       __nv_bfloat16* out) {
    dim3 grid(ceildiv(kN, QRows), ceildiv(kBH, BHPack), 1);
    freq_attention60_cutile_padded_out60_bhpack_kernel<
        QRows, BHPack, SumBF16Denom, ConstNegInf, UseExp2, Prob, Normalize>
        <<<grid, 1>>>(q, k, v, out);
}

template <int QRows, int VCols, bool SumBF16Denom = true,
          bool ConstNegInf = false, bool UseExp2 = false,
          int Prob = kFreqProbExp, bool Normalize = true>
void launch_cutile_padded_out60_vsplit(const __nv_bfloat16* q,
                                       const __nv_bfloat16* k,
                                       const __nv_bfloat16* v,
                                       __nv_bfloat16* out) {
    dim3 grid(ceildiv(kN, QRows), kBH, 1);
    freq_attention60_cutile_padded_out60_vsplit_kernel<
        QRows, VCols, SumBF16Denom, ConstNegInf, UseExp2, Prob, Normalize>
        <<<grid, 1>>>(q, k, v, out);
}

void launch_split_packed_qkv_identity_freq60_pad64(const __nv_bfloat16* qkv,
                                                   __nv_bfloat16* q,
                                                   __nv_bfloat16* k,
                                                   __nv_bfloat16* v) {
    long long total = (long long)kBatches * kHeads * kNPad * (kD / 2);
    split_packed_qkv_identity_freq60_pad64_kernel<<<ceildiv(static_cast<int>(total), kInitTile), 1>>>(
        qkv, q, k, v, total);
}

void launch_split_packed_qkv_identity_freq60_pad64_u32(const __nv_bfloat16* qkv,
                                                       __nv_bfloat16* q,
                                                       __nv_bfloat16* k,
                                                       __nv_bfloat16* v) {
    long long total = (long long)kBatches * kHeads * kNPad * (kD / 2);
    split_packed_qkv_identity_freq60_pad64_u32_kernel
        <<<ceildiv(static_cast<int>(total), kInitTile), 1>>>(qkv, q, k, v, total);
}

template <bool VU32 = false>
void launch_split_packed_qkv_rotary_freq60_pad64(const __nv_bfloat16* qkv,
                                                 const __nv_bfloat16* cos_f,
                                                 const __nv_bfloat16* sin_f,
                                                 __nv_bfloat16* q,
                                                 __nv_bfloat16* k,
                                                 __nv_bfloat16* v) {
    long long total = (long long)kBatches * kHeads * kNPad * (kD / 2);
    split_packed_qkv_rotary_freq60_pad64_kernel<VU32>
        <<<ceildiv(static_cast<int>(total), kInitTile), 1>>>(
            qkv, cos_f, sin_f, q, k, v, total);
}

template <bool VU32 = false>
void launch_split_packed_qkv_rotary_freq60_compact(const __nv_bfloat16* qkv,
                                                   const __nv_bfloat16* cos_f,
                                                   const __nv_bfloat16* sin_f,
                                                   __nv_bfloat16* q,
                                                   __nv_bfloat16* k,
                                                   __nv_bfloat16* v) {
    long long total = (long long)kBatches * kHeads * kN * (kD / 2);
    split_packed_qkv_rotary_freq60_compact_kernel<VU32>
        <<<ceildiv(static_cast<int>(total), kInitTile), 1>>>(
            qkv, cos_f, sin_f, q, k, v, total);
}

template <bool VU32 = false>
void launch_split_packed_qkv_rotary_freq60_validpad(const __nv_bfloat16* qkv,
                                                    const __nv_bfloat16* cos_f,
                                                    const __nv_bfloat16* sin_f,
                                                    __nv_bfloat16* q,
                                                    __nv_bfloat16* k,
                                                    __nv_bfloat16* v) {
    long long total = (long long)kBatches * kHeads * kN * (kD / 2);
    split_packed_qkv_rotary_freq60_validpad_kernel<VU32>
        <<<ceildiv(static_cast<int>(total), kInitTile), 1>>>(
            qkv, cos_f, sin_f, q, k, v, total);
}

template <int QRows, bool UseExp2 = false>
void launch_packed_qkv_identity_attention(const __nv_bfloat16* qkv,
                                          __nv_bfloat16* out) {
    dim3 grid(ceildiv(kN, QRows), kBH, 1);
    freq_attention60_packed_qkv_identity_kernel<QRows, UseExp2>
        <<<grid, 1>>>(qkv, out);
}

template <int QRows, bool UseExp2 = false>
void launch_packed_qkv_rotary_attention(const __nv_bfloat16* qkv,
                                        const __nv_bfloat16* cos_f,
                                        const __nv_bfloat16* sin_f,
                                        __nv_bfloat16* out) {
    dim3 grid(ceildiv(kN, QRows), kBH, 1);
    freq_attention60_packed_qkv_rotary_kernel<QRows, UseExp2>
        <<<grid, 1>>>(qkv, cos_f, sin_f, out);
}

bool should_describe(const Options& opts, const char* name) {
    return opts.variant == "all" || opts.variant == name;
}

template <typename Kernel>
void describe_kernel(const Options& opts, const char* name, Kernel kernel, dim3 grid) {
    if (!should_describe(opts, name)) return;

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
        "%-34s grid=(%u,%u,%u) waves/SM=%.1f attr_regs=%d "
        "static_shared=%zuB shared_limit=%d occupancy_active_cta_per_sm=%d "
        "max_threads_per_block=%d local=%zuB const=%zuB ptx=%d binary=%d\n",
        name,
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

void describe_all(const Options& opts) {
    cudaDeviceProp prop{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("GPU: %s SM %d.%d SMs=%d shared_per_sm=%zuB regs_per_sm=%d max_blocks_per_sm=%d\n",
                prop.name,
                prop.major,
                prop.minor,
                prop.multiProcessorCount,
                prop.sharedMemPerMultiprocessor,
                prop.regsPerMultiprocessor,
                prop.maxBlocksPerMultiProcessor);

    describe_kernel(opts,
                    "q8s",
                    freq_attention60_cutile_padded_out60_kernel<8, false, false, false>,
                    dim3(ceildiv(kN, 8), kBH, 1));
    describe_kernel(opts,
                    "q16s",
                    freq_attention60_cutile_padded_out60_kernel<16, false, false, false>,
                    dim3(ceildiv(kN, 16), kBH, 1));
    describe_kernel(opts,
                    "q16s_exp2",
                    freq_attention60_cutile_padded_out60_kernel<16, false, false, true>,
                    dim3(ceildiv(kN, 16), kBH, 1));
    describe_kernel(opts,
                    "q16s_poly3_noclamp",
                    freq_attention60_cutile_padded_out60_kernel<
                        16, false, false, false, kFreqProbPoly3NoClamp>,
                    dim3(ceildiv(kN, 16), kBH, 1));
    describe_kernel(opts,
                    "q16s_poly3_clamp",
                    freq_attention60_cutile_padded_out60_kernel<
                        16, false, false, false, kFreqProbPoly3Clamp>,
                    dim3(ceildiv(kN, 16), kBH, 1));
    describe_kernel(opts,
                    "q16s_poly2_l4",
                    freq_attention60_cutile_padded_out60_kernel<
                        16, false, false, false, kFreqProbPoly2L4>,
                    dim3(ceildiv(kN, 16), kBH, 1));
    describe_kernel(opts,
                    "q16s_nodenom",
                    freq_attention60_cutile_padded_out60_kernel<
                        16, false, false, false, kFreqProbExp, false>,
                    dim3(ceildiv(kN, 16), kBH, 1));
    describe_kernel(opts,
                    "q16s_poly3_clamp_nodenom",
                    freq_attention60_cutile_padded_out60_kernel<
                        16, false, false, false, kFreqProbPoly3Clamp, false>,
                    dim3(ceildiv(kN, 16), kBH, 1));
    describe_kernel(opts,
                    "q16s_bh2",
                    freq_attention60_cutile_padded_out60_bhpack_kernel<
                        16, 2, false, false, false>,
                    dim3(ceildiv(kN, 16), ceildiv(kBH, 2), 1));
    describe_kernel(opts,
                    "q16s_bh4",
                    freq_attention60_cutile_padded_out60_bhpack_kernel<
                        16, 4, false, false, false>,
                    dim3(ceildiv(kN, 16), ceildiv(kBH, 4), 1));
    describe_kernel(opts,
                    "q16s_v32",
                    freq_attention60_cutile_padded_out60_vsplit_kernel<
                        16, 32, false, false, false>,
                    dim3(ceildiv(kN, 16), kBH, 1));
    describe_kernel(opts,
                    "q16s_v16",
                    freq_attention60_cutile_padded_out60_vsplit_kernel<
                        16, 16, false, false, false>,
                    dim3(ceildiv(kN, 16), kBH, 1));
    describe_kernel(opts,
                    "q16sb",
                    freq_attention60_cutile_padded_out60_kernel<16, true, false, false>,
                    dim3(ceildiv(kN, 16), kBH, 1));
    describe_kernel(opts,
                    "q16sc",
                    freq_attention60_cutile_padded_out60_kernel<16, false, true, false>,
                    dim3(ceildiv(kN, 16), kBH, 1));
    describe_kernel(opts,
                    "q16sc_exp2",
                    freq_attention60_cutile_padded_out60_kernel<16, false, true, true>,
                    dim3(ceildiv(kN, 16), kBH, 1));
    describe_kernel(opts,
                    "q32s",
                    freq_attention60_cutile_padded_out60_kernel<32, false, false, false>,
                    dim3(ceildiv(kN, 32), kBH, 1));
    describe_kernel(opts,
                    "q64s",
                    freq_attention60_cutile_padded_out60_kernel<64, false, false, false>,
                    dim3(ceildiv(kN, 64), kBH, 1));
    describe_kernel(opts,
                    "q128s",
                    freq_attention60_cutile_padded_out60_kernel<128, false, false, false>,
                    dim3(ceildiv(kN, 128), kBH, 1));

    describe_kernel(opts,
                    "qkv_rot_split",
                    split_packed_qkv_rotary_freq60_pad64_kernel<false>,
                    dim3(ceildiv((long long)kBatches * kHeads * kNPad * (kD / 2),
                                 kInitTile),
                         1,
                         1));
    describe_kernel(opts,
                    "qkv_rot_split_vu32",
                    split_packed_qkv_rotary_freq60_pad64_kernel<true>,
                    dim3(ceildiv((long long)kBatches * kHeads * kNPad * (kD / 2),
                                 kInitTile),
                         1,
                         1));
    describe_kernel(opts,
                    "qkv_rot_compact_split",
                    split_packed_qkv_rotary_freq60_compact_kernel<false>,
                    dim3(ceildiv((long long)kBatches * kHeads * kN * (kD / 2),
                                 kInitTile),
                         1,
                         1));
    describe_kernel(opts,
                    "qkv_rot_validpad_split",
                    split_packed_qkv_rotary_freq60_validpad_kernel<false>,
                    dim3(ceildiv((long long)kBatches * kHeads * kN * (kD / 2),
                                 kInitTile),
                         1,
                         1));
    describe_kernel(opts,
                    "q16qkv_direct",
                    freq_attention60_packed_qkv_identity_kernel<16, false>,
                    dim3(ceildiv(kN, 16), kBH, 1));
    describe_kernel(opts,
                    "q16qkv_rot_direct",
                    freq_attention60_packed_qkv_rotary_kernel<16, false>,
                    dim3(ceildiv(kN, 16), kBH, 1));
}

void validate_q16(const __nv_bfloat16* d_q,
                  const __nv_bfloat16* d_k,
                  const __nv_bfloat16* d_v,
                  const __nv_bfloat16* d_out) {
    size_t elems = static_cast<size_t>(kNPad) * kD;
    std::vector<__nv_bfloat16> q(elems);
    std::vector<__nv_bfloat16> k(elems);
    std::vector<__nv_bfloat16> v(elems);
    std::vector<__nv_bfloat16> out(elems);
    CUDA_CHECK(cudaMemcpy(q.data(), d_q, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k.data(), d_k, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v.data(), d_v, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    float rms = 0.0f;
    int count = 0;
    for (int r = 0; r < kN; ++r) {
        float scores[kN];
        float row_max = -3.402823466e38f;
        for (int c = 0; c < kN; ++c) {
            float acc = 0.0f;
            for (int d = 0; d < kD; ++d) {
                acc += __bfloat162float(q[r * kD + d]) *
                       __bfloat162float(k[c * kD + d]);
            }
            scores[c] = acc * kScale;
            row_max = std::max(row_max, scores[c]);
        }
        float probs[kN];
        float denom = 0.0f;
        for (int c = 0; c < kN; ++c) {
            __nv_bfloat16 p = __float2bfloat16(std::exp(scores[c] - row_max));
            probs[c] = __bfloat162float(p);
            denom += probs[c];
        }
        for (int d = 0; d < kD; ++d) {
            float acc = 0.0f;
            for (int c = 0; c < kN; ++c) {
                acc += probs[c] * __bfloat162float(v[c * kD + d]);
            }
            float ref = __bfloat162float(__float2bfloat16(acc / denom));
            float got = __bfloat162float(out[r * kD + d]);
            float diff = std::fabs(ref - got);
            max_abs = std::max(max_abs, diff);
            rms += diff * diff;
            ++count;
        }
    }
    rms = std::sqrt(rms / static_cast<float>(count));
    std::printf("  validate BH0 max_abs=%.8g rms=%.8g\n", max_abs, rms);
}

void validate_unpadded(const __nv_bfloat16* d_q,
                       const __nv_bfloat16* d_k,
                       const __nv_bfloat16* d_v,
                       const __nv_bfloat16* d_out,
                       bool sum_bf16_denom = true) {
    size_t elems = static_cast<size_t>(kN) * kD;
    std::vector<__nv_bfloat16> q(elems);
    std::vector<__nv_bfloat16> k(elems);
    std::vector<__nv_bfloat16> v(elems);
    std::vector<__nv_bfloat16> out(elems);
    CUDA_CHECK(cudaMemcpy(q.data(), d_q, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k.data(), d_k, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v.data(), d_v, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    float rms = 0.0f;
    int count = 0;
    for (int r = 0; r < kN; ++r) {
        float scores[kN];
        float row_max = -3.402823466e38f;
        for (int c = 0; c < kN; ++c) {
            float acc = 0.0f;
            for (int d = 0; d < kD; ++d) {
                acc += __bfloat162float(q[r * kD + d]) *
                       __bfloat162float(k[c * kD + d]);
            }
            scores[c] = acc * kScale;
            row_max = std::max(row_max, scores[c]);
        }
        float probs[kN];
        float denom = 0.0f;
        for (int c = 0; c < kN; ++c) {
            float p_f32 = std::exp(scores[c] - row_max);
            __nv_bfloat16 p = __float2bfloat16(p_f32);
            probs[c] = __bfloat162float(p);
            denom += sum_bf16_denom ? probs[c] : p_f32;
        }
        for (int d = 0; d < kD; ++d) {
            float acc = 0.0f;
            for (int c = 0; c < kN; ++c) {
                acc += probs[c] * __bfloat162float(v[c * kD + d]);
            }
            float ref = __bfloat162float(__float2bfloat16(acc / denom));
            float got = __bfloat162float(out[r * kD + d]);
            float diff = std::fabs(ref - got);
            max_abs = std::max(max_abs, diff);
            rms += diff * diff;
            ++count;
        }
    }
    rms = std::sqrt(rms / static_cast<float>(count));
    std::printf("  validate BH0 max_abs=%.8g rms=%.8g\n", max_abs, rms);
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
    std::printf("  compare_vs_qrows_baseline elems=%zu max_abs=%.9g rms=%.9g ref_rms=%.9g rel_rms=%.9g\n",
                elems, max_abs, rms, ref_rms, rel_rms);
}

template <int QRows, bool SumBF16Denom = false, bool ConstNegInf = false,
          bool UseExp2 = false, int Prob = kFreqProbExp, bool Normalize = true,
          int BHPack = 1, int SplitVCols = 0>
void run_padded_out60_source_variant(const Options& opts, const char* name) {
    size_t in_elems = static_cast<size_t>(kBH) * kN * kD;
    size_t pad_elems = static_cast<size_t>(kBH) * kNPad * kD;
    double bytes_gib = static_cast<double>((3 * pad_elems + in_elems) *
                                           sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    __nv_bfloat16* d_out = nullptr;
    __nv_bfloat16* d_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, in_elems * sizeof(__nv_bfloat16)));
    if (opts.compare_baseline) {
        CUDA_CHECK(cudaMalloc(&d_ref, in_elems * sizeof(__nv_bfloat16)));
    }
    init_bf16(d_q, pad_elems);
    init_bf16(d_k, pad_elems);
    init_bf16(d_v, pad_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (opts.compare_baseline) {
        launch_cutile_padded_out60<QRows, false, false>(d_q, d_k, d_v, d_ref);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < opts.warmup; ++i) {
        if constexpr (SplitVCols > 0) {
            launch_cutile_padded_out60_vsplit<QRows, SplitVCols, SumBF16Denom,
                                              ConstNegInf, UseExp2, Prob,
                                              Normalize>(
                d_q, d_k, d_v, d_out);
        } else if constexpr (BHPack == 1) {
            launch_cutile_padded_out60<QRows, SumBF16Denom, ConstNegInf, UseExp2,
                                       Prob, Normalize>(d_q, d_k, d_v, d_out);
        } else {
            launch_cutile_padded_out60_bhpack<QRows, BHPack, SumBF16Denom,
                                              ConstNegInf, UseExp2, Prob,
                                              Normalize>(d_q, d_k, d_v, d_out);
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
        if constexpr (SplitVCols > 0) {
            launch_cutile_padded_out60_vsplit<QRows, SplitVCols, SumBF16Denom,
                                              ConstNegInf, UseExp2, Prob,
                                              Normalize>(
                d_q, d_k, d_v, d_out);
        } else if constexpr (BHPack == 1) {
            launch_cutile_padded_out60<QRows, SumBF16Denom, ConstNegInf, UseExp2,
                                       Prob, Normalize>(d_q, d_k, d_v, d_out);
        } else {
            launch_cutile_padded_out60_bhpack<QRows, BHPack, SumBF16Denom,
                                              ConstNegInf, UseExp2, Prob,
                                              Normalize>(d_q, d_k, d_v, d_out);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.validate) {
        validate_unpadded(d_q, d_k, d_v, d_out, false);
    }
    if (opts.compare_baseline) {
        compare_outputs_to_baseline(d_ref, d_out, in_elems);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_out, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));
    if (d_ref) {
        CUDA_CHECK(cudaFree(d_ref));
    }

    double flops = 4.0 * kBH * kN * kN * kD;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-5s qrows=%d grid=(%d,%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.3f TF/s roof70=%.1f%% checksum=%.4f\n",
        name, QRows, ceildiv(kN, QRows), kBH, bytes_gib, best_ms, median_ms,
        tflops, tflops / kA10gDenseBf16Tflops * 100.0,
        __bfloat162float(checksum));
}

template <bool UseU32Split = false>
void run_packed_qkv_pipeline_variant(const Options& opts, const char* name) {
    size_t qkv_elems = static_cast<size_t>(kBatches) * kN * kQkvFeatures;
    size_t pad_elems = static_cast<size_t>(kBH) * kNPad * kD;
    size_t out_elems = static_cast<size_t>(kBH) * kN * kD;
    double bytes_gib = static_cast<double>((qkv_elems + 3 * pad_elems + 3 * pad_elems +
                                            out_elems) * sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_qkv = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    __nv_bfloat16* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_qkv, qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_qkv, qkv_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto launch_pipeline = [&]() {
        if constexpr (UseU32Split) {
            launch_split_packed_qkv_identity_freq60_pad64_u32(d_qkv, d_q, d_k, d_v);
        } else {
            launch_split_packed_qkv_identity_freq60_pad64(d_qkv, d_q, d_k, d_v);
        }
        launch_cutile_padded_out60<16, false, false, false>(d_q, d_k, d_v, d_out);
    };

    for (int i = 0; i < opts.warmup; ++i) {
        launch_pipeline();
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
        launch_pipeline();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_out, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));

    double flops = 4.0 * kBH * kN * kN * kD;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-14s qrows=16 grid=(%d,%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.3f TF/s roof70=%.1f%% checksum=%.4f\n",
        name, ceildiv(kN, 16), kBH, bytes_gib, best_ms, median_ms,
        tflops, tflops / kA10gDenseBf16Tflops * 100.0,
        __bfloat162float(checksum));
}

template <bool VU32 = false>
void run_packed_qkv_rotary_split_variant(const Options& opts, const char* name) {
    size_t qkv_elems = static_cast<size_t>(kBatches) * kN * kQkvFeatures;
    size_t trig_elems = static_cast<size_t>(kN) * (kD / 2);
    size_t pad_elems = static_cast<size_t>(kBH) * kNPad * kD;
    double bytes_gib = static_cast<double>((qkv_elems + 2 * trig_elems +
                                            3 * pad_elems) *
                                           sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_qkv = nullptr;
    __nv_bfloat16* d_cos = nullptr;
    __nv_bfloat16* d_sin = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    __nv_bfloat16* d_ref_q = nullptr;
    __nv_bfloat16* d_ref_k = nullptr;
    __nv_bfloat16* d_ref_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_qkv, qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_cos, trig_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_sin, trig_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, pad_elems * sizeof(__nv_bfloat16)));
    if (opts.compare_baseline) {
        CUDA_CHECK(cudaMalloc(&d_ref_q, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref_k, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref_v, pad_elems * sizeof(__nv_bfloat16)));
    }
    init_bf16(d_qkv, qkv_elems);
    init_trig_bf16(d_cos, d_sin, trig_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (opts.compare_baseline) {
        launch_split_packed_qkv_rotary_freq60_pad64<false>(
            d_qkv, d_cos, d_sin, d_ref_q, d_ref_k, d_ref_v);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < opts.warmup; ++i) {
        launch_split_packed_qkv_rotary_freq60_pad64<VU32>(d_qkv, d_cos, d_sin,
                                                          d_q, d_k, d_v);
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
        launch_split_packed_qkv_rotary_freq60_pad64<VU32>(d_qkv, d_cos, d_sin,
                                                          d_q, d_k, d_v);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.compare_baseline) {
        compare_outputs_to_baseline(d_ref_q, d_q, pad_elems);
        compare_outputs_to_baseline(d_ref_k, d_k, pad_elems);
        compare_outputs_to_baseline(d_ref_v, d_v, pad_elems);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_q, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    if (d_ref_q) {
        CUDA_CHECK(cudaFree(d_ref_q));
        CUDA_CHECK(cudaFree(d_ref_k));
        CUDA_CHECK(cudaFree(d_ref_v));
    }

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double gbps = bytes_gib / (static_cast<double>(best_ms) * 1.0e-3);
    std::printf(
        "%-18s grid=(%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.1f GiB/s checksum=%.4f\n",
        name,
        ceildiv(static_cast<int>((long long)kBatches * kHeads * kNPad * (kD / 2)),
                kInitTile),
        bytes_gib, best_ms, median_ms, gbps, __bfloat162float(checksum));
}

template <bool VU32 = false>
void run_packed_qkv_rotary_pipeline_variant(const Options& opts, const char* name) {
    size_t qkv_elems = static_cast<size_t>(kBatches) * kN * kQkvFeatures;
    size_t trig_elems = static_cast<size_t>(kN) * (kD / 2);
    size_t pad_elems = static_cast<size_t>(kBH) * kNPad * kD;
    size_t out_elems = static_cast<size_t>(kBH) * kN * kD;
    double bytes_gib = static_cast<double>((qkv_elems + 2 * trig_elems +
                                            3 * pad_elems + 3 * pad_elems +
                                            out_elems) * sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_qkv = nullptr;
    __nv_bfloat16* d_cos = nullptr;
    __nv_bfloat16* d_sin = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    __nv_bfloat16* d_out = nullptr;
    __nv_bfloat16* d_ref_q = nullptr;
    __nv_bfloat16* d_ref_k = nullptr;
    __nv_bfloat16* d_ref_v = nullptr;
    __nv_bfloat16* d_ref_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_qkv, qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_cos, trig_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_sin, trig_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(__nv_bfloat16)));
    if (opts.compare_baseline) {
        CUDA_CHECK(cudaMalloc(&d_ref_q, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref_k, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref_v, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref_out, out_elems * sizeof(__nv_bfloat16)));
    }
    init_bf16(d_qkv, qkv_elems);
    init_trig_bf16(d_cos, d_sin, trig_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (opts.compare_baseline) {
        launch_split_packed_qkv_rotary_freq60_pad64<false>(
            d_qkv, d_cos, d_sin, d_ref_q, d_ref_k, d_ref_v);
        launch_cutile_padded_out60<16, false, false, false>(
            d_ref_q, d_ref_k, d_ref_v, d_ref_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto launch_pipeline = [&]() {
        launch_split_packed_qkv_rotary_freq60_pad64<VU32>(d_qkv, d_cos, d_sin,
                                                          d_q, d_k, d_v);
        launch_cutile_padded_out60<16, false, false, false>(d_q, d_k, d_v, d_out);
    };

    for (int i = 0; i < opts.warmup; ++i) {
        launch_pipeline();
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
        launch_pipeline();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.compare_baseline) {
        compare_outputs_to_baseline(d_ref_out, d_out, out_elems);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_out, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));
    if (d_ref_q) {
        CUDA_CHECK(cudaFree(d_ref_q));
        CUDA_CHECK(cudaFree(d_ref_k));
        CUDA_CHECK(cudaFree(d_ref_v));
        CUDA_CHECK(cudaFree(d_ref_out));
    }

    double flops = 4.0 * kBH * kN * kN * kD;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-18s qrows=16 grid=(%d,%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.3f TF/s roof70=%.1f%% checksum=%.4f\n",
        name, ceildiv(kN, 16), kBH, bytes_gib, best_ms, median_ms,
        tflops, tflops / kA10gDenseBf16Tflops * 100.0,
        __bfloat162float(checksum));
}

template <bool VU32 = false>
void run_packed_qkv_rotary_compact_split_variant(const Options& opts, const char* name) {
    size_t qkv_elems = static_cast<size_t>(kBatches) * kN * kQkvFeatures;
    size_t trig_elems = static_cast<size_t>(kN) * (kD / 2);
    size_t compact_elems = static_cast<size_t>(kBH) * kN * kD;
    double bytes_gib = static_cast<double>((qkv_elems + 2 * trig_elems +
                                            3 * compact_elems) *
                                           sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_qkv = nullptr;
    __nv_bfloat16* d_cos = nullptr;
    __nv_bfloat16* d_sin = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    __nv_bfloat16* d_ref_q = nullptr;
    __nv_bfloat16* d_ref_k = nullptr;
    __nv_bfloat16* d_ref_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_qkv, qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_cos, trig_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_sin, trig_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q, compact_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, compact_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, compact_elems * sizeof(__nv_bfloat16)));
    if (opts.compare_baseline) {
        CUDA_CHECK(cudaMalloc(&d_ref_q, compact_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref_k, compact_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref_v, compact_elems * sizeof(__nv_bfloat16)));
    }
    init_bf16(d_qkv, qkv_elems);
    init_trig_bf16(d_cos, d_sin, trig_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (opts.compare_baseline) {
        launch_split_packed_qkv_rotary_freq60_compact<false>(
            d_qkv, d_cos, d_sin, d_ref_q, d_ref_k, d_ref_v);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < opts.warmup; ++i) {
        launch_split_packed_qkv_rotary_freq60_compact<VU32>(
            d_qkv, d_cos, d_sin, d_q, d_k, d_v);
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
        launch_split_packed_qkv_rotary_freq60_compact<VU32>(
            d_qkv, d_cos, d_sin, d_q, d_k, d_v);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.compare_baseline) {
        compare_outputs_to_baseline(d_ref_q, d_q, compact_elems);
        compare_outputs_to_baseline(d_ref_k, d_k, compact_elems);
        compare_outputs_to_baseline(d_ref_v, d_v, compact_elems);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_q, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    if (d_ref_q) {
        CUDA_CHECK(cudaFree(d_ref_q));
        CUDA_CHECK(cudaFree(d_ref_k));
        CUDA_CHECK(cudaFree(d_ref_v));
    }

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double gbps = bytes_gib / (static_cast<double>(best_ms) * 1.0e-3);
    std::printf(
        "%-26s grid=(%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.1f GiB/s checksum=%.4f\n",
        name,
        ceildiv(static_cast<int>((long long)kBatches * kHeads * kN * (kD / 2)),
                kInitTile),
        bytes_gib, best_ms, median_ms, gbps, __bfloat162float(checksum));
}

template <bool VU32 = false>
void run_packed_qkv_rotary_compact_pipeline_variant(const Options& opts,
                                                    const char* name) {
    size_t qkv_elems = static_cast<size_t>(kBatches) * kN * kQkvFeatures;
    size_t trig_elems = static_cast<size_t>(kN) * (kD / 2);
    size_t compact_elems = static_cast<size_t>(kBH) * kN * kD;
    size_t pad_elems = static_cast<size_t>(kBH) * kNPad * kD;
    size_t out_elems = compact_elems;
    double bytes_gib = static_cast<double>((qkv_elems + 2 * trig_elems +
                                            3 * compact_elems + 3 * compact_elems +
                                            out_elems) * sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_qkv = nullptr;
    __nv_bfloat16* d_cos = nullptr;
    __nv_bfloat16* d_sin = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    __nv_bfloat16* d_out = nullptr;
    __nv_bfloat16* d_ref_q = nullptr;
    __nv_bfloat16* d_ref_k = nullptr;
    __nv_bfloat16* d_ref_v = nullptr;
    __nv_bfloat16* d_ref_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_qkv, qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_cos, trig_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_sin, trig_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q, compact_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, compact_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, compact_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(__nv_bfloat16)));
    if (opts.compare_baseline) {
        CUDA_CHECK(cudaMalloc(&d_ref_q, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref_k, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref_v, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref_out, out_elems * sizeof(__nv_bfloat16)));
    }
    init_bf16(d_qkv, qkv_elems);
    init_trig_bf16(d_cos, d_sin, trig_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (opts.compare_baseline) {
        launch_split_packed_qkv_rotary_freq60_pad64<false>(
            d_qkv, d_cos, d_sin, d_ref_q, d_ref_k, d_ref_v);
        launch_cutile_padded_out60<16, false, false, false>(
            d_ref_q, d_ref_k, d_ref_v, d_ref_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto launch_pipeline = [&]() {
        launch_split_packed_qkv_rotary_freq60_compact<VU32>(
            d_qkv, d_cos, d_sin, d_q, d_k, d_v);
        launch_cutile_masked<16>(d_q, d_k, d_v, d_out);
    };

    for (int i = 0; i < opts.warmup; ++i) {
        launch_pipeline();
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
        launch_pipeline();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.compare_baseline) {
        compare_outputs_to_baseline(d_ref_out, d_out, out_elems);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_out, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));
    if (d_ref_q) {
        CUDA_CHECK(cudaFree(d_ref_q));
        CUDA_CHECK(cudaFree(d_ref_k));
        CUDA_CHECK(cudaFree(d_ref_v));
        CUDA_CHECK(cudaFree(d_ref_out));
    }

    double flops = 4.0 * kBH * kN * kN * kD;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-26s qrows=16 grid=(%d,%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.3f TF/s roof70=%.1f%% checksum=%.4f\n",
        name, ceildiv(kN, 16), kBH, bytes_gib, best_ms, median_ms,
        tflops, tflops / kA10gDenseBf16Tflops * 100.0,
        __bfloat162float(checksum));
}

template <bool VU32 = false>
void run_packed_qkv_rotary_validpad_split_variant(const Options& opts,
                                                  const char* name) {
    size_t qkv_elems = static_cast<size_t>(kBatches) * kN * kQkvFeatures;
    size_t trig_elems = static_cast<size_t>(kN) * (kD / 2);
    size_t pad_elems = static_cast<size_t>(kBH) * kNPad * kD;
    double bytes_gib = static_cast<double>((qkv_elems + 2 * trig_elems +
                                            3 * static_cast<size_t>(kBH) * kN * kD) *
                                           sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_qkv = nullptr;
    __nv_bfloat16* d_cos = nullptr;
    __nv_bfloat16* d_sin = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    __nv_bfloat16* d_ref_q = nullptr;
    __nv_bfloat16* d_ref_k = nullptr;
    __nv_bfloat16* d_ref_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_qkv, qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_cos, trig_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_sin, trig_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, pad_elems * sizeof(__nv_bfloat16)));
    if (opts.compare_baseline) {
        CUDA_CHECK(cudaMalloc(&d_ref_q, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref_k, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref_v, pad_elems * sizeof(__nv_bfloat16)));
    }
    init_bf16(d_qkv, qkv_elems);
    init_trig_bf16(d_cos, d_sin, trig_elems);
    CUDA_CHECK(cudaMemset(d_q, 0, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_k, 0, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_v, 0, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (opts.compare_baseline) {
        CUDA_CHECK(cudaMemset(d_ref_q, 0, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_ref_k, 0, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_ref_v, 0, pad_elems * sizeof(__nv_bfloat16)));
        launch_split_packed_qkv_rotary_freq60_validpad<false>(
            d_qkv, d_cos, d_sin, d_ref_q, d_ref_k, d_ref_v);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < opts.warmup; ++i) {
        launch_split_packed_qkv_rotary_freq60_validpad<VU32>(
            d_qkv, d_cos, d_sin, d_q, d_k, d_v);
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
        launch_split_packed_qkv_rotary_freq60_validpad<VU32>(
            d_qkv, d_cos, d_sin, d_q, d_k, d_v);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.compare_baseline) {
        compare_outputs_to_baseline(d_ref_q, d_q, pad_elems);
        compare_outputs_to_baseline(d_ref_k, d_k, pad_elems);
        compare_outputs_to_baseline(d_ref_v, d_v, pad_elems);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_q, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    if (d_ref_q) {
        CUDA_CHECK(cudaFree(d_ref_q));
        CUDA_CHECK(cudaFree(d_ref_k));
        CUDA_CHECK(cudaFree(d_ref_v));
    }

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double gbps = bytes_gib / (static_cast<double>(best_ms) * 1.0e-3);
    std::printf(
        "%-26s grid=(%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.1f GiB/s checksum=%.4f\n",
        name,
        ceildiv(static_cast<int>((long long)kBatches * kHeads * kN * (kD / 2)),
                kInitTile),
        bytes_gib, best_ms, median_ms, gbps, __bfloat162float(checksum));
}

template <bool VU32 = false>
void run_packed_qkv_rotary_validpad_pipeline_variant(const Options& opts,
                                                     const char* name) {
    size_t qkv_elems = static_cast<size_t>(kBatches) * kN * kQkvFeatures;
    size_t trig_elems = static_cast<size_t>(kN) * (kD / 2);
    size_t pad_elems = static_cast<size_t>(kBH) * kNPad * kD;
    size_t out_elems = static_cast<size_t>(kBH) * kN * kD;
    double bytes_gib = static_cast<double>((qkv_elems + 2 * trig_elems +
                                            3 * static_cast<size_t>(kBH) * kN * kD +
                                            3 * pad_elems + out_elems) *
                                           sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_qkv = nullptr;
    __nv_bfloat16* d_cos = nullptr;
    __nv_bfloat16* d_sin = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    __nv_bfloat16* d_out = nullptr;
    __nv_bfloat16* d_ref_q = nullptr;
    __nv_bfloat16* d_ref_k = nullptr;
    __nv_bfloat16* d_ref_v = nullptr;
    __nv_bfloat16* d_ref_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_qkv, qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_cos, trig_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_sin, trig_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(__nv_bfloat16)));
    if (opts.compare_baseline) {
        CUDA_CHECK(cudaMalloc(&d_ref_q, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref_k, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref_v, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_ref_out, out_elems * sizeof(__nv_bfloat16)));
    }
    init_bf16(d_qkv, qkv_elems);
    init_trig_bf16(d_cos, d_sin, trig_elems);
    CUDA_CHECK(cudaMemset(d_q, 0, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_k, 0, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_v, 0, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (opts.compare_baseline) {
        launch_split_packed_qkv_rotary_freq60_pad64<false>(
            d_qkv, d_cos, d_sin, d_ref_q, d_ref_k, d_ref_v);
        launch_cutile_padded_out60<16, false, false, false>(
            d_ref_q, d_ref_k, d_ref_v, d_ref_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto launch_pipeline = [&]() {
        launch_split_packed_qkv_rotary_freq60_validpad<VU32>(
            d_qkv, d_cos, d_sin, d_q, d_k, d_v);
        launch_cutile_padded_out60<16, false, false, false>(d_q, d_k, d_v, d_out);
    };

    for (int i = 0; i < opts.warmup; ++i) {
        launch_pipeline();
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
        launch_pipeline();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.compare_baseline) {
        compare_outputs_to_baseline(d_ref_out, d_out, out_elems);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_out, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));
    if (d_ref_q) {
        CUDA_CHECK(cudaFree(d_ref_q));
        CUDA_CHECK(cudaFree(d_ref_k));
        CUDA_CHECK(cudaFree(d_ref_v));
        CUDA_CHECK(cudaFree(d_ref_out));
    }

    double flops = 4.0 * kBH * kN * kN * kD;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-26s qrows=16 grid=(%d,%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.3f TF/s checksum=%.4f\n",
        name, ceildiv(kN, 16), kBH, bytes_gib, best_ms, median_ms,
        tflops, __bfloat162float(checksum));
}

template <int QRows, bool UseExp2 = false>
void run_packed_qkv_direct_variant(const Options& opts, const char* name) {
    size_t qkv_elems = static_cast<size_t>(kBatches) * kN * kQkvFeatures;
    size_t pad_elems = static_cast<size_t>(kBH) * kNPad * kD;
    size_t out_elems = static_cast<size_t>(kBH) * kN * kD;
    double bytes_gib = static_cast<double>((qkv_elems + out_elems) *
                                           sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_qkv = nullptr;
    __nv_bfloat16* d_out = nullptr;
    __nv_bfloat16* d_ref = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_qkv, qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(__nv_bfloat16)));
    if (opts.compare_baseline) {
        CUDA_CHECK(cudaMalloc(&d_ref, out_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_q, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_k, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_v, pad_elems * sizeof(__nv_bfloat16)));
    }
    init_bf16(d_qkv, qkv_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (opts.compare_baseline) {
        launch_split_packed_qkv_identity_freq60_pad64(d_qkv, d_q, d_k, d_v);
        launch_cutile_padded_out60<16, false, false, false>(d_q, d_k, d_v, d_ref);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < opts.warmup; ++i) {
        launch_packed_qkv_identity_attention<QRows, UseExp2>(d_qkv, d_out);
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
        launch_packed_qkv_identity_attention<QRows, UseExp2>(d_qkv, d_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.compare_baseline) {
        compare_outputs_to_baseline(d_ref, d_out, out_elems);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_out, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_out));
    if (d_ref) {
        CUDA_CHECK(cudaFree(d_ref));
        CUDA_CHECK(cudaFree(d_q));
        CUDA_CHECK(cudaFree(d_k));
        CUDA_CHECK(cudaFree(d_v));
    }

    double flops = 4.0 * kBH * kN * kN * kD;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-14s qrows=%d grid=(%d,%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.3f TF/s checksum=%.4f\n",
        name, QRows, ceildiv(kN, QRows), kBH, bytes_gib, best_ms, median_ms,
        tflops, __bfloat162float(checksum));
}

template <int QRows, bool UseExp2 = false>
void run_packed_qkv_rotary_direct_variant(const Options& opts, const char* name) {
    size_t qkv_elems = static_cast<size_t>(kBatches) * kN * kQkvFeatures;
    size_t trig_elems = static_cast<size_t>(kN) * (kD / 2);
    size_t pad_elems = static_cast<size_t>(kBH) * kNPad * kD;
    size_t out_elems = static_cast<size_t>(kBH) * kN * kD;
    double bytes_gib = static_cast<double>((qkv_elems + 2 * trig_elems +
                                            out_elems) *
                                           sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_qkv = nullptr;
    __nv_bfloat16* d_cos = nullptr;
    __nv_bfloat16* d_sin = nullptr;
    __nv_bfloat16* d_out = nullptr;
    __nv_bfloat16* d_ref = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_qkv, qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_cos, trig_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_sin, trig_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(__nv_bfloat16)));
    if (opts.compare_baseline) {
        CUDA_CHECK(cudaMalloc(&d_ref, out_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_q, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_k, pad_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_v, pad_elems * sizeof(__nv_bfloat16)));
    }
    init_bf16(d_qkv, qkv_elems);
    init_trig_bf16(d_cos, d_sin, trig_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (opts.compare_baseline) {
        launch_split_packed_qkv_rotary_freq60_pad64<false>(
            d_qkv, d_cos, d_sin, d_q, d_k, d_v);
        launch_cutile_padded_out60<16, false, false, false>(d_q, d_k, d_v, d_ref);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < opts.warmup; ++i) {
        launch_packed_qkv_rotary_attention<QRows, UseExp2>(d_qkv, d_cos, d_sin,
                                                           d_out);
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
        launch_packed_qkv_rotary_attention<QRows, UseExp2>(d_qkv, d_cos, d_sin,
                                                           d_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.compare_baseline) {
        compare_outputs_to_baseline(d_ref, d_out, out_elems);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_out, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_out));
    if (d_ref) {
        CUDA_CHECK(cudaFree(d_ref));
        CUDA_CHECK(cudaFree(d_q));
        CUDA_CHECK(cudaFree(d_k));
        CUDA_CHECK(cudaFree(d_v));
    }

    double flops = 4.0 * kBH * kN * kN * kD;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-19s qrows=%d grid=(%d,%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.3f TF/s checksum=%.4f\n",
        name, QRows, ceildiv(kN, QRows), kBH, bytes_gib, best_ms, median_ms,
        tflops, __bfloat162float(checksum));
}

template <int QRows>
void run_variant(const Options& opts, const char* name) {
    size_t elems = static_cast<size_t>(kBH) * kNPad * kD;
    double bytes_gib = static_cast<double>(4 * elems * sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    __nv_bfloat16* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, elems * sizeof(__nv_bfloat16)));
    init_bf16(d_q, elems);
    init_bf16(d_k, elems);
    init_bf16(d_v, elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch_cutile<QRows>(d_q, d_k, d_v, d_out);
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
        launch_cutile<QRows>(d_q, d_k, d_v, d_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.validate) {
        validate_q16(d_q, d_k, d_v, d_out);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_out, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));

    double flops = 4.0 * kBH * kN * kN * kD;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-4s qrows=%d grid=(%d,%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.3f TF/s checksum=%.4f\n",
        name, QRows, ceildiv(kN, QRows), kBH, bytes_gib, best_ms, median_ms,
        tflops, __bfloat162float(checksum));
}

template <int QRows>
void run_masked_variant(const Options& opts, const char* name) {
    size_t elems = static_cast<size_t>(kBH) * kN * kD;
    double bytes_gib = static_cast<double>(4 * elems * sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    __nv_bfloat16* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, elems * sizeof(__nv_bfloat16)));
    init_bf16(d_q, elems);
    init_bf16(d_k, elems);
    init_bf16(d_v, elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch_cutile_masked<QRows>(d_q, d_k, d_v, d_out);
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
        launch_cutile_masked<QRows>(d_q, d_k, d_v, d_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.validate) {
        validate_unpadded(d_q, d_k, d_v, d_out);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_out, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));

    double flops = 4.0 * kBH * kN * kN * kD;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-4s qrows=%d grid=(%d,%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.3f TF/s checksum=%.4f\n",
        name, QRows, ceildiv(kN, QRows), kBH, bytes_gib, best_ms, median_ms,
        tflops, __bfloat162float(checksum));
}

template <int QRows>
void run_padded_pipeline_variant(const Options& opts, const char* name) {
    size_t in_elems = static_cast<size_t>(kBH) * kN * kD;
    size_t pad_elems = static_cast<size_t>(kBH) * kNPad * kD;
    double bytes_gib = static_cast<double>((3 * in_elems + 3 * pad_elems + in_elems) *
                                           sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    __nv_bfloat16* d_qp = nullptr;
    __nv_bfloat16* d_kp = nullptr;
    __nv_bfloat16* d_vp = nullptr;
    __nv_bfloat16* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, in_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, in_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, in_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_qp, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_kp, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_vp, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, in_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_q, in_elems);
    init_bf16(d_k, in_elems);
    init_bf16(d_v, in_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto launch_pipeline = [&]() {
        long long total = static_cast<long long>(pad_elems);
        pad3_freq60_to64_kernel<<<ceildiv(static_cast<int>(pad_elems), kInitTile), 1>>>(
            d_q, d_k, d_v, d_qp, d_kp, d_vp, total);
        launch_cutile_padded_out60<QRows>(d_qp, d_kp, d_vp, d_out);
    };

    for (int i = 0; i < opts.warmup; ++i) {
        launch_pipeline();
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
        launch_pipeline();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.validate) {
        validate_unpadded(d_q, d_k, d_v, d_out);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_out, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_qp));
    CUDA_CHECK(cudaFree(d_kp));
    CUDA_CHECK(cudaFree(d_vp));
    CUDA_CHECK(cudaFree(d_out));

    double flops = 4.0 * kBH * kN * kN * kD;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-5s qrows=%d grid=(%d,%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.3f TF/s checksum=%.4f\n",
        name, QRows, ceildiv(kN, QRows), kBH, bytes_gib, best_ms, median_ms,
        tflops, __bfloat162float(checksum));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        if (opts.describe) {
            describe_all(opts);
            return 0;
        }
        if (opts.compare_baseline && opts.variant == "all") {
            throw std::runtime_error("--compare-baseline requires --variant NAME");
        }
        if (opts.variant == "all" || opts.variant == "q8") {
            run_variant<8>(opts, "q8");
        }
        if (opts.variant == "all" || opts.variant == "q16") {
            run_variant<16>(opts, "q16");
        }
        if (opts.variant == "all" || opts.variant == "q32") {
            run_variant<32>(opts, "q32");
        }
        if (opts.variant == "all" || opts.variant == "q64") {
            run_variant<64>(opts, "q64");
        }
        if (opts.variant == "all" || opts.variant == "q128") {
            run_variant<128>(opts, "q128");
        }
        if (opts.variant == "all" || opts.variant == "q8m") {
            run_masked_variant<8>(opts, "q8m");
        }
        if (opts.variant == "all" || opts.variant == "q16m") {
            run_masked_variant<16>(opts, "q16m");
        }
        if (opts.variant == "all" || opts.variant == "q32m") {
            run_masked_variant<32>(opts, "q32m");
        }
        if (opts.variant == "all" || opts.variant == "q64m") {
            run_masked_variant<64>(opts, "q64m");
        }
        if (opts.variant == "all" || opts.variant == "q128m") {
            run_masked_variant<128>(opts, "q128m");
        }
        if (opts.variant == "all" || opts.variant == "q8p") {
            run_padded_pipeline_variant<8>(opts, "q8p");
        }
        if (opts.variant == "all" || opts.variant == "q16p") {
            run_padded_pipeline_variant<16>(opts, "q16p");
        }
        if (opts.variant == "all" || opts.variant == "q32p") {
            run_padded_pipeline_variant<32>(opts, "q32p");
        }
        if (opts.variant == "all" || opts.variant == "q64p") {
            run_padded_pipeline_variant<64>(opts, "q64p");
        }
        if (opts.variant == "all" || opts.variant == "q128p") {
            run_padded_pipeline_variant<128>(opts, "q128p");
        }
        if (opts.variant == "all" || opts.variant == "q8s") {
            run_padded_out60_source_variant<8>(opts, "q8s");
        }
        if (opts.variant == "all" || opts.variant == "q16s") {
            run_padded_out60_source_variant<16>(opts, "q16s");
        }
        if (opts.variant == "all" || opts.variant == "q16s_exp2") {
            run_padded_out60_source_variant<16, false, false, true>(opts, "q16s_exp2");
        }
        if (opts.variant == "all" || opts.variant == "q16s_poly3_noclamp") {
            run_padded_out60_source_variant<16, false, false, false,
                                            kFreqProbPoly3NoClamp>(
                opts, "q16s_poly3_noclamp");
        }
        if (opts.variant == "all" || opts.variant == "q16s_poly3_clamp") {
            run_padded_out60_source_variant<16, false, false, false,
                                            kFreqProbPoly3Clamp>(
                opts, "q16s_poly3_clamp");
        }
        if (opts.variant == "all" || opts.variant == "q16s_poly2_l4") {
            run_padded_out60_source_variant<16, false, false, false,
                                            kFreqProbPoly2L4>(
                opts, "q16s_poly2_l4");
        }
        if (opts.variant == "all" || opts.variant == "q16s_nodenom") {
            run_padded_out60_source_variant<16, false, false, false,
                                            kFreqProbExp, false>(
                opts, "q16s_nodenom");
        }
        if (opts.variant == "all" || opts.variant == "q16s_poly3_clamp_nodenom") {
            run_padded_out60_source_variant<16, false, false, false,
                                            kFreqProbPoly3Clamp, false>(
                opts, "q16s_poly3_clamp_nodenom");
        }
        if (opts.variant == "all" || opts.variant == "q16s_bh2") {
            run_padded_out60_source_variant<16, false, false, false,
                                            kFreqProbExp, true, 2>(
                opts, "q16s_bh2");
        }
        if (opts.variant == "all" || opts.variant == "q16s_bh4") {
            run_padded_out60_source_variant<16, false, false, false,
                                            kFreqProbExp, true, 4>(
                opts, "q16s_bh4");
        }
        if (opts.variant == "all" || opts.variant == "q16s_v32") {
            run_padded_out60_source_variant<16, false, false, false,
                                            kFreqProbExp, true, 1, 32>(
                opts, "q16s_v32");
        }
        if (opts.variant == "all" || opts.variant == "q16s_v16") {
            run_padded_out60_source_variant<16, false, false, false,
                                            kFreqProbExp, true, 1, 16>(
                opts, "q16s_v16");
        }
        if (opts.variant == "all" || opts.variant == "q16sb") {
            run_padded_out60_source_variant<16, true>(opts, "q16sb");
        }
        if (opts.variant == "all" || opts.variant == "q16sc") {
            run_padded_out60_source_variant<16, false, true>(opts, "q16sc");
        }
        if (opts.variant == "all" || opts.variant == "q16sc_exp2") {
            run_padded_out60_source_variant<16, false, true, true>(opts, "q16sc_exp2");
        }
        if (opts.variant == "all" || opts.variant == "q16qkv_pipe") {
            run_packed_qkv_pipeline_variant<false>(opts, "q16qkv_pipe");
        }
        if (opts.variant == "all" || opts.variant == "q16qkv_pipe_u32") {
            run_packed_qkv_pipeline_variant<true>(opts, "q16qkv_pipe_u32");
        }
        if (opts.variant == "all" || opts.variant == "qkv_rot_split") {
            run_packed_qkv_rotary_split_variant<false>(opts, "qkv_rot_split");
        }
        if (opts.variant == "all" || opts.variant == "qkv_rot_split_vu32") {
            run_packed_qkv_rotary_split_variant<true>(opts, "qkv_rot_split_vu32");
        }
        if (opts.variant == "all" || opts.variant == "q16qkv_rot_pipe") {
            run_packed_qkv_rotary_pipeline_variant<false>(opts, "q16qkv_rot_pipe");
        }
        if (opts.variant == "all" || opts.variant == "q16qkv_rot_pipe_vu32") {
            run_packed_qkv_rotary_pipeline_variant<true>(opts, "q16qkv_rot_pipe_vu32");
        }
        if (opts.variant == "all" || opts.variant == "qkv_rot_compact_split") {
            run_packed_qkv_rotary_compact_split_variant<false>(
                opts, "qkv_rot_compact_split");
        }
        if (opts.variant == "all" || opts.variant == "qkv_rot_compact_split_vu32") {
            run_packed_qkv_rotary_compact_split_variant<true>(
                opts, "qkv_rot_compact_split_vu32");
        }
        if (opts.variant == "all" || opts.variant == "q16qkv_rot_compact_pipe") {
            run_packed_qkv_rotary_compact_pipeline_variant<false>(
                opts, "q16qkv_rot_compact_pipe");
        }
        if (opts.variant == "all" || opts.variant == "q16qkv_rot_compact_pipe_vu32") {
            run_packed_qkv_rotary_compact_pipeline_variant<true>(
                opts, "q16qkv_rot_compact_pipe_vu32");
        }
        if (opts.variant == "all" || opts.variant == "qkv_rot_validpad_split") {
            run_packed_qkv_rotary_validpad_split_variant<false>(
                opts, "qkv_rot_validpad_split");
        }
        if (opts.variant == "all" || opts.variant == "qkv_rot_validpad_split_vu32") {
            run_packed_qkv_rotary_validpad_split_variant<true>(
                opts, "qkv_rot_validpad_split_vu32");
        }
        if (opts.variant == "all" || opts.variant == "q16qkv_rot_validpad_pipe") {
            run_packed_qkv_rotary_validpad_pipeline_variant<false>(
                opts, "q16qkv_rot_validpad_pipe");
        }
        if (opts.variant == "all" || opts.variant == "q16qkv_rot_validpad_pipe_vu32") {
            run_packed_qkv_rotary_validpad_pipeline_variant<true>(
                opts, "q16qkv_rot_validpad_pipe_vu32");
        }
        if (opts.variant == "all" || opts.variant == "q16qkv_rot_direct") {
            run_packed_qkv_rotary_direct_variant<16>(opts, "q16qkv_rot_direct");
        }
        if (opts.variant == "all" || opts.variant == "q32qkv_rot_direct") {
            run_packed_qkv_rotary_direct_variant<32>(opts, "q32qkv_rot_direct");
        }
        if (opts.variant == "all" || opts.variant == "q64qkv_rot_direct") {
            run_packed_qkv_rotary_direct_variant<64>(opts, "q64qkv_rot_direct");
        }
        if (opts.variant == "all" || opts.variant == "q16qkv_direct") {
            run_packed_qkv_direct_variant<16>(opts, "q16qkv_direct");
        }
        if (opts.variant == "all" || opts.variant == "q32qkv_direct") {
            run_packed_qkv_direct_variant<32>(opts, "q32qkv_direct");
        }
        if (opts.variant == "all" || opts.variant == "q64qkv_direct") {
            run_packed_qkv_direct_variant<64>(opts, "q64qkv_direct");
        }
        if (opts.variant == "all" || opts.variant == "q32s") {
            run_padded_out60_source_variant<32>(opts, "q32s");
        }
        if (opts.variant == "all" || opts.variant == "q64s") {
            run_padded_out60_source_variant<64>(opts, "q64s");
        }
        if (opts.variant == "all" || opts.variant == "q128s") {
            run_padded_out60_source_variant<128>(opts, "q128s");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
