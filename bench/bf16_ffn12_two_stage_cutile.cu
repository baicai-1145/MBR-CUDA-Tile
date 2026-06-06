#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            throw std::runtime_error(std::string(#call) + " failed: " +          \
                                     cudaGetErrorString(err__));                 \
        }                                                                        \
    } while (0)

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kM = 78048;
constexpr int kIn = 256;
constexpr int kHidden = 1024;
constexpr int kOut = 256;
constexpr int kInitTile = 256;
constexpr double kA10gDenseBf16Tflops = 70.0;

struct Options {
    std::string variant = "all";
    int warmup = 10;
    int iters = 100;
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
        if (std::strcmp(argv[i], "--variant") == 0) {
            opts.variant = need_value(argv[i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0) {
            opts.warmup = parse_int_arg(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            opts.iters = parse_int_arg(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--describe") == 0) {
            opts.describe = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: bench_bf16_ffn12_two_stage_cutile [options]\n"
                "  --variant NAME  all, fused_pairh32_lat2,\n"
                "                  fused_tm16_pairh32_lat2_outnoround,\n"
                "                  fused_pairh16_lat2_outnoround,\n"
                "                  fused_pairh16_tk32_lat2_outnoround,\n"
                "                  fused_pairh32_tk32_lat2_outnoround,\n"
                "                  fused_pairh32_tk128_lat2_outnoround,\n"
                "                  fused_pairh32_lat1_outnoround,\n"
                "                  fused_pairh32_w1lat1_w2lat2_outnoround,\n"
                "                  fused_pairh32_w1lat2_w2lat1_outnoround,\n"
                "                  fused_pairh32_lat3_outnoround,\n"
                "                  fused_pairh32_a_persist_lat2_outnoround,\n"
                "                  fused_pairh32_serial_out_halves_lat2_outnoround,\n"
                "                  fused_pairh32_tn64_y2_lat2_outnoround,\n"
                "                  fused_pairh32_tn64x4_lat2_outnoround,\n"
                "                  fused_pairh32_out256_lat2_outnoround,\n"
                "                  fused_pairh32_pair2_lat2_outnoround,\n"
                "                  fused_pairh32_w2stream_lat2_outnoround,\n"
                "                  fused_pairh32_w2pair_lat2_outnoround,\n"
                "                  fused_pairh32_bias_broadcast_lat2,\n"
                "                  fused_pairh32_bias_broadcast_odd5_lat2,\n"
                "                  fused_pairh32_bias_broadcast_odd5_w2byhidden_lat2,\n"
                "                  fused_tm16_pairh32_bias_broadcast_odd5_lat2,\n"
                "                  fused_pairh32_bias_broadcast_odd5_lat1,\n"
                "                  fused_pairh32_bias_broadcast_odd5_w1lat1_w2lat2,\n"
                "                  fused_pairh32_bias_broadcast_odd5_w1lat2_w2lat1,\n"
                "                  fused_pairh32_bias_broadcast_odd5_lat3,\n"
                "                  two_stage_poly9_tk64,\n"
                "                  two_stage_poly9_tk64_ffn2_tn32,\n"
                "                  two_stage_poly9_tk64_ffn2_tn64,\n"
                "                  two_stage_poly9_tk64_ffn2_tn128\n"
                "  --warmup N      warmup launches, default 10\n"
                "  --iters N       measured launches, default 100\n"
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

template <typename T>
static __tile__ auto bf16_round(T value) {
    return ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(value));
}

template <typename TileT>
static __tile__ auto gelu_erf_poly9_l30(TileT x) {
    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto ax = ct::abs(x);
    auto z = ax * ax;
    auto p = ((((((((0.00000005422539767f * z - 0.000002440964777f) * z +
                    0.00004855766724f) * z - 0.0005709642654f) * z +
                  0.004507274577f) * z - 0.02579950512f) * z +
                0.1120213868f) * z - 0.3758834075f) * z +
              1.128367753f);
    auto erf_abs = ct::min(ct::max(ax * p, zero), one);
    auto erf_approx = ct::select(x < zero, zero - erf_abs, erf_abs);
    return 0.5f * x * (one + erf_approx);
}

template <typename TileT>
static __tile__ auto gelu_erf_odd5_l175(TileT x) {
    auto zero = x * 0.0f;
    auto one = zero + 1.0f;
    auto ax = ct::abs(x);
    auto z = ax * ax;
    auto p = ((0.00671723077f * z - 0.116092831f) * z + 1.12144713f);
    auto erf_abs = ct::min(ct::max(ax * p, zero), one);
    auto erf_approx = ct::select(x < zero, zero - erf_abs, erf_abs);
    return 0.5f * x * (one + erf_approx);
}

template <bool Odd5, typename TileT>
static __tile__ auto gelu_selected(TileT x) {
    if constexpr (Odd5) {
        return gelu_erf_odd5_l175(x);
    } else {
        return gelu_erf_poly9_l30(x);
    }
}

using I64InitTile = ct::tile<long long, ct::shape<kInitTile>>;
using F32InitTile = ct::tile<float, ct::shape<kInitTile>>;

__tile_global__ void fill_bf16_kernel(__nv_bfloat16* __restrict__ dst, long long total) {
    dst = ct::assume_aligned(dst, 16_ic);
    I64InitTile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64InitTile>();
    auto in_bounds = idx < total;
    F32InitTile values =
        0.125f + ct::element_cast<float>((idx * 17LL) & 1023LL) * 0.000244140625f;
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

void init_bf16(__nv_bfloat16* ptr, size_t elems) {
    fill_bf16_kernel<<<ceildiv(static_cast<int>(elems), kInitTile), 1>>>(
        ptr, static_cast<long long>(elems));
    CUDA_CHECK(cudaGetLastError());
}

template <int TM, int TN, int TK>
__tile_global__ void ffn1_poly9_tk64_kernel(const __nv_bfloat16* __restrict__ a,
                                            const __nv_bfloat16* __restrict__ w1_nt,
                                            const __nv_bfloat16* __restrict__ b1,
                                            __nv_bfloat16* __restrict__ hidden) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using I64Tile = ct::tile<long long, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    hidden = ct::assume_aligned(hidden, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto hidden_view = ct::partition_view{
        ct::tensor_span{hidden, ct::shape<kM, kHidden>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
        acc = ct::mma(a_view.load(tile_m, kk), w1_view.load(kk, tile_n), acc);
    }

    I64Tile local = ct::iota<I64Tile>();
    auto cols = static_cast<long long>(tile_n) * TN + (local % TN);
    auto bias = ct::element_cast<float>(ct::load(b1 + cols));
    auto value = gelu_erf_poly9_l30(bf16_round(acc) + bias);
    hidden_view.store(ct::element_cast<__nv_bfloat16>(value), tile_m, tile_n);
}

template <int TM, int TN, int TK>
__tile_global__ void ffn2_tn64_kernel(const __nv_bfloat16* __restrict__ hidden,
                                      const __nv_bfloat16* __restrict__ w2_nt,
                                      const __nv_bfloat16* __restrict__ b2,
                                      __nv_bfloat16* __restrict__ out) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using I64Tile = ct::tile<long long, ct::shape<TM, TN>>;

    hidden = ct::assume_aligned(hidden, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto hidden_view = ct::partition_view{
        ct::tensor_span{hidden, ct::shape<kM, kHidden>{}},
        ct::shape<TM, TK>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{kHidden / TK})) {
        acc = ct::mma(hidden_view.load(tile_m, kk), w2_view.load(kk, tile_n), acc);
    }

    I64Tile local = ct::iota<I64Tile>();
    auto cols = static_cast<long long>(tile_n) * TN + (local % TN);
    auto bias = ct::element_cast<float>(ct::load(b2 + cols));
    auto value = bf16_round(acc) + bias;
    out_view.store(ct::element_cast<__nv_bfloat16>(value), tile_m, tile_n);
}

template <int TM, int TK, int THidden, int MemoryLatency = 2, int W2MemoryLatency = MemoryLatency>
__tile_global__ void ffn12_fused_pairh32_lat2_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutHalf>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using W1Tile = ct::tile<__nv_bfloat16, ct::shape<TK, THidden>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<THidden, OutHalf>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, OutHalf>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<THidden, OutHalf>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, OutHalf>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    auto out_acc0 = ct::full<OutAccTile>(0.0f);
    auto out_acc1 = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            ATile a_tile;
            W1Tile w1_0;
            W1Tile w1_1;
            [[cutile::hint(0, latency=MemoryLatency)]]
            a_tile = a_view.load(tile_m, kk);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_0 = w1_view.load(kk, hidden_tile0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_1 = w1_view.load(kk, hidden_tile1);
            hidden_acc0 = ct::mma(a_tile, w1_0, hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_1, hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_erf_poly9_l30(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_erf_poly9_l30(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        W2Tile w2_00;
        W2Tile w2_01;
        W2Tile w2_10;
        W2Tile w2_11;
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_00 = w2_view.load(hidden_tile0, 0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_01 = w2_view.load(hidden_tile0, 1);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_10 = w2_view.load(hidden_tile1, 0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_11 = w2_view.load(hidden_tile1, 1);
        out_acc0 = ct::mma(hidden_bf16_0, w2_00, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_0, w2_01, out_acc1);
        out_acc0 = ct::mma(hidden_bf16_1, w2_10, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_1, w2_11, out_acc1);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % OutHalf;
    auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto out_bias1 = ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols));
    auto value0 = out_acc0 + out_bias0;
    auto value1 = out_acc1 + out_bias1;
    out_view.store(ct::element_cast<__nv_bfloat16>(value0), tile_m, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(value1), tile_m, 1);
}

template <int TM, int TK, int THidden, int MemoryLatency = 2, int W2MemoryLatency = MemoryLatency>
__tile_global__ void ffn12_fused_pairh32_serial_out_halves_lat2_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutHalf>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using W1Tile = ct::tile<__nv_bfloat16, ct::shape<TK, THidden>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<THidden, OutHalf>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, OutHalf>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<THidden, OutHalf>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, OutHalf>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % OutHalf;

    auto out_acc = ct::full<OutAccTile>(0.0f);
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            ATile a_tile;
            W1Tile w1_0;
            W1Tile w1_1;
            [[cutile::hint(0, latency=MemoryLatency)]]
            a_tile = a_view.load(tile_m, kk);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_0 = w1_view.load(kk, hidden_tile0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_1 = w1_view.load(kk, hidden_tile1);
            hidden_acc0 = ct::mma(a_tile, w1_0, hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_1, hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_erf_poly9_l30(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_erf_poly9_l30(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        W2Tile w2_00;
        W2Tile w2_10;
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_00 = w2_view.load(hidden_tile0, 0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_10 = w2_view.load(hidden_tile1, 0);
        out_acc = ct::mma(hidden_bf16_0, w2_00, out_acc);
        out_acc = ct::mma(hidden_bf16_1, w2_10, out_acc);
    }

    auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc + out_bias0), tile_m, 0);

    out_acc = ct::full<OutAccTile>(0.0f);
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            ATile a_tile;
            W1Tile w1_0;
            W1Tile w1_1;
            [[cutile::hint(0, latency=MemoryLatency)]]
            a_tile = a_view.load(tile_m, kk);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_0 = w1_view.load(kk, hidden_tile0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_1 = w1_view.load(kk, hidden_tile1);
            hidden_acc0 = ct::mma(a_tile, w1_0, hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_1, hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_erf_poly9_l30(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_erf_poly9_l30(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        W2Tile w2_01;
        W2Tile w2_11;
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_01 = w2_view.load(hidden_tile0, 1);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_11 = w2_view.load(hidden_tile1, 1);
        out_acc = ct::mma(hidden_bf16_0, w2_01, out_acc);
        out_acc = ct::mma(hidden_bf16_1, w2_11, out_acc);
    }

    auto out_bias1 = ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols));
    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc + out_bias1), tile_m, 1);
}

template <int TM, int TK, int THidden, int MemoryLatency = 2, int W2MemoryLatency = MemoryLatency>
__tile_global__ void ffn12_fused_pairh32_a_persist_lat2_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    static_assert(kIn == 4 * TK);
    constexpr int OutHalf = kOut / 2;
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutHalf>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using W1Tile = ct::tile<__nv_bfloat16, ct::shape<TK, THidden>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<THidden, OutHalf>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, OutHalf>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<THidden, OutHalf>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, OutHalf>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    ATile a0;
    ATile a1;
    ATile a2;
    ATile a3;
    [[cutile::hint(0, latency=MemoryLatency)]]
    a0 = a_view.load(tile_m, 0);
    [[cutile::hint(0, latency=MemoryLatency)]]
    a1 = a_view.load(tile_m, 1);
    [[cutile::hint(0, latency=MemoryLatency)]]
    a2 = a_view.load(tile_m, 2);
    [[cutile::hint(0, latency=MemoryLatency)]]
    a3 = a_view.load(tile_m, 3);

    auto out_acc0 = ct::full<OutAccTile>(0.0f);
    auto out_acc1 = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);

        W1Tile w1_0;
        W1Tile w1_1;
        [[cutile::hint(0, latency=MemoryLatency)]]
        w1_0 = w1_view.load(0, hidden_tile0);
        [[cutile::hint(0, latency=MemoryLatency)]]
        w1_1 = w1_view.load(0, hidden_tile1);
        hidden_acc0 = ct::mma(a0, w1_0, hidden_acc0);
        hidden_acc1 = ct::mma(a0, w1_1, hidden_acc1);

        [[cutile::hint(0, latency=MemoryLatency)]]
        w1_0 = w1_view.load(1, hidden_tile0);
        [[cutile::hint(0, latency=MemoryLatency)]]
        w1_1 = w1_view.load(1, hidden_tile1);
        hidden_acc0 = ct::mma(a1, w1_0, hidden_acc0);
        hidden_acc1 = ct::mma(a1, w1_1, hidden_acc1);

        [[cutile::hint(0, latency=MemoryLatency)]]
        w1_0 = w1_view.load(2, hidden_tile0);
        [[cutile::hint(0, latency=MemoryLatency)]]
        w1_1 = w1_view.load(2, hidden_tile1);
        hidden_acc0 = ct::mma(a2, w1_0, hidden_acc0);
        hidden_acc1 = ct::mma(a2, w1_1, hidden_acc1);

        [[cutile::hint(0, latency=MemoryLatency)]]
        w1_0 = w1_view.load(3, hidden_tile0);
        [[cutile::hint(0, latency=MemoryLatency)]]
        w1_1 = w1_view.load(3, hidden_tile1);
        hidden_acc0 = ct::mma(a3, w1_0, hidden_acc0);
        hidden_acc1 = ct::mma(a3, w1_1, hidden_acc1);

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_erf_poly9_l30(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_erf_poly9_l30(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        W2Tile w2_00;
        W2Tile w2_01;
        W2Tile w2_10;
        W2Tile w2_11;
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_00 = w2_view.load(hidden_tile0, 0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_01 = w2_view.load(hidden_tile0, 1);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_10 = w2_view.load(hidden_tile1, 0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_11 = w2_view.load(hidden_tile1, 1);
        out_acc0 = ct::mma(hidden_bf16_0, w2_00, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_0, w2_01, out_acc1);
        out_acc0 = ct::mma(hidden_bf16_1, w2_10, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_1, w2_11, out_acc1);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % OutHalf;
    auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto out_bias1 = ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols));
    auto value0 = out_acc0 + out_bias0;
    auto value1 = out_acc1 + out_bias1;
    out_view.store(ct::element_cast<__nv_bfloat16>(value0), tile_m, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(value1), tile_m, 1);
}

template <int TM,
          int TK,
          int THidden,
          int MemoryLatency = 2,
          int W2MemoryLatency = MemoryLatency,
          bool Odd5 = false>
__tile_global__ void ffn12_fused_pairh32_bias_broadcast_lat2_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutHalf>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using W1Tile = ct::tile<__nv_bfloat16, ct::shape<TK, THidden>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<THidden, OutHalf>>;
    using I64HiddenBiasTile = ct::tile<long long, ct::shape<1, THidden>>;
    using I64OutBiasTile = ct::tile<long long, ct::shape<1, OutHalf>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<THidden, OutHalf>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, OutHalf>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    auto out_acc0 = ct::full<OutAccTile>(0.0f);
    auto out_acc1 = ct::full<OutAccTile>(0.0f);
    I64HiddenBiasTile hidden_local = ct::iota<I64HiddenBiasTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            ATile a_tile;
            W1Tile w1_0;
            W1Tile w1_1;
            [[cutile::hint(0, latency=MemoryLatency)]]
            a_tile = a_view.load(tile_m, kk);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_0 = w1_view.load(kk, hidden_tile0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_1 = w1_view.load(kk, hidden_tile1);
            hidden_acc0 = ct::mma(a_tile, w1_0, hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_1, hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + hidden_local;
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + hidden_local;
        auto hidden_bias0 = ct::broadcast(
            ct::element_cast<float>(ct::load(b1 + hidden_cols0)),
            ct::shape<TM, THidden>{});
        auto hidden_bias1 = ct::broadcast(
            ct::element_cast<float>(ct::load(b1 + hidden_cols1)),
            ct::shape<TM, THidden>{});
        auto hidden_value0 = gelu_selected<Odd5>(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_selected<Odd5>(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        W2Tile w2_00;
        W2Tile w2_01;
        W2Tile w2_10;
        W2Tile w2_11;
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_00 = w2_view.load(hidden_tile0, 0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_01 = w2_view.load(hidden_tile0, 1);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_10 = w2_view.load(hidden_tile1, 0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_11 = w2_view.load(hidden_tile1, 1);
        out_acc0 = ct::mma(hidden_bf16_0, w2_00, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_0, w2_01, out_acc1);
        out_acc0 = ct::mma(hidden_bf16_1, w2_10, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_1, w2_11, out_acc1);
    }

    I64OutBiasTile out_local = ct::iota<I64OutBiasTile>();
    auto out_cols = out_local;
    auto out_bias0 = ct::broadcast(
        ct::element_cast<float>(ct::load(b2 + out_cols)),
        ct::shape<TM, OutHalf>{});
    auto out_bias1 = ct::broadcast(
        ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols)),
        ct::shape<TM, OutHalf>{});
    auto value0 = out_acc0 + out_bias0;
    auto value1 = out_acc1 + out_bias1;
    out_view.store(ct::element_cast<__nv_bfloat16>(value0), tile_m, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(value1), tile_m, 1);
}

template <int TM,
          int TK,
          int THidden,
          int MemoryLatency = 2,
          int W2MemoryLatency = MemoryLatency,
          bool Odd5 = false>
__tile_global__ void ffn12_fused_pairh32_bias_broadcast_w2byhidden_lat2_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutHalf>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using W1Tile = ct::tile<__nv_bfloat16, ct::shape<TK, THidden>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<THidden, OutHalf>>;
    using I64HiddenBiasTile = ct::tile<long long, ct::shape<1, THidden>>;
    using I64OutBiasTile = ct::tile<long long, ct::shape<1, OutHalf>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<THidden, OutHalf>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, OutHalf>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    auto out_acc0 = ct::full<OutAccTile>(0.0f);
    auto out_acc1 = ct::full<OutAccTile>(0.0f);
    I64HiddenBiasTile hidden_local = ct::iota<I64HiddenBiasTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            ATile a_tile;
            W1Tile w1_0;
            W1Tile w1_1;
            [[cutile::hint(0, latency=MemoryLatency)]]
            a_tile = a_view.load(tile_m, kk);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_0 = w1_view.load(kk, hidden_tile0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_1 = w1_view.load(kk, hidden_tile1);
            hidden_acc0 = ct::mma(a_tile, w1_0, hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_1, hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + hidden_local;
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + hidden_local;
        auto hidden_bias0 = ct::broadcast(
            ct::element_cast<float>(ct::load(b1 + hidden_cols0)),
            ct::shape<TM, THidden>{});
        auto hidden_bias1 = ct::broadcast(
            ct::element_cast<float>(ct::load(b1 + hidden_cols1)),
            ct::shape<TM, THidden>{});
        auto hidden_value0 = gelu_selected<Odd5>(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_selected<Odd5>(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);

        {
            W2Tile w2_00;
            W2Tile w2_01;
            [[cutile::hint(0, latency=W2MemoryLatency)]]
            w2_00 = w2_view.load(hidden_tile0, 0);
            [[cutile::hint(0, latency=W2MemoryLatency)]]
            w2_01 = w2_view.load(hidden_tile0, 1);
            out_acc0 = ct::mma(hidden_bf16_0, w2_00, out_acc0);
            out_acc1 = ct::mma(hidden_bf16_0, w2_01, out_acc1);
        }
        {
            W2Tile w2_10;
            W2Tile w2_11;
            [[cutile::hint(0, latency=W2MemoryLatency)]]
            w2_10 = w2_view.load(hidden_tile1, 0);
            [[cutile::hint(0, latency=W2MemoryLatency)]]
            w2_11 = w2_view.load(hidden_tile1, 1);
            out_acc0 = ct::mma(hidden_bf16_1, w2_10, out_acc0);
            out_acc1 = ct::mma(hidden_bf16_1, w2_11, out_acc1);
        }
    }

    I64OutBiasTile out_local = ct::iota<I64OutBiasTile>();
    auto out_cols = out_local;
    auto out_bias0 = ct::broadcast(
        ct::element_cast<float>(ct::load(b2 + out_cols)),
        ct::shape<TM, OutHalf>{});
    auto out_bias1 = ct::broadcast(
        ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols)),
        ct::shape<TM, OutHalf>{});
    auto value0 = out_acc0 + out_bias0;
    auto value1 = out_acc1 + out_bias1;
    out_view.store(ct::element_cast<__nv_bfloat16>(value0), tile_m, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(value1), tile_m, 1);
}

template <int TM, int TK, int THidden, int TN, int MemoryLatency = 2, int W2MemoryLatency = MemoryLatency>
__tile_global__ void ffn12_fused_pairh32_tn64_y2_lat2_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    static_assert(kOut % (2 * TN) == 0);
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using W1Tile = ct::tile<__nv_bfloat16, ct::shape<TK, THidden>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<THidden, TN>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<THidden, TN>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_y, tile_z] = ct::bid();
    (void)tile_z;
    auto out_tile0 = tile_y * 2;
    auto out_tile1 = out_tile0 + 1;

    auto out_acc0 = ct::full<OutAccTile>(0.0f);
    auto out_acc1 = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            ATile a_tile;
            W1Tile w1_0;
            W1Tile w1_1;
            [[cutile::hint(0, latency=MemoryLatency)]]
            a_tile = a_view.load(tile_m, kk);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_0 = w1_view.load(kk, hidden_tile0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_1 = w1_view.load(kk, hidden_tile1);
            hidden_acc0 = ct::mma(a_tile, w1_0, hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_1, hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_erf_poly9_l30(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_erf_poly9_l30(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        W2Tile w2_00;
        W2Tile w2_01;
        W2Tile w2_10;
        W2Tile w2_11;
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_00 = w2_view.load(hidden_tile0, out_tile0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_01 = w2_view.load(hidden_tile0, out_tile1);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_10 = w2_view.load(hidden_tile1, out_tile0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_11 = w2_view.load(hidden_tile1, out_tile1);
        out_acc0 = ct::mma(hidden_bf16_0, w2_00, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_0, w2_01, out_acc1);
        out_acc0 = ct::mma(hidden_bf16_1, w2_10, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_1, w2_11, out_acc1);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols0 = static_cast<long long>(out_tile0) * TN + (out_local % TN);
    auto out_cols1 = static_cast<long long>(out_tile1) * TN + (out_local % TN);
    auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols0));
    auto out_bias1 = ct::element_cast<float>(ct::load(b2 + out_cols1));
    auto value0 = out_acc0 + out_bias0;
    auto value1 = out_acc1 + out_bias1;
    out_view.store(ct::element_cast<__nv_bfloat16>(value0), tile_m, out_tile0);
    out_view.store(ct::element_cast<__nv_bfloat16>(value1), tile_m, out_tile1);
}

template <int TM, int TK, int THidden, int TN, int MemoryLatency = 2, int W2MemoryLatency = MemoryLatency>
__tile_global__ void ffn12_fused_pairh32_tn64x4_lat2_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    static_assert(kOut == 4 * TN);
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using W1Tile = ct::tile<__nv_bfloat16, ct::shape<TK, THidden>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<THidden, TN>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<THidden, TN>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    auto out_acc0 = ct::full<OutAccTile>(0.0f);
    auto out_acc1 = ct::full<OutAccTile>(0.0f);
    auto out_acc2 = ct::full<OutAccTile>(0.0f);
    auto out_acc3 = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            ATile a_tile;
            W1Tile w1_0;
            W1Tile w1_1;
            [[cutile::hint(0, latency=MemoryLatency)]]
            a_tile = a_view.load(tile_m, kk);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_0 = w1_view.load(kk, hidden_tile0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_1 = w1_view.load(kk, hidden_tile1);
            hidden_acc0 = ct::mma(a_tile, w1_0, hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_1, hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_erf_poly9_l30(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_erf_poly9_l30(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        W2Tile w2_00;
        W2Tile w2_01;
        W2Tile w2_02;
        W2Tile w2_03;
        W2Tile w2_10;
        W2Tile w2_11;
        W2Tile w2_12;
        W2Tile w2_13;
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_00 = w2_view.load(hidden_tile0, 0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_01 = w2_view.load(hidden_tile0, 1);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_02 = w2_view.load(hidden_tile0, 2);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_03 = w2_view.load(hidden_tile0, 3);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_10 = w2_view.load(hidden_tile1, 0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_11 = w2_view.load(hidden_tile1, 1);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_12 = w2_view.load(hidden_tile1, 2);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_13 = w2_view.load(hidden_tile1, 3);
        out_acc0 = ct::mma(hidden_bf16_0, w2_00, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_0, w2_01, out_acc1);
        out_acc2 = ct::mma(hidden_bf16_0, w2_02, out_acc2);
        out_acc3 = ct::mma(hidden_bf16_0, w2_03, out_acc3);
        out_acc0 = ct::mma(hidden_bf16_1, w2_10, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_1, w2_11, out_acc1);
        out_acc2 = ct::mma(hidden_bf16_1, w2_12, out_acc2);
        out_acc3 = ct::mma(hidden_bf16_1, w2_13, out_acc3);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % TN;
    auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto out_bias1 = ct::element_cast<float>(ct::load(b2 + TN + out_cols));
    auto out_bias2 = ct::element_cast<float>(ct::load(b2 + 2 * TN + out_cols));
    auto out_bias3 = ct::element_cast<float>(ct::load(b2 + 3 * TN + out_cols));
    auto value0 = out_acc0 + out_bias0;
    auto value1 = out_acc1 + out_bias1;
    auto value2 = out_acc2 + out_bias2;
    auto value3 = out_acc3 + out_bias3;
    out_view.store(ct::element_cast<__nv_bfloat16>(value0), tile_m, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(value1), tile_m, 1);
    out_view.store(ct::element_cast<__nv_bfloat16>(value2), tile_m, 2);
    out_view.store(ct::element_cast<__nv_bfloat16>(value3), tile_m, 3);
}

template <int TM, int TK, int THidden, int MemoryLatency = 2, int W2MemoryLatency = MemoryLatency>
__tile_global__ void ffn12_fused_pairh32_out256_lat2_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, kOut>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using W1Tile = ct::tile<__nv_bfloat16, ct::shape<TK, THidden>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<THidden, kOut>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, kOut>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<THidden, kOut>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, kOut>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    auto out_acc = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            ATile a_tile;
            W1Tile w1_0;
            W1Tile w1_1;
            [[cutile::hint(0, latency=MemoryLatency)]]
            a_tile = a_view.load(tile_m, kk);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_0 = w1_view.load(kk, hidden_tile0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_1 = w1_view.load(kk, hidden_tile1);
            hidden_acc0 = ct::mma(a_tile, w1_0, hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_1, hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_erf_poly9_l30(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_erf_poly9_l30(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        W2Tile w2_0;
        W2Tile w2_1;
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_0 = w2_view.load(hidden_tile0, 0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_1 = w2_view.load(hidden_tile1, 0);
        out_acc = ct::mma(hidden_bf16_0, w2_0, out_acc);
        out_acc = ct::mma(hidden_bf16_1, w2_1, out_acc);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % kOut;
    auto out_bias = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto value = out_acc + out_bias;
    out_view.store(ct::element_cast<__nv_bfloat16>(value), tile_m, 0);
}

template <int TM, int TK, int THidden, int MemoryLatency = 2, int W2MemoryLatency = MemoryLatency>
__tile_global__ void ffn12_fused_pairh32_pair2_lat2_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    static_assert(kHidden % (4 * THidden) == 0);
    constexpr int OutHalf = kOut / 2;
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutHalf>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using W1Tile = ct::tile<__nv_bfloat16, ct::shape<TK, THidden>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<THidden, OutHalf>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, OutHalf>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<THidden, OutHalf>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, OutHalf>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    auto out_acc0 = ct::full<OutAccTile>(0.0f);
    auto out_acc1 = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_quad : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (4 * THidden)})) {
        auto hidden_tile0 = hidden_quad * 4;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_tile2 = hidden_tile0 + 2;
        auto hidden_tile3 = hidden_tile0 + 3;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc2 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc3 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            ATile a_tile;
            W1Tile w1_0;
            W1Tile w1_1;
            W1Tile w1_2;
            W1Tile w1_3;
            [[cutile::hint(0, latency=MemoryLatency)]]
            a_tile = a_view.load(tile_m, kk);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_0 = w1_view.load(kk, hidden_tile0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_1 = w1_view.load(kk, hidden_tile1);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_2 = w1_view.load(kk, hidden_tile2);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_3 = w1_view.load(kk, hidden_tile3);
            hidden_acc0 = ct::mma(a_tile, w1_0, hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_1, hidden_acc1);
            hidden_acc2 = ct::mma(a_tile, w1_2, hidden_acc2);
            hidden_acc3 = ct::mma(a_tile, w1_3, hidden_acc3);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_cols2 =
            static_cast<long long>(hidden_tile2) * THidden + (hidden_local % THidden);
        auto hidden_cols3 =
            static_cast<long long>(hidden_tile3) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_bias2 = ct::element_cast<float>(ct::load(b1 + hidden_cols2));
        auto hidden_bias3 = ct::element_cast<float>(ct::load(b1 + hidden_cols3));
        auto hidden_value0 = gelu_erf_poly9_l30(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_erf_poly9_l30(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_value2 = gelu_erf_poly9_l30(bf16_round(hidden_acc2) + hidden_bias2);
        auto hidden_value3 = gelu_erf_poly9_l30(bf16_round(hidden_acc3) + hidden_bias3);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);
        auto hidden_bf16_2 = ct::element_cast<__nv_bfloat16>(hidden_value2);
        auto hidden_bf16_3 = ct::element_cast<__nv_bfloat16>(hidden_value3);
        W2Tile w2_00;
        W2Tile w2_01;
        W2Tile w2_10;
        W2Tile w2_11;
        W2Tile w2_20;
        W2Tile w2_21;
        W2Tile w2_30;
        W2Tile w2_31;
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_00 = w2_view.load(hidden_tile0, 0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_01 = w2_view.load(hidden_tile0, 1);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_10 = w2_view.load(hidden_tile1, 0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_11 = w2_view.load(hidden_tile1, 1);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_20 = w2_view.load(hidden_tile2, 0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_21 = w2_view.load(hidden_tile2, 1);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_30 = w2_view.load(hidden_tile3, 0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_31 = w2_view.load(hidden_tile3, 1);
        out_acc0 = ct::mma(hidden_bf16_0, w2_00, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_0, w2_01, out_acc1);
        out_acc0 = ct::mma(hidden_bf16_1, w2_10, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_1, w2_11, out_acc1);
        out_acc0 = ct::mma(hidden_bf16_2, w2_20, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_2, w2_21, out_acc1);
        out_acc0 = ct::mma(hidden_bf16_3, w2_30, out_acc0);
        out_acc1 = ct::mma(hidden_bf16_3, w2_31, out_acc1);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % OutHalf;
    auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto out_bias1 = ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols));
    auto value0 = out_acc0 + out_bias0;
    auto value1 = out_acc1 + out_bias1;
    out_view.store(ct::element_cast<__nv_bfloat16>(value0), tile_m, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(value1), tile_m, 1);
}

template <int TM, int TK, int THidden, int MemoryLatency = 2, int W2MemoryLatency = MemoryLatency>
__tile_global__ void ffn12_fused_pairh32_w2stream_lat2_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutHalf>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using W1Tile = ct::tile<__nv_bfloat16, ct::shape<TK, THidden>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<THidden, OutHalf>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, OutHalf>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<THidden, OutHalf>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, OutHalf>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    auto out_acc0 = ct::full<OutAccTile>(0.0f);
    auto out_acc1 = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            ATile a_tile;
            W1Tile w1_0;
            W1Tile w1_1;
            [[cutile::hint(0, latency=MemoryLatency)]]
            a_tile = a_view.load(tile_m, kk);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_0 = w1_view.load(kk, hidden_tile0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_1 = w1_view.load(kk, hidden_tile1);
            hidden_acc0 = ct::mma(a_tile, w1_0, hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_1, hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_erf_poly9_l30(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_erf_poly9_l30(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);

        W2Tile w2_tile;
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_tile = w2_view.load(hidden_tile0, 0);
        out_acc0 = ct::mma(hidden_bf16_0, w2_tile, out_acc0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_tile = w2_view.load(hidden_tile0, 1);
        out_acc1 = ct::mma(hidden_bf16_0, w2_tile, out_acc1);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_tile = w2_view.load(hidden_tile1, 0);
        out_acc0 = ct::mma(hidden_bf16_1, w2_tile, out_acc0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_tile = w2_view.load(hidden_tile1, 1);
        out_acc1 = ct::mma(hidden_bf16_1, w2_tile, out_acc1);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % OutHalf;
    auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto out_bias1 = ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols));
    auto value0 = out_acc0 + out_bias0;
    auto value1 = out_acc1 + out_bias1;
    out_view.store(ct::element_cast<__nv_bfloat16>(value0), tile_m, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(value1), tile_m, 1);
}

template <int TM, int TK, int THidden, int MemoryLatency = 2, int W2MemoryLatency = MemoryLatency>
__tile_global__ void ffn12_fused_pairh32_w2pair_lat2_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    using HiddenAccTile = ct::tile<float, ct::shape<TM, THidden>>;
    using OutAccTile = ct::tile<float, ct::shape<TM, OutHalf>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using W1Tile = ct::tile<__nv_bfloat16, ct::shape<TK, THidden>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<THidden, OutHalf>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<TM, THidden>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, OutHalf>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<TM, TK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<TK, THidden>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<THidden, OutHalf>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<TM, OutHalf>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    auto out_acc0 = ct::full<OutAccTile>(0.0f);
    auto out_acc1 = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * THidden)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / TK})) {
            ATile a_tile;
            W1Tile w1_0;
            W1Tile w1_1;
            [[cutile::hint(0, latency=MemoryLatency)]]
            a_tile = a_view.load(tile_m, kk);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_0 = w1_view.load(kk, hidden_tile0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w1_1 = w1_view.load(kk, hidden_tile1);
            hidden_acc0 = ct::mma(a_tile, w1_0, hidden_acc0);
            hidden_acc1 = ct::mma(a_tile, w1_1, hidden_acc1);
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * THidden + (hidden_local % THidden);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * THidden + (hidden_local % THidden);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_erf_poly9_l30(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_erf_poly9_l30(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);

        W2Tile w2_00;
        W2Tile w2_10;
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_00 = w2_view.load(hidden_tile0, 0);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_10 = w2_view.load(hidden_tile1, 0);
        out_acc0 = ct::mma(hidden_bf16_0, w2_00, out_acc0);
        out_acc0 = ct::mma(hidden_bf16_1, w2_10, out_acc0);

        W2Tile w2_01;
        W2Tile w2_11;
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_01 = w2_view.load(hidden_tile0, 1);
        [[cutile::hint(0, latency=W2MemoryLatency)]]
        w2_11 = w2_view.load(hidden_tile1, 1);
        out_acc1 = ct::mma(hidden_bf16_0, w2_01, out_acc1);
        out_acc1 = ct::mma(hidden_bf16_1, w2_11, out_acc1);
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % OutHalf;
    auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto out_bias1 = ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols));
    auto value0 = out_acc0 + out_bias0;
    auto value1 = out_acc1 + out_bias1;
    out_view.store(ct::element_cast<__nv_bfloat16>(value0), tile_m, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(value1), tile_m, 1);
}

template <int Ffn2TN>
void launch_two_stage_ffn2_tn(const __nv_bfloat16* d_a,
                              const __nv_bfloat16* d_w1,
                              const __nv_bfloat16* d_b1,
                              const __nv_bfloat16* d_w2,
                              const __nv_bfloat16* d_b2,
                              __nv_bfloat16* d_hidden,
                              __nv_bfloat16* d_out) {
    dim3 ffn1_grid(kM / 32, kHidden / 64, 1);
    ffn1_poly9_tk64_kernel<32, 64, 64>
        <<<ffn1_grid, 1>>>(d_a, d_w1, d_b1, d_hidden);
    dim3 ffn2_grid(kM / 32, kOut / Ffn2TN, 1);
    ffn2_tn64_kernel<32, Ffn2TN, 64>
        <<<ffn2_grid, 1>>>(d_hidden, d_w2, d_b2, d_out);
}

template <int TM, int TK, int THidden, int MemoryLatency, int W2MemoryLatency>
void launch_fused_shape_latency(const __nv_bfloat16* d_a,
                                const __nv_bfloat16* d_w1,
                                const __nv_bfloat16* d_b1,
                                const __nv_bfloat16* d_w2,
                                const __nv_bfloat16* d_b2,
                                __nv_bfloat16* d_out) {
    static_assert(kM % TM == 0);
    static_assert(kIn % TK == 0);
    static_assert(kHidden % (2 * THidden) == 0);
    dim3 grid(kM / TM, 1, 1);
    ffn12_fused_pairh32_lat2_kernel<TM, TK, THidden, MemoryLatency, W2MemoryLatency>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

template <int MemoryLatency, int W2MemoryLatency = MemoryLatency>
void launch_fused_latency(const __nv_bfloat16* d_a,
                          const __nv_bfloat16* d_w1,
                          const __nv_bfloat16* d_b1,
                          const __nv_bfloat16* d_w2,
                          const __nv_bfloat16* d_b2,
                          __nv_bfloat16* d_out) {
    launch_fused_shape_latency<32, 64, 32, MemoryLatency, W2MemoryLatency>(
        d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

void launch_fused(const __nv_bfloat16* d_a,
                  const __nv_bfloat16* d_w1,
                  const __nv_bfloat16* d_b1,
                  const __nv_bfloat16* d_w2,
                  const __nv_bfloat16* d_b2,
                  __nv_bfloat16* d_out) {
    launch_fused_latency<2, 2>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

void launch_fused_bias_broadcast(const __nv_bfloat16* d_a,
                                 const __nv_bfloat16* d_w1,
                                 const __nv_bfloat16* d_b1,
                                 const __nv_bfloat16* d_w2,
                                 const __nv_bfloat16* d_b2,
                                 __nv_bfloat16* d_out) {
    dim3 grid(kM / 32, 1, 1);
    ffn12_fused_pairh32_bias_broadcast_lat2_kernel<32, 64, 32, 2, 2>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

template <int TM, int MemoryLatency, int W2MemoryLatency>
void launch_fused_bias_broadcast_odd5_latency(const __nv_bfloat16* d_a,
                                              const __nv_bfloat16* d_w1,
                                              const __nv_bfloat16* d_b1,
                                              const __nv_bfloat16* d_w2,
                                              const __nv_bfloat16* d_b2,
                                              __nv_bfloat16* d_out) {
    dim3 grid(kM / TM, 1, 1);
    ffn12_fused_pairh32_bias_broadcast_lat2_kernel<TM,
                                                   64,
                                                   32,
                                                   MemoryLatency,
                                                   W2MemoryLatency,
                                                   true>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

void launch_fused_bias_broadcast_odd5(const __nv_bfloat16* d_a,
                                      const __nv_bfloat16* d_w1,
                                      const __nv_bfloat16* d_b1,
                                      const __nv_bfloat16* d_w2,
                                      const __nv_bfloat16* d_b2,
                                      __nv_bfloat16* d_out) {
    launch_fused_bias_broadcast_odd5_latency<32, 2, 2>(
        d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

void launch_fused_bias_broadcast_odd5_w2byhidden(const __nv_bfloat16* d_a,
                                                 const __nv_bfloat16* d_w1,
                                                 const __nv_bfloat16* d_b1,
                                                 const __nv_bfloat16* d_w2,
                                                 const __nv_bfloat16* d_b2,
                                                 __nv_bfloat16* d_out) {
    dim3 grid(kM / 32, 1, 1);
    ffn12_fused_pairh32_bias_broadcast_w2byhidden_lat2_kernel<32,
                                                              64,
                                                              32,
                                                              2,
                                                              2,
                                                              true>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

void launch_fused_tn64_y2(const __nv_bfloat16* d_a,
                          const __nv_bfloat16* d_w1,
                          const __nv_bfloat16* d_b1,
                          const __nv_bfloat16* d_w2,
                          const __nv_bfloat16* d_b2,
                          __nv_bfloat16* d_out) {
    dim3 grid(kM / 32, kOut / (2 * 64), 1);
    ffn12_fused_pairh32_tn64_y2_lat2_kernel<32, 64, 32, 64, 2, 2>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

void launch_fused_tn64x4(const __nv_bfloat16* d_a,
                         const __nv_bfloat16* d_w1,
                         const __nv_bfloat16* d_b1,
                         const __nv_bfloat16* d_w2,
                         const __nv_bfloat16* d_b2,
                         __nv_bfloat16* d_out) {
    dim3 grid(kM / 32, 1, 1);
    ffn12_fused_pairh32_tn64x4_lat2_kernel<32, 64, 32, 64, 2, 2>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

void launch_fused_out256(const __nv_bfloat16* d_a,
                         const __nv_bfloat16* d_w1,
                         const __nv_bfloat16* d_b1,
                         const __nv_bfloat16* d_w2,
                         const __nv_bfloat16* d_b2,
                         __nv_bfloat16* d_out) {
    dim3 grid(kM / 32, 1, 1);
    ffn12_fused_pairh32_out256_lat2_kernel<32, 64, 32, 2, 2>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

void launch_fused_pair2(const __nv_bfloat16* d_a,
                        const __nv_bfloat16* d_w1,
                        const __nv_bfloat16* d_b1,
                        const __nv_bfloat16* d_w2,
                        const __nv_bfloat16* d_b2,
                        __nv_bfloat16* d_out) {
    dim3 grid(kM / 32, 1, 1);
    ffn12_fused_pairh32_pair2_lat2_kernel<32, 64, 32, 2, 2>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

void launch_fused_w2stream(const __nv_bfloat16* d_a,
                           const __nv_bfloat16* d_w1,
                           const __nv_bfloat16* d_b1,
                           const __nv_bfloat16* d_w2,
                           const __nv_bfloat16* d_b2,
                           __nv_bfloat16* d_out) {
    dim3 grid(kM / 32, 1, 1);
    ffn12_fused_pairh32_w2stream_lat2_kernel<32, 64, 32, 2, 2>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

void launch_fused_w2pair(const __nv_bfloat16* d_a,
                         const __nv_bfloat16* d_w1,
                         const __nv_bfloat16* d_b1,
                         const __nv_bfloat16* d_w2,
                         const __nv_bfloat16* d_b2,
                         __nv_bfloat16* d_out) {
    dim3 grid(kM / 32, 1, 1);
    ffn12_fused_pairh32_w2pair_lat2_kernel<32, 64, 32, 2, 2>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

void launch_fused_a_persist(const __nv_bfloat16* d_a,
                            const __nv_bfloat16* d_w1,
                            const __nv_bfloat16* d_b1,
                            const __nv_bfloat16* d_w2,
                            const __nv_bfloat16* d_b2,
                            __nv_bfloat16* d_out) {
    dim3 grid(kM / 32, 1, 1);
    ffn12_fused_pairh32_a_persist_lat2_kernel<32, 64, 32, 2, 2>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

void launch_fused_serial_out_halves(const __nv_bfloat16* d_a,
                                    const __nv_bfloat16* d_w1,
                                    const __nv_bfloat16* d_b1,
                                    const __nv_bfloat16* d_w2,
                                    const __nv_bfloat16* d_b2,
                                    __nv_bfloat16* d_out) {
    dim3 grid(kM / 32, 1, 1);
    ffn12_fused_pairh32_serial_out_halves_lat2_kernel<32, 64, 32, 2, 2>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
}

bool should_run(const Options& opts, const char* name) {
    return opts.variant == "all" || opts.variant == name;
}

template <typename Kernel>
void describe_kernel(const Options& opts, const char* name, Kernel kernel, dim3 grid) {
    if (!should_run(opts, name)) return;

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
        "%-44s grid=(%u,%u,%u) waves/SM=%.1f attr_regs=%d "
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
                    "fused_pairh32_lat2",
                    ffn12_fused_pairh32_lat2_kernel<32, 64, 32, 2, 2>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_tm16_pairh32_lat2_outnoround",
                    ffn12_fused_pairh32_lat2_kernel<16, 64, 32, 2, 2>,
                    dim3(kM / 16, 1, 1));
    describe_kernel(opts,
                    "fused_pairh16_lat2_outnoround",
                    ffn12_fused_pairh32_lat2_kernel<32, 64, 16, 2, 2>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh16_tk32_lat2_outnoround",
                    ffn12_fused_pairh32_lat2_kernel<32, 32, 16, 2, 2>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_tk32_lat2_outnoround",
                    ffn12_fused_pairh32_lat2_kernel<32, 32, 32, 2, 2>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_tk128_lat2_outnoround",
                    ffn12_fused_pairh32_lat2_kernel<32, 128, 32, 2, 2>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_lat1_outnoround",
                    ffn12_fused_pairh32_lat2_kernel<32, 64, 32, 1, 1>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_w1lat1_w2lat2_outnoround",
                    ffn12_fused_pairh32_lat2_kernel<32, 64, 32, 1, 2>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_w1lat2_w2lat1_outnoround",
                    ffn12_fused_pairh32_lat2_kernel<32, 64, 32, 2, 1>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_lat3_outnoround",
                    ffn12_fused_pairh32_lat2_kernel<32, 64, 32, 3, 3>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_a_persist_lat2_outnoround",
                    ffn12_fused_pairh32_a_persist_lat2_kernel<32, 64, 32, 2, 2>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_serial_out_halves_lat2_outnoround",
                    ffn12_fused_pairh32_serial_out_halves_lat2_kernel<32, 64, 32, 2, 2>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_tn64_y2_lat2_outnoround",
                    ffn12_fused_pairh32_tn64_y2_lat2_kernel<32, 64, 32, 64, 2, 2>,
                    dim3(kM / 32, kOut / (2 * 64), 1));
    describe_kernel(opts,
                    "fused_pairh32_tn64x4_lat2_outnoround",
                    ffn12_fused_pairh32_tn64x4_lat2_kernel<32, 64, 32, 64, 2, 2>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_out256_lat2_outnoround",
                    ffn12_fused_pairh32_out256_lat2_kernel<32, 64, 32, 2, 2>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_pair2_lat2_outnoround",
                    ffn12_fused_pairh32_pair2_lat2_kernel<32, 64, 32, 2, 2>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_w2stream_lat2_outnoround",
                    ffn12_fused_pairh32_w2stream_lat2_kernel<32, 64, 32, 2, 2>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_w2pair_lat2_outnoround",
                    ffn12_fused_pairh32_w2pair_lat2_kernel<32, 64, 32, 2, 2>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_bias_broadcast_lat2",
                    ffn12_fused_pairh32_bias_broadcast_lat2_kernel<32, 64, 32, 2, 2>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_bias_broadcast_odd5_lat2",
                    ffn12_fused_pairh32_bias_broadcast_lat2_kernel<32, 64, 32, 2, 2, true>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_bias_broadcast_odd5_w2byhidden_lat2",
                    ffn12_fused_pairh32_bias_broadcast_w2byhidden_lat2_kernel<
                        32, 64, 32, 2, 2, true>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_tm16_pairh32_bias_broadcast_odd5_lat2",
                    ffn12_fused_pairh32_bias_broadcast_lat2_kernel<16, 64, 32, 2, 2, true>,
                    dim3(kM / 16, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_bias_broadcast_odd5_lat1",
                    ffn12_fused_pairh32_bias_broadcast_lat2_kernel<32, 64, 32, 1, 1, true>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_bias_broadcast_odd5_w1lat1_w2lat2",
                    ffn12_fused_pairh32_bias_broadcast_lat2_kernel<32, 64, 32, 1, 2, true>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_bias_broadcast_odd5_w1lat2_w2lat1",
                    ffn12_fused_pairh32_bias_broadcast_lat2_kernel<32, 64, 32, 2, 1, true>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "fused_pairh32_bias_broadcast_odd5_lat3",
                    ffn12_fused_pairh32_bias_broadcast_lat2_kernel<32, 64, 32, 3, 3, true>,
                    dim3(kM / 32, 1, 1));
    describe_kernel(opts,
                    "two_stage_poly9_tk64_ffn1",
                    ffn1_poly9_tk64_kernel<32, 64, 64>,
                    dim3(kM / 32, kHidden / 64, 1));
    describe_kernel(opts,
                    "two_stage_poly9_tk64_ffn2_tn64",
                    ffn2_tn64_kernel<32, 64, 64>,
                    dim3(kM / 32, kOut / 64, 1));
    describe_kernel(opts,
                    "two_stage_poly9_tk64_ffn2_tn32",
                    ffn2_tn64_kernel<32, 32, 64>,
                    dim3(kM / 32, kOut / 32, 1));
    describe_kernel(opts,
                    "two_stage_poly9_tk64_ffn2_tn128",
                    ffn2_tn64_kernel<32, 128, 64>,
                    dim3(kM / 32, kOut / 128, 1));
}

template <typename Launch>
void run_variant(const char* name,
                 int launches,
                 double hidden_rw_gib,
                 const Options& opts,
                 __nv_bfloat16* d_out,
                 Launch launch) {
    for (int i = 0; i < opts.warmup; ++i) {
        launch();
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
        launch();
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

    double useful_flops = 2.0 * kM * kHidden * kIn + 2.0 * kM * kOut * kHidden;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double useful_tf = useful_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-20s launches=%d best=%.4f ms median=%.4f ms useful=%.2f TF/s "
        "roof=%.1f%% hidden_rw=%.3f GiB checksum=%.4f\n",
        name, launches, best_ms, median_ms, useful_tf,
        useful_tf * 100.0 / kA10gDenseBf16Tflops,
        hidden_rw_gib, __bfloat162float(checksum));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        if (opts.describe) {
            describe_all(opts);
            return 0;
        }
        size_t a_elems = static_cast<size_t>(kM) * kIn;
        size_t w1_elems = static_cast<size_t>(kHidden) * kIn;
        size_t b1_elems = kHidden;
        size_t hidden_elems = static_cast<size_t>(kM) * kHidden;
        size_t w2_elems = static_cast<size_t>(kOut) * kHidden;
        size_t b2_elems = kOut;
        size_t out_elems = static_cast<size_t>(kM) * kOut;

        __nv_bfloat16* d_a = nullptr;
        __nv_bfloat16* d_w1 = nullptr;
        __nv_bfloat16* d_b1 = nullptr;
        __nv_bfloat16* d_hidden = nullptr;
        __nv_bfloat16* d_w2 = nullptr;
        __nv_bfloat16* d_b2 = nullptr;
        __nv_bfloat16* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_w1, w1_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_b1, b1_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_hidden, hidden_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_w2, w2_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_b2, b2_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(__nv_bfloat16)));

        init_bf16(d_a, a_elems);
        init_bf16(d_w1, w1_elems);
        init_bf16(d_b1, b1_elems);
        init_bf16(d_w2, w2_elems);
        init_bf16(d_b2, b2_elems);
        CUDA_CHECK(cudaDeviceSynchronize());

        double hidden_rw_gib =
            (2.0 * static_cast<double>(hidden_elems) * sizeof(__nv_bfloat16)) /
            (1024.0 * 1024.0 * 1024.0);

        if (should_run(opts, "fused_pairh32_lat2")) {
            run_variant("fused_pairh32_lat2", 1, 0.0, opts, d_out, [&] {
                launch_fused(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_tm16_pairh32_lat2_outnoround")) {
            run_variant("fused_tm16_pairh32_lat2_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_shape_latency<16, 64, 32, 2, 2>(
                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh16_lat2_outnoround")) {
            run_variant("fused_pairh16_lat2_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_shape_latency<32, 64, 16, 2, 2>(
                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh16_tk32_lat2_outnoround")) {
            run_variant("fused_pairh16_tk32_lat2_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_shape_latency<32, 32, 16, 2, 2>(
                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_tk32_lat2_outnoround")) {
            run_variant("fused_pairh32_tk32_lat2_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_shape_latency<32, 32, 32, 2, 2>(
                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_tk128_lat2_outnoround")) {
            run_variant("fused_pairh32_tk128_lat2_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_shape_latency<32, 128, 32, 2, 2>(
                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_lat1_outnoround")) {
            run_variant("fused_pairh32_lat1_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_latency<1, 1>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_w1lat1_w2lat2_outnoround")) {
            run_variant("fused_pairh32_w1lat1_w2lat2_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_latency<1, 2>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_w1lat2_w2lat1_outnoround")) {
            run_variant("fused_pairh32_w1lat2_w2lat1_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_latency<2, 1>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_lat3_outnoround")) {
            run_variant("fused_pairh32_lat3_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_latency<3, 3>(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_a_persist_lat2_outnoround")) {
            run_variant("fused_pairh32_a_persist_lat2_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_a_persist(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_serial_out_halves_lat2_outnoround")) {
            run_variant("fused_pairh32_serial_out_halves_lat2_outnoround",
                        1,
                        0.0,
                        opts,
                        d_out,
                        [&] {
                launch_fused_serial_out_halves(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_tn64_y2_lat2_outnoround")) {
            run_variant("fused_pairh32_tn64_y2_lat2_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_tn64_y2(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_tn64x4_lat2_outnoround")) {
            run_variant("fused_pairh32_tn64x4_lat2_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_tn64x4(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_out256_lat2_outnoround")) {
            run_variant("fused_pairh32_out256_lat2_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_out256(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_pair2_lat2_outnoround")) {
            run_variant("fused_pairh32_pair2_lat2_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_pair2(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_w2stream_lat2_outnoround")) {
            run_variant("fused_pairh32_w2stream_lat2_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_w2stream(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_w2pair_lat2_outnoround")) {
            run_variant("fused_pairh32_w2pair_lat2_outnoround", 1, 0.0, opts, d_out, [&] {
                launch_fused_w2pair(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_bias_broadcast_lat2")) {
            run_variant("fused_pairh32_bias_broadcast_lat2", 1, 0.0, opts, d_out, [&] {
                launch_fused_bias_broadcast(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_bias_broadcast_odd5_lat2")) {
            run_variant("fused_pairh32_bias_broadcast_odd5_lat2", 1, 0.0, opts, d_out, [&] {
                launch_fused_bias_broadcast_odd5(d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_bias_broadcast_odd5_w2byhidden_lat2")) {
            run_variant("fused_pairh32_bias_broadcast_odd5_w2byhidden_lat2",
                        1,
                        0.0,
                        opts,
                        d_out,
                        [&] {
                launch_fused_bias_broadcast_odd5_w2byhidden(
                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_tm16_pairh32_bias_broadcast_odd5_lat2")) {
            run_variant("fused_tm16_pairh32_bias_broadcast_odd5_lat2",
                        1,
                        0.0,
                        opts,
                        d_out,
                        [&] {
                launch_fused_bias_broadcast_odd5_latency<16, 2, 2>(
                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_bias_broadcast_odd5_lat1")) {
            run_variant("fused_pairh32_bias_broadcast_odd5_lat1", 1, 0.0, opts, d_out, [&] {
                launch_fused_bias_broadcast_odd5_latency<32, 1, 1>(
                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_bias_broadcast_odd5_w1lat1_w2lat2")) {
            run_variant("fused_pairh32_bias_broadcast_odd5_w1lat1_w2lat2",
                        1,
                        0.0,
                        opts,
                        d_out,
                        [&] {
                launch_fused_bias_broadcast_odd5_latency<32, 1, 2>(
                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_bias_broadcast_odd5_w1lat2_w2lat1")) {
            run_variant("fused_pairh32_bias_broadcast_odd5_w1lat2_w2lat1",
                        1,
                        0.0,
                        opts,
                        d_out,
                        [&] {
                launch_fused_bias_broadcast_odd5_latency<32, 2, 1>(
                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "fused_pairh32_bias_broadcast_odd5_lat3")) {
            run_variant("fused_pairh32_bias_broadcast_odd5_lat3", 1, 0.0, opts, d_out, [&] {
                launch_fused_bias_broadcast_odd5_latency<32, 3, 3>(
                    d_a, d_w1, d_b1, d_w2, d_b2, d_out);
            });
        }
        if (should_run(opts, "two_stage_poly9_tk64")) {
            run_variant("two_stage_poly9_tk64", 2, hidden_rw_gib, opts, d_out, [&] {
                launch_two_stage_ffn2_tn<64>(
                    d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out);
            });
        }
        if (should_run(opts, "two_stage_poly9_tk64_ffn2_tn32")) {
            run_variant("two_stage_poly9_tk64_ffn2_tn32", 2, hidden_rw_gib, opts, d_out, [&] {
                launch_two_stage_ffn2_tn<32>(
                    d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out);
            });
        }
        if (should_run(opts, "two_stage_poly9_tk64_ffn2_tn64")) {
            run_variant("two_stage_poly9_tk64_ffn2_tn64", 2, hidden_rw_gib, opts, d_out, [&] {
                launch_two_stage_ffn2_tn<64>(
                    d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out);
            });
        }
        if (should_run(opts, "two_stage_poly9_tk64_ffn2_tn128")) {
            run_variant("two_stage_poly9_tk64_ffn2_tn128", 2, hidden_rw_gib, opts, d_out, [&] {
                launch_two_stage_ffn2_tn<128>(
                    d_a, d_w1, d_b1, d_w2, d_b2, d_hidden, d_out);
            });
        }

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_w1));
        CUDA_CHECK(cudaFree(d_b1));
        CUDA_CHECK(cudaFree(d_hidden));
        CUDA_CHECK(cudaFree(d_w2));
        CUDA_CHECK(cudaFree(d_b2));
        CUDA_CHECK(cudaFree(d_out));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
