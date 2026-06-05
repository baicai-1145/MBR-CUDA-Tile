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

constexpr int kInitTile = 256;
constexpr int kM = 78048;
constexpr int kIn = 256;
constexpr int kHidden = 1024;
constexpr int kOut = 256;
constexpr int kTileM = 32;
constexpr int kTileK = 64;
constexpr int kHiddenTile = 32;
constexpr double kA10gDenseBf16Tflops = 70.0;

using I64InitTile = ct::tile<long long, ct::shape<kInitTile>>;
using F32InitTile = ct::tile<float, ct::shape<kInitTile>>;

struct Options {
    std::string variant = "all";
    int warmup = 20;
    int iters = 300;
};

int ceildiv(long long a, int b) {
    return static_cast<int>((a + b - 1) / b);
}

int parse_positive_int(const char* name, const char* value) {
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
            opts.warmup = parse_positive_int(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            opts.iters = parse_positive_int(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: bench_bf16_ffn12_residual_cutile [options]\n"
                "  --variant NAME  all, old, outnoround_lat0,\n"
                "                  fast_lat2_outnoround, fast_lat2_outnoround_nostore,\n"
                "                  nores_old, nores_outnoround_lat0,\n"
                "                  nores_fast_lat2_outnoround, two_kernel_fast\n"
                "  --warmup N      warmup launches, default 20\n"
                "  --iters N       measured launches, default 300\n");
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

__tile_global__ void fill_bf16_kernel(__nv_bfloat16* __restrict__ dst, long long total) {
    dst = ct::assume_aligned(dst, 16_ic);
    I64InitTile idx = static_cast<long long>(ct::bid().x) * kInitTile +
                      ct::iota<I64InitTile>();
    auto in_bounds = idx < total;
    F32InitTile values = 0.25f +
                         ct::element_cast<float>((idx * 13LL) & 1023LL) *
                             0.0009765625f;
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

void init_bf16(__nv_bfloat16* dst, long long total) {
    fill_bf16_kernel<<<ceildiv(total, kInitTile), 1>>>(dst, total);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static __tile__ auto bf16_round(T value) {
    return ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(value));
}

template <bool Round, typename T>
static __tile__ auto bf16_round_if(T value) {
    if constexpr (Round) {
        return bf16_round(value);
    }
    return value;
}

template <typename TileT>
static __tile__ auto gelu_erf_poly9_l30_fast(TileT x) {
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

template <bool AddResidual,
          bool RoundOutputAcc,
          int MemoryLatency,
          bool ALoadLatencyHint = (MemoryLatency > 0),
          bool W1WeightLatencyHint = (MemoryLatency > 0),
          bool StoreLatencyHint = (MemoryLatency > 0)>
__tile_global__ void ffn12_residual_pairh32_poly9_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ w1_nt,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2_nt,
    const __nv_bfloat16* __restrict__ b2,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ out) {
    constexpr int OutHalf = kOut / 2;
    using HiddenAccTile = ct::tile<float, ct::shape<kTileM, kHiddenTile>>;
    using OutAccTile = ct::tile<float, ct::shape<kTileM, OutHalf>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<kTileM, kTileK>>;
    using W1Tile = ct::tile<__nv_bfloat16, ct::shape<kTileK, kHiddenTile>>;
    using W2Tile = ct::tile<__nv_bfloat16, ct::shape<kHiddenTile, OutHalf>>;
    using I64HiddenTile = ct::tile<long long, ct::shape<kTileM, kHiddenTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<kTileM, OutHalf>>;

    a = ct::assume_aligned(a, 16_ic);
    w1_nt = ct::assume_aligned(w1_nt, 16_ic);
    b1 = ct::assume_aligned(b1, 16_ic);
    w2_nt = ct::assume_aligned(w2_nt, 16_ic);
    b2 = ct::assume_aligned(b2, 16_ic);
    if constexpr (AddResidual) {
        residual = ct::assume_aligned(residual, 16_ic);
    }
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kIn>{}},
        ct::shape<kTileM, kTileK>{}
    };
    auto w1_view = ct::partition_view{
        ct::tensor_span{w1_nt, ct::shape<kIn, kHidden>{}, ct::layout_left{}},
        ct::shape<kTileK, kHiddenTile>{}
    };
    auto w2_view = ct::partition_view{
        ct::tensor_span{w2_nt, ct::shape<kHidden, kOut>{}, ct::layout_left{}},
        ct::shape<kHiddenTile, OutHalf>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<kTileM, OutHalf>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;

    auto out_acc0 = ct::full<OutAccTile>(0.0f);
    auto out_acc1 = ct::full<OutAccTile>(0.0f);
    I64HiddenTile hidden_local = ct::iota<I64HiddenTile>();
    for (auto hidden_pair : ct::irange(std::size_t{0},
                                       std::size_t{kHidden / (2 * kHiddenTile)})) {
        auto hidden_tile0 = hidden_pair * 2;
        auto hidden_tile1 = hidden_tile0 + 1;
        auto hidden_acc0 = ct::full<HiddenAccTile>(0.0f);
        auto hidden_acc1 = ct::full<HiddenAccTile>(0.0f);
        for (auto kk : ct::irange(std::size_t{0}, std::size_t{kIn / kTileK})) {
            if constexpr (MemoryLatency > 0 || ALoadLatencyHint || W1WeightLatencyHint) {
                ATile a_tile;
                W1Tile w1_0;
                W1Tile w1_1;
                if constexpr (ALoadLatencyHint) {
                    [[cutile::hint(0, latency=MemoryLatency)]]
                    a_tile = a_view.load(tile_m, kk);
                } else {
                    a_tile = a_view.load(tile_m, kk);
                }
                if constexpr (W1WeightLatencyHint) {
                    [[cutile::hint(0, latency=MemoryLatency)]]
                    w1_0 = w1_view.load(kk, hidden_tile0);
                    [[cutile::hint(0, latency=MemoryLatency)]]
                    w1_1 = w1_view.load(kk, hidden_tile1);
                } else {
                    w1_0 = w1_view.load(kk, hidden_tile0);
                    w1_1 = w1_view.load(kk, hidden_tile1);
                }
                hidden_acc0 = ct::mma(a_tile, w1_0, hidden_acc0);
                hidden_acc1 = ct::mma(a_tile, w1_1, hidden_acc1);
            } else {
                auto a_tile = a_view.load(tile_m, kk);
                hidden_acc0 = ct::mma(a_tile, w1_view.load(kk, hidden_tile0), hidden_acc0);
                hidden_acc1 = ct::mma(a_tile, w1_view.load(kk, hidden_tile1), hidden_acc1);
            }
        }

        auto hidden_cols0 =
            static_cast<long long>(hidden_tile0) * kHiddenTile +
            (hidden_local % kHiddenTile);
        auto hidden_cols1 =
            static_cast<long long>(hidden_tile1) * kHiddenTile +
            (hidden_local % kHiddenTile);
        auto hidden_bias0 = ct::element_cast<float>(ct::load(b1 + hidden_cols0));
        auto hidden_bias1 = ct::element_cast<float>(ct::load(b1 + hidden_cols1));
        auto hidden_value0 = gelu_erf_poly9_l30_fast(bf16_round(hidden_acc0) + hidden_bias0);
        auto hidden_value1 = gelu_erf_poly9_l30_fast(bf16_round(hidden_acc1) + hidden_bias1);
        auto hidden_bf16_0 = ct::element_cast<__nv_bfloat16>(hidden_value0);
        auto hidden_bf16_1 = ct::element_cast<__nv_bfloat16>(hidden_value1);

        if constexpr (MemoryLatency > 0) {
            W2Tile w2_00;
            W2Tile w2_01;
            W2Tile w2_10;
            W2Tile w2_11;
            [[cutile::hint(0, latency=MemoryLatency)]]
            w2_00 = w2_view.load(hidden_tile0, 0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w2_01 = w2_view.load(hidden_tile0, 1);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w2_10 = w2_view.load(hidden_tile1, 0);
            [[cutile::hint(0, latency=MemoryLatency)]]
            w2_11 = w2_view.load(hidden_tile1, 1);
            out_acc0 = ct::mma(hidden_bf16_0, w2_00, out_acc0);
            out_acc1 = ct::mma(hidden_bf16_0, w2_01, out_acc1);
            out_acc0 = ct::mma(hidden_bf16_1, w2_10, out_acc0);
            out_acc1 = ct::mma(hidden_bf16_1, w2_11, out_acc1);
        } else {
            out_acc0 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 0), out_acc0);
            out_acc1 = ct::mma(hidden_bf16_0, w2_view.load(hidden_tile0, 1), out_acc1);
            out_acc0 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 0), out_acc0);
            out_acc1 = ct::mma(hidden_bf16_1, w2_view.load(hidden_tile1, 1), out_acc1);
        }
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_cols = out_local % OutHalf;
    auto out_bias0 = ct::element_cast<float>(ct::load(b2 + out_cols));
    auto out_bias1 = ct::element_cast<float>(ct::load(b2 + OutHalf + out_cols));
    auto value0 = bf16_round_if<RoundOutputAcc>(out_acc0) + out_bias0;
    auto value1 = bf16_round_if<RoundOutputAcc>(out_acc1) + out_bias1;
    auto out_value0 = ct::element_cast<__nv_bfloat16>(value0);
    auto out_value1 = ct::element_cast<__nv_bfloat16>(value1);
    if constexpr (AddResidual) {
        auto residual_view = ct::partition_view{
            ct::tensor_span{residual, ct::shape<kM, kOut>{}},
            ct::shape<kTileM, OutHalf>{}
        };
        out_value0 = residual_view.load(tile_m, 0) + out_value0;
        out_value1 = residual_view.load(tile_m, 1) + out_value1;
    } else {
        (void)residual;
    }
    if constexpr (StoreLatencyHint) {
        [[cutile::hint(0, latency=MemoryLatency)]]
        out_view.store(out_value0, tile_m, 0);
        [[cutile::hint(0, latency=MemoryLatency)]]
        out_view.store(out_value1, tile_m, 1);
    } else {
        out_view.store(out_value0, tile_m, 0);
        out_view.store(out_value1, tile_m, 1);
    }
}

__tile_global__ void residual_add_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                              const __nv_bfloat16* __restrict__ residual,
                                              __nv_bfloat16* __restrict__ out) {
    using AddTile = ct::tile<__nv_bfloat16, ct::shape<kTileM, kOut>>;

    x = ct::assume_aligned(x, 16_ic);
    residual = ct::assume_aligned(residual, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto x_view = ct::partition_view{
        ct::tensor_span{x, ct::shape<kM, kOut>{}},
        ct::shape<kTileM, kOut>{}
    };
    auto residual_view = ct::partition_view{
        ct::tensor_span{residual, ct::shape<kM, kOut>{}},
        ct::shape<kTileM, kOut>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kM, kOut>{}},
        ct::shape<kTileM, kOut>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;
    AddTile value = x_view.load(tile_m, 0) + residual_view.load(tile_m, 0);
    out_view.store(value, tile_m, 0);
}

template <bool AddResidual,
          bool RoundOutputAcc,
          int MemoryLatency,
          bool StoreLatencyHint = (MemoryLatency > 0)>
void launch_residual(const __nv_bfloat16* d_a,
                     const __nv_bfloat16* d_w1,
                     const __nv_bfloat16* d_b1,
                     const __nv_bfloat16* d_w2,
                     const __nv_bfloat16* d_b2,
                     const __nv_bfloat16* d_residual,
                     __nv_bfloat16* d_out) {
    dim3 grid(kM / kTileM, 1, 1);
    ffn12_residual_pairh32_poly9_kernel<AddResidual,
                                        RoundOutputAcc,
                                        MemoryLatency,
                                        (MemoryLatency > 0),
                                        (MemoryLatency > 0),
                                        StoreLatencyHint>
        <<<grid, 1>>>(d_a, d_w1, d_b1, d_w2, d_b2, d_residual, d_out);
}

void launch_residual_add(const __nv_bfloat16* d_x,
                         const __nv_bfloat16* d_residual,
                         __nv_bfloat16* d_out) {
    dim3 grid(kM / kTileM, 1, 1);
    residual_add_bf16_kernel<<<grid, 1>>>(d_x, d_residual, d_out);
}

template <typename Launch>
void run_variant(const char* name,
                 const Options& opts,
                 const __nv_bfloat16* d_a,
                 const __nv_bfloat16* d_w1,
                 const __nv_bfloat16* d_b1,
                 const __nv_bfloat16* d_w2,
                 const __nv_bfloat16* d_b2,
                 const __nv_bfloat16* d_residual,
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

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double useful_flops = 2.0 * kM * kHidden * kIn +
                          2.0 * kM * kOut * kHidden;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double useful_tf = useful_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-22s best=%.3f ms median=%.3f ms useful=%.2f TF/s roof=%.1f%% checksum=%.4f\n",
        name, best_ms, median_ms, useful_tf,
        useful_tf * 100.0 / kA10gDenseBf16Tflops, __bfloat162float(checksum_bf16));

    (void)d_a;
    (void)d_w1;
    (void)d_b1;
    (void)d_w2;
    (void)d_b2;
    (void)d_residual;
}

bool should_run(const Options& opts, const char* name) {
    return opts.variant == "all" || opts.variant == name;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        size_t a_elems = static_cast<size_t>(kM) * kIn;
        size_t w1_elems = static_cast<size_t>(kHidden) * kIn;
        size_t b1_elems = kHidden;
        size_t w2_elems = static_cast<size_t>(kOut) * kHidden;
        size_t b2_elems = kOut;
        size_t out_elems = static_cast<size_t>(kM) * kOut;

        __nv_bfloat16* d_a = nullptr;
        __nv_bfloat16* d_w1 = nullptr;
        __nv_bfloat16* d_b1 = nullptr;
        __nv_bfloat16* d_w2 = nullptr;
        __nv_bfloat16* d_b2 = nullptr;
        __nv_bfloat16* d_residual = nullptr;
        __nv_bfloat16* d_out = nullptr;
        __nv_bfloat16* d_tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_w1, w1_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_b1, b1_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_w2, w2_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_b2, b2_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_residual, out_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_tmp, out_elems * sizeof(__nv_bfloat16)));

        init_bf16(d_a, a_elems);
        init_bf16(d_w1, w1_elems);
        init_bf16(d_b1, b1_elems);
        init_bf16(d_w2, w2_elems);
        init_bf16(d_b2, b2_elems);
        init_bf16(d_residual, out_elems);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::printf("FFN12 residual CUDA Tile fusion probe\n");
        std::printf("shape: M=%d, in=%d, hidden=%d, out=%d, BF16 storage, FP32 mma accumulate\n",
                    kM, kIn, kHidden, kOut);

        if (should_run(opts, "old")) {
            run_variant("old_round_lat0", opts, d_a, d_w1, d_b1, d_w2, d_b2,
                        d_residual, d_out,
                        [&] {
                            launch_residual<true, true, 0>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_residual, d_out);
                        });
        }
        if (should_run(opts, "outnoround_lat0")) {
            run_variant("outnoround_lat0", opts, d_a, d_w1, d_b1, d_w2, d_b2,
                        d_residual, d_out,
                        [&] {
                            launch_residual<true, false, 0>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_residual, d_out);
                        });
        }
        if (should_run(opts, "fast_lat2_outnoround")) {
            run_variant("fast_lat2_outnoround", opts, d_a, d_w1, d_b1, d_w2, d_b2,
                        d_residual, d_out,
                        [&] {
                            launch_residual<true, false, 2>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_residual, d_out);
                        });
        }
        if (should_run(opts, "fast_lat2_outnoround_nostore")) {
            run_variant("fast_lat2_outnoround_nostore", opts, d_a, d_w1, d_b1,
                        d_w2, d_b2, d_residual, d_out,
                        [&] {
                            launch_residual<true, false, 2, false>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_residual, d_out);
                        });
        }
        if (should_run(opts, "nores_old")) {
            run_variant("nores_round_lat0", opts, d_a, d_w1, d_b1, d_w2, d_b2,
                        d_residual, d_out,
                        [&] {
                            launch_residual<false, true, 0>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_residual, d_out);
                        });
        }
        if (should_run(opts, "nores_outnoround_lat0")) {
            run_variant("nores_outnoround_lat0", opts, d_a, d_w1, d_b1, d_w2,
                        d_b2, d_residual, d_out,
                        [&] {
                            launch_residual<false, false, 0>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_residual, d_out);
                        });
        }
        if (should_run(opts, "nores_fast_lat2_outnoround")) {
            run_variant("nores_fast_lat2_outnoround", opts, d_a, d_w1, d_b1,
                        d_w2, d_b2, d_residual, d_out,
                        [&] {
                            launch_residual<false, false, 2>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_residual, d_out);
                        });
        }
        if (should_run(opts, "two_kernel_fast")) {
            run_variant("two_kernel_fast", opts, d_a, d_w1, d_b1, d_w2, d_b2,
                        d_residual, d_out,
                        [&] {
                            launch_residual<false, false, 2>(
                                d_a, d_w1, d_b1, d_w2, d_b2, d_residual, d_tmp);
                            launch_residual_add(d_tmp, d_residual, d_out);
                        });
        }

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_w1));
        CUDA_CHECK(cudaFree(d_b1));
        CUDA_CHECK(cudaFree(d_w2));
        CUDA_CHECK(cudaFree(d_b2));
        CUDA_CHECK(cudaFree(d_residual));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_tmp));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
