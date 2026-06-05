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
constexpr int kM = 78060;
constexpr int kFullM = (kM / 32) * 32;
constexpr int kTokens = 1301;
constexpr int kHeads = 8;
constexpr int kHeadDim = 64;
constexpr int kN = 1536;
constexpr int kK = 256;
constexpr int kFullM64 = (kM / 64) * 64;
constexpr double kA10gDenseBf16Tflops = 70.0;

using I64Tile = ct::tile<long long, ct::shape<kInitTile>>;
using F32Tile = ct::tile<float, ct::shape<kInitTile>>;

struct Options {
    std::string variant = "all";
    int warmup = 20;
    int iters = 300;
    bool describe = false;
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
        } else if (std::strcmp(argv[i], "--describe") == 0) {
            opts.describe = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: bench_bf16_qkv_bkn_cutile [options]\n"
                "  --variant NAME  all, baseline, lat1, lat2, lat4, a2_b2_s0,\n"
                "                  a2_b1_s0, compound2x128, compound2x128_lat2,\n"
                "                  tn32_lat2, tn64_lat2, tn128_lat2,\n"
                "                  m16_tn256_lat2, m64_tn64_lat2,\n"
                "                  m64_tn32_lat2,\n"
                "                  m64_tn128_lat2,\n"
                "                  m64_tn128_lat1, m64_tn128_lat4,\n"
                "                  m64_tn128_a2_b2_s0,\n"
                "                  m64_tn128_a2_b1_s0,\n"
                "                  m64_tn128_a1_b2_s2,\n"
                "                  m64_tn128_lat1_masktail,\n"
                "                  split_contig_lat2, split_contig_tk8_lat2,\n"
                "                  split_contig_a2_b2_s0,\n"
                "                  split_contig_a1_b2_s2,\n"
                "                  split_contig_a2_b1_s2,\n"
                "                  split_contig_a1_b1_s2,\n"
                "                  split_contig_a0_b2_s2,\n"
                "                  split_contig_a2_b0_s2,\n"
                "                  split_contig_gridz_lat2,\n"
                "                  split_contig_pairk, split_contig_pairk_lat2,\n"
                "                  split_contig_compound2x128_lat2,\n"
                "                  headmajor_lat2,\n"
                "                  split_contig_tn32_lat2, split_contig_tn64_lat2,\n"
                "                  split_contig_tn128_lat2,\n"
                "                  split_contig_m64_tn64_lat2,\n"
                "                  split_contig_m64_tn128_lat2,\n"
                "                  split_contig_tk32_lat2, split_contig_tk64_lat2,\n"
                "                  store_one_lat2, store_f32_lat2\n"
                "  --warmup N      warmup launches, default 20\n"
                "  --iters N       measured launches, default 300\n"
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

__tile_global__ void fill_bf16_kernel(__nv_bfloat16* __restrict__ dst, long long total) {
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = static_cast<long long>(ct::bid().x) * kInitTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    F32Tile values = 0.25f + ct::element_cast<float>((idx * 13LL) & 1023LL) * 0.0009765625f;
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

void init_bf16(__nv_bfloat16* dst, long long total) {
    fill_bf16_kernel<<<ceildiv(total, kInitTile), 1>>>(dst, total);
    CUDA_CHECK(cudaGetLastError());
}

template <int TM, int TN, int TK, int M, int N, int K>
__tile_global__ void qkv_bkn_baseline_kernel(const __nv_bfloat16* __restrict__ a,
                                             const __nv_bfloat16* __restrict__ b_kn,
                                             __nv_bfloat16* __restrict__ c) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;

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

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        acc = ct::mma(a_view.load(tile_m, kk), b_view.load(kk, tile_n), acc);
    }
    c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
__tile_global__ void qkv_bkn_latency_kernel(const __nv_bfloat16* __restrict__ a,
                                            const __nv_bfloat16* __restrict__ b_kn,
                                            __nv_bfloat16* __restrict__ c) {
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

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        ATile a_tile;
        BTile b_tile;
        if constexpr (LoadLatency > 0) {
            [[ cutile::hint(0, latency=LoadLatency) ]]
            a_tile = a_view.load(tile_m, kk);
        } else {
            a_tile = a_view.load(tile_m, kk);
        }
        if constexpr (BLoadLatency > 0) {
            [[ cutile::hint(0, latency=BLoadLatency) ]]
            b_tile = b_view.load(kk, tile_n);
        } else {
            b_tile = b_view.load(kk, tile_n);
        }
        acc = ct::mma(a_tile, b_tile, acc);
    }
    if constexpr (StoreLatency > 0) {
        [[ cutile::hint(0, latency=StoreLatency) ]]
        c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
    } else {
        c_view.store(ct::element_cast<__nv_bfloat16>(acc), tile_m, tile_n);
    }
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int TailStart,
          int LoadLatency>
__tile_global__ void qkv_bkn_masked_tail_kernel(const __nv_bfloat16* __restrict__ a,
                                                const __nv_bfloat16* __restrict__ b_kn,
                                                __nv_bfloat16* __restrict__ c) {
    static_assert(TailStart % TM == 0);
    static_assert(TailStart < M);
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

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
__tile_global__ void qkv_bkn_split_contig_gridz_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v) {
    static_assert(N == 3 * 8 * 64);
    static_assert((N / 3) % TN == 0);
    static_assert(K % TK == 0);
    constexpr int kComponentN = N / 3;
    constexpr int kComponentTiles = kComponentN / TN;
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
        ct::tensor_span{q, ct::shape<M, kComponentN>{}},
        ct::shape<TM, TN>{}
    };
    auto k_view = ct::partition_view{
        ct::tensor_span{k_out, ct::shape<M, kComponentN>{}},
        ct::shape<TM, TN>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v, ct::shape<M, kComponentN>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_component_n, component] = ct::bid();
    auto packed_tile_n = component * kComponentTiles + tile_component_n;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        ATile a_tile;
        BTile b_tile;
        if constexpr (LoadLatency > 0) {
            [[ cutile::hint(0, latency=LoadLatency) ]]
            a_tile = a_view.load(tile_m, kk);
        } else {
            a_tile = a_view.load(tile_m, kk);
        }
        if constexpr (BLoadLatency > 0) {
            [[ cutile::hint(0, latency=BLoadLatency) ]]
            b_tile = b_view.load(kk, packed_tile_n);
        } else {
            b_tile = b_view.load(kk, packed_tile_n);
        }
        acc = ct::mma(a_tile, b_tile, acc);
    }

    auto out = ct::element_cast<__nv_bfloat16>(acc);
    if (component == 0) {
        if constexpr (StoreLatency > 0) {
            [[ cutile::hint(0, latency=StoreLatency) ]]
            q_view.store(out, tile_m, tile_component_n);
        } else {
            q_view.store(out, tile_m, tile_component_n);
        }
    } else if (component == 1) {
        if constexpr (StoreLatency > 0) {
            [[ cutile::hint(0, latency=StoreLatency) ]]
            k_view.store(out, tile_m, tile_component_n);
        } else {
            k_view.store(out, tile_m, tile_component_n);
        }
    } else {
        if constexpr (StoreLatency > 0) {
            [[ cutile::hint(0, latency=StoreLatency) ]]
            v_view.store(out, tile_m, tile_component_n);
        } else {
            v_view.store(out, tile_m, tile_component_n);
        }
    }
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
__tile_global__ void qkv_bkn_split_contig_pairk_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v) {
    static_assert(N == 3 * 8 * 64);
    static_assert((N / 3) % TN == 0);
    static_assert(K % (2 * TK) == 0);
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
    for (auto kpair : ct::irange(std::size_t{0}, std::size_t{K / (2 * TK)})) {
        auto kk0 = kpair * 2;
        auto kk1 = kk0 + 1;
        ATile a_tile0;
        ATile a_tile1;
        BTile b_tile0;
        BTile b_tile1;
        if constexpr (LoadLatency > 0) {
            [[ cutile::hint(0, latency=LoadLatency) ]]
            a_tile0 = a_view.load(tile_m, kk0);
            [[ cutile::hint(0, latency=LoadLatency) ]]
            a_tile1 = a_view.load(tile_m, kk1);
        } else {
            a_tile0 = a_view.load(tile_m, kk0);
            a_tile1 = a_view.load(tile_m, kk1);
        }
        if constexpr (BLoadLatency > 0) {
            [[ cutile::hint(0, latency=BLoadLatency) ]]
            b_tile0 = b_view.load(kk0, tile_n);
            [[ cutile::hint(0, latency=BLoadLatency) ]]
            b_tile1 = b_view.load(kk1, tile_n);
        } else {
            b_tile0 = b_view.load(kk0, tile_n);
            b_tile1 = b_view.load(kk1, tile_n);
        }
        acc = ct::mma(a_tile0, b_tile0, acc);
        acc = ct::mma(a_tile1, b_tile1, acc);
    }

    auto out = ct::element_cast<__nv_bfloat16>(acc);
    if (tile_n < kComponentTiles) {
        if constexpr (StoreLatency > 0) {
            [[ cutile::hint(0, latency=StoreLatency) ]]
            q_view.store(out, tile_m, tile_n);
        } else {
            q_view.store(out, tile_m, tile_n);
        }
    } else if (tile_n < 2 * kComponentTiles) {
        if constexpr (StoreLatency > 0) {
            [[ cutile::hint(0, latency=StoreLatency) ]]
            k_view.store(out, tile_m, tile_n - kComponentTiles);
        } else {
            k_view.store(out, tile_m, tile_n - kComponentTiles);
        }
    } else {
        if constexpr (StoreLatency > 0) {
            [[ cutile::hint(0, latency=StoreLatency) ]]
            v_view.store(out, tile_m, tile_n - 2 * kComponentTiles);
        } else {
            v_view.store(out, tile_m, tile_n - 2 * kComponentTiles);
        }
    }
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
__tile_global__ void qkv_bkn_split_contig_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v) {
    static_assert(N == 3 * 8 * 64);
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
        ATile a_tile;
        BTile b_tile;
        if constexpr (LoadLatency > 0) {
            [[ cutile::hint(0, latency=LoadLatency) ]]
            a_tile = a_view.load(tile_m, kk);
        } else {
            a_tile = a_view.load(tile_m, kk);
        }
        if constexpr (BLoadLatency > 0) {
            [[ cutile::hint(0, latency=BLoadLatency) ]]
            b_tile = b_view.load(kk, tile_n);
        } else {
            b_tile = b_view.load(kk, tile_n);
        }
        acc = ct::mma(a_tile, b_tile, acc);
    }

    auto out = ct::element_cast<__nv_bfloat16>(acc);
    if (tile_n < kComponentTiles) {
        if constexpr (StoreLatency > 0) {
            [[ cutile::hint(0, latency=StoreLatency) ]]
            q_view.store(out, tile_m, tile_n);
        } else {
            q_view.store(out, tile_m, tile_n);
        }
    } else if (tile_n < 2 * kComponentTiles) {
        if constexpr (StoreLatency > 0) {
            [[ cutile::hint(0, latency=StoreLatency) ]]
            k_view.store(out, tile_m, tile_n - kComponentTiles);
        } else {
            k_view.store(out, tile_m, tile_n - kComponentTiles);
        }
    } else {
        if constexpr (StoreLatency > 0) {
            [[ cutile::hint(0, latency=StoreLatency) ]]
            v_view.store(out, tile_m, tile_n - 2 * kComponentTiles);
        } else {
            v_view.store(out, tile_m, tile_n - 2 * kComponentTiles);
        }
    }
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency,
          int BLoadLatency = LoadLatency>
__tile_global__ void qkv_bkn_headmajor_kernel(const __nv_bfloat16* __restrict__ a,
                                              const __nv_bfloat16* __restrict__ b_kn,
                                              __nv_bfloat16* __restrict__ q,
                                              __nv_bfloat16* __restrict__ k_out,
                                              __nv_bfloat16* __restrict__ v) {
    static_assert(N == 3 * kHeads * kHeadDim);
    static_assert(TN == 4 * kHeadDim);
    static_assert(K % TK == 0);
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using BTile = ct::tile<__nv_bfloat16, ct::shape<TK, TN>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, TN>>;

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

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        ATile a_tile;
        BTile b_tile;
        if constexpr (LoadLatency > 0) {
            [[ cutile::hint(0, latency=LoadLatency) ]]
            a_tile = a_view.load(tile_m, kk);
        } else {
            a_tile = a_view.load(tile_m, kk);
        }
        if constexpr (BLoadLatency > 0) {
            [[ cutile::hint(0, latency=BLoadLatency) ]]
            b_tile = b_view.load(kk, tile_n);
        } else {
            b_tile = b_view.load(kk, tile_n);
        }
        acc = ct::mma(a_tile, b_tile, acc);
    }

    I64OutTile local = ct::iota<I64OutTile>();
    auto row = static_cast<long long>(tile_m) * TM + local / TN;
    auto col = local % TN;
    auto batch = row / kTokens;
    auto token = row - batch * kTokens;
    constexpr int kComponentTiles = (N / 3) / TN;
    constexpr int kHeadsPerTile = TN / kHeadDim;
    auto component = static_cast<long long>(tile_n) / kComponentTiles;
    auto component_tile = static_cast<long long>(tile_n) - component * kComponentTiles;
    auto head = component_tile * kHeadsPerTile + col / kHeadDim;
    auto dim = col % kHeadDim;
    auto out_idx = ((batch * kHeads + head) * kTokens + token) * kHeadDim + dim;
    auto out = ct::element_cast<__nv_bfloat16>(acc);
    if (component == 0) {
        ct::store(q + out_idx, out);
    } else if (component == 1) {
        ct::store(k_out + out_idx, out);
    } else {
        ct::store(v + out_idx, out);
    }
}

template <int M, int N, int K, int LoadLatency = 2, int BLoadLatency = LoadLatency>
__tile_global__ void qkv_bkn_compound2x128_split_contig_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_kn,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v) {
    static_assert(N == 3 * 8 * 64);
    static_assert((N / 3) % 256 == 0);
    static_assert(K % 16 == 0);
    constexpr int TM = 32;
    constexpr int TN = 128;
    constexpr int TK = 16;
    constexpr int kComponentN = N / 3;
    constexpr int kComponentTiles = kComponentN / 256;
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using BTile = ct::tile<__nv_bfloat16, ct::shape<TK, TN>>;
    using BShape = ct::shape<K, TN>;
    using OutShape = ct::shape<M, TN>;
    using BStrides = ct::shape<N, 1>;
    using OutStrides = ct::shape<kComponentN, 1>;
    using BLayout = ct::layout_strided<BStrides>;
    using OutLayout = ct::layout_strided<OutStrides>;
    using BMapping = typename BLayout::template mapping<BShape>;
    using OutMapping = typename OutLayout::template mapping<OutShape>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    q = ct::assume_aligned(q, 16_ic);
    k_out = ct::assume_aligned(k_out, 16_ic);
    v = ct::assume_aligned(v, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<TM, TK>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto component = tile_n / kComponentTiles;
    auto component_tile = tile_n - component * kComponentTiles;
    std::size_t packed_n_base = static_cast<std::size_t>(tile_n) * 256;
    std::size_t out_n_base = static_cast<std::size_t>(component_tile) * 256;

    auto b0_view = ct::partition_view{
        ct::tensor_span{b_kn + packed_n_base, BMapping{BShape{}, BStrides{}}},
        ct::shape<TK, TN>{}
    };
    auto b1_view = ct::partition_view{
        ct::tensor_span{b_kn + packed_n_base + TN, BMapping{BShape{}, BStrides{}}},
        ct::shape<TK, TN>{}
    };

    __nv_bfloat16* out_ptr = component == 0 ? q : (component == 1 ? k_out : v);
    auto out0_view = ct::partition_view{
        ct::tensor_span{out_ptr + out_n_base, OutMapping{OutShape{}, OutStrides{}}},
        ct::shape<TM, TN>{}
    };
    auto out1_view = ct::partition_view{
        ct::tensor_span{out_ptr + out_n_base + TN, OutMapping{OutShape{}, OutStrides{}}},
        ct::shape<TM, TN>{}
    };

    auto acc0 = ct::full<AccTile>(0.0f);
    auto acc1 = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        ATile a_tile;
        BTile b0_tile;
        BTile b1_tile;
        if constexpr (LoadLatency > 0) {
            [[cutile::hint(0, latency=LoadLatency)]]
            a_tile = a_view.load(tile_m, kk);
        } else {
            a_tile = a_view.load(tile_m, kk);
        }
        if constexpr (BLoadLatency > 0) {
            [[cutile::hint(0, latency=BLoadLatency)]]
            b0_tile = b0_view.load(kk, 0);
            [[cutile::hint(0, latency=BLoadLatency)]]
            b1_tile = b1_view.load(kk, 0);
        } else {
            b0_tile = b0_view.load(kk, 0);
            b1_tile = b1_view.load(kk, 0);
        }
        acc0 = ct::mma(a_tile, b0_tile, acc0);
        acc1 = ct::mma(a_tile, b1_tile, acc1);
    }

    out0_view.store(ct::element_cast<__nv_bfloat16>(acc0), tile_m, 0);
    out1_view.store(ct::element_cast<__nv_bfloat16>(acc1), tile_m, 0);
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency,
          int BLoadLatency = LoadLatency>
__tile_global__ void qkv_bkn_store_one_kernel(const __nv_bfloat16* __restrict__ a,
                                              const __nv_bfloat16* __restrict__ b_kn,
                                              __nv_bfloat16* __restrict__ c) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<TM, TK>>;
    using BTile = ct::tile<__nv_bfloat16, ct::shape<TK, TN>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, TN>>;

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

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        ATile a_tile;
        BTile b_tile;
        if constexpr (LoadLatency > 0) {
            [[ cutile::hint(0, latency=LoadLatency) ]]
            a_tile = a_view.load(tile_m, kk);
        } else {
            a_tile = a_view.load(tile_m, kk);
        }
        if constexpr (BLoadLatency > 0) {
            [[ cutile::hint(0, latency=BLoadLatency) ]]
            b_tile = b_view.load(kk, tile_n);
        } else {
            b_tile = b_view.load(kk, tile_n);
        }
        acc = ct::mma(a_tile, b_tile, acc);
    }

    I64OutTile local = ct::iota<I64OutTile>();
    auto row = static_cast<long long>(tile_m) * TM + local / TN;
    auto col = static_cast<long long>(tile_n) * TN + local % TN;
    ct::store_masked(c + row * N + col,
                     ct::element_cast<__nv_bfloat16>(acc),
                     local == 0);
}

template <int TM,
          int TN,
          int TK,
          int M,
          int N,
          int K,
          int LoadLatency,
          int BLoadLatency = LoadLatency>
__tile_global__ void qkv_bkn_store_f32_kernel(const __nv_bfloat16* __restrict__ a,
                                              const __nv_bfloat16* __restrict__ b_kn,
                                              float* __restrict__ c) {
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

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        ATile a_tile;
        BTile b_tile;
        if constexpr (LoadLatency > 0) {
            [[ cutile::hint(0, latency=LoadLatency) ]]
            a_tile = a_view.load(tile_m, kk);
        } else {
            a_tile = a_view.load(tile_m, kk);
        }
        if constexpr (BLoadLatency > 0) {
            [[ cutile::hint(0, latency=BLoadLatency) ]]
            b_tile = b_view.load(kk, tile_n);
        } else {
            b_tile = b_view.load(kk, tile_n);
        }
        acc = ct::mma(a_tile, b_tile, acc);
    }
    c_view.store(acc, tile_m, tile_n);
}

template <int M, int N, int K, int LoadLatency = 0, int BLoadLatency = LoadLatency>
__tile_global__ void qkv_bkn_compound2x128_kernel(const __nv_bfloat16* __restrict__ a,
                                                  const __nv_bfloat16* __restrict__ b_kn,
                                                  __nv_bfloat16* __restrict__ c) {
    static_assert(N % 256 == 0);
    using AccTile = ct::tile<float, ct::shape<32, 128>>;
    using ATile = ct::tile<__nv_bfloat16, ct::shape<32, 16>>;
    using BTile = ct::tile<__nv_bfloat16, ct::shape<16, 128>>;
    using BShape = ct::shape<K, 128>;
    using CShape = ct::shape<M, 128>;
    using Strides = ct::shape<N, 1>;
    using StridedLayout = ct::layout_strided<Strides>;
    using BMapping = typename StridedLayout::template mapping<BShape>;
    using CMapping = typename StridedLayout::template mapping<CShape>;

    a = ct::assume_aligned(a, 16_ic);
    b_kn = ct::assume_aligned(b_kn, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<M, K>{}},
        ct::shape<32, 16>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    std::size_t n_base = static_cast<std::size_t>(tile_n) * 256;
    auto b0_view = ct::partition_view{
        ct::tensor_span{b_kn + n_base, BMapping{BShape{}, Strides{}}},
        ct::shape<16, 128>{}
    };
    auto b1_view = ct::partition_view{
        ct::tensor_span{b_kn + n_base + 128, BMapping{BShape{}, Strides{}}},
        ct::shape<16, 128>{}
    };
    auto c0_view = ct::partition_view{
        ct::tensor_span{c + n_base, CMapping{CShape{}, Strides{}}},
        ct::shape<32, 128>{}
    };
    auto c1_view = ct::partition_view{
        ct::tensor_span{c + n_base + 128, CMapping{CShape{}, Strides{}}},
        ct::shape<32, 128>{}
    };

    auto acc0 = ct::full<AccTile>(0.0f);
    auto acc1 = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / 16})) {
        ATile a_tile;
        BTile b0_tile;
        BTile b1_tile;
        if constexpr (LoadLatency > 0) {
            [[ cutile::hint(0, latency=LoadLatency) ]]
            a_tile = a_view.load(tile_m, kk);
        } else {
            a_tile = a_view.load(tile_m, kk);
        }
        if constexpr (BLoadLatency > 0) {
            [[ cutile::hint(0, latency=BLoadLatency) ]]
            b0_tile = b0_view.load(kk, 0);
            [[ cutile::hint(0, latency=BLoadLatency) ]]
            b1_tile = b1_view.load(kk, 0);
        } else {
            b0_tile = b0_view.load(kk, 0);
            b1_tile = b1_view.load(kk, 0);
        }
        acc0 = ct::mma(a_tile, b0_tile, acc0);
        acc1 = ct::mma(a_tile, b1_tile, acc1);
    }
    c0_view.store(ct::element_cast<__nv_bfloat16>(acc0), tile_m, 0);
    c1_view.store(ct::element_cast<__nv_bfloat16>(acc1), tile_m, 0);
}

template <typename Launch>
void run_variant(const Options& opts, const char* name, Launch launch) {
    if (opts.variant != "all" && opts.variant != name) return;

    constexpr int kTM = 32;
    constexpr int kTN = 256;
    constexpr int kTK = 16;
    size_t a_elems = static_cast<size_t>(kM) * kK;
    size_t b_elems = static_cast<size_t>(kN) * kK;
    size_t c_elems = static_cast<size_t>(kM) * kN;
    double gib = static_cast<double>(a_elems + b_elems + c_elems) *
                 sizeof(__nv_bfloat16) / (1024.0 * 1024.0 * 1024.0);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, static_cast<long long>(a_elems));
    init_bf16(d_b, static_cast<long long>(b_elems));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch(d_a, d_b, d_c);
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
        launch(d_a, d_b, d_c);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_c, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    double flops = 2.0 * kFullM * kN * kK;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-12s tile=%dx%dx%d grid=(%d,%d) fullM=%d mem=%.2f GiB "
        "best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% checksum=%.4f\n",
        name, kTM, kTN, kTK, kFullM / kTM, kN / kTN, kFullM, gib,
        best_ms, median_ms, tflops, tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum_bf16));
}

template <int TM, int TN, int TK, int MShape, typename Launch>
void run_variant_shape(const Options& opts, const char* name, Launch launch) {
    if (opts.variant != "all" && opts.variant != name) return;

    size_t a_elems = static_cast<size_t>(kM) * kK;
    size_t b_elems = static_cast<size_t>(kN) * kK;
    size_t c_elems = static_cast<size_t>(kM) * kN;
    double gib = static_cast<double>(a_elems + b_elems + c_elems) *
                 sizeof(__nv_bfloat16) / (1024.0 * 1024.0 * 1024.0);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, static_cast<long long>(a_elems));
    init_bf16(d_b, static_cast<long long>(b_elems));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch(d_a, d_b, d_c);
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
        launch(d_a, d_b, d_c);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_c, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    double flops = 2.0 * MShape * kN * kK;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-18s tile=%dx%dx%d grid=(%d,%d) M=%d mem=%.2f GiB "
        "best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% checksum=%.4f\n",
        name, TM, TN, TK, MShape / TM, kN / TN, MShape, gib,
        best_ms, median_ms, tflops, tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum_bf16));
}

template <int TM,
          int TN,
          int TK,
          int MShape,
          int TailStart,
          typename LaunchFull,
          typename LaunchTail>
void run_variant_shape_masked_tail(const Options& opts,
                                   const char* name,
                                   LaunchFull launch_full,
                                   LaunchTail launch_tail) {
    if (opts.variant != "all" && opts.variant != name) return;

    size_t a_elems = static_cast<size_t>(kM) * kK;
    size_t b_elems = static_cast<size_t>(kN) * kK;
    size_t c_elems = static_cast<size_t>(kM) * kN;
    double gib = static_cast<double>(a_elems + b_elems + c_elems) *
                 sizeof(__nv_bfloat16) / (1024.0 * 1024.0 * 1024.0);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, static_cast<long long>(a_elems));
    init_bf16(d_b, static_cast<long long>(b_elems));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch_full(d_a, d_b, d_c);
        launch_tail(d_a, d_b, d_c);
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
        launch_full(d_a, d_b, d_c);
        launch_tail(d_a, d_b, d_c);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16,
                          d_c + static_cast<size_t>(TailStart) * kN,
                          sizeof(checksum_bf16),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    double flops = 2.0 * kM * kN * kK;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-26s tile=%dx%dx%d full_grid=(%d,%d) tail_grid=(%d,%d) fullM=%d M=%d "
        "mem=%.2f GiB best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% "
        "tail_checksum=%.4f\n",
        name, TM, TN, TK, MShape / TM, kN / TN,
        ceildiv(kM - TailStart, TM), kN / TN, MShape, kM, gib,
        best_ms, median_ms, tflops, tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum_bf16));
}

template <typename Launch>
void run_variant_f32_output(const Options& opts, const char* name, Launch launch) {
    if (opts.variant != "all" && opts.variant != name) return;

    constexpr int kTM = 32;
    constexpr int kTN = 256;
    constexpr int kTK = 16;
    size_t a_elems = static_cast<size_t>(kM) * kK;
    size_t b_elems = static_cast<size_t>(kN) * kK;
    size_t c_elems = static_cast<size_t>(kM) * kN;
    double gib = (static_cast<double>(a_elems + b_elems) * sizeof(__nv_bfloat16) +
                  static_cast<double>(c_elems) * sizeof(float)) /
                 (1024.0 * 1024.0 * 1024.0);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * sizeof(float)));
    init_bf16(d_a, static_cast<long long>(a_elems));
    init_bf16(d_b, static_cast<long long>(b_elems));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch(d_a, d_b, d_c);
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
        launch(d_a, d_b, d_c);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    float checksum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&checksum, d_c, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    double flops = 2.0 * kFullM * kN * kK;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-12s tile=%dx%dx%d grid=(%d,%d) fullM=%d mem=%.2f GiB "
        "best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% checksum=%.4f\n",
        name, kTM, kTN, kTK, kFullM / kTM, kN / kTN, kFullM, gib,
        best_ms, median_ms, tflops, tflops * 100.0 / kA10gDenseBf16Tflops,
        checksum);
}

template <int TN, int TK, typename Launch>
void run_variant_split_contig_shape(const Options& opts, const char* name, Launch launch) {
    if (opts.variant != "all" && opts.variant != name) return;

    constexpr int kTM = 32;
    size_t a_elems = static_cast<size_t>(kM) * kK;
    size_t b_elems = static_cast<size_t>(kN) * kK;
    size_t split_elems = static_cast<size_t>(kM) * (kN / 3);
    size_t out_elems = split_elems * 3;
    double gib = static_cast<double>(a_elems + b_elems + out_elems) *
                 sizeof(__nv_bfloat16) / (1024.0 * 1024.0 * 1024.0);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q, split_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, split_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, split_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, static_cast<long long>(a_elems));
    init_bf16(d_b, static_cast<long long>(b_elems));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch(d_a, d_b, d_q, d_k, d_v);
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
        launch(d_a, d_b, d_q, d_k, d_v);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_q, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));

    double flops = 2.0 * kFullM * kN * kK;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-18s tile=%dx%dx%d grid=(%d,%d) fullM=%d mem=%.2f GiB "
        "best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% checksum=%.4f\n",
        name, kTM, TN, TK, kFullM / kTM, kN / TN, kFullM, gib,
        best_ms, median_ms, tflops, tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum_bf16));
}

template <int TM, int TN, int TK, int MShape, typename Launch>
void run_variant_split_contig_shape_m(const Options& opts, const char* name, Launch launch) {
    if (opts.variant != "all" && opts.variant != name) return;

    size_t a_elems = static_cast<size_t>(kM) * kK;
    size_t b_elems = static_cast<size_t>(kN) * kK;
    size_t split_elems = static_cast<size_t>(kM) * (kN / 3);
    size_t out_elems = split_elems * 3;
    double gib = static_cast<double>(a_elems + b_elems + out_elems) *
                 sizeof(__nv_bfloat16) / (1024.0 * 1024.0 * 1024.0);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_q, split_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, split_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, split_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, static_cast<long long>(a_elems));
    init_bf16(d_b, static_cast<long long>(b_elems));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch(d_a, d_b, d_q, d_k, d_v);
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
        launch(d_a, d_b, d_q, d_k, d_v);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_q, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));

    double flops = 2.0 * MShape * kN * kK;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-26s tile=%dx%dx%d grid=(%d,%d) M=%d mem=%.2f GiB "
        "best=%.3f ms median=%.3f ms %.2f TF/s roof=%.1f%% checksum=%.4f\n",
        name, TM, TN, TK, MShape / TM, kN / TN, MShape, gib,
        best_ms, median_ms, tflops, tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum_bf16));
}

template <int TK, typename Launch>
void run_variant_split_contig(const Options& opts, const char* name, Launch launch) {
    run_variant_split_contig_shape<256, TK>(opts, name, launch);
}

template <int LoadLatency, int BLoadLatency = LoadLatency, int StoreLatency = LoadLatency>
void launch_latency(const __nv_bfloat16* d_a,
                    const __nv_bfloat16* d_b,
                    __nv_bfloat16* d_c) {
    dim3 grid(kFullM / 32, kN / 256, 1);
    qkv_bkn_latency_kernel<32,
                           256,
                           16,
                           kFullM,
                           kN,
                           kK,
                           LoadLatency,
                           BLoadLatency,
                           StoreLatency><<<grid, 1>>>(d_a, d_b, d_c);
}

template <int TM,
          int TN,
          int TK,
          int MShape,
          int LoadLatency,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
void launch_latency_shape(const __nv_bfloat16* d_a,
                          const __nv_bfloat16* d_b,
                          __nv_bfloat16* d_c) {
    dim3 grid(MShape / TM, kN / TN, 1);
    qkv_bkn_latency_kernel<TM,
                           TN,
                           TK,
                           MShape,
                           kN,
                           kK,
                           LoadLatency,
                           BLoadLatency,
                           StoreLatency><<<grid, 1>>>(d_a, d_b, d_c);
}

template <int TM, int TN, int TK, int TailStart, int LoadLatency>
void launch_masked_tail_shape(const __nv_bfloat16* d_a,
                              const __nv_bfloat16* d_b,
                              __nv_bfloat16* d_c) {
    dim3 grid(ceildiv(kM - TailStart, TM), kN / TN, 1);
    qkv_bkn_masked_tail_kernel<TM,
                               TN,
                               TK,
                               kM,
                               kN,
                               kK,
                               TailStart,
                               LoadLatency><<<grid, 1>>>(d_a, d_b, d_c);
}

void launch_baseline(const __nv_bfloat16* d_a,
                     const __nv_bfloat16* d_b,
                     __nv_bfloat16* d_c) {
    dim3 grid(kFullM / 32, kN / 256, 1);
    qkv_bkn_baseline_kernel<32, 256, 16, kFullM, kN, kK><<<grid, 1>>>(
        d_a, d_b, d_c);
}

template <int TK,
          int LoadLatency,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency,
          int TN = 256>
void launch_split_contig(const __nv_bfloat16* d_a,
                         const __nv_bfloat16* d_b,
                         __nv_bfloat16* d_q,
                         __nv_bfloat16* d_k,
                         __nv_bfloat16* d_v) {
    dim3 grid(kFullM / 32, kN / TN, 1);
    qkv_bkn_split_contig_kernel<32,
                                TN,
                                TK,
                                kFullM,
                                kN,
                                kK,
                                LoadLatency,
                                BLoadLatency,
                                StoreLatency>
        <<<grid, 1>>>(d_a, d_b, d_q, d_k, d_v);
}

template <int TM,
          int TN,
          int TK,
          int MShape,
          int LoadLatency,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
void launch_split_contig_m(const __nv_bfloat16* d_a,
                           const __nv_bfloat16* d_b,
                           __nv_bfloat16* d_q,
                           __nv_bfloat16* d_k,
                           __nv_bfloat16* d_v) {
    dim3 grid(MShape / TM, kN / TN, 1);
    qkv_bkn_split_contig_kernel<TM,
                                TN,
                                TK,
                                MShape,
                                kN,
                                kK,
                                LoadLatency,
                                BLoadLatency,
                                StoreLatency>
        <<<grid, 1>>>(d_a, d_b, d_q, d_k, d_v);
}

template <int TK, int LoadLatency, int BLoadLatency = LoadLatency>
void launch_headmajor(const __nv_bfloat16* d_a,
                      const __nv_bfloat16* d_b,
                      __nv_bfloat16* d_q,
                      __nv_bfloat16* d_k,
                      __nv_bfloat16* d_v) {
    dim3 grid(kFullM / 32, kN / 256, 1);
    qkv_bkn_headmajor_kernel<32,
                             256,
                             TK,
                             kFullM,
                             kN,
                             kK,
                             LoadLatency,
                             BLoadLatency>
        <<<grid, 1>>>(d_a, d_b, d_q, d_k, d_v);
}

template <int TK,
          int LoadLatency,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
void launch_split_contig_gridz(const __nv_bfloat16* d_a,
                               const __nv_bfloat16* d_b,
                               __nv_bfloat16* d_q,
                               __nv_bfloat16* d_k,
                               __nv_bfloat16* d_v) {
    dim3 grid(kFullM / 32, (kN / 3) / 256, 3);
    qkv_bkn_split_contig_gridz_kernel<32,
                                      256,
                                      TK,
                                      kFullM,
                                      kN,
                                      kK,
                                      LoadLatency,
                                      BLoadLatency,
                                      StoreLatency>
        <<<grid, 1>>>(d_a, d_b, d_q, d_k, d_v);
}

template <int TK,
          int LoadLatency,
          int BLoadLatency = LoadLatency,
          int StoreLatency = LoadLatency>
void launch_split_contig_pairk(const __nv_bfloat16* d_a,
                               const __nv_bfloat16* d_b,
                               __nv_bfloat16* d_q,
                               __nv_bfloat16* d_k,
                               __nv_bfloat16* d_v) {
    dim3 grid(kFullM / 32, kN / 256, 1);
    qkv_bkn_split_contig_pairk_kernel<32,
                                      256,
                                      TK,
                                      kFullM,
                                      kN,
                                      kK,
                                      LoadLatency,
                                      BLoadLatency,
                                      StoreLatency>
        <<<grid, 1>>>(d_a, d_b, d_q, d_k, d_v);
}

template <int LoadLatency = 2, int BLoadLatency = LoadLatency>
void launch_split_contig_compound2x128(const __nv_bfloat16* d_a,
                                       const __nv_bfloat16* d_b,
                                       __nv_bfloat16* d_q,
                                       __nv_bfloat16* d_k,
                                       __nv_bfloat16* d_v) {
    dim3 grid(kFullM / 32, kN / 256, 1);
    qkv_bkn_compound2x128_split_contig_kernel<kFullM,
                                              kN,
                                              kK,
                                              LoadLatency,
                                              BLoadLatency>
        <<<grid, 1>>>(d_a, d_b, d_q, d_k, d_v);
}

template <int LoadLatency = 0, int BLoadLatency = LoadLatency>
void launch_store_one(const __nv_bfloat16* d_a,
                      const __nv_bfloat16* d_b,
                      __nv_bfloat16* d_c) {
    dim3 grid(kFullM / 32, kN / 256, 1);
    qkv_bkn_store_one_kernel<32,
                             256,
                             16,
                             kFullM,
                             kN,
                             kK,
                             LoadLatency,
                             BLoadLatency><<<grid, 1>>>(d_a, d_b, d_c);
}

template <int LoadLatency = 0, int BLoadLatency = LoadLatency>
void launch_store_f32(const __nv_bfloat16* d_a,
                      const __nv_bfloat16* d_b,
                      float* d_c) {
    dim3 grid(kFullM / 32, kN / 256, 1);
    qkv_bkn_store_f32_kernel<32,
                             256,
                             16,
                             kFullM,
                             kN,
                             kK,
                             LoadLatency,
                             BLoadLatency><<<grid, 1>>>(d_a, d_b, d_c);
}

template <int LoadLatency = 0, int BLoadLatency = LoadLatency>
void launch_compound2x128(const __nv_bfloat16* d_a,
                          const __nv_bfloat16* d_b,
                          __nv_bfloat16* d_c) {
    dim3 grid(kFullM / 32, kN / 256, 1);
    qkv_bkn_compound2x128_kernel<kFullM, kN, kK, LoadLatency, BLoadLatency>
        <<<grid, 1>>>(d_a, d_b, d_c);
}

template <typename Kernel>
void describe_kernel(const Options& opts, const char* name, Kernel kernel, dim3 grid) {
    if (opts.variant != "all" && opts.variant != name) return;

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
        "%-30s grid=(%u,%u,%u) waves/SM=%.1f attr_regs=%d "
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
                    "baseline",
                    qkv_bkn_baseline_kernel<32, 256, 16, kFullM, kN, kK>,
                    dim3(kFullM / 32, kN / 256, 1));
    describe_kernel(opts,
                    "lat2",
                    qkv_bkn_latency_kernel<32, 256, 16, kFullM, kN, kK, 2, 2, 2>,
                    dim3(kFullM / 32, kN / 256, 1));
    describe_kernel(opts,
                    "a2_b2_s0",
                    qkv_bkn_latency_kernel<32, 256, 16, kFullM, kN, kK, 2, 2, 0>,
                    dim3(kFullM / 32, kN / 256, 1));
    describe_kernel(opts,
                    "a2_b1_s0",
                    qkv_bkn_latency_kernel<32, 256, 16, kFullM, kN, kK, 2, 1, 0>,
                    dim3(kFullM / 32, kN / 256, 1));
    describe_kernel(opts,
                    "tn32_lat2",
                    qkv_bkn_latency_kernel<32, 32, 16, kFullM, kN, kK, 2, 2, 2>,
                    dim3(kFullM / 32, kN / 32, 1));
    describe_kernel(opts,
                    "tn64_lat2",
                    qkv_bkn_latency_kernel<32, 64, 16, kFullM, kN, kK, 2, 2, 2>,
                    dim3(kFullM / 32, kN / 64, 1));
    describe_kernel(opts,
                    "tn128_lat2",
                    qkv_bkn_latency_kernel<32, 128, 16, kFullM, kN, kK, 2, 2, 2>,
                    dim3(kFullM / 32, kN / 128, 1));
    describe_kernel(opts,
                    "m16_tn256_lat2",
                    qkv_bkn_latency_kernel<16, 256, 16, kFullM, kN, kK, 2, 2, 2>,
                    dim3(kFullM / 16, kN / 256, 1));
    describe_kernel(opts,
                    "m64_tn64_lat2",
                    qkv_bkn_latency_kernel<64, 64, 16, kFullM64, kN, kK, 2, 2, 2>,
                    dim3(kFullM64 / 64, kN / 64, 1));
    describe_kernel(opts,
                    "m64_tn32_lat2",
                    qkv_bkn_latency_kernel<64, 32, 16, kFullM64, kN, kK, 2, 2, 2>,
                    dim3(kFullM64 / 64, kN / 32, 1));
    describe_kernel(opts,
                    "m64_tn128_lat2",
                    qkv_bkn_latency_kernel<64, 128, 16, kFullM64, kN, kK, 2, 2, 2>,
                    dim3(kFullM64 / 64, kN / 128, 1));
    describe_kernel(opts,
                    "m64_tn128_lat1",
                    qkv_bkn_latency_kernel<64, 128, 16, kFullM64, kN, kK, 1, 1, 1>,
                    dim3(kFullM64 / 64, kN / 128, 1));
    describe_kernel(opts,
                    "m64_tn128_lat4",
                    qkv_bkn_latency_kernel<64, 128, 16, kFullM64, kN, kK, 4, 4, 4>,
                    dim3(kFullM64 / 64, kN / 128, 1));
    describe_kernel(opts,
                    "m64_tn128_a2_b2_s0",
                    qkv_bkn_latency_kernel<64, 128, 16, kFullM64, kN, kK, 2, 2, 0>,
                    dim3(kFullM64 / 64, kN / 128, 1));
    describe_kernel(opts,
                    "m64_tn128_a2_b1_s0",
                    qkv_bkn_latency_kernel<64, 128, 16, kFullM64, kN, kK, 2, 1, 0>,
                    dim3(kFullM64 / 64, kN / 128, 1));
    describe_kernel(opts,
                    "m64_tn128_a1_b2_s2",
                    qkv_bkn_latency_kernel<64, 128, 16, kFullM64, kN, kK, 1, 2, 2>,
                    dim3(kFullM64 / 64, kN / 128, 1));
    describe_kernel(opts,
                    "m64_tn128_lat1_masktail",
                    qkv_bkn_masked_tail_kernel<64, 128, 16, kM, kN, kK, kFullM64, 1>,
                    dim3(ceildiv(kM - kFullM64, 64), kN / 128, 1));
    describe_kernel(opts,
                    "compound2x128_lat2",
                    qkv_bkn_compound2x128_kernel<kFullM, kN, kK, 2, 2>,
                    dim3(kFullM / 32, kN / 256, 1));
    describe_kernel(opts,
                    "split_contig_lat2",
                    qkv_bkn_split_contig_kernel<32, 256, 16, kFullM, kN, kK, 2, 2, 2>,
                    dim3(kFullM / 32, kN / 256, 1));
    describe_kernel(opts,
                    "split_contig_a2_b2_s0",
                    qkv_bkn_split_contig_kernel<32, 256, 16, kFullM, kN, kK, 2, 2, 0>,
                    dim3(kFullM / 32, kN / 256, 1));
    describe_kernel(opts,
                    "split_contig_a1_b2_s2",
                    qkv_bkn_split_contig_kernel<32, 256, 16, kFullM, kN, kK, 1, 2, 2>,
                    dim3(kFullM / 32, kN / 256, 1));
    describe_kernel(opts,
                    "split_contig_a2_b1_s2",
                    qkv_bkn_split_contig_kernel<32, 256, 16, kFullM, kN, kK, 2, 1, 2>,
                    dim3(kFullM / 32, kN / 256, 1));
    describe_kernel(opts,
                    "split_contig_a1_b1_s2",
                    qkv_bkn_split_contig_kernel<32, 256, 16, kFullM, kN, kK, 1, 1, 2>,
                    dim3(kFullM / 32, kN / 256, 1));
    describe_kernel(opts,
                    "split_contig_a0_b2_s2",
                    qkv_bkn_split_contig_kernel<32, 256, 16, kFullM, kN, kK, 0, 2, 2>,
                    dim3(kFullM / 32, kN / 256, 1));
    describe_kernel(opts,
                    "split_contig_a2_b0_s2",
                    qkv_bkn_split_contig_kernel<32, 256, 16, kFullM, kN, kK, 2, 0, 2>,
                    dim3(kFullM / 32, kN / 256, 1));
    describe_kernel(opts,
                    "split_contig_gridz_lat2",
                    qkv_bkn_split_contig_gridz_kernel<32, 256, 16, kFullM, kN, kK, 2, 2, 2>,
                    dim3(kFullM / 32, (kN / 3) / 256, 3));
    describe_kernel(opts,
                    "split_contig_tn32_lat2",
                    qkv_bkn_split_contig_kernel<32, 32, 16, kFullM, kN, kK, 2, 2, 2>,
                    dim3(kFullM / 32, kN / 32, 1));
    describe_kernel(opts,
                    "split_contig_tn64_lat2",
                    qkv_bkn_split_contig_kernel<32, 64, 16, kFullM, kN, kK, 2, 2, 2>,
                    dim3(kFullM / 32, kN / 64, 1));
    describe_kernel(opts,
                    "split_contig_tn128_lat2",
                    qkv_bkn_split_contig_kernel<32, 128, 16, kFullM, kN, kK, 2, 2, 2>,
                    dim3(kFullM / 32, kN / 128, 1));
    describe_kernel(opts,
                    "split_contig_m64_tn64_lat2",
                    qkv_bkn_split_contig_kernel<64, 64, 16, kFullM64, kN, kK, 2, 2, 2>,
                    dim3(kFullM64 / 64, kN / 64, 1));
    describe_kernel(opts,
                    "split_contig_m64_tn128_lat2",
                    qkv_bkn_split_contig_kernel<64, 128, 16, kFullM64, kN, kK, 2, 2, 2>,
                    dim3(kFullM64 / 64, kN / 128, 1));
    describe_kernel(opts,
                    "split_contig_compound2x128_lat2",
                    qkv_bkn_compound2x128_split_contig_kernel<kFullM,
                                                               kN,
                                                               kK,
                                                               2,
                                                              2>,
                    dim3(kFullM / 32, kN / 256, 1));
    describe_kernel(opts,
                    "headmajor_lat2",
                    qkv_bkn_headmajor_kernel<32, 256, 16, kFullM, kN, kK, 2, 2>,
                    dim3(kFullM / 32, kN / 256, 1));
    describe_kernel(opts,
                    "split_contig_tk32_lat2",
                    qkv_bkn_split_contig_kernel<32, 256, 32, kFullM, kN, kK, 2, 2, 2>,
                    dim3(kFullM / 32, kN / 256, 1));
    describe_kernel(opts,
                    "split_contig_tk64_lat2",
                    qkv_bkn_split_contig_kernel<32, 256, 64, kFullM, kN, kK, 2, 2, 2>,
                    dim3(kFullM / 32, kN / 256, 1));
}

void run_all(const Options& opts) {
    std::printf("infer_qkv_bkn M=%d N=%d K=%d fullM=%d\n", kM, kN, kK, kFullM);
    if (opts.describe) {
        describe_all(opts);
        return;
    }
    run_variant(opts, "baseline", launch_baseline);
    run_variant(opts, "lat1", launch_latency<1>);
    run_variant(opts, "lat2", launch_latency<2>);
    run_variant(opts, "lat4", launch_latency<4>);
    run_variant(opts, "a2_b2_s0", launch_latency<2, 2, 0>);
    run_variant(opts, "a2_b1_s0", launch_latency<2, 1, 0>);
    run_variant_shape<32, 32, 16, kFullM>(
        opts,
        "tn32_lat2",
        launch_latency_shape<32, 32, 16, kFullM, 2>);
    run_variant_shape<32, 64, 16, kFullM>(
        opts,
        "tn64_lat2",
        launch_latency_shape<32, 64, 16, kFullM, 2>);
    run_variant_shape<32, 128, 16, kFullM>(
        opts,
        "tn128_lat2",
        launch_latency_shape<32, 128, 16, kFullM, 2>);
    run_variant_shape<16, 256, 16, kFullM>(
        opts,
        "m16_tn256_lat2",
        launch_latency_shape<16, 256, 16, kFullM, 2>);
    run_variant_shape<64, 64, 16, kFullM64>(
        opts,
        "m64_tn64_lat2",
        launch_latency_shape<64, 64, 16, kFullM64, 2>);
    run_variant_shape<64, 32, 16, kFullM64>(
        opts,
        "m64_tn32_lat2",
        launch_latency_shape<64, 32, 16, kFullM64, 2>);
    run_variant_shape<64, 128, 16, kFullM64>(
        opts,
        "m64_tn128_lat2",
        launch_latency_shape<64, 128, 16, kFullM64, 2>);
    run_variant_shape<64, 128, 16, kFullM64>(
        opts,
        "m64_tn128_lat1",
        launch_latency_shape<64, 128, 16, kFullM64, 1>);
    run_variant_shape<64, 128, 16, kFullM64>(
        opts,
        "m64_tn128_lat4",
        launch_latency_shape<64, 128, 16, kFullM64, 4>);
    run_variant_shape<64, 128, 16, kFullM64>(
        opts,
        "m64_tn128_a2_b2_s0",
        launch_latency_shape<64, 128, 16, kFullM64, 2, 2, 0>);
    run_variant_shape<64, 128, 16, kFullM64>(
        opts,
        "m64_tn128_a2_b1_s0",
        launch_latency_shape<64, 128, 16, kFullM64, 2, 1, 0>);
    run_variant_shape<64, 128, 16, kFullM64>(
        opts,
        "m64_tn128_a1_b2_s2",
        launch_latency_shape<64, 128, 16, kFullM64, 1, 2, 2>);
    run_variant_shape_masked_tail<64, 128, 16, kFullM64, kFullM64>(
        opts,
        "m64_tn128_lat1_masktail",
        launch_latency_shape<64, 128, 16, kFullM64, 1>,
        launch_masked_tail_shape<64, 128, 16, kFullM64, 1>);
    run_variant(opts, "compound2x128", launch_compound2x128<>);
    run_variant(opts, "compound2x128_lat2", launch_compound2x128<2>);
    run_variant_split_contig<16>(
        opts, "split_contig_lat2", launch_split_contig<16, 2>);
    run_variant_split_contig<16>(
        opts, "split_contig_a2_b2_s0", launch_split_contig<16, 2, 2, 0>);
    run_variant_split_contig<16>(
        opts, "split_contig_a1_b2_s2", launch_split_contig<16, 1, 2, 2>);
    run_variant_split_contig<16>(
        opts, "split_contig_a2_b1_s2", launch_split_contig<16, 2, 1, 2>);
    run_variant_split_contig<16>(
        opts, "split_contig_a1_b1_s2", launch_split_contig<16, 1, 1, 2>);
    run_variant_split_contig<16>(
        opts, "split_contig_a0_b2_s2", launch_split_contig<16, 0, 2, 2>);
    run_variant_split_contig<16>(
        opts, "split_contig_a2_b0_s2", launch_split_contig<16, 2, 0, 2>);
    run_variant_split_contig<16>(
        opts, "split_contig_gridz_lat2", launch_split_contig_gridz<16, 2>);
    run_variant_split_contig_shape<32, 16>(
        opts, "split_contig_tn32_lat2", launch_split_contig<16, 2, 2, 2, 32>);
    run_variant_split_contig_shape<64, 16>(
        opts, "split_contig_tn64_lat2", launch_split_contig<16, 2, 2, 2, 64>);
    run_variant_split_contig_shape<128, 16>(
        opts, "split_contig_tn128_lat2", launch_split_contig<16, 2, 2, 2, 128>);
    run_variant_split_contig_shape_m<64, 64, 16, kFullM64>(
        opts,
        "split_contig_m64_tn64_lat2",
        launch_split_contig_m<64, 64, 16, kFullM64, 2>);
    run_variant_split_contig_shape_m<64, 128, 16, kFullM64>(
        opts,
        "split_contig_m64_tn128_lat2",
        launch_split_contig_m<64, 128, 16, kFullM64, 2>);
    run_variant_split_contig<16>(
        opts, "split_contig_pairk", launch_split_contig_pairk<16, 0>);
    run_variant_split_contig<16>(
        opts, "split_contig_pairk_lat2", launch_split_contig_pairk<16, 2>);
    run_variant_split_contig<16>(
        opts, "split_contig_compound2x128_lat2", launch_split_contig_compound2x128<2>);
    run_variant_split_contig<16>(
        opts, "headmajor_lat2", launch_headmajor<16, 2>);
    run_variant_split_contig<8>(
        opts, "split_contig_tk8_lat2", launch_split_contig<8, 2>);
    run_variant_split_contig<32>(
        opts, "split_contig_tk32_lat2", launch_split_contig<32, 2>);
    run_variant_split_contig<64>(
        opts, "split_contig_tk64_lat2", launch_split_contig<64, 2>);
    run_variant(opts, "store_one_lat2", launch_store_one<2>);
    run_variant_f32_output(opts, "store_f32_lat2", launch_store_f32<2>);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        run_all(opts);
        CUDA_CHECK(cudaDeviceSynchronize());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
