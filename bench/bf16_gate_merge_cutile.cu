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

#define CUDA_CHECK(call)                                                        \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            throw std::runtime_error(std::string(#call) + " failed: " +        \
                                     cudaGetErrorString(err__));               \
        }                                                                      \
    } while (0)

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kInitTile = 256;
constexpr int kBatches = 60;
constexpr int kNTime = 1301;
constexpr int kHeads = 8;
constexpr int kD = 64;
constexpr int kMerged = kHeads * kD;
constexpr int kTokensTime = kBatches * kNTime;
constexpr int kFreqN = 60;
constexpr int kTokensFreq = kBatches * kFreqN;

using I64InitTile = ct::tile<long long, ct::shape<kInitTile>>;
using F32InitTile = ct::tile<float, ct::shape<kInitTile>>;

struct Options {
    std::string variant = "all";
    int tokens = kTokensTime;
    int n_tokens = kNTime;
    int warmup = 30;
    int iters = 300;
    bool describe = false;
    bool compare_baseline = false;
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
        } else if (std::strcmp(argv[i], "--shape") == 0) {
            const char* shape = need_value(argv[i]);
            if (std::strcmp(shape, "time") == 0) {
                opts.tokens = kTokensTime;
                opts.n_tokens = kNTime;
            } else if (std::strcmp(shape, "freq") == 0) {
                opts.tokens = kTokensFreq;
                opts.n_tokens = kFreqN;
            } else {
                throw std::runtime_error(std::string("unknown --shape: ") + shape);
            }
        } else if (std::strcmp(argv[i], "--warmup") == 0) {
            opts.warmup = parse_positive_int(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            opts.iters = parse_positive_int(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--describe") == 0) {
            opts.describe = true;
        } else if (std::strcmp(argv[i], "--compare-baseline") == 0) {
            opts.compare_baseline = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: bench_bf16_gate_merge_cutile [options]\n"
                "  --variant NAME      all, token1, token2, token4, flat256\n"
                "  --shape NAME        time or freq, default time\n"
                "  --warmup N          warmup launches, default 30\n"
                "  --iters N           measured launches, default 300\n"
                "  --describe          print CUDA runtime resource diagnostics\n"
                "  --compare-baseline  compare output against token1\n");
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

template <bool FullBF16, typename T>
static __tile__ auto maybe_bf16_round(T value) {
    if constexpr (FullBF16) {
        return bf16_round(value);
    } else {
        return value;
    }
}

__tile_global__ void fill_bf16_kernel(__nv_bfloat16* __restrict__ dst, long long total) {
    dst = ct::assume_aligned(dst, 16_ic);
    I64InitTile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64InitTile>();
    auto in_bounds = idx < total;
    F32InitTile values = 0.125f +
        ct::element_cast<float>((idx * 13LL + 7LL) & 255LL) * 0.0009765625f;
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

void init_bf16(__nv_bfloat16* ptr, size_t elems) {
    fill_bf16_kernel<<<ceildiv(static_cast<long long>(elems), kInitTile), 1>>>(
        ptr, static_cast<long long>(elems));
    CUDA_CHECK(cudaGetLastError());
}

template <bool FullBF16 = false>
__tile_global__ void gate_merge_token1_kernel(const __nv_bfloat16* __restrict__ attn,
                                              const __nv_bfloat16* __restrict__ gates,
                                              __nv_bfloat16* __restrict__ merged,
                                              int tokens,
                                              int n_tokens) {
    using I64Tile = ct::tile<long long, ct::shape<kMerged>>;
    using F32Tile = ct::tile<float, ct::shape<kMerged>>;

    attn = ct::assume_aligned(attn, 16_ic);
    gates = ct::assume_aligned(gates, 16_ic);
    merged = ct::assume_aligned(merged, 16_ic);

    int token = static_cast<int>(ct::bid().x);
    I64Tile e = ct::iota<I64Tile>();
    auto in_bounds = token < tokens;

    int n = token % n_tokens;
    int b = token / n_tokens;
    auto h = e / kD;
    auto d = e % kD;

    auto src_idx = ((static_cast<long long>(b) * kHeads + h) * n_tokens + n) * kD + d;
    auto gate_idx = static_cast<long long>(token) * kHeads + h;
    auto dst_idx = static_cast<long long>(token) * kMerged + e;

    F32Tile attn_values =
        ct::element_cast<float>(ct::load_masked(attn + src_idx, in_bounds));
    F32Tile gate_values =
        ct::element_cast<float>(ct::load_masked(gates + gate_idx, in_bounds));
    attn_values = maybe_bf16_round<FullBF16>(attn_values);
    gate_values = maybe_bf16_round<FullBF16>(gate_values);
    auto values = maybe_bf16_round<FullBF16>(attn_values * gate_values);
    ct::store_masked(merged + dst_idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

template <int TokensPerCTA, bool FullBF16 = false>
__tile_global__ void gate_merge_tokens_kernel(const __nv_bfloat16* __restrict__ attn,
                                              const __nv_bfloat16* __restrict__ gates,
                                              __nv_bfloat16* __restrict__ merged,
                                              int tokens,
                                              int n_tokens) {
    using I64Tile = ct::tile<long long, ct::shape<TokensPerCTA, kMerged>>;
    using F32Tile = ct::tile<float, ct::shape<TokensPerCTA, kMerged>>;

    attn = ct::assume_aligned(attn, 16_ic);
    gates = ct::assume_aligned(gates, 16_ic);
    merged = ct::assume_aligned(merged, 16_ic);

    int token_base = static_cast<int>(ct::bid().x) * TokensPerCTA;
    I64Tile local = ct::iota<I64Tile>();
    auto local_token = local / kMerged;
    auto e = local % kMerged;
    auto token = token_base + local_token;
    auto in_bounds = token < tokens;

    auto n = token % n_tokens;
    auto b = token / n_tokens;
    auto h = e / kD;
    auto d = e % kD;

    auto src_idx = ((b * kHeads + h) * n_tokens + n) * kD + d;
    auto gate_idx = token * kHeads + h;
    auto dst_idx = token * kMerged + e;

    F32Tile attn_values =
        ct::element_cast<float>(ct::load_masked(attn + src_idx, in_bounds));
    F32Tile gate_values =
        ct::element_cast<float>(ct::load_masked(gates + gate_idx, in_bounds));
    attn_values = maybe_bf16_round<FullBF16>(attn_values);
    gate_values = maybe_bf16_round<FullBF16>(gate_values);
    auto values = maybe_bf16_round<FullBF16>(attn_values * gate_values);
    ct::store_masked(merged + dst_idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

template <bool FullBF16 = false>
__tile_global__ void gate_merge_flat256_kernel(const __nv_bfloat16* __restrict__ attn,
                                               const __nv_bfloat16* __restrict__ gates,
                                               __nv_bfloat16* __restrict__ merged,
                                               long long total,
                                               int n_tokens) {
    using I64Tile = ct::tile<long long, ct::shape<kInitTile>>;
    using F32Tile = ct::tile<float, ct::shape<kInitTile>>;

    attn = ct::assume_aligned(attn, 16_ic);
    gates = ct::assume_aligned(gates, 16_ic);
    merged = ct::assume_aligned(merged, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto d = idx % kD;
    auto n = (idx / kD) % n_tokens;
    auto h = (idx / ((long long)kD * n_tokens)) % kHeads;
    auto b = idx / ((long long)kD * n_tokens * kHeads);
    auto token = b * n_tokens + n;
    auto merged_idx = token * kMerged + h * kD + d;
    auto gate_idx = token * kHeads + h;

    F32Tile attn_values =
        ct::element_cast<float>(ct::load_masked(attn + idx, in_bounds));
    F32Tile gate_values =
        ct::element_cast<float>(ct::load_masked(gates + gate_idx, in_bounds));
    attn_values = maybe_bf16_round<FullBF16>(attn_values);
    gate_values = maybe_bf16_round<FullBF16>(gate_values);
    auto values = maybe_bf16_round<FullBF16>(attn_values * gate_values);
    ct::store_masked(merged + merged_idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
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
    double waves = static_cast<double>(grid.x) *
                   static_cast<double>(grid.y) *
                   static_cast<double>(std::max(1u, grid.z)) /
                   static_cast<double>(prop.multiProcessorCount);

    std::printf(
        "%-10s grid=(%u,%u,%u) waves/SM=%.1f attr_regs=%d "
        "static_shared=%zuB shared_limit=%d occupancy_active_cta_per_sm=%d "
        "max_threads_per_block=%d local=%zuB const=%zuB ptx=%d binary=%d\n",
        name,
        grid.x,
        grid.y,
        grid.z,
        waves,
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
    std::printf("shape: tokens=%d n_tokens=%d heads=%d dim=%d merged=%d\n",
                opts.tokens, opts.n_tokens, kHeads, kD, kMerged);

    describe_kernel(opts, "token1", gate_merge_token1_kernel<false>,
                    dim3(opts.tokens, 1, 1));
    describe_kernel(opts, "token2", gate_merge_tokens_kernel<2, false>,
                    dim3(ceildiv(opts.tokens, 2), 1, 1));
    describe_kernel(opts, "token4", gate_merge_tokens_kernel<4, false>,
                    dim3(ceildiv(opts.tokens, 4), 1, 1));
    describe_kernel(opts, "flat256", gate_merge_flat256_kernel<false>,
                    dim3(ceildiv((long long)opts.tokens * kMerged, kInitTile), 1, 1));
}

template <typename Launch>
void run_variant(const Options& opts, const char* name, Launch launch) {
    if (opts.variant != "all" && opts.variant != name) return;

    size_t attn_elems = static_cast<size_t>(opts.tokens) * kMerged;
    size_t gate_elems = static_cast<size_t>(opts.tokens) * kHeads;
    size_t out_elems = attn_elems;
    double unique_gib = static_cast<double>((attn_elems + gate_elems + out_elems) *
                                            sizeof(__nv_bfloat16)) /
                        (1024.0 * 1024.0 * 1024.0);
    double tile_logic_gib = static_cast<double>((attn_elems + attn_elems + out_elems) *
                                                sizeof(__nv_bfloat16)) /
                            (1024.0 * 1024.0 * 1024.0);

    __nv_bfloat16* d_attn = nullptr;
    __nv_bfloat16* d_gates = nullptr;
    __nv_bfloat16* d_out = nullptr;
    __nv_bfloat16* d_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_attn, attn_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_gates, gate_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(__nv_bfloat16)));
    if (opts.compare_baseline) {
        CUDA_CHECK(cudaMalloc(&d_ref, out_elems * sizeof(__nv_bfloat16)));
    }
    init_bf16(d_attn, attn_elems);
    init_bf16(d_gates, gate_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (opts.compare_baseline) {
        gate_merge_token1_kernel<false><<<opts.tokens, 1>>>(
            d_attn, d_gates, d_ref, opts.tokens, opts.n_tokens);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < opts.warmup; ++i) {
        launch(d_attn, d_gates, d_out);
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
        launch(d_attn, d_gates, d_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.compare_baseline) {
        std::vector<__nv_bfloat16> ref(out_elems);
        std::vector<__nv_bfloat16> out(out_elems);
        CUDA_CHECK(cudaMemcpy(ref.data(), d_ref, out_elems * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_elems * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        double max_abs = 0.0;
        double sum_sq = 0.0;
        double ref_sq = 0.0;
        for (size_t i = 0; i < out_elems; ++i) {
            double r = __bfloat162float(ref[i]);
            double o = __bfloat162float(out[i]);
            double diff = o - r;
            max_abs = std::max(max_abs, std::abs(diff));
            sum_sq += diff * diff;
            ref_sq += r * r;
        }
        double rms = std::sqrt(sum_sq / static_cast<double>(out_elems));
        double ref_rms = std::sqrt(ref_sq / static_cast<double>(out_elems));
        double rel_rms = ref_rms > 0.0 ? rms / ref_rms : 0.0;
        std::printf("  compare_vs_token1 max_abs=%.9g rms=%.9g rel_rms=%.9g\n",
                    max_abs, rms, rel_rms);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_out, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_attn));
    CUDA_CHECK(cudaFree(d_gates));
    CUDA_CHECK(cudaFree(d_out));
    if (d_ref) {
        CUDA_CHECK(cudaFree(d_ref));
    }

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double unique_gibs = unique_gib / (static_cast<double>(best_ms) * 1.0e-3);
    double tile_logic_gibs = tile_logic_gib / (static_cast<double>(best_ms) * 1.0e-3);
    std::printf(
        "%-10s tokens=%d n_tokens=%d best=%.4f ms median=%.4f ms "
        "unique=%.1f GiB/s tile_logic=%.1f GiB/s checksum=%.4f\n",
        name,
        opts.tokens,
        opts.n_tokens,
        best_ms,
        median_ms,
        unique_gibs,
        tile_logic_gibs,
        __bfloat162float(checksum));
}

void run_all(const Options& opts) {
    if (opts.describe) {
        describe_all(opts);
        return;
    }

    run_variant(opts, "token1", [&](auto a, auto g, auto o) {
        gate_merge_token1_kernel<false><<<opts.tokens, 1>>>(
            a, g, o, opts.tokens, opts.n_tokens);
    });
    run_variant(opts, "token2", [&](auto a, auto g, auto o) {
        gate_merge_tokens_kernel<2, false><<<ceildiv(opts.tokens, 2), 1>>>(
            a, g, o, opts.tokens, opts.n_tokens);
    });
    run_variant(opts, "token4", [&](auto a, auto g, auto o) {
        gate_merge_tokens_kernel<4, false><<<ceildiv(opts.tokens, 4), 1>>>(
            a, g, o, opts.tokens, opts.n_tokens);
    });
    run_variant(opts, "flat256", [&](auto a, auto g, auto o) {
        gate_merge_flat256_kernel<false>
            <<<ceildiv((long long)opts.tokens * kMerged, kInitTile), 1>>>(
                a, g, o, (long long)opts.tokens * kMerged, opts.n_tokens);
    });
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        if (opts.compare_baseline && opts.variant == "all") {
            throw std::runtime_error("--compare-baseline requires --variant NAME");
        }
        run_all(opts);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
