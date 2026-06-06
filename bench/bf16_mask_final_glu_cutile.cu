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

constexpr int kM = 1301;
constexpr int kK = 1024;
constexpr int kMaskStride = 7916;
constexpr int kInitTile = 256;
constexpr double kA10gDenseBf16Tflops = 70.0;

struct Options {
    std::string variant = "all";
    int warmup = 20;
    int iters = 300;
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
                "Usage: bench_bf16_mask_final_glu_cutile [options]\n"
                "  --variant NAME  all, model_strided, n24_contig, n24_strided,\n"
                "                  n28_strided, n64_strided, n520_strided\n"
                "  --warmup N      warmup launches, default 20\n"
                "  --iters N       measured launches, default 300\n"
                "  --describe      print CUDA runtime resource diagnostics\n");
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

template <int TM, int TN, int TK, int NOut, int OutStride>
__tile_global__ void linear_glu_static_store_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_nt,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int output_offset,
    bool full_bf16) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using I64OutTile = ct::tile<long long, ct::shape<TM, TN>>;

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    bias = ct::assume_aligned(bias, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<kM, kK>{}},
        ct::shape<TM, TK>{}
    };
    auto b_first_view = ct::partition_view{
        ct::tensor_span{b_nt, ct::shape<kK, NOut>{}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto b_gate_view = ct::partition_view{
        ct::tensor_span{b_nt + static_cast<std::size_t>(NOut) * kK,
                        ct::shape<kK, NOut>{},
                        ct::layout_left{}},
        ct::shape<TK, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    bool full_m_tile = tile_m < kM / TM;
    bool full_n_tile = false;
    if constexpr (NOut >= TN) {
        full_n_tile = tile_n < NOut / TN;
    }

    auto first = ct::full<AccTile>(0.0f);
    auto gate = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{kK / TK})) {
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
    auto rows = static_cast<long long>(tile_m) * TM + local / TN;
    auto cols = static_cast<long long>(tile_n) * TN + local % TN;
    auto valid = (rows < kM) & (cols < NOut);

    first = bf16_round(first);
    gate = bf16_round(gate);
    auto first_bias = ct::element_cast<float>(ct::load_masked(bias + cols, cols < NOut));
    auto gate_bias = ct::element_cast<float>(
        ct::load_masked(bias + static_cast<long long>(NOut) + cols, cols < NOut));
    first = first + first_bias;
    gate = gate + gate_bias;
    first = ct::select(full_bf16, bf16_round(first), first);
    gate = ct::select(full_bf16, bf16_round(gate), gate);
    gate = 1.0f / (1.0f + ct::exp(-gate));
    gate = ct::select(full_bf16, bf16_round(gate), gate);
    auto value = first * gate;
    value = ct::select(full_bf16, bf16_round(value), value);

    ct::store_masked(out + rows * OutStride + output_offset + cols,
                     ct::element_cast<__nv_bfloat16>(value),
                     valid);
}

struct DeviceBuffers {
    __nv_bfloat16* x = nullptr;
    __nv_bfloat16* w = nullptr;
    __nv_bfloat16* b = nullptr;
    __nv_bfloat16* out = nullptr;
};

struct BenchResult {
    float best = 0.0f;
    float median = 0.0f;
    float p90 = 0.0f;
    double tflops = 0.0;
};

void free_buffers(DeviceBuffers& bufs) {
    if (bufs.x) CUDA_CHECK(cudaFree(bufs.x));
    if (bufs.w) CUDA_CHECK(cudaFree(bufs.w));
    if (bufs.b) CUDA_CHECK(cudaFree(bufs.b));
    if (bufs.out) CUDA_CHECK(cudaFree(bufs.out));
    bufs = {};
}

template <int NOut, int TN, int OutStride>
void alloc_buffers(DeviceBuffers& bufs) {
    CUDA_CHECK(cudaMalloc(&bufs.x, static_cast<size_t>(kM) * kK * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&bufs.w, static_cast<size_t>(2 * NOut) * kK * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&bufs.b, static_cast<size_t>(2 * NOut) * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&bufs.out, static_cast<size_t>(kM) * OutStride * sizeof(__nv_bfloat16)));
    init_bf16(bufs.x, static_cast<size_t>(kM) * kK);
    init_bf16(bufs.w, static_cast<size_t>(2 * NOut) * kK);
    init_bf16(bufs.b, static_cast<size_t>(2 * NOut));
    init_bf16(bufs.out, static_cast<size_t>(kM) * OutStride);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <int NOut, int TN, int OutStride>
void launch_variant(const DeviceBuffers& bufs, int output_offset) {
    dim3 grid(ceildiv(kM, 32), ceildiv(NOut, TN), 1);
    linear_glu_static_store_bf16_kernel<32, TN, 64, NOut, OutStride>
        <<<grid, 1>>>(bufs.x, bufs.w, bufs.b, bufs.out, output_offset, true);
}

template <int NOut, int TN, int OutStride>
void describe_variant(const char* name) {
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(
        &attr, linear_glu_static_store_bf16_kernel<32, TN, 64, NOut, OutStride>));
    std::printf(
        "describe %-16s NOut=%d TN=%d OutStride=%d regs=%d smem_static=%zu "
        "local=%zu maxThreads=%d\n",
        name,
        NOut,
        TN,
        OutStride,
        attr.numRegs,
        attr.sharedSizeBytes,
        attr.localSizeBytes,
        attr.maxThreadsPerBlock);
}

template <int NOut, int TN, int OutStride>
BenchResult run_variant(const char* name, const Options& opts, int output_offset) {
    DeviceBuffers bufs;
    alloc_buffers<NOut, TN, OutStride>(bufs);
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < opts.warmup; ++i) {
        launch_variant<NOut, TN, OutStride>(bufs, output_offset);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;
    times.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_variant<NOut, TN, OutStride>(bufs, output_offset);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float best = *std::min_element(times.begin(), times.end());
    float median = percentile(times, 0.5f);
    float p90 = percentile(times, 0.9f);
    double flops = 4.0 * static_cast<double>(kM) * kK * NOut;
    double tflops = flops / (static_cast<double>(best) * 1.0e-3) / 1.0e12;
    BenchResult result{best, median, p90, tflops};
    std::printf(
        "%-16s best=%.4f ms median=%.4f ms p90=%.4f ms useful=%.2f TF/s "
        "roof70=%.1f%% NOut=%d TN=%d stride=%d offset=%d\n",
        name,
        best,
        median,
        p90,
        tflops,
        100.0 * tflops / kA10gDenseBf16Tflops,
        NOut,
        TN,
        OutStride,
        output_offset);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free_buffers(bufs);
    return result;
}

bool selected(const Options& opts, const char* name) {
    return opts.variant == "all" || opts.variant == name;
}

template <int NOut, int Count>
void run_model_width(const Options& opts, double& best_per_chunk_ms, double& median_per_chunk_ms) {
    char name[32];
    std::snprintf(name, sizeof(name), "model_n%d", NOut);
    if constexpr (NOut <= 32) {
        BenchResult result = run_variant<NOut, 32, kMaskStride>(name, opts, 0);
        best_per_chunk_ms += result.best * Count;
        median_per_chunk_ms += result.median * Count;
    } else {
        BenchResult result = run_variant<NOut, 64, kMaskStride>(name, opts, 0);
        best_per_chunk_ms += result.best * Count;
        median_per_chunk_ms += result.median * Count;
    }
}

void run_model_strided_suite(const Options& opts) {
    double best_per_chunk_ms = 0.0;
    double median_per_chunk_ms = 0.0;
    run_model_width<24, 28>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<28, 8>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<36, 6>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<40, 4>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<44, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<52, 6>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<60, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<64, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<68, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<76, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<80, 4>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<88, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<96, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<104, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<112, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<116, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<124, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<132, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<144, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<156, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<164, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<176, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<188, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<200, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<216, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<228, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<244, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<264, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<284, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<304, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<320, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<344, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<372, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<396, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<420, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<452, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<488, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);
    run_model_width<520, 2>(opts, best_per_chunk_ms, median_per_chunk_ms);

    std::printf(
        "model_strided subtotal best_per_chunk=%.4f ms median_per_chunk=%.4f ms "
        "best_test_clean_4chunk=%.4f ms median_test_clean_4chunk=%.4f ms\n",
        best_per_chunk_ms,
        median_per_chunk_ms,
        best_per_chunk_ms * 4.0,
        median_per_chunk_ms * 4.0);
}

void print_gpu() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        print_gpu();
        if (opts.describe) {
            describe_variant<24, 32, 24>("n24_contig");
            describe_variant<24, 32, kMaskStride>("n24_strided");
            describe_variant<28, 32, kMaskStride>("n28_strided");
            describe_variant<64, 64, kMaskStride>("n64_strided");
            describe_variant<520, 64, kMaskStride>("n520_strided");
        }
        if (selected(opts, "n24_contig")) {
            run_variant<24, 32, 24>("n24_contig", opts, 0);
        }
        if (selected(opts, "n24_strided")) {
            run_variant<24, 32, kMaskStride>("n24_strided", opts, 0);
        }
        if (selected(opts, "n28_strided")) {
            run_variant<28, 32, kMaskStride>("n28_strided", opts, 48);
        }
        if (selected(opts, "n64_strided")) {
            run_variant<64, 64, kMaskStride>("n64_strided", opts, 1024);
        }
        if (selected(opts, "n520_strided")) {
            run_variant<520, 64, kMaskStride>("n520_strided", opts, 7396);
        }
        if (opts.variant == "model_strided") {
            run_model_strided_suite(opts);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
