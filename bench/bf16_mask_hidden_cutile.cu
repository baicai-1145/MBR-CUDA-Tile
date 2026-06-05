#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "cuda_tile.h"

namespace {

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr double kA10gDenseBf16Tflops = 70.0;
constexpr int kMActual = 1301;
constexpr int kN = 1024;
constexpr int kInitTile = 256;
using I64InitTile = ct::tile<long long, ct::shape<kInitTile>>;
using F32InitTile = ct::tile<float, ct::shape<kInitTile>>;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(err__));                \
        }                                                                       \
    } while (0)

struct Options {
    std::string variant = "all";
    int k = 1024;
    int warmup = 30;
    int iters = 300;
};

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
            opts.variant = argv[++i];
        } else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            opts.k = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            opts.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            opts.iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: bench_bf16_mask_hidden_cutile [options]\n"
                "  --k N           256 or 1024; default 1024\n"
                "  --variant NAME  all, t32x64x64, t16x64x64, t64x64x64,\n"
                "                  t32x32x64, t32x64x32, t32x64x128,\n"
                "                  t32x128x32, t32x128x64, t32x128x128,\n"
                "                  t16x128x64, default all\n"
                "  --warmup N      warmup launches, default 30\n"
                "  --iters N       measured launches, default 300\n");
            std::exit(0);
        } else {
            throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
        }
    }
    if (opts.k != 256 && opts.k != 1024) {
        throw std::runtime_error("--k must be 256 or 1024");
    }
    return opts;
}

__tile_global__ void init_bf16_kernel(__nv_bfloat16* __restrict__ data,
                                      long long n) {
    data = ct::assume_aligned(data, 16_ic);
    I64InitTile idx =
        static_cast<long long>(ct::bid().x) * kInitTile + ct::iota<I64InitTile>();
    auto in_bounds = idx < n;
    F32InitTile value =
        ct::element_cast<float>((idx * 17LL + 13LL) & 255LL) * 0.0009765625f -
        0.125f;
    ct::store_masked(data + idx, ct::element_cast<__nv_bfloat16>(value), in_bounds);
}

void init_bf16(__nv_bfloat16* data, size_t n) {
    init_bf16_kernel<<<(unsigned int)((n + kInitTile - 1) / kInitTile), 1>>>(
        data, static_cast<long long>(n));
    CUDA_CHECK(cudaGetLastError());
}

template <int TM, int TN, int TK, int MPad, int K>
__tile_global__ void mask_hidden_padded_m_kernel(const __nv_bfloat16* __restrict__ a,
                                                 const __nv_bfloat16* __restrict__ b_nt,
                                                 __nv_bfloat16* __restrict__ c) {
    using AccTile = ct::tile<float, ct::shape<TM, TN>>;
    using I64ATile = ct::tile<long long, ct::shape<TM, TK>>;
    using I64CTile = ct::tile<long long, ct::shape<TM, TN>>;

    static_assert(MPad % TM == 0);
    static_assert(kN % TN == 0);
    static_assert(K % TK == 0);

    a = ct::assume_aligned(a, 16_ic);
    b_nt = ct::assume_aligned(b_nt, 16_ic);
    c = ct::assume_aligned(c, 16_ic);

    auto a_view = ct::partition_view{
        ct::tensor_span{a, ct::shape<MPad, K>{}},
        ct::shape<TM, TK>{}
    };
    auto b_view = ct::partition_view{
        ct::tensor_span{b_nt, ct::shape<K, kN>{}, ct::layout_left{}},
        ct::shape<TK, TN>{}
    };
    auto c_view = ct::partition_view{
        ct::tensor_span{c, ct::shape<MPad, kN>{}},
        ct::shape<TM, TN>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_z;
    bool full_m_tile = tile_m < kMActual / TM;
    auto acc = ct::full<AccTile>(0.0f);
    for (auto kk : ct::irange(std::size_t{0}, std::size_t{K / TK})) {
        auto b_tile = b_view.load(kk, tile_n);
        if (full_m_tile) {
            acc = ct::mma(a_view.load(tile_m, kk), b_tile, acc);
        } else {
            I64ATile local = ct::iota<I64ATile>();
            auto rows = static_cast<long long>(tile_m) * TM + local / TK;
            auto cols = static_cast<long long>(kk) * TK + local % TK;
            auto valid = rows < kMActual;
            auto a_tile = ct::load_masked(a + rows * K + cols, valid);
            acc = ct::mma(a_tile, b_tile, acc);
        }
    }

    auto out = ct::element_cast<__nv_bfloat16>(acc);
    if (full_m_tile) {
        c_view.store(out, tile_m, tile_n);
    } else {
        I64CTile local = ct::iota<I64CTile>();
        auto rows = static_cast<long long>(tile_m) * TM + local / TN;
        auto cols = static_cast<long long>(tile_n) * TN + local % TN;
        auto valid = rows < kMActual;
        ct::store_masked(c + rows * kN + cols, out, valid);
    }
}

template <int TM, int TN, int TK, int K>
void run_variant(const Options& opts, const char* name) {
    if (opts.variant != "all" && opts.variant != name) return;
    if (K != opts.k) return;

    constexpr int MPad = ((kMActual + TM - 1) / TM) * TM;
    size_t a_elems = static_cast<size_t>(kMActual) * K;
    size_t b_elems = static_cast<size_t>(kN) * K;
    size_t c_elems = static_cast<size_t>(kMActual) * kN;
    double gib = (static_cast<double>(a_elems + b_elems + c_elems) *
                  sizeof(__nv_bfloat16)) /
                 (1024.0 * 1024.0 * 1024.0);
    dim3 grid(MPad / TM, kN / TN, 1);

    __nv_bfloat16* d_a = nullptr;
    __nv_bfloat16* d_b = nullptr;
    __nv_bfloat16* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_a, a_elems);
    init_bf16(d_b, b_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto launch = [&]() {
        mask_hidden_padded_m_kernel<TM, TN, TK, MPad, K><<<grid, 1>>>(d_a, d_b, d_c);
    };
    for (int i = 0; i < opts.warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times;
    times.reserve(opts.iters);
    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_c, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    std::sort(times.begin(), times.end());
    float best_ms = times.front();
    float median_ms = times[times.size() / 2];
    double useful_flops = 2.0 * kMActual * kN * K;
    double issued_flops = 2.0 * MPad * kN * K;
    double useful_tflops = useful_flops / (best_ms * 1.0e-3) / 1.0e12;
    double issued_tflops = issued_flops / (best_ms * 1.0e-3) / 1.0e12;
    std::printf(
        "%-14s K=%d tile=%dx%dx%d grid=(%u,%u) MPad=%d mem=%.3f GiB "
        "best=%.4f ms median=%.4f ms useful=%.2f TF/s issued=%.2f TF/s "
        "roof=%.1f%% checksum=%.4f\n",
        name, K, TM, TN, TK, grid.x, grid.y, MPad, gib, best_ms, median_ms,
        useful_tflops, issued_tflops, useful_tflops * 100.0 / kA10gDenseBf16Tflops,
        __bfloat162float(checksum));
}

template <int K>
void run_k(const Options& opts) {
    run_variant<32, 64, 64, K>(opts, "t32x64x64");
    run_variant<16, 64, 64, K>(opts, "t16x64x64");
    run_variant<64, 64, 64, K>(opts, "t64x64x64");
    run_variant<32, 32, 64, K>(opts, "t32x32x64");
    run_variant<32, 64, 32, K>(opts, "t32x64x32");
    run_variant<32, 64, 128, K>(opts, "t32x64x128");
    run_variant<32, 128, 32, K>(opts, "t32x128x32");
    run_variant<32, 128, 64, K>(opts, "t32x128x64");
    run_variant<32, 128, 128, K>(opts, "t32x128x128");
    run_variant<16, 128, 64, K>(opts, "t16x128x64");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        if (opts.k == 256) {
            run_k<256>(opts);
        } else {
            run_k<1024>(opts);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
