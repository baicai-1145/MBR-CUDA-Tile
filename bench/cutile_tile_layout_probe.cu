#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                            \
    do {                                                                           \
        cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                \
            throw std::runtime_error(std::string(#call) + " failed: " +            \
                                     cudaGetErrorString(err__));                   \
        }                                                                          \
    } while (0)

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kRows = 32;
constexpr int kCols = 256;
constexpr int kHeads = 4;
constexpr int kHalfDim = 32;
constexpr int kPair = 2;
constexpr int kElems = kRows * kCols;

__tile_global__ void reshape_cat_identity_kernel(const float* __restrict__ src,
                                                 float* __restrict__ dst) {
    using Tile = ct::tile<float, ct::shape<kRows, kCols>>;
    using Tile4 = ct::tile<float, ct::shape<kRows, kHeads, kHalfDim, kPair>>;
    using PairTile = ct::tile<float, ct::shape<kRows, kHeads, kHalfDim, 1>>;

    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    auto src_view = ct::partition_view{
        ct::tensor_span{src, ct::shape<kRows, kCols>{}},
        ct::shape<kRows, kCols>{}
    };
    auto dst_view = ct::partition_view{
        ct::tensor_span{dst, ct::shape<kRows, kCols>{}},
        ct::shape<kRows, kCols>{}
    };

    Tile x = src_view.load(0, 0);
    Tile4 x4 = ct::reshape(x, ct::shape<kRows, kHeads, kHalfDim, kPair>{});
    PairTile even = ct::extract(x4, ct::shape<kRows, kHeads, kHalfDim, 1>{},
                                0, 0, 0, 0);
    PairTile odd = ct::extract(x4, ct::shape<kRows, kHeads, kHalfDim, 1>{},
                               0, 0, 0, 1);
    Tile rebuilt = ct::reshape(ct::cat<3>(even, odd), ct::shape<kRows, kCols>{});
    dst_view.store(rebuilt, 0, 0);
}

__tile_global__ void reshape_cat_rot90_kernel(const float* __restrict__ src,
                                              float* __restrict__ dst) {
    using Tile = ct::tile<float, ct::shape<kRows, kCols>>;
    using Tile4 = ct::tile<float, ct::shape<kRows, kHeads, kHalfDim, kPair>>;
    using PairTile = ct::tile<float, ct::shape<kRows, kHeads, kHalfDim, 1>>;

    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    auto src_view = ct::partition_view{
        ct::tensor_span{src, ct::shape<kRows, kCols>{}},
        ct::shape<kRows, kCols>{}
    };
    auto dst_view = ct::partition_view{
        ct::tensor_span{dst, ct::shape<kRows, kCols>{}},
        ct::shape<kRows, kCols>{}
    };

    Tile x = src_view.load(0, 0);
    Tile4 x4 = ct::reshape(x, ct::shape<kRows, kHeads, kHalfDim, kPair>{});
    PairTile even = ct::extract(x4, ct::shape<kRows, kHeads, kHalfDim, 1>{},
                                0, 0, 0, 0);
    PairTile odd = ct::extract(x4, ct::shape<kRows, kHeads, kHalfDim, 1>{},
                               0, 0, 0, 1);
    Tile rotated = ct::reshape(ct::cat<3>(-odd, even), ct::shape<kRows, kCols>{});
    dst_view.store(rotated, 0, 0);
}

float expected_rot90(const std::vector<float>& src, int row, int col) {
    int dim = col % 64;
    int base = row * kCols + col - (dim % 2);
    float even = src[base];
    float odd = src[base + 1];
    return (dim % 2 == 0) ? -odd : even;
}

void check_identity(const std::vector<float>& src, const std::vector<float>& out) {
    float max_abs = 0.0f;
    int first = -1;
    for (int i = 0; i < kElems; ++i) {
        float diff = std::fabs(out[i] - src[i]);
        if (diff > max_abs) max_abs = diff;
        if (diff != 0.0f && first < 0) first = i;
    }
    std::printf("identity max_abs=%g", max_abs);
    if (first >= 0) {
        std::printf(" first_mismatch row=%d col=%d got=%g expected=%g",
                    first / kCols, first % kCols, out[first], src[first]);
    }
    std::printf("\n");
}

void check_rot90(const std::vector<float>& src, const std::vector<float>& out) {
    float max_abs = 0.0f;
    int first = -1;
    float first_expected = 0.0f;
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            int i = row * kCols + col;
            float expected = expected_rot90(src, row, col);
            float diff = std::fabs(out[i] - expected);
            if (diff > max_abs) max_abs = diff;
            if (diff != 0.0f && first < 0) {
                first = i;
                first_expected = expected;
            }
        }
    }
    std::printf("rot90 max_abs=%g", max_abs);
    if (first >= 0) {
        std::printf(" first_mismatch row=%d col=%d got=%g expected=%g",
                    first / kCols, first % kCols, out[first], first_expected);
    }
    std::printf("\n");
}

}  // namespace

int main() {
    std::vector<float> src(kElems);
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            src[row * kCols + col] = static_cast<float>(row * 100000 + col);
        }
    }

    float* d_src = nullptr;
    float* d_identity = nullptr;
    float* d_rot90 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, kElems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_identity, kElems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rot90, kElems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src, src.data(), kElems * sizeof(float),
                          cudaMemcpyHostToDevice));

    reshape_cat_identity_kernel<<<1, 1>>>(d_src, d_identity);
    reshape_cat_rot90_kernel<<<1, 1>>>(d_src, d_rot90);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> identity(kElems);
    std::vector<float> rot90(kElems);
    CUDA_CHECK(cudaMemcpy(identity.data(), d_identity, kElems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rot90.data(), d_rot90, kElems * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_identity(src, identity);
    check_rot90(src, rot90);

    CUDA_CHECK(cudaFree(d_rot90));
    CUDA_CHECK(cudaFree(d_identity));
    CUDA_CHECK(cudaFree(d_src));
    return 0;
}
