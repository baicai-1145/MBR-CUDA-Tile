#include "mbr_cuda_tile.h"

#include "cuda_tile.h"
#include <cuda_bf16.h>
#include <cstdint>

namespace cudasep::mbr_tile {
namespace {

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kResidualTileM = 32;
constexpr int kResidualDim = 256;
constexpr int kResidualStaticM = 78048;
constexpr int kResidualExpectedM = 78060;

__tile_global__ void residual_add_bf16_main_kernel(const __nv_bfloat16* __restrict__ x,
                                                   const __nv_bfloat16* __restrict__ residual,
                                                   __nv_bfloat16* __restrict__ out) {
    using AddTile = ct::tile<__nv_bfloat16, ct::shape<kResidualTileM, kResidualDim>>;

    x = ct::assume_aligned(x, 16_ic);
    residual = ct::assume_aligned(residual, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto x_view = ct::partition_view{
        ct::tensor_span{x, ct::shape<kResidualStaticM, kResidualDim>{}},
        ct::shape<kResidualTileM, kResidualDim>{}
    };
    auto residual_view = ct::partition_view{
        ct::tensor_span{residual, ct::shape<kResidualStaticM, kResidualDim>{}},
        ct::shape<kResidualTileM, kResidualDim>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out, ct::shape<kResidualStaticM, kResidualDim>{}},
        ct::shape<kResidualTileM, kResidualDim>{}
    };

    auto [tile_m, tile_n, tile_z] = ct::bid();
    (void)tile_n;
    (void)tile_z;
    AddTile value = x_view.load(tile_m, 0) + residual_view.load(tile_m, 0);
    out_view.store(value, tile_m, 0);
}

__tile_global__ void residual_add_bf16_tail_kernel(const __nv_bfloat16* __restrict__ x,
                                                   const __nv_bfloat16* __restrict__ residual,
                                                   __nv_bfloat16* __restrict__ out) {
    constexpr int TailRows = kResidualExpectedM - kResidualStaticM;
    using AddTile = ct::tile<__nv_bfloat16, ct::shape<kResidualTileM, kResidualDim>>;
    using IndexTile = ct::tile<long long, ct::shape<kResidualTileM, kResidualDim>>;

    x = ct::assume_aligned(x, 16_ic);
    residual = ct::assume_aligned(residual, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    IndexTile local = ct::iota<IndexTile>();
    auto local_rows = local / kResidualDim;
    auto cols = local % kResidualDim;
    auto rows = static_cast<long long>(kResidualStaticM) + local_rows;
    auto mask = local_rows < TailRows;
    auto offsets = rows * kResidualDim + cols;
    AddTile value = ct::load_masked(x + offsets, mask) +
                    ct::load_masked(residual + offsets, mask);
    ct::store_masked(out + offsets, value, mask);
}

}  // namespace

bool try_residual_add_bf16_cutile(const Tensor& x, const Tensor& residual, Tensor& out) {
    if (x.dtype() != DType::BFloat16 || residual.dtype() != DType::BFloat16) return false;
    if (x.shape() != residual.shape() || x.numel() != residual.numel()) return false;
    if (x.numel() % kResidualDim != 0) return false;
    int64_t rows = x.numel() / kResidualDim;
    if (rows != kResidualExpectedM) return false;

    Tensor x_contig = x.is_contiguous() ? x : x.contiguous();
    Tensor residual_contig = residual.is_contiguous() ? residual : residual.contiguous();
    Tensor out_flat = Tensor::empty({rows, kResidualDim}, DType::BFloat16);

    residual_add_bf16_main_kernel<<<kResidualStaticM / kResidualTileM, 1>>>(
        x_contig.data_bf16(),
        residual_contig.data_bf16(),
        out_flat.data_bf16());
    residual_add_bf16_tail_kernel<<<1, 1>>>(
        x_contig.data_bf16(),
        residual_contig.data_bf16(),
        out_flat.data_bf16());
    CUDA_CHECK(cudaGetLastError());

    out = out_flat.reshape(x.shape());
    return true;
}

}  // namespace cudasep::mbr_tile
