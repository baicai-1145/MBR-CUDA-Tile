#include "chunk_cuda_tile.h"

#include "cuda_tile.h"

namespace cudasep::chunk_tile {
namespace {

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kTile = 256;
using I64Tile = ct::tile<long long, ct::shape<kTile>>;

static inline int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

__tile_global__ void accumulate_chunk_kernel(float* __restrict__ dest,
                                             float* __restrict__ weight_sum,
                                             const float* __restrict__ src,
                                             const float* __restrict__ window,
                                             long long total,
                                             int64_t offset,
                                             int64_t chunk_len,
                                             int64_t dest_total_len) {
    dest = ct::assume_aligned(dest, 16_ic);
    weight_sum = ct::assume_aligned(weight_sum, 16_ic);
    src = ct::assume_aligned(src, 16_ic);
    window = ct::assume_aligned(window, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    auto c = idx / chunk_len;
    auto i = idx % chunk_len;
    auto w = ct::load_masked(window + i, in_bounds);
    auto value = ct::load_masked(src + idx, in_bounds) * w;
    auto dest_idx = c * dest_total_len + offset + i;

    ct::atomic_add_masked<ct::memory_order::relaxed>(dest + dest_idx, value, in_bounds);
    ct::atomic_add_masked<ct::memory_order::relaxed>(weight_sum + offset + i, w, in_bounds && (c == 0));
}

__tile_global__ void normalize_by_weights_kernel(float* __restrict__ data,
                                                 const float* __restrict__ weight_sum,
                                                 long long total,
                                                 int64_t total_len) {
    data = ct::assume_aligned(data, 16_ic);
    weight_sum = ct::assume_aligned(weight_sum, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    auto w = ct::load_masked(weight_sum + (idx % total_len), in_bounds);
    auto value = ct::load_masked(data + idx, in_bounds);
    auto normalized = ct::select(w > 1.0e-8f, value / w, value);
    ct::store_masked(data + idx, normalized, in_bounds);
}

}  // namespace

void accumulate_chunk(Tensor& dest,
                      Tensor& weight_sum,
                      const Tensor& src,
                      const Tensor& window,
                      int64_t offset) {
    int64_t chunk_len = src.size(-1);
    int64_t num_channels = src.numel() / chunk_len;
    int64_t total = num_channels * chunk_len;
    accumulate_chunk_kernel<<<(int)ceildiv(total, kTile), 1>>>(
        dest.data_f32(), weight_sum.data_f32(), src.data_f32(), window.data_f32(),
        total, offset, chunk_len, dest.size(-1));
    CUDA_CHECK(cudaGetLastError());
}

void normalize_by_weights(Tensor& data, const Tensor& weight_sum) {
    int64_t total_len = data.size(-1);
    int64_t num_channels = data.numel() / total_len;
    int64_t total = num_channels * total_len;
    normalize_by_weights_kernel<<<(int)ceildiv(total, kTile), 1>>>(
        data.data_f32(), weight_sum.data_f32(), total, total_len);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudasep::chunk_tile
