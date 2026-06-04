#include "mbr_cuda_tile.h"

#include "cuda_tile.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace cudasep::mbr_tile {
namespace {

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kTimeAttnN = 1301;
constexpr int kTimeAttnD = 64;
constexpr int kTimeAttnMainN = 1280;
constexpr int kTimeAttnCutileQRows16 = 16;
constexpr int kTimeAttnCutileQRows32 = 32;
constexpr int kTimeAttnCutileQRows64 = 64;
constexpr int kTimeAttnCutileQRows128 = 128;
constexpr int kTimeAttnCutileKTile32 = 32;
constexpr int kTimeAttnCutileKTile64 = 64;
constexpr int kTimeAttnCutileKTile128 = 128;
constexpr float kLog2E = 1.44269504088896340736f;

static inline int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

template <bool UseExp2, typename TileT>
static __tile__ auto softmax_exp(TileT x) {
    if constexpr (UseExp2) {
        return ct::exp2(x * kLog2E);
    }
    return ct::exp(x);
}

template <int QRows, int KTile, int QBlockOffset = 0, bool UseExp2 = false>
__tile_global__ void time_attention1301_cutile_qk_softmax_av_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int KTiles = (kTimeAttnN + KTile - 1) / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kTimeAttnD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kTimeAttnD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block_local, bh, tile_z] = ct::bid();
    (void)tile_z;
    auto q_block = q_block_local + QBlockOffset;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kTimeAttnN * kTimeAttnD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kTimeAttnN * kTimeAttnD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kTimeAttnN * kTimeAttnD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kTimeAttnN * kTimeAttnD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kTimeAttnN, kTimeAttnD>{}},
        ct::shape<QRows, kTimeAttnD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kTimeAttnD, kTimeAttnN>{}, ct::layout_left{}},
        ct::shape<kTimeAttnD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kTimeAttnN, kTimeAttnD>{}},
        ct::shape<KTile, kTimeAttnD>{}
    };

    auto q_tile = q_view.load_masked(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows =
        static_cast<long long>(q_block) * QRows + score_local / KTile;
    auto score_cols_local = score_local % KTile;

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{KTiles})) {
        auto key_cols =
            static_cast<long long>(kt) * KTile + score_cols_local;
        auto valid = (score_rows < kTimeAttnN) && (key_cols < kTimeAttnN);
        ScoreTile scores = ct::mma(q_tile,
                                   k_t_view.load_masked(0, kt),
                                   ct::full<ScoreTile>(0.0f));
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores * scale, neg_inf);

        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load_masked(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    out_acc = out_acc / row_l;
    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kTimeAttnD;
    auto out_cols = out_local % kTimeAttnD;
    auto out_valid = out_rows < kTimeAttnN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kTimeAttnD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows, int KTile, bool UseExp2 = false>
__tile_global__ void time_attention1301_main1280_cutile_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kTimeAttnMainN / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kTimeAttnD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const std::size_t batch_offset =
        static_cast<std::size_t>(bh) * kTimeAttnN * kTimeAttnD;
    const __nv_bfloat16* q_batch = q + batch_offset;
    const __nv_bfloat16* k_batch = k + batch_offset;
    const __nv_bfloat16* v_batch = v + batch_offset;
    __nv_bfloat16* out_batch = out + batch_offset;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kTimeAttnN, kTimeAttnD>{}},
        ct::shape<QRows, kTimeAttnD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kTimeAttnD, kTimeAttnN>{}, ct::layout_left{}},
        ct::shape<kTimeAttnD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kTimeAttnN, kTimeAttnD>{}},
        ct::shape<KTile, kTimeAttnD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kTimeAttnN, kTimeAttnD>{}},
        ct::shape<QRows, kTimeAttnD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kTimeAttnN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f));
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(valid, scores * scale, neg_inf);
    auto tile_m = ct::reduce_max<1>(scores);
    auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
    auto alpha = softmax_exp<UseExp2>(row_m - new_m);
    auto probs_f32 = ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto tile_l = ct::sum<1>(probs_f32);
    out_acc = out_acc * alpha +
              ct::mma(probs_bf16,
                      v_view.load_masked(FullKTiles, 0),
                      ct::full<OutTile>(0.0f));
    row_l = row_l * alpha + tile_l;

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

}  // namespace

void launch_time_attention1301_split_tail_cutile(const Tensor& q,
                                                 const Tensor& k,
                                                 const Tensor& v,
                                                 Tensor& out,
                                                 int64_t bh,
                                                 float scale,
                                                 bool use_k32,
                                                 bool use_tail_q32,
                                                 bool use_exp2) {
    dim3 grid_main(kTimeAttnMainN / kTimeAttnCutileQRows64, static_cast<unsigned int>(bh));
    dim3 grid_tail_q64(1, static_cast<unsigned int>(bh));
    dim3 grid_tail_q32(
        static_cast<unsigned int>(ceildiv(kTimeAttnN - kTimeAttnMainN,
                                          kTimeAttnCutileQRows32)),
        static_cast<unsigned int>(bh));

    if (use_k32) {
        if (use_exp2) {
            time_attention1301_main1280_cutile_kernel<kTimeAttnCutileQRows64,
                                                      kTimeAttnCutileKTile32,
                                                      true>
                <<<grid_main, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                   out.data_bf16(), scale);
        } else {
            time_attention1301_main1280_cutile_kernel<kTimeAttnCutileQRows64,
                                                      kTimeAttnCutileKTile32>
                <<<grid_main, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                   out.data_bf16(), scale);
        }
    } else {
        if (use_exp2) {
            time_attention1301_main1280_cutile_kernel<kTimeAttnCutileQRows64,
                                                      kTimeAttnCutileKTile64,
                                                      true>
                <<<grid_main, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                   out.data_bf16(), scale);
        } else {
            time_attention1301_main1280_cutile_kernel<kTimeAttnCutileQRows64,
                                                      kTimeAttnCutileKTile64>
                <<<grid_main, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                   out.data_bf16(), scale);
        }
    }

    if (use_tail_q32) {
        if (use_exp2) {
            time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows32,
                                                           kTimeAttnCutileKTile32,
                                                           40,
                                                           true>
                <<<grid_tail_q32, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                       out.data_bf16(), scale);
        } else {
            time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows32,
                                                           kTimeAttnCutileKTile32,
                                                           40>
                <<<grid_tail_q32, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                       out.data_bf16(), scale);
        }
    } else if (use_exp2) {
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows64,
                                                       kTimeAttnCutileKTile64,
                                                       20,
                                                       true>
            <<<grid_tail_q64, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                   out.data_bf16(), scale);
    } else {
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows64,
                                                       kTimeAttnCutileKTile64,
                                                       20>
            <<<grid_tail_q64, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                                   out.data_bf16(), scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_time_attention1301_full_cutile(const Tensor& q,
                                           const Tensor& k,
                                           const Tensor& v,
                                           Tensor& out,
                                           int64_t bh,
                                           float scale,
                                           int qrows,
                                           int ktile) {
    if (qrows == kTimeAttnCutileQRows64 && ktile == kTimeAttnCutileKTile128) {
        dim3 grid(static_cast<unsigned int>(ceildiv(kTimeAttnN, kTimeAttnCutileQRows64)),
                  static_cast<unsigned int>(bh));
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows64,
                                                       kTimeAttnCutileKTile128>
            <<<grid, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                          out.data_bf16(), scale);
    } else if (qrows == kTimeAttnCutileQRows64 && ktile == kTimeAttnCutileKTile32) {
        dim3 grid(static_cast<unsigned int>(ceildiv(kTimeAttnN, kTimeAttnCutileQRows64)),
                  static_cast<unsigned int>(bh));
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows64,
                                                       kTimeAttnCutileKTile32>
            <<<grid, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                          out.data_bf16(), scale);
    } else if (qrows == kTimeAttnCutileQRows128 && ktile == kTimeAttnCutileKTile64) {
        dim3 grid(static_cast<unsigned int>(ceildiv(kTimeAttnN, kTimeAttnCutileQRows128)),
                  static_cast<unsigned int>(bh));
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows128,
                                                       kTimeAttnCutileKTile64>
            <<<grid, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                          out.data_bf16(), scale);
    } else if (qrows == kTimeAttnCutileQRows32 && ktile == kTimeAttnCutileKTile64) {
        dim3 grid(static_cast<unsigned int>(ceildiv(kTimeAttnN, kTimeAttnCutileQRows32)),
                  static_cast<unsigned int>(bh));
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows32,
                                                       kTimeAttnCutileKTile64>
            <<<grid, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                          out.data_bf16(), scale);
    } else if (qrows == kTimeAttnCutileQRows16 && ktile == kTimeAttnCutileKTile64) {
        dim3 grid(static_cast<unsigned int>(ceildiv(kTimeAttnN, kTimeAttnCutileQRows16)),
                  static_cast<unsigned int>(bh));
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows16,
                                                       kTimeAttnCutileKTile64>
            <<<grid, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                          out.data_bf16(), scale);
    } else if (qrows == kTimeAttnCutileQRows64 && ktile == kTimeAttnCutileKTile64) {
        dim3 grid(static_cast<unsigned int>(ceildiv(kTimeAttnN, kTimeAttnCutileQRows64)),
                  static_cast<unsigned int>(bh));
        time_attention1301_cutile_qk_softmax_av_kernel<kTimeAttnCutileQRows64,
                                                       kTimeAttnCutileKTile64>
            <<<grid, 1>>>(q.data_bf16(), k.data_bf16(), v.data_bf16(),
                          out.data_bf16(), scale);
    } else {
        throw std::runtime_error("unsupported time attention CUDA Tile shape qrows=" +
                                 std::to_string(qrows) + " ktile=" + std::to_string(ktile));
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudasep::mbr_tile
