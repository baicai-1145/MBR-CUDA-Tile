#include "cuda_tile.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ct = cuda::tiles;
using namespace ct::literals;

#define CUDA_CHECK(call)                                                             \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            throw std::runtime_error(std::string(#call) + " failed: " +             \
                                     cudaGetErrorString(err__));                    \
        }                                                                           \
    } while (0)

constexpr int kN = 1301;
constexpr int kNPad = 1344;
constexpr int kNMain = 1280;
constexpr int kD = 64;
constexpr int kBH = 480;
constexpr int kKTile = 64;
constexpr int kInitTile = 256;
constexpr float kLog2E = 1.44269504088896340736f;

struct Options {
    int warmup = 1;
    int iters = 3;
    bool validate = false;
};

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
        if (std::strcmp(argv[i], "--warmup") == 0) {
            opts.warmup = parse_int_arg(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            opts.iters = parse_int_arg(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--validate") == 0) {
            opts.validate = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: bench_bf16_time_attention_cutile [options]\n"
                "  --warmup N  warmup launches, default 1\n"
                "  --iters N   measured launches, default 3\n"
                "  --validate  compare several BH0 rows against CPU reference\n");
            std::exit(0);
        } else {
            throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
        }
    }
    return opts;
}

int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

template <bool UseExp2, typename TileT>
static __tile__ auto softmax_exp(TileT x) {
    if constexpr (UseExp2) {
        return ct::exp2(x * kLog2E);
    }
    return ct::exp(x);
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
    using I64Tile = ct::tile<long long, ct::shape<kInitTile>>;
    using F32Tile = ct::tile<float, ct::shape<kInitTile>>;
    dst = ct::assume_aligned(dst, 16_ic);
    I64Tile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    F32Tile values =
        0.125f + ct::element_cast<float>((idx * 17LL) & 1023LL) * 0.000244140625f;
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

__tile_global__ void transpose_k_nd_to_dn_kernel(const __nv_bfloat16* __restrict__ src,
                                                 __nv_bfloat16* __restrict__ dst,
                                                 long long total) {
    using I64Tile = ct::tile<long long, ct::shape<256>>;
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    I64Tile idx = (long long)ct::bid().x * 256 + ct::iota<I64Tile>();
    auto valid = idx < total;
    auto d = idx % kD;
    auto n = (idx / kD) % kN;
    auto bh = idx / ((long long)kN * kD);
    auto dst_idx = (bh * kD + d) * kN + n;
    auto values = ct::load_masked(src + idx, valid);
    ct::store_masked(dst + dst_idx, values, valid);
}

template <int QRows>
__tile_global__ void time_attention1301_cutile_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    using ScoreTile = ct::tile<float, ct::shape<QRows, kKTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, kKTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kNPad * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kNPad, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kNPad>{}, ct::layout_left{}},
        ct::shape<kD, kKTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kNPad, kD>{}},
        ct::shape<kKTile, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / kKTile;
    auto score_cols_local = score_local % kKTile;

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{kNPad / kKTile})) {
        auto key_cols = static_cast<long long>(kt) * kKTile + score_cols_local;
        auto valid = (score_rows < kN) && (key_cols < kN);
        ScoreTile scores = ct::mma(q_tile, k_t_view.load(0, kt),
                                   ct::full<ScoreTile>(0.0f));
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores * scale, neg_inf);

        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = ct::exp(row_m - new_m);
        auto probs_f32 = ct::select(valid, ct::exp(scores - new_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(ct::element_cast<float>(probs_bf16));

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16, v_view.load(kt, 0), ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    out_acc = out_acc / row_l;
    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_valid = out_rows < kN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows, int KTile>
__tile_global__ void time_attention1301_cutile_masked_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int KTiles = (kN + KTile - 1) / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };

    auto q_tile = q_view.load_masked(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / KTile;
    auto score_cols_local = score_local % KTile;

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{KTiles})) {
        auto key_cols = static_cast<long long>(kt) * KTile + score_cols_local;
        auto valid = (score_rows < kN) && (key_cols < kN);
        ScoreTile scores = ct::mma(q_tile, k_t_view.load_masked(0, kt),
                                   ct::full<ScoreTile>(0.0f));
        auto neg_inf = scores * 0.0f - 3.402823466e38f;
        scores = ct::select(valid, scores * scale, neg_inf);

        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = ct::exp(row_m - new_m);
        auto probs_f32 = ct::select(valid, ct::exp(scores - new_m), scores * 0.0f);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(ct::element_cast<float>(probs_bf16));

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16, v_view.load_masked(kt, 0), ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    out_acc = out_acc / row_l;
    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_valid = out_rows < kN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows, int KTile>
__tile_global__ void time_attention1301_cutile_masked_score_av_lb_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int KTiles = (kN + KTile - 1) / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };

    auto q_tile = q_view.load_masked(q_block, 0);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / KTile;
    auto score_cols_local = score_local % KTile;

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{KTiles})) {
        auto key_cols = static_cast<long long>(kt) * KTile + score_cols_local;
        auto valid = (score_rows < kN) && (key_cols < kN);
        ScoreTile scores = ct::mma(q_tile, k_t_view.load_masked(0, kt),
                                   ct::full<ScoreTile>(0.0f));
        auto score_values = ct::select(valid, scores * scale, scores * 0.0f);
        auto score_bf16 = ct::element_cast<__nv_bfloat16>(score_values);
        out_acc = out_acc + ct::mma(score_bf16,
                                    v_view.load_masked(kt, 0),
                                    ct::full<OutTile>(0.0f));
    }

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_valid = out_rows < kN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows>
__tile_global__ void time_attention1301_main1280_av_const_kernel(
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out) {
    constexpr int KTile = 64;
    constexpr int FullKTiles = kNMain / KTile;
    using ProbTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ProbTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(ct::full<ProbTile>(0.125f));
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        out_acc = out_acc +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
    }

    I64ProbTile prob_local = ct::iota<I64ProbTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + prob_local % KTile;
    auto valid = key_cols < kN;
    auto tail_probs = ct::element_cast<__nv_bfloat16>(
        ct::select(valid, ct::full<ProbTile>(0.125f), ct::full<ProbTile>(0.0f)));
    out_acc = out_acc +
              ct::mma(tail_probs,
                      v_view.load_masked(FullKTiles, 0),
                      ct::full<OutTile>(0.0f));

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc), q_block, 0);
}

template <int QRows>
__tile_global__ void time_attention1301_main1280_qk_only_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int KTile = 64;
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    ScoreTile score_acc = ct::full<ScoreTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        score_acc = score_acc +
                    ct::mma(q_tile,
                            k_t_view.load(0, kt),
                            ct::full<ScoreTile>(0.0f)) * scale;
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto tail_scores = ct::mma(q_tile,
                               k_t_view.load_masked(0, FullKTiles),
                               ct::full<ScoreTile>(0.0f)) * scale;
    score_acc = score_acc + ct::select(valid, tail_scores, tail_scores * 0.0f);

    out_view.store(ct::element_cast<__nv_bfloat16>(score_acc), q_block, 0);
}

template <int QRows>
__tile_global__ void time_attention1301_main1280_qk_only_kt_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_t,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int KTile = 64;
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    q = ct::assume_aligned(q, 16_ic);
    k_t = ct::assume_aligned(k_t, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_t_batch =
        k_t + static_cast<std::size_t>(bh) * kD * kN;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_t_batch, ct::shape<kD, kN>{}},
        ct::shape<kD, KTile>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    ScoreTile score_acc = ct::full<ScoreTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        score_acc = score_acc +
                    ct::mma(q_tile,
                            k_t_view.load(0, kt),
                            ct::full<ScoreTile>(0.0f)) * scale;
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto tail_scores = ct::mma(q_tile,
                               k_t_view.load_masked(0, FullKTiles),
                               ct::full<ScoreTile>(0.0f)) * scale;
    score_acc = score_acc + ct::select(valid, tail_scores, tail_scores * 0.0f);

    out_view.store(ct::element_cast<__nv_bfloat16>(score_acc), q_block, 0);
}

template <int QRows>
__tile_global__ void time_attention1301_main1280_qk_store_p_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ p,
    float scale) {
    constexpr int KTile = 64;
    constexpr int QBlocks = kNMain / QRows;
    constexpr int FullKTiles = kNMain / KTile;
    constexpr int KTiles = (kN + KTile - 1) / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    p = ct::assume_aligned(p, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* p_block =
        p + ((static_cast<std::size_t>(bh) * QBlocks + q_block) *
             KTiles * QRows * KTile);

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    I64ScoreTile local = ct::iota<I64ScoreTile>();
    auto all_valid = local >= 0LL;

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        ct::store_masked(p_block + static_cast<std::size_t>(kt) * QRows * KTile + local,
                         ct::element_cast<__nv_bfloat16>(scores),
                         all_valid);
    }

    auto key_cols = static_cast<long long>(FullKTiles) * KTile + local % KTile;
    auto valid = key_cols < kN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f));
    auto p_values = ct::select(valid, scores * scale, scores * 0.0f);
    ct::store_masked(p_block + static_cast<std::size_t>(FullKTiles) * QRows * KTile + local,
                     ct::element_cast<__nv_bfloat16>(p_values),
                     all_valid);
}

template <int QRows>
__tile_global__ void time_attention1301_main1280_av_load_p_kernel(
    const __nv_bfloat16* __restrict__ p,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out) {
    constexpr int KTile = 64;
    constexpr int QBlocks = kNMain / QRows;
    constexpr int FullKTiles = kNMain / KTile;
    constexpr int KTiles = (kN + KTile - 1) / KTile;
    using ProbTile = ct::tile<__nv_bfloat16, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ProbTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    p = ct::assume_aligned(p, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* p_block =
        p + ((static_cast<std::size_t>(bh) * QBlocks + q_block) *
             KTiles * QRows * KTile);
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    I64ProbTile local = ct::iota<I64ProbTile>();
    auto all_valid = local >= 0LL;
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        ProbTile probs = ct::load_masked(
            p_block + static_cast<std::size_t>(kt) * QRows * KTile + local,
            all_valid);
        out_acc = out_acc +
                  ct::mma(probs,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
    }

    ProbTile probs = ct::load_masked(
        p_block + static_cast<std::size_t>(FullKTiles) * QRows * KTile + local,
        all_valid);
    out_acc = out_acc +
              ct::mma(probs,
                      v_view.load_masked(FullKTiles, 0),
                      ct::full<OutTile>(0.0f));

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc), q_block, 0);
}

__tile_global__ void time_attention1301_q64k64_main1280_score_av_lb_split_d32_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int QRows = 64;
    constexpr int KTile = 64;
    constexpr int DTile = 32;
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, DTile>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, DTile>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, DTile>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    OutTile out_acc0 = ct::full<OutTile>(0.0f);
    OutTile out_acc1 = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(scores);
        out_acc0 = out_acc0 +
                   ct::mma(probs_bf16,
                           v_view.load(kt, 0),
                           ct::full<OutTile>(0.0f));
        out_acc1 = out_acc1 +
                   ct::mma(probs_bf16,
                           v_view.load(kt, 1),
                           ct::full<OutTile>(0.0f));
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f));
    auto score_values = ct::select(valid, scores * scale, scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(score_values);
    out_acc0 = out_acc0 +
               ct::mma(probs_bf16,
                       v_view.load_masked(FullKTiles, 0),
                       ct::full<OutTile>(0.0f));
    out_acc1 = out_acc1 +
               ct::mma(probs_bf16,
                       v_view.load_masked(FullKTiles, 1),
                       ct::full<OutTile>(0.0f));

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc0), q_block, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc1), q_block, 1);
}

__tile_global__ void time_attention1301_q64k64_main1280_score_av_lb_prescale_q_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int QRows = 64;
    constexpr int KTile = 64;
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = ct::element_cast<__nv_bfloat16>(
        ct::element_cast<float>(q_view.load(q_block, 0)) * scale);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f));
        out_acc = out_acc +
                  ct::mma(ct::element_cast<__nv_bfloat16>(scores),
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f));
    auto score_values = ct::select(valid, scores, scores * 0.0f);
    out_acc = out_acc +
              ct::mma(ct::element_cast<__nv_bfloat16>(score_values),
                      v_view.load_masked(FullKTiles, 0),
                      ct::full<OutTile>(0.0f));

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc), q_block, 0);
}

template <int QRows, int KTile = 64, bool SumF32 = false, bool UseExp2 = false,
          bool ScoreBf16 = false, bool PrescaleQ = false>
__tile_global__ void time_attention1301_main1280_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    if constexpr (PrescaleQ) {
        q_tile = ct::element_cast<__nv_bfloat16>(
            ct::element_cast<float>(q_tile) * scale);
    }
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f));
        if constexpr (!PrescaleQ) {
            scores = scores * scale;
        }
        if constexpr (ScoreBf16) {
            scores = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(scores));
        }
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<UseExp2>(row_m - new_m);
        auto probs_f32 = softmax_exp<UseExp2>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);
        if constexpr (!SumF32) {
            tile_l = ct::sum<1>(ct::element_cast<float>(probs_bf16));
        }

        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f));
    if constexpr (!PrescaleQ) {
        scores = scores * scale;
    }
    if constexpr (ScoreBf16) {
        scores = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(scores));
    }
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(valid, scores, neg_inf);
    auto tile_m = ct::reduce_max<1>(scores);
    auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
    auto alpha = softmax_exp<UseExp2>(row_m - new_m);
    auto probs_f32 = ct::select(valid, softmax_exp<UseExp2>(scores - new_m), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto tile_l = ct::sum<1>(probs_f32);
    if constexpr (!SumF32) {
        tile_l = ct::sum<1>(ct::element_cast<float>(probs_bf16));
    }
    out_acc = out_acc * alpha +
              ct::mma(probs_bf16,
                      v_view.load_masked(FullKTiles, 0),
                      ct::full<OutTile>(0.0f));
    row_l = row_l * alpha + tile_l;

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc / row_l), q_block, 0);
}

__tile_global__ void time_attention1301_main1280_q64k32_exp2_split_d32_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int QRows = 64;
    constexpr int KTile = 32;
    constexpr int DTile = 32;
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, DTile>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, DTile>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, DTile>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc0 = ct::full<OutTile>(0.0f);
    OutTile out_acc1 = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = softmax_exp<true>(row_m - new_m);
        auto probs_f32 = softmax_exp<true>(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        out_acc0 = out_acc0 * alpha +
                   ct::mma(probs_bf16,
                           v_view.load(kt, 0),
                           ct::full<OutTile>(0.0f));
        out_acc1 = out_acc1 * alpha +
                   ct::mma(probs_bf16,
                           v_view.load(kt, 1),
                           ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f)) * scale;
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(valid, scores, neg_inf);
    auto tile_m = ct::reduce_max<1>(scores);
    auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
    auto alpha = softmax_exp<true>(row_m - new_m);
    auto probs_f32 = ct::select(valid, softmax_exp<true>(scores - new_m), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto tile_l = ct::sum<1>(probs_f32);
    out_acc0 = out_acc0 * alpha +
               ct::mma(probs_bf16,
                       v_view.load_masked(FullKTiles, 0),
                       ct::full<OutTile>(0.0f));
    out_acc1 = out_acc1 * alpha +
               ct::mma(probs_bf16,
                       v_view.load_masked(FullKTiles, 1),
                       ct::full<OutTile>(0.0f));
    row_l = row_l * alpha + tile_l;

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc0 / row_l), q_block, 0);
    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc1 / row_l), q_block, 1);
}

template <int QRows, int KTile>
__tile_global__ void time_attention1301_main1280_direct_ptr_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int FullKTiles = kNMain / KTile;
    using QIndexTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using KIndexTile = ct::tile<long long, ct::shape<kD, KTile>>;
    using VIndexTile = ct::tile<long long, ct::shape<KTile, kD>>;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const std::size_t batch_offset = static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* q_batch = q + batch_offset;
    const __nv_bfloat16* k_batch = k + batch_offset;
    const __nv_bfloat16* v_batch = v + batch_offset;
    __nv_bfloat16* out_batch = out + batch_offset;

    QIndexTile q_local = ct::iota<QIndexTile>();
    auto q_rows = static_cast<long long>(q_block) * QRows + q_local / kD;
    auto q_cols = q_local % kD;
    auto q_tile = ct::load(q_batch + q_rows * kD + q_cols);

    KIndexTile k_local = ct::iota<KIndexTile>();
    auto k_d = k_local / KTile;
    auto k_cols_local = k_local % KTile;

    VIndexTile v_local = ct::iota<VIndexTile>();
    auto v_rows_local = v_local / kD;
    auto v_cols = v_local % kD;

    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto k_cols = static_cast<long long>(kt) * KTile + k_cols_local;
        auto k_tile = ct::load(k_batch + k_cols * kD + k_d);

        auto scores = ct::mma(q_tile,
                              k_tile,
                              ct::full<ScoreTile>(0.0f)) * scale;
        auto tile_m = ct::reduce_max<1>(scores);
        auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
        auto alpha = ct::exp(row_m - new_m);
        auto probs_f32 = ct::exp(scores - new_m);
        auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
        auto tile_l = ct::sum<1>(probs_f32);

        auto v_rows = static_cast<long long>(kt) * KTile + v_rows_local;
        auto v_tile = ct::load(v_batch + v_rows * kD + v_cols);
        out_acc = out_acc * alpha +
                  ct::mma(probs_bf16,
                          v_tile,
                          ct::full<OutTile>(0.0f));
        row_l = row_l * alpha + tile_l;
        row_m = new_m;
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto score_valid = score_key_cols < kN;
    auto tail_k_cols = static_cast<long long>(FullKTiles) * KTile + k_cols_local;
    auto tail_k_valid = tail_k_cols < kN;
    auto tail_k_safe_cols = ct::select(tail_k_valid, tail_k_cols, tail_k_cols * 0LL);
    auto k_tile = ct::load_masked(k_batch + tail_k_safe_cols * kD + k_d, tail_k_valid);
    auto scores = ct::mma(q_tile,
                          k_tile,
                          ct::full<ScoreTile>(0.0f)) * scale;
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(score_valid, scores, neg_inf);
    auto tile_m = ct::reduce_max<1>(scores);
    auto new_m = ct::select(row_m > tile_m, row_m, tile_m);
    auto alpha = ct::exp(row_m - new_m);
    auto probs_f32 = ct::select(score_valid, ct::exp(scores - new_m), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto tile_l = ct::sum<1>(probs_f32);

    auto tail_v_rows = static_cast<long long>(FullKTiles) * KTile + v_rows_local;
    auto tail_v_valid = tail_v_rows < kN;
    auto tail_v_safe_rows = ct::select(tail_v_valid, tail_v_rows, tail_v_rows * 0LL);
    auto v_tile = ct::load_masked(v_batch + tail_v_safe_rows * kD + v_cols, tail_v_valid);
    out_acc = out_acc * alpha +
              ct::mma(probs_bf16,
                      v_tile,
                      ct::full<OutTile>(0.0f));
    row_l = row_l * alpha + tile_l;

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    ct::store_masked(out_batch + out_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc / row_l),
                     out_rows < kN);
}

template <int QRows, int KTile, int QBlockOffset, bool UseExp2 = false>
__tile_global__ void time_attention1301_tail_offset_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int KTiles = (kN + KTile - 1) / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using RowTile = ct::tile<float, ct::shape<QRows, 1>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block_local, bh, tile_z] = ct::bid();
    (void)tile_z;
    auto q_block = q_block_local + QBlockOffset;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };

    auto q_tile = q_view.load_masked(q_block, 0);
    RowTile row_m = ct::full<RowTile>(-3.402823466e38f);
    RowTile row_l = ct::full<RowTile>(0.0f);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / KTile;
    auto score_cols_local = score_local % KTile;

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{KTiles})) {
        auto key_cols = static_cast<long long>(kt) * KTile + score_cols_local;
        auto valid = (score_rows < kN) && (key_cols < kN);
        auto scores = ct::mma(q_tile,
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
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_valid = out_rows < kN;
    auto safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + safe_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows>
__tile_global__ void time_attention1301_main1280_score_av_lb_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    float scale) {
    constexpr int KTile = 64;
    constexpr int FullKTiles = kNMain / KTile;
    using ScoreTile = ct::tile<float, ct::shape<QRows, KTile>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, KTile>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch =
        q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch =
        k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch =
        v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch =
        out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kN>{}, ct::layout_left{}},
        ct::shape<kD, KTile>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kN, kD>{}},
        ct::shape<KTile, kD>{}
    };
    auto out_view = ct::partition_view{
        ct::tensor_span{out_batch, ct::shape<kN, kD>{}},
        ct::shape<QRows, kD>{}
    };

    auto q_tile = q_view.load(q_block, 0);
    OutTile out_acc = ct::full<OutTile>(0.0f);

    for (auto kt : ct::irange(std::size_t{0}, std::size_t{FullKTiles})) {
        auto scores = ct::mma(q_tile,
                              k_t_view.load(0, kt),
                              ct::full<ScoreTile>(0.0f)) * scale;
        out_acc = out_acc +
                  ct::mma(ct::element_cast<__nv_bfloat16>(scores),
                          v_view.load(kt, 0),
                          ct::full<OutTile>(0.0f));
    }

    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto key_cols = static_cast<long long>(FullKTiles) * KTile + score_local % KTile;
    auto valid = key_cols < kN;
    auto scores = ct::mma(q_tile,
                          k_t_view.load_masked(0, FullKTiles),
                          ct::full<ScoreTile>(0.0f));
    auto score_values = ct::select(valid, scores * scale, scores * 0.0f);
    out_acc = out_acc +
              ct::mma(ct::element_cast<__nv_bfloat16>(score_values),
                      v_view.load_masked(FullKTiles, 0),
                      ct::full<OutTile>(0.0f));

    out_view.store(ct::element_cast<__nv_bfloat16>(out_acc), q_block, 0);
}

float deterministic_bf16_value(size_t idx) {
    float v = 0.125f + static_cast<float>((idx * 17ULL) & 1023ULL) * 0.000244140625f;
    return __bfloat162float(__float2bfloat16(v));
}

template <int KTile>
void validate_rows(const std::vector<__nv_bfloat16>& out, float scale) {
    int rows[] = {0, 37, kN - 1};
    double max_abs = 0.0;
    double sum_sq = 0.0;
    long long count = 0;
    for (int row : rows) {
        std::vector<float> acc(kD, 0.0f);
        float row_m = -3.402823466e38f;
        float row_l = 0.0f;
        for (int kt = 0; kt < ceildiv(kN, KTile) * KTile; kt += KTile) {
            float tile_m = -3.402823466e38f;
            float scores[KTile];
            for (int j = 0; j < KTile; ++j) {
                int col = kt + j;
                float score = -3.402823466e38f;
                if (col < kN) {
                    float dot = 0.0f;
                    for (int d = 0; d < kD; ++d) {
                        float qv = deterministic_bf16_value(static_cast<size_t>(row) * kD + d);
                        float kv = deterministic_bf16_value(static_cast<size_t>(col) * kD + d);
                        dot += qv * kv;
                    }
                    score = dot * scale;
                    tile_m = std::max(tile_m, score);
                }
                scores[j] = score;
            }
            float new_m = std::max(row_m, tile_m);
            float alpha = std::exp(row_m - new_m);
            for (int d = 0; d < kD; ++d) {
                acc[d] *= alpha;
            }
            float tile_l = 0.0f;
            for (int j = 0; j < KTile; ++j) {
                int col = kt + j;
                if (col >= kN) continue;
                float p = std::exp(scores[j] - new_m);
                p = __bfloat162float(__float2bfloat16(p));
                tile_l += p;
                for (int d = 0; d < kD; ++d) {
                    float vv = deterministic_bf16_value(static_cast<size_t>(col) * kD + d);
                    acc[d] += p * vv;
                }
            }
            row_l = row_l * alpha + tile_l;
            row_m = new_m;
        }
        for (int d = 0; d < kD; ++d) {
            float ref = __bfloat162float(__float2bfloat16(acc[d] / row_l));
            float got = __bfloat162float(out[static_cast<size_t>(row) * kD + d]);
            double diff = static_cast<double>(got) - static_cast<double>(ref);
            max_abs = std::max(max_abs, std::abs(diff));
            sum_sq += diff * diff;
            ++count;
        }
    }
    std::printf("validate BH0 rows=3 max_abs=%.9g rms=%.9g\n",
                max_abs, std::sqrt(sum_sq / static_cast<double>(count)));
}

template <int QRows>
void run_padded_variant(const Options& opts,
                        const __nv_bfloat16* d_q,
                        const __nv_bfloat16* d_k,
                        const __nv_bfloat16* d_v,
                        __nv_bfloat16* d_out,
                        float scale,
                        bool run_validation) {
    dim3 grid(ceildiv(kN, QRows), kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_cutile_kernel<QRows><<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
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
        time_attention1301_cutile_kernel<QRows><<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    if (run_validation) {
        std::vector<__nv_bfloat16> out_bh0(static_cast<size_t>(kN) * kD);
        CUDA_CHECK(cudaMemcpy(out_bh0.data(), d_out, out_bh0.size() * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        validate_rows<kKTile>(out_bh0, scale);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kN * kN * kD;
    double padded_flops = 4.0 * static_cast<double>(kBH) * kN * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double padded_tflops = padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("padded qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s padded_math=%.2f TF/s checksum=%.6f\n",
                QRows, kKTile, best_ms, median_ms, real_tflops, padded_tflops, checksum);
}

template <int QRows, int KTile>
void run_masked_variant(const Options& opts,
                        const __nv_bfloat16* d_q,
                        const __nv_bfloat16* d_k,
                        const __nv_bfloat16* d_v,
                        __nv_bfloat16* d_out,
                        float scale,
                        bool run_validation) {
    dim3 grid(ceildiv(kN, QRows), kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_cutile_masked_kernel<QRows, KTile>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
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
        time_attention1301_cutile_masked_kernel<QRows, KTile>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    if (run_validation) {
        std::vector<__nv_bfloat16> out_bh0(static_cast<size_t>(kN) * kD);
        CUDA_CHECK(cudaMemcpy(out_bh0.data(), d_out, out_bh0.size() * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        validate_rows<KTile>(out_bh0, scale);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kN * kN * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kN * (ceildiv(kN, KTile) * KTile) * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("masked qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s checksum=%.6f\n",
                QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops, checksum);
}

template <int QRows, int KTile>
void run_score_av_lower_bound_variant(const Options& opts,
                                      const __nv_bfloat16* d_q,
                                      const __nv_bfloat16* d_k,
                                      const __nv_bfloat16* d_v,
                                      __nv_bfloat16* d_out,
                                      float scale) {
    dim3 grid(ceildiv(kN, QRows), kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_cutile_masked_score_av_lb_kernel<QRows, KTile>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
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
        time_attention1301_cutile_masked_score_av_lb_kernel<QRows, KTile>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kN * kN * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kN * (ceildiv(kN, KTile) * KTile) * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("score_av_lb qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s checksum=%.6f\n",
                QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops, checksum);
}

template <int QRows, bool ScoreAvLowerBound, bool SumF32 = false, bool UseExp2 = false,
          int KTile = 64, bool ScoreBf16 = false, bool PrescaleQ = false>
void run_main1280_variant(const Options& opts,
                          const __nv_bfloat16* d_q,
                          const __nv_bfloat16* d_k,
                          const __nv_bfloat16* d_v,
                          __nv_bfloat16* d_out,
                          float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        if constexpr (ScoreAvLowerBound) {
            time_attention1301_main1280_score_av_lb_kernel<QRows>
                <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        } else {
            time_attention1301_main1280_kernel<QRows, KTile, SumF32, UseExp2,
                                               ScoreBf16, PrescaleQ>
                <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        }
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
        if constexpr (ScoreAvLowerBound) {
            time_attention1301_main1280_score_av_lb_kernel<QRows>
                <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        } else {
            time_attention1301_main1280_kernel<QRows, KTile, SumF32, UseExp2,
                                               ScoreBf16, PrescaleQ>
                <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops = 4.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    const char* name = "main1280";
    if constexpr (ScoreAvLowerBound) {
        name = "main1280_score_av_lb";
    } else if constexpr (PrescaleQ && UseExp2 && SumF32) {
        name = "main1280_sum_f32_exp2_prescale_q";
    } else if constexpr (PrescaleQ && SumF32) {
        name = "main1280_sum_f32_prescale_q";
    } else if constexpr (PrescaleQ) {
        name = "main1280_prescale_q";
    } else if constexpr (ScoreBf16 && SumF32) {
        name = "main1280_sum_f32_score_bf16";
    } else if constexpr (ScoreBf16) {
        name = "main1280_score_bf16";
    } else if constexpr (UseExp2 && SumF32) {
        name = "main1280_sum_f32_exp2";
    } else if constexpr (UseExp2) {
        name = "main1280_exp2";
    } else if constexpr (SumF32) {
        name = "main1280_sum_f32";
    }
    std::printf("%s qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                name, QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

void run_main1280_q64k32_exp2_split_d32_variant(const Options& opts,
                                                const __nv_bfloat16* d_q,
                                                const __nv_bfloat16* d_k,
                                                const __nv_bfloat16* d_v,
                                                __nv_bfloat16* d_out,
                                                float scale) {
    dim3 grid(kNMain / 64, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_q64k32_exp2_split_d32_kernel
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
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
        time_attention1301_main1280_q64k32_exp2_split_d32_kernel
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops = 4.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_sum_f32_exp2_split_d32 qrows=64 ktile=32 best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

template <int MainKTile, int TailQRows = 64, int TailKTile = 64, bool UseExp2 = false>
void run_split_tail_pair_variant(const Options& opts,
                                 const __nv_bfloat16* d_q,
                                 const __nv_bfloat16* d_k,
                                 const __nv_bfloat16* d_v,
                                 __nv_bfloat16* d_out,
                                 float scale) {
    constexpr int QRows = 64;
    static_assert(kNMain % TailQRows == 0);
    dim3 grid_main(kNMain / QRows, kBH);
    dim3 grid_tail(ceildiv(kN - kNMain, TailQRows), kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_kernel<QRows, MainKTile, true, UseExp2>
            <<<grid_main, 1>>>(d_q, d_k, d_v, d_out, scale);
        time_attention1301_tail_offset_kernel<TailQRows, TailKTile, kNMain / TailQRows, UseExp2>
            <<<grid_tail, 1>>>(d_q, d_k, d_v, d_out, scale);
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
        time_attention1301_main1280_kernel<QRows, MainKTile, true, UseExp2>
            <<<grid_main, 1>>>(d_q, d_k, d_v, d_out, scale);
        time_attention1301_tail_offset_kernel<TailQRows, TailKTile, kNMain / TailQRows, UseExp2>
            <<<grid_tail, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum0_bf16{};
    __nv_bfloat16 checksum_tail_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum0_bf16, d_out, sizeof(checksum0_bf16),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&checksum_tail_bf16, d_out + static_cast<std::size_t>(kNMain) * kD,
                          sizeof(checksum_tail_bf16), cudaMemcpyDeviceToHost));
    float checksum0 = __bfloat162float(checksum0_bf16);
    float checksum_tail = __bfloat162float(checksum_tail_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kN * kN * kD;
    double logical_main_keys = static_cast<double>(kNMain + MainKTile);
    double logical_tail_qrows =
        static_cast<double>(ceildiv(kN - kNMain, TailQRows) * TailQRows);
    double logical_tail_keys = static_cast<double>(ceildiv(kN, TailKTile) * TailKTile);
    double logical_flops =
        4.0 * static_cast<double>(kBH) * kNMain * logical_main_keys * kD +
        4.0 * static_cast<double>(kBH) * logical_tail_qrows * logical_tail_keys * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops = logical_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("split_tail_pair_sum_f32%s main_ktile=%d tail_qrows=%d tail_ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum0=%.6f checksum_tail=%.6f\n",
                UseExp2 ? "_exp2" : "", MainKTile, TailQRows, TailKTile,
                best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum0, checksum_tail);
}

template <int QRows, int KTile, bool UseExp2 = false>
void run_tail_only_variant(const Options& opts,
                           const __nv_bfloat16* d_q,
                           const __nv_bfloat16* d_k,
                           const __nv_bfloat16* d_v,
                           __nv_bfloat16* d_out,
                           float scale) {
    static_assert(kNMain % QRows == 0);
    constexpr int QBlockOffset = kNMain / QRows;
    constexpr int TailRows = kN - kNMain;
    dim3 grid(ceildiv(TailRows, QRows), kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_tail_offset_kernel<QRows, KTile, QBlockOffset, UseExp2>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
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
        time_attention1301_tail_offset_kernel<QRows, KTile, QBlockOffset, UseExp2>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out + static_cast<std::size_t>(kNMain) * kD,
                          sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * TailRows * kN * kD;
    double logical_qrows = static_cast<double>(ceildiv(TailRows, QRows) * QRows);
    double logical_keys = static_cast<double>(ceildiv(kN, KTile) * KTile);
    double logical_flops = 4.0 * static_cast<double>(kBH) * logical_qrows * logical_keys * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops = logical_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("tail_only%s qrows=%d ktile=%d launches=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                UseExp2 ? "_exp2" : "", QRows, KTile, ceildiv(TailRows, QRows),
                best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

template <int QRows, int KTile>
void run_main1280_direct_ptr_variant(const Options& opts,
                                     const __nv_bfloat16* d_q,
                                     const __nv_bfloat16* d_k,
                                     const __nv_bfloat16* d_v,
                                     __nv_bfloat16* d_out,
                                     float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_direct_ptr_kernel<QRows, KTile>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
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
        time_attention1301_main1280_direct_ptr_kernel<QRows, KTile>
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16),
                          cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops =
        4.0 * static_cast<double>(kBH) * kNMain * (kNMain + KTile) * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_direct_ptr_sum_f32 qrows=%d ktile=%d best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                QRows, KTile, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

template <int QRows>
void run_main1280_qk_only_variant(const Options& opts,
                                  const __nv_bfloat16* d_q,
                                  const __nv_bfloat16* d_k,
                                  __nv_bfloat16* d_out,
                                  float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_qk_only_kernel<QRows>
            <<<grid, 1>>>(d_q, d_k, d_out, scale);
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
        time_attention1301_main1280_qk_only_kernel<QRows>
            <<<grid, 1>>>(d_q, d_k, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 2.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops = 2.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_qk_only qrows=%d ktile=64 best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                QRows, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

template <int QRows>
void run_main1280_qk_only_kt_variant(const Options& opts,
                                     const __nv_bfloat16* d_q,
                                     const __nv_bfloat16* d_k_t,
                                     __nv_bfloat16* d_out,
                                     float scale) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_qk_only_kt_kernel<QRows>
            <<<grid, 1>>>(d_q, d_k_t, d_out, scale);
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
        time_attention1301_main1280_qk_only_kt_kernel<QRows>
            <<<grid, 1>>>(d_q, d_k_t, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 2.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops = 2.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_qk_only_kt qrows=%d ktile=64 best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                QRows, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

template <int QRows>
void run_main1280_av_const_variant(const Options& opts,
                                   const __nv_bfloat16* d_v,
                                   __nv_bfloat16* d_out) {
    dim3 grid(kNMain / QRows, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_av_const_kernel<QRows>
            <<<grid, 1>>>(d_v, d_out);
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
        time_attention1301_main1280_av_const_kernel<QRows>
            <<<grid, 1>>>(d_v, d_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 2.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops = 2.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_av_const qrows=%d ktile=64 best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                QRows, best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

template <int QRows>
void run_main1280_split_global_variant(const Options& opts,
                                       const __nv_bfloat16* d_q,
                                       const __nv_bfloat16* d_k,
                                       const __nv_bfloat16* d_v,
                                       __nv_bfloat16* d_p,
                                       __nv_bfloat16* d_out,
                                       float scale) {
    constexpr int QBlocks = kNMain / QRows;
    constexpr int KTiles = (kN + kKTile - 1) / kKTile;
    dim3 grid(QBlocks, kBH);

    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_main1280_qk_store_p_kernel<QRows>
            <<<grid, 1>>>(d_q, d_k, d_p, scale);
        time_attention1301_main1280_av_load_p_kernel<QRows>
            <<<grid, 1>>>(d_p, d_v, d_out);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> qk_times_ms;
    std::vector<float> av_times_ms;
    std::vector<float> pair_times_ms;
    qk_times_ms.reserve(opts.iters);
    av_times_ms.reserve(opts.iters);
    pair_times_ms.reserve(opts.iters);

    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_qk_store_p_kernel<QRows>
            <<<grid, 1>>>(d_q, d_k, d_p, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        qk_times_ms.push_back(ms);
    }

    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_av_load_p_kernel<QRows>
            <<<grid, 1>>>(d_p, d_v, d_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        av_times_ms.push_back(ms);
    }

    for (int i = 0; i < opts.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        time_attention1301_main1280_qk_store_p_kernel<QRows>
            <<<grid, 1>>>(d_q, d_k, d_p, scale);
        time_attention1301_main1280_av_load_p_kernel<QRows>
            <<<grid, 1>>>(d_p, d_v, d_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        pair_times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float qk_best_ms = *std::min_element(qk_times_ms.begin(), qk_times_ms.end());
    float av_best_ms = *std::min_element(av_times_ms.begin(), av_times_ms.end());
    float pair_best_ms = *std::min_element(pair_times_ms.begin(), pair_times_ms.end());
    float pair_median_ms = percentile(pair_times_ms, 0.5f);
    double half_flops = 2.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double pair_flops = 2.0 * half_flops;
    double qk_tflops = half_flops / (static_cast<double>(qk_best_ms) * 1.0e-3) / 1.0e12;
    double av_tflops = half_flops / (static_cast<double>(av_best_ms) * 1.0e-3) / 1.0e12;
    double pair_tflops = pair_flops / (static_cast<double>(pair_best_ms) * 1.0e-3) / 1.0e12;
    double p_elems = static_cast<double>(kBH) * QBlocks * KTiles * QRows * kKTile;
    double p_roundtrip_gb = p_elems * sizeof(__nv_bfloat16) * 2.0 / 1.0e9;
    double p_roundtrip_gbs =
        p_roundtrip_gb / (static_cast<double>(pair_best_ms) * 1.0e-3);
    std::printf("main1280_split_global qrows=%d ktile=64 qk=%.3f ms %.2f TF/s av=%.3f ms %.2f TF/s pair=%.3f ms median=%.3f ms pair_math=%.2f TF/s roof70=%.1f%% p_roundtrip=%.2f GB %.1f GB/s checksum=%.6f\n",
                QRows, qk_best_ms, qk_tflops, av_best_ms, av_tflops,
                pair_best_ms, pair_median_ms, pair_tflops,
                pair_tflops / 70.0 * 100.0, p_roundtrip_gb,
                p_roundtrip_gbs, checksum);
}

void run_main1280_split_d32_lower_bound_variant(const Options& opts,
                                                const __nv_bfloat16* d_q,
                                                const __nv_bfloat16* d_k,
                                                const __nv_bfloat16* d_v,
                                                __nv_bfloat16* d_out,
                                                float scale) {
    dim3 grid(kNMain / 64, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_q64k64_main1280_score_av_lb_split_d32_kernel
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
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
        time_attention1301_q64k64_main1280_score_av_lb_split_d32_kernel
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops = 4.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_score_av_lb_split_d32 qrows=64 ktile=64 best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

void run_main1280_prescale_q_lower_bound_variant(const Options& opts,
                                                 const __nv_bfloat16* d_q,
                                                 const __nv_bfloat16* d_k,
                                                 const __nv_bfloat16* d_v,
                                                 __nv_bfloat16* d_out,
                                                 float scale) {
    dim3 grid(kNMain / 64, kBH);
    for (int i = 0; i < opts.warmup; ++i) {
        time_attention1301_q64k64_main1280_score_av_lb_prescale_q_kernel
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
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
        time_attention1301_q64k64_main1280_score_av_lb_prescale_q_kernel
            <<<grid, 1>>>(d_q, d_k, d_v, d_out, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    __nv_bfloat16 checksum_bf16{};
    CUDA_CHECK(cudaMemcpy(&checksum_bf16, d_out, sizeof(checksum_bf16), cudaMemcpyDeviceToHost));
    float checksum = __bfloat162float(checksum_bf16);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double real_flops = 4.0 * static_cast<double>(kBH) * kNMain * kN * kD;
    double logical_padded_flops = 4.0 * static_cast<double>(kBH) * kNMain * kNPad * kD;
    double real_tflops = real_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    double logical_tflops =
        logical_padded_flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf("main1280_score_av_lb_prescale_q qrows=64 ktile=64 best=%.3f ms median=%.3f ms real_math=%.2f TF/s logical_tile_math=%.2f TF/s roof70=%.1f%% checksum=%.6f\n",
                best_ms, median_ms, real_tflops, logical_tflops,
                real_tflops / 70.0 * 100.0, checksum);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::printf("GPU: %s, SM %d.%d, SMs %d\n",
                    prop.name, prop.major, prop.minor, prop.multiProcessorCount);
        std::printf("Shape: BH=%d N=%d NPad=%d D=%d, CUDA Tile streaming QK-softmax-AV\n",
                    kBH, kN, kNPad, kD);

        size_t padded_elems = static_cast<size_t>(kBH) * kNPad * kD;
        size_t unpadded_elems = static_cast<size_t>(kBH) * kN * kD;
        size_t out_elems = static_cast<size_t>(kBH) * kN * kD;
        constexpr int split_qrows = 64;
        constexpr int split_qblocks = kNMain / split_qrows;
        constexpr int split_ktiles = (kN + kKTile - 1) / kKTile;
        size_t split_p_elems =
            static_cast<size_t>(kBH) * split_qblocks * split_ktiles * split_qrows * kKTile;
        __nv_bfloat16* d_q = nullptr;
        __nv_bfloat16* d_k = nullptr;
        __nv_bfloat16* d_v = nullptr;
        __nv_bfloat16* d_q_masked = nullptr;
        __nv_bfloat16* d_k_masked = nullptr;
        __nv_bfloat16* d_k_t_masked = nullptr;
        __nv_bfloat16* d_v_masked = nullptr;
        __nv_bfloat16* d_out = nullptr;
        __nv_bfloat16* d_p_split_q64 = nullptr;
        CUDA_CHECK(cudaMalloc(&d_q, padded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_k, padded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_v, padded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_q_masked, unpadded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_k_masked, unpadded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_k_t_masked, unpadded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_v_masked, unpadded_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_p_split_q64, split_p_elems * sizeof(__nv_bfloat16)));

        int fill_blocks = static_cast<int>((padded_elems + kInitTile - 1) / kInitTile);
        fill_bf16_kernel<<<fill_blocks, 1>>>(d_q, static_cast<long long>(padded_elems));
        fill_bf16_kernel<<<fill_blocks, 1>>>(d_k, static_cast<long long>(padded_elems));
        fill_bf16_kernel<<<fill_blocks, 1>>>(d_v, static_cast<long long>(padded_elems));
        int fill_masked_blocks =
            static_cast<int>((unpadded_elems + kInitTile - 1) / kInitTile);
        fill_bf16_kernel<<<fill_masked_blocks, 1>>>(
            d_q_masked, static_cast<long long>(unpadded_elems));
        fill_bf16_kernel<<<fill_masked_blocks, 1>>>(
            d_k_masked, static_cast<long long>(unpadded_elems));
        fill_bf16_kernel<<<fill_masked_blocks, 1>>>(
            d_v_masked, static_cast<long long>(unpadded_elems));
        transpose_k_nd_to_dn_kernel<<<fill_masked_blocks, 1>>>(
            d_k_masked, d_k_t_masked, static_cast<long long>(unpadded_elems));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float scale = 1.0f / std::sqrt(static_cast<float>(kD));
        run_padded_variant<8>(opts, d_q, d_k, d_v, d_out, scale, false);
        run_padded_variant<16>(opts, d_q, d_k, d_v, d_out, scale, opts.validate);
        run_padded_variant<32>(opts, d_q, d_k, d_v, d_out, scale, false);
        run_padded_variant<64>(opts, d_q, d_k, d_v, d_out, scale, false);
        run_padded_variant<128>(opts, d_q, d_k, d_v, d_out, scale, false);
        run_masked_variant<16, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale, false);
        run_masked_variant<16, 64>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale, false);
        run_masked_variant<32, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale, false);
        run_masked_variant<32, 64>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale, opts.validate);
        run_masked_variant<64, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale, opts.validate);
        run_masked_variant<64, 64>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale, false);
        run_masked_variant<128, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale, false);
        run_score_av_lower_bound_variant<64, 32>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_score_av_lower_bound_variant<64, 64>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_qk_only_variant<32>(opts, d_q_masked, d_k_masked, d_out, scale);
        run_main1280_qk_only_variant<64>(opts, d_q_masked, d_k_masked, d_out, scale);
        run_main1280_qk_only_kt_variant<64>(opts, d_q_masked, d_k_t_masked, d_out, scale);
        run_main1280_qk_only_variant<128>(opts, d_q_masked, d_k_masked, d_out, scale);
        run_main1280_av_const_variant<32>(opts, d_v_masked, d_out);
        run_main1280_av_const_variant<64>(opts, d_v_masked, d_out);
        run_main1280_av_const_variant<128>(opts, d_v_masked, d_out);
        run_main1280_variant<32, false>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<32, false, true>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<32, false, true, false, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<32, false, true, true, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<32, true>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, false, 16>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, true, 16>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, false, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, true, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, true, 32, false, true>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_direct_ptr_variant<64, 32>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, false, 128>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_tail_only_variant<64, 64>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_tail_only_variant<32, 64>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_tail_only_variant<16, 64>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_tail_only_variant<32, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_tail_only_variant<64, 64, true>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_tail_only_variant<32, 32, true>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_split_tail_pair_variant<64>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_split_tail_pair_variant<32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_split_tail_pair_variant<32, 32, 64>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_split_tail_pair_variant<32, 32, 32>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_split_tail_pair_variant<32, 64, 64, true>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_split_tail_pair_variant<32, 32, 32, true>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, false, 64, true>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, false, 32, true>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, false, true, true>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_q64k32_exp2_split_d32_variant(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<64, true>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_prescale_q_lower_bound_variant(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_split_d32_lower_bound_variant(
            opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_split_global_variant<64>(
            opts, d_q_masked, d_k_masked, d_v_masked, d_p_split_q64, d_out, scale);
        run_main1280_variant<128, false>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<128, false, true>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<128, false, true, false, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<128, false, true, true, 32>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        run_main1280_variant<128, true>(opts, d_q_masked, d_k_masked, d_v_masked, d_out, scale);
        CUDA_CHECK(cudaFree(d_q));
        CUDA_CHECK(cudaFree(d_k));
        CUDA_CHECK(cudaFree(d_v));
        CUDA_CHECK(cudaFree(d_q_masked));
        CUDA_CHECK(cudaFree(d_k_masked));
        CUDA_CHECK(cudaFree(d_k_t_masked));
        CUDA_CHECK(cudaFree(d_v_masked));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_p_split_q64));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
