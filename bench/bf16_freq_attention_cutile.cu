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

constexpr int kN = 60;
constexpr int kNPad = 64;
constexpr int kD = 64;
constexpr int kBH = 10408;
constexpr int kInitTile = 256;
constexpr float kScale = 0.125f;

using I64InitTile = ct::tile<long long, ct::shape<kInitTile>>;
using F32InitTile = ct::tile<float, ct::shape<kInitTile>>;

struct Options {
    int warmup = 1;
    int iters = 5;
    std::string variant = "all";
    bool validate = false;
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
        if (std::strcmp(argv[i], "--warmup") == 0) {
            opts.warmup = parse_int_arg(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            opts.iters = parse_int_arg(argv[i], need_value(argv[i]));
        } else if (std::strcmp(argv[i], "--variant") == 0) {
            opts.variant = need_value(argv[i]);
        } else if (std::strcmp(argv[i], "--validate") == 0) {
            opts.validate = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: bench_bf16_freq_attention_cutile [options]\n"
                "  --variant NAME  all, q8, q16, q32, q64, q128, q8m, q16m, q32m,\n"
                "                  q64m, q128m, q8p, q16p, q32p, q64p, q128p,\n"
                "                  q8s, q16s, q32s, q64s, q128s, q16sc,\n"
                "                  default all\n"
                "  --warmup N      warmup launches, default 1\n"
                "  --iters N       measured launches, default 5\n"
                "  --validate      compare BH=0 output against CPU reference\n");
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
    I64InitTile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64InitTile>();
    auto in_bounds = idx < total;
    F32InitTile values = 0.125f +
        ct::element_cast<float>((idx * 17LL) & 1023LL) * 0.000244140625f;
    ct::store_masked(dst + idx, ct::element_cast<__nv_bfloat16>(values), in_bounds);
}

__tile_global__ void pad3_freq60_to64_kernel(const __nv_bfloat16* __restrict__ q,
                                             const __nv_bfloat16* __restrict__ k,
                                             const __nv_bfloat16* __restrict__ v,
                                             __nv_bfloat16* __restrict__ q_pad,
                                             __nv_bfloat16* __restrict__ k_pad,
                                             __nv_bfloat16* __restrict__ v_pad,
                                             long long total) {
    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    q_pad = ct::assume_aligned(q_pad, 16_ic);
    k_pad = ct::assume_aligned(k_pad, 16_ic);
    v_pad = ct::assume_aligned(v_pad, 16_ic);

    I64InitTile idx = (long long)ct::bid().x * kInitTile + ct::iota<I64InitTile>();
    auto in_bounds = idx < total;
    auto d = idx % kD;
    auto row = (idx / kD) % kNPad;
    auto bh = idx / (kNPad * kD);
    auto valid = in_bounds && (row < kN);
    auto safe_row = ct::select(valid, row, row * 0LL);
    auto src = bh * (long long)kN * kD + safe_row * kD + d;
    auto zero = ct::element_cast<__nv_bfloat16>(ct::element_cast<float>(idx * 0LL));
    auto qv = ct::load_masked(q + src, valid, zero);
    auto kv = ct::load_masked(k + src, valid, zero);
    auto vv = ct::load_masked(v + src, valid, zero);
    ct::store_masked(q_pad + idx, qv, in_bounds);
    ct::store_masked(k_pad + idx, kv, in_bounds);
    ct::store_masked(v_pad + idx, vv, in_bounds);
}

template <int QRows>
__tile_global__ void freq_attention60_cutile_kernel(const __nv_bfloat16* __restrict__ q,
                                                    const __nv_bfloat16* __restrict__ k,
                                                    const __nv_bfloat16* __restrict__ v,
                                                    __nv_bfloat16* __restrict__ out) {
    using ScoreTile = ct::tile<float, ct::shape<QRows, kNPad>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, kNPad>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch = q + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* k_batch = k + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* v_batch = v + static_cast<std::size_t>(bh) * kNPad * kD;
    __nv_bfloat16* out_batch = out + static_cast<std::size_t>(bh) * kNPad * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kNPad, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kNPad>{}, ct::layout_left{}},
        ct::shape<kD, kNPad>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kNPad, kD>{}},
        ct::shape<kNPad, kD>{}
    };

    auto scores = ct::mma(q_view.load(q_block, 0), k_t_view.load(0, 0),
                          ct::full<ScoreTile>(0.0f));
    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / kNPad;
    auto score_cols = score_local % kNPad;
    auto score_valid = (score_rows < kN) && (score_cols < kN);
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(score_valid, scores * kScale, neg_inf);

    auto row_max = ct::reduce_max<1>(scores);
    auto probs_f32 = ct::select(score_valid, ct::exp(scores - row_max), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto denom = ct::sum<1>(ct::element_cast<float>(probs_bf16));

    auto out_acc = ct::mma(probs_bf16, v_view.load(0, 0), ct::full<OutTile>(0.0f));
    out_acc = out_acc / denom;

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_valid = out_rows < kN;
    ct::store_masked(out_batch + out_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

template <int QRows, bool SumBF16Denom = true, bool ConstNegInf = false>
__tile_global__ void freq_attention60_cutile_padded_out60_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out) {
    using ScoreTile = ct::tile<float, ct::shape<QRows, kNPad>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, kNPad>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch = q + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* k_batch = k + static_cast<std::size_t>(bh) * kNPad * kD;
    const __nv_bfloat16* v_batch = v + static_cast<std::size_t>(bh) * kNPad * kD;
    __nv_bfloat16* out_batch = out + static_cast<std::size_t>(bh) * kN * kD;

    auto q_view = ct::partition_view{
        ct::tensor_span{q_batch, ct::shape<kNPad, kD>{}},
        ct::shape<QRows, kD>{}
    };
    auto k_t_view = ct::partition_view{
        ct::tensor_span{k_batch, ct::shape<kD, kNPad>{}, ct::layout_left{}},
        ct::shape<kD, kNPad>{}
    };
    auto v_view = ct::partition_view{
        ct::tensor_span{v_batch, ct::shape<kNPad, kD>{}},
        ct::shape<kNPad, kD>{}
    };

    auto scores = ct::mma(q_view.load(q_block, 0), k_t_view.load(0, 0),
                          ct::full<ScoreTile>(0.0f));
    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / kNPad;
    auto score_cols = score_local % kNPad;
    auto score_valid = (score_rows < kN) && (score_cols < kN);
    auto neg_inf = [&]() {
        if constexpr (ConstNegInf) {
            return ct::full<ScoreTile>(-3.402823466e38f);
        } else {
            return scores * 0.0f - 3.402823466e38f;
        }
    }();
    scores = ct::select(score_valid, scores * kScale, neg_inf);

    auto row_max = ct::reduce_max<1>(scores);
    auto probs_f32 = ct::select(score_valid, ct::exp(scores - row_max), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto denom = [&]() {
        if constexpr (SumBF16Denom) {
            return ct::sum<1>(ct::element_cast<float>(probs_bf16));
        } else {
            return ct::sum<1>(probs_f32);
        }
    }();

    auto out_acc = ct::mma(probs_bf16, v_view.load(0, 0), ct::full<OutTile>(0.0f));
    out_acc = out_acc / denom;

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
__tile_global__ void freq_attention60_cutile_masked_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out) {
    using QTile = ct::tile<__nv_bfloat16, ct::shape<QRows, kD>>;
    using KTile = ct::tile<__nv_bfloat16, ct::shape<kD, kNPad>>;
    using VTile = ct::tile<__nv_bfloat16, ct::shape<kNPad, kD>>;
    using ScoreTile = ct::tile<float, ct::shape<QRows, kNPad>>;
    using OutTile = ct::tile<float, ct::shape<QRows, kD>>;
    using I64QTile = ct::tile<long long, ct::shape<QRows, kD>>;
    using I64KTile = ct::tile<long long, ct::shape<kD, kNPad>>;
    using I64VTile = ct::tile<long long, ct::shape<kNPad, kD>>;
    using I64ScoreTile = ct::tile<long long, ct::shape<QRows, kNPad>>;
    using I64OutTile = ct::tile<long long, ct::shape<QRows, kD>>;

    q = ct::assume_aligned(q, 16_ic);
    k = ct::assume_aligned(k, 16_ic);
    v = ct::assume_aligned(v, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto [q_block, bh, tile_z] = ct::bid();
    (void)tile_z;
    const __nv_bfloat16* q_batch = q + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* k_batch = k + static_cast<std::size_t>(bh) * kN * kD;
    const __nv_bfloat16* v_batch = v + static_cast<std::size_t>(bh) * kN * kD;
    __nv_bfloat16* out_batch = out + static_cast<std::size_t>(bh) * kN * kD;

    I64QTile q_local = ct::iota<I64QTile>();
    auto q_rows = static_cast<long long>(q_block) * QRows + q_local / kD;
    auto q_cols = q_local % kD;
    auto q_valid = q_rows < kN;
    auto q_safe_rows = ct::select(q_valid, q_rows, q_rows * 0LL);
    QTile q_tile = ct::load_masked(q_batch + q_safe_rows * kD + q_cols, q_valid);

    I64KTile k_local = ct::iota<I64KTile>();
    auto k_dim = k_local / kNPad;
    auto k_col = k_local % kNPad;
    auto k_valid = k_col < kN;
    auto k_safe_col = ct::select(k_valid, k_col, k_col * 0LL);
    KTile k_tile = ct::load_masked(k_batch + k_safe_col * kD + k_dim, k_valid);

    I64VTile v_local = ct::iota<I64VTile>();
    auto v_row = v_local / kD;
    auto v_col = v_local % kD;
    auto v_valid = v_row < kN;
    auto v_safe_row = ct::select(v_valid, v_row, v_row * 0LL);
    VTile v_tile = ct::load_masked(v_batch + v_safe_row * kD + v_col, v_valid);

    auto scores = ct::mma(q_tile, k_tile, ct::full<ScoreTile>(0.0f));
    I64ScoreTile score_local = ct::iota<I64ScoreTile>();
    auto score_rows = static_cast<long long>(q_block) * QRows + score_local / kNPad;
    auto score_cols = score_local % kNPad;
    auto score_valid = (score_rows < kN) && (score_cols < kN);
    auto neg_inf = scores * 0.0f - 3.402823466e38f;
    scores = ct::select(score_valid, scores * kScale, neg_inf);

    auto row_max = ct::reduce_max<1>(scores);
    auto probs_f32 = ct::select(score_valid, ct::exp(scores - row_max), scores * 0.0f);
    auto probs_bf16 = ct::element_cast<__nv_bfloat16>(probs_f32);
    auto denom = ct::sum<1>(ct::element_cast<float>(probs_bf16));

    auto out_acc = ct::mma(probs_bf16, v_tile, ct::full<OutTile>(0.0f));
    out_acc = out_acc / denom;

    I64OutTile out_local = ct::iota<I64OutTile>();
    auto out_rows = static_cast<long long>(q_block) * QRows + out_local / kD;
    auto out_cols = out_local % kD;
    auto out_valid = out_rows < kN;
    auto out_safe_rows = ct::select(out_valid, out_rows, out_rows * 0LL);
    ct::store_masked(out_batch + out_safe_rows * kD + out_cols,
                     ct::element_cast<__nv_bfloat16>(out_acc),
                     out_valid);
}

void init_bf16(__nv_bfloat16* ptr, size_t elems) {
    fill_bf16_kernel<<<ceildiv(static_cast<int>(elems), kInitTile), 1>>>(
        ptr, static_cast<long long>(elems));
    CUDA_CHECK(cudaGetLastError());
}

template <int QRows>
void launch_cutile(const __nv_bfloat16* q,
                   const __nv_bfloat16* k,
                   const __nv_bfloat16* v,
                   __nv_bfloat16* out) {
    dim3 grid(ceildiv(kN, QRows), kBH, 1);
    freq_attention60_cutile_kernel<QRows><<<grid, 1>>>(q, k, v, out);
}

template <int QRows>
void launch_cutile_masked(const __nv_bfloat16* q,
                          const __nv_bfloat16* k,
                          const __nv_bfloat16* v,
                          __nv_bfloat16* out) {
    dim3 grid(ceildiv(kN, QRows), kBH, 1);
    freq_attention60_cutile_masked_kernel<QRows><<<grid, 1>>>(q, k, v, out);
}

template <int QRows, bool SumBF16Denom = true, bool ConstNegInf = false>
void launch_cutile_padded_out60(const __nv_bfloat16* q,
                                const __nv_bfloat16* k,
                                const __nv_bfloat16* v,
                                __nv_bfloat16* out) {
    dim3 grid(ceildiv(kN, QRows), kBH, 1);
    freq_attention60_cutile_padded_out60_kernel<QRows, SumBF16Denom, ConstNegInf>
        <<<grid, 1>>>(q, k, v, out);
}

void validate_q16(const __nv_bfloat16* d_q,
                  const __nv_bfloat16* d_k,
                  const __nv_bfloat16* d_v,
                  const __nv_bfloat16* d_out) {
    size_t elems = static_cast<size_t>(kNPad) * kD;
    std::vector<__nv_bfloat16> q(elems);
    std::vector<__nv_bfloat16> k(elems);
    std::vector<__nv_bfloat16> v(elems);
    std::vector<__nv_bfloat16> out(elems);
    CUDA_CHECK(cudaMemcpy(q.data(), d_q, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k.data(), d_k, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v.data(), d_v, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    float rms = 0.0f;
    int count = 0;
    for (int r = 0; r < kN; ++r) {
        float scores[kN];
        float row_max = -3.402823466e38f;
        for (int c = 0; c < kN; ++c) {
            float acc = 0.0f;
            for (int d = 0; d < kD; ++d) {
                acc += __bfloat162float(q[r * kD + d]) *
                       __bfloat162float(k[c * kD + d]);
            }
            scores[c] = acc * kScale;
            row_max = std::max(row_max, scores[c]);
        }
        float probs[kN];
        float denom = 0.0f;
        for (int c = 0; c < kN; ++c) {
            __nv_bfloat16 p = __float2bfloat16(std::exp(scores[c] - row_max));
            probs[c] = __bfloat162float(p);
            denom += probs[c];
        }
        for (int d = 0; d < kD; ++d) {
            float acc = 0.0f;
            for (int c = 0; c < kN; ++c) {
                acc += probs[c] * __bfloat162float(v[c * kD + d]);
            }
            float ref = __bfloat162float(__float2bfloat16(acc / denom));
            float got = __bfloat162float(out[r * kD + d]);
            float diff = std::fabs(ref - got);
            max_abs = std::max(max_abs, diff);
            rms += diff * diff;
            ++count;
        }
    }
    rms = std::sqrt(rms / static_cast<float>(count));
    std::printf("  validate BH0 max_abs=%.8g rms=%.8g\n", max_abs, rms);
}

void validate_unpadded(const __nv_bfloat16* d_q,
                       const __nv_bfloat16* d_k,
                       const __nv_bfloat16* d_v,
                       const __nv_bfloat16* d_out,
                       bool sum_bf16_denom = true) {
    size_t elems = static_cast<size_t>(kN) * kD;
    std::vector<__nv_bfloat16> q(elems);
    std::vector<__nv_bfloat16> k(elems);
    std::vector<__nv_bfloat16> v(elems);
    std::vector<__nv_bfloat16> out(elems);
    CUDA_CHECK(cudaMemcpy(q.data(), d_q, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k.data(), d_k, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v.data(), d_v, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    float rms = 0.0f;
    int count = 0;
    for (int r = 0; r < kN; ++r) {
        float scores[kN];
        float row_max = -3.402823466e38f;
        for (int c = 0; c < kN; ++c) {
            float acc = 0.0f;
            for (int d = 0; d < kD; ++d) {
                acc += __bfloat162float(q[r * kD + d]) *
                       __bfloat162float(k[c * kD + d]);
            }
            scores[c] = acc * kScale;
            row_max = std::max(row_max, scores[c]);
        }
        float probs[kN];
        float denom = 0.0f;
        for (int c = 0; c < kN; ++c) {
            float p_f32 = std::exp(scores[c] - row_max);
            __nv_bfloat16 p = __float2bfloat16(p_f32);
            probs[c] = __bfloat162float(p);
            denom += sum_bf16_denom ? probs[c] : p_f32;
        }
        for (int d = 0; d < kD; ++d) {
            float acc = 0.0f;
            for (int c = 0; c < kN; ++c) {
                acc += probs[c] * __bfloat162float(v[c * kD + d]);
            }
            float ref = __bfloat162float(__float2bfloat16(acc / denom));
            float got = __bfloat162float(out[r * kD + d]);
            float diff = std::fabs(ref - got);
            max_abs = std::max(max_abs, diff);
            rms += diff * diff;
            ++count;
        }
    }
    rms = std::sqrt(rms / static_cast<float>(count));
    std::printf("  validate BH0 max_abs=%.8g rms=%.8g\n", max_abs, rms);
}

template <int QRows, bool ConstNegInf = false>
void run_padded_out60_source_variant(const Options& opts, const char* name) {
    size_t in_elems = static_cast<size_t>(kBH) * kN * kD;
    size_t pad_elems = static_cast<size_t>(kBH) * kNPad * kD;
    double bytes_gib = static_cast<double>((3 * pad_elems + in_elems) *
                                           sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    __nv_bfloat16* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, in_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_q, pad_elems);
    init_bf16(d_k, pad_elems);
    init_bf16(d_v, pad_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch_cutile_padded_out60<QRows, false, ConstNegInf>(d_q, d_k, d_v, d_out);
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
        launch_cutile_padded_out60<QRows, false, ConstNegInf>(d_q, d_k, d_v, d_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.validate) {
        validate_unpadded(d_q, d_k, d_v, d_out, false);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_out, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));

    double flops = 4.0 * kBH * kN * kN * kD;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-5s qrows=%d grid=(%d,%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.3f TF/s checksum=%.4f\n",
        name, QRows, ceildiv(kN, QRows), kBH, bytes_gib, best_ms, median_ms,
        tflops, __bfloat162float(checksum));
}

template <int QRows>
void run_variant(const Options& opts, const char* name) {
    size_t elems = static_cast<size_t>(kBH) * kNPad * kD;
    double bytes_gib = static_cast<double>(4 * elems * sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    __nv_bfloat16* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, elems * sizeof(__nv_bfloat16)));
    init_bf16(d_q, elems);
    init_bf16(d_k, elems);
    init_bf16(d_v, elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch_cutile<QRows>(d_q, d_k, d_v, d_out);
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
        launch_cutile<QRows>(d_q, d_k, d_v, d_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.validate) {
        validate_q16(d_q, d_k, d_v, d_out);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_out, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));

    double flops = 4.0 * kBH * kN * kN * kD;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-4s qrows=%d grid=(%d,%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.3f TF/s checksum=%.4f\n",
        name, QRows, ceildiv(kN, QRows), kBH, bytes_gib, best_ms, median_ms,
        tflops, __bfloat162float(checksum));
}

template <int QRows>
void run_masked_variant(const Options& opts, const char* name) {
    size_t elems = static_cast<size_t>(kBH) * kN * kD;
    double bytes_gib = static_cast<double>(4 * elems * sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    __nv_bfloat16* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, elems * sizeof(__nv_bfloat16)));
    init_bf16(d_q, elems);
    init_bf16(d_k, elems);
    init_bf16(d_v, elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < opts.warmup; ++i) {
        launch_cutile_masked<QRows>(d_q, d_k, d_v, d_out);
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
        launch_cutile_masked<QRows>(d_q, d_k, d_v, d_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.validate) {
        validate_unpadded(d_q, d_k, d_v, d_out);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_out, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));

    double flops = 4.0 * kBH * kN * kN * kD;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-4s qrows=%d grid=(%d,%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.3f TF/s checksum=%.4f\n",
        name, QRows, ceildiv(kN, QRows), kBH, bytes_gib, best_ms, median_ms,
        tflops, __bfloat162float(checksum));
}

template <int QRows>
void run_padded_pipeline_variant(const Options& opts, const char* name) {
    size_t in_elems = static_cast<size_t>(kBH) * kN * kD;
    size_t pad_elems = static_cast<size_t>(kBH) * kNPad * kD;
    double bytes_gib = static_cast<double>((3 * in_elems + 3 * pad_elems + in_elems) *
                                           sizeof(__nv_bfloat16)) /
                       (1024.0 * 1024.0 * 1024.0);
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    __nv_bfloat16* d_qp = nullptr;
    __nv_bfloat16* d_kp = nullptr;
    __nv_bfloat16* d_vp = nullptr;
    __nv_bfloat16* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, in_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k, in_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v, in_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_qp, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_kp, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_vp, pad_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, in_elems * sizeof(__nv_bfloat16)));
    init_bf16(d_q, in_elems);
    init_bf16(d_k, in_elems);
    init_bf16(d_v, in_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto launch_pipeline = [&]() {
        long long total = static_cast<long long>(pad_elems);
        pad3_freq60_to64_kernel<<<ceildiv(static_cast<int>(pad_elems), kInitTile), 1>>>(
            d_q, d_k, d_v, d_qp, d_kp, d_vp, total);
        launch_cutile_padded_out60<QRows>(d_qp, d_kp, d_vp, d_out);
    };

    for (int i = 0; i < opts.warmup; ++i) {
        launch_pipeline();
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
        launch_pipeline();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    if (opts.validate) {
        validate_unpadded(d_q, d_k, d_v, d_out);
    }

    __nv_bfloat16 checksum{};
    CUDA_CHECK(cudaMemcpy(&checksum, d_out, sizeof(checksum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_qp));
    CUDA_CHECK(cudaFree(d_kp));
    CUDA_CHECK(cudaFree(d_vp));
    CUDA_CHECK(cudaFree(d_out));

    double flops = 4.0 * kBH * kN * kN * kD;
    float best_ms = *std::min_element(times_ms.begin(), times_ms.end());
    float median_ms = percentile(times_ms, 0.5f);
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::printf(
        "%-5s qrows=%d grid=(%d,%d) mem=%.2f GiB best=%.3f ms median=%.3f ms %.3f TF/s checksum=%.4f\n",
        name, QRows, ceildiv(kN, QRows), kBH, bytes_gib, best_ms, median_ms,
        tflops, __bfloat162float(checksum));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        if (opts.variant == "all" || opts.variant == "q8") {
            run_variant<8>(opts, "q8");
        }
        if (opts.variant == "all" || opts.variant == "q16") {
            run_variant<16>(opts, "q16");
        }
        if (opts.variant == "all" || opts.variant == "q32") {
            run_variant<32>(opts, "q32");
        }
        if (opts.variant == "all" || opts.variant == "q64") {
            run_variant<64>(opts, "q64");
        }
        if (opts.variant == "all" || opts.variant == "q128") {
            run_variant<128>(opts, "q128");
        }
        if (opts.variant == "all" || opts.variant == "q8m") {
            run_masked_variant<8>(opts, "q8m");
        }
        if (opts.variant == "all" || opts.variant == "q16m") {
            run_masked_variant<16>(opts, "q16m");
        }
        if (opts.variant == "all" || opts.variant == "q32m") {
            run_masked_variant<32>(opts, "q32m");
        }
        if (opts.variant == "all" || opts.variant == "q64m") {
            run_masked_variant<64>(opts, "q64m");
        }
        if (opts.variant == "all" || opts.variant == "q128m") {
            run_masked_variant<128>(opts, "q128m");
        }
        if (opts.variant == "all" || opts.variant == "q8p") {
            run_padded_pipeline_variant<8>(opts, "q8p");
        }
        if (opts.variant == "all" || opts.variant == "q16p") {
            run_padded_pipeline_variant<16>(opts, "q16p");
        }
        if (opts.variant == "all" || opts.variant == "q32p") {
            run_padded_pipeline_variant<32>(opts, "q32p");
        }
        if (opts.variant == "all" || opts.variant == "q64p") {
            run_padded_pipeline_variant<64>(opts, "q64p");
        }
        if (opts.variant == "all" || opts.variant == "q128p") {
            run_padded_pipeline_variant<128>(opts, "q128p");
        }
        if (opts.variant == "all" || opts.variant == "q8s") {
            run_padded_out60_source_variant<8>(opts, "q8s");
        }
        if (opts.variant == "all" || opts.variant == "q16s") {
            run_padded_out60_source_variant<16>(opts, "q16s");
        }
        if (opts.variant == "all" || opts.variant == "q16sc") {
            run_padded_out60_source_variant<16, true>(opts, "q16sc");
        }
        if (opts.variant == "all" || opts.variant == "q32s") {
            run_padded_out60_source_variant<32>(opts, "q32s");
        }
        if (opts.variant == "all" || opts.variant == "q64s") {
            run_padded_out60_source_variant<64>(opts, "q64s");
        }
        if (opts.variant == "all" || opts.variant == "q128s") {
            run_padded_out60_source_variant<128>(opts, "q128s");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
