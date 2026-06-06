#include "stft_cuda_tile.h"

#include "cuda_tile.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace cudasep::stft_tile {
namespace {

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kTile = 256;
using I64Tile = ct::tile<long long, ct::shape<kTile>>;
using F32Tile = ct::tile<float, ct::shape<kTile>>;

constexpr int kFixedFft = 2048;
constexpr int kFixedHalfFft = kFixedFft / 2;
constexpr int kFixedFreqBins = kFixedHalfFft + 1;
constexpr float kTwoPi = 6.28318530717958647692f;

struct ComplexTile {
    F32Tile r;
    F32Tile i;
};

static inline int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static Tensor ensure_f32(const Tensor& x) {
    if (x.dtype() == DType::Float32) {
        return x.contiguous();
    }
    return x.to_f32().contiguous();
}

static bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' &&
           std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0;
}

static bool istft_window_sum_cache_enabled() {
    static int disabled =
        env_flag_enabled("CUDASEP_DISABLE_ISTFT_WINDOW_SUM_CACHE") ? 1 : 0;
    return disabled == 0;
}

static __tile__ I64Tile bit_reverse_11(I64Tile x) {
    x = ((x & 0x555ll) << 1) | ((x >> 1) & 0x555ll);
    x = ((x & 0x333ll) << 2) | ((x >> 2) & 0x333ll);
    x = ((x & 0x0f0fll) << 4) | ((x >> 4) & 0x0f0fll);
    x = ((x & 0x00ffll) << 8) | ((x >> 8) & 0x00ffll);
    return (x >> 5) & 2047ll;
}

__tile_global__ void fft2048_stage_kernel(const float* __restrict__ src,
                                          float* __restrict__ dst,
                                          long long total,
                                          int len,
                                          int inverse) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto n = idx % kFixedFft;
    int half = len / 2;
    auto j = n % len;
    auto lower = j < half;
    auto j2 = ct::select(lower, j, j - half);
    auto base = idx - j + j2;
    auto upper = base + half;

    auto ur = ct::load_masked(src + base * 2, in_bounds);
    auto ui = ct::load_masked(src + base * 2 + 1, in_bounds);
    auto vr = ct::load_masked(src + upper * 2, in_bounds);
    auto vi = ct::load_masked(src + upper * 2 + 1, in_bounds);
    auto angle = ct::element_cast<float>(j2) *
                 ((inverse ? kTwoPi : -kTwoPi) / static_cast<float>(len));
    auto c = ct::cos(angle);
    auto s = ct::sin(angle);
    auto tr = c * vr - s * vi;
    auto ti = c * vi + s * vr;
    auto out_r = ct::select(lower, ur + tr, ur - tr);
    auto out_i = ct::select(lower, ui + ti, ui - ti);

    ct::store_masked(dst + idx * 2, out_r, in_bounds);
    ct::store_masked(dst + idx * 2 + 1, out_i, in_bounds);
}

__tile_global__ void fft2048_two_stage_kernel(const float* __restrict__ src,
                                              float* __restrict__ dst,
                                              long long total,
                                              int len,
                                              int inverse) {
    src = ct::assume_aligned(src, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto n = idx % kFixedFft;
    int half1 = len / 2;
    int len2 = len * 2;

    auto j_b = n % len2;
    auto lower_b = j_b < len;
    auto jb2 = ct::select(lower_b, j_b, j_b - len);
    auto q0 = idx - j_b + jb2;
    auto q1 = q0 + len;

    auto n0 = q0 % kFixedFft;
    auto j0 = n0 % len;
    auto lower0 = j0 < half1;
    auto j02 = ct::select(lower0, j0, j0 - half1);
    auto base0 = q0 - j0 + j02;
    auto upper0 = base0 + half1;
    auto u0r = ct::load_masked(src + base0 * 2, in_bounds);
    auto u0i = ct::load_masked(src + base0 * 2 + 1, in_bounds);
    auto v0r = ct::load_masked(src + upper0 * 2, in_bounds);
    auto v0i = ct::load_masked(src + upper0 * 2 + 1, in_bounds);
    auto angle0 = ct::element_cast<float>(j02) *
                  ((inverse ? kTwoPi : -kTwoPi) / static_cast<float>(len));
    auto c0 = ct::cos(angle0);
    auto s0 = ct::sin(angle0);
    auto t0r = c0 * v0r - s0 * v0i;
    auto t0i = c0 * v0i + s0 * v0r;
    auto a0r = ct::select(lower0, u0r + t0r, u0r - t0r);
    auto a0i = ct::select(lower0, u0i + t0i, u0i - t0i);

    auto n1 = q1 % kFixedFft;
    auto j1 = n1 % len;
    auto lower1 = j1 < half1;
    auto j12 = ct::select(lower1, j1, j1 - half1);
    auto base1 = q1 - j1 + j12;
    auto upper1 = base1 + half1;
    auto u1r = ct::load_masked(src + base1 * 2, in_bounds);
    auto u1i = ct::load_masked(src + base1 * 2 + 1, in_bounds);
    auto v1r = ct::load_masked(src + upper1 * 2, in_bounds);
    auto v1i = ct::load_masked(src + upper1 * 2 + 1, in_bounds);
    auto angle1 = ct::element_cast<float>(j12) *
                  ((inverse ? kTwoPi : -kTwoPi) / static_cast<float>(len));
    auto c1 = ct::cos(angle1);
    auto s1 = ct::sin(angle1);
    auto t1r = c1 * v1r - s1 * v1i;
    auto t1i = c1 * v1i + s1 * v1r;
    auto a1r = ct::select(lower1, u1r + t1r, u1r - t1r);
    auto a1i = ct::select(lower1, u1i + t1i, u1i - t1i);

    auto angle_b = ct::element_cast<float>(jb2) *
                   ((inverse ? kTwoPi : -kTwoPi) / static_cast<float>(len2));
    auto cb = ct::cos(angle_b);
    auto sb = ct::sin(angle_b);
    auto tbr = cb * a1r - sb * a1i;
    auto tbi = cb * a1i + sb * a1r;
    auto out_r = ct::select(lower_b, a0r + tbr, a0r - tbr);
    auto out_i = ct::select(lower_b, a0i + tbi, a0i - tbi);

    ct::store_masked(dst + idx * 2, out_r, in_bounds);
    ct::store_masked(dst + idx * 2 + 1, out_i, in_bounds);
}

template <typename Mask>
static __tile__ ComplexTile stft2048_window_bitrev_len2_two_stage_value(
    const float* __restrict__ signal,
    const float* __restrict__ window,
    I64Tile idx,
    Mask in_bounds,
    int signal_length,
    int hop_length,
    int T) {
    auto n = idx % kFixedFft;
    auto j_b = n & 3LL;
    auto lower_b = j_b < 2LL;
    auto jb2 = ct::select(lower_b, j_b, j_b - 2LL);
    auto q0 = idx - j_b + jb2;
    auto q1 = q0 + 2LL;

    auto j0 = q0 & 1LL;
    auto base0 = q0 - j0;
    auto lower0 = j0 == 0LL;

    auto p0u = base0 % kFixedFft;
    auto n0u = bit_reverse_11(p0u);
    auto frame0u = base0 / kFixedFft;
    auto t0u = frame0u % T;
    auto b0u = frame0u / T;
    auto src0u = b0u * signal_length + t0u * hop_length + n0u;
    auto u0r = ct::load_masked(signal + src0u, in_bounds) *
               ct::load_masked(window + n0u, in_bounds);
    auto u0i = u0r * 0.0f;

    auto q0v = base0 + 1LL;
    auto p0v = q0v % kFixedFft;
    auto n0v = bit_reverse_11(p0v);
    auto frame0v = q0v / kFixedFft;
    auto t0v = frame0v % T;
    auto b0v = frame0v / T;
    auto src0v = b0v * signal_length + t0v * hop_length + n0v;
    auto v0r = ct::load_masked(signal + src0v, in_bounds) *
               ct::load_masked(window + n0v, in_bounds);
    auto v0i = v0r * 0.0f;

    auto a0r = ct::select(lower0, u0r + v0r, u0r - v0r);
    auto a0i = ct::select(lower0, u0i + v0i, u0i - v0i);

    auto j1 = q1 & 1LL;
    auto base1 = q1 - j1;
    auto lower1 = j1 == 0LL;

    auto p1u = base1 % kFixedFft;
    auto n1u = bit_reverse_11(p1u);
    auto frame1u = base1 / kFixedFft;
    auto t1u = frame1u % T;
    auto b1u = frame1u / T;
    auto src1u = b1u * signal_length + t1u * hop_length + n1u;
    auto u1r = ct::load_masked(signal + src1u, in_bounds) *
               ct::load_masked(window + n1u, in_bounds);
    auto u1i = u1r * 0.0f;

    auto q1v = base1 + 1LL;
    auto p1v = q1v % kFixedFft;
    auto n1v = bit_reverse_11(p1v);
    auto frame1v = q1v / kFixedFft;
    auto t1v = frame1v % T;
    auto b1v = frame1v / T;
    auto src1v = b1v * signal_length + t1v * hop_length + n1v;
    auto v1r = ct::load_masked(signal + src1v, in_bounds) *
               ct::load_masked(window + n1v, in_bounds);
    auto v1i = v1r * 0.0f;

    auto a1r = ct::select(lower1, u1r + v1r, u1r - v1r);
    auto a1i = ct::select(lower1, u1i + v1i, u1i - v1i);

    constexpr float kCosHalfPiF = -4.371138828673793e-08f;
    auto zero_twiddle = jb2 == 0LL;
    auto rot_r = kCosHalfPiF * a1r + a1i;
    auto rot_i = kCosHalfPiF * a1i - a1r;
    auto tbr = ct::select(zero_twiddle, a1r, rot_r);
    auto tbi = ct::select(zero_twiddle, a1i, rot_i);
    return {
        ct::select(lower_b, a0r + tbr, a0r - tbr),
        ct::select(lower_b, a0i + tbi, a0i - tbi),
    };
}

template <typename SpecT, typename Mask>
static __tile__ ComplexTile istft2048_prepare_bitrev_len2_two_stage_value(
    const SpecT* __restrict__ spec,
    I64Tile idx,
    Mask in_bounds,
    int T,
    float spec_scale) {
    auto n = idx % kFixedFft;
    auto j_b = n & 3LL;
    auto lower_b = j_b < 2LL;
    auto jb2 = ct::select(lower_b, j_b, j_b - 2LL);
    auto q0 = idx - j_b + jb2;
    auto q1 = q0 + 2LL;

    auto j0 = q0 & 1LL;
    auto base0 = q0 - j0;
    auto lower0 = j0 == 0LL;

    auto p0u = base0 % kFixedFft;
    auto n0u = bit_reverse_11(p0u);
    auto frame0u = base0 / kFixedFft;
    auto t0u = frame0u % T;
    auto b0u = frame0u / T;
    auto mirrored0u = n0u > kFixedHalfFft;
    auto f0u = ct::select(mirrored0u, kFixedFft - n0u, n0u);
    auto src0u = (b0u * kFixedFreqBins * (long long)T +
                  f0u * (long long)T + t0u) * 2;
    auto u0r = ct::element_cast<float>(ct::load_masked(spec + src0u, in_bounds)) * spec_scale;
    auto u0i = ct::element_cast<float>(ct::load_masked(spec + src0u + 1, in_bounds)) * spec_scale;
    u0i = ct::select(mirrored0u, -u0i, u0i);

    auto q0v = base0 + 1LL;
    auto p0v = q0v % kFixedFft;
    auto n0v = bit_reverse_11(p0v);
    auto frame0v = q0v / kFixedFft;
    auto t0v = frame0v % T;
    auto b0v = frame0v / T;
    auto mirrored0v = n0v > kFixedHalfFft;
    auto f0v = ct::select(mirrored0v, kFixedFft - n0v, n0v);
    auto src0v = (b0v * kFixedFreqBins * (long long)T +
                  f0v * (long long)T + t0v) * 2;
    auto v0r = ct::element_cast<float>(ct::load_masked(spec + src0v, in_bounds)) * spec_scale;
    auto v0i = ct::element_cast<float>(ct::load_masked(spec + src0v + 1, in_bounds)) * spec_scale;
    v0i = ct::select(mirrored0v, -v0i, v0i);

    auto a0r = ct::select(lower0, u0r + v0r, u0r - v0r);
    auto a0i = ct::select(lower0, u0i + v0i, u0i - v0i);

    auto j1 = q1 & 1LL;
    auto base1 = q1 - j1;
    auto lower1 = j1 == 0LL;

    auto p1u = base1 % kFixedFft;
    auto n1u = bit_reverse_11(p1u);
    auto frame1u = base1 / kFixedFft;
    auto t1u = frame1u % T;
    auto b1u = frame1u / T;
    auto mirrored1u = n1u > kFixedHalfFft;
    auto f1u = ct::select(mirrored1u, kFixedFft - n1u, n1u);
    auto src1u = (b1u * kFixedFreqBins * (long long)T +
                  f1u * (long long)T + t1u) * 2;
    auto u1r = ct::element_cast<float>(ct::load_masked(spec + src1u, in_bounds)) * spec_scale;
    auto u1i = ct::element_cast<float>(ct::load_masked(spec + src1u + 1, in_bounds)) * spec_scale;
    u1i = ct::select(mirrored1u, -u1i, u1i);

    auto q1v = base1 + 1LL;
    auto p1v = q1v % kFixedFft;
    auto n1v = bit_reverse_11(p1v);
    auto frame1v = q1v / kFixedFft;
    auto t1v = frame1v % T;
    auto b1v = frame1v / T;
    auto mirrored1v = n1v > kFixedHalfFft;
    auto f1v = ct::select(mirrored1v, kFixedFft - n1v, n1v);
    auto src1v = (b1v * kFixedFreqBins * (long long)T +
                  f1v * (long long)T + t1v) * 2;
    auto v1r = ct::element_cast<float>(ct::load_masked(spec + src1v, in_bounds)) * spec_scale;
    auto v1i = ct::element_cast<float>(ct::load_masked(spec + src1v + 1, in_bounds)) * spec_scale;
    v1i = ct::select(mirrored1v, -v1i, v1i);

    auto a1r = ct::select(lower1, u1r + v1r, u1r - v1r);
    auto a1i = ct::select(lower1, u1i + v1i, u1i - v1i);

    constexpr float kCosHalfPiF = -4.371138828673793e-08f;
    auto zero_twiddle = jb2 == 0LL;
    auto rot_r = kCosHalfPiF * a1r - a1i;
    auto rot_i = kCosHalfPiF * a1i + a1r;
    auto tbr = ct::select(zero_twiddle, a1r, rot_r);
    auto tbi = ct::select(zero_twiddle, a1i, rot_i);
    return {
        ct::select(lower_b, a0r + tbr, a0r - tbr),
        ct::select(lower_b, a0i + tbi, a0i - tbi),
    };
}

__tile_global__ void stft2048_window_bitrev_two_stage_kernel(
    const float* __restrict__ signal,
    const float* __restrict__ window,
    float* __restrict__ dst,
    long long total,
    int signal_length,
    int hop_length,
    int T) {
    signal = ct::assume_aligned(signal, 16_ic);
    window = ct::assume_aligned(window, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto out = stft2048_window_bitrev_len2_two_stage_value(
        signal, window, idx, in_bounds, signal_length, hop_length, T);
    ct::store_masked(dst + idx * 2, out.r, in_bounds);
    ct::store_masked(dst + idx * 2 + 1, out.i, in_bounds);
}

__tile_global__ void istft2048_prepare_bitrev_two_stage_kernel(
    const float* __restrict__ spec,
    float* __restrict__ dst,
    long long total,
    int T,
    float spec_scale) {
    spec = ct::assume_aligned(spec, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto out = istft2048_prepare_bitrev_len2_two_stage_value(
        spec, idx, in_bounds, T, spec_scale);
    ct::store_masked(dst + idx * 2, out.r, in_bounds);
    ct::store_masked(dst + idx * 2 + 1, out.i, in_bounds);
}

__tile_global__ void istft2048_prepare_bitrev_two_stage_bf16_kernel(
    const __nv_bfloat16* __restrict__ spec,
    float* __restrict__ dst,
    long long total,
    int T,
    float spec_scale) {
    spec = ct::assume_aligned(spec, 16_ic);
    dst = ct::assume_aligned(dst, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto out = istft2048_prepare_bitrev_len2_two_stage_value(
        spec, idx, in_bounds, T, spec_scale);
    ct::store_masked(dst + idx * 2, out.r, in_bounds);
    ct::store_masked(dst + idx * 2 + 1, out.i, in_bounds);
}

__tile_global__ void stft2048_extract_r2c_kernel(const float* __restrict__ scratch,
                                                 float* __restrict__ output,
                                                 long long total,
                                                 int T) {
    scratch = ct::assume_aligned(scratch, 16_ic);
    output = ct::assume_aligned(output, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto t = idx % T;
    auto f = (idx / T) % kFixedFreqBins;
    auto b = idx / ((long long)T * kFixedFreqBins);
    auto frame = b * T + t;
    auto src_idx = (frame * kFixedFft + f) * 2;
    auto dst_idx = idx * 2;

    ct::store_masked(output + dst_idx,
                     ct::load_masked(scratch + src_idx, in_bounds),
                     in_bounds);
    ct::store_masked(output + dst_idx + 1,
                     ct::load_masked(scratch + src_idx + 1, in_bounds),
                     in_bounds);
}

__tile_global__ void istft2048_overlap_add_kernel(const float* __restrict__ scratch,
                                                  const float* __restrict__ window,
                                                  float* __restrict__ output,
                                                  long long total,
                                                  int T,
                                                  int hop_length,
                                                  int signal_length) {
    scratch = ct::assume_aligned(scratch, 16_ic);
    window = ct::assume_aligned(window, 16_ic);
    output = ct::assume_aligned(output, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto n = idx % kFixedFft;
    auto frame = idx / kFixedFft;
    auto t = frame % T;
    auto b = frame / T;
    auto out_pos = t * hop_length + n;
    auto out_idx = b * signal_length + out_pos;
    float norm = 1.0f / static_cast<float>(kFixedFft);
    auto value = ct::load_masked(scratch + idx * 2, in_bounds) *
                 ct::load_masked(window + n, in_bounds) * norm;
    ct::atomic_add_masked<ct::memory_order::relaxed>(output + out_idx, value, in_bounds);
}

__tile_global__ void scale_kernel(float* __restrict__ data,
                                  long long total,
                                  float factor) {
    data = ct::assume_aligned(data, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto values = ct::load_masked(data + idx, in_bounds) * factor;
    ct::store_masked(data + idx, values, in_bounds);
}

__tile_global__ void window_sum_kernel(const float* __restrict__ window,
                                       float* __restrict__ window_sum,
                                       int n_fft,
                                       int hop_length,
                                       int T,
                                       int signal_length) {
    window = ct::assume_aligned(window, 16_ic);
    window_sum = ct::assume_aligned(window_sum, 16_ic);

    I64Tile pos = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = pos < signal_length;
    auto sum = ct::zeros<F32Tile>();

    for (int t = 0; t < T; t++) {
        auto n = pos - (long long)t * hop_length;
        auto valid = in_bounds && (n >= 0) && (n < n_fft);
        auto w = ct::load_masked(window + n, valid);
        sum = sum + ct::select(valid, w * w, w * 0.0f);
    }

    ct::store_masked(window_sum + pos, sum, in_bounds);
}

__tile_global__ void normalize_by_window_kernel(float* __restrict__ output,
                                                const float* __restrict__ window_sum,
                                                long long total,
                                                int signal_length) {
    output = ct::assume_aligned(output, 16_ic);
    window_sum = ct::assume_aligned(window_sum, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;
    auto pos = idx % signal_length;
    auto ws = ct::load_masked(window_sum + pos, in_bounds);
    auto values = ct::load_masked(output + idx, in_bounds);
    auto normalized = ct::select(ws > 1.0e-8f, values / ws, values);
    ct::store_masked(output + idx, normalized, in_bounds);
}

__tile_global__ void hann_window_kernel(float* __restrict__ out, int size) {
    out = ct::assume_aligned(out, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < size;
    F32Tile idx_f = ct::element_cast<float>(idx);
    auto phase = idx_f * (6.28318530717958647692f / (float)size);
    auto values = 0.5f * (1.0f - ct::cos(phase));
    ct::store_masked(out + idx, values, in_bounds);
}

}  // namespace

Tensor hann_window(int size) {
    Tensor out = Tensor::empty({(int64_t)size}, DType::Float32);
    if (size <= 0) {
        return out;
    }

    if (size == 1) {
        out.fill_(1.0f);
        return out;
    }

    hann_window_kernel<<<(int)ceildiv(size, kTile), 1>>>(out.data_f32(), size);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor stft(const Tensor& signal,
            int n_fft,
            int hop_length,
            int win_length,
            const Tensor& window,
            bool center,
            bool normalized) {
    Tensor sig = ensure_f32(signal);
    Tensor win = ensure_f32(window);

    if (sig.ndim() == 1) {
        sig = sig.reshape({1, sig.size(0)});
    }

    int B = (int)sig.size(0);
    if (center) {
        int pad_amount = n_fft / 2;
        sig = sig.pad_reflect({(int64_t)pad_amount, (int64_t)pad_amount});
    }

    int padded_length = (int)sig.size(1);
    int T = (padded_length - n_fft) / hop_length + 1;
    int F = n_fft / 2 + 1;

    if (n_fft != kFixedFft || win_length != kFixedFft ||
        F != kFixedFreqBins || win.numel() != kFixedFft) {
        throw std::runtime_error("custom STFT only supports n_fft=2048, win_length=2048");
    }

    Tensor output = Tensor::empty({(int64_t)B, (int64_t)F, (int64_t)T, 2},
                                  DType::Float32);
    int frames = B * T;
    int64_t fft_elems = (int64_t)frames * kFixedFft;
    Tensor scratch_a = Tensor::empty({(int64_t)frames, kFixedFft, 2}, DType::Float32);
    Tensor scratch_b = Tensor::empty({(int64_t)frames, kFixedFft, 2}, DType::Float32);

    stft2048_window_bitrev_two_stage_kernel
        <<<(int)ceildiv(fft_elems, kTile), 1>>>(
            sig.data_f32(), win.data_f32(), scratch_b.data_f32(), fft_elems,
            padded_length, hop_length, T);
    CUDA_CHECK(cudaGetLastError());

    float* src = scratch_b.data_f32();
    float* dst = scratch_a.data_f32();
    for (int len = 8; len < kFixedFft; len <<= 2) {
        fft2048_two_stage_kernel<<<(int)ceildiv(fft_elems, kTile), 1>>>(
            src, dst, fft_elems, len, 0);
        CUDA_CHECK(cudaGetLastError());
        std::swap(src, dst);
    }
    fft2048_stage_kernel<<<(int)ceildiv(fft_elems, kTile), 1>>>(
        src, dst, fft_elems, kFixedFft, 0);
    CUDA_CHECK(cudaGetLastError());
    std::swap(src, dst);

    int64_t complex_elems = (int64_t)B * F * T;
    stft2048_extract_r2c_kernel<<<(int)ceildiv(complex_elems, kTile), 1>>>(
        src, output.data_f32(), complex_elems, T);
    CUDA_CHECK(cudaGetLastError());

    if (normalized) {
        float norm_factor = 1.0f / std::sqrt((float)n_fft);
        int64_t N = output.numel();
        scale_kernel<<<(int)ceildiv(N, kTile), 1>>>(output.data_f32(), N, norm_factor);
        CUDA_CHECK(cudaGetLastError());
    }

    return output;
}

Tensor istft(const Tensor& complex_spec,
             int n_fft,
             int hop_length,
             int win_length,
             const Tensor& window,
             int64_t length,
             bool center,
             bool normalized) {
    Tensor spec = (complex_spec.dtype() == DType::BFloat16)
        ? complex_spec.contiguous()
        : ensure_f32(complex_spec);
    Tensor win = ensure_f32(window);

    int B = (int)spec.size(0);
    int F = (int)spec.size(1);
    int T = (int)spec.size(2);

    if (n_fft != kFixedFft || win_length != kFixedFft ||
        F != kFixedFreqBins || win.numel() != kFixedFft) {
        throw std::runtime_error("custom iSTFT only supports n_fft=2048, win_length=2048");
    }

    float spec_scale = normalized ? std::sqrt((float)n_fft) : 1.0f;
    int signal_length = n_fft + (T - 1) * hop_length;
    Tensor output = Tensor::zeros({(int64_t)B, (int64_t)signal_length}, DType::Float32);
    int frames = B * T;
    int64_t fft_elems = (int64_t)frames * kFixedFft;
    Tensor scratch_a = Tensor::empty({(int64_t)frames, kFixedFft, 2}, DType::Float32);
    Tensor scratch_b = Tensor::empty({(int64_t)frames, kFixedFft, 2}, DType::Float32);

    if (spec.dtype() == DType::BFloat16) {
        istft2048_prepare_bitrev_two_stage_bf16_kernel
            <<<(int)ceildiv(fft_elems, kTile), 1>>>(
                spec.data_bf16(), scratch_b.data_f32(), fft_elems, T, spec_scale);
    } else {
        istft2048_prepare_bitrev_two_stage_kernel
            <<<(int)ceildiv(fft_elems, kTile), 1>>>(
                spec.data_f32(), scratch_b.data_f32(), fft_elems, T, spec_scale);
    }
    CUDA_CHECK(cudaGetLastError());

    float* src = scratch_b.data_f32();
    float* dst = scratch_a.data_f32();
    for (int len = 8; len < kFixedFft; len <<= 2) {
        fft2048_two_stage_kernel<<<(int)ceildiv(fft_elems, kTile), 1>>>(
            src, dst, fft_elems, len, 1);
        CUDA_CHECK(cudaGetLastError());
        std::swap(src, dst);
    }
    fft2048_stage_kernel<<<(int)ceildiv(fft_elems, kTile), 1>>>(
        src, dst, fft_elems, kFixedFft, 1);
    CUDA_CHECK(cudaGetLastError());
    std::swap(src, dst);

    istft2048_overlap_add_kernel<<<(int)ceildiv(fft_elems, kTile), 1>>>(
        src, win.data_f32(), output.data_f32(), fft_elems,
        T, hop_length, signal_length);
    CUDA_CHECK(cudaGetLastError());

    Tensor window_sum;
    if (istft_window_sum_cache_enabled()) {
        static Tensor cached_window_sum;
        static const void* cached_window_ptr = nullptr;
        static int cached_n_fft = 0;
        static int cached_hop_length = 0;
        static int cached_T = 0;
        static int cached_signal_length = 0;
        bool cache_hit = !cached_window_sum.is_empty() &&
                         cached_window_ptr == win.data_ptr() &&
                         cached_n_fft == n_fft &&
                         cached_hop_length == hop_length &&
                         cached_T == T &&
                         cached_signal_length == signal_length;
        if (!cache_hit) {
            cached_window_sum =
                Tensor::empty({(int64_t)signal_length}, DType::Float32);
            window_sum_kernel<<<(int)ceildiv(signal_length, kTile), 1>>>(
                win.data_f32(), cached_window_sum.data_f32(), n_fft, hop_length,
                T, signal_length);
            CUDA_CHECK(cudaGetLastError());
            cached_window_ptr = win.data_ptr();
            cached_n_fft = n_fft;
            cached_hop_length = hop_length;
            cached_T = T;
            cached_signal_length = signal_length;
        }
        window_sum = cached_window_sum;
    } else {
        window_sum = Tensor::empty({(int64_t)signal_length}, DType::Float32);
        window_sum_kernel<<<(int)ceildiv(signal_length, kTile), 1>>>(
            win.data_f32(), window_sum.data_f32(), n_fft, hop_length, T,
            signal_length);
        CUDA_CHECK(cudaGetLastError());
    }

    int64_t output_elems = output.numel();
    normalize_by_window_kernel<<<(int)ceildiv(output_elems, kTile), 1>>>(
        output.data_f32(), window_sum.data_f32(), output_elems, signal_length);
    CUDA_CHECK(cudaGetLastError());

    if (center) {
        int pad = n_fft / 2;
        output = output.slice(1, pad, signal_length - pad);
        signal_length = (int)output.size(1);
    }

    if (length > 0) {
        if (length < signal_length) {
            output = output.slice(1, 0, length);
        } else if (length > signal_length) {
            output = output.pad({0, length - signal_length}, 0.0f);
        }
    }

    return output;
}

}  // namespace cudasep::stft_tile
