#include "stft_cuda_tile.h"

#include "cuda_context.h"
#include "cuda_tile.h"

#include <cmath>
#include <cufft.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace cudasep::stft_tile {
namespace {

#define CUFFT_CHECK(call) do { \
    cufftResult _err = (call); \
    if (_err != CUFFT_SUCCESS) { \
        throw std::runtime_error(std::string("cuFFT error: ") + \
            std::to_string((int)_err) + " at " + __FILE__ + ":" + \
            std::to_string(__LINE__)); \
    } \
} while(0)

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int kTile = 256;
using I64Tile = ct::tile<long long, ct::shape<kTile>>;
using F32Tile = ct::tile<float, ct::shape<kTile>>;

struct FftPlanKey {
    int n_fft = 0;
    int batch = 0;
    int type = 0;

    bool operator==(const FftPlanKey& other) const {
        return n_fft == other.n_fft && batch == other.batch && type == other.type;
    }
};

struct FftPlanKeyHash {
    size_t operator()(const FftPlanKey& key) const {
        size_t h1 = std::hash<int>{}(key.n_fft);
        size_t h2 = std::hash<int>{}(key.batch);
        size_t h3 = std::hash<int>{}(key.type);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

static cufftHandle get_cached_fft_plan(int n_fft, int batch, cufftType type) {
    static std::mutex plan_mutex;
    static std::unordered_map<FftPlanKey, cufftHandle, FftPlanKeyHash> cache;

    FftPlanKey key{n_fft, batch, (int)type};
    std::lock_guard<std::mutex> lock(plan_mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        CUFFT_CHECK(cufftSetStream(it->second, CudaContext::instance().stream()));
        return it->second;
    }

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n_fft, type, batch));
    CUFFT_CHECK(cufftSetStream(plan, CudaContext::instance().stream()));
    cache.emplace(key, plan);
    return plan;
}

static void* get_fft_complex_buffer(size_t required_bytes) {
    static std::mutex buffer_mutex;
    static void* buffer = nullptr;
    static size_t buffer_size = 0;

    std::lock_guard<std::mutex> lock(buffer_mutex);
    if (required_bytes <= buffer_size && buffer != nullptr) {
        return buffer;
    }
    if (buffer != nullptr) {
        cudaFree(buffer);
        buffer = nullptr;
        buffer_size = 0;
    }
    CUDA_CHECK(cudaMalloc(&buffer, required_bytes));
    buffer_size = required_bytes;
    return buffer;
}

static inline int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static Tensor ensure_f32(const Tensor& x) {
    if (x.dtype() == DType::Float32) {
        return x.contiguous();
    }
    return x.to_f32().contiguous();
}

    __tile_global__ void extract_windowed_frames_kernel(const float* __restrict__ signal,
                                                        const float* __restrict__ window,
                                                        float* __restrict__ frames,
                                                        long long total,
                                                        int signal_length,
                                                        int n_fft,
                                                        int hop_length,
                                                        int T) {
    signal = ct::assume_aligned(signal, 16_ic);
    window = ct::assume_aligned(window, 16_ic);
    frames = ct::assume_aligned(frames, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    auto n = idx % n_fft;
    auto t = (idx / n_fft) % T;
    auto b = idx / ((long long)n_fft * T);
    auto src_idx = b * signal_length + t * hop_length + n;

    auto values = ct::load_masked(signal + src_idx, in_bounds) *
                  ct::load_masked(window + n, in_bounds);
    ct::store_masked(frames + idx, values, in_bounds);
}

__tile_global__ void complex_to_real_imag_kernel(const float* __restrict__ complex_data,
                                                 float* __restrict__ output,
                                                 long long total,
                                                 int F,
                                                 int T) {
    complex_data = ct::assume_aligned(complex_data, 16_ic);
    output = ct::assume_aligned(output, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    auto t = idx % T;
    auto f = (idx / T) % F;
    auto b = idx / ((long long)T * F);
    auto complex_idx = b * T * F + t * F + f;
    auto out_base = (b * F * T + f * T + t) * 2;

    auto complex_base = complex_idx * 2;
    ct::store_masked(output + out_base,
                     ct::load_masked(complex_data + complex_base, in_bounds),
                     in_bounds);
    ct::store_masked(output + out_base + 1,
                     ct::load_masked(complex_data + complex_base + 1, in_bounds),
                     in_bounds);
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

__tile_global__ void real_imag_to_complex_kernel(const float* __restrict__ input,
                                                 float* __restrict__ complex_data,
                                                 long long total,
                                                 int F,
                                                 int T) {
    input = ct::assume_aligned(input, 16_ic);
    complex_data = ct::assume_aligned(complex_data, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    auto t = idx % T;
    auto f = (idx / T) % F;
    auto b = idx / ((long long)T * F);
    auto in_base = (b * F * T + f * T + t) * 2;
    auto complex_idx = b * T * F + t * F + f;
    auto complex_base = complex_idx * 2;

    ct::store_masked(complex_data + complex_base,
                     ct::load_masked(input + in_base, in_bounds),
                     in_bounds);
    ct::store_masked(complex_data + complex_base + 1,
                     ct::load_masked(input + in_base + 1, in_bounds),
                     in_bounds);
}

__tile_global__ void overlap_add_windowed_kernel(const float* __restrict__ frames,
                                                 const float* __restrict__ window,
                                                 float* __restrict__ output,
                                                 long long total,
                                                 int n_fft,
                                                 int hop_length,
                                                 int T,
                                                 int signal_length,
                                                 float scale) {
    frames = ct::assume_aligned(frames, 16_ic);
    window = ct::assume_aligned(window, 16_ic);
    output = ct::assume_aligned(output, 16_ic);

    I64Tile idx = (long long)ct::bid().x * kTile + ct::iota<I64Tile>();
    auto in_bounds = idx < total;

    auto n = idx % n_fft;
    auto t = (idx / n_fft) % T;
    auto b = idx / ((long long)n_fft * T);
    auto out_pos = t * hop_length + n;
    auto valid = in_bounds && (out_pos < signal_length);
    auto values = ct::load_masked(frames + idx, valid) *
                  ct::load_masked(window + n, valid) * scale;
    auto out_idx = b * signal_length + out_pos;

    ct::atomic_add_masked<ct::memory_order::relaxed>(output + out_idx, values, valid);
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
    (void)win_length;

    Tensor sig = ensure_f32(signal);
    Tensor win = ensure_f32(window);

    bool was_1d = (sig.ndim() == 1);
    if (was_1d) {
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

    int64_t frame_elems = (int64_t)B * T * n_fft;
    Tensor frames = Tensor::empty({(int64_t)B, (int64_t)T, (int64_t)n_fft}, DType::Float32);
    extract_windowed_frames_kernel<<<(int)ceildiv(frame_elems, kTile), 1>>>(
        sig.data_f32(), win.data_f32(), frames.data_f32(), frame_elems,
        padded_length, n_fft, hop_length, T);
    CUDA_CHECK(cudaGetLastError());

    int64_t complex_elems = (int64_t)B * T * F;
    cufftComplex* d_complex = static_cast<cufftComplex*>(
        get_fft_complex_buffer((size_t)complex_elems * sizeof(cufftComplex)));

    cufftHandle plan = get_cached_fft_plan(n_fft, B * T, CUFFT_R2C);
    CUFFT_CHECK(cufftExecR2C(plan, frames.data_f32(), d_complex));

    Tensor output = Tensor::empty({(int64_t)B, (int64_t)F, (int64_t)T, 2}, DType::Float32);
    complex_to_real_imag_kernel<<<(int)ceildiv(complex_elems, kTile), 1>>>(
        reinterpret_cast<const float*>(d_complex), output.data_f32(), complex_elems, F, T);
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
    (void)win_length;

    Tensor spec = ensure_f32(complex_spec);
    Tensor win = ensure_f32(window);

    int B = (int)spec.size(0);
    int F = (int)spec.size(1);
    int T = (int)spec.size(2);

    if (normalized) {
        float norm_factor = std::sqrt((float)n_fft);
        int64_t N = spec.numel();
        spec = spec.clone();
        scale_kernel<<<(int)ceildiv(N, kTile), 1>>>(spec.data_f32(), N, norm_factor);
        CUDA_CHECK(cudaGetLastError());
    }

    int64_t complex_elems = (int64_t)B * F * T;
    cufftComplex* d_complex = static_cast<cufftComplex*>(
        get_fft_complex_buffer((size_t)complex_elems * sizeof(cufftComplex)));

    real_imag_to_complex_kernel<<<(int)ceildiv(complex_elems, kTile), 1>>>(
        spec.data_f32(), reinterpret_cast<float*>(d_complex), complex_elems, F, T);
    CUDA_CHECK(cudaGetLastError());

    Tensor frames = Tensor::empty({(int64_t)B * T, (int64_t)n_fft}, DType::Float32);
    cufftHandle plan = get_cached_fft_plan(n_fft, B * T, CUFFT_C2R);
    CUFFT_CHECK(cufftExecC2R(plan, d_complex, frames.data_f32()));

    int64_t frame_elems = frames.numel();
    frames = frames.reshape({(int64_t)B, (int64_t)T, (int64_t)n_fft});

    int signal_length = n_fft + (T - 1) * hop_length;
    Tensor output = Tensor::zeros({(int64_t)B, (int64_t)signal_length}, DType::Float32);
    overlap_add_windowed_kernel<<<(int)ceildiv(frame_elems, kTile), 1>>>(
        frames.data_f32(), win.data_f32(), output.data_f32(), frame_elems,
        n_fft, hop_length, T, signal_length, 1.0f / (float)n_fft);
    CUDA_CHECK(cudaGetLastError());

    Tensor window_sum = Tensor::empty({(int64_t)signal_length}, DType::Float32);
    window_sum_kernel<<<(int)ceildiv(signal_length, kTile), 1>>>(
        win.data_f32(), window_sum.data_f32(), n_fft, hop_length, T, signal_length);
    CUDA_CHECK(cudaGetLastError());

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
