#pragma once

#include "tensor.h"

namespace cudasep::stft_tile {

Tensor hann_window(int size);

Tensor stft(const Tensor& signal,
            int n_fft,
            int hop_length,
            int win_length,
            const Tensor& window,
            bool center = true,
            bool normalized = false);

Tensor istft(const Tensor& complex_spec,
             int n_fft,
             int hop_length,
             int win_length,
             const Tensor& window,
             int64_t length = -1,
             bool center = true,
             bool normalized = false);

}  // namespace cudasep::stft_tile
