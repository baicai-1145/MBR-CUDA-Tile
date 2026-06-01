#pragma once
#include "tensor.h"
#include <string>

namespace cudasep {

struct AudioData {
    Tensor samples;     // [channels, num_samples] Float32 on GPU
    int sample_rate;
    int channels;
    int64_t num_samples;
};

// Read WAV file. Returns samples as [channels, num_samples] Float32 Tensor on GPU.
AudioData load_wav(const std::string& path);

// Write audio to WAV file. samples: [channels, num_samples] Float32 (on GPU, will be copied to CPU).
void save_wav(const std::string& path, const Tensor& samples, int sample_rate);

} // namespace cudasep
