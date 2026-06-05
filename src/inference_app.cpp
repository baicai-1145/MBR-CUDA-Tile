#include "inference_app.h"

#include "chunk_cuda_tile.h"
#include "cuda_context.h"
#include "mbr_cuda_tile.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include <sstream>
#include <utility>

namespace fs = std::filesystem;

namespace cudasep::app {

namespace {

double elapsed_ms(const std::chrono::high_resolution_clock::time_point& start,
                  const std::chrono::high_resolution_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string format_ms(double ms) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(ms >= 100.0 ? 1 : 3) << ms << " ms";
    return ss.str();
}

struct TimeAttentionChunkContextGuard {
    explicit TimeAttentionChunkContextGuard(int chunk_index) {
        mbr_tile::set_time_attention_context_chunk(chunk_index);
    }

    ~TimeAttentionChunkContextGuard() {
        mbr_tile::set_time_attention_context_chunk(-1);
    }
};

int default_num_overlap(const JsonValue& config) {
    return std::max(1, config.get_int("num_overlap", 1));
}

int resolve_step(int chunk_size, float overlap_arg, int default_overlap) {
    if (overlap_arg > 0.0f && overlap_arg < 1.0f) {
        int step = (int)std::lround(chunk_size * (1.0f - overlap_arg));
        return std::max(1, step);
    }

    int num_overlap = default_overlap;
    if (overlap_arg >= 1.0f) {
        num_overlap = std::max(1, (int)std::lround(overlap_arg));
    }
    return std::max(1, chunk_size / std::max(1, num_overlap));
}

void require_mel_band_roformer_config(const JsonValue& config) {
    if (config.has("model_type")) {
        std::string model_type = config.get_string("model_type", "");
        if (model_type != "mel_band_roformer" && model_type != "MelBandRoformer") {
            throw std::runtime_error("Only MelBandRoformer .csm files are supported");
        }
    }

    if (!config.has("num_bands") || !config.has("stft_n_fft")) {
        throw std::runtime_error("Only MelBandRoformer .csm files are supported");
    }
}

Tensor process_chunked(const Tensor& audio, int chunk_size, int step, int chunk_batch_size,
                       std::function<Tensor(const Tensor&)> model_forward,
                       LogCallback logger) {
    Tensor mix = audio;
    int64_t length_init = audio.size(-1);

    int fade_size = chunk_size / 10;
    int border = chunk_size - step;
    if (length_init > 2LL * border && border > 0) {
        mix = mix.pad_reflect({border, border});
    }

    int64_t total_samples = mix.size(-1);
    std::vector<int64_t> chunk_starts;
    chunk_starts.reserve((size_t)((total_samples + step - 1) / step));
    for (int64_t start = 0; start < total_samples; start += step) {
        chunk_starts.push_back(start);
    }

    int64_t chunk_count = (int64_t)chunk_starts.size();
    int effective_batch = std::max(1, (int)std::min<int64_t>((int64_t)chunk_batch_size, chunk_count));
    bool single_chunk_mode = (effective_batch == 1);

    auto prepare_input_chunk = [&](int64_t start, int64_t end) {
        Tensor chunk = mix.slice(-1, start, end);
        int64_t actual_len = end - start;
        if (actual_len < chunk_size) {
            if (actual_len > chunk_size / 2) {
                chunk = chunk.pad_reflect({0, chunk_size - actual_len});
            } else {
                chunk = chunk.pad({0, chunk_size - actual_len}, 0.0f);
            }
        }
        return chunk;
    };

    Tensor first_batch_out;
    if (single_chunk_mode) {
        int64_t start = chunk_starts.front();
        int64_t end = std::min(start + chunk_size, total_samples);
        Tensor first_chunk = prepare_input_chunk(start, end);
        TimeAttentionChunkContextGuard guard(0);
        first_batch_out = model_forward(first_chunk);
    } else {
        std::vector<Tensor> first_group_inputs;
        int first_group = std::min<int64_t>(effective_batch, chunk_count);
        first_group_inputs.reserve((size_t)first_group);
        for (int i = 0; i < first_group; ++i) {
            int64_t start = chunk_starts[(size_t)i];
            int64_t end = std::min(start + chunk_size, total_samples);
            first_group_inputs.push_back(prepare_input_chunk(start, end));
        }

        Tensor first_batch_in = Tensor::cat(first_group_inputs, 0);
        TimeAttentionChunkContextGuard guard(0);
        first_batch_out = model_forward(first_batch_in);
    }

    std::vector<int64_t> out_shape = first_batch_out.shape();
    out_shape[0] = 1;
    out_shape.back() = total_samples;

    std::vector<float> fade(chunk_size, 1.0f);
    if (fade_size > 0) {
        if (fade_size == 1) {
            fade.front() = 0.0f;
            fade.back() = 0.0f;
        } else {
            for (int i = 0; i < fade_size; ++i) {
                float alpha = (float)i / (float)(fade_size - 1);
                fade[i] = alpha;
                fade[chunk_size - fade_size + i] = 1.0f - alpha;
            }
        }
    }
    Tensor fade_tensor = Tensor::from_cpu_f32(fade.data(), {(int64_t)chunk_size});

    Tensor output = Tensor::zeros(out_shape);
    Tensor weight_sum = Tensor::zeros({total_samples});
    int64_t num_channels = output.numel() / total_samples;
    output = output.reshape({num_channels, total_samples});

    int64_t chunk_index = 0;
    if (logger) {
        logger("[chunk] start");
        logger("[chunk] size: " + std::to_string(chunk_size));
        logger("[chunk] step: " + std::to_string(step));
        logger("[chunk] count: " + std::to_string(chunk_count));
        logger("[chunk] batch size: " + std::to_string(effective_batch));
    }

    for (int64_t group_start = 0; group_start < chunk_count; group_start += effective_batch) {
        int group_size = (int)std::min<int64_t>(effective_batch, chunk_count - group_start);
        Tensor batch_out;
        if (group_start == 0) {
            batch_out = first_batch_out;
        } else if (single_chunk_mode) {
            int64_t start = chunk_starts[(size_t)group_start];
            int64_t end = std::min(start + chunk_size, total_samples);
            Tensor chunk_in = prepare_input_chunk(start, end);
            TimeAttentionChunkContextGuard guard((int)group_start);
            batch_out = model_forward(chunk_in);
        } else {
            std::vector<Tensor> batch_inputs;
            batch_inputs.reserve((size_t)group_size);
            for (int local = 0; local < group_size; ++local) {
                int64_t start = chunk_starts[(size_t)(group_start + local)];
                int64_t end = std::min(start + chunk_size, total_samples);
                batch_inputs.push_back(prepare_input_chunk(start, end));
            }
            Tensor batch_in = Tensor::cat(batch_inputs, 0);
            TimeAttentionChunkContextGuard guard((int)group_start);
            batch_out = model_forward(batch_in);
        }

        for (int local = 0; local < group_size; ++local) {
            int64_t start = chunk_starts[(size_t)(group_start + local)];
            int64_t end = std::min(start + chunk_size, total_samples);
            int64_t actual_len = end - start;
            bool last_chunk = (end >= total_samples);
            chunk_index++;

            Tensor chunk_out = single_chunk_mode ? batch_out : batch_out.slice(0, local, local + 1).contiguous();
            if (chunk_out.size(-1) > actual_len) {
                chunk_out = chunk_out.slice(-1, 0, actual_len);
            }
            chunk_out = chunk_out.reshape({num_channels, actual_len}).contiguous();

            Tensor window = fade_tensor;
            if (fade_size > 0 && (start == 0 || last_chunk)) {
                std::vector<float> window_cpu = fade;
                if (start == 0) {
                    std::fill(window_cpu.begin(), window_cpu.begin() + fade_size, 1.0f);
                }
                if (last_chunk) {
                    std::fill(window_cpu.end() - fade_size, window_cpu.end(), 1.0f);
                }
                window = Tensor::from_cpu_f32(window_cpu.data(), {(int64_t)chunk_size});
            }

            Tensor fade_crop = (actual_len < chunk_size) ? window.slice(0, 0, actual_len).contiguous() : window;
            chunk_tile::accumulate_chunk(output, weight_sum, chunk_out, fade_crop, start);

            if (logger && (chunk_index == 1 || chunk_index == chunk_count || (chunk_index % 4) == 0)) {
                logger("[chunk] completed " + std::to_string(chunk_index) + "/" + std::to_string(chunk_count));
            }
        }
    }

    if (logger) {
        logger("[chunk] merge complete");
    }

    chunk_tile::normalize_by_weights(output, weight_sum);
    output = output.reshape(out_shape);
    if (length_init > 2LL * border && border > 0) {
        output = output.slice(output.ndim() - 1, border, border + length_init);
    }

    return output;
}

}  // namespace

Tensor LoadedModel::forward(const Tensor& input) {
    return model.forward(input);
}

std::vector<std::string> collect_stem_names(const LoadedModel& model) {
    std::vector<std::string> names;
    const JsonValue& config = model.weights.config();
    if (config.has("instruments") && config["instruments"].is_array()) {
        const JsonValue& instruments = config["instruments"];
        for (size_t i = 0; i < instruments.size(); ++i) {
            names.push_back(instruments[i].as_string());
        }
    }
    while ((int)names.size() < model.num_sources) {
        names.push_back("stem_" + std::to_string(names.size()));
    }
    if ((int)names.size() > model.num_sources) {
        names.resize(model.num_sources);
    }
    return names;
}

LoadedModel load_model(const std::string& model_path, int device,
                       bool quantize_fp16, bool quantize_bf16,
                       LogCallback logger) {
    using Clock = std::chrono::high_resolution_clock;
    auto total_start = Clock::now();
    cudaSetDevice(device);

    LoadedModel loaded;
    loaded.model_path = model_path;
    loaded.quantize_fp16 = quantize_fp16;
    loaded.quantize_bf16 = quantize_bf16;
    g_quantize_fp16 = quantize_fp16;
    g_quantize_bf16 = quantize_bf16;

    auto weights_start = Clock::now();
    loaded.weights = ModelWeights::load(model_path);
    if (logger) {
        logger("[time] weights load: " + format_ms(elapsed_ms(weights_start, Clock::now())));
    }

    require_mel_band_roformer_config(loaded.weights.config());

    if (quantize_fp16) {
        auto fp16_start = Clock::now();
        loaded.weights.convert_linear_weights_to_fp16();
        if (logger) {
            logger("[time] FP16 prepare: " + format_ms(elapsed_ms(fp16_start, Clock::now())));
        }
    } else if (quantize_bf16) {
        auto bf16_start = Clock::now();
        loaded.weights.convert_linear_weights_to_bf16();
        if (logger) {
            logger("[time] BF16 prepare: " + format_ms(elapsed_ms(bf16_start, Clock::now())));
        }
    }

    auto init_start = Clock::now();
    loaded.model.load(loaded.weights);
    loaded.num_sources = loaded.model.config().num_stems;
    loaded.sample_rate = loaded.model.config().sample_rate;
    loaded.chunk_size = loaded.weights.config().get_int("chunk_size", loaded.model.config().stft_n_fft * 256);
    loaded.num_overlap = default_num_overlap(loaded.weights.config());
    loaded.chunk_batch_size = 1;
    cudaDeviceSynchronize();
    if (logger) {
        logger("[time] model init: " + format_ms(elapsed_ms(init_start, Clock::now())));
    }

    loaded.stem_names = collect_stem_names(loaded);
    if (logger) {
        logger("[time] model load total: " + format_ms(elapsed_ms(total_start, Clock::now())));
    }
    return loaded;
}

InferenceResult run_inference(LoadedModel& model, const std::string& input_path, float overlap,
                              LogCallback logger) {
    using Clock = std::chrono::high_resolution_clock;
    auto load_start = Clock::now();
    AudioData audio = load_wav(input_path);
    if (logger) {
        logger("[time] WAV load: " + format_ms(elapsed_ms(load_start, Clock::now())));
        logger("[audio] sample rate: " + std::to_string(audio.sample_rate));
        logger("[audio] channels: " + std::to_string(audio.channels));
        logger("[audio] samples: " + std::to_string(audio.num_samples));
    }

    Tensor input = audio.samples.unsqueeze(0);
    cudaDeviceSynchronize();
    auto t0 = Clock::now();

    Tensor output;
    if (model.chunk_size > 0 && audio.num_samples > model.chunk_size) {
        int step = resolve_step(model.chunk_size, overlap, model.num_overlap);
        if (logger) {
            logger("[infer] chunked");
        }
        output = process_chunked(input, model.chunk_size, step, model.chunk_batch_size,
                                 [&](const Tensor& x) { return model.forward(x); },
                                 logger);
    } else {
        if (logger) {
            logger("[infer] single chunk");
        }
        output = model.forward(input);
        cudaDeviceSynchronize();
    }

    auto t1 = Clock::now();
    InferenceResult result;
    result.audio = std::move(audio);
    result.output = std::move(output);
    result.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.rtf = (double)result.audio.num_samples / model.sample_rate / (result.infer_ms / 1000.0);
    if (logger) {
        logger("[infer] time: " + format_ms(result.infer_ms));
        logger("[infer] RTF: " + std::to_string(result.rtf));
    }
    return result;
}

Tensor extract_stem_audio(const Tensor& output, int stem) {
    if (output.ndim() != 4) {
        return output.squeeze(0);
    }
    int sources = (int)output.size(1);
    if (stem < 0 || stem >= sources) {
        throw std::runtime_error("Stem index out of range");
    }
    return output.slice(1, stem, stem + 1).squeeze(0).squeeze(0);
}

std::string stem_label(const LoadedModel& model, int stem) {
    if (stem >= 0 && stem < (int)model.stem_names.size()) {
        return model.stem_names[stem];
    }
    return "stem_" + std::to_string(stem);
}

std::vector<fs::path> save_outputs(const LoadedModel& model, const Tensor& output,
                                   const fs::path& out_path, int stem) {
    std::vector<fs::path> saved;
    bool multi_source = (output.ndim() == 4);

    if (multi_source) {
        int sources = (int)output.size(1);
        if (stem >= 0 && stem < sources) {
            Tensor stem_audio = extract_stem_audio(output, stem);
            fs::path file_path = out_path;
            if (out_path.extension() != ".wav") {
                fs::create_directories(out_path);
                file_path = out_path / (stem_label(model, stem) + ".wav");
            }
            save_wav(file_path.string(), stem_audio, model.sample_rate);
            saved.push_back(file_path);
            return saved;
        }

        if (stem != -1) {
            throw std::runtime_error("Stem index out of range");
        }

        fs::create_directories(out_path);
        for (int s = 0; s < sources; ++s) {
            Tensor stem_audio = extract_stem_audio(output, s);
            fs::path file_path = out_path / (stem_label(model, s) + ".wav");
            save_wav(file_path.string(), stem_audio, model.sample_rate);
            saved.push_back(file_path);
        }
        return saved;
    }

    Tensor stem_audio = output.squeeze(0);
    fs::path file_path = out_path;
    if (out_path.extension() != ".wav") {
        fs::create_directories(out_path);
        file_path = out_path / "output.wav";
    }
    save_wav(file_path.string(), stem_audio, model.sample_rate);
    saved.push_back(file_path);
    return saved;
}

}  // namespace cudasep::app
