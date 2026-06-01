#include "inference_app.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Args {
    std::string model_path;
    std::string input_path;
    std::string output_path = "output";
    int stem = 0;
    float overlap = -1.0f;
    int device = 0;
    bool help = false;
    bool quantize_fp16 = false;
    int chunk_batch_size = 0;
};

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--model" || a == "-m") && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if ((a == "--input" || a == "-i") && i + 1 < argc) {
            args.input_path = argv[++i];
        } else if ((a == "--output" || a == "-o") && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if ((a == "--stem" || a == "-s") && i + 1 < argc) {
            args.stem = std::stoi(argv[++i]);
        } else if (a == "--overlap" && i + 1 < argc) {
            args.overlap = std::stof(argv[++i]);
        } else if ((a == "--device" || a == "-d") && i + 1 < argc) {
            args.device = std::stoi(argv[++i]);
        } else if (a == "--fp16") {
            args.quantize_fp16 = true;
        } else if (a == "--chunk-batch-size" && i + 1 < argc) {
            args.chunk_batch_size = std::stoi(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            args.help = true;
        }
    }
    return args;
}

static void print_usage(const char* progname) {
    std::cout << "CUDA MelBandRoformer WAV CLI\n\n"
              << "Usage:\n"
              << "  " << progname << " --model <model.csm> --input <audio.wav> [options]\n\n"
              << "Options:\n"
              << "  --output, -o <path>       Output directory or WAV file path (default: output)\n"
              << "  --stem, -s <int>          Stem index to save (default: 0, -1 saves all)\n"
              << "  --fp16                    Use FP16 linear weights\n"
              << "  --chunk-batch-size <n>    Override chunk inference batch size\n"
              << "  --overlap <float>         (0,1)=overlap ratio, >=1=num_overlap\n"
              << "  --device, -d <int>        CUDA device ID (default: 0)\n"
              << "  --help, -h                Show this help message\n";
}

static void print_gpu_info(int device) {
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "GPU: " << prop.name << " (SM " << prop.major << '.' << prop.minor
              << ", " << (prop.totalGlobalMem / (1024 * 1024)) << " MB)" << std::endl;
}

static int run_cli(const Args& args) {
    if (args.model_path.empty()) {
        std::cerr << "Error: --model is required" << std::endl;
        return 1;
    }

    std::cout << "Loading model: " << args.model_path << std::endl;
    cudasep::app::LogCallback cli_logger = [](const std::string& line) {
        std::cout << "  " << line << std::endl;
    };

    cudasep::app::LoadedModel model =
        cudasep::app::load_model(args.model_path, args.device, args.quantize_fp16, cli_logger);
    if (args.chunk_batch_size > 0) {
        model.chunk_batch_size = args.chunk_batch_size;
        cli_logger("[config] chunk batch size: " + std::to_string(model.chunk_batch_size));
    }

    std::cout << "  Model type: MelBandRoformer" << std::endl;
    std::cout << "  Sources: " << model.num_sources
              << ", Sample rate: " << model.sample_rate
              << ", Chunk size: " << model.chunk_size
              << ", Num overlap: " << model.num_overlap << std::endl;

    if (args.input_path.empty()) {
        std::cerr << "Error: --input is required" << std::endl;
        return 1;
    }

    std::cout << "\nLoading WAV: " << args.input_path << std::endl;
    cudasep::app::InferenceResult result =
        cudasep::app::run_inference(model, args.input_path, args.overlap, cli_logger);

    std::cout << "  " << result.audio.channels << "ch, " << result.audio.sample_rate << " Hz, "
              << result.audio.num_samples << " samples ("
              << std::fixed << std::setprecision(1)
              << (double)result.audio.num_samples / result.audio.sample_rate << "s)" << std::endl;
    if (result.audio.sample_rate != model.sample_rate) {
        std::cerr << "Warning: WAV sample rate (" << result.audio.sample_rate
                  << ") != model sample rate (" << model.sample_rate << ")." << std::endl;
    }

    std::cout << "\nRunning inference..." << std::endl;
    std::cout << "  Inference time: " << std::fixed << std::setprecision(1)
              << result.infer_ms << " ms (RTF: " << std::setprecision(2) << result.rtf << "x)"
              << std::endl;

    std::vector<fs::path> saved =
        cudasep::app::save_outputs(model, result.output, fs::path(args.output_path), args.stem);
    for (const auto& path : saved) {
        std::cout << "Saving: " << path.string() << std::endl;
    }
    std::cout << "Done." << std::endl;
    return 0;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }

    try {
        print_gpu_info(args.device);
        return run_cli(args);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA: " << cudaGetErrorString(err) << std::endl;
        }
        return 1;
    }
}
