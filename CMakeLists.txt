cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(CudaInfer LANGUAGES CXX CUDA)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures — use "-real" to embed SASS only (no PTX), keeping
# the last arch with PTX for forward-compatibility via JIT.
# Override with -DCMAKE_CUDA_ARCHITECTURES=... for local single-GPU builds.
set(CMAKE_CUDA_ARCHITECTURES "80-real;86-real;89")

# Require CUDA 12.0+
find_package(CUDAToolkit 12.0 REQUIRED)

# NVCC flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr --allow-unsupported-compiler")

# Platform-specific compiler and linker flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /DNDEBUG /utf-8")
else()
    # Strip debug symbols in Release builds (saves 50-100 MB)
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -s")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG")
endif()

# Minimal CLI-only build: WAV input + MelBandRoformer .csm inference.
set(ALL_SOURCES
    src/main.cpp
    src/inference_app.cpp
    src/audio_io.cpp
    src/tensor.cu
    src/weights.cu
    src/model_mel_band_roformer.cu
    src/ops_attention.cu
    src/ops_elementwise.cu
    src/ops_fused.cu
    src/ops_linear.cu
    src/ops_misc.cu
    src/ops_norm.cu
    src/ops_stft.cu
)

# Main executable
add_executable(cudasep_infer ${ALL_SOURCES})

target_link_libraries(cudasep_infer PRIVATE
    CUDA::cublas
    CUDA::cufft
    CUDA::cudart
)

target_include_directories(cudasep_infer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
