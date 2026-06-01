cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# CMake 3.22 does not ship CUDA20 flag mappings, but CUDA 13.3 nvcc supports
# them. Teach CMake the compile option before enabling the CUDA language.
set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "--std=c++20")
set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "--std=c++20")

project(CudaInfer LANGUAGES CXX CUDA)

# C++20 / CUDA20 required for CUDA Tile experiments.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures — use "-real" to embed SASS only (no PTX), keeping
# the last arch with PTX for forward-compatibility via JIT.
# Override with -DCMAKE_CUDA_ARCHITECTURES=... for local single-GPU builds.
set(CMAKE_CUDA_ARCHITECTURES "80-real;86-real;89")

# CMake may read stale toolkit metadata on CUDA preview installs; use the
# actual nvcc version for the CUDA Tile requirement.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.3)
    message(FATAL_ERROR "CUDA Tile requires nvcc 13.3 or newer")
endif()
find_package(CUDAToolkit REQUIRED)

# NVCC flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr --allow-unsupported-compiler --enable-tile")

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
    src/tensor_cuda_tile.cu
    src/tensor_cuda_glue.cu
    src/weights.cu
    src/cuda_context.cu
    src/chunk_cuda_tile.cu
    src/mbr_cuda_tile.cu
    src/stft_cuda_tile.cu
    src/model_mel_band_roformer.cu
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
