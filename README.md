# CUDA-MSST-Infer

Minimal C++/CUDA CLI for running MelBandRoformer `.csm` inference on WAV files.

This branch intentionally keeps only the runtime path needed for experiments:

- CLI executable: `cudasep_infer`
- input audio: WAV only
- model: MelBandRoformer `.csm` only
- output: WAV stems
- no server mode
- no Python bindings
- no other model architectures
- no non-WAV decoding fallback
- no cuDNN dependency

## Build

Requirements:

- NVIDIA GPU
- CUDA Toolkit 12.0+
- CMake 3.20+
- Ninja
- C++17 compiler

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86

ninja -C build -j$(nproc)
```

Adjust `CMAKE_CUDA_ARCHITECTURES` for the target GPU if needed.

## Usage

Run inference and save all stems:

```bash
./build/cudasep_infer \
  --model path/to/model.csm \
  --input input.wav \
  --output outputs/run \
  --fp16 \
  --chunk-batch-size 1 \
  --stem -1
```

Save one stem:

```bash
./build/cudasep_infer \
  --model path/to/model.csm \
  --input input.wav \
  --output vocals.wav \
  --fp16 \
  --stem 0
```

## Runtime Files

```text
src/main.cpp                    CLI argument parsing and entry point
src/inference_app.*             MelBandRoformer load/infer/save orchestration
src/audio_io.*                  WAV read/write
src/model_mel_band_roformer.*   MelBandRoformer runtime
src/ops_*.cu                    CUDA ops used by MelBandRoformer
src/tensor.*                    GPU tensor helper
src/weights.*                   .csm weight/config loader
```

## Verification Example

```bash
./build/cudasep_infer \
  --model mel-band-roformer-deux/becruily_deux.csm \
  --input test_clean.wav \
  --output outputs/test_clean_min \
  --fp16 \
  --chunk-batch-size 1 \
  --stem -1
```

Expected outputs:

```text
outputs/test_clean_min/Vocals.wav
outputs/test_clean_min/Instrumental.wav
```

## License

AGPL-3.0
