#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

binary="${1:-build-min/cudasep_infer}"
forbidden_source='cuBLAS|cublas|cuFFT|cufft|cuDNN|cudnn|nvcuda::wmma|wmma::|mma\.sync|asm\s*\(|__global__'
forbidden_binary='cublas|cufft|cudnn|nvcuda::wmma|wmma::|mma\.sync'

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

echo "[audit] source/CMake/bench forbidden path scan"
if rg -n "$forbidden_source" CMakeLists.txt src bench \
    -g '!bench/README.md' \
    -g '!bench/audit_cuda_tile_first.sh' \
    -g '!bench/.ipynb_checkpoints/**' \
    >"$tmp"; then
    cat "$tmp"
    echo "[audit] FAIL: forbidden source/CMake/bench pattern found" >&2
    exit 1
fi
echo "[audit] OK: no forbidden source/CMake/bench matches"

if [[ ! -x "$binary" ]]; then
    echo "[audit] FAIL: binary not found or not executable: $binary" >&2
    exit 1
fi

echo "[audit] dynamic dependency scan: $binary"
objdump -p "$binary" | rg -i 'NEEDED|cublas|cufft|cudnn' || true
if objdump -p "$binary" | rg -i 'cublas|cufft|cudnn' >"$tmp"; then
    cat "$tmp"
    echo "[audit] FAIL: forbidden CUDA library dependency found" >&2
    exit 1
fi
echo "[audit] OK: no cuBLAS/cuFFT/cuDNN dynamic dependency"

echo "[audit] dynamic symbol scan: $binary"
if nm -D "$binary" 2>/dev/null | c++filt | rg -i "$forbidden_binary" >"$tmp"; then
    cat "$tmp"
    echo "[audit] FAIL: forbidden dynamic symbol found" >&2
    exit 1
fi
echo "[audit] OK: no forbidden dynamic symbols"

echo "[audit] binary string scan: $binary"
if strings "$binary" | rg -i "$forbidden_binary" >"$tmp"; then
    cat "$tmp"
    echo "[audit] FAIL: forbidden binary string found" >&2
    exit 1
fi
echo "[audit] OK: no forbidden binary strings"

echo "[audit] PASS: CUDA Tile-first purity gate passed for $binary"
