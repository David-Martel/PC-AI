#3. Setup Script (setup_cuda.sh)
#This script ensures your environment is correctly configured to build Rust with CUDA support (candle-core features).

Bash
#!/bin/bash
set -e

# setup_cuda.sh
# Usage: ./setup_cuda.sh

echo "[Setup] Checking NVIDIA environment..."

if ! command -v nvcc &>/dev/null; then
    echo "[Error] 'nvcc' not found. Please install the CUDA Toolkit."
    echo "Ubuntu: sudo apt install nvidia-cuda-toolkit"
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
echo "[Info] Found CUDA version: $NVCC_VERSION"

# Set environment variables required for candle-core build
export CUDA_ROOT="/usr/local/cuda"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CANDLE_FLASH_ATTN_BUILD_DIR="./target/flash_attn" # Optional optimization

echo "[Setup] Environment variables set."
echo "        CUDA_ROOT=$CUDA_ROOT"

echo "[Build] Cleaning previous builds..."
cargo clean

echo "[Build] compiling with CUDA features..."
# We explicitly enable the 'cuda' feature for candle-core
#
# Make sure your Cargo.toml is set up to include candle-core with the 'cuda' feature
export CUDA_COMPUTE_CAP=89 # Set to your GPU arch (e.g., 89 for RTX 4090, 86 for 3090)

cargo build --release --features "cuda"

echo "[Success] Build complete. Run with: ./target/release/janus-rust"
