#!/bin/bash
# Cargo wrapper script for WSL - enables sccache for Rust compilations
# This script sets up sccache environment and calls cargo with proper wrapper configuration

# Set sccache as the Rust compiler wrapper
export RUSTC_WRAPPER=sccache

# Optionally centralize WSL cargo/rustup data on the Windows T: drive
# Toggle with WSL_USE_SHARED_RUST_CACHE=0 to keep WSL-local caches
if [ "${WSL_USE_SHARED_RUST_CACHE:-1}" = "1" ]; then
    export CARGO_HOME="${WSL_CARGO_HOME:-/mnt/t/RustCache/wsl-cargo-home}"
    export RUSTUP_HOME="${WSL_RUSTUP_HOME:-/mnt/t/RustCache/wsl-rustup}"
fi

# Configure sccache cache directory to use shared Windows cache
# The cache is at T:\RustCache\sccache on Windows
# Which is accessible at /mnt/t/RustCache/sccache in WSL
if [ -z "$SCCACHE_DIR" ]; then
    export SCCACHE_DIR="/mnt/t/RustCache/sccache"
fi

# Set cache compression and size if not already set
if [ -z "$SCCACHE_CACHE_COMPRESSION" ]; then
    export SCCACHE_CACHE_COMPRESSION=zstd
fi

if [ -z "$SCCACHE_CACHE_SIZE" ]; then
    export SCCACHE_CACHE_SIZE="30G"
fi

if [ -z "$SCCACHE_IDLE_TIMEOUT" ]; then
    export SCCACHE_IDLE_TIMEOUT=1800
fi

# Call the real cargo with all arguments
exec cargo "$@"
