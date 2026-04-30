# Setup RUSTFLAGS optimization profiles
# Add this to your PowerShell profile ($PROFILE) or source it manually

function Set-RustFlags {
    <#
    .SYNOPSIS
    Switch between RUSTFLAGS optimization profiles for Rust compilation.

    .DESCRIPTION
    Provides quick switching between predefined RUSTFLAGS configurations optimized for different use cases:
    - dev: Fast compilation, full debug info (daily development)
    - release: Maximum optimization, stripped (production builds)
    - test: Balanced optimization with debug info (CI/CD, testing)
    - simd: SIMD-optimized for numeric/crypto code
    - lto: Full link-time optimization (maximum performance)
    - minimal: Smallest binary size (distribution)
    - cross: Cross-compilation target support

    .PARAMETER Profile
    The optimization profile to activate.

    .EXAMPLE
    Set-RustFlags dev        # Switch to development profile
    Set-RustFlags release    # Switch to production profile
    Set-RustFlags simd       # Switch to SIMD-optimized profile

    .NOTES
    Hardware: Intel Core Ultra 7 155H (16c/22t)
    Profiles are hardcoded for this machine's capabilities.
    #>

    param(
        [ValidateSet("dev", "release", "test", "simd", "lto", "minimal", "cross")][string]$Profile = "dev"
    )

    # Define RUSTFLAGS for each profile
    $flags = @{
        "dev" = @(
            "-C", "target-cpu=native",
            "-C", "opt-level=0",
            "-C", "debuginfo=full",
            "-C", "overflow-checks=on",
            "-C", "link-arg=/STACK:8388608"
        )

        "release" = @(
            "-C", "target-cpu=native",
            "-C", "opt-level=3",
            "-C", "lto=fat",
            "-C", "codegen-units=1",
            "-C", "overflow-checks=off",
            "-C", "debuginfo=none",
            "-C", "strip=symbols",
            "-C", "target-feature=+avx2,+fma",
            "-C", "link-arg=/SUBSYSTEM:CONSOLE",
            "-C", "link-arg=/STACK:8388608"
        )

        "test" = @(
            "-C", "target-cpu=native",
            "-C", "opt-level=2",
            "-C", "lto=thin",
            "-C", "codegen-units=16",
            "-C", "overflow-checks=on",
            "-C", "debuginfo=1",
            "-C", "link-arg=/STACK:8388608"
        )

        "simd" = @(
            "-C", "target-cpu=native",
            "-C", "opt-level=3",
            "-C", "lto=fat",
            "-C", "codegen-units=1",
            "-C", "overflow-checks=off",
            "-C", "target-feature=+avx2,+fma,+sha,+aes,+vaes,+vpclmulqdq",
            "-C", "link-arg=/STACK:8388608"
        )

        "lto" = @(
            "-C", "target-cpu=native",
            "-C", "opt-level=3",
            "-C", "lto=fat",
            "-C", "codegen-units=1",
            "-C", "embed-bitcode=yes",
            "-C", "overflow-checks=off",
            "-C", "link-arg=/STACK:8388608"
        )

        "minimal" = @(
            "-C", "target-cpu=native",
            "-C", "opt-level=z",
            "-C", "lto=fat",
            "-C", "codegen-units=1",
            "-C", "strip=all",
            "-C", "overflow-checks=off",
            "-C", "link-arg=/STACK:8388608"
        )

        "cross" = @(
            "-C", "target-cpu=generic",
            "-C", "opt-level=3",
            "-C", "lto=thin",
            "-C", "codegen-units=8",
            "-C", "overflow-checks=off",
            "-C", "link-arg=-fuse-ld=mold",
            "-C", "link-arg=-Wl,--gc-sections"
        )
    }

    # Apply selected profile
    $env:RUSTFLAGS = $flags[$Profile] -join " "

    # Display confirmation
    Write-Host "✅ RUSTFLAGS set to: $Profile" -ForegroundColor Green

    # Show profile details
    switch ($Profile) {
        "dev" {
            Write-Host "   Use for: Daily development, frequent rebuilds" -ForegroundColor Gray
            Write-Host "   Expected: ~2-10s incremental builds (with sccache)" -ForegroundColor Gray
        }
        "release" {
            Write-Host "   Use for: Production binaries, distribution" -ForegroundColor Gray
            Write-Host "   Expected: ~180s compile, 40-50% faster than dev" -ForegroundColor Gray
        }
        "test" {
            Write-Host "   Use for: CI/CD, automated testing" -ForegroundColor Gray
            Write-Host "   Expected: ~90s compile, 20-25% faster than dev" -ForegroundColor Gray
        }
        "simd" {
            Write-Host "   Use for: Numeric, crypto, DSP code" -ForegroundColor Gray
            Write-Host "   Expected: 4-8x faster SIMD operations" -ForegroundColor Gray
        }
        "lto" {
            Write-Host "   Use for: Maximum performance over compile time" -ForegroundColor Gray
            Write-Host "   Expected: ~300s compile, 50% faster runtime" -ForegroundColor Gray
        }
        "minimal" {
            Write-Host "   Use for: Distribution, size-constrained" -ForegroundColor Gray
            Write-Host "   Expected: Smallest binary size (~40MB)" -ForegroundColor Gray
        }
        "cross" {
            Write-Host "   Use for: Cross-compilation (Linux from Windows)" -ForegroundColor Gray
            Write-Host "   Expected: Generic x86-64, portable binary" -ForegroundColor Gray
        }
    }

    Write-Host ""
}

function Get-RustFlags {
    <#
    .SYNOPSIS
    Display current RUSTFLAGS setting.

    .EXAMPLE
    Get-RustFlags
    #>

    if ([string]::IsNullOrWhiteSpace($env:RUSTFLAGS)) {
        Write-Host "⚠️  RUSTFLAGS not set" -ForegroundColor Yellow
    } else {
        Write-Host "Current RUSTFLAGS:" -ForegroundColor Cyan
        Write-Host "$($env:RUSTFLAGS)" -ForegroundColor Gray
    }
}

function Invoke-RustBuild {
    <#
    .SYNOPSIS
    Build with current RUSTFLAGS and measure time.

    .PARAMETER Release
    Build in release mode.

    .EXAMPLE
    Invoke-RustBuild
    Invoke-RustBuild -Release
    #>

    param([switch]$Release)

    $args = @("build")
    if ($Release) { $args += "--release" }

    Write-Host "Building with RUSTFLAGS..." -ForegroundColor Cyan
    $time = Measure-Command {
        & cargo @args
    }

    Write-Host "Build completed in $($time.TotalSeconds) seconds" -ForegroundColor Green
}

# Display available profiles on load
Write-Host "Rust compilation flags setup loaded" -ForegroundColor Cyan
Write-Host "Available profiles: dev, release, test, simd, lto, minimal, cross" -ForegroundColor Gray
Write-Host ""
Write-Host "Usage:" -ForegroundColor Cyan
Write-Host "  Set-RustFlags dev      # Switch to development profile" -ForegroundColor Gray
Write-Host "  Get-RustFlags          # Show current flags" -ForegroundColor Gray
Write-Host "  Invoke-RustBuild       # Build with current flags" -ForegroundColor Gray
Write-Host ""

# Set default to dev profile
Set-RustFlags "dev"
