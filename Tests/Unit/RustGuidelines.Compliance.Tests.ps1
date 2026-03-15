<#
.SYNOPSIS
    Post-compilation compliance tests for Microsoft Pragmatic Rust Guidelines.

.DESCRIPTION
    Validates that the PC_AI Rust codebase conforms to the M-* guidelines from
    ~/.agents/rust-guidelines.txt. These tests run against source files and
    Cargo.toml configuration — they do not require compilation.
#>

BeforeAll {
    $script:RepoRoot = Join-Path $PSScriptRoot '..\..'
    $script:NativeRoot = Join-Path $script:RepoRoot 'Native\pcai_core'
    $script:DeployRoot = Join-Path $script:RepoRoot 'Deploy'

    function Get-RustSourceFiles {
        param(
            [string]$Root,
            [switch]$ExcludeTests,
            [switch]$ExcludeVendor
        )
        $excludePatterns = @()
        if ($ExcludeTests) { $excludePatterns += 'tests' }
        if ($ExcludeVendor) { $excludePatterns += 'vendor'; $excludePatterns += 'third_party'; $excludePatterns += 'target'; $excludePatterns += 'target-ffi'; $excludePatterns += 'target-ffi-nosccache' }

        Get-ChildItem -Path $Root -Recurse -Filter '*.rs' -File | Where-Object {
            $path = $_.FullName
            $excluded = $false
            foreach ($p in $excludePatterns) {
                if ($path -match [regex]::Escape([IO.Path]::DirectorySeparatorChar) + $p + [regex]::Escape([IO.Path]::DirectorySeparatorChar)) {
                    $excluded = $true
                    break
                }
                if ($path -match '[\\/]' + $p + '[\\/]') {
                    $excluded = $true
                    break
                }
            }
            -not $excluded
        }
    }
}

# ─── M-STATIC-VERIFICATION: Cargo.toml has [lints.clippy] ────────────────────

Describe "M-STATIC-VERIFICATION: Clippy lints configured" -Tag 'Unit', 'RustGuidelines', 'Fast' {
    It "Native workspace Cargo.toml should have [workspace.lints.clippy]" {
        $cargoToml = Get-Content -Path (Join-Path $script:NativeRoot 'Cargo.toml') -Raw
        $cargoToml | Should -Match '\[workspace\.lints\.clippy\]'
    }

    It "Should enable pedantic clippy lints" {
        $cargoToml = Get-Content -Path (Join-Path $script:NativeRoot 'Cargo.toml') -Raw
        $cargoToml | Should -Match 'pedantic\s*=\s*\{.*level.*=.*"warn"'
    }

    It "Should enable undocumented_unsafe_blocks lint" {
        $cargoToml = Get-Content -Path (Join-Path $script:NativeRoot 'Cargo.toml') -Raw
        $cargoToml | Should -Match 'undocumented_unsafe_blocks\s*=\s*"warn"'
    }
}

# ─── M-LINT-OVERRIDE-EXPECT: #[expect] over #[allow] ─────────────────────────

Describe "M-LINT-OVERRIDE-EXPECT: Prefer #[expect] over #[allow]" -Tag 'Unit', 'RustGuidelines', 'Fast' {
    It "Should have few #[allow(...)] without reason in Native src/" {
        $files = Get-RustSourceFiles -Root $script:NativeRoot -ExcludeTests -ExcludeVendor
        $allows = @($files | Select-String -Pattern '#\[allow\(' |
            Where-Object { $_.Line -notmatch 'reason\s*=' -and $_.Line -notmatch '^\s*//' })
        # Allow up to 25 — some are in generated code or attribute macros
        $allows.Count | Should -BeLessOrEqual 25 -Because "M-LINT-OVERRIDE-EXPECT: use #[expect] with reason instead of #[allow]"
    }
}

# ─── M-AVOID-STATICS: No static mut ──────────────────────────────────────────

Describe "M-AVOID-STATICS: No static mut in production code" -Tag 'Unit', 'RustGuidelines', 'Fast' {
    It "Should have zero 'static mut' declarations in Native src/" {
        $files = Get-RustSourceFiles -Root $script:NativeRoot -ExcludeTests -ExcludeVendor
        $staticMut = @($files | Select-String -Pattern 'static\s+mut\s+' |
            Where-Object { $_.Line -notmatch '^\s*//' })
        $staticMut.Count | Should -Be 0 -Because "M-AVOID-STATICS: use atomic types or interior mutability instead"
    }

    It "Should have zero 'static mut' declarations in Deploy src/" {
        $deployFiles = Get-RustSourceFiles -Root $script:DeployRoot -ExcludeTests -ExcludeVendor
        if ($deployFiles.Count -eq 0) { Set-ItResult -Skipped -Because "No Deploy Rust files found" ; return }
        $staticMut = @($deployFiles | Select-String -Pattern 'static\s+mut\s+' |
            Where-Object { $_.Line -notmatch '^\s*//' })
        $staticMut.Count | Should -Be 0
    }
}

# ─── M-NO-GLOB-REEXPORTS: No pub use ::* ─────────────────────────────────────

Describe "M-NO-GLOB-REEXPORTS: No glob re-exports" -Tag 'Unit', 'RustGuidelines', 'Fast' {
    It "Should have zero 'pub use ...::*' in Native src/" {
        $files = Get-RustSourceFiles -Root $script:NativeRoot -ExcludeTests -ExcludeVendor
        $globs = @($files | Select-String -Pattern 'pub\s+use\s+.*::\*\s*;' |
            Where-Object { $_.Line -notmatch '^\s*//' })
        $globs.Count | Should -Be 0 -Because "M-NO-GLOB-REEXPORTS: list re-exports explicitly"
    }
}

# ─── M-MIMALLOC-APPS: Binary crates use mimalloc ─────────────────────────────

Describe "M-MIMALLOC-APPS: Binary crates use mimalloc" -Tag 'Unit', 'RustGuidelines', 'Fast' {
    BeforeAll {
        $script:BinaryCrates = @(
            @{ Name = 'pcai_media_server'; Main = Join-Path $script:NativeRoot 'pcai_media_server\src\main.rs' },
            @{ Name = 'pcai_ollama_rs'; Main = Join-Path $script:NativeRoot 'pcai_ollama_rs\src\main.rs' },
            @{ Name = 'pcai_perf_cli'; Main = Join-Path $script:NativeRoot 'pcai_perf_cli\src\main.rs' }
        )
    }

    It "Should have #[global_allocator] with mimalloc in <Name>" -ForEach @(
        @{ Name = 'pcai_media_server'; Main = "$PSScriptRoot\..\..\Native\pcai_core\pcai_media_server\src\main.rs" },
        @{ Name = 'pcai_ollama_rs'; Main = "$PSScriptRoot\..\..\Native\pcai_core\pcai_ollama_rs\src\main.rs" },
        @{ Name = 'pcai_perf_cli'; Main = "$PSScriptRoot\..\..\Native\pcai_core\pcai_perf_cli\src\main.rs" }
    ) {
        if (-not (Test-Path $Main)) { Set-ItResult -Skipped -Because "$Name main.rs not found"; return }
        $content = Get-Content -Path $Main -Raw
        $content | Should -Match 'global_allocator' -Because "M-MIMALLOC-APPS: $Name should use mimalloc"
    }
}

# ─── Tech Debt: No expect("TODO") markers ────────────────────────────────────

Describe "Tech Debt: No TODO expect markers" -Tag 'Unit', 'RustGuidelines', 'Fast' {
    It "Should have zero expect('TODO: Verify unwrap') in all Rust files" {
        $allRs = Get-ChildItem -Path $script:RepoRoot -Recurse -Filter '*.rs' -File |
            Where-Object { $_.FullName -notmatch '[\\/](target|vendor|third_party)[\\/]' }
        $todoExpects = @($allRs | Select-String -Pattern 'expect\("TODO: Verify unwrap"\)')
        $todoExpects.Count | Should -Be 0 -Because "all TODO expects should have descriptive messages"
    }
}

# ─── M-MODULE-DOCS: Library crates have module documentation ─────────────────

Describe "M-MODULE-DOCS: Library crates have module docs" -Tag 'Unit', 'RustGuidelines', 'Fast' {
    It "pcai_inference lib.rs should have //! module documentation" {
        $libRs = Join-Path $script:NativeRoot 'pcai_inference\src\lib.rs'
        if (-not (Test-Path $libRs)) { Set-ItResult -Skipped; return }
        $content = Get-Content -Path $libRs -Raw
        $content | Should -Match '^//!' -Because "M-MODULE-DOCS: library crates need //! documentation"
    }

    It "pcai_core_lib lib.rs should have //! module documentation" {
        $libRs = Join-Path $script:NativeRoot 'pcai_core_lib\src\lib.rs'
        if (-not (Test-Path $libRs)) { Set-ItResult -Skipped; return }
        $content = Get-Content -Path $libRs -Raw
        $content | Should -Match '^//!' -Because "M-MODULE-DOCS: library crates need //! documentation"
    }
}

# ─── M-FEATURES-ADDITIVE: Feature flags don't subtract ───────────────────────

Describe "M-FEATURES-ADDITIVE: Features are additive" -Tag 'Unit', 'RustGuidelines', 'Fast' {
    It "pcai_inference features should all be additive" {
        $cargoToml = Get-Content -Path (Join-Path $script:NativeRoot 'pcai_inference\Cargo.toml') -Raw
        # Check for cfg(not(feature = ...)) patterns that would indicate subtractive features
        $srcFiles = Get-RustSourceFiles -Root (Join-Path $script:NativeRoot 'pcai_inference\src') -ExcludeTests -ExcludeVendor
        $subtractive = @($srcFiles | Select-String -Pattern '#\[cfg\(not\(feature\s*=')
        # Some cfg(not(feature)) is normal for default behavior — just flag excessive use
        $subtractive.Count | Should -BeLessOrEqual 10 -Because "M-FEATURES-ADDITIVE: features should add, not subtract"
    }
}

# ─── M-STRONG-TYPES: Path parameters use Path types ──────────────────────────

Describe "M-STRONG-TYPES: Path parameters use proper types" -Tag 'Unit', 'RustGuidelines', 'Advisory' {
    It "Should not have excessive String path parameters in production code" {
        $files = Get-RustSourceFiles -Root $script:NativeRoot -ExcludeTests -ExcludeVendor
        # Look for fn signatures with path/file/dir parameters typed as &str or String
        $stringPaths = @($files | Select-String -Pattern 'fn\s+\w+.*\b(path|file_path|dir)\s*:\s*(&str|String)' |
            Where-Object { $_.Line -notmatch '^\s*//' -and $_.Line -notmatch 'ffi' })
        # Advisory: allow some but flag if excessive
        $stringPaths.Count | Should -BeLessOrEqual 20 -Because "M-STRONG-TYPES: prefer Path/PathBuf/impl AsRef<Path> for path parameters"
    }
}

# ─── CI/CD: Guidelines workflow exists ────────────────────────────────────────

Describe "CI/CD: Rust guidelines workflow configured" -Tag 'Unit', 'RustGuidelines', 'Fast' {
    It "Should have rust-guidelines.yml workflow" {
        Test-Path (Join-Path $script:RepoRoot '.github\workflows\rust-guidelines.yml') | Should -Be $true
    }

    It "Workflow should check format and clippy" {
        $workflow = Get-Content -Path (Join-Path $script:RepoRoot '.github\workflows\rust-guidelines.yml') -Raw
        $workflow | Should -Match 'cargo fmt.*--check'
        $workflow | Should -Match 'cargo clippy'
    }
}

# ─── Module Export Completeness ───────────────────────────────────────────────

Describe "PC-AI Framework: All modules load correctly" -Tag 'Unit', 'RustGuidelines', 'Fast' {
    It "PC-AI.Drivers should import without errors" {
        $manifest = Join-Path $script:RepoRoot 'Modules\PC-AI.Drivers\PC-AI.Drivers.psd1'
        { Import-Module $manifest -Force -ErrorAction Stop } | Should -Not -Throw
    }

    It "PC-AI.Acceleration should import without errors" {
        $manifest = Join-Path $script:RepoRoot 'Modules\PC-AI.Acceleration\PC-AI.Acceleration.psd1'
        { Import-Module $manifest -Force -ErrorAction Stop } | Should -Not -Throw
    }

    It "PC-AI.Common should import without errors" {
        $manifest = Join-Path $script:RepoRoot 'Modules\PC-AI.Common\PC-AI.Common.psm1'
        { Import-Module $manifest -Force -ErrorAction Stop } | Should -Not -Throw
    }

    It "PC-AI.CLI should import without errors" {
        $manifest = Join-Path $script:RepoRoot 'Modules\PC-AI.CLI\PC-AI.CLI.psm1'
        { Import-Module $manifest -Force -ErrorAction Stop } | Should -Not -Throw
    }
}
