#Requires -Version 7.0

function Disable-PcaiBuildCompilerCaches {
    <#
    .SYNOPSIS
    Prevents build initialization from re-enabling compiler caches in this process.
    #>
    [CmdletBinding()]
    param()
    $env:SCCACHE_DISABLE = '1'
    $env:CCACHE_DISABLE = '1'
    $env:RUSTC_WRAPPER = $null
    $env:CMAKE_C_COMPILER_LAUNCHER = $null
    $env:CMAKE_CXX_COMPILER_LAUNCHER = $null
}

function Get-PcaiExplicitCpuTarget {
    param([string[]]$Arguments)
    for ($index = 0; $index -lt $Arguments.Count; $index++) {
        $argument = $Arguments[$index]
        if ($argument -match '^(?:-C|--codegen=)target-cpu=(.+)$') { return $Matches[1] }
        if ($argument -in @('-C', '--codegen') -and $index + 1 -lt $Arguments.Count -and
            $Arguments[$index + 1] -match '^target-cpu=(.+)$') { return $Matches[1] }
    }
    return $null
}

function Enable-PcaiNativeBuildOptimization {
    <#
    .SYNOPSIS
    Enables explicit host CPU tuning without replacing a selected CPU target.
    .DESCRIPTION
    Changes this build process only. Resulting native-target binaries require
    qualification on their destination machine. Encoded Cargo flags take
    precedence over RUSTFLAGS and retain their original argument boundaries.
    #>
    [CmdletBinding()]
    param()

    if ($env:CARGO_ENCODED_RUSTFLAGS) {
        $arguments = [Collections.Generic.List[string]]::new()
        foreach ($argument in $env:CARGO_ENCODED_RUSTFLAGS.Split([char]31)) {
            $arguments.Add($argument)
        }
        $cpu = Get-PcaiExplicitCpuTarget -Arguments $arguments.ToArray()
        if (-not $cpu) {
            $arguments.Add('-C')
            $arguments.Add('target-cpu=native')
        }
        $env:CARGO_ENCODED_RUSTFLAGS = $arguments -join [char]31
    } else {
        $cpu = Get-PcaiExplicitCpuTarget -Arguments ([string]$env:RUSTFLAGS -split '\s+')
        if (-not $cpu) {
            $env:RUSTFLAGS = "$($env:RUSTFLAGS) -C target-cpu=native".Trim()
        }
    }
    $env:CARGO_PROFILE_RELEASE_CODEGEN_UNITS = '1'
    [pscustomobject]@{
        CpuTarget = if ($cpu) { $cpu } else { 'native' }
        Host = [Environment]::MachineName
        Architecture = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
        ReleaseCodegenUnits = 1
    }
}
