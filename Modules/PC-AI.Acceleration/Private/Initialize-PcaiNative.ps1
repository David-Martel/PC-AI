#Requires -Version 7.0
<#
.SYNOPSIS
    Initializes PCAI Native DLLs for high-performance operations

.DESCRIPTION
    Loads the native Rust DLLs and C# P/Invoke wrapper for:
    - pcai_core_lib.dll - Unified Core engine (Search, Duplicates, System)
    - PcaiNative.dll    - Managed C# bridge

    These provide 5-15x speedup over PowerShell equivalents.

    IMPORTANT: Requires PowerShell 7+ (.NET 8) for the C# wrapper to load.
#>

# Module-level state for native tools
$script:PcaiNativeLoaded = $false
$script:PcaiNativeVersion = $null
$script:PcaiNativeDllPath = $null

function Initialize-PcaiNative {
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter()]
        [switch]$Force
    )

    if ($script:PcaiNativeLoaded -and -not $Force) {
        Write-Verbose 'PCAI Native already loaded, skipping initialization'
        return $true
    }

    # Verify PowerShell 7+
    if ($PSVersionTable.PSVersion.Major -lt 7) {
        Write-Warning "PCAI Native requires PowerShell 7+ (you have $($PSVersionTable.PSVersion))"
        $script:PcaiNativeLoaded = $false
        return $false
    }

    Write-Verbose 'Initializing PCAI Native tools...'

    # Resolve the repo root first so moved checkouts do not depend on ~/PC_AI.
    $repoRoot = $null
    if (Get-Command Resolve-PcaiRepoRoot -ErrorAction SilentlyContinue) {
        try {
            $repoRoot = Resolve-PcaiRepoRoot -StartPath $PSScriptRoot
        } catch {}
    }
    if (-not $repoRoot -and $env:PCAI_ROOT -and (Test-Path $env:PCAI_ROOT)) {
        $repoRoot = $env:PCAI_ROOT
    }
    if (-not $repoRoot) {
        try {
            $repoRoot = (Join-Path $PSScriptRoot '..\..\..' | Resolve-Path -ErrorAction Stop).Path
        } catch {}
    }
    if (-not $repoRoot) {
        foreach ($candidate in @('C:\codedev\PC_AI', (Join-Path $HOME 'PC_AI')) | Select-Object -Unique) {
            if (-not $candidate -or -not (Test-Path $candidate)) {
                continue
            }

            try {
                $repoRoot = (Resolve-Path -Path $candidate -ErrorAction Stop).Path
                break
            } catch {}
        }
    }

    $bridgeModule = Get-Module -ListAvailable -Name 'PcaiNativeBridge' -ErrorAction SilentlyContinue |
        Sort-Object -Property Version -Descending |
        Select-Object -First 1
    $bridgeBin = $null
    if ($bridgeModule -and $bridgeModule.Path) {
        $bridgeModuleRoot = Split-Path -Parent $bridgeModule.Path
        $bridgeBin = Join-Path $bridgeModuleRoot 'bin'
    }

    # Find DLL locations
    $searchPaths = @(
        $(if ($repoRoot) { Join-Path $repoRoot 'bin' })                                  # Repo bin
        $(if ($repoRoot) { Join-Path $repoRoot 'Native\PcaiNative\bin\Release\net8.0\win-x64' })
        $(if ($repoRoot) { Join-Path $repoRoot '.pcai\build\artifacts\pcai-native' })
        $(if ($repoRoot) { Join-Path $repoRoot 'Native\pcai_core\target\release' })      # Native workspace target
        $bridgeBin                                                                         # Installed bridge bin
        (Join-Path $env:USERPROFILE 'PC_AI\bin')                                          # Legacy compatibility path
    ) | Select-Object -Unique | Where-Object { $_ }

    $dllPath = $null
    foreach ($searchPath in $searchPaths) {
        $resolved = Resolve-Path -Path $searchPath -ErrorAction SilentlyContinue
        if ($resolved) {
            $coreDll = Join-Path $resolved.Path 'pcai_core_lib.dll'
            $wrapperDll = Join-Path $resolved.Path 'PcaiNative.dll'


            if ((Test-Path $coreDll) -and (Test-Path $wrapperDll)) {
                $dllPath = $resolved.Path
                break
            }
        }
    }

    if (-not $dllPath) {
        Write-Verbose 'PCAI Native DLLs (Core or Wrapper) not found in search paths'
        $script:PcaiNativeLoaded = $false
        return $false
    }

    Write-Verbose "Found PCAI Native DLLs at: $dllPath"
    $script:PcaiNativeDllPath = $dllPath

    try {
        # CRITICAL: Add the DLL directory to the process PATH so native DLLs can be found
        # This allows the C# wrapper to locate specialized Rust DLLs
        $currentPath = [System.Environment]::GetEnvironmentVariable('PATH', 'Process')
        if ($currentPath -notlike "*$dllPath*") {
            [System.Environment]::SetEnvironmentVariable('PATH', "$dllPath;$currentPath", 'Process')
            Write-Verbose "Added $dllPath to process PATH"
        }

        # Load the C# wrapper assembly
        $wrapperPath = Join-Path $dllPath 'PcaiNative.dll'

        # Check if already loaded
        $loadedAssembly = [System.AppDomain]::CurrentDomain.GetAssemblies() |
            Where-Object { $_.GetName().Name -eq 'PcaiNative' }

        if (-not $loadedAssembly) {
            Add-Type -Path $wrapperPath -ErrorAction Stop
            Write-Verbose 'Loaded PcaiNative.dll assembly'
        } else {
            Write-Verbose 'PcaiNative assembly already loaded'
        }

        # Test core availability
        if ([PcaiNative.PcaiCore]::IsAvailable) {
            $script:PcaiNativeVersion = [PcaiNative.PcaiCore]::Version
            Write-Verbose "PCAI System version: $($script:PcaiNativeVersion)"

            Write-Verbose 'PCAI Core loaded and functional'

            $script:PcaiNativeLoaded = $true
            return $true
        } else {
            Write-Warning 'PCAI Core DLL loaded but not functional'
            $script:PcaiNativeLoaded = $false
            return $false
        }
    } catch {
        Write-Warning "Failed to load PCAI Native: $_"
        $script:PcaiNativeLoaded = $false
        return $false
    }
}

function Test-PcaiNativeAvailable {
    <#
    .SYNOPSIS
        Tests if PCAI Native tools are available and functional
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    if (-not $script:PcaiNativeLoaded) {
        # Try to initialize
        $null = Initialize-PcaiNative
    }

    return $script:PcaiNativeLoaded
}

function Get-PcaiNativeStatus {
    <#
    .SYNOPSIS
        Gets detailed status of PCAI Native tools
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param()

    # Ensure initialized
    $available = Test-PcaiNativeAvailable
    $dllPath = $script:PcaiNativeDllPath

    $coreAvailable = $false
    $searchAvailable = $false
    $systemAvailable = $false
    $fsAvailable = $false
    $performanceAvailable = $false

    if ($available) {
        $coreAvailable = [PcaiNative.PcaiCore]::IsAvailable
        if (([System.Management.Automation.PSTypeName]'PcaiNative.PcaiSearch').Type) {
            $searchAvailable = [PcaiNative.PcaiSearch]::IsAvailable
        }
        if (([System.Management.Automation.PSTypeName]'PcaiNative.SystemModule').Type) {
            $systemAvailable = [PcaiNative.SystemModule]::IsAvailable
        }
        if (([System.Management.Automation.PSTypeName]'PcaiNative.FsModule').Type) {
            $fsAvailable = [PcaiNative.FsModule]::IsAvailable
        }
        $performanceAvailable = $coreAvailable
    }

    $dlls = $null
    if ($dllPath) {
        $dlls = [PSCustomObject]@{
            PcaiNative = (Test-Path (Join-Path $dllPath 'PcaiNative.dll'))
            CoreLib    = (Test-Path (Join-Path $dllPath 'pcai_core_lib.dll'))
            Inference  = (Test-Path (Join-Path $dllPath 'pcai_inference.dll'))
        }
    }

    [PSCustomObject]@{
        Available     = $available
        Version       = $script:PcaiNativeVersion
        DllPath       = $dllPath
        CoreAvailable = $coreAvailable
        CpuCount      = if ($coreAvailable) { [PcaiNative.PcaiCore]::CpuCount } else { [uint32][Environment]::ProcessorCount }
        Modules       = if ($available) {
            [PSCustomObject]@{
                Core        = $coreAvailable
                Search      = $searchAvailable
                System      = $systemAvailable
                Performance = $performanceAvailable
                Fs          = $fsAvailable
            }
        } else { $null }
        Dlls          = $dlls
    }
}

function Invoke-PcaiNativeDuplicates {
    <#
    .SYNOPSIS
        Finds duplicate files using native parallel SHA-256 hashing
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [ValidateScript({ Test-Path $_ -PathType Container })]
        [string]$Path,

        [Parameter()]
        [int64]$MinimumSize = 0,

        [Parameter()]
        [string]$IncludePattern,

        [Parameter()]
        [string]$ExcludePattern,

        [Parameter()]
        [switch]$StatsOnly
    )

    if (-not (Test-PcaiNativeAvailable)) {
        throw 'PCAI Native tools not available.'
    }

    $resolvedPath = Resolve-Path $Path | Select-Object -ExpandProperty Path

    if ($StatsOnly) {
        return [PcaiNative.PcaiSearch]::FindDuplicatesStats($resolvedPath, [uint64]$MinimumSize, $IncludePattern, $ExcludePattern)
    }

    $result = [PcaiNative.PcaiSearch]::FindDuplicates($resolvedPath, [uint64]$MinimumSize, $IncludePattern, $ExcludePattern)
    return $result
    return $null
}

function Invoke-PcaiNativeFileSearch {
    <#
    .SYNOPSIS
        Fast file search using native parallel directory walking
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Pattern,

        [Parameter()]
        [string]$Path,

        [Parameter()]
        [int64]$MaxResults = 0,

        [Parameter()]
        [switch]$StatsOnly
    )

    if (-not (Test-PcaiNativeAvailable)) {
        throw 'PCAI Native tools not available.'
    }

    $resolvedPath = if ($Path) {
        Resolve-Path $Path | Select-Object -ExpandProperty Path
    } else {
        $null
    }

    if ($StatsOnly) {
        return [PcaiNative.PcaiSearch]::FindFilesStats($Pattern, $resolvedPath, [uint64]$MaxResults)
    }

    $result = [PcaiNative.PcaiSearch]::FindFiles($Pattern, $resolvedPath, [uint32]$MaxResults)
    return $result
    return $null
}

function Invoke-PcaiNativeDirectoryManifest {
    <#
    .SYNOPSIS
        Collects a shallow or deep directory manifest using native traversal
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [ValidateScript({ Test-Path $_ -PathType Container })]
        [string]$Path,

        [Parameter()]
        [uint32]$MaxDepth = 0,

        [Parameter()]
        [uint64]$MaxResults = 0,

        [Parameter()]
        [switch]$StatsOnly
    )

    if (-not (Test-PcaiNativeAvailable)) {
        throw 'PCAI Native tools not available.'
    }

    $resolvedPath = Resolve-Path $Path | Select-Object -ExpandProperty Path

    if ($StatsOnly) {
        return [PcaiNative.PcaiSearch]::CollectDirectoryManifestStats($resolvedPath, $MaxDepth, $MaxResults)
    }

    return [PcaiNative.PcaiSearch]::CollectDirectoryManifest($resolvedPath, $MaxDepth, $MaxResults)
}

function Invoke-PcaiNativeContentSearch {
    <#
    .SYNOPSIS
        Fast content search using native parallel regex matching
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Pattern,

        [Parameter()]
        [string]$Path,

        [Parameter()]
        [string]$FilePattern,

        [Parameter()]
        [int64]$MaxResults = 0,

        [Parameter()]
        [int]$ContextLines = 0,

        [Parameter()]
        [switch]$StatsOnly
    )

    if (-not (Test-PcaiNativeAvailable)) {
        throw 'PCAI Native tools not available.'
    }

    $resolvedPath = if ($Path) {
        Resolve-Path $Path | Select-Object -ExpandProperty Path
    } else {
        $null
    }

    if ($StatsOnly) {
        return [PcaiNative.PcaiSearch]::SearchContentStats($Pattern, $resolvedPath, $FilePattern, [uint64]$MaxResults)
    }

    $result = [PcaiNative.PcaiSearch]::SearchContent($Pattern, $resolvedPath, $FilePattern, [uint32]$MaxResults, [uint32]$ContextLines)
    return $result
    return $null
}

function Invoke-PcaiNativeSystemInfo {
    <#
    .SYNOPSIS
        Native system interrogation for hardware and OS telemetry
    #>
    [CmdletBinding()]
    param(
        [Parameter()]
        [switch]$MetricsOnly,

        [Parameter()]
        [switch]$HighFidelity
    )

    if (-not (Test-PcaiNativeAvailable)) { return $null }

    $result = if ($MetricsOnly) {
        [PcaiNative.PerformanceModule]::QueryHardwareMetrics()
    } else {
        [PcaiNative.SystemModule]::QuerySystemInfo()
    }
    return $result
    return $null
}

function Test-PcaiResourceSafety {
    <#
    .SYNOPSIS
        Checks if system resources are within safety limits (e.g. 80% load)
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter()]
        [float]$GpuLimit = 0.8
    )

    if (-not (Test-PcaiNativeAvailable)) { return $true }
    return [PcaiNative.PcaiCore]::CheckResourceSafety($GpuLimit)
}

function Invoke-PcaiNativeUnifiedHardwareReport {
    <#
    .SYNOPSIS
        Invokes unified hardware report generation through the native bridge.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter()]
        [ValidateSet("Basic", "Normal", "Full")]
        [string]$Verbosity = "Normal"
    )

    if (-not (Test-PcaiNativeAvailable)) {
        throw 'PCAI Native tools not available.'
    }

    $verbosityEnum = [PcaiNative.DiagnosticVerbosity]::$Verbosity
    return [PcaiNative.PcaiDiagnostics]::GetUnifiedHardwareReportJson($verbosityEnum)
}

function Invoke-PcaiNativeEstimateTokens {
    <#
    .SYNOPSIS
        Estimates tokens using the native core implementation.
    #>
    [CmdletBinding()]
    [OutputType([uint64])]
    param(
        [Parameter(Mandatory, Position = 0)]
        [AllowEmptyString()]
        [string]$Text
    )

    if (-not (Test-PcaiNativeAvailable)) {
        return [uint64](($Text.Split(' ') | Where-Object { $_ }).Count * 1.2 + 1)
    }

    return [PcaiNative.PcaiCore]::EstimateTokens($Text)
}

function Get-PcaiTokenEstimate {
    <#
    .SYNOPSIS
        Estimates the number of tokens in a string for Gemma-like models natively
    #>
    [CmdletBinding()]
    [OutputType([uint64])]
    param(
        [Parameter(Mandatory, Position = 0)]
        [AllowEmptyString()]
        [string]$Text
    )

    return Invoke-PcaiNativeEstimateTokens -Text $Text
}
