#Requires -Version 7.0
<#
.SYNOPSIS
    Initializes PCAI Native DLLs for high-performance operations

.DESCRIPTION
    Loads the native Rust DLLs and C# P/Invoke wrapper for:
    - pcai_core_lib.dll - Unified Core engine (Search, Duplicates, System)
    - PcaiNative.dll    - Managed C# bridge

    Performance depends on the operation and must be measured on the target host.

    IMPORTANT: Requires PowerShell 7+ (.NET 8) for the C# wrapper to load.
#>

# Module-level state for native tools
$script:PcaiNativeLoaded = $false
$script:PcaiNativeVersion = $null
$script:PcaiNativeDllPath = $null
$script:PcaiNativeCoreLibraryPath = $null
$script:PcaiNativeWrapperPath = $null
$script:PcaiNativeArtifactHash = $null
$script:PcaiNativeRid = $null

function Get-PcaiNativeRuntimeIdentifier {
    [CmdletBinding()]
    [OutputType([string])]
    param()

    $osPart = if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
        'win'
    } elseif ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::OSX)) {
        'osx'
    } else {
        'linux'
    }

    $archPart = switch ([System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture) {
        'X64'   { 'x64' }
        'Arm64' { 'arm64' }
        'X86'   { 'x86' }
        'Arm'   { 'arm' }
        default { $_.ToString().ToLowerInvariant() }
    }

    return "$osPart-$archPart"
}

function Get-PcaiNativeLibraryFileName {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter()]
        [string]$LibraryName = 'pcai_core_lib'
    )

    $stem = [System.IO.Path]::GetFileNameWithoutExtension($LibraryName.Trim())
    if ($stem.StartsWith('libpcai_', [System.StringComparison]::OrdinalIgnoreCase)) {
        $stem = $stem.Substring(3)
    }

    if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
        return "$stem.dll"
    }
    if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::OSX)) {
        return "lib$stem.dylib"
    }
    return "lib$stem.so"
}

function Get-PcaiNativeCandidatePaths {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string[]]$BasePaths
    )

    $coreFileName = Get-PcaiNativeLibraryFileName -LibraryName 'pcai_core_lib'
    $rid = Get-PcaiNativeRuntimeIdentifier
    $candidates = [System.Collections.Generic.List[string]]::new()

    foreach ($basePath in $BasePaths | Where-Object { $_ } | Select-Object -Unique) {
        $resolved = Resolve-Path -Path $basePath -ErrorAction SilentlyContinue
        if (-not $resolved) {
            continue
        }

        $resolvedPath = $resolved.Path
        if ((Test-Path (Join-Path $resolvedPath $coreFileName)) -or
            (Test-Path (Join-Path $resolvedPath 'PcaiNative.dll')) -or
            (Test-Path (Join-Path $resolvedPath "runtimes/$rid/native/$coreFileName"))) {
            $candidates.Add($resolvedPath)
        }

        $bundleRoots = @(
            (Join-Path $resolvedPath 'native-bundles')
            (Join-Path $resolvedPath 'bundles')
            (Join-Path $resolvedPath 'versions')
        )

        foreach ($bundleRoot in $bundleRoots) {
            if (-not (Test-Path $bundleRoot -PathType Container)) {
                continue
            }

            Get-ChildItem -Path $bundleRoot -Directory -ErrorAction SilentlyContinue |
                Sort-Object -Property LastWriteTime -Descending |
                ForEach-Object {
                    $bundlePath = $_.FullName
                    if ((Test-Path (Join-Path $bundlePath $coreFileName)) -or
                        (Test-Path (Join-Path $bundlePath 'PcaiNative.dll'))) {
                        $candidates.Add($bundlePath)
                    }
                }
        }
    }

    return @($candidates | Select-Object -Unique)
}

function Initialize-PcaiNative {
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter()]
        [switch]$Force
    )

    # Explicit overrides are authoritative, including when they are invalid. Validate
    # before discovery or the loaded-state shortcut so fallback cannot replace intent.
    $coreFileName = Get-PcaiNativeLibraryFileName -LibraryName 'pcai_core_lib'
    $explicitCorePath = $null
    $overrideName = $null
    foreach ($name in @('PCAI_CORE_LIB_PATH', 'PCAI_TEST_CORE_DLL', 'PCAI_NATIVE_DIR')) {
        $value = [System.Environment]::GetEnvironmentVariable($name)
        if ([string]::IsNullOrWhiteSpace($value)) { continue }
        $overrideName = $name
        $explicitCorePath = if ($name -eq 'PCAI_NATIVE_DIR' -or
            (Test-Path -LiteralPath $value -PathType Container)) {
            Join-Path $value $coreFileName
        } else { $value }
        break
    }
    if ($overrideName -and -not (Test-Path -LiteralPath $explicitCorePath -PathType Leaf)) {
        Write-Warning "PCAI Native override $overrideName does not identify an existing core library"
        $script:PcaiNativeLoaded = $false
        $script:PcaiNativeVersion = $null
        $script:PcaiNativeDllPath = $null
        $script:PcaiNativeCoreLibraryPath = $null
        $script:PcaiNativeWrapperPath = $null
        $script:PcaiNativeArtifactHash = $null
        return $false
    }

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
    $rid = Get-PcaiNativeRuntimeIdentifier
    $coreFileName = Get-PcaiNativeLibraryFileName -LibraryName 'pcai_core_lib'
    $script:PcaiNativeRid = $rid

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
        foreach ($candidate in @((Join-Path $env:USERPROFILE 'PC_AI'), (Join-Path $HOME 'PC_AI')) | Select-Object -Unique) {
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

    $userHome = if ($HOME) { $HOME } else { $env:USERPROFILE }
    $baseSearchPaths = @(
        $(if ($repoRoot) { Join-Path $repoRoot 'bin' })                                  # Repo bin
        $(if ($repoRoot) { Join-Path $repoRoot "Native/PcaiNative/bin/Release/net8.0/$rid" })
        $(if ($repoRoot) { Join-Path $repoRoot 'Native/PcaiNative/bin/Release/net8.0' })
        $(if ($repoRoot) { Join-Path $repoRoot "Native/PcaiNative/bin/Debug/net8.0/$rid" })
        $(if ($repoRoot) { Join-Path $repoRoot 'Native/PcaiNative/bin/Debug/net8.0' })
        $(if ($repoRoot) { Join-Path $repoRoot '.pcai/build/artifacts/pcai-native' })
        $(if ($repoRoot) { Join-Path $repoRoot 'Native/pcai_core/target/release' })      # Native workspace release
        $(if ($repoRoot) { Join-Path $repoRoot 'target/release' })
        $(if ($repoRoot) { Join-Path $repoRoot 'Native/pcai_core/target/debug' })        # Native workspace debug
        $(if ($repoRoot) { Join-Path $repoRoot 'target/debug' })
        $bridgeBin                                                                         # Installed bridge bin
        $(if ($userHome) { Join-Path $userHome '.local/bin' })
        $(if ($userHome) { Join-Path $userHome 'PC_AI/bin' })                              # Legacy compatibility path
    ) | Select-Object -Unique | Where-Object { $_ }

    $searchPaths = Get-PcaiNativeCandidatePaths -BasePaths $baseSearchPaths

    $wrapperPath = $null
    $coreLibPath = $null

    if ($explicitCorePath) {
        $coreLibPath = (Resolve-Path -LiteralPath $explicitCorePath -ErrorAction Stop).Path
    }

    foreach ($searchPath in $searchPaths) {
        $candidateCore = Join-Path $searchPath $coreFileName
        $candidateRuntimeCore = Join-Path $searchPath "runtimes/$rid/native/$coreFileName"
        $candidateWrapper = Join-Path $searchPath 'PcaiNative.dll'

        if ((Test-Path $candidateWrapper -PathType Leaf) -and (Test-Path $candidateCore -PathType Leaf)) {
            $wrapperPath = $candidateWrapper
            if (-not $coreLibPath) { $coreLibPath = $candidateCore }
            break
        }
        if (-not $wrapperPath -and (Test-Path $candidateWrapper -PathType Leaf)) {
            $wrapperPath = $candidateWrapper
        }
        if (-not $coreLibPath) {
            if (Test-Path $candidateCore -PathType Leaf) {
                $coreLibPath = $candidateCore
            } elseif (Test-Path $candidateRuntimeCore -PathType Leaf) {
                $coreLibPath = $candidateRuntimeCore
            }
        }
    }

    if (-not $wrapperPath -or -not $coreLibPath) {
        Write-Verbose "PCAI Native artifacts not found in search paths (Wrapper=$wrapperPath, CoreLib=$coreLibPath, Expected=$coreFileName)"
        $script:PcaiNativeLoaded = $false
        return $false
    }

    $dllPath = Split-Path -Parent $coreLibPath
    Write-Verbose "Found PCAI Native wrapper at '$wrapperPath' and core library at '$coreLibPath'"
    $script:PcaiNativeDllPath = $dllPath
    $script:PcaiNativeCoreLibraryPath = $coreLibPath
    $script:PcaiNativeWrapperPath = $wrapperPath
    $script:PcaiNativeArtifactHash = (Get-FileHash -LiteralPath $coreLibPath -Algorithm SHA256).Hash

    try {
        # Ensure C# NativeResolver sees the exact resolved core library path
        [System.Environment]::SetEnvironmentVariable('PCAI_CORE_LIB_PATH', $coreLibPath, 'Process')

        # Prepend the native library directory to PATH using the OS path separator
        $pathSep = [System.IO.Path]::PathSeparator
        $currentPath = [System.Environment]::GetEnvironmentVariable('PATH', 'Process')
        $pathEntries = @($currentPath -split [regex]::Escape([string]$pathSep) | Where-Object { $_ })
        if ($pathEntries -notcontains $dllPath) {
            [System.Environment]::SetEnvironmentVariable('PATH', "$dllPath$pathSep$currentPath", 'Process')
            Write-Verbose "Added $dllPath to process PATH"
        }

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
            Write-Warning 'PCAI Core library loaded but not functional'
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

    # Initialization validates explicit overrides before its cached-state shortcut.
    # Always enter it so status callers cannot bypass that validation.
    $null = Initialize-PcaiNative
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
    $coreFileName = Get-PcaiNativeLibraryFileName -LibraryName 'pcai_core_lib'
    $inferenceFileName = Get-PcaiNativeLibraryFileName -LibraryName 'pcai_inference'

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
            PcaiNative = [bool]($script:PcaiNativeWrapperPath -and (Test-Path $script:PcaiNativeWrapperPath))
            CoreLib    = [bool]($script:PcaiNativeCoreLibraryPath -and (Test-Path $script:PcaiNativeCoreLibraryPath))
            Inference  = (Test-Path (Join-Path $dllPath $inferenceFileName))
        }
    }

    [PSCustomObject]@{
        Available         = $available
        Version           = $script:PcaiNativeVersion
        RuntimeIdentifier = if ($script:PcaiNativeRid) { $script:PcaiNativeRid } else { Get-PcaiNativeRuntimeIdentifier }
        DllPath           = $dllPath
        CoreLibraryPath   = $script:PcaiNativeCoreLibraryPath
        WrapperPath       = $script:PcaiNativeWrapperPath
        ArtifactHash      = $script:PcaiNativeArtifactHash
        CoreLibraryName   = $coreFileName
        CoreAvailable     = $coreAvailable
        CpuCount          = if ($coreAvailable) { [PcaiNative.PcaiCore]::CpuCount } else { [uint32][Environment]::ProcessorCount }
        Modules           = if ($available) {
            [PSCustomObject]@{
                Core        = $coreAvailable
                Search      = $searchAvailable
                System      = $systemAvailable
                Performance = $performanceAvailable
                Fs          = $fsAvailable
            }
        } else { $null }
        Dlls              = $dlls
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
        if ([string]::IsNullOrEmpty($Text)) {
            return [uint64]0
        }

        # Preserve the existing BMP heuristic and avoid a per-scalar loop when
        # there are no surrogate code units to decode.
        if (-not [regex]::IsMatch($Text, '[\uD800-\uDFFF]')) {
            $count = [uint64]0
            foreach ($match in [regex]::Matches($Text, '[\p{L}\p{N}]+|\s+|[^\p{L}\p{N}\s]')) {
                $byteLen = [System.Text.Encoding]::UTF8.GetByteCount($match.Value)
                if ($byteLen -gt 8) {
                    $count += [uint64][Math]::Ceiling($byteLen / 4.0)
                } else {
                    $count += [uint64]1
                }
            }
            return [uint64]($count + 1)
        }

        # Rust regex operates on Unicode scalars, not UTF-16 code units. Keep
        # L/N runs, whitespace runs and individual other scalars before sizing.
        $count = [uint64]1 # BOS for nonempty input
        $pieceBytes = 0
        $previousKind = -1
        foreach ($rune in $Text.EnumerateRunes()) {
            $kind = if ([System.Text.Rune]::IsLetter($rune) -or [System.Text.Rune]::IsNumber($rune)) {
                0
            } elseif ([System.Text.Rune]::IsWhiteSpace($rune)) { 1 } else { 2 }
            if ($pieceBytes -gt 0 -and ($kind -ne $previousKind -or $kind -eq 2)) {
                $count += if ($pieceBytes -gt 8) { [uint64][Math]::Ceiling($pieceBytes / 4.0) } else { [uint64]1 }
                $pieceBytes = 0
            }
            $pieceBytes += $rune.Utf8SequenceLength
            $previousKind = $kind
        }
        if ($pieceBytes -gt 0) {
            $count += if ($pieceBytes -gt 8) { [uint64][Math]::Ceiling($pieceBytes / 4.0) } else { [uint64]1 }
        }
        return $count
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

function Invoke-PcaiNativeProcessLassoSnapshot {
    <#
    .SYNOPSIS
        Gets a native snapshot of Process Lasso config and recent log activity.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter()]
        [string]$ConfigPath = 'C:\ProgramData\ProcessLasso\config\prolasso.ini',

        [Parameter()]
        [string]$LogPath = 'C:\ProgramData\ProcessLasso\logs\processlasso.log',

        [Parameter()]
        [int]$LookbackMinutes = 60
    )

    if (-not (Test-PcaiNativeAvailable)) {
        throw 'PCAI Native tools not available.'
    }

    if (-not ([System.Management.Automation.PSTypeName]'PcaiNative.ProcessLassoModule').Type) {
        throw 'PcaiNative.ProcessLassoModule is not loaded.'
    }

    return [PcaiNative.ProcessLassoModule]::GetSnapshotJson($ConfigPath, $LogPath, [uint32]$LookbackMinutes)
}
