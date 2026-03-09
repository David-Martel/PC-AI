function Write-Success {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "WARNING: $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "ERROR: $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Cyan
}

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Magenta
    Write-Host ""
}

function Write-SubHeader {
    param([string]$Message)
    Write-Host "--- $Message ---" -ForegroundColor DarkCyan
}

function Write-Bullet {
    param([string]$Message, [string]$Color = 'White')
    Write-Host "  * $Message" -ForegroundColor $Color
}

# Singleton initialization state
$script:PcaiNativeInitialized = $false
$script:PcaiAccelerationManifestPath = $null
$script:PcaiDirectCoreInteropTypeName = 'PcaiDirectCoreInterop'
$script:PcaiDirectCoreInteropLoaded = $false
$script:PcaiDirectCoreInteropDllPath = $null

function Resolve-PcaiAccelerationManifestPath {
    if ($script:PcaiAccelerationManifestPath -and (Test-Path -LiteralPath $script:PcaiAccelerationManifestPath)) {
        return $script:PcaiAccelerationManifestPath
    }

    $moduleRoot = Split-Path -Parent $PSScriptRoot
    $candidates = @(
        (Join-Path $moduleRoot 'PC-AI.Acceleration\PC-AI.Acceleration.psd1'),
        (Join-Path (Join-Path $env:LOCALAPPDATA 'PowerShell\Modules') 'PC-AI.Acceleration\PC-AI.Acceleration.psd1')
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            $script:PcaiAccelerationManifestPath = $candidate
            return $candidate
        }
    }

    $availableModule = Get-Module -ListAvailable -Name 'PC-AI.Acceleration' | Sort-Object Version -Descending | Select-Object -First 1
    if ($availableModule -and $availableModule.Path) {
        $script:PcaiAccelerationManifestPath = $availableModule.Path
        return $availableModule.Path
    }

    return $null
}

function Find-PcaiProbeExistingPath {
    param(
        [Parameter(Mandatory)]
        [string[]]$Candidates,
        [switch]$Directory
    )

    foreach ($candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }

        $pathType = if ($Directory) { 'Container' } else { 'Any' }
        if (-not (Test-Path -LiteralPath $candidate -PathType $pathType)) {
            continue
        }

        try {
            return (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).Path
        }
        catch {
            return $candidate
        }
    }

    return $null
}

function Get-PcaiNonEmptyCandidates {
    [CmdletBinding()]
    [OutputType([string[]])]
    param(
        [AllowNull()]
        [AllowEmptyCollection()]
        [string[]]$Candidates
    )

    return @(
        foreach ($candidate in @($Candidates)) {
            if (-not [string]::IsNullOrWhiteSpace($candidate)) {
                $candidate
            }
        }
    )
}

function Get-PcaiAccelerationProbe {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [switch]$IncludeToolPaths
    )

    $manifestPath = Resolve-PcaiAccelerationManifestPath
    $moduleRoot = if ($manifestPath) { Split-Path -Parent $manifestPath } else { $null }
    $repoRoot = $null

    if ($moduleRoot) {
        $candidateRepoRoot = Split-Path -Parent $moduleRoot
        if (Test-Path -LiteralPath (Join-Path $candidateRepoRoot 'PC-AI.ps1')) {
            $repoRoot = $candidateRepoRoot
        }
    }

    if (-not $repoRoot) {
        $repoRootCandidates = Get-PcaiNonEmptyCandidates -Candidates @(
            $env:PCAI_ROOT,
            'C:\codedev\PC_AI',
            (Join-Path $HOME 'PC_AI')
        )
        if ($repoRootCandidates.Count -gt 0) {
            $repoRoot = Find-PcaiProbeExistingPath -Candidates $repoRootCandidates -Directory
        }
    }

    $dllRoots = @(
        $(if ($repoRoot) { Join-Path $repoRoot 'bin' }),
        $(if ($repoRoot) { Join-Path $repoRoot 'Native\PcaiNative\bin\Release\net8.0\win-x64' }),
        $(if ($repoRoot) { Join-Path $repoRoot '.pcai\build\artifacts\pcai-native' }),
        $(if ($repoRoot) { Join-Path $repoRoot 'Native\pcai_core\target\release' }),
        (Join-Path $env:USERPROFILE 'PC_AI\bin')
    ) | Select-Object -Unique

    $nativeRoot = $null
    foreach ($candidateRoot in $dllRoots) {
        if (-not $candidateRoot -or -not (Test-Path -LiteralPath $candidateRoot -PathType Container)) {
            continue
        }

        if ((Test-Path -LiteralPath (Join-Path $candidateRoot 'PcaiNative.dll')) -and
            (Test-Path -LiteralPath (Join-Path $candidateRoot 'pcai_core_lib.dll'))) {
            $nativeRoot = Find-PcaiProbeExistingPath -Candidates @($candidateRoot) -Directory
            break
        }
    }

    $toolSearchRoots = @(
        (Join-Path $env:USERPROFILE '.cargo\bin'),
        (Join-Path $env:USERPROFILE 'bin'),
        (Join-Path $env:USERPROFILE '.local\bin'),
        (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links'),
        'C:\Program Files\ripgrep',
        'C:\Program Files\fd'
    ) | Select-Object -Unique

    $tools = foreach ($toolName in @('rg', 'fd', 'hyperfine')) {
        $toolPath = $null
        foreach ($searchRoot in $toolSearchRoots) {
            if (-not $searchRoot) {
                continue
            }

            $candidateToolPath = Join-Path $searchRoot "$toolName.exe"
            if (Test-Path -LiteralPath $candidateToolPath) {
                $toolPath = Find-PcaiProbeExistingPath -Candidates @($candidateToolPath)
                break
            }
        }

        if (-not $toolPath) {
            try {
                $whereResult = & where.exe $toolName 2>$null | Select-Object -First 1
                if ($whereResult -and (Test-Path -LiteralPath $whereResult)) {
                    $toolPath = Find-PcaiProbeExistingPath -Candidates @($whereResult)
                }
            }
            catch {
            }
        }

        [PSCustomObject]@{
            Name      = $toolName
            Available = [bool]$toolPath
            Path      = if ($IncludeToolPaths) { $toolPath } else { $null }
        }
    }

    [PSCustomObject]@{
        Timestamp          = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
        RepoRoot           = $repoRoot
        PowerShell7OrNewer = [bool]($PSVersionTable.PSVersion.Major -ge 7)
        AccelerationModule = [PSCustomObject]@{
            ManifestPath = $manifestPath
            Available    = [bool]$manifestPath
        }
        Native = [PSCustomObject]@{
            Root          = $nativeRoot
            PcaiNativeDll = [bool]($nativeRoot -and (Test-Path -LiteralPath (Join-Path $nativeRoot 'PcaiNative.dll')))
            CoreLibDll    = [bool]($nativeRoot -and (Test-Path -LiteralPath (Join-Path $nativeRoot 'pcai_core_lib.dll')))
            InferenceDll  = [bool]($nativeRoot -and (Test-Path -LiteralPath (Join-Path $nativeRoot 'pcai_inference.dll')))
        }
        Tools = @($tools)
    }
}

function Initialize-PcaiDirectCoreInterop {
    [CmdletBinding()]
    [OutputType([type])]
    param(
        [Parameter(Mandatory)]
        [string]$DllPath
    )

    if ($script:PcaiDirectCoreInteropLoaded -and
        $script:PcaiDirectCoreInteropDllPath -eq $DllPath -and
        ($script:PcaiDirectCoreInteropTypeName -as [type])) {
        return ($script:PcaiDirectCoreInteropTypeName -as [type])
    }

    $dllDir = Split-Path -Parent $DllPath
    $currentPath = [System.Environment]::GetEnvironmentVariable('PATH', 'Process')
    if ($currentPath -notlike "*$dllDir*") {
        [System.Environment]::SetEnvironmentVariable('PATH', "$dllDir;$currentPath", 'Process')
    }

    $escapedDllPath = $DllPath.Replace('\', '\\')
    $typeDefinition = @"
using System;
using System.Runtime.InteropServices;

public static class $($script:PcaiDirectCoreInteropTypeName) {
    [DllImport(@"$escapedDllPath", CallingConvention = CallingConvention.Cdecl)]
    public static extern uint pcai_core_test();

    [DllImport(@"$escapedDllPath", CallingConvention = CallingConvention.Cdecl)]
    public static extern uint pcai_cpu_count();

    [DllImport(@"$escapedDllPath", CallingConvention = CallingConvention.Cdecl)]
    public static extern UIntPtr pcai_estimate_tokens([MarshalAs(UnmanagedType.LPUTF8Str)] string text);
}
"@

    if (-not ($script:PcaiDirectCoreInteropTypeName -as [type])) {
        Add-Type -TypeDefinition $typeDefinition -Language CSharp -ErrorAction Stop | Out-Null
    }

    $script:PcaiDirectCoreInteropLoaded = $true
    $script:PcaiDirectCoreInteropDllPath = $DllPath
    return ($script:PcaiDirectCoreInteropTypeName -as [type])
}

function Get-PcaiDirectCoreProbe {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [string]$SampleText = 'The quick brown fox jumps over the lazy dog.'
    )

    $probe = Get-PcaiAccelerationProbe
    $nativeRoot = $probe.Native.Root
    $coreDllPath = if ($nativeRoot) { Join-Path $nativeRoot 'pcai_core_lib.dll' } else { $null }

    if (-not $coreDllPath -or -not (Test-Path -LiteralPath $coreDllPath)) {
        return [PSCustomObject]@{
            Timestamp      = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
            Available      = $false
            DllPath        = $coreDllPath
            CoreAvailable  = $false
            CpuCount       = $null
            TokenEstimate  = $null
            Error          = 'pcai_core_lib.dll not found'
            DirectToRust   = $false
        }
    }

    try {
        $interopType = Initialize-PcaiDirectCoreInterop -DllPath $coreDllPath
        $magic = $interopType::pcai_core_test()
        $cpuCount = $interopType::pcai_cpu_count()
        $tokenEstimate = [uint64]($interopType::pcai_estimate_tokens($SampleText).ToUInt64())

        [PSCustomObject]@{
            Timestamp      = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
            Available      = $true
            DllPath        = $coreDllPath
            CoreAvailable  = ($magic -eq 0x50434149)
            CpuCount       = $cpuCount
            TokenEstimate  = $tokenEstimate
            Error          = $null
            DirectToRust   = $true
        }
    }
    catch {
        [PSCustomObject]@{
            Timestamp      = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
            Available      = $false
            DllPath        = $coreDllPath
            CoreAvailable  = $false
            CpuCount       = $null
            TokenEstimate  = $null
            Error          = $_.Exception.Message
            DirectToRust   = $true
        }
    }
}

function Get-PcaiDirectTokenEstimate {
    [CmdletBinding()]
    [OutputType([uint64])]
    param(
        [Parameter(Mandatory, Position = 0)]
        [AllowEmptyString()]
        [string]$Text
    )

    $probe = Get-PcaiAccelerationProbe
    $nativeRoot = $probe.Native.Root
    $coreDllPath = if ($nativeRoot) { Join-Path $nativeRoot 'pcai_core_lib.dll' } else { $null }

    if (-not $coreDllPath -or -not (Test-Path -LiteralPath $coreDllPath)) {
        throw 'pcai_core_lib.dll not found for direct token estimate.'
    }

    $interopType = Initialize-PcaiDirectCoreInterop -DllPath $coreDllPath
    return [uint64]($interopType::pcai_estimate_tokens($Text).ToUInt64())
}

function Initialize-PcaiNative {
    [CmdletBinding()]
    param([switch]$Force)

    if ($script:PcaiNativeInitialized -and -not $Force) {
        return $true
    }

    $accelModule = Get-Module PC-AI.Acceleration
    if (-not $accelModule) {
        $accelManifest = Resolve-PcaiAccelerationManifestPath
        if ($accelManifest) {
            $accelModule = Import-Module -Name $accelManifest -PassThru -ErrorAction SilentlyContinue
        }
    }

    if ($accelModule) {
        $result = & $accelModule { Initialize-PcaiNative -Force:$Force }
        $script:PcaiNativeInitialized = $result
        return $result
    }

    Write-Error "PC-AI.Acceleration module not found. Native initialization failed."
    return $false
}

Export-ModuleMember -Function Write-Success, Write-Warning, Write-Error, Write-Info, Write-Header, Write-SubHeader, Write-Bullet, Initialize-PcaiNative, Get-PcaiAccelerationProbe, Get-PcaiDirectCoreProbe, Get-PcaiDirectTokenEstimate
