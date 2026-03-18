#Requires -Version 5.1
<#
.SYNOPSIS
    Initializes CUDA environment variables for the current PowerShell session.

.DESCRIPTION
    Detects installed CUDA toolkits using a preferred version list and common
    environment variables, then sets CUDA_PATH/CUDA_HOME and updates PATH with
    CUDA bin and nvvm/bin. Intended for build and doc pipelines (non-destructive).

.PARAMETER PreferredVersions
    Ordered list of CUDA versions to probe under the standard Windows install path.

.PARAMETER CudaPath
    Explicit CUDA installation path to prefer over auto-detection.

.PARAMETER Quiet
    Suppress informational output (still returns a result object).
#>
[CmdletBinding()]
param(
    [string[]]$PreferredVersions = @('v13.1', 'v13.0', 'v12.9', 'v12.8', 'v12.6', 'v12.5'),
    [string]$CudaPath,
    [switch]$Quiet,
    [switch]$InstallMachineGlobal,
    [switch]$WorkaroundMsvc1944
)

function Add-ToPath {
    param([string]$PathSegment)
    if (-not $PathSegment) { return $false }
    if (-not (Test-Path $PathSegment)) { return $false }
    if ($env:PATH -notlike "*$PathSegment*") {
        $env:PATH = "$PathSegment;$env:PATH"
        return $true
    }
    return $false
}

function Add-ToEnvList {
    param(
        [string]$Name,
        [string]$Value
    )
    if (-not $Name -or -not $Value) { return $false }
    if (-not (Test-Path $Value)) { return $false }
    $item = Get-Item -Path "Env:$Name" -ErrorAction SilentlyContinue
    $current = if ($item) { $item.Value } else { $null }
    if ($current -notlike "*$Value*") {
        if ($current) {
            Set-Item -Path "Env:$Name" -Value "$Value;$current"
        } else {
            Set-Item -Path "Env:$Name" -Value "$Value"
        }
        return $true
    }
    return $false
}

function Get-ShortPath {
    param([string]$Path)
    if (-not $Path) { return $null }
    try {
        $short = & cmd /c 'for %I in (\'$Path\") do @echo %~sI" 2>$null
        $short = $short | Select-Object -First 1
        if ($short) {
            $short = $short.Trim()
            if ($short -and (Test-Path $short)) { return $short }
        }
    } catch {
        return $null
    }
    return $null
}

function Get-CudaCandidates {
    param(
        [string[]]$PreferredVersions,
        [string]$CudaPath
    )
    $candidates = @()
    if ($CudaPath) { $candidates += $CudaPath }

    foreach ($ver in $PreferredVersions) {
        $candidates += "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$ver"
    }

    if ($env:CUDA_PATH) { $candidates += $env:CUDA_PATH }
    if ($env:CUDA_HOME) { $candidates += $env:CUDA_HOME }

    $envCandidates = Get-ChildItem Env:CUDA_PATH_V* -ErrorAction SilentlyContinue |
        ForEach-Object { $_.Value }
    if ($envCandidates) { $candidates += $envCandidates }

    return $candidates |
        Where-Object { $_ -and $_.Trim().Length -gt 0 } |
        Select-Object -Unique
}

function Resolve-MsvcCompilerBin {
    $candidates = [System.Collections.Generic.List[string]]::new()

    $currentCcbin = $env:NVCC_CCBIN
    if ($currentCcbin -and (Test-Path -LiteralPath $currentCcbin)) {
        if (Test-Path -LiteralPath (Join-Path $currentCcbin 'cl.exe')) {
            $candidates.Add($currentCcbin)
        }
    }

    $clCommand = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($clCommand -and $clCommand.Source -and (Test-Path -LiteralPath $clCommand.Source)) {
        $candidates.Add((Split-Path -Parent $clCommand.Source))
    }

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $installRoots = & $vswhere -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        foreach ($root in @($installRoots)) {
            if (-not $root) { continue }
            $msvcRoot = Join-Path $root 'VC\Tools\MSVC'
            if (-not (Test-Path -LiteralPath $msvcRoot)) { continue }
            $toolsetRoots = Get-ChildItem -Path $msvcRoot -Directory -ErrorAction SilentlyContinue
            foreach ($toolsetRoot in $toolsetRoots) {
                $binDir = Join-Path $toolsetRoot.FullName 'bin\Hostx64\x64'
                if (Test-Path -LiteralPath (Join-Path $binDir 'cl.exe')) {
                    $candidates.Add($binDir)
                }
            }
        }
    }

    if (-not $candidates.Count) {
        foreach ($base in @('C:\Program Files\Microsoft Visual Studio', 'C:\Program Files (x86)\Microsoft Visual Studio')) {
            if (-not (Test-Path -LiteralPath $base)) { continue }
            $fallback = Get-ChildItem -Path $base -Filter cl.exe -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -match '\\VC\\Tools\\MSVC\\[^\\]+\\bin\\Hostx64\\x64\\cl\.exe$' } |
                Select-Object -ExpandProperty FullName
            foreach ($clPath in @($fallback)) {
                if ($clPath -and (Test-Path -LiteralPath $clPath)) {
                    $candidates.Add((Split-Path -Parent $clPath))
                }
            }
        }
    }

    if (-not $candidates.Count) { return $null }

    return $candidates |
        Select-Object -Unique |
        Sort-Object {
            if ($_ -match '\\MSVC\\(?<ver>[^\\]+)\\bin\\Hostx64\\x64$') {
                return [version]$Matches.ver
            }
            return [version]'0.0.0'
        } -Descending |
        Select-Object -First 1
}

function Initialize-CudaEnvironment {
    [CmdletBinding()]
    param(
        [string[]]$PreferredVersions = @('v13.1', 'v13.0', 'v12.9', 'v12.8', 'v12.6', 'v12.5'),
        [string]$CudaPath,
        [switch]$Quiet,
        [switch]$InstallMachineGlobal,
        [switch]$WorkaroundMsvc1944
    )

    $candidates = Get-CudaCandidates -PreferredVersions $PreferredVersions -CudaPath $CudaPath
    $selected = $null
    $reasons = @()

    foreach ($candidate in $candidates) {
        if (-not (Test-Path $candidate)) {
            $reasons += "Not found: $candidate"
            continue
        }

        $nvcc = Join-Path $candidate 'bin\nvcc.exe'
        $include = Join-Path $candidate 'include\cuda_runtime.h'

        if (-not (Test-Path $nvcc)) {
            $reasons += "nvcc missing: $candidate"
            continue
        }
        if (-not (Test-Path $include)) {
            $reasons += "cuda_runtime.h missing: $candidate"
            continue
        }

        $selected = $candidate
        break
    }

    if (-not $selected) {
        return [PSCustomObject]@{
            Found       = $false
            CudaPath    = $null
            Nvcc        = $null
            Cicc        = $null
            Include     = $null
            PathUpdated = $false
            Notes       = $reasons
        }
    }

    $shortPath = Get-ShortPath -Path $selected
    if ($shortPath -and ($shortPath -ne $selected)) {
        $selected = $shortPath
    }

    # SET EXPLICIT ENVIRONMENT VARIABLES (no fallbacks)
    $env:CUDA_PATH = $selected
    $env:CUDA_HOME = $selected
    $env:CUDA_DIR = $selected

    $pathUpdated = $false
    $pathUpdated = (Add-ToPath (Join-Path $selected 'bin')) -or $pathUpdated
    $pathUpdated = (Add-ToPath (Join-Path $selected 'nvvm\bin')) -or $pathUpdated
    $pathUpdated = (Add-ToPath (Join-Path $selected 'libnvvp')) -or $pathUpdated

    $includeDir = Join-Path $selected 'include'
    $libDir = Join-Path $selected 'lib\x64'
    $includeUpdated = Add-ToEnvList -Name 'INCLUDE' -Value $includeDir
    $libUpdated = Add-ToEnvList -Name 'LIB' -Value $libDir

    $nvccPath = Join-Path $selected 'bin\nvcc.exe'
    $ciccPath = Join-Path $selected 'nvvm\bin\cicc.exe'
    $includePath = Join-Path $selected 'include\cuda_runtime.h'
    $libPath = Join-Path $selected 'lib\x64'

    # Set CMake-specific CUDA variables (no fallbacks)
    $env:CMAKE_CUDA_COMPILER = ($nvccPath -replace '\\', '/')
    $env:CUDAToolkit_ROOT = ($selected -replace '\\', '/')
    $msvcBin = Resolve-MsvcCompilerBin
    if ($msvcBin) {
        $env:NVCC_CCBIN = ($msvcBin -replace '\\', '/')
        Add-ToPath $msvcBin | Out-Null

        # Resolve MSVC include/lib paths from the bin directory.
        # nvcc host compiler (cl.exe) requires INCLUDE and LIB to find
        # MSVC standard headers and the Windows SDK (ucrt, um, shared).
        # Without these, nvcc fails with:
        #   "nvcc fatal: Failed to preprocess host compiler properties."
        $msvcToolsetRoot = ($msvcBin -replace '[\\/]bin[\\/]Hostx64[\\/]x64$', '')
        $msvcInclude = Join-Path $msvcToolsetRoot 'include'
        $msvcLib = Join-Path $msvcToolsetRoot 'lib\x64'
        if (Test-Path $msvcInclude) {
            $includeUpdated = (Add-ToEnvList -Name 'INCLUDE' -Value $msvcInclude) -or $includeUpdated
        }
        if (Test-Path $msvcLib) {
            $libUpdated = (Add-ToEnvList -Name 'LIB' -Value $msvcLib) -or $libUpdated
        }

        # Resolve Windows SDK include/lib paths (ucrt, um, shared).
        $sdkRoot = "${env:ProgramFiles(x86)}\Windows Kits\10"
        if (Test-Path $sdkRoot) {
            $sdkIncludeBase = Join-Path $sdkRoot 'Include'
            $sdkLibBase = Join-Path $sdkRoot 'Lib'
            # Pick the latest SDK version
            $sdkVer = Get-ChildItem -Path $sdkIncludeBase -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '^\d+\.\d+\.\d+\.\d+$' } |
                Sort-Object { [version]$_.Name } -Descending |
                Select-Object -First 1 -ExpandProperty Name
            if ($sdkVer) {
                foreach ($sub in 'ucrt', 'um', 'shared') {
                    $sdkInc = Join-Path $sdkIncludeBase "$sdkVer\$sub"
                    if (Test-Path $sdkInc) {
                        $includeUpdated = (Add-ToEnvList -Name 'INCLUDE' -Value $sdkInc) -or $includeUpdated
                    }
                }
                foreach ($sub in 'ucrt', 'um') {
                    $sdkLib = Join-Path $sdkLibBase "$sdkVer\$sub\x64"
                    if (Test-Path $sdkLib) {
                        $libUpdated = (Add-ToEnvList -Name 'LIB' -Value $sdkLib) -or $libUpdated
                    }
                }
                if (-not $Quiet) {
                    Write-Host "  Windows SDK $sdkVer INCLUDE/LIB paths added" -ForegroundColor DarkGray
                }
            }
        }
        if (-not $Quiet) {
            Write-Host "  MSVC INCLUDE/LIB paths added for nvcc host compiler" -ForegroundColor DarkGray
        }
    } elseif (Test-Path Env:NVCC_CCBIN) {
        Remove-Item Env:NVCC_CCBIN -ErrorAction SilentlyContinue
    }

    if (-not $env:CUDA_DEVICE_ORDER) {
        $env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
    }

    # Set CUDA compute capabilities for common architectures
    # 75=Turing, 80=Ampere, 86=GA102, 89=Ada, 90=Hopper, 120=Blackwell
    if (-not $env:CUDAARCHS) {
        $env:CUDAARCHS = '75;80;86;89;120'
    }

    if ($WorkaroundMsvc1944) {
        $msvcBin = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64'
        if (Test-Path $msvcBin) {
            $env:NVCC_CCBIN = $msvcBin
            $pathUpdated = (Add-ToPath $msvcBin) -or $pathUpdated
            if (-not $Quiet) { Write-Host "  NVCC_CCBIN workaround applied: $msvcBin" -ForegroundColor DarkGray }
        } else {
            if (-not $Quiet) { Write-Warning 'MSVC 19.44 Hostx64 bin not found; skipping WorkaroundMsvc1944.' }
        }
    }

    if ($InstallMachineGlobal) {
        $currentCudaPath = [System.Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine')
        if ($currentCudaPath -ne $selected) {
            [System.Environment]::SetEnvironmentVariable('CUDA_PATH', $selected, 'Machine')
            if (-not $Quiet) { Write-Host 'Registered CUDA_PATH globally to Machine scope.' -ForegroundColor Green }
        }

        $machinePath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
        $binDir = Join-Path $selected 'bin'
        $nvvmDir = Join-Path $selected 'nvvm\bin'
        $libnvvpDir = Join-Path $selected 'libnvvp'

        $envParts = @()
        if ($machinePath -notlike "*$binDir*") { $envParts += $binDir }
        if ($machinePath -notlike "*$nvvmDir*") { $envParts += $nvvmDir }
        if ((Test-Path $libnvvpDir) -and ($machinePath -notlike "*$libnvvpDir*")) { $envParts += $libnvvpDir }

        if ($envParts.Count -gt 0) {
            $newMachinePath = ($envParts -join ';') + ';' + $machinePath
            [System.Environment]::SetEnvironmentVariable('Path', $newMachinePath, 'Machine')
            if (-not $Quiet) { Write-Host "Added $($envParts.Count) CUDA locations to Machine Path." -ForegroundColor Green }
        }
    }

    $result = [PSCustomObject]@{
        Found                = $true
        CudaPath             = $selected
        SelectedVersion      = (Split-Path -Leaf $selected)
        Nvcc                 = $nvccPath
        Cicc                 = $ciccPath
        Include              = $includePath
        Lib                  = $libPath
        PathUpdated          = $pathUpdated
        IncludeUpdated       = $includeUpdated
        LibUpdated           = $libUpdated
        CudaArchs            = $env:CUDAARCHS
        NvccCcbin            = $env:NVCC_CCBIN
        CompatibilityWarning = $null
        Notes                = @()
    }

    $majorVersion = 0
    if ($result.SelectedVersion -match '^v(?<major>\d+)\.(?<minor>\d+)$') {
        $majorVersion = [int]$Matches.major
    }
    if ($majorVersion -lt 13) {
        $warn = "CUDA $($result.SelectedVersion) selected as fallback. Preferred default for this repository is CUDA v13.1."
        $result.CompatibilityWarning = $warn
        $result.Notes += $warn
        if (-not $Quiet) {
            Write-Warning $warn
        }
    }

    if (-not $Quiet) {
        Write-Host "CUDA detected at: $selected" -ForegroundColor Green
        Write-Host "  nvcc:                 $nvccPath" -ForegroundColor DarkGray
        Write-Host "  cicc:                 $ciccPath" -ForegroundColor DarkGray
        Write-Host "  CMAKE_CUDA_COMPILER:  $nvccPath" -ForegroundColor DarkGray
        Write-Host "  NVCC_CCBIN:           $($env:NVCC_CCBIN)" -ForegroundColor DarkGray
        Write-Host "  CUDA_DEVICE_ORDER:    $($env:CUDA_DEVICE_ORDER)" -ForegroundColor DarkGray
        Write-Host "  CUDAARCHS:            $($env:CUDAARCHS)" -ForegroundColor DarkGray
    }

    return $result
}

if ($MyInvocation.InvocationName -ne '.') {
    Initialize-CudaEnvironment -PreferredVersions $PreferredVersions -CudaPath $CudaPath -Quiet:$Quiet -InstallMachineGlobal:$InstallMachineGlobal -WorkaroundMsvc1944:$WorkaroundMsvc1944
}
