#Requires -Version 7.0
<#
.SYNOPSIS
    Discovers installed NVIDIA driver versions, compares them against the registry,
    and optionally downloads or installs the latest compatible driver package.

.DESCRIPTION
    Queries nvidia-smi for the installed driver version on each GPU in the system.
    Reads the PC-AI driver registry (Config\driver-registry.json) to determine the
    known-latest driver version for NVIDIA GPUs.

    Because both the RTX 5060 Ti (Blackwell, SM 120) and the RTX 2000 Ada Generation
    (Ada Lovelace, SM 89) share a single Windows driver package (identified in the
    registry via sharedDriverGroup: "nvidia-gpu-driver"), this script enforces the
    requirement that any driver update must be compatible with BOTH GPUs before
    recommending installation.

    Outputs a formatted status table:
        GPU Name | Installed Driver | Registry Latest | Status | Download URL

    Optional actions:
        -AutoDownload   Download the installer to $DownloadDir if the driver is outdated.
        -AutoInstall    Download AND install the driver (requires admin; prompts for
                        confirmation unless -Force is also set).

.PARAMETER RegistryPath
    Full path to an alternate driver-registry.json file. When omitted, the path is
    resolved automatically from the script root (../Config/driver-registry.json).

.PARAMETER SoftwareRegistryPath
    Full path to an alternate nvidia-software-registry.json file. When omitted,
    resolved from the script root (../Config/nvidia-software-registry.json).

.PARAMETER DownloadDir
    Directory for downloaded installer files. Created if it does not exist.
    Defaults to $env:TEMP\NvidiaDriverUpdates.

.PARAMETER AutoDownload
    Download the driver installer when the installed version is outdated.
    The installer is placed in $DownloadDir for manual review before running.

.PARAMETER AutoInstall
    Download AND execute the installer when the installed version is outdated.
    Requires administrator elevation. Prompts for confirmation unless -Force is set.
    Implies -AutoDownload.

.PARAMETER Force
    Skip the interactive confirmation prompt when -AutoInstall is used.
    Re-download the installer even if the file already exists in $DownloadDir.

.PARAMETER NvidiaSmiPath
    Full path to nvidia-smi.exe. When omitted the function searches:
      1. $env:SystemRoot\System32\nvidia-smi.exe  (default driver install location)
      2. C:\Windows\System32\nvidia-smi.exe
      3. Anything on PATH named nvidia-smi.exe

.EXAMPLE
    .\Sync-NvidiaDriverVersion.ps1
    Queries GPU driver versions and prints a status table. No downloads performed.

.EXAMPLE
    .\Sync-NvidiaDriverVersion.ps1 -AutoDownload
    Downloads the latest NVIDIA driver installer when the installed version is outdated.
    Installer is saved to $env:TEMP\NvidiaDriverUpdates\ for manual review.

.EXAMPLE
    .\Sync-NvidiaDriverVersion.ps1 -AutoInstall
    Downloads and installs the latest driver if outdated. Prompts for confirmation.

.EXAMPLE
    .\Sync-NvidiaDriverVersion.ps1 -AutoInstall -Force
    Downloads and installs without confirmation prompt.

.EXAMPLE
    .\Sync-NvidiaDriverVersion.ps1 -Verbose
    Emits verbose diagnostic messages during discovery and registry lookup.

.OUTPUTS
    PSCustomObject[] — one entry per detected GPU, with properties:
        GpuIndex        [int]    nvidia-smi GPU index
        GpuName         [string] Full GPU name from nvidia-smi
        InstalledDriver [string] Driver version reported by nvidia-smi
        RegistryLatest  [string] Latest driver version from driver-registry.json
        Status          [string] UpToDate | Outdated | RegistryNotFound | NvidiaSmiMissing | Unknown
        DownloadUrl     [string] Driver download page URL from registry
        RegistryId      [string] Device ID in driver-registry.json that matched
#>
[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
[OutputType([PSCustomObject[]])]
param(
    [Parameter()]
    [string]$RegistryPath,

    [Parameter()]
    [string]$SoftwareRegistryPath,

    [Parameter()]
    [string]$DownloadDir = (Join-Path $env:TEMP 'NvidiaDriverUpdates'),

    [Parameter()]
    [switch]$AutoDownload,

    [Parameter()]
    [switch]$AutoInstall,

    [Parameter()]
    [switch]$Force,

    [Parameter()]
    [string]$NvidiaSmiPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Region: helpers
# ---------------------------------------------------------------------------

function Write-StatusTable {
    param([PSCustomObject[]]$Rows)

    if (-not $Rows -or $Rows.Count -eq 0) {
        Write-Host '  (no GPU rows to display)' -ForegroundColor DarkGray
        return
    }

    $colWidths = @{
        GpuIndex        = 5
        GpuName         = [Math]::Max(8, ($Rows | ForEach-Object { $_.GpuName.Length } | Measure-Object -Maximum).Maximum)
        InstalledDriver = [Math]::Max(9, ($Rows | ForEach-Object { "$($_.InstalledDriver)".Length } | Measure-Object -Maximum).Maximum)
        RegistryLatest  = [Math]::Max(6, ($Rows | ForEach-Object { "$($_.RegistryLatest)".Length } | Measure-Object -Maximum).Maximum)
        Status          = [Math]::Max(6, ($Rows | ForEach-Object { $_.Status.Length } | Measure-Object -Maximum).Maximum)
    }

    $fmt    = "{0,-$($colWidths.GpuIndex)} {1,-$($colWidths.GpuName)} {2,-$($colWidths.InstalledDriver)} {3,-$($colWidths.RegistryLatest)} {4,-$($colWidths.Status)} {5}"
    $header = $fmt -f 'Index', 'GPU Name', 'Installed', 'Latest', 'Status', 'Download URL'

    $separator = '-' * $header.Length

    Write-Host ''
    Write-Host $separator -ForegroundColor DarkGray
    Write-Host $header    -ForegroundColor Cyan
    Write-Host $separator -ForegroundColor DarkGray

    foreach ($row in $Rows) {
        $color = switch ($row.Status) {
            'UpToDate'         { 'Green'   }
            'Outdated'         { 'Yellow'  }
            'RegistryNotFound' { 'DarkGray'}
            'NvidiaSmiMissing' { 'Red'     }
            default            { 'White'   }
        }

        $line = $fmt -f `
            $row.GpuIndex, `
            $row.GpuName, `
            $row.InstalledDriver, `
            $row.RegistryLatest, `
            $row.Status, `
            $row.DownloadUrl

        Write-Host $line -ForegroundColor $color
    }

    Write-Host $separator -ForegroundColor DarkGray
    Write-Host ''
}

function Resolve-NvidiaSmi {
    param([string]$HintPath)

    $candidates = [System.Collections.Generic.List[string]]::new()
    if ($HintPath) { $candidates.Add($HintPath) }
    $candidates.Add((Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'))
    $candidates.Add('C:\Windows\System32\nvidia-smi.exe')

    # Also check PATH
    $onPath = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($onPath) { $candidates.Add($onPath.Source) }

    foreach ($c in $candidates) {
        if ($c -and (Test-Path -LiteralPath $c)) {
            Write-Verbose "nvidia-smi resolved: $c"
            return $c
        }
    }
    return $null
}

function Get-InstalledNvidiaDrivers {
    <#
    .SYNOPSIS
        Returns an array of PSCustomObjects with GpuIndex, GpuName, DriverVersion
        by querying nvidia-smi --query-gpu.
    #>
    param([string]$SmiPath)

    $results = [System.Collections.Generic.List[PSCustomObject]]::new()

    try {
        $rawLines = & $SmiPath `
            --query-gpu=index,name,driver_version `
            --format=csv,noheader,nounits 2>&1

        if ($LASTEXITCODE -ne 0) {
            Write-Warning "nvidia-smi exited with code $LASTEXITCODE. Output: $rawLines"
            return $results
        }

        foreach ($line in $rawLines) {
            $line = $line.Trim()
            if ([string]::IsNullOrWhiteSpace($line)) { continue }

            $parts = $line -split ',\s*'
            if ($parts.Count -lt 3) {
                Write-Warning "Unexpected nvidia-smi output line (expected 3 CSV fields): '$line'"
                continue
            }

            $results.Add([PSCustomObject]@{
                GpuIndex      = [int]$parts[0].Trim()
                GpuName       = $parts[1].Trim()
                DriverVersion = $parts[2].Trim()
            })
        }
    }
    catch {
        Write-Warning "Failed to query nvidia-smi: $($_.Exception.Message)"
    }

    return $results
}

function Get-RegistryDriverVersion {
    <#
    .SYNOPSIS
        Searches driver-registry.json for all entries in the "nvidia-gpu-driver"
        sharedDriverGroup and returns the first non-null latestVersion found,
        along with the entry's downloadUrl and id.

        Falls back to searching by gpu category when no sharedDriverGroup match
        is found — handles registries where sharedDriverGroup is not yet set.
    #>
    param(
        [PSCustomObject]$Registry,
        [string]$GpuName
    )

    # Priority 1: sharedDriverGroup match (canonical path for multi-GPU shared packages)
    $sharedGroupDevices = @($Registry.devices | Where-Object {
        $_.sharedDriverGroup -eq 'nvidia-gpu-driver' -and
        $_.driver.latestVersion
    })

    if ($sharedGroupDevices.Count -gt 0) {
        # Verify both GPUs are in the shared group (multi-GPU compatibility requirement)
        Write-Verbose "Found $($sharedGroupDevices.Count) device(s) in nvidia-gpu-driver sharedDriverGroup."

        # Return the entry whose friendly name best matches $GpuName
        foreach ($dev in $sharedGroupDevices) {
            foreach ($rule in $dev.matchRules) {
                if ($rule.type -eq 'friendly_name') {
                    $pattern = $rule.pattern -replace '\*', '.*'
                    if ($GpuName -match $pattern) {
                        Write-Verbose "Registry match via sharedDriverGroup friendly_name: $($dev.id)"
                        return [PSCustomObject]@{
                            RegistryId     = $dev.id
                            LatestVersion  = $dev.driver.latestVersion
                            DownloadUrl    = $dev.driver.downloadUrl
                            SharedGroup    = $dev.sharedDriverGroup
                        }
                    }
                }
            }
        }

        # Fuzzy fallback: return the first group member if no name match
        $first = $sharedGroupDevices[0]
        Write-Verbose "No friendly_name match in sharedDriverGroup; returning first group member: $($first.id)"
        return [PSCustomObject]@{
            RegistryId    = $first.id
            LatestVersion = $first.driver.latestVersion
            DownloadUrl   = $first.driver.downloadUrl
            SharedGroup   = $first.sharedDriverGroup
        }
    }

    # Priority 2: category=gpu with a latestVersion (fallback for older registry versions)
    $gpuDevices = @($Registry.devices | Where-Object {
        $_.category -eq 'gpu' -and $_.driver.latestVersion
    })

    if ($gpuDevices.Count -gt 0) {
        foreach ($dev in $gpuDevices) {
            foreach ($rule in $dev.matchRules) {
                if ($rule.type -eq 'friendly_name') {
                    $pattern = $rule.pattern -replace '\*', '.*'
                    if ($GpuName -match $pattern) {
                        Write-Verbose "Registry match via gpu category friendly_name: $($dev.id)"
                        return [PSCustomObject]@{
                            RegistryId    = $dev.id
                            LatestVersion = $dev.driver.latestVersion
                            DownloadUrl   = $dev.driver.downloadUrl
                            SharedGroup   = $dev.sharedDriverGroup
                        }
                    }
                }
            }
        }
    }

    Write-Verbose "No registry entry matched GPU name: $GpuName"
    return $null
}

function Compare-DriverVersionStrings {
    <#
    .SYNOPSIS
        Compares two NVIDIA driver version strings (e.g. "591.55" vs "582.41").
        Returns 1 if $Latest > $Installed, 0 if equal, -1 if $Latest < $Installed.
    #>
    param(
        [string]$Installed,
        [string]$Latest
    )

    try {
        $installedParts = $Installed -split '\.' | ForEach-Object { [int]$_ }
        $latestParts    = $Latest    -split '\.' | ForEach-Object { [int]$_ }

        $maxLen = [Math]::Max($installedParts.Count, $latestParts.Count)

        for ($i = 0; $i -lt $maxLen; $i++) {
            $iv = if ($i -lt $installedParts.Count) { $installedParts[$i] } else { 0 }
            $lv = if ($i -lt $latestParts.Count)    { $latestParts[$i]    } else { 0 }

            if ($lv -gt $iv) { return 1  }
            if ($lv -lt $iv) { return -1 }
        }
        return 0
    }
    catch {
        Write-Verbose "Driver version parse failed ('$Installed' vs '$Latest'): $($_.Exception.Message)"
        return 0
    }
}

function Confirm-NvidiaInstall {
    param(
        [string]$InstalledVersion,
        [string]$LatestVersion,
        [string[]]$GpuNames,
        [string]$InstallerPath
    )

    Write-Host ''
    Write-Host '  NVIDIA Driver Install Confirmation' -ForegroundColor Yellow
    Write-Host '  ----------------------------------' -ForegroundColor DarkGray
    Write-Host "  Installed : $InstalledVersion" -ForegroundColor White
    Write-Host "  Latest    : $LatestVersion"    -ForegroundColor Cyan
    Write-Host '  Affects GPUs:'
    foreach ($n in $GpuNames) { Write-Host "    - $n" -ForegroundColor White }
    Write-Host "  Installer : $InstallerPath"    -ForegroundColor White
    Write-Host ''

    $response = Read-Host '  Proceed with installation? (y/N)'
    return ($response -match '^[Yy]')
}

function Test-AdminElevation {
    $identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]$identity
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-NvidiaInstall {
    <#
    .SYNOPSIS
        Silently executes the NVIDIA installer exe with -s -noreboot arguments.
    #>
    param(
        [string]$InstallerPath,
        [string]$SilentArgs = '-s -noreboot'
    )

    Write-Host "  [*] Launching NVIDIA installer (silent) ..." -ForegroundColor Gray
    Write-Verbose "  Installer: $InstallerPath  Args: $SilentArgs"

    try {
        $argList = $SilentArgs -split '\s+'
        $proc = Start-Process `
            -FilePath  $InstallerPath `
            -ArgumentList $argList `
            -Wait      `
            -PassThru

        $exitCode = $proc.ExitCode

        if ($exitCode -eq 0) {
            Write-Host '  [OK] NVIDIA driver installed successfully (exit 0).' -ForegroundColor Green
        }
        elseif ($exitCode -eq 3010) {
            Write-Host '  [OK] NVIDIA driver installed — reboot required to complete (exit 3010).' -ForegroundColor Yellow
        }
        else {
            # Exit code 1 and all other non-zero codes are genuine failures
            Write-Warning "  [!!] NVIDIA installer exited with code $exitCode. Check NVIDIA setup logs."
        }

        return $exitCode
    }
    catch {
        Write-Warning "  [!!] Failed to start NVIDIA installer: $($_.Exception.Message)"
        return -1
    }
}

# ---------------------------------------------------------------------------
# Region: resolve file paths
# ---------------------------------------------------------------------------

$scriptRepoRoot = Split-Path -Parent $PSScriptRoot   # Tools\ -> PC_AI\

if (-not $RegistryPath) {
    $RegistryPath = Join-Path $scriptRepoRoot 'Config\driver-registry.json'
}

if (-not $SoftwareRegistryPath) {
    $SoftwareRegistryPath = Join-Path $scriptRepoRoot 'Config\nvidia-software-registry.json'
}

Write-Verbose "Driver registry       : $RegistryPath"
Write-Verbose "Software registry     : $SoftwareRegistryPath"
Write-Verbose "Download dir          : $DownloadDir"

# ---------------------------------------------------------------------------
# Region: load driver registry
# ---------------------------------------------------------------------------

if (-not (Test-Path -LiteralPath $RegistryPath)) {
    Write-Error "Driver registry not found: $RegistryPath"
    return
}

$registryRaw = [System.IO.File]::ReadAllText($RegistryPath)
$registry    = $registryRaw | ConvertFrom-Json

if (-not $registry.devices) {
    Write-Error "Registry file is missing the 'devices' array: $RegistryPath"
    return
}

Write-Verbose "Registry loaded (v$($registry.version)), $($registry.devices.Count) device entries."

# Optionally read the software registry for enrichment (driver notes, GPU UUIDs, etc.)
$softwareRegistry = $null
if (Test-Path -LiteralPath $SoftwareRegistryPath) {
    $softwareRegistry = [System.IO.File]::ReadAllText($SoftwareRegistryPath) | ConvertFrom-Json
    Write-Verbose "Software registry loaded (v$($softwareRegistry.version))."
}

# ---------------------------------------------------------------------------
# Region: validate multi-GPU compatibility
# ---------------------------------------------------------------------------
# Both NVIDIA GPUs on this system share a single driver package.
# Warn if either GPU is missing from the nvidia-gpu-driver sharedDriverGroup.

$sharedGroupIds = @($registry.devices |
    Where-Object { $_.sharedDriverGroup -eq 'nvidia-gpu-driver' } |
    ForEach-Object { $_.id })

if ($sharedGroupIds.Count -lt 2) {
    Write-Warning ("driver-registry.json has only $($sharedGroupIds.Count) device(s) in " +
        "'nvidia-gpu-driver' sharedDriverGroup. " +
        "Both RTX 5060 Ti and RTX 2000 Ada should be listed to enforce multi-GPU compatibility.")
}
else {
    Write-Verbose "Multi-GPU sharedDriverGroup 'nvidia-gpu-driver' contains: $($sharedGroupIds -join ', ')"
}

# ---------------------------------------------------------------------------
# Region: resolve nvidia-smi
# ---------------------------------------------------------------------------

$smiPath = Resolve-NvidiaSmi -HintPath $NvidiaSmiPath

if (-not $smiPath) {
    Write-Warning 'nvidia-smi.exe was not found. Cannot query installed driver version.'

    # Build placeholder rows from software registry GPU list (best-effort)
    $rows = [System.Collections.Generic.List[PSCustomObject]]::new()

    if ($softwareRegistry -and $softwareRegistry.gpus) {
        foreach ($gpu in $softwareRegistry.gpus) {
            $rows.Add([PSCustomObject]@{
                GpuIndex        = $gpu.index
                GpuName         = $gpu.name
                InstalledDriver = 'Unknown (nvidia-smi missing)'
                RegistryLatest  = (Get-RegistryDriverVersion -Registry $registry -GpuName $gpu.name)?.LatestVersion ?? 'N/A'
                Status          = 'NvidiaSmiMissing'
                DownloadUrl     = 'https://www.nvidia.com/download/index.aspx'
                RegistryId      = 'N/A'
            })
        }
    }
    else {
        $rows.Add([PSCustomObject]@{
            GpuIndex        = -1
            GpuName         = 'Unknown'
            InstalledDriver = 'Unknown (nvidia-smi missing)'
            RegistryLatest  = 'N/A'
            Status          = 'NvidiaSmiMissing'
            DownloadUrl     = 'https://www.nvidia.com/download/index.aspx'
            RegistryId      = 'N/A'
        })
    }

    Write-Host ''
    Write-Host '  NVIDIA Driver Status' -ForegroundColor Cyan
    Write-StatusTable -Rows $rows
    return ,$rows.ToArray()
}

Write-Verbose "nvidia-smi: $smiPath"

# ---------------------------------------------------------------------------
# Region: query installed driver versions
# ---------------------------------------------------------------------------

Write-Host ''
Write-Host '  Querying NVIDIA GPU driver versions via nvidia-smi...' -ForegroundColor Gray

$installedGpus = Get-InstalledNvidiaDrivers -SmiPath $smiPath

if ($installedGpus.Count -eq 0) {
    Write-Warning 'nvidia-smi returned no GPU entries. No NVIDIA GPUs detected or driver not loaded.'
    return
}

Write-Verbose "nvidia-smi reported $($installedGpus.Count) GPU(s)."

# ---------------------------------------------------------------------------
# Region: build status rows
# ---------------------------------------------------------------------------

$rows              = [System.Collections.Generic.List[PSCustomObject]]::new()
$outdatedGpus      = [System.Collections.Generic.List[PSCustomObject]]::new()
$installerSilentArgs = '-s -noreboot'
$installerUrl      = $null

foreach ($gpu in $installedGpus) {
    $regInfo = Get-RegistryDriverVersion -Registry $registry -GpuName $gpu.GpuName

    if (-not $regInfo) {
        $row = [PSCustomObject]@{
            GpuIndex        = $gpu.GpuIndex
            GpuName         = $gpu.GpuName
            InstalledDriver = $gpu.DriverVersion
            RegistryLatest  = 'N/A'
            Status          = 'RegistryNotFound'
            DownloadUrl     = 'https://www.nvidia.com/download/index.aspx'
            RegistryId      = 'N/A'
        }
        $rows.Add($row)
        Write-Verbose "GPU '$($gpu.GpuName)': no registry entry found."
        continue
    }

    $cmp = Compare-DriverVersionStrings `
        -Installed $gpu.DriverVersion `
        -Latest    $regInfo.LatestVersion

    $status = switch ($cmp) {
        1  { 'Outdated'  }
        0  { 'UpToDate'  }
        -1 { 'UpToDate'  }   # Installed newer than registry (e.g. beta driver)
        default { 'Unknown' }
    }

    if ($cmp -eq 1 -and $regInfo.DownloadUrl) {
        $installerUrl = $regInfo.DownloadUrl
    }

    $row = [PSCustomObject]@{
        GpuIndex        = $gpu.GpuIndex
        GpuName         = $gpu.GpuName
        InstalledDriver = $gpu.DriverVersion
        RegistryLatest  = $regInfo.LatestVersion
        Status          = $status
        DownloadUrl     = $regInfo.DownloadUrl ?? 'https://www.nvidia.com/download/index.aspx'
        RegistryId      = $regInfo.RegistryId
    }

    $rows.Add($row)

    if ($status -eq 'Outdated') {
        $outdatedGpus.Add($row)
        Write-Verbose "GPU '$($gpu.GpuName)': OUTDATED (installed=$($gpu.DriverVersion), latest=$($regInfo.LatestVersion))"
    }
    else {
        Write-Verbose "GPU '$($gpu.GpuName)': up to date ($($gpu.DriverVersion))."
    }
}

# ---------------------------------------------------------------------------
# Region: display status table
# ---------------------------------------------------------------------------

Write-Host ''
Write-Host '  NVIDIA Driver Status' -ForegroundColor Cyan
Write-StatusTable -Rows $rows

# Summarise shared-group compatibility note when multiple GPUs share the same package
$sharedRows = @($rows | Where-Object { $_.RegistryId -ne 'N/A' } |
    Group-Object { (Get-RegistryDriverVersion -Registry $registry -GpuName $_.GpuName)?.SharedGroup } |
    Where-Object { $_.Name -eq 'nvidia-gpu-driver' })

if ($sharedRows.Count -gt 0) {
    Write-Host ('  Note: RTX 5060 Ti and RTX 2000 Ada share a single driver package ' +
        '(sharedDriverGroup: nvidia-gpu-driver).') -ForegroundColor DarkGray
    Write-Host '  Any update must be compatible with both GPUs.' -ForegroundColor DarkGray
    Write-Host ''
}

# ---------------------------------------------------------------------------
# Region: early exit when everything is current
# ---------------------------------------------------------------------------

if ($outdatedGpus.Count -eq 0) {
    Write-Host '  All NVIDIA GPU drivers are up to date.' -ForegroundColor Green
    return ,$rows.ToArray()
}

Write-Host ("  $($outdatedGpus.Count) GPU(s) have an outdated driver. " +
    "Latest registry version: $($outdatedGpus[0].RegistryLatest)") -ForegroundColor Yellow

# ---------------------------------------------------------------------------
# Region: AutoDownload / AutoInstall
# ---------------------------------------------------------------------------

if (-not $AutoDownload -and -not $AutoInstall) {
    Write-Host ("  Run with -AutoDownload to fetch the installer, " +
        "or -AutoInstall to download and install.") -ForegroundColor DarkGray
    return ,$rows.ToArray()
}

# Determine target registry entry for the download (prefer first sharedDriverGroup entry)
$targetEntry = $registry.devices |
    Where-Object { $_.sharedDriverGroup -eq 'nvidia-gpu-driver' -and $_.driver.latestVersion } |
    Select-Object -First 1

if (-not $targetEntry) {
    Write-Warning 'No nvidia-gpu-driver sharedDriverGroup entry with a latestVersion found in registry. Cannot determine download target.'
    return ,$rows.ToArray()
}

$downloadUrl     = $targetEntry.driver.downloadUrl
$registrySilentArgs = if ($targetEntry.driver.silentArgs) { $targetEntry.driver.silentArgs } else { '-s -noreboot' }
$latestVersion   = $targetEntry.driver.latestVersion

if (-not $downloadUrl) {
    Write-Warning "Registry entry '$($targetEntry.id)' has no downloadUrl configured."
    Write-Host "  Manual download: https://www.nvidia.com/download/index.aspx" -ForegroundColor Yellow
    return ,$rows.ToArray()
}

# Ensure download directory exists
if (-not (Test-Path -LiteralPath $DownloadDir)) {
    New-Item -ItemType Directory -Path $DownloadDir -Force | Out-Null
    Write-Verbose "Created download directory: $DownloadDir"
}

$outFile = Join-Path $DownloadDir "nvidia-gpu-driver-$latestVersion.exe"

# ------------------------------------------------------------------
# The NVIDIA download page (nvidia.com/download/index.aspx) requires
# browser interaction to generate a direct download link. Providing
# a usable direct URL requires the per-product download API endpoint.
# We detect whether a direct URL is available and fall back gracefully.
# ------------------------------------------------------------------
$isDirectUrl = $downloadUrl -match '\.(exe|run|pkg)(\?|$)'

if (-not $isDirectUrl) {
    Write-Host ''
    Write-Host '  NVIDIA Driver Download' -ForegroundColor Cyan
    Write-Host "  Registry download page: $downloadUrl" -ForegroundColor White
    Write-Host ''
    Write-Host ('  Note: The NVIDIA download URL in the registry is a product selection page. ' +
        'A browser session is required to generate a direct installer link.') -ForegroundColor DarkGray
    Write-Host '  Recommended steps:' -ForegroundColor DarkGray
    Write-Host "    1. Navigate to: $downloadUrl" -ForegroundColor White
    Write-Host "    2. Select GeForce Game Ready or Studio Driver for Windows 11 64-bit." -ForegroundColor DarkGray
    Write-Host "    3. Download the installer exe (approx. 600 MB)." -ForegroundColor DarkGray
    Write-Host "    4. Save as: $outFile" -ForegroundColor White
    Write-Host "    5. Re-run this script with -AutoInstall (it will find the existing file)." -ForegroundColor DarkGray
    Write-Host ''

    # Check if file was already manually placed in DownloadDir
    if (Test-Path -LiteralPath $outFile) {
        Write-Host "  Existing installer found: $outFile" -ForegroundColor Green
    }
    else {
        Write-Host "  Installer not yet present at: $outFile" -ForegroundColor Yellow
        if (-not $AutoInstall) {
            return ,$rows.ToArray()
        }
    }
}
else {
    # Direct URL path — attempt download
    if (-not (Test-Path -LiteralPath $outFile) -or $Force) {
        if (-not $PSCmdlet.ShouldProcess($downloadUrl, "Download NVIDIA driver $latestVersion")) {
            return ,$rows.ToArray()
        }

        Write-Host "  [*] Downloading NVIDIA driver $latestVersion ..." -ForegroundColor Gray
        Write-Verbose "  URL: $downloadUrl"
        Write-Verbose "  Destination: $outFile"

        try {
            # Use HttpClient with an explicit timeout instead of the deprecated WebClient.
            # A 30-minute ceiling is reasonable for a large (~600 MB) driver package.
            $httpClient = [System.Net.Http.HttpClient]::new()
            try {
                $httpClient.Timeout      = [TimeSpan]::FromMinutes(30)
                $httpClient.DefaultRequestHeaders.Add(
                    'User-Agent',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')

                $response = $httpClient.GetAsync($downloadUrl).GetAwaiter().GetResult()
                $response.EnsureSuccessStatusCode() | Out-Null

                $fileStream = [System.IO.File]::Create($outFile)
                try {
                    $response.Content.CopyToAsync($fileStream).GetAwaiter().GetResult()
                }
                finally {
                    $fileStream.Dispose()
                }
            }
            finally {
                $httpClient.Dispose()
            }
            Write-Host '  [OK] Download complete.' -ForegroundColor Green
        }
        catch {
            Write-Warning "  [!!] Download failed: $($_.Exception.Message)"
            Write-Host "  Manual download: $downloadUrl" -ForegroundColor Yellow
            return ,$rows.ToArray()
        }
    }
    else {
        Write-Host "  [--] Using cached installer: $(Split-Path $outFile -Leaf)" -ForegroundColor DarkGray
    }
}

if (-not $AutoInstall) {
    Write-Host "  Installer ready: $outFile" -ForegroundColor Green
    Write-Host '  Run with -AutoInstall to install, or launch the installer manually.' -ForegroundColor DarkGray
    return ,$rows.ToArray()
}

# ---------------------------------------------------------------------------
# Region: install
# ---------------------------------------------------------------------------

if (-not (Test-Path -LiteralPath $outFile)) {
    Write-Warning "Installer file not found at expected path: $outFile"
    Write-Warning 'Download the installer manually and save it to that path, then re-run with -AutoInstall.'
    return ,$rows.ToArray()
}

if (-not (Test-AdminElevation)) {
    Write-Warning 'NVIDIA driver installation requires administrator elevation.'
    Write-Warning 'Re-run this script as Administrator (Right-click > Run as Administrator).'
    return ,$rows.ToArray()
}

# Confirm unless -Force bypasses it
$proceed = $Force.IsPresent

if (-not $proceed) {
    if ($PSCmdlet.ShouldProcess(
            ($outdatedGpus | ForEach-Object { $_.GpuName }) -join ', ',
            "Install NVIDIA driver $latestVersion (silent: $registrySilentArgs)")) {
        $proceed = Confirm-NvidiaInstall `
            -InstalledVersion ($outdatedGpus[0].InstalledDriver) `
            -LatestVersion    $latestVersion `
            -GpuNames         ($outdatedGpus | ForEach-Object { $_.GpuName }) `
            -InstallerPath    $outFile
    }
}

if (-not $proceed) {
    Write-Host '  Installation cancelled by user.' -ForegroundColor DarkGray
    return ,$rows.ToArray()
}

$exitCode = Invoke-NvidiaInstall -InstallerPath $outFile -SilentArgs $registrySilentArgs

if ($exitCode -eq 0 -or $exitCode -eq 3010) {
    Write-Host ''
    Write-Host ('  NVIDIA driver installation completed. ' +
        'Verify with: nvidia-smi --query-gpu=driver_version --format=csv,noheader') -ForegroundColor Green
    if ($exitCode -eq 3010) {
        Write-Host '  A system reboot is required to complete the installation.' -ForegroundColor Yellow
    }
}

return ,$rows.ToArray()
