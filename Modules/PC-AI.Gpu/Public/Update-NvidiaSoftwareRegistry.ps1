#Requires -Version 5.1
<#
.SYNOPSIS
    Updates the local nvidia-software-registry.json with patched component
    entries or with versions auto-detected from the running system.

.DESCRIPTION
    Provides two mutually complementary update modes:

    PATCH MODE (supply -ComponentId with one or more of -LatestVersion,
    -DownloadUrl, -Sha256):
        Locates the single registry entry whose "id" matches -ComponentId and
        updates only the supplied fields, leaving all other fields untouched.

    REFRESH-FROM-SYSTEM MODE (-RefreshFromSystem):
        Re-runs Get-NvidiaSoftwareStatus to detect the currently installed
        version of every component, then updates the "installedVersion" field
        in the registry for each component that reports a non-null detected
        version. Does not modify latestVersion, downloadUrl, or sha256.

    Both modes:
      - Write a timestamped backup of the original registry to
        .pcai\nvidia-backup\ before overwriting.
      - Validate that the modified JSON round-trips correctly (ConvertTo-Json
        -> ConvertFrom-Json) before writing to disk.
      - Return a summary object listing every changed field.

    Supports -WhatIf to preview changes without writing to disk.
    -ComponentId and -RefreshFromSystem may be combined: the refresh then
    limits itself to the single specified component.

.PARAMETER ComponentId
    Registry component ID to patch (e.g. 'cuda-toolkit', 'gpu-driver').
    Required in patch mode; optional in refresh-from-system mode.

.PARAMETER LatestVersion
    New value for the "latestVersion" field of the matched component.

.PARAMETER DownloadUrl
    New value for the "downloadUrl" field of the matched component.

.PARAMETER Sha256
    New value for the "sha256" field of the matched component.

.PARAMETER RefreshFromSystem
    Re-detect installed versions from the live system and update
    "installedVersion" fields in the registry.

.PARAMETER RegistryPath
    Full path to an alternate registry JSON file. When omitted the default
    Config\nvidia-software-registry.json is resolved from the module root.

.OUTPUTS
    [PSCustomObject] with properties:
        RegistryPath     - Path of the registry file that was updated.
        BackupPath       - Path of the backup taken before the update.
        ChangesApplied   - [int] number of field-level changes written.
        Changes          - [PSCustomObject[]] per-change detail records
                           (ComponentId, Field, OldValue, NewValue).
        Success          - $true when the file was written successfully.
        WhatIf           - $true when the function ran in -WhatIf mode.

.EXAMPLE
    Update-NvidiaSoftwareRegistry -ComponentId 'gpu-driver' -LatestVersion '591.55'
    Updates the latestVersion for the gpu-driver entry.

.EXAMPLE
    Update-NvidiaSoftwareRegistry -ComponentId 'cuda-toolkit' `
        -LatestVersion '13.3' `
        -DownloadUrl   'https://developer.nvidia.com/cuda-13-3-0-download-archive'
    Patches both latestVersion and downloadUrl in a single call.

.EXAMPLE
    Update-NvidiaSoftwareRegistry -RefreshFromSystem
    Detects all installed component versions and updates installedVersion fields.

.EXAMPLE
    Update-NvidiaSoftwareRegistry -RefreshFromSystem -ComponentId 'cudnn'
    Refreshes only the cuDNN installedVersion from the live system.

.EXAMPLE
    Update-NvidiaSoftwareRegistry -ComponentId 'tensorrt' -LatestVersion '10.10.0' -WhatIf
    Previews the change without writing to disk.

.NOTES
    Phase 3 implementation.
    Backup files follow the naming pattern:
        .pcai\nvidia-backup\nvidia-software-registry-<yyyyMMdd-HHmmss>.json
    The "lastUpdated" and "detectionTimestamp" fields are also refreshed to
    the current UTC ISO-8601 timestamp whenever the file is written.
#>
function Update-NvidiaSoftwareRegistry {
    [CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'Medium')]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$ComponentId,

        [Parameter()]
        [string]$LatestVersion,

        [Parameter()]
        [string]$DownloadUrl,

        [Parameter()]
        [string]$Sha256,

        [Parameter()]
        [switch]$RefreshFromSystem,

        [Parameter()]
        [string]$RegistryPath
    )

    $ErrorActionPreference = 'Stop'

    # --- Build result skeleton ---
    $result = [PSCustomObject]@{
        RegistryPath   = $null
        BackupPath     = $null
        ChangesApplied = 0
        Changes        = @()
        Success        = $false
        WhatIf         = $false
    }

    # -------------------------------------------------------------------------
    # Validate that the caller supplied at least one meaningful action
    # -------------------------------------------------------------------------
    $patchMode   = $ComponentId -and ($LatestVersion -or $DownloadUrl -or $Sha256)
    $refreshMode = $RefreshFromSystem.IsPresent

    if (-not $patchMode -and -not $refreshMode) {
        $msg = "Update-NvidiaSoftwareRegistry: Nothing to do. Provide -ComponentId with " +
            "at least one of -LatestVersion, -DownloadUrl, -Sha256, or supply -RefreshFromSystem."
        $PSCmdlet.WriteError((New-Object System.Management.Automation.ErrorRecord (New-Object System.ArgumentException $msg), "MissingParameters", [System.Management.Automation.ErrorCategory]::InvalidArgument, $null))
        return $result
    }

    # -------------------------------------------------------------------------
    # Resolve registry file path
    # -------------------------------------------------------------------------
    if (-not $RegistryPath) {
        # Derive PC_AI root from this script's location:
        #   Public\ -> PC-AI.Gpu\ -> Modules\ -> PC_AI\
        $modulePublicDir = $PSScriptRoot
        $moduleDir       = Split-Path $modulePublicDir -Parent
        $modulesDir      = Split-Path $moduleDir -Parent
        $pcAiRoot        = Split-Path $modulesDir -Parent
        $RegistryPath    = Join-Path $pcAiRoot 'Config\nvidia-software-registry.json'
    }

    $result.RegistryPath = $RegistryPath

    if (-not (Test-Path -LiteralPath $RegistryPath)) {
        $msg = "Registry file not found: $RegistryPath"
        $PSCmdlet.WriteError((New-Object System.Management.Automation.ErrorRecord (New-Object System.IO.FileNotFoundException $msg), "RegistryNotFound", [System.Management.Automation.ErrorCategory]::ObjectNotFound, $RegistryPath))
        return $result
    }

    # -------------------------------------------------------------------------
    # Load raw JSON text and deserialise
    # -------------------------------------------------------------------------
    Write-Verbose "Update-NvidiaSoftwareRegistry: Loading registry from '$RegistryPath'..."
    $rawJson  = [System.IO.File]::ReadAllText($RegistryPath)
    try {
        $registry = $rawJson | ConvertFrom-Json
    } catch {
        $msg = "Registry file is not valid JSON: $RegistryPath"
        $PSCmdlet.WriteError((New-Object System.Management.Automation.ErrorRecord $_.Exception, "JsonParseError", [System.Management.Automation.ErrorCategory]::ParserError, $RegistryPath))
        return $result
    }

    if ($null -eq $registry -or -not $registry.PSObject.Properties.Match('components').Count) {
        $msg = "Registry file is missing the 'components' array: $RegistryPath"
        $PSCmdlet.WriteError((New-Object System.Management.Automation.ErrorRecord (New-Object System.Exception $msg), "MissingComponentsArray", [System.Management.Automation.ErrorCategory]::InvalidData, $RegistryPath))
        return $result
    }

    if ($registry.components -isnot [array]) {
        $registry.components = @($registry.components)
    }

    # Work with a mutable copy of the components array
    $components = [System.Collections.Generic.List[object]]::new()
    foreach ($c in $registry.components) { $components.Add($c) }

    $changeList = [System.Collections.Generic.List[PSCustomObject]]::new()

    # -------------------------------------------------------------------------
    # PATCH MODE — update specific fields on a named component
    # -------------------------------------------------------------------------
    if ($patchMode) {
        $target = $null
        foreach ($c in $components) {
            if ($c.id -eq $ComponentId) { $target = $c; break }
        }

        if (-not $target) {
            $msg = "Component '$ComponentId' was not found in the registry."
            $PSCmdlet.WriteError((New-Object System.Management.Automation.ErrorRecord (New-Object System.Exception $msg), "ComponentNotFound", [System.Management.Automation.ErrorCategory]::ObjectNotFound, $ComponentId))
            return $result
        }

        Write-Verbose "Update-NvidiaSoftwareRegistry: Patch mode — component '$ComponentId'."

        if ($LatestVersion) {
            $old = $target.latestVersion
            if ($old -ne $LatestVersion) {
                $target.latestVersion = $LatestVersion
                $changeList.Add([PSCustomObject]@{
                    ComponentId = $ComponentId
                    Field       = 'latestVersion'
                    OldValue    = $old
                    NewValue    = $LatestVersion
                })
                Write-Verbose "  latestVersion: '$old' -> '$LatestVersion'"
            }
        }

        if ($DownloadUrl) {
            $old = $target.downloadUrl
            if ($old -ne $DownloadUrl) {
                $target.downloadUrl = $DownloadUrl
                $changeList.Add([PSCustomObject]@{
                    ComponentId = $ComponentId
                    Field       = 'downloadUrl'
                    OldValue    = $old
                    NewValue    = $DownloadUrl
                })
                Write-Verbose "  downloadUrl: '$old' -> '$DownloadUrl'"
            }
        }

        if ($Sha256) {
            $oldSha = $null
            if ($target.PSObject.Properties['sha256']) {
                $oldSha = $target.sha256
            }
            if ($oldSha -ne $Sha256) {
                # Add the sha256 property if it does not already exist
                if (-not $target.PSObject.Properties['sha256']) {
                    $target | Add-Member -NotePropertyName 'sha256' -NotePropertyValue $Sha256
                }
                else {
                    $target.sha256 = $Sha256
                }
                $changeList.Add([PSCustomObject]@{
                    ComponentId = $ComponentId
                    Field       = 'sha256'
                    OldValue    = $oldSha
                    NewValue    = $Sha256
                })
                Write-Verbose "  sha256: '$oldSha' -> '$Sha256'"
            }
        }
    }

    # -------------------------------------------------------------------------
    # REFRESH-FROM-SYSTEM MODE — update installedVersion from live detection
    # -------------------------------------------------------------------------
    if ($refreshMode) {
        Write-Verbose "Update-NvidiaSoftwareRegistry: Refresh-from-system mode..."

        $statusParams = @{}
        if ($RegistryPath)  { $statusParams['RegistryPath'] = $RegistryPath }
        if ($ComponentId)   { $statusParams['ComponentId']  = $ComponentId }

        $statusResults = Get-NvidiaSoftwareStatus @statusParams

        foreach ($status in $statusResults) {
            if ($null -eq $status.InstalledVersion) {
                Write-Verbose "  Skipping '$($status.ComponentId)' — no installed version detected."
                continue
            }

            $target = $null
            foreach ($c in $components) {
                if ($c.id -eq $status.ComponentId) { $target = $c; break }
            }

            if (-not $target) {
                Write-Verbose "  Component '$($status.ComponentId)' not found in registry (skipping)."
                continue
            }

            $old = $target.installedVersion
            if ($old -ne $status.InstalledVersion) {
                $target.installedVersion = $status.InstalledVersion
                $changeList.Add([PSCustomObject]@{
                    ComponentId = $status.ComponentId
                    Field       = 'installedVersion'
                    OldValue    = $old
                    NewValue    = $status.InstalledVersion
                })
                Write-Verbose "  $($status.ComponentId) installedVersion: '$old' -> '$($status.InstalledVersion)'"
            }
        }
    }

    # -------------------------------------------------------------------------
    # Nothing changed — return early
    # -------------------------------------------------------------------------
    if ($changeList.Count -eq 0) {
        Write-Verbose "Update-NvidiaSoftwareRegistry: No field values changed — registry is already up to date."
        $result.Success        = $true
        $result.Changes        = @()
        $result.ChangesApplied = 0
        return $result
    }

    # -------------------------------------------------------------------------
    # Update registry-level timestamps
    # -------------------------------------------------------------------------
    $nowIso = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    $registry.lastUpdated         = $nowIso
    $registry.detectionTimestamp  = $nowIso

    # Re-attach the mutated components list
    $registry.components = @($components)

    # -------------------------------------------------------------------------
    # Validate JSON round-trip before writing
    # -------------------------------------------------------------------------
    Write-Verbose "Update-NvidiaSoftwareRegistry: Validating JSON schema round-trip..."
    try {
        $newJson  = $registry | ConvertTo-Json -Depth 20
        $testBack = $newJson | ConvertFrom-Json
        if (-not $testBack.components) {
            throw "Round-trip validation failed: 'components' array missing after re-serialisation."
        }
    }
    catch {
        $msg = "Update-NvidiaSoftwareRegistry: JSON validation failed — registry NOT written: $($_.Exception.Message)"
        $PSCmdlet.WriteError((New-Object System.Management.Automation.ErrorRecord $_.Exception, "JsonValidationFailed", [System.Management.Automation.ErrorCategory]::InvalidData, $RegistryPath))
        return $result
    }

    # -------------------------------------------------------------------------
    # WhatIf guard
    # -------------------------------------------------------------------------
    if (-not $PSCmdlet.ShouldProcess($RegistryPath, "Write updated registry ($($changeList.Count) change(s))")) {
        $result.WhatIf         = $true
        $result.Changes        = @($changeList)
        $result.ChangesApplied = $changeList.Count
        Write-Verbose "WhatIf: Would write $($changeList.Count) change(s) to '$RegistryPath'."
        foreach ($ch in $changeList) {
            Write-Verbose "  $($ch.ComponentId).$($ch.Field): '$($ch.OldValue)' -> '$($ch.NewValue)'"
        }
        return $result
    }

    # -------------------------------------------------------------------------
    # Backup current registry
    # -------------------------------------------------------------------------
    $pcAiRoot  = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
    $backupDir = Join-Path $pcAiRoot '.pcai\nvidia-backup'
    if (-not (Test-Path -LiteralPath $backupDir)) {
        New-Item -Path $backupDir -ItemType Directory -Force | Out-Null
        Write-Verbose "Update-NvidiaSoftwareRegistry: Created backup directory: $backupDir"
    }

    $timestamp  = (Get-Date -Format 'yyyyMMdd-HHmmss')
    $backupFile = Join-Path $backupDir "nvidia-software-registry-$timestamp.json"

    Write-Verbose "Update-NvidiaSoftwareRegistry: Backing up registry to '$backupFile'..."
    [System.IO.File]::WriteAllText($backupFile, $rawJson)
    $result.BackupPath = $backupFile

    # -------------------------------------------------------------------------
    # Write updated registry (atomic: write to .tmp then rename)
    # -------------------------------------------------------------------------
    Write-Verbose "Update-NvidiaSoftwareRegistry: Writing $($changeList.Count) change(s) to '$RegistryPath'..."
    $tempPath = "$RegistryPath.tmp"
    [System.IO.File]::WriteAllText($tempPath, $newJson)
    Move-Item -Path $tempPath -Destination $RegistryPath -Force

    $result.Success        = $true
    $result.Changes        = @($changeList)
    $result.ChangesApplied = $changeList.Count

    Write-Verbose "Update-NvidiaSoftwareRegistry: Done. $($changeList.Count) field(s) updated. Backup: '$backupFile'."
    return $result
}
