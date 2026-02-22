#Requires -Version 5.1
<#
.SYNOPSIS
    Create links/junctions under Models/ for discovered model files and folders.

.DESCRIPTION
    Reads MODELS.md (from Invoke-ModelDiscovery) and creates hardlinks or symlinks
    to model files under Models\linked. Also creates junctions for HF-style model
    directories (detected via config.json).

.PARAMETER InventoryPath
    Path to MODELS.md (default: repo root MODELS.md).

.PARAMETER OutputRoot
    Root folder for links (default: Models\linked).

.PARAMETER Force
    Overwrite existing links/files.

.PARAMETER DryRun
    Show actions without creating links.
#>
[CmdletBinding()]
param(
    [string]$InventoryPath = (Join-Path (Resolve-Path .) 'MODELS.md'),
    [string]$OutputRoot = (Join-Path (Resolve-Path .) 'Models\linked'),
    [switch]$Force,
    [switch]$DryRun
)

function Get-SafeName {
    param([string]$Name)
    $safe = $Name -replace '[\\/:*?"<>|]', '_' -replace '\s+', '_'
    $safe = $safe.Trim('_')
    if (-not $safe) { $safe = 'model' }
    return $safe
}

function Get-ShortHash {
    param([string]$Path)
    try {
        $hash = Get-FileHash -Algorithm SHA1 -Path $Path -ErrorAction Stop
        return $hash.Hash.Substring(0, 8).ToLowerInvariant()
    } catch {
        return ([guid]::NewGuid().ToString('N').Substring(0, 8))
    }
}

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        if (-not $DryRun) {
            New-Item -ItemType Directory -Path $Path -Force | Out-Null
        }
    }
}

if (-not (Test-Path $InventoryPath)) {
    throw "Inventory file not found: $InventoryPath"
}

$repoRoot = Resolve-Path .
Ensure-Dir -Path $OutputRoot
$hfRoot = Join-Path $OutputRoot 'hf'
Ensure-Dir -Path $hfRoot

$lines = Get-Content -Path $InventoryPath
$rows = $lines | Where-Object { $_ -match '^\|.+\|.+\|.+\|.+\|.+\|.+\|$' -and $_ -notmatch 'Name \| Ext' -and $_ -notmatch '^\| ---' }

$entries = @()
foreach ($line in $rows) {
    $parts = $line.Trim('|').Split('|') | ForEach-Object { $_.Trim() }
    if ($parts.Count -ge 6) {
        $entries += [PSCustomObject]@{
            Name = $parts[0]
            Ext = $parts[1]
            SizeMB = $parts[2]
            Modified = $parts[3]
            Backend = $parts[4]
            Path = $parts[5]
        }
    }
}

if (-not $entries -or $entries.Count -eq 0) {
    Write-Host "No model entries parsed from $InventoryPath"
    return
}

$outputRootDrive = [System.IO.Path]::GetPathRoot((Resolve-Path $OutputRoot).Path)
$created = 0
$skipped = 0
$failed = 0
$hfDirs = New-Object System.Collections.Generic.HashSet[string]

foreach ($entry in $entries) {
    $path = $entry.Path
    if (-not $path -or -not (Test-Path $path)) {
        $skipped++
        continue
    }

    $fileName = [System.IO.Path]::GetFileName($path)
    $ext = [System.IO.Path]::GetExtension($path)
    $safeBase = Get-SafeName -Name ($entry.Name)
    if (-not $ext) {
        $ext = [System.IO.Path]::GetExtension($fileName)
    }
    $destFile = if ($ext) { "$safeBase$ext" } else { $safeBase }
    $destPath = Join-Path $OutputRoot $destFile

    if (Test-Path $destPath) {
        if ($Force) {
            if (-not $DryRun) { Remove-Item -Path $destPath -Force -ErrorAction SilentlyContinue }
        } else {
            $skipped++
            continue
        }
    }

    $sameDrive = ([System.IO.Path]::GetPathRoot($path) -eq $outputRootDrive)
    $linked = $false

    if ($DryRun) {
        Write-Host "[DryRun] Link $destPath -> $path"
        $created++
        $linked = $true
    } else {
        if ($sameDrive) {
            try {
                New-Item -ItemType HardLink -Path $destPath -Target $path -ErrorAction Stop | Out-Null
                $created++
                $linked = $true
            } catch {
                # fall through to symlink/junction
            }
        }
        if (-not $linked) {
            try {
                New-Item -ItemType SymbolicLink -Path $destPath -Target $path -ErrorAction Stop | Out-Null
                $created++
                $linked = $true
            } catch {
                $linked = $false
            }
        }
        if (-not $linked) {
            # Fallback: create junction to parent directory and write a pointer file
            $parent = Split-Path -Parent $path
            $junctionName = Get-SafeName -Name ((Split-Path -Leaf $parent) + "_" + (Get-ShortHash -Path $path))
            $junctionPath = Join-Path $OutputRoot $junctionName
            if (-not (Test-Path $junctionPath)) {
                try {
                    New-Item -ItemType Junction -Path $junctionPath -Target $parent -ErrorAction Stop | Out-Null
                } catch {
                    $failed++
                    continue
                }
            }
            $pointer = $destPath + '.link.txt'
            Set-Content -Path $pointer -Value $path -Encoding UTF8
            $created++
        }
    }

    # Track HF-style model directories for junctions
    $parentDir = Split-Path -Parent $path
    if (Test-Path (Join-Path $parentDir 'config.json')) {
        $hfDirs.Add($parentDir) | Out-Null
    }
}

foreach ($dir in $hfDirs) {
    $name = Get-SafeName -Name ([System.IO.Path]::GetFileName($dir))
    $dest = Join-Path $hfRoot $name
    if (Test-Path $dest) {
        if ($Force -and -not $DryRun) { Remove-Item -Path $dest -Force -ErrorAction SilentlyContinue }
        else { continue }
    }
    if ($DryRun) {
        Write-Host "[DryRun] Junction $dest -> $dir"
        continue
    }
    try {
        New-Item -ItemType Junction -Path $dest -Target $dir -ErrorAction Stop | Out-Null
    } catch {
        $failed++
    }
}

Write-Host "Link summary: created=$created, skipped=$skipped, failed=$failed"
