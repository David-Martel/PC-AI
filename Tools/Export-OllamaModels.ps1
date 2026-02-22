#Requires -Version 5.1
<#
.SYNOPSIS
    Export Ollama model blobs as hardlinked GGUF files into the repo.

.DESCRIPTION
    Reads Ollama manifests under %USERPROFILE%\.ollama\models\manifests and
    hardlinks the model blob (sha256-*) into a usable .gguf filename
    under the repo's Models\ollama directory.

.PARAMETER OutputRoot
    Destination root (default: .\Models\ollama).

.PARAMETER Force
    Overwrite existing links/files.

.PARAMETER DryRun
    Only print actions, do not create links.

.PARAMETER NoFallbackCopy
    Do not fallback to file copy if hardlink fails (e.g., cross-volume).
#>
[CmdletBinding()]
param(
    [string]$OutputRoot = (Join-Path (Resolve-Path .) 'Models\ollama'),
    [switch]$Force,
    [switch]$DryRun,
    [switch]$NoFallbackCopy
)

function Get-SafeName {
    param([string]$Name)
    $safe = $Name -replace '[\\/:*?"<>|]', '_' -replace '\s+', '_'
    $safe = $safe.Trim('_')
    if (-not $safe) { $safe = 'model' }
    return $safe
}

function Test-GgufMagic {
    param([string]$Path)
    try {
        $stream = [System.IO.File]::OpenRead($Path)
        try {
            $buffer = New-Object byte[] 4
            $read = $stream.Read($buffer, 0, 4)
            if ($read -eq 4) {
                $magicStr = [System.Text.Encoding]::ASCII.GetString($buffer)
                return $magicStr -eq 'GGUF'
            }
        }
        finally {
            $stream.Dispose()
        }
    }
    catch { }
    return $false
}

$userHome = $env:USERPROFILE
if (-not $userHome) {
    throw "USERPROFILE not set."
}

$ollamaRoot = Join-Path $userHome '.ollama\models'
$manifestRoot = Join-Path $ollamaRoot 'manifests'
$blobRoot = Join-Path $ollamaRoot 'blobs'

if (-not (Test-Path $manifestRoot)) {
    throw "Ollama manifests not found at $manifestRoot"
}

if (-not (Test-Path $blobRoot)) {
    throw "Ollama blobs not found at $blobRoot"
}

if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
}

$manifestFiles = Get-ChildItem -Path $manifestRoot -Recurse -File -ErrorAction SilentlyContinue
if (-not $manifestFiles -or $manifestFiles.Count -eq 0) {
    Write-Host "No Ollama manifest files found under $manifestRoot"
    return
}

$exported = 0
foreach ($file in $manifestFiles) {
    try {
        $json = Get-Content $file.FullName -Raw -ErrorAction Stop | ConvertFrom-Json
        $modelLayer = $json.layers | Where-Object { $_.mediaType -match 'ollama\.image\.model' } | Select-Object -First 1
        if (-not $modelLayer) { continue }

        $digest = $modelLayer.digest
        if (-not $digest) { continue }

        $digestId = $digest -replace '^sha256:', ''
        $blobPath = Join-Path $blobRoot ("sha256-" + $digestId)
        if (-not (Test-Path $blobPath)) { continue }

        $relativeModel = $file.FullName.Replace($manifestRoot, '').TrimStart('\')
        $safeModel = Get-SafeName $relativeModel
        $shortDigest = if ($digestId.Length -ge 8) { $digestId.Substring(0, 8) } else { $digestId }
        $filename = "${safeModel}__${shortDigest}.gguf"
        $destPath = Join-Path $OutputRoot $filename

        $isGguf = Test-GgufMagic -Path $blobPath
        if (-not $isGguf) {
            Write-Host "Skipping non-GGUF blob: $relativeModel ($blobPath)"
            continue
        }

        if (Test-Path $destPath) {
            if ($Force) {
                if (-not $DryRun) {
                    Remove-Item -Path $destPath -Force -ErrorAction SilentlyContinue
                }
            }
            else {
                Write-Host "Exists, skipping: $destPath"
                continue
            }
        }

        if ($DryRun) {
            Write-Host "[DryRun] Hardlink $destPath -> $blobPath"
            $exported++
            continue
        }

        try {
            New-Item -ItemType HardLink -Path $destPath -Target $blobPath -ErrorAction Stop | Out-Null
            $exported++
            Write-Host "Linked: $destPath"
        }
        catch {
            Write-Warning "Hardlink failed: $destPath -> $blobPath ($($_.Exception.Message))"
            if (-not $NoFallbackCopy) {
                Copy-Item -Path $blobPath -Destination $destPath -Force
                $exported++
                Write-Host "Copied: $destPath"
            }
        }
    }
    catch {
        Write-Warning "Failed to parse manifest: $($file.FullName) ($($_.Exception.Message))"
    }
}

Write-Host "Exported $exported Ollama models to $OutputRoot"
