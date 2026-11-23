#Requires -Version 7.4
<#
.SYNOPSIS
    Stages one digest-pinned upstream llama.cpp runtime.

.DESCRIPTION
    Selects an archive for the current operating system and PowerShell process
    architecture. The release manifest records the GitHub release asset hashes
    and lengths. The installer verifies cached and newly downloaded archives,
    extracts supported ZIP entries into a private staging directory, and moves
    the completed tree to a previously unused destination on the same volume.
    Device enumeration and workload qualification are separate from staging.

.PARAMETER Route
    An explicit build archive selection. Windows x64 supports CPU, IntelVulkan,
    and NvidiaCuda. The CUDA route includes the matched CUDA 13.5 libraries.

.PARAMETER OutputRoot
    Caller-controlled root for the version, platform and route directories.

.PARAMETER CacheRoot
    Optional read-only directory of release archives named as in the manifest.
    A cache miss downloads into temporary staging. A digest or size mismatch
    in an existing cache is an error.

.PARAMETER DryRun
    Returns the selected route and destination before any filesystem writes or downloads.

.EXAMPLE
    ./Tools/Install-PcaiLlamaRuntime.ps1 -Route IntelVulkan -OutputRoot ./llama-runtime -DryRun

.LINK
    https://github.com/ggml-org/llama.cpp/releases/tag/b12635
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)]
    [ValidateSet('CPU', 'IntelVulkan', 'NvidiaCuda')]
    [string]$Route,

    [Parameter(Mandatory)]
    [string]$OutputRoot,

    [string]$CacheRoot,

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Expand-VerifiedLlamaZip {
    param(
        [string]$ArchivePath,
        [System.Collections.IDictionary]$Asset,
        [string]$PackageRoot
    )

    $stream = [IO.File]::Open($ArchivePath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    try {
        if ($stream.Length -ne $Asset.sizeBytes) {
            throw "Size mismatch for $($Asset.name). Expected $($Asset.sizeBytes) bytes but found $($stream.Length)."
        }
        $actual = (Get-FileHash -InputStream $stream -Algorithm SHA256).Hash
        if ($actual -ne $Asset.sha256) {
            throw "SHA-256 mismatch for $($Asset.name)."
        }

        $stream.Position = 0
        $zip = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Read, $true)
        try {
            if ($zip.Entries.Count -gt 1024) {
                throw "Too many ZIP entries for $($Asset.name)."
            }
            $entryNames = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
            $rootPrefix = $PackageRoot + [IO.Path]::DirectorySeparatorChar
            [long]$expandedBytes = 0
            foreach ($entry in $zip.Entries) {
                $name = $entry.FullName.Replace('\', '/')
                $directory = $name.EndsWith('/')
                $relativePath = $name.TrimEnd('/')
                $fileType = ($entry.ExternalAttributes -shr 16) -band 0xF000
                if ($fileType -notin @(0, 0x4000, 0x8000) -or ($entry.ExternalAttributes -band 0x400)) {
                    throw "Unsupported ZIP entry type in $($Asset.name)."
                }
                foreach ($part in $relativePath.Split('/')) {
                    if ($part -match '[\x00-\x1f<>:"|?*]' -or $part -match '[. ]\z' -or
                        $part -match '\A(?:CON|PRN|AUX|NUL|COM[0-9¹²³]|LPT[0-9¹²³])(?:\.|\z)' -or
                        [string]::IsNullOrEmpty($part) -or $part -ceq '$null') {
                        throw "Invalid ZIP entry path in $($Asset.name)."
                    }
                }
                if (-not $entryNames.Add($relativePath)) {
                    throw "Duplicate ZIP entry path in $($Asset.name)."
                }
                $target = [IO.Path]::GetFullPath((Join-Path $PackageRoot $relativePath))
                if (-not $target.StartsWith($rootPrefix, [StringComparison]::Ordinal)) {
                    throw "ZIP entry path leaves the stage for $($Asset.name)."
                }
                if ($directory) {
                    [void][IO.Directory]::CreateDirectory($target)
                } else {
                    if ($entry.Length -gt (4GB - $expandedBytes)) {
                        throw "Expanded ZIP size limit exceeded for $($Asset.name)."
                    }
                    $expandedBytes += $entry.Length
                    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target))
                    [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $false)
                }
            }
        } finally {
            $zip.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

$manifestPath = Join-Path $PSScriptRoot '../Config/llama-cpp-runtime.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -AsHashtable
if ($manifest.schemaVersion -ne 1 -or $manifest.version -notmatch '\Ab[0-9]+\z' -or
    $manifest.commit -notmatch '\A[0-9a-f]{40}\z') {
    throw 'Invalid llama.cpp release manifest identity.'
}
$architecture = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString().ToLowerInvariant()
$os = if ($IsWindows) { 'windows' } elseif ($IsLinux) { 'linux' } else { 'unsupported' }
$platform = "$os-$architecture"
$result = [ordered]@{
    Status = 'Unsupported'
    Version = $manifest.version
    Commit = $manifest.commit
    Platform = $platform
    Route = $Route
    Destination = $null
    ServerPath = $null
    BenchmarkPath = $null
    ReceiptPath = $null
    Assets = @()
    Reason = $null
}
$profile = $manifest.platforms[$platform]
if ($null -eq $profile -or $platform -ne 'windows-x64') {
    $result.Reason = "The pinned $platform archive format has no supported staging profile."
    return [pscustomobject]$result
}
$canonicalRoute = @('CPU', 'IntelVulkan', 'NvidiaCuda') | Where-Object { $_ -eq $Route }
$ids = @($profile.routes[$canonicalRoute])
if ($ids.Count -eq 0 -or $null -eq $ids[0] -or ($Route -eq 'NvidiaCuda' -and $ids.Count -ne 2)) {
    throw "Invalid asset selection for $platform $Route."
}
$assets = foreach ($id in $ids) {
    $asset = $profile.assets[$id]
    if ($null -eq $asset -or $asset.name -notmatch '\A[a-z0-9][a-z0-9.-]+\.zip\z' -or
        $asset.sha256 -notmatch '\A[0-9a-f]{64}\z' -or
        ($asset.sizeBytes -isnot [long] -and $asset.sizeBytes -isnot [int]) -or
        $asset.sizeBytes -le 0 -or $asset.sizeBytes -gt 4GB) {
        throw "Invalid ZIP asset declaration '$id' for $platform $Route."
    }
    $asset
}
$parent = Join-Path ($ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot)) "$($manifest.version)/$platform"
$destination = Join-Path $parent $canonicalRoute
$cache = if ($CacheRoot) { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($CacheRoot) }
$result.Route = $canonicalRoute
$result.Destination = $destination
$result.Assets = @($assets)
if ($DryRun) {
    $result.Status = 'DryRun'
    return [pscustomobject]$result
}
if (-not $PSCmdlet.ShouldProcess($destination, "Stage llama.cpp $($manifest.version) $canonicalRoute")) {
    $result.Status = if ($WhatIfPreference) { 'WhatIf' } else { 'Declined' }
    return [pscustomobject]$result
}
if (Test-Path -LiteralPath $destination) {
    throw "Runtime destination is already occupied: $destination"
}

[void][IO.Directory]::CreateDirectory($parent)
$stage = Join-Path $parent ('.stage-' + [guid]::NewGuid().ToString('N'))
New-Item -Path $stage -ItemType Directory | Out-Null
try {
    $package = Join-Path $stage 'runtime'
    [void][IO.Directory]::CreateDirectory($package)
    foreach ($asset in $assets) {
        $cachedPath = if ($cache) { Join-Path $cache $asset.name }
        if ($cachedPath -and (Test-Path -LiteralPath $cachedPath)) {
            $archivePath = $cachedPath
        } else {
            $archivePath = Join-Path $stage $asset.name
            $url = "https://github.com/ggml-org/llama.cpp/releases/download/$($manifest.version)/$($asset.name)"
            Invoke-WebRequest -Uri $url -OutFile $archivePath -TimeoutSec 600
        }
        Expand-VerifiedLlamaZip -ArchivePath $archivePath -Asset $asset -PackageRoot $package
    }

    $files = Get-ChildItem -LiteralPath $package -File -Recurse -Force
    $servers = @($files | Where-Object Name -EQ 'llama-server.exe')
    $benchmarks = @($files | Where-Object Name -EQ 'llama-bench.exe')
    if ($servers.Count -ne 1 -or $benchmarks.Count -ne 1 -or
        $servers[0].Length -eq 0 -or $benchmarks[0].Length -eq 0 -or
        $servers[0].DirectoryName -ne $benchmarks[0].DirectoryName) {
        throw 'The staged archive must contain one nonempty server and one nonempty benchmark in the same directory.'
    }

    $serverRelative = [IO.Path]::GetRelativePath($package, $servers[0].FullName)
    $benchmarkRelative = [IO.Path]::GetRelativePath($package, $benchmarks[0].FullName)
    $receiptName = 'pcai-runtime-receipt.json'
    $receiptPath = Join-Path $package $receiptName
    if (Test-Path -LiteralPath $receiptPath) { throw 'The archive contains the reserved runtime receipt path.' }
    $receipt = [ordered]@{
        Schema = 'pcai.llama-runtime-receipt/v1'
        Route = $canonicalRoute
        Version = $manifest.version
        Commit = $manifest.commit
        BenchmarkRelativePath = $benchmarkRelative.Replace('\', '/')
        Binaries = [ordered]@{}
    }
    foreach ($file in $files | Where-Object Extension -In '.exe', '.dll' | Sort-Object FullName) {
        $relative = [IO.Path]::GetRelativePath($package, $file.FullName).Replace('\', '/')
        $receipt.Binaries[$relative] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    [IO.File]::WriteAllText($receiptPath, (ConvertTo-Json -InputObject $receipt -Depth 5))
    [IO.Directory]::Move($package, $destination)
    $result.ServerPath = Join-Path $destination $serverRelative
    $result.BenchmarkPath = Join-Path $destination $benchmarkRelative
    $result.ReceiptPath = Join-Path $destination $receiptName
    $result.Status = 'Installed'
    [pscustomobject]$result
} finally {
    # This directory was created by this invocation under its resolved destination parent.
    $resolvedStage = [IO.Path]::GetFullPath($stage)
    $resolvedParent = [IO.Path]::GetFullPath($parent)
    if ([IO.Path]::GetDirectoryName($resolvedStage) -eq $resolvedParent -and [IO.Directory]::Exists($resolvedStage)) {
        try {
            [IO.Directory]::Delete($resolvedStage, $true)
        } catch {
            Write-Warning "Could not clean runtime staging directory '$resolvedStage': $($_.Exception.Message)"
        }
    }
}
