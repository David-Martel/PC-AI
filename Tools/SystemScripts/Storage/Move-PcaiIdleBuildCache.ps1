#Requires -Version 7.3
<#
.SYNOPSIS
Relocates an idle workspace build cache with verified byte custody.
.DESCRIPTION
Copies into a new destination, verifies every SHA256, and records original paths
before retiring identical source files. New or changed source files are retained.
Callers must release active owners before use. No junction or persistent setting
is created; future builds must select the destination directly.
.PARAMETER SourceDirectory
Existing cache directory within WorkspaceRoot.
.PARAMETER DestinationDirectory
New, non-overlapping destination on a volume with sufficient free space.
.PARAMETER WorkspaceRoot
Workspace that owns the source cache.
.PARAMETER DryRun
Reports the plan without writing files or changing source data.
.PARAMETER Help
Displays usage. -h and --help are accepted.
.EXAMPLE
./Move-PcaiIdleBuildCache.ps1 -SourceDirectory C:/codedev/PC_AI/.pcai/integration/work-perf-target -DestinationDirectory D:/pcai-relocation/work-perf-target/r1 -DryRun
#>
[CmdletBinding(SupportsShouldProcess, PositionalBinding = $false)]
param(
    [string]$SourceDirectory,
    [string]$DestinationDirectory,
    [string]$WorkspaceRoot = (Join-Path $PSScriptRoot '../../..'),
    [switch]$DryRun,
    [Alias('h')][switch]$Help,
    [Parameter(ValueFromRemainingArguments)][string[]]$CliArgs
)
$ErrorActionPreference = 'Stop'
if ($Help -or @($CliArgs) -contains '--help') {
    'Usage: Move-PcaiIdleBuildCache.ps1 -SourceDirectory path -DestinationDirectory path [-WorkspaceRoot path] [-DryRun|-WhatIf] [-h|--help]'
    return
}
if ($CliArgs) { throw 'Unknown relocation arguments.' }
if ([string]::IsNullOrWhiteSpace($SourceDirectory) -or [string]::IsNullOrWhiteSpace($DestinationDirectory)) {
    throw 'Both source and destination directories are required.'
}

function Assert-PcaiRelocationPath {
    param([string]$Path)
    $cursor = $Path
    while ($cursor) {
        $leaf = [IO.Path]::GetFileName($cursor.TrimEnd([IO.Path]::DirectorySeparatorChar))
        if ($leaf -eq '.git') { throw 'A Git store requires separate custody.' }
        if ($leaf.TrimEnd(' ', '.') -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.|$)') {
            throw "Reserved relocation path: $cursor"
        }
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            # Copying a regular hard-linked file preserves its bytes in the new
            # cache. Retiring this directory entry leaves outside aliases intact.
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
                ($item.LinkType -and ($item.PSIsContainer -or $item.LinkType -ne 'HardLink'))) {
                throw "Linked relocation path requires separate custody: $cursor"
            }
        }
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ($parent -eq $cursor) { break }
        $cursor = $parent
    }
}
function Get-PcaiRelocationFiles {
    param([string]$Root)
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($Root)
    while ($pending.Count) {
        foreach ($item in Get-ChildItem -LiteralPath $pending.Pop() -Force) {
            if ($item.Name -eq '.git') { throw 'A Git store requires separate custody.' }
            Assert-PcaiRelocationPath -Path $item.FullName
            if ($item.PSIsContainer) { $pending.Push($item.FullName) }
            else { $item }
        }
    }
}
function Assert-PcaiRelocationIdle {
    param([string]$Root)
    if ($IsWindows) {
        $users = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            $_.ProcessId -ne $PID -and $_.CommandLine -and
            $_.CommandLine.Replace('/', '\').IndexOf($Root.Replace('/', '\'), [StringComparison]::OrdinalIgnoreCase) -ge 0
        })
        if ($users.Count) { throw 'An active process references the source cache; keep it in place.' }
    } else { throw 'This workstation relocation requires Windows process-custody validation.' }
}

$workspace = [IO.Path]::GetFullPath($WorkspaceRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)
$source = [IO.Path]::GetFullPath($SourceDirectory).TrimEnd([IO.Path]::DirectorySeparatorChar)
$destination = [IO.Path]::GetFullPath($DestinationDirectory).TrimEnd([IO.Path]::DirectorySeparatorChar)
$workspacePrefix = $workspace + [IO.Path]::DirectorySeparatorChar
$sourcePrefix = $source + [IO.Path]::DirectorySeparatorChar
$destinationPrefix = $destination + [IO.Path]::DirectorySeparatorChar
if (-not $source.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The source must be contained by the explicitly selected workspace.'
}
if ($destination -eq [IO.Path]::GetPathRoot($destination) -or
    $source.Equals($destination, [StringComparison]::OrdinalIgnoreCase) -or
    $destination.StartsWith($sourcePrefix, [StringComparison]::OrdinalIgnoreCase) -or
    $source.StartsWith($destinationPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Source and destination must be distinct, non-overlapping directories.'
}
Assert-PcaiRelocationPath -Path $source
Assert-PcaiRelocationPath -Path $destination
if (-not (Test-Path -LiteralPath $source -PathType Container)) { throw 'Source cache is absent.' }
$staging = $destination + '.pending'
$receiptPath = $destination + '.relocation.json'
foreach ($path in @($destination, $staging, $receiptPath)) {
    Assert-PcaiRelocationPath -Path $path
    if (Test-Path -LiteralPath $path) { throw "Existing destination custody must be retained: $path" }
}
Assert-PcaiRelocationIdle -Root $source
$files = @(Get-PcaiRelocationFiles -Root $source)
$bytes = [long](($files | Measure-Object Length -Sum).Sum)
$volume = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($destination))
if (-not $volume.IsReady -or $volume.AvailableFreeSpace -lt ($bytes + 1GB)) {
    throw 'Destination does not have adequate verified free space.'
}
if ($DryRun -or -not $PSCmdlet.ShouldProcess($source, "Relocate verified idle cache to $destination")) {
    [pscustomobject]@{ State = 'Planned'; Source = $source; Destination = $destination; FileCount = $files.Count; Bytes = $bytes }
    return
}
[void](New-Item -ItemType Directory -Path $staging)
$manifest = [Collections.Generic.List[object]]::new()
foreach ($file in $files) {
    $relative = [IO.Path]::GetRelativePath($source, $file.FullName)
    $copy = Join-Path $staging $relative
    [void](New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($copy)) -Force)
    $originalHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    Copy-Item -LiteralPath $file.FullName -Destination $copy -ErrorAction Stop
    $copyHash = (Get-FileHash -LiteralPath $copy -Algorithm SHA256).Hash
    if ($copyHash -ne $originalHash) { throw 'Staged bytes differ; original and staging remain preserved.' }
    $manifest.Add([pscustomobject]@{ RelativePath = $relative; Bytes = $file.Length; SHA256 = $originalHash; SourceLinkType = $file.LinkType })
}
Assert-PcaiRelocationIdle -Root $source
$currentFiles = @(Get-PcaiRelocationFiles -Root $source)
if ($currentFiles.Count -ne $manifest.Count) { throw 'Source inventory changed; original and staging remain preserved.' }
foreach ($entry in $manifest) {
    if ((Get-FileHash -LiteralPath (Join-Path $source $entry.RelativePath) -Algorithm SHA256).Hash -ne $entry.SHA256) {
        throw 'Source bytes changed; original and staging remain preserved.'
    }
}
# This rename stays within the destination volume. The original is retained
# until a complete verified manifest is durably recorded outside the source.
[IO.Directory]::Move($staging, $destination)
$receipt = [ordered]@{ Version = 1; ObservedAtUtc = [DateTimeOffset]::UtcNow.ToString('o'); State = 'VerifiedCopy'; Source = $source; Destination = $destination; Bytes = $bytes; Files = $manifest }
$receiptStream = [IO.File]::Open($receiptPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::ReadWrite, [IO.FileShare]::Read)
try {
    $receiptBytes = [Text.UTF8Encoding]::new($false).GetBytes(($receipt | ConvertTo-Json -Depth 5))
    $receiptStream.Write($receiptBytes, 0, $receiptBytes.Length)
    $receiptStream.Flush($true)
    $receiptStream.Position = 0
    $recorded = [byte[]]::new($receiptBytes.Length)
    $receiptStream.ReadExactly($recorded)
    if (-not [Linq.Enumerable]::SequenceEqual[byte]($receiptBytes, $recorded)) {
        throw 'Receipt verification failed; originals and destination remain retained.'
    }
    Assert-PcaiRelocationIdle -Root $source
    foreach ($entry in $manifest) {
        $original = [IO.Path]::GetFullPath((Join-Path $source $entry.RelativePath))
        if (-not $original.StartsWith($sourcePrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Manifest path escaped the source.' }
        Assert-PcaiRelocationPath -Path $original
        if ((Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash -ne $entry.SHA256) {
            throw 'A source changed after publication; remaining originals and destination are retained.'
        }
        Remove-Item -LiteralPath $original -Force -ErrorAction Stop
    }
    # Remove only empty directories. Any new/unmapped work prevents retirement.
    $directories = @(Get-ChildItem -LiteralPath $source -Directory -Recurse -Force | Sort-Object { $_.FullName.Length } -Descending)
    foreach ($directory in $directories) {
        Assert-PcaiRelocationPath -Path $directory.FullName
        [IO.Directory]::Delete($directory.FullName, $false)
    }
    [IO.Directory]::Delete($source, $false)
    $receipt.State = 'Relocated'
    $receipt.CompletedAtUtc = [DateTimeOffset]::UtcNow.ToString('o')
    $receiptBytes = [Text.UTF8Encoding]::new($false).GetBytes(($receipt | ConvertTo-Json -Depth 5))
    $receiptStream.Position = 0
    $receiptStream.Write($receiptBytes, 0, $receiptBytes.Length)
    $receiptStream.SetLength($receiptBytes.Length)
    $receiptStream.Flush($true)
} finally { $receiptStream.Dispose() }
[pscustomobject]@{ State = $receipt.State; Source = $source; Destination = $destination; FileCount = $manifest.Count; Bytes = $bytes; Receipt = $receiptPath }
