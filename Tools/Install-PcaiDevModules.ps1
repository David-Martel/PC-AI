#Requires -Version 7.0
<#
.SYNOPSIS
Installs development modules with verified staging and recoverable replacements.
.DESCRIPTION
Copies modules by default. Existing installations are renamed into
.pcai-module-install/<module>/rN/previous and retained with SHA256 receipts.
Failed publication restores the previous installation and preserves staging.
Never deletes an installation or follows reparse points during copy enumeration.
.PARAMETER DryRun
Plans without filesystem, process-environment or registry writes.
.PARAMETER Help
Displays usage. -h and --help are accepted.
.PARAMETER IncludeCargoTools
Uses PcaiModuleBootstrap resolver precedence, including environment overrides.
Root .git metadata is excluded; nested Git stores and linked payloads are rejected.
.PARAMETER PSModulePathScope
Updates Process, User or Machine paths while preserving each existing value.
.EXAMPLE
pwsh -NoProfile -File .\Tools\Install-PcaiDevModules.ps1 -DryRun
.NOTES
Stable receipt revisions include observation times in metadata. Retained previous
installations require separate custody review before retirement. Junction mode
is explicit and keeps its dependency on the selected source directory.
#>
[CmdletBinding(SupportsShouldProcess = $true, PositionalBinding = $false)]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA 'PowerShell\Modules'),
    [ValidateSet('Auto', 'Junction', 'Copy')][string]$Mode = 'Auto',
    [bool]$UpdatePSModulePath = $true,
    [ValidateSet('Process', 'User', 'Machine')][string]$PSModulePathScope = 'User',
    [switch]$IncludeCargoTools,
    [switch]$DryRun,
    [Alias('h')][switch]$Help,
    [Parameter(ValueFromRemainingArguments)][string[]]$CliArgs
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($Help -or @($CliArgs) -contains '--help') {
    'Usage: Install-PcaiDevModules.ps1 [-RepoRoot path] [-InstallRoot path] [-Mode Auto|Copy|Junction] [-IncludeCargoTools] [-PSModulePathScope Process|User|Machine] [-DryRun|-WhatIf] [-h|--help]'
    return
}
if ($CliArgs) { throw "Unknown arguments: $($CliArgs -join ' ')" }
if ($DryRun) { $WhatIfPreference = $true }
. (Join-Path $PSScriptRoot 'PcaiModuleBootstrap.ps1')

function Test-ContainedPath {
    param([string]$Path, [string]$Root)
    $prefix = $Root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    return $Path.Equals($Root, [StringComparison]::OrdinalIgnoreCase) -or $Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}
function Assert-SafePath {
    param([string]$Path, [switch]$AllowLeafLink)
    if ($Path -eq [IO.Path]::GetPathRoot($Path)) { throw "Refusing filesystem root: $Path" }
    $cursor = $Path
    $leaf = $true
    while ($cursor) {
        $name = [IO.Path]::GetFileName($cursor).TrimEnd(' ', '.')
        if ($name -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.|$)') { throw "Unsafe Windows path: $cursor" }
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -and -not ($leaf -and $AllowLeafLink)) {
                throw "Reparse point in installation path: $cursor"
            }
        }
        $leaf = $false
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ($parent -eq $cursor) { break }
        $cursor = $parent
    }
}
function Get-ModuleInventory {
    param([string]$Root, [switch]$ExcludeRootGit)
    $rootItem = Get-Item -LiteralPath $Root -Force
    if ($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Linked module root requires explicit resolution: $Root" }
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($Root)
    $files = [Collections.Generic.List[object]]::new()
    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        foreach ($item in (Get-ChildItem -LiteralPath $directory -Force)) {
            if ($item.Name -eq '.git') {
                if ($ExcludeRootGit -and $directory -eq $Root) { continue }
                throw "Nested Git store requires custody review: $($item.FullName)"
            }
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Linked payload requires custody review: $($item.FullName)" }
            Assert-SafePath -Path $item.FullName
            if ($item.PSIsContainer) { $pending.Push($item.FullName); continue }
            $files.Add([pscustomobject]@{
                Path = [IO.Path]::GetRelativePath($Root, $item.FullName)
                Length = $item.Length
                Sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash
            })
        }
    }
    return @($files | Sort-Object Path)
}
function Test-SameInventory {
    param([object[]]$Left, [object[]]$Right)
    return ($Left.Count -eq $Right.Count) -and ((ConvertTo-Json -InputObject @($Left) -Depth 4 -Compress) -ceq (ConvertTo-Json -InputObject @($Right) -Depth 4 -Compress))
}
function Merge-ModulePath {
    param([string]$Existing, [string]$Preferred)
    $parts = [Collections.Generic.List[string]]::new()
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($part in @($Preferred) + @($Existing -split ';')) {
        if ([string]::IsNullOrWhiteSpace($part)) { continue }
        if ($seen.Add($part.Trim().TrimEnd('\', '/'))) { $parts.Add($part.Trim()) }
    }
    return $parts -join ';'
}
$RepoRoot = [IO.Path]::GetFullPath($RepoRoot).TrimEnd('\', '/')
$InstallRoot = (Get-PcaiStableModuleInstallRoot -InstallRoot $InstallRoot).TrimEnd('\', '/')
Assert-SafePath -Path $InstallRoot
$sourceModulesRoot = Join-Path $RepoRoot 'Modules'
$sources = [Collections.Generic.List[object]]::new()
foreach ($directory in (Get-ChildItem -LiteralPath $sourceModulesRoot -Directory | Sort-Object Name)) {
    if ((Test-Path -LiteralPath (Join-Path $directory.FullName "$($directory.Name).psd1")) -or
        (Test-Path -LiteralPath (Join-Path $directory.FullName "$($directory.Name).psm1"))) {
        $sources.Add([pscustomobject]@{ Name = $directory.Name; SourcePath = $directory.FullName; SourceType = 'repo-module' })
    }
}
if ($IncludeCargoTools) {
    $manifest = Resolve-PcaiModuleManifestPath -ModuleName CargoTools -RepoRoot $RepoRoot -InstallRoot $InstallRoot
    if ($manifest) {
        $sources.Add([pscustomobject]@{ Name = 'CargoTools'; SourcePath = (Split-Path -Parent $manifest); SourceType = 'external-module' })
    } else { Write-Warning 'CargoTools source not found; skipping CargoTools install.' }
}
$names = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($module in $sources) {
    if (-not $names.Add($module.Name)) { throw "Module name collision: $($module.Name)" }
}
foreach ($module in $sources) {
    $source = [IO.Path]::GetFullPath($module.SourcePath).TrimEnd('\', '/')
    $installMode = if ($Mode -eq 'Auto') { 'Copy' } else { $Mode }
    $sourceItem = Get-Item -LiteralPath $source -Force
    # Resolve only the explicitly selected source root; inventory never follows children.
    if ($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { $source = $sourceItem.ResolveLinkTarget($true).FullName }
    if ((Test-ContainedPath -Path $InstallRoot -Root $source) -or (Test-ContainedPath -Path $source -Root $InstallRoot)) {
        throw "Source and installation paths overlap: $source / $InstallRoot"
    }
    $destination = Join-Path $InstallRoot $module.Name
    Assert-SafePath -Path $destination -AllowLeafLink
    $desired = @(Get-ModuleInventory -Root $source -ExcludeRootGit)
    $previous = @()
    $previousLink = $null
    $exists = Test-Path -LiteralPath $destination
    if ($exists) {
        $item = Get-Item -LiteralPath $destination -Force
        if (-not $item.PSIsContainer) { throw "Installation destination is a file: $destination" }
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            $previousLink = @($item.Target) -join ';'
            if ($installMode -eq 'Junction' -and $previousLink -eq $source) {
                [pscustomobject]@{ Name = $module.Name; InstalledPath = $destination; Mode = $installMode; State = 'Unchanged'; Receipt = $null }
                continue
            }
        } else {
            $previous = @(Get-ModuleInventory -Root $destination)
            if ($installMode -eq 'Copy' -and (Test-SameInventory -Left $desired -Right $previous)) {
                [pscustomobject]@{ Name = $module.Name; InstalledPath = $destination; Mode = $installMode; State = 'Unchanged'; Receipt = $null }
                continue
            }
        }
    }
    if (-not $PSCmdlet.ShouldProcess($destination, "Stage, verify and publish $($module.Name) using $installMode; retain previous installation")) { continue }
    $receiptRoot = Join-Path $InstallRoot ".pcai-module-install\$($module.Name)"
    Assert-SafePath -Path $receiptRoot
    [void](New-Item -ItemType Directory -Path $receiptRoot -Force)
    $receiptPath = $null
    $receipt = $null
    $movedPrevious = $false
    $published = $false
    # Acquire custody before selecting a revision or creating transaction state.
    $lockStream = [IO.File]::Open((Join-Path $receiptRoot 'publish.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    try {
        $revision = 1
        while (Test-Path -LiteralPath (Join-Path $receiptRoot "r$revision")) { $revision++ }
        $transaction = Join-Path $receiptRoot "r$revision"
        [void](New-Item -ItemType Directory -Path $transaction)
        $stage = Join-Path $transaction 'staged'
        $backup = Join-Path $transaction 'previous'
        $receiptPath = Join-Path $transaction 'receipt.json'
        $receipt = [ordered]@{
            SchemaVersion = 1; Module = $module.Name; Source = $source; Destination = $destination
            Mode = $installMode; ObservedUtc = [DateTime]::UtcNow.ToString('o'); State = 'Staging'
            PreviousPath = $(if ($exists) { $backup } else { $null }); PreviousLinkTarget = $previousLink
            Files = $desired; PreviousFiles = $previous; Error = $null
        }
        $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
        if ($installMode -eq 'Junction') { [void](New-Item -ItemType Junction -Path $stage -Target $source) }
        else {
            [void](New-Item -ItemType Directory -Path $stage)
            foreach ($file in $desired) {
                $target = Join-Path $stage $file.Path
                [void](New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force)
                Copy-Item -LiteralPath (Join-Path $source $file.Path) -Destination $target
            }
            if (-not (Test-SameInventory -Left $desired -Right @(Get-ModuleInventory -Root $stage))) { throw 'Staged file hashes differ from selected source.' }
        }
        $manifestPath = Join-Path $stage "$($module.Name).psd1"
        if (Test-Path -LiteralPath $manifestPath) {
            $data = Import-PowerShellDataFile -LiteralPath $manifestPath
            if ($data.ContainsKey('RootModule') -and $data.RootModule) {
                $rootModulePath = [IO.Path]::GetFullPath((Join-Path $stage $data.RootModule))
                if (-not (Test-ContainedPath -Path $rootModulePath -Root $stage) -or -not (Test-Path -LiteralPath $rootModulePath -PathType Leaf)) {
                    throw 'Manifest RootModule is absent or outside staged payload.'
                }
            }
        }
        if (-not (Test-SameInventory -Left $desired -Right @(Get-ModuleInventory -Root $source -ExcludeRootGit))) { throw 'Source changed during installation.' }
        if ($exists) {
            if ($previousLink) {
                if ((@((Get-Item -LiteralPath $destination -Force).Target) -join ';') -ne $previousLink) { throw 'Destination link changed during installation.' }
            } elseif (-not (Test-SameInventory -Left $previous -Right @(Get-ModuleInventory -Root $destination))) { throw 'Destination changed during installation.' }
            Move-Item -LiteralPath $destination -Destination $backup
            $movedPrevious = $true
        } elseif (Test-Path -LiteralPath $destination) { throw 'Destination appeared during installation.' }
        Move-Item -LiteralPath $stage -Destination $destination
        $published = $true
        $receipt.State = 'Installed'
        $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
    } catch {
        $failure = $_
        if ($published) { Move-Item -LiteralPath $destination -Destination $stage }
        if ($movedPrevious) { Move-Item -LiteralPath $backup -Destination $destination }
        if ($receipt -and $receiptPath) {
            $receipt.State = 'RolledBack'
            $receipt.Error = $failure.Exception.Message
            $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
        }
        throw $failure
    } finally {
        if ($lockStream) { $lockStream.Dispose() }
    }
    [pscustomobject]@{ Name = $module.Name; SourcePath = $source; InstalledPath = $destination; Mode = $installMode; State = 'Installed'; Receipt = $receiptPath }
}
if ($UpdatePSModulePath -and $PSCmdlet.ShouldProcess("PSModulePath ($PSModulePathScope and process)", 'Prepend installation root while retaining existing module paths')) {
    $processValue = Merge-ModulePath -Existing $env:PSModulePath -Preferred $InstallRoot
    if ($PSModulePathScope -ne 'Process') {
        $oldValue = [Environment]::GetEnvironmentVariable('PSModulePath', $PSModulePathScope)
        $newValue = Merge-ModulePath -Existing $oldValue -Preferred $InstallRoot
        if ($newValue -ne $oldValue) { [Environment]::SetEnvironmentVariable('PSModulePath', $newValue, $PSModulePathScope) }
    }
    $env:PSModulePath = $processValue
}
