#Requires -Version 7.0
<#
.SYNOPSIS
Installs development modules with verified staging and recoverable replacements.
.DESCRIPTION
Copies modules by default. Existing installations are renamed into
.pcai-module-install/<module>/rN/previous and retained with SHA256 receipts.
Failed publication restores the previous installation and preserves staging.
Never deletes an installation or follows name-surrogate links during enumeration.
Admits only the 16 documented Cloud Files tags, queried through no-follow handles;
other placeholder families and unknown reparse tags remain rejected.
.PARAMETER DryRun
Plans without filesystem, process-environment or registry writes.
.PARAMETER Help
Displays usage. -h and --help are accepted.
.PARAMETER CargoToolsSourcePath
Explicit CargoTools directory or canonical CargoTools.psd1 file. Other manifest
filenames are rejected before staging to preserve the selected module identity.
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
    [string[]]$ModuleNames,
    [string]$CargoToolsSourcePath,
    [string]$BackupRoot,
    [switch]$DryRun,
    [Alias('h')][switch]$Help,
    [Parameter(ValueFromRemainingArguments)][string[]]$CliArgs
)


Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'


function Test-ContainedPath {
    param([string]$Path, [string]$Root)
    $prefix = $Root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    return $Path.Equals($Root, [StringComparison]::OrdinalIgnoreCase) -or $Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}

function Test-PcaiCloudReparseTag {
    param([uint32]$Tag)
    # MS-FSCC documents exactly CLOUD and CLOUD_1..CLOUD_F. Never admit a
    # name-surrogate (junction/symlink) or another provider/placeholder family.
    return (($Tag -band [uint32]536870912) -eq 0) -and
        (($Tag -band [Convert]::ToUInt32('FFFF0FFF', 16)) -eq [Convert]::ToUInt32('9000001A', 16))
}

function Get-PcaiInstallReparseTag {
    param([Parameter(Mandatory)][string]$Path)
    if (-not $IsWindows) { throw "Windows reparse tag query unavailable: $Path" }
    if (-not ('Pcai.ModuleInstallReparseTagV1' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
namespace Pcai {
    public static class ModuleInstallReparseTagV1 {
        [StructLayout(LayoutKind.Sequential)]
        private struct AttributeTagInfo { public uint Attributes; public uint Tag; }
        [DllImport("kernel32.dll", CharSet=CharSet.Unicode, ExactSpelling=true, SetLastError=true)]
        private static extern SafeFileHandle CreateFileW(string path, uint access,
            uint share, IntPtr security, uint disposition, uint flags, IntPtr template);
        [DllImport("kernel32.dll", ExactSpelling=true, SetLastError=true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GetFileInformationByHandleEx(SafeFileHandle handle,
            int infoClass, out AttributeTagInfo info, uint size);
        public static uint ReadTag(string path) {
            // Read metadata only; OPEN_EXISTING + OPEN_REPARSE_POINT never
            // follows the leaf link. BACKUP_SEMANTICS admits directory handles.
            using (SafeFileHandle handle = CreateFileW(path, 0x80, 7,
                IntPtr.Zero, 3, 0x02200000, IntPtr.Zero)) {
                if (handle.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateFileW no-follow metadata open failed");
                AttributeTagInfo info;
                if (!GetFileInformationByHandleEx(handle, 9, out info, 8))
                    throw new Win32Exception(Marshal.GetLastWin32Error(), "FileAttributeTagInfo metadata query failed");
                return info.Tag;
            }
        }
    }
}
'@ -ErrorAction Stop
    }
    try { return [Pcai.ModuleInstallReparseTagV1]::ReadTag($Path) }
    catch { throw "Cannot inspect Windows reparse tag for $Path : $($_.Exception.Message)" }
}

function Test-PcaiCloudReparsePath {
    param([Parameter(Mandatory)][string]$Path)
    return Test-PcaiCloudReparseTag -Tag (Get-PcaiInstallReparseTag -Path $Path)
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
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                $tag = Get-PcaiInstallReparseTag -Path $cursor
                $knownLeafLink = $leaf -and $AllowLeafLink -and
                    ($tag -in @([Convert]::ToUInt32('A0000003', 16), [Convert]::ToUInt32('A000000C', 16))) -and
                    ($item.LinkType -in @('Junction', 'SymbolicLink'))
                if (-not (Test-PcaiCloudReparseTag -Tag $tag) -and -not $knownLeafLink) {
                    throw "Reparse point in installation path: $cursor"
                }
            }
        }
        $leaf = $false
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ($parent -eq $cursor) { break }
        $cursor = $parent
    }
}

function Get-ModuleInventory {
    param([string]$Root, [switch]$ExcludeRootGit, [string[]]$RelativeFiles)
    $rootItem = Get-Item -LiteralPath $Root -Force
    if (($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -and
        -not (Test-PcaiCloudReparsePath -Path $Root)) { throw "Linked module root requires explicit resolution: $Root" }
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($Root)
    $files = [Collections.Generic.List[object]]::new()
    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        $items = if ($RelativeFiles) {
            foreach ($relativeFile in $RelativeFiles) {
                $selectedPath = [IO.Path]::GetFullPath((Join-Path $Root $relativeFile))
                if (-not (Test-ContainedPath -Path $selectedPath -Root $Root)) { throw 'Selected payload escapes source root.' }
                Assert-SafePath -Path $selectedPath
                $selectedItem = Get-Item -LiteralPath $selectedPath -Force
                if ($selectedItem.PSIsContainer) { throw 'Selected standalone payload must be a file.' }
                $selectedItem
            }
        } else { Get-ChildItem -LiteralPath $directory -Force }
        foreach ($item in $items) {
            if ($item.Name -eq '.git') {
                if ($ExcludeRootGit -and $directory -eq $Root) { continue }
                throw "Nested Git store requires custody review: $($item.FullName)"
            }
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -and
                -not (Test-PcaiCloudReparsePath -Path $item.FullName)) { throw "Linked payload requires custody review: $($item.FullName)" }
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

function Merge-ModuleConfigObject {
    <# .SYNOPSIS Preserve installed JSON values while adding new source defaults. #>
    param([Collections.IDictionary]$Defaults, [Collections.IDictionary]$Installed)
    $result = @{}
    foreach ($key in $Defaults.Keys) { $result[$key] = $Defaults[$key] }
    foreach ($key in $Installed.Keys) {
        if ($result.ContainsKey($key) -and $result[$key] -is [Collections.IDictionary] -and
            $Installed[$key] -is [Collections.IDictionary]) {
            $result[$key] = Merge-ModuleConfigObject -Defaults $result[$key] -Installed $Installed[$key]
        } else { $result[$key] = $Installed[$key] }
    }
    return $result
}

function Copy-InstalledModuleConfig {
    <# .SYNOPSIS Merge installed Config objects and preserve other installed Config files. #>
    param([string]$Installed, [string]$Staged)
    if (-not [IO.Directory]::Exists($Installed)) { return }
    if (([IO.File]::GetAttributes($Installed) -band [IO.FileAttributes]::ReparsePoint) -ne 0 -and -not (Test-PcaiCloudReparsePath $Installed)) {
        throw "Installed Config cannot be a reparse point: '$Installed'."
    }
    [IO.Directory]::CreateDirectory($Staged) | Microsoft.PowerShell.Core\Out-Null
    foreach ($entry in Microsoft.PowerShell.Management\Get-ChildItem -LiteralPath $Installed -Force) {
        if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -and -not (Test-PcaiCloudReparsePath $entry.FullName)) { throw "Installed Config contains a reparse point: '$($entry.FullName)'." }
        Assert-SafePath -Path $entry.FullName
        $target = [IO.Path]::Combine($Staged, $entry.Name)
        if ($entry.PSIsContainer) {
            Copy-InstalledModuleConfig -Installed $entry.FullName -Staged $target
        } elseif ($entry.Extension -eq '.json') {
            $previous = Microsoft.PowerShell.Utility\ConvertFrom-Json -InputObject ([IO.File]::ReadAllText($entry.FullName)) -AsHashtable
            $defaults = if ([IO.File]::Exists($target)) {
                Microsoft.PowerShell.Utility\ConvertFrom-Json -InputObject ([IO.File]::ReadAllText($target)) -AsHashtable
            } else { @{} }
            if ($previous -isnot [Collections.IDictionary] -or $defaults -isnot [Collections.IDictionary]) {
                throw "Config JSON must have an object root: '$($entry.FullName)'."
            }
            $merged = Merge-ModuleConfigObject -Defaults $defaults -Installed $previous
            $json = Microsoft.PowerShell.Utility\ConvertTo-Json -InputObject $merged -Depth 50 -WarningAction Stop
            [IO.File]::WriteAllText($target, $json, [Text.UTF8Encoding]::new($false))
        } else { [IO.File]::Copy($entry.FullName, $target, $true) }
    }
}

function Test-ModuleManifestFileReference {
    <# .SYNOPSIS Check local manifest references while allowing named modules and assemblies. #>
    param([object]$Reference, [string]$Field, [string]$Root, [string]$PayloadRoot)
    if ($Reference -is [Collections.IDictionary]) {
        if ($Field -ne 'NestedModules' -or -not $Reference.Contains('ModuleName')) {
            throw "Unsupported manifest reference in '$Field'."
        }
        $Reference = $Reference['ModuleName']
    }
    if ([string]::IsNullOrWhiteSpace([string]$Reference)) { throw "Empty manifest reference in '$Field'." }
    $name = [string]$Reference
    $isFile = $name -match '[\\/:]' -or [IO.Path]::GetExtension($name) -in @('.ps1', '.psm1', '.psd1', '.dll', '.ps1xml', '.exe')
    if ($Field -in @('NestedModules', 'RequiredAssemblies') -and -not $isFile) { return }
    $candidate = [IO.Path]::GetFullPath([IO.Path]::Combine($Root, $name))
    if (-not (Test-ContainedPath -Path $candidate -Root $PayloadRoot) -or -not [IO.File]::Exists($candidate)) {
        if ($Field -ceq 'RootModule') {
            throw "Manifest RootModule is absent or outside staged payload (outside the staged module): '$name'."
        }
        throw "Manifest payload missing or outside the staged module: '$Field' reference '$name'."
    }
    return $candidate
}

function Test-StagedModulePayload {
    <# .SYNOPSIS Validate copied scripts/data and active manifest references without importing modules. #>
    param([string]$Name, [string]$Path)
    $manifest = [IO.Path]::Combine($Path, "$Name.psd1")
    $scriptFile = [IO.Path]::Combine($Path, "$Name.psm1")
    if (-not [IO.File]::Exists($manifest) -and -not [IO.File]::Exists($scriptFile)) {
        throw "Module '$Name' has no manifest or root script."
    }

    # Traverse explicitly so a development junction payload cannot introduce nested links.
    $metadataFiles = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::OrdinalIgnoreCase)
    $directories = [Collections.Generic.Stack[string]]::new()
    $directories.Push($Path)
    while ($directories.Count) {
        $directory = $directories.Pop()
        foreach ($entry in Microsoft.PowerShell.Management\Get-ChildItem -LiteralPath $directory -Force) {
            if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -and -not (Test-PcaiCloudReparsePath $entry.FullName)) {
                throw "Staged payload contains a reparse point: '$($entry.FullName)'."
            }
            if ($entry.Name -ceq '.git' -and $directory -ceq $Path) { continue }
            Assert-SafePath -Path $entry.FullName
            if ($entry.PSIsContainer) { $directories.Push($entry.FullName); continue }
            if ($entry.Extension -eq '.psd1') {
                $metadata = Microsoft.PowerShell.Utility\Import-PowerShellDataFile -LiteralPath $entry.FullName
                $metadataFiles.Add($entry.FullName, $metadata)
            } elseif ($entry.Extension -in @('.ps1', '.psm1')) {
                $tokens = $null
                $parseErrors = $null
                [Management.Automation.Language.Parser]::ParseFile($entry.FullName, [ref]$tokens, [ref]$parseErrors) | Microsoft.PowerShell.Core\Out-Null
                if ($parseErrors.Count) {
                    throw "Script syntax failed for '$($entry.FullName)': $($parseErrors[0].Message)"
                }
            }
        }
    }
    $active = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $pending = [Collections.Generic.Stack[string]]::new()
    if ([IO.File]::Exists($manifest)) { $pending.Push($manifest) }
    while ($pending.Count) {
        $activePath = $pending.Pop()
        if (-not $active.Add($activePath)) { continue }
        if (-not $metadataFiles.ContainsKey($activePath)) { throw "Active manifest was not staged: '$activePath'." }
        $metadata = $metadataFiles[$activePath]
        if ($metadata -isnot [Collections.IDictionary] -or -not $metadata.Contains('ModuleVersion') -or
            -not [Version]::TryParse([string]$metadata.ModuleVersion, [ref]([Version]$null))) {
            throw "Invalid ModuleVersion in '$activePath'."
        }
        foreach ($field in @('RootModule', 'FileList', 'ScriptsToProcess', 'TypesToProcess', 'FormatsToProcess', 'RequiredAssemblies', 'NestedModules')) {
            if (-not $metadata.Contains($field) -or -not $metadata[$field]) { continue }
            foreach ($reference in @($metadata[$field])) {
                $localFile = Test-ModuleManifestFileReference -Reference $reference -Field $field -Root ([IO.Path]::GetDirectoryName($activePath)) -PayloadRoot $Path
                if ($localFile -and $field -in @('RootModule', 'NestedModules') -and [IO.Path]::GetExtension($localFile) -eq '.psd1') {
                    $pending.Push($localFile)
                }
            }
        }
    }
    $diagnostics = [Collections.Generic.List[string]]::new()
    foreach ($pair in $metadataFiles.GetEnumerator()) {
        if ($active.Contains($pair.Key) -or $pair.Value -isnot [Collections.IDictionary] -or
            -not $pair.Value.Contains('ModuleVersion') -or -not $pair.Value.Contains('FileList')) { continue }
        foreach ($reference in @($pair.Value['FileList'])) {
            try {
                Test-ModuleManifestFileReference -Reference $reference -Field FileList -Root ([IO.Path]::GetDirectoryName($pair.Key)) -PayloadRoot $Path |
                    Microsoft.PowerShell.Core\Out-Null
            } catch {
                $message = "Inactive manifest '$($pair.Key)' has a stale FileList: $($_.Exception.Message)"
                $diagnostics.Add($message)
                Microsoft.PowerShell.Utility\Write-Verbose $message
            }
        }
    }
    return [pscustomobject]@{ ActiveManifestCount = $active.Count; InactiveManifestDiagnostics = $diagnostics.ToArray() }
}

# Resolve and validate all plan inputs before creating transaction state.
function Resolve-CombinedDeploymentPath {
    param([string]$Path,[switch]$AllowLeafLink)
    if ([string]::IsNullOrWhiteSpace($Path) -or -not [IO.Path]::IsPathFullyQualified($Path)) { throw 'Deployment paths must be absolute filesystem paths.' }
    if ($Path.Substring([IO.Path]::GetPathRoot($Path).Length).Contains(':')) { throw 'Deployment paths must not use alternate data streams.' }
    # Validate raw Windows components before normalization can erase aliases.
    # Keep the consolidated resolver and cloud-file custody policy; do not
    # resurrect the older installer-only resolver from the merged branch.
    if ($IsWindows) {
        foreach ($part in $Path.Substring([IO.Path]::GetPathRoot($Path).Length) -split '[\\/]') {
            if ($part -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:[. ]|$)') { throw "Unsafe Windows path: $Path" }
            if ($part -notin @('.', '..') -and $part -match '[. ]$') {
                throw "Deployment path components must not end with a dot or space: '$Path'."
            }
        }
    }
    $resolved = [IO.Path]::GetFullPath($Path).TrimEnd('\','/')
    Assert-SafePath -Path $resolved -AllowLeafLink:$AllowLeafLink
    return $resolved
}
function Get-MergedPsModulePath {
    param([string]$PreferredRoot,[string]$CurrentValue)
    return Merge-ModulePath -Preferred $PreferredRoot -Existing $CurrentValue
}
function Assert-CombinedUnchangedPlanFresh {
    param([object]$Plan)
    if(-not (Test-SameInventory $Plan.Desired @(Get-ModuleInventory -Root $Plan.Source -ExcludeRootGit -RelativeFiles $Plan.Module.PayloadFiles))){throw 'Source changed during unchanged installation verification.'}
    if(-not (Test-Path -LiteralPath $Plan.Destination -PathType Container)){throw 'Unchanged destination disappeared or changed type.'}
    Assert-SafePath -Path $Plan.Destination -AllowLeafLink
    if($Plan.Mode -eq 'Copy'){
        if(-not (Test-SameInventory $Plan.Previous @(Get-ModuleInventory -Root $Plan.Destination))){throw 'Unchanged destination changed during installation.'}
    }elseif((@((Get-Item -LiteralPath $Plan.Destination -Force).Target) -join ';') -cne $Plan.PreviousLink){throw 'Unchanged destination link changed during installation.'}
}
if ($Help -or @($CliArgs) -contains '--help') {
    'Usage: Install-PcaiDevModules.ps1 [-RepoRoot path] [-InstallRoot path] [-ModuleNames names] [-CargoToolsSourcePath path] [-BackupRoot path] [-Mode Auto|Copy|Junction] [-IncludeCargoTools] [-PSModulePathScope Process|User|Machine] [-DryRun|-WhatIf] [-h|--help]'
    return
}
if ($CliArgs) { throw "Unknown arguments: $($CliArgs -join ' ')" }
if ($DryRun) { $WhatIfPreference = $true }
. (Join-Path $PSScriptRoot 'PcaiModuleBootstrap.ps1')
$RepoRoot = Resolve-CombinedDeploymentPath $RepoRoot
$InstallRoot = Resolve-CombinedDeploymentPath $InstallRoot
$InstallRoot = Resolve-CombinedDeploymentPath (Get-PcaiStableModuleInstallRoot -InstallRoot $InstallRoot)
$receiptStore = Join-Path $InstallRoot '.pcai-module-install'
$backupStore = if ($BackupRoot) { Resolve-CombinedDeploymentPath $BackupRoot } else { $receiptStore }
if ([IO.Path]::GetPathRoot($backupStore) -ine [IO.Path]::GetPathRoot($InstallRoot)) { throw 'BackupRoot and InstallRoot must be on the same volume.' }
$sourceModulesRoot = Join-Path $RepoRoot 'Modules'
$sources = [Collections.Generic.List[object]]::new()
foreach ($directory in (Get-ChildItem -LiteralPath $sourceModulesRoot -Directory | Sort-Object Name)) {
    if ((Test-Path -LiteralPath (Join-Path $directory.FullName "$($directory.Name).psd1")) -or
        (Test-Path -LiteralPath (Join-Path $directory.FullName "$($directory.Name).psm1"))) {
        $sources.Add([pscustomobject]@{Name=$directory.Name;SourcePath=$directory.FullName;SourceType='repo-module';PayloadFiles=$null})
    }
}
foreach ($manifest in (Get-ChildItem -LiteralPath $sourceModulesRoot -File -Filter '*.psd1' | Sort-Object Name)) {
    Assert-SafePath -Path $manifest.FullName
    $name=$manifest.BaseName; $data=Import-PowerShellDataFile -LiteralPath $manifest.FullName
    if (-not $data.ContainsKey('RootModule') -or $data.RootModule -cne "$name.psm1") { throw "Standalone manifest must select its paired script: $($manifest.FullName)" }
    $sources.Add([pscustomobject]@{Name=$name;SourcePath=$sourceModulesRoot;SourceType='standalone-module';PayloadFiles=@($manifest.Name,[string]$data.RootModule)})
}
$wanted=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach($name in $ModuleNames){if([string]::IsNullOrWhiteSpace($name) -or $name -match '[\\/:]' -or -not $wanted.Add($name)){throw 'ModuleNames contains an invalid or duplicate name.'}}
if($IncludeCargoTools -or $CargoToolsSourcePath -or $wanted.Contains('CargoTools')){
    $manifest=if($CargoToolsSourcePath){
        $cargoSource=Resolve-CombinedDeploymentPath $CargoToolsSourcePath
        if([IO.Directory]::Exists($cargoSource)){Join-Path $cargoSource 'CargoTools.psd1'}else{$cargoSource}
    }else{Resolve-PcaiModuleManifestPath -ModuleName CargoTools -RepoRoot $RepoRoot -InstallRoot $InstallRoot}
    if(-not $manifest -or -not (Test-Path -LiteralPath $manifest -PathType Leaf)){throw 'CargoTools source manifest was not found.'}
    if([IO.Path]::GetFileName($manifest) -ine 'CargoTools.psd1'){throw 'Explicit CargoTools manifest must be named CargoTools.psd1; refusing to select a different sibling manifest.'}
    $sources.Add([pscustomobject]@{Name='CargoTools';SourcePath=(Split-Path -Parent $manifest);SourceType='external-module';PayloadFiles=$null})
}
if($wanted.Count){$sources=[Collections.Generic.List[object]]::new([object[]]@($sources | Where-Object {$wanted.Contains($_.Name)}))}
$seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach($module in $sources){if(-not $seen.Add($module.Name)){throw "Module name collision: $($module.Name)"}}
foreach($name in $wanted){if(-not $seen.Contains($name)){throw "Selected module was not found: $name"}}
$plans=[Collections.Generic.List[object]]::new()
foreach($module in $sources){
    $source=Resolve-CombinedDeploymentPath $module.SourcePath -AllowLeafLink
    $sourceItem=Get-Item -LiteralPath $source -Force
    if(($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -and -not (Test-PcaiCloudReparsePath $source)){
        $source=Resolve-CombinedDeploymentPath $sourceItem.ResolveLinkTarget($true).FullName
    }
    $destination=Join-Path $InstallRoot $module.Name
    if((Test-ContainedPath $InstallRoot $source) -or (Test-ContainedPath $source $InstallRoot)){throw 'Source and installation paths must not overlap.'}
    if((Test-ContainedPath $backupStore $source) -or (Test-ContainedPath $source $backupStore) -or
        (Test-ContainedPath $backupStore $destination) -or (Test-ContainedPath $destination $backupStore)){throw 'BackupRoot must not overlap source or active module payloads.'}
    Assert-SafePath -Path $destination -AllowLeafLink
    $installMode=if($Mode -eq 'Auto'){'Copy'}else{$Mode}
    if($installMode -eq 'Junction' -and $module.PayloadFiles){throw "Standalone module $($module.Name) requires Copy or Auto mode; Junction would expose unrelated modules."}
    if($installMode -eq 'Junction' -and ($module.SourceType -ne 'repo-module' -or
        $source -match '(?i)[\\/]OneDrive(?:[\\/]|$)' -or -not (Test-ContainedPath $source $RepoRoot))){throw 'Junction requires a repo-local directory module; use Copy.'}
    $desired=@(Get-ModuleInventory -Root $source -ExcludeRootGit -RelativeFiles $module.PayloadFiles)
    $exists=Test-Path -LiteralPath $destination; $previous=@();$previousLink=$null
    if($exists){
        $item=Get-Item -LiteralPath $destination -Force
        if(-not $item.PSIsContainer){throw "Installed target is a file: $destination"}
        if(($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -and -not (Test-PcaiCloudReparsePath $destination)){$previousLink=@($item.Target) -join ';'}
        else{$previous=@(Get-ModuleInventory -Root $destination)}
        if($installMode -eq 'Junction' -and (Test-Path -LiteralPath (Join-Path $destination 'Config')) -and
            -not [string]::Equals($previousLink,$source,[StringComparison]::OrdinalIgnoreCase)){throw 'Installed Config requires preservation; use Copy.'}
    }
    # Copy identity is whole source/prior inventory equality, including Config.
    # Different installed Config remains a changed plan requiring a real overlay.
    $unchanged=$exists -and (($installMode -eq 'Copy' -and -not $previousLink -and (Test-SameInventory $desired $previous)) -or
        ($installMode -eq 'Junction' -and $previousLink -and [string]::Equals($previousLink,$source,[StringComparison]::OrdinalIgnoreCase)))
    $plans.Add([pscustomobject]@{Module=$module;Source=$source;Destination=$destination;Mode=$installMode;Desired=$desired;Previous=$previous;PreviousLink=$previousLink;Existed=$exists;Unchanged=[bool]$unchanged;Lock=$null;Transaction=$null;Stage=$null;Backup=$null;ReceiptPath=$null;Receipt=$null;Effective=@();Validation=$null;MovedPrevious=$false;Published=$false})
}
if(-not $PSCmdlet.ShouldProcess($InstallRoot,"Stage and validate all $($plans.Count) selected modules before publishing; retain rN receipts and previous payloads")){
    foreach($plan in $plans){[pscustomobject]@{Name=$plan.Module.Name;InstalledPath=$plan.Destination;Mode=$plan.Mode;State='DryRun';Status='DryRun';Receipt=$null;BackupPath=$null}}
    return
}
$oldProcessPath=[Environment]::GetEnvironmentVariable('PSModulePath','Process')
$oldScopedPath=[Environment]::GetEnvironmentVariable('PSModulePath',$PSModulePathScope)
$first=$null;$secondary=[Collections.Generic.List[object]]::new();$pathUpdateAttempted=$false
try{
    # All locks, stages, config overlays and validations precede any active rename.
    foreach($plan in $plans){
        if($plan.Unchanged){
            Assert-CombinedUnchangedPlanFresh $plan
            $validationRoot=if($plan.Mode -eq 'Copy'){$plan.Destination}else{$plan.Source}
            $plan.Validation=Test-StagedModulePayload -Name $plan.Module.Name -Path $validationRoot
            Assert-CombinedUnchangedPlanFresh $plan
            continue
        }
        $receiptRoot=Join-Path $receiptStore $plan.Module.Name
        Assert-SafePath $receiptRoot
        [IO.Directory]::CreateDirectory($receiptRoot) | Out-Null
        $plan.Lock=[IO.File]::Open((Join-Path $receiptRoot 'publish.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
        $revision=1
        $backupModuleRoot=Join-Path $backupStore $plan.Module.Name
        while((Test-Path -LiteralPath (Join-Path $receiptRoot "r$revision")) -or (Test-Path -LiteralPath (Join-Path $backupModuleRoot "r$revision"))){$revision++}
        $plan.Transaction=Join-Path $receiptRoot "r$revision"
        [IO.Directory]::CreateDirectory($plan.Transaction) | Out-Null
        $plan.Stage=Join-Path $plan.Transaction 'staged'
        $plan.Backup=Join-Path (Join-Path $backupModuleRoot "r$revision") 'previous'
        $plan.ReceiptPath=Join-Path $plan.Transaction 'receipt.json'
        $plan.Receipt=[ordered]@{SchemaVersion=1;Module=$plan.Module.Name;Source=$plan.Source;Destination=$plan.Destination;Mode=$plan.Mode;ObservedUtc=[DateTime]::UtcNow.ToString('o');State='Staging';PreviousPath=$(if($plan.Existed){$plan.Backup}else{$null});PreviousLinkTarget=$plan.PreviousLink;Files=$plan.Desired;EffectiveFiles=@();PreviousFiles=$plan.Previous;Error=$null;SecondaryErrors=@()}
        $plan.Receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $plan.ReceiptPath -Encoding utf8
        if($plan.Mode -eq 'Copy'){
            [IO.Directory]::CreateDirectory($plan.Stage) | Out-Null
            foreach($file in $plan.Desired){$target=Join-Path $plan.Stage $file.Path;[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target)) | Out-Null;Copy-Item -LiteralPath (Join-Path $plan.Source $file.Path) -Destination $target}
            if(-not (Test-SameInventory $plan.Desired @(Get-ModuleInventory -Root $plan.Stage))){throw 'Staged file hashes differ from selected source.'}
            # Never follow a prior installation link to read Config implicitly.
            if($plan.Existed -and -not $plan.PreviousLink){Copy-InstalledModuleConfig -Installed (Join-Path $plan.Destination 'Config') -Staged (Join-Path $plan.Stage 'Config')}
            $plan.Effective=@(Get-ModuleInventory -Root $plan.Stage)
            $plan.Validation=Test-StagedModulePayload -Name $plan.Module.Name -Path $plan.Stage
        }else{
            $plan.Validation=Test-StagedModulePayload -Name $plan.Module.Name -Path $plan.Source
            $plan.Effective=$plan.Desired
        }
        if(-not (Test-SameInventory $plan.Desired @(Get-ModuleInventory -Root $plan.Source -ExcludeRootGit -RelativeFiles $plan.Module.PayloadFiles))){throw 'Source changed during installation.'}
        $plan.Receipt.EffectiveFiles=$plan.Effective;$plan.Receipt.State='Validated'
        $plan.Receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $plan.ReceiptPath -Encoding utf8
    }
    foreach($plan in $plans){
        if($plan.Unchanged){continue}
        if(-not (Test-SameInventory $plan.Desired @(Get-ModuleInventory -Root $plan.Source -ExcludeRootGit -RelativeFiles $plan.Module.PayloadFiles))){throw 'Source changed before publication.'}
        if($plan.Mode -eq 'Copy' -and -not (Test-SameInventory $plan.Effective @(Get-ModuleInventory -Root $plan.Stage))){throw 'Effective staged payload changed before publication.'}
        if($plan.Existed){
            if($plan.PreviousLink){if((@((Get-Item -LiteralPath $plan.Destination -Force).Target) -join ';') -cne $plan.PreviousLink){throw 'Destination link changed during installation.'}}
            elseif(-not (Test-SameInventory $plan.Previous @(Get-ModuleInventory -Root $plan.Destination))){throw 'Destination changed during installation.'}
            Assert-SafePath -Path ([IO.Path]::GetDirectoryName($plan.Backup))
            [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($plan.Backup)) | Out-Null
            Move-Item -LiteralPath $plan.Destination -Destination $plan.Backup
            $plan.MovedPrevious=$true
        }elseif(Test-Path -LiteralPath $plan.Destination){throw 'Destination appeared during installation.'}
        if($plan.Mode -eq 'Junction'){New-Item -ItemType Junction -Path $plan.Stage -Target $plan.Source | Out-Null}
        Move-Item -LiteralPath $plan.Stage -Destination $plan.Destination;$plan.Published=$true
        if($plan.Mode -eq 'Copy' -and -not (Test-SameInventory $plan.Effective @(Get-ModuleInventory -Root $plan.Destination))){throw 'Published payload differs from validated effective inventory.'}
        $plan.Receipt.State='Installed';$plan.Receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $plan.ReceiptPath -Encoding utf8
    }
    if($UpdatePSModulePath){
        $pathUpdateAttempted=$true
        $newScoped=Merge-ModulePath -Existing $oldScopedPath -Preferred $InstallRoot
        [Environment]::SetEnvironmentVariable('PSModulePath',$newScoped,$PSModulePathScope)
        if($PSModulePathScope -ne 'Process'){[Environment]::SetEnvironmentVariable('PSModulePath',(Merge-ModulePath -Existing $oldProcessPath -Preferred $InstallRoot),'Process')}
    }
    foreach($plan in $plans){if($plan.Unchanged){Assert-CombinedUnchangedPlanFresh $plan}}
}catch{
    $first=$_
    if($pathUpdateAttempted){
        try{[Environment]::SetEnvironmentVariable('PSModulePath',$oldScopedPath,$PSModulePathScope)}catch{$secondary.Add($_)}
        try{[Environment]::SetEnvironmentVariable('PSModulePath',$oldProcessPath,'Process')}catch{$secondary.Add($_)}
    }
    for($i=$plans.Count-1;$i -ge 0;$i--){
        $plan=$plans[$i];$restored=$true
        if($plan.Published){try{Move-Item -LiteralPath $plan.Destination -Destination $plan.Stage;$plan.Published=$false}catch{$secondary.Add($_);$restored=$false}}
        if($plan.MovedPrevious -and -not $plan.Published){try{Move-Item -LiteralPath $plan.Backup -Destination $plan.Destination;$plan.MovedPrevious=$false}catch{$secondary.Add($_);$restored=$false}}
        if($plan.Receipt){
            $plan.Receipt.State=if($restored){'RolledBack'}else{'RollbackIncomplete'}
            $plan.Receipt.Error=$first.Exception.Message;$plan.Receipt.SecondaryErrors=@($secondary | ForEach-Object {$_.Exception.Message})
            try{$plan.Receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $plan.ReceiptPath -Encoding utf8}catch{$secondary.Add($_)}
        }
    }
}finally{
    foreach($plan in $plans){if($plan.Lock){try{$plan.Lock.Dispose()}catch{if($null -eq $first){$first=$_}else{$secondary.Add($_)}}}}
}
if($first){$first.Exception.Data['PcaiInstallerSecondaryOriginals']=$secondary;$first.Exception.Data['PcaiInstallerOriginalPlans']=$plans;throw $first}
foreach($plan in $plans){[pscustomobject]@{Name=$plan.Module.Name;SourcePath=$plan.Source;InstalledPath=$plan.Destination;Mode=$plan.Mode;State=$(if($plan.Unchanged){'Unchanged'}else{'Installed'});Status=$(if($plan.Unchanged){'Unchanged'}else{'Installed'});Receipt=$plan.ReceiptPath;BackupPath=$(if($plan.Existed -and -not $plan.Unchanged){$plan.Backup}else{$null});InactiveManifestDiagnostics=$plan.Validation.InactiveManifestDiagnostics}}
