#Requires -Version 7.0

<#
.SYNOPSIS
Publish selected development modules with staged validation and recoverable backups.
.DESCRIPTION
Auto uses Copy. Existing module directories are renamed into a unique backup
directory; they are never deleted. Existing Config values override source defaults.
All selected payloads are staged and validated without importing modules before
publication. Root and referenced nested manifests receive file-reference checks;
other data manifests are parsed and preserved, with stale FileList diagnostics.
Named external dependencies require verification when importing in a fresh shell.
PSModulePath entries are preserved in both the selected scope and
the current process. Use Junction only for a single repo-local development module
without installed Config overrides. Run deployments from a fresh profile-free shell;
already loaded modules continue to use their existing code.
.PARAMETER ModuleNames
Names to publish. Omit to publish all eligible repo modules.
.PARAMETER CargoToolsSourcePath
Explicit CargoTools source directory or manifest path. CargoTools is included when
selected by ModuleNames or IncludeCargoTools. The legacy OneDrive source is a fallback.
.PARAMETER BackupRoot
Backup directory on the same volume as InstallRoot, outside selected module payloads.
Defaults to InstallRoot\.pcai-module-backups. Backups and failed stages are retained.
.PARAMETER DryRun
Validate the deployment plan and return it without file or environment writes.
.EXAMPLE
.\Install-PcaiDevModules.ps1 -ModuleNames CargoTools -CargoToolsSourcePath C:\codedev\CargoTools -Mode Copy -DryRun
.EXAMPLE
.\Install-PcaiDevModules.ps1 -ModuleNames CargoTools -CargoToolsSourcePath C:\codedev\CargoTools -Mode Copy
#>
[CmdletBinding(SupportsShouldProcess, PositionalBinding = $false)]
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
    [Alias('h', '-help')][switch]$Help,
    [Parameter(ValueFromRemainingArguments)][string[]]$RemainingArguments
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-ModuleDeploymentPath {
    <# .SYNOPSIS Validate an absolute Windows path and its existing destination ancestors. #>
    param([string]$Path, [switch]$Destination)
    if ($Path) { $Path = $Path.Replace('/', '\') }
    if ([string]::IsNullOrWhiteSpace($Path) -or -not [IO.Path]::IsPathFullyQualified($Path)) {
        throw "Deployment paths must be absolute: '$Path'."
    }
    if ($Path -notmatch '^[A-Za-z]:\\' -or $Path.Substring(2).Contains(':')) {
        throw "Deployment paths must use a local drive without alternate data streams: '$Path'."
    }
    foreach ($part in $Path.Substring(3) -split '[\\/]') {
        if ($part -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:[. ]|$)') {
            throw "Reserved path component in '$Path'."
        }
    }
    $resolved = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    if ($resolved -eq [IO.Path]::GetPathRoot($resolved).TrimEnd('\', '/')) {
        throw "A deployment directory cannot be a drive root: '$Path'."
    }
    if ($Destination) {
        $cursor = $resolved
        while ($cursor) {
            if ([IO.File]::Exists($cursor)) { throw "Destination ancestor is a file: '$cursor'." }
            if ([IO.Directory]::Exists($cursor) -and
                (([IO.File]::GetAttributes($cursor) -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
                throw "Destination ancestor is a reparse point: '$cursor'."
            }
            $parent = [IO.Path]::GetDirectoryName($cursor)
            if ($parent -eq $cursor) { break }
            $cursor = $parent
        }
    }
    return $resolved
}

function Test-ModulePathContained {
    <# .SYNOPSIS Compare path containment with a directory boundary. #>
    param([string]$Path, [string]$Root)
    return $Path.StartsWith($Root.TrimEnd('\', '/') + '\', [StringComparison]::OrdinalIgnoreCase)
}

function Get-MergedPsModulePath {
    <# .SYNOPSIS Prepend a module root while preserving and deduplicating existing entries. #>
    param([string]$PreferredRoot, [AllowNull()][string]$CurrentValue)
    $parts = [Collections.Generic.List[string]]::new()
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in @($PreferredRoot) + @($CurrentValue -split ';')) {
        if ([string]::IsNullOrWhiteSpace($entry)) { continue }
        $display = $entry.Trim()
        $key = [Environment]::ExpandEnvironmentVariables($display).TrimEnd('\', '/')
        if ([IO.Path]::IsPathFullyQualified($key)) { $key = [IO.Path]::GetFullPath($key).TrimEnd('\', '/') }
        if ($seen.Add($key)) { $parts.Add($display) }
    }
    return $parts -join ';'
}

function Copy-ModulePayload {
    <# .SYNOPSIS Copy a source payload without following directory or file links. #>
    param([string]$Source, [string]$Destination, [switch]$TopLevel)
    if (([IO.File]::GetAttributes($Source) -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Source reparse points are unsupported: '$Source'."
    }
    [IO.Directory]::CreateDirectory($Destination) | Microsoft.PowerShell.Core\Out-Null
    foreach ($entry in Microsoft.PowerShell.Management\Get-ChildItem -LiteralPath $Source -Force) {
        if ($TopLevel -and $entry.Name -in @('.git', '.claude', '.github', '.venv', '.pcai', 'target', 'worktrees', 'Reports')) { continue }
        if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Source reparse points are unsupported: '$($entry.FullName)'."
        }
        $target = [IO.Path]::Combine($Destination, $entry.Name)
        if ($entry.PSIsContainer) { Copy-ModulePayload -Source $entry.FullName -Destination $target }
        else { [IO.File]::Copy($entry.FullName, $target, $true) }
    }
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
    if (([IO.File]::GetAttributes($Installed) -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Installed Config cannot be a reparse point: '$Installed'."
    }
    [IO.Directory]::CreateDirectory($Staged) | Microsoft.PowerShell.Core\Out-Null
    foreach ($entry in Microsoft.PowerShell.Management\Get-ChildItem -LiteralPath $Installed -Force) {
        if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Installed Config contains a reparse point: '$($entry.FullName)'." }
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
    if (-not (Test-ModulePathContained -Path $candidate -Root $PayloadRoot) -or -not [IO.File]::Exists($candidate)) {
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
            if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Staged payload contains a reparse point: '$($entry.FullName)'."
            }
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

function Move-ModuleDeploymentDirectory {
    <# .SYNOPSIS Rename a directory on the same volume for publication or rollback. #>
    param([string]$Source, [string]$Destination)
    [IO.Directory]::Move($Source, $Destination)
}

if ($RemainingArguments) {
    if ($RemainingArguments.Count -eq 1 -and $RemainingArguments[0] -eq '--help') { $Help = $true }
    else { throw "Unexpected arguments: $($RemainingArguments -join ' ')" }
}
if ($Help) {
    Microsoft.PowerShell.Core\Get-Help -Name $PSCommandPath -Detailed
    return
}
if (-not $IsWindows) { throw 'This deployment helper requires Windows.' }
$RepoRoot = Resolve-ModuleDeploymentPath -Path $RepoRoot
$InstallRoot = Resolve-ModuleDeploymentPath -Path $InstallRoot -Destination
if (-not $BackupRoot) { $BackupRoot = [IO.Path]::Combine($InstallRoot, '.pcai-module-backups') }
$BackupRoot = Resolve-ModuleDeploymentPath -Path $BackupRoot -Destination
if ([IO.Path]::GetPathRoot($BackupRoot) -ne [IO.Path]::GetPathRoot($InstallRoot)) {
    throw 'BackupRoot and InstallRoot must be on the same volume for atomic directory renames.'
}
$selected = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($name in $ModuleNames) {
    if ($name -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$') { throw "Invalid module name '$name'." }
    [void]$selected.Add($name)
}
$plans = [Collections.Generic.List[object]]::new()
$modulesRoot = [IO.Path]::Combine($RepoRoot, 'Modules')
if ([IO.Directory]::Exists($modulesRoot)) {
    foreach ($directory in Microsoft.PowerShell.Management\Get-ChildItem -LiteralPath $modulesRoot -Directory |
        Microsoft.PowerShell.Utility\Sort-Object Name) {
        if ($selected.Count -and -not $selected.Contains($directory.Name)) { continue }
        if ([IO.File]::Exists([IO.Path]::Combine($directory.FullName, "$($directory.Name).psd1")) -or
            [IO.File]::Exists([IO.Path]::Combine($directory.FullName, "$($directory.Name).psm1"))) {
            $plans.Add([pscustomobject]@{ Name = $directory.Name; SourcePath = $directory.FullName; SourceType = 'repo-module' })
        }
    }
}
if ($IncludeCargoTools -or $selected.Contains('CargoTools')) {
    $cargoSource = if ($CargoToolsSourcePath) { $CargoToolsSourcePath }
        else { [IO.Path]::Combine($env:USERPROFILE, 'OneDrive\Documents\PowerShell\Modules\CargoTools') }
    $cargoSource = Resolve-ModuleDeploymentPath -Path $cargoSource
    if ([IO.File]::Exists($cargoSource) -and [IO.Path]::GetExtension($cargoSource) -eq '.psd1') {
        $cargoSource = [IO.Path]::GetDirectoryName($cargoSource)
    }
    if (-not [IO.File]::Exists([IO.Path]::Combine($cargoSource, 'CargoTools.psd1'))) {
        throw "CargoTools source manifest was not found. Supply -CargoToolsSourcePath: '$cargoSource'."
    }
    if (-not @($plans | Microsoft.PowerShell.Core\Where-Object Name -eq 'CargoTools').Count) {
        $plans.Add([pscustomobject]@{ Name = 'CargoTools'; SourcePath = $cargoSource; SourceType = 'external-module' })
    }
}
foreach ($name in $selected) {
    if (-not @($plans | Microsoft.PowerShell.Core\Where-Object Name -eq $name).Count) { throw "Selected module '$name' was not found." }
}
if (-not $plans.Count) { throw 'No eligible modules were selected.' }
$effectiveMode = if ($Mode -eq 'Auto') { 'Copy' } else { $Mode }
foreach ($plan in $plans) {
    $sourcePath = Resolve-ModuleDeploymentPath -Path $plan.SourcePath
    $plan.SourcePath = $sourcePath
    $targetPath = Resolve-ModuleDeploymentPath -Path ([IO.Path]::Combine($InstallRoot, $plan.Name))
    if ($sourcePath -eq $targetPath -or $InstallRoot -eq $sourcePath -or (Test-ModulePathContained $InstallRoot $sourcePath) -or
        (Test-ModulePathContained $sourcePath $targetPath) -or $BackupRoot -eq $targetPath -or
        (Test-ModulePathContained $BackupRoot $targetPath) -or $BackupRoot -eq $sourcePath -or
        (Test-ModulePathContained $BackupRoot $sourcePath)) {
        throw "Source, target and backup payloads must not overlap for '$($plan.Name)'."
    }
    if ($effectiveMode -eq 'Junction' -and ($plans.Count -ne 1 -or
        -not (Test-ModulePathContained $sourcePath $RepoRoot) -or $sourcePath -match '(?i)[\\/]OneDrive(?:[\\/]|$)' -or
        [IO.Directory]::Exists([IO.Path]::Combine($targetPath, 'Config')))) {
        throw 'Junction requires one repo-local, non-OneDrive source without installed Config overrides. Use Copy.'
    }
    $plan | Microsoft.PowerShell.Utility\Add-Member -NotePropertyMembers @{
        InstalledPath = $targetPath; Mode = $effectiveMode; BackupRoot = $BackupRoot
    }
}
if ($DryRun) {
    foreach ($plan in $plans) { $plan | Microsoft.PowerShell.Utility\Add-Member -NotePropertyName Status -NotePropertyValue 'DryRun' -PassThru }
    return
}
if (-not $PSCmdlet.ShouldProcess($InstallRoot, "Stage and publish $($plans.Count) module(s), retain backups, update module path: $UpdatePSModulePath")) { return }
$transaction = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '-' + [Guid]::NewGuid().ToString('N')
$stageRoot = Resolve-ModuleDeploymentPath -Path ([IO.Path]::Combine($InstallRoot, ".pcai-stage-$transaction")) -Destination
$backupTransaction = Resolve-ModuleDeploymentPath -Path ([IO.Path]::Combine($BackupRoot, $transaction)) -Destination
$records = [Collections.Generic.List[object]]::new()
$oldProcessPath = [Environment]::GetEnvironmentVariable('PSModulePath', 'Process')
$oldScopedPath = [Environment]::GetEnvironmentVariable('PSModulePath', $PSModulePathScope)
$environmentTouched = $false
try {
    [IO.Directory]::CreateDirectory($stageRoot) | Microsoft.PowerShell.Core\Out-Null
    foreach ($plan in $plans) {
        $stage = [IO.Path]::Combine($stageRoot, $plan.Name)
        $record = [pscustomobject]@{ Plan = $plan; Stage = $stage; Backup = [IO.Path]::Combine($backupTransaction, $plan.Name); OldMoved = $false; NewMoved = $false; Validation = $null }
        $records.Add($record)
        if ($effectiveMode -eq 'Junction') {
            Microsoft.PowerShell.Management\New-Item -ItemType Junction -Path $stage -Target $plan.SourcePath | Microsoft.PowerShell.Core\Out-Null
        } else {
            Copy-ModulePayload -Source $plan.SourcePath -Destination $stage -TopLevel
            Copy-InstalledModuleConfig -Installed ([IO.Path]::Combine($plan.InstalledPath, 'Config')) -Staged ([IO.Path]::Combine($stage, 'Config'))
        }
        $record.Validation = Test-StagedModulePayload -Name $plan.Name -Path $stage
    }
    [IO.Directory]::CreateDirectory($backupTransaction) | Microsoft.PowerShell.Core\Out-Null
    foreach ($record in $records) {
        if ([IO.File]::Exists($record.Plan.InstalledPath)) { throw "Installed target is a file: '$($record.Plan.InstalledPath)'." }
        if ([IO.Directory]::Exists($record.Plan.InstalledPath)) {
            Move-ModuleDeploymentDirectory -Source $record.Plan.InstalledPath -Destination $record.Backup
            $record.OldMoved = $true
        }
        Move-ModuleDeploymentDirectory -Source $record.Stage -Destination $record.Plan.InstalledPath
        $record.NewMoved = $true
    }
    if ($UpdatePSModulePath) {
        $environmentTouched = $true
        $mergedScope = Get-MergedPsModulePath -PreferredRoot $InstallRoot -CurrentValue $oldScopedPath
        [Environment]::SetEnvironmentVariable('PSModulePath', $mergedScope, $PSModulePathScope)
        if ($PSModulePathScope -ne 'Process') {
            $mergedProcess = Get-MergedPsModulePath -PreferredRoot $InstallRoot -CurrentValue $oldProcessPath
            [Environment]::SetEnvironmentVariable('PSModulePath', $mergedProcess, 'Process')
        }
    }
} catch {
    $failure = $_.Exception.Message
    $rollbackErrors = [Collections.Generic.List[string]]::new()
    if ($environmentTouched) {
        try {
            [Environment]::SetEnvironmentVariable('PSModulePath', $oldScopedPath, $PSModulePathScope)
            [Environment]::SetEnvironmentVariable('PSModulePath', $oldProcessPath, 'Process')
        } catch { $rollbackErrors.Add("Environment rollback: $($_.Exception.Message)") }
    }
    for ($index = $records.Count - 1; $index -ge 0; $index--) {
        $record = $records[$index]
        try {
            if ($record.NewMoved) {
                Move-ModuleDeploymentDirectory -Source $record.Plan.InstalledPath -Destination ($record.Stage + '.failed')
            }
            if ($record.OldMoved) {
                Move-ModuleDeploymentDirectory -Source $record.Backup -Destination $record.Plan.InstalledPath
            }
        } catch { $rollbackErrors.Add("$($record.Plan.Name): $($_.Exception.Message)") }
    }
    $detail = if ($rollbackErrors.Count) { " Rollback errors: $($rollbackErrors -join '; ')" } else { ' Existing module directories were restored.' }
    throw "Deployment failed: $failure.$detail Staged evidence retained at '$stageRoot'."
}
foreach ($record in $records) {
    [pscustomobject]@{
        Name = $record.Plan.Name; SourcePath = $record.Plan.SourcePath; InstalledPath = $record.Plan.InstalledPath
        Mode = $effectiveMode; SourceType = $record.Plan.SourceType; Status = 'Installed'
        BackupPath = if ($record.OldMoved) { $record.Backup } else { $null }
        StageRoot = $stageRoot
        InactiveManifestDiagnostics = $record.Validation.InactiveManifestDiagnostics
    }
}
