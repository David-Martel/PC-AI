#Requires -Version 7.0
<#
.SYNOPSIS
    Applies machine-path.policy.json to one host: PATH, redundant binaries and agent-home-hostile wrappers.

.DESCRIPTION
    Dry run by default; nothing is written without -Apply. With -Apply the script:

      1. backs up both PATH values (raw text and registry kind) to <BackupDir>\<host>.<scope>.path.bak-<hash>
         and records a resolution baseline of the policy commands (<host>.baseline.before.json);
      2. rewrites the USER PATH: absolute entries only, dead and duplicate entries removed, forbidden entries
         removed, the policy head block first, the remaining entries in their existing order. The machine PATH is
         only touched with -IncludeMachine (needs an elevated session);
      3. archives, never deletes, to <managed bin>\.superseded\<name>-<sha12><ext>:
           * SHA-suffixed or pre-<sha> variants of a binary (agent-bus-<sha>.exe and so on),
           * pass-through wrappers (a .cmd that only calls the same-named program elsewhere on PATH),
           * exe copies identical to a machine-scope or WinGet/Go copy of the same program,
           * the older of two different copies of one command in the two managed bin directories
             (a hardlink to the newer copy is left behind when something references the old path);
         but only after a reference search of live config, scheduled tasks, services and running processes finds
         nothing that names the exact file. Referenced files are skipped and reported;
      4. rewrites wrappers that build paths from USERPROFILE or HOME for paths that exist only in the real profile
         (agent homes redirect both) to %~dp0 or the real profile path, keeping the original in .superseded;
      5. broadcasts WM_SETTINGCHANGE, re-runs the baseline and prints every command whose winner changed.

    Revert: restore PATH from the backup files with Restore-MachinePath.ps1 semantics, which is
    Set-ItemProperty on the registry value named in the first line of each file (see the printed command).

.PARAMETER Apply
    Execute. Without it the script prints the plan only.

.PARAMETER BackupDir
    Defaults to the real profile's .machine\backups\path.

.EXAMPLE
    pwsh -NoProfile -File Repair-MachinePath.ps1            # dry run
    pwsh -NoProfile -File Repair-MachinePath.ps1 -Apply
#>
[CmdletBinding()]
param(
    [switch]$Apply,
    [switch]$IncludeMachine,
    [string[]]$MachineAdditions = @(),
    [string]$PolicyPath = (Join-Path $PSScriptRoot 'machine-path.policy.json'),
    [string]$UserHome,
    [string]$BackupDir,
    [switch]$SkipArchive,
    [switch]$SkipWrappers
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'MachinePath.psm1') -Force

$ctx = Get-MachinePathContext -UserHome $UserHome
$policy = Read-MachinePathPolicy -Path $PolicyPath -Context $ctx
if (-not $BackupDir) { $BackupDir = Join-Path $ctx.UserHome '.machine\backups\path' }
$hostName = $env:COMPUTERNAME.ToLowerInvariant()
$actions = [System.Collections.Generic.List[object]]::new()
$sha12 = { param($p) ((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash).Substring(0, 12).ToLowerInvariant() }

# ---- reference search -------------------------------------------------------------------------------------------
# Archival fails CLOSED: if any source that could name a binary (a config file, a scheduled task, a service, a running process)
# cannot be read, "nothing references this file" is not a safe conclusion, so no file is archived in this run.
$referenceFiles = @(
    '.claude.json', '.claude\settings.json', '.codex\config.toml', '.mcp.json', '.gemini\settings.json',
    '.gemini\antigravity-cli\mcp_config.json', '.copilot\mcp-config.json', '.config\agent-bus\config.json',
    'AppData\Roaming\Code\User\settings.json', 'AppData\Roaming\Code\User\mcp.json',
    '.gitconfig', '.ssh\config', '.wezterm.lua', '.config\wezterm\wezterm.lua'
) | ForEach-Object { Join-Path $ctx.UserHome $_ }
$referenceDirs = @('.machine', 'Documents\PowerShell', 'OneDrive\Documents\PowerShell', '.config\powershell', '.config\git', '.config\wezterm', '.agents',
    '.codex\hooks', '.claude\hooks', '.local\share\agent-hooks', '.local\share\git-guard\current\bin') | ForEach-Object { Join-Path $ctx.UserHome $_ }
$script:refCorpus = $null
if (-not $SkipArchive) {
    $script:refCorpus = Get-MachineReferenceCorpus -Files $referenceFiles -Directories $referenceDirs
    if ($script:refCorpus.Failures.Count -gt 0) {
        "reference discovery INCOMPLETE ($($script:refCorpus.Failures.Count) sources unreadable); archival is disabled for this run:"
        $script:refCorpus.Failures | Select-Object -First 10 | ForEach-Object { "  $_" }
        $SkipArchive = $true
    }
}
function Test-Referenced {
    param([string]$Path)
    if ($null -eq $script:refCorpus) { return $true }   # no corpus means no evidence: treat as referenced
    $c = $script:refCorpus.Text
    $needle = $Path.ToLowerInvariant()
    $rel = $needle.Replace($ctx.UserHome.ToLowerInvariant(), '')
    $c.Contains($needle) -or $c.Contains('~' + $rel) -or $c.Contains('%userprofile%' + $rel) -or $c.Contains('$home' + $rel)
}

# ---- 1. PATH plan -----------------------------------------------------------------------------------------------
$user = Get-MachinePathRegistry -Scope User
$machine = Get-MachinePathRegistry -Scope Machine
$plan = Get-MachinePathPlan -UserEntries $user.Entries -MachineEntries $machine.Entries -Policy $policy -Context $ctx
# Entries added to the MACHINE PATH (tokens allowed, e.g. '{CargoHome}\bin') so services and sshd sessions, which never see the user PATH
# as a logon would, can find them. Applied only with -IncludeMachine.
if ($policy.HostOverrideReason) { "host override: $($policy.HostOverrideReason)" }
# Required entries that the (possibly empty) head block did not place are appended, never reordered.
foreach ($req in $policy.RequiredUserEntries) {
    if ((Test-Path -LiteralPath $req -PathType Container) -and -not ($plan.UserEntries | Where-Object { $_ -ieq $req.TrimEnd('\') })) { $plan.UserEntries = @($plan.UserEntries) + $req.TrimEnd('\') }
}
foreach ($add in (@($MachineAdditions) + @($policy.MachineAdditions))) {
    $abs = Expand-MachinePathToken -Text $add -Context $ctx
    if ((Test-Path -LiteralPath $abs -PathType Container) -and -not ($plan.MachineEntries | Where-Object { (Resolve-MachinePathEntry -Entry $_ -Context $ctx) -ieq $abs })) {
        $plan.MachineEntries = @($plan.MachineEntries) + $abs
    }
}
$userChanged = ($plan.UserEntries -join ';') -ne $user.Raw
$machineChanged = $IncludeMachine -and (($plan.MachineEntries -join ';') -ne $machine.Raw)
"PATH plan: user {0} -> {1} entries ({2} -> {3} chars); machine {4} -> {5}{6}" -f $user.Entries.Count, $plan.UserEntries.Count, $user.Raw.Length, ($plan.UserEntries -join ';').Length, $machine.Entries.Count, $plan.MachineEntries.Count, $(if (-not $IncludeMachine) { ' (not applied without -IncludeMachine)' } else { '' })
foreach ($r in $plan.Removed | Where-Object { $_.Scope -eq 'User' -or $IncludeMachine }) { "  remove [{0}] {1} ({2})" -f $r.Scope, $r.Entry, $r.Reason }
$cargoMissing = @($policy.RequiredUserEntries | Where-Object { $_ -and ($plan.UserEntries -notcontains $_) -and (Test-Path -LiteralPath $_) })
foreach ($c in $cargoMissing) { "  note: required entry not placed: $c" }

# ---- 2. archive candidates --------------------------------------------------------------------------------------
$archive = [System.Collections.Generic.List[object]]::new()
if (-not $SkipArchive) {
    $managed = $policy.ManagedBinDirs
    foreach ($d in $managed) { foreach ($v in @(Get-MachineVariantFile -Directory $d)) { $archive.Add([pscustomobject]@{ Path = $v; Why = 'versioned variant' }) } }

    $machineDirs = @($machine.Entries | ForEach-Object { Resolve-MachinePathEntry -Entry $_ -Context $ctx } | Where-Object { [IO.Directory]::Exists($_) })
    $userDirs = @($plan.UserEntries | Where-Object { [IO.Directory]::Exists($_) })
    $allDirs = @($userDirs + $machineDirs | Select-Object -Unique)
    $index = Get-MachineCommandIndex -Directories $allDirs
    $cargoBin = (Join-Path $ctx.CargoHome 'bin').ToLowerInvariant()
    $procDirs = @($env:Path -split ';' | Where-Object { $_ } | ForEach-Object { (Resolve-MachinePathEntry -Entry $_ -Context $ctx).TrimEnd('\').ToLowerInvariant() })
    foreach ($name in $index.Keys) {
        $cands = $index[$name]
        $mine = @($cands | Where-Object { $managed -contains $_.Dir.TrimEnd('\') })
        $others = @($cands | Where-Object { $managed -notcontains $_.Dir.TrimEnd('\') })
        if (-not $mine) { continue }
        if ($policy.IntentionalShadows.ContainsKey($name)) { continue }
        foreach ($m in $mine) {
            if ((Get-Item -LiteralPath $m.Path -Force).LinkType -eq 'HardLink') { continue }   # hardlinks (rustup proxies, pdf-tools) are maintained by Sync-MachineEnvironment
            if ($m.Ext -in '.cmd', '.bat') {
                # pass-through wrapper: one command line that calls an existing program of the same name
                $lines = @(Get-Content -LiteralPath $m.Path | Where-Object { $_ -and $_ -notmatch '^\s*(@echo off|rem\b|::)' })
                if ($lines.Count -eq 1 -and ($lines -join ' ') -match '^\s*(?:call\s+)?"?(?<t>[A-Za-z]:\\[^"]+?\.(?:exe|cmd|bat))"?\s+%\*\s*$' -and
                    ([IO.Path]::GetFileNameWithoutExtension($Matches['t']).ToLowerInvariant() -eq $name) -and (Test-Path -LiteralPath $Matches['t']) -and
                    ($others | Where-Object { $_.Path -eq $Matches['t'] }) -and
                    $Matches['t'] -notlike ((Join-Path $ctx.UserHome 'AppData') + '\*') -and
                    ($procDirs -contains (Split-Path $Matches['t'] -Parent).TrimEnd('\').ToLowerInvariant())) {   # the target's directory must be on THIS process's PATH too: a service started before a PATH change keeps the old one
                    $archive.Add([pscustomobject]@{ Path = $m.Path; Why = "pass-through to $($Matches['t'])" })
                }
            } elseif ($m.Ext -eq '.exe' -and $m.Dir.TrimEnd('\') -eq (Join-Path $ctx.UserHome 'bin')) {
                # Only ~/bin copies are candidates: ~/.local/bin belongs to self-updating installers (uv, claude, python).
                # Repo build output (C:\codedev) and cargo bin directories are not stable enough to be the surviving copy.
                # The twin must be the copy that WINS once this one is gone: the first runnable candidate on PATH, not merely any identical one
                # (gen7: an identical Program Files copy sat behind a WindowsApps alias that would have started winning).
                $firstOther = @($others | Where-Object { $_.Ext -in '.com', '.exe', '.bat', '.cmd' } | Select-Object -First 1)
                $twin = @($firstOther | Where-Object { $_.Ext -eq '.exe' -and $_.Dir.ToLowerInvariant() -ne $cargoBin -and $_.Dir.ToLowerInvariant() -notlike '*\.cargo\bin' -and $_.Dir -notlike 'C:\codedev\*' -and $_.Dir -notlike ((Join-Path $ctx.UserHome 'AppData') + '\*') })
                # A twin under the profile's AppData (WinGet Links, WindowsApps) is not enough: the Codex sandbox users cannot read
                # %LOCALAPPDATA%, so the ~/bin copy is the one they can run.
                # Get-MachineFileHash returns $null for an unreadable file (WinGet app-execution aliases are reparse points that cannot be opened); never equal.
                $hm = if ($twin) { Get-MachineFileHash $m.Path } else { $null }
                if ($twin -and $hm -and $hm -eq (Get-MachineFileHash $twin[0].Path)) {
                    $archive.Add([pscustomobject]@{ Path = $m.Path; Why = "identical to $($twin[0].Path)" })
                }
            }
        }
        # Same name in both managed dirs with different content is reported by Test-MachinePath (PATH-SHADOW-MISMATCH) and left
        # for an owner decision: modification time alone does not say which build is the correct one.
    }
}
if (-not $SkipArchive) {
    # Extensionless bash shims (`exec "C:/.../name.exe" "$@"`) are written for Git Bash, but PowerShell lists them first and then fails
    # with "Cannot run a document in the middle of a pipeline" as soon as the same-named .cmd that used to mask them is archived.
    foreach ($d in $policy.ManagedBinDirs) {
        if (-not [IO.Directory]::Exists($d)) { continue }
        foreach ($f in [IO.Directory]::EnumerateFiles($d)) {
            if ([IO.Path]::GetExtension($f) -ne '' -or ([IO.FileInfo]::new($f)).Length -gt 2048) { continue }
            $txt = Get-Content -LiteralPath $f -Raw -ErrorAction SilentlyContinue
            if ($txt -match '(?s)^#!\S*\s*\r?\n\s*exec\s+"(?<t>[A-Za-z]:/[^"]+?\.exe)"\s+"\$@"\s*$' -and
                ([IO.Path]::GetFileNameWithoutExtension($Matches['t']).ToLowerInvariant() -eq [IO.Path]::GetFileName($f).ToLowerInvariant()) -and
                [IO.File]::Exists($Matches['t'].Replace('/', '\'))) {
                # The target must stay reachable under the same name: once this shim (and any wrapper already queued for archival) is
                # gone, the first runnable candidate on PATH has to BE the target. A target outside PATH would lose its command.
                $shimName = [IO.Path]::GetFileName($f).ToLowerInvariant()
                $queued = @($archive | ForEach-Object { $_.Path })
                $remaining = if ($index.ContainsKey($shimName)) { @($index[$shimName] | Where-Object { $_.Ext -in '.com', '.exe', '.bat', '.cmd' -and $queued -notcontains $_.Path } | ForEach-Object { $_.Path }) } else { @() }
                if (-not (Test-MachineShimReplacement -Target $Matches['t'] -Candidate $remaining)) { continue }
                $archive.Add([pscustomobject]@{ Path = $f; Why = "bash pass-through to $($Matches['t'])" })
            }
        }
    }
}
$archive = @($archive | Sort-Object Path -Unique)
foreach ($a in $archive) {
    $ref = Test-Referenced -Path $a.Path
    $a | Add-Member -NotePropertyName Referenced -NotePropertyValue $ref -Force
    '  archive{0} {1} ({2})' -f $(if ($ref) { ' SKIP(referenced)' } else { '' }), $a.Path.Replace($ctx.UserHome, '~'), $a.Why
}

# ---- 3. wrappers that break in agent homes -----------------------------------------------------------------------
function Get-WrapperRewrite {
    # %USERPROFILE%\bin\ becomes %~dp0 inside ~/bin; any other %USERPROFILE% or %HOME% becomes the real profile path.
    param([string]$Path)
    $text = [IO.File]::ReadAllText($Path)
    $inHomeBin = (Split-Path $Path -Parent).TrimEnd('\') -eq (Join-Path $ctx.UserHome 'bin')
    $t = if ($inHomeBin) { [regex]::Replace($text, '(?i)%(USERPROFILE|HOME)%\\bin\\', '%~dp0') } else { $text }
    [pscustomobject]@{ Old = $text; New = [regex]::Replace($t, '(?i)%(USERPROFILE|HOME)%', { $ctx.UserHome }) }
}
$wrapperFixes = @()
# A REAL agent home (the current one when redirected, else the newest on this host): only paths that exist in the real
# profile and not there are broken. Also used for the ground-truth lookups.
$agentHome = if ($env:USERPROFILE -ne $ctx.UserHome) { $env:USERPROFILE } else {
    Get-ChildItem (Join-Path $ctx.LocalAppData 'ClaudeCode\agent-homes') -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 | ForEach-Object { Join-Path $_.FullName 'home' }
}
if (-not $SkipWrappers) {
    if (-not $agentHome -or -not (Test-Path -LiteralPath $agentHome)) { 'wrapper check skipped: no agent home found on this host' }
    else {
        $wrapperFixes = @(Test-MachinePathWrappers -Directories $policy.ManagedBinDirs -Context $ctx -AgentHome $agentHome | Where-Object Code -eq 'PATH-WRAPPER-ENVREL')
        foreach ($w in $wrapperFixes) {
            if ($w.Subject -like '*.ps1') { '  wrapper MANUAL (ps1) {0}' -f $w.Subject.Replace($ctx.UserHome, '~'); continue }
            $rw = Get-WrapperRewrite -Path $w.Subject
            '  wrapper fix {0}' -f $w.Subject.Replace($ctx.UserHome, '~')
            $o = $rw.Old -split "\r?\n"; $n = $rw.New -split "\r?\n"
            for ($k = 0; $k -lt $o.Count; $k++) { if ($o[$k] -ne $n[$k]) { "    - $($o[$k])"; "    + $($n[$k])" } }
        }
    }
}
if (-not $SkipArchive) {
    if ($script:refCorpus) { "reference corpus: $($script:refCorpus.Sources) sources, $($script:refCorpus.Text.Length) chars, $($script:refCorpus.Failures.Count) unreadable" }
    # Positive control: the live Claude config always names the canonical agent-bus binary.
    "reference control (agent-bus-mcp.exe is referenced): $(Test-Referenced -Path (Join-Path $ctx.UserHome 'bin\agent-bus-mcp.exe'))"
}

if (-not $Apply) { 'Dry run. Re-run with -Apply to execute.'; return }

# ---- apply --------------------------------------------------------------------------------------------------------
$backups = Backup-MachinePath -Directory $BackupDir -HostName $hostName
'backups: ' + ($backups -join '; ')
'revert user PATH:    pwsh -NoProfile -Command "$l=(Get-Content ''{0}'')[1]; Set-ItemProperty HKCU:\Environment Path $l -Type ExpandString"' -f ($backups | Where-Object { $_ -like '*.user.*' })
$baselineDirs = @(@($user.Entries + $machine.Entries) | ForEach-Object { Resolve-MachinePathEntry -Entry $_ -Context $ctx } | Where-Object { [IO.Directory]::Exists($_) } | Select-Object -Unique)
$before = @(Get-MachineResolutionBaseline -Commands $policy.BaselineCommands -Directories $baselineDirs)
$before | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $BackupDir "$hostName.baseline.before.json") -Encoding utf8NoBOM


function Get-GroundTruth {
    # Real lookups from a FRESH pwsh whose PATH is the current registry composition (user first), optionally in an agent-home
    # environment: Get-Command -All for every policy command, and the System32 where.exe (not the ~/bin one).
    param([string]$AgentHome)
    $u = Get-MachinePathRegistry -Scope User; $m = Get-MachinePathRegistry -Scope Machine
    $composed = [Environment]::ExpandEnvironmentVariables(((@($u.Entries) + @($m.Entries)) -join ';'))
    $names = ($policy.BaselineCommands | ForEach-Object { "'$_'" }) -join ','
    $script = "`$ErrorActionPreference='SilentlyContinue'; @($names) | ForEach-Object { `$n=`$_; [pscustomobject]@{ Name=`$n; All=@(Get-Command `$n -All | ForEach-Object { `$_.Source ?? `$_.Name }) } } | ConvertTo-Json -Depth 4 -Compress"
    $psi = [Diagnostics.ProcessStartInfo]::new((Get-Command pwsh).Source)
    foreach ($a in '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', $script) { $psi.ArgumentList.Add($a) }
    $psi.RedirectStandardOutput = $true; $psi.UseShellExecute = $false
    $psi.Environment['PATH'] = $composed
    if ($AgentHome) { $psi.Environment['USERPROFILE'] = $AgentHome; $psi.Environment['HOME'] = $AgentHome }
    $proc = [Diagnostics.Process]::Start($psi)
    $out = $proc.StandardOutput.ReadToEnd(); [void]$proc.WaitForExit(180000)
    $out | ConvertFrom-Json
}
function Save-GroundTruth {
    param([string]$Tag)
    $sets = [ordered]@{ normal = (Get-GroundTruth) }
    if ($agentHome -and (Test-Path -LiteralPath $agentHome)) { $sets['agenthome'] = (Get-GroundTruth -AgentHome $agentHome) }
    $sets['where-system32'] = @(& "$env:SystemRoot\System32\where.exe" git, gh, pwsh, python, uv, cargo, rustc, rg, node, npm, agent-bus 2>$null)
    $sets | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $BackupDir "$hostName.groundtruth.$Tag.json") -Encoding utf8NoBOM
    $sets
}

function Move-ToSuperseded {
    param([string]$Path, [string]$LinkTo)
    $dir = Split-Path $Path -Parent
    $sup = Join-Path $dir '.superseded'
    New-Item -ItemType Directory -Force -Path $sup | Out-Null
    $hash = & $sha12 $Path
    $name = [IO.Path]::GetFileNameWithoutExtension($Path); $ext = [IO.Path]::GetExtension($Path)
    $dest = Join-Path $sup ('{0}-{1}{2}' -f $name, $hash, $ext)
    if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $Path -Force } else { Move-Item -LiteralPath $Path -Destination $dest }
    if ($LinkTo) { New-Item -ItemType HardLink -Path $Path -Target $LinkTo | Out-Null }
    $dest
}

$truthBefore = Save-GroundTruth -Tag before   # before any change, in a fresh pwsh and in an agent-home environment
foreach ($a in $archive) {
    if ($a.Referenced -and -not $a.LinkTo) { $actions.Add([pscustomobject]@{ Action = 'skip-referenced'; Path = $a.Path }); continue }
    try {
        $dest = Move-ToSuperseded -Path $a.Path -LinkTo $(if ($a.Referenced) { $a.LinkTo })
        $actions.Add([pscustomobject]@{ Action = 'archived'; Path = $a.Path; To = $dest; Why = $a.Why })
    } catch { $actions.Add([pscustomobject]@{ Action = 'archive-failed'; Path = $a.Path; Error = $_.Exception.Message }) }
}

foreach ($w in $wrapperFixes) {
    $p = $w.Subject
    # A textual rewrite of $HOME is unsafe inside PowerShell expressions, so .ps1 files are reported, not rewritten.
    if ($p -like '*.ps1') { $actions.Add([pscustomobject]@{ Action = 'wrapper-needs-manual-fix'; Path = $p }); continue }
    $rw = Get-WrapperRewrite -Path $p
    if ($rw.New -ne $rw.Old) {
        $dest = Move-ToSuperseded -Path $p
        [IO.File]::WriteAllText($p, $rw.New, [Text.UTF8Encoding]::new($false))
        $actions.Add([pscustomobject]@{ Action = 'wrapper-rewritten'; Path = $p; Original = $dest })
    }
}

foreach ($e in $policy.RequiredUserEnv.PSObject.Properties) {
    if ([Environment]::GetEnvironmentVariable($e.Name, 'User') -ne $e.Value) {
        [Environment]::SetEnvironmentVariable($e.Name, $e.Value, 'User'); $actions.Add([pscustomobject]@{ Action = 'user-env-set'; Name = $e.Name; Value = $e.Value })
    }
}
if ($userChanged) { Set-MachinePathValue -Scope User -Entries $plan.UserEntries; $actions.Add([pscustomobject]@{ Action = 'user-path-written'; Entries = $plan.UserEntries.Count }) }
if ($machineChanged) { Set-MachinePathValue -Scope Machine -Entries $plan.MachineEntries; $actions.Add([pscustomobject]@{ Action = 'machine-path-written'; Entries = $plan.MachineEntries.Count }) }
Send-MachineEnvironmentChange

$userAfter = Get-MachinePathRegistry -Scope User
$machineAfter = Get-MachinePathRegistry -Scope Machine
$afterDirs = @(@($userAfter.Entries + $machineAfter.Entries) | ForEach-Object { Resolve-MachinePathEntry -Entry $_ -Context $ctx } | Where-Object { [IO.Directory]::Exists($_) } | Select-Object -Unique)
$after = @(Get-MachineResolutionBaseline -Commands $policy.BaselineCommands -Directories $afterDirs)
$after | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $BackupDir "$hostName.baseline.after.json") -Encoding utf8NoBOM
$truthAfter = Save-GroundTruth -Tag after
foreach ($set in 'normal', 'agenthome') {
    if (-not $truthBefore.Contains($set) -or -not $truthAfter.Contains($set)) { continue }
    $b = @{}; foreach ($r in $truthBefore[$set]) { $b[$r.Name] = @($r.All) }
    $changed = @($truthAfter[$set] | Where-Object { (@($_.All)[0]) -ne ($b[$_.Name] | Select-Object -First 1) })
    "ground truth [$set] (fresh pwsh, Get-Command -All): $($changed.Count) of $($truthAfter[$set].Count) commands resolve differently"
    $changed | ForEach-Object { '  {0}: {1} -> {2}' -f $_.Name, ($b[$_.Name] | Select-Object -First 1), (@($_.All)[0]) }
    $lost = @($truthAfter[$set] | Where-Object { @($_.All).Count -eq 0 -and @($b[$_.Name]).Count -gt 0 })
    if ($lost) { Write-Error "Commands lost in ${set}: $($lost.Name -join ', '). Revert with the command below." }
}
$diff = @(Compare-MachineResolutionBaseline -Before $before -After $after)
$actions | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $BackupDir "$hostName.actions.json") -Encoding utf8NoBOM
$actions | Group-Object Action | ForEach-Object { 'applied {0}: {1}' -f $_.Name, $_.Count }
'baseline diff ({0} of {1} commands changed):' -f $diff.Count, $before.Count
$diff | ForEach-Object { '  {0} {1}: {2} -> {3}' -f $_.Status, $_.Command, $_.Before, $_.After }
'backups: ' + ($backups -join '; ')
'revert user PATH:    pwsh -NoProfile -Command "$l=(Get-Content ''{0}'')[1]; Set-ItemProperty HKCU:\Environment Path $l -Type ExpandString"' -f ($backups | Where-Object { $_ -like '*.user.*' })
if ($diff | Where-Object Status -eq 'LOST') { Write-Error 'A baseline command no longer resolves. Revert with the command above.'; exit 2 }
