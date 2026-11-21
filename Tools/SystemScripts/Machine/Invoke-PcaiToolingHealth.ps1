#Requires -Version 7.2

<#
.SYNOPSIS
Validates the PowerShell profile chain, the ~/bin tool shims and the repositories behind the tooling.
.DESCRIPTION
Runs from a scheduled task or systemd user timer, never from the profile, so it cannot slow an
interactive shell. It checks:

- Profile chain: every profile file in the chain exists and parses; the deployed PC-AI shim
  matches Config/PowerShell/Profile in a PC-AI checkout; a child pwsh loads the chain within the
  time budget (skipped by -Quick).
- Tool shims: each tool resolves on PATH, answers --help, and any shadowing copies further down
  PATH are reported with their hashes (LEARNED_RULES 30: identify by path and --help, never by
  --version alone).
- Repositories (dtm-claude, dtm-codex, PC-AI, google-access, agent-hub by default): fetch, then
  fast-forward a checkout only when it is on its default branch, has no tracked changes, no
  operation in progress and no local commits. Everything else is reported, never changed.
  Linked worktrees, stale local branches and your open pull requests are listed for action.
  Nothing is deleted.

Writes status.json and posts a compact summary to agent-bus when the result changes.
.PARAMETER Quick
Skip network fetch, pull-request listing and the timed profile load (well under 2 s).
.PARAMETER NoFastForward
Report what would be fast-forwarded without merging.
.PARAMETER DryRun
Collect and print only: no fetch, no merge, no files, no bus.
.EXAMPLE
pwsh -NoProfile -File Tools/SystemScripts/Machine/Invoke-PcaiToolingHealth.ps1 -Quick -DryRun
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet('Manual', 'Logon', 'Boot', 'Event', 'Timer', 'Systemd')][string]$Trigger = 'Manual',
    [string]$StatusRoot,
    [string]$ConfigPath,
    [ValidateRange(0, 86400)][int]$MinIntervalSeconds = 600,
    [switch]$Quick,
    [switch]$NoFastForward,
    [switch]$NoBus,
    [switch]$ForceNotify,
    [switch]$DryRun,
    [switch]$PassThru,
    [Alias('h')][switch]$Help,
    # PowerShell cannot alias `--help`; it arrives here and is honored below (AGENTS.md: -h, --help).
    [Parameter(ValueFromRemainingArguments, DontShow)][string[]]$RemainingArguments = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:PcaiToolingSchema = 'pcai.tooling-health-status/v1'

function Get-PcaiToolingHostName {
    [CmdletBinding()]
    param()
    $name = if ($env:COMPUTERNAME) { $env:COMPUTERNAME } else { [System.Net.Dns]::GetHostName() }
    return ($name -split '\.')[0].ToLowerInvariant()
}

function Invoke-PcaiToolingCommand {
    <# Runs a native command with a deadline; never throws for a non-zero exit. #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$FilePath, [string[]]$ArgumentList = @(), [ValidateRange(1, 900)][int]$TimeoutSeconds = 30, [string]$WorkingDirectory, [string[]]$UnsetEnvironment = @(), [hashtable]$Environment = @{})
    $psi = [Diagnostics.ProcessStartInfo]::new($FilePath)
    foreach ($a in $ArgumentList) { $psi.ArgumentList.Add($a) }
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.RedirectStandardInput = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.Environment['GIT_TERMINAL_PROMPT'] = '0'
    # Read-only inspection must not refresh .git/index (git status otherwise rewrites stat data and
    # takes index.lock, even in -DryRun, and can collide with an interactive git command).
    $psi.Environment['GIT_OPTIONAL_LOCKS'] = '0'
    foreach ($name in $UnsetEnvironment) { [void]$psi.Environment.Remove($name) }
    foreach ($name in $Environment.Keys) { $psi.Environment[$name] = [string]$Environment[$name] }
    $psi.Environment['GCM_INTERACTIVE'] = 'never'
    if ($WorkingDirectory) { $psi.WorkingDirectory = $WorkingDirectory }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try { $p = [Diagnostics.Process]::Start($psi) } catch {
        return [pscustomobject]@{ ExitCode = -1; Stdout = ''; Stderr = $_.Exception.Message; TimedOut = $false; Seconds = 0 }
    }
    $p.StandardInput.Close()
    $out = $p.StandardOutput.ReadToEndAsync()
    $err = $p.StandardError.ReadToEndAsync()
    $timedOut = -not $p.WaitForExit($TimeoutSeconds * 1000)
    if ($timedOut) { try { $p.Kill($true) } catch { Write-Verbose "kill failed: $_" }; $null = $p.WaitForExit(5000) }
    return [pscustomobject]@{
        ExitCode = if ($timedOut) { -2 } else { $p.ExitCode }
        Stdout   = $out.GetAwaiter().GetResult(); Stderr = $err.GetAwaiter().GetResult()
        TimedOut = $timedOut; Seconds = [math]::Round($sw.Elapsed.TotalSeconds, 2)
    }
}

function Read-PcaiToolingConfig {
    [CmdletBinding()]
    param([string]$Path)
    $codedev = if ($IsWindows) { 'C:\codedev' } else { Join-Path $HOME 'codedev' }
    $roots = @($codedev)
    if ($IsLinux) { $roots += @("/srv/workspaces/$env:USER/repos", (Join-Path $HOME 'repos')) }
    $config = [ordered]@{
        repoNames            = @('dtm-claude', 'dtm-codex', 'PC-AI', 'google-access', 'agent-hub')
        repoRoots            = $roots
        repoPaths            = @((Join-Path $HOME '.claude'), (Join-Path $HOME '.codex'))
        tools                = if ($IsWindows) { @('agent-bus', 'pcai-tb', 'git', 'gh', 'rg', 'uv', 'rclone', 'pwsh', 'sccache', 'ssh') } else { @('agent-bus', 'pcai-tb', 'git', 'gh', 'rg', 'uv', 'pwsh', 'ssh') }
        profileLoadBudgetSec = 5
        busEnabled           = $true
        busThread            = 'machine-hygiene'
    }
    if ($Path -and (Test-Path -LiteralPath $Path)) {
        $raw = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable
        foreach ($key in @($raw.Keys)) {
            if (-not $config.Contains($key)) { throw "Unknown tooling-health configuration key '$key' in $Path" }
            $config[$key] = $raw[$key]
        }
    }
    return [pscustomobject]$config
}

function Get-PcaiProfileChain {
    [CmdletBinding()]
    param()
    $paths = [Collections.Generic.List[string]]::new()
    # $PROFILE.CurrentUserCurrentHost: the shim pwsh actually loads (OneDrive Documents when
    # Known Folder Move is on). The other Documents copy is reported but optional.
    $active = $PROFILE.CurrentUserCurrentHost
    if ($active) { $paths.Add($active) }
    if ($IsWindows) {
        $docs = [Environment]::GetFolderPath('MyDocuments')
        foreach ($p in @((Join-Path $docs 'PowerShell\Microsoft.PowerShell_profile.ps1'), (Join-Path $HOME 'Documents\PowerShell\Microsoft.PowerShell_profile.ps1'),
                (Join-Path $HOME 'OneDrive\Documents\PowerShell\Microsoft.PowerShell_profile.ps1'))) { if ($p -notin $paths) { $paths.Add($p) } }
    }
    $private = Join-Path $HOME '.config/powershell/Microsoft.PowerShell_profile.ps1'
    if ($private -ne $active) { $paths.Add($private) }
    return , @(foreach ($p in $paths) {
            $exists = Test-Path -LiteralPath $p -PathType Leaf
            $errors = $null
            if ($exists) { $tokens = $null; [void][Management.Automation.Language.Parser]::ParseFile($p, [ref]$tokens, [ref]$errors) }
            [pscustomobject]@{
                Path        = $p
                Exists      = $exists
                ParseErrors = if ($exists) { @($errors).Count } else { $null }
                Sha256      = if ($exists) { (Get-FileHash -LiteralPath $p).Hash.ToLowerInvariant() } else { $null }
                Role        = if ($p -like '*.config*') { 'private' } else { 'shim' }
                Required    = ($p -eq $active) -or ($p -like '*.config*')
            }
        })
}

function Measure-PcaiProfileLoad {
    <# Starts a child pwsh that loads the profile chain and exits; returns seconds and outcome. #>
    [CmdletBinding()]
    param([int]$TimeoutSeconds = 60)
    $pwsh = Join-Path $PSHOME $(if ($IsWindows) { 'pwsh.exe' } else { 'pwsh' })
    # A profile that fails at runtime still lets `exit 0` succeed, so the child reports the errors
    # the chain left in $Error and exits 3 when there are any.
    $probe = 'if ($Error.Count) { $Error | Select-Object -Last 3 | ForEach-Object { [Console]::Error.WriteLine("profile error: " + $_.ToString()) }; exit 3 }; exit 0'
    # The profile shim returns early for SSH command sessions (it keeps ssh transfers clean), so a
    # probe started from an SSH session would never load the private profile. Clear the SSH
    # variables for the child so it loads the full chain, as a local terminal does.
    $r = Invoke-PcaiToolingCommand -FilePath $pwsh -ArgumentList @('-NoLogo', '-Command', $probe) -TimeoutSeconds $TimeoutSeconds `
        -UnsetEnvironment @('SSH_CONNECTION', 'SSH_CLIENT', 'SSH_TTY')
    $firstErr = (@($r.Stderr -split "`n" | Where-Object { $_.Trim() }) | Select-Object -First 1)
    return [pscustomobject]@{ Seconds = $r.Seconds; ExitCode = $r.ExitCode; TimedOut = $r.TimedOut; FirstError = $firstErr }
}

function Test-PcaiHelpOutcome {
    <#
    .SYNOPSIS
    Decides whether what a tool printed for --help (its first few lines) shows a working tool. Pure function.
    .DESCRIPTION
    Printing something is not proof of health: a missing shared library or a shim traceback prints
    a line too. A failure diagnostic is never healthy. Otherwise exit 0 is healthy, and a nonzero
    exit is accepted only when the line reads as a usage or help message, because some tools
    (several Windows CLIs and wrappers) exit 1 or 2 from --help.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Line, $ExitCode)
    $diagnostic = '(?i)error while loading shared libraries|cannot open shared object|traceback \(most recent|unhandled exception|the code execution cannot proceed|is not recognized as|no such file or directory|exec format error|bad cpu type|cannot execute|permission denied'
    if ($Line -match $diagnostic) { return $false }
    if ($ExitCode -eq 0) { return $true }
    return ($Line -match '(?i)\b(usage|options|commands|synopsis|help)\b')
}

function Test-PcaiToolShim {    <#
    .SYNOPSIS
    Resolves each tool on PATH, runs every `--help` concurrently, and reports shadowing copies.
    .DESCRIPTION
    LEARNED_RULES 30: ~/bin and ~/.local/bin shadow system commands at different versions, and a
    proxied --version can name another product. A tool is identified by its resolved path, its
    --help banner and, for shadowing copies, by size and SHA-256.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string[]]$Name, [int]$TimeoutSeconds = 10, [switch]$ResolveOnly)
    $entries = foreach ($n in $Name) {
        $all = @(Get-Command $n -CommandType Application -All -ErrorAction SilentlyContinue | ForEach-Object Source | Select-Object -Unique)
        $proc = $null; $out = $null; $err = $null
        # gh, rclone and pwsh each take 1.5-2 s to print --help on dtm-p1gen9, so -Quick only resolves.
        if ($all -and -not $ResolveOnly) {
            $psi = [Diagnostics.ProcessStartInfo]::new($all[0])
            $psi.ArgumentList.Add('--help')
            $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.RedirectStandardInput = $true
            $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true
            try { $proc = [Diagnostics.Process]::Start($psi); $proc.StandardInput.Close(); $out = $proc.StandardOutput.ReadToEndAsync(); $err = $proc.StandardError.ReadToEndAsync() } catch { $proc = $null }
        }
        [pscustomobject]@{ Name = $n; All = $all; Proc = $proc; Out = $out; Err = $err }
    }
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $hashOf = { param($p) (Get-FileHash -LiteralPath $p).Hash.Substring(0, 12).ToLowerInvariant() }
    return , @(foreach ($e in $entries) {
            if (-not $e.All) { [pscustomobject]@{ Name = $e.Name; Path = $null; Hash = $null; Ok = $false; Help = 'not found on PATH'; Shadowed = @() }; continue }
            $line = $null; $exitCode = $null; $sample = $null
            if ($e.Proc) {
                $left = [math]::Max(0, ($deadline - [DateTime]::UtcNow).TotalMilliseconds)
                if ($e.Proc.WaitForExit([int]$left)) {
                    # Classify on the first few lines: ssh prints an option complaint before its usage line.
                    $lines = @(($e.Out.GetAwaiter().GetResult() + "`n" + $e.Err.GetAwaiter().GetResult()) -split "`n" | Where-Object { $_.Trim() } | Select-Object -First 6)
                    $line = $lines | Select-Object -First 1
                    $sample = $lines -join "`n"
                    $exitCode = $e.Proc.ExitCode
                } else { try { $e.Proc.Kill($true) } catch { Write-Verbose "kill: $_" } }
            }
            $first = $e.All[0]
            $firstItem = Get-Item -LiteralPath $first
            # Hash only when a shadowing copy has the same size; a size difference already proves
            # a different binary, and hashing every large tool would cost seconds.
            $firstHash = $null
            $shadowed = @($e.All | Select-Object -Skip 1 | Select-Object -First 4 | ForEach-Object {
                    $i = Get-Item -LiteralPath $_
                    $same = $false
                    if ($i.Length -eq $firstItem.Length) {
                        if (-not $firstHash) { $firstHash = & $hashOf $first }
                        $same = (& $hashOf $_) -eq $firstHash
                    }
                    [pscustomobject]@{ Path = $_; Length = $i.Length; SameAsFirst = $same }
                })
            [pscustomobject]@{
                Name = $e.Name; Path = $first; Length = $firstItem.Length; ExitCode = $exitCode
                Ok = [bool]($ResolveOnly -or ($line -and (Test-PcaiHelpOutcome -Line $sample -ExitCode $exitCode)))
                Help = if ($line) { $line.Trim().Substring(0, [math]::Min(100, $line.Trim().Length)) } elseif ($ResolveOnly) { 'resolved (quick: --help not run)' } else { 'no --help output within the deadline' }
                Shadowed = $shadowed
            }
        })
}

function Get-PcaiRemoteSlug {
    [CmdletBinding()]
    param([string]$Url)
    if ($Url -match 'github\.com[:/](?<o>[^/]+)/(?<r>[^/]+?)(\.git)?/?$') { return "$($Matches.o)/$($Matches.r)" }
    return $null
}

function Get-PcaiGitOriginUrl {
    <# The url of [remote "origin"] in a git config file's text. Pure function. #>
    [CmdletBinding()]
    param([AllowEmptyString()][string]$ConfigText)
    $inOrigin = $false
    foreach ($line in ($ConfigText -split "`r?`n")) {
        $t = $line.Trim()
        if ($t -match '^\[(.+)\]$') { $inOrigin = $Matches[1] -match '^remote\s+"origin"$'; continue }
        if ($inOrigin -and $t -match '^url\s*=\s*(?<u>.+)$') { return $Matches.u.Trim() }
    }
    return $null
}

function Invoke-PcaiParallel {
    <# Runs a script block per item in parallel runspaces with this file's functions loaded. #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][object[]]$Item, [Parameter(Mandatory)][scriptblock]$Body, [int]$Throttle = 8, [hashtable]$Argument = @{})
    $functions = @('Invoke-PcaiToolingCommand', 'Get-PcaiRepoState') | ForEach-Object { "function $_ {`n$((Get-Item "function:$_").Definition)`n}" }
    $prelude = ($functions -join "`n")
    $bodyText = $Body.ToString()
    return , @($Item | ForEach-Object -ThrottleLimit $Throttle -Parallel {
            Set-StrictMode -Version Latest
            $ErrorActionPreference = 'Stop'
            . ([scriptblock]::Create($using:prelude))
            $arg = $using:Argument
            & ([scriptblock]::Create($using:bodyText)) $_ $arg
        })
}

function Find-PcaiToolingRepo {
    <# Primary checkouts (a .git directory, not a linked worktree) whose origin names a watched repo. #>
    [CmdletBinding()]
    param([Parameter(Mandatory)]$Config)
    $candidates = [Collections.Generic.List[string]]::new()
    foreach ($p in @($Config.repoPaths)) { if ($p) { $candidates.Add($p) } }
    foreach ($root in @($Config.repoRoots)) {
        if ($root -and (Test-Path -LiteralPath $root)) { foreach ($d in @(Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue)) { $candidates.Add($d.FullName) } }
    }
    $found = [Collections.Generic.List[object]]::new()
    foreach ($c in ($candidates | Select-Object -Unique)) {
        $configPath = Join-Path $c '.git/config'
        if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) { continue }
        # Reading the config file directly keeps discovery fast over a large code root (one git
        # process per directory took 14 s on dtm-p1gen9).
        $slug = Get-PcaiRemoteSlug -Url (Get-PcaiGitOriginUrl -ConfigText (Get-Content -LiteralPath $configPath -Raw))
        if (-not $slug) { continue }
        $name = ($slug -split '/')[1]
        if ($name -in @($Config.repoNames)) { $found.Add([pscustomobject]@{ Path = $c; Slug = $slug; Name = $name }) }
    }
    return , $found.ToArray()
}

function Get-PcaiRepoAction {
    <#
    .SYNOPSIS
    Decides whether a checkout may be fast-forwarded. Pure function.
    .DESCRIPTION
    Fast-forward only a clean checkout on its default branch that is strictly behind its
    remote. Anything else is reported for its owner; this check never resets, rebases,
    stashes or deletes.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)]$State)
    if (-not $State.FetchOk) { return 'FetchFailed' }
    if ($State.PSObject.Properties['InspectOk'] -and -not $State.InspectOk) { return 'InspectFailed' }
    if ($State.Busy) { return 'Busy' }
    if (-not $State.Branch) { return 'Detached' }
    if ($State.Branch -cne $State.DefaultBranch) { return 'OffDefault' }
    if ($State.TrackedChanges -gt 0) { return 'Dirty' }
    if ($State.Ahead -gt 0 -and $State.Behind -gt 0) { return 'Diverged' }
    if ($State.Ahead -gt 0) { return 'Ahead' }
    if ($State.Behind -gt 0) { return 'FastForward' }
    return 'UpToDate'
}

function Get-PcaiRepoState {
    [CmdletBinding()]
    param([Parameter(Mandatory)]$Repo, [switch]$NoFetch)
    $git = { param([string[]]$a, [int]$t = 30) Invoke-PcaiToolingCommand -FilePath 'git' -ArgumentList (@('-C', $Repo.Path) + $a) -TimeoutSeconds $t }
    $fetchOk = $true; $fetchNote = 'skipped'
    if (-not $NoFetch) {
        $f = & $git @('fetch', '--prune', '--quiet', 'origin') 120
        $fetchOk = $f.ExitCode -eq 0
        $fetchNote = if ($fetchOk) { 'ok' } else { ($f.Stderr.Trim() -split "`n")[0] }
    }
    $def = (& $git @('symbolic-ref', '--quiet', '--short', 'refs/remotes/origin/HEAD')).Stdout.Trim() -replace '^origin/', ''
    if (-not $def) { $def = if ((& $git @('rev-parse', '--verify', '--quiet', 'refs/remotes/origin/main')).ExitCode -eq 0) { 'main' } else { 'master' } }
    # Every inspection that feeds the fast-forward decision must succeed: an empty stdout from a
    # failed or timed-out git call would otherwise read as "clean" or "up to date".
    $failed = [Collections.Generic.List[string]]::new()
    $run = { param([string[]]$a) $r = & $git $a; if ($r.ExitCode -ne 0) { $failed.Add("git $($a[0]) exit $($r.ExitCode)") }; $r }
    $branch = (& $git @('branch', '--show-current')).Stdout.Trim()
    $head = (& $run @('rev-parse', 'HEAD')).Stdout.Trim()
    $target = (& $run @('rev-parse', "refs/remotes/origin/$def")).Stdout.Trim()
    $tracked = @((& $run @('status', '--porcelain=v1', '--untracked-files=no')).Stdout -split "`n" | Where-Object { $_.Trim() }).Count
    $untracked = @((& $git @('ls-files', '--others', '--exclude-standard', '--directory')).Stdout -split "`n" | Where-Object { $_.Trim() }).Count
    $counts = (& $run @('rev-list', '--left-right', '--count', "HEAD...refs/remotes/origin/$def")).Stdout.Trim() -split '\s+'
    if ($counts.Count -ne 2) { $failed.Add('git rev-list returned no ahead/behind counts') }
    $ahead = if ($counts.Count -eq 2) { [int]$counts[0] } else { 0 }
    $behind = if ($counts.Count -eq 2) { [int]$counts[1] } else { 0 }
    $gitDir = Join-Path $Repo.Path '.git'
    $busy = @('MERGE_HEAD', 'rebase-merge', 'rebase-apply', 'CHERRY_PICK_HEAD', 'REVERT_HEAD', 'index.lock' | Where-Object { Test-Path -LiteralPath (Join-Path $gitDir $_) }).Count -gt 0

    $worktrees = [Collections.Generic.List[object]]::new()
    $current = $null
    foreach ($line in ((& $git @('worktree', 'list', '--porcelain')).Stdout -split "`n")) {
        if ($line -like 'worktree *') { if ($current) { $worktrees.Add([pscustomobject]$current) }; $current = [ordered]@{ Path = $line.Substring(9).Trim(); Branch = $null; Detached = $false; Locked = $false; Prunable = $false } }
        elseif ($line -like 'branch *' -and $current) { $current.Branch = $line.Substring(7).Trim() -replace '^refs/heads/', '' }
        elseif ($line -eq 'detached' -and $current) { $current.Detached = $true }
        elseif ($line -like 'locked*' -and $current) { $current.Locked = $true }
        elseif ($line -like 'prunable*' -and $current) { $current.Prunable = $true }
    }
    if ($current) { $worktrees.Add([pscustomobject]$current) }
    $merged = @((& $git @('branch', '--format=%(refname:lstrip=2)', '--merged', "refs/remotes/origin/$def")).Stdout -split "`n" | ForEach-Object Trim | Where-Object { $_ })
    $gone = @((& $git @('for-each-ref', '--format=%(refname:lstrip=2)|%(upstream:track)', 'refs/heads')).Stdout -split "`n" |
            Where-Object { $_ -like '*|`[gone`]*' } | ForEach-Object { ($_ -split '\|')[0] })
    $staleBranches = @(@($merged + $gone) | Sort-Object -Unique -CaseSensitive | Where-Object { $_ -and $_ -cne $def -and $_ -cne $branch })
    $linked = @($worktrees | Select-Object -Skip 1 | ForEach-Object {
            $why = @()
            if ($_.Prunable) { $why += 'prunable' }
            if ($_.Branch -and $_.Branch -cin $merged) { $why += 'merged' }
            if ($_.Branch -and $_.Branch -cin $gone) { $why += 'upstream gone' }
            if ($_.Detached) { $why += 'detached' }
            [pscustomobject]@{ Path = $_.Path; Branch = $_.Branch; Locked = $_.Locked; NeedsAction = [bool]$why; Why = ($why -join ', ') }
        })
    return [pscustomobject]@{
        Path = $Repo.Path; Slug = $Repo.Slug; Name = $Repo.Name; DefaultBranch = $def; Branch = $branch; Head = $head; Target = $target
        TrackedChanges = $tracked; Untracked = $untracked; Ahead = $ahead; Behind = $behind; Busy = $busy
        FetchOk = $fetchOk; Fetch = $fetchNote; InspectOk = ($failed.Count -eq 0); InspectErrors = $failed.ToArray(); Worktrees = $linked; StaleBranches = $staleBranches
    }
}

function Test-PcaiRepoUnchanged {
    <#
    .SYNOPSIS
    Re-reads a checkout immediately before a fast-forward and confirms it still matches the
    sampled state: same branch, same HEAD, no tracked changes and no operation in progress.
    `git merge --ff-only` only refuses a non-fast-forward; it does not check the branch.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)]$State)
    $git = { param([string[]]$a) Invoke-PcaiToolingCommand -FilePath 'git' -ArgumentList (@('-C', $State.Path) + $a) -TimeoutSeconds 30 }
    $b = & $git @('branch', '--show-current')
    $h = & $git @('rev-parse', 'HEAD')
    $st = & $git @('status', '--porcelain=v1', '--untracked-files=no')
    if ($b.ExitCode -ne 0 -or $h.ExitCode -ne 0 -or $st.ExitCode -ne 0) { return 'inspection failed' }
    if ($b.Stdout.Trim() -cne $State.Branch) { return 'branch changed' }
    if ($h.Stdout.Trim() -ne $State.Head) { return 'HEAD moved' }
    if (@($st.Stdout -split "`n" | Where-Object { $_.Trim() }).Count) { return 'tracked changes appeared' }
    $gitDir = Join-Path $State.Path '.git'
    if (@('MERGE_HEAD', 'rebase-merge', 'rebase-apply', 'CHERRY_PICK_HEAD', 'REVERT_HEAD', 'index.lock' | Where-Object { Test-Path -LiteralPath (Join-Path $gitDir $_) }).Count) { return 'operation in progress' }
    return $null
}

function Get-PcaiOpenPullRequest {
    [CmdletBinding()]
    param([string[]]$Slug)
    $gh = Get-Command gh -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $gh) { return , @() }
    return , @(foreach ($s in ($Slug | Select-Object -Unique)) {
            $r = Invoke-PcaiToolingCommand -FilePath $gh.Source -ArgumentList @('pr', 'list', '--repo', $s, '--author', '@me', '--state', 'open', '--json', 'number,title,headRefName,isDraft,updatedAt', '--limit', '50') -TimeoutSeconds 30
            if ($r.ExitCode -ne 0) { [pscustomobject]@{ Repo = $s; Error = ($r.Stderr.Trim() -split "`n")[0]; Items = @() }; continue }
            [pscustomobject]@{ Repo = $s; Error = $null; Items = @($r.Stdout | ConvertFrom-Json) }
        })
}

function Invoke-PcaiFastForward {
    <#
    .SYNOPSIS
    Fast-forwards one checkout to the commit that was sampled, and reports any ref this run moved
    other than the sampled branch.
    .DESCRIPTION
    Revalidation and `git merge --ff-only` are separate commands, and merge advances whichever
    branch is checked out when it runs. If a person switches branch, or detaches HEAD, inside that
    window, the checks pass and that ref moves; they may even switch back afterwards. Git offers no
    way to pin the merge to a branch, so the merge leaves its own evidence instead: it runs with
    GIT_REFLOG_ACTION set to a nonce unique to this run and with core.logAllRefUpdates forced on, so
    every ref it moves gets a reflog entry carrying the nonce even where the repository disables
    reflogs. Afterwards one `git log -g --all` lists those entries, and the outcome is decided by
    what moved, not by what is checked out now:
      - only refs/heads/<sampled branch>: FastForwarded;
      - nothing: Changed (the merge was refused, or a no-op because the ref already held the target);
      - any other branch, or a detached HEAD: FastForwardMisapplied (severity warn), naming each
        ref, its tip before this run's merge and the command that undoes the move.
    Nothing is reset automatically: any reset acts on whatever is checked out when it runs, and the
    person who switched may switch again (codex reviews of PR 345 at f5f39c97 and 7c4e7735).
    Commits are only added to the advanced ref, never lost. Branch names compare case-sensitively,
    as Git does. -BeforeMerge and -AfterMerge run on either side of the merge so tests can open the
    window deterministically.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)]$State, [scriptblock]$BeforeMerge, [scriptblock]$AfterMerge)
    $nonce = "pcai-tooling-health fast-forward $([guid]::NewGuid().ToString('n'))"
    $git = { param([string[]]$a, [int]$t = 120, [hashtable]$e = @{}) Invoke-PcaiToolingCommand -FilePath 'git' -ArgumentList (@('-C', $State.Path) + $a) -TimeoutSeconds $t -Environment $e }
    $changedSince = Test-PcaiRepoUnchanged -State $State
    if ($changedSince) { return [pscustomobject]@{ Action = 'Changed'; Note = "not fast-forwarded: $changedSince since sampling" } }
    if ($BeforeMerge) { & $BeforeMerge }
    # Merge the exact commit that was sampled, not whatever the ref points at now. An ignored local
    # file that the target starts tracking is never overwritten: its contents were never committed,
    # so nothing could recover them (codex review of PR 345 at 57f35770).
    $m = & $git @('-c', 'core.logAllRefUpdates=true', 'merge', '--ff-only', '--no-overwrite-ignore', '--quiet', $State.Target) 120 @{ GIT_REFLOG_ACTION = $nonce }
    if ($AfterMerge) { & $AfterMerge }
    $log = & $git @('log', '--walk-reflogs', '--all', '-n', '500', '--format=%gD%x09%H%x09%gs') 60
    if ($log.ExitCode -ne 0) {
        return [pscustomobject]@{ Action = 'FastForwardMisapplied'; Note = "the reflog could not be read after the merge (exit $($log.ExitCode)), so what this run moved is unknown; check $($State.Branch) and the checked-out ref" }
    }
    # Each line: <ref>@{<n>} TAB <new value> TAB <subject>. Group by ref, newest first.
    $byRef = [ordered]@{}
    foreach ($line in ($log.Stdout -split "`n")) {
        $parts = $line.TrimEnd("`r") -split "`t", 3
        if ($parts.Count -ne 3 -or $parts[0] -notmatch '^(?<ref>.+)@\{(?<n>\d+)\}$') { continue }
        $ref = $Matches.ref
        if (-not $byRef.Contains($ref)) { $byRef[$ref] = [Collections.Generic.List[object]]::new() }
        $byRef[$ref].Add([pscustomobject]@{ N = [int]$Matches.n; Hash = $parts[1]; Subject = $parts[2] })
    }
    $moved = @(foreach ($ref in $byRef.Keys) {
            $list = @($byRef[$ref] | Sort-Object N)
            for ($k = 0; $k -lt $list.Count; $k++) {
                if ($list[$k].Subject.StartsWith($nonce)) {
                    $prev = $list | Where-Object N -eq ($list[$k].N + 1) | Select-Object -First 1
                    [pscustomobject]@{ Ref = $ref; Before = if ($prev) { $prev.Hash } else { $null }; After = $list[$k].Hash }
                    break
                }
            }
        })
    # A merge on an attached HEAD logs both HEAD and the branch; HEAD alone means it was detached.
    $branches = @($moved | Where-Object { $_.Ref -cne 'HEAD' })
    if ($branches.Count) { $moved = $branches }
    $sampled = "refs/heads/$($State.Branch)"
    if ($moved.Count -eq 0) {
        $branchNow = (& $git @('branch', '--show-current') 30).Stdout.Trim()
        if ($m.ExitCode -ne 0) {
            if ($branchNow -ceq $State.Branch) { return [pscustomobject]@{ Action = 'FastForwardFailed'; Note = ($m.Stderr.Trim() -split "`n")[0] } }
            return [pscustomobject]@{ Action = 'Changed'; Note = "checkout changed during fast-forward; the merge was refused and nothing moved" }
        }
        return [pscustomobject]@{ Action = 'Changed'; Note = 'this run''s merge moved nothing (the checked-out ref already held the target)' }
    }
    if ($moved.Count -eq 1 -and $moved[0].Ref -ceq $sampled) { return [pscustomobject]@{ Action = 'FastForwarded'; Note = "behind $($State.Behind)" } }
    $short = { param([string]$x) if ($x) { $x.Substring(0, [Math]::Min(8, $x.Length)) } else { 'unknown' } }
    $q = "git -C `"$($State.Path)`""
    $notes = @(foreach ($mv in $moved) {
            $name = if ($mv.Ref -ceq 'HEAD') { 'a detached HEAD' } else { "branch '$($mv.Ref -replace '^refs/heads/', '')'" }
            $undo = if (-not $mv.Before) { 'its previous tip is not in the reflog' }
            elseif ($mv.Ref -ceq 'HEAD') { "while still detached there: $q reset --keep $($mv.Before)" }
            else { "if it is not checked out: $q update-ref $($mv.Ref) $($mv.Before) $($mv.After); if it is: $q reset --keep $($mv.Before)" }
            "$name advanced from $(& $short $mv.Before) to $(& $short $mv.After); to undo: $undo"
        })
    return [pscustomobject]@{ Action = 'FastForwardMisapplied'; Note = "this run's merge moved a ref other than $($State.Branch) and did not reset it automatically: $($notes -join ' | ')" }
}
function Format-PcaiToolingSummary {    [CmdletBinding()]
    param([Parameter(Mandatory)]$Status)
    $p = $Status.profile
    $missing = @($p.Chain | Where-Object { $_.Required -and -not $_.Exists } | ForEach-Object Path)
    $profileText = if (@($p.Chain | Where-Object { $_.Exists -and $_.ParseErrors }).Count) { 'PARSE ERRORS' } else { 'parses' }
    if ($missing) { $profileText += " MISSING $($missing -join ', ')" }
    if ($p.Load) {
        $profileText += " load=$($p.Load.Seconds)s"
        if ($p.Load.ExitCode -ne 0) { $profileText += " exit=$($p.Load.ExitCode)" + $(if ($p.Load.FirstError) { " ($($p.Load.FirstError))" } else { '' }) }
    }
    if ($null -ne $p.ShimMatchesSource) { $profileText += " shim=$(if ($p.ShimMatchesSource) { 'matches PC-AI source' } else { 'DRIFT in ' + (@($p.ShimDrift) -join ', ') })" }
    $tools = @($Status.tools)
    $bad = @($tools | Where-Object { -not $_.Ok } | ForEach-Object Name)
    $shadow = @($tools | Where-Object { @($_.Shadowed | Where-Object { -not $_.SameAsFirst }).Count } | ForEach-Object Name)
    $repos = @($Status.repos)
    $ff = @($repos | Where-Object Action -eq 'FastForwarded' | ForEach-Object { "$($_.Name)" })
    $wouldFf = @($repos | Where-Object Action -eq 'FastForward' | ForEach-Object { "$($_.Name) (behind $($_.Behind))" })
    $off = @($repos | Where-Object Action -in @('OffDefault', 'Detached', 'Dirty', 'Diverged', 'Ahead', 'Busy', 'FetchFailed', 'FastForwardFailed', 'Changed', 'InspectFailed', 'RolledBack', 'FastForwardMisapplied') | ForEach-Object { "$($_.Name)@$($_.Branch)($($_.Action))" })
    $wt = @($repos | ForEach-Object { @($_.Worktrees | Where-Object NeedsAction) }).Count
    $stale = @($repos | ForEach-Object { @($_.StaleBranches) }).Count
    $prs = @($Status.pullRequests | ForEach-Object { @($_.Items) }).Count
    $text = "TOOLING-HEALTH $($Status.host) [$($Status.trigger)] severity=$($Status.severity) profile: $profileText; tools ok $(@($tools | Where-Object Ok).Count)/$($tools.Count)"
    if ($bad) { $text += " (failing: $($bad -join ', '))" }
    if ($shadow) { $text += " (shadowed by a different binary: $($shadow -join ', '))" }
    $text += "; repos $($repos.Count): fast-forwarded [$($ff -join ', ')]"
    if ($wouldFf) { $text += ", would fast-forward [$($wouldFf -join ', ')]" }
    $text += ", needs owner [$($off -join ', ')]; worktrees needing action $wt; stale branches $stale; my open PRs $prs"
    if ($Status.quick) { $text += ' (quick: repositories not checked)' }
    return $text
}

function Invoke-PcaiToolingHealth {
    [CmdletBinding()]
    param([string]$RunTrigger = 'Manual', [string]$Root, [string]$Config, [int]$MinInterval = 600,
        [switch]$IsQuick, [switch]$SkipFastForward, [switch]$SkipBus, [switch]$Notify, [switch]$IsDryRun)
    if (-not $Root) {
        $Root = if ($IsWindows) { Join-Path $env:LOCALAPPDATA 'PC-AI\tooling-health' } else { Join-Path $(if ($env:XDG_STATE_HOME) { $env:XDG_STATE_HOME } else { Join-Path $HOME '.local/state' }) 'pcai/tooling-health' }
    }
    if (-not $Config) { $Config = Join-Path $Root 'config.json' }
    $cfg = Read-PcaiToolingConfig -Path $Config
    $hostName = Get-PcaiToolingHostName
    if (-not $IsDryRun -and -not $IsQuick -and -not $SkipBus) {
        $null = New-Item -ItemType Directory -Force -Path $Root
        $lastRun = Join-Path $Root 'last-run.txt'
        if ($RunTrigger -ne 'Manual' -and $MinInterval -gt 0 -and (Test-Path -LiteralPath $lastRun)) {
            $last = [DateTime]::Parse((Get-Content -LiteralPath $lastRun -Raw).Trim(), [Globalization.CultureInfo]::InvariantCulture, 'RoundtripKind')
            if (([DateTime]::UtcNow - $last).TotalSeconds -lt $MinInterval) { Write-Output "TOOLING-HEALTH $hostName debounced"; return $null }
        }
        Set-Content -LiteralPath $lastRun -Value ([DateTime]::UtcNow.ToString('o')) -NoNewline
    }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $chain = Get-PcaiProfileChain
    $found = Find-PcaiToolingRepo -Config $cfg
    # Loading the profile executes it, which a dry run must not do.
    $load = if ($IsQuick -or $IsDryRun) { $null } else { Measure-PcaiProfileLoad }
    $tools = Test-PcaiToolShim -Name @($cfg.tools) -ResolveOnly:$IsQuick
    # -Quick covers the profile chain and tool shims only, which stay under about 2 s; repository
    # checks need git and the network and run in the full (scheduled) pass.
    $states = if ($IsQuick -or -not $found) { @() } else {
        Invoke-PcaiParallel -Item $found -Throttle 4 -Argument @{ NoFetch = [bool]$IsDryRun } -Body { param($repo, $a) Get-PcaiRepoState -Repo $repo -NoFetch:($a.NoFetch) }
    }
    $repos = @(foreach ($state in @($states | Sort-Object Path)) {
            $repo = $state
            $action = Get-PcaiRepoAction -State $state
            $note = $null
            if ($action -eq 'FastForward' -and -not $SkipFastForward -and -not $IsDryRun) {
                $ff = Invoke-PcaiFastForward -State $state
                $action = $ff.Action
                $note = $ff.Note
            }
            $state | Add-Member -NotePropertyName Action -NotePropertyValue $action -PassThru | Add-Member -NotePropertyName Note -NotePropertyValue $note -PassThru
        })
    $prs = if ($IsQuick -or $IsDryRun) { @() } else { Get-PcaiOpenPullRequest -Slug @($found | ForEach-Object Slug) }

    # Compare the deployed shims with the PC-AI source after the fetch above, at the same default
    # branch the repository check resolved (origin/HEAD, else main, else master).
    $shimMatches = $null
    $shimDrift = @()
    $pcai = @($found | Where-Object Name -eq 'PC-AI') | Select-Object -First 1
    if ($pcai) {
        $pcaiState = @($repos | Where-Object Path -eq $pcai.Path) | Select-Object -First 1
        $ref = if ($pcaiState) { "refs/remotes/origin/$($pcaiState.DefaultBranch)" } else { 'refs/remotes/origin/HEAD' }
        $src = Invoke-PcaiToolingCommand -FilePath 'git' -ArgumentList @('-C', $pcai.Path, 'show', "${ref}:Config/PowerShell/Profile/Microsoft.PowerShell_profile.ps1") -TimeoutSeconds 10
        if ($src.ExitCode -ne 0 -and -not $pcaiState) {
            $src = Invoke-PcaiToolingCommand -FilePath 'git' -ArgumentList @('-C', $pcai.Path, 'show', 'refs/remotes/origin/main:Config/PowerShell/Profile/Microsoft.PowerShell_profile.ps1') -TimeoutSeconds 10
        }
        if ($src.ExitCode -eq 0) {
            $norm = { param([string]$t) ($t -replace "`r`n", "`n").TrimEnd() }
            $srcText = & $norm $src.Stdout
            $shims = @($chain | Where-Object { $_.Role -eq 'shim' -and $_.Exists })
            $shimDrift = @($shims | Where-Object { (& $norm (Get-Content -LiteralPath $_.Path -Raw)) -ne $srcText } | ForEach-Object Path)
            if ($shims) { $shimMatches = -not $shimDrift.Count }
        }
    }

    $severity = 'ok'
    if (@($chain | Where-Object { $_.Exists -and $_.ParseErrors }).Count -or @($tools | Where-Object { -not $_.Ok -and $_.Name -in @('git', 'pwsh') }).Count) { $severity = 'fail' }
    elseif (($load -and ($load.ExitCode -ne 0 -or $load.Seconds -gt $cfg.profileLoadBudgetSec)) -or $shimMatches -eq $false -or
        @($chain | Where-Object { $_.Required -and -not $_.Exists }).Count -or
        @($tools | Where-Object { -not $_.Ok }).Count -or @($repos | Where-Object Action -in @('FastForwardFailed', 'FetchFailed', 'InspectFailed', 'Diverged', 'RolledBack', 'FastForwardMisapplied')).Count) { $severity = 'warn' }

    $status = [ordered]@{
        schema = $script:PcaiToolingSchema; host = $hostName; evaluatedAtUtc = [DateTime]::UtcNow.ToString('o'); trigger = $RunTrigger
        quick = [bool]$IsQuick; dryRun = [bool]$IsDryRun; severity = $severity
        profile = [ordered]@{ Chain = $chain; ShimMatchesSource = $shimMatches; ShimDrift = $shimDrift; Load = $load }
        tools = $tools; repos = $repos; pullRequests = $prs; elapsedSeconds = [math]::Round($sw.Elapsed.TotalSeconds, 2)
    }
    $summary = Format-PcaiToolingSummary -Status ([pscustomobject]$status)
    $status.summary = $summary
    $facts = [ordered]@{
        sev = $severity; chain = @($chain | ForEach-Object { "$($_.Path)|$($_.Exists)|$($_.ParseErrors)" }); shim = $shimMatches
        tools = @($tools | ForEach-Object { "$($_.Name)|$($_.Path)|$($_.Ok)" })
        repos = @($repos | ForEach-Object { "$($_.Path)|$($_.Branch)|$($_.Action)|$(@($_.Worktrees | Where-Object NeedsAction).Count)|$(@($_.StaleBranches).Count)" })
        prs = @($prs | ForEach-Object { @($_.Items) | ForEach-Object number })
    }
    $status.fingerprint = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes(($facts | ConvertTo-Json -Compress -Depth 4)))).ToLowerInvariant()
    $status.bus = 'not posted'
    # A quick or no-bus run reports only: it must not overwrite the scheduled pass's status,
    # debounce stamp or notification fingerprint.
    if (-not $IsDryRun -and -not $IsQuick -and -not $SkipBus) {
        $fp = Join-Path $Root 'last-fingerprint.txt'
        $prev = if (Test-Path -LiteralPath $fp) { (Get-Content -LiteralPath $fp -Raw).Trim() } else { '' }
        $changed = $prev -ne $status.fingerprint
        if (($changed -or $Notify -or @($repos | Where-Object Action -eq 'FastForwarded').Count) -and $cfg.busEnabled) {
            $exe = if ($IsWindows) { 'agent-bus.exe' } else { 'agent-bus' }
            $bus = @((Get-Command agent-bus -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1 | ForEach-Object Source), (Join-Path $HOME "bin/$exe"), (Join-Path $HOME ".local/bin/$exe")) |
                Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
            if ($bus) {
                $r = Invoke-PcaiToolingCommand -FilePath $bus -TimeoutSeconds 30 -ArgumentList @('send', '--from-agent', "pcai-tooling-health-$hostName", '--to-agent', 'all', '--topic', 'status',
                    '--thread-id', $cfg.busThread, '--tag', 'repo:PC-AI', '--tag', "host:$hostName", '--body', $summary, '--encoding', 'compact')
                $status.bus = if ($r.ExitCode -eq 0) { 'posted' } else { "agent-bus exit $($r.ExitCode)" }
            } else { $status.bus = 'agent-bus not found' }
        } elseif (-not $changed) { $status.bus = 'unchanged' }
        $tmp = Join-Path $Root "status.json.tmp-$PID"
        [IO.File]::WriteAllText($tmp, ([pscustomobject]$status | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $tmp -Destination (Join-Path $Root 'status.json') -Force
        if ($status.bus -in @('posted', 'unchanged') -or -not $cfg.busEnabled) { Set-Content -LiteralPath $fp -Value $status.fingerprint -NoNewline }
    }
    Write-Output $summary
    return [pscustomobject]$status
}

if ($MyInvocation.InvocationName -ne '.') {
    $unknown = @($RemainingArguments | Where-Object { $_ -ne '--help' })
    if ($unknown) { throw "Unknown arguments: $($unknown -join ' '). Use -h or --help." }
    if ($Help -or $RemainingArguments -contains '--help') { Get-Help $PSCommandPath -Detailed; return }
    $result = Invoke-PcaiToolingHealth -RunTrigger $Trigger -Root $StatusRoot -Config $ConfigPath -MinInterval $MinIntervalSeconds `
        -IsQuick:$Quick -SkipFastForward:$NoFastForward -SkipBus:$NoBus -Notify:$ForceNotify -IsDryRun:$DryRun
    $statusObject = @($result | Where-Object { $_ -isnot [string] }) | Select-Object -Last 1
    @($result | Where-Object { $_ -is [string] }) | ForEach-Object { Write-Host $_ }
    if ($PassThru -and $statusObject) { $statusObject }
    if (-not $statusObject) { exit 0 }
    exit $(switch ($statusObject.severity) { 'ok' { 0 } 'warn' { 2 } default { 1 } })
}
