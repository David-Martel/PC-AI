# Controls for the profile and tooling validator (docs/usb4/link-health.md, "Tooling health").
# The fast-forward decision is the safety boundary: it must never move a checkout that is off its
# default branch, dirty, busy, ahead or diverged. These tests change no repository.

BeforeAll {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
    . (Join-Path $repoRoot 'Tools/SystemScripts/Machine/Invoke-PcaiToolingHealth.ps1')
    function New-RepoState {
        param([string]$Branch = 'main', [int]$Tracked = 0, [int]$Ahead = 0, [int]$Behind = 0, [bool]$Busy = $false, [bool]$FetchOk = $true)
        [pscustomobject]@{ Branch = $Branch; DefaultBranch = 'main'; TrackedChanges = $Tracked; Ahead = $Ahead; Behind = $Behind; Busy = $Busy; FetchOk = $FetchOk }
    }
}

Describe 'Get-PcaiRepoAction fast-forwards only clean default-branch checkouts that are strictly behind' -Tag 'Unit', 'Portable' {
    It 'fast-forwards a clean default-branch checkout that is behind' { Get-PcaiRepoAction -State (New-RepoState -Behind 3) | Should -Be 'FastForward' }
    It 'leaves an up-to-date checkout alone' { Get-PcaiRepoAction -State (New-RepoState) | Should -Be 'UpToDate' }
    It 'never moves a checkout on another branch' { Get-PcaiRepoAction -State (New-RepoState -Branch 'feat/x' -Behind 3) | Should -Be 'OffDefault' }
    It 'never moves a detached checkout' { Get-PcaiRepoAction -State (New-RepoState -Branch '' -Behind 3) | Should -Be 'Detached' }
    It 'never moves a checkout with tracked changes' { Get-PcaiRepoAction -State (New-RepoState -Tracked 1 -Behind 3) | Should -Be 'Dirty' }
    It 'never moves a checkout with a merge, rebase or index lock in progress' { Get-PcaiRepoAction -State (New-RepoState -Busy $true -Behind 3) | Should -Be 'Busy' }
    It 'reports local commits instead of merging' { Get-PcaiRepoAction -State (New-RepoState -Ahead 1) | Should -Be 'Ahead' }
    It 'reports divergence instead of merging' { Get-PcaiRepoAction -State (New-RepoState -Ahead 1 -Behind 1) | Should -Be 'Diverged' }
    It 'does not act on stale refs after a failed fetch' { Get-PcaiRepoAction -State (New-RepoState -Behind 3 -FetchOk $false) | Should -Be 'FetchFailed' }
    It 'does not act when a git inspection failed (empty output must not read as clean)' {
        $state = New-RepoState -Behind 3
        $state | Add-Member -NotePropertyName InspectOk -NotePropertyValue $false
        Get-PcaiRepoAction -State $state | Should -Be 'InspectFailed'
    }
    It 'reports a failed re-inspection instead of merging' {
        Mock Invoke-PcaiToolingCommand { [pscustomobject]@{ ExitCode = 128; Stdout = ''; Stderr = 'fatal'; TimedOut = $false; Seconds = 0 } }
        Test-PcaiRepoUnchanged -State ([pscustomobject]@{ Path = 'C:ake'; Branch = 'main'; Head = 'aaa' }) | Should -Be 'inspection failed'
    }
}

Describe 'Repository discovery helpers' -Tag 'Unit', 'Portable' {
    It 'reads the origin url from git config text without running git' {
        $text = "[core]`n`tbare = false`n[remote `"upstream`"]`n`turl = https://example.com/a.git`n[remote `"origin`"]`n`turl = https://github.com/David-Martel/PC-AI.git`n`tfetch = +refs/heads/*:refs/remotes/origin/*`n"
        Get-PcaiGitOriginUrl -ConfigText $text | Should -Be 'https://github.com/David-Martel/PC-AI.git'
    }
    It 'returns nothing when there is no origin' { Get-PcaiGitOriginUrl -ConfigText "[core]`n`tbare = false`n" | Should -BeNullOrEmpty }
    It 'parses https and ssh GitHub remotes into owner/name' {
        Get-PcaiRemoteSlug -Url 'https://github.com/David-Martel/PC-AI.git' | Should -Be 'David-Martel/PC-AI'
        Get-PcaiRemoteSlug -Url 'git@github.com:David-Martel/dtm-claude.git' | Should -Be 'David-Martel/dtm-claude'
        Get-PcaiRemoteSlug -Url 'https://example.com/x/y.git' | Should -BeNullOrEmpty
    }
}

Describe 'Summary reports every repository outcome and missing profile files' -Tag 'Unit', 'Portable' {
    BeforeAll {
        function New-Status {
            param([object[]]$Repos = @(), [object[]]$Chain = @())
            [pscustomobject]@{
                host = 'h'; trigger = 'Manual'; severity = 'warn'; quick = $false
                profile = [pscustomobject]@{ Chain = $Chain; ShimMatchesSource = $null; ShimDrift = @(); Load = $null }
                tools = @(); repos = $Repos; pullRequests = @()
            }
        }
        function New-Repo { param([string]$Name, [string]$Action) [pscustomobject]@{ Name = $Name; Branch = 'main'; Action = $Action; Behind = 2; Worktrees = @(); StaleBranches = @() } }
    }
    It 'lists a checkout that would be fast-forwarded (-NoFastForward, -DryRun)' {
        Format-PcaiToolingSummary -Status (New-Status -Repos @(New-Repo 'PC-AI' 'FastForward')) | Should -Match 'would fast-forward \[PC-AI \(behind 2\)\]'
    }
    It 'lists a failed fast-forward under needs owner' {
        Format-PcaiToolingSummary -Status (New-Status -Repos @(New-Repo 'PC-AI' 'FastForwardFailed')) | Should -Match 'needs owner \[PC-AI@main\(FastForwardFailed\)\]'
    }
    It 'names a missing required profile file' {
        $chain = @([pscustomobject]@{ Path = 'C:\x\private.ps1'; Exists = $false; ParseErrors = $null; Required = $true; Role = 'private' })
        Format-PcaiToolingSummary -Status (New-Status -Chain $chain) | Should -Match 'MISSING C:\\x\\private.ps1'
    }
    It 'marks the active profile and the private profile as required' {
        $chain = Get-PcaiProfileChain
        ($chain | Where-Object Path -eq $PROFILE.CurrentUserCurrentHost).Required | Should -BeTrue
        @($chain | Where-Object Role -eq 'private').Required | Should -Not -Contain $false
    }
}

Describe 'Invoke-PcaiToolingHealth side effects' -Tag 'Unit', 'Portable' {
    BeforeEach {
        $script:behind = [pscustomobject]@{
            Path = 'C:\fake\PC-AI'; Slug = 'David-Martel/PC-AI'; Name = 'PC-AI'; DefaultBranch = 'main'; Branch = 'main'; Head = 'aaa'; Target = 'bbb'
            TrackedChanges = 0; Untracked = 0; Ahead = 0; Behind = 2; Busy = $false; FetchOk = $true; Fetch = 'ok'; Worktrees = @(); StaleBranches = @()
        }
        Mock Find-PcaiToolingRepo { , @([pscustomobject]@{ Path = 'C:\fake\PC-AI'; Slug = 'David-Martel/PC-AI'; Name = 'PC-AI' }) }
        Mock Invoke-PcaiParallel { , @($script:behind) }
        Mock Get-PcaiProfileChain { , @() }
        Mock Measure-PcaiProfileLoad { [pscustomobject]@{ Seconds = 0.5; ExitCode = 0; TimedOut = $false; FirstError = $null } }
        Mock Test-PcaiToolShim { , @([pscustomobject]@{ Name = 'git'; Path = 'git'; Length = 1; Ok = $true; Help = 'usage'; Shadowed = @() }) }
        Mock Get-PcaiOpenPullRequest { , @() }
        Mock Invoke-PcaiToolingCommand { [pscustomobject]@{ ExitCode = 0; Stdout = ''; Stderr = ''; TimedOut = $false; Seconds = 0 } }
    }
    It 'writes no file, merges nothing, queries no PR and posts nothing with -IsDryRun' {
        $root = Join-Path $TestDrive 'dry'
        $status = @(Invoke-PcaiToolingHealth -RunTrigger Event -Root $root -IsDryRun) | Where-Object { $_ -isnot [string] }
        Test-Path -LiteralPath $root | Should -BeFalse
        $status.repos[0].Action | Should -Be 'FastForward'
        Should -Invoke Get-PcaiOpenPullRequest -Times 0 -Exactly
        Should -Invoke Invoke-PcaiToolingCommand -Times 0 -Exactly -ParameterFilter { $ArgumentList -contains 'merge' -or $ArgumentList -contains 'send' }
        Should -Invoke Invoke-PcaiParallel -Times 1 -Exactly -ParameterFilter { $Argument.NoFetch -eq $true }
        # Loading the profile executes it, so a dry run must not start the probe.
        Should -Invoke Measure-PcaiProfileLoad -Times 0 -Exactly
    }
    It 'does not fast-forward a checkout whose branch changed after sampling' {
        Mock Test-PcaiRepoUnchanged { 'branch changed' }
        $root = Join-Path $TestDrive 'moved'
        $status = @(Invoke-PcaiToolingHealth -RunTrigger Event -Root $root -SkipBus) | Where-Object { $_ -isnot [string] }
        $status.repos[0].Action | Should -Be 'Changed'
        # A -NoBus run must not leave a debounce stamp that would hold back the scheduled pass.
        Test-Path -LiteralPath (Join-Path $root 'last-run.txt') | Should -BeFalse
        Should -Invoke Invoke-PcaiToolingCommand -Times 0 -Exactly -ParameterFilter { $ArgumentList -contains 'merge' }
    }
    It 'fast-forwards to the sampled commit when nothing changed' {
        Mock Test-PcaiRepoUnchanged { $null }
        # The verification step re-reads the branch after the merge.
        Mock Invoke-PcaiToolingCommand { [pscustomobject]@{ ExitCode = 0; Stdout = 'main'; Stderr = ''; TimedOut = $false; Seconds = 0 } } -ParameterFilter { $ArgumentList -contains '--show-current' }
        $status = @(Invoke-PcaiToolingHealth -RunTrigger Manual -Root (Join-Path $TestDrive 'ff') -SkipBus) | Where-Object { $_ -isnot [string] }
        $status.repos[0].Action | Should -Be 'FastForwarded'
        Should -Invoke Invoke-PcaiToolingCommand -Times 1 -Exactly -ParameterFilter { $ArgumentList -contains 'merge' -and $ArgumentList -contains 'bbb' }
    }
    It 'reports a rolled-back fast-forward as a warning needing its owner' {
        Mock Test-PcaiRepoUnchanged { $null }
        $script:revParseCalls = 0
        Mock Invoke-PcaiToolingCommand {
            $out = if ($ArgumentList -contains '--show-current') { 'feat/x' } elseif ($ArgumentList -contains 'rev-parse') { $script:revParseCalls++; if ($script:revParseCalls -eq 1) { 'bbb' } else { 'aaa' } } else { '' }
            [pscustomobject]@{ ExitCode = 0; Stdout = $out; Stderr = ''; TimedOut = $false; Seconds = 0 }
        }
        $status = @(Invoke-PcaiToolingHealth -RunTrigger Manual -Root (Join-Path $TestDrive 'rolled') -SkipBus) | Where-Object { $_ -isnot [string] }
        $status.repos[0].Action | Should -Be 'RolledBack'
        $status.severity | Should -Be 'warn'
        $status.summary | Should -Match 'needs owner \[PC-AI@main\(RolledBack\)\]'
        Should -Invoke Invoke-PcaiToolingCommand -Times 1 -Exactly -ParameterFilter { $ArgumentList -contains 'reset' -and $ArgumentList -contains '--keep' -and $ArgumentList -contains 'aaa' }
    }
    It 'raises the severity when a git inspection failed' {
        $script:behind | Add-Member -NotePropertyName InspectOk -NotePropertyValue $false
        $status = @(Invoke-PcaiToolingHealth -RunTrigger Manual -Root (Join-Path $TestDrive 'insp') -SkipBus) | Where-Object { $_ -isnot [string] }
        $status.repos[0].Action | Should -Be 'InspectFailed'
        $status.severity | Should -Be 'warn'
    }
    It 'raises the severity when the profile load reports runtime errors' {
        Mock Measure-PcaiProfileLoad { [pscustomobject]@{ Seconds = 0.5; ExitCode = 3; TimedOut = $false; FirstError = 'profile error: Import-Module failed' } }
        $status = @(Invoke-PcaiToolingHealth -RunTrigger Manual -Root (Join-Path $TestDrive 'err') -SkipBus -SkipFastForward) | Where-Object { $_ -isnot [string] }
        $status.severity | Should -Be 'warn'
        $status.summary | Should -Match 'exit=3 \(profile error: Import-Module failed\)'
    }
}

Describe 'Fast-forward cannot advance a branch other than the sampled one' -Tag 'Unit', 'Portable' -Skip:(-not [bool](Get-Command git -ErrorAction SilentlyContinue)) {
    # Why: revalidation and `git merge --ff-only` are separate commands, and merge advances whichever
    # branch is checked out when it runs. A switch to a feature branch at the same HEAD inside that
    # window used to pass the checks and then fast-forward the feature branch (codex review of
    # PR 345 at 93d54db7). -BeforeMerge is the seam that opens the window deterministically.
    BeforeAll {
        function Invoke-TestGit {
            param([string]$Path, [string[]]$Argument)
            $out = & git -C $Path -c user.name=t -c user.email=t@example.invalid -c commit.gpgsign=false -c core.hooksPath=.git/hooks @Argument 2>&1
            if ($LASTEXITCODE -ne 0) { throw "git $($Argument -join ' ') failed: $out" }
            ($out | Out-String).Trim()
        }
        function New-BehindCheckout {
            param([string]$Name)
            $root = Join-Path $TestDrive $Name
            $src = Join-Path $root 'src'
            $work = Join-Path $root 'work'
            $null = New-Item -ItemType Directory -Path $src
            Invoke-TestGit $src @('init', '-q', '-b', 'main') | Out-Null
            Set-Content -LiteralPath (Join-Path $src 'a.txt') -Value 'one'
            Invoke-TestGit $src @('add', 'a.txt') | Out-Null
            Invoke-TestGit $src @('commit', '-q', '-m', 'one') | Out-Null
            Invoke-TestGit $root @('clone', '-q', $src, $work) | Out-Null
            Invoke-TestGit $work @('branch', 'feat') | Out-Null
            Set-Content -LiteralPath (Join-Path $src 'a.txt') -Value 'two'
            Invoke-TestGit $src @('commit', '-qam', 'two') | Out-Null
            Invoke-TestGit $work @('fetch', '-q', 'origin') | Out-Null
            $state = Get-PcaiRepoState -Repo ([pscustomobject]@{ Path = $work; Slug = 'x/y'; Name = 'y' }) -NoFetch
            [pscustomobject]@{ Work = $work; State = $state }
        }
    }
    It 'fast-forwards the sampled branch when nothing changes' {
        $c = New-BehindCheckout 'happy'
        (Get-PcaiRepoAction -State $c.State) | Should -Be 'FastForward'
        $r = Invoke-PcaiFastForward -State $c.State
        $r.Action | Should -Be 'FastForwarded'
        Invoke-TestGit $c.Work @('rev-parse', 'main') | Should -Be $c.State.Target
        (Get-Content -LiteralPath (Join-Path $c.Work 'a.txt')) | Should -Be 'two'
    }
    It 'restores a feature branch at the same HEAD that was checked out inside the revalidation window' {
        $c = New-BehindCheckout 'window'
        $r = Invoke-PcaiFastForward -State $c.State -BeforeMerge { Invoke-TestGit $c.Work @('switch', '-q', 'feat') | Out-Null }
        $r.Action | Should -Be 'RolledBack'
        Invoke-TestGit $c.Work @('rev-parse', 'feat') | Should -Be $c.State.Head -Because 'the feature branch must not advance'
        Invoke-TestGit $c.Work @('rev-parse', 'main') | Should -Be $c.State.Head
        Invoke-TestGit $c.Work @('branch', '--show-current') | Should -Be 'feat'
        (Get-Content -LiteralPath (Join-Path $c.Work 'a.txt')) | Should -Be 'one'
        Invoke-TestGit $c.Work @('status', '--porcelain=v1', '--untracked-files=no') | Should -BeNullOrEmpty
    }
    It 'reports Changed and moves nothing when the branch differs before the merge starts' {
        $c = New-BehindCheckout 'early'
        Invoke-TestGit $c.Work @('switch', '-q', 'feat') | Out-Null
        $r = Invoke-PcaiFastForward -State $c.State
        $r.Action | Should -Be 'Changed'
        Invoke-TestGit $c.Work @('rev-parse', 'feat') | Should -Be $c.State.Head
    }
}
Describe 'Read-only inspection leaves repositories and state untouched' -Tag 'Unit', 'Portable' {
    It 'does not rewrite .git/index when a tracked file''s timestamp changed' -Skip:(-not (Get-Command git -ErrorAction SilentlyContinue)) {
        $repo = Join-Path $TestDrive 'repo'
        $null = New-Item -ItemType Directory -Path $repo
        & git -C $repo init -q -b main 2>$null
        Set-Content -LiteralPath (Join-Path $repo 'a.txt') -Value 'one'
        & git -C $repo add a.txt
        & git -C $repo -c user.name=t -c user.email=t@example.invalid -c commit.gpgsign=false commit -q -m init 2>$null
        (Get-Item -LiteralPath (Join-Path $repo 'a.txt')).LastWriteTimeUtc = [DateTime]::UtcNow.AddMinutes(5)
        $index = Join-Path $repo '.git/index'
        $before = (Get-FileHash -LiteralPath $index).Hash
        $null = Get-PcaiRepoState -Repo ([pscustomobject]@{ Path = $repo; Slug = 'x/y'; Name = 'y' }) -NoFetch
        (Get-FileHash -LiteralPath $index).Hash | Should -Be $before
    }
    It 'clears the SSH variables for the profile-load probe so the shim loads the full chain' {
        Mock Invoke-PcaiToolingCommand { [pscustomobject]@{ ExitCode = 0; Stdout = ''; Stderr = ''; TimedOut = $false; Seconds = 0.1 } }
        $null = Measure-PcaiProfileLoad
        Should -Invoke Invoke-PcaiToolingCommand -Times 1 -Exactly -ParameterFilter { $UnsetEnvironment -contains 'SSH_CONNECTION' -and $UnsetEnvironment -contains 'SSH_TTY' }
    }
}

Describe 'Help switches (AGENTS.md: -h and --help)' -Tag 'Unit', 'Portable' {
    It 'prints the comment-based help for <_>' -ForEach @('-h', '--help') {
        $path = Join-Path $repoRoot 'Tools/SystemScripts/Machine/Invoke-PcaiToolingHealth.ps1'
        $out = & (Join-Path $PSHOME 'pwsh') -NoProfile -NonInteractive -File $path $_ 2>&1 | Out-String
        $LASTEXITCODE | Should -Be 0
        $out | Should -Match 'SYNOPSIS'
    }
}

Describe 'Tool shim resolution' -Tag 'Unit', 'Portable' {
    It 'resolves pwsh and reports a missing tool as not ok' {
        $r = Test-PcaiToolShim -Name @('pwsh', 'pcai-no-such-tool-0b9f') -ResolveOnly
        ($r | Where-Object Name -eq 'pwsh').Ok | Should -BeTrue
        ($r | Where-Object Name -eq 'pcai-no-such-tool-0b9f').Ok | Should -BeFalse
    }
}
