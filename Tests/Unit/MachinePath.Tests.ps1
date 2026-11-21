#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }
<#
    Why these tests exist: the PATH policy decides which binary every shell and agent runs, and a wrong
    rewrite of the registry value can break every tool at once. Each Context pins one failure that the
    audit and the repair plan must catch, using temporary directories and injected directory probes so no
    test reads or writes the real PATH. Policy: Docs/MACHINE_PATH_POLICY.md. Code: Tools/SystemScripts/Machine.
#>

BeforeAll {
    $machineDir = Join-Path $PSScriptRoot '..\..\Tools\SystemScripts\Machine'
    Import-Module (Join-Path $machineDir 'MachinePath.psm1') -Force
    $script:Ctx = [pscustomobject]@{ UserHome = 'C:\Users\tester'; CargoHome = 'T:\Cargo'; LocalAppData = 'C:\Users\tester\AppData\Local'; SystemRoot = 'C:\Windows' }
    $script:Policy = Read-MachinePathPolicy -Path (Join-Path $machineDir 'machine-path.policy.json') -Context $script:Ctx
    $script:AllExist = { param($p) $true }
}

Describe 'Resolve-MachinePathEntry' {
    It 'expands USERPROFILE to the real profile, not the redirected one' {
        # Why: in an agent home $env:USERPROFILE points into AgentHomes; the registry entry means the real profile.
        $old = $env:USERPROFILE
        try {
            $env:USERPROFILE = 'C:\Agent\home'
            Resolve-MachinePathEntry -Entry '%USERPROFILE%\bin' -Context $script:Ctx | Should -Be 'C:\Users\tester\bin'
        } finally { $env:USERPROFILE = $old }
    }
    It 'expands 8.3 short names and parent segments so two spellings of one directory are duplicates' {
        # Why: dtm-work lists C:\PROGRA~1\CMake\bin and C:\Program Files\CMake\bin side by side; they are one directory.
        (Resolve-MachinePathEntry -Entry 'C:\PROGRA~1' -Context $script:Ctx) | Should -Be 'C:\Program Files'
        (Resolve-MachinePathEntry -Entry 'C:\Windows\System32\..\System32' -Context $script:Ctx) | Should -Be 'C:\Windows\System32'
    }
    It 'does not forbid the bin directory of the CARGO_HOME in use' {
        # Why: when CARGO_HOME is unset the canonical home is ~/.cargo, so ~/.cargo/bin is required, not forbidden.
        $ctx = [pscustomobject]@{ UserHome = 'C:\Users\tester'; CargoHome = 'C:\Users\tester\.cargo'; LocalAppData = 'C:\Users\tester\AppData\Local'; SystemRoot = 'C:\Windows' }
        $pol = Read-MachinePathPolicy -Path (Join-Path $PSScriptRoot '..\..\Tools\SystemScripts\Machine\machine-path.policy.json') -Context $ctx
        $pol.ForbiddenEntries | Should -Not -Contain 'C:\Users\tester\.cargo\bin'
    }
    It 'strips quotes and a trailing backslash but keeps a drive root' {
        Resolve-MachinePathEntry -Entry '"C:\Tools\"' -Context $script:Ctx | Should -Be 'C:\Tools'
        Resolve-MachinePathEntry -Entry 'C:\' -Context $script:Ctx | Should -Be 'C:\'
    }
}

Describe 'Test-MachinePathEntries' {
    It 'flags dead, duplicate, cross-scope duplicate, forbidden, relative and USERPROFILE entries' {
        $dead = { param($p) $p -ne 'C:\Gone' }
        $f = Test-MachinePathEntries -UserEntries @('C:\Gone', 'C:\Tools', 'c:\tools\', '%USERPROFILE%\bin', 'relative\dir', 'C:\Users\tester\.cargo\bin') `
            -MachineEntries @('C:\Windows', 'C:\Tools') -Policy $script:Policy -Context $script:Ctx -DirectoryExists $dead
        $codes = $f.Code
        $codes | Should -Contain 'PATH-DEAD'
        $codes | Should -Contain 'PATH-DUP'
        ($f | Where-Object Code -eq 'PATH-DUP-CROSS').Severity | Should -Be 'Info'
        $codes | Should -Contain 'PATH-ENVREL'
        $codes | Should -Contain 'PATH-RELATIVE'
        $codes | Should -Contain 'PATH-FORBIDDEN'
    }
    It 'reports an error when a REG_SZ value contains %VAR% because it is never expanded' {
        # Why: the classic silent breakage; Windows only expands REG_EXPAND_SZ.
        $f = Test-MachinePathEntries -UserEntries @('%SystemRoot%\System32') -UserKind String -Policy $script:Policy -Context $script:Ctx -DirectoryExists $script:AllExist
        ($f | Where-Object Code -eq 'PATH-KIND').Severity | Should -Be 'Error'
    }
    It 'reports required entries that are missing' {
        $f = Test-MachinePathEntries -UserEntries @('C:\Tools') -Policy $script:Policy -Context $script:Ctx -DirectoryExists $script:AllExist
        ($f | Where-Object Code -eq 'PATH-MISSING').Subject | Should -Contain 'T:\Cargo\bin'
    }
    It 'is quiet for a clean PATH (positive control for the checks above)' {
        $clean = @($script:Policy.RequiredUserEntries) + @('C:\Tools')
        $f = Test-MachinePathEntries -UserEntries $clean -MachineEntries @('C:\Windows') -Policy $script:Policy -Context $script:Ctx -DirectoryExists $script:AllExist
        @($f | Where-Object Severity -ne 'Info') | Should -BeNullOrEmpty
    }
}

Describe 'Get-MachinePathPlan' {
    It 'puts the policy head first, removes dead, duplicate and forbidden entries, and makes user entries absolute' {
        $exists = { param($p) $p -notin 'C:\Gone' }
        $plan = Get-MachinePathPlan -UserEntries @('C:\Tools', 'C:\Gone', 'C:\Users\tester\.cargo\bin', '%USERPROFILE%\bin', 'c:\tools') `
            -MachineEntries @('C:\Windows') -Policy $script:Policy -Context $script:Ctx -DirectoryExists $exists
        $plan.UserEntries[0] | Should -Be 'C:\Users\tester\.local\bin'
        $plan.UserEntries[1] | Should -Be 'C:\Users\tester\bin'
        $plan.UserEntries | Should -Contain 'T:\Cargo\bin'
        $plan.UserEntries | Should -Not -Contain 'C:\Users\tester\.cargo\bin'
        @($plan.UserEntries | Where-Object { $_ -ieq 'C:\Tools' }).Count | Should -Be 1
        ($plan.UserEntries -join ';') | Should -Not -Match '%'
        ($plan.Removed | Where-Object Reason -eq 'dead').Entry | Should -Contain 'C:\Gone'
        ($plan.Removed | Where-Object Reason -eq 'forbidden').Entry | Should -Contain 'C:\Users\tester\.cargo\bin'
    }
    It 'does not report an entry that merely moved into the head block as removed' {
        $plan = Get-MachinePathPlan -UserEntries @('C:\Tools', 'C:\Users\tester\bin') -Policy $script:Policy -Context $script:Ctx -DirectoryExists $script:AllExist
        $plan.Removed | Where-Object { $_.Entry -like '*tester\bin' } | Should -BeNullOrEmpty
    }
    It 'removes agent-home entries the dotnet SDK appends on first run' {
        # Why: observed on dtm-p1gen9; the SDK added <agent home>\.dotnet\tools to HKCU PATH, and every agent home leaves one behind.
        $agent = 'C:\Users\tester\AppData\Local\ClaudeCode\agent-homes\abc\home\.dotnet\tools'
        $plan = Get-MachinePathPlan -UserEntries @('C:\Tools', $agent) -Policy $script:Policy -Context $script:Ctx -DirectoryExists $script:AllExist
        $plan.UserEntries | Should -Not -Contain $agent
        ($plan.Removed | Where-Object Reason -eq 'agent-home').Entry | Should -Contain $agent
        $f = Test-MachinePathEntries -UserEntries @($agent) -Policy $script:Policy -Context $script:Ctx -DirectoryExists $script:AllExist
        $f.Code | Should -Contain 'PATH-AGENTHOME'
    }
    It 'keeps a directory that is in both scopes and leaves the machine text untouched' {
        # Why: dtm-work keeps ~/bin in the machine PATH on purpose (services and sshd never see the user PATH); only same-scope duplicates go.
        $plan = Get-MachinePathPlan -UserEntries @('C:\Windows\System32\OpenSSH', 'C:\Tools', 'C:\Tools') -MachineEntries @('%SystemRoot%\System32\OpenSSH\') -Policy $script:Policy -Context $script:Ctx -DirectoryExists $script:AllExist
        $plan.MachineEntries | Should -Contain '%SystemRoot%\System32\OpenSSH\'
        $plan.UserEntries | Should -Contain 'C:\Windows\System32\OpenSSH'
        ($plan.Removed | Where-Object { $_.Scope -eq 'User' }).Entry | Should -Be @('C:\Tools')
    }
}

Describe 'Shadow and wrapper checks on real directories' {
    BeforeEach {
        $script:Mine = Join-Path $TestDrive 'mine'; $script:Sys = Join-Path $TestDrive 'sys'
        Remove-Item -LiteralPath $script:Mine, $script:Sys -Recurse -Force -ErrorAction SilentlyContinue   # TestDrive persists across tests in a Describe
        New-Item -ItemType Directory -Force -Path $script:Mine, $script:Sys | Out-Null
        $pol = [pscustomobject]@{ ManagedBinDirs = @($script:Mine); IntentionalShadows = @{ sorty = 'intended' } }
        $script:ShadowPolicy = $pol
    }
    It 'flags a name that resolves differently under user-first and machine-first composition' {
        Set-Content (Join-Path $script:Mine 'tool.exe') 'a'; Set-Content (Join-Path $script:Sys 'tool.exe') 'b'
        $f = Test-MachinePathShadows -UserDirectories @($script:Mine) -MachineDirectories @($script:Sys) -Policy $script:ShadowPolicy
        $f.Code | Should -Contain 'PATH-ORDER-DEPENDENT'
    }
    It 'does not flag an intentional override' {
        Set-Content (Join-Path $script:Mine 'sorty.exe') 'a'; Set-Content (Join-Path $script:Sys 'sorty.exe') 'b'
        $f = Test-MachinePathShadows -UserDirectories @($script:Mine) -MachineDirectories @($script:Sys) -Policy $script:ShadowPolicy -Hash
        $f | Should -BeNullOrEmpty
    }
    It 'reports different content as a mismatch and identical content as a redundant copy' {
        Set-Content (Join-Path $script:Mine 'diff.exe') 'a'; Set-Content (Join-Path $script:Sys 'diff.exe') 'b'
        Set-Content (Join-Path $script:Mine 'same.exe') 'a'; Set-Content (Join-Path $script:Sys 'same.exe') 'a'
        $f = Test-MachinePathShadows -UserDirectories @($script:Mine) -MachineDirectories @($script:Sys) -Policy $script:ShadowPolicy -Hash
        ($f | Where-Object Code -eq 'PATH-SHADOW-MISMATCH').Subject | Should -Contain 'diff'
        ($f | Where-Object Code -eq 'PATH-SHADOW-DUP').Subject | Should -Contain 'same'
    }
    It 'flags an extensionless shim that PowerShell would pick over a real exe, but not one masked by a same-dir sibling' {
        # Why: archiving gpg.cmd exposed ~/bin/gpg (a bash shim) as the first `gpg` in PowerShell, which then failed to run.
        Set-Content (Join-Path $script:Mine 'gpg') '#!/bin/bash'; Set-Content (Join-Path $script:Sys 'gpg.exe') 'x'
        Set-Content (Join-Path $script:Mine 'safe') '#!/bin/bash'; Set-Content (Join-Path $script:Mine 'safe.cmd') 'x'; Set-Content (Join-Path $script:Sys 'safe.exe') 'x'
        $f = Test-MachinePathExtensionless -Directories @($script:Mine) -AllDirectories @($script:Mine, $script:Sys)
        $f.Subject | Should -Contain (Join-Path $script:Mine 'gpg')
        $f.Subject | Should -Not -Contain (Join-Path $script:Mine 'safe')
    }
    It 'finds SHA-suffixed variants but not the canonical binaries' {
        foreach ($n in 'agent-bus.exe', 'agent-bus-http.exe', 'agent-bus-mcp.exe', 'agent-bus-2fbca7ec8812a1158df61027a41c38eb8ad65fa6.exe', 'agent-bus-http.pre-2525c4d.exe', 'agent-bus-mcp-d2d57ef.exe', 'agent-bus.exe~RF9a118f1.TMP') {
            Set-Content (Join-Path $script:Mine $n) 'x'
        }
        $names = @(Get-MachineVariantFile -Directory $script:Mine | ForEach-Object { Split-Path $_ -Leaf })
        $names | Should -Contain 'agent-bus-2fbca7ec8812a1158df61027a41c38eb8ad65fa6.exe'
        $names | Should -Contain 'agent-bus-http.pre-2525c4d.exe'
        $names | Should -Contain 'agent-bus-mcp-d2d57ef.exe'
        $names | Should -Not -Contain 'agent-bus.exe'
        $names | Should -Not -Contain 'agent-bus-http.exe'
        $names | Should -Not -Contain 'agent-bus-mcp.exe'
    }
    It 'flags a wrapper whose every target is missing, ignores URLs, and flags USERPROFILE paths that exist only in the real profile' {
        $real = Join-Path $TestDrive 'realhome'; $agent = Join-Path $TestDrive 'agenthome'
        New-Item -ItemType Directory -Force -Path (Join-Path $real 'tools'), $agent | Out-Null
        Set-Content (Join-Path $real 'tools\x.exe') 'x'
        $ctx = [pscustomobject]@{ UserHome = $real }
        Set-Content (Join-Path $script:Mine 'dead.cmd') '@echo off', 'Z:\definitely\missing\thing.exe %*'
        Set-Content (Join-Path $script:Mine 'url.cmd') '@echo off', 'echo get https://example.com/cli/install.ps1'
        Set-Content (Join-Path $script:Mine 'home.cmd') '@echo off', '"%USERPROFILE%\tools\x.exe" %*'
        $f = Test-MachinePathWrappers -Directories @($script:Mine) -Context $ctx -AgentHome $agent
        ($f | Where-Object Code -eq 'PATH-WRAPPER-DEAD').Subject | Should -Contain (Join-Path $script:Mine 'dead.cmd')
        ($f | Where-Object Subject -like '*url.cmd') | Should -BeNullOrEmpty
        ($f | Where-Object Code -eq 'PATH-WRAPPER-ENVREL').Subject | Should -Contain (Join-Path $script:Mine 'home.cmd')
    }
}

Describe 'Resolution baseline' {
    It 'reports LOST, MOVED, GAINED and CONTENT and stays quiet for unchanged rows' {
        $b = @(
            [pscustomobject]@{ Command = 'a'; Winner = 'C:\x\a.exe'; Hash = '1' }, [pscustomobject]@{ Command = 'b'; Winner = 'C:\x\b.exe'; Hash = '1' },
            [pscustomobject]@{ Command = 'c'; Winner = $null; Hash = $null }, [pscustomobject]@{ Command = 'd'; Winner = 'C:\x\d.exe'; Hash = '1' },
            [pscustomobject]@{ Command = 'e'; Winner = 'C:\x\e.exe'; Hash = '1' })
        $a = @(
            [pscustomobject]@{ Command = 'a'; Winner = $null; Hash = $null }, [pscustomobject]@{ Command = 'b'; Winner = 'C:\y\b.exe'; Hash = '1' },
            [pscustomobject]@{ Command = 'c'; Winner = 'C:\y\c.exe'; Hash = '2' }, [pscustomobject]@{ Command = 'd'; Winner = 'C:\x\d.exe'; Hash = '2' },
            [pscustomobject]@{ Command = 'e'; Winner = 'C:\x\e.exe'; Hash = '1' })
        $d = Compare-MachineResolutionBaseline -Before $b -After $a
        ($d | Where-Object Command -eq 'a').Status | Should -Be 'LOST'
        ($d | Where-Object Command -eq 'b').Status | Should -Be 'MOVED'
        ($d | Where-Object Command -eq 'c').Status | Should -Be 'GAINED'
        ($d | Where-Object Command -eq 'd').Status | Should -Be 'CONTENT'
        $d | Where-Object Command -eq 'e' | Should -BeNullOrEmpty
    }
    It 'resolves the first PATHEXT match in directory order' {
        $one = Join-Path $TestDrive 'one'; $two = Join-Path $TestDrive 'two'
        New-Item -ItemType Directory -Force -Path $one, $two | Out-Null
        Set-Content (Join-Path $one 'zz.cmd') 'x'; Set-Content (Join-Path $two 'zz.exe') 'x'
        Resolve-MachineCommand -Name 'zz' -Directories @($one, $two) | Should -Be (Join-Path $one 'zz.cmd')
        Resolve-MachineCommand -Name 'nonexistent-zz' -Directories @($one, $two) | Should -BeNullOrEmpty
    }
}

Describe 'Get-MachineBuildJobCap' {
    # Why: the synchronizer persisted eight build jobs everywhere, overriding the NUC and dtm-work profiles that cap native builds at two.
    BeforeAll {
        $script:Prof = Join-Path $TestDrive 'profiles'; New-Item -ItemType Directory -Force -Path $script:Prof | Out-Null
        '{ "hostName": "capped-host", "buildPolicy": { "maxNativeBuildJobs": 2 } }' | Set-Content (Join-Path $script:Prof 'capped-host.json')
        '{ "hostName": "free-host" }' | Set-Content (Join-Path $script:Prof 'free-host.json')
        'not json' | Set-Content (Join-Path $script:Prof 'broken.json')
    }
    It 'honours the cap of the matching host profile only' {
        Get-MachineBuildJobCap -HostName 'CAPPED-HOST' -ProfileDirectories @($script:Prof) | Should -Be 2
        Get-MachineBuildJobCap -HostName 'free-host' -ProfileDirectories @($script:Prof) | Should -Be 8
        Get-MachineBuildJobCap -HostName 'unknown-host' -ProfileDirectories @($script:Prof) | Should -Be 8
    }
    It 'never raises the default and ignores unreadable profiles' {
        Get-MachineBuildJobCap -HostName 'capped-host' -ProfileDirectories @($script:Prof) -Default 1 | Should -Be 1
        Get-MachineBuildJobCap -HostName 'x' -ProfileDirectories @((Join-Path $TestDrive 'missing')) | Should -Be 8
    }
}

Describe 'Get-MachineReferenceCorpus' {
    # Why: an incomplete reference search must not look like "nothing references this file" (codex review of PR 343, P1).
    BeforeEach {
        $script:Cfg = Join-Path $TestDrive 'cfg.json'; Set-Content $script:Cfg 'C:\Users\x\bin\tool.exe'
        $script:Ok = { @('') }
    }
    It 'collects text from files and providers and reports no failures when everything is readable' {
        $c = Get-MachineReferenceCorpus -Files @($script:Cfg, (Join-Path $TestDrive 'optional-missing.json')) -Directories @() -TaskProvider { @('task C:\a\b.exe') } -ServiceProvider $script:Ok -ProcessProvider $script:Ok
        $c.Failures.Count | Should -Be 0
        $c.Text | Should -Match 'c:\\users\\x\\bin\\tool\.exe'
        $c.Text | Should -Match 'task c:\\a\\b\.exe'
    }
    It 'reads task actions that have no Execute property (COM handlers) instead of failing the whole search' {
        # Why: found on dtm-p1gen9 the first time the search ran fail-closed; strict mode made a handler action throw.
        $tasks = @(
            [pscustomobject]@{ Actions = @([pscustomobject]@{ Execute = 'C:\a\b.exe'; Arguments = '-x' }) },
            [pscustomobject]@{ Actions = @([pscustomobject]@{ ClassId = '{abc}' }) }
        )
        @(Get-MachineTaskActionText -Task $tasks) | Should -Be @('C:\a\b.exe -x')
    }
    It 'reports a failure for each provider that throws' {
        $c = Get-MachineReferenceCorpus -Files @() -Directories @() -TaskProvider { throw 'access denied' } -ServiceProvider { throw 'rpc unavailable' } -ProcessProvider $script:Ok
        $c.Failures.Count | Should -Be 2
        ($c.Failures -join ' ') | Should -Match 'access denied'
    }
    It 'reports a failure for a file that exists but cannot be read' {
        $locked = [IO.File]::Open($script:Cfg, 'Open', 'ReadWrite', 'None')
        try { $c = Get-MachineReferenceCorpus -Files @($script:Cfg) -Directories @() -TaskProvider $script:Ok -ServiceProvider $script:Ok -ProcessProvider $script:Ok }
        finally { $locked.Dispose() }
        $c.Failures.Count | Should -Be 1
    }
}

Describe 'Sync-MachineEnvironment.ps1 safety contract' {
    # Why: write-capable session scripts must offer -DryRun and --help that prove no side effects (AGENTS.md), and PATH must be
    # planned from the CARGO_HOME the script selects, not the one it started with.
    BeforeAll {
        $script:Sync = (Resolve-Path (Join-Path $PSScriptRoot '..\..\Tools\SystemScripts\Machine\Sync-MachineEnvironment.ps1')).Path
        $script:SyncText = Get-Content $script:Sync -Raw
    }
    It '-DryRun leaves HKCU environment, PATH and the backup directory untouched' {
        $snap = { $k = Get-Item -LiteralPath 'HKCU:\Environment'; ($k.GetValueNames() | Sort-Object | ForEach-Object { "$_=$($k.GetValue($_, $null, 'DoNotExpandEnvironmentNames'))" }) -join "`n" }
        $before = & $snap
        $bk = Join-Path (Get-MachinePathContext).UserHome '.machine\backups'
        $stamp = { if (Test-Path $bk) { [string](Get-ChildItem $bk -Force -Recurse -ErrorAction SilentlyContinue | Measure-Object LastWriteTimeUtc -Maximum).Maximum + '|' + @(Get-ChildItem $bk -Force -Recurse -ErrorAction SilentlyContinue).Count } else { 'absent' } }
        $bkBefore = & $stamp
        $out = & (Get-Command pwsh).Source -NoProfile -File $script:Sync -DryRun 2>&1 | Out-String
        $LASTEXITCODE | Should -Be 0
        $out | Should -Match 'Dry run'
        & $snap | Should -Be $before
        & $stamp | Should -Be $bkBefore
    }
    It '-h and --help print usage and exit 0 without side effects' {
        foreach ($flag in '-h', '--help') {
            $out = & (Get-Command pwsh).Source -NoProfile -File $script:Sync $flag 2>&1 | Out-String
            $LASTEXITCODE | Should -Be 0
            $out | Should -Match '-DryRun'
        }
    }
    It 'rebuilds the PATH context after CARGO_HOME is selected' {
        $envLoop = $script:SyncText.IndexOf('$canonicalEnv.GetEnumerator()')
        $rebuild = $script:SyncText.IndexOf('Get-MachinePathContext -CargoHome $cargoHome')
        $envLoop | Should -BeGreaterThan 0
        $rebuild | Should -BeGreaterThan $envLoop
        $script:SyncText.LastIndexOf('Read-MachinePathPolicy') | Should -BeGreaterThan $rebuild
    }
    It 'does not hard-code the build job count' {
        $script:SyncText | Should -Not -Match "'CARGO_BUILD_JOBS'\s*=\s*'8'"
        $script:SyncText | Should -Not -Match "'MAKEFLAGS'\s*=\s*'-j8'"
    }
}


Describe 'Test-MachineShimReplacement' {
    # Why: a bash shim whose target exists is only redundant if the command still resolves to that target once the shim is gone;
    # the target need not be on PATH (codex review of PR 343, round 2, P2).
    It 'accepts a shim when the first remaining candidate is its target, whatever the slash style or case' {
        Test-MachineShimReplacement -Target 'C:/Program Files/GnuPG/bin/gpg.exe' -Candidate @('C:\program files\gnupg\bin\gpg.exe', 'C:\other\gpg.exe') | Should -BeTrue
    }
    It 'rejects a shim whose target is not on PATH or loses to another command' {
        Test-MachineShimReplacement -Target 'D:/tools/jq.exe' -Candidate @() | Should -BeFalse
        Test-MachineShimReplacement -Target 'D:/tools/jq.exe' -Candidate @('C:\elsewhere\jq.exe', 'D:\tools\jq.exe') | Should -BeFalse
    }
}


Describe 'Test-MachineTextReference' {
    # Why: a hook that names a binary as $env:USERPROFILE\bin\x.exe, $env:HOME\... or %HOME%\... was not seen as a reference, so the
    # binary could be archived while still in use (codex review of PR 343, round 4, P2).
    BeforeAll {
        $script:UH = 'C:\Users\tester'
        $script:Exe = 'C:\Users\tester\bin\agent-bus-mcp-abc1234.exe'
    }
    It 'recognises every spelling of the profile-relative path, with either slash' -ForEach @(
        @{ Text = 'C:\Users\tester\bin\agent-bus-mcp-abc1234.exe' }, @{ Text = 'C:/Users/tester/bin/agent-bus-mcp-abc1234.exe' },
        @{ Text = '~/bin/agent-bus-mcp-abc1234.exe' }, @{ Text = '%USERPROFILE%\bin\agent-bus-mcp-abc1234.exe' },
        @{ Text = '%HOME%\bin\agent-bus-mcp-abc1234.exe' }, @{ Text = '$HOME/bin/agent-bus-mcp-abc1234.exe' },
        @{ Text = '$env:USERPROFILE\bin\agent-bus-mcp-abc1234.exe' }, @{ Text = '"$env:HOME/bin/agent-bus-mcp-abc1234.exe"' }
    ) {
        $corpus = $Text.Replace('\\', '\').Replace('/', '\').ToLowerInvariant()
        Test-MachineTextReference -CorpusText $corpus -Path $script:Exe -UserHome $script:UH | Should -BeTrue
    }
    It 'does not report an unrelated or merely similar path' {
        $corpus = '$env:userprofile\bin\agent-bus-mcp-other.exe c:\users\tester\bin\agent-bus.exe'
        Test-MachineTextReference -CorpusText $corpus -Path $script:Exe -UserHome $script:UH | Should -BeFalse
    }
}

Describe 'Machine scripts contain no stray control characters' {
    # Why: a Python-side escape slip put literal backspace and tab characters into directory names in Sync-MachineEnvironment.ps1,
    # so a normal run would have stopped after writing the environment variables. The dry-run tests return before that block.
    It 'has no control characters other than tab, CR and LF in <_.Name>' -ForEach @(
        Get-ChildItem (Join-Path $PSScriptRoot '..\..\Tools\SystemScripts\Machine') -Include *.ps1, *.psm1, *.json -File -Recurse
    ) {
        $bad = @(Get-Content -LiteralPath $_.FullName | ForEach-Object { $i = 0 } { $i++; if ($_ -match '[\x00-\x08\x0B\x0C\x0E-\x1F]') { $i } })
        $bad | Should -BeNullOrEmpty
    }
    It 'has no tab or control character inside a quoted path argument in the synchronizer' {
        $text = Get-Content -LiteralPath (Join-Path $PSScriptRoot '..\..\Tools\SystemScripts\Machine\Sync-MachineEnvironment.ps1') -Raw
        [regex]::Matches($text, "Join-Path\s+\S+\s+'[^'\r\n]*[\t\x00-\x1F][^'\r\n]*'").Count | Should -Be 0
        $text | Should -Not -Match "\.dotnet\tools"
    }
}
