#Requires -Version 5.1
BeforeAll {
    $repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    # Complete maintained functions; no module import, native bridge or tool discovery.
    . (Join-Path $repo 'Modules/PC-AI.Acceleration/Private/Initialize-RustTools.ps1')
    . (Join-Path $repo 'Modules/PC-AI.Acceleration/Public/Get-RustToolStatus.ps1')
    $script:SavedRustStatusCaches = @{}
    foreach ($name in @('ToolPaths','RustToolCache')) {
        $existing = Get-Variable -Name $name -Scope Script -ErrorAction SilentlyContinue
        $script:SavedRustStatusCaches[$name] = @{ Existed = $null -ne $existing; Value = $(if ($existing) { $existing.Value } else { $null }) }
    }

    function Get-RustStatusFixtureCalls {
        param([string]$Path)
        $receipt = [IO.Path]::Combine([IO.Path]::GetDirectoryName($Path), 'version-calls.jsonl')
        if (-not [IO.File]::Exists($receipt)) { return }
        foreach ($line in [IO.File]::ReadAllLines($receipt)) { $line | ConvertFrom-Json }
    }
    function New-RustStatusVersionFixture {
        param([string]$Name = 'owned-version.ps1', [string]$Body = "'owned fixture 1.2.3'")
        $directory = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        [void][IO.Directory]::CreateDirectory($directory)
        $path = Join-Path $directory $Name
        # The invoked script has its own scope; record actual calls in owned bytes.
        $header = @'
$receipt = @{ Arguments = @($args); Source = $PSCommandPath } | ConvertTo-Json -Compress
[IO.File]::AppendAllText([IO.Path]::Combine($PSScriptRoot, 'version-calls.jsonl'), $receipt + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
'@ + [Environment]::NewLine
        [IO.File]::WriteAllText($path, $header + $Body, [Text.UTF8Encoding]::new($false))
        return $path
    }
}

AfterAll {
    foreach ($name in $script:SavedRustStatusCaches.Keys) {
        $saved = $script:SavedRustStatusCaches[$name]
        if ($saved.Existed) { Set-Variable -Name $name -Scope Script -Value $saved.Value }
        else { Remove-Variable -Name $name -Scope Script -ErrorAction SilentlyContinue }
    }
}

Describe 'Rust tool status availability and version contracts' -Tag 'Unit', 'Acceleration' {
    BeforeEach {
        $script:SelectedVersionPath = [IO.Path]::Combine($TestDrive, 'uninvoked-version.ps1')
        Mock Get-RustToolPath { $null }
    }

    It 'reports a missing discovery result without querying or invoking a path' {
        Mock Test-Path { throw 'A null discovery result must short-circuit existence checks.' }
        $result = @(Get-RustToolStatus -Tool rg)
        $result | Should -HaveCount 1
        $result[0].Tool | Should -BeExactly 'rg'
        $result[0].Available | Should -BeFalse
        $result[0].Path | Should -BeNullOrEmpty
        $result[0].Version | Should -BeNullOrEmpty
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath).Count | Should -Be 0
        Should -Invoke Get-RustToolPath -Exactly -Times 1 -ParameterFilter { $ToolName -ceq 'rg' }
        Should -Invoke Test-Path -Exactly -Times 0
    }

    It 'retains a stale discovered path but does not invoke it or invent a version' {
        $script:SelectedVersionPath = Join-Path $TestDrive 'absent-version.ps1'
        Mock Get-RustToolPath { $script:SelectedVersionPath }
        $result = Get-RustToolStatus -Tool fd
        $result.Path | Should -BeExactly $script:SelectedVersionPath
        $result.Available | Should -BeFalse
        $result.Version | Should -BeNullOrEmpty
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath).Count | Should -Be 0
    }

    # Regression detector: real literal bytes previously produced Available=false.
    It 'invokes an existing bracket-containing path literally and preserves its source bytes' {
        $script:SelectedVersionPath = New-RustStatusVersionFixture -Name 'owned[rg].ps1'
        $before = (Get-FileHash -LiteralPath $script:SelectedVersionPath).Hash
        Mock Get-RustToolPath { $script:SelectedVersionPath }
        $result = Get-RustToolStatus -Tool rg
        $result.Available | Should -BeTrue
        $result.Path | Should -BeExactly $script:SelectedVersionPath
        $result.Version | Should -BeExactly '1.2.3'
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath) | Should -HaveCount 1
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath)[0].Arguments | Should -Be @('--version')
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath)[0].Source | Should -BeExactly $script:SelectedVersionPath
        (Get-FileHash -LiteralPath $script:SelectedVersionPath).Hash | Should -BeExactly $before
    }

    It 'uses only the first output line and passes the exact version argument' {
        $script:SelectedVersionPath = New-RustStatusVersionFixture -Body "'owned release 2.3.4'; 'ignored release 99.0.0'"
        Mock Get-RustToolPath { $script:SelectedVersionPath }
        $result = Get-RustToolStatus -Tool bat
        $result.Available | Should -BeTrue
        $result.Version | Should -BeExactly '2.3.4'
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath) | Should -HaveCount 1
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath)[0].Arguments | Should -Be @('--version')
    }

    It 'retains trimmed non-semver first-line output rather than dropping the available tool' {
        $script:SelectedVersionPath = New-RustStatusVersionFixture -Body "'  owned development build  '; 'later 7.8.9'"
        Mock Get-RustToolPath { $script:SelectedVersionPath }
        $result = Get-RustToolStatus -Tool procs
        $result.Tool | Should -BeExactly 'procs'
        $result.Available | Should -BeTrue
        $result.Version | Should -BeExactly 'owned development build'
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath) | Should -HaveCount 1
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath)[0].Arguments | Should -Be @('--version')
    }

    It 'keeps an available tool with a null version when its owned version script emits nothing' {
        $script:SelectedVersionPath = New-RustStatusVersionFixture -Body '# Deliberately no version output.'
        Mock Get-RustToolPath { $script:SelectedVersionPath }
        $result = Get-RustToolStatus -Tool tokei
        $result.Available | Should -BeTrue
        $result.Version | Should -BeNullOrEmpty
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath) | Should -HaveCount 1
    }

    It 'reports Unknown and retains the row when the owned version script throws' {
        $script:SelectedVersionPath = New-RustStatusVersionFixture -Body "throw 'Owned version refusal.'"
        Mock Get-RustToolPath { $script:SelectedVersionPath }
        $result = @(Get-RustToolStatus -Tool hyperfine)
        $result | Should -HaveCount 1
        $result[0].Tool | Should -BeExactly 'hyperfine'
        $result[0].Available | Should -BeTrue
        $result[0].Version | Should -BeExactly 'Unknown'
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath) | Should -HaveCount 1
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath)[0].Arguments | Should -Be @('--version')
    }

    It 'returns the ten distinct public identifiers and acceleration mappings for All' {
        $result = @(Get-RustToolStatus)
        $result | Should -HaveCount 10
        @($result.Tool | Sort-Object) | Should -Be @('bat','btm','dust','eza','fd','hyperfine','procs','rg','sd','tokei')
        @($result.Tool | Select-Object -Unique) | Should -HaveCount 10
        @($result | Where-Object Available) | Should -HaveCount 0
        @($result | Where-Object { $null -ne $_.Path -or $null -ne $_.Version }) | Should -HaveCount 0
        ($result | Where-Object Tool -EQ 'rg').Name | Should -BeExactly 'ripgrep'
        ($result | Where-Object Tool -EQ 'rg').Accelerates | Should -BeExactly 'Search-LogsFast, Search-ContentFast, Get-SystemEvents'
        ($result | Where-Object Tool -EQ 'fd').Accelerates | Should -BeExactly 'Find-FilesFast, Find-DuplicatesFast'
        ($result | Where-Object Tool -EQ 'sd').Accelerates | Should -BeExactly 'N/A'
        ($result | Where-Object Tool -EQ 'btm').Name | Should -BeExactly 'bottom'
        Should -Invoke Get-RustToolPath -Exactly -Times 10
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath).Count | Should -Be 0
    }

    It 'rejects an unsupported tool before discovery' {
        { Get-RustToolStatus -Tool rustc } | Should -Throw '*Cannot validate argument*'
        Should -Invoke Get-RustToolPath -Exactly -Times 0
    }
}

Describe 'Rust tool preferred path and cached discovery boundaries' -Tag 'Unit', 'Acceleration' {
    BeforeEach {
        $script:SelectedVersionPath = [IO.Path]::Combine($TestDrive, 'uninvoked-version.ps1')
        $script:ToolPaths = @{}
        $script:RustToolCache = @{}
    }

    It 'uses the preferred cached tool path without fallback discovery' {
        $script:SelectedVersionPath = New-RustStatusVersionFixture
        $script:ToolPaths['rg'] = $script:SelectedVersionPath
        Mock Find-RustTool { throw 'Preferred path must bypass discovery.' }
        $result = Get-RustToolStatus -Tool rg
        $result.Path | Should -BeExactly $script:SelectedVersionPath
        $result.Available | Should -BeTrue
        $result.Version | Should -BeExactly '1.2.3'
        Should -Invoke Find-RustTool -Exactly -Times 0
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath) | Should -HaveCount 1
    }

    It 'falls back when the preferred path is empty without overwriting unrelated cached tools' {
        $script:SelectedVersionPath = New-RustStatusVersionFixture
        $script:ToolPaths = @{ rg = ''; fd = 'unrelated-owned-cache-value' }
        Mock Find-RustTool { $script:SelectedVersionPath }
        $result = Get-RustToolStatus -Tool rg
        $result.Path | Should -BeExactly $script:SelectedVersionPath
        $result.Available | Should -BeTrue
        $result.Version | Should -BeExactly '1.2.3'
        $script:ToolPaths['fd'] | Should -BeExactly 'unrelated-owned-cache-value'
        Should -Invoke Find-RustTool -Exactly -Times 1 -ParameterFilter { $ToolName -ceq 'rg' }
    }

    It 'does not treat a stale preferred path as available or invoke its fallback implicitly' {
        $script:SelectedVersionPath = Join-Path $TestDrive 'stale-preferred.ps1'
        $script:ToolPaths['rg'] = $script:SelectedVersionPath
        Mock Find-RustTool { throw 'This status operation must not silently rewrite discovery policy.' }
        $result = Get-RustToolStatus -Tool rg
        $result.Path | Should -BeExactly $script:SelectedVersionPath
        $result.Available | Should -BeFalse
        $result.Version | Should -BeNullOrEmpty
        Should -Invoke Find-RustTool -Exactly -Times 0
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath).Count | Should -Be 0
    }

    It 'reuses the secondary discovery cache without searching PATH or invoking where.exe' {
        $script:SelectedVersionPath = New-RustStatusVersionFixture
        $script:RustToolCache['rg'] = $script:SelectedVersionPath
        Mock Join-Path { throw 'A cache hit must not construct discovery paths.' }
        $result = Get-RustToolStatus -Tool rg
        $result.Path | Should -BeExactly $script:SelectedVersionPath
        $result.Available | Should -BeTrue
        $result.Version | Should -BeExactly '1.2.3'
        $script:RustToolCache['rg'] | Should -BeExactly $script:SelectedVersionPath
        Should -Invoke Join-Path -Exactly -Times 0
    }

    It 'checks cached bracket-path availability literally without running the version script' {
        $script:SelectedVersionPath = New-RustStatusVersionFixture -Name 'owned[rg].ps1'
        $script:ToolPaths['rg'] = $script:SelectedVersionPath
        Mock Find-RustTool { throw 'Availability must use the preferred path.' }
        $available = Test-RustToolAvailable -Tool rg
        $available | Should -BeOfType ([bool])
        $available | Should -BeTrue
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath).Count | Should -Be 0
        Should -Invoke Find-RustTool -Exactly -Times 0
    }

    It 'returns false for cached missing-tool availability without invoking discovery or version' {
        $script:ToolPaths['rg'] = Join-Path $TestDrive 'absent-availability.ps1'
        Mock Find-RustTool { throw 'Availability must not refresh a stale preferred path.' }
        $available = Test-RustToolAvailable -Tool rg
        $available | Should -BeOfType ([bool])
        $available | Should -BeFalse
        @(Get-RustStatusFixtureCalls $script:SelectedVersionPath).Count | Should -Be 0
        Should -Invoke Find-RustTool -Exactly -Times 0
    }
}
