#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

# Why this file exists: the full PSScriptAnalyzer pass over ~850 files takes 8-12 minutes and
# PSScriptAnalyzer cannot be run in parallel runspaces, so Tools/Test-ChangedPowerShellLint.ps1
# analyzes only the files a branch changed. A gate that silently analyzes nothing is worse than
# none, so these tests pin both directions: a bad file fails, a good file passes, an empty change
# set says so, and the analyzed count is reported.
# See Tools/Test-ChangedPowerShellLint.ps1 and Tests/Invoke-PcaiCi.ps1.

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:Tool = Join-Path $script:RepoRoot 'Tools/Test-ChangedPowerShellLint.ps1'
    $script:Settings = Join-Path $script:RepoRoot 'PSScriptAnalyzerSettings.psd1'
    $script:HasAnalyzer = [bool](Get-Module -ListAvailable PSScriptAnalyzer)

    function New-FixtureRepo {
        param([string]$Name)
        $dir = Join-Path $TestDrive $Name
        [IO.Directory]::CreateDirectory($dir) | Out-Null
        git -C $dir init -q -b main 2>&1 | Out-Null
        git -C $dir config user.email 'fixture@example.invalid' | Out-Null
        git -C $dir config user.name 'fixture' | Out-Null
        git -C $dir config commit.gpgsign false | Out-Null
        # The fixture is a throwaway repository: host-wide hooks (core.hooksPath in the global config, such as a
        # credential or Invoke-Expression scanner) must not decide whether its commits land.
        git -C $dir config core.hooksPath .git/hooks | Out-Null
        [IO.File]::WriteAllText((Join-Path $dir 'base.ps1'), "Write-Output 'base'`n", [Text.UTF8Encoding]::new($false))
        git -C $dir add -A | Out-Null
        Invoke-FixtureCommit $dir 'base'
        $dir
    }

    # A fixture commit that does not land would leave its files staged and change what the tool sees, so fail loudly.
    function Invoke-FixtureCommit {
        param([string]$Repo, [string]$Message)
        $out = git -C $Repo commit -q -m $Message 2>&1
        if ($LASTEXITCODE -ne 0) { throw "fixture commit '$Message' failed in ${Repo}: $($out | Out-String)" }
    }

    function Invoke-Tool {
        param([string]$Repo)
        $out = pwsh -NoProfile -File $script:Tool -RepoRoot $Repo -BaseRef main -SettingsPath $script:Settings 2>&1
        [pscustomobject]@{ Exit = $LASTEXITCODE; Text = ($out | Out-String) }
    }
}

Describe 'Test-ChangedPowerShellLint' -Tag 'Unit', 'Portable' -Skip:(-not [bool](Get-Module -ListAvailable PSScriptAnalyzer)) {
    It 'fails and names the rule for a changed file with an analyzer error' {
        $repo = New-FixtureRepo 'bad'
        # PSAvoidUsingInvokeExpression is enabled in the repo settings.
        [IO.File]::WriteAllText((Join-Path $repo 'bad.ps1'), "Invoke-Expression 'Get-Date'`n", [Text.UTF8Encoding]::new($false))
        $r = Invoke-Tool $repo
        $r.Exit | Should -Be 1
        $r.Text | Should -Match 'bad\.ps1'
        $r.Text | Should -Match 'PSAvoidUsingInvokeExpression'
    }

    It 'passes and reports the analyzed count for a clean changed file' {
        $repo = New-FixtureRepo 'good'
        [IO.File]::WriteAllText((Join-Path $repo 'good.ps1'), "Write-Output 'ok'`n", [Text.UTF8Encoding]::new($false))
        $r = Invoke-Tool $repo
        $r.Exit | Should -Be 0
        $r.Text | Should -Match 'analyzed 1 changed PowerShell file'
    }

    It 'ignores unchanged files even when they would fail' {
        $repo = New-FixtureRepo 'unchanged'
        [IO.File]::WriteAllText((Join-Path $repo 'old-bad.ps1'), "Invoke-Expression 'Get-Date'`n", [Text.UTF8Encoding]::new($false))
        git -C $repo add -A | Out-Null
        Invoke-FixtureCommit $repo 'bad but committed on main'
        $r = Invoke-Tool $repo
        $r.Exit | Should -Be 0
        $r.Text | Should -Match 'no changed PowerShell files'
    }

    It 'sees files committed on the branch and files that are untracked' {
        $repo = New-FixtureRepo 'branch'
        git -C $repo switch -q -c topic 2>&1 | Out-Null
        [IO.File]::WriteAllText((Join-Path $repo 'committed.ps1'), "Invoke-Expression 'Get-Date'`n", [Text.UTF8Encoding]::new($false))
        git -C $repo add -A | Out-Null
        Invoke-FixtureCommit $repo 'topic'
        [IO.File]::WriteAllText((Join-Path $repo 'untracked.ps1'), "Write-Output 'u'`n", [Text.UTF8Encoding]::new($false))
        $r = Invoke-Tool $repo
        $r.Exit | Should -Be 1
        $r.Text | Should -Match 'committed\.ps1'
        $r.Text | Should -Match 'analyzed 2 changed PowerShell file'
    }
    # Protects: a file whose name Git would quote (non-ASCII) is still analyzed. Detects: selection from Git's quoted
    # display names, where "t\303\251st.ps1" matches no extension filter and the file disappears (codex review of #337).
    # Needs: PSScriptAnalyzer and git. Breadcrumb: Tools/Test-ChangedPowerShellLint.ps1 Get-GitPathList.
    It 'analyzes staged and untracked files with non-ASCII names' {
        $repo = New-FixtureRepo 'nonascii'
        [IO.File]::WriteAllText((Join-Path $repo 'stagéd.ps1'), "Invoke-Expression 'Get-Date'`n", [Text.UTF8Encoding]::new($false))
        git -C $repo add -A | Out-Null
        [IO.File]::WriteAllText((Join-Path $repo 'untrackéd.ps1'), "Write-Output 'u'`n", [Text.UTF8Encoding]::new($false))
        $r = Invoke-Tool $repo
        $r.Exit | Should -Be 1
        $r.Text | Should -Match 'analyzed 2 changed PowerShell file'
        $r.Text | Should -Match 'PSAvoidUsingInvokeExpression'
    }
}
