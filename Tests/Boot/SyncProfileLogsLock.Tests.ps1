#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
Regression tests for the sync-lock deadlock in Sync-ProfileLogs.ps1.

A lock file left behind by a killed run used to block every subsequent run
forever: Acquire-SyncLock opens with FileMode::CreateNew, which fails against an
existing file, and nothing decided whether the holder was still alive. On this
machine a lock left at 2026-02-21 failed every hourly sync for over six months,
silently, because Task Scheduler reports only exit 0x1.

The pair of tests below is the point: breaking a stale lock is worthless if it
also breaks a live one, so the fresh-lock case is a negative control.
#>

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:Script = Join-Path $script:RepoRoot 'Tools\SystemScripts\Machine\Sync-ProfileLogs.ps1'

    function New-SandboxPair {
        $base = Join-Path ([IO.Path]::GetTempPath()) "plsync-$([guid]::NewGuid())"
        $local = Join-Path $base 'local'
        $remote = Join-Path $base 'remote'
        New-Item -ItemType Directory -Path $local, $remote -Force | Out-Null
        # Give the sync something to do.
        Set-Content -LiteralPath (Join-Path $local 'sample.log') -Value 'entry one' -Encoding utf8
        [pscustomobject]@{ Base = $base; Local = $local; Remote = $remote }
    }

    function New-LockFile {
        param([string]$Dir, [datetime]$WriteTimeUtc, [string]$Content = '')
        $lock = Join-Path $Dir '.sync.lock'
        Set-Content -LiteralPath $lock -Value $Content -NoNewline -Encoding utf8
        (Get-Item -LiteralPath $lock -Force).LastWriteTimeUtc = $WriteTimeUtc
        $lock
    }

    function Invoke-Sync {
        param([string]$Local, [string]$Remote)
        $out = & pwsh -NoProfile -NonInteractive -File $script:Script `
            -LocalPath $Local -OneDrivePath $Remote 2>&1
        [pscustomobject]@{ Output = ($out | Out-String); ExitCode = $LASTEXITCODE }
    }
}

Describe 'Sync-ProfileLogs sync lock' {

    It 'parses without error' {
        $errors = $null
        [void][System.Management.Automation.Language.Parser]::ParseFile(
            $script:Script, [ref]$null, [ref]$errors)
        @($errors) | Should -BeNullOrEmpty
    }

    It 'breaks an abandoned lock and completes' {
        $s = New-SandboxPair
        try {
            # Zero-byte, months old - exactly what a killed run leaves behind,
            # since the original code never flushed the lock's contents.
            New-LockFile -Dir $s.Remote -WriteTimeUtc ([DateTime]::UtcNow.AddDays(-200)) | Out-Null

            $r = Invoke-Sync -Local $s.Local -Remote $s.Remote

            $r.ExitCode | Should -Be 0 -Because 'a 200-day-old lock is abandoned'
            $r.Output | Should -Not -Match 'Sync lock held'
        } finally {
            Remove-Item -LiteralPath $s.Base -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It 'NEGATIVE CONTROL: still respects a lock held by a live process' {
        $s = New-SandboxPair
        try {
            # Fresh, and naming this very process - a genuinely running sync.
            $info = @{
                Machine = $env:COMPUTERNAME
                Pid     = $PID
                Time    = [DateTime]::UtcNow.ToString('o')
            } | ConvertTo-Json -Compress
            New-LockFile -Dir $s.Remote -WriteTimeUtc ([DateTime]::UtcNow) -Content $info | Out-Null

            $r = Invoke-Sync -Local $s.Local -Remote $s.Remote

            # If this passes trivially the stale-lock fix has made the lock
            # useless, which would be worse than the deadlock it replaced.
            $r.ExitCode | Should -Not -Be 0 -Because 'a fresh lock must still block'
            $r.Output | Should -Match 'lock'
        } finally {
            Remove-Item -LiteralPath $s.Base -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It 'leaves no lock behind after a successful run' {
        $s = New-SandboxPair
        try {
            $r = Invoke-Sync -Local $s.Local -Remote $s.Remote
            $r.ExitCode | Should -Be 0
            Test-Path -LiteralPath (Join-Path $s.Remote '.sync.lock') | Should -BeFalse
        } finally {
            Remove-Item -LiteralPath $s.Base -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
