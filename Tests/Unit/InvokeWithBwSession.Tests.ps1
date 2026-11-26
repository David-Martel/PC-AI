#Requires -Version 7.0
# Drives the real Invoke-WithBwSession.ps1 against a stub Unlock-BwVault.ps1 in a private
# temporary directory. No vault, backend or credential is touched.
Set-StrictMode -Version Latest

Describe 'Invoke-WithBwSession argument forwarding' -Tag 'Unit' {
    BeforeAll {
        $script:fixture = Join-Path ([IO.Path]::GetTempPath()) ('pcai-bwsession-' + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $script:fixture | Out-Null
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/Invoke-WithBwSession.ps1') -Destination $script:fixture
        # Stub unlocker: same function contract as the real launcher, with a fixed non-secret session.
        Set-Content -LiteralPath (Join-Path $script:fixture 'Unlock-BwVault.ps1') -Value @'
function Invoke-BwVaultUnlock { param([switch]$Quiet) $env:BW_SESSION = 'stub-nonsecret-session'; return 0 }
'@
        # Child that reports exactly what it received.
        $script:echo = Join-Path $script:fixture 'echo-args.ps1'
        Set-Content -LiteralPath $script:echo -Value @'
"COUNT=$($args.Count)"
foreach ($a in $args) { "ARG=$a" }
"SESSION=$env:BW_SESSION"
'@
        $script:wrapper = Join-Path $script:fixture 'Invoke-WithBwSession.ps1'
        $script:pwsh = (Get-Process -Id $PID).Path
        function script:Invoke-Wrapper([string[]]$Arguments) {
            & $script:pwsh -NoLogo -NoProfile -File $script:wrapper @Arguments 2>&1 | ForEach-Object { "$_" }
        }
    }
    AfterAll { Remove-Item -LiteralPath $script:fixture -Recurse -Force -ErrorAction SilentlyContinue }

    # Regression: `$exe, $rest = $args` leaves $rest a [string] when exactly one argument
    # follows the executable, and splatting a string passes it character by character.
    # On dtm-p1gen9 (2026-11-26) `Invoke-WithBwSession.ps1 bash.exe script.sh` ran `bash o`.
    It 'passes a single argument after the executable intact' {
        $out = Invoke-Wrapper @($script:pwsh, $script:echo)
        $out | Should -Contain 'COUNT=0'
        $out | Should -Contain 'SESSION=stub-nonsecret-session'
    }
    It 'passes one argument to the child intact' {
        $out = Invoke-Wrapper @($script:pwsh, '-NoProfile', '-File', $script:echo, 'only-arg')
        $out | Should -Contain 'COUNT=1'
        $out | Should -Contain 'ARG=only-arg'
    }
    It 'passes several arguments in order' {
        $out = Invoke-Wrapper @($script:pwsh, '-NoProfile', '-File', $script:echo, 'a', 'b c', '-Flag')
        $out | Should -Contain 'COUNT=3'
        ($out | Where-Object { $_ -like 'ARG=*' }) | Should -Be @('ARG=a', 'ARG=b c', 'ARG=-Flag')
    }
    It 'runs an executable with no arguments' {
        $out = Invoke-Wrapper @('hostname')
        $LASTEXITCODE | Should -Be 0
        @($out).Count | Should -BeGreaterThan 0
    }
}
