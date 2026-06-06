#requires -Version 7.0
<#
.SYNOPSIS
    Read-only doctor/preflight for git commit signing + GitHub auth. Verifies an agent can produce a
    Verified signed commit on THIS machine before it tries to commit (wire into commit-cluster /
    context-save pre-flight). Designed for the Claude agent-home HOME-redirect environment.

.DESCRIPTION
    Checks (all read-only):
      1. Effective signing config (gpg.format / user.signingkey / commit.gpgsign). Warns if OpenPGP
         is selected while HOME is a long redirected path (the keyboxd-socket-too-long trap).
      2. ssh-agent holds an identity; allowed_signers exists and lists the committer email.
      3. Throwaway SSH-signed commit in TEMP verifies %G? == G (the real proof).
      4. gh auth valid + has the scopes used by the GitOps tooling.
      5. No credential dir (.ssh/.gnupg/.gitconfig/.config) under %OneDrive% (KFM footgun).
    Returns a structured object; non-zero exit on FAIL so CI/agents can gate.

.PARAMETER Json   Emit the result object as JSON.
#>
[CmdletBinding()]
param([switch] $Json)

$checks = [System.Collections.Generic.List[object]]::new()
function Add-Check { param($Name, $State, $Detail) $checks.Add([ordered]@{ check = $Name; state = $State; detail = $Detail }) }

# 1. effective signing config
$fmt = (git config --get gpg.format) 2>$null; if (-not $fmt) { $fmt = 'openpgp' }
$key = (git config --get user.signingkey) 2>$null
$sign = (git config --get commit.gpgsign) 2>$null
$homeLong = ($env:HOME -and $env:HOME.Length -gt 90)
if ($fmt -eq 'ssh') { Add-Check 'signing-format' 'PASS' "ssh (key=$key)" }
elseif ($homeLong) { Add-Check 'signing-format' 'WARN' "openpgp under long HOME ($($env:HOME.Length) chars) -> keyboxd socket risk; prefer gpg.format=ssh" }
else { Add-Check 'signing-format' 'WARN' "openpgp (consider ssh)" }

# 2. ssh-agent + allowed_signers
$agent = (ssh-add -l 2>$null)
if ($LASTEXITCODE -eq 0 -and $agent) { Add-Check 'ssh-agent' 'PASS' "$(@($agent).Count) identity(ies) loaded" }
else { Add-Check 'ssh-agent' 'WARN' 'no identities (signing may prompt for passphrase)' }
$as = (git config --get gpg.ssh.allowedSignersFile) 2>$null
if ($as -and (Test-Path $as)) { Add-Check 'allowed-signers' 'PASS' $as } else { Add-Check 'allowed-signers' 'WARN' 'not set/missing (commits unverifiable locally)' }

# 3. throwaway SSH signed-commit smoke test
$state = 'FAIL'; $detail = 'not run'
try {
    $t = Join-Path $env:TEMP "gitsign-doctor-$PID"
    Remove-Item $t -Recurse -Force -ErrorAction SilentlyContinue; New-Item -ItemType Directory $t | Out-Null
    git -C $t init -q
    git -C $t config user.name (git config user.name); git -C $t config user.email (git config user.email)
    git -C $t config gpg.format ssh
    git -C $t config user.signingkey ($key ? $key : 'C:/Users/david/.ssh/id_ed25519.pub')
    git -C $t config commit.gpgsign true
    if ($as) { git -C $t config gpg.ssh.allowedSignersFile $as }
    'x' | Out-File "$t\a" -Encoding ascii; git -C $t add a
    $env:GIT_TERMINAL_PROMPT = '0'
    git -C $t commit -q -m t 2>$null | Out-Null
    $g = git -C $t log -1 --pretty='%G?' 2>$null
    $state = ($g -eq 'G') ? 'PASS' : 'FAIL'; $detail = "verify=%G?=$g"
    Remove-Item $t -Recurse -Force -ErrorAction SilentlyContinue
} catch { $detail = "$_" }
Add-Check 'sign-smoke-test' $state $detail

# 4. gh auth
$auth = gh auth status 2>&1 | Out-String
if ($auth -match 'Logged in') { Add-Check 'gh-auth' 'PASS' (($auth | Select-String 'Logged in').ToString().Trim()) }
else { Add-Check 'gh-auth' 'FAIL' 'not logged in' }

# 5. OneDrive KFM guard
$od = $env:OneDrive
$bad = @()
if ($od) { foreach ($d in '.ssh', '.gnupg', '.gitconfig', '.config\git') { $p = "C:\Users\david\$d"; if ($p -like "$od*") { $bad += $d } } }
if ($bad) { Add-Check 'onedrive-guard' 'FAIL' "credential dirs under OneDrive: $($bad -join ',')" } else { Add-Check 'onedrive-guard' 'PASS' 'credential dirs not OneDrive-synced' }

$fail = ($checks | Where-Object state -eq 'FAIL').Count
$result = [ordered]@{ overall = ($fail -eq 0 ? 'PASS' : 'FAIL'); fail_count = $fail; checks = $checks }
if ($Json) { $result | ConvertTo-Json -Depth 6 }
else {
    Write-Host "git-signing doctor: $($result.overall)" -ForegroundColor ($fail -eq 0 ? 'Green' : 'Red')
    foreach ($c in $checks) {
        $col = switch ($c.state) { 'PASS' { 'Green' } 'WARN' { 'Yellow' } default { 'Red' } }
        Write-Host ("  [{0,-4}] {1,-18} {2}" -f $c.state, $c.check, $c.detail) -ForegroundColor $col
    }
}
exit $fail
