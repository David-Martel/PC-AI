#Requires -Version 5.1

<#
.SYNOPSIS
    Scans PowerShell sources for hard-coded credentials.

.DESCRIPTION
    The scan this replaces lived inline in maintenance.yml and had never
    executed: its patterns used ["\'] inside a single-quoted PowerShell string,
    and PowerShell escapes ' by DOUBLING it, so the string terminated early and
    every scheduled run died on "Unexpected token ']'".

    Once the quoting was corrected the naive patterns produced six findings on
    this repository and all six were false positives:

      Token = '%ProgramFiles%'                 PATH substitution, not auth
      $env:JULES_API_KEY = 'test-key-abc'      test fixture
      [string]$Password = 'testpassword'       test parameter default
      SMB_PASSWORD="$7"                        shell positional arg, no literal
      ConvertTo-SecureString -String $Password  a variable, not a literal

    A gate that is always red gets muted just as fast as one that is always
    green, so the patterns here require the assigned value to actually look
    like a secret: a quoted literal that is not a %PLACEHOLDER%, not a
    variable reference, not an obvious test/example value, and long enough to
    plausibly be real.

.PARAMETER Path
    Root to scan. Defaults to the repository root.

.PARAMETER ExcludeDirectory
    Directory names to skip anywhere in the tree.

.PARAMETER PassThru
    Emit finding objects in addition to the host output.

.EXAMPLE
    pwsh Tools/Test-CredentialScan.ps1

.NOTES
    Exit code 0 = no findings, 1 = findings or a failed self-test.
#>

[CmdletBinding()]
param(
    [string]$Path,
    [string[]]$ExcludeDirectory = @('.git', 'node_modules', '.pcai', 'worktrees', '.venv', 'target', 'Release'),
    [switch]$PassThru
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $Path) { $Path = Split-Path -Parent $PSScriptRoot }
if (-not (Test-Path $Path)) { throw "Scan path not found: $Path" }

# A value is only interesting if it is a quoted literal that does not look like
# a placeholder. These are the disqualifiers, checked against the captured value.
$placeholderPattern = '^\s*$|^%.*%$|^\$|^\$\{|^<.*>$|^\*+$|^x+$|^\.\.\.$'
$testValuePattern = 'test|example|dummy|sample|placeholder|changeme|your[-_ ]|redacted|fake|foo|bar|password123|<[^>]*>'
$minimumValueLength = 8

$rules = @(
    @{ Name = 'HardcodedPassword'; Pattern = '(?i)\bpassword\s*=\s*(["''])(?<value>[^"'']+)\1'; ValueGuard = $true }
    @{ Name = 'HardcodedApiKey'; Pattern = '(?i)\bapi[_-]?key\s*=\s*(["''])(?<value>[^"'']+)\1'; ValueGuard = $true }
    @{ Name = 'HardcodedSecret'; Pattern = '(?i)\bsecret\s*=\s*(["''])(?<value>[^"'']+)\1'; ValueGuard = $true }
    # "token" is heavily overloaded (PATH substitution tokens, parser tokens),
    # so it requires an auth-flavoured prefix rather than the bare word.
    @{ Name = 'HardcodedAuthToken'; Pattern = '(?i)\b(auth|access|bearer|refresh|api|pat|session)[_-]?token\s*=\s*(["''])(?<value>[^"'']+)\2'; ValueGuard = $true }
    @{ Name = 'HardcodedConnectionString'; Pattern = '(?i)(pwd|password)=(?<value>[^;"''\s]{8,})[;"'']'; ValueGuard = $true }
    # Only a LITERAL is a finding here. Converting a variable is the normal,
    # correct way to build a SecureString from user input.
    @{ Name = 'PlaintextSecureString'; Pattern = '(?i)ConvertTo-SecureString\s+(-String\s+)?(["''])(?<value>[^"'']+)\2[^\r\n]*-AsPlainText'; ValueGuard = $true }
    # Provider-specific formats are high confidence on their own.
    @{ Name = 'AwsAccessKeyId'; Pattern = '\b(AKIA|ASIA)[0-9A-Z]{16}\b'; ValueGuard = $false }
    @{ Name = 'PrivateKeyBlock'; Pattern = '-----BEGIN (RSA |OPENSSH |EC |DSA |PGP )?PRIVATE KEY-----'; ValueGuard = $false }
    @{ Name = 'GitHubToken'; Pattern = '\bgh[pousr]_[A-Za-z0-9]{36,}\b'; ValueGuard = $false }
    @{ Name = 'SlackToken'; Pattern = '\bxox[abprs]-[A-Za-z0-9-]{10,}\b'; ValueGuard = $false }
)

function Test-SuspiciousValue {
    <#
    .SYNOPSIS Returns $true when a captured literal plausibly is a real secret.
    #>
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) { return $false }
    if ($Value.Length -lt $minimumValueLength) { return $false }
    if ($Value -match $placeholderPattern) { return $false }
    if ($Value -match $testValuePattern) { return $false }
    # A value that is entirely one repeated character, or contains no variety,
    # is a mask rather than a credential.
    if (@($Value.ToCharArray() | Sort-Object -Unique).Count -lt 5) { return $false }
    return $true
}

function Find-CredentialInText {
    param([string]$Text, [string]$Source)

    $found = @()
    foreach ($rule in $rules) {
        foreach ($m in [regex]::Matches($Text, $rule.Pattern)) {
            if ($rule.ValueGuard) {
                $value = $m.Groups['value'].Value
                if (-not (Test-SuspiciousValue -Value $value)) { continue }
            }
            $lineNumber = ($Text.Substring(0, $m.Index) -split "`n").Count
            $found += [PSCustomObject]@{
                Rule   = $rule.Name
                Source = $Source
                Line   = $lineNumber
            }
        }
    }
    # Callers wrap this in @() -- PowerShell unrolls an empty array to $null on
    # return, and .Count on $null throws under StrictMode.
    return $found
}

# ---------------------------------------------------------------------------
# Positive control. A scanner that cannot demonstrate a catch is not a scanner.
# ---------------------------------------------------------------------------
$canarySamples = @(
    '$password = "Gg7#kQ2vRm9Lz"'                                      # pragma: allowlist secret
    '$apiKey = "sk-9fJ2mQ7vXb3nR8tZ4wY6cD1a"'                          # pragma: allowlist secret
    '$authToken = "hZ3q9WvK2mT7xR4bN8pL"'                              # pragma: allowlist secret
    'ConvertTo-SecureString -String "Vb7#nQ2mZx9Kt" -AsPlainText -Force' # pragma: allowlist secret
    'AKIAIOSFODNN7EXAMPLE'  # pragma: allowlist secret
)
foreach ($sample in $canarySamples) {
    if (@(Find-CredentialInText -Text $sample -Source '(canary)').Count -eq 0) {
        Write-Host "Positive control FAILED: scanner did not flag: $sample" -ForegroundColor Red
        exit 1
    }
}

# Negative control: the known false positives must stay quiet, or the gate goes
# permanently red and stops being read.
$benignSamples = @(
    "[PSCustomObject]@{ Token = '%ProgramFiles%'; Literal = `$env:ProgramFiles }"
    "`$env:JULES_API_KEY = 'test-key-abc'"
    "[string]`$Password = 'testpassword'"
    'SMB_PASSWORD="$7"'
    'ConvertTo-SecureString -String $Password -AsPlainText -Force'
    "`$apiKey = `$env:MY_API_KEY"
)
foreach ($sample in $benignSamples) {
    $noise = @(Find-CredentialInText -Text $sample -Source '(benign)')
    if ($noise.Count -gt 0) {
        Write-Host "Negative control FAILED: known-benign line flagged as $($noise[0].Rule): $sample" -ForegroundColor Red
        exit 1
    }
}
Write-Host 'Self-test OK: scanner flags planted secrets and ignores known-benign lines.' -ForegroundColor Green

# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------
# -ErrorAction SilentlyContinue matters: this repo has broken symlinks under
# Models\linked\hf that otherwise abort the whole enumeration.
$files = @(
    Get-ChildItem -Path $Path -Recurse -Include *.ps1, *.psm1, *.psd1 -File -ErrorAction SilentlyContinue |
        Where-Object {
            $relative = $_.FullName.Substring($Path.Length).TrimStart('\', '/')
            $segments = $relative -split '[\\/]'
            -not ($segments | Where-Object { $ExcludeDirectory -contains $_ })
        }
)

if ($files.Count -eq 0) {
    Write-Host "No PowerShell files found under $Path - refusing to report a clean scan of nothing." -ForegroundColor Red
    exit 1
}

$findings = @()
foreach ($file in $files) {
    $content = Get-Content -LiteralPath $file.FullName -Raw -ErrorAction SilentlyContinue
    if (-not $content) { continue }
    if ($content -match '#\s*pragma:\s*allowlist\s+secret') {
        # Line-level allowlisting would be better, but file-level keeps this
        # honest and visible: the marker has to be written into the source.
        $lines = $content -split "`r?`n"
        $content = ($lines | Where-Object { $_ -notmatch '#\s*pragma:\s*allowlist\s+secret' }) -join "`n"
    }
    $relative = $file.FullName.Substring($Path.Length).TrimStart('\', '/')
    $findings += @(Find-CredentialInText -Text $content -Source $relative)
}

Write-Host "Scanned $($files.Count) PowerShell file(s)." -ForegroundColor Cyan

if ($findings.Count -gt 0) {
    Write-Host "$($findings.Count) possible hard-coded credential(s):" -ForegroundColor Red
    foreach ($finding in $findings) {
        Write-Host "  [$($finding.Rule)] $($finding.Source):$($finding.Line)" -ForegroundColor Yellow
    }
    Write-Host 'If a hit is genuinely not a secret, add "# pragma: allowlist secret" to that line.' -ForegroundColor Gray
    if ($PassThru) { $findings }
    exit 1
}

Write-Host 'No hard-coded credentials found.' -ForegroundColor Green
if ($PassThru) { $findings }
exit 0
