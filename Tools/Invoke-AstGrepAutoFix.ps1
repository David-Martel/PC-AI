#Requires -Version 7.0
<#
.SYNOPSIS
    Integrates ast-grep findings with a local LLM for automated fix proposals.
.DESCRIPTION
    Runs `sg scan --json` over a target directory, reads 10 lines of context per
    finding, sends each to Ollama for a concise fix proposal, and outputs results.
    Duplicate rule+snippet combos are served from an in-memory cache.
    Mode=Apply writes fixes back after per-file confirmation (ShouldProcess).
.PARAMETER Path
    Directory to scan. Relative paths resolve from repo root. Default: Native/pcai_core/
.PARAMETER Model
    Ollama model. Default: qwen2.5-coder:3b (~115 tok/s, ~20 findings < 60 s)
.PARAMETER Mode
    Scan=findings only | Propose=(default) LLM proposals | Apply=write fixes
.PARAMETER OllamaUrl
    Ollama base URL. Default: http://127.0.0.1:11434
.PARAMETER MaxFindings
    Cap on findings sent to LLM. Default: 20
.PARAMETER Format
    Table=(default) | Json | Markdown
.EXAMPLE
    pwsh Tools\Invoke-AstGrepAutoFix.ps1 -Mode Propose -MaxFindings 10
.EXAMPLE
    pwsh Tools\Invoke-AstGrepAutoFix.ps1 -Mode Scan -Format Json | ConvertFrom-Json
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$Path        = 'Native/pcai_core/',
    [string]$Model       = 'qwen2.5-coder:3b',
    [ValidateSet('Scan','Propose','Apply')][string]$Mode = 'Propose',
    [string]$OllamaUrl   = 'http://127.0.0.1:11434',
    [int]$MaxFindings    = 20,
    [ValidateSet('Table','Json','Markdown')][string]$Format = 'Table'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-ContextLines([string]$FilePath, [int]$LineNumber) {
    if (-not (Test-Path $FilePath)) { return '' }
    [string[]]$all = [System.IO.File]::ReadAllLines($FilePath)
    $s = [Math]::Max(0, $LineNumber - 6)
    $e = [Math]::Min($all.Length - 1, $LineNumber + 4)
    ($s..$e | ForEach-Object { '{0,4}: {1}' -f ($_ + 1), $all[$_] }) -join "`n"
}

function Invoke-OllamaFix([string]$Rule, [string]$Msg, [string]$Ctx, [string]$Key) {
    if ($script:Cache.ContainsKey($Key)) { return $script:Cache[$Key] }
    $prompt = "Rust ast-grep rule '$Rule': $Msg`n`nCode:`n``````rust`n$Ctx`n```````n`nProvide ONLY the fixed code, no explanation."
    $body   = @{ model = $Model; prompt = $prompt; stream = $false } | ConvertTo-Json -Compress
    try {
        $r   = Invoke-RestMethod -Uri "$OllamaUrl/api/generate" -Method Post `
                   -Body $body -ContentType 'application/json' -TimeoutSec 15
        $fix = ($r.response ?? '').Trim() -replace '^```[a-z]*\r?\n','' -replace '\r?\n```$',''
        $script:Cache[$Key] = $fix
        return $fix
    } catch {
        Write-Verbose "LLM timeout/error for rule '$Rule': $_"
        return ''
    }
}

# ── resolve paths ──────────────────────────────────────────────────────────────
$repoRoot = if ($env:PCAI_ROOT) { $env:PCAI_ROOT } else { Split-Path -Parent $PSScriptRoot }
$scanPath = if ([System.IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $repoRoot $Path }
if (-not (Test-Path $scanPath)) { Write-Error "Scan path not found: $scanPath"; exit 1 }
if (-not (Get-Command sg -ErrorAction SilentlyContinue)) {
    Write-Error "ast-grep (sg) not found on PATH. Install from https://ast-grep.github.io"; exit 1
}

# ── build sg args with rule dirs ──────────────────────────────────────────────
$ruleDirs = @("$HOME/.claude/rules/rust", "$HOME/.claude/ast-grep-rules") | Where-Object { Test-Path $_ }
if ($ruleDirs.Count -eq 0) { Write-Warning "No rule dirs under ~/.claude — using project-local rules only." }
$sgArgs = @('scan','--json')
foreach ($d in $ruleDirs) { $sgArgs += '--rule', $d }
$sgArgs += $scanPath

# ── run scan ──────────────────────────────────────────────────────────────────
Write-Verbose "sg $($sgArgs -join ' ')"
$rawJson = sg @sgArgs 2>$null
if ([string]::IsNullOrWhiteSpace($rawJson)) { Write-Host "No findings." -ForegroundColor Green; exit 0 }

$raw = $rawJson | ConvertFrom-Json
if (-not $raw -or $raw.Count -eq 0) { Write-Host "No findings." -ForegroundColor Green; exit 0 }
Write-Verbose "$($raw.Count) raw finding(s)"

# Normalise schema across sg versions
$findings = $raw | ForEach-Object {
    [PSCustomObject]@{
        File     = $_.file     ?? $_.path     ?? ''
        Line     = [int]($_.range.start.line  ?? $_.lines ?? 0) + 1
        Rule     = $_.rule_id  ?? $_.ruleId   ?? $_.check_id ?? 'unknown'
        Severity = ($_.severity ?? 'warning').ToLower()
        Message  = $_.message  ?? ''
        Snippet  = $_.text     ?? ''
    }
} | Select-Object -First $MaxFindings

$script:Cache   = [System.Collections.Generic.Dictionary[string,string]]::new()
$results        = [System.Collections.Generic.List[PSCustomObject]]::new()

foreach ($f in $findings) {
    $ctx = Get-ContextLines -FilePath $f.File -LineNumber $f.Line
    $key = "$($f.Rule)|$($f.Snippet.GetHashCode())"
    $fix = if ($Mode -ne 'Scan') { Invoke-OllamaFix $f.Rule $f.Message $ctx $key } else { '' }
    $results.Add([PSCustomObject]@{
        File = $f.File; Line = $f.Line; Rule = $f.Rule; Severity = $f.Severity
        Original = $f.Snippet; Context = $ctx; ProposedFix = $fix; FixAvailable = ($fix -ne '')
    })
}

# ── apply ─────────────────────────────────────────────────────────────────────
if ($Mode -eq 'Apply') {
    foreach ($r in $results | Where-Object FixAvailable) {
        $rel = $r.File.Replace($repoRoot,'').TrimStart('/\')
        Write-Host "`n[$($r.Rule)] $rel`:$($r.Line)" -ForegroundColor Yellow
        Write-Host "Original : $($r.Original)"   -ForegroundColor Red
        Write-Host "Proposed : $($r.ProposedFix)" -ForegroundColor Green
        if ($PSCmdlet.ShouldProcess("$rel line $($r.Line)", "Apply fix for $($r.Rule)")) {
            $src = [System.IO.File]::ReadAllText($r.File)
            if ($src.Contains($r.Original.Trim())) {
                [System.IO.File]::WriteAllText($r.File,
                    $src.Replace($r.Original.Trim(), $r.ProposedFix.Trim()),
                    [System.Text.Encoding]::UTF8)
                Write-Host "  Applied." -ForegroundColor Green
            } else {
                Write-Warning "  Snippet not found verbatim in $($r.File) — skipped."
            }
        }
    }
}

# ── output ────────────────────────────────────────────────────────────────────
switch ($Format) {
    'Json'     { $results | ConvertTo-Json -Depth 4 }
    'Markdown' {
        "# ast-grep Auto-Fix Report`n"
        "Scan: ``$scanPath``  |  Model: ``$Model``  |  $(Get-Date -Format 'yyyy-MM-dd HH:mm')`n"
        foreach ($r in $results) {
            $rel = $r.File.Replace($repoRoot,'').TrimStart('/\')
            "## ``$($r.Rule)`` — ${rel}:$($r.Line)`n> **$($r.Severity)** — $($r.Message)`n"
            "**Original**`n``````rust`n$($r.Original)`n```````n"
            if ($r.FixAvailable) { "**Proposed**`n``````rust`n$($r.ProposedFix)`n```````n" }
            else                 { "_No fix proposed_`n" }
        }
    }
    default {  # Table
        $hdr = '{0,-30} {1,5} {2,-26} {3,-8} {4}' -f 'File','Line','Rule','Severity','Fix Available'
        Write-Host $hdr -ForegroundColor Cyan
        Write-Host ('-' * 80) -ForegroundColor DarkGray
        foreach ($r in $results) {
            $color = switch ($r.Severity) { 'error' {'Red'} 'warning' {'Yellow'} default {'White'} }
            $label = if ($r.FixAvailable) { "Yes ($Model)" } else { 'No' }
            Write-Host ('{0,-30} {1,5} {2,-26} {3,-8} {4}' -f `
                (Split-Path -Leaf $r.File), $r.Line, $r.Rule, $r.Severity, $label) -ForegroundColor $color
        }
        Write-Host "`n$($results.Count) finding(s) | Mode=$Mode" -ForegroundColor DarkGray
    }
}
