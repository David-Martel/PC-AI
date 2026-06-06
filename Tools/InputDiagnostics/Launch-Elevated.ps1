#Requires -Version 7.0
<#
.SYNOPSIS
    Generic non-elevated launcher: runs any -File elevated via UAC, waits, and prints
    the elevated log. Target script must self-transcript and write
    C:\codedev\PC_AI\Logs\elevated\last-run.txt with its log path (as the toolkit's
    elevated scripts do).
.PARAMETER Script      Full path to the elevated target script.
.PARAMETER WhatIfPass  Append -WhatIf to the target.
.PARAMETER WaitSeconds Max wait for UAC + completion. Default 120.
#>
[CmdletBinding()]
param([Parameter(Mandatory)][string]$Script,[switch]$WhatIfPass,[int]$WaitSeconds=120)

$marker = 'C:\codedev\PC_AI\Logs\elevated\last-run.txt'
New-Item -ItemType Directory -Path (Split-Path $marker) -Force | Out-Null
Remove-Item $marker -ErrorAction SilentlyContinue

$argl = @('-NoProfile','-File',$Script)
if ($WhatIfPass) { $argl += '-WhatIf' }
try {
    Start-Process pwsh -Verb RunAs -ArgumentList $argl -ErrorAction Stop
} catch {
    Write-Host "Could not launch elevated: $_" -ForegroundColor Red
    $hint = if ($WhatIfPass) { ' -WhatIf' } else { '' }
    Write-Host "Run manually from elevated PowerShell:`n  pwsh -NoProfile -File `"$Script`"$hint" -ForegroundColor Yellow
    return
}
Write-Host ">>> APPROVE THE UAC PROMPT (target: $(Split-Path $Script -Leaf)$(if($WhatIfPass){' -WhatIf'})). Waiting up to $WaitSeconds s..." -ForegroundColor Cyan
$deadline = (Get-Date).AddSeconds($WaitSeconds)
while ((Get-Date) -lt $deadline -and -not (Test-Path $marker)) { Start-Sleep -Seconds 2 }
if (Test-Path $marker) {
    $lp = (Get-Content $marker -Raw).Trim()
    Write-Host "`n================ ELEVATED LOG: $lp ================" -ForegroundColor Green
    Get-Content $lp
} else {
    Write-Host "No completion detected (UAC not approved or still running)." -ForegroundColor Yellow
}
