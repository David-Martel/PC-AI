#Requires -Version 7.0
<#
.SYNOPSIS
    Elevation wrapper that runs Repair-WorkstationInputReliability.ps1 under a
    transcript so its output is captured to a log file (the elevated window is
    separate from the launching session). Used so an automated/launching context
    can read the result after the user approves the UAC prompt.
.PARAMETER Mode
    WhatIf (default, preview only) | Apply (make changes) | Revert (undo from newest backup).
.EXAMPLE
    Start-Process pwsh -Verb RunAs -ArgumentList '-NoProfile','-File','<thisfile>','-Mode','WhatIf'
#>
[CmdletBinding()]
param([ValidateSet('WhatIf','Apply','Revert')][string]$Mode = 'WhatIf')

$target = 'C:\codedev\PC_AI\Tools\InputDiagnostics\Repair-WorkstationInputReliability.ps1'
$logDir = 'C:\codedev\PC_AI\Logs\elevated'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $logDir "repair-$Mode-$stamp.log"

Start-Transcript -Path $log -Force | Out-Null
Write-Host "=== Elevated run: Mode=$Mode  $(Get-Date -Format o) ==="
$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
Write-Host "Elevated: $elevated"
try {
    switch ($Mode) {
        'WhatIf' { & $target -WhatIf }
        'Apply'  { & $target }
        'Revert' { & $target -Revert }
    }
    Write-Host "=== EXIT CODE: $LASTEXITCODE ==="
} catch {
    Write-Host "ERROR: $_"
} finally {
    Stop-Transcript | Out-Null
}
# marker file so a watcher knows the run finished and where the log is
Set-Content -Path (Join-Path $logDir 'last-run.txt') -Value $log -Encoding utf8
