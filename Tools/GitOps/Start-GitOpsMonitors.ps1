#requires -Version 7.0
<#
.SYNOPSIS
    Tiny launcher: fire Invoke-GitOpsMonitors.ps1 DETACHED and return immediately. Called from the
    git pre-push hook (lefthook). Always exits 0 so a monitor issue can NEVER block a push.
    Kept as a separate -File script to avoid the sh->pwsh nested-quoting that broke an inline command.
#>
try {
    $mon = Join-Path $PSScriptRoot 'Invoke-GitOpsMonitors.ps1'
    # Win32_Process.Create orphans the worker (parent = wmiprvse), breaking the git/lefthook job
    # object so the pre-push hook returns INSTANTLY. Start-Process keeps the child in the hook's job,
    # which makes git wait for the full 20s monitor — defeating "non-blocking".
    $cmd = "pwsh -NoProfile -WindowStyle Hidden -File `"$mon`""
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd } | Out-Null
} catch {
    # Never block the push on a monitor-launch failure.
    Write-Host "gitops monitor launch skipped: $_" -ForegroundColor DarkYellow
}
exit 0
