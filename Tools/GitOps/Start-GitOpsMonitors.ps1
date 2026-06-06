#requires -Version 7.0
<#
.SYNOPSIS
    Tiny launcher: fire Invoke-GitOpsMonitors.ps1 DETACHED and return immediately. Called from the
    git pre-push hook (lefthook). Always exits 0 so a monitor issue can NEVER block a push.
    Kept as a separate -File script to avoid the sh->pwsh nested-quoting that broke an inline command.
#>
try {
    $mon = Join-Path $PSScriptRoot 'Invoke-GitOpsMonitors.ps1'
    $tmp = $env:TEMP
    # Redirect the child's stdout/stderr to files so it does NOT inherit/hold the hook's console
    # pipe — otherwise the push appears to hang until the 20s monitor finishes. True fire-and-forget.
    Start-Process -FilePath 'pwsh' -WindowStyle Hidden `
        -ArgumentList '-NoProfile', '-WindowStyle', 'Hidden', '-File', $mon `
        -RedirectStandardOutput (Join-Path $tmp 'gitops-monitor.out.log') `
        -RedirectStandardError  (Join-Path $tmp 'gitops-monitor.err.log') | Out-Null
} catch {
    # Never block the push on a monitor-launch failure.
    Write-Host "gitops monitor launch skipped: $_" -ForegroundColor DarkYellow
}
exit 0
