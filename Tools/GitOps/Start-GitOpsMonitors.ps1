#requires -Version 7.0
<#
.SYNOPSIS
    Tiny launcher: fire Invoke-GitOpsMonitors.ps1 DETACHED and return immediately. Called from the
    git pre-push hook (lefthook). Always exits 0 so a monitor issue can NEVER block a push.
    Kept as a separate -File script to avoid the sh->pwsh nested-quoting that broke an inline command.
#>
try {
    $mon = Join-Path $PSScriptRoot 'Invoke-GitOpsMonitors.ps1'
    Start-Process -FilePath 'pwsh' -WindowStyle Hidden `
        -ArgumentList '-NoProfile', '-WindowStyle', 'Hidden', '-File', $mon | Out-Null
} catch {
    # Never block the push on a monitor-launch failure.
    Write-Host "gitops monitor launch skipped: $_" -ForegroundColor DarkYellow
}
exit 0
