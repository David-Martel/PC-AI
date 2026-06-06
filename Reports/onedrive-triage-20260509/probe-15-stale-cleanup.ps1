$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\15-stale-cleanup.txt'
"=== Stale-SID OneDrive task cleanup at $(Get-Date -Format o) ===" | Out-File $out
"Mode: $(if ($env:CLEANUP_APPLY -eq '1') { 'APPLY' } else { 'DRY-RUN' })" | Out-File $out -Append

$Apply = ($env:CLEANUP_APPLY -eq '1')

# Re-inventory tasks (don't trust stale on-disk CSV)
$tasks = Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {
    $_.TaskName -match 'OneDrive' -or ($_.Actions.Execute -join ';') -match 'OneDrive|FileSyncHelper|FileCoAuth'
}

# Active user SID for this session (the one we MUST preserve)
$mySid = ([System.Security.Principal.WindowsIdentity]::GetCurrent()).User.Value
"My SID: $mySid" | Out-File $out -Append

$staleTargets = @()
$keepers = @()
foreach ($t in $tasks) {
    $p = $t.Principal.UserId
    $resolved = $p
    try {
        if ($p -match '^S-1-') {
            $resolved = ([System.Security.Principal.SecurityIdentifier]$p).Translate([System.Security.Principal.NTAccount]).Value
        }
    } catch { $resolved = "UNRESOLVABLE:$p" }

    $isMine = ($p -eq $mySid)
    $isSystemPerMachine = ($p -eq 'SYSTEM' -and $t.TaskName -eq 'OneDrive Per-Machine Standalone Update Task')
    $isStaleAccount = ($resolved -match '^(WsiAccount|DevToolsUser|CodexSandboxOffline)$') -or ($resolved -match '^UNRESOLVABLE:')

    if ($isStaleAccount) {
        $staleTargets += $t
    } else {
        $keepers += [PSCustomObject]@{Task=$t.TaskName; Principal=$p; Resolved=$resolved; Mine=$isMine; SystemPerMachine=$isSystemPerMachine}
    }
}

"`n--- Keepers ($($keepers.Count)) ---" | Out-File $out -Append
$keepers | Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- Stale targets ($($staleTargets.Count)) ---" | Out-File $out -Append
$staleTargets | Select-Object TaskName, TaskPath, @{n='Principal';e={$_.Principal.UserId}} | Format-Table -AutoSize | Out-String | Out-File $out -Append

if ($Apply) {
    "`n--- Applying deletion ---" | Out-File $out -Append
    foreach ($t in $staleTargets) {
        try {
            Unregister-ScheduledTask -TaskName $t.TaskName -TaskPath $t.TaskPath -Confirm:$false -ErrorAction Stop
            "  DELETED: $($t.TaskName)" | Out-File $out -Append
        } catch {
            "  FAILED: $($t.TaskName): $($_.Exception.Message)" | Out-File $out -Append
        }
    }
} else {
    "`n--- DRY-RUN: would delete the $($staleTargets.Count) tasks above ---" | Out-File $out -Append
    "  Set CLEANUP_APPLY=1 to actually delete." | Out-File $out -Append
}

# Verify post-state
"`n--- Post-state OneDrive task inventory ---" | Out-File $out -Append
Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {
    $_.TaskName -match 'OneDrive' -or ($_.Actions.Execute -join ';') -match 'OneDrive|FileSyncHelper|FileCoAuth'
} | Select-Object TaskName, State, @{n='Principal';e={$_.Principal.UserId}} | Format-Table -AutoSize | Out-String | Out-File $out -Append

Write-Host "Wrote $out"
Get-Content $out
