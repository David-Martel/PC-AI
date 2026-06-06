$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\05-stale-tasks.txt'
$csv = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\05-stale-tasks.csv'
"=== OneDrive scheduled task inventory at $(Get-Date -Format o) ===" | Out-File $out

$tasks = Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {
    $_.TaskName -match 'OneDrive' -or ($_.Actions.Execute -join ';') -match 'OneDrive|FileSyncHelper|FileCoAuth'
}
"Total OneDrive-related tasks: $($tasks.Count)" | Out-File $out -Append

$rows = foreach ($t in $tasks) {
    $info = Get-ScheduledTaskInfo -TaskName $t.TaskName -TaskPath $t.TaskPath -ErrorAction SilentlyContinue
    $principalUser = $t.Principal.UserId
    $resolvedUser = ''
    try {
        if ($principalUser -match '^S-1-') {
            $sid = New-Object System.Security.Principal.SecurityIdentifier($principalUser)
            $resolvedUser = $sid.Translate([System.Security.Principal.NTAccount]).Value
        } else {
            $resolvedUser = $principalUser
        }
    } catch {
        $resolvedUser = "UNRESOLVABLE_SID:$principalUser"
    }
    [PSCustomObject]@{
        TaskName       = $t.TaskName
        TaskPath       = $t.TaskPath
        State          = $t.State
        Author         = $t.Author
        Principal      = $principalUser
        ResolvedUser   = $resolvedUser
        LastRun        = $info.LastRunTime
        LastResultHex  = if ($info) { '0x{0:X8}' -f ($info.LastTaskResult -band 0xFFFFFFFF) } else { '' }
        LastResultDec  = if ($info) { $info.LastTaskResult } else { '' }
        Executable     = ($t.Actions.Execute -join ';')
        Args           = (($t.Actions.Arguments -join ';') -replace "`r`n", ' ')
    }
}

$rows | Sort-Object Principal, TaskName | Format-Table -AutoSize -Wrap | Out-String -Width 280 | Out-File $out -Append

"`n--- Likely STALE (UNRESOLVABLE_SID, WsiAccount, DevToolsUser, CodexSandboxOffline) ---" | Out-File $out -Append
$stale = $rows | Where-Object {
    $_.ResolvedUser -match 'UNRESOLVABLE' -or
    $_.Principal -match 'WsiAccount|DevToolsUser|CodexSandboxOffline' -or
    $_.ResolvedUser -match 'WsiAccount|DevToolsUser|CodexSandboxOffline'
}
$stale | Format-Table -AutoSize -Wrap | Out-String -Width 280 | Out-File $out -Append
"Stale count: $($stale.Count)" | Out-File $out -Append

$rows | Export-Csv $csv -NoTypeInformation
Write-Host "Total: $($rows.Count) tasks. Stale: $($stale.Count). Wrote $out and $csv"
