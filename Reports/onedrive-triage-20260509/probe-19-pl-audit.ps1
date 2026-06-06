$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\19-processlasso-audit.txt'
"=== Process Lasso audit for OneDrive at $(Get-Date -Format o) ===" | Out-File $out

"`n--- Process Governor (Lasso) state ---" | Out-File $out -Append
Get-Process -Name 'ProcessGovernor','prolasso' -ErrorAction SilentlyContinue |
    Select-Object Name, Id, Responding, StartTime, @{n='Mem(MB)';e={[math]::Round($_.WorkingSet64/1MB,1)}} |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- prolasso.ini live (relevant OneDrive sections) ---" | Out-File $out -Append
$cfg = 'C:\ProgramData\ProcessLasso\config\prolasso.ini'
if (Test-Path $cfg) {
    "Path: $cfg, mtime $(Get-Item $cfg | ForEach-Object LastWriteTime), size $((Get-Item $cfg).Length) bytes" | Out-File $out -Append
    $content = Get-Content $cfg -Raw
    "OneDrive references in config:" | Out-File $out -Append
    $matches = [regex]::Matches($content, '(?im)^.*?(?:OneDrive|FileSyncHelper|FileCoAuth|OneDriveSetup|OneDriveLauncher|OneDriveStandaloneUpdater|OneDrive\.Sync\.Service).*$')
    if ($matches.Count -eq 0) {
        "  (no OneDrive references found)" | Out-File $out -Append
    } else {
        "Total references: $($matches.Count)" | Out-File $out -Append
        $matches | ForEach-Object { "  $($_.Value.Trim())" } | Out-File $out -Append
    }

    "`nExclusion-related sections containing OneDrive:" | Out-File $out -Append
    $sections = @('ProBalanceExclude','ProBalanceInclude','SmartTrimExclude','HighPerformance','Watchdog','LimitProcessExceptions','HighIOPriority','LowIOPriority','HighCPUPriority','LowCPUPriority')
    foreach ($s in $sections) {
        $sectionRegex = "(?ms)^\[$s\].*?(?=^\[|\z)"
        $sec = [regex]::Match($content, $sectionRegex)
        if ($sec.Success) {
            $sectionMatches = [regex]::Matches($sec.Value, '(?im)^.*(?:OneDrive|FileSyncHelper|FileCoAuth).*$')
            if ($sectionMatches.Count -gt 0) {
                "[$s]" | Out-File $out -Append
                $sectionMatches | ForEach-Object { "  $($_.Value.Trim())" } | Out-File $out -Append
            }
        }
    }
} else {
    "prolasso.ini NOT found at $cfg" | Out-File $out -Append
}

"`n--- Live OneDrive process priority/IO snapshot ---" | Out-File $out -Append
$processes = Get-Process -Name OneDrive,FileSyncHelper,FileCoAuth,OneDrive.Sync.Service -ErrorAction SilentlyContinue
foreach ($p in $processes) {
    "$($p.Name) PID $($p.Id) BasePriority=$($p.BasePriority) PriorityClass=$($p.PriorityClass)" | Out-File $out -Append
}

"`n--- WMIC priority class for OneDrive procs ---" | Out-File $out -Append
$procs = Get-CimInstance Win32_Process -Filter "Name LIKE 'OneDrive%' OR Name LIKE 'FileSync%' OR Name LIKE 'FileCoAuth%'" -ErrorAction SilentlyContinue
$procs | Select-Object Name, ProcessId, Priority, ThreadCount, WorkingSetSize, IoReadOperationCount, IoWriteOperationCount, KernelModeTime, UserModeTime |
    Format-Table -AutoSize | Out-String -Width 200 | Out-File $out -Append

"`n--- Process Lasso recent log lines (mentioning OneDrive) ---" | Out-File $out -Append
$plLog = 'C:\ProgramData\ProcessLasso\logs'
if (Test-Path $plLog) {
    $recent = Get-ChildItem $plLog -Filter '*.txt' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($recent) {
        "Latest log: $($recent.Name) ($([math]::Round($recent.Length/1KB,1)) KB)" | Out-File $out -Append
        Get-Content $recent.FullName -Tail 2000 | Where-Object { $_ -match 'OneDrive|FileSync|FileCoAuth' } |
            Select-Object -Last 30 | Out-File $out -Append
    }
}

Write-Host "Wrote $out"
