$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\27-stepC.txt'
"=== Step C: Process Lasso + Defender optimization at $(Get-Date -Format o) ===" | Out-File $out

# Phase 1: Defender exclusions (additive, low risk)
"`n=== Defender exclusions (Microsoft recommendation) ===" | Out-File $out -Append
"`n--- BEFORE ---" | Out-File $out -Append
$pre = Get-MpPreference -ErrorAction SilentlyContinue
"ExclusionPath count: $($pre.ExclusionPath.Count)" | Out-File $out -Append
$preOd = $pre.ExclusionPath | Where-Object { $_ -match 'OneDrive' }
"OneDrive paths: $($preOd.Count)" | Out-File $out -Append
"ExclusionProcess count: $($pre.ExclusionProcess.Count)" | Out-File $out -Append
$preOdProc = $pre.ExclusionProcess | Where-Object { $_ -match 'OneDrive|FileSync|FileCoAuth' }
"OneDrive process exclusions: $($preOdProc.Count)" | Out-File $out -Append

# Apply per Microsoft recommendation
$folderToExclude = "$env:USERPROFILE\OneDrive"
$processesToExclude = @(
    'C:\Program Files\Microsoft OneDrive\OneDrive.exe',
    'C:\Program Files\Microsoft OneDrive\26.070.0414.0001\FileSyncHelper.exe',
    'C:\Program Files\Microsoft OneDrive\26.070.0414.0001\FileCoAuth.exe',
    'C:\Program Files\Microsoft OneDrive\26.070.0414.0001\OneDrive.Sync.Service.exe',
    'C:\Program Files\Microsoft OneDrive\OneDriveStandaloneUpdater.exe'
)

"`n--- Adding Defender exclusions ---" | Out-File $out -Append
try {
    Add-MpPreference -ExclusionPath $folderToExclude -ErrorAction Stop
    "  ADDED path: $folderToExclude" | Out-File $out -Append
} catch {
    "  FAILED path: $($_.Exception.Message)" | Out-File $out -Append
}
foreach ($p in $processesToExclude) {
    if (-not (Test-Path $p)) {
        "  SKIPPED process (not found): $p" | Out-File $out -Append
        continue
    }
    try {
        Add-MpPreference -ExclusionProcess $p -ErrorAction Stop
        "  ADDED process: $p" | Out-File $out -Append
    } catch {
        "  FAILED process: $($_.Exception.Message)" | Out-File $out -Append
    }
}

"`n--- AFTER ---" | Out-File $out -Append
$post = Get-MpPreference -ErrorAction SilentlyContinue
$postOd = $post.ExclusionPath | Where-Object { $_ -match 'OneDrive' }
"OneDrive paths: $($postOd.Count)" | Out-File $out -Append
$postOd | ForEach-Object { "  $_" } | Out-File $out -Append
$postOdProc = $post.ExclusionProcess | Where-Object { $_ -match 'OneDrive|FileSync|FileCoAuth' }
"OneDrive process exclusions: $($postOdProc.Count)" | Out-File $out -Append
$postOdProc | ForEach-Object { "  $_" } | Out-File $out -Append

# Phase 2: Process Lasso config edit (snapshot first)
"`n=== Process Lasso priority adjustment ===" | Out-File $out -Append
$cfg = 'C:\ProgramData\ProcessLasso\config\prolasso.ini'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backup = "$cfg.bak-stepC-$timestamp"

if (-not (Test-Path $cfg)) {
    "ERR: prolasso.ini not found at $cfg" | Out-File $out -Append
} else {
    Copy-Item $cfg $backup -Force
    "Backup: $backup" | Out-File $out -Append

    # Read content
    $content = Get-Content $cfg -Raw

    # Save the original lines we will modify
    "`n--- Lines BEFORE ---" | Out-File $out -Append
    $matches = [regex]::Matches($content, '(?im)^DefaultPriorities=.*$|^DefaultIOPriorities=.*$')
    foreach ($m in $matches) {
        "  $($m.Value)" | Out-File $out -Append
    }

    # Replace OneDrive priorities to NORMAL (no priority hint = use default)
    # CPU: change "onedrive.exe,below normal" -> "onedrive.exe,normal"
    # IO: change "onedrive.exe,1" -> "onedrive.exe,2" (Normal-Low IO; Microsoft default is 3)
    # NOTE: leave OneDrive.Sync.Service and FileSyncHelper at below normal/IO=1 because they handle background scanning
    # ONLY relax the user-facing OneDrive.exe

    $newContent = $content -replace '(?i)onedrive\.exe,below normal', 'onedrive.exe,normal'
    $newContent = $newContent -replace '(?i)onedrive\.exe,1', 'onedrive.exe,2'

    # Verify changes were made
    $changes = [regex]::Matches($newContent, '(?im)^DefaultPriorities=.*$|^DefaultIOPriorities=.*$')
    "`n--- Lines AFTER (preview) ---" | Out-File $out -Append
    foreach ($m in $changes) {
        $hasOd = ($m.Value -match '(?i)onedrive\.exe,(normal|2)')
        if ($hasOd) {
            "  $($m.Value)" | Out-File $out -Append
        }
    }

    # Write the file (PowerShell defaults to UTF-16 with -Encoding default; force UTF-8 no BOM)
    [System.IO.File]::WriteAllText($cfg, $newContent, [System.Text.UTF8Encoding]::new($false))
    "Written updated $cfg" | Out-File $out -Append

    # Reload Process Lasso configuration (Process Lasso watches the file or via prolasso.exe)
    "`n--- Reloading Process Lasso config ---" | Out-File $out -Append
    $proLasso = 'C:\Program Files\Process Lasso\prolasso.exe'
    if (Test-Path $proLasso) {
        try {
            Start-Process $proLasso -ArgumentList '/reloadconfig' -WindowStyle Hidden -Wait
            "  /reloadconfig issued" | Out-File $out -Append
        } catch {
            "  reload failed: $($_.Exception.Message)" | Out-File $out -Append
        }
    }
    Start-Sleep -Seconds 3

    "`n--- Live OneDrive priority post-reload (may take a few sec for governor to apply) ---" | Out-File $out -Append
    Get-Process -Name OneDrive,FileSyncHelper,OneDrive.Sync.Service,FileCoAuth -ErrorAction SilentlyContinue |
        Select-Object Name, Id, BasePriority, PriorityClass |
        Format-Table -AutoSize | Out-String | Out-File $out -Append
}

Write-Host "Wrote $out"
