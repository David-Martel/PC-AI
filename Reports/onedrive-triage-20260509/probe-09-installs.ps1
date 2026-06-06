$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\09-installs.txt'
"=== OneDrive install layout at $(Get-Date -Format o) ===" | Out-File $out

"`n--- Per-Machine install (C:\Program Files\Microsoft OneDrive) ---" | Out-File $out -Append
$pm = 'C:\Program Files\Microsoft OneDrive'
if (Test-Path $pm) {
    Get-ChildItem $pm -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object Name, LastWriteTime, @{n='Type';e={if ($_.PSIsContainer) { 'Dir' } else { '{0:N1} KB' -f ($_.Length/1KB) }}} |
        Format-Table -AutoSize | Out-String | Out-File $out -Append
    "`n--- Versioned subdirs ---" | Out-File $out -Append
    Get-ChildItem $pm -Directory | Where-Object { $_.Name -match '^\d' } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 5 |
        ForEach-Object {
            "--- $($_.Name) ---" | Out-File $out -Append
            Get-ChildItem $_.FullName -Filter '*.exe' | Select-Object Name, @{n='SizeKB';e={[math]::Round($_.Length/1KB,1)}}, @{n='Ver';e={$_.VersionInfo.FileVersion}} | Format-Table -AutoSize | Out-String | Out-File $out -Append
            "FileCoAuth.exe present: $(Test-Path "$($_.FullName)\FileCoAuth.exe")" | Out-File $out -Append
            "OneDriveLauncher.exe present: $(Test-Path "$($_.FullName)\OneDriveLauncher.exe")" | Out-File $out -Append
            "OneDrive.exe present: $(Test-Path "$($_.FullName)\OneDrive.exe")" | Out-File $out -Append
        }
} else {
    "Per-machine install NOT found" | Out-File $out -Append
}

"`n--- Per-User install (LOCALAPPDATA) ---" | Out-File $out -Append
$pu = "$env:LOCALAPPDATA\Microsoft\OneDrive"
Get-ChildItem $pu -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object Name, LastWriteTime, @{n='Type';e={if ($_.PSIsContainer) { 'Dir' } else { '{0:N1} KB' -f ($_.Length/1KB) }}} |
    Format-Table -AutoSize | Out-String | Out-File $out -Append

"`n--- Currently running OneDrive image path ---" | Out-File $out -Append
Get-CimInstance Win32_Process -Filter "Name='OneDrive.exe'" -ErrorAction SilentlyContinue |
    Select-Object ProcessId, ExecutablePath, CreationDate, CommandLine |
    Format-List | Out-String | Out-File $out -Append

"`n--- HKCU OneDrive Update info ---" | Out-File $out -Append
Get-ItemProperty 'HKCU:\Software\Microsoft\OneDrive' -ErrorAction SilentlyContinue |
    Select-Object Version, InstallationType, IsPerMachineInstall, OneDriveTrigger_Background, LastUpdate |
    Format-List | Out-String | Out-File $out -Append

"`n--- HKLM OneDrive ---" | Out-File $out -Append
Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\OneDrive' -ErrorAction SilentlyContinue |
    Select-Object * -ExcludeProperty PS* | Format-List | Out-String | Out-File $out -Append

Write-Host "Wrote $out"
Get-Content $out -TotalCount 60
