$ErrorActionPreference = 'Continue'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\12-writetest.txt'
"=== OneDrive write-test at $(Get-Date -Format o) ===" | Out-File $out

$testRoot = "$env:USERPROFILE\OneDrive\_synctest_20260509"
$null = New-Item -ItemType Directory -Path $testRoot -Force -ErrorAction SilentlyContinue
$testFile = Join-Path $testRoot "synctest-$(Get-Date -Format yyyyMMdd-HHmmss).txt"
$content = @"
OneDrive sync write-test
Generated at $(Get-Date -Format o)
Computer: $env:COMPUTERNAME
User: $env:USERNAME
Test ID: $([Guid]::NewGuid())
Purpose: verify upload pipeline by introducing a fresh small file
"@
$content | Out-File $testFile -Encoding UTF8 -NoNewline
"Test file created: $testFile (size=$((Get-Item $testFile).Length) bytes)" | Out-File $out -Append

# Capture pre-state from running OneDrive diagnostic via repair tool
$preDir = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\writetest-pre'
$null = New-Item -ItemType Directory -Path $preDir -Force
$preEv = Join-Path $preDir 'pre.json'
"Capturing pre-state via Repair-OneDriveSync (DryRun)..." | Out-File $out -Append
& 'C:\codedev\PC_AI\Tools\Repair-OneDriveSync.ps1' -OutputDirectory $preDir -SinceMinutes 5 -DryRun *>&1 | Out-Null
$pre = Get-ChildItem $preDir -Filter 'pre-repair-evidence.json' | Select-Object -First 1
if ($pre) {
    $preData = (Get-Content $pre.FullName -Raw | ConvertFrom-Json).SyncDiagnostics[0].Values
    "Pre-state at $(Get-Date -Format o):" | Out-File $out -Append
    foreach ($k in 'driveChangesToSend','driveSentChanges','numLocalChanges','numFileUploads','numFileFailedUploads','BytesUploaded','BytesToUpload','FilesToUpload','syncProgressState','scanState','uptimeSecs','timeUtc') {
        "  pre.$k = $($preData.$k)" | Out-File $out -Append
    }
}

"`n--- Waiting 90 seconds for sync ---" | Out-File $out -Append
Start-Sleep -Seconds 90

# Capture post-state
$postDir = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\writetest-post'
$null = New-Item -ItemType Directory -Path $postDir -Force
& 'C:\codedev\PC_AI\Tools\Repair-OneDriveSync.ps1' -OutputDirectory $postDir -SinceMinutes 5 -DryRun *>&1 | Out-Null
$post = Get-ChildItem $postDir -Filter 'pre-repair-evidence.json' | Select-Object -First 1
if ($post) {
    $postData = (Get-Content $post.FullName -Raw | ConvertFrom-Json).SyncDiagnostics[0].Values
    "`nPost-state at $(Get-Date -Format o):" | Out-File $out -Append
    foreach ($k in 'driveChangesToSend','driveSentChanges','numLocalChanges','numFileUploads','numFileFailedUploads','BytesUploaded','BytesToUpload','FilesToUpload','syncProgressState','scanState','uptimeSecs','timeUtc') {
        "  post.$k = $($postData.$k)" | Out-File $out -Append
    }
}

"`n--- Compare ---" | Out-File $out -Append
foreach ($k in 'driveChangesToSend','driveSentChanges','numLocalChanges','numFileUploads','BytesUploaded','timeUtc') {
    $a = $preData.$k; $b = $postData.$k
    "  $k : $a -> $b" | Out-File $out -Append
}

"`n--- File status check ---" | Out-File $out -Append
$f = Get-Item $testFile -Force
$attrs = $f.Attributes
"Test file attrs: $attrs" | Out-File $out -Append
"  IsOffline (placeholder): $(($attrs -band [IO.FileAttributes]::Offline) -ne 0)" | Out-File $out -Append
"  IsSparseFile: $(($attrs -band [IO.FileAttributes]::SparseFile) -ne 0)" | Out-File $out -Append

# Look for the file in CldFlt2 ETL via shell sync status
try {
    $shell = New-Object -ComObject Shell.Application
    $folderItem = $shell.Namespace((Split-Path $testFile)).ParseName((Split-Path $testFile -Leaf))
    if ($folderItem) {
        $folder = $shell.Namespace((Split-Path $testFile))
        for ($i = 0; $i -lt 320; $i++) {
            $h = $folder.GetDetailsOf($null, $i)
            if ($h -in 'Status','Sync status','Availability status','State','Sharing status') {
                $v = $folder.GetDetailsOf($folderItem, $i)
                if ($v) { "  $h (col $i) = $v" | Out-File $out -Append }
            }
        }
    }
} catch {
    "Shell COM error: $($_.Exception.Message)" | Out-File $out -Append
}

Write-Host "Wrote $out"
Get-Content $out
