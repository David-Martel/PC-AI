$ErrorActionPreference = 'Continue'
$pre = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\liverepair-readonly\pre-repair-evidence.json'
$out = 'C:\codedev\PC_AI\Reports\onedrive-triage-20260509\11-live-sync-current.txt'
"=== Live sync diagnostic at $(Get-Date -Format o) ===" | Out-File $out
"Source: $pre" | Out-File $out -Append
$j = Get-Content $pre -Raw | ConvertFrom-Json

"`n--- Top-level keys ---" | Out-File $out -Append
$j.PSObject.Properties.Name | Out-File $out -Append

"`n--- SyncValues block (full) ---" | Out-File $out -Append
$sv = $j.SyncValues
if ($sv) { $sv | Format-List | Out-String | Out-File $out -Append }
else { '(no SyncValues field)' | Out-File $out -Append }

"`n--- Specific upload-related fields ---" | Out-File $out -Append
$pick = 'clientVersion','timeUtc','uptimeSecs','vaultState','placeholdersEnabled',
        'driveChangesToSend','driveSentChanges','scanState','scanStateStallDetected',
        'syncProgressState','syncStallDetected','numLocalChanges',
        'activeHydrations','passiveHydrations',
        'numFileUploads','numFileDownloads','numFileFailedUploads','numFileFailedDownloads',
        'numUploadErrorsReported','numDownloadErrorsReported','numHashMismatchErrorsReported',
        'BytesToUpload','BytesUploaded','FilesToUpload','ChangesToProcess',
        'wasFileDBReset','fullScanCount','preciseScanCount','invalidatedScanCount',
        'drivesScanRequested','drivesChangeEnumPending','drivesWaitingForInitialSync',
        'numAbortedReSyncNeeded','numResyncs','dailyCogsImpact'
foreach ($k in $pick) {
    if ($null -ne $sv.$k) { "  $k = $($sv.$k)" | Out-File $out -Append }
    elseif ($null -ne $j.$k) { "  $k = $($j.$k)" | Out-File $out -Append }
}

Write-Host "Wrote $out"
Get-Content $out
