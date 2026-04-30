$since = (Get-Date).AddDays(-7)

# Check which virtualization logs exist
Write-Host "=== Available Virtualization Logs ===" -ForegroundColor Cyan
Get-WinEvent -ListLog * -ErrorAction SilentlyContinue | Where-Object { $_.LogName -match 'Hyper|Virtual|Lxss' } | Select-Object LogName, RecordCount, IsEnabled | Format-Table -AutoSize

# Check for recent Hyper-V errors
Write-Host "`n=== Recent Hyper-V/Virtualization Errors ===" -ForegroundColor Cyan
$hvLogs = @('Microsoft-Windows-Hyper-V-Compute-Admin', 'Microsoft-Windows-Hyper-V-Compute-Operational', 'Microsoft-Windows-Hyper-V-VMMS-Admin')
foreach ($log in $hvLogs) {
    $events = Get-WinEvent -LogName $log -ErrorAction SilentlyContinue | Where-Object { $_.TimeCreated -ge $since -and $_.LevelDisplayName -in 'Error','Warning' } | Select-Object -First 5
    if ($events) {
        Write-Host "`nLog: $log" -ForegroundColor Yellow
        $events | Select-Object TimeCreated, Id, LevelDisplayName, Message | Format-List
    }
}

# Check Application log for .NET OutOfMemory
Write-Host "`n=== .NET OutOfMemory Exceptions ===" -ForegroundColor Cyan
$dotnetErrors = Get-WinEvent -LogName Application -ErrorAction SilentlyContinue | Where-Object { $_.TimeCreated -ge $since -and $_.Message -match 'OutOfMemoryException' } | Select-Object -First 10
if ($dotnetErrors) {
    $dotnetErrors | Select-Object TimeCreated, ProviderName, Id, @{n='Message';e={$_.Message.Substring(0, [Math]::Min(200, $_.Message.Length))}} | Format-Table -AutoSize
} else {
    Write-Host "No OutOfMemoryException events found"
}
