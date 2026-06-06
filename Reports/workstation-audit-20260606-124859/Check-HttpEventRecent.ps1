[CmdletBinding()]
param(
    [int]$Minutes = 2
)

$start = (Get-Date).AddMinutes(-$Minutes)
Get-WinEvent -FilterHashtable @{
    LogName = 'System'
    ProviderName = 'Microsoft-Windows-HttpEvent'
    StartTime = $start
} -ErrorAction SilentlyContinue |
    Select-Object TimeCreated,Id,Message |
    ConvertTo-Json -Depth 4
