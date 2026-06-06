[CmdletBinding()]
param([int]$Minutes = 5)

$start = (Get-Date).AddMinutes(-$Minutes)
Get-WinEvent -FilterHashtable @{
    LogName = @('System','Application')
    StartTime = $start
    Level = 1,2,3
} -ErrorAction SilentlyContinue |
    Group-Object ProviderName,Id,LevelDisplayName |
    Sort-Object Count -Descending |
    Select-Object Count,Name |
    ConvertTo-Json -Depth 4
