[CmdletBinding()]
param([int]$SinceHours = 24)

$reportRoot = Split-Path -Parent $PSCommandPath
$start = (Get-Date).AddHours(-$SinceHours)
$providers = @('Bonjour Service','Netwaw18','Microsoft-Windows-DistributedCOM','Microsoft-Windows-CertificateServicesClient-AutoEnrollment','Microsoft-Windows-Smartcard-Server','Service Control Manager','VBScriptDeprecationAlert')
$allEvents = foreach ($logName in @('System','Application')) {
    Get-WinEvent -FilterHashtable @{ LogName = $logName; StartTime = $start; Level = 1,2,3 } -ErrorAction SilentlyContinue
}

foreach ($provider in $providers) {
    $safe = $provider -replace '[\\/:*?"<>| ]','_'
    $allEvents |
        Where-Object { $_.ProviderName -eq $provider } |
        Sort-Object TimeCreated -Descending |
        Select-Object -First 20 TimeCreated,ProviderName,Id,LevelDisplayName,Message |
        ConvertTo-Json -Depth 5 |
        Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot "events-provider-$safe.json")
}
