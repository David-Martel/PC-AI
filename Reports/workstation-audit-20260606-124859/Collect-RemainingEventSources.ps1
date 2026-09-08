#Requires -Version 7.0
[CmdletBinding()]
param([int]$SinceHours = 24)

$reportRoot = Split-Path -Parent $PSCommandPath
$start = (Get-Date).AddHours(-$SinceHours)
$providers = @('Bonjour Service','Netwaw18','Microsoft-Windows-DistributedCOM','Microsoft-Windows-CertificateServicesClient-AutoEnrollment','Microsoft-Windows-Smartcard-Server','Service Control Manager','VBScriptDeprecationAlert')

# Why this is not a plain pipeline any more
#
# The original was:
#
#   $allEvents = Get-WinEvent ... -ErrorAction SilentlyContinue
#   $allEvents | Where-Object {...} | ConvertTo-Json | Out-File ...
#
# which produced ZERO-BYTE .json files, for two compounding reasons:
#
#   1. -ErrorAction SilentlyContinue collapsed two different outcomes into one
#      empty result: a query that succeeded and matched nothing, and a query
#      that FAILED (log unreadable, access denied). Get-WinEvent also raises
#      "No events were found..." as an error for the successful-but-empty
#      case, so both landed in the same bucket.
#   2. ConvertTo-Json on an EMPTY PIPELINE emits nothing at all -- not "[]" --
#      so Out-File wrote a zero-length file, which is not valid JSON.
#
# Those two outcomes must stay distinguishable. Writing "[]" after a failed
# query would assert "this provider logged nothing", a claim the run cannot
# support. So an empty-but-successful query writes [], and a failed query
# writes a record saying so.
#
# Note the `,$rows |` below. It is load-bearing, and the obvious alternatives
# are not:
#     @()  | ConvertTo-Json -AsArray               -> '' (empty pipeline again)
#     ConvertTo-Json -InputObject @() -AsArray     -> '[[]]' (double wrapped)
#     ,$rows | ConvertTo-Json                      -> '[]', '[{...}]' correctly
# A List rather than `+=`: `$a += @(...)` reallocates and copies the whole
# array on every iteration, which is wasteful when a busy System log returns
# tens of thousands of events.
$allEvents = [System.Collections.Generic.List[object]]::new()
$queryFailures = [System.Collections.Generic.List[object]]::new()
foreach ($logName in @('System', 'Application')) {
    try {
        $allEvents.AddRange(@(Get-WinEvent -FilterHashtable @{ LogName = $logName; StartTime = $start; Level = 1, 2, 3 } -ErrorAction Stop))
    } catch {
        # Classify on the STABLE FullyQualifiedErrorId, never on message text.
        # Get-WinEvent's "No events were found that match the specified
        # selection criteria." is localized, so a message match would record a
        # successful empty query as a FAILURE on any non-English Windows --
        # converting a locale difference into false evidence, which is the
        # exact failure mode this rewrite exists to prevent. Id verified by
        # triggering the case directly:
        #   NoMatchingEventsFound,Microsoft.PowerShell.Commands.GetWinEventCommand
        if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound,*') {
            continue   # successful query, zero matches
        }
        $queryFailures.Add([pscustomobject]@{
                LogName = $logName
                Error   = $_.Exception.Message
                ErrorId = $_.FullyQualifiedErrorId
            })
    }
}

foreach ($provider in $providers) {
    $safe = $provider -replace '[\\/:*?"<>| ]', '_'
    $path = Join-Path $reportRoot "events-provider-$safe.json"

    $rows = @($allEvents |
        Where-Object { $_.ProviderName -eq $provider } |
        Sort-Object TimeCreated -Descending |
        Select-Object -First 20 TimeCreated, ProviderName, Id, LevelDisplayName, Message)

    if ($queryFailures.Count -gt 0) {
        [pscustomobject]@{
            captureStatus = 'partial-or-failed'
            provider      = $provider
            queryFailures = $queryFailures
            events        = $rows
        } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $path -Encoding utf8
    } else {
        , $rows | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $path -Encoding utf8
    }
}
