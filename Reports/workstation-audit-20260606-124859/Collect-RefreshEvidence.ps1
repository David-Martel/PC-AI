[CmdletBinding()]
param(
    [int]$SinceMinutes = 240
)

$ErrorActionPreference = 'Continue'
$reportRoot = Split-Path -Parent $PSCommandPath
$repoRoot = Split-Path -Parent (Split-Path -Parent $reportRoot)

function Write-Step {
    param([string]$Name)
    Write-Host "== $Name =="
}

Push-Location $repoRoot
try {
    Write-Step 'Sync provider health'
    & .\Tools\Test-SyncProviderHealth.ps1 -SinceMinutes $SinceMinutes -PassThru |
        Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'sync-provider-health-refresh.txt')

    Write-Step 'Boot mount health'
    & .\Tools\Test-BootMountHealth.ps1 -SinceMinutes $SinceMinutes -PassThru |
        Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'boot-mount-health-refresh.txt')

    Write-Step 'Process Lasso boot safety'
    & .\Tools\Test-ProcessLassoBootSafety.ps1 |
        Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'process-lasso-refresh.txt')

    Write-Step 'Present PnP devices not OK'
    Get-PnpDevice -PresentOnly |
        Where-Object Status -ne 'OK' |
        Select-Object Class,FriendlyName,InstanceId,Status,Problem |
        Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'pnp-present-not-ok-refresh.csv')

    Write-Step 'Recent critical/error/warning event summary'
    $start = (Get-Date).AddMinutes(-$SinceMinutes)
    Get-WinEvent -FilterHashtable @{ LogName = @('System','Application'); StartTime = $start; Level = 1,2,3 } -ErrorAction SilentlyContinue |
        Group-Object ProviderName,Id,LevelDisplayName |
        Sort-Object Count -Descending |
        Select-Object Count,Name |
        Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'events-refresh-summary.txt')

    Write-Step 'Recent high-signal event samples'
    $providers = @(
        'Microsoft-Windows-Kernel-Power',
        'Microsoft-Windows-Kernel-PnP',
        'Microsoft-Windows-DriverFrameworks-UserMode',
        'Microsoft-Windows-FilterManager',
        'disk',
        'Application Error',
        'Windows Error Reporting'
    )
    # See Collect-RemainingEventSources.ps1 for the full rationale. In short:
    # -ErrorAction SilentlyContinue made a FAILED query look identical to a
    # successful one that matched nothing, and ConvertTo-Json on an empty
    # pipeline emits nothing, so Out-File wrote a zero-byte file that is not
    # valid JSON. Keep the two outcomes distinguishable, and always write JSON.
    $refreshPath = Join-Path $reportRoot 'events-refresh-samples.json'
    $queryError = $null
    $events = @()
    try {
        $events = @(Get-WinEvent -FilterHashtable @{ LogName = @('System', 'Application'); StartTime = $start; Level = 1, 2, 3 } -ErrorAction Stop)
    } catch {
        # Stable error id, not localized message text -- see the note in
        # Collect-RemainingEventSources.ps1. A message match would classify a
        # successful empty query as a failure on non-English Windows.
        if ($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound,*') {
            $queryError = '{0} [{1}]' -f $_.Exception.Message, $_.FullyQualifiedErrorId
        }
    }

    $rows = @($events |
        Where-Object { $providers -contains $_.ProviderName } |
        Sort-Object TimeCreated -Descending |
        Select-Object -First 80 TimeCreated, ProviderName, Id, LevelDisplayName, Message)

    if ($queryError) {
        [pscustomobject]@{
            captureStatus = 'failed'
            queryError    = $queryError
            events        = $rows
        } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $refreshPath -Encoding utf8
    } else {
        # The leading comma is load-bearing: an empty pipeline makes
        # ConvertTo-Json emit nothing, and -AsArray does not help because the
        # cmdlet is never invoked. `,$rows` passes the array as a single object.
        , $rows | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $refreshPath -Encoding utf8
    }

    Write-Step 'Docker storage'
    docker system df | Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'docker-system-df-refresh.txt')
    docker ps -a --format 'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}\t{{.Size}}' |
        Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'docker-ps-refresh.txt')

    Write-Step 'WSL state'
    wsl --status | Out-File -Encoding unicode -LiteralPath (Join-Path $reportRoot 'wsl-status-refresh.txt')
    wsl --list --verbose | Out-File -Encoding unicode -LiteralPath (Join-Path $reportRoot 'wsl-list-verbose-refresh.txt')

    Write-Step 'Logical disks'
    Get-CimInstance Win32_LogicalDisk |
        Select-Object DeviceID,VolumeName,FileSystem,
            @{Name='SizeGB';Expression={[math]::Round($_.Size / 1GB, 2)}},
            @{Name='FreeGB';Expression={[math]::Round($_.FreeSpace / 1GB, 2)}},
            @{Name='FreePct';Expression={if ($_.Size) { [math]::Round(100 * $_.FreeSpace / $_.Size, 2) } else { $null }}} |
        Format-List |
        Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'logical-disks-refresh.txt')

    Write-Step 'Done'
} finally {
    Pop-Location
}
