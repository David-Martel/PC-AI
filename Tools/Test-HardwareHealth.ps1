#Requires -Version 7.0
<#
.SYNOPSIS
Reports device-level hardware faults, biometric/Windows Hello readiness, and
NVIDIA driver-version consistency.

.DESCRIPTION
Device Manager's yellow triangle is the only surface most of these faults have,
and nobody opens Device Manager on a schedule. This turns that inspection into
something runnable, diffable and gate-able.

Three checks, chosen because each has bitten this machine:

  Faulted devices   Any present device not reporting OK, with its CM problem
                    code resolved to text. Code 31 (driver failed to load) and
                    43 (device reported a problem) are the ones that appear here
                    and mean very different things.

  Windows Hello     The biometric devices can all report OK while Hello still
                    does not work, because readiness needs three separate
                    things: healthy sensors, a RUNNING WbioSrvc, and at least
                    one enrolled NGC credential container. WbioSrvc is
                    AUTO_START, so finding it stopped is a real fault, not a
                    trigger-start service idling. An empty NGC store means
                    nothing is enrolled and no amount of service fixing will
                    help - that needs interactive enrolment.

  NVIDIA drivers    Windows ships ONE NVIDIA driver package per system. If two
                    NVIDIA display devices report different driver versions,
                    one of them will typically fail to load with problem 31.
                    That is exactly the state found on 2026-09-09: an eGPU on
                    32.0.15.9636 working while the internal laptop dGPU sat on
                    32.0.16.1088 in error, invisible to CUDA.

.PARAMETER PassThru
Emit the finding objects to the pipeline.

.PARAMETER FailOnIssue
Exit 1 when any ERROR-severity finding exists. WARN findings do not fail.

.PARAMETER IncludeAbsent
Also report devices that are not currently present. Off by default: a laptop
accumulates hundreds of stale enumerations from docks and hot-plug, and they are
not faults.

.PARAMETER OutputJson
Write the machine-readable report to this path. Not written when DryRun is set.

.PARAMETER DryRun
Run every check but write no report file. The long form `--DryRun` also works.

.PARAMETER Help
Print this help and exit. The aliases `-h`, `-?` and `--help` also work.

.EXAMPLE
./Test-HardwareHealth.ps1
Print the hardware health summary.

.EXAMPLE
./Test-HardwareHealth.ps1 -FailOnIssue -OutputJson Reports\hardware-health.json
Write a report and exit non-zero on any error-severity finding.

.NOTES
Read-only. Queries PnP, services and the registry; changes nothing. Starting a
stopped WbioSrvc or reinstalling a GPU driver is deliberately NOT done here -
both are system changes that belong to a human decision, so they are reported
with the exact remedy instead.
#>
[CmdletBinding()]
param(
    [switch]$PassThru,
    [switch]$FailOnIssue,
    [switch]$IncludeAbsent,
    [string]$OutputJson,
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if (@($CliArgs) -contains '--help') {
    $Help = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--help' })
}
if (@($CliArgs) -contains '--DryRun') {
    $DryRun = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--DryRun' })
}
if ($Help) {
    $helpMatch = [regex]::Match((Get-Content -LiteralPath $PSCommandPath -Raw), '(?s)<#\s*(.*?)\s*#>')
    if ($helpMatch.Success) { $helpMatch.Groups[1].Value.Trim() } else { Get-Help -Detailed $PSCommandPath }
    return
}
if (@($CliArgs).Count -gt 0) {
    throw "Unknown CLI argument(s): $($CliArgs -join ', ')"
}

# CM_PROB_* codes actually seen on this class of machine. Anything unlisted is
# reported by number rather than guessed at.
$ProblemCodes = @{
    1  = 'Device not configured correctly'
    3  = 'Driver may be corrupted, or the system is low on memory'
    10 = 'Device cannot start'
    12 = 'Cannot find enough free resources'
    14 = 'Requires a restart to work properly'
    18 = 'Reinstall the drivers for this device'
    19 = 'Registry configuration is incomplete or damaged'
    22 = 'Device is disabled'
    28 = 'Drivers are not installed'
    31 = 'Windows cannot load the drivers required for this device'
    43 = 'Windows stopped this device because it reported problems'
    45 = 'Device is not connected (stale enumeration)'
    52 = 'Cannot verify the digital signature of the drivers'
}

$findings = [System.Collections.Generic.List[object]]::new()
function Add-Finding {
    param(
        [ValidateSet('ERROR', 'WARN', 'INFO')][string]$Severity,
        [string]$Area,
        [string]$Item,
        [string]$Detail,
        [string]$Remedy = ''
    )
    $findings.Add([pscustomobject]@{
            Severity = $Severity
            Area     = $Area
            Item     = $Item
            Detail   = $Detail
            Remedy   = $Remedy
        })
}

# --------------------------------------------------------- faulted devices ---
# -ErrorAction Stop, not SilentlyContinue: a PnP query that fails must not read
# as "no faulted devices". Failing closed matters more than a tidy run.
try {
    $devices = @(Get-PnpDevice -ErrorAction Stop)
} catch {
    Write-Host "FATAL: could not enumerate PnP devices - $($_.Exception.Message)" -ForegroundColor Red
    exit 2
}
if ($devices.Count -eq 0) {
    Write-Host 'FATAL: Get-PnpDevice returned nothing - refusing to report health.' -ForegroundColor Red
    exit 2
}

$scope = if ($IncludeAbsent) { $devices } else { @($devices | Where-Object { $_.Present }) }
foreach ($d in ($scope | Where-Object { $_.Status -ne 'OK' -and $_.Status -ne 'Unknown' })) {
    $code = $null
    try {
        $code = (Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName 'DEVPKEY_Device_ProblemCode' -ErrorAction Stop).Data
    } catch { }
    # A device can report Error while exposing no problem code - some virtual
    # adapters do. Say that plainly rather than printing "problem  :".
    $codeText = if ($null -eq $code -or "$code" -eq '') { 'not reported' } else { "$code" }
    $text = if ($null -ne $code -and "$code" -ne '' -and $ProblemCodes.ContainsKey([int]$code)) {
        $ProblemCodes[[int]$code]
    } elseif ($codeText -eq 'not reported') {
        'Device reports Error but exposes no CM problem code'
    } else {
        'Unrecognised problem code'
    }
    $remedy = switch ($(if ($codeText -eq 'not reported') { -1 } else { [int]$code })) {
        31 { 'Driver failed to load. If another device of the same vendor works, compare driver versions - a mixed-version install is the usual cause.' }
        43 { 'Device reported a fault. Often a failed USB enumeration on a hub or dock port rather than the peripheral itself.' }
        -1 { 'Often a virtual adapter left behind by an uninstalled VPN or hypervisor client. Remove it if the product is gone.' }
        default { 'Inspect in Device Manager.' }
    }
    Add-Finding -Severity 'ERROR' -Area 'Device' -Item $d.FriendlyName `
        -Detail ("$($d.Class) - problem $codeText : $text") -Remedy $remedy
}

# ------------------------------------------------------------ Windows Hello ---
# Readiness is three independent things. Reporting only one of them is how "the
# sensors are fine so Hello should work" happens.
$bio = @($devices | Where-Object { $_.Class -eq 'Biometric' -and $_.Present })
if ($bio.Count -eq 0) {
    Add-Finding -Severity 'INFO' -Area 'Hello' -Item 'Biometric devices' -Detail 'None present.'
} else {
    $badBio = @($bio | Where-Object { $_.Status -ne 'OK' })
    if ($badBio.Count -gt 0) {
        Add-Finding -Severity 'ERROR' -Area 'Hello' -Item 'Biometric devices' `
            -Detail ("Not OK: " + (($badBio | ForEach-Object { $_.FriendlyName }) -join '; '))
    } else {
        Add-Finding -Severity 'INFO' -Area 'Hello' -Item 'Biometric devices' `
            -Detail ("$($bio.Count) present, all OK")
    }
}

$wbio = Get-Service -Name WbioSrvc -ErrorAction SilentlyContinue
if ($null -eq $wbio) {
    Add-Finding -Severity 'WARN' -Area 'Hello' -Item 'WbioSrvc' -Detail 'Service not installed.'
} elseif ($wbio.Status -ne 'Running') {
    # AUTO_START and stopped is a genuine fault. Do not confuse this with a
    # trigger-start service that idles until first use.
    Add-Finding -Severity 'ERROR' -Area 'Hello' -Item 'WbioSrvc' `
        -Detail "Windows Biometric Service is $($wbio.Status) (StartType $($wbio.StartType))." `
        -Remedy 'Start-Service WbioSrvc. If StartType is Automatic, being stopped is a fault, not idling.'
} else {
    Add-Finding -Severity 'INFO' -Area 'Hello' -Item 'WbioSrvc' -Detail 'Running.'
}

# NGC holds the Hello credential containers. Empty means nothing is enrolled,
# which no amount of service or driver repair will fix.
$ngc = Join-Path $env:SystemRoot 'ServiceProfiles\LocalService\AppData\Local\Microsoft\Ngc'
if (Test-Path -LiteralPath $ngc) {
    $containers = @(Get-ChildItem -LiteralPath $ngc -Force -Directory -ErrorAction SilentlyContinue)
    if ($containers.Count -eq 0) {
        Add-Finding -Severity 'WARN' -Area 'Hello' -Item 'NGC credential store' `
            -Detail 'Zero containers - no Windows Hello credential is enrolled.' `
            -Remedy 'Enrol interactively: Settings > Accounts > Sign-in options. Cannot be scripted.'
    } else {
        Add-Finding -Severity 'INFO' -Area 'Hello' -Item 'NGC credential store' `
            -Detail "$($containers.Count) container(s) enrolled."
    }
} else {
    Add-Finding -Severity 'WARN' -Area 'Hello' -Item 'NGC credential store' -Detail 'Path not found.'
}

# ------------------------------------------------------ NVIDIA driver parity ---
$nv = @($devices | Where-Object { $_.Class -eq 'Display' -and $_.Present -and $_.FriendlyName -match 'NVIDIA' })
$nvInfo = foreach ($g in $nv) {
    $ver = $null
    try { $ver = (Get-PnpDeviceProperty -InstanceId $g.InstanceId -KeyName 'DEVPKEY_Device_DriverVersion' -ErrorAction Stop).Data } catch { }
    [pscustomobject]@{ Name = $g.FriendlyName; Status = $g.Status; Driver = $ver }
}
$nvInfo = @($nvInfo)
$distinct = @($nvInfo | Where-Object { $_.Driver } | Select-Object -ExpandProperty Driver -Unique)
if ($distinct.Count -gt 1) {
    Add-Finding -Severity 'ERROR' -Area 'GPU' -Item 'NVIDIA driver versions' `
        -Detail ("Mixed versions across $($nvInfo.Count) NVIDIA GPUs: " + ($distinct -join ', ')) `
        -Remedy 'Windows ships one NVIDIA package per system. Reinstall a single driver (clean install) covering every NVIDIA GPU; a mixed install is the usual cause of problem 31.'
} elseif ($nvInfo.Count -gt 0) {
    Add-Finding -Severity 'INFO' -Area 'GPU' -Item 'NVIDIA driver versions' `
        -Detail ("$($nvInfo.Count) NVIDIA GPU(s), all on $($distinct -join ', ')")
}

# ------------------------------------------------------------------ output ---
$errors = @($findings | Where-Object { $_.Severity -eq 'ERROR' })
$warns = @($findings | Where-Object { $_.Severity -eq 'WARN' })

$summary = [pscustomobject]@{
    GeneratedAt   = (Get-Date).ToString('o')
    Computer      = $env:COMPUTERNAME
    DevicesTotal  = $devices.Count
    DevicesPresent = @($devices | Where-Object { $_.Present }).Count
    Errors        = $errors.Count
    Warnings      = $warns.Count
    NvidiaGpus    = $nvInfo
    Findings      = @($findings)
}

foreach ($f in ($findings | Sort-Object { switch ($_.Severity) { 'ERROR' { 0 } 'WARN' { 1 } default { 2 } } })) {
    $colour = switch ($f.Severity) { 'ERROR' { 'Red' } 'WARN' { 'Yellow' } default { 'Gray' } }
    Write-Host ("[{0,-5}] {1,-7} {2}" -f $f.Severity, $f.Area, $f.Item) -ForegroundColor $colour
    Write-Host "         $($f.Detail)"
    if ($f.Remedy) { Write-Host "         -> $($f.Remedy)" -ForegroundColor DarkGray }
}
Write-Host ''
Write-Host ("Devices present {0} | errors {1} | warnings {2}" -f $summary.DevicesPresent, $errors.Count, $warns.Count)

if (-not [string]::IsNullOrWhiteSpace($OutputJson) -and -not $DryRun) {
    $dir = Split-Path -Parent $OutputJson
    if (-not [string]::IsNullOrWhiteSpace($dir) -and -not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $OutputJson -Encoding utf8
    Write-Host "Report written to $OutputJson"
}

if ($PassThru) { $findings }
if ($FailOnIssue -and $errors.Count -gt 0) { exit 1 }
exit 0
