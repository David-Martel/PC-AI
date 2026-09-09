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

  Windows Hello     Reported from sources that are actually READABLE, which is
                    the whole difficulty. The obvious one is not: the NGC
                    credential directory is ACL'd to SYSTEM and NgcCtnrSvc only,
                    so even an elevated enumeration is denied - and with
                    -ErrorAction SilentlyContinue that denial returns an empty
                    collection indistinguishable from "nothing is enrolled". On
                    2026-09-09 that produced a confidently wrong verdict ("no
                    Hello credential enrolled") on a machine where PIN and Face
                    both worked. Enrolment is therefore read from WinBio
                    AccountInfo\<SID>\EnrolledFactors instead.

                    WbioSrvc state is reported as CONTEXT, never as a verdict.
                    It is Start=2 but carries RPC start triggers: LogonUI starts
                    it on demand at the lock screen and it idle-stops after, so
                    Stopped is its normal resting state mid-session and starting
                    it by hand does not stick. Recent 1609 sensor errors ARE
                    diagnostic and are surfaced instead.

  NVIDIA drivers    The fault to look for is a VERSION split, not a "branch"
                    incompatibility. nvlddmkm.sys is one shared kernel driver,
                    so two packages from different releases cannot both load and
                    the loser reports problem 31. Different INFs at the same
                    version are normal - one release ships ~45 OEM-specific INFs
                    and none lists every device. Verified 2026-09-09: an RTX 2000
                    Ada (nvltsi.inf) and a GeForce RTX 5060 Ti (nv_dispsi.inf)
                    from the SAME 610.88 package both work.

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

# NGC holds the Hello credential containers. Establish enrolment FIRST, because
# it decides whether a stopped biometric service is a fault or just idle.
$ngc = Join-Path $env:SystemRoot 'ServiceProfiles\LocalService\AppData\Local\Microsoft\Ngc'
$enrolled = $null   # $null = unknown, otherwise a count
if (Test-Path -LiteralPath $ngc) {
    # -ErrorAction Stop, not SilentlyContinue. The NGC store is ACL'd to SYSTEM
    # and NgcCtnrSvc, so an unelevated or restricted run gets an access denial -
    # which SilentlyContinue would turn into an empty collection, i.e. "nothing
    # enrolled". That is a false negative twice over: it hides real enrolments
    # AND it downgrades a genuinely stopped WbioSrvc from ERROR to INFO.
    # $enrolled stays $null on failure so downstream checks know it is UNKNOWN.
    $containers = $null
    try {
        $containers = @(Get-ChildItem -LiteralPath $ngc -Force -Directory -ErrorAction Stop)
    } catch {
        # Expected, and NOT worth a warning. The ACL grants SYSTEM and
        # NgcCtnrSvc only - Administrators are excluded - so elevation does not
        # help and telling the operator to re-run elevated wastes their time.
        # Recorded as context purely so nobody mistakes the denial for an empty
        # store again. Enrolment comes from WinBio AccountInfo below.
        Add-Finding -Severity 'INFO' -Area 'Hello' -Item 'NGC credential store' `
            -Detail 'Not enumerable (ACL grants SYSTEM/NgcCtnrSvc only). Expected - enrolment is read from WinBio instead, never from this count.'
    }
    if ($null -ne $containers) {
        $enrolled = $containers.Count
        Add-Finding -Severity 'INFO' -Area 'Hello' -Item 'NGC credential store' `
            -Detail "$enrolled container(s) readable."
    }
} else {
    Add-Finding -Severity 'WARN' -Area 'Hello' -Item 'NGC credential store' -Detail 'Path not found.'
}

# The NGC directory is ACL'd to SYSTEM and NgcCtnrSvc ONLY - not Administrators -
# so even an elevated run cannot enumerate it. Reading "0 containers" from it is
# therefore an access denial wearing a costume, and treating that as "nothing is
# enrolled" is exactly the false negative that produced a wrong verdict here on
# 2026-09-09 (reported "no Hello credential enrolled" on a machine with PIN and
# Face both working). Authoritative, readable sources instead:
#   WinBio AccountInfo\<SID>\EnrolledFactors - biometric factors, per user
#   Passport KSP key list                    - PIN (uvkey-*) and FIDO passkeys
$factors = 0
$bioAccounts = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WinBio\AccountInfo'
$mySid = ([System.Security.Principal.WindowsIdentity]::GetCurrent()).User.Value
if (Test-Path -LiteralPath $bioAccounts) {
    $mine = Get-ItemProperty -Path (Join-Path $bioAccounts $mySid) -ErrorAction SilentlyContinue
    $factors = if ($mine -and $mine.PSObject.Properties.Name -contains 'EnrolledFactors') { [int]$mine.EnrolledFactors } else { 0 }
    # WINBIO_TYPE_* bitmask: 2 = FacialFeatures, 8 = Fingerprint.
    $names = @()
    if ($factors -band 2) { $names += 'Face' }
    if ($factors -band 8) { $names += 'Fingerprint' }
    if ($factors -eq 0) {
        Add-Finding -Severity 'INFO' -Area 'Hello' -Item 'Biometric enrolment' `
            -Detail 'No biometric factor enrolled for the current user.' `
            -Remedy 'Settings > Accounts > Sign-in options. Enrolment is interactive by design.'
    } else {
        $missing = @()
        if (-not ($factors -band 2)) { $missing += 'Face' }
        if (-not ($factors -band 8)) { $missing += 'Fingerprint' }
        $detail = "Enrolled: $($names -join ', ') (EnrolledFactors=$factors)."
        if ($missing.Count -gt 0) { $detail += " Not enrolled: $($missing -join ', ')." }
        Add-Finding -Severity 'INFO' -Area 'Hello' -Item 'Biometric enrolment' -Detail $detail
    }
} else {
    Add-Finding -Severity 'WARN' -Area 'Hello' -Item 'Biometric enrolment' -Detail 'WinBio AccountInfo key absent.'
}

$wbio = Get-Service -Name WbioSrvc -ErrorAction SilentlyContinue
if ($null -eq $wbio) {
    Add-Finding -Severity 'WARN' -Area 'Hello' -Item 'WbioSrvc' -Detail 'Service not installed.'
} elseif ($wbio.Status -eq 'Running') {
    Add-Finding -Severity 'INFO' -Area 'Hello' -Item 'WbioSrvc' -Detail 'Running.'
} else {
    # Service state is NOT a health verdict here, and three attempts to make it
    # one all produced false positives. WbioSrvc is Start=2 but carries
    # RPC-interface start triggers: LogonUI starts it at the lock screen when a
    # biometric is actually needed, and it idle-stops afterwards. So Stopped is
    # its normal resting state DURING a session, whether or not anything is
    # enrolled, and starting it by hand does not stick.
    # Report it as context. The real signals are enrolment (below), device status,
    # and recent Microsoft-Windows-Biometrics/Operational errors.
    Add-Finding -Severity 'INFO' -Area 'Hello' -Item 'WbioSrvc' `
        -Detail "Stopped (StartType $($wbio.StartType)). Not diagnostic - it is trigger-started by LogonUI on demand and idle-stops after."
}
# Recent sensor errors ARE diagnostic, unlike service state. 1609 is the
# secure-component connection failure that blocks biometric enrolment and
# sign-in; a burst of them is worth surfacing, a stale pair from days ago is not.
$since = (Get-Date).AddDays(-2)
$bioErr = @()
try {
    $bioErr = @(Get-WinEvent -FilterHashtable @{
            LogName = 'Microsoft-Windows-Biometrics/Operational'; Id = 1609; StartTime = $since
        } -ErrorAction Stop)
} catch {
    # No matching events is thrown, not returned empty - that is the healthy case.
}
if ($bioErr.Count -gt 0) {
    Add-Finding -Severity 'WARN' -Area 'Hello' -Item 'Biometric sensor errors' `
        -Detail ("$($bioErr.Count) secure-connection failure(s) (event 1609) in the last 2 days, newest $($bioErr[0].TimeCreated).") `
        -Remedy 'The sensor could not establish its secure channel. Re-seat/re-enumerate it, and check the sensor driver version.'
}

# ------------------------------------------------------ NVIDIA driver parity ---
$nv = @($devices | Where-Object { $_.Class -eq 'Display' -and $_.Present -and $_.FriendlyName -match 'NVIDIA' })
$nvInfo = foreach ($g in $nv) {
    $ver = $null; $inf = $null; $hw = $null
    try { $ver = (Get-PnpDeviceProperty -InstanceId $g.InstanceId -KeyName 'DEVPKEY_Device_DriverVersion' -ErrorAction Stop).Data } catch { }
    try { $inf = (Get-PnpDeviceProperty -InstanceId $g.InstanceId -KeyName 'DEVPKEY_Device_DriverInfPath' -ErrorAction Stop).Data } catch { }
    try { $hw = @((Get-PnpDeviceProperty -InstanceId $g.InstanceId -KeyName 'DEVPKEY_Device_HardwareIds' -ErrorAction Stop).Data)[0] } catch { }
    $devId = if ($hw -match 'DEV_([0-9A-Fa-f]{4})') { "DEV_$($Matches[1].ToUpper())" } else { $null }
    [pscustomobject]@{ Name = $g.FriendlyName; Status = $g.Status; Driver = $ver; Inf = $inf; DeviceId = $devId }
}
$nvInfo = @($nvInfo)
# A GPU whose DriverVersion could not be read must NOT be silently dropped from
# the parity comparison - doing so can hide the very version split this check
# exists to find (two GPUs, one unreadable, "all on one version").
$unreadable = @($nvInfo | Where-Object { -not $_.Driver })
if ($unreadable.Count -gt 0) {
    Add-Finding -Severity 'WARN' -Area 'GPU' -Item 'Driver version unreadable' `
        -Detail ("Could not read DriverVersion for: " + (($unreadable | ForEach-Object { $_.Name }) -join '; ') +
                 ". Version parity below is computed from the remaining GPUs only and may be incomplete.") `
        -Remedy 'Usually means the device is in a fault state. Resolve its device error first, then re-run.'
}
$distinct = @($nvInfo | Where-Object { $_.Driver } | Select-Object -ExpandProperty Driver -Unique)

if ($distinct.Count -gt 1) {
    # Mixed versions is the symptom. The question that decides the remedy is
    # whether any ONE installed package covers every NVIDIA device present.
    # NVIDIA splits GeForce (consumer) from RTX/Quadro (professional) into
    # separate driver branches with DISJOINT device lists, while nvlddmkm.sys is
    # a single shared kernel driver - so on a machine mixing the two, no
    # reinstall of an existing package can make both work. Saying "just
    # reinstall one driver" there sends the operator after something impossible.
    $infDirs = @(Get-ChildItem (Join-Path $env:SystemRoot 'System32\DriverStore\FileRepository') -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^nv.*\.inf_' })
    $wantIds = @($nvInfo | Where-Object { $_.DeviceId } | Select-Object -ExpandProperty DeviceId -Unique)
    $covering = foreach ($dir in $infDirs) {
        $infFile = @(Get-ChildItem $dir.FullName -Filter '*.inf' -File -ErrorAction SilentlyContinue)[0]
        if (-not $infFile) { continue }
        $text = Get-Content -LiteralPath $infFile.FullName -Raw -ErrorAction SilentlyContinue
        if (-not $text) { continue }
        $missing = @($wantIds | Where-Object { $text -notmatch [regex]::Escape($_) })
        if ($missing.Count -eq 0) { $infFile.Name }
    }
    $covering = @($covering)

    # The fault is the VERSION SPLIT, not the INF split. nvlddmkm.sys is a single
    # shared kernel driver, so two packages from different RELEASES cannot both
    # load and whichever loses reports problem 31. Different INFs at the SAME
    # version are completely normal - one NVIDIA release ships ~45 OEM-specific
    # INFs (nv_dispsi, nvltsi, nvmisi ...) and no single one of them lists every
    # device. Verified on this machine 2026-09-09: an RTX 2000 Ada (nvltsi) and a
    # GeForce RTX 5060 Ti (nv_dispsi) were served by DIFFERENT INFs from the SAME
    # 610.88 package and both reported OK once the version split was removed.
    $coverNote = if ($covering.Count -gt 0) {
        "One installed package covers every device ($($covering -join ', '))."
    } else {
        "No single INF lists every device ($($wantIds -join ', ')) - which is normal; NVIDIA splits device coverage across many OEM-specific INFs within one release."
    }
    Add-Finding -Severity 'ERROR' -Area 'GPU' -Item 'NVIDIA driver version split' `
        -Detail ("$($nvInfo.Count) NVIDIA GPUs are on DIFFERENT driver versions: " + ($distinct -join ', ') + ". $coverNote") `
        -Remedy ('Install ONE NVIDIA release that supports every GPU present, so all of them share a single nvlddmkm.sys. ' +
                 'Do not chase a single INF containing every device - that is not how NVIDIA packages are laid out. ' +
                 'After installing, an externally-attached GPU may need its enclosure re-enumerated (or a reboot) to reallocate PCIe resources; ' +
                 'problem 12 immediately after a driver swap means resource allocation, not an unsupported device.')
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
