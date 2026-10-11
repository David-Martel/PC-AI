#Requires -Version 7.0

<#
.SYNOPSIS
Reports observed Windows hardware capabilities separately from fleet expectations.
.DESCRIPTION
Uses bounded, module-qualified CIM queries without profiles, native PC-AI DLLs,
vendor executables or model dependencies. Device presence, current drivers and
workload validation are distinct. Missing providers remain unknown. Device
instance identifiers, serial numbers, GPU UUIDs, MAC addresses and IP addresses
are excluded. Power adapter budgets are user statements, not sensor readings.
.PARAMETER ProfilePath
Optional fleet JSON profile. Defaults to Config/machine-profiles/<hostname>.json
when present. Profiles describe expectations and conservative build policy only.
.PARAMETER QueryTimeoutSeconds
CIM operation timeout per query, between one and ten seconds. Default: three.
.PARAMETER AsJson
Return JSON instead of a PowerShell object.
.PARAMETER OutputPath
Optional absolute JSON output file. Existing files are preserved unless Force is
specified. Reserved Windows device/null paths and stream syntax are rejected.
.PARAMETER Force
Allow replacing the explicitly specified output file.
.PARAMETER DryRun
Return intended operations without hardware queries, reading profiles or writes.
.PARAMETER Help
Display help; aliases -h and --help are supported.
.EXAMPLE
pwsh -NoLogo -NoProfile -File .\Tools\Get-PcaiMachineCapabilities.ps1 -AsJson
.EXAMPLE
pwsh -NoLogo -NoProfile -File .\Tools\Get-PcaiMachineCapabilities.ps1 -DryRun -OutputPath C:\Temp\machine.json
#>
[CmdletBinding()]
param(
    [string]$ProfilePath,
    [ValidateRange(1, 10)] [int]$QueryTimeoutSeconds = 3,
    [switch]$AsJson,
    [string]$OutputPath,
    [switch]$Force,
    [switch]$DryRun,
    [Alias('h', '-help')] [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($Help) { Microsoft.PowerShell.Core\Get-Help -Name $PSCommandPath -Detailed; return }
if ($DryRun) {
    $preview = [pscustomobject]@{
        status = 'DryRun'; outputPath = $OutputPath; profilePath = $ProfilePath
        queryTimeoutSeconds = $QueryTimeoutSeconds; hardwareQueried = $false; writesPerformed = $false
    }
    if ($AsJson) { Microsoft.PowerShell.Utility\ConvertTo-Json -InputObject $preview -Compress }
    else { $preview }
    return
}
if (-not $IsWindows) { throw 'Machine capability discovery requires Windows.' }
$destination = $null
if ($OutputPath) {
    if (-not [System.IO.Path]::IsPathFullyQualified($OutputPath)) { throw 'OutputPath must be absolute.' }
    $destination = [System.IO.Path]::GetFullPath($OutputPath)
    foreach ($part in $destination.Split([char[]]'\/')) {
        $trimmed = $part.TrimEnd('.', ' ')
        if ($trimmed -eq '$null' -or $trimmed -match '^(?i:AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.|$)') {
            throw 'OutputPath contains a reserved Windows path component.'
        }
    }
    if ($destination.Substring(2).Contains(':')) { throw 'OutputPath cannot contain alternate stream syntax.' }
    if ([System.IO.Directory]::Exists($destination)) { throw 'OutputPath must name a JSON file.' }
    if ([System.IO.File]::Exists($destination) -and -not $Force) { throw 'OutputPath already exists. Use -Force to replace it.' }
}

function Get-MachineValue {
    param($Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    if ($Object -is [System.Collections.IDictionary]) {
        if ($Object.Contains($Name)) { return $Object[$Name] }
    }
    elseif ($null -ne $Object.PSObject.Properties[$Name]) { return $Object.$Name }
    return $null
}

function Invoke-MachineCimQuery {
    param([string]$ClassName, [string]$Namespace = 'root/cimv2', [string[]]$Property)
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $query = [ordered]@{
        source = $ClassName; namespace = $Namespace; status = 'Unknown'
        operationTimeoutSeconds = $QueryTimeoutSeconds; durationMilliseconds = $null
        instanceCount = $null; error = $null; data = @()
    }
    try {
        $parameters = @{ ClassName = $ClassName; Namespace = $Namespace; OperationTimeoutSec = $QueryTimeoutSeconds; ErrorAction = 'Stop' }
        if ($Property) { $parameters.Property = $Property }
        $query.data = @(CimCmdlets\Get-CimInstance @parameters)
        $query.instanceCount = $query.data.Count
        $query.status = if ($query.data.Count -gt 0) { 'Supported' } else { 'SupportedNoInstances' }
    }
    catch {
        $statusCode = Get-MachineValue $_.Exception 'StatusCode'
        if ([string]$statusCode -in @('InvalidClass', 'InvalidNamespace', 'NotSupported')) { $query.status = 'Unsupported' }
        $query.error = [ordered]@{
            exceptionType = $_.Exception.GetType().FullName; category = $_.CategoryInfo.Category.ToString()
            statusCode = if ($null -ne $statusCode) { $statusCode.ToString() } else { $null }
            errorId = $_.FullyQualifiedErrorId
        }
    }
    $query.durationMilliseconds = [math]::Round($timer.Elapsed.TotalMilliseconds, 1)
    return $query
}

function Get-MachineHardwareKey {
    param([string]$DeviceId)
    # Only bus/vendor/product fields are emitted. Instance suffixes may be serials.
    $bus = if ($DeviceId -match '^([^\\]+)\\') { $Matches[1] } else { 'Unknown' }
    $vendor = if ($DeviceId -match '(?:VEN|VID)_([0-9A-Fa-f]{4})') { $Matches[1].ToUpperInvariant() } else { $null }
    $product = if ($DeviceId -match '(?:DEV|PID)_([0-9A-Fa-f]{4})') { $Matches[1].ToUpperInvariant() } else { $null }
    return [ordered]@{ bus = $bus; vendorId = $vendor; productId = $product }
}

$startedUtc = [datetimeoffset]::UtcNow
$hostName = [environment]::MachineName.ToLowerInvariant()
$machineProfile = $null
$profileSource = $null
if (-not $ProfilePath) {
    $candidate = [System.IO.Path]::Combine((Microsoft.PowerShell.Management\Split-Path -Parent $PSScriptRoot), 'Config', 'machine-profiles', "$hostName.json")
    if ([System.IO.File]::Exists($candidate)) { $ProfilePath = $candidate }
}
if ($ProfilePath) {
    $profileFile = [System.IO.FileInfo]::new([System.IO.Path]::GetFullPath($ProfilePath))
    if ($profileFile.Length -gt 1048576) { throw 'Machine profile exceeds the 1 MiB limit.' }
    $machineProfile = Microsoft.PowerShell.Utility\ConvertFrom-Json -InputObject ([System.IO.File]::ReadAllText($profileFile.FullName))
    if ((Get-MachineValue $machineProfile 'schemaVersion') -ne 1) { throw 'Unsupported machine profile schemaVersion.' }
    $profileHost = Get-MachineValue $machineProfile 'hostName'
    if ($profileHost -and $profileHost -ne $hostName) { throw 'Machine profile hostName does not match this host.' }
    $profileSource = $profileFile.Name
}

$queries = [System.Collections.Generic.List[object]]::new()
$computer = Invoke-MachineCimQuery Win32_ComputerSystem -Property Name,Manufacturer,Model,TotalPhysicalMemory,NumberOfLogicalProcessors
$processor = Invoke-MachineCimQuery Win32_Processor -Property Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors
$os = Invoke-MachineCimQuery Win32_OperatingSystem -Property Caption,Version,BuildNumber,FreePhysicalMemory
$pnp = Invoke-MachineCimQuery Win32_PnPEntity -Property Name,PNPClass,PNPDeviceID,Present,Status,ConfigManagerErrorCode,Manufacturer
$drivers = Invoke-MachineCimQuery Win32_PnPSignedDriver -Property DeviceID,DriverVersion,DriverProviderName,DriverDate,IsSigned
$video = Invoke-MachineCimQuery Win32_VideoController -Property Name,PNPDeviceID,DriverVersion,Status,ConfigManagerErrorCode
$network = Invoke-MachineCimQuery Win32_NetworkAdapter -Property Name,PNPDeviceID,PhysicalAdapter,NetConnectionStatus,Speed
foreach ($query in @($computer, $processor, $os, $pnp, $drivers, $video, $network)) { $queries.Add($query) }
$providers = [System.Collections.Generic.List[object]]::new()
foreach ($provider in @(
        @{ Class = 'Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter'; Kind = 'ChannelEnergy'; Property = @('Name', 'Energy', 'Time', 'Power') }
        @{ Class = 'Win32_PerfFormattedData_Counters_ProcessorInformation'; Kind = 'CpuPerformance'; Property = @('Name', 'ProcessorFrequency', 'PerformanceLimitFlags') }
        @{ Class = 'Win32_PerfFormattedData_PerfOS_Memory'; Kind = 'MemoryCommit'; Property = @('AvailableBytes', 'CommittedBytes', 'CommitLimit') }
        @{ Class = 'Win32_PerfFormattedData_Counters_ThermalZoneInformation'; Kind = 'AcpiThermalZone'; Property = @('Name', 'HighPrecisionTemperature') }
        @{ Class = 'MSAcpi_ThermalZoneTemperature'; Namespace = 'root/wmi'; Kind = 'AcpiThermalZone'; Property = @('InstanceName', 'CurrentTemperature') }
    )) {
    $namespace = if ($provider.ContainsKey('Namespace')) { $provider.Namespace } else { 'root/cimv2' }
    $probe = Invoke-MachineCimQuery -ClassName $provider.Class -Namespace $namespace -Property $provider.Property
    $queries.Add($probe)
    $providers.Add([ordered]@{
        kind = $provider.Kind; source = $probe.source; status = $probe.status
        instanceCount = $probe.instanceCount; workloadValidation = 'Unknown'; error = $probe.error
        interpretation = 'Provider enumeration only; no proof of valid readings, sensor health or CPU core temperature'
    })
}

$driverById = @{}
foreach ($driver in $drivers.data) {
    $key = [string](Get-MachineValue $driver 'DeviceID')
    if ($key) { $driverById[$key] = $driver }
}
$videoById = @{}
foreach ($controller in $video.data) {
    $key = [string](Get-MachineValue $controller 'PNPDeviceID')
    if ($key) { $videoById[$key] = $controller }
}
$networkById = @{}
foreach ($adapter in $network.data) {
    $key = [string](Get-MachineValue $adapter 'PNPDeviceID')
    if ($key) { $networkById[$key] = $adapter }
}
$inventory = [System.Collections.Generic.List[object]]::new()
$observedDisplayIds = @{}
foreach ($device in $pnp.data) {
    $id = [string](Get-MachineValue $device 'PNPDeviceID')
    $name = [string](Get-MachineValue $device 'Name')
    $class = [string](Get-MachineValue $device 'PNPClass')
    $hardware = Get-MachineHardwareKey $id
    $categories = [System.Collections.Generic.List[string]]::new()
    if ($class -eq 'Display') { $categories.Add('Gpu'); $observedDisplayIds[$id] = $true }
    if ($class -eq 'Net') { $categories.Add('Network') }
    if ($class -in @('Media', 'AudioEndpoint')) { $categories.Add('AudioVideo') }
    if ($class -eq 'Camera' -or $class -eq 'Image') { $categories.Add('AudioVideo') }
    if ($hardware.bus -eq 'USB' -or $class -eq 'USB') { $categories.Add('Usb') }
    if ($name -match 'Thunderbolt|USB4') { $categories.Add('ThunderboltUsb4') }
    if ($hardware.vendorId -eq '8086' -or $name -match 'Intel') { $categories.Add('Intel') }
    if ($hardware.vendorId -eq '10DE' -or $name -match 'NVIDIA') { $categories.Add('Nvidia') }
    if ($categories.Count -eq 0) { continue }
    $present = Get-MachineValue $device 'Present'
    $errorCode = Get-MachineValue $device 'ConfigManagerErrorCode'
    $presence = if ($present -eq $true -and $errorCode -ne 45) { 'Present' }
        elseif ($present -eq $false -or $errorCode -eq 45) { 'Disconnected' } else { 'Unknown' }
    $status = Get-MachineValue $device 'Status'
    $driver = $driverById[$id]
    $driverVersion = Get-MachineValue $driver 'DriverVersion'
    $driverState = if ($presence -eq 'Present' -and $errorCode -eq 0 -and $status -eq 'OK' -and $driverVersion) {
        'CurrentDeviceDriverReported'
    } elseif ($presence -eq 'Disconnected') { 'HistoricalOrDisconnected' } else { 'UnknownOrProblem' }
    $row = [ordered]@{
        name = $name; deviceClass = $class; categories = $categories.ToArray(); hardware = $hardware
        presence = $presence; statusRaw = $status; configManagerErrorCodeRaw = $errorCode
        driver = [ordered]@{
            state = $driverState; version = $driverVersion
            provider = Get-MachineValue $driver 'DriverProviderName'; signed = Get-MachineValue $driver 'IsSigned'
        }
        source = 'Win32_PnPEntity + Win32_PnPSignedDriver'; workloadValidation = 'Unknown'
        videoDriverVersion = $null; videoDriverMatches = $null; network = $null
    }
    if ($class -eq 'Display' -and $videoById.ContainsKey($id)) {
        $row.videoDriverVersion = Get-MachineValue $videoById[$id] 'DriverVersion'
        if ($driverVersion -and $row.videoDriverVersion) { $row.videoDriverMatches = $driverVersion -eq $row.videoDriverVersion }
        if ($row.videoDriverMatches -eq $false) { $row.driver.state = 'DriverVersionMismatch' }
    }
    if ($networkById.ContainsKey($id)) {
        $adapter = $networkById[$id]
        $row.network = [ordered]@{
            physicalAdapter = Get-MachineValue $adapter 'PhysicalAdapter'
            connectionStatusRaw = Get-MachineValue $adapter 'NetConnectionStatus'
            speedBitsPerSecondRaw = Get-MachineValue $adapter 'Speed'
        }
    }
    $inventory.Add($row)
}
foreach ($controller in $video.data) {
    $id = [string](Get-MachineValue $controller 'PNPDeviceID')
    if ($observedDisplayIds.ContainsKey($id)) { continue }
    $inventory.Add([ordered]@{
        name = Get-MachineValue $controller 'Name'; deviceClass = 'Display'; categories = @('Gpu')
        hardware = Get-MachineHardwareKey $id; presence = 'Unknown'
        statusRaw = Get-MachineValue $controller 'Status'; configManagerErrorCodeRaw = Get-MachineValue $controller 'ConfigManagerErrorCode'
        driver = [ordered]@{ state = 'UnknownOrProblem'; version = Get-MachineValue $controller 'DriverVersion'; provider = $null; signed = $null }
        source = 'Win32_VideoController only; PnP presence unverified'; workloadValidation = 'Unknown'
        videoDriverVersion = Get-MachineValue $controller 'DriverVersion'; videoDriverMatches = $null; network = $null
    })
}

$nvidiaCandidates = @($inventory | Microsoft.PowerShell.Core\Where-Object {
    $_.deviceClass -eq 'Display' -and $_.hardware.vendorId -eq '10DE' -and
    $_.presence -eq 'Present' -and $_.driver.state -eq 'CurrentDeviceDriverReported'
})
$intelCandidates = @($inventory | Microsoft.PowerShell.Core\Where-Object {
    $_.deviceClass -eq 'Display' -and $_.hardware.vendorId -eq '8086' -and
    $_.presence -eq 'Present' -and $_.driver.state -eq 'CurrentDeviceDriverReported'
})
$computerRow = if ($computer.data.Count -gt 0) { $computer.data[0] } else { $null }
$osRow = if ($os.data.Count -gt 0) { $os.data[0] } else { $null }
$logical = Get-MachineValue $computerRow 'NumberOfLogicalProcessors'
$freeKiB = Get-MachineValue $osRow 'FreePhysicalMemory'
$profileBuild = Get-MachineValue $machineProfile 'buildPolicy'
$configuredJobs = Get-MachineValue $profileBuild 'maxNativeBuildJobs'
if ($null -eq $configuredJobs) { $configuredJobs = 2 }
if ($configuredJobs -lt 1 -or $configuredJobs -gt 64) { throw 'Profile maxNativeBuildJobs must be between 1 and 64.' }
$recommendedJobs = 1
if ($logical -gt 0 -and $freeKiB -gt 0) {
    $recommendedJobs = [int][math]::Max(1, [math]::Min($configuredJobs,
        [math]::Min([math]::Floor($logical / 4.0), [math]::Floor([double]$freeKiB * 1024 / 4GB))))
}
$queryProvenance = @($queries | Microsoft.PowerShell.Core\ForEach-Object {
    [ordered]@{ source = $_.source; namespace = $_.namespace; status = $_.status
        operationTimeoutSeconds = $_.operationTimeoutSeconds; durationMilliseconds = $_.durationMilliseconds
        instanceCount = $_.instanceCount; error = $_.error }
})
$expectedHardware = @(Get-MachineValue $machineProfile 'expectedHardware' | Microsoft.PowerShell.Core\Where-Object { $null -ne $_ })
$report = [ordered]@{
    schemaVersion = 1; reportType = 'WindowsMachineCapabilities'
    utc = $startedUtc.ToString('o'); hostName = $hostName; platform = 'Windows'
    identity = [ordered]@{
        manufacturer = Get-MachineValue $computerRow 'Manufacturer'; model = Get-MachineValue $computerRow 'Model'
        totalPhysicalBytes = Get-MachineValue $computerRow 'TotalPhysicalMemory'; logicalProcessors = $logical
        processors = @($processor.data | Microsoft.PowerShell.Core\ForEach-Object {
            [ordered]@{ name = Get-MachineValue $_ 'Name'; manufacturer = Get-MachineValue $_ 'Manufacturer'
                cores = Get-MachineValue $_ 'NumberOfCores'; logicalProcessors = Get-MachineValue $_ 'NumberOfLogicalProcessors' }
        })
        os = [ordered]@{ caption = Get-MachineValue $osRow 'Caption'; version = Get-MachineValue $osRow 'Version'; buildNumber = Get-MachineValue $osRow 'BuildNumber' }
        source = 'Live CIM identity; serials and unique device identifiers excluded'
    }
    devices = $inventory.ToArray(); providers = $providers.ToArray(); queryProvenance = $queryProvenance
    gpuCapabilities = [ordered]@{
        nvidiaDriverCandidateCount = $nvidiaCandidates.Count; intelDriverCandidateCount = $intelCandidates.Count
        cudaWorkloadValidation = 'Unknown'; intelComputeWorkloadValidation = 'Unknown'
        nvidiaTelemetry = 'NotProbed'; cpuCoreTemperature = 'Unknown'; gpuCoreTemperature = 'Unknown'
        interpretation = 'Present device plus current driver is a candidate, not validated compute capability; DLL presence is irrelevant'
    }
    expectedHardware = $expectedHardware
    statedPowerPath = Get-MachineValue $machineProfile 'statedPowerPath'
    powerObservation = [ordered]@{ wholePcInputWatts = $null; inputVoltageVolts = $null; usbPdNegotiation = 'Unknown'; source = 'No input-power or voltage sensor queried' }
    buildPolicy = [ordered]@{
        configuredMaxNativeBuildJobs = [int]$configuredJobs; recommendedNativeBuildJobs = $recommendedJobs
        availablePhysicalBytesAtCapture = if ($null -ne $freeKiB) { [long]$freeKiB * 1024 } else { $null }
        rule = 'min(profile cap, floor(logical CPUs/4), floor(available GiB/4)); minimum1; unknown telemetry=>1'
        applied = $false; source = if ($profileSource) { $profileSource } else { 'Conservative default' }
    }
    profileSource = $profileSource
    collectionMilliseconds = [math]::Round(([datetimeoffset]::UtcNow - $startedUtc).TotalMilliseconds, 1)
}
$json = Microsoft.PowerShell.Utility\ConvertTo-Json -InputObject $report -Depth 12
if ($destination) {
    $parent = [System.IO.Path]::GetDirectoryName($destination)
    [System.IO.Directory]::CreateDirectory($parent) | Microsoft.PowerShell.Core\Out-Null
    $mode = if ($Force) { [System.IO.FileMode]::Create } else { [System.IO.FileMode]::CreateNew }
    $stream = [System.IO.FileStream]::new($destination, $mode, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
    try {
        $writer = [System.IO.StreamWriter]::new($stream, [System.Text.UTF8Encoding]::new($false))
        try { $writer.WriteLine($json); $writer.Flush(); $stream.Flush($true) }
        finally { $writer.Dispose() }
    }
    finally { $stream.Dispose() }
}
if ($AsJson) { $json } else { [pscustomobject]$report }
