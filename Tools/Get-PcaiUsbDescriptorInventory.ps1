#Requires -Version 7.0

<#
.SYNOPSIS
Reports USB configuration descriptors and declared bus-current limits.
.DESCRIPTION
Profile-independent Windows USB hub IOCTL queries. No port reset, driver change
or vendor transfer is issued. Declared maximum current is not measured current,
USB-C Power Delivery negotiation, charger capability or PC input power.
Raw device paths and serial strings are excluded. Hub indices are snapshot-local.
.PARAMETER AsJson
Return JSON instead of a PowerShell object.
.PARAMETER OutputPath
Optional absolute JSON file. Existing files require explicit Force.
.PARAMETER Force
Replace the explicitly named output file.
.PARAMETER DryRun
Return a preview without native loading, hardware queries or file writes.
.PARAMETER Help
Display help; supports -h and --help.
.EXAMPLE
pwsh -NoLogo -NoProfile -File .\Tools\Get-PcaiUsbDescriptorInventory.ps1 -AsJson
.EXAMPLE
pwsh -NoLogo -NoProfile -File .\Tools\Get-PcaiUsbDescriptorInventory.ps1 -DryRun -OutputPath C:\Temp\usb.json
#>
[CmdletBinding()]
param(
    [switch]$AsJson,
    [string]$OutputPath,
    [switch]$Force,
    [switch]$DryRun,
    [Alias('h', '-help')] [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Read-PcaiUsbHubs {
    if (-not ('PcAi.Diagnostics.UsbInventory' -as [type])) {
        Microsoft.PowerShell.Utility\Add-Type -Path (Join-Path $PSScriptRoot 'Native/UsbHubInventory.cs') -ErrorAction Stop
    }
    [PcAi.Diagnostics.UsbInventory]::Read()
}

function Invoke-PcaiUsbDescriptorInventory {
    [CmdletBinding()]
    param([switch]$AsJson, [string]$OutputPath, [switch]$Force, [switch]$DryRun)

    if ($DryRun) {
        $preview = [pscustomobject]@{
            status = 'DryRun'; outputPath = $OutputPath
            hardwareQueried = $false; nativeLoaded = $false; writesPerformed = $false
        }
        if ($AsJson) { Microsoft.PowerShell.Utility\ConvertTo-Json -InputObject $preview -Compress }
        else { $preview }
        return
    }
    if (-not $IsWindows) { throw 'USB descriptor inventory requires Windows.' }
    $destination = $null
    if ($OutputPath) {
        if (-not [IO.Path]::IsPathFullyQualified($OutputPath)) { throw 'OutputPath must be absolute.' }
        # Validate the original spelling first: GetFullPath can turn a NUL leaf into a device path.
        foreach ($part in $OutputPath.Split([char[]]'\/')) {
            $trimmed = $part.TrimEnd('.', ' ')
            if ($trimmed -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)') {
                throw 'OutputPath contains a reserved Windows path component.'
            }
        }
        $destination = [IO.Path]::GetFullPath($OutputPath)
        if ($destination -match '^\\\\[?.]\\') { throw 'OutputPath cannot use Windows device namespace syntax.' }
        if ($destination.Substring(2).Contains(':')) { throw 'OutputPath cannot contain alternate stream syntax.' }
        if ([IO.Directory]::Exists($destination)) { throw 'OutputPath must name a JSON file.' }
        if ([IO.Path]::GetExtension($destination) -ne '.json') { throw 'OutputPath must name a JSON file.' }
        if ([IO.File]::Exists($destination) -and -not $Force) { throw 'OutputPath already exists. Use -Force to replace it.' }
    }

    # Fail enumeration/initialization before creating output or reporting an empty inventory.
    $hubs = @(Read-PcaiUsbHubs)
    $ports = @($hubs | ForEach-Object { $_.Ports })
    $errors = @($hubs | Where-Object { $_.Error })
    $portErrors = @($ports | Where-Object { $_.Error -or $_.SpeedQueryError -or $_.ConfigurationQueryError })
    $report = [pscustomobject]@{
        schemaVersion = 1; reportType = 'WindowsUsbDescriptorInventory'
        utc = [DateTime]::UtcNow.ToString('o'); hostName = [Environment]::MachineName
        status = if ($errors.Count -gt 0 -or $portErrors.Count -gt 0) { 'Partial' } else { 'Complete' }
        source = 'Windows USB hub query-only IOCTLs; configuration descriptor index0 and EX_V2 speed flags'
        interpretation = 'Declared configuration maximum only; not measured current, USB-C PD negotiation, charger capacity or PC input power. Remote wake is descriptor capability, not enabled state or a wake test.'
        measuredBusCurrentMa = $null; pcInputWatts = $null; usbPdNegotiation = 'Unknown'
        hubCount = $hubs.Count; portCount = $ports.Count
        connectedPortCount = @($ports | Where-Object { $_.ConnectionStatus -eq 1 }).Count
        hubs = $hubs
    }
    $json = Microsoft.PowerShell.Utility\ConvertTo-Json -InputObject $report -Depth 10
    if ($destination) {
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination)) | Out-Null
        $mode = if ($Force) { [IO.FileMode]::Create } else { [IO.FileMode]::CreateNew }
        $stream = [IO.FileStream]::new($destination, $mode, [IO.FileAccess]::Write, [IO.FileShare]::Read)
        try {
            $writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false))
            try { $writer.WriteLine($json); $writer.Flush(); $stream.Flush($true) }
            finally { $writer.Dispose() }
        }
        finally { $stream.Dispose() }
    }
    if ($AsJson) { $json } else { $report }
}

if ($Help) { Microsoft.PowerShell.Core\Get-Help -Name $PSCommandPath -Detailed; return }
Invoke-PcaiUsbDescriptorInventory -AsJson:$AsJson -OutputPath $OutputPath -Force:$Force -DryRun:$DryRun
