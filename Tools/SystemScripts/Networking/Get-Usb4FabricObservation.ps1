#Requires -Version 7.0
<#
.SYNOPSIS
Collect present Windows USB4 endpoints for the native read-only topology planner.
.DESCRIPTION
Captures interface GUID, MAC, present PnP ancestry, addresses and neighbor-cache
observations. No packet probes, addresses, routes, firewall, shares or files change.
A hidden child runs under the existing bounded process runner. Missing providers
fail explicitly. Merge host observations only over an independently trusted
management channel; source labels and neighbor caches are not authentication.
.PARAMETER DryRun
Return the collection plan without starting providers or writing files.
.EXAMPLE
./Get-Usb4FabricObservation.ps1 -DryRun
.EXAMPLE
$observation = ./Get-Usb4FabricObservation.ps1
# Pipe the observation as JSON to an explicitly chosen private evidence file.
# See docs/usb4/fabric-topology.md for the native topology-plan contract.
#>
[CmdletBinding()]
param(
    [ValidateRange(30, 300)][int]$TtlSeconds = 90,
    [ValidateSet('local', 'management-ssh')][string]$Source = 'local',
    [ValidateRange(1, 30)][int]$TimeoutSeconds = 10,
    [switch]$DryRun,
    [Alias('h', 'help')][switch]$ShowHelp,
    [Parameter(DontShow)][switch]$Collect
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($ShowHelp) { Get-Help $PSCommandPath -Detailed; return }
if ($DryRun) {
    [pscustomobject]@{ Schema = 'pcai.usb4-fabric-collection-plan/v1'; ReadOnly = $true;
        TimeoutSeconds = $TimeoutSeconds; TtlSeconds = $TtlSeconds; Source = $Source;
        ChangesNetworking = $false; RequiresThunderboltShare = $false }
    return
}
if (-not $IsWindows) { throw 'The USB4 Windows observation provider is unsupported on this platform.' }

if (-not $Collect) {
    . (Join-Path $PSScriptRoot 'Invoke-VigilBoundedProcess.ps1')
    $result = Invoke-VigilBoundedProcess -FilePath (Join-Path $PSHOME 'pwsh.exe') `
        -Arguments @('-NoLogo', '-NoProfile', '-NonInteractive', '-File', $PSCommandPath,
            '-Collect', '-TtlSeconds', [string]$TtlSeconds, '-Source', $Source) `
        -TimeoutSeconds $TimeoutSeconds
    if ($result.ExitCode -ne 0) { throw "USB4 collection failed ($($result.ExitCode)): $($result.Stderr)" }
    if ([Text.Encoding]::UTF8.GetByteCount($result.Stdout) -gt 262144) { throw 'USB4 collection exceeds its output contract.' }
    $value = $result.Stdout | ConvertFrom-Json
    if ($value.schema -cne 'pcai.usb4-fabric-observation/v1' -or @($value.hosts).Count -ne 1) {
        throw 'USB4 provider returned an invalid observation envelope.'
    }
    $value
    return
}

function Get-Usb4FabricHost {
    <# .SYNOPSIS
    Read one host's present USB4 identities without probing or changing networking.
    #>
    param([ValidateSet('local', 'management-ssh')][string]$ObservationSource = 'local')
$interfaces = @()
$candidates = @(Get-NetAdapter -IncludeHidden -ErrorAction Stop | Where-Object {
    ([string]$_.PnPDeviceID).StartsWith('USB4\', [StringComparison]::OrdinalIgnoreCase) -and
    ([string]$_.PnPDeviceID).IndexOf('PROT_USB4NET', [StringComparison]::OrdinalIgnoreCase) -ge 0
})
if ($candidates.Count -gt 64) { throw 'USB4 inventory exceeds the 64-interface bound.' }
$presentIds = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
if ($candidates.Count -gt 0) {
    # Provider failure must not masquerade as a successful empty/undocked report.
    $presentDevices = @(Get-PnpDevice -PresentOnly -ErrorAction Stop)
    if ($presentDevices.Count -gt 16384) { throw 'Present-device inventory exceeds its bound.' }
    foreach ($device in $presentDevices) { [void]$presentIds.Add([string]$device.InstanceId) }
}
foreach ($adapter in $candidates) {
    $pnp = [string]$adapter.PnPDeviceID
    if (-not $presentIds.Contains($pnp)) { continue }
    $ancestry = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $cursor = $pnp; $root = $null
    for ($depth = 0; $cursor -and $depth -lt 16; $depth++) {
        if (-not $ancestry.Add($cursor)) { throw 'USB4 PnP ancestry contains a cycle.' }
        if ($cursor.StartsWith('USB4\ROOT_DEVICE_ROUTER', [StringComparison]::OrdinalIgnoreCase)) {
            if ($presentIds.Contains($cursor)) { $root = $cursor }
            break
        }
        $cursor = [string](Get-PnpDeviceProperty -InstanceId $cursor -KeyName DEVPKEY_Device_Parent -ErrorAction Stop).Data
    }
    $index = [int]$adapter.ifIndex
    $addresses = @(Get-NetIPAddress -InterfaceIndex $index -PolicyStore ActiveStore -ErrorAction Stop |
        Where-Object { [string]$_.AddressState -eq 'Preferred' } | ForEach-Object { [string]$_.IPAddress })
    try { $neighbors = @(Get-NetNeighbor -InterfaceIndex $index -ErrorAction Stop) }
    catch {
        if ($_.FullyQualifiedErrorId -cne 'CmdletizationQuery_NotFound,Get-NetNeighbor' -or
            $_.CategoryInfo.Category -ne [Management.Automation.ErrorCategory]::ObjectNotFound) { throw }
        $neighbors = @()
    }
    if ($neighbors.Count -gt 64 -or $addresses.Count -gt 64) { throw 'USB4 address/neighbor inventory exceeds its bound.' }
    $interfaces += @([ordered]@{
        interface_guid = ([guid]$adapter.InterfaceGuid).ToString()
        mac = ([string]$adapter.MacAddress).ToUpperInvariant()
        pnp_instance_id = $pnp; root_router_id = $root
        interface_index = $index; interface_alias = [string]$adapter.Name
        up = ($adapter.InterfaceOperationalStatus -eq 1 -and $adapter.AdminStatus -eq 1)
        addresses = $addresses
        neighbors = @($neighbors | Where-Object {
            $macText = ([string]$_.LinkLayerAddress).Replace('-', '').Replace(':', '')
            [string]$_.State -in @('Reachable', 'Stale', 'Delay', 'Probe', 'Permanent') -and
            $macText -match '^[0-9a-fA-F]{12}$' -and $macText -ne '000000000000' -and
            (([Convert]::ToInt32($macText.Substring(0, 2), 16) -band 1) -eq 0)
        } | ForEach-Object { [ordered]@{ address = [string]$_.IPAddress; mac = [string]$_.LinkLayerAddress } })
    })
}
    [ordered]@{ observed_at_utc = [DateTimeOffset]::UtcNow.ToString('o');
        host = [Environment]::MachineName.ToLowerInvariant(); source = $ObservationSource;
        interfaces = $interfaces; reference_applications = @() }
}
[ordered]@{ schema = 'pcai.usb4-fabric-observation/v1'; ttl_seconds = $TtlSeconds;
    hosts = @((Get-Usb4FabricHost -ObservationSource $Source)) } | ConvertTo-Json -Depth 12 -Compress
