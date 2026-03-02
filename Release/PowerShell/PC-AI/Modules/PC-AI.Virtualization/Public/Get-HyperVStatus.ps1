#Requires -Version 5.1
<#
.SYNOPSIS
    Gets Hyper-V status and configuration

.DESCRIPTION
    Checks Hyper-V installation status, services, and virtual machine status.

.PARAMETER IncludeVMs
    Include virtual machine list

.EXAMPLE
    Get-HyperVStatus
    Returns Hyper-V status

.OUTPUTS
    PSCustomObject with Hyper-V status information
#>
function Get-HyperVStatus {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [switch]$IncludeVMs
    )

    $result = [PSCustomObject]@{
        Installed       = $false
        Enabled         = $false
        Services        = @()
        VirtualMachines = @()
        VirtualSwitches = @()
        Severity        = 'Unknown'
    }

    try {
        # Check if Hyper-V is installed
        $hyperv = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All -ErrorAction SilentlyContinue
        if ($hyperv) {
            $result.Installed = $true
            $result.Enabled = ($hyperv.State -eq 'Enabled')
        }

        if (-not $result.Enabled) {
            $result.Severity = 'Warning'
            return $result
        }

        # Check Hyper-V services
        $serviceNames = @('vmcompute', 'vmms', 'hvhost')
        foreach ($serviceName in $serviceNames) {
            $service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
            if ($service) {
                $result.Services += [PSCustomObject]@{
                    Name   = $service.Name
                    Status = $service.Status
                    StartType = $service.StartType
                }
            }
        }

        # Check for stopped services
        $stoppedServices = $result.Services | Where-Object { $_.Status -ne 'Running' }
        if ($stoppedServices) {
            $result.Severity = 'Warning'
        }
        else {
            $result.Severity = 'OK'
        }

        # Get virtual switches
        try {
            $result.VirtualSwitches = Get-VMSwitch -ErrorAction SilentlyContinue |
                Select-Object Name, SwitchType, NetAdapterInterfaceDescription
        }
        catch {
            Write-Verbose "Could not get virtual switches: $_"
        }

        # Get VMs if requested
        if ($IncludeVMs) {
            try {
                $result.VirtualMachines = Get-VM -ErrorAction SilentlyContinue |
                    Select-Object Name, State, CPUUsage, MemoryAssigned, Uptime
            }
            catch {
                Write-Verbose "Could not get virtual machines: $_"
            }
        }

        return $result

    }
    catch {
        Write-Error "Failed to get Hyper-V status: $($_.Exception.Message)"
        $result.Severity = 'Error'
        return $result
    }
}
