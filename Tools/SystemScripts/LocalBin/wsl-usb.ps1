#!/usr/bin/env pwsh
#Requires -Version 7.0

<#
.SYNOPSIS
    WSL USB Device Manager - Simplified wrapper for usbipd-win

.DESCRIPTION
    Manages USB devices between Windows and WSL distributions with automatic
    elevation, error handling, and a user-friendly interface.

.PARAMETER Action
    Action to perform: list, attach, detach, bind, unbind, auto-attach, status

.PARAMETER BusId
    USB device Bus ID (e.g., 1-3, 18-4)

.PARAMETER Distribution
    WSL distribution name (default: Ubuntu)

.PARAMETER Force
    Force operations (bypass safety checks)

.PARAMETER Filter
    Filter string for listing devices (e.g., "FTDI", "0403:")

.EXAMPLE
    wsl-usb.ps1 list
    Lists all USB devices

.EXAMPLE
    wsl-usb.ps1 list -Filter "FTDI"
    Lists only FTDI devices

.EXAMPLE
    wsl-usb.ps1 attach 18-4
    Attaches device 18-4 to default WSL distribution (Ubuntu)

.EXAMPLE
    wsl-usb.ps1 attach 18-4 -Distribution Ubuntu -Auto
    Attaches device with auto-reattach on hotplug

.EXAMPLE
    wsl-usb.ps1 detach 18-4
    Detaches device from WSL

.EXAMPLE
    wsl-usb.ps1 status
    Shows current WSL USB device status
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0, Mandatory = $false)]
    [ValidateSet('list', 'attach', 'detach', 'bind', 'unbind', 'auto-attach', 'status', 'help')]
    [string]$Action = 'list',

    [Parameter(Position = 1, Mandatory = $false)]
    [string]$BusId,

    [Parameter(Mandatory = $false)]
    [string]$Distribution = 'Ubuntu',

    [Parameter(Mandatory = $false)]
    [switch]$Force,

    [Parameter(Mandatory = $false)]
    [string]$Filter,

    [Parameter(Mandatory = $false)]
    [switch]$Auto,

    [Parameter(Mandatory = $false)]
    [Alias('h')]
    [switch]$Help
)

# Color output helpers
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = 'White'
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success { param([string]$Message) Write-ColorOutput "✓ $Message" 'Green' }
function Write-Error { param([string]$Message) Write-ColorOutput "✗ $Message" 'Red' }
function Write-Warning { param([string]$Message) Write-ColorOutput "⚠ $Message" 'Yellow' }
function Write-Info { param([string]$Message) Write-ColorOutput "ℹ $Message" 'Cyan' }

# Check if running as administrator
function Test-IsAdmin {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Auto-elevate if needed
function Invoke-ElevatedCommand {
    param([string]$Command)

    if (Test-IsAdmin) {
        return $true
    }

    Write-Warning "This operation requires administrator privileges."
    $response = Read-Host "Elevate to administrator? (Y/n)"

    if ($response -match '^[Yy]?$') {
        $scriptPath = $PSCommandPath
        $arguments = $PSBoundParameters.GetEnumerator() | ForEach-Object {
            "-$($_.Key) $($_.Value)"
        }

        Start-Process pwsh -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" $($arguments -join ' ')"
        exit
    }

    return $false
}

# Fallback: Use raw USB device enumeration via WMI
function Get-UsbDevicesFallback {
    Write-Info "Using fallback USB enumeration (WMI)..."

    try {
        $devices = Get-CimInstance -ClassName Win32_PnPEntity | Where-Object {
            $_.Service -match 'usbser|ftdibus|usbccgp' -or
            $_.DeviceID -match '^USB\\VID_'
        } | Select-Object @{
            Name = 'DeviceID'; Expression = { $_.DeviceID }
        }, @{
            Name = 'Description'; Expression = { $_.Caption }
        }, @{
            Name = 'Status'; Expression = { $_.Status }
        }, @{
            Name = 'VID_PID'; Expression = {
                if ($_.DeviceID -match 'VID_([0-9A-F]{4}).*PID_([0-9A-F]{4})') {
                    "$($matches[1]):$($matches[2])"
                }
            }
        }

        return $devices
    }
    catch {
        Write-Error "Failed to enumerate USB devices: $_"
        return $null
    }
}

# Get USB device list with error handling
function Get-UsbDevices {
    param([string]$FilterString)

    try {
        # Try usbipd first
        $output = & usbipd list 2>&1

        # Check if command failed
        if ($LASTEXITCODE -ne 0 -or $output -match 'Unhandled exception|ERROR_PATH_NOT_FOUND') {
            Write-Warning "usbipd list failed with driver enumeration error."
            Write-Info "This is a known issue with usbipd-win 5.3 when certain drivers are installed."
            Write-Info ""

            # Offer fallback
            $devices = Get-UsbDevicesFallback

            if ($devices) {
                Write-Info ""
                Write-ColorOutput "USB Devices (via WMI fallback):" 'Yellow'
                Write-ColorOutput ("=" * 80) 'DarkGray'

                $devices | Format-Table -AutoSize | Out-String | Write-Host

                Write-Info ""
                Write-Info "To fix usbipd, try:"
                Write-Info "  1. Reboot Windows"
                Write-Info "  2. Uninstall VirtualBox USB drivers if present"
                Write-Info "  3. Run: usbipd list (as Admin)"
            }

            return $null
        }

        # Parse successful output
        $lines = $output -split "`n"

        if ($FilterString) {
            $lines = $lines | Where-Object { $_ -match $FilterString }
        }

        return $lines -join "`n"
    }
    catch {
        Write-Error "Failed to list USB devices: $_"
        return $null
    }
}

# Attach device to WSL
function Attach-UsbDevice {
    param(
        [string]$BusId,
        [string]$Distro,
        [bool]$AutoAttach,
        [bool]$ForceAttach
    )

    if (-not (Test-IsAdmin)) {
        if (-not (Invoke-ElevatedCommand)) {
            Write-Error "Administrator privileges required for attach operation."
            return $false
        }
    }

    Write-Info "Attaching device $BusId to WSL distribution '$Distro'..."

    try {
        $args = @('attach', '--wsl', $Distro, '--busid', $BusId)

        if ($AutoAttach) {
            $args += '--auto-attach'
        }

        $output = & usbipd @args 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Device $BusId attached to $Distro"

            if ($AutoAttach) {
                Write-Info "Auto-attach enabled for this device"
            }

            Write-Info ""
            Write-Info "Verify in WSL with:"
            Write-Info "  wsl -d $Distro -- lsusb"
            Write-Info "  wsl -d $Distro -- ls -l /dev/ttyUSB*"

            return $true
        }
        else {
            Write-Error "Failed to attach device: $output"
            return $false
        }
    }
    catch {
        Write-Error "Attach operation failed: $_"
        return $false
    }
}

# Detach device from WSL
function Detach-UsbDevice {
    param(
        [string]$BusId,
        [bool]$Unbind
    )

    if (-not (Test-IsAdmin)) {
        if (-not (Invoke-ElevatedCommand)) {
            Write-Error "Administrator privileges required for detach operation."
            return $false
        }
    }

    Write-Info "Detaching device $BusId from WSL..."

    try {
        $output = & usbipd detach --busid $BusId 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Device $BusId detached from WSL"

            if ($Unbind) {
                Write-Info "Unbinding device to restore Windows driver..."
                $output = & usbipd unbind --busid $BusId 2>&1

                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Device unbound - Windows driver restored"
                }
                else {
                    Write-Warning "Unbind failed: $output"
                    Write-Info "Device may need to be unplugged/replugged to restore COM port"
                }
            }

            return $true
        }
        else {
            Write-Error "Failed to detach device: $output"
            return $false
        }
    }
    catch {
        Write-Error "Detach operation failed: $_"
        return $false
    }
}

# Bind device for sharing
function Bind-UsbDevice {
    param(
        [string]$BusId,
        [bool]$ForceBinding
    )

    if (-not (Test-IsAdmin)) {
        if (-not (Invoke-ElevatedCommand)) {
            Write-Error "Administrator privileges required for bind operation."
            return $false
        }
    }

    Write-Info "Binding device $BusId for sharing..."

    try {
        $args = @('bind', '--busid', $BusId)

        if ($ForceBinding) {
            $args += '--force'
        }

        $output = & usbipd @args 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Device $BusId bound for sharing"
            Write-Info "Device can now be attached to WSL with: wsl-usb attach $BusId"
            return $true
        }
        else {
            Write-Error "Failed to bind device: $output"

            if ($output -match 'incompatible.*filter' -and -not $ForceBinding) {
                Write-Warning "Incompatible USB filter detected."
                $retry = Read-Host "Retry with --force? (Y/n)"

                if ($retry -match '^[Yy]?$') {
                    return Bind-UsbDevice -BusId $BusId -ForceBinding $true
                }
            }

            return $false
        }
    }
    catch {
        Write-Error "Bind operation failed: $_"
        return $false
    }
}

# Show WSL USB status
function Show-WslUsbStatus {
    Write-Info "WSL USB Device Status"
    Write-ColorOutput ("=" * 80) 'DarkGray'

    # Check WSL distributions
    Write-Info ""
    Write-ColorOutput "WSL Distributions:" 'Cyan'
    wsl -l -v

    Write-Info ""
    Write-ColorOutput "USB Devices:" 'Cyan'
    $devices = Get-UsbDevices

    if ($devices) {
        Write-Host $devices
    }

    # Check usbipd service
    Write-Info ""
    Write-ColorOutput "usbipd Service:" 'Cyan'
    $service = Get-Service usbipd -ErrorAction SilentlyContinue

    if ($service) {
        Write-Host "Status: $($service.Status)"
        Write-Host "StartType: $($service.StartType)"
    }
    else {
        Write-Warning "usbipd service not found - reinstall usbipd-win"
    }

    # Check Ubuntu usbip client
    Write-Info ""
    Write-ColorOutput "Ubuntu usbip client:" 'Cyan'
    $usbipVersion = wsl -d Ubuntu -- bash -c "usbip version 2>&1" 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Success $usbipVersion
    }
    else {
        Write-Warning "usbip not found in Ubuntu - run: wsl -d Ubuntu -- sudo apt install linux-tools-generic"
    }
}

# Show help for specific command
function Show-CommandHelp {
    param([string]$Command)

    switch ($Command) {
        'list' {
            Write-ColorOutput @"

COMMAND: list

DESCRIPTION:
    List all USB devices available on the system. Uses usbipd when available,
    falls back to WMI enumeration if usbipd fails.

USAGE:
    wsl-usb list [-Filter <pattern>] [-Help]

OPTIONS:
    -Filter <pattern>    Filter results by pattern (case-insensitive)
                         Examples: "FTDI", "0403:", "Serial"
    -Help, -h            Show this help message

EXAMPLES:
    # List all USB devices
    wsl-usb list

    # List only FTDI devices
    wsl-usb list -Filter "FTDI"

    # List devices by VID:PID
    wsl-usb list -Filter "0403:"

    # Show help for list command
    wsl-usb list --help

"@ 'White'
        }

        'bind' {
            Write-ColorOutput @"

COMMAND: bind

DESCRIPTION:
    Bind a USB device for sharing with WSL. This is required before you can
    attach the device to a WSL distribution. Requires administrator privileges.

USAGE:
    wsl-usb bind <busid> [-Force] [-Help]

PARAMETERS:
    <busid>              USB Bus ID (e.g., 18-4, 21-2)
                         Find Bus IDs with: wsl-usb list

OPTIONS:
    -Force               Force binding even with incompatible USB filters
                         (required when VirtualBox/USBPcap detected)
    -Help, -h            Show this help message

EXAMPLES:
    # Bind device 18-4 for sharing
    wsl-usb bind 18-4

    # Force bind (bypass filter warnings)
    wsl-usb bind 18-4 -Force

    # Show help
    wsl-usb bind --help

NOTES:
    - Only needs to be done once per device
    - Device remains available to Windows until attached to WSL
    - Use 'unbind' to restore original Windows driver state

"@ 'White'
        }

        'attach' {
            Write-ColorOutput @"

COMMAND: attach

DESCRIPTION:
    Attach a USB device to a WSL distribution. The device must be bound first
    using 'wsl-usb bind'. Once attached, the device is no longer accessible
    from Windows. Requires administrator privileges.

USAGE:
    wsl-usb attach <busid> [-Distribution <name>] [-Auto] [-Help]

PARAMETERS:
    <busid>              USB Bus ID (e.g., 18-4, 21-2)

OPTIONS:
    -Distribution <name> Target WSL distribution (default: Ubuntu)
    -Auto                Enable auto-attach on hotplug/distro start
    -Force               Force attach operation
    -Help, -h            Show this help message

EXAMPLES:
    # Attach device to default distribution (Ubuntu)
    wsl-usb attach 18-4

    # Attach to specific distribution
    wsl-usb attach 18-4 -Distribution Alpine

    # Attach with auto-reattach on hotplug
    wsl-usb attach 18-4 -Auto

    # Show help
    wsl-usb attach --help

VERIFICATION:
    After attaching, verify in WSL with:
        wsl -d Ubuntu -- lsusb
        wsl -d Ubuntu -- ls -l /dev/ttyUSB*

NOTES:
    - Device must be bound first: wsl-usb bind <busid>
    - Close Windows apps using the device before attaching
    - Device becomes unavailable to Windows while attached
    - Use 'detach' to return device to Windows

"@ 'White'
        }

        'detach' {
            Write-ColorOutput @"

COMMAND: detach

DESCRIPTION:
    Detach a USB device from WSL and return it to Windows. The device will
    be available to Windows applications again. Requires administrator privileges.

USAGE:
    wsl-usb detach <busid> [-Force] [-Help]

PARAMETERS:
    <busid>              USB Bus ID (e.g., 18-4, 21-2)

OPTIONS:
    -Force               Also unbind device to restore Windows driver
                         (COM port will reappear immediately)
    -Help, -h            Show this help message

EXAMPLES:
    # Detach device from WSL
    wsl-usb detach 18-4

    # Detach and restore Windows driver
    wsl-usb detach 18-4 -Force

    # Show help
    wsl-usb detach --help

NOTES:
    - After detach, device remains bound (shareable state)
    - Use 'unbind' separately to fully restore Windows driver
    - Or use -Force flag to detach and unbind in one step
    - If COM port doesn't reappear, unplug/replug device

"@ 'White'
        }

        'unbind' {
            Write-ColorOutput @"

COMMAND: unbind

DESCRIPTION:
    Unbind a USB device to restore the original Windows driver. This returns
    the device to normal Windows operation (COM port reappears). The device
    must be detached from WSL first. Requires administrator privileges.

USAGE:
    wsl-usb unbind <busid> [-Help]

PARAMETERS:
    <busid>              USB Bus ID (e.g., 18-4, 21-2)

OPTIONS:
    -Help, -h            Show this help message

EXAMPLES:
    # Unbind device to restore Windows driver
    wsl-usb unbind 18-4

    # Show help
    wsl-usb unbind --help

NOTES:
    - Detach from WSL first if currently attached
    - Restores original Windows driver (COM port reappears)
    - Device can be re-bound later with 'bind' command
    - If issues persist, unplug and replug the device

"@ 'White'
        }

        'status' {
            Write-ColorOutput @"

COMMAND: status

DESCRIPTION:
    Show complete system status including WSL distributions, USB devices,
    usbipd service status, and Ubuntu usbip client status. Useful for
    diagnosing issues.

USAGE:
    wsl-usb status [-Help]

OPTIONS:
    -Help, -h            Show this help message

EXAMPLE:
    wsl-usb status

OUTPUT INCLUDES:
    - WSL distributions and versions
    - USB devices (via usbipd or WMI fallback)
    - usbipd service status
    - Ubuntu usbip client version

"@ 'White'
        }

        default {
            Show-Help
        }
    }
}

# Show help
function Show-Help {
    Write-ColorOutput @"

WSL USB Device Manager - Simplified Interface for usbipd-win

USAGE:
    wsl-usb <action> [busid] [options]
    wsl-usb <action> --help    (show detailed help for a command)

ACTIONS:
    list              List all USB devices
    attach <busid>    Attach device to WSL
    detach <busid>    Detach device from WSL
    bind <busid>      Bind device for sharing (required before first attach)
    unbind <busid>    Unbind device (restore Windows driver)
    auto-attach <id>  Attach device with auto-reattach on hotplug
    status            Show WSL and USB device status
    help              Show this help message

OPTIONS:
    -Distribution <name>  WSL distribution name (default: Ubuntu)
    -Force                Force operation (bypass safety checks)
    -Filter <string>      Filter device list (e.g., "FTDI", "0403:")
    -Auto                 Enable auto-attach for attach operations
    --help, -h            Show help (general or command-specific)

EXAMPLES:
    # List all USB devices
    wsl-usb list

    # List only FTDI devices
    wsl-usb list -Filter "FTDI"

    # Bind and attach device 18-4 to Ubuntu
    wsl-usb bind 18-4
    wsl-usb attach 18-4

    # Attach with auto-reattach on hotplug
    wsl-usb attach 18-4 -Auto

    # Detach and restore Windows driver
    wsl-usb detach 18-4 -Force

    # Show complete status
    wsl-usb status

    # Get detailed help for a command
    wsl-usb bind --help
    wsl-usb attach --help

NOTES:
    - Admin privileges required for bind/attach/detach operations
    - Device must be bound before first attach
    - Close Windows apps using the device before attaching to WSL
    - Use 'wsl-usb status' to diagnose issues
    - Use 'wsl-usb <command> --help' for detailed command help

"@ 'White'
}

# Check for --help flag (before any admin checks)
if ($Help.IsPresent) {
    Show-CommandHelp -Command $Action
    exit 0
}

# Main execution
switch ($Action) {
    'list' {
        $devices = Get-UsbDevices -FilterString $Filter
        if ($devices) {
            Write-Host $devices
        }
    }

    'attach' {
        if (-not $BusId) {
            Write-Error "BusId required for attach operation"
            Write-Info "Usage: wsl-usb attach <busid> [-Distribution <name>] [-Auto]"
            Write-Info "For detailed help: wsl-usb attach --help"
            exit 1
        }

        Attach-UsbDevice -BusId $BusId -Distro $Distribution -AutoAttach $Auto.IsPresent -ForceAttach $Force.IsPresent
    }

    'detach' {
        if (-not $BusId) {
            Write-Error "BusId required for detach operation"
            Write-Info "Usage: wsl-usb detach <busid>"
            Write-Info "For detailed help: wsl-usb detach --help"
            exit 1
        }

        Detach-UsbDevice -BusId $BusId -Unbind $Force.IsPresent
    }

    'bind' {
        if (-not $BusId) {
            Write-Error "BusId required for bind operation"
            Write-Info "Usage: wsl-usb bind <busid> [-Force]"
            Write-Info "For detailed help: wsl-usb bind --help"
            exit 1
        }

        Bind-UsbDevice -BusId $BusId -ForceBinding $Force.IsPresent
    }

    'unbind' {
        if (-not $BusId) {
            Write-Error "BusId required for unbind operation"
            Write-Info "Usage: wsl-usb unbind <busid>"
            Write-Info "For detailed help: wsl-usb unbind --help"
            exit 1
        }

        if (-not (Test-IsAdmin)) {
            if (-not (Invoke-ElevatedCommand)) {
                Write-Error "Administrator privileges required for unbind operation."
                exit 1
            }
        }

        $output = & usbipd unbind --busid $BusId 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Device $BusId unbound"
        }
        else {
            Write-Error "Unbind failed: $output"
        }
    }

    'auto-attach' {
        if (-not $BusId) {
            Write-Error "BusId required for auto-attach operation"
            Write-Info "Usage: wsl-usb auto-attach <busid> [-Distribution <name>]"
            Write-Info "For detailed help: wsl-usb attach --help"
            exit 1
        }

        Attach-UsbDevice -BusId $BusId -Distro $Distribution -AutoAttach $true -ForceAttach $Force.IsPresent
    }

    'status' {
        Show-WslUsbStatus
    }

    'help' {
        Show-Help
    }

    default {
        Write-Error "Unknown action: $Action"
        Show-Help
        exit 1
    }
}
