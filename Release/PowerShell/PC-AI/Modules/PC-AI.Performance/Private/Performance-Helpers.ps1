#Requires -Version 5.1
<#
.SYNOPSIS
    Internal helper functions for PC-AI.Performance module.

.DESCRIPTION
    Contains private utility functions used by public module functions.
    These functions are not exported from the module.
#>

function Get-DriveMediaType {
    <#
    .SYNOPSIS
        Determines if a drive is SSD or HDD.

    .DESCRIPTION
        Uses multiple methods to detect drive type for optimal operation selection.

    .PARAMETER DriveLetter
        The drive letter to check (e.g., 'C').

    .OUTPUTS
        System.String - Returns 'SSD', 'HDD', or 'Unknown'
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true)]
        [ValidatePattern('^[A-Za-z]$')]
        [string]$DriveLetter
    )

    try {
        # Try using Get-PhysicalDisk (Windows 8+)
        $partition = Get-Partition -DriveLetter $DriveLetter -ErrorAction SilentlyContinue
        if ($partition) {
            $disk = Get-Disk -Number $partition.DiskNumber -ErrorAction SilentlyContinue
            if ($disk) {
                $physicalDisk = Get-PhysicalDisk | Where-Object { $_.DeviceId -eq $disk.Number } | Select-Object -First 1
                if ($physicalDisk) {
                    switch ($physicalDisk.MediaType) {
                        'SSD' { return 'SSD' }
                        'HDD' { return 'HDD' }
                        'Unspecified' {
                            # Check if it's NVMe (typically SSD)
                            if ($physicalDisk.BusType -eq 'NVMe') {
                                return 'SSD'
                            }
                        }
                    }
                }
            }
        }

        # Fallback: Check using WMI MSFT_PhysicalDisk
        $wmiDisk = Get-CimInstance -Namespace 'Root\Microsoft\Windows\Storage' -ClassName 'MSFT_PhysicalDisk' -ErrorAction SilentlyContinue |
            Where-Object { $_.FriendlyName -and $_.MediaType }

        if ($wmiDisk) {
            # MediaType: 0 = Unspecified, 3 = HDD, 4 = SSD
            $ssdDisks = $wmiDisk | Where-Object { $_.MediaType -eq 4 }
            if ($ssdDisks) {
                return 'SSD'
            }
        }

        # Final fallback: Check for SSD indicators in model name
        $diskDrive = Get-CimInstance -ClassName Win32_DiskDrive -ErrorAction SilentlyContinue |
            Where-Object { $_.DeviceID }

        foreach ($drive in $diskDrive) {
            if ($drive.Model -match 'SSD|NVMe|Solid State|Flash') {
                return 'SSD'
            }
        }

        return 'Unknown'
    }
    catch {
        Write-Warning "Could not determine drive type for $DriveLetter`: $_"
        return 'Unknown'
    }
}

function Test-IsAdmin {
    <#
    .SYNOPSIS
        Tests if the current session has administrator privileges.

    .OUTPUTS
        System.Boolean
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Format-ByteSize {
    <#
    .SYNOPSIS
        Formats a byte count into a human-readable string.

    .PARAMETER Bytes
        The number of bytes to format.

    .PARAMETER Precision
        Number of decimal places (default: 2).

    .OUTPUTS
        System.String
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true)]
        [long]$Bytes,

        [Parameter()]
        [int]$Precision = 2
    )

    $sizes = @('B', 'KB', 'MB', 'GB', 'TB', 'PB')
    $order = 0

    while ($Bytes -ge 1024 -and $order -lt $sizes.Count - 1) {
        $order++
        $Bytes = $Bytes / 1024
    }

    return "{0:N$Precision} {1}" -f $Bytes, $sizes[$order]
}

function Get-PercentageColor {
    <#
    .SYNOPSIS
        Returns a console color based on a percentage value.

    .DESCRIPTION
        Used for color-coding resource usage in console output.
        Green = Good (low usage), Yellow = Warning, Red = Critical (high usage)

    .PARAMETER Percentage
        The percentage value (0-100).

    .PARAMETER InvertScale
        If true, higher values are better (e.g., free space).

    .OUTPUTS
        System.ConsoleColor
    #>
    [CmdletBinding()]
    [OutputType([System.ConsoleColor])]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateRange(0, 100)]
        [double]$Percentage,

        [Parameter()]
        [switch]$InvertScale
    )

    if ($InvertScale) {
        $Percentage = 100 - $Percentage
    }

    if ($Percentage -ge 90) {
        return [System.ConsoleColor]::Red
    }
    elseif ($Percentage -ge 75) {
        return [System.ConsoleColor]::Yellow
    }
    else {
        return [System.ConsoleColor]::Green
    }
}

function Get-ProcessOwner {
    <#
    .SYNOPSIS
        Gets the owner of a process.

    .PARAMETER ProcessId
        The process ID.

    .OUTPUTS
        System.String - The process owner or 'N/A' if unavailable.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true)]
        [int]$ProcessId
    )

    try {
        $process = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
        if ($process) {
            $owner = Invoke-CimMethod -InputObject $process -MethodName GetOwner -ErrorAction SilentlyContinue
            if ($owner -and $owner.User) {
                if ($owner.Domain) {
                    return "$($owner.Domain)\$($owner.User)"
                }
                return $owner.User
            }
        }
    }
    catch {
        # Silently fail for processes we can't query
    }

    return 'N/A'
}

function Write-ColoredOutput {
    <#
    .SYNOPSIS
        Writes colored output to the console.

    .PARAMETER Message
        The message to write.

    .PARAMETER ForegroundColor
        The text color.

    .PARAMETER NoNewline
        If specified, doesn't add a newline at the end.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,

        [Parameter()]
        [System.ConsoleColor]$ForegroundColor = [System.ConsoleColor]::White,

        [Parameter()]
        [switch]$NoNewline
    )

    $params = @{
        Object = $Message
        ForegroundColor = $ForegroundColor
    }

    if ($NoNewline) {
        $params['NoNewline'] = $true
    }

    Write-Host @params
}

function Get-DiskIOCounters {
    <#
    .SYNOPSIS
        Gets current disk I/O counters.

    .OUTPUTS
        PSCustomObject with read/write bytes and operations.
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param()

    try {
        $counters = Get-Counter -Counter @(
            '\PhysicalDisk(_Total)\Disk Read Bytes/sec',
            '\PhysicalDisk(_Total)\Disk Write Bytes/sec',
            '\PhysicalDisk(_Total)\Disk Reads/sec',
            '\PhysicalDisk(_Total)\Disk Writes/sec'
        ) -ErrorAction SilentlyContinue

        if ($counters) {
            $samples = $counters.CounterSamples
            return [PSCustomObject]@{
                ReadBytesPerSec  = [long]($samples | Where-Object { $_.Path -like '*Read Bytes*' }).CookedValue
                WriteBytesPerSec = [long]($samples | Where-Object { $_.Path -like '*Write Bytes*' }).CookedValue
                ReadsPerSec      = [double]($samples | Where-Object { $_.Path -like '*Reads/sec*' }).CookedValue
                WritesPerSec     = [double]($samples | Where-Object { $_.Path -like '*Writes/sec*' }).CookedValue
            }
        }
    }
    catch {
        Write-Warning "Could not get disk I/O counters: $_"
    }

    return [PSCustomObject]@{
        ReadBytesPerSec  = 0
        WriteBytesPerSec = 0
        ReadsPerSec      = 0
        WritesPerSec     = 0
    }
}
