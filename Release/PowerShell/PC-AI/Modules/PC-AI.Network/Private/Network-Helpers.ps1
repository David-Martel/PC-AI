#Requires -Version 5.1
<#
.SYNOPSIS
    Internal helper functions for PC-AI.Network module

.DESCRIPTION
    Provides network utility functions used by public module functions including:
    - Adapter status formatting
    - Network metric calculations
    - Registry value helpers
    - Connectivity test utilities
#>

function Get-AdapterStatusDescription {
    <#
    .SYNOPSIS
        Converts adapter operational status code to description
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [int]$StatusCode
    )

    $statusMap = @{
        1 = 'Up'
        2 = 'Down'
        3 = 'Testing'
        4 = 'Unknown'
        5 = 'Dormant'
        6 = 'NotPresent'
        7 = 'LowerLayerDown'
    }

    if ($statusMap.ContainsKey($StatusCode)) {
        return $statusMap[$StatusCode]
    }
    return "Unknown ($StatusCode)"
}

function Get-NetworkSeverity {
    <#
    .SYNOPSIS
        Determines severity level based on network issue type
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$IssueType
    )

    $critical = @('NoConnectivity', 'DNSFailure', 'AdapterDisabled', 'IPConflict')
    $warning = @('HighLatency', 'PacketLoss', 'SlowDNS', 'SuboptimalMTU')
    $info = @('UnusedAdapter', 'IPv6Disabled', 'VirtualAdapter')

    if ($IssueType -in $critical) { return 'Critical' }
    if ($IssueType -in $warning) { return 'Warning' }
    if ($IssueType -in $info) { return 'Info' }
    return 'Unknown'
}

function Test-RegistryKeyExists {
    <#
    .SYNOPSIS
        Tests if a registry key exists
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    try {
        return Test-Path -Path $Path -ErrorAction SilentlyContinue
    }
    catch {
        return $false
    }
}

function Get-RegistryValueSafe {
    <#
    .SYNOPSIS
        Safely retrieves a registry value with default fallback
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [string]$Name,

        [Parameter()]
        $DefaultValue = $null
    )

    try {
        if (Test-Path -Path $Path) {
            $value = Get-ItemProperty -Path $Path -Name $Name -ErrorAction SilentlyContinue
            if ($null -ne $value.$Name) {
                return $value.$Name
            }
        }
    }
    catch {
        # Silent failure, return default
    }
    return $DefaultValue
}

function Set-RegistryValueSafe {
    <#
    .SYNOPSIS
        Safely sets a registry value with backup capability
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [string]$Name,

        [Parameter(Mandatory)]
        $Value,

        [Parameter(Mandatory)]
        [ValidateSet('String', 'DWord', 'QWord', 'Binary', 'MultiString', 'ExpandString')]
        [string]$PropertyType,

        [Parameter()]
        [string]$BackupPath
    )

    try {
        # Create backup if path provided
        if ($BackupPath -and (Test-Path -Path $Path)) {
            $existingValue = Get-ItemProperty -Path $Path -Name $Name -ErrorAction SilentlyContinue
            if ($null -ne $existingValue.$Name) {
                $backup = @{
                    Path = $Path
                    Name = $Name
                    Value = $existingValue.$Name
                    Timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
                }
                $backupDir = Split-Path $BackupPath -Parent
                if (-not (Test-Path $backupDir)) {
                    New-Item -Path $backupDir -ItemType Directory -Force | Out-Null
                }
                $backup | ConvertTo-Json | Out-File -FilePath $BackupPath -Append
            }
        }

        # Ensure path exists
        if (-not (Test-Path -Path $Path)) {
            if ($PSCmdlet.ShouldProcess($Path, 'Create Registry Key')) {
                New-Item -Path $Path -Force | Out-Null
            }
        }

        # Set value
        if ($PSCmdlet.ShouldProcess("$Path\$Name", "Set to $Value")) {
            Set-ItemProperty -Path $Path -Name $Name -Value $Value -Type $PropertyType -Force
            return $true
        }
        return $false
    }
    catch {
        Write-Warning "Failed to set registry value: $_"
        return $false
    }
}

function Format-BytesPerSecond {
    <#
    .SYNOPSIS
        Formats bytes per second into human-readable format
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [decimal]$BytesPerSecond
    )

    $units = @('B/s', 'KB/s', 'MB/s', 'GB/s', 'TB/s')
    $unitIndex = 0
    $value = $BytesPerSecond

    while ($value -ge 1024 -and $unitIndex -lt ($units.Count - 1)) {
        $value /= 1024
        $unitIndex++
    }

    return '{0:N2} {1}' -f $value, $units[$unitIndex]
}

function Format-Latency {
    <#
    .SYNOPSIS
        Formats latency value with appropriate units
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [decimal]$Milliseconds
    )

    if ($Milliseconds -lt 1) {
        return '{0:N2} us' -f ($Milliseconds * 1000)
    }
    elseif ($Milliseconds -ge 1000) {
        return '{0:N2} s' -f ($Milliseconds / 1000)
    }
    return '{0:N2} ms' -f $Milliseconds
}

function Get-WSLDistributions {
    <#
    .SYNOPSIS
        Gets list of installed WSL distributions
    #>
    [CmdletBinding()]
    param()

    try {
        $wslOutput = wsl -l -v 2>&1
        if ($LASTEXITCODE -ne 0) {
            return @()
        }

        $distributions = @()
        $lines = $wslOutput -split "`n" | Where-Object { $_ -match '\S' }

        foreach ($line in $lines | Select-Object -Skip 1) {
            # Remove asterisk for default and parse
            $cleanLine = $line -replace '^\*\s*', '' -replace '\s+', ' '
            $parts = $cleanLine.Trim().Split(' ')

            if ($parts.Count -ge 3) {
                $distributions += [PSCustomObject]@{
                    Name = $parts[0]
                    State = $parts[1]
                    Version = $parts[2]
                }
            }
        }

        return $distributions
    }
    catch {
        return @()
    }
}

function Test-PortConnectivity {
    <#
    .SYNOPSIS
        Tests TCP connectivity to a specific port
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Host,

        [Parameter(Mandatory)]
        [int]$Port,

        [Parameter()]
        [int]$TimeoutMs = 3000
    )

    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $connect = $tcpClient.BeginConnect($Host, $Port, $null, $null)
        $wait = $connect.AsyncWaitHandle.WaitOne($TimeoutMs, $false)

        if ($wait) {
            try {
                $tcpClient.EndConnect($connect)
                $result = [PSCustomObject]@{
                    Host = $Host
                    Port = $Port
                    Success = $true
                    Message = 'Connected'
                }
            }
            catch {
                $result = [PSCustomObject]@{
                    Host = $Host
                    Port = $Port
                    Success = $false
                    Message = $_.Exception.Message
                }
            }
        }
        else {
            $result = [PSCustomObject]@{
                Host = $Host
                Port = $Port
                Success = $false
                Message = 'Connection timeout'
            }
        }

        $tcpClient.Close()
        return $result
    }
    catch {
        return [PSCustomObject]@{
            Host = $Host
            Port = $Port
            Success = $false
            Message = $_.Exception.Message
        }
    }
}

function Measure-NetworkLatency {
    <#
    .SYNOPSIS
        Measures network latency to a host
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Target,

        [Parameter()]
        [int]$Count = 4
    )

    try {
        $pingResults = Test-Connection -ComputerName $Target -Count $Count -ErrorAction SilentlyContinue

        if ($pingResults) {
            $latencies = $pingResults | ForEach-Object { $_.ResponseTime }
            return [PSCustomObject]@{
                Target = $Target
                Success = $true
                MinLatency = ($latencies | Measure-Object -Minimum).Minimum
                MaxLatency = ($latencies | Measure-Object -Maximum).Maximum
                AvgLatency = ($latencies | Measure-Object -Average).Average
                PacketLoss = (($Count - $pingResults.Count) / $Count) * 100
            }
        }
        else {
            return [PSCustomObject]@{
                Target = $Target
                Success = $false
                MinLatency = $null
                MaxLatency = $null
                AvgLatency = $null
                PacketLoss = 100
            }
        }
    }
    catch {
        return [PSCustomObject]@{
            Target = $Target
            Success = $false
            MinLatency = $null
            MaxLatency = $null
            AvgLatency = $null
            PacketLoss = 100
            Error = $_.Exception.Message
        }
    }
}

function ConvertTo-NetworkReportSection {
    <#
    .SYNOPSIS
        Formats network data into a report section
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Title,

        [Parameter()]
        [object]$Data,

        [Parameter()]
        [string]$EmptyMessage = 'No data found.'
    )

    $output = @()
    $output += "== $Title =="
    $output += ''

    if ($null -eq $Data -or ($Data -is [array] -and $Data.Count -eq 0)) {
        $output += $EmptyMessage
    }
    else {
        if ($Data -is [string]) {
            $output += $Data
        }
        else {
            $output += ($Data | Format-Table -AutoSize | Out-String).Trim()
        }
    }

    $output += ''
    return $output -join "`r`n"
}
