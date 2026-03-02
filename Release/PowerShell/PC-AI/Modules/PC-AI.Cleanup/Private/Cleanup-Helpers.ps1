#Requires -Version 5.1
<#
.SYNOPSIS
    Internal helper functions for PC-AI.Cleanup module.

.DESCRIPTION
    Contains private helper functions used by public cleanup functions.
    These functions are not exported and should not be called directly.

.NOTES
    Module: PC-AI.Cleanup
    Author: PC_AI Project
#>

function Write-CleanupLog {
    <#
    .SYNOPSIS
        Writes a log entry to the module log file.

    .PARAMETER Message
        The message to log.

    .PARAMETER Level
        Log level: Info, Warning, Error. Default is Info.

    .PARAMETER LogFile
        Optional specific log file name. Default is 'cleanup.log'.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string]$Message,

        [Parameter()]
        [ValidateSet('Info', 'Warning', 'Error')]
        [string]$Level = 'Info',

        [Parameter()]
        [string]$LogFile = 'cleanup.log'
    )

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $logPath = Join-Path -Path $script:LogPath -ChildPath $LogFile
    $logEntry = "[$timestamp] [$Level] $Message"

    try {
        Add-Content -Path $logPath -Value $logEntry -ErrorAction Stop
    }
    catch {
        Write-Verbose "Could not write to log file: $_"
    }

    # Also output based on level
    switch ($Level) {
        'Warning' { Write-Warning $Message }
        'Error' { Write-Error $Message }
        default { Write-Verbose $Message }
    }
}

function Test-IsAdministrator {
    <#
    .SYNOPSIS
        Tests if the current session is running with administrator privileges.

    .OUTPUTS
        System.Boolean - True if running as administrator, False otherwise.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-FileHashSafe {
    <#
    .SYNOPSIS
        Gets file hash with error handling for locked/inaccessible files.

    .PARAMETER Path
        Path to the file.

    .PARAMETER Algorithm
        Hash algorithm to use. Default is SHA256.

    .OUTPUTS
        String - The computed hash, or $null if file cannot be read.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter()]
        [ValidateSet('MD5', 'SHA1', 'SHA256', 'SHA384', 'SHA512')]
        [string]$Algorithm = 'SHA256'
    )

    try {
        $hash = Get-FileHash -Path $Path -Algorithm $Algorithm -ErrorAction Stop
        return $hash.Hash
    }
    catch {
        Write-Verbose "Could not hash file '$Path': $_"
        return $null
    }
}

function Format-FileSize {
    <#
    .SYNOPSIS
        Formats a file size in bytes to human-readable format.

    .PARAMETER Bytes
        Size in bytes.

    .OUTPUTS
        String - Formatted size (e.g., "1.5 GB").
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true)]
        [long]$Bytes
    )

    if ($Bytes -ge 1TB) {
        return "{0:N2} TB" -f ($Bytes / 1TB)
    }
    elseif ($Bytes -ge 1GB) {
        return "{0:N2} GB" -f ($Bytes / 1GB)
    }
    elseif ($Bytes -ge 1MB) {
        return "{0:N2} MB" -f ($Bytes / 1MB)
    }
    elseif ($Bytes -ge 1KB) {
        return "{0:N2} KB" -f ($Bytes / 1KB)
    }
    else {
        return "$Bytes Bytes"
    }
}

function Backup-EnvironmentVariable {
    <#
    .SYNOPSIS
        Creates a backup of an environment variable value.

    .PARAMETER Name
        Name of the environment variable.

    .PARAMETER Target
        Target scope: User or Machine.

    .PARAMETER BackupPath
        Optional path for backup file. Default is in PC-AI Logs folder.

    .OUTPUTS
        String - Path to the backup file.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [ValidateSet('User', 'Machine')]
        [string]$Target,

        [Parameter()]
        [string]$BackupPath
    )

    $value = [Environment]::GetEnvironmentVariable($Name, $Target)

    if ([string]::IsNullOrEmpty($value)) {
        Write-Verbose "Environment variable '$Name' ($Target) is empty, skipping backup."
        return $null
    }

    if (-not $BackupPath) {
        $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $BackupPath = Join-Path -Path $script:LogPath -ChildPath "${Name}_${Target}_$timestamp.bak"
    }

    try {
        [System.IO.File]::WriteAllText($BackupPath, $value, [System.Text.Encoding]::UTF8)
        Write-CleanupLog -Message "Backed up $Name ($Target) to: $BackupPath" -Level Info
        return $BackupPath
    }
    catch {
        Write-CleanupLog -Message "Failed to backup $Name ($Target): $_" -Level Error
        return $null
    }
}

function Test-PathExists {
    <#
    .SYNOPSIS
        Tests if a path exists, handling various edge cases.

    .PARAMETER Path
        Path to test.

    .OUTPUTS
        Boolean - True if path exists and is accessible.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    # Handle empty or whitespace
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $false
    }

    # Expand environment variables
    $expandedPath = [Environment]::ExpandEnvironmentVariables($Path)

    # Test existence
    return Test-Path -Path $expandedPath -ErrorAction SilentlyContinue
}

function Get-TempPaths {
    <#
    .SYNOPSIS
        Gets a list of common temporary file locations.

    .OUTPUTS
        Array of PSCustomObjects with Name and Path properties.
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param()

    $tempPaths = @()

    # Windows Temp
    $windowsTemp = Join-Path -Path $env:SystemRoot -ChildPath 'Temp'
    if (Test-Path -Path $windowsTemp) {
        $tempPaths += [PSCustomObject]@{
            Name = 'Windows Temp'
            Path = $windowsTemp
            RequiresAdmin = $true
        }
    }

    # User Temp
    $userTemp = $env:TEMP
    if (Test-Path -Path $userTemp) {
        $tempPaths += [PSCustomObject]@{
            Name = 'User Temp'
            Path = $userTemp
            RequiresAdmin = $false
        }
    }

    # Windows Prefetch
    $prefetch = Join-Path -Path $env:SystemRoot -ChildPath 'Prefetch'
    if (Test-Path -Path $prefetch) {
        $tempPaths += [PSCustomObject]@{
            Name = 'Windows Prefetch'
            Path = $prefetch
            RequiresAdmin = $true
        }
    }

    # Windows SoftwareDistribution Download
    $wuDownload = Join-Path -Path $env:SystemRoot -ChildPath 'SoftwareDistribution\Download'
    if (Test-Path -Path $wuDownload) {
        $tempPaths += [PSCustomObject]@{
            Name = 'Windows Update Download Cache'
            Path = $wuDownload
            RequiresAdmin = $true
        }
    }

    # User Downloads (optional, not included by default for safety)
    # Browser caches
    $localAppData = $env:LOCALAPPDATA

    # Chrome Cache
    $chromeCache = Join-Path -Path $localAppData -ChildPath 'Google\Chrome\User Data\Default\Cache'
    if (Test-Path -Path $chromeCache) {
        $tempPaths += [PSCustomObject]@{
            Name = 'Chrome Cache'
            Path = $chromeCache
            RequiresAdmin = $false
        }
    }

    # Edge Cache
    $edgeCache = Join-Path -Path $localAppData -ChildPath 'Microsoft\Edge\User Data\Default\Cache'
    if (Test-Path -Path $edgeCache) {
        $tempPaths += [PSCustomObject]@{
            Name = 'Edge Cache'
            Path = $edgeCache
            RequiresAdmin = $false
        }
    }

    # Firefox Cache
    $firefoxProfiles = Join-Path -Path $localAppData -ChildPath 'Mozilla\Firefox\Profiles'
    if (Test-Path -Path $firefoxProfiles) {
        Get-ChildItem -Path $firefoxProfiles -Directory | ForEach-Object {
            $ffCache = Join-Path -Path $_.FullName -ChildPath 'cache2'
            if (Test-Path -Path $ffCache) {
                $tempPaths += [PSCustomObject]@{
                    Name = "Firefox Cache ($($_.Name))"
                    Path = $ffCache
                    RequiresAdmin = $false
                }
            }
        }
    }

    # Thumbnail Cache
    $thumbCache = Join-Path -Path $localAppData -ChildPath 'Microsoft\Windows\Explorer'
    if (Test-Path -Path $thumbCache) {
        $tempPaths += [PSCustomObject]@{
            Name = 'Windows Thumbnail Cache'
            Path = $thumbCache
            RequiresAdmin = $false
            Filter = 'thumbcache_*.db'
        }
    }

    return $tempPaths
}

function Remove-FilesSafely {
    <#
    .SYNOPSIS
        Removes files safely, skipping those in use.

    .PARAMETER Path
        Path to clean (directory).

    .PARAMETER Filter
        Optional file filter pattern.

    .PARAMETER Recurse
        Whether to recurse into subdirectories.

    .PARAMETER OlderThanDays
        Only remove files older than this many days.

    .OUTPUTS
        PSCustomObject with deletion statistics.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter()]
        [string]$Filter = '*',

        [Parameter()]
        [switch]$Recurse,

        [Parameter()]
        [int]$OlderThanDays = 0
    )

    $result = [PSCustomObject]@{
        Path = $Path
        FilesDeleted = 0
        FilesSkipped = 0
        BytesReclaimed = 0
        Errors = @()
    }

    if (-not (Test-Path -Path $Path)) {
        return $result
    }

    $cutoffDate = (Get-Date).AddDays(-$OlderThanDays)

    $getChildParams = @{
        Path = $Path
        File = $true
        ErrorAction = 'SilentlyContinue'
    }

    if ($Filter -ne '*') {
        $getChildParams['Filter'] = $Filter
    }

    if ($Recurse) {
        $getChildParams['Recurse'] = $true
    }

    $files = Get-ChildItem @getChildParams

    foreach ($file in $files) {
        # Skip if newer than cutoff
        if ($OlderThanDays -gt 0 -and $file.LastWriteTime -gt $cutoffDate) {
            $result.FilesSkipped++
            continue
        }

        if ($PSCmdlet.ShouldProcess($file.FullName, 'Delete')) {
            try {
                $size = $file.Length
                Remove-Item -Path $file.FullName -Force -ErrorAction Stop
                $result.FilesDeleted++
                $result.BytesReclaimed += $size
            }
            catch {
                $result.FilesSkipped++
                $result.Errors += "Could not delete '$($file.FullName)': $($_.Exception.Message)"
            }
        }
    }

    return $result
}
