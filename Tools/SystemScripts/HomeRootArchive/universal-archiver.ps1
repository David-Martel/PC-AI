#Requires -Version 5.1
<#
.SYNOPSIS
    Universal Archiver - Advanced file archiving tool with smart exclusions and multiple format support

.DESCRIPTION
    A comprehensive PowerShell archiving tool that supports multiple archive formats,
    password protection, smart exclusions, and detailed reporting. Designed for efficient
    backup and archiving of directories with configurable rules.

.PARAMETER Path
    The source path to archive (file or directory)

.PARAMETER OutputPath
    The destination path for the archive file (optional - defaults to source directory)

.PARAMETER Format
    Archive format to use. Valid options: 7z, zip, tar, gzip, bzip2, xz
    Default: 7z

.PARAMETER Level
    Compression level. Valid options: store, fastest, fast, normal, maximum, ultra
    Default: normal

.PARAMETER Password
    Password for encrypted archives (SecureString). Use with -Encrypt flag.

.PARAMETER Encrypt
    Enable password protection for the archive

.PARAMETER Exclude
    Array of patterns to exclude from the archive

.PARAMETER Include
    Array of patterns to include in the archive (overrides exclusions)

.PARAMETER ConfigFile
    Path to JSON configuration file with exclusion rules and settings

.PARAMETER ProfileName
    Name of exclusion profile to use from config file

.PARAMETER Incremental
    Enable incremental backup mode (only archive changed files)

.PARAMETER DedupeCheck
    Enable file deduplication detection before archiving

.PARAMETER LogPath
    Path for detailed log file (optional)

.PARAMETER Verify
    Verify archive integrity after creation

.PARAMETER DryRun
    Show what would be archived without creating the archive

.PARAMETER Quiet
    Suppress progress output (except errors)

.EXAMPLE
    .\universal-archiver.ps1 -Path "C:\MyFolder" -Format 7z -Level maximum

    Creates a 7z archive with maximum compression

.EXAMPLE
    .\universal-archiver.ps1 -Path "C:\MyFolder" -Encrypt -Format zip -Exclude "*.tmp","*.log"

    Creates an encrypted zip archive excluding temporary and log files

.EXAMPLE
    .\universal-archiver.ps1 -Path "C:\MyFolder" -ConfigFile ".\archiver-config.json" -ProfileName "development"

    Uses predefined exclusion profile from config file

.NOTES
    Author: Universal Archiver Tool
    Version: 2.0.0
    Requires: 7-Zip for advanced compression formats

    Dependencies:
    - 7-Zip (for 7z, tar, gzip, bzip2, xz formats)
    - PowerShell 5.1 or higher
#>

[CmdletBinding(DefaultParameterSetName = 'Default')]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateScript({Test-Path $_ -PathType Any})]
    [string]$Path,

    [Parameter()]
    [string]$OutputPath,

    [Parameter()]
    [ValidateSet('7z', 'zip', 'tar', 'gzip', 'bzip2', 'xz')]
    [string]$Format = '7z',

    [Parameter()]
    [ValidateSet('store', 'fastest', 'fast', 'normal', 'maximum', 'ultra')]
    [string]$Level = 'normal',

    [Parameter()]
    [SecureString]$Password,

    [Parameter()]
    [switch]$Encrypt,

    [Parameter()]
    [string[]]$Exclude = @(),

    [Parameter()]
    [string[]]$Include = @(),

    [Parameter()]
    [ValidateScript({Test-Path $_ -PathType Leaf})]
    [string]$ConfigFile,

    [Parameter()]
    [string]$ProfileName,

    [Parameter()]
    [switch]$Incremental,

    [Parameter()]
    [switch]$DedupeCheck,

    [Parameter()]
    [string]$LogPath,

    [Parameter()]
    [switch]$Verify,

    [Parameter()]
    [switch]$DryRun,

    [Parameter()]
    [switch]$Quiet
)

# Global variables
$script:LogFile = $null
$script:StartTime = Get-Date
$script:Config = $null

# Default smart exclusions for common scenarios
$script:DefaultExclusions = @{
    'GoogleDrive' = @(
        '*.gsheet', '*.gdoc', '*.gslides', '*.gdraw', '*.gtable',
        '*.desktop.ini', 'desktop.ini', 'thumbs.db', 'Thumbs.db'
    )
    'Development' = @(
        'node_modules', '.git', '.svn', '.hg',
        'bin', 'obj', '*.pdb', '*.exe', '*.dll',
        '.vs', '.vscode', '*.log', '*.tmp'
    )
    'System' = @(
        '$RECYCLE.BIN', 'System Volume Information',
        'pagefile.sys', 'hiberfil.sys', 'swapfile.sys',
        '~$*', '.DS_Store', '*.lnk'
    )
    'Media' = @(
        '*.tmp', '*.temp', '*.cache',
        'Adobe Premiere Pro Auto-Save', 'Adobe After Effects Auto-Save'
    )
}

# Compression level mappings
$script:CompressionLevels = @{
    '7z' = @{
        'store' = '-mx0'; 'fastest' = '-mx1'; 'fast' = '-mx3';
        'normal' = '-mx5'; 'maximum' = '-mx7'; 'ultra' = '-mx9'
    }
    'zip' = @{
        'store' = '-mx0'; 'fastest' = '-mx1'; 'fast' = '-mx3';
        'normal' = '-mx5'; 'maximum' = '-mx7'; 'ultra' = '-mx9'
    }
}

# Initialize logging
function Initialize-Logging {
    if ($LogPath) {
        $script:LogFile = $LogPath
    } else {
        $logDir = Split-Path $Path -Parent
        $baseName = Split-Path $Path -Leaf
        $script:LogFile = Join-Path $logDir "archiver-$baseName-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"
    }

    if (-not $Quiet) {
        Write-Host "Logging to: $script:LogFile" -ForegroundColor Green
    }

    Write-LogEntry "=== Universal Archiver Started ===" -Level Info
    Write-LogEntry "Parameters:" -Level Info
    Write-LogEntry "  Path: $Path" -Level Info
    Write-LogEntry "  Format: $Format" -Level Info
    Write-LogEntry "  Level: $Level" -Level Info
    Write-LogEntry "  Encrypt: $Encrypt" -Level Info
}

# Logging function
function Write-LogEntry {
    param(
        [string]$Message,
        [ValidateSet('Info', 'Warning', 'Error', 'Debug')]
        [string]$Level = 'Info'
    )

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $logMessage = "[$timestamp] [$Level] $Message"

    if ($script:LogFile) {
        Add-Content -Path $script:LogFile -Value $logMessage -Encoding UTF8
    }

    if (-not $Quiet -or $Level -eq 'Error') {
        $color = switch ($Level) {
            'Info' { 'White' }
            'Warning' { 'Yellow' }
            'Error' { 'Red' }
            'Debug' { 'Gray' }
        }
        Write-Host $logMessage -ForegroundColor $color
    }
}

# Load configuration file
function Load-Configuration {
    if ($ConfigFile -and (Test-Path $ConfigFile)) {
        try {
            $script:Config = Get-Content $ConfigFile -Raw | ConvertFrom-Json
            Write-LogEntry "Loaded configuration from: $ConfigFile" -Level Info
        } catch {
            Write-LogEntry "Failed to load configuration: $($_.Exception.Message)" -Level Error
            throw
        }
    } else {
        # Create default configuration
        $script:Config = [PSCustomObject]@{
            profiles = [PSCustomObject]@{
                default = [PSCustomObject]@{
                    exclude = $script:DefaultExclusions.GoogleDrive + $script:DefaultExclusions.System
                    include = @()
                }
                development = [PSCustomObject]@{
                    exclude = $script:DefaultExclusions.Development + $script:DefaultExclusions.System
                    include = @()
                }
                media = [PSCustomObject]@{
                    exclude = $script:DefaultExclusions.Media + $script:DefaultExclusions.System
                    include = @()
                }
            }
            settings = [PSCustomObject]@{
                defaultFormat = '7z'
                defaultLevel = 'normal'
                verifyByDefault = $true
                logRetentionDays = 30
            }
        }
    }
}

# Get exclusion patterns
function Get-ExclusionPatterns {
    $patterns = @()

    # Add profile exclusions
    if ($ProfileName -and $script:Config.profiles.$ProfileName) {
        $patterns += $script:Config.profiles.$ProfileName.exclude
        Write-LogEntry "Using profile '$ProfileName' with $($script:Config.profiles.$ProfileName.exclude.Count) exclusions" -Level Info
    } elseif ($script:Config.profiles.default) {
        $patterns += $script:Config.profiles.default.exclude
        Write-LogEntry "Using default profile with $($script:Config.profiles.default.exclude.Count) exclusions" -Level Info
    }

    # Add command-line exclusions
    if ($Exclude.Count -gt 0) {
        $patterns += $Exclude
        Write-LogEntry "Added $($Exclude.Count) command-line exclusions" -Level Info
    }

    return $patterns | Select-Object -Unique
}

# Get inclusion patterns
function Get-InclusionPatterns {
    $patterns = @()

    # Add profile inclusions
    if ($ProfileName -and $script:Config.profiles.$ProfileName) {
        $patterns += $script:Config.profiles.$ProfileName.include
    } elseif ($script:Config.profiles.default) {
        $patterns += $script:Config.profiles.default.include
    }

    # Add command-line inclusions
    if ($Include.Count -gt 0) {
        $patterns += $Include
    }

    return $patterns | Select-Object -Unique
}

# Check for 7-Zip installation
function Test-7ZipAvailable {
    $sevenZipPaths = @(
        "${env:ProgramFiles}\7-Zip\7z.exe",
        "${env:ProgramFiles(x86)}\7-Zip\7z.exe",
        "7z.exe"  # Check PATH
    )

    foreach ($path in $sevenZipPaths) {
        try {
            $result = if ($path -eq "7z.exe") {
                Get-Command "7z.exe" -ErrorAction SilentlyContinue
            } else {
                if (Test-Path $path) { $path }
            }

            if ($result) {
                $sevenZipPath = if ($result.Source) { $result.Source } else { $result }
                Write-LogEntry "Found 7-Zip at: $sevenZipPath" -Level Info
                return $sevenZipPath
            }
        } catch {
            continue
        }
    }

    return $null
}

# Get file information for deduplication
function Get-FileHash {
    param([string]$FilePath)

    try {
        $hash = Get-FileHash $FilePath -Algorithm SHA256
        return $hash.Hash
    } catch {
        Write-LogEntry "Failed to calculate hash for: $FilePath" -Level Warning
        return $null
    }
}

# Perform deduplication check
function Test-Duplication {
    param([string[]]$FilePaths)

    if (-not $DedupeCheck) {
        return $FilePaths
    }

    Write-LogEntry "Performing deduplication check..." -Level Info
    $hashTable = @{}
    $uniqueFiles = @()
    $duplicateCount = 0

    foreach ($file in $FilePaths) {
        if (Test-Path $file -PathType Leaf) {
            $hash = Get-FileHash $file
            if ($hashTable.ContainsKey($hash)) {
                Write-LogEntry "Duplicate found: $file (same as $($hashTable[$hash]))" -Level Debug
                $duplicateCount++
            } else {
                $hashTable[$hash] = $file
                $uniqueFiles += $file
            }
        } else {
            $uniqueFiles += $file
        }
    }

    if ($duplicateCount -gt 0) {
        Write-LogEntry "Deduplication: Removed $duplicateCount duplicate files" -Level Info
    }

    return $uniqueFiles
}

# Get files to archive with filtering
function Get-FilesToArchive {
    param(
        [string]$SourcePath,
        [string[]]$ExcludePatterns,
        [string[]]$IncludePatterns
    )

    Write-LogEntry "Scanning files in: $SourcePath" -Level Info

    $allFiles = @()
    if (Test-Path $SourcePath -PathType Container) {
        $allFiles = Get-ChildItem $SourcePath -Recurse -File | ForEach-Object { $_.FullName }
    } else {
        $allFiles = @($SourcePath)
    }

    Write-LogEntry "Found $($allFiles.Count) total files" -Level Info

    # Apply exclusions
    $filteredFiles = $allFiles
    foreach ($pattern in $ExcludePatterns) {
        $beforeCount = $filteredFiles.Count
        $filteredFiles = $filteredFiles | Where-Object { $_ -notlike "*$pattern*" }
        $excluded = $beforeCount - $filteredFiles.Count
        if ($excluded -gt 0) {
            Write-LogEntry "Excluded $excluded files matching pattern: $pattern" -Level Debug
        }
    }

    # Apply inclusions (override exclusions)
    if ($IncludePatterns.Count -gt 0) {
        $includedFiles = @()
        foreach ($pattern in $IncludePatterns) {
            $matches = $allFiles | Where-Object { $_ -like "*$pattern*" }
            $includedFiles += $matches
            Write-LogEntry "Included $($matches.Count) files matching pattern: $pattern" -Level Debug
        }
        $filteredFiles = ($filteredFiles + $includedFiles) | Select-Object -Unique
    }

    # Deduplication check
    if ($DedupeCheck) {
        $filteredFiles = Test-Duplication -FilePaths $filteredFiles
    }

    Write-LogEntry "Final file count after filtering: $($filteredFiles.Count)" -Level Info
    return $filteredFiles
}

# Generate archive filename
function Get-ArchiveFileName {
    param(
        [string]$SourcePath,
        [string]$Format,
        [string]$OutputPath
    )

    if ($OutputPath) {
        if (Test-Path $OutputPath -PathType Container) {
            $baseName = Split-Path $SourcePath -Leaf
            return Join-Path $OutputPath "$baseName-$(Get-Date -Format 'yyyyMMdd-HHmmss').$Format"
        } else {
            return $OutputPath
        }
    } else {
        $parentPath = Split-Path $SourcePath -Parent
        $baseName = Split-Path $SourcePath -Leaf
        return Join-Path $parentPath "$baseName-$(Get-Date -Format 'yyyyMMdd-HHmmss').$Format"
    }
}

# Create archive using appropriate method
function New-Archive {
    param(
        [string]$SourcePath,
        [string]$DestinationPath,
        [string]$Format,
        [string]$Level,
        [SecureString]$Password,
        [string[]]$ExcludePatterns,
        [string[]]$IncludePatterns
    )

    try {
        $sevenZipPath = Test-7ZipAvailable

        if ($Format -in @('7z', 'tar', 'gzip', 'bzip2', 'xz') -and -not $sevenZipPath) {
            throw "7-Zip is required for $Format format but was not found. Please install 7-Zip."
        }

        if ($Format -eq 'zip' -and -not $sevenZipPath) {
            # Use PowerShell native compression for ZIP
            return New-ZipArchive -SourcePath $SourcePath -DestinationPath $DestinationPath -Level $Level -Password $Password -ExcludePatterns $ExcludePatterns -IncludePatterns $IncludePatterns
        } else {
            # Use 7-Zip for all formats
            return New-7ZipArchive -SourcePath $SourcePath -DestinationPath $DestinationPath -Format $Format -Level $Level -Password $Password -ExcludePatterns $ExcludePatterns -IncludePatterns $IncludePatterns -SevenZipPath $sevenZipPath
        }
    } catch {
        Write-LogEntry "Archive creation failed: $($_.Exception.Message)" -Level Error
        throw
    }
}

# Create ZIP archive using PowerShell
function New-ZipArchive {
    param(
        [string]$SourcePath,
        [string]$DestinationPath,
        [string]$Level,
        [SecureString]$Password,
        [string[]]$ExcludePatterns,
        [string[]]$IncludePatterns
    )

    Write-LogEntry "Creating ZIP archive using PowerShell compression..." -Level Info

    if ($Password) {
        Write-LogEntry "Warning: PowerShell ZIP compression does not support password protection. Consider using 7z format." -Level Warning
    }

    # Get files to archive
    $filesToArchive = Get-FilesToArchive -SourcePath $SourcePath -ExcludePatterns $ExcludePatterns -IncludePatterns $IncludePatterns

    if ($DryRun) {
        Write-LogEntry "DRY RUN: Would archive $($filesToArchive.Count) files to $DestinationPath" -Level Info
        return $true
    }

    # Create temporary directory structure
    $tempDir = Join-Path $env:TEMP "UniversalArchiver-$(Get-Date -Format 'yyyyMMddHHmmss')"
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

    try {
        if (Test-Path $SourcePath -PathType Container) {
            # Copy directory structure
            $sourceParent = Split-Path $SourcePath -Parent
            $sourceName = Split-Path $SourcePath -Leaf
            $tempTarget = Join-Path $tempDir $sourceName

            foreach ($file in $filesToArchive) {
                $relativePath = $file.Substring($SourcePath.Length + 1)
                $targetFile = Join-Path $tempTarget $relativePath
                $targetDir = Split-Path $targetFile -Parent

                if (-not (Test-Path $targetDir)) {
                    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
                }
                Copy-Item $file $targetFile -Force
            }

            Compress-Archive -Path "$tempTarget\*" -DestinationPath $DestinationPath -CompressionLevel Optimal -Force
        } else {
            # Single file
            Copy-Item $SourcePath $tempDir -Force
            Compress-Archive -Path "$tempDir\*" -DestinationPath $DestinationPath -CompressionLevel Optimal -Force
        }

        Write-LogEntry "ZIP archive created successfully: $DestinationPath" -Level Info
        return $true
    } finally {
        # Clean up temporary directory
        if (Test-Path $tempDir) {
            Remove-Item $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# Create archive using 7-Zip
function New-7ZipArchive {
    param(
        [string]$SourcePath,
        [string]$DestinationPath,
        [string]$Format,
        [string]$Level,
        [SecureString]$Password,
        [string[]]$ExcludePatterns,
        [string[]]$IncludePatterns,
        [string]$SevenZipPath
    )

    Write-LogEntry "Creating $Format archive using 7-Zip..." -Level Info

    # Build 7-Zip command
    $arguments = @('a')  # Add command

    # Archive format
    $arguments += "-t$Format"

    # Compression level
    if ($script:CompressionLevels[$Format][$Level]) {
        $arguments += $script:CompressionLevels[$Format][$Level]
    }

    # Password protection
    if ($Password) {
        $plainPassword = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($Password))
        $arguments += "-p$plainPassword"
        $arguments += '-mhe=on'  # Encrypt headers
    }

    # Exclusion patterns
    foreach ($pattern in $ExcludePatterns) {
        $arguments += "-xr!*$pattern*"
    }

    # Output and source
    $arguments += "`"$DestinationPath`""
    $arguments += "`"$SourcePath`""

    if ($DryRun) {
        Write-LogEntry "DRY RUN: Would execute: $SevenZipPath $($arguments -join ' ')" -Level Info
        return $true
    }

    # Execute 7-Zip
    Write-LogEntry "Executing: $SevenZipPath $($arguments -join ' ')" -Level Debug

    try {
        $process = Start-Process -FilePath $SevenZipPath -ArgumentList $arguments -Wait -PassThru -NoNewWindow

        if ($process.ExitCode -eq 0) {
            Write-LogEntry "$Format archive created successfully: $DestinationPath" -Level Info
            return $true
        } else {
            Write-LogEntry "7-Zip failed with exit code $($process.ExitCode)" -Level Error
            return $false
        }
    } catch {
        Write-LogEntry "Failed to execute 7-Zip: $($_.Exception.Message)" -Level Error
        return $false
    }
}

# Verify archive integrity
function Test-ArchiveIntegrity {
    param(
        [string]$ArchivePath,
        [string]$Format
    )

    Write-LogEntry "Verifying archive integrity..." -Level Info

    try {
        $sevenZipPath = Test-7ZipAvailable

        if ($Format -eq 'zip' -and -not $sevenZipPath) {
            # Use PowerShell for ZIP verification
            $tempExtract = Join-Path $env:TEMP "ArchiveTest-$(Get-Random)"
            try {
                Expand-Archive -Path $ArchivePath -DestinationPath $tempExtract -Force
                Remove-Item $tempExtract -Recurse -Force -ErrorAction SilentlyContinue
                Write-LogEntry "Archive verification successful" -Level Info
                return $true
            } catch {
                Write-LogEntry "Archive verification failed: $($_.Exception.Message)" -Level Error
                return $false
            }
        } elseif ($sevenZipPath) {
            # Use 7-Zip test command
            $arguments = @('t', "`"$ArchivePath`"")

            $process = Start-Process -FilePath $sevenZipPath -ArgumentList $arguments -Wait -PassThru -NoNewWindow

            if ($process.ExitCode -eq 0) {
                Write-LogEntry "Archive verification successful" -Level Info
                return $true
            } else {
                $error = $process.StandardError.ReadToEnd()
                Write-LogEntry "Archive verification failed: $error" -Level Error
                return $false
            }
        } else {
            Write-LogEntry "Cannot verify archive: No verification tool available" -Level Warning
            return $true
        }
    } catch {
        Write-LogEntry "Archive verification error: $($_.Exception.Message)" -Level Error
        return $false
    }
}

# Generate final report
function Write-FinalReport {
    param(
        [string]$ArchivePath,
        [bool]$Success,
        [DateTime]$StartTime,
        [int]$FileCount
    )

    $endTime = Get-Date
    $duration = $endTime - $StartTime

    Write-LogEntry "=== Archive Operation Complete ===" -Level Info
    Write-LogEntry "Status: $(if ($Success) { 'SUCCESS' } else { 'FAILED' })" -Level $(if ($Success) { 'Info' } else { 'Error' })

    if ($Success) {
        Write-LogEntry "Archive Path: $ArchivePath" -Level Info

        if (Test-Path $ArchivePath) {
            $archiveSize = (Get-Item $ArchivePath).Length
            $sizeGB = [math]::Round($archiveSize / 1GB, 2)
            $sizeMB = [math]::Round($archiveSize / 1MB, 2)

            Write-LogEntry "Archive Size: $($sizeMB) MB ($($sizeGB) GB)" -Level Info
        }

        Write-LogEntry "Files Processed: $FileCount" -Level Info
    }

    Write-LogEntry "Duration: $($duration.ToString('hh\:mm\:ss'))" -Level Info
    Write-LogEntry "Log File: $script:LogFile" -Level Info
}

# Request password securely
function Request-SecurePassword {
    param([string]$Prompt = "Enter archive password")

    $securePassword = Read-Host -Prompt $Prompt -AsSecureString
    return $securePassword
}

# Main execution function
function Invoke-UniversalArchiver {
    try {
        # Initialize
        Initialize-Logging
        Load-Configuration

        # Get password if encryption requested
        if ($Encrypt -and -not $Password) {
            $Password = Request-SecurePassword
        }

        # Get exclusion and inclusion patterns
        $excludePatterns = Get-ExclusionPatterns
        $includePatterns = Get-InclusionPatterns

        Write-LogEntry "Exclusion patterns: $($excludePatterns.Count)" -Level Info
        Write-LogEntry "Inclusion patterns: $($includePatterns.Count)" -Level Info

        # Generate archive filename
        $archiveFileName = Get-ArchiveFileName -SourcePath $Path -Format $Format -OutputPath $OutputPath
        Write-LogEntry "Target archive: $archiveFileName" -Level Info

        # Get file count for reporting
        $filesToArchive = Get-FilesToArchive -SourcePath $Path -ExcludePatterns $excludePatterns -IncludePatterns $includePatterns

        if ($DryRun) {
            Write-LogEntry "=== DRY RUN SUMMARY ===" -Level Info
            Write-LogEntry "Source: $Path" -Level Info
            Write-LogEntry "Target: $archiveFileName" -Level Info
            Write-LogEntry "Format: $Format" -Level Info
            Write-LogEntry "Files to archive: $($filesToArchive.Count)" -Level Info
            Write-LogEntry "Encryption: $(if ($Encrypt) { 'Enabled' } else { 'Disabled' })" -Level Info
            return
        }

        # Create archive
        $success = New-Archive -SourcePath $Path -DestinationPath $archiveFileName -Format $Format -Level $Level -Password $Password -ExcludePatterns $excludePatterns -IncludePatterns $includePatterns

        # Verify archive if requested and creation was successful
        if ($success -and ($Verify -or $script:Config.settings.verifyByDefault)) {
            $verifySuccess = Test-ArchiveIntegrity -ArchivePath $archiveFileName -Format $Format
            $success = $success -and $verifySuccess
        }

        # Generate report
        Write-FinalReport -ArchivePath $archiveFileName -Success $success -StartTime $script:StartTime -FileCount $filesToArchive.Count

        if ($success) {
            exit 0
        } else {
            exit 1
        }

    } catch {
        Write-LogEntry "Fatal error: $($_.Exception.Message)" -Level Error
        Write-LogEntry $_.ScriptStackTrace -Level Error
        exit 1
    }
}

# Execute main function
Invoke-UniversalArchiver