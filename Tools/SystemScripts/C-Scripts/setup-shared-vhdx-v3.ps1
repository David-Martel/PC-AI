# WSL Shared VHDX Setup Script v3.0
# Advanced shared development environment for WSL and Windows
# Integrates with David's optimized New-PersistentVHDX function

param(
    [string]$VhdxPath = 'T:\vm\shared-dev.vhdx',
    [string]$Size = '256GB',
    [string]$Label = 'WSL-Shared-Dev',
    [string]$DriveLetterPreference = '',
    [switch]$AutoMount = $false,
    [switch]$Force = $false,
    [switch]$UseOptimizedFunction = $true
)

# Ensure running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] 'Administrator')) {
    Write-Error 'This script must be run as Administrator!'
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

Write-Host 'WSL Shared VHDX Setup v3.0' -ForegroundColor Green
Write-Host '==============================' -ForegroundColor Green

# Import David's optimized New-PersistentVHDX function if available and requested
$optimizedFunctionPath = 'C:\Users\david\Documents\PowerShell\New-PersistentVHDX.ps1'
$hasOptimizedFunction = $false

if ($UseOptimizedFunction -and (Test-Path $optimizedFunctionPath)) {
    try {
        . $optimizedFunctionPath
        $hasOptimizedFunction = $true
        Write-Host "✓ Loaded David's optimized New-PersistentVHDX function" -ForegroundColor Green
    } catch {
        Write-Warning "Could not load optimized function: $_"
        Write-Host 'Falling back to standard VHDX creation' -ForegroundColor Yellow
    }
} elseif ($UseOptimizedFunction) {
    Write-Warning "Optimized function not found at $optimizedFunctionPath"
    Write-Host 'Falling back to standard VHDX creation' -ForegroundColor Yellow
}

# Function to convert size string to bytes (for fallback method)
function Convert-SizeToBytes {
    param([string]$SizeString)

    $SizeString = $SizeString.ToUpper()
    if ($SizeString -match '(\d+)(GB|TB|MB)') {
        $number = [int]$matches[1]
        $unit = $matches[2]

        switch ($unit) {
            'MB' { return $number * 1MB }
            'GB' { return $number * 1GB }
            'TB' { return $number * 1TB }
        }
    }
    throw "Invalid size format. Use format like '50GB', '1TB', etc."
}

# Parse VHDX path components
$vhdxDir = Split-Path $VhdxPath -Parent
$vhdxName = [System.IO.Path]::GetFileNameWithoutExtension($VhdxPath)

Write-Host "`nConfiguration:" -ForegroundColor Yellow
Write-Host "  VHDX Path: $VhdxPath" -ForegroundColor Cyan
Write-Host "  Size: $Size" -ForegroundColor Cyan
Write-Host "  Label: $Label" -ForegroundColor Cyan
if ($DriveLetterPreference) {
    Write-Host "  Preferred Drive Letter: $DriveLetterPreference" -ForegroundColor Cyan
}
Write-Host "  Using Optimized Function: $($hasOptimizedFunction -and $UseOptimizedFunction)" -ForegroundColor Cyan

# Check if VHDX already exists
if (Test-Path $VhdxPath) {
    if (-not $Force) {
        Write-Warning "VHDX already exists at $VhdxPath"
        $choice = Read-Host 'Do you want to continue? This will mount the existing VHDX. (y/N)'
        if ($choice -ne 'y' -and $choice -ne 'Y') {
            Write-Host 'Operation cancelled.' -ForegroundColor Yellow
            exit 0
        }
    }
    Write-Host "Using existing VHDX at $VhdxPath" -ForegroundColor Cyan

    # Mount existing VHDX
    try {
        Write-Host 'Mounting existing VHDX...' -ForegroundColor Yellow
        $mountResult = Mount-VHD -Path $VhdxPath -Passthru
        $diskNumber = $mountResult.DiskNumber
        Write-Host "✓ VHDX mounted as Disk $diskNumber" -ForegroundColor Green

        # Get existing drive letter
        $partition = Get-Partition -DiskNumber $diskNumber | Where-Object { $_.Type -eq 'Basic' }
        if ($partition) {
            $driveLetter = $partition.DriveLetter
            Write-Host "✓ Existing partition found at drive $driveLetter`:" -ForegroundColor Green
        }
    } catch {
        Write-Error "Failed to mount existing VHDX: $_"
        exit 1
    }
} else {
    # Create VHDX using optimized function or fallback
    if ($hasOptimizedFunction -and $UseOptimizedFunction) {
        Write-Host "`nUsing David's optimized New-PersistentVHDX function..." -ForegroundColor Green

        # Extract size number from string like "50GB"
        if ($Size -match '(\d+)') {
            $sizeGB = [int]$matches[1]
        } else {
            throw "Could not parse size from '$Size'"
        }

        # Prepare parameters for optimized function
        $vhdxParams = @{
            Location    = $vhdxDir
            Name        = $vhdxName
            SizeInGB    = $sizeGB
            VolumeLabel = $Label
        }

        if ($DriveLetterPreference -and $DriveLetterPreference -match '^[A-Za-z]$') {
            $vhdxParams.DriveLetterPreference = $DriveLetterPreference
        }

        try {
            # Create directory if it doesn't exist
            if (-not (Test-Path $vhdxDir)) {
                Write-Host "Creating directory: $vhdxDir" -ForegroundColor Yellow
                New-Item -Path $vhdxDir -ItemType Directory -Force | Out-Null
            }

            # Use David's optimized function
            $result = New-PersistentVHDX @vhdxParams -Verbose

            if ($result) {
                Write-Host '✓ VHDX created successfully using optimized function' -ForegroundColor Green

                # Get the mounted drive information
                $mountedVHD = Get-VHD -Path $VhdxPath
                if ($mountedVHD.Attached) {
                    $disk = Get-Disk -Number $mountedVHD.DiskNumber
                    $partition = Get-Partition -DiskNumber $disk.Number | Where-Object { $_.Type -eq 'Basic' }
                    if ($partition) {
                        $driveLetter = $partition.DriveLetter
                        Write-Host "✓ VHDX mounted at drive $driveLetter`:" -ForegroundColor Green
                    }
                } else {
                    throw 'VHDX not mounted after creation'
                }
            } else {
                throw 'Optimized function returned false'
            }
        } catch {
            Write-Warning "Optimized function failed: $_"
            Write-Host 'Falling back to standard method...' -ForegroundColor Yellow
            $hasOptimizedFunction = $false
        }
    }

    # Fallback to standard method if optimized function failed or not available
    if (-not $hasOptimizedFunction -or -not $UseOptimizedFunction) {
        Write-Host "`nUsing standard VHDX creation method..." -ForegroundColor Yellow

        # Create directory for VHDX if it doesn't exist
        if (-not (Test-Path $vhdxDir)) {
            Write-Host "Creating directory: $vhdxDir" -ForegroundColor Yellow
            New-Item -Path $vhdxDir -ItemType Directory -Force | Out-Null
        }

        # Convert size to bytes
        try {
            $SizeBytes = Convert-SizeToBytes $Size
            Write-Host "Creating VHDX: $VhdxPath ($Size)" -ForegroundColor Yellow

            # Create the VHDX
            New-VHD -Path $VhdxPath -SizeBytes $SizeBytes -Dynamic -Confirm:$false
            Write-Host '✓ VHDX created successfully' -ForegroundColor Green
        } catch {
            Write-Error "Failed to create VHDX: $_"
            exit 1
        }

        try {
            # Mount the VHDX
            Write-Host 'Mounting VHDX...' -ForegroundColor Yellow
            $mountResult = Mount-VHD -Path $VhdxPath -Passthru
            $diskNumber = $mountResult.DiskNumber
            Write-Host "✓ VHDX mounted as Disk $diskNumber" -ForegroundColor Green

            # Get disk information
            $disk = Get-Disk -Number $diskNumber

            # Check if disk needs to be initialized
            if ($disk.PartitionStyle -eq 'RAW') {
                Write-Host 'Initializing disk with GPT partition style...' -ForegroundColor Yellow
                Initialize-Disk -Number $diskNumber -PartitionStyle GPT -Confirm:$false
                Write-Host '✓ Disk initialized' -ForegroundColor Green

                # Create partition
                Write-Host 'Creating partition...' -ForegroundColor Yellow
                if ($DriveLetterPreference -and $DriveLetterPreference -match '^[A-Za-z]$' -and -not (Get-Volume -DriveLetter $DriveLetterPreference -ErrorAction SilentlyContinue)) {
                    $partition = New-Partition -DiskNumber $diskNumber -UseMaximumSize -DriveLetter $DriveLetterPreference
                    $driveLetter = $DriveLetterPreference
                } else {
                    $partition = New-Partition -DiskNumber $diskNumber -UseMaximumSize -AssignDriveLetter
                    $driveLetter = $partition.DriveLetter
                }
                Write-Host "✓ Partition created with drive letter $driveLetter" -ForegroundColor Green

                # Format the volume
                Write-Host 'Formatting volume with NTFS...' -ForegroundColor Yellow
                Format-Volume -DriveLetter $driveLetter -FileSystem NTFS -NewFileSystemLabel $Label -Confirm:$false -Force
                Write-Host "✓ Volume formatted as $driveLetter`: with label '$Label'" -ForegroundColor Green
            } else {
                # Get existing drive letter
                $partition = Get-Partition -DiskNumber $diskNumber | Where-Object { $_.Type -eq 'Basic' }
                if ($partition) {
                    $driveLetter = $partition.DriveLetter
                    Write-Host "✓ Existing partition found at drive $driveLetter`:" -ForegroundColor Green
                } else {
                    Write-Warning 'No basic partition found on the disk'
                }
            }
        } catch {
            Write-Error "Failed to setup VHDX: $_"

            # Cleanup on failure
            try {
                if ($diskNumber) {
                    Write-Host 'Attempting cleanup...' -ForegroundColor Yellow
                    Dismount-VHD -Path $VhdxPath -Confirm:$false
                }
            } catch {
                Write-Warning 'Could not dismount VHDX during cleanup'
            }
            exit 1
        }
    }
}

# Create initial directory structure for WSL development
if ($driveLetter) {
    Write-Host "`nCreating WSL development directory structure..." -ForegroundColor Yellow
    $basePath = "${driveLetter}:\"

    $directories = @(
        'cross-platform',
        'docker-shared',
        'multi-wsl',
        'sync',
        'backup',
        'unison-profiles'
    )

    foreach ($dir in $directories) {
        $fullPath = Join-Path $basePath $dir
        if (-not (Test-Path $fullPath)) {
            New-Item -Path $fullPath -ItemType Directory -Force | Out-Null
            Write-Host "  ✓ Created: $fullPath" -ForegroundColor Cyan
        }
    }

    # Create optimized Unison profiles for cross-platform development
    $unisonProfilePath = Join-Path $basePath 'unison-profiles\cross-platform-template.prf'
    $unisonProfileContent = @"
# Unison profile template for cross-platform development
# Copy and customize for specific projects

# Example configuration:
# root = /mnt/wsl/shared/cross-platform/web-audio-agents
# root = ${basePath}sync\web-audio-agents

# Performance optimizations
fastcheck = true          # Quick change detection
maxthreads = 4           # Parallel sync operations
fat = false              # Precise file comparison
times = true             # Use modification times
perms = 0                # Don't sync permissions (WSL/Windows compatibility)

# Sync behavior
prefer = newer           # Auto-resolve: newer file wins
auto = true             # Automatic conflict resolution
batch = true            # Non-interactive mode
silent = false          # Show sync operations
confirmbigdel = false   # Don't confirm large deletions
confirmmerge = false    # Don't confirm merges

# Cross-platform ignore patterns
ignore = Name .git
ignore = Name .gitignore
ignore = Name node_modules
ignore = Name .next
ignore = Name dist
ignore = Name build
ignore = Name target
ignore = Name *.log
ignore = Name .env.local
ignore = Name .env
ignore = Name .vscode
ignore = Name __pycache__
ignore = Name *.pyc
ignore = Name .DS_Store
ignore = Name thumbs.db
ignore = Name desktop.ini
ignore = Name .tmp
ignore = Name *.tmp
ignore = Name .cache
ignore = Name npm-debug.log*
ignore = Name yarn-debug.log*
ignore = Name yarn-error.log*

# Development tool ignores
ignore = Name coverage
ignore = Name .nyc_output
ignore = Name .pytest_cache
ignore = Name .coverage
ignore = Name htmlcov
ignore = Name .tox
ignore = Name .nox
ignore = Name .venv
ignore = Name venv
ignore = Name ENV
ignore = Name env.bak
ignore = Name venv.bak

# IDE and editor ignores
ignore = Name .idea
ignore = Name .vscode/settings.json
ignore = Name *.swp
ignore = Name *.swo
ignore = Name *~
ignore = Name .#*

# OS-specific ignores
ignore = Name .fseventsd
ignore = Name .Spotlight-V100
ignore = Name .TemporaryItems
ignore = Name .Trashes
ignore = Name .vol
ignore = Name .com.apple.timemachine.donotpresent
ignore = Name .AppleDB
ignore = Name .AppleDesktop
ignore = Name Network Trash Folder
ignore = Name Temporary Items
ignore = Name .apdisk
"@

    Set-Content -Path $unisonProfilePath -Value $unisonProfileContent -Encoding UTF8
    Write-Host "  ✓ Created optimized Unison profile template: $unisonProfilePath" -ForegroundColor Cyan

    # Create a web development specific profile
    $webdevProfilePath = Join-Path $basePath 'unison-profiles\web-development.prf'
    $webdevProfileContent = @"
# Unison profile for web development projects
# Optimized for JavaScript/TypeScript/React/Next.js projects

# CONFIGURE THESE PATHS FOR YOUR PROJECT:
# root = /mnt/wsl/shared/cross-platform/your-web-project
# root = ${basePath}sync\your-web-project

# Web development optimizations
fastcheck = true
maxthreads = 6           # More threads for web projects
fat = false
times = true
perms = 0

# Auto-resolution
prefer = newer
auto = true
batch = true
silent = false

# Web-specific ignores (more aggressive)
ignore = Name node_modules
ignore = Name .next
ignore = Name .nuxt
ignore = Name dist
ignore = Name build
ignore = Name .output
ignore = Name .vercel
ignore = Name .netlify
ignore = Name .cache
ignore = Name .parcel-cache
ignore = Name .vite
ignore = Name coverage
ignore = Name .nyc_output
ignore = Name .storybook-out

# Package manager artifacts
ignore = Name package-lock.json
ignore = Name yarn.lock
ignore = Name pnpm-lock.yaml
ignore = Name npm-debug.log*
ignore = Name yarn-debug.log*
ignore = Name yarn-error.log*

# Environment and config
ignore = Name .env
ignore = Name .env.local
ignore = Name .env.development.local
ignore = Name .env.test.local
ignore = Name .env.production.local

# Version control
ignore = Name .git
ignore = Name .gitignore
ignore = Name .gitattributes

# IDEs and editors
ignore = Name .vscode/settings.json
ignore = Name .idea
ignore = Name *.swp
ignore = Name *.swo
ignore = Name *~

# OS files
ignore = Name .DS_Store
ignore = Name thumbs.db
ignore = Name desktop.ini
"@

    Set-Content -Path $webdevProfilePath -Value $webdevProfileContent -Encoding UTF8
    Write-Host "  ✓ Created web development Unison profile: $webdevProfilePath" -ForegroundColor Cyan

    # Create Unison management scripts
    $unisonScriptPath = Join-Path $basePath 'unison-profiles\manage-sync.sh'
    $unisonScriptContent = @'
#!/bin/bash
# Unison Sync Management Script
# Usage: ./manage-sync.sh <profile> <action>

PROFILE="$1"
ACTION="$2"

if [ -z "$PROFILE" ] || [ -z "$ACTION" ]; then
    echo "Usage: $0 <profile> <start|stop|status|sync>"
    echo ""
    echo "Available profiles:"
    ls ~/.unison/*.prf 2>/dev/null | sed 's/.*\///g; s/\.prf//g' | sed 's/^/  - /'
    exit 1
fi

PIDFILE="/tmp/unison-$PROFILE.pid"
LOGFILE="$HOME/.unison/$PROFILE.log"

case "$ACTION" in
    start)
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "Unison sync for $PROFILE is already running (PID: $(cat "$PIDFILE"))"
            exit 1
        fi

        echo "Starting Unison sync for profile: $PROFILE"
        nohup unison "$PROFILE" -repeat watch -logfile "$LOGFILE" > /dev/null 2>&1 &
        echo $! > "$PIDFILE"
        echo "Started with PID: $(cat "$PIDFILE")"
        echo "Log file: $LOGFILE"
        ;;

    stop)
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            if kill -0 "$PID" 2>/dev/null; then
                kill "$PID"
                rm -f "$PIDFILE"
                echo "Stopped Unison sync for $PROFILE (PID: $PID)"
            else
                echo "Process $PID not found, cleaning up PID file"
                rm -f "$PIDFILE"
            fi
        else
            echo "No running Unison sync found for $PROFILE"
        fi
        ;;

    status)
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "Unison sync for $PROFILE is running (PID: $(cat "$PIDFILE"))"
            echo "Log file: $LOGFILE"
            if [ -f "$LOGFILE" ]; then
                echo "Last 5 log entries:"
                tail -5 "$LOGFILE"
            fi
        else
            echo "Unison sync for $PROFILE is not running"
            [ -f "$PIDFILE" ] && rm -f "$PIDFILE"
        fi
        ;;

    sync)
        echo "Performing one-time sync for profile: $PROFILE"
        unison "$PROFILE"
        ;;

    *)
        echo "Invalid action: $ACTION"
        echo "Available actions: start, stop, status, sync"
        exit 1
        ;;
esac
'@

    Set-Content -Path $unisonScriptPath -Value $unisonScriptContent -Encoding UTF8
    Write-Host "  ✓ Created Unison management script: $unisonScriptPath" -ForegroundColor Cyan

    Set-Content -Path $unisonProfilePath -Value $unisonProfileContent -Encoding UTF8
    Write-Host "  ✓ Created Unison profile template: $unisonProfilePath" -ForegroundColor Cyan
}

# Generate WSL mount commands
Write-Host "`n" + '='*60 -ForegroundColor Green
Write-Host 'SETUP COMPLETE!' -ForegroundColor Green
Write-Host '='*60 -ForegroundColor Green

Write-Host "`nVHDX Information:" -ForegroundColor Yellow
Write-Host "  Path: $VhdxPath"
Write-Host "  Drive Letter: $driveLetter`:"
Write-Host "  Label: $Label"
Write-Host "  Size: $Size"

if ($hasOptimizedFunction -and $UseOptimizedFunction) {
    Write-Host '  Auto-mount: ✓ Configured (via scheduled task)' -ForegroundColor Green
    Write-Host '  Optimization: ✓ Fixed VHDX for best performance' -ForegroundColor Green
} else {
    Write-Host '  Auto-mount: ⚠ Not configured (manual mount required)' -ForegroundColor Yellow
}

Write-Host "`nTo mount in WSL, run these commands:" -ForegroundColor Yellow
Write-Host 'sudo mkdir -p /mnt/wsl/shared' -ForegroundColor Cyan
Write-Host "sudo mount -t drvfs $driveLetter`: /mnt/wsl/shared -o uid=1000,gid=1000" -ForegroundColor Cyan

Write-Host "`nTo make it persistent, add to /etc/fstab:" -ForegroundColor Yellow
Write-Host "$driveLetter`: /mnt/wsl/shared drvfs defaults,uid=1000,gid=1000 0 0" -ForegroundColor Cyan

Write-Host "`nDirectory structure created:" -ForegroundColor Yellow
foreach ($dir in $directories) {
    Write-Host "  /mnt/wsl/shared/$dir" -ForegroundColor Cyan
}

# Optional auto-mount in WSL
if ($AutoMount) {
    Write-Host "`nAttempting to mount in WSL..." -ForegroundColor Yellow
    try {
        # Check if WSL is available
        $wslDistros = wsl --list --quiet 2>$null
        if ($wslDistros) {
            $defaultDistro = (wsl --list --quiet)[0]
            if ($defaultDistro) {
                Write-Host "Mounting in WSL distro: $defaultDistro" -ForegroundColor Cyan

                wsl -d $defaultDistro sudo mkdir -p /mnt/wsl/shared
                wsl -d $defaultDistro sudo mount -t drvfs "$driveLetter`:" /mnt/wsl/shared -o uid=1000, gid=1000

                Write-Host '✓ Mounted in WSL at /mnt/wsl/shared' -ForegroundColor Green
            } else {
                Write-Warning 'No default WSL distro found'
            }
        } else {
            Write-Warning 'WSL not available or no distros installed'
        }
    } catch {
        Write-Warning "Could not auto-mount in WSL: $_"
        Write-Warning 'Please mount manually using the commands above'
    }
}

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host '1. Mount the VHDX in WSL using the commands above' -ForegroundColor White
Write-Host '2. Run the migration script: ~/wsl-performance-migration.sh' -ForegroundColor White
Write-Host '3. Setup Unison for bidirectional sync using the template profile' -ForegroundColor White
Write-Host '4. Test your development workflow with the new structure' -ForegroundColor White

if ($hasOptimizedFunction) {
    Write-Host "`nAdvanced Features Enabled:" -ForegroundColor Green
    Write-Host '✓ Fixed VHDX for optimal performance on ReFS' -ForegroundColor Green
    Write-Host '✓ Automatic mounting via scheduled task' -ForegroundColor Green
    Write-Host '✓ Persistent configuration across reboots' -ForegroundColor Green
}

Write-Host "`n" + '='*60 -ForegroundColor Green
Write-Host 'WSL Shared VHDX is ready for development!' -ForegroundColor Green
Write-Host '='*60 -ForegroundColor Green
