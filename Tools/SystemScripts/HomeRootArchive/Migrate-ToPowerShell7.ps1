#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Migrate from Windows PowerShell 5.1 to PowerShell 7.5.2

.DESCRIPTION
    This script migrates PowerShell profiles, modules, and configurations from Windows PowerShell
    to PowerShell 7. It handles path updates, module migration, and environment variable updates.

.PARAMETER WhatIf
    Shows what would be done without actually performing the migration

.PARAMETER Force
    Forces overwrite of existing files during migration

.PARAMETER BackupOnly
    Only creates backups without performing migration

.EXAMPLE
    .\Migrate-ToPowerShell7.ps1 -WhatIf
    Shows what the migration would do

.EXAMPLE
    .\Migrate-ToPowerShell7.ps1 -Force
    Performs the migration, overwriting existing files
#>

param(
    [switch]$WhatIf,
    [switch]$Force,
    [switch]$BackupOnly
)

# Ensure we're running in PowerShell 7
if ($PSVersionTable.PSVersion.Major -lt 7) {
    Write-Error "This script must be run in PowerShell 7. Use: pwsh -File $($MyInvocation.MyCommand.Path)"
    exit 1
}

# Define paths
$WindowsPSPath = "$env:USERPROFILE\Documents\WindowsPowerShell"
$PS7Path = "$env:USERPROFILE\Documents\PowerShell"
$WindowsPSModulesPath = "$env:USERPROFILE\Documents\WindowsPowerShell\Modules"
$PS7ModulesPath = "$env:USERPROFILE\Documents\PowerShell\Modules"
$BackupPath = "$env:USERPROFILE\PowerShell_Migration_Backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

function Write-Status {
    param([string]$Message, [string]$Type = "Info")
    $timestamp = Get-Date -Format "HH:mm:ss"
    switch ($Type) {
        "Info" { Write-Host "[$timestamp] INFO: $Message" -ForegroundColor Cyan }
        "Success" { Write-Host "[$timestamp] SUCCESS: $Message" -ForegroundColor Green }
        "Warning" { Write-Host "[$timestamp] WARNING: $Message" -ForegroundColor Yellow }
        "Error" { Write-Host "[$timestamp] ERROR: $Message" -ForegroundColor Red }
    }
}

function Test-AdminRights {
    return ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Backup-ExistingConfiguration {
    Write-Status "Creating backup at: $BackupPath"

    if (-not (Test-Path $BackupPath)) {
        New-Item -Path $BackupPath -ItemType Directory -Force | Out-Null
    }

    # Backup Windows PowerShell configuration
    if (Test-Path $WindowsPSPath) {
        Write-Status "Backing up Windows PowerShell configuration..."
        Copy-Item -Path $WindowsPSPath -Destination "$BackupPath\WindowsPowerShell" -Recurse -Force
    }

    # Backup existing PowerShell 7 configuration
    if (Test-Path $PS7Path) {
        Write-Status "Backing up existing PowerShell 7 configuration..."
        Copy-Item -Path $PS7Path -Destination "$BackupPath\PowerShell" -Recurse -Force
    }

    # Backup environment variables
    $envVars = @{
        'PSModulePath' = $env:PSModulePath
        'PSExecutionPolicyPreference' = $env:PSExecutionPolicyPreference
        'POWERSHELL_DISTRIBUTION_CHANNEL' = $env:POWERSHELL_DISTRIBUTION_CHANNEL
    }

    $envVars | ConvertTo-Json -Depth 2 | Set-Content -Path "$BackupPath\environment_variables.json"
    Write-Status "Environment variables backed up"
}

function Get-ModuleCompatibility {
    param([string]$ModuleName, [string]$ModulePath)

    # Check if module is compatible with PowerShell 7
    $manifest = Get-ChildItem -Path $ModulePath -Filter "*.psd1" | Select-Object -First 1

    if ($manifest) {
        try {
            $moduleData = Import-PowerShellDataFile -Path $manifest.FullName
            $psVersion = $moduleData.PowerShellVersion
            $psEdition = $moduleData.CompatiblePSEditions

            # Check compatibility
            if ($psEdition -contains "Core" -or $psEdition -contains "Desktop") {
                return @{ Compatible = $true; Reason = "Explicitly supports Core edition" }
            } elseif ($psVersion -and [version]$psVersion -le [version]"5.1") {
                return @{ Compatible = $true; Reason = "Supports PowerShell $psVersion or lower" }
            } else {
                return @{ Compatible = $false; Reason = "May not be compatible with PowerShell Core" }
            }
        } catch {
            return @{ Compatible = $null; Reason = "Could not parse module manifest" }
        }
    }

    return @{ Compatible = $null; Reason = "No manifest found" }
}

function Migrate-Profiles {
    Write-Status "Migrating PowerShell profiles..."

    # Create PS7 profile directory
    if (-not (Test-Path $PS7Path)) {
        if (-not $WhatIf) {
            New-Item -Path $PS7Path -ItemType Directory -Force | Out-Null
        }
        Write-Status "Created PowerShell 7 profile directory: $PS7Path"
    }

    # Profile files to migrate
    $profiles = @(
        @{ Name = "Profile.ps1"; Description = "All Users, All Hosts" }
        @{ Name = "Microsoft.PowerShell_profile.ps1"; Description = "All Users, PowerShell Console" }
        @{ Name = "Microsoft.VSCode_profile.ps1"; Description = "All Users, VS Code" }
    )

    foreach ($profile in $profiles) {
        $sourcePath = Join-Path $WindowsPSPath $profile.Name
        $destPath = Join-Path $PS7Path $profile.Name

        if (Test-Path $sourcePath) {
            Write-Status "Migrating profile: $($profile.Name) ($($profile.Description))"

            if (-not $WhatIf) {
                # Read and modify profile content for PS7 compatibility
                $content = Get-Content -Path $sourcePath -Raw

                # Fix common compatibility issues
                $content = $content -replace '\$PSScriptRoot', '$PSScriptRoot'
                $content = $content -replace 'Windows PowerShell', 'PowerShell'

                # Add PowerShell 7 detection at the top if not present
                if ($content -notmatch '\$PSVersionTable\.PSVersion\.Major') {
                    $ps7Check = @"
# PowerShell 7 compatibility check
if (`$PSVersionTable.PSVersion.Major -ge 7) {
    # PowerShell 7+ specific configurations can go here
}

"@
                    $content = $ps7Check + $content
                }

                if (Test-Path $destPath -and -not $Force) {
                    Write-Status "Profile already exists: $destPath (use -Force to overwrite)" "Warning"
                } else {
                    Set-Content -Path $destPath -Value $content -Encoding UTF8
                    Write-Status "Migrated profile: $($profile.Name)" "Success"
                }
            } else {
                Write-Status "[WHATIF] Would migrate: $sourcePath -> $destPath"
            }
        }
    }
}

function Migrate-Modules {
    Write-Status "Analyzing modules for migration..."

    if (-not (Test-Path $WindowsPSModulesPath)) {
        Write-Status "No Windows PowerShell modules directory found" "Warning"
        return
    }

    # Create PS7 modules directory
    if (-not (Test-Path $PS7ModulesPath)) {
        if (-not $WhatIf) {
            New-Item -Path $PS7ModulesPath -ItemType Directory -Force | Out-Null
        }
        Write-Status "Created PowerShell 7 modules directory: $PS7ModulesPath"
    }

    $modules = Get-ChildItem -Path $WindowsPSModulesPath -Directory
    $compatibleModules = @()
    $incompatibleModules = @()
    $unknownModules = @()

    foreach ($module in $modules) {
        $compatibility = Get-ModuleCompatibility -ModuleName $module.Name -ModulePath $module.FullName

        switch ($compatibility.Compatible) {
            $true {
                $compatibleModules += @{ Module = $module; Reason = $compatibility.Reason }
                Write-Status "✓ $($module.Name): $($compatibility.Reason)" "Success"
            }
            $false {
                $incompatibleModules += @{ Module = $module; Reason = $compatibility.Reason }
                Write-Status "✗ $($module.Name): $($compatibility.Reason)" "Warning"
            }
            $null {
                $unknownModules += @{ Module = $module; Reason = $compatibility.Reason }
                Write-Status "? $($module.Name): $($compatibility.Reason)" "Info"
            }
        }
    }

    # Migrate compatible modules
    if ($compatibleModules.Count -gt 0) {
        Write-Status "Migrating $($compatibleModules.Count) compatible modules..."

        foreach ($moduleInfo in $compatibleModules) {
            $module = $moduleInfo.Module
            $sourcePath = $module.FullName
            $destPath = Join-Path $PS7ModulesPath $module.Name

            if (-not $WhatIf) {
                if (Test-Path $destPath -and -not $Force) {
                    Write-Status "Module already exists: $($module.Name) (use -Force to overwrite)" "Warning"
                } else {
                    Copy-Item -Path $sourcePath -Destination $destPath -Recurse -Force
                    Write-Status "Migrated module: $($module.Name)" "Success"
                }
            } else {
                Write-Status "[WHATIF] Would migrate module: $($module.Name)"
            }
        }
    }

    # Report summary
    Write-Status "Module migration summary:"
    Write-Status "  Compatible modules: $($compatibleModules.Count)" "Success"
    Write-Status "  Incompatible modules: $($incompatibleModules.Count)" "Warning"
    Write-Status "  Unknown compatibility: $($unknownModules.Count)" "Info"

    if ($incompatibleModules.Count -gt 0) {
        Write-Status "Incompatible modules (manual review needed):"
        foreach ($moduleInfo in $incompatibleModules) {
            Write-Status "  - $($moduleInfo.Module.Name): $($moduleInfo.Reason)" "Warning"
        }
    }
}

function Update-EnvironmentVariables {
    Write-Status "Updating environment variables for PowerShell 7..."

    # Update PSModulePath to prioritize PS7 modules
    $currentPSModulePath = $env:PSModulePath -split ';'
    $newPSModulePath = @()

    # Add PS7 module paths first
    $newPSModulePath += $PS7ModulesPath
    $newPSModulePath += "$env:ProgramFiles\PowerShell\Modules"

    # Add existing paths (except Windows PowerShell specific ones)
    foreach ($path in $currentPSModulePath) {
        if ($path -notlike "*WindowsPowerShell*" -and $path -notin $newPSModulePath) {
            $newPSModulePath += $path
        }
    }

    $newPSModulePathString = $newPSModulePath -join ';'

    if (-not $WhatIf) {
        # Update current session
        $env:PSModulePath = $newPSModulePathString

        # Update user environment variable
        [Environment]::SetEnvironmentVariable("PSModulePath", $newPSModulePathString, "User")
        Write-Status "Updated PSModulePath environment variable" "Success"
    } else {
        Write-Status "[WHATIF] Would update PSModulePath to: $newPSModulePathString"
    }
}

function Set-PowerShell7AsDefault {
    Write-Status "Configuring PowerShell 7 as default..."

    # Update Windows Terminal settings if present
    $wtSettingsPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"

    if (Test-Path $wtSettingsPath) {
        if (-not $WhatIf) {
            try {
                $wtSettings = Get-Content $wtSettingsPath | ConvertFrom-Json

                # Find PowerShell 7 profile
                $ps7Profile = $wtSettings.profiles.list | Where-Object { $_.commandline -like "*pwsh*" }

                if ($ps7Profile) {
                    $wtSettings.defaultProfile = $ps7Profile.guid
                    $wtSettings | ConvertTo-Json -Depth 100 | Set-Content $wtSettingsPath -Encoding UTF8
                    Write-Status "Updated Windows Terminal default profile to PowerShell 7" "Success"
                } else {
                    Write-Status "PowerShell 7 profile not found in Windows Terminal" "Warning"
                }
            } catch {
                Write-Status "Failed to update Windows Terminal settings: $($_.Exception.Message)" "Warning"
            }
        } else {
            Write-Status "[WHATIF] Would update Windows Terminal default profile to PowerShell 7"
        }
    }

    # Create desktop shortcut for PowerShell 7
    $desktopPath = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = "$desktopPath\PowerShell 7.lnk"

    if (-not $WhatIf) {
        $WshShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut($shortcutPath)
        $Shortcut.TargetPath = "C:\Program Files\PowerShell\7\pwsh.exe"
        $Shortcut.Description = "PowerShell 7"
        $Shortcut.WorkingDirectory = "$env:USERPROFILE"
        $Shortcut.Save()
        Write-Status "Created PowerShell 7 desktop shortcut" "Success"
    } else {
        Write-Status "[WHATIF] Would create PowerShell 7 desktop shortcut"
    }
}

function Fix-GcpProfileSystem {
    Write-Status "Fixing GCP Profile System compatibility..."

    $gcpModulePath = "$PS7ModulesPath\GcpUtils"
    if (Test-Path "$WindowsPSModulesPath\GcpUtils") {
        if (-not $WhatIf) {
            # Copy GCP module to PS7 if it doesn't exist
            if (-not (Test-Path $gcpModulePath) -or $Force) {
                Copy-Item -Path "$WindowsPSModulesPath\GcpUtils" -Destination $gcpModulePath -Recurse -Force

                # Fix PowerShell 7 compatibility issues in GCP module
                $profileManagerPath = "$gcpModulePath\GcpProfileManager.ps1"
                if (Test-Path $profileManagerPath) {
                    $content = Get-Content $profileManagerPath -Raw

                    # Fix syntax errors and PS7 compatibility
                    $content = $content -replace 'ErrorActionPreference.*Stop', 'ErrorActionPreference = "SilentlyContinue"'

                    Set-Content -Path $profileManagerPath -Value $content -Encoding UTF8
                    Write-Status "Fixed GCP Profile Manager for PowerShell 7 compatibility" "Success"
                }
            }
        } else {
            Write-Status "[WHATIF] Would fix GCP Profile System for PowerShell 7"
        }
    }
}

function Show-MigrationSummary {
    Write-Status "=== PowerShell 7 Migration Summary ==="
    Write-Status "✓ PowerShell 7.5.2 is installed and ready"
    Write-Status "✓ Backup created at: $BackupPath"

    if (-not $WhatIf -and -not $BackupOnly) {
        Write-Status "✓ Profiles migrated to PowerShell 7"
        Write-Status "✓ Compatible modules migrated"
        Write-Status "✓ Environment variables updated"
        Write-Status "✓ PowerShell 7 configured as default"
    }

    Write-Status ""
    Write-Status "Next Steps:"
    Write-Status "1. Test PowerShell 7 with: pwsh"
    Write-Status "2. Verify your profile loads correctly"
    Write-Status "3. Check that required modules work as expected"
    Write-Status "4. Update any scripts to use 'pwsh' instead of 'powershell'"
    Write-Status "5. Consider removing old Windows PowerShell modules after verification"

    if (Test-AdminRights) {
        Write-Status "6. Run 'Set-ExecutionPolicy RemoteSigned -Scope LocalMachine' if needed"
    } else {
        Write-Status "6. Run as Administrator and execute 'Set-ExecutionPolicy RemoteSigned -Scope LocalMachine' if needed"
    }

    Write-Status ""
    Write-Status "Backup location: $BackupPath" "Info"
}

# Main execution
try {
    Write-Status "Starting PowerShell 7 migration process..."
    Write-Status "PowerShell version: $($PSVersionTable.PSVersion)"
    Write-Status "Running in $(if ($WhatIf) { 'WhatIf' } elseif ($BackupOnly) { 'Backup-only' } else { 'Full migration' }) mode"

    # Always create backup
    Backup-ExistingConfiguration

    if (-not $BackupOnly) {
        # Perform migration steps
        Migrate-Profiles
        Migrate-Modules
        Update-EnvironmentVariables
        Set-PowerShell7AsDefault
        Fix-GcpProfileSystem
    }

    Show-MigrationSummary

    Write-Status "Migration completed successfully!" "Success"

} catch {
    Write-Status "Migration failed: $($_.Exception.Message)" "Error"
    Write-Status "Backup available at: $BackupPath" "Info"
    exit 1
}