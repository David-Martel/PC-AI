#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Safe removal of 1,389 duplicate package.json files following PRIME DIRECTIVE
#>

param(
    [switch]$DryRun = $false,
    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

# PRIME DIRECTIVE: Define canonical package.json files that MUST be preserved
$CanonicalFiles = @(
    "C:\Users\david\.claude\package.json",
    "C:\Users\david\.claude\hooks\package.json",
    "C:\Users\david\.claude\windows\package.json"
)

function Write-Status {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [$Level] $Message"
}

Write-Status "=== DUPLICATE PACKAGE.JSON CLEANUP SCRIPT ===" "SUCCESS"
Write-Status "Following PRIME DIRECTIVE: Zero tolerance for duplicates" "SUCCESS"

if ($DryRun) {
    Write-Status "DRY RUN MODE: No files will be modified" "WARN"
}

# Verify canonical files exist
foreach ($file in $CanonicalFiles) {
    if (Test-Path $file) {
        Write-Status "Found canonical file: $file" "SUCCESS"
    } else {
        Write-Status "Missing canonical file: $file" "ERROR"
        exit 1
    }
}

# Find all package.json files
$allFiles = Get-ChildItem -Path "C:\Users\david\.claude" -Name "package.json" -Recurse -File
Write-Status "Found $($allFiles.Count) total package.json files" "INFO"

# Separate canonical from duplicates
$duplicates = @()
foreach ($file in $allFiles) {
    if ($file.FullName -notin $CanonicalFiles) {
        if ($file.FullName -match "(node_modules|compiled-servers|dist|build|\.cache|temp|tmp)") {
            $duplicates += $file.FullName
        }
    }
}

Write-Status "Duplicate files to remove: $($duplicates.Count)" "INFO"

# Create backup list
$backupFile = "C:\Users\david\.claude\duplicate-cleanup-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"
if (!$DryRun) {
    $duplicates | Out-File -FilePath $backupFile -Encoding UTF8
    Write-Status "Backup list created: $backupFile" "SUCCESS"
}

# Remove duplicates
$successCount = 0
foreach ($file in $duplicates) {
    if ($DryRun) {
        Write-Status "DRY RUN: Would remove: $file" "WARN"
        $successCount++
    } else {
        try {
            Remove-Item -Path $file -Force
            $successCount++
            if ($successCount % 100 -eq 0) {
                Write-Status "Removed $successCount files..." "INFO"
            }
        }
        catch {
            Write-Status "Failed to remove: $file" "ERROR"
        }
    }
}

Write-Status "Operation completed. Files processed: $successCount" "SUCCESS"
