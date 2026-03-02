#Requires -Version 5.1
<#
.SYNOPSIS
    Optimizes .wslconfig for performance

.DESCRIPTION
    Creates or updates .wslconfig with optimized settings based on system resources.

.PARAMETER Memory
    Memory allocation for WSL (e.g., '8GB', '16GB'). Auto-calculated if not specified.

.PARAMETER Processors
    Number of CPU cores for WSL. Auto-calculated if not specified.

.PARAMETER SwapPath
    Path for swap file. Default: system determined.

.PARAMETER DryRun
    Show what would be changed without making changes

.PARAMETER Force
    Skip confirmation prompt

.EXAMPLE
    Optimize-WSLConfig
    Automatically optimizes based on system resources

.EXAMPLE
    Optimize-WSLConfig -Memory 16GB -Processors 8 -DryRun
    Preview optimization with specific settings

.OUTPUTS
    PSCustomObject with applied settings
#>
function Optimize-WSLConfig {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$Memory,

        [Parameter()]
        [int]$Processors,

        [Parameter()]
        [string]$SwapPath,

        [Parameter()]
        [switch]$DryRun,

        [Parameter()]
        [switch]$Force
    )

    $wslConfigPath = "$env:USERPROFILE\.wslconfig"

    # Calculate optimal settings based on system
    $os = Get-CimInstance Win32_OperatingSystem
    $totalMemoryGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 0)
    $cpuCount = [Environment]::ProcessorCount

    # Default to 25-30% of total memory for WSL
    $defaultMemoryGB = [math]::Max(4, [math]::Min([math]::Floor($totalMemoryGB * 0.30), 64))
    $defaultProcessors = [math]::Max(2, [math]::Min([math]::Floor($cpuCount * 0.6), $cpuCount - 2))

    $memoryValue = if ($Memory) { $Memory } else { "${defaultMemoryGB}GB" }
    $processorsValue = if ($Processors -gt 0) { $Processors } else { $defaultProcessors }

    function Normalize-WSLPath {
        param([string]$Path)
        if (-not $Path) { return $Path }
        return ($Path -replace '\\', '/')
    }

    $swapConfig = ''
    if ($SwapPath) {
        $swapNormalized = Normalize-WSLPath $SwapPath
        $swapConfig = @"

# Swap file location
swapFile=$swapNormalized
"@
    }

    $config = @"
# WSL Configuration - Optimized by PC-AI
# Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
# System: $totalMemoryGB GB RAM, $cpuCount cores

[wsl2]
# Memory allocation
memory=$memoryValue

# CPU allocation
processors=$processorsValue
$swapConfig

# Network configuration - NAT for broad compatibility
networkingMode=nat

# VM idle timeout (ms)
vmIdleTimeout=60000

# Enhanced networking features
dnsTunneling=true
firewall=true
autoProxy=true

# GUI applications support
guiApplications=true

# Nested virtualization for Docker
nestedVirtualization=true

[experimental]
# Sparse VHD for efficient storage
sparseVhd=true

# Memory reclaim settings
autoMemoryReclaim=gradual

# Page reporting for memory efficiency
pageReporting=true

# Network optimizations
hostAddressLoopback=true
bestEffortDnsParsing=true
"@

    $result = [PSCustomObject]@{
        Path         = $wslConfigPath
        Memory       = $memoryValue
        Processors   = $processorsValue
        BackupPath   = $null
        Applied      = $false
        Config       = $config
    }

    if ($DryRun) {
        Write-Host "DRY RUN - Would create/update: $wslConfigPath" -ForegroundColor Yellow
        Write-Host ""
        Write-Host $config -ForegroundColor Cyan
        return $result
    }

    if ($PSCmdlet.ShouldProcess($wslConfigPath, 'Update WSL configuration')) {
        # Create backup if file exists
        if (Test-Path $wslConfigPath) {
            $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
            $backupPath = "$wslConfigPath.backup_$timestamp"
            Copy-Item $wslConfigPath $backupPath
            $result.BackupPath = $backupPath
            Write-Host "[*] Backup created: $backupPath" -ForegroundColor Green
        }

        # Write new config
        [System.IO.File]::WriteAllText($wslConfigPath, $config, [System.Text.Encoding]::UTF8)
        $result.Applied = $true

        Write-Host "[*] .wslconfig updated successfully" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "  1. Run: wsl --shutdown" -ForegroundColor White
        Write-Host "  2. Wait 5 seconds" -ForegroundColor White
        Write-Host "  3. Restart WSL: wsl" -ForegroundColor White
    }

    return $result
}
