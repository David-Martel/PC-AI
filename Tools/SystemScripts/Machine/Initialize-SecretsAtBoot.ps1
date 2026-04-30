#Requires -Version 5.1
<#
.SYNOPSIS
    Initializes machine secrets at Windows boot/login
.DESCRIPTION
    Runs at Windows login via Task Scheduler to:
    1. Load BWS access token from User environment
    2. Sync secrets from BWS Cloud (or use cache if offline)
    3. Set secrets as persistent User environment variables
    4. Broadcast environment change to running applications

    CRITICAL: This script must NEVER fail catastrophically.
    All errors are logged and the script continues gracefully.
.NOTES
    Machine: dtm-p1gen7
    Trigger: Task Scheduler at user logon
    Run As: Current user (required for DPAPI)
#>

param(
    [switch]$Silent,
    [switch]$ForceSync
)

#region Configuration
$script:Config = @{
    MachineDir    = "$env:USERPROFILE\.machine"
    ModulePath    = "$env:USERPROFILE\.machine\SecretsTier.psm1"
    LogDir        = "$env:USERPROFILE\.machine\logs"
    LogFile       = "$env:USERPROFILE\.machine\logs\boot-init.log"
    MaxLogSizeKB  = 500

    # Secrets to set as persistent User environment variables
    # These will be available to ALL applications without profile load
    PersistentSecrets = @(
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "VERTEX_AI_API_KEY",
        "GOOGLE_API_KEY",
        "CLOUDFLARE_API_KEY",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_ZONE_ID",
        "HF_TOKEN",
        "DOCKER_PAT",
        "GITHUB_PERSONAL_ACCESS_TOKEN"
    )
}
#endregion

#region Logging
function Initialize-Logging {
    if (-not (Test-Path $script:Config.LogDir)) {
        New-Item -ItemType Directory -Path $script:Config.LogDir -Force | Out-Null
    }

    # Rotate log if too large
    if (Test-Path $script:Config.LogFile) {
        $logSize = (Get-Item $script:Config.LogFile).Length / 1KB
        if ($logSize -gt $script:Config.MaxLogSizeKB) {
            $backupPath = $script:Config.LogFile -replace '\.log$', "-$(Get-Date -Format 'yyyyMMdd').log"
            Move-Item $script:Config.LogFile $backupPath -Force
        }
    }
}

function Write-Log {
    param(
        [string]$Message,
        [ValidateSet('INFO', 'WARN', 'ERROR', 'OK')]
        [string]$Level = 'INFO'
    )

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $entry = "[$timestamp] [$Level] $Message"

    # Always log to file
    Add-Content -Path $script:Config.LogFile -Value $entry -ErrorAction SilentlyContinue

    # Console output unless silent
    if (-not $Silent) {
        $color = switch ($Level) {
            'OK'    { 'Green' }
            'WARN'  { 'Yellow' }
            'ERROR' { 'Red' }
            default { 'White' }
        }
        Write-Host $entry -ForegroundColor $color
    }
}
#endregion

#region Core Functions
function Import-SecretsModule {
    if (-not (Test-Path $script:Config.ModulePath)) {
        Write-Log "SecretsTier module not found" "ERROR"
        return $false
    }

    try {
        Import-Module $script:Config.ModulePath -Force -DisableNameChecking -ErrorAction Stop
        return $true
    } catch {
        Write-Log "Failed to import module: $_" "ERROR"
        return $false
    }
}

function Get-BwsToken {
    $token = $null
    if (Get-Command Resolve-BwsTokenFromBackends -ErrorAction SilentlyContinue) {
        $token = Resolve-BwsTokenFromBackends -Quiet
    }

    if ($token) {
        $env:BWS_ACCESS_TOKEN = $token
        return $true
    }

    Write-Log "BWS_ACCESS_TOKEN not found in User environment" "WARN"
    return $false
}

function Sync-FromBwsCloud {
    <#
    .SYNOPSIS
    Attempts to sync secrets from BWS Cloud
    .OUTPUTS
    Hashtable of secrets, or $null on failure
    #>

    try {
        $secrets = Get-BwsSecretsList

        if ($secrets -and $secrets.Count -gt 0) {
            # Update local cache
            Save-SecretsToCache -Secrets $secrets
            Write-Log "Synced $($secrets.Count) secrets from BWS Cloud" "OK"
            return $secrets
        }

        Write-Log "BWS returned no secrets" "WARN"
        return $null
    } catch {
        Write-Log "BWS sync failed: $_" "WARN"
        return $null
    }
}

function Get-FromLocalCache {
    <#
    .SYNOPSIS
    Loads secrets from local DPAPI-encrypted cache
    .OUTPUTS
    Hashtable of secrets, or $null on failure
    #>

    try {
        $cacheStatus = Get-CacheStatus

        if (-not $cacheStatus.Exists) {
            Write-Log "No local cache exists" "WARN"
            return $null
        }

        if ($cacheStatus.IsExpired) {
            Write-Log "Cache is $([int]$cacheStatus.Age.TotalDays) days old (expired)" "WARN"
        }

        # Use cache even if expired (better than nothing)
        $secrets = Get-SecretsFromCache -IgnoreAge

        if ($secrets -and $secrets.Count -gt 0) {
            $ageStr = if ($cacheStatus.Age.TotalHours -lt 24) {
                "$([int]$cacheStatus.Age.TotalHours)h"
            } else {
                "$([int]$cacheStatus.Age.TotalDays)d"
            }
            Write-Log "Loaded $($secrets.Count) secrets from cache ($ageStr old)" "OK"
            return $secrets
        }

        Write-Log "Cache exists but couldn't be read" "ERROR"
        return $null
    } catch {
        Write-Log "Cache read error: $_" "ERROR"
        return $null
    }
}

function Set-PersistentEnvironment {
    <#
    .SYNOPSIS
    Sets specified secrets as persistent User environment variables
    .PARAMETER Secrets
    Hashtable of all secrets
    .OUTPUTS
    Number of variables updated
    #>
    param([hashtable]$Secrets)

    $updated = 0
    $skipped = 0

    foreach ($key in $script:Config.PersistentSecrets) {
        if ($Secrets.ContainsKey($key)) {
            $currentValue = [Environment]::GetEnvironmentVariable($key, 'User')

            # Only update if different (or force sync)
            if ($currentValue -ne $Secrets[$key] -or $ForceSync) {
                try {
                    [Environment]::SetEnvironmentVariable($key, $Secrets[$key], 'User')
                    $updated++
                } catch {
                    Write-Log "Failed to set $key : $_" "ERROR"
                }
            } else {
                $skipped++
            }
        }
    }

    if ($updated -gt 0) {
        Write-Log "Updated $updated persistent env vars ($skipped unchanged)" "OK"
    } else {
        Write-Log "All $skipped env vars already current" "INFO"
    }

    return $updated
}

function Broadcast-EnvironmentChange {
    <#
    .SYNOPSIS
    Broadcasts WM_SETTINGCHANGE to notify running applications
    #>

    try {
        # Only add type if not already defined
        if (-not ([System.Management.Automation.PSTypeName]'BootNativeMethods').Type) {
            Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class BootNativeMethods {
    [DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    public static extern IntPtr SendMessageTimeout(
        IntPtr hWnd, uint Msg, UIntPtr wParam, string lParam,
        uint fuFlags, uint uTimeout, out UIntPtr lpdwResult);
    public static readonly IntPtr HWND_BROADCAST = new IntPtr(0xffff);
    public const uint WM_SETTINGCHANGE = 0x001A;
    public const uint SMTO_ABORTIFHUNG = 0x0002;
}
"@
        }

        $result = [UIntPtr]::Zero
        [BootNativeMethods]::SendMessageTimeout(
            [BootNativeMethods]::HWND_BROADCAST,
            [BootNativeMethods]::WM_SETTINGCHANGE,
            [UIntPtr]::Zero,
            "Environment",
            [BootNativeMethods]::SMTO_ABORTIFHUNG,
            5000,
            [ref]$result
        ) | Out-Null

        Write-Log "Environment broadcast sent" "INFO"
    } catch {
        Write-Log "Broadcast failed (non-critical): $_" "WARN"
    }
}

function Set-ProcessEnvironment {
    <#
    .SYNOPSIS
    Sets all secrets as Process environment variables for current session
    #>
    param([hashtable]$Secrets)

    foreach ($key in $Secrets.Keys) {
        [Environment]::SetEnvironmentVariable($key, $Secrets[$key], 'Process')
    }
}
#endregion

#region Main Execution
function Main {
    $startTime = Get-Date

    Initialize-Logging
    Write-Log "=== Boot initialization starting ===" "INFO"
    Write-Log "Machine: $env:COMPUTERNAME, User: $env:USERNAME" "INFO"

    # Step 1: Import module
    if (-not (Import-SecretsModule)) {
        Write-Log "Cannot continue without module - exiting" "ERROR"
        return 1
    }

    # Step 2: Load BWS token
    $hasToken = Get-BwsToken

    # Step 3: Get secrets (tiered fallback)
    $secrets = $null
    $source = "NONE"

    if ($hasToken) {
        # Try BWS Cloud first
        $secrets = Sync-FromBwsCloud
        if ($secrets) {
            $source = "BWS Cloud"
        }
    }

    if (-not $secrets) {
        # Fallback to local cache
        $secrets = Get-FromLocalCache
        if ($secrets) {
            $source = "Local Cache"
        }
    }

    if (-not $secrets) {
        Write-Log "No secrets available from any source!" "ERROR"
        Write-Log "Run 'secrets sync' when online to initialize" "WARN"
        return 1
    }

    # Step 4: Set persistent environment variables
    $updatedCount = Set-PersistentEnvironment -Secrets $secrets

    # Step 5: Broadcast if we made changes
    if ($updatedCount -gt 0) {
        Broadcast-EnvironmentChange
    }

    # Step 6: Set process environment for this session
    Set-ProcessEnvironment -Secrets $secrets

    # Summary
    $elapsed = ((Get-Date) - $startTime).TotalSeconds
    Write-Log "=== Initialization complete ===" "OK"
    Write-Log "Source: $source, Secrets: $($secrets.Count), Updated: $updatedCount, Time: $($elapsed.ToString('F1'))s" "INFO"

    return 0
}

# Execute and return exit code
$exitCode = Main
exit $exitCode
#endregion
