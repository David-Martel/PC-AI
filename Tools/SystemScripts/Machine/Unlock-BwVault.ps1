<#
.SYNOPSIS
    Auto-unlocks the Bitwarden CLI vault using Windows Credential Manager.
.DESCRIPTION
    Reads the master password from CredMan target 'bitwarden/master_password',
    handles login if needed (via bitwarden/client_id + bitwarden/client_secret),
    unlocks the vault, and persists BW_SESSION as both process and user env var.

    Designed to run at login (scheduled task) or on-demand from profile.
.PARAMETER Quiet
    Suppresses informational output (for profile/scheduled task use).
.PARAMETER Force
    Re-unlocks even if BW_SESSION is already set and valid.
.PARAMETER PersistSession
    Persists BW_SESSION as a User environment variable (survives new terminals).
.EXAMPLE
    # From profile (quiet, persist)
    .\Unlock-BwVault.ps1 -Quiet -PersistSession

    # Interactive
    .\Unlock-BwVault.ps1

    # Force re-unlock
    .\Unlock-BwVault.ps1 -Force -PersistSession
#>
[CmdletBinding()]
param(
    [switch]$Quiet,
    [switch]$Force,
    [switch]$PersistSession
)

$ErrorActionPreference = 'Stop'

function Write-Status {
    param([string]$Message, [string]$Color = 'Gray')
    if (-not $Quiet) { Write-Host $Message -ForegroundColor $Color }
}

function Invoke-BwSyncIfDue {
    # Sync the BW vault from server if lastSync is missing or older than -MaxAgeHours.
    # Idempotent + cheap; safe to call before every successful return.
    # Requires BW_SESSION to be set (caller's responsibility).
    param(
        [Parameter(Mandatory)][string]$BwPath,
        [int]$MaxAgeHours = 6
    )
    try {
        $st = & $BwPath status 2>$null | ConvertFrom-Json -ErrorAction Stop
        $age = $null
        $needSync = $true
        if ($st.lastSync) {
            $age = (Get-Date) - [DateTime]::Parse($st.lastSync).ToLocalTime()
            $needSync = $age.TotalHours -gt $MaxAgeHours
        }
        if ($needSync) {
            $lastDesc = if ($st.lastSync) { $st.lastSync } else { 'never' }
            Write-Status "Syncing BW vault (last sync: $lastDesc)..." Cyan
            $null = & $BwPath sync 2>&1
            if ($LASTEXITCODE -eq 0) { Write-Status 'BW vault synced' Green }
            else                     { Write-Status "BW sync exit=$LASTEXITCODE (non-fatal; continuing)" Yellow }
        } else {
            Write-Status ("BW vault sync skipped (last sync {0:N1}h ago)" -f $age.TotalHours) Gray
        }
    } catch {
        Write-Status "BW sync check failed: $_ (non-fatal)" Yellow
    }
}

# -- Locate bw binary --
$bwCmd = Get-Command bw -ErrorAction SilentlyContinue
if (-not $bwCmd) {
    $fallbacks = @(
        "$env:USERPROFILE\bin\bw.exe",
        "$env:LOCALAPPDATA\Programs\Bitwarden CLI\bw.exe"
    )
    foreach ($fb in $fallbacks) {
        if (Test-Path $fb) { $bwCmd = Get-Item $fb; break }
    }
}
if (-not $bwCmd) {
    Write-Warning "Bitwarden CLI (bw) not found in PATH or fallback locations."
    return $false
}
$bw = if ($bwCmd -is [System.Management.Automation.ApplicationInfo]) { $bwCmd.Source } else { $bwCmd.FullName }

# -- Check current status --
$statusJson = & $bw status 2>$null | ConvertFrom-Json -ErrorAction SilentlyContinue
$vaultStatus = $statusJson.status  # unauthenticated, locked, unlocked

if ($vaultStatus -eq 'unlocked' -and -not $Force) {
    # Already unlocked - just ensure BW_SESSION is set
    if ($env:BW_SESSION) {
        Write-Status "BW vault already unlocked (session active)" Green
        Invoke-BwSyncIfDue -BwPath $bw
        return $true
    }
    # Unlocked but no session var - need to re-unlock to get session token
}

# -- Read master password from Windows Credential Manager --
function Get-CredManPassword {
    param([string]$Target)
    try {
        $cred = Get-StoredCredential -Target $Target -ErrorAction SilentlyContinue
        if ($cred) {
            return $cred.GetNetworkCredential().Password
        }
    } catch {}

    # Fallback: use cmdkey + .NET CredRead
    try {
        Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class CredRead {
    [DllImport("advapi32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
    public static extern bool CredReadW(string target, int type, int flags, out IntPtr cred);
    [DllImport("advapi32.dll")]
    public static extern void CredFree(IntPtr cred);

    [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
    public struct CREDENTIAL {
        public int Flags; public int Type; public string TargetName; public string Comment;
        public long LastWritten; public int CredentialBlobSize; public IntPtr CredentialBlob;
        public int Persist; public int AttributeCount; public IntPtr Attributes;
        public string TargetAlias; public string UserName;
    }

    public static string Read(string target) {
        IntPtr credPtr;
        if (!CredReadW(target, 1, 0, out credPtr)) return null;
        try {
            var cred = (CREDENTIAL)Marshal.PtrToStructure(credPtr, typeof(CREDENTIAL));
            if (cred.CredentialBlobSize > 0)
                return Marshal.PtrToStringUni(cred.CredentialBlob, cred.CredentialBlobSize / 2);
            return null;
        } finally { CredFree(credPtr); }
    }
}
"@ -ErrorAction SilentlyContinue
        return [CredRead]::Read($Target)
    } catch {
        return $null
    }
}

$masterPassword = Get-CredManPassword -Target 'bitwarden/master_password'
if (-not $masterPassword) {
    Write-Warning "No master password found in Credential Manager (target: bitwarden/master_password)."
    Write-Warning "Store it with: New-StoredCredential -Target 'bitwarden/master_password' -UserName 'bw' -Password '<master>' -Type Generic -Persist LocalMachine"
    return $false
}

# -- Handle unauthenticated state (login first) --
if ($vaultStatus -eq 'unauthenticated') {
    Write-Status "BW vault is unauthenticated, attempting API key login..." Yellow

    $clientId = Get-CredManPassword -Target 'bitwarden/client_id'
    $clientSecret = Get-CredManPassword -Target 'bitwarden/client_secret'

    if ($clientId -and $clientSecret) {
        $env:BW_CLIENTID = $clientId
        $env:BW_CLIENTSECRET = $clientSecret
        $loginOut = & $bw login --apikey 2>&1
        Remove-Item Env:\BW_CLIENTID -ErrorAction SilentlyContinue
        Remove-Item Env:\BW_CLIENTSECRET -ErrorAction SilentlyContinue

        if ($LASTEXITCODE -ne 0) {
            Write-Warning "BW API key login failed: $loginOut"
            return $false
        }
        Write-Status "BW login successful" Green
    } else {
        Write-Warning "BW is unauthenticated and no API key credentials found in Credential Manager."
        Write-Warning "Store them with:"
        Write-Warning "  New-StoredCredential -Target 'bitwarden/client_id' -UserName 'bw' -Password '<client_id>' -Type Generic -Persist LocalMachine"
        Write-Warning "  New-StoredCredential -Target 'bitwarden/client_secret' -UserName 'bw' -Password '<secret>' -Type Generic -Persist LocalMachine"
        return $false
    }
}

# -- Unlock vault --
Write-Status "Unlocking BW vault..." Cyan

# Pass master password via environment variable (never as CLI argument)
$env:_BW_MP = $masterPassword
$session = & $bw unlock --passwordenv _BW_MP --raw 2>&1
Remove-Item Env:\_BW_MP -ErrorAction SilentlyContinue

if ($LASTEXITCODE -ne 0 -or -not $session -or $session -match 'Invalid master password') {
    Write-Warning "BW unlock failed: $session"
    return $false
}

# -- Set BW_SESSION --
$env:BW_SESSION = $session

if ($PersistSession) {
    [Environment]::SetEnvironmentVariable('BW_SESSION', $session, 'User')
    Write-Status "BW_SESSION persisted as User environment variable" Green
}

Write-Status "BW vault unlocked successfully" Green
Invoke-BwSyncIfDue -BwPath $bw
return $true
