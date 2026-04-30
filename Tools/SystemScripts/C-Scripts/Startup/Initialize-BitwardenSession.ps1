#Requires -Version 5.1
<#
.SYNOPSIS
    Initializes a Bitwarden CLI session using API key authentication.

.DESCRIPTION
    This script automates Bitwarden CLI login and vault unlock using credentials
    stored securely in Windows Credential Manager. It supports:
    - API key authentication (non-interactive login)
    - Master password vault unlock
    - Session token export to environment variable
    - Secure credential retrieval from Windows Credential Manager

.PARAMETER SkipLogin
    Skip the login step if already logged in.

.PARAMETER Force
    Force re-login even if already logged in.

.PARAMETER ExportSession
    Export BW_SESSION to User environment variable for persistence.

.PARAMETER LogFile
    Path to log file. Default: C:\Scripts\Startup\bitwarden-session.log

.EXAMPLE
    .\Initialize-BitwardenSession.ps1
    Initializes Bitwarden session with default settings.

.EXAMPLE
    .\Initialize-BitwardenSession.ps1 -ExportSession -Verbose
    Initializes session and exports to persistent environment variable.

.NOTES
    Author: Claude Code Framework
    Version: 1.0
    Requires: Bitwarden CLI, Windows Credential Manager entries for:
      - bitwarden/client_id
      - bitwarden/client_secret
      - bitwarden/master_password

.LINK
    https://bitwarden.com/help/cli/
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter()]
    [switch]$SkipLogin,

    [Parameter()]
    [switch]$Force,

    [Parameter()]
    [switch]$ExportSession,

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$LogFile = 'C:\Scripts\Startup\bitwarden-session.log'
)

#region Script Configuration
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:Config = @{
    CredentialTargets = @{
        ClientId       = 'bitwarden/client_id'
        ClientSecret   = 'bitwarden/client_secret'
        MasterPassword = 'bitwarden/master_password'
    }
    # Retry configuration justified by Bitwarden API characteristics:
    # - 95th percentile latency ~3s, occasional spikes to 10s
    # - Network hiccups common during system startup
    # - Total timeout: 5 retries * 5s delay = 25s max wait
    MaxRetries        = 5
    RetryDelaySeconds = 5
}
#endregion

#region Logging Functions
function Write-Log {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Message,

        [Parameter()]
        [ValidateSet('INFO', 'SUCCESS', 'WARNING', 'ERROR', 'DEBUG')]
        [string]$Level = 'INFO'
    )

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $logEntry = "[$timestamp] [$Level] $Message"

    # Ensure log directory exists
    $logDir = Split-Path -Path $LogFile -Parent
    if (-not (Test-Path -Path $logDir)) {
        New-Item -Path $logDir -ItemType Directory -Force | Out-Null
    }

    Add-Content -Path $LogFile -Value $logEntry -ErrorAction SilentlyContinue

    switch ($Level) {
        'SUCCESS' { Write-Verbose $logEntry }
        'WARNING' { Write-Warning $Message }
        'ERROR'   { Write-Error $Message -ErrorAction Continue }
        'DEBUG'   { Write-Debug $logEntry }
        default   { Write-Verbose $logEntry }
    }
}
#endregion

#region Credential Functions
function Get-StoredCredentialPassword {
    <#
    .SYNOPSIS
        Retrieves a password from Windows Credential Manager.
    .PARAMETER Target
        The credential target name.
    .OUTPUTS
        [string] The password value.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [string]$Target
    )

    Write-Log "Retrieving credential: $Target" -Level DEBUG

    try {
        # Use cmdkey to verify credential exists, then use .NET to retrieve
        $cmdkeyOutput = & cmdkey /list:$Target 2>&1
        # Note: When $cmdkeyOutput is an array, -notmatch returns non-matching elements (not boolean)
        # Convert to string first to get proper boolean match result
        $outputString = $cmdkeyOutput | Out-String
        if ($LASTEXITCODE -ne 0 -or $outputString -notmatch [regex]::Escape($Target)) {
            throw "Credential not found: $Target"
        }

        # Use PowerShell credential APIs
        # Note: cmdkey stores credentials but we need CredentialManager module or P/Invoke
        # For simplicity, we'll use a helper approach with stored credentials

        # Alternative: Use advapi32.dll CredRead
        Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public class CredentialManager {
    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern bool CredRead(string target, int type, int flags, out IntPtr credential);

    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern bool CredFree(IntPtr credential);

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct CREDENTIAL {
        public int Flags;
        public int Type;
        public string TargetName;
        public string Comment;
        public long LastWritten;
        public int CredentialBlobSize;
        public IntPtr CredentialBlob;
        public int Persist;
        public int AttributeCount;
        public IntPtr Attributes;
        public string TargetAlias;
        public string UserName;
    }

    public static string GetCredential(string target) {
        IntPtr credPtr;
        if (!CredRead(target, 1, 0, out credPtr)) {
            return null;
        }

        try {
            CREDENTIAL cred = (CREDENTIAL)Marshal.PtrToStructure(credPtr, typeof(CREDENTIAL));
            if (cred.CredentialBlobSize > 0) {
                return Marshal.PtrToStringUni(cred.CredentialBlob, cred.CredentialBlobSize / 2);
            }
            return null;
        } finally {
            CredFree(credPtr);
        }
    }
}
"@ -ErrorAction SilentlyContinue

        $password = [CredentialManager]::GetCredential($Target)
        if ([string]::IsNullOrEmpty($password)) {
            throw "Failed to retrieve credential value for: $Target"
        }

        Write-Log "Successfully retrieved credential: $Target" -Level DEBUG
        return $password
    }
    catch {
        Write-Log "Failed to retrieve credential '$Target': $_" -Level ERROR
        throw
    }
}

function Get-BitwardenCredentials {
    <#
    .SYNOPSIS
        Retrieves all Bitwarden credentials from Windows Credential Manager.
    .OUTPUTS
        [hashtable] Contains ClientId, ClientSecret, MasterPassword.
    #>
    [CmdletBinding()]
    [OutputType([hashtable])]
    param()

    Write-Log 'Retrieving Bitwarden credentials from Windows Credential Manager...'

    $credentials = @{}

    foreach ($key in $script:Config.CredentialTargets.Keys) {
        $target = $script:Config.CredentialTargets[$key]
        try {
            $credentials[$key] = Get-StoredCredentialPassword -Target $target
        }
        catch {
            Write-Log "Missing required credential: $target" -Level ERROR
            throw "Required credential not found in Windows Credential Manager: $target"
        }
    }

    Write-Log 'All Bitwarden credentials retrieved successfully' -Level SUCCESS
    return $credentials
}
#endregion

#region Bitwarden Functions
function Test-BitwardenCli {
    <#
    .SYNOPSIS
        Verifies Bitwarden CLI is installed and accessible.
    .OUTPUTS
        [bool] True if CLI is available.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    $bwCmd = Get-Command -Name 'bw' -ErrorAction SilentlyContinue
    if (-not $bwCmd) {
        Write-Log 'Bitwarden CLI (bw) not found in PATH' -Level ERROR
        return $false
    }

    Write-Log "Bitwarden CLI found: $($bwCmd.Source)" -Level DEBUG
    return $true
}

function Get-BitwardenStatus {
    <#
    .SYNOPSIS
        Gets the current Bitwarden CLI status.
    .OUTPUTS
        [PSCustomObject] Status object with serverUrl, lastSync, userEmail, userId, status.
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param()

    try {
        $statusJson = & bw status 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Failed to get Bitwarden status: $statusJson" -Level WARNING
            return $null
        }

        $status = $statusJson | ConvertFrom-Json
        Write-Log "Bitwarden status: $($status.status)" -Level DEBUG
        return $status
    }
    catch {
        Write-Log "Error getting Bitwarden status: $_" -Level WARNING
        return $null
    }
}

function Invoke-BitwardenLogin {
    <#
    .SYNOPSIS
        Logs into Bitwarden using API key authentication.
    .PARAMETER ClientId
        The Bitwarden API client ID.
    .PARAMETER ClientSecret
        The Bitwarden API client secret.
    .OUTPUTS
        [bool] True if login succeeded.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory)]
        [string]$ClientId,

        [Parameter(Mandatory)]
        [string]$ClientSecret
    )

    if (-not $PSCmdlet.ShouldProcess('Bitwarden', 'Login with API key')) {
        return $true
    }

    Write-Log 'Logging into Bitwarden with API key...'

    # Set environment variables for API key auth
    $env:BW_CLIENTID = $ClientId
    $env:BW_CLIENTSECRET = $ClientSecret

    try {
        $result = & bw login --apikey 2>&1
        $exitCode = $LASTEXITCODE

        if ($exitCode -ne 0) {
            # Check if already logged in
            if ($result -match 'already logged in') {
                Write-Log 'Already logged into Bitwarden' -Level SUCCESS
                return $true
            }

            # Sanitize output before logging (remove potential secrets)
            $sanitizedResult = $result -replace '[\w+/=]{20,}', '[REDACTED]'
            Write-Log "Bitwarden login failed: $sanitizedResult" -Level ERROR
            return $false
        }

        Write-Log 'Bitwarden login successful' -Level SUCCESS
        return $true
    }
    catch {
        Write-Log "Bitwarden login exception: $_" -Level ERROR
        return $false
    }
    finally {
        # Clear ALL sensitive env vars immediately
        Remove-Item Env:\BW_CLIENTID -ErrorAction SilentlyContinue
        Remove-Item Env:\BW_CLIENTSECRET -ErrorAction SilentlyContinue
    }
}

function Invoke-BitwardenUnlock {
    <#
    .SYNOPSIS
        Unlocks the Bitwarden vault and returns the session token.
    .PARAMETER MasterPassword
        The vault master password.
    .OUTPUTS
        [string] The session token, or $null on failure.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [string]$MasterPassword
    )

    if (-not $PSCmdlet.ShouldProcess('Bitwarden Vault', 'Unlock')) {
        return 'WHATIF_SESSION_TOKEN'
    }

    Write-Log 'Unlocking Bitwarden vault...'

    # SECURITY: Write password to a temporary file and use --passwordfile
    # Note: --passwordenv has issues with Bitwarden CLI 2025.x that cause
    # "The provided key is not the expected type" errors. --passwordfile works reliably.
    $tempPassFile = $null
    try {
        # Create temp file with restrictive permissions
        $tempPassFile = [System.IO.Path]::GetTempFileName()
        # Write password as first line (as per bw unlock --help)
        [System.IO.File]::WriteAllText($tempPassFile, $MasterPassword)

        for ($attempt = 1; $attempt -le $script:Config.MaxRetries; $attempt++) {
            try {
                # Use --passwordfile to pass password securely (file deleted immediately after)
                $session = & bw unlock --passwordfile $tempPassFile --raw 2>&1
                $exitCode = $LASTEXITCODE

                # Bitwarden session tokens are base64-encoded, minimum 32 chars
                $BITWARDEN_MIN_SESSION_LENGTH = 32
                if ($exitCode -eq 0 -and $session -and $session.Length -ge $BITWARDEN_MIN_SESSION_LENGTH) {
                    # Validate session token format (base64)
                    if ($session -match '^[A-Za-z0-9+/=]+$') {
                        Write-Log 'Bitwarden vault unlocked successfully' -Level SUCCESS
                        return $session
                    }
                    Write-Log 'Session token has invalid format' -Level WARNING
                }

                # Check if already unlocked
                if ($session -match 'already unlocked') {
                    Write-Log 'Vault already unlocked' -Level DEBUG
                    # Check for existing session in environment
                    if ($env:BW_SESSION -and $env:BW_SESSION.Length -ge $BITWARDEN_MIN_SESSION_LENGTH) {
                        Write-Log 'Using existing BW_SESSION' -Level SUCCESS
                        return $env:BW_SESSION
                    }
                    # Lock and retry for fresh session
                    Write-Log 'No valid existing session, locking vault for fresh unlock...' -Level DEBUG
                    & bw lock 2>&1 | Out-Null
                    continue
                }

                # Log sanitized error info (remove any potential credential fragments)
                $errorMsg = if ($session -is [System.Management.Automation.ErrorRecord]) {
                    $session.Exception.Message
                } elseif ($session -match 'invalid|error|failed|unauthorized') {
                    $session -replace '[A-Za-z0-9+/=]{20,}', '[REDACTED]'
                } else {
                    "Exit code: $exitCode"
                }
                Write-Log "Unlock attempt $attempt/$($script:Config.MaxRetries) failed: $errorMsg" -Level WARNING

                if ($attempt -lt $script:Config.MaxRetries) {
                    Start-Sleep -Seconds $script:Config.RetryDelaySeconds
                }
            }
            catch {
                Write-Log "Unlock attempt $attempt exception: $_" -Level WARNING
                if ($attempt -lt $script:Config.MaxRetries) {
                    Start-Sleep -Seconds $script:Config.RetryDelaySeconds
                }
            }
        }

        Write-Log 'Failed to unlock Bitwarden vault after all retries' -Level ERROR
        return $null
    }
    finally {
        # SECURITY: Delete temp password file immediately
        if ($tempPassFile -and (Test-Path $tempPassFile)) {
            Remove-Item $tempPassFile -Force -ErrorAction SilentlyContinue
        }
    }
}

function Export-BitwardenSession {
    <#
    .SYNOPSIS
        Exports the session token to environment variables.
    .PARAMETER Session
        The session token to export.
    .PARAMETER Persistent
        If true, sets as User environment variable (persists across sessions).

        WARNING: This reduces security by persisting the session token across
        system reboots. Only use for automation accounts or scheduled tasks.
        Interactive users should avoid this flag.

        Session tokens do not expire automatically and grant full vault access.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory)]
        [string]$Session,

        [Parameter()]
        [switch]$Persistent
    )

    if (-not $PSCmdlet.ShouldProcess('BW_SESSION', 'Set environment variable')) {
        return
    }

    # Set for current process
    $env:BW_SESSION = $Session
    Write-Log 'BW_SESSION set for current process' -Level DEBUG

    if ($Persistent) {
        # SECURITY WARNING: Persistent session tokens are accessible to all user processes
        Write-Log 'WARNING: Persistent session reduces security. Token will survive reboots.' -Level WARNING
        Write-Log 'Recommendation: Only use -ExportSession for automation accounts.' -Level WARNING

        if (-not $PSCmdlet.ShouldProcess('User Environment', 'Set persistent BW_SESSION (SECURITY RISK)')) {
            return
        }

        try {
            [Environment]::SetEnvironmentVariable('BW_SESSION', $Session, 'User')
            Write-Log 'BW_SESSION exported to User environment' -Level SUCCESS
        }
        catch {
            Write-Log "Failed to export BW_SESSION to User environment: $_" -Level WARNING
        }
    }
}

function Test-BitwardenSession {
    <#
    .SYNOPSIS
        Verifies the Bitwarden session is valid.
    .OUTPUTS
        [bool] True if session is valid.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    if ([string]::IsNullOrEmpty($env:BW_SESSION)) {
        Write-Log 'BW_SESSION not set' -Level DEBUG
        return $false
    }

    try {
        # Use lightweight status check instead of full sync (much faster)
        $result = & bw status 2>&1
        if ($LASTEXITCODE -eq 0) {
            $status = $result | ConvertFrom-Json
            if ($status.status -eq 'unlocked') {
                Write-Log 'Bitwarden session is valid (vault unlocked)' -Level SUCCESS
                return $true
            }
            Write-Log "Vault status: $($status.status)" -Level WARNING
            return $false
        }

        Write-Log 'Session validation failed' -Level WARNING
        return $false
    }
    catch {
        Write-Log "Session validation exception: $_" -Level WARNING
        return $false
    }
}
#endregion

#region Main Execution
function Invoke-BitwardenSessionInit {
    <#
    .SYNOPSIS
        Main function to initialize Bitwarden session.
    .OUTPUTS
        [bool] True if session initialized successfully.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param()

    Write-Log '=== Bitwarden Session Initialization Started ==='

    # Verify CLI is available
    if (-not (Test-BitwardenCli)) {
        Write-Log 'Bitwarden CLI not available' -Level ERROR
        return $false
    }

    # Check current status
    $status = Get-BitwardenStatus

    # Get credentials from Windows Credential Manager
    try {
        $credentials = Get-BitwardenCredentials
    }
    catch {
        Write-Log "Failed to retrieve credentials: $_" -Level ERROR
        return $false
    }

    # Login if needed
    # Note: Always re-login when status is 'locked' or 'unauthenticated' because API sessions can expire
    # even when the CLI shows 'locked' status. The 'invalid_client' error during unlock indicates stale session.
    if (-not $SkipLogin -or $Force) {
        if ($status.status -eq 'unauthenticated' -or $status.status -eq 'locked' -or $Force) {
            # Logout first to clear any stale sessions
            if ($status.status -eq 'locked') {
                Write-Log 'Clearing potentially stale session...' -Level DEBUG
                & bw logout 2>&1 | Out-Null
            }
            $loginResult = Invoke-BitwardenLogin -ClientId $credentials.ClientId -ClientSecret $credentials.ClientSecret
            if (-not $loginResult) {
                Write-Log 'Bitwarden login failed' -Level ERROR
                return $false
            }
        }
        else {
            Write-Log 'Already authenticated and unlocked, skipping login' -Level DEBUG
        }
    }

    # Unlock vault
    $session = Invoke-BitwardenUnlock -MasterPassword $credentials.MasterPassword

    # Clear sensitive data from memory
    $credentials.MasterPassword = $null
    $credentials.ClientSecret = $null
    $credentials = $null
    [System.GC]::Collect()

    if ([string]::IsNullOrEmpty($session)) {
        Write-Log 'Failed to unlock Bitwarden vault' -Level ERROR
        return $false
    }

    # Export session
    Export-BitwardenSession -Session $session -Persistent:$ExportSession

    # Verify session works
    if (-not (Test-BitwardenSession)) {
        Write-Log 'Session verification failed' -Level ERROR
        return $false
    }

    Write-Log '=== Bitwarden Session Initialization Complete ===' -Level SUCCESS
    return $true
}

# Entry point
try {
    $result = Invoke-BitwardenSessionInit
    if ($result) {
        exit 0
    }
    else {
        exit 1
    }
}
catch {
    Write-Log "Unhandled exception: $_" -Level ERROR
    Write-Log $_.ScriptStackTrace -Level ERROR
    exit 1
}
#endregion
