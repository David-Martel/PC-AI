#Requires -Version 5.1
<#
.SYNOPSIS
    Synchronizes container registry credentials from Bitwarden to Windows Credential Manager.

.DESCRIPTION
    This script retrieves registry credentials from Bitwarden Personal Vault and stores them
    in Windows Credential Manager using docker-credential-wincred. This enables both Docker
    and Podman to authenticate to registries using a shared credential store.

    Supported registries:
    - Docker Hub (https://index.docker.io/v1/)
    - GitHub Container Registry (https://ghcr.io)

.PARAMETER BitwardenItems
    Hashtable mapping Bitwarden item names to registry URLs.
    Default: @{ 'Docker Hub' = 'https://index.docker.io/v1/'; 'GitHub Container Registry' = 'https://ghcr.io' }

.PARAMETER Force
    Force credential update even if credentials already exist.

.PARAMETER DryRun
    Show what would be done without making changes.

.PARAMETER LogFile
    Path to log file. Default: C:\Scripts\Startup\registry-credentials.log

.EXAMPLE
    .\Sync-RegistryCredentials.ps1
    Synchronizes all configured registry credentials.

.EXAMPLE
    .\Sync-RegistryCredentials.ps1 -Force -Verbose
    Forces credential update with verbose output.

.NOTES
    Author: Claude Code Framework
    Version: 1.0
    Requires: Bitwarden CLI with valid BW_SESSION, docker-credential-wincred

.LINK
    https://docs.docker.com/engine/reference/commandline/login/#credential-helpers
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter()]
    [hashtable]$BitwardenItems = @{
        # Item names must match exactly as they appear in Bitwarden vault
        'docker.com'                 = 'https://index.docker.io/v1/'
        # Note: Create 'ghcr.io' item in Bitwarden with GitHub username + PAT
        # (PAT needs read:packages and write:packages scopes)
        'ghcr.io'                    = 'https://ghcr.io'
    },

    [Parameter()]
    [switch]$Force,

    [Parameter()]
    [switch]$DryRun,

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$LogFile = 'C:\Scripts\Startup\registry-credentials.log'
)

#region Script Configuration
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:Config = @{
    CredentialHelper = 'docker-credential-wincred'
    MaxLogSizeBytes  = 5MB
    LogRetentionDays = 30
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

    # Rotate log if too large
    if ((Test-Path $LogFile) -and (Get-Item $LogFile).Length -gt $script:Config.MaxLogSizeBytes) {
        $archiveName = $LogFile -replace '\.log$', "-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"
        Move-Item -Path $LogFile -Destination $archiveName -Force
        Write-Verbose "Log rotated to: $archiveName"
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

#region Validation Functions
function Test-BitwardenSession {
    <#
    .SYNOPSIS
        Verifies that a valid Bitwarden session exists.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    if ([string]::IsNullOrEmpty($env:BW_SESSION)) {
        Write-Log 'BW_SESSION environment variable not set' -Level WARNING
        return $false
    }

    try {
        $result = & bw status 2>&1
        if ($LASTEXITCODE -eq 0) {
            $status = $result | ConvertFrom-Json
            if ($status.status -eq 'unlocked') {
                Write-Log 'Bitwarden session is valid' -Level DEBUG
                return $true
            }
            Write-Log "Bitwarden vault status: $($status.status)" -Level WARNING
        }
        return $false
    }
    catch {
        Write-Log "Failed to check Bitwarden session: $_" -Level WARNING
        return $false
    }
}

function Test-CredentialHelper {
    <#
    .SYNOPSIS
        Verifies docker-credential-wincred is available.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    $helper = Get-Command -Name $script:Config.CredentialHelper -ErrorAction SilentlyContinue
    if (-not $helper) {
        Write-Log "$($script:Config.CredentialHelper) not found in PATH" -Level ERROR
        return $false
    }

    Write-Log "Credential helper found: $($helper.Source)" -Level DEBUG
    return $true
}
#endregion

#region Bitwarden Functions
function Get-BitwardenItem {
    <#
    .SYNOPSIS
        Retrieves an item from Bitwarden vault by name.
    .PARAMETER ItemName
        The name of the Bitwarden item to retrieve.
    .OUTPUTS
        [PSCustomObject] The Bitwarden item with login details.
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [string]$ItemName
    )

    Write-Log "Retrieving Bitwarden item: $ItemName" -Level DEBUG

    try {
        $itemJson = & bw get item $ItemName 2>&1
        if ($LASTEXITCODE -ne 0) {
            # Check if item not found
            if ($itemJson -match 'Not found') {
                Write-Log "Bitwarden item not found: $ItemName" -Level WARNING
                return $null
            }
            throw "Failed to retrieve item: $itemJson"
        }

        $item = $itemJson | ConvertFrom-Json

        # Validate it's a login item with credentials
        if ($item.type -ne 1) {  # Type 1 = Login
            Write-Log "Bitwarden item '$ItemName' is not a login item (type=$($item.type))" -Level WARNING
            return $null
        }

        if ([string]::IsNullOrEmpty($item.login.username) -or [string]::IsNullOrEmpty($item.login.password)) {
            Write-Log "Bitwarden item '$ItemName' is missing username or password" -Level WARNING
            return $null
        }

        Write-Log "Successfully retrieved Bitwarden item: $ItemName" -Level DEBUG
        return $item
    }
    catch {
        Write-Log "Error retrieving Bitwarden item '$ItemName': $_" -Level ERROR
        return $null
    }
}
#endregion

#region Credential Store Functions
function Get-StoredRegistryCredential {
    <#
    .SYNOPSIS
        Gets existing credential from docker-credential-wincred.
    .PARAMETER ServerUrl
        The registry server URL.
    .OUTPUTS
        [PSCustomObject] The stored credential or $null.
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [string]$ServerUrl
    )

    try {
        $result = $ServerUrl | & $script:Config.CredentialHelper get 2>&1
        if ($LASTEXITCODE -eq 0 -and $result) {
            $cred = $result | ConvertFrom-Json
            if ($cred.Username) {
                Write-Log "Found existing credential for: $ServerUrl (user: $($cred.Username))" -Level DEBUG
                return $cred
            }
        }
        return $null
    }
    catch {
        Write-Log "Error checking existing credential for '$ServerUrl': $_" -Level DEBUG
        return $null
    }
}

function Set-RegistryCredential {
    <#
    .SYNOPSIS
        Stores a credential in docker-credential-wincred.
    .PARAMETER ServerUrl
        The registry server URL.
    .PARAMETER Username
        The username/login.
    .PARAMETER Secret
        The password or token.
    .OUTPUTS
        [bool] True if credential was stored successfully.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory)]
        [string]$ServerUrl,

        [Parameter(Mandatory)]
        [string]$Username,

        [Parameter(Mandatory)]
        [string]$Secret
    )

    if (-not $PSCmdlet.ShouldProcess($ServerUrl, 'Store registry credential')) {
        return $true
    }

    Write-Log "Storing credential for: $ServerUrl (user: $Username)"

    $tempFile = $null
    try {
        $credJson = @{
            ServerURL = $ServerUrl
            Username  = $Username
            Secret    = $Secret
        } | ConvertTo-Json -Compress

        # SECURITY: Write to temp file with UTF8 no BOM, then pipe via cmd.exe
        # PowerShell's native piping adds BOM which causes "invalid character 'ï'" errors
        $tempFile = [System.IO.Path]::GetTempFileName()
        [System.IO.File]::WriteAllText($tempFile, $credJson, [System.Text.UTF8Encoding]::new($false))

        $result = & cmd /c "type `"$tempFile`" | $($script:Config.CredentialHelper) store" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Failed to store credential for '$ServerUrl': $result" -Level ERROR
            return $false
        }

        Write-Log "Successfully stored credential for: $ServerUrl" -Level SUCCESS
        return $true
    }
    catch {
        Write-Log "Error storing credential for '$ServerUrl': $_" -Level ERROR
        return $false
    }
    finally {
        # Clear sensitive data
        $credJson = $null
        $Secret = $null
        # Delete temp file with credential
        if ($tempFile -and (Test-Path $tempFile)) {
            Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
        }
    }
}

function Get-AllStoredCredentials {
    <#
    .SYNOPSIS
        Lists all credentials stored in docker-credential-wincred.
    .OUTPUTS
        [string[]] Array of server URLs with stored credentials.
    #>
    [CmdletBinding()]
    [OutputType([string[]])]
    param()

    try {
        $result = & $script:Config.CredentialHelper list 2>&1
        if ($LASTEXITCODE -eq 0 -and $result) {
            $list = $result | ConvertFrom-Json
            return $list.PSObject.Properties.Name
        }
        return @()
    }
    catch {
        Write-Log "Error listing credentials: $_" -Level WARNING
        return @()
    }
}
#endregion

#region Main Functions
function Sync-SingleCredential {
    <#
    .SYNOPSIS
        Synchronizes a single credential from Bitwarden to wincred.
    .PARAMETER ItemName
        The Bitwarden item name.
    .PARAMETER ServerUrl
        The registry server URL.
    .OUTPUTS
        [bool] True if sync was successful.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory)]
        [string]$ItemName,

        [Parameter(Mandatory)]
        [string]$ServerUrl
    )

    Write-Log "Syncing: $ItemName -> $ServerUrl"

    # Check if credential already exists (skip if not forcing)
    if (-not $Force) {
        $existing = Get-StoredRegistryCredential -ServerUrl $ServerUrl
        if ($existing) {
            Write-Log "Credential already exists for $ServerUrl, skipping (use -Force to override)" -Level DEBUG
            return $true
        }
    }

    # Get credential from Bitwarden
    $item = Get-BitwardenItem -ItemName $ItemName
    if (-not $item) {
        Write-Log "Could not retrieve Bitwarden item: $ItemName" -Level WARNING
        return $false
    }

    # Store credential
    if ($DryRun) {
        Write-Log "[DryRun] Would store credential for $ServerUrl (user: $($item.login.username))" -Level INFO
        return $true
    }

    $result = Set-RegistryCredential -ServerUrl $ServerUrl -Username $item.login.username -Secret $item.login.password

    # For Docker Hub, also store under short format for Podman compatibility
    # Docker uses: https://index.docker.io/v1/
    # Podman uses: docker.io
    if ($ServerUrl -eq 'https://index.docker.io/v1/') {
        Write-Log 'Also storing Docker Hub credential for Podman (docker.io)' -Level DEBUG
        Set-RegistryCredential -ServerUrl 'docker.io' -Username $item.login.username -Secret $item.login.password | Out-Null
    }

    # Clear sensitive data from memory
    $item = $null
    [System.GC]::Collect()

    return $result
}

function Invoke-CredentialSync {
    <#
    .SYNOPSIS
        Main function to synchronize all registry credentials.
    .OUTPUTS
        [int] Exit code (0 = success, 1 = partial failure, 2 = total failure).
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([int])]
    param()

    Write-Log '=== Registry Credential Sync Started ==='

    # Validate prerequisites
    if (-not (Test-BitwardenSession)) {
        Write-Log 'No valid Bitwarden session. Run Initialize-BitwardenSession.ps1 first.' -Level ERROR
        return 2
    }

    if (-not (Test-CredentialHelper)) {
        Write-Log 'Credential helper not available' -Level ERROR
        return 2
    }

    # Show current state
    $existingCreds = Get-AllStoredCredentials
    if ($existingCreds.Count -gt 0) {
        Write-Log "Existing credentials: $($existingCreds -join ', ')" -Level DEBUG
    }

    # Sync each configured credential
    $successCount = 0
    $failureCount = 0

    foreach ($itemName in $BitwardenItems.Keys) {
        $serverUrl = $BitwardenItems[$itemName]
        try {
            $success = Sync-SingleCredential -ItemName $itemName -ServerUrl $serverUrl
            if ($success) {
                $successCount++
            }
            else {
                $failureCount++
            }
        }
        catch {
            Write-Log "Error syncing '$itemName': $_" -Level ERROR
            $failureCount++
        }
    }

    # Report results
    Write-Log "Sync complete: $successCount succeeded, $failureCount failed"

    if ($failureCount -eq 0) {
        Write-Log '=== Registry Credential Sync Complete ===' -Level SUCCESS
        return 0
    }
    elseif ($successCount -gt 0) {
        Write-Log '=== Registry Credential Sync Partial ===' -Level WARNING
        return 1
    }
    else {
        Write-Log '=== Registry Credential Sync Failed ===' -Level ERROR
        return 2
    }
}
#endregion

#region Entry Point
try {
    $exitCode = Invoke-CredentialSync
    exit $exitCode
}
catch {
    Write-Log "Unhandled exception: $_" -Level ERROR
    Write-Log $_.ScriptStackTrace -Level ERROR
    exit 2
}
#endregion
