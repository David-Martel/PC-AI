#Requires -Version 7.0
<#
.SYNOPSIS
    Three-Tier Secrets Management Module for Windows
.DESCRIPTION
    Provides resilient secrets management with:
    - Tier 1: BWS Cloud (primary, online)
    - Tier 2: Local DPAPI-encrypted cache (offline)
    - Tier 3: BW Vault backup (manual recovery)
.NOTES
    Machine: explicit MachineName; defaults to current COMPUTERNAME
    Created: 2025-12-28
.LINK
    ../README.md#credential-backend-code-and-custody
#>

[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingPlainTextForPassword','',Justification='CredentialRoot is a filesystem directory path, not a password or secret value.')]
param([ValidateNotNullOrEmpty()][string]$MachineName = $env:COMPUTERNAME,
    [string]$MachineRoot = (Join-Path $env:USERPROFILE '.machine'),
    [string]$CacheRoot,[string]$CredentialRoot = $env:PCAI_CREDENTIAL_ROOT)

# Load required assemblies for DPAPI encryption
Add-Type -AssemblyName System.Security

# Module Configuration
$script:MachineDir = [IO.Path]::GetFullPath($MachineRoot)
$script:IdentityFile = "$script:MachineDir\identity.json"
$script:MaxCacheAgeDays = 7
$script:BwSecureNoteName = "[MACHINE] $MachineName Secrets"

# Module import performs no provisioning, provider registration or secret access.
$backendHelperPath = Join-Path $PSScriptRoot 'SecretBackendUtilities.ps1'
if (-not (Test-Path -LiteralPath $backendHelperPath -PathType Leaf)) { throw 'The reviewed sibling backend helper is required.' }
. $backendHelperPath
# Explicit storage is authoritative. Import admits metadata only; it never
# provisions directories, authenticates, reads secret files or changes ACLs.
foreach($name in @('MachineRoot','CacheRoot','CredentialRoot')) {
    if ($PSBoundParameters.ContainsKey($name) -and [string]::IsNullOrWhiteSpace((Get-Variable -Name $name -ValueOnly))) { throw 'Empty explicit credential storage refused.' }
}
$script:LegacyCacheAllowed = -not $PSBoundParameters.ContainsKey('MachineRoot') -and -not $PSBoundParameters.ContainsKey('CacheRoot') -and [string]::IsNullOrWhiteSpace($CredentialRoot)
if ($PSBoundParameters.ContainsKey('MachineRoot')) { $null=Resolve-CredentialPrivateStorage -RootOverride $MachineRoot -Purpose Explicit }
if (-not [string]::IsNullOrWhiteSpace($CredentialRoot)) { $null=Resolve-CredentialPrivateStorage -RootOverride $CredentialRoot -Purpose Cache -MachineName $MachineName }
$script:CacheSelection = if ($PSBoundParameters.ContainsKey('CacheRoot')) {
    Resolve-CredentialPrivateStorage -RootOverride $CacheRoot -Purpose Explicit
} elseif ($PSBoundParameters.ContainsKey('MachineRoot')) {
    Resolve-CredentialPrivateStorage -RootOverride $MachineRoot -Purpose Explicit
} else {
    Resolve-CredentialPrivateStorage -RootOverride $CredentialRoot -Purpose Cache -MachineName $MachineName
}
$script:CacheDir=$script:CacheSelection.Directory
$script:CacheFile=Join-Path $script:CacheDir 'secrets-cache.enc'
$script:ManifestFile=Join-Path $script:CacheDir 'secrets-manifest.json'
$script:LegacyCacheFile=Join-Path $script:MachineDir 'secrets-cache.enc'
$script:LegacyManifestFile=Join-Path $script:MachineDir 'secrets-manifest.json'
#region Helper Functions

function Test-InternetConnection {
    <#
    .SYNOPSIS
    Quick test for internet connectivity
    #>
    try {
        $null = [System.Net.Dns]::GetHostEntry("api.bitwarden.com")
        return $true
    } catch {
        return $false
    }
}

function Get-MachineIdentityInternal {
    <#
    .SYNOPSIS
    Gets machine identity from local file (internal module function)
    #>
    if (Test-Path $script:IdentityFile) {
        try {
            return Get-Content $script:IdentityFile -Raw | ConvertFrom-SecretBackendJson
        } catch {
            return $null
        }
    }
    # Fallback: return basic identity
    return @{
        machineId = (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Cryptography' -Name MachineGuid -ErrorAction SilentlyContinue).MachineGuid
        computerName = $env:COMPUTERNAME
    }
}

function ConvertTo-EncryptedString {
    <#
    .SYNOPSIS
    Encrypt explicit synthetic or selected secret text using current-user DPAPI.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param([Parameter(Mandatory)][string]$PlainText)
    $bytes=[Text.Encoding]::UTF8.GetBytes($PlainText);$encrypted=$null
    try{
        $encrypted=[Security.Cryptography.ProtectedData]::Protect($bytes,$null,[Security.Cryptography.DataProtectionScope]::CurrentUser)
        return [Convert]::ToBase64String($encrypted)
    }finally{[Array]::Clear($bytes,0,$bytes.Length);if($encrypted){[Array]::Clear($encrypted,0,$encrypted.Length)}}
}

function ConvertFrom-EncryptedString {
    <#
    .SYNOPSIS
    Decrypt selected cache text and clear intermediate byte buffers.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param([Parameter(Mandatory)][string]$EncryptedText)
    $encrypted=$null;$decrypted=$null
    try{
        $encrypted=[Convert]::FromBase64String($EncryptedText)
        $decrypted=[Security.Cryptography.ProtectedData]::Unprotect($encrypted,$null,[Security.Cryptography.DataProtectionScope]::CurrentUser)
        return [Text.Encoding]::UTF8.GetString($decrypted)
    }catch{throw [Security.Cryptography.CryptographicException]::new('Selected cache could not be decrypted; private content is withheld.')}
    finally{if($encrypted){[Array]::Clear($encrypted,0,$encrypted.Length)};if($decrypted){[Array]::Clear($decrypted,0,$decrypted.Length)}}
}

#endregion

#region SecretManagement and CredentialManager Integration

$script:PreferredSecretVaultName = if ($env:MACHINE_SECRETS_VAULT) {
    [string]$env:MACHINE_SECRETS_VAULT
} else {
    'LocalStore'
}

function ConvertFrom-SecureStringToPlainText {
    param([Parameter(Mandatory)][Security.SecureString]$SecureString)

    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureString)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    } finally {
        if ($bstr -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        }
    }
}

function Test-SecretManagementAvailable {
    return [bool](Get-Command Microsoft.PowerShell.SecretManagement\Get-SecretVault -ErrorAction SilentlyContinue)
}

function Get-PreferredSecretVaultName {
    <#
    .SYNOPSIS
    Return the configured preferred local vault name.
    #>

    return $script:PreferredSecretVaultName
}

function Register-LocalSecretVault {
    <#
    .SYNOPSIS
    Register a selected local vault after explicit confirmation.
    #>

    [CmdletBinding(SupportsShouldProcess)]
    param(
        [string]$VaultName = $script:PreferredSecretVaultName,
        [switch]$SetDefault
    )
    if (-not $PSCmdlet.ShouldProcess('Ensure-LocalSecretVaultRegistration state', 'Apply requested backend operation')) { return $false }


    if (-not (Test-SecretManagementAvailable)) {
        return $false
    }

    $registered = Microsoft.PowerShell.SecretManagement\Get-SecretVault -Name $VaultName -ErrorAction SilentlyContinue
    if ($registered) {
        return $true
    }

    $secretStoreModule = Get-Module -ListAvailable Microsoft.PowerShell.SecretStore |
        Sort-Object Version -Descending |
        Select-Object -First 1
    if (-not $secretStoreModule) {
        return $false
    }

    try {
        $params = @{
            Name       = $VaultName
            ModuleName = 'Microsoft.PowerShell.SecretStore'
        }
        if ($SetDefault) {
            $params.DefaultVault = $true
        }

        Microsoft.PowerShell.SecretManagement\Register-SecretVault @params -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Get-SecretManagementStatus {
    <#
    .SYNOPSIS
    Return local provider and vault registration metadata.
    #>

    [CmdletBinding()]
    param()

    $vaults = @()
    if (Test-SecretManagementAvailable) {
        $vaults = @(Microsoft.PowerShell.SecretManagement\Get-SecretVault -ErrorAction SilentlyContinue)
    }

    [pscustomobject]@{
        Available            = (Test-SecretManagementAvailable)
        PreferredVault       = $script:PreferredSecretVaultName
        RegisteredVaultCount = $vaults.Count
        RegisteredVaults     = @($vaults | Select-Object -ExpandProperty Name)
        LocalStoreAvailable  = [bool](Get-Module -ListAvailable Microsoft.PowerShell.SecretStore)
        BitwardenVaultModule = [bool](Get-Module -ListAvailable SecretManagement.BitWarden)
    }
}

function Get-PlainTextSecretFromSecretManagement {
    [OutputType([string])]
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Name,
        [string]$VaultName = $script:PreferredSecretVaultName
    )

    if (-not (Test-SecretManagementAvailable)) {
        return $null
    }

    try {
        $secret = Microsoft.PowerShell.SecretManagement\Get-Secret -Name $Name -Vault $VaultName -AsPlainText -ErrorAction Stop
        if (-not [string]::IsNullOrWhiteSpace([string]$secret)) {
            return [string]$secret
        }
    } catch { Write-Verbose 'Optional backend operation failed in Get-PlainTextSecretFromSecretManagement; private provider output is withheld.' }

    return $null
}

function Set-PlainTextSecretInSecretManagement {
    [OutputType([bool])]
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Secret,
        [string]$VaultName = $script:PreferredSecretVaultName,
        [hashtable]$Metadata
    )
    if (-not $PSCmdlet.ShouldProcess('Set-PlainTextSecretInSecretManagement state', 'Apply requested backend operation')) { return $false }


    if (-not (Ensure-LocalSecretVaultRegistration -VaultName $VaultName)) {
        return $false
    }

    if (-not $PSCmdlet.ShouldProcess($Name, "Set secret in vault '$VaultName'")) {
        return $false
    }

    try {
        $params = @{
            Name   = $Name
            Vault  = $VaultName
            Secret = $Secret
        }
        if ($Metadata) {
            $params.Metadata = $Metadata
        }

        Microsoft.PowerShell.SecretManagement\Set-Secret @params -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Import-CredentialManagerSupport {
    if (Get-Command Get-StoredCredential -ErrorAction SilentlyContinue) {
        return $true
    }

    try {
        Import-Module CredentialManager -ErrorAction Stop
    } catch {
        return $false
    }

    return [bool](Get-Command Get-StoredCredential -ErrorAction SilentlyContinue)
}

function Get-PlainTextSecretFromCredentialManager {
    [OutputType([string])]
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Target)

    if (-not (Import-CredentialManagerSupport)) {
        return $null
    }

    try {
        $credObject = Get-StoredCredential -Target $Target -AsCredentialObject -ErrorAction SilentlyContinue
        if ($credObject -and $credObject.PSObject.Properties.Name -contains 'Password' -and
            -not [string]::IsNullOrWhiteSpace([string]$credObject.Password)) {
            return [string]$credObject.Password
        }
    } catch { Write-Verbose 'Optional backend operation failed in Get-PlainTextSecretFromCredentialManager; private provider output is withheld.' }

    try {
        $psCredential = Get-StoredCredential -Target $Target -ErrorAction SilentlyContinue
        if ($psCredential -is [pscredential]) {
            return (ConvertFrom-SecureStringToPlainText -SecureString $psCredential.Password)
        }
    } catch { Write-Verbose 'Optional backend operation failed in Get-PlainTextSecretFromCredentialManager; private provider output is withheld.' }

    return $null
}

function Unlock-BitwardenCliSession {
    <#
    .SYNOPSIS
    Validate and initialize a protected Bitwarden session.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([pscustomobject])]
    param()
    if(-not$PSCmdlet.ShouldProcess('Bitwarden session','Delegate guarded session validation')){return [pscustomobject]@{Unlocked=$false;Reason='not-applied';PasswordSource=$null;ClientSource=$null}}
    $state=Initialize-BitwardenSessionFromBackends -Quiet -WhatIf:$WhatIfPreference -Confirm:$false
    [pscustomobject]@{Unlocked=[bool]$state.Success;Reason=$state.Source;PasswordSource=$null;ClientSource=$null}
}

function Publish-ConnectionCredential {
    <#
    .SYNOPSIS
    Publish confirmed connection credentials without plaintext process arguments.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([pscustomobject])]
    param([Parameter(Mandatory)][string]$Name,[Parameter(Mandatory)][string]$Username,[Parameter(Mandatory)][WorkProfileBackend.SecureSecretArgument()][Security.SecureString]$Password,[string[]]$Targets=@(),[string]$VaultName=$script:PreferredSecretVaultName,[string]$Comment='')
    $secretName="connection/$Name";$stored=$false;$results=@()
    if($PSCmdlet.ShouldProcess($secretName,'Publish selected credential to local vault')){
        $plain=ConvertFrom-SecureStringToPlainText -SecureString $Password
        try{
            $payload=@{name=$Name;username=$Username;password=$plain;targets=$Targets;comment=$Comment;updatedAt=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress -Depth 5
            $stored=Set-PlainTextSecretInSecretManagement -Name $secretName -Secret $payload -VaultName $VaultName -Confirm:$false
        }finally{$plain=$null;$payload=$null}
    }
    foreach($target in $Targets){
        $success=$false
        if($PSCmdlet.ShouldProcess($target,'Publish selected Windows credential')){
            $kind=if($target-match'^(?i:TERMSRV/)' -or$target.Contains(':')){'generic'}else{'domain'}
            $success=Set-CredentialManagerCredential -Target $target -UserName $Username -Password $Password -Kind $kind -Confirm:$false
        }
        $results+=[pscustomobject]@{Target=$target;Success=$success}
    }
    [pscustomobject]@{Name=$Name;SecretName=$secretName;SecretStored=$stored;CredentialTargets=$results}
}

function Get-ConnectionCredential {
    <#
    .SYNOPSIS
    Read a selected connection credential from a local vault.
    #>

    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Name,
        [string]$VaultName = $script:PreferredSecretVaultName
    )

    $secretName = "connection/$Name"
    $payload = Get-PlainTextSecretFromSecretManagement -Name $secretName -VaultName $VaultName
    if ([string]::IsNullOrWhiteSpace($payload)) {
        return $null
    }

    try {
        return ($payload | ConvertFrom-SecretBackendJson -ErrorAction Stop)
    } catch {
        return $null
    }
}

#endregion

#region Tier 1: BWS Cloud Functions

function Get-BwsSecretsList {
    <#
    .SYNOPSIS
    Read BWS secrets using an explicitly available process token.
    #>
    [CmdletBinding()]
    [OutputType([hashtable])]
    param()
    if(-not$env:BWS_ACCESS_TOKEN){Write-Verbose 'No explicit process BWS token is available.';return $null}
    $command=Get-Command bws.exe -CommandType Application -ErrorAction SilentlyContinue|Select-Object -First 1
    if(-not$command){Write-Verbose 'BWS executable is unavailable.';return $null}
    $response=Invoke-SecretBackendProcess -FilePath $command.Source -Arguments @('secret','list','-o','json')
    if(-not$response.Success){Write-Verbose 'BWS command failed; private output is withheld.';return $null}
    try{
        $result=@{}
        foreach($entry in @($response.StdOut|ConvertFrom-SecretBackendJson -ErrorAction Stop)){
            if([string]::IsNullOrWhiteSpace([string]$entry.key)-or$result.ContainsKey([string]$entry.key)){throw 'Invalid or duplicate BWS key.'}
            $result[[string]$entry.key]=[string]$entry.value
        }
        return $result
    }catch{throw [FormatException]::new('BWS response is invalid; private content is withheld.')}
}

function Sync-SecretsFromBws {
    <#
    .SYNOPSIS
    Fetches secrets from BWS and updates local cache
    .OUTPUTS
    Boolean indicating success
    #>
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns','',Justification='Existing compatibility API Sync-SecretsFromBws is referenced by installed profiles or sibling backend code; preserve its exact name.')]
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [switch]$Force
    )
    if (-not $PSCmdlet.ShouldProcess('Sync-SecretsFromBws state', 'Apply requested backend operation')) { return $false }


    if (-not $Force -and (Get-CacheStatus).IsValid) { return $true }
    Write-Verbose "Attempting BWS sync..."

    if (-not (Test-InternetConnection)) {
        Write-Warning "No internet connection - cannot sync from BWS"
        return $false
    }

    $secrets = Get-BwsSecretsList
    if (-not $secrets -or $secrets.Count -eq 0) {
        Write-Verbose "No secrets returned from BWS"
        return $false
    }

    # Update local cache
    Save-SecretsToCache -Secrets $secrets

    Write-Verbose "Synced $($secrets.Count) secrets from BWS Cloud"
    return $true
}

#endregion

#region Tier 2: Local Cache Functions

function Save-SecretsToCache {
    <#
    .SYNOPSIS
    Write protected DPAPI cache with a bound metadata digest.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param([Parameter(Mandatory)][hashtable]$Secrets)
    if(-not$PSCmdlet.ShouldProcess($script:CacheDir,'Write encrypted cache and bound metadata')){return}
    $null=Initialize-CredentialPrivateStorage -Selection $script:CacheSelection
    $json=$Secrets|ConvertTo-Json -Compress -Depth 20
    try{$encrypted=ConvertTo-EncryptedString -PlainText $json}finally{$json=$null}
    Write-BitwardenPrivateFile -LiteralPath $script:CacheFile -Value $encrypted -Overwrite -Confirm:$false
    $manifest=@{lastUpdated=[DateTime]::UtcNow.ToString('o');secretCount=$Secrets.Count;source='explicit';machineName=$MachineName;cacheSha256=(Get-FileHash -LiteralPath $script:CacheFile).Hash}
    Write-BitwardenPrivateFile -LiteralPath $script:ManifestFile -Value ($manifest|ConvertTo-Json -Compress) -Overwrite -Confirm:$false
}

function Get-SecretsCachePair {
    $cache=$script:CacheFile;$manifest=$script:ManifestFile;$source='Protected'
    $cacheExists=Test-Path -LiteralPath $cache
    $manifestExists=Test-Path -LiteralPath $manifest
    # A partial/new invalid pair never authorizes rollback to an older cache.
    if (($cacheExists -or $manifestExists) -and -not ($cacheExists -and $manifestExists)) { throw 'Selected cache pair is incomplete; private content is withheld.' }
    if ($cacheExists -and $manifestExists -and
        (-not (Test-Path -LiteralPath $cache -PathType Leaf) -or -not (Test-Path -LiteralPath $manifest -PathType Leaf))) {
        throw 'Selected cache pair must contain ordinary files; private content is withheld.'
    }
    if (-not $cacheExists -and -not $manifestExists -and $script:LegacyCacheAllowed) {
        $cache=$script:LegacyCacheFile;$manifest=$script:LegacyManifestFile;$source='LegacyReadOnly'
        $cacheExists=Test-Path -LiteralPath $cache -PathType Leaf
        $manifestExists=Test-Path -LiteralPath $manifest -PathType Leaf
        if (($cacheExists -or $manifestExists) -and -not ($cacheExists -and $manifestExists)) { throw 'Selected legacy pair is incomplete; private content is withheld.' }
    }
    return [pscustomobject]@{Exists=($cacheExists -and $manifestExists);Cache=$cache;Manifest=$manifest;Source=$source}
}

function Read-SecretsCachePair {
    param([Parameter(Mandatory)]$Pair,[switch]$Decrypt)
    $cache=$null;$manifest=$null;$json=$null
    try {
        $cache=Open-CredentialPrivateRead $Pair.Cache
        $manifest=Open-CredentialPrivateRead $Pair.Manifest
        $metadata=$manifest.Text|ConvertFrom-SecretBackendJson -AsHashtable -ErrorAction Stop
        if ($metadata -isnot [hashtable] -or -not $metadata.ContainsKey('lastUpdated') -or -not $metadata.ContainsKey('secretCount')) { throw 'Invalid cache metadata schema.' }
        if ($metadata.ContainsKey('machineName') -and $metadata.machineName -cne $MachineName) { throw 'Selected cache machine identity mismatch.' }
        $updated=if($metadata.lastUpdated -is [DateTime]){$metadata.lastUpdated.ToUniversalTime()}else{[DateTime]::Parse([string]$metadata.lastUpdated,[Globalization.CultureInfo]::InvariantCulture,[Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()}
        $age=[DateTime]::UtcNow-$updated
        if ($metadata.secretCount -isnot [int] -and $metadata.secretCount -isnot [long]) { throw 'Invalid cache count type.' }
        $count=[int]$metadata.secretCount
        if ($age.TotalSeconds -lt -300 -or $count -lt 0) { throw 'Invalid cache age or count.' }
        $bound=$metadata.ContainsKey('cacheSha256')
        if ($bound -and ($metadata.cacheSha256 -notmatch '^[A-Fa-f0-9]{64}$' -or $metadata.cacheSha256 -ine $cache.Snapshot.Hash)) { throw 'Cache digest mismatch.' }
        $secrets=$null
        if ($Decrypt) {
            $json=ConvertFrom-EncryptedString -EncryptedText $cache.Text
            $secrets=$json|ConvertFrom-SecretBackendJson -AsHashtable -ErrorAction Stop
            if ($secrets -isnot [hashtable] -or $secrets.Count -ne $count) { throw 'Cache secret data schema/count mismatch.' }
        }
        Assert-CredentialPrivateReadCurrent $cache
        Assert-CredentialPrivateReadCurrent $manifest
        return [pscustomobject]@{Age=$age;SecretCount=$count;LastUpdated=$updated;BoundDigest=$bound;Secrets=$secrets;Source=$Pair.Source}
    } catch { throw [IO.InvalidDataException]::new('Selected cache is invalid; private content is withheld.') }
    finally {
        $json=$null
        if ($cache) { $cache.Text=$null;$cache.Stream.Dispose() }
        if ($manifest) { $manifest.Text=$null;$manifest.Stream.Dispose() }
    }
}

function Get-SecretsFromCache {
    <#
    .SYNOPSIS
    Read a validated cache without environment hydration.
    #>
    [CmdletBinding()]
    [OutputType([hashtable])]
    param([switch]$IgnoreAge)
    $pair=Get-SecretsCachePair
    if (-not $pair.Exists) { return $null }
    $read=Read-SecretsCachePair -Pair $pair -Decrypt
    if (-not $IgnoreAge -and $read.Age.TotalDays -gt $script:MaxCacheAgeDays) { return $null }
    return $read.Secrets
}

function Get-CacheStatus {
    <#
    .SYNOPSIS
    Validate cache metadata and digest without decrypting secrets.
    #>
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param()
    $pair=Get-SecretsCachePair
    $state=@{Exists=$pair.Exists;Age=$null;SecretCount=0;IsValid=$false;IsExpired=$false;LastUpdated=$null;BoundDigest=$false;Source=$pair.Source}
    if (-not $pair.Exists) { return [pscustomobject]$state }
    $read=Read-SecretsCachePair -Pair $pair
    $state.Age=$read.Age;$state.SecretCount=$read.SecretCount;$state.LastUpdated=$read.LastUpdated;$state.BoundDigest=$read.BoundDigest
    $state.IsExpired=$state.Age.TotalDays -gt $script:MaxCacheAgeDays
    $state.IsValid=-not $state.IsExpired
    return [pscustomobject]$state
}

#endregion

#region Tier 3: BW Vault Functions

function Test-BwSession {
    <#
    .SYNOPSIS
    Tests if BW vault is unlocked
    #>
    try {
        $status = Invoke-MachineBitwardenCommand status 2>&1 | ConvertFrom-SecretBackendJson
        return $status.status -eq "unlocked"
    } catch {
        return $false
    }
}

function Set-BwNoteTextInternal {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param([Parameter(Mandatory)][string]$Name,[Parameter(Mandatory)][string]$Text)
    if(-not$PSCmdlet.ShouldProcess($Name,'Write selected secure note without secret command arguments')){return $false}
    $items=Invoke-MachineBitwardenCommand list items --search $Name|ConvertFrom-SecretBackendJson -ErrorAction Stop
    $matching=@($items|Where-Object {$_.name-eq$Name-and$_.type-eq2})
    if($matching.Count-gt1){throw 'Ambiguous secure-note identity refused.'}
    if($matching.Count-eq1){
        $item=Invoke-MachineBitwardenCommand get item $matching[0].id|ConvertFrom-SecretBackendJson -ErrorAction Stop
        if(-not$item.id-or$item.id-ne$matching[0].id){throw 'Selected secure-note identity could not be verified.'}
        $item.notes=$Text;$arguments=@('edit','item',[string]$item.id)
    }else{$item=@{type=2;name=$Name;notes=$Text;secureNote=@{type=0}};$arguments=@('create','item')}
    $json=$item|ConvertTo-Json -Compress -Depth 20
    try{
        $encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($json))
        $result=Invoke-BitwardenCli -Arguments $arguments -StandardInput $encoded
        if(-not$result.Success){throw 'Secure-note write failed; private command output is withheld.'}
        return $true
    }finally{$json=$null;$encoded=$null}
}

function Get-BwNoteTextInternal {
    [CmdletBinding()]
    [OutputType([string])]
    param([Parameter(Mandatory)][string]$Name)
    $matching=@(Invoke-MachineBitwardenCommand list items --search $Name|ConvertFrom-SecretBackendJson -ErrorAction Stop|Where-Object {$_.name-eq$Name-and$_.type-eq2})
    if($matching.Count-gt1){throw 'Exact secure-note identity is ambiguous.'}
    if($matching.Count-eq1){return [string]$matching[0].notes}
    return $null
}

# Backup v2 uses an immutable generation namespace. The canonical note is only
# a commit manifest; readers never assemble numbered parts from different writes.
$script:BwBackupPrefix = '#pcai-secrets-backup-v2' + [char]10
$script:BwBackupChunkSize = 4500
$script:BwBackupMaxParts = 1024
$script:BwBackupMaxRawLength = 3456000

function Get-BwBackupHashInternal {
    param([Parameter(Mandatory)][byte[]]$Bytes)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash($Bytes)).Replace('-','').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}

function Get-BwBackupIntegerInternal {
    param([System.Text.Json.JsonElement]$Element,[long]$Minimum,[long]$Maximum)
    $value = 0L
    if ($Element.ValueKind -ne [System.Text.Json.JsonValueKind]::Number -or
        -not $Element.TryGetInt64([ref]$value) -or $value -lt $Minimum -or $value -gt $Maximum) {
        throw 'Backup integer is outside the admitted bound.'
    }
    return $value
}

function Get-BwBackupManifestInternal {
    param([Parameter(Mandatory)][string]$CanonicalText)
    if (-not $CanonicalText.StartsWith($script:BwBackupPrefix,[StringComparison]::Ordinal) -or
        [Text.Encoding]::UTF8.GetByteCount($CanonicalText) -gt 2048) { throw 'Invalid backup commit manifest.' }
    $document = $null
    try {
        $document = [System.Text.Json.JsonDocument]::Parse($CanonicalText.Substring($script:BwBackupPrefix.Length))
        $root = $document.RootElement
        if ($root.ValueKind -ne [System.Text.Json.JsonValueKind]::Object) { throw 'Backup manifest object required.' }
        $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        foreach ($property in $root.EnumerateObject()) {
            if (-not $names.Add($property.Name)) { throw 'Duplicate backup manifest field.' }
        }
        if ((@($names | Sort-Object) -join ',') -cne 'encodedLength,encoding,generation,machine,partCount,rawLength,sha256,version') {
            throw 'Unexpected backup manifest fields.'
        }
        $values = @{}
        foreach ($name in @('machine','generation','sha256','encoding')) {
            $element = $root.GetProperty($name)
            if ($element.ValueKind -ne [System.Text.Json.JsonValueKind]::String) { throw 'Backup string field required.' }
            $values[$name] = $element.GetString()
        }
        if ($values.machine -cne $MachineName -or $values.encoding -cne 'utf8-base64' -or
            $values.generation -cnotmatch '\A[0-9a-f]{32}\z' -or $values.sha256 -cnotmatch '\A[0-9a-f]{64}\z') {
            throw 'Backup identity or encoding mismatch.'
        }
        $values.version = Get-BwBackupIntegerInternal $root.GetProperty('version') 2 2
        $values.partCount = Get-BwBackupIntegerInternal $root.GetProperty('partCount') 1 $script:BwBackupMaxParts
        $values.rawLength = Get-BwBackupIntegerInternal $root.GetProperty('rawLength') 1 $script:BwBackupMaxRawLength
        $values.encodedLength = Get-BwBackupIntegerInternal $root.GetProperty('encodedLength') 4 ($script:BwBackupMaxParts * $script:BwBackupChunkSize)
        if ($values.encodedLength -ne (4L * [long][Math]::Ceiling($values.rawLength / 3.0)) -or
            $values.partCount -ne [long][Math]::Ceiling($values.encodedLength / [double]$script:BwBackupChunkSize)) {
            throw 'Backup count and length mismatch.'
        }
        return [pscustomobject]$values
    } catch { throw [IO.InvalidDataException]::new('Backup commit manifest is invalid; private content is withheld.') }
    finally { if ($document) { $document.Dispose() } }
}

function ConvertFrom-BwBackupElementInternal {
    param([System.Text.Json.JsonElement]$Element,[int]$Depth = 0)
    if ($Depth -gt 20) { throw 'Backup nesting limit exceeded.' }
    switch ($Element.ValueKind) {
        Object {
            $result = @{}
            foreach ($property in $Element.EnumerateObject()) {
                if ($result.ContainsKey($property.Name)) { throw 'Duplicate backup object key.' }
                $result[$property.Name] = ConvertFrom-BwBackupElementInternal -Element $property.Value -Depth ($Depth + 1)
            }
            return $result
        }
        Array {
            $result = [Collections.Generic.List[object]]::new()
            foreach ($value in $Element.EnumerateArray()) {
                $result.Add((ConvertFrom-BwBackupElementInternal -Element $value -Depth ($Depth + 1)))
            }
            return ,$result.ToArray()
        }
        String { return $Element.GetString() }
        Number {
            $integer = 0L
            if ($Element.TryGetInt64([ref]$integer)) { return $integer }
            return $Element.GetDecimal()
        }
        True { return $true }
        False { return $false }
        Null { return $null }
        default { throw 'Unsupported backup value.' }
    }
}

function Get-BwBackupPartNameInternal {
    param([Parameter(Mandatory)][string]$Generation,[Parameter(Mandatory)][int]$Index)
    return ('{0} [pcai-v2:{1}:{2}]' -f $script:BwSecureNoteName,$Generation,$Index)
}

function Get-BwBackupPartHeaderInternal {
    param([Parameter(Mandatory)][string]$Generation,[Parameter(Mandatory)][int]$Index,[Parameter(Mandatory)][int]$Count)
    $header = ('#pcai-secrets-v2 generation={0} part={1} parts={2}' -f $Generation,$Index,$Count) + [char]10
    if ($header.Length -gt 128) { throw 'Backup header limit exceeded.' }
    return $header
}

function Read-BwBackupSnapshotInternal {
    param([Parameter(Mandatory)][string]$CanonicalText)
    $manifest = Get-BwBackupManifestInternal -CanonicalText $CanonicalText
    $encoded = [Text.StringBuilder]::new([int]$manifest.encodedLength)
    $names = [Collections.Generic.List[string]]::new()
    $texts = [Collections.Generic.List[string]]::new()
    for ($index = 1; $index -le $manifest.partCount; $index++) {
        $name = Get-BwBackupPartNameInternal $manifest.generation $index
        $text = Get-BwNoteTextInternal -Name $name
        $header = Get-BwBackupPartHeaderInternal $manifest.generation $index ([int]$manifest.partCount)
        $length = [Math]::Min($script:BwBackupChunkSize,[int]$manifest.encodedLength - (($index - 1) * $script:BwBackupChunkSize))
        if ($null -eq $text -or $text.Length -ne ($header.Length + $length) -or
            -not $text.StartsWith($header,[StringComparison]::Ordinal)) { throw 'Backup part identity or size mismatch.' }
        $chunk = $text.Substring($header.Length)
        if ($chunk -cnotmatch '\A[A-Za-z0-9+/]+={0,2}\z' -or
            ($index -lt $manifest.partCount -and $chunk.Contains('='))) { throw 'Invalid backup base64 part.' }
        [void]$encoded.Append($chunk)
        $names.Add($name)
        $texts.Add($text)
    }
    $bytes = $null
    $document = $null
    try {
        $joined = $encoded.ToString()
        $bytes = [Convert]::FromBase64String($joined)
        if ($bytes.Length -ne $manifest.rawLength -or [Convert]::ToBase64String($bytes) -cne $joined -or
            (Get-BwBackupHashInternal -Bytes $bytes) -cne $manifest.sha256) { throw 'Backup payload integrity mismatch.' }
        $json = [Text.UTF8Encoding]::new($false,$true).GetString($bytes)
        $document = [System.Text.Json.JsonDocument]::Parse($json)
        $backup = ConvertFrom-BwBackupElementInternal -Element $document.RootElement
        if ($backup -isnot [hashtable] -or $backup.machine -cne $MachineName -or
            $backup.secrets -isnot [hashtable] -or $backup.secretCount -isnot [long] -or
            $backup.secretCount -lt 1 -or $backup.secretCount -ne $backup.secrets.Count) {
            throw 'Backup payload schema or machine mismatch.'
        }
        return [pscustomobject]@{Secrets=$backup.secrets;Manifest=$manifest;CanonicalText=$CanonicalText;PartNames=$names.ToArray();PartTexts=$texts.ToArray()}
    } catch { throw [IO.InvalidDataException]::new('Backup generation is invalid; private content is withheld.') }
    finally {
        if ($document) { $document.Dispose() }
        if ($bytes) { [Array]::Clear($bytes,0,$bytes.Length) }
        $json = $null
        $joined = $null
    }
}

function Read-BwBackupCommittedInternal {
    param([Parameter(Mandatory)][string]$CanonicalText,[ValidateRange(0,1)][int]$RetryLimit = 1)
    # A reader racing legitimate generation retirement may retry once, but only
    # after observing a different valid v2 generation at the canonical name.
    for ($attempt = 0; $attempt -le $RetryLimit; $attempt++) {
        $manifest = Get-BwBackupManifestInternal -CanonicalText $CanonicalText
        $failure = $null
        try { $snapshot = Read-BwBackupSnapshotInternal -CanonicalText $CanonicalText }
        catch { $failure = $_ }
        $current = Get-BwNoteTextInternal -Name $script:BwSecureNoteName
        if ($current -ceq $CanonicalText) {
            if ($failure) { throw [IO.InvalidDataException]::new('Backup generation is invalid; private content is withheld.') }
            return $snapshot
        }
        if ($attempt -ge $RetryLimit -or [string]::IsNullOrEmpty($current)) { throw 'Backup commit changed; retry the explicit recovery.' }
        $next = Get-BwBackupManifestInternal -CanonicalText $current
        if ($next.generation -ceq $manifest.generation) { throw 'Backup commit changed without a new generation.' }
        $CanonicalText = $current
    }
    throw 'Backup commit could not be read coherently.'
}

function Remove-BwNoteTextInternal {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param([Parameter(Mandatory)][string]$Name,[Parameter(Mandatory)][string]$ExpectedText)
    if (-not $PSCmdlet.ShouldProcess($Name,'Remove exact observed backup part')) { return $false }
    $matching = @(Invoke-MachineBitwardenCommand list items --search $Name |
        ConvertFrom-SecretBackendJson -ErrorAction Stop | Where-Object { $_.name -ceq $Name -and $_.type -eq 2 })
    if ($matching.Count -eq 0) { return $true }
    if ($matching.Count -ne 1) { return $false }
    $item = Invoke-MachineBitwardenCommand get item $matching[0].id | ConvertFrom-SecretBackendJson -ErrorAction Stop
    if ($item.id -cne $matching[0].id -or $item.name -cne $Name -or $item.type -ne 2 -or $item.notes -cne $ExpectedText) { return $false }
    $result = Invoke-BitwardenCli -Arguments @('delete','item',[string]$item.id)
    if (-not $result.Success) { return $false }
    return $null -eq (Get-BwNoteTextInternal -Name $Name)
}

function Remove-BwBackupPartsInternal {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param([Parameter(Mandatory)]$Snapshot,[AllowNull()][string]$CanonicalText)
    if (-not $PSCmdlet.ShouldProcess('Exact observed backup generation','Retire known parts without touching unrelated notes')) { return $false }
    $changed = $false
    for ($index = 0; $index -lt $Snapshot.PartNames.Count; $index++) {
        $current = Get-BwNoteTextInternal -Name $script:BwSecureNoteName
        if ([string]$current -cne [string]$CanonicalText) {
            if ($changed -or [string]::IsNullOrEmpty($CanonicalText) -or [string]::IsNullOrEmpty($current)) { return $false }
            $before = Get-BwBackupManifestInternal -CanonicalText $CanonicalText
            $after = Get-BwBackupManifestInternal -CanonicalText $current
            if ($before.generation -ceq $after.generation) { return $false }
            $observed = Read-BwBackupCommittedInternal -CanonicalText $current -RetryLimit 0
            $CanonicalText = $observed.CanonicalText
            $current = $CanonicalText
            $changed = $true
        }
        if ($current -and $current.StartsWith($script:BwBackupPrefix,[StringComparison]::Ordinal)) {
            $active = Get-BwBackupManifestInternal -CanonicalText $current
            if ($active.generation -ceq $Snapshot.Manifest.generation) { return $false }
        }
        if (-not (Remove-BwNoteTextInternal -Name $Snapshot.PartNames[$index] -ExpectedText $Snapshot.PartTexts[$index] -Confirm:$false)) { return $false }
    }
    return $true
}

function Export-SecretsToBwVault {
    <#
    .SYNOPSIS
    Commit a generation-bound backup after all immutable parts validate.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param([switch]$Force)
    if (-not $PSCmdlet.ShouldProcess($script:BwSecureNoteName,'Export selected machine secrets')) { return $false }
    if (-not (Test-BwSession)) { Write-Error 'BW vault is locked. Unlock it before exporting.'; return $false }
    if (-not $Force -and (Get-CacheStatus).IsExpired) { throw 'Expired backup requires explicit Force acknowledgement.' }
    $secrets = Get-BwsSecretsList
    if (-not $secrets) { $secrets = Get-SecretsFromCache -IgnoreAge }
    if (-not $secrets -or $secrets.Count -eq 0) { Write-Error 'No secrets to export'; return $false }
    $generation = [guid]::NewGuid().ToString('N')
    $attemptedNames = [Collections.Generic.List[string]]::new()
    $attemptedTexts = [Collections.Generic.List[string]]::new()
    $previous = $null
    $previousText = $null
    $bytes = $null
    $committed = $false
    try {
        $previousText = Get-BwNoteTextInternal -Name $script:BwSecureNoteName
        if ($previousText -and $previousText.StartsWith($script:BwBackupPrefix,[StringComparison]::Ordinal)) {
            $previous = Read-BwBackupCommittedInternal -CanonicalText $previousText
            $previousText = $previous.CanonicalText
        }
        $payload = @{exported=[DateTime]::UtcNow.ToString('o');machine=$MachineName;machineId=(Get-MachineIdentityInternal).machineId;secretCount=$secrets.Count;secrets=$secrets} | ConvertTo-Json -Compress -Depth 20
        $bytes = [Text.UTF8Encoding]::new($false,$true).GetBytes($payload)
        if ($bytes.Length -lt 1 -or $bytes.Length -gt $script:BwBackupMaxRawLength) { throw 'Backup payload exceeds the admitted byte bound.' }
        $encoded = [Convert]::ToBase64String($bytes)
        $count = [int][Math]::Ceiling($encoded.Length / [double]$script:BwBackupChunkSize)
        $manifest = [ordered]@{version=2;machine=$MachineName;generation=$generation;sha256=(Get-BwBackupHashInternal $bytes);encoding='utf8-base64';partCount=$count;encodedLength=$encoded.Length;rawLength=$bytes.Length}
        $canonical = $script:BwBackupPrefix + ($manifest | ConvertTo-Json -Compress)
        $null = Get-BwBackupManifestInternal -CanonicalText $canonical
        for ($index = 1; $index -le $count; $index++) {
            $name = Get-BwBackupPartNameInternal $generation $index
            if ($null -ne (Get-BwNoteTextInternal -Name $name)) { throw 'Unique backup part namespace is already occupied.' }
            $offset = ($index - 1) * $script:BwBackupChunkSize
            $body = (Get-BwBackupPartHeaderInternal $generation $index $count) + $encoded.Substring($offset,[Math]::Min($script:BwBackupChunkSize,$encoded.Length - $offset))
            $attemptedNames.Add($name)
            $attemptedTexts.Add($body)
            if (-not (Set-BwNoteTextInternal -Name $name -Text $body -Confirm:$false)) { throw 'Backup part publication failed.' }
        }
        $null = Read-BwBackupSnapshotInternal -CanonicalText $canonical
        if ((Get-BwNoteTextInternal -Name $script:BwSecureNoteName) -cne $previousText) { throw 'Another backup writer changed the canonical commit.' }
        if (-not (Set-BwNoteTextInternal -Name $script:BwSecureNoteName -Text $canonical -Confirm:$false)) { throw 'Backup commit publication failed.' }
        $readback = Read-BwBackupCommittedInternal -CanonicalText $canonical
        if ($readback.Manifest.generation -cne $generation -or $readback.CanonicalText -cne $canonical) { throw 'Another backup generation became current.' }
        $committed = $true
        if ($previous) {
            try { $cleanupConfirmed = Remove-BwBackupPartsInternal -Snapshot $previous -CanonicalText $canonical -Confirm:$false }
            catch { $cleanupConfirmed = $false }
            if (-not $cleanupConfirmed) {
                Write-Warning 'Backup committed; prior generation cleanup was not confirmed. Unknown or concurrent data was preserved.'
            }
        }
        Invoke-MachineBitwardenCommand sync | Out-Null
        return $true
    } catch {
        if (-not $committed -and $attemptedNames.Count -gt 0) {
            try {
                $own = [pscustomobject]@{Manifest=[pscustomobject]@{generation=$generation};PartNames=$attemptedNames.ToArray();PartTexts=$attemptedTexts.ToArray()}
                $null = Remove-BwBackupPartsInternal -Snapshot $own -CanonicalText $previousText -Confirm:$false
            } catch { Write-Verbose 'Uncommitted backup cleanup was not confirmed; exact attempted parts are preserved.' }
        }
        Write-Error 'Backup export failed; a coherent current generation is required. Private provider details are withheld.'
        return $false
    } finally {
        if ($bytes) { [Array]::Clear($bytes,0,$bytes.Length) }
        $payload = $null
        $encoded = $null
        $secrets = $null
    }
}

function Import-SecretsFromBwVault {
    <#
    .SYNOPSIS
    Restore a fully validated backup before publishing cache or environment.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param([switch]$SetEnvironment)
    if (-not $PSCmdlet.ShouldProcess($script:BwSecureNoteName,'Restore selected machine secrets')) { return $null }
    if (-not (Test-BwSession)) { Write-Error 'BW vault is locked. Unlock it before restoring.'; return $null }
    $noteText = Get-BwNoteTextInternal -Name $script:BwSecureNoteName
    if ([string]::IsNullOrEmpty($noteText)) { Write-Error 'Selected backup was not found.'; return $null }
    # Legacy multipart has no generation integrity and cannot prove a coherent
    # snapshot. The operator explicitly selected one-time re-export instead.
    if ($noteText.StartsWith('#chunked',[StringComparison]::Ordinal)) {
        throw [IO.InvalidDataException]::new('Legacy multipart backup cannot be verified. Re-export once using Export-SecretsToBwVault, then retry recovery; cache and environment were not changed.')
    }
    try {
        if ($noteText.StartsWith($script:BwBackupPrefix,[StringComparison]::Ordinal)) {
            $snapshot = Read-BwBackupCommittedInternal -CanonicalText $noteText
            $secrets = $snapshot.Secrets
        } else {
            # Legitimate historical single-note JSON remains readable. It is not
            # promoted to a v2 integrity claim merely because JSON parses.
            $document = $null
            try {
                $document = [System.Text.Json.JsonDocument]::Parse($noteText)
                $backup = ConvertFrom-BwBackupElementInternal -Element $document.RootElement
            } finally { if ($document) { $document.Dispose() } }
            if ($backup -isnot [hashtable] -or $backup.secrets -isnot [hashtable] -or
                ($backup.ContainsKey('machine') -and $backup.machine -cne $MachineName)) { throw 'Legacy single-note backup schema or machine mismatch.' }
            $secrets = $backup.secrets
        }
        Save-SecretsToCache -Secrets $secrets
        if ($SetEnvironment) { Set-SecretsAsEnvironment -Secrets $secrets }
        return $secrets
    } catch {
        Write-Error 'Backup recovery failed validation or publication; private provider details are withheld.'
        return $null
    }
}

#endregion

#region Environment Variable Functions

function Set-SecretsAsEnvironment {
    <#
    .SYNOPSIS
    Set explicitly supplied values in a confirmed environment scope.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param([Parameter(Mandatory)][hashtable]$Secrets,[ValidateSet('Process','User')][string]$Scope='Process')
    foreach($key in $Secrets.Keys){if([string]::IsNullOrWhiteSpace([string]$key)-or[string]$key-match'[=\x00]'){throw 'Invalid environment variable name.'}}
    foreach($key in $Secrets.Keys){
        if($PSCmdlet.ShouldProcess("$Scope environment variable $key",'Set explicitly selected secret')){
            [Environment]::SetEnvironmentVariable([string]$key,[string]$Secrets[$key],$Scope)
        }
    }
}

function Set-PersistentSecrets {
    <#
    .SYNOPSIS
    Persist only explicitly selected cached secret names.
    #>
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns','',Justification='Existing public selected-key API; retain callers while guarding mutation.')]
    [CmdletBinding(SupportsShouldProcess)]
    param([Parameter(Mandatory)][ValidateNotNullOrEmpty()][string[]]$SecretNames)
    if(-not$PSCmdlet.ShouldProcess('Explicitly selected User variables','Read cache and persist selected values')){return}
    $secrets=Get-SecretsFromCache -IgnoreAge
    if(-not$secrets){throw 'No local secrets are available for selected-key persistence.'}
    $selected=@{}
    foreach($name in $SecretNames){
        if(-not$secrets.ContainsKey($name)){throw 'A requested secret name is missing; no User variables were changed.'}
        $selected[$name]=$secrets[$name]
    }
    Set-SecretsAsEnvironment -Secrets $selected -Scope User -Confirm:$false
}

#endregion

#region Main Orchestration

function Set-InitializedSecretsEnvironment {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory)][hashtable]$Secrets,
        [string[]]$PersistentKeys = @()
    )

    if (-not $PSCmdlet.ShouldProcess('process and requested User variables', 'Initialize machine secrets')) { return }
    Set-SecretsAsEnvironment -Secrets $Secrets -Scope Process
    foreach ($key in $PersistentKeys) {
        if ($Secrets.ContainsKey($key)) {
            [Environment]::SetEnvironmentVariable($key, $Secrets[$key], 'User')
        }
    }
}

function Initialize-MachineSecretsTiered {
    <#
    .SYNOPSIS
    Main entry point - initializes secrets using three-tier fallback
    .DESCRIPTION
    Called at profile load. Tries BWS first, then cache, then warns about BW recovery.
    .PARAMETER PersistentKeys
    Optional list of secret keys to also set as persistent User variables
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [string[]]$PersistentKeys = @()
    )
    if (-not $PSCmdlet.ShouldProcess('Initialize-MachineSecretsTiered state', 'Apply requested backend operation')) { return [pscustomobject]@{Success=$false;Source='not-applied';SecretCount=0;Warnings=@()} }


    $result = @{
        Source = $null
        SecretCount = 0
        Success = $false
        Warnings = @()
    }

    # Tier 1: Try BWS Cloud
    Write-Verbose "Tier 1: Attempting BWS Cloud..."
    if (-not $env:BWS_ACCESS_TOKEN -and (Get-Command Resolve-BwsTokenFromBackends -ErrorAction SilentlyContinue)) {
        try {
            [void](Resolve-BwsTokenFromBackends -Quiet)
        } catch {
            $result.Warnings += 'BWS token resolution failed; trying the local cache.'
        }
    }
    if ($env:BWS_ACCESS_TOKEN -and (Test-InternetConnection)) {
        $secrets = Get-BwsSecretsList
        if ($secrets -and $secrets.Count -gt 0) {
            Save-SecretsToCache -Secrets $secrets
            Set-InitializedSecretsEnvironment -Secrets $secrets -PersistentKeys $PersistentKeys
            $result.Source = "BWS Cloud"
            $result.SecretCount = $secrets.Count
            $result.Success = $true

            return [PSCustomObject]$result
        }
    }

    # Tier 2: Try Local Cache
    Write-Verbose "Tier 2: Attempting local cache..."
    $cacheStatus = Get-CacheStatus
    if ($cacheStatus.IsValid) {
        $secrets = Get-SecretsFromCache
        if ($secrets -and $secrets.Count -gt 0) {
            Set-InitializedSecretsEnvironment -Secrets $secrets -PersistentKeys $PersistentKeys
            $result.Source = "Local Cache"
            $result.SecretCount = $secrets.Count
            $result.Success = $true
            $result.Warnings += "Using cached secrets from $($cacheStatus.LastUpdated.ToString('yyyy-MM-dd HH:mm'))"
            return [PSCustomObject]$result
        }
    } elseif ($cacheStatus.Exists -and $cacheStatus.IsExpired) {
        # Cache exists but expired - use it anyway with warning
        $secrets = Get-SecretsFromCache -IgnoreAge
        if ($secrets -and $secrets.Count -gt 0) {
            Set-InitializedSecretsEnvironment -Secrets $secrets -PersistentKeys $PersistentKeys
            $result.Source = "Local Cache (EXPIRED)"
            $result.SecretCount = $secrets.Count
            $result.Success = $true
            $result.Warnings += "WARNING: Cache is $([int]$cacheStatus.Age.TotalDays) days old!"
            $result.Warnings += "Run 'Sync-MachineSecrets' when online to refresh"
            return [PSCustomObject]$result
        }
    }

    # Tier 3: Manual BW Recovery Required
    $result.Source = "NONE"
    $result.Success = $false
    $result.Warnings += "No secrets available!"
    $result.Warnings += "Run 'Import-SecretsFromBwVault' after unlocking BW vault"

    return [PSCustomObject]$result
}

function Get-SecretsStatus {
    <#
    .SYNOPSIS
    Return metadata without fetching secrets or hydrating environment.
    #>
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param()
    [pscustomobject]@{MachineName=$MachineName;BackupNote=$script:BwSecureNoteName;Cache=Get-CacheStatus;BwsTokenInProcess=[bool]$env:BWS_ACCESS_TOKEN;BwStatus=(Get-BwStatusSafe).status}
}

function Sync-MachineSecrets {
    <#
    .SYNOPSIS
    Force synchronization from BWS Cloud
    #>
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns','',Justification='Existing compatibility API Sync-MachineSecrets is referenced by installed profiles or sibling backend code; preserve its exact name.')]
    [CmdletBinding(SupportsShouldProcess)]
    param()
    if (-not $PSCmdlet.ShouldProcess('Sync-MachineSecrets state', 'Apply requested backend operation')) { return $null }


    if (Sync-SecretsFromBws -Force) {
        $secrets = Get-SecretsFromCache
        Set-SecretsAsEnvironment -Secrets $secrets -Scope Process
        Write-Verbose "Secrets synced and loaded into environment"
    } else {
        Write-Error "Sync failed. Check internet connection and BWS token."
    }
}

#endregion

Set-Alias -Name Ensure-LocalSecretVaultRegistration -Value Register-LocalSecretVault -Scope Script
# Export functions
Export-ModuleMember -Alias Ensure-LocalSecretVaultRegistration -Function @(
    # Secret backend integration
    'Get-PreferredSecretVaultName',
    'Register-LocalSecretVault',
    'Get-SecretManagementStatus',
    'Get-SecretBackendStatus',
    'Initialize-LocalSecretBackends',
    'Resolve-BwsTokenFromBackends',
    'Resolve-BwBootstrapState',
    'Unlock-BitwardenCliSession',
    'Initialize-BitwardenSessionFromBackends',
    'Publish-ConnectionCredential',
    'Get-ConnectionCredential',

    # Tier 1
    'Get-BwsSecretsList',
    'Sync-SecretsFromBws',

    # Tier 2
    'Save-SecretsToCache',
    'Get-SecretsFromCache',
    'Get-CacheStatus',

    # Tier 3
    'Test-BwSession',
    'Export-SecretsToBwVault',
    'Import-SecretsFromBwVault',
    'Initialize-BitwardenSession',

    # Local backends
    'Get-CredentialManagerCredential',
    'Set-CredentialManagerCredential',
    'Get-ResourceCredentialManifest',
    'Get-ResourceCredentialReconciliationReport',
    'Sync-ResourceCredentialTargetsFromVault',
    'Sync-ResourceCredentialsFromBitwarden',
    'Repair-ResourceDriveMappings',

    # Environment
    'Set-SecretsAsEnvironment',
    'Set-PersistentSecrets',

    # Main
    'Initialize-MachineSecretsTiered',
    'Get-SecretsStatus',
    'Sync-MachineSecrets'
)
