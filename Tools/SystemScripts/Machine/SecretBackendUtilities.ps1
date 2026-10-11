if(-not('WorkProfileBackend.SecureSecretArgumentAttribute'-as[type])){
    Add-Type -TypeDefinition @'
using System;
using System.Security;
using System.Management.Automation;
using System.Runtime.InteropServices;
namespace WorkProfileBackend {
    public sealed class SecureSecretArgumentAttribute : ArgumentTransformationAttribute {
        public override object Transform(EngineIntrinsics engine, object input) {
            if(input is PSObject wrapper) input = wrapper.BaseObject;
            if(input is SecureString secure) return secure;
            if(input is string text) {
                var result=new SecureString();
                foreach(char c in text) result.AppendChar(c);
                result.MakeReadOnly();
                return result;
            }
            throw new ArgumentTransformationMetadataException("Credential input must be SecureString or a legacy string.");
        }
    }
    public static class NativeCredential {
        [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
        private struct Credential {
            public uint Flags, Type;
            public string TargetName, Comment;
            public long LastWritten;
            public uint CredentialBlobSize;
            public IntPtr CredentialBlob;
            public uint Persist, AttributeCount;
            public IntPtr Attributes;
            public string TargetAlias, UserName;
        }
        [DllImport("advapi32.dll", EntryPoint="CredWriteW", CharSet=CharSet.Unicode, SetLastError=true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool CredWrite(ref Credential credential, uint flags);
        public static void Write(string target, string user, SecureString password, bool domain) {
            if(!OperatingSystem.IsWindows() || IntPtr.Size!=8) throw new PlatformNotSupportedException("The reviewed credential ABI requires Windows x64.");
            if(password==null || password.Length>1280) throw new ArgumentException("Credential blob is absent or too large.");
            if(String.IsNullOrWhiteSpace(target) || target.Length>(domain?337:32767)) throw new ArgumentException("Credential target is invalid.");
            if(String.IsNullOrWhiteSpace(user)) throw new ArgumentException("Credential user is required.");
            IntPtr secret=Marshal.SecureStringToGlobalAllocUnicode(password);
            try {
                var credential=new Credential { Type=domain?2u:1u, TargetName=target, UserName=user,
                    CredentialBlob=secret, CredentialBlobSize=(uint)password.Length*2u, Persist=2u };
                if(!CredWrite(ref credential,0))
                    throw new InvalidOperationException("Windows credential write failed with code "+Marshal.GetLastWin32Error()+".");
            } finally { Marshal.ZeroFreeGlobalAllocUnicode(secret); }
        }
    }
}
'@
}

function Write-NativeWindowsCredential {
    [CmdletBinding(SupportsShouldProcess)]
    param([Parameter(Mandatory)][string]$Target,[Parameter(Mandatory)][string]$UserName,[Parameter(Mandatory)][WorkProfileBackend.SecureSecretArgument()][Security.SecureString]$Password,[ValidateSet('generic','domain')][string]$Kind='generic')
    if($PSCmdlet.ShouldProcess($Target,'Write Windows credential through protected native memory')){
        [WorkProfileBackend.NativeCredential]::Write($Target,$UserName,$Password,($Kind-eq'domain'))
    }
}

Set-StrictMode -Version 3.0

function Import-LocalSecretManagement {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [switch]$InstallMissing
    )

    $state = [ordered]@{
        SecretManagementAvailable = $false
        SecretStoreAvailable = $false
        CredManStoreAvailable = $false
        CredentialManagerAvailable = $false
        BitwardenVaultAvailable = $false
        Imported = @()
    }

    $moduleSpecs = @(
        @{ Name = 'Microsoft.PowerShell.SecretManagement'; InstallName = 'Microsoft.PowerShell.SecretManagement' },
        @{ Name = 'Microsoft.PowerShell.SecretStore'; InstallName = 'Microsoft.PowerShell.SecretStore' },
        @{ Name = 'Microsoft.PowerShell.CredManStore'; InstallName = 'Microsoft.PowerShell.CredManStore' },
        @{ Name = 'SecretManagement.BitWarden'; InstallName = 'SecretManagement.BitWarden' }
    )

    foreach ($spec in $moduleSpecs) {
        $available = Get-Module -ListAvailable -Name $spec.Name | Sort-Object Version -Descending | Select-Object -First 1
        if (-not $available -and $InstallMissing -and $PSCmdlet.ShouldProcess($spec.Name, 'Install selected secret provider')) {
            try {
                Install-Module -Name $spec.InstallName -Repository PSGallery -Scope CurrentUser -Force -AllowClobber -AcceptLicense -ErrorAction Stop | Out-Null
                $available = Get-Module -ListAvailable -Name $spec.Name | Sort-Object Version -Descending | Select-Object -First 1
            } catch { Write-Verbose 'Optional backend operation failed in Import-LocalSecretManagement; private provider output is withheld.' }
        }

        if ($available) {
            try {
                Import-Module $available.Path -Force -ErrorAction Stop | Out-Null
                $state.Imported += $spec.Name
            } catch { Write-Verbose 'Optional backend operation failed in Import-LocalSecretManagement; private provider output is withheld.' }
        }
    }

    $credentialManagerModule = Get-Module -ListAvailable -Name CredentialManager | Sort-Object Version -Descending | Select-Object -First 1
    if ($credentialManagerModule) {
        try {
            Import-Module $credentialManagerModule.Path -Force -ErrorAction Stop | Out-Null
            $state.Imported += 'CredentialManager'
        } catch { Write-Verbose 'Optional backend operation failed in Import-LocalSecretManagement; private provider output is withheld.' }
    }

    $state.SecretManagementAvailable = [bool](Get-Module -ListAvailable -Name Microsoft.PowerShell.SecretManagement)
    $state.SecretStoreAvailable = [bool](Get-Module -ListAvailable -Name Microsoft.PowerShell.SecretStore)
    $state.CredManStoreAvailable = [bool](Get-Module -ListAvailable -Name Microsoft.PowerShell.CredManStore)
    $state.CredentialManagerAvailable = [bool](Get-Module -ListAvailable -Name CredentialManager)
    $state.BitwardenVaultAvailable = [bool](Get-Module -ListAvailable -Name SecretManagement.BitWarden)

    [pscustomobject]$state
}

function Get-BitwardenPasswordFileCandidates {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns','',Justification='Existing compatibility API Get-BitwardenPasswordFileCandidates is referenced by installed profiles or sibling backend code; preserve its exact name.')]
    [OutputType([array])]
    [CmdletBinding()]
    param()

    @(
        (Join-Path $env:USERPROFILE '.bwdata\bw_pass.txt'),
        (Join-Path $env:USERPROFILE '.bw\bw_pass.txt')
    )
}

function Get-BitwardenPasswordFile {
    foreach ($candidate in Get-BitwardenPasswordFileCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    $null
}

function Get-BitwardenSessionFileCandidates {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns','',Justification='Existing compatibility API Get-BitwardenSessionFileCandidates is referenced by installed profiles or sibling backend code; preserve its exact name.')]
    [OutputType([array])]
    [CmdletBinding()]
    param()

    @(
        (Join-Path $env:USERPROFILE '.bwdata\session.txt'),
        (Join-Path $env:USERPROFILE '.bw\session.txt')
    )
}

function Get-BitwardenSessionFile {
    foreach ($candidate in Get-BitwardenSessionFileCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    (Get-BitwardenSessionFileCandidates | Select-Object -First 1)
}

function Get-BitwardenCliProcessSpec {
    [CmdletBinding()]
    param(
        [string]$NativeExecutablePath = $env:PCAI_BW_EXECUTABLE,
        [string]$NativeExecutableSha256 = $env:PCAI_BW_EXECUTABLE_SHA256
    )

    # Explicit fallback requires a fresh installed-file hash. Preserve normal
    # native command selection when no override was requested.
    if (-not [string]::IsNullOrWhiteSpace($NativeExecutablePath)) {
        $actualPackage = Get-RealWinGetPackageExecutable -Tool bw -ExecutablePath $NativeExecutablePath -ExpectedSha256 $NativeExecutableSha256
        return [pscustomobject]@{FilePath=$actualPackage;ArgumentPrefix=@();DisplayPath=$actualPackage;Kind='native-executable'}
    }
    # Prefer the installed native CLI; an old bw.cmd may outlive its npm package.
    $nativeBw = Get-Command bw.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($nativeBw) {
        return [pscustomobject]@{
            FilePath = $nativeBw.Source
            ArgumentPrefix = @()
            DisplayPath = $nativeBw.Source
            Kind = 'native-executable'
        }
    }

    # Discover an actual package only when normal native discovery is absent.
    $actualPackage = Get-RealWinGetPackageExecutable -Tool bw
    if ($actualPackage) {
        return [pscustomobject]@{FilePath=$actualPackage;ArgumentPrefix=@();DisplayPath=$actualPackage;Kind='native-executable'}
    }
    $npmRoot = Join-Path $env:APPDATA 'npm'
    $bwJsPath = Join-Path $npmRoot 'node_modules\@bitwarden\cli\build\bw.js'
    $nodeCommand = Get-Command node -CommandType Application -ErrorAction SilentlyContinue
    if ($nodeCommand -and (Test-Path -LiteralPath $bwJsPath)) {
        return [pscustomobject]@{
            FilePath = $nodeCommand.Source
            ArgumentPrefix = @($bwJsPath)
            DisplayPath = $bwJsPath
            Kind = 'node-script'
        }
    }

    $homeBinBwCmd = Join-Path $env:USERPROFILE 'bin\bw.cmd'
    if (Test-Path -LiteralPath $homeBinBwCmd) {
        return [pscustomobject]@{
            FilePath = $homeBinBwCmd
            ArgumentPrefix = @()
            DisplayPath = $homeBinBwCmd
            Kind = 'home-bin-cmd'
        }
    }

    $bwCmdPath = Join-Path $npmRoot 'bw.cmd'
    if (Test-Path -LiteralPath $bwCmdPath) {
        return [pscustomobject]@{
            FilePath = $bwCmdPath
            ArgumentPrefix = @()
            DisplayPath = $bwCmdPath
            Kind = 'cmd-shim'
        }
    }

    $bwCommand = Get-Command bw -CommandType Application -ErrorAction SilentlyContinue
    if ($bwCommand) {
        return [pscustomobject]@{
            FilePath = $bwCommand.Source
            ArgumentPrefix = @()
            DisplayPath = $bwCommand.Source
            Kind = 'command'
        }
    }

    $null
}

function ConvertTo-ProcessArgumentString {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string[]]$Arguments
    )

    $escaped = foreach ($argument in @($Arguments)) {
        if ($null -eq $argument) { continue }

        $value = [string]$argument
        if ($value -notmatch '[\s"]') {
            $value
            continue
        }

        '"' + (($value -replace '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1') + '"'
    }

    @($escaped) -join ' '
}

function Invoke-BitwardenCli {
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param([Parameter(Mandatory)][string[]]$Arguments,[ValidateRange(1,120)][int]$TimeoutSeconds=20,[string]$StandardInput)
    $spec=Get-BitwardenCliProcessSpec
    if(-not$spec){return [pscustomobject]@{Success=$false;TimedOut=$false;ExitCode=$null;StdOut=$null;StdErr='bw CLI is not available'}}
    if($spec.Kind-in@('home-bin-cmd','cmd-shim','command')){
        throw [InvalidOperationException]::new('Shell-based Bitwarden launch is refused; install native CLI or select an explicit hash-attested native package.')
    }
    $parameters=@{FilePath=$spec.FilePath;Arguments=@($spec.ArgumentPrefix)+$Arguments;TimeoutSeconds=$TimeoutSeconds}
    if($PSBoundParameters.ContainsKey('StandardInput')){$parameters.StandardInput=$StandardInput}
    Invoke-SecretBackendProcess @parameters
}

function Get-BwStatusSafe {
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param()
    try{
        $response=Invoke-BitwardenCli -Arguments @('status') -TimeoutSeconds 15
        if(-not$response.Success){return [pscustomobject]@{status=if($response.StdErr-eq'bw CLI is not available'){'missing'}elseif($response.TimedOut){'timeout'}else{'unknown'};ExitCode=$response.ExitCode}}
        $parsed=$response.StdOut|ConvertFrom-SecretBackendJson -ErrorAction Stop
        if($parsed.status-notin@('locked','unlocked','unauthenticated')){throw [FormatException]::new('Unexpected status metadata.')}
        return [pscustomobject]@{status=[string]$parsed.status;ExitCode=$response.ExitCode}
    }catch{
        Write-Verbose 'Bitwarden status metadata could not be validated; private response is withheld.'
        return [pscustomobject]@{status='error';ExitCode=$null}
    }
}

function Get-PlainTextSecretFromCandidateVaults {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns','',Justification='Existing compatibility API Get-PlainTextSecretFromCandidateVaults is referenced by installed profiles or sibling backend code; preserve its exact name.')]
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Name,
        [string[]]$VaultNames = @('LocalStore', 'SecretStore', 'CredMan', 'Bitwarden')
    )

    if (-not (Get-Command Microsoft.PowerShell.SecretManagement\Get-Secret -ErrorAction SilentlyContinue)) {
        return $null
    }

    foreach ($vaultName in @($VaultNames)) {
        if ([string]::IsNullOrWhiteSpace($vaultName)) { continue }
        try {
            $value = Microsoft.PowerShell.SecretManagement\Get-Secret -Name $Name -Vault $vaultName -AsPlainText -ErrorAction Stop -WarningAction SilentlyContinue
            if (-not [string]::IsNullOrWhiteSpace([string]$value)) {
                return [pscustomobject]@{
                    Name = $Name
                    Vault = $vaultName
                    Value = [string]$value
                }
            }
        } catch { Write-Verbose 'Optional backend operation failed in Get-PlainTextSecretFromCandidateVaults; private provider output is withheld.' }
    }

    $null
}

function Resolve-BwBootstrapState {
    [CmdletBinding()]
    param()

    [void](Import-LocalSecretManagement)

    $result = [ordered]@{
        Password = $null
        PasswordSource = $null
        ClientId = $null
        ClientIdSource = $null
        ClientSecret = $null
        ClientSecretSource = $null
        PasswordFile = (Get-BitwardenPasswordFile)
        SessionFile = (Get-BitwardenSessionFile)
    }

    $passwordSecret = Get-PlainTextSecretFromCandidateVaults -Name 'bitwarden/master_password'
    if ($passwordSecret) {
        $result.Password = $passwordSecret.Value
        $result.PasswordSource = "vault:$($passwordSecret.Vault)"
    } elseif ($result.PasswordFile -and (Test-Path -LiteralPath $result.PasswordFile)) {
        try {
            Assert-BitwardenPrivateInput -Path $result.PasswordFile
            $result.Password = (Get-Content -LiteralPath $result.PasswordFile -Raw -ErrorAction Stop).Trim()
            if ($result.Password) {
                $result.PasswordSource = "file:$($result.PasswordFile)"
            }
        } catch { Write-Warning "BW password file exists but is unreadable ($($result.PasswordFile)): failure details withheld" }
    }

    # API credentials are admitted as a complete pair from one source. Partial
    # vault/file/process inputs must never be combined into a different identity.
    foreach ($vaultName in @('LocalStore', 'SecretStore', 'CredMan', 'Bitwarden')) {
        $clientIdSecret = Get-PlainTextSecretFromCandidateVaults -Name 'bitwarden/client_id' -VaultNames @($vaultName)
        $clientSecretSecret = Get-PlainTextSecretFromCandidateVaults -Name 'bitwarden/client_secret' -VaultNames @($vaultName)
        if ($clientIdSecret -and $clientSecretSecret) {
            $result.ClientId = $clientIdSecret.Value
            $result.ClientSecret = $clientSecretSecret.Value
            $result.ClientIdSource = "vault:$vaultName"
            $result.ClientSecretSource = "vault:$vaultName"
            break
        }
    }

    $apiSecretsPath = Join-Path $env:USERPROFILE '.bwdata\secrets.json'
    if (-not $result.ClientId -and (Test-Path -LiteralPath $apiSecretsPath)) {
        try {
            Assert-BitwardenPrivateInput -Path $apiSecretsPath
            $apiSecrets = Get-Content -LiteralPath $apiSecretsPath -Raw | ConvertFrom-SecretBackendJson -ErrorAction Stop
            if (-not [string]::IsNullOrWhiteSpace([string]$apiSecrets.client_id) -and -not [string]::IsNullOrWhiteSpace([string]$apiSecrets.client_secret)) {
                $result.ClientId = [string]$apiSecrets.client_id
                $result.ClientSecret = [string]$apiSecrets.client_secret
                $result.ClientIdSource = "file:$apiSecretsPath"
                $result.ClientSecretSource = "file:$apiSecretsPath"
            }
        } catch { Write-Warning "BW API secrets file exists but failed to qualify a complete pair ($apiSecretsPath): failure details withheld" }
    }

    if (-not $result.ClientId -and -not [string]::IsNullOrWhiteSpace($env:BW_CLIENTID) -and -not [string]::IsNullOrWhiteSpace($env:BW_CLIENTSECRET)) {
        $result.ClientId = $env:BW_CLIENTID
        $result.ClientSecret = $env:BW_CLIENTSECRET
        $result.ClientIdSource = 'process:BW_CLIENTID'
        $result.ClientSecretSource = 'process:BW_CLIENTSECRET'
    }

    [pscustomobject]$result
}

function Set-PlainTextSecretInSecretManagement {
    [OutputType([bool])]
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Secret,
        [string]$VaultName = 'LocalStore'
    )
    if (-not $PSCmdlet.ShouldProcess('Set-PlainTextSecretInSecretManagement state', 'Apply requested backend operation')) { return $false }


    $state = Import-LocalSecretManagement
    if (-not $state.SecretManagementAvailable) {
        return $false
    }

    try {
        if (-not (Microsoft.PowerShell.SecretManagement\Get-SecretVault -Name $VaultName -ErrorAction SilentlyContinue -WarningAction SilentlyContinue)) {
            return $false
        }
        Microsoft.PowerShell.SecretManagement\Set-Secret -Name $Name -Vault $VaultName -Secret $Secret -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Initialize-CredentialFileInterop {
    if ('Pcai.Credential.FileV1' -as [type]) {
        if ([Pcai.Credential.FileV1]::ProtocolVersion -ne 1) { throw 'Unknown credential file interop version.' }
        return
    }
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
namespace Pcai.Credential {
 public static class FileV1 {
  public const int ProtocolVersion = 1;
  [StructLayout(LayoutKind.Sequential)] private struct FileIdInfo { public ulong Volume, Low, High; }
  [StructLayout(LayoutKind.Sequential)] private struct FileStandardInfo { public long AllocationSize, EndOfFile; public uint Links; public byte DeletePending, Directory; }
  [DllImport("kernel32.dll", SetLastError=true, ExactSpelling=true)] [return: MarshalAs(UnmanagedType.Bool)]
  private static extern bool GetFileInformationByHandleEx(SafeFileHandle handle, int information, out FileIdInfo identity, uint size);
  [DllImport("kernel32.dll", SetLastError=true, ExactSpelling=true)] [return: MarshalAs(UnmanagedType.Bool)]
  private static extern bool GetFileInformationByHandleEx(SafeFileHandle handle, int information, out FileStandardInfo standard, uint size);
  [DllImport("ntdll.dll", ExactSpelling=true)] private static extern int NtQuerySecurityObject(SafeFileHandle handle, uint information, [Out] byte[] descriptor, uint length, out uint required);
  [DllImport("ntdll.dll", ExactSpelling=true)] private static extern int NtSetSecurityObject(SafeFileHandle handle, uint information, [In] byte[] descriptor);
  [DllImport("ntdll.dll", ExactSpelling=true)] private static extern uint RtlNtStatusToDosError(int status);
  public static string Identity(SafeFileHandle handle) {
   FileIdInfo identity;
   if (!GetFileInformationByHandleEx(handle, 18, out identity, 24)) throw new Win32Exception(Marshal.GetLastWin32Error());
   return identity.Volume.ToString("X16") + ":" + identity.High.ToString("X16") + identity.Low.ToString("X16");
  }
  public static uint Links(SafeFileHandle handle) {
   FileStandardInfo standard;
   if (!GetFileInformationByHandleEx(handle, 1, out standard, (uint)Marshal.SizeOf<FileStandardInfo>())) throw new Win32Exception(Marshal.GetLastWin32Error());
   if (standard.DeletePending != 0 || standard.Directory != 0) throw new InvalidOperationException("Unsafe file object.");
   return standard.Links;
  }
  public static byte[] ReadDescriptor(SafeFileHandle handle) {
   byte[] descriptor = new byte[65536]; uint required;
   int status = NtQuerySecurityObject(handle, 7, descriptor, (uint)descriptor.Length, out required);
   if (status != 0) throw new Win32Exception(unchecked((int)RtlNtStatusToDosError(status)));
   if (required < 20 || required > descriptor.Length) throw new InvalidOperationException("Invalid security descriptor length.");
   Array.Resize(ref descriptor, (int)required); return descriptor;
  }
  public static void WriteDescriptor(SafeFileHandle handle, byte[] descriptor) {
   int status = NtSetSecurityObject(handle, 7, descriptor);
   if (status != 0) throw new Win32Exception(unchecked((int)RtlNtStatusToDosError(status)));
  }
 }
}
'@
}

function Get-CredentialBytesHash {
    param([byte[]]$Bytes)
    $hash = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hash.ComputeHash($Bytes))).Replace('-','') }
    finally { $hash.Dispose() }
}

function Open-CredentialOwnedFile {
    param([string]$Path, [switch]$RestoreRights)
    $rights = [Security.AccessControl.FileSystemRights]'ReadData,ReadPermissions'
    if ($RestoreRights) { $rights = $rights -bor [Security.AccessControl.FileSystemRights]'ChangePermissions,TakeOwnership' }
    return [IO.FileSystemAclExtensions]::Create([IO.FileInfo]::new($Path), [IO.FileMode]::Open, $rights, [IO.FileShare]'ReadWrite,Delete', 4096, [IO.FileOptions]::None, $null)
}

function Read-CredentialOwnedSnapshot {
    param([IO.FileStream]$Stream)
    $position = $Stream.Position
    $hash = [Security.Cryptography.SHA256]::Create()
    try {
        $Stream.Position = 0
        $digest = ([BitConverter]::ToString($hash.ComputeHash($Stream))).Replace('-','')
        return [pscustomobject]@{Identity=[Pcai.Credential.FileV1]::Identity($Stream.SafeFileHandle);Hash=$digest;Bytes=[byte[]]@();Descriptor=[Pcai.Credential.FileV1]::ReadDescriptor($Stream.SafeFileHandle)}
    } finally { $Stream.Position = $position; $hash.Dispose() }
}

function Get-CredentialDescriptorSddl {
    param([byte[]]$Descriptor)
    return ([Security.AccessControl.RawSecurityDescriptor]::new($Descriptor,0)).GetSddlForm([Security.AccessControl.AccessControlSections]::All)
}

function Set-CredentialOwnedDescriptor {
    [CmdletBinding()]
    param([IO.FileStream]$Stream,[byte[]]$Descriptor)
    $raw = [Security.AccessControl.RawSecurityDescriptor]::new($Descriptor,0)
    if ($raw.ControlFlags -band [Security.AccessControl.ControlFlags]::DiscretionaryAclAutoInherited) {
        $raw.SetFlags($raw.ControlFlags -bor [Security.AccessControl.ControlFlags]::DiscretionaryAclAutoInheritRequired)
    }
    $submitted = [byte[]]::new($raw.BinaryLength)
    $raw.GetBinaryForm($submitted,0)
    [Pcai.Credential.FileV1]::WriteDescriptor($Stream.SafeFileHandle,$submitted)
    if ((Get-CredentialDescriptorSddl ([Pcai.Credential.FileV1]::ReadDescriptor($Stream.SafeFileHandle))) -cne (Get-CredentialDescriptorSddl $Descriptor)) {
        throw 'Exact owner/group/DACL restoration is unsupported.'
    }
}

function New-CredentialPrivateAcl {
    param([switch]$Directory)
    $acl = if ($Directory) { [Security.AccessControl.DirectorySecurity]::new() } else { [Security.AccessControl.FileSecurity]::new() }
    $acl.SetOwner([Security.Principal.WindowsIdentity]::GetCurrent().User)
    $acl.SetAccessRuleProtection($true,$false)
    $inheritance = if ($Directory) { [Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit' } else { [Security.AccessControl.InheritanceFlags]::None }
    foreach ($sid in @([Security.Principal.WindowsIdentity]::GetCurrent().User,[Security.Principal.SecurityIdentifier]::new('S-1-5-18'),[Security.Principal.SecurityIdentifier]::new('S-1-5-32-544'))) {
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($sid,[Security.AccessControl.FileSystemRights]::FullControl,$inheritance,[Security.AccessControl.PropagationFlags]::None,[Security.AccessControl.AccessControlType]::Allow))
    }
    return $acl
}

function Assert-CredentialPrivateDirectory {
    param([string]$Path,[switch]$ExistingOnly)
    if (-not [IO.Directory]::Exists($Path)) {
        if ($ExistingOnly) { throw 'Selected private directory does not exist.' }
        [IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new($Path),(New-CredentialPrivateAcl -Directory))
    }
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Unsafe private custody directory.' }
    $acl = Get-Acl -LiteralPath $Path -ErrorAction Stop
    $allowed = @([Security.Principal.WindowsIdentity]::GetCurrent().User.Value,'S-1-5-18','S-1-5-32-544')
    if (-not $acl.AreAccessRulesProtected -or $acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -cne $allowed[0]) { throw 'Private custody owner or protection mismatch.' }
    foreach ($rule in $acl.GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier])) {
        if ($rule.AccessControlType -eq [Security.AccessControl.AccessControlType]::Allow -and $rule.IdentityReference.Value -notin $allowed) { throw 'Private custody admits an unsupported principal.' }
    }
}

function New-CredentialPrivateStage {
    param([string]$Path,[byte[]]$Bytes)
    $stream = [IO.FileSystemAclExtensions]::Create([IO.FileInfo]::new($Path),[IO.FileMode]::CreateNew,[Security.AccessControl.FileSystemRights]'ReadData,WriteData,ReadPermissions,ChangePermissions,TakeOwnership',[IO.FileShare]'ReadWrite,Delete',4096,[IO.FileOptions]::None,(New-CredentialPrivateAcl))
    try { if ($Bytes) { $stream.Write($Bytes,0,$Bytes.Length) }; $stream.Flush($true); return $stream }
    catch { $stream.Dispose(); throw }
}

function Get-CredentialStreamPath {
    param([string]$Path)
    # The stream provider uses Win32 enumeration and needs extended long paths.
    $full=[IO.Path]::GetFullPath($Path)
    if ($full.StartsWith('\\?\',[StringComparison]::Ordinal)) { return $full }
    if ($full.StartsWith('\\',[StringComparison]::Ordinal)) { return '\\?\UNC\'+$full.Substring(2) }
    return '\\?\'+$full
}

function Assert-CredentialTrustedNamespace {
    param([string]$Parent)
    $trusted = @([Security.Principal.WindowsIdentity]::GetCurrent().User.Value,'S-1-5-18','S-1-5-32-544')
    # Windows' volume root is normally owned by this privileged servicing SID.
    # Resolve the service locally; an unknown service/owner fails closed.
    $trusted += ([Security.Principal.NTAccount]::new('NT SERVICE','TrustedInstaller')).Translate([Security.Principal.SecurityIdentifier]).Value
    $mutation = [Security.AccessControl.FileSystemRights]'WriteData,WriteExtendedAttributes,WriteAttributes,Delete,DeleteSubdirectoriesAndFiles,ChangePermissions,TakeOwnership'
    # WinNT generic WRITE and ALL grants also confer namespace mutation.
    $genericMutation = [uint32]0x50000000
    $cursor = (Get-Item -LiteralPath $Parent -Force -ErrorAction Stop).FullName
    $directParent = $true
    while ($cursor) {
        $directory = Get-Item -LiteralPath $cursor -Force -ErrorAction Stop
        if (-not $directory.PSIsContainer -or ($directory.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Unsupported credential namespace directory.' }
        $acl = Get-Acl -LiteralPath $directory.FullName -ErrorAction Stop
        $raw = [Security.AccessControl.RawSecurityDescriptor]::new($acl.GetSecurityDescriptorBinaryForm(),0)
        if ($null -eq $raw.Owner -or $raw.Owner.Value -notin $trusted -or $null -eq $raw.DiscretionaryAcl) { throw 'Untrusted credential namespace ownership or DACL.' }
        $mask = [uint32][int]$mutation
        # Ancestor-only AddSubdirectory cannot replace an already existing
        # selected child. The direct parent must also exclude that authority.
        if ($directParent) { $mask = $mask -bor [uint32][int][Security.AccessControl.FileSystemRights]::CreateDirectories }
        foreach ($ace in $raw.DiscretionaryAcl) {
            if ($ace -isnot [Security.AccessControl.CommonAce] -or $ace.IsCallback -or $ace.AceQualifier -notin @([Security.AccessControl.AceQualifier]::AccessAllowed,[Security.AccessControl.AceQualifier]::AccessDenied)) { throw 'Unsupported credential namespace DACL entry.' }
            if ($ace.AceFlags -band [Security.AccessControl.AceFlags]::InheritOnly) { continue }
            # Do not infer effective access by subtracting deny ACEs. Any
            # applicable untrusted allow is sufficient to refuse publication.
            $access = [BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$ace.AccessMask),0)
            if ($ace.AceQualifier -eq [Security.AccessControl.AceQualifier]::AccessAllowed -and $ace.SecurityIdentifier.Value -notin $trusted -and ($access -band ($mask -bor $genericMutation))) { throw 'Foreign credential namespace mutation authority refused.' }
        }
        $cursor = [IO.Path]::GetDirectoryName($directory.FullName)
        $directParent = $false
    }
}

function Resolve-CredentialPrivateStorage {
    [CmdletBinding()]
    param([string]$RootOverride,[ValidateSet('Bootstrap','Cache','Explicit')][string]$Purpose,[string]$MachineName)
    if (-not $IsWindows) { throw 'Qualified Windows credential storage required.' }
    $explicitRoot = -not [string]::IsNullOrWhiteSpace($RootOverride)
    $requested = if ($explicitRoot) { $RootOverride } else { Join-Path $env:USERPROFILE '.pcai-credential-state' }
    $provider=$null;$drive=$null
    $root=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($requested,[ref]$provider,[ref]$drive)
    if ($provider.Name -ne 'FileSystem') { throw 'Filesystem credential storage required.' }
    $root=[IO.Path]::GetFullPath($root)
    if ($root.Substring([IO.Path]::GetPathRoot($root).Length).Contains(':')) { throw 'Alternate credential storage stream refused.' }
    if ([IO.DriveInfo]::new([IO.Path]::GetPathRoot($root)).DriveFormat -cne 'NTFS') { throw 'Qualified NTFS credential storage required.' }
    if ($explicitRoot -and -not (Test-Path -LiteralPath $root) -and -not [IO.Directory]::Exists([IO.Path]::GetDirectoryName($root))) { throw 'Explicit credential root requires an existing admitted parent.' }
    $selected = switch($Purpose) {
        Bootstrap { Join-Path $root 'bootstrap' }
        Cache {
            if ([string]::IsNullOrWhiteSpace($MachineName)) { throw 'Explicit machine cache identity required.' }
            Join-Path (Join-Path $root 'cache') (Get-CredentialBytesHash ([Text.Encoding]::UTF8.GetBytes($MachineName.ToUpperInvariant())))
        }
        Explicit { $root }
    }
    $cursor=$selected;$missing=[Collections.Generic.List[string]]::new()
    while ($cursor -and -not (Test-Path -LiteralPath $cursor)) {
        if ([IO.Path]::GetFileName($cursor).TrimEnd(' ','.') -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.|$)') { throw 'Reserved credential storage path refused.' }
        $missing.Add($cursor);$cursor=[IO.Path]::GetDirectoryName($cursor)
    }
    if (-not $cursor) { throw 'Existing credential storage parent required.' }
    Assert-CredentialTrustedNamespace $cursor
    # An existing selected/root directory must be private; metadata admission
    # never repairs its ACL or provisions a path during import.
    foreach($path in @($root,$selected)) {
        if (Test-Path -LiteralPath $path) { Assert-CredentialPrivateDirectory $path -ExistingOnly }
    }
    return [pscustomobject]@{Root=$root;Directory=$selected;Missing=@($missing.ToArray());Purpose=$Purpose}
}

function Initialize-CredentialPrivateStorage {
    param([Parameter(Mandatory)]$Selection)
    # Re-admit the chosen namespace instead of trusting an earlier import.
    $cursor=$Selection.Directory;$missing=[Collections.Generic.List[string]]::new()
    while (-not (Test-Path -LiteralPath $cursor)) {
        $missing.Add($cursor);$cursor=[IO.Path]::GetDirectoryName($cursor)
        if (-not $cursor) { throw 'Existing credential storage parent required.' }
    }
    Assert-CredentialTrustedNamespace $cursor
    for($index=$missing.Count-1;$index -ge 0;$index--) {
        Assert-CredentialPrivateDirectory $missing[$index]
        Assert-CredentialTrustedNamespace $missing[$index]
    }
    Assert-CredentialPrivateDirectory $Selection.Root -ExistingOnly
    Assert-CredentialPrivateDirectory $Selection.Directory -ExistingOnly
    return $Selection.Directory
}

function Open-CredentialPrivateRead {
    param([string]$Path)
    $stream=$null;$reader=$null
    try {
        Assert-BitwardenPrivateInput $Path
        $item=Get-Item -LiteralPath $Path -Force -ErrorAction Stop
        if ($item.Attributes -band (-bnot ([IO.FileAttributes]::Normal -bor [IO.FileAttributes]::Archive))) { throw 'Unsupported private input attributes.' }
        if (@(Get-Item -LiteralPath (Get-CredentialStreamPath $Path) -Stream * -ErrorAction Stop | Where-Object Stream -NE ':$DATA').Count) { throw 'Private input alternate streams refused.' }
        Initialize-CredentialFileInterop
        $stream=Open-CredentialOwnedFile $Path
        if ([Pcai.Credential.FileV1]::Links($stream.SafeFileHandle) -ne 1) { throw 'Private input hardlinks refused.' }
        $snapshot=Read-CredentialOwnedSnapshot $stream
        $raw=[Security.AccessControl.RawSecurityDescriptor]::new($snapshot.Descriptor,0)
        $allowed=@([Security.Principal.WindowsIdentity]::GetCurrent().User.Value,'S-1-5-18','S-1-5-32-544')
        if ($null -eq $raw.Owner -or $raw.Owner.Value -notin $allowed -or $null -eq $raw.DiscretionaryAcl) { throw 'Unsupported private input owner or DACL.' }
        foreach($ace in $raw.DiscretionaryAcl) {
            if ($ace -isnot [Security.AccessControl.CommonAce] -or $ace.IsCallback -or $ace.AceQualifier -notin @([Security.AccessControl.AceQualifier]::AccessAllowed,[Security.AccessControl.AceQualifier]::AccessDenied)) { throw 'Unsupported private input ACE.' }
            if ($ace.AceQualifier -eq [Security.AccessControl.AceQualifier]::AccessAllowed -and $ace.SecurityIdentifier.Value -notin $allowed) { throw 'Foreign private input grant refused.' }
        }
        $stream.Position=0
        $reader=[IO.StreamReader]::new($stream,[Text.UTF8Encoding]::new($false,$true),$true,4096,$true)
        $text=$reader.ReadToEnd()
        return [pscustomobject]@{Path=$Path;Stream=$stream;Snapshot=$snapshot;Text=$text}
    } catch {
        if ($stream) { $stream.Dispose() }
        throw [IO.InvalidDataException]::new('Selected private input is invalid; private content is withheld.')
    } finally { if ($reader) { $reader.Dispose() } }
}

function Assert-CredentialPrivateReadCurrent {
    param([Parameter(Mandatory)]$Read)
    $now=Read-CredentialOwnedSnapshot $Read.Stream
    if ($now.Identity -cne $Read.Snapshot.Identity -or $now.Hash -cne $Read.Snapshot.Hash -or (Get-CredentialDescriptorSddl $now.Descriptor) -cne (Get-CredentialDescriptorSddl $Read.Snapshot.Descriptor) -or -not (Test-CredentialCurrentIdentity $Read.Path $now.Identity $now.Hash)) { throw 'Selected private input changed; private content is withheld.' }
}

function Invoke-CredentialFileReplace {
    param([string]$Stage,[string]$Target,[string]$Backup)
    [IO.File]::Replace($Stage,$Target,$Backup,$false)
}

function Invoke-CredentialFileMove {
    param([string]$Stage,[string]$Target)
    [IO.File]::Move($Stage,$Target,$false)
}

function Test-CredentialCurrentIdentity {
    param([string]$Path,[string]$Identity,[string]$Hash)
    $current = $null
    try {
        $current = Open-CredentialOwnedFile $Path
        $snapshot = Read-CredentialOwnedSnapshot $current
        try { return $snapshot.Identity -ceq $Identity -and $snapshot.Hash -ceq $Hash }
        finally { [Array]::Clear($snapshot.Bytes,0,$snapshot.Bytes.Length) }
    } catch [IO.FileNotFoundException] { return $false }
    finally { if ($current) { $current.Dispose() } }
}

function Write-BitwardenPrivateFile {
    <#
    .SYNOPSIS
    Publishes protected bytes with retained original custody.
    .DESCRIPTION
    Windows NTFS only. Recovery captures owner/group/DACL, not SACL/audit
    metadata, and remains RequiresReview when audit preservation is unverifiable.
    The cooperating lock and identity observations are not arbitrary-writer CAS.
    Existing parent/ancestors must have trusted ownership and applicable
    mutation grants before secret staging. Unsafe shared temp roots are refused.
    All private custody remains available after interrupted publication.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param([Parameter(Mandatory)][string]$LiteralPath,[Parameter(Mandatory)][string]$Value,[switch]$Overwrite)
    if (-not $IsWindows) { throw [PlatformNotSupportedException]::new('Private credential publication requires qualified Windows NTFS semantics.') }
    # Admit the supplied leaf before Windows/provider normalization can turn a
    # trailing-space reserved basename into a device path.
    if ([IO.Path]::GetFileName($LiteralPath).TrimEnd(' ','.') -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.|$)') { throw 'Reserved private-file destination refused.' }
    $provider = $null; $drive = $null
    $full = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($LiteralPath,[ref]$provider,[ref]$drive)
    if ($provider.Name -ne 'FileSystem') { throw 'Filesystem private-file destination required.' }
    $full = [IO.Path]::GetFullPath($full)
    if ($full.Substring([IO.Path]::GetPathRoot($full).Length).Contains(':')) { throw 'Alternate stream destinations refused.' }
    $parent = [IO.Path]::GetDirectoryName($full)
    if (-not [IO.Directory]::Exists($parent)) { throw 'Private-file parent must already exist.' }
    $cursor = $full
    while ($cursor) {
        if ([IO.Path]::GetFileName($cursor).TrimEnd(' ','.') -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.|$)') { throw 'Reserved private-file destination refused.' }
        if (Test-Path -LiteralPath $cursor) {
            if ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked private-file path refused.' }
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    if ([IO.DriveInfo]::new([IO.Path]::GetPathRoot($full)).DriveFormat -cne 'NTFS') { throw 'Private publication requires qualified NTFS storage.' }
    $exists = Test-Path -LiteralPath $full
    if ($exists) {
        $item = Get-Item -LiteralPath $full -Force
        if (-not $Overwrite -or $item.PSIsContainer -or ($item.Attributes -band (-bnot ([IO.FileAttributes]::Archive -bor [IO.FileAttributes]::Normal)))) { throw 'Unsafe private-file overwrite refused; unsupported attributes or permission.' }
        if (@(Get-Item -LiteralPath (Get-CredentialStreamPath $full) -Stream * -ErrorAction Stop | Where-Object Stream -NE ':$DATA').Count) { throw 'Alternate data streams require separate qualification.' }
    }
    if (-not $PSCmdlet.ShouldProcess($full,'Publish protected backend data')) { return }
    Assert-CredentialTrustedNamespace $parent
    Initialize-CredentialFileInterop
    $original = $null; $owned = $null; $lock = $null; $snapshot = $null; $bytes = $null; $receipt = $null; $transaction = $null
    try {
        if ($exists) {
            $original = Open-CredentialOwnedFile $full
            if ([Pcai.Credential.FileV1]::Links($original.SafeFileHandle) -ne 1) { throw 'Hard-linked private-file destination refused.' }
            $snapshot = Read-CredentialOwnedSnapshot $original
            $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
            $admittedOwners = @($identity.User.Value,$identity.Owner.Value)
            $rawDescriptor = [Security.AccessControl.RawSecurityDescriptor]::new($snapshot.Descriptor,0)
            if ($rawDescriptor.Owner.Value -notin $admittedOwners) { throw 'Foreign-owned private-file metadata refused.' }
            # ReplaceFile merges this DACL before the owned candidate can be
            # tightened. Refuse public predecessors before staging any secret,
            # rather than exposing new bytes during that publication boundary.
            $privatePrincipals = @($identity.User.Value,'S-1-5-18','S-1-5-32-544')
            if ($null -eq $rawDescriptor.DiscretionaryAcl) { throw 'Private predecessor requires an explicitly qualified DACL.' }
            foreach ($ace in $rawDescriptor.DiscretionaryAcl) {
                if ($ace -isnot [Security.AccessControl.QualifiedAce] -or $ace.AceQualifier -notin @([Security.AccessControl.AceQualifier]::AccessAllowed,[Security.AccessControl.AceQualifier]::AccessDenied)) { throw 'Unsupported private predecessor DACL entry.' }
                if ($ace.AceQualifier -eq [Security.AccessControl.AceQualifier]::AccessAllowed -and $ace.SecurityIdentifier.Value -notin $privatePrincipals) { throw 'Public predecessor ACL requires explicit repair before private publication.' }
            }
        }
        # Coordinate the admitted filesystem namespace across replacement IDs.
        # Actual directory/leaf names expand existing aliases; hardlinks and
        # reparses are already refused. Identity still binds every file object.
        $actualParent = (Get-Item -LiteralPath $parent -Force).FullName
        $actualLeaf = if ($exists) { (Get-Item -LiteralPath $full -Force).Name } else { [IO.Path]::GetFileName($full).TrimEnd(' ','.') }
        $keyText = ([IO.Path]::Combine($actualParent,$actualLeaf)).ToUpperInvariant()
        $key = Get-CredentialBytesHash ([Text.Encoding]::UTF8.GetBytes($keyText))
        $custody = Join-Path $parent '.pcai-private-write'
        Assert-CredentialPrivateDirectory $custody
        $bucket = Join-Path $custody $key
        Assert-CredentialPrivateDirectory $bucket
        $lockPath = Join-Path $bucket 'publication.lock'
        if (Test-Path -LiteralPath $lockPath) {
            $lockItem = Get-Item -LiteralPath $lockPath -Force
            if ($lockItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked publication lock refused.' }
        }
        $lock = [IO.FileSystemAclExtensions]::Create([IO.FileInfo]::new($lockPath),[IO.FileMode]::OpenOrCreate,[Security.AccessControl.FileSystemRights]'ReadData,WriteData,ReadPermissions',[IO.FileShare]::None,4096,[IO.FileOptions]::None,(New-CredentialPrivateAcl))
        if ([Pcai.Credential.FileV1]::Links($lock.SafeFileHandle) -ne 1) { throw 'Hard-linked publication lock refused.' }
        $transaction = Join-Path $bucket ('r-' + [guid]::NewGuid().ToString('N'))
        Assert-CredentialPrivateDirectory $transaction
        $receipt = [ordered]@{Schema='credential-publication-v1';State='Prepared';MetadataScope='OwnerGroupDaclOnly';AuditMetadataVerified=$false;OriginalIdentity=if($snapshot){$snapshot.Identity}else{$null};OriginalSHA256=if($snapshot){$snapshot.Hash}else{$null};RecoveryRequiresReview=$false}
        if ($snapshot) {
            $probe = New-CredentialPrivateStage (Join-Path $transaction 'metadata-probe.bin') ([byte[]]@())
            try { Set-CredentialOwnedDescriptor -Stream $probe -Descriptor $snapshot.Descriptor } finally { $probe.Dispose() }
            $descriptorStage = New-CredentialPrivateStage (Join-Path $transaction 'original-owner-group-dacl.bin') $snapshot.Descriptor
            $descriptorStage.Dispose()
            if (-not (Test-CredentialCurrentIdentity $full $snapshot.Identity $snapshot.Hash)) { throw 'Private-file predecessor changed before publication.' }
            $fresh = [Pcai.Credential.FileV1]::ReadDescriptor($original.SafeFileHandle)
            if ((Get-CredentialDescriptorSddl $fresh) -cne (Get-CredentialDescriptorSddl $snapshot.Descriptor)) { throw 'Private-file predecessor metadata changed.' }
        } elseif (Test-Path -LiteralPath $full) { throw 'Private-file destination appeared before publication.' }
        $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
        $stage = Join-Path $transaction 'candidate.bin'
        $owned = New-CredentialPrivateStage $stage $bytes
        # ReplaceFile's replacement sharing excludes an open data writer. Flush
        # and close it, then pin the same candidate with a read-only handle.
        $owned.Dispose(); $owned = Open-CredentialOwnedFile $stage
        $candidateIdentity = [Pcai.Credential.FileV1]::Identity($owned.SafeFileHandle)
        $candidateHash = Get-CredentialBytesHash $bytes
        $candidateDescriptor = [Pcai.Credential.FileV1]::ReadDescriptor($owned.SafeFileHandle)
        $receipt.CandidateIdentity = $candidateIdentity; $receipt.CandidateSHA256 = $candidateHash
        $backup = Join-Path $transaction 'actual-displaced.bin'
        # ReplaceFile opens its replacement with exclusive sharing. Retain its
        # captured identity, close it for publication, and qualify the resulting
        # target against that exact identity; the original stays pinned.
        $owned.Dispose(); $owned = $null
        if ($exists) { Invoke-CredentialFileReplace $stage $full $backup }
        else { Invoke-CredentialFileMove $stage $full }
        $owned = Open-CredentialOwnedFile $full -RestoreRights
        if (-not (Test-CredentialCurrentIdentity $full $candidateIdentity $candidateHash)) { throw 'Published private file was replaced by another writer.' }
        if ([Pcai.Credential.FileV1]::Identity($owned.SafeFileHandle) -cne $candidateIdentity) { throw 'Published file handle names another writer.' }
        # Successful writes intentionally retain the candidate's restrictive ACL;
        # only failed-publication recovery reinstates the predecessor metadata.
        Set-CredentialOwnedDescriptor -Stream $owned -Descriptor $candidateDescriptor
        if ($snapshot) {
            $displaced = Open-CredentialOwnedFile $backup
            try { $actual = Read-CredentialOwnedSnapshot $displaced } finally { $displaced.Dispose() }
            try {
                $receipt.DisplacedIdentity = $actual.Identity; $receipt.DisplacedSHA256 = $actual.Hash
                if ($actual.Identity -cne $snapshot.Identity -or $actual.Hash -cne $snapshot.Hash) { throw 'Actual displaced predecessor differed.' }
            } finally { [Array]::Clear($actual.Bytes,0,$actual.Bytes.Length) }
        }
        if (-not (Test-CredentialCurrentIdentity $full $candidateIdentity $candidateHash)) { throw 'Publication ownership changed during verification.' }
        $receipt.State = 'Published'
    } catch {
        $operation = $_.Exception
        if ($receipt) {
            $receipt.State = 'Failed'; $receipt.RecoveryRequiresReview = $true
            try {
                if ($snapshot -and (Test-CredentialCurrentIdentity $full $snapshot.Identity $snapshot.Hash)) {
                    $descriptor = [Pcai.Credential.FileV1]::ReadDescriptor($original.SafeFileHandle)
                    if ((Get-CredentialDescriptorSddl $descriptor) -ceq (Get-CredentialDescriptorSddl $snapshot.Descriptor)) { $receipt.State='OriginalPreserved';$receipt.RecoveryRequiresReview=$false }
                } elseif ($snapshot -and -not (Test-Path -LiteralPath $full) -and (Test-Path -LiteralPath (Join-Path $transaction 'actual-displaced.bin') -PathType Leaf)) {
                    $displaced = Open-CredentialOwnedFile (Join-Path $transaction 'actual-displaced.bin')
                    try { $actual = Read-CredentialOwnedSnapshot $displaced } finally { $displaced.Dispose() }
                    try {
                        $receipt.DisplacedIdentity=$actual.Identity; $receipt.DisplacedSHA256=$actual.Hash
                        if ($actual.Identity -cne $snapshot.Identity -or $actual.Hash -cne $snapshot.Hash) { throw 'Foreign displaced original requires review.' }
                        $recoveryPath = Join-Path $transaction 'recovery.bin'
                        $recovery = New-CredentialPrivateStage $recoveryPath ([byte[]]@())
                        try {
                            $source = Open-CredentialOwnedFile (Join-Path $transaction 'actual-displaced.bin')
                            try {
                                $sourceSnapshot = Read-CredentialOwnedSnapshot $source
                                if ($sourceSnapshot.Identity -cne $actual.Identity -or $sourceSnapshot.Hash -cne $actual.Hash) { throw 'Displaced recovery source changed.' }
                                $source.Position=0; $source.CopyTo($recovery); $recovery.Flush($true)
                            } finally { $source.Dispose() }
                            if ((Read-CredentialOwnedSnapshot $recovery).Hash -cne $actual.Hash) { throw 'Recovery copy differs from the actual displaced bytes.' }
                            $recoveryIdentity=[Pcai.Credential.FileV1]::Identity($recovery.SafeFileHandle)
                            Invoke-CredentialFileMove $recoveryPath $full
                            Set-CredentialOwnedDescriptor -Stream $recovery -Descriptor $snapshot.Descriptor
                            if (Test-CredentialCurrentIdentity $full $recoveryIdentity $snapshot.Hash) {
                                $receipt.State='OwnerGroupDaclRestoredAuditUnverified'
                                $receipt.RecoveryIdentity=$recoveryIdentity
                            } else { $receipt.State='LaterWriterPreserved' }
                        } finally { $recovery.Dispose() }
                    } finally { [Array]::Clear($actual.Bytes,0,$actual.Bytes.Length) }
                } elseif (Test-Path -LiteralPath $full) { $receipt.State='LaterWriterOrPublishedCandidatePreserved' }
            } catch { $receipt.RecoveryFailureType=$_.Exception.GetType().FullName }
        }
        $failure=[InvalidOperationException]::new('Private-file publication failed; protected custody retained for inspection.',$operation)
        $failure.Data['OriginalOperationException']=$operation
        if ($receipt) { $failure.Data['RecoveryRequiresReview']=$receipt.RecoveryRequiresReview; $failure.Data['RecoveryState']=$receipt.State; $failure.Data['CustodyPath']=$transaction }
        throw $failure
    } finally {
        if ($receipt -and $transaction) {
            try {
                $receiptBytes=[Text.Encoding]::UTF8.GetBytes(($receipt | ConvertTo-Json -Depth 4 -WarningAction Stop))
                $receiptStream=New-CredentialPrivateStage (Join-Path $transaction 'receipt.json') $receiptBytes
                $receiptStream.Dispose()
            } catch { Write-Warning 'Private publication receipt could not be recorded; retained custody requires inspection.' }
        }
        if ($owned) { $owned.Dispose() }; if ($original) { $original.Dispose() }; if ($lock) { $lock.Dispose() }
        if ($bytes) { [Array]::Clear($bytes,0,$bytes.Length) }; if ($snapshot) { [Array]::Clear($snapshot.Bytes,0,$snapshot.Bytes.Length) }
    }
}


function Initialize-BitwardenSessionFromBackends {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingPlainTextForPassword','',Justification='CredentialRoot is a filesystem directory path, not a password or secret value.')]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns','',Justification='Public compatibility entrypoint used by profiles and tooling; preserve its exact name.')]
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([pscustomobject])]
    param([switch]$Quiet,[switch]$Refresh,[string]$CredentialRoot = $env:PCAI_CREDENTIAL_ROOT)
    $previousSession=$env:BW_SESSION
    $previousId=$env:BW_CLIENTID
    $previousSecret=$env:BW_CLIENTSECRET
    $tempFile=$null;$accepted=$false
    $storage=$null
    # An explicit override must not be ignored by the existing-session fast
    # path. Admit it without provisioning or calling a provider/status command.
    if ($PSBoundParameters.ContainsKey('CredentialRoot') -or -not [string]::IsNullOrWhiteSpace($CredentialRoot)) {
        try {
            if ([string]::IsNullOrWhiteSpace($CredentialRoot)) { throw 'Empty explicit credential root refused.' }
            $storage=Resolve-CredentialPrivateStorage -RootOverride $CredentialRoot -Purpose Bootstrap
        } catch { return [pscustomobject]@{Success=$false;Source='bootstrap-failed';Message='Configured storage admission failed; prior process state preserved.'} }
    }
    if(-not$Refresh-and$previousSession-and(Get-BwStatusSafe).status-eq'unlocked'){
        return [pscustomobject]@{Success=$true;Source='process';Message='Validated existing process session.'}
    }
    if(-not$PSCmdlet.ShouldProcess('Bitwarden process session and protected session file','Validate or initialize session from configured backends')){
        return [pscustomobject]@{Success=$false;Source='not-applied';Message='Session bootstrap was not applied.'}
    }
    try{
        if ($PSBoundParameters.ContainsKey('CredentialRoot') -and [string]::IsNullOrWhiteSpace($CredentialRoot)) { throw 'Empty explicit credential root refused.' }
        if (-not $storage) { $storage=Resolve-CredentialPrivateStorage -RootOverride $CredentialRoot -Purpose Bootstrap }
        $bootstrap=Resolve-BwBootstrapState
        $sessionFile=if($bootstrap.SessionFile){$bootstrap.SessionFile}else{Get-BitwardenSessionFile}
        if(-not(Test-Path -LiteralPath ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($sessionFile))) -PathType Container)){
            return [pscustomobject]@{Success=$false;Source='provisioning-required';Message='Protected session parent is missing; provision it explicitly before requesting bootstrap.'}
        }
        if(-not$Refresh-and$bootstrap.SessionFile-and(Test-Path -LiteralPath $bootstrap.SessionFile -PathType Leaf)){
            Assert-BitwardenPrivateInput -Path $bootstrap.SessionFile
            $candidate=[IO.File]::ReadAllText($bootstrap.SessionFile).Trim()
            if($candidate.Length-gt20){
                $env:BW_SESSION=$candidate
                if((Get-BwStatusSafe).status-eq'unlocked'){$accepted=$true;return [pscustomobject]@{Success=$true;Source='session-file';Message='Validated protected session file.'}}
                $env:BW_SESSION=$previousSession
            }
        }
        $status=Get-BwStatusSafe
        if($status.status-notin@('locked','unlocked','unauthenticated')){throw 'Bitwarden status is not qualified for authentication mutation.'}
        if($status.status-eq'unauthenticated'-and$bootstrap.ClientId-and$bootstrap.ClientSecret){
            $env:BW_CLIENTID=$bootstrap.ClientId;$env:BW_CLIENTSECRET=$bootstrap.ClientSecret
            $login=Invoke-BitwardenCli -Arguments @('login','--apikey') -TimeoutSeconds 30
            if(-not$login.Success){throw 'Bitwarden API login failed.'}
        }
        if(-not$bootstrap.Password){return [pscustomobject]@{Success=$false;Source='missing-password';Message='No configured master password source was available.'}}
        $bootstrapRoot=Initialize-CredentialPrivateStorage -Selection $storage
        $tempFile=Join-Path $bootstrapRoot ([IO.Path]::GetRandomFileName())
        Write-BitwardenPrivateFile -LiteralPath $tempFile -Value $bootstrap.Password -Confirm:$false
        $response=Invoke-BitwardenCli -Arguments @('unlock','--passwordfile',$tempFile,'--raw') -TimeoutSeconds 30
        if(-not$response.Success-or[string]::IsNullOrWhiteSpace($response.StdOut)-or$response.StdOut.Trim().Length-le20){throw 'Bitwarden unlock did not produce a usable candidate session.'}
        $env:BW_SESSION=$response.StdOut.Trim()
        if((Get-BwStatusSafe).status-ne'unlocked'){throw 'Candidate session was rejected.'}
        # Parent creation must be an explicit provisioning step, not an unlock side effect.
        Write-BitwardenPrivateFile -LiteralPath $sessionFile -Value $env:BW_SESSION -Overwrite -Confirm:$false
        $accepted=$true
        return [pscustomobject]@{Success=$true;Source=$bootstrap.PasswordSource;Message='Validated and persisted the protected session.'}
    }catch{
        if(-not$Quiet){Write-Verbose 'Bitwarden bootstrap failed; prior process credentials will be restored. Private output is withheld.'}
        return [pscustomobject]@{Success=$false;Source='bootstrap-failed';Message='Configured bootstrap or candidate validation failed; prior process state restored.'}
    }finally{
        if(-not$accepted){$env:BW_SESSION=$previousSession}
        $env:BW_CLIENTID=$previousId;$env:BW_CLIENTSECRET=$previousSecret
        if($tempFile-and(Test-Path -LiteralPath $tempFile -PathType Leaf)){[IO.File]::Delete($tempFile)}
    }
}

function Resolve-BwsTokenFromBackends {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns','',Justification='Existing compatibility API Resolve-BwsTokenFromBackends is referenced by installed profiles or sibling backend code; preserve its exact name.')]
    [OutputType([string])]
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [switch]$Quiet
    )
    if (-not $PSCmdlet.ShouldProcess('Resolve-BwsTokenFromBackends state', 'Apply requested backend operation')) { return $null }


    if ($env:BWS_ACCESS_TOKEN) {
        return $env:BWS_ACCESS_TOKEN
    }

    $tokenFile = Join-Path $env:USERPROFILE '.config\bws\.token'
    if (Test-Path -LiteralPath $tokenFile) {
        try {
            $secureToken = Get-Content -LiteralPath $tokenFile | ConvertTo-SecureString
            $bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
            try {
                $token = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
                if ($token) {
                    $env:BWS_ACCESS_TOKEN = $token
                    return $token
                }
            } finally {
                [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
            }
        } catch { Write-Warning "BWS token file exists but failed to decrypt ($tokenFile) - likely DPAPI/user mismatch: failure details withheld" }
    }

    $moduleState = Import-LocalSecretManagement
    if ($moduleState.SecretManagementAvailable) {
        foreach ($vaultName in @('LocalStore', 'SecretStore', 'Bitwarden', 'CredMan')) {
            foreach ($secretName in @('BWS_ACCESS_TOKEN', 'bws/access-token', 'bitwarden-secrets/access-token')) {
                try {
                    $value = Microsoft.PowerShell.SecretManagement\Get-Secret -Name $secretName -Vault $vaultName -AsPlainText -ErrorAction Stop -WarningAction SilentlyContinue
                    if ($value) {
                        $env:BWS_ACCESS_TOKEN = $value
                        return $value
                    }
                } catch { Write-Verbose 'Optional backend operation failed in Resolve-BwsTokenFromBackends; private provider output is withheld.' }
            }
        }
    }

    $userToken = [Environment]::GetEnvironmentVariable('BWS_ACCESS_TOKEN', 'User')
    if ($userToken) {
        $env:BWS_ACCESS_TOKEN = $userToken
        return $userToken
    }

    if (-not $Quiet) {
        Write-Verbose 'Unable to resolve BWS_ACCESS_TOKEN from configured backends.'
    }

    $null
}

function Initialize-LocalSecretBackends {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns','',Justification='Existing compatibility API Initialize-LocalSecretBackends is referenced by installed profiles or sibling backend code; preserve its exact name.')]
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [switch]$InstallMissing,
        [switch]$RegisterLocalStore,
        [switch]$RegisterCredManVault,
        [switch]$RegisterBitwardenVault
    )
    if (-not $PSCmdlet.ShouldProcess('Initialize-LocalSecretBackends state', 'Apply requested backend operation')) { return [pscustomobject]@{Applied=$false} }


    $state = Import-LocalSecretManagement -InstallMissing:$InstallMissing
    if (-not $state.SecretManagementAvailable) {
        return Get-SecretBackendStatus
    }

    if ($RegisterLocalStore -and $state.SecretStoreAvailable) {
        try {
            if (-not (Microsoft.PowerShell.SecretManagement\Get-SecretVault -Name 'LocalStore' -ErrorAction SilentlyContinue)) {
                Microsoft.PowerShell.SecretManagement\Register-SecretVault -Name 'LocalStore' -ModuleName Microsoft.PowerShell.SecretStore -DefaultVault -WarningAction SilentlyContinue | Out-Null
            }
        } catch { Write-Verbose 'Optional backend operation failed in Initialize-LocalSecretBackends; private provider output is withheld.' }
    }

    if ($RegisterCredManVault -and $state.CredManStoreAvailable) {
        try {
            if (-not (Microsoft.PowerShell.SecretManagement\Get-SecretVault -Name 'CredMan' -ErrorAction SilentlyContinue)) {
                Microsoft.PowerShell.SecretManagement\Register-SecretVault -Name 'CredMan' -ModuleName Microsoft.PowerShell.CredManStore -WarningAction SilentlyContinue | Out-Null
            }
        } catch { Write-Verbose 'Optional backend operation failed in Initialize-LocalSecretBackends; private provider output is withheld.' }
    }

    if ($RegisterBitwardenVault -and $state.BitwardenVaultAvailable) {
        try {
            if (-not (Microsoft.PowerShell.SecretManagement\Get-SecretVault -Name 'Bitwarden' -ErrorAction SilentlyContinue)) {
                Microsoft.PowerShell.SecretManagement\Register-SecretVault -Name 'Bitwarden' -ModuleName SecretManagement.BitWarden -WarningAction SilentlyContinue | Out-Null
            }
        } catch { Write-Verbose 'Optional backend operation failed in Initialize-LocalSecretBackends; private provider output is withheld.' }
    }

    Get-SecretBackendStatus
}

function Get-SecretBackendStatus {
    [CmdletBinding()]
    param()

    $moduleState = [pscustomobject]@{
        SecretManagementAvailable = [bool](Get-Module -ListAvailable Microsoft.PowerShell.SecretManagement)
        SecretStoreAvailable = [bool](Get-Module -ListAvailable Microsoft.PowerShell.SecretStore)
        CredManStoreAvailable = [bool](Get-Module -ListAvailable Microsoft.PowerShell.CredManStore)
        CredentialManagerAvailable = [bool](Get-Module -ListAvailable CredentialManager)
        BitwardenVaultAvailable = [bool](Get-Module -ListAvailable SecretManagement.BitWarden)
    }
    $vaults = @()
    if ($moduleState.SecretManagementAvailable -and (Get-Command Microsoft.PowerShell.SecretManagement\Get-SecretVault -ListImported -ErrorAction SilentlyContinue)) {
        try {
            $vaults = @(Microsoft.PowerShell.SecretManagement\Get-SecretVault -ErrorAction SilentlyContinue -WarningAction SilentlyContinue | Select-Object Name, ModuleName, IsDefault)
        } catch { Write-Verbose 'Optional backend operation failed in Get-SecretBackendStatus; private provider output is withheld.' }
    }

    $bwStatus = Get-BwStatusSafe
    $passwordFile = Get-BitwardenPasswordFile
    $sessionFile = Get-BitwardenSessionFile
    $bwsToken = [bool]$env:BWS_ACCESS_TOKEN

    [pscustomobject]@{
        SecretManagementAvailable = $moduleState.SecretManagementAvailable
        SecretStoreAvailable      = $moduleState.SecretStoreAvailable
        CredManStoreAvailable     = $moduleState.CredManStoreAvailable
        CredentialManagerAvailable = $moduleState.CredentialManagerAvailable
        BitwardenVaultAvailable   = $moduleState.BitwardenVaultAvailable
        RegisteredVaults          = $vaults
        BitwardenPasswordFile     = $passwordFile
        BitwardenSessionFile      = $sessionFile
        BwStatus                  = $bwStatus.status
        BwSessionInProcess        = [bool]$env:BW_SESSION
        BwsTokenAvailable         = [bool]$bwsToken
        ResourceConfigPath        = (Join-Path $env:USERPROFILE '.machine\resource-credentials.json')
    }
}

function ConvertFrom-SecureStringSafe {
    param([Security.SecureString]$SecureString)

    if (-not $SecureString) {
        return $null
    }

    $bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureString)
    try {
        [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
    } finally {
        [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

function Resolve-SecretCredentialInternal {
    param(
        [Parameter(Mandatory)]
        [object]$Secret,
        [string]$DefaultUserName
    )

    if ($Secret -is [pscredential]) {
        return [pscustomobject]@{
            UserName = $Secret.UserName
            Password = (ConvertFrom-SecureStringSafe -SecureString $Secret.Password)
        }
    }

    if ($Secret -is [securestring]) {
        return [pscustomobject]@{
            UserName = $DefaultUserName
            Password = (ConvertFrom-SecureStringSafe -SecureString $Secret)
        }
    }

    if ($Secret -is [string]) {
        try {
            $jsonValue = $Secret | ConvertFrom-SecretBackendJson -ErrorAction Stop
            if ($jsonValue.password) {
                return [pscustomobject]@{
                    UserName = if ($jsonValue.username) { $jsonValue.username } else { $DefaultUserName }
                    Password = [string]$jsonValue.password
                }
            }
        } catch { Write-Verbose 'Optional backend operation failed in Resolve-SecretCredentialInternal; private provider output is withheld.' }

        return [pscustomobject]@{
            UserName = $DefaultUserName
            Password = $Secret
        }
    }

    if ($Secret.PSObject.Properties.Name -contains 'password') {
        return [pscustomobject]@{
            UserName = if ($Secret.PSObject.Properties.Name -contains 'username') { [string]$Secret.username } else { $DefaultUserName }
            Password = [string]$Secret.password
        }
    }

    $null
}

function Get-SecretCredentialFromVaultInternal {
    param(
        [Parameter(Mandatory)]
        [string]$SecretName,
        [string]$VaultName,
        [string]$DefaultUserName
    )

    if (-not (Get-Command Microsoft.PowerShell.SecretManagement\Get-Secret -ErrorAction SilentlyContinue)) {
        return $null
    }

    try {
        if ($VaultName -and (Get-Command Microsoft.PowerShell.SecretManagement\Get-SecretVault -ErrorAction SilentlyContinue)) {
            $registeredVault = Microsoft.PowerShell.SecretManagement\Get-SecretVault -Name $VaultName -ErrorAction SilentlyContinue -WarningAction SilentlyContinue
            if (-not $registeredVault) {
                return $null
            }
        }

        $secret = if ($VaultName) {
            Microsoft.PowerShell.SecretManagement\Get-Secret -Name $SecretName -Vault $VaultName -ErrorAction Stop -WarningAction SilentlyContinue
        } else {
            Microsoft.PowerShell.SecretManagement\Get-Secret -Name $SecretName -ErrorAction Stop -WarningAction SilentlyContinue
        }

        return Resolve-SecretCredentialInternal -Secret $secret -DefaultUserName $DefaultUserName
    } catch {
        $null
    }
}

function Sync-ResourceCredentialTargetsFromVault {
    [OutputType([object[]])]
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [string]$ConfigPath = (Join-Path $env:USERPROFILE '.machine\resource-credentials.json'),
        [string]$VaultName = 'LocalStore'

    )
    if (-not $PSCmdlet.ShouldProcess('Sync-ResourceCredentialTargetsFromVault state', 'Apply requested backend operation')) { return @() }


    if (-not (Test-Path -LiteralPath $ConfigPath)) {
        throw "Resource credential configuration not found: $ConfigPath"
    }

    $config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-SecretBackendJson -ErrorAction Stop
    $results = New-Object System.Collections.Generic.List[object]

    foreach ($resource in @($config.resources)) {
        if (-not $resource.secretName) {
            $results.Add([pscustomobject]@{
                Resource = $resource.name
                Target   = $null
                Status   = 'skipped'
                Message  = 'secretName not configured'
            })
            continue
        }

        $credential = Get-SecretCredentialFromVaultInternal -SecretName $resource.secretName -VaultName $VaultName -DefaultUserName $resource.preferredUsername
        if (-not $credential -or -not $credential.Password) {
            $results.Add([pscustomobject]@{
                Resource = $resource.name
                Target   = $null
                Status   = 'missing-secret'
                Message  = "No usable credential found in vault '$VaultName' for secret '$($resource.secretName)'"
            })
            continue
        }

        foreach ($target in @($resource.targets)) {
            $userName = if ($target.username) { [string]$target.username } else { [string]$credential.UserName }
            if (-not $userName) {
                $results.Add([pscustomobject]@{
                    Resource = $resource.name
                    Target   = $target.target
                    Status   = 'missing-username'
                    Message  = 'No username available for target'
                })
                continue
            }

            if (-not $PSCmdlet.ShouldProcess([string]$target.target, 'Update selected credential target')) {
                $results.Add([pscustomobject]@{
                    Resource = $resource.name
                    Target   = $target.target
                    Status   = 'whatif'
                    Message  = "Would update $($target.kind) target for $userName"
                })
                continue
            }

            try {
                $kind = if ($target.kind -eq 'domain') { 'domain' } else { 'generic' }
                $written = Set-CredentialManagerCredential -Target $target.target -UserName $userName -Password $credential.Password -Kind $kind -Confirm:$false
                if ($written) {
                    $results.Add([pscustomobject]@{
                        Resource = $resource.name
                        Target   = $target.target
                        Status   = 'updated'
                        Message  = "Updated $($target.kind) target for $userName"
                    })
                } else {
                    $results.Add([pscustomobject]@{
                        Resource = $resource.name
                        Target   = $target.target
                        Status   = 'failed'
                        Message  = "Windows credential write failed."
                    })
                }
            } catch {
                $results.Add([pscustomobject]@{
                    Resource = $resource.name
                    Target   = $target.target
                    Status   = 'failed'
                    Message  = 'Backend operation failed; private details withheld.'
                })
            }
        }
    }

    $results
}

function Initialize-BitwardenSession {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [switch]$Quiet,
        [switch]$SyncBootstrapSecretsToLocalStore
    )
    if (-not $PSCmdlet.ShouldProcess('Initialize-BitwardenSession state', 'Apply requested backend operation')) { return [pscustomobject]@{Success=$false;Source='not-applied';Message='Session initialization not applied.'} }


    $result = Initialize-BitwardenSessionFromBackends -Quiet:$Quiet
    if ($result.Success -and $SyncBootstrapSecretsToLocalStore -and (Get-Command Resolve-BwBootstrapState -ErrorAction SilentlyContinue) -and (Get-Command Set-PlainTextSecretInSecretManagement -ErrorAction SilentlyContinue)) {
        try {
            $bootstrap = Resolve-BwBootstrapState
            foreach ($secretName in @('bitwarden/master_password', 'bitwarden/client_id', 'bitwarden/client_secret')) {
                $value = switch ($secretName) {
                    'bitwarden/master_password' { $bootstrap.Password }
                    'bitwarden/client_id' { $bootstrap.ClientId }
                    'bitwarden/client_secret' { $bootstrap.ClientSecret }
                }
                if (-not [string]::IsNullOrWhiteSpace($value)) {
                    [void](Set-PlainTextSecretInSecretManagement -Name $secretName -Secret $value -VaultName 'LocalStore')
                }
            }
        } catch { Write-Verbose 'Optional backend operation failed in Initialize-BitwardenSession; private provider output is withheld.' }
    }

    [pscustomobject]@{
        Success = [bool]$result.Success
        Source  = $result.Source
        Message = $result.Message
        SessionAvailable = [bool]$env:BW_SESSION
    }
}

function Get-CmdKeyListOutput {
    [CmdletBinding()]
    [OutputType([string])]
    param([Parameter(Mandatory)][string]$Target,[ValidateRange(1,120)][int]$TimeoutSeconds=5)
    $result=Invoke-SecretBackendProcess -FilePath (Join-Path $env:SystemRoot 'System32/cmdkey.exe') -Arguments @("/list:$Target") -TimeoutSeconds $TimeoutSeconds
    if($result.Success){return $result.StdOut}
    Write-Verbose 'Credential metadata listing failed; private output is withheld.'
    return $null
}

function Get-CredentialManagerCredential {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Target
    )

    try {
        $listOutput = Get-CmdKeyListOutput -Target $Target
        if (-not $listOutput) {
            return $null
        }

        $normalizedTarget = $Target.Trim().ToLowerInvariant()
        $blocks = ($listOutput -split "Currently stored credentials:")[-1] -split "(?=Target:\s)"
        foreach ($block in $blocks) {
            if ($block -match 'Target:\s*(.+)') {
                $targetValue = $matches[1].Trim()
                $candidateTargets = @($targetValue)
                if ($targetValue -match '^LegacyGeneric:target=(.+)$') {
                    $candidateTargets += $matches[1].Trim()
                } elseif ($targetValue -match '^Domain:target=(.+)$') {
                    $candidateTargets += $matches[1].Trim()
                }

                $normalizedCandidates = @($candidateTargets | ForEach-Object { $_.Trim().ToLowerInvariant() } | Select-Object -Unique)
                if ($normalizedCandidates -contains $normalizedTarget) {
                    $userName = $null
                    $type = $null
                    if ($block -match 'User:\s*(.+)') {
                        $userName = $matches[1].Trim()
                    }
                    if ($block -match 'Type:\s*(.+)') {
                        $type = $matches[1].Trim()
                    }
                    return [pscustomobject]@{
                        Target = $targetValue
                        RequestedTarget = $Target
                        Type = $type
                        UserName = $userName
                    }
                }
            }
        }
    } catch { Write-Verbose 'Optional backend operation failed in Get-CredentialManagerCredential; private provider output is withheld.' }

    $null
}

function Set-CredentialManagerCredential {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([bool])]
    param([Parameter(Mandatory)][string]$Target,[Parameter(Mandatory)][string]$UserName,[Parameter(Mandatory)][WorkProfileBackend.SecureSecretArgument()][Security.SecureString]$Password,[ValidateSet('generic','domain')][string]$Kind='generic')
    if(-not$PSCmdlet.ShouldProcess($Target,"Set Credential Manager $Kind credential")){return $false}
    Write-NativeWindowsCredential -Target $Target -UserName $UserName -Password $Password -Kind $Kind
    return $true
}

function Resolve-BitwardenItemForResource {
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param([Parameter(Mandatory)][psobject]$Resource)
    if($Resource.PSObject.Properties.Name-contains'bitwardenItemId'-and$Resource.bitwardenItemId){
        $response=Invoke-BitwardenCli -Arguments @('get','item',[string]$Resource.bitwardenItemId)
        if(-not$response.Success){throw 'Configured Bitwarden item read failed; fuzzy fallback is refused.'}
        $item=$response.StdOut|ConvertFrom-SecretBackendJson -ErrorAction Stop
        if($item.id-ne$Resource.bitwardenItemId){throw 'Configured Bitwarden item identity mismatch.'}
        return [pscustomobject]@{Item=$item;Source="id:$($Resource.bitwardenItemId)"}
    }
    $name=if($Resource.PSObject.Properties.Name-contains'bitwardenItemName'-and$Resource.bitwardenItemName){[string]$Resource.bitwardenItemName}else{[string]$Resource.name}
    if([string]::IsNullOrWhiteSpace($name)){throw 'An exact Bitwarden item name or identifier is required.'}
    $response=Invoke-BitwardenCli -Arguments @('list','items','--search',$name)
    if(-not$response.Success){throw 'Bitwarden resource search failed.'}
    $matching=@($response.StdOut|ConvertFrom-SecretBackendJson -ErrorAction Stop|Where-Object {$_.name-eq$name-and$_.login-and$_.login.password})
    if($matching.Count-ne1){throw 'Exact Bitwarden resource identity is missing or ambiguous; first fuzzy match is refused.'}
    [pscustomobject]@{Item=$matching[0];Source="name:$name"}
}

function Get-ResourceCredentialManifest {
    [CmdletBinding()]
    param(
        [string]$ConfigPath = (Join-Path $env:USERPROFILE '.machine\resource-credentials.json')
    )

    if (-not (Test-Path -LiteralPath $ConfigPath)) {
        return $null
    }

    Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-SecretBackendJson -ErrorAction Stop
}

function Get-ResourceCredentialReconciliationReport {
    [OutputType([object[]])]
    [CmdletBinding()]
    param(
        [string]$ConfigPath = (Join-Path $env:USERPROFILE '.machine\resource-credentials.json'),
        [string]$VaultName = 'LocalStore'
    )

    $manifest = Get-ResourceCredentialManifest -ConfigPath $ConfigPath
    if (-not $manifest) {
        return @()
    }

    $resources = New-Object System.Collections.Generic.List[object]
    $moduleState = Import-LocalSecretManagement
    $registeredVault = $null
    if ($moduleState.SecretManagementAvailable) {
        try {
            $registeredVault = Microsoft.PowerShell.SecretManagement\Get-SecretVault -Name $VaultName -ErrorAction SilentlyContinue -WarningAction SilentlyContinue
        } catch { Write-Verbose 'Optional backend operation failed in Get-ResourceCredentialReconciliationReport; private provider output is withheld.' }
    }

    foreach ($resource in @($manifest.resources)) {
        $localSecretPresent = $false
        if ($registeredVault -and $resource.secretName -and (Get-Command Microsoft.PowerShell.SecretManagement\Get-SecretInfo -ErrorAction SilentlyContinue)) {
            try {
                $localSecretPresent = [bool](Microsoft.PowerShell.SecretManagement\Get-SecretInfo -Name $resource.secretName -Vault $VaultName -ErrorAction Stop -WarningAction SilentlyContinue)
            } catch {
                $localSecretPresent = $false
            }
        }

        $targetStates = @()
        foreach ($target in @($resource.targets)) {
            $existing = Get-CredentialManagerCredential -Target $target.target
            $targetStates += [pscustomobject]@{
                Target = $target.target
                Kind = $target.kind
                Exists = [bool]$existing
                CredentialType = if ($existing) { $existing.Type } else { $null }
                UserName = if ($existing) { $existing.UserName } else { $null }
            }
        }

        $driveState = $null
        if ($resource.PSObject.Properties.Name -contains 'expectedDriveLetter' -and -not [string]::IsNullOrWhiteSpace([string]$resource.expectedDriveLetter)) {
            $driveName = [string]$resource.expectedDriveLetter
            if ($driveName.EndsWith(':')) {
                $driveName = $driveName.Substring(0, $driveName.Length - 1)
            }

            $drive = Get-PSDrive -Name $driveName -ErrorAction SilentlyContinue
            $driveState = [pscustomobject]@{
                Letter = [string]$resource.expectedDriveLetter
                Present = [bool]$drive
                Root = if ($drive) { $drive.Root } else { $null }
                DisplayRoot = if ($drive -and $drive.DisplayRoot) { $drive.DisplayRoot } else { $null }
                MatchesHint = if ($drive -and ($resource.PSObject.Properties.Name -contains 'sharePathHint')) {
                    ($drive.Root -eq [string]$resource.sharePathHint) -or ($drive.DisplayRoot -eq [string]$resource.sharePathHint)
                } else {
                    $null
                }
            }
        }

        $resources.Add([pscustomobject]@{
            Name = $resource.name
            ResourceType = $resource.resourceType
            BitwardenItemId = if ($resource.bitwardenItemId) { $resource.bitwardenItemId } elseif ($resource.PSObject.Properties.Name -contains 'bitwardenItemSearch') { "(search) $($resource.bitwardenItemSearch)" } else { $null }
            LocalSecretName = $resource.secretName
            LocalSecretPresent = $localSecretPresent
            PreferredUserName = $resource.preferredUsername
            ExpectedDriveLetter = if ($resource.PSObject.Properties.Name -contains 'expectedDriveLetter') { $resource.expectedDriveLetter } else { $null }
            SharePathHint = if ($resource.PSObject.Properties.Name -contains 'sharePathHint') { $resource.sharePathHint } else { $null }
            SharePathCandidates = if ($resource.PSObject.Properties.Name -contains 'sharePathCandidates') {
                @($resource.sharePathCandidates)
            } elseif ($resource.PSObject.Properties.Name -contains 'sharePathHint' -and -not [string]::IsNullOrWhiteSpace([string]$resource.sharePathHint)) {
                @([string]$resource.sharePathHint)
            } else {
                @()
            }
            CredentialTargets = @($targetStates | ForEach-Object {
                [pscustomobject]@{
                    Target = $_.Target
                    Type = $_.Kind
                    Exists = $_.Exists
                    CredentialType = $_.CredentialType
                    UserName = $_.UserName
                }
            })
            Drive = $driveState
            Notes = $resource.notes
        })
    }

    [pscustomobject]@{
        ManifestPath = $ConfigPath
        Vault = [pscustomobject]@{
            Name = $VaultName
            Ready = [bool]$registeredVault
            Message = if ($registeredVault) {
                $null
            } elseif (-not $moduleState.SecretManagementAvailable) {
                'SecretManagement is not available locally.'
            } else {
                "Vault '$VaultName' is not registered yet."
            }
        }
        Resources = if ($resources.Count -gt 0) { $resources.ToArray() } else { @() }
    }
}

function Get-UncPathHost {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path
    )

    if ($Path -match '^[\\]{2}([^\\]+)\\') {
        return $matches[1]
    }

    $null
}

function Test-TcpEndpointReachable {
    [OutputType([bool])]
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$HostName,
        [int]$Port = 445,
        [int]$TimeoutMilliseconds = 3000
    )

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $asyncResult = $client.BeginConnect($HostName, $Port, $null, $null)
        $connected = $asyncResult.AsyncWaitHandle.WaitOne($TimeoutMilliseconds, $false) -and $client.Connected
        if ($client.Connected) {
            try {
                $client.EndConnect($asyncResult) | Out-Null
            } catch { Write-Verbose 'Optional backend operation failed in Test-TcpEndpointReachable; private provider output is withheld.' }
        }
        return $connected
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Resolve-ResourceSharePathCandidate {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][psobject]$Resource,
        [int]$TimeoutMilliseconds = 3000
    )

    $candidates = New-Object System.Collections.Generic.List[string]
    foreach ($propertyName in @('sharePathHint', 'sharePathCandidates')) {
        if (-not ($Resource.PSObject.Properties.Name -contains $propertyName)) {
            continue
        }

        $value = $Resource.$propertyName
        foreach ($candidate in @($value)) {
            $candidateText = [string]$candidate
            if (-not [string]::IsNullOrWhiteSpace($candidateText) -and -not $candidates.Contains($candidateText)) {
                $candidates.Add($candidateText)
            }
        }
    }

    $probeResults = New-Object System.Collections.Generic.List[object]
    foreach ($candidatePath in $candidates) {
        $hostName = Get-UncPathHost -Path $candidatePath
        $reachable = $false
        if (-not [string]::IsNullOrWhiteSpace($hostName)) {
            $reachable = Test-TcpEndpointReachable -HostName $hostName -TimeoutMilliseconds $TimeoutMilliseconds
        }

        $probe = [pscustomobject]@{
            Path = $candidatePath
            Host = $hostName
            Reachable = $reachable
        }
        $probeResults.Add($probe)

        if ($reachable) {
            return [pscustomobject]@{
                SelectedPath = $candidatePath
                Probes = if ($probeResults.Count -gt 0) { $probeResults.ToArray() } else { @() }
            }
        }
    }

    [pscustomobject]@{
        SelectedPath = $null
        Probes = if ($probeResults.Count -gt 0) { $probeResults.ToArray() } else { @() }
    }
}

function Repair-ResourceDriveMappings {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseSingularNouns','',Justification='Existing compatibility API Repair-ResourceDriveMappings is referenced by installed profiles or sibling backend code; preserve its exact name.')]
    [OutputType([object[]],[array])]
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [string]$ConfigPath = (Join-Path $env:USERPROFILE '.machine\resource-credentials.json')
    )
    if (-not $PSCmdlet.ShouldProcess('Repair-ResourceDriveMappings state', 'Apply requested backend operation')) { return @() }


    $manifest = Get-ResourceCredentialManifest -ConfigPath $ConfigPath
    if (-not $manifest) {
        return @([pscustomobject]@{
            Name = $null
            Success = $false
            Message = "Resource credential configuration not found: $ConfigPath"
        })
    }

    $results = New-Object System.Collections.Generic.List[object]

    foreach ($resource in @($manifest.resources)) {
        if (-not ($resource.PSObject.Properties.Name -contains 'expectedDriveLetter') -or
            -not ($resource.PSObject.Properties.Name -contains 'sharePathHint')) {
            continue
        }

        $driveLetter = [string]$resource.expectedDriveLetter
        $shareResolution = Resolve-ResourceSharePathCandidate -Resource $resource
        $sharePath = if ($shareResolution.SelectedPath) { [string]$shareResolution.SelectedPath } else { [string]$resource.sharePathHint }
        if ([string]::IsNullOrWhiteSpace($driveLetter) -or [string]::IsNullOrWhiteSpace($sharePath)) {
            continue
        }

        if (-not $shareResolution.SelectedPath) {
            $probeSummary = @($shareResolution.Probes | ForEach-Object {
                if ($_.Host) {
                    "{0} ({1}) reachable={2}" -f $_.Path, $_.Host, $_.Reachable
                } else {
                    "{0} reachable={1}" -f $_.Path, $_.Reachable
                }
            }) -join '; '

            $results.Add([pscustomobject]@{
                Name = $resource.name
                Drive = $driveLetter
                SharePath = $sharePath
                Success = $false
                Status = 'unreachable'
                Message = if ($probeSummary) {
                    "No reachable SMB endpoint found for $driveLetter. Checked: $probeSummary"
                } else {
                    "No reachable SMB endpoint found for $driveLetter."
                }
            })
            continue
        }

        $driveName = if ($driveLetter.EndsWith(':')) { $driveLetter.Substring(0, $driveLetter.Length - 1) } else { $driveLetter }
        $existing = Get-PSDrive -Name $driveName -ErrorAction SilentlyContinue
        if ($existing -and (($existing.Root -eq $sharePath) -or ($existing.DisplayRoot -eq $sharePath))) {
            $results.Add([pscustomobject]@{
                Name = $resource.name
                Drive = $driveLetter
                SharePath = $sharePath
                Success = $true
                Status = 'already-mounted'
                Message = "Drive $driveLetter already maps to $sharePath"
            })
            continue
        }

        if (-not $PSCmdlet.ShouldProcess($driveLetter, "Map $driveLetter to $sharePath")) {
            $results.Add([pscustomobject]@{
                Name = $resource.name
                Drive = $driveLetter
                SharePath = $sharePath
                Success = $true
                Status = 'whatif'
                Message = "Would map $driveLetter to $sharePath"
            })
            continue
        }

        if ($existing) { throw 'A different drive mapping exists; explicit mapping custody is required.' }
        $response = Invoke-SecretBackendProcess -FilePath (Join-Path $env:SystemRoot 'System32/net.exe') -Arguments @('use', $driveLetter, $sharePath, '/persistent:yes')
        $success = $response.Success
        if ($success) {
            $observed = Get-PSDrive -Name $driveName -ErrorAction Stop
            if ($observed.Root -ne $sharePath -and $observed.DisplayRoot -ne $sharePath) { throw 'Mapped drive postcondition failed.' }
        }

        $results.Add([pscustomobject]@{
            Name = $resource.name
            Drive = $driveLetter
            SharePath = $sharePath
            Success = $success
            Status = if ($success) { 'mapped' } else { 'failed' }
            Message = if ($success) { "Mapped $driveLetter to $sharePath" } else { "Failed to map $driveLetter to $sharePath" }
        })
    }

    if ($results.Count -gt 0) { $results.ToArray() } else { @() }
}

function Sync-ResourceCredentialsFromBitwarden {
    [OutputType([object[]])]
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [string]$ConfigPath = (Join-Path $env:USERPROFILE '.machine\resource-credentials.json'),
        [string]$VaultName = 'LocalStore',
        [string[]]$ResourceNames
    )
    if (-not $PSCmdlet.ShouldProcess('Sync-ResourceCredentialsFromBitwarden state', 'Apply requested backend operation')) { return @() }


    $manifest = Get-ResourceCredentialManifest -ConfigPath $ConfigPath
    if (-not $manifest) {
        return @([pscustomobject]@{ Name = $null; Success = $false; Message = "Resource credential configuration not found: $ConfigPath"; Targets = @() })
    }

    $moduleState = Import-LocalSecretManagement
    $vaultRegistered = $false
    if ($moduleState.SecretManagementAvailable) {
        try {
            $vaultRegistered = [bool](Microsoft.PowerShell.SecretManagement\Get-SecretVault -Name $VaultName -ErrorAction SilentlyContinue -WarningAction SilentlyContinue)
        } catch {
            $vaultRegistered = $false
        }
    }

    $sessionState = Initialize-BitwardenSessionFromBackends -Quiet
    if (-not $sessionState.Success) {
        return @([pscustomobject]@{ Name = $null; Success = $false; Message = $sessionState.Message; Targets = @() })
    }

    $bwSpec = Get-BitwardenCliProcessSpec
    if (-not $bwSpec) {
        return @([pscustomobject]@{ Name = $null; Success = $false; Message = 'bw CLI is not available'; Targets = @() })
    }

    $resources = @($manifest.resources)
    if ($ResourceNames -and $ResourceNames.Count -gt 0) {
        $requested = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        foreach ($resourceName in @($ResourceNames)) {
            if (-not [string]::IsNullOrWhiteSpace([string]$resourceName)) {
                [void]$requested.Add([string]$resourceName)
            }
        }

        $resources = @($resources | Where-Object { $requested.Contains([string]$_.name) })
    }

    if ($resources.Count -eq 0) {
        return @()
    }

    $syncResponse = Invoke-BitwardenCli -Arguments @('sync') -TimeoutSeconds 30
    if (-not $syncResponse.Success) { throw 'Bitwarden sync failed; refusing resource credential deployment.' }

    $results = New-Object System.Collections.Generic.List[object]

    foreach ($resource in $resources) {
        $resolvedItem = Resolve-BitwardenItemForResource -Resource $resource
        if (-not $resolvedItem) {
            $results.Add([pscustomobject]@{
                Name = $resource.name
                Success = $false
                Message = 'No matching Bitwarden item could be resolved from id or search keys'
                LocalSecretStored = $false
                Targets = @()
            })
            continue
        }

        $item = $resolvedItem.Item

        if (-not $item.login -or [string]::IsNullOrWhiteSpace([string]$item.login.password)) {
            $results.Add([pscustomobject]@{
                Name = $resource.name
                Success = $false
                Message = 'Bitwarden item does not contain a login password'
                LocalSecretStored = $false
                Targets = @()
            })
            continue
        }

        $preferredUserName = if ($resource.preferredUsername) { [string]$resource.preferredUsername } elseif ($item.login.username) { [string]$item.login.username } else { $null }
        $localSecretStored = $false
        if ($vaultRegistered -and $resource.secretName) {
            $payload = @{ username = $preferredUserName; password = [string]$item.login.password } | ConvertTo-Json -Compress
            $localSecretStored = Set-PlainTextSecretInSecretManagement -Name $resource.secretName -Secret $payload -VaultName $VaultName
        }

        $targetResults = New-Object System.Collections.Generic.List[object]
        foreach ($target in @($resource.targets)) {
            $userName = if ($target.username) { [string]$target.username } else { $preferredUserName }
            if (-not $userName) {
                $targetResults.Add([pscustomobject]@{
                    Resource = $resource.name
                    Target = $target.target
                    Status = 'missing-username'
                    Message = 'No username available for target'
                })
                continue
            }

            $kind = if ($target.kind -eq 'domain') { 'domain' } else { 'generic' }
            $updated = $false
            if ($PSCmdlet.ShouldProcess($target.target, 'Deploy resource credential target')) {
                $updated = Set-CredentialManagerCredential -Target $target.target -UserName $userName -Password ([string]$item.login.password) -Kind $kind
            }

            $targetResults.Add([pscustomobject]@{
                Resource = $resource.name
                Target = $target.target
                Status = if ($updated) { 'updated' } else { 'failed' }
                Message = if ($updated) { "Updated $kind target for $userName" } else { 'Failed to update target' }
            })
        }

        $failures = @($targetResults | Where-Object { $_.Status -ne 'updated' })
        $messages = New-Object System.Collections.Generic.List[string]
        if (-not $vaultRegistered) {
            $messages.Add("SecretManagement vault '$VaultName' is not registered")
        } elseif ($resource.secretName -and -not $localSecretStored) {
            $messages.Add("Failed to mirror secret '$($resource.secretName)' into $VaultName")
        }
        if ($failures.Count -gt 0) {
            $messages.Add((($failures | Select-Object -ExpandProperty Message | Select-Object -Unique) -join '; '))
        }
        if ($messages.Count -eq 0) {
            $messages.Add('Credential targets updated from Bitwarden')
        }
        $messages.Insert(0, "Bitwarden source $($resolvedItem.Source)")

        $results.Add([pscustomobject]@{
            Name = $resource.name
            Success = ($failures.Count -eq 0)
            Message = ($messages -join '; ')
            LocalSecretStored = $localSecretStored
            Targets = $targetResults
        })
    }

    $results
}

function Get-RealWinGetPackageExecutable {
    [CmdletBinding()]
    [OutputType([string])]
    param([Parameter(Mandatory)][ValidateSet('bw','rclone')][string]$Tool,[string]$UserProfileRoot=$env:USERPROFILE,[string]$ExecutablePath,[string]$ExpectedSha256)
    $packages=[IO.Path]::GetFullPath((Join-Path $UserProfileRoot 'AppData/Local/Microsoft/WinGet/Packages'))
    $packagePrefix=if($Tool-eq'bw'){'Bitwarden.CLI_'}else{'Rclone.Rclone_'}
    if($ExecutablePath){
        if($ExpectedSha256-notmatch'^[A-Fa-f0-9]{64}$'){throw 'Explicit package selection requires a fresh SHA256 attestation.'}
        $full=[IO.Path]::GetFullPath($ExecutablePath)
        if(-not$full.StartsWith((Join-Path $packages $packagePrefix),[StringComparison]::OrdinalIgnoreCase)-or[IO.Path]::GetFileName($full)-ne"$Tool.exe"-or-not(Test-Path -LiteralPath $full -PathType Leaf)){throw 'Explicit executable must belong to the actual tool package, not WinGet Links.'}
        Assert-RealPackageAncestry -Path $full
        if((Get-FileHash -LiteralPath $full).Hash-ne$ExpectedSha256){throw 'Explicit package executable hash mismatch.'}
        return $full
    }
    if(-not(Test-Path -LiteralPath $packages -PathType Container)){return $null}
    Assert-RealPackageAncestry -Path $packages
    $executables=@(foreach($package in Get-ChildItem -LiteralPath $packages -Directory -Filter ($packagePrefix+'*')){
        Assert-RealPackageAncestry -Path $package.FullName
        $directories=if($Tool-eq'bw'){@($package)}else{@(Get-ChildItem -LiteralPath $package.FullName -Directory -Filter 'rclone-v*-windows-amd64')}
        foreach($directory in $directories){
            Assert-RealPackageAncestry -Path $directory.FullName
            $path=Join-Path $directory.FullName "$Tool.exe"
            if(Test-Path -LiteralPath $path -PathType Leaf){Assert-RealPackageAncestry -Path $path;$path}
        }
    })
    if($executables.Count-gt1){throw "Ambiguous actual package executable for $Tool; an explicit reviewed package selection is required."}
    if($executables.Count-eq1){return $executables[0]}
    return $null
}
function Invoke-MachineBitwardenCommand {
    [CmdletBinding()]
    [OutputType([string])]
    param([Parameter(ValueFromRemainingArguments)][string[]]$Arguments)
    $response = Invoke-BitwardenCli -Arguments $Arguments
    $global:LASTEXITCODE = if ($null -ne $response.ExitCode) { $response.ExitCode } else { 1 }
    if (-not $response.Success) {
        # Never attach process stdout/stderr: unlock/list/get output may contain
        # authentication material. Metadata is sufficient at this boundary.
        throw "Bitwarden command failed (exit=$($response.ExitCode), timeout=$($response.TimedOut))."
    }
    return $response.StdOut
}

function ConvertFrom-SecretBackendJson {
    [CmdletBinding()]
    param([Parameter(Mandatory,ValueFromPipeline)][string]$InputObject,[switch]$AsHashtable)
    process {
        try {ConvertFrom-Json -InputObject $InputObject -AsHashtable:$AsHashtable -ErrorAction Stop}
        catch {throw [FormatException]::new('Backend JSON is invalid; private response content is withheld.')}
    }
}
function Assert-RealPackageAncestry {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)
    $ancestor=[IO.Path]::GetFullPath($Path)
    while($ancestor){
        $item=Get-Item -LiteralPath $ancestor -Force -ErrorAction Stop
        if($item.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'Reparse-point package ancestry refused.'}
        $ancestor=[IO.Path]::GetDirectoryName($ancestor)
    }
}
function Assert-BitwardenPrivateInput {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)
    if(-not(Test-Path -LiteralPath $Path -PathType Leaf)){throw 'Private input must be an existing ordinary file.'}
    Assert-RealPackageAncestry -Path $Path
    $allowed=@([Security.Principal.WindowsIdentity]::GetCurrent().User.Value,'S-1-5-18','S-1-5-32-544')
    $acl=Get-Acl -LiteralPath $Path -ErrorAction Stop
    foreach($rule in $acl.GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier])){
        if($rule.AccessControlType-eq[Security.AccessControl.AccessControlType]::Allow-and$allowed-notcontains$rule.IdentityReference.Value){throw 'Private input has an unqualified access rule; explicit ACL repair is required.'}
    }
}
# Private, process-local custody survives module reloads; it is never persisted.
if (-not ('WorkProfileBackend.PendingProcessCustody' -as [type])) {
    Add-Type -TypeDefinition @'
namespace WorkProfileBackend {
    public static class PendingProcessCustody {
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, object> entries =
            new System.Collections.Concurrent.ConcurrentDictionary<string, object>();
        public static string Retain(object entry) {
            var id = System.Guid.NewGuid().ToString("N");
            if (!entries.TryAdd(id, entry)) throw new System.InvalidOperationException("Custody registration failed.");
            return id;
        }
        public static object Find(string id) {
            object entry;
            return entries.TryGetValue(id, out entry) ? entry : null;
        }
        public static bool Release(string id) {
            object entry;
            return entries.TryRemove(id, out entry);
        }
        public static string[] Ids() {
            var keys = entries.Keys;
            var ids = new string[keys.Count];
            keys.CopyTo(ids, 0);
            return ids;
        }
    }
}
'@
}

function Invoke-SecretBackendOwnedTermination {
    [CmdletBinding()]
    param([Parameter(Mandatory)][Diagnostics.Process]$Process)
    $Process.Kill($true)
}

function Read-SecretBackendOwnedOutput {
    [CmdletBinding()][OutputType([Threading.Tasks.Task[string]])]
    param([Parameter(Mandatory)][IO.StreamReader]$Reader)
    return $Reader.ReadToEndAsync()
}

function Wait-SecretBackendOwnedProcess {
    [CmdletBinding()][OutputType([bool])]
    param([Parameter(Mandatory)][Diagnostics.Process]$Process,
        [Parameter(Mandatory)][ValidateRange(0,120000)][int]$Milliseconds)
    return $Process.WaitForExit($Milliseconds)
}

function Wait-SecretBackendOwnedPipeClosure {
    [CmdletBinding()][OutputType([bool])]
    param([Parameter(Mandatory)][Threading.Tasks.Task[]]$Tasks)
    return [Threading.Tasks.Task]::WhenAll($Tasks).Wait(5000)
}

function Close-SecretBackendOwnedProcess {
    [CmdletBinding()]
    param([Parameter(Mandatory)][object]$Entry)
    # Never reacquire by PID: retries use the original Process and pipe tasks.
    if (-not $Entry.Process.HasExited) {
        try { Invoke-SecretBackendOwnedTermination -Process $Entry.Process }
        catch [InvalidOperationException] {
            if (-not $Entry.Process.HasExited) { throw }
        }
    }
    if (-not (Wait-SecretBackendOwnedProcess -Process $Entry.Process -Milliseconds 5000)) {
        throw [TimeoutException]::new('Owned backend exit remains unconfirmed.')
    }
    $tasks = [Threading.Tasks.Task[]]@(@($Entry.StdOutTask,$Entry.StdErrTask,$Entry.InputTask) | Where-Object { $null -ne $_ })
    if ($tasks.Count) {
        try { $completed = Wait-SecretBackendOwnedPipeClosure -Tasks $tasks }
        catch {
            # Faulted reads/writes are observed, but only settled tasks permit disposal.
            $completed = @($tasks | Where-Object { -not $_.IsCompleted }).Count -eq 0
            if (-not $completed) { throw }
        }
        if (-not $completed) { throw [TimeoutException]::new('Owned backend pipes remain unconfirmed.') }
    }
    # Explicit stream disposal and Process disposal happen only after exit/task confirmation.
    if ($Entry.RedirectInput) { $Entry.Process.StandardInput.Dispose() }
    $Entry.Process.StandardOutput.Dispose()
    $Entry.Process.StandardError.Dispose()
    $Entry.Process.Dispose()
}

function Get-SecretBackendProcessRecovery {
    [CmdletBinding()][OutputType([pscustomobject])]
    param([ValidatePattern('^[a-f0-9]{32}$')][string]$CustodyId)
    $ids = if ($PSBoundParameters.ContainsKey('CustodyId')) { @($CustodyId) } else { [WorkProfileBackend.PendingProcessCustody]::Ids() }
    foreach ($id in $ids) {
        $entry = [WorkProfileBackend.PendingProcessCustody]::Find($id)
        [pscustomobject]@{ CustodyId=$id; CleanupPending=($null -ne $entry) }
    }
}

function Repair-SecretBackendProcessCustody {
    [CmdletBinding()][OutputType([pscustomobject])]
    param([Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$CustodyId)
    $entry = [WorkProfileBackend.PendingProcessCustody]::Find($CustodyId)
    if ($null -eq $entry) { throw [InvalidOperationException]::new('Owned backend recovery identity is unavailable.') }
    [Threading.Monitor]::Enter($entry)
    try {
        if ($null -eq [WorkProfileBackend.PendingProcessCustody]::Find($CustodyId)) {
            throw [InvalidOperationException]::new('Owned backend recovery identity is unavailable.')
        }
        try { Close-SecretBackendOwnedProcess -Entry $entry }
        catch {
            $failure = [InvalidOperationException]::new('Backend owned-child recovery remains pending; private command output is withheld.')
            $failure.Data['CustodyId'] = $CustodyId
            $failure.Data['CleanupPending'] = $true
            throw $failure
        }
        if (-not [WorkProfileBackend.PendingProcessCustody]::Release($CustodyId)) {
            throw [InvalidOperationException]::new('Owned backend custody release could not be confirmed.')
        }
        return [pscustomobject]@{ CustodyId=$CustodyId; CleanupPending=$false; RecoveryCompleted=$true }
    } finally { [Threading.Monitor]::Exit($entry) }
}

function Invoke-SecretBackendProcess {
    [CmdletBinding()][OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments=@(),
        [ValidateRange(1,120)][int]$TimeoutSeconds=20,
        [string]$StandardInput
    )
    # Admission is process-wide: known pending custody blocks every new backend
    # invocation until explicit exact-object recovery releases it. This snapshot
    # precedes Process allocation. It does not serialize calls already admitted
    # before another invocation registers a cleanup failure.
    $pendingCustodyIds = @([WorkProfileBackend.PendingProcessCustody]::Ids())
    if ($pendingCustodyIds.Count) {
        $failure = [InvalidOperationException]::new('Backend owned-child cleanup is pending; recover owned custody before starting another invocation. Private command output is withheld.')
        $failure.Data['CustodyIds'] = [string[]]$pendingCustodyIds
        $failure.Data['CleanupPending'] = $true
        $failure.Data['AdmissionBlocked'] = $true
        throw $failure
    }
    $process = [Diagnostics.Process]::new()
    $started = $false
    $retained = $false
    $disposed = $false
    $entry = [pscustomobject]@{ Process=$process; StdOutTask=$null; StdErrTask=$null; InputTask=$null; RedirectInput=$false }
    try {
        $info = [Diagnostics.ProcessStartInfo]::new()
        $info.FileName = $FilePath
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $info.RedirectStandardInput = $PSBoundParameters.ContainsKey('StandardInput')
        $entry.RedirectInput = $info.RedirectStandardInput
        foreach ($argument in $Arguments) { [void]$info.ArgumentList.Add($argument) }
        $process.StartInfo = $info
        $started = $process.Start()
        if (-not $started) { throw [InvalidOperationException]::new('Backend process did not start.') }
        $entry.StdOutTask = Read-SecretBackendOwnedOutput -Reader $process.StandardOutput
        $entry.StdErrTask = Read-SecretBackendOwnedOutput -Reader $process.StandardError
        $deadline = [Diagnostics.Stopwatch]::StartNew()
        $inputCompleted = $true
        if ($info.RedirectStandardInput) {
            $entry.InputTask = $process.StandardInput.WriteAsync($StandardInput)
            $inputCompleted = $entry.InputTask.Wait($TimeoutSeconds * 1000)
            if ($inputCompleted) { $process.StandardInput.Close() }
        }
        $remaining = [Math]::Max(0, $TimeoutSeconds * 1000 - [int]$deadline.ElapsedMilliseconds)
        if (-not $inputCompleted -or -not (Wait-SecretBackendOwnedProcess -Process $process -Milliseconds $remaining)) {
            Close-SecretBackendOwnedProcess -Entry $entry
            $disposed = $true
            return [pscustomobject]@{ Success=$false; TimedOut=$true; ExitCode=$null; StdOut=$null; StdErr='Backend process timed out.' }
        }
        if (-not (Wait-SecretBackendOwnedPipeClosure -Tasks ([Threading.Tasks.Task[]]@($entry.StdOutTask,$entry.StdErrTask)))) {
            throw [TimeoutException]::new('Backend output pipes did not close.')
        }
        $output = $entry.StdOutTask.GetAwaiter().GetResult()
        $null = $entry.StdErrTask.GetAwaiter().GetResult()
        return [pscustomobject]@{
            Success=($process.ExitCode -eq 0); TimedOut=$false; ExitCode=$process.ExitCode
            StdOut=if ($process.ExitCode -eq 0) { $output.Trim() } else { $null }
            StdErr=if ($process.ExitCode -eq 0) { $null } else { 'Backend process returned a nonzero exit.' }
        }
    } catch {
        if ($started -and -not $disposed) {
            try { Close-SecretBackendOwnedProcess -Entry $entry; $disposed = $true }
            catch {
                $custodyId = [WorkProfileBackend.PendingProcessCustody]::Retain($entry)
                $retained = $true
                $failure = [InvalidOperationException]::new('Backend process failed and owned-child cleanup remains pending; private command output is withheld.')
                $failure.Data['CustodyId'] = $custodyId
                $failure.Data['CleanupPending'] = $true
                throw $failure
            }
        }
        throw [InvalidOperationException]::new('Backend process could not be completed; private command output is withheld.')
    } finally {
        if (-not $retained -and -not $disposed) { $process.Dispose() }
    }
}
