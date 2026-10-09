#Requires -PSEdition Core

function Get-LLMConfigBytesHash {
    param([byte[]]$Bytes)
    $hash = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hash.ComputeHash($Bytes))).Replace('-', '') }
    finally { $hash.Dispose() }
}

function Assert-LLMConfigWritePath {
    param([Parameter(Mandatory)][string]$Path)
    $provider = $null
    $drive = $null
    $providerPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path, [ref]$provider, [ref]$drive)
    if ($provider.Name -ne 'FileSystem') { throw 'Configuration writes require a filesystem path.' }
    $fullPath = [IO.Path]::GetFullPath($providerPath)
    if ($fullPath -eq [IO.Path]::GetPathRoot($fullPath)) { throw 'Configuration path cannot select a filesystem root.' }
    if ($IsWindows -and $fullPath.Substring([IO.Path]::GetPathRoot($fullPath).Length).Contains(':')) { throw 'Configuration writes cannot select alternate data streams.' }
    $cursor = $fullPath
    while ($cursor) {
        $name = [IO.Path]::GetFileName($cursor).TrimEnd(' ', '.')
        if ($name -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.|$)') { throw 'Configuration path contains an unsafe Windows name.' }
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force -ErrorAction Stop
            if ($item.PSProvider.Name -ne 'FileSystem' -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
                ($item.PSObject.Properties['LinkType'] -and $item.LinkType -eq 'HardLink')) {
                throw 'Configuration writes cannot use non-filesystem or linked paths.'
            }
        }
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ($parent -eq $cursor) { break }
        $cursor = $parent
    }
    return $fullPath
}

function Read-LLMConfigSnapshot {
    param([Parameter(Mandatory)][string]$Path)
    $pathValue = Assert-LLMConfigWritePath -Path $Path
    $exists = Test-Path -LiteralPath $pathValue -PathType Leaf
    if ((Test-Path -LiteralPath $pathValue) -and -not $exists) { throw 'Configuration path must select a file.' }
    $bytes = [byte[]]@()
    if ($exists) { $bytes = [IO.File]::ReadAllBytes($pathValue) }
    $configuration = if ($exists) {
        $text = [Text.UTF8Encoding]::new($false, $true).GetString($bytes).TrimStart([char]0xFEFF)
        if (-not $text.TrimStart().StartsWith('{', [StringComparison]::Ordinal)) { throw 'LLM configuration must be a JSON object.' }
        $text | ConvertFrom-Json -Depth 100 -ErrorAction Stop
    } else { [pscustomobject]@{} }
    if ($configuration -isnot [pscustomobject]) { throw 'LLM configuration must be a JSON object.' }
    return [pscustomobject]@{ Path=$pathValue; Exists=$exists; Bytes=$bytes; Hash=(Get-LLMConfigBytesHash -Bytes $bytes); Configuration=$configuration }
}

function Set-LLMConfigPrivateAcl {
    param([Parameter(Mandatory)][string]$Path, [switch]$Directory)
    if ($IsWindows) {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent().User
        $acl = if ($Directory) { [Security.AccessControl.DirectorySecurity]::new() } else { [Security.AccessControl.FileSecurity]::new() }
        $acl.SetOwner($identity)
        $acl.SetAccessRuleProtection($true, $false)
        $inheritance = if ($Directory) { [Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit' } else { [Security.AccessControl.InheritanceFlags]::None }
        foreach ($sid in @($identity, [Security.Principal.SecurityIdentifier]::new('S-1-5-18'), [Security.Principal.SecurityIdentifier]::new('S-1-5-32-544'))) {
            $rule = [Security.AccessControl.FileSystemAccessRule]::new($sid, [Security.AccessControl.FileSystemRights]::FullControl, $inheritance, [Security.AccessControl.PropagationFlags]::None, [Security.AccessControl.AccessControlType]::Allow)
            [void]$acl.AddAccessRule($rule)
        }
        Set-Acl -LiteralPath $Path -AclObject $acl -ErrorAction Stop
    } else {
        # Private custody requires native permissions; fail closed if unavailable.
        $mode = if ($Directory) { [IO.UnixFileMode]'UserRead,UserWrite,UserExecute' } else { [IO.UnixFileMode]'UserRead,UserWrite' }
        [IO.File]::SetUnixFileMode($Path, $mode)
    }
}

function Write-LLMConfigStage {
    param([Parameter(Mandatory)][string]$Path, [byte[]]$Bytes)
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        Set-LLMConfigPrivateAcl -Path $Path
        $stream.Write($Bytes, 0, $Bytes.Length)
        $stream.Flush($true)
    } finally { $stream.Dispose() }
}

function Get-LLMConfigCurrentHash {
    param([Parameter(Mandatory)][string]$Path)
    return Get-LLMConfigBytesHash -Bytes ([IO.File]::ReadAllBytes($Path))
}

function Initialize-LLMConfigNativeAcl {
    # The documented user-mode Nt APIs preserve raw ACE order and legacy
    # inheritance flags that SetSecurityInfo (including the managed setter) changes.
    if ('Pcai.Config.NativeAclV1' -as [type]) {
        if ([Pcai.Config.NativeAclV1]::ProtocolVersion -ne 1) { throw 'Unknown configuration ACL interop version.' }
        return
    }
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
namespace Pcai.Config {
 public static class NativeAclV1 {
  public const int ProtocolVersion = 1;
  [StructLayout(LayoutKind.Sequential)] private struct FileIdInfo { public ulong Volume, Low, High; }
  [DllImport("kernel32.dll", SetLastError=true, ExactSpelling=true)]
  [return: MarshalAs(UnmanagedType.Bool)]
  private static extern bool GetFileInformationByHandleEx(SafeFileHandle handle, int information, out FileIdInfo identity, uint size);
  [DllImport("ntdll.dll", ExactSpelling=true)]
  private static extern int NtQuerySecurityObject(SafeFileHandle handle, uint information, [Out] byte[] descriptor, uint length, out uint required);
  [DllImport("ntdll.dll", ExactSpelling=true)]
  private static extern int NtSetSecurityObject(SafeFileHandle handle, uint information, [In] byte[] descriptor);
  [DllImport("ntdll.dll", ExactSpelling=true)]
  private static extern uint RtlNtStatusToDosError(int status);
  public static string Identity(SafeFileHandle handle) {
   FileIdInfo identity;
   if (!GetFileInformationByHandleEx(handle, 18, out identity, 24)) throw new Win32Exception(Marshal.GetLastWin32Error());
   return identity.Volume.ToString("X16") + ":" + identity.High.ToString("X16") + identity.Low.ToString("X16");
  }
  public static byte[] ReadDescriptor(SafeFileHandle handle) {
   // NtQuerySecurityObject documents the NTFS on-disk descriptor limit as 64 KiB.
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

function Read-LLMConfigOwnedBytes {
    param([IO.FileStream]$Stream)
    $position = $Stream.Position
    $buffer = [IO.MemoryStream]::new()
    try { $Stream.Position = 0; $Stream.CopyTo($buffer); return ,$buffer.ToArray() }
    finally { $Stream.Position = $position; $buffer.Dispose() }
}

function Set-LLMConfigOwnedRecoveryAcl {
    param([IO.FileStream]$Stream, [Security.AccessControl.FileSecurity]$Acl, [byte[]]$Descriptor)
    Initialize-LLMConfigNativeAcl
    if (-not $Descriptor) { $Descriptor = $Acl.GetSecurityDescriptorBinaryForm() }
    $raw = [Security.AccessControl.RawSecurityDescriptor]::new($Descriptor, 0)
    # Request the original auto-inheritance model without converting legacy ACLs.
    if ($raw.ControlFlags -band [Security.AccessControl.ControlFlags]::DiscretionaryAclAutoInherited) {
        $raw.SetFlags($raw.ControlFlags -bor [Security.AccessControl.ControlFlags]::DiscretionaryAclAutoInheritRequired)
    }
    $submitted = [byte[]]::new($raw.BinaryLength)
    $raw.GetBinaryForm($submitted, 0)
    [Pcai.Config.NativeAclV1]::WriteDescriptor($Stream.SafeFileHandle, $submitted)
}

function Test-LLMConfigOwnedTarget {
    param([string]$Path, [IO.FileStream]$Owned, [string]$ExpectedHash, [Collections.IDictionary]$Receipt)
    $current = $null
    try {
        $current = [IO.FileSystemAclExtensions]::Create([IO.FileInfo]::new($Path), [IO.FileMode]::Open,
            [Security.AccessControl.FileSystemRights]'ReadData,ReadPermissions', [IO.FileShare]'ReadWrite,Delete', 4096, [IO.FileOptions]::None, $null)
        $Receipt.FailureCurrentFileIdentity = [Pcai.Config.NativeAclV1]::Identity($current.SafeFileHandle)
        $Receipt.FailureCurrentSHA256 = Get-LLMConfigBytesHash -Bytes (Read-LLMConfigOwnedBytes -Stream $current)
        return ($Receipt.FailureCurrentFileIdentity -ceq [Pcai.Config.NativeAclV1]::Identity($Owned.SafeFileHandle) -and
            $Receipt.FailureCurrentSHA256 -ceq $ExpectedHash)
    } catch [IO.FileNotFoundException] {
        $Receipt.FailureCurrentFileIdentity = $null
        return $false
    } catch [IO.DirectoryNotFoundException] {
        $Receipt.FailureCurrentFileIdentity = $null
        return $false
    } finally { if ($current) { $current.Dispose() } }
}

function Restore-LLMConfigMissingTarget {
    param([string]$SourcePath, [string]$TargetPath, [string]$TransactionPath, [Collections.IDictionary]$Receipt, [psobject]$OriginalMetadata)
    # ReplaceFile error1177 can move the original to backup before failing.
    # Keep that actual backup and restore only into an absent target, never over
    # a writer that arrived afterwards. Move without overwrite decides the race.
    $source = [IO.FileSystemAclExtensions]::Create([IO.FileInfo]::new($SourcePath), [IO.FileMode]::Open,
        [Security.AccessControl.FileSystemRights]'ReadData,ReadPermissions', [IO.FileShare]'ReadWrite,Delete', 4096, [IO.FileOptions]::None, $null)
    try {
        $bytes = Read-LLMConfigOwnedBytes -Stream $source
        $displacedIdentity = [Pcai.Config.NativeAclV1]::Identity($source.SafeFileHandle)
        $displacedDescriptor = [Pcai.Config.NativeAclV1]::ReadDescriptor($source.SafeFileHandle)
    } finally { $source.Dispose() }
    $restoreHash = Get-LLMConfigBytesHash -Bytes $bytes
    $Receipt.DisplacedSHA256 = $restoreHash
    $Receipt.DisplacedFileIdentity = $displacedIdentity
    $Receipt.DisplacedWindowsDescriptorSHA256 = Get-LLMConfigBytesHash -Bytes $displacedDescriptor
    Write-LLMConfigStage -Path (Join-Path $TransactionPath 'displaced-security-descriptor.bin') -Bytes $displacedDescriptor
    if (-not $OriginalMetadata -or $displacedIdentity -cne $OriginalMetadata.Identity -or $restoreHash -cne $OriginalMetadata.Hash) {
        throw 'Actual displaced original identity or bytes differ; original metadata will not be applied to a foreign writer.'
    }
    $originalAcl = $OriginalMetadata.Acl
    $originalDescriptor = $OriginalMetadata.Descriptor
    $originalSddl = ([Security.AccessControl.RawSecurityDescriptor]::new($originalDescriptor, 0)).GetSddlForm([Security.AccessControl.AccessControlSections]::All)
    $aclHash = Get-LLMConfigBytesHash -Bytes ([Text.Encoding]::UTF8.GetBytes($originalSddl))
    Set-LLMConfigPrivateAcl -Path $SourcePath
    if (Test-Path -LiteralPath $TargetPath) {
        $Receipt.FailureCurrentSHA256 = Get-LLMConfigCurrentHash -Path $TargetPath
        return 'LaterWriterPreserved'
    }
    $stage = Join-Path $TransactionPath 'recovery-missing-target.json'
    Write-LLMConfigStage -Path $stage -Bytes $bytes
    if ((Get-LLMConfigCurrentHash -Path $stage) -cne $restoreHash) { throw 'Missing-target recovery candidate hash mismatch.' }
    $Receipt.Recovery.Add([pscustomobject]@{ RestoredSHA256=$restoreHash; StagePath=$stage; PreservedWindowsAclSHA256=$aclHash; MetadataSource='IdentityBoundPrePublicationWindowsAcl' })
    # Pin our own candidate with WRITE_DAC and delete sharing across the move.
    # Applying inherited ACLs while inside custody derives the wrong parent;
    # apply them after the move through this handle, never the target pathname.
    # 4096 is the standard FileStream buffer, not a workload/resource setting.
    $rights = [Security.AccessControl.FileSystemRights]'ReadData,ReadPermissions,ChangePermissions,TakeOwnership'
    $owned = [IO.FileSystemAclExtensions]::Create([IO.FileInfo]::new($stage), [IO.FileMode]::Open, $rights, [IO.FileShare]'ReadWrite,Delete', 4096, [IO.FileOptions]::None, $null)
    try {
        try { [IO.File]::Move($stage, $TargetPath) }
        catch {
            if (Test-Path -LiteralPath $TargetPath -PathType Leaf) {
                $Receipt.FailureCurrentSHA256 = Get-LLMConfigCurrentHash -Path $TargetPath
                return 'LaterWriterPreserved'
            }
            throw
        }
        Set-LLMConfigOwnedRecoveryAcl -Stream $owned -Acl $originalAcl -Descriptor $originalDescriptor
        if (-not (Test-LLMConfigOwnedTarget -Path $TargetPath -Owned $owned -ExpectedHash $restoreHash -Receipt $Receipt)) { return 'LaterWriterPreserved' }
        $actualDescriptor = [Pcai.Config.NativeAclV1]::ReadDescriptor($owned.SafeFileHandle)
        $actualSddl = ([Security.AccessControl.RawSecurityDescriptor]::new($actualDescriptor, 0)).GetSddlForm([Security.AccessControl.AccessControlSections]::All)
        $actualAclHash = Get-LLMConfigBytesHash -Bytes ([Text.Encoding]::UTF8.GetBytes($actualSddl))
        if ($actualAclHash -cne $aclHash) { throw 'Missing-target recovery Windows ACL mismatch.' }
        # A different file can contain identical bytes. Keep the owned candidate
        # open through the final identity observation, not just its hash check.
        if (-not (Test-LLMConfigOwnedTarget -Path $TargetPath -Owned $owned -ExpectedHash $restoreHash -Receipt $Receipt)) { return 'LaterWriterPreserved' }
        return 'Restored'
    } finally { $owned.Dispose() }
}

function Restore-LLMConfigDisplaced {
    param([string]$SourcePath, [string]$ExpectedCurrentHash, [string]$TargetPath, [string]$TransactionPath, [Collections.IDictionary]$Receipt)
    # Bound non-cooperating writer races exactly as the profile repair does.
    # Every replacement captures its displaced bytes; a newer current is retained.
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        if (-not (Test-Path -LiteralPath $TargetPath -PathType Leaf)) { return 'LaterWriterPreserved' }
        $currentHash = Get-LLMConfigCurrentHash -Path $TargetPath
        $Receipt.FailureCurrentSHA256 = $currentHash
        if ($currentHash -cne $ExpectedCurrentHash) { return 'LaterWriterPreserved' }
        $bytes = [IO.File]::ReadAllBytes($SourcePath)
        $restoreHash = Get-LLMConfigBytesHash -Bytes $bytes
        $stage = Join-Path $TransactionPath "recovery-stage-r$attempt.json"
        $displaced = Join-Path $TransactionPath "recovery-displaced-r$attempt.bin"
        Write-LLMConfigStage -Path $stage -Bytes $bytes
        [IO.File]::Replace($stage, $TargetPath, $displaced)
        Set-LLMConfigPrivateAcl -Path $displaced
        $displacedHash = Get-LLMConfigCurrentHash -Path $displaced
        $Receipt.Recovery.Add([pscustomobject]@{ RestoredSHA256=$restoreHash; DisplacedPath=$displaced; DisplacedSHA256=$displacedHash; ExpectedDisplacedSHA256=$ExpectedCurrentHash })
        if ($displacedHash -ceq $ExpectedCurrentHash) {
            if ((Get-LLMConfigCurrentHash -Path $TargetPath) -cne $restoreHash) { return 'LaterWriterPreserved' }
            return 'Restored'
        }
        $SourcePath = $displaced
        $ExpectedCurrentHash = $restoreHash
    }
    return 'RequiresReview'
}

function Save-LLMConfigAtomically {
    <#
    .SYNOPSIS
        Preserves original config bytes before atomic publication.
    .DESCRIPTION
        Rejects serialization warnings before custody or publication. Retains
        every captured version under ignored private same-volume custody.
        Original-byte checks detect concurrent writers; replacement captures
        boundary races, with bounded recovery rather than a hash compare-and-swap.
        Missing-target recovery binds the original bytes and raw owner/group/DACL
        descriptor to a retained original file handle before publication. Exact
        descriptor and file-identity observations decide successful restoration;
        mismatches preserve private custody for review. SACL/audit preservation
        is outside this descriptor capture and recovery contract.
        Publication currently requires Windows. Unix File.Replace does not
        atomically capture the exchanged original; other platforms fail closed
        pending independently qualified atomic exchange and metadata admission.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][psobject]$Configuration, [Parameter(Mandatory)][psobject]$Snapshot)
    if (-not $IsWindows) {
        throw [PlatformNotSupportedException]::new('LLM configuration publication requires Windows; other platforms await qualified atomic exchange and metadata preservation.')
    }
    # 100 is the PowerShell serializer maximum, not a tuning or load limit.
    $json = $Configuration | ConvertTo-Json -Depth 100 -WarningAction Stop -ErrorAction Stop
    if (-not $json.TrimStart().StartsWith('{', [StringComparison]::Ordinal)) { throw 'Serialized LLM configuration must remain a JSON object.' }
    $validated = $json | ConvertFrom-Json -Depth 100 -ErrorAction Stop
    if ($validated -isnot [pscustomobject]) { throw 'Serialized LLM configuration must remain a JSON object.' }
    $bytes = [Text.UTF8Encoding]::new($false, $true).GetBytes($json)
    $afterHash = Get-LLMConfigBytesHash -Bytes $bytes
    $target = Assert-LLMConfigWritePath -Path $Snapshot.Path
    $selectedPath = if (-not [string]::IsNullOrWhiteSpace($script:ModuleConfig.ProjectConfigPath)) { $script:ModuleConfig.ProjectConfigPath } else { $script:ModuleConfig.ConfigPath }
    $selectedTarget = Assert-LLMConfigWritePath -Path $selectedPath
    $comparison = if ($IsWindows) { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
    if (-not [string]::Equals($target, $selectedTarget, $comparison)) { throw 'Configuration snapshot is outside the selected module configuration path.' }
    if ((Get-LLMConfigBytesHash -Bytes $Snapshot.Bytes) -cne $Snapshot.Hash) { throw 'Original configuration snapshot hash is inconsistent.' }
    $parent = [IO.Path]::GetDirectoryName($target)
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) { [void](New-Item -ItemType Directory -Path $parent -Force -ErrorAction Stop) }
    $identityPath = if ($IsWindows) { $target.ToUpperInvariant() } else { $target }
    $pathHash = Get-LLMConfigBytesHash -Bytes ([Text.Encoding]::UTF8.GetBytes($identityPath))
    $custody = Join-Path $parent ('.pcai/config-write/' + $pathHash.ToLowerInvariant())
    [void](Assert-LLMConfigWritePath -Path $custody)
    [void](New-Item -ItemType Directory -Path $custody -Force -ErrorAction Stop)
    Set-LLMConfigPrivateAcl -Path $custody -Directory
    $lockPath = Join-Path $custody 'publish.lock'
    [void](Assert-LLMConfigWritePath -Path $lockPath)
    $lock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $receipt = $null
    $receiptPath = $null
    $published = $false
    $originalOwned = $null
    $originalMetadata = $null
    try {
        Set-LLMConfigPrivateAcl -Path $lockPath
        $revision = 1
        while (Test-Path -LiteralPath (Join-Path $custody "r$revision")) { $revision++ }
        $transaction = Join-Path $custody "r$revision"
        [void](New-Item -ItemType Directory -Path $transaction -ErrorAction Stop)
        Set-LLMConfigPrivateAcl -Path $transaction -Directory
        $stage = Join-Path $transaction 'staged.json'
        $original = Join-Path $transaction 'original.bin'
        $displaced = Join-Path $transaction 'displaced-original.bin'
        $receiptPath = Join-Path $transaction 'receipt.json'
        $receipt = [ordered]@{ SchemaVersion=1; TargetPath=$target; OriginalExists=$Snapshot.Exists; OriginalSHA256=$Snapshot.Hash; ProposedSHA256=$afterHash; DisplacedSHA256=$null; PublishedObservedSHA256=$null; FailureCurrentSHA256=$null; PartialPublicationObserved=$false; State='Staging'; RecoveryState=$null; RecoveryErrorType=$null; RecoveryErrorHResult=$null; Recovery=[Collections.Generic.List[object]]::new() }
        Write-LLMConfigStage -Path $receiptPath -Bytes ([Text.Encoding]::UTF8.GetBytes(($receipt | ConvertTo-Json -Depth 6)))
        if ($Snapshot.Exists) { Write-LLMConfigStage -Path $original -Bytes $Snapshot.Bytes }
        Write-LLMConfigStage -Path $stage -Bytes $bytes
        if ((Get-LLMConfigCurrentHash -Path $stage) -cne $afterHash) { throw 'Staged configuration hash mismatch.' }
        [void](Assert-LLMConfigWritePath -Path $target)
        if ($Snapshot.Exists) {
            # Capture before Move/Replace can alter inheritance in protected custody.
            # Keep this exact original open so file identity cannot be recycled.
            Initialize-LLMConfigNativeAcl
            $originalOwned = [IO.FileSystemAclExtensions]::Create([IO.FileInfo]::new($target), [IO.FileMode]::Open,
                [Security.AccessControl.FileSystemRights]'ReadData,ReadPermissions', [IO.FileShare]'ReadWrite,Delete', 4096, [IO.FileOptions]::None, $null)
            $ownedHash = Get-LLMConfigBytesHash -Bytes (Read-LLMConfigOwnedBytes -Stream $originalOwned)
            if ($ownedHash -cne $Snapshot.Hash) { throw 'Configuration changed after reading; publication refused.' }
            $originalMetadata = [pscustomobject]@{
                Identity=[Pcai.Config.NativeAclV1]::Identity($originalOwned.SafeFileHandle)
                Hash=$ownedHash
                Descriptor=[Pcai.Config.NativeAclV1]::ReadDescriptor($originalOwned.SafeFileHandle)
                Acl=[IO.FileSystemAclExtensions]::GetAccessControl($originalOwned)
            }
            $receipt.OriginalFileIdentity = $originalMetadata.Identity
            $receipt.OriginalWindowsDescriptorSHA256 = Get-LLMConfigBytesHash -Bytes $originalMetadata.Descriptor
            Write-LLMConfigStage -Path (Join-Path $transaction 'original-security-descriptor.bin') -Bytes $originalMetadata.Descriptor
            if (-not (Test-Path -LiteralPath $target -PathType Leaf) -or (Get-LLMConfigCurrentHash -Path $target) -cne $Snapshot.Hash) { throw 'Configuration changed after reading; publication refused.' }
            [IO.File]::Replace($stage, $target, $displaced)
            $published = $true
            Set-LLMConfigPrivateAcl -Path $displaced
            $receipt.DisplacedSHA256 = Get-LLMConfigCurrentHash -Path $displaced
            if ($receipt.DisplacedSHA256 -cne $Snapshot.Hash) { throw 'Concurrent writer arrived at replacement boundary; its actual bytes are retained.' }
        } else {
            if (Test-Path -LiteralPath $target) { throw 'Configuration appeared after reading; publication refused.' }
            [IO.File]::Move($stage, $target)
            $published = $true
        }
        $receipt.PublishedObservedSHA256 = Get-LLMConfigCurrentHash -Path $target
        if ($receipt.PublishedObservedSHA256 -cne $afterHash) { throw 'Published configuration changed; current writer bytes are retained.' }
        $receipt.State = 'Published'
        [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
    } catch {
        $failure = $_
        if (-not $published -and $receipt -and $Snapshot.Exists -and (Test-Path -LiteralPath $displaced -PathType Leaf)) {
            $receipt.PartialPublicationObserved = $true
            try { $receipt.RecoveryState = Restore-LLMConfigMissingTarget -SourcePath $displaced -TargetPath $target -TransactionPath $transaction -Receipt $receipt -OriginalMetadata $originalMetadata }
            catch { $receipt.RecoveryState = 'RequiresReview'; $receipt.RecoveryErrorType = $_.Exception.GetBaseException().GetType().FullName; $receipt.RecoveryErrorHResult = $_.Exception.GetBaseException().HResult }
        } elseif ($published -and $Snapshot.Exists) {
            try { $receipt.RecoveryState = Restore-LLMConfigDisplaced -SourcePath $displaced -ExpectedCurrentHash $afterHash -TargetPath $target -TransactionPath $transaction -Receipt $receipt }
            catch { $receipt.RecoveryState = 'RequiresReview'; $receipt.RecoveryErrorType = $_.Exception.GetBaseException().GetType().FullName; $receipt.RecoveryErrorHResult = $_.Exception.GetBaseException().HResult }
        } elseif ($published) {
            # There was no original to restore. Retain the valid new/current file
            # and all private stage/custody rather than removing a later writer.
            $receipt.RecoveryState = 'NewFileRetainedForReview'
        }
        if ($receipt) {
            $receipt.State = if (-not $published -and -not $receipt.PartialPublicationObserved) { 'FailedBeforePublish' }
                elseif ($receipt.RecoveryState -eq 'Restored') { 'RolledBack' }
                elseif ($receipt.RecoveryState -eq 'LaterWriterPreserved') { 'LaterWriterPreserved' }
                else { 'RecoveryRequiresReview' }
            try { [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false)) }
            catch { Write-Warning "Configuration custody retained; receipt update failed: $receiptPath" }
        }
        throw $failure
    } finally { if ($originalOwned) { $originalOwned.Dispose() }; $lock.Dispose() }
}
