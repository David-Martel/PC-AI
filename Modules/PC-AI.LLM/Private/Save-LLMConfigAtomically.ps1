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

function Set-LLMConfigOwnedRecoveryAcl {
    param([IO.FileStream]$Stream, [Security.AccessControl.FileSecurity]$Acl)
    [IO.FileSystemAclExtensions]::SetAccessControl($Stream, $Acl)
}

function Restore-LLMConfigMissingTarget {
    param([string]$SourcePath, [string]$TargetPath, [string]$TransactionPath, [Collections.IDictionary]$Receipt)
    # ReplaceFile error1177 can move the original to backup before failing.
    # Keep that actual backup and restore only into an absent target, never over
    # a writer that arrived afterwards. Move without overwrite decides the race.
    $bytes = [IO.File]::ReadAllBytes($SourcePath)
    $restoreHash = Get-LLMConfigBytesHash -Bytes $bytes
    $Receipt.DisplacedSHA256 = $restoreHash
    $originalAcl = Get-Acl -LiteralPath $SourcePath -ErrorAction Stop
    $aclHash = Get-LLMConfigBytesHash -Bytes ([Text.Encoding]::UTF8.GetBytes($originalAcl.GetSecurityDescriptorSddlForm([Security.AccessControl.AccessControlSections]::All)))
    Set-LLMConfigPrivateAcl -Path $SourcePath
    if (Test-Path -LiteralPath $TargetPath) {
        $Receipt.FailureCurrentSHA256 = Get-LLMConfigCurrentHash -Path $TargetPath
        return 'LaterWriterPreserved'
    }
    $stage = Join-Path $TransactionPath 'recovery-missing-target.json'
    Write-LLMConfigStage -Path $stage -Bytes $bytes
    if ((Get-LLMConfigCurrentHash -Path $stage) -cne $restoreHash) { throw 'Missing-target recovery candidate hash mismatch.' }
    $Receipt.Recovery.Add([pscustomobject]@{ RestoredSHA256=$restoreHash; StagePath=$stage; PreservedWindowsAclSHA256=$aclHash; MetadataSource='ActualDisplacedWindowsAcl' })
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
        $restoreAcl = [Security.AccessControl.FileSecurity]::new()
        $restoreAcl.SetSecurityDescriptorBinaryForm($originalAcl.GetSecurityDescriptorBinaryForm(), [Security.AccessControl.AccessControlSections]'Access,Owner,Group')
        Set-LLMConfigOwnedRecoveryAcl -Stream $owned -Acl $restoreAcl
        $Receipt.FailureCurrentSHA256 = Get-LLMConfigCurrentHash -Path $TargetPath
        if ($Receipt.FailureCurrentSHA256 -cne $restoreHash) { return 'LaterWriterPreserved' }
        $actualAcl = [IO.FileSystemAclExtensions]::GetAccessControl($owned)
        $actualAclHash = Get-LLMConfigBytesHash -Bytes ([Text.Encoding]::UTF8.GetBytes($actualAcl.GetSecurityDescriptorSddlForm([Security.AccessControl.AccessControlSections]::All)))
        if ($actualAclHash -cne $aclHash) { throw 'Missing-target recovery Windows ACL mismatch.' }
    } finally { $owned.Dispose() }
    $Receipt.FailureCurrentSHA256 = Get-LLMConfigCurrentHash -Path $TargetPath
    if ($Receipt.FailureCurrentSHA256 -cne $restoreHash) { return 'LaterWriterPreserved' }
    return 'Restored'
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
            try { $receipt.RecoveryState = Restore-LLMConfigMissingTarget -SourcePath $displaced -TargetPath $target -TransactionPath $transaction -Receipt $receipt }
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
    } finally { $lock.Dispose() }
}
