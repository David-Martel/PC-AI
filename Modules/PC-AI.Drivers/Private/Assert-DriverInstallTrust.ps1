#Requires -Version 7.0

function Get-DriverInstallTrustRoot {
    $programFiles = [Environment]::GetFolderPath('ProgramFiles')
    if (-not [IO.Path]::IsPathFullyQualified($programFiles)) {
        throw 'The Windows Program Files path is unavailable.'
    }
    return Join-Path $programFiles 'PC-AI.DriverTrust'
}

function Assert-DriverInstallTrustPath {
    param([string]$Path, [switch]$Directory)

    if ($Path -notmatch '^(?i:[a-z]):\\' -or $Path.Substring(3).Contains(':')) {
        throw 'Driver installation requires an ordinary local Windows path.'
    }

    $current = [IO.Path]::GetFullPath($Path)
    $root = [IO.Path]::GetPathRoot($current)
    if ([IO.DriveInfo]::new($root).DriveType -ne [IO.DriveType]::Fixed) {
        throw "Driver installation trust requires a fixed drive: $root"
    }

    $admins = 'S-1-5-32-544'
    $system = 'S-1-5-18'
    $installer = 'S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464'
    $isLeaf = $true
    while ($current) {
        $item = Get-Item -LiteralPath $current -Force -ErrorAction Stop
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "A reparse point cannot establish driver installation trust: $current"
        }
        if ($item.PSIsContainer -ne ((-not $isLeaf) -or $Directory)) {
            throw "Unexpected driver installation trust path type: $current"
        }

        $acl = Get-Acl -LiteralPath $current -ErrorAction Stop
        $descriptor = [Security.AccessControl.RawSecurityDescriptor]::new($acl.GetSecurityDescriptorBinaryForm(), 0)
        $owners = if ($isLeaf) { @($admins, $system) } else { @($admins, $system, $installer) }
        if ($descriptor.Owner.Value -notin $owners -or $null -eq $descriptor.DiscretionaryAcl) {
            throw "Driver installation trust path requires a protected owner and DACL: $current"
        }

        # Deny any allow ACE that could replace an existing protected name.
        # Ancestors may permit new, unrelated names, for example at a drive root.
        $unsafe = [int]([Security.AccessControl.FileSystemRights]'Delete,DeleteSubdirectoriesAndFiles,ChangePermissions,TakeOwnership,WriteAttributes,WriteExtendedAttributes')
        if ($isLeaf) {
            $unsafe = $unsafe -bor [int]([Security.AccessControl.FileSystemRights]'WriteData,AppendData')
        }
        # GENERIC_ALL and GENERIC_WRITE can expand to the rights above.
        $unsafe = $unsafe -bor 0x50000000
        foreach ($rule in $descriptor.DiscretionaryAcl) {
            if ($rule.AceFlags -band [Security.AccessControl.AceFlags]::InheritOnly) { continue }
            if ($rule -isnot [Security.AccessControl.QualifiedAce]) {
                throw "Cannot establish driver installation trust for an unrecognized ACE: $current"
            }
            if ($rule.AceQualifier -eq [Security.AccessControl.AceQualifier]::AccessAllowed -and
                $rule.SecurityIdentifier.Value -notin @($admins, $system, $installer) -and
                ($rule.AccessMask -band $unsafe)) {
                throw "An untrusted principal can modify or replace the driver installation path: $current"
            }
        }

        $current = if ($current -eq $root) { $null } else { [IO.Path]::GetDirectoryName($current) }
        $isLeaf = $false
    }
}

function Assert-DriverInstallerSignature {
    param([string]$Path, [string]$PublisherCertificateSha256)

    $signature = Get-AuthenticodeSignature -LiteralPath $Path -ErrorAction Stop
    if ($signature.Status -ne 'Valid' -or -not $signature.SignerCertificate) {
        throw "A valid Authenticode signature is required for '$Path'."
    }

    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $certificateSha256 = [BitConverter]::ToString($algorithm.ComputeHash($signature.SignerCertificate.RawData)).Replace('-', '')
    }
    finally {
        $algorithm.Dispose()
    }
    if ($certificateSha256 -ne $PublisherCertificateSha256) {
        throw "The Authenticode signer for '$Path' does not match the pinned publisher certificate."
    }
}

function New-DriverInstallStage {
    param([string]$TrustRoot)

    $stage = Join-Path $TrustRoot ('install-' + [guid]::NewGuid().ToString('N'))
    $acl = [Security.AccessControl.DirectorySecurity]::new()
    $acl.SetAccessRuleProtection($true, $false)
    $administrators = [Security.Principal.SecurityIdentifier]::new('S-1-5-32-544')
    $system = [Security.Principal.SecurityIdentifier]::new('S-1-5-18')
    $acl.SetOwner($administrators)
    foreach ($identity in @($administrators, $system)) {
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new(
            $identity, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow'
        ))
    }
    $null = [IO.FileSystemAclExtensions]::CreateDirectory($acl, $stage)
    return $stage
}

function Assert-DriverArchivePayload {
    param([string]$Path)

    $items = @(Get-ChildItem -LiteralPath $Path -Recurse -Force -ErrorAction Stop)
    foreach ($item in $items) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Driver archives cannot contain reparse points: $($item.FullName)"
        }
        Assert-DriverInstallTrustPath -Path $item.FullName -Directory:$item.PSIsContainer
    }
    $executables = @($items | Where-Object { -not $_.PSIsContainer -and $_.Extension -eq '.exe' })
    if ($executables.Count -ne 1) {
        throw 'Driver archives must contain exactly one executable installer.'
    }
    return $executables[0].FullName
}
