#Requires -Version 7.0
<#
.SYNOPSIS
Repairs three known profile startup anchors with guarded byte-preserving custody.
.DESCRIPTION
Defaults to a read-only plan. Apply requires the observed profile SHA256. Each
anchor must occur exactly once or match the already-repaired form; unknown and
duplicate layouts fail closed. Preserves encoding, BOM and line endings. A private
backup and hash receipt outside Git precede staged parsing and atomic replacement.
Every replacement also preserves the bytes actually displaced by that operation.
Hash checks detect concurrent writers; bounded recovery restores their displaced
bytes or reports retained custody requiring review. This is not an atomic hash
compare-and-swap and never evaluates profiles. BackupRoot must share the profile volume.
.PARAMETER Apply
Publishes the reviewed patch. Requires ExpectedSha256 unless DryRun or WhatIf.
.PARAMETER ExpectedSha256
SHA256 of the existing profile, checked before planning and again before replacing.
.PARAMETER BackupRoot
Private custody directory outside Git on the profile volume. Default resolution
tries LOCALAPPDATA/PC_AI/ProfileRepairs, then ProgramData/PC_AI/ProfileRepairs/<user-SID>.
If neither qualifies, read-only plans require an explicit -BackupRoot before Apply.
Candidate discovery does not create directories or prove write permission.
.PARAMETER DryRun
Plans without filesystem, environment or registry writes.
.PARAMETER Help
Displays usage; -h and --help are accepted.
.NOTES
Minimal sessions retain existing process/user agent-bus tokens, and skip secret
backend/config fallback. Full sessions retain their existing fallback behavior.
Backups use stable rN names and contain private original bytes; never add them to Git.
.EXAMPLE
.\Tools\Repair-PcaiProfileStartup.ps1 -DryRun
.EXAMPLE
.\Tools\Repair-PcaiProfileStartup.ps1 -Apply -ExpectedSha256 <observed-sha256>
#>
[CmdletBinding(SupportsShouldProcess = $true, PositionalBinding = $false)]
param(
    [string]$ProfilePath = (Join-Path $HOME '.config/powershell/Microsoft.PowerShell_profile.ps1'),
    [string]$BackupRoot,
    [ValidatePattern('^[A-Fa-f0-9]{64}$')][string]$ExpectedSha256,
    [switch]$Apply,
    [switch]$DryRun,
    [Alias('h')][switch]$Help,
    [Parameter(ValueFromRemainingArguments)][string[]]$CliArgs
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($Help -or @($CliArgs) -contains '--help') {
    'Usage: Repair-PcaiProfileStartup.ps1 [-ProfilePath path] [-BackupRoot private-path] [-Apply -ExpectedSha256 hash] [-DryRun|-WhatIf] [-h|--help]'
    return
}
if ($CliArgs) { throw "Unknown arguments: $($CliArgs -join ' ')" }
if ($DryRun) { $WhatIfPreference = $true }

function Get-BytesSha256 {
    param([byte[]]$Bytes)
    return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($Bytes))
}
function Assert-RepairPath {
    param([string]$Path, [switch]$OutsideGit)
    $cursor = $Path
    if ($Path -eq [IO.Path]::GetPathRoot($Path)) { throw "Refusing filesystem root: $Path" }
    while ($cursor) {
        $name = [IO.Path]::GetFileName($cursor).TrimEnd(' ', '.')
        if ($name -match '^(?i:\$null|AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.|$)') { throw 'Unsafe Windows path in repair request.' }
        if ((Test-Path -LiteralPath $cursor) -and ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Linked repair paths require custody review: $cursor"
        }
        if ($OutsideGit -and (Test-Path -LiteralPath (Join-Path $cursor '.git'))) { throw 'Private backup path must remain outside Git.' }
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ($parent -eq $cursor) { break }
        $cursor = $parent
    }
}
function Resolve-PrivateProfileBackupRoot {
    param([string]$TargetPath, [string]$RequestedRoot)
    $volume = [IO.Path]::GetPathRoot($TargetPath)
    if ($RequestedRoot) {
        $resolved = [IO.Path]::GetFullPath($RequestedRoot)
        if (-not $volume.Equals([IO.Path]::GetPathRoot($resolved), [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Private backup custody must share the profile volume for atomic displacement capture. Supply -BackupRoot with a path on that volume outside Git.'
        }
        Assert-RepairPath -Path $resolved -OutsideGit
        return [pscustomobject]@{ Path = $resolved; Required = $false; Issue = $null }
    }
    $candidates = [Collections.Generic.List[string]]::new()
    if ($env:LOCALAPPDATA) { $candidates.Add((Join-Path $env:LOCALAPPDATA 'PC_AI/ProfileRepairs')) }
    if ($IsWindows -and $env:ProgramData) {
        $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
        $candidates.Add((Join-Path $env:ProgramData "PC_AI/ProfileRepairs/$sid"))
    }
    foreach ($candidate in $candidates) {
        try {
            $resolved = [IO.Path]::GetFullPath($candidate)
            if (-not $volume.Equals([IO.Path]::GetPathRoot($resolved), [StringComparison]::OrdinalIgnoreCase)) { continue }
            Assert-RepairPath -Path $resolved -OutsideGit
            return [pscustomobject]@{ Path = $resolved; Required = $false; Issue = $null }
        } catch { continue }
    }
    return [pscustomobject]@{ Path = $null; Required = $true; Issue = 'No automatic same-volume custody path outside Git qualifies. Supply -BackupRoot with a writable private directory on the profile volume outside every Git checkout.' }
}
function Set-PrivateRepairAcl {
    param([string]$Path, [switch]$Directory)
    if (-not $IsWindows) { throw 'Private profile repair requires Windows ACL support.' }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent().User
    $acl = if ($Directory) { [Security.AccessControl.DirectorySecurity]::new() } else { [Security.AccessControl.FileSecurity]::new() }
    $acl.SetOwner($identity)
    $acl.SetAccessRuleProtection($true, $false)
    $inheritance = if ($Directory) { [Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit' } else { [Security.AccessControl.InheritanceFlags]::None }
    foreach ($sid in @($identity, [Security.Principal.SecurityIdentifier]::new('S-1-5-18'), [Security.Principal.SecurityIdentifier]::new('S-1-5-32-544'))) {
        $rule = [Security.AccessControl.FileSystemAccessRule]::new($sid, [Security.AccessControl.FileSystemRights]::FullControl, $inheritance, [Security.AccessControl.PropagationFlags]::None, [Security.AccessControl.AccessControlType]::Allow)
        [void]$acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $Path -AclObject $acl
}
function Update-UniqueProfileAnchor {
    param([string]$Text, [string]$Old, [string]$New, [string]$Name)
    # Existing private profiles may contain mixed CRLF/LF from independent edits.
    # Match only the exact known line contents; retain their individual separators.
    $oldPattern = (($Old.Replace("`r`n", "`n").Split("`n") | ForEach-Object { [regex]::Escape($_) }) -join '(?:\r\n|\n)')
    $newPattern = (($New.Replace("`r`n", "`n").Split("`n") | ForEach-Object { [regex]::Escape($_) }) -join '(?:\r\n|\n)')
    $alreadyMatches = [regex]::Matches($Text, $newPattern)
    $already = $alreadyMatches.Count
    $remaining = [regex]::Replace($Text, $newPattern, '')
    $oldMatches = [regex]::Matches($remaining, $oldPattern)
    $unpatched = $oldMatches.Count
    if ($already -eq 1 -and $unpatched -eq 0) { return [pscustomobject]@{ Text = $Text; Changed = $false; Name = $Name } }
    if ($already -ne 0 -or $unpatched -ne 1) { throw "Unknown or duplicate profile anchor: $Name" }
    $match = $oldMatches[0]
    $following = $Text.Substring($match.Index + $match.Length)
    $separator = if ($following.StartsWith("`r`n")) { "`r`n" } else { "`n" }
    $replacement = $New.Replace("`r`n", "`n").Replace("`n", $separator)
    if ($Name -eq 'MinimalTokenFallback') {
        $secretAnchor = '    $secretValue = Get-ProfileSecretFromSecretManagement -Name @('
        $insertion = ('    if ($script:ProfileLevel -eq ''minimal'') {', '        return $false', '    }', '') -join $separator
        $replacement = $match.Value.Replace($secretAnchor, $insertion + $separator + $secretAnchor)
    }
    return [pscustomobject]@{ Text = $Text.Substring(0, $match.Index) + $replacement + $following; Changed = $true; Name = $Name }
}
function Assert-ProfileParses {
    param([string]$Text)
    $tokens = $null; $errors = $null
    [void][Management.Automation.Language.Parser]::ParseInput($Text, [ref]$tokens, [ref]$errors)
    if ($errors.Count -gt 0) { throw 'Profile parse validation failed; contents withheld.' }
}
function Write-PrivateProfileStage {
    param([string]$Path, [byte[]]$Bytes)
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        Set-PrivateRepairAcl -Path $Path
        $stream.Write($Bytes, 0, $Bytes.Length)
        $stream.Flush($true)
    } finally { $stream.Dispose() }
}
function Restore-DisplacedProfile {
    param([string]$SourcePath, [string]$ExpectedCurrentHash, [string]$TargetPath, [string]$StagePath, [string]$TransactionPath, [Collections.IDictionary]$Receipt)
    # A second non-cooperating writer can race recovery too. Each swap captures
    # exactly what it displaced, then restores that newer version on the next pass.
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        if ((Get-FileHash -LiteralPath $TargetPath).Hash -ne $ExpectedCurrentHash) { return 'InterveningCurrentPreserved' }
        $restoreBytes = [IO.File]::ReadAllBytes($SourcePath)
        $restoreHash = Get-BytesSha256 -Bytes $restoreBytes
        Write-PrivateProfileStage -Path $StagePath -Bytes $restoreBytes
        $displacedPath = Join-Path $TransactionPath "recovery-displaced-r$attempt.bin"
        if (Test-Path -LiteralPath $displacedPath) { throw 'Recovery custody path already exists; refusing overwrite.' }
        [IO.File]::Replace($StagePath, $TargetPath, $displacedPath)
        Set-PrivateRepairAcl -Path $displacedPath
        $displacedHash = (Get-FileHash -LiteralPath $displacedPath).Hash
        $Receipt.RecoveryAttempts.Add([pscustomobject]@{ SourcePath = $SourcePath; RestoredSha256 = $restoreHash; DisplacedPath = $displacedPath; DisplacedSha256 = $displacedHash; ExpectedDisplacedSha256 = $ExpectedCurrentHash })
        if ($displacedHash -eq $ExpectedCurrentHash) {
            if ((Get-FileHash -LiteralPath $TargetPath).Hash -ne $restoreHash) { return 'InterveningCurrentPreserved' }
            return 'Restored'
        }
        $SourcePath = $displacedPath
        $ExpectedCurrentHash = $restoreHash
    }
    return 'RequiresReview'
}

$ProfilePath = [IO.Path]::GetFullPath($ProfilePath)
Assert-RepairPath -Path $ProfilePath
if ($PSBoundParameters.ContainsKey('BackupRoot') -and [string]::IsNullOrWhiteSpace($BackupRoot)) { throw 'Explicit BackupRoot cannot be empty.' }
$custody = Resolve-PrivateProfileBackupRoot -TargetPath $ProfilePath -RequestedRoot $BackupRoot
$BackupRoot = $custody.Path
if (-not (Test-Path -LiteralPath $ProfilePath -PathType Leaf)) { throw 'Canonical profile file is missing.' }
if ($Apply -and -not $WhatIfPreference -and -not $ExpectedSha256) { throw 'Apply requires ExpectedSha256 from the reviewed profile.' }
$originalBytes = [IO.File]::ReadAllBytes($ProfilePath)
$beforeHash = Get-BytesSha256 -Bytes $originalBytes
if ($ExpectedSha256 -and $ExpectedSha256 -ne $beforeHash) { throw 'Expected profile SHA256 does not match current bytes.' }
$offset = 0
$encoding = [Text.UTF8Encoding]::new($false, $true)
if ($originalBytes.Length -ge 3 -and $originalBytes[0] -eq 239 -and $originalBytes[1] -eq 187 -and $originalBytes[2] -eq 191) { $offset = 3 }
elseif ($originalBytes.Length -ge 2 -and $originalBytes[0] -eq 255 -and $originalBytes[1] -eq 254) {
    if ($originalBytes.Length -ge 4 -and $originalBytes[2] -eq 0 -and $originalBytes[3] -eq 0) { throw 'Unsupported UTF32 profile encoding.' }
    $offset = 2; $encoding = [Text.UnicodeEncoding]::new($false, $false, $true)
} elseif ($originalBytes.Length -ge 2 -and $originalBytes[0] -eq 254 -and $originalBytes[1] -eq 255) {
    $offset = 2; $encoding = [Text.UnicodeEncoding]::new($true, $false, $true)
}
$text = $encoding.GetString($originalBytes, $offset, $originalBytes.Length - $offset)
Assert-ProfileParses -Text $text
foreach ($dependency in @('Get-PreferredModulesRoot', 'Set-ProfileEnvFromUser')) {
    if ([regex]::Matches($text, "(?m)^function (?:script:)?$([regex]::Escape($dependency))\s*\{").Count -ne 1) {
        throw "Unknown or duplicate profile dependency: $dependency"
    }
}
$newline = if ($text.Contains("`r`n")) { "`r`n" } else { "`n" }
if ($text.Replace("`r`n", '').Contains("`r")) { throw 'Unsupported profile line endings.' }
$anchors = @(
    @{
        Name = 'ModulePathSkip'
        Old = 'Optimize-PSModulePath -AllowNetworkPaths:$script:AllowNetworkModulePath'
        New = ('if ($env:PS_SKIP_PSMODULEPATH_OPTIMIZE -ne ''1'') {', '    Optimize-PSModulePath -AllowNetworkPaths:$script:AllowNetworkModulePath', '}') -join $newline
    },
    @{
        Name = 'DeveloperModuleRoot'
        Old = ('$script:UserPowerShellDir = Split-Path -Parent $PROFILE.CurrentUserCurrentHost', '$script:UserPSModulesRoot = Join-Path $script:UserPowerShellDir ''Modules''') -join $newline
        New = '$script:UserPSModulesRoot = Get-PreferredModulesRoot'
    },
    @{
        Name = 'MinimalTokenFallback'
        Old = ('function script:Initialize-AgentBusAuthToken {', '    if (Set-ProfileEnvFromUser -Name ''AGENT_BUS_AUTH_TOKEN'') {', '        return $true', '    }', '', '    $secretValue = Get-ProfileSecretFromSecretManagement -Name @(') -join $newline
        New = ('function script:Initialize-AgentBusAuthToken {', '    if (Set-ProfileEnvFromUser -Name ''AGENT_BUS_AUTH_TOKEN'') {', '        return $true', '    }', '', '    if ($script:ProfileLevel -eq ''minimal'') {', '        return $false', '    }', '', '    $secretValue = Get-ProfileSecretFromSecretManagement -Name @(') -join $newline
    }
)
$changes = [Collections.Generic.List[string]]::new()
foreach ($anchor in $anchors) {
    $update = Update-UniqueProfileAnchor -Text $text -Old $anchor.Old -New $anchor.New -Name $anchor.Name
    $text = $update.Text
    if ($update.Changed) { $changes.Add($update.Name) }
}
Assert-ProfileParses -Text $text
$body = $encoding.GetBytes($text)
$newBytes = [byte[]]::new($offset + $body.Length)
if ($offset) { [Array]::Copy($originalBytes, 0, $newBytes, 0, $offset) }
[Array]::Copy($body, 0, $newBytes, $offset, $body.Length)
$afterHash = Get-BytesSha256 -Bytes $newBytes
$result = [ordered]@{ ProfilePath = $ProfilePath; BackupRoot = $BackupRoot; BackupRootRequired = $custody.Required; CustodyIssue = $custody.Issue; BeforeSha256 = $beforeHash; AfterSha256 = $afterHash; Changes = @($changes); State = 'Planned'; Receipt = $null }
if ($changes.Count -eq 0) { $result.State = 'AlreadyApplied'; [pscustomobject]$result; return }
if (-not $Apply -or -not $PSCmdlet.ShouldProcess($ProfilePath, 'Preserve private original bytes and apply three guarded startup repairs')) {
    [pscustomobject]$result; return
}
if ($custody.Required) { throw $custody.Issue }

$pathHash = Get-BytesSha256 -Bytes ([Text.Encoding]::UTF8.GetBytes($ProfilePath.ToUpperInvariant()))
$custodyRoot = Join-Path $BackupRoot ("profile-" + $pathHash.Substring(0, 16).ToLowerInvariant())
Assert-RepairPath -Path $custodyRoot -OutsideGit
try { [void](New-Item -ItemType Directory -Path $custodyRoot -Force) }
catch { throw 'Cannot create private backup custody. Supply -BackupRoot with a writable same-volume private directory outside Git; profile bytes have not been replaced.' }
Set-PrivateRepairAcl -Path $custodyRoot -Directory
$lock = [IO.File]::Open((Join-Path $custodyRoot 'repair.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
$receipt = $null
$receiptPath = $null
$published = $false
try {
    $revision = 1
    while (Test-Path -LiteralPath (Join-Path $custodyRoot "r$revision")) { $revision++ }
    $transaction = Join-Path $custodyRoot "r$revision"
    [void](New-Item -ItemType Directory -Path $transaction)
    Set-PrivateRepairAcl -Path $transaction -Directory
    $backupPath = Join-Path $transaction 'original.bin'
    $displacedPath = Join-Path $transaction 'displaced-original.bin'
    $receiptPath = Join-Path $transaction 'receipt.json'
    # A whole-home Git checkout must never see a transient copy of private profile contents.
    # File.Replace requires the same volume, but does not require the same directory.
    $stagePath = Join-Path $transaction 'staged-profile.ps1'
    Assert-RepairPath -Path $stagePath -OutsideGit
    if (Test-Path -LiteralPath $stagePath) { throw 'Profile staging path already exists; preserve and review it.' }
    $receipt = [ordered]@{ SchemaVersion = 2; ProfilePath = $ProfilePath; BackupPath = $backupPath; DisplacedPath = $displacedPath; DisplacedSha256 = $null; StagePath = $stagePath; BeforeSha256 = $beforeHash; AfterSha256 = $afterHash; Changes = @($changes); ObservedUtc = [DateTime]::UtcNow.ToString('o'); State = 'Preserving'; RecoveryState = $null; RecoveryAttempts = [Collections.Generic.List[object]]::new(); Error = $null }
    $receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
    [IO.File]::WriteAllBytes($backupPath, $originalBytes)
    if ((Get-FileHash -LiteralPath $backupPath).Hash -ne $beforeHash) { throw 'Private original backup failed SHA256 validation.' }
    Write-PrivateProfileStage -Path $stagePath -Bytes $newBytes
    $tokens = $null; $errors = $null
    [void][Management.Automation.Language.Parser]::ParseFile($stagePath, [ref]$tokens, [ref]$errors)
    if ($errors.Count -gt 0) { throw 'Staged profile parse validation failed; contents withheld.' }
    if ((Get-FileHash -LiteralPath $stagePath).Hash -ne $afterHash) { throw 'Staged profile hash mismatch.' }
    if ((Get-FileHash -LiteralPath $ProfilePath).Hash -ne $beforeHash) { throw 'Profile changed after review; refusing replacement.' }
    if (Test-Path -LiteralPath $displacedPath) { throw 'Publication custody path already exists; refusing overwrite.' }
    [IO.File]::Replace($stagePath, $ProfilePath, $displacedPath)
    $published = $true
    Set-PrivateRepairAcl -Path $displacedPath
    $receipt.DisplacedSha256 = (Get-FileHash -LiteralPath $displacedPath).Hash
    if ($receipt.DisplacedSha256 -ne $beforeHash) { throw 'Profile changed at replacement boundary; actual displaced writer bytes retained for recovery.' }
    if ((Get-FileHash -LiteralPath $ProfilePath).Hash -ne $afterHash) { throw 'Published profile hash mismatch.' }
    $receipt.State = 'Applied'
    $receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
    $result.State = 'Applied'
    $result.Receipt = $receiptPath
} catch {
    $failure = $_
    if ($published) {
        try {
            $receipt.RecoveryState = Restore-DisplacedProfile -SourcePath $displacedPath -ExpectedCurrentHash $afterHash -TargetPath $ProfilePath -StagePath $stagePath -TransactionPath $transaction -Receipt $receipt
        } catch {
            $receipt.RecoveryState = 'RequiresReview'
            $receipt.Error = "Publication: $($failure.Exception.Message); recovery: $($_.Exception.Message)"
        }
    }
    if ($receipt) {
        $receipt.State = if (-not $published) { 'FailedBeforeReplace' }
            elseif ($receipt.RecoveryState -eq 'Restored') { 'RolledBack' }
            elseif ($receipt.RecoveryState -eq 'InterveningCurrentPreserved') { 'ConcurrentWriterPreserved' }
            else { 'RecoveryRequiresReview' }
        if (-not $receipt.Error) { $receipt.Error = $failure.Exception.Message }
        $receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
    }
    if ($published -and $receipt.RecoveryState -eq 'RequiresReview') { throw "Profile recovery requires review; every displaced version is retained in $transaction. $($receipt.Error)" }
    throw $failure
} finally { $lock.Dispose() }
[pscustomobject]$result
