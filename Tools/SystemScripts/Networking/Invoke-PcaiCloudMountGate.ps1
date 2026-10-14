#Requires -Version 7.3
<#
.SYNOPSIS
Run a pinned cloud mount admission and retain bounded structured receipts.
.DESCRIPTION
Default, DryRun, WhatIf and help make no runtime probes or writes. Apply runs
the reviewed launcher in the exact Limited interactive owner session. Existing
receipts are replaced atomically only after the new receipt is flushed. This
runner never restarts, stops, creates or changes provider tasks.
.PARAMETER ValidateOnly
Qualify an existing provider without permitting a provider start.
.EXAMPLE
./Invoke-PcaiCloudMountGate.ps1 -Drive U -DryRun
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidateSet('U','V')][string]$Drive = 'U',
    [string]$TaskName,
    [string]$ProviderUnc,
    [string]$CacheDirectory,
    [string]$ExpectedUserSid,
    [string]$ExpectedTaskXmlSha256,
    [string]$ExpectedWrapperSha256,
    [string]$ExpectedHelperSha256,
    [string]$ReceiptDirectory,
    [ValidateRange(1,600)][int]$TimeoutSeconds = 180,
    [switch]$Apply,
    [switch]$ValidateOnly,
    [switch]$DryRun,
    [Alias('h','?')][switch]$Help,
    [Parameter(ValueFromRemainingArguments)][string[]]$CliArgs
)
$script:PcaiGateWrapper = Join-Path $PSScriptRoot 'Start-PcaiCloudMount.ps1'
$script:PcaiGateHelper = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'Repair-CloudCacheMount.ps1'

function Get-PcaiGateContext {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    [pscustomobject]@{ Sid = $identity.User.Value; SessionId = [Diagnostics.Process]::GetCurrentProcess().SessionId
        Elevated = ([Security.Principal.WindowsPrincipal]::new($identity)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator) }
}
function Assert-PcaiGateLocalPath {
    param([string]$Path, [switch]$Directory)
    $full = [IO.Path]::GetFullPath($Path)
    if (-not [IO.Path]::IsPathFullyQualified($Path) -or $Path.StartsWith('\\') -or $full[1] -ne ':') { throw 'A fully qualified local path is required.' }
    $item = Get-Item -LiteralPath $full -Force -ErrorAction Stop
    if ($Directory -and -not $item.PSIsContainer) { throw 'Receipt root must be an existing directory.' }
    if (-not $Directory -and $item.PSIsContainer) { throw 'Pinned source must be a file.' }
    $current = $full
    while ($current) {
        if (([IO.File]::GetAttributes($current) -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Reparse source/receipt ancestry is refused.' }
        $current = [IO.Path]::GetDirectoryName($current)
    }
    return $full
}
function Open-PcaiGateSource {
    param([string]$Path, [string]$ExpectedHash)
    if ($ExpectedHash -notmatch '^[a-fA-F0-9]{64}$') { throw 'Reviewed source SHA256 is required.' }
    $full = Assert-PcaiGateLocalPath $Path
    $stream = [IO.FileStream]::new($full,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
    try {
        if ($stream.Length -gt 2MB) { throw 'Pinned source exceeds the two MiB limit.' }
        $hash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($stream))
        if ($hash -ne $ExpectedHash) { throw 'Pinned source bytes changed; no provider admission requested.' }
        $stream.Position = 0
        return $stream
    } catch { $stream.Dispose(); throw }
}
function Write-PcaiGateReceipt {
    param($Receipt, [string]$Directory)
    $root = Assert-PcaiGateLocalPath $Directory -Directory
    $latest = Join-Path $root "deployment-gate-$($Receipt.Drive)-latest.json"
    $previous = Join-Path $root "deployment-gate-$($Receipt.Drive)-previous.json"
    $lockPath = Join-Path $root "deployment-gate-$($Receipt.Drive).lock"
    foreach ($path in @($latest,$previous,$lockPath)) {
        if ([IO.File]::Exists($path) -and (([IO.File]::GetAttributes($path) -band [IO.FileAttributes]::ReparsePoint) -ne 0)) { throw 'Receipt leaf reparse point refused.' }
    }
    $lock = [IO.FileStream]::new($lockPath,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    $temporary = Join-Path $root ('.deployment-gate-' + [guid]::NewGuid().ToString('N') + '.pending')
    $ownedTemporary = $false
    try {
        foreach ($path in @($latest,$previous)) {
            if ([IO.File]::Exists($path)) {
                if ([IO.FileInfo]::new($path).Length -gt 128KB) { throw 'Existing receipt exceeds the limit; preserved.' }
                $old = [IO.File]::ReadAllText($path) | ConvertFrom-Json -ErrorAction Stop
                if (@('SchemaVersion','Owner','Drive','TaskName','UserSid' | Where-Object {-not $old.PSObject.Properties[$_]}).Count -gt 0 -or
                    $old.SchemaVersion -ne 1 -or $old.Owner -ne 'PcaiCloudMountGate' -or $old.Drive -ne $Receipt.Drive -or $old.TaskName -ne $Receipt.TaskName -or $old.UserSid -ne $Receipt.UserSid) { throw 'Foreign receipt identity refused; preserved.' }
            }
        }
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($Receipt | ConvertTo-Json -Depth 20 -Compress))
        if ($bytes.Length -gt 128KB) { throw 'New receipt exceeds the limit; existing receipts preserved.' }
        $file = [IO.FileStream]::new($temporary,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        $ownedTemporary = $true
        try { $file.Write($bytes,0,$bytes.Length); $file.Flush($true) } finally { $file.Dispose() }
        if ([IO.File]::Exists($latest)) { [IO.File]::Replace($temporary,$latest,$previous) }
        else { [IO.File]::Move($temporary,$latest,$false) }
        $ownedTemporary = $false
    } finally {
        if ($ownedTemporary -and [IO.File]::Exists($temporary)) { [IO.File]::Delete($temporary) }
        $lock.Dispose()
    }
}
function Invoke-PcaiGateRunner {
    [CmdletBinding(SupportsShouldProcess)]
    param([string]$Drive='U',[string]$TaskName,[string]$ProviderUnc,[string]$CacheDirectory,
        [string]$ExpectedUserSid,[string]$ExpectedTaskXmlSha256,[string]$ExpectedWrapperSha256,
        [string]$ExpectedHelperSha256,[string]$ReceiptDirectory,[int]$TimeoutSeconds=180,
        [switch]$Apply,[switch]$ValidateOnly,[switch]$DryRun,[switch]$Help,[string[]]$CliArgs)
    if ($Help -or ($CliArgs -contains '--help')) { return [pscustomobject]@{State='Help';Usage='Use -Apply with reviewed source/task hashes, owner SID and receipt directory; -ValidateOnly prohibits provider starts.';SideEffects=$false} }
    if ($CliArgs -and @($CliArgs | Where-Object {$_ -ne '--help'}).Count -gt 0) { throw 'Unknown arguments refused.' }
    if (-not $Apply -or $DryRun -or $WhatIfPreference -or -not $PSCmdlet.ShouldProcess("$Drive / $TaskName",'Run pinned cloud admission and save receipt')) { return [pscustomobject]@{State='Plan';Drive=$Drive;TaskName=$TaskName;SideEffects=$false} }
    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'
    if ($Drive -notin @('U','V') -or $TaskName -notin @('Rclone-Dropbox-Gmail-U','Rclone-VIGIL-Team') -or
        ($Drive -eq 'U' -and $TaskName -ne 'Rclone-Dropbox-Gmail-U') -or ($Drive -eq 'V' -and $TaskName -ne 'Rclone-VIGIL-Team') -or
        $ExpectedTaskXmlSha256 -notmatch '^[a-fA-F0-9]{64}$' -or $TimeoutSeconds -lt 1 -or $TimeoutSeconds -gt 600) { throw 'Reviewed U/V provider binding is required.' }
    $context = Get-PcaiGateContext
    if ($context.Elevated -or $context.SessionId -eq 0 -or $context.Sid -ne $ExpectedUserSid) { throw 'Exact Limited interactive owner session required.' }
    $wrapperLease = $null; $helperLease = $null
    $receipt = [pscustomobject]@{ SchemaVersion=1;Owner='PcaiCloudMountGate';Drive=$Drive;TaskName=$TaskName
        UserSid=$ExpectedUserSid;SessionId=$context.SessionId;ObservedUtc=[DateTime]::UtcNow.ToString('o')
        WrapperSha256=$ExpectedWrapperSha256;HelperSha256=$ExpectedHelperSha256;TaskXmlSha256=$ExpectedTaskXmlSha256
        ValidateOnly=[bool]$ValidateOnly;State='Failed';Result=$null;Error=$null }
    try {
        $null = Assert-PcaiGateLocalPath $ReceiptDirectory -Directory
        $wrapperLease = Open-PcaiGateSource $script:PcaiGateWrapper $ExpectedWrapperSha256
        $helperLease = Open-PcaiGateSource $script:PcaiGateHelper $ExpectedHelperSha256
        $invoke = @{ TaskName=$TaskName;Drive=$Drive;ProviderUnc=$ProviderUnc;CacheDirectory=$CacheDirectory
            ExpectedUserSid=$ExpectedUserSid;ExpectedTaskXmlSha256=$ExpectedTaskXmlSha256;TimeoutSeconds=$TimeoutSeconds;Apply=$true;ValidateOnly=[bool]$ValidateOnly }
        $result = @(& $script:PcaiGateWrapper @invoke)
        if ($result.Count -ne 1 -or $result[0].State -notin @('AlreadyRunning','Started','Hold','Busy')) { throw 'Launcher returned an unqualified outcome.' }
        $receipt.Result = $result[0]; $receipt.State = $result[0].State
    } catch { $receipt.State='Failed';$receipt.Error=$_.Exception.Message }
    finally { if ($helperLease) {$helperLease.Dispose()};if ($wrapperLease) {$wrapperLease.Dispose()} }
    Write-PcaiGateReceipt $receipt $ReceiptDirectory
    return $receipt
}
if ($MyInvocation.InvocationName -ne '.') {
    $result = Invoke-PcaiGateRunner @PSBoundParameters
    $result | ConvertTo-Json -Depth 20
    if ($result.State -in @('Plan','Help','AlreadyRunning','Started')) { exit 0 }
    if ($result.State -eq 'Hold') { exit 2 }
    if ($result.State -eq 'Busy') { exit 4 }
    exit 1
}
