#Requires -Version 7.0
<#
.SYNOPSIS
Archives the synchronized Bitwarden vault inside protected staging storage.
.DESCRIPTION
Uses the existing machine Bitwarden backend. Plain JSON remains the compatibility
default; encrypted export requires a separately validated recovery procedure.
DryRun and WhatIf return before authentication, file writes, or pruning.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$Dir = (Join-Path $env:USERPROFILE '.bwdata'),
    [ValidateRange(1, 3650)][int]$KeepDays = 30,
    [ValidateSet('json', 'encrypted_json')][string]$Format = 'json',
    [ValidateRange(1, 600)][int]$TimeoutSeconds = 120,
    [switch]$Quiet,
    [Alias('-DryRun')][switch]$DryRun,
    [Alias('h', '-help')][switch]$Help
)

function Set-BwArchiveAcl {
    [CmdletBinding(SupportsShouldProcess)]
    param([Parameter(Mandatory)][string]$LiteralPath)
    $item = Get-Item -LiteralPath $LiteralPath -ErrorAction Stop
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw 'Archive storage must not be a reparse point.'
    }
    $sids = @(
        [Security.Principal.WindowsIdentity]::GetCurrent().User,
        [Security.Principal.SecurityIdentifier]::new('S-1-5-18'),
        [Security.Principal.SecurityIdentifier]::new('S-1-5-32-544')
    )
    $acl = if ($item.PSIsContainer) { [Security.AccessControl.DirectorySecurity]::new() }
    else { [Security.AccessControl.FileSecurity]::new() }
    $acl.SetAccessRuleProtection($true, $false)
    $inheritance = if ($item.PSIsContainer) { [Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit' }
    else { [Security.AccessControl.InheritanceFlags]::None }
    foreach ($sid in $sids) {
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new(
                $sid, [Security.AccessControl.FileSystemRights]::FullControl,
                $inheritance, [Security.AccessControl.PropagationFlags]::None,
                [Security.AccessControl.AccessControlType]::Allow
            ))
    }
    if (-not $PSCmdlet.ShouldProcess($item.FullName, 'Restrict archive ACL to current user, SYSTEM, and Administrators')) { return }
    Set-Acl -LiteralPath $item.FullName -AclObject $acl -ErrorAction Stop
    $actual = Get-Acl -LiteralPath $item.FullName -ErrorAction Stop
    $rules = @($actual.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]))
    if (-not $actual.AreAccessRulesProtected -or $rules.Count -ne $sids.Count) { throw 'Archive ACL verification failed.' }
    foreach ($rule in $rules) {
        if ($rule.IdentityReference.Value -notin $sids.Value -or $rule.IsInherited -or
            $rule.AccessControlType -ne 'Allow' -or
            $rule.FileSystemRights -ne [Security.AccessControl.FileSystemRights]::FullControl) {
            throw 'Archive ACL verification failed.'
        }
    }
}

function Invoke-BwArchiveCommand {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string[]]$Arguments, [Parameter(Mandatory)][int]$TimeoutSeconds)
    $response = Invoke-BitwardenCli -Arguments $Arguments -TimeoutSeconds $TimeoutSeconds
    if (-not $response.Success) {
        # Native stdout/stderr may contain secrets; report metadata only.
        throw "Bitwarden $($Arguments[0]) failed (exit=$($response.ExitCode), timeout=$($response.TimedOut))."
    }
    return $response.StdOut
}

function Get-BwArchiveStatus {
    param([int]$TimeoutSeconds)
    $output = Invoke-BwArchiveCommand -Arguments @('status') -TimeoutSeconds $TimeoutSeconds
    try {
        $status = ($output | ConvertFrom-Json -ErrorAction Stop).status
        if ($status -notin @('unlocked', 'locked', 'unauthenticated')) { throw 'Invalid status.' }
        return $status
    }
    catch { throw 'Bitwarden returned an invalid status response.' }
}

function Invoke-BwArchiveRefresh {
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([int])]
    param(
        [string]$Dir = (Join-Path $env:USERPROFILE '.bwdata'),
        [ValidateRange(1, 3650)][int]$KeepDays = 30,
        [ValidateSet('json', 'encrypted_json')][string]$Format = 'json',
        [ValidateRange(1, 600)][int]$TimeoutSeconds = 120,
        [switch]$Quiet,
        [switch]$DryRun
    )
    $archiveRoot = [IO.Path]::GetFullPath($Dir)
    if ($DryRun -or -not $PSCmdlet.ShouldProcess($archiveRoot, 'Synchronize and archive Bitwarden; prune expired archives')) { return 0 }
    $null = New-Item -ItemType Directory -Path $archiveRoot -Force -ErrorAction Stop
    Set-BwArchiveAcl -LiteralPath $archiveRoot
    $backend = Join-Path $env:USERPROFILE '.machine\SecretBackendUtilities.ps1'
    if (-not (Test-Path -LiteralPath $backend -PathType Leaf)) { throw 'The machine Bitwarden backend is unavailable.' }
    if (-not (Get-Command Invoke-BitwardenCli -ErrorAction SilentlyContinue) -or
        -not (Get-Command Initialize-BitwardenSessionFromBackends -ErrorAction SilentlyContinue)) {
        . $backend
    }
    $status = Get-BwArchiveStatus -TimeoutSeconds $TimeoutSeconds
    if ($status -ne 'unlocked') {
        $session = Initialize-BitwardenSessionFromBackends -Quiet
        if (-not $session.Success) { throw 'The machine Bitwarden backend could not establish an unlocked session.' }
        $status = Get-BwArchiveStatus -TimeoutSeconds $TimeoutSeconds
        if ($status -ne 'unlocked') { throw 'The Bitwarden session is not unlocked.' }
    }
    $staging = Join-Path $archiveRoot ('.bitwarden_archive_' + [guid]::NewGuid().ToString('N'))
    $null = New-Item -ItemType Directory -Path $staging -ErrorAction Stop
    $extension = if ($Format -eq 'encrypted_json') { 'enc.json' } else { 'json' }
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')
    $name = "bitwarden_archive_$stamp.$extension"
    $stagedArchive = Join-Path $staging $name
    $stagedLatest = Join-Path $staging "bitwarden_archive_latest.$extension"
    $target = Join-Path $archiveRoot $name
    $latest = Join-Path $archiveRoot "bitwarden_archive_latest.$extension"
    try {
        # Protect the empty directory before any vault data is written.
        Set-BwArchiveAcl -LiteralPath $staging
        $null = Invoke-BwArchiveCommand -Arguments @('sync') -TimeoutSeconds $TimeoutSeconds
        $null = Invoke-BwArchiveCommand -Arguments @('export', '--format', $Format, '--output', $stagedArchive) -TimeoutSeconds $TimeoutSeconds
        if (-not (Test-Path -LiteralPath $stagedArchive -PathType Leaf) -or (Get-Item -LiteralPath $stagedArchive).Length -eq 0) {
            throw 'Bitwarden export did not produce a nonempty archive.'
        }
        Set-BwArchiveAcl -LiteralPath $stagedArchive
        Copy-Item -LiteralPath $stagedArchive -Destination $stagedLatest -ErrorAction Stop
        Set-BwArchiveAcl -LiteralPath $stagedLatest
        # Same-volume moves preserve explicit file ACLs established in staging.
        Move-Item -LiteralPath $stagedArchive -Destination $target -ErrorAction Stop
        Move-Item -LiteralPath $stagedLatest -Destination $latest -Force -ErrorAction Stop
        $cutoff = (Get-Date).AddDays(-$KeepDays)
        foreach ($file in Get-ChildItem -LiteralPath $archiveRoot -File -ErrorAction Stop) {
            if ($file.Name -match '^bitwarden_archive_\d{8}T\d{6,13}Z\.(?:enc\.)?json$' -and
                $file.LastWriteTime -lt $cutoff -and -not ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                Remove-Item -LiteralPath $file.FullName -Force -ErrorAction Stop
            }
        }
        if (-not $Quiet) { Write-Information "Bitwarden archive refreshed: $target" -InformationAction Continue }
        return 0
    }
    finally {
        # Delete only our two known files and empty directory; never recurse.
        foreach ($path in @($stagedArchive, $stagedLatest)) {
            if (Test-Path -LiteralPath $path -PathType Leaf) { Remove-Item -LiteralPath $path -Force -ErrorAction Stop }
        }
        if (Test-Path -LiteralPath $staging -PathType Container) { Remove-Item -LiteralPath $staging -Force -ErrorAction Stop }
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    if ($Help -or $Dir -eq '--help' -or $args -contains '--help') {
        Write-Output 'Update-BwArchive.ps1 [-Dir PATH] [-KeepDays 30] [-Format json|encrypted_json] [-TimeoutSeconds 120] [-Quiet] [-DryRun] [-WhatIf]'
        exit 0
    }
    try {
        $options = @{} + $PSBoundParameters
        $options.Remove('Help')
        if ($Dir -eq '--DryRun') {
            $options.Remove('Dir')
            $options['DryRun'] = $true
        }
        exit (Invoke-BwArchiveRefresh @options -ErrorAction Stop)
    }
    catch {
        $message = 'Bitwarden archive failed: ' + $_.Exception.Message
        try {
            $log = Join-Path $env:USERPROFILE '.bwdata\bitwarden_archive.log'
            Add-Content -LiteralPath $log -Value ([DateTime]::UtcNow.ToString('o') + ' [Error] ' + $message) -ErrorAction Stop
        }
        catch { Write-Error 'The Bitwarden archive error log could not be written.' }
        Write-Error $message
        exit 1
    }
}
