#Requires -Version 7.3
<#
.SYNOPSIS
    Archive held Windows file bytes and optionally delete their verified identity.
.DESCRIPTION
    Preview performs no writes. Execution rejects active writers and cloud/reparse
    sources, hashes held bytes before upload, verifies remote MD5, size and count,
    then flushes the verified receipt before identity-bound deletion. Remote files
    are never overwritten. SHA256 revision keys retain changed source versions.
.PARAMETER SourcePath
    Explicit local source files; directories and reparse sources are rejected.
.PARAMETER Category
    Archive category below the batch folder.
.PARAMETER BatchId
    Stable archive batch name; observation times remain in receipt metadata.
.PARAMETER RemoteRoot
    Root inside the selected rclone remote.
.PARAMETER RcloneRemote
    Remote name including its trailing colon.
.PARAMETER ManifestPath
    JSONL receipts; verified records are flushed to disk before deletion.
.PARAMETER DriveChunkSize
    Google Drive upload chunk size forwarded to rclone.
.PARAMETER Execute
    Opt in to uploads and verification.
.PARAMETER RecordDryRun
    Explicitly append preview receipts; ordinary preview performs no writes.
.PARAMETER DeleteAfterVerify
    Delete only the captured file identity after durable verification.
.PARAMETER DryRun
    Suppress all writes including RecordDryRun, even with Execute; --DryRun is also supported.
.PARAMETER Help
    Print help; -h and --help are also supported.
.PARAMETER CliArgs
    Compatibility spellings --help and --DryRun.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(ValueFromPipeline, ValueFromPipelineByPropertyName)]
    [Alias('FullName')][string[]]$SourcePath,
    [string]$Category = '04-reports',
    [string]$BatchId = 'cleanup-archive',
    [string]$RemoteRoot = 'Archives/Workstation Cleanup',
    [string]$RcloneRemote = 'gdrive-personal:',
    [string]$ManifestPath = (Join-Path (Get-Location).Path 'Reports/cleanup-archive-manifest.jsonl'),
    [string]$DriveChunkSize = '512M',
    [switch]$Execute,
    [switch]$RecordDryRun,
    [switch]$DeleteAfterVerify,
    [switch]$DryRun,
    [Alias('h', '?')][switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArgs
)
begin {
    $ErrorActionPreference = 'Stop'
    $CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($CliArgs -contains '--help') { $Help = $true }
    if ($CliArgs -contains '--DryRun') { $DryRun = $true }
    $unknown = @($CliArgs | Where-Object { $_ -notin @('--help', '--DryRun') })
    if ($unknown.Count) { throw "Unknown archive argument: $($unknown -join ', ')" }
    if ($Help) {
        [regex]::Match((Get-Content -LiteralPath $PSCommandPath -Raw), '(?s)<#\s*(.*?)\s*#>').Groups[1].Value.Trim()
        $script:HelpShown = $true
        return
    }
    $script:IsExecuting = [bool]$Execute -and -not $DryRun
    if ($script:IsExecuting -and -not (Get-Command rclone -ErrorAction SilentlyContinue)) { throw 'rclone is required for execution.' }
    if (-not $RcloneRemote.EndsWith(':')) { throw 'RcloneRemote must include its trailing colon.' }
    $script:ManifestFile = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ManifestPath)
    $script:ManifestDir = Split-Path $script:ManifestFile -Parent
    $script:SeenSource = @{}
    $script:SeenIdentity = @{}
    $script:Records = [System.Collections.Generic.List[object]]::new()
    function Initialize-ArchiveCustody {
        <# .SYNOPSIS
        Load native custody only for explicitly requested writes.
        #>
        if (-not ('Pcai.Storage.PcaiArchiveSource' -as [type])) {
            Add-Type -Path (Join-Path $PSScriptRoot 'SystemScripts/Storage/PcaiArchiveSource.cs') -ErrorAction Stop
        }
    }
    function Write-ArchiveReceipt {
        <# .SYNOPSIS
        Append a receipt with durable native flushing.
        #>
        param([Parameter(Mandatory)]$Record, $Custody)
        if ($null -ne $Custody) {
            $Custody.AcquireReceipt($script:ManifestFile)
            $Record.receipt_guid_path = $Custody.ReceiptGuidPath
            $Record.receipt_volume_serial = $Custody.ReceiptVolumeSerial
            $Record.receipt_file_id = $Custody.ReceiptFileId
        }
        $json = [pscustomobject]$Record | ConvertTo-Json -Compress -Depth 8
        if ($null -ne $Custody -and $Record.event -eq 'verified') {
            $Custody.PersistVerifiedReceipt($script:ManifestFile, $json, $Record.remote_md5, $Record.remote_size_bytes, $Record.remote_count)
        } elseif ($null -ne $Custody) {
            $Custody.AppendReceipt($script:ManifestFile, $json)
        } else {
            [Pcai.Storage.PcaiArchiveSource]::AppendDurableReceipt($script:ManifestFile, $json)
        }
    }
    function Invoke-ArchiveRclone {
        <# .SYNOPSIS
        Run rclone and fail on unsuccessful external operations.
        #>
        param([Parameter(Mandatory)][string[]]$ArgumentList)
        $output = @(& rclone @ArgumentList)
        if ($LASTEXITCODE -ne 0) { throw "rclone $($ArgumentList[0]) failed with exit $LASTEXITCODE." }
        return $output
    }
    function ConvertTo-ArchiveInteger {
        <# .SYNOPSIS
        Require an exact nonnegative numeric scalar within Int64 limits.
        #>
        param([AllowNull()]$Value)
        if ($null -eq $Value) { throw 'Remote size requires a numeric scalar.' }
        $type = $Value.GetType()
        if ($type -in @([byte], [sbyte], [short], [ushort], [int], [uint], [long], [ulong], [System.Numerics.BigInteger])) {
            $integer = [System.Numerics.BigInteger]::Parse($Value.ToString([Globalization.CultureInfo]::InvariantCulture))
        } elseif ($type -in @([double], [single])) {
            $number = [double]$Value
            if ([double]::IsNaN($number) -or [double]::IsInfinity($number) -or [math]::Truncate($number) -ne $number) { throw 'Remote size must be an integral numeric scalar.' }
            $integer = [System.Numerics.BigInteger]::new($number)
        } elseif ($type -eq [decimal]) {
            if ([decimal]::Truncate($Value) -ne $Value) { throw 'Remote size must be an integral numeric scalar.' }
            $integer = [System.Numerics.BigInteger]::new([decimal]$Value)
        } else { throw 'Remote size requires a numeric scalar; coercion is forbidden.' }
        if ($integer -lt [System.Numerics.BigInteger]::Zero -or $integer -gt [System.Numerics.BigInteger]::new([long]::MaxValue)) { throw 'Remote size is outside nonnegative Int64 bounds.' }
        return [long]$integer
    }
}
process {
    if ($script:HelpShown) { return }
    foreach ($path in @($SourcePath)) {
        $item = Get-Item -LiteralPath (Resolve-Path -LiteralPath $path).ProviderPath -ErrorAction Stop
        if ($item.PSIsContainer) { throw "SourcePath must be a file: $($item.FullName)" }
        if ($script:SeenSource.ContainsKey($item.FullName)) { throw "Duplicate archive source: $($item.FullName)" }
        $script:SeenSource[$item.FullName] = $true
        $record = [ordered]@{
            schema = 'pcai-cleanup-archive.v3'; event = 'planned'; batch_id = $BatchId; category = $Category
            observed_utc = [datetime]::UtcNow.ToString('o'); source_path = $item.FullName; source_leaf = $item.Name; size_bytes = $item.Length
            created_time_utc = $item.CreationTimeUtc.ToString('o'); modified_time_utc = $item.LastWriteTimeUtc.ToString('o')
            source_guid_path = $null; source_volume_serial = $null; source_file_id = $null
            hardlink_count_before = $null; hardlink_count_before_delete = $null; allocated_bytes = $null; reclaimed_bytes_estimate = $null
            archive_remote = $null; execute = $script:IsExecuting; delete_after_verify = [bool]$DeleteAfterVerify
            uploaded = $false; upload_attempted = $false; copy_completed = $false; upload_outcome = 'not-attempted'
            verified = $false; deleted = $false; local_md5 = $null; source_sha256 = $null
            remote_md5 = $null; remote_size_bytes = $null; remote_count = $null; rclone_log = $null
            receipt_guid_path = $null; receipt_volume_serial = $null; receipt_file_id = $null
            verified_at_utc = $null; deleted_at_utc = $null; error = $null
        }
        $custody = $null
        $performed = $false
        try {
            if ($script:IsExecuting -and $PSCmdlet.ShouldProcess($item.FullName, 'Archive held bytes and verify remote receipt')) {
                $performed = $true
                Initialize-ArchiveCustody
                $custody = [Pcai.Storage.PcaiArchiveSource]::new($item.FullName)
                $item.Refresh()
                $record.created_time_utc = $item.CreationTimeUtc.ToString('o')
                $record.modified_time_utc = $item.LastWriteTimeUtc.ToString('o')
                $record.size_bytes = $custody.Length
                $record.source_guid_path = $custody.CanonicalGuidPath
                $record.source_volume_serial = $custody.VolumeSerialHex
                $record.source_file_id = $custody.FileId128
                $identityKey = "$($custody.VolumeSerialHex):$($custody.FileId128)"
                if ($script:SeenIdentity.ContainsKey($identityKey)) { throw 'Duplicate held file identity in archive run; alias preserved.' }
                $script:SeenIdentity[$identityKey] = $true
                $record.hardlink_count_before = $custody.HardLinkCount
                $record.allocated_bytes = $custody.AllocatedBytes
                $record.local_md5 = $custody.Md5
                $record.source_sha256 = $custody.Sha256
                $pathHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($custody.CanonicalGuidPath.ToLowerInvariant()))).Substring(0, 16).ToLowerInvariant()
                $remoteDir = "$RcloneRemote$RemoteRoot/$BatchId/$Category"
                $extension = if ($item.Extension -match '^\.[a-zA-Z][a-zA-Z0-9]{0,9}$' -and $item.Extension -notmatch '\d{8}') { $item.Extension.ToLowerInvariant() } else { '.bin' }
                $leaf = 'payload-{0}-{1}{2}' -f $pathHash, $custody.Sha256, $extension
                $record.archive_remote = "$remoteDir/$leaf"
                $record.rclone_log = Join-Path $script:ManifestDir "rclone-$pathHash-$($custody.Sha256).log"
                $record.event = 'attempted'
                Write-ArchiveReceipt -Record $record -Custody $custody
                Invoke-ArchiveRclone -ArgumentList @('mkdir', $remoteDir) | Out-Null
                $record.upload_attempted = $true
                $record.uploaded = $null
                $record.upload_outcome = 'copy-requested'
                Invoke-ArchiveRclone -ArgumentList @('copyto', $custody.SourcePath, $record.archive_remote, '--ignore-existing', '--drive-chunk-size', $DriveChunkSize, '--transfers', '1', '--checkers', '4', '--checksum', '--log-file', $record.rclone_log) | Out-Null
                $record.copy_completed = $true
                $record.upload_outcome = 'copied-or-existing'
                $hashLines = @(Invoke-ArchiveRclone -ArgumentList @('md5sum', $record.archive_remote))
                if ($hashLines.Count -ne 1 -or $hashLines[0] -notmatch '^([a-fA-F0-9]{32})\s+.+$') { throw 'Remote MD5 must contain exactly one valid file hash.' }
                $record.remote_md5 = $Matches[1].ToLowerInvariant()
                $remoteSize = ConvertFrom-Json -InputObject ((Invoke-ArchiveRclone -ArgumentList @('size', $record.archive_remote, '--json')) -join "`n") -NoEnumerate -ErrorAction Stop
                if ($null -eq $remoteSize -or $remoteSize.GetType() -ne [System.Management.Automation.PSCustomObject]) { throw 'Remote size must be exactly one JSON object.' }
                $record.remote_size_bytes = ConvertTo-ArchiveInteger $remoteSize.bytes
                $record.remote_count = ConvertTo-ArchiveInteger $remoteSize.count
                if ($record.local_md5 -ne $record.remote_md5 -or $custody.Length -ne $record.remote_size_bytes -or $record.remote_count -ne 1) { throw 'Remote hash, size or count differs from held source bytes.' }
                $record.verified = $true
                $record.verified_at_utc = [datetime]::UtcNow.ToString('o')
                $record.event = 'verified'
                Write-ArchiveReceipt -Record $record -Custody $custody
                if ($DeleteAfterVerify -and $PSCmdlet.ShouldProcess($item.FullName, 'Delete the exact verified file identity')) {
                    $custody.DeleteVerified()
                    $record.hardlink_count_before_delete = $custody.HardLinkCountBeforeDeletion
                    if ($custody.HardLinkCountBeforeDeletion -eq 1) { $record.reclaimed_bytes_estimate = $custody.AllocatedBytes }
                    $record.deleted = $true
                    $record.deleted_at_utc = [datetime]::UtcNow.ToString('o')
                    $record.event = 'deleted'
                    Write-ArchiveReceipt -Record $record -Custody $custody
                }
            } elseif ($RecordDryRun -and -not $DryRun -and $PSCmdlet.ShouldProcess($script:ManifestFile, 'Append planned archive receipt')) {
                Initialize-ArchiveCustody
                Write-ArchiveReceipt -Record $record
            }
        } catch {
            $record.error = $_.Exception.Message
            $record.event = 'error'
            if ($performed -and ('Pcai.Storage.PcaiArchiveSource' -as [type])) {
                try { Write-ArchiveReceipt -Record $record -Custody $custody } catch { throw "Archive failed: $($record.error) Receipt failed: $($_.Exception.Message)" }
            }
            throw
        } finally {
            if ($null -ne $custody) { $custody.Dispose() }
            $script:Records.Add([pscustomobject]$record)
        }
    }
}
end { if (-not $script:HelpShown) { $script:Records } }
