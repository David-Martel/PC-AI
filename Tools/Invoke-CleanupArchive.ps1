#Requires -Version 7.0
<#
.SYNOPSIS
    Archive cleanup candidates to Google Drive through rclone.

.DESCRIPTION
    Uploads explicit source files to a dated Drive archive folder, verifies the
    upload with rclone check, records a JSONL manifest, and optionally deletes
    local files only after verification succeeds.

    The default mode is a dry run. Pass -Execute to upload. Pass
    -DeleteAfterVerify only when verified archive evidence is sufficient for
    local removal.

.PARAMETER SourcePath
    One or more files to archive.

.PARAMETER Category
    Archive subfolder under the batch folder, such as 01-docker-vhds.

.PARAMETER BatchId
    Dated archive batch folder name.

.PARAMETER RemoteRoot
    Root path inside the rclone remote.

.PARAMETER RcloneRemote
    rclone remote name, including trailing colon.

.PARAMETER ManifestPath
    JSONL manifest path to append.

.PARAMETER Execute
    Perform uploads and verification. Without this switch, only records planned actions.

.PARAMETER RecordDryRun
    Append dry-run records to the manifest. By default, dry runs do not write files.

.PARAMETER DeleteAfterVerify
    Delete each local source file only after rclone check succeeds.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory, ValueFromPipeline, ValueFromPipelineByPropertyName)]
    [Alias('FullName')]
    [string[]]$SourcePath,

    [string]$Category = '04-reports',
    [string]$BatchId = '2026-06-26-cleanup-lane',
    [string]$RemoteRoot = 'Archives/Workstation Cleanup',
    [string]$RcloneRemote = 'gdrive-personal:',
    [string]$ManifestPath = (Join-Path (Resolve-Path .) 'Reports/cleanup-archive-manifest-20260626.jsonl'),
    [string]$DriveChunkSize = '512M',
    [switch]$Execute,
    [switch]$RecordDryRun,
    [switch]$DeleteAfterVerify
)

begin {
    $rclone = Get-Command rclone -ErrorAction SilentlyContinue
    if (-not $rclone) {
        throw 'rclone is required but was not found on PATH.'
    }

    if (-not $RcloneRemote.EndsWith(':')) {
        throw "RcloneRemote must include a trailing colon, for example 'gdrive-personal:'."
    }

    $manifestDir = Split-Path -Path $ManifestPath -Parent
    if ($manifestDir -and -not (Test-Path -LiteralPath $manifestDir)) {
        New-Item -ItemType Directory -Path $manifestDir -Force | Out-Null
    }

    $records = New-Object System.Collections.Generic.List[object]
}

process {
    foreach ($path in $SourcePath) {
        $resolved = Resolve-Path -LiteralPath $path -ErrorAction Stop
        $item = Get-Item -LiteralPath $resolved.ProviderPath -ErrorAction Stop
        if ($item.PSIsContainer) {
            throw "SourcePath must be a file, not a directory: $($item.FullName)"
        }

        $remoteDir = "$RcloneRemote$RemoteRoot/$BatchId/$Category"
        $remoteFile = "$remoteDir/$($item.Name)"
        $logPath = Join-Path $manifestDir ("rclone-$($item.BaseName)-$(Get-Date -Format 'yyyyMMdd-HHmmss').log")

        $record = [ordered]@{
            schema = 'pcai-cleanup-archive.v1'
            batch_id = $BatchId
            category = $Category
            source_path = $item.FullName
            archive_remote = $remoteFile
            size_bytes = $item.Length
            created_time_utc = $item.CreationTimeUtc.ToString('o')
            modified_time_utc = $item.LastWriteTimeUtc.ToString('o')
            execute = [bool]$Execute
            delete_after_verify = [bool]$DeleteAfterVerify
            uploaded = $false
            verified = $false
            deleted = $false
            local_md5 = $null
            remote_md5 = $null
            remote_size_bytes = $null
            rclone_log = $logPath
            verified_at_utc = $null
            deleted_at_utc = $null
            error = $null
        }

        try {
            if ($Execute) {
                if ($PSCmdlet.ShouldProcess($item.FullName, "Archive to $remoteFile")) {
                    & rclone mkdir $remoteDir
                    if ($LASTEXITCODE -ne 0) {
                        throw "rclone mkdir failed for $remoteDir"
                    }

                    & rclone copyto $item.FullName $remoteFile --drive-chunk-size $DriveChunkSize --transfers 1 --checkers 4 --checksum --log-file $logPath
                    if ($LASTEXITCODE -ne 0) {
                        throw "rclone copyto failed for $($item.FullName)"
                    }
                    $record.uploaded = $true

                    $localHashLine = (& rclone md5sum $item.FullName) | Select-Object -First 1
                    if ($LASTEXITCODE -ne 0) {
                        throw "rclone md5sum failed for $($item.FullName)"
                    }

                    $remoteHashLine = (& rclone md5sum $remoteFile) | Select-Object -First 1
                    if ($LASTEXITCODE -ne 0) {
                        throw "rclone md5sum failed for $remoteFile"
                    }

                    $remoteSizeJson = (& rclone size $remoteFile --json) -join "`n"
                    if ($LASTEXITCODE -ne 0) {
                        throw "rclone size failed for $remoteFile"
                    }

                    $remoteSize = $remoteSizeJson | ConvertFrom-Json
                    $record.local_md5 = (($localHashLine -split '\s+', 2)[0]).ToLowerInvariant()
                    $record.remote_md5 = (($remoteHashLine -split '\s+', 2)[0]).ToLowerInvariant()
                    $record.remote_size_bytes = [int64]$remoteSize.bytes

                    if ($record.local_md5 -ne $record.remote_md5) {
                        throw "MD5 mismatch for $($item.FullName): local=$($record.local_md5) remote=$($record.remote_md5)"
                    }

                    if ($item.Length -ne $record.remote_size_bytes) {
                        throw "Size mismatch for $($item.FullName): local=$($item.Length) remote=$($record.remote_size_bytes)"
                    }

                    $record.verified = $true
                    $record.verified_at_utc = (Get-Date).ToUniversalTime().ToString('o')

                    if ($DeleteAfterVerify) {
                        if ($PSCmdlet.ShouldProcess($item.FullName, 'Delete verified local archive candidate')) {
                            Remove-Item -LiteralPath $item.FullName -Force -ErrorAction Stop
                            $record.deleted = $true
                            $record.deleted_at_utc = (Get-Date).ToUniversalTime().ToString('o')
                        }
                    }
                }
            }
        }
        catch {
            $record.error = $_.Exception.Message
            Write-Error $record.error
        }
        finally {
            if ($Execute -or $RecordDryRun) {
                $json = [pscustomobject]$record | ConvertTo-Json -Compress -Depth 5
                Add-Content -LiteralPath $ManifestPath -Value $json -Encoding UTF8
            }
            $records.Add([pscustomobject]$record) | Out-Null
        }
    }
}

end {
    $records
}
