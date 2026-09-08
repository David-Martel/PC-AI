#Requires -Version 7.0
<#
.SYNOPSIS
    Archive cleanup candidates to Google Drive through rclone.

.DESCRIPTION
    Uploads explicit source files to a dated Drive archive folder, verifies each
    upload by comparing rclone md5sum and rclone size against the local file,
    records a JSONL manifest, and optionally deletes local files only after that
    verification succeeds.

    The default mode is a preview. Pass -Execute to upload. Pass
    -DeleteAfterVerify only when verified archive evidence is sufficient for
    local removal.

    A preview performs no writes at all: it does not require rclone, does not
    create the manifest directory, and does not append to the manifest unless
    -RecordDryRun is given.

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
    Perform uploads and verification. Without this switch, only records planned
    actions.

.PARAMETER RecordDryRun
    Append preview records to the manifest. By default a preview writes nothing.

.PARAMETER DeleteAfterVerify
    Delete each local source file only after its md5 and size verification
    succeeds AND the verified record has been durably written to the manifest.

.PARAMETER DryRun
    Force preview mode even if -Execute is given. The long CLI form `--DryRun`
    is also accepted.

.PARAMETER Help
    Print script help and exit. The aliases `-h` and `--help` are also accepted.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(ValueFromPipeline, ValueFromPipelineByPropertyName)]
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
    [switch]$DeleteAfterVerify,
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

begin {
    $CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if (@($CliArgs) -contains '--help') {
        $Help = $true
        $CliArgs = @($CliArgs | Where-Object { $_ -ne '--help' })
    }
    if (@($CliArgs) -contains '--DryRun') {
        $DryRun = $true
        $CliArgs = @($CliArgs | Where-Object { $_ -ne '--DryRun' })
    }
    if ($Help) {
        $helpMatch = [regex]::Match((Get-Content -LiteralPath $PSCommandPath -Raw), '(?s)<#\s*(.*?)\s*#>')
        if ($helpMatch.Success) { $helpMatch.Groups[1].Value.Trim() } else { Get-Help -Detailed $PSCommandPath }
        $script:HelpShown = $true
        return
    }

    # -DryRun wins over -Execute so a caller can force a safe preview.
    $script:IsExecuting = [bool]$Execute -and -not $DryRun

    # rclone is only needed to actually transfer. Requiring it in a preview made
    # the documented default mode fail on any machine without rclone installed,
    # which also made the tool untestable.
    if ($script:IsExecuting) {
        if (-not (Get-Command rclone -ErrorAction SilentlyContinue)) {
            throw 'rclone is required but was not found on PATH.'
        }
    }

    if (-not $RcloneRemote.EndsWith(':')) {
        throw "RcloneRemote must include a trailing colon, for example 'gdrive-personal:'."
    }

    $script:ManifestDir = Split-Path -Path $ManifestPath -Parent

    # Destination collision guard. The remote key used to be the leaf name
    # alone, so two sources with the same file name -- from different
    # directories, or a re-run of the same batch/category -- mapped to one
    # remote path. The second copyto replaced the first payload, yet each source
    # was verified against the remote straight after its own upload, so both
    # "passed" and both were eligible for deletion. That silently destroyed the
    # first file while the manifest claimed both were archived. The key now
    # carries a hash of the full source path, and duplicates within a run are
    # rejected outright.
    $script:SeenRemote = @{}
    $script:Records = New-Object System.Collections.Generic.List[object]
    $script:ManifestDirReady = $false

    # Directory creation is deferred until something is actually going to be
    # written. It used to run unconditionally in begin{}, so the documented
    # default preview mutated the filesystem -- creating directories while
    # promising to write nothing.
    function Write-ManifestDirectory {
        if ($script:ManifestDirReady) { return }
        if ($script:ManifestDir -and -not (Test-Path -LiteralPath $script:ManifestDir)) {
            New-Item -ItemType Directory -Path $script:ManifestDir -Force | Out-Null
        }
        $script:ManifestDirReady = $true
    }

    function Write-ManifestRecord {
        param([Parameter(Mandatory)][object]$Record)
        Write-ManifestDirectory
        $json = [pscustomobject]$Record | ConvertTo-Json -Compress -Depth 5
        Add-Content -LiteralPath $ManifestPath -Value $json -Encoding UTF8
    }
}

process {
    if ($script:HelpShown) { return }
    foreach ($path in @($SourcePath)) {
        $resolved = Resolve-Path -LiteralPath $path -ErrorAction Stop
        $item = Get-Item -LiteralPath $resolved.ProviderPath -ErrorAction Stop
        if ($item.PSIsContainer) {
            throw "SourcePath must be a file, not a directory: $($item.FullName)"
        }

        # Short, stable discriminator derived from the absolute source path.
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            $pathHash = -join ($sha.ComputeHash(
                    [System.Text.Encoding]::UTF8.GetBytes($item.FullName.ToLowerInvariant())
                ) | Select-Object -First 4 | ForEach-Object { $_.ToString('x2') })
        } finally { $sha.Dispose() }

        $remoteDir = "$RcloneRemote$RemoteRoot/$BatchId/$Category"
        $remoteLeaf = '{0}__{1}{2}' -f $item.BaseName, $pathHash, $item.Extension
        $remoteFile = "$remoteDir/$remoteLeaf"

        if ($script:SeenRemote.ContainsKey($remoteFile)) {
            throw "Duplicate archive destination '$remoteFile' for '$($item.FullName)'; already claimed by '$($script:SeenRemote[$remoteFile])'."
        }
        $script:SeenRemote[$remoteFile] = $item.FullName

        $logPath = Join-Path $script:ManifestDir ("rclone-$($item.BaseName)-$pathHash-$(Get-Date -Format 'yyyyMMdd-HHmmss').log")

        $record = [ordered]@{
            schema              = 'pcai-cleanup-archive.v2'
            event               = if ($script:IsExecuting) { 'attempted' } else { 'planned' }
            batch_id            = $BatchId
            category            = $Category
            source_path         = $item.FullName
            archive_remote      = $remoteFile
            size_bytes          = $item.Length
            created_time_utc    = $item.CreationTimeUtc.ToString('o')
            modified_time_utc   = $item.LastWriteTimeUtc.ToString('o')
            execute             = [bool]$script:IsExecuting
            delete_after_verify = [bool]$DeleteAfterVerify
            uploaded            = $false
            verified            = $false
            deleted             = $false
            local_md5           = $null
            remote_md5          = $null
            remote_size_bytes   = $null
            rclone_log          = $logPath
            verified_at_utc     = $null
            deleted_at_utc      = $null
            error               = $null
        }

        $performed = $false
        try {
            if ($script:IsExecuting -and $PSCmdlet.ShouldProcess($item.FullName, "Archive to $remoteFile")) {
                $performed = $true

                & rclone mkdir $remoteDir
                if ($LASTEXITCODE -ne 0) { throw "rclone mkdir failed for $remoteDir" }

                Write-ManifestDirectory
                & rclone copyto $item.FullName $remoteFile --drive-chunk-size $DriveChunkSize --transfers 1 --checkers 4 --checksum --log-file $logPath
                if ($LASTEXITCODE -ne 0) { throw "rclone copyto failed for $($item.FullName)" }
                $record.uploaded = $true

                $localHashLine = (& rclone md5sum $item.FullName) | Select-Object -First 1
                if ($LASTEXITCODE -ne 0) { throw "rclone md5sum failed for $($item.FullName)" }

                $remoteHashLine = (& rclone md5sum $remoteFile) | Select-Object -First 1
                if ($LASTEXITCODE -ne 0) { throw "rclone md5sum failed for $remoteFile" }

                $remoteSizeJson = (& rclone size $remoteFile --json) -join "`n"
                if ($LASTEXITCODE -ne 0) { throw "rclone size failed for $remoteFile" }

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

                # Persist the verified state BEFORE removing the source. If the
                # process dies, or the manifest append fails because the file is
                # locked or the disk is full, the previous ordering had already
                # deleted the local file with no durable record that it was ever
                # verified -- destroying both the data and its audit trail.
                $record.event = 'verified'
                Write-ManifestRecord $record

                if ($DeleteAfterVerify -and $PSCmdlet.ShouldProcess($item.FullName, 'Delete verified local archive candidate')) {
                    Remove-Item -LiteralPath $item.FullName -Force -ErrorAction Stop
                    $record.deleted = $true
                    $record.deleted_at_utc = (Get-Date).ToUniversalTime().ToString('o')
                    $record.event = 'deleted'
                    Write-ManifestRecord $record
                }
            }
        }
        catch {
            $record.error = $_.Exception.Message
            $record.event = 'error'
            Write-Error $record.error
            if ($performed) { Write-ManifestRecord $record }
        }
        finally {
            # Preview records are opt-in. Previously this keyed off -Execute
            # rather than whether anything actually happened, so `-Execute
            # -WhatIf` still appended to the manifest -- making -WhatIf mutate
            # state, which is precisely what it must never do.
            if (-not $performed -and $RecordDryRun) {
                Write-ManifestRecord $record
            }
            $script:Records.Add([pscustomobject]$record) | Out-Null
        }
    }
}

end {
    if ($script:HelpShown) { return }
    $script:Records
}

