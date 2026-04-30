#Requires -Version 5.1
<#
.SYNOPSIS
    Migrates text-based profile logs to JSONL format.

.DESCRIPTION
    This utility converts legacy text-based profile logs to the new JSONL format
    with checksums, machine IDs, and structured fields.

    Features:
    - Parses existing [timestamp][Level][Component] format
    - Adds machine ID and process ID (marked as 'migrated')
    - Computes CRC32 checksums for each entry
    - Preserves original timestamp information
    - Handles multi-month log files
    - Creates backup before migration
    - Reports migration statistics

.PARAMETER LogPath
    Path to the logs directory. Defaults to $env:USERPROFILE\.machine\logs

.PARAMETER BackupPath
    Path for backup files. Defaults to $LogPath\backup-pre-jsonl

.PARAMETER WhatIf
    Shows what would be migrated without making changes.

.EXAMPLE
    .\Migrate-ProfileLogs.ps1
    Migrates all text logs in default location.

.EXAMPLE
    .\Migrate-ProfileLogs.ps1 -WhatIf
    Shows migration plan without making changes.

.NOTES
    Version: 1.0.0
    Created: 2026-01-05
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$LogPath = (Join-Path $env:USERPROFILE '.machine\logs'),
    [string]$BackupPath
)

#region Configuration
$script:LogVersion = 2
$script:MachineId = $null

# CRC32 lookup table
$script:CrcTable = @(
    0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3,
    0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988, 0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91,
    0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
    0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5,
    0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172, 0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B,
    0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
    0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F,
    0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924, 0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D,
    0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
    0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D, 0x91646C97, 0xE6635C01,
    0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E, 0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457,
    0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
    0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB,
    0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0, 0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9,
    0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
    0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81, 0xB7BD5C3B, 0xC0BA6CAD,
    0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A, 0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683,
    0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
    0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7,
    0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC, 0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5,
    0xD6D6A3E8, 0xA1D1937E, 0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
    0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55, 0x316E8EEF, 0x4669BE79,
    0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236, 0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F,
    0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
    0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F, 0x72076785, 0x05005713,
    0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38, 0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21,
    0x86D3D2D4, 0xF1D4E242, 0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
    0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45,
    0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2, 0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB,
    0xAED16A4A, 0xD9D65ADC, 0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
    0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD706B3, 0x54DE5729, 0x23D967BF,
    0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94, 0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
)
#endregion

#region Helper Functions
function Get-MachineId {
    if ($script:MachineId) { return $script:MachineId }

    try {
        $regPath = 'HKLM:\SOFTWARE\Microsoft\Cryptography'
        $guid = (Get-ItemProperty -Path $regPath -Name MachineGuid -ErrorAction Stop).MachineGuid
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($guid)
        $hash = [System.Security.Cryptography.SHA256]::Create().ComputeHash($bytes)
        $script:MachineId = [BitConverter]::ToString($hash[0..2]).Replace('-', '').ToLower()
    } catch {
        $hostname = $env:COMPUTERNAME
        if (-not $hostname) { $hostname = 'unknown' }
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($hostname)
        $hash = [System.Security.Cryptography.SHA256]::Create().ComputeHash($bytes)
        $script:MachineId = [BitConverter]::ToString($hash[0..2]).Replace('-', '').ToLower()
    }

    return $script:MachineId
}

function Get-CRC32 {
    param([string]$InputString)

    $bytes = [System.Text.Encoding]::UTF8.GetBytes($InputString)
    # Use long to avoid PS5.1 uint32 issues
    [long]$crc = 4294967295  # 0xFFFFFFFF as decimal

    foreach ($byte in $bytes) {
        $index = [int](($crc -bxor $byte) -band 0xFF)
        $crc = (($crc -shr 8) -bxor [long]$script:CrcTable[$index]) -band 4294967295
    }

    return (($crc -bxor 4294967295) -band 4294967295).ToString('x8')
}

function Convert-TextLogLineToJsonl {
    <#
    .SYNOPSIS
        Converts a single text log line to JSONL format.
    #>
    param(
        [string]$Line,
        [int]$SequenceNumber
    )

    # Parse format: [timestamp][Level][Component] Message
    # Example: [2026-01-05 14:17:44.800][Info][Profile] Profile loaded in 1692ms

    $pattern = '^\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]\[(\w+)\]\[(\w+)\]\s*(.*)$'

    if ($Line -match $pattern) {
        $timestamp = $Matches[1]
        $level = $Matches[2]
        $component = $Matches[3]
        $message = $Matches[4]

        # Convert timestamp to ISO 8601 UTC
        try {
            $dt = [DateTime]::ParseExact($timestamp, @('yyyy-MM-dd HH:mm:ss.fff', 'yyyy-MM-dd HH:mm:ss'), $null)
            $isoTimestamp = $dt.ToUniversalTime().ToString('o')
        } catch {
            $isoTimestamp = [DateTime]::UtcNow.ToString('o')
        }

        # Build entry without checksum
        $entry = [ordered]@{
            v    = $script:LogVersion
            ts   = $isoTimestamp
            lvl  = $level
            cmp  = $component
            msg  = $message
            mid  = Get-MachineId
            pid  = 0  # Unknown - marked as 0 for migrated entries
            seq  = $SequenceNumber
            migrated = $true  # Flag indicating this was migrated from text format
        }

        # Compute checksum
        $json = $entry | ConvertTo-Json -Compress
        $entry['chk'] = Get-CRC32 -InputString $json

        return $entry
    }

    # Return null for unparseable lines
    return $null
}
#endregion

#region Main Migration Logic
function Start-LogMigration {
    param(
        [string]$SourcePath,
        [string]$BackupDir,
        [switch]$WhatIfMode
    )

    Write-Host "=== Profile Log Migration ===" -ForegroundColor Cyan
    Write-Host "Source: $SourcePath"
    Write-Host "Machine ID: $(Get-MachineId)"
    Write-Host ""

    # Find all text log files (profile_YYYY-MM.log format, not .jsonl)
    $textLogs = Get-ChildItem -Path $SourcePath -Filter 'profile_*.log' -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch '\.jsonl$' }

    if (-not $textLogs) {
        Write-Host "No text log files found to migrate." -ForegroundColor Yellow
        return
    }

    Write-Host "Found $($textLogs.Count) text log file(s) to migrate:" -ForegroundColor Yellow
    foreach ($log in $textLogs) {
        $lineCount = (Get-Content $log.FullName | Measure-Object).Count
        Write-Host "  - $($log.Name) ($lineCount lines)"
    }
    Write-Host ""

    if ($WhatIfMode) {
        Write-Host "WhatIf mode - no changes will be made." -ForegroundColor Magenta
        return
    }

    # Create backup directory
    if (-not [System.IO.Directory]::Exists($BackupDir)) {
        [System.IO.Directory]::CreateDirectory($BackupDir) | Out-Null
        Write-Host "Created backup directory: $BackupDir" -ForegroundColor Green
    }

    $totalStats = @{
        FilesProcessed = 0
        EntriesMigrated = 0
        EntriesSkipped = 0
        Errors = 0
    }

    foreach ($textLog in $textLogs) {
        Write-Host "Processing: $($textLog.Name)..." -ForegroundColor Yellow

        # Extract month stamp from filename
        if ($textLog.Name -match 'profile_(\d{4}-\d{2})\.log$') {
            $monthStamp = $Matches[1]
        } else {
            Write-Host "  Skipping - unexpected filename format" -ForegroundColor Red
            continue
        }

        # Backup original file
        $backupFile = Join-Path $BackupDir $textLog.Name
        Copy-Item -Path $textLog.FullName -Destination $backupFile -Force
        Write-Host "  Backed up to: $backupFile" -ForegroundColor Gray

        # Read source file
        $lines = [System.IO.File]::ReadAllLines($textLog.FullName, [System.Text.Encoding]::UTF8)

        # Convert to JSONL
        $jsonlLines = [System.Collections.Generic.List[string]]::new()
        $sequence = 0

        foreach ($line in $lines) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }

            $sequence++
            $entry = Convert-TextLogLineToJsonl -Line $line -SequenceNumber $sequence

            if ($entry) {
                $jsonlLines.Add(($entry | ConvertTo-Json -Compress))
                $totalStats.EntriesMigrated++
            } else {
                # Keep unparseable lines as raw entries
                $rawEntry = [ordered]@{
                    v = $script:LogVersion
                    ts = [DateTime]::UtcNow.ToString('o')
                    lvl = 'Info'
                    cmp = 'Migration'
                    msg = $line
                    mid = Get-MachineId
                    pid = 0
                    seq = $sequence
                    migrated = $true
                    raw = $true
                }
                $json = $rawEntry | ConvertTo-Json -Compress
                $rawEntry['chk'] = Get-CRC32 -InputString $json
                $jsonlLines.Add(($rawEntry | ConvertTo-Json -Compress))
                $totalStats.EntriesSkipped++
            }
        }

        # Write JSONL file
        $jsonlFile = Join-Path $SourcePath "profile_$monthStamp.jsonl"

        # If JSONL file already exists, append; otherwise create
        if ([System.IO.File]::Exists($jsonlFile)) {
            # Prepend migrated entries (they're older)
            $existingContent = [System.IO.File]::ReadAllText($jsonlFile, [System.Text.Encoding]::UTF8)
            $migratedContent = ($jsonlLines -join "`n") + "`n"
            [System.IO.File]::WriteAllText($jsonlFile, $migratedContent + $existingContent, [System.Text.Encoding]::UTF8)
            Write-Host "  Prepended to existing JSONL file" -ForegroundColor Gray
        } else {
            [System.IO.File]::WriteAllLines($jsonlFile, $jsonlLines, [System.Text.Encoding]::UTF8)
            Write-Host "  Created: $jsonlFile" -ForegroundColor Green
        }

        # Rename original to indicate it's been migrated
        $migratedName = $textLog.FullName + '.migrated'
        Rename-Item -Path $textLog.FullName -NewName $migratedName -Force
        Write-Host "  Renamed original to: $($textLog.Name).migrated" -ForegroundColor Gray

        $totalStats.FilesProcessed++
        Write-Host "  Migrated $($jsonlLines.Count) entries" -ForegroundColor Green
        Write-Host ""
    }

    # Summary
    Write-Host "=== Migration Complete ===" -ForegroundColor Cyan
    Write-Host "Files processed:   $($totalStats.FilesProcessed)"
    Write-Host "Entries migrated:  $($totalStats.EntriesMigrated)" -ForegroundColor Green
    Write-Host "Entries as raw:    $($totalStats.EntriesSkipped)" -ForegroundColor Yellow
    Write-Host "Backup location:   $BackupDir"
    Write-Host ""
    Write-Host "Original files renamed to *.log.migrated" -ForegroundColor Gray
    Write-Host "To rollback: rename *.log.migrated back to *.log and delete *.jsonl files" -ForegroundColor Gray
}
#endregion

#region Entry Point
# Set default backup path
if (-not $BackupPath) {
    $BackupPath = Join-Path $LogPath 'backup-pre-jsonl'
}

# Verify log path exists
if (-not [System.IO.Directory]::Exists($LogPath)) {
    Write-Host "Log path does not exist: $LogPath" -ForegroundColor Red
    exit 1
}

# Run migration
Start-LogMigration -SourcePath $LogPath -BackupDir $BackupPath -WhatIfMode:$WhatIf
#endregion
