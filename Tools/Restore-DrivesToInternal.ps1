<#
.SYNOPSIS
Moves cloud-cache-disk.vhdx (F:) and shared-dev.vhdx (W:) off the removable D:
enclosure onto internal T:, then repoints their scheduled tasks.

.DESCRIPTION
UNTESTED until D: is reconnected - it could not be exercised when written
(2026-09-09) because the ACASIS enclosure was detached. Run -DryRun FIRST.

Order matters. Step 0 is a survey that REFUSES to continue if the expected files
are missing, because D: has been described as scratch space that gets emptied,
and the VHDXs exist nowhere else on this machine (verified by a reparse-skipping
walk of C:\ and T:\ with positive controls, 2026-09-09).

The space problem this solves:
  T: free    2,117 GB
  F: volume  2,362 GB   <- does not fit; compaction must reclaim >= ~245 GB
W: (~306 GB compacted) fits comfortably and is moved first, which also frees
nothing on T: but gets the smaller, safer move done and verified first.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$DryRun,
    [string]$TargetDir = 'T:\vm',
    [int]$RequiredHeadroomGB = 150,
    # Drive letter of the source enclosure, e.g. 'E'. Omit to locate it by
    # identity (the NTFS volume on a disk larger than 7 TB). The letter D: is
    # NOT assumed: it is already held by a different, FAT32 disk, so the
    # returning enclosure gets whatever letter is free. Hard-wiring D: made this
    # script's own documented recovery unreachable.
    [string]$SourceDriveLetter,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)
Set-StrictMode -Version 2.0

# CLI contract (AGENTS.md): -h/--help and --DryRun long forms, handled before
# anything resolves a path or queries a disk.
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
    return
}
if (@($CliArgs).Count -gt 0) { throw "Unknown CLI argument(s): $($CliArgs -join ', ')" }
$ErrorActionPreference = 'Stop'
if ($DryRun) { $WhatIfPreference = $true }

function Say { param($m, $l = 'INFO') Write-Host ("[{0}] {1}" -f $l, $m) }

# ---------------------------------------------------------------- Step 0: survey
Say 'STEP 0 - survey the source volume before touching anything'

# The drive LETTER is not an identity. Measured 2026-09-09: while the 8 TB
# ACASIS/SN850X that actually hosts these VHDXs was detached, letter D: was taken
# by a completely different disk - a 953 GB FAT32 "FATDRIVE" on a Sabrent USB SSD
# holding audiometric calibration data. A guard that only checks `Test-Path D:\`
# passes on that wrong disk. Verify the volume can plausibly BE the source:
# NTFS (FAT32 caps files at 4 GB, so it cannot hold a 2.2 TB VHDX) and large enough.
if ($PSBoundParameters.ContainsKey('SourceDriveLetter') -and -not [string]::IsNullOrWhiteSpace($SourceDriveLetter)) {
    $srcLetter = $SourceDriveLetter.TrimEnd(':', '\')
    Say "  using caller-supplied source drive ${srcLetter}:"
} else {
    # Locate by identity, not by letter. Every disk over 7 TB with a lettered
    # partition is a candidate; the NTFS + size checks below still have to pass,
    # so a wrong guess fails loudly rather than copying to the wrong place.
    Say '  locating the source enclosure by size (>7 TB), since D: is not a reliable identity...'
    $candidates = @(
        Get-Disk -ErrorAction SilentlyContinue |
            Where-Object { $_.Size -gt 7TB } |
            Get-Partition -ErrorAction SilentlyContinue |
            Where-Object { $_.DriveLetter } |
            ForEach-Object { [string]$_.DriveLetter }
    )
    if ($candidates.Count -eq 0) {
        throw ('No disk larger than 7 TB with a lettered partition is attached. Connect the ACASIS ' +
            'TBU405Pro enclosure, or pass -SourceDriveLetter explicitly. NOTE: do NOT assume D: - ' +
            'that letter is held by a different, FAT32 disk.')
    }
    if ($candidates.Count -gt 1) {
        throw ("Multiple disks over 7 TB have lettered partitions ($($candidates -join ', ')). " +
            'Disambiguate with -SourceDriveLetter.')
    }
    $srcLetter = $candidates[0]
    Say "  found source enclosure at ${srcLetter}:"
}

if (-not (Test-Path "${srcLetter}:\")) { throw "${srcLetter}: is not attached. Connect the source enclosure first." }
$srcVol = Get-Volume -DriveLetter $srcLetter -ErrorAction Stop
Say ("  {0}: = label '{1}' {2} {3:N1} GB" -f $srcLetter, $srcVol.FileSystemLabel, $srcVol.FileSystem, ($srcVol.Size / 1GB))
if ($srcVol.FileSystem -ne 'NTFS') {
    throw ("${srcLetter}: is $($srcVol.FileSystem), not NTFS (label '$($srcVol.FileSystemLabel)'). " +
        'This is NOT the enclosure that holds the VHDXs - FAT32 cannot store a file over 4 GB.')
}
if ($srcVol.Size -lt 2TB) {
    throw ("${srcLetter}: is only {0:N0} GB - too small to be the 8 TB source enclosure. Wrong disk." -f ($srcVol.Size / 1GB))
}
$SourceRoot = "${srcLetter}:\vm"

# W: ONLY. cloud-cache-disk was deliberately REMOVED from this list on 2026-09-09.
#
# F: was rebuilt empty on internal T: (500 GB dynamic) because it only ever backed
# Google Drive's content cache, which re-fetches on demand. The old 1,860 GB cache
# VHDX on the enclosure is NOT needed. Moving it back would (a) not fit - T: has
# ~1.37 TB free - and (b) overwrite the working F: this script's own siblings now
# depend on. shared-dev.vhdx is different: 710,352 files of real, non-reproducible
# data, which is why it is still worth recovering.
$items = @(
    @{ Name = 'shared-dev'; Src = (Join-Path $SourceRoot 'shared-dev.vhdx'); Task = 'AutoMount_VHDX_shared-dev'; Letter = 'W'; Label = 'WSL-Shared-Dev'; Delay = 'PT1M'; StartupDelay = 60 }
)

# Belt and braces: refuse outright if a working cloud-cache-disk is mounted, in case
# someone re-adds F: to $items without reading the note above.
$existingF = Get-Volume -FileSystemLabel 'cloud-cache-disk' -ErrorAction SilentlyContinue
if ($existingF -and ($items | Where-Object { $_.Label -eq 'cloud-cache-disk' })) {
    throw ("A 'cloud-cache-disk' volume is already mounted at $($existingF.DriveLetter): - " +
        'refusing to restore an older copy over it. Remove cloud-cache-disk from $items.')
}

$missing = @()
foreach ($i in $items) {
    if (Test-Path -LiteralPath $i.Src) {
        $gb = [math]::Round((Get-Item -LiteralPath $i.Src).Length / 1GB, 1)
        Say ("  FOUND {0}  {1} GB" -f $i.Src, $gb)
    } else { $missing += $i.Src; Say ("  MISSING {0}" -f $i.Src) 'ERROR' }
}

# Sole-copy guard. Per D:\staging-for-deletion\DO-NOT-WIPE.md these paths held the
# ONLY copies of four repos. D: being called "scratch" does not make that false.
foreach ($p in 'D:\repo-backups', 'D:\archive', 'D:\staging-for-deletion\DO-NOT-WIPE.md') {
    Say ("  sole-copy check: {0} = {1}" -f $p, $(if (Test-Path $p) { 'present' } else { '*** ABSENT ***' }))
}

if ($missing.Count) {
    throw ("Refusing to continue - these do not exist on D: and exist nowhere else: {0}. " -f ($missing -join ', ') +
        'If D: was wiped, F:/W: must be rebuilt from scratch and the cloud clients re-synced instead.')
}

# ------------------------------------------------------- Step 1: detach if mounted
Say 'STEP 1 - detach any currently-attached copy'
foreach ($i in $items) {
    $d = Get-Disk | Where-Object { $_.Location -eq $i.Src }
    if ($d) {
        if ($PSCmdlet.ShouldProcess($i.Src, 'Dismount-VHD')) { Dismount-VHD -Path $i.Src }
        Say ("  dismounted {0}" -f $i.Src)
    } else { Say ("  not attached: {0}" -f $i.Src) }
}

# -------------------------------------------------------------- Step 2: compact
Say 'STEP 2 - compact in place on D: (cheap there; avoids moving dead space)'
foreach ($i in $items) {
    $before = (Get-Item -LiteralPath $i.Src).Length / 1GB
    if ($PSCmdlet.ShouldProcess($i.Src, 'Optimize-VHD -Mode Full')) {
        # Optimize-VHD -Mode Full requires the VHDX to be ATTACHED read-only.
        # Dismounting first does not error - it silently degrades to Prezeroed
        # and reclaims less, which matters because the fit check below throws if
        # the result does not fit on the target. Keep the mount across the
        # optimize and release it in finally so a failure cannot strand it.
        Mount-VHD -Path $i.Src -ReadOnly -NoDriveLetter
        try {
            Optimize-VHD -Path $i.Src -Mode Full
        } finally {
            Dismount-VHD -Path $i.Src -ErrorAction SilentlyContinue
        }
    }
    $after = if ($DryRun) { $before } else { (Get-Item -LiteralPath $i.Src).Length / 1GB }
    Say ("  {0}: {1:N1} -> {2:N1} GB (reclaimed {3:N1})" -f $i.Name, $before, $after, ($before - $after))
}

# ----------------------------------------------------------- Step 3: fit check
Say 'STEP 3 - prove it fits BEFORE copying'
$need = 0; foreach ($i in $items) { $need += (Get-Item -LiteralPath $i.Src).Length }
$freeT = (Get-Volume -DriveLetter T).SizeRemaining
Say ("  need {0:N1} GB, T: free {1:N1} GB, headroom required {2} GB" -f ($need / 1GB), ($freeT / 1GB), $RequiredHeadroomGB)
if (($freeT - $need) -lt ($RequiredHeadroomGB * 1GB)) {
    throw ('Will not fit with the required headroom. Shrink F: further (Shrink-Partition + Resize-VHD) or free space on T:.')
}

# ------------------------------------------------- Step 4: copy, verify, then swap
# Copy (not move) so the source survives until the copy is proven.
Say 'STEP 4 - copy to T:, verify size, then repoint'
foreach ($i in ($items | Sort-Object { (Get-Item -LiteralPath $_.Src).Length })) {
    $dst = Join-Path $TargetDir (Split-Path $i.Src -Leaf)
    if ($PSCmdlet.ShouldProcess($i.Src, "copy -> $dst")) {
        Copy-Item -LiteralPath $i.Src -Destination $dst -Force
        $s = (Get-Item -LiteralPath $i.Src).Length; $d = (Get-Item -LiteralPath $dst).Length
        if ($s -ne $d) { throw ("Copy size mismatch for {0}: {1} vs {2}" -f $i.Name, $s, $d) }
        Say ("  copied and size-verified: {0}" -f $dst)
    }

    $exe = 'C:\Program Files\PowerShell\7\pwsh.exe'
    $args = '-NoLogo -NoProfile -ExecutionPolicy Bypass -File "C:\codedev\PC_AI\Tools\Mount-PersistentVHDX.ps1" ' +
    ('-VhdPath "{0}" -TaskName "{1}" -ExpectedState Volume -StartupDelaySeconds {2} -WaitForVhdSeconds 120 ' -f $dst, $i.Task, $i.StartupDelay) +
    ('-LogRoot "C:\codedev\PC_AI\Tools\..\Logs\VHDMount" -ExpectedVolumeLabel "{0}" -ExpectedDriveLetter {1} -ExpectedFileSystem NTFS' -f $i.Label, $i.Letter)
    if ($PSCmdlet.ShouldProcess($i.Task, 'repoint scheduled task to T:')) {
        $tr = New-ScheduledTaskTrigger -AtStartup; $tr.Delay = $i.Delay
        Set-ScheduledTask -TaskName $i.Task -Trigger $tr -Action (New-ScheduledTaskAction -Execute $exe -Argument $args) | Out-Null
        $v = Get-ScheduledTask -TaskName $i.Task
        if ($v.Actions[0].Arguments -notlike "*$dst*") { throw ("Task {0} did not persist the new path" -f $i.Task) }
        Say ("  task repointed and re-read OK: {0}" -f $i.Task)
    }
}

# ------------------------------------------------------------- Step 5: prove it
Say 'STEP 5 - run the tasks and verify by LABEL (a stick can take F:)'
if (-not $DryRun) {
    foreach ($i in $items) { Start-ScheduledTask -TaskName $i.Task; Start-Sleep -Seconds 20 }
    foreach ($i in $items) {
        $v = Get-Volume -FileSystemLabel $i.Label -ErrorAction SilentlyContinue
        if ($v) { Say ("  OK {0}: mounted at {1}: " -f $i.Label, $v.DriveLetter) }
        else { Say ("  FAILED to mount {0}" -f $i.Label) 'ERROR' }
    }
    Say 'Now run: Start-ScheduledTask -TaskName CloudClients-AfterVHDX'
}

Say 'STEP 6 - MANUAL: delete D:\vm\*.vhdx only after F:/W: have survived a real reboot from T:.'
