#Requires -Version 5.1
<#
.SYNOPSIS
Starts cloud sync clients only after the F: cloud-cache-disk VHDX is mounted.

.DESCRIPTION
Run at logon in the user's session. Waits for the cloud-cache VHDX volume to
appear and validate, then launches the configured cloud clients.

If the volume never appears the clients are deliberately NOT started and the
script exits nonzero. Starting a sync client whose cache/root is missing is the
failure that silently relocates a sync root onto C: - which is exactly what this
gating exists to prevent.

Validation binds the expected letter and NTFS volume to the attached expected
VHDX image. When required, a successful receipt from this boot must arrive within
the same deadline; an old or absent receipt remains pending until timeout.
If Storage queries cannot resolve the image in the Limited session, a fresh
receipt's exact volume GUID must match live Win32_Volume identity at F:.

.PARAMETER ExpectedLabel
Volume label that identifies the cloud cache disk.

.PARAMETER ExpectedVhdPath
Absolute path to the VHDX whose attached disk must own the expected drive letter.

.PARAMETER TimeoutSeconds
How long to wait for the volume before giving up.

.PARAMETER RequireMountLog
Also require the newest AutoMount result JSON to report ExitCode 0.

.PARAMETER WhatIf
Validate and report without launching anything.

.PARAMETER DryRun
Non-mutating preview: validate the volume and report which clients would be
launched, without launching any and without creating the log directory or
writing a log file. The long CLI form `--DryRun` is also accepted.

.PARAMETER Help
Print script help and exit. The aliases `-h` and `--help` are also accepted.

.EXAMPLE
pwsh -File .\Tools\Start-CloudClientsAfterVHDX.ps1 -WhatIf
.EXAMPLE
pwsh -File .\Tools\Start-CloudClientsAfterVHDX.ps1 -DryRun
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [ValidateNotNullOrEmpty()]
    [string]$ExpectedLabel = 'cloud-cache-disk',
    [ValidatePattern('^[A-Za-z]$')]
    [string]$ExpectedDriveLetter = 'F',
    [ValidateScript({ [System.IO.Path]::IsPathRooted($_) })]
    [string]$ExpectedVhdPath = 'T:\vm\cloud-cache-disk.vhdx',
    [ValidateRange(1, 3600)]
    [int]$TimeoutSeconds = 180,
    [ValidateRange(1, 60)]
    [int]$PollSeconds = 3,
    [switch]$RequireMountLog,
    [switch]$AllowElevated,
    [string]$MountLogRoot = 'C:\codedev\PC_AI\Logs\VHDMount\AutoMount_VHDX_cloud-cache-disk',
    [string]$LogRoot = 'C:\codedev\PC_AI\Logs\CloudClientStart',
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)


# Dot-sourcing exposes the complete gate for in-process contract tests.
$script:CloudClientGateSource = $PSCommandPath
function Invoke-CloudClientGate {
    [CmdletBinding(SupportsShouldProcess = $true)]
    param(
        [ValidateNotNullOrEmpty()]
        [string]$ExpectedLabel = 'cloud-cache-disk',
        [ValidatePattern('^[A-Za-z]$')]
        [string]$ExpectedDriveLetter = 'F',
        [ValidateScript({ [System.IO.Path]::IsPathRooted($_) })]
        [string]$ExpectedVhdPath = 'T:\vm\cloud-cache-disk.vhdx',
        [ValidateRange(1, 3600)]
        [int]$TimeoutSeconds = 180,
        [ValidateRange(1, 60)]
        [int]$PollSeconds = 3,
        [switch]$RequireMountLog,
        [switch]$AllowElevated,
        [string]$MountLogRoot = 'C:\codedev\PC_AI\Logs\VHDMount\AutoMount_VHDX_cloud-cache-disk',
        [string]$LogRoot = 'C:\codedev\PC_AI\Logs\CloudClientStart',
        [switch]$DryRun,
        [Alias('h', '?')]
        [switch]$Help,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$CliArgs
    )

    Set-StrictMode -Version 2.0
    $ErrorActionPreference = 'Stop'

    # CLI contract (AGENTS.md). Handled before the log directory is created, so
    # `-h` and `-DryRun` do not leave a directory behind on a machine that was only
    # ever asked for help.
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
        $helpMatch = [regex]::Match((Get-Content -LiteralPath $script:CloudClientGateSource -Raw), '(?s)<#\s*(.*?)\s*#>')
        if ($helpMatch.Success) {
            Write-Information $helpMatch.Groups[1].Value.Trim() -InformationAction Continue
        } else {
            Get-Help -Detailed $script:CloudClientGateSource | Out-String | Write-Information -InformationAction Continue
        }
        return 0
    }
    if ($DryRun) { $WhatIfPreference = $true }

    # The log file is the one side effect NOT behind ShouldProcess, so -DryRun has
    # to suppress it explicitly. AGENTS.md requires dry run to suppress file writes,
    # and a "preview" that creates a directory and a log file is not a preview.
    $logFile = $null
    if (-not $DryRun -and -not $WhatIfPreference) {
        if (-not (Test-Path $LogRoot)) { New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null }
        $logFile = Join-Path $LogRoot 'current.log'
    }

    function Write-Log {
        param([string]$Message, [string]$Level = 'INFO')
        $line = '{0} [{1}] {2}' -f (Get-Date -Format 'HH:mm:ss'), $Level, $Message
        Write-Information $line -InformationAction Continue
        if ($logFile) { Add-Content -LiteralPath $logFile -Value $line }
    }

    # Clients to launch once the volume is ready. Order matters only cosmetically.
    $clients = @(
        @{ Name    = 'GoogleDriveFS'
           Process = 'GoogleDriveFS'
           Path    = 'C:\Program Files\Google\Drive File Stream\130.0.2.0\GoogleDriveFS.exe'
           Args    = '--startup_mode'
           Glob    = 'C:\Program Files\Google\Drive File Stream\*\GoogleDriveFS.exe' },
        @{ Name    = 'Dropbox'
           Process = 'Dropbox'
           Path    = 'C:\Program Files (x86)\Dropbox\Client\Dropbox.exe'
           Args    = '/systemstartup'
           Glob    = 'C:\Program Files (x86)\Dropbox\Client\Dropbox.exe' }
    )

    # Cloud sync clients refuse to run elevated (Google Drive exits immediately, and
    # an elevated Dropbox would own the sync root as Administrator). The scheduled
    # task runs Limited; this guard catches a manual run from an elevated shell.
    $isElevated = (New-Object Security.Principal.WindowsPrincipal(
        [Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    if ($isElevated -and -not $AllowElevated) {
        Write-Log 'Running ELEVATED. Cloud clients must start unelevated - Google Drive exits on launch.' 'ERROR'
        Write-Log 'Use the CloudClients-AfterVHDX scheduled task, or pass -AllowElevated to override.' 'ERROR'
        return 4
    }

    $expectedImage = [System.IO.Path]::GetFullPath($ExpectedVhdPath).TrimEnd('\')
    $bootTime = $null
    if ($RequireMountLog) {
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
        if ($null -ne $os -and $os.PSObject.Properties.Name -contains 'LastBootUpTime') {
            $bootTime = $os.LastBootUpTime
        }
        if (-not $bootTime) {
            Write-Log 'Cannot prove receipt freshness without boot time; clients NOT started.' 'ERROR'
            return 3
        }
    }

    Write-Log "Waiting for $($ExpectedDriveLetter): NTFS label '$ExpectedLabel' on '$expectedImage' (max $($TimeoutSeconds)s)"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $vol = $null
    $pending = 'Expected image/volume not ready.'
    $lastDiagnostic = $null
    $receiptReady = -not $RequireMountLog
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        # Poll the receipt in the same loop as the volume. A previous-boot receipt
        # does not prove readiness, but neither should it defeat a logon race.
        if ($RequireMountLog) {
            $receiptReady = $false
            $newest = Get-ChildItem -LiteralPath $MountLogRoot -Filter '*.result.json' -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($newest) {
                $fatalReceipt = $false
                try {
                    $res = Get-Content -LiteralPath $newest.FullName -Raw | ConvertFrom-Json
                    foreach ($field in @('StartedAt', 'ExitCode', 'Status', 'VhdPath')) {
                        if ($null -eq $res -or $res.PSObject.Properties.Name -notcontains $field) {
                            throw "Mount receipt is incomplete: missing $field."
                        }
                    }
                    $mountStartedAt = [datetime]$res.StartedAt
                    if ($mountStartedAt -gt (Get-Date)) {
                        $fatalReceipt = $true
                        throw 'Receipt StartedAt is in the future.'
                    }
                    if ($mountStartedAt -ge $bootTime) {
                        if ($null -eq $res.ExitCode -or $res.ExitCode -ne 0 -or $res.Status -ne 'Success') {
                            $fatalReceipt = $true
                            throw 'Fresh mount receipt does not report Success and ExitCode 0.'
                        }
                        if ([System.IO.Path]::GetFullPath($res.VhdPath).TrimEnd('\') -ne $expectedImage) {
                            $fatalReceipt = $true
                            throw 'Mount receipt belongs to a different VHDX image.'
                        }
                        $receiptReady = $true
                    } else {
                        $pending = 'Waiting for a receipt from this boot.'
                    }
                } catch {
                    if ($fatalReceipt) {
                        Write-Log "Failed mount receipt '$($newest.Name)': $($_.Exception.Message); clients NOT started." 'ERROR'
                        return 3
                    }
                    # The mount producer writes JSON directly to its final path.
                    # A partial/read-locked receipt remains pending; it grants
                    # no readiness and cannot extend the original deadline.
                    $pending = "Waiting for a complete mount receipt: $($_.Exception.Message)"
                }
            } else {
                $pending = 'Waiting for the mount receipt.'
            }
        }

        $vol = $null
        try {
            # Try the Storage image/disk association first. Some Limited-session
            # providers cannot resolve it; the exact-volume receipt fallback below
            # remains bound to this boot's successful expected-image mount.
            $images = @(Get-DiskImage -ImagePath $expectedImage -ErrorAction Stop)
            if ($images.Count -gt 1) {
                Write-Log 'Expected image has ambiguous returned identities; clients NOT started.' 'ERROR'
                return 2
            }
            if ($images.Count -eq 0) { throw 'Expected image metadata is unavailable.' }
            $image = $images[0]
            if ($image.Attached -and [System.IO.Path]::GetFullPath($image.ImagePath).TrimEnd('\') -eq $expectedImage) {
                # Resolve the actual nullable image association through the same
                # parser used by cache repair. The child scope preserves this
                # launcher's parameters when dot-sourcing the declaration-only
                # helper; it never invokes repair or another task.
                try {
                    $imageNumber = & {
                        param($InputImage, $RepairSource)
                        . $RepairSource
                        Get-CacheImageDiskNumber $InputImage
                    } $image (Join-Path (Split-Path $script:CloudClientGateSource -Parent) 'Repair-CloudCacheMount.ps1')
                } catch {
                    Write-Log "Image disk identity refused: $($_.Exception.Message); clients NOT started." 'ERROR'
                    return 2
                }
                if ($null -eq $imageNumber) { throw 'Expected image disk identity is incomplete.' }
                $disk = @(Get-Disk -Number $imageNumber -ErrorAction Stop)
                # A query failure or empty metadata can use the privileged
                # receipt fallback. Observed contradictory identities cannot.
                if ($disk.Count -gt 1 -or ($disk.Count -eq 1 -and $null -ne $disk[0].Number -and $disk[0].Number -ne $imageNumber)) {
                    Write-Log 'Expected image resolved to contradictory or ambiguous disk identities; clients NOT started.' 'ERROR'
                    return 2
                }
                if ($disk.Count -eq 0 -or $null -eq $disk[0].Number) { throw 'Expected image disk metadata is unavailable or incomplete.' }
                $partitions = @(Get-Partition -DiskNumber $disk[0].Number -ErrorAction Stop)
                if (@($partitions | Where-Object { $null -ne $_.DiskNumber -and $_.DiskNumber -ne $disk[0].Number }).Count -gt 0) {
                    Write-Log 'Returned partition identity contradicts the expected image disk; clients NOT started.' 'ERROR'
                    return 2
                }
                $partition = @($partitions | Where-Object { [string]$_.DriveLetter -eq $ExpectedDriveLetter })
                if ($partition.Count -eq 0) {
                    $pending = "Expected image has no partition at $($ExpectedDriveLetter):."
                } elseif ($partition.Count -ne 1) {
                    Write-Log 'Expected drive letter has ambiguous partition identities; clients NOT started.' 'ERROR'
                    return 2
                } else {
                    if ($null -eq $partition[0].DiskNumber) { throw 'Expected partition disk metadata is incomplete.' }
                    $candidate = @(Get-Volume -Partition $partition[0] -ErrorAction Stop)
                    if ($candidate.Count -ne 1 -or [string]$candidate[0].DriveLetter -ne $ExpectedDriveLetter -or
                        $candidate[0].FileSystemLabel -ne $ExpectedLabel -or $candidate[0].FileSystem -ne 'NTFS') {
                        Write-Log 'Expected image volume has the wrong drive letter, label or filesystem; clients NOT started.' 'ERROR'
                        return 2
                    }
                    if (Test-Path -LiteralPath ('{0}:\' -f $ExpectedDriveLetter)) {
                        $vol = $candidate[0]
                    }
                }
            }
        } catch {
            $pending = "Image/volume validation pending: $($_.Exception.Message)"
            if ($RequireMountLog -and $receiptReady) {
                try {
                    # Limited sessions may not query the Storage image/disk
                    # association. Use the privileged mount receipt's exact
                    # volume GUID, not its label alone, with live WMI identity.
                    $receiptVolume = @($res.Volumes | Where-Object {
                        [string]$_.DriveLetter -eq $ExpectedDriveLetter -and
                        $_.FileSystemLabel -eq $ExpectedLabel -and $_.FileSystem -eq 'NTFS'
                    })
                    if ($receiptVolume.Count -ne 1 -or
                        $receiptVolume[0].Path -notmatch '^\\\\\?\\Volume\{[0-9a-fA-F-]{36}\}\\$') {
                        throw 'Fresh receipt lacks one exact expected volume GUID.'
                    }
                    $liveVolume = @(Get-CimInstance -ClassName Win32_Volume -Filter ("DriveLetter='{0}:'" -f $ExpectedDriveLetter) -ErrorAction Stop)
                    if ($liveVolume.Count -ne 1 -or
                        $liveVolume[0].DeviceID -ne $receiptVolume[0].Path -or
                        $liveVolume[0].DriveLetter -ne ("{0}:" -f $ExpectedDriveLetter) -or
                        $liveVolume[0].Label -ne $ExpectedLabel -or $liveVolume[0].FileSystem -ne 'NTFS') {
                        throw 'Live volume does not match the fresh expected-image receipt identity.'
                    }
                    if (Test-Path -LiteralPath ('{0}:\' -f $ExpectedDriveLetter)) {
                        $vol = $receiptVolume[0]
                        Write-Log 'Validated live volume GUID against the fresh expected-image mount receipt.'
                    }
                } catch {
                    $pending = "Receipt/live-volume validation pending: $($_.Exception.Message)"
                }
            }
            if (-not $vol -and $pending -ne $lastDiagnostic) {
                Write-Log $pending 'WARN'
                $lastDiagnostic = $pending
            }
        }
        if ($sw.Elapsed.TotalSeconds -ge $TimeoutSeconds) {
            $vol = $null
            break
        }
        if ($vol -and $receiptReady) { break }
        $remainingMs = [math]::Ceiling(($TimeoutSeconds - $sw.Elapsed.TotalSeconds) * 1000)
        if ($remainingMs -gt 0) {
            Start-Sleep -Milliseconds ([int][math]::Min($PollSeconds * 1000, $remainingMs))
        }
    }
    $sw.Stop()
    if (-not $vol -or -not $receiptReady) {
        Write-Log "Gate deadline expired after $($TimeoutSeconds)s. $pending Clients NOT started." 'ERROR'
        if (-not $receiptReady) { return 3 }
        return 2
    }
    Write-Log ("Validated {0}: on expected VHDX after {1:N1}s." -f $ExpectedDriveLetter, $sw.Elapsed.TotalSeconds)

    $started = 0; $skipped = 0; $failed = 0
    foreach ($c in $clients) {
        if (Get-Process -Name $c.Process -ErrorAction SilentlyContinue) {
            Write-Log "$($c.Name): already running - skip"
            $skipped++
            continue
        }
        $exe = $c.Path
        if (-not (Test-Path $exe)) {
            # version-numbered install dirs move on update; fall back to newest match
            $cand = Get-ChildItem $c.Glob -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($cand) { $exe = $cand.FullName; Write-Log "$($c.Name): resolved to $exe" }
            else { Write-Log "$($c.Name): executable not found - skip" 'WARN'; $failed++; continue }
        }
        if ($PSCmdlet.ShouldProcess($c.Name, "Start $exe $($c.Args)")) {
            try {
                Start-Process -FilePath $exe -ArgumentList $c.Args -WindowStyle Hidden -ErrorAction Stop
                Write-Log "$($c.Name): started"
                $started++
            } catch {
                Write-Log "$($c.Name): FAILED to start - $($_.Exception.Message)" 'ERROR'
                $failed++
            }
        }
    }

    Write-Log "Done. started=$started skipped=$skipped failed=$failed"
    if ($failed -gt 0) { return 1 }
    return 0
}

if ($MyInvocation.InvocationName -ne '.') {
    exit (Invoke-CloudClientGate @PSBoundParameters)
}
