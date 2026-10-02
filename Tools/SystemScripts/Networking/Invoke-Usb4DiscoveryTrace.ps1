<#
.SYNOPSIS
Capture bounded USB4 router discovery telemetry on Windows.
.DESCRIPTION
Plans by default. Apply in elevated PowerShell to collect only the official USB4
HostRouter and DeviceRouter TraceLogging providers. Output stays in the supplied
report directory. Each invocation owns a unique collector and filenames; existing
files are never overwritten. DryRun and WhatIf create no files, native processes,
trace sessions or sleeps. This captures router metadata, not network traffic.
.EXAMPLE
./Invoke-Usb4DiscoveryTrace.ps1 -ReportDirectory C:/Reports/usb4 -DryRun
.EXAMPLE
./Invoke-Usb4DiscoveryTrace.ps1 -ReportDirectory C:/Reports/usb4 -DurationSeconds 5 -Apply
.LINK
https://learn.microsoft.com/en-us/windows-hardware/design/component-guidelines/usb4-tracelogging-rundown-events
.LINK
https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/logman-create-trace
.LINK
https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/logman-update-trace
#>
#Requires -Version 7.0
[CmdletBinding(SupportsShouldProcess, PositionalBinding = $false)]
param(
    [string]$ReportDirectory,
    [ValidateRange(1, 60)][int]$DurationSeconds = 5,
    [ValidateRange(32, 128)][int]$MaxMegabytes = 64,
    [switch]$Apply,
    [switch]$DryRun,
    [Alias('h', 'help')][switch]$ShowHelp,
    [Parameter(ValueFromRemainingArguments)][string[]]$RemainingArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Test-Usb4TraceAdministrator {
    [CmdletBinding()]
    param()
    if (-not $IsWindows) { return $false }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    try {
        $principal = [Security.Principal.WindowsPrincipal]::new($identity)
        return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    }
    finally { $identity.Dispose() }
}

function Invoke-Usb4TraceNative {
    [CmdletBinding()]
    param([string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory, [int]$TimeoutSeconds = 30)
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $FilePath
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($argument in $Arguments) { $startInfo.ArgumentList.Add($argument) }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $started = $false
    try {
        $started = $process.Start()
        if (-not $started) { throw "Could not start $FilePath." }
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            $process.Kill($true)
            $process.WaitForExit()
            return [pscustomobject]@{ ExitCode = -1; StandardOutput = $stdout.GetAwaiter().GetResult();
                StandardError = "$FilePath timed out after $TimeoutSeconds seconds. " + $stderr.GetAwaiter().GetResult()
            }
        }
        return [pscustomobject]@{ ExitCode = $process.ExitCode; StandardOutput = $stdout.GetAwaiter().GetResult();
            StandardError = $stderr.GetAwaiter().GetResult()
        }
    }
    catch {
        return [pscustomobject]@{ ExitCode = -1; StandardOutput = ''; StandardError = $_.Exception.Message }
    }
    finally {
        if ($started -and -not $process.HasExited) { $process.Kill($true) }
        $process.Dispose()
    }
}

function Invoke-Usb4TraceCommand {
    [CmdletBinding()]
    param([object]$Command, [string]$Directory, [string]$LogPath)
    $result = Invoke-Usb4TraceNative -FilePath $Command.FilePath -Arguments $Command.Arguments -WorkingDirectory $Directory
    [ordered]@{ TimestampUtc = [DateTime]::UtcNow.ToString('o'); FilePath = $Command.FilePath;
        Arguments = $Command.Arguments; ExitCode = $result.ExitCode; StandardOutput = $result.StandardOutput;
        StandardError = $result.StandardError
    } | ConvertTo-Json -Compress -Depth 4 | Add-Content -LiteralPath $LogPath -Encoding utf8
    if ($result.ExitCode -ne 0) { throw "$($Command.FilePath) exited $($result.ExitCode): $($result.StandardError) $($result.StandardOutput)" }
    return $result
}

function Invoke-Usb4DiscoveryTraceMain {
    [CmdletBinding(SupportsShouldProcess)]
    param([string]$Directory, [ValidateRange(1, 60)][int]$Seconds = 5,
        [ValidateRange(32, 128)][int]$Megabytes = 64, [switch]$EnableApply, [switch]$IsDryRun)
    $id = [Guid]::NewGuid().ToString('N')
    $name = "pcai-usb4-$id"
    $directoryPath = $(if ($Directory) { [IO.Path]::GetFullPath($Directory) } else { $null })
    $providerLines = @(
        '{575BA31F-2B45-58C2-64FD-F5DC757B6137} 0xffffffffffffffff 5',
        '{AE795D36-2B11-5EFB-C7E0-5D552BC55D6C} 0xffffffffffffffff 5'
    )
    $paths = [ordered]@{}
    foreach ($entry in @(
            @('Etl', '.etl'), @('Xml', '.xml'), @('Providers', '.providers.txt'),
            @('NativeLog', '.native.jsonl'), @('Summary', '.summary.txt'), @('Report', '.report.xml'))) {
        $paths[$entry[0]] = $(if ($directoryPath) { Join-Path $directoryPath ($name + $entry[1]) } else { $name + $entry[1] })
    }
    # The watchdog bounds orphan capture if the caller is interrupted. Normal capture
    # stops after Seconds; the extra ten seconds allow normal stop without a race.
    $watchdog = [TimeSpan]::FromSeconds($Seconds + 10).ToString('hh\:mm\:ss')
    $commands = [ordered]@{
        Create = [pscustomobject]@{ FilePath = 'logman.exe'; Arguments = @('create', 'trace', $name,
                '-o', $paths.Etl, '-f', 'bincirc', '-max', "$Megabytes", '-rf', $watchdog,
                '-p', '{575BA31F-2B45-58C2-64FD-F5DC757B6137}', '0xffffffffffffffff', '5')
        }
        Update = [pscustomobject]@{ FilePath = 'logman.exe'; Arguments = @('update', 'trace', $name, '-pf', $paths.Providers) }
        Start  = [pscustomobject]@{ FilePath = 'logman.exe'; Arguments = @('start', $name) }
        Stop   = [pscustomobject]@{ FilePath = 'logman.exe'; Arguments = @('stop', $name) }
        Delete = [pscustomobject]@{ FilePath = 'logman.exe'; Arguments = @('delete', $name) }
        Decode = [pscustomobject]@{ FilePath = 'tracerpt.exe'; Arguments = @($paths.Etl, '-o', $paths.Xml, '-of', 'XML',
                '-summary', $paths.Summary, '-report', $paths.Report)
        }
    }
    $metadata = [ordered]@{ Applied = $false; State = 'Planned'; CollectorName = $name;
        DurationSeconds = $Seconds; MaxMegabytes = $Megabytes; ReportDirectory = $directoryPath;
        Paths = [pscustomobject]$paths; Providers = $providerLines; Commands = [pscustomobject]$commands
    }
    if (-not $EnableApply -or $IsDryRun) { return [pscustomobject]$metadata }
    if (-not $PSCmdlet.ShouldProcess($directoryPath, "Capture $Seconds seconds of USB4 router discovery metadata")) {
        return [pscustomobject]$metadata
    }
    if (-not $directoryPath) { throw 'Apply requires an explicit ReportDirectory.' }
    if (-not (Test-Usb4TraceAdministrator)) { throw 'Apply requires elevated Windows PowerShell.' }
    if (Test-Path -LiteralPath $directoryPath -PathType Leaf) { throw 'ReportDirectory is a file.' }
    foreach ($path in $paths.Values) {
        if (Test-Path -LiteralPath $path) { throw "Refusing to overwrite existing artifact $path." }
    }
    $null = New-Item -ItemType Directory -Path $directoryPath -Force
    $log = [IO.File]::Open($paths.NativeLog, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $log.Dispose()
    $providerFile = [IO.File]::Open($paths.Providers, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $providerWriter = [IO.StreamWriter]::new($providerFile, [Text.UTF8Encoding]::new($false))
    try { foreach ($line in $providerLines) { $providerWriter.WriteLine($line) } } finally { $providerWriter.Dispose() }
    $createAttempted = $false
    $startAttempted = $false
    $failure = $null
    $cleanupErrors = [Collections.Generic.List[string]]::new()
    try {
        $createAttempted = $true
        $null = Invoke-Usb4TraceCommand -Command $commands.Create -Directory $directoryPath -LogPath $paths.NativeLog
        $null = Invoke-Usb4TraceCommand -Command $commands.Update -Directory $directoryPath -LogPath $paths.NativeLog
        $startAttempted = $true
        $null = Invoke-Usb4TraceCommand -Command $commands.Start -Directory $directoryPath -LogPath $paths.NativeLog
        Start-Sleep -Seconds $Seconds
    }
    catch { $failure = $_.Exception.Message } finally {
        # The unique name is generated internally; attempt deletion even if create
        # returned an error after partially creating its collector.
        if ($createAttempted) {
            if ($startAttempted) {
                try { $null = Invoke-Usb4TraceCommand -Command $commands.Stop -Directory $directoryPath -LogPath $paths.NativeLog }
                catch { $cleanupErrors.Add($_.Exception.Message) }
            }
            try { $null = Invoke-Usb4TraceCommand -Command $commands.Delete -Directory $directoryPath -LogPath $paths.NativeLog }
            catch { $cleanupErrors.Add($_.Exception.Message) }
        }
    }
    if ($failure -or $cleanupErrors.Count) {
        throw "USB4 trace failed. $failure $($cleanupErrors -join ' ') Native log: $($paths.NativeLog)"
    }
    # Saved logman collectors may append their numeric file version even without
    # an explicit -v option. Resolve only this invocation's files in its directory.
    $pattern = '^' + [Regex]::Escape($name) + '(?:_[0-9]+)?\.etl$'
    $capturedFiles = @(Get-ChildItem -LiteralPath $directoryPath -Filter "$name*.etl" -File |
            Where-Object { $_.Name -match $pattern -and $_.Length -gt 0 })
    if ($capturedFiles.Count -ne 1) {
        throw "Capture requires exactly one nonempty ETL matching $name with an optional numeric suffix; found $($capturedFiles.Count)."
    }
    $paths.Etl = $capturedFiles[0].FullName
    $metadata.Paths.Etl = $paths.Etl
    $commands.Decode.Arguments[0] = $paths.Etl
    $null = Invoke-Usb4TraceCommand -Command $commands.Decode -Directory $directoryPath -LogPath $paths.NativeLog
    if (-not (Test-Path -LiteralPath $paths.Xml -PathType Leaf) -or (Get-Item -LiteralPath $paths.Xml).Length -eq 0) {
        throw 'tracerpt produced no nonempty XML.'
    }
    $metadata.Applied = $true
    $metadata.State = 'Captured'
    return [pscustomobject]$metadata
}

if ($MyInvocation.InvocationName -ne '.') {
    if ($ShowHelp -or '--help' -in $RemainingArguments) { Get-Help $PSCommandPath -Detailed; return }
    if ($RemainingArguments -and $RemainingArguments.Count) { throw "Unknown arguments: $($RemainingArguments -join ' ')." }
    Invoke-Usb4DiscoveryTraceMain -Directory $ReportDirectory -Seconds $DurationSeconds -Megabytes $MaxMegabytes `
        -EnableApply:$Apply -IsDryRun:$DryRun -WhatIf:$WhatIfPreference
}
