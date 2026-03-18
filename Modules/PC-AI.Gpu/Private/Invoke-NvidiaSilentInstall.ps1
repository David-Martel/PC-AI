#Requires -Version 5.1

function Invoke-NvidiaSilentInstall {
<#
.SYNOPSIS
    Runs an NVIDIA component installer silently and captures its outcome.

.DESCRIPTION
    Supports three installer types, auto-detected from the file extension:

        .exe  - Executed directly with the supplied -SilentArgs.
        .msi  - Delegated to msiexec.exe with /i /qb /norestart.
        .zip  - Extracted to a caller-supplied or auto-derived target directory.

    Requires Administrator elevation. Logs stdout/stderr to a timestamped file
    under .pcai\nvidia-install-logs\ in the PC_AI repository root.

    Exit code semantics:
        0    - Success.
        3010 - Success; a system restart is required to complete the install.
        Any other non-zero code is treated as failure.

    Supports -WhatIf: when specified the installer command is written to the
    verbose stream but not executed.

.PARAMETER InstallerPath
    Full path to the downloaded NVIDIA installer file (.exe, .msi, or .zip).

.PARAMETER SilentArgs
    Arguments appended when running an .exe installer. Defaults to the
    canonical NVIDIA silent flags: /s /n /noreboot.
    Ignored for .msi and .zip installer types.

.PARAMETER ComponentId
    Identifier string included in the log file name so log files from multiple
    components are easy to distinguish (e.g. 'cuda-toolkit', 'gpu-driver').
    When omitted the base name of InstallerPath is used.

.PARAMETER TimeoutSeconds
    Maximum seconds to wait for the installer process to exit. Default: 600.

.OUTPUTS
    [PSCustomObject] with properties:
        Success          - $true when exit code indicates success.
        ExitCode         - Raw process exit code, or -1 on timeout/error.
        RebootRequired   - $true when exit code is 3010.
        LogPath          - Full path of the captured installer log file.
        InstallerPath    - The installer path that was passed in.
        Duration         - [TimeSpan] elapsed wall-clock time.

.EXAMPLE
    Invoke-NvidiaSilentInstall -InstallerPath 'C:\Temp\cuda_13.1_win.exe' -ComponentId 'cuda-toolkit'

.EXAMPLE
    Invoke-NvidiaSilentInstall -InstallerPath 'C:\Temp\cudnn_9.exe' -WhatIf

.EXAMPLE
    Invoke-NvidiaSilentInstall -InstallerPath 'C:\Temp\driver.exe' `
        -SilentArgs @('/s', '/n', '/noreboot') -TimeoutSeconds 900

.NOTES
    Phase 3 implementation.
    The log directory is created lazily on first use. Log files are never
    deleted by this function — callers are responsible for log rotation.
    For .zip installers, the content is extracted to a sibling directory
    named <zip-basename>-extracted beside the zip file unless -TargetDir
    is specified (future extension point).
#>
    [CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [ValidateNotNullOrEmpty()]
        [string]$InstallerPath,

        [Parameter()]
        [string[]]$SilentArgs = @('/s', '/n', '/noreboot'),

        [Parameter()]
        [string]$ComponentId,

        [Parameter()]
        [ValidateRange(30, 7200)]
        [int]$TimeoutSeconds = 600
    )

    $ErrorActionPreference = 'Stop'

    # --- Build result skeleton ---
    $result = [PSCustomObject]@{
        Success        = $false
        ExitCode       = -1
        RebootRequired = $false
        LogPath        = $null
        InstallerPath  = $InstallerPath
        Duration       = [TimeSpan]::Zero
    }

    # --- Admin elevation check (pattern from Test-AdminElevation) ---
    $identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Invoke-NvidiaSilentInstall requires Administrator elevation.'
    }

    # --- Validate installer path ---
    if (-not (Test-Path -LiteralPath $InstallerPath)) {
        throw "Installer not found: $InstallerPath"
    }

    $installerItem = Get-Item -LiteralPath $InstallerPath
    $extension     = $installerItem.Extension.ToLower()

    if ($extension -notin @('.exe', '.msi', '.zip')) {
        throw "Unsupported installer type '$extension'. Expected .exe, .msi, or .zip."
    }

    # --- Resolve component label for log filename ---
    if (-not $ComponentId) {
        $ComponentId = $installerItem.BaseName
    }

    # --- Prepare log directory ---
    # Derive PC_AI root from this script's location:
    #   Private\ -> PC-AI.Gpu\ -> Modules\ -> PC_AI\
    $modulePrivateDir = $PSScriptRoot
    $moduleDir        = Split-Path $modulePrivateDir -Parent
    $modulesDir       = Split-Path $moduleDir -Parent
    $pcAiRoot         = Split-Path $modulesDir -Parent

    $logDir = Join-Path $pcAiRoot '.pcai\nvidia-install-logs'
    if (-not (Test-Path -LiteralPath $logDir)) {
        New-Item -Path $logDir -ItemType Directory -Force | Out-Null
        Write-Verbose "Created log directory: $logDir"
    }

    $timestamp = (Get-Date -Format 'yyyyMMdd-HHmmss')
    $logFile   = Join-Path $logDir "$($ComponentId)-$timestamp.log"
    $result.LogPath = $logFile

    # --- WhatIf guard ---
    if (-not $PSCmdlet.ShouldProcess($InstallerPath, "Run silent NVIDIA installer ($extension)")) {
        Write-Verbose "WhatIf: Would run silent install for '$InstallerPath' (type: $extension)"
        return $result
    }

    # --- Execute installer ---
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    try {
        switch ($extension) {
            '.exe' {
                # Ensure each argument is quoted if it contains spaces
                $escapedArgs = @($SilentArgs | ForEach-Object {
                    if ($_ -match ' ') { "`"$_`"" } else { $_ }
                })
                $argList = $escapedArgs -join ' '
                Write-Verbose "Invoke-NvidiaSilentInstall: Launching EXE '$InstallerPath' with args: $argList"

                $psi                        = New-Object System.Diagnostics.ProcessStartInfo
                $psi.FileName               = $InstallerPath
                $psi.Arguments              = $argList
                $psi.RedirectStandardOutput = $true
                $psi.RedirectStandardError  = $true
                $psi.UseShellExecute        = $false
                $psi.CreateNoWindow         = $true

                $proc = New-Object System.Diagnostics.Process
                $proc.StartInfo = $psi

                # Accumulate output asynchronously to avoid deadlock on full buffers
                $stdoutLines = [System.Collections.Generic.List[string]]::new()
                $stderrLines = [System.Collections.Generic.List[string]]::new()

                $proc.add_OutputDataReceived({ param($s, $e); if ($null -ne $e.Data) { $stdoutLines.Add($e.Data) } })
                $proc.add_ErrorDataReceived({  param($s, $e); if ($null -ne $e.Data) { $stderrLines.Add($e.Data) } })

                try {
                    [void]$proc.Start()
                    $proc.BeginOutputReadLine()
                    $proc.BeginErrorReadLine()

                    $timeoutMs = $TimeoutSeconds * 1000
                    $exited    = $proc.WaitForExit($timeoutMs)

                    if (-not $exited) {
                        try { $proc.Kill() } catch { }
                        throw "Installer timed out after $TimeoutSeconds seconds."
                    }

                    $result.ExitCode = $proc.ExitCode
                }
                finally {
                    $proc.Dispose()
                }

                # Write captured output to log
                $logContent = [System.Text.StringBuilder]::new()
                [void]$logContent.AppendLine("=== NVIDIA Silent Install Log ===")
                [void]$logContent.AppendLine("Component : $ComponentId")
                [void]$logContent.AppendLine("Installer : $InstallerPath")
                [void]$logContent.AppendLine("Args      : $argList")
                [void]$logContent.AppendLine("Started   : $timestamp")
                [void]$logContent.AppendLine("ExitCode  : $($result.ExitCode)")
                [void]$logContent.AppendLine("")
                [void]$logContent.AppendLine("--- STDOUT ---")
                foreach ($line in $stdoutLines) { [void]$logContent.AppendLine($line) }
                [void]$logContent.AppendLine("")
                [void]$logContent.AppendLine("--- STDERR ---")
                foreach ($line in $stderrLines) { [void]$logContent.AppendLine($line) }
                [System.IO.File]::WriteAllText($logFile, $logContent.ToString())
            }

            '.msi' {
                $msiArgs = "/i `"$InstallerPath`" /qb /norestart /l*v `"$logFile`""
                Write-Verbose "Invoke-NvidiaSilentInstall: Launching MSI via msiexec.exe with args: $msiArgs"

                $psi               = New-Object System.Diagnostics.ProcessStartInfo
                $psi.FileName      = 'msiexec.exe'
                $psi.Arguments     = $msiArgs
                $psi.UseShellExecute = $false
                $psi.CreateNoWindow  = $true

                $proc = New-Object System.Diagnostics.Process
                $proc.StartInfo = $psi

                try {
                    [void]$proc.Start()

                    $timeoutMs = $TimeoutSeconds * 1000
                    $exited    = $proc.WaitForExit($timeoutMs)

                    if (-not $exited) {
                        try { $proc.Kill() } catch { }
                        throw "MSI installer timed out after $TimeoutSeconds seconds."
                    }

                    $result.ExitCode = $proc.ExitCode
                }
                finally {
                    $proc.Dispose()
                }

                # msiexec writes its own verbose log via /l*v; append a summary header
                $header = "=== NVIDIA Silent MSI Install ===" + [Environment]::NewLine +
                          "Component : $ComponentId" + [Environment]::NewLine +
                          "Installer : $InstallerPath" + [Environment]::NewLine +
                          "Started   : $timestamp" + [Environment]::NewLine +
                          "ExitCode  : $($result.ExitCode)" + [Environment]::NewLine
                if (Test-Path -LiteralPath $logFile) {
                    $existing = [System.IO.File]::ReadAllText($logFile)
                    [System.IO.File]::WriteAllText($logFile, $header + $existing)
                }
                else {
                    [System.IO.File]::WriteAllText($logFile, $header)
                }
            }

            '.zip' {
                $targetDir = Join-Path (Split-Path $InstallerPath -Parent) "$($installerItem.BaseName)-extracted"
                Write-Verbose "Invoke-NvidiaSilentInstall: Extracting ZIP '$InstallerPath' to '$targetDir'"

                if (-not (Test-Path -LiteralPath $targetDir)) {
                    New-Item -Path $targetDir -ItemType Directory -Force | Out-Null
                }

                Add-Type -AssemblyName System.IO.Compression.FileSystem
                [System.IO.Compression.ZipFile]::ExtractToDirectory($InstallerPath, $targetDir)

                $result.ExitCode = 0

                $logContent = "=== NVIDIA ZIP Extraction Log ===" + [Environment]::NewLine +
                              "Component : $ComponentId" + [Environment]::NewLine +
                              "Archive   : $InstallerPath" + [Environment]::NewLine +
                              "Target    : $targetDir" + [Environment]::NewLine +
                              "Started   : $timestamp" + [Environment]::NewLine +
                              "ExitCode  : 0 (extraction succeeded)" + [Environment]::NewLine
                [System.IO.File]::WriteAllText($logFile, $logContent)
            }
        }
    }
    catch {
        $stopwatch.Stop()
        $result.Duration = $stopwatch.Elapsed

        # Append error to log if file already exists, otherwise create it
        $errorLine = "FATAL ERROR: $($_.Exception.Message)"
        if (Test-Path -LiteralPath $logFile) {
            [System.IO.File]::AppendAllText($logFile, [Environment]::NewLine + $errorLine + [Environment]::NewLine)
        }
        else {
            [System.IO.File]::WriteAllText($logFile, $errorLine + [Environment]::NewLine)
        }

        Write-Error "Invoke-NvidiaSilentInstall: Installer execution failed: $($_.Exception.Message)"
        return $result
    }

    $stopwatch.Stop()
    $result.Duration = $stopwatch.Elapsed

    # --- Interpret exit code ---
    switch ($result.ExitCode) {
        0    {
            $result.Success        = $true
            $result.RebootRequired = $false
            Write-Verbose "Invoke-NvidiaSilentInstall: Installation succeeded (exit 0)."
        }
        3010 {
            $result.Success        = $true
            $result.RebootRequired = $true
            Write-Verbose "Invoke-NvidiaSilentInstall: Installation succeeded; reboot required (exit 3010)."
        }
        default {
            $result.Success = $false
            Write-Warning "Invoke-NvidiaSilentInstall: Installer exited with code $($result.ExitCode). See log: $logFile"
        }
    }

    Write-Verbose "Invoke-NvidiaSilentInstall: Duration $($result.Duration.TotalSeconds.ToString('F1'))s. Log: $logFile"
    return $result
}
