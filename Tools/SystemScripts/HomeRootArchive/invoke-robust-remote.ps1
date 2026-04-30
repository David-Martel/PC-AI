# invoke-robust-remote.ps1
# Robust remote command execution with automatic fallback

param(
    [Parameter(Mandatory=$true)]
    [string]$Command,

    [string]$Server = "10.10.20.214",
    [pscredential]$Credential,
    [switch]$ShowDiagnostics
)

if (-not $Credential) {
    $Credential = Get-Credential -Message "Enter credentials for $Server"
}

Write-Host "=== Robust Remote Command Execution ===" -ForegroundColor Cyan
Write-Host "Server: $Server" -ForegroundColor Yellow
Write-Host "Command: $Command`n" -ForegroundColor Yellow

$methodsAttempted = @()
$result = $null
$success = $false

# Method 1: PowerShell Remoting (WinRM)
Write-Host "[Method 1/4] Attempting PowerShell Remoting (WinRM)..." -ForegroundColor Cyan
try {
    $startTime = Get-Date

    $result = Invoke-Command -ComputerName $Server -ScriptBlock {
        param($cmd)
        Invoke-Expression $cmd 2>&1 | Out-String
    } -ArgumentList $Command -Credential $Credential -ErrorAction Stop

    $duration = ((Get-Date) - $startTime).TotalMilliseconds

    Write-Host "  SUCCESS via WinRM ($([math]::Round($duration, 0)) ms)" -ForegroundColor Green
    $methodsAttempted += @{Method="WinRM"; Status="Success"; Duration=$duration}
    $success = $true
} catch {
    $errorMsg = $_.Exception.Message
    Write-Host "  FAILED: $errorMsg" -ForegroundColor Red
    $methodsAttempted += @{Method="WinRM"; Status="Failed"; Error=$errorMsg}

    if ($ShowDiagnostics) {
        Write-Host "    Diagnostics: Checking WinRM availability..." -ForegroundColor Gray
        $winrmTest = Test-NetConnection -ComputerName $Server -Port 5985 -InformationLevel Quiet
        Write-Host "    Port 5985 accessible: $winrmTest" -ForegroundColor Gray
    }
}

# Method 2: PsExec (if Method 1 failed)
if (-not $success) {
    Write-Host "`n[Method 2/4] Attempting PsExec..." -ForegroundColor Cyan

    $psexecPath = "C:\Tools\PsExec64.exe"
    if (-not (Test-Path $psexecPath)) {
        $psexecPath = "$env:TEMP\PsExec64.exe"

        if (-not (Test-Path $psexecPath)) {
            Write-Host "  Downloading PsExec..." -ForegroundColor Yellow
            try {
                Invoke-WebRequest -Uri "https://live.sysinternals.com/PsExec64.exe" -OutFile $psexecPath -UseBasicParsing
            } catch {
                Write-Host "  FAILED: Cannot download PsExec" -ForegroundColor Red
                $methodsAttempted += @{Method="PsExec"; Status="Failed"; Error="Download failed"}
            }
        }
    }

    if (Test-Path $psexecPath) {
        try {
            $startTime = Get-Date

            $username = $Credential.UserName
            $password = $Credential.GetNetworkCredential().Password

            # Create a script block to execute
            $scriptBlock = "powershell.exe -Command `"$Command`""

            $result = & $psexecPath -accepteula \\$Server -u $username -p $password $scriptBlock 2>&1

            $duration = ((Get-Date) - $startTime).TotalMilliseconds

            if ($LASTEXITCODE -eq 0) {
                Write-Host "  SUCCESS via PsExec ($([math]::Round($duration, 0)) ms)" -ForegroundColor Green
                $methodsAttempted += @{Method="PsExec"; Status="Success"; Duration=$duration}
                $success = $true
            } else {
                Write-Host "  FAILED: PsExec exit code $LASTEXITCODE" -ForegroundColor Red
                $methodsAttempted += @{Method="PsExec"; Status="Failed"; Error="Exit code $LASTEXITCODE"}
            }
        } catch {
            Write-Host "  FAILED: $($_.Exception.Message)" -ForegroundColor Red
            $methodsAttempted += @{Method="PsExec"; Status="Failed"; Error=$_.Exception.Message}
        }
    }
}

# Method 3: WMI/CIM (if previous methods failed)
if (-not $success) {
    Write-Host "`n[Method 3/4] Attempting WMI/CIM..." -ForegroundColor Cyan

    try {
        $startTime = Get-Date

        # Create CIM session
        $cimSession = New-CimSession -ComputerName $Server -Credential $Credential -ErrorAction Stop

        # Encode command to Base64 for safe execution
        $bytes = [System.Text.Encoding]::Unicode.GetBytes($Command)
        $encodedCommand = [Convert]::ToBase64String($bytes)

        # Create a temporary file on remote system to capture output
        $outputFile = "C:\Windows\Temp\wmi-output-$(Get-Random).txt"
        $commandLine = "powershell.exe -EncodedCommand $encodedCommand > $outputFile 2>&1"

        $processResult = Invoke-CimMethod -CimSession $cimSession -ClassName Win32_Process -MethodName Create -Arguments @{
            CommandLine = $commandLine
        }

        if ($processResult.ReturnValue -eq 0) {
            Write-Host "  Process started (PID: $($processResult.ProcessId))" -ForegroundColor Yellow

            # Wait for process to complete (with timeout)
            $timeout = 30
            $elapsed = 0
            $processRunning = $true

            while ($processRunning -and $elapsed -lt $timeout) {
                Start-Sleep -Seconds 1
                $elapsed++

                $process = Get-CimInstance -CimSession $cimSession -ClassName Win32_Process -Filter "ProcessId = $($processResult.ProcessId)" -ErrorAction SilentlyContinue

                if (-not $process) {
                    $processRunning = $false
                }
            }

            if ($processRunning) {
                Write-Host "  WARNING: Process timeout after $timeout seconds" -ForegroundColor Yellow
            }

            # Try to retrieve output
            Start-Sleep -Seconds 2 # Give file system time to flush

            # Use SMB to read the output file
            try {
                $remotePath = "\\$Server\C$\Windows\Temp\$(Split-Path $outputFile -Leaf)"
                $result = Get-Content $remotePath -ErrorAction Stop

                # Cleanup
                Remove-Item $remotePath -Force -ErrorAction SilentlyContinue

                $duration = ((Get-Date) - $startTime).TotalMilliseconds

                Write-Host "  SUCCESS via WMI ($([math]::Round($duration, 0)) ms)" -ForegroundColor Green
                $methodsAttempted += @{Method="WMI"; Status="Success"; Duration=$duration}
                $success = $true
            } catch {
                Write-Host "  PARTIAL: Process executed but output unavailable" -ForegroundColor Yellow
                $methodsAttempted += @{Method="WMI"; Status="Partial"; Error="Output retrieval failed"}
                $result = "Command executed via WMI (PID: $($processResult.ProcessId)) but output unavailable"
            }
        } else {
            Write-Host "  FAILED: Process creation failed (return code: $($processResult.ReturnValue))" -ForegroundColor Red
            $methodsAttempted += @{Method="WMI"; Status="Failed"; Error="Return code $($processResult.ReturnValue)"}
        }

        Remove-CimSession $cimSession
    } catch {
        Write-Host "  FAILED: $($_.Exception.Message)" -ForegroundColor Red
        $methodsAttempted += @{Method="WMI"; Status="Failed"; Error=$_.Exception.Message}
    }
}

# Method 4: Custom TCP Listener (if all previous methods failed)
if (-not $success) {
    Write-Host "`n[Method 4/4] Attempting Custom TCP Listener (port 9999)..." -ForegroundColor Cyan

    try {
        $startTime = Get-Date

        $client = New-Object System.Net.Sockets.TcpClient
        $asyncResult = $client.BeginConnect($Server, 9999, $null, $null)
        $wait = $asyncResult.AsyncWaitHandle.WaitOne(5000, $false)

        if ($wait -and $client.Connected) {
            $stream = $client.GetStream()
            $reader = New-Object System.IO.StreamReader($stream)
            $writer = New-Object System.IO.StreamWriter($stream)
            $writer.AutoFlush = $true

            # Send command
            $writer.WriteLine($Command)

            # Read response
            $status = $reader.ReadLine()
            $result = $reader.ReadToEnd()

            $client.Close()

            $duration = ((Get-Date) - $startTime).TotalMilliseconds

            if ($status -eq "SUCCESS") {
                Write-Host "  SUCCESS via Custom TCP ($([math]::Round($duration, 0)) ms)" -ForegroundColor Green
                $methodsAttempted += @{Method="Custom TCP"; Status="Success"; Duration=$duration}
                $success = $true
            } else {
                Write-Host "  FAILED: Server returned error" -ForegroundColor Red
                $methodsAttempted += @{Method="Custom TCP"; Status="Failed"; Error=$result}
            }
        } else {
            Write-Host "  FAILED: Connection timeout or refused" -ForegroundColor Red
            $methodsAttempted += @{Method="Custom TCP"; Status="Failed"; Error="Connection timeout"}

            if ($ShowDiagnostics) {
                Write-Host "    Diagnostics: Custom TCP listener may not be running" -ForegroundColor Gray
                Write-Host "    Start listener: tcp-listener-server.ps1" -ForegroundColor Gray
            }
        }
    } catch {
        Write-Host "  FAILED: $($_.Exception.Message)" -ForegroundColor Red
        $methodsAttempted += @{Method="Custom TCP"; Status="Failed"; Error=$_.Exception.Message}
    }
}

# Summary
Write-Host "`n=== Execution Summary ===" -ForegroundColor Cyan

if ($success) {
    Write-Host "Status: SUCCESS" -ForegroundColor Green
    Write-Host "`nOutput:" -ForegroundColor Yellow
    Write-Host $result
} else {
    Write-Host "Status: FAILED - All methods exhausted" -ForegroundColor Red
    Write-Host "`nRecommendations:" -ForegroundColor Yellow
    Write-Host "  1. Enable WinRM on server: run setup-server-winrm.ps1" -ForegroundColor White
    Write-Host "  2. Start TCP listener: tcp-listener-server.ps1" -ForegroundColor White
    Write-Host "  3. Verify firewall rules allow connections" -ForegroundColor White
    Write-Host "  4. Check network connectivity: ping $Server" -ForegroundColor White
}

Write-Host "`nMethods Attempted:" -ForegroundColor Cyan
$methodsAttempted | ForEach-Object {
    $color = switch ($_.Status) {
        "Success" { "Green" }
        "Partial" { "Yellow" }
        "Failed" { "Red" }
    }

    if ($_.Duration) {
        Write-Host ("  {0,-15} {1,-10} ({2} ms)" -f $_.Method, $_.Status, [math]::Round($_.Duration, 0)) -ForegroundColor $color
    } else {
        Write-Host ("  {0,-15} {1,-10} - {2}" -f $_.Method, $_.Status, $_.Error) -ForegroundColor $color
    }
}

# Return exit code
if ($success) {
    exit 0
} else {
    exit 1
}
