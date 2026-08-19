# tcp-listener-server.ps1
# Run this on the SERVER (10.10.20.214) to enable fallback remote command execution

[System.Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingInvokeExpression', '', Justification = 'Purpose-built remote command listener: executes an arbitrary command line received over the IP-allowlisted socket; there is no fixed command+args to refactor to the call operator.')]
param(
    [int]$Port = 9999,
    [string[]]$AllowedIPs = @("10.10.15.150", "10.10.20.199")
)

Write-Host "=== TCP Command Listener ===" -ForegroundColor Cyan
Write-Host "Listening on port $Port" -ForegroundColor Yellow
Write-Host "Allowed IPs: $($AllowedIPs -join ', ')" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Gray

# Create TCP listener
$endpoint = New-Object System.Net.IPEndPoint([System.Net.IPAddress]::Any, $Port)
$listener = New-Object System.Net.Sockets.TcpListener $endpoint

try {
    $listener.Start()
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Listener started successfully" -ForegroundColor Green

    while ($true) {
        # Wait for connection
        if ($listener.Pending()) {
            $client = $listener.AcceptTcpClient()
            $clientIP = $client.Client.RemoteEndPoint.Address.ToString()

            Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] Connection from $clientIP" -ForegroundColor Cyan

            # Check if IP is allowed
            if ($clientIP -notin $AllowedIPs) {
                Write-Host "  REJECTED: IP not in allowed list" -ForegroundColor Red
                $client.Close()
                continue
            }

            try {
                $stream = $client.GetStream()
                $reader = New-Object System.IO.StreamReader($stream)
                $writer = New-Object System.IO.StreamWriter($stream)
                $writer.AutoFlush = $true

                # Read command
                $command = $reader.ReadLine()

                if ($command) {
                    Write-Host "  Command: $command" -ForegroundColor Yellow

                    # Execute command
                    try {
                        $result = Invoke-Expression $command 2>&1 | Out-String
                        $writer.WriteLine("SUCCESS")
                        $writer.WriteLine($result)
                        Write-Host "  Executed successfully" -ForegroundColor Green
                    } catch {
                        $errorMsg = $_.Exception.Message
                        $writer.WriteLine("ERROR")
                        $writer.WriteLine($errorMsg)
                        Write-Host "  ERROR: $errorMsg" -ForegroundColor Red
                    }
                } else {
                    $writer.WriteLine("ERROR")
                    $writer.WriteLine("No command received")
                }

            } catch {
                Write-Host "  ERROR processing request: $_" -ForegroundColor Red
            } finally {
                $client.Close()
            }
        }

        Start-Sleep -Milliseconds 100
    }

} catch {
    Write-Host "FATAL ERROR: $_" -ForegroundColor Red
} finally {
    $listener.Stop()
    Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] Listener stopped" -ForegroundColor Yellow
}
