# tcp-client-invoke.ps1
# Run this on the CLIENT to execute commands via TCP listener

param(
    [Parameter(Mandatory=$true)]
    [string]$Command,

    [string]$Server = "10.10.20.214",
    [int]$Port = 9999,
    [int]$TimeoutSeconds = 30
)

function Invoke-TCPCommand {
    param(
        [string]$Command,
        [string]$Server,
        [int]$Port,
        [int]$Timeout
    )

    Write-Host "Connecting to $Server`:$Port..." -ForegroundColor Yellow

    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $asyncResult = $client.BeginConnect($Server, $Port, $null, $null)
        $wait = $asyncResult.AsyncWaitHandle.WaitOne($Timeout * 1000, $false)

        if (-not $wait -or -not $client.Connected) {
            throw "Connection timeout or refused"
        }

        Write-Host "Connected" -ForegroundColor Green

        $stream = $client.GetStream()
        $reader = New-Object System.IO.StreamReader($stream)
        $writer = New-Object System.IO.StreamWriter($stream)
        $writer.AutoFlush = $true

        # Send command
        Write-Host "Sending command: $Command" -ForegroundColor Yellow
        $writer.WriteLine($Command)

        # Read response
        $status = $reader.ReadLine()
        $result = $reader.ReadToEnd()

        $client.Close()

        if ($status -eq "SUCCESS") {
            Write-Host "`nResult:" -ForegroundColor Green
            Write-Host $result
            return @{Success=$true; Output=$result}
        } else {
            Write-Host "`nError:" -ForegroundColor Red
            Write-Host $result
            return @{Success=$false; Error=$result}
        }

    } catch {
        Write-Host "ERROR: $_" -ForegroundColor Red
        return @{Success=$false; Error=$_.Exception.Message}
    }
}

# Execute
$result = Invoke-TCPCommand -Command $Command -Server $Server -Port $Port -Timeout $TimeoutSeconds

# Exit with appropriate code
if ($result.Success) {
    exit 0
} else {
    exit 1
}
