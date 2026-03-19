param(
    [string]$Device = 'cuda:0',
    [int]$Port = 18203,
    [int]$TimeoutSec = 120
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent

$proc = Start-Process -FilePath (Join-Path $repoRoot 'pcai-media.exe') `
    -ArgumentList @('--model', (Join-Path $repoRoot 'Models/Janus-Pro-1B'), '--device', $Device, '--port', "$Port") `
    -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $repoRoot "pcai-media-stdout-$Port.log") `
    -RedirectStandardError (Join-Path $repoRoot "pcai-media-stderr-$Port.log")

try {
    Write-Host "Starting pcai-media (device=$Device, port=$Port)..."
    for ($i = 0; $i -lt $TimeoutSec; $i++) {
        Start-Sleep -Seconds 1
        if ($proc.HasExited) {
            Write-Host "Server exited: $($proc.ExitCode)" -ForegroundColor Red
            Get-Content (Join-Path $repoRoot "pcai-media-stdout-$Port.log") -Tail 5
            return
        }
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
            if ($h.model_loaded) {
                Write-Host "SERVER READY in $i seconds" -ForegroundColor Green
                break
            }
        } catch { }
    }

    if (-not $h -or -not $h.model_loaded) {
        Write-Host "Server did not become ready" -ForegroundColor Red
        Get-Content (Join-Path $repoRoot "pcai-media-stdout-$Port.log") -Tail 10
        return
    }

    # Generation test
    Write-Host "`nTest 1: Image Generation" -ForegroundColor Cyan
    $body = @{ prompt = 'A red apple on a white table'; cfg_scale = 5.0; temperature = 1.0 } | ConvertTo-Json
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $gen = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/images/generate" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 300
    $sw.Stop()
    $tokPerSec = [math]::Round(576 / $sw.Elapsed.TotalSeconds, 1)
    Write-Host "  Generated: $($gen.width)x$($gen.height) in $([math]::Round($sw.Elapsed.TotalSeconds, 1))s (~$tokPerSec tok/s)" -ForegroundColor Green

    # Understanding test
    $imgPath = Join-Path $repoRoot 'Reports/media/understand-red.png'
    if (Test-Path $imgPath) {
        Write-Host "`nTest 2: Image Understanding" -ForegroundColor Cyan
        $imgB64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($imgPath))
        $undBody = @{ image_base64 = $imgB64; prompt = 'Describe this image in detail. What colors, shapes, and objects do you see?'; max_tokens = 128; temperature = 0.1 } | ConvertTo-Json
        $sw2 = [System.Diagnostics.Stopwatch]::StartNew()
        $und = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/images/understand" -Method Post -ContentType 'application/json' -Body $undBody -TimeoutSec 120
        $sw2.Stop()
        Write-Host "  Response ($($und.text.Length) chars, $([math]::Round($sw2.Elapsed.TotalSeconds, 1))s):" -ForegroundColor Green
        Write-Host "  $($und.text)" -ForegroundColor White
    }

    Write-Host "`nAll tests passed!" -ForegroundColor Green
} finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
