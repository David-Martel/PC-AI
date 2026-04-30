# Test PowerShell syntax for GcpUtils files
$files = @(
    'C:\Users\david\OneDrive\Documents\PowerShell\Modules\GcpUtils\GcpProfileManagerV2.ps1',
    'C:\Users\david\OneDrive\Documents\PowerShell\Modules\GcpUtils\Enable-WarpIntegration.ps1',
    'C:\Users\david\OneDrive\Documents\PowerShell\Modules\GcpUtils\Enable-WindowsTerminalWarp.ps1',
    'C:\Users\david\OneDrive\Documents\PowerShell\Modules\GcpUtils\Enable-GcpProfileSync.ps1'
)

foreach ($file in $files) {
    Write-Host "Testing: $(Split-Path $file -Leaf)" -ForegroundColor Cyan
    try {
        $parseErrors = $null
        [void][System.Management.Automation.Language.Parser]::ParseFile($file, [ref]$null, [ref]$parseErrors)
        if ($parseErrors.Count -eq 0) {
            Write-Host "  Syntax OK" -ForegroundColor Green
        } else {
            Write-Host "  Syntax Errors Found:" -ForegroundColor Red
            foreach ($error in $parseErrors) {
                Write-Host "    Line $($error.Extent.StartLineNumber): $($error.Message)" -ForegroundColor Yellow
            }
        }
    } catch {
        Write-Host "  Parse Error: $_" -ForegroundColor Red
    }
    Write-Host
}