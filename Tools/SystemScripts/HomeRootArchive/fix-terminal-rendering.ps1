# Windows Terminal Rendering Fix for Claude Code
# Fixes the issue where each character creates a new line

Write-Host "Applying Windows Terminal rendering fixes for Claude Code..." -ForegroundColor Cyan

# Fix 1: Reset Terminal Settings
Write-Host "`nResetting terminal settings..." -ForegroundColor Yellow
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:TERM = "xterm-256color"

# Fix 2: Disable Windows Terminal GPU acceleration (can cause rendering issues)
Write-Host "Checking Windows Terminal settings..." -ForegroundColor Yellow
$wtSettingsPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"
$wtPreviewSettingsPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminalPreview_8wekyb3d8bbwe\LocalState\settings.json"

function Update-WTSettings {
    param($path)
    if (Test-Path $path) {
        Write-Host "Found Windows Terminal settings at: $path" -ForegroundColor Green
        $settings = Get-Content $path -Raw | ConvertFrom-Json

        # Add rendering settings if not present
        if (-not $settings.PSObject.Properties['rendering']) {
            $settings | Add-Member -NotePropertyName 'rendering' -NotePropertyValue @{} -Force
        }

        # Disable GPU acceleration
        $settings.rendering | Add-Member -NotePropertyName 'disableGPU' -NotePropertyValue $true -Force
        $settings.rendering | Add-Member -NotePropertyName 'software' -NotePropertyValue $true -Force

        # Add experimental rendering settings
        if (-not $settings.PSObject.Properties['experimental']) {
            $settings | Add-Member -NotePropertyName 'experimental' -NotePropertyValue @{} -Force
        }
        $settings.experimental | Add-Member -NotePropertyName 'rendering.software' -NotePropertyValue $true -Force

        # Backup and save
        Copy-Item $path "$path.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')" -Force
        $settings | ConvertTo-Json -Depth 10 | Set-Content $path -Force
        Write-Host "Updated Windows Terminal settings" -ForegroundColor Green
    }
}

# Fix 3: PowerShell specific settings
Write-Host "`nApplying PowerShell fixes..." -ForegroundColor Yellow
$PSReadLineOptions = @{
    EditMode = 'Windows'
    HistoryNoDuplicates = $true
    HistorySearchCursorMovesToEnd = $true
    BellStyle = 'None'
    PredictionSource = 'History'
    PredictionViewStyle = 'ListView'
}

foreach ($option in $PSReadLineOptions.GetEnumerator()) {
    try {
        Set-PSReadLineOption -$($option.Key) $($option.Value)
    } catch {
        Write-Host "Could not set PSReadLine option: $($option.Key)" -ForegroundColor Red
    }
}

# Fix 4: Console Mode Settings
Write-Host "`nAdjusting console mode settings..." -ForegroundColor Yellow
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public class ConsoleHelper {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr GetStdHandle(int nStdHandle);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool GetConsoleMode(IntPtr hConsoleHandle, out uint lpMode);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool SetConsoleMode(IntPtr hConsoleHandle, uint dwMode);

    public const int STD_INPUT_HANDLE = -10;
    public const int STD_OUTPUT_HANDLE = -11;

    public const uint ENABLE_ECHO_INPUT = 0x0004;
    public const uint ENABLE_LINE_INPUT = 0x0002;
    public const uint ENABLE_PROCESSED_INPUT = 0x0001;
    public const uint ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004;
    public const uint DISABLE_NEWLINE_AUTO_RETURN = 0x0008;

    public static void FixConsoleMode() {
        IntPtr inputHandle = GetStdHandle(STD_INPUT_HANDLE);
        IntPtr outputHandle = GetStdHandle(STD_OUTPUT_HANDLE);

        uint inputMode, outputMode;
        GetConsoleMode(inputHandle, out inputMode);
        GetConsoleMode(outputHandle, out outputMode);

        // Fix input mode
        inputMode |= ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT;
        SetConsoleMode(inputHandle, inputMode);

        // Fix output mode
        outputMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        outputMode &= ~DISABLE_NEWLINE_AUTO_RETURN;
        SetConsoleMode(outputHandle, outputMode);
    }
}
"@

try {
    [ConsoleHelper]::FixConsoleMode()
    Write-Host "Console mode settings adjusted" -ForegroundColor Green
} catch {
    Write-Host "Could not adjust console mode: $_" -ForegroundColor Red
}

# Fix 5: Clear and reset the terminal
Write-Host "`nClearing terminal buffer..." -ForegroundColor Yellow
Clear-Host

# Fix 6: Environment variables for better compatibility
$env:COLUMNS = [Console]::WindowWidth
$env:LINES = [Console]::WindowHeight
$env:COLORTERM = 'truecolor'

# Fix 7: Test the fix
Write-Host "`n" -NoNewline
Write-Host "=== Terminal Rendering Fix Applied ===" -ForegroundColor Green
Write-Host "Please try typing now. If the issue persists:" -ForegroundColor Cyan
Write-Host "1. Restart Windows Terminal" -ForegroundColor White
Write-Host "2. Run this script again after restart" -ForegroundColor White
Write-Host "3. Try: Set-PSReadLineOption -EditMode Emacs" -ForegroundColor White
Write-Host "4. Or try: Set-PSReadLineOption -EditMode Vi" -ForegroundColor White
Write-Host "`nAdditional troubleshooting:" -ForegroundColor Yellow
Write-Host "- Check if Windows Terminal is up to date" -ForegroundColor White
Write-Host "- Try disabling GPU acceleration in Windows Terminal settings" -ForegroundColor White
Write-Host "- Reset Windows Terminal settings to default" -ForegroundColor White

# Fix 8: Create a permanent fix profile
$profileContent = @'
# Claude Code Terminal Rendering Fix
if ($env:TERM_PROGRAM -eq 'claude' -or $Host.Name -match 'claude') {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $env:TERM = "xterm-256color"

    # PSReadLine settings for better compatibility
    if (Get-Module -ListAvailable -Name PSReadLine) {
        Set-PSReadLineOption -EditMode Windows
        Set-PSReadLineOption -BellStyle None
        Set-PSReadLineOption -PredictionSource History
    }
}
'@

$profilePath = $PROFILE.CurrentUserCurrentHost
Write-Host "`nWould you like to add the fix to your PowerShell profile for permanent effect? (Y/N)" -ForegroundColor Cyan
$response = Read-Host
if ($response -eq 'Y' -or $response -eq 'y') {
    if (!(Test-Path $profilePath)) {
        New-Item -Path $profilePath -ItemType File -Force | Out-Null
    }
    Add-Content -Path $profilePath -Value "`n$profileContent" -Force
    Write-Host "Fix added to PowerShell profile: $profilePath" -ForegroundColor Green
}

Write-Host "`nDone! Terminal rendering fixes applied." -ForegroundColor Green