@echo off
REM DNS Proxy management convenience script
REM Usage: dns-proxy [start|stop|restart|status|enable-autostart|disable-autostart]

setlocal enabledelayedexpansion

if "%1"=="" (
    echo Usage: dns-proxy [start^|stop^|restart^|status^|enable-autostart^|disable-autostart]
    exit /b 1
)

REM Run PowerShell script with admin privileges if needed
if "%1"=="start" (
    powershell -NoProfile -Command "& {$env:PSExecutionPolicy='Bypass'; & 'C:\Users\david\bin\LocalDNSProxy.ps1' -Action start}"
) else if "%1"=="stop" (
    powershell -NoProfile -Command "& {$env:PSExecutionPolicy='Bypass'; & 'C:\Users\david\bin\LocalDNSProxy.ps1' -Action stop}"
) else if "%1"=="restart" (
    powershell -NoProfile -Command "& {$env:PSExecutionPolicy='Bypass'; & 'C:\Users\david\bin\LocalDNSProxy.ps1' -Action restart}"
) else if "%1"=="status" (
    powershell -NoProfile -Command "& {$env:PSExecutionPolicy='Bypass'; & 'C:\Users\david\bin\LocalDNSProxy.ps1' -Action status}"
) else if "%1"=="enable-autostart" (
    echo Enabling DNS proxy autostart...
    powershell -NoProfile -Command "& {$env:PSExecutionPolicy='Bypass'; Set-Service -Name dnsproxy -StartupType Automatic -ErrorAction SilentlyContinue; Write-Host 'DNS Proxy autostart enabled' -ForegroundColor Green}"
) else if "%1"=="disable-autostart" (
    echo Disabling DNS proxy autostart...
    powershell -NoProfile -Command "& {$env:PSExecutionPolicy='Bypass'; Set-Service -Name dnsproxy -StartupType Manual -ErrorAction SilentlyContinue; Write-Host 'DNS Proxy autostart disabled' -ForegroundColor Green}"
) else (
    echo Unknown command: %1
    exit /b 1
)

endlocal
