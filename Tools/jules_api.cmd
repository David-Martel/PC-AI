@echo off
REM Jules API wrapper -- delegates to Invoke-JulesSession.ps1
REM
REM Usage:
REM   jules_api.cmd -Action List -Format Table
REM   jules_api.cmd -Action New -Prompt "Fix lint errors in PC-AI.Hardware"
REM   jules_api.cmd -Action Status -SessionId abc123
REM   jules_api.cmd -Action Approve -SessionId abc123
REM   jules_api.cmd -Action GetPatch -SessionId abc123
REM
REM All arguments are forwarded verbatim to the PowerShell script.
REM Requires pwsh (PowerShell 7+) on PATH; falls back to powershell.exe for PS 5.1.

setlocal

set "SCRIPT=%~dp0Invoke-JulesSession.ps1"

where pwsh >nul 2>&1
if %ERRORLEVEL% equ 0 (
    pwsh -NoLogo -NoProfile -File "%SCRIPT%" %*
) else (
    powershell -NoLogo -NoProfile -File "%SCRIPT%" %*
)

endlocal
