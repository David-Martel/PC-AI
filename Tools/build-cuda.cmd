@echo off
setlocal

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%" >nul

where pwsh >nul 2>nul
if %ERRORLEVEL%==0 (
  pwsh -NoProfile -ExecutionPolicy Bypass -File ".\Build.ps1" -Component inference -EnableCuda -RunTests
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File ".\Build.ps1" -Component inference -EnableCuda -RunTests
)

set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%
