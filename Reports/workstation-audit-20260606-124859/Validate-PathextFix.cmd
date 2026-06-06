@echo off
setlocal
set "PROBE_DIR=%~dp0pathext-probe"
set "PATH=%PROBE_DIR%;%PATH%"
set "PATHEXT=.COM;.EXE;.BAT;.CMD;.MSC;.PY;.PYW;.CPL"

where probe >nul 2>nul
if errorlevel 1 (
  echo FAIL: probe.cmd was not command-discoverable
  exit /b 1
)

for /f "delims=" %%I in ('where probe 2^>nul') do echo FOUND_CMD=%%I
for /f "delims=" %%I in ('probe') do echo RUN_CMD=%%I

where probejs >nul 2>nul
if not errorlevel 1 (
  echo FAIL: probejs.js was command-discoverable as bare command probejs
  where probejs
  exit /b 2
)

echo PASS: sanitized PATHEXT discovers .cmd and excludes .js
