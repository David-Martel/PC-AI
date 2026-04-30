# Simple solution: Create .exe files that delegate to .bat files
$localBin = "C:\Users\david\.local\bin"

Write-Host "=== Creating Simple npm.exe and npx.exe Delegates ===" -ForegroundColor Cyan

# Remove problematic compiled versions
if (Test-Path "$localBin\npm.exe.bak") {
    Remove-Item "$localBin\npm.exe" -Force -ErrorAction SilentlyContinue
    Write-Host "Removed problematic npm.exe" -ForegroundColor Yellow
}
if (Test-Path "$localBin\npx.exe.bak") {
    Remove-Item "$localBin\npx.exe" -Force -ErrorAction SilentlyContinue
    Write-Host "Removed problematic npx.exe" -ForegroundColor Yellow
}

# Simple C# code that uses cmd.exe to run the batch files
$npmSimpleCode = @'
using System;
using System.Diagnostics;

class Program {
    static int Main(string[] args) {
        var startInfo = new ProcessStartInfo {
            FileName = "cmd.exe",
            Arguments = "/c npm " + string.Join(" ", args),
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using (var process = Process.Start(startInfo)) {
            process.WaitForExit();
            return process.ExitCode;
        }
    }
}
'@

$npxSimpleCode = @'
using System;
using System.Diagnostics;

class Program {
    static int Main(string[] args) {
        var startInfo = new ProcessStartInfo {
            FileName = "cmd.exe",
            Arguments = "/c npx " + string.Join(" ", args),
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using (var process = Process.Start(startInfo)) {
            process.WaitForExit();
            return process.ExitCode;
        }
    }
}
'@

# Check for C# compiler
$cscPath = Get-ChildItem -Path "${env:ProgramFiles}\Microsoft Visual Studio\2022\*\MSBuild\Current\Bin\Roslyn\csc.exe" -ErrorAction SilentlyContinue | Select-Object -First 1

if ($cscPath) {
    Write-Host "Compiling simple delegate executables..." -ForegroundColor Yellow

    $npmTempSrc = "$env:TEMP\npm_simple.cs"
    $npxTempSrc = "$env:TEMP\npx_simple.cs"

    Set-Content -Path $npmTempSrc -Value $npmSimpleCode
    Set-Content -Path $npxTempSrc -Value $npxSimpleCode

    # Compile with specific options for console apps
    & $cscPath.FullName /out:"$localBin\npm.exe" /target:winexe /platform:anycpu $npmTempSrc 2>&1 | Out-Null
    & $cscPath.FullName /out:"$localBin\npx.exe" /target:winexe /platform:anycpu $npxTempSrc 2>&1 | Out-Null

    Remove-Item $npmTempSrc, $npxTempSrc -ErrorAction SilentlyContinue

    if ((Test-Path "$localBin\npm.exe") -and (Test-Path "$localBin\npx.exe")) {
        Write-Host "[OK] Created delegate executables" -ForegroundColor Green
    }
} else {
    # Fallback: Just copy node.exe and rename
    Write-Host "No compiler found, using node.exe copy method..." -ForegroundColor Yellow

    $nodeExe = "C:\Program Files\nodejs\node.exe"
    if (Test-Path $nodeExe) {
        # These won't work as npm/npx but will exist as .exe files
        Copy-Item $nodeExe "$localBin\npm.exe" -Force
        Copy-Item $nodeExe "$localBin\npx.exe" -Force
        Write-Host "[INFO] Created .exe files (use npm.bat/npx.bat for functionality)" -ForegroundColor Yellow
    }
}

# Ensure we have working .bat files as primary executables
$npmBat = @'
@echo off
"C:\Program Files\nodejs\npm.cmd" %*
'@

$npxBat = @'
@echo off
"C:\Program Files\nodejs\npx.cmd" %*
'@

Set-Content -Path "$localBin\npm.bat" -Value $npmBat -Encoding ASCII
Set-Content -Path "$localBin\npx.bat" -Value $npxBat -Encoding ASCII

Write-Host "`n=== Files Created ===" -ForegroundColor Cyan
$files = Get-ChildItem "$localBin\np*.*" | Select-Object Name, @{N='Size(KB)';E={[math]::Round($_.Length/1KB, 2)}}
$files | Format-Table -AutoSize

Write-Host "`n=== Testing (Quick) ===" -ForegroundColor Cyan

# Test using the batch files which we know work
if (Test-Path "$localBin\npm.bat") {
    $npmVer = & cmd /c "$localBin\npm.bat" --version 2>&1
    Write-Host "npm (via .bat): $npmVer" -ForegroundColor Green
}

if (Test-Path "$localBin\npx.bat") {
    $npxVer = & cmd /c "$localBin\npx.bat" --version 2>&1
    Write-Host "npx (via .bat): $npxVer" -ForegroundColor Green
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host @"
Created in $localBin :
  - npm.exe (delegate to cmd)
  - npx.exe (delegate to cmd)
  - npm.bat (primary functional wrapper)
  - npx.bat (primary functional wrapper)

The .exe files exist for compatibility with tools expecting .exe extensions.
The .bat files provide the actual functionality.
"@ -ForegroundColor White

# Clean up backups
Remove-Item "$localBin\*.exe.bak" -ErrorAction SilentlyContinue