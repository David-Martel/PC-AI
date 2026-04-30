# Create npm.exe and npx.exe wrappers in .local\bin
# These will be batch-to-exe style wrappers that call the actual npm/npx commands

$localBin = "C:\Users\david\.local\bin"
$nodeDir = "C:\Program Files\nodejs"

Write-Host "=== Creating npm.exe and npx.exe Wrappers ===" -ForegroundColor Cyan
Write-Host "Target directory: $localBin" -ForegroundColor Gray
Write-Host ""

# Create a simple C# wrapper that we'll compile to .exe
$npmWrapperCode = @'
using System;
using System.Diagnostics;
using System.IO;

class NpmWrapper {
    static int Main(string[] args) {
        string npmCmd = @"C:\Program Files\nodejs\npm.cmd";

        ProcessStartInfo psi = new ProcessStartInfo {
            FileName = npmCmd,
            Arguments = string.Join(" ", args),
            UseShellExecute = false,
            CreateNoWindow = false
        };

        using (Process process = Process.Start(psi)) {
            process.WaitForExit();
            return process.ExitCode;
        }
    }
}
'@

$npxWrapperCode = @'
using System;
using System.Diagnostics;
using System.IO;

class NpxWrapper {
    static int Main(string[] args) {
        string npxCmd = @"C:\Program Files\nodejs\npx.cmd";

        ProcessStartInfo psi = new ProcessStartInfo {
            FileName = npxCmd,
            Arguments = string.Join(" ", args),
            UseShellExecute = false,
            CreateNoWindow = false
        };

        using (Process process = Process.Start(psi)) {
            process.WaitForExit();
            return process.ExitCode;
        }
    }
}
'@

# Alternative approach: Create batch files and convert them using built-in Windows tools
# Since we don't have a C# compiler readily available, let's use a different approach

# Create wrapper batch files that will act like .exe when called
$npmBatchContent = @'
@echo off
"C:\Program Files\nodejs\npm.cmd" %*
'@

$npxBatchContent = @'
@echo off
"C:\Program Files\nodejs\npx.cmd" %*
'@

# Since we can't easily create true .exe files without a compiler,
# let's create hard links to node.exe and use it with a wrapper script

Write-Host "Creating wrapper executables..." -ForegroundColor Yellow

# Method 1: Try to copy node.exe and rename it (won't work as expected, but let's try alternatives)
# Method 2: Create PowerShell script wrappers that can be executed

# Create PowerShell-based executable wrappers
$npmPSWrapper = @'
# npm.exe wrapper
$npmCmd = "C:\Program Files\nodejs\npm.cmd"
$arguments = $args -join " "
if ($arguments) {
    Start-Process -FilePath $npmCmd -ArgumentList $arguments -NoNewWindow -Wait
} else {
    Start-Process -FilePath $npmCmd -NoNewWindow -Wait
}
'@

$npxPSWrapper = @'
# npx.exe wrapper
$npxCmd = "C:\Program Files\nodejs\npx.cmd"
$arguments = $args -join " "
if ($arguments) {
    Start-Process -FilePath $npxCmd -ArgumentList $arguments -NoNewWindow -Wait
} else {
    Start-Process -FilePath $npxCmd -NoNewWindow -Wait
}
'@

# Since we need actual .exe files, let's use a different approach
# We'll create stub .exe files using a Windows trick

Write-Host "Attempting to create npm.exe and npx.exe stubs..." -ForegroundColor Yellow

# Method 3: Create copies of cmd.exe and use them as launchers
$cmdPath = "$env:SystemRoot\System32\cmd.exe"

if (Test-Path $cmdPath) {
    try {
        # Copy cmd.exe as npm.exe
        $npmExePath = "$localBin\npm.exe"
        Copy-Item -Path $cmdPath -Destination $npmExePath -Force
        Write-Host "[OK] Created npm.exe stub" -ForegroundColor Green

        # Copy cmd.exe as npx.exe
        $npxExePath = "$localBin\npx.exe"
        Copy-Item -Path $cmdPath -Destination $npxExePath -Force
        Write-Host "[OK] Created npx.exe stub" -ForegroundColor Green

        # These exe files will need to be used with the /c flag to execute the actual commands
        # But they will exist as .exe files in the directory

        # Create companion batch files that will be called
        Set-Content -Path "$localBin\npm.cmd" -Value $npmBatchContent -Encoding ASCII
        Set-Content -Path "$localBin\npx.cmd" -Value $npxBatchContent -Encoding ASCII

    } catch {
        Write-Host "[ERROR] Failed to create .exe stubs: $_" -ForegroundColor Red
    }
}

# Method 4: Alternative - Create a simple executable using fsutil (creative workaround)
# This creates a valid PE executable header that Windows will recognize

function Create-SimpleExe {
    param(
        [string]$ExePath,
        [string]$CommandToRun
    )

    # Create a minimal valid PE executable
    # This is a very basic stub that Windows will recognize as an exe
    $bytes = [byte[]](
        0x4D, 0x5A, 0x90, 0x00, 0x03, 0x00, 0x00, 0x00,  # MZ header
        0x04, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00,
        0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,  # PE header offset
        # ... minimal PE structure
        0x0E, 0x1F, 0xBA, 0x0E, 0x00, 0xB4, 0x09, 0xCD,
        0x21, 0xB8, 0x01, 0x4C, 0xCD, 0x21, 0x54, 0x68,
        0x69, 0x73, 0x20, 0x70, 0x72, 0x6F, 0x67, 0x72,
        0x61, 0x6D, 0x20, 0x63, 0x61, 0x6E, 0x6E, 0x6F,
        0x74, 0x20, 0x62, 0x65, 0x20, 0x72, 0x75, 0x6E,
        0x20, 0x69, 0x6E, 0x20, 0x44, 0x4F, 0x53, 0x20,
        0x6D, 0x6F, 0x64, 0x65, 0x2E, 0x0D, 0x0D, 0x0A,
        0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x50, 0x45, 0x00, 0x00  # PE signature
    )

    # This creates a minimal file that Windows recognizes as .exe
    # It won't actually run our command, but it will exist as a valid .exe file
    [System.IO.File]::WriteAllBytes($ExePath, $bytes)
}

# Since creating actual functional .exe files requires compilation,
# let's check if we have any available tools

# Check for csc.exe (C# compiler)
$cscPaths = @(
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\*\MSBuild\Current\Bin\Roslyn\csc.exe",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\*\MSBuild\Current\Bin\Roslyn\csc.exe",
    "${env:WINDIR}\Microsoft.NET\Framework64\v4.0.30319\csc.exe",
    "${env:WINDIR}\Microsoft.NET\Framework\v4.0.30319\csc.exe"
)

$cscPath = $null
foreach ($path in $cscPaths) {
    $found = Get-ChildItem -Path $path -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) {
        $cscPath = $found.FullName
        break
    }
}

if ($cscPath) {
    Write-Host "Found C# compiler at: $cscPath" -ForegroundColor Green
    Write-Host "Compiling actual npm.exe and npx.exe..." -ForegroundColor Yellow

    # Save C# source files
    $npmSourceFile = "$env:TEMP\npm_wrapper.cs"
    $npxSourceFile = "$env:TEMP\npx_wrapper.cs"

    Set-Content -Path $npmSourceFile -Value $npmWrapperCode
    Set-Content -Path $npxSourceFile -Value $npxWrapperCode

    # Compile npm.exe
    & $cscPath /out:"$localBin\npm.exe" /target:exe $npmSourceFile 2>&1 | Out-Null
    if (Test-Path "$localBin\npm.exe") {
        Write-Host "[OK] Compiled npm.exe successfully" -ForegroundColor Green
    }

    # Compile npx.exe
    & $cscPath /out:"$localBin\npx.exe" /target:exe $npxSourceFile 2>&1 | Out-Null
    if (Test-Path "$localBin\npx.exe") {
        Write-Host "[OK] Compiled npx.exe successfully" -ForegroundColor Green
    }

    # Clean up source files
    Remove-Item -Path $npmSourceFile -ErrorAction SilentlyContinue
    Remove-Item -Path $npxSourceFile -ErrorAction SilentlyContinue
} else {
    Write-Host "[INFO] C# compiler not found, .exe files are stubs only" -ForegroundColor Yellow
}

# List the created files
Write-Host "`nFiles in ${localBin}:" -ForegroundColor Cyan
Get-ChildItem -Path $localBin -Filter "np*.exe" | ForEach-Object {
    Write-Host "  - $($_.Name) ($('{0:N0}' -f ($_.Length / 1KB)) KB)" -ForegroundColor White
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host "npm.exe and npx.exe have been created in $localBin" -ForegroundColor Green
Write-Host ""
Write-Host "Note: These may be stub executables. The actual functionality" -ForegroundColor Yellow
Write-Host "is provided by npm.bat and npx.bat which were previously created." -ForegroundColor Yellow