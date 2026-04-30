# Create symlinks for Scoop-installed development tools in .local/bin
$scoopDir = "C:\Users\david\scoop"
$localBinDir = "C:\Users\david\.local\bin"

# Ensure .local/bin directory exists
if (!(Test-Path $localBinDir)) {
    New-Item -ItemType Directory -Path $localBinDir -Force | Out-Null
}

Write-Host "Creating symlinks for development tools in $localBinDir"

# Define tool mappings: app name -> executable paths relative to app\current\
$tools = @{
    'meson' = @('bin\meson.exe')
    'ninja' = @('ninja.exe')
    'pkg-config' = @('bin\pkg-config.exe')
    'doxygen' = @('doxygen.exe', 'doxyindexer.exe', 'doxysearch.cgi.exe')
    'llvm' = @('bin\clang.exe', 'bin\clang++.exe', 'bin\llvm-config.exe', 'bin\llc.exe', 'bin\opt.exe')
    'gdb' = @('bin\gdb.exe', 'bin\gdbserver.exe')
    'ccache' = @('ccache.exe')
    '7zip' = @('7z.exe')
}

$successCount = 0
$totalCount = 0

foreach ($app in $tools.Keys) {
    $appDir = "$scoopDir\apps\$app\current"
    if (Test-Path $appDir) {
        Write-Host "Processing $app..."
        foreach ($exe in $tools[$app]) {
            $sourcePath = Join-Path $appDir $exe
            $exeName = Split-Path $exe -Leaf
            $targetPath = Join-Path $localBinDir $exeName
            $totalCount++

            if (Test-Path $sourcePath) {
                try {
                    # Remove existing symlink/file if it exists
                    if (Test-Path $targetPath) {
                        Remove-Item $targetPath -Force -ErrorAction SilentlyContinue
                    }

                    # Create symlink
                    New-Item -ItemType SymbolicLink -Path $targetPath -Target $sourcePath -Force | Out-Null
                    Write-Host "  ✓ Created symlink: $exeName -> $sourcePath"
                    $successCount++
                } catch {
                    Write-Host "  ✗ Failed to create symlink for $exeName`: $_" -ForegroundColor Red
                }
            } else {
                Write-Host "  ✗ Source not found: $sourcePath" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "App directory not found: $appDir" -ForegroundColor Yellow
    }
}

Write-Host "`nSummary:"
Write-Host "Successfully created $successCount out of $totalCount symlinks"
Write-Host "Symlinks location: $localBinDir"

# List all created symlinks
Write-Host "`nCreated symlinks:"
Get-ChildItem $localBinDir -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.LinkType -eq 'SymbolicLink') {
        Write-Host "  $($_.Name) -> $($_.Target)"
    }
}