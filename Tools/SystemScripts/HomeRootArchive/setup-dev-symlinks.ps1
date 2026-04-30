# Developer Tools Symlink Setup Script
# Creates symlinks for all essential development tools in ~/bin and ~/.local/bin

$Bin="$HOME\bin"
$LocalBin="$HOME\.local\bin"

function New-ToolLink {
  param(
    [Parameter(Mandatory)][string]$LinkPath,
    [Parameter(Mandatory)][string]$TargetPath
  )
  $LinkDir=Split-Path -Parent $LinkPath
  New-Item -ItemType Directory -Force -Path $LinkDir | Out-Null

  if (Test-Path $LinkPath) { Remove-Item -Force $LinkPath }

  try {
    New-Item -ItemType SymbolicLink -Path $LinkPath -Target $TargetPath -ErrorAction Stop | Out-Null
    Write-Host "✓ Symlink: $(Split-Path -Leaf $LinkPath)" -ForegroundColor Green
    return
  } catch {}

  # Try hardlink if same drive
  if (([IO.Path]::GetPathRoot($LinkPath)).ToLower() -eq ([IO.Path]::GetPathRoot($TargetPath)).ToLower()) {
    cmd /c mklink /H "$LinkPath" "$TargetPath" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
      Write-Host "✓ Hardlink: $(Split-Path -Leaf $LinkPath)" -ForegroundColor Yellow
      return
    }
  }

  # Final fallback: .cmd shim
  $ext=[IO.Path]::GetExtension($LinkPath)
  if ($ext -ieq '.exe') {
    $LinkPath=[IO.Path]::ChangeExtension($LinkPath,'.cmd')
  }
  "@echo off`r`n`"$TargetPath`" %*" | Set-Content -Encoding ASCII $LinkPath
  Write-Host "✓ Shim: $(Split-Path -Leaf $LinkPath)" -ForegroundColor Cyan
}

Write-Host "`n=== Discovering Tool Locations ===" -ForegroundColor Magenta

# MSVC cl.exe
$vswhere="${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$clTarget = if (Test-Path $vswhere) {
  & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe' | Select-Object -First 1
}
if (-not $clTarget) {
  $clTarget='C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe'
}

# Intel oneAPI
$oneApiRoot='C:\Program Files (x86)\Intel\oneAPI'
$icxBin="$oneApiRoot\compiler\2025.0\bin"
if (-not (Test-Path $icxBin)) {
  $icxBin=(Get-ChildItem "$oneApiRoot\compiler" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1 | ForEach-Object { Join-Path $_.FullName 'bin' })
}
$icxTarget=Join-Path $icxBin 'icx.exe'
$icxclTarget=Join-Path $icxBin 'icx-cl.exe'
$setvarsTarget=Join-Path $oneApiRoot 'setvars.bat'

# Build tools
$ninjaTarget='C:\codedev\msys64\mingw64\bin\ninja.exe'
$vcpkgTarget='C:\codedev\vcpkg\vcpkg.exe'
$cmakeTarget='C:\Program Files\CMake\bin\cmake.exe'
$makeTarget='C:\codedev\msys64\mingw64\bin\mingw32-make.exe'

# VCS and Python tools
$ghTarget='C:\Program Files\GitHub CLI\gh.exe'
$uvTarget="$HOME\.cargo\bin\uv.exe"
$uvxTarget="$HOME\.cargo\bin\uvx.exe"
$claudeTarget="$HOME\AppData\Roaming\npm\claude.cmd"

# Display discovery results
$tools = @{
  'cl.exe'=$clTarget; 'icx.exe'=$icxTarget; 'icx-cl.exe'=$icxclTarget
  'setvars.bat'=$setvarsTarget; 'ninja.exe'=$ninjaTarget; 'vcpkg.exe'=$vcpkgTarget
  'cmake.exe'=$cmakeTarget; 'make.exe'=$makeTarget; 'gh.exe'=$ghTarget
  'uv.exe'=$uvTarget; 'uvx.exe'=$uvxTarget; 'claude.cmd'=$claudeTarget
}

foreach($k in $tools.Keys | Sort-Object) {
  $exists = Test-Path $tools[$k]
  $status = if($exists){"✓"}else{"✗ MISSING"}
  $color = if($exists){"Green"}else{"Red"}
  Write-Host "$status $k" -ForegroundColor $color -NoNewline
  Write-Host " -> $($tools[$k])" -ForegroundColor Gray
}

Write-Host "`n=== Creating Symlinks in $Bin ===" -ForegroundColor Magenta
if (Test-Path $clTarget) { New-ToolLink -LinkPath "$Bin\cl.exe" -TargetPath $clTarget }
if (Test-Path $icxTarget) { New-ToolLink -LinkPath "$Bin\icx.exe" -TargetPath $icxTarget }
if (Test-Path $icxclTarget) { New-ToolLink -LinkPath "$Bin\icx-cl.exe" -TargetPath $icxclTarget }
if (Test-Path $setvarsTarget) { New-ToolLink -LinkPath "$Bin\setvars.bat" -TargetPath $setvarsTarget }
if (Test-Path $ninjaTarget) { New-ToolLink -LinkPath "$Bin\ninja.exe" -TargetPath $ninjaTarget }
if (Test-Path $vcpkgTarget) { New-ToolLink -LinkPath "$Bin\vcpkg.exe" -TargetPath $vcpkgTarget }
if (Test-Path $cmakeTarget) { New-ToolLink -LinkPath "$Bin\cmake.exe" -TargetPath $cmakeTarget }
if (Test-Path $makeTarget) { New-ToolLink -LinkPath "$Bin\make.exe" -TargetPath $makeTarget }
if (Test-Path $ghTarget) { New-ToolLink -LinkPath "$Bin\gh.exe" -TargetPath $ghTarget }
if (Test-Path $uvTarget) { New-ToolLink -LinkPath "$Bin\uv.exe" -TargetPath $uvTarget }
if (Test-Path $uvxTarget) { New-ToolLink -LinkPath "$Bin\uvx.exe" -TargetPath $uvxTarget }
if (Test-Path $claudeTarget) { New-ToolLink -LinkPath "$Bin\claude.cmd" -TargetPath $claudeTarget }

Write-Host "`n=== Mirroring Symlinks in $LocalBin ===" -ForegroundColor Magenta
$map=@{
  "$LocalBin\cl.exe"        = $clTarget
  "$LocalBin\icx.exe"       = $icxTarget
  "$LocalBin\icx-cl.exe"    = $icxclTarget
  "$LocalBin\setvars.bat"   = $setvarsTarget
  "$LocalBin\ninja.exe"     = $ninjaTarget
  "$LocalBin\vcpkg.exe"     = $vcpkgTarget
  "$LocalBin\cmake.exe"     = $cmakeTarget
  "$LocalBin\make.exe"      = $makeTarget
  "$LocalBin\gh.exe"        = $ghTarget
  "$LocalBin\uv.exe"        = $uvTarget
  "$LocalBin\uvx.exe"       = $uvxTarget
  "$LocalBin\claude.cmd"    = $claudeTarget
}
foreach($k in $map.Keys){
  $t=$map[$k]
  if($t -and (Test-Path $t)){ New-ToolLink -LinkPath $k -TargetPath $t }
}

Write-Host "`n=== Setup Complete! ===" -ForegroundColor Green
Write-Host "Both $Bin and $LocalBin now contain symlinks to your dev tools."
Write-Host "PATH priority: $Bin (highest) > $LocalBin"
