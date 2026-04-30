$ErrorActionPreference = 'Stop'

# Get current Machine PATH
$pathM = [Environment]::GetEnvironmentVariable('Path','Machine')
$entries = $pathM -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }

# Entries to remove
$remove = @(
  'C:\codedev\msys64\mingw64\bin',
  'C:\msys64\mingw64\bin',
  'C:\msys64\usr\bin',
  'C:\Program Files\Git\usr\bin',
  'C:\Program Files (x86)\Windows Kits\10\Wi',
  'C:\users\david\.local\bin',
  'C:/Program Files/CMake/share/cmake-4.1'
)

# Filter out unwanted entries
$cleaned = $entries | Where-Object {
  $e = $_ -replace '/','\\'  # Normalize slashes
  -not ($remove | Where-Object { $_ -eq $e })
}

# Remove duplicates (case-insensitive)
$seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$final = $cleaned | Where-Object { $seen.Add($_) }

# Set Machine PATH
$newPath = $final -join ';'
[Environment]::SetEnvironmentVariable('Path', $newPath, 'Machine')

Write-Host "[OK] Cleaned Machine PATH"
Write-Host "Removed entries:"
$remove | ForEach-Object { Write-Host "  - $_" }
Write-Host "`nMachine PATH now has $($final.Count) entries"
