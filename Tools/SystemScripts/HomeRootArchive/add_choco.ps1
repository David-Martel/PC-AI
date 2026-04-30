$m = [Environment]::GetEnvironmentVariable('Path','Machine')
if ($m -notlike '*chocolatey*') {
  $new = $m + ';C:\ProgramData\chocolatey\bin'
  [Environment]::SetEnvironmentVariable('Path', $new, 'Machine')
  Write-Host 'Added chocolatey to Machine PATH'
} else {
  Write-Host 'Chocolatey already in PATH'
}
