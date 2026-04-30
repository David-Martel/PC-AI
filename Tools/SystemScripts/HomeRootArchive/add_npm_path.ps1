$u = [Environment]::GetEnvironmentVariable('Path','User')
$npm = 'C:\Users\david\AppData\Roaming\npm'
if ($u -notlike "*$npm*") {
  $new = $npm + ';' + $u
  [Environment]::SetEnvironmentVariable('Path', $new, 'User')
  Write-Host "Added npm global to User PATH"
} else {
  Write-Host "npm global already in User PATH"
}
