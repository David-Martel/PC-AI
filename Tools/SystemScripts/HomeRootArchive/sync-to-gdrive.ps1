$dest = 'G:\My Drive\scripts\dtm-remote-setup'

# Copy updated main scripts
Copy-Item 'C:\Users\david\.claude\scripts\dtm-remote-setup.ps1' $dest -Force
Copy-Item 'C:\Users\david\.claude\scripts\dtm-remote-setup.sh' $dest -Force
Write-Host "Copied: dtm-remote-setup.ps1"
Write-Host "Copied: dtm-remote-setup.sh"

# Copy new network scripts
$newScripts = @(
    'invoke-robust-remote.ps1',
    'setup-server-winrm.ps1',
    'tcp-client-invoke.ps1',
    'tcp-listener-server.ps1',
    'test-connection-methods.ps1',
    'test-smb-performance.ps1'
)

foreach ($script in $newScripts) {
    $src = Join-Path 'C:\Users\david' $script
    if (Test-Path $src) {
        Copy-Item $src $dest -Force
        Write-Host "Copied: $script"
    }
}

# List final contents
Write-Host "`nGoogle Drive folder contents:"
Get-ChildItem $dest | Format-Table Name, Length, LastWriteTime -AutoSize
