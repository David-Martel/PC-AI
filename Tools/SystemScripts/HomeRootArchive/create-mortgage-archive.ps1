# Create password-protected mortgage archive
$securePassword = Read-Host -Prompt "Enter mortgage archive password" -AsSecureString

# Archive parameters
$params = @{
    Path = "T:\cloud-cache\google\My Drive\DTM-Haus-Two\mortage-details"
    Password = $securePassword
    Format = "zip"
    Level = "maximum"
    Exclude = @("*.gdoc", "*.gsheet", "*.gslides", "*.gdraw", "*.gtable")
    OutputPath = "C:\Users\david\mortgage-archive.zip"
}

Write-Host "Creating password-protected mortgage archive..." -ForegroundColor Green
Write-Host "Password: <provided interactively>" -ForegroundColor Yellow
Write-Host ""

# Call the universal archiver
& "C:\Users\david\universal-archiver.ps1" @params
