# This script sets the Git clone directory based on hostname
$hostname = [System.Net.Dns]::GetHostName().ToLower()

# Define machine-specific paths
switch ($hostname) {
	"dtm-p1gen7" { $env:GIT_CLONE_DIR = "C:/codedev/" }
	"desktop-pc" { $env:GIT_CLONE_DIR = "E:\GitRepos" }
	"macbook-pro" { $env:GIT_CLONE_DIR = "$env:USERPROFILE\codedev" }
	default { $env:GIT_CLONE_DIR = "$env:USERPROFILE\Documents\GitHub" }
}

# Output the selected directory
Write-Host "Git clone directory set to: $env:GIT_CLONE_DIR"