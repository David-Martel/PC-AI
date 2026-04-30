# SSH Key Management Helper for Windows
# Save as fix-ssh-agent.ps1

# Check if SSH agent is running
function Test-SSHAgentRunning {
	$service = Get-Service -Name 'ssh-agent' -ErrorAction SilentlyContinue
	return ($service -and $service.Status -eq 'Running')
}

# Ensure SSH agent is running and set to automatic startup
function Enable-SSHAgent {
	$service = Get-Service -Name 'ssh-agent' -ErrorAction SilentlyContinue

	if (-not $service) {
		Write-Host 'OpenSSH Authentication Agent service not found. Make sure OpenSSH Client is installed.' -ForegroundColor Red
		Write-Host 'You can install it via Settings > Apps > Optional Features > Add a feature > OpenSSH Client' -ForegroundColor Yellow
		return $false
	}

	if ($service.Status -ne 'Running') {
		Write-Host 'Starting OpenSSH Authentication Agent service...' -ForegroundColor Yellow
		Start-Service 'ssh-agent'
	}

	# Set startup type to automatic
	$startupType = (Get-Service 'ssh-agent').StartType
	if ($startupType -ne 'Automatic') {
		Write-Host 'Setting OpenSSH Authentication Agent service to start automatically...' -ForegroundColor Yellow
		Set-Service -Name 'ssh-agent' -StartupType Automatic
	}

	Write-Host 'OpenSSH Authentication Agent service is running and set to start automatically.' -ForegroundColor Green
	return $true
}

# List currently loaded keys
function Get-SSHKeys {
	Write-Host 'Currently loaded SSH keys:' -ForegroundColor Cyan
	ssh-add -l

	Write-Host "`nKeys available in ~/.ssh:" -ForegroundColor Cyan
	Get-ChildItem -Path "$env:USERPROFILE\.ssh\id_*" -Exclude '*.pub' | ForEach-Object {
		Write-Host " - $($_.Name)"
	}
}

# Add keys to SSH agent
function Add-SSHKeys {
	$keyFiles = Get-ChildItem -Path "$env:USERPROFILE\.ssh\id_*" -Exclude '*.pub', '*.ppk'

	if ($keyFiles.Count -eq 0) {
		Write-Host 'No SSH keys found in ~/.ssh directory.' -ForegroundColor Yellow
		return
	}

	foreach ($keyFile in $keyFiles) {
		Write-Host "Adding key: $($keyFile.Name)..." -ForegroundColor Yellow
		ssh-add $keyFile.FullName
	}

	Write-Host 'Keys added to SSH agent.' -ForegroundColor Green
}

# Check Pageant integration
function Test-PageantIntegration {
	Write-Host 'Checking Pageant integration...' -ForegroundColor Cyan

	$pageantProcess = Get-Process -Name 'pageant' -ErrorAction SilentlyContinue
	if (-not $pageantProcess) {
		Write-Host "Pageant is not running. This may be fine if you're only using the Windows SSH agent." -ForegroundColor Yellow
		return
	}

	Write-Host "Pageant is running (PID: $($pageantProcess.Id))." -ForegroundColor Green
	Write-Host 'If you want OpenSSH to work with Pageant, ensure you have the correct IdentityAgent directive in your SSH config.' -ForegroundColor Yellow

	# Check for ppk keys that could be loaded into Pageant
	$ppkKeys = Get-ChildItem -Path "$env:USERPROFILE\.ssh\*.ppk" -ErrorAction SilentlyContinue
	if ($ppkKeys.Count -gt 0) {
		Write-Host "Found $($ppkKeys.Count) PuTTY .ppk keys. These should be loaded into Pageant, not OpenSSH." -ForegroundColor Yellow
		foreach ($key in $ppkKeys) {
			Write-Host " - $($key.Name)" -ForegroundColor Yellow
		}
	}
}

# Test SSH config
function Test-SSHConfig {
	Write-Host 'Checking SSH config...' -ForegroundColor Cyan

	$configPath = "$env:USERPROFILE\.ssh\config"
	if (-not (Test-Path $configPath)) {
		Write-Host "SSH config file not found at $configPath" -ForegroundColor Red
		return
	}

	# Look for common issues
	$configContent = Get-Content $configPath -Raw

	if ($configContent -match 'IdentityFile\s+C:\\') {
		Write-Host 'Warning: Found Windows-style backslash paths in your config file.' -ForegroundColor Yellow
		Write-Host '  Consider using forward slashes for better compatibility: ~/.ssh/your_key' -ForegroundColor Yellow
	}

	Write-Host "SSH config file exists. Run 'ssh -G hostname' to see effective settings for a specific host." -ForegroundColor Green
}

# Main script execution
Write-Host 'Windows SSH Key Management Helper' -ForegroundColor Cyan
Write-Host '=================================' -ForegroundColor Cyan

# Step 1: Ensure SSH agent is running
if (Test-SSHAgentRunning) {
	Write-Host 'OpenSSH Authentication Agent is running.' -ForegroundColor Green
} else {
	Write-Host 'OpenSSH Authentication Agent is not running.' -ForegroundColor Yellow
	$startAgent = Read-Host 'Would you like to start it? (y/n)'
	if ($startAgent -eq 'y') {
		Enable-SSHAgent
	}
}

# Step 2: Show currently loaded keys
Get-SSHKeys

# Step 3: Check config and Pageant integration
Test-SSHConfig
Test-PageantIntegration

# Step 4: Add keys if needed
$addKeys = Read-Host "`nWould you like to add your keys to the SSH agent? (y/n)"
if ($addKeys -eq 'y') {
	Add-SSHKeys
}

# Step 5: Test connection
$testHost = Read-Host "`nEnter a host to test connection (leave empty to skip)"
if ($testHost) {
	Write-Host "`nTesting SSH connection to $testHost..." -ForegroundColor Cyan
	Write-Host "Command: ssh -vT $testHost" -ForegroundColor Yellow
	ssh -vT $testHost
}

Write-Host "`nSSH agent setup complete!" -ForegroundColor Green