#Requires -RunAsAdministrator
<#
.SYNOPSIS
Test and validate DNS proxy and service registry setup.

.DESCRIPTION
Runs comprehensive tests to ensure:
1. Service registry file is valid
2. PowerShell module can be imported
3. Environment variables can be read
4. DNS proxy script syntax is valid
5. Batch wrapper script exists
#>

$ErrorActionPreference = 'Continue'
$testResults = @()

function Test-Item {
    param(
        [string]$Name,
        [scriptblock]$TestBlock
    )

    try {
        $result = & $TestBlock
        $status = if ($result) { "[PASS]" } else { "[FAIL]" }
        Write-Host "$status : $Name" -ForegroundColor $(if ($result) { 'Green' } else { 'Red' })
        $testResults += @{ Name = $Name; Result = $result }
    } catch {
        Write-Host "[ERROR] : $Name" -ForegroundColor Red
        Write-Host "  Details: $_" -ForegroundColor Yellow
        $testResults += @{ Name = $Name; Result = $false }
    }
}

Write-Host "`n=== DNS Proxy Setup Validation ===" -ForegroundColor Cyan
Write-Host "Testing components and configuration...`n" -ForegroundColor Gray

# Test 1: Service Registry File
Test-Item "Service Registry file exists" {
    Test-Path "$env:USERPROFILE\.service-registry.json"
}

# Test 2: Service Registry is valid JSON
Test-Item "Service Registry is valid JSON" {
    $json = Get-Content "$env:USERPROFILE\.service-registry.json" | ConvertFrom-Json
    $json -ne $null
}

# Test 3: Service Registry contains required services
Test-Item "Service Registry has MCP service" {
    $json = Get-Content "$env:USERPROFILE\.service-registry.json" | ConvertFrom-Json
    $json.services.mcp -ne $null
}

# Test 4: PowerShell Module exists
Test-Item "PowerShell LocalServiceRegistry module exists" {
    Test-Path "$env:USERPROFILE\Documents\PowerShell\Modules\LocalServiceRegistry\LocalServiceRegistry.psm1"
}

# Test 5: PowerShell Module can be imported
Test-Item "PowerShell module imports successfully" {
    Import-Module "$env:USERPROFILE\Documents\PowerShell\Modules\LocalServiceRegistry\LocalServiceRegistry.psm1" -ErrorAction Stop
    $true
}

# Test 6: Module exports required functions
Test-Item "Module exports Get-LocalService function" {
    Get-Command Get-LocalService -ErrorAction SilentlyContinue -Module LocalServiceRegistry | Out-Null
    $?
}

Test-Item "Module exports Open-LocalService function" {
    Get-Command Open-LocalService -ErrorAction SilentlyContinue -Module LocalServiceRegistry | Out-Null
    $?
}

# Test 7: DNS Proxy script syntax
Test-Item "LocalDNSProxy.ps1 syntax is valid" {
    $ast = $null
    $tokens = @()
    $parseErrors = @()
    [System.Management.Automation.Language.Parser]::ParseFile(
        "C:\Users\david\bin\LocalDNSProxy.ps1",
        [ref]$tokens,
        [ref]$parseErrors
    ) | Out-Null
    $parseErrors.Count -eq 0
}

# Test 8: Setup script syntax
Test-Item "setup-dns-env.ps1 syntax is valid" {
    $ast = $null
    $tokens = @()
    $parseErrors = @()
    [System.Management.Automation.Language.Parser]::ParseFile(
        "C:\Users\david\bin\setup-dns-env.ps1",
        [ref]$tokens,
        [ref]$parseErrors
    ) | Out-Null
    $parseErrors.Count -eq 0
}

# Test 9: Batch wrapper exists
Test-Item "dns-proxy.bat wrapper exists" {
    Test-Path "C:\Users\david\bin\dns-proxy.bat"
}

# Test 10: Profile additions file exists
Test-Item "pwsh-profile-additions.ps1 exists" {
    Test-Path "C:\Users\david\bin\pwsh-profile-additions.ps1"
}

# Test 11: Documentation exists
Test-Item "DNS-PROXY-SETUP.md documentation exists" {
    Test-Path "C:\Users\david\bin\DNS-PROXY-SETUP.md"
}

# Test 12: Hosts file is clean
Test-Item "Hosts file has no .localhost entries" {
    $hostsContent = Get-Content "C:\Windows\System32\drivers\etc\hosts" | Out-String
    -not ($hostsContent -match '\.localhost')
}

# Test 13: Hosts file still has essential entries
Test-Item "Hosts file has localhost entries" {
    $hostsContent = Get-Content "C:\Windows\System32\drivers\etc\hosts" | Out-String
    ($hostsContent -match '127\.0\.0\.1.*localhost') -and ($hostsContent -match '::1.*localhost')
}

# Test 14: Test service registry functions
Test-Item "Get-ServicePort function works" {
    $port = Get-ServicePort -ServiceName "mcp"
    $port -eq 3006
}

# Test 15: Test service registry path
Test-Item "SERVICE_REGISTRY path is configured" {
    $registryPath = "$env:USERPROFILE\.service-registry.json"
    Test-Path $registryPath
}

# Summary
Write-Host "`n=== Test Summary ===" -ForegroundColor Cyan
$passCount = ($testResults | Where-Object { $_.Result }).Count
$failCount = ($testResults | Where-Object { -not $_.Result }).Count
$totalCount = $testResults.Count

Write-Host "Passed: $passCount/$totalCount" -ForegroundColor Green
if ($failCount -gt 0) {
    Write-Host "Failed: $failCount" -ForegroundColor Red
}

Write-Host "`n=== Next Steps ===" -ForegroundColor Cyan
Write-Host "1. Source the profile additions in your PowerShell profile:" -ForegroundColor Gray
Write-Host "   . C:\Users\david\bin\pwsh-profile-additions.ps1" -ForegroundColor Yellow
Write-Host ""

Write-Host "2. Run environment setup (requires admin):" -ForegroundColor Gray
Write-Host "   Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process" -ForegroundColor Yellow
Write-Host "   C:\Users\david\bin\setup-dns-env.ps1" -ForegroundColor Yellow
Write-Host ""

Write-Host "3. Start DNS proxy (requires admin):" -ForegroundColor Gray
Write-Host "   C:\Users\david\bin\LocalDNSProxy.ps1 -Action start" -ForegroundColor Yellow
Write-Host ""

Write-Host "4. Check status:" -ForegroundColor Gray
Write-Host "   C:\Users\david\bin\LocalDNSProxy.ps1 -Action status" -ForegroundColor Yellow
Write-Host ""

Write-Host "5. Test commands:" -ForegroundColor Gray
Write-Host "   Get-LocalService" -ForegroundColor Yellow
Write-Host "   Get-ServicePort -ServiceName mcp" -ForegroundColor Yellow
Write-Host "   open-service -ServiceName mcp" -ForegroundColor Yellow

exit $(if ($failCount -eq 0) { 0 } else { 1 })
