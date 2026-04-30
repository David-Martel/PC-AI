#Requires -RunAsAdministrator
<#
.SYNOPSIS
Setup environment variables and shortcuts for DNS proxy and service registry.

.DESCRIPTION
Configures environment variables that will be available system-wide for accessing
local services through DNS proxy or service registry.
#>

$ErrorActionPreference = 'Stop'

Write-Host "Setting up DNS Proxy and Local Service Environment Variables..." -ForegroundColor Cyan

# Define environment variables
$envVars = @{
    'DNS_PROXY_PORT' = '5354'
    'DNS_PROXY_HOST' = 'localhost'
    'DNS_PROXY_UPSTREAM' = '8.8.8.8,1.1.1.1'
    'LOCAL_SERVICE_REGISTRY' = "$env:USERPROFILE\.service-registry.json"
    'LOCAL_SERVICE_PORT_BASE' = '3000'
    'VERTEX_CODE_REVIEWER_PORT' = '3001'
    'VERTEX_CODE_GENERATOR_PORT' = '3002'
    'VERTEX_MASTER_ARCHITECT_PORT' = '3003'
    'VERTEX_WORKSPACE_ANALYZER_PORT' = '3004'
    'VERTEX_DOC_GENERATOR_PORT' = '3005'
    'MCP_PORT' = '3006'
    'GEMINI_CLI_PORT' = '3007'
    'VERTEX_CODE_REVIEWER_URL' = 'http://localhost:3001'
    'VERTEX_CODE_GENERATOR_URL' = 'http://localhost:3002'
    'VERTEX_MASTER_ARCHITECT_URL' = 'http://localhost:3003'
    'VERTEX_WORKSPACE_ANALYZER_URL' = 'http://localhost:3004'
    'VERTEX_DOC_GENERATOR_URL' = 'http://localhost:3005'
    'MCP_LOCAL_URL' = 'http://localhost:3006'
    'GEMINI_CLI_URL' = 'http://localhost:3007'
}

# Set system environment variables
foreach ($name in $envVars.Keys) {
    try {
        [System.Environment]::SetEnvironmentVariable($name, $envVars[$name], 'Machine')
        Write-Host "✓ Set $name = $($envVars[$name])" -ForegroundColor Green
    } catch {
        Write-Warning "Failed to set $name : $_"
    }
}

Write-Host "`n✓ Environment variables configured" -ForegroundColor Green
Write-Host "Note: You may need to restart your shell/IDE for changes to take effect.`n" -ForegroundColor Yellow
