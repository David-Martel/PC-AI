# Simple Vertex AI MCP Integration
param([switch]$DryRun)

$ConfigPath = "C:\Users\david\.claude\mcp.json"

Write-Host "Adding Vertex AI servers to MCP configuration..." -ForegroundColor Green

# Read config
$config = Get-Content $ConfigPath -Raw | ConvertFrom-Json

# Define servers
$servers = @{
    "vertex-ai-code-reviewer" = 8000
    "vertex-ai-data-analyzer" = 8001
    "vertex-ai-document-processor" = 8002
    "vertex-ai-image-analyzer" = 8003
    "vertex-ai-chat-assistant" = 8004
    "vertex-ai-orchestrator" = 8005
}

# Add each server
foreach ($name in $servers.Keys) {
    $port = $servers[$name]

    if ($config.mcpServers.PSObject.Properties.Name -contains $name) {
        Write-Host "Server $name already exists, skipping..." -ForegroundColor Yellow
        continue
    }

    $serverConfig = [PSCustomObject]@{
        transport = [PSCustomObject]@{
            type = "sse"
            url = "http://localhost:$port"
        }
        env = [PSCustomObject]@{
            "GOOGLE_APPLICATION_CREDENTIALS" = "/home/david/.auth/business/service-account-key.json"
            "VERTEX_PROJECT_ID" = "auricleinc-gemini"
            "VERTEX_LOCATION" = "us-central1"
            "LOG_LEVEL" = "ERROR"
            "MCP_SERVER_NAME" = $name
        }
    }

    $config.mcpServers | Add-Member -MemberType NoteProperty -Name $name -Value $serverConfig
    Write-Host "Added server: $name on port $port" -ForegroundColor Green
}

# Save or preview
if ($DryRun) {
    Write-Host "`nDRY RUN - Would save this configuration:" -ForegroundColor Magenta
    $config | ConvertTo-Json -Depth 10
} else {
    $backup = "$ConfigPath.backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    Copy-Item $ConfigPath $backup
    Write-Host "Backup created: $backup" -ForegroundColor Yellow

    $config | ConvertTo-Json -Depth 10 | Set-Content $ConfigPath -Encoding UTF8
    Write-Host "Configuration updated successfully!" -ForegroundColor Green
}