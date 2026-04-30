# Vertex AI MCP Server Integration Script
# Integrates WSL-based Vertex AI servers into Windows MCP configuration

param(
    [switch]$DryRun,
    [switch]$TestConnectivity,
    [string]$ConfigPath = "C:\Users\david\.claude\mcp.json"
)

Write-Host "🚀 Vertex AI MCP Server Integration Script" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Define Vertex AI server configurations
$vertexAiServers = @{
    "vertex-ai-code-reviewer" = @{
        port = 8000
        description = "Vertex AI Code Reviewer MCP Server"
    }
    "vertex-ai-data-analyzer" = @{
        port = 8001
        description = "Vertex AI Data Analyzer MCP Server"
    }
    "vertex-ai-document-processor" = @{
        port = 8002
        description = "Vertex AI Document Processor MCP Server"
    }
    "vertex-ai-image-analyzer" = @{
        port = 8003
        description = "Vertex AI Image Analyzer MCP Server"
    }
    "vertex-ai-chat-assistant" = @{
        port = 8004
        description = "Vertex AI Chat Assistant MCP Server"
    }
    "vertex-ai-orchestrator" = @{
        port = 8005
        description = "Vertex AI Orchestrator MCP Server"
    }
}

# Test connectivity to Vertex AI servers if requested
if ($TestConnectivity) {
    Write-Host "`n🔍 Testing connectivity to Vertex AI servers..." -ForegroundColor Yellow

    foreach ($serverName in $vertexAiServers.Keys) {
        $port = $vertexAiServers[$serverName].port
        Write-Host "  Testing $serverName on port $port..." -NoNewline

        try {
            $result = Test-NetConnection -ComputerName "localhost" -Port $port -WarningAction SilentlyContinue
            if ($result.TcpTestSucceeded) {
                Write-Host " ✅ Connected" -ForegroundColor Green
            } else {
                Write-Host " ❌ Not accessible" -ForegroundColor Red
            }
        } catch {
            Write-Host " ❌ Connection failed" -ForegroundColor Red
        }
    }
}

# Read existing MCP configuration
Write-Host "`n📖 Reading existing MCP configuration..." -ForegroundColor Yellow
if (-not (Test-Path $ConfigPath)) {
    Write-Error "MCP configuration file not found at: $ConfigPath"
    exit 1
}

try {
    $mcpConfig = Get-Content $ConfigPath -Raw | ConvertFrom-Json
    Write-Host "✅ MCP configuration loaded successfully" -ForegroundColor Green
    Write-Host "   Current servers: $($mcpConfig.mcpServers.PSObject.Properties.Name -join ', ')" -ForegroundColor Cyan
} catch {
    Write-Error "Failed to parse MCP configuration: $_"
    exit 1
}

# Backup existing configuration
if (-not $DryRun) {
    $backupPath = "$ConfigPath.backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    Write-Host "`n💾 Creating backup at: $backupPath" -ForegroundColor Yellow
    Copy-Item $ConfigPath $backupPath
    Write-Host "✅ Backup created successfully" -ForegroundColor Green
}

# Add Vertex AI servers to configuration
Write-Host "`n🔧 Adding Vertex AI servers to MCP configuration..." -ForegroundColor Yellow

$serversAdded = 0
foreach ($serverName in $vertexAiServers.Keys) {
    $serverConfig = $vertexAiServers[$serverName]
    Write-Host "  Adding $serverName..." -NoNewline

    # Check if server already exists
    if ($mcpConfig.mcpServers.PSObject.Properties.Name -contains $serverName) {
        Write-Host " ⚠️  Already exists (skipping)" -ForegroundColor Yellow
        continue
    }

    # Create server configuration for SSE transport
    $newServerConfig = [PSCustomObject]@{
        transport = [PSCustomObject]@{
            type = "sse"
            url = "http://localhost:$($serverConfig.port)"
        }
        env = [PSCustomObject]@{
            "GOOGLE_APPLICATION_CREDENTIALS" = "/home/david/.auth/business/service-account-key.json"
            "VERTEX_PROJECT_ID" = "auricleinc-gemini"
            "VERTEX_LOCATION" = "us-central1"
            "LOG_LEVEL" = "ERROR"
            "MCP_SERVER_NAME" = $serverName
        }
    }

    # Add server to configuration
    $mcpConfig.mcpServers | Add-Member -MemberType NoteProperty -Name $serverName -Value $newServerConfig
    Write-Host " ✅ Added" -ForegroundColor Green
    $serversAdded++
}

# Display configuration preview
if ($DryRun) {
    Write-Host "`n🔍 DRY RUN - Configuration preview:" -ForegroundColor Magenta
    Write-Host "Would add the following servers:" -ForegroundColor Magenta
    foreach ($serverName in $vertexAiServers.Keys) {
        if (-not ($mcpConfig.mcpServers.PSObject.Properties.Name -contains $serverName)) {
            $port = $vertexAiServers[$serverName].port
            Write-Host "  - $serverName (http://localhost:$port)" -ForegroundColor Magenta
        }
    }
    Write-Host "`nDry run completed. No changes were made." -ForegroundColor Magenta
    exit 0
}

# Save updated configuration
if ($serversAdded -gt 0) {
    Write-Host "`n💾 Saving updated MCP configuration..." -ForegroundColor Yellow
    try {
        $mcpConfig | ConvertTo-Json -Depth 10 | Set-Content $ConfigPath -Encoding UTF8
        Write-Host "✅ MCP configuration updated successfully" -ForegroundColor Green
    } catch {
        Write-Error "Failed to save MCP configuration: $_"
        # Restore backup if it exists
        if (Test-Path $backupPath) {
            Copy-Item $backupPath $ConfigPath
            Write-Host "🔄 Configuration restored from backup" -ForegroundColor Yellow
        }
        exit 1
    }
}

# Create WSL server startup script
Write-Host "`n🚀 Creating WSL server startup script..." -ForegroundColor Yellow
$bashScript = @'
#!/bin/bash
# Vertex AI MCP Servers Startup Script
echo "🚀 Starting Vertex AI MCP Servers..."

export GOOGLE_APPLICATION_CREDENTIALS="/home/david/.auth/business/service-account-key.json"
export VERTEX_PROJECT_ID="auricleinc-gemini"
export VERTEX_LOCATION="us-central1"
export LOG_LEVEL="ERROR"

cd /home/david/agents

echo "Starting vertex-ai-code-reviewer on port 8000..."
export MCP_SERVER_NAME="vertex-ai-code-reviewer"
nohup uv run python -m vertex_ai_servers.code_reviewer --port 8000 > /tmp/vertex-ai-code-reviewer.log 2>&1 &

echo "Starting vertex-ai-data-analyzer on port 8001..."
export MCP_SERVER_NAME="vertex-ai-data-analyzer"
nohup uv run python -m vertex_ai_servers.data_analyzer --port 8001 > /tmp/vertex-ai-data-analyzer.log 2>&1 &

echo "Starting vertex-ai-document-processor on port 8002..."
export MCP_SERVER_NAME="vertex-ai-document-processor"
nohup uv run python -m vertex_ai_servers.document_processor --port 8002 > /tmp/vertex-ai-document-processor.log 2>&1 &

echo "Starting vertex-ai-image-analyzer on port 8003..."
export MCP_SERVER_NAME="vertex-ai-image-analyzer"
nohup uv run python -m vertex_ai_servers.image_analyzer --port 8003 > /tmp/vertex-ai-image-analyzer.log 2>&1 &

echo "Starting vertex-ai-chat-assistant on port 8004..."
export MCP_SERVER_NAME="vertex-ai-chat-assistant"
nohup uv run python -m vertex_ai_servers.chat_assistant --port 8004 > /tmp/vertex-ai-chat-assistant.log 2>&1 &

echo "Starting vertex-ai-orchestrator on port 8005..."
export MCP_SERVER_NAME="vertex-ai-orchestrator"
nohup uv run python -m vertex_ai_servers.orchestrator --port 8005 > /tmp/vertex-ai-orchestrator.log 2>&1 &

echo "✅ All Vertex AI MCP Servers started!"
echo "📝 Check logs in /tmp/ directory"
echo "🔍 Check status: ps aux | grep vertex-ai"
'@

$wslStartupPath = "C:\Users\david\start-vertex-ai-servers.sh"
$bashScript | Set-Content $wslStartupPath -Encoding UTF8
Write-Host "✅ WSL startup script created: $wslStartupPath" -ForegroundColor Green

# Summary
Write-Host "`n📋 Integration Summary:" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
if (Test-Path $backupPath) {
    Write-Host "✅ Backup created: $backupPath" -ForegroundColor Green
}
Write-Host "✅ Added $serversAdded Vertex AI servers to MCP configuration" -ForegroundColor Green
Write-Host "✅ Created WSL startup script: $wslStartupPath" -ForegroundColor Green

Write-Host "`n📚 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Copy startup script to WSL: wsl cp `"$wslStartupPath`" `"/home/david/`"" -ForegroundColor White
Write-Host "2. Make script executable: wsl chmod +x `"/home/david/start-vertex-ai-servers.sh`"" -ForegroundColor White
Write-Host "3. Start servers: wsl `"/home/david/start-vertex-ai-servers.sh`"" -ForegroundColor White
Write-Host "4. Test MCP servers in Claude Code" -ForegroundColor White

Write-Host "`n🔧 Troubleshooting:" -ForegroundColor Cyan
Write-Host "- Check server logs: wsl tail -f /tmp/vertex-ai-*.log" -ForegroundColor White
Write-Host "- Check server processes: wsl ps aux | grep vertex-ai" -ForegroundColor White
Write-Host "- Stop servers: wsl pkill -f vertex-ai" -ForegroundColor White
if (Test-Path $backupPath) {
    Write-Host "- Restore backup: Copy-Item `"$backupPath`" `"$ConfigPath`"" -ForegroundColor White
}

Write-Host "`nIntegration completed successfully!" -ForegroundColor Green