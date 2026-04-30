# Vertex AI MCP Server Integration Script
# Integrates WSL-based Vertex AI servers into Windows MCP configuration

param(
    [switch]$DryRun,
    [switch]$TestConnectivity,
    [string]$ConfigPath = "C:\Users\david\.claude\mcp.json"
)

Write-Host "🚀 Vertex AI MCP Server Integration Script" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Test WSL connectivity
Write-Host "`n📡 Testing WSL connectivity..." -ForegroundColor Yellow
try {
    $wslTest = wsl echo "WSL connection test" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "WSL is not accessible. Please ensure WSL is running."
        exit 1
    }
    Write-Host "✅ WSL connectivity confirmed" -ForegroundColor Green
} catch {
    Write-Error "Failed to test WSL connectivity: $_"
    exit 1
}

# Define Vertex AI server configurations
$vertexAiServers = @{
    "vertex-ai-code-reviewer" = @{
        port = 8000
        description = "Vertex AI Code Reviewer MCP Server"
        env = @{
            "GOOGLE_APPLICATION_CREDENTIALS" = "/home/david/.auth/business/service-account-key.json"
            "VERTEX_PROJECT_ID" = "auricleinc-gemini"
            "VERTEX_LOCATION" = "us-central1"
            "LOG_LEVEL" = "ERROR"
            "MCP_SERVER_NAME" = "vertex-ai-code-reviewer"
        }
    }
    "vertex-ai-data-analyzer" = @{
        port = 8001
        description = "Vertex AI Data Analyzer MCP Server"
        env = @{
            "GOOGLE_APPLICATION_CREDENTIALS" = "/home/david/.auth/business/service-account-key.json"
            "VERTEX_PROJECT_ID" = "auricleinc-gemini"
            "VERTEX_LOCATION" = "us-central1"
            "LOG_LEVEL" = "ERROR"
            "MCP_SERVER_NAME" = "vertex-ai-data-analyzer"
        }
    }
    "vertex-ai-document-processor" = @{
        port = 8002
        description = "Vertex AI Document Processor MCP Server"
        env = @{
            "GOOGLE_APPLICATION_CREDENTIALS" = "/home/david/.auth/business/service-account-key.json"
            "VERTEX_PROJECT_ID" = "auricleinc-gemini"
            "VERTEX_LOCATION" = "us-central1"
            "LOG_LEVEL" = "ERROR"
            "MCP_SERVER_NAME" = "vertex-ai-document-processor"
        }
    }
    "vertex-ai-image-analyzer" = @{
        port = 8003
        description = "Vertex AI Image Analyzer MCP Server"
        env = @{
            "GOOGLE_APPLICATION_CREDENTIALS" = "/home/david/.auth/business/service-account-key.json"
            "VERTEX_PROJECT_ID" = "auricleinc-gemini"
            "VERTEX_LOCATION" = "us-central1"
            "LOG_LEVEL" = "ERROR"
            "MCP_SERVER_NAME" = "vertex-ai-image-analyzer"
        }
    }
    "vertex-ai-chat-assistant" = @{
        port = 8004
        description = "Vertex AI Chat Assistant MCP Server"
        env = @{
            "GOOGLE_APPLICATION_CREDENTIALS" = "/home/david/.auth/business/service-account-key.json"
            "VERTEX_PROJECT_ID" = "auricleinc-gemini"
            "VERTEX_LOCATION" = "us-central1"
            "LOG_LEVEL" = "ERROR"
            "MCP_SERVER_NAME" = "vertex-ai-chat-assistant"
        }
    }
    "vertex-ai-orchestrator" = @{
        port = 8005
        description = "Vertex AI Orchestrator MCP Server"
        env = @{
            "GOOGLE_APPLICATION_CREDENTIALS" = "/home/david/.auth/business/service-account-key.json"
            "VERTEX_PROJECT_ID" = "auricleinc-gemini"
            "VERTEX_LOCATION" = "us-central1"
            "LOG_LEVEL" = "ERROR"
            "MCP_SERVER_NAME" = "vertex-ai-orchestrator"
        }
    }
}

# Test connectivity to Vertex AI servers
if ($TestConnectivity) {
    Write-Host "`n🔍 Testing connectivity to Vertex AI servers..." -ForegroundColor Yellow

    foreach ($serverName in $vertexAiServers.Keys) {
        $port = $vertexAiServers[$serverName].port
        Write-Host "  Testing $serverName on port $port..." -NoNewline

        try {
            $result = Test-NetConnection -ComputerName "localhost" -Port $port -WarningAction SilentlyContinue
            if ($result.TcpTestSucceeded) {
                Write-Host " ✅ Connected" -ForegroundColor Green

                # Test health endpoint if available
                try {
                    $healthCheck = Invoke-RestMethod -Uri "http://localhost:$port/health" -Method GET -TimeoutSec 5 -ErrorAction SilentlyContinue
                    Write-Host "    Health check: ✅ OK" -ForegroundColor Green
                } catch {
                    Write-Host "    Health check: ⚠️  No health endpoint" -ForegroundColor Yellow
                }
            } else {
                Write-Host " ❌ Not accessible" -ForegroundColor Red
            }
        } catch {
            Write-Host " ❌ Connection failed: $_" -ForegroundColor Red
        }
    }

    if (-not $DryRun) {
        $continue = Read-Host "`nDo you want to continue with integration? (y/N)"
        if ($continue -ne "y" -and $continue -ne "Y") {
            Write-Host "Integration cancelled." -ForegroundColor Yellow
            exit 0
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
$backupPath = "$ConfigPath.backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
Write-Host "`n💾 Creating backup at: $backupPath" -ForegroundColor Yellow
Copy-Item $ConfigPath $backupPath
Write-Host "✅ Backup created successfully" -ForegroundColor Green

# Add Vertex AI servers to configuration
Write-Host "`n🔧 Adding Vertex AI servers to MCP configuration..." -ForegroundColor Yellow

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
        env = [PSCustomObject]@{}
    }

    # Add environment variables
    foreach ($envKey in $serverConfig.env.Keys) {
        $newServerConfig.env | Add-Member -MemberType NoteProperty -Name $envKey -Value $serverConfig.env[$envKey]
    }

    # Add server to configuration
    $mcpConfig.mcpServers | Add-Member -MemberType NoteProperty -Name $serverName -Value $newServerConfig
    Write-Host " ✅ Added" -ForegroundColor Green
}

# Save updated configuration
if ($DryRun) {
    Write-Host "`n🔍 DRY RUN - Configuration changes:" -ForegroundColor Magenta
    $mcpConfig | ConvertTo-Json -Depth 10 | Write-Host
    Write-Host "`nDry run completed. No changes were made." -ForegroundColor Magenta
} else {
    Write-Host "`n💾 Saving updated MCP configuration..." -ForegroundColor Yellow
    try {
        $mcpConfig | ConvertTo-Json -Depth 10 | Set-Content $ConfigPath -Encoding UTF8
        Write-Host "✅ MCP configuration updated successfully" -ForegroundColor Green
    } catch {
        Write-Error "Failed to save MCP configuration: $_"
        # Restore backup
        Copy-Item $backupPath $ConfigPath
        Write-Host "🔄 Configuration restored from backup" -ForegroundColor Yellow
        exit 1
    }
}

# Create WSL server startup script
Write-Host "`n🚀 Creating WSL server startup script..." -ForegroundColor Yellow
$startupScript = @'
#!/bin/bash
# Vertex AI MCP Servers Startup Script
# Generated by integrate-vertex-ai-servers.ps1

echo "🚀 Starting Vertex AI MCP Servers..."

# Set common environment variables
export GOOGLE_APPLICATION_CREDENTIALS="/home/david/.auth/business/service-account-key.json"
export VERTEX_PROJECT_ID="auricleinc-gemini"
export VERTEX_LOCATION="us-central1"
export LOG_LEVEL="ERROR"

# Function to start a server
start_server() {
    local server_name=$1
    local port=$2
    local script_path=$3

    echo "Starting $server_name on port $port..."

    # Check if port is already in use
    if netstat -tuln | grep -q ":$port "; then
        echo "⚠️  Port $port is already in use, skipping $server_name"
        return
    fi

    # Start server in background
    export MCP_SERVER_NAME="$server_name"
    cd /home/david/agents
    nohup uv run python -m $script_path --port $port > "/tmp/$server_name.log" 2>&1 &
    echo $! > "/tmp/$server_name.pid"
    echo "✅ Started $server_name (PID: $!)"
}

# Start all Vertex AI servers
start_server "vertex-ai-code-reviewer" 8000 "vertex_ai_servers.code_reviewer"
start_server "vertex-ai-data-analyzer" 8001 "vertex_ai_servers.data_analyzer"
start_server "vertex-ai-document-processor" 8002 "vertex_ai_servers.document_processor"
start_server "vertex-ai-image-analyzer" 8003 "vertex_ai_servers.image_analyzer"
start_server "vertex-ai-chat-assistant" 8004 "vertex_ai_servers.chat_assistant"
start_server "vertex-ai-orchestrator" 8005 "vertex_ai_servers.orchestrator"

echo "🎉 Vertex AI MCP Servers startup complete!"
echo "📊 Server status:"
ps aux | grep -E "(vertex-ai|python -m vertex_ai_servers)" | grep -v grep

echo ""
echo "📝 Log files are in /tmp/"
echo "🔍 To check server status: ps aux | grep vertex-ai"
echo "🛑 To stop all servers: pkill -f 'vertex-ai'"
'@

$wslStartupPath = "start-vertex-ai-servers.sh"
$startupScript | Set-Content $wslStartupPath -Encoding UTF8
Write-Host "✅ WSL startup script created: $wslStartupPath" -ForegroundColor Green

# Summary
Write-Host "`n📋 Integration Summary:" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
Write-Host "✅ Backup created: $backupPath" -ForegroundColor Green
Write-Host "✅ Added $($vertexAiServers.Count) Vertex AI servers to MCP configuration" -ForegroundColor Green
Write-Host "✅ Created WSL startup script: $wslStartupPath" -ForegroundColor Green

Write-Host "`n📚 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Copy startup script to WSL: wsl cp '$wslStartupPath' '/home/david/'" -ForegroundColor White
Write-Host "2. Make script executable: wsl chmod +x '/home/david/$wslStartupPath'" -ForegroundColor White
Write-Host "3. Start servers: wsl '/home/david/$wslStartupPath'" -ForegroundColor White
Write-Host "4. Test MCP servers in Claude Code" -ForegroundColor White

Write-Host "`n🔧 Troubleshooting:" -ForegroundColor Cyan
Write-Host "- Check server logs: wsl tail -f /tmp/vertex-ai-*.log" -ForegroundColor White
Write-Host "- Check server processes: wsl ps aux | grep vertex-ai" -ForegroundColor White
Write-Host "- Stop servers: wsl pkill -f vertex-ai" -ForegroundColor White
Write-Host "- Restore backup if needed: Copy-Item '`$backupPath' '`$ConfigPath'" -ForegroundColor White

if (-not $DryRun) {
    Write-Host "`n✨ Integration completed successfully!" -ForegroundColor Green
}