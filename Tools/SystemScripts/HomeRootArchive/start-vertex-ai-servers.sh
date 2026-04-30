#!/bin/bash
# Vertex AI MCP Servers Startup Script
# This script starts all Vertex AI MCP servers in WSL for Windows integration

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
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        echo "⚠️  Port $port is already in use, skipping $server_name"
        return
    fi

    # Start server in background
    export MCP_SERVER_NAME="$server_name"
    cd /home/david/agents

    # Start with nohup and redirect output
    nohup uv run python -m $script_path --port $port > "/tmp/$server_name.log" 2>&1 &
    local pid=$!
    echo $pid > "/tmp/$server_name.pid"
    echo "✅ Started $server_name (PID: $pid)"

    # Give the server a moment to start
    sleep 1
}

# Create agents directory if it doesn't exist
mkdir -p /home/david/agents
cd /home/david/agents

# Start all Vertex AI servers
start_server "vertex-ai-code-reviewer" 8000 "vertex_ai_servers.code_reviewer"
start_server "vertex-ai-data-analyzer" 8001 "vertex_ai_servers.data_analyzer"
start_server "vertex-ai-document-processor" 8002 "vertex_ai_servers.document_processor"
start_server "vertex-ai-image-analyzer" 8003 "vertex_ai_servers.image_analyzer"
start_server "vertex-ai-chat-assistant" 8004 "vertex_ai_servers.chat_assistant"
start_server "vertex-ai-orchestrator" 8005 "vertex_ai_servers.orchestrator"

echo ""
echo "🎉 Vertex AI MCP Servers startup complete!"
echo ""
echo "📊 Server status:"
ps aux | grep -E "(vertex-ai|python -m vertex_ai_servers)" | grep -v grep || echo "No servers found running"

echo ""
echo "📝 Log files are in /tmp/ directory:"
ls -la /tmp/vertex-ai-*.log 2>/dev/null || echo "No log files found yet"

echo ""
echo "🔍 Useful commands:"
echo "  Check status:      ps aux | grep vertex-ai"
echo "  View logs:         tail -f /tmp/vertex-ai-*.log"
echo "  Stop all servers:  pkill -f 'vertex-ai'"
echo "  Test connectivity: curl http://localhost:8000/health"

echo ""
echo "⚡ Ready for Claude Code integration!"