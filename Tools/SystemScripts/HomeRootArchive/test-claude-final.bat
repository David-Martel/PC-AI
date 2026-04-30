@echo off
echo ========================================
echo Testing Claude.exe MCP Configuration
echo ========================================
echo.

REM Test with Windows path
echo [1/3] Testing with Windows path format...
"C:\Users\david\.local\bin\claude.exe" --model claude-opus-4-1-20250805 --mcp-config "C:\Users\david\.claude\mcp.json" --add-dir "C:\users\david\" --add-dir "T:\projects\" --dangerously-skip-permissions --continue 2>nul
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: Windows path format works
) else (
    echo FAILED: Windows path format issue
)
echo.

REM Test MCP server listing
echo [2/3] Testing MCP server listing...
"C:\Users\david\.local\bin\claude.exe" mcp list 2>nul | findstr /i "rust-link rust-sequential-thinking"
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: Rust MCP servers found
) else (
    echo WARNING: Rust MCP servers not listed
)
echo.

REM Test a simple MCP command
echo [3/3] Testing MCP connectivity...
echo {"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05"},"id":1} | "C:\users\david\.local\bin\rust-link-claude.cmd" 2>nul | findstr /B "{"
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: MCP protocol working
) else (
    echo WARNING: MCP protocol issue
)
echo.

echo ========================================
echo Configuration Summary:
echo ========================================
echo MCP Config: C:\Users\david\.claude\mcp.json
echo Rust Servers:
echo   - rust-link-claude.cmd (optimized)
echo   - rust-sequential-thinking-claude.cmd (optimized)
echo   - rust-fs.exe (standard)
echo.
echo To use claude.exe with MCP:
echo claude.exe --mcp-config "C:\Users\david\.claude\mcp.json" --continue
echo ========================================