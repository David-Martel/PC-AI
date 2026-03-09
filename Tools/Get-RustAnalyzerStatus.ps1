<#
.SYNOPSIS
    Checks rust-analyzer and lspmux status, configuration, and all LSP clients.
#>

Write-Host "=== rust-analyzer / lspmux Process Status ===" -ForegroundColor Cyan

# Active processes
$raProcs = Get-Process -Name 'rust-analyzer*' -ErrorAction SilentlyContinue
$lspmuxProcs = Get-Process -Name 'lspmux*' -ErrorAction SilentlyContinue

Write-Host "`nrust-analyzer instances: $($raProcs.Count)"
foreach ($p in $raProcs) {
    $wmi = Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)" -ErrorAction SilentlyContinue
    $parentName = ''
    if ($wmi) {
        $parent = Get-Process -Id $wmi.ParentProcessId -ErrorAction SilentlyContinue
        $parentName = if ($parent) { "$($parent.ProcessName) (PID $($parent.Id))" } else { "PID $($wmi.ParentProcessId) (dead)" }
    }
    Write-Host "  PID $($p.Id): WS=$([math]::Round($p.WorkingSet64/1MB,1)) MB, Private=$([math]::Round($p.PrivateMemorySize64/1MB,1)) MB, Handles=$($p.HandleCount), Parent=$parentName"
    Write-Host "    Started: $($p.StartTime)"
    Write-Host "    CPU: $([math]::Round($p.TotalProcessorTime.TotalSeconds,1))s"
    if ($wmi.CommandLine) {
        $cmd = if ($wmi.CommandLine.Length -gt 200) { $wmi.CommandLine.Substring(0,200) + '...' } else { $wmi.CommandLine }
        Write-Host "    CmdLine: $cmd"
    }
}

Write-Host "`nlspmux instances: $($lspmuxProcs.Count)"
foreach ($p in $lspmuxProcs) {
    $wmi = Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)" -ErrorAction SilentlyContinue
    $parentName = ''
    if ($wmi) {
        $parent = Get-Process -Id $wmi.ParentProcessId -ErrorAction SilentlyContinue
        $parentName = if ($parent) { "$($parent.ProcessName) (PID $($parent.Id))" } else { "PID $($wmi.ParentProcessId) (dead)" }
    }
    Write-Host "  PID $($p.Id): WS=$([math]::Round($p.WorkingSet64/1MB,1)) MB, Private=$([math]::Round($p.PrivateMemorySize64/1MB,1)) MB, Parent=$parentName"
    if ($wmi.CommandLine) {
        $cmd = if ($wmi.CommandLine.Length -gt 300) { $wmi.CommandLine.Substring(0,300) + '...' } else { $wmi.CommandLine }
        Write-Host "    CmdLine: $cmd"
    }
}

# Find all potential LSP clients
Write-Host "`n=== Potential LSP Clients Spawning rust-analyzer ===" -ForegroundColor Cyan
$clients = @(
    @{ Name = 'VS Code'; Process = 'Code'; ConfigPath = "$env:APPDATA\Code\User\settings.json" }
    @{ Name = 'Cursor'; Process = 'Cursor'; ConfigPath = "$env:APPDATA\Cursor\User\settings.json" }
    @{ Name = 'Claude Code'; Process = 'claude'; ConfigPath = "$env:USERPROFILE\.claude\settings.json" }
    @{ Name = 'Codex CLI'; Process = 'codex'; ConfigPath = '' }
    @{ Name = 'Neovim'; Process = 'nvim'; ConfigPath = "$env:LOCALAPPDATA\nvim\init.lua" }
    @{ Name = 'Helix'; Process = 'hx'; ConfigPath = "$env:APPDATA\helix\languages.toml" }
    @{ Name = 'Zed'; Process = 'zed'; ConfigPath = "$env:APPDATA\Zed\settings.json" }
    @{ Name = 'WezTerm'; Process = 'wezterm*'; ConfigPath = "$env:USERPROFILE\.wezterm.lua" }
)

foreach ($client in $clients) {
    $procs = Get-Process -Name $client.Process -ErrorAction SilentlyContinue
    $running = $procs.Count -gt 0
    $configExists = if ($client.ConfigPath) { Test-Path $client.ConfigPath } else { $false }
    if ($running) {
        Write-Host "  [RUNNING] $($client.Name): $($procs.Count) process(es)" -ForegroundColor Green
        if ($configExists) {
            Write-Host "    Config: $($client.ConfigPath)" -ForegroundColor DarkGray
        }
    }
}

# Check VS Code rust-analyzer extension settings
Write-Host "`n=== VS Code rust-analyzer Settings ===" -ForegroundColor Cyan
$vsCodeSettings = "$env:APPDATA\Code\User\settings.json"
if (Test-Path $vsCodeSettings) {
    $content = Get-Content $vsCodeSettings -Raw
    $raSettings = @(
        'rust-analyzer.server.path'
        'rust-analyzer.files.excludeDirs'
        'rust-analyzer.checkOnSave'
        'rust-analyzer.cargo.buildScripts.enable'
        'rust-analyzer.procMacro.enable'
        'rust-analyzer.files.watcher'
        'rust-analyzer.lru.capacity'
        'rust-analyzer.cargo.sysroot'
    )
    foreach ($setting in $raSettings) {
        if ($content -match [regex]::Escape($setting)) {
            $line = ($content -split "`n" | Where-Object { $_ -match [regex]::Escape($setting) } | Select-Object -First 1).Trim()
            Write-Host "  $line"
        } else {
            Write-Host "  $setting`: (not set)" -ForegroundColor DarkGray
        }
    }
} else {
    Write-Host "  No VS Code settings found"
}

# Check for lspmux configuration
Write-Host "`n=== lspmux Configuration ===" -ForegroundColor Cyan
$lspmuxConfigs = @(
    "$env:APPDATA\lspmux\config\config.toml"
    "$env:USERPROFILE\.config\lspmux\config.toml"
    "$env:APPDATA\lspmux\config.toml"
    "$env:LOCALAPPDATA\lspmux\config.toml"
    "$env:USERPROFILE\.lspmux.toml"
)
$foundConfig = $false
foreach ($path in $lspmuxConfigs) {
    if (Test-Path $path) {
        Write-Host "  Found: $path" -ForegroundColor Green
        Get-Content $path | ForEach-Object { Write-Host "    $_" }
        $foundConfig = $true
    }
}
if (-not $foundConfig) {
    Write-Host "  No lspmux config found at any standard location" -ForegroundColor Yellow
    Write-Host "  Searched: $($lspmuxConfigs -join ', ')"
}

# lspmux server health check
Write-Host "`n=== lspmux Server Health ===" -ForegroundColor Cyan
$lspmuxRunning = $lspmuxProcs.Count -gt 0
if ($lspmuxRunning) {
    Write-Host "  Server: RUNNING (PID $($lspmuxProcs[0].Id))" -ForegroundColor Green
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.Connect('127.0.0.1', 27631)
        $tcp.Close()
        Write-Host "  TCP 127.0.0.1:27631: LISTENING" -ForegroundColor Green
    } catch {
        Write-Host "  TCP 127.0.0.1:27631: NOT RESPONDING" -ForegroundColor Red
    }
} else {
    Write-Host "  Server: NOT RUNNING" -ForegroundColor Red
    Write-Host "  Start with: & `"$env:USERPROFILE\.local\bin\Start-LspmuxServer.ps1`"" -ForegroundColor Yellow
}

# Check scheduled task
$task = Get-ScheduledTask -TaskName 'LspmuxServer' -ErrorAction SilentlyContinue
if ($task) {
    Write-Host "  Scheduled Task: $($task.State)" -ForegroundColor $(if ($task.State -eq 'Ready') { 'Green' } else { 'Yellow' })
} else {
    Write-Host "  Scheduled Task: NOT REGISTERED" -ForegroundColor Yellow
}

# Check routing: is rust-analyzer.cmd routing through lspmux?
Write-Host "`n=== rust-analyzer Routing ===" -ForegroundColor Cyan
$raCmd = "$env:USERPROFILE\bin\rust-analyzer.cmd"
if (Test-Path $raCmd) {
    $content = Get-Content $raCmd -Raw
    if ($content -match 'lspmux') {
        Write-Host "  $raCmd -> lspmux (CONFIGURED)" -ForegroundColor Green
    } else {
        Write-Host "  $raCmd -> CargoTools wrapper (legacy)" -ForegroundColor Yellow
    }
}
$lspmuxRa = "$env:USERPROFILE\.local\bin\lspmux-ra.cmd"
if (Test-Path $lspmuxRa) {
    Write-Host "  ${lspmuxRa}: EXISTS" -ForegroundColor Green
}

# Check Claude Code MCP / LSP settings
Write-Host "`n=== Claude Code LSP Settings ===" -ForegroundColor Cyan
$claudeSettings = "$env:USERPROFILE\.claude\settings.json"
if (Test-Path $claudeSettings) {
    $content = Get-Content $claudeSettings -Raw
    if ($content -match 'rust-analyzer|lspmux|LSP') {
        Write-Host "  References found in settings.json"
        $content -split "`n" | Where-Object { $_ -match 'rust-analyzer|lspmux|LSP' } | ForEach-Object { Write-Host "    $($_.Trim())" }
    } else {
        Write-Host "  No rust-analyzer/lspmux references in Claude settings"
    }
}

# Workspace-level .vscode/settings.json files
Write-Host "`n=== Workspace rust-analyzer Configs (Rust projects) ===" -ForegroundColor Cyan
$rustWorkspaces = @(
    'C:\codedev\PC_AI'
    'T:\projects\serena-source'
    'T:\projects\rust-mcp'
    'C:\codedev\codex'
    'C:\codedev\gitoxide'
    'C:\codedev\deepwiki-rs'
    'C:\codedev\litho-book'
)
foreach ($ws in $rustWorkspaces) {
    $wsSettings = Join-Path $ws '.vscode\settings.json'
    if (Test-Path $wsSettings) {
        $content = Get-Content $wsSettings -Raw
        if ($content -match 'rust-analyzer') {
            Write-Host "  $ws\.vscode\settings.json: HAS rust-analyzer config" -ForegroundColor Green
            $content -split "`n" | Where-Object { $_ -match 'rust-analyzer' } | ForEach-Object { Write-Host "    $($_.Trim())" }
        }
    } elseif (Test-Path $ws) {
        Write-Host "  ${ws}: no .vscode/settings.json" -ForegroundColor DarkGray
    }
}

# lspmux binary path check
Write-Host "`n=== lspmux Binary Location ===" -ForegroundColor Cyan
$lspmuxPaths = @(
    "$env:USERPROFILE\.cargo\bin\lspmux.exe"
    "T:\RustCache\cargo-home\bin\lspmux.exe"
)
foreach ($path in $lspmuxPaths) {
    if (Test-Path $path) {
        $ver = & $path --version 2>&1
        $size = [math]::Round((Get-Item $path).Length / 1MB, 1)
        Write-Host "  $path ($size MB) - $ver" -ForegroundColor Green
    }
}

# Process tree: what spawned each rust-analyzer?
Write-Host "`n=== rust-analyzer Process Tree ===" -ForegroundColor Cyan
foreach ($p in $raProcs) {
    Write-Host "  rust-analyzer PID $($p.Id):" -ForegroundColor Yellow
    $current = $p.Id
    $depth = 0
    while ($current -and $depth -lt 8) {
        $wmi = Get-CimInstance Win32_Process -Filter "ProcessId=$current" -ErrorAction SilentlyContinue
        if (-not $wmi) { break }
        $indent = '    ' + ('  ' * $depth)
        $shortCmd = if ($wmi.CommandLine.Length -gt 100) { $wmi.CommandLine.Substring(0,100) + '...' } else { $wmi.CommandLine }
        Write-Host "${indent}PID $current ($($wmi.Name)): $shortCmd"
        $current = $wmi.ParentProcessId
        $depth++
    }
}
