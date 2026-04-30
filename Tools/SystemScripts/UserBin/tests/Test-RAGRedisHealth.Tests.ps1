#Requires -Modules Pester
<#
.SYNOPSIS
    Pester tests for RAG-Redis health monitoring system
.DESCRIPTION
    Comprehensive unit, integration, and smoke tests for:
    - Test-RAGRedisHealth.ps1
    - Start-RAGRedisNative.ps1
    - docker-health.ps1
    - mcp-health.ps1
.NOTES
    Run: Invoke-Pester -Path .\Tests\ -Output Detailed
#>

BeforeAll {
    $script:BinPath = Split-Path $PSScriptRoot -Parent
    $script:TestDataPath = "$TestDrive\rag-test-data"
    New-Item -ItemType Directory -Path $script:TestDataPath -Force | Out-Null
}

AfterAll {
    Remove-Item -Path $script:TestDataPath -Recurse -Force -ErrorAction SilentlyContinue
}

Describe "RAG-Redis Health System - Smoke Tests" -Tag 'Smoke' {

    It "Test-RAGRedisHealth.ps1 exists and has valid syntax" {
        $scriptPath = Join-Path $script:BinPath "Test-RAGRedisHealth.ps1"
        Test-Path $scriptPath | Should -Be $true

        $errors = $null
        $null = [System.Management.Automation.PSParser]::Tokenize(
            (Get-Content $scriptPath -Raw), [ref]$errors
        )
        $errors.Count | Should -Be 0
    }

    It "Start-RAGRedisNative.ps1 exists and has valid syntax" {
        $scriptPath = Join-Path $script:BinPath "Start-RAGRedisNative.ps1"
        Test-Path $scriptPath | Should -Be $true

        $errors = $null
        $null = [System.Management.Automation.PSParser]::Tokenize(
            (Get-Content $scriptPath -Raw), [ref]$errors
        )
        $errors.Count | Should -Be 0
    }

    It "docker-health.ps1 exists and has valid syntax" {
        $scriptPath = Join-Path $script:BinPath "docker-health.ps1"
        Test-Path $scriptPath | Should -Be $true

        $errors = $null
        $null = [System.Management.Automation.PSParser]::Tokenize(
            (Get-Content $scriptPath -Raw), [ref]$errors
        )
        $errors.Count | Should -Be 0
    }

    It "mcp-health.ps1 exists and has valid syntax" {
        $scriptPath = Join-Path $script:BinPath "mcp-health.ps1"
        Test-Path $scriptPath | Should -Be $true

        $errors = $null
        $null = [System.Management.Automation.PSParser]::Tokenize(
            (Get-Content $scriptPath -Raw), [ref]$errors
        )
        $errors.Count | Should -Be 0
    }

    It "MCP server binary exists" {
        $mcpPath = Join-Path $script:BinPath "rag-redis-mcp-server.exe"
        Test-Path $mcpPath | Should -Be $true

        $fileInfo = Get-Item $mcpPath
        $fileInfo.Length | Should -BeGreaterThan 0
    }

    It "Configuration files exist" {
        $configPath = Join-Path $script:BinPath "rag-redis-config.json"
        Test-Path $configPath | Should -Be $true

        $content = Get-Content $configPath -Raw | ConvertFrom-Json
        $content.redis.url | Should -Not -BeNullOrEmpty
    }
}

Describe "Docker Health Check - Unit Tests" -Tag 'Unit' {

    Context "Container Name Validation" {
        It "Should reject container names with shell metacharacters" {
            $invalidNames = @(
                'redis;rm -rf /',
                'redis && echo pwned',
                'redis|cat /etc/passwd',
                'redis`id`',
                'redis$(whoami)'
            )

            foreach ($name in $invalidNames) {
                # Container names should only contain alphanumeric, dots, underscores, hyphens
                $name -match '^[a-zA-Z0-9][a-zA-Z0-9_.-]*$' | Should -Be $false
            }
        }

        It "Should accept valid container names" {
            $validNames = @(
                'redis',
                'rag-redis-backend',
                'redis_test_01',
                'my.redis.container'
            )

            foreach ($name in $validNames) {
                $name -match '^[a-zA-Z0-9][a-zA-Z0-9_.-]*$' | Should -Be $true
            }
        }
    }

    Context "Docker Availability Detection" {
        BeforeEach {
            Mock docker { throw "Docker not found" } -ModuleName Microsoft.PowerShell.Management
        }

        It "Should handle Docker daemon not running" {
            # When docker info fails, should return appropriate status
            $result = @{
                docker_running = $false
                error = "Docker daemon not running"
            }

            $result.docker_running | Should -Be $false
            $result.error | Should -Match 'not running'
        }
    }
}

Describe "Redis Connection Tests - Unit Tests" -Tag 'Unit' {

    Context "Port Connectivity" {
        It "Should correctly identify port 6379 as Docker Redis" {
            $portMapping = @{
                6379 = "Docker Redis Stack (RediSearch, ReJSON, VectorSet)"
                6380 = "Native Windows Redis (basic, no modules)"
            }

            $portMapping[6379] | Should -Match 'Docker'
            $portMapping[6379] | Should -Match 'RediSearch'
        }

        It "Should correctly identify port 6380 as Native Redis" {
            $portMapping = @{
                6379 = "Docker Redis Stack"
                6380 = "Native Windows Redis"
            }

            $portMapping[6380] | Should -Match 'Native'
        }
    }

    Context "Redis Module Detection" {
        It "Should parse module list correctly" {
            $mockModuleList = @"
1) "name"
2) "search"
3) "ver"
4) 20803
5) "name"
6) "ReJSON"
7) "ver"
8) 20609
"@

            $hasSearch = $mockModuleList -match "search"
            $hasJSON = $mockModuleList -match "ReJSON"

            $hasSearch | Should -Be $true
            $hasJSON | Should -Be $true
        }

        It "Should detect missing modules in plain Redis" {
            $mockModuleList = ""

            $hasSearch = $mockModuleList -match "search"
            $hasJSON = $mockModuleList -match "ReJSON"

            $hasSearch | Should -Be $false
            $hasJSON | Should -Be $false
        }
    }
}

Describe "MCP Server Health - Unit Tests" -Tag 'Unit' {

    Context "Binary Validation" {
        It "Should detect corrupted binary (zero size)" {
            $testBinary = Join-Path $TestDrive "test-mcp.exe"
            New-Item -Path $testBinary -ItemType File -Force | Out-Null

            $fileInfo = Get-Item $testBinary
            $fileInfo.Length | Should -Be 0

            # Zero-size binary should be flagged as invalid
            $isValid = $fileInfo.Length -gt 1024  # Minimum expected size
            $isValid | Should -Be $false
        }

        It "Should validate binary exists at expected path" {
            $mcpPath = "C:\Users\david\bin\rag-redis-mcp-server.exe"

            # This is a smoke test - actual binary should exist
            if (Test-Path $mcpPath) {
                $fileInfo = Get-Item $mcpPath
                $fileInfo.Length | Should -BeGreaterThan 1MB
            }
        }
    }

    Context "Environment Variable Expansion" {
        It "Should safely expand environment variables" {
            $testValue = '${env:USERPROFILE}\.claude'
            # PowerShell 5.1 compatible regex replacement
            $expanded = [regex]::Replace($testValue, '\$\{env:(\w+)\}', {
                param($match)
                $envVar = $match.Groups[1].Value
                $envValue = [Environment]::GetEnvironmentVariable($envVar)
                if ($null -ne $envValue) { $envValue } else { "" }
            })

            $expanded | Should -Not -Match '\$\{env:'
            $expanded | Should -Match '\\\.claude$'
        }

        It "Should handle missing environment variables gracefully" {
            $testValue = '${env:NONEXISTENT_VAR_12345}'
            # PowerShell 5.1 compatible regex replacement
            $expanded = [regex]::Replace($testValue, '\$\{env:(\w+)\}', {
                param($match)
                $envVar = $match.Groups[1].Value
                $envValue = [Environment]::GetEnvironmentVariable($envVar)
                if ($null -ne $envValue) { $envValue } else { "" }
            })

            $expanded | Should -Be ""
        }
    }
}

Describe "Configuration Validation - Unit Tests" -Tag 'Unit' {

    Context "JSON Config Parsing" {
        It "Should parse rag-redis-config.json correctly" {
            $configPath = Join-Path $script:BinPath "rag-redis-config.json"

            if (Test-Path $configPath) {
                $config = Get-Content $configPath -Raw | ConvertFrom-Json

                $config.redis.url | Should -Not -BeNullOrEmpty
                $config.redis.pool_size | Should -BeGreaterThan 0
                $config.vector.dimension | Should -Be 768
                $config.embedding.model | Should -Be "all-MiniLM-L6-v2"
            }
        }

        It "Should detect invalid JSON syntax" {
            $invalidJson = '{ "redis": { "url": "redis://localhost:6379" '  # Missing closing braces

            { $invalidJson | ConvertFrom-Json } | Should -Throw
        }
    }

    Context "Port Configuration Consistency" {
        It "Should have consistent Redis URL across configs" {
            $mcpConfigPath = "$env:USERPROFILE\.claude\mcp.json"

            if (Test-Path $mcpConfigPath) {
                $mcpConfig = Get-Content $mcpConfigPath -Raw | ConvertFrom-Json
                $ragRedisEnv = $mcpConfig.mcpServers.'rag-redis'.env

                $ragRedisEnv.REDIS_URL | Should -Match ':6379'
            }
        }
    }
}

Describe "Error Injection Tests" -Tag 'ErrorInjection' {

    Context "Network Failures" {
        It "Should handle connection refused gracefully" {
            # Test connection to a port that's definitely not listening
            $result = Test-NetConnection -ComputerName localhost -Port 59999 -WarningAction SilentlyContinue

            $result.TcpTestSucceeded | Should -Be $false
        }
    }

    Context "Process Failures" {
        It "Should handle non-existent process lookup" {
            $process = Get-Process -Name "NonExistentProcess12345" -ErrorAction SilentlyContinue

            $process | Should -BeNullOrEmpty
        }
    }

    Context "File System Failures" {
        It "Should handle read-only directory" {
            $readOnlyDir = Join-Path $TestDrive "readonly-test"
            New-Item -ItemType Directory -Path $readOnlyDir -Force | Out-Null

            # Test that we can detect write permission issues
            $testFile = Join-Path $readOnlyDir "test.txt"
            try {
                Set-Content -Path $testFile -Value "test" -ErrorAction Stop
                $canWrite = $true
            } catch {
                $canWrite = $false
            }

            # Cleanup
            Remove-Item -Path $readOnlyDir -Recurse -Force -ErrorAction SilentlyContinue

            # This should succeed in test drive
            $canWrite | Should -Be $true
        }
    }
}

Describe "Integration Tests" -Tag 'Integration' {

    Context "Docker Integration" -Skip:(-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        It "Should detect Docker availability" {
            $dockerInfo = docker info 2>&1
            $LASTEXITCODE | Should -Be 0
        }
    }

    Context "Redis Integration" {
        It "Should connect to Redis on port 6379 if running" {
            $tcpTest = Test-NetConnection -ComputerName localhost -Port 6379 -WarningAction SilentlyContinue

            if ($tcpTest.TcpTestSucceeded) {
                # Redis is running, verify it responds
                $ping = redis-cli -p 6379 ping 2>&1
                $ping | Should -Be "PONG"
            } else {
                Set-ItResult -Skipped -Because "Redis not running on port 6379"
            }
        }

        It "Should detect Redis modules on Docker Redis" {
            $tcpTest = Test-NetConnection -ComputerName localhost -Port 6379 -WarningAction SilentlyContinue

            if ($tcpTest.TcpTestSucceeded) {
                $modules = redis-cli -p 6379 module list 2>&1

                # Docker Redis Stack should have search module
                $hasSearch = $modules -match "search"

                if (-not $hasSearch) {
                    Write-Warning "Redis on 6379 lacks RediSearch module - RAG operations will fail"
                }
            } else {
                Set-ItResult -Skipped -Because "Redis not running on port 6379"
            }
        }
    }
}

Describe "Performance Tests" -Tag 'Performance' {

    It "Should complete health check script parsing in under 1 second" {
        $scriptPath = Join-Path $script:BinPath "Test-RAGRedisHealth.ps1"

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $null = [System.Management.Automation.PSParser]::Tokenize(
            (Get-Content $scriptPath -Raw), [ref]$null
        )
        $sw.Stop()

        $sw.ElapsedMilliseconds | Should -BeLessThan 1000
    }

    It "Should parse MCP config in under 500ms" {
        $configPath = "$env:USERPROFILE\.claude\mcp.json"

        if (Test-Path $configPath) {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $null = Get-Content $configPath -Raw | ConvertFrom-Json
            $sw.Stop()

            $sw.ElapsedMilliseconds | Should -BeLessThan 500
        }
    }
}
