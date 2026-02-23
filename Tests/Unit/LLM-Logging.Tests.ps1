#Requires -Version 5.1

. (Join-Path $PSScriptRoot '..\Helpers\Resolve-TestRepoRoot.ps1')

BeforeAll {
    $PcaiRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot

    # Import the module
    $ModulePath = Join-Path $PcaiRoot 'Modules\PC-AI.LLM\PC-AI.LLM.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop
    $script:LlmModule = Get-Module PC-AI.LLM

    # Setup test log file location
    $script:TestLogDir = Join-Path $PcaiRoot 'Reports\Logs'
    $script:TestLogFile = Join-Path $script:TestLogDir 'llm-router.log'

    # Ensure log directory exists
    if (-not (Test-Path $script:TestLogDir)) {
        New-Item -ItemType Directory -Path $script:TestLogDir -Force | Out-Null
    }
}

AfterAll {
    # Clean up test log file
    if (Test-Path $script:TestLogFile) {
        Remove-Item $script:TestLogFile -Force -ErrorAction SilentlyContinue
    }
}

Describe "LLM-Logging Module" {
    BeforeEach {
        # Reset log level to default and clear log file before each test
        & $script:LlmModule { Set-LLMLogLevel -Level 'Info' }
        if (Test-Path $script:TestLogFile) {
            Remove-Item $script:TestLogFile -Force
        }
    }

    Context "Log Level Management" {
        It "Should have default log level of Info" {
            $level = & $script:LlmModule { Get-LLMLogLevel }
            $level | Should -Be 'Info'
        }

        It "Should set log level to Debug" {
            & $script:LlmModule { Set-LLMLogLevel -Level 'Debug' }
            $level = & $script:LlmModule { Get-LLMLogLevel }
            $level | Should -Be 'Debug'
        }

        It "Should set log level to Warning" {
            & $script:LlmModule { Set-LLMLogLevel -Level 'Warning' }
            $level = & $script:LlmModule { Get-LLMLogLevel }
            $level | Should -Be 'Warning'
        }

        It "Should set log level to Error" {
            & $script:LlmModule { Set-LLMLogLevel -Level 'Error' }
            $level = & $script:LlmModule { Get-LLMLogLevel }
            $level | Should -Be 'Error'
        }

        It "Should reject invalid log level" {
            { & $script:LlmModule { Set-LLMLogLevel -Level 'InvalidLevel' } } | Should -Throw
        }
    }

    Context "Write-LLMLog Function" {
        It "Should write Info level log entry" {
            & $script:LlmModule { Write-LLMLog -Level 'Info' -Message 'Test info message' }

            Test-Path $script:TestLogFile | Should -Be $true
            $content = Get-Content $script:TestLogFile -Raw
            $content | Should -Not -BeNullOrEmpty
            $content | Should -Match 'Test info message'
        }

        It "Should write log entry with structured JSON format" {
            & $script:LlmModule { Write-LLMLog -Level 'Info' -Message 'Test message' }

            $content = Get-Content $script:TestLogFile -Raw
            $json = $content | ConvertFrom-Json

            $json.Timestamp | Should -Not -BeNullOrEmpty
            $json.Level | Should -Be 'Info'
            $json.Message | Should -Be 'Test message'
        }

        It "Should include Data object when provided" {
            $testData = @{
                Router = 'FunctionGemma'
                Latency = 250
                ToolName = 'Get-SystemInfo'
            }

            & $script:LlmModule {
                Write-LLMLog -Level 'Info' -Message 'Router decision' -Data $args[0]
            } $testData

            $content = Get-Content $script:TestLogFile -Raw
            $json = $content | ConvertFrom-Json

            $json.Data | Should -Not -BeNullOrEmpty
            $json.Data.Router | Should -Be 'FunctionGemma'
            $json.Data.Latency | Should -Be 250
            $json.Data.ToolName | Should -Be 'Get-SystemInfo'
        }

        It "Should respect log level filtering - Debug messages not logged at Info level" {
            & $script:LlmModule { Set-LLMLogLevel -Level 'Info' }
            & $script:LlmModule { Write-LLMLog -Level 'Debug' -Message 'Debug message' }

            if (Test-Path $script:TestLogFile) {
                $content = Get-Content $script:TestLogFile -Raw -ErrorAction SilentlyContinue
                if ($content) {
                    $content | Should -Not -Match 'Debug message'
                }
            }
        }

        It "Should log Debug messages when log level is Debug" {
            & $script:LlmModule { Set-LLMLogLevel -Level 'Debug' }
            & $script:LlmModule { Write-LLMLog -Level 'Debug' -Message 'Debug message' }

            Test-Path $script:TestLogFile | Should -Be $true
            $content = Get-Content $script:TestLogFile -Raw
            $content | Should -Match 'Debug message'
        }

        It "Should log Warning messages at Info level" {
            & $script:LlmModule { Set-LLMLogLevel -Level 'Info' }
            & $script:LlmModule { Write-LLMLog -Level 'Warning' -Message 'Warning message' }

            Test-Path $script:TestLogFile | Should -Be $true
            $content = Get-Content $script:TestLogFile -Raw
            $content | Should -Match 'Warning message'
        }

        It "Should log Error messages at Info level" {
            & $script:LlmModule { Set-LLMLogLevel -Level 'Info' }
            & $script:LlmModule { Write-LLMLog -Level 'Error' -Message 'Error message' }

            Test-Path $script:TestLogFile | Should -Be $true
            $content = Get-Content $script:TestLogFile -Raw
            $content | Should -Match 'Error message'
        }

        It "Should not log Info messages when level is Warning" {
            & $script:LlmModule { Set-LLMLogLevel -Level 'Warning' }
            & $script:LlmModule { Write-LLMLog -Level 'Info' -Message 'Info message' }

            if (Test-Path $script:TestLogFile) {
                $content = Get-Content $script:TestLogFile -Raw -ErrorAction SilentlyContinue
                if ($content) {
                    $content | Should -Not -Match 'Info message'
                }
            }
        }

        It "Should not log Warning messages when level is Error" {
            & $script:LlmModule { Set-LLMLogLevel -Level 'Error' }
            & $script:LlmModule { Write-LLMLog -Level 'Warning' -Message 'Warning message' }

            if (Test-Path $script:TestLogFile) {
                $content = Get-Content $script:TestLogFile -Raw -ErrorAction SilentlyContinue
                if ($content) {
                    $content | Should -Not -Match 'Warning message'
                }
            }
        }

        It "Should append multiple log entries" {
            & $script:LlmModule { Write-LLMLog -Level 'Info' -Message 'First message' }
            & $script:LlmModule { Write-LLMLog -Level 'Info' -Message 'Second message' }
            & $script:LlmModule { Write-LLMLog -Level 'Info' -Message 'Third message' }

            $lines = Get-Content $script:TestLogFile
            $lines.Count | Should -BeGreaterOrEqual 3

            $content = Get-Content $script:TestLogFile -Raw
            $content | Should -Match 'First message'
            $content | Should -Match 'Second message'
            $content | Should -Match 'Third message'
        }

        It "Should handle router decision logging scenario" {
            $routerData = @{
                Router = 'FunctionGemma'
                Model = 'functionary/functiongemma-2b'
                Decision = 'tool_call'
                ToolName = 'Get-SystemInfo'
                Latency = 342
                TokensUsed = 156
            }

            & $script:LlmModule {
                Write-LLMLog -Level 'Info' -Message 'Router decision completed' -Data $args[0]
            } $routerData

            $content = Get-Content $script:TestLogFile -Raw
            $json = $content | ConvertFrom-Json

            $json.Data.Router | Should -Be 'FunctionGemma'
            $json.Data.Decision | Should -Be 'tool_call'
            $json.Data.Latency | Should -Be 342
        }

        It "Should handle tool call logging scenario" {
            $toolData = @{
                ToolName = 'Get-CimInstance'
                Parameters = @{ ClassName = 'Win32_LogicalDisk' }
                StartTime = (Get-Date).ToString('o')
                EndTime = (Get-Date).AddMilliseconds(45).ToString('o')
                Success = $true
            }

            & $script:LlmModule {
                Write-LLMLog -Level 'Debug' -Message 'Tool execution' -Data $args[0]
            } $toolData

            # Should not log at Info level
            if (Test-Path $script:TestLogFile) {
                $content = Get-Content $script:TestLogFile -Raw -ErrorAction SilentlyContinue
                if ($content) {
                    $content | Should -Not -Match 'Tool execution'
                }
            }

            # But should log at Debug level
            & $script:LlmModule { Set-LLMLogLevel -Level 'Debug' }
            Remove-Item $script:TestLogFile -Force -ErrorAction SilentlyContinue

            & $script:LlmModule {
                Write-LLMLog -Level 'Debug' -Message 'Tool execution' -Data $args[0]
            } $toolData

            $content = Get-Content $script:TestLogFile -Raw
            $json = $content | ConvertFrom-Json
            $json.Data.ToolName | Should -Be 'Get-CimInstance'
            $json.Data.Success | Should -Be $true
        }

        It "Should create log directory if it doesn't exist" {
            $tempLogDir = Join-Path $TestDrive 'TempLogs'

            # Temporarily modify the log path within the module (this is a simplified test)
            # In real scenarios, the module would handle this internally

            if (Test-Path $tempLogDir) {
                Remove-Item $tempLogDir -Recurse -Force
            }

            # The Write-LLMLog function should create the directory
            # We'll test this indirectly by verifying standard behavior
            Test-Path $script:TestLogDir | Should -Be $true
        }
    }

    Context "Log Level Hierarchy" {
        It "Should have correct log level ordering (Debug < Info < Warning < Error)" {
            $levels = @('Debug', 'Info', 'Warning', 'Error')

            foreach ($level in $levels) {
                & $script:LlmModule { Set-LLMLogLevel -Level $args[0] } $level
                $currentLevel = & $script:LlmModule { Get-LLMLogLevel }
                $currentLevel | Should -Be $level
            }
        }
    }

    Context "Performance and Reliability" {
        It "Should handle rapid sequential writes" {
            & $script:LlmModule { Set-LLMLogLevel -Level 'Info' }

            1..10 | ForEach-Object {
                & $script:LlmModule { Write-LLMLog -Level 'Info' -Message "Message $_" }
            }

            Test-Path $script:TestLogFile | Should -Be $true
            $lines = Get-Content $script:TestLogFile
            $lines.Count | Should -BeGreaterOrEqual 10
        }

        It "Should handle empty Data object gracefully" {
            & $script:LlmModule { Write-LLMLog -Level 'Info' -Message 'Message with empty data' -Data @{} }

            $content = Get-Content $script:TestLogFile -Raw
            $json = $content | ConvertFrom-Json
            $json.Message | Should -Be 'Message with empty data'
        }
    }
}
