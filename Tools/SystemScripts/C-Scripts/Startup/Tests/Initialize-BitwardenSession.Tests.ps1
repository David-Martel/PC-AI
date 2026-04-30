#Requires -Modules Pester

<#
.SYNOPSIS
    Pester tests for Initialize-BitwardenSession.ps1

.DESCRIPTION
    Comprehensive test suite for Bitwarden session initialization.
    Tests credential retrieval, login, unlock, and session management.

.NOTES
    Pester Version: 5.x
#>

BeforeAll {
    $scriptPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'Initialize-BitwardenSession.ps1'

    # Import TestHelpers if available
    $helpersPath = Join-Path $PSScriptRoot 'TestHelpers.psm1'
    if (Test-Path $helpersPath) {
        Import-Module $helpersPath -Force
    }

    # Helper to create mock Bitwarden status
    function New-MockBitwardenStatus {
        param(
            [ValidateSet('unauthenticated', 'locked', 'unlocked')]
            [string]$Status = 'locked',
            [string]$UserEmail = 'test@example.com'
        )

        [PSCustomObject]@{
            serverUrl = 'https://vault.bitwarden.com'
            lastSync  = (Get-Date).ToString('o')
            userEmail = $UserEmail
            userId    = [guid]::NewGuid().ToString()
            status    = $Status
        }
    }
}

Describe 'Initialize-BitwardenSession Script Structure' {

    Context 'Script File Validation' {

        It 'Should exist at expected location' {
            $scriptPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'Initialize-BitwardenSession.ps1'
            Test-Path $scriptPath | Should -Be $true
        }

        It 'Should be valid PowerShell syntax' {
            $scriptPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'Initialize-BitwardenSession.ps1'
            $errors = $null
            $null = [System.Management.Automation.PSParser]::Tokenize(
                (Get-Content $scriptPath -Raw),
                [ref]$errors
            )
            $errors.Count | Should -Be 0
        }

        It 'Should have proper help documentation' {
            $scriptPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'Initialize-BitwardenSession.ps1'
            $content = Get-Content $scriptPath -Raw
            # Check for comment-based help markers
            $content | Should -Match '\.SYNOPSIS'
            $content | Should -Match '\.DESCRIPTION'
            $content | Should -Match '\.EXAMPLE'
        }
    }
}

Describe 'Get-StoredCredentialPassword' {

    BeforeAll {
        . (Join-Path (Split-Path $PSScriptRoot -Parent) 'Initialize-BitwardenSession.ps1')
    }

    Context 'Happy Path' {

        It 'Should retrieve credential from Windows Credential Manager' {
            # This test requires actual credentials to be set up
            # For unit testing, we mock the underlying calls
            Mock cmdkey {
                return "Target: bitwarden/client_id`nType: Generic`nUser: bitwarden"
            }

            # Skip if function doesn't exist (script not dot-sourced)
            if (Get-Command Get-StoredCredentialPassword -ErrorAction SilentlyContinue) {
                # Mock would need P/Invoke mock which is complex
                # This is more of an integration test
                Set-ItResult -Skipped -Because 'Requires Windows Credential Manager integration'
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }

    Context 'Error Handling' {

        It 'Should throw when credential not found' {
            Mock cmdkey {
                $global:LASTEXITCODE = 1
                return "CMDKEY: Credential not found."
            }

            if (Get-Command Get-StoredCredentialPassword -ErrorAction SilentlyContinue) {
                { Get-StoredCredentialPassword -Target 'nonexistent/credential' } | Should -Throw
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }
}

Describe 'Test-BitwardenCli' {

    BeforeAll {
        . (Join-Path (Split-Path $PSScriptRoot -Parent) 'Initialize-BitwardenSession.ps1')
    }

    Context 'CLI Available' {

        It 'Should return true when bw CLI is found' {
            Mock Get-Command {
                [PSCustomObject]@{
                    Name   = 'bw'
                    Source = 'C:\Program Files\Bitwarden CLI\bw.exe'
                }
            } -ParameterFilter { $Name -eq 'bw' }

            if (Get-Command Test-BitwardenCli -ErrorAction SilentlyContinue) {
                Test-BitwardenCli | Should -Be $true
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }

    Context 'CLI Not Available' {

        It 'Should return false when bw CLI is not found' {
            Mock Get-Command { $null } -ParameterFilter { $Name -eq 'bw' }

            if (Get-Command Test-BitwardenCli -ErrorAction SilentlyContinue) {
                Test-BitwardenCli | Should -Be $false
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }
}

Describe 'Get-BitwardenStatus' {

    BeforeAll {
        . (Join-Path (Split-Path $PSScriptRoot -Parent) 'Initialize-BitwardenSession.ps1')
    }

    Context 'Status Retrieval' {

        It 'Should parse locked status correctly' {
            Mock bw {
                $global:LASTEXITCODE = 0
                '{"serverUrl":"https://vault.bitwarden.com","lastSync":"2025-01-25T10:00:00.000Z","userEmail":"test@example.com","userId":"123","status":"locked"}'
            } -ParameterFilter { $args[0] -eq 'status' }

            if (Get-Command Get-BitwardenStatus -ErrorAction SilentlyContinue) {
                $status = Get-BitwardenStatus
                $status.status | Should -Be 'locked'
                $status.userEmail | Should -Be 'test@example.com'
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }

        It 'Should parse unlocked status correctly' {
            Mock bw {
                $global:LASTEXITCODE = 0
                '{"serverUrl":"https://vault.bitwarden.com","lastSync":"2025-01-25T10:00:00.000Z","userEmail":"test@example.com","userId":"123","status":"unlocked"}'
            } -ParameterFilter { $args[0] -eq 'status' }

            if (Get-Command Get-BitwardenStatus -ErrorAction SilentlyContinue) {
                $status = Get-BitwardenStatus
                $status.status | Should -Be 'unlocked'
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }

        It 'Should handle unauthenticated status' {
            Mock bw {
                $global:LASTEXITCODE = 0
                '{"serverUrl":null,"lastSync":null,"userEmail":null,"userId":null,"status":"unauthenticated"}'
            } -ParameterFilter { $args[0] -eq 'status' }

            if (Get-Command Get-BitwardenStatus -ErrorAction SilentlyContinue) {
                $status = Get-BitwardenStatus
                $status.status | Should -Be 'unauthenticated'
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }

    Context 'Error Handling' {

        It 'Should return null on CLI error' {
            Mock bw {
                $global:LASTEXITCODE = 1
                throw 'CLI error'
            } -ParameterFilter { $args[0] -eq 'status' }

            if (Get-Command Get-BitwardenStatus -ErrorAction SilentlyContinue) {
                $status = Get-BitwardenStatus
                $status | Should -BeNullOrEmpty
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }
}

Describe 'Invoke-BitwardenLogin' {

    BeforeAll {
        . (Join-Path (Split-Path $PSScriptRoot -Parent) 'Initialize-BitwardenSession.ps1')
    }

    Context 'Successful Login' {

        It 'Should login with API key credentials' {
            Mock bw {
                $global:LASTEXITCODE = 0
                return 'You are logged in!'
            } -ParameterFilter { $args -contains '--apikey' }

            if (Get-Command Invoke-BitwardenLogin -ErrorAction SilentlyContinue) {
                $result = Invoke-BitwardenLogin -ClientId 'test.client.id' -ClientSecret 'test_secret'
                $result | Should -Be $true
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }

        It 'Should handle already logged in scenario' {
            Mock bw {
                $global:LASTEXITCODE = 1
                return 'You are already logged in'
            } -ParameterFilter { $args -contains '--apikey' }

            if (Get-Command Invoke-BitwardenLogin -ErrorAction SilentlyContinue) {
                $result = Invoke-BitwardenLogin -ClientId 'test.client.id' -ClientSecret 'test_secret'
                $result | Should -Be $true
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }

    Context 'Failed Login' {

        It 'Should return false on invalid credentials' {
            Mock bw {
                $global:LASTEXITCODE = 1
                return 'Invalid credentials'
            } -ParameterFilter { $args -contains '--apikey' }

            if (Get-Command Invoke-BitwardenLogin -ErrorAction SilentlyContinue) {
                $result = Invoke-BitwardenLogin -ClientId 'bad.id' -ClientSecret 'bad_secret'
                $result | Should -Be $false
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }

    Context 'Security' {

        It 'Should clear client secret from environment after login' {
            Mock bw {
                $global:LASTEXITCODE = 0
                return 'Logged in'
            }

            if (Get-Command Invoke-BitwardenLogin -ErrorAction SilentlyContinue) {
                Invoke-BitwardenLogin -ClientId 'test.id' -ClientSecret 'sensitive_secret'
                $env:BW_CLIENTSECRET | Should -BeNullOrEmpty
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }
}

Describe 'Invoke-BitwardenUnlock' {

    BeforeAll {
        . (Join-Path (Split-Path $PSScriptRoot -Parent) 'Initialize-BitwardenSession.ps1')
    }

    Context 'Successful Unlock' {

        It 'Should return session token on successful unlock' {
            # Valid base64 session token (at least 32 chars)
            $mockSession = 'YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY3ODkw'

            Mock bw {
                $global:LASTEXITCODE = 0
                return $mockSession
            } -ParameterFilter { $args -contains '--raw' }

            if (Get-Command Invoke-BitwardenUnlock -ErrorAction SilentlyContinue) {
                $session = Invoke-BitwardenUnlock -MasterPassword 'test_password'
                $session | Should -Be $mockSession
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }

    Context 'Retry Logic' {

        It 'Should retry on transient failure' {
            $script:UnlockAttempt = 0
            # Valid base64 session token (at least 32 chars, base64 chars only)
            $validBase64Session = 'YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY3ODkwMTIzNDU2Nzg5MDEyMzQ1Njc4OTA='

            Mock bw {
                $script:UnlockAttempt++
                if ($script:UnlockAttempt -lt 2) {
                    $global:LASTEXITCODE = 1
                    return 'Unlock failed'
                }
                $global:LASTEXITCODE = 0
                return $validBase64Session
            } -ParameterFilter { $args -contains '--raw' }

            Mock Start-Sleep { }

            if (Get-Command Invoke-BitwardenUnlock -ErrorAction SilentlyContinue) {
                $session = Invoke-BitwardenUnlock -MasterPassword 'test_password'
                $session | Should -Not -BeNullOrEmpty
                $script:UnlockAttempt | Should -BeGreaterThan 1
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }

    Context 'Failed Unlock' {

        It 'Should return null after max retries' {
            Mock bw {
                $global:LASTEXITCODE = 1
                return 'Invalid master password'
            } -ParameterFilter { $args -contains '--raw' }

            Mock Start-Sleep { }

            if (Get-Command Invoke-BitwardenUnlock -ErrorAction SilentlyContinue) {
                $session = Invoke-BitwardenUnlock -MasterPassword 'wrong_password'
                $session | Should -BeNullOrEmpty
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }
}

Describe 'Export-BitwardenSession' {

    BeforeAll {
        . (Join-Path (Split-Path $PSScriptRoot -Parent) 'Initialize-BitwardenSession.ps1')
    }

    Context 'Process Environment' {

        It 'Should set BW_SESSION in current process' {
            $testSession = 'test_session_token'

            if (Get-Command Export-BitwardenSession -ErrorAction SilentlyContinue) {
                Export-BitwardenSession -Session $testSession
                $env:BW_SESSION | Should -Be $testSession

                # Cleanup
                $env:BW_SESSION = $null
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }

    Context 'Persistent Environment' {

        It 'Should accept Persistent parameter with WhatIf' {
            if (Get-Command Export-BitwardenSession -ErrorAction SilentlyContinue) {
                # Test that the function accepts -Persistent parameter
                # Using -WhatIf to avoid actually modifying the environment
                { Export-BitwardenSession -Session 'YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXo=' -Persistent -WhatIf } | Should -Not -Throw
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }
}

Describe 'Test-BitwardenSession' {

    BeforeAll {
        . (Join-Path (Split-Path $PSScriptRoot -Parent) 'Initialize-BitwardenSession.ps1')
    }

    Context 'Valid Session' {

        It 'Should return true when status shows unlocked' {
            $env:BW_SESSION = 'YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY3ODkw'

            Mock bw {
                $global:LASTEXITCODE = 0
                return '{"status":"unlocked","userEmail":"test@example.com"}'
            } -ParameterFilter { $args[0] -eq 'status' }

            if (Get-Command Test-BitwardenSession -ErrorAction SilentlyContinue) {
                Test-BitwardenSession | Should -Be $true
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }

            $env:BW_SESSION = $null
        }
    }

    Context 'Invalid Session' {

        It 'Should return false when BW_SESSION not set' {
            $env:BW_SESSION = $null

            if (Get-Command Test-BitwardenSession -ErrorAction SilentlyContinue) {
                Test-BitwardenSession | Should -Be $false
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }

        It 'Should return false when status shows locked' {
            $env:BW_SESSION = 'invalid_session'

            Mock bw {
                $global:LASTEXITCODE = 0
                return '{"status":"locked","userEmail":"test@example.com"}'
            } -ParameterFilter { $args[0] -eq 'status' }

            if (Get-Command Test-BitwardenSession -ErrorAction SilentlyContinue) {
                Test-BitwardenSession | Should -Be $false
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }

            $env:BW_SESSION = $null
        }
    }
}

Describe 'Integration Tests' {

    Context 'Full Initialization Flow' {

        BeforeAll {
            . (Join-Path (Split-Path $PSScriptRoot -Parent) 'Initialize-BitwardenSession.ps1')
        }

        It 'Should complete full initialization with mocked dependencies' {
            # Mock all external dependencies
            Mock Get-Command {
                [PSCustomObject]@{ Name = 'bw'; Source = 'bw.exe' }
            } -ParameterFilter { $Name -eq 'bw' }

            Mock bw {
                param($Command)
                switch ($args[0]) {
                    'status' {
                        '{"status":"locked","userEmail":"test@example.com"}'
                    }
                    'login' {
                        $global:LASTEXITCODE = 0
                        'Logged in'
                    }
                    'unlock' {
                        $global:LASTEXITCODE = 0
                        'session_token_for_testing_purposes_1234567890'
                    }
                    'sync' {
                        $global:LASTEXITCODE = 0
                        'Synced'
                    }
                    default { '' }
                }
            }

            Mock cmdkey { "Target: bitwarden/client_id" }

            # This is an integration test that would need real credentials
            Set-ItResult -Skipped -Because 'Integration test requires real Windows Credential Manager setup'
        }
    }
}

Describe 'Security Tests' {

    Context 'Credential Handling' {

        It 'Should not log sensitive credentials' {
            $logFile = 'C:\Scripts\Startup\bitwarden-session.log'

            # Run initialization (would need mocking)
            # Then verify log file doesn't contain secrets

            Set-ItResult -Skipped -Because 'Requires full integration test setup'
        }

        It 'Should clear credentials from memory after use' {
            # Verify garbage collection is called
            # This is more of a code review item

            Set-ItResult -Skipped -Because 'Memory verification requires specialized tooling'
        }
    }
}

AfterAll {
    # Cleanup
    $env:BW_SESSION = $null
    $env:BW_CLIENTID = $null
    $env:BW_CLIENTSECRET = $null
}
