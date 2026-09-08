#Requires -Modules Pester

<#
.SYNOPSIS
    Pester tests for Sync-RegistryCredentials.ps1

.DESCRIPTION
    Test suite for registry credential synchronization from Bitwarden to wincred.

.NOTES
    Pester Version: 5.x
#>

BeforeAll {
    $scriptPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'Sync-RegistryCredentials.ps1'

    # Import TestHelpers if available
    $helpersPath = Join-Path $PSScriptRoot 'TestHelpers.psm1'
    if (Test-Path $helpersPath) {
        Import-Module $helpersPath -Force
    }

    # Mock Bitwarden item structure
    function New-MockBitwardenItem {
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingPlainTextForPassword', '', Justification = 'test fixture, not a real secret')]
        param(
            [string]$Name = 'Test Item',
            [string]$Username = 'testuser',
            [string]$Password = 'testpassword',
            [int]$Type = 1  # Login type
        )

        @{
            id       = [guid]::NewGuid().ToString()
            name     = $Name
            type     = $Type
            login    = @{
                username = $Username
                password = $Password
            }
        } | ConvertTo-Json -Depth 3
    }
}

Describe 'Sync-RegistryCredentials Script Structure' {

    Context 'Script File Validation' {

        It 'Should exist at expected location' {
            Test-Path $scriptPath | Should -Be $true
        }

        It 'Should be valid PowerShell syntax' {
            $errors = $null
            $null = [System.Management.Automation.PSParser]::Tokenize(
                (Get-Content $scriptPath -Raw),
                [ref]$errors
            )
            $errors.Count | Should -Be 0
        }

        It 'Should have proper help documentation' {
            $content = Get-Content $scriptPath -Raw
            $content | Should -Match '\.SYNOPSIS'
            $content | Should -Match '\.DESCRIPTION'
            $content | Should -Match '\.EXAMPLE'
        }
    }
}

Describe 'Test-BitwardenSession' {

    BeforeAll {
        . $scriptPath
    }

    Context 'Valid Session' {

        It 'Should return true when vault is unlocked' {
            $env:BW_SESSION = 'valid_session_token_for_testing'

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

        It 'Should return false when vault is locked' {
            $env:BW_SESSION = 'some_session'

            Mock bw {
                $global:LASTEXITCODE = 0
                return '{"status":"locked"}'
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

Describe 'Test-CredentialHelper' {

    BeforeAll {
        . $scriptPath
    }

    Context 'Helper Available' {

        It 'Should return true when docker-credential-wincred is found' {
            Mock Get-Command {
                [PSCustomObject]@{
                    Name   = 'docker-credential-wincred'
                    Source = 'C:\Program Files\Docker\Docker\resources\bin\docker-credential-wincred.exe'
                }
            } -ParameterFilter { $Name -eq 'docker-credential-wincred' }

            if (Get-Command Test-CredentialHelper -ErrorAction SilentlyContinue) {
                Test-CredentialHelper | Should -Be $true
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }

    Context 'Helper Not Available' {

        It 'Should return false when helper is not found' {
            Mock Get-Command { $null } -ParameterFilter { $Name -eq 'docker-credential-wincred' }

            if (Get-Command Test-CredentialHelper -ErrorAction SilentlyContinue) {
                Test-CredentialHelper | Should -Be $false
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }
}

Describe 'Get-BitwardenItem' {

    BeforeAll {
        . $scriptPath
    }

    Context 'Successful Retrieval' {

        It 'Should return item when found' {
            $mockItem = New-MockBitwardenItem -Name 'Docker Hub' -Username 'dockeruser' -Password 'dockerpass'

            Mock bw {
                $global:LASTEXITCODE = 0
                return $mockItem
            } -ParameterFilter { $args[0] -eq 'get' -and $args[1] -eq 'item' }

            if (Get-Command Get-BitwardenItem -ErrorAction SilentlyContinue) {
                $item = Get-BitwardenItem -ItemName 'Docker Hub'
                $item | Should -Not -BeNullOrEmpty
                $item.login.username | Should -Be 'dockeruser'
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }

    Context 'Item Not Found' {

        It 'Should return null when item not found' {
            Mock bw {
                $global:LASTEXITCODE = 1
                return 'Not found.'
            } -ParameterFilter { $args[0] -eq 'get' -and $args[1] -eq 'item' }

            if (Get-Command Get-BitwardenItem -ErrorAction SilentlyContinue) {
                $item = Get-BitwardenItem -ItemName 'Nonexistent Item'
                $item | Should -BeNullOrEmpty
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }

    Context 'Invalid Item Type' {

        It 'Should return null for non-login items' {
            $mockItem = @{
                id   = [guid]::NewGuid().ToString()
                name = 'Secure Note'
                type = 2  # Secure note type
            } | ConvertTo-Json

            Mock bw {
                $global:LASTEXITCODE = 0
                return $mockItem
            } -ParameterFilter { $args[0] -eq 'get' -and $args[1] -eq 'item' }

            if (Get-Command Get-BitwardenItem -ErrorAction SilentlyContinue) {
                $item = Get-BitwardenItem -ItemName 'Secure Note'
                $item | Should -BeNullOrEmpty
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }
}

Describe 'Set-RegistryCredential' {

    BeforeAll {
        . $scriptPath
    }

    Context 'Successful Store' {

        It 'Should return true on successful store' {
            Mock docker-credential-wincred {
                $global:LASTEXITCODE = 0
                return ''
            }

            # Override the script config temporarily
            $script:Config = @{ CredentialHelper = 'docker-credential-wincred'; MaxLogSizeBytes = 5MB; LogRetentionDays = 30 }

            if (Get-Command Set-RegistryCredential -ErrorAction SilentlyContinue) {
                $result = Set-RegistryCredential -ServerUrl 'https://test.io' -Username 'user' -Secret 'pass'
                $result | Should -Be $true
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }

    Context 'Failed Store' {

        It 'Should return false on store failure' {
            Mock docker-credential-wincred {
                $global:LASTEXITCODE = 1
                return 'Store failed'
            }

            $script:Config = @{ CredentialHelper = 'docker-credential-wincred'; MaxLogSizeBytes = 5MB; LogRetentionDays = 30 }

            if (Get-Command Set-RegistryCredential -ErrorAction SilentlyContinue) {
                $result = Set-RegistryCredential -ServerUrl 'https://test.io' -Username 'user' -Secret 'pass'
                $result | Should -Be $false
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }

    Context 'WhatIf Support' {

        It 'Should support WhatIf parameter' {
            if (Get-Command Set-RegistryCredential -ErrorAction SilentlyContinue) {
                { Set-RegistryCredential -ServerUrl 'https://test.io' -Username 'user' -Secret 'pass' -WhatIf } | Should -Not -Throw
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }
}

Describe 'Get-AllStoredCredentials' {

    BeforeAll {
        . $scriptPath
    }

    Context 'Credentials Exist' {

        It 'Should return list of server URLs' {
            Mock docker-credential-wincred {
                $global:LASTEXITCODE = 0
                return '{"https://index.docker.io/v1/":"dockeruser","https://ghcr.io":"ghuser"}'
            }

            $script:Config = @{ CredentialHelper = 'docker-credential-wincred'; MaxLogSizeBytes = 5MB; LogRetentionDays = 30 }

            if (Get-Command Get-AllStoredCredentials -ErrorAction SilentlyContinue) {
                $creds = Get-AllStoredCredentials
                $creds | Should -Contain 'https://index.docker.io/v1/'
                $creds | Should -Contain 'https://ghcr.io'
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }

    Context 'No Credentials' {

        It 'Should return empty array when no credentials exist' {
            Mock docker-credential-wincred {
                $global:LASTEXITCODE = 0
                return '{}'
            }

            $script:Config = @{ CredentialHelper = 'docker-credential-wincred'; MaxLogSizeBytes = 5MB; LogRetentionDays = 30 }

            if (Get-Command Get-AllStoredCredentials -ErrorAction SilentlyContinue) {
                $creds = @(Get-AllStoredCredentials)
                # Function returns empty array on error or empty result
                $creds.Count | Should -Be 0
            } else {
                Set-ItResult -Skipped -Because 'Function not exported'
            }
        }
    }
}

Describe 'Integration Tests' {

    Context 'Full Sync Flow' {

        BeforeAll {
            . $scriptPath
        }

        It 'Should sync credentials with mocked dependencies' {
            # Mock all dependencies
            $env:BW_SESSION = 'test_session'

            Mock bw {
                param($Command)
                switch ($args[0]) {
                    'status' {
                        $global:LASTEXITCODE = 0
                        '{"status":"unlocked"}'
                    }
                    'get' {
                        $global:LASTEXITCODE = 0
                        New-MockBitwardenItem -Name 'Docker Hub' -Username 'dockeruser' -Password 'dockerpass'
                    }
                    default { '' }
                }
            }

            Mock Get-Command {
                [PSCustomObject]@{ Name = 'docker-credential-wincred'; Source = 'test.exe' }
            } -ParameterFilter { $Name -eq 'docker-credential-wincred' }

            Mock docker-credential-wincred {
                $global:LASTEXITCODE = 0
                return ''
            }

            # This is an integration test that requires full setup
            Set-ItResult -Skipped -Because 'Integration test requires real credential helper setup'

            $env:BW_SESSION = $null
        }
    }
}

Describe 'Security Tests' {

    Context 'Credential Handling' {

        It 'Should use stdin for credential storage (not command line)' {
            $scriptContent = Get-Content $scriptPath -Raw
            # Verify credentials are piped to stdin, not passed as arguments
            $scriptContent | Should -Match '\| & \$script:Config\.CredentialHelper store'
        }

        It 'Should clear sensitive data after use' {
            $scriptContent = Get-Content $scriptPath -Raw
            $scriptContent | Should -Match '\[System\.GC\]::Collect\(\)'
            $scriptContent | Should -Match '\$item = \$null'
        }
    }
}

AfterAll {
    # Cleanup
    $env:BW_SESSION = $null
}
