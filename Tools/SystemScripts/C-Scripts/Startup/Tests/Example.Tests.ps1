#Requires -Modules Pester

<#
.SYNOPSIS
    Example Pester test demonstrating test framework usage

.DESCRIPTION
    This is a simple example showing how to use the test framework,
    helper functions, and best practices.

.NOTES
    Use this as a template for new tests
#>

BeforeAll {
    # Import test helpers
    Import-Module (Join-Path $PSScriptRoot 'TestHelpers.psm1') -Force

    # Define a simple function to test (normally this would be in the main script)
    function Test-SimpleFunction {
        param(
            [string]$Input,
            [switch]$ThrowError
        )

        if ($ThrowError) {
            throw "Error occurred"
        }

        return "Processed: $Input"
    }

    function Test-RetryFunction {
        param(
            [int]$MaxRetries = 3
        )

        $attempt = 0

        while ($attempt -lt $MaxRetries) {
            $attempt++

            try {
                # Simulate external call
                $result = Get-ExternalData
                return $result
            } catch {
                if ($attempt -eq $MaxRetries) {
                    throw "Max retries exceeded"
                }

                Start-Sleep -Milliseconds 100
            }
        }
    }

    function Get-ExternalData {
        # This would be mocked in tests
        throw "Not implemented"
    }
}

Describe 'Example Test Structure' {

    Context 'Basic Test Pattern' {

        It 'Should demonstrate simple assertion' {
            # Arrange
            $input = 'test'

            # Act
            $result = Test-SimpleFunction -Input $input

            # Assert
            $result | Should -Be 'Processed: test'
        }

        It 'Should demonstrate exception testing' {
            # Assert that function throws
            { Test-SimpleFunction -Input 'test' -ThrowError } | Should -Throw
        }

        It 'Should demonstrate exception message testing' {
            # Assert specific error message
            { Test-SimpleFunction -Input 'test' -ThrowError } |
                Should -Throw -ExpectedMessage 'Error occurred'
        }
    }

    Context 'Using Mock Objects' {

        It 'Should demonstrate mock service creation' {
            # Create mock service using helper
            $service = New-MockService -Name 'TestService' -Status 'Running'

            # Verify properties
            $service.Name | Should -Be 'TestService'
            $service.Status | Should -Be 'Running'
            $service.StartType | Should -Be 'Automatic'
        }

        It 'Should demonstrate mock process creation' {
            # Create mock process
            $process = New-MockProcess -ProcessName 'TestProcess' -Id 9999

            # Verify properties
            $process.ProcessName | Should -Be 'TestProcess'
            $process.Id | Should -Be 9999
            $process.HasExited | Should -Be $false
        }
    }

    Context 'Testing with Mocks' {

        BeforeAll {
            # Setup mocks for this context
            Mock Get-Service {
                New-MockService -Name 'TestService' -Status 'Running'
            }
        }

        It 'Should use mocked Get-Service' {
            # Act
            $service = Get-Service -Name 'TestService'

            # Assert
            $service.Status | Should -Be 'Running'

            # Verify mock was called
            Should -Invoke Get-Service -Times 1 -Exactly
        }

        It 'Should verify mock with parameter filter' {
            # Act
            Get-Service -Name 'TestService'

            # Assert with parameter filter
            Should -Invoke Get-Service -ParameterFilter {
                $Name -eq 'TestService'
            } -Times 1
        }
    }

    Context 'Retry Scenario Testing' {

        BeforeAll {
            Reset-RetryCounter
        }

        It 'Should demonstrate retry scenario' {
            # Create a retry scenario: fail 2 times, then succeed
            $retryBlock = New-RetryScenario -FailCount 2 -SuccessValue 'Success'

            # First call - fails
            $result1 = & $retryBlock
            $result1 | Should -Be $false

            # Second call - fails
            $result2 = & $retryBlock
            $result2 | Should -Be $false

            # Third call - succeeds
            $result3 = & $retryBlock
            $result3 | Should -Be 'Success'

            # Verify retry count
            Assert-RetryCount -ExpectedCount 3
        }

        It 'Should demonstrate retry with exceptions' {
            Reset-RetryCounter

            # Create scenario that throws exceptions
            $retryBlock = New-RetryScenario `
                -FailCount 2 `
                -SuccessValue 'Success' `
                -FailureException 'Service not ready'

            # First call - throws
            { & $retryBlock } | Should -Throw -ExpectedMessage 'Service not ready'

            # Second call - throws
            { & $retryBlock } | Should -Throw -ExpectedMessage 'Service not ready'

            # Third call - succeeds
            $result = & $retryBlock
            $result | Should -Be 'Success'

            Assert-RetryCount -ExpectedCount 3
        }
    }

    Context 'Testing Retry Logic in Functions' {

        BeforeEach {
            Reset-RetryCounter
        }

        It 'Should retry on failure' {
            # Mock external dependency with retry scenario
            Mock Get-ExternalData {
                $script:RetryAttempt++

                if ($script:RetryAttempt -lt 3) {
                    throw "Not ready"
                }

                return "Data ready"
            }

            # Act
            $result = Test-RetryFunction -MaxRetries 5

            # Assert
            $result | Should -Be "Data ready"
            Assert-RetryCount -ExpectedCount 3
        }

        It 'Should throw after max retries' {
            # Mock always fails
            Mock Get-ExternalData {
                throw "Always fails"
            }

            # Act & Assert
            { Test-RetryFunction -MaxRetries 3 } | Should -Throw
        }
    }

    Context 'Timing Assertions' {

        It 'Should measure execution time' {
            # Measure how long something takes
            $elapsed = Measure-ExecutionTime -ScriptBlock {
                Start-Sleep -Milliseconds 100
            }

            # Verify it took at least 100ms
            $elapsed.TotalMilliseconds | Should -BeGreaterOrEqual 100
        }

        It 'Should assert execution time range' {
            # Assert execution is between 100-200ms
            Assert-ExecutionTime -ScriptBlock {
                Start-Sleep -Milliseconds 150
            } -MinSeconds 0.1 -MaxSeconds 0.3
        }

        It 'Should fail if execution too slow' {
            # This should fail because sleep is 200ms but max is 100ms
            {
                Assert-ExecutionTime -ScriptBlock {
                    Start-Sleep -Milliseconds 200
                } -MaxSeconds 0.1
            } | Should -Throw
        }
    }

    Context 'WSL Mock Helpers' {

        It 'Should create WSL list distros output' {
            $output = New-MockWSLOutput -Command 'ListDistros' -Success $true

            $output | Should -Contain 'Ubuntu'
        }

        It 'Should create WSL network test output' {
            $output = New-MockWSLOutput -Command 'NetworkTest' -Success $true

            $output | Should -Match 'google.com'
        }

        It 'Should simulate WSL failure' {
            {
                New-MockWSLOutput -Command 'Echo' -Success $false
            } | Should -Throw
        }
    }

    Context 'Docker Mock Helpers' {

        It 'Should create Docker version output' {
            $output = New-MockDockerOutput -Command 'Version' -Success $true

            $output | Should -Match 'Docker Engine'
        }

        It 'Should create Docker container list' {
            $output = New-MockDockerOutput -Command 'ContainerList' -Success $true

            $output | Should -Be 'rag-redis'
        }

        It 'Should simulate Docker daemon not running' {
            {
                New-MockDockerOutput -Command 'PS' -Success $false
            } | Should -Throw -ExpectedMessage 'Cannot connect to Docker daemon'
        }
    }
}

Describe 'Test Best Practices Examples' {

    Context 'Good Test Structure' {

        It 'Has a clear, descriptive name' {
            # Test names should clearly state what is being tested
            # Good: "Should start service when stopped"
            # Bad: "Test1" or "Works"

            $true | Should -Be $true
        }

        It 'Tests one thing' {
            # Each test should verify one specific behavior
            # Don't combine multiple unrelated assertions

            $result = Test-SimpleFunction -Input 'test'
            $result | Should -Match 'Processed'
        }

        It 'Follows Arrange-Act-Assert pattern' {
            # Arrange - Setup test data
            $input = 'test value'

            # Act - Execute the function
            $result = Test-SimpleFunction -Input $input

            # Assert - Verify the result
            $result | Should -Be 'Processed: test value'
        }

        It 'Uses Should -Invoke to verify behavior' {
            # Mock external dependency
            Mock Write-Host { }

            # Call function that uses Write-Host (if it existed)
            # Test-SimpleFunction -Input 'test'

            # Verify Write-Host was called
            # Should -Invoke Write-Host -Times 1
            $true | Should -Be $true
        }
    }

    Context 'Common Patterns' {

        It 'Should test null handling' {
            # Good practice: test edge cases
            { Test-SimpleFunction -Input $null } | Should -Not -Throw
        }

        It 'Should test empty strings' {
            $result = Test-SimpleFunction -Input ''
            $result | Should -Be 'Processed: '
        }

        It 'Should test boundary conditions' {
            # Test limits, edge cases, boundaries
            # Example: max length, min value, etc.
            $true | Should -Be $true
        }
    }
}

AfterAll {
    # Cleanup
    Clear-TestState
}
