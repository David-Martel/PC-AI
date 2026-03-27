<#
.SYNOPSIS
    Unit tests for LLM-ErrorHandling module

.DESCRIPTION
    Tests error categorization, retry logic with exponential backoff, and structured error reporting
#>

BeforeAll {
    # Import module under test
    $ModulePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.LLM\PC-AI.LLM.psd1'
    Import-Module $ModulePath -Force -ErrorAction Stop

    # Import the private function directly for testing
    $PrivatePath = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.LLM\Private\LLM-ErrorHandling.ps1'
    . $PrivatePath
}

Describe "Get-LLMErrorCategory" -Tag 'Unit', 'LLM', 'ErrorHandling', 'Fast', 'Portable' {
    Context "When detecting connectivity errors" {
        It "Should classify connection refused as Connectivity" {
            $ex = New-Object System.Net.WebException("Connection refused", [System.Net.WebExceptionStatus]::ConnectFailure)
            $result = Get-LLMErrorCategory -Exception $ex
            $result | Should -Be ([LLMErrorCategory]::Connectivity)
        }

        It "Should classify unable to connect as Connectivity" {
            $ex = New-Object System.Exception("Unable to connect to the remote server")
            $result = Get-LLMErrorCategory -Exception $ex
            $result | Should -Be ([LLMErrorCategory]::Connectivity)
        }

        It "Should classify no such host as Connectivity" {
            $ex = New-Object System.Net.Sockets.SocketException(11001) # Host not found
            $result = Get-LLMErrorCategory -Exception $ex
            $result | Should -Be ([LLMErrorCategory]::Connectivity)
        }
    }

    Context "When detecting timeout errors" {
        It "Should classify timeout as Timeout" {
            $ex = New-Object System.Net.WebException("The operation has timed out", [System.Net.WebExceptionStatus]::Timeout)
            $result = Get-LLMErrorCategory -Exception $ex
            $result | Should -Be ([LLMErrorCategory]::Timeout)
        }

        It "Should classify request timeout as Timeout" {
            $ex = New-Object System.Exception("Request timeout")
            $result = Get-LLMErrorCategory -Exception $ex
            $result | Should -Be ([LLMErrorCategory]::Timeout)
        }
    }

    Context "When detecting rate limit errors" {
        It "Should classify 429 status as RateLimited" {
            $ex = New-Object System.Exception("429: Too Many Requests")
            $result = Get-LLMErrorCategory -Exception $ex
            $result | Should -Be ([LLMErrorCategory]::RateLimited)
        }

        It "Should classify rate limit message as RateLimited" {
            $ex = New-Object System.Exception("Rate limit exceeded")
            $result = Get-LLMErrorCategory -Exception $ex
            $result | Should -Be ([LLMErrorCategory]::RateLimited)
        }
    }

    Context "When detecting invalid request errors" {
        It "Should classify 400 status as InvalidRequest" {
            $ex = New-Object System.Exception("400: Bad Request")
            $result = Get-LLMErrorCategory -Exception $ex
            $result | Should -Be ([LLMErrorCategory]::InvalidRequest)
        }

        It "Should classify invalid model error as InvalidRequest" {
            $ex = New-Object System.Exception("Invalid model specified")
            $result = Get-LLMErrorCategory -Exception $ex
            $result | Should -Be ([LLMErrorCategory]::InvalidRequest)
        }
    }

    Context "When detecting server errors" {
        It "Should classify 500 status as ServerError" {
            $ex = New-Object System.Exception("500: Internal Server Error")
            $result = Get-LLMErrorCategory -Exception $ex
            $result | Should -Be ([LLMErrorCategory]::ServerError)
        }

        It "Should classify 502 status as ServerError" {
            $ex = New-Object System.Exception("502: Bad Gateway")
            $result = Get-LLMErrorCategory -Exception $ex
            $result | Should -Be ([LLMErrorCategory]::ServerError)
        }

        It "Should classify 503 status as ServerError" {
            $ex = New-Object System.Exception("503: Service Unavailable")
            $result = Get-LLMErrorCategory -Exception $ex
            $result | Should -Be ([LLMErrorCategory]::ServerError)
        }
    }

    Context "When detecting parse errors" {
        It "Should classify JSON parse error as ParseError" {
            $ex = New-Object System.ArgumentException("Invalid JSON")
            $result = Get-LLMErrorCategory -Exception $ex
            $result | Should -Be ([LLMErrorCategory]::ParseError)
        }

        It "Should classify conversion error as ParseError" {
            $ex = New-Object System.Exception("Failed to parse response")
            $result = Get-LLMErrorCategory -Exception $ex
            $result | Should -Be ([LLMErrorCategory]::ParseError)
        }
    }

    Context "When detecting unknown errors" {
        It "Should classify unrecognized error as Unknown" {
            $ex = New-Object System.Exception("Something unexpected happened")
            $result = Get-LLMErrorCategory -Exception $ex
            $result | Should -Be ([LLMErrorCategory]::Unknown)
        }
    }
}

Describe "Invoke-WithRetry" -Tag 'Unit', 'LLM', 'ErrorHandling', 'Slow', 'Portable' {
    Context "When operation succeeds on first try" {
        It "Should return result without retry" {
            $script:callCount = 0
            $result = Invoke-WithRetry -ScriptBlock {
                $script:callCount++
                return "Success"
            }

            $result | Should -Be "Success"
            $script:callCount | Should -Be 1
        }

        It "Should not delay if successful on first attempt" {
            $start = Get-Date
            $result = Invoke-WithRetry -ScriptBlock { "Success" }
            $elapsed = (Get-Date) - $start

            $elapsed.TotalSeconds | Should -BeLessThan 1
        }
    }

    Context "When operation succeeds after transient failure" {
        It "Should retry on connectivity error and succeed" {
            $script:attemptCount = 0
            $result = Invoke-WithRetry -ScriptBlock {
                $script:attemptCount++
                if ($script:attemptCount -eq 1) {
                    throw (New-Object System.Net.WebException("Connection refused", [System.Net.WebExceptionStatus]::ConnectFailure))
                }
                return "Success after retry"
            }

            $result | Should -Be "Success after retry"
            $script:attemptCount | Should -Be 2
        }

        It "Should retry on timeout error and succeed" {
            $script:attemptCount = 0
            $result = Invoke-WithRetry -ScriptBlock {
                $script:attemptCount++
                if ($script:attemptCount -eq 1) {
                    throw (New-Object System.Net.WebException("Timeout", [System.Net.WebExceptionStatus]::Timeout))
                }
                return "Success"
            }

            $result | Should -Be "Success"
            $script:attemptCount | Should -Be 2
        }

        It "Should retry on rate limit error and succeed" {
            $script:attemptCount = 0
            $result = Invoke-WithRetry -ScriptBlock {
                $script:attemptCount++
                if ($script:attemptCount -eq 1) {
                    throw (New-Object System.Exception("429: Too Many Requests"))
                }
                return "Success"
            }

            $result | Should -Be "Success"
            $script:attemptCount | Should -Be 2
        }

        It "Should retry on server error and succeed" {
            $script:attemptCount = 0
            $result = Invoke-WithRetry -ScriptBlock {
                $script:attemptCount++
                if ($script:attemptCount -eq 1) {
                    throw (New-Object System.Exception("500: Internal Server Error"))
                }
                return "Success"
            }

            $result | Should -Be "Success"
            $script:attemptCount | Should -Be 2
        }
    }

    Context "When operation fails with non-retryable error" {
        It "Should not retry on InvalidRequest error" {
            $script:attemptCount = 0
            {
                Invoke-WithRetry -ScriptBlock {
                    $script:attemptCount++
                    throw (New-Object System.Exception("400: Bad Request"))
                } -ErrorAction Stop
            } | Should -Throw

            $script:attemptCount | Should -Be 1
        }

        It "Should not retry on ParseError" {
            $script:attemptCount = 0
            {
                Invoke-WithRetry -ScriptBlock {
                    $script:attemptCount++
                    throw (New-Object System.ArgumentException("Invalid JSON"))
                } -ErrorAction Stop
            } | Should -Throw

            $script:attemptCount | Should -Be 1
        }
    }

    Context "When operation exhausts retries" {
        It "Should retry up to MaxRetries times" {
            $script:attemptCount = 0
            {
                Invoke-WithRetry -ScriptBlock {
                    $script:attemptCount++
                    throw (New-Object System.Net.WebException("Connection refused", [System.Net.WebExceptionStatus]::ConnectFailure))
                } -MaxRetries 3 -ErrorAction Stop
            } | Should -Throw

            $script:attemptCount | Should -Be 4  # Initial attempt + 3 retries
        }

        It "Should throw the last exception after all retries" {
            {
                Invoke-WithRetry -ScriptBlock {
                    throw (New-Object System.Exception("Persistent connection error"))
                } -MaxRetries 2 -ErrorAction Stop
            } | Should -Throw -ExpectedMessage "*Persistent connection error*"
        }
    }

    Context "When using exponential backoff" {
        It "Should apply backoff delays between retries" {
            $script:attemptCount = 0
            $start = Get-Date

            {
                Invoke-WithRetry -ScriptBlock {
                    $script:attemptCount++
                    throw (New-Object System.Net.WebException("Timeout", [System.Net.WebExceptionStatus]::Timeout))
                } -MaxRetries 2 -BackoffSeconds @(1, 2) -ErrorAction Stop
            } | Should -Throw

            $elapsed = (Get-Date) - $start
            # Should have waited at least 1 + 2 = 3 seconds
            $elapsed.TotalSeconds | Should -BeGreaterThan 2.5
        }
    }

    Context "When customizing retryable categories" {
        It "Should only retry specified categories" {
            $script:attemptCount = 0
            {
                Invoke-WithRetry -ScriptBlock {
                    $script:attemptCount++
                    throw (New-Object System.Net.WebException("Connection refused", [System.Net.WebExceptionStatus]::ConnectFailure))
                } -RetryableCategories @('Timeout', 'RateLimited') -ErrorAction Stop
            } | Should -Throw

            $script:attemptCount | Should -Be 1  # No retry because Connectivity not in retryable list
        }

        It "Should retry when error matches custom retryable category" {
            $script:attemptCount = 0
            $result = Invoke-WithRetry -ScriptBlock {
                $script:attemptCount++
                if ($script:attemptCount -eq 1) {
                    throw (New-Object System.Exception("429: Too Many Requests"))
                }
                return "Success"
            } -RetryableCategories @('RateLimited')

            $result | Should -Be "Success"
            $script:attemptCount | Should -Be 2
        }
    }
}

Describe "New-LLMErrorReport" -Tag 'Unit', 'LLM', 'ErrorHandling', 'Fast', 'Portable' {
    Context "When creating error report from exception" {
        It "Should include exception details" {
            $ex = New-Object System.Exception("Test error message")
            $report = New-LLMErrorReport -Exception $ex

            $report | Should -Not -BeNullOrEmpty
            $report.Message | Should -Be "Test error message"
            $report.Category | Should -Be ([LLMErrorCategory]::Unknown)
            $report.Timestamp | Should -Not -BeNullOrEmpty
        }

        It "Should classify error category" {
            $ex = New-Object System.Net.WebException("Connection refused", [System.Net.WebExceptionStatus]::ConnectFailure)
            $report = New-LLMErrorReport -Exception $ex

            $report.Category | Should -Be ([LLMErrorCategory]::Connectivity)
        }

        It "Should include exception type" {
            $ex = New-Object System.ArgumentException("Invalid argument")
            $report = New-LLMErrorReport -Exception $ex

            $report.ExceptionType | Should -Be "System.ArgumentException"
        }
    }

    Context "When including operation context" {
        It "Should include operation name" {
            $ex = New-Object System.Exception("Test error")
            $report = New-LLMErrorReport -Exception $ex -Operation "Send-OllamaRequest"

            $report.Operation | Should -Be "Send-OllamaRequest"
        }

        It "Should include context data" {
            $ex = New-Object System.Exception("Test error")
            $context = @{
                Model = "llama3.2"
                ApiUrl = "http://localhost:11434"
                Timeout = 30
            }
            $report = New-LLMErrorReport -Exception $ex -Operation "Invoke-LLMChat" -Context $context

            $report.Context | Should -Not -BeNullOrEmpty
            $report.Context.Model | Should -Be "llama3.2"
            $report.Context.ApiUrl | Should -Be "http://localhost:11434"
            $report.Context.Timeout | Should -Be 30
        }
    }

    Context "When formatting error report" {
        It "Should create PSCustomObject with required properties" {
            $ex = New-Object System.Exception("Test error")
            $report = New-LLMErrorReport -Exception $ex

            $report | Should -BeOfType [PSCustomObject]
            $report.PSObject.Properties.Name | Should -Contain "Message"
            $report.PSObject.Properties.Name | Should -Contain "Category"
            $report.PSObject.Properties.Name | Should -Contain "ExceptionType"
            $report.PSObject.Properties.Name | Should -Contain "Timestamp"
        }

        It "Should include inner exception if present" {
            $inner = New-Object System.Exception("Inner error")
            $outer = New-Object System.Exception("Outer error", $inner)
            $report = New-LLMErrorReport -Exception $outer

            $report.InnerException | Should -Not -BeNullOrEmpty
            $report.InnerException | Should -BeLike "*Inner error*"
        }

        It "Should be convertible to JSON" {
            $ex = New-Object System.Exception("Test error")
            $report = New-LLMErrorReport -Exception $ex -Operation "TestOp" -Context @{ Key = "Value" }

            { $report | ConvertTo-Json -Depth 5 } | Should -Not -Throw
        }
    }

    Context "When handling complex error scenarios" {
        It "Should handle WebException with response" {
            $ex = New-Object System.Net.WebException("HTTP error", $null, [System.Net.WebExceptionStatus]::ProtocolError, $null)
            $report = New-LLMErrorReport -Exception $ex

            $report.Category | Should -Be ([LLMErrorCategory]::ServerError)
        }

        It "Should handle nested exceptions" {
            $inner1 = New-Object System.Exception("Root cause")
            $inner2 = New-Object System.Exception("Middle layer", $inner1)
            $outer = New-Object System.Exception("Top level", $inner2)
            $report = New-LLMErrorReport -Exception $outer

            $report.InnerException | Should -Not -BeNullOrEmpty
        }
    }
}

AfterAll {
    Remove-Module PC-AI.LLM -Force -ErrorAction SilentlyContinue
}
