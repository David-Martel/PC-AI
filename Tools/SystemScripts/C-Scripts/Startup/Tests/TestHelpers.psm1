<#
.SYNOPSIS
    Reusable test utilities for WSL/Docker Services Pester tests

.DESCRIPTION
    This module provides helper functions for creating mocks, simulating scenarios,
    and asserting complex conditions in Pester tests.

.NOTES
    Import in BeforeAll: Import-Module .\TestHelpers.psm1 -Force
#>

#region Mock Object Factories

function New-MockService {
    <#
    .SYNOPSIS
        Creates a mock Windows Service object

    .PARAMETER Name
        Service name

    .PARAMETER Status
        Service status: Running, Stopped, Paused

    .PARAMETER StartType
        Service start type: Automatic, Manual, Disabled

    .EXAMPLE
        $service = New-MockService -Name 'LxssManager' -Status 'Running'
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Name,

        [Parameter()]
        [ValidateSet('Running', 'Stopped', 'Paused', 'StartPending', 'StopPending')]
        [string]$Status = 'Running',

        [Parameter()]
        [ValidateSet('Automatic', 'Manual', 'Disabled')]
        [string]$StartType = 'Automatic'
    )

    $service = New-Object PSObject -Property @{
        Name          = $Name
        Status        = $Status
        StartType     = $StartType
        DisplayName   = $Name
        ServiceName   = $Name
        DependentServices = @()
        ServicesDependedOn = @()
        CanStop       = $true
        CanShutdown   = $true
        CanPauseAndContinue = $false
    }

    $service.PSObject.TypeNames.Insert(0, 'System.ServiceProcess.ServiceController')
    return $service
}

function New-MockProcess {
    <#
    .SYNOPSIS
        Creates a mock Process object

    .PARAMETER ProcessName
        Name of the process

    .PARAMETER Id
        Process ID

    .PARAMETER HasExited
        Whether the process has exited

    .PARAMETER StartTime
        Process start time

    .EXAMPLE
        $process = New-MockProcess -ProcessName 'Docker Desktop' -Id 1234
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ProcessName,

        [Parameter()]
        [int]$Id = (Get-Random -Minimum 1000 -Maximum 9999),

        [Parameter()]
        [bool]$HasExited = $false,

        [Parameter()]
        [datetime]$StartTime = (Get-Date).AddMinutes(-10)
    )

    [PSCustomObject]@{
        PSTypeName   = 'System.Diagnostics.Process'
        ProcessName  = $ProcessName
        Id           = $Id
        HasExited    = $HasExited
        StartTime    = $StartTime
        Responding   = $true
        MainWindowHandle = [IntPtr]::Zero
        MainWindowTitle  = $ProcessName
    }
}

function New-MockDockerContainer {
    <#
    .SYNOPSIS
        Creates a mock Docker container object

    .PARAMETER Name
        Container name

    .PARAMETER Id
        Container ID

    .PARAMETER Status
        Container status: running, exited, created, paused

    .PARAMETER Image
        Container image

    .EXAMPLE
        $container = New-MockDockerContainer -Name 'rag-redis' -Status 'running'
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Name,

        [Parameter()]
        [string]$Id = (New-Guid).ToString().Substring(0, 12),

        [Parameter()]
        [ValidateSet('running', 'exited', 'created', 'paused', 'restarting')]
        [string]$Status = 'running',

        [Parameter()]
        [string]$Image = 'redis:latest'
    )

    [PSCustomObject]@{
        ID      = $Id
        Name    = $Name
        Status  = $Status
        Image   = $Image
        Created = (Get-Date).AddHours(-1).ToString('yyyy-MM-dd HH:mm:ss')
        Ports   = '6379/tcp'
    }
}

#endregion

#region Retry Scenario Helpers

function New-RetryScenario {
    <#
    .SYNOPSIS
        Creates a scriptblock that simulates retry behavior

    .PARAMETER FailCount
        Number of times to fail before succeeding

    .PARAMETER SuccessValue
        Value to return on success

    .PARAMETER FailureValue
        Value to return on failure

    .PARAMETER FailureException
        Exception to throw on failure (if specified, throws instead of returning FailureValue)

    .EXAMPLE
        $script:RetryAttempt = 0
        $retryBlock = New-RetryScenario -FailCount 2 -SuccessValue $true
        $result = & $retryBlock  # Returns $false first 2 times, then $true

    .EXAMPLE
        # Throw exception on failure
        $retryBlock = New-RetryScenario -FailCount 2 -SuccessValue 'OK' -FailureException 'Service not ready'
    #>
    [CmdletBinding()]
    param(
        [Parameter()]
        [int]$FailCount = 2,

        [Parameter()]
        [object]$SuccessValue = $true,

        [Parameter()]
        [object]$FailureValue = $false,

        [Parameter()]
        [string]$FailureException
    )

    if (-not (Get-Variable -Name RetryAttempt -Scope Script -ErrorAction SilentlyContinue)) {
        $script:RetryAttempt = 0
    }

    return {
        $script:RetryAttempt++

        if ($script:RetryAttempt -le $using:FailCount) {
            if ($using:FailureException) {
                throw $using:FailureException
            } else {
                return $using:FailureValue
            }
        }

        return $using:SuccessValue
    }.GetNewClosure()
}

function Reset-RetryCounter {
    <#
    .SYNOPSIS
        Resets the retry attempt counter

    .EXAMPLE
        Reset-RetryCounter
    #>
    [CmdletBinding()]
    param()

    $script:RetryAttempt = 0
}

function Assert-RetryCount {
    <#
    .SYNOPSIS
        Asserts that retry count matches expected value

    .PARAMETER ExpectedCount
        Expected number of retry attempts

    .PARAMETER Message
        Custom assertion message

    .EXAMPLE
        Assert-RetryCount -ExpectedCount 3 -Message "Should retry 3 times"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [int]$ExpectedCount,

        [Parameter()]
        [string]$Message = "Retry count mismatch"
    )

    if (-not (Get-Variable -Name RetryAttempt -Scope Script -ErrorAction SilentlyContinue)) {
        throw "RetryAttempt variable not found. Use New-RetryScenario first."
    }

    $script:RetryAttempt | Should -Be $ExpectedCount -Because $Message
}

#endregion

#region Timing Helpers

function Measure-ExecutionTime {
    <#
    .SYNOPSIS
        Measures execution time of a scriptblock

    .PARAMETER ScriptBlock
        Code to measure

    .PARAMETER MaxSeconds
        Maximum allowed execution time

    .EXAMPLE
        $elapsed = Measure-ExecutionTime -ScriptBlock { Start-Sleep -Seconds 1 }
        $elapsed.TotalSeconds | Should -BeLessThan 2
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [scriptblock]$ScriptBlock,

        [Parameter()]
        [double]$MaxSeconds
    )

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    try {
        & $ScriptBlock
    } finally {
        $stopwatch.Stop()
    }

    if ($MaxSeconds -and $stopwatch.Elapsed.TotalSeconds -gt $MaxSeconds) {
        throw "Execution exceeded maximum time: $($stopwatch.Elapsed.TotalSeconds)s > $MaxSeconds s"
    }

    return $stopwatch.Elapsed
}

function Assert-ExecutionTime {
    <#
    .SYNOPSIS
        Asserts that execution time is within expected range

    .PARAMETER ScriptBlock
        Code to measure

    .PARAMETER MinSeconds
        Minimum expected execution time

    .PARAMETER MaxSeconds
        Maximum allowed execution time

    .EXAMPLE
        Assert-ExecutionTime -ScriptBlock { Start-Sleep -Milliseconds 500 } -MinSeconds 0.4 -MaxSeconds 0.6
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [scriptblock]$ScriptBlock,

        [Parameter()]
        [double]$MinSeconds = 0,

        [Parameter(Mandatory)]
        [double]$MaxSeconds
    )

    $elapsed = Measure-ExecutionTime -ScriptBlock $ScriptBlock

    $elapsed.TotalSeconds | Should -BeGreaterOrEqual $MinSeconds -Because "Execution too fast"
    $elapsed.TotalSeconds | Should -BeLessOrEqual $MaxSeconds -Because "Execution too slow"
}

#endregion

#region Mock Verification Helpers

function Assert-MockInvoked {
    <#
    .SYNOPSIS
        Extended mock invocation assertion with better error messages

    .PARAMETER CommandName
        Name of mocked command

    .PARAMETER Times
        Expected number of invocations

    .PARAMETER Exactly
        Whether count must be exact

    .PARAMETER ParameterFilter
        Parameter filter scriptblock

    .PARAMETER Message
        Custom error message

    .EXAMPLE
        Assert-MockInvoked -CommandName 'Start-Service' -Times 1 -Exactly -Message "Should start service once"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$CommandName,

        [Parameter()]
        [int]$Times = 1,

        [Parameter()]
        [switch]$Exactly,

        [Parameter()]
        [scriptblock]$ParameterFilter,

        [Parameter()]
        [string]$Message
    )

    $invokeParams = @{
        CommandName = $CommandName
        Times       = $Times
        Exactly     = $Exactly
    }

    if ($ParameterFilter) {
        $invokeParams['ParameterFilter'] = $ParameterFilter
    }

    try {
        Should -Invoke @invokeParams
    } catch {
        if ($Message) {
            throw "$Message - $_"
        } else {
            throw
        }
    }
}

#endregion

#region WSL-Specific Helpers

function New-MockWSLOutput {
    <#
    .SYNOPSIS
        Creates realistic WSL command output

    .PARAMETER Command
        WSL command type

    .PARAMETER Success
        Whether the command succeeded

    .EXAMPLE
        $output = New-MockWSLOutput -Command 'ListDistros' -Success $true
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('ListDistros', 'Echo', 'NetworkTest', 'ServiceStatus')]
        [string]$Command,

        [Parameter()]
        [bool]$Success = $true
    )

    switch ($Command) {
        'ListDistros' {
            if ($Success) {
                return @('Ubuntu', 'Ubuntu-20.04')
            } else {
                throw 'WSL is not installed'
            }
        }

        'Echo' {
            if ($Success) {
                return 'test'
            } else {
                throw 'WSL command failed'
            }
        }

        'NetworkTest' {
            if ($Success) {
                return @(
                    'Server:  8.8.8.8',
                    'Address: 8.8.8.8#53',
                    '',
                    'Name:    google.com',
                    'Address: 142.250.185.46'
                )
            } else {
                throw 'Network unreachable'
            }
        }

        'ServiceStatus' {
            if ($Success) {
                return 'running'
            } else {
                return 'stopped'
            }
        }
    }
}

#endregion

#region Docker-Specific Helpers

function New-MockDockerOutput {
    <#
    .SYNOPSIS
        Creates realistic Docker command output

    .PARAMETER Command
        Docker command type

    .PARAMETER Success
        Whether the command succeeded

    .EXAMPLE
        $output = New-MockDockerOutput -Command 'Version' -Success $true
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('Version', 'PS', 'Info', 'ContainerList', 'ContainerStart')]
        [string]$Command,

        [Parameter()]
        [bool]$Success = $true
    )

    switch ($Command) {
        'Version' {
            if ($Success) {
                return @(
                    'Client: Docker Engine - Community',
                    ' Version:           24.0.7',
                    ' API version:       1.43',
                    'Server: Docker Desktop',
                    ' Version:          24.0.7'
                )
            } else {
                throw 'Cannot connect to Docker daemon'
            }
        }

        'PS' {
            if ($Success) {
                return @(
                    'CONTAINER ID   IMAGE          COMMAND                  STATUS',
                    'abc123def456   redis:latest   "docker-entrypoint.s…"   Up 5 minutes'
                )
            } else {
                throw 'Cannot connect to Docker daemon'
            }
        }

        'Info' {
            if ($Success) {
                return @(
                    'Containers: 1',
                    ' Running: 1',
                    ' Paused: 0',
                    ' Stopped: 0',
                    'Images: 5'
                )
            } else {
                throw 'Cannot connect to Docker daemon'
            }
        }

        'ContainerList' {
            if ($Success) {
                return 'rag-redis'
            } else {
                return ''
            }
        }

        'ContainerStart' {
            if ($Success) {
                return 'rag-redis'
            } else {
                throw 'No such container'
            }
        }
    }
}

#endregion

#region Cleanup Helpers

function Clear-TestState {
    <#
    .SYNOPSIS
        Clears all test state variables

    .EXAMPLE
        Clear-TestState
    #>
    [CmdletBinding()]
    param()

    # Remove common test variables
    $testVars = @(
        'RetryAttempt',
        'PhaseOrder',
        'SleepDurations',
        'InvocationCount',
        'LastCommand'
    )

    foreach ($var in $testVars) {
        if (Get-Variable -Name $var -Scope Script -ErrorAction SilentlyContinue) {
            Remove-Variable -Name $var -Scope Script -Force
        }
    }
}

#endregion

# Export module members
Export-ModuleMember -Function @(
    # Mock factories
    'New-MockService',
    'New-MockProcess',
    'New-MockDockerContainer',

    # Retry helpers
    'New-RetryScenario',
    'Reset-RetryCounter',
    'Assert-RetryCount',

    # Timing helpers
    'Measure-ExecutionTime',
    'Assert-ExecutionTime',

    # Mock verification
    'Assert-MockInvoked',

    # WSL helpers
    'New-MockWSLOutput',

    # Docker helpers
    'New-MockDockerOutput',

    # Cleanup
    'Clear-TestState'
)
