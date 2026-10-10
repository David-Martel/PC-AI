#Requires -Version 7.0
# Actual canonical function; every job/HTTP/wait/progress boundary is inert.
BeforeAll {
    $script:CanonicalHelperPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../Modules/PC-AI.LLM/Private/LLM-Helpers.ps1'))
    $tokens = $null
    $parseErrors = $null
    $helperAst = [Management.Automation.Language.Parser]::ParseFile($script:CanonicalHelperPath, [ref]$tokens, [ref]$parseErrors)
    $top = @($helperAst.EndBlock.Statements)
    if ($parseErrors.Count -ne 0 -or $helperAst.ParamBlock -or $helperAst.BeginBlock -or $helperAst.ProcessBlock -or $helperAst.DynamicParamBlock -or $helperAst.UsingStatements.Count -ne 0 -or $helperAst.EndBlock.Traps.Count -ne 0 -or $top.Count -ne 23 -or @($top | Where-Object { $_ -isnot [Management.Automation.Language.FunctionDefinitionAst] }).Count -ne 0) {
        throw 'Canonical helper must contain only its 23 function declarations before dot-sourcing.'
    }
    $script:CanonicalProgressFunction = @($top | Where-Object { $_.Name -ceq 'Invoke-OpenAIChatWithProgress' })
    if ($script:CanonicalProgressFunction.Count -ne 1) { throw 'Exact canonical progress function required.' }
    $script:CanonicalProgressFunction = $script:CanonicalProgressFunction[0]
    Add-Type -Path ([IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../Fixtures/InertProgressJob.cs'))) -ErrorAction Stop -WarningAction Stop
    . $script:CanonicalHelperPath
    $global:PCAI_ProgressBoundFileR1 = (Get-Command Invoke-OpenAIChatWithProgress).ScriptBlock.File
    function Resolve-PcaiEndpoint { param($ApiUrl, $ProviderName) throw 'Unmocked endpoint boundary forbidden' }
    function Get-VLLMModelInfo { param($ApiUrl, $ModelName) throw 'Unmocked HTTP model boundary forbidden' }
    function Get-VLLMMetricsSnapshot { param($ApiUrl, $ModelName, $TimeoutSeconds) throw 'Unmocked HTTP metric boundary forbidden' }
    function Invoke-FixtureProgress {
        Invoke-OpenAIChatWithProgress -Messages @(@{role='user';content='inert fixture'}) -Model 'inert-model' -TimeoutSeconds 7 -ApiUrl 'http://fixture.invalid:18080' -ProgressIntervalSeconds 1 -ErrorAction Stop
    }
    function Get-FixtureExceptionEvidence {
        param([System.Management.Automation.ErrorRecord]$Record)
        $parts = @($Record.ToString(), $Record.Exception.ToString())
        foreach ($value in $Record.Exception.Data.Values) {
            # Retained custody references are inspected by identity, never formatted.
            if ($value -is [System.Management.Automation.Job]) { continue }
            $parts += ($value | Out-String -Width 4096)
        }
        return $parts -join [Environment]::NewLine
    }
}

Describe 'Actual progress function exact-job custody' -Tag 'Unit', 'LLM', 'Portable', 'PredecessorDetecting' {
    BeforeEach {
        $script:OwnedJob = [PcaiProgressCustodyR1.InertJob]::new('owned', $false)
        $script:ForeignJob = [PcaiProgressCustodyR1.InertJob]::new('foreign', $true)
        $script:PrimaryFailure = [InvalidOperationException]::new('primary-model-failure')
        $script:StopFailure = [InvalidOperationException]::new('secondary-stop-failure')
        $script:RemoveFailure = [InvalidOperationException]::new('secondary-remove-failure')
        $script:CleanupCalls = [Collections.Generic.List[string]]::new()
        Mock Resolve-PcaiEndpoint { 'http://fixture.invalid:18080' }
        Mock Start-Job { $script:OwnedJob }
        Mock Get-VLLMModelInfo { [pscustomobject]@{MaxModelLen=64} }
        Mock Get-VLLMMetricsSnapshot { [pscustomobject]@{KVCacheUsagePerc=0.2;NumRequestsRunning=1;NumRequestsWaiting=0} }
        Mock Invoke-RestMethod { throw 'Unmocked HTTP invocation forbidden' }
        Mock Start-Sleep { throw 'Unconfigured wait forbidden' }
        Mock Write-Progress {}
        Mock Receive-Job {
            if (@($Job).Count -ne 1 -or -not [object]::ReferenceEquals(@($Job)[0], $script:OwnedJob)) { throw 'Wrong receive-job custody' }
            [pscustomobject]@{choices=@([pscustomobject]@{message=[pscustomobject]@{content='useful-response'}})}
        }
        Mock Stop-Job {
            if (@($Job).Count -ne 1 -or -not [object]::ReferenceEquals(@($Job)[0], $script:OwnedJob)) { throw 'Wrong stop-job custody' }
            $script:CleanupCalls.Add('Stop')
        }
        Mock Remove-Job {
            if (@($Job).Count -ne 1 -or -not [object]::ReferenceEquals(@($Job)[0], $script:OwnedJob)) { throw 'Wrong remove-job custody' }
            $script:CleanupCalls.Add('Remove')
        }
    }
    AfterEach {
        # Disposes only two locally constructed inert fixtures; does not invoke a job cmdlet.
        $script:OwnedJob.Dispose()
        $script:ForeignJob.Dispose()
    }

    It 'preserves the response and removes only its exact completed job' {
        $bound = Get-Command Invoke-OpenAIChatWithProgress
        $bound.ScriptBlock.File | Should -Be $script:CanonicalHelperPath
        $bound.ScriptBlock.Ast.Extent.Text | Should -BeExactly $script:CanonicalProgressFunction.Extent.Text
        $actual = Invoke-FixtureProgress
        $actual.message.content | Should -Be 'useful-response'
        $actual.provider | Should -Be 'openai'
        $actual.raw.choices[0].message.content | Should -Be 'useful-response'
        Should -Invoke Start-Job -Exactly -Times 1 -ParameterFilter { $ArgumentList[0] -eq 'http://fixture.invalid:18080/v1/chat/completions' -and $ArgumentList[3] -eq 7 }
        Should -Invoke Remove-Job -Exactly -Times 1 -ParameterFilter { [object]::ReferenceEquals(@($Job)[0], $script:OwnedJob) }
        Should -Invoke Stop-Job -Exactly -Times 0
        Should -Invoke Write-Progress -Exactly -Times 1 -ParameterFilter { $Completed }
        Should -Invoke Invoke-RestMethod -Exactly -Times 0
    }

    It 'stops and removes its started job after model lookup failure while preserving the cause' {
        $script:OwnedJob.Dispose()
        $script:OwnedJob = [PcaiProgressCustodyR1.InertJob]::new('owned', $true)
        Mock Get-VLLMModelInfo { throw $script:PrimaryFailure }
        $caught = $null
        try { Invoke-FixtureProgress | Out-Null } catch { $caught = $_ }
        [object]::ReferenceEquals($caught.Exception, $script:PrimaryFailure) | Should -BeTrue
        Should -Invoke Stop-Job -Exactly -Times 1 -ParameterFilter { [object]::ReferenceEquals(@($Job)[0], $script:OwnedJob) }
        Should -Invoke Remove-Job -Exactly -Times 1 -ParameterFilter { [object]::ReferenceEquals(@($Job)[0], $script:OwnedJob) }
        Should -Invoke Write-Progress -Exactly -Times 1 -ParameterFilter { $Completed }
        Should -Invoke Receive-Job -Exactly -Times 0
    }

    It 'removes its completed job and finishes progress after a receive failure' {
        $script:PrimaryFailure = [InvalidOperationException]::new('primary-receive-failure')
        Mock Receive-Job { throw $script:PrimaryFailure }
        $caught = $null
        try { Invoke-FixtureProgress | Out-Null } catch { $caught = $_ }
        [object]::ReferenceEquals($caught.Exception, $script:PrimaryFailure) | Should -BeTrue
        Should -Invoke Remove-Job -Exactly -Times 1 -ParameterFilter { [object]::ReferenceEquals(@($Job)[0], $script:OwnedJob) }
        Should -Invoke Stop-Job -Exactly -Times 0
        Should -Invoke Write-Progress -Exactly -Times 1 -ParameterFilter { $Completed }
    }

    It 'stops and removes only its active job after polling cancellation' {
        $script:OwnedJob.Dispose()
        $script:OwnedJob = [PcaiProgressCustodyR1.InertJob]::new('owned', $true)
        $script:PrimaryFailure = [OperationCanceledException]::new('primary-poll-cancellation')
        Mock Start-Sleep { throw $script:PrimaryFailure }
        $caught = $null
        try { Invoke-FixtureProgress | Out-Null } catch { $caught = $_ }
        [object]::ReferenceEquals($caught.Exception, $script:PrimaryFailure) | Should -BeTrue
        Should -Invoke Start-Sleep -Exactly -Times 1
        Should -Invoke Stop-Job -Exactly -Times 1 -ParameterFilter { [object]::ReferenceEquals(@($Job)[0], $script:OwnedJob) }
        Should -Invoke Remove-Job -Exactly -Times 1 -ParameterFilter { [object]::ReferenceEquals(@($Job)[0], $script:OwnedJob) }
        Should -Invoke Receive-Job -Exactly -Times 0
        Should -Invoke Write-Progress -Exactly -Times 1 -ParameterFilter { $Completed }
    }

    It 'attempts both cleanup actions and retains secondary failures alongside the primary cause' {
        $script:OwnedJob.Dispose()
        $script:OwnedJob = [PcaiProgressCustodyR1.InertJob]::new('owned', $true)
        Mock Get-VLLMModelInfo { throw $script:PrimaryFailure }
        Mock Stop-Job {
            if (@($Job).Count -ne 1 -or -not [object]::ReferenceEquals(@($Job)[0], $script:OwnedJob)) { throw 'Wrong stop-job custody' }
            $script:CleanupCalls.Add('Stop')
            throw $script:StopFailure
        }
        Mock Remove-Job {
            if (@($Job).Count -ne 1 -or -not [object]::ReferenceEquals(@($Job)[0], $script:OwnedJob)) { throw 'Wrong remove-job custody' }
            $script:CleanupCalls.Add('Remove')
            throw $script:RemoveFailure
        }
        $caught = $null
        try { Invoke-FixtureProgress | Out-Null } catch { $caught = $_ }
        [object]::ReferenceEquals($caught.Exception, $script:PrimaryFailure) | Should -BeTrue
        ($script:CleanupCalls -join ',') | Should -Be 'Stop,Remove'
        $evidence = Get-FixtureExceptionEvidence $caught
        $evidence | Should -Match 'secondary-stop-failure'
        $evidence | Should -Match 'secondary-remove-failure'
        Should -Invoke Write-Progress -Exactly -Times 1 -ParameterFilter { $Completed }
    }

    It 'does not clean an unrelated job when job creation fails' {
        $script:PrimaryFailure = [InvalidOperationException]::new('primary-start-failure')
        Mock Start-Job { throw $script:PrimaryFailure }
        $caught = $null
        try { Invoke-FixtureProgress | Out-Null } catch { $caught = $_ }
        [object]::ReferenceEquals($caught.Exception, $script:PrimaryFailure) | Should -BeTrue
        Should -Invoke Start-Job -Exactly -Times 1
        Should -Invoke Stop-Job -Exactly -Times 0
        Should -Invoke Remove-Job -Exactly -Times 0
        Should -Invoke Receive-Job -Exactly -Times 0
        Should -Invoke Get-VLLMModelInfo -Exactly -Times 0
        $script:ForeignJob.State | Should -Be 'Running'
        Should -Invoke Invoke-RestMethod -Exactly -Times 0
    }

    It 'raises a cleanup-only removal failure and retains the exact unremoved job' {
        $script:RemoveFailure = [InvalidOperationException]::new('cleanup-only-remove-failure')
        Mock Remove-Job {
            if (@($Job).Count -ne 1 -or -not [object]::ReferenceEquals(@($Job)[0], $script:OwnedJob)) { throw 'Wrong remove-job custody' }
            $script:CleanupCalls.Add('Remove')
            throw $script:RemoveFailure
        }
        $caught = $null
        $actual = $null
        try { $actual = Invoke-FixtureProgress } catch { $caught = $_ }
        $actual | Should -BeNullOrEmpty
        [object]::ReferenceEquals($caught.Exception, $script:RemoveFailure) | Should -BeTrue
        @($caught.Exception.Data['PcaiProgressCleanupErrors']).Count | Should -Be 1
        [object]::ReferenceEquals(@($caught.Exception.Data['PcaiProgressCleanupErrors'])[0].Exception, $script:RemoveFailure) | Should -BeTrue
        [object]::ReferenceEquals($caught.Exception.Data['PcaiProgressOwnedJob'], $script:OwnedJob) | Should -BeTrue
        $script:ForeignJob.State | Should -Be 'Running'
        ($script:CleanupCalls -join ',') | Should -Be 'Remove'
        Should -Invoke Receive-Job -Exactly -Times 1
        Should -Invoke Stop-Job -Exactly -Times 0
        Should -Invoke Write-Progress -Exactly -Times 1 -ParameterFilter { $Completed }
    }

    It 'raises a progress-finalization-only failure after removing its completed job' {
        $script:ProgressFailure = [InvalidOperationException]::new('cleanup-only-progress-failure')
        Mock Write-Progress { throw $script:ProgressFailure } -ParameterFilter { $Completed }
        $caught = $null
        $actual = $null
        try { $actual = Invoke-FixtureProgress } catch { $caught = $_ }
        $actual | Should -BeNullOrEmpty
        [object]::ReferenceEquals($caught.Exception, $script:ProgressFailure) | Should -BeTrue
        @($caught.Exception.Data['PcaiProgressCleanupErrors']).Count | Should -Be 1
        [object]::ReferenceEquals(@($caught.Exception.Data['PcaiProgressCleanupErrors'])[0].Exception, $script:ProgressFailure) | Should -BeTrue
        $caught.Exception.Data.Contains('PcaiProgressOwnedJob') | Should -BeFalse
        ($script:CleanupCalls -join ',') | Should -Be 'Remove'
        Should -Invoke Receive-Job -Exactly -Times 1
        Should -Invoke Remove-Job -Exactly -Times 1 -ParameterFilter { [object]::ReferenceEquals(@($Job)[0], $script:OwnedJob) }
        Should -Invoke Stop-Job -Exactly -Times 0
        Should -Invoke Write-Progress -Exactly -Times 1 -ParameterFilter { $Completed }
    }
}
