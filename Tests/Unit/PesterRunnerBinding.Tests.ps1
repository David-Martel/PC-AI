Describe 'test runner candidate module binding' -Tag 'Unit', 'Fast', 'Portable' -Skip:($PSVersionTable.PSVersion.Major -lt 7) {
    BeforeAll {
        $script:CandidateRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        $script:BindingChild = Join-Path $TestDrive 'runner-binding-child.ps1'
        @'
param([string]$Runner,[string]$Kind,[int]$Fail)
$ErrorActionPreference='Stop'
Import-Module Pester -RequiredVersion 5.7.1
$global:BindingFailure=$Fail
$global:BindingObservation=$null
function global:Invoke-Pester {
    param($Configuration)
    $global:BindingObservation=[ordered]@{
        First=($env:PSModulePath -split [regex]::Escape([string][IO.Path]::PathSeparator))[0]
        Resolved=@(Get-Module PC-AI.Common -ListAvailable)[0].Path
    }
    if($global:BindingFailure){throw 'Synthetic runner failure before test execution'}
    [pscustomobject]@{TotalCount=1;PassedCount=1;FailedCount=0;SkippedCount=0;NotRunCount=0;FailedContainersCount=0;FailedBlocksCount=0;Duration=[TimeSpan]::Zero;CodeCoverage=$null}
}
$global:BindingDispatch=${function:Invoke-Pester}
function global:Import-Module {
    [CmdletBinding()]
    param([Parameter(Position=0)][string[]]$Name,[version]$MinimumVersion,[version]$RequiredVersion,[switch]$Force)
    Microsoft.PowerShell.Core\Import-Module @PSBoundParameters
    # The runner's real Pester import can re-export Invoke-Pester. Retain only
    # the inert dispatch boundary after forwarding the complete actual import.
    Set-Item -LiteralPath Function:\global:Invoke-Pester -Value $global:BindingDispatch
}
$original='inert-module-path-sentinel'+[IO.Path]::PathSeparator+$env:PSModulePath
$env:PSModulePath=$original
$originalLocation=(Get-Location).Path
$caught=$false
try {
    if($Kind -eq 'Pester'){& $Runner -Type Unit -Tag 'InertBindingFixtureOnly'}
    else {& $Runner -Suite Unit -Tag 'InertBindingFixtureOnly'}
} catch {$caught=$true}
'PCAI_BINDING_RESULT='+([ordered]@{Observed=$global:BindingObservation;Restored=($env:PSModulePath -ceq $original);LocationRestored=((Get-Location).Path -ceq $originalLocation);Caught=$caught}|ConvertTo-Json -Compress)
'@ | Set-Content -LiteralPath $script:BindingChild -Encoding utf8
    }

    # Execute each complete maintained runner with only its test-dispatch boundary
    # inert. Detect stale installed/junction module selection and environment leaks
    # on success and exceptional exit, without discovering/executing product tests.
    It 'pins candidate modules and restores process state for <Runner> failure=<Fail>' -ForEach @(
        @{Runner='.pester.ps1';Kind='Pester';Fail=0},
        @{Runner='.pester.ps1';Kind='Pester';Fail=1},
        @{Runner='Invoke-AllTests.ps1';Kind='All';Fail=0},
        @{Runner='Invoke-AllTests.ps1';Kind='All';Fail=1}
    ) {
        $runnerPath=Join-Path $script:CandidateRoot "Tests/$Runner"
        $start=[Diagnostics.ProcessStartInfo]::new()
        $start.FileName=Join-Path $PSHOME $(if($IsWindows){'pwsh.exe'}else{'pwsh'})
        $start.UseShellExecute=$false
        $start.WorkingDirectory=$TestDrive
        $start.RedirectStandardOutput=$true
        $start.RedirectStandardError=$true
        foreach($argument in @('-NoLogo','-NoProfile','-File',$script:BindingChild,'-Runner',$runnerPath,'-Kind',$Kind,'-Fail',[string]$Fail)){
            $start.ArgumentList.Add($argument)
        }
        $process=[Diagnostics.Process]::new()
        $process.StartInfo=$start
        try {
            [void]$process.Start()
            $stdout=$process.StandardOutput.ReadToEndAsync()
            $stderr=$process.StandardError.ReadToEndAsync()
            if(-not $process.WaitForExit(60000)){
                $process.Kill($true)
                $process.WaitForExit()
                throw 'Owned runner-binding child exceeded its 60-second bound.'
            }
            $output=$stdout.GetAwaiter().GetResult()
            $errorText=$stderr.GetAwaiter().GetResult()
            $process.ExitCode | Should -Be 0 -Because $errorText
            $receiptLine=@($output -split '\r?\n'|Where-Object {$_ -like 'PCAI_BINDING_RESULT=*'})
            $receiptLine.Count | Should -Be 1 -Because $output
            $receipt=$receiptLine[0].Substring('PCAI_BINDING_RESULT='.Length)|ConvertFrom-Json
            $receipt.Observed.First | Should -Be (Join-Path $script:CandidateRoot 'Modules') -Because $output
            $receipt.Observed.Resolved | Should -Be (Join-Path $script:CandidateRoot 'Modules/PC-AI.Common/PC-AI.Common.psm1')
            $receipt.Restored | Should -BeTrue
            $receipt.LocationRestored | Should -BeTrue
            $receipt.Caught | Should -Be ([bool]$Fail)
        } finally {
            $process.Dispose()
        }
    }
}
