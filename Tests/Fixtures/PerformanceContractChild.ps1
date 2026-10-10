#Requires -Version 7.0
[CmdletBinding()]
param([switch]$RunChild, [string]$RequestPath)

function Assert-PerformanceContractResult {
    [CmdletBinding()]
    param([Parameter(Mandatory)]$Result, [Parameter(Mandatory)][int]$ExpectedCount)
    if ($ExpectedCount -le 0) { throw 'Expected test cardinality must be positive.' }
    foreach ($name in @('Passed','Failed','Skipped','NotRun','Total','FailedContainers','FailedBlocks')) {
        $property=$Result.PSObject.Properties[$name]
        if ($null -eq $property -or $null -eq $property.Value -or ($property.Value -isnot [int] -and $property.Value -isnot [long]) -or $property.Value -lt 0) { throw "Missing or invalid test count: $name" }
    }
    if ($Result.Failed -or $Result.Skipped -or $Result.NotRun -or $Result.FailedContainers -or $Result.FailedBlocks) { throw 'Body, hook, container or skipped-test result prevents qualification.' }
    if ($Result.Total -ne $ExpectedCount -or $Result.Passed -ne $ExpectedCount -or ($Result.Passed+$Result.Failed+$Result.Skipped+$Result.NotRun) -ne $Result.Total) { throw 'Test discovery/cardinality reconciliation failed.' }
    if ($Result.SourceStable -isnot [bool] -or -not $Result.SourceStable) { throw 'Child source drift prevents qualification.' }
    if (@($Result.Cases).Count -ne $ExpectedCount) { throw 'Raw test-case cardinality mismatch.' }
    foreach ($case in $Result.Cases) { if ($case.Result -cne 'Passed') { throw 'Raw test-case result is not passed.' } }
}

function Get-PerformanceContractTypes {
    @([AppDomain]::CurrentDomain.GetAssemblies() | ForEach-Object {
        $assembly=$_
        foreach ($name in @('PcaiNative.OptimizerModule','PcaiNative.NativeCore')) {
            $type=$assembly.GetType($name,$false)
            if ($type) { [pscustomobject]@{Type=$type.FullName;Assembly=$assembly.FullName;Location=if($assembly.IsDynamic){$null}else{$assembly.Location}} }
        }
    })
}

function Get-PerformanceContractBindings {
    param([string[]]$Paths)
    @($Paths | ForEach-Object { [pscustomobject]@{Path=[IO.Path]::GetFullPath($_);SHA=(Get-FileHash -LiteralPath $_ -ErrorAction Stop).Hash} })
}

function Invoke-PerformanceContractChild {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$FixturePath,
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [Parameter(Mandatory)][string]$EvidenceRoot,
        [Parameter(Mandatory)][string]$PesterManifest,
        [ValidateSet('Consumer','ZeroDiscovery','AfterAllFailure','ContainerFailure','PositiveControl')][string]$Kind='Consumer',
        [ValidateRange(1,180000)][int]$TimeoutMilliseconds=120000
    )
    if (Test-Path -LiteralPath $EvidenceRoot) { throw 'Child evidence collision; preserved output cannot be replaced.' }
    [void][IO.Directory]::CreateDirectory($EvidenceRoot)
    $request=[ordered]@{FixturePath=[IO.Path]::GetFullPath($FixturePath);RepositoryRoot=[IO.Path]::GetFullPath($RepositoryRoot);EvidenceRoot=[IO.Path]::GetFullPath($EvidenceRoot);PesterManifest=[IO.Path]::GetFullPath($PesterManifest);Kind=$Kind;ExpectedCount=if($Kind -ceq 'Consumer'){63}else{1};ParentPid=$PID}
    $requestFile=Join-Path $EvidenceRoot 'request.json'
    [IO.File]::WriteAllText($requestFile,($request|ConvertTo-Json -Depth 4),[Text.UTF8Encoding]::new($false))
    $executable=Join-Path $PSHOME $(if([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT){'pwsh.exe'}else{'pwsh'})
    $start=[Diagnostics.ProcessStartInfo]::new($executable)
    $start.UseShellExecute=$false;$start.CreateNoWindow=$true;$start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
    $helperPath=$MyInvocation.MyCommand.ScriptBlock.File
    if ([string]::IsNullOrWhiteSpace($helperPath)) { throw 'File-backed child helper is required.' }
    foreach ($argument in @('-NoLogo','-NoProfile','-File',$helperPath,'-RunChild','-RequestPath',$requestFile)) { $start.ArgumentList.Add($argument) }
    $start.Environment['TEMP']=Join-Path $EvidenceRoot 'scratch';$start.Environment['TMP']=$start.Environment['TEMP'];$start.Environment['TMPDIR']=$start.Environment['TEMP']
    [void][IO.Directory]::CreateDirectory($start.Environment['TEMP'])
    $process=[Diagnostics.Process]::Start($start)
    if ($null -eq $process) { throw 'Child process was not created.' }
    $resolved=$false;$output=$null;$errors=$null
    try {
        if ($process.Handle -eq [IntPtr]::Zero) { throw 'Owned child handle admission failed.' }
        $output=$process.StandardOutput.ReadToEndAsync();$errors=$process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutMilliseconds)) {
            $process.Kill()
            if (-not $process.WaitForExit(5000)) { throw 'Owned child closure is unresolved after deadline.' }
            $resolved=$true
            $timeoutException=[TimeoutException]::new('Performance contract child exceeded its fixture deadline.')
            $timeoutException.Data['EvidenceRoot']=$EvidenceRoot
            $timeoutException.Data['ChildPid']=$process.Id
            $timeoutException.Data['TimeoutMilliseconds']=$TimeoutMilliseconds
            $timeoutException.Data['ExitCode']=$process.ExitCode
            $timeoutException.Data['OutputTask']=$output
            $timeoutException.Data['ErrorTask']=$errors
            $timeoutException.Data['OutputCompleted']=$output.IsCompletedSuccessfully
            $timeoutException.Data['ErrorCompleted']=$errors.IsCompletedSuccessfully
            $stdoutPath=Join-Path $EvidenceRoot 'stdout.log'
            $stderrPath=Join-Path $EvidenceRoot 'stderr.log'
            $resultPath=Join-Path $EvidenceRoot 'result.json'
            $timeoutException.Data['StdoutPath']=$stdoutPath
            $timeoutException.Data['StderrPath']=$stderrPath
            $timeoutException.Data['ChildResultPath']=$resultPath
            $diagnosticErrors=[Collections.Generic.List[string]]::new()
            foreach($stream in @(@{Task=$output;Path=$stdoutPath},@{Task=$errors;Path=$stderrPath})) {
                if($stream.Task.IsCompletedSuccessfully) {
                    try { [IO.File]::WriteAllText($stream.Path,$stream.Task.GetAwaiter().GetResult(),[Text.UTF8Encoding]::new($false)) }
                    catch { $diagnosticErrors.Add($_.Exception.Message) }
                }
            }
            try {
                if(Test-Path -LiteralPath $resultPath) { $timeoutException.Data['ChildResult']=Get-Content -LiteralPath $resultPath -Raw|ConvertFrom-Json -ErrorAction Stop }
            } catch { $diagnosticErrors.Add($_.Exception.Message) }
            $timeoutException.Data['DiagnosticCaptureErrors']=$diagnosticErrors.ToArray()
            throw $timeoutException
        }
        $resolved=$true
        [IO.File]::WriteAllText((Join-Path $EvidenceRoot 'stdout.log'),$output.GetAwaiter().GetResult(),[Text.UTF8Encoding]::new($false))
        [IO.File]::WriteAllText((Join-Path $EvidenceRoot 'stderr.log'),$errors.GetAwaiter().GetResult(),[Text.UTF8Encoding]::new($false))
        $resultFile=Join-Path $EvidenceRoot 'result.json'
        if (-not (Test-Path -LiteralPath $resultFile)) { throw 'Child emitted no result receipt.' }
        $result=Get-Content -LiteralPath $resultFile -Raw|ConvertFrom-Json -ErrorAction Stop
        if ($result.ChildPid -ne $process.Id -or $result.ParentPid -ne $PID -or $result.Kind -cne $Kind) { throw 'Child receipt process/scope identity mismatch.' }
        try {
            Assert-PerformanceContractResult $result $request.ExpectedCount
            if ($process.ExitCode -ne 0) { throw 'Child exit disagrees with its result receipt.' }
        } catch { $_.Exception.Data['ChildResult']=$result; $_.Exception.Data['EvidenceRoot']=$EvidenceRoot; throw }
        return $result
    } catch {
        if (-not $resolved) { $_.Exception.Data['OwnedProcess']=$process; $_.Exception.Data['OutputTask']=$output; $_.Exception.Data['ErrorTask']=$errors; $_.Exception.Data['EvidenceRoot']=$EvidenceRoot }
        throw
    } finally { if ($resolved) { $process.Dispose() } }
}

if ($RunChild) {
    $ErrorActionPreference='Stop'
    $request=Get-Content -LiteralPath $RequestPath -Raw|ConvertFrom-Json -ErrorAction Stop
    $coverageHelper=Join-Path $request.RepositoryRoot 'Tools/Merge-PesterChildCoverage.ps1'
    $coveragePaths=@(foreach($name in @('Get-PcaiMemoryPressure','Get-PcaiProcessCategories','Get-PcaiOptimizationPlan')){Join-Path $request.RepositoryRoot ('Modules/PC-AI.Performance/Public/'+$name+'.ps1')})
    $paths=@($request.FixturePath,$PSCommandPath,$coverageHelper,(Join-Path $request.RepositoryRoot 'Native/PcaiNative/OptimizerModule.cs'))+$coveragePaths
    $before=Get-PerformanceContractBindings $paths;$typesBefore=@(Get-PerformanceContractTypes)
    if($typesBefore.Count){throw 'Fresh child already contains a PcaiNative boundary.'}
    Import-Module $request.PesterManifest -Force -ErrorAction Stop
    $config=New-PesterConfiguration
    $config.Run.Container=New-PesterContainer -Path $request.FixturePath -Data @{IsolatedChild=$true;RepositoryRoot=$request.RepositoryRoot;ChildCaseKind=$request.Kind}
    $config.Run.PassThru=$true;$config.Output.Verbosity='Detailed';$config.TestResult.Enabled=$true;$config.TestResult.OutputFormat='NUnitXml';$config.TestResult.OutputPath=Join-Path $request.EvidenceRoot 'tests.xml'
    if ($request.Kind -ceq 'Consumer') {
        $config.CodeCoverage.Enabled=$true
        $config.CodeCoverage.Path=$coveragePaths
        $config.CodeCoverage.OutputPath=Join-Path $request.EvidenceRoot 'coverage.xml'
        $config.CodeCoverage.OutputFormat='JaCoCo'
    }
    $result=Invoke-Pester -Configuration $config
    $after=Get-PerformanceContractBindings $paths
    $stable=($before|ConvertTo-Json -Compress)-ceq($after|ConvertTo-Json -Compress)
    $receipt=[pscustomobject]@{Scope='Actual C# wrapper/inert synthetic transport only; no native DLL activation or provider diagnostics';ChildPid=$PID;ParentPid=$request.ParentPid;Kind=$request.Kind;Pester=(Get-Module Pester).Version.ToString();Passed=$result.PassedCount;Failed=$result.FailedCount;Skipped=$result.SkippedCount;NotRun=$result.NotRunCount;Total=$result.TotalCount;FailedContainers=$result.FailedContainersCount;FailedBlocks=$result.FailedBlocksCount;SourceStable=$stable;Before=$before;After=$after;TypesBefore=$typesBefore;TypesAfter=@(Get-PerformanceContractTypes);Cases=@($result.Tests|Select-Object ExpandedName,Result)}
    if ($request.Kind -ceq 'Consumer' -and $stable) {
        . $coverageHelper
        $snapshot=New-PesterCoverageSnapshot -Coverage $result.CodeCoverage -SourceBindings @($before|Where-Object Path -in $coveragePaths)
        $receipt | Add-Member -NotePropertyName Coverage -NotePropertyValue $snapshot
    }
    $receipt|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $request.EvidenceRoot 'result.json') -Encoding utf8
    try { Assert-PerformanceContractResult $receipt $request.ExpectedCount } catch { Write-Error $_ -ErrorAction Continue;exit 1 }
    exit 0
}
