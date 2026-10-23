#Requires -Version 7.0
[CmdletBinding()]
param([string]$CacheSourcePath=(Join-Path $PSScriptRoot '../../Modules/PC-AI.Common/Public/Get-PcaiSharedCache.ps1'))
$ErrorActionPreference='Stop'
. $CacheSourcePath
$pwsh=(Get-Process -Id $PID).Path
$child=(Resolve-Path (Join-Path $PSScriptRoot '../Fixtures/PortableCaptureChild.ps1')).Path
$ownedRoot=Join-Path ([IO.Path]::GetTempPath()) ('pcai-portable-capture-'+[guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($ownedRoot)
$rows=[Collections.Generic.List[object]]::new()
function Assert-Control([bool]$Condition,[string]$Message){if(-not $Condition){throw $Message}}
function Invoke-Owned([string]$Mode,[int]$TimeoutMs=10000,[Threading.CancellationToken]$Token=[Threading.CancellationToken]::None,[string]$PidPath){
    $arguments=@('-NoLogo','-NoProfile','-File',$child,'-Mode',$Mode)
    if($PidPath){$arguments+=@('-PidPath',$PidPath)}
    $watch=[Diagnostics.Stopwatch]::StartNew()
    $parameters=@{FilePath=$pwsh;Arguments=$arguments;TimeoutMs=$TimeoutMs}
    if((Get-Command Invoke-PcaiPortableProcessCapture).Parameters.ContainsKey('CancellationToken')){$parameters.CancellationToken=$Token}
    $result=Invoke-PcaiPortableProcessCapture @parameters
    $watch.Stop()
    Assert-Control ($watch.ElapsedMilliseconds -lt $TimeoutMs+4000) "Unbounded capture: $Mode"
    foreach($field in @('Success','ExitCode','TimedOut','StdOut','StdErr')){Assert-Control ($result.PSObject.Properties.Name -contains $field) "Missing predecessor field: $field"}
    $rows.Add([pscustomobject]@{Mode=$Mode;Success=$result.Success;ExitCode=$result.ExitCode;TimedOut=$result.TimedOut;Cancelled=$result.Cancelled;Closure=$result.Closure;ElapsedMs=$watch.ElapsedMilliseconds;StdoutBytes=$result.StdoutBytes;StderrBytes=$result.StderrBytes;OriginalPid=$result.OriginalPid;Descendants=$result.Descendants})
    return $result
}
function Close-OwnedDescendant([string]$Path){
    if(-not(Test-Path -LiteralPath $Path)){return}
    $identity=Get-Content -LiteralPath $Path -Raw|ConvertFrom-Json
    $process=Get-Process -Id $identity.Pid -ErrorAction SilentlyContinue
    if($process){
        try{
            # An exact observed start identity, never PID alone, authorizes fixture cleanup.
            if($IsLinux){
                $statPath="/proc/$($identity.Pid)/stat"
                try{$stat=[IO.File]::ReadAllText($statPath)}catch{
                    if(-not [IO.File]::Exists($statPath)){return}
                    throw
                }
                $fields=$stat.Substring($stat.LastIndexOf(')')+2).Split(' ',[StringSplitOptions]::RemoveEmptyEntries)
                $bootId=[IO.File]::ReadAllText('/proc/sys/kernel/random/boot_id').Trim()
                Assert-Control ($fields[19] -ceq $identity.LinuxStartTicks -and $bootId -ceq $identity.LinuxBootId) 'Owned descendant kernel identity changed'
            }else{
                Assert-Control ($process.StartTime.ToUniversalTime().Ticks -eq $identity.StartUtcTicks) 'Owned descendant identity changed'
            }
            if(-not $process.HasExited){$process.Kill($true)}
            Assert-Control ($process.WaitForExit(5000)) 'Owned descendant cleanup exceeded its bound'
        }finally{$process.Dispose()}
    }
}
try{
    # Both writes exceed ordinary pipe capacities. The old wait-before-drain
    # implementation times out here; both complete outputs are required.
    $result=Invoke-Owned Both
    Assert-Control ($result.Success -and $result.ExitCode -eq 0) 'Concurrent large-pipe capture failed'
    Assert-Control ($result.StdOut.Length -eq 262144 -and $result.StdErr.Length -eq 262144) 'Lost large stdout/stderr bytes'
    Assert-Control ($result.ReadersSettled -and $result.HandlesClosed -and $result.ParentExited) 'Successful capture lost original closure'
    Write-Output 'PASS: simultaneous large stdout/stderr, five fields and original closure'

    $result=Invoke-Owned Nonzero
    Assert-Control (-not $result.Success -and $result.ExitCode -eq 23 -and -not $result.TimedOut) 'Nonzero exit misclassified'
    Assert-Control ($result.StdOut -ceq 'owned-out' -and $result.StdErr -ceq 'owned-error') 'Nonzero output lost'
    Write-Output 'PASS: nonzero exit retains both outputs'

    $result=Invoke-Owned Sleep 1200
    Assert-Control (-not $result.Success -and $result.TimedOut -and $result.ExitCode -eq -2) 'Deadline was not enforced'
    Assert-Control ($result.ParentExited -or $result.Closure -ceq 'HOLD') 'Timeout fabricated root closure'
    Write-Output 'PASS: operation deadline and truthful bounded cleanup'

    $cancel=[Threading.CancellationTokenSource]::new()
    try{
        $cancel.CancelAfter(1200)
        $result=Invoke-Owned Sleep 10000 $cancel.Token
        Assert-Control (-not $result.Success -and $result.Cancelled -and -not $result.TimedOut -and $result.ExitCode -eq -4) 'Caller cancellation was ignored/misclassified'
    }finally{$cancel.Dispose()}
    Write-Output 'PASS: in-flight caller cancellation'

    $cancel=[Threading.CancellationTokenSource]::new()
    try{
        $cancel.Cancel()
        $result=Invoke-PcaiPortableProcessCapture -FilePath $pwsh -TimeoutMs 1000 -CancellationToken $cancel.Token
        Assert-Control ($result.ExitCode -eq -4 -and $result.Closure -ceq 'NOT_STARTED') 'Precancelled work launched'
    }finally{$cancel.Dispose()}
    foreach($timeout in @(0,-1)){
        $result=Invoke-PcaiPortableProcessCapture -FilePath $pwsh -TimeoutMs $timeout
        Assert-Control ($result.ExitCode -eq -3 -and $result.Closure -ceq 'NOT_STARTED') 'Nonpositive deadline was admitted'
    }
    Write-Output 'PASS: precancelled and invalid deadline work refused'

    # The fifth submission must fail before launching while four original
    # captures retain their slots. Cancellation must close those same owners.
    $admissionDeadline=[DateTime]::UtcNow.AddSeconds(5)
    while(@([Pcai.Common.PortableProcessCaptureV1]::PendingCustody()).Count -and [DateTime]::UtcNow -lt $admissionDeadline){[Threading.Thread]::Sleep(20)}
    Assert-Control (@([Pcai.Common.PortableProcessCaptureV1]::PendingCustody()).Count -eq 0) 'Prior original custody still occupies admission'
    $admitted=[Collections.Generic.List[object]]::new()
    $rejected=$null
    try{
        Assert-Control ([Pcai.Common.PortableProcessCaptureV1]::MaximumOutstanding -eq 4) 'Reviewed custody capacity changed'
        for($i=0;$i -lt 4;$i++){
            $admitted.Add([Pcai.Common.PortableProcessCaptureV1]::Begin($pwsh,@('-NoLogo','-NoProfile','-File',$child,'-Mode','Sleep'),10000,[Threading.CancellationToken]::None))
        }
        $rejected=[Pcai.Common.PortableProcessCaptureV1]::Begin($pwsh,@('-NoLogo','-NoProfile','-File',$child,'-Mode','Sleep'),10000,[Threading.CancellationToken]::None)
        Assert-Control $rejected.Completion.IsCompleted 'Capacity rejection queued another owned worker'
        $refusal=$rejected.Completion.GetAwaiter().GetResult()
        Assert-Control (-not $refusal.Success -and $refusal.ExitCode -eq -3 -and $refusal.PrimaryFailure -ceq 'ADMISSION') 'Fifth capture bypassed finite admission'
        Assert-Control ($null -eq $refusal.OriginalPid -and $refusal.Closure -ceq 'NOT_STARTED') 'Rejected capture launched or fabricated custody'
    }finally{
        if($rejected -and -not $rejected.Completion.IsCompleted){$rejected.Cancel();$admitted.Add($rejected)}
        foreach($original in $admitted){$original.Cancel()}
        foreach($original in $admitted){
            Assert-Control ($original.Completion.Wait(5000)) 'Original admitted capture did not settle after cancellation'
            $closed=$original.Completion.GetAwaiter().GetResult()
            Assert-Control ($closed.HandlesClosed -and $closed.Closure -in @('ROOT_AND_PIPES_CLOSED','NOT_STARTED')) 'Original admitted capture lost exact closure'
        }
    }
    Write-Output 'PASS: four-slot admission rejects fifth before launch and closes original owners'

    $result=Invoke-Owned Overflow
    Assert-Control (-not $result.Success -and -not $result.TimedOut -and $result.ExitCode -eq -3) 'Unbounded output admitted'
    Assert-Control ($result.StdoutBytes -le [Pcai.Common.PortableProcessCaptureV1]::StdoutLimit) 'Output limit exceeded'
    Write-Output 'PASS: finite output budget'

    foreach($mode in @('TreeParent','HeldParent')){
        $identity=Join-Path $ownedRoot ($mode+'.json')
        try{
            $result=Invoke-Owned $mode 3000 ([Threading.CancellationToken]::None) $identity
            Assert-Control (Test-Path -LiteralPath $identity) "Descendant fixture did not start: $mode"
            Assert-Control ($result.Descendants -ceq 'UNVERIFIED') 'Unproved descendant closure claimed'
            if($mode -eq 'TreeParent'){Assert-Control ($result.TimedOut -and -not $result.Success) 'Live tree escaped capture deadline'}
            else{Assert-Control ($result.TimedOut -and -not $result.Success -and $result.ParentExited) 'Exited parent with held pipe was not bounded by total deadline'}
        }finally{Close-OwnedDescendant $identity}
    }
    Write-Output 'PASS: live descendant and exited-parent held-pipe cases; descendants remain unverified'

    if(-not $IsWindows){
        $cancel=[Threading.CancellationTokenSource]::new()
        try{
            $cancel.CancelAfter(1200)
            $result=Invoke-PcaiProcessCapture -FilePath $pwsh -Arguments @('-NoLogo','-NoProfile','-File',$child,'-Mode','Sleep') -TimeoutMs 10000 -CancellationToken $cancel.Token
            Assert-Control ($result.Cancelled -and $result.ExitCode -eq -4 -and -not $result.TimedOut) 'Public portable dispatch lost cancellation'
        }finally{$cancel.Dispose()}
        Write-Output 'PASS: real non-Windows public dispatch forwards cancellation'
    }else{Write-Output 'SKIP: non-Windows public dispatch requires Linux/macOS execution'}
    $deadline=[DateTime]::UtcNow.AddSeconds(5)
    while(@([Pcai.Common.PortableProcessCaptureV1]::PendingCustody()).Count -and [DateTime]::UtcNow -lt $deadline){[Threading.Thread]::Sleep(20)}
    Assert-Control (@([Pcai.Common.PortableProcessCaptureV1]::PendingCustody()).Count -eq 0) 'Original custody remains held after owned-child cleanup'
    $rows|ConvertTo-Json -Depth 4|Write-Output
    Write-Output 'PASS: portable process capture owned-child contracts'
}finally{
    foreach($mode in @('TreeParent','HeldParent')){Close-OwnedDescendant (Join-Path $ownedRoot ($mode+'.json'))}
    $resolved=[IO.Path]::GetFullPath($ownedRoot)
    Assert-Control ($resolved.StartsWith([IO.Path]::GetFullPath([IO.Path]::GetTempPath()),[StringComparison]::OrdinalIgnoreCase)) 'Fixture cleanup escaped temporary root'
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
