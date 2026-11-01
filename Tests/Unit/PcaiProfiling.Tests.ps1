#Requires -Version 7.0
#Requires -Modules @{ModuleName='Pester';ModuleVersion='5.0.0'}
BeforeAll {
    $script:Profiler=Join-Path $PSScriptRoot '../Benchmarks/Invoke-PcaiProfiling.ps1'
    $script:Pwsh=(Get-Command pwsh -CommandType Application | Select-Object -First 1).Source
    $script:Fixture=Join-Path $TestDrive 'startup-fixture.ps1'
    @'
param([string]$Mode,[string]$PidFile)
if($PidFile){[IO.File]::WriteAllText($PidFile,[string]$PID)}
switch($Mode){
 'good' {[Console]::Out.WriteLine('matched useful output');[Console]::Error.WriteLine('diagnostic')}
 'error' {exit 7}
 'hang' {Start-Sleep -Seconds 30}
 'oversize' {[Console]::Out.Write('x'*1048577)}
}
'@ | Set-Content -LiteralPath $script:Fixture -Encoding utf8NoBOM
}
Describe 'Bounded startup profiling' {
    It 'retains stable repetitions, exact output hashes and source binding' {
        $root=Join-Path $TestDrive 'receipts'
        $first=& $script:Profiler -Scenario custom -CommandPath $script:Pwsh -ArgumentList @('-NoProfile','-File',$script:Fixture,'good') -OutputRoot $root -PassThru
        $second=& $script:Profiler -Scenario custom -CommandPath $script:Pwsh -ArgumentList @('-NoProfile','-File',$script:Fixture,'good') -OutputRoot $root -PassThru
        Split-Path (Split-Path $first.ReceiptPath) -Leaf | Should -Be 'profile-r1'
        Split-Path (Split-Path $second.ReceiptPath) -Leaf | Should -Be 'profile-r2'
        $receipt=Get-Content -LiteralPath $first.ReceiptPath -Raw|ConvertFrom-Json
        $receipt.Measurement.Stdout.Sha256 | Should -Be $second.Measurement.Stdout.Sha256
        $receipt.CommandSha256 | Should -Be (Get-FileHash $script:Pwsh).Hash
        $receipt.ScriptSha256 | Should -Be (Get-FileHash $script:Profiler).Hash
        $receipt.Measurement.ObserverCpuMs | Should -BeGreaterOrEqual 0
        $receipt.Measurement.ElapsedMs | Should -BeGreaterThan 0
        Get-Process -Id $receipt.Measurement.OwnedPid -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
        Get-Content -LiteralPath $first.ReceiptPath -Raw | Should -Not -Match 'matched useful output|diagnostic'
    }
    It 'DryRun and WhatIf create neither directory nor process' {
        foreach($switchName in 'DryRun','WhatIf'){
            $root=Join-Path $TestDrive $switchName
            $pidFile=Join-Path $TestDrive "$switchName.pid"
            $options=@{Scenario='custom';CommandPath=$script:Pwsh;ArgumentList=@('-NoProfile','-File',$script:Fixture,'good',$pidFile);OutputRoot=$root;$switchName=$true}
            & $script:Profiler @options | Out-Null
            Test-Path $root | Should -BeFalse
            Test-Path $pidFile | Should -BeFalse
        }
    }
    It 'supports both help aliases without writes' {
        & $script:Profiler -h | Out-Null
        & $script:Profiler --help | Out-Null
    }
    It 'rejects a real failed child and records no success receipt' {
        $root=Join-Path $TestDrive 'failed'
        { & $script:Profiler -Scenario custom -CommandPath $script:Pwsh -ArgumentList @('-NoProfile','-File',$script:Fixture,'error') -OutputRoot $root } | Should -Throw '*exit code 7*'
        @(Get-ChildItem $root -Recurse -Filter startup-profile.json).Count | Should -Be 0
    }
    It 'closes the owned child at the deadline and preserves unrelated processes' {
        $foreign=Start-Process -FilePath $script:Pwsh -ArgumentList @('-NoProfile','-Command','Start-Sleep -Seconds 30') -WindowStyle Hidden -PassThru
        try {
            foreach($deadline in 1800,1){
                $pidFile=Join-Path $TestDrive "timeout-$deadline.pid"
                $failure=$null
                try { & $script:Profiler -Scenario custom -CommandPath $script:Pwsh -ArgumentList @('-NoProfile','-File',$script:Fixture,'hang',$pidFile) -TimeoutMilliseconds $deadline -OutputRoot (Join-Path $TestDrive "timeout-$deadline") }
                catch { $failure=$_.Exception }
                $failure | Should -Not -BeNullOrEmpty
                $failure.Message | Should -BeLike '*deadline*'
                $owned=[int]$failure.Data['PcaiOwnedPid']
                $owned | Should -BeGreaterThan 0
                $failure.Data['PcaiOwnedProcessExited'] | Should -BeTrue
                if(Test-Path -LiteralPath $pidFile){[int](Get-Content $pidFile)|Should -Be $owned}
                Get-Process -Id $owned -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
                $foreign.Refresh();$foreign.HasExited | Should -BeFalse
            }
        } finally { if(-not $foreign.HasExited){$foreign.Kill();$null=$foreign.WaitForExit(1000)};$foreign.Dispose() }
    }
    It 'bounds actual output and closes its producer' {
        $pidFile=Join-Path $TestDrive 'oversize.pid'
        { & $script:Profiler -Scenario custom -CommandPath $script:Pwsh -ArgumentList @('-NoProfile','-File',$script:Fixture,'oversize',$pidFile) -OutputRoot (Join-Path $TestDrive 'oversize') } | Should -Throw '*capture limit*'
        Get-Process -Id ([int](Get-Content $pidFile)) -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
    }
    It 'honors explicit incomplete bundles without global fallback or writes' {
        $previous=$env:PCAI_NATIVE_BUNDLE_ROOT
        try {
            $env:PCAI_NATIVE_BUNDLE_ROOT=Join-Path $TestDrive 'empty-bundle'
            $null=New-Item -ItemType Directory $env:PCAI_NATIVE_BUNDLE_ROOT
            $root=Join-Path $TestDrive 'wrong-bundle'
            {& $script:Profiler -Scenario chat-tui-help -OutputRoot $root -DryRun} | Should -Throw '*selected build roots*'
            Test-Path $root | Should -BeFalse
        } finally {$env:PCAI_NATIVE_BUNDLE_ROOT=$previous}
    }
}
