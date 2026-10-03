BeforeAll {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
    . (Join-Path $repoRoot 'Tools/SystemScripts/Networking/Invoke-VigilBoundedProcess.ps1')
    $pwshPath = (Get-Command pwsh -CommandType Application | Select-Object -First 1).Source
}
Describe 'Shared Windows owned process runner' {
    It 'preserves exact quoted arguments, UTF8, empty values and the requested working directory' {
        $argumentScript = Join-Path $TestDrive 'argument capture.ps1'
        '[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); [Console]::Write((@{arguments=@($args); cwd=[Environment]::CurrentDirectory} | ConvertTo-Json -Compress))' | Set-Content -LiteralPath $argumentScript
        $expected = @('space value', 'a"b', '', 'C:\trailing\', 'éΩ')
        $result = Invoke-VigilBoundedProcess -FilePath $pwshPath -Arguments (@('-NoProfile', '-File', $argumentScript) + $expected) -WorkingDirectory $TestDrive -TimeoutSeconds 5
        $result.ExitCode | Should -Be 0
        $result.Stderr | Should -BeNullOrEmpty
        $actual = $result.Stdout | ConvertFrom-Json
        $actual.arguments.Count | Should -Be 5
        for ($i = 0; $i -lt $expected.Count; $i++) { $actual.arguments[$i] | Should -BeExactly $expected[$i] }
        $actual.cwd | Should -BeExactly $TestDrive
    }
    It 'streams large UTF8 stdin to EOF without truncation or a BOM and captures both outputs' {
        $inputValue = ('å' * 1000000) + "`ncomplete"
        $command = '[Console]::Write([Console]::In.ReadToEnd()); [Console]::Error.Write("error-receipt"); exit 9'
        $result = Invoke-VigilBoundedProcess -FilePath $pwshPath -Arguments @('-NoProfile', '-Command', $command) -InputText $inputValue -TimeoutSeconds 8
        $result.ExitCode | Should -Be 9
        $result.Stdout | Should -BeExactly $inputValue
        $result.Stderr | Should -BeExactly 'error-receipt'
    }
    It 'gives absent stdin immediate EOF instead of inheriting an interactive console' {
        $result = Invoke-VigilBoundedProcess -FilePath $pwshPath -Arguments @('-NoProfile', '-Command', '[Console]::Write("<"+[Console]::In.ReadToEnd()+">")') -TimeoutSeconds 5
        $result.ExitCode | Should -Be 0
        $result.Stdout | Should -BeExactly '<>'
    }
    It 'bounds large stdin to a child that never reads and terminates the owned child' {
        $pidFile = Join-Path $TestDrive 'no-read.pid'
        $command = "[IO.File]::WriteAllText('$($pidFile.Replace("'", "''"))', [string]`$PID); Start-Sleep -Seconds 30"
        $clock = [Diagnostics.Stopwatch]::StartNew()
        { Invoke-VigilBoundedProcess -FilePath $pwshPath -Arguments @('-NoProfile', '-Command', $command) -InputText ('x' * 8000000) -TimeoutSeconds 2 } | Should -Throw '*timed out*process tree terminated*'
        $clock.Elapsed.TotalSeconds | Should -BeLessThan 4
        Test-Path -LiteralPath $pidFile | Should -BeTrue
        Get-Process -Id ([int](Get-Content -LiteralPath $pidFile -Raw)) -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
    }
    It 'kills pipe-inheriting orphan descendants after a successful parent exit' {
        $pidFile = Join-Path $TestDrive 'orphan.pid'
        $command = @"
`$start = [Diagnostics.ProcessStartInfo]::new()
`$start.FileName = '$($pwshPath.Replace("'", "''"))'
`$start.UseShellExecute = `$false
`$start.CreateNoWindow = `$true
foreach (`$value in @('-NoProfile', '-Command', 'Start-Sleep -Seconds 30')) { `$start.ArgumentList.Add(`$value) }
`$child = [Diagnostics.Process]::Start(`$start)
[IO.File]::WriteAllText('$($pidFile.Replace("'", "''"))', [string]`$child.Id)
[Console]::Write('finished-parent')
exit 0
"@
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $result = Invoke-VigilBoundedProcess -FilePath $pwshPath -Arguments @('-NoProfile', '-Command', $command) -TimeoutSeconds 3
        $clock.Elapsed.TotalSeconds | Should -BeLessThan 3
        $result.ExitCode | Should -Be 0
        $result.Stdout | Should -BeExactly 'finished-parent'
        Get-Process -Id ([int](Get-Content -LiteralPath $pidFile -Raw)) -ErrorAction SilentlyContinue | Should -BeNullOrEmpty
    }
    It 'retains launch failure and rejects invalid deadline before launching' {
        { Invoke-VigilBoundedProcess -FilePath (Join-Path $TestDrive 'absent.exe') -TimeoutSeconds 1 } | Should -Throw '*CreateProcess suspended*'
        { Invoke-VigilBoundedProcess -FilePath $pwshPath -TimeoutSeconds 0 } | Should -Throw
        { Invoke-VigilBoundedProcess -FilePath $pwshPath -TimeoutSeconds 3601 } | Should -Throw
        { Invoke-VigilBoundedProcess -FilePath $pwshPath -WorkingDirectory (Join-Path $TestDrive 'absent-dir') } | Should -Throw '*CreateProcess suspended*'
    }
}
