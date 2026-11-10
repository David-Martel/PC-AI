#Requires -Version 7.0
<#
.SYNOPSIS
Detect stale OpenSSH descriptor inheritance through real Windows child processes.
.DESCRIPTION
Offline controls inject inert ancestor metadata and preserve unrelated environment,
arguments, memory-only UTF8 stdin, separate output and process custody. Live SSH
requires an explicitly supplied installed client and already pinned peer alias.
#>
[CmdletBinding()]
param(
    [string]$SourcePath = (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Invoke-VigilBoundedProcess.ps1'),
    [string]$LiveSshClientPath,
    [string]$LivePeerAlias,
    [switch]$LiveWindowsPeer
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
if (-not $IsWindows) { Write-Output 'NOT_RUN: Windows native process contract'; return }
. $SourcePath
$pwsh=Join-Path $PSHOME 'pwsh.exe'
$workspace=Join-Path ([IO.Path]::GetTempPath()) ('pcai-ssh-stdio-test-'+[guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($workspace)
$fixture=Join-Path $workspace 'environment fixture.ps1'
$names=@('c28fc6f98a2c44abbbd89d6a3037d0d9_POSIX_FD_STATE','OPENSSH_STDIO_MODE','c28fc6f98a2c44abbbd89d6a3037d0d9_POSIX_CHROOT','PCAI_STDIO_INERT_FIXTURE')
$original=@{};foreach($name in $names){$original[$name]=[Environment]::GetEnvironmentVariable($name)}
$script:checks=0
function Assert-Stdio([bool]$Condition,[string]$Message){if(-not $Condition){throw $Message};$script:checks++}
function Restore-FixtureEnvironment {
    foreach($name in $names){
        # Passing PowerShell $null to a .NET string parameter can create an
        # empty-valued variable. Empty OpenSSH chroot metadata is meaningful.
        if($null -eq $original[$name]){Remove-Item -LiteralPath ('Env:'+$name) -ErrorAction SilentlyContinue}
        else{[Environment]::SetEnvironmentVariable($name,$original[$name])}
    }
}
try {
    # Selector group: FreshChildHandlesDiscardOnlyAncestorStdioMetadata
    # Protects: fresh synchronous handles, independent child environment and jail custody.
    # Detects: inheriting incompatible OpenSSH descriptor/mode state, altering the
    # parent, stripping unrelated Unicode values or dropping a chroot declaration.
    # Needs: Windows + pwsh; all payloads are inert and no SSH connection is made.
    # Breadcrumb: Invoke-VigilBoundedProcess.ps1::CreateChildEnvironment/CreateProcess.
    $body=@'
[Console]::InputEncoding=[Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
[Console]::Write((@{state=[Environment]::GetEnvironmentVariable('c28fc6f98a2c44abbbd89d6a3037d0d9_POSIX_FD_STATE');mode=[Environment]::GetEnvironmentVariable('OPENSSH_STDIO_MODE');chroot=[Environment]::GetEnvironmentVariable('c28fc6f98a2c44abbbd89d6a3037d0d9_POSIX_CHROOT');other=[Environment]::GetEnvironmentVariable('PCAI_STDIO_INERT_FIXTURE');args=@($args);cwd=[Environment]::CurrentDirectory;input=[Console]::In.ReadToEnd()}|ConvertTo-Json -Compress))
[Console]::Error.Write('inert-stderr');exit 9
'@
    [IO.File]::WriteAllText($fixture,$body,[Text.UTF8Encoding]::new($false))
    $injected=@('inert-ancestor-descriptors','nonsock','inert-preserved-jail','åΩ; literal value')
    for($i=0;$i -lt $names.Count;$i++){[Environment]::SetEnvironmentVariable($names[$i],$injected[$i])}
    $arguments=@('space value','a"b','','C:\tail\','éΩ','$literal; & |')
    $payload=('å'*200000)+"`ncomplete"
    $result=Invoke-VigilBoundedProcess -FilePath $pwsh -Arguments (@('-NoLogo','-NoProfile','-File',$fixture)+$arguments) -WorkingDirectory $workspace -InputText $payload -TimeoutSeconds 10
    $actual=$result.Stdout|ConvertFrom-Json
    Assert-Stdio ($null -eq $actual.state -and $null -eq $actual.mode) 'Child inherited stale OpenSSH stdio metadata.'
    Assert-Stdio ($actual.chroot -ceq $injected[2] -and $actual.other -ceq $injected[3]) 'Unrelated or jail environment changed.'
    for($i=0;$i -lt $names.Count;$i++){Assert-Stdio ([Environment]::GetEnvironmentVariable($names[$i]) -ceq $injected[$i]) 'Parent environment changed.'}
    Assert-Stdio ($actual.args.Count -eq $arguments.Count) 'Argument count changed.'
    for($i=0;$i -lt $arguments.Count;$i++){Assert-Stdio ($actual.args[$i] -ceq $arguments[$i]) "Argument $i changed."}
    Assert-Stdio ($actual.cwd -ceq $workspace -and $actual.input -ceq $payload) 'Working directory or UTF8 stdin changed.'
    Assert-Stdio ($result.ExitCode -eq 9 -and $result.Stderr -ceq 'inert-stderr') 'Exit status or separate stderr changed.'
    Write-Output 'PASS: fresh child handles discard only ancestor stdio metadata'

    # Restore before optional authentication; inert jail declarations must never
    # be propagated by this test to a real SSH executable.
    Restore-FixtureEnvironment
    foreach($name in $names){Assert-Stdio ([Environment]::GetEnvironmentVariable($name) -ceq $original[$name]) 'Fixture restoration changed null/empty environment semantics.'}

    # Selector group: ActualPinnedNativeSshCompletesThroughFreshPipes
    # Protects: real installed native SSH completion, not fixture-only claims.
    # Detects: the observed EOF hang; optional Windows echo also detects truncated
    # or recoded stdin, merged diagnostics and discarded remote failure status.
    # Needs: caller-supplied client and existing strict pinned alias; peer pwsh for
    # LiveWindowsPeer. No enrollment, services, routes or keys are changed.
    # Breadcrumb: same public bounded runner; upstream Win32 descriptor import.
    if($LiveSshClientPath -and $LivePeerAlias){
        $sshArguments=@('-n','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=5',$LivePeerAlias,'hostname')
        $result=Invoke-VigilBoundedProcess -FilePath $LiveSshClientPath -Arguments $sshArguments -TimeoutSeconds 10
        $diagnostic=$result.Stderr.Substring(0,[Math]::Min(500,$result.Stderr.Length))
        Assert-Stdio ($result.ExitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($result.Stdout)) "Pinned native SSH command failed: exit=$($result.ExitCode); stderr=$diagnostic"
        if($LiveWindowsPeer){
            $command='[Console]::InputEncoding=[Text.UTF8Encoding]::new($false);[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);[Console]::Write([Console]::In.ReadToEnd());[Console]::Out.Flush();[Console]::Error.Write("inert-error");[Console]::Error.Flush();exit 9'
            $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
            $result=Invoke-VigilBoundedProcess -FilePath $LiveSshClientPath -Arguments @('-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=5',$LivePeerAlias,'pwsh','-NoLogo','-NoProfile','-NonInteractive','-EncodedCommand',$encoded) -InputText $payload -TimeoutSeconds 10
            Assert-Stdio ($result.Stdout -ceq $payload -and $result.Stderr -ceq 'inert-error' -and $result.ExitCode -ne 0) 'Actual native SSH lost inert input, diagnostics or remote failure status.'
        }
        Write-Output 'PASS: actual pinned native SSH completes through fresh pipes'
    } elseif($LiveSshClientPath -or $LivePeerAlias -or $LiveWindowsPeer){throw 'Live checks require both installed client and pinned peer alias.'}
    else{Write-Output 'NOT_RUN: live SSH requires explicit installed client and pinned peer alias'}
    Write-Output "PASS: $script:checks stdio assertions"
} finally {
    Restore-FixtureEnvironment
    if([IO.File]::Exists($fixture)){[IO.File]::Delete($fixture)}
    [IO.Directory]::Delete($workspace,$false)
}
