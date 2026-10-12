#Requires -Version 7.0
<#
.SYNOPSIS
    Measures a bounded startup scenario and retains a hash-bound receipt.
.DESCRIPTION
    Runs a fresh process, captures interval CPU and bounded output digests, and
    closes only the process it owns on timeout. Unconfirmed termination or failed
    pipe/handle disposal retains exact custody in the shared helper and error;
    a replacement in the same runspace is blocked until resolved, including later
    script invocations. Stderr uses that same cleanup path. Runspace custody uses
    a versioned process-local store and needs no profile or native assembly.
    Internal states are rooted before child launch. Active states do not block
    unrelated launches; failed closure blocks replacement until exact retry.
    Failures expose PcaiOwnedPid and PcaiOwnedProcessExited exception metadata
    when a child was started, even if its script never reached a PID-file write.
    Output text and authentication
    settings are not persisted. This is startup profiling, not workload throughput
    or an EventPipe trace. Repetitions use profile-rN names with dates in metadata.
    An explicit native bundle never falls back to an unrelated executable.
    DryRun and WhatIf return the plan without starting a process or writing files.
.EXAMPLE
    ./Invoke-PcaiProfiling.ps1 -Scenario acceleration-import -DryRun
.EXAMPLE
    ./Invoke-PcaiProfiling.ps1 -Scenario chat-tui-help -PassThru
#>
[CmdletBinding(SupportsShouldProcess,PositionalBinding=$false)]
param(
    [ValidateSet('acceleration-import','native-init','chat-tui-help','service-host-provider-show','custom')]
    [string]$Scenario = 'acceleration-import',
    [string]$OutputRoot,
    [string]$CommandPath,
    [string[]]$ArgumentList = @(),
    [ValidateRange(1,60000)][int]$TimeoutMilliseconds = 10000,
    [System.Threading.CancellationToken]$CancellationToken = [System.Threading.CancellationToken]::None,
    [switch]$DryRun,
    [Alias('h')][switch]$Help,
    [switch]$PassThru,
    [Parameter(ValueFromRemainingArguments)][string[]]$CliArgs
)
if ($Help -or @($CliArgs) -contains '--help') { Get-Help $PSCommandPath -Detailed; return }
if ($CliArgs) { throw 'Unknown profiling arguments.' }
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))

function Resolve-PcaiStartupCommand {
    param([string]$Name, [string]$Artifact)
    $roots = if ($env:PCAI_NATIVE_BUNDLE_ROOT) {
        @((Resolve-Path -LiteralPath $env:PCAI_NATIVE_BUNDLE_ROOT -ErrorAction Stop).Path)
    } else {
        @((Join-Path $repoRoot ".pcai/build/artifacts/$Artifact"),
          (Join-Path $repoRoot "Native/$Name/bin/Release/net8.0/win-x64/publish"),
          (Join-Path $repoRoot "Native/$Name/bin/Release/net8.0/win-x64"),
          (Join-Path $repoRoot "Native/$Name/bin/Release/net8.0"))
    }
    foreach ($root in $roots) {
        $exe = Join-Path $root "$Name.exe"
        if (Test-Path -LiteralPath $exe -PathType Leaf) { return @{Path=$exe;Arguments=@()} }
        $dll = Join-Path $root "$Name.dll"
        if (Test-Path -LiteralPath $dll -PathType Leaf) {
            return @{Path=(Get-Command dotnet -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source;Arguments=@($dll);Artifact=$dll}
        }
    }
    throw "$Name was not found in the selected build roots. Build it or select its exact bundle."
}

function Invoke-PcaiStartupProcess {
    param([string]$Path, [string[]]$Arguments)
    $CancellationToken.ThrowIfCancellationRequested()
    Assert-PcaiPerfNoPendingCustody
    $custody = [pscustomobject]@{Process=$null;OwnedPid=$null;Input=$null;Output=$null;Protocol=$null;Remaining='';OriginRegistry=$script:PcaiPerfCustodyRegistry;Gate=[Threading.SemaphoreSlim]::new(1,1)}
    $process=$null
    $admitted=$false
    $ownedPid = $null
    $observer = [Diagnostics.Process]::GetCurrentProcess()
    $observerCpu = $observer.TotalProcessorTime.TotalMilliseconds
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $operationException = $null
    try {
        Register-PcaiPerfPendingState $custody
        $admitted=$true
        $info = [Diagnostics.ProcessStartInfo]::new()
        $info.FileName = $Path
        foreach ($argument in $Arguments) { $info.ArgumentList.Add($argument) }
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $info.StandardOutputEncoding = [Text.UTF8Encoding]::new($false,$true)
        $info.StandardErrorEncoding = [Text.UTF8Encoding]::new($false,$true)
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $info
        if (-not $process.Start()) { throw 'Startup process did not start.' }
        $ownedPid = $process.Id
        $custody.Process = $process
        $custody.OwnedPid = $ownedPid
        $custody.Output = $process.StandardOutput
        $custody | Add-Member -NotePropertyName Stderr -NotePropertyValue $process.StandardError
        $readers = @($process.StandardOutput,$process.StandardError)
        $buffers = @([char[]]::new(4096),[char[]]::new(4096))
        $texts = @([Text.StringBuilder]::new(),[Text.StringBuilder]::new())
        $tasks = @($readers[0].ReadAsync($buffers[0],0,4096),$readers[1].ReadAsync($buffers[1],0,4096))
        while ($tasks[0] -or $tasks[1] -or -not $process.HasExited) {
            $CancellationToken.ThrowIfCancellationRequested()
            $remaining = $TimeoutMilliseconds - [int]$timer.ElapsedMilliseconds
            if ($remaining -le 0) { throw [TimeoutException]::new('Startup profiling deadline exceeded.') }
            $pending = @($tasks | Where-Object { $_ })
            if ($pending.Count) {
                $null = [Threading.Tasks.Task]::WhenAny([Threading.Tasks.Task[]]$pending).Wait([Math]::Min($remaining,20),$CancellationToken)
            } else {
                $null = $process.WaitForExit([Math]::Min($remaining,20))
            }
            for ($index=0;$index -lt 2;$index++) {
                if ($tasks[$index] -and $tasks[$index].IsCompleted) {
                    $count = $tasks[$index].GetAwaiter().GetResult()
                    if ($count -eq 0) { $tasks[$index]=$null; continue }
                    if ($texts[$index].Length + $count -gt 1048576) {
                        throw [IO.InvalidDataException]::new('Startup output exceeds the capture limit.')
                    }
                    $null = $texts[$index].Append($buffers[$index],0,$count)
                    $tasks[$index] = $readers[$index].ReadAsync($buffers[$index],0,4096)
                }
            }
        }
        $timer.Stop()
        $process.Refresh()
        $observer.Refresh()
        if ($process.ExitCode -ne 0) { throw "Startup process failed with exit code $($process.ExitCode)." }
        $digests = foreach ($text in $texts) {
            $bytes = [Text.Encoding]::UTF8.GetBytes($text.ToString())
            @{Bytes=$bytes.Length;Sha256=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes))}
        }
        return [pscustomobject]@{
            OwnedPid=$ownedPid;ExitCode=$process.ExitCode;ElapsedMs=$timer.Elapsed.TotalMilliseconds
            ChildCpuMs=$process.TotalProcessorTime.TotalMilliseconds
            ObserverCpuMs=($observer.TotalProcessorTime.TotalMilliseconds-$observerCpu)
            ChildCpuFraction=($process.TotalProcessorTime.TotalMilliseconds/[Math]::Max(1,$timer.Elapsed.TotalMilliseconds)/[Environment]::ProcessorCount)
            Stdout=$digests[0];Stderr=$digests[1]
        }
    } catch {
        $operationException = $_.Exception
        if ($null -ne $ownedPid) {
            $operationException.Data['PcaiOwnedPid'] = $ownedPid
            $operationException.Data['PcaiOwnedProcessExited'] = $false
        }
        throw
    } finally {
        try {
            if ($custody.Process) {
                Close-PcaiPerfOwnedProcess $custody -OperationException $operationException
                if ($operationException -and $null -ne $ownedPid) {
                    $operationException.Data['PcaiOwnedProcessExited'] = $true
                }
            } else { if($process){$process.Dispose()}; if($admitted){Unregister-PcaiPerfPendingState $custody} }
        } finally { $observer.Dispose() }
    }
}

$spec = switch ($Scenario) {
    { $_ -in 'acceleration-import','native-init' } {
        $modulePath = (Join-Path $repoRoot 'Modules/PC-AI.Acceleration/PC-AI.Acceleration.psd1').Replace("'","''")
        $code = "`$ErrorActionPreference='Stop'; `$module=Import-Module '$modulePath' -Force -PassThru;"
        if ($Scenario -eq 'native-init') { $code += ' if(-not (& $module {Initialize-PcaiNative -Force})){throw "Native initialization failed"};' }
        @{Path=(Get-Command pwsh -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
          Arguments=@('-NoLogo','-NoProfile','-EncodedCommand',[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($code)))}
    }
    'chat-tui-help' { $selected=Resolve-PcaiStartupCommand PcaiChatTui pcai-chattui; $selected.Arguments+=@('--help'); $selected }
    'service-host-provider-show' { $selected=Resolve-PcaiStartupCommand PcaiServiceHost pcai-servicehost; $selected.Arguments+=@('provider','show'); $selected }
    'custom' {
        if (-not $CommandPath) { throw 'The custom scenario requires CommandPath.' }
        @{Path=$CommandPath;Arguments=$ArgumentList}
    }
}
$spec.Path = (Resolve-Path -LiteralPath $spec.Path -ErrorAction Stop).Path
if (-not $OutputRoot) { $OutputRoot=Join-Path $repoRoot '.pcai/profiling' }
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot)
$plan=[pscustomobject]@{Scenario=$Scenario;CommandPath=$spec.Path;OutputRoot=$OutputRoot;TimeoutMilliseconds=$TimeoutMilliseconds;DryRun=[bool]$DryRun}
if ($DryRun -or -not $PSCmdlet.ShouldProcess($OutputRoot,'Run bounded startup profiling and preserve receipt')) { $plan; return }
. (Join-Path $repoRoot 'Modules/PC-AI.Acceleration/Private/Invoke-PcaiPerfWorker.ps1')
Assert-PcaiPerfNoPendingCustody
. (Join-Path $repoRoot 'Tools/PcaiArtifactDirectories.ps1')
$revision=New-PcaiArtifactDirectory -Root $OutputRoot -Name 'profile'
$measurement=Invoke-PcaiStartupProcess $spec.Path $spec.Arguments
$receipt=[ordered]@{
    SchemaVersion=1;ObservedAt=[DateTimeOffset]::UtcNow.ToString('o');Scenario=$Scenario;Profiler='ProcessMetrics'
    Host=$env:COMPUTERNAME;LogicalProcessors=[Environment]::ProcessorCount;PowerShell=$PSVersionTable.PSVersion.ToString()
    CommandPath=$spec.Path;CommandSha256=(Get-FileHash -LiteralPath $spec.Path -Algorithm SHA256).Hash
    ScriptSha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash
    ArtifactSha256=if($spec.ContainsKey('Artifact')){(Get-FileHash -LiteralPath $spec.Artifact -Algorithm SHA256).Hash}else{$null}
    NativeBundleRoot=$env:PCAI_NATIVE_BUNDLE_ROOT;Measurement=$measurement
    Boundary='One startup call; observer CPU is included separately. Output text and command arguments are not persisted.'
}
$receiptPath=Join-Path $revision 'startup-profile.json'
$receipt|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $receiptPath -Encoding utf8NoBOM
$result=[pscustomobject]@{Scenario=$Scenario;ReceiptPath=$receiptPath;Measurement=$measurement;Profiler='ProcessMetrics'}
if ($PassThru) { $result } else { $result|Format-List }
