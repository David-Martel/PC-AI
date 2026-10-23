#Requires -Version 7.0

# One gate and one owned child per module/runspace. Shared callers retain this
# object, including its gate, across restarts.
$existingWorkerState = Get-Variable -Name PcaiPerfWorkerState -Scope Script -ValueOnly -ErrorAction SilentlyContinue
if (-not $existingWorkerState -or -not $existingWorkerState.PSObject.Properties['Gate']) {
    if ($existingWorkerState -and $existingWorkerState.Process) {
        throw 'Legacy worker custody requires a fresh PowerShell session.'
    }
    $script:PcaiPerfWorkerState = [PSCustomObject]@{
        Gate = [System.Threading.SemaphoreSlim]::new(1, 1)
        Process = $null
        OwnedPid = $null
        Input = $null
        Output = $null
        Remaining = ''
        ToolPath = $null
        Signature = $null
        BundleRoot = $null
        ToolSha256 = $null
        Protocol = $null
        LastUse = [datetime]::MinValue
    }
}

# Script invocations have separate script scopes. Root pending exact objects in
# this runspace so a later profiler/module invocation cannot bypass failed closure.
$custodyRunspace=[System.Management.Automation.Runspaces.Runspace]::DefaultRunspace
if (-not $custodyRunspace) { throw 'pcai-perf custody requires an initialized PowerShell runspace.' }
$custodyRunspaceId=$custodyRunspace.InstanceId
$custodyRegistryKey="PC_AI.PcaiPerfPendingCustody.v1:$($custodyRunspaceId.ToString('D'))"
$custodyRegistry=[AppDomain]::CurrentDomain.GetData($custodyRegistryKey)
$localPending=Get-Variable -Name PcaiPerfPendingCustody -Scope Script -ValueOnly -ErrorAction SilentlyContinue
if ($null -ne $localPending -and $localPending -isnot [Collections.Generic.List[object]]) {
    throw 'pcai-perf local pending custody has an unknown type; exact objects were preserved.'
}
if ($null -eq $custodyRegistry) {
    $pending=$localPending
    if ($null -eq $pending) { $pending=[Collections.Generic.List[object]]::new() }
    $custodyRegistry=[pscustomobject]@{Version=1;RunspaceId=$custodyRunspaceId;Pending=$pending;Gate=[object]::new()}
    $custodyRegistry.PSObject.TypeNames.Insert(0,'PC_AI.PcaiPerfPendingCustodyRegistry.v1')
    [AppDomain]::CurrentDomain.SetData($custodyRegistryKey,$custodyRegistry)
} else {
    $validRegistry=$custodyRegistry -is [System.Management.Automation.PSCustomObject] -and
        $custodyRegistry.PSObject.TypeNames[0] -ceq 'PC_AI.PcaiPerfPendingCustodyRegistry.v1'
    foreach($field in @('Version','RunspaceId','Pending','Gate')) {
        if (-not $custodyRegistry.PSObject.Properties[$field] -or $custodyRegistry.PSObject.Properties[$field].MemberType -ne 'NoteProperty') {
            $validRegistry=$false
        }
    }
    if (-not $validRegistry -or $custodyRegistry.Version -isnot [int] -or $custodyRegistry.Version -ne 1 -or
        $custodyRegistry.RunspaceId -isnot [guid] -or $custodyRegistry.RunspaceId -ne $custodyRunspaceId -or
        $custodyRegistry.Pending -isnot [Collections.Generic.List[object]] -or
        $null -eq $custodyRegistry.Gate -or $custodyRegistry.Gate.GetType() -ne [object]) {
        throw 'pcai-perf pending registry type, version or runspace is unknown; rooted custody was preserved.'
    }
    if ($localPending -and $localPending.Count -gt 0 -and -not [object]::ReferenceEquals($localPending,$custodyRegistry.Pending)) {
        throw 'pcai-perf local pending custody differs from the runspace registry; close the original objects before reloading.'
    }
}
$script:PcaiPerfPendingCustody=$custodyRegistry.Pending
$script:PcaiPerfCustodyRegistry=$custodyRegistry
if (-not $script:PcaiPerfWorkerState.PSObject.Properties['OriginRegistry']) {
    $script:PcaiPerfWorkerState | Add-Member -NotePropertyName OriginRegistry -NotePropertyValue $custodyRegistry
}

function Get-PcaiPerfStateRegistry {
    param($State)
    # Preserve legacy raw cleanup-state inputs; internal constructors bind origin
    # before launch. A retained State never changes its captured origin reference.
    if (-not $State.PSObject.Properties['OriginRegistry']) {
        $State | Add-Member -NotePropertyName OriginRegistry -NotePropertyValue $script:PcaiPerfCustodyRegistry
    }
    $registry=$State.OriginRegistry
    Assert-PcaiPerfCustodyRegistry $registry
    return $registry
}

function Assert-PcaiPerfCustodyRegistry {
    param($Registry)
    $registry=$Registry
    $valid=$registry -is [System.Management.Automation.PSCustomObject] -and
        $registry.PSObject.TypeNames[0] -ceq 'PC_AI.PcaiPerfPendingCustodyRegistry.v1'
    foreach($field in @('Version','RunspaceId','Pending','Gate')) {
        if (-not $registry.PSObject.Properties[$field] -or $registry.PSObject.Properties[$field].MemberType -ne 'NoteProperty') { $valid=$false }
    }
    if (-not $valid -or $registry.Version -isnot [int] -or $registry.Version -ne 1 -or
        $registry.RunspaceId -isnot [guid] -or $registry.Pending -isnot [Collections.Generic.List[object]] -or
        $null -eq $registry.Gate -or $registry.Gate.GetType() -ne [object]) {
        throw 'pcai-perf origin registry type or version is unknown; exact custody was preserved.'
    }
    $key="PC_AI.PcaiPerfPendingCustody.v1:$($registry.RunspaceId.ToString('D'))"
    if (-not [object]::ReferenceEquals([AppDomain]::CurrentDomain.GetData($key),$registry)) {
        throw 'pcai-perf rooted origin registry changed; exact custody was preserved.'
    }
}

function Get-PcaiPerfPendingSnapshot {
    param($Registry)
    Assert-PcaiPerfCustodyRegistry $Registry
    if (-not [Threading.Monitor]::TryEnter($Registry.Gate,1000)) { throw [TimeoutException]::new('pcai-perf custody registry is busy; replacement remains blocked.') }
    try { return $Registry.Pending.ToArray() } finally { [Threading.Monitor]::Exit($Registry.Gate) }
}

function Register-PcaiPerfPendingState {
    param($State)
    $registry=Get-PcaiPerfStateRegistry $State
    if (-not [Threading.Monitor]::TryEnter($registry.Gate,1000)) { throw [TimeoutException]::new('pcai-perf origin registry is busy; exact custody was preserved.') }
    try { if (-not $registry.Pending.Contains($State)) { $registry.Pending.Add($State) } } finally { [Threading.Monitor]::Exit($registry.Gate) }
}

function Unregister-PcaiPerfPendingState {
    param($State)
    $registry=Get-PcaiPerfStateRegistry $State
    if (-not [Threading.Monitor]::TryEnter($registry.Gate,1000)) { throw [TimeoutException]::new('pcai-perf origin registry is busy; exact custody was preserved.') }
    try { $null=$registry.Pending.Remove($State) } finally { [Threading.Monitor]::Exit($registry.Gate) }
}

function Get-PcaiPerfRemainingTime {
    param([Diagnostics.Stopwatch]$Deadline, [int]$TimeoutMilliseconds)
    $remaining = $TimeoutMilliseconds - [int]$Deadline.ElapsedMilliseconds
    if ($remaining -le 0) { throw [TimeoutException]::new('pcai-perf operation deadline exceeded.') }
    return $remaining
}

function Wait-PcaiPerfTask {
    param(
        [System.Threading.Tasks.Task]$Task,
        [Diagnostics.Stopwatch]$Deadline,
        [int]$TimeoutMilliseconds,
        [System.Threading.CancellationToken]$CancellationToken
    )
    $CancellationToken.ThrowIfCancellationRequested()
    try {
        if (-not $Task.Wait((Get-PcaiPerfRemainingTime $Deadline $TimeoutMilliseconds), $CancellationToken)) {
            throw [TimeoutException]::new('pcai-perf operation deadline exceeded.')
        }
    } catch [AggregateException] {
        throw $_.Exception.InnerException
    }
}

function Read-PcaiPerfFrame {
    param($State, [Diagnostics.Stopwatch]$Deadline, [int]$TimeoutMilliseconds, [System.Threading.CancellationToken]$CancellationToken)
    $buffer = [System.Text.StringBuilder]::new()
    $remaining = $State.Remaining
    $State.Remaining = ''
    $newline = $remaining.IndexOf([char]10)
    if ($newline -ge 0) {
        $State.Remaining = $remaining.Substring($newline + 1)
        return $remaining.Substring(0,$newline).TrimEnd([char]13)
    }
    $null = $buffer.Append($remaining)
    $chunk = [char[]]::new(4096)
    while ($true) {
        if ($buffer.Length -gt 1048576) { throw [System.IO.InvalidDataException]::new('pcai-perf response exceeds the character limit.') }
        $readCancellation = [Threading.CancellationTokenSource]::CreateLinkedTokenSource($CancellationToken)
        try {
            $readCancellation.CancelAfter((Get-PcaiPerfRemainingTime $Deadline $TimeoutMilliseconds))
            $read = $State.Output.ReadAsync([Memory[char]]::new($chunk),$readCancellation.Token).AsTask()
            try { Wait-PcaiPerfTask $read $Deadline $TimeoutMilliseconds $CancellationToken }
            catch {
                $CancellationToken.ThrowIfCancellationRequested()
                if ($Deadline.ElapsedMilliseconds -ge $TimeoutMilliseconds) { throw [TimeoutException]::new('pcai-perf operation deadline exceeded.') }
                throw
            }
        } finally {
            $readCancellation.Cancel()
            $readCancellation.Dispose()
        }
        if ($read.Result -eq 0) { throw [System.IO.EndOfStreamException]::new('pcai-perf closed before a complete response frame.') }
        $newline = [Array]::IndexOf($chunk,[char]10,0,$read.Result)
        $count = if ($newline -ge 0) { $newline } else { $read.Result }
        if ($buffer.Length + $count -gt 1048576) { throw [IO.InvalidDataException]::new('pcai-perf response exceeds the character limit.') }
        $null = $buffer.Append($chunk,0,$count)
        if ($newline -ge 0) {
            $State.Remaining = [string]::new($chunk,$newline+1,$read.Result-$newline-1)
            $line = $buffer.ToString().TrimEnd([char]13)
            if ([Text.Encoding]::UTF8.GetByteCount($line) -gt 1048576) { throw [IO.InvalidDataException]::new('pcai-perf response exceeds the byte limit.') }
            return $line
        }
    }
}

function Assert-PcaiPerfBundleBinding {
    param([string]$ToolPath)
    $resolved = [IO.Path]::GetFullPath((Get-Item -LiteralPath $ToolPath -ErrorAction Stop).FullName)
    if ($env:PCAI_NATIVE_BUNDLE_ROOT) {
        $root = [IO.Path]::GetFullPath((Get-Item -LiteralPath $env:PCAI_NATIVE_BUNDLE_ROOT -ErrorAction Stop).FullName)
        if (-not [string]::Equals($resolved, (Join-Path $root 'pcai-perf.exe'), [StringComparison]::OrdinalIgnoreCase)) {
            throw [System.IO.InvalidDataException]::new('pcai-perf does not belong to the explicitly selected native bundle.')
        }
    }
    return $resolved
}

function Test-PcaiPerfWorkerHealthy {
    [CmdletBinding()]
    param([string]$ToolPath)
    $state = $script:PcaiPerfWorkerState
    if (($state.PSObject.Properties['ClosurePending'] -and $state.ClosurePending) -or -not $state.Process -or $state.Process.HasExited -or $state.OwnedPid -ne $state.Process.Id) { return $false }
    if ($ToolPath -and -not [string]::Equals($state.ToolPath, $ToolPath, [StringComparison]::OrdinalIgnoreCase)) { return $false }
    return ($state.Input -and $state.Output -and $state.Protocol -eq 1)
}

function Close-PcaiPerfOwnedProcess {
    param($State, [Exception]$OperationException, [switch]$LockHeld, [ValidateRange(1,60000)][int]$TimeoutMilliseconds=1000)
    $acquired=$false
    try {
        $null=Get-PcaiPerfStateRegistry $State
        if (-not $State.PSObject.Properties['Gate']) { $State | Add-Member -NotePropertyName Gate -NotePropertyValue ([Threading.SemaphoreSlim]::new(1,1)) }
        if ($State.Gate -isnot [Threading.SemaphoreSlim]) { throw 'pcai-perf cleanup gate has an unknown type; exact custody was preserved.' }
        if (-not $LockHeld) {
            $acquired=$State.Gate.Wait($TimeoutMilliseconds)
            if (-not $acquired) { throw [TimeoutException]::new('pcai-perf worker is busy; exact custody was preserved.') }
        }
        Close-PcaiPerfOwnedProcessLocked $State -OperationException $OperationException
    }
    catch {
        # Registry/gate failures also keep the original operation and exact state
        # available to callers; they must not replace the failure being cleaned up.
        $cause=if($State.PSObject.Properties['ClosureOperationException'] -and $State.ClosureOperationException){$State.ClosureOperationException}else{$OperationException}
        if($cause -and -not $_.Exception.Data.Contains('OperationException')) {
            $closureFailure=[InvalidOperationException]::new("Operation failed: $($cause.Message) Cleanup: $($_.Exception.Message)",$_.Exception)
            $closureFailure.Data['PcaiProcessCustody']=$State
            $closureFailure.Data['OperationException']=$cause
            throw $closureFailure
        }
        $_.Exception.Data['PcaiProcessCustody']=$State
        if($cause){$_.Exception.Data['OperationException']=$cause}
        throw
    }
    finally { if($acquired){$null=$State.Gate.Release()} }
}

function Close-PcaiPerfOwnedProcessLocked {
    param($State, [Exception]$OperationException)
    # Custody stays with the exact handle until termination is confirmed. Never
    # select descendants, foreign PIDs, or processes by name.
    $resourceNames=@('Input','Output','Stderr','Process')
    if (-not @($resourceNames | Where-Object { $State.PSObject.Properties[$_] -and $State.$_ }).Count) {
        Unregister-PcaiPerfPendingState $State
        if($State.PSObject.Properties['ClosurePending']){$State.ClosurePending=$false}
        if($State.PSObject.Properties['ProcessExitConfirmed']){$State.ProcessExitConfirmed=$false}
        if($State.PSObject.Properties['ConfirmedProcess']){$State.ConfirmedProcess=$null}
        if($State.PSObject.Properties['ClosureOperationException']){$State.ClosureOperationException=$null}
        $State.OwnedPid=$null
        return
    }
    $State | Add-Member -NotePropertyName ClosurePending -NotePropertyValue $true -Force
    if ($OperationException -and (-not $State.PSObject.Properties['ClosureOperationException'] -or -not $State.ClosureOperationException)) {
        $State | Add-Member -NotePropertyName ClosureOperationException -NotePropertyValue $OperationException -Force
    }
    # Internal callers already rooted this exact state before launch. External
    # raw states must be admitted before any process/pipe mutation. A busy
    # registry rejects their cleanup unchanged; it cannot root a caller's earlier
    # resources retroactively. Internal failed states stay rooted even here.
    Register-PcaiPerfPendingState $State
    $operationCause=if($State.PSObject.Properties['ClosureOperationException']){$State.ClosureOperationException}else{$null}
    $cleanupException = $null
    $confirmed = $State.PSObject.Properties['ProcessExitConfirmed'] -and $State.ProcessExitConfirmed
    if ($confirmed -and $State.Process -and -not [object]::ReferenceEquals($State.Process,$State.ConfirmedProcess)) {
        $confirmed=$false
        $cleanupException=[InvalidOperationException]::new('Confirmed process custody changed; foreign handles were preserved.')
    } elseif ($State.Process -and -not $confirmed) {
        try {
            if ($State.OwnedPid -ne $State.Process.Id) {
                throw [InvalidOperationException]::new('Process custody mismatch; foreign handles were preserved.')
            }
            if (-not $State.Process.HasExited) {
                $State.Process.Kill()
                if (-not $State.Process.WaitForExit(1000) -and -not $State.Process.HasExited) {
                    throw [TimeoutException]::new('Owned process termination was not confirmed within 1000 ms.')
                }
            }
            $confirmed = $State.Process.HasExited
        } catch {
            $cleanupException = $_.Exception
            # Kill can race a normal exit. Only a matching handle with confirmed
            # exit permits disposal, even when Kill itself raised an exception.
            try { $confirmed = $State.OwnedPid -eq $State.Process.Id -and $State.Process.HasExited } catch { $confirmed = $false }
        }
    }
    if (-not $confirmed) {
        Register-PcaiPerfPendingState $State
        $detail = if ($cleanupException) { $cleanupException.Message } else { 'Termination could not be confirmed.' }
        $message = "pcai-perf closure failed for owned PID $($State.OwnedPid); exact process and pipe custody retained. Retry Stop-PcaiPerfWorker after resolving closure. $detail"
        if ($operationCause) { $message = "Operation failed: $($operationCause.Message) Cleanup: $message" }
        $closureFailure = [InvalidOperationException]::new($message, $cleanupException)
        $closureFailure.Data['PcaiProcessCustody'] = $State
        if ($operationCause) { $closureFailure.Data['OperationException'] = $operationCause }
        throw $closureFailure
    }
    if (-not $State.PSObject.Properties['ProcessExitConfirmed'] -or -not $State.ProcessExitConfirmed) {
        $State | Add-Member -NotePropertyName ProcessExitConfirmed -NotePropertyValue $true -Force
        $State | Add-Member -NotePropertyName ConfirmedProcess -NotePropertyValue $State.Process -Force
    }
    # A successful close releases only that object. Failed objects remain in the
    # state even if process disposal succeeds, allowing an exact pipe-only retry.
    $disposeFailures=[Collections.Generic.List[Exception]]::new()
    $disposeDetails=[Collections.Generic.List[string]]::new()
    foreach($resourceName in $resourceNames) {
        $property=$State.PSObject.Properties[$resourceName]
        if (-not $property -or -not $property.Value) { continue }
        try { $property.Value.Dispose(); $property.Value=$null }
        catch { $disposeFailures.Add($_.Exception); $disposeDetails.Add("$resourceName Dispose: $($_.Exception.Message)") }
    }
    if ($disposeFailures.Count) {
        Register-PcaiPerfPendingState $State
        $message="pcai-perf closure failed for exited owned PID $($State.OwnedPid); exact failed pipe/handle custody retained. Retry Stop-PcaiPerfWorker. $($disposeDetails -join '; ')"
        if ($operationCause) { $message="Operation failed: $($operationCause.Message) Cleanup: $message" }
        $closureFailure=[InvalidOperationException]::new($message,[AggregateException]::new([Exception[]]$disposeFailures.ToArray()))
        $closureFailure.Data['PcaiProcessCustody']=$State
        $closureFailure.Data['PcaiProcessExitConfirmed']=$true
        if ($operationCause) { $closureFailure.Data['OperationException']=$operationCause }
        throw $closureFailure
    }
    Unregister-PcaiPerfPendingState $State
    $State.OwnedPid = $null
    $State.Protocol = $null
    $State.Remaining = ''
    $State.ClosurePending = $false
    $State.ProcessExitConfirmed=$false
    $State.ConfirmedProcess=$null
    if ($State.PSObject.Properties['ClosureOperationException']) { $State.ClosureOperationException=$null }
}

function Test-PcaiPerfFatalTransportException {
    <#
    .SYNOPSIS
        Identifies transport failures that prohibit replacement work.
    .DESCRIPTION
        Follows wrapped operation and cleanup causes without replacing the
        original exception or retained process and resource objects.
    .PARAMETER Exception
        Exception received from the CLI or worker transport.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param([Parameter(Mandatory)][Exception]$Exception)

    $pending = [Collections.Generic.Stack[Exception]]::new()
    $visited = [Collections.Generic.HashSet[Exception]]::new()
    $pending.Push($Exception)
    while ($pending.Count -gt 0) {
        $current = $pending.Pop()
        if (-not $visited.Add($current)) { continue }
        if ($current -is [TimeoutException] -or $current -is [OperationCanceledException] -or
            $current -is [IO.InvalidDataException] -or $current -is [IO.EndOfStreamException] -or
            $current.GetType().FullName -ceq 'Newtonsoft.Json.JsonReaderException' -or
            $current.Data.Contains('PcaiProcessCustody')) { return $true }
        if ($current.InnerException) { $pending.Push($current.InnerException) }
        if ($current.Data['OperationException'] -is [Exception]) {
            $pending.Push($current.Data['OperationException'])
        }
        if ($current -is [AggregateException]) {
            foreach ($inner in $current.InnerExceptions) { $pending.Push($inner) }
        }
    }
    return $false
}

function Assert-PcaiPerfNoPendingCustody {
    $origin=Get-PcaiPerfStateRegistry $script:PcaiPerfWorkerState
    $retained=@(Get-PcaiPerfPendingSnapshot $script:PcaiPerfCustodyRegistry)
    if(-not [object]::ReferenceEquals($origin,$script:PcaiPerfCustodyRegistry)){$retained+=@(Get-PcaiPerfPendingSnapshot $origin)}
    if (@($retained|Where-Object {$_.PSObject.Properties['ClosurePending'] -and $_.ClosurePending}).Count -gt 0) {
        throw [InvalidOperationException]::new('pcai-perf has unconfirmed process or resource closure; retained custody blocks replacement. Retry Stop-PcaiPerfWorker first.')
    }
}
function Stop-PcaiPerfWorker {
    [CmdletBinding()]
    param([ValidateRange(1, 60000)][int]$TimeoutMilliseconds = 10000)
    $state = $script:PcaiPerfWorkerState
    $deadline=[Diagnostics.Stopwatch]::StartNew()
    $origin=Get-PcaiPerfStateRegistry $state
    # Release registry monitors before acquiring any exact worker gate.
    $pendingStates=@(Get-PcaiPerfPendingSnapshot $script:PcaiPerfCustodyRegistry)
    if (-not [object]::ReferenceEquals($origin,$script:PcaiPerfCustodyRegistry)) { $pendingStates+=@(Get-PcaiPerfPendingSnapshot $origin) }
    foreach($pending in $pendingStates|Where-Object {$_.PSObject.Properties['ClosurePending'] -and $_.ClosurePending}){Close-PcaiPerfOwnedProcess $pending -TimeoutMilliseconds (Get-PcaiPerfRemainingTime $deadline $TimeoutMilliseconds)}
    Close-PcaiPerfOwnedProcess $state -TimeoutMilliseconds (Get-PcaiPerfRemainingTime $deadline $TimeoutMilliseconds)
}

function Invoke-PcaiPerfFrame {
    param($State, [hashtable]$Request, [Diagnostics.Stopwatch]$Deadline, [int]$TimeoutMilliseconds, [System.Threading.CancellationToken]$CancellationToken, [switch]$Negotiate, [string]$SerializedRequest)
    $json = if ($SerializedRequest) { $SerializedRequest } else { $Request | ConvertTo-Json -Compress -Depth 20 -WarningAction Stop }
    if ([Text.Encoding]::UTF8.GetByteCount($json) -gt 1048576) {
        throw [System.IO.InvalidDataException]::new('pcai-perf request exceeds the byte limit.')
    }
    Wait-PcaiPerfTask ($State.Input.WriteLineAsync($json)) $Deadline $TimeoutMilliseconds $CancellationToken
    Wait-PcaiPerfTask ($State.Input.FlushAsync()) $Deadline $TimeoutMilliseconds $CancellationToken
    $line = Read-PcaiPerfFrame $State $Deadline $TimeoutMilliseconds $CancellationToken
    $response = ConvertFrom-Json -InputObject $line -Depth 20 -ErrorAction Stop
    if ($Negotiate -and (-not $response.PSObject.Properties['protocol'] -or $response.protocol -ne 1 -or -not $response.PSObject.Properties['request_id'])) {
        throw [NotSupportedException]::new('pcai-perf worker does not support correlated protocol1; use bounded direct CLI or rebuild it.')
    }
    if (-not $response.PSObject.Properties['request_id'] -or $response.request_id -cne $Request.request_id) {
        throw [System.IO.InvalidDataException]::new('pcai-perf response request ID mismatch.')
    }
    if (-not $response.PSObject.Properties['protocol'] -or $response.protocol -ne 1) {
        throw [IO.InvalidDataException]::new('pcai-perf response protocol changed during the session.')
    }
    if (-not $response.PSObject.Properties['ok'] -or $response.ok -isnot [bool]) {
        throw [System.IO.InvalidDataException]::new('pcai-perf response is missing a boolean status.')
    }
    if (-not $response.ok) { throw [InvalidOperationException]::new([string]$response.error) }
    if (-not $response.PSObject.Properties['result']) { throw [IO.InvalidDataException]::new('pcai-perf success response is missing its result.') }
    return $response.result
}

function Start-PcaiPerfWorker {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$ToolPath,
        [string[]]$WorkerArguments = @('worker'),
        [ValidateRange(1, 60000)][int]$TimeoutMilliseconds = 10000,
        [System.Threading.CancellationToken]$CancellationToken = [System.Threading.CancellationToken]::None,
        [Diagnostics.Stopwatch]$Deadline = [Diagnostics.Stopwatch]::StartNew(),
        [switch]$LockHeld
    )
    $state = $script:PcaiPerfWorkerState
    $acquired = $false
    $admitted = $false
    if (-not $LockHeld) {
        $acquired = $state.Gate.Wait((Get-PcaiPerfRemainingTime $Deadline $TimeoutMilliseconds), $CancellationToken)
        if (-not $acquired) { throw [TimeoutException]::new('pcai-perf worker is busy.') }
    }
    try {
        Assert-PcaiPerfNoPendingCustody
        $resolved = Assert-PcaiPerfBundleBinding $ToolPath
        $file = Get-Item -LiteralPath $resolved -ErrorAction Stop
        $signature = "$resolved|$($file.Length)|$($file.LastWriteTimeUtc.Ticks)|$($WorkerArguments -join [char]0)"
        if ((Test-PcaiPerfWorkerHealthy -ToolPath $resolved) -and $state.Signature -ceq $signature -and $state.BundleRoot -ceq $env:PCAI_NATIVE_BUNDLE_ROOT) {
            return $state
        }
        Close-PcaiPerfOwnedProcess $state -LockHeld
        Register-PcaiPerfPendingState $state
        $admitted=$true
        $startInfo = [Diagnostics.ProcessStartInfo]::new()
        $startInfo.FileName = $resolved
        foreach ($argument in $WorkerArguments) { $startInfo.ArgumentList.Add($argument) }
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardInput = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.StandardInputEncoding = [Text.UTF8Encoding]::new($false)
        $startInfo.StandardOutputEncoding = [Text.UTF8Encoding]::new($false, $true)
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $startInfo
        try {
            $toolSha256 = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash
            $CancellationToken.ThrowIfCancellationRequested()
            $null = Get-PcaiPerfRemainingTime $Deadline $TimeoutMilliseconds
            if (-not $process.Start()) { throw 'pcai-perf worker failed to start.' }
        } catch {
            $process.Dispose()
            Unregister-PcaiPerfPendingState $state
            $admitted=$false
            throw
        }
        $state.Process = $process
        $state.OwnedPid = $process.Id
        $state.Input = $process.StandardInput
        $state.Output = $process.StandardOutput
        $state.ToolPath = $resolved
        $state.Signature = $signature
        $state.BundleRoot = $env:PCAI_NATIVE_BUNDLE_ROOT
        $state.ToolSha256 = $toolSha256
        $state.Protocol = $null
        $state.Remaining = ''
        try {
            if ((Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash -ne $toolSha256) {
                throw [IO.InvalidDataException]::new('pcai-perf executable changed during startup.')
            }
            $hello = Invoke-PcaiPerfFrame $state @{command='hello';request_id=[guid]::NewGuid().ToString('N');protocol=1} $Deadline $TimeoutMilliseconds $CancellationToken -Negotiate
            if ($hello.protocol -ne 1) { throw [NotSupportedException]::new('pcai-perf worker protocol negotiation failed.') }
            $state.Protocol = 1
            $state.LastUse = Get-Date
            return $state
        } catch {
            Close-PcaiPerfOwnedProcess $state -OperationException $_.Exception -LockHeld
            throw
        }
    } catch {
        if($admitted -and -not $state.Process){Unregister-PcaiPerfPendingState $state}
        throw
    } finally {
        if ($acquired) { $null = $state.Gate.Release() }
    }
}

function Invoke-PcaiPerfWorkerRequest {
    <#
    .SYNOPSIS
        Sends one correlated, bounded request to the owned pcai-perf worker.
    .DESCRIPTION
        One deadline covers serialization, lock acquisition, negotiation, writes
        and the response. Synchronous startup/filesystem calls are checked at
        their boundaries; the deadline does not preempt those operating-system calls.
        Cancellation or a corrupt/late frame closes only this helper's child.
        Unconfirmed termination retains exact handle and pipe custody and blocks
        replacement until Stop-PcaiPerfWorker confirms closure.
        Legacy workers are explicitly rejected, allowing bounded direct CLI fallback.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$ToolPath,
        [Parameter(Mandatory)][string]$Command,
        [hashtable]$Payload = @{},
        [ValidateRange(1, 60000)][int]$TimeoutMilliseconds = 10000,
        [System.Threading.CancellationToken]$CancellationToken = [System.Threading.CancellationToken]::None,
        [string[]]$WorkerArguments = @('worker')
    )
    $deadline = [Diagnostics.Stopwatch]::StartNew()
    $CancellationToken.ThrowIfCancellationRequested()
    foreach ($key in @('command', 'request_id', 'protocol')) {
        if ($Payload.ContainsKey($key)) { throw [ArgumentException]::new("Reserved worker payload key: $key") }
    }
    $request = @{command=$Command;request_id=[guid]::NewGuid().ToString('N');protocol=1}
    foreach ($entry in $Payload.GetEnumerator()) { $request[$entry.Key] = $entry.Value }
    $serializedRequest = $request | ConvertTo-Json -Compress -Depth 20 -WarningAction Stop
    if ([Text.Encoding]::UTF8.GetByteCount($serializedRequest) -gt 1048576) {
        throw [System.IO.InvalidDataException]::new('pcai-perf request exceeds the byte limit.')
    }
    $CancellationToken.ThrowIfCancellationRequested()
    $state = $script:PcaiPerfWorkerState
    $acquired = $state.Gate.Wait((Get-PcaiPerfRemainingTime $deadline $TimeoutMilliseconds), $CancellationToken)
    if (-not $acquired) { throw [TimeoutException]::new('pcai-perf worker is busy.') }
    try {
        $null = Start-PcaiPerfWorker -ToolPath $ToolPath -WorkerArguments $WorkerArguments -TimeoutMilliseconds $TimeoutMilliseconds -CancellationToken $CancellationToken -Deadline $deadline -LockHeld
        $result = Invoke-PcaiPerfFrame $state $request $deadline $TimeoutMilliseconds $CancellationToken -SerializedRequest $serializedRequest
        $state.LastUse = Get-Date
        return $result
    } catch {
        Close-PcaiPerfOwnedProcess $state -OperationException $_.Exception -LockHeld
        throw
    } finally {
        $null = $state.Gate.Release()
    }
}

function Invoke-PcaiPerfCliCommand {
    <#
    .SYNOPSIS
        Runs a bounded direct CLI command, retaining legacy JSON compatibility.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$ToolPath,
        [Parameter(Mandatory)][string[]]$Arguments,
        [ValidateRange(1, 60000)][int]$TimeoutMilliseconds = 10000,
        [System.Threading.CancellationToken]$CancellationToken = [System.Threading.CancellationToken]::None
    )
    $CancellationToken.ThrowIfCancellationRequested()
    Assert-PcaiPerfNoPendingCustody
    $resolved = Assert-PcaiPerfBundleBinding $ToolPath
    $deadline = [Diagnostics.Stopwatch]::StartNew()
    $state = [PSCustomObject]@{ Process=$null;OwnedPid=$null;Input=$null;Output=$null;Protocol=$null;Remaining='';OriginRegistry=$script:PcaiPerfCustodyRegistry;Gate=[Threading.SemaphoreSlim]::new(1,1) }
    $process=$null
    $admitted=$false
    $operationException = $null
    try {
        Register-PcaiPerfPendingState $state
        $admitted=$true
        $info = [Diagnostics.ProcessStartInfo]::new()
        $info.FileName = $resolved
        foreach ($argument in $Arguments) { $info.ArgumentList.Add($argument) }
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.StandardOutputEncoding = [Text.UTF8Encoding]::new($false, $true)
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $info
        if (-not $process.Start()) { throw 'pcai-perf CLI failed to start.' }
        $state.Process = $process
        $state.OwnedPid = $process.Id
        $state.Output = $process.StandardOutput
        $line = Read-PcaiPerfFrame $state $deadline $TimeoutMilliseconds $CancellationToken
        Wait-PcaiPerfTask ($process.WaitForExitAsync($CancellationToken)) $deadline $TimeoutMilliseconds $CancellationToken
        $CancellationToken.ThrowIfCancellationRequested()
        if ($process.ExitCode -ne 0) { throw [InvalidOperationException]::new("pcai-perf CLI failed with exit code $($process.ExitCode).") }
        return ConvertFrom-Json -InputObject $line -Depth 20 -ErrorAction Stop
    } catch {
        $operationException = $_.Exception
        throw
    } finally {
        if ($state.Process) { Close-PcaiPerfOwnedProcess $state -OperationException $operationException }
        else { if($process){$process.Dispose()}; if($admitted){Unregister-PcaiPerfPendingState $state} }
    }
}

$workerModule = $ExecutionContext.SessionState.Module
if ($workerModule -and -not $workerModule.OnRemove) {
    $workerModule.OnRemove = { Stop-PcaiPerfWorker }
}
