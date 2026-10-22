#Requires -Version 7.0
# Synthetic data and private child processes only; no live credential provider is invoked.
Set-StrictMode -Version Latest
Describe 'Isolated Windows credential custody fixtures' -Tag 'Unit', 'Windows' {
BeforeAll {
    . (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretBackendUtilities.ps1')
    if (-not $IsWindows) { throw 'These custody fixtures require Windows NTFS.' }
    function New-CustodySyntheticSecureString {
        param([string]$Text)
        $value=[Security.SecureString]::new()
        foreach($character in $Text.ToCharArray()){$value.AppendChar($character)}
        $value.MakeReadOnly()
        return $value
    }
    $script:SavedFixtureEnvironment = @{}
    $script:SavedFixtureEnvironmentPresence = @{}
    $environmentNames = @('USERPROFILE','TEMP','TMP','APPDATA','LOCALAPPDATA','PATH',
        'BITWARDENCLI_APPDATA_DIR','BW_SESSION','BW_CLIENTID','BW_CLIENTSECRET','BWS_ACCESS_TOKEN',
        'PCAI_BW_EXECUTABLE','PCAI_BW_EXECUTABLE_SHA256','PCAI_CREDENTIAL_ROOT',
        'PCAI_BACKEND_FIXTURE_VAR','PCAI_CONSUMER_SYNTHETIC')
    foreach ($name in $environmentNames) {
        $script:SavedFixtureEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        $script:SavedFixtureEnvironmentPresence[$name] = [Environment]::GetEnvironmentVariables('Process').Keys -icontains $name
    }
    # Pester creates TestDrive before BeforeAll. Its ordinary TEMP ancestry can
    # be unsafe for credential publication, so use a fresh protected namespace.
    # No production credential file is opened and no existing ACL is changed.
    $realProfile = [Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)
    Assert-CredentialTrustedNamespace $realProfile
    $script:FixtureDrive = Join-Path $realProfile ('.pcai-custody-fixture-' + [guid]::NewGuid().ToString('N'))
    if (Test-Path -LiteralPath $script:FixtureDrive) { throw 'Fresh fixture root required.' }
    Assert-CredentialPrivateDirectory $script:FixtureDrive
    Assert-CredentialTrustedNamespace $script:FixtureDrive
    $script:FixtureProfileRoot=$realProfile
    Initialize-CredentialFileInterop
    Initialize-CredentialCleanupInterop
    $script:FixtureRootPin=[Pcai.Credential.CleanupV1]::PinDirectory($script:FixtureDrive)
    $script:FixtureRootIdentity=[Pcai.Credential.FileV1]::Identity($script:FixtureRootPin)
    $env:USERPROFILE = Join-Path $script:FixtureDrive 'user'
    $env:TEMP = Join-Path $script:FixtureDrive 'temp'
    $env:TMP = $env:TEMP
    $env:APPDATA = Join-Path $env:USERPROFILE 'AppData/Roaming'
    $env:LOCALAPPDATA = Join-Path $env:USERPROFILE 'AppData/Local'
    $env:BITWARDENCLI_APPDATA_DIR = Join-Path $script:FixtureDrive 'bw-cli-state'
    foreach ($path in @($env:USERPROFILE,$env:TEMP,(Join-Path $env:USERPROFILE 'AppData'),$env:APPDATA,$env:LOCALAPPDATA,$env:BITWARDENCLI_APPDATA_DIR)) {
        Assert-CredentialPrivateDirectory $path
    }
    $powerShellDirectory = Split-Path -Parent (Get-Process -Id $PID).Path
    $env:PATH = $powerShellDirectory + [IO.Path]::PathSeparator + (Join-Path $env:SystemRoot 'System32')
    foreach ($name in $environmentNames | Where-Object { $_ -match '^(BW_|BWS_|PCAI_)' }) {
        [Environment]::SetEnvironmentVariable($name, $null, 'Process')
    }
}
AfterAll {
    Remove-Module SecretsTier -Force -ErrorAction SilentlyContinue
    foreach ($name in $script:SavedFixtureEnvironment.Keys) {
        if ($script:SavedFixtureEnvironmentPresence[$name]) {
            [Environment]::SetEnvironmentVariable($name, $script:SavedFixtureEnvironment[$name], 'Process')
        } else {
            Remove-Item -LiteralPath ('Env:' + $name) -ErrorAction SilentlyContinue
        }
    }
    # This exact fresh process-owned tree contains synthetic data only. Expected
    # failure-control witnesses are redundant after their assertions complete.
    try {
        $root=[IO.Path]::GetFullPath($script:FixtureDrive)
        if([IO.Path]::GetDirectoryName($root) -cne $script:FixtureProfileRoot -or
           [IO.Path]::GetFileName($root) -cnotmatch '^\.pcai-custody-fixture-[a-f0-9]{32}$' -or
           [Pcai.Credential.FileV1]::Identity($script:FixtureRootPin) -cne $script:FixtureRootIdentity){throw 'Synthetic root custody changed.'}
        Assert-CredentialTrustedNamespace $root
        Assert-CredentialPrivateDirectory $root
        if('WorkProfileBackend.PendingProcessCustody' -as [type]){
            if(@([WorkProfileBackend.PendingProcessCustody]::Ids()).Count){throw 'Synthetic process custody remains pending.'}
        }
        $entries=[Collections.Generic.List[IO.FileSystemInfo]]::new()
        $queue=[Collections.Generic.Queue[string]]::new();$queue.Enqueue($root)
        while($queue.Count){
            foreach($entry in Get-ChildItem -LiteralPath $queue.Dequeue() -Force -ErrorAction Stop){
                if(-not $entry.FullName.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or
                   ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'Unqualified synthetic descendant retained.'}
                $entries.Add($entry)
                if($entry.PSIsContainer){$queue.Enqueue($entry.FullName)}
            }
        }
        foreach($entry in @($entries|Sort-Object {$_.FullName.Length} -Descending)){
            if($entry.PSIsContainer){[IO.Directory]::Delete($entry.FullName,$false)}
            else{[IO.File]::SetAttributes($entry.FullName,[IO.FileAttributes]::Normal);[IO.File]::Delete($entry.FullName)}
        }
        [Pcai.Credential.CleanupV1]::MarkDelete($script:FixtureRootPin)
    } catch {throw 'Synthetic fixture cleanup was not confirmed; the exact owned root was retained for inspection.'}
    finally {if($script:FixtureRootPin){$script:FixtureRootPin.Dispose()}}
    if(Test-Path -LiteralPath $script:FixtureDrive){throw 'Synthetic fixture root cleanup postcondition failed.'}
}

Describe 'Owned process custody and passive configuration' {
# Synthetic fixtures only. Real child processes exercise custody; no credential provider or authentication is invoked.
BeforeAll {
    . (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretBackendUtilities.ps1')
    $script:pwshPath = (Get-Process -Id $PID).Path
    $script:helperPath = Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretBackendUtilities.ps1'
}

Describe 'Exact owned process custody with real children' {
    BeforeEach {
        $script:injectKillFailure = $false
        $script:injectWaitFailure = $false
        $script:injectPipeFailure = $false
        $script:ownedProcess = $null
        $script:ownedHandle = $null
        $script:capturedTasks = [Collections.Generic.List[Threading.Tasks.Task]]::new()
        $script:custodyId = $null
        $script:control = [Diagnostics.Process]::new()
        $script:control.StartInfo = [Diagnostics.ProcessStartInfo]::new($pwshPath)
        $script:control.StartInfo.UseShellExecute = $false
        $script:control.StartInfo.CreateNoWindow = $true
        foreach ($arg in @('-NoLogo','-NoProfile','-Command','Start-Sleep -Seconds 60')) { [void]$script:control.StartInfo.ArgumentList.Add($arg) }
        if (-not $script:control.Start()) { throw 'Control process did not start.' }
        Mock Invoke-SecretBackendOwnedTermination {
            param($Process)
            $script:ownedProcess = $Process
            if ($null -eq $script:ownedHandle) { $script:ownedHandle = $Process.SafeHandle }
            if ($script:injectKillFailure) { throw [IO.IOException]::new('private-custody-sentinel kill refusal') }
            $Process.Kill($true)
        }
        Mock Wait-SecretBackendOwnedProcess {
            param($Process,$Milliseconds)
            if ($script:injectWaitFailure -and $Milliseconds -eq 5000) { return $false }
            return $Process.WaitForExit($Milliseconds)
        }
        Mock Read-SecretBackendOwnedOutput {
            param($Reader)
            $task = $Reader.ReadToEndAsync()
            $script:capturedTasks.Add($task)
            return $task
        }
        Mock Wait-SecretBackendOwnedPipeClosure {
            param($Tasks)
            if ($script:injectPipeFailure) { return $false }
            return [Threading.Tasks.Task]::WhenAll([Threading.Tasks.Task[]]$Tasks).Wait(5000)
        }
    }
    AfterEach {
        $script:injectKillFailure = $false
        $script:injectWaitFailure = $false
        $script:injectPipeFailure = $false
        if ($script:custodyId -and (Get-SecretBackendProcessRecovery $script:custodyId).CleanupPending) {
            $null = Repair-SecretBackendProcessCustody -CustodyId $script:custodyId
        }
        if (-not $script:control.HasExited) { $script:control.Kill($true) }
        if (-not $script:control.WaitForExit(5000)) { throw 'Control cleanup failed.' }
        $script:control.Dispose()
    }
    # Protects: retains the exact live child and output tasks after Kill refuses, then recovers without touching a control.
    # Detects: Lost child/handle/pipe identity or unconfirmed cleanup would admit another backend while the original remains active.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-SecretBackendProcessRecovery, Invoke-SecretBackendProcess, Repair-SecretBackendProcessCustody
    It 'retains the exact live child and output tasks after Kill refuses, then recovers without touching a control' {
        $script:injectKillFailure = $true
        $failure = $null
        $streams = @(& {
            try { Invoke-SecretBackendProcess -FilePath $pwshPath -Arguments @('-NoProfile','-Command',"[Console]::Out.Write('private-custody-sentinel');[Console]::Error.Write('private-custody-sentinel');Start-Sleep -Seconds 60") -TimeoutSeconds 2 }
            catch { $script:failure = $_ }
        } *>&1)
        $failure = $script:failure
        $failure.Exception.Message | Should -Match 'cleanup remains pending'
        $script:custodyId = [string]$failure.Exception.Data['CustodyId']
        $failure.Exception.Data['CleanupPending'] | Should -BeTrue
        (($streams | Out-String) + ($failure | Out-String)) | Should -Not -Match 'private-custody-sentinel'
        $streams.Count | Should -Be 0
        $entry = [WorkProfileBackend.PendingProcessCustody]::Find($custodyId)
        [object]::ReferenceEquals($entry.Process,$ownedProcess) | Should -BeTrue
        [object]::ReferenceEquals($entry.StdOutTask,$capturedTasks[0]) | Should -BeTrue
        [object]::ReferenceEquals($entry.StdErrTask,$capturedTasks[1]) | Should -BeTrue
        $entry.Process.HasExited | Should -BeFalse
        $ownedHandle.IsClosed | Should -BeFalse
        $stdoutReader = $entry.Process.StandardOutput
        $stderrReader = $entry.Process.StandardError
        $entry.StdOutTask.IsCompleted | Should -BeFalse
        $entry.StdErrTask.IsCompleted | Should -BeFalse
        $control.HasExited | Should -BeFalse
        (Get-SecretBackendProcessRecovery $custodyId).PSObject.Properties.Name -join ',' | Should -Be 'CustodyId,CleanupPending'
        # Metadata discovery remains possible if an outer provider sanitized its error.
        $metadata = @(Get-SecretBackendProcessRecovery)
        $metadata.CustodyId | Should -Contain $custodyId
        $metadata[0].PSObject.Properties.Name -join ',' | Should -Be 'CustodyId,CleanupPending'
        # A failed ordinary retry must retain the same identity and tasks.
        { Repair-SecretBackendProcessCustody $custodyId } | Should -Throw '*recovery remains pending*'
        [object]::ReferenceEquals([WorkProfileBackend.PendingProcessCustody]::Find($custodyId),$entry) | Should -BeTrue
        # Reloading the helper must not discard pending custody.
        . $helperPath
        [object]::ReferenceEquals([WorkProfileBackend.PendingProcessCustody]::Find($custodyId),$entry) | Should -BeTrue
        $script:injectKillFailure = $false
        $result = Repair-SecretBackendProcessCustody $custodyId
        $result.RecoveryCompleted | Should -BeTrue
        $result.CleanupPending | Should -BeFalse
        $entry.StdOutTask.IsCompleted | Should -BeTrue
        $entry.StdErrTask.IsCompleted | Should -BeTrue
        $ownedHandle.IsClosed | Should -BeTrue
        { $stdoutReader.Read() } | Should -Throw
        { $stderrReader.Read() } | Should -Throw
        [WorkProfileBackend.PendingProcessCustody]::Find($custodyId) | Should -BeNullOrEmpty
        (Get-SecretBackendProcessRecovery $custodyId).CleanupPending | Should -BeFalse
        $control.HasExited | Should -BeFalse
    }
    # Protects: retains exact process and tasks when OS exit confirmation refuses, then verifies normal recovery.
    # Detects: Lost child/handle/pipe identity or unconfirmed cleanup would admit another backend while the original remains active.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Invoke-SecretBackendProcess, Repair-SecretBackendProcessCustody
    It 'retains exact process and tasks when OS exit confirmation refuses, then verifies normal recovery' {
        $script:injectWaitFailure = $true
        try { Invoke-SecretBackendProcess -FilePath $pwshPath -Arguments @('-NoProfile','-Command','Start-Sleep -Seconds 60') -TimeoutSeconds 1; throw 'Expected failure.' }
        catch { $failure = $_ }
        $failure.Exception.Message | Should -Match 'cleanup remains pending'
        $script:custodyId = [string]$failure.Exception.Data['CustodyId']
        $entry = [WorkProfileBackend.PendingProcessCustody]::Find($custodyId)
        [object]::ReferenceEquals($entry.Process,$ownedProcess) | Should -BeTrue
        [object]::ReferenceEquals($entry.StdOutTask,$capturedTasks[0]) | Should -BeTrue
        [object]::ReferenceEquals($entry.StdErrTask,$capturedTasks[1]) | Should -BeTrue
        { Repair-SecretBackendProcessCustody $custodyId } | Should -Throw '*recovery remains pending*'
        $script:injectWaitFailure = $false
        (Repair-SecretBackendProcessCustody $custodyId).RecoveryCompleted | Should -BeTrue
        $ownedHandle.IsClosed | Should -BeTrue
        $control.HasExited | Should -BeFalse
    }
    # Protects: cleans up after a genuine disposed output reader raises an external IO failure.
    # Detects: Lost child/handle/pipe identity or unconfirmed cleanup would admit another backend while the original remains active.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Invoke-SecretBackendProcess
    It 'cleans up after a genuine disposed output reader raises an external IO failure' {
        Mock Read-SecretBackendOwnedOutput {
            param($Reader)
            $Reader.Dispose()
            # The actual .NET read raises ObjectDisposedException; runner is unmocked.
            return $Reader.ReadToEndAsync()
        }
        { Invoke-SecretBackendProcess -FilePath $pwshPath -Arguments @('-NoProfile','-Command','Start-Sleep -Seconds 60') } | Should -Throw '*could not be completed; private command output is withheld*'
        $ownedProcess | Should -Not -BeNullOrEmpty
        $ownedHandle.IsClosed | Should -BeTrue
        $control.HasExited | Should -BeFalse
    }
    # Protects: rejects an unknown recovery identity without touching the unrelated control.
    # Detects: Lost child/handle/pipe identity or unconfirmed cleanup would admit another backend while the original remains active.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Repair-SecretBackendProcessCustody
    It 'rejects an unknown recovery identity without touching the unrelated control' {
        { Repair-SecretBackendProcessCustody -CustodyId ('0'*32) } | Should -Throw '*identity is unavailable*'
        $control.HasExited | Should -BeFalse
    }
    # Protects: withholds successful output and retains exact completed pipes when pipe confirmation refuses.
    # Detects: Lost child/handle/pipe identity or unconfirmed cleanup would admit another backend while the original remains active.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Invoke-SecretBackendProcess, Repair-SecretBackendProcessCustody
    It 'withholds successful output and retains exact completed pipes when pipe confirmation refuses' {
        $script:injectPipeFailure = $true
        try { Invoke-SecretBackendProcess -FilePath $pwshPath -Arguments @('-NoProfile','-Command',"[Console]::Out.Write('private-custody-sentinel')"); throw 'Expected failure.' }
        catch { $failure = $_ }
        $failure.Exception.Message | Should -Match 'cleanup remains pending'
        ($failure | Out-String) | Should -Not -Match 'private-custody-sentinel'
        $script:custodyId = [string]$failure.Exception.Data['CustodyId']
        $entry = [WorkProfileBackend.PendingProcessCustody]::Find($custodyId)
        $handle = $entry.Process.SafeHandle
        $reader = $entry.Process.StandardOutput
        [object]::ReferenceEquals($entry.StdOutTask,$capturedTasks[0]) | Should -BeTrue
        [object]::ReferenceEquals($entry.StdErrTask,$capturedTasks[1]) | Should -BeTrue
        $entry.Process.HasExited | Should -BeTrue
        $handle.IsClosed | Should -BeFalse
        { Repair-SecretBackendProcessCustody $custodyId } | Should -Throw '*recovery remains pending*'
        $script:injectPipeFailure = $false
        (Repair-SecretBackendProcessCustody $custodyId).RecoveryCompleted | Should -BeTrue
        $handle.IsClosed | Should -BeTrue
        { $reader.Read() } | Should -Throw
        $control.HasExited | Should -BeFalse
    }
    # Protects: retains partial real read setup and recovers after an IO failure plus refused Kill.
    # Detects: Lost child/handle/pipe identity or unconfirmed cleanup would admit another backend while the original remains active.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Invoke-SecretBackendProcess, Repair-SecretBackendProcessCustody
    It 'retains partial real read setup and recovers after an IO failure plus refused Kill' {
        $script:injectKillFailure = $true
        Mock Read-SecretBackendOwnedOutput {
            param($Reader)
            if ($script:capturedTasks.Count -eq 1) {
                $Reader.Dispose()
                return $Reader.ReadToEndAsync()
            }
            $task = $Reader.ReadToEndAsync()
            $script:capturedTasks.Add($task)
            return $task
        }
        try { Invoke-SecretBackendProcess -FilePath $pwshPath -Arguments @('-NoProfile','-Command','Start-Sleep -Seconds 60'); throw 'Expected failure.' }
        catch { $failure = $_ }
        $failure.Exception.Message | Should -Match 'cleanup remains pending'
        $script:custodyId = [string]$failure.Exception.Data['CustodyId']
        $entry = [WorkProfileBackend.PendingProcessCustody]::Find($custodyId)
        [object]::ReferenceEquals($entry.Process,$ownedProcess) | Should -BeTrue
        [object]::ReferenceEquals($entry.StdOutTask,$capturedTasks[0]) | Should -BeTrue
        $entry.StdErrTask | Should -BeNullOrEmpty
        $entry.Process.HasExited | Should -BeFalse
        $script:injectKillFailure = $false
        (Repair-SecretBackendProcessCustody $custodyId).RecoveryCompleted | Should -BeTrue
        $entry.StdOutTask.IsCompleted | Should -BeTrue
        $ownedHandle.IsClosed | Should -BeTrue
        $control.HasExited | Should -BeFalse
    }
    # Protects: retains the exact blocked stdin task and original child through refused termination and explicit recovery.
    # Detects: Lost child/handle/pipe identity or unconfirmed cleanup would admit another backend while the original remains active.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Invoke-SecretBackendProcess, Repair-SecretBackendProcessCustody
    It 'retains the exact blocked stdin task and original child through refused termination and explicit recovery' {
        $script:injectKillFailure = $true
        $script:inputFailure = $null
        $streams = @(& {
            try {
                Invoke-SecretBackendProcess -FilePath $pwshPath -Arguments @('-NoLogo','-NoProfile','-Command',"[Console]::Out.Write('private-custody-sentinel');[Console]::Error.Write('private-custody-sentinel');Start-Sleep -Seconds 60") -StandardInput ('x' * 1048576) -TimeoutSeconds 1
            } catch { $script:inputFailure = $_ }
        } *>&1)
        $inputFailure | Should -Not -BeNullOrEmpty
        $script:custodyId = [string]$inputFailure.Exception.Data['CustodyId']
        $inputFailure.Exception.Data['CleanupPending'] | Should -BeTrue
        @($inputFailure.Exception.Data.Keys | Sort-Object) -join ',' | Should -Be 'CleanupPending,CustodyId'
        $streams.Count | Should -Be 0
        ($inputFailure | Out-String) | Should -Not -Match 'private-custody-sentinel|pwsh|Start-Sleep'
        $entry = [WorkProfileBackend.PendingProcessCustody]::Find($custodyId)
        [object]::ReferenceEquals($entry.Process,$ownedProcess) | Should -BeTrue
        [object]::ReferenceEquals($entry.StdOutTask,$capturedTasks[0]) | Should -BeTrue
        [object]::ReferenceEquals($entry.StdErrTask,$capturedTasks[1]) | Should -BeTrue
        $entry.RedirectInput | Should -BeTrue
        $entry.InputTask | Should -Not -BeNullOrEmpty
        $entry.InputTask.IsCompleted | Should -BeFalse
        $entry.Process.HasExited | Should -BeFalse
        $inputTask = $entry.InputTask
        { Repair-SecretBackendProcessCustody -CustodyId $custodyId } | Should -Throw '*recovery remains pending*'
        $retained = [WorkProfileBackend.PendingProcessCustody]::Find($custodyId)
        [object]::ReferenceEquals($retained,$entry) | Should -BeTrue
        [object]::ReferenceEquals($retained.Process,$ownedProcess) | Should -BeTrue
        [object]::ReferenceEquals($retained.InputTask,$inputTask) | Should -BeTrue
        [object]::ReferenceEquals($retained.StdOutTask,$capturedTasks[0]) | Should -BeTrue
        [object]::ReferenceEquals($retained.StdErrTask,$capturedTasks[1]) | Should -BeTrue
        $ownedHandle.IsClosed | Should -BeFalse
        $control.HasExited | Should -BeFalse
        $script:injectKillFailure = $false
        $recovered = Repair-SecretBackendProcessCustody -CustodyId $custodyId
        $recovered.RecoveryCompleted | Should -BeTrue
        $recovered.CleanupPending | Should -BeFalse
        [object]::ReferenceEquals($entry.InputTask,$inputTask) | Should -BeTrue
        $inputTask.IsCompleted | Should -BeTrue
        $entry.StdOutTask.IsCompleted | Should -BeTrue
        $entry.StdErrTask.IsCompleted | Should -BeTrue
        $ownedHandle.IsClosed | Should -BeTrue
        [WorkProfileBackend.PendingProcessCustody]::Find($custodyId) | Should -BeNullOrEmpty
        $control.HasExited | Should -BeFalse
    }

}

BeforeAll {
    . (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretBackendUtilities.ps1')
    $script:helperPath=Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretBackendUtilities.ps1'
    $script:pwshPath=(Get-Process -Id $PID).Path
}
Describe 'Pending exact custody blocks new real backend admission' {
    BeforeEach {
        $script:denyOwnedTermination=$true
        $script:custodyId=$null
        $script:control=[Diagnostics.Process]::new()
        $script:control.StartInfo=[Diagnostics.ProcessStartInfo]::new($pwshPath)
        $script:control.StartInfo.UseShellExecute=$false
        $script:control.StartInfo.CreateNoWindow=$true
        foreach($arg in @('-NoLogo','-NoProfile','-Command','Start-Sleep -Seconds 60')){$script:control.StartInfo.ArgumentList.Add($arg)}
        $control.Start()|Should -BeTrue
        Mock Invoke-SecretBackendOwnedTermination {
            param($Process)
            if($script:denyOwnedTermination){throw [IO.IOException]::new('synthetic-private-os-refusal')}
            $Process.Kill($true)
        }
        $firstFailure=$null
        try{$null=Invoke-SecretBackendProcess -FilePath $pwshPath -Arguments @('-NoLogo','-NoProfile','-Command',"[Console]::Out.Write('synthetic-private-output');Start-Sleep -Seconds 60") -TimeoutSeconds 1}
        catch{$firstFailure=$_.Exception}
        $firstFailure.Data['CleanupPending']|Should -BeTrue
        $script:custodyId=[string]$firstFailure.Data['CustodyId']
        $script:entry=[WorkProfileBackend.PendingProcessCustody]::Find($custodyId)
        $entry.Process.HasExited|Should -BeFalse
        $script:originalProcess=$entry.Process
        $script:originalStdout=$entry.StdOutTask
        $script:originalStderr=$entry.StdErrTask
    }
    AfterEach {
        $script:denyOwnedTermination=$false
        if($script:custodyId -and [WorkProfileBackend.PendingProcessCustody]::Find($custodyId)){$null=Repair-SecretBackendProcessCustody -CustodyId $custodyId}
        if(-not $control.HasExited){$control.Kill($true)}
        if(-not $control.WaitForExit(5000)){throw 'Owned unrelated control cleanup unconfirmed.'}
        $control.Dispose()
    }
    # Protects: withholds all output and exposes only opaque pending metadata before and after helper reload, then admits after exact recovery.
    # Detects: Lost child/handle/pipe identity or unconfirmed cleanup would admit another backend while the original remains active.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Invoke-SecretBackendProcess, Repair-SecretBackendProcessCustody
    It 'withholds all output and exposes only opaque pending metadata before and after helper reload, then admits after exact recovery' {
        {Repair-SecretBackendProcessCustody -CustodyId $custodyId}|Should -Throw '*recovery remains pending*'
        $marker=Join-Path $script:FixtureDrive 'never-started.txt'
        foreach($reload in @($false,$true)){
            if($reload){. $helperPath}
            $script:blockedFailure=$null
            $captured=@(& {
                try{Invoke-SecretBackendProcess -FilePath $pwshPath -Arguments @('-NoLogo','-NoProfile','-Command',"[IO.File]::WriteAllText('$($marker.Replace("'","''"))','started');[Console]::Out.Write('synthetic-private-new-output')") -StandardInput 'synthetic-private-stdin' -TimeoutSeconds 5}
                catch{$script:blockedFailure=$_.Exception}
            } *>&1)
            $captured.Count|Should -Be 0
            $blockedFailure|Should -Not -BeNullOrEmpty
            $blockedFailure.Data['CleanupPending']|Should -BeTrue
            $blockedFailure.Data['AdmissionBlocked']|Should -BeTrue
            @($blockedFailure.Data.Keys|Sort-Object) -join ','|Should -Be 'AdmissionBlocked,CleanupPending,CustodyIds'
            @($blockedFailure.Data['CustodyIds']).Count|Should -Be 1
            $blockedFailure.Data['CustodyIds'][0]|Should -Be $custodyId
            $blockedFailure.Data['CustodyIds'][0]|Should -Match '^[a-f0-9]{32}$'
            $blockedFailure.Message|Should -Not -Match 'synthetic-private|pwsh|never-started|\.ps1'
            Test-Path -LiteralPath $marker|Should -BeFalse
            [object]::ReferenceEquals([WorkProfileBackend.PendingProcessCustody]::Find($custodyId),$entry)|Should -BeTrue
            [object]::ReferenceEquals($entry.Process,$originalProcess)|Should -BeTrue
            [object]::ReferenceEquals($entry.StdOutTask,$originalStdout)|Should -BeTrue
            [object]::ReferenceEquals($entry.StdErrTask,$originalStderr)|Should -BeTrue
            $control.HasExited|Should -BeFalse
        }
        $script:denyOwnedTermination=$false
        (Repair-SecretBackendProcessCustody -CustodyId $custodyId).RecoveryCompleted|Should -BeTrue
        [WorkProfileBackend.PendingProcessCustody]::Ids().Length|Should -Be 0
        $result=Invoke-SecretBackendProcess -FilePath $pwshPath -Arguments @('-NoLogo','-NoProfile','-Command',"[Console]::Out.Write('after-exact-recovery')") -TimeoutSeconds 5
        $result.Success|Should -BeTrue
        $result.StdOut|Should -Be 'after-exact-recovery'
        # The unchanged helper contract returns exactly null on successful stderr.
        ($null -eq $result.StdErr)|Should -BeTrue
        $control.HasExited|Should -BeFalse
    }
    # Protects: blocks a newly imported module from starting a different backend while retaining the original host entry.
    # Detects: Lost child/handle/pipe identity or unconfirmed cleanup would admit another backend while the original remains active.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Invoke-SecretBackendProcess, Repair-SecretBackendProcessCustody
    It 'blocks a newly imported module from starting a different backend while retaining the original host entry' {
        $absentRoot=Join-Path $script:FixtureDrive 'module-machine-root'
        $warnings=@()
        $module=Import-Module (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretsTier.psm1') -ArgumentList 'DTM-WORK',$absentRoot -PassThru -Force -WarningVariable warnings
        try{
            $module.ExportedFunctions.Count|Should -Be 32
            $module.ExportedAliases.Count|Should -Be 1
            $warnings.Count|Should -Be 0
            Test-Path -LiteralPath $absentRoot|Should -BeFalse
            $marker=Join-Path $script:FixtureDrive 'never-cmd-started.txt'
            $failure=& $module {
                param($Marker)
                try{Invoke-SecretBackendProcess -FilePath (Join-Path $env:SystemRoot 'System32/cmd.exe') -Arguments @('/c',"echo started > `"$Marker`"") -TimeoutSeconds 5;throw 'A backend unexpectedly ran.'}
                catch{if(-not $_.Exception.Data['AdmissionBlocked']){throw};return $_.Exception}
            } $marker
            $failure.Data['CleanupPending']|Should -BeTrue
            $failure.Data['CustodyIds'][0]|Should -Be $custodyId
            Test-Path -LiteralPath $marker|Should -BeFalse
            [object]::ReferenceEquals([WorkProfileBackend.PendingProcessCustody]::Find($custodyId),$entry)|Should -BeTrue
            $entry.Process.HasExited|Should -BeFalse
            $control.HasExited|Should -BeFalse
        }finally{Remove-Module $module -Force}
        $script:denyOwnedTermination=$false
        (Repair-SecretBackendProcessCustody -CustodyId $custodyId).RecoveryCompleted|Should -BeTrue
        [WorkProfileBackend.PendingProcessCustody]::Ids().Length|Should -Be 0
    }
}

Describe 'Canonical credential source and passive machine configuration' {
    # Protects: declares exactly the reviewed two sibling managed sources.
    # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1 and SecretBackendUtilities.ps1: passive import/custody contract
    It 'declares exactly the reviewed two sibling managed sources' {
        $manifestPath = Join-Path (Split-Path -Parent $helperPath) 'managed-files.json'
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        @($manifest.PSObject.Properties.Name | Sort-Object) -join ',' | Should -BeExactly 'files,version'
        $manifest.version | Should -Be 1
        @($manifest.files).Count | Should -Be 2
        $manifest.files[0] | Should -BeExactly 'SecretBackendUtilities.ps1'
        $manifest.files[1] | Should -BeExactly 'SecretsTier.psm1'
        foreach ($name in $manifest.files) { Test-Path -LiteralPath (Join-Path (Split-Path -Parent $helperPath) $name) -PathType Leaf | Should -BeTrue }
    }
    # Protects: imports the actual sibling module without provisioning or changing process credentials and exposes the intended contract.
    # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1 and SecretBackendUtilities.ps1: passive import/custody contract
    It 'imports the actual sibling module without provisioning or changing process credentials and exposes the intended contract' {
        $absentRoot = Join-Path $script:FixtureDrive 'passive-machine-root'
        $warnings = @()
        $previousSession = $env:BW_SESSION
        $previousToken = $env:BWS_ACCESS_TOKEN
        $module = Import-Module (Join-Path (Split-Path -Parent $helperPath) 'SecretsTier.psm1') -ArgumentList 'fixture-machine',$absentRoot -Force -PassThru -WarningVariable warnings -ErrorAction Stop
        try {
            $warnings.Count | Should -Be 0
            Test-Path -LiteralPath $absentRoot | Should -BeFalse
            $env:BW_SESSION | Should -Be $previousSession
            $env:BWS_ACCESS_TOKEN | Should -Be $previousToken
            $module.ExportedFunctions.Count | Should -Be 32
            $module.ExportedAliases.Count | Should -Be 1
            $module.ExportedAliases['Ensure-LocalSecretVaultRegistration'].Definition | Should -Be 'Register-LocalSecretVault'
            $module.ExportedFunctions.ContainsKey('Get-SecretBackendProcessRecovery') | Should -BeFalse
            $module.ExportedFunctions.ContainsKey('Repair-SecretBackendProcessCustody') | Should -BeFalse
        } finally { Remove-Module $module -Force }
    }
    # Protects: uses the explicit machine selector and cache root without provisioning the selected path.
    # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1 and SecretBackendUtilities.ps1: passive import/custody contract
    It 'uses the explicit machine selector and cache root without provisioning the selected path' {
        $selectedRoot = Join-Path $script:FixtureDrive 'selected-machine-root'
        $module = Import-Module (Join-Path (Split-Path -Parent $helperPath) 'SecretsTier.psm1') -ArgumentList 'fixture-selected-host',$selectedRoot -Force -PassThru -ErrorAction Stop
        try {
            $actual = & $module { [pscustomobject]@{MachineDir=$script:MachineDir;CacheFile=$script:CacheFile;ManifestFile=$script:ManifestFile;BackupNote=$script:BwSecureNoteName} }
            $actual.MachineDir | Should -Be ([IO.Path]::GetFullPath($selectedRoot))
            $actual.CacheFile | Should -Be (Join-Path $selectedRoot 'secrets-cache.enc')
            $actual.ManifestFile | Should -Be (Join-Path $selectedRoot 'secrets-manifest.json')
            $actual.BackupNote | Should -BeExactly '[MACHINE] fixture-selected-host Secrets'
            Test-Path -LiteralPath $selectedRoot | Should -BeFalse
        } finally { Remove-Module $module -Force }
    }
}

}
Describe 'Protected runtime consumers and authenticated legacy reads' {
BeforeAll {
    $script:ConsumerHostEnvironment=@{USERPROFILE=$env:USERPROFILE;TEMP=$env:TEMP;TMP=$env:TMP}
    . (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretBackendUtilities.ps1')
    Initialize-CredentialFileInterop
    function New-ConsumerPrivateText {
        param([string]$Path,[string]$Text)
        $stream=New-CredentialPrivateStage $Path ([Text.Encoding]::UTF8.GetBytes($Text))
        $stream.Dispose()
    }
    function New-ConsumerLegacyPair {
        $dir=Join-Path $env:USERPROFILE '.machine'
        [IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new($dir),(New-CredentialPrivateAcl -Directory))
        $acl=Get-Acl $dir
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]'CreateFiles,DeleteSubdirectoriesAndFiles',[Security.AccessControl.AccessControlType]::Allow))
        Set-Acl $dir $acl
        $plain=[Text.Encoding]::UTF8.GetBytes('{"PCAI_CONSUMER_SYNTHETIC":"fixture-secret"}')
        try{$cipher=[Convert]::ToBase64String([Security.Cryptography.ProtectedData]::Protect($plain,$null,[Security.Cryptography.DataProtectionScope]::CurrentUser))}finally{[Array]::Clear($plain,0,$plain.Length)}
        $script:legacyCache=Join-Path $dir 'secrets-cache.enc'
        $script:legacyManifest=Join-Path $dir 'secrets-manifest.json'
        New-ConsumerPrivateText $legacyCache $cipher
        New-ConsumerPrivateText $legacyManifest (@{machineName='fixture-machine';secretCount=1;lastUpdated=[DateTime]::UtcNow.ToString('o');cacheSha256=(Get-FileHash $legacyCache).Hash}|ConvertTo-Json -Compress)
        $script:legacyBefore=@($legacyCache,$legacyManifest|ForEach-Object{[pscustomobject]@{Path=$_;Hash=(Get-FileHash $_).Hash;Sddl=(Get-Acl $_).Sddl}})
    }
    function Assert-ConsumerLegacyUnchanged {
        foreach($file in $legacyBefore){(Get-FileHash $file.Path).Hash|Should -BeExactly $file.Hash;(Get-Acl $file.Path).Sddl|Should -BeExactly $file.Sddl}
    }
}
Describe 'Private credential consumer runtime contract' {
BeforeEach {
    $script:caseRoot=Join-Path $script:FixtureDrive ([guid]::NewGuid().ToString('N'))
    [IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new($caseRoot),(New-CredentialPrivateAcl -Directory))
    $env:USERPROFILE=Join-Path $caseRoot 'user'
    $env:TEMP=Join-Path $caseRoot 'unsafe-temp';$env:TMP=$env:TEMP
    foreach($dir in @($env:USERPROFILE,$env:TEMP)){[IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new($dir),(New-CredentialPrivateAcl -Directory))}
    $acl=Get-Acl $env:TEMP
    $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]'CreateFiles,DeleteSubdirectoriesAndFiles',[Security.AccessControl.AccessControlType]::Allow))
    Set-Acl $env:TEMP $acl
    $env:PCAI_CREDENTIAL_ROOT=$null;$env:BW_SESSION=$null;$env:BW_CLIENTID=$null;$env:BW_CLIENTSECRET=$null;$env:PCAI_CONSUMER_SYNTHETIC=$null
    $script:module=$null
}
AfterEach {
    if($module){Remove-Module $module -Force}
    $env:PCAI_CREDENTIAL_ROOT=$null;$env:BW_SESSION=$null;$env:PCAI_CONSUMER_SYNTHETIC=$null
    foreach($name in $script:ConsumerHostEnvironment.Keys){[Environment]::SetEnvironmentVariable($name,$script:ConsumerHostEnvironment[$name],'Process')}
}
Describe 'Actual default protected bootstrap integration' {
    BeforeEach {
        . (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretBackendUtilities.ps1')
        $dir=Join-Path $env:USERPROFILE '.bwdata'
        [IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new($dir),(New-CredentialPrivateAcl -Directory))
        $script:unlocked=$false;$script:passwordPath=$null
        Mock Resolve-BwBootstrapState {[pscustomobject]@{Password='test-synthetic-master';PasswordSource='fixture';SessionFile=(Join-Path $env:USERPROFILE '.bwdata/session.txt');ClientId=$null;ClientSecret=$null}}
        Mock Invoke-BitwardenCli {
            if($Arguments[0] -eq 'status'){return [pscustomobject]@{ExitCode=0;Success=$true;StdOut=if($script:unlocked){'{"status":"unlocked"}'}else{'{"status":"locked"}'}}}
            if($Arguments[0] -eq 'unlock'){
                $script:passwordPath=$Arguments[2]
                [IO.File]::ReadAllText($passwordPath)|Should -BeExactly 'test-synthetic-master'
                Assert-BitwardenPrivateInput $passwordPath
                $script:unlocked=$true
                return [pscustomobject]@{ExitCode=0;Success=$true;StdOut='accepted-consumer-synthetic-session'}
            }
            throw 'Unexpected provider boundary.'
        }
    }
    # Protects: uses protected profile bootstrap despite actual foreign-write synthetic TEMP.
    # Detects: Mixed credential sources or premature session acceptance would use the wrong identity or lose the prior process state.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Initialize-BitwardenSessionFromBackends
    It 'uses protected profile bootstrap despite actual foreign-write synthetic TEMP' {
        $result=Initialize-BitwardenSessionFromBackends -Refresh -Confirm:$false
        $result.Success|Should -BeTrue
        $passwordPath.StartsWith((Join-Path $env:USERPROFILE '.pcai-credential-state/bootstrap'),[StringComparison]::OrdinalIgnoreCase)|Should -BeTrue
        Test-Path $passwordPath|Should -BeFalse
        @(Get-ChildItem $env:TEMP -Force).Count|Should -Be 0
        [IO.File]::ReadAllText((Join-Path $env:USERPROFILE '.bwdata/session.txt'))|Should -BeExactly 'accepted-consumer-synthetic-session'
    }
    # Protects: WhatIf provisions no storage and calls no provider.
    # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Initialize-BitwardenSessionFromBackends
    It 'WhatIf provisions no storage and calls no provider' {
        (Initialize-BitwardenSessionFromBackends -WhatIf).Success|Should -BeFalse
        Test-Path (Join-Path $env:USERPROFILE '.pcai-credential-state')|Should -BeFalse
        Should -Invoke Resolve-BwBootstrapState -Times 0 -Exactly
        Should -Invoke Invoke-BitwardenCli -Times 0 -Exactly
    }
    # Protects: invalid explicit root blocks even existing-session status without mutation.
    # Detects: Mixed credential sources or premature session acceptance would use the wrong identity or lose the prior process state.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Initialize-BitwardenSessionFromBackends
    It 'invalid explicit root blocks even existing-session status without mutation' {
        $env:BW_SESSION='prior-synthetic-session'
        (Initialize-BitwardenSessionFromBackends -CredentialRoot $env:TEMP -Confirm:$false).Success|Should -BeFalse
        $env:BW_SESSION|Should -BeExactly 'prior-synthetic-session'
        Should -Invoke Resolve-BwBootstrapState -Times 0 -Exactly
        Should -Invoke Invoke-BitwardenCli -Times 0 -Exactly
        @(Get-ChildItem $env:TEMP -Force).Count|Should -Be 0
    }
}
Describe 'Actual protected default cache and read-only authenticated legacy integration' {
    BeforeEach {
        $script:module=Import-Module (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretsTier.psm1') -ArgumentList @('fixture-machine') -Force -PassThru -ErrorAction Stop
    }

    BeforeAll {
        function New-ConsumerProtectedLegacyPair {
            $script:protectedCachePaths=& $script:module {
                $null=Initialize-CredentialPrivateStorage -Selection $script:CacheSelection
                [pscustomobject]@{Cache=$script:CacheFile;Manifest=$script:ManifestFile}
            }
            $plain=[Text.Encoding]::UTF8.GetBytes('{"PCAI_CONSUMER_SYNTHETIC":"previous-cache"}')
            try{$cipher=[Convert]::ToBase64String([Security.Cryptography.ProtectedData]::Protect($plain,$null,[Security.Cryptography.DataProtectionScope]::CurrentUser))}
            finally{[Array]::Clear($plain,0,$plain.Length)}
            New-ConsumerPrivateText $script:protectedCachePaths.Cache $cipher
            New-ConsumerPrivateText $script:protectedCachePaths.Manifest (@{machineName='fixture-machine';secretCount=1;lastUpdated=[DateTime]::UtcNow.ToString('o');cacheSha256=(Get-FileHash $script:protectedCachePaths.Cache).Hash}|ConvertTo-Json -Compress)
            $script:protectedBefore=@(foreach($path in @($script:protectedCachePaths.Cache,$script:protectedCachePaths.Manifest)){
                $read=Open-CredentialPrivateRead $path
                try{[pscustomobject]@{Path=$path;Hash=$read.Snapshot.Hash;Identity=$read.Snapshot.Identity;Sddl=(Get-CredentialDescriptorSddl $read.Snapshot.Descriptor)}}
                finally{$read.Stream.Dispose()}
            })
        }
        function Assert-ConsumerProtectedPairUnchanged {
            foreach($before in $script:protectedBefore){
                $read=Open-CredentialPrivateRead $before.Path
                try{
                    $read.Snapshot.Hash|Should -BeExactly $before.Hash
                    $read.Snapshot.Identity|Should -BeExactly $before.Identity
                    (Get-CredentialDescriptorSddl $read.Snapshot.Descriptor)|Should -BeExactly $before.Sddl
                }finally{$read.Stream.Dispose()}
            }
        }
    }
    # Protects: a usable cache through failure of the obsolete second publication boundary.
    # Detects: replacing ciphertext before a failing manifest write destroys the previous coherent fallback.
    # Needs: Real protected NTFS legacy pair and DPAPI, one injected manifest failure; no providers.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Save-SecretsToCache,Get-SecretsFromCache
    It 'keeps a coherent usable cache when a separate manifest write is unavailable' {
        New-ConsumerProtectedLegacyPair
        $script:cacheWriteImplementation=& $script:module {(Get-Command Write-BitwardenPrivateFile).ScriptBlock}
        Mock -ModuleName SecretsTier Write-BitwardenPrivateFile {
            param($LiteralPath,$Value,[switch]$Overwrite)
            if($LiteralPath -eq $script:protectedCachePaths.Manifest){throw 'Synthetic second publication unavailable.'}
            & $script:cacheWriteImplementation -LiteralPath $LiteralPath -Value $Value -Overwrite:$Overwrite -Confirm:$false
        }
        $failed=$false
        try{& $script:module {Save-SecretsToCache @{PCAI_CONSUMER_SYNTHETIC='new-cache'} -Confirm:$false}}
        catch{$failed=$true}
        $read=& $script:module {Get-SecretsFromCache}
        $read.PCAI_CONSUMER_SYNTHETIC|Should -BeExactly $(if($failed){'previous-cache'}else{'new-cache'})
        (& $script:module {Get-CacheStatus}).IsValid|Should -BeTrue
        Should -Invoke -ModuleName SecretsTier Write-BitwardenPrivateFile -Times 0 -Exactly -ParameterFilter {$LiteralPath -eq $script:protectedCachePaths.Manifest}
    }
    # Protects: the exact previous pair when the sole canonical cache publication fails.
    # Detects: failed writes must preserve predecessor identities, ciphertext bytes, metadata bytes and owner/group/DACL.
    # Needs: Actual protected NTFS pair and an injected prepublication failure; no live credentials.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Save-SecretsToCache
    It 'preserves the previous readable pair exactly when canonical cache publication fails' {
        New-ConsumerProtectedLegacyPair
        Mock -ModuleName SecretsTier Write-BitwardenPrivateFile {throw 'Synthetic canonical publication refused.'}
        {& $script:module {Save-SecretsToCache @{PCAI_CONSUMER_SYNTHETIC='new-cache'} -Confirm:$false}}|Should -Throw
        Assert-ConsumerProtectedPairUnchanged
        (& $script:module {Get-SecretsFromCache}).PCAI_CONSUMER_SYNTHETIC|Should -BeExactly 'previous-cache'
    }


    # Protects: one atomic cache artifact with no redundant v1 manifest after qualified migration.
    # Detects: successful updates must not create another publication boundary or retain unused metadata copies.
    # Needs: Real protected v1 fixture, NTFS replacement and current-user DPAPI.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Save-SecretsToCache,Remove-SecretsLegacyManifestInternal
    It 'publishes one self-contained cache file and retires only qualified old metadata' {
        New-ConsumerProtectedLegacyPair
        & $script:module {Save-SecretsToCache @{PCAI_CONSUMER_SYNTHETIC='single-envelope'} -Confirm:$false}
        $text=[IO.File]::ReadAllText($script:protectedCachePaths.Cache)
        $text.StartsWith(('#pcai-dpapi-cache-v2'+[char]10),[StringComparison]::Ordinal)|Should -BeTrue
        Test-Path -LiteralPath $script:protectedCachePaths.Manifest|Should -BeFalse
        $text|Should -Not -Match 'single-envelope'
        (& $script:module {Get-SecretsFromCache}).PCAI_CONSUMER_SYNTHETIC|Should -BeExactly 'single-envelope'
    }
    # Protects: status observation without plaintext recovery or implicit environment publication.
    # Detects: metadata inspection must not invoke DPAPI decryption or return cached secrets.
    # Needs: An actual saved encrypted envelope and a fail-closed decryption mock.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Get-CacheStatus,Read-SecretsCacheEnvelopeInternal
    It 'reports v2 cache metadata without decrypting or exposing values' {
        & $script:module {Save-SecretsToCache @{PCAI_CONSUMER_SYNTHETIC='status-private-sentinel'} -Confirm:$false}
        Mock -ModuleName SecretsTier ConvertFrom-EncryptedString {throw 'Status must not decrypt.'}
        $status=& $script:module {Get-CacheStatus}
        $status.IsValid|Should -BeTrue;$status.Source|Should -BeExactly 'Protected';$status.SecretCount|Should -Be 1
        ($status|ConvertTo-Json -Depth 5)|Should -Not -Match 'status-private-sentinel|ciphertext'
        Should -Invoke -ModuleName SecretsTier ConvertFrom-EncryptedString -Times 0 -Exactly
        $env:PCAI_CONSUMER_SYNTHETIC|Should -BeNullOrEmpty
    }
    # Protects: recognized new cache formats from fallback or acceptance after metadata tampering.
    # Detects: unknown versions, malformed fields or changed public metadata must not recover secrets through the valid legacy cache.
    # Needs: Real DPAPI envelope plus unchanged usable legacy fixture; all values are synthetic.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Get-SecretsCachePair,Read-SecretsCacheEnvelopeInternal
    It 'refuses tampered v2 cache <Field> without legacy rollback or hydration' -TestCases @(
        @{Field='Version'},@{Field='Schema'},@{Field='Machine'},@{Field='Count'},@{Field='Generation'},
        @{Field='Timestamp'},@{Field='Source'},@{Field='Digest'},@{Field='Ciphertext'},@{Field='Duplicate'},@{Field='Prefix'}
    ){
        param($Field)
        New-ConsumerLegacyPair
        & $script:module {Save-SecretsToCache @{PCAI_CONSUMER_SYNTHETIC='new-private-sentinel'} -Confirm:$false}
        $path=& $script:module {$script:CacheFile}
        $text=[IO.File]::ReadAllText($path);$body=$text.Substring($text.IndexOf([char]10)+1)
        $metadata=$body|ConvertFrom-Json -AsHashtable;$prefix='#pcai-dpapi-cache-v2'+[char]10
        switch($Field){
            Version {$metadata.version=3} Schema {$metadata.schema='unsupported'}
            Machine {$metadata.machineName='different-machine'} Count {$metadata.secretCount=2}
            Generation {$metadata.generation='f'*32}
            Timestamp {$metadata.lastUpdated=[DateTime]::UtcNow.AddDays(1).ToString('o')}
            Source {$metadata.source='unqualified'} Digest {$metadata.cacheSha256='0'*64}
            Ciphertext {$metadata.ciphertext='not-base64'}
            Duplicate {$body=$body.Substring(0,$body.Length-1)+',"version":2}'}
            Prefix {$prefix='#pcai-dpapi-cache-v3'+[char]10}
        }
        if($Field -ne 'Duplicate'){$body=$metadata|ConvertTo-Json -Compress}
        [IO.File]::WriteAllText($path,($prefix+$body),[Text.UTF8Encoding]::new($false))
        {& $script:module {Get-SecretsFromCache}}|Should -Throw '*private content is withheld*'
        $env:PCAI_CONSUMER_SYNTHETIC|Should -BeNullOrEmpty
        Assert-ConsumerLegacyUnchanged
    }
    # Protects: agreement between authenticated DPAPI metadata and its public status envelope.
    # Detects: a structurally valid DPAPI payload with different machine/count/time/generation/source must not be consumed.
    # Needs: Synthetic current-user DPAPI payload mutations with recomputed ciphertext digest.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Read-SecretsCacheEnvelopeInternal
    It 'refuses authenticated cache payload disagreement in <Field>' -TestCases @(
        @{Field='machineName'},@{Field='generation'},@{Field='lastUpdated'},@{Field='secretCount'},@{Field='source'},@{Field='schema'}
    ){
        param($Field)
        & $script:module {Save-SecretsToCache @{PCAI_CONSUMER_SYNTHETIC='bound-private-sentinel'} -Confirm:$false}
        $path=& $script:module {$script:CacheFile}
        $text=[IO.File]::ReadAllText($path);$metadata=$text.Substring($text.IndexOf([char]10)+1)|ConvertFrom-Json -AsHashtable
        $plain=& $script:module {param($cipher) ConvertFrom-EncryptedString $cipher} $metadata.ciphertext
        $payload=$plain|ConvertFrom-Json -AsHashtable;$plain=$null
        if($Field -eq 'secretCount'){$payload[$Field]=2}else{$payload[$Field]='different-synthetic-metadata'}
        $metadata.ciphertext=& $script:module {param($json) ConvertTo-EncryptedString $json} ($payload|ConvertTo-Json -Depth 20 -Compress)
        $metadata.cacheSha256=Get-CredentialBytesHash ([Text.Encoding]::UTF8.GetBytes($metadata.ciphertext))
        [IO.File]::WriteAllText($path,('#pcai-dpapi-cache-v2'+[char]10+($metadata|ConvertTo-Json -Compress)),[Text.UTF8Encoding]::new($false))
        {& $script:module {Get-SecretsFromCache}}|Should -Throw '*private content is withheld*'
        $env:PCAI_CONSUMER_SYNTHETIC|Should -BeNullOrEmpty
    }
    # Protects: preexisting metadata custody through qualification-to-capture races.
    # Detects: changed metadata arriving after legacy validation must not be adopted as a safe deletion target.
    # Needs: A real validated v1 pair with a synthetic post-read content change.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Save-SecretsToCache
    It 'preserves old metadata changed immediately after legacy pair validation' {
        New-ConsumerProtectedLegacyPair
        $script:cacheReadImplementation=& $script:module {(Get-Command Read-SecretsCachePair).ScriptBlock}
        Mock -ModuleName SecretsTier Read-SecretsCachePair {
            param($Pair,[switch]$Decrypt)
            $result=& $script:cacheReadImplementation -Pair $Pair -Decrypt:$Decrypt
            if($Pair.Layout -eq 'PairV1'){
                $metadata=[IO.File]::ReadAllText($Pair.Manifest)|ConvertFrom-Json -AsHashtable
                $metadata.uniqueNote='arrived-after-qualification'
                [IO.File]::WriteAllText($Pair.Manifest,($metadata|ConvertTo-Json -Compress),[Text.UTF8Encoding]::new($false))
            }
            return $result
        }
        $warnings=@()
        & $script:module {Save-SecretsToCache @{PCAI_CONSUMER_SYNTHETIC='new-qualified-cache'} -Confirm:$false -WarningVariable warnings -WarningAction SilentlyContinue;[pscustomobject]@{Warnings=@($warnings)}}|ForEach-Object{$warnings=$_.Warnings}
        @($warnings).Count|Should -Be 1
        [IO.File]::ReadAllText($script:protectedCachePaths.Manifest)|Should -Match 'arrived-after-qualification'
        (& $script:module {Get-SecretsFromCache}).PCAI_CONSUMER_SYNTHETIC|Should -BeExactly 'new-qualified-cache'
    }
    # Protects: changed or unknown old manifest data after the single-file commit.
    # Detects: cleanup must preserve modified contents, replacement identities, ACLs, streams and unsupported attributes.
    # Needs: Actual NTFS metadata with isolated post-publication mutation before guarded cleanup.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Remove-SecretsLegacyManifestInternal
    It 'preserves old cache metadata when cleanup sees <Mode>' -TestCases @(
        @{Mode='Text'},@{Mode='Identity'},@{Mode='Acl'},@{Mode='Stream'},@{Mode='ReadOnly'}
    ){
        param($Mode)
        New-ConsumerProtectedLegacyPair
        $script:cacheMetadataMutation=$Mode
        $script:cacheWriteImplementation=& $script:module {(Get-Command Write-BitwardenPrivateFile).ScriptBlock}
        Mock -ModuleName SecretsTier Write-BitwardenPrivateFile {
            param($LiteralPath,$Value,[switch]$Overwrite)
            & $script:cacheWriteImplementation -LiteralPath $LiteralPath -Value $Value -Overwrite:$Overwrite -Confirm:$false
            if($LiteralPath -eq $script:protectedCachePaths.Cache){
                $path=$script:protectedCachePaths.Manifest
                switch($script:cacheMetadataMutation){
                    Text {[IO.File]::AppendAllText($path,' ')}
                    Identity {
                        $text=[IO.File]::ReadAllText($path)
                        [IO.File]::Move($path,(Join-Path $env:USERPROFILE 'owned-retired-manifest.json'))
                        New-ConsumerPrivateText $path $text
                    }
                    Acl {
                        $acl=Get-Acl $path
                        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.WindowsIdentity]::GetCurrent().User,[Security.AccessControl.FileSystemRights]::WriteData,[Security.AccessControl.AccessControlType]::Deny))
                        Set-Acl $path $acl
                    }
                    Stream {[IO.File]::WriteAllText(($path+':synthetic-extra'),'unrecognized-synthetic-stream')}
                    ReadOnly {[IO.File]::SetAttributes($path,[IO.FileAttributes]::ReadOnly)}
                }
            }
        }
        $warnings=& $script:module {
            $warnings=@()
            Save-SecretsToCache @{PCAI_CONSUMER_SYNTHETIC='new-qualified-cache'} -Confirm:$false -WarningVariable warnings -WarningAction SilentlyContinue
            ,@($warnings)
        }
        @($warnings).Count|Should -Be 1
        Test-Path -LiteralPath $script:protectedCachePaths.Manifest|Should -BeTrue
        if($Mode -eq 'Stream'){[IO.File]::ReadAllText(($script:protectedCachePaths.Manifest+':synthetic-extra'))|Should -BeExactly 'unrecognized-synthetic-stream'}
        (& $script:module {Get-SecretsFromCache}).PCAI_CONSUMER_SYNTHETIC|Should -BeExactly 'new-qualified-cache'
    }
    # Protects: a coherent fallback after actual partial NTFS displacement during canonical publication.
    # Detects: failed single-file replacement must recover the previous DPAPI cache and exact metadata bytes/DACL.
    # Needs: Real NTFS move and existing private-file recovery; all content is synthetic.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretsTier.psm1,SecretBackendUtilities.ps1}: Save-SecretsToCache,Write-BitwardenPrivateFile
    It 'keeps the old cache usable after actual partial canonical displacement' {
        New-ConsumerProtectedLegacyPair
        Mock -ModuleName SecretsTier Invoke-CredentialFileReplace {
            param($Stage,$Target,$Backup)
            [IO.File]::Move($Target,$Backup)
            throw 'Synthetic partial canonical displacement.'
        }
        {& $script:module {Save-SecretsToCache @{PCAI_CONSUMER_SYNTHETIC='new-cache'} -Confirm:$false}}|Should -Throw
        foreach($before in $script:protectedBefore){
            (Get-FileHash $before.Path).Hash|Should -BeExactly $before.Hash
            (Get-Acl $before.Path).Sddl|Should -BeExactly $before.Sddl
        }
        (& $script:module {Get-SecretsFromCache}).PCAI_CONSUMER_SYNTHETIC|Should -BeExactly 'previous-cache'
    }

    # Protects: import and WhatIf create no directories or provider effects.
    # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Save-SecretsToCache
    It 'import and WhatIf create no directories or provider effects' {
        Test-Path (Join-Path $env:USERPROFILE '.pcai-credential-state')|Should -BeFalse
        Test-Path (Join-Path $env:USERPROFILE '.machine')|Should -BeFalse
        Mock -ModuleName SecretsTier Invoke-BitwardenCli {throw 'No provider permitted.'}
        & $module {Save-SecretsToCache @{fixture='synthetic'} -WhatIf}
        Test-Path (Join-Path $env:USERPROFILE '.pcai-credential-state')|Should -BeFalse
        Should -Invoke -ModuleName SecretsTier Invoke-BitwardenCli -Times 0 -Exactly
    }
    # Protects: default save publishes actual private DPAPI cache without changing legacy namespace.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-CacheStatus, Get-SecretsFromCache, Save-SecretsToCache
    It 'default save publishes actual private DPAPI cache without changing legacy namespace' {
        New-ConsumerLegacyPair
        & $module {Save-SecretsToCache @{PCAI_CONSUMER_SYNTHETIC='new-fixture'} -Confirm:$false}
        $selected=& $module {$script:CacheDir}
        $selected.StartsWith((Join-Path $env:USERPROFILE '.pcai-credential-state/cache'),[StringComparison]::OrdinalIgnoreCase)|Should -BeTrue
        [IO.Path]::GetFileName($selected)|Should -Match '^[A-F0-9]{64}$'
        $cache=Join-Path $selected 'secrets-cache.enc'
        [IO.File]::ReadAllText($cache)|Should -Not -Match 'new-fixture'
        (Get-Acl $cache).AreAccessRulesProtected|Should -BeTrue
        (& $module {Get-SecretsFromCache}).PCAI_CONSUMER_SYNTHETIC|Should -BeExactly 'new-fixture'
        (& $module {Get-CacheStatus}).Source|Should -BeExactly 'Protected'
        & $module {Save-SecretsToCache @{PCAI_CONSUMER_SYNTHETIC='replacement-fixture'} -Confirm:$false}
        (& $module {Get-SecretsFromCache}).PCAI_CONSUMER_SYNTHETIC|Should -BeExactly 'replacement-fixture'
        Assert-ConsumerLegacyUnchanged
    }
    # Protects: refuses preferred path directories instead of falling back to legacy (<Both>).
    # Detects: A directory substituted for either preferred cache file must cause rejection instead of rollback to an otherwise usable legacy pair.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-SecretsFromCache, Initialize-CredentialPrivateStorage, New-CredentialPrivateAcl
    It 'refuses preferred path directories instead of falling back to legacy (<Both>)' -TestCases @(@{Both=$false},@{Both=$true}) {
        param($Both)
        New-ConsumerLegacyPair
        $selected=& $module {Initialize-CredentialPrivateStorage $script:CacheSelection}
        [IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new((Join-Path $selected 'secrets-cache.enc')),(New-CredentialPrivateAcl -Directory))
        if($Both){[IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new((Join-Path $selected 'secrets-manifest.json')),(New-CredentialPrivateAcl -Directory))}
        {& $module {Get-SecretsFromCache}}|Should -Throw '*private content is withheld*'
        Assert-ConsumerLegacyUnchanged
    }
    # Protects: reads a literal long private path and refuses an actual named alternate stream.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: New-CredentialPrivateAcl, Open-CredentialPrivateRead
    It 'reads a literal long private path and refuses an actual named alternate stream' {
        $directory=$env:USERPROFILE
        while($directory.Length -lt 280){$directory=Join-Path $directory 'long-literal-segment';[IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new($directory),(New-CredentialPrivateAcl -Directory))}
        $path=Join-Path $directory 'synthetic-private.txt'
        $path.Length|Should -BeGreaterThan 260
        New-ConsumerPrivateText $path 'synthetic-long-path'
        $read=Open-CredentialPrivateRead $path
        try{$read.Text|Should -BeExactly 'synthetic-long-path'}finally{$read.Stream.Dispose()}
        [IO.File]::WriteAllText(('{0}:foreign' -f $path),'synthetic-stream')
        {Open-CredentialPrivateRead $path}|Should -Throw '*private content is withheld*'
        [IO.File]::ReadAllText($path)|Should -BeExactly 'synthetic-long-path'
    }
    # Protects: reads actual private legacy DPAPI data without provisioning migrating or hydrating.
    # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-CacheStatus, Get-SecretsFromCache
    It 'reads actual private legacy DPAPI data without provisioning migrating or hydrating' {
        New-ConsumerLegacyPair
        (& $module {Get-SecretsFromCache}).PCAI_CONSUMER_SYNTHETIC|Should -BeExactly 'fixture-secret'
        (& $module {Get-CacheStatus}).Source|Should -BeExactly 'LegacyReadOnly'
        $env:PCAI_CONSUMER_SYNTHETIC|Should -BeNullOrEmpty
        Test-Path (Join-Path $env:USERPROFILE '.pcai-credential-state')|Should -BeFalse
        Test-Path (Join-Path $env:USERPROFILE '.machine/.pcai-private-write')|Should -BeFalse
        Assert-ConsumerLegacyUnchanged
    }
    # Protects: legacy metadata status never decrypts secret plaintext.
    # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-CacheStatus
    It 'legacy metadata status never decrypts secret plaintext' {
        New-ConsumerLegacyPair
        Mock -ModuleName SecretsTier ConvertFrom-EncryptedString {throw 'Unexpected decrypt.'}
        $state=& $module {Get-CacheStatus}
        $state.IsValid|Should -BeTrue;$state.Source|Should -BeExactly 'LegacyReadOnly'
        Should -Invoke -ModuleName SecretsTier ConvertFrom-EncryptedString -Times 0 -Exactly
        Assert-ConsumerLegacyUnchanged
    }
    # Protects: partial preferred pair never falls back to usable legacy.
    # Detects: An incomplete preferred pair must fail without returning stale legacy values or changing the legacy files.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-SecretsFromCache, Initialize-CredentialPrivateStorage
    It 'partial preferred pair never falls back to usable legacy' {
        New-ConsumerLegacyPair
        $selected=& $module {Initialize-CredentialPrivateStorage $script:CacheSelection}
        New-ConsumerPrivateText (Join-Path $selected 'secrets-cache.enc') 'partial-synthetic'
        {& $module {Get-SecretsFromCache}}|Should -Throw '*pair is incomplete*'
        Assert-ConsumerLegacyUnchanged
    }
    # Protects: invalid authenticated ciphertext is refused without private content in errors.
    # Detects: Hash-matching ciphertext that fails DPAPI authentication must be refused with sanitized errors and without provisioning preferred storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-SecretsFromCache
    It 'invalid authenticated ciphertext is refused without private content in errors' {
        New-ConsumerLegacyPair
        [IO.File]::WriteAllText($legacyCache,'private-sentinel-not-dpapi')
        $meta=Get-Content $legacyManifest -Raw|ConvertFrom-Json -AsHashtable
        $meta.cacheSha256=(Get-FileHash $legacyCache).Hash
        [IO.File]::WriteAllText($legacyManifest,($meta|ConvertTo-Json -Compress))
        try{& $module {Get-SecretsFromCache};throw 'Expected rejection.'}catch{$_.Exception.Message|Should -Match 'private content is withheld';$_.Exception.Message|Should -Not -Match 'private-sentinel'}
        Test-Path (Join-Path $env:USERPROFILE '.pcai-credential-state')|Should -BeFalse
    }
    # Protects: valid DPAPI with an invalid payload schema fails before returning values.
    # Detects: Successful DPAPI decryption must not admit a nonobject secret payload or hydrate the process environment.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-SecretsFromCache
    It 'valid DPAPI with an invalid payload schema fails before returning values' {
        New-ConsumerLegacyPair
        $plain=[Text.Encoding]::UTF8.GetBytes('["synthetic"]')
        try{$cipher=[Convert]::ToBase64String([Security.Cryptography.ProtectedData]::Protect($plain,$null,[Security.Cryptography.DataProtectionScope]::CurrentUser))}finally{[Array]::Clear($plain,0,$plain.Length)}
        [IO.File]::WriteAllText($legacyCache,$cipher)
        $meta=Get-Content $legacyManifest -Raw|ConvertFrom-Json -AsHashtable;$meta.cacheSha256=(Get-FileHash $legacyCache).Hash
        [IO.File]::WriteAllText($legacyManifest,($meta|ConvertTo-Json -Compress))
        {& $module {Get-SecretsFromCache}}|Should -Throw '*private content is withheld*'
        $env:PCAI_CONSUMER_SYNTHETIC|Should -BeNullOrEmpty
    }
    # Protects: foreign-read legacy file grant is rejected without ACL repair.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-SecretsFromCache
    It 'foreign-read legacy file grant is rejected without ACL repair' {
        New-ConsumerLegacyPair
        $acl=Get-Acl $legacyCache;$acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]::Read,[Security.AccessControl.AccessControlType]::Allow));Set-Acl $legacyCache $acl
        $sddl=(Get-Acl $legacyCache).Sddl
        {& $module {Get-SecretsFromCache}}|Should -Throw '*private content is withheld*'
        (Get-Acl $legacyCache).Sddl|Should -BeExactly $sddl
    }
    # Protects: retained identity refuses same-byte namespace substitution after real DPAPI decrypt.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-SecretsFromCache, New-CredentialPrivateStage
    It 'retained identity refuses same-byte namespace substitution after real DPAPI decrypt' {
        New-ConsumerLegacyPair
        Mock -ModuleName SecretsTier ConvertFrom-EncryptedString {
            $cipher=[Convert]::FromBase64String($EncryptedText)
            $bytes=[Security.Cryptography.ProtectedData]::Unprotect($cipher,$null,[Security.Cryptography.DataProtectionScope]::CurrentUser)
            try{$json=[Text.Encoding]::UTF8.GetString($bytes)}finally{[Array]::Clear($bytes,0,$bytes.Length)}
            $path=& (Get-Module SecretsTier) {$script:LegacyCacheFile}
            [IO.File]::Move($path,($path+'.retired'))
            $replacement=New-CredentialPrivateStage $path ([Text.Encoding]::UTF8.GetBytes($EncryptedText));$replacement.Dispose()
            return $json
        }
        {& $module {Get-SecretsFromCache}}|Should -Throw '*private content is withheld*'
        Test-Path ($legacyCache+'.retired')|Should -BeTrue
        (Get-FileHash $legacyCache).Hash|Should -BeExactly (Get-FileHash ($legacyCache+'.retired')).Hash
        $env:PCAI_CONSUMER_SYNTHETIC|Should -BeNullOrEmpty
    }
    # Protects: invalid explicit cache root is refused during passive import.
    # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1 and SecretBackendUtilities.ps1: passive import/custody contract
    It 'invalid explicit cache root is refused during passive import' {
        Remove-Module $module -Force;$script:module=$null
        {Import-Module (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretsTier.psm1') -ArgumentList @('fixture-machine',(Join-Path $env:USERPROFILE '.machine'),$env:TEMP) -Force -ErrorAction Stop}|Should -Throw
        @(Get-ChildItem $env:TEMP -Force).Count|Should -Be 0
        Test-Path (Join-Path $env:USERPROFILE '.machine')|Should -BeFalse
    }
    # Protects: explicit machine root preserves its selected cache path and ignores legacy fallback.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-SecretsFromCache, Save-SecretsToCache
    It 'explicit machine root preserves its selected cache path and ignores legacy fallback' {
        New-ConsumerLegacyPair
        Remove-Module $module -Force
        $explicit=Join-Path $env:USERPROFILE 'explicit-machine'
        $script:module=Import-Module (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretsTier.psm1') -ArgumentList @('fixture-machine',$explicit) -Force -PassThru -ErrorAction Stop
        (& $module {Get-SecretsFromCache})|Should -BeNullOrEmpty
        & $module {Save-SecretsToCache @{fixture='explicit-synthetic'} -Confirm:$false}
        Test-Path (Join-Path $explicit 'secrets-cache.enc')|Should -BeTrue
        Assert-ConsumerLegacyUnchanged
    }
}
}

}

Describe 'Read-only root envelope and strictly private selected storage' {
    # Protects the frontend contract that a traversable envelope is not a secret
    # leaf. Uses real synthetic NTFS ACLs; no production path or provider is used.
    BeforeAll {
        . (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretBackendUtilities.ps1')
        function Add-EnvelopeFixtureGrant {
            param([string]$Path,[Security.AccessControl.FileSystemRights]$Rights)
            $acl=Get-Acl -LiteralPath $Path
            $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),$Rights,[Security.AccessControl.AccessControlType]::Allow))
            Set-Acl -LiteralPath $Path -AclObject $acl
        }
        function Get-EnvelopeFixtureSelection {
            param([string]$Purpose)
            $directory=if($Purpose -eq 'Bootstrap'){Join-Path $script:envelope 'bootstrap'}elseif($Purpose -eq 'Cache'){
                Join-Path (Join-Path $script:envelope 'cache') (Get-CredentialBytesHash ([Text.Encoding]::UTF8.GetBytes('FIXTURE-MACHINE')))
            }else{$script:envelope}
            [pscustomobject]@{Root=$script:envelope;Directory=$directory;Purpose=$Purpose}
        }
        function Invoke-EnvelopeFixtureFrontend {
            param([string]$Operation,[string]$Purpose)
            if($Operation -eq 'Resolve'){
                if($Purpose -eq 'Explicit'){Resolve-CredentialPrivateStorage -RootOverride $script:envelope -Purpose Explicit}
                else{Resolve-CredentialPrivateStorage -Purpose $Purpose -MachineName 'fixture-machine'}
            }else{Initialize-CredentialPrivateStorage (Get-EnvelopeFixtureSelection $Purpose)}
        }
    }
    BeforeEach {
        $script:envelopeProfileBefore=$env:USERPROFILE
        $script:envelopeCase=Join-Path $script:FixtureDrive ([guid]::NewGuid().ToString('N'))
        Assert-CredentialPrivateDirectory $script:envelopeCase
        $env:USERPROFILE=Join-Path $script:envelopeCase 'user'
        Assert-CredentialPrivateDirectory $env:USERPROFILE
        $script:envelope=Join-Path $env:USERPROFILE '.pcai-credential-state'
        Assert-CredentialPrivateDirectory $script:envelope
        foreach($dir in @((Join-Path $script:envelope 'bootstrap'),(Join-Path $script:envelope 'cache'),(Get-EnvelopeFixtureSelection Cache).Directory)){
            Assert-CredentialPrivateDirectory $dir
        }
    }
    AfterEach {$env:USERPROFILE=$script:envelopeProfileBefore}

    # Protects: admits a read-only envelope for <Operation> <Purpose> without changing any ACL.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretBackendUtilities.ps1: Resolve-CredentialPrivateStorage,Initialize-CredentialPrivateStorage
    It 'admits a read-only envelope for <Operation> <Purpose> without changing any ACL' -TestCases @(
        @{Operation='Resolve';Purpose='Bootstrap'},@{Operation='Resolve';Purpose='Cache'},
        @{Operation='Initialize';Purpose='Bootstrap'},@{Operation='Initialize';Purpose='Cache'}
    ) {
        param($Operation,$Purpose)
        Add-EnvelopeFixtureGrant $script:envelope ReadAndExecute
        $selection=Get-EnvelopeFixtureSelection $Purpose
        $before=@($script:envelope,$selection.Directory|ForEach-Object{[pscustomobject]@{Path=$_;Sddl=(Get-Acl -LiteralPath $_).Sddl}})
        $result=Invoke-EnvelopeFixtureFrontend $Operation $Purpose
        $actual=if($Operation -eq 'Resolve'){$result.Directory}else{$result}
        $actual|Should -BeExactly $selection.Directory
        foreach($item in $before){(Get-Acl -LiteralPath $item.Path).Sddl|Should -BeExactly $item.Sddl}
        Assert-CredentialPrivateDirectory $selection.Directory -ExistingOnly
    }

    # Protects: provisions missing private <Purpose> children under an unchanged read-only envelope.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Assert-CredentialPrivateDirectory, Initialize-CredentialPrivateStorage, Resolve-CredentialPrivateStorage
    It 'provisions missing private <Purpose> children under an unchanged read-only envelope' -TestCases @(@{Purpose='Bootstrap'},@{Purpose='Cache'}) {
        param($Purpose)
        $selection=Get-EnvelopeFixtureSelection $Purpose
        Remove-Item -LiteralPath $selection.Directory
        if($Purpose -eq 'Cache'){Remove-Item -LiteralPath (Join-Path $script:envelope 'cache')}
        Add-EnvelopeFixtureGrant $script:envelope ReadAndExecute
        $sddl=(Get-Acl -LiteralPath $script:envelope).Sddl
        $admitted=Resolve-CredentialPrivateStorage -Purpose $Purpose -MachineName 'fixture-machine'
        Test-Path -LiteralPath $selection.Directory|Should -BeFalse
        Initialize-CredentialPrivateStorage $admitted|Should -BeExactly $selection.Directory
        Assert-CredentialPrivateDirectory $selection.Directory -ExistingOnly
        if($Purpose -eq 'Cache'){Assert-CredentialPrivateDirectory (Join-Path $script:envelope 'cache') -ExistingOnly}
        (Get-Acl -LiteralPath $script:envelope).Sddl|Should -BeExactly $sddl
    }

    # Protects: refuses a read grant on the Explicit root selected as the secret leaf during <Operation>.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1 and SecretBackendUtilities.ps1: passive import/custody contract
    It 'refuses a read grant on the Explicit root selected as the secret leaf during <Operation>' -TestCases @(@{Operation='Resolve'},@{Operation='Initialize'}) {
        param($Operation)
        Add-EnvelopeFixtureGrant $script:envelope ReadAndExecute
        $sddl=(Get-Acl -LiteralPath $script:envelope).Sddl
        {Invoke-EnvelopeFixtureFrontend $Operation Explicit}|Should -Throw '*unsupported principal*'
        (Get-Acl -LiteralPath $script:envelope).Sddl|Should -BeExactly $sddl
    }

    # Protects: refuses a foreign-readable selected <Purpose> leaf during <Operation> without ACL repair.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1 and SecretBackendUtilities.ps1: passive import/custody contract
    It 'refuses a foreign-readable selected <Purpose> leaf during <Operation> without ACL repair' -TestCases @(
        @{Operation='Resolve';Purpose='Bootstrap'},@{Operation='Resolve';Purpose='Cache'},
        @{Operation='Initialize';Purpose='Bootstrap'},@{Operation='Initialize';Purpose='Cache'}
    ) {
        param($Operation,$Purpose)
        $selection=Get-EnvelopeFixtureSelection $Purpose
        Add-EnvelopeFixtureGrant $selection.Directory ReadAndExecute
        $sddl=(Get-Acl -LiteralPath $selection.Directory).Sddl
        {Invoke-EnvelopeFixtureFrontend $Operation $Purpose}|Should -Throw '*unsupported principal*'
        (Get-Acl -LiteralPath $selection.Directory).Sddl|Should -BeExactly $sddl
    }

    # Protects: refuses foreign ancestor deletion authority during <Operation> <Purpose> despite protected children.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1 and SecretBackendUtilities.ps1: passive import/custody contract
    It 'refuses foreign ancestor deletion authority during <Operation> <Purpose> despite protected children' -TestCases @(
        @{Operation='Resolve';Purpose='Bootstrap'},@{Operation='Resolve';Purpose='Cache'},
        @{Operation='Initialize';Purpose='Bootstrap'},@{Operation='Initialize';Purpose='Cache'}
    ) {
        param($Operation,$Purpose)
        Add-EnvelopeFixtureGrant $script:envelopeCase DeleteSubdirectoriesAndFiles
        $sddl=(Get-Acl -LiteralPath $script:envelopeCase).Sddl
        {Invoke-EnvelopeFixtureFrontend $Operation $Purpose}|Should -Throw '*namespace mutation authority refused*'
        (Get-Acl -LiteralPath $script:envelopeCase).Sddl|Should -BeExactly $sddl
    }
}


Describe 'Synthetic cloud, backup and environment orchestration' {
    # Protects provider selection and serialization boundaries in SecretsTier.
    # All cloud/native commands and persistent environment writes are mocked;
    # only explicit synthetic process data can cross the tested boundaries.
    BeforeAll {
        $script:providerModule=Import-Module (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretsTier.psm1') -ArgumentList @('fixture-machine') -Force -PassThru -ErrorAction Stop
        $script:providerTokenBefore=$env:BWS_ACCESS_TOKEN
        function Initialize-SyntheticBackupBackend {
            $script:backupItems=@{};$script:backupNextId=0
            $script:backupSecrets=@{alpha='synthetic-roundtrip'}
            $script:backupFailSecondPart=$false;$script:backupRecovered=$null
            Mock -ModuleName SecretsTier Test-BwSession {$true}
            Mock -ModuleName SecretsTier Get-CacheStatus {[pscustomobject]@{IsExpired=$false}}
            Mock -ModuleName SecretsTier Get-BwsSecretsList {$script:backupSecrets}
            Mock -ModuleName SecretsTier Get-MachineIdentityInternal {@{machineId='synthetic-id'}}
            Mock -ModuleName SecretsTier Save-SecretsToCache {$script:backupRecovered=$Secrets}
            Mock -ModuleName SecretsTier Invoke-MachineBitwardenCommand {
                switch($Arguments[0]){
                    list {return @($script:backupItems.Values)|ConvertTo-Json -Depth 20 -AsArray -Compress}
                    get {return $script:backupItems[$Arguments[2]]|ConvertTo-Json -Depth 20 -Compress}
                    delete {[void]$script:backupItems.Remove($Arguments[2]);return ''}
                    sync {return ''}
                    default {throw 'Unexpected synthetic vault operation.'}
                }
            }
            Mock -ModuleName SecretsTier Invoke-BitwardenCli {
                if($Arguments[0] -eq 'delete'){
                    if($Arguments.Count -ne 3 -or $Arguments[1] -ne 'item'){throw 'Unexpected synthetic delete operation.'}
                    [void]$script:backupItems.Remove($Arguments[2])
                    return [pscustomobject]@{Success=$true}
                }
                $item=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($StandardInput))|ConvertFrom-Json -AsHashtable
                if($script:backupFailSecondPart -and ($item.name -match '(?: #2$|\[pcai-v2:[a-f0-9]{32}:2\]$)')){return [pscustomobject]@{Success=$false}}
                if($Arguments[0] -eq 'create'){$script:backupNextId++;$item.id='synthetic-'+$script:backupNextId}
                $script:backupItems[$item.id]=[pscustomobject]$item
                [pscustomobject]@{Success=$true}
            }
        }
        function Set-SyntheticBackupPayload {
            param([byte[]]$Bytes)
            $canonical=@($script:backupItems.Values|Where-Object name -EQ '[MACHINE] fixture-machine Secrets')[0]
            $manifest=$canonical.notes.Substring($canonical.notes.IndexOf([char]10)+1)|ConvertFrom-Json -AsHashtable
            $encoded=[Convert]::ToBase64String($Bytes)
            $encoded.Length|Should -BeLessOrEqual 4500
            foreach($id in @($script:backupItems.Keys)){
                if($script:backupItems[$id].name -ne $canonical.name){[void]$script:backupItems.Remove($id)}
            }
            $digest=[Security.Cryptography.SHA256]::HashData($Bytes)
            $manifest.sha256=[Convert]::ToHexString($digest).ToLowerInvariant()
            $manifest.rawLength=$Bytes.Length;$manifest.encodedLength=$encoded.Length;$manifest.partCount=1
            $name=$canonical.name+' [pcai-v2:'+$manifest.generation+':1]'
            $header='#pcai-secrets-v2 generation='+$manifest.generation+' part=1 parts=1'+[char]10
            $script:backupItems['synthetic-payload']=[pscustomobject]@{id='synthetic-payload';name=$name;type=2;notes=($header+$encoded)}
            $canonical.notes='#pcai-secrets-backup-v2'+[char]10+($manifest|ConvertTo-Json -Compress)
        }
    }
    BeforeEach {
        $script:providerComputerNameBefore=$env:COMPUTERNAME
        $env:COMPUTERNAME='fixture-machine'
        $env:BWS_ACCESS_TOKEN=$null
        Mock -ModuleName SecretsTier Invoke-SecretBackendProcess {throw 'Unmocked external process forbidden.'}
        Mock -ModuleName SecretsTier Invoke-MachineBitwardenCommand {throw 'Unmocked Bitwarden command forbidden.'}
        Mock -ModuleName SecretsTier Invoke-BitwardenCli {throw 'Unmocked Bitwarden boundary forbidden.'}
        Mock -ModuleName SecretsTier Test-InternetConnection {$false}
        Mock -ModuleName SecretsTier Save-SecretsToCache {throw 'Unmocked cache publication forbidden.'}
        Mock -ModuleName SecretsTier Set-SecretsAsEnvironment {throw 'Unmocked environment publication forbidden.'}
        Mock -ModuleName SecretsTier Set-InitializedSecretsEnvironment {throw 'Unmocked environment hydration forbidden.'}
        Mock -ModuleName SecretsTier Resolve-BwsTokenFromBackends {$null}
    }
    AfterEach {$env:BWS_ACCESS_TOKEN=$null;$env:COMPUTERNAME=$script:providerComputerNameBefore}
    AfterAll {Remove-Module $script:providerModule -Force;$env:BWS_ACCESS_TOKEN=$script:providerTokenBefore}

    # Protects: explicit-machine backup identity through actual serializer/import.
    # Detects: a note selected for another machine cannot roundtrip its own data.
    # Needs: inert in-memory vault and synthetic values only.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1::Export-SecretsToBwVault,Import-SecretsFromBwVault
    It 'roundtrips a backup using the explicit machine selector instead of the current host' {
        Initialize-SyntheticBackupBackend
        $env:COMPUTERNAME='synthetic-current-host'
        (& $script:providerModule {$MachineName})|Should -BeExactly 'fixture-machine'
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $canonical=@($script:backupItems.Values|Where-Object name -eq '[MACHINE] fixture-machine Secrets')[0]
        $body=($canonical.notes -split [string][char]10,2)[1]|ConvertFrom-Json
        $body.machine|Should -BeExactly 'fixture-machine'
        $r=& $script:providerModule {Import-SecretsFromBwVault -Confirm:$false -ErrorAction Stop}
        $r.alpha|Should -BeExactly 'synthetic-roundtrip'
        $script:backupRecovered.alpha|Should -BeExactly 'synthetic-roundtrip'
    }
    # Protects: the previous committed snapshot through partial same-count writes.
    # Detects: a new first part combined with an old suffix silently changes a secret.
    # Needs: inert in-memory vault, forced second-part write failure, equal-size data.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1::Export-SecretsToBwVault,Import-SecretsFromBwVault
    It 'preserves the prior complete backup after an interrupted same-count replacement' {
        Initialize-SyntheticBackupBackend
        $script:backupSecrets=@{alpha=('A'*6000)}
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $script:backupSecrets=@{alpha=('B'*6000)};$script:backupFailSecondPart=$true
        try{$replacement=& $script:providerModule {Export-SecretsToBwVault -Confirm:$false -ErrorAction Stop}}catch{$replacement=$false}
        $replacement|Should -BeFalse
        $r=& $script:providerModule {Import-SecretsFromBwVault -Confirm:$false -ErrorAction Stop}
        $r.alpha|Should -BeExactly ('A'*6000)
    }
    # Protects: the operator's documented forced refresh despite valid cache.
    # Detects: cache fast-path masks cloud credential rotation during explicit sync.
    # Needs: synthetic valid cache and mocked cloud/cache/environment boundaries.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1::Sync-MachineSecrets,Sync-SecretsFromBws
    It 'forces cloud refresh when the operator explicitly synchronizes a valid cache' {
        Mock -ModuleName SecretsTier Get-CacheStatus {[pscustomobject]@{IsValid=$true}}
        Mock -ModuleName SecretsTier Test-InternetConnection {$true}
        Mock -ModuleName SecretsTier Get-BwsSecretsList {@{alpha='synthetic-rotated'}}
        Mock -ModuleName SecretsTier Save-SecretsToCache {}
        Mock -ModuleName SecretsTier Get-SecretsFromCache {@{alpha='synthetic-rotated'}}
        Mock -ModuleName SecretsTier Set-SecretsAsEnvironment {}
        & $script:providerModule {Sync-MachineSecrets -Confirm:$false}
        Should -Invoke -ModuleName SecretsTier Get-BwsSecretsList -Times 1 -Exactly
        Should -Invoke -ModuleName SecretsTier Save-SecretsToCache -Times 1 -Exactly -ParameterFilter {$Secrets.alpha -eq 'synthetic-rotated'}
    }


    # Protects: the ASCII ciphertext margin and UTF-8 integrity of actual backups.
    # Detects: Unicode byte expansion or surrogate splitting corrupts a roundtrip.
    # Needs: CJK/emoji synthetic data and the in-memory vault; no provider.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1::Export-SecretsToBwVault,Import-SecretsFromBwVault
    It 'roundtrips CJK and emoji through bounded ASCII parts and explicit encoding metadata' {
        Initialize-SyntheticBackupBackend
        $script:backupSecrets=@{alpha=('漢字😀'*1200)}
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $canonical=@($script:backupItems.Values|Where-Object name -eq '[MACHINE] fixture-machine Secrets')[0]
        $canonical.notes|Should -Match '^#pcai-secrets-backup-v2\n'
        [Text.Encoding]::UTF8.GetByteCount($canonical.notes)|Should -BeLessOrEqual 2048
        $manifest=($canonical.notes -split [string][char]10,2)[1]|ConvertFrom-Json
        $manifest.encoding|Should -BeExactly 'utf8-base64'
        $parts=@($script:backupItems.Values|Where-Object name -ne $canonical.name)
        $parts.Count|Should -BeGreaterThan 1
        foreach($part in $parts){
            $part.notes|Should -Not -Match '[^\x00-\x7F]'
            $split=$part.notes -split [string][char]10,2
            [Text.Encoding]::ASCII.GetByteCount($split[0])|Should -BeLessOrEqual 128
            $split[1].Length|Should -BeLessOrEqual 4500
        }
        $r=& $script:providerModule {Import-SecretsFromBwVault -Confirm:$false -ErrorAction Stop}
        $r.alpha|Should -BeExactly ('漢字😀'*1200)
    }
    # Protects: manifest identity, encoding, bounds and content-integrity admission.
    # Detects: edited metadata is accepted before cache publication or hydration.
    # Needs: each named tamper case over an actual synthetic exported snapshot.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1::Read-BwBackupSnapshotInternal,Import-SecretsFromBwVault
    It 'refuses tampered v2 <Field> metadata before cache or environment publication' -TestCases @(
        @{Field='version'},@{Field='machine'},@{Field='encoding'},@{Field='partCount'},
        @{Field='encodedLength'},@{Field='rawLength'},@{Field='sha256'},@{Field='generation'}
    ) {
        param($Field)
        Initialize-SyntheticBackupBackend
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $canonical=@($script:backupItems.Values|Where-Object name -eq '[MACHINE] fixture-machine Secrets')[0]
        $manifest=($canonical.notes -split [string][char]10,2)[1]|ConvertFrom-Json -AsHashtable
        switch($Field){
            version {$manifest.version=1}
            machine {$manifest.machine='wrong-machine'}
            encoding {$manifest.encoding='other'}
            partCount {$manifest.partCount=1025}
            encodedLength {$manifest.encodedLength=[int]$manifest.encodedLength+4}
            rawLength {$manifest.rawLength=[int]$manifest.rawLength+1}
            sha256 {$manifest.sha256='0'*64}
            generation {$manifest.generation='f'*32}
        }
        $canonical.notes='#pcai-secrets-backup-v2'+[char]10+($manifest|ConvertTo-Json -Compress)
        $script:backupRecovered=$null
        {& $script:providerModule {Import-SecretsFromBwVault -SetEnvironment -Confirm:$false -ErrorAction Stop}}|Should -Throw
        $script:backupRecovered|Should -BeNullOrEmpty
        Should -Invoke -ModuleName SecretsTier Set-SecretsAsEnvironment -Times 0 -Exactly
    }
    # Protects: byte-level content binding even when altered JSON remains valid.
    # Detects: valid base64 containing a different credential bypasses the digest.
    # Needs: actual multi-part synthetic snapshot and a single changed ASCII value.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1::Read-BwBackupSnapshotInternal
    It 'rejects a valid-base64 valid-JSON credential change with an unchanged manifest' {
        Initialize-SyntheticBackupBackend
        $script:backupSecrets=@{alpha=('A'*6000)}
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $parts=@($script:backupItems.Values|Where-Object name -match '\[pcai-v2:[a-f0-9]{32}:\d+\]$'|Sort-Object {[int]([regex]::Match($_.name,':(\d+)\]$').Groups[1].Value)})
        $encoded=($parts|ForEach-Object{($_.notes -split [string][char]10,2)[1]}) -join ''
        $bytes=[Convert]::FromBase64String($encoded)
        $index=[Array]::IndexOf($bytes,[byte][char]'A');$index|Should -BeGreaterOrEqual 0
        $bytes[$index]=[byte][char]'B';$tampered=[Convert]::ToBase64String($bytes)
        for($i=0;$i -lt $parts.Count;$i++){
            $header=($parts[$i].notes -split [string][char]10,2)[0]
            $parts[$i].notes=$header+[char]10+$tampered.Substring($i*4500,[Math]::Min(4500,$tampered.Length-$i*4500))
        }
        {& $script:providerModule {Import-SecretsFromBwVault -Confirm:$false -ErrorAction Stop}}|Should -Throw
        $script:backupRecovered|Should -BeNullOrEmpty
    }
    # Protects: exact prior-generation retirement after verified replacement.
    # Detects: obsolete committed parts accumulate or unrelated notes are deleted.
    # Needs: two actual synthetic exports plus unrelated and lookalike notes.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1::Export-SecretsToBwVault,Remove-BwNoteTextInternal
    It 'retires only the prior committed generation after full replacement validation' {
        Initialize-SyntheticBackupBackend
        $script:backupSecrets=@{alpha=('A'*6000)}
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $oldNames=@($script:backupItems.Values|Where-Object name -match '\[pcai-v2:[a-f0-9]{32}:\d+\]$'|ForEach-Object name)
        $oldNames.Count|Should -BeGreaterThan 0
        $script:backupItems['unrelated']=[pscustomobject]@{id='unrelated';type=2;name='unrelated-note';notes='synthetic-unrelated'}
        $script:backupItems['lookalike']=[pscustomobject]@{id='lookalike';type=2;name=($oldNames[0]+' extra');notes='synthetic-lookalike'}
        $script:backupSecrets=@{alpha=('B'*6000)}
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        foreach($old in $oldNames){@($script:backupItems.Values|Where-Object name -eq $old).Count|Should -Be 0}
        $script:backupItems.unrelated.notes|Should -BeExactly 'synthetic-unrelated'
        $script:backupItems.lookalike.notes|Should -BeExactly 'synthetic-lookalike'
        (& $script:providerModule {Import-SecretsFromBwVault -Confirm:$false -ErrorAction Stop}).alpha|Should -BeExactly ('B'*6000)
    }
    # Protects: cleanup of only this failed attempt's uniquely owned staging.
    # Detects: failure leaves redundant parts or removes the committed snapshot.
    # Needs: unique generation, synthetic second-part failure, existing snapshot.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1::Export-SecretsToBwVault,Remove-BwNoteTextInternal
    It 'cleans its own interrupted staging generation while preserving all prior items' {
        Initialize-SyntheticBackupBackend
        $script:backupSecrets=@{alpha=('A'*6000)}
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $before=@($script:backupItems.Values|ForEach-Object{[pscustomobject]@{Id=$_.id;Name=$_.name;Text=$_.notes}})
        $script:backupSecrets=@{alpha=('B'*6000)};$script:backupFailSecondPart=$true
        try{$ok=& $script:providerModule {Export-SecretsToBwVault -Confirm:$false -ErrorAction Stop}}catch{$ok=$false}
        $ok|Should -BeFalse;$script:backupItems.Count|Should -Be $before.Count
        foreach($item in $before){$script:backupItems[$item.Id].notes|Should -BeExactly $item.Text}
    }
    # Protects: concurrent readers during exact prior-generation retirement.
    # Detects: one demonstrable canonical change strands an otherwise valid read.
    # Needs: stale canonical once, current valid generation, inert synthetic vault.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1::Read-BwBackupCommittedInternal
    It 'retries once when a retired-part read demonstrates a newer canonical generation' {
        Initialize-SyntheticBackupBackend
        $script:backupSecrets=@{alpha=('A'*6000)}
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $script:backupOldCanonical=@($script:backupItems.Values|Where-Object name -eq '[MACHINE] fixture-machine Secrets')[0]|ConvertTo-Json -Depth 20 -Compress
        $script:backupSecrets=@{alpha=('B'*6000)}
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $script:backupServedOld=$false;$script:backupCanonicalReads=0
        Mock -ModuleName SecretsTier Invoke-MachineBitwardenCommand {
            if($Arguments[0] -ne 'list'){throw 'Unexpected write during import.'}
            if($Arguments[3] -eq '[MACHINE] fixture-machine Secrets'){
                $script:backupCanonicalReads++
                if(-not $script:backupServedOld){$script:backupServedOld=$true;return '['+$script:backupOldCanonical+']'}
            }
            @($script:backupItems.Values)|ConvertTo-Json -Depth 20 -AsArray -Compress
        }
        $r=& $script:providerModule {Import-SecretsFromBwVault -Confirm:$false -ErrorAction Stop}
        $r.alpha|Should -BeExactly ('B'*6000)
        $script:backupCanonicalReads|Should -BeGreaterOrEqual 2
        $script:backupCanonicalReads|Should -BeLessOrEqual 3
    }


    # Protects: strict manifest field uniqueness, shape and bounded transport metadata.
    # Detects: permissive JSON coercion or duplicate keys could reinterpret a committed snapshot.
    # Needs: Actual in-memory serializer/import; malformed synthetic manifest only.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Get-BwBackupManifestInternal
    It 'refuses malformed v2 manifest <Mode> without publishing values' -TestCases @(
        @{Mode='Duplicate'},@{Mode='Unknown'},@{Mode='StringInteger'},@{Mode='Fraction'},
        @{Mode='NullString'},@{Mode='UpperGeneration'},@{Mode='Oversize'},@{Mode='Array'}
    ) {
        param($Mode)
        Initialize-SyntheticBackupBackend
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $canonical=@($script:backupItems.Values|Where-Object name -EQ '[MACHINE] fixture-machine Secrets')[0]
        $body=$canonical.notes.Substring($canonical.notes.IndexOf([char]10)+1)
        $manifest=$body|ConvertFrom-Json -AsHashtable
        switch($Mode){
            Duplicate {$body=$body.Substring(0,$body.Length-1)+',"version":2}'}
            Unknown {$manifest.extra='unexpected';$body=$manifest|ConvertTo-Json -Compress}
            StringInteger {$manifest.partCount='1';$body=$manifest|ConvertTo-Json -Compress}
            Fraction {$manifest.partCount=1.5;$body=$manifest|ConvertTo-Json -Compress}
            NullString {$manifest.machine=$null;$body=$manifest|ConvertTo-Json -Compress}
            UpperGeneration {$manifest.generation='A'*32;$body=$manifest|ConvertTo-Json -Compress}
            Oversize {$body=$body+(' '*2100)}
            Array {$body='[]'}
        }
        $canonical.notes='#pcai-secrets-backup-v2'+[char]10+$body
        {& $script:providerModule {Import-SecretsFromBwVault -SetEnvironment -Confirm:$false -ErrorAction Stop}}|Should -Throw
        Should -Invoke -ModuleName SecretsTier Save-SecretsToCache -Times 0 -Exactly
        Should -Invoke -ModuleName SecretsTier Set-SecretsAsEnvironment -Times 0 -Exactly
    }
    # Protects: complete payload decoding and schema validation before private publication.
    # Detects: correct hashes alone cannot legitimize invalid UTF-8, duplicate keys or mismatched machine/count.
    # Needs: Hash-consistent synthetic wire payloads; no real provider.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Read-BwBackupSnapshotInternal,ConvertFrom-BwBackupElementInternal
    It 'rejects a hash-consistent but invalid snapshot <Mode>' -TestCases @(
        @{Mode='Utf8'},@{Mode='Duplicate'},@{Mode='WrongMachine'},@{Mode='WrongCount'},
        @{Mode='NonObject'},@{Mode='NonSecretMap'},@{Mode='CaseCollision'},@{Mode='TooDeep'}
    ){
        param($Mode)
        Initialize-SyntheticBackupBackend
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $json='{"machine":"fixture-machine","secretCount":1,"secrets":{"alpha":"synthetic"}}'
        switch($Mode){
            Utf8 {}
            Duplicate {$json='{"machine":"fixture-machine","secretCount":1,"secrets":{"alpha":"first","alpha":"second"}}'}
            WrongMachine {$json=$json.Replace('fixture-machine','other-machine')}
            WrongCount {$json=$json.Replace('"secretCount":1','"secretCount":2')}
            NonObject {$json='[]'}
            NonSecretMap {$json=$json.Replace('{"alpha":"synthetic"}','["synthetic"]')}
            CaseCollision {$json='{"machine":"fixture-machine","secretCount":2,"secrets":{"alpha":"first","ALPHA":"second"}}'}
            TooDeep {$json='{"machine":"fixture-machine","secretCount":1,"secrets":{"alpha":'+('{"a":'*22)+'"synthetic"'+('}'*22)+'}}'}
        }
        $bytes=if($Mode -eq 'Utf8'){[byte[]]@(0xC3,0x28)}else{[Text.Encoding]::UTF8.GetBytes($json)}
        Set-SyntheticBackupPayload $bytes
        {& $script:providerModule {Import-SecretsFromBwVault -SetEnvironment -Confirm:$false -ErrorAction Stop}}|Should -Throw
        Should -Invoke -ModuleName SecretsTier Save-SecretsToCache -Times 0 -Exactly
        Should -Invoke -ModuleName SecretsTier Set-SecretsAsEnvironment -Times 0 -Exactly
    }
    # Protects: exact generation/index/count, chunk bounds and strict base64 encoding.
    # Detects: invalid or noncanonical wire data must not reach cache even when note names match.
    # Needs: Actual exported synthetic note with one deliberately malformed part.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Read-BwBackupSnapshotInternal
    It 'refuses malformed v2 part <Mode> before publication' -TestCases @(
        @{Mode='Header'},@{Mode='Short'},@{Mode='Long'},@{Mode='Base64'},@{Mode='Missing'}
    ){
        param($Mode)
        Initialize-SyntheticBackupBackend
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $part=@($script:backupItems.Values|Where-Object name -NE '[MACHINE] fixture-machine Secrets')[0]
        switch($Mode){
            Header {$part.notes=$part.notes.Replace(' part=1 ',' part=2 ')}
            Short {$part.notes=$part.notes.Substring(0,$part.notes.Length-1)}
            Long {$part.notes+='A'}
            Base64 {$at=$part.notes.IndexOf([char]10)+1;$part.notes=$part.notes.Substring(0,$at)+'!'+$part.notes.Substring($at+1)}
            Missing {[void]$script:backupItems.Remove($part.id)}
        }
        {& $script:providerModule {Import-SecretsFromBwVault -SetEnvironment -Confirm:$false -ErrorAction Stop}}|Should -Throw
        Should -Invoke -ModuleName SecretsTier Save-SecretsToCache -Times 0 -Exactly
        Should -Invoke -ModuleName SecretsTier Set-SecretsAsEnvironment -Times 0 -Exactly
    }
    # Protects: exact identity and content custody at cloud cleanup boundaries.
    # Detects: ambiguity, changed identities or changed bodies must preserve every candidate.
    # Needs: Synthetic note lookup/delete interface, no real provider.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Remove-BwNoteTextInternal
    It 'preserves notes when guarded cleanup sees <Mode>' -TestCases @(
        @{Mode='Ambiguous'},@{Mode='WrongId'},@{Mode='WrongName'},@{Mode='WrongType'},@{Mode='ChangedText'},@{Mode='DeleteFailure'}
    ){
        param($Mode)
        $script:cleanupMode=$Mode
        Mock -ModuleName SecretsTier Invoke-MachineBitwardenCommand {
            if($Arguments[0] -eq 'list'){
                $rows=@([pscustomobject]@{id='synthetic-one';name='synthetic-part';type=2;notes='expected'})
                if($script:cleanupMode -eq 'Ambiguous'){$rows+=[pscustomobject]@{id='synthetic-two';name='synthetic-part';type=2;notes='expected'}}
                return $rows|ConvertTo-Json -AsArray -Compress
            }
            $row=@{id='synthetic-one';name='synthetic-part';type=2;notes='expected'}
            switch($script:cleanupMode){
                WrongId {$row.id='different'} WrongName {$row.name='different'}
                WrongType {$row.type=1} ChangedText {$row.notes='changed'}
            }
            $row|ConvertTo-Json -Compress
        }
        Mock -ModuleName SecretsTier Invoke-BitwardenCli {[pscustomobject]@{Success=$false}}
        (& $script:providerModule {Remove-BwNoteTextInternal -Name synthetic-part -ExpectedText expected -Confirm:$false})|Should -BeFalse
        if($Mode -eq 'DeleteFailure'){Should -Invoke -ModuleName SecretsTier Invoke-BitwardenCli -Times 1 -Exactly}
        else{Should -Invoke -ModuleName SecretsTier Invoke-BitwardenCli -Times 0 -Exactly}
    }


    # Protects: a finite committed read under repeated canonical publication.
    # Detects: continuous writer activity or same-generation edits must not trigger unbounded retries.
    # Needs: Three complete synthetic generations; all transport reads remain in memory.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Read-BwBackupCommittedInternal
    It 'bounds canonical recovery reads under <Mode>' -TestCases @(@{Mode='Continuous'},@{Mode='SameGeneration'}){
        param($Mode)
        $script:retryNotes=@();$allParts=@()
        foreach($value in @('first','second','third')){
            Initialize-SyntheticBackupBackend
            $script:backupSecrets=@{alpha=$value}
            (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
            $items=@($script:backupItems.Values)
            $script:retryNotes+=@($items|Where-Object name -EQ '[MACHINE] fixture-machine Secrets')[0].notes
            $allParts+=@($items|Where-Object name -NE '[MACHINE] fixture-machine Secrets')
        }
        if($Mode -eq 'SameGeneration'){
            $manifest=$script:retryNotes[0].Substring($script:retryNotes[0].IndexOf([char]10)+1)|ConvertFrom-Json -AsHashtable
            $manifest.sha256='0'*64
            $script:retryNotes[1]='#pcai-secrets-backup-v2'+[char]10+($manifest|ConvertTo-Json -Compress)
        }
        $script:retryParts=$allParts;$script:retryReads=0
        Mock -ModuleName SecretsTier Invoke-MachineBitwardenCommand {
            if($Arguments[0] -ne 'list'){throw 'Unexpected recovery transport operation.'}
            if($Arguments[-1] -eq '[MACHINE] fixture-machine Secrets'){
                $index=[Math]::Min($script:retryReads,2);$script:retryReads++
                return @([pscustomobject]@{id='canonical';name='[MACHINE] fixture-machine Secrets';type=2;notes=$script:retryNotes[$index]})|ConvertTo-Json -AsArray -Compress
            }
            $script:retryParts|ConvertTo-Json -Depth 20 -AsArray -Compress
        }
        {& $script:providerModule {Import-SecretsFromBwVault -SetEnvironment -Confirm:$false -ErrorAction Stop}}|Should -Throw
        $script:retryReads|Should -Be $(if($Mode -eq 'Continuous'){3}else{2})
        Should -Invoke -ModuleName SecretsTier Save-SecretsToCache -Times 0 -Exactly
        Should -Invoke -ModuleName SecretsTier Set-SecretsAsEnvironment -Times 0 -Exactly
    }
    # Protects: the currently committed generation from cleanup even when the caller owns its parts.
    # Detects: cleanup must recheck the canonical generation rather than deleting active recovery data.
    # Needs: An actual validated synthetic snapshot and exact-delete invocation accounting.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Remove-BwBackupPartsInternal
    It 'preserves an active generation and unrelated notes during cleanup' {
        Initialize-SyntheticBackupBackend
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $canonical=@($script:backupItems.Values|Where-Object name -EQ '[MACHINE] fixture-machine Secrets')[0].notes
        $snapshot=& $script:providerModule {param($text) Read-BwBackupSnapshotInternal -CanonicalText $text} $canonical
        $script:backupItems['foreign']=[pscustomobject]@{id='foreign';name='unrelated';type=2;notes='unrelated-synthetic'}
        $before=$script:backupItems|ConvertTo-Json -Depth 20 -Compress
        (& $script:providerModule {param($item,$text) Remove-BwBackupPartsInternal -Snapshot $item -CanonicalText $text -Confirm:$false} $snapshot $canonical)|Should -BeFalse
        ($script:backupItems|ConvertTo-Json -Depth 20 -Compress)|Should -BeExactly $before
        Should -Invoke -ModuleName SecretsTier Invoke-BitwardenCli -Times 0 -Exactly -ParameterFilter {$Arguments[0] -eq 'delete'}
    }


    # Protects: successful committed backup publication despite incomplete prior-generation retirement.
    # Detects: silent cleanup failures hide redundant data, while cleanup diagnostics must never expose private provider details.
    # Needs: Valid in-memory generations plus true/false/throw cleanup-boundary outcomes.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Export-SecretsToBwVault
    It 'keeps a validated commit successful and reports only sanitized cleanup outcome <Outcome>' -TestCases @(
        @{Outcome='True'},@{Outcome='False'},@{Outcome='Throw'}
    ){
        param($Outcome)
        Initialize-SyntheticBackupBackend
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $script:cleanupOutcome=$Outcome
        Mock -ModuleName SecretsTier Remove-BwBackupPartsInternal {
            if($script:cleanupOutcome -eq 'Throw'){throw 'PRIVATE-PROVIDER-DETAILS-SENTINEL'}
            $script:cleanupOutcome -eq 'True'
        }
        $script:backupSecrets=@{alpha='synthetic-new-commit'}
        $publication=& $script:providerModule {
            $warnings=@()
            $success=Export-SecretsToBwVault -Confirm:$false -WarningVariable warnings -WarningAction SilentlyContinue
            [pscustomobject]@{Success=$success;Warnings=@($warnings)}
        }
        $publication.Success|Should -BeTrue
        $warnings=$publication.Warnings
        $r=& $script:providerModule {Import-SecretsFromBwVault -Confirm:$false -ErrorAction Stop}
        $r.alpha|Should -BeExactly 'synthetic-new-commit'
        Should -Invoke -ModuleName SecretsTier Remove-BwBackupPartsInternal -Times 1 -Exactly
        if($Outcome -eq 'True'){@($warnings).Count|Should -Be 0}
        else{
            @($warnings).Count|Should -Be 1
            $warnings[0].ToString()|Should -BeExactly 'Backup committed; prior generation cleanup was not confirmed. Unknown or concurrent data was preserved.'
            ($warnings|Out-String)|Should -Not -Match 'PRIVATE-PROVIDER-DETAILS-SENTINEL'
        }
    }

    # Protects: refuses BWS invocation without an explicit token.
    # Detects: Missing-token invocation must fail before the guarded external process boundary is called.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-BwsSecretsList
    It 'refuses BWS invocation without an explicit token' {
        (& $script:providerModule {Get-BwsSecretsList})|Should -BeNullOrEmpty
        Should -Invoke -ModuleName SecretsTier Invoke-SecretBackendProcess -Times 0 -Exactly
    }
    # Protects: parses unique synthetic BWS keys without invoking a real executable.
    # Detects: Cloud JSON parsing must retain each unique key/value while passing the token only to the mocked process environment.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-BwsSecretsList
    It 'parses unique synthetic BWS keys without invoking a real executable' {
        $env:BWS_ACCESS_TOKEN='synthetic-never-valid' # pragma: allowlist secret - isolated synthetic fixture; never a live credential
        Mock -ModuleName SecretsTier Get-Command {[pscustomobject]@{Source='C:/synthetic/bws.exe'}} -ParameterFilter {$Name -eq 'bws.exe'}
        Mock -ModuleName SecretsTier Invoke-SecretBackendProcess {[pscustomobject]@{Success=$true;StdOut='[{"key":"alpha","value":"synthetic-a"},{"key":"beta","value":"synthetic-b"}]'}}
        $secrets=& $script:providerModule {Get-BwsSecretsList}
        $secrets.Count|Should -Be 2;$secrets.alpha|Should -BeExactly 'synthetic-a'
        Should -Invoke -ModuleName SecretsTier Invoke-SecretBackendProcess -Times 1 -Exactly -ParameterFilter {($Arguments -join ',') -eq 'secret,list,-o,json'}
    }
    # Protects: withholds invalid or duplicate BWS payloads (<Payload>).
    # Detects: Malformed JSON, nonarray input and duplicate secret keys must be refused rather than partially accepted.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-BwsSecretsList
    It 'withholds invalid or duplicate BWS payloads (<Payload>)' -TestCases @(
        @{Payload='not-json-private-sentinel'},@{Payload='[{"key":"","value":"private-sentinel"}]'},
        @{Payload='[{"key":"same","value":"private-sentinel"},{"key":"same","value":"private-sentinel"}]'}
    ) {
        param($Payload)
        $env:BWS_ACCESS_TOKEN='synthetic-never-valid' # pragma: allowlist secret - isolated synthetic fixture; never a live credential
        Mock -ModuleName SecretsTier Get-Command {[pscustomobject]@{Source='C:/synthetic/bws.exe'}} -ParameterFilter {$Name -eq 'bws.exe'}
        Mock -ModuleName SecretsTier Invoke-SecretBackendProcess {[pscustomobject]@{Success=$true;StdOut=$Payload}}
        try{& $script:providerModule {Get-BwsSecretsList};throw 'Expected refusal'}catch{$_.Exception.Message|Should -Match 'private content is withheld';$_.Exception.Message|Should -Not -Match 'private-sentinel'}
    }
    # Protects: retains a valid cache without querying cloud unless forced.
    # Detects: A valid cache fast path must perform no cloud fetch or cache rewrite without an explicit Force request.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Sync-SecretsFromBws
    It 'retains a valid cache without querying cloud unless forced' {
        Mock -ModuleName SecretsTier Get-CacheStatus {[pscustomobject]@{IsValid=$true}}
        (& $script:providerModule {Sync-SecretsFromBws -Confirm:$false})|Should -BeTrue
        Should -Invoke -ModuleName SecretsTier Test-InternetConnection -Times 0 -Exactly
    }
    # Protects: publishes fetched synthetic BWS values only when online.
    # Detects: Fetched online values must reach the cache exactly once without an implicit environment assignment.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Sync-SecretsFromBws
    It 'publishes fetched synthetic BWS values only when online' {
        Mock -ModuleName SecretsTier Test-InternetConnection {$true}
        Mock -ModuleName SecretsTier Get-BwsSecretsList {@{alpha='synthetic-only'}}
        Mock -ModuleName SecretsTier Save-SecretsToCache {}
        (& $script:providerModule {Sync-SecretsFromBws -Force -Confirm:$false})|Should -BeTrue
        Should -Invoke -ModuleName SecretsTier Save-SecretsToCache -Times 1 -Exactly -ParameterFilter {$Secrets.alpha -eq 'synthetic-only'}
    }
    # Protects: performs no publication when forced cloud sync is offline or empty (<Online>).
    # Detects: Offline connectivity or an empty cloud result must not publish a cache or environment update.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Sync-SecretsFromBws
    It 'performs no publication when forced cloud sync is offline or empty (<Online>)' -TestCases @(@{Online=$false},@{Online=$true}) {
        param($Online)
        Mock -ModuleName SecretsTier Test-InternetConnection {$Online}
        Mock -ModuleName SecretsTier Get-BwsSecretsList {@{}}
        (& $script:providerModule {Sync-SecretsFromBws -Force -Confirm:$false -WarningAction SilentlyContinue})|Should -BeFalse
        Should -Invoke -ModuleName SecretsTier Save-SecretsToCache -Times 0 -Exactly
    }
    # Protects: serializes secure-note edits on stdin and never puts synthetic private data in arguments.
    # Detects: Private note content must travel through encoded standard input rather than process arguments, and edits must retain the exact item ID.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Set-BwNoteTextInternal
    It 'serializes secure-note edits on stdin and never puts synthetic private data in arguments' {
        Mock -ModuleName SecretsTier Invoke-MachineBitwardenCommand {
            if($Arguments[0] -eq 'list'){return '[{"id":"synthetic-note","name":"fixture-note","type":2}]'}
            '{"id":"synthetic-note","name":"fixture-note","type":2,"notes":"old"}'
        }
        Mock -ModuleName SecretsTier Invoke-BitwardenCli {
            $json=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($StandardInput))|ConvertFrom-Json
            $json.notes|Should -BeExactly 'synthetic-private-note'
            ($Arguments -join ',')|Should -BeExactly 'edit,item,synthetic-note'
            ($Arguments -join ',')|Should -Not -Match 'synthetic-private-note'
            [pscustomobject]@{Success=$true}
        }
        (& $script:providerModule {Set-BwNoteTextInternal -Name fixture-note -Text synthetic-private-note -Confirm:$false})|Should -BeTrue
    }
    # Protects: creates a new secure note only from an unambiguous empty search.
    # Detects: An empty exact search must create one secure-note item with the selected name and body rather than edit an unrelated item.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Set-BwNoteTextInternal
    It 'creates a new secure note only from an unambiguous empty search' {
        Mock -ModuleName SecretsTier Invoke-MachineBitwardenCommand {'[]'}
        Mock -ModuleName SecretsTier Invoke-BitwardenCli {
            $json=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($StandardInput))|ConvertFrom-Json
            $json.type|Should -Be 2;$json.secureNote.type|Should -Be 0;$json.name|Should -BeExactly 'fixture-note'
            [pscustomobject]@{Success=$true}
        }
        (& $script:providerModule {Set-BwNoteTextInternal fixture-note synthetic-body -Confirm:$false})|Should -BeTrue
        Should -Invoke -ModuleName SecretsTier Invoke-BitwardenCli -Times 1 -Exactly -ParameterFilter {($Arguments -join ',') -eq 'create,item'}
    }
    # Protects: rejects ambiguous secure-note lookup before writing.
    # Detects: Duplicate exact note names must prevent both reads and writes without invoking the mutation boundary.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-BwNoteTextInternal, Set-BwNoteTextInternal
    It 'rejects ambiguous secure-note lookup before writing' {
        Mock -ModuleName SecretsTier Invoke-MachineBitwardenCommand {'[{"id":"one","name":"fixture-note","type":2},{"id":"two","name":"fixture-note","type":2}]'}
        {& $script:providerModule {Set-BwNoteTextInternal fixture-note synthetic-body -Confirm:$false}}|Should -Throw '*Ambiguous*'
        {& $script:providerModule {Get-BwNoteTextInternal fixture-note}}|Should -Throw '*ambiguous*'
        Should -Invoke -ModuleName SecretsTier Invoke-BitwardenCli -Times 0 -Exactly
    }
    # Protects: rejects a changed secure-note identity before writing.
    # Detects: An item ID changed between lookup and retrieval must stop the edit before any private payload is sent.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Set-BwNoteTextInternal
    It 'rejects a changed secure-note identity before writing' {
        Mock -ModuleName SecretsTier Invoke-MachineBitwardenCommand {
            if($Arguments[0] -eq 'list'){return '[{"id":"one","name":"fixture-note","type":2}]'}
            '{"id":"changed","name":"fixture-note","type":2,"notes":"old"}'
        }
        {& $script:providerModule {Set-BwNoteTextInternal fixture-note synthetic-body -Confirm:$false}}|Should -Throw '*identity could not be verified*'
        Should -Invoke -ModuleName SecretsTier Invoke-BitwardenCli -Times 0 -Exactly
    }
    # Protects: returns only the exact secure-note name and ignores fuzzy/type mismatches.
    # Detects: Fuzzy name matches and non-secure-note items must not substitute their contents for the selected exact note.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-BwNoteTextInternal
    It 'returns only the exact secure-note name and ignores fuzzy/type mismatches' {
        Mock -ModuleName SecretsTier Invoke-MachineBitwardenCommand {'[{"id":"one","name":"fixture-note","type":2,"notes":"synthetic"},{"id":"two","name":"fixture-note-extra","type":2,"notes":"wrong"},{"id":"three","name":"fixture-note","type":1,"notes":"wrong"}]'}
        (& $script:providerModule {Get-BwNoteTextInternal fixture-note})|Should -BeExactly 'synthetic'
    }
    # Protects: exports a large snapshot using bounded v2 parts and a final canonical commit.
    # Detects: The old positional overwrite could publish a mixed-generation credential after interruption.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Export-SecretsToBwVault
    It 'splits a large synthetic export into bounded v2 parts before committing the canonical note' {
        Initialize-SyntheticBackupBackend
        $script:backupSecrets=@{alpha=('a'*11000)}
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $items=@($script:backupItems.Values)
        $canonical=@($items|Where-Object name -EQ '[MACHINE] fixture-machine Secrets')
        $canonical.Count|Should -Be 1
        $manifest=$canonical[0].notes.Substring($canonical[0].notes.IndexOf([char]10)+1)|ConvertFrom-Json
        $parts=@($items|Where-Object name -NE $canonical[0].name)
        $parts.Count|Should -Be $manifest.partCount
        $parts.Count|Should -BeGreaterThan 1
        foreach($part in $parts){
            $part.name|Should -Match '\[pcai-v2:[a-f0-9]{32}:[1-9][0-9]*\]$'
            $part.notes|Should -Match '^#pcai-secrets-v2 generation=[a-f0-9]{32} part=[1-9][0-9]* parts=[1-9][0-9]*\n'
            $part.notes.Substring($part.notes.IndexOf([char]10)+1).Length|Should -BeLessOrEqual 4500
        }
        $last=@($items|Sort-Object {[int]($_.id -replace '^synthetic-','')})[-1]
        $last.name|Should -BeExactly $canonical[0].name
        (& $script:providerModule {Import-SecretsFromBwVault -Confirm:$false}).alpha|Should -BeExactly ('a'*11000)
    }
    # Protects: refuses an expired backup without Force before reading or writing values.
    # Detects: An expired cache without explicit Force must abort export before fetching any cloud secret values.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Export-SecretsToBwVault
    It 'refuses an expired backup without Force before reading or writing values' {
        Mock -ModuleName SecretsTier Test-BwSession {$true}
        Mock -ModuleName SecretsTier Get-CacheStatus {[pscustomobject]@{IsExpired=$true}}
        Mock -ModuleName SecretsTier Get-BwsSecretsList {throw 'No cloud read permitted.'}
        {& $script:providerModule {Export-SecretsToBwVault -Confirm:$false}}|Should -Throw '*explicit Force*'
        Should -Invoke -ModuleName SecretsTier Get-BwsSecretsList -Times 0 -Exactly
    }
    # Protects: refuses legacy multipart backups without changing cache or environment.
    # Detects: Positional legacy parts cannot establish that credential fragments belong to one export.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Import-SecretsFromBwVault
    It 'refuses a legacy multipart backup with re-export guidance before cache or environment publication' {
        Initialize-SyntheticBackupBackend
        $script:backupItems['legacy-1']=[pscustomobject]@{id='legacy-1';name='[MACHINE] fixture-machine Secrets';type=2;notes=('#chunked part=1 parts=2'+[char]10+'{"machine":"fixture-machine","exported":"synthetic-date",')}
        $script:backupItems['legacy-2']=[pscustomobject]@{id='legacy-2';name='[MACHINE] fixture-machine Secrets #2';type=2;notes=('#chunked part=2 parts=2'+[char]10+'"secrets":{"alpha":"synthetic-restored"}}')}
        $script:backupRecovered=@{alpha='previous-synthetic-cache'}
        $before=($script:backupItems|ConvertTo-Json -Depth 20 -Compress)
        {& $script:providerModule {Import-SecretsFromBwVault -SetEnvironment -Confirm:$false -ErrorAction Stop}}|Should -Throw '*re-export*'
        $script:backupRecovered.alpha|Should -BeExactly 'previous-synthetic-cache'
        ($script:backupItems|ConvertTo-Json -Depth 20 -Compress)|Should -BeExactly $before
        Should -Invoke -ModuleName SecretsTier Save-SecretsToCache -Times 0 -Exactly
        Should -Invoke -ModuleName SecretsTier Set-SecretsAsEnvironment -Times 0 -Exactly
    }
    # Protects: preserves legitimate single-note JSON compatibility and explicit environment intent.
    # Detects: Refusing unsafe multipart input must not strand complete legacy single-note snapshots.
    # Needs: In-memory synthetic provider; no real credentials or environment writes.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1: Import-SecretsFromBwVault
    It 'imports a complete legacy single-note backup and honors the explicit environment request' {
        Initialize-SyntheticBackupBackend
        $script:backupItems['legacy-single']=[pscustomobject]@{id='legacy-single';name='[MACHINE] fixture-machine Secrets';type=2;notes='{"machine":"fixture-machine","exported":"synthetic-date","secrets":{"alpha":"synthetic-restored"}}'}
        Mock -ModuleName SecretsTier Set-SecretsAsEnvironment {}
        $secrets=& $script:providerModule {Import-SecretsFromBwVault -SetEnvironment -Confirm:$false}
        $secrets.alpha|Should -BeExactly 'synthetic-restored'
        Should -Invoke -ModuleName SecretsTier Save-SecretsToCache -Times 1 -Exactly -ParameterFilter {$Secrets.alpha -eq 'synthetic-restored'}
        Should -Invoke -ModuleName SecretsTier Set-SecretsAsEnvironment -Times 1 -Exactly
    }
    # Protects: rejects invalid v2 part counts before cache publication (<Count>).
    # Detects: Unbounded or zero-count manifests must not drive provider enumeration or publication.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Import-SecretsFromBwVault
    It 'rejects invalid backup part counts before cache publication (<Count>)' -TestCases @(@{Count=0},@{Count=1025}) {
        param($Count)
        Initialize-SyntheticBackupBackend
        (& $script:providerModule {Export-SecretsToBwVault -Confirm:$false})|Should -BeTrue
        $canonical=@($script:backupItems.Values|Where-Object name -EQ '[MACHINE] fixture-machine Secrets')[0]
        $manifest=$canonical.notes.Substring($canonical.notes.IndexOf([char]10)+1)|ConvertFrom-Json -AsHashtable
        $manifest.partCount=$Count
        $canonical.notes='#pcai-secrets-backup-v2'+[char]10+($manifest|ConvertTo-Json -Compress)
        {& $script:providerModule {Import-SecretsFromBwVault -Confirm:$false -ErrorAction Stop}}|Should -Throw
        Should -Invoke -ModuleName SecretsTier Save-SecretsToCache -Times 0 -Exactly
    }
    # Protects: withholds mismatched-machine backup contents and refuses hydration.
    # Detects: A legacy single-note payload naming another machine must return no secrets and perform no cache or environment publication.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Import-SecretsFromBwVault
    It 'withholds mismatched-machine backup contents and refuses hydration' {
        Mock -ModuleName SecretsTier Test-BwSession {$true}
        Mock -ModuleName SecretsTier Get-BwNoteTextInternal {'{"machine":"other-machine","exported":"synthetic-date","secrets":{"alpha":"private-sentinel"}}'}
        (& $script:providerModule {Import-SecretsFromBwVault -Confirm:$false -ErrorAction SilentlyContinue})|Should -BeNullOrEmpty
        Should -Invoke -ModuleName SecretsTier Save-SecretsToCache -Times 0 -Exactly
        Should -Invoke -ModuleName SecretsTier Set-SecretsAsEnvironment -Times 0 -Exactly
    }
    # Protects: validated cache fallback after actual BWS launch or parsing failure (<Failure>/<Scenario>).
    # Detects: cloud retrieval exceptions abort initialization, leak native output, or hydrate unvalidated cloud values.
    # Needs: actual BWS parser with inert executable discovery/native process boundary; synthetic cache/environment substitutes only.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1::Get-BwsSecretsList,Initialize-MachineSecretsTiered
    It 'uses validated <Scenario> cache after actual BWS <Failure> failure' -TestCases @(
        @{Failure='Launch';Scenario='Valid';Expected='Local Cache'},@{Failure='Malformed';Scenario='Valid';Expected='Local Cache'},
        @{Failure='Launch';Scenario='Expired';Expected='Local Cache (EXPIRED)'},@{Failure='Malformed';Scenario='Expired';Expected='Local Cache (EXPIRED)'},
        @{Failure='Launch';Scenario='Missing';Expected='NONE'},@{Failure='Malformed';Scenario='Missing';Expected='NONE'}
    ) {
        param($Failure,$Scenario,$Expected)
        $env:BWS_ACCESS_TOKEN='synthetic-token' # pragma: allowlist secret - isolated synthetic fixture; never a live credential
        Mock -ModuleName SecretsTier Test-InternetConnection {$true}
        Mock -ModuleName SecretsTier Get-Command {[pscustomobject]@{Source='C:/synthetic/bws.exe'}} -ParameterFilter {$Name -eq 'bws.exe'}
        Mock -ModuleName SecretsTier Invoke-SecretBackendProcess {
            if($Failure -eq 'Launch'){throw [ComponentModel.Win32Exception]::new('private-cloud-launch-sentinel')}
            [pscustomobject]@{Success=$true;StdOut='private-malformed-json-sentinel'}
        }
        Mock -ModuleName SecretsTier Get-CacheStatus {[pscustomobject]@{IsValid=($Scenario -eq 'Valid');Exists=($Scenario -eq 'Expired');IsExpired=($Scenario -eq 'Expired');LastUpdated=[datetime]::UtcNow;Age=[timespan]::FromDays(8)}}
        Mock -ModuleName SecretsTier Get-SecretsFromCache {@{alpha='synthetic-cache-value'}}
        Mock -ModuleName SecretsTier Set-InitializedSecretsEnvironment {}
        $result=& $script:providerModule {Initialize-MachineSecretsTiered -PersistentKeys alpha -Confirm:$false}
        $result.Source|Should -BeExactly $Expected
        $result.Success|Should -Be ($Scenario -ne 'Missing')
        $result.SecretCount|Should -Be $(if($Scenario -eq 'Missing'){0}else{1})
        $result.Warnings|Should -Contain 'Cloud secret retrieval failed; trying the validated local cache.'
        ($result|ConvertTo-Json -Depth 5)|Should -Not -Match 'private-cloud-launch-sentinel|private-malformed-json-sentinel|synthetic-cache-value'
        Should -Invoke -ModuleName SecretsTier Invoke-SecretBackendProcess -Times 1 -Exactly -ParameterFilter {$FilePath -eq 'C:/synthetic/bws.exe' -and ($Arguments -join ',') -eq 'secret,list,-o,json'}
        Should -Invoke -ModuleName SecretsTier Save-SecretsToCache -Times 0 -Exactly
        Should -Invoke -ModuleName SecretsTier Get-CacheStatus -Times 1 -Exactly
        if($Scenario -eq 'Missing'){
            Should -Invoke -ModuleName SecretsTier Get-SecretsFromCache -Times 0 -Exactly
            Should -Invoke -ModuleName SecretsTier Set-InitializedSecretsEnvironment -Times 0 -Exactly
        }else{
            Should -Invoke -ModuleName SecretsTier Get-SecretsFromCache -Times 1 -Exactly -ParameterFilter {[bool]$IgnoreAge -eq ($Scenario -eq 'Expired')}
            Should -Invoke -ModuleName SecretsTier Set-InitializedSecretsEnvironment -Times 1 -Exactly -ParameterFilter {$Secrets.alpha -eq 'synthetic-cache-value' -and $PersistentKeys.Count -eq 1 -and $PersistentKeys[0] -eq 'alpha'}
        }
    }
    # Protects: selected-cache admission and hydration errors remain visible after recoverable cloud retrieval failure (<Boundary>).
    # Detects: an overly broad cloud catch swallows downstream validation or environment errors.
    # Needs: actual BWS retrieval with inert launch refusal, synthetic cache and hydration boundaries.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1::Get-BwsSecretsList,Initialize-MachineSecretsTiered
    It 'propagates <Boundary> failure after actual BWS launch refusal' -TestCases @(@{Boundary='Admission'},@{Boundary='Hydration'}) {
        param($Boundary)
        $env:BWS_ACCESS_TOKEN='synthetic-token' # pragma: allowlist secret - isolated synthetic fixture; never a live credential
        Mock -ModuleName SecretsTier Test-InternetConnection {$true}
        Mock -ModuleName SecretsTier Get-Command {[pscustomobject]@{Source='C:/synthetic/bws.exe'}} -ParameterFilter {$Name -eq 'bws.exe'}
        Mock -ModuleName SecretsTier Invoke-SecretBackendProcess {throw [ComponentModel.Win32Exception]::new('private-cloud-launch-sentinel')}
        Mock -ModuleName SecretsTier Get-CacheStatus {if($Boundary -eq 'Admission'){throw 'Synthetic admission error.'};[pscustomobject]@{IsValid=$true}}
        Mock -ModuleName SecretsTier Get-SecretsFromCache {@{alpha='synthetic-cache-value'}}
        Mock -ModuleName SecretsTier Set-InitializedSecretsEnvironment {throw 'Synthetic hydration error.'}
        {& $script:providerModule {Initialize-MachineSecretsTiered -Confirm:$false}}|Should -Throw ("*Synthetic $($Boundary.ToLowerInvariant()) error*")
        Should -Invoke -ModuleName SecretsTier Save-SecretsToCache -Times 0 -Exactly
        Should -Invoke -ModuleName SecretsTier Get-CacheStatus -Times 1 -Exactly
        Should -Invoke -ModuleName SecretsTier Set-InitializedSecretsEnvironment -Times $(if($Boundary -eq 'Admission'){0}else{1}) -Exactly
    }
    # Protects: validated local fallback when fetched cloud secrets cannot be cached (<Scenario>).
    # Detects: a publication exception escapes before cache admission, hydrates uncommitted cloud values, or leaks private error text.
    # Needs: inert provider/cache/environment boundaries; no real credentials, storage or User-variable writes.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1::Initialize-MachineSecretsTiered
    It 'continues through validated <Scenario> cache after cloud cache publication fails' -TestCases @(
        @{Scenario='Valid';Expected='Local Cache'},@{Scenario='Expired';Expected='Local Cache (EXPIRED)'},@{Scenario='Missing';Expected='NONE'}
    ) {
        param($Scenario,$Expected)
        $env:BWS_ACCESS_TOKEN='synthetic-token' # pragma: allowlist secret - isolated synthetic fixture; never a live credential
        $script:initializationEvents=[Collections.Generic.List[string]]::new()
        Mock -ModuleName SecretsTier Test-InternetConnection {$true}
        Mock -ModuleName SecretsTier Get-BwsSecretsList {@{alpha='synthetic-cloud-value'}}
        Mock -ModuleName SecretsTier Save-SecretsToCache {$script:initializationEvents.Add('Save');throw [IO.IOException]::new('private-publication-error-sentinel')}
        Mock -ModuleName SecretsTier Get-CacheStatus {$script:initializationEvents.Add('Status');[pscustomobject]@{IsValid=($Scenario -eq 'Valid');Exists=($Scenario -eq 'Expired');IsExpired=($Scenario -eq 'Expired');LastUpdated=[datetime]::UtcNow;Age=[timespan]::FromDays(8)}}
        Mock -ModuleName SecretsTier Get-SecretsFromCache {$script:initializationEvents.Add('Read');@{alpha='synthetic-cache-value'}}
        Mock -ModuleName SecretsTier Set-InitializedSecretsEnvironment {$script:initializationEvents.Add('Hydrate')}
        $result=& $script:providerModule {Initialize-MachineSecretsTiered -PersistentKeys alpha -Confirm:$false}
        $result.Source|Should -BeExactly $Expected
        $result.Success|Should -Be ($Scenario -ne 'Missing')
        $result.SecretCount|Should -Be $(if($Scenario -eq 'Missing'){0}else{1})
        $result.Warnings|Should -Contain 'Cloud cache publication failed; trying the validated local cache.'
        ($result|ConvertTo-Json -Depth 5)|Should -Not -Match 'private-publication-error-sentinel|synthetic-cloud-value|synthetic-cache-value'
        Should -Invoke -ModuleName SecretsTier Save-SecretsToCache -Times 1 -Exactly
        Should -Invoke -ModuleName SecretsTier Get-CacheStatus -Times 1 -Exactly
        if($Scenario -eq 'Missing'){
            ($script:initializationEvents -join ',')|Should -BeExactly 'Save,Status'
            Should -Invoke -ModuleName SecretsTier Get-SecretsFromCache -Times 0 -Exactly
            Should -Invoke -ModuleName SecretsTier Set-InitializedSecretsEnvironment -Times 0 -Exactly
        }else{
            ($script:initializationEvents -join ',')|Should -BeExactly 'Save,Status,Read,Hydrate'
            Should -Invoke -ModuleName SecretsTier Get-SecretsFromCache -Times 1 -Exactly -ParameterFilter {[bool]$IgnoreAge -eq ($Scenario -eq 'Expired')}
            Should -Invoke -ModuleName SecretsTier Set-InitializedSecretsEnvironment -Times 1 -Exactly -ParameterFilter {$Secrets.Count -eq 1 -and $Secrets.alpha -eq 'synthetic-cache-value' -and $PersistentKeys.Count -eq 1 -and $PersistentKeys[0] -eq 'alpha'}
        }
    }
    # Protects: unsafe selected storage remains fail-closed after a recoverable cloud cache-write failure.
    # Detects: the new publication catch swallows cache admission errors or hydrates any unvalidated input.
    # Needs: inert cloud/cache/environment boundaries and one synthetic invalid-storage exception.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretsTier.psm1::Initialize-MachineSecretsTiered
    It 'propagates selected cache admission failure after cloud cache publication fails' {
        $env:BWS_ACCESS_TOKEN='synthetic-token' # pragma: allowlist secret - isolated synthetic fixture; never a live credential
        Mock -ModuleName SecretsTier Test-InternetConnection {$true}
        Mock -ModuleName SecretsTier Get-BwsSecretsList {@{alpha='synthetic-cloud-value'}}
        Mock -ModuleName SecretsTier Save-SecretsToCache {throw [IO.IOException]::new('private-publication-error-sentinel')}
        Mock -ModuleName SecretsTier Get-CacheStatus {throw [IO.InvalidDataException]::new('Selected synthetic cache is invalid.')}
        Mock -ModuleName SecretsTier Get-SecretsFromCache {throw 'Unvalidated cache read forbidden.'}
        {& $script:providerModule {Initialize-MachineSecretsTiered -Confirm:$false}}|Should -Throw '*Selected synthetic cache is invalid*'
        Should -Invoke -ModuleName SecretsTier Get-CacheStatus -Times 1 -Exactly
        Should -Invoke -ModuleName SecretsTier Get-SecretsFromCache -Times 0 -Exactly
        Should -Invoke -ModuleName SecretsTier Set-InitializedSecretsEnvironment -Times 0 -Exactly
    }
    # Protects: selects cloud, valid cache, expired cache and manual recovery deterministically (<State>).
    # Detects: Each available tier must report its actual source; the missing-data path must remain unsuccessful and perform no hydration.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Initialize-MachineSecretsTiered
    It 'selects cloud, valid cache, expired cache and manual recovery deterministically (<State>)' -TestCases @(
        @{State='Cloud';Expected='BWS Cloud'},@{State='Valid';Expected='Local Cache'},
        @{State='Expired';Expected='Local Cache (EXPIRED)'},@{State='Missing';Expected='NONE'}
    ) {
        param($State,$Expected)
        if($State -eq 'Cloud'){$env:BWS_ACCESS_TOKEN='synthetic-token'} # pragma: allowlist secret - isolated synthetic fixture; never a live credential
        Mock -ModuleName SecretsTier Test-InternetConnection {$true}
        Mock -ModuleName SecretsTier Get-BwsSecretsList {@{alpha='synthetic-value'}}
        Mock -ModuleName SecretsTier Save-SecretsToCache {}
        Mock -ModuleName SecretsTier Get-CacheStatus {[pscustomobject]@{IsValid=($State -eq 'Valid');Exists=($State -eq 'Expired');IsExpired=($State -eq 'Expired');LastUpdated=[datetime]::UtcNow;Age=[timespan]::FromDays(8)}}
        Mock -ModuleName SecretsTier Get-SecretsFromCache {@{alpha='synthetic-value'}}
        Mock -ModuleName SecretsTier Set-InitializedSecretsEnvironment {}
        $result=& $script:providerModule {Initialize-MachineSecretsTiered -Confirm:$false}
        $result.Source|Should -BeExactly $Expected
        $result.Success|Should -Be ($State -ne 'Missing')
        if($State -eq 'Missing'){Should -Invoke -ModuleName SecretsTier Set-InitializedSecretsEnvironment -Times 0 -Exactly}
        else{Should -Invoke -ModuleName SecretsTier Set-InitializedSecretsEnvironment -Times 1 -Exactly}
    }
    # Protects: persists only requested keys and refuses missing keys before any User write.
    # Detects: User-scope persistence must contain only requested keys, and a missing selection must fail before a second environment write.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Set-PersistentSecrets
    It 'persists only requested keys and refuses missing keys before any User write' {
        Mock -ModuleName SecretsTier Get-SecretsFromCache {@{alpha='synthetic-a';beta='synthetic-b'}}
        Mock -ModuleName SecretsTier Set-SecretsAsEnvironment {}
        & $script:providerModule {Set-PersistentSecrets -SecretNames alpha -Confirm:$false}
        Should -Invoke -ModuleName SecretsTier Set-SecretsAsEnvironment -Times 1 -Exactly -ParameterFilter {$Scope -eq 'User' -and $Secrets.Count -eq 1 -and $Secrets.alpha -eq 'synthetic-a'}
        {& $script:providerModule {Set-PersistentSecrets -SecretNames missing -Confirm:$false}}|Should -Throw '*missing*'
        Should -Invoke -ModuleName SecretsTier Set-SecretsAsEnvironment -Times 1 -Exactly
    }
}


Describe 'Synthetic resource coordination and mapping boundaries' {
    # Protects resource filtering, metadata projection, credential identity and
    # observed mapping postconditions. All providers/network/native writes use
    # inert process-local substitutes; synthetic config lives under FixtureDrive.
    BeforeAll {
        . (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretBackendUtilities.ps1')
        $script:syntheticSecretManagement=New-Module -Name Microsoft.PowerShell.SecretManagement -ScriptBlock {
            function Get-SecretVault {[CmdletBinding()]param([string]$Name) throw 'Synthetic provider boundary not mocked.'}
            function Get-SecretInfo {[CmdletBinding()]param([string]$Name,[string]$Vault) throw 'Synthetic provider boundary not mocked.'}
            function Get-Secret {[CmdletBinding()]param([string]$Name,[string]$Vault,[switch]$AsPlainText) throw 'Synthetic provider boundary not mocked.'}
            function Set-Secret {[CmdletBinding()]param([string]$Name,[string]$Vault,[object]$Secret,[hashtable]$Metadata) throw 'Synthetic provider boundary not mocked.'}
            function Register-SecretVault {[CmdletBinding()]param([string]$Name,[string]$ModuleName,[switch]$DefaultVault) throw 'Synthetic provider boundary not mocked.'}
            Export-ModuleMember -Function *
        }
        $script:syntheticSecretManagement|Import-Module -Global -Force
    }
    BeforeEach {
        Mock 'Microsoft.PowerShell.SecretManagement\Get-SecretVault' {$null}
        Mock 'Microsoft.PowerShell.SecretManagement\Get-SecretInfo' {$null}
        Mock 'Microsoft.PowerShell.SecretManagement\Get-Secret' {throw 'Unexpected provider read.'}
        Mock 'Microsoft.PowerShell.SecretManagement\Set-Secret' {throw 'Unexpected provider write.'}
        Mock Invoke-SecretBackendProcess {throw 'Unexpected native process.'}
        Mock Test-TcpEndpointReachable {throw 'Unexpected network probe.'}
        Mock Set-CredentialManagerCredential -RemoveParameterType Password {throw 'Unexpected native credential write.'}
    }
    AfterAll {Remove-Module $script:syntheticSecretManagement -Force}


    Describe 'Synthetic backend discovery and credential normalization' {
        # Protects command selection and attested package identity. Synthetic
        # files are discovery fixtures and are never executed.
        BeforeAll {
            # PowerShellGet is optional on the host. Supply only an inert test
            # boundary so a mock can verify intent without installing anything.
            $script:archiveDeadlineRunner=(Get-Command Invoke-SecretBackendProcess).ScriptBlock
            $script:archiveDeadlineExecutable=(Get-Process -Id $PID).Path
            function Install-Module {
                [CmdletBinding()]
                param([string]$Name,[string]$Repository,[string]$Scope,[switch]$Force,[switch]$AllowClobber,[switch]$AcceptLicense)
                throw 'Unmocked synthetic package installation forbidden.'
            }
        }
        BeforeEach {
            $script:discoverySaved=@{}
            foreach($name in @('USERPROFILE','APPDATA','BW_CLIENTID','BW_CLIENTSECRET','BW_SESSION','PCAI_BW_EXECUTABLE','PCAI_BW_EXECUTABLE_SHA256','BWS_ACCESS_TOKEN')){$script:discoverySaved[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
            $script:discoveryRoot=Join-Path $script:FixtureDrive ([guid]::NewGuid().ToString('N'))
            Assert-CredentialPrivateDirectory $script:discoveryRoot
            $env:USERPROFILE=$script:discoveryRoot;$env:APPDATA=Join-Path $script:discoveryRoot 'AppData/Roaming'
            [IO.Directory]::CreateDirectory($env:APPDATA)|Out-Null
            $env:BW_CLIENTID=$null;$env:BW_CLIENTSECRET=$null;$env:BW_SESSION=$null;$env:PCAI_BW_EXECUTABLE=$null;$env:PCAI_BW_EXECUTABLE_SHA256=$null;$env:BWS_ACCESS_TOKEN=$null
        }
        AfterEach {foreach($name in $script:discoverySaved.Keys){[Environment]::SetEnvironmentVariable($name,$script:discoverySaved[$name],'Process')}}


        # Protects: the existing archive caller's finite deadline through CLI validation and the actual process runner.
        # Detects: supported deadlines above 120 seconds must not fail parameter binding before a fast owned child can run.
        # Needs: A hidden profile-free PowerShell child that exits immediately; no Bitwarden executable/provider.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,Update-BwArchive.ps1}: Invoke-BitwardenCli,Invoke-SecretBackendProcess
        It 'accepts archive-compatible finite deadline <Timeout> through the real bounded runner' -TestCases @(@{Timeout=121},@{Timeout=600}) {
            param($Timeout)
            Mock Get-BitwardenCliProcessSpec {[pscustomobject]@{Kind='native-executable';FilePath=$script:archiveDeadlineExecutable;ArgumentPrefix=@('-NoLogo','-NoProfile','-NonInteractive','-Command')}}
            Mock Invoke-SecretBackendProcess {[pscustomobject]@{Success=$true;StdOut='synthetic-cli-boundary'}}
            $result=Invoke-BitwardenCli -Arguments @("[Console]::Out.Write('synthetic-deadline-ready')") -TimeoutSeconds $Timeout
            $result.Success|Should -BeTrue;$result.StdOut|Should -BeExactly 'synthetic-cli-boundary'
            Should -Invoke Invoke-SecretBackendProcess -Times 1 -Exactly -ParameterFilter {$TimeoutSeconds -eq $Timeout}
            $native=& $script:archiveDeadlineRunner -FilePath $script:archiveDeadlineExecutable -Arguments @('-NoLogo','-NoProfile','-NonInteractive','-Command',"[Console]::Out.Write('synthetic-deadline-ready')") -TimeoutSeconds $Timeout
            $native.Success|Should -BeTrue;$native.StdOut|Should -BeExactly 'synthetic-deadline-ready'
        }
        # Protects: finite supported bounds and the independent short cmdkey limit.
        # Detects: zero or over-600 deadlines must be rejected before any child launch; cmdkey's existing 120-second limit stays unchanged.
        # Needs: Validation-only calls and process invocation counters.
        # Breadcrumb: Tools/SystemScripts/Machine/SecretBackendUtilities.ps1: Invoke-BitwardenCli,Invoke-SecretBackendProcess,Get-CmdKeyListOutput
        It 'refuses unsupported archive deadline <Timeout> before launch' -TestCases @(@{Timeout=0},@{Timeout=601}) {
            param($Timeout)
            {Invoke-BitwardenCli -Arguments status -TimeoutSeconds $Timeout}|Should -Throw
            {& $script:archiveDeadlineRunner -FilePath $script:archiveDeadlineExecutable -Arguments @('-NoLogo') -TimeoutSeconds $Timeout}|Should -Throw
            {Get-CmdKeyListOutput -Target fixture -TimeoutSeconds 121}|Should -Throw
            Should -Invoke Invoke-SecretBackendProcess -Times 0 -Exactly
        }

        # Protects: attests unique package files and refuses ambiguous or stale selections (<Tool>).
        # Detects: Stale, ambiguous or unattested command selection would invoke the wrong executable or cross the refused shell boundary.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-RealWinGetPackageExecutable
        It 'attests unique package files and refuses ambiguous or stale selections (<Tool>)' -TestCases @(@{Tool='bw'},@{Tool='rclone'}) {
            param($Tool)
            $root=Join-Path $env:USERPROFILE 'AppData/Local/Microsoft/WinGet/Packages'
            $name=if($Tool -eq 'bw'){'Bitwarden.CLI_synthetic'}else{'Rclone.Rclone_synthetic'}
            $directory=Join-Path $root $name
            if($Tool -eq 'rclone'){$directory=Join-Path $directory 'rclone-v99-windows-amd64'}
            [IO.Directory]::CreateDirectory($directory)|Out-Null
            $path=Join-Path $directory "$Tool.exe";[IO.File]::WriteAllText($path,'synthetic-never-executable')
            (Get-RealWinGetPackageExecutable -Tool $Tool)|Should -BeExactly $path
            (Get-RealWinGetPackageExecutable -Tool $Tool -ExecutablePath $path -ExpectedSha256 (Get-FileHash $path).Hash)|Should -BeExactly $path
            {Get-RealWinGetPackageExecutable -Tool $Tool -ExecutablePath $path -ExpectedSha256 ('0'*64)}|Should -Throw '*hash mismatch*'
            {Get-RealWinGetPackageExecutable -Tool $Tool -ExecutablePath $path}|Should -Throw '*SHA256*'
            $other=Join-Path $root ($name+'-second')
            if($Tool -eq 'rclone'){$other=Join-Path $other 'rclone-v98-windows-amd64'}
            [IO.Directory]::CreateDirectory($other)|Out-Null;[IO.File]::WriteAllText((Join-Path $other "$Tool.exe"),'synthetic-other')
            {Get-RealWinGetPackageExecutable -Tool $Tool}|Should -Throw '*Ambiguous*'
        }
        # Protects: keeps absent packages missing and refuses WinGet Links as attested packages.
        # Detects: Stale, ambiguous or unattested command selection would invoke the wrong executable or cross the refused shell boundary.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-RealWinGetPackageExecutable
        It 'keeps absent packages missing and refuses WinGet Links as attested packages' {
            Get-RealWinGetPackageExecutable -Tool bw|Should -BeNullOrEmpty
            $links=Join-Path $env:USERPROFILE 'AppData/Local/Microsoft/WinGet/Links';[IO.Directory]::CreateDirectory($links)|Out-Null
            $path=Join-Path $links 'bw.exe';[IO.File]::WriteAllText($path,'synthetic')
            {Get-RealWinGetPackageExecutable bw -ExecutablePath $path -ExpectedSha256 (Get-FileHash $path).Hash}|Should -Throw '*actual tool package*'
        }
        # Protects: selects discovery paths deterministically without executing them (<Kind>).
        # Detects: Stale, ambiguous or unattested command selection would invoke the wrong executable or cross the refused shell boundary.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-BitwardenCliProcessSpec
        It 'selects discovery paths deterministically without executing them (<Kind>)' -TestCases @(
            @{Kind='Native'},@{Kind='Package'},@{Kind='Explicit'},@{Kind='Node'},@{Kind='HomeCmd'},@{Kind='NpmCmd'},@{Kind='Command'},@{Kind='Missing'}
        ) {
            param($Kind)
            Mock Get-Command {if(($Kind -eq 'Native' -and $Name -eq 'bw.exe') -or ($Kind -eq 'Node' -and $Name -eq 'node') -or ($Kind -eq 'Command' -and $Name -eq 'bw')){[pscustomobject]@{Source="C:/synthetic/$Name"}}} -ParameterFilter {$Name -in @('bw.exe','node','bw')}
            Mock Get-RealWinGetPackageExecutable {if($Kind -in @('Package','Explicit')){'C:/synthetic/package/bw.exe'}}
            $path=switch($Kind){Node{Join-Path $env:APPDATA 'npm/node_modules/@bitwarden/cli/build/bw.js'}HomeCmd{Join-Path $env:USERPROFILE 'bin/bw.cmd'}NpmCmd{Join-Path $env:APPDATA 'npm/bw.cmd'}}
            if($path){[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($path))|Out-Null;[IO.File]::WriteAllText($path,'synthetic-never-run')}
            if($Kind -eq 'Explicit'){$env:PCAI_BW_EXECUTABLE='C:/synthetic/reviewed/bw.exe';$env:PCAI_BW_EXECUTABLE_SHA256='A'*64}
            $spec=Get-BitwardenCliProcessSpec
            $expected=switch($Kind){Native{'native-executable'}Package{'native-executable'}Explicit{'native-executable'}Node{'node-script'}HomeCmd{'home-bin-cmd'}NpmCmd{'cmd-shim'}Command{'command'}}
            if($Kind -eq 'Missing'){$spec|Should -BeNullOrEmpty}else{$spec.Kind|Should -BeExactly $expected}
            if($Kind -eq 'Node'){$spec.ArgumentPrefix.Count|Should -Be 1}
            Should -Invoke Invoke-SecretBackendProcess -Times 0 -Exactly
        }
        # Protects: launches only native or node specs through the guarded process boundary (<Kind>).
        # Detects: Stale, ambiguous or unattested command selection would invoke the wrong executable or cross the refused shell boundary.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Invoke-BitwardenCli
        It 'launches only native or node specs through the guarded process boundary (<Kind>)' -TestCases @(
            @{Kind='native-executable'},@{Kind='node-script'},@{Kind='home-bin-cmd'},@{Kind='cmd-shim'},@{Kind='command'},@{Kind='missing'}
        ) {
            param($Kind)
            Mock Get-BitwardenCliProcessSpec {if($Kind -ne 'missing'){[pscustomobject]@{Kind=$Kind;FilePath='C:/synthetic/not-run.exe';ArgumentPrefix=if($Kind -eq 'node-script'){@('C:/synthetic/bw.js')}else{@()}}}}
            Mock Invoke-SecretBackendProcess {[pscustomobject]@{Success=$true;StdOut='synthetic-status'}}
            if($Kind -in @('home-bin-cmd','cmd-shim','command')){
                {Invoke-BitwardenCli -Arguments status}|Should -Throw '*Shell-based*'
                Should -Invoke Invoke-SecretBackendProcess -Times 0 -Exactly
            }elseif($Kind -eq 'missing'){
                (Invoke-BitwardenCli -Arguments status).Success|Should -BeFalse
                Should -Invoke Invoke-SecretBackendProcess -Times 0 -Exactly
            }else{
                (Invoke-BitwardenCli -Arguments status -StandardInput synthetic-stdin -TimeoutSeconds 7).Success|Should -BeTrue
                Should -Invoke Invoke-SecretBackendProcess -Times 1 -Exactly -ParameterFilter {$StandardInput -eq 'synthetic-stdin' -and $TimeoutSeconds -eq 7}
            }
        }
        # Protects: reports only safe CLI status metadata (<RequestedState>).
        # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-BwStatusSafe
        It 'reports only safe CLI status metadata (<RequestedState>)' -TestCases @(
            @{RequestedState='locked'},@{RequestedState='unlocked'},@{RequestedState='unauthenticated'},@{RequestedState='missing'},@{RequestedState='timeout'},@{RequestedState='unknown'},@{RequestedState='invalid'}
        ) {
            param($RequestedState)
            Mock Invoke-BitwardenCli {
                if($RequestedState -in @('locked','unlocked','unauthenticated','invalid')){
                    [pscustomobject]@{Success=$true;StdOut=if($RequestedState -eq 'invalid'){'private-invalid-json'}else{'{"status":"'+$RequestedState+'"}'};ExitCode=0}
                }else{[pscustomobject]@{Success=$false;StdErr=if($RequestedState -eq 'missing'){'bw CLI is not available'}else{'private-stderr'};TimedOut=($RequestedState -eq 'timeout');ExitCode=1}}
            }
            $result=Get-BwStatusSafe
            $result.status|Should -BeExactly $(if($RequestedState -eq 'invalid'){'error'}else{$RequestedState})
            ($result|ConvertTo-Json)|Should -Not -Match 'private-'
        }
        # Protects: imports or installs only selected inert providers and honors WhatIf (<Mode>).
        # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Import-LocalSecretManagement
        It 'imports or installs only selected inert providers and honors WhatIf (<Mode>)' -TestCases @(@{Mode='Available'},@{Mode='Install'},@{Mode='WhatIf'}) {
            param($Mode)
            Mock Get-Module {if($Mode -eq 'Available'){[pscustomobject]@{Name=[string]$Name;Version=[version]'1.0';Path=('C:/synthetic/'+[string]$Name+'.psd1')}}} -ParameterFilter {$ListAvailable}
            Mock Import-Module {}
            Mock Install-Module {}
            $state=if($Mode -eq 'WhatIf'){Import-LocalSecretManagement -InstallMissing -WhatIf}else{Import-LocalSecretManagement -InstallMissing -Confirm:$false}
            if($Mode -eq 'Available'){$state.SecretManagementAvailable|Should -BeTrue;$state.Imported.Count|Should -Be 5;Should -Invoke Import-Module -Times 5 -Exactly;Should -Invoke Install-Module -Times 0 -Exactly}
            elseif($Mode -eq 'Install'){$state.SecretManagementAvailable|Should -BeFalse;Should -Invoke Install-Module -Times 4 -Exactly -ParameterFilter {$Repository -eq 'PSGallery' -and $Scope -eq 'CurrentUser'}}
            else{Should -Invoke Install-Module -Times 0 -Exactly}
        }
        # Protects: never combines incomplete API credential identities (<Source>).
        # Detects: Mixed credential sources or premature session acceptance would use the wrong identity or lose the prior process state.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Assert-CredentialPrivateDirectory, New-CredentialPrivateStage, Resolve-BwBootstrapState
        It 'never combines incomplete API credential identities (<Source>)' -TestCases @(@{Source='Vault'},@{Source='File'},@{Source='Process'},@{Source='Partial'}) {
            param($Source)
            Mock Import-LocalSecretManagement {[pscustomobject]@{SecretManagementAvailable=$true}}
            Mock Get-PlainTextSecretFromCandidateVaults {
                if($Source -eq 'Vault'){
                    if($Name -eq 'bitwarden/master_password'){return [pscustomobject]@{Value='synthetic-master';Vault='SecretStore'}}
                    if($VaultNames[0] -eq 'SecretStore'){return [pscustomobject]@{Value=("synthetic-"+$Name);Vault='SecretStore'}}
                }
            }
            $dir=Join-Path $env:USERPROFILE '.bwdata';Assert-CredentialPrivateDirectory $dir
            $stream=New-CredentialPrivateStage (Join-Path $dir 'bw_pass.txt') ([Text.Encoding]::UTF8.GetBytes(' synthetic-file-master '));$stream.Dispose()
            if($Source -eq 'File'){$stream=New-CredentialPrivateStage (Join-Path $dir 'secrets.json') ([Text.Encoding]::UTF8.GetBytes('{"client_id":"synthetic-file-id","client_secret":"synthetic-file-secret"}'));$stream.Dispose()}
            if($Source -in @('Process','Partial')){$env:BW_CLIENTID='synthetic-process-id'}
            if($Source -eq 'Process'){$env:BW_CLIENTSECRET='synthetic-process-secret'} # pragma: allowlist secret - isolated synthetic fixture; never a live credential
            $state=Resolve-BwBootstrapState
            if($Source -eq 'Vault'){$state.Password|Should -BeExactly 'synthetic-master';$state.ClientIdSource|Should -BeExactly 'vault:SecretStore';$state.ClientSecretSource|Should -BeExactly $state.ClientIdSource}
            else{$state.Password|Should -BeExactly 'synthetic-file-master'}
            switch($Source){
                File {$state.ClientId|Should -BeExactly 'synthetic-file-id';$state.ClientIdSource|Should -BeExactly $state.ClientSecretSource}
                Process {$state.ClientId|Should -BeExactly 'synthetic-process-id';$state.ClientSecret|Should -BeExactly 'synthetic-process-secret'}
                Partial {$state.ClientId|Should -BeNullOrEmpty;$state.ClientSecret|Should -BeNullOrEmpty}
            }
        }
        # Protects: normalizes explicit synthetic credential shapes without providers (<Shape>).
        # Detects: Wrong resource/target identity or an unobserved mapping outcome would be reported as a successful deployment.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Resolve-SecretCredentialInternal
        It 'normalizes explicit synthetic credential shapes without providers (<Shape>)' -TestCases @(
            @{Shape='PSCredential'},@{Shape='SecureString'},@{Shape='Json'},@{Shape='Plain'},@{Shape='Object'},@{Shape='Unknown'}
        ) {
            param($Shape)
            $secure=New-CustodySyntheticSecureString synthetic-value
            $value=switch($Shape){PSCredential{[pscredential]::new('explicit-user',$secure)}SecureString{$secure}Json{'{"username":"explicit-user","password":"synthetic-value"}'}Plain{'synthetic-value'}Object{[pscustomobject]@{username='explicit-user';password='synthetic-value'}}Unknown{[pscustomobject]@{other='unsupported'}}} # pragma: allowlist secret - isolated synthetic fixture; never a live credential
            $credential=Resolve-SecretCredentialInternal -Secret $value -DefaultUserName default-user
            if($Shape -eq 'Unknown'){$credential|Should -BeNullOrEmpty}else{$credential.Password|Should -BeExactly 'synthetic-value';$credential.UserName|Should -BeExactly $(if($Shape -in @('SecureString','Plain')){'default-user'}else{'explicit-user'})}
        }
        # Protects: refuses changed IDs and fuzzy or ambiguous resource names (<Mode>).
        # Detects: Wrong resource/target identity or an unobserved mapping outcome would be reported as a successful deployment.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Resolve-BitwardenItemForResource
        It 'refuses changed IDs and fuzzy or ambiguous resource names (<Mode>)' -TestCases @(
            @{Mode='Id'},@{Mode='ChangedId'},@{Mode='ReadFailure'},@{Mode='Name'},@{Mode='Fuzzy'},@{Mode='Ambiguous'}
        ) {
            param($Mode)
            Mock Invoke-BitwardenCli {[pscustomobject]@{Success=($Mode -ne 'ReadFailure');StdOut=switch($Mode){
                Id{'{"id":"synthetic-id","login":{"username":"fixture","password":"synthetic"}}'}
                ChangedId{'{"id":"changed","login":{"username":"fixture","password":"synthetic"}}'}
                Name{'[{"name":"exact","login":{"username":"fixture","password":"synthetic"}},{"name":"exact-fuzzy","login":{"username":"fixture","password":"wrong"}}]'}
                Fuzzy{'[{"name":"exact-fuzzy","login":{"username":"fixture","password":"wrong"}}]'}
                Ambiguous{'[{"name":"exact","login":{"username":"one","password":"synthetic"}},{"name":"exact","login":{"username":"two","password":"synthetic"}}]'}
                default{'{}'}
            }}}
            $resource=if($Mode -in @('Id','ChangedId','ReadFailure')){[pscustomobject]@{bitwardenItemId='synthetic-id'}}else{[pscustomobject]@{name='exact'}}
            if($Mode -in @('Id','Name')){(Resolve-BitwardenItemForResource $resource).Source|Should -BeExactly $(if($Mode -eq 'Id'){'id:synthetic-id'}else{'name:exact'})}
            else{{Resolve-BitwardenItemForResource $resource}|Should -Throw}
            Should -Invoke Invoke-BitwardenCli -Times 1 -Exactly
        }
        # Protects: parses only exact synthetic native credential targets (<Prefix>).
        # Detects: Wrong resource/target identity or an unobserved mapping outcome would be reported as a successful deployment.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-CredentialManagerCredential
        It 'parses only exact synthetic native credential targets (<Prefix>)' -TestCases @(@{Prefix='Domain:target='},@{Prefix='LegacyGeneric:target='},@{Prefix=''}) {
            param($Prefix)
            Mock Get-CmdKeyListOutput {(@('Currently stored credentials:','Target: wrong','Type: Generic','User: wrong',('Target: '+$Prefix+'Fixture-Target'),'Type: Domain Password','User: fixture-user') -join [Environment]::NewLine)}
            $r=Get-CredentialManagerCredential -Target ' fixture-target '
            $r.UserName|Should -BeExactly 'fixture-user';$r.Type|Should -BeExactly 'Domain Password';$r.RequestedTarget|Should -BeExactly ' fixture-target '
        }
        # Protects: withholds private CLI output from public command errors.
        # Detects: Failed or timed-out CLI stdout/stderr must not appear in the public error; only bounded exit/timeout metadata is reported.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Invoke-MachineBitwardenCommand
        It 'withholds private CLI output from public command errors' {
            Mock Invoke-BitwardenCli {[pscustomobject]@{Success=$false;ExitCode=3;TimedOut=$true;StdOut='private-sentinel';StdErr='private-sentinel'}}
            try{Invoke-MachineBitwardenCommand status;throw 'Expected refusal'}catch{$_.Exception.Message|Should -Match 'exit=3, timeout=True';$_.Exception.Message|Should -Not -Match 'private-sentinel'}
        }
    }


    Describe 'Synthetic local vault, connection and backend state interfaces' {
        # Protects passive metadata, explicit publication and guarded provider
        # registration. The SecretManagement namespace above is an inert module;
        # real native credential and external process boundaries stay mocked.
        BeforeAll {
            $script:interfaceModule=Import-Module (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretsTier.psm1') -ArgumentList @('fixture-machine') -Force -PassThru -ErrorAction Stop
        }
        BeforeEach {
            $script:interfaceSession=$env:BW_SESSION
            $script:interfaceToken=$env:BWS_ACCESS_TOKEN
            $env:BW_SESSION=$null;$env:BWS_ACCESS_TOKEN=$null
        }
        AfterEach {$env:BW_SESSION=$script:interfaceSession;$env:BWS_ACCESS_TOKEN=$script:interfaceToken}
        AfterAll {Remove-Module $script:interfaceModule -Force}

        # Protects: reports backend metadata without fetching credentials or importing providers.
        # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-SecretBackendStatus
        It 'reports backend metadata without fetching credentials or importing providers' {
            Mock Get-Module {[pscustomobject]@{Name=[string]$Name}} -ParameterFilter {$ListAvailable}
            Mock Get-BwStatusSafe {[pscustomobject]@{status='locked'}}
            Mock Get-BitwardenPasswordFile {'C:/synthetic/password-not-read'}
            Mock Get-BitwardenSessionFile {'C:/synthetic/session-not-read'}
            Mock 'Microsoft.PowerShell.SecretManagement\Get-SecretVault' {[pscustomobject]@{Name='synthetic-vault';ModuleName='synthetic';IsDefault=$true}}
            $r=Get-SecretBackendStatus
            $r.BwStatus|Should -BeExactly 'locked';$r.SecretManagementAvailable|Should -BeTrue
            $r.RegisteredVaults[0].Name|Should -BeExactly 'synthetic-vault'
            Should -Invoke 'Microsoft.PowerShell.SecretManagement\Get-Secret' -Times 0 -Exactly
        }
        # Protects: registers only requested and available synthetic provider backends.
        # Detects: Explicitly requested available providers must register exactly once, with only LocalStore selected as the default vault.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Initialize-LocalSecretBackends
        It 'registers only requested and available synthetic provider backends' {
            Mock Import-LocalSecretManagement {[pscustomobject]@{SecretManagementAvailable=$true;SecretStoreAvailable=$true;CredManStoreAvailable=$true;BitwardenVaultAvailable=$true}}
            Mock 'Microsoft.PowerShell.SecretManagement\Register-SecretVault' {}
            Mock Get-SecretBackendStatus {[pscustomobject]@{State='synthetic-only'}}
            (Initialize-LocalSecretBackends -RegisterLocalStore -RegisterCredManVault -RegisterBitwardenVault -Confirm:$false).State|Should -BeExactly 'synthetic-only'
            Should -Invoke 'Microsoft.PowerShell.SecretManagement\Register-SecretVault' -Times 3 -Exactly
            Should -Invoke 'Microsoft.PowerShell.SecretManagement\Register-SecretVault' -Times 1 -Exactly -ParameterFilter {$Name -eq 'LocalStore' -and $DefaultVault}
        }
        # Protects: does not re-register an existing synthetic vault or unavailable provider.
        # Detects: Already registered vaults and unavailable provider modules must produce no registration writes.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Initialize-LocalSecretBackends
        It 'does not re-register an existing synthetic vault or unavailable provider' {
            Mock Import-LocalSecretManagement {[pscustomobject]@{SecretManagementAvailable=$true;SecretStoreAvailable=$true;CredManStoreAvailable=$false;BitwardenVaultAvailable=$false}}
            Mock 'Microsoft.PowerShell.SecretManagement\Get-SecretVault' {[pscustomobject]@{Name='LocalStore'}}
            Mock 'Microsoft.PowerShell.SecretManagement\Register-SecretVault' {throw 'Unexpected registration.'}
            Mock Get-SecretBackendStatus {[pscustomobject]@{State='unchanged'}}
            (Initialize-LocalSecretBackends -RegisterLocalStore -RegisterCredManVault -RegisterBitwardenVault -Confirm:$false).State|Should -BeExactly 'unchanged'
            Should -Invoke 'Microsoft.PowerShell.SecretManagement\Register-SecretVault' -Times 0 -Exactly
        }
        # Protects: publishes a selected helper secret only to an admitted inert vault (<Scenario>).
        # Detects: Missing modules or vault registration must prevent secret publication, and provider write failures must not be reported as success.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Set-PlainTextSecretInSecretManagement
        It 'publishes a selected helper secret only to an admitted inert vault (<Scenario>)' -TestCases @(@{Scenario='Present'},@{Scenario='MissingModule'},@{Scenario='MissingVault'},@{Scenario='WriteFailure'}) {
            param($Scenario)
            Mock Import-LocalSecretManagement {[pscustomobject]@{SecretManagementAvailable=($Scenario -ne 'MissingModule')}}
            Mock 'Microsoft.PowerShell.SecretManagement\Get-SecretVault' {if($Scenario -ne 'MissingVault'){[pscustomobject]@{Name='LocalStore'}}}
            Mock 'Microsoft.PowerShell.SecretManagement\Set-Secret' {if($Scenario -eq 'WriteFailure'){throw 'synthetic-private-error'}}
            (Set-PlainTextSecretInSecretManagement fixture synthetic-only -Confirm:$false)|Should -Be ($Scenario -eq 'Present')
            if($Scenario -in @('MissingModule','MissingVault')){Should -Invoke 'Microsoft.PowerShell.SecretManagement\Set-Secret' -Times 0 -Exactly}
        }
        # Protects: returns the first usable candidate vault and never exposes failed provider output.
        # Detects: Blank, failing or empty candidates must be skipped; the first usable vault must stop later provider reads without disclosing earlier errors.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-PlainTextSecretFromCandidateVaults
        It 'returns the first usable candidate vault and never exposes failed provider output' {
            Mock 'Microsoft.PowerShell.SecretManagement\Get-Secret' {if($Vault -eq 'first'){throw 'private-sentinel'};if($Vault -eq 'selected'){'synthetic-only'}}
            $r=Get-PlainTextSecretFromCandidateVaults -Name fixture -VaultNames @('','first','empty','selected','later')
            $r.Value|Should -BeExactly 'synthetic-only';$r.Vault|Should -BeExactly 'selected'
            Should -Invoke 'Microsoft.PowerShell.SecretManagement\Get-Secret' -Times 0 -Exactly -ParameterFilter {$Vault -eq 'later'}
        }
        # Protects: reads only an admitted vault and normalizes its explicit synthetic payload (<Registered>).
        # Detects: An unregistered vault must perform no secret read; admitted synthetic JSON must preserve its selected username and password.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-SecretCredentialFromVaultInternal
        It 'reads only an admitted vault and normalizes its explicit synthetic payload (<Registered>)' -TestCases @(@{Registered=$true},@{Registered=$false}) {
            param($Registered)
            Mock 'Microsoft.PowerShell.SecretManagement\Get-SecretVault' {if($Registered){[pscustomobject]@{Name='fixture'}}}
            Mock 'Microsoft.PowerShell.SecretManagement\Get-Secret' {'{"username":"fixture-user","password":"synthetic-only"}'}
            $r=Get-SecretCredentialFromVaultInternal -SecretName fixture -VaultName fixture -DefaultUserName default
            if($Registered){$r.UserName|Should -BeExactly 'fixture-user';$r.Password|Should -BeExactly 'synthetic-only'}
            else{$r|Should -BeNullOrEmpty;Should -Invoke 'Microsoft.PowerShell.SecretManagement\Get-Secret' -Times 0 -Exactly}
        }
        # Protects: initializes and mirrors only the complete selected bootstrap values.
        # Detects: Mixed credential sources or premature session acceptance would use the wrong identity or lose the prior process state.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Initialize-BitwardenSession
        It 'initializes and mirrors only the complete selected bootstrap values' {
            Mock Initialize-BitwardenSessionFromBackends {[pscustomobject]@{Success=$true;Source='fixture';Message='synthetic-ready'}}
            Mock Resolve-BwBootstrapState {[pscustomobject]@{Password='synthetic-password';ClientId='synthetic-id';ClientSecret='synthetic-secret'}} # pragma: allowlist secret - isolated synthetic fixture; never a live credential
            Mock Set-PlainTextSecretInSecretManagement {$true}
            $r=Initialize-BitwardenSession -SyncBootstrapSecretsToLocalStore -Confirm:$false
            $r.Success|Should -BeTrue;$r.Source|Should -BeExactly 'fixture'
            Should -Invoke Set-PlainTextSecretInSecretManagement -Times 3 -Exactly -ParameterFilter {$VaultName -eq 'LocalStore'}
        }
        # Protects: registers a module vault through its inert provider interface (<Exists>).
        # Detects: A missing selected vault must register once; an existing registration must remain untouched.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Register-LocalSecretVault
        It 'registers a module vault through its inert provider interface (<Exists>)' -TestCases @(@{Exists=$true},@{Exists=$false}) {
            param($Exists)
            Mock -ModuleName SecretsTier Get-Module {[pscustomobject]@{Version=[version]'1.0'}} -ParameterFilter {$ListAvailable}
            Mock -ModuleName SecretsTier 'Microsoft.PowerShell.SecretManagement\Get-SecretVault' {if($Exists){[pscustomobject]@{Name='fixture'}}}
            Mock -ModuleName SecretsTier 'Microsoft.PowerShell.SecretManagement\Register-SecretVault' {}
            (& $script:interfaceModule {Register-LocalSecretVault -VaultName fixture -SetDefault -Confirm:$false})|Should -BeTrue
            Should -Invoke -ModuleName SecretsTier 'Microsoft.PowerShell.SecretManagement\Register-SecretVault' -Times $(if($Exists){0}else{1}) -Exactly
        }
        # Protects: projects module provider status from metadata without secret reads.
        # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-SecretManagementStatus
        It 'projects module provider status from metadata without secret reads' {
            Mock -ModuleName SecretsTier Get-Module {[pscustomobject]@{Version=[version]'1.0'}} -ParameterFilter {$ListAvailable}
            Mock -ModuleName SecretsTier 'Microsoft.PowerShell.SecretManagement\Get-SecretVault' {[pscustomobject]@{Name='fixture'}}
            Mock -ModuleName SecretsTier 'Microsoft.PowerShell.SecretManagement\Get-Secret' {throw 'Unexpected secret read.'}
            $r=& $script:interfaceModule {Get-SecretManagementStatus}
            $r.Available|Should -BeTrue;$r.RegisteredVaultCount|Should -Be 1
            Should -Invoke -ModuleName SecretsTier 'Microsoft.PowerShell.SecretManagement\Get-Secret' -Times 0 -Exactly
        }
        # Protects: publishes only explicit connection payload and correctly typed native target kinds.
        # Detects: Selected connection JSON, SecureString conversion and domain/generic target classification must stay exact without returning the password.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Publish-ConnectionCredential
        It 'publishes only explicit connection payload and correctly typed native target kinds' {
            Mock -ModuleName SecretsTier Set-PlainTextSecretInSecretManagement {
                $j=$Secret|ConvertFrom-Json
                $j.username|Should -BeExactly 'fixture-user';$j.password|Should -BeExactly 'synthetic-only'
                $true
            }
            Mock -ModuleName SecretsTier Write-NativeWindowsCredential {$Password|Should -BeOfType ([Security.SecureString])}
            $r=& $script:interfaceModule {Publish-ConnectionCredential -Name fixture -Username fixture-user -Password synthetic-only -Targets @('fixture-host','TERMSRV/fixture','fixture:445') -Confirm:$false}
            $r.SecretStored|Should -BeTrue;$r.CredentialTargets.Count|Should -Be 3
            Should -Invoke -ModuleName SecretsTier Write-NativeWindowsCredential -Times 1 -Exactly -ParameterFilter {$Kind -eq 'domain'}
            Should -Invoke -ModuleName SecretsTier Write-NativeWindowsCredential -Times 2 -Exactly -ParameterFilter {$Kind -eq 'generic'}
            ($r|ConvertTo-Json -Depth 8)|Should -Not -Match 'synthetic-only'
        }
        # Protects: reads only valid explicit connection JSON and treats missing or invalid data as absent (<Payload>).
        # Detects: Missing or malformed connection JSON must remain absent; valid selected JSON must preserve its explicit username.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-ConnectionCredential
        It 'reads only valid explicit connection JSON and treats missing or invalid data as absent (<Payload>)' -TestCases @(
            @{Payload='{"username":"fixture-user","password":"synthetic-only"}'},@{Payload='not-json'},@{Payload=''}
        ) {
            param($Payload)
            Mock -ModuleName SecretsTier Get-PlainTextSecretFromSecretManagement {$Payload}
            $r=& $script:interfaceModule {Get-ConnectionCredential fixture}
            if($Payload.StartsWith('{')){$r.username|Should -BeExactly 'fixture-user'}else{$r|Should -BeNullOrEmpty}
        }

        # Protects: sets selected module vault values with metadata through the inert provider only.
        # Detects: The explicit secret name and metadata must reach exactly one selected provider write.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Set-PlainTextSecretInSecretManagement
        It 'sets selected module vault values with metadata through the inert provider only' {
            Mock -ModuleName SecretsTier 'Microsoft.PowerShell.SecretManagement\Get-SecretVault' {[pscustomobject]@{Name='fixture'}}
            Mock -ModuleName SecretsTier 'Microsoft.PowerShell.SecretManagement\Set-Secret' {}
            (& $script:interfaceModule {Set-PlainTextSecretInSecretManagement -Name fixture -Secret synthetic-only -VaultName fixture -Metadata @{purpose='synthetic'} -Confirm:$false})|Should -BeTrue
            Should -Invoke -ModuleName SecretsTier 'Microsoft.PowerShell.SecretManagement\Set-Secret' -Times 1 -Exactly -ParameterFilter {$Name -eq 'fixture' -and $Metadata.purpose -eq 'synthetic'}
        }
        # Protects: withholds failed module provider reads and returns only explicit synthetic values (<Scenario>).
        # Detects: A failed provider read must return no value, while a successful selected read returns only its synthetic payload.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-PlainTextSecretFromSecretManagement
        It 'withholds failed module provider reads and returns only explicit synthetic values (<Scenario>)' -TestCases @(@{Scenario='Value'},@{Scenario='Failure'}) {
            param($Scenario)
            Mock -ModuleName SecretsTier 'Microsoft.PowerShell.SecretManagement\Get-Secret' {if($Scenario -eq 'Failure'){throw 'private-sentinel'};'synthetic-only'}
            $r=& $script:interfaceModule {Get-PlainTextSecretFromSecretManagement -Name fixture -VaultName fixture}
            if($Scenario -eq 'Value'){$r|Should -BeExactly 'synthetic-only'}else{$r|Should -BeNullOrEmpty}
        }
        # Protects: reads synthetic machine identity from the isolated module root without registry fallback.
        # Detects: A valid isolated identity file must determine machineId, while malformed content must return no invented identity.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Assert-CredentialPrivateDirectory, Get-MachineIdentityInternal
        It 'reads synthetic machine identity from the isolated module root without registry fallback' {
            $dir=& $script:interfaceModule {$script:MachineDir}
            $dir.StartsWith($script:FixtureDrive,[StringComparison]::OrdinalIgnoreCase)|Should -BeTrue
            Assert-CredentialPrivateDirectory $dir
            $p=Join-Path $dir 'identity.json'
            [IO.File]::WriteAllText($p,'{"machineId":"synthetic-identity","computerName":"fixture-machine"}')
            (& $script:interfaceModule {Get-MachineIdentityInternal}).machineId|Should -BeExactly 'synthetic-identity'
            [IO.File]::WriteAllText($p,'invalid-private-sentinel')
            (& $script:interfaceModule {Get-MachineIdentityInternal})|Should -BeNullOrEmpty
        }
        # Protects: projects cache and session metadata without decrypting or hydrating values.
        # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-SecretsStatus
        It 'projects cache and session metadata without decrypting or hydrating values' {
            Mock -ModuleName SecretsTier Get-CacheStatus {[pscustomobject]@{Exists=$false;IsValid=$false}}
            Mock -ModuleName SecretsTier Get-BwStatusSafe {[pscustomobject]@{status='locked'}}
            Mock -ModuleName SecretsTier Get-SecretsFromCache {throw 'Unexpected plaintext read.'}
            $r=& $script:interfaceModule {Get-SecretsStatus}
            $r.MachineName|Should -BeExactly 'fixture-machine';$r.BwStatus|Should -BeExactly 'locked';$r.Cache.Exists|Should -BeFalse
            Should -Invoke -ModuleName SecretsTier Get-SecretsFromCache -Times 0 -Exactly
        }
        # Protects: hydrates only a successful explicit sync and reports failure without environment mutation (<Succeeded>).
        # Detects: Failed explicit synchronization must perform no environment publication; successful synchronization must hydrate once.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Sync-MachineSecrets
        It 'hydrates only a successful explicit sync and reports failure without environment mutation (<Succeeded>)' -TestCases @(@{Succeeded=$true},@{Succeeded=$false}) {
            param($Succeeded)
            Mock -ModuleName SecretsTier Sync-SecretsFromBws {$Succeeded}
            Mock -ModuleName SecretsTier Get-SecretsFromCache {@{alpha='synthetic-only'}}
            Mock -ModuleName SecretsTier Set-SecretsAsEnvironment {}
            & $script:interfaceModule {Sync-MachineSecrets -Confirm:$false -ErrorAction SilentlyContinue}
            Should -Invoke -ModuleName SecretsTier Set-SecretsAsEnvironment -Times $(if($Succeeded){1}else{0}) -Exactly
        }
        # Protects: validates every environment name before any synthetic process assignment.
        # Detects: Any invalid key must prevent all assignments, and WhatIf must preserve a previously assigned synthetic process value.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Set-SecretsAsEnvironment
        It 'validates every environment name before any synthetic process assignment' {
            $before=$env:PCAI_CONSUMER_SYNTHETIC
            try{
                $env:PCAI_CONSUMER_SYNTHETIC=$null
                {& $script:interfaceModule {Set-SecretsAsEnvironment -Secrets @{PCAI_CONSUMER_SYNTHETIC='synthetic-only';'bad=name'='synthetic'} -Scope Process -Confirm:$false}}|Should -Throw '*Invalid environment variable name*'
                $env:PCAI_CONSUMER_SYNTHETIC|Should -BeNullOrEmpty
                & $script:interfaceModule {Set-SecretsAsEnvironment -Secrets @{PCAI_CONSUMER_SYNTHETIC='synthetic-only'} -Scope Process -Confirm:$false}
                $env:PCAI_CONSUMER_SYNTHETIC|Should -BeExactly 'synthetic-only'
                & $script:interfaceModule {Set-SecretsAsEnvironment -Secrets @{PCAI_CONSUMER_SYNTHETIC='must-not-apply'} -Scope Process -WhatIf}
                $env:PCAI_CONSUMER_SYNTHETIC|Should -BeExactly 'synthetic-only'
            }finally{$env:PCAI_CONSUMER_SYNTHETIC=$before}
        }
        # Protects: does not persist absent keys during initialized environment selection.
        # Detects: A missing persistent-key selection must not create a User-scope write while Process hydration still occurs once.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Set-InitializedSecretsEnvironment
        It 'does not persist absent keys during initialized environment selection' {
            Mock -ModuleName SecretsTier Set-SecretsAsEnvironment {}
            & $script:interfaceModule {Set-InitializedSecretsEnvironment -Secrets @{alpha='synthetic-only'} -PersistentKeys absent -Confirm:$false}
            Should -Invoke -ModuleName SecretsTier Set-SecretsAsEnvironment -Times 1 -Exactly -ParameterFilter {$Scope -eq 'Process' -and $Secrets.alpha -eq 'synthetic-only'}
        }

        # Protects: delegates guarded unlock and exposes metadata only.
        # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Unlock-BitwardenCliSession
        It 'delegates guarded unlock and exposes metadata only' {
            Mock -ModuleName SecretsTier Initialize-BitwardenSessionFromBackends {[pscustomobject]@{Success=$true;Source='synthetic-source'}}
            $r=& $script:interfaceModule {Unlock-BitwardenCliSession -Confirm:$false}
            $r.Unlocked|Should -BeTrue;$r.Reason|Should -BeExactly 'synthetic-source'
            $r.PasswordSource|Should -BeNullOrEmpty;$r.ClientSource|Should -BeNullOrEmpty
        }
    }

    Describe 'Synthetic resource sync decisions' {
        BeforeEach {
            $script:syncManifest=[pscustomobject]@{resources=@(
                [pscustomobject]@{name='alpha';preferredUsername='preferred-user';secretName='synthetic/alpha';targets=@(
                    [pscustomobject]@{target='synthetic-good';kind='domain';username='override-user'},
                    [pscustomobject]@{target='synthetic-fail';kind='other';username=$null})},
                [pscustomobject]@{name='beta';preferredUsername='not-selected';secretName=$null;targets=@()})}
            Mock Get-ResourceCredentialManifest {$script:syncManifest}
            Mock Import-LocalSecretManagement {[pscustomobject]@{SecretManagementAvailable=$false}}
            Mock Initialize-BitwardenSessionFromBackends {[pscustomobject]@{Success=$true;Message='synthetic-ready'}}
            Mock Get-BitwardenCliProcessSpec {[pscustomobject]@{Kind='synthetic-never-launched'}}
            Mock Invoke-BitwardenCli {[pscustomobject]@{Success=$true}}
            Mock Resolve-BitwardenItemForResource {[pscustomobject]@{Source='id:synthetic';Item=[pscustomobject]@{login=[pscustomobject]@{username='item-user';password='test-synthetic-only'}}}}
            Mock Set-PlainTextSecretInSecretManagement {throw 'Unexpected mirror.'}
            Mock Set-CredentialManagerCredential -RemoveParameterType Password {$Target -eq 'synthetic-good'}
        }
        # Protects: filters case-insensitively and preserves target precedence, status and redaction.
        # Detects: Wrong resource/target identity or an unobserved mapping outcome would be reported as a successful deployment.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Sync-ResourceCredentialsFromBitwarden
        It 'filters case-insensitively and preserves target precedence, status and redaction' {
            $rows=@(Sync-ResourceCredentialsFromBitwarden -ResourceNames @('ALPHA',' ') -Confirm:$false)
            $rows.Count|Should -Be 1;$rows[0].Name|Should -BeExactly 'alpha';$rows[0].Success|Should -BeFalse
            $rows[0].Targets.Count|Should -Be 2
            $rows[0].Targets[0].Status|Should -BeExactly 'updated';$rows[0].Targets[1].Status|Should -BeExactly 'failed'
            Should -Invoke Resolve-BitwardenItemForResource -Times 1 -Exactly -ParameterFilter {$Resource.name -eq 'alpha'}
            Should -Invoke Set-CredentialManagerCredential -Times 1 -Exactly -ParameterFilter {$Target -eq 'synthetic-good' -and $UserName -eq 'override-user' -and $Kind -eq 'domain'}
            Should -Invoke Set-CredentialManagerCredential -Times 1 -Exactly -ParameterFilter {$Target -eq 'synthetic-fail' -and $UserName -eq 'preferred-user' -and $Kind -eq 'generic'}
            Should -Invoke Set-PlainTextSecretInSecretManagement -Times 0 -Exactly
            ($rows|ConvertTo-Json -Depth 10)|Should -Not -Match 'test-synthetic-only'
        }
        # Protects: refuses deployment after failed Bitwarden sync.
        # Detects: Wrong resource/target identity or an unobserved mapping outcome would be reported as a successful deployment.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Sync-ResourceCredentialsFromBitwarden
        It 'refuses deployment after failed Bitwarden sync' {
            Mock Invoke-BitwardenCli {[pscustomobject]@{Success=$false}}
            {Sync-ResourceCredentialsFromBitwarden -Confirm:$false}|Should -Throw '*sync failed*'
            Should -Invoke Resolve-BitwardenItemForResource -Times 0 -Exactly
            Should -Invoke Set-CredentialManagerCredential -Times 0 -Exactly
        }
        # Protects: performs no credential deployment when prerequisite <Missing> is absent.
        # Detects: Wrong resource/target identity or an unobserved mapping outcome would be reported as a successful deployment.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Sync-ResourceCredentialsFromBitwarden
        It 'performs no credential deployment when prerequisite <Missing> is absent' -TestCases @(
            @{Missing='Manifest'},@{Missing='Session'},@{Missing='Cli'},@{Missing='RequestedResource'}
        ) {
            param($Missing)
            switch($Missing){
                Manifest {Mock Get-ResourceCredentialManifest {$null}}
                Session {Mock Initialize-BitwardenSessionFromBackends {[pscustomobject]@{Success=$false;Message='synthetic-unavailable'}}}
                Cli {Mock Get-BitwardenCliProcessSpec {$null}}
            }
            $names=if($Missing -eq 'RequestedResource'){@('absent')}else{@('alpha')}
            $rows=@(Sync-ResourceCredentialsFromBitwarden -ResourceNames $names -Confirm:$false)
            if($Missing -eq 'RequestedResource'){$rows.Count|Should -Be 0}else{$rows[0].Success|Should -BeFalse}
            Should -Invoke Invoke-BitwardenCli -Times 0 -Exactly
            Should -Invoke Set-CredentialManagerCredential -Times 0 -Exactly
        }
        # Protects: returns usable missing-item/password/username metadata without leaking values (<Missing>).
        # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Sync-ResourceCredentialsFromBitwarden
        It 'returns usable missing-item/password/username metadata without leaking values (<Missing>)' -TestCases @(
            @{Missing='Item'},@{Missing='Password'},@{Missing='Username'}
        ) {
            param($Missing)
            switch($Missing){
                Item {Mock Resolve-BitwardenItemForResource {$null}}
                Password {Mock Resolve-BitwardenItemForResource {[pscustomobject]@{Source='fixture';Item=[pscustomobject]@{login=[pscustomobject]@{username='fixture';password=''}}}}}
                Username {
                    $script:syncManifest.resources[0].preferredUsername=$null
                    $script:syncManifest.resources[0].targets[0].username=$null
                    Mock Resolve-BitwardenItemForResource {[pscustomobject]@{Source='fixture';Item=[pscustomobject]@{login=[pscustomobject]@{username=$null;password='synthetic-only'}}}} # pragma: allowlist secret - isolated synthetic fixture; never a live credential
                }
            }
            $rows=@(Sync-ResourceCredentialsFromBitwarden -ResourceNames alpha -Confirm:$false)
            $rows[0].Success|Should -BeFalse
            if($Missing -eq 'Username'){$rows[0].Targets[0].Status|Should -BeExactly 'missing-username'}
            Should -Invoke Set-CredentialManagerCredential -Times 0 -Exactly
        }
        # Protects: mirrors only selected synthetic credentials when the inert vault is registered.
        # Detects: Wrong resource/target identity or an unobserved mapping outcome would be reported as a successful deployment.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Sync-ResourceCredentialsFromBitwarden
        It 'mirrors only selected synthetic credentials when the inert vault is registered' {
            Mock Import-LocalSecretManagement {[pscustomobject]@{SecretManagementAvailable=$true}}
            Mock 'Microsoft.PowerShell.SecretManagement\Get-SecretVault' {[pscustomobject]@{Name='LocalStore'}}
            Mock Set-PlainTextSecretInSecretManagement {$true}
            Mock Set-CredentialManagerCredential -RemoveParameterType Password {$true}
            $r=@(Sync-ResourceCredentialsFromBitwarden -ResourceNames alpha -Confirm:$false)[0]
            $r.Success|Should -BeTrue;$r.LocalSecretStored|Should -BeTrue
            Should -Invoke Set-PlainTextSecretInSecretManagement -Times 1 -Exactly -ParameterFilter {
                $Name -eq 'synthetic/alpha' -and ($Secret|ConvertFrom-Json).username -eq 'preferred-user'
            }
        }
        # Protects: WhatIf does not discover config, authenticate or synchronize.
        # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Sync-ResourceCredentialsFromBitwarden
        It 'WhatIf does not discover config, authenticate or synchronize' {
            @(Sync-ResourceCredentialsFromBitwarden -WhatIf).Count|Should -Be 0
            Should -Invoke Get-ResourceCredentialManifest -Times 0 -Exactly
            Should -Invoke Initialize-BitwardenSessionFromBackends -Times 0 -Exactly
        }
    }

    Describe 'Synthetic resource reconciliation projection' {
        BeforeEach {
            $script:projection=[pscustomobject]@{resources=@([pscustomobject]@{
                name='synthetic-resource';resourceType='smb';bitwardenItemId='synthetic-id';secretName='fixture/secret';preferredUsername='fixture-user';expectedDriveLetter='Z:';sharePathHint='\\fixture-host\share';sharePathCandidates=@('\\fixture-host\share','\\second-host\share');notes='synthetic-only';targets=@(
                    [pscustomobject]@{target='present';kind='domain'},[pscustomobject]@{target='absent';kind='generic'})})}
            Mock Get-ResourceCredentialManifest {$script:projection}
            Mock Import-LocalSecretManagement {[pscustomobject]@{SecretManagementAvailable=$false}}
            Mock Get-CredentialManagerCredential {if($Target -eq 'present'){[pscustomobject]@{Type=2;UserName='existing-user'}}}
            Mock Get-PSDrive {[pscustomobject]@{Root='Z:\';DisplayRoot='\\fixture-host\share'}}
        }
        # Protects: projects native metadata and matching display root without deployment.
        # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-ResourceCredentialReconciliationReport
        It 'projects native metadata and matching display root without deployment' {
            $report=Get-ResourceCredentialReconciliationReport -ConfigPath 'synthetic-config'
            $report.Vault.Ready|Should -BeFalse;$report.Vault.Message|Should -BeExactly 'SecretManagement is not available locally.'
            @($report.Resources).Count|Should -Be 1;$r=$report.Resources[0]
            $r.LocalSecretPresent|Should -BeFalse;$r.Drive.Present|Should -BeTrue;$r.Drive.MatchesHint|Should -BeTrue
            @($r.SharePathCandidates).Count|Should -Be 2
            $r.CredentialTargets[0].Exists|Should -BeTrue;$r.CredentialTargets[0].CredentialType|Should -Be 2;$r.CredentialTargets[0].UserName|Should -BeExactly 'existing-user'
            $r.CredentialTargets[1].Exists|Should -BeFalse;$r.CredentialTargets[1].CredentialType|Should -BeNullOrEmpty
            Should -Invoke Get-PSDrive -Times 1 -Exactly -ParameterFilter {$Name -eq 'Z'}
        }
        # Protects: does not invent matches or candidates for missing projection fields (<Field>).
        # Detects: Wrong resource/target identity or an unobserved mapping outcome would be reported as a successful deployment.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-ResourceCredentialReconciliationReport
        It 'does not invent matches or candidates for missing projection fields (<Field>)' -TestCases @(
            @{Field='Drive'},@{Field='Candidates'},@{Field='Hint'},@{Field='Manifest'}
        ) {
            param($Field)
            switch($Field){
                Drive {Mock Get-PSDrive {$null}}
                Candidates {$script:projection.resources[0].PSObject.Properties.Remove('sharePathCandidates')}
                Hint {$script:projection.resources[0].PSObject.Properties.Remove('sharePathCandidates');$script:projection.resources[0].PSObject.Properties.Remove('sharePathHint')}
                Manifest {Mock Get-ResourceCredentialManifest {$null}}
            }
            $r=Get-ResourceCredentialReconciliationReport
            switch($Field){
                Drive {$r.Resources[0].Drive.Present|Should -BeFalse;$r.Resources[0].Drive.MatchesHint|Should -BeNullOrEmpty}
                Candidates {@($r.Resources[0].SharePathCandidates).Count|Should -Be 1}
                Hint {@($r.Resources[0].SharePathCandidates).Count|Should -Be 0;$r.Resources[0].Drive.MatchesHint|Should -BeNullOrEmpty}
                Manifest {@($r).Count|Should -Be 0}
            }
        }
        # Protects: projects inert registered-vault metadata without fetching plaintext.
        # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-ResourceCredentialReconciliationReport
        It 'projects inert registered-vault metadata without fetching plaintext' {
            Mock Import-LocalSecretManagement {[pscustomobject]@{SecretManagementAvailable=$true}}
            Mock 'Microsoft.PowerShell.SecretManagement\Get-SecretVault' {[pscustomobject]@{Name='LocalStore'}}
            Mock 'Microsoft.PowerShell.SecretManagement\Get-SecretInfo' {[pscustomobject]@{Name='fixture/secret'}}
            $r=Get-ResourceCredentialReconciliationReport
            $r.Vault.Ready|Should -BeTrue;$r.Resources[0].LocalSecretPresent|Should -BeTrue
            Should -Invoke 'Microsoft.PowerShell.SecretManagement\Get-Secret' -Times 0 -Exactly
        }
    }

    Describe 'Synthetic drive mapping custody decisions' {
        BeforeEach {
            $script:share='\\fixture-host\share';$script:driveReads=0
            Mock Get-ResourceCredentialManifest {[pscustomobject]@{resources=@([pscustomobject]@{name='fixture';expectedDriveLetter='Z:';sharePathHint=$script:share})}}
            Mock Resolve-ResourceSharePathCandidate {[pscustomobject]@{SelectedPath=$script:share;Probes=@([pscustomobject]@{Path=$script:share;Host='fixture-host';Reachable=$true})}}
            Mock Get-PSDrive {$script:driveReads++;if($script:driveReads -gt 1){[pscustomobject]@{Root=$script:share;DisplayRoot=$null}}}
            Mock Invoke-SecretBackendProcess {[pscustomobject]@{Success=$true}}
        }
        # Protects: publishes only after the exact mapping postcondition is observed.
        # Detects: Wrong resource/target identity or an unobserved mapping outcome would be reported as a successful deployment.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Repair-ResourceDriveMappings
        It 'publishes only after the exact mapping postcondition is observed' {
            $r=@(Repair-ResourceDriveMappings -Confirm:$false)
            $r.Count|Should -Be 1;$r[0].Success|Should -BeTrue;$r[0].Status|Should -BeExactly 'mapped'
            Should -Invoke Get-PSDrive -Times 2 -Exactly -ParameterFilter {$Name -eq 'Z'}
            Should -Invoke Invoke-SecretBackendProcess -Times 1 -Exactly -ParameterFilter {$FilePath -eq (Join-Path $env:SystemRoot 'System32/net.exe') -and ($Arguments -join ',') -eq 'use,Z:,\\fixture-host\share,/persistent:yes'}
        }
        # Protects: refuses replacement of a different mapping or a false observed success (<State>).
        # Detects: Wrong resource/target identity or an unobserved mapping outcome would be reported as a successful deployment.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Repair-ResourceDriveMappings
        It 'refuses replacement of a different mapping or a false observed success (<State>)' -TestCases @(@{State='Existing'},@{State='Postcondition'}) {
            param($State)
            Mock Get-PSDrive {$script:driveReads++;if($State -eq 'Existing' -or $script:driveReads -gt 1){[pscustomobject]@{Root='\\wrong-fixture\share';DisplayRoot=$null}}}
            {Repair-ResourceDriveMappings -Confirm:$false}|Should -Throw
            if($State -eq 'Existing'){Should -Invoke Invoke-SecretBackendProcess -Times 0 -Exactly}
        }
        # Protects: does not invoke a native mapper for existing exact Root or DisplayRoot (<Property>).
        # Detects: Wrong resource/target identity or an unobserved mapping outcome would be reported as a successful deployment.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Repair-ResourceDriveMappings
        It 'does not invoke a native mapper for existing exact Root or DisplayRoot (<Property>)' -TestCases @(@{Property='Root'},@{Property='DisplayRoot'}) {
            param($Property)
            Mock Get-PSDrive {if($Property -eq 'Root'){[pscustomobject]@{Root=$script:share;DisplayRoot=$null}}else{[pscustomobject]@{Root='Z:\';DisplayRoot=$script:share}}}
            $r=@(Repair-ResourceDriveMappings -Confirm:$false)[0]
            $r.Status|Should -BeExactly 'already-mounted';$r.Success|Should -BeTrue
            Should -Invoke Invoke-SecretBackendProcess -Times 0 -Exactly
        }
        # Protects: preserves unreachable or failed mapping status without claiming deployment (<State>).
        # Detects: Wrong resource/target identity or an unobserved mapping outcome would be reported as a successful deployment.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Repair-ResourceDriveMappings
        It 'preserves unreachable or failed mapping status without claiming deployment (<State>)' -TestCases @(@{State='Unreachable'},@{State='NativeFailure'}) {
            param($State)
            if($State -eq 'Unreachable'){Mock Resolve-ResourceSharePathCandidate {[pscustomobject]@{SelectedPath=$null;Probes=@([pscustomobject]@{Path='not-unc';Host=$null;Reachable=$false})}}}
            else{Mock Invoke-SecretBackendProcess {[pscustomobject]@{Success=$false}}}
            $r=@(Repair-ResourceDriveMappings -Confirm:$false)[0]
            $r.Success|Should -BeFalse
            $r.Status|Should -BeExactly $(if($State -eq 'Unreachable'){'unreachable'}else{'failed'})
            if($State -eq 'Unreachable'){Should -Invoke Invoke-SecretBackendProcess -Times 0 -Exactly}
        }
        # Protects: WhatIf performs no manifest discovery or mapping.
        # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Repair-ResourceDriveMappings
        It 'WhatIf performs no manifest discovery or mapping' {
            @(Repair-ResourceDriveMappings -WhatIf).Count|Should -Be 0
            Should -Invoke Get-ResourceCredentialManifest -Times 0 -Exactly
            Should -Invoke Invoke-SecretBackendProcess -Times 0 -Exactly
        }
    }

    Describe 'Ordered share discovery and exact vault targets' {
        # Protects: deduplicates candidate paths and stops at the first reachable synthetic endpoint.
        # Detects: Wrong resource/target identity or an unobserved mapping outcome would be reported as a successful deployment.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Resolve-ResourceSharePathCandidate
        It 'deduplicates candidate paths and stops at the first reachable synthetic endpoint' {
            Mock Test-TcpEndpointReachable {$HostName -eq 'second-fixture'}
            $resource=[pscustomobject]@{sharePathHint='\\first-fixture\share';sharePathCandidates=@(' ','\\first-fixture\share','not-unc','\\second-fixture\share','\\third-fixture\share')}
            $r=Resolve-ResourceSharePathCandidate $resource -TimeoutMilliseconds 7
            $r.SelectedPath|Should -BeExactly '\\second-fixture\share'
            @($r.Probes).Count|Should -Be 3
            Should -Invoke Test-TcpEndpointReachable -Times 2 -Exactly
            Should -Invoke Test-TcpEndpointReachable -Times 0 -Exactly -ParameterFilter {$HostName -eq 'third-fixture'}
        }
        # Protects: keeps empty discovery unresolved without issuing network traffic.
        # Detects: Wrong resource/target identity or an unobserved mapping outcome would be reported as a successful deployment.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Resolve-ResourceSharePathCandidate
        It 'keeps empty discovery unresolved without issuing network traffic' {
            $r=Resolve-ResourceSharePathCandidate ([pscustomobject]@{name='empty'})
            $r.SelectedPath|Should -BeNullOrEmpty;@($r.Probes).Count|Should -Be 0
            Should -Invoke Test-TcpEndpointReachable -Times 0 -Exactly
        }
        # Protects: deploys synthetic vault target outcomes and catches private write failures.
        # Detects: Wrong resource/target identity or an unobserved mapping outcome would be reported as a successful deployment.
        # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
        # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Sync-ResourceCredentialTargetsFromVault
        It 'deploys synthetic vault target outcomes and catches private write failures' {
            $path=Join-Path $script:FixtureDrive ('synthetic-resource-'+[guid]::NewGuid().ToString('N')+'.json')
            $config=@{resources=@(
                @{name='skip';secretName=$null},
                @{name='missing';secretName='missing';preferredUsername='fixture'},
                @{name='targets';secretName='present';preferredUsername='fixture';targets=@(
                    @{target='ok';username='override';kind='domain'},@{target='fail';username=$null;kind='generic'},@{target='throw';username=$null;kind='other'})})}
            [IO.File]::WriteAllText($path,($config|ConvertTo-Json -Depth 8))
            Mock Get-SecretCredentialFromVaultInternal {if($SecretName -eq 'present'){[pscustomobject]@{UserName='fixture';Password=(New-CustodySyntheticSecureString synthetic-only)}}}
            Mock Set-CredentialManagerCredential -RemoveParameterType Password {if($Target -eq 'throw'){throw 'private-sentinel'};$Target -eq 'ok'}
            $rows=@(Sync-ResourceCredentialTargetsFromVault -ConfigPath $path -Confirm:$false)
            ($rows.Status -join ',')|Should -BeExactly 'skipped,missing-secret,updated,failed,failed'
            ($rows|ConvertTo-Json)|Should -Not -Match 'private-sentinel|synthetic-only'
            Should -Invoke Set-CredentialManagerCredential -Times 1 -Exactly -ParameterFilter {$Target -eq 'ok' -and $UserName -eq 'override' -and $Kind -eq 'domain'}
        }
    }
}

Describe 'Private publication and recovery' {
BeforeAll {
    . (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretBackendUtilities.ps1')
    Initialize-CredentialFileInterop
    function New-PublicationFixtureFile {
        param([string]$Path,[string]$Value='original-synthetic',[switch]$DifferentAcl)
        $acl=[Security.AccessControl.FileSecurity]::new()
        $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User
        $acl.SetOwner($sid);$acl.SetAccessRuleProtection($true,$false)
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($sid,[Security.AccessControl.FileSystemRights]::FullControl,[Security.AccessControl.AccessControlType]::Allow))
        if($DifferentAcl){$acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-5-18'),[Security.AccessControl.FileSystemRights]::Read,[Security.AccessControl.AccessControlType]::Allow))}
        $file=[IO.FileSystemAclExtensions]::Create([IO.FileInfo]::new($Path),[IO.FileMode]::CreateNew,[Security.AccessControl.FileSystemRights]::Write,[IO.FileShare]::None,4096,[IO.FileOptions]::None,$acl)
        try{$bytes=[Text.Encoding]::UTF8.GetBytes($Value);$file.Write($bytes);$file.Flush($true)}finally{$file.Dispose()}
    }
    function Get-PublicationFixtureState {
        param([string]$Path)
        $file=Open-CredentialOwnedFile $Path
        try{$state=Read-CredentialOwnedSnapshot $file;return [pscustomobject]@{Identity=$state.Identity;Hash=$state.Hash;Sddl=(Get-CredentialDescriptorSddl $state.Descriptor)}}finally{$file.Dispose()}
    }
    function Assert-PublicationOriginal {
        $now=Get-PublicationFixtureState $script:target
        $now.Hash|Should -BeExactly $script:before.Hash
        $now.Sddl|Should -BeExactly $script:before.Sddl
    }
    function New-PublicationLaterWriter {
        param([string]$LaterText='later-synthetic')
        $later=Join-Path $script:caseRoot ([guid]::NewGuid().ToString('N')+'.bin')
        New-PublicationFixtureFile -Path $later -Value $LaterText -DifferentAcl
        # MoveFileEx overwrite refuses this retained Windows handle. A real
        # writer can instead rename the delete-shared object, then publish its
        # distinct file into the vacant name; retain both synthetic objects.
        if([IO.File]::Exists($script:target)){
            [IO.File]::Move($script:target,($later+'.retired'))
        }
        [IO.File]::Move($later,$script:target,$false)
        $script:laterState=Get-PublicationFixtureState $script:target
    }
}
Describe 'Private successor credential publication with actual Windows file custody' {
    BeforeEach {
        $script:caseRoot=Join-Path $script:FixtureDrive ([guid]::NewGuid().ToString('N'))
        $null=New-Item -ItemType Directory $caseRoot
        $script:target=Join-Path $caseRoot 'credential.fixture'
        New-PublicationFixtureFile $target
        $script:before=Get-PublicationFixtureState $target
        $script:failure=$null
        $script:laterState=$null
        $script:recoveryDenied=$false
        $script:descriptorTrace=[Collections.Generic.List[string]]::new()
    }
    # Protects: successful creation verifies private output and retires redundant transaction artifacts.
    # Detects: a validated replacement must not retain unrequested secret backups or receipts indefinitely.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'creates a genuinely private file without retaining successful transaction artifacts' {
        $new=Join-Path $caseRoot 'new.fixture'
        Write-BitwardenPrivateFile $new 'synthetic-new' -Confirm:$false
        [IO.File]::ReadAllText($new)|Should -BeExactly 'synthetic-new'
        (Get-Acl $new).AreAccessRulesProtected|Should -BeTrue
        @(Get-ChildItem (Join-Path $caseRoot '.pcai-private-write') -File -Recurse|Where-Object Name -NE 'publication.lock').Count|Should -Be 0
        @(Get-ChildItem (Join-Path $caseRoot '.pcai-private-write') -Directory -Recurse|Where-Object Name -Match '^r-[a-f0-9]{32}$').Count|Should -Be 0
    }
    # Protects: successful overwrite verifies restrictive ACL and retires its displaced original copy.
    # Detects: validated writes must not accumulate redundant secret-bearing backups.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Assert-BitwardenPrivateInput, Write-BitwardenPrivateFile
    It 'overwrites with restrictive private ACL without retaining a redundant displaced original' {
        Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false
        [IO.File]::ReadAllText($target)|Should -BeExactly 'synthetic-replacement'
        (Get-Acl -LiteralPath $target).AreAccessRulesProtected|Should -BeTrue
        Assert-BitwardenPrivateInput -Path $target
        @(Get-ChildItem (Join-Path $caseRoot '.pcai-private-write') -File -Recurse|Where-Object Name -NE 'publication.lock').Count|Should -Be 0
        @(Get-ChildItem (Join-Path $caseRoot '.pcai-private-write') -Directory -Recurse|Where-Object Name -Match '^r-[a-f0-9]{32}$').Count|Should -Be 0
    }

    # Protects: unknown or changed transaction custody from success-path deletion.
    # Detects: post-publication substitutions, links or unsupported attributes must preserve every artifact and warn.
    # Needs: Actual NTFS publication and isolated injected synthetic artifacts, no live credentials.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretBackendUtilities.ps1: Remove-CredentialPublishedCustody,Write-BitwardenPrivateFile
    It 'preserves complete successful custody when cleanup sees <Mode>' -TestCases @(
        @{Mode='Unknown'},@{Mode='Changed'},@{Mode='Identity'},@{Mode='Hardlink'},@{Mode='ReadOnly'},
        @{Mode='ArtifactAcl'},@{Mode='TargetAcl'}
    ){
        param($Mode)
        $script:cleanupInjection=$Mode
        $script:stageImplementation=(Get-Command New-CredentialPrivateStage).ScriptBlock
        Mock New-CredentialPrivateStage {
            param($Path,$Bytes)
            $stream=& $script:stageImplementation -Path $Path -Bytes $Bytes
            if([IO.Path]::GetFileName($Path) -eq 'receipt.json'){
                $directory=[IO.Path]::GetDirectoryName($Path)
                $displaced=Join-Path $directory 'actual-displaced.bin'
                switch($script:cleanupInjection){
                    Unknown {
                        $extra=& $script:stageImplementation -Path (Join-Path $directory 'unknown.bin') -Bytes ([Text.Encoding]::UTF8.GetBytes('synthetic-unknown'))
                        $extra.Dispose()
                    }
                    Changed {[IO.File]::WriteAllText($displaced,'synthetic-changed')}
                    Identity {
                        $saved=[IO.File]::ReadAllBytes($displaced)
                        [IO.File]::Move($displaced,(Join-Path $script:caseRoot 'owned-retired-original.bin'))
                        $replacement=& $script:stageImplementation -Path $displaced -Bytes $saved
                        $replacement.Dispose()
                    }
                    Hardlink {New-Item -ItemType HardLink -Path (Join-Path $script:caseRoot 'owned-hardlink.bin') -Target $displaced|Out-Null}
                    ReadOnly {[IO.File]::SetAttributes($displaced,[IO.FileAttributes]::ReadOnly)}
                    ArtifactAcl {
                        $acl=Get-Acl -LiteralPath $displaced
                        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.WindowsIdentity]::GetCurrent().User,[Security.AccessControl.FileSystemRights]::WriteData,[Security.AccessControl.AccessControlType]::Deny))
                        Set-Acl -LiteralPath $displaced -AclObject $acl
                    }
                    TargetAcl {
                        $acl=Get-Acl -LiteralPath $script:target
                        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.WindowsIdentity]::GetCurrent().User,[Security.AccessControl.FileSystemRights]::WriteData,[Security.AccessControl.AccessControlType]::Deny))
                        Set-Acl -LiteralPath $script:target -AclObject $acl
                    }
                }
            }
            return $stream
        }
        $warnings=@()
        Write-BitwardenPrivateFile $target 'synthetic-qualified-new' -Overwrite -Confirm:$false -WarningVariable warnings -WarningAction SilentlyContinue
        [IO.File]::ReadAllText($target)|Should -BeExactly 'synthetic-qualified-new'
        @($warnings).Count|Should -Be 1
        $warnings[0].ToString()|Should -BeExactly 'Private file published; successful transaction cleanup was not confirmed. Unqualified custody was preserved.'
        $transactions=@(Get-ChildItem (Join-Path $caseRoot '.pcai-private-write') -Directory -Recurse|Where-Object Name -Match '^r-[a-f0-9]{32}$')
        $transactions.Count|Should -Be 1
        $children=@(Get-ChildItem -LiteralPath $transactions[0].FullName -File -Force)
        $children.Count|Should -Be $(if($Mode -eq 'Unknown'){5}else{4})
        foreach($name in @('metadata-probe.bin','original-owner-group-dacl.bin','actual-displaced.bin','receipt.json')){
            @($children|Where-Object Name -EQ $name).Count|Should -Be 1
        }
    }


    # Protects: prepublication predecessor metadata from being silently retired after a Replace boundary change.
    # Detects: a same-identity same-byte displaced file with changed owner/group/DACL remains unqualified custody.
    # Needs: Actual NTFS Replace plus a synthetic DACL edit before displaced snapshot capture.
    # Breadcrumb: Tools/SystemScripts/Machine/SecretBackendUtilities.ps1: Write-BitwardenPrivateFile,Remove-CredentialPublishedCustody
    It 'preserves same-byte displaced custody whose descriptor changed during replacement' {
        $script:replaceImplementation=(Get-Command Invoke-CredentialFileReplace).ScriptBlock
        Mock Invoke-CredentialFileReplace {
            param($Stage,$Target,$Backup)
            & $script:replaceImplementation -Stage $Stage -Target $Target -Backup $Backup
            $acl=Get-Acl -LiteralPath $Backup
            $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.WindowsIdentity]::GetCurrent().User,[Security.AccessControl.FileSystemRights]::WriteData,[Security.AccessControl.AccessControlType]::Deny))
            Set-Acl -LiteralPath $Backup -AclObject $acl
        }
        $warnings=@()
        Write-BitwardenPrivateFile $target 'synthetic-qualified-new' -Overwrite -Confirm:$false -WarningVariable warnings -WarningAction SilentlyContinue
        [IO.File]::ReadAllText($target)|Should -BeExactly 'synthetic-qualified-new'
        @($warnings).Count|Should -Be 1
        $warnings[0].ToString()|Should -BeExactly 'Private file published; successful transaction cleanup was not confirmed. Unqualified custody was preserved.'
        $backup=@(Get-ChildItem (Join-Path $caseRoot '.pcai-private-write') -File -Recurse|Where-Object Name -EQ 'actual-displaced.bin')
        $backup.Count|Should -Be 1
        $retained=Get-PublicationFixtureState $backup[0].FullName
        $retained.Identity|Should -BeExactly $before.Identity
        $retained.Hash|Should -BeExactly $before.Hash
        $retained.Sddl|Should -Not -BeExactly $before.Sddl
        @(Get-ChildItem -LiteralPath $backup[0].DirectoryName -File -Force).Count|Should -Be 4
    }

    # Protects: never calls Set-Acl on the original target.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'never calls Set-Acl on the original target' {
        Mock Set-Acl {throw 'Path ACL mutation forbidden'}
        Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false
        Should -Invoke Set-Acl -Times 0 -Exactly
    }
    # Protects: preserves original bytes and exact SDDL on actual absent-stage replacement failure.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'preserves original bytes and exact SDDL on actual absent-stage replacement failure' {
        Mock Invoke-CredentialFileReplace {
            param($Stage,$Target,$Backup)
            [IO.File]::Move($Stage,($Stage+'.preserved'))
            [IO.File]::Replace($Stage,$Target,$Backup,$false)
        }
        try{Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}catch{$script:failure=$_}
        $failure|Should -Not -BeNullOrEmpty
        $failure.Exception.Data['OriginalOperationException']|Should -Not -BeNullOrEmpty
        $failure.Exception.Data['RecoveryState']|Should -BeExactly 'OriginalPreserved'
        $failure.Exception.Data['RecoveryRequiresReview']|Should -BeFalse
        Assert-PublicationOriginal
    }
    # Protects: restores actual displaced bytes and strict pre-move DACL after partial 1177 failure.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'restores actual displaced bytes and strict pre-move DACL after partial 1177 failure' {
        Mock Invoke-CredentialFileReplace {param($Stage,$Target,$Backup) [IO.File]::Move($Target,$Backup);throw [ComponentModel.Win32Exception]::new(1177)}
        try{Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}catch{$script:failure=$_}
        $failure.Exception.Data['OriginalOperationException'].NativeErrorCode|Should -Be 1177
        $failure.Exception.Data['RecoveryState']|Should -BeExactly 'OwnerGroupDaclRestoredAuditUnverified'
        $failure.Exception.Data['RecoveryRequiresReview']|Should -BeTrue
        Assert-PublicationOriginal
        $backup=@(Get-ChildItem (Join-Path $caseRoot '.pcai-private-write') -Filter actual-displaced.bin -Recurse)[0]
        (Get-PublicationFixtureState $backup.FullName).Identity|Should -BeExactly $before.Identity
    }
    # Protects: preserves a real later writer already present after partial displacement.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'preserves a real later writer already present after partial displacement' {
        Mock Invoke-CredentialFileReplace {param($Stage,$Target,$Backup) [IO.File]::Move($Target,$Backup);New-PublicationLaterWriter;throw [ComponentModel.Win32Exception]::new(1177)}
        try{Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}catch{$script:failure=$_}
        $failure.Exception.Data['OriginalOperationException'].NativeErrorCode|Should -Be 1177
        $now=Get-PublicationFixtureState $target
        $now.Identity|Should -BeExactly $laterState.Identity
        $now.Hash|Should -BeExactly $laterState.Hash
        $now.Sddl|Should -BeExactly $laterState.Sddl
    }
    # Protects: refuses to overwrite a later writer arriving at the recovery move boundary.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'refuses to overwrite a later writer arriving at the recovery move boundary' {
        Mock Invoke-CredentialFileReplace {param($Stage,$Target,$Backup) [IO.File]::Move($Target,$Backup);throw [ComponentModel.Win32Exception]::new(1177)}
        Mock Invoke-CredentialFileMove {param($Stage,$Target) New-PublicationLaterWriter;[IO.File]::Move($Stage,$Target,$false)}
        try{Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}catch{$script:failure=$_}
        $now=Get-PublicationFixtureState $target
        $now.Identity|Should -BeExactly $laterState.Identity
        $now.Sddl|Should -BeExactly $laterState.Sddl
        $now.Hash|Should -BeExactly $laterState.Hash
        $failure.Exception.Data['OriginalOperationException'].NativeErrorCode|Should -Be 1177
    }
    # Protects: detects a same-bytes foreign file after owned recovery ACL restoration.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Test-CredentialCurrentIdentity, Write-BitwardenPrivateFile
    It 'detects a same-bytes foreign file after owned recovery ACL restoration' {
        Mock Invoke-CredentialFileReplace {param($Stage,$Target,$Backup) [IO.File]::Move($Target,$Backup);throw [ComponentModel.Win32Exception]::new(1177)}
        Mock Set-CredentialOwnedDescriptor {
            param($Stream,$Descriptor)
            $script:descriptorTrace.Add([string]$Stream.Name)
            $raw=[Security.AccessControl.RawSecurityDescriptor]::new($Descriptor,0)
            if($raw.ControlFlags-band[Security.AccessControl.ControlFlags]::DiscretionaryAclAutoInherited){$raw.SetFlags($raw.ControlFlags-bor[Security.AccessControl.ControlFlags]::DiscretionaryAclAutoInheritRequired)}
            $submitted=[byte[]]::new($raw.BinaryLength);$raw.GetBinaryForm($submitted,0)
            [Pcai.Credential.FileV1]::WriteDescriptor($Stream.SafeFileHandle,$submitted)
            if(Test-CredentialCurrentIdentity -Path $script:target -Identity ([Pcai.Credential.FileV1]::Identity($Stream.SafeFileHandle)) -Hash $script:before.Hash){try{New-PublicationLaterWriter -LaterText 'original-synthetic'}catch{Write-Host ('Synthetic laterwriter failure: '+$_.Exception.Message);throw}}
        }
        try{Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}catch{$script:failure=$_}
        $laterState.Hash|Should -BeExactly $before.Hash
        $laterState.Identity|Should -Not -BeExactly $before.Identity
        $failure.Exception.Data['RecoveryState']|Should -BeExactly 'LaterWriterPreserved'
        $now=Get-PublicationFixtureState $target
        $now.Identity|Should -BeExactly $laterState.Identity
        $now.Sddl|Should -BeExactly $laterState.Sddl
    }
    # Protects: retains the original operation error when recovery ACL restoration fails.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Test-CredentialCurrentIdentity, Write-BitwardenPrivateFile
    It 'retains the original operation error when recovery ACL restoration fails' {
        Mock Invoke-CredentialFileReplace {param($Stage,$Target,$Backup) [IO.File]::Move($Target,$Backup);throw [ComponentModel.Win32Exception]::new(1177)}
        Mock Set-CredentialOwnedDescriptor {param($Stream,$Descriptor) if(Test-CredentialCurrentIdentity -Path $script:target -Identity ([Pcai.Credential.FileV1]::Identity($Stream.SafeFileHandle)) -Hash $script:before.Hash){$script:recoveryDenied=$true;throw 'Synthetic recovery descriptor denial'};[Pcai.Credential.FileV1]::WriteDescriptor($Stream.SafeFileHandle,$Descriptor)}
        try{Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}catch{$script:failure=$_}
        $failure.Exception.Data['OriginalOperationException'].NativeErrorCode|Should -Be 1177
        $failure.Exception.Data['RecoveryRequiresReview']|Should -BeTrue
        $script:recoveryDenied|Should -BeTrue
        @(Get-ChildItem (Join-Path $caseRoot '.pcai-private-write') -Filter actual-displaced.bin -Recurse).Count|Should -Be 1
    }
    # Protects: rejects a same-bytes changed predecessor before any secret stage.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'rejects a same-bytes changed predecessor before any secret stage' {
        Mock Set-CredentialOwnedDescriptor {param($Stream,$Descriptor) [Pcai.Credential.FileV1]::WriteDescriptor($Stream.SafeFileHandle,$Descriptor);try{New-PublicationLaterWriter -LaterText 'original-synthetic'}catch{Write-Host ('Synthetic laterwriter failure: '+$_.Exception.Message);throw}}
        try{Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}catch{$script:failure=$_}
        $failure|Should -Not -BeNullOrEmpty
        $laterState.Hash|Should -BeExactly $before.Hash
        $now=Get-PublicationFixtureState $target
        $now.Identity|Should -BeExactly $laterState.Identity
        $now.Sddl|Should -BeExactly $laterState.Sddl
        @(Get-ChildItem (Join-Path $caseRoot '.pcai-private-write') -Filter candidate.bin -Recurse).Count|Should -Be 0
    }
    # Protects: does not mutate files or create custody for WhatIf.
    # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'does not mutate files or create custody for WhatIf' {
        Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -WhatIf
        Assert-PublicationOriginal
        Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
    }
    # Protects: rejects overwrite without permission before custody creation.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'rejects overwrite without permission before custody creation' {
        {Write-BitwardenPrivateFile $target 'synthetic-replacement' -Confirm:$false}|Should -Throw
        Assert-PublicationOriginal
        Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
    }
    # Protects: rejects actual hardlink identity before custody creation.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'rejects actual hardlink identity before custody creation' {
        $alias=Join-Path $caseRoot 'alias.fixture'
        New-Item -ItemType HardLink -Path $alias -Target $target|Out-Null
        {Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}|Should -Throw
        Assert-PublicationOriginal
        Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
    }
    # Protects: rejects actual alternate data streams before custody creation.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-CredentialStreamPath, Write-BitwardenPrivateFile
    It 'rejects actual alternate data streams before custody creation' {
        Set-Content -LiteralPath (Get-CredentialStreamPath $target) -Stream synthetic -Value 'nonsecret-stream'
        {Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}|Should -Throw
        Assert-PublicationOriginal
        Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
    }
    # Protects: rejects actual read-only attributes before custody creation.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'rejects actual read-only attributes before custody creation' {
        [IO.File]::SetAttributes($target,[IO.FileAttributes]::ReadOnly)
        try{{Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}|Should -Throw;Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse}
        finally{[IO.File]::SetAttributes($target,[IO.FileAttributes]::Normal)}
    }
    # Protects: rejects reserved path components without creating files.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'rejects reserved path components without creating files' {
        {Write-BitwardenPrivateFile (Join-Path $caseRoot 'NUL.txt') 'synthetic' -Confirm:$false}|Should -Throw
        Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
    }
    # Protects: fails closed for unsupported platform before allocation.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'fails closed for unsupported platform before allocation' {
        $old=$IsWindows
        try{Set-Variable IsWindows -Value $false -Force;{Write-BitwardenPrivateFile $target 'synthetic' -Overwrite -Confirm:$false}|Should -Throw}
        finally{Set-Variable IsWindows -Value $old -Force}
        Assert-PublicationOriginal
    }
    # Protects: rejects a genuine junction ancestor before stage allocation.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'rejects a genuine junction ancestor before stage allocation' {
        $link=Join-Path $script:FixtureDrive ([guid]::NewGuid().ToString('N'))
        New-Item -ItemType Junction -Path $link -Target $caseRoot|Out-Null
        try {
            {Write-BitwardenPrivateFile (Join-Path $link 'credential.fixture') 'synthetic' -Overwrite -Confirm:$false}|Should -Throw
            Assert-PublicationOriginal
            Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
        } finally {[IO.Directory]::Delete($link,$false)}
    }
    # Protects: rejects an insecure pre-existing custody directory without fixing its ACL.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'rejects an insecure pre-existing custody directory without fixing its ACL' {
        $directory=Join-Path $caseRoot '.pcai-private-write'
        $null=New-Item -ItemType Directory $directory
        $acl=Get-Acl $directory
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]::Read,[Security.AccessControl.AccessControlType]::Allow))
        Set-Acl $directory $acl
        $sddl=(Get-Acl $directory).Sddl
        {Write-BitwardenPrivateFile $target 'synthetic' -Overwrite -Confirm:$false}|Should -Throw
        (Get-Acl $directory).Sddl|Should -BeExactly $sddl
        Assert-PublicationOriginal
        @(Get-ChildItem $directory -File -Recurse).Count|Should -Be 0
    }
    # Protects: refuses unsupported exact metadata restoration before any secret bytes or publication.
    # Detects: Unexpected provisioning, provider access, value disclosure or fabricated readiness would violate the passive/no-change contract.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'refuses unsupported exact metadata restoration before any secret bytes or publication' {
        Mock Set-CredentialOwnedDescriptor {throw [UnauthorizedAccessException]::new('Synthetic unsupported owner/group/DACL')}
        Mock Invoke-CredentialFileReplace {throw 'Publication should never start'}
        {Write-BitwardenPrivateFile $target 'synthetic' -Overwrite -Confirm:$false}|Should -Throw
        Assert-PublicationOriginal
        Should -Invoke Invoke-CredentialFileReplace -Times 0 -Exactly
        @(Get-ChildItem (Join-Path $caseRoot '.pcai-private-write') -Filter candidate.bin -Recurse).Count|Should -Be 0
    }
    # Protects: captures a same-bytes foreign displacement rather than claiming compare-and-swap.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'captures a same-bytes foreign displacement rather than claiming compare-and-swap' {
        Mock Invoke-CredentialFileReplace {
            param($Stage,$Target,$Backup)
            New-PublicationLaterWriter -LaterText 'original-synthetic'
            [IO.File]::Replace($Stage,$Target,$Backup,$false)
        }
        try{Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}catch{$script:failure=$_}
        $failure|Should -Not -BeNullOrEmpty
        $laterState.Hash|Should -BeExactly $before.Hash
        $failure.Exception.Data['RecoveryRequiresReview']|Should -BeTrue
        $backup=Join-Path $failure.Exception.Data['CustodyPath'] 'actual-displaced.bin'
        $saved=Get-PublicationFixtureState $backup
        $saved.Identity|Should -BeExactly $laterState.Identity
        $saved.Hash|Should -BeExactly $laterState.Hash
        $saved.Identity|Should -Not -BeExactly $before.Identity
    }
    # Protects: blocks another cooperating writer while the original publication lock is owned.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'blocks another cooperating writer while the original publication lock is owned' {
        $script:nestedFailure=$null
        Mock Invoke-CredentialFileReplace {
            param($Stage,$Target,$Backup)
            try{Write-BitwardenPrivateFile $Target 'nested-synthetic' -Overwrite -Confirm:$false}catch{$script:nestedFailure=$_}
            [IO.File]::Replace($Stage,$Target,$Backup,$false)
        }
        Write-BitwardenPrivateFile $target 'outer-synthetic' -Overwrite -Confirm:$false
        $nestedFailure|Should -Not -BeNullOrEmpty
        [IO.File]::ReadAllText($target)|Should -BeExactly 'outer-synthetic'
    }
    # Protects: captures the original raw descriptor before the actual protected-directory move.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Get-CredentialDescriptorSddl, Write-BitwardenPrivateFile
    It 'captures the original raw descriptor before the actual protected-directory move' {
        $script:afterMoveSddl=$null
        Mock Invoke-CredentialFileReplace {
            param($Stage,$Target,$Backup)
            [IO.File]::Move($Target,$Backup)
            $script:afterMoveSddl=(Get-PublicationFixtureState $Backup).Sddl
            throw [ComponentModel.Win32Exception]::new(1177)
        }
        try{Write-BitwardenPrivateFile $target 'synthetic' -Overwrite -Confirm:$false}catch{$script:failure=$_}
        $saved=[IO.File]::ReadAllBytes((Join-Path $failure.Exception.Data['CustodyPath'] 'original-owner-group-dacl.bin'))
        (Get-CredentialDescriptorSddl $saved)|Should -BeExactly $before.Sddl
        $afterMoveSddl|Should -Not -BeNullOrEmpty
        Assert-PublicationOriginal
    }
    # Protects: qualifies an ordinary token-owner inherited file and publishes a restrictive replacement.
    # Detects: An admitted inherited predecessor must become a distinct private successor without retaining successful displaced backup artifacts.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Assert-BitwardenPrivateInput, Write-BitwardenPrivateFile
    It 'qualifies an ordinary token-owner inherited file and publishes a restrictive replacement' {
        # Qualify the supported inherited predecessor separately from a public
        # ordinary predecessor: actual private inheritable parent, no repair of
        # the existing target and no changes to the success/recovery assertions.
        $privateParent=Join-Path $caseRoot 'private-inherited'
        $directoryAcl=[Security.AccessControl.DirectorySecurity]::new()
        $directorySid=[Security.Principal.WindowsIdentity]::GetCurrent().User
        $directoryAcl.SetOwner($directorySid);$directoryAcl.SetAccessRuleProtection($true,$false)
        $directoryAcl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($directorySid,[Security.AccessControl.FileSystemRights]::FullControl,[Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit',[Security.AccessControl.PropagationFlags]::None,[Security.AccessControl.AccessControlType]::Allow))
        [IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new($privateParent),$directoryAcl)
        $ordinary=Join-Path $privateParent 'ordinary.fixture'
        [IO.File]::WriteAllText($ordinary,'ordinary-synthetic')
        $oldState=Get-PublicationFixtureState $ordinary
        (Get-Acl $ordinary).AreAccessRulesProtected|Should -BeFalse
        Write-BitwardenPrivateFile $ordinary 'private-replacement-synthetic' -Overwrite -Confirm:$false
        [IO.File]::ReadAllText($ordinary)|Should -BeExactly 'private-replacement-synthetic'
        (Get-Acl $ordinary).AreAccessRulesProtected|Should -BeTrue
        Assert-BitwardenPrivateInput -Path $ordinary
        (Get-PublicationFixtureState $ordinary).Identity|Should -Not -BeExactly $oldState.Identity
        @(Get-ChildItem (Join-Path $privateParent '.pcai-private-write') -File -Recurse|Where-Object Name -NE 'publication.lock').Count|Should -Be 0
    }
    # Protects: restores ordinary inherited owner group and DACL exactly after actual partial displacement.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'restores ordinary inherited owner group and DACL exactly after actual partial displacement' {
        # Qualify the supported inherited predecessor separately from a public
        # ordinary predecessor: actual private inheritable parent, no repair of
        # the existing target and no changes to the success/recovery assertions.
        $privateParent=Join-Path $caseRoot 'private-inherited'
        $directoryAcl=[Security.AccessControl.DirectorySecurity]::new()
        $directorySid=[Security.Principal.WindowsIdentity]::GetCurrent().User
        $directoryAcl.SetOwner($directorySid);$directoryAcl.SetAccessRuleProtection($true,$false)
        $directoryAcl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($directorySid,[Security.AccessControl.FileSystemRights]::FullControl,[Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit',[Security.AccessControl.PropagationFlags]::None,[Security.AccessControl.AccessControlType]::Allow))
        [IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new($privateParent),$directoryAcl)
        $script:target=Join-Path $privateParent 'ordinary.fixture'
        [IO.File]::WriteAllText($target,'ordinary-synthetic')
        $script:before=Get-PublicationFixtureState $target
        (Get-Acl $target).AreAccessRulesProtected|Should -BeFalse
        Mock Invoke-CredentialFileReplace {param($Stage,$Target,$Backup) [IO.File]::Move($Target,$Backup);throw [ComponentModel.Win32Exception]::new(1177)}
        try{Write-BitwardenPrivateFile $target 'synthetic' -Overwrite -Confirm:$false}catch{$script:failure=$_}
        $failure.Exception.Data['OriginalOperationException'].NativeErrorCode|Should -Be 1177
        $failure.Exception.Data['RecoveryState']|Should -BeExactly 'OwnerGroupDaclRestoredAuditUnverified'
        Assert-PublicationOriginal
    }
    # Protects: rejects supplied reserved trailing space before provider device normalization.
    # Detects: Wrong provider selection, invalid payload/identity or unintended publication would expose or overwrite credential data.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'rejects supplied reserved trailing space before provider device normalization' {
        {Write-BitwardenPrivateFile (Join-Path $caseRoot 'NUL ') 'synthetic' -Confirm:$false}|Should -Throw '*Reserved*'
        Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
    }
    # Protects: holds cooperating namespace ownership after a swap changes the file identity.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'holds cooperating namespace ownership after a swap changes the file identity' {
        $script:replaceCalls=0; $script:nestedSucceeded=$false; $script:outerSucceeded=$false
        Mock Invoke-CredentialFileReplace {
            param($Stage,$Target,$Backup)
            $script:replaceCalls++
            [IO.File]::Replace($Stage,$Target,$Backup,$false)
            if($script:replaceCalls -eq 1){
                try{Write-BitwardenPrivateFile $Target 'nested-after-swap-synthetic' -Overwrite -Confirm:$false;$script:nestedSucceeded=$true}catch{$script:nestedFailure=$_}
            }
        }
        try{Write-BitwardenPrivateFile $target 'outer-synthetic' -Overwrite -Confirm:$false;$script:outerSucceeded=$true}catch{$script:failure=$_}
        [ordered]@{HelperSha256=(Get-FileHash (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretBackendUtilities.ps1')).Hash;NestedSucceeded=$nestedSucceeded;OuterSucceeded=$outerSucceeded;ActualReplaceCalls=$replaceCalls}|ConvertTo-Json|Set-Content (Join-Path $script:FixtureDrive 'namespace-lock-witness.json') -Encoding utf8
        $nestedSucceeded|Should -BeFalse
        $outerSucceeded|Should -BeTrue
        $replaceCalls|Should -Be 1
        [IO.File]::ReadAllText($target)|Should -BeExactly 'outer-synthetic'
    }
    # Protects: refuses a broadly readable predecessor before real Replace can expose staged secret bytes.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'refuses a broadly readable predecessor before real Replace can expose staged secret bytes' {
        $acl=Get-Acl $target
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]::Read,[Security.AccessControl.AccessControlType]::Allow))
        Set-Acl $target $acl
        $script:before=Get-PublicationFixtureState $target
        $script:replaceCalls=0; $script:publicDuringReplace=$false; $script:sentinelVisible=$false
        Mock Invoke-CredentialFileReplace {
            param($Stage,$Target,$Backup)
            $script:replaceCalls++
            [IO.File]::Replace($Stage,$Target,$Backup,$false)
            $rules=(Get-Acl $Target).GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier])
            $script:publicDuringReplace=@($rules|Where-Object {$_.AccessControlType -eq 'Allow' -and $_.IdentityReference.Value -eq 'S-1-1-0' -and ($_.FileSystemRights -band [Security.AccessControl.FileSystemRights]::ReadData)}).Count -gt 0
            $script:sentinelVisible=[IO.File]::ReadAllText($Target) -ceq 'synthetic-confidential-sentinel'
        }
        try{Write-BitwardenPrivateFile $target 'synthetic-confidential-sentinel' -Overwrite -Confirm:$false}catch{$script:failure=$_}
        [ordered]@{HelperSha256=(Get-FileHash (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretBackendUtilities.ps1')).Hash;ActualReplaceCalls=$replaceCalls;EveryoneReadPermittedAtPublishedBoundary=$publicDuringReplace;SyntheticSentinelAtPublishedBoundary=$sentinelVisible}|ConvertTo-Json|Set-Content (Join-Path $script:FixtureDrive 'public-predecessor-witness.json') -Encoding utf8
        $replaceCalls|Should -Be 0
        $failure|Should -Not -BeNullOrEmpty
        Assert-PublicationOriginal
        Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
    }
    # Protects: refuses foreign parent namespace authority before a broad predecessor can arrive at real Replace.
    # Detects: Lost file identity, bytes or descriptor custody would overwrite a later writer or misreport failed recovery as success.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'refuses foreign parent namespace authority before a broad predecessor can arrive at real Replace' {
        $directoryAcl=Get-Acl $caseRoot
        $directoryAcl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]'CreateFiles,DeleteSubdirectoriesAndFiles',[Security.AccessControl.AccessControlType]::Allow))
        Set-Acl $caseRoot $directoryAcl
        # Directory ACL propagation can alter an existing protected child's
        # AUTO_INHERITED flag. Bind the original after fixture setup, before
        # production is called, while retaining the exact SDDL assertion.
        $script:before=Get-PublicationFixtureState $target
        $script:replaceCalls=0; $script:exposed=$false; $script:foreignIdentity=$null
        Mock Invoke-CredentialFileReplace {
            param($Stage,$Target,$Backup)
            $script:replaceCalls++
            # Initially private original passed its admission. The actual owned
            # fixture boundary publishes a distinct broadly readable object;
            # no foreign login or real credential is involved.
            New-PublicationLaterWriter -LaterText 'foreign-predecessor-synthetic'
            $broadAcl=Get-Acl $Target
            $broadAcl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]::Read,[Security.AccessControl.AccessControlType]::Allow))
            Set-Acl $Target $broadAcl
            $script:foreignIdentity=(Get-PublicationFixtureState $Target).Identity
            [IO.File]::Replace($Stage,$Target,$Backup,$false)
            $rules=(Get-Acl $Target).GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier])
            $worldRead=@($rules|Where-Object {$_.AccessControlType -eq 'Allow' -and $_.IdentityReference.Value -eq 'S-1-1-0' -and ($_.FileSystemRights -band [Security.AccessControl.FileSystemRights]::ReadData)}).Count -gt 0
            $script:exposed=$worldRead -and ([IO.File]::ReadAllText($Target) -ceq 'synthetic-confidential-sentinel')
        }
        try{Write-BitwardenPrivateFile $target 'synthetic-confidential-sentinel' -Overwrite -Confirm:$false}catch{$script:failure=$_}
        [ordered]@{HelperSha256=(Get-FileHash (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretBackendUtilities.ps1')).Hash;InitialIdentity=$before.Identity;InitialPrivateSddl=$before.Sddl;ForeignIdentity=$foreignIdentity;ActualReplaceCalls=$replaceCalls;PublishedWorldReadableSyntheticSentinel=$exposed;ForeignLoginUsed=$false;FailureState=if($failure){$failure.Exception.Data['RecoveryState']}else{$null}}|ConvertTo-Json -Depth 4|Set-Content (Join-Path $script:FixtureDrive 'namespace-authority-witness.json') -Encoding utf8
        $replaceCalls|Should -Be 0
        $failure|Should -Not -BeNullOrEmpty
        Assert-PublicationOriginal
        Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
    }
}

}
Describe 'Directory namespace admission' {
BeforeAll {
    . (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretBackendUtilities.ps1')
}
Describe 'Trusted namespace before private credential staging' {
    BeforeEach {
        $script:root=Join-Path $script:FixtureDrive ([guid]::NewGuid().ToString('N'))
        [IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new($root),(New-CredentialPrivateAcl -Directory))
        $script:target=Join-Path $root 'new-synthetic.fixture'
    }
    # Protects: refuses applicable foreign <Right> before private stage creation.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'refuses applicable foreign <Right> before private stage creation' -TestCases @(
        @{Right='CreateFiles'},@{Right='DeleteSubdirectoriesAndFiles'},
        @{Right='ChangePermissions'},@{Right='TakeOwnership'},@{Right='CreateDirectories'}
    ) {
        param($Right)
        $acl=Get-Acl $root
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]$Right,[Security.AccessControl.AccessControlType]::Allow))
        Set-Acl $root $acl
        Mock New-CredentialPrivateStage { throw 'Unexpected secret stage.' }
        {Write-BitwardenPrivateFile $target 'synthetic-namespace-only' -Confirm:$false}|Should -Throw '*namespace mutation authority refused*'
        Should -Invoke New-CredentialPrivateStage -Times 0 -Exactly
        Test-Path $target|Should -BeFalse
        Test-Path (Join-Path $root '.pcai-private-write')|Should -BeFalse
    }
    # Protects: refuses ancestor DeleteChild even when the immediate parent is protected.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: New-CredentialPrivateAcl, Write-BitwardenPrivateFile
    It 'refuses ancestor DeleteChild even when the immediate parent is protected' {
        $parent=Join-Path $root 'protected-existing-child'
        [IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new($parent),(New-CredentialPrivateAcl -Directory))
        $acl=Get-Acl $root
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles,[Security.AccessControl.AccessControlType]::Allow))
        Set-Acl $root $acl
        $selected=Join-Path $parent 'new.fixture'
        {Write-BitwardenPrivateFile $selected 'synthetic' -Confirm:$false}|Should -Throw '*namespace mutation authority refused*'
        Test-Path $selected|Should -BeFalse
        Test-Path (Join-Path $parent '.pcai-private-write')|Should -BeFalse
    }
    # Protects: refuses an actually foreign-owned directory despite private mutation grants.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Write-BitwardenPrivateFile
    It 'refuses an actually foreign-owned directory despite private mutation grants' {
        $acl=Get-Acl $root
        $acl.SetOwner([Security.Principal.SecurityIdentifier]::new('S-1-5-32-545'))
        Set-Acl $root $acl
        (Get-Acl $root).GetOwner([Security.Principal.SecurityIdentifier]).Value|Should -BeExactly 'S-1-5-32-545'
        {Write-BitwardenPrivateFile $target 'synthetic' -Confirm:$false}|Should -Throw '*namespace ownership or DACL*'
        Test-Path $target|Should -BeFalse
        Test-Path (Join-Path $root '.pcai-private-write')|Should -BeFalse
    }
    # Protects: admits foreign read-only directory grants and still publishes private bytes.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Assert-BitwardenPrivateInput, Write-BitwardenPrivateFile
    It 'admits foreign read-only directory grants and still publishes private bytes' {
        $acl=Get-Acl $root
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]::ReadAndExecute,[Security.AccessControl.AccessControlType]::Allow))
        Set-Acl $root $acl
        Write-BitwardenPrivateFile $target 'synthetic-readonly-parent' -Confirm:$false
        [IO.File]::ReadAllText($target)|Should -BeExactly 'synthetic-readonly-parent'
        Assert-BitwardenPrivateInput $target
    }
    # Protects: ignores InheritOnly grants that do not apply to an admitted existing directory.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Assert-BitwardenPrivateInput, Write-BitwardenPrivateFile
    It 'ignores InheritOnly grants that do not apply to an admitted existing directory' {
        $acl=Get-Acl $root
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]::FullControl,[Security.AccessControl.InheritanceFlags]::ObjectInherit,[Security.AccessControl.PropagationFlags]::InheritOnly,[Security.AccessControl.AccessControlType]::Allow))
        Set-Acl $root $acl
        Write-BitwardenPrivateFile $target 'synthetic-inherit-only' -Confirm:$false
        [IO.File]::ReadAllText($target)|Should -BeExactly 'synthetic-inherit-only'
        Assert-BitwardenPrivateInput $target
    }
    # Protects: conservatively refuses a foreign mutation allow even when a deny also exists.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Assert-CredentialTrustedNamespace
    It 'conservatively refuses a foreign mutation allow even when a deny also exists' {
        $acl=Get-Acl $root
        foreach($kind in @([Security.AccessControl.AccessControlType]::Allow,[Security.AccessControl.AccessControlType]::Deny)) {
            $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]::CreateFiles,$kind))
        }
        Set-Acl $root $acl
        {Assert-CredentialTrustedNamespace $root}|Should -Throw '*namespace mutation authority refused*'
        Test-Path (Join-Path $root '.pcai-private-write')|Should -BeFalse
    }
    # Protects: admits ancestor-only AddSubdirectory grants without granting direct parent mutation.
    # Detects: Unsafe namespace/private-leaf admission, rejected harmless traversal or silent ACL repair would break safe low-friction storage.
    # Needs: Windows x64 NTFS, Pester 5.7, isolated synthetic fixtures; no live provider or hardware.
    # Breadcrumb: Tools/SystemScripts/Machine/{SecretBackendUtilities.ps1,SecretsTier.psm1}: Assert-BitwardenPrivateInput, New-CredentialPrivateAcl, Write-BitwardenPrivateFile
    It 'admits ancestor-only AddSubdirectory grants without granting direct parent mutation' {
        $parent=Join-Path $root 'protected-existing-child'
        [IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new($parent),(New-CredentialPrivateAcl -Directory))
        $acl=Get-Acl $root
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]::CreateDirectories,[Security.AccessControl.AccessControlType]::Allow))
        Set-Acl $root $acl
        $selected=Join-Path $parent 'new.fixture'
        Write-BitwardenPrivateFile $selected 'synthetic-ancestor' -Confirm:$false
        [IO.File]::ReadAllText($selected)|Should -BeExactly 'synthetic-ancestor'
        Assert-BitwardenPrivateInput $selected
    }
}

}

}
