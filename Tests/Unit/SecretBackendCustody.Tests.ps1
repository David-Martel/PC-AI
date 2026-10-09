#Requires -Version 7.0
# Synthetic data and private child processes only; no live credential provider is invoked.
Describe 'Isolated Windows credential custody fixtures' -Tag 'Unit', 'Windows' {
BeforeAll {
    . (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretBackendUtilities.ps1')
    if (-not $IsWindows) { throw 'These custody fixtures require Windows NTFS.' }
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
    # Failed custody cases retain synthetic recovery files for review. Do not
    # delete their evidence merely because the test process is ending.
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
    It 'rejects an unknown recovery identity without touching the unrelated control' {
        { Repair-SecretBackendProcessCustody -CustodyId ('0'*32) } | Should -Throw '*identity is unavailable*'
        $control.HasExited | Should -BeFalse
    }
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
        Mock Resolve-BwBootstrapState {[pscustomobject]@{Password='synthetic-master';PasswordSource='fixture';SessionFile=(Join-Path $env:USERPROFILE '.bwdata/session.txt');ClientId=$null;ClientSecret=$null}}
        Mock Invoke-BitwardenCli {
            if($Arguments[0] -eq 'status'){return [pscustomobject]@{ExitCode=0;Success=$true;StdOut=if($script:unlocked){'{"status":"unlocked"}'}else{'{"status":"locked"}'}}}
            if($Arguments[0] -eq 'unlock'){
                $script:passwordPath=$Arguments[2]
                [IO.File]::ReadAllText($passwordPath)|Should -BeExactly 'synthetic-master'
                Assert-BitwardenPrivateInput $passwordPath
                $script:unlocked=$true
                return [pscustomobject]@{ExitCode=0;Success=$true;StdOut='accepted-consumer-synthetic-session'}
            }
            throw 'Unexpected provider boundary.'
        }
    }
    It 'uses protected profile bootstrap despite actual foreign-write synthetic TEMP' {
        $result=Initialize-BitwardenSessionFromBackends -Refresh -Confirm:$false
        $result.Success|Should -BeTrue
        $passwordPath.StartsWith((Join-Path $env:USERPROFILE '.pcai-credential-state/bootstrap'),[StringComparison]::OrdinalIgnoreCase)|Should -BeTrue
        Test-Path $passwordPath|Should -BeFalse
        @(Get-ChildItem $env:TEMP -Force).Count|Should -Be 0
        [IO.File]::ReadAllText((Join-Path $env:USERPROFILE '.bwdata/session.txt'))|Should -BeExactly 'accepted-consumer-synthetic-session'
    }
    It 'WhatIf provisions no storage and calls no provider' {
        (Initialize-BitwardenSessionFromBackends -WhatIf).Success|Should -BeFalse
        Test-Path (Join-Path $env:USERPROFILE '.pcai-credential-state')|Should -BeFalse
        Should -Invoke Resolve-BwBootstrapState -Times 0 -Exactly
        Should -Invoke Invoke-BitwardenCli -Times 0 -Exactly
    }
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
    It 'import and WhatIf create no directories or provider effects' {
        Test-Path (Join-Path $env:USERPROFILE '.pcai-credential-state')|Should -BeFalse
        Test-Path (Join-Path $env:USERPROFILE '.machine')|Should -BeFalse
        Mock -ModuleName SecretsTier Invoke-BitwardenCli {throw 'No provider permitted.'}
        & $module {Save-SecretsToCache @{fixture='synthetic'} -WhatIf}
        Test-Path (Join-Path $env:USERPROFILE '.pcai-credential-state')|Should -BeFalse
        Should -Invoke -ModuleName SecretsTier Invoke-BitwardenCli -Times 0 -Exactly
    }
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
    It 'refuses preferred path directories instead of falling back to legacy (<Both>)' -TestCases @(@{Both=$false},@{Both=$true}) {
        param($Both)
        New-ConsumerLegacyPair
        $selected=& $module {Initialize-CredentialPrivateStorage $script:CacheSelection}
        [IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new((Join-Path $selected 'secrets-cache.enc')),(New-CredentialPrivateAcl -Directory))
        if($Both){[IO.FileSystemAclExtensions]::Create([IO.DirectoryInfo]::new((Join-Path $selected 'secrets-manifest.json')),(New-CredentialPrivateAcl -Directory))}
        {& $module {Get-SecretsFromCache}}|Should -Throw '*private content is withheld*'
        Assert-ConsumerLegacyUnchanged
    }
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
    It 'reads actual private legacy DPAPI data without provisioning migrating or hydrating' {
        New-ConsumerLegacyPair
        (& $module {Get-SecretsFromCache}).PCAI_CONSUMER_SYNTHETIC|Should -BeExactly 'fixture-secret'
        (& $module {Get-CacheStatus}).Source|Should -BeExactly 'LegacyReadOnly'
        $env:PCAI_CONSUMER_SYNTHETIC|Should -BeNullOrEmpty
        Test-Path (Join-Path $env:USERPROFILE '.pcai-credential-state')|Should -BeFalse
        Test-Path (Join-Path $env:USERPROFILE '.machine/.pcai-private-write')|Should -BeFalse
        Assert-ConsumerLegacyUnchanged
    }
    It 'legacy metadata status never decrypts secret plaintext' {
        New-ConsumerLegacyPair
        Mock -ModuleName SecretsTier ConvertFrom-EncryptedString {throw 'Unexpected decrypt.'}
        $state=& $module {Get-CacheStatus}
        $state.IsValid|Should -BeTrue;$state.Source|Should -BeExactly 'LegacyReadOnly'
        Should -Invoke -ModuleName SecretsTier ConvertFrom-EncryptedString -Times 0 -Exactly
        Assert-ConsumerLegacyUnchanged
    }
    It 'partial preferred pair never falls back to usable legacy' {
        New-ConsumerLegacyPair
        $selected=& $module {Initialize-CredentialPrivateStorage $script:CacheSelection}
        New-ConsumerPrivateText (Join-Path $selected 'secrets-cache.enc') 'partial-synthetic'
        {& $module {Get-SecretsFromCache}}|Should -Throw '*pair is incomplete*'
        Assert-ConsumerLegacyUnchanged
    }
    It 'invalid authenticated ciphertext is refused without private content in errors' {
        New-ConsumerLegacyPair
        [IO.File]::WriteAllText($legacyCache,'private-sentinel-not-dpapi')
        $meta=Get-Content $legacyManifest -Raw|ConvertFrom-Json -AsHashtable
        $meta.cacheSha256=(Get-FileHash $legacyCache).Hash
        [IO.File]::WriteAllText($legacyManifest,($meta|ConvertTo-Json -Compress))
        try{& $module {Get-SecretsFromCache};throw 'Expected rejection.'}catch{$_.Exception.Message|Should -Match 'private content is withheld';$_.Exception.Message|Should -Not -Match 'private-sentinel'}
        Test-Path (Join-Path $env:USERPROFILE '.pcai-credential-state')|Should -BeFalse
    }
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
    It 'foreign-read legacy file grant is rejected without ACL repair' {
        New-ConsumerLegacyPair
        $acl=Get-Acl $legacyCache;$acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]::Read,[Security.AccessControl.AccessControlType]::Allow));Set-Acl $legacyCache $acl
        $sddl=(Get-Acl $legacyCache).Sddl
        {& $module {Get-SecretsFromCache}}|Should -Throw '*private content is withheld*'
        (Get-Acl $legacyCache).Sddl|Should -BeExactly $sddl
    }
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
    It 'invalid explicit cache root is refused during passive import' {
        Remove-Module $module -Force;$script:module=$null
        {Import-Module (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Machine/SecretsTier.psm1') -ArgumentList @('fixture-machine',(Join-Path $env:USERPROFILE '.machine'),$env:TEMP) -Force -ErrorAction Stop}|Should -Throw
        @(Get-ChildItem $env:TEMP -Force).Count|Should -Be 0
        Test-Path (Join-Path $env:USERPROFILE '.machine')|Should -BeFalse
    }
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
    It 'creates a genuinely private file and records successful candidate identity' {
        $new=Join-Path $caseRoot 'new.fixture'
        Write-BitwardenPrivateFile $new 'synthetic-new' -Confirm:$false
        [IO.File]::ReadAllText($new)|Should -BeExactly 'synthetic-new'
        (Get-Acl $new).AreAccessRulesProtected|Should -BeTrue
        $receipt=@(Get-ChildItem (Join-Path $caseRoot '.pcai-private-write') -Filter receipt.json -Recurse)[0]|Get-Content -Raw|ConvertFrom-Json
        $receipt.State|Should -BeExactly 'Published'
        (Get-PublicationFixtureState $new).Identity|Should -BeExactly $receipt.CandidateIdentity
    }
    It 'overwrites with restrictive private ACL and retains the actual original object' {
        Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false
        [IO.File]::ReadAllText($target)|Should -BeExactly 'synthetic-replacement'
        (Get-Acl -LiteralPath $target).AreAccessRulesProtected|Should -BeTrue
        Assert-BitwardenPrivateInput -Path $target
        $backup=@(Get-ChildItem (Join-Path $caseRoot '.pcai-private-write') -Filter actual-displaced.bin -Recurse)[0]
        $saved=Get-PublicationFixtureState $backup.FullName
        $saved.Identity|Should -BeExactly $before.Identity
        $saved.Hash|Should -BeExactly $before.Hash
    }
    It 'never calls Set-Acl on the original target' {
        Mock Set-Acl {throw 'Path ACL mutation forbidden'}
        Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false
        Should -Invoke Set-Acl -Times 0 -Exactly
    }
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
    It 'preserves a real later writer already present after partial displacement' {
        Mock Invoke-CredentialFileReplace {param($Stage,$Target,$Backup) [IO.File]::Move($Target,$Backup);New-PublicationLaterWriter;throw [ComponentModel.Win32Exception]::new(1177)}
        try{Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}catch{$script:failure=$_}
        $failure.Exception.Data['OriginalOperationException'].NativeErrorCode|Should -Be 1177
        $now=Get-PublicationFixtureState $target
        $now.Identity|Should -BeExactly $laterState.Identity
        $now.Hash|Should -BeExactly $laterState.Hash
        $now.Sddl|Should -BeExactly $laterState.Sddl
    }
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
    It 'retains the original operation error when recovery ACL restoration fails' {
        Mock Invoke-CredentialFileReplace {param($Stage,$Target,$Backup) [IO.File]::Move($Target,$Backup);throw [ComponentModel.Win32Exception]::new(1177)}
        Mock Set-CredentialOwnedDescriptor {param($Stream,$Descriptor) if(Test-CredentialCurrentIdentity -Path $script:target -Identity ([Pcai.Credential.FileV1]::Identity($Stream.SafeFileHandle)) -Hash $script:before.Hash){$script:recoveryDenied=$true;throw 'Synthetic recovery descriptor denial'};[Pcai.Credential.FileV1]::WriteDescriptor($Stream.SafeFileHandle,$Descriptor)}
        try{Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}catch{$script:failure=$_}
        $failure.Exception.Data['OriginalOperationException'].NativeErrorCode|Should -Be 1177
        $failure.Exception.Data['RecoveryRequiresReview']|Should -BeTrue
        $script:recoveryDenied|Should -BeTrue
        @(Get-ChildItem (Join-Path $caseRoot '.pcai-private-write') -Filter actual-displaced.bin -Recurse).Count|Should -Be 1
    }
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
    It 'does not mutate files or create custody for WhatIf' {
        Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -WhatIf
        Assert-PublicationOriginal
        Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
    }
    It 'rejects overwrite without permission before custody creation' {
        {Write-BitwardenPrivateFile $target 'synthetic-replacement' -Confirm:$false}|Should -Throw
        Assert-PublicationOriginal
        Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
    }
    It 'rejects actual hardlink identity before custody creation' {
        $alias=Join-Path $caseRoot 'alias.fixture'
        New-Item -ItemType HardLink -Path $alias -Target $target|Out-Null
        {Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}|Should -Throw
        Assert-PublicationOriginal
        Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
    }
    It 'rejects actual alternate data streams before custody creation' {
        Set-Content -LiteralPath (Get-CredentialStreamPath $target) -Stream synthetic -Value 'nonsecret-stream'
        {Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}|Should -Throw
        Assert-PublicationOriginal
        Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
    }
    It 'rejects actual read-only attributes before custody creation' {
        [IO.File]::SetAttributes($target,[IO.FileAttributes]::ReadOnly)
        try{{Write-BitwardenPrivateFile $target 'synthetic-replacement' -Overwrite -Confirm:$false}|Should -Throw;Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse}
        finally{[IO.File]::SetAttributes($target,[IO.FileAttributes]::Normal)}
    }
    It 'rejects reserved path components without creating files' {
        {Write-BitwardenPrivateFile (Join-Path $caseRoot 'NUL.txt') 'synthetic' -Confirm:$false}|Should -Throw
        Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
    }
    It 'fails closed for unsupported platform before allocation' {
        $old=$IsWindows
        try{Set-Variable IsWindows -Value $false -Force;{Write-BitwardenPrivateFile $target 'synthetic' -Overwrite -Confirm:$false}|Should -Throw}
        finally{Set-Variable IsWindows -Value $old -Force}
        Assert-PublicationOriginal
    }
    It 'rejects a genuine junction ancestor before stage allocation' {
        $link=Join-Path $script:FixtureDrive ([guid]::NewGuid().ToString('N'))
        New-Item -ItemType Junction -Path $link -Target $caseRoot|Out-Null
        {Write-BitwardenPrivateFile (Join-Path $link 'credential.fixture') 'synthetic' -Overwrite -Confirm:$false}|Should -Throw
        Assert-PublicationOriginal
        Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
    }
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
    It 'refuses unsupported exact metadata restoration before any secret bytes or publication' {
        Mock Set-CredentialOwnedDescriptor {throw [UnauthorizedAccessException]::new('Synthetic unsupported owner/group/DACL')}
        Mock Invoke-CredentialFileReplace {throw 'Publication should never start'}
        {Write-BitwardenPrivateFile $target 'synthetic' -Overwrite -Confirm:$false}|Should -Throw
        Assert-PublicationOriginal
        Should -Invoke Invoke-CredentialFileReplace -Times 0 -Exactly
        @(Get-ChildItem (Join-Path $caseRoot '.pcai-private-write') -Filter candidate.bin -Recurse).Count|Should -Be 0
    }
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
        $backup=@(Get-ChildItem (Join-Path $privateParent '.pcai-private-write') -Filter actual-displaced.bin -Recurse)[0]
        (Get-PublicationFixtureState $backup.FullName).Identity|Should -BeExactly $oldState.Identity
    }
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
    It 'rejects supplied reserved trailing space before provider device normalization' {
        {Write-BitwardenPrivateFile (Join-Path $caseRoot 'NUL ') 'synthetic' -Confirm:$false}|Should -Throw '*Reserved*'
        Test-Path (Join-Path $caseRoot '.pcai-private-write')|Should -BeFalse
    }
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
    It 'refuses an actually foreign-owned directory despite private mutation grants' {
        $acl=Get-Acl $root
        $acl.SetOwner([Security.Principal.SecurityIdentifier]::new('S-1-5-32-545'))
        Set-Acl $root $acl
        (Get-Acl $root).GetOwner([Security.Principal.SecurityIdentifier]).Value|Should -BeExactly 'S-1-5-32-545'
        {Write-BitwardenPrivateFile $target 'synthetic' -Confirm:$false}|Should -Throw '*namespace ownership or DACL*'
        Test-Path $target|Should -BeFalse
        Test-Path (Join-Path $root '.pcai-private-write')|Should -BeFalse
    }
    It 'admits foreign read-only directory grants and still publishes private bytes' {
        $acl=Get-Acl $root
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]::ReadAndExecute,[Security.AccessControl.AccessControlType]::Allow))
        Set-Acl $root $acl
        Write-BitwardenPrivateFile $target 'synthetic-readonly-parent' -Confirm:$false
        [IO.File]::ReadAllText($target)|Should -BeExactly 'synthetic-readonly-parent'
        Assert-BitwardenPrivateInput $target
    }
    It 'ignores InheritOnly grants that do not apply to an admitted existing directory' {
        $acl=Get-Acl $root
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]::FullControl,[Security.AccessControl.InheritanceFlags]::ObjectInherit,[Security.AccessControl.PropagationFlags]::InheritOnly,[Security.AccessControl.AccessControlType]::Allow))
        Set-Acl $root $acl
        Write-BitwardenPrivateFile $target 'synthetic-inherit-only' -Confirm:$false
        [IO.File]::ReadAllText($target)|Should -BeExactly 'synthetic-inherit-only'
        Assert-BitwardenPrivateInput $target
    }
    It 'conservatively refuses a foreign mutation allow even when a deny also exists' {
        $acl=Get-Acl $root
        foreach($kind in @([Security.AccessControl.AccessControlType]::Allow,[Security.AccessControl.AccessControlType]::Deny)) {
            $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'),[Security.AccessControl.FileSystemRights]::CreateFiles,$kind))
        }
        Set-Acl $root $acl
        {Assert-CredentialTrustedNamespace $root}|Should -Throw '*namespace mutation authority refused*'
        Test-Path (Join-Path $root '.pcai-private-write')|Should -BeFalse
    }
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
