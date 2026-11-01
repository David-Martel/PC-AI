# These fixtures execute the maintained functions with all host boundaries replaced.
# They never contact WSL, Docker, HTTP endpoints, GPUs, scheduled tasks or services.
BeforeAll {
    $script:VirtualRoot = Join-Path $PSScriptRoot '../../Modules/PC-AI.Virtualization'
    foreach ($leaf in @(
        'Private/Get-PcaiRuntimeDefaults.ps1', 'Public/Get-PcaiServiceHealth.ps1',
        'Public/Invoke-PcaiDoctor.ps1', 'Public/Get-WSLEnvironmentHealth.ps1',
        'Public/Get-WSLVsockBridgeStatus.ps1', 'Public/Get-HVSockProxyStatus.ps1',
        'Public/Enable-WSLSystemd.ps1', 'Public/Set-PCaiServiceState.ps1',
        'Public/Get-DockerStatus.ps1', 'Public/Get-HyperVStatus.ps1', 'Public/Get-WSLStatus.ps1'
    )) { . (Join-Path $script:VirtualRoot $leaf) }

    # Local stubs keep even a missing mock from launching a host executable.
    function wsl { param([Parameter(ValueFromRemainingArguments)][object[]]$Tokens) throw 'Unmocked WSL boundary forbidden.' }
    function docker { param([Parameter(ValueFromRemainingArguments)][object[]]$Tokens) throw 'Unmocked Docker boundary forbidden.' }
    function docker-compose { param([Parameter(ValueFromRemainingArguments)][object[]]$Tokens) throw 'Unmocked compose boundary forbidden.' }
    function Get-PcaiRuntimeConfig { param($ProjectRoot, $ConfigPath) throw 'Unmocked runtime config forbidden.' }
    function Get-WindowsOptionalFeature { [CmdletBinding()] param([switch]$Online, $FeatureName) throw 'Unmocked feature query forbidden.' }
    function Get-VMSwitch { [CmdletBinding()] param() throw 'Unmocked switch query forbidden.' }
    function Get-VM { [CmdletBinding()] param() throw 'Unmocked VM query forbidden.' }
    function Get-ScheduledTask { [CmdletBinding()] param($TaskName) throw 'Unmocked task query forbidden.' }
    function Get-ScheduledTaskInfo { [CmdletBinding()] param($TaskName) throw 'Unmocked task query forbidden.' }

    function New-FixtureRuntimeDefaults {
        [pscustomobject]@{
            ProjectRoot = 'fixture-root'; ConfigPath = 'fixture-config'
            PcaiInferenceUrl = 'http://fixture-inference:9080'; FunctionGemmaUrl = 'http://fixture-router:9081'
            OllamaBaseUrl = 'http://fixture-ollama:9082'; vLLMBaseUrl = 'http://fixture-vllm:9083'
            NativeDllSearchPaths = @('fixture-missing.dll', 'fixture-core.dll')
        }
    }
    function New-FixtureComponent {
        param([string]$Status = 'OK', [object[]]$Issues = @(), [string]$RecoveryAction)
        [pscustomobject]@{ Status = $Status; Issues = $Issues; RecoveryAction = $RecoveryAction }
    }
    function New-FixtureServiceHealth {
        [pscustomobject]@{
            PcaiInference = @{ Status = 'OK'; ModelLoaded = $true }
            FunctionGemma = @{ Status = 'OK' }; NativeFFI = @{ DllExists = $true }
            Gpu = @{ Status = 'OK'; Devices = @([pscustomobject]@{ Name = 'Fixture integrated GPU' }); NvidiaSmi = $false; NvidiaRuntime = $false }
            Docker = @{ Running = $false }
        }
    }
}

Describe 'Virtualization runtime configuration projection' -Tag 'Unit', 'Virtualization', 'Portable' {
    It 'preserves configured host endpoints, native search order and caller config selection' {
        Mock Get-PcaiRuntimeConfig { New-FixtureRuntimeDefaults }
        $result = Get-PcaiRuntimeDefaults -ConfigPath 'private-host.json'
        $result.PcaiInferenceUrl | Should -Be 'http://fixture-inference:9080'
        $result.FunctionGemmaUrl | Should -Be 'http://fixture-router:9081'
        $result.OllamaBaseUrl | Should -Be 'http://fixture-ollama:9082'
        $result.vLLMBaseUrl | Should -Be 'http://fixture-vllm:9083'
        $result.NativeDllSearchPaths | Should -Be @('fixture-missing.dll', 'fixture-core.dll')
        $result.ConfigPath | Should -Be 'fixture-config'
        Should -Invoke Get-PcaiRuntimeConfig -Exactly -Times 1 -ParameterFilter { $ConfigPath -eq 'private-host.json' }
    }
    It 'retains explicit config and conservative endpoints when shared resolver is unavailable' {
        Mock Get-PcaiRuntimeConfig { throw 'Unexpected shared resolver call' }
        Mock Get-Command { $null } -ParameterFilter { $Name -eq 'Get-PcaiRuntimeConfig' }
        $result = Get-PcaiRuntimeDefaults -ConfigPath 'explicit-host.json'
        $result.ConfigPath | Should -Be 'explicit-host.json'
        $result.PcaiInferenceUrl | Should -Be 'http://127.0.0.1:8080'
        $result.FunctionGemmaUrl | Should -Be 'http://127.0.0.1:8000'
        $result.NativeDllSearchPaths | Should -HaveCount 0
        Should -Invoke Get-PcaiRuntimeConfig -Exactly -Times 0
    }
}

Describe 'Inference health routing and optional host observations' -Tag 'Unit', 'Virtualization', 'Portable' {
    BeforeEach {
        Mock Get-PcaiRuntimeDefaults { New-FixtureRuntimeDefaults }
        Mock Invoke-RestMethod { throw 'Synthetic endpoint unavailable' }
        Mock Test-Path { $false }
        Mock wsl { $global:LASTEXITCODE = 0; '* FixtureDistro Stopped 2' }
        Mock Get-Process { $null }
        Mock Get-CimInstance { @() }
        Mock Get-Command { $null } -ParameterFilter { $Name -eq 'nvidia-smi' }
        Mock docker { throw 'Unexpected Docker call' }
    }
    It 'uses health backend/model fields and configured URLs without legacy-provider probes' {
        Mock Invoke-RestMethod {
            if ($Uri -like '*/health') { return [pscustomobject]@{ backend = 'fixture-cpu'; model_loaded = $true } }
            [pscustomobject]@{ data = @() }
        }
        $result = Get-PcaiServiceHealth -ConfigPath 'selected-config'
        $result.OverallStatus | Should -Be 'Healthy'
        $result.PcaiInference.Backend | Should -Be 'fixture-cpu'
        $result.PcaiInference.ModelLoaded | Should -BeTrue
        $result.NativeFFI.Status | Should -Be 'NotBuilt'
        $result.WSL.Status | Should -Be 'Stopped'
        $result.Docker.Status | Should -Be 'NotRunning'
        $result.PSObject.Properties.Name | Should -Not -Contain 'Ollama'
        Should -Invoke Invoke-RestMethod -Exactly -Times 1 -ParameterFilter { $Uri -eq 'http://fixture-inference:9080/health' -and $TimeoutSec -eq 3 }
        Should -Invoke Invoke-RestMethod -Exactly -Times 1 -ParameterFilter { $Uri -eq 'http://fixture-router:9081/v1/models' -and $TimeoutSec -eq 2 }
        Should -Invoke Get-PcaiRuntimeDefaults -Exactly -Times 1 -ParameterFilter { $ConfigPath -eq 'selected-config' }
        Should -Invoke Invoke-RestMethod -Exactly -Times 0 -ParameterFilter { $Uri -match 'fixture-ollama|fixture-vllm' }
    }
    It 'uses models and router health fallback with explicit endpoint authority' {
        Mock Invoke-RestMethod {
            if ($Uri -eq 'http://override-inference/v1/models') { return [pscustomobject]@{ data = @([pscustomobject]@{ id = 'fixture-loaded-model' }) } }
            if ($Uri -eq 'http://override-router/health') { return [pscustomobject]@{ status = 'healthy' } }
            throw 'Synthetic primary route failure'
        }
        $result = Get-PcaiServiceHealth -PcaiInferenceUrl 'http://override-inference' -FunctionGemmaUrl 'http://override-router'
        $result.PcaiInference.Backend | Should -Be 'fixture-loaded-model'
        $result.PcaiInference.ModelLoaded | Should -BeTrue
        $result.FunctionGemma.Responding | Should -BeTrue
        Should -Invoke Invoke-RestMethod -Exactly -Times 4
        Should -Invoke Invoke-RestMethod -Exactly -Times 0 -ParameterFilter { $Uri -match 'fixture-inference|fixture-router' }
    }
    It 'does not infer a loaded model from an empty models list' {
        Mock Invoke-RestMethod { [pscustomobject]@{ data = @() } } -ParameterFilter { $Uri -like '*/v1/models' }
        $result = Get-PcaiServiceHealth
        $result.PcaiInference.Responding | Should -BeTrue
        $result.PcaiInference.ModelLoaded | Should -BeFalse
        $result.PcaiInference.Backend | Should -BeNullOrEmpty
    }
    It 'distinguishes DLL presence from an HTTP server and honors explicit search order' {
        Mock Test-Path { $Path -eq 'explicit.dll' }
        $result = Get-PcaiServiceHealth -NativeDllSearchPaths @('absent.dll', 'explicit.dll', 'later.dll')
        $result.OverallStatus | Should -Be 'FFIAvailable'
        $result.PcaiInference.Status | Should -Be 'NotRunning'
        $result.NativeFFI.Path | Should -Be 'explicit.dll'
        Should -Invoke Test-Path -Exactly -Times 0 -ParameterFilter { $Path -eq 'later.dll' -or $Path -like 'fixture-*' }
    }
    It 'reports degraded service and unknown GPU when every probe fails' {
        Mock wsl { throw 'Synthetic WSL absence' }
        Mock Get-Process { throw 'Synthetic process query failure' }
        Mock Get-CimInstance { throw 'Synthetic GPU query refusal' }
        $result = Get-PcaiServiceHealth
        $result.OverallStatus | Should -Be 'Degraded'
        $result.FunctionGemma.Status | Should -Be 'NotRunning'
        $result.WSL.Status | Should -Be 'NotInstalled'
        $result.Docker.Status | Should -Be 'NotInstalled'
        $result.Gpu.Status | Should -Be 'Unknown'
    }
    It 'checks legacy providers only on request and retains a version lookup failure' {
        Mock Invoke-RestMethod {
            if ($Uri -like '*/api/tags') { return [pscustomobject]@{ models = @() } }
            if ($Uri -eq 'http://fixture-vllm:9083/v1/models') { return [pscustomobject]@{ data = @() } }
            throw 'Synthetic endpoint refusal'
        }
        $result = Get-PcaiServiceHealth -CheckLegacyProviders
        $result.Ollama.Status | Should -Be 'OK'
        $result.Ollama.Version | Should -BeNullOrEmpty
        $result.vLLM.Status | Should -Be 'OK'
        Should -Invoke Invoke-RestMethod -Exactly -Times 1 -ParameterFilter { $Uri -eq 'http://fixture-ollama:9082/api/version' }
    }
    It 'projects GPU metadata, Docker NVIDIA runtime and active bridge count' {
        Mock Get-CimInstance { [pscustomobject]@{ Name = 'NVIDIA Fixture'; DriverVersion = 'fixture'; Status = 'OK'; PNPDeviceID = 'SYNTHETIC' } }
        Mock Get-Command { [pscustomobject]@{ Name = 'nvidia-smi' } } -ParameterFilter { $Name -eq 'nvidia-smi' }
        Mock Get-Process { [pscustomobject]@{ Id = 123 } }
        Mock docker { $global:LASTEXITCODE = 0; if ($Tokens[-1] -like '*Runtimes*') { '{"nvidia":{}}' } else { 'fixture-engine' } }
        Mock wsl { $global:LASTEXITCODE = 0; if ($Tokens -contains 'pgrep') { '3' } else { '* FixtureDistro Running 2' } }
        $result = Get-PcaiServiceHealth -Distribution FixtureDistro
        $result.Gpu.Devices[0].PnpDeviceId | Should -Be 'SYNTHETIC'
        $result.Gpu.NvidiaSmi | Should -BeTrue
        $result.Gpu.NvidiaRuntime | Should -BeTrue
        $result.Bridges.Count | Should -Be 3
        $result.Bridges.Status | Should -Be 'OK'
        $result.Docker.Status | Should -Be 'OK'
        Should -Invoke wsl -Exactly -Times 1 -ParameterFilter { $Tokens -contains 'pgrep' -and $Tokens -contains 'FixtureDistro' }
    }
}

Describe 'Doctor recommendations are based on actual health fields' -Tag 'Unit', 'Virtualization', 'Portable' {
    BeforeEach { Mock Get-PcaiServiceHealth { New-FixtureServiceHealth } }
    It 'does not manufacture recommendations for healthy integrated graphics' {
        (Invoke-PcaiDoctor).Recommendations | Should -HaveCount 0
    }
    It 'passes only supplied nonblank endpoints and native paths to health evaluation' {
        $result = Invoke-PcaiDoctor -ConfigPath 'host.json' -PcaiInferenceUrl 'http://chosen' -FunctionGemmaUrl ' ' -NativeDllSearchPaths @('chosen.dll') -CheckLegacyProviders
        $result.Health.PcaiInference.ModelLoaded | Should -BeTrue
        Should -Invoke Get-PcaiServiceHealth -Exactly -Times 1 -ParameterFilter {
            $ConfigPath -eq 'host.json' -and $PcaiInferenceUrl -eq 'http://chosen' -and
            -not $FunctionGemmaUrl -and $NativeDllSearchPaths[0] -eq 'chosen.dll' -and $CheckLegacyProviders
        }
    }
    It 'separates missing model, router, DLL, NVIDIA tool and container runtime actions' {
        Mock Get-PcaiServiceHealth {
            $health = New-FixtureServiceHealth
            $health.PcaiInference.ModelLoaded = $false; $health.FunctionGemma.Status = 'NotRunning'
            $health.NativeFFI.DllExists = $false; $health.Gpu.Devices[0].Name = 'NVIDIA Fixture'
            $health.Docker.Running = $true; $health
        }
        $result = Invoke-PcaiDoctor
        $result.Recommendations | Should -HaveCount 5
        $result.Recommendations -join ' ' | Should -Match 'no model is loaded'
        $result.Recommendations -join ' ' | Should -Match 'nvidia-smi is missing'
        $result.Recommendations -join ' ' | Should -Match 'NVIDIA runtime not detected'
    }
    It 'uses explicit inference target and missing GPU recommendation on failure' {
        Mock Get-PcaiServiceHealth {
            $health = New-FixtureServiceHealth
            $health.PcaiInference.Status = 'NotRunning'; $health.Gpu.Status = 'NotFound'; $health
        }
        $result = Invoke-PcaiDoctor -PcaiInferenceUrl 'http://chosen'
        $result.Recommendations | Should -HaveCount 2
        $result.Recommendations[0] | Should -Match 'http://chosen'
        $result.Recommendations[1] | Should -Match 'No GPU detected'
    }
}

Describe 'Environment health aggregation and recovery selection' -Tag 'Unit', 'Virtualization', 'Portable' {
    BeforeEach {
        Mock Test-WSLHealth { New-FixtureComponent }
        Mock Test-DockerHealth { New-FixtureComponent }
        Mock Test-VSockBridgeHealth { New-FixtureComponent }
        Mock Test-RAGRedisHealth { New-FixtureComponent }
        Mock Test-WSLNetworkHealth { New-FixtureComponent }
        Mock Test-WSLStartupTask { New-FixtureComponent }
        Mock Write-Host {}
    }
    It 'skips only network in Quick mode and propagates explicit distro and recovery choice' {
        $result = Get-WSLEnvironmentHealth -Quick -Distribution FixtureDistro -AutoRecover
        $result.OverallHealth | Should -Be 'Healthy'
        $result.Network.Status | Should -Be 'Skipped'
        Should -Invoke Test-WSLNetworkHealth -Exactly -Times 0
        Should -Invoke Test-WSLHealth -Exactly -Times 1 -ParameterFilter { $Distribution -eq 'FixtureDistro' -and $AutoRecover }
        Should -Invoke Test-DockerHealth -Exactly -Times 1 -ParameterFilter { $Distribution -eq 'FixtureDistro' -and $AutoRecover }
        Should -Invoke Test-RAGRedisHealth -Exactly -Times 1 -ParameterFilter { $AutoRecover }
    }
    It 'preserves issue severity, recovery evidence and full network choice' -ForEach @(
        @{ Severity = 'Critical'; Expected = 'Critical' },
        @{ Severity = 'Warning'; Expected = 'Warning' },
        @{ Severity = 'Info'; Expected = 'Degraded' }
    ) {
        Mock Test-WSLHealth { New-FixtureComponent -Status Error -Issues @([pscustomobject]@{ Severity = $Severity; Component = 'WSL'; Message = 'fixture issue' }) -RecoveryAction 'fixture recovery' }
        $result = Get-WSLEnvironmentHealth -Distribution FixtureDistro
        $result.OverallHealth | Should -Be $Expected
        $result.Issues[0].Message | Should -Be 'fixture issue'
        $result.RecoveryActions | Should -Be @('fixture recovery')
        Should -Invoke Test-WSLNetworkHealth -Exactly -Times 1 -ParameterFilter { $Distribution -eq 'FixtureDistro' }
    }
}

Describe 'WSL configuration edits preserve independent settings' -Tag 'Unit', 'Virtualization', 'Portable' {
    BeforeEach {
        Mock wsl { 'synthetic acknowledgement' }
        Mock Test-Path { $true }
        Mock New-Item { throw 'Unexpected file creation' }
        Mock Get-Content { @('[network]', 'generateResolvConf=false') }
        $script:CapturedConfig = $null
        Mock Set-Content { $script:CapturedConfig = @($Value) }
        Mock Start-Sleep {}
    }
    It 'adds a boot section without replacing network settings or restarting when disabled' {
        $result = Enable-WSLSystemd -Distribution FixtureDistro -RestartWSL:$false
        $result.Updated | Should -BeTrue
        $result.Restarted | Should -BeFalse
        $script:CapturedConfig | Should -HaveCount 1
        $script:CapturedConfig[0] | Should -Be "[network]`r`ngenerateResolvConf=false`r`n`r`n[boot]`r`nsystemd=true`r`n"
        Should -Invoke Set-Content -Exactly -Times 1 -ParameterFilter { $Path -like '*FixtureDistro*' }
        Should -Invoke wsl -Exactly -Times 0 -ParameterFilter { $Tokens -contains '--shutdown' }
    }
    It 'updates an existing boot key and retains the following section' {
        Mock Get-Content { @('[boot]', 'systemd=false', 'command=fixture', '[user]', 'default=fixture') }
        $result = Enable-WSLSystemd -Distribution FixtureDistro
        $result.Restarted | Should -BeTrue
        Should -Invoke Set-Content -Exactly -Times 1 -ParameterFilter { $Value -eq "[boot]`r`nsystemd=true`r`ncommand=fixture`r`n[user]`r`ndefault=fixture`r`n" }
        Should -Invoke wsl -Exactly -Times 1 -ParameterFilter { $Tokens -contains '--shutdown' }
        Should -Invoke wsl -Exactly -Times 1 -ParameterFilter { $Tokens -contains 'WSL restarted' -and $Tokens -contains 'FixtureDistro' }
    }
    It 'inserts the boot key before the next section or at end' -ForEach @(
        @{ Lines = @('[boot]', 'command=fixture', '[user]', 'default=fixture'); Expected = "[boot]`r`ncommand=fixture`r`nsystemd=true`r`n[user]`r`ndefault=fixture`r`n" },
        @{ Lines = @('[boot]', 'command=fixture'); Expected = "[boot]`r`ncommand=fixture`r`nsystemd=true`r`n" }
    ) {
        Mock Get-Content { $Lines }
        (Enable-WSLSystemd -RestartWSL:$false).Error | Should -BeNullOrEmpty
        Should -Invoke Set-Content -Exactly -Times 1 -ParameterFilter { $Value -eq $Expected }
    }
    It 'does not restart WSL after a configuration write failure' {
        Mock Set-Content { throw 'synthetic write denied' }
        $result = Enable-WSLSystemd
        $result.Updated | Should -BeFalse
        $result.Error | Should -Be 'synthetic write denied'
        Should -Invoke wsl -Exactly -Times 0 -ParameterFilter { $Tokens -contains '--shutdown' }
    }
}

Describe 'Read-only bridge and service state contracts' -Tag 'Unit', 'Virtualization', 'Portable' {
    It 'returns an empty proxy result without querying a process for missing state' {
        Mock Test-Path { $false }; Mock Get-Process { throw 'Unexpected process query' }
        $result = Get-HVSockProxyStatus -StatePath 'fixture-state.json'
        $result.Running | Should -Be 0
        $result.Entries | Should -HaveCount 0
        Should -Invoke Get-Process -Exactly -Times 0
    }
    It 'projects legacy proxy entries as unverified without treating PID existence as custody' {
        Mock Resolve-HVSockStatePath { $Path }
        Mock Test-Path { $true }
        Mock Get-Content { '[{"Name":"fixture-a","Pid":101,"ServiceId":"fixture-id","TcpTarget":"fixture:80","Started":"fixture"},{"Name":"fixture-b","Pid":102}]' }
        Mock Get-Process { if ($Id -eq 101) { [pscustomobject]@{ Id = 101 } } }
        $result = Get-HVSockProxyStatus -StatePath 'fixture-state.json'
        $result.Running | Should -Be 0
        $result.Entries | Should -HaveCount 2
        $result.Entries[0].TcpTarget | Should -Be 'fixture:80'
        $result.Entries[0].CustodyVerified | Should -BeFalse
        $result.Entries[0].CustodyError | Should -Not -BeNullOrEmpty
        $result.Entries[1].Running | Should -BeFalse
        Should -Invoke Get-Process -Exactly -Times 0
    }
    It 'retains WSL service/bridge output and explicit distribution' {
        Mock wsl { if ($Tokens -contains 'systemctl') { 'active' } else { "port=8000`nport=8080" } }
        $result = Get-WSLVsockBridgeStatus -Distribution FixtureDistro
        $result.ServiceState | Should -Be 'active'
        $result.BridgeStatus | Should -Be "port=8000`nport=8080"
        Should -Invoke wsl -Exactly -Times 2 -ParameterFilter { $Tokens -contains 'FixtureDistro' }
    }
    It 'preserves a failed WSL status query as error evidence' {
        Mock wsl { throw 'synthetic unavailable' }
        $result = Get-WSLVsockBridgeStatus
        $result.Errors | Should -Be @('synthetic unavailable')
        $result.ServiceState | Should -BeNullOrEmpty
    }
    It 'dispatches exactly the requested service action before reporting its status' -ForEach @(
        @{ Action = 'start'; Command = 'Start-Service' }, @{ Action = 'stop'; Command = 'Stop-Service' }, @{ Action = 'restart'; Command = 'Restart-Service' }
    ) {
        Mock Start-Service {}; Mock Stop-Service {}; Mock Restart-Service {}
        Mock Get-Service { [pscustomobject]@{ Name = 'fixture-service'; Status = 'FixtureStatus' } }
        (Set-PCaiServiceState -Name fixture-service -Action $Action).Status | Should -Be 'FixtureStatus'
        Should -Invoke $Command -Exactly -Times 1 -ParameterFilter { $Name -eq 'fixture-service' -and $ErrorAction -eq 'Stop' }
        foreach ($other in @('Start-Service', 'Stop-Service', 'Restart-Service') | Where-Object { $_ -ne $Command }) { Should -Invoke $other -Exactly -Times 0 }
    }
    It 'propagates service operation failure before querying success state' {
        Mock Start-Service { throw 'synthetic service denied' }; Mock Get-Service { throw 'Unexpected service query' }
        { Set-PCaiServiceState -Name fixture-service -Action start } | Should -Throw '*synthetic service denied*'
        Should -Invoke Get-Service -Exactly -Times 0
    }
}

Describe 'Environment component failures and bounded recovery decisions' -Tag 'Unit', 'Virtualization', 'Portable' {
    BeforeEach {
        Mock wsl { throw 'Unexpected WSL route' }
        Mock docker { throw 'Unexpected Docker route' }
        Mock docker-compose { throw 'Unexpected compose route' }
        Mock Start-Sleep {}
        Mock Write-Host {}
        Mock Test-Path { $false }
    }
    It 'reports failed WSL enumeration without issuing start or systemd commands' {
        Mock wsl { $global:LASTEXITCODE = 9; 'enumeration refused' }
        $result = Test-WSLHealth -Distribution FixtureDistro -AutoRecover
        $result.Status | Should -Be 'Error'
        $result.Issues[0].Severity | Should -Be 'Critical'
        Should -Invoke wsl -Exactly -Times 1
    }
    It 'leaves a stopped distribution stopped without automatic recovery' {
        Mock wsl { $global:LASTEXITCODE = 0; '* FixtureDistro Stopped 2' }
        $result = Test-WSLHealth -Distribution FixtureDistro
        $result.DistroRunning | Should -BeFalse
        $result.Issues[0].Message | Should -Match 'not running'
        Should -Invoke wsl -Exactly -Times 1
    }
    It 'rechecks a recovered distribution and accepts degraded systemd with explicit evidence' {
        $script:WslListCalls = 0
        Mock wsl {
            $global:LASTEXITCODE = 0
            if ($Tokens -contains '-l') {
                $script:WslListCalls++
                if ($script:WslListCalls -eq 1) { '* FixtureDistro Stopped 2' } else { '* FixtureDistro Running 2' }
            } elseif ($Tokens -contains 'systemctl') { 'degraded' } else { 'Started' }
        }
        $result = Test-WSLHealth -Distribution FixtureDistro -AutoRecover
        $result.DistroRunning | Should -BeTrue
        $result.SystemdStatus | Should -Be 'degraded'
        $result.RecoveryAction | Should -Be 'Started FixtureDistro'
        Should -Invoke wsl -Exactly -Times 1 -ParameterFilter { $Tokens -contains 'Started' -and $Tokens -contains 'FixtureDistro' }
    }
    It 'reports unhealthy systemd without silently claiming health' {
        Mock wsl { $global:LASTEXITCODE = 0; if ($Tokens -contains '-l') { '* FixtureDistro Running 2' } else { 'maintenance' } }
        $result = Test-WSLHealth -Distribution FixtureDistro
        $result.Status | Should -Be 'Warning'
        $result.Issues[0].Message | Should -Be 'Systemd status: maintenance'
    }
    It 'does not accept negated or partial systemd states as running' -ForEach @(
        @{ State = 'not running' }, @{ State = 'running-away' }, @{ State = 'undegraded' }
    ) {
        Mock wsl { $global:LASTEXITCODE = 0; if ($Tokens -contains '-l') { '* FixtureDistro Running 2' } else { $State } }
        $result = Test-WSLHealth -Distribution FixtureDistro
        $result.Status | Should -Be 'Warning'
        $result.SystemdStatus | Should -Be $State
        $result.Issues[0].Message | Should -Be "Systemd status: $State"
    }
    It 'prefers a responding Windows Docker engine and counts useful container output' {
        Mock Get-Command { [pscustomobject]@{ Name = 'docker' } } -ParameterFilter { $Name -eq 'docker' }
        Mock docker { $global:LASTEXITCODE = 0; if ($Tokens -contains 'ps') { 'container-a'; 'container-b' } else { 'fixture engine' } }
        $result = Test-DockerHealth -Distribution FixtureDistro
        $result.Status | Should -Be 'OK'
        $result.ServiceActive | Should -BeTrue
        $result.ContainerCount | Should -Be 2
        Should -Invoke wsl -Exactly -Times 0
    }
    It 'uses an already active WSL Docker service when no host engine exists' {
        Mock Get-Command { $null } -ParameterFilter { $Name -eq 'docker' }
        Mock wsl {
            $global:LASTEXITCODE = 0
            if ($Tokens -contains 'systemctl') { 'active' } elseif ($Tokens -contains 'ps') { 'container-a' } else { 'fixture engine' }
        }
        $result = Test-DockerHealth -Distribution FixtureDistro
        $result.DaemonResponding | Should -BeTrue
        $result.ContainerCount | Should -Be 1
        Should -Invoke wsl -Exactly -Times 0 -ParameterFilter { $Tokens -contains 'sudo' }
    }
    It 'does not start inactive Docker unless automatic recovery is requested' {
        Mock Get-Command { $null } -ParameterFilter { $Name -eq 'docker' }
        Mock wsl { $global:LASTEXITCODE = 0; 'inactive' }
        $result = Test-DockerHealth -Distribution FixtureDistro
        $result.Status | Should -Be 'Error'
        $result.ServiceActive | Should -BeFalse
        Should -Invoke wsl -Exactly -Times 1
    }
    It 'records recovered Docker separately from daemon connectivity failure' {
        Mock Get-Command { $null } -ParameterFilter { $Name -eq 'docker' }
        $script:DockerServiceChecks = 0
        Mock wsl {
            $global:LASTEXITCODE = 0
            if ($Tokens -contains 'is-active') {
                $script:DockerServiceChecks++
                if ($script:DockerServiceChecks -eq 1) { 'inactive' } else { 'active' }
            } elseif ($Tokens -contains 'info') { $global:LASTEXITCODE = 1; 'refused' } else { 'started' }
        }
        $result = Test-DockerHealth -Distribution FixtureDistro -AutoRecover
        $result.ServiceActive | Should -BeTrue
        $result.DaemonResponding | Should -BeFalse
        $result.RecoveryAction | Should -Be 'Started Docker service'
        $result.Issues.Message | Should -Contain 'Docker daemon not responding'
        Should -Invoke wsl -Exactly -Times 1 -ParameterFilter { $Tokens -contains 'sudo' -and $Tokens -contains 'start' }
    }
    It 'reports optional bridges absent without attempting to install anything' {
        Mock wsl { $global:LASTEXITCODE = 1; if ($Tokens -contains 'systemctl') { 'failed' } elseif ($Tokens -contains 'pgrep') { '0' } else { '' } }
        $result = Test-VSockBridgeHealth -Distribution FixtureDistro -AutoRecover
        $result.Status | Should -Be 'Info'
        $result.SocatProcesses | Should -Be 0
        $result.Bridges | Should -HaveCount 0
        Should -Invoke wsl -Exactly -Times 0 -ParameterFilter { $Tokens -contains 'start' -or $Tokens -contains 'sudo' }
    }
    It 'retains observed bridge ports and positive process count' {
        Mock wsl {
            $global:LASTEXITCODE = 0
            if ($Tokens -contains 'systemctl') { 'active' }
            elseif ($Tokens -contains 'pgrep') { '2' }
            elseif ($Tokens -contains 'sport = :8000') { 'LISTEN 0 1 *:8000' }
            else { '' }
        }
        $result = Test-VSockBridgeHealth -Distribution FixtureDistro
        $result.Status | Should -Be 'OK'
        $result.SocatProcesses | Should -Be 2
        $result.Bridges.Port | Should -Be 8000
    }
    It 'does not launch absent Redis without explicit recovery and an existing compose path' {
        Mock docker { $global:LASTEXITCODE = 1; '' }
        $result = Test-RAGRedisHealth -ComposePath 'fixture-compose.yml' -AutoRecover
        $result.Status | Should -Be 'Error'
        Should -Invoke docker-compose -Exactly -Times 0
    }
    It 'requires both Redis ping and search module for a healthy backend' -ForEach @(
        @{ Ping = 'PONG'; Modules = 'search'; Expected = 'OK'; Redis = $true; Search = $true },
        @{ Ping = 'PONG'; Modules = 'other'; Expected = 'Warning'; Redis = $true; Search = $false },
        @{ Ping = 'refused'; Modules = 'search'; Expected = 'Error'; Redis = $false; Search = $false }
    ) {
        Mock docker {
            $global:LASTEXITCODE = 0
            if ($Tokens -contains 'ps') { 'fixture-id|Up 5 minutes' }
            elseif ($Tokens -contains 'ping') { $Ping } else { $Modules }
        }
        $result = Test-RAGRedisHealth -ComposePath 'fixture-compose.yml'
        $result.Status | Should -Be $Expected
        $result.ContainerID | Should -Be 'fixture-id'
        $result.RedisPing | Should -Be $Redis
        $result.SearchModule | Should -Be $Search
        Should -Invoke docker-compose -Exactly -Times 0
    }
    It 'uses only the explicitly selected compose path during recovery and rechecks the backend' {
        Mock Test-Path { $Path -eq 'fixture-compose.yml' }
        $script:RedisChecks = 0
        Mock docker {
            $global:LASTEXITCODE = 0
            if ($Tokens -contains 'ps') {
                $script:RedisChecks++
                if ($script:RedisChecks -gt 1) { 'fixture-id|Up' }
            } elseif ($Tokens -contains 'ping') { 'PONG' } else { 'search' }
        }
        Mock docker-compose {}
        $result = Test-RAGRedisHealth -ComposePath 'fixture-compose.yml' -AutoRecover
        $result.Status | Should -Be 'Warning'
        $result.RedisPing | Should -BeTrue
        $result.SearchModule | Should -BeTrue
        $result.Issues[0].Message | Should -Be 'RAG-Redis container not running'
        $result.RecoveryAction | Should -Be 'Started RAG-Redis container'
        Should -Invoke docker-compose -Exactly -Times 1 -ParameterFilter {
            $Tokens -contains 'fixture-compose.yml' -and $Tokens -contains 'up' -and $Tokens -contains 'redis'
        }
    }
    It 'distinguishes DNS failure from external connectivity failure' -ForEach @(
        @{ DnsExit = 0; PingExit = 0; Expected = 'OK'; Issues = 0 },
        @{ DnsExit = 0; PingExit = 1; Expected = 'Warning'; Issues = 1 },
        @{ DnsExit = 1; PingExit = 0; Expected = 'Error'; Issues = 1 }
    ) {
        Mock wsl { $global:LASTEXITCODE = if ($Tokens -contains 'nslookup') { $DnsExit } else { $PingExit }; 'synthetic response' }
        $result = Test-WSLNetworkHealth -Distribution FixtureDistro
        $result.Status | Should -Be $Expected
        $result.Issues | Should -HaveCount $Issues
        Should -Invoke wsl -Exactly -Times 2 -ParameterFilter { $Tokens -contains 'FixtureDistro' }
    }
    It 'distinguishes enabled, disabled and missing startup tasks without mutation' -ForEach @(
        @{ State = 'Ready'; Expected = 'OK'; Exists = $true; Enabled = $true },
        @{ State = 'Disabled'; Expected = 'Warning'; Exists = $true; Enabled = $false },
        @{ State = $null; Expected = 'Info'; Exists = $false; Enabled = $false }
    ) {
        Mock Get-ScheduledTask { if ($State) { [pscustomobject]@{ State = $State } } }
        Mock Get-ScheduledTaskInfo { [pscustomobject]@{ LastRunTime = [datetime]'2026-01-01' } }
        $result = Test-WSLStartupTask
        $result.Status | Should -Be $Expected
        $result.Exists | Should -Be $Exists
        $result.Enabled | Should -Be $Enabled
        if ($Exists) { $result.LastRun | Should -Be ([datetime]'2026-01-01') }
    }
}

Describe 'Virtualization inventory preserves useful data and failures' -Tag 'Unit', 'Virtualization', 'Portable' {
    It 'parses WSL and kernel versions from native line arrays and joined text' -ForEach @(
        @{ VersionOutput = @('WSL version: 2.6.1.0', 'Kernel version: 6.6.87.2-1') },
        @{ VersionOutput = "WSL version: 2.6.1.0`r`nKernel version: 6.6.87.2-1`r`n" }
    ) {
        Mock Get-Command { [pscustomobject]@{ Name = 'wsl.exe' } } -ParameterFilter { $Name -eq 'wsl.exe' }
        Mock wsl { if ($Tokens -contains '--version') { $VersionOutput } else { '* FixtureDistro Running 2' } }
        $result = Get-WSLStatus
        $result.Version | Should -Be '2.6.1.0'
        $result.KernelVersion | Should -Be '6.6.87.2-1'
    }
    It 'parses Docker container columns and distinguishes Linux from Windows backend' -ForEach @(
        @{ OsType = 'linux'; Backend = 'WSL2' }, @{ OsType = 'windows'; Backend = 'Windows' }
    ) {
        Mock Get-Command { [pscustomobject]@{ Name = 'docker.exe' } } -ParameterFilter { $Name -eq 'docker.exe' }
        Mock docker {
            $global:LASTEXITCODE = 0
            if ($Tokens -contains 'version') { 'fixture-version' }
            elseif ($Tokens -contains 'info') { $OsType }
            else { "fixture-id`tfixture-name`tUp 1 minute`tfixture-image`nmalformed row" }
        }
        $result = Get-DockerStatus -IncludeContainers
        $result.Running | Should -BeTrue
        $result.Backend | Should -Be $Backend
        $result.Version | Should -Be 'fixture-version'
        $result.Containers | Should -HaveCount 1
        $result.Containers[0].Name | Should -Be 'fixture-name'
        $result.Containers[0].Image | Should -Be 'fixture-image'
    }
    It 'does not query containers after Docker reports a failed server version' {
        Mock Get-Command { [pscustomobject]@{ Name = 'docker.exe' } } -ParameterFilter { $Name -eq 'docker.exe' }
        Mock docker { $global:LASTEXITCODE = 1; 'synthetic daemon refused' }
        $result = Get-DockerStatus -IncludeContainers
        $result.Installed | Should -BeTrue
        $result.Running | Should -BeFalse
        $result.Severity | Should -Be 'Warning'
        Should -Invoke docker -Exactly -Times 0 -ParameterFilter { $Tokens -contains 'ps' }
    }
    It 'returns before service and VM enumeration when Hyper-V is disabled' {
        Mock Get-WindowsOptionalFeature { [pscustomobject]@{ State = 'Disabled' } }
        Mock Get-Service { throw 'Unexpected service query' }; Mock Get-VM { throw 'Unexpected VM query' }
        $result = Get-HyperVStatus -IncludeVMs
        $result.Installed | Should -BeTrue
        $result.Enabled | Should -BeFalse
        $result.Severity | Should -Be 'Warning'
        Should -Invoke Get-Service -Exactly -Times 0
        Should -Invoke Get-VM -Exactly -Times 0
    }
    It 'projects service, switch and requested VM metadata with stopped-service warning' {
        Mock Get-WindowsOptionalFeature { [pscustomobject]@{ State = 'Enabled' } }
        Mock Get-Service { [pscustomobject]@{ Name = $Name; Status = if ($Name -eq 'vmms') { 'Stopped' } else { 'Running' }; StartType = 'Automatic' } }
        Mock Get-VMSwitch { [pscustomobject]@{ Name = 'fixture-switch'; SwitchType = 'Internal'; NetAdapterInterfaceDescription = 'fixture' } }
        Mock Get-VM { [pscustomobject]@{ Name = 'fixture-vm'; State = 'Running'; CPUUsage = 3; MemoryAssigned = 1024; Uptime = [timespan]::FromMinutes(1) } }
        $result = Get-HyperVStatus -IncludeVMs
        $result.Severity | Should -Be 'Warning'
        $result.Services | Should -HaveCount 3
        $result.VirtualSwitches.Name | Should -Be 'fixture-switch'
        $result.VirtualMachines.Name | Should -Be 'fixture-vm'
        $result.VirtualMachines.MemoryAssigned | Should -Be 1024
    }
    It 'does not enumerate VMs unless requested and survives a failed optional switch query' {
        Mock Get-WindowsOptionalFeature { [pscustomobject]@{ State = 'Enabled' } }
        Mock Get-Service { [pscustomobject]@{ Name = $Name; Status = 'Running'; StartType = 'Automatic' } }
        Mock Get-VMSwitch { throw 'synthetic switch query denied' }; Mock Get-VM { throw 'Unexpected VM query' }
        $result = Get-HyperVStatus
        $result.Severity | Should -Be 'OK'
        $result.VirtualMachines | Should -HaveCount 0
        Should -Invoke Get-VM -Exactly -Times 0
    }
    It 'parses the default distribution and propagates its running/stopped severity' -ForEach @(
        @{ State = 'Running'; Expected = 'OK' }, @{ State = 'Stopped'; Expected = 'Info' }
    ) {
        Mock Get-Command { [pscustomobject]@{ Name = 'wsl.exe' } } -ParameterFilter { $Name -eq 'wsl.exe' }
        Mock wsl { if ($Tokens -contains '--version') { '' } else { "  NAME STATE VERSION`n* FixtureDistro $State 2`n  OtherDistro Stopped 1" } }
        Mock Test-Path { $true }; Mock Get-Content { '[wsl2] fixture=true' }
        $result = Get-WSLStatus -Detailed
        $result.DefaultDistro | Should -Be 'FixtureDistro'
        $result.Distributions | Should -HaveCount 2
        $result.Distributions[0].Version | Should -Be 2
        $result.Distributions[1].IsDefault | Should -BeFalse
        $result.Severity | Should -Be $Expected
        $result.WSLConfig | Should -Be '[wsl2] fixture=true'
    }
    It 'reports an installed WSL with no distributions without reading user configuration' {
        Mock Get-Command { [pscustomobject]@{ Name = 'wsl.exe' } } -ParameterFilter { $Name -eq 'wsl.exe' }
        Mock wsl { '' }; Mock Get-Content { throw 'Unexpected user config read' }
        $result = Get-WSLStatus
        $result.Installed | Should -BeTrue
        $result.Severity | Should -Be 'Warning'
        $result.Distributions | Should -HaveCount 0
        Should -Invoke Get-Content -Exactly -Times 0
    }
}
