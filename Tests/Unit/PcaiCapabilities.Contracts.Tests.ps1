#Requires -Version 7.0
# Exercises the actual producer with inert metadata dependencies; no native module import.
param([string]$RepositoryRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..')))
BeforeAll {
    $script:capabilitySource=Join-Path $RepositoryRoot 'Modules/PC-AI.Acceleration/Public/Get-PcaiCapabilities.ps1'
    . $script:capabilitySource
    if([IO.Path]::GetFullPath((Get-Command Get-PcaiCapabilities).ScriptBlock.File)-cne[IO.Path]::GetFullPath($script:capabilitySource)){throw 'Actual producer source binding failed.'}
    function Get-PcaiNativeStatus {throw 'Unmocked status/native discovery forbidden.'}
    function Initialize-PcaiNative {throw 'Native initialization forbidden.'}
    function Test-PcaiNativeAvailable {throw 'Native availability activation forbidden.'}
    function Get-CimInstance {param($ClassName)throw 'Unmocked GPU collection forbidden.'}
    function Get-PcaiServiceHealth {throw 'Unmocked service/provider calls forbidden.'}
    function Invoke-PcaiNativeEstimateTokens {throw 'Native execution forbidden.'}
    function Invoke-PcaiNativeDirectoryManifest {throw 'Native execution forbidden.'}
    function Invoke-PcaiNativeFileSearch {throw 'Native execution forbidden.'}
    function Invoke-PcaiNativeContentSearch {throw 'Native execution forbidden.'}
    function Invoke-PcaiNativeSystemInfo {throw 'Native execution forbidden.'}
    function Get-DiskUsageFast {throw 'Disk/native execution forbidden.'}
    $script:metadataCommands=@{}
    $script:wrapperNames=@('Invoke-PcaiNativeEstimateTokens','Invoke-PcaiNativeDirectoryManifest','Invoke-PcaiNativeFileSearch','Invoke-PcaiNativeContentSearch','Invoke-PcaiNativeSystemInfo','Get-DiskUsageFast','Get-PcaiServiceHealth')
    foreach($name in $script:wrapperNames){$script:metadataCommands[$name]=Get-Command $name}
    function New-SyntheticCapabilityStatus([bool]$Enabled=$false){
        [pscustomobject]@{Available=$Enabled;Version='synthetic';DllPath=$null;CoreAvailable=$Enabled;CpuCount=[uint32]4;Modules=[pscustomobject]@{Core=$Enabled;Search=$Enabled;System=$Enabled;Performance=$Enabled;Fs=$Enabled};Dlls=$null}
    }
    function Get-OperationRow($Capabilities,[string]$Operation){
        $rows=@($Capabilities.BackendCoverage|Where-Object Operation -CEQ $Operation)
        $rows.Count|Should -Be 1
        return $rows[0]
    }
}
Describe 'Actual capability producer with inert metadata boundaries' {
    BeforeEach {
        Set-StrictMode -Version Latest
        $script:nativeStatus=New-SyntheticCapabilityStatus
        $script:presentCommands=@{}
        $script:gpuRows=@();$script:servicePayload=[pscustomobject]@{Schema='synthetic-health';Services=@([pscustomobject]@{Name='inert';Healthy=$true})}
        $script:expectedGpuCalls=0;$script:expectedServiceCalls=0
        Mock Get-PcaiNativeStatus {$script:nativeStatus}
        Mock Get-Command {
            $queryName=[string](@($Name)[0])
            if($script:presentCommands.ContainsKey($queryName)-and$script:presentCommands[$queryName]){$script:metadataCommands[$queryName]}else{$null}
        } -ParameterFilter {@($Name).Count-eq1-and([string](@($Name)[0]))-in$script:wrapperNames}
        Mock Get-CimInstance {$script:gpuRows}
        Mock Get-PcaiServiceHealth {$script:servicePayload}
        Mock Initialize-PcaiNative {throw 'Forbidden activation.'}
        Mock Test-PcaiNativeAvailable {throw 'Forbidden activation.'}
        foreach($name in @('Invoke-PcaiNativeEstimateTokens','Invoke-PcaiNativeDirectoryManifest','Invoke-PcaiNativeFileSearch','Invoke-PcaiNativeContentSearch','Invoke-PcaiNativeSystemInfo','Get-DiskUsageFast')){Mock $name {throw 'Forbidden native execution.'}}
    }
    AfterEach {
        Should -Invoke Initialize-PcaiNative -Times 0 -Exactly
        Should -Invoke Test-PcaiNativeAvailable -Times 0 -Exactly
        foreach($name in @('Invoke-PcaiNativeEstimateTokens','Invoke-PcaiNativeDirectoryManifest','Invoke-PcaiNativeFileSearch','Invoke-PcaiNativeContentSearch','Invoke-PcaiNativeSystemInfo','Get-DiskUsageFast')){Should -Invoke $name -Times 0 -Exactly}
        Should -Invoke Get-CimInstance -Times $script:expectedGpuCalls -Exactly
        Should -Invoke Get-PcaiServiceHealth -Times $script:expectedServiceCalls -Exactly
    }
    It 'reports six unique operations without collecting optional GPU or services by default' {
        $result=Get-PcaiCapabilities
        @($result.BackendCoverage).Count|Should -Be 6
        @($result.BackendCoverage.Operation|Sort-Object)|Should -BeExactly @('ContentSearch','DirectoryManifest','DiskUsage','FileSearch','FullContext','TokenEstimate')
        $result.Cpu.LogicalCores|Should -Be 4
        $result.Gpu|Should -BeNullOrEmpty;$result.Services|Should -BeNullOrEmpty
        foreach($row in $result.BackendCoverage){$row.RustAvailable|Should -BeFalse;$row.CSharpBridgeAvailable|Should -BeFalse}
    }
    It 'keeps an unchanged full available and callable native contract' {
        $script:nativeStatus=New-SyntheticCapabilityStatus $true
        foreach($name in $script:wrapperNames){$script:presentCommands[$name]=$true}
        $result=Get-PcaiCapabilities
        foreach($row in $result.BackendCoverage){$row.RustAvailable|Should -BeTrue;$row.PowerShellSurface|Should -BeTrue;$row.CoverageState|Should -BeExactly 'Rust+CSharp+PS';$row.Gap|Should -BeNullOrEmpty;$row.PreferredBackend|Should -BeExactly 'Rust+C#'}
        $result.Features.ContentSearch|Should -BeTrue
    }
    It 'preserves independent search availability without inventing a core capability' {
        $script:nativeStatus.Available=$true;$script:nativeStatus.Modules.Search=$true
        foreach($name in @('Invoke-PcaiNativeDirectoryManifest','Invoke-PcaiNativeFileSearch','Invoke-PcaiNativeContentSearch')){$script:presentCommands[$name]=$true}
        $result=Get-PcaiCapabilities
        (Get-OperationRow $result ContentSearch).RustAvailable|Should -BeTrue
        (Get-OperationRow $result TokenEstimate).RustAvailable|Should -BeFalse
        $result.Features.ContentSearch|Should -BeTrue;$result.Features.PromptAssembly|Should -BeFalse
    }
    It 'accepts the real unavailable-status null Modules shape under StrictMode Latest' {
        # Exact legal predecessor producer shape: Private/Initialize-PcaiNative.ps1:300-308.
        Set-StrictMode -Version Latest
        $script:nativeStatus.Modules=$null
        $result=Get-PcaiCapabilities -ErrorAction Stop
        @($result.BackendCoverage).Count|Should -Be 6
        foreach($row in $result.BackendCoverage){$row.RustAvailable|Should -BeFalse}
    }
    It 'keeps native operation flags boolean for known unavailable null Modules' {
        $script:nativeStatus.Modules=$null
        Set-StrictMode -Off
        $result=Get-PcaiCapabilities
        foreach($name in @('DirectoryManifest','FileSearch','ContentSearch','DuplicateScan','LogSearch','DiskUsage','MemoryStats','FsReplace')){$result.Features.$name|Should -BeOfType ([bool]);$result.Features.$name|Should -BeFalse}
    }
    It 'does not prefer a native backend without its callable token wrapper' -Tag ProposedPolicy {
        $script:nativeStatus=New-SyntheticCapabilityStatus $true
        $row=Get-OperationRow (Get-PcaiCapabilities) TokenEstimate
        $row.PowerShellSurface|Should -BeFalse;$row.CoverageState|Should -BeExactly 'Unavailable'
        $row.PreferredBackend|Should -BeExactly 'PowerShell'
    }
    It 'does not prefer a native search backend without its callable wrapper' -Tag ProposedPolicy {
        $script:nativeStatus=New-SyntheticCapabilityStatus $true
        $row=Get-OperationRow (Get-PcaiCapabilities) ContentSearch
        $row.PowerShellSurface|Should -BeFalse;$row.PreferredBackend|Should -BeExactly 'PowerShell/rg'
    }
    It 'does not fabricate a disk PowerShell surface from a native module flag' -Tag ProposedPolicy {
        $script:nativeStatus=New-SyntheticCapabilityStatus $true
        $row=Get-OperationRow (Get-PcaiCapabilities) DiskUsage
        $row.PowerShellSurface|Should -BeFalse;$row.CoverageState|Should -BeExactly 'Unavailable';$row.PreferredBackend|Should -BeExactly 'PowerShell'
    }
    It 'does not turn an unknown <Field> flag into a known native capability' -Tag ProposedPolicy -TestCases @(@{Field='CoreAvailable';Value='false'},@{Field='Search';Value='unknown'},@{Field='Search';Value=$null}) {
        param($Field,$Value)
        $script:nativeStatus=New-SyntheticCapabilityStatus $true
        if($Field-ceq'CoreAvailable'){$script:nativeStatus.CoreAvailable=$Value}else{$script:nativeStatus.Modules.Search=$Value}
        # Proposed malformed-metadata policy: actionable refusal, never invented true or zero.
        {Get-PcaiCapabilities -ErrorAction Stop}|Should -Throw
    }
    It 'preserves a status producer failure instead of claiming usable fallback metadata' {
        Mock Get-PcaiNativeStatus {throw [IO.InvalidDataException]::new('Synthetic status unavailable')}
        {Get-PcaiCapabilities -ErrorAction Stop}|Should -Throw '*Synthetic status unavailable*'
    }
    It 'returns requested GPU metadata as a stable array with <Count> entries' -TestCases @(@{Count=0},@{Count=1},@{Count=2}) {
        param($Count)
        $script:gpuRows=@(for($i=0;$i-lt$Count;$i++){[pscustomobject]@{Name=('synthetic-gpu-'+$i);DriverVersion='synthetic-driver';Status='synthetic-ok';PNPDeviceID=('synthetic-id-'+$i)}})
        $script:expectedGpuCalls=1
        $result=Get-PcaiCapabilities -IncludeGpu
        @($result.Gpu).Count|Should -Be $Count
        if($Count){$result.Gpu[0].Name|Should -BeExactly 'synthetic-gpu-0';$result.Gpu[0].PnpDeviceId|Should -BeExactly 'synthetic-id-0'}
    }
    It 'contains an optional GPU collector failure without inventing inventory' {
        $script:expectedGpuCalls=1
        Mock Get-CimInstance {throw 'Synthetic GPU failure'}
        $result=Get-PcaiCapabilities -IncludeGpu
        @($result.Gpu).Count|Should -Be 0
        @($result.BackendCoverage).Count|Should -Be 6
    }
    It 'does not call an absent service-health command even when explicitly requested' {
        (Get-PcaiCapabilities -IncludeServices).Services|Should -BeNullOrEmpty
    }
    It 'preserves the requested service producer payload without contacting providers itself' {
        $script:presentCommands['Get-PcaiServiceHealth']=$true;$script:expectedServiceCalls=1
        $result=Get-PcaiCapabilities -IncludeServices
        $result.Services.Schema|Should -BeExactly 'synthetic-health'
        @($result.Services.Services).Count|Should -Be 1
        $result.Services.Services[0].Name|Should -BeExactly 'inert'
        $result.Services.Services[0].Healthy|Should -BeTrue
    }
    It 'contains an optional service producer exception without fabricating healthy services' {
        $script:presentCommands['Get-PcaiServiceHealth']=$true;$script:expectedServiceCalls=1
        Mock Get-PcaiServiceHealth {throw 'Synthetic service failure'}
        (Get-PcaiCapabilities -IncludeServices).Services|Should -BeNullOrEmpty
    }
}
