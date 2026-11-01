BeforeAll {
    $script:HelperPath = (Resolve-Path (Join-Path $PSScriptRoot '../../Modules/PC-AI.LLM/Private/LLM-Helpers.ps1')).ProviderPath
    $script:ChildScript = Join-Path $TestDrive 'endpoint-consumer.ps1'
    @'
param([string]$HelperPath,[string]$Mapping)
$ErrorActionPreference='Stop'
Import-Module Microsoft.PowerShell.Management,Microsoft.PowerShell.Utility -ErrorAction Stop
$PSModuleAutoLoadingPreference='None'
. $HelperPath
if($ExecutionContext.SessionState.InvokeCommand.GetCommand('Resolve-PcaiPath',[Management.Automation.CommandTypes]::All)){throw 'Fresh consumer must have no optional resolver'}
$missingErrors=@()
$missingResult=Resolve-PcaiEndpoint -ApiUrl 'http://127.0.0.1:18765' -ErrorVariable +missingErrors -ErrorAction Stop
function Resolve-PcaiPath {param($PathType) if($PathType-cne'HVSockConfig'){throw 'Wrong path contract'};$Mapping}
$presentErrors=@()
$presentResult=Resolve-PcaiEndpoint -ApiUrl 'hvsock://fixture' -ErrorVariable +presentErrors -ErrorAction Stop
$explicitErrors=@()
$explicitResult=Resolve-PcaiEndpoint -ApiUrl 'hvsock://fixture' -ConfigPath $Mapping -ErrorVariable +explicitErrors -ErrorAction Stop
function Resolve-PcaiPath {param($PathType) throw 'Owned resolver failure'}
$resolverFailure=$null
try{Resolve-PcaiEndpoint -ApiUrl 'http://127.0.0.1:18765' -ErrorAction Stop|Out-Null}catch{$resolverFailure=$_}
@{MissingResult=$missingResult;MissingErrorCount=$missingErrors.Count;PresentResult=$presentResult;PresentErrorCount=$presentErrors.Count;ExplicitResult=$explicitResult;ExplicitErrorCount=$explicitErrors.Count;ResolverFailureRetained=($null-ne$resolverFailure);ResolverFailureMessage=if($null-ne$resolverFailure){$resolverFailure.Exception.Message}else{$null}}|ConvertTo-Json -Compress
'@ | Set-Content -LiteralPath $script:ChildScript -Encoding utf8NoBOM
    $mapping = Join-Path $TestDrive 'hvsock-map.txt'
    'fixture:unused:localhost:18766' | Set-Content -LiteralPath $mapping -Encoding utf8NoBOM
    $raw = & (Join-Path $PSHOME $(if ($IsWindows) {'pwsh.exe'} else {'pwsh'})) -NoLogo -NoProfile -NonInteractive -File $script:ChildScript -HelperPath $script:HelperPath -Mapping $mapping
    if ($LASTEXITCODE -ne 0) { throw 'Real standalone endpoint consumer failed.' }
    $script:Actual = $raw | ConvertFrom-Json
}

Describe 'Standalone endpoint optional resolver errors' -Tag 'Unit', 'LLM', 'Portable' {
    # Protects standalone HTTP callers; detects a retained command-not-found error.
    # Needs a fresh no-profile child. Breadcrumb: HTTP R3 original error metadata.
    It 'returns an explicit HTTP endpoint without emitting an error for an absent optional resolver' {
        $script:Actual.MissingResult | Should -Be 'http://127.0.0.1:18765'
        $script:Actual.MissingErrorCount | Should -Be 0
    }
    # Protects installed resolver integration; detects lost mapping discovery.
    # Needs the real owned resolver function and mapping file. Breadcrumb: child consumer above.
    It 'uses the actual optional resolver when installed' {
        $script:Actual.PresentResult | Should -Be 'http://localhost:18766'
        $script:Actual.PresentErrorCount | Should -Be 0
    }
    # Protects explicit configuration; detects discovery errors or changed endpoint mapping.
    # Needs the owned mapping file. Breadcrumb: explicit ConfigPath child call above.
    It 'uses an explicit mapping without discovery errors' {
        $script:Actual.ExplicitResult | Should -Be 'http://localhost:18766'
        $script:Actual.ExplicitErrorCount | Should -Be 0
    }
    # Protects original failures; detects accidental swallowing of resolver exceptions.
    # Needs the throwing owned resolver. Breadcrumb: retained child ErrorRecord above.
    It 'retains a genuine resolver failure' {
        $script:Actual.ResolverFailureRetained | Should -BeTrue
        $script:Actual.ResolverFailureMessage | Should -Be 'Owned resolver failure'
    }
}
