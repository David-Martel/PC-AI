#Requires -Version 7.5
param([string]$CandidateRoot=(Join-Path $PSScriptRoot '../../Modules/PC-AI.Common'),
    [string]$SourceManifestSHA256,[string]$PwshPath=(Join-Path $PSHOME 'pwsh.exe'),
    [string]$EvidenceDirectory,[string]$SourceManifestPath=(Join-Path $PSScriptRoot 'capture-source-register.json'))
BeforeAll {
    . (Join-Path $PSScriptRoot 'Test-ProcessCaptureSources.ps1')
    $manifestPath=$SourceManifestPath
    if([string]::IsNullOrEmpty($SourceManifestSHA256)){$SourceManifestSHA256=(Get-FileHash -LiteralPath $manifestPath).Hash}
    $script:ProjectionManifest=Read-ProcessCaptureSources $CandidateRoot $PSScriptRoot $manifestPath $SourceManifestSHA256
    $script:Manifest=$script:ProjectionManifest
    . (Join-Path $CandidateRoot 'Public/Get-PcaiSharedCache.ps1')
    # Definitions only allow spies on the actual loader; no Add-Type invocation here.
    . (Join-Path $CandidateRoot 'Private/Initialize-PcaiProcessCapture.ps1')
}
Describe 'Current cache consumers at the declared transport boundary' {
    BeforeEach {
        Set-StrictMode -Version Latest
        $script:Config=[pscustomobject]@{Provider='redis';Enabled=$true;RedisCliPath='PUBLIC_INERT_CLI';RedisHost='localhost';RedisPort=16379;KeyPrefix='public[*]:';TimeoutMs=300}
        $script:PcaiSharedCacheProviderState.Signature='';$script:PcaiSharedCacheProviderState.Status=$null;$script:PcaiSharedCacheProviderState.CheckedAtUtc=[datetime]::MinValue
        Mock Get-PcaiExternalCacheConfig { $script:Config }
        Mock Invoke-PcaiProcessCapture { [pscustomobject]@{Success=$false;ExitCode=-3;TimedOut=$false;StdOut='';StdErr='PUBLIC_CAPTURE_FAILURE'} }
    }
    # Protects: Redis arguments remain individual literal strings and keep configured timeouts.
    # Detects: shell quoting/splitting, dropped Unicode or altered five-field result identity.
    # Needs: actual maintained Redis wrapper, public argv and declared capture-boundary spy only.
    # Breadcrumb: unchanged Invoke-PcaiRedisCli ArgumentList construction.
    It 'preserves literal Redis argv timeout and result fields' {
        $script:PublicResult=[pscustomobject]@{Success=$true;ExitCode=0;TimedOut=$false;StdOut='PUBLIC_OK';StdErr=''}
        Mock Invoke-PcaiProcessCapture { $script:PublicResult }
        $value=Invoke-PcaiRedisCli -Config $script:Config -Arguments @('SET','public π','quote" bracket[owned]') -TimeoutMs 701
        [object]::ReferenceEquals($value,$script:PublicResult)|Should -BeTrue
        @($value.PSObject.Properties.Name).Count|Should -Be 5
        Should -Invoke Invoke-PcaiProcessCapture -Times 1 -Exactly -ParameterFilter {
            $FilePath-ceq'PUBLIC_INERT_CLI'-and$TimeoutMs-eq701-and$Arguments.Count-eq8-and
            $Arguments[0]-ceq'-h'-and$Arguments[1]-ceq'localhost'-and$Arguments[2]-ceq'-p'-and$Arguments[3]-ceq'16379'-and
            $Arguments[4]-ceq'--raw'-and$Arguments[5]-ceq'SET'-and$Arguments[6]-ceq'public π'-and$Arguments[7]-ceq'quote" bracket[owned]'
        }
    }
    # Protects: failed bounded GET is a miss and cannot delete a remotely valid value.
    # Detects: failed/truncated response decoded or transport failure treated as corrupt payload.
    # Needs: actual GET consumer with availability-only collaborator and failed capture spy.
    # Breadcrumb: Get-PcaiExternalCacheEntryRecord result.Success gate precedes payload decode/delete.
    It 'treats a failed capture GET as a miss without deletion' {
        Mock Get-PcaiExternalCacheStatus { [pscustomobject]@{Enabled=$true;Available=$true} }
        Get-PcaiExternalCacheEntryRecord -Namespace 'owned' -Key 'k'|Should -BeNullOrEmpty
        Should -Invoke Invoke-PcaiProcessCapture -Times 1 -Exactly
        Should -Invoke Invoke-PcaiProcessCapture -Times 0 -Exactly -ParameterFilter { $Arguments-ccontains'DEL' }
    }
    # Protects: SET cannot report persistence when bounded capture fails.
    # Detects: partial OK or nonzero transport accepted without Success.
    # Needs: actual SET/payload serializer with public value, availability collaborator only.
    # Breadcrumb: Set-PcaiExternalCacheEntry boolean success-and-OK contract.
    It 'returns false for failed SET even with a partial OK payload' {
        Mock Get-PcaiExternalCacheStatus { [pscustomobject]@{Enabled=$true;Available=$true} }
        Mock Invoke-PcaiProcessCapture { [pscustomobject]@{Success=$false;ExitCode=-3;TimedOut=$false;StdOut='OK';StdErr='PUBLIC_CAPTURE_FAILURE'} }
        Set-PcaiExternalCacheEntry -Namespace 'owned' -Key 'k' -Value @{public='π'}|Should -BeFalse
        Should -Invoke Invoke-PcaiProcessCapture -Times 1 -Exactly -ParameterFilter { $Arguments[5]-ceq'SET' }
    }
    # Protects: partial SCAN cannot authorize namespace deletion.
    # Detects: consumer uses captured prefix after overflow/failure.
    # Needs: actual SCAN consumer with public matching key text and failed capture boundary.
    # Breadcrumb: Clear-PcaiExternalCache checks scan.Success before key parsing.
    It 'does not delete keys from failed SCAN output' {
        Mock Get-PcaiExternalCacheStatus { [pscustomobject]@{Enabled=$true;Available=$true} }
        Mock Invoke-PcaiProcessCapture { [pscustomobject]@{Success=$false;ExitCode=-3;TimedOut=$false;StdOut='public[*]:owned::k';StdErr='PUBLIC_CAPTURE_FAILURE'} }
        Clear-PcaiExternalCache -Namespace 'owned'
        Should -Invoke Invoke-PcaiProcessCapture -Times 1 -Exactly -ParameterFilter { $Arguments-ccontains'--scan' }
        Should -Invoke Invoke-PcaiProcessCapture -Times 0 -Exactly -ParameterFilter { $Arguments-ccontains'DEL' }
    }
    # Protects: bounded health failure remains unavailable and retains health TTL caching.
    # Detects: failed PONG promoted, repeated child launches inside current health window.
    # Needs: actual provider signature/status code, declared failed transport only.
    # Breadcrumb: Get-PcaiExternalCacheStatus literal five-second signature health cache.
    It 'caches unavailable health without launching repeatedly' {
        $first=Get-PcaiExternalCacheStatus -Refresh
        $second=Get-PcaiExternalCacheStatus
        $first.Enabled|Should -BeTrue;$first.Available|Should -BeFalse;$first.Reachable|Should -BeFalse
        $first.Error|Should -BeExactly 'PUBLIC_CAPTURE_FAILURE'
        [object]::ReferenceEquals($first,$second)|Should -BeTrue
        Should -Invoke Invoke-PcaiProcessCapture -Times 1 -Exactly
    }
    # Protects: portable memory use requires neither external process nor lazy compiler.
    # Detects: new capture initialization added to disabled provider or local-hit path.
    # Needs: current complete Public functions and actual initializer/capture spies, public value.
    # Breadcrumb: local-first Get/Set plus disabled Get-PcaiExternalCacheStatus.
    It 'keeps local memory cache usable with the external provider disabled' {
        $script:Config.Enabled=$false;$script:Config.Provider='memory'
        Mock Initialize-PcaiProcessCaptureRunner { throw 'PUBLIC_UNEXPECTED_INITIALIZER' }
        $status=Get-PcaiExternalCacheStatus -Refresh
        $status.Enabled|Should -BeFalse
        Set-PcaiSharedCacheEntry -Namespace 'capture-collaborator' -Key 'memory' -Value 'PUBLIC_MEMORY'
        Get-PcaiSharedCacheEntry -Namespace 'capture-collaborator' -Key 'memory'|Should -BeExactly 'PUBLIC_MEMORY'
        Should -Invoke Invoke-PcaiProcessCapture -Times 0 -Exactly
        Should -Invoke Initialize-PcaiProcessCaptureRunner -Times 0 -Exactly
    }
}
AfterAll {
    [void](Read-ProcessCaptureSources $CandidateRoot $PSScriptRoot $manifestPath $SourceManifestSHA256)
}
