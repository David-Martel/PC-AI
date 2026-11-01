#Requires -Version 7.0
param(
    [string]$HelperSourcePath = (Join-Path $PSScriptRoot '../../Tools/GitOps/Invoke-GitOpsMonitors.ps1'),
    [string]$ProjectionSourceRoot = (Join-Path $PSScriptRoot '../../Tools/GitOps')
)
BeforeAll {
    $tokens=$null;$errors=$null
    $ast=[Management.Automation.Language.Parser]::ParseFile($HelperSourcePath,[ref]$tokens,[ref]$errors)
    if($errors.Count){throw 'Maintained GitOps helper must parse'}
    $definition=@($ast.FindAll({param($n)$n-is[Management.Automation.Language.FunctionDefinitionAst]},$true)|Where-Object Name -CEQ Get-GitOpsResponse)
    if($definition.Count-ne1){throw 'One complete maintained classifier required'}
    Set-Item -LiteralPath function:Get-GitOpsResponse -Value $definition[0].Body.GetScriptBlock()
    function Invoke-GitOpsProcess {param($FilePath,$Arguments,$WorkingDirectory,$TimeoutSeconds)throw 'Unmocked GitHub/process boundary forbidden'}
    function Start-Process {throw 'Real process launch forbidden'}
    function Invoke-RestMethod {throw 'Real HTTP forbidden'}
    function Invoke-WebRequest {throw 'Real HTTP forbidden'}
}
Describe 'Maintained GitOps provider reasons' -Tag Unit,GitOps,Fast,Portable {
    BeforeEach {
        $script:response=[pscustomobject]@{ExitCode=1;Stdout='fixture-secret-DO-NOT-LOG C:/private/credential-file';Stderr=''}
        Mock Invoke-GitOpsProcess {$script:response}
        Mock Start-Process {throw 'Unexpected real process'}
        Mock Invoke-RestMethod {throw 'Unexpected real HTTP'}
        Mock Invoke-WebRequest {throw 'Unexpected real HTTP'}
    }
    AfterEach {
        Should -Invoke Invoke-GitOpsProcess -Exactly -Times 1 -ParameterFilter {
            $FilePath -ceq 'gh' -and $WorkingDirectory -ceq 'fixture-root' -and $TimeoutSeconds -eq 2 -and
            $Arguments.Count -eq 2 -and $Arguments[0] -ceq 'api' -and $Arguments[1] -ceq 'fixture/route'
        }
        Should -Invoke Start-Process -Exactly -Times 0
        Should -Invoke Invoke-RestMethod -Exactly -Times 0
        Should -Invoke Invoke-WebRequest -Exactly -Times 0
    }
    # Protects the original successful value/shape contract, independent of diagnostics.
    It 'retains the complete successful <Scenario> value' -ForEach @(
        @{Scenario='empty array';JSON='[]';Shape='Array'},
        @{Scenario='singleton array';JSON='[{"Name":"fixture"}]';Shape='Array'},
        @{Scenario='object';JSON='{"Name":"fixture","Zero":0,"Null":null}';Shape='Object'}
    ) {
        $script:response=[pscustomobject]@{ExitCode=0;Stdout=$JSON;Stderr=''}
        $result=Get-GitOpsResponse -Operation fixture -Arguments @('api','fixture/route') -WorkingDirectory fixture-root -TimeoutSeconds 2 -Shape $Shape
        $result.status|Should -Be Available
        $result.exit_code|Should -Be 0
        $result.reason|Should -BeNullOrEmpty
        ($result.value|ConvertTo-Json -Compress -Depth 5)|Should -BeExactly ((ConvertFrom-Json -InputObject $JSON -NoEnumerate)|ConvertTo-Json -Compress -Depth 5)
    }
    # Actual unchanged status assertions prevent enum enrichment from turning unavailable into safe.
    It 'retains <Scenario> availability semantics' -ForEach @(
        @{Scenario='generic404';Operation='code-scanning';Stderr='gh: Not Found (HTTP 404)';Status='Unavailable'},
        @{Scenario='explicit disabled feature';Operation='code-scanning';Stderr='gh: Code scanning is not enabled (HTTP 404)';Status='Unsupported'},
        @{Scenario='disabled phrase outside code scanning';Operation='billing';Stderr='gh: Code scanning is not enabled (HTTP 404)';Status='Unavailable'}
    ) {
        $script:response.Stderr=$Stderr
        $result=Get-GitOpsResponse -Operation $Operation -Arguments @('api','fixture/route') -WorkingDirectory fixture-root -TimeoutSeconds 2 -Shape Array
        $result.status|Should -BeExactly $Status
        $result.value|Should -BeNullOrEmpty
    }
    # Known provider forms are exact allowlist entries. Generic/ambiguous failures
    # retain RequestFailure and cannot imply feature, permission or authentication cause.
    It 'emits only safe bounded reason fields for <Scenario>' -ForEach @(
        @{Scenario='no analysis';Operation='code-scanning';Stderr='gh: no analysis found (HTTP 404)';Code='NoAnalysisFound';HTTP=404},
        @{Scenario='no analysis repository';Operation='code-scanning';Stderr='gh: No analysis found for this repository (HTTP 404)';Code='NoAnalysisFound';HTTP=404},
        @{Scenario='no analysis wrong operation';Operation='billing';Stderr='gh: no analysis found (HTTP 404)';Code='RequestFailure';HTTP=404},
        @{Scenario='unrecognized no-analysis suffix';Operation='code-scanning';Stderr='gh: no analysis found for private branch (HTTP 404)';Code='RequestFailure';HTTP=404},
        @{Scenario='unprefixed no-analysis phrase';Operation='code-scanning';Stderr='no analysis found (HTTP 404)';Code='RequestFailure';HTTP=404},
        @{Scenario='generic404';Operation='code-scanning';Stderr='gh: Not Found (HTTP 404)';Code='RequestFailure';HTTP=404},
        @{Scenario='explicit disabled';Operation='code-scanning';Stderr='gh: Code scanning is not enabled (HTTP 404)';Code='FeatureDisabled';HTTP=404},
        @{Scenario='HTTP100';Operation='runs';Stderr='gh: failure (HTTP 100)';Code='RequestFailure';HTTP=100},
        @{Scenario='HTTP599';Operation='runs';Stderr='gh: failure (HTTP 599)';Code='RequestFailure';HTTP=599},
        @{Scenario='HTTP401';Operation='runs';Stderr='gh: failure (HTTP 401)';Code='RequestFailure';HTTP=401},
        @{Scenario='HTTP403';Operation='runs';Stderr='gh: failure (HTTP 403)';Code='RequestFailure';HTTP=403},
        @{Scenario='HTTP429';Operation='runs';Stderr='gh: failure (HTTP 429)';Code='RequestFailure';HTTP=429},
        @{Scenario='HTTP503';Operation='runs';Stderr='gh: failure (HTTP 503)';Code='RequestFailure';HTTP=503},
        @{Scenario='out-of-range099';Operation='runs';Stderr='gh: failure (HTTP 099)';Code='RequestFailure';HTTP=$null},
        @{Scenario='out-of-range600';Operation='runs';Stderr='gh: failure (HTTP 600)';Code='RequestFailure';HTTP=$null},
        @{Scenario='four-digit code';Operation='runs';Stderr='gh: failure (HTTP 4040)';Code='RequestFailure';HTTP=$null},
        @{Scenario='ambiguous two codes';Operation='code-scanning';Stderr="gh: no analysis found (HTTP 404) and upstream HTTP 503";Code='RequestFailure';HTTP=$null},
        @{Scenario='duplicate status occurrences';Operation='runs';Stderr='gh: HTTP 404 and HTTP 404';Code='RequestFailure';HTTP=$null},
        @{Scenario='absent status';Operation='runs';Stderr='private failure fixture-secret-DO-NOT-LOG C:/private/credential-file';Code='RequestFailure';HTTP=$null}
    ) {
        $script:response.Stderr=$Stderr
        $result=Get-GitOpsResponse -Operation $Operation -Arguments @('api','fixture/route') -WorkingDirectory fixture-root -TimeoutSeconds 2 -Shape Array
        $result.PSObject.Properties.Name|Should -Contain reason_code
        $result.PSObject.Properties.Name|Should -Contain http_status
        $result.reason_code|Should -BeExactly $Code
        $result.http_status|Should -Be $HTTP
        $result.status|Should -BeExactly $(if($Code-ceq'FeatureDisabled'){'Unsupported'}else{'Unavailable'})
        $result.value|Should -BeNullOrEmpty
        ($result|ConvertTo-Json -Depth 5)|Should -Not -Match 'fixture-secret|credential-file|private branch|ErrorKind|InspectionError|Stderr|Stdout'
    }
    It 'collapses <Scenario> private launcher provenance to RequestFailure' -ForEach @(
        @{Scenario='negative launch sentinel';Exit=-1;Field='ErrorKind';Text='PrivateException fixture-secret-DO-NOT-LOG'},
        @{Scenario='private exception kind';Exit=1;Field='ErrorKind';Text='PrivateException fixture-secret-DO-NOT-LOG'},
        @{Scenario='inspection error';Exit=1;Field='InspectionError';Text='C:/private/credential-file fixture-secret-DO-NOT-LOG'}
    ) {
        $script:response.ExitCode=$Exit
        $script:response|Add-Member -NotePropertyName $Field -NotePropertyValue $Text
        $result=Get-GitOpsResponse -Operation runs -Arguments @('api','fixture/route') -WorkingDirectory fixture-root -TimeoutSeconds 2 -Shape Array
        $result.PSObject.Properties.Name|Should -Contain reason_code
        $result.reason_code|Should -BeExactly RequestFailure
        $result.http_status|Should -BeNullOrEmpty
        $result.value|Should -BeNullOrEmpty
        ($result|ConvertTo-Json -Depth 5)|Should -Not -Match 'fixture-secret|credential-file|PrivateException|ErrorKind|InspectionError'
    }
    It 'labels <Scenario> invalid JSON or shape without emitting its bytes' -ForEach @(
        @{Scenario='malformed';JSON='{fixture-secret-DO-NOT-LOG';Shape='Object'},
        @{Scenario='empty';JSON='';Shape='Array'},
        @{Scenario='null';JSON='null';Shape='Object'},
        @{Scenario='object instead of array';JSON='{"Secret":"fixture-secret-DO-NOT-LOG"}';Shape='Array'},
        @{Scenario='array instead of object';JSON='["fixture-secret-DO-NOT-LOG"]';Shape='Object'},
        @{Scenario='string instead of object';JSON='"fixture-secret-DO-NOT-LOG"';Shape='Object'},
        @{Scenario='boolean instead of object';JSON='true';Shape='Object'},
        @{Scenario='number instead of object';JSON='7';Shape='Object'}
    ) {
        $script:response=[pscustomobject]@{ExitCode=0;Stdout=$JSON;Stderr=''}
        $result=Get-GitOpsResponse -Operation fixture -Arguments @('api','fixture/route') -WorkingDirectory fixture-root -TimeoutSeconds 2 -Shape $Shape
        $result.status|Should -BeExactly Unavailable
        $result.value|Should -BeNullOrEmpty
        $result.PSObject.Properties.Name|Should -Contain reason_code
        $result.reason_code|Should -BeExactly InvalidResponse
        ($result|ConvertTo-Json -Depth 5)|Should -Not -Match 'fixture-secret'
    }
    # Protects launcher provenance precedence over provider-like stderr. Real
    # launch failure remains unavailable and does not become a feature assertion.
    It 'preserves RequestFailure when <Scenario> collides with known provider text' -ForEach @(
        @{Scenario='launch sentinel no-analysis404';Exit=-1;Field=$null;Stderr='gh: no analysis found (HTTP 404)'},
        @{Scenario='private kind feature-disabled404';Exit=1;Field='ErrorKind';Stderr='gh: Code scanning is not enabled (HTTP 404)'},
        @{Scenario='inspection error feature-disabled404';Exit=1;Field='InspectionError';Stderr='gh: Code scanning is not enabled (HTTP 404)'}
    ) {
        $script:response.ExitCode=$Exit;$script:response.Stderr=$Stderr
        if($Field){$script:response|Add-Member -NotePropertyName $Field -NotePropertyValue 'fixture-secret-DO-NOT-LOG C:/private/credential-file'}
        $result=Get-GitOpsResponse -Operation code-scanning -Arguments @('api','fixture/route') -WorkingDirectory fixture-root -TimeoutSeconds 2 -Shape Array
        $result.status|Should -BeExactly Unavailable
        $result.PSObject.Properties.Name|Should -Contain reason_code
        $result.reason_code|Should -BeExactly RequestFailure
        $result.http_status|Should -Be 404
        $result.value|Should -BeNullOrEmpty
        ($result|ConvertTo-Json -Depth 5)|Should -Not -Match 'fixture-secret|credential-file|ErrorKind|InspectionError'
    }
}
Describe 'Actual GitOps report inspection projections' -Tag Unit,GitOps,Fast,Portable {
    # Execute the actual maintained inspection pipeline extent, not a mirror.
    It '<Leaf> retains only safe inspection fields' -ForEach @(
        @{Leaf='Get-UpstreamReviews.ps1'},@{Leaf='Watch-WorkflowHealth.ps1'}
    ) {
        $tokens=$null;$errors=$null
        $ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $ProjectionSourceRoot $Leaf),[ref]$tokens,[ref]$errors)
        if($errors.Count){throw 'Report source must parse'}
        $pipelines=@($ast.FindAll({param($n)$n-is[Management.Automation.Language.PipelineAst]-and$n.Extent.Text-match'^\$inspections\s*\|\s*Select-Object\s+operation'},$true))
        $pipelines.Count|Should -Be 1
        $inspections=@([pscustomobject]@{operation='code-scanning';status='Unavailable';exit_code=1;reason='GitHub inspection failed.';reason_code='NoAnalysisFound';http_status=404;value=@{PrivateBilling=1000000};ErrorKind='fixture-secret-DO-NOT-LOG';Stderr='C:/private/credential-file'})
        $actual=& ([scriptblock]::Create($pipelines[0].Extent.Text))
        @($actual.PSObject.Properties.Name)|Should -Be @('operation','status','exit_code','reason','reason_code','http_status')
        $actual.reason_code|Should -BeExactly NoAnalysisFound
        $actual.http_status|Should -Be 404
        ($actual|ConvertTo-Json -Depth 5)|Should -Not -Match 'fixture-secret|credential-file|PrivateBilling|1000000|ErrorKind|Stderr'
    }
}
