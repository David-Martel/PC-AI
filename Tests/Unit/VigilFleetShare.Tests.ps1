BeforeAll {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
    $script:helper = Join-Path $repoRoot 'Tools/SystemScripts/Networking/Mount-VigilFleetShare.ps1'
}

Describe 'Retired ASUS fleet share' {
    # Protects: callers can identify retirement during an offline preview.
    # Detects: an obsolete mount is still advertised as a usable plan.
    # Needs: PowerShell and Pester only.
    # Breadcrumb: Tools/SystemScripts/Networking/Mount-VigilFleetShare.ps1
    It 'reports retirement even for an old apply command in dry run' {
        $result = & $script:helper -Action Mount -Drive 'N' -Apply -DryRun -NativeTimeoutSeconds 8 -ProbeTimeoutMilliseconds 750
        $result.State | Should -BeExactly 'Retired'
        $result.Action | Should -BeExactly 'Mount'
        $result.Drive | Should -BeExactly 'N:'
        $result.Endpoint | Should -BeNullOrEmpty
        $result.Native | Should -BeNullOrEmpty
    }

    # Protects: a retired service cannot regain automatic triggers through the helper.
    # Detects: a task template with the old live mount action is still emitted.
    # Needs: PowerShell and Pester only.
    # Breadcrumb: Tools/SystemScripts/Networking/Mount-VigilFleetShare.ps1
    It 'refuses to emit a scheduler template' {
        { & $script:helper -ShowTaskTemplate } | Should -Throw '*retired*'
    }

    # Protects: scheduled invocations and manual apply calls fail before changing drive mappings.
    # Detects: a filesystem mutation remains reachable through an existing command line.
    # Needs: PowerShell and Pester only.
    # Breadcrumb: Tools/SystemScripts/Networking/Mount-VigilFleetShare.ps1
    It 'rejects live <Action> for the retired share' -ForEach @(
        @{ Action = 'Mount' }
        @{ Action = 'Unmount' }
    ) {
        { & $script:helper -Action $Action -Apply } | Should -Throw '*retired*'
    }

    # Protects: status and WhatIf remain suitable for offline diagnostics.
    # Detects: either availability or a scheduled action is offered for the retired endpoint.
    # Needs: PowerShell and Pester only.
    # Breadcrumb: Tools/SystemScripts/Networking/Mount-VigilFleetShare.ps1
    It 'reports retirement in status and WhatIf without an endpoint' {
        (& $script:helper).State | Should -BeExactly 'Retired'
        $plan = & $script:helper -Action Mount -Apply -WhatIf
        $plan.State | Should -BeExactly 'Retired'
        $plan.Endpoint | Should -BeNullOrEmpty
    }

    # Protects: both legacy help spellings remain available without network access.
    # Detects: an invocation for help falls through to an operational action.
    # Needs: PowerShell and Pester only.
    # Breadcrumb: Tools/SystemScripts/Networking/Mount-VigilFleetShare.ps1
    It 'describes retirement for short and long help requests' {
        (& $script:helper -h).Synopsis | Should -Match 'retirement'
        (& $script:helper --help).Synopsis | Should -Match 'retirement'
    }
}
