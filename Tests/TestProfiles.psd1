@{
    # Paths are relative to the selected repository. Legacy globs preserve the
    # original Invoke-AllTests flat-file cohorts, including its four-suite All.
    Unit              = @{ Path = @('Tests/Unit') }
    Integration       = @{ Path = @('Tests/Integration') }
    All               = @{ Path = @('Tests/Unit', 'Tests/Integration') }
    LegacyUnit        = @{ Path = @('Tests/Unit/*.Tests.ps1') }
    LegacyIntegration = @{ Path = @('Tests/Integration/*.Tests.ps1') }
    LegacyE2E         = @{ Path = @('Tests/E2E/*.Tests.ps1') }
    LegacyFunctional  = @{ Path = @('Tests/Functional/*.Tests.ps1') }
    LegacyAll         = @{
        Path = @('Tests/Unit/*.Tests.ps1', 'Tests/Integration/*.Tests.ps1', 'Tests/E2E/*.Tests.ps1', 'Tests/Functional/*.Tests.ps1')
    }
    Portable          = @{ Path = @('Tests/Unit'); Tag = @('Portable') }
    # Exact maintained Windows acceptance cohort; do not narrow this profile.
    WindowsCI         = @{
        Path = @(
            'Tests/Unit'
            'Tests/Evaluation/PC-AI.Evaluation.Tests.ps1'
            'Tests/Integration/ModuleLoading.Tests.ps1'
            'Tests/Boot/WorkloadResourceProfile.Tests.ps1'
            'Tests/Boot/CloudClientGate.Tests.ps1'
            'Tests/Boot/ArchiveSourceCustody.Tests.ps1'
            'Tests/Boot/CloudCacheRepairIdentity.Tests.ps1'
            'Tests/Boot/CloudMountGate.Tests.ps1'
            'Tests/Boot/CloudMountGateRunner.Tests.ps1'
            'Tests/Boot/MachineCapabilities.Tests.ps1'
            'Tests/Boot/MediaToolchain.Tests.ps1'
            'Tests/Boot/HardwareMeasurementContracts.Tests.ps1'
            'Tests/Boot/DevModuleDeployment.Tests.ps1'
            'Tests/Boot/FleetToolRouting.Tests.ps1'
            'Tests/Boot/TelemetryPromptContracts.Tests.ps1'
            'Tests/Boot/PackagedRuntimeRoot.Tests.ps1'
            'Tests/Boot/ProfileBootstrap.Tests.ps1'
            'Tests/Boot/CrashProfileSafety.Tests.ps1'
            'Tests/Boot/CrashTelemetry.Tests.ps1'
        )
    }
}
