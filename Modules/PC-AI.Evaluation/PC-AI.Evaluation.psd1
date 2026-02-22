@{
    RootModule = 'PC-AI.Evaluation.psm1'
    ModuleVersion = '1.0.0'
    GUID = 'f8e3a2b1-c4d5-6e7f-8a9b-0c1d2e3f4a5b'
    Author = 'PC-AI Team'
    CompanyName = 'PC-AI'
    Copyright = '(c) 2026 PC-AI Team. All rights reserved.'
    Description = 'LLM Evaluation Framework for PC-AI Inference Backends'
    PowerShellVersion = '7.0'

    # PcaiInference is required for native backend evaluation
    RequiredModules = @(
        @{ ModuleName = 'PcaiInference'; ModuleVersion = '1.0.0'; Guid = '8e7c4f1a-3b9d-4c5e-8f0a-1b2c3d4e5f6a' }
    )

    # Script to run before importing module - validates DLL availability
    ScriptsToProcess = @('ValidateDependencies.ps1')

    FunctionsToExport = @(
        # Core Evaluation
        'New-EvaluationSuite'
        'Invoke-EvaluationSuite'
        'Get-EvaluationResults'

        # Metrics
        'Measure-InferenceLatency'
        'Measure-TokenThroughput'
        'Measure-MemoryUsage'
        'Compare-ResponseSimilarity'

        # LLM-as-Judge
        'Invoke-LLMJudge'
        'Compare-ResponsePair'
        'Evaluate-DiagnosticQuality'

        # Regression Testing
        'New-BaselineSnapshot'
        'Test-ForRegression'
        'Get-RegressionReport'

        # A/B Testing
        'New-ABTest'
        'Add-ABTestResult'
        'Get-ABTestAnalysis'

        # Test Datasets
        'Get-EvaluationDataset'
        'New-EvaluationTestCase'
        'Import-EvaluationDataset'
        'Export-EvaluationDataset'

        # Evaluation Run Utilities
        'Get-PcaiArtifactsRoot'
        'Initialize-EvaluationPaths'
        'New-PcaiEvaluationRunContext'
        'Get-EvaluationRunState'
        'Stop-EvaluationRun'

        # Quality Metrics (internal but exported for testing)
        'Measure-Coherence'
        'Measure-Toxicity'
        'Measure-Groundedness'
    )

    VariablesToExport = @()
    AliasesToExport = @()

    PrivateData = @{
        PSData = @{
            Tags = @('LLM', 'Evaluation', 'Testing', 'Inference', 'AI')
            ProjectUri = 'https://github.com/David-Martel/PC-AI'
        }
    }
}
