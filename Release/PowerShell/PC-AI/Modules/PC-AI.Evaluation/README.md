# PC-AI Evaluation Framework

Comprehensive LLM evaluation suite for testing PC-AI inference backends, including automated metrics, LLM-as-judge patterns, regression testing, and A/B testing.

## Quick Start

```powershell
# Import the module
Import-Module PC-AI.Evaluation

# Create an evaluation suite
$suite = New-EvaluationSuite -Name "DiagnosticQuality" `
    -Metrics @('latency', 'similarity', 'coherence') `
    -IncludeDefaultMetrics

# Load test cases
$testCases = Get-EvaluationDataset -Name 'diagnostic'
foreach ($tc in $testCases) {
    $suite.AddTestCase($tc)
}

# Run evaluation
$results = Invoke-EvaluationSuite -Suite $suite `
    -Backend 'llamacpp' `
    -ModelPath "C:\models\llama-3.2-1b.gguf"

# View results
Get-EvaluationResults -Suite $suite -Format summary
```

## Features

### 1. Automated Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `latency` | Response generation time | Lower is better |
| `throughput` | Tokens per second | Higher is better |
| `memory` | Memory usage in MB | Lower is better |
| `similarity` | Semantic similarity to expected | 0-1 |
| `groundedness` | Response grounded in context | 0-1 |
| `coherence` | Logical flow and consistency | 0-1 |
| `toxicity` | Harmful content detection | 0-1 (lower is safer) |
| `accuracy` | Factual correctness | 0-1 |

### 2. LLM-as-Judge

Use an LLM to evaluate response quality:

```powershell
$judgment = Invoke-LLMJudge -Response $response `
    -Question $question `
    -Criteria @('accuracy', 'helpfulness', 'clarity')
```

Pairwise comparison:

```powershell
$comparison = Compare-ResponsePair -Question $q `
    -ResponseA $responseFromModelA `
    -ResponseB $responseFromModelB
```

Diagnostic-specific evaluation:

```powershell
$quality = Evaluate-DiagnosticQuality -DiagnosticOutput $output `
    -DiagnosticInput $input
```

### 3. Regression Testing

Create baselines and detect performance regressions:

```powershell
# Create baseline
$baseline = New-BaselineSnapshot -Name "v1.0.0" -Suite $suite -Backend 'llamacpp'

# Later: test for regression
$regression = Test-ForRegression -BaselineName "v1.0.0" -Suite $currentSuite
```

### 4. A/B Testing

Compare different backends or models:

```powershell
# Create A/B test
$test = New-ABTest -Name "llamacpp-vs-mistralrs" `
    -VariantAName "llamacpp" `
    -VariantBName "mistralrs"

# Add results
Add-ABTestResult -TestName $test.Name -Variant "A" -Score 0.85
Add-ABTestResult -TestName $test.Name -Variant "B" -Score 0.90

# Analyze
$analysis = Get-ABTestAnalysis -TestName $test.Name
```

## Built-in Datasets

| Dataset | Description | Test Cases |
|---------|-------------|------------|
| `diagnostic` | PC-AI diagnostic scenarios | ~10 |
| `general` | General LLM capabilities | ~3 |
| `safety` | Safety and refusal testing | ~2 |

Load custom datasets:

```powershell
$dataset = Get-EvaluationDataset -Name "path/to/custom.json"
```

## Evaluation Runner Script

Run comprehensive evaluations from command line:

```powershell
# Single backend evaluation
.\Tests\Evaluation\Invoke-InferenceEvaluation.ps1 `
    -Backend llamacpp `
    -ModelPath "C:\models\model.gguf" `
    -Dataset diagnostic

# A/B test between backends
.\Tests\Evaluation\Invoke-InferenceEvaluation.ps1 `
    -ABTest "llamacpp:mistralrs" `
    -ModelPath "C:\models\model.gguf"

# Create baseline
.\Tests\Evaluation\Invoke-InferenceEvaluation.ps1 `
    -Backend llamacpp `
    -CreateBaseline "v1.0.0-baseline"

# Compare to baseline
.\Tests\Evaluation\Invoke-InferenceEvaluation.ps1 `
    -Backend llamacpp `
    -CompareBaseline "v1.0.0-baseline"
```

## Test Dataset Format

Custom datasets use JSON format:

```json
[
  {
    "id": "test-001",
    "category": "diagnostic",
    "prompt": "Analyze this diagnostic...",
    "expected": "Expected response content",
    "context": {"type": "disk-health"},
    "tags": ["disk", "smart"]
  }
]
```

## Integration with CI/CD

Add to GitHub Actions workflow:

```yaml
- name: Run Evaluation Suite
  shell: pwsh
  run: |
    Import-Module ./Modules/PC-AI.Evaluation
    $suite = New-EvaluationSuite -Name "CI-Eval" -Metrics @('latency', 'similarity')
    # ... configure and run
    $results = Get-EvaluationResults -Format summary
    if ($results.PassRate -lt 80) { exit 1 }
```

## Metrics Interpretation

### Pass Rate Thresholds
- **≥ 90%**: Excellent - production ready
- **≥ 80%**: Good - minor improvements needed
- **≥ 60%**: Fair - requires attention
- **< 60%**: Poor - significant issues

### Effect Size (A/B Testing)
- **< 0.2**: Negligible difference
- **0.2-0.5**: Small effect
- **0.5-0.8**: Medium effect
- **> 0.8**: Large effect

## Architecture

```
PC-AI.Evaluation/
├── PC-AI.Evaluation.psd1    # Module manifest
├── PC-AI.Evaluation.psm1    # Core module
├── Datasets/
│   └── pcai-diagnostic-eval.json
├── Baselines/               # Saved baseline snapshots
└── Results/                 # Evaluation results
```

## Dependencies

- PowerShell 7.0+
- PcaiInference module (for native backend testing)
- Pester 5.x (for running tests)

## Running Tests

```powershell
Invoke-Pester -Path ./Tests/Evaluation/PC-AI.Evaluation.Tests.ps1 -Output Detailed
```
