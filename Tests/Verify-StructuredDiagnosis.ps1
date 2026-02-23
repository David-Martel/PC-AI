. (Join-Path $PSScriptRoot 'Helpers\Resolve-TestRepoRoot.ps1')

# Verification script for PC-AI Structured Diagnostic Output
$projectRoot = Resolve-TestRepoRoot -StartPath $PSScriptRoot
$modulesPath = Join-Path $projectRoot 'Modules'

# Import all modules
Get-ChildItem -Path $modulesPath -Directory | ForEach-Object {
	$manifest = Join-Path $_.FullName "$($_.Name).psd1"
	if (Test-Path $manifest) {
		Write-Host "Importing module: $($_.Name)" -ForegroundColor Gray
		Import-Module $manifest -Force -ErrorAction SilentlyContinue
	}
}

$testPath = Join-Path $projectRoot 'Native'
$reportPath = Join-Path $env:TEMP 'PC-AI-Test-Report.json'
$model = 'qwen2.5-coder:7b'

Write-Host "`nStep 1: Running Invoke-SmartDiagnosis with -SaveReport using $model..." -ForegroundColor Cyan
$result = Invoke-SmartDiagnosis -Path $testPath -AnalysisType Quick -SaveReport -OutputPath $reportPath -Model $model

Write-Host "`nStep 2: Verifying Report File Existence..." -ForegroundColor Cyan
if (Test-Path $reportPath) {
	Write-Host "[PASS] Report saved to $reportPath" -ForegroundColor Green
} else {
	Write-Host '[FAIL] Report not found!' -ForegroundColor Red
	exit 1
}

Write-Host "`nStep 3: Validating JSON Structure..." -ForegroundColor Cyan
try {
	$report = Get-Content $reportPath -Raw | ConvertFrom-Json
	Write-Host '[PASS] Valid JSON' -ForegroundColor Green

	Write-Host "`nStep 4: Checking Mandatory Metadata..." -ForegroundColor Cyan
	$fields = @('Metadata', 'LLMAnalysis', 'Timestamp', 'AnalysisType')
	foreach ($f in $fields) {
		if ($report.$f) {
			Write-Host "[PASS] Field '$f' exists" -ForegroundColor Green
		} else {
			Write-Host "[FAIL] Field '$f' is missing!" -ForegroundColor Red
		}
	}

	Write-Host "`nStep 5: Verifying LLM Analysis Schema..." -ForegroundColor Cyan
	$analysis = $report.LLMAnalysis
	$analysisFields = @('diagnosis_version', 'timestamp', 'findings', 'recommendations')
	foreach ($af in $analysisFields) {
		if ($analysis.$af) {
			Write-Host "[PASS] Analysis field '$af' exists" -ForegroundColor Green
		} else {
			Write-Host "[FAIL] Analysis field '$af' is missing!" -ForegroundColor Red
		}
	}

	Write-Host "`nStep 6: Checking PC-AI Tooling Version in Metadata..." -ForegroundColor Cyan
	if ($report.Metadata.pcai_version -eq '2.0.0') {
		Write-Host '[PASS] pcai_version is 2.0.0' -ForegroundColor Green
	} else {
		Write-Host "[FAIL] Incorrect pcai_version: $($report.Metadata.pcai_version)" -ForegroundColor Red
	}

} catch {
	Write-Host "[FAIL] JSON parsing failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`nVerification Complete!" -ForegroundColor Cyan
