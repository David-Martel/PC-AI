#Requires -Version 5.1

<#
.SYNOPSIS
    Validates .github/workflows/*.yml: YAML well-formedness plus the syntax of
    every embedded PowerShell `run:` block.

.DESCRIPTION
    Two classes of defect have shipped to this repository undetected because
    nothing parsed workflow files before GitHub did:

      1. maintenance.yml's Credential Scan used ["\'] inside a single-quoted
         PowerShell string. PowerShell escapes ' by DOUBLING it, so the string
         terminated early and every scheduled run died with "Unexpected token
         ']'". The scan had never executed once.

      2. release-cuda.yml embedded PowerShell here-strings whose terminators
         ("@) must sit at column 0. That broke out of the YAML block scalar and
         made the file invalid YAML, so the CUDA release workflow could never
         load -- and it had never run.

    Both are caught here. `${{ ... }}` expressions are stubbed before parsing
    because GitHub substitutes them before any shell sees the script.

.PARAMETER WorkflowPath
    Directory containing workflow files. Defaults to .github/workflows relative
    to the repository root.

.PARAMETER PassThru
    Emit the finding objects instead of only writing to the host.

.EXAMPLE
    pwsh Tools/Test-WorkflowSyntax.ps1

.EXAMPLE
    pwsh Tools/Test-WorkflowSyntax.ps1 -PassThru | Format-Table
#>

[CmdletBinding()]
param(
    [string]$WorkflowPath,
    [switch]$PassThru
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $WorkflowPath) {
    $WorkflowPath = Join-Path $repoRoot '.github/workflows'
}

if (-not (Test-Path $WorkflowPath)) {
    throw "Workflow directory not found: $WorkflowPath"
}

$files = @(Get-ChildItem -Path $WorkflowPath -Filter *.yml -File)
if ($files.Count -eq 0) {
    throw "No workflow files found under $WorkflowPath - refusing to report success on an empty scan."
}

# powershell-yaml gives real YAML parsing. Without it we can still parse the
# embedded PowerShell, but we must say so rather than imply a full check ran.
$haveYaml = $null -ne (Get-Module -ListAvailable -Name 'powershell-yaml')
if ($haveYaml) {
    Import-Module powershell-yaml -ErrorAction Stop
} else {
    Write-Warning 'powershell-yaml is not installed; YAML structure will not be validated. Install-Module powershell-yaml -Scope CurrentUser'
}

$findings = [System.Collections.Generic.List[psobject]]::new()

function Add-Finding {
    param([string]$File, [string]$Scope, [string]$Kind, [string]$Message)
    $findings.Add([PSCustomObject]@{
            File    = $File
            Scope   = $Scope
            Kind    = $Kind
            Message = $Message
        })
}

function Test-EmbeddedPowerShell {
    param([string]$File, [string]$Scope, [string]$Code)

    # GitHub expands ${{ ... }} before invoking the shell, so a raw expression
    # is not a PowerShell syntax error. Replace with a bare token.
    $stubbed = [regex]::Replace($Code, '\$\{\{[^}]*\}\}', 'GHA_EXPR')
    $parseErrors = $null
    $null = [System.Management.Automation.Language.Parser]::ParseInput($stubbed, [ref]$null, [ref]$parseErrors)
    if ($parseErrors -and $parseErrors.Count -gt 0) {
        foreach ($parseError in $parseErrors) {
            Add-Finding -File $File -Scope $Scope -Kind 'PowerShellSyntax' `
                -Message "line $($parseError.Extent.StartLineNumber): $($parseError.Message)"
        }
    }
}

$stepsScanned = 0

foreach ($file in $files) {
    $raw = Get-Content -LiteralPath $file.FullName -Raw

    # GitHub evaluates ${{ ... }} ANYWHERE in a workflow, including inside a
    # run: block's shell comments. An empty or whitespace-only expression is
    # rejected at workflow-load time with "An expression was expected", which
    # kills the workflow before a single step runs. powershell-yaml parses such
    # a file happily, so this has to be checked separately.
    foreach ($expr in [regex]::Matches($raw, '\$\{\{(.*?)\}\}')) {
        if ([string]::IsNullOrWhiteSpace($expr.Groups[1].Value)) {
            $lineNumber = ($raw.Substring(0, $expr.Index) -split "`n").Count
            Add-Finding -File $file.Name -Scope "line $lineNumber" -Kind 'EmptyActionsExpression' `
                -Message 'Empty ${{ }} expression. GitHub rejects the whole workflow with "An expression was expected".'
        }
    }

    if (-not $haveYaml) { continue }

    $doc = $null
    try {
        $doc = ConvertFrom-Yaml -Yaml $raw -ErrorAction Stop
    } catch {
        Add-Finding -File $file.Name -Scope '(document)' -Kind 'InvalidYaml' -Message $_.Exception.Message
        continue
    }

    if (-not $doc.ContainsKey('jobs')) {
        Add-Finding -File $file.Name -Scope '(document)' -Kind 'NoJobs' -Message 'Workflow declares no jobs.'
        continue
    }

    $workflowShell = $null
    if ($doc.ContainsKey('defaults') -and $doc.defaults -and $doc.defaults.ContainsKey('run') `
            -and $doc.defaults.run -and $doc.defaults.run.ContainsKey('shell')) {
        $workflowShell = $doc.defaults.run.shell
    }

    foreach ($jobName in $doc.jobs.Keys) {
        $job = $doc.jobs[$jobName]
        if (-not $job -or -not $job.ContainsKey('steps')) { continue }

        if (-not $job.ContainsKey('runs-on') -and -not $job.ContainsKey('uses')) {
            Add-Finding -File $file.Name -Scope "job '$jobName'" -Kind 'MissingRunsOn' `
                -Message 'Job has neither runs-on nor uses.'
        }

        $jobShell = $workflowShell
        if ($job.ContainsKey('defaults') -and $job.defaults -and $job.defaults.ContainsKey('run') `
                -and $job.defaults.run -and $job.defaults.run.ContainsKey('shell')) {
            $jobShell = $job.defaults.run.shell
        }
        $runsOn = if ($job.ContainsKey('runs-on')) { "$($job['runs-on'])" } else { '' }

        foreach ($step in @($job.steps)) {
            if (-not $step -or -not $step.ContainsKey('run')) { continue }

            $shell = if ($step.ContainsKey('shell')) { $step.shell }
            elseif ($jobShell) { $jobShell }
            elseif ($runsOn -like 'windows*') { 'powershell' }
            else { 'bash' }

            if ($shell -notin @('pwsh', 'powershell')) { continue }

            $stepsScanned++
            $stepName = if ($step.ContainsKey('name')) { $step.name } else { '(unnamed)' }
            Test-EmbeddedPowerShell -File $file.Name -Scope "job '$jobName' / step '$stepName'" -Code $step.run
        }
    }
}

Write-Host "Scanned $($files.Count) workflow file(s) and $stepsScanned embedded PowerShell step(s)." -ForegroundColor Cyan

if ($haveYaml -and $stepsScanned -eq 0) {
    # Every workflow in this repo runs PowerShell somewhere. Zero means the
    # walker silently stopped matching, not that there is nothing to check.
    Write-Host 'No PowerShell steps were found. That indicates a broken scan, not a clean repo.' -ForegroundColor Red
    exit 1
}

if ($findings.Count -gt 0) {
    Write-Host "$($findings.Count) workflow problem(s):" -ForegroundColor Red
    foreach ($finding in $findings) {
        Write-Host "  [$($finding.Kind)] $($finding.File) :: $($finding.Scope)" -ForegroundColor Yellow
        Write-Host "      $($finding.Message)" -ForegroundColor Gray
    }
    if ($PassThru) { $findings }
    exit 1
}

Write-Host 'All workflow files parse and every embedded PowerShell step is syntactically valid.' -ForegroundColor Green
if ($PassThru) { $findings }
exit 0
