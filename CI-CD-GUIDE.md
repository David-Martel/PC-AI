# CI/CD Pipeline Guide

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline for PC_AI.

## Overview

The PC_AI project uses **GitHub Actions** for automated testing, security scanning, and releases. The pipeline is optimized for PowerShell-based Windows applications with cross-version compatibility testing.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Git Push/PR                          │
└─────────────────────┬───────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    ┌──────────┐          ┌────────────┐
    │   Lint   │          │  Security  │
    │ (PSAnalyzer) │     │   Scan     │
    └─────┬────┘          └─────┬──────┘
          │                     │
          │     ┌───────────────┘
          │     │
          ▼     ▼
    ┌──────────────┐
    │   Tests      │
    │ (PS 5.1/7.4) │
    └──────┬───────┘
           │
           ▼
    ┌────────────────┐
    │  Integration   │
    │   Tests        │
    └──────┬─────────┘
           │
           ▼
    ┌────────────────┐
    │   Coverage     │
    │   Report       │
    └────────────────┘
```

## Workflows

### Workflow Inventory

- **PowerShell Tests**: `.github/workflows/powershell-tests.yml`
- **Rust Inference Build & Test**: `.github/workflows/rust-inference.yml`
- **CUDA/CPU Native Releases**: `.github/workflows/release-cuda.yml`
- **Documentation Pipeline**: `.github/workflows/docs-pipeline.yml`
- **Evaluation Smoke**: `.github/workflows/evaluation-smoke.yml`
- **Tooling Automation**: `.github/workflows/tooling-automation.yml`
- **Security Scan**: `.github/workflows/security.yml`
- **Release**: `.github/workflows/release.yml`
- **Scheduled Checks**: `.github/workflows/scheduled-checks.yml`

### 1. PowerShell Tests (`.github/workflows/powershell-tests.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` branch

**Jobs:**

#### Lint Job
- Runs PSScriptAnalyzer with custom rules
- Validates PowerShell syntax
- Fails on errors and warnings

```powershell
# Run locally:
Invoke-ScriptAnalyzer -Path . -Recurse -Settings PSScriptAnalyzerSettings.psd1
```

#### Test Job
- Matrix strategy: PowerShell 5.1 and 7.4
- Runs Pester tests with coverage
- Uploads coverage to Codecov

```powershell
# Run locally:
.\Tests\.pester.ps1 -CI
```

#### Integration Job
- Tests module imports
- Validates script syntax
- Verifies end-to-end functionality

```powershell
# Run locally:
Get-ChildItem -Path Modules -Directory | ForEach-Object {
    Import-Module (Join-Path $_.FullName "$($_.Name).psd1") -Force
}
```

### 2. Security Scan (`.github/workflows/security.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests
- Weekly schedule (Monday at midnight UTC)

**Security Checks:**

1. **Hardcoded Credentials**
   - Scans for password patterns
   - Detects API keys and secrets
   - Checks for plain text credentials

2. **File Permissions**
   - Validates executable scripts
   - Checks for overly permissive files

3. **Module Manifests**
   - Validates `.psd1` files
   - Ensures proper metadata

4. **Dangerous Commands**
   - Detects `Invoke-Expression`
   - Flags destructive operations
   - Warns about privilege escalation

5. **Dependency Audit**
   - Lists required modules
   - Checks for known vulnerabilities

```powershell
# Run locally:
# Check for credentials
Select-String -Path *.ps1 -Pattern "password\s*=\s*['\"][^'\"]+['\"]"

# Validate manifests
Get-ChildItem -Recurse -Filter *.psd1 | ForEach-Object {
    Test-ModuleManifest $_.FullName
}
```

### 3. Rust Inference Build & Test (`.github/workflows/rust-inference.yml`)

**Triggers:**
- Push to `main`/`develop` for inference code paths
- PRs to `main`

**Jobs:**
- Cargo checks/tests/clippy/fmt for `Native/pcai_core/pcai_inference`
- Windows MSVC build
- Optional CUDA build (push only)
- Integration tests for the inference DLL

### 4. CUDA/CPU Native Releases (`.github/workflows/release-cuda.yml`)

**Triggers:**
- Version tags (`v*`) and manual dispatch

**What it does:**
- Builds CPU + CUDA binaries for `llamacpp` and `mistralrs`
- Packages artifacts from `.pcai/build/artifacts`
- Uploads release assets

### 5. Documentation Pipeline (`.github/workflows/docs-pipeline.yml`)

**Triggers:**
- Pushes that affect docs or tooling
- Manual dispatch

**What it does:**
- Runs `Tools/Invoke-DocPipeline.ps1 -Mode DocsOnly`
- Runs `Tools/generate-auto-docs.ps1 -BuildDocs`
- Generates API signature report
- Uploads `Reports/**` and `.pcai/**` artifacts

### 6. Evaluation Smoke (`.github/workflows/evaluation-smoke.yml`)

**Triggers:**
- Manual dispatch

**What it does:**
- Starts a local mock OpenAI-compatible server
- Runs `Tests/Evaluation/Invoke-InferenceEvaluation.ps1` against `-Backend http`
- Uploads `.pcai/evaluation/runs/**` outputs

### 7. Tooling Automation (`.github/workflows/tooling-automation.yml`)

**Triggers:**
- Manual dispatch with toggles

**What it does:**
- Runs helper tooling in `Tools/` on demand (docs, help coverage, FunctionGemma prep/tests)

### 8. Release (`.github/workflows/release.yml`)

**Triggers:**
- Git tags matching `v*` (e.g., `v1.0.0`)

**Release Process:**

1. **Extract Version** - Parse version from tag
2. **Run Tests** - Full test suite must pass
3. **Build Package** - Create release archive with:
   - All modules
   - Main scripts
   - Documentation
   - Installation script
4. **Generate Release Notes** - Auto-generated from template
5. **Create GitHub Release** - Upload artifacts and notes

For native inference binaries (CPU/CUDA), see **release-cuda.yml**.

```powershell
# Create a release:
git tag v1.0.0
git push origin v1.0.0
```

**Release Package Structure:**
```
PC-AI-1.0.0/
├── Modules/
│   ├── PC-AI.Hardware/
│   ├── PC-AI.Virtualization/
│   └── ...
├── Reports/
├── Config/
├── PC-AI.ps1
├── Install.ps1
├── DIAGNOSE.md
├── DIAGNOSE_LOGIC.md
└── README.md
```

## Artifacts and Local Outputs

- Build and evaluation outputs are stored in `.pcai/` (gitignored but `rg`-searchable).
- Evaluation runs create `.pcai\evaluation\runs\<timestamp-label>\` with `events.jsonl`, `progress.log`, and `summary.json`.
- CI workflows upload `Reports/**` and `.pcai/**` as artifacts when enabled.

### 9. Scheduled Checks (`.github/workflows/scheduled-checks.yml`)

**Triggers:**
- Daily at 6 AM UTC
- Manual workflow dispatch

**Automated Checks:**

1. **Dependency Check**
   - Checks PowerShell Gallery for updates
   - Reports outdated modules

2. **Workflow Validation**
   - Validates YAML syntax
   - Checks for common issues

3. **Link Check**
   - Scans documentation for broken links
   - Validates internal references

4. **Health Report**
   - Repository statistics
   - Module coverage
   - Configuration status

```powershell
# View health report:
Get-Content health-report.json | ConvertFrom-Json
```

## Pre-Commit Hooks

### Setup

```powershell
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Hooks Configured

1. **PSScriptAnalyzer** - Lint PowerShell code
2. **Syntax Check** - Validate PowerShell syntax
3. **Pester Tests** - Run unit tests
4. **Module Manifests** - Validate `.psd1` files
5. **Trailing Whitespace** - Remove trailing spaces
6. **End of File Fixer** - Ensure files end with newline
7. **YAML Check** - Validate YAML syntax
8. **JSON Check** - Validate JSON syntax
9. **Mixed Line Endings** - Enforce CRLF for Windows
10. **Large Files Check** - Prevent commits >500KB
11. **Merge Conflict Check** - Detect unresolved conflicts
12. **Prettier** - Format YAML, JSON, and Markdown

## Code Quality Standards

### PSScriptAnalyzer Rules

The project uses custom PSScriptAnalyzer rules defined in `PSScriptAnalyzerSettings.psd1`:

**Enabled Rules:**
- Avoid cmdlet aliases
- Avoid `Invoke-Expression`
- Use approved verbs
- Proper credential handling
- Consistent indentation (4 spaces)
- Consistent whitespace
- Open brace on same line
- Correct casing

**Excluded Rules:**
- `PSAvoidUsingWriteHost` - Allowed for user interaction

### Code Coverage

**Requirements:**
- Minimum 85% code coverage
- All critical paths must be tested
- Integration tests for each module

```powershell
# Generate coverage report:
.\Tests\.pester.ps1 -Coverage

# View coverage:
Invoke-Item Tests\coverage.html
```

## Editor Configuration

The `.editorconfig` file enforces consistent coding style across editors:

- **Indent Style:** Spaces (4 spaces for PowerShell)
- **Line Endings:** CRLF (Windows)
- **Charset:** UTF-8 with BOM for PowerShell
- **Max Line Length:** 120 characters
- **Trim Trailing Whitespace:** Yes
- **Insert Final Newline:** Yes

Supported editors: VS Code, Visual Studio, JetBrains IDEs, Sublime Text, Atom

## Local Development Workflow

### 1. Clone and Setup

```powershell
git clone https://github.com/yourusername/PC_AI.git
cd PC_AI

# Install development dependencies
Install-Module PSScriptAnalyzer -Scope CurrentUser -Force
Install-Module Pester -MinimumVersion 5.6.1 -Scope CurrentUser -Force

# Install pre-commit hooks
pre-commit install
```

### 2. Create a Feature Branch

```powershell
git checkout -b feature/my-new-feature
```

### 3. Make Changes

```powershell
# Edit code
code Modules/PC-AI.Hardware/PC-AI.Hardware.psm1

# Run linter
Invoke-ScriptAnalyzer -Path . -Recurse -Settings PSScriptAnalyzerSettings.psd1

# Run tests
.\Tests\.pester.ps1
```

### 4. Commit Changes

```powershell
# Pre-commit hooks run automatically
git add .
git commit -m "feat: Add new hardware detection feature"

# If hooks fail, fix issues and retry
Invoke-ScriptAnalyzer -Path ChangedFile.ps1 -Fix
git add ChangedFile.ps1
git commit -m "feat: Add new hardware detection feature"
```

### 5. Push and Create PR

```powershell
git push origin feature/my-new-feature

# Create PR via GitHub UI or CLI
gh pr create --title "Add new hardware detection" --body "Description of changes"
```

### 6. CI Pipeline Runs

- Lint check runs first
- Tests run on PS 5.1 and 7.4
- Security scan completes
- Integration tests verify module imports
- Coverage report generated

### 7. Merge

Once all checks pass and PR is approved, merge to `main`:

```powershell
gh pr merge --squash
```

## Release Process

### Creating a Release

1. **Update Version Numbers**
   ```powershell
   # Update module manifests
   Update-ModuleManifest -Path Modules/PC-AI.Hardware/PC-AI.Hardware.psd1 -ModuleVersion "1.1.0"
   ```

2. **Create and Push Tag**
   ```powershell
   git tag -a v1.1.0 -m "Release version 1.1.0"
   git push origin v1.1.0
   ```

3. **GitHub Actions Automatically:**
   - Runs full test suite
   - Builds release package
   - Generates release notes
   - Creates GitHub release
   - Uploads artifacts

4. **Manual Steps (Optional):**
   - Edit release notes on GitHub
   - Announce release
   - Update documentation

### Versioning Strategy

We follow **Semantic Versioning (SemVer)**:

- **MAJOR** (1.0.0) - Breaking changes
- **MINOR** (1.1.0) - New features, backward compatible
- **PATCH** (1.1.1) - Bug fixes, backward compatible

Examples:
```powershell
v1.0.0  # Initial release
v1.1.0  # Added USB diagnostics module
v1.1.1  # Fixed SMART status parsing bug
v2.0.0  # Breaking change: New diagnostics API
```

## Troubleshooting

### Pre-Commit Hooks Failing

```powershell
# Skip hooks temporarily (not recommended)
git commit --no-verify

# Fix specific hook
pre-commit run psscriptanalyzer --all-files

# Update hooks
pre-commit autoupdate
```

### Test Failures

```powershell
# Run tests with verbose output
.\Tests\.pester.ps1 -Verbose

# Run specific test
.\Tests\.pester.ps1 -TestName "PC-AI.Hardware"

# Debug test
.\Tests\.pester.ps1 -Debug
```

### PSScriptAnalyzer Issues

```powershell
# Auto-fix issues
Invoke-ScriptAnalyzer -Path . -Recurse -Fix

# Check specific file
Invoke-ScriptAnalyzer -Path Modules/PC-AI.Hardware/PC-AI.Hardware.psm1

# Exclude specific rules
Invoke-ScriptAnalyzer -Path . -ExcludeRule PSAvoidUsingWriteHost
```

### GitHub Actions Failures

1. **Check workflow logs** on GitHub Actions tab
2. **Run locally** to reproduce:
   ```powershell
   # Simulate CI environment
   $env:CI = $true
   .\Tests\.pester.ps1 -CI
   ```
3. **Check permissions** for repository secrets
4. **Validate YAML syntax** using online validators

## Best Practices

### Commit Messages

Follow **Conventional Commits**:

```
feat: Add new hardware detection feature
fix: Correct SMART status parsing
docs: Update CI/CD documentation
test: Add tests for USB diagnostics
chore: Update dependencies
refactor: Simplify network adapter code
perf: Optimize disk scanning
```

### Testing Guidelines

1. **Unit Tests** - Test individual functions
2. **Integration Tests** - Test module interactions
3. **Mocking** - Mock external dependencies (WMI, CIM)
4. **Coverage** - Aim for >85% coverage
5. **Edge Cases** - Test error conditions

Example test:
```powershell
Describe "Get-HardwareDiagnostics" {
    It "Returns diagnostic results" {
        $result = Get-HardwareDiagnostics
        $result | Should -Not -BeNullOrEmpty
    }

    It "Handles missing WMI data gracefully" {
        Mock Get-CimInstance { throw "WMI error" }
        { Get-HardwareDiagnostics } | Should -Not -Throw
    }
}
```

### Security Guidelines

1. **Never commit credentials** to the repository
2. **Use environment variables** for sensitive data
3. **Encrypt secrets** using GitHub Secrets
4. **Review dependencies** regularly
5. **Follow least privilege** principle

### Documentation Requirements

1. **Comment-based help** for all functions
2. **README** in each module directory
3. **Examples** in documentation
4. **Changelog** for releases

## Continuous Improvement

### Metrics to Track

1. **Test Coverage** - Maintain >85%
2. **Build Time** - Keep under 10 minutes
3. **Code Quality** - Zero PSScriptAnalyzer errors
4. **Security Issues** - Zero high-severity findings
5. **PR Turnaround** - Merge within 24 hours

### Regular Maintenance

- **Weekly:** Review scheduled check reports
- **Monthly:** Update dependencies
- **Quarterly:** Security audit
- **Annually:** Major version planning

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PSScriptAnalyzer Rules](https://github.com/PowerShell/PSScriptAnalyzer)
- [Pester Documentation](https://pester.dev/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)

## Support

For CI/CD issues:
1. Check this guide
2. Review GitHub Actions logs
3. Run tests locally
4. Open an issue on GitHub
