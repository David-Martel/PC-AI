# CI/CD Pipeline Guide

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline for PC_AI.

## Overview

The PC_AI project uses **GitHub Actions** for automated testing, security scanning, and releases. The workflow files are
the source of truth: this guide is rebuilt from `.github/workflows/*.yml`, and `Tools/validate-doc-accuracy.ps1` (check
`workflows`) fails when a workflow file is missing from this guide or when the guide names a workflow that does not exist.

**Which workflows run.** Workflow state is a repository setting on GitHub, not something the files record. As of
2026-11-22 (`gh workflow list --all`) these nine are `disabled_manually`: `changelog`, `ci`, `evaluation-smoke`,
`jules-review`, `maintenance`, `nvidia-validation`, `release-cuda`, `release` and `rust-guidelines`. The other ten are
active. The disabled `ci.yml` was the broad gate; every job in it runs on `windows-latest` and the repository has no
Windows runner. `linux-ci.yml` is the gate that runs today, on the default branch `main` (there is no `develop` branch).
The validator cannot see enabled state offline, so re-check with `gh workflow list --all` before relying on the table.

## Pipeline Architecture

```
                 Pull request / push to main
                              |
      +-----------------------+------------------------+
      |                                                |
 always (no path filter)                     path-filtered contract workflows
 linux-ci.yml                                functiongemma-runtime, media-upscale,
   portable-contracts                        native-override-admission, portable-bootstrap,
   pester-lint                               usb4-lifecycle, usb4-peer-ssh-access,
   rust-linux                                usb4-route-verification
   gitops-contracts  (calls gitops-contracts.yml)
   ci-gate           (hosted runner, needs the four above)
```

Manual and scheduled workflows sit outside this path: `tooling-automation` is active; `evaluation-smoke`,
`maintenance`, `jules-review`, `release`, `release-cuda`, `changelog`, `nvidia-validation`, `rust-guidelines` and `ci`
are disabled.

## Workflows

### Workflow Inventory

| Workflow file | Trigger | State (2026-11-22) |
|---|---|---|
| `.github/workflows/linux-ci.yml` | PR and push to `main`, manual | active |
| `.github/workflows/gitops-contracts.yml` | `workflow_call` (from `linux-ci.yml`), manual | active |
| `.github/workflows/portable-bootstrap.yml` | PR and push, path-filtered | active |
| `.github/workflows/functiongemma-runtime.yml` | PR and push, path-filtered | active |
| `.github/workflows/media-upscale.yml` | PR and push, path-filtered | active |
| `.github/workflows/native-override-admission.yml` | PR (path-filtered), manual | active |
| `.github/workflows/usb4-lifecycle.yml` | PR (path-filtered), manual | active |
| `.github/workflows/usb4-peer-ssh-access.yml` | PR (path-filtered), manual | active |
| `.github/workflows/usb4-route-verification.yml` | PR and push, path-filtered | active |
| `.github/workflows/tooling-automation.yml` | Manual with toggles | active |
| `.github/workflows/ci.yml` | PR and push to `main` | disabled_manually |
| `.github/workflows/rust-guidelines.yml` | PR and push touching Rust or Cargo files | disabled_manually |
| `.github/workflows/nvidia-validation.yml` | PR and push touching the GPU stack, manual | disabled_manually |
| `.github/workflows/maintenance.yml` | Monday 06:00 UTC, manual | disabled_manually |
| `.github/workflows/jules-review.yml` | Monday 06:00 UTC, issue label, manual | disabled_manually |
| `.github/workflows/evaluation-smoke.yml` | Manual | disabled_manually |
| `.github/workflows/release.yml` | Tags `v*` | disabled_manually |
| `.github/workflows/release-cuda.yml` | Tags `v*`, manual | disabled_manually |
| `.github/workflows/changelog.yml` | Push to `main` (not `CHANGELOG.md`), manual | disabled_manually |

Contract workflows run Linux legs on the `pc-ai-local` self-hosted runner when `PCAI_LINUX_RUNNER_ENABLED` is `true`
for trusted (non-fork) pull requests, and on `ubuntu-latest` otherwise; their Windows legs use `windows-latest`.

### Active workflows

#### `linux-ci.yml` (CI (Linux self-hosted))

The gate. Jobs `portable-contracts`, `pester-lint`, `rust-linux`, `gitops-contracts` and `ci-gate`. See "Linux
self-hosted gate" below for what each checks and what it does not cover.

#### `gitops-contracts.yml` (GitOps monitor contracts)

Job `contracts` on a Linux and a Windows leg. Runs the GitOps monitor fixture contracts
(`Tests/Unit/GitOpsMonitors.Tests.ps1`) with job-local Pester. Called by `linux-ci.yml` and runnable by hand.

#### `portable-bootstrap.yml` (Portable bootstrap)

Job `bootstrap`: profile-free bootstrap, Rust-tool discovery, module-manifest and benchmark-admission contracts and the
Python admission tests, on Linux and Windows legs, when Common, Acceleration, Native, Deploy Cargo or `Tests/Portable`
paths change. Details: [portable bootstrap](docs/portable-bootstrap.md).

#### `functiongemma-runtime.yml` (FunctionGemma runtime)

Job `runtime-tests` runs the runtime HTTP contracts and the shared core's CPU tests on
Windows and Ubuntu. See [FunctionGemma validation](docs/functiongemma-runtime-validation.md).

#### `media-upscale.yml` (Media upscale contracts)

Job `upscale-contracts`: optional ONNX upscale build and tests for `pcai_media` and `pcai_media_model`. Details:
[media upscale validation](docs/media-upscale-validation.md).

#### `native-override-admission.yml` (Native override admission)

Job `admission`: builds the host native core and runs the native-override admission contracts in fresh processes.
Details: [native override admission](docs/native-override-admission.md).

#### USB4 workflows

- `usb4-lifecycle.yml` (job `lifecycle`): lifecycle contracts for `pcai_tb_cli` and the USB4 networking scripts.
- `usb4-peer-ssh-access.yml` (job `fixtures`): peer SSH access fixture contracts.
- `usb4-route-verification.yml` (job `route-postcondition`): route postcondition verification. Details:
  [route verification](docs/usb4-route-verification.md).

#### `tooling-automation.yml` (Tooling Automation)

Manual dispatch with toggles on `windows-latest`. Runs helper tooling in `Tools/` on demand: documentation generation
and `validate-doc-accuracy.ps1`, help and tool-coverage scripts, FunctionGemma preparation and LLM validation helpers.

### Disabled workflows

These files stay in the repository and run again if re-enabled; do not treat them as current gates.

- **`ci.yml`** (CI): jobs `security-scan`, `rust-fmt`, `powershell-lint`, `rust-check`, `rust-test`, `powershell-test`,
  `build-llamacpp`, `integration-tests`, `dotnet-build`, `deploy-runtime` and `gate`, all on `windows-latest`. Stays
  disabled until a Windows runner for this repository exists.
- **`rust-guidelines.yml`** (Rust Guidelines): format check, clippy with the guideline lints, unit tests, `cargo audit`
  and a compliance scan for Rust and Cargo changes. Clippy now runs in `linux-ci.yml`.
- **`nvidia-validation.yml`** (NVIDIA Stack Validation): jobs `powershell-tests`, `rust-nvml-check` and
  `registry-validation` for the `PC-AI.Gpu` module and the NVIDIA and driver registries.
- **`maintenance.yml`** (Weekly Maintenance): jobs `security-scan` (credential scan, manifest validation, `cargo audit`),
  `dependency-health`, `compatibility` (PowerShell 5.1 tests, Rust MSRV), `documentation` and `health-report`.
- **`jules-review.yml`** (Jules Review): weekly, issue-label and manual Jules review jobs.
- **`evaluation-smoke.yml`** (Evaluation Smoke): manual run of `Tests/Evaluation/Invoke-InferenceEvaluation.ps1` against
  a local mock OpenAI-compatible server (`-Backend http`), uploading `.pcai/evaluation/runs/**`.
- **`release.yml`** (Release): on a `v*` tag, extracts the version, runs tests, builds the module release ZIP and its
  generated `Install.ps1`, writes release notes and creates the GitHub release. The
  [module smoke check](Tools/Test-PcaiReleaseModule.ps1) verifies that the staged module itself exports the required functions.
- **`release-cuda.yml`** (Release Native Binaries): on a `v*` tag or manual dispatch, builds and tests via `Build.ps1`,
  builds CPU and CUDA binaries for `llamacpp` and `mistralrs`, packages `.pcai/build/artifacts` and uploads release
  assets.
- **`changelog.yml`** (Changelog): regenerates `CHANGELOG.md` with git-cliff on pushes to `main`.

Run the equivalent checks locally when a disabled workflow would have covered your change:

```powershell
Invoke-ScriptAnalyzer -Path . -Recurse -Settings PSScriptAnalyzerSettings.psd1   # lint
.\Tests\.pester.ps1 -CI                                                         # Pester with CI settings
pwsh Tests\Invoke-PortableTests.ps1                                             # what linux-ci runs, locally
pwsh Tools\validate-doc-accuracy.ps1                                             # documentation accuracy gate
```

## Linux self-hosted gate

`.github/workflows/linux-ci.yml` ("CI (Linux self-hosted)") runs on persistent repo-scoped self-hosted Linux runners.
Its jobs select runners with `runs-on: [self-hosted, Linux, X64, pc-ai-local]`: the role label `pc-ai-local` and no host
label, so any runner that carries `pc-ai-local` can serve them. It exists because the main `CI` workflow is disabled:
every job in it runs on `windows-latest` and PC-AI has no Windows runner, so it cannot run on an available runner.
`CI` stays disabled until a Windows runner for this repo exists.

| Job | What it checks | Source of the check |
|---|---|---|
| `portable-contracts` | actionlint on the workflow, profile-free bootstrap, Rust-tool discovery, module-manifest and benchmark-admission contracts, Python admission tests | the Linux leg of `portable-bootstrap.yml`, without its .NET steps |
| `pester-lint` | the Pester files listed in `Tests/linux-ci-pester.txt` (Portable tag), then PSScriptAnalyzer at Error severity over `Modules/` | `Tests/Invoke-PortableTests.ps1` sections Pester and Lint |
| `rust-linux` | `cargo fmt --check`, clippy (`-D warnings`), `cargo nextest run` and `cargo test --doc` for `Native/pcai_core` with `--no-default-features --features server,ffi` | `Tests/Invoke-PortableTests.ps1` sections RustFmt, RustClippy, Rust |
| `ci-gate` | fails unless every job above succeeded; runs on a hosted runner because it checks out nothing, and on `pc-ai-local` it queued a median of 8 minutes behind other jobs | single check name for this workflow |

**What it does not cover.** The Windows-only PowerShell modules (CIM/WMI, registry, drivers, network adapters,
Hyper-V and WSL), the `PcaiNative` C# bridge (the runner image has no .NET SDK), CUDA builds and the NVIDIA stack.
This is a bounded Linux subset, not the PowerShell suite: the repository stays Windows-first (see CLAUDE.md, "Platform
scope"). No test was changed to make it pass on Linux. `Tests/linux-ci-pester.txt` lists only files that already pass
there; 22 other files fail on Linux because they call Windows APIs.

**Build environment on the runner.** `rust-linux` uses `.github/actions/setup-job-rust` to install stable Rust with
clippy and rustfmt in job-temporary `CARGO_HOME` and `RUSTUP_HOME` directories. Installer `HOME` and `ZDOTDIR` are
isolated too. This Cargo home does not inherit the image's Cargo-home configuration, so the job explicitly sets
`RUSTC_WRAPPER=sccache`. It calls `cargo` directly because the Rust sections of `Tests/Invoke-PortableTests.ps1`
clear that wrapper. `CARGO_INCREMENTAL=0` is explicit in the workflow because `.cargo/config.toml` enables
incremental builds, and an incremental unit is never cached by sccache. The job-temporary homes sit under a fixed
`$RUNNER_TEMP/pcai-rust` name, because sccache 0.19.0 keys Rust compiles on the absolute `CARGO_HOME` path even when
`SCCACHE_BASEDIRS` names it: a per-job `pcai-rust-<guid>` name gave 0 Rust hits in 640 compiles (run 44893813238).
The runner empties `RUNNER_TEMP` between jobs, so the homes stay job-local. The final step prints per-run
`sccache --show-stats`; compare `Cache hits (Rust)` across two runs on the same runner to see cross-run reuse.
`portable-contracts` also uses the job-local Rust action because its Python admission tests compile native Rust fixtures.
Runners also apply the committed machine profile's build-job cap while retaining any stricter existing Cargo limit.
Details and measurements are in the separate vigil-utils repository (`docs/rust-build-acceleration.md`, "Runner job
environment"); that file is not part of this repository.

**Artifacts.** `pester-lint` uploads `pester-linux.xml` only when the job fails, with `continue-on-error` and a
1-day retention. Nothing reads it, a passing result is already in the log, and a GitHub artifact quota must not turn a
green gate red. The fleet artifact store is not an option here, because this public workflow carries no secrets
(see "Trust boundary").

**Switch.** Every job is skipped until the repository variable `PCAI_LINUX_RUNNER_ENABLED` is `true`. Set it after
the runner is registered: `gh variable set PCAI_LINUX_RUNNER_ENABLED --body true -R David-Martel/PC-AI`. Without the
switch each pull request would queue jobs for a label that no runner carries.

**Trust boundary.** PC-AI is public and the runner is persistent. Fork pull requests are skipped by the job `if:`, but a
fork can edit the workflow, so the repository setting "Approval for running fork pull request workflows" must be
`all_external_contributors`. Check it before registering the runner:
`gh api repos/David-Martel/PC-AI/actions/permissions/fork-pr-contributor-approval`.

## Artifacts and Local Outputs

- Build and evaluation outputs are stored in `.pcai/` (gitignored but `rg`-searchable).
- Evaluation runs create `.pcai\evaluation\runs\<timestamp-label>\` with `events.jsonl`, `progress.log`, and `summary.json`.
- CI workflows upload `Reports/**` and `.pcai/**` as artifacts when enabled.

## Pre-Commit Hooks

### Setup

```powershell
# Install pre-commit (any isolated tool installer; this repo uses uv)
uv tool install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Hooks Configured

1. **PSScriptAnalyzer** - Lint PowerShell code
2. **Syntax Check** - Parse literal source paths with [Test-PowerShellSyntax.ps1](Tools/Test-PowerShellSyntax.ps1), without executing staged files; [safety tests](Tests/Unit/PowerShellSyntaxHook.Tests.ps1) and [hash-bound review](Tests/Unit/powershell-syntax-review.md) cover the configured entrypoint
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
git clone https://github.com/David-Martel/PC-AI.git
cd PC-AI

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

- `linux-ci.yml` runs on every pull request: portable contracts, the Pester subset, Rust fmt/clippy/tests and GitOps contracts
- Path-filtered contract workflows run when their paths change (see the Workflow Inventory)
- Disabled workflows do not run; run their checks locally when your change touches that area

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

3. **GitHub Actions (when `release.yml` and `release-cuda.yml` are enabled; both are `disabled_manually` as of 2026-11-22):**
   - Runs the test suite
   - Builds the release package
   - Generates release notes
   - Creates the GitHub release
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
# Hooks are never skipped: fix the failure instead. Fix a specific hook:
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

- **Weekly:** Review dependency and security reports (`maintenance.yml` produces them only while it is enabled)
- **Monthly:** Update dependencies
- **Quarterly:** Security audit
- **Annually:** Major version planning

## Additional Resources

### Evaluation smoke lifecycle

The manually dispatched [evaluation smoke workflow](.github/workflows/evaluation-smoke.yml)
uses environment variables for literal dispatch arguments, starts its Windows mock
helper with a hidden window, and stops the recorded process from an always-run step.
The [portable lifecycle tests](Tests/Unit/EvaluationSmokeWorkflow.Tests.ps1) execute
the cleanup and argument blocks without killing processes or contacting a model.
Their [review record](Tests/Unit/evaluation-smoke-review.md) describes the failing
controls and the [hash-bound register](Tests/Unit/evaluation-smoke-review.json).
The Linux allowlist runs these tests in portable CI. A passing lifecycle suite
does not qualify evaluation-module behavior, model quality, or GPU inference;
the actual mock evaluation must also complete before enabling the dispatch.

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
