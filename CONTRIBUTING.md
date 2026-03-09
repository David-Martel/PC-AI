# Contributing to PC-AI

Thank you for your interest in contributing to PC-AI. This guide will help you get started.

## Reporting Issues

- Use [GitHub Issues](https://github.com/David-Martel/PC-AI/issues) to report bugs
- Include your PowerShell version (`$PSVersionTable`), Windows version, and steps to reproduce
- For hardware-related issues, include relevant diagnostic output

## Suggesting Features

- Open a GitHub Issue with the `enhancement` label
- Describe the use case and expected behavior
- If proposing a new diagnostic module, outline what data it would collect

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/David-Martel/PC-AI.git
   cd PC-AI
   ```

2. Install prerequisites:
   - **PowerShell 7.0+** (`winget install Microsoft.PowerShell`)
   - **Pester 5+** (`Install-Module Pester -Force -Scope CurrentUser`)
   - **Optional**: Rust toolchain for native components (`rustup`)
   - **Optional**: .NET 8 SDK for C# interop layer

3. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Code Style

### PowerShell
- Follow [PSScriptAnalyzer](https://github.com/PowerShell/PSScriptAnalyzer) rules
- Use approved verbs (`Get-`, `Set-`, `Invoke-`, etc.)
- Include `[CmdletBinding()]` on public functions
- Run: `Invoke-ScriptAnalyzer -Path . -Recurse`

### Rust
- Follow standard `rustfmt` formatting
- Pass `cargo clippy --all-targets -- -D warnings`
- Use `cargo test` before submitting

### C#
- Follow .NET conventions
- Target .NET 8

## Testing

```powershell
# Run all tests
pwsh Tests\Invoke-AllTests.ps1 -Suite All

# Run specific suite
pwsh Tests\Invoke-AllTests.ps1 -Suite Unit
pwsh Tests\Invoke-AllTests.ps1 -Suite Integration

# Rust tests
cargo test --manifest-path Native\pcai_core\Cargo.toml

# Lint check
.\Build.ps1 -Component lint
```

All PRs must pass existing tests. Add tests for new functionality.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Run the full test suite and lint checks
4. Submit a PR with a clear description of changes and motivation

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(hardware): add NVMe temperature monitoring
fix(acceleration): handle missing fd binary gracefully
refactor(common): simplify path resolution logic
docs: update installation instructions
test(usb): add USB hub enumeration tests
chore: update Rust dependencies
```

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
