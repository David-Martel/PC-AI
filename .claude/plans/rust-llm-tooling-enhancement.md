# Rust Debugging & LLM-Friendly Tooling Enhancement

**Created**: 2026-01-28
**Status**: Planning
**Priority**: High

## Objective

Extend CargoTools module with:
1. Additional Rust debugging and optimization tools
2. LLM-friendly interfaces with structured output
3. Context-rich diagnostics for AI-assisted development

## Proposed Tools

### Debugging Tools

| Tool | Command | Purpose | LLM Value |
|------|---------|---------|-----------|
| `cargo-expand` | `Invoke-CargoExpand` | Macro expansion | Understand generated code |
| `cargo-tree` | `Invoke-CargoDeps` | Dependency visualization | Analyze dep conflicts |
| `cargo-audit` | `Invoke-CargoAudit` | Security vulnerabilities | Automated CVE detection |
| `cargo-bloat` | `Invoke-CargoBloat` | Binary size analysis | Optimize binary size |
| `cargo-udeps` | `Invoke-CargoUnused` | Unused dependencies | Clean up Cargo.toml |
| `cargo-flamegraph` | `Invoke-CargoProfile` | CPU profiling | Identify bottlenecks |
| `cargo-llvm-lines` | `Invoke-CargoLlvmLines` | Monomorphization analysis | Reduce compile times |

### Optimization Tools

| Tool | Command | Purpose | LLM Value |
|------|---------|---------|-----------|
| `cargo-cache` | `Get-CargoCache` | Cache management | Disk usage optimization |
| `cargo-chef` | `Build-CargoChef` | Docker layer caching | Faster CI builds |
| `cargo-nextest` | `Invoke-CargoTest` | Fast test runner | Better test output |
| `clippy --fix` | `Repair-CargoClippy` | Auto-fix lints | Automated code quality |

## LLM-Friendly Output Format

### Structured Diagnostics Schema

```json
{
  "tool": "cargo-audit",
  "version": "0.18.0",
  "timestamp": "2026-01-28T18:00:00Z",
  "status": "issues_found",
  "summary": {
    "total_issues": 3,
    "critical": 1,
    "high": 2,
    "medium": 0,
    "low": 0
  },
  "issues": [
    {
      "id": "RUSTSEC-2024-0001",
      "severity": "critical",
      "package": "hyper",
      "affected_version": "0.14.0",
      "fixed_version": "0.14.28",
      "description": "HTTP request smuggling vulnerability",
      "recommendation": "Upgrade hyper to 0.14.28 or later",
      "cve": "CVE-2024-12345",
      "url": "https://rustsec.org/advisories/RUSTSEC-2024-0001"
    }
  ],
  "context": {
    "workspace_root": "C:\\Users\\david\\PC_AI\\Native\\pcai_core",
    "manifest_path": "Cargo.toml",
    "lockfile_hash": "abc123"
  },
  "suggested_actions": [
    {
      "priority": 1,
      "action": "Update Cargo.toml dependency",
      "command": "cargo update -p hyper@0.14.0 --precise 0.14.28",
      "risk": "low",
      "automated": true
    }
  ]
}
```

### Common Output Contract

All CargoTools functions will support:

```powershell
# Human-readable (default)
Invoke-CargoAudit

# LLM-friendly JSON
Invoke-CargoAudit -OutputFormat Json

# Structured object
$results = Invoke-CargoAudit -OutputFormat Object

# Piped to LLM analysis
Invoke-CargoAudit -OutputFormat Json | ConvertFrom-Json | Invoke-LLMAnalysis
```

## Implementation Phases

### Phase 1: Core LLM Interface (Priority)

1. Add `-OutputFormat` parameter to all public functions
2. Create `Format-CargoOutput` helper for consistent formatting
3. Add structured error schema
4. Create `ConvertTo-LlmContext` for rich context extraction

### Phase 2: Debugging Tools

1. `Invoke-CargoExpand` - Macro expansion with context
2. `Invoke-CargoAudit` - Security scanning with recommendations
3. `Invoke-CargoBloat` - Binary analysis with optimization hints

### Phase 3: Optimization Tools

1. `Get-CargoCache` - Cache analysis and cleanup
2. `Invoke-CargoUnused` - Dependency cleanup
3. `Invoke-CargoProfile` - Performance profiling

### Phase 4: Integration

1. MCP server for cargo tools (optional)
2. VS Code extension integration
3. CI/CD pipeline helpers

## LLM Context Extraction

### Automatic Context for AI Assistants

```powershell
function Get-RustProjectContext {
    <#
    .SYNOPSIS
    Extracts comprehensive project context for LLM analysis.
    #>
    [CmdletBinding()]
    param(
        [string]$Path = '.'
    )

    return [PSCustomObject]@{
        # Project structure
        CargoToml = Get-Content (Join-Path $Path 'Cargo.toml') -Raw
        Dependencies = cargo tree --depth 1 2>$null
        Features = cargo tree --features 2>$null

        # Build state
        LastBuildErrors = Get-CargoLastBuildErrors
        ClippyWarnings = Invoke-CargoClippy -OutputFormat Object

        # Performance metrics
        CompileTime = Measure-CargoBuild -Timing
        BinarySize = Get-CargoBinarySize

        # Security
        Vulnerabilities = Invoke-CargoAudit -OutputFormat Object

        # Suggested improvements
        Recommendations = Get-CargoRecommendations
    }
}
```

### Error Context for Debugging

```powershell
function Format-CargoError {
    <#
    .SYNOPSIS
    Formats cargo errors with rich context for LLM debugging.
    #>
    param(
        [string]$ErrorOutput,
        [string]$SourceFile
    )

    return [PSCustomObject]@{
        ErrorType = 'compilation'
        ErrorCode = 'E0382'  # Extracted from output
        Message = 'borrow of moved value'
        Location = @{
            File = $SourceFile
            Line = 42
            Column = 15
            Context = Get-SourceContext -File $SourceFile -Line 42 -Before 3 -After 3
        }
        Explanation = Get-RustcExplain 'E0382'
        SuggestedFixes = @(
            @{
                Description = 'Clone the value before moving'
                Code = '$value.clone()'
                Location = 'line 41'
            },
            @{
                Description = 'Use a reference instead'
                Code = '&$value'
                Location = 'line 42'
            }
        )
        RelatedDocs = @(
            'https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html',
            'https://doc.rust-lang.org/error_codes/E0382.html'
        )
    }
}
```

## Module Updates Required

### CargoTools.psd1

```powershell
FunctionsToExport = @(
    # Existing
    'Invoke-CargoWrapper',
    'Invoke-RustAnalyzerWrapper',
    # New debugging tools
    'Invoke-CargoExpand',
    'Invoke-CargoAudit',
    'Invoke-CargoBloat',
    'Invoke-CargoUnused',
    'Invoke-CargoProfile',
    # New optimization tools
    'Get-CargoCache',
    'Clear-CargoCache',
    'Invoke-CargoTest',  # nextest wrapper
    'Repair-CargoClippy',
    # LLM helpers
    'Get-RustProjectContext',
    'Format-CargoOutput',
    'Format-CargoError',
    'ConvertTo-LlmContext'
)
```

## Success Metrics

- [ ] All tools support `-OutputFormat Json|Object|Text`
- [ ] JSON schema documented for each tool
- [ ] Error messages include actionable suggestions
- [ ] Context extraction includes relevant source code
- [ ] Integration with Test-RustAnalyzerHealth
- [ ] Pester tests for all new functions

## Next Steps

1. Implement Phase 1 core LLM interface
2. Add `-OutputFormat` to existing functions
3. Create schema documentation
4. Implement highest-value debugging tools first
