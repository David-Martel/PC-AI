# PC_AI Tooling Benchmark Report

Generated: 2026-03-07 03:16:21
Suite: quick
RepoRoot: `C:\codedev\PC_AI`

## Backend Coverage

| Operation | Coverage | Preferred | Gap |
| --- | --- | --- | --- |
| TokenEstimate | Rust+CSharp+PS | Rust+C# |  |
| DirectoryManifest | Rust+CSharp+PS | Rust+C# |  |
| FileSearch | Rust+CSharp+PS | Rust+C# |  |
| ContentSearch | Rust+CSharp+PS | Rust+C# |  |
| FullContext | Rust+CSharp+PS | Rust+C# |  |
| DiskUsage | Rust+CSharp+PS | Rust+C# |  |

## Tool Schema Coverage

- Tool schema count: 28
- Tool mapping count: 0
- Backend coverage rows: 6

## Performance Results

| Case | Backend | Mean ms | Median ms | StdDev ms | Speedup vs PS | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| command-map | powershell | 45.5 | 42.15 | 8.4 | 1 | PowerShellOnly |
| directory-manifest | native | 19.05 | 19.24 | 3.41 | 25.76 | Rust+CSharp+PS |
| directory-manifest | powershell | 490.73 | 491.58 | 50.49 | 1 | Rust+CSharp+PS |
| file-search | native | 94.28 | 97.08 | 12.66 | 27.84 | Rust+CSharp+PS |
| file-search | powershell | 2625.22 | 2624.04 | 367.45 | 1 | Rust+CSharp+PS |
| runtime-config | powershell | 19.82 | 17.29 | 6.97 | 1 | PowerShellOnly |
| token-estimate | native | 32.06 | 32.87 | 2.81 | 2.35 | Rust+CSharp+PS |
| token-estimate | powershell | 75.48 | 64.31 | 32.37 | 1 | Rust+CSharp+PS |

