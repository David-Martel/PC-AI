# PC_AI Tooling Benchmark Report

Generated: 2026-03-07 20:28:39
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
| command-map | powershell | 44.85 | 41.68 | 10.12 | 1 | PowerShellOnly |
| directory-manifest | native | 45.99 | 40.75 | 12.13 | 16.88 | Rust+CSharp+PS |
| directory-manifest | powershell | 776.15 | 716.99 | 205.97 | 1 | Rust+CSharp+PS |
| file-search | accelerated | 16.96 | 11.28 | 11.68 | 246.7 | Rust+CSharp+PS |
| file-search | native | 185.78 | 215.29 | 52.36 | 22.52 | Rust+CSharp+PS |
| file-search | powershell | 4183.98 | 3854.19 | 735.27 | 1 | Rust+CSharp+PS |
| runtime-config | powershell | 23.65 | 15.86 | 16.63 | 1 | PowerShellOnly |
| token-estimate | native | 28.82 | 24.18 | 9.81 | 4.35 | Rust+CSharp+PS |
| token-estimate | powershell | 125.37 | 80.86 | 84.95 | 1 | Rust+CSharp+PS |

