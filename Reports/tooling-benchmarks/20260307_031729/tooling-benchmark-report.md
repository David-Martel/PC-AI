# PC_AI Tooling Benchmark Report

Generated: 2026-03-07 03:18:09
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
| command-map | powershell | 76.83 | 72.01 | 14.26 | 1 | PowerShellOnly |
| directory-manifest | native | 28.55 | 28.39 | 7.04 | 40.69 | Rust+CSharp+PS |
| directory-manifest | powershell | 1161.75 | 1254.69 | 203.89 | 1 | Rust+CSharp+PS |
| file-search | accelerated | 8.7 | 4.63 | 5.64 | 394.92 | Rust+CSharp+PS |
| file-search | native | 104.07 | 94.01 | 26.95 | 33.01 | Rust+CSharp+PS |
| file-search | powershell | 3435.82 | 3451.68 | 698.26 | 1 | Rust+CSharp+PS |
| runtime-config | powershell | 44.92 | 39.12 | 16.31 | 1 | PowerShellOnly |
| token-estimate | native | 35.81 | 37.34 | 7.45 | 2.9 | Rust+CSharp+PS |
| token-estimate | powershell | 103.82 | 80.19 | 49.35 | 1 | Rust+CSharp+PS |

