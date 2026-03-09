# PC_AI Tooling Benchmark Report

Generated: 2026-03-07 20:18:44
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
| command-map | powershell | 56.11 | 48.62 | 21.55 | 1 | PowerShellOnly |
| directory-manifest | native | 89.68 | 82.87 | 39.9 | 5.24 | Rust+CSharp+PS |
| directory-manifest | powershell | 470.35 | 453.76 | 70.22 | 1 | Rust+CSharp+PS |
| file-search | accelerated | 15.92 | 14.05 | 6.42 | 124.18 | Rust+CSharp+PS |
| file-search | native | 155.65 | 146.91 | 31.13 | 12.7 | Rust+CSharp+PS |
| file-search | powershell | 1976.95 | 2013.03 | 200.43 | 1 | Rust+CSharp+PS |
| runtime-config | powershell | 18.36 | 16.56 | 7.16 | 1 | PowerShellOnly |
| token-estimate | native | 22.12 | 21.81 | 1.89 | 5.09 | Rust+CSharp+PS |
| token-estimate | powershell | 112.61 | 116.56 | 40.98 | 1 | Rust+CSharp+PS |

