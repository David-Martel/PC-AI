# PC_AI Tooling Benchmark Report

Generated: 2026-03-07 20:28:42
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
| command-map | powershell | 53.06 | 53.57 | 3.61 | 1 | PowerShellOnly |
| directory-manifest | native | 144.53 | 134.99 | 68.37 | 5.06 | Rust+CSharp+PS |
| directory-manifest | powershell | 730.75 | 772.05 | 143.38 | 1 | Rust+CSharp+PS |
| file-search | accelerated | 34.66 | 20.45 | 29.65 | 99.69 | Rust+CSharp+PS |
| file-search | native | 172.99 | 157.07 | 41.79 | 19.97 | Rust+CSharp+PS |
| file-search | powershell | 3455.26 | 3660.48 | 481.33 | 1 | Rust+CSharp+PS |
| runtime-config | powershell | 21.92 | 20.16 | 5.2 | 1 | PowerShellOnly |
| token-estimate | native | 19.6 | 19.89 | 1.43 | 5.03 | Rust+CSharp+PS |
| token-estimate | powershell | 98.53 | 102.66 | 25.39 | 1 | Rust+CSharp+PS |

