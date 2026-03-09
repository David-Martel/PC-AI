# PC_AI Tooling Benchmark Report

Generated: 2026-03-07 20:25:47
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
| command-map | powershell | 63.19 | 60.01 | 16.53 | 1 | PowerShellOnly |
| directory-manifest | native | 111.74 | 106.55 | 31.72 | 5.38 | Rust+CSharp+PS |
| directory-manifest | powershell | 601.62 | 584.16 | 141.08 | 1 | Rust+CSharp+PS |
| file-search | accelerated | 21.19 | 11.99 | 19.69 | 199.06 | Rust+CSharp+PS |
| file-search | native | 94.5 | 99.38 | 17.27 | 44.64 | Rust+CSharp+PS |
| file-search | powershell | 4218.17 | 3660.83 | 852.67 | 1 | Rust+CSharp+PS |
| runtime-config | powershell | 23.4 | 17.22 | 8.64 | 1 | PowerShellOnly |
| token-estimate | native | 64.5 | 67.27 | 8.41 | 3.12 | Rust+CSharp+PS |
| token-estimate | powershell | 201.32 | 191.29 | 68.89 | 1 | Rust+CSharp+PS |

