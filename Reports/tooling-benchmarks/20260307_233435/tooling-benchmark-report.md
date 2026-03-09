# PC_AI Tooling Benchmark Report

Generated: 2026-03-07 23:35:29
Suite: default
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
- Backend coverage rows: 0

## Performance Results

| Case | Backend | Mean ms | Median ms | StdDev ms | WS delta KB | Private delta KB | Managed delta KB | Managed alloc KB | Speedup vs PS | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| acceleration-import | powershell | 1861.89 | 1701.36 | 238.68 | 1609.33 | 196 | 102.07 | 96.44 | 1 | PowerShellOnly |
| content-search | powershell | 1714.44 | 1759.67 | 174.46 | 112 | 72 | 7.62 | 16996.56 | 1 | Rust+CSharp+PS |
| content-search | native | 1722.48 | 1732.9 | 63.02 | -388.8 | -1822.4 | -3678.22 | 228.28 | 1 | Rust+CSharp+PS |
| content-search | accelerated | 1776.51 | 1741.04 | 127.94 | 77.6 | -27.2 | 1560.79 | 1554.72 | 0.97 | Rust+CSharp+PS |
| direct-core-probe | powershell | 31.03 | 28.97 | 10.29 | 1052 | 1042.4 | -606.11 | 1383.99 | 1 | PowerShellOnly |
| file-search | native | 21.77 | 19.07 | 6.08 | 468 | 450.4 | 203.95 | 203.52 | 67.93 | Rust+CSharp+PS |
| file-search | accelerated | 120.07 | 123.26 | 14.43 | 44.8 | 10.4 | 1871.96 | 4378.23 | 12.32 | Rust+CSharp+PS |
| file-search | powershell | 1478.91 | 1509.44 | 108.92 | 1255.2 | 400.8 | 991.24 | 12097.02 | 1 | Rust+CSharp+PS |
| runtime-config | powershell | 71.66 | 17.11 | 87.94 | 651.2 | 632 | -1131.99 | 849.81 | 1 | PowerShellOnly |

