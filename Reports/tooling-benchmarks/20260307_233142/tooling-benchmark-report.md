# PC_AI Tooling Benchmark Report

Generated: 2026-03-07 23:32:55
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
| acceleration-import | powershell | 3789.01 | 3966.07 | 311.2 | 970.67 | 18.67 | 3057.91 | 3044.07 | 1 | PowerShellOnly |
| content-search | accelerated | 4.37 | 3.97 | 0.98 | 20 | 0.8 | 260.21 | 257.85 | 613.21 | Rust+CSharp+PS |
| content-search | native | 2390.42 | 2496.54 | 252.98 | 345.6 | 303.2 | 155.77 | 155.38 | 1.12 | Rust+CSharp+PS |
| content-search | powershell | 2679.74 | 2612.31 | 121.99 | -393.6 | -406.4 | 2500.7 | 16988.38 | 1 | Rust+CSharp+PS |
| direct-core-probe | powershell | 41.68 | 42.9 | 10.14 | 24.8 | 1.6 | -664.03 | 1381.8 | 1 | PowerShellOnly |
| file-search | accelerated | 13.52 | 9.16 | 9.48 | 349.6 | 319.2 | -1759.85 | 423.27 | 191.18 | Rust+CSharp+PS |
| file-search | native | 54.1 | 57.16 | 7.68 | 553.6 | 660.8 | 221.39 | 220.93 | 47.78 | Rust+CSharp+PS |
| file-search | powershell | 2584.79 | 2687.93 | 170.95 | 1687.2 | 62.4 | 1052.24 | 12133.65 | 1 | Rust+CSharp+PS |
| runtime-config | powershell | 27.4 | 20.49 | 17.06 | 529.6 | 485.6 | 856.88 | 854.4 | 1 | PowerShellOnly |

