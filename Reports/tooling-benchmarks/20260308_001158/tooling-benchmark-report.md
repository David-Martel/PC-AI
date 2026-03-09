# PC_AI Tooling Benchmark Report

Generated: 2026-03-08 00:12:28
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
| content-search | native | 13.33 | 12.61 | 3.09 | 168 | 159.2 | 151.35 | 154.16 | 143.56 | Rust+CSharp+PS |
| content-search | accelerated | 26.91 | 28.39 | 4.32 | 39.2 | 4.8 | 1552.56 | 1548.1 | 71.11 | Rust+CSharp+PS |
| content-search | powershell | 1913.69 | 1868.46 | 210.37 | 1552 | 512.8 | -749.09 | 23439.82 | 1 | Rust+CSharp+PS |
| file-search | native | 31.48 | 27.87 | 11.02 | 337.6 | 291.2 | 223.56 | 219.88 | 68.35 | Rust+CSharp+PS |
| file-search | accelerated | 194.81 | 199.06 | 27.39 | 1127.2 | 1004.8 | -63.33 | 4377.66 | 11.04 | Rust+CSharp+PS |
| file-search | powershell | 2151.65 | 2171.36 | 65.97 | 2347.2 | 1395.2 | 976.83 | 12112.79 | 1 | Rust+CSharp+PS |

