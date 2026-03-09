# PC_AI Tooling Benchmark Report

Generated: 2026-03-07 03:20:46
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
- Backend coverage rows: 6

## Performance Results

| Case | Backend | Mean ms | Median ms | StdDev ms | Speedup vs PS | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| content-search | accelerated | 11.58 | 11.6 | 5.97 | 408.89 | Rust+CSharp+PS |
| content-search | native | 3283.99 | 3508.14 | 443.77 | 1.44 | Rust+CSharp+PS |
| content-search | powershell | 4734.92 | 4719.06 | 1116.89 | 1 | Rust+CSharp+PS |
| full-context | native | 5045.05 | 5297.83 | 467.49 | 1.5 | Rust+CSharp+PS |
| full-context | powershell | 7567.51 | 7815.68 | 373.98 | 1 | Rust+CSharp+PS |

