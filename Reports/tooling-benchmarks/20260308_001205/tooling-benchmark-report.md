# PC_AI Tooling Benchmark Report

Generated: 2026-03-08 00:12:22
Suite: default
RepoRoot: `C:\codedev\PC_AI`

## Backend Coverage

| Operation | Coverage | Preferred | Gap |
| --- | --- | --- | --- |

## Tool Schema Coverage

- Tool schema count: 28
- Tool mapping count: 0
- Backend coverage rows: 0

## Performance Results

| Case | Backend | Mean ms | Median ms | StdDev ms | WS delta KB | Private delta KB | Managed delta KB | Managed alloc KB | Speedup vs PS | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| content-search | native | 12.59 | 11.11 | 3.5 | 88.8 | 28.8 | 150.77 | 150.39 | 181.25 | PowerShellOnly |
| content-search | accelerated | 52.5 | 48.11 | 11.34 | 438.4 | 331.2 | -403.38 | 1672.04 | 43.47 | PowerShellOnly |
| content-search | powershell | 2282 | 2305.69 | 130.83 | 2463.2 | 765.6 | -763.81 | 23399.28 | 1 | PowerShellOnly |

