# PC_AI Tooling Benchmark Report

Generated: 2026-03-11 00:18:59
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
| content-search | native | 4172.07 | 4123.42 | 172.94 | 481.6 | 173.6 | 215.58 | 214.27 | 1.08 | PowerShellOnly |
| content-search | powershell | 4497.59 | 4461.74 | 424.42 | 1037.6 | 1169.6 | -202.84 | 33519.52 | 1 | PowerShellOnly |
| content-search | accelerated | 4842.11 | 4744.04 | 453.68 | -562.4 | -615.2 | -1568.77 | 1699.33 | 0.93 | PowerShellOnly |

