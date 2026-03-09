# PC_AI Tooling Benchmark Report

Generated: 2026-03-07 22:32:56
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
| acceleration-import | powershell | 8859.11 | 9475.15 | 3732.93 | -177.33 | -136 | 105.03 | 102.06 | 1 | PowerShellOnly |
| acceleration-probe | powershell | 43.34 | 25.63 | 29.88 | 900.8 | 961.6 | 1297 | 1293.45 | 1 | PowerShellOnly |
| direct-core-probe | powershell | 63.03 | 64.32 | 25.44 | 766.4 | 602.4 | 1282.58 | 1278.83 | 1 | PowerShellOnly |
| runtime-config | powershell | 20.6 | 11.05 | 18.19 | 327.2 | 248 | 841.88 | 839.45 | 1 | PowerShellOnly |

