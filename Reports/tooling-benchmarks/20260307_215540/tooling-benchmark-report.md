# PC_AI Tooling Benchmark Report

Generated: 2026-03-07 21:56:54
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
| acceleration-import | powershell | 9692.61 | 9692.68 | 1681.96 | -3802.67 | -3649.33 | 111.3 | 105.74 | 1 | PowerShellOnly |
| acceleration-probe | powershell | 39.5 | 34.85 | 11.73 | 1137.6 | 1052 | 1305.75 | 1305.3 | 1 | PowerShellOnly |
| direct-core-probe | powershell | 44.23 | 41.09 | 11.79 | 67.2 | 0.8 | 1295.21 | 1291.44 | 1 | PowerShellOnly |

