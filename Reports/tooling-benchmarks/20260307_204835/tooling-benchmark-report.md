# PC_AI Tooling Benchmark Report

Generated: 2026-03-07 20:49:21
Suite: quick
RepoRoot: `C:\codedev\PC_AI`

## Backend Coverage

| Operation | Coverage | Preferred | Gap |
| --- | --- | --- | --- |

## Tool Schema Coverage

- Tool schema count: 28
- Tool mapping count: 0
- Backend coverage rows: 0

## Performance Results

| Case | Backend | Mean ms | Median ms | StdDev ms | Speedup vs PS | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| acceleration-import | powershell | 1754.22 | 1593.53 | 229.09 | 1 | PowerShellOnly |
| acceleration-probe | powershell | 22.25 | 18.47 | 8.77 | 1 | PowerShellOnly |
| command-map | powershell | 51.38 | 49.68 | 6.46 | 1 | PowerShellOnly |
| directory-manifest | powershell | 282.41 | 269.54 | 30.43 | 1 | PowerShellOnly |
| file-search | accelerated | 352.36 | 350.6 | 22.22 | 5.82 | PowerShellOnly |
| file-search | powershell | 2052.42 | 1950.62 | 323.28 | 1 | PowerShellOnly |
| runtime-config | powershell | 18.52 | 19.16 | 2.56 | 1 | PowerShellOnly |
| token-estimate | powershell | 67 | 71.83 | 14.68 | 1 | PowerShellOnly |

