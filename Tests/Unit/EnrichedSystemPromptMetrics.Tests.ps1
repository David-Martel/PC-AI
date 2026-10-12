#Requires -Version 7.0

param(
    [string]$HelperSourcePath = (Join-Path $PSScriptRoot '../../Modules/PC-AI.LLM/Private/LLM-Helpers.ps1')
)

# These controls execute the complete maintained renderer over owned UTF-8 files.
# Only the dashboard/provider boundaries are replaced; no native/module bootstrap.
BeforeAll {
    $sourcePath = (Resolve-Path -LiteralPath $HelperSourcePath).ProviderPath
    $tokens = $null
    $errors = $null
    $sourceAst = [Management.Automation.Language.Parser]::ParseFile($sourcePath, [ref]$tokens, [ref]$errors)
    if ($errors.Count) { throw 'Canonical prompt helper source must parse.' }
    $definitions = @($sourceAst.FindAll({
        param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq 'Get-EnrichedSystemPrompt'
    }, $true))
    if ($definitions.Count -ne 1) { throw 'One complete canonical prompt renderer is required.' }
    Set-Item -LiteralPath function:Get-EnrichedSystemPrompt -Value ($definitions[0].Body.GetScriptBlock())

    function Get-PcaiNativeDashboardSnapshot { throw 'Unmocked dashboard boundary.' }
    function Get-PcaiResourceMetrics { throw 'Unexpected resource-provider call.' }
    function Get-CimInstance { param([string]$ClassName) throw 'Unexpected hardware-provider call.' }
    function Invoke-RestMethod { throw 'Unexpected backend request.' }
    function Invoke-WebRequest { throw 'Unexpected backend request.' }
    function Start-Process { throw 'Unexpected backend launch.' }
    $script:caseNumber = 0
}

Describe 'Canonical prompt metrics with owned documents' -Tag 'Unit', 'LLM', 'Fast', 'Portable' {
    BeforeEach {
        $script:caseNumber++
        $script:documentRoot = Join-Path $TestDrive "metrics-case-$script:caseNumber"
        [void][IO.Directory]::CreateDirectory($script:documentRoot)
        $script:promptText = "資料 café 🧭`n[SYSTEM_RESOURCE_STATUS]`n末尾"
        $script:documentPath = Join-Path $script:documentRoot 'CHAT.md'
        [IO.File]::WriteAllText($script:documentPath, $script:promptText, [Text.UTF8Encoding]::new($false))
        $script:documentHash = (Get-FileHash -LiteralPath $script:documentPath).Hash
        $script:snapshot = $null
        Mock Get-PcaiNativeDashboardSnapshot { $script:snapshot }
        Mock Get-PcaiResourceMetrics { throw 'Unexpected resource-provider call.' }
        Mock Get-CimInstance { throw 'Unexpected hardware-provider call.' }
        Mock Invoke-RestMethod { throw 'Unexpected backend request.' }
        Mock Invoke-WebRequest { throw 'Unexpected backend request.' }
        Mock Start-Process { throw 'Unexpected backend launch.' }
    }

    AfterEach {
        (Get-FileHash -LiteralPath $script:documentPath).Hash | Should -BeExactly $script:documentHash
        [IO.File]::ReadAllText($script:documentPath) | Should -BeExactly $script:promptText
        Should -Invoke Get-PcaiNativeDashboardSnapshot -Exactly -Times 1
        Should -Invoke Get-PcaiResourceMetrics -Exactly -Times 0
        Should -Invoke Get-CimInstance -Exactly -Times 0
        Should -Invoke Invoke-RestMethod -Exactly -Times 0
        Should -Invoke Invoke-WebRequest -Exactly -Times 0
        Should -Invoke Start-Process -Exactly -Times 0
    }

    # Protects: modern CPU values, including zero, outrank legacy aliases.
    # Detects: falsey-zero fallback, ignored avg_load or incorrect byte units.
    # Needs: complete renderer, literal dashboard JSON and real owned document.
    # Breadcrumb: Get-EnrichedSystemPrompt native summary and alias precedence.
    It 'reports <Scenario> without discarding authoritative zero or legacy identity' -ForEach @(
        @{ Scenario = 'nonzero modern avg_load'; Snapshot = '{"Metrics":{"avg_load":12.376,"cpu_usage_perc":90,"CpuUsage":99,"memory_usage_bytes":3221225472,"total_memory_bytes":17179869184}}'; Cpu = 'CPU Usage: 12.38%'; Memory = 'Memory: 3 GB / 16 GB' },
        @{ Scenario = 'modern CPU and used-memory zero'; Snapshot = '{"Metrics":{"avg_load":0,"cpu_usage_perc":90,"CpuUsage":99,"memory_usage_bytes":0,"MemoryUsage":3221225472,"total_memory_bytes":17179869184}}'; Cpu = 'CPU Usage: 0%'; Memory = 'Memory: 0 GB / 16 GB' },
        @{ Scenario = 'legacy snake-case CPU zero'; Snapshot = '{"Metrics":{"cpu_usage_perc":0,"CpuUsage":99,"memory_usage_bytes":1073741824,"total_memory_bytes":8589934592}}'; Cpu = 'CPU Usage: 0%'; Memory = 'Memory: 1 GB / 8 GB' },
        @{ Scenario = 'legacy snake-case CPU precedence'; Snapshot = '{"Metrics":{"cpu_usage_perc":6.75,"CpuUsage":99,"memory_usage_bytes":1073741824,"total_memory_bytes":8589934592}}'; Cpu = 'CPU Usage: 6.75%'; Memory = 'Memory: 1 GB / 8 GB' },
        @{ Scenario = 'PascalCase-only legacy metrics'; Snapshot = '{"Metrics":{"CpuUsage":25,"MemoryUsage":1073741824,"TotalMemory":8589934592}}'; Cpu = 'CPU Usage: 25%'; Memory = 'Memory: 1 GB / 8 GB' },
        @{ Scenario = 'null modern CPU warmup'; Snapshot = '{"Metrics":{"avg_load":null,"cpu_usage_perc":90,"memory_usage_bytes":1073741824,"total_memory_bytes":8589934592}}'; Cpu = 'CPU Usage: n/a'; Memory = 'Memory: 1 GB / 8 GB' },
        @{ Scenario = 'all-null warmup metrics'; Snapshot = '{"Metrics":{"avg_load":null,"memory_usage_bytes":null,"total_memory_bytes":null}}'; Cpu = 'CPU Usage: n/a'; Memory = 'Memory: n/a' }
    ) {
        $script:snapshot = $Snapshot
        Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:documentRoot |
            Should -BeExactly "資料 café 🧭`n### LIVE SYSTEM TELEMETRY (NATIVE)`n$Cpu`n$Memory`n$Snapshot`n末尾"
    }

    # Protects: invalid/absent JSON remains available for diagnosis without guessed data.
    # Detects: parse exceptions escaping, zero invention or silent raw-snapshot loss.
    # Needs: literal malformed and nullable snapshots; no filesystem/provider mock.
    # Breadcrumb: Get-EnrichedSystemPrompt native parsing catch and n/a summary.
    It 'retains <Scenario> exactly while presenting unavailable summaries' -ForEach @(
        @{ Scenario = 'malformed JSON'; Snapshot = '{broken-json' },
        @{ Scenario = 'missing metrics'; Snapshot = '{"Note":"owned opaque snapshot"}' },
        @{ Scenario = 'explicit null metrics'; Snapshot = '{"Metrics":null,"Note":"owned warmup"}' },
        @{ Scenario = 'empty metrics object'; Snapshot = '{"Metrics":{}}' }
    ) {
        $script:snapshot = $Snapshot
        Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:documentRoot |
            Should -BeExactly "資料 café 🧭`n### LIVE SYSTEM TELEMETRY (NATIVE)`nCPU Usage: n/a`nMemory: n/a`n$Snapshot`n末尾"
    }

    # Protects: impossible or untyped memory measurements are unavailable, not numeric.
    # Detects: negative/over-total usage and coercion of nonnumeric wire representations.
    # Needs: complete real renderer; exact independent JSON/value expectations.
    # Breadcrumb: memory_usage_bytes / total_memory_bytes represent nonnegative bytes.
    It 'does not describe <Scenario> as valid physical memory' -ForEach @(
        @{ Scenario = 'zero total'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":1073741824,"total_memory_bytes":0}}' },
        @{ Scenario = 'negative total'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":1073741824,"total_memory_bytes":-8589934592}}' },
        @{ Scenario = 'negative used bytes'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":-1073741824,"total_memory_bytes":8589934592}}' },
        @{ Scenario = 'used bytes greater than total'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":9663676416,"total_memory_bytes":8589934592}}' },
        @{ Scenario = 'quoted numeric total'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":1073741824,"total_memory_bytes":"8589934592"}}' },
        @{ Scenario = 'quoted numeric used bytes'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":"1073741824","total_memory_bytes":8589934592}}' }
    ) {
        $script:snapshot = $Snapshot
        Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:documentRoot |
            Should -BeExactly "資料 café 🧭`n### LIVE SYSTEM TELEMETRY (NATIVE)`nCPU Usage: 18.5%`nMemory: n/a`n$Snapshot`n末尾"
    }

    # Protects: opaque snapshot bytes and Unicode context survive append-mode rendering.
    # Detects: serializer rewrites, missing data or prompt-file mutation.
    # Needs: real UTF-8 document with no resource marker and independent summary values.
    # Breadcrumb: native telemetry append branch retains the complete supplied snapshot.
    It 'appends one truthful block to a marker-free owned Unicode document' {
        $script:promptText = '診断 é 🧭 literal [OTHER_STATUS] terminal'
        [IO.File]::WriteAllText($script:documentPath, $script:promptText, [Text.UTF8Encoding]::new($false))
        $script:documentHash = (Get-FileHash -LiteralPath $script:documentPath).Hash
        $script:snapshot = "{`"Metrics`":{`"avg_load`":0,`"memory_usage_bytes`":0,`"total_memory_bytes`":8589934592},`"Note`":`"資料 é 🧭`"}"
        Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:documentRoot |
            Should -BeExactly "$script:promptText`n`n[SYSTEM_RESOURCE_STATUS]`n### LIVE SYSTEM TELEMETRY (NATIVE)`nCPU Usage: 0%`nMemory: 0 GB / 8 GB`n$script:snapshot"
    }

    # Protects: bad byte representations do not erase a valid CPU summary.
    # Detects: boolean/string/array coercion, nonfinite values and unsigned overflow.
    # Needs: literal JSON through the sole dashboard mock and original document bytes.
    # Breadcrumb: byte fields carry integral unsigned numeric measurements.
    It 'preserves CPU and raw JSON when memory has <Scenario>' -ForEach @(
        @{ Scenario = 'boolean used value'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":false,"total_memory_bytes":8589934592}}' },
        @{ Scenario = 'boolean total value'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":0,"total_memory_bytes":true}}' },
        @{ Scenario = 'array used value'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":[1073741824],"total_memory_bytes":8589934592}}' },
        @{ Scenario = 'object total value'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":1073741824,"total_memory_bytes":{"Bytes":8589934592}}}' },
        @{ Scenario = 'null used value'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":null,"total_memory_bytes":8589934592}}' },
        @{ Scenario = 'null total value'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":1073741824,"total_memory_bytes":null}}' },
        @{ Scenario = 'NaN used value'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":NaN,"total_memory_bytes":8589934592}}' },
        @{ Scenario = 'infinite total value'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":1073741824,"total_memory_bytes":Infinity}}' },
        @{ Scenario = 'fractional byte value'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":1.5,"total_memory_bytes":8589934592}}' },
        @{ Scenario = 'UInt64 overflow total'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":1073741824,"total_memory_bytes":18446744073709551616}}' },
        @{ Scenario = 'nonnumeric text'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":"unavailable","total_memory_bytes":8589934592}}' }
    ) {
        $script:snapshot = $Snapshot
        Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:documentRoot |
            Should -BeExactly "資料 café 🧭`n### LIVE SYSTEM TELEMETRY (NATIVE)`nCPU Usage: 18.5%`nMemory: n/a`n$Snapshot`n末尾"
    }

    # Protects: large unsigned counters and legacy rounded formatting remain usable.
    # Detects: narrowing/overflow refusal or new trailing-zero text from validation.
    # Needs: literal independent maximum and fractional-GiB byte values.
    # Breadcrumb: validation is exact in bytes; display retains two-place rounding.
    It 'retains <Scenario> as valid measurements with unchanged display formatting' -ForEach @(
        @{ Scenario = 'UInt64 maximum total'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":1073741824,"total_memory_bytes":18446744073709551615}}'; Memory = 'Memory: 1 GB / 17179869184 GB' },
        @{ Scenario = 'fractional GiB rounded without trailing zero'; Snapshot = '{"Metrics":{"avg_load":18.5,"memory_usage_bytes":1181116006,"total_memory_bytes":8589934592}}'; Memory = 'Memory: 1.1 GB / 8 GB' }
    ) {
        $script:snapshot = $Snapshot
        Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:documentRoot |
            Should -BeExactly "資料 café 🧭`n### LIVE SYSTEM TELEMETRY (NATIVE)`nCPU Usage: 18.5%`n$Memory`n$Snapshot`n末尾"
    }
}
