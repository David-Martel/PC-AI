#Requires -Version 7.0

# Complete canonical helper; ParseFile retains coverage coordinates. No module/native bootstrap.
BeforeAll {
    $sourcePath = (Resolve-Path (Join-Path $PSScriptRoot '../../Modules/PC-AI.LLM/Private/LLM-Helpers.ps1')).ProviderPath
    $sourceTokens = $null
    $sourceErrors = $null
    $sourceAst = [Management.Automation.Language.Parser]::ParseFile($sourcePath, [ref]$sourceTokens, [ref]$sourceErrors)
    if ($sourceErrors.Count) { throw 'Canonical LLM helper source must parse.' }
    $definitions = @($sourceAst.FindAll({
        param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq 'Get-EnrichedSystemPrompt'
    }, $true))
    if ($definitions.Count -ne 1) { throw 'Exactly one complete canonical enrichment helper required.' }
    Set-Item -LiteralPath function:Get-EnrichedSystemPrompt -Value ($definitions[0].Body.GetScriptBlock())
    $nativeDefinitions = @($sourceAst.FindAll({
        param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq 'Get-PcaiNativeDashboardSnapshot'
    }, $true))
    if ($nativeDefinitions.Count -ne 1) { throw 'Exactly one complete native dashboard boundary required.' }
    Set-Item -LiteralPath function:Get-PcaiNativeDashboardSnapshot -Value ($nativeDefinitions[0].Body.GetScriptBlock())

    # Discovery and telemetry collaborators only. Unexpected hardware calls fail the case.
    function Get-PcaiResourceMetrics { throw 'Unmocked resource telemetry boundary.' }
    function Resolve-PcaiPath { param([string]$PathType) throw 'Unmocked project root boundary.' }
    function Get-CimInstance { param([string]$ClassName) throw 'Unmocked hardware telemetry boundary.' }
    $script:promptCaseNumber = 0
    # A canonical here-string retains its physical source newline, even in a mixed-EOL checkout.
    # Derive only that lexical separator; metric values/text remain independently specified.
    $telemetryTokens = @($sourceTokens | Where-Object {
        $_.Kind -eq 'HereStringExpandable' -and $_.Extent.StartOffset -ge $definitions[0].Extent.StartOffset -and
        $_.Extent.EndOffset -le $definitions[0].Extent.EndOffset
    })
    if ($telemetryTokens.Count -ne 1) { throw 'One canonical telemetry here-string separator required.' }
    $separator = [regex]::Match($telemetryTokens[0].Extent.Text, '^@"(\r?\n)').Groups[1].Value
    if (-not $separator) { throw 'Canonical telemetry source newline required.' }
    $script:literalTelemetry = @('CPU Usage: 12.5%', 'Memory Usage: 3 GB / 16 GB', 'GPU Usage: 7%', 'System Uptime: 01d 02h 03m') -join $separator
    function Write-PromptDocument([string]$Name, [string]$Content) {
        [IO.File]::WriteAllText((Join-Path $script:promptFixture $Name), $Content, [Text.UTF8Encoding]::new($false))
    }
}

Describe 'Canonical prompt enrichment with owned documents' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        $script:promptCaseNumber++
        $script:promptFixture = Join-Path $TestDrive "prompt-case-$script:promptCaseNumber"
        [void][IO.Directory]::CreateDirectory($script:promptFixture)
        Mock Get-PcaiResourceMetrics {
            [pscustomobject]@{ CpuUsage = 12.5; MemoryUsage = 3GB; TotalMemory = 16GB; GpuUsage = 7; Uptime = 93780 }
        }
        Mock Get-PcaiNativeDashboardSnapshot { $null }
        Mock Get-CimInstance { throw 'Unexpected hardware query.' }
        Mock Resolve-PcaiPath { $script:promptFixture }
        Mock Get-Command { [pscustomobject]@{ Name = $Name } } -ParameterFilter { $Name -in @('Get-PcaiResourceMetrics', 'Resolve-PcaiPath') }
    }

    # Protects: chat mode uses only the requested prompt and telemetry.
    # Detects: selecting DIAGNOSE or leaking its optional reasoning into chat.
    # Needs: real owned UTF8 documents; mocked native/resource/CIM boundaries.
    # Breadcrumb: LLM-Helpers.ps1 Get-EnrichedSystemPrompt prompt selection and reasoning guard.
    It 'selects CHAT without appending diagnose reasoning' {
        Write-PromptDocument CHAT.md "chat-only`n[SYSTEM_RESOURCE_STATUS]"
        Write-PromptDocument DIAGNOSE.md 'diagnose-only'
        Write-PromptDocument DIAGNOSE_LOGIC.md 'reasoning-only'
        $actual = Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:promptFixture
        $actual | Should -BeExactly "chat-only`n$script:literalTelemetry"
        Should -Invoke Get-PcaiResourceMetrics -Exactly -Times 1
        Should -Invoke Get-CimInstance -Exactly -Times 0
    }

    # Protects: diagnose mode retains complete Unicode reasoning exactly once.
    # Detects: wrong document, omitted logic, duplicate framework, or text loss.
    # Needs: owned CHAT/DIAGNOSE/logic documents; resource metrics boundary.
    # Breadcrumb: LLM-Helpers.ps1 Get-EnrichedSystemPrompt DIAGNOSE_LOGIC insertion.
    It 'selects DIAGNOSE and inserts complete optional reasoning once' {
        Write-PromptDocument CHAT.md 'chat-only'
        Write-PromptDocument DIAGNOSE.md "diagnose-only`n[SYSTEM_RESOURCE_STATUS]"
        Write-PromptDocument DIAGNOSE_LOGIC.md "第一步: preserve context`nsecond step"
        $actual = Get-EnrichedSystemPrompt -Mode diagnose -ProjectRoot $script:promptFixture
        $actual | Should -BeExactly "diagnose-only`n$script:literalTelemetry`n`n## REASONING FRAMEWORK`n`n第一步: preserve context`nsecond step"
        ([regex]::Matches($actual, '## REASONING FRAMEWORK')).Count | Should -Be 1
    }

    # Protects: optional missing reasoning does not invalidate diagnosis.
    # Detects: mandatory logic dependency or changed standalone prompt content.
    # Needs: owned DIAGNOSE with no logic file; concrete resource fixture.
    # Breadcrumb: LLM-Helpers.ps1 Get-EnrichedSystemPrompt logic Test-Path guard.
    It 'keeps diagnose usable when the optional reasoning file is absent' {
        Write-PromptDocument DIAGNOSE.md 'standalone diagnosis'
        Get-EnrichedSystemPrompt -Mode diagnose -ProjectRoot $script:promptFixture |
            Should -BeExactly "standalone diagnosis`n`n[SYSTEM_RESOURCE_STATUS]`n$script:literalTelemetry"
    }

    # Protects: each missing mode emits its exact path warning and empty prompt.
    # Detects: wrong missing filename, telemetry side effects, or nonempty fallback.
    # Needs: empty owned directory; mocked telemetry with invocation counts.
    # Breadcrumb: LLM-Helpers.ps1 Get-EnrichedSystemPrompt missing-prompt early return.
    It 'warns exactly and returns empty for missing <Mode> without telemetry' -ForEach @(
        @{ Mode = 'chat'; Leaf = 'CHAT.md' }, @{ Mode = 'diagnose'; Leaf = 'DIAGNOSE.md' }
    ) {
        $warnings = @()
        $actual = Get-EnrichedSystemPrompt -Mode $Mode -ProjectRoot $script:promptFixture -WarningVariable warnings -WarningAction SilentlyContinue
        $actual | Should -BeExactly ''
        $warnings.Count | Should -Be 1
        $warnings[0].Message | Should -BeExactly "System prompt not found at $(Join-Path $script:promptFixture $Leaf)"
        Should -Invoke Get-PcaiResourceMetrics -Exactly -Times 0
        Should -Invoke Get-CimInstance -Exactly -Times 0
    }

    # Protects: every exact resource placeholder is substituted in place.
    # Detects: first-only replacement, residual markers, or extra appended block.
    # Needs: owned two-marker prompt; independently specified telemetry text.
    # Breadcrumb: LLM-Helpers.ps1 Get-EnrichedSystemPrompt resource marker replacement.
    It 'replaces both exact placeholders and does not append an extra section' {
        Write-PromptDocument CHAT.md "first [SYSTEM_RESOURCE_STATUS] middle [SYSTEM_RESOURCE_STATUS] last"
        $actual = Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:promptFixture
        $actual | Should -BeExactly "first $script:literalTelemetry middle $script:literalTelemetry last"
        $actual | Should -Not -Match '\[SYSTEM_RESOURCE_STATUS\]'
        ([regex]::Matches($actual, 'CPU Usage:')).Count | Should -Be 2
    }

    # Protects: a marker-free Unicode document stays lossless and gains one block.
    # Detects: truncation, duplicate insertion, unrelated marker edits, or file mutation.
    # Needs: real large UTF8 document and exact appended resource fixture.
    # Breadcrumb: LLM-Helpers.ps1 Get-EnrichedSystemPrompt append branch and file read.
    It 'appends a single telemetry section to a large lossless Unicode document' {
        $literalPrompt = ('診断 café 🧭 é literal [OTHER_STATUS] ' * 4096) + 'terminal-marker'
        Write-PromptDocument CHAT.md $literalPrompt
        $actual = Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:promptFixture
        $actual | Should -BeExactly "$literalPrompt`n`n[SYSTEM_RESOURCE_STATUS]`n$script:literalTelemetry"
        ([regex]::Matches($actual, '\[SYSTEM_RESOURCE_STATUS\]')).Count | Should -Be 1
        [IO.File]::ReadAllText((Join-Path $script:promptFixture 'CHAT.md')) | Should -BeExactly $literalPrompt
    }

    # Protects: telemetry-provider failure leaves the original prompt usable.
    # Detects: losing text/placeholder or falling through to an extra hardware query.
    # Needs: owned prompt; throwing resource boundary; zero CIM invocation check.
    # Breadcrumb: LLM-Helpers.ps1 Get-EnrichedSystemPrompt manual telemetry catch.
    It 'preserves the exact original prompt when resource metrics throws' {
        $original = "留 intact`n[SYSTEM_RESOURCE_STATUS]`nend"
        Write-PromptDocument CHAT.md $original
        Mock Get-PcaiResourceMetrics { throw 'Owned telemetry provider failure.' }
        Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:promptFixture | Should -BeExactly $original
        Should -Invoke Get-PcaiResourceMetrics -Exactly -Times 1
        Should -Invoke Get-CimInstance -Exactly -Times 0
    }

    # Protects: documented CPU average and KiB-to-GiB fallback reporting.
    # Detects: wrong averaging/units, wrong telemetry provider, or duplicate queries.
    # Needs: absent resource discovery; mocked two CPUs and OS memory values.
    # Breadcrumb: LLM-Helpers.ps1 Get-EnrichedSystemPrompt basic CIM fallback.
    It 'uses concrete CIM fallback values only when the metrics command is absent' {
        Write-PromptDocument CHAT.md '[SYSTEM_RESOURCE_STATUS]'
        Mock Get-Command { $null } -ParameterFilter { $Name -eq 'Get-PcaiResourceMetrics' }
        Mock Get-CimInstance { @([pscustomobject]@{ LoadPercentage = 10 }, [pscustomobject]@{ LoadPercentage = 30 }) } -ParameterFilter { $ClassName -eq 'Win32_Processor' }
        Mock Get-CimInstance { [pscustomobject]@{ FreePhysicalMemory = 2MB; TotalVisibleMemorySize = 8MB } } -ParameterFilter { $ClassName -eq 'Win32_OperatingSystem' }
        Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:promptFixture | Should -BeExactly "CPU Usage: 20%`nMemory: 2 GB free / 8 GB total"
        Should -Invoke Get-PcaiResourceMetrics -Exactly -Times 0
        Should -Invoke Get-CimInstance -Exactly -Times 1 -ParameterFilter { $ClassName -eq 'Win32_Processor' }
        Should -Invoke Get-CimInstance -Exactly -Times 1 -ParameterFilter { $ClassName -eq 'Win32_OperatingSystem' }
    }

    # Protects: failed CIM collection preserves the exact original prompt.
    # Detects: exception escape, partial telemetry append, or extra provider calls.
    # Needs: absent resource discovery; owned prompt; throwing CIM boundary.
    # Breadcrumb: LLM-Helpers.ps1 Get-EnrichedSystemPrompt fallback catch.
    It 'preserves the document when the fallback hardware provider fails' {
        Write-PromptDocument CHAT.md 'fallback failure preserves me'
        Mock Get-Command { $null } -ParameterFilter { $Name -eq 'Get-PcaiResourceMetrics' }
        Mock Get-CimInstance { throw 'Owned CIM boundary failure.' }
        Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:promptFixture | Should -BeExactly 'fallback failure preserves me'
        Should -Invoke Get-CimInstance -Exactly -Times 1
    }

    # Protects: omitted explicit root uses the known Root resolver contract.
    # Detects: bypassing the resolver or reading documents from an unrelated root.
    # Needs: mocked Resolve-PcaiPath Root and an owned document at its result.
    # Breadcrumb: LLM-Helpers.ps1 Get-EnrichedSystemPrompt ProjectRoot resolution.
    It 'resolves the known project root when the caller omits it' {
        Write-PromptDocument CHAT.md '[SYSTEM_RESOURCE_STATUS]'
        Get-EnrichedSystemPrompt -Mode chat | Should -BeExactly $script:literalTelemetry
        Should -Invoke Resolve-PcaiPath -Exactly -Times 1 -ParameterFilter { $PathType -eq 'Root' }
    }

    # Protects: concrete summary units/rounding plus verbatim snapshot retention.
    # Detects: wrong metric conversion, raw JSON loss, or redundant manual queries.
    # Needs: synthetic JSON returned only through the mocked dashboard boundary.
    # Breadcrumb: LLM-Helpers.ps1 Get-EnrichedSystemPrompt native summary; no native-runtime proof.
    It 'formats concrete native-boundary metrics and retains the complete snapshot' {
        Write-PromptDocument CHAT.md '[SYSTEM_RESOURCE_STATUS]'
        $script:syntheticSnapshot = '{"Metrics":{"cpu_usage_perc":12.345,"memory_usage_bytes":3221225472,"total_memory_bytes":17179869184},"Note":"owned synthetic dashboard"}'
        Mock Get-PcaiNativeDashboardSnapshot { $script:syntheticSnapshot }
        Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:promptFixture |
            Should -BeExactly "### LIVE SYSTEM TELEMETRY (NATIVE)`nCPU Usage: 12.34%`nMemory: 3 GB / 16 GB`n$script:syntheticSnapshot"
        Should -Invoke Get-PcaiResourceMetrics -Exactly -Times 0
        Should -Invoke Get-CimInstance -Exactly -Times 0
    }

    # Protects: compatibility CPU/memory property names retain their public meaning.
    # Detects: lost legacy metric names, incorrect units, or resource fallback.
    # Needs: exact synthetic PascalCase JSON through mocked dashboard boundary.
    # Breadcrumb: LLM-Helpers.ps1 Get-EnrichedSystemPrompt compatibility metric lookup.
    It 'formats compatibility metric names from a native-boundary snapshot' {
        Write-PromptDocument CHAT.md '[SYSTEM_RESOURCE_STATUS]'
        $script:syntheticSnapshot = '{"Metrics":{"CpuUsage":25,"MemoryUsage":1073741824,"TotalMemory":8589934592}}'
        Mock Get-PcaiNativeDashboardSnapshot { $script:syntheticSnapshot }
        Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:promptFixture |
            Should -BeExactly "### LIVE SYSTEM TELEMETRY (NATIVE)`nCPU Usage: 25%`nMemory: 1 GB / 8 GB`n$script:syntheticSnapshot"
        Should -Invoke Get-PcaiResourceMetrics -Exactly -Times 0
    }

    # Protects: malformed/absent/zero-total metrics report n/a and retain raw data.
    # Detects: invalid division, JSON exception escape, data loss, or extra collection.
    # Needs: three synthetic boundary strings; no type injection/native execution.
    # Breadcrumb: LLM-Helpers.ps1 Get-EnrichedSystemPrompt JSON catch and n/a summary.
    It 'reports unavailable summary values for <Scenario> while preserving raw snapshot' -ForEach @(
        @{ Scenario = 'malformed JSON'; Snapshot = '{broken-json' },
        @{ Scenario = 'absent metrics'; Snapshot = '{"Note":"no metrics"}' },
        @{ Scenario = 'zero memory total'; Snapshot = '{"Metrics":{"memory_usage_bytes":1073741824,"total_memory_bytes":0}}' }
    ) {
        Write-PromptDocument CHAT.md '[SYSTEM_RESOURCE_STATUS]'
        $script:syntheticSnapshot = $Snapshot
        Mock Get-PcaiNativeDashboardSnapshot { $script:syntheticSnapshot }
        Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:promptFixture |
            Should -BeExactly "### LIVE SYSTEM TELEMETRY (NATIVE)`nCPU Usage: n/a`nMemory: n/a`n$Snapshot"
        Should -Invoke Get-PcaiResourceMetrics -Exactly -Times 0
        Should -Invoke Get-CimInstance -Exactly -Times 0
    }

    # Protects: native dashboard failure preserves the resource metrics fallback.
    # Detects: escaping dashboard failure, dropped telemetry, or hardware fallback.
    # Needs: throwing dashboard mock; concrete resource result; zero CIM check.
    # Breadcrumb: LLM-Helpers.ps1 Get-EnrichedSystemPrompt outer native catch and fallback.
    It 'falls back to concrete resource metrics when the native boundary throws' {
        Write-PromptDocument CHAT.md '[SYSTEM_RESOURCE_STATUS]'
        Mock Get-PcaiNativeDashboardSnapshot { throw 'Owned dashboard boundary failure.' }
        Get-EnrichedSystemPrompt -Mode chat -ProjectRoot $script:promptFixture | Should -BeExactly $script:literalTelemetry
        Should -Invoke Get-PcaiResourceMetrics -Exactly -Times 1
        Should -Invoke Get-CimInstance -Exactly -Times 0
    }
}
