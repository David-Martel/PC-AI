# The complete adapters and serializers run; only their provider/native CLI
# boundary is inert. No model, HTTP provider, native binary or job is invoked.
BeforeAll {
    . (Join-Path $PSScriptRoot '../../Modules/PC-AI.LLM/Private/LLM-Helpers.ps1')
    . (Join-Path $PSScriptRoot '../../Modules/PC-AI.LLM/Public/Send-OllamaRequest.ps1')
    function New-InertMetadataDictionary {
        param([string]$Kind, [System.Collections.IDictionary]$Entries)
        $dictionary = if ($Kind -ceq 'Generic') { [Collections.Generic.Dictionary[string, object]]::new() }
        elseif ($Kind -ceq 'Ordered') { [Collections.Specialized.OrderedDictionary]::new() }
        else { throw 'Unknown inert dictionary type.' }
        foreach ($key in $Entries.PSBase.Keys) { $dictionary.Add($key, $Entries[$key]) }
        return ,$dictionary
    }
}

Describe 'Actual native chat optional metadata' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        Set-StrictMode -Version Latest
        $script:ModuleConfig = @{ DefaultModel = 'fixture'; DefaultTimeout = 7; ToolsPath = 'fixture-tools'; OllamaToolModel = 'fixture' }
        $script:NativeReply = [pscustomobject]@{ ok = $true; content = 'useful answer'; model = 'fixture' }
        $script:CapturedRequest = $null
        Mock Invoke-OllamaNativeCli {
            $Arguments[0] | Should -Be 'chat'
            $Arguments[1] | Should -Be '--request-file'
            $script:CapturedRequest = Get-Content -LiteralPath $Arguments[2] -Raw | ConvertFrom-Json
            $script:NativeReply
        }
    }

    # Protects useful provider answers when metadata is optional. Detects
    # strict missing/null field access through the real generation adapter.
    It 'retains useful native responses with omitted optional metadata' {
        $result = Invoke-OllamaGenerate -Prompt 'inert fixture' -Model fixture -ErrorAction Stop
        $result.message.content | Should -Be 'useful answer'
        $result.model | Should -Be fixture
        @($result.ToolCalls) | Should -HaveCount 0
        @($result.ExecutedTools) | Should -HaveCount 0
        $result.total_duration | Should -BeNullOrEmpty
        [object]::ReferenceEquals($result.raw, $script:NativeReply) | Should -BeTrue
    }
    It 'treats explicit null optional metadata as empty tools and no timing' {
        $script:NativeReply | Add-Member toolCalls $null
        $script:NativeReply | Add-Member executedTools $null
        $script:NativeReply | Add-Member timing $null
        $result = Invoke-OllamaGenerate -Prompt 'inert fixture' -Model fixture -ErrorAction Stop
        $result.message.content | Should -Be 'useful answer'
        @($result.ToolCalls) | Should -HaveCount 0
        @($result.ExecutedTools) | Should -HaveCount 0
        $result.total_duration | Should -BeNullOrEmpty
    }
    It 'retains a timing object with omitted duration without inventing a metric' {
        $script:NativeReply | Add-Member toolCalls @()
        $script:NativeReply | Add-Member executedTools @()
        $script:NativeReply | Add-Member timing ([pscustomobject]@{ otherMetric = 4 })
        $result = Invoke-OllamaGenerate -Prompt 'inert fixture' -Model fixture -ErrorAction Stop
        $result.message.content | Should -Be 'useful answer'
        $result.total_duration | Should -BeNullOrEmpty
        $result.raw.timing.otherMetric | Should -Be 4
    }
    It 'preserves populated tool arrays and explicit zero duration by identity' {
        $calls = @([pscustomobject]@{ name = 'one' }, [pscustomobject]@{ name = 'two' })
        $executed = @([pscustomobject]@{ name = 'one'; result = 'inert' })
        $timing = [pscustomobject]@{ totalDurationNs = 0 }
        $script:NativeReply | Add-Member toolCalls $calls
        $script:NativeReply | Add-Member executedTools $executed
        $script:NativeReply | Add-Member timing $timing
        $result = Invoke-OllamaGenerate -Prompt 'inert fixture' -Model fixture -ErrorAction Stop
        @($result.ToolCalls) | Should -HaveCount 2
        @($result.ExecutedTools) | Should -HaveCount 1
        [object]::ReferenceEquals($result.ToolCalls[0], $calls[0]) | Should -BeTrue
        [object]::ReferenceEquals($result.ExecutedTools[0], $executed[0]) | Should -BeTrue
        $result.total_duration | Should -Be 0
        [object]::ReferenceEquals($result.raw.timing, $timing) | Should -BeTrue
    }
    # Protects the actual wire request. Needs literal zero options, not defaults
    # or a replacement formatter, and a real serialized temporary request.
    It 'preserves explicit zero sampling settings in the actual serialized request' {
        $script:NativeReply | Add-Member toolCalls @()
        $script:NativeReply | Add-Member executedTools @()
        $script:NativeReply | Add-Member timing ([pscustomobject]@{ totalDurationNs = 1 })
        $result = Invoke-OllamaGenerate -Prompt 'inert fixture' -Model fixture -Temperature 0 -TopP 0 -RepeatPenalty 0 -TfsZ 0 -Seed 0 -ErrorAction Stop
        $result.message.content | Should -Be 'useful answer'
        foreach ($name in 'temperature', 'topP', 'repeatPenalty', 'tfsZ', 'seed') {
            $script:CapturedRequest.PSObject.Properties[$name].Value | Should -Be 0
        }
    }
    # Missing required payload data must remain a failure, not a blank success.
    It 'rejects a native response missing required content' {
        $script:NativeReply.PSObject.Properties.Remove('content')
        { Invoke-OllamaGenerate -Prompt 'inert fixture' -Model fixture -ErrorAction Stop } | Should -Throw '*content*'
    }
    It 'rejects a native response missing required model' {
        $script:NativeReply.PSObject.Properties.Remove('model')
        { Invoke-OllamaGenerate -Prompt 'inert fixture' -Model fixture -ErrorAction Stop } | Should -Throw '*model*'
    }
    # Protects real IDictionary implementations, case-compatible metadata and
    # original values. Detects Contains overload assumptions and Keys shadowing.
    It 'accepts a <Kind> native response with <Metadata> metadata' -ForEach @(
        @{ Kind = 'Generic'; Metadata = 'missing' }
        @{ Kind = 'Ordered'; Metadata = 'missing' }
        @{ Kind = 'Generic'; Metadata = 'populated' }
        @{ Kind = 'Ordered'; Metadata = 'populated' }
    ) {
        $script:NativeReply = New-InertMetadataDictionary -Kind $Kind -Entries ([ordered]@{ ok = $true; content = 'dictionary answer'; model = 'fixture'; Keys = 'inert shadow' })
        $calls = @([pscustomobject]@{ name = 'one' }, [pscustomobject]@{ name = 'two' })
        $executed = @([pscustomobject]@{ name = 'one'; result = 'inert' })
        $timing = New-InertMetadataDictionary -Kind $Kind -Entries ([ordered]@{ TOTALDURATIONNS = 0; Keys = 'inert shadow' })
        if ($Metadata -ceq 'populated') {
            $script:NativeReply.Add('TOOLCALLS', $calls)
            $script:NativeReply.Add('EXECUTEDTOOLS', $executed)
            $script:NativeReply.Add('TIMING', $timing)
        }
        $result = Invoke-OllamaGenerate -Prompt 'inert fixture' -Model fixture -ErrorAction Stop
        $result.message.content | Should -Be 'dictionary answer'
        [object]::ReferenceEquals($result.raw, $script:NativeReply) | Should -BeTrue
        if ($Metadata -ceq 'missing') {
            @($result.ToolCalls) | Should -HaveCount 0
            @($result.ExecutedTools) | Should -HaveCount 0
            $result.total_duration | Should -BeNullOrEmpty
        } else {
            @($result.ToolCalls) | Should -HaveCount 2
            @($result.ExecutedTools) | Should -HaveCount 1
            [object]::ReferenceEquals($result.ToolCalls[0], $calls[0]) | Should -BeTrue
            [object]::ReferenceEquals($result.ExecutedTools[0], $executed[0]) | Should -BeTrue
            $result.total_duration | Should -Be 0
        }
    }
    It 'rejects ambiguous case-distinct native metadata in <Kind> dictionaries' -ForEach @(
        @{ Kind = 'Generic' }
        @{ Kind = 'Ordered' }
    ) {
        $script:NativeReply = New-InertMetadataDictionary -Kind $Kind -Entries ([ordered]@{ ok = $true; content = 'dictionary answer'; model = 'fixture' })
        $script:NativeReply.Add('toolCalls', @())
        $script:NativeReply.Add('TOOLCALLS', @([pscustomobject]@{ name = 'ambiguous' }))
        { Invoke-OllamaGenerate -Prompt 'inert fixture' -Model fixture -ErrorAction Stop } | Should -Throw '*Ambiguous*toolCalls*'
    }
}

Describe 'Actual public generation response optional metadata' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeEach {
        Set-StrictMode -Version Latest
        $script:ModuleConfig = @{ DefaultModel = 'fixture'; DefaultTimeout = 7 }
        $script:GenerationReply = [pscustomobject]@{ message = [pscustomobject]@{ content = 'useful answer' }; model = 'fixture' }
        Mock Test-OllamaConnection { $true }
        Mock Get-OllamaModels { [pscustomobject]@{ Name = 'fixture' } }
        Mock Invoke-OllamaGenerate { $script:GenerationReply }
    }
    # Protects the public schema without requiring provider-specific metrics.
    # Needs omitted/null raw, omitted/null timing, and null/present tool arrays.
    It 'retains useful generation responses when raw and tools are omitted' {
        $result = Send-OllamaRequest -Prompt 'inert fixture' -Model fixture -MaxRetries 0 -ErrorAction Stop
        $result.Response | Should -Be 'useful answer'
        $result.Model | Should -Be fixture
        $result.Usage | Should -BeNullOrEmpty
        @($result.ToolCalls) | Should -HaveCount 0
        @($result.ExecutedTools) | Should -HaveCount 0
        Should -Invoke Invoke-OllamaGenerate -Exactly -Times 1
    }
    It 'accepts explicitly null raw and tool metadata' {
        $script:GenerationReply | Add-Member raw $null
        $script:GenerationReply | Add-Member ToolCalls $null
        $script:GenerationReply | Add-Member ExecutedTools $null
        $result = Send-OllamaRequest -Prompt 'inert fixture' -Model fixture -MaxRetries 0 -ErrorAction Stop
        $result.Response | Should -Be 'useful answer'
        $result.Usage | Should -BeNullOrEmpty
        @($result.ToolCalls) | Should -HaveCount 0
        @($result.ExecutedTools) | Should -HaveCount 0
    }
    It 'accepts a raw dictionary with omitted timing' {
        $script:GenerationReply | Add-Member raw @{ otherMetric = 4 }
        $script:GenerationReply | Add-Member ToolCalls @()
        $script:GenerationReply | Add-Member ExecutedTools @()
        $result = Send-OllamaRequest -Prompt 'inert fixture' -Model fixture -MaxRetries 0 -ErrorAction Stop
        $result.Response | Should -Be 'useful answer'
        $result.Usage | Should -BeNullOrEmpty
    }
    It 'accepts explicitly null timing inside a raw object' {
        $script:GenerationReply | Add-Member raw ([pscustomobject]@{ timing = $null })
        $script:GenerationReply | Add-Member ToolCalls @()
        $script:GenerationReply | Add-Member ExecutedTools @()
        $result = Send-OllamaRequest -Prompt 'inert fixture' -Model fixture -MaxRetries 0 -ErrorAction Stop
        $result.Response | Should -Be 'useful answer'
        $result.Usage | Should -BeNullOrEmpty
    }
    It 'preserves supplied usage and tool objects without replacing them' {
        $timing = @{ totalDurationNs = 3 }
        $calls = @([pscustomobject]@{ name = 'one' }, [pscustomobject]@{ name = 'two' })
        $executed = @([pscustomobject]@{ name = 'one'; result = 'inert' })
        $script:GenerationReply | Add-Member raw @{ timing = $timing }
        $script:GenerationReply | Add-Member ToolCalls $calls
        $script:GenerationReply | Add-Member ExecutedTools $executed
        $result = Send-OllamaRequest -Prompt 'inert fixture' -Model fixture -MaxRetries 0 -ErrorAction Stop
        [object]::ReferenceEquals($result.Usage, $timing) | Should -BeTrue
        @($result.ToolCalls) | Should -HaveCount 2
        @($result.ExecutedTools) | Should -HaveCount 1
        [object]::ReferenceEquals($result.ToolCalls[0], $calls[0]) | Should -BeTrue
        [object]::ReferenceEquals($result.ExecutedTools[0], $executed[0]) | Should -BeTrue
    }
    It 'rejects a generation response missing required message content' {
        $script:GenerationReply.message.PSObject.Properties.Remove('content')
        { Send-OllamaRequest -Prompt 'inert fixture' -Model fixture -MaxRetries 0 -ErrorAction Stop } | Should -Throw '*content*'
    }
    It 'rejects a generation response missing required model' {
        $script:GenerationReply.PSObject.Properties.Remove('model')
        { Send-OllamaRequest -Prompt 'inert fixture' -Model fixture -MaxRetries 0 -ErrorAction Stop } | Should -Throw '*model*'
    }
    # Protects actual outer and nested dictionary metadata through the full
    # public wrapper. Needs both ordinary Dictionary and OrderedDictionary.
    It 'accepts a <Kind> generation response with <Metadata> metadata' -ForEach @(
        @{ Kind = 'Generic'; Metadata = 'missing' }
        @{ Kind = 'Ordered'; Metadata = 'missing' }
        @{ Kind = 'Generic'; Metadata = 'populated' }
        @{ Kind = 'Ordered'; Metadata = 'populated' }
    ) {
        $script:GenerationReply = New-InertMetadataDictionary -Kind $Kind -Entries ([ordered]@{ message = [pscustomobject]@{ content = 'dictionary answer' }; model = 'fixture'; Keys = 'inert shadow' })
        $calls = @([pscustomobject]@{ name = 'one' }, [pscustomobject]@{ name = 'two' })
        $executed = @([pscustomobject]@{ name = 'one'; result = 'inert' })
        $timing = New-InertMetadataDictionary -Kind $Kind -Entries ([ordered]@{ totalDurationNs = 0; Keys = 'inert shadow' })
        $raw = New-InertMetadataDictionary -Kind $Kind -Entries ([ordered]@{ TIMING = $timing; Keys = 'inert shadow' })
        if ($Metadata -ceq 'populated') {
            $script:GenerationReply.Add('RAW', $raw)
            $script:GenerationReply.Add('TOOLCALLS', $calls)
            $script:GenerationReply.Add('EXECUTEDTOOLS', $executed)
        }
        $result = Send-OllamaRequest -Prompt 'inert fixture' -Model fixture -MaxRetries 0 -ErrorAction Stop
        $result.Response | Should -Be 'dictionary answer'
        if ($Metadata -ceq 'missing') {
            $result.Usage | Should -BeNullOrEmpty
            @($result.ToolCalls) | Should -HaveCount 0
            @($result.ExecutedTools) | Should -HaveCount 0
        } else {
            [object]::ReferenceEquals($result.Usage, $timing) | Should -BeTrue
            @($result.ToolCalls) | Should -HaveCount 2
            @($result.ExecutedTools) | Should -HaveCount 1
            [object]::ReferenceEquals($result.ToolCalls[0], $calls[0]) | Should -BeTrue
            [object]::ReferenceEquals($result.ExecutedTools[0], $executed[0]) | Should -BeTrue
            $result.Usage['totalDurationNs'] | Should -Be 0
        }
    }
    It 'rejects ambiguous case-distinct generation metadata in <Kind> dictionaries' -ForEach @(
        @{ Kind = 'Generic' }
        @{ Kind = 'Ordered' }
    ) {
        $script:GenerationReply = New-InertMetadataDictionary -Kind $Kind -Entries ([ordered]@{ message = [pscustomobject]@{ content = 'dictionary answer' }; model = 'fixture' })
        $script:GenerationReply.Add('ToolCalls', @())
        $script:GenerationReply.Add('TOOLCALLS', @([pscustomobject]@{ name = 'ambiguous' }))
        { Send-OllamaRequest -Prompt 'inert fixture' -Model fixture -MaxRetries 0 -ErrorAction Stop } | Should -Throw '*Ambiguous*ToolCalls*'
    }
}

Describe 'Actual progress declarations-only parser guard' -Tag 'Unit', 'LLM', 'Portable' {
    BeforeAll {
        $script:HelperText = [IO.File]::ReadAllText((Join-Path $PSScriptRoot '../../Modules/PC-AI.LLM/Private/LLM-Helpers.ps1'))
        $tokens = $null
        $errors = $null
        $progressAst = [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'LLMProgressJobCustody.Tests.ps1'), [ref]$tokens, [ref]$errors)
        if ($null -ne $errors -and @($errors).Count -ne 0) { throw 'Progress fixture must parse before its actual guard is evaluated.' }
        $guards = @($progressAst.FindAll({
            param($node)
            $node -is [Management.Automation.Language.IfStatementAst] -and
            $node.Extent.Text.Contains("throw 'Canonical helper must contain only its 23 function declarations before dot-sourcing.'", [StringComparison]::Ordinal)
        }, $true))
        if ($guards.Count -ne 1) { throw 'One exact maintained declarations-only guard required.' }
        $script:ActualProgressGuard = [scriptblock]::Create($guards[0].Clauses[0].Item1.Extent.Text)
    }
    BeforeEach { Set-StrictMode -Version Latest }
    # Protects the real admission predicate and every retained AST check.
    # Detects null report access without executing source or using fake jobs.
    It 'admits the actual valid helper declarations with the parser error report' {
        $tokens = $null
        $parseErrors = $null
        $helperAst = [Management.Automation.Language.Parser]::ParseInput($script:HelperText, [ref]$tokens, [ref]$parseErrors)
        $top = @($helperAst.EndBlock.Statements)
        @($top) | Should -HaveCount 23
        (& $script:ActualProgressGuard) | Should -BeFalse
    }
    It 'admits valid helper declarations when the parse error report is null' {
        $tokens = $null
        $parseErrors = $null
        $helperAst = [Management.Automation.Language.Parser]::ParseInput($script:HelperText, [ref]$tokens, [ref]$parseErrors)
        $top = @($helperAst.EndBlock.Statements)
        $parseErrors = $null
        (& $script:ActualProgressGuard) | Should -BeFalse
    }
    It 'rejects malformed helper source using actual parser errors' {
        $tokens = $null
        $parseErrors = $null
        $helperAst = [Management.Automation.Language.Parser]::ParseInput(($script:HelperText + "`nfunction Malformed {"), [ref]$tokens, [ref]$parseErrors)
        $top = @($helperAst.EndBlock.Statements)
        @($parseErrors).Count | Should -BeGreaterThan 0
        (& $script:ActualProgressGuard) | Should -BeTrue
    }
    It 'rejects executable top-level source without invoking it' {
        $tokens = $null
        $parseErrors = $null
        $helperAst = [Management.Automation.Language.Parser]::ParseInput(($script:HelperText + "`nWrite-Output 'inert AST only'"), [ref]$tokens, [ref]$parseErrors)
        $top = @($helperAst.EndBlock.Statements)
        (& $script:ActualProgressGuard) | Should -BeTrue
    }
    # Protects the separate trap check: trap-only source retains exactly the
    # same 23 declarations, so declaration count cannot reject it by itself.
    It 'rejects a parsed top-level trap while retaining the 23 declarations' {
        $tokens = $null
        $parseErrors = $null
        $helperAst = [Management.Automation.Language.Parser]::ParseInput(($script:HelperText + "`ntrap { continue }"), [ref]$tokens, [ref]$parseErrors)
        $top = @($helperAst.EndBlock.Statements)
        $top | Should -HaveCount 23
        @($helperAst.EndBlock.Traps) | Should -HaveCount 1
        (& $script:ActualProgressGuard) | Should -BeTrue
    }
}
