#Requires -Version 7.0

# Environment variables are saved and restored around every test. No network, model or GPU is used.
BeforeAll {
    $script:RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
    . (Join-Path $script:RepoRoot 'Modules/PC-AI.Common/Public/Get-PcaiOriginBearerHeader.ps1')

    # The hardware runner is a standalone script that needs a CUDA server and a model, so only the
    # function that builds its request parameters is lifted out of the file and tested.
    $runner = Join-Path $script:RepoRoot 'Tests/Hardware/Invoke-PcaiMediaCudaFunctionalTests.ps1'
    $parserErrors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($runner, [ref]$null, [ref]$parserErrors)
    if ($parserErrors) { throw ($parserErrors | Out-String) }
    $function = $ast.Find({
            param($node)
            $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq 'Get-MediaRequestParameter'
        }, $false)
    if (-not $function) { throw 'Get-MediaRequestParameter is missing from the media runner.' }
    . ([scriptblock]::Create($function.Extent.Text))
}

Describe 'Origin-scoped bearer headers' -Tag 'Unit', 'Portable' {
BeforeEach {
    $script:SavedTokens = @{}
    foreach ($name in 'PCAI_INFERENCE_TOKEN', 'PCAI_MEDIA_TOKEN') {
        $script:SavedTokens[$name] = [Environment]::GetEnvironmentVariable($name)
        [Environment]::SetEnvironmentVariable($name, $null)
    }
}

AfterEach {
    foreach ($name in $script:SavedTokens.Keys) {
        [Environment]::SetEnvironmentVariable($name, $script:SavedTokens[$name])
    }
}

Context 'Get-PcaiOriginBearerHeader' {
    # Protects: the one rule every client uses to attach a server token: same origin only.
    # Detects: the header sent to a different host, port or scheme (credential leak to another
    # provider), the origin check being skipped, or a header sent with no token configured.
    # Needs: PowerShell 7 and Pester 5; environment variables only.
    # Breadcrumb: Modules/PC-AI.Common/Public/Get-PcaiOriginBearerHeader.ps1.
    It 'returns the bearer header for the same origin: <Request> against <Base>' -TestCases @(
        @{ Request = 'http://127.0.0.1:18080/v1/models'; Base = 'http://127.0.0.1:18080' }
        @{ Request = 'http://127.0.0.1:18080/v1/chat/completions'; Base = 'http://127.0.0.1:18080/' }
        @{ Request = 'HTTP://LocalHost:8080/v1/models'; Base = 'http://localhost:8080' }
        @{ Request = 'https://inference.lan/v1/models'; Base = 'https://inference.lan:443' }
        @{ Request = 'http://[::1]:8080/v1/models'; Base = 'http://[::1]:8080' }
    ) {
        param($Request, $Base)
        [Environment]::SetEnvironmentVariable('PCAI_INFERENCE_TOKEN', ' unit-token ')
        $header = Get-PcaiOriginBearerHeader -RequestUrl $Request -BaseUrl $Base -TokenVariable 'PCAI_INFERENCE_TOKEN'
        $header.Authorization | Should -BeExactly 'Bearer unit-token'
    }

    It 'returns nothing for another origin: <Request> against <Base>' -TestCases @(
        @{ Request = 'http://127.0.0.1:11434/api/generate'; Base = 'http://127.0.0.1:18080' }
        @{ Request = 'http://localhost:18080/v1/models'; Base = 'http://127.0.0.1:18080' }
        @{ Request = 'https://127.0.0.1:18080/v1/models'; Base = 'http://127.0.0.1:18080' }
        @{ Request = 'http://127.0.0.1:18080.evil.example/v1/models'; Base = 'http://127.0.0.1:18080' }
        @{ Request = 'http://evil.example/v1/models'; Base = 'http://127.0.0.1:18080' }
        @{ Request = 'http://[::1]:8080/v1/models'; Base = 'http://127.0.0.1:8080' }
    ) {
        param($Request, $Base)
        [Environment]::SetEnvironmentVariable('PCAI_INFERENCE_TOKEN', 'unit-token')
        $header = Get-PcaiOriginBearerHeader -RequestUrl $Request -BaseUrl $Base -TokenVariable 'PCAI_INFERENCE_TOKEN'
        $header.Count | Should -Be 0
    }

    It 'returns nothing without a usable token or URL: <Case>' -TestCases @(
        @{ Case = 'unset token'; Token = $null; Request = 'http://127.0.0.1:1/x'; Base = 'http://127.0.0.1:1' }
        @{ Case = 'empty token'; Token = ''; Request = 'http://127.0.0.1:1/x'; Base = 'http://127.0.0.1:1' }
        @{ Case = 'blank token'; Token = '   '; Request = 'http://127.0.0.1:1/x'; Base = 'http://127.0.0.1:1' }
        @{ Case = 'relative request URL'; Token = 't'; Request = '/v1/models'; Base = 'http://127.0.0.1:1' }
        @{ Case = 'auto base URL'; Token = 't'; Request = 'http://127.0.0.1:1/x'; Base = 'auto' }
        @{ Case = 'empty base URL'; Token = 't'; Request = 'http://127.0.0.1:1/x'; Base = '' }
        @{ Case = 'null request URL'; Token = 't'; Request = $null; Base = 'http://127.0.0.1:1' }
        @{ Case = 'non-http scheme'; Token = 't'; Request = 'ftp://127.0.0.1:1/x'; Base = 'ftp://127.0.0.1:1' }
    ) {
        param($Token, $Request, $Base)
        [Environment]::SetEnvironmentVariable('PCAI_INFERENCE_TOKEN', $Token)
        { Get-PcaiOriginBearerHeader -RequestUrl $Request -BaseUrl $Base -TokenVariable 'PCAI_INFERENCE_TOKEN' } | Should -Not -Throw
        (Get-PcaiOriginBearerHeader -RequestUrl $Request -BaseUrl $Base -TokenVariable 'PCAI_INFERENCE_TOKEN').Count | Should -Be 0
    }

    It 'reads only the named variable' {
        [Environment]::SetEnvironmentVariable('PCAI_INFERENCE_TOKEN', 'inference-token')
        (Get-PcaiOriginBearerHeader -RequestUrl 'http://127.0.0.1:1/x' -BaseUrl 'http://127.0.0.1:1' -TokenVariable 'PCAI_MEDIA_TOKEN').Count | Should -Be 0
    }

    It 'rejects a token variable name that is not an identifier' {
        { Get-PcaiOriginBearerHeader -RequestUrl 'http://127.0.0.1:1/x' -BaseUrl 'http://127.0.0.1:1' -TokenVariable 'a b' } | Should -Throw
    }
}

Context 'Media CUDA runner request parameters' {
    # Protects: the hardware runner against a token-protected media server. The child inherits
    # PCAI_MEDIA_TOKEN and then refuses every route but /health, so the runner's own POSTs must carry
    # the bearer header, and only to the child's loopback URL.
    # Detects: the header dropped (every case would fail with 401), or the pcai-inference token used
    # for the media server.
    # Needs: PowerShell 7 and Pester 5; the function is lifted from the runner, no server, model or GPU.
    # Breadcrumb: Tests/Hardware/Invoke-PcaiMediaCudaFunctionalTests.ps1 Get-MediaRequestParameter.
    BeforeEach {
        $script:Case = @{ Name = 'generate/simple'; Operation = 'generate'; Payload = @{ prompt = 'a red apple'; temperature = 1.0 } }
    }

    It 'adds the bearer header to the POST for the child server URL' {
        [Environment]::SetEnvironmentVariable('PCAI_MEDIA_TOKEN', 'media-token')
        $parameters = Get-MediaRequestParameter -BaseUri 'http://127.0.0.1:43210' -Case $script:Case -TimeoutSeconds 30
        $parameters.Uri | Should -BeExactly 'http://127.0.0.1:43210/v1/images/generate'
        $parameters.Method | Should -BeExactly 'Post'
        $parameters.Headers.Authorization | Should -BeExactly 'Bearer media-token'
        $parameters.TimeoutSec | Should -Be 30
        $parameters.NoProxy | Should -BeTrue
        $parameters.MaximumRedirection | Should -Be 0
        ($parameters.Body | ConvertFrom-Json).prompt | Should -Be 'a red apple'
    }

    It 'sends no header when no media token is configured' {
        $parameters = Get-MediaRequestParameter -BaseUri 'http://127.0.0.1:43210' -Case $script:Case -TimeoutSeconds 30
        $parameters.ContainsKey('Headers') | Should -BeFalse
    }

    It 'does not use the pcai-inference token for the media server' {
        [Environment]::SetEnvironmentVariable('PCAI_INFERENCE_TOKEN', 'inference-token')
        $parameters = Get-MediaRequestParameter -BaseUri 'http://127.0.0.1:43210' -Case $script:Case -TimeoutSeconds 30
        $parameters.ContainsKey('Headers') | Should -BeFalse
    }
}
}
