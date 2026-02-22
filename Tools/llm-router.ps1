#Requires -Version 5.1

<#
.SYNOPSIS
  Lightweight Ollama-compatible router with LM Studio fallback.

.DESCRIPTION
  Listens on a local port and forwards Ollama API requests to Ollama when available.
  If Ollama is down, it converts requests to LM Studio's OpenAI-compatible API.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File .\llm-router.ps1

.EXAMPLE
  .\llm-router.ps1 -ListenPort 11435
#>

[CmdletBinding()]
param(
    [Parameter()]
    [ValidateRange(1024, 65535)]
    [int]$ListenPort = 11435,

    [Parameter()]
    [string]$OllamaBaseUrl,

    [Parameter()]
    [string]$LMStudioBaseUrl,

    [Parameter()]
    [ValidateSet('Ollama', 'LMStudio')]
    [string]$Prefer = 'Ollama'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Read-RequestBody {
    param([System.Net.HttpListenerRequest]$Request)

    if (-not $Request.HasEntityBody) {
        return ''
    }

    $reader = New-Object System.IO.StreamReader($Request.InputStream, $Request.ContentEncoding)
    try {
        return $reader.ReadToEnd()
    } finally {
        $reader.Close()
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$cfgPath = Join-Path $repoRoot 'Config\llm-config.json'
if (Test-Path $cfgPath) {
    try {
        $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
        if (-not $OllamaBaseUrl -and $cfg.providers.ollama.baseUrl) {
            $OllamaBaseUrl = $cfg.providers.ollama.baseUrl
        }
        if (-not $LMStudioBaseUrl -and $cfg.providers.lmstudio.baseUrl) {
            $LMStudioBaseUrl = $cfg.providers.lmstudio.baseUrl
        }
    } catch {}
}

function Write-JsonResponse {
    param(
        [System.Net.HttpListenerResponse]$Response,
        [int]$StatusCode,
        [string]$Json
    )

    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Json)
    $Response.StatusCode = $StatusCode
    $Response.ContentType = 'application/json'
    $Response.ContentLength64 = $bytes.Length
    $Response.OutputStream.Write($bytes, 0, $bytes.Length)
    $Response.OutputStream.Close()
}

function Invoke-HttpJson {
    param(
        [string]$Method,
        [string]$Url,
        [string]$Body,
        [int]$TimeoutSec = 60
    )

    $params = @{
        Uri         = $Url
        Method      = $Method
        TimeoutSec  = $TimeoutSec
        ErrorAction = 'Stop'
    }

    if ($Body) {
        $params['Body'] = $Body
        $params['ContentType'] = 'application/json'
    }

    $resp = Invoke-WebRequest @params
    return [PSCustomObject]@{
        StatusCode = $resp.StatusCode
        Content    = $resp.Content
    }
}

function Try-Ollama {
    param([string]$Method, [string]$Path, [string]$Body)

    $url = "$OllamaBaseUrl$Path"
    try {
        return Invoke-HttpJson -Method $Method -Url $url -Body $Body
    } catch {
        return $null
    }
}

function Try-LMStudio {
    param([string]$Method, [string]$Path, [string]$Body)

    $url = "$LMStudioBaseUrl$Path"
    try {
        return Invoke-HttpJson -Method $Method -Url $url -Body $Body
    } catch {
        return $null
    }
}

function Convert-LMStudioModelsToOllamaTags {
    param([string]$LmJson)

    $lm = $LmJson | ConvertFrom-Json
    $models = @()

    foreach ($item in $lm.data) {
        $models += [PSCustomObject]@{
            name        = $item.id
            modified_at = $null
            size        = $null
            digest      = $null
            details     = @{}
        }
    }

    return (@{ models = $models } | ConvertTo-Json -Depth 6)
}

function Convert-LMStudioChatToOllamaResponse {
    param(
        [string]$LmJson,
        [string]$Model,
        [switch]$GenerateMode
    )

    $lm = $LmJson | ConvertFrom-Json
    $content = $null

    if ($lm.choices -and $lm.choices.Count -gt 0) {
        $choice = $lm.choices[0]
        if ($choice.message -and $choice.message.content) {
            $content = $choice.message.content
        } elseif ($choice.text) {
            $content = $choice.text
        }
    }

    if (-not $content) {
        $content = ''
    }

    if (-not $Model) {
        $Model = $lm.model
    }

    if ($GenerateMode) {
        $payload = [PSCustomObject]@{
            model      = $Model
            created_at = (Get-Date).ToString('o')
            response   = $content
            done       = $true
        }
    } else {
        $payload = [PSCustomObject]@{
            model      = $Model
            created_at = (Get-Date).ToString('o')
            message    = [PSCustomObject]@{ role = 'assistant'; content = $content }
            done       = $true
        }
    }

    return ($payload | ConvertTo-Json -Depth 6)
}

function Convert-OllamaGenerateToLMStudioChat {
    param([PSCustomObject]$OllamaRequest)

    $messages = @()
    if ($OllamaRequest.system) {
        $messages += @{ role = 'system'; content = $OllamaRequest.system }
    }

    $messages += @{ role = 'user'; content = $OllamaRequest.prompt }

    $body = @{
        model    = $OllamaRequest.model
        messages = $messages
        stream   = $OllamaRequest.stream
    }

    if ($OllamaRequest.options) {
        if ($null -ne $OllamaRequest.options.temperature) {
            $body['temperature'] = $OllamaRequest.options.temperature
        }
        if ($OllamaRequest.options.num_predict) {
            $body['max_tokens'] = $OllamaRequest.options.num_predict
        }
    }

    return ($body | ConvertTo-Json -Depth 8)
}

function Convert-OllamaChatToLMStudioChat {
    param([PSCustomObject]$OllamaRequest)

    $body = @{
        model    = $OllamaRequest.model
        messages = $OllamaRequest.messages
        stream   = $OllamaRequest.stream
    }

    if ($OllamaRequest.options) {
        if ($null -ne $OllamaRequest.options.temperature) {
            $body['temperature'] = $OllamaRequest.options.temperature
        }
        if ($OllamaRequest.options.num_predict) {
            $body['max_tokens'] = $OllamaRequest.options.num_predict
        }
    }

    return ($body | ConvertTo-Json -Depth 8)
}

$listener = New-Object System.Net.HttpListener
$prefix = "http://*:$ListenPort/"
$listener.Prefixes.Add($prefix)
$listener.Start()

Write-Host "LLM router listening on $prefix" -ForegroundColor Green
Write-Host "Ollama:  $OllamaBaseUrl" -ForegroundColor Gray
Write-Host "LM Studio: $LMStudioBaseUrl" -ForegroundColor Gray

try {
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $request = $context.Request
        $response = $context.Response

        $path = $request.Url.AbsolutePath
        $method = $request.HttpMethod.ToUpperInvariant()

        try {
            switch ($path) {
                '/api/tags' {
                    if ($Prefer -eq 'Ollama') {
                        $ollama = Try-Ollama -Method 'GET' -Path '/api/tags' -Body $null
                        if ($ollama) {
                            Write-JsonResponse -Response $response -StatusCode 200 -Json $ollama.Content
                            break
                        }
                    }

                    $lm = Try-LMStudio -Method 'GET' -Path '/v1/models' -Body $null
                    if (-not $lm) {
                        throw 'Neither Ollama nor LM Studio responded'
                    }

                    $json = Convert-LMStudioModelsToOllamaTags -LmJson $lm.Content
                    Write-JsonResponse -Response $response -StatusCode 200 -Json $json
                    break
                }
                '/api/generate' {
                    if ($method -ne 'POST') {
                        Write-JsonResponse -Response $response -StatusCode 405 -Json '{"error":"Method not allowed"}'
                        break
                    }

                    $bodyText = Read-RequestBody -Request $request

                    if ($Prefer -eq 'Ollama') {
                        $ollama = Try-Ollama -Method 'POST' -Path '/api/generate' -Body $bodyText
                        if ($ollama) {
                            Write-JsonResponse -Response $response -StatusCode 200 -Json $ollama.Content
                            break
                        }
                    }

                    $reqObj = $bodyText | ConvertFrom-Json
                    if ($reqObj.stream -eq $true) {
                        Write-JsonResponse -Response $response -StatusCode 400 -Json '{"error":"Streaming fallback not supported"}'
                        break
                    }

                    $lmBody = Convert-OllamaGenerateToLMStudioChat -OllamaRequest $reqObj
                    $lm = Try-LMStudio -Method 'POST' -Path '/v1/chat/completions' -Body $lmBody
                    if (-not $lm) {
                        throw 'LM Studio did not respond'
                    }

                    $json = Convert-LMStudioChatToOllamaResponse -LmJson $lm.Content -Model $reqObj.model -GenerateMode
                    Write-JsonResponse -Response $response -StatusCode 200 -Json $json
                    break
                }
                '/api/chat' {
                    if ($method -ne 'POST') {
                        Write-JsonResponse -Response $response -StatusCode 405 -Json '{"error":"Method not allowed"}'
                        break
                    }

                    $bodyText = Read-RequestBody -Request $request

                    if ($Prefer -eq 'Ollama') {
                        $ollama = Try-Ollama -Method 'POST' -Path '/api/chat' -Body $bodyText
                        if ($ollama) {
                            Write-JsonResponse -Response $response -StatusCode 200 -Json $ollama.Content
                            break
                        }
                    }

                    $reqObj = $bodyText | ConvertFrom-Json
                    if ($reqObj.stream -eq $true) {
                        Write-JsonResponse -Response $response -StatusCode 400 -Json '{"error":"Streaming fallback not supported"}'
                        break
                    }

                    $lmBody = Convert-OllamaChatToLMStudioChat -OllamaRequest $reqObj
                    $lm = Try-LMStudio -Method 'POST' -Path '/v1/chat/completions' -Body $lmBody
                    if (-not $lm) {
                        throw 'LM Studio did not respond'
                    }

                    $json = Convert-LMStudioChatToOllamaResponse -LmJson $lm.Content -Model $reqObj.model
                    Write-JsonResponse -Response $response -StatusCode 200 -Json $json
                    break
                }
                '/api/version' {
                    $payload = @{ version = 'router-1.0' } | ConvertTo-Json
                    Write-JsonResponse -Response $response -StatusCode 200 -Json $payload
                    break
                }
                default {
                    Write-JsonResponse -Response $response -StatusCode 404 -Json '{"error":"Not found"}'
                    break
                }
            }
        } catch {
            $msg = $_.Exception.Message.Replace('"', "'")
            Write-JsonResponse -Response $response -StatusCode 500 -Json '{\'error\":\"$msg\"}"
        }
    }
} finally {
    $listener.Stop()
    $listener.Close()
}
