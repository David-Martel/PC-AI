#Requires -PSEdition Core

$sharedCacheHelper = Join-Path $PSScriptRoot 'Get-PcaiSharedCache.ps1'
if ((-not (Get-Command Get-PcaiSharedCacheEntry -ErrorAction SilentlyContinue)) -and (Test-Path $sharedCacheHelper)) {
    . $sharedCacheHelper
}

if (-not ('Pcai.Common.RuntimeConfigBridge' -as [type])) {
    Add-Type -Language CSharp -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace Pcai.Common {
    public static class RuntimeConfigBridge {
        public static string FindRepoRoot(string startPath) {
            if (string.IsNullOrWhiteSpace(startPath)) {
                return null;
            }

            string cursor;
            try {
                cursor = Path.GetFullPath(startPath);
            } catch {
                return null;
            }

            if (File.Exists(cursor)) {
                cursor = Path.GetDirectoryName(cursor);
            }

            while (!string.IsNullOrWhiteSpace(cursor)) {
                if (File.Exists(Path.Combine(cursor, "PC-AI.ps1")) ||
                    File.Exists(Path.Combine(cursor, "AGENTS.md")) ||
                    Directory.Exists(Path.Combine(cursor, ".git"))) {
                    return cursor;
                }

                var parent = Directory.GetParent(cursor);
                if (parent == null || string.Equals(parent.FullName, cursor, StringComparison.OrdinalIgnoreCase)) {
                    break;
                }
                cursor = parent.FullName;
            }

            return null;
        }

        private static bool TryNavigate(JsonElement root, string[] segments, out JsonElement value) {
            value = root;
            if (segments == null || segments.Length == 0) {
                return true;
            }

            foreach (var segment in segments) {
                if (value.ValueKind != JsonValueKind.Object) {
                    return false;
                }
                JsonElement next;
                if (!value.TryGetProperty(segment, out next)) {
                    return false;
                }
                value = next;
            }

            return true;
        }

        public static string TryGetString(string json, string[] pathSegments) {
            if (string.IsNullOrWhiteSpace(json)) {
                return null;
            }

            try {
                using (var doc = JsonDocument.Parse(json)) {
                    JsonElement value;
                    if (!TryNavigate(doc.RootElement, pathSegments, out value)) {
                        return null;
                    }
                    if (value.ValueKind == JsonValueKind.String) {
                        return value.GetString();
                    }
                    if (value.ValueKind == JsonValueKind.Number || value.ValueKind == JsonValueKind.True || value.ValueKind == JsonValueKind.False) {
                        return value.ToString();
                    }
                }
            } catch {
                return null;
            }

            return null;
        }

        public static int? TryGetInt(string json, string[] pathSegments) {
            if (string.IsNullOrWhiteSpace(json)) {
                return null;
            }

            try {
                using (var doc = JsonDocument.Parse(json)) {
                    JsonElement value;
                    if (!TryNavigate(doc.RootElement, pathSegments, out value)) {
                        return null;
                    }
                    if (value.ValueKind == JsonValueKind.Number) {
                        int intValue;
                        if (value.TryGetInt32(out intValue)) {
                            return intValue;
                        }
                    }
                    if (value.ValueKind == JsonValueKind.String) {
                        int parsed;
                        if (Int32.TryParse(value.GetString(), out parsed)) {
                            return parsed;
                        }
                    }
                }
            } catch {
                return null;
            }

            return null;
        }

        public static string[] TryGetStringArray(string json, string[] pathSegments) {
            if (string.IsNullOrWhiteSpace(json)) {
                return Array.Empty<string>();
            }

            try {
                using (var doc = JsonDocument.Parse(json)) {
                    JsonElement value;
                    if (!TryNavigate(doc.RootElement, pathSegments, out value)) {
                        return Array.Empty<string>();
                    }
                    if (value.ValueKind != JsonValueKind.Array) {
                        return Array.Empty<string>();
                    }

                    var output = new List<string>();
                    foreach (var item in value.EnumerateArray()) {
                        if (item.ValueKind == JsonValueKind.String) {
                            var str = item.GetString();
                            if (!String.IsNullOrWhiteSpace(str)) {
                                output.Add(str);
                            }
                        }
                    }
                    return output.ToArray();
                }
            } catch {
                return Array.Empty<string>();
            }
        }
    }
}
"@
}

function Resolve-PcaiRepoRoot {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [string]$StartPath
    )

    $searchRoot = if ($StartPath) { $StartPath } elseif ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).ProviderPath }
    if (Test-Path $searchRoot -PathType Leaf) {
        $searchRoot = Split-Path -Parent $searchRoot
    }

    $resolvedSearchRoot = try {
        (Resolve-Path -Path $searchRoot -ErrorAction Stop).Path
    } catch {
        $searchRoot
    }

    $cacheKey = $resolvedSearchRoot
    if (Get-Command Get-PcaiSharedCacheEntry -ErrorAction SilentlyContinue) {
        $cachedRoot = Get-PcaiSharedCacheEntry -Namespace 'pcai-common' -Key "repo-root::$cacheKey" -TtlSeconds 300
        if ($cachedRoot) {
            return [string]$cachedRoot
        }
    }

    $resolvedRoot = [Pcai.Common.RuntimeConfigBridge]::FindRepoRoot($resolvedSearchRoot)
    if (-not [string]::IsNullOrWhiteSpace($resolvedRoot)) {
        if (Get-Command Set-PcaiSharedCacheEntry -ErrorAction SilentlyContinue) {
            Set-PcaiSharedCacheEntry -Namespace 'pcai-common' -Key "repo-root::$cacheKey" -Value $resolvedRoot | Out-Null
        }
        return $resolvedRoot
    }

    $commonCandidates = @(
        'C:\codedev\PC_AI'
    ) | Select-Object -Unique

    foreach ($candidate in $commonCandidates) {
        if (-not $candidate -or -not (Test-Path $candidate)) {
            continue
        }

        $resolvedCandidate = [Pcai.Common.RuntimeConfigBridge]::FindRepoRoot($candidate)
        if (-not [string]::IsNullOrWhiteSpace($resolvedCandidate)) {
            if (Get-Command Set-PcaiSharedCacheEntry -ErrorAction SilentlyContinue) {
                Set-PcaiSharedCacheEntry -Namespace 'pcai-common' -Key "repo-root::$cacheKey" -Value $resolvedCandidate | Out-Null
            }
            return $resolvedCandidate
        }
    }

    try {
        $fallbackRoot = (Resolve-Path -Path $resolvedSearchRoot -ErrorAction Stop).Path
        if (Get-Command Set-PcaiSharedCacheEntry -ErrorAction SilentlyContinue) {
            Set-PcaiSharedCacheEntry -Namespace 'pcai-common' -Key "repo-root::$cacheKey" -Value $fallbackRoot | Out-Null
        }
        return $fallbackRoot
    } catch {
        if (Get-Command Set-PcaiSharedCacheEntry -ErrorAction SilentlyContinue) {
            Set-PcaiSharedCacheEntry -Namespace 'pcai-common' -Key "repo-root::$cacheKey" -Value $resolvedSearchRoot | Out-Null
        }
        return $resolvedSearchRoot
    }
}

function Get-PcaiRuntimeConfig {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [string]$ProjectRoot,
        [string]$ConfigPath,
        [string]$ToolsPath
    )

    $resolvedRoot = if ($ProjectRoot) {
        Resolve-PcaiRepoRoot -StartPath $ProjectRoot
    } else {
        Resolve-PcaiRepoRoot -StartPath $PSScriptRoot
    }

    $resolvedConfigPath = if ($ConfigPath) {
        if ([System.IO.Path]::IsPathRooted($ConfigPath)) { $ConfigPath } else { Join-Path $resolvedRoot $ConfigPath }
    } else {
        Join-Path $resolvedRoot 'Config\llm-config.json'
    }

    $resolvedToolsPath = if ($ToolsPath) {
        if ([System.IO.Path]::IsPathRooted($ToolsPath)) { $ToolsPath } else { Join-Path $resolvedRoot $ToolsPath }
    } else {
        Join-Path $resolvedRoot 'Config\pcai-tools.json'
    }

    $runtimeCacheKey = '{0}|{1}|{2}' -f $resolvedRoot, $resolvedConfigPath, $resolvedToolsPath
    $runtimeConfigStamp = if (Get-Command Get-PcaiDependencyStamp -ErrorAction SilentlyContinue) {
        Get-PcaiDependencyStamp -InputObject @($resolvedConfigPath)
    } else {
        $null
    }
    if (Get-Command Get-PcaiSharedCacheEntry -ErrorAction SilentlyContinue) {
        $cachedRuntime = Get-PcaiSharedCacheEntry -Namespace 'pcai-common' -Key "runtime-config::$runtimeCacheKey" -DependencyStamp $runtimeConfigStamp
        if ($cachedRuntime) {
            return [PSCustomObject]$cachedRuntime
        }
    }

    $runtime = [ordered]@{
        ProjectRoot = $resolvedRoot
        ConfigPath = $resolvedConfigPath
        ToolsPath = $resolvedToolsPath
        Exists = $false

        PcaiInferenceUrl = 'http://127.0.0.1:18080'
        PcaiInferenceModel = 'llama.cpp'
        PcaiInferenceTimeoutMs = 120000

        OllamaBaseUrl = 'http://127.0.0.1:11434'
        OllamaModel = 'qwen2.5-coder:3b'
        OllamaToolModel = ''
        OllamaSummaryModel = ''
        OllamaTimeoutMs = 90000
        OllamaCliSearchPaths = @()
        OllamaToolInvokerPath = (Join-Path $resolvedRoot 'Tools\Invoke-PcaiMappedTool.ps1')

        FunctionGemmaUrl = 'http://127.0.0.1:8000'
        FunctionGemmaModel = 'functiongemma-270m-it'

        RouterBaseUrl = 'http://127.0.0.1:8000'
        RouterModel = 'functiongemma-270m-it'
        LMStudioApiUrl = 'http://127.0.0.1:1234'

        vLLMBaseUrl = 'http://127.0.0.1:8001'
        vLLMModel = 'functiongemma-270m-it'

        FallbackOrder = @('ollama', 'pcai-inference')
        NativeDllSearchPaths = @()
    }

    if (Test-Path $resolvedConfigPath) {
        try {
            $json = [System.IO.File]::ReadAllText($resolvedConfigPath)
            $runtime.Exists = $true

            $pcaiUrl = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('providers', 'pcai-inference', 'baseUrl'))
            if (-not [string]::IsNullOrWhiteSpace($pcaiUrl)) { $runtime.PcaiInferenceUrl = $pcaiUrl }

            $pcaiModel = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('providers', 'pcai-inference', 'defaultModel'))
            if (-not [string]::IsNullOrWhiteSpace($pcaiModel)) { $runtime.PcaiInferenceModel = $pcaiModel }

            $pcaiTimeout = [Pcai.Common.RuntimeConfigBridge]::TryGetInt($json, @('providers', 'pcai-inference', 'timeout'))
            if ($pcaiTimeout -and $pcaiTimeout -gt 0) { $runtime.PcaiInferenceTimeoutMs = [int]$pcaiTimeout }

            $fgUrl = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('providers', 'functiongemma', 'baseUrl'))
            if (-not [string]::IsNullOrWhiteSpace($fgUrl)) { $runtime.FunctionGemmaUrl = $fgUrl }

            $fgModel = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('providers', 'functiongemma', 'defaultModel'))
            if (-not [string]::IsNullOrWhiteSpace($fgModel)) { $runtime.FunctionGemmaModel = $fgModel }

            $routerUrl = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('router', 'baseUrl'))
            if (-not [string]::IsNullOrWhiteSpace($routerUrl)) { $runtime.RouterBaseUrl = $routerUrl } else { $runtime.RouterBaseUrl = $runtime.FunctionGemmaUrl }

            $routerModel = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('router', 'model'))
            if (-not [string]::IsNullOrWhiteSpace($routerModel)) { $runtime.RouterModel = $routerModel } else { $runtime.RouterModel = $runtime.FunctionGemmaModel }

            $ollamaUrl = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('ollama', 'base_url'))
            if (-not [string]::IsNullOrWhiteSpace($ollamaUrl)) { $runtime.OllamaBaseUrl = $ollamaUrl }
            if ([string]::IsNullOrWhiteSpace($ollamaUrl)) {
                $ollamaUrl = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('providers', 'ollama', 'baseUrl'))
            }
            if (-not [string]::IsNullOrWhiteSpace($ollamaUrl)) { $runtime.OllamaBaseUrl = $ollamaUrl }

            $ollamaModel = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('ollama', 'model'))
            if (-not [string]::IsNullOrWhiteSpace($ollamaModel)) { $runtime.OllamaModel = $ollamaModel }
            if ([string]::IsNullOrWhiteSpace($ollamaModel)) {
                $ollamaModel = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('providers', 'ollama', 'defaultModel'))
            }
            if (-not [string]::IsNullOrWhiteSpace($ollamaModel)) { $runtime.OllamaModel = $ollamaModel }

            $ollamaToolModel = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('ollama', 'tool_model'))
            if (-not [string]::IsNullOrWhiteSpace($ollamaToolModel)) { $runtime.OllamaToolModel = $ollamaToolModel }

            $ollamaSummaryModel = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('ollama', 'summary_model'))
            if (-not [string]::IsNullOrWhiteSpace($ollamaSummaryModel)) { $runtime.OllamaSummaryModel = $ollamaSummaryModel }

            $ollamaTimeout = [Pcai.Common.RuntimeConfigBridge]::TryGetInt($json, @('ollama', 'timeout_ms'))
            if ($ollamaTimeout -and $ollamaTimeout -gt 0) { $runtime.OllamaTimeoutMs = [int]$ollamaTimeout }

            $lmstudioUrl = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('providers', 'lmstudio', 'baseUrl'))
            if (-not [string]::IsNullOrWhiteSpace($lmstudioUrl)) { $runtime.LMStudioApiUrl = $lmstudioUrl }

            $vllmUrl = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('providers', 'vllm', 'baseUrl'))
            if (-not [string]::IsNullOrWhiteSpace($vllmUrl)) { $runtime.vLLMBaseUrl = $vllmUrl } else { $runtime.vLLMBaseUrl = $runtime.RouterBaseUrl }

            $vllmModel = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('providers', 'vllm', 'defaultModel'))
            if (-not [string]::IsNullOrWhiteSpace($vllmModel)) { $runtime.vLLMModel = $vllmModel } else { $runtime.vLLMModel = $runtime.RouterModel }

            $fallback = [Pcai.Common.RuntimeConfigBridge]::TryGetStringArray($json, @('fallbackOrder'))
            if ($fallback -and $fallback.Length -gt 0) {
                $runtime.FallbackOrder = @($fallback)
            }

            $ollamaCliSearchPaths = [Pcai.Common.RuntimeConfigBridge]::TryGetStringArray($json, @('ollama', 'cliSearchPaths'))
            foreach ($cliPath in $ollamaCliSearchPaths) {
                if ([string]::IsNullOrWhiteSpace($cliPath)) { continue }
                $normalizedPath = if ([System.IO.Path]::IsPathRooted($cliPath)) {
                    $cliPath
                } else {
                    Join-Path $resolvedRoot $cliPath
                }
                $runtime.OllamaCliSearchPaths += $normalizedPath
            }

            $toolInvokerPath = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('ollama', 'toolInvokerPath'))
            if (-not [string]::IsNullOrWhiteSpace($toolInvokerPath)) {
                $runtime.OllamaToolInvokerPath = if ([System.IO.Path]::IsPathRooted($toolInvokerPath)) { $toolInvokerPath } else { Join-Path $resolvedRoot $toolInvokerPath }
            }

            $dllPaths = [Pcai.Common.RuntimeConfigBridge]::TryGetStringArray($json, @('nativeInference', 'dllSearchPaths'))
            foreach ($dllPath in $dllPaths) {
                if ([string]::IsNullOrWhiteSpace($dllPath)) { continue }
                $normalizedPath = if ([System.IO.Path]::IsPathRooted([string]$dllPath)) {
                    [string]$dllPath
                } else {
                    Join-Path $resolvedRoot ([string]$dllPath)
                }
                $runtime.NativeDllSearchPaths += $normalizedPath
            }

            $toolsFromConfig = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('router', 'toolsPath'))
            if (-not [string]::IsNullOrWhiteSpace($toolsFromConfig)) {
                $runtime.ToolsPath = if ([System.IO.Path]::IsPathRooted($toolsFromConfig)) { $toolsFromConfig } else { Join-Path $resolvedRoot $toolsFromConfig }
            }
        } catch {
            Write-Verbose "Get-PcaiRuntimeConfig failed to parse ${resolvedConfigPath}: $($_.Exception.Message)"
        }
    }

    if (-not $runtime.NativeDllSearchPaths -or $runtime.NativeDllSearchPaths.Count -eq 0) {
        $runtime.NativeDllSearchPaths = @(
            (Join-Path $resolvedRoot '.local\bin\pcai_inference.dll'),
            (Join-Path $resolvedRoot 'Native\pcai_core\pcai_inference\target\release\pcai_inference.dll'),
            (Join-Path $resolvedRoot '.pcai\build\artifacts\pcai-llamacpp\pcai_inference.dll'),
            (Join-Path $resolvedRoot '.pcai\build\artifacts\pcai-mistralrs\pcai_inference.dll')
        )
    }

    if (-not $runtime.OllamaCliSearchPaths -or $runtime.OllamaCliSearchPaths.Count -eq 0) {
        $runtime.OllamaCliSearchPaths = @(
            (Join-Path $resolvedRoot 'Native\pcai_core\bin\pcai-ollama-rs.exe'),
            (Join-Path $resolvedRoot 'Native\pcai_core\target\release\pcai-ollama-rs.exe'),
            (Join-Path $resolvedRoot 'Native\pcai_core\pcai_ollama_rs\target\release\pcai-ollama-rs.exe')
        )
    }

    if (Get-Command Set-PcaiSharedCacheEntry -ErrorAction SilentlyContinue) {
        Set-PcaiSharedCacheEntry -Namespace 'pcai-common' -Key "runtime-config::$runtimeCacheKey" -Value $runtime -DependencyStamp $runtimeConfigStamp | Out-Null
    }

    return [PSCustomObject]$runtime
}
