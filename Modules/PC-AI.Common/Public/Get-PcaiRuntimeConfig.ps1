#Requires -Version 5.1

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

    if ($env:PCAI_ROOT -and (Test-Path $env:PCAI_ROOT)) {
        try {
            return (Resolve-Path -Path $env:PCAI_ROOT -ErrorAction Stop).Path
        } catch {
            return $env:PCAI_ROOT
        }
    }

    $searchRoot = if ($StartPath) { $StartPath } elseif ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).ProviderPath }
    if (Test-Path $searchRoot -PathType Leaf) {
        $searchRoot = Split-Path -Parent $searchRoot
    }

    $resolvedRoot = [Pcai.Common.RuntimeConfigBridge]::FindRepoRoot($searchRoot)
    if (-not [string]::IsNullOrWhiteSpace($resolvedRoot)) {
        return $resolvedRoot
    }

    try {
        return (Resolve-Path -Path $searchRoot -ErrorAction Stop).Path
    } catch {
        return $searchRoot
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

    $runtime = [ordered]@{
        ProjectRoot = $resolvedRoot
        ConfigPath = $resolvedConfigPath
        ToolsPath = $resolvedToolsPath
        Exists = $false

        PcaiInferenceUrl = 'http://127.0.0.1:8080'
        PcaiInferenceModel = 'pcai-inference'
        PcaiInferenceTimeoutMs = 120000

        FunctionGemmaUrl = 'http://127.0.0.1:8000'
        FunctionGemmaModel = 'functiongemma-270m-it'

        RouterBaseUrl = 'http://127.0.0.1:8000'
        RouterModel = 'functiongemma-270m-it'

        OllamaBaseUrl = 'http://127.0.0.1:11434'
        OllamaModel = 'llama3.2'
        LMStudioApiUrl = 'http://127.0.0.1:1234'

        vLLMBaseUrl = 'http://127.0.0.1:8001'
        vLLMModel = 'functiongemma-270m-it'

        FallbackOrder = @('pcai-inference')
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

            $ollamaUrl = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('providers', 'ollama', 'baseUrl'))
            if (-not [string]::IsNullOrWhiteSpace($ollamaUrl)) { $runtime.OllamaBaseUrl = $ollamaUrl }

            $ollamaModel = [Pcai.Common.RuntimeConfigBridge]::TryGetString($json, @('providers', 'ollama', 'defaultModel'))
            if (-not [string]::IsNullOrWhiteSpace($ollamaModel)) { $runtime.OllamaModel = $ollamaModel }

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

            $dllPaths = [Pcai.Common.RuntimeConfigBridge]::TryGetStringArray($json, @('nativeInference', 'dllSearchPaths'))
            foreach ($dllPath in $dllPaths) {
                if ([string]::IsNullOrWhiteSpace($dllPath)) { continue }
                $expanded = [Environment]::ExpandEnvironmentVariables([string]$dllPath)
                $normalizedPath = if ([System.IO.Path]::IsPathRooted($expanded)) {
                    $expanded
                } else {
                    Join-Path $resolvedRoot $expanded
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

    return [PSCustomObject]$runtime
}
