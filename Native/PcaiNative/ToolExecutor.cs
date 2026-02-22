using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace PcaiNative;

/// <summary>
/// Executes PC-AI tools defined in pcai-tools.json.
/// Handles mapping from LLM function calls to PowerShell cmdlets or native modules.
/// Includes safety interlocks for destructive actions.
/// </summary>
public sealed class ToolExecutor
{
    private const int DefaultTimeoutMs = 30000;
    private const int DefaultMaxOutputChars = 20000;
    private const int DefaultRetryCount = 1;
    private const int DefaultRetryDelayMs = 500;

    private readonly string _toolsPath;
    private readonly JsonNode? _toolsConfig;
    private readonly PowerShellHost? _psHost;
    private readonly SafetyInterlock? _interlock;

    public JsonNode? Tools => _toolsConfig?["tools"];

    public ToolExecutor(string toolsPath, PowerShellHost? psHost = null, SafetyInterlock? interlock = null)
    {
        _toolsPath = toolsPath;
        _psHost = psHost;
        _interlock = interlock;

        if (File.Exists(toolsPath))
        {
            try
            {
                var json = File.ReadAllText(toolsPath);
                _toolsConfig = JsonNode.Parse(json);
            }
            catch
            {
                // Soft failure - caller handles null config
            }
        }
    }

    /// <summary>
    /// Executes a tool call by name and arguments.
    /// </summary>
    public async Task<string> ExecuteToolAsync(string name, JsonElement arguments)
    {
        if (_toolsConfig?["tools"] is not JsonArray tools)
        {
            return BuildEnvelope(name, ToolExecutionResult.Fail("Tools configuration not loaded or invalid."));
        }

        var toolDef = tools.FirstOrDefault(t => t?["function"]?["name"]?.ToString() == name);
        if (toolDef == null)
        {
            return BuildEnvelope(name, ToolExecutionResult.Fail($"Unknown tool: {name}"));
        }

        var mapping = toolDef["pcai_mapping"];
        if (mapping == null)
        {
            return BuildEnvelope(name, ToolExecutionResult.Fail($"No mapping found for tool: {name}"));
        }

        var limits = GetToolLimits(toolDef);

        // Safety INTERLOCK Check
        if (_interlock != null)
        {
            var isDestructive = mapping["is_destructive"]?.GetValue<bool>() ?? false;
            var description = toolDef["function"]?["description"]?.ToString() ?? "No description available.";

            if (!await _interlock.VerifyActionAsync(name, description, isDestructive))
            {
                return BuildEnvelope(name, ToolExecutionResult.Fail("User rejected destructive tool execution."));
            }
        }

        // Special handling for native-first tools to avoid PS overhead
        if (name == "pcai_get_usb_diagnostics" || name == "pcai_get_usb_list")
        {
            var native = PcaiCore.GetUsbDeepDiagnostics();
            return BuildEnvelope(name, native != null
                ? ToolExecutionResult.Ok(native, exitCode: 0, durationMs: 0)
                : ToolExecutionResult.Fail("Native USB diagnostics failed."));
        }
        if (name == "pcai_get_network_info")
        {
            var native = PcaiCore.GetNetworkThroughput();
            return BuildEnvelope(name, native != null
                ? ToolExecutionResult.Ok(native, exitCode: 0, durationMs: 0)
                : ToolExecutionResult.Fail("Native network diagnostics failed."));
        }

        // Standard PowerShell mapping
        var cmdlet = mapping["cmdlet"]?.ToString();
        var module = mapping["module"]?.ToString();

        if (string.IsNullOrEmpty(cmdlet))
        {
            return BuildEnvelope(name, ToolExecutionResult.Fail($"No cmdlet mapped for tool: {name}"));
        }

        var lastResult = ToolExecutionResult.Fail("Tool execution failed.");
        for (var attempt = 0; attempt <= limits.RetryCount; attempt++)
        {
            lastResult = _psHost != null
                ? await RunHostCommandAsync(cmdlet, module, mapping["params"], arguments, limits)
                : await RunPowerShellCommandAsync(cmdlet, module, mapping["params"], arguments, limits);

            if (lastResult.Success)
            {
                return BuildEnvelope(name, lastResult, limits);
            }

            if (!IsRetryable(lastResult.Error))
            {
                break;
            }

            if (attempt < limits.RetryCount)
            {
                await Task.Delay(limits.RetryDelayMs);
            }
        }

        return BuildEnvelope(name, lastResult, limits);
    }

    private async Task<ToolExecutionResult> RunHostCommandAsync(string cmdlet, string? module, JsonNode? paramMappings, JsonElement arguments, ToolExecutionLimits limits)
    {
        if (!IsSafeCommandToken(cmdlet))
        {
            return ToolExecutionResult.Fail($"Unsafe cmdlet token: {cmdlet}");
        }

        var sb = new StringBuilder();
        if (!string.IsNullOrEmpty(module))
        {
            sb.Append($"Import-Module '{EscapeSingleQuotedLiteral(module)}' -ErrorAction SilentlyContinue; ");
        }

        sb.Append(cmdlet);
        AppendParameters(sb, paramMappings, arguments);
        sb.Append(" | ConvertTo-Json -Depth 6 -Compress");

        var sw = Stopwatch.StartNew();
        var output = await _psHost!.ExecuteAsync(sb.ToString(), TimeSpan.FromMilliseconds(limits.TimeoutMs));
        sw.Stop();

        if (IsErrorPayload(output))
        {
            return ToolExecutionResult.Fail(output, exitCode: 1, durationMs: sw.ElapsedMilliseconds);
        }

        return ToolExecutionResult.Ok(output, exitCode: 0, durationMs: sw.ElapsedMilliseconds);
    }

    private void AppendParameters(StringBuilder sb, JsonNode? paramMappings, JsonElement arguments)
    {
        if (paramMappings is JsonObject pm)
        {
            foreach (var kvp in pm)
            {
                var pName = kvp.Key;
                if (!IsSafeParameterName(pName))
                {
                    continue;
                }

                var pValue = kvp.Value?.ToString();

                if (pValue != null && pValue.StartsWith("$"))
                {
                    var argName = pValue.TrimStart('$');
                    if (arguments.TryGetProperty(argName, out var argValue))
                    {
                        var valStr = argValue.ValueKind == JsonValueKind.String
                            ? argValue.GetString()
                            : argValue.GetRawText();

                        sb.Append($" -{pName} '{EscapeSingleQuotedLiteral(valStr ?? string.Empty)}'");
                    }
                }
                else if (pValue != null)
                {
                    sb.Append($" -{pName} '{EscapeSingleQuotedLiteral(pValue)}'");
                }
            }
        }
    }

    private async Task<ToolExecutionResult> RunPowerShellCommandAsync(string cmdlet, string? module, JsonNode? paramMappings, JsonElement arguments, ToolExecutionLimits limits)
    {
        if (!IsSafeCommandToken(cmdlet))
        {
            return ToolExecutionResult.Fail($"Unsafe cmdlet token: {cmdlet}");
        }

        var sb = new StringBuilder();
        if (!string.IsNullOrEmpty(module))
        {
            sb.Append($"Import-Module '{EscapeSingleQuotedLiteral(module)}' -ErrorAction SilentlyContinue; ");
        }

        sb.Append(cmdlet);
        AppendParameters(sb, paramMappings, arguments);

        // Add redirection to ensure clean JSON output
        sb.Append(" | ConvertTo-Json -Depth 6 -Compress");

        var sw = Stopwatch.StartNew();
        try
        {
            var encodedCommand = Convert.ToBase64String(Encoding.Unicode.GetBytes(sb.ToString()));
            var startInfo = new ProcessStartInfo
            {
                FileName = "pwsh",
                Arguments = $"-NoProfile -NonInteractive -EncodedCommand {encodedCommand}",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = Process.Start(startInfo);
            if (process == null)
            {
                return ToolExecutionResult.Fail("Failed to start PowerShell.", exitCode: 1, durationMs: sw.ElapsedMilliseconds);
            }

            var outputTask = process.StandardOutput.ReadToEndAsync();
            var errorTask = process.StandardError.ReadToEndAsync();
            var waitTask = process.WaitForExitAsync();

            var completed = await Task.WhenAny(waitTask, Task.Delay(limits.TimeoutMs));
            if (completed != waitTask)
            {
                try { process.Kill(entireProcessTree: true); } catch { }
                return ToolExecutionResult.Fail("PowerShell execution timed out.", exitCode: 124, durationMs: sw.ElapsedMilliseconds, timedOut: true);
            }

            var output = await outputTask;
            var error = await errorTask;

            if (process.ExitCode != 0 && string.IsNullOrWhiteSpace(output))
            {
                return ToolExecutionResult.Fail($"PowerShell exited with code {process.ExitCode}: {error}", exitCode: process.ExitCode, durationMs: sw.ElapsedMilliseconds);
            }

            return ToolExecutionResult.Ok(string.IsNullOrWhiteSpace(output) ? "{}" : output, process.ExitCode, sw.ElapsedMilliseconds);
        }
        catch (Exception ex)
        {
            return ToolExecutionResult.Fail($"Execution failed: {ex.Message}", exitCode: 1, durationMs: sw.ElapsedMilliseconds);
        }
    }

    private static bool IsErrorPayload(string payload)
    {
        if (string.IsNullOrWhiteSpace(payload))
        {
            return true;
        }
        return payload.Contains("\"error\"", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsRetryable(string? error)
    {
        if (string.IsNullOrWhiteSpace(error))
        {
            return false;
        }

        if (error.Contains("User rejected", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        if (error.Contains("Unknown tool", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        return true;
    }

    private static ToolExecutionLimits GetToolLimits(JsonNode? toolDef)
    {
        var limitsNode = toolDef?["pcai_limits"] ?? toolDef?["limits"];
        var timeoutMs = limitsNode?["timeout_ms"]?.GetValue<int>() ?? DefaultTimeoutMs;
        var maxOutput = limitsNode?["max_output_chars"]?.GetValue<int>() ?? DefaultMaxOutputChars;
        var retries = limitsNode?["retries"]?.GetValue<int>() ?? DefaultRetryCount;
        var retryDelay = limitsNode?["retry_delay_ms"]?.GetValue<int>() ?? DefaultRetryDelayMs;
        return new ToolExecutionLimits(timeoutMs, maxOutput, retries, retryDelay);
    }

    private static string BuildEnvelope(string toolName, ToolExecutionResult result, ToolExecutionLimits? limits = null)
    {
        var output = result.Output ?? string.Empty;
        var warnings = new JsonArray();

        if (limits.HasValue && output.Length > limits.Value.MaxOutputChars)
        {
            output = output[..limits.Value.MaxOutputChars];
            warnings.Add("output_truncated");
        }

        JsonNode? parsed = null;
        try
        {
            parsed = JsonNode.Parse(output);
        }
        catch
        {
            parsed = output;
        }

        var envelope = new JsonObject
        {
            ["tool"] = toolName,
            ["success"] = result.Success,
            ["exit_code"] = result.ExitCode,
            ["duration_ms"] = result.DurationMs,
            ["warnings"] = warnings,
            ["error"] = result.Success ? null : result.Error,
            ["result"] = parsed
        };

        if (result.TimedOut)
        {
            envelope["timed_out"] = true;
        }

        return envelope.ToJsonString(new JsonSerializerOptions { WriteIndented = false });
    }

    private static string EscapeSingleQuotedLiteral(string value)
    {
        return value.Replace("'", "''");
    }

    private static bool IsSafeCommandToken(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return false;
        }

        foreach (var ch in value)
        {
            if (!(char.IsLetterOrDigit(ch) || ch == '-' || ch == '_' || ch == '.'))
            {
                return false;
            }
        }

        return true;
    }

    private static bool IsSafeParameterName(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return false;
        }

        foreach (var ch in value)
        {
            if (!(char.IsLetterOrDigit(ch) || ch == '_'))
            {
                return false;
            }
        }

        return true;
    }

    private readonly record struct ToolExecutionLimits(int TimeoutMs, int MaxOutputChars, int RetryCount, int RetryDelayMs);

    private readonly record struct ToolExecutionResult(string? Output, bool Success, int ExitCode, string? Error, bool TimedOut, long DurationMs)
    {
        public static ToolExecutionResult Ok(string output, int exitCode, long durationMs) =>
            new(output, true, exitCode, null, false, durationMs);

        public static ToolExecutionResult Fail(string error, int exitCode = 1, long durationMs = 0, bool timedOut = false) =>
            new(null, false, exitCode, error, timedOut, durationMs);
    }
}
