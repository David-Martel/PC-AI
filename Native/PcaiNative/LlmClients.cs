using System.Text;
using System.Text.Json;

namespace PcaiNative;

/// <summary>
/// Native high-performance LLM orchestrator.
/// Bypasses PowerShell HTTP and JSON bottlenecks.
/// </summary>
public sealed class PcaiLlmClient : IDisposable
{
    private readonly HttpClient _http;
    private readonly string _baseUrl;

    public PcaiLlmClient(string baseUrl = "")
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _http = new HttpClient();
    }

    /// <summary>
    /// Synchronously sends a completion request to pcai-inference and returns the parsed response.
    /// Optimized for low-latency structural ingestion.
    /// </summary>
    public string? Chat(string model, string prompt, float temperature = 0.3f)
    {
        var request = new
        {
            model = model,
            prompt = prompt,
            stream = false,
            temperature = temperature
        };

        var json = JsonSerializer.Serialize(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        try
        {
            var response = _http.PostAsync($"{_baseUrl}/v1/completions", content).GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();
            return response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            return $"{{\"error\": \"LLM call failed: {ex.Message}\"}}";
        }
    }

    public void Dispose() => _http.Dispose();
}

/// <summary>
/// OpenAI-compatible chat client (vLLM/LM Studio) for tool-calling workflows.
/// </summary>
public sealed class PcaiOpenAiClient : IDisposable
{
    private readonly HttpClient _http;
    private readonly string _baseUrl;

    public PcaiOpenAiClient(string baseUrl)
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _http = new HttpClient();
    }

    /// <summary>
    /// Sends a raw JSON payload to /v1/chat/completions and returns the raw JSON response.
    /// </summary>
    public string ChatCompletionsRaw(string payloadJson)
    {
        var content = new StringContent(payloadJson, Encoding.UTF8, "application/json");
        try
        {
            var response = _http.PostAsync($"{_baseUrl}/v1/chat/completions", content).GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();
            return response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            return $"{{\"error\": \"OpenAI chat call failed: {ex.Message}\"}}";
        }
    }

    public void Dispose() => _http.Dispose();
}

/// <summary>
/// Centralized ReAct engine for automated tool execution.
/// Unifies logic for CLI (automated) and TUI (agent mode).
/// </summary>
public sealed class ReActOrchestrator
{
    private readonly PcaiOpenAiClient _client;
    private readonly ToolExecutor _executor;
    private readonly string _model;

    public event Action<string>? OnThought;
    public event Action<string, string>? OnToolCall;
    public event Action<string, string>? OnToolResult;
    public event Action<string>? OnFinalAnswer;
    public event Action<string>? OnError;

    public ReActOrchestrator(PcaiOpenAiClient client, ToolExecutor executor, string model)
    {
        _client = client;
        _executor = executor;
        _model = model;
    }

    /// <summary>
    /// Executes the ReAct loop for a given prompt.
    /// </summary>
    public async Task RunAsync(string prompt, int maxSteps = 5)
    {
        var tools = _executor.Tools?.AsArray().Select(t => t?.DeepClone()).ToList();

        var messages = new List<object>
        {
            new { role = "system", content = "You are a PC diagnosis agent. Use the provided tools to help the user. If you need more information, call a tool. If you have enough information, provide a 'Final Answer:'." },
            new { role = "user", content = prompt }
        };

        for (int step = 0; step < maxSteps; step++)
        {
            var payload = new
            {
                model = _model,
                messages = messages,
                tools = tools,
                tool_choice = "auto",
                temperature = 0.1
            };

            var responseJson = _client.ChatCompletionsRaw(JsonSerializer.Serialize(payload));
            if (responseJson.Contains("\"error\""))
            {
                OnError?.Invoke(responseJson);
                return;
            }

            using var doc = JsonDocument.Parse(responseJson);
            var choice = doc.RootElement.GetProperty("choices")[0];
            var message = choice.GetProperty("message");
            var content = message.GetProperty("content").GetString() ?? "";

            if (!string.IsNullOrEmpty(content))
            {
                if (content.Contains("Final Answer:"))
                {
                    OnFinalAnswer?.Invoke(content);
                    return;
                }
                OnThought?.Invoke(content);
            }

            // Look for tool calls in the message (standard OpenAI tool_calls)
            if (message.TryGetProperty("tool_calls", out var toolCalls))
            {
                var assistantMessage = new { role = "assistant", content = content, tool_calls = toolCalls };
                messages.Add(assistantMessage);

                foreach (var tc in toolCalls.EnumerateArray())
                {
                    var func = tc.GetProperty("function");
                    var name = func.GetProperty("name").GetString()!;
                    var argsRaw = func.GetProperty("arguments").GetString()!;

                    OnToolCall?.Invoke(name, argsRaw);

                    var result = await _executor.ExecuteToolAsync(name, JsonDocument.Parse(argsRaw).RootElement);
                    OnToolResult?.Invoke(name, result);

                    messages.Add(new
                    {
                        role = "tool",
                        tool_call_id = tc.GetProperty("id").GetString(),
                        name = name,
                        content = result
                    });
                }
            }
            else if (!string.IsNullOrEmpty(content))
            {
                // If the model spoke but didn't call a tool or give a Final Answer, treat as part of thought
                messages.Add(new { role = "assistant", content = content });

                if (content.Contains("Final Answer:")) return;
            }
            else
            {
                OnError?.Invoke("LLM returned empty response and no tool calls.");
                return;
            }
        }

        OnError?.Invoke("ReAct loop exceeded maximum steps.");
    }
}
