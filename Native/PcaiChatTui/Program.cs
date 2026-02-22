#nullable enable
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using PcaiNative;

namespace PcaiChatTui;

/// <summary>Application entry point for the PC_AI interactive chat TUI.</summary>
public static class Program
{
    private const string DefaultBaseUrl = "";
    private const string DefaultModel = "pcai-inference";
    private const int DefaultTimeoutSec = 120;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true
    };

    public static async Task<int> Main(string[] args)
    {
        var options = CliOptions.Parse(args);
        var config = LlmConfig.Load(options.ConfigPath);

        var provider = options.Provider ?? config.DefaultProvider ?? "pcai-inference";
        var providerConfig = config.GetProvider(provider);
        var nativeConfig = config.NativeProvider;

        var preferNative = string.Equals(provider, "pcai-native", StringComparison.OrdinalIgnoreCase) ||
                           string.Equals(providerConfig?.Type, "ffi", StringComparison.OrdinalIgnoreCase);

        var backend = options.Backend;
        var backendExplicit = backend != BackendType.Auto && backend != BackendType.Http;
        if (backend == BackendType.Auto && preferNative && !string.IsNullOrWhiteSpace(nativeConfig?.Backend))
        {
            backend = nativeConfig.Backend!.ToLowerInvariant() switch
            {
                "llamacpp" => BackendType.LlamaCpp,
                "mistralrs" => BackendType.MistralRs,
                "http" => BackendType.Http,
                _ => BackendType.Auto
            };
        }
        if (backend == BackendType.Auto && !preferNative)
        {
            backend = BackendType.Http;
        }

        var baseUrl = NormalizeBaseUrl(options.BaseUrl ?? providerConfig?.BaseUrl ?? DefaultBaseUrl);
        baseUrl = ResolveHvsockEndpoint(baseUrl, options.HvsockConfigPath ?? config.HvsockConfigPath);
        var model = options.Model ?? providerConfig?.DefaultModel ?? DefaultModel;
        var timeoutSec = options.TimeoutSec ?? providerConfig?.TimeoutSec ?? DefaultTimeoutSec;
        var mode = options.Mode ?? "chat"; // Default to chat mode

        var modelPath = options.ModelPath ?? nativeConfig?.ModelPath;
        var gpuLayers = options.GpuLayers;
        if (options.GpuLayers == -1 && nativeConfig?.GpuLayers != null)
        {
            gpuLayers = nativeConfig.GpuLayers.Value;
        }
        var defaultMaxTokens = nativeConfig?.DefaultMaxTokens ?? 2048;
        var defaultTemperature = nativeConfig?.DefaultTemperature ?? 0.7f;

        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(timeoutSec) };

        var systemPrompt = options.SystemPrompt;
        if (string.IsNullOrWhiteSpace(systemPrompt) && !string.IsNullOrWhiteSpace(options.SystemFile))
        {
            if (File.Exists(options.SystemFile))
            {
                systemPrompt = File.ReadAllText(options.SystemFile);
            }
        }

        // If no system prompt provided, load default for mode
        if (string.IsNullOrWhiteSpace(systemPrompt))
        {
            systemPrompt = LoadModePrompt(mode);
        }

        IInferenceBackend? nativeBackend = null;
        var useNative = backend != BackendType.Http;

        if (useNative)
        {
            nativeBackend = await BackendFactory.CreateAsync(backend, baseUrl);

            if (nativeBackend is HttpBackend)
            {
                await nativeBackend.DisposeAsync();
                nativeBackend = null;
                useNative = false;
                backend = BackendType.Http;
            }
            else if (!await nativeBackend.CheckAvailabilityAsync())
            {
                var message = "pcai_inference.dll not available or backend failed to initialize";
                if (backendExplicit || preferNative)
                {
                    Console.WriteLine($"error> {message}");
                    return 1;
                }

                Console.WriteLine($"warning> {message}. Falling back to HTTP.");
                await nativeBackend.DisposeAsync();
                nativeBackend = null;
                useNative = false;
                backend = BackendType.Http;
            }
        }

        if (useNative && string.IsNullOrWhiteSpace(modelPath))
        {
            var message = "Native backend selected but no model path provided. Use --model-path or set providers.pcai-native.modelPath.";
            if (backendExplicit || preferNative)
            {
                Console.WriteLine($"error> {message}");
                return 1;
            }

            Console.WriteLine($"warning> {message} Falling back to HTTP.");
            if (nativeBackend != null)
            {
                await nativeBackend.DisposeAsync();
                nativeBackend = null;
            }
            useNative = false;
            backend = BackendType.Http;
        }

        if (options.HealthOnly)
        {
            var health = useNative
                ? await GetNativeHealthAsync(nativeBackend, provider)
                : await GetHealthAsync(http, baseUrl, provider);
            PrintHealth(health, options.JsonOutput);
            return health.Ok ? 0 : 1;
        }

        if (options.ModelsOnly)
        {
            var models = useNative
                ? GetNativeModels(modelPath)
                : await GetModelsAsync(http, baseUrl, provider);
            if (options.JsonOutput)
            {
                Console.WriteLine(JsonSerializer.Serialize(models, JsonOptions));
            }
            else
            {
                Console.WriteLine("Available models:");
                foreach (var name in models)
                {
                    Console.WriteLine($"- {name}");
                }
            }
            return 0;
        }

        if (useNative && nativeBackend != null && !string.IsNullOrWhiteSpace(modelPath))
        {
            var loaded = await nativeBackend.LoadModelAsync(modelPath, gpuLayers);
            if (!loaded)
            {
                var error = InferenceModule.GetLastError() ?? "Unknown error";
                if (backendExplicit || preferNative)
                {
                    Console.WriteLine($"error> Failed to load model: {error}");
                    return 1;
                }

                Console.WriteLine($"warning> Failed to load model: {error}. Falling back to HTTP.");
                await nativeBackend.DisposeAsync();
                nativeBackend = null;
                useNative = false;
                backend = BackendType.Http;
            }
        }

        PrintBanner(baseUrl, model, provider, mode, backend, useNative, modelPath);

        if (string.Equals(mode, "single", StringComparison.OrdinalIgnoreCase))
        {
            var prompt = options.Prompt;
            if (string.IsNullOrWhiteSpace(prompt))
            {
                Console.Write("prompt> ");
                prompt = Console.ReadLine();
            }

            if (!string.IsNullOrWhiteSpace(prompt))
            {
                if (useNative && nativeBackend != null)
                {
                    await RunSingleNativeAsync(nativeBackend, systemPrompt, prompt, defaultMaxTokens, defaultTemperature);
                }
                else
                {
                    await RunSingleAsync(http, baseUrl, model, provider, systemPrompt, prompt);
                }
            }
            return 0;
        }

        var stream = string.Equals(mode, "stream", StringComparison.OrdinalIgnoreCase);
        var react = string.Equals(mode, "react", StringComparison.OrdinalIgnoreCase) || mode == "diagnose";
        var toolsPath = options.ToolsPath ?? config.ToolsPath ?? PathResolver.ResolvePath("Config/pcai-tools.json");

        await RunInteractiveAsync(
            http,
            baseUrl,
            model,
            provider,
            systemPrompt,
            stream,
            react,
            toolsPath,
            mode,
            options.JsonOutput,
            nativeBackend,
            useNative,
            defaultMaxTokens,
            defaultTemperature,
            modelPath);

        if (nativeBackend != null)
        {
            await nativeBackend.DisposeAsync();
        }
        return 0;
    }

    private static void PrintBanner(string baseUrl, string model, string provider, string mode, BackendType backend, bool useNative, string? modelPath)
    {
        Console.WriteLine("PC_AI Chat TUI");
        Console.WriteLine($"Provider: {provider}");
        Console.WriteLine($"Backend: {backend}");
        if (useNative)
        {
            Console.WriteLine($"ModelPath: {modelPath ?? "not set"}");
        }
        else
        {
            Console.WriteLine($"Endpoint: {baseUrl}");
            Console.WriteLine($"Model: {model}");
        }
        Console.WriteLine($"Mode: {mode}");
        Console.WriteLine("Commands: /mode [chat|diagnose], /health, /models, /reset, /exit");
        Console.WriteLine();
    }

    private static string? LoadModePrompt(string mode)
    {
        var path = mode == "diagnose"
            ? PathResolver.ResolvePath("DIAGNOSE.md")
            : PathResolver.ResolvePath("CHAT.md");

        if (File.Exists(path))
        {
            var content = File.ReadAllText(path);
            if (mode == "diagnose")
            {
                var logicPath = PathResolver.ResolvePath("DIAGNOSE_LOGIC.md");
                if (File.Exists(logicPath))
                {
                    content += "\n\n" + File.ReadAllText(logicPath);
                }
            }
            return content;
        }

        return null;
    }

    private static async Task RunInteractiveAsync(
        HttpClient http,
        string baseUrl,
        string model,
        string provider,
        string? systemPrompt,
        bool stream,
        bool react,
        string toolsPath,
        string initialMode,
        bool jsonOutput,
        IInferenceBackend? nativeBackend,
        bool useNative,
        int defaultMaxTokens,
        float defaultTemperature,
        string? modelPath)
    {
        var mode = initialMode;
        var history = new List<ChatMessage>();
        if (!string.IsNullOrWhiteSpace(systemPrompt))
        {
            history.Add(new ChatMessage("system", systemPrompt));
            Console.WriteLine("System prompt loaded. Use /reset to clear context.");
        }

        if (useNative && react)
        {
            Console.WriteLine("warning> ReAct/tool routing requires HTTP providers. Running native chat without tools.");
            react = false;
        }

        // Initialize PowerShell host + orchestrator for tool routing (HTTP only)
        using var psHost = react ? new PowerShellHost() : null;
        var interlock = react ? new SafetyInterlock(SafetyInterlock.ConsoleConfirmationHandler) : null;
        var toolExecutor = react ? new ToolExecutor(toolsPath, psHost, interlock) : null;
        using var openAiClient = react ? new PcaiOpenAiClient(baseUrl) : null;
        var orchestrator = react ? new ReActOrchestrator(openAiClient!, toolExecutor!, model) : null;

        if (orchestrator != null)
        {
            // Bind orchestration events for TUI feedback
            orchestrator.OnThought += (thought) => Console.WriteLine($"thought> {thought}");
            orchestrator.OnToolCall += (name, args) => Console.WriteLine($"executing> {name}({args})");
            orchestrator.OnToolResult += (name, result) => Console.WriteLine($"result> {name} completed.");
            orchestrator.OnFinalAnswer += (answer) =>
            {
                if (string.Equals(mode, "diagnose", StringComparison.OrdinalIgnoreCase))
                {
                    if (!TryValidateDiagnoseJson(answer, out var error))
                    {
                        Console.WriteLine($"error> Diagnose mode requires valid JSON output. {error}");
                        Console.WriteLine($"raw> {answer}");
                        return;
                    }

                    if (jsonOutput)
                    {
                        using var doc = JsonDocument.Parse(answer);
                        Console.WriteLine(JsonSerializer.Serialize(doc.RootElement, JsonOptions));
                        Console.WriteLine();
                        return;
                    }

                    var summary = BuildDiagnoseSummary(answer);
                    if (!string.IsNullOrWhiteSpace(summary))
                    {
                        Console.WriteLine("assistant>");
                        Console.WriteLine(summary);
                        Console.WriteLine();
                        return;
                    }
                }

                Console.WriteLine($"assistant> {answer}\n");
            };
            orchestrator.OnError += (error) => Console.WriteLine($"error> {error}");
        }

        while (true)
        {
            Console.Write($"{mode}> ");
            var input = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(input))
            {
                continue;
            }

            if (input.StartsWith("/", StringComparison.OrdinalIgnoreCase))
            {
                if (input.StartsWith("/mode", StringComparison.OrdinalIgnoreCase))
                {
                    var parts = input.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length < 2)
                    {
                        Console.WriteLine($"Current mode: {mode}");
                        continue;
                    }

                    var newMode = parts[1].Trim().ToLowerInvariant();
                    if (newMode is "chat" or "diagnose")
                    {
                        mode = newMode;
                        react = newMode == "diagnose";
                        systemPrompt = LoadModePrompt(mode);
                        history.Clear();
                        if (!string.IsNullOrWhiteSpace(systemPrompt))
                        {
                            history.Add(new ChatMessage("system", systemPrompt));
                        }
                        Console.WriteLine($"Switched to {mode} mode. Context reset.");
                        continue;
                    }
                    else
                    {
                        Console.WriteLine("Invalid mode. Use 'chat' or 'diagnose'.");
                        continue;
                    }
                }

                var command = input.Trim().ToLowerInvariant();
                switch (command)
                {
                    case "/exit":
                    case "/quit":
                        return;
                    case "/reset":
                        history.Clear();
                        if (!string.IsNullOrWhiteSpace(systemPrompt))
                        {
                            history.Add(new ChatMessage("system", systemPrompt));
                        }
                        Console.WriteLine("Context cleared.");
                        continue;
                    case "/health":
                        var health = useNative
                            ? await GetNativeHealthAsync(nativeBackend, provider)
                            : await GetHealthAsync(http, baseUrl, provider);
                        PrintHealth(health, jsonOutput: false);
                        continue;
                    case "/models":
                        var models = useNative
                            ? GetNativeModels(modelPath)
                            : await GetModelsAsync(http, baseUrl, provider);
                        Console.WriteLine("Available models:");
                        foreach (var name in models)
                        {
                            Console.WriteLine($"- {name}");
                        }
                        continue;
                    default:
                        Console.WriteLine("Unknown command.");
                        continue;
                }
            }

            history.Add(new ChatMessage("user", input));

            try
            {
                if (react && orchestrator != null)
                {
                    // Use the unified ReAct orchestrator
                    // We need to inject native context if in diagnose mode
                    if (mode == "diagnose" && PcaiCore.IsAvailable)
                    {
                        var context = PcaiCore.QueryFullContextJson();
                        if (!string.IsNullOrEmpty(context))
                        {
                            var oldContext = history.FirstOrDefault(m => m.Content.Contains("[NATIVE_CONTEXT]"));
                            if (oldContext != null) history.Remove(oldContext);
                            history.Insert(1, new ChatMessage("system", $"[NATIVE_CONTEXT]\n{context}"));
                        }
                    }

                    // Build full prompt from history for the orchestrator
                    // (Orchestrator currently takes a simple prompt, we should enhance it later to take history)
                    var fullPrompt = string.Join("\n", history.Select(m => $"{m.Role}: {m.Content}"));
                    await orchestrator.RunAsync(fullPrompt);
                    continue;
                }

                if (useNative && nativeBackend != null)
                {
                    var nativePrompt = BuildPrompt(history);
                    if (stream)
                    {
                        Console.Write("assistant> ");
                        var sb = new StringBuilder();
                        await foreach (var token in nativeBackend.GenerateStreamingAsync(nativePrompt, defaultMaxTokens, defaultTemperature))
                        {
                            if (!string.IsNullOrEmpty(token))
                            {
                                Console.Write(token);
                                sb.Append(token);
                            }
                        }
                        Console.WriteLine();
                        history.Add(new ChatMessage("assistant", sb.ToString()));
                    }
                    else
                    {
                        var content = await nativeBackend.GenerateAsync(nativePrompt, defaultMaxTokens, defaultTemperature);
                        history.Add(new ChatMessage("assistant", content));
                        Console.WriteLine($"assistant> {content}\n");
                    }
                }
                else if (stream)
                {
                    Console.Write("assistant> ");
                    var content = await StreamOpenAiChatAsync(http, baseUrl, model, history);
                    history.Add(new ChatMessage("assistant", content));
                }
                else
                {
                    var content = await SendOpenAiChatAsync(http, baseUrl, model, history);
                    history.Add(new ChatMessage("assistant", content));
                    Console.WriteLine($"assistant> {content}\n");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"error> {ex.Message}");
            }
        }
    }

    private static async Task RunSingleAsync(
        HttpClient http,
        string baseUrl,
        string model,
        string provider,
        string? systemPrompt,
        string prompt)
    {
        var messages = new List<ChatMessage>();
        if (!string.IsNullOrWhiteSpace(systemPrompt))
        {
            messages.Add(new ChatMessage("system", systemPrompt));
        }
        messages.Add(new ChatMessage("user", prompt));

        var openAiContent = await SendOpenAiChatAsync(http, baseUrl, model, messages);
        Console.WriteLine(openAiContent);
    }

    private static async Task RunSingleNativeAsync(
        IInferenceBackend nativeBackend,
        string? systemPrompt,
        string prompt,
        int maxTokens,
        float temperature)
    {
        var messages = new List<ChatMessage>();
        if (!string.IsNullOrWhiteSpace(systemPrompt))
        {
            messages.Add(new ChatMessage("system", systemPrompt));
        }
        messages.Add(new ChatMessage("user", prompt));

        var nativePrompt = BuildPrompt(messages);
        var content = await nativeBackend.GenerateAsync(nativePrompt, maxTokens, temperature);
        Console.WriteLine(content);
    }

    private static async Task<HealthResult> GetHealthAsync(HttpClient http, string baseUrl, string provider)
    {
        var result = new HealthResult
        {
            Endpoint = baseUrl,
            Native = PcaiCore.GetDiagnostics()
        };

        try
        {
            var response = await http.GetAsync($"{baseUrl}/v1/models");
            result.Ok = response.IsSuccessStatusCode;
            result.Version = response.IsSuccessStatusCode ? "ok" : "unknown";
        }
        catch (Exception ex)
        {
            result.Ok = false;
            result.Error = ex.Message;
        }

        return result;
    }

    private static async Task<HealthResult> GetNativeHealthAsync(IInferenceBackend? nativeBackend, string provider)
    {
        var result = new HealthResult
        {
            Endpoint = "pcai-native",
            Native = PcaiCore.GetDiagnostics()
        };

        if (nativeBackend == null)
        {
            result.Ok = false;
            result.Error = "Native backend not initialized";
            return result;
        }

        result.Ok = await nativeBackend.CheckAvailabilityAsync();
        if (!result.Ok)
        {
            result.Error = InferenceModule.GetLastError() ?? "Native backend unavailable";
        }

        return result;
    }

    private static async Task<IReadOnlyList<string>> GetModelsAsync(HttpClient http, string baseUrl, string provider)
    {
        var openAi = await http.GetFromJsonAsync<OpenAiModels>($"{baseUrl}/v1/models", JsonOptions);
        if (openAi?.Data == null)
        {
            return Array.Empty<string>();
        }

        return openAi.Data.Select(m => m.Id).ToArray();
    }

    private static IReadOnlyList<string> GetNativeModels(string? modelPath)
    {
        if (!string.IsNullOrWhiteSpace(modelPath))
        {
            return new[] { modelPath };
        }

        return new[] { "pcai-native" };
    }

    private static async Task<string> SendOllamaChatAsync(HttpClient http, string baseUrl, string model, List<ChatMessage> messages)
    {
        var request = new OllamaChatRequest
        {
            Model = model,
            Stream = false,
            Messages = messages
        };

        var response = await http.PostAsJsonAsync($"{baseUrl}/api/chat", request, JsonOptions);
        response.EnsureSuccessStatusCode();

        var payload = await response.Content.ReadFromJsonAsync<OllamaChatResponse>(JsonOptions);
        return payload?.Message?.Content ?? string.Empty;
    }

    private static bool TryValidateDiagnoseJson(string content, out string error)
    {
        error = string.Empty;
        if (string.IsNullOrWhiteSpace(content))
        {
            error = "Response is empty.";
            return false;
        }

        try
        {
            using var doc = JsonDocument.Parse(content);
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
            {
                error = "Root JSON must be an object.";
                return false;
            }
            return true;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return false;
        }
    }

    private static string BuildDiagnoseSummary(string content)
    {
        using var doc = JsonDocument.Parse(content);
        var root = doc.RootElement;
        var sb = new StringBuilder();
        const int maxItems = 5;

        AppendStringArray(root, "summary", "Summary", sb, maxItems);
        AppendFindings(root, sb, maxItems);
        AppendRecommendations(root, sb, maxItems);
        AppendStringArray(root, "what_is_missing", "Missing Data", sb, maxItems);

        return sb.ToString().TrimEnd();
    }

    private static string BuildPrompt(List<ChatMessage> history)
    {
        var sb = new StringBuilder();
        foreach (var message in history)
        {
            var label = message.Role switch
            {
                "system" => "System",
                "user" => "User",
                "assistant" => "Assistant",
                _ => message.Role
            };
            sb.AppendLine($"{label}: {message.Content}");
        }
        sb.Append("Assistant: ");
        return sb.ToString();
    }

    private static void AppendStringArray(JsonElement root, string propName, string title, StringBuilder sb, int maxItems)
    {
        if (!root.TryGetProperty(propName, out var array) || array.ValueKind != JsonValueKind.Array)
        {
            return;
        }

        var count = array.GetArrayLength();
        if (count == 0)
        {
            return;
        }

        sb.AppendLine($"{title}:");
        var i = 0;
        foreach (var item in array.EnumerateArray())
        {
            if (i >= maxItems)
            {
                sb.AppendLine($"- ... ({count - maxItems} more)");
                break;
            }
            var text = item.ValueKind == JsonValueKind.String ? item.GetString() : item.ToString();
            if (!string.IsNullOrWhiteSpace(text))
            {
                sb.AppendLine($"- {text}");
            }
            i++;
        }
        sb.AppendLine();
    }

    private static void AppendFindings(JsonElement root, StringBuilder sb, int maxItems)
    {
        if (!root.TryGetProperty("findings", out var findings) || findings.ValueKind != JsonValueKind.Array)
        {
            return;
        }

        var count = findings.GetArrayLength();
        if (count == 0)
        {
            return;
        }

        sb.AppendLine("Findings:");
        var i = 0;
        foreach (var item in findings.EnumerateArray())
        {
            if (i >= maxItems)
            {
                sb.AppendLine($"- ... ({count - maxItems} more)");
                break;
            }

            var criticality = GetStringProperty(item, "criticality");
            var category = GetStringProperty(item, "category");
            var issue = GetStringProperty(item, "issue");

            var line = new StringBuilder("- ");
            if (!string.IsNullOrWhiteSpace(criticality))
            {
                line.Append($"[{criticality}] ");
            }
            if (!string.IsNullOrWhiteSpace(category))
            {
                line.Append($"{category}: ");
            }
            line.Append(string.IsNullOrWhiteSpace(issue) ? item.ToString() : issue);
            sb.AppendLine(line.ToString());
            i++;
        }
        sb.AppendLine();
    }

    private static void AppendRecommendations(JsonElement root, StringBuilder sb, int maxItems)
    {
        if (!root.TryGetProperty("recommendations", out var recs) || recs.ValueKind != JsonValueKind.Array)
        {
            return;
        }

        var count = recs.GetArrayLength();
        if (count == 0)
        {
            return;
        }

        sb.AppendLine("Recommendations:");
        var i = 0;
        foreach (var item in recs.EnumerateArray())
        {
            if (i >= maxItems)
            {
                sb.AppendLine($"- ... ({count - maxItems} more)");
                break;
            }

            var step = GetStringProperty(item, "step");
            var action = GetStringProperty(item, "action");
            var risk = GetStringProperty(item, "risk");

            var prefix = string.IsNullOrWhiteSpace(step) ? $"{i + 1}." : $"{step}.";
            var line = new StringBuilder($"{prefix} ");
            line.Append(string.IsNullOrWhiteSpace(action) ? item.ToString() : action);
            if (!string.IsNullOrWhiteSpace(risk))
            {
                line.Append($" (risk: {risk})");
            }
            sb.AppendLine(line.ToString());
            i++;
        }
        sb.AppendLine();
    }

    private static string GetStringProperty(JsonElement element, string name)
    {
        if (!element.TryGetProperty(name, out var value))
        {
            return string.Empty;
        }
        return value.ValueKind == JsonValueKind.String ? (value.GetString() ?? string.Empty) : value.ToString();
    }

    private static async Task<string> StreamOllamaChatAsync(HttpClient http, string baseUrl, string model, List<ChatMessage> messages)
    {
        var request = new OllamaChatRequest
        {
            Model = model,
            Stream = true,
            Messages = messages
        };

        var json = JsonSerializer.Serialize(request, JsonOptions);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        var requestMessage = new HttpRequestMessage(HttpMethod.Post, $"{baseUrl}/api/chat") { Content = content };
        using var response = await http.SendAsync(requestMessage, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync();
        using var reader = new StreamReader(stream);
        var sb = new StringBuilder();

        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync();
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            try
            {
                using var doc = JsonDocument.Parse(line);
                if (doc.RootElement.TryGetProperty("message", out var messageElement))
                {
                    var contentText = messageElement.GetProperty("content").GetString() ?? string.Empty;
                    if (!string.IsNullOrEmpty(contentText))
                    {
                        Console.Write(contentText);
                        sb.Append(contentText);
                    }
                }

                if (doc.RootElement.TryGetProperty("done", out var doneElement) && doneElement.GetBoolean())
                {
                    Console.WriteLine();
                    break;
                }
            }
            catch
            {
                continue;
            }
        }

        return sb.ToString();
    }

    private static async Task<string> SendOpenAiChatAsync(HttpClient http, string baseUrl, string model, List<ChatMessage> messages)
    {
        var payload = new
        {
            model,
            messages = messages.Select(m => new { role = m.Role, content = m.Content }).ToArray(),
            temperature = 0.2
        };

        var response = await http.PostAsJsonAsync($"{baseUrl}/v1/chat/completions", payload, JsonOptions);
        response.EnsureSuccessStatusCode();
        var json = await response.Content.ReadFromJsonAsync<OpenAiChatResponse>(JsonOptions);
        return json?.Choices?.FirstOrDefault()?.Message?.Content ?? string.Empty;
    }

    private static async Task<string> StreamOpenAiChatAsync(HttpClient http, string baseUrl, string model, List<ChatMessage> messages)
    {
        var payload = new
        {
            model,
            messages = messages.Select(m => new { role = m.Role, content = m.Content }).ToArray(),
            temperature = 0.2,
            stream = true
        };

        var json = JsonSerializer.Serialize(payload, JsonOptions);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        var request = new HttpRequestMessage(HttpMethod.Post, $"{baseUrl}/v1/chat/completions") { Content = content };
        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync();
        using var reader = new StreamReader(stream);
        var sb = new StringBuilder();

        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync();
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            if (!line.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var payloadLine = line.Substring(5).Trim();
            if (string.Equals(payloadLine, "[DONE]", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine();
                break;
            }

            try
            {
                using var doc = JsonDocument.Parse(payloadLine);
                var choice = doc.RootElement.GetProperty("choices")[0];
                if (choice.TryGetProperty("delta", out var delta) &&
                    delta.TryGetProperty("content", out var contentElement))
                {
                    var contentText = contentElement.GetString() ?? string.Empty;
                    if (!string.IsNullOrEmpty(contentText))
                    {
                        Console.Write(contentText);
                        sb.Append(contentText);
                    }
                }
            }
            catch
            {
                // Ignore malformed chunks
            }
        }

        return sb.ToString();
    }

    private static async Task<OpenAiChatResponse?> SendOpenAiToolCallAsync(HttpClient http, string baseUrl, string model, List<ChatMessage> messages, List<object> tools)
    {
        var payload = new
        {
            model,
            messages = messages.Select(m => new { role = m.Role, content = m.Content }).ToArray(),
            tools = tools,
            tool_choice = "auto",
            temperature = 0.2
        };

        var response = await http.PostAsJsonAsync($"{baseUrl}/v1/chat/completions", payload, JsonOptions);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<OpenAiChatResponse>(JsonOptions);
    }

    private static List<object> LoadTools(string toolsPath)
    {
        if (!File.Exists(toolsPath))
        {
            return new List<object>();
        }

        try
        {
            using var stream = File.OpenRead(toolsPath);
            using var doc = JsonDocument.Parse(stream);
            if (!doc.RootElement.TryGetProperty("tools", out var tools))
            {
                return new List<object>();
            }

            return tools.EnumerateArray()
                .Select(t => JsonSerializer.Deserialize<object>(t.GetRawText(), JsonOptions)!)
                .ToList();
        }
        catch
        {
            return new List<object>();
        }
    }

    private static string ResolveHvsockEndpoint(string url, string? configPath)
    {
        if (!url.StartsWith("hvsock://", StringComparison.OrdinalIgnoreCase) &&
            !url.StartsWith("vsock://", StringComparison.OrdinalIgnoreCase))
        {
            return url;
        }

        var name = url.Split("://", 2, StringSplitOptions.RemoveEmptyEntries).LastOrDefault();
        if (string.IsNullOrWhiteSpace(name))
        {
            return url;
        }

        var path = configPath ?? PathResolver.ResolvePath("Config/hvsock-proxy.conf");
        if (!File.Exists(path))
        {
            return url;
        }

        foreach (var line in File.ReadAllLines(path))
        {
            var trimmed = line.Trim();
            if (string.IsNullOrWhiteSpace(trimmed) || trimmed.StartsWith("#"))
            {
                continue;
            }

            var parts = trimmed.Split(':');
            if (parts.Length < 4)
            {
                continue;
            }

            if (!string.Equals(parts[0], name, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var host = parts[2];
            var port = parts[3];
            return $"http://{host}:{port}";
        }

        return url;
    }

    private static void PrintHealth(HealthResult health, bool jsonOutput)
    {
        if (jsonOutput)
        {
            Console.WriteLine(JsonSerializer.Serialize(health, JsonOptions));
            return;
        }

        Console.WriteLine("LLM Health");
        Console.WriteLine($"  Endpoint: {health.Endpoint}");
        Console.WriteLine($"  Status: {(health.Ok ? "OK" : "ERROR")}");
        if (!string.IsNullOrWhiteSpace(health.Version))
        {
            Console.WriteLine($"  Version: {health.Version}");
        }
        if (!string.IsNullOrWhiteSpace(health.Error))
        {
            Console.WriteLine($"  Error: {health.Error}");
        }

        Console.WriteLine("Native Diagnostics");
        Console.WriteLine($"  Available: {health.Native.IsAvailable}");
        Console.WriteLine($"  Version: {health.Native.Version}");
        Console.WriteLine($"  CPU: {health.Native.CpuCount}");
    }

    private static string NormalizeBaseUrl(string url)
    {
        return url.TrimEnd('/');
    }
}

/// <summary>Parsed command-line options for the PcaiChatTui executable.</summary>
public sealed class CliOptions
{
    public string? BaseUrl { get; init; }
    public string? Model { get; init; }
    public string? ConfigPath { get; init; }
    public int? TimeoutSec { get; init; }
    public bool HealthOnly { get; init; }
    public bool ModelsOnly { get; init; }
    public bool JsonOutput { get; init; }
    public string? SystemPrompt { get; init; }
    public string? SystemFile { get; init; }
    public string? Provider { get; init; }
    public string? Mode { get; init; }
    public string? Prompt { get; init; }
    public string? ToolsPath { get; init; }
    public string? HvsockConfigPath { get; init; }

    // Native inference options
    public BackendType Backend { get; init; } = BackendType.Auto;
    public string? ModelPath { get; init; }
    public int GpuLayers { get; init; } = -1;

    public static CliOptions Parse(string[] args)
    {
        string? baseUrl = null;
        string? model = null;
        string? configPath = null;
        int? timeoutSec = null;
        bool health = false;
        bool models = false;
        bool json = false;
        string? systemPrompt = null;
        string? systemFile = null;
        string? provider = null;
        string? mode = null;
        string? prompt = null;
        string? toolsPath = null;
        string? hvsockConfigPath = null;

        // Native inference options
        var backend = BackendType.Auto;
        string? modelPath = null;
        int gpuLayers = -1;

        for (var i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            switch (arg)
            {
                case "--base-url":
                case "-u":
                    baseUrl = NextValue(args, ref i);
                    break;
                case "--model":
                case "-m":
                    model = NextValue(args, ref i);
                    break;
                case "--config":
                case "-c":
                    configPath = NextValue(args, ref i);
                    break;
                case "--timeout":
                    if (int.TryParse(NextValue(args, ref i), out var parsed))
                    {
                        timeoutSec = parsed;
                    }
                    break;
                case "--health":
                    health = true;
                    break;
                case "--models":
                    models = true;
                    break;
                case "--json":
                    json = true;
                    break;
                case "--system":
                    systemPrompt = NextValue(args, ref i);
                    break;
                case "--system-file":
                    systemFile = NextValue(args, ref i);
                    break;
                case "--provider":
                    provider = NextValue(args, ref i);
                    break;
                case "--mode":
                    mode = NextValue(args, ref i);
                    break;
                case "--prompt":
                    prompt = NextValue(args, ref i);
                    break;
                case "--tools":
                    toolsPath = NextValue(args, ref i);
                    break;
                case "--hvsock-config":
                    hvsockConfigPath = NextValue(args, ref i);
                    break;
                case "--backend":
                case "-b":
                    var backendStr = NextValue(args, ref i).ToLowerInvariant();
                    backend = backendStr switch
                    {
                        "auto" => BackendType.Auto,
                        "http" => BackendType.Http,
                        "llamacpp" => BackendType.LlamaCpp,
                        "mistralrs" => BackendType.MistralRs,
                        _ => BackendType.Auto
                    };
                    break;
                case "--model-path":
                    modelPath = NextValue(args, ref i);
                    break;
                case "--gpu-layers":
                    if (int.TryParse(NextValue(args, ref i), out var parsedLayers))
                    {
                        gpuLayers = parsedLayers;
                    }
                    break;
                case "--help":
                case "-h":
                    PrintHelp();
                    Environment.Exit(0);
                    break;
            }
        }

        return new CliOptions
        {
            BaseUrl = baseUrl,
            Model = model,
            ConfigPath = configPath,
            TimeoutSec = timeoutSec,
            HealthOnly = health,
            ModelsOnly = models,
            JsonOutput = json,
            SystemPrompt = systemPrompt,
            SystemFile = systemFile,
            Provider = provider,
            Mode = mode,
            Prompt = prompt,
            ToolsPath = toolsPath,
            HvsockConfigPath = hvsockConfigPath,
            Backend = backend,
            ModelPath = modelPath,
            GpuLayers = gpuLayers
        };
    }

    private static void PrintHelp()
    {
        Console.WriteLine(@"PC-AI Chat TUI

Usage: PcaiChatTui [options]

Options:
  --base-url, -u <url>      LLM endpoint URL (default: http://127.0.0.1:8080)
  --model, -m <name>        Model name for HTTP providers
  --provider <name>         Provider: pcai-inference, pcai-native, vllm, lmstudio (default: pcai-inference)
  --mode <mode>             Mode: chat, diagnose, stream, single, react
  --config, -c <path>       Config file path
  --timeout <seconds>       Request timeout
  --system <prompt>         System prompt text
  --system-file <path>      System prompt file
  --tools <path>            Tools JSON path
  --health                  Check endpoint health
  --models                  List available models
  --json                    Output as JSON

Native Inference:
  --backend, -b <type>      Backend: auto, http, llamacpp, mistralrs
  --model-path <path>       Path to GGUF/SafeTensors model file
  --gpu-layers <n>          GPU layers (-1 = all, 0 = CPU only)

  --help, -h                Show this help
");
    }

    private static string NextValue(string[] args, ref int index)
    {
        if (index + 1 >= args.Length)
        {
            return string.Empty;
        }

        index++;
        return args[index];
    }
}

/// <summary>Strongly-typed representation of <c>Config/llm-config.json</c>.</summary>
public sealed class LlmConfig
{
    public string? DefaultProvider { get; init; }
    public Dictionary<string, ProviderConfig> Providers { get; init; } = new(StringComparer.OrdinalIgnoreCase);
    public NativeProviderConfig? NativeProvider { get; init; }
    public string? ToolsPath { get; init; }
    public string? HvsockConfigPath { get; init; }

    public ProviderConfig? GetProvider(string name)
    {
        if (Providers.TryGetValue(name, out var provider))
        {
            return provider;
        }
        return null;
    }

    public static LlmConfig Load(string? configPath)
    {
        var path = configPath ?? PathResolver.ResolvePath("Config/llm-config.json");
        if (!File.Exists(path))
        {
            return new LlmConfig();
        }

        try
        {
            using var stream = File.OpenRead(path);
            using var doc = JsonDocument.Parse(stream);

            var providersMap = new Dictionary<string, ProviderConfig>(StringComparer.OrdinalIgnoreCase);
            string? defaultProvider = null;
            string? toolsPath = null;
            var hvsockConfigPath = PathResolver.ResolvePath("Config/hvsock-proxy.conf");
            NativeProviderConfig? nativeProvider = null;

            if (doc.RootElement.TryGetProperty("providers", out var providers))
            {
                foreach (var provider in providers.EnumerateObject())
                {
                    var providerElement = provider.Value;
                    var type = providerElement.TryGetProperty("type", out var typeElement)
                        ? typeElement.GetString()
                        : null;
                    var baseUrl = providerElement.TryGetProperty("baseUrl", out var baseUrlElement)
                        ? baseUrlElement.GetString()
                        : null;
                    var model = providerElement.TryGetProperty("defaultModel", out var modelElement)
                        ? modelElement.GetString()
                        : null;
                    var timeout = providerElement.TryGetProperty("timeout", out var timeoutElement)
                        ? timeoutElement.GetInt32() / 1000
                        : (int?)null;

                    providersMap[provider.Name] = new ProviderConfig
                    {
                        Type = type,
                        BaseUrl = baseUrl,
                        DefaultModel = model,
                        TimeoutSec = timeout
                    };

                    if (string.Equals(type, "ffi", StringComparison.OrdinalIgnoreCase) ||
                        string.Equals(provider.Name, "pcai-native", StringComparison.OrdinalIgnoreCase))
                    {
                        var backend = providerElement.TryGetProperty("backend", out var backendElement)
                            ? backendElement.GetString()
                            : null;
                        var modelPath = providerElement.TryGetProperty("modelPath", out var modelPathElement)
                            ? modelPathElement.GetString()
                            : null;
                        var gpuLayers = providerElement.TryGetProperty("gpuLayers", out var gpuElement)
                            ? gpuElement.GetInt32()
                            : (int?)null;
                        var defaultTemp = providerElement.TryGetProperty("defaultTemperature", out var tempElement)
                            ? tempElement.GetSingle()
                            : (float?)null;
                        var defaultMaxTokens = providerElement.TryGetProperty("defaultMaxTokens", out var maxTokensElement)
                            ? maxTokensElement.GetInt32()
                            : (int?)null;

                        nativeProvider = new NativeProviderConfig
                        {
                            Backend = backend,
                            ModelPath = modelPath,
                            GpuLayers = gpuLayers,
                            DefaultTemperature = defaultTemp,
                            DefaultMaxTokens = defaultMaxTokens
                        };
                    }
                }
            }

            if (doc.RootElement.TryGetProperty("fallbackOrder", out var fallback))
            {
                defaultProvider = fallback.EnumerateArray().Select(e => e.GetString()).FirstOrDefault(s => !string.IsNullOrWhiteSpace(s));
            }

            if (doc.RootElement.TryGetProperty("router", out var router) &&
                router.TryGetProperty("toolsPath", out var toolsElement))
            {
                toolsPath = toolsElement.GetString();
            }

            return new LlmConfig
            {
                DefaultProvider = defaultProvider,
                ToolsPath = toolsPath,
                HvsockConfigPath = hvsockConfigPath,
                NativeProvider = nativeProvider,
                Providers = providersMap
            };
        }
        catch
        {
            return new LlmConfig();
        }
    }
}

/// <summary>Configuration for a single named LLM provider entry.</summary>
public sealed class ProviderConfig
{
    public string? Type { get; init; }
    public string? BaseUrl { get; init; }
    public string? DefaultModel { get; init; }
    public int? TimeoutSec { get; init; }
}

/// <summary>Native FFI provider settings read from the <c>pcai-native</c> provider block.</summary>
public sealed class NativeProviderConfig
{
    public string? Backend { get; init; }
    public string? ModelPath { get; init; }
    public int? GpuLayers { get; init; }
    public float? DefaultTemperature { get; init; }
    public int? DefaultMaxTokens { get; init; }
}

/// <summary>Result of an LLM endpoint health check, serialised for JSON output.</summary>
public sealed class HealthResult
{
    [JsonPropertyName("ok")]
    public bool Ok { get; set; }

    [JsonPropertyName("endpoint")]
    public string Endpoint { get; set; } = string.Empty;

    [JsonPropertyName("version")]
    public string? Version { get; set; }

    [JsonPropertyName("error")]
    public string? Error { get; set; }

    [JsonPropertyName("native")]
    public NativeDiagnostics Native { get; set; } = new();
}

/// <summary>Ollama version response model.</summary>
public sealed class OllamaVersion
{
    [JsonPropertyName("version")]
    public string Version { get; set; } = string.Empty;
}

/// <summary>Ollama model tags list response.</summary>
public sealed class OllamaTags
{
    [JsonPropertyName("models")]
    public List<OllamaModel>? Models { get; set; }
}

/// <summary>A single model entry in the Ollama tags list.</summary>
public sealed class OllamaModel
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;
}

/// <summary>OpenAI-compatible <c>/v1/models</c> response envelope.</summary>
public sealed class OpenAiModels
{
    [JsonPropertyName("data")]
    public List<OpenAiModel>? Data { get; set; }
}

/// <summary>A single model entry from the OpenAI models list.</summary>
public sealed class OpenAiModel
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;
}

/// <summary>OpenAI-compatible chat completion response.</summary>
public sealed class OpenAiChatResponse
{
    [JsonPropertyName("choices")]
    public List<OpenAiChoice>? Choices { get; set; }
}

/// <summary>A single choice within a chat completion response.</summary>
public sealed class OpenAiChoice
{
    [JsonPropertyName("message")]
    public OpenAiMessage? Message { get; set; }
}

/// <summary>Message payload within an OpenAI choice, including optional tool calls.</summary>
public sealed class OpenAiMessage
{
    [JsonPropertyName("content")]
    public string Content { get; set; } = string.Empty;

    [JsonPropertyName("tool_calls")]
    public JsonElement? ToolCalls { get; set; }
}

/// <summary>Request body for the Ollama <c>/api/chat</c> endpoint.</summary>
public sealed class OllamaChatRequest
{
    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    [JsonPropertyName("stream")]
    public bool Stream { get; set; }

    [JsonPropertyName("messages")]
    public List<ChatMessage> Messages { get; set; } = new();
}

/// <summary>Response body from the Ollama <c>/api/chat</c> endpoint.</summary>
public sealed class OllamaChatResponse
{
    [JsonPropertyName("message")]
    public ChatMessage? Message { get; set; }
}

/// <summary>A single message in a chat conversation, compatible with both OpenAI and Ollama formats.</summary>
public sealed class ChatMessage
{
    /// <summary>Initialises an empty <see cref="ChatMessage"/> for deserialisation.</summary>
    public ChatMessage() { }

    /// <summary>Initialises a <see cref="ChatMessage"/> with the given role and content.</summary>
    public ChatMessage(string role, string content)
    {
        Role = role;
        Content = content;
    }

    [JsonPropertyName("role")]
    public string Role { get; set; } = string.Empty;

    [JsonPropertyName("content")]
    public string Content { get; set; } = string.Empty;

    [JsonPropertyName("tool_call_id")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ToolCallId { get; set; }

    [JsonPropertyName("tool_calls")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? ToolCalls { get; set; }
}

internal static class PathResolver
{
    private static readonly Lazy<string> ProjectRoot = new(() =>
    {
        var envRoot = Environment.GetEnvironmentVariable("PCAI_ROOT");
        if (!string.IsNullOrWhiteSpace(envRoot) && Directory.Exists(envRoot))
        {
            return envRoot;
        }

        var current = AppContext.BaseDirectory;
        while (!string.IsNullOrWhiteSpace(current))
        {
            if (File.Exists(Path.Combine(current, "PC-AI.ps1")) || Directory.Exists(Path.Combine(current, "Config")))
            {
                return current;
            }

            current = Directory.GetParent(current)?.FullName;
        }

        return AppContext.BaseDirectory;
    });

    public static string ResolvePath(string relativePath)
    {
        return Path.Combine(ProjectRoot.Value, relativePath);
    }
}
