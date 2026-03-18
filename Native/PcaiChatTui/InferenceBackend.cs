// InferenceBackend.cs - Backend selection for TUI
#nullable enable
using System.Net.Http.Json;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading.Channels;
using PcaiNative;

namespace PcaiChatTui;

/// <summary>Identifies which inference backend the TUI should connect to.</summary>
public enum BackendType
{
    /// <summary>Automatically select the best available backend.</summary>
    Auto,
    /// <summary>Connect to pcai-inference via the OpenAI-compatible HTTP API.</summary>
    Http,
    /// <summary>Use the native llama.cpp backend via FFI.</summary>
    LlamaCpp,
    /// <summary>Use the native mistral.rs backend via FFI.</summary>
    MistralRs
}

/// <summary>Abstraction over native FFI and HTTP inference backends.</summary>
public interface IInferenceBackend : IAsyncDisposable
{
    /// <summary>Human-readable name of the backend including the endpoint or variant.</summary>
    string Name { get; }

    /// <summary>Returns <c>true</c> if the backend is reachable and ready to serve requests.</summary>
    Task<bool> CheckAvailabilityAsync(CancellationToken cancellationToken = default);

    /// <summary>Loads the model at <paramref name="modelPath"/> with the specified GPU layer count.</summary>
    Task<bool> LoadModelAsync(string modelPath, int gpuLayers = -1);

    /// <summary>Generates a completion for <paramref name="prompt"/> and returns the full result.</summary>
    Task<string> GenerateAsync(string prompt, int maxTokens = 2048, float temperature = 0.7f);

    /// <summary>Generates a streaming completion, yielding tokens as they are produced.</summary>
    IAsyncEnumerable<string> GenerateStreamingAsync(string prompt, int maxTokens = 2048, float temperature = 0.7f);
}

/// <summary>Creates <see cref="IInferenceBackend"/> instances for the requested <see cref="BackendType"/>.</summary>
public static class BackendFactory
{
    /// <summary>Creates and returns the backend for <paramref name="type"/>, performing auto-detection when needed.</summary>
    public static async Task<IInferenceBackend> CreateAsync(BackendType type, string? httpEndpoint = null)
    {
        return type switch
        {
            BackendType.Http => CreateHttpBackend(httpEndpoint),
            BackendType.LlamaCpp => new NativeBackend("llamacpp"),
            BackendType.MistralRs => new NativeBackend("mistralrs"),
            BackendType.Auto => await ResolveAutoAsync(httpEndpoint),
            _ => throw new ArgumentException($"Unknown backend type: {type}")
        };
    }

    private static async Task<IInferenceBackend> ResolveAutoAsync(string? httpEndpoint)
    {
        // Try native first, fall back to HTTP
        var native = new NativeBackend("mistralrs");
        if (await native.CheckAvailabilityAsync(default))
            return native;

        native = new NativeBackend("llamacpp");
        if (await native.CheckAvailabilityAsync(default))
            return native;

        return CreateHttpBackend(httpEndpoint);
    }

    private static IInferenceBackend CreateHttpBackend(string? httpEndpoint)
    {
        if (string.IsNullOrWhiteSpace(httpEndpoint))
        {
            throw new InvalidOperationException(
                "HTTP endpoint is not configured. Set providers.pcai-inference.baseUrl in Config/llm-config.json or pass --base-url.");
        }

        return new HttpBackend(httpEndpoint);
    }
}

/// <summary>Inference backend that calls the native Rust DLL via FFI (P/Invoke).</summary>
public class NativeBackend : IInferenceBackend
{
    private readonly string _backendName;
    private bool _initialized;
    private bool _disposed;


    private readonly INativeInferenceModule _module;


        /// <summary>Initialises the backend wrapper for the named Rust backend variant.</summary>
    public NativeBackend(string backendName, INativeInferenceModule? module = null)
    {
        _backendName = backendName;
        _module = module ?? new NativeInferenceModuleWrapper();
    }

    /// <inheritdoc/>
    public string Name => $"Native ({_backendName})";

    /// <inheritdoc/>
    public Task<bool> CheckAvailabilityAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            // Check if DLL exists and can initialize
            var result = _module.pcai_init(_backendName);
            if (result == 0)
            {
                _initialized = true;
                return Task.FromResult(true);
            }
            return Task.FromResult(false);
        }
        catch (DllNotFoundException)
        {
            return Task.FromResult(false);
        }
        catch (EntryPointNotFoundException)
        {
            return Task.FromResult(false);
        }
    }

    /// <inheritdoc/>
    public Task<bool> LoadModelAsync(string modelPath, int gpuLayers = -1)
    {
        if (!_initialized)
        {
            var initResult = _module.pcai_init(_backendName);
            if (initResult != 0)
                return Task.FromResult(false);
            _initialized = true;
        }

        var result = _module.pcai_load_model(modelPath, gpuLayers);
        return Task.FromResult(result == 0);
    }

    /// <inheritdoc/>
    public Task<string> GenerateAsync(string prompt, int maxTokens = 2048, float temperature = 0.7f)
    {
        var result = _module.Generate(prompt, (uint)maxTokens, temperature);
        if (result == null)
        {
            var error = _module.GetLastError() ?? "Unknown error";
            throw new InvalidOperationException($"Generation failed: {error}");
        }

        return Task.FromResult(result);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<string> GenerateStreamingAsync(string prompt, int maxTokens = 2048, float temperature = 0.7f)
    {
        var channel = Channel.CreateUnbounded<string>();
        InferenceModule.TokenCallback? callback = null;
        callback = (tokenPtr, _) =>
        {
            if (tokenPtr == IntPtr.Zero) return;
            var token = Marshal.PtrToStringUTF8(tokenPtr);
            if (!string.IsNullOrEmpty(token))
            {
                channel.Writer.TryWrite(token);
            }
        };

        _ = Task.Run(() =>
        {
            var result = _module.pcai_generate_streaming(prompt, (uint)maxTokens, temperature, callback, IntPtr.Zero);
            if (result != 0)
            {
                var error = _module.GetLastError() ?? "Unknown error";
                channel.Writer.TryComplete(new InvalidOperationException($"Streaming failed: {error}"));
                return;
            }
            channel.Writer.TryComplete();
        });

        await foreach (var token in channel.Reader.ReadAllAsync())
        {
            yield return token;
        }
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync()
    {
        if (!_disposed && _initialized)
        {
            _module.pcai_shutdown();
            _disposed = true;
        }
        return ValueTask.CompletedTask;
    }
}

/// <summary>Inference backend that communicates with pcai-inference over the OpenAI-compatible HTTP API.</summary>
public class HttpBackend : IInferenceBackend
{
    private readonly HttpClient _client;
    private readonly string _endpoint;


    /// <summary>Initialises the HTTP backend targeting the given base <paramref name="endpoint"/> URL.</summary>
    public HttpBackend(string endpoint)
    {
        _endpoint = endpoint.TrimEnd('/');
        _client = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
    }

    internal HttpBackend(string endpoint, HttpMessageHandler handler)
    {
        _endpoint = endpoint.TrimEnd('/');
        _client = new HttpClient(handler) { Timeout = TimeSpan.FromMinutes(5) };
    }

    /// <inheritdoc/>
    public string Name => $"HTTP ({_endpoint})";

    /// <inheritdoc/>
    public async Task<bool> CheckAvailabilityAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var response = await _client.GetAsync($"{_endpoint}/v1/models", cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    /// <inheritdoc/>
    public Task<bool> LoadModelAsync(string modelPath, int gpuLayers = -1)
    {
        // HTTP backends auto-load models
        return Task.FromResult(true);
    }

    /// <inheritdoc/>
    public async Task<string> GenerateAsync(string prompt, int maxTokens = 2048, float temperature = 0.7f)
    {
        var request = new
        {
            model = "pcai-inference",
            prompt = prompt,
            stream = false,
            temperature = temperature,
            max_tokens = maxTokens
        };

        var content = new StringContent(JsonSerializer.Serialize(request), System.Text.Encoding.UTF8, "application/json");
        var response = await _client.PostAsync($"{_endpoint}/v1/completions", content);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(json);
        var choices = doc.RootElement.GetProperty("choices");
        if (choices.GetArrayLength() == 0)
        {
            return string.Empty;
        }
        return choices[0].GetProperty("text").GetString() ?? "";
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<string> GenerateStreamingAsync(string prompt, int maxTokens = 2048, float temperature = 0.7f)
    {
        var request = new
        {
            model = "pcai-inference",
            prompt = prompt,
            stream = true,
            temperature = temperature,
            max_tokens = maxTokens
        };

        var content = new StringContent(JsonSerializer.Serialize(request), System.Text.Encoding.UTF8, "application/json");
        var response = await _client.PostAsync($"{_endpoint}/v1/completions", content);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync();
        using var reader = new StreamReader(stream);

        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync();
            if (string.IsNullOrEmpty(line)) continue;

            if (!line.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var payload = line.Substring(5).Trim();
            if (string.Equals(payload, "[DONE]", StringComparison.OrdinalIgnoreCase))
            {
                yield break;
            }

            using var doc = JsonDocument.Parse(payload);
            if (doc.RootElement.TryGetProperty("choices", out var choices) && choices.GetArrayLength() > 0)
            {
                var choice = choices[0];
                if (choice.TryGetProperty("text", out var token))
                {
                    yield return token.GetString() ?? "";
                }
            }
        }
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync()
    {
        _client.Dispose();
        return ValueTask.CompletedTask;
    }
}
