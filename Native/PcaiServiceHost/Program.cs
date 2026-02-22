using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using PcaiNative;

namespace PcaiServiceHost;

public static class Program
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true
    };

    private static readonly JsonSerializerOptions JsonWriteOptions = new()
    {
        WriteIndented = true
    };

    public static async Task<int> Main(string[] args)
    {
        if (args.Length == 0)
        {
            PrintUsage();
            return 1;
        }

        var command = args[0].ToLowerInvariant();
        var remaining = args.Skip(1).ToArray();

        switch (command)
        {
            case "vllm":
                return await HandleVllmAsync(remaining);
            case "hvsock":
                return await HandleHvsockAsync(remaining);
            case "provider":
                return HandleProvider(remaining);
            case "inference":
                return HandleInference(remaining);
            default:
                PrintUsage();
                return 1;
        }
    }

    private static void PrintUsage()
    {
        Console.WriteLine("PcaiServiceHost");
        Console.WriteLine("Usage:");
        Console.WriteLine("  pcai-servicehost vllm start|stop|restart|status|ensure [--compose <path>] [--interval <sec>]");
        Console.WriteLine("  pcai-servicehost hvsock start|stop|status|run [--config <path>] [--state <path>]");
        Console.WriteLine("  pcai-servicehost provider show|set-order <comma-separated>");
        Console.WriteLine("  pcai-servicehost inference status|init|load|generate|shutdown [options]");
    }

    private static async Task<int> HandleVllmAsync(string[] args)
    {
        if (args.Length == 0)
        {
            PrintUsage();
            return 1;
        }

        var action = args[0].ToLowerInvariant();
        var composePath = GetArg(args, "--compose") ?? @"C:\Users\david\PC_AI\Deploy\docker\vllm\docker-compose.yml";
        var intervalSec = int.TryParse(GetArg(args, "--interval"), out var parsed) ? parsed : 15;

        return action switch
        {
            "start" => RunDockerCompose(new[] { "-f", composePath, "up", "-d" }),
            "stop" => RunDockerCompose(new[] { "-f", composePath, "down" }),
            "restart" => RunDockerCompose(new[] { "-f", composePath, "down" }) + RunDockerCompose(new[] { "-f", composePath, "up", "-d" }),
            "status" => PrintVllmStatus(),
            "ensure" => await EnsureVllmAsync(composePath, intervalSec),
            _ => 1
        };
    }

    private static int RunDockerCompose(string[] args)
    {
        var docker = FindExecutable("docker.exe");
        if (docker == null)
        {
            Console.Error.WriteLine("Docker not found in PATH.");
            return 2;
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = docker,
            ArgumentList = { "compose" },
            UseShellExecute = false
        };

        foreach (var arg in args)
        {
            startInfo.ArgumentList.Add(arg);
        }

        using var proc = Process.Start(startInfo);
        proc?.WaitForExit();
        return proc?.ExitCode ?? 1;
    }

    private static int PrintVllmStatus()
    {
        var docker = FindExecutable("docker.exe");
        if (docker == null)
        {
            Console.Error.WriteLine("Docker not found in PATH.");
            return 2;
        }

        var output = RunProcessCapture(docker, "ps", "--filter", "name=vllm-functiongemma", "--format", "{{.Status}}");
        if (string.IsNullOrWhiteSpace(output))
        {
            Console.WriteLine("vLLM container not running");
            return 1;
        }

        Console.WriteLine($"vLLM container status: {output.Trim()}");
        return 0;
    }

    private static async Task<int> EnsureVllmAsync(string composePath, int intervalSec)
    {
        Console.WriteLine("Ensuring vLLM container stays running...");
        while (true)
        {
            var status = PrintVllmStatus();
            if (status != 0)
            {
                RunDockerCompose(new[] { "-f", composePath, "up", "-d" });
            }
            await Task.Delay(TimeSpan.FromSeconds(intervalSec));
        }
    }

    private static async Task<int> HandleHvsockAsync(string[] args)
    {
        if (args.Length == 0)
        {
            PrintUsage();
            return 1;
        }

        var action = args[0].ToLowerInvariant();
        var configPath = GetArg(args, "--config") ?? @"C:\Users\david\PC_AI\Config\hvsock-proxy.conf";
        var statePath = GetArg(args, "--state") ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), "PC_AI", "hvsock-proxy", "state.json");

        return action switch
        {
            "start" => StartHvsockProxy(configPath, statePath),
            "stop" => StopHvsockProxy(statePath),
            "status" => PrintHvsockStatus(statePath),
            "run" => await RunHvsockProxyLoop(configPath, statePath),
            _ => 1
        };
    }

    private static int StartHvsockProxy(string configPath, string statePath)
    {
        var winsocat = FindExecutable("winsocat.exe");
        if (winsocat == null)
        {
            Console.Error.WriteLine("winsocat.exe not found in PATH.");
            return 2;
        }

        var entries = LoadHvsockConfig(configPath);
        if (entries.Count == 0)
        {
            Console.Error.WriteLine("No hvsock entries found.");
            return 1;
        }

        var state = new List<HvsockProxyStateEntry>();
        foreach (var entry in entries)
        {
            var args = $"HVSock-LISTEN:{entry.ServiceId} TCP:{entry.TcpHost}:{entry.TcpPort}";
            var proc = Process.Start(new ProcessStartInfo
            {
                FileName = winsocat,
                Arguments = args,
                UseShellExecute = false,
                CreateNoWindow = true
            });

            if (proc == null)
            {
                Console.Error.WriteLine($"Failed to start winsocat for {entry.Name}");
                continue;
            }

            state.Add(new HvsockProxyStateEntry
            {
                Name = entry.Name,
                ServiceId = entry.ServiceId,
                TcpTarget = $"{entry.TcpHost}:{entry.TcpPort}",
                Pid = proc.Id,
                Started = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
            });
        }

        SaveState(statePath, state);
        Console.WriteLine($"Started {state.Count} HVSOCK proxies.");
        return 0;
    }

    private static int StopHvsockProxy(string statePath)
    {
        if (!File.Exists(statePath))
        {
            Console.WriteLine("No state file found.");
            return 0;
        }

        var state = JsonSerializer.Deserialize<List<HvsockProxyStateEntry>>(File.ReadAllText(statePath), JsonOptions) ?? new();
        var stopped = 0;
        foreach (var entry in state)
        {
            try
            {
                var proc = Process.GetProcessById(entry.Pid);
                proc.Kill(true);
                stopped++;
            }
            catch
            {
                // ignore
            }
        }

        File.Delete(statePath);
        Console.WriteLine($"Stopped {stopped} HVSOCK proxies.");
        return 0;
    }

    private static int PrintHvsockStatus(string statePath)
    {
        if (!File.Exists(statePath))
        {
            Console.WriteLine("No HVSOCK proxies running.");
            return 1;
        }

        var state = JsonSerializer.Deserialize<List<HvsockProxyStateEntry>>(File.ReadAllText(statePath), JsonOptions) ?? new();
        var running = 0;
        foreach (var entry in state)
        {
            var alive = true;
            try
            {
                Process.GetProcessById(entry.Pid);
            }
            catch
            {
                alive = false;
            }

            Console.WriteLine($"{entry.Name}: {(alive ? "RUNNING" : "STOPPED")} pid={entry.Pid} target={entry.TcpTarget}");
            if (alive) running++;
        }

        Console.WriteLine($"Active: {running}/{state.Count}");
        return running == state.Count ? 0 : 1;
    }

    private static async Task<int> RunHvsockProxyLoop(string configPath, string statePath)
    {
        Console.WriteLine("Running HVSOCK proxy supervisor...");
        while (true)
        {
            var status = PrintHvsockStatus(statePath);
            if (status != 0)
            {
                StartHvsockProxy(configPath, statePath);
            }
            await Task.Delay(TimeSpan.FromSeconds(15));
        }
    }

    private static int HandleProvider(string[] args)
    {
        if (args.Length == 0)
        {
            PrintUsage();
            return 1;
        }

        var action = args[0].ToLowerInvariant();
        var configPath = @"C:\Users\david\PC_AI\Config\llm-config.json";

        if (!File.Exists(configPath))
        {
            Console.Error.WriteLine("llm-config.json not found.");
            return 2;
        }

        var configNode = JsonNode.Parse(File.ReadAllText(configPath)) as JsonObject;
        if (configNode == null)
        {
            Console.Error.WriteLine("Failed to parse llm-config.json.");
            return 2;
        }

        if (action == "show")
        {
            if (configNode.TryGetPropertyValue("fallbackOrder", out var value) && value != null)
            {
                Console.WriteLine(value.ToJsonString(JsonWriteOptions));
                return 0;
            }

            Console.WriteLine("fallbackOrder not set.");
            return 1;
        }

        if (action == "set-order" && args.Length > 1)
        {
            var order = args[1].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToArray();
            var orderArray = new JsonArray(order.Select(item => JsonValue.Create(item)).ToArray());
            configNode["fallbackOrder"] = orderArray;
            File.WriteAllText(configPath, configNode.ToJsonString(JsonWriteOptions));
            Console.WriteLine("Updated fallbackOrder.");
            return 0;
        }

        return 1;
    }

    private static int HandleInference(string[] args)
    {
        if (args.Length == 0)
        {
            PrintUsage();
            return 1;
        }

        var action = args[0].ToLowerInvariant();
        switch (action)
        {
            case "status":
                return PrintInferenceStatus();
            case "init":
                return InitInference(args.Skip(1).ToArray());
            case "load":
                return LoadInferenceModel(args.Skip(1).ToArray());
            case "generate":
                return GenerateInference(args.Skip(1).ToArray());
            case "shutdown":
                InferenceModule.pcai_shutdown();
                Console.WriteLine("Inference backend shutdown.");
                return 0;
            default:
                PrintUsage();
                return 1;
        }
    }

    private static int PrintInferenceStatus()
    {
        var status = new InferenceStatus
        {
            DllAvailable = InferenceModule.IsAvailable,
            Initialized = SafeBoolCall(() => InferenceModule.pcai_is_initialized() != 0),
            ModelLoaded = SafeBoolCall(() => InferenceModule.pcai_is_model_loaded() != 0),
            Backend = GetBackendName(),
            LastError = InferenceModule.GetLastError()
        };

        Console.WriteLine(JsonSerializer.Serialize(status, JsonOptions));
        return status.DllAvailable ? 0 : 1;
    }

    private static int InitInference(string[] args)
    {
        var backend = GetArg(args, "--backend") ?? "auto";
        if (string.Equals(backend, "auto", StringComparison.OrdinalIgnoreCase))
        {
            if (TryInitBackend("mistralrs") || TryInitBackend("llamacpp"))
            {
                return 0;
            }

            Console.Error.WriteLine("Failed to initialize any backend.");
            return 1;
        }

        if (!TryInitBackend(backend))
        {
            Console.Error.WriteLine($"Failed to initialize backend: {backend}");
            return 1;
        }

        return 0;
    }

    private static bool TryInitBackend(string backend)
    {
        var result = InferenceModule.pcai_init(backend);
        if (result == 0)
        {
            Console.WriteLine($"Initialized backend: {backend}");
            return true;
        }

        var error = InferenceModule.GetLastError() ?? "Unknown error";
        Console.Error.WriteLine($"Init failed for {backend}: {error}");
        return false;
    }

    private static int LoadInferenceModel(string[] args)
    {
        var modelPath = GetArg(args, "--model-path");
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            Console.Error.WriteLine("Missing --model-path for inference load.");
            return 1;
        }

        var gpuLayers = int.TryParse(GetArg(args, "--gpu-layers"), out var parsed) ? parsed : -1;

        var result = InferenceModule.pcai_load_model(modelPath, gpuLayers);
        if (result == 0)
        {
            Console.WriteLine($"Model loaded: {modelPath}");
            return 0;
        }

        var error = InferenceModule.GetLastError() ?? "Unknown error";
        Console.Error.WriteLine($"Model load failed: {error}");
        return 1;
    }

    private static int GenerateInference(string[] args)
    {
        var prompt = GetArg(args, "--prompt") ?? string.Empty;
        if (string.IsNullOrWhiteSpace(prompt))
        {
            Console.Error.WriteLine("Missing --prompt for inference generate.");
            return 1;
        }

        var maxTokens = uint.TryParse(GetArg(args, "--max-tokens"), out var parsedTokens) ? parsedTokens : 512u;
        var temperature = float.TryParse(GetArg(args, "--temperature"), out var parsedTemp) ? parsedTemp : 0.7f;

        var result = InferenceModule.Generate(prompt, maxTokens, temperature);
        if (result == null)
        {
            var error = InferenceModule.GetLastError() ?? "Unknown error";
            Console.Error.WriteLine($"Generate failed: {error}");
            return 1;
        }

        Console.WriteLine(result);
        return 0;
    }

    private static string? GetArg(string[] args, string name)
    {
        for (var i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
            {
                return args[i + 1];
            }
        }
        return null;
    }

    private static bool SafeBoolCall(Func<bool> fn)
    {
        try { return fn(); }
        catch { return false; }
    }

    private static string? GetBackendName()
    {
        try
        {
            var ptr = InferenceModule.pcai_get_backend_name();
            return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
        }
        catch
        {
            return null;
        }
    }

    private static string? FindExecutable(string exe)
    {
        var paths = (Environment.GetEnvironmentVariable("PATH") ?? string.Empty).Split(';', StringSplitOptions.RemoveEmptyEntries);
        foreach (var path in paths)
        {
            var candidate = Path.Combine(path.Trim(), exe);
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }
        return null;
    }

    private static string RunProcessCapture(string file, params string[] args)
    {
        var psi = new ProcessStartInfo
        {
            FileName = file,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };

        foreach (var arg in args)
        {
            psi.ArgumentList.Add(arg);
        }

        using var proc = Process.Start(psi);
        if (proc == null)
        {
            return string.Empty;
        }

        var output = proc.StandardOutput.ReadToEnd();
        proc.WaitForExit();
        return output;
    }

    private static List<HvsockProxyEntry> LoadHvsockConfig(string path)
    {
        var entries = new List<HvsockProxyEntry>();
        foreach (var raw in File.ReadAllLines(path))
        {
            var line = raw.Trim();
            if (string.IsNullOrWhiteSpace(line) || line.StartsWith('#'))
            {
                continue;
            }

            var parts = line.Split(':');
            if (parts.Length < 4)
            {
                continue;
            }

            entries.Add(new HvsockProxyEntry
            {
                Name = parts[0],
                ServiceId = parts[1],
                TcpHost = parts[2],
                TcpPort = parts[3]
            });
        }
        return entries;
    }

    private static void SaveState(string path, List<HvsockProxyStateEntry> state)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(dir))
        {
            Directory.CreateDirectory(dir);
        }

        File.WriteAllText(path, JsonSerializer.Serialize(state, JsonOptions));
    }
}

public sealed class HvsockProxyEntry
{
    public string Name { get; set; } = string.Empty;
    public string ServiceId { get; set; } = string.Empty;
    public string TcpHost { get; set; } = string.Empty;
    public string TcpPort { get; set; } = string.Empty;
}

public sealed class HvsockProxyStateEntry
{
    public string Name { get; set; } = string.Empty;
    public string ServiceId { get; set; } = string.Empty;
    public string TcpTarget { get; set; } = string.Empty;
    public int Pid { get; set; }
    public string Started { get; set; } = string.Empty;
}

public sealed class InferenceStatus
{
    public bool DllAvailable { get; set; }
    public bool Initialized { get; set; }
    public bool ModelLoaded { get; set; }
    public string? Backend { get; set; }
    public string? LastError { get; set; }
}
