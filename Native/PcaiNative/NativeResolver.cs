using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading;

#nullable enable

namespace PcaiNative
{
    /// <summary>
    /// Shared cross-platform native library import resolver for all PCAI native modules
    /// (<c>pcai_core_lib</c>, <c>pcai_inference</c>, <c>pcai_media</c>).
    /// <see cref="NativeLibrary.SetDllImportResolver"/> allows only one resolver per assembly;
    /// this class consolidates resolution across Windows (<c>.dll</c>), Linux (<c>lib*.so</c>),
    /// and macOS (<c>lib*.dylib</c>) and records resolved artifact provenance.
    /// </summary>
    public static class NativeResolver
    {
        private static int _registered;

        private static readonly ConcurrentDictionary<string, (string Path, string ProvenanceSource)> ResolvedPaths =
            new(StringComparer.OrdinalIgnoreCase);

        private static readonly Dictionary<string, string> ConfigSections = new(StringComparer.OrdinalIgnoreCase)
        {
            ["pcai_core_lib"] = "nativeCore",
            ["pcai_inference"] = "nativeInference",
            ["pcai_media"] = "nativeMedia",
        };

        /// <summary>
        /// Ensures the shared resolver is registered for the <c>PcaiNative</c> assembly.
        /// Safe to call from multiple static constructors concurrently.
        /// </summary>
        public static void EnsureRegistered()
        {
            EnsureRegistered(typeof(NativeResolver).Assembly);
        }

        /// <summary>
        /// Ensures the shared resolver is registered exactly once for the given assembly.
        /// Safe to call from multiple static constructors concurrently.
        /// </summary>
        public static void EnsureRegistered(Assembly assembly)
        {
            ArgumentNullException.ThrowIfNull(assembly);
            if (Interlocked.CompareExchange(ref _registered, 1, 0) == 0)
            {
                try
                {
                    NativeLibrary.SetDllImportResolver(assembly, Resolve);
                }
                catch (InvalidOperationException)
                {
                    // A resolver was already registered for this assembly by an earlier caller.
                }
            }
        }

        /// <summary>
        /// Normalizes a raw P/Invoke library identifier or filename (such as
        /// <c>pcai_core_lib.dll</c> or <c>libpcai_core_lib.so</c>) to its canonical logical stem.
        /// </summary>
        public static string NormalizeLibraryName(string? libraryName)
        {
            if (string.IsNullOrWhiteSpace(libraryName))
                return string.Empty;

            var trimmed = libraryName.Trim();
            var name = Path.GetFileName(trimmed);
            if (string.IsNullOrWhiteSpace(name))
                return string.Empty;

            if (name.EndsWith(".dll", StringComparison.OrdinalIgnoreCase))
                name = name[..^4];
            else if (name.EndsWith(".dylib", StringComparison.OrdinalIgnoreCase))
                name = name[..^6];
            else if (name.EndsWith(".so", StringComparison.OrdinalIgnoreCase))
                name = name[..^3];

            if (name.StartsWith("libpcai_", StringComparison.OrdinalIgnoreCase))
                name = name[3..];

            return name.ToLowerInvariant();
        }

        /// <summary>
        /// Returns the .NET Runtime Identifier (RID) for the current OS and process architecture.
        /// </summary>
        public static string GetCurrentRuntimeIdentifier()
        {
            var platform = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? OSPlatform.Windows
                : RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
                    ? OSPlatform.OSX
                    : OSPlatform.Linux;
            return GetRuntimeIdentifier(platform, RuntimeInformation.ProcessArchitecture);
        }

        /// <summary>
        /// Returns the .NET Runtime Identifier (RID) for the specified OS and architecture.
        /// </summary>
        public static string GetRuntimeIdentifier(OSPlatform platform, Architecture architecture)
        {
            var osPart = platform == OSPlatform.Windows
                ? "win"
                : platform == OSPlatform.OSX
                    ? "osx"
                    : "linux";

            var archPart = architecture switch
            {
                Architecture.X64 => "x64",
                Architecture.Arm64 => "arm64",
                Architecture.X86 => "x86",
                Architecture.Arm => "arm",
                _ => architecture.ToString().ToLowerInvariant(),
            };

            return $"{osPart}-{archPart}";
        }

        /// <summary>
        /// Returns the platform-expected shared library filename for the current operating system.
        /// </summary>
        public static string GetExpectedFileName(string logicalName)
        {
            var platform = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? OSPlatform.Windows
                : RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
                    ? OSPlatform.OSX
                    : OSPlatform.Linux;
            return GetExpectedFileName(logicalName, platform);
        }

        /// <summary>
        /// Returns the platform-expected shared library filename for the specified operating system:
        /// Windows <c>&lt;name&gt;.dll</c>, Linux <c>lib&lt;name&gt;.so</c>, macOS <c>lib&lt;name&gt;.dylib</c>.
        /// </summary>
        public static string GetExpectedFileName(string logicalName, OSPlatform platform)
        {
            var stem = NormalizeLibraryName(logicalName);
            if (string.IsNullOrEmpty(stem))
                throw new ArgumentException("Library name cannot be null or whitespace.", nameof(logicalName));

            if (platform == OSPlatform.Windows)
                return $"{stem}.dll";
            if (platform == OSPlatform.OSX)
                return $"lib{stem}.dylib";
            return $"lib{stem}.so";
        }

        /// <summary>
        /// Gets the resolved absolute path of a native library if it has been resolved or can be resolved now.
        /// </summary>
        public static string? GetResolvedLibraryPath(string libraryName)
        {
            var logical = NormalizeLibraryName(libraryName);
            if (string.IsNullOrEmpty(logical))
                return null;

            if (ResolvedPaths.TryGetValue(logical, out var recorded))
                return recorded.Path;

            return TryResolveLibraryPath(logical, out var resolvedPath, out _) ? resolvedPath : null;
        }

        /// <summary>
        /// Gets the provenance source label for a resolved native library.
        /// </summary>
        public static string? GetResolvedProvenance(string libraryName)
        {
            var logical = NormalizeLibraryName(libraryName);
            if (string.IsNullOrEmpty(logical))
                return null;

            if (ResolvedPaths.TryGetValue(logical, out var recorded))
                return recorded.ProvenanceSource;

            return TryResolveLibraryPath(logical, out _, out var provenance) ? provenance : null;
        }

        /// <summary>
        /// Resolves the candidate filesystem path for a native library without loading it,
        /// returning the matched path and provenance label.
        /// </summary>
        public static bool TryResolveLibraryPath(
            string libraryName,
            out string? resolvedPath,
            out string? provenanceSource,
            Assembly? assembly = null)
        {
            resolvedPath = null;
            provenanceSource = null;

            var logical = NormalizeLibraryName(libraryName);
            if (!ConfigSections.TryGetValue(logical, out var configSection))
                return false;

            var targetAssembly = assembly ?? typeof(NativeResolver).Assembly;
            var expectedFileName = GetExpectedFileName(logical);
            var rid = GetCurrentRuntimeIdentifier();

            // 1. Explicit environment variable file overrides (authoritative when set).
            if (string.Equals(logical, "pcai_core_lib", StringComparison.OrdinalIgnoreCase))
            {
                foreach (var envVar in new[] { "PCAI_CORE_LIB_PATH", "PCAI_TEST_CORE_DLL" })
                {
                    var envValue = Environment.GetEnvironmentVariable(envVar);
                    if (!string.IsNullOrWhiteSpace(envValue))
                    {
                        var candidate = Directory.Exists(envValue)
                            ? Path.Combine(envValue, expectedFileName)
                            : envValue;
                        if (File.Exists(candidate))
                        {
                            resolvedPath = Path.GetFullPath(candidate);
                            provenanceSource = $"env:{envVar}";
                            return true;
                        }

                        // Fail closed when an explicit file override is set to a non-existent path.
                        provenanceSource = $"env:{envVar}:missing";
                        return false;
                    }
                }
            }

            var nativeDirOverride = Environment.GetEnvironmentVariable("PCAI_NATIVE_DIR");
            if (!string.IsNullOrWhiteSpace(nativeDirOverride))
            {
                var candidate = Path.Combine(nativeDirOverride, expectedFileName);
                if (File.Exists(candidate))
                {
                    resolvedPath = Path.GetFullPath(candidate);
                    provenanceSource = "env:PCAI_NATIVE_DIR";
                    return true;
                }

                provenanceSource = "env:PCAI_NATIVE_DIR:missing";
                return false;
            }

            // Build prioritised candidate list with provenance labels.
            var candidates = new List<(string Path, string Provenance)>();

            // 2. Config-driven search paths from Config/llm-config.json
            foreach (var configCandidate in LoadConfigSearchPaths(targetAssembly, configSection))
            {
                if (Directory.Exists(configCandidate))
                    candidates.Add((Path.Combine(configCandidate, expectedFileName), "config:llm-config.json"));
                else
                    candidates.Add((configCandidate, "config:llm-config.json"));
            }

            // 3. Same directory as PcaiNative.dll and runtimes/<rid>/native
            var assemblyDir = Path.GetDirectoryName(targetAssembly.Location);
            if (!string.IsNullOrEmpty(assemblyDir))
            {
                candidates.Add((Path.Combine(assemblyDir, expectedFileName), "assembly-dir"));
                candidates.Add((Path.Combine(assemblyDir, "runtimes", rid, "native", expectedFileName), $"assembly-runtimes:{rid}"));
            }

            // 4. AppContext.BaseDirectory and runtimes/<rid>/native
            if (!string.IsNullOrEmpty(AppContext.BaseDirectory))
            {
                candidates.Add((Path.Combine(AppContext.BaseDirectory, expectedFileName), "appcontext-base"));
                candidates.Add((Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native", expectedFileName), $"appcontext-runtimes:{rid}"));
            }

            // 5. Repository root outputs (bin/, runtimes/<rid>/native, Native/pcai_core/target/release, target/release)
            var repoRoot = FindRepoRoot(targetAssembly);
            if (!string.IsNullOrEmpty(repoRoot))
            {
                candidates.Add((Path.Combine(repoRoot, "bin", expectedFileName), "repo:bin"));
                candidates.Add((Path.Combine(repoRoot, "bin", "runtimes", rid, "native", expectedFileName), $"repo:bin/runtimes/{rid}"));
                candidates.Add((Path.Combine(repoRoot, "Native", "PcaiNative", "runtimes", rid, "native", expectedFileName), $"repo:PcaiNative/runtimes/{rid}"));
                candidates.Add((Path.Combine(repoRoot, ".pcai", "build", "artifacts", "pcai-native", expectedFileName), "repo:artifacts/pcai-native"));
                candidates.Add((Path.Combine(repoRoot, "Native", "pcai_core", "target", "release", expectedFileName), "repo:Native/pcai_core/target/release"));
                candidates.Add((Path.Combine(repoRoot, "target", "release", expectedFileName), "repo:target/release"));
                candidates.Add((Path.Combine(repoRoot, "Native", "pcai_core", "target", "debug", expectedFileName), "repo:Native/pcai_core/target/debug"));
                candidates.Add((Path.Combine(repoRoot, "target", "debug", expectedFileName), "repo:target/debug"));
            }

            // 6. Common user-local bin locations
            var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            if (!string.IsNullOrEmpty(userProfile))
            {
                candidates.Add((Path.Combine(userProfile, ".local", "bin", expectedFileName), "user:.local/bin"));
                candidates.Add((Path.Combine(userProfile, "PC_AI", "bin", expectedFileName), "user:PC_AI/bin"));
            }

            foreach (var (candidatePath, source) in candidates)
            {
                if (File.Exists(candidatePath))
                {
                    resolvedPath = Path.GetFullPath(candidatePath);
                    provenanceSource = source;
                    return true;
                }
            }

            return false;
        }

        private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            var logical = NormalizeLibraryName(libraryName);
            if (!ConfigSections.ContainsKey(logical))
                return IntPtr.Zero;

            if (TryResolveLibraryPath(logical, out var resolvedPath, out var provenance, assembly) &&
                !string.IsNullOrEmpty(resolvedPath))
            {
                if (NativeLibrary.TryLoad(resolvedPath, out var handle))
                {
                    ResolvedPaths[logical] = (resolvedPath, provenance ?? "resolved");
                    return handle;
                }

                // If an explicit override environment variable was provided and failed to load, fail closed.
                if (provenance is not null && provenance.StartsWith("env:", StringComparison.Ordinal))
                    return IntPtr.Zero;
            }

            // If an explicit override pointed to a missing path, do not fall back to system/PATH libraries.
            if (provenance is not null && provenance.EndsWith(":missing", StringComparison.Ordinal))
                return IntPtr.Zero;

            if (NativeLibrary.TryLoad(logical, assembly, searchPath, out var defaultHandle))
            {
                ResolvedPaths[logical] = (logical, "default-loader");
                return defaultHandle;
            }

            return IntPtr.Zero;
        }

        private static IEnumerable<string> LoadConfigSearchPaths(Assembly assembly, string sectionName)
        {
            var configPath = FindConfigPath(assembly);
            if (string.IsNullOrEmpty(configPath))
                return Array.Empty<string>();

            var projectRoot = Directory.GetParent(Path.GetDirectoryName(configPath) ?? string.Empty)?.FullName;
            if (string.IsNullOrEmpty(projectRoot))
                return Array.Empty<string>();

            try
            {
                var results = new List<string>();
                using var doc = JsonDocument.Parse(File.ReadAllText(configPath));

                if (!doc.RootElement.TryGetProperty(sectionName, out var section))
                    return results;

                if (!section.TryGetProperty("dllSearchPaths", out var paths) || paths.ValueKind != JsonValueKind.Array)
                    return results;

                foreach (var element in paths.EnumerateArray())
                {
                    if (element.ValueKind != JsonValueKind.String)
                        continue;

                    var path = element.GetString();
                    if (string.IsNullOrWhiteSpace(path))
                        continue;

                    results.Add(Path.IsPathRooted(path)
                        ? path
                        : Path.Combine(projectRoot, path));
                }

                return results;
            }
            catch
            {
                return Array.Empty<string>();
            }
        }

        private static string? FindRepoRoot(Assembly assembly)
        {
            var envRoot = Environment.GetEnvironmentVariable("PCAI_ROOT");
            if (!string.IsNullOrWhiteSpace(envRoot) && Directory.Exists(envRoot))
                return Path.GetFullPath(envRoot);

            foreach (var startDir in new[] { Path.GetDirectoryName(assembly.Location), AppContext.BaseDirectory })
            {
                var current = startDir;
                for (var i = 0; i < 8 && !string.IsNullOrEmpty(current); i++)
                {
                    if (File.Exists(Path.Combine(current, "PC-AI.ps1")) ||
                        Directory.Exists(Path.Combine(current, "Native", "pcai_core")))
                    {
                        return current;
                    }

                    current = Directory.GetParent(current)?.FullName;
                }
            }

            return null;
        }

        private static string? FindConfigPath(Assembly assembly)
        {
            var repoRoot = FindRepoRoot(assembly);
            if (!string.IsNullOrEmpty(repoRoot))
            {
                var repoConfig = Path.Combine(repoRoot, "Config", "llm-config.json");
                if (File.Exists(repoConfig))
                    return repoConfig;
            }

            var current = Path.GetDirectoryName(assembly.Location);
            for (var i = 0; i < 6 && !string.IsNullOrEmpty(current); i++)
            {
                var candidate = Path.Combine(current, "Config", "llm-config.json");
                if (File.Exists(candidate))
                    return candidate;

                current = Directory.GetParent(current)?.FullName;
            }
            return null;
        }
    }
}
