using System;
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
    /// Shared DLL import resolver for all native modules (pcai_inference, pcai_media, etc.).
    /// <see cref="NativeLibrary.SetDllImportResolver"/> allows only one resolver per assembly;
    /// this class consolidates resolution for all native DLLs so that both
    /// <see cref="InferenceModule"/> and <see cref="MediaModule"/> are handled correctly.
    /// </summary>
    internal static class NativeResolver
    {
        private static int _registered;

        /// <summary>
        /// Ensures the shared resolver is registered exactly once for the given assembly.
        /// Safe to call from multiple static constructors concurrently.
        /// </summary>
        internal static void EnsureRegistered(Assembly assembly)
        {
            if (Interlocked.CompareExchange(ref _registered, 1, 0) == 0)
                NativeLibrary.SetDllImportResolver(assembly, Resolve);
        }

        private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            return libraryName switch
            {
                "pcai_inference" => ResolveLibrary(libraryName, "nativeInference", "pcai_inference.dll", assembly, searchPath),
                "pcai_media" => ResolveLibrary(libraryName, "nativeMedia", "pcai_media.dll", assembly, searchPath),
                _ => IntPtr.Zero
            };
        }

        private static IntPtr ResolveLibrary(
            string libraryName,
            string configSection,
            string dllFileName,
            Assembly assembly,
            DllImportSearchPath? searchPath)
        {
            // Try default resolution first (handles PATH, system directories, etc.)
            if (NativeLibrary.TryLoad(libraryName, assembly, searchPath, out IntPtr handle))
                return handle;

            // Build prioritised candidate list
            var candidates = new List<string>();

            // 0. Config-driven search paths from Config/llm-config.json
            foreach (var path in LoadConfigSearchPaths(assembly, configSection))
                candidates.Add(path);

            // 1. Runtime native folder deployed by the build script
            var assemblyDir = Path.GetDirectoryName(assembly.Location);
            if (!string.IsNullOrEmpty(assemblyDir))
            {
                candidates.Add(Path.Combine(assemblyDir, "runtimes", "win-x64", "native", dllFileName));
                candidates.Add(Path.Combine(assemblyDir, dllFileName));
            }

            // 2. AppContext base directory
            candidates.Add(Path.Combine(AppContext.BaseDirectory, "runtimes", "win-x64", "native", dllFileName));
            candidates.Add(Path.Combine(AppContext.BaseDirectory, dllFileName));

            // 3. Common project locations relative to %USERPROFILE%
            var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            candidates.Add(Path.Combine(userProfile, "PC_AI", "bin", dllFileName));
            candidates.Add(Path.Combine(userProfile, ".local", "bin", dllFileName));

            foreach (var path in candidates)
            {
                if (File.Exists(path) && NativeLibrary.TryLoad(path, out handle))
                    return handle;
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

        private static string? FindConfigPath(Assembly assembly)
        {
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
