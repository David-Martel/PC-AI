using System;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Collections.Generic;
using System.IO;

namespace PcaiNative
{
    /// <summary>
    /// P/Invoke interop with pcai_inference.dll
    /// </summary>
    public static class InferenceModule
    {
        private const string DllName = "pcai_inference";

        /// <summary>
        /// Static constructor to register native library resolver
        /// </summary>
        static InferenceModule()
        {
            NativeLibrary.SetDllImportResolver(typeof(InferenceModule).Assembly, ResolveDll);
        }

        /// <summary>
        /// Custom DLL resolver for pcai_inference.dll
        /// </summary>
        private static IntPtr ResolveDll(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (libraryName != DllName)
                return IntPtr.Zero;

            // Try default resolution first
            if (NativeLibrary.TryLoad(libraryName, assembly, searchPath, out IntPtr handle))
                return handle;

            // Search paths in priority order
            var searchPaths = new List<string>();

            // 0. Configured search paths from Config/llm-config.json
            foreach (var path in LoadConfigSearchPaths(assembly))
            {
                searchPaths.Add(path);
            }

            // 1. Runtime native folder (deployed by build script)
            var assemblyDir = Path.GetDirectoryName(assembly.Location);
            if (!string.IsNullOrEmpty(assemblyDir))
            {
                searchPaths.Add(Path.Combine(assemblyDir, "runtimes", "win-x64", "native", "pcai_inference.dll"));
                searchPaths.Add(Path.Combine(assemblyDir, "pcai_inference.dll"));
            }

            // 2. Common project locations
            var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            searchPaths.Add(Path.Combine(userProfile, "PC_AI", "bin", "pcai_inference.dll"));
            searchPaths.Add(Path.Combine(userProfile, ".local", "bin", "pcai_inference.dll"));
            searchPaths.Add(Path.Combine(userProfile, "PC_AI", "Native", "pcai_core", "pcai_inference", "target", "release", "pcai_inference.dll"));
            searchPaths.Add(Path.Combine(userProfile, "PC_AI", "Deploy", "pcai-inference", "target", "release", "pcai_inference.dll"));

            foreach (var path in searchPaths)
            {
                if (File.Exists(path) && NativeLibrary.TryLoad(path, out handle))
                    return handle;
            }

            return IntPtr.Zero;
        }

        private static IEnumerable<string> LoadConfigSearchPaths(Assembly assembly)
        {
            var configPath = FindConfigPath(assembly);
            if (string.IsNullOrEmpty(configPath))
            {
                yield break;
            }

            var projectRoot = Directory.GetParent(Path.GetDirectoryName(configPath) ?? string.Empty)?.FullName;
            if (string.IsNullOrEmpty(projectRoot))
            {
                yield break;
            }

            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
                if (!doc.RootElement.TryGetProperty("nativeInference", out var nativeInference))
                {
                    yield break;
                }
                if (!nativeInference.TryGetProperty("dllSearchPaths", out var paths) || paths.ValueKind != JsonValueKind.Array)
                {
                    yield break;
                }

                foreach (var element in paths.EnumerateArray())
                {
                    if (element.ValueKind != JsonValueKind.String)
                    {
                        continue;
                    }
                    var path = element.GetString();
                    if (string.IsNullOrWhiteSpace(path))
                    {
                        continue;
                    }
                    if (Path.IsPathRooted(path))
                    {
                        yield return path;
                    }
                    else
                    {
                        yield return Path.Combine(projectRoot, path);
                    }
                }
            }
            catch
            {
                yield break;
            }
        }

        private static string? FindConfigPath(Assembly assembly)
        {
            var start = Path.GetDirectoryName(assembly.Location);
            var current = start;
            for (var i = 0; i < 6 && !string.IsNullOrEmpty(current); i++)
            {
                var candidate = Path.Combine(current, "Config", "llm-config.json");
                if (File.Exists(candidate))
                {
                    return candidate;
                }
                current = Directory.GetParent(current)?.FullName;
            }
            return null;
        }

        #region Native Imports

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int pcai_init([MarshalAs(UnmanagedType.LPUTF8Str)] string backendName);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int pcai_load_model([MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath, int gpuLayers);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr pcai_generate([MarshalAs(UnmanagedType.LPUTF8Str)] string prompt, uint maxTokens, float temperature);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int pcai_generate_streaming([MarshalAs(UnmanagedType.LPUTF8Str)] string prompt, uint maxTokens, float temperature, TokenCallback callback, IntPtr userData);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pcai_shutdown();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr pcai_last_error();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pcai_last_error_code();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pcai_free_string(IntPtr ptr);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pcai_is_initialized();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pcai_is_model_loaded();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr pcai_get_backend_name();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void TokenCallback(IntPtr token, IntPtr userData);

        #endregion

        #region High-level Wrappers

        public static bool IsAvailable
        {
            get
            {
                try
                {
                    return pcai_is_initialized() != 0 || IsDllAvailable();
                }
                catch (EntryPointNotFoundException)
                {
                    return IsDllAvailable();
                }
                catch (DllNotFoundException)
                {
                    return false;
                }
            }
        }

        private static bool IsDllAvailable()
        {
            try { return pcai_last_error_code() >= 0; }
            catch { return false; }
        }

        public static string? Generate(string prompt, uint maxTokens = 512, float temperature = 0.7f)
        {
            var ptr = pcai_generate(prompt, maxTokens, temperature);
            if (ptr == IntPtr.Zero) return null;
            try { return Marshal.PtrToStringUTF8(ptr); }
            finally { pcai_free_string(ptr); }
        }

        public static string? GetLastError()
        {
            var ptr = pcai_last_error();
            return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
        }

        #endregion
    }
}
