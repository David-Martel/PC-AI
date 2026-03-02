using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

#nullable enable

namespace PcaiNative
{
    /// <summary>
    /// P/Invoke interop with pcai_media.dll (Janus-Pro multimodal inference).
    /// Provides image generation and visual understanding capabilities backed by
    /// the Janus-Pro model running natively via the pcai_media Rust library.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The DLL resolver searches the following locations in priority order:
    /// <list type="number">
    ///   <item>Paths listed under <c>nativeMedia.dllSearchPaths</c> in <c>Config/llm-config.json</c>.</item>
    ///   <item>The <c>runtimes/win-x64/native/</c> sub-folder next to the calling assembly.</item>
    ///   <item>The calling assembly's directory.</item>
    ///   <item>Common project-relative locations under <c>%USERPROFILE%</c>.</item>
    /// </list>
    /// </para>
    /// <para>
    /// All native calls use <see cref="CallingConvention.Cdecl"/>.
    /// Heap-allocated string pointers returned by the native layer <b>must</b> be
    /// freed with <see cref="pcai_media_free_string"/>; the high-level wrappers
    /// handle this automatically.
    /// </para>
    /// </remarks>
    public static class MediaModule
    {
        private const string DllName = "pcai_media";

        /// <summary>
        /// Static constructor — registers the custom DLL import resolver for this assembly
        /// so that <c>pcai_media.dll</c> can be located from non-standard paths at runtime.
        /// </summary>
        static MediaModule()
        {
            NativeLibrary.SetDllImportResolver(typeof(MediaModule).Assembly, ResolveDll);
        }

        /// <summary>
        /// Custom DLL resolver for <c>pcai_media.dll</c>.
        /// Called by the .NET runtime before the default OS search is attempted.
        /// </summary>
        /// <param name="libraryName">The DLL name requested by a <see cref="DllImportAttribute"/>.</param>
        /// <param name="assembly">The assembly that contains the import declaration.</param>
        /// <param name="searchPath">The default search path hint supplied by the runtime.</param>
        /// <returns>
        /// A valid native library handle, or <see cref="IntPtr.Zero"/> to fall through to the
        /// default OS loader.
        /// </returns>
        private static IntPtr ResolveDll(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (libraryName != DllName)
                return IntPtr.Zero;

            // Try default resolution first (handles PATH, system directories, etc.)
            if (NativeLibrary.TryLoad(libraryName, assembly, searchPath, out IntPtr handle))
                return handle;

            // Build prioritised candidate list.
            var candidates = new List<string>();

            // 0. Config-driven search paths from Config/llm-config.json
            foreach (var path in LoadConfigSearchPaths(assembly))
                candidates.Add(path);

            // 1. Runtime native folder deployed by the build script
            var assemblyDir = Path.GetDirectoryName(assembly.Location);
            if (!string.IsNullOrEmpty(assemblyDir))
            {
                candidates.Add(Path.Combine(assemblyDir, "runtimes", "win-x64", "native", "pcai_media.dll"));
                candidates.Add(Path.Combine(assemblyDir, "pcai_media.dll"));
            }

            // 2. AppContext base directory (used by single-file publish and some hosts)
            candidates.Add(Path.Combine(AppContext.BaseDirectory, "runtimes", "win-x64", "native", "pcai_media.dll"));
            candidates.Add(Path.Combine(AppContext.BaseDirectory, "pcai_media.dll"));

            // 3. Common project locations relative to %USERPROFILE%
            var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            candidates.Add(Path.Combine(userProfile, "PC_AI", "bin", "pcai_media.dll"));
            candidates.Add(Path.Combine(userProfile, ".local", "bin", "pcai_media.dll"));
            candidates.Add(Path.Combine(userProfile, "PC_AI", "Native", "pcai_core", "pcai_media", "target", "release", "pcai_media.dll"));
            candidates.Add(Path.Combine(userProfile, "PC_AI", "Deploy", "pcai-media", "target", "release", "pcai_media.dll"));

            foreach (var path in candidates)
            {
                if (File.Exists(path) && NativeLibrary.TryLoad(path, out handle))
                    return handle;
            }

            return IntPtr.Zero;
        }

        /// <summary>
        /// Loads DLL search paths from <c>Config/llm-config.json</c> under the
        /// <c>nativeMedia.dllSearchPaths</c> array. Both absolute and project-relative
        /// paths are supported.
        /// </summary>
        private static IEnumerable<string> LoadConfigSearchPaths(Assembly assembly)
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

                if (!doc.RootElement.TryGetProperty("nativeMedia", out var nativeMedia))
                    return results;

                if (!nativeMedia.TryGetProperty("dllSearchPaths", out var paths) || paths.ValueKind != JsonValueKind.Array)
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

        /// <summary>
        /// Walks up the directory tree from the assembly location looking for
        /// <c>Config/llm-config.json</c>, stopping after six levels.
        /// </summary>
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

        #region Native Imports

        /// <summary>
        /// Initialises the pcai_media backend for the specified compute device.
        /// Must be called once before any other media function.
        /// </summary>
        /// <param name="device">
        /// Device selector string, e.g. <c>"cuda:0"</c>, <c>"cpu"</c>, or <c>null</c>
        /// to use the library default (CUDA when available, otherwise CPU).
        /// </param>
        /// <returns>0 on success; non-zero on failure. Inspect <see cref="pcai_media_last_error"/> for details.</returns>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int pcai_media_init([MarshalAs(UnmanagedType.LPUTF8Str)] string? device);

        /// <summary>
        /// Loads the Janus-Pro model weights from disk into the native backend.
        /// </summary>
        /// <param name="modelPath">
        /// Absolute path to the model directory or a single-file weight blob
        /// (format is backend-specific, e.g. a SafeTensors directory or GGUF file).
        /// </param>
        /// <param name="gpuLayers">
        /// Number of transformer layers to offload to the GPU.
        /// Pass 0 to run entirely on CPU, or -1 to offload all layers.
        /// </param>
        /// <returns>0 on success; non-zero on failure.</returns>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int pcai_media_load_model([MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath, int gpuLayers);

        /// <summary>
        /// Generates an image from a text prompt and writes the result to <paramref name="outputPath"/>.
        /// This is a synchronous, blocking call.
        /// </summary>
        /// <param name="prompt">Natural-language description of the desired image (UTF-8).</param>
        /// <param name="cfgScale">
        /// Classifier-free guidance scale. Higher values adhere more strictly to the prompt.
        /// Typical range: 1.0–10.0; default 5.0.
        /// </param>
        /// <param name="temperature">
        /// Sampling temperature controlling creativity/diversity.
        /// Typical range: 0.5–1.5; default 1.0.
        /// </param>
        /// <param name="outputPath">
        /// Absolute path where the generated image will be saved (PNG format recommended).
        /// </param>
        /// <returns>0 on success; non-zero on failure.</returns>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int pcai_media_generate_image(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string prompt,
            float cfgScale,
            float temperature,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string outputPath);

        /// <summary>
        /// Runs visual understanding (VQA / image captioning) on an existing image file.
        /// Returns a heap-allocated UTF-8 string pointer containing the model's response.
        /// </summary>
        /// <param name="imagePath">Absolute path to the image file (PNG, JPEG, or BMP).</param>
        /// <param name="prompt">Question or instruction for the model, e.g. <c>"Describe this image."</c></param>
        /// <param name="maxTokens">Maximum number of tokens to generate in the response.</param>
        /// <param name="temperature">Sampling temperature. 0.0 = deterministic, 1.0 = creative.</param>
        /// <returns>
        /// A non-null heap-allocated UTF-8 string pointer containing the model response,
        /// or <see cref="IntPtr.Zero"/> on failure. The caller <b>must</b> free the pointer
        /// with <see cref="pcai_media_free_string"/>.
        /// </returns>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr pcai_media_understand_image(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string imagePath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string prompt,
            uint maxTokens,
            float temperature);

        /// <summary>
        /// Generates an image from a text prompt and returns raw PNG bytes.
        /// The caller must free the buffer with <see cref="pcai_media_free_bytes"/>.
        /// </summary>
        /// <param name="prompt">Natural-language description of the desired image (UTF-8).</param>
        /// <param name="cfgScale">Classifier-free guidance scale. Pass 0 for default.</param>
        /// <param name="temperature">Sampling temperature. Pass 0 for default.</param>
        /// <param name="outData">Receives pointer to the PNG byte buffer.</param>
        /// <param name="outLen">Receives the byte count of the buffer.</param>
        /// <returns>0 on success; non-zero on failure.</returns>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int pcai_media_generate_image_bytes(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string prompt,
            float cfgScale,
            float temperature,
            out IntPtr outData,
            out UIntPtr outLen);

        /// <summary>
        /// Frees a byte buffer previously returned by <see cref="pcai_media_generate_image_bytes"/>.
        /// Passing <see cref="IntPtr.Zero"/> is a safe no-op.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pcai_media_free_bytes(IntPtr data, UIntPtr len);

        /// <summary>
        /// Shuts down the pcai_media backend and releases all model resources.
        /// Safe to call multiple times; subsequent calls after the first are no-ops.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pcai_media_shutdown();

        /// <summary>
        /// Returns a pointer to the last error message recorded on the calling thread,
        /// or <see cref="IntPtr.Zero"/> if no error has been set.
        /// The returned pointer is valid until the next native call on the same thread.
        /// Do <b>not</b> free this pointer.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr pcai_media_last_error();

        /// <summary>
        /// Returns the integer error code of the last native media operation on the
        /// calling thread, or 0 if no error has occurred.
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pcai_media_last_error_code();

        /// <summary>
        /// Frees a heap-allocated string previously returned by the native media library.
        /// Passing <see cref="IntPtr.Zero"/> is a safe no-op.
        /// </summary>
        /// <param name="ptr">Pointer returned by a native function (e.g. <see cref="pcai_media_understand_image"/>).</param>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pcai_media_free_string(IntPtr ptr);

        #endregion

        #region High-level Wrappers

        /// <summary>
        /// Returns <c>true</c> when the native DLL is loadable from the current process.
        /// Does not require the backend to be initialised.
        /// </summary>
        public static bool IsAvailable
        {
            get
            {
                try { return pcai_media_last_error_code() >= 0; }
                catch (DllNotFoundException) { return false; }
                catch (EntryPointNotFoundException) { return false; }
            }
        }

        /// <summary>
        /// Returns the last error message recorded by the native library on the calling thread,
        /// or <c>null</c> if no error has been set.
        /// </summary>
        public static string? GetLastError()
        {
            var ptr = pcai_media_last_error();
            return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
        }

        /// <summary>
        /// Generates an image from a text prompt and writes the result to <paramref name="outputPath"/>.
        /// </summary>
        /// <param name="prompt">Natural-language description of the desired image.</param>
        /// <param name="outputPath">
        /// Absolute path where the generated image will be saved (PNG format recommended).
        /// </param>
        /// <param name="cfgScale">
        /// Classifier-free guidance scale. Typical range 1.0–10.0; defaults to 5.0.
        /// </param>
        /// <param name="temperature">
        /// Sampling temperature. Typical range 0.5–1.5; defaults to 1.0.
        /// </param>
        /// <returns>
        /// <c>null</c> on success, or an error message string if the native call failed.
        /// </returns>
        public static string? GenerateImage(
            string prompt,
            string outputPath,
            float cfgScale = 5.0f,
            float temperature = 1.0f)
        {
            var result = pcai_media_generate_image(prompt, cfgScale, temperature, outputPath);
            return result == 0 ? null : GetLastError();
        }

        /// <summary>
        /// Runs visual understanding on an existing image file and returns the model's response text.
        /// </summary>
        /// <param name="imagePath">Absolute path to the image file (PNG, JPEG, or BMP).</param>
        /// <param name="prompt">Question or instruction, e.g. <c>"What objects are in this image?"</c></param>
        /// <param name="maxTokens">Maximum tokens to generate. Defaults to 512.</param>
        /// <param name="temperature">Sampling temperature. Defaults to 0.7.</param>
        /// <returns>
        /// The model's response text, or <c>null</c> if the native call returned a null pointer
        /// (typically indicating an uninitialised backend or missing model).
        /// </returns>
        public static string? UnderstandImage(
            string imagePath,
            string prompt,
            uint maxTokens = 512,
            float temperature = 0.7f)
        {
            var ptr = pcai_media_understand_image(imagePath, prompt, maxTokens, temperature);
            if (ptr == IntPtr.Zero) return null;
            try
            {
                return Marshal.PtrToStringUTF8(ptr);
            }
            finally
            {
                pcai_media_free_string(ptr);
            }
        }

        /// <summary>
        /// Generates an image and returns the raw PNG bytes as a managed byte array.
        /// </summary>
        /// <param name="prompt">Natural-language description of the desired image.</param>
        /// <param name="cfgScale">Classifier-free guidance scale. Defaults to 5.0.</param>
        /// <param name="temperature">Sampling temperature. Defaults to 1.0.</param>
        /// <returns>PNG bytes on success, or <c>null</c> on failure.</returns>
        public static byte[]? GenerateImageBytes(
            string prompt,
            float cfgScale = 5.0f,
            float temperature = 1.0f)
        {
            var result = pcai_media_generate_image_bytes(prompt, cfgScale, temperature, out var dataPtr, out var lenPtr);
            if (result != 0 || dataPtr == IntPtr.Zero)
                return null;

            var len = (int)lenPtr;
            try
            {
                var bytes = new byte[len];
                Marshal.Copy(dataPtr, bytes, 0, len);
                return bytes;
            }
            finally
            {
                pcai_media_free_bytes(dataPtr, lenPtr);
            }
        }

        /// <summary>
        /// Asynchronously generates an image from a text prompt by running the blocking
        /// native call on a thread-pool thread.
        /// </summary>
        /// <param name="prompt">Natural-language description of the desired image.</param>
        /// <param name="outputPath">
        /// Absolute path where the generated image will be saved (PNG format recommended).
        /// </param>
        /// <param name="cfgScale">
        /// Classifier-free guidance scale. Typical range 1.0–10.0; defaults to 5.0.
        /// </param>
        /// <param name="temperature">
        /// Sampling temperature. Typical range 0.5–1.5; defaults to 1.0.
        /// </param>
        /// <param name="cancellationToken">
        /// Token to observe for cooperative cancellation. Note: cancellation is best-effort —
        /// once the native call has started it runs to completion or failure.
        /// </param>
        /// <returns>
        /// A task that resolves to <c>null</c> on success, or an error message string on failure.
        /// </returns>
        /// <exception cref="OperationCanceledException">
        /// Thrown if <paramref name="cancellationToken"/> is cancelled before the native call begins.
        /// </exception>
        public static Task<string?> GenerateImageAsync(
            string prompt,
            string outputPath,
            float cfgScale = 5.0f,
            float temperature = 1.0f,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.Run(
                () => GenerateImage(prompt, outputPath, cfgScale, temperature),
                cancellationToken);
        }

        /// <summary>
        /// Asynchronously runs visual understanding on an existing image file by offloading
        /// the blocking native call to a thread-pool thread.
        /// </summary>
        /// <param name="imagePath">Absolute path to the image file (PNG, JPEG, or BMP).</param>
        /// <param name="prompt">Question or instruction for the model.</param>
        /// <param name="maxTokens">Maximum tokens to generate. Defaults to 512.</param>
        /// <param name="temperature">Sampling temperature. Defaults to 0.7.</param>
        /// <param name="cancellationToken">
        /// Token to observe for cooperative cancellation. Best-effort once the native call starts.
        /// </param>
        /// <returns>
        /// A task that resolves to the model's response text, or <c>null</c> if the native
        /// backend returned a null pointer.
        /// </returns>
        /// <exception cref="OperationCanceledException">
        /// Thrown if <paramref name="cancellationToken"/> is cancelled before the native call begins.
        /// </exception>
        public static Task<string?> UnderstandImageAsync(
            string imagePath,
            string prompt,
            uint maxTokens = 512,
            float temperature = 0.7f,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.Run(
                () => UnderstandImage(imagePath, prompt, maxTokens, temperature),
                cancellationToken);
        }

        #endregion
    }
}
