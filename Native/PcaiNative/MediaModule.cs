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
    /// <summary>
    /// Result of polling an async media request via <see cref="MediaModule.pcai_media_poll_result"/>.
    /// </summary>
    /// <remarks>
    /// Status codes: 0=pending, 1=running, 2=complete, 3=failed, 4=cancelled, -1=unknown ID.
    /// When <see cref="Status"/> is 2 or 3, <see cref="Text"/> is non-null and must be freed
    /// with <c>pcai_media_free_string</c>.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct PcaiMediaAsyncResult
    {
        /// <summary>Status code (0=pending, 1=running, 2=complete, 3=failed, 4=cancelled, -1=unknown).</summary>
        public int Status;
        /// <summary>Result text (output path on success, error on failure). Must be freed by caller.</summary>
        public IntPtr Text;
    }

    public static class MediaModule
    {
        private const string DllName = "pcai_media";

        /// <summary>
        /// Static constructor — registers the custom DLL import resolver for this assembly
        /// so that <c>pcai_media.dll</c> can be located from non-standard paths at runtime.
        /// </summary>
        static MediaModule()
        {
            NativeResolver.EnsureRegistered(typeof(MediaModule).Assembly);
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

        /// <summary>
        /// Submits an asynchronous image generation request to the native backend.
        /// Returns a request ID immediately; use <see cref="pcai_media_poll_result"/> to check progress.
        /// </summary>
        /// <param name="prompt">Text prompt (UTF-8).</param>
        /// <param name="cfgScale">CFG scale. Pass 0 for default.</param>
        /// <param name="temperature">Sampling temperature. Pass 0 for default.</param>
        /// <param name="outputPath">Absolute path where the PNG will be written on success.</param>
        /// <returns>Request ID &gt; 0 on success, or -1 on error.</returns>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern long pcai_media_generate_image_async(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string prompt,
            float cfgScale,
            float temperature,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string outputPath);

        /// <summary>
        /// Polls the status of an async media request.
        /// When status is 2 (complete) or 3 (failed), the text pointer must be freed
        /// with <see cref="pcai_media_free_string"/>.
        /// </summary>
        /// <param name="requestId">ID returned by <see cref="pcai_media_generate_image_async"/>.</param>
        /// <returns>A <see cref="PcaiMediaAsyncResult"/> with status and optional text.</returns>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern PcaiMediaAsyncResult pcai_media_poll_result(long requestId);

        /// <summary>
        /// Cancels an in-progress async media request.
        /// </summary>
        /// <param name="requestId">ID returned by <see cref="pcai_media_generate_image_async"/>.</param>
        /// <returns>0 if cancelled successfully, -1 if ID not found or already finished.</returns>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pcai_media_cancel(long requestId);

        /// <summary>
        /// Upscale an image 4x using RealESRGAN ONNX model.
        /// </summary>
        /// <param name="modelPath">Path to the RealESRGAN ONNX model file.</param>
        /// <param name="inputPath">Path to the input image.</param>
        /// <param name="outputPath">Path to save the upscaled output image.</param>
        /// <returns>0 on success, negative error code on failure.</returns>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pcai_media_upscale_image(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string inputPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string outputPath);

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
        /// Generates an image using the native async FFI (no thread-pool blocking).
        /// The native backend runs generation on its own OS thread; this method polls
        /// for completion with cooperative cancellation.
        /// </summary>
        /// <param name="prompt">Natural-language description of the desired image.</param>
        /// <param name="outputPath">Absolute path where the generated PNG will be saved.</param>
        /// <param name="cfgScale">CFG scale. Defaults to 5.0.</param>
        /// <param name="temperature">Sampling temperature. Defaults to 1.0.</param>
        /// <param name="pollIntervalMs">Polling interval in milliseconds. Defaults to 100.</param>
        /// <param name="cancellationToken">Token for cooperative cancellation.</param>
        /// <returns><c>null</c> on success, or an error message on failure.</returns>
        public static async Task<string?> GenerateImageNativeAsync(
            string prompt,
            string outputPath,
            float cfgScale = 5.0f,
            float temperature = 1.0f,
            int pollIntervalMs = 100,
            CancellationToken cancellationToken = default)
        {
            var requestId = pcai_media_generate_image_async(prompt, cfgScale, temperature, outputPath);
            if (requestId < 0)
                return GetLastError() ?? "Failed to submit async generation request";

            try
            {
                while (true)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var result = pcai_media_poll_result(requestId);
                    switch (result.Status)
                    {
                        case 0: // pending
                        case 1: // running
                            await Task.Delay(pollIntervalMs, cancellationToken).ConfigureAwait(false);
                            continue;
                        case 2: // complete
                            if (result.Text != IntPtr.Zero)
                                pcai_media_free_string(result.Text);
                            return null;
                        case 3: // failed
                            string? error = null;
                            if (result.Text != IntPtr.Zero)
                            {
                                error = Marshal.PtrToStringUTF8(result.Text);
                                pcai_media_free_string(result.Text);
                            }
                            return error ?? "Unknown generation error";
                        case 4: // cancelled
                            throw new OperationCanceledException("Native request was cancelled");
                        default: // -1 or unknown
                            return "Unknown request ID or internal error";
                    }
                }
            }
            catch (OperationCanceledException)
            {
                pcai_media_cancel(requestId);
                throw;
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

        /// <summary>
        /// Upscale an image 4x using the RealESRGAN ONNX model.
        /// Requires the native DLL built with the <c>upscale</c> feature.
        /// </summary>
        /// <param name="modelPath">Path to the RealESRGAN ONNX model file.</param>
        /// <param name="inputPath">Path to the input image.</param>
        /// <param name="outputPath">Path where the 4x-upscaled image will be saved.</param>
        /// <returns><c>null</c> on success, or an error message on failure.</returns>
        public static string? UpscaleImage(string modelPath, string inputPath, string outputPath)
        {
            int rc = pcai_media_upscale_image(modelPath, inputPath, outputPath);
            return rc == 0 ? null : GetLastError() ?? $"Upscale failed (code {rc})";
        }

        #endregion
    }
}
