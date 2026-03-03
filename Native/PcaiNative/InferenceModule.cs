using System;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

#nullable enable

namespace PcaiNative
{
    /// <summary>
    /// P/Invoke interop with pcai_inference.dll
    /// </summary>
    public static class InferenceModule
    {
        private const string DllName = "pcai_inference";

        /// <summary>
        /// Status codes for async inference requests.
        /// Matches the integer values returned by <c>pcai_poll_result</c>.
        /// </summary>
        public enum AsyncRequestStatus
        {
            /// <summary>The request ID is not recognised by the native layer.</summary>
            Unknown = -1,
            /// <summary>The request has been accepted and is queued for processing.</summary>
            Pending = 0,
            /// <summary>A worker thread is actively running inference for this request.</summary>
            Running = 1,
            /// <summary>Inference finished successfully. The result text is available.</summary>
            Complete = 2,
            /// <summary>Inference failed. The error message is available via the text field.</summary>
            Failed = 3,
            /// <summary>The request was cancelled before or during inference.</summary>
            Cancelled = 4,
        }

        /// <summary>
        /// Matches the Rust <c>repr(C)</c> <c>PcaiAsyncResult</c> struct.
        /// On 64-bit Windows the layout is: 4-byte Status + 4-byte padding + 8-byte Text pointer = 16 bytes.
        /// </summary>
        /// <remarks>
        /// When <see cref="AsyncRequestStatus.Complete"/> or <see cref="AsyncRequestStatus.Failed"/>
        /// is observed the <see cref="Text"/> pointer is non-null and <b>must</b> be freed by the
        /// caller via <c>pcai_free_string</c>. For all other statuses <see cref="Text"/> is null.
        /// </remarks>
        [StructLayout(LayoutKind.Sequential)]
        public struct PcaiAsyncResult
        {
            /// <summary>
            /// Status code of the async request.
            /// Cast to <see cref="AsyncRequestStatus"/> for a typed representation.
            /// </summary>
            public int Status;

            /// <summary>
            /// Result text pointer. Non-null only when <see cref="Status"/> is 2 (complete) or
            /// 3 (failed). The caller must free this pointer with <c>pcai_free_string</c>.
            /// </summary>
            public IntPtr Text;
        }

        /// <summary>
        /// Static constructor to register native library resolver
        /// </summary>
        static InferenceModule()
        {
            NativeResolver.EnsureRegistered(typeof(InferenceModule).Assembly);
        }

        #region Native Imports

        /// <summary>Initialises the native inference backend. Returns 0 on success.</summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int pcai_init([MarshalAs(UnmanagedType.LPUTF8Str)] string backendName);

        /// <summary>Loads a GGUF/SafeTensors model from disk. Returns 0 on success.</summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int pcai_load_model([MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath, int gpuLayers);

        /// <summary>Runs synchronous inference and returns a heap-allocated UTF-8 string pointer. Free with <see cref="pcai_free_string"/>.</summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr pcai_generate([MarshalAs(UnmanagedType.LPUTF8Str)] string prompt, uint maxTokens, float temperature);

        /// <summary>Runs streaming inference, invoking <paramref name="callback"/> for each token. Returns 0 on success.</summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int pcai_generate_streaming([MarshalAs(UnmanagedType.LPUTF8Str)] string prompt, uint maxTokens, float temperature, TokenCallback callback, IntPtr userData);

        /// <summary>Shuts down the native backend and releases all loaded model resources.</summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pcai_shutdown();

        /// <summary>Returns a pointer to the last thread-local error message, or <see cref="IntPtr.Zero"/> if none.</summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr pcai_last_error();

        /// <summary>Returns the integer error code of the last native operation, or 0 if no error occurred.</summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pcai_last_error_code();

        /// <summary>Frees a heap-allocated string previously returned by the native library.</summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pcai_free_string(IntPtr ptr);

        /// <summary>Returns non-zero if the native backend has been successfully initialised.</summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pcai_is_initialized();

        /// <summary>Returns non-zero if a model is currently loaded into the native backend.</summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pcai_is_model_loaded();

        /// <summary>Returns a pointer to a static string identifying the active backend name (e.g. "llamacpp").</summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr pcai_get_backend_name();

        /// <summary>Callback signature invoked by the native layer for each streamed token.</summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void TokenCallback(IntPtr token, IntPtr userData);

        /// <summary>
        /// Submits an asynchronous inference request to the native backend.
        /// The caller must poll with <see cref="pcai_poll_result"/> until a terminal
        /// status (complete, failed, or cancelled) is observed.
        /// </summary>
        /// <param name="prompt">Input text prompt (null-terminated UTF-8, max 100 KB).</param>
        /// <param name="maxTokens">Maximum tokens to generate. 0 uses the backend default.</param>
        /// <param name="temperature">Sampling temperature. 0.0 = greedy, 1.0 = creative.</param>
        /// <returns>A positive request ID on success, or -1 on error (inspect <c>pcai_last_error</c>).</returns>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern long pcai_generate_async([MarshalAs(UnmanagedType.LPUTF8Str)] string prompt, uint maxTokens, float temperature);

        /// <summary>
        /// Polls the status of an async request. When the returned status is 2 (complete)
        /// or 3 (failed) the request is removed from the internal map and the
        /// <see cref="PcaiAsyncResult.Text"/> pointer is non-null. The caller must free it
        /// with <see cref="pcai_free_string"/>.
        /// </summary>
        /// <param name="requestId">Request ID returned by <see cref="pcai_generate_async"/>.</param>
        /// <returns>
        /// A <see cref="PcaiAsyncResult"/> whose <c>Status</c> field encodes one of:
        /// 0=pending, 1=running, 2=complete, 3=failed, 4=cancelled, -1=unknown ID.
        /// </returns>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern PcaiAsyncResult pcai_poll_result(long requestId);

        /// <summary>
        /// Marks a pending or running async request as cancelled. If inference is already
        /// in progress the worker thread observes the flag and exits without writing a result.
        /// </summary>
        /// <param name="requestId">Request ID returned by <see cref="pcai_generate_async"/>.</param>
        /// <returns>0 if successfully cancelled, -1 if the ID was not found or already finished.</returns>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pcai_cancel(long requestId);

        #endregion

        #region High-level Wrappers

        /// <summary>Returns <c>true</c> if the native DLL is loaded and the backend is reachable.</summary>
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

        /// <summary>Runs synchronous inference and returns the generated text, or <c>null</c> on failure.</summary>
        public static string? Generate(string prompt, uint maxTokens = 512, float temperature = 0.7f)
        {
            var ptr = pcai_generate(prompt, maxTokens, temperature);
            if (ptr == IntPtr.Zero) return null;
            try { return Marshal.PtrToStringUTF8(ptr); }
            finally { pcai_free_string(ptr); }
        }

        /// <summary>
        /// Gets the last error message recorded by the native library on the current thread.
        /// </summary>
        /// <returns>The error message, or <c>null</c> if no error has been set.</returns>
        public static string? GetLastError()
        {
            var ptr = pcai_last_error();
            return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
        }

        /// <summary>
        /// Starts an asynchronous inference request and returns a <see cref="Task{T}"/>
        /// that resolves when the native backend produces a result.
        /// </summary>
        /// <remarks>
        /// The method polls the native backend at <paramref name="pollIntervalMs"/> millisecond
        /// intervals without blocking the calling thread. On <see cref="AsyncRequestStatus.Complete"/>
        /// or <see cref="AsyncRequestStatus.Failed"/> the internal request entry is cleaned up
        /// automatically (the native layer removes it after the first terminal poll).
        /// </remarks>
        /// <param name="prompt">The input prompt text.</param>
        /// <param name="maxTokens">Maximum tokens to generate. Defaults to 512.</param>
        /// <param name="temperature">Sampling temperature. Defaults to 0.7.</param>
        /// <param name="pollIntervalMs">Milliseconds between successive poll attempts. Defaults to 50.</param>
        /// <param name="cancellationToken">Token to cancel the operation. Cancelling signals the native
        /// layer via <c>pcai_cancel</c> and the task returns <c>null</c>.</param>
        /// <returns>
        /// The generated text on success, or <c>null</c> if the request was cancelled.
        /// </returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the native layer rejects the request or when inference fails.
        /// </exception>
        public static async Task<string?> GenerateAsync(
            string prompt,
            uint maxTokens = 512,
            float temperature = 0.7f,
            int pollIntervalMs = 50,
            CancellationToken cancellationToken = default)
        {
            long requestId = pcai_generate_async(prompt, maxTokens, temperature);
            if (requestId < 0)
            {
                var error = GetLastError();
                throw new InvalidOperationException($"Failed to start async generation: {error ?? "unknown error"}");
            }

            try
            {
                while (!cancellationToken.IsCancellationRequested)
                {
                    var result = pcai_poll_result(requestId);
                    var status = (AsyncRequestStatus)result.Status;

                    switch (status)
                    {
                        case AsyncRequestStatus.Complete:
                            // Text pointer is non-null; extract then free.
                            try
                            {
                                return result.Text != IntPtr.Zero
                                    ? Marshal.PtrToStringUTF8(result.Text)
                                    : null;
                            }
                            finally
                            {
                                if (result.Text != IntPtr.Zero)
                                    pcai_free_string(result.Text);
                            }

                        case AsyncRequestStatus.Failed:
                            string? errorMsg = null;
                            try
                            {
                                errorMsg = result.Text != IntPtr.Zero
                                    ? Marshal.PtrToStringUTF8(result.Text)
                                    : null;
                            }
                            finally
                            {
                                if (result.Text != IntPtr.Zero)
                                    pcai_free_string(result.Text);
                            }
                            throw new InvalidOperationException(
                                $"Async generation failed: {errorMsg ?? "unknown error"}");

                        case AsyncRequestStatus.Cancelled:
                            return null;

                        case AsyncRequestStatus.Unknown:
                            throw new InvalidOperationException(
                                $"Unknown request ID returned by native layer: {requestId}");

                        case AsyncRequestStatus.Pending:
                        case AsyncRequestStatus.Running:
                            await Task.Delay(pollIntervalMs, cancellationToken).ConfigureAwait(false);
                            break;

                        default:
                            throw new InvalidOperationException(
                                $"Unexpected async status value: {result.Status}");
                    }
                }

                // CancellationToken fired — cancel the native request and signal to caller.
                pcai_cancel(requestId);
                return null;
            }
            catch (OperationCanceledException)
            {
                // Propagate cancellation but ensure the native request is cleaned up first.
                pcai_cancel(requestId);
                throw;
            }
        }

        /// <summary>
        /// Polls the status of an async request without blocking.
        /// When the status is terminal (complete or failed) the native entry is consumed
        /// and the text is returned; subsequent calls for the same ID will return
        /// <see cref="AsyncRequestStatus.Unknown"/>.
        /// </summary>
        /// <param name="requestId">Request ID obtained from <see cref="pcai_generate_async"/>.</param>
        /// <returns>
        /// A tuple of <c>(Status, Text)</c> where <c>Text</c> is non-null only for
        /// <see cref="AsyncRequestStatus.Complete"/> or <see cref="AsyncRequestStatus.Failed"/>.
        /// </returns>
        public static (AsyncRequestStatus Status, string? Text) PollResult(long requestId)
        {
            var result = pcai_poll_result(requestId);
            var status = (AsyncRequestStatus)result.Status;
            string? text = null;

            if (result.Text != IntPtr.Zero)
            {
                try
                {
                    text = Marshal.PtrToStringUTF8(result.Text);
                }
                finally
                {
                    pcai_free_string(result.Text);
                }
            }

            return (status, text);
        }

        /// <summary>
        /// Cancels a pending or running async request. If inference is already executing
        /// the worker thread will observe the cancellation flag before storing a result.
        /// </summary>
        /// <param name="requestId">Request ID obtained from <see cref="pcai_generate_async"/>.</param>
        /// <returns>
        /// <c>true</c> if the request was successfully marked as cancelled;
        /// <c>false</c> if the ID was not found or the request had already reached a terminal state.
        /// </returns>
        public static bool CancelRequest(long requestId)
        {
            return pcai_cancel(requestId) == 0;
        }

        #endregion
    }
}
