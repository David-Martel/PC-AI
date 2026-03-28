using System.Runtime.InteropServices;

namespace PcaiNative;

/// <summary>
/// A <see cref="SafeHandle"/> subclass for heap-allocated C strings returned by pcai_core_lib.
///
/// Use this type only for Rust FFI functions that allocate a new string on the heap and
/// document that the caller must free the pointer via <c>pcai_free_string</c>. Examples:
/// <list type="bullet">
///   <item><description><c>pcai_string_copy</c> — allocates a heap copy of the input string.</description></item>
///   <item><description><c>pcai_extract_json</c> — allocates extracted JSON on the heap.</description></item>
/// </list>
///
/// Do NOT use this type for functions that return pointers into static Rust memory (string
/// literals baked into the binary), such as <c>pcai_core_version</c>,
/// <c>pcai_search_version</c>, or <c>pcai_status_description</c>. Those functions must use
/// <see cref="IntPtr"/> with <c>Marshal.PtrToStringAnsi</c> or
/// <c>Marshal.PtrToStringUTF8</c> and must never call <c>pcai_free_string</c>.
/// Doing so would pass a non-heap pointer to the Rust allocator — undefined behaviour and
/// a crash.
/// </summary>
public sealed class SafeRustStringHandle : SafeHandle
{
    /// <summary>
    /// Initializes a new instance of <see cref="SafeRustStringHandle"/>.
    /// The P/Invoke runtime calls this constructor when the native function returns.
    /// </summary>
    public SafeRustStringHandle() : base(IntPtr.Zero, ownsHandle: true) { }

    /// <inheritdoc/>
    public override bool IsInvalid => handle == IntPtr.Zero;

    /// <summary>
    /// Frees the heap-allocated Rust string via <c>pcai_free_string</c>.
    /// Called automatically by the SafeHandle finalizer / Dispose pattern.
    /// </summary>
    /// <returns><see langword="true"/> on success (always, because pcai_free_string is void).</returns>
    protected override bool ReleaseHandle()
    {
        NativeCore.pcai_free_string(handle);
        return true;
    }

    /// <summary>
    /// Reads the native UTF-8 string into a managed <see cref="string"/>.
    /// Returns <see cref="string.Empty"/> if the handle is invalid.
    /// </summary>
    public override string ToString()
    {
        if (IsInvalid) return string.Empty;
        return Marshal.PtrToStringUTF8(handle) ?? string.Empty;
    }
}
