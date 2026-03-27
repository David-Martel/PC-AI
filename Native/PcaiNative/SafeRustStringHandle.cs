using System;
using System.Runtime.InteropServices;

namespace PcaiNative;

/// <summary>
/// A SafeHandle to safely wrap and manage the lifetime of a string pointer allocated by the native Rust core library.
/// Ensures the pointer is always freed by calling <c>pcai_free_string</c>.
/// </summary>
public sealed class SafeRustStringHandle : SafeHandle
{
    // Required default constructor for P/Invoke marshalling
    public SafeRustStringHandle() : base(IntPtr.Zero, true)
    {
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            NativeCore.pcai_free_string(handle);
            handle = IntPtr.Zero;
        }
        return true;
    }

    /// <summary>
    /// Converts the native string to a managed string using UTF-8 encoding.
    /// Returns null if the handle is invalid or the pointer is null.
    /// </summary>
    public string? ToManagedString()
    {
        if (IsInvalid) return null;
        return Marshal.PtrToStringUTF8(handle);
    }
}
