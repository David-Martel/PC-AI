using System;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using Xunit;

namespace PcaiNative.Tests;

/// <summary>
/// Verifies P/Invoke declarations in PcaiNative are structurally correct without
/// loading native DLLs.  These tests run on any platform (Linux, Windows, macOS)
/// because they only inspect assembly metadata via reflection.
///
/// Design decisions:
///   - <see cref="InferenceModule"/> (public) is used as the assembly anchor.
///     <c>NativeCore</c> is internal and lives in the same assembly, so all
///     DllImport declarations across both types are found by scanning all types.
///   - DLL names are normalised (trimmed, .dll suffix stripped, lower-cased) before
///     comparison so "pcai_core_lib.dll" and "pcai_core_lib" both match.
/// </summary>
public class PInvokeSignatureTests
{
    // Use a public type from PcaiNative as the assembly entry point.
    private static readonly Assembly PcaiAssembly = typeof(InferenceModule).Assembly;

    // Canonical set of Rust library stems that P/Invoke declarations may target.
    // Comparison is performed after stripping the ".dll" suffix (see NormaliseDllName).
    private static readonly string[] AllowedLibraries =
    [
        "pcai_inference",
        "pcai_core_lib",
        "pcai_media",
    ];

    // ──────────────────────────────────────────────────────────────────────────
    // Calling convention
    // ──────────────────────────────────────────────────────────────────────────

    [Fact]
    public void AllDllImportsSpecifyCallingConventionCdecl()
    {
        // Rust's default extern "C" ABI maps to Cdecl on all platforms.
        // A mismatch causes stack corruption at runtime on x86 (silent on x64
        // but still a correctness violation).
        var violations = GetDllImportMethods()
            .Where(m => m.GetCustomAttribute<DllImportAttribute>()!.CallingConvention
                        != CallingConvention.Cdecl)
            .Select(m => $"{m.DeclaringType?.Name}.{m.Name}")
            .ToList();

        Assert.True(violations.Count == 0,
            $"These P/Invoke methods do not specify CallingConvention.Cdecl " +
            $"(required for Rust FFI):\n  {string.Join("\n  ", violations)}");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Library name allow-list
    // ──────────────────────────────────────────────────────────────────────────

    [Fact]
    public void AllDllImportsReferenceKnownLibraries()
    {
        var violations = GetDllImportMethods()
            .Select(m => (Method: m, Lib: NormaliseDllName(m.GetCustomAttribute<DllImportAttribute>()!.Value)))
            .Where(x => !AllowedLibraries.Contains(x.Lib))
            .Select(x => $"{x.Method.DeclaringType?.Name}.{x.Method.Name} → '{x.Lib}'")
            .ToList();

        Assert.True(violations.Count == 0,
            $"These P/Invoke methods reference libraries not in the allowed set " +
            $"({string.Join(", ", AllowedLibraries)}):\n  {string.Join("\n  ", violations)}");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // String return-type safety
    // ──────────────────────────────────────────────────────────────────────────

    [Fact]
    public void NoDllImportMethodsReturnRawString()
    {
        // Returning a raw managed string from a DllImport is always wrong:
        // the runtime will attempt to free the Rust-owned pointer via CoTaskMemFree,
        // causing heap corruption.  All string returns must use IntPtr and be
        // manually marshalled, or use a SafeHandle subclass.
        var violations = GetDllImportMethods()
            .Where(m => m.ReturnType == typeof(string))
            .Select(m => $"{m.DeclaringType?.Name}.{m.Name}")
            .ToList();

        Assert.True(violations.Count == 0,
            $"These P/Invoke methods return raw string, which causes heap corruption " +
            $"(use IntPtr + manual marshal or SafeHandle instead):\n  " +
            $"{string.Join("\n  ", violations)}");
    }

    [Fact]
    public void StringReturnTypesUseIntPtrOrSafeHandle()
    {
        // Any method whose return carries a native string pointer must use either:
        //   IntPtr  — for static/borrowed strings (caller must NOT free them), or
        //   SafeHandle — for owned strings (finalizer calls the free function).
        var stringReturners = GetDllImportMethods()
            .Where(m => m.ReturnType == typeof(string)
                     || m.ReturnType == typeof(IntPtr)
                     || m.ReturnType.IsSubclassOf(typeof(SafeHandle)));

        var violations = stringReturners
            .Where(m => m.ReturnType == typeof(string)) // raw string is the only bad case
            .Select(m => $"{m.DeclaringType?.Name}.{m.Name}")
            .ToList();

        Assert.True(violations.Count == 0,
            $"String-returning FFI methods must use IntPtr or SafeHandle:\n  " +
            $"{string.Join("\n  ", violations)}");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Sanity checks
    // ──────────────────────────────────────────────────────────────────────────

    [Fact]
    public void AssemblyContainsDllImportDeclarations()
    {
        // If this fires the assembly is empty or something went wrong with the
        // project reference — all other tests would trivially pass on zero methods.
        Assert.NotEmpty(GetDllImportMethods());
    }

    [Fact]
    public void AllowedLibraryCoverageIsComplete()
    {
        // Every allowed library stem must be referenced by at least one DllImport.
        // This catches stale entries in AllowedLibraries after a refactor removes a DLL.
        var actualLibs = GetDllImportMethods()
            .Select(m => NormaliseDllName(m.GetCustomAttribute<DllImportAttribute>()!.Value))
            .ToHashSet();

        var unused = AllowedLibraries.Where(lib => !actualLibs.Contains(lib)).ToList();

        Assert.True(unused.Count == 0,
            $"These entries in AllowedLibraries are not referenced by any DllImport — " +
            $"remove them or add the missing declarations:\n  {string.Join("\n  ", unused)}");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────────────────────────────────

    private static MethodInfo[] GetDllImportMethods()
    {
        return PcaiAssembly
            .GetTypes()
            .SelectMany(t => t.GetMethods(
                BindingFlags.Static |
                BindingFlags.NonPublic |
                BindingFlags.Public))
            .Where(m => m.GetCustomAttribute<DllImportAttribute>() != null)
            .ToArray();
    }

    /// <summary>
    /// Normalise a DLL name for comparison: trim whitespace, strip ".dll" suffix,
    /// lower-case.  Handles both "pcai_core_lib.dll" and "pcai_core_lib".
    /// </summary>
    private static string NormaliseDllName(string? name)
    {
        if (name is null) return string.Empty;
        var s = name.Trim();
        if (s.EndsWith(".dll", StringComparison.OrdinalIgnoreCase))
            s = s[..^4];
        return s.ToLowerInvariant();
    }
}
