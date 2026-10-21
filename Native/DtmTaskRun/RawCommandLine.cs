using System.Runtime.InteropServices;
using System.Text;

namespace DtmTaskRun;

/// <summary>
/// Recovers the child's command line exactly as it was written after the launcher's first bare
/// <c>--</c>, and builds one from split arguments when no raw text is available.
/// </summary>
/// <remarks>
/// <c>conhost.exe --headless &lt;cmd&gt;</c> handed <c>&lt;cmd&gt;</c> to CreateProcess unchanged. Splitting it into
/// argv and quoting each element again (what <c>ProcessStartInfo.ArgumentList</c> does) is not the
/// identity for programs that parse their own command line: <c>cmd.exe /c ""C:\a b\job.cmd" "x y""</c>
/// comes out with different quotes and a different meaning. So the launcher passes the original text.
/// </remarks>
internal static class RawCommandLine
{
    /// <summary>The process's own command line as Windows stores it (GetCommandLineW), or null.</summary>
    public static string? Current()
    {
        var p = GetCommandLineW();
        return p == IntPtr.Zero ? null : Marshal.PtrToStringUni(p);
    }

    /// <summary>
    /// Returns the text after the first argument whose value is <c>--</c>, with the separating
    /// whitespace removed, or null when there is no such argument, nothing follows it, or the
    /// tokens before it do not match <paramref name="parsedArgs"/> (then the caller falls back to
    /// <see cref="Build"/>, which is never worse than the old behaviour).
    /// </summary>
    /// <param name="raw">The full command line, program name included.</param>
    /// <param name="parsedArgs">The runtime's argv (program name excluded), used as a cross-check.</param>
    public static string? ChildPart(string? raw, IReadOnlyList<string> parsedArgs)
    {
        if (string.IsNullOrEmpty(raw))
        {
            return null;
        }

        var i = SkipProgramName(raw);
        var index = 0;
        while (true)
        {
            i = SkipWhitespace(raw, i);
            if (i >= raw.Length)
            {
                return null;
            }

            var (value, next) = ReadArgument(raw, i);
            if (index >= parsedArgs.Count || !string.Equals(value, parsedArgs[index], StringComparison.Ordinal))
            {
                return null;
            }

            index++;
            i = next;
            if (value == "--")
            {
                var rest = raw[SkipWhitespace(raw, i)..].TrimEnd();
                return rest.Length == 0 ? null : rest;
            }
        }
    }

    /// <summary>
    /// Builds a command line from an executable and split arguments with the quoting rules the C
    /// runtime's argv parser reverses (the same rules as <c>ProcessStartInfo.ArgumentList</c>).
    /// </summary>
    public static string Build(string exe, IReadOnlyList<string> args)
    {
        var sb = new StringBuilder();
        // The program name is parsed without backslash escapes: quote it only when it needs it.
        _ = exe.Length == 0 || exe.Any(char.IsWhiteSpace) ? sb.Append('"').Append(exe).Append('"') : sb.Append(exe);
        foreach (var a in args)
        {
            _ = sb.Append(' ');
            AppendQuoted(sb, a);
        }

        return sb.ToString();
    }

    private static void AppendQuoted(StringBuilder sb, string arg)
    {
        if (arg.Length > 0 && arg.IndexOfAny([' ', '\t', '\n', '\v', '"']) < 0)
        {
            _ = sb.Append(arg);
            return;
        }

        _ = sb.Append('"');
        for (var i = 0; i < arg.Length; i++)
        {
            var backslashes = 0;
            while (i < arg.Length && arg[i] == '\\')
            {
                backslashes++;
                i++;
            }

            if (i == arg.Length)
            {
                _ = sb.Append('\\', backslashes * 2);
                break;
            }

            if (arg[i] == '"')
            {
                _ = sb.Append('\\', (backslashes * 2) + 1).Append('"');
            }
            else
            {
                _ = sb.Append('\\', backslashes).Append(arg[i]);
            }
        }

        _ = sb.Append('"');
    }

    // argv[0]: a quoted name ends at the next quote, an unquoted one at the first space or tab; no escapes.
    private static int SkipProgramName(string s)
    {
        if (s.Length > 0 && s[0] == '"')
        {
            var close = s.IndexOf('"', 1);
            return close < 0 ? s.Length : close + 1;
        }

        var i = 0;
        while (i < s.Length && s[i] != ' ' && s[i] != '\t')
        {
            i++;
        }

        return i;
    }

    private static int SkipWhitespace(string s, int i)
    {
        while (i < s.Length && (s[i] == ' ' || s[i] == '\t'))
        {
            i++;
        }

        return i;
    }

    // One argument after argv[0], by the CommandLineToArgvW / MSVC rules: 2n backslashes + quote
    // gives n backslashes and toggles quoting, 2n+1 gives n backslashes and a literal quote, and
    // "" inside a quoted run gives a literal quote.
    private static (string Value, int Next) ReadArgument(string s, int i)
    {
        var sb = new StringBuilder();
        var inQuotes = false;
        while (i < s.Length)
        {
            var c = s[i];
            if (!inQuotes && (c == ' ' || c == '\t'))
            {
                break;
            }

            if (c == '\\')
            {
                var n = 0;
                while (i < s.Length && s[i] == '\\')
                {
                    n++;
                    i++;
                }

                if (i < s.Length && s[i] == '"')
                {
                    _ = sb.Append('\\', n / 2);
                    if (n % 2 == 1)
                    {
                        _ = sb.Append('"');
                        i++;
                    }
                }
                else
                {
                    _ = sb.Append('\\', n);
                }

                continue;
            }

            if (c == '"')
            {
                if (inQuotes && i + 1 < s.Length && s[i + 1] == '"')
                {
                    _ = sb.Append('"');
                    i += 2;
                    continue;
                }

                inQuotes = !inQuotes;
                i++;
                continue;
            }

            _ = sb.Append(c);
            i++;
        }

        return (sb.ToString(), i);
    }

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCommandLineW();
}
