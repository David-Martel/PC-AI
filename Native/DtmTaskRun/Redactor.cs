using System.Text.RegularExpressions;

namespace DtmTaskRun;

/// <summary>
/// Best-effort secret masking for command lines, log lines, status JSON and event-log text.
/// This is defence in depth, not a guarantee: the stderr tail is deliberately kept out of
/// the agent-bus message, which is the only channel that leaves the machine.
/// </summary>
internal static partial class Redactor
{
    /// <summary>Replacement text for a masked value.</summary>
    public const string Mask = "***";

    // name=value / name:value where the NAME marks the value as secret.
    [GeneratedRegex(
        """(?<name>["']?[\w.\-]*(?:token|passw(?:or)?d|passwd|pwd|secret|credential|api[-_]?key|apikey|private[-_]?key|access[-_]?key|key)["']?)(?<sep>\s*[:=]\s*)(?<val>"[^"]*"|'[^']*'|\S+)""",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex NamedValue();

    [GeneratedRegex(@"\bBearer\s+[A-Za-z0-9._~+/=\-]+", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex BearerToken();

    [GeneratedRegex(@"\bAuthorization\s*:\s*\S+(?:[ \t]+\S+)?", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex AuthorizationHeader();

    // JWT-ish blobs, GitHub / OpenAI / Slack / AWS style tokens.
    [GeneratedRegex(@"\beyJ[\w\-]{20,}(?:\.[\w\-]*){0,2}|\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}|\bsk-[A-Za-z0-9_\-]{20,}|\bxox[abprs]-[A-Za-z0-9\-]{10,}|\bAKIA[0-9A-Z]{16}\b", RegexOptions.CultureInvariant)]
    private static partial Regex WellKnownTokens();

    // scheme://user:password@host
    [GeneratedRegex(@"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)", RegexOptions.CultureInvariant)]
    private static partial Regex UrlUserInfo();

    /// <summary>True when a flag or key name marks its value as secret (token, password, secret, ...key).</summary>
    public static bool IsSensitiveName(string name)
    {
        var n = name.TrimStart('-', '/').ToLowerInvariant();
        return n.Contains("token", StringComparison.Ordinal)
            || n.Contains("passw", StringComparison.Ordinal)
            || n.Contains("pwd", StringComparison.Ordinal)
            || n.Contains("secret", StringComparison.Ordinal)
            || n.Contains("credential", StringComparison.Ordinal)
            || n.Contains("bearer", StringComparison.Ordinal)
            || n.EndsWith("key", StringComparison.Ordinal);
    }

    /// <summary>Masks secrets inside free text (one log line, one stderr line).</summary>
    public static string RedactText(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return text;
        }

        var s = NamedValue().Replace(text, m => m.Groups["name"].Value + m.Groups["sep"].Value + Mask);
        s = AuthorizationHeader().Replace(s, "Authorization: " + Mask);
        s = BearerToken().Replace(s, "Bearer " + Mask);
        s = WellKnownTokens().Replace(s, Mask);
        s = UrlUserInfo().Replace(s, Mask);
        return s;
    }

    /// <summary>
    /// Masks secrets in an argument vector: <c>name=value</c> forms, the value that follows a
    /// sensitive flag (<c>--token abc</c>), and anything that looks like a bare token.
    /// </summary>
    public static IReadOnlyList<string> RedactArgs(IReadOnlyList<string> args)
    {
        var result = new List<string>(args.Count);
        var maskNext = false;
        foreach (var a in args)
        {
            if (maskNext)
            {
                result.Add(Mask);
                maskNext = false;
                continue;
            }

            var eq = a.IndexOfAny(['=', ':']);
            if (eq > 0 && IsSensitiveName(a[..eq]) && !LooksLikePath(a, eq) && !a[..eq].Any(char.IsWhiteSpace))
            {
                result.Add(a[..(eq + 1)] + Mask);
                continue;
            }

            if (a.Length > 1 && (a[0] == '-' || a[0] == '/') && eq < 0 && IsSensitiveName(a))
            {
                result.Add(a);
                maskNext = true;
                continue;
            }

            result.Add(RedactText(a));
        }

        return result;
    }

    // "C:\keys\x" is a drive path, not key:value; do not mangle it.
    private static bool LooksLikePath(string arg, int sepIndex)
        => arg[sepIndex] == ':' && sepIndex == 1 && char.IsLetter(arg[0]);

    /// <summary>Joins an argument vector into a display command line, quoting where needed.</summary>
    public static string JoinForDisplay(string exe, IReadOnlyList<string> args)
    {
        var parts = new List<string>(args.Count + 1) { Quote(exe) };
        foreach (var a in args)
        {
            parts.Add(Quote(a));
        }

        return string.Join(' ', parts);
    }

    private static string Quote(string s)
        => s.Length == 0 || s.IndexOfAny([' ', '\t', '"']) >= 0 ? "\"" + s.Replace("\"", "\\\"", StringComparison.Ordinal) + "\"" : s;
}
