namespace PcaiNative;

/// <summary>
/// Handles safety verification for destructive tools.
/// Defaults to requiring user confirmation for impactful actions.
/// </summary>
public sealed class SafetyInterlock
{
    /// <summary>
    /// Delegate for requesting user confirmation.
    /// Returns true if the user approves the action.
    /// </summary>
    public delegate Task<bool> ConfirmationHandler(string toolName, string description);

    private readonly ConfirmationHandler _handler;

    public SafetyInterlock(ConfirmationHandler handler)
    {
        _handler = handler;
    }

    /// <summary>
    /// Verifies if a tool call is safe to proceed.
    /// </summary>
    public async Task<bool> VerifyActionAsync(string toolName, string description, bool isDestructive)
    {
        // If the tool is not marked as destructive, proceed
        if (!isDestructive) return true;

        // Otherwise, request confirmation
        return await _handler(toolName, description);
    }

    /// <summary>
    /// Default TUI confirmation handler that uses Console.ReadLine.
    /// </summary>
    public static Task<bool> ConsoleConfirmationHandler(string toolName, string description)
    {
        Console.Beep(); // Audio alert for safety prompt
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("\n[SAFETY INTERLOCK]");
        Console.WriteLine($"The agent wants to execute: {toolName}");
        Console.WriteLine($"Description: {description}");
        Console.Write("Are you sure you want to proceed? (y/n): ");
        Console.ResetColor();

        var input = Console.ReadLine()?.Trim().ToLowerInvariant();
        return Task.FromResult(input == "y" || input == "yes");
    }
}
