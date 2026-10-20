using System.Runtime.Versioning;

// The launcher uses Job Objects, the Application event log and the registry. Windows only,
// by design (this repo targets the Windows workstation; see CLAUDE.md "Platform scope").
[assembly: SupportedOSPlatform("windows")]
