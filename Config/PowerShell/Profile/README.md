# PowerShell Profile Bootstrap

`Microsoft.PowerShell_profile.ps1` is the sanitized, Git-backed bootstrap for
this workstation's PowerShell 7 profile system.

Deploy the same file to both possible user-document roots:

- `%USERPROFILE%\Documents\PowerShell\Microsoft.PowerShell_profile.ps1`
- `%USERPROFILE%\OneDrive\Documents\PowerShell\Microsoft.PowerShell_profile.ps1`

The bootstrap intentionally contains no credentials or machine-specific secret
values. It delegates to the private machine-local canonical profile at
`%USERPROFILE%\.config\powershell\Microsoft.PowerShell_profile.ps1` and keeps
the local module root authoritative.

Do not copy the canonical profile into this public repository without first
decomposing it into sanitized public configuration and private local fragments.
After deployment, compare SHA-256 hashes of the Git source and both installed
copies, run PSScriptAnalyzer, and start a fresh interactive PowerShell session.
