# Local machine regression tests

This suite validates the installed workstation credential modules as well as the
repo-owned archive script. It is invoked explicitly because `Tools/SystemScripts`
policy intentionally keeps `.machine` secret modules outside the repository.

Requirements: Windows, PowerShell 7, Pester 5, and the installed
`~/.machine/SecretsTier.psm1` and `~/.machine/SecretBackendUtilities.ps1`.

```powershell
$result = Invoke-Pester -Path ./Tests/LocalMachine/CredentialRepairs.Tests.ps1 -Output Detailed -PassThru
if ($result.Result -ne 'Passed') { throw 'Credential regression tests failed.' }
```

The tests exercise actual initializer, private-file ACL, and archive logic.
External Bitwarden operations and cloud/cache access use nonsecret fixtures;
the suite does not login, unlock, synchronize, export, or decrypt the real vault.
Temporary User environment variables use unique names and are removed afterward.
