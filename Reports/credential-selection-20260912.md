# Bitwarden command selection — 2026-09-12

The installed machine helper selected a stale `~/bin/bw.cmd` wrapper before the
working native `bw.exe`. The wrapper failed and the status helper returned
`unknown`; both installed native executable paths independently returned
`locked` with exit code zero.

`~/.machine/SecretBackendUtilities.ps1` now resolves `bw.exe` with
`Get-Command -CommandType Application` before considering the existing Node and
CMD fallbacks. A native result uses an empty argument prefix. If no native
executable exists, the existing direct Node execution and other fallback logic
remain available. The private module remains outside Git under the repository's
machine-secret policy.

The explicit `Tests/LocalMachine/CredentialRepairs.Local.ps1` suite passed
**28 tests, zero failures, zero skipped**. Two new regressions verify native
selection despite stale wrappers and preservation of the direct Node fallback.
The remaining fixtures exercise the installed secrets initializer, protected
archive flow, and session bootstrap using nonsecret external-I/O fixtures.

A fresh live invocation of the real helper returned `Kind: native-executable`
and `VaultStatus: locked`. These are status-only checks: no real vault login,
unlock, synchronization, export, or decryption was performed. A locked vault
still requires the normal authenticated session before operations needing
vault contents. The tests remain opt-in through `New-PesterContainer`; ordinary
CI discovery does not import workstation-private modules.
