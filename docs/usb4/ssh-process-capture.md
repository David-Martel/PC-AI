# Windows SSH process inheritance

Dot-source the [bounded runner](../../Tools/SystemScripts/Networking/Invoke-VigilBoundedProcess.ps1) in a fresh PowerShell 7 process. No package build or service installation is needed. Run the [offline controls](../../Tests/Portable/Test-SshStdioInheritance.ps1), then execute an already pinned command:

```powershell
pwsh -NoProfile -File Tests/Portable/Test-SshStdioInheritance.ps1
. ./Tools/SystemScripts/Networking/Invoke-VigilBoundedProcess.ps1
Invoke-VigilBoundedProcess -FilePath ssh.exe -Arguments @('-n', '-T', 'enrolled-peer', 'hostname') -TimeoutSeconds 10
```

The [peer SSH guide](peer-ssh-access.md) covers enrollment and firewall scope. This helper creates new synchronous anonymous stdio pipes and a suspended child, places that child in a private kill-on-close Windows job, and then runs it. Input, execution and output share one deadline. Parent exit closes the job and terminates descendants, including those that inherit stdio. Existing custody tracking and retry remain unchanged. `ExitCode`, `Stdout`, `Stderr`, UTF8 input without a BOM, exact Windows arguments and the public function signature are unchanged. Stdin remains memory-only; no capture or credential spool file is created.

## Root cause and correction

On Carbon-two's reverse USB4 path, installed Microsoft-signed Windows OpenSSH 10.0 and 10.4 printed the peer hostname but hung during channel shutdown. Git OpenSSH 10.6 completed through the same anonymous pipes. The remote PowerShell parent carried OpenSSH's private `c28fc6f98a2c44abbbd89d6a3037d0d9_POSIX_FD_STATE` environment entry, which describes handles inherited from its own ancestor. The [upstream descriptor initialization](https://github.com/PowerShell/openssh-portable/blob/v10.4.0.0/contrib/win32/win32compat/w32fd.c) warns that a regular intermediary can pass stale descriptor state to another POSIX child. Importing that state reclassified the runner's freshly created handles incorrectly.

The runner now snapshots the native Windows environment and constructs a separate Unicode child block omitting only that private descriptor entry and `OPENSSH_STDIO_MODE`. Both describe stdio handles that this launcher has replaced. The child's native OpenSSH therefore uses its normal synchronous-handle interpretation. Hidden drive-directory entries and all unrelated environment entries remain byte-for-character unchanged in their native ordering. The parent environment stays unchanged. The private OpenSSH chroot declaration remains intact; the fix changes no jail, authentication, key, configuration, service, profile, route or firewall.

A file-capture experiment completed small commands but exposed further large-output and handle-mode interactions. It was retired from the implementation after the inherited-descriptor cause was verified. No software reinstall was needed for this defect. Qualification used the actual installed signed native 10.0 and 10.4 clients, existing strict pinned aliases and unchanged endpoints: hostname exited zero; a 400 KB inert UTF8 input returned exactly with separate stderr and a nonzero remote command status. These are process-transport results, separate from repository authorization or physical link lifecycle validation.

## Detecting controls

The [selector register](../../Tests/Portable/ssh-process-capture-review.json) binds source and tests. The offline selector group injects inert ancestor metadata, launches a real PowerShell child and checks that only the incompatible metadata disappears. It also verifies unrelated Unicode environment and chroot preservation, parent immutability, exact argv, working directory, 400 KB UTF8 stdin, separate stderr and exit 9. The baseline runner fails this selector with `Child inherited stale OpenSSH stdio metadata`; the corrected runner passes. The existing [six unit selectors](../../Tests/Unit/VigilBoundedProcess.Tests.ps1) protect argument quoting, large stdin, immediate EOF, blocked-input deadlines, orphan termination and launch failures.

Live qualification is opt-in and needs a caller-supplied installed client plus an existing strict pinned alias. `LiveWindowsPeer` additionally sends a large inert UTF8 payload to profile-free peer PowerShell. Both input and output encoding are declared by the peer fixture, so a remote console code page cannot masquerade as transport corruption. Missing declarations produce `NOT_RUN`; offline results never imply peer readiness.

```powershell
pwsh -NoProfile -File Tests/Portable/Test-SshStdioInheritance.ps1 -LiveSshClientPath 'C:/Program Files/OpenSSH-Server/ssh.exe' -LivePeerAlias dtm-p1gen9-usb4 -LiveWindowsPeer
```

The runner preserves the SSH process status it receives. A remote shell can map its command's nonzero status before SSH returns it; that is a separate shell behavior. SFTP binary checks, Git authentication and deployment admission remain separate qualifications. This remains a UTF8 text process API with its existing memory capture behavior, not a binary transfer or output-quota API.
