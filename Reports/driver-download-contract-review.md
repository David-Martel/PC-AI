# Driver download safety and existing-byte integrity

[Install-DriverUpdate](../Modules/PC-AI.Drivers/Public/Install-DriverUpdate.ps1)
now obtains ShouldProcess approval before creating its download directory or
calling the downloader in either install or DownloadOnly mode. `-WhatIf` returns
the existing WhatIf result without creating a directory, reading a cached hash,
downloading, extracting or launching an installer. Normal DownloadOnly still
returns the downloaded file without executing it.

[Invoke-TrustedDownload](../Modules/PC-AI.Drivers/Private/Invoke-TrustedDownload.ps1)
now verifies existing bytes when ExpectedSha256 is supplied. A mismatch or hash
read failure refuses the cached file and preserves it without deletion or
implicit redownload. Matching files are reused. The existing optional-digest and
explicit ForceDownload contracts remain available. Cached lookup and hashing
use literal paths, including filenames containing brackets. Cleanup of a failed
new download also uses its literal filename and preserves wildcard-matching
siblings.

The retained [before-source pins](../.pcai/integration/driver-download-contract-r1/before-pins.json)
bind the original public function, helper and fixture. The
[R2 selector register](../.pcai/integration/driver-download-contract-r2/selector-review-register.json)
shows all 48 original selector names and body hashes unchanged at that repair.
The [R3 register](../.pcai/integration/driver-download-contract-r3/selector-review-register.json)
records two subsequent fixture corrections, with the other 53 bodies unchanged.
Seven added
selectors in the [driver fixture](../Tests/Unit/PC-AI.Drivers.Tests.ps1) expand to
ten cases, each with direct Protects, Detects, Needs and Breadcrumb comments.
The fixture invokes the maintained functions with a private registry and known
three-byte files. Its web-response boundary writes only inert private fixture
bytes; real network, installer and extraction calls are forbidden.

Validation used the installed Pester 5.7.1 package in fresh PowerShell processes:

| Receipt | Passed | Failed | Skipped | Not run | Scope |
|---|---:|---:|---:|---:|---|
| [Original failing first](../.pcai/integration/driver-download-contract-r1/failing-first/result.json) | 3 | 6 | 0 | 48 | Original source, first nine controls |
| [Historical R1 strict caller](../.pcai/integration/driver-download-contract-r1/full-after/result.json) | 55 | 2 | 0 | 0 | First repair, entire fixture, StrictMode Latest |
| [Literal-cleanup failing first](../.pcai/integration/driver-download-contract-r2/failing-first/result.json) | 9 | 1 | 0 | 48 | Current ten controls against pre-cleanup helper |
| [Current narrow after](../.pcai/integration/driver-download-contract-r2/narrow-after/result.json) | 10 | 0 | 0 | 48 | Current source, new tag, StrictMode Latest |
| [Current full CI-equivalent caller](../.pcai/integration/driver-download-contract-r2/full-ci-mode/result.json) | 58 | 0 | 0 | 0 | Current source, entire fixture, default CI strictness |
| [Exact R2 strict baseline](../.pcai/integration/driver-download-contract-r3/full-failing-first-r2/result.json) | 56 | 2 | 0 | 0 | Before the two legacy fixture corrections |
| [R3 strict caller](../.pcai/integration/driver-download-contract-r3/full-strict-after/result.json) | 58 | 0 | 0 | 0 | Current fixture, StrictMode Latest |
| [R3 CI-equivalent caller](../.pcai/integration/driver-download-contract-r3/full-ci-mode/result.json) | 58 | 0 | 0 | 0 | Current fixture, default CI strictness |

All runs retain complete NUnit XML, JaCoCo XML, logs and source-before/after
hashes. No source changed during a run. Containers and blocks have zero failures.
The historical strict failures accessed `.Count` on a null Compare-DriverVersion
result. The two fixtures now collect pipeline output into arrays with terminating
errors enabled, retaining their zero-count expectations. Each also invokes the
same classifier with its inclusion flag and checks one exact status/device row;
the Current case checks both versions. An always-empty or erroring classifier
cannot satisfy these controls. Production classifier behavior is unchanged.
The two full current runs close this fixture gap; the original failure receipts
remain preserved. The mistakenly broadened initial R3 baseline is explicitly
non-admitted; only the exact corrected selection above supports comparison.

[Parser and source pins](../.pcai/integration/driver-download-contract-r2/parse-and-pins.json)
record zero parse errors.
[Original analyzer comparison](../.pcai/integration/driver-download-contract-r1/analyzer-comparison.json)
and [current analyzer receipt](../.pcai/integration/driver-download-contract-r2/analyzer.json)
record zero diagnostics for the two functions with the repository configuration.
Full current fixture coverage of these two files is
117/212 commands (55.19%); this is diagnostic scope only. Repository coverage,
current-head CI, actual vendor downloads and driver installation were not run or
qualified here. No performance gain is claimed.

Current source SHA256:

| File | SHA256 |
|---|---|
| Public function | `2599DBE6F21FA74E1A60579DCB8043D5BBE765077E57077F295F10A4A11484AA` |
| Private helper | `0C89B47AEF3361DA46413E09BBC7E8A8DBF089AFF20B015A3289DBDE51B0398C` |
| Driver fixture | `6A9897A31C14D682F0468A881A208E1762235A88EE255A5842DF759FEDE64EE8` |

The [R1 preservation inventory](../.pcai/integration/driver-download-contract-r2/r1-preservation.json)
retains the entire earlier source/evidence packet and its manifest unchanged.
Distinct [R2 source/controls review](../.pcai/integration/driver-download-contract-peer-r2/review.json)
and [R3 fixture review](../.pcai/integration/driver-download-contract-peer-r3/review.json)
accept the frozen source and retained Pester evidence within this scope. The
repository entry point links this report. The integrating owner must commit the
bounded change, then run the full
unchanged 85% repository gate. The private validation packet is retained locally;
its receipts must accompany any review that depends on them.
