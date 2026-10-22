# System Scripts

This folder is the repo-owned home for workstation scripts that were previously
spread across Task Scheduler actions, `C:\Scripts`, `~\.machine`,
`~\.local\bin`, `~\bin`, OneDrive PowerShell script folders, and selected
`unifi_api` startup helpers.

## Migration Tool

Use the migration script from the repo root:

```powershell
pwsh .\Tools\Migrate-SystemScriptsIntoRepo.ps1 -DryRun
pwsh .\Tools\Migrate-SystemScriptsIntoRepo.ps1 -Apply
```

The script supports `-h`, `--help`, `-DryRun`, and `--DryRun`. Apply mode
emits a JSON report under `Reports\system-script-migration-<timestamp>.json`.

## Scheduled Tasks Repointed

The 2026-04-30 migration repointed these PowerShell scheduled tasks to repo
paths:

| Task | Repo script |
|------|-------------|
| `\BW-Auto-Unlock` | `Tools\SystemScripts\Machine\Unlock-BwVault.ps1` |
| `\Bitwarden\Initialize-MachineSecrets` | `Tools\SystemScripts\Machine\Initialize-SecretsAtBoot.ps1` |
| `\DevEnvironmentStartup` | `Tools\SystemScripts\TaskScheduler\PowerShellScripts\Start-DevEnvironment.ps1` |
| `\Gemini-CLI-Update-stable` | `Tools\SystemScripts\TaskScheduler\Gemini\check-releases.ps1` |
| `\LspmuxServer` | `Tools\SystemScripts\LocalBin\Start-LspmuxServer.ps1` |
| `\PowerShell\ProfileLogSync` | `Tools\SystemScripts\Machine\Sync-ProfileLogs.ps1` |
| `\UDP Socket Monitor` | `Tools\SystemScripts\Machine\Monitor-UDPSockets.ps1` |
| `\UnifiUdmDriveStackStartup` | `Tools\SystemScripts\unifi_api\scripts\windows\Start-UDMDriveStack.ps1` |

`UnifiUdmDriveStackStartup` remains disabled until OneDrive has a clean
post-repair sync-health window.

## Source Groups

- `C-Scripts`: legacy WSL, Docker, VHD, startup, and registry-credential tools
  moved from `C:\Scripts`.
- `HomeRootArchive`: archived home-root scripts with historical registry,
  network, cloud-sync, GCP, npm, MCP, and encoding repair utilities.
- `LocalBin`: selected system-modifying scripts moved from `~\.local\bin`.
- `Machine`: selected scheduled-task scripts and reviewed credential backend
  code; private configuration, logs, and caches remain outside this repo.
- `TaskScheduler`: scripts that are direct scheduled-task targets or companions.
- `UserBin`: selected system-modifying scripts moved from `~\bin`.
- `unifi_api`: UDM Windows and on-boot helper scripts required by the migrated
  UDM drive-stack task.

## Bitwarden archive maintenance

`Machine/Update-BwArchive.ps1` is the canonical archive entrypoint; the installed
`~/.machine/Update-BwArchive.ps1` forwards to it. `-DryRun`, `-WhatIf`, and `--help`
return before vault access. A real run validates the unlocked session, protects
and verifies archive storage ACLs, then synchronizes and exports into private
staging before publishing the timestamped and latest archives. Native failures
and timeouts leave the previous latest archive intact.

The archive deadline remains 1–600 seconds (default 120). Its Bitwarden CLI
bridge and owned-process wait accept the same finite upper bound; ordinary
backend calls keep their 20-second default. The separate `cmdkey` reader keeps
its 1–120-second bound and five-second default.

Run the Windows fixture suite in `Tests/LocalMachine` after changing the archive
or installed machine credential modules. It does not access the real vault.
Plain JSON remains the compatibility default. Do not change scheduled jobs to
encrypted exports until their recovery procedure has been validated separately.

## Credential backend code and custody

The [backend helper](Machine/SecretBackendUtilities.ps1) and
[tiered module](Machine/SecretsTier.psm1) are canonical code sources. This slice
extracts the existing pair from source commit
`900a5330f6f977bb56d5e5ebff8c9bda5d3a97d2`; it does not integrate the other
work in the fleet-runtime draft.

Install/deploy only reviewed code. Reconcile the two-file
[source manifest](Machine/managed-files.json) with the installed manifest;
preserve unrelated managed files, private configuration and active consumers.
Do not replace a broader installed manifest with this two-file list. There is
no separate build command: PowerShell loads source and compiles its inline
interop when needed. Run a passive import with:

```powershell
Import-Module ./Tools/SystemScripts/Machine/SecretsTier.psm1 -Force
```

Imports neither authenticate nor register providers, read credentials, create
storage or change ACLs. The default `.pcai-credential-state` root is an envelope
for bootstrap/cache children. Its trusted namespace may permit foreign
read/traversal; it must still exclude namespace mutation and reparse ancestry.
The selected bootstrap or machine-hashed cache leaf remains strictly private.
An Explicit override selects its root as the leaf and keeps the strict private
contract. Missing parents are created privately. Existing ACLs are validated,
never repaired by admission or initialization.

Explicit overrides fail rather than falling back. Legacy cache reads require
current-user DPAPI, validated metadata/schema and retained file identity.
Partial or invalid preferred storage prevents rollback to legacy data.
Private publication requires Windows x64 NTFS, trusted ancestry and ordinary
private files. After verified success, the current transaction's exact known
artifacts are pinned and identity/hash qualified before handle-based deletion
under the publication lock. Unknown, changed or interrupted custody is retained
with a sanitized warning; earlier transactions are never swept. Recovery retains
exact file/process custody until cleanup can be confirmed.
The native directory pin normalizes extended-length Win32 paths before opening
deep protected cache/bootstrap transactions; .NET long-path behavior alone is
insufficient for an unprefixed CreateFileW call.
Owner/group/DACL custody does not establish SACL preservation or
an atomic comparison against arbitrary writers.

### Atomic local cache publication

New cache writes publish one DPAPI envelope at the existing `secrets-cache.enc`
path. One atomic publication binds ciphertext, machine selector, generation,
timestamp, source and secret count. The authenticated plaintext repeats those
metadata fields, so consuming the cache verifies both the public envelope and
the decrypted payload before returning secrets. A failed publication preserves
the previous usable cache; there is no second metadata publication to fail.

Valid protected v1 cache/manifest pairs and authenticated read-only legacy pairs
remain readable. A recognized malformed or unsupported envelope is refused
without rolling back to legacy data. `Get-CacheStatus` checks metadata and
ciphertext integrity without decrypting; it does not certify the DPAPI payload.
After a verified replacement, obsolete v1 metadata is retired only when its
previously observed identity, bytes and owner/group/DACL still match, under
retained file custody. Changed, unqualified or ambiguous metadata is preserved
with a sanitized warning. Cache publication does not repair ACLs or sweep
earlier transaction artifacts.

The new reader accepts legacy v1 pairs and new v2 envelopes. An already loaded
older SecretsTier module cannot read v2. Accepted deployment must refresh or
reimport cache-reading and cache-writing module instances on each host before
its first v2 cache publication, using the existing module/runtime qualification.
Copying source files alone does not establish runtime convergence. No live cache
migration is included in this qualification.

At profile initialization, a failed cloud cache publication adds a sanitized
warning and continues through the existing local-cache admission path. Valid
cache data is used normally; expired data retains the existing explicit age
warning and IgnoreAge behavior. Missing data remains unsuccessful. Cache
validation and environment-hydration errors remain outside the publication
catch, so this recovery does not admit unsafe storage or unvalidated values.

### Cloud backup publication and compatibility

`Export-SecretsToBwVault` binds the payload and note name to the explicitly
selected machine. New exports encode UTF-8 JSON as base64, stage bounded ASCII
parts under a unique generation, and publish the canonical v2 manifest last.
The manifest binds machine, generation, part count, lengths, encoding and raw
payload SHA-256. Import validates the whole snapshot before saving cache or
hydrating the environment. An interrupted part write leaves the previous
committed snapshot usable. Cleanup targets only exact observed owned parts;
it does not enumerate or delete notes by wildcard, or serialize all writers.

Complete legacy single-note JSON remains readable. Legacy `#chunked` multipart
imports are refused because their positional parts cannot establish one
export generation. Re-export once from a known complete source to create v2;
refusal leaves the existing cache and environment unchanged. Explicit
`Sync-MachineSecrets` forces a cloud refresh rather than accepting a valid but
potentially stale cache. The synthetic controls below qualify these software
boundaries without accessing a real vault or changing deployed credentials.

### Windows synthetic qualification

The [custody selectors](../../Tests/Unit/SecretBackendCustody.Tests.ps1) use
synthetic protected directories, owned child processes, inert process-local
provider interfaces and mocked network/native boundaries. After assertions and
process recovery, each run retires only its freshly created synthetic root, with
containment, identity and reparse checks; it never sweeps earlier roots. They exercise no live
vault, drive mapping, cloud note, hardware or authentication. Run the explicit
pair-source gate below; the ordinary Modules-only coverage glob does not
measure these shipped helpers. The Windows CI jobs are disabled, so a green
portable CI job does not replace this Windows execution.

```powershell
Import-Module Pester -MinimumVersion 5.7
$c = New-PesterConfiguration
$c.Run.Path = './Tests/Unit/SecretBackendCustody.Tests.ps1'
$c.Run.PassThru = $true
$c.CodeCoverage.Enabled = $true
$c.CodeCoverage.Path = @('./Tools/SystemScripts/Machine/SecretBackendUtilities.ps1','./Tools/SystemScripts/Machine/SecretsTier.psm1')
$c.CodeCoverage.OutputPath = Join-Path $env:TEMP 'pcai-credential-coverage.xml'
$c.CodeCoverage.CoveragePercentTarget = 85
$r = Invoke-Pester -Configuration $c
if ($r.Result -cne 'Passed' -or $r.CodeCoverage.CoveragePercent -lt 85) { throw 'Credential qualification failed' }
```

The imported baseline passed 69 Windows cases but covered only 808/2336
commands (34.589%) across both files. The envelope controls first reproduced
six failures at the root-private guard, then passed all 16 focused cases after
the selected-leaf correction; the ten unsafe-root/leaf/ancestor controls kept
refusing their inputs. The three backup/refresh regressions then reproduced
explicit-machine mismatch, same-count mixed-generation import after interruption,
and a valid cache masking an explicit refresh. The v2 controls cover Unicode,
manifest and payload tampering, malformed encodings, bounded concurrent reads,
exact cleanup custody and the selected legacy compatibility contract.

Successful private-file cleanup first failed two controls (one retained receipt
for creation; four retained artifacts for overwrite). Metadata-change controls
then reproduced same-identity/same-byte ACL changes being deleted; the repair
preserves changed owner/group/DACL. The native directory pin also requires an
extended-length Win32 path for deep default cache/bootstrap transactions.

The initial signed extraction gate passed 271 cases with no failures or skips in 25.899 seconds,
covering 2287/2648 commands (86.367%): helper 1566/1815 (86.281%), module 721/833
(86.555%). The current synthetic root was removed after process/handle recovery.
The unchanged repository analyzer settings admitted zero Warning/Error findings;
unconfigured heuristic warnings remain documented for review. These are synthetic
software results; live provider authentication and fleet deployment are separate.

The 271-case execution used test LF pin
`D793DCC33A704D6154BF170BBC35D813CAEF24B27FF5A3D76CE4C746100BB1BE`.
Subsequent rationale/breadcrumb corrections changed comments only: the
noncomment-token SHA-256 stayed
`BAB8367CA0FC3CE1B7FB3C0B50FC722666B127B3552EA2EE39EF5E6818579628`;
all 190 `It`/setup/teardown body hashes and all 153 selector hashes were unchanged
during that rationale-only transition. Seven subsequent exact-line secret-scan
pragmas label proven synthetic fixtures; their seven textual selector hashes
changed while the same noncomment-token seal remained intact.
The historical full test pin below includes those reviewed comment corrections.

The subsequent atomic-cache/deadline revision passed 303 cases with no failures,
skips, failed blocks or failed containers in 39.438 seconds. Explicit coverage
across both complete shipped files was 2451/2819 commands (86.946%): helper
1566/1815 (86.281%), module 885/1004 (88.147%). The unchanged configured analyzer
reported zero Warning/Error findings. An independent Windows replay passed all
303 cases in 25.820 seconds, with empty stderr and no remaining newly owned
fixture roots. Its code and tests are bound separately below; the original
153 selector bodies remain identical to signed commit `75b815231d78199d5516b1bbad12131de1c5dfe1`.

The subsequent profile-fallback repair passed 307 cases in 39.745 seconds,
with no test/block/container failures or skips and empty stderr. Pair coverage
was 2455/2823 commands (86.964%): helper 1566/1815 (86.281%), module 889/1008
(88.194%). The unchanged analyzer configuration again reported zero
Warning/Error findings. An independent Windows replay passed all 307 cases in
26.611 seconds with no test/block/container failures or skips, empty stderr
and no remaining newly owned fixture roots. Its review and pins are recorded
separately below.

### Historical signed extraction review register

Each row identifies one `It` selector or its explicit parameterized group.
Protects/Detects/Needs/source breadcrumbs sit directly above each selector.
Hashes use UTF-8 with CRLF converted to LF. Each row inherits the historical
source/test pins and review record below. These records describe signed commit
`75b815231d78199d5516b1bbad12131de1c5dfe1`; current source changes require the
separate revision review below.

Review base: `761c658a3ec7569661a3476f69477aa15d277fa8`.
Reviewer/type/date: independent Codex reviewers listed per row / 2026-10-22 UTC.
Scope: root reviewed the complete six-path extraction, production dependency
closure, 62 selector groups and the independent 271-case Windows execution;
credential_custody_audit reviewed 87 custody/helper groups; bus_installer_resume
reviewed four private-file cleanup groups and helper source only. The bus author
does not approve its own SecretsTier module. All 153 groups are accounted for.
Root additionally reviewed the seven synthetic-fixture pragma annotations
on 2026-10-22 (annotation-only scope); row authorship reviews remain partitioned.
Root independently verified the comments-only pin transition and all selector
hashes; reviewers used the source/test pins below. These reviews certify the
synthetic software boundary, not live provider authentication or fleet deployment.

Tools/SystemScripts/Machine/SecretBackendUtilities.ps1: `F9E9F13853B291752FD379E4C4CAF943168A8CD0DCC40A1E843958B8F9E61CBD`.
Tools/SystemScripts/Machine/SecretsTier.psm1: `DEA3EBA8D532F9D2CDA94BB8AC5CA634836E737098A7BA6CB2D677818C409BDA`.
Tests/Unit/SecretBackendCustody.Tests.ps1: `07B7FAB5EFB28D74AB9D14F8371F6403416EAF38B0C6A3721A3A6F4CCAD80E39`.

| Selector/group | Selector SHA256 | Disposition | Reviewer | Review type | Date (UTC) |
| --- | --- | --- | --- | --- | --- |
| Isolated Windows credential custody fixtures / Owned process custody and passive configuration / Exact owned process custody with real children / retains the exact live child and output tasks after Kill refuses, then recovers without touching a control | `96139008CF0597B20F94BF39EB9FCBC976C78B00CF9B19956C0D2F533ED5A072` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Owned process custody and passive configuration / Exact owned process custody with real children / retains exact process and tasks when OS exit confirmation refuses, then verifies normal recovery | `A1B15D76542BBDBCC7E1B7AA5FD3F704BBD0DA37E2F4A794FA0C1A8BF2ABE0AE` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Owned process custody and passive configuration / Exact owned process custody with real children / cleans up after a genuine disposed output reader raises an external IO failure | `C0E6507942B0EB46F08E8B5A0B96698366624A4C6DDF6324277FDD5FF04C81BB` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Owned process custody and passive configuration / Exact owned process custody with real children / rejects an unknown recovery identity without touching the unrelated control | `B763CC6DE93B1F39F122563E498E899BBE3649ED7C489F0B6AEF5DA2D372F307` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Owned process custody and passive configuration / Exact owned process custody with real children / withholds successful output and retains exact completed pipes when pipe confirmation refuses | `DDEAD0A3C54AFDDD5DD8C5497B28824907FED643D08F3F8EE54F702670B75D93` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Owned process custody and passive configuration / Exact owned process custody with real children / retains partial real read setup and recovers after an IO failure plus refused Kill | `F439955889E4B5E40121E59F3126812840297B7422FAED79E63DEA4986A446CD` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Owned process custody and passive configuration / Exact owned process custody with real children / retains the exact blocked stdin task and original child through refused termination and explicit recovery | `86832AE23C392D555F64A1141BA146B0BC0F66C6893832342D5C790596167D95` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Owned process custody and passive configuration / Pending exact custody blocks new real backend admission / withholds all output and exposes only opaque pending metadata before and after helper reload, then admits after exact recovery | `C0F84F9F50B5431DD2311CAB4245F0C436406CEC4D6F147D3806894D5AA3ADCC` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Owned process custody and passive configuration / Pending exact custody blocks new real backend admission / blocks a newly imported module from starting a different backend while retaining the original host entry | `3DEC6332485C1004AB33E94F14F36B18C8C8C617D9777884B36E94326BDF1D15` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Owned process custody and passive configuration / Canonical credential source and passive machine configuration / declares exactly the reviewed two sibling managed sources | `5835B7D44DA6A3D7754ED9F14F58E1423ABF5D191F91C9BFE35E0A6AF85AA62D` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Owned process custody and passive configuration / Canonical credential source and passive machine configuration / imports the actual sibling module without provisioning or changing process credentials and exposes the intended contract | `D143D6B150D809143E8E6AA83E072967745C8B800E009A02B72AF7FE3A900702` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Owned process custody and passive configuration / Canonical credential source and passive machine configuration / uses the explicit machine selector and cache root without provisioning the selected path | `396A87152E66F041DDF0AA5F3B29FA76E6E89E7D25F62A9B355B7E200416070F` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual default protected bootstrap integration / uses protected profile bootstrap despite actual foreign-write synthetic TEMP | `8A0E74F2582C6CBFD58DE99C0CFF466055022E44F41511A60E5941D4C7C07A19` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual default protected bootstrap integration / WhatIf provisions no storage and calls no provider | `16B940CFC64260B56609301765C279A0E68240FD9AE7DFBD194D1D91FB32088C` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual default protected bootstrap integration / invalid explicit root blocks even existing-session status without mutation | `9846F808F4C3F0003891F0C79FACE68AFAD690D8BF1CB072C525FB4C98F971D0` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / import and WhatIf create no directories or provider effects | `551C272EF3786FB2BF7F9B4762973D8BD801373B9B3DC3F00408BDA7D47A5394` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / default save publishes actual private DPAPI cache without changing legacy namespace | `3A58C4B4D1FF1F538D0E39AFC6FC00B65CEE04FDE5A288F39DAE0B74311B20EA` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / refuses preferred path directories instead of falling back to legacy (<Both>) | `98561200A297DA00833741C85281437D0DCB2C85A455F7D6E3BF90B1229A2B2C` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / reads a literal long private path and refuses an actual named alternate stream | `302052906565F7BD8C81E894038E0C4AB3620141A2C7167E46D2FAEE54B83B1F` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / reads actual private legacy DPAPI data without provisioning migrating or hydrating | `ACB4C1D117AD917D518C7B81D458F3D2DA236440FA35BC992E4E5524D5A9C64E` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / legacy metadata status never decrypts secret plaintext | `4E4F8FE8982CFA64CECCEA2439711637D9D513C746658EE0292028ABDE1058A5` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / partial preferred pair never falls back to usable legacy | `C50034BF4391C706499C0DA0F7CD55CAB33A5CBE1946CCBAABE130116ADA9008` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / invalid authenticated ciphertext is refused without private content in errors | `1C4917A5ED0445087E30EB559071CBE9F5EA201C01B55886FE9E4F1DB7BE963C` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / valid DPAPI with an invalid payload schema fails before returning values | `75FC0C2879BF7EF98BA0B047155D624F45FE5096EE8749700905948A05140B67` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / foreign-read legacy file grant is rejected without ACL repair | `F6BC7745B24F621673AAF07D88C967FBE89BAF7C1164ADABFFEF93C0849646DE` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / retained identity refuses same-byte namespace substitution after real DPAPI decrypt | `2CD5592B62BF27C23EAECCAEE80D3EEB62F9F502B321A66E8E65F66CD9A228CB` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / invalid explicit cache root is refused during passive import | `0735D44A79E83504CC72DB103FDE022CD31D0A492A15F33865970A30C585E9D5` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / explicit machine root preserves its selected cache path and ignores legacy fallback | `E8B9A66849EDD6D9927AF32FE1FF8667F517448E3ADAF112F4F92271AEF02C28` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Read-only root envelope and strictly private selected storage / admits a read-only envelope for <Operation> <Purpose> without changing any ACL | `A055A74FD335C036A9FA8A3CAAD17E7D3E4301EFEC3A17B17062F7EB5E1AF835` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Read-only root envelope and strictly private selected storage / provisions missing private <Purpose> children under an unchanged read-only envelope | `9F51E9A7696787BAA6828DD91ACB3D365575DB1F6ED42664452BD91DA5702EE5` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Read-only root envelope and strictly private selected storage / refuses a read grant on the Explicit root selected as the secret leaf during <Operation> | `273FAF7F42B21EBD61DE22A45B1A312E9768D92C4BAE3FBC71761A91D128BF69` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Read-only root envelope and strictly private selected storage / refuses a foreign-readable selected <Purpose> leaf during <Operation> without ACL repair | `54E0FE8FE6DA9B1546AD53C392A9323B813D552621DFBF4A2D7EEBCC02F511CD` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Read-only root envelope and strictly private selected storage / refuses foreign ancestor deletion authority during <Operation> <Purpose> despite protected children | `6A43F24294FE117E3ABBD6278BDB542E66183878BD768FF4A4676138688A0FF4` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / roundtrips a backup using the explicit machine selector instead of the current host | `C20989AFF26AA97544F9A5F117EEAC9CADC815CC6425AC509A3ABEFE35B00E36` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / preserves the prior complete backup after an interrupted same-count replacement | `5B492FFBAC57E8E7531534314C9C5082465DBF321AC6166FB17D40EB3ADC4F97` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / forces cloud refresh when the operator explicitly synchronizes a valid cache | `4623A791AE5084686B544103D16A675291F29CEA1C45F98BEF5E4AB568C7E285` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / roundtrips CJK and emoji through bounded ASCII parts and explicit encoding metadata | `76B96921D49F0899371E3CB33305C61B7D5BDFA520BC7519EFA448BE1447977C` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / refuses tampered v2 <Field> metadata before cache or environment publication | `E99E541B6170BE37ED8265442B87BD9949B9278E26CAE306B97BF1DF83DE00AC` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / rejects a valid-base64 valid-JSON credential change with an unchanged manifest | `180367182B0033C230F74CFD2F91D93B05EDA0AE1395EFCDCE2B328FC5BBD724` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / retires only the prior committed generation after full replacement validation | `282CAF13F7F2812EB68F7AC496DBC45A28CA57E6FE78C4254930DE64D7175431` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / cleans its own interrupted staging generation while preserving all prior items | `486FA393C123D64AE1040ADABE28CC977338D4CA1EA1D2BD44CABD1B4378F34B` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / retries once when a retired-part read demonstrates a newer canonical generation | `35CE9CDBE129D32D47086E16DDC4C871F651246B8778F537C0921B6F6BDEE7D2` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / refuses malformed v2 manifest <Mode> without publishing values | `F211A27F5CB99DAF34BA09FA1EE7157716EE8201A140AFBBFC94D25D4C92CE67` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / rejects a hash-consistent but invalid snapshot <Mode> | `8B68EFD2F8F8450B5FDAEE36EC8B44F416B53397B5C86673B7B8266FF96F6BCF` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / refuses malformed v2 part <Mode> before publication | `FE49353490BA9A6165B9CD9DE201453FAA9BC2B71E40A65F0C0372F26FB8922B` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / preserves notes when guarded cleanup sees <Mode> | `E1DF869CCF4C7F1A09CA23595656EDEB6804AD3805FB2CB93BE42ECEF2C2BCEB` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / bounds canonical recovery reads under <Mode> | `F9ABBE99BC8E552D89C9D684660189EDF8C1ACB0AFFA34DB9EB4204EB4067E3D` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / preserves an active generation and unrelated notes during cleanup | `6916E80D01E832E95563D2EDD6F556676A9F6A7F3590D08FB4B51E61F6351208` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / keeps a validated commit successful and reports only sanitized cleanup outcome <Outcome> | `C24E5F20FADBCD66097C2151E4F5A28115AF01B668AE6F2A29DE09761760B3F9` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / refuses BWS invocation without an explicit token | `AF05C2052841C29D8304624C4D50C078FA087476A897F27406C186F5550DD86E` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / parses unique synthetic BWS keys without invoking a real executable | `315902BFF60666A64F50F19811BF384E7B4A7E12FA9AAEE9C9354B45B0918FD5` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / withholds invalid or duplicate BWS payloads (<Payload>) | `27115E5A91AB55C102A2EC7A160A86104A1B609FAC1282431B5EF3DC4430B1CB` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / retains a valid cache without querying cloud unless forced | `98003A782BA7B204D5B7471414973A4D1E66EA52488FF379AD56A92CA754ED5A` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / publishes fetched synthetic BWS values only when online | `B6CBCAF02E2863BCAEB80F8141750D08795D507A7CC91087F3A226591CE345F5` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / performs no publication when forced cloud sync is offline or empty (<Online>) | `32A91FB6CDDD30696C90EBB56CD3CB23F42B60988961F43F091732B842582A20` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / serializes secure-note edits on stdin and never puts synthetic private data in arguments | `3B4136EC493D0635065B57E6CF844A543E3CD61F64B7AD9AC3A98DC33A2E3F62` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / creates a new secure note only from an unambiguous empty search | `75E157045148B645595BA6BD995EB0A840BE24D7CDDA2743BBE2F6FEDE9F07C7` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / rejects ambiguous secure-note lookup before writing | `DC00E555F06349AEA5D2ED7E1C1A7D4320C616A3A3B3D942C1FD6BE901886112` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / rejects a changed secure-note identity before writing | `0C59B64AD6F04252E483AC2F8CA52E2A1B0E169F1A56CA482B42ACCC408D4CB7` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / returns only the exact secure-note name and ignores fuzzy/type mismatches | `43DBE03A801B889C9A5285CE30026537E3AF1E6A649F0BBBB2C47FD2DAF5A1A5` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / splits a large synthetic export into bounded v2 parts before committing the canonical note | `3FDC70F62935287ECED05BE36495B5B2C90BBBFD6647F1F6ECA2F89B2871E5DE` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / refuses an expired backup without Force before reading or writing values | `C6A37049B2F3E1AD1C1AB83263035439FDD04790AA83D76EAF3E467EE26299AE` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / refuses a legacy multipart backup with re-export guidance before cache or environment publication | `50EC5253191F6FC97E0312A2D249F82C826F7D16FB94BC5EDCEA54F4EE3BEF35` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / imports a complete legacy single-note backup and honors the explicit environment request | `C5042DDBD78D8FA490D53BC12B0BF2DEEBF100892A6D39E97C1F61180514234D` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / rejects invalid backup part counts before cache publication (<Count>) | `720D398E26EDE0A648445EAFF6FBF3A21C482ADB8B795F90CFC2DB6B86730D81` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / withholds mismatched-machine backup contents and refuses hydration | `33176157CA833DC6433924EBC26CF71AA7296AFC695CE1F84D288F0CF396BD13` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / selects cloud, valid cache, expired cache and manual recovery deterministically (<State>) | `535AC2D28594F42CCF48551ADA446F8F0E7B0F45B061C72FDCEADAB9A63732AD` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / persists only requested keys and refuses missing keys before any User write | `02D6578B6FC78469575B621386311832DC6535A452B9680FD72951C32494CBFF` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic backend discovery and credential normalization / attests unique package files and refuses ambiguous or stale selections (<Tool>) | `DA3797370309F19D2A44328B6A412A8C6093196303E41F33DF19A8FA31A12DB8` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic backend discovery and credential normalization / keeps absent packages missing and refuses WinGet Links as attested packages | `6C7CEDA1B069457DCBA5A01C550A5C3F2FFB55B249CB363818A7CD4BA3F68B88` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic backend discovery and credential normalization / selects discovery paths deterministically without executing them (<Kind>) | `BAEC4AC7F84CE678201E8B6B53264324414B1B7B2ADFE9A5EA9849A2F744A67D` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic backend discovery and credential normalization / launches only native or node specs through the guarded process boundary (<Kind>) | `21EA742F9A809E28E371B890937EC0AE231C40F089BE913108D7B7FBCDE0B6E1` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic backend discovery and credential normalization / reports only safe CLI status metadata (<RequestedState>) | `1F43123CC1E9C2654044630F1261AA5D6BC1B9189D1D7E8C57A456FFF345AA7A` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic backend discovery and credential normalization / imports or installs only selected inert providers and honors WhatIf (<Mode>) | `DA3BB7D7D708A0EEFE1DE6077B20037064639DB0F619CE6C5A253DF1D627CBBF` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic backend discovery and credential normalization / never combines incomplete API credential identities (<Source>) | `A5A436FA2C2BCD63C601EB127EC9F5D119E4345568DDDE7A878F3BDFA6463453` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic backend discovery and credential normalization / normalizes explicit synthetic credential shapes without providers (<Shape>) | `FD5EF1DFD244587D85B28E5B69580419879A4B64A337C40279592F7EE2E893B3` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic backend discovery and credential normalization / refuses changed IDs and fuzzy or ambiguous resource names (<Mode>) | `CA7115996AC4295133256B4EEECB70F28FD2F9B2268FAEB9BF7E18FB9EDE808B` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic backend discovery and credential normalization / parses only exact synthetic native credential targets (<Prefix>) | `7DC435B6F427323103D42CB1BBF6FB7AAC1F72162FD3BF6EB002ECE78FE82D9D` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic backend discovery and credential normalization / withholds private CLI output from public command errors | `1D881F2D8EBABCB14AEBAB0011AD2BA2C58B7A498E9522D29D5D0F05FD9D350D` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / reports backend metadata without fetching credentials or importing providers | `70488A8C2BD204A92A36D5FD999937E7AEAE49F8A80F045D9CEC937AE5DE3F25` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / registers only requested and available synthetic provider backends | `1279235CB3360DAD0513FB1BFE7E7639854DBEB72CE689F77A5884A754E29F67` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / does not re-register an existing synthetic vault or unavailable provider | `BCB828D89FD51D49E076EA4CD5C62E55A0E26529CB11C40963D1F75612D54158` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / publishes a selected helper secret only to an admitted inert vault (<Scenario>) | `5C1A46518FDBECC63E857F3CCEAF80D6E43F3E36E42F504CE27742473996C025` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / returns the first usable candidate vault and never exposes failed provider output | `68DAA3F87D71C52D29CCD8023FE82431D642CE0AF1E36EA430C1F2063540A314` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / reads only an admitted vault and normalizes its explicit synthetic payload (<Registered>) | `2CE54E27EBA3F59D3E7DE3DCC1D5E372F364B1DEA82A48C392C6F30BCCC91D5A` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / initializes and mirrors only the complete selected bootstrap values | `AFD72D45CCFE9F8C8165675A42E5F1430057BEEA0CF6577FFCAA7350B5F4712F` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / registers a module vault through its inert provider interface (<Exists>) | `D108AF51550B09C9C9372703E627AC11B8C6A5278740AAC88A99917814BBA1F8` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / projects module provider status from metadata without secret reads | `5803EAB2137889287D78DD7CB8728B8A40333033493F9BB157CDA320327A55CC` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / publishes only explicit connection payload and correctly typed native target kinds | `CC56C82824C515B9DEF43FEB40782731E3E744728670C2B55555CF0637980595` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / reads only valid explicit connection JSON and treats missing or invalid data as absent (<Payload>) | `38897F9E2104F847DA4D35CEF6CBB8B91B2ACA35CC5E5E6C0D7FC5493CBD05B3` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / sets selected module vault values with metadata through the inert provider only | `7B3D614A283B382E2F366ACF2D87F7496CFEDE85D331FCA9DFAA4D2E52524E97` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / withholds failed module provider reads and returns only explicit synthetic values (<Scenario>) | `CF1A4EE1A13FE4EEC6DC28BA88FD3985ACCF83A97B4642FC7549DACA9734E823` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / reads synthetic machine identity from the isolated module root without registry fallback | `DBBCE9DF49BFFDFABEAE6CD90472DD513196666121DA2DEF486F8BE54D76CC0A` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / projects cache and session metadata without decrypting or hydrating values | `F10013381B448CFEA1945AD4E9DDD7D7E4CC18F2A9DB27660C3BD794E125153F` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / hydrates only a successful explicit sync and reports failure without environment mutation (<Succeeded>) | `4635C0A9C6ABF036B7DA279B1CD975B565E0925729FDF35CF2EB7FB5CEFF1DDE` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / validates every environment name before any synthetic process assignment | `D6760164918BA85EF1E850CACAA49E96A5ECFB6C7291D92CEF4EFF8115AD5D67` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / does not persist absent keys during initialized environment selection | `8354FD3CDB1E41D8E1405B4F25F38CD829014EA1DA11FFA159B913157ED663FC` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic local vault, connection and backend state interfaces / delegates guarded unlock and exposes metadata only | `8442FB2A4593794B0A351291DBAAE4273684EE5B38E91827C16AE9E789A7F45C` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic resource sync decisions / filters case-insensitively and preserves target precedence, status and redaction | `1B0FD91DD25CE802B3D59AD80688B1D5029B27FF9A85AF148DB608244BF84E23` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic resource sync decisions / refuses deployment after failed Bitwarden sync | `4A06F1E391A1FC1A44976E86D68064CB3841DF1A901A1A14E559036D2B00BD15` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic resource sync decisions / performs no credential deployment when prerequisite <Missing> is absent | `0B6D4F5D52C0699816A8CB3155455A140D28E853D42A57E2B4B7EDC2013BE6B2` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic resource sync decisions / returns usable missing-item/password/username metadata without leaking values (<Missing>) | `D8716647FF686F4981E5DF395411C8B4E3B7181B268FAE74B6950EE207C9D6BF` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic resource sync decisions / mirrors only selected synthetic credentials when the inert vault is registered | `18BDEC211CC343F2F724ED05D8F14B914C7042284E4680BBF3D7135B75F91405` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic resource sync decisions / WhatIf does not discover config, authenticate or synchronize | `0FADF01A08DF6E2E032A2BDF95839E9B625EC7302929B43BEE5E85FDA2C6CAF1` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic resource reconciliation projection / projects native metadata and matching display root without deployment | `6F37037BB70E074C0C61CF2558D31F28B55559BD19E452A72EDBA3E25774A85E` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic resource reconciliation projection / does not invent matches or candidates for missing projection fields (<Field>) | `D03D7DDF15C516E115C4C8A88F8DFA422055E23E25983543D2ECA9320627D19D` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic resource reconciliation projection / projects inert registered-vault metadata without fetching plaintext | `2BF6D94051E1E49A72F46E32D794BD930DDA301B243FE186DE77241599EA64BF` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic drive mapping custody decisions / publishes only after the exact mapping postcondition is observed | `0F9ED575CD8D4EBE93C4C55D2A0C8240307C4797D88EEA29DD3ECB69338FA8D9` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic drive mapping custody decisions / refuses replacement of a different mapping or a false observed success (<State>) | `CA85FBC104DBCCB129768BC74420450F90D3EC9DFABE3D47F05A72720997E28E` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic drive mapping custody decisions / does not invoke a native mapper for existing exact Root or DisplayRoot (<Property>) | `72C5D470C2B697EA27D09B49E70FBD8A4D7F0082D8734703AAFCB7F8C2C060A5` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic drive mapping custody decisions / preserves unreachable or failed mapping status without claiming deployment (<State>) | `8669A44CEF1E95A89E4D2AE8D1AC22BC9F74C7BE6171345B8C3F48042ABBAE81` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic drive mapping custody decisions / WhatIf performs no manifest discovery or mapping | `8B041EA439F7A5C003D92B5586494F8B386612C3D18B822CF9871A4A361D8B5E` | KEEP | codex-gen9-fleet-resume (/root) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Ordered share discovery and exact vault targets / deduplicates candidate paths and stops at the first reachable synthetic endpoint | `2E167B90BBB3E433D336ADCD100F19231325F2E91D1C356115510593C5CAF666` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Ordered share discovery and exact vault targets / keeps empty discovery unresolved without issuing network traffic | `68A570744F1BB3899E5D35010DC941E9F5D21CA4488C01986389D3660FF3EB25` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Ordered share discovery and exact vault targets / deploys synthetic vault target outcomes and catches private write failures | `92C56B2CA19A6D9981594020E4DC58720AF3C26210A9E5F36353FE2960B8D0FA` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / creates a genuinely private file without retaining successful transaction artifacts | `6772BEB7C66DB4EF9A2D315249844C17087A028685F681EC4691058DFBC40449` | KEEP | codex-gen9-bus-installer-resume (/root/bus_installer_resume) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / overwrites with restrictive private ACL without retaining a redundant displaced original | `551DD1927BA7EC1ED7CBE83EF3C3DD0634337DF97E770BE091A0AC90F1EC3FE6` | KEEP | codex-gen9-bus-installer-resume (/root/bus_installer_resume) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / preserves complete successful custody when cleanup sees <Mode> | `DCDA752A3A17A988A293A865EC4A939B7792B63D2CC3CE7FE585C75DBD2E262F` | KEEP | codex-gen9-bus-installer-resume (/root/bus_installer_resume) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / preserves same-byte displaced custody whose descriptor changed during replacement | `3FCF4ACC70DD93515BC0BD624E64201B4F58CA5C48B9D47C2B3138034D55B7DE` | KEEP | codex-gen9-bus-installer-resume (/root/bus_installer_resume) | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / never calls Set-Acl on the original target | `1E15C7958322B5498B1888A712C05CD7FE5D46A1E9BBE928143CDDD2B88E06C1` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / preserves original bytes and exact SDDL on actual absent-stage replacement failure | `C7480A1459D1E6392F7C9FAA6FBC59C7ACFD6AB1A7CC3B9E085C4D89EC4DE875` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / restores actual displaced bytes and strict pre-move DACL after partial 1177 failure | `1C89C4D4CB4AFCFA9DAE46F3A628F5BA461BB3286A5D184AF823BAE1FF65D5FA` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / preserves a real later writer already present after partial displacement | `A8660CA8A0C93CDDCE23510A8A834E13F631D0A7A89372EB55656ECE2FD1BC49` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / refuses to overwrite a later writer arriving at the recovery move boundary | `E634B6A6AF158B9A46C7187C2DEC812AC763668FABC9CA03814768B6E40326E2` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / detects a same-bytes foreign file after owned recovery ACL restoration | `CAF2EAB059373B2B682A1E736A5FFEDC19CAC3A8B3D9011B5277D2CC308D54AC` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / retains the original operation error when recovery ACL restoration fails | `6DE820E6E5ACC0CF5596A6547A948FF289A63312E3EC690E1D75E4C4A9C4E794` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / rejects a same-bytes changed predecessor before any secret stage | `43BA3A610BA710C229382723F3D63BA16EE29EDFBBB1515A53EF679A43B8411A` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / does not mutate files or create custody for WhatIf | `BDE887CB533ACAF82C13B88CD823E608B49BF5D9275A373C52E5A4DE8FA030D0` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / rejects overwrite without permission before custody creation | `E2AE8C7BE50ABBE5D7376F551446606DE8F31EF15C28557FBDDAC9A234361EC7` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / rejects actual hardlink identity before custody creation | `7D9D71CA1AF587C9786989DF08DC47965162B5379D316A15303BBE8055579DF7` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / rejects actual alternate data streams before custody creation | `4094790D5C89AC6921FEBD383640C1D5125F05446E416B06C8AA5864FCF20B65` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / rejects actual read-only attributes before custody creation | `DD75EBA7D8B86594B9574CECD6B51DE01BF33783C77C295A318F48CABBF8FF32` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / rejects reserved path components without creating files | `47D1BA81A9579CC15887A1B8672B4934AE9D4803E66F9D395E0D8A1EE0878CD1` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / fails closed for unsupported platform before allocation | `9661A11288A372E5EE1D1B7C0A1BCACA1906F48760703AAED3D39EBA229F0875` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / rejects a genuine junction ancestor before stage allocation | `A674C86722CA1FEB1B859931C9CE76EA469F29FAEEDAF57265CCF8A2C001B7CB` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / rejects an insecure pre-existing custody directory without fixing its ACL | `BC141AEF9A6113486FFFAFE9199D590EBDE90C3A7000F65EB933B5BA776F56AC` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / refuses unsupported exact metadata restoration before any secret bytes or publication | `6850C595F4A0C9220998031688E7837B1A3CCAB9CFB17327F26BF58F7B3002CC` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / captures a same-bytes foreign displacement rather than claiming compare-and-swap | `D25DEB5603A8B965CF48DF47FEE9872AAFF549CD278DD8ACBB70842BE26D8481` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / blocks another cooperating writer while the original publication lock is owned | `D0077A4A7B38EB3BAB8601BF43B120C60D8B9C3F24B1B654101FCE46A08FE768` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / captures the original raw descriptor before the actual protected-directory move | `3AEBBA8FE4BB65A8B8F5D81CDF95A6604635140AE5605D2F41C6119C0941AA98` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / qualifies an ordinary token-owner inherited file and publishes a restrictive replacement | `E91BECE60FB5B0F1D628F9D65C2C72CAA67282D51053F64BFDC357A4CC40231B` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / restores ordinary inherited owner group and DACL exactly after actual partial displacement | `1768F82A0B44B64B9817AC16AA265C6C1BC6DF91A86F99C5CBEB2AD3C7079640` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / rejects supplied reserved trailing space before provider device normalization | `59B001CBEF5056F2AE22D4CEF0C02A0F81DD3B193473A76B4BDD334975C18C73` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / holds cooperating namespace ownership after a swap changes the file identity | `D96199EDD67E66A0DC45934DEA958D94C01032CCDB3156B0B6D40A791338C691` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / refuses a broadly readable predecessor before real Replace can expose staged secret bytes | `B5FF8D9570812373D5BB9004C7C9671AA50075E94478C84C8DAD37ED05AA00C8` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Private publication and recovery / Private successor credential publication with actual Windows file custody / refuses foreign parent namespace authority before a broad predecessor can arrive at real Replace | `4447703E4BCE95201D30CE944B9F98E80B9434334EBB482993FB89E53E378AD6` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Directory namespace admission / Trusted namespace before private credential staging / refuses applicable foreign <Right> before private stage creation | `2AA8220E9590DC71F004C4AEF6CCA7C3DFA0D452C40F07AA7512FE2EDC841A7E` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Directory namespace admission / Trusted namespace before private credential staging / refuses ancestor DeleteChild even when the immediate parent is protected | `FB56548D9188B1704246C8CB16D30B634D3F7A2EE1D84C5ABE4DF9419A68B082` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Directory namespace admission / Trusted namespace before private credential staging / refuses an actually foreign-owned directory despite private mutation grants | `115E87408B02E340D29ACB9752B0EB72C8DB27D05145B3208D35EDA36152BE87` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Directory namespace admission / Trusted namespace before private credential staging / admits foreign read-only directory grants and still publishes private bytes | `02963120DCA7D6D9D29038975BDB7CA3FD47E1A8F83022F5D6E0A793D064CDE3` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Directory namespace admission / Trusted namespace before private credential staging / ignores InheritOnly grants that do not apply to an admitted existing directory | `1C79DE8B171A290A1DB8030D57BD7B3AC6D5443DA489F5DA31DC5759393CA628` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Directory namespace admission / Trusted namespace before private credential staging / conservatively refuses a foreign mutation allow even when a deny also exists | `BDE7F19DDD6FDE0AC3F14DB7EE80C8CA48AC1951674CA2446F1660E761066D5E` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Directory namespace admission / Trusted namespace before private credential staging / admits ancestor-only AddSubdirectory grants without granting direct parent mutation | `C1964ACF7BAA24AB81764F14D8EDBE1FD2B622B982B3A5621419FB82EB5E14A4` | KEEP | /root/credential_custody_audit | independent Codex static source/test review | 2026-10-22 |

### Atomic-cache, finite-deadline and profile-fallback revision review

These revisions address the normal PR review's cache-pair publication failure,
archive deadline mismatch and uncaught profile cache-write failure. The original 153 `It` bodies are byte-identical
after LF normalization to signed commit `75b815231d78199d5516b1bbad12131de1c5dfe1`.
Thirteen new groups add 36 cases, and two `BeforeAll` fixture extensions supply
isolated legacy-pair helpers and captured real process-runner entrypoints.

The cache regressions first reproduced unusable fallback after a separate
manifest failure. Further failing controls exposed a manifest-observation race
and alternate-stream metadata being retired; the final implementation preserves
both changed inputs. Deadline controls first reproduced rejection of 121/600
seconds, then exposed the private wait's remaining 120000-ms cap. The qualified
runner now uses a finite 600000-ms maximum while the cmdkey contract stays intact.

Current LF SHA-256 source/test pins for the thirteen revision groups:

- Tools/SystemScripts/Machine/SecretBackendUtilities.ps1: `269C70104FB24B502EA7233347D5B1FFEB5F0A5C1B16C0661DDDCB904EF6E4E5`.
- Tools/SystemScripts/Machine/SecretsTier.psm1: `C4162491245879653F27AAC76B1395FC02E1671A9C736A2A25CFAF616F3949D5`.
- Tests/Unit/SecretBackendCustody.Tests.ps1: `7923321917C39649AA0FFB6FF020AA762F65B0D5D5B6A320D30F714D01FE15AB`.

Execution evidence: author `fce7e9`, independent root `dd7d38`; both pass
303 Windows cases with no test/block/container failures or skips. Source/test
equivalence and new-selector inventory were checked against the signed baseline
in `eb32e1` and independently in `78ae78`. The latter review KEEP covers the
revised production, the first eleven groups and both fixture extensions; the
reviewer authored none of that code. Those 303-case results used signed
`238524ca83f3af8463e5b3def0a6e5923a74a877`, with module pin
`B6A7014269E536233BB8F1F43413282D29C2C1E7B7A253FF22328C6BC87DD3E1`
and test pin `012D29D4C1518BBB8CDB6C875DA96E0DB55F33760976F73DD1E1CEDF9F7C5635`.
The setup LF SHA-256 pins are
`25336FC37001D332EF437DB6A6C2270ABAE6BC2EB3030B3B3B330A34404961CA`
and `0999371F0E40E51FC646426754D8D968A061411E9F769F1CD538F6172A93BAF3`.
Historical approvals above retain their original source pins.

The profile-publication regressions then failed all four cases against that
signed revision (`7eeda3`). The Save-only catch passed the four focused controls
(`689cc9`) and the full 307-case gate (`def3f0`). The independent 307-case
Windows replay (`2888fa`) binds the current pins. Independent static review
`62679c` KEEP covers the narrow production catch and two new groups, and rebinds
the earlier eleven groups to the current pins: all 164 prior selector bodies,
their cache/cleanup/deadline production and all fixture setup bodies are
unchanged. The two added synthetic-token literals carry narrowly scoped scanner
pragmas and were reviewed in their isolated fixture context. All thirteen rows
below are reviewed by `/root/codex_deployment_audit`, independent Codex static
source/test review, 2026-10-22 UTC. This does not recertify all 153 historical
rows at changed production pins; full-suite execution is separate evidence.

Status controls qualify structural/digest checks until an actual DPAPI read.
Fast-child controls qualify deadline admission and execution at 121/600 seconds,
rather than waiting 600 seconds. The five cleanup-mutation variants qualify
retention; only the alternate-stream variant verifies changed stream contents,
so they do not claim complete per-variant postmutation byte/ACL readback.
No installed credential code, real provider, cache or vault was changed.

| Selector/group | Selector SHA256 | Disposition | Reviewer | Review type | Date (UTC) |
| --- | --- | --- | --- | --- | --- |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / keeps a coherent usable cache when a separate manifest write is unavailable | `4D7A767E13CCABC5C5673E0C02C47526E14206490D350141472192204F74F95E` | KEEP | /root/codex_deployment_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / preserves the previous readable pair exactly when canonical cache publication fails | `ABF4DD3ED57B2AE66EFEB3B2B6454D507BED8F54B756C690624D266405401F6B` | KEEP | /root/codex_deployment_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / publishes one self-contained cache file and retires only qualified old metadata | `248343C4AFB0EB87615028E59F673A40D8ABC9C7D8D678BD9D77FB0D2979E404` | KEEP | /root/codex_deployment_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / reports v2 cache metadata without decrypting or exposing values | `2333D4F02B10B0F2ABB3CA4D8D025619EF157A2D3546C590B04C28A490B2A86B` | KEEP | /root/codex_deployment_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / refuses tampered v2 cache <Field> without legacy rollback or hydration | `BC833E2215AE2CA0455518D3B2804D2F2FF8AEB315C62EF5B3F69E4378A0105C` | KEEP | /root/codex_deployment_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / refuses authenticated cache payload disagreement in <Field> | `F26843EF0A9DE956D4945927A08CD7242D797FEBD8EA4993D2116F4BF1201140` | KEEP | /root/codex_deployment_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / preserves old metadata changed immediately after legacy pair validation | `5A15814F63C4F810F2922CE77D582F344AC2D65F5B68CC326C9BC7AACF789074` | KEEP | /root/codex_deployment_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / preserves old cache metadata when cleanup sees <Mode> | `1BCFDD1A338DF51D5A2CD2F54785C7A8E34B3F93648AA9848F0DF8DACF91739E` | KEEP | /root/codex_deployment_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Protected runtime consumers and authenticated legacy reads / Private credential consumer runtime contract / Actual protected default cache and read-only authenticated legacy integration / keeps the old cache usable after actual partial canonical displacement | `D864CB215B9A338577123E23E92CB31C000F6065A9A1E9F7DEC5DCCF72B8B3D4` | KEEP | /root/codex_deployment_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic backend discovery and credential normalization / accepts archive-compatible finite deadline <Timeout> through the real bounded runner | `204F923AA640D7FC5631249A227C602B5288299351817142B76B3969DF2B22DA` | KEEP | /root/codex_deployment_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic resource coordination and mapping boundaries / Synthetic backend discovery and credential normalization / refuses unsupported archive deadline <Timeout> before launch | `D47B1DC6A0802E56DEF382B45A670565642675626E334751FE7A1A4F652C8F86` | KEEP | /root/codex_deployment_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / continues through validated <Scenario> cache after cloud cache publication fails | `5A41DB5983F2CB4F25672D1A2B69D84EC7DFE50AD008E283E45CE75B342F64C5` | KEEP | /root/codex_deployment_audit | independent Codex static source/test review | 2026-10-22 |
| Isolated Windows credential custody fixtures / Synthetic cloud, backup and environment orchestration / propagates selected cache admission failure after cloud cache publication fails | `C8013D686DBACE68102501511CAA8ED7C2C4ED7C2EFD17070DAB5D8163277B6F` | KEEP | /root/codex_deployment_audit | independent Codex static source/test review | 2026-10-22 |

## Safety Rules

The [Windows USB4 startup policy](../../docs/usb4/windows-boot.md) is an observer.
It preserves native V&V failures and refuses discovery-driven route mutation.
The [entrypoint](Networking/Configure-ThunderboltSwitchRoutes.ps1) distinguishes
undocked/degraded, collection failure and pending acceptance; dry-run writes no
receipts. It does not grant authenticated data-edge readiness.

- Treat migrated scripts as production workstation automation, not scratch
  snippets.
- Prefer `-DryRun` and evidence capture before enabling or registering any
  startup/logon behavior.
- Do not force-add logs, cache files, secret material, or backup files that are
  ignored by git.
- Before enabling Task Scheduler usage for a migrated script, check whether it
  needs `-h`/`--help`, `-DryRun`, idempotency, structured logs, and nonzero exit
  codes for failure.
