# PC-AI release signing and provenance

Use the accepted Git revision and the artifact hashes to identify a PC-AI
release. Git signatures identify the source author or tag signer.
Authenticode identifies the publisher of an individual Windows file. Artifact
attestations tie the final archive to the repository and workflow that built it.
Windows privileges remain governed by account, ACL, and task policy.

## Source and version

Use an annotated, signed SemVer release tag with the prefix v. Verify the tag
signature against the authorized release signer and resolve the tag to an
accepted main commit before pushing it. Keep tag updates and deletion restricted.
Both the [native release workflow](.github/workflows/release-cuda.yml) and the
[PowerShell release workflow](.github/workflows/release.yml) react to version
tags. Inventory both sets of assets before a production tag is pushed. Keep
manual native workflow runs for artifact testing.

    git verify-tag v1.2.3
    git rev-parse "v1.2.3^{commit}"

For organization-backed signing, the DTMVentures Artifact Signing account needs
a verified identity and a certificate profile. Give the release workflow
the profile-level Artifact Signing Certificate Profile Signer role through
short-lived federated identity. Scope that identity to the protected release
workflow and environment. Keep signing privileges and key material away from PR
jobs and ordinary workstation accounts.
[Artifact Signing integrations](https://learn.microsoft.com/en-us/azure/artifact-signing/how-to-signing-integrations) ·
[profile roles](https://learn.microsoft.com/en-us/azure/artifact-signing/tutorial-assign-roles).

## Windows payloads and cross-platform archives

Sign the final Windows EXE, DLL and PowerShell files after building, staging and
choosing their final encodings. Apply a trusted timestamp and verify each file
with a fresh Windows process before packaging. Record the observed publisher,
certificate chain, timestamp and SHA-256. Admit a verified publisher through
organization policy and manage certificate rotation there. Changing a signed
script's encoding or line endings requires a new signature. System task actions
still require protected directories, interpreter and configuration.

    signtool verify /pa /all /v .\pcai-tb.exe
    Get-AuthenticodeSignature -LiteralPath .\Invoke-PcaiLinkHealth.ps1

Use the [module package builder](Tools/New-PcaiPowerShellRelease.ps1) to produce
the release payload. Verify the final archive after compression. Publish SHA-256
checksums and build attestations for Windows and Linux archives with their source
commit, architecture, and workflow reference. In a future single-publisher
pipeline, assemble all required assets on a draft before publishing an immutable
release. ZIP verification remains an admission check at deployment.
[PowerShell signature verification](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.security/get-authenticodesignature) ·
[GitHub artifact attestations](https://docs.github.com/en/actions/concepts/security/artifact-attestations) ·
[immutable releases](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases).

DTMVentures profile access and installer-side Authenticode enforcement require
a verified identity, a signing credential and a signed acceptance artifact.
A build from a signed Git commit can be tracked with hashes during a
controlled rollout. Its Windows files still need the separate publisher
signature for Authenticode admission.
