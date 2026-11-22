# Versioning policy

PC-AI has one product version for the whole repository. It is a [SemVer 2.0](https://semver.org/spec/v2.0.0.html)
version, it comes from a signed, annotated git tag on `main`, and the Rust workspace carries the same number. This
document says what the number means, who creates a tag and when, how untagged builds are named, which components follow
the product version and which keep their own, and how builds and installs read it.

**Baseline version:** `0.2.0`. This is the first release, approved by the owner. When this policy was written
(2026-11-22) the repository had no tags, so no earlier build has a release version. The `versioning` check in
[validate-doc-accuracy.ps1](../Tools/validate-doc-accuracy.ps1) reads this line.

## Why this exists

Until this policy, the repository had no tag and no release. [Get-BuildVersion.ps1](../Tools/Get-BuildVersion.ps1)
called every build `0.1.0.N+hash`, and its header described a different string, `{semver}-{commits}+{hash}.{timestamp}`,
from the one its code produced. Four of the eight workspace crates hard-coded `0.1.0` instead of inheriting the
workspace version, so a version bump would have missed the main inference binary. Nothing failed while these
disagreed. The tag `v0.2.0`, created on the merge commit of the pull request that adds this file, starts the release
history. The check described under [Enforcement](#enforcement) keeps the documentation and the code in agreement from
then on.

## Product version

- **Format:** `MAJOR.MINOR.PATCH`, or `MAJOR.MINOR.PATCH-rc.N` for a release candidate.
- **Tags:** `vMAJOR.MINOR.PATCH` (for example `v0.2.0`) and `vMAJOR.MINOR.PATCH-rc.N` (for example `v0.3.0-rc.1`). Each
  tag is annotated and signed (`git tag -s`), never lightweight. Its message is `PC-AI vX.Y.Z`.
- **Branch:** tags go only on commits reachable from `origin/main`, and only on the merge commit of a release PR. A tag
  is never created on a feature branch or a commit that has not been pushed.
- **Immutability:** a published tag is never moved or deleted. A broken release gets a new PATCH version.

### What a change means before 1.0

The public surface is the `PC-AI.ps1` CLI, the exported PowerShell commands, the native FFI exports, the HTTP routes
and the `Config/` schemas. While MAJOR is 0, it is not stable:

| Bump | When |
|---|---|
| MINOR (`0.2.0` to `0.3.0`) | New features, and any breaking change to the public surface |
| PATCH (`0.2.0` to `0.2.1`) | Bug fixes, documentation and dependency updates that change no public behaviour |
| MAJOR (`1.0.0`) | Declaring the public surface stable. This is an owner decision, not a routine bump |

After 1.0, normal SemVer applies: a breaking change bumps MAJOR.

## Releasing: who tags and when

The [merge autopilot](https://github.com/David-Martel/vigil-utils/blob/main/tools/merge-autopilot/README.md) merges
pull requests. **It never creates tags.** A tag is created by the owner, or by an agent the owner authorized for that
specific release, in this order:

1. **Release PR.** One pull request sets `[workspace.package] version` in
   [Native/pcai_core/Cargo.toml](../Native/pcai_core/Cargo.toml) to the version it will tag. It updates `Cargo.lock`
   with `cargo update -w`, which changes only the workspace members' own versions, and regenerates
   [CHANGELOG.md](../CHANGELOG.md) (see [Changelog](#changelog)). For `v0.2.0`, that pull request is the one that
   added this file.
2. **Merge.** The PR merges through the normal route.
3. **Green on the merge commit.** Every required check passes on the merge commit itself, not just on the PR head.
4. **Tag.** Create the tag on that commit, verify it, then push it:

   ```bash
   git fetch origin
   git tag -s v0.2.0 -m "PC-AI v0.2.0" <merge-commit-sha>
   git tag -v v0.2.0
   git push origin v0.2.0
   ```

Pushing a `v*` tag is the trigger for [release.yml](../.github/workflows/release.yml) (the PowerShell module package)
and [release-cuda.yml](../.github/workflows/release-cuda.yml) (the native ZIPs). Both workflows are disabled in the
repository settings today (see the workflow table in [CLAUDE.md](../CLAUDE.md)), so a tag records the release but
publishes nothing. Re-enabling them is a separate decision.

## Untagged builds

Every build that is not exactly on a release tag is named after the last tag:

**Version Format:** `{semver}[.{commits}]+{hash}[.dirty]`

- `{semver}` is the latest `v*` tag without its `v`. With no tag, it is the `[workspace.package]` version.
- `{commits}` is the number of commits since that tag (since the root commit when there is no tag). It is omitted on
  the tagged commit.
- `{hash}` is the 7-character commit hash, or `unknown` when git is unavailable.
- `.dirty` marks uncommitted changes.

Examples:

- Example: `0.2.0+abc1234` is the tagged `v0.2.0` commit.
- Example: `0.2.0.15+abc1234` is 15 commits after `v0.2.0`.
- Example: `0.2.0.3+abc1234.dirty` is 3 commits after `v0.2.0`, with uncommitted changes.
- Example: `0.3.0-rc.1.4+abc1234` is 4 commits after the release candidate `v0.3.0-rc.1`.

**Next-development convention.** There is no `-dev` suffix. A `-dev` pre-release such as `0.3.0-dev` would sort
*before* `0.3.0` under SemVer, and before the release it was built after. Development builds are identified by the
`.{commits}` field instead: `0.2.0.15` is the fifteenth commit after `0.2.0`, and it sorts after `0.2.0` as a
four-part file version. The workspace version stays at the last release until the next release PR raises it. That PR
may raise it early, but only to the version it is preparing, so the check accepts either the latest tag or the next
PATCH, MINOR or MAJOR after it.

## Components

| Component | Version | Source |
|---|---|---|
| Product (tags, release packages) | Product version | `vX.Y.Z` tags |
| Rust workspace `Native/pcai_core` (all eight members) | Product version | `[workspace.package] version`. Every member declares `version.workspace = true` |
| Vendored third-party crates | Upstream version, excluded | [ggml-sys](../Native/pcai_core/third_party/ggml-sys/Cargo.toml) keeps upstream's `0.2.0-dev`; it is a `[patch]` for the llama-rs git source and has to match that source. The same applies to `vendor/` and `Deploy/vendor/` |
| FunctionGemma workspace [Deploy/Cargo.toml](../Deploy/Cargo.toml) | `0.1.0`, independent for now | A separate Cargo workspace with no `[workspace.package]`. Moving it to the product version is a follow-up |
| PowerShell modules (`Modules/*.psd1`) | Independent `ModuleVersion` | Each module keeps its own SemVer and bumps it when its exported surface changes. For example, [PC-AI.Drivers](../Modules/PC-AI.Drivers/PC-AI.Drivers.psd1) is at 1.1.0 while most modules are at 1.0.0. `PC-AI.Common` has no manifest |
| PowerShell release package | Product version | [New-PcaiPowerShellRelease.ps1](../Tools/New-PcaiPowerShellRelease.ps1) takes the generated root module's version from `Get-BuildVersion` `SemVer` and writes the full `Get-BuildVersion` object to `RELEASE_INFO.json` as `BuildVersion` |
| C# assemblies published by `Build.ps1` | Product version, stamped at build time | `Invoke-DotnetComponentBuild` in [Build.ps1](../Build.ps1) passes `-p:Version` (SemVer), `-p:AssemblyVersion` (`X.Y.Z.0`), `-p:FileVersion` (`X.Y.Z.{commits}`) and `-p:InformationalVersion` from `Get-BuildVersion`. Today it publishes PcaiNative and PcaiServiceHost |
| C# `<Version>` in `*.csproj` | Not authoritative | The literal (for example `0.1.0` in [PcaiNative.csproj](../Native/PcaiNative/PcaiNative.csproj), `1.0.0` in [DtmTaskRun.csproj](../Native/DtmTaskRun/DtmTaskRun.csproj)) applies only to a bare `dotnet build`, and is not bumped per release. PcaiChatTui and DtmTaskRun are built outside `Invoke-DotnetComponentBuild` and are not stamped yet; that is a follow-up |
| `AI-Media/pyproject.toml` | Independent | The Python media agent is not versioned with the product |

## How builds, installs and tests read the version

- **[Get-BuildVersion.ps1](../Tools/Get-BuildVersion.ps1)** is the one place the version is derived. It uses CargoTools'
  `Get-BuildVersionInfo` when that module is installed, and the script's own git fallback otherwise. Both paths return
  the same object (`Version`, `SemVer`, `ReleaseTag`, `CommitsSinceTag`, `AssemblyVersion`, `FileVersion`,
  `InformationalVersion` and so on), in `Object`, `Json`, `Env`, `Cargo` or `CMake` format. With no tag it reads the
  default from `[workspace.package] version`. `-SetEnv` exports `PCAI_VERSION`, `PCAI_SEMVER`, `PCAI_GIT_HASH`,
  `PCAI_RELEASE_TAG` and related variables.
- **[Build.ps1](../Build.ps1)** runs it once with `-SetEnv` (`Initialize-BuildVersion`). Cargo builds see the `PCAI_*`
  variables, .NET publishes are stamped as described above, and the build manifest
  `.pcai/build/artifacts/manifest.json` records `pcaiVersion`, `semver`, `releaseTag`, `gitCommit` and `buildType`
  next to the artifact hashes. Native bundles record the same in `bundle.buildinfo.json`.
- **[pcai_inference/build.rs](../Native/pcai_core/pcai_inference/build.rs)** embeds `PCAI_VERSION` when `Build.ps1` sets
  it (or `PCAI_BUILD_VERSION`). Otherwise it derives the version from `git describe`, falling back to the crate
  (product) version. Cargo caches the script, so it reruns only when one of its inputs changes:
  - every environment variable it embeds (`PCAI_VERSION`, `PCAI_BUILD_VERSION`, `PCAI_SEMVER`, `PCAI_GIT_HASH`,
    `PCAI_GIT_HASH_SHORT`, `PCAI_GIT_BRANCH`, `PCAI_BUILD_TIMESTAMP`, `PCAI_BUILD_TYPE`, `CUDA_PATH`, `CUDA_HOME`),
    each read through `tracked_env`, which declares it to Cargo whether or not it is set;
  - `PATH`, but only while CUDA is configured, because the script then runs the `nvcc` found on `PATH`;
  - HEAD, the HEAD reflog, the branch ref, `packed-refs` and `refs/tags`, resolved with
    `git rev-parse --git-path` so linked worktrees resolve too.

  [test_inference_build_metadata.py](../Tests/Portable/test_inference_build_metadata.py) changes each input alone
  in a warm build and checks that the embedded constant changed, and that an unchanged rebuild stays cached.
  `pcai-llamacpp --version-json` and
  `pcai-mistralrs --version-json` print the embedded build info, and the FFI export `pcai_version` returns the crate
  version.
- **[pcai_core_lib/build.rs](../Native/pcai_core/pcai_core_lib/build.rs)** embeds `PCAI_BUILD_VERSION`, falling back to
  `{crate version}+unknown`. `pcai_core_version` returns it.
- **`pcai-tb --version`** ([main.rs](../Native/pcai_core/pcai_tb_cli/src/main.rs)) prints the crate version, which is
  now the product version, with no commit hash. Embedding the build version there is a follow-up.
- **Deployment receipts:** recording the installed product version in install and deployment receipts, so a test can
  assert what is deployed, is a follow-up pull request. Until then, read `manifest.json` or `RELEASE_INFO.json`.

## Changelog

[CHANGELOG.md](../CHANGELOG.md) is generated by [git-cliff](https://git-cliff.org/) from [cliff.toml](../cliff.toml).
The conventional commit type sets the section. A subject that is not a conventional commit, such as a plain squash
title, goes under "Other" rather than being dropped. Merge commits and `context:` session checkpoints are skipped, only
the first line of each message is printed, identical subjects appear once, and `commit_preprocessors` replace fleet
host names with `[host]`. Regenerate it in each release PR, before the tag exists, so the new commits appear as
`[Unreleased]`:

```bash
git-cliff --config cliff.toml --output CHANGELOG.md
```

Once the tag exists, the same command files those commits under the tag's version and date. Review the output before
committing, because the repository is public. Search it for IP addresses, tokens, e-mail addresses and host names. If
a commit subject leaks one, add a `commit_preprocessors` pattern to `cliff.toml` and regenerate; do not edit
`CHANGELOG.md` by hand, because the next regeneration would restore the leak.
[changelog.yml](../.github/workflows/changelog.yml), which regenerated the file on every push to `main`, is disabled.
The release PR is where the changelog is updated.

## Enforcement

The `versioning` check in [validate-doc-accuracy.ps1](../Tools/validate-doc-accuracy.ps1) fails when:

- the workspace version is neither the latest `v*` tag nor the next PATCH, MINOR or MAJOR version after it;
- with a complete history that has no tag, the workspace version is not the baseline above;
- the workspace version is below the baseline, or is not a plain `X.Y.Z`;
- the header of `Get-BuildVersion.ps1`, the CLAUDE.md "Version Information" section or this file documents a format
  other than `{semver}[.{commits}]+{hash}[.dirty]`, or gives an example that does not fit it.

Tags are read with local git only. A shallow clone (CI's default checkout) does not carry tags, so there the check
verifies only that the workspace version is not below the baseline, and its summary says so. The tag rule is enforced
wherever the history is complete, including the checkout used to create a tag.
[DocAccuracy.Tests.ps1](../Tests/Unit/DocAccuracy.Tests.ps1) holds the positive controls (each rule fails on a fixture
that breaks it). It also matches the live output of `Get-BuildVersion.ps1`, through both the CargoTools path and the
fallback path, against the pattern the check uses.

## Follow-ups

- Move the `Deploy/` FunctionGemma workspace to the product version through a `[workspace.package]`.
- Stamp PcaiChatTui and DtmTaskRun from `Get-BuildVersion`, as `Invoke-DotnetComponentBuild` already does.
- Embed the build version (with commit hash) in `pcai-tb --version`.
- Replace the remaining last-resort `0.1.0+unknown` literals in `Build.ps1` (`Initialize-BuildVersion` and the
  manifest writer, used only when `Get-BuildVersion.ps1` is missing) and in `pcai_inference/Invoke-PcaiBuild.ps1`.
- Record the product version in deployment receipts.
- CargoTools (external): `Get-BuildVersionInfo` invents a `ReleaseTag` of `v{default}` when no tag exists, and when git
  fails it returns a version with no commit hash. Both should be fixed in CargoTools.
