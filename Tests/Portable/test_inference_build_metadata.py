"""Native build-script inputs are tracked, so warm rebuilds never embed stale metadata.

Why this exists: PR #403 fixed the pcai-inference build script's Git watch paths.
Before that fix Cargo reran the script on every build (a missing
``rerun-if-changed`` path counts as changed). That hid the fact that only some of
the environment variables it embeds had ``rerun-if-env-changed`` triggers. Once
the script became cacheable, a build that changed only ``PCAI_BUILD_VERSION`` kept
the previous version embedded (reproduced by the merge autopilot's review of
00f2c317). The rule these tests enforce: every external input a build script
reads has a rerun trigger.

- The static test needs no toolchain. It reads every
  ``Native/pcai_core/*/build.rs`` and fails when an environment variable the
  script reads is not declared through ``cargo:rerun-if-env-changed`` (directly or
  through ``tracked_env``). Variables Cargo itself sets are exempt: a change to
  them already produces a new build-script run. The scan sees declarations, not
  whether a declaration is reached at run time; the warm-rebuild test covers that
  (the pre-fix script declared CUDA_PATH only when CUDA was already configured).
- The warm-rebuild test needs cargo and network access for a cold dependency
  fetch. In one isolated target directory it checks pcai-inference
  (``--no-default-features --features ffi``), then changes exactly one input per
  step and asserts that the generated ``build_info.rs`` constant changed. Its
  negative control asserts that an unchanged rebuild neither reruns the build
  script nor recompiles the crate, so the cache still works.
- ``PATH`` is an input of every build, because the script looks ``git`` up on it.
  Codex found (PR #403, 15c00d73) that the declaration sat inside ``if has_cuda``:
  a first CPU build without git on PATH embeds ``unknown``, records no Git watch
  paths, and a later warm build with git available never reran the script. Three
  tests cover it at three costs: a static check that the declaration is
  unconditional, a fast check of the build script's ``output`` file for a CPU-only
  feature set, and the exact warm-rebuild sequence (build without git, restore
  PATH, rebuild warm, expect the real commit, then withdraw git again and expect
  ``unknown`` rather than the stale commit).

Breadcrumbs: Native/pcai_core/pcai_inference/build.rs (``tracked_env``);
docs/VERSIONING.md, "How builds, installs and tests read the version". CI runs
this file in linux-ci.yml's portable-contracts job, which discovers
``Tests/Portable/test_*.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO / "Native" / "pcai_core"
INFERENCE_BUILD_RS = WORKSPACE / "pcai_inference" / "build.rs"

# Variables Cargo sets for a build script. A change to any of them (target,
# profile, features, package metadata) already gives the script a new run, so
# they need no trigger of their own. CARGO_* names are exempt by prefix.
CARGO_PROVIDED = {
    "OUT_DIR",
    "TARGET",
    "HOST",
    "PROFILE",
    "OPT_LEVEL",
    "DEBUG",
    "NUM_JOBS",
    "RUSTC",
    "RUSTDOC",
}

# Every metadata input of pcai_inference/build.rs: the variable, the value one
# step sets (None removes it), and the generated line that must then appear.
# Order matters: each step changes exactly one variable from the previous step.
INFERENCE_STEPS = [
    (
        "PCAI_BUILD_VERSION",
        "9.8.7.1+aaaaaaa",
        'pub const VERSION: &str = "9.8.7.1+aaaaaaa";',
    ),
    ("PCAI_VERSION", "9.8.7.2+bbbbbbb", 'pub const VERSION: &str = "9.8.7.2+bbbbbbb";'),
    ("PCAI_SEMVER", "9.8.7", 'pub const SEMVER: &str = "9.8.7";'),
    ("PCAI_GIT_HASH", "c" * 40, f'pub const GIT_HASH: &str = "{"c" * 40}";'),
    ("PCAI_GIT_HASH_SHORT", "ddddddd", 'pub const GIT_HASH_SHORT: &str = "ddddddd";'),
    (
        "PCAI_GIT_BRANCH",
        "metadata-probe",
        'pub const GIT_BRANCH: &str = "metadata-probe";',
    ),
    (
        "PCAI_BUILD_TIMESTAMP",
        "2001-02-03T04:05:06Z",
        'pub const BUILD_TIMESTAMP: &str = "2001-02-03T04:05:06Z";',
    ),
    ("PCAI_BUILD_TYPE", "probe", 'pub const BUILD_TYPE: &str = "probe";'),
]

# Host CUDA settings a build script must not consult: whether the artifact has CUDA
# is a property of the compiled features (codex, PR #403), not of the build host.
HOST_CUDA_PROBES = ('"CUDA_PATH"', '"CUDA_HOME"', '"nvidia-smi"')

# pcai-inference declares crate-type = ["cdylib", "rlib"]; its kind is never "lib".
LIBRARY_KINDS = {"lib", "rlib", "cdylib", "dylib", "staticlib"}

# build.rs writes the identity as MARKER + JSON + NUL; build_info_json() returns JSON.
IDENTITY_MARKER = b"PCAI_BUILD_IDENTITY_V1\0"

# A build script that runs git must declare PATH as a top-level statement of
# ``main`` (rustfmt indents that by exactly four spaces). A declaration nested in
# a conditional is the bug: it covers only the builds that take that branch.
RUNS_GIT = re.compile(r'Command::new\(\s*"git"\s*\)')
PATH_DECLARED_UNCONDITIONALLY = re.compile(
    r'^    println!\("cargo:rerun-if-env-changed=PATH"\);$', re.MULTILINE
)

ENV_READ = re.compile(r'(?:env::var(?:_os)?|option_env!|env!)\(\s*"([A-Z0-9_]+)"')
RERUN_ENV = re.compile(r"rerun-if-env-changed=([A-Z0-9_]+)")
TRACKED_CALL = re.compile(r'tracked_env\(\s*"([A-Z0-9_]+)"')


def build_scripts() -> list[Path]:
    return sorted(WORKSPACE.glob("*/build.rs"))


def untracked_inputs(source: str) -> set[str]:
    """Environment variables the script reads that no rerun trigger covers."""
    read = {
        name
        for name in ENV_READ.findall(source)
        if name not in CARGO_PROVIDED and not name.startswith("CARGO_")
    }
    tracked = set(RERUN_ENV.findall(source)) | set(TRACKED_CALL.findall(source))
    return read - tracked


@dataclass(frozen=True)
class CheckResult:
    """One cargo check: build_info.rs, mtime, crate freshness, script output."""

    text: str
    mtime: int
    lib_fresh: bool
    output: str = ""
    # The JSON frame `--version-json` prints (build_identity.bin, see build.rs).
    identity: dict[str, object] = field(default_factory=dict)


def constant(text: str, name: str) -> str:
    """The value of ``pub const NAME: &str = "...";`` in build_info.rs."""
    match = re.search(rf'pub const {name}: &str = "([^"]*)";', text)
    if match is None:
        raise AssertionError(f"build_info.rs has no string constant {name}")
    return match.group(1)


class BuildScriptInputsAreDeclared(unittest.TestCase):
    def test_every_environment_input_has_a_rerun_trigger(self):
        """Protects: each native build script declares every variable it reads.

        Detects: an ``env::var`` read without a ``cargo:rerun-if-env-changed``
        (directly or through ``tracked_env``), which makes a warm build keep stale
        generated output. Needs: only the sources.
        """
        scripts = build_scripts()
        self.assertIn(
            INFERENCE_BUILD_RS,
            scripts,
            "the inference build script must be in the scanned set",
        )
        for script in scripts:
            with self.subTest(script=str(script.relative_to(REPO))):
                missing = untracked_inputs(script.read_text(encoding="utf-8"))
                self.assertFalse(
                    missing, f"read without a rerun trigger: {sorted(missing)}"
                )

    def test_the_inference_script_does_not_probe_host_cuda(self):
        """Protects: HAS_CUDA describes the compiled features, not the build host.

        Detects: a build script that reads CUDA_PATH, CUDA_HOME or nvidia-smi again;
        that made no-CUDA artifacts report ``has_cuda: true`` (codex, PR #403).
        """
        source = INFERENCE_BUILD_RS.read_text(encoding="utf-8")
        found = [probe for probe in HOST_CUDA_PROBES if probe in source]
        self.assertFalse(found, f"build.rs consults host CUDA settings: {found}")
        self.assertIn("CARGO_FEATURE_CUDA_LLAMACPP", source)
        self.assertIn("CARGO_FEATURE_CUDA_MISTRALRS", source)

    def test_a_build_script_that_runs_git_tracks_path_unconditionally(self):
        """Protects: PATH is a rerun trigger on every build of a script that runs git.

        Detects: the PR #403 regression where ``rerun-if-env-changed=PATH`` was emitted
        only inside ``if has_cuda``, so CPU builds kept metadata computed under a
        different PATH. Needs: only the sources. This sees the declaration's position,
        not whether it is reached; the build-script output test covers that.
        """
        scripts = [
            script
            for script in build_scripts()
            if RUNS_GIT.search(script.read_text(encoding="utf-8"))
        ]
        self.assertIn(
            INFERENCE_BUILD_RS,
            scripts,
            "the inference script runs git and must be scanned",
        )
        for script in scripts:
            with self.subTest(script=str(script.relative_to(REPO))):
                source = script.read_text(encoding="utf-8")
                self.assertIsNotNone(
                    PATH_DECLARED_UNCONDITIONALLY.search(source),
                    "runs git but does not declare PATH at the top level of main",
                )

    def test_the_path_scan_rejects_a_conditional_declaration(self):
        """Positive control: the scan flags the PR #403 shape and accepts the fix."""
        conditional = (
            "fn main() {\n    if has_cuda {\n"
            '        println!("cargo:rerun-if-env-changed=PATH");\n    }\n}\n'
        )
        unconditional = (
            'fn main() {\n    println!("cargo:rerun-if-env-changed=PATH");\n'
            "    if has_cuda {\n    }\n}\n"
        )
        self.assertIsNone(PATH_DECLARED_UNCONDITIONALLY.search(conditional))
        self.assertIsNotNone(PATH_DECLARED_UNCONDITIONALLY.search(unconditional))
        self.assertIsNotNone(RUNS_GIT.search('Command::new("git")'))
        self.assertIsNone(RUNS_GIT.search('Command::new("nvcc")'))

    def test_the_scan_finds_an_untracked_read(self):
        """Positive control: the scan flags an untracked read, accepts a tracked one."""
        self.assertEqual(
            untracked_inputs('let v = env::var("PCAI_SEMVER");'), {"PCAI_SEMVER"}
        )
        declared = (
            'println!("cargo:rerun-if-env-changed=PCAI_SEMVER"); '
            'let v = env::var("PCAI_SEMVER");'
        )
        self.assertEqual(untracked_inputs(declared), set())
        self.assertEqual(untracked_inputs('let v = tracked_env("PCAI_SEMVER");'), set())
        cargo_set = (
            'let o = env::var_os("OUT_DIR"); let t = env::var("CARGO_FEATURE_FFI");'
        )
        self.assertEqual(untracked_inputs(cargo_set), set())


class InferenceWarmRebuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cargo = shutil.which("cargo")
        if not cls.cargo:
            raise RuntimeError(
                "cargo is required; no rebuild behaviour is inferred without it"
            )
        cls.scratch_dir = tempfile.TemporaryDirectory(prefix="pcai-build-metadata-")
        cls.addClassCleanup(cls.scratch_dir.cleanup)
        cls.scratch = Path(cls.scratch_dir.name)
        # PCAI_BUILD_METADATA_TARGET_DIR keeps one target directory across runs
        # on a disk that cannot hold several; every test here is independent of
        # what an earlier run left in it.
        persistent = os.environ.get("PCAI_BUILD_METADATA_TARGET_DIR")
        cls.target = Path(persistent) if persistent else cls.scratch / "target"
        # Start from no metadata input at all, and never share a target directory.
        cls.base_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("PCAI_")
            and key not in {"CUDA_PATH", "CUDA_HOME", "CARGO_TARGET_DIR"}
        }
        cls.nonce = 0

    def fresh_nonce(self, label: str) -> str:
        """A PCAI_BUILD_TYPE no earlier run used, so the script reruns for it."""
        type(self).nonce += 1
        return f"{label}-{os.getpid()}-{type(self).nonce}"

    def path_without_git(self) -> str:
        """PATH on which ``git`` cannot run, leaving cargo, rustc and the linker.

        Windows: drop every PATH entry that holds a git executable (git lives in
        its own directories there). Elsewhere git shares /usr/bin with the C
        toolchain, so put a ``git`` that always fails first instead.
        """
        original = self.base_env["PATH"]
        self.assertIsNotNone(
            shutil.which("git", path=original),
            "git must be installed: without it this scenario proves nothing",
        )
        if os.name == "nt":
            kept = [
                entry
                for entry in original.split(os.pathsep)
                if entry and shutil.which("git", path=entry) is None
            ]
            stripped = os.pathsep.join(kept)
            self.assertIsNone(shutil.which("git", path=stripped))
            self.assertIsNotNone(
                shutil.which("rustc", path=stripped)
                or shutil.which("cargo", path=stripped),
                "stripping git from PATH also removed the Rust toolchain",
            )
            return stripped
        stub_dir = self.scratch / "no-git"
        stub_dir.mkdir(exist_ok=True)
        stub = stub_dir / "git"
        stub.write_text("#!/bin/sh\nexit 127\n", encoding="utf-8")
        stub.chmod(0o755)
        return os.pathsep.join([str(stub_dir), original])

    def check(self, overrides: Mapping[str, str | None]) -> CheckResult:
        env = dict(self.base_env)
        for key, value in overrides.items():
            if value is not None:
                env[key] = value.replace("{scratch}", str(self.scratch))
        argv = [
            self.cargo,
            "check",
            "--manifest-path",
            str(WORKSPACE / "Cargo.toml"),
            "-p",
            "pcai-inference",
            "--lib",
            "--no-default-features",
            "--features",
            "ffi",
            "--locked",
            "--target-dir",
            str(self.target),
            "--message-format=json-render-diagnostics",
        ]
        result = subprocess.run(
            argv, env=env, capture_output=True, text=True, timeout=1500, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr[-6000:])
        out_dir = None
        lib_fresh = None
        for line in result.stdout.splitlines():
            if not line.startswith("{"):
                continue
            item = json.loads(line)
            package = item.get("package_id", "")
            if "#pcai-inference@" not in package and "pcai-inference " not in package:
                continue
            kinds = set(item.get("target", {}).get("kind", []))
            if item.get("reason") == "build-script-executed":
                out_dir = Path(item["out_dir"])
            elif item.get("reason") == "compiler-artifact" and LIBRARY_KINDS & kinds:
                lib_fresh = bool(item.get("fresh"))
        # Explicit checks rather than assertIsNotNone, so the type checker narrows.
        if out_dir is None:
            self.fail("cargo reported no build-script output directory")
        if lib_fresh is None:
            self.fail("cargo reported no pcai-inference library artifact")
        info = out_dir / "build_info.rs"
        # <target>/<profile>/build/pcai-inference-<hash>/{out,output}: `output`
        # is what the script printed.
        output = out_dir.parent / "output"
        frame = (out_dir / "build_identity.bin").read_bytes()
        if not (frame.startswith(IDENTITY_MARKER) and frame.endswith(b"\0")):
            self.fail("build_identity.bin is not MARKER + JSON + NUL")
        return CheckResult(
            info.read_text(encoding="utf-8"),
            info.stat().st_mtime_ns,
            lib_fresh,
            output.read_text(encoding="utf-8"),
            json.loads(frame[len(IDENTITY_MARKER) : -1]),
        )

    def test_each_metadata_input_change_reaches_the_generated_constants(self):
        """Protects: a rebuild that changes one version input embeds the new value.

        Detects: an input with no rerun trigger (the script is skipped and the old
        constant stays). Needs: cargo, network for the cold dependency fetch, and a
        few minutes for one cold check plus one warm check per step.
        """
        overrides: dict[str, str | None] = {}
        previous = self.check(overrides)
        for name, value, expected in INFERENCE_STEPS:
            overrides[name] = value
            current = self.check(overrides)
            with self.subTest(input=name, value=value):
                self.assertIn(
                    expected,
                    current.text,
                    f"changing only {name} did not regenerate build_info.rs",
                )
                self.assertNotEqual(previous.text, current.text)
                self.assertFalse(
                    current.lib_fresh,
                    f"changing only {name} did not recompile the crate",
                )
            previous = current

    def test_host_cuda_settings_do_not_claim_cuda(self):
        """Protects: a build without a CUDA feature never reports CUDA.

        Detects: HAS_CUDA derived from the build host. Before the fix, CUDA_PATH set
        to any directory made a no-CUDA build embed ``has_cuda: true``, which is what
        ``--version-json`` prints. Needs: cargo (shares the class's target directory).
        """
        fake = {
            "CUDA_PATH": "{scratch}/fake-cuda/v12.9",
            "CUDA_HOME": "{scratch}/fake-cuda-home/v12.0",
        }
        result = self.check(fake)
        self.assertIn("pub const HAS_CUDA: bool = false;", result.text)
        self.assertIn('pub const CUDA_VERSION: &str = "none";', result.text)

    def test_a_cpu_build_script_declares_path_in_its_output(self):
        """Protects: a CPU-only build tracks PATH (fast form of the warm rebuild).

        Detects: ``rerun-if-env-changed=PATH`` emitted only for CUDA builds. Cargo
        records what the script printed in
        ``<target>/<profile>/build/<pkg>-<hash>/output``.
        The nonce forces a fresh script run, so the file is not left over from an
        earlier run. Needs: cargo (shares the class's target directory).
        """
        result = self.check({"PCAI_BUILD_TYPE": self.fresh_nonce("output")})
        self.assertNotIn(
            "cargo:rustc-cfg=has_cuda",
            result.output,
            "this must be a CPU-only build for the check to mean anything",
        )
        self.assertIn("cargo:rerun-if-env-changed=PATH", result.output.splitlines())

    def test_git_becoming_available_reaches_a_warm_cpu_rebuild(self):
        """Protects: a warm CPU rebuild embeds the real commit once git is on PATH.

        Detects: the PR #403 failure. (1) A CPU build with git off PATH must say
        ``unknown`` for the commit, the branch and the version; (2) restoring PATH
        and rebuilding warm, in the same target directory, must embed HEAD; (3)
        removing git again must go back to ``unknown``, never keep the old commit.
        PCAI_BUILD_TYPE stays fixed (and unique to this run) across the steps, so
        PATH is the only input that changes between them. Needs: cargo, git.
        """
        head = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        original_path = self.base_env["PATH"]
        no_git_path = self.path_without_git()
        same = {"PCAI_BUILD_TYPE": self.fresh_nonce("gitpath")}

        without_git = self.check({**same, "PATH": no_git_path})
        for name in ("GIT_HASH", "GIT_HASH_SHORT", "GIT_BRANCH"):
            self.assertEqual(
                constant(without_git.text, name),
                "unknown",
                f"{name} must say unknown when git is unavailable",
            )
        self.assertTrue(
            constant(without_git.text, "VERSION").endswith("+unknown"),
            "VERSION must carry an explicit +unknown build",
        )
        self.assertEqual(without_git.identity["git_hash"], "unknown")
        self.assertEqual(without_git.identity["git_branch"], "unknown")
        self.assertEqual(
            without_git.identity["version"], constant(without_git.text, "VERSION")
        )

        with_git = self.check({**same, "PATH": original_path})
        self.assertEqual(
            constant(with_git.text, "GIT_HASH"),
            head,
            "a warm rebuild after git became available kept the stale metadata",
        )
        self.assertEqual(
            with_git.identity["git_hash"],
            head,
            "--version-json would still print the stale commit",
        )
        self.assertEqual(
            with_git.identity["version"], constant(with_git.text, "VERSION")
        )
        self.assertEqual(constant(with_git.text, "GIT_HASH_SHORT"), head[:7])
        self.assertIn(head[:7], constant(with_git.text, "VERSION"))
        self.assertNotIn("unknown", constant(with_git.text, "VERSION"))
        self.assertFalse(
            with_git.lib_fresh, "the crate was not rebuilt with the new metadata"
        )

        withdrawn = self.check({**same, "PATH": no_git_path})
        self.assertEqual(
            constant(withdrawn.text, "GIT_HASH"),
            "unknown",
            "git withdrawn from PATH left the previous commit embedded",
        )
        self.assertEqual(withdrawn.identity["git_hash"], "unknown")

    def test_unchanged_inputs_keep_the_build_cached(self):
        """Negative control: with no input changed, nothing reruns or recompiles.

        Detects: a rerun trigger on a path that does not exist, or on an input that
        changes on every build. Either reruns the script on every build, the state of
        this crate before #403.
        """
        overrides = {"PCAI_BUILD_VERSION": "1.2.3.4+eeeeeee"}
        self.check(overrides)
        first = self.check(overrides)
        second = self.check(overrides)
        self.assertEqual(
            first.mtime,
            second.mtime,
            "the build script ran again with no input changed",
        )
        self.assertTrue(
            second.lib_fresh, "the crate was recompiled with no input changed"
        )


if __name__ == "__main__":
    unittest.main()
