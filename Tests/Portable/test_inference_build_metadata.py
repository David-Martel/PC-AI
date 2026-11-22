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

Breadcrumbs: Native/pcai_core/pcai_inference/build.rs (``tracked_env``);
docs/VERSIONING.md, "How builds, installs and tests read the version". CI runs
this file in linux-ci.yml's portable-contracts job, which discovers
``Tests/Portable/test_*.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
    """One cargo check: the generated build_info.rs, its mtime, crate freshness."""

    text: str
    mtime: int
    lib_fresh: bool


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
        cls.target = cls.scratch / "target"
        # Start from no metadata input at all, and never share a target directory.
        cls.base_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("PCAI_")
            and key not in {"CUDA_PATH", "CUDA_HOME", "CARGO_TARGET_DIR"}
        }

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
        return CheckResult(
            info.read_text(encoding="utf-8"), info.stat().st_mtime_ns, lib_fresh
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
