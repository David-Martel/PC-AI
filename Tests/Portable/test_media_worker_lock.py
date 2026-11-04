"""Bounded process controls for the actual async FFI lock-acquisition source slice.

The probe compiles the verbatim acquisition block using minimal state types. It
does not load a model, media DLL, GPU, camera or network provider. This qualifies
the lock/error subpath, not the complete media FFI or its pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SETTINGS = None
ARTIFACT = None


def build_probe(settings) -> Path:
    manifest = Path(__file__).with_name("media-worker-lock") / "Cargo.toml"
    cargo = settings.cargo or shutil.which("cargo")
    if not cargo:
        raise RuntimeError("Cargo is required; no prerequisite success is inferred")
    source = settings.source.resolve(strict=True)
    target = settings.target_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env.update(
        PCAI_MEDIA_FFI_SOURCE=str(source),
        CARGO_BUILD_JOBS="2",
        RUSTC_WRAPPER="",
        RUSTC_WORKSPACE_WRAPPER="",
        RUSTFLAGS="",
        CARGO_ENCODED_RUSTFLAGS="",
        SCCACHE_DISABLE="1",
    )
    argv = [
        cargo,
        "build",
        "--manifest-path",
        str(manifest),
        "--locked",
        "--offline",
        "--jobs",
        "2",
        "--target-dir",
        str(target),
        "--message-format=json-render-diagnostics",
    ]
    result = subprocess.run(
        argv,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"probe compilation failed ({result.returncode}): {result.stderr[-6000:]}"
        )
    artifacts = []
    for line in result.stdout.splitlines():
        item = json.loads(line)
        if item.get("reason") == "compiler-artifact" and item.get("executable"):
            artifacts.append(Path(item["executable"]).resolve(strict=True))
    if len(artifacts) != 1:
        raise RuntimeError("one exact probe executable is required")
    executable = artifacts[0]
    if not executable.is_relative_to(target) or executable.is_symlink():
        raise RuntimeError("probe executable must be a regular owned target artifact")
    receipt = {
        "source": str(source),
        "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "artifact": str(executable),
        "artifactSha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "cargoArgv": argv,
        "fixtureFiles": {
            str(path.relative_to(Path(__file__).parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(__file__),
                manifest,
                manifest.with_name("Cargo.lock"),
                manifest.with_name("build.rs"),
                manifest.parent / "src/main.rs",
            )
        },
        "scope": (
            "Actual source acquisition slice; minimal model-free state; "
            "no complete FFI/hardware qualification"
        ),
    }
    settings.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)
    return executable


class MediaWorkerLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global ARTIFACT
        settings = SETTINGS
        if settings is None:
            cls.scratch = tempfile.TemporaryDirectory(prefix="pcai-media-lock-")
            cls.addClassCleanup(cls.scratch.cleanup)
            scratch = Path(cls.scratch.name)
            settings = argparse.Namespace(
                source=Path(__file__).resolve().parents[2]
                / "Native/pcai_core/pcai_media/src/ffi/mod.rs",
                target_dir=scratch / "target",
                receipt=scratch / "receipt.json",
                cargo=None,
            )
        ARTIFACT = build_probe(settings)

    def run_owned(self, scenario: str) -> subprocess.CompletedProcess:
        # This exact binary creates no descendants. subprocess.run kills and
        # reaps only its own handle on timeout; no external PID/name is selected.
        probe_env = os.environ.copy()
        probe_env["RUST_BACKTRACE"] = "0"
        probe_env["RUST_LIB_BACKTRACE"] = "0"
        try:
            result = subprocess.run(
                [str(ARTIFACT), scenario],
                capture_output=True,
                text=True,
                timeout=3,
                env=probe_env,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            stderr = error.stderr or b""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            self.fail(f"owned acquisition child exceeded 3s: {stderr[-2000:]}")
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        self.assertLessEqual(len(result.stdout) + len(result.stderr), 16384)
        return result

    def test_healthy_running_acquisition_preserves_existing_states(self):
        """Protects: healthy acquisition preserves running and cancelled requests.

        Detects: wrong statuses, extra requests or newly poisoned healthy state.
        Needs: native Cargo/linker and an owned std-only child; no model or device.
        Breadcrumb: Native/pcai_core/pcai_media/src/ffi/mod.rs
        """
        result = self.run_owned("healthy-running")
        self.assertIn("PCAI_HEALTHY_ACQUISITION_COMPLETE id=7", result.stdout)

    def test_healthy_cancelled_acquisition_does_not_rewrite_cancellation(self):
        """Protects: acquiring healthy state leaves an existing cancellation intact.

        Detects: lock handling overwrites cancellation or another request.
        Needs: exact source-bound std-only child and Cargo; no device/model/DLL.
        Breadcrumb: Native/pcai_core/pcai_media/src/ffi/mod.rs
        """
        result = self.run_owned("healthy-cancelled")
        self.assertIn("PCAI_HEALTHY_ACQUISITION_COMPLETE id=8", result.stdout)

    def test_poisoned_acquisition_records_failure_without_relocking(self):
        """Protects: poisoned acquisition records failure within a finite deadline.

        Detects: held-guard deadlock, wrong failure or unrelated lost cancellation.
        Needs: one poisoned owned child, 3s supervisor and native Rust; no hardware.
        Breadcrumb: Native/pcai_core/pcai_media/src/ffi/mod.rs
        """
        result = self.run_owned("poisoned-running")
        self.assertIn("PCAI_OWNED_POISON_ACQUISITION_ENTER", result.stderr)
        self.assertIn("PCAI_OWNED_POISON_ACQUISITION_RETURNED", result.stderr)
        self.assertIn("PCAI_OWNED_POISON_FAILURE_RECORDED", result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--cargo")
    SETTINGS, rest = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0], *rest])
