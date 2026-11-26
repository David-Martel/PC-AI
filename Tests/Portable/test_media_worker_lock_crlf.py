"""CRLF checkouts must parse exactly like LF for the media worker lock contract.

Windows checkouts with autocrlf hand build.rs a CRLF copy of the media FFI
source. The contract extracts functions by their closing brace at column zero,
so the build script has to normalise line endings before it parses. These
controls build the probe from a CRLF copy of the real source and require the
same generated probe source as the LF build, and the same refusals for a
regressed source.

The generated source is located from cargo's `build-script-executed` message,
not a path glob, so the test holds under CARGO_BUILD_TARGET and any target dir.

Needs: native Cargo and a linker, as for test_media_worker_lock; no model,
media DLL, GPU or network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import test_media_worker_lock as contract

SOURCE = Path(__file__).resolve().parents[2] / "Native/pcai_core/pcai_media/src/ffi/mod.rs"


def lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def crlf(data: bytes) -> bytes:
    return lf(data).replace(b"\n", b"\r\n")


class CrlfContractTests(unittest.TestCase):
    def build(self, scratch: Path, name: str, source_bytes: bytes) -> tuple[Path, Path]:
        """Build the probe from `source_bytes`; return the executable and generated source."""
        source = scratch / f"{name}.rs"
        source.write_bytes(source_bytes)
        target = scratch / f"target-{name}"
        settings = argparse.Namespace(
            source=source,
            target_dir=target,
            receipt=scratch / f"receipt-{name}.json",
            cargo=None,
        )
        executable, out_dir = contract.build_probe_with_out_dir(settings)
        generated = out_dir / "worker_acquisition.rs"
        self.assertTrue(generated.is_file(), f"build script wrote no {generated}")
        return executable, generated

    def test_crlf_source_generates_the_same_probe_as_lf(self):
        """Protects: Windows autocrlf checkouts build the lock contract.

        Detects: build.rs matching raw LF-brace markers, which fail to find the
        function end at column zero in a CRLF source, or generating a probe
        that differs from the LF one.
        Needs: native Cargo and a linker; no model or device.
        Breadcrumb: Tests/Portable/media-worker-lock/build.rs
        """
        original = SOURCE.read_bytes()
        with tempfile.TemporaryDirectory(prefix="pcai-media-lock-crlf-") as scratch_name:
            scratch = Path(scratch_name)
            lf_exe, lf_generated = self.build(scratch, "lf", lf(original))
            crlf_exe, crlf_generated = self.build(scratch, "crlf", crlf(original))
            self.assertIn(b"\r\n", crlf(original))
            self.assertEqual(
                lf_generated.read_bytes(),
                crlf_generated.read_bytes(),
                "CRLF and LF sources must generate byte-identical probe code",
            )
            result = subprocess.run(
                [str(crlf_exe), "poisoned-running"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            self.assertIn("PCAI_OWNED_POISON_FAILURE_RECORDED", result.stdout)

    def test_crlf_regressed_source_is_still_refused(self):
        """Protects: CRLF normalisation does not weaken the structural assertions.

        Detects: a regression (panic instead of into_inner) that slips through
        because the CRLF copy is parsed more loosely than the LF one.
        Needs: native Cargo and a linker; no model or device.
        Breadcrumb: Tests/Portable/media-worker-lock/build.rs
        """
        regressed = lf(SOURCE.read_bytes()).replace(
            b"get_state().lock().unwrap_or_else(PoisonError::into_inner)",
            b'get_state().lock().expect("state")',
            1,
        )
        self.assertNotEqual(regressed, lf(SOURCE.read_bytes()))
        with tempfile.TemporaryDirectory(prefix="pcai-media-lock-crlf-") as scratch_name:
            with self.assertRaises(RuntimeError) as raised:
                self.build(Path(scratch_name), "regressed", crlf(regressed))
        self.assertIn("poisoned-lock recovery must use into_inner", str(raised.exception))

    def test_missing_build_script_message_fails_closed(self):
        """Protects: the generated-source lookup never passes on an empty result.

        Detects: a lookup that returns a guessed path or skips when cargo reports
        no build-script-executed message for the probe package (for example a
        cached build, or messages for other packages only).
        Needs: no cargo; synthetic cargo JSON.
        Breadcrumb: Tests/Portable/test_media_worker_lock.py probe_out_dir
        """
        with tempfile.TemporaryDirectory(prefix="pcai-media-lock-outdir-") as name:
            target = Path(name).resolve()
            out = target / "x" / "out"
            out.mkdir(parents=True)
            other = json.dumps(
                {"reason": "build-script-executed", "package_id": "other-crate 1.0.0", "out_dir": str(out)}
            )
            mine = json.dumps(
                {
                    "reason": "build-script-executed",
                    "package_id": f"path+file:///w#{contract.PROBE_PACKAGE}@0.1.0",
                    "out_dir": str(out),
                }
            )
            with self.assertRaisesRegex(RuntimeError, "no single build-script-executed"):
                contract.probe_out_dir("", target)
            with self.assertRaisesRegex(RuntimeError, "no single build-script-executed"):
                contract.probe_out_dir(other, target)
            with self.assertRaisesRegex(RuntimeError, "no single build-script-executed"):
                contract.probe_out_dir(mine + "\n" + mine, target)
            self.assertEqual(contract.probe_out_dir(other + "\n" + mine, target), out.resolve())
            outside = Path(name).parent.resolve()
            with self.assertRaisesRegex(RuntimeError, "inside the owned target"):
                contract.probe_out_dir(mine, outside / "elsewhere")


if __name__ == "__main__":
    unittest.main()
