"""Detect source-custody violations before building the Candle security backport."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "candle_vendor", ROOT / "Tools/verify_candle_vendor.py"
)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CandleVendorCustody(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="pcai-candle-custody-")
        self.addCleanup(self.scratch.cleanup)
        self.vendor = Path(self.scratch.name) / "vendor"
        shutil.copytree(ROOT / "vendor", self.vendor)
        self.package = self.vendor / "candle-core-0.11.0"
        self.manifest = self.vendor / "candle-core-provenance.json"

    def update_manifest(self, action):
        data = json.loads(self.manifest.read_text(encoding="utf-8"))
        action(data)
        self.manifest.write_text(json.dumps(data), encoding="utf-8")

    # Protects: the actual complete published-source packet remains admissible.
    # Detects: a checker that refuses every packet or omits one locked Candle version.
    # Needs: all three byte-bound vendor packages; no build or network.
    # Breadcrumb: Tools/verify_candle_vendor.py and the vendor provenance manifest.
    def test_actual_packet_is_complete(self):
        self.assertEqual(
            CHECKER.verify(self.vendor),
            {"packages": 3, "files": 319, "modified_files": 9},
        )

    # Protects: unchanged upstream bytes cannot silently drift under a valid receipt.
    # Detects: a consumed package file edited without its reviewed source binding.
    # Needs: an owned byte-altered copy of the actual upstream LICENSE file.
    # Breadcrumb: Tools/verify_candle_vendor.py digest comparison.
    def test_changed_upstream_bytes_are_refused(self):
        path = self.package / "LICENSE"
        path.write_bytes(path.read_bytes() + b"\nowned custody mutation\n")
        with self.assertRaisesRegex(ValueError, "Source bytes differ"):
            CHECKER.verify(self.vendor)

    # Protects: all published files except the expressly omitted library lock.
    # Detects: silently dropped source or license files.
    # Needs: an owned package copy with its LICENSE removed.
    # Breadcrumb: Tools/verify_candle_vendor.py regular-file admission.
    def test_missing_upstream_file_is_refused(self):
        (self.package / "LICENSE").unlink()
        with self.assertRaises(FileNotFoundError):
            CHECKER.verify(self.vendor)

    # Protects: an obsolete nested library lock cannot re-enter the dependency inputs.
    # Detects: copied generated/unreviewed package members ignored by hash iteration.
    # Needs: an owned additional Cargo.lock beneath an actual package.
    # Breadcrumb: Tools/verify_candle_vendor.py exact package-file set comparison.
    def test_unrecorded_library_lock_is_refused(self):
        (self.package / "Cargo.lock").write_text(
            "# owned extra input\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Unexpected/missing package files"):
            CHECKER.verify(self.vendor)

    # Protects: provenance identifies the checksum-verified original published crate.
    # Detects: a receipt silently naming a different package source.
    # Needs: an owned manifest copy with only its first crate checksum changed.
    # Breadcrumb: Tools/verify_candle_vendor.py PUBLISHED identity map.
    def test_replaced_crate_identity_is_refused(self):
        self.update_manifest(
            lambda data: data["packages"][0].update(crate_sha256="0" * 64)
        )
        with self.assertRaisesRegex(ValueError, "Published crate identity mismatch"):
            CHECKER.verify(self.vendor)

    # Protects: this backport changes only the manifest and two archive consumers.
    # Detects: a new upstream edit disguised by updating only its current-byte hash.
    # Needs: an owned provenance copy marking LICENSE as changed upstream code.
    # Breadcrumb: Tools/verify_candle_vendor.py CHANGED scope comparison.
    def test_new_unreviewed_edit_scope_is_refused(self):
        def alter(data):
            row = next(
                row for row in data["packages"][0]["files"] if row["path"] == "LICENSE"
            )
            row["upstream_sha256"] = "0" * 64

        self.update_manifest(alter)
        with self.assertRaisesRegex(ValueError, "Backport exceeds"):
            CHECKER.verify(self.vendor)

    # Protects: manifest paths cannot escape the selected package during validation.
    # Detects: traversal in a forged source receipt before any external file read.
    # Needs: an owned receipt with one ../ path; no outside source changes.
    # Breadcrumb: Tools/verify_candle_vendor.py relative.
    def test_nonlocal_receipt_path_is_refused(self):
        self.update_manifest(
            lambda data: data["packages"][0]["files"][0].update(path="../outside")
        )
        with self.assertRaisesRegex(ValueError, "Nonlocal manifest path"):
            CHECKER.verify(self.vendor)

    # Protects: reviewers can reconstruct the exact bounded upstream source delta.
    # Detects: altered patch receipts despite unchanged final source files.
    # Needs: an owned patch copy with one extra byte.
    # Breadcrumb: Tools/verify_candle_vendor.py patch_sha256 comparison.
    def test_changed_patch_receipt_is_refused(self):
        path = self.vendor / "patches/candle-core-0.11.0-zip9.patch"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Backport patch bytes differ"):
            CHECKER.verify(self.vendor)


if __name__ == "__main__":
    unittest.main()
