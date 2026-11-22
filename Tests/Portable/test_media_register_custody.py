"""Hash custody for the media weight-safety test registers.

Two registers bind individual test selectors to LF-normalised SHA-256 hashes:
`media-worker-lock-review.json` (Python selectors) and
`media-weight-safety-review.json` (Rust selectors). A register nobody can fail
is paper, so these checks recompute every hash from the working tree and fail on
drift, on an unbound Python test, and on any entry that claims more review than
it has. They need no Cargo, model, GPU or network.

Hashing rules, matching the registers' `hash_normalization` field:
- files: UTF-8 bytes with CRLF converted to LF;
- Python selectors: `ast.get_source_segment` of the test function;
- Rust selectors: the contiguous comment block above the test attribute through
  the function's closing brace at its own indent, plus a final newline.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
PORTABLE = ROOT / "Tests" / "Portable"
PYTHON_REGISTER = PORTABLE / "media-worker-lock-review.json"
RUST_REGISTER = PORTABLE / "media-weight-safety-review.json"

TEST_ATTR = re.compile(r"^\s*#\[(?:tokio::)?test(?:\(.*\))?\]\s*$")
FN = re.compile(r"^(\s*)(?:async\s+)?fn\s+([A-Za-z0-9_]+)\s*\(")
MOD = re.compile(r"^(\s*)(?:pub\s+)?mod\s+([A-Za-z0-9_]+)\s*\{")


def read_lf(relative: str) -> str:
    return (ROOT / relative).read_bytes().decode("utf-8").replace("\r\n", "\n")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def python_selector_hash(relative: str, name: str) -> str:
    text = read_lf(relative)
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(text, node)
            assert segment is not None
            return sha256(segment)
    raise KeyError(f"{name} not found in {relative}")


def rust_test_spans(relative: str) -> dict[str, str]:
    """Return {module-relative selector: span text} for every Rust test in a file."""
    lines = read_lf(relative).split("\n")
    spans: dict[str, str] = {}
    modules: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        module = MOD.match(line)
        if module:
            modules.append((len(module.group(1)), module.group(2)))
            continue
        while modules and line.strip() == "}" and len(line) - len(line.lstrip()) == modules[-1][0]:
            modules.pop()
        if not TEST_ATTR.match(line):
            continue
        cursor = index + 1
        while cursor < len(lines) and lines[cursor].strip().startswith("#["):
            cursor += 1
        function = FN.match(lines[cursor])
        if not function:
            continue
        indent, name = function.groups()
        end = cursor
        while lines[end] != indent + "}":
            end += 1
        start = index
        while start > 0 and lines[start - 1].lstrip().startswith("//"):
            start -= 1
        spans["::".join([m for _, m in modules] + [name])] = "\n".join(lines[start : end + 1]) + "\n"
    return spans


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class PythonRegisterCustody(unittest.TestCase):
    def test_every_python_binding_matches_the_working_tree(self):
        """Protects: the Python selectors in media-worker-lock-review.json stay bound to real bytes.

        Detects: a test, selector body, fixture peer or the FFI source edited
        without re-binding its hash (a stale register that still looks reviewed).
        Needs: the repository files only; no Cargo, model or device.
        """
        register = load(PYTHON_REGISTER)
        self.assertGreaterEqual(len(register["reviews"]), 6, "register must not be empty")
        for review in register["reviews"]:
            with self.subTest(selector=review["selector"]):
                self.assertEqual(sha256(read_lf(review["test_path"])), review["test_sha256"])
                self.assertEqual(
                    python_selector_hash(review["test_path"], review["selector"].split(".")[-1]),
                    review["selector_sha256"],
                )
                self.assertEqual(sha256(read_lf(review["source_path"])), review["source_sha256"])
                for peer in review["fixture_peers"]:
                    self.assertEqual(sha256(read_lf(peer["path"])), peer["sha256"], peer["path"])

    def test_every_lock_contract_test_is_bound_individually(self):
        """Protects: no lock-contract test runs without its own register entry.

        Detects: a new test_* method added to the lock or CRLF test files that
        was never bound, which would let it count as covered without a hash.
        Needs: the repository files only.
        """
        bound = {r["selector"] for r in load(PYTHON_REGISTER)["reviews"]}
        for relative in (
            "Tests/Portable/test_media_worker_lock.py",
            "Tests/Portable/test_media_worker_lock_crlf.py",
        ):
            tree = ast.parse(read_lf(relative))
            for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
                for method in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
                    if method.name.startswith("test_"):
                        with self.subTest(selector=f"{cls.name}.{method.name}"):
                            self.assertIn(f"{cls.name}.{method.name}", bound)

    def test_python_entries_claim_no_independent_review_they_lack(self):
        """Protects: an author self-review is never recorded as an independent one.

        Detects: a verdict or reviewer type implying independent approval while
        the receipt hash is empty.
        Needs: the repository files only.
        """
        for review in load(PYTHON_REGISTER)["reviews"]:
            with self.subTest(selector=review["selector"]):
                if review["review_receipt_sha256"] is None:
                    self.assertTrue(review["review_verdict"].startswith("pending"), review["review_verdict"])
                    self.assertNotIn("independent-approved", review["review_verdict"])
                    self.assertTrue(
                        review["reviewer_type"].startswith("author"), review["reviewer_type"]
                    )


class RustRegisterCustody(unittest.TestCase):
    def test_every_rust_binding_matches_the_working_tree(self):
        """Protects: each Rust selector in media-weight-safety-review.json stays bound to its text.

        Detects: a media test or its source file changing without a re-bound hash,
        a selector that no longer exists, or a duplicate entry.
        Needs: the repository files only; Rust is parsed as text, not built.
        """
        register = load(RUST_REGISTER)
        reviews = register["reviews"]
        self.assertGreaterEqual(len(reviews), 50, "register must keep every new or changed selector")
        keys = [(r["test_path"], r["selector"]) for r in reviews]
        self.assertEqual(len(keys), len(set(keys)), "duplicate selector entries")
        cache: dict[str, dict[str, str]] = {}
        for review in reviews:
            with self.subTest(selector=review["selector"], path=review["test_path"]):
                spans = cache.setdefault(review["test_path"], rust_test_spans(review["test_path"]))
                module_less = review["selector"]
                candidates = [k for k in spans if module_less == k or module_less.endswith("::" + k)]
                self.assertEqual(len(candidates), 1, f"selector not found uniquely: {candidates}")
                self.assertEqual(sha256(spans[candidates[0]]), review["selector_sha256"])
                for source in review["source_files"]:
                    self.assertEqual(sha256(read_lf(source["path"])), source["sha256"], source["path"])

    def test_rust_entries_are_author_self_review_only(self):
        """Protects: nothing in the Rust register is presented as independently reviewed.

        Detects: reviewer metadata that claims independence, a missing rationale
        field, or a disposition other than keep without an explanation.
        Needs: the repository files only.
        """
        for review in load(RUST_REGISTER)["reviews"]:
            with self.subTest(selector=review["selector"]):
                self.assertIn("independent review pending", review["reviewer_type"])
                for field in ("protects", "detects", "needs"):
                    self.assertTrue(review[field].strip(), field)
                self.assertEqual(review["disposition"], "keep")


if __name__ == "__main__":
    unittest.main()
