"""Verify published sources and reviewed archive patches without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath

PUBLISHED = {
    "0.9.2": "c15b675b80d994b2eadb20a4bbe434eabeb454eac3ee5e2b4cf6f147ee9be091",
    "0.10.2": "6bd9895436c1ba5dc1037a19935d084b838db066ff4e15ef7dded020b7c12a4a",
    "0.11.0": "5ecb245093b0f791b89d3420c3df9c6d49c60ab63ba54db896bf8a3baf486706",
}
CHANGED = {"Cargo.toml", "src/npy.rs", "src/pickle.rs"}


def regular(path: Path) -> None:
    """Refuse symlinks, Windows reparse points and non-regular source files."""
    for current in (path, *path.parents):
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Linked source path: {current}")
    if not path.is_file():
        raise ValueError(f"Expected regular file: {path}")


def relative(value: str) -> str:
    """Accept a single canonical portable file path inside its package."""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or ":" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError(f"Nonlocal manifest path: {value!r}")
    return value


def digest(path: Path) -> str:
    regular(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(vendor: Path) -> dict:
    """Validate every package file, narrow edit scope and complete patch receipts."""
    manifest = vendor / "candle-core-provenance.json"
    regular(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data["schema_version"] != 1 or len(data["packages"]) != len(PUBLISHED):
        raise ValueError("Unexpected provenance schema/package count")
    versions: set[str] = set()
    count = 0
    for package in data["packages"]:
        version = package["version"]
        if (
            package["package"] != "candle-core"
            or version in versions
            or PUBLISHED.get(version) != package["crate_sha256"]
        ):
            raise ValueError("Published crate identity mismatch")
        versions.add(version)
        root = vendor / f"candle-core-{version}"
        expected: set[str] = set()
        changed: set[str] = set()
        for row in package["files"]:
            name = relative(row["path"])
            if name in expected or name == "Cargo.lock":
                raise ValueError("Duplicate file or non-authoritative library lock")
            expected.add(name)
            for key in ("sha256", "upstream_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                    raise ValueError("Invalid source hash")
            if digest(root / name) != row["sha256"]:
                raise ValueError(f"Source bytes differ: {root / name}")
            if row["sha256"] != row["upstream_sha256"]:
                changed.add(name)
        if changed != CHANGED or set(package["modified_paths"]) != CHANGED:
            raise ValueError("Backport exceeds the three reviewed source files")
        omitted = package["omitted"]
        if len(omitted) != 1 or omitted[0]["path"] != "Cargo.lock":
            raise ValueError("Unexpected published-source omission")
        actual: set[str] = set()
        for path in root.rglob("*"):
            info = path.lstat()
            if (
                stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400
            ):
                raise ValueError(f"Linked vendor member: {path}")
            if not path.is_dir():
                regular(path)
                actual.add(path.relative_to(root).as_posix())
        if actual != expected:
            raise ValueError(
                f"Unexpected/missing package files: {sorted(actual ^ expected)}"
            )
        patch = relative(package["patch_path"])
        if patch != f"patches/candle-core-{version}-zip9.patch":
            raise ValueError("Unexpected patch receipt path")
        if digest(vendor / patch) != package["patch_sha256"]:
            raise ValueError("Backport patch bytes differ")
        count += len(expected)
    return {
        "packages": len(versions),
        "files": count,
        "modified_files": len(CHANGED) * len(versions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor", type=Path, default=Path(__file__).resolve().parent.parent / "vendor"
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.vendor.absolute()), sort_keys=True))


if __name__ == "__main__":
    main()
