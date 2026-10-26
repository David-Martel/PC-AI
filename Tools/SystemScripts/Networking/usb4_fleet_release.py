"""Validate fleet records and package exact committed USB4 deployment inputs.

This publisher does not discover peers, install software, or change networking.
Windows uses the maintained PowerShell installer; Linux uses the native agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import TypedDict, cast
from urllib.parse import urlsplit

PLATFORMS = {"windows", "linux", "macos"}
ARCHITECTURES = {"x86", "x86_64", "aarch64"}
KINDS = {"deployable", "templates-only"}
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}\Z")
RESERVED = re.compile(
    r"(?:\$null|aux|con|nul|prn|com[1-9]|lpt[1-9])(?:\..*)?\Z", re.IGNORECASE
)
PRIVATE_PARTS = {".git", ".ssh", ".gnupg", ".aws", "secrets", "credentials", "private"}
PRIVATE_SUFFIXES = {".pem", ".key", ".pfx", ".p12", ".kdbx", ".env"}
SECRET_KEYS = {
    "password",
    "secret",
    "authtoken",
    "apikey",
    "privatekey",
    "credential",
    "credentials",
}
PRIVATE_HEADER = re.compile(rb"(?m)^-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----\r?$")
FIRST_PRINTABLE_CODE_POINT = 32
ATTRIBUTE_RECORD_WIDTH = 3
ATTRIBUTE_OUTPUT_FIELD_COUNT = 7


class Payload(TypedDict):
    platform: str
    kind: str
    supportedArchitectures: list[str]
    paths: list[str]


class HostIdentity(TypedDict):
    id: str
    platform: str
    architecture: str
    backend: str


class Host(HostIdentity, total=False):
    policyPath: str | None


class Registry(TypedDict):
    schemaVersion: int
    repository: str
    payloads: dict[str, Payload]
    hosts: list[Host]


class FileRecord(TypedDict):
    path: str
    sha256: str
    size: int
    gitBlob: str


class CheckoutValidation(TypedDict):
    rule: str
    attributeSource: str
    attributeInputs: list[str]
    textAttributes: list[str]
    eolAttributes: list[str]
    requiresUtf8WithoutNul: bool
    artifactNormalizationPerformed: bool


class SourceIdentity(TypedDict):
    repository: str
    commit: str
    contentOrigin: str
    checkoutValidation: CheckoutValidation
    registryPath: str
    registrySha256: str


class Manifest(TypedDict):
    schemaVersion: int
    artifactType: str
    source: SourceIdentity
    host: HostIdentity
    kind: str
    runtimeQualified: bool
    installationPerformed: bool
    policy: FileRecord | None
    files: list[FileRecord]


class RequiredReceipt(TypedDict):
    state: str
    output: str
    manifest: Manifest


class BuildReceipt(RequiredReceipt, total=False):
    archiveSha256: str


class Arguments(argparse.Namespace):
    source_root: Path = Path(__file__).absolute().parents[3]
    registry: str = "Config/usb4-fleet/fleet.json"
    command: str = ""
    host: str = ""
    output: str = ""
    dry_run: bool = False


class ReleaseError(ValueError):
    """A release input or publication condition was not safe or reproducible."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def relative_path(value: object) -> str:
    """Accept one portable, non-secret, repository-relative regular-file name."""
    if not isinstance(value, str) or not value:
        raise ReleaseError("A nonempty relative path is required")
    require(
        not any(ord(c) < FIRST_PRINTABLE_CODE_POINT for c in value),
        "Control characters in path",
    )
    require(not any(c in value for c in '\\:<>"|?*'), f"Nonportable path: {value}")
    parts = value.split("/")
    require(
        not PurePosixPath(value).is_absolute()
        and all(p not in {"", ".", ".."} for p in parts),
        f"Unsafe relative path: {value}",
    )
    for part in parts:
        require(part == part.rstrip(" ."), f"Trailing space/dot in path: {value}")
        require(not RESERVED.fullmatch(part), f"Reserved Windows path: {value}")
        require(part.casefold() not in PRIVATE_PARTS, f"Private material path: {value}")
        lower = part.casefold()
        require(
            not (
                lower.startswith(".env") or lower.endswith(".env") or ".env." in lower
            ),
            f"Private dotenv material path: {value}",
        )
    require(
        Path(parts[-1]).suffix.lower() not in PRIVATE_SUFFIXES,
        f"Private material path: {value}",
    )
    require(
        value.casefold() != "manifest.json",
        "manifest.json is reserved for release metadata",
    )
    return value


def no_links(path: Path) -> tuple[Path, ...]:
    """Reject links and explicitly classify components that do not exist yet."""
    missing: list[Path] = []
    for component in [*reversed(path.parents), path]:
        try:
            info = component.lstat()
        except FileNotFoundError:
            missing.append(component)
        else:
            require(
                not stat.S_ISLNK(info.st_mode)
                and not (getattr(info, "st_file_attributes", 0) & 0x400),
                f"Link/reparse path refused: {component}",
            )
    return tuple(missing)


def source_file(root: Path, name: object) -> Path:
    name = relative_path(name)
    path = root.joinpath(*name.split("/"))
    require(not no_links(path), f"Missing regular source file: {name}")
    require(path.is_file(), f"Missing regular source file: {name}")
    require(path.resolve().is_relative_to(root), f"Source escapes repository: {name}")
    return path


def no_secrets(value: object) -> None:
    if isinstance(value, dict):
        for key, child in cast(dict[object, object], value).items():
            require(
                re.sub(r"[_-]", "", str(key)).casefold() not in SECRET_KEYS,
                "Secret-bearing configuration fields must not be packaged",
            )
            no_secrets(child)
    elif isinstance(value, list):
        for child in cast(list[object], value):
            no_secrets(child)


def object_map(value: object, message: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ReleaseError(message)
    mapping = cast(dict[object, object], value)
    require(all(isinstance(key, str) for key in mapping), message)
    return cast(dict[str, object], mapping)


def object_list(value: object, message: str) -> list[object]:
    if not isinstance(value, list):
        raise ReleaseError(message)
    return cast(list[object], value)


def string_value(value: object, message: str) -> str:
    if not isinstance(value, str):
        raise ReleaseError(message)
    return value


def parse_json(text: str) -> object:
    """Treat untrusted JSON as an object until its structure is validated."""
    return cast(object, json.loads(text))


def load_registry(root: Path, registry_path: str) -> Registry:
    """Validate records and available payload files without creating any artifacts."""
    registry = object_map(
        parse_json(source_file(root, registry_path).read_text(encoding="utf-8-sig")),
        "Registry must be an object",
    )
    no_secrets(registry)
    require(
        type(registry.get("schemaVersion")) is int and registry["schemaVersion"] == 1,
        "Unknown registry schema",
    )
    url = urlsplit(
        string_value(registry.get("repository"), "Repository must be a URL string")
    )
    require(
        url.scheme == "https"
        and bool(url.hostname)
        and bool(url.path.strip("/"))
        and not url.username
        and not url.password
        and not url.query
        and not url.fragment,
        "Repository must be an HTTPS URL without credentials, query or fragment",
    )
    payloads = object_map(registry.get("payloads"), "Missing payload descriptors")
    hosts = object_list(registry.get("hosts"), "Missing host records")
    require(bool(payloads), "Missing payload descriptors")
    require(bool(hosts), "Missing host records")
    for backend, payload_value in payloads.items():
        require(
            bool(IDENTIFIER.fullmatch(backend)),
            "Invalid backend ID",
        )
        payload = object_map(payload_value, f"Invalid payload: {backend}")
        require(
            isinstance(payload.get("platform"), str)
            and string_value(payload["platform"], "Invalid payload platform")
            in PLATFORMS
            and isinstance(payload.get("kind"), str)
            and string_value(payload["kind"], "Invalid payload kind") in KINDS,
            f"Unknown payload platform/kind: {backend}",
        )
        architectures = object_list(
            payload.get("supportedArchitectures"), f"Unknown architecture: {backend}"
        )
        paths = object_list(payload.get("paths"), f"Invalid payload paths: {backend}")
        require(
            bool(architectures)
            and all(isinstance(a, str) and a in ARCHITECTURES for a in architectures),
            f"Unknown architecture: {backend}",
        )
        require(
            bool(paths)
            and all(isinstance(p, str) for p in paths)
            and len(set(paths)) == len(paths),
            f"Invalid payload paths: {backend}",
        )
        require(
            not (payload["platform"] == "macos" and payload["kind"] == "deployable"),
            "macOS deployment is not implemented",
        )
        for name in paths:
            _ = source_file(root, name)
    seen: set[str] = set()
    for host_value in hosts:
        host = object_map(host_value, "Host must be an object")
        host_id = string_value(host.get("id", ""), "Invalid/duplicate host ID")
        require(
            bool(IDENTIFIER.fullmatch(host_id)) and host_id not in seen,
            "Invalid/duplicate host ID",
        )
        seen.add(host_id)
        require(
            isinstance(host.get("platform"), str)
            and string_value(host["platform"], "Invalid host platform") in PLATFORMS
            and isinstance(host.get("architecture"), str)
            and string_value(host["architecture"], "Invalid host architecture")
            in ARCHITECTURES,
            f"Unknown host platform/architecture: {host_id}",
        )
        backend = string_value(host.get("backend"), f"Unknown host backend: {host_id}")
        require(backend in payloads, f"Unknown host backend: {host_id}")
        payload = object_map(payloads[backend], f"Invalid payload: {backend}")
        require(
            host["platform"] == payload["platform"]
            and host["architecture"]
            in object_list(payload["supportedArchitectures"], "Invalid architectures"),
            f"Backend does not support host: {host_id}",
        )
        policy_name = host.get("policyPath")
        require(
            policy_name is None or isinstance(policy_name, str),
            f"Policy path must be a string or absent: {host_id}",
        )
        if host["platform"] == "windows" and payload["kind"] == "deployable":
            require(
                bool(policy_name),
                f"Deployable Windows host requires a policy: {host_id}",
            )
        if policy_name:
            policy = object_map(
                parse_json(
                    source_file(root, policy_name).read_text(encoding="utf-8-sig")
                ),
                f"Policy must be an object: {host_id}",
            )
            no_secrets(policy)
            if host["platform"] == "windows":
                require(
                    isinstance(policy.get("hostName"), str)
                    and string_value(
                        policy["hostName"], "Invalid policy host"
                    ).casefold()
                    == host_id.casefold(),
                    f"Policy belongs to another host: {host_id}",
                )
            elif host["platform"] == "linux" and host["backend"] == "linux-native":
                epoch = policy.get("epoch")
                require(
                    type(policy.get("schema")) is int
                    and policy["schema"] == 1
                    and type(epoch) is int
                    and epoch > 0
                    and policy.get("node_name") == host_id,
                    f"Native policy identity/schema/epoch mismatch: {host_id}",
                )
    return cast(Registry, cast(object, registry))


def git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
        timeout=30,
    )
    require(result.returncode == 0, f"Git provenance operation failed: {arguments[0]}")
    return result.stdout


def attributed_crlf_checkout(
    root: Path, commit: str, name: str, current: bytes, committed: bytes
) -> bool:
    """Recognize only Git-attributed text checkout EOL differences, never artifacts."""
    if (
        b"\0" in current
        or b"\0" in committed
        or current.replace(b"\r\n", b"\n") != committed
    ):
        return False
    try:
        _ = current.decode("utf-8")
        _ = committed.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReleaseError(
            f"Selected bytes differ from committed blob: {name}; invalid UTF-8"
        ) from error
    # --source selects the commit tree, but Git still applies local/global/system
    # attribute overrides. This effective policy admits checkout EOL differences
    # only; payload bytes always come from the committed blob. No filter runs.
    raw = git(root, "check-attr", "--source", commit, "-z", "text", "eol", "--", name)
    fields = raw.split(b"\0")
    if len(fields) != ATTRIBUTE_OUTPUT_FIELD_COUNT or fields[-1] != b"":
        return False
    attributes: dict[bytes, bytes] = {}
    for index in range(0, len(fields) - 1, ATTRIBUTE_RECORD_WIDTH):
        path, attribute, value = fields[index : index + ATTRIBUTE_RECORD_WIDTH]
        if path != name.encode("utf-8") or attribute in attributes:
            return False
        attributes[attribute] = value
    return (
        set(attributes) == {b"text", b"eol"}
        and attributes[b"text"] in {b"set", b"auto"}
        and attributes[b"eol"] in {b"lf", b"crlf"}
    )


def committed_files(
    root: Path, names: list[str]
) -> tuple[str, dict[str, tuple[str, bytes, int]]]:
    """Read immutable blobs and refuse selected worktree/index drift or Git links."""
    top = Path(git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    require(top == root, "Source root must be the exact Git worktree root")
    commit = git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    require(
        bool(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit)),
        "Invalid Git commit identity",
    )
    require(
        not git(root, "diff", "--cached", "--name-only", commit, "--", *names).strip(),
        "Selected staged files differ from the commit",
    )
    files: dict[str, tuple[str, bytes, int]] = {}
    for name in names:
        current = source_file(root, name).read_bytes()
        entry = git(root, "ls-tree", "-z", commit, "--", name)
        require(
            entry.count(b"\0") == 1 and b"\t" in entry,
            f"Source is not a tracked file: {name}",
        )
        header, raw_name = entry[:-1].split(b"\t", 1)
        mode, kind, object_id = header.decode().split()
        require(
            raw_name.decode() == name
            and kind == "blob"
            and mode in {"100644", "100755"},
            f"Git link/non-file refused: {name}",
        )
        data = git(root, "cat-file", "blob", object_id)
        require(
            current == data
            or attributed_crlf_checkout(root, commit, name, current, data),
            f"Selected file bytes differ from committed blob: {name}",
        )
        require(not PRIVATE_HEADER.search(data), f"Private-key payload refused: {name}")
        files[name] = (object_id, data, 0o755 if mode == "100755" else 0o644)
    require(
        git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip() == commit,
        "Git HEAD changed during packaging",
    )
    return commit, files


def output_path(value: str) -> Path:
    path = Path(value)
    require(path.is_absolute(), "Output ZIP path must be absolute")
    require(path.suffix.lower() == ".zip", "Output must have a .zip extension")
    require(
        not any(
            ord(c) < FIRST_PRINTABLE_CODE_POINT or c in '<>:"|?*' for c in path.name
        ),
        "Unsafe output filename",
    )
    require(
        not RESERVED.fullmatch(path.name.rstrip(" ."))
        and path.name == path.name.rstrip(" ."),
        "Unsafe output filename",
    )
    missing = no_links(path)
    require(
        not missing or missing == (path,), "Output parent directory must already exist"
    )
    require(path.parent.is_dir(), "Output parent directory must already exist")
    require(not path.exists(), "Output already exists; refusing overwrite")
    return path


def build_bundle(
    root: Path, registry_path: str, host_id: str, output: str, *, dry_run: bool = False
) -> BuildReceipt:
    registry = load_registry(root, registry_path)
    matches = [h for h in registry["hosts"] if h["id"] == host_id]
    require(len(matches) == 1, f"Unknown host: {host_id}")
    host = matches[0]
    payload = registry["payloads"][host["backend"]]
    names = {relative_path(registry_path), *payload["paths"]}
    policy_name = host.get("policyPath")
    if policy_name:
        names.add(policy_name)
    require(
        len({n.casefold() for n in names}) == len(names),
        "Case-colliding archive entries",
    )
    commit, files = committed_files(root, sorted(names))
    # Reparse the authoritative bytes: a concurrent registry edit cannot pick files
    # from a different plan while still obtaining the old registry's blob identity.
    require(
        parse_json(files[registry_path][1].decode("utf-8-sig")) == registry,
        "Registry changed during packaging",
    )
    records: list[FileRecord] = [
        {
            "path": n,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "gitBlob": oid,
        }
        for n, (oid, data, _) in files.items()
    ]
    by_name = {r["path"]: r for r in records}
    manifest: Manifest = {
        "schemaVersion": 1,
        "artifactType": "pcai-usb4-fleet-release",
        "source": {
            "repository": registry["repository"],
            "commit": commit,
            "contentOrigin": "CommittedGitBlobs",
            "checkoutValidation": {
                "rule": "ExactBytesOrAttributedTextCRLFToLFOnly",
                "attributeSource": "GitEffectiveCheckoutPolicy",
                "attributeInputs": [
                    "ResolvedCommitTree",
                    "GitLocalGlobalAndSystemOverrides",
                ],
                "textAttributes": ["set", "auto"],
                "eolAttributes": ["lf", "crlf"],
                "requiresUtf8WithoutNul": True,
                "artifactNormalizationPerformed": False,
            },
            "registryPath": registry_path,
            "registrySha256": by_name[registry_path]["sha256"],
        },
        "host": {
            "id": host["id"],
            "platform": host["platform"],
            "architecture": host["architecture"],
            "backend": host["backend"],
        },
        "kind": payload["kind"],
        "runtimeQualified": False,
        "installationPerformed": False,
        "policy": by_name.get(policy_name) if policy_name else None,
        "files": records,
    }
    target = output_path(output)
    receipt: BuildReceipt = {
        "state": "Planned" if dry_run else "Packaged",
        "output": str(target),
        "manifest": manifest,
    }
    if dry_run:
        return receipt
    staged: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".pcai-usb4-", suffix=".zip", dir=target.parent, delete=False
        ) as handle:
            staged = Path(handle.name)
            with zipfile.ZipFile(
                handle, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                entries = {n: (data, mode) for n, (_, data, mode) in files.items()}
                entries["manifest.json"] = (
                    (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
                    0o644,
                )
                for name in sorted(entries):
                    data, mode = entries[name]
                    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = (stat.S_IFREG | mode) << 16
                    archive.writestr(info, data)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and cannot replace an existing target.
        # Unsupported filesystems fail closed instead of falling back to overwrite.
        require(
            not no_links(target.parent), "Output parent disappeared during publication"
        )
        os.link(staged, target)
        receipt["archiveSha256"] = hashlib.sha256(staged.read_bytes()).hexdigest()
        return receipt
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).absolute().parents[3]
    )
    _ = parser.add_argument(
        "--registry",
        default="Config/usb4-fleet/fleet.json",
        help="Repository-relative registry path",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _ = commands.add_parser(
        "validate", help="Validate registry/payload paths without writing"
    )
    build = commands.add_parser(
        "build", help="Package exact committed files; never install"
    )
    _ = build.add_argument("--host", required=True)
    _ = build.add_argument("--output", required=True)
    _ = build.add_argument("--dry-run", "-DryRun", action="store_true")
    args = Arguments()
    _ = parser.parse_args(argv, namespace=args)
    try:
        require(not no_links(args.source_root.absolute()), "Source root is missing")
        root = args.source_root.resolve(strict=True)
        require(root.is_dir(), "Source root is not a directory")
        result: object
        if args.command == "validate":
            registry = load_registry(root, args.registry)
            result = {
                "state": "Validated",
                "writesPerformed": False,
                "hosts": [h["id"] for h in registry["hosts"]],
            }
        else:
            result = build_bundle(
                root, args.registry, args.host, args.output, dry_run=args.dry_run
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ReleaseError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"USB4 release refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
