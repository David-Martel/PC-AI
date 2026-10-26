"""Detect provenance, no-write, archive-integrity, and preservation failures."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from unittest.mock import patch

if TYPE_CHECKING:
    from usb4_fleet_release import BuildReceipt


class FixtureHost(TypedDict, total=False):
    id: object
    platform: object
    architecture: object
    backend: object
    policyPath: object


class FixturePayload(TypedDict):
    platform: str
    kind: str
    supportedArchitectures: list[str]
    paths: list[str]


class FixtureRegistry(TypedDict):
    schemaVersion: int
    repository: str
    hosts: list[FixtureHost]
    payloads: dict[str, FixturePayload]


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "Tools/SystemScripts/Networking/usb4_fleet_release.py"
SPEC = importlib.util.spec_from_file_location("usb4_fleet_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
COMMIT = "1" * 40
OTHER_COMMIT = "2" * 40


class ReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="pcai-usb4-release-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.registry_path = "Config/usb4-fleet/fleet.json"
        self.policy_path = "Config/usb4-fleet/hosts/host-a.json"
        self.script_path = "Tools/peer.ps1"
        self.output = self.root / "release.zip"
        self.registry: FixtureRegistry = {
            "schemaVersion": 1,
            "repository": "https://github.com/example/project",
            "hosts": [
                {
                    "id": "host-a",
                    "platform": "windows",
                    "architecture": "x86_64",
                    "backend": "windows-peer",
                    "policyPath": self.policy_path,
                }
            ],
            "payloads": {
                "windows-peer": {
                    "platform": "windows",
                    "kind": "deployable",
                    "supportedArchitectures": ["x86_64", "aarch64"],
                    "paths": [self.script_path],
                }
            },
        }
        self.write(self.script_path, b"Write-Output 'actual payload'\n")
        self.write(
            self.policy_path, b'{"schemaVersion":1,"hostName":"HOST-A","mode":"Peer"}\n'
        )
        self.save_registry()
        self.capture_commit()
        self.calls: list[tuple[str, ...]] = []
        self.git_options: dict[str, object] = {}

    def write(self, name: str, data: bytes) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def save_registry(self) -> None:
        self.write(self.registry_path, (json.dumps(self.registry) + "\n").encode())

    def capture_commit(self) -> None:
        self.committed = {
            p.relative_to(self.root).as_posix(): p.read_bytes()
            for p in self.root.rglob("*")
            if p.is_file()
        }
        self.objects = {
            hashlib.sha1(data).hexdigest(): data for data in self.committed.values()
        }

    def git_io(self, root: Path, *args: str) -> bytes:
        self.assertEqual(root, self.root)
        self.calls.append(args)
        if args[:2] == ("rev-parse", "--show-toplevel"):
            return str(self.root).encode() + b"\n"
        if args[:2] == ("rev-parse", "--verify"):
            heads = [c for c in self.calls if c[:2] == ("rev-parse", "--verify")]
            return (
                OTHER_COMMIT
                if self.git_options.get("changed_head") and len(heads) > 1
                else COMMIT
            ).encode() + b"\n"
        if args[:2] == ("diff", "--cached"):
            return b"Tools/peer.ps1\n" if self.git_options.get("staged") else b""
        if args[0] == "ls-tree":
            self.assertEqual(args[2], COMMIT)
            name = args[-1]
            if name not in self.committed:
                return b""
            mode = "120000" if self.git_options.get("git_link") else "100644"
            oid = hashlib.sha1(self.committed[name]).hexdigest()
            return f"{mode} blob {oid}\t{name}\0".encode()
        if args[:2] == ("cat-file", "blob"):
            return self.objects[args[2]]
        if args[0] == "check-attr":
            self.assertEqual(
                args[1:8], ("--source", COMMIT, "-z", "text", "eol", "--", args[-1])
            )
            if self.git_options.get("malformed_attributes"):
                return b"incomplete\0"
            name = args[-1]
            text = self.git_options.get("text_attribute", "set")
            eol = self.git_options.get("eol_attribute", "crlf")
            return f"{name}\0text\0{text}\0{name}\0eol\0{eol}\0".encode()
        self.fail(f"Unexpected Git I/O: {args}")

    def build(self, **options: object) -> BuildReceipt:
        # Only the external Git I/O adapter is replaced. Validation, actual
        # filesystem reads, immutable-byte checks and ZIP publication are real.
        with patch.object(release, "git", self.git_io):
            return release.build_bundle(
                self.root, self.registry_path, "host-a", str(self.output), **options
            )

    def assert_no_artifact(self) -> None:
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".pcai-usb4-*")), [])

    def test_real_zip_binds_all_bytes_manifest_and_commit(self) -> None:
        result = self.build()
        self.assertEqual(result["state"], "Packaged")
        self.assertEqual(
            result["archiveSha256"],
            hashlib.sha256(self.output.read_bytes()).hexdigest(),
        )
        with zipfile.ZipFile(self.output) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["source"]["commit"], COMMIT)
            self.assertEqual(manifest["host"]["id"], "host-a")
            self.assertFalse(manifest["installationPerformed"])
            self.assertFalse(manifest["runtimeQualified"])
            self.assertEqual(
                set(archive.namelist()), {*self.committed, "manifest.json"}
            )
            for record in manifest["files"]:
                data = archive.read(record["path"])
                self.assertEqual(data, self.committed[record["path"]])
                self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
                self.assertEqual(record["size"], len(data))
            self.assertEqual(manifest["policy"]["path"], self.policy_path)
            self.assertTrue(
                all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in archive.infolist())
            )
        self.assertEqual(self.calls[-1], ("rev-parse", "--verify", "HEAD^{commit}"))

    def test_plan_and_validate_write_nothing(self) -> None:
        before = set(self.root.rglob("*"))
        self.assertEqual(
            release.load_registry(self.root, self.registry_path)["schemaVersion"], 1
        )
        result = self.build(dry_run=True)
        self.assertEqual(result["state"], "Planned")
        self.assertEqual(set(self.root.rglob("*")), before)
        self.assert_no_artifact()

    def test_attributed_crlf_checkout_archives_committed_lf_bytes_and_hash(
        self,
    ) -> None:
        committed = self.committed[self.script_path]
        self.write(self.script_path, committed.replace(b"\n", b"\r\n"))
        for text, eol in (
            ("set", "crlf"),
            ("set", "lf"),
            ("auto", "crlf"),
            ("auto", "lf"),
        ):
            with self.subTest(text=text, eol=eol):
                self.git_options.update(text_attribute=text, eol_attribute=eol)
                result = self.build()
                source = result["manifest"]["source"]
                self.assertEqual(source["contentOrigin"], "CommittedGitBlobs")
                self.assertEqual(
                    source["checkoutValidation"]["attributeSource"],
                    "GitEffectiveCheckoutPolicy",
                )
                self.assertFalse(
                    source["checkoutValidation"]["artifactNormalizationPerformed"]
                )
                with zipfile.ZipFile(self.output) as archive:
                    self.assertEqual(archive.read(self.script_path), committed)
                    record = next(
                        r
                        for r in result["manifest"]["files"]
                        if r["path"] == self.script_path
                    )
                    self.assertEqual(
                        record["sha256"], hashlib.sha256(committed).hexdigest()
                    )
                    self.assertEqual(record["size"], len(committed))
                self.output.unlink()
        self.assertTrue(
            any(
                c[0] == "check-attr" and c[1:3] == ("--source", COMMIT)
                for c in self.calls
            )
        )

    def test_real_git_local_override_is_declared_and_never_changes_payload_bytes(
        self,
    ) -> None:
        # Clone existing local Git objects without checkout, commits, hook
        # overrides, networking, or any mutation of the real source checkout.
        checkout = self.root / "attribute-fixture"
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--shared",
                "--no-checkout",
                str(ROOT),
                str(checkout),
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        source_commit = (
            subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                capture_output=True,
                check=True,
                timeout=30,
            )
            .stdout.decode()
            .strip()
        )
        real_git = release.git

        def actual_attribute_io(root: Path, *arguments: str) -> bytes:
            if arguments[0] == "check-attr":
                self.assertEqual(arguments[1:3], ("--source", COMMIT))
                return real_git(checkout, *arguments[:2], source_commit, *arguments[3:])
            return self.git_io(root, *arguments)

        committed = self.committed[self.script_path]
        self.write(self.script_path, committed.replace(b"\n", b"\r\n"))
        overrides = checkout / ".git/info/attributes"
        overrides.write_text(f"{self.script_path} -text -eol\n", encoding="utf-8")
        with (
            patch.object(release, "git", actual_attribute_io),
            self.assertRaises(release.ReleaseError),
        ):
            release.build_bundle(
                self.root, self.registry_path, "host-a", str(self.output)
            )
        self.assert_no_artifact()
        overrides.write_text(f"{self.script_path} text eol=crlf\n", encoding="utf-8")
        with patch.object(release, "git", actual_attribute_io):
            result = release.build_bundle(
                self.root, self.registry_path, "host-a", str(self.output)
            )
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(archive.read(self.script_path), committed)
            manifest = json.loads(archive.read("manifest.json"))
        policy = manifest["source"]["checkoutValidation"]
        self.assertEqual(policy["attributeSource"], "GitEffectiveCheckoutPolicy")
        self.assertIn("GitLocalGlobalAndSystemOverrides", policy["attributeInputs"])
        self.assertFalse(policy["artifactNormalizationPerformed"])
        record = next(
            r for r in result["manifest"]["files"] if r["path"] == self.script_path
        )
        self.assertEqual(record["sha256"], hashlib.sha256(committed).hexdigest())

    def test_real_windows_autocrlf_checkout_keeps_release_sources_canonical(
        self,
    ) -> None:
        checkout = self.root / "windows-checkout-fixture"
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--shared",
                "--no-checkout",
                str(ROOT),
                str(checkout),
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        # Use the actual proposed repository policy, including before it is
        # committed. The immutable file contents still come from Git HEAD.
        (checkout / ".gitattributes").write_bytes(
            (ROOT / ".gitattributes").read_bytes()
        )
        names = [
            "Tools/SystemScripts/Networking/usb4_fleet_release.py",
            "Tests/Portable/test_usb4_fleet_release.py",
        ]
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "-c",
                "core.autocrlf=true",
                "checkout",
                "HEAD",
                "--",
                *names,
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        for name in names:
            with self.subTest(path=name):
                committed = subprocess.run(
                    ["git", "-C", str(checkout), "show", f"HEAD:{name}"],
                    capture_output=True,
                    check=True,
                    timeout=30,
                ).stdout
                actual = (checkout / name).read_bytes()
                self.assertNotIn(b"\r\n", actual)
                self.assertEqual(actual, committed)

    def test_crlf_with_substantive_change_is_not_a_clean_checkout(self) -> None:
        changed = (
            self.committed[self.script_path].replace(b"\n", b"\r\n")
            + b"# foreign edit\r\n"
        )
        self.write(self.script_path, changed)
        with self.assertRaisesRegex(release.ReleaseError, "differ from committed"):
            self.build()
        self.assertEqual((self.root / self.script_path).read_bytes(), changed)
        self.assert_no_artifact()

    def test_crlf_binary_nul_and_invalid_utf8_cannot_use_text_exception(self) -> None:
        for data in (b"binary\0record\n", b"invalid\xffrecord\n"):
            with self.subTest(data=data):
                self.write(self.script_path, data)
                self.capture_commit()
                self.write(self.script_path, data.replace(b"\n", b"\r\n"))
                with self.assertRaisesRegex(
                    release.ReleaseError, "differ from committed"
                ):
                    self.build()
                self.assert_no_artifact()

    def test_crlf_unknown_unset_binary_or_uncontrolled_attributes_refuse(self) -> None:
        self.write(
            self.script_path, self.committed[self.script_path].replace(b"\n", b"\r\n")
        )
        for text, eol in (
            ("unset", "crlf"),
            ("unspecified", "crlf"),
            ("set", "unspecified"),
            ("auto", "unset"),
            ("set", "native"),
        ):
            with self.subTest(text=text, eol=eol):
                self.git_options.update(text_attribute=text, eol_attribute=eol)
                with self.assertRaisesRegex(
                    release.ReleaseError, "differ from committed"
                ):
                    self.build()
                self.assert_no_artifact()

    def test_crlf_malformed_attribute_output_fails_closed(self) -> None:
        self.write(
            self.script_path, self.committed[self.script_path].replace(b"\n", b"\r\n")
        )
        self.git_options["malformed_attributes"] = True
        with self.assertRaisesRegex(release.ReleaseError, "differ from committed"):
            self.build()
        self.assert_no_artifact()

    def test_selected_mutation_staged_drift_unknown_git_file_and_changed_head_refuse(
        self,
    ) -> None:
        for mutation in (
            "worktree",
            "registry",
            "policy",
            "staged",
            "untracked",
            "changed_head",
            "git_link",
        ):
            with self.subTest(mutation=mutation):
                originals = {
                    name: (self.root / name).read_bytes() for name in self.committed
                }
                self.calls.clear()
                self.git_options.clear()
                if mutation in {"worktree", "policy", "registry"}:
                    name = {
                        "worktree": self.script_path,
                        "policy": self.policy_path,
                        "registry": self.registry_path,
                    }[mutation]
                    (self.root / name).write_bytes(originals[name] + b" ")
                elif mutation == "untracked":
                    self.committed.pop(self.script_path)
                else:
                    self.git_options[mutation] = True
                with self.assertRaises(release.ReleaseError):
                    self.build()
                self.assert_no_artifact()
                for name, data in originals.items():
                    self.write(name, data)
                self.capture_commit()

    def test_unrelated_dirty_file_does_not_prevent_release(self) -> None:
        self.write("unrelated.txt", b"other agent's preserved work")
        result = self.build()
        self.assertEqual(result["state"], "Packaged")
        self.assertEqual(
            (self.root / "unrelated.txt").read_bytes(), b"other agent's preserved work"
        )
        with zipfile.ZipFile(self.output) as archive:
            self.assertNotIn("unrelated.txt", archive.namelist())

    def test_existing_output_and_racing_creator_are_preserved(self) -> None:
        self.output.write_bytes(b"other artifact")
        with self.assertRaises(release.ReleaseError):
            self.build()
        self.assertEqual(self.output.read_bytes(), b"other artifact")
        self.output.unlink()
        original_link = os.link

        def competitor(source: Path, target: Path) -> None:
            target.write_bytes(b"concurrent writer")
            original_link(source, target)

        with (
            patch.object(release.os, "link", competitor),
            self.assertRaises(FileExistsError),
        ):
            self.build()
        self.assertEqual(self.output.read_bytes(), b"concurrent writer")
        self.assertEqual(list(self.root.glob(".pcai-usb4-*")), [])

    def test_publication_io_failure_cleans_only_owned_stage(self) -> None:
        survivor = self.root / ".pcai-usb4-foreign.zip"
        survivor.write_bytes(b"preserve")
        with (
            patch.object(
                release.os, "link", side_effect=OSError("filesystem cannot link")
            ),
            self.assertRaises(OSError),
        ):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".pcai-usb4-*")), [survivor])
        self.assertEqual(survivor.read_bytes(), b"preserve")

    def test_traversal_secret_reserved_and_ambiguous_paths_refuse(self) -> None:
        for name in (
            "../outside",
            "/absolute",
            "C:/drive",
            "a\\b",
            "a//b",
            "a/./b",
            "NUL.txt",
            "a/COM9.",
            "a/$null",
            "a/file ",
            'a/bad"name',
            "a/file?",
            "a/file*",
            ".git/config",
            "credentials/token.json",
            "keys/private.key",
            "a\nfile",
            "manifest.json",
        ):
            with self.subTest(name=name), self.assertRaises(release.ReleaseError):
                release.relative_path(name)
        with self.assertRaises(release.ReleaseError):
            release.output_path("relative.zip")
        with self.assertRaises(release.ReleaseError):
            release.output_path(str(self.root / "absent" / "release.zip"))
        with self.assertRaises(release.ReleaseError):
            release.output_path(str(self.root / "release.zip:stream.zip"))

    def test_dotenv_payloads_never_publish_private_bytes(self) -> None:
        sentinel = b"private dotenv sentinel: preserve this input\n"
        for name in (
            "Config/.env",
            "Config/.env.local",
            "Config/Config.env.local",
            "Config/.ENV.PRODUCTION",
            "Config/.envrc",
            "Config/sub.env/secret.txt",
        ):
            with self.subTest(name=name):
                self.write(name, sentinel)
                self.registry["payloads"]["windows-peer"]["paths"] = [
                    self.script_path,
                    name,
                ]
                self.save_registry()
                self.capture_commit()
                try:
                    with self.assertRaises(release.ReleaseError):
                        self.build()
                    self.assert_no_artifact()
                    self.assertEqual((self.root / name).read_bytes(), sentinel)
                finally:
                    # Clean only the test-owned output if the implementation
                    # regresses, so later cases cannot pass due to a collision.
                    self.output.unlink(missing_ok=True)

    def test_native_linux_policy_is_bound_to_host_schema_and_epoch(self) -> None:
        self.registry["hosts"][0]["platform"] = "linux"
        self.registry["hosts"][0]["backend"] = "linux-native"
        payload = self.registry["payloads"].pop("windows-peer")
        payload["platform"] = "linux"
        payload["kind"] = "templates-only"
        self.registry["payloads"]["linux-native"] = payload
        self.save_registry()
        policy: dict[str, object] = {"schema": 1, "epoch": 2, "node_name": "host-a"}
        self.write(self.policy_path, json.dumps(policy).encode())
        self.capture_commit()
        result = self.build(dry_run=True)
        self.assertEqual(result["manifest"]["kind"], "templates-only")
        self.assertFalse(result["manifest"]["runtimeQualified"])
        self.assertIsNotNone(result["manifest"]["policy"])
        self.assert_no_artifact()
        for field, value in (
            ("node_name", "foreign"),
            ("schema", True),
            ("schema", 2),
            ("epoch", True),
            ("epoch", 0),
        ):
            with self.subTest(field=field, value=value):
                invalid = dict(policy)
                invalid[field] = value
                self.write(self.policy_path, json.dumps(invalid).encode())
                self.capture_commit()
                with self.assertRaises(release.ReleaseError):
                    self.build()
                self.assert_no_artifact()

    def test_schema_platform_architecture_backend_policy_and_secret_fields_refuse(
        self,
    ) -> None:
        cases = (
            ("schemaVersion", True),
            ("repository", "https://user:password@example.test/repo"),
            ("platform", "plan9"),
            ("architecture", "invented"),
            ("backend", "missing"),
            ("policyPath", None),
            ("hostName", "foreign"),
            ("secret", "never-package"),
            ("duplicateHost", True),
            ("crossPlatform", True),
            ("macosDeploy", True),
        )
        for field, value in cases:
            with self.subTest(field=field):
                registry = json.loads(json.dumps(self.registry))
                policy: dict[str, object] = {"hostName": "HOST-A"}
                if field in {"schemaVersion", "repository", "secret"}:
                    registry[field] = value
                elif field == "hostName":
                    policy[field] = value
                elif field == "duplicateHost":
                    registry["hosts"].append(registry["hosts"][0])
                elif field == "crossPlatform":
                    registry["hosts"][0]["platform"] = "linux"
                elif field == "macosDeploy":
                    registry["hosts"][0]["platform"] = "macos"
                    registry["payloads"]["windows-peer"]["platform"] = "macos"
                else:
                    registry["hosts"][0][field] = value
                self.write(self.registry_path, json.dumps(registry).encode())
                self.write(self.policy_path, json.dumps(policy).encode())
                with self.assertRaises(release.ReleaseError):
                    release.load_registry(self.root, self.registry_path)
                self.assert_no_artifact()

    def test_malformed_container_and_field_types_refuse_cleanly(self) -> None:
        mutations: tuple[tuple[tuple[str | int, ...], object], ...] = (
            (("repository",), []),
            (("payloads",), []),
            (("hosts",), {}),
            (("hosts", 0), None),
            (("payloads", "windows-peer"), None),
            (("payloads", "windows-peer", "platform"), []),
            (("payloads", "windows-peer", "kind"), []),
            (("payloads", "windows-peer", "supportedArchitectures"), [[]]),
            (("payloads", "windows-peer", "paths"), [[]]),
            (("hosts", 0, "platform"), []),
            (("hosts", 0, "architecture"), {}),
            (("hosts", 0, "backend"), []),
        )
        for location, value in mutations:
            with self.subTest(location=location):
                registry = json.loads(json.dumps(self.registry))
                container = registry
                for component in location[:-1]:
                    container = container[component]
                container[location[-1]] = value
                self.write(self.registry_path, json.dumps(registry).encode())
                with self.assertRaises(release.ReleaseError):
                    release.load_registry(self.root, self.registry_path)
                self.assert_no_artifact()

    def test_templates_remain_explicitly_not_deployed_or_runtime_qualified(
        self,
    ) -> None:
        self.registry["payloads"]["windows-peer"]["platform"] = "linux"
        self.registry["payloads"]["windows-peer"]["kind"] = "templates-only"
        self.registry["hosts"][0]["platform"] = "linux"
        self.registry["hosts"][0]["architecture"] = "aarch64"
        self.registry["hosts"][0].pop("policyPath")
        self.save_registry()
        self.capture_commit()
        result = self.build()
        self.assertEqual(result["manifest"]["kind"], "templates-only")
        self.assertFalse(result["manifest"]["runtimeQualified"])
        self.assertFalse(result["manifest"]["installationPerformed"])
        self.assertIsNone(result["manifest"]["policy"])

    def test_private_key_contents_and_case_collision_refuse(self) -> None:
        self.write(
            self.script_path,
            # Synthetic header only; this fixture contains no key material.
            b"-----BEGIN OPENSSH PRIVATE KEY-----\n"  # pragma: allowlist secret
            b"not for shipping\n",
        )
        self.capture_commit()
        with self.assertRaisesRegex(release.ReleaseError, "Private-key"):
            self.build()
        self.write(self.script_path.upper(), b"other")
        self.registry["payloads"]["windows-peer"]["paths"].append(
            self.script_path.upper()
        )
        self.save_registry()
        self.capture_commit()
        with self.assertRaisesRegex(release.ReleaseError, "Case-colliding"):
            self.build()

    def test_symlink_inputs_and_output_parents_refuse(self) -> None:
        target = self.root / "external.txt"
        target.write_bytes(b"external")
        linked = self.root / "link.ps1"
        try:
            linked.symlink_to(target)
        except OSError as exc:
            self.skipTest(f"OS denied symlink fixture creation: {exc}")
        self.registry["payloads"]["windows-peer"]["paths"] = ["link.ps1"]
        self.save_registry()
        with self.assertRaisesRegex(release.ReleaseError, "Link/reparse"):
            release.load_registry(self.root, self.registry_path)
        self.assertEqual(target.read_bytes(), b"external")
        parent_link = self.root / "parent-link"
        parent_link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(release.ReleaseError, "Link/reparse"):
            release.output_path(str(parent_link / "release.zip"))

    def test_git_adapter_checks_real_process_result_and_no_shell(self) -> None:
        with patch.object(
            release.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, b"actual", b""),
        ) as run:
            self.assertEqual(release.git(self.root, "rev-parse", "HEAD"), b"actual")
            run.assert_called_once_with(
                ["git", "-C", str(self.root), "rev-parse", "HEAD"],
                capture_output=True,
                check=False,
                timeout=30,
            )
        with (
            patch.object(
                release.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 23, b"", b"error"),
            ),
            self.assertRaises(release.ReleaseError),
        ):
            release.git(self.root, "rev-parse", "HEAD")

    def test_cli_validate_is_read_only_and_errors_are_nonzero(self) -> None:
        before = set(self.root.rglob("*"))
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(
                release.main(["--source-root", str(self.root), "validate"]), 0
            )
        self.assertEqual(json.loads(stdout.getvalue())["writesPerformed"], False)
        self.assertEqual(set(self.root.rglob("*")), before)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                release.main(
                    [
                        "--source-root",
                        str(self.root),
                        "--registry",
                        "../wrong",
                        "validate",
                    ]
                ),
                1,
            )

    def test_cli_dry_run_and_git_failure_cannot_claim_publication(self) -> None:
        arguments = [
            "--source-root",
            str(self.root),
            "build",
            "--host",
            "host-a",
            "--output",
            str(self.output),
            "--dry-run",
        ]
        with (
            patch.object(release, "git", self.git_io),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            self.assertEqual(release.main(arguments), 0)
        self.assertEqual(json.loads(stdout.getvalue())["state"], "Planned")
        self.assert_no_artifact()
        with (
            patch.object(
                release, "git", side_effect=subprocess.TimeoutExpired("git", 30)
            ),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(release.main(arguments), 1)
        self.assert_no_artifact()


if __name__ == "__main__":
    unittest.main()
