"""Black-box topology planner controls; needs an explicitly built pcai-tb binary.

Run: uv run --no-project python Tests/Portable/test_usb4_fabric_cli.py --exe PATH
Breadcrumb: Native/pcai_core/pcai_tb_cli/src/{main,fabric_plan}.rs.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class FabricCliTests(unittest.TestCase):
    executable: Path | None = None

    @classmethod
    def setUpClass(cls):
        # Generic admission discovery does not build this native executable.
        # The USB4 workflow invokes this script with mandatory --exe after build;
        # missing or invalid explicit paths still fail there, rather than skip.
        if cls.executable is None:
            raise unittest.SkipTest(
                "UNSUPPORTED: explicit built CLI required; run this script with --exe"
            )

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="pcai-fabric-cli-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)

    def invoke(self, *args):
        return subprocess.run(
            [str(self.executable), "topology-plan", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    def fixture(self):
        hosts = []
        for i, host in enumerate(("fixture-a", "fixture-b"), 1):
            peer = 3 - i
            hosts.append(
                {
                    "host": host,
                    "source": "local" if i == 1 else "management-ssh",
                    "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "interfaces": [
                        {
                            "interface_guid": f"00000000-0000-0000-0000-{i:012x}",
                            "mac": f"02:00:00:00:00:{i:02x}",
                            "interface_index": i + 10,
                            "interface_alias": "Ethernet display name",
                            "up": True,
                            "pnp_instance_id": (
                                f"USB4\\VEN_045E&DEV_0000083F&PROT_USB4NET\\{i}"
                            ),
                            "root_router_id": f"USB4\\ROOT_DEVICE_ROUTER\\{i}",
                            "addresses": [f"fe80::{i}%{i + 10}"],
                            "neighbors": [
                                {
                                    "address": f"fe80::{peer}%{i + 10}",
                                    "mac": f"02:00:00:00:00:{peer:02x}",
                                }
                            ],
                        }
                    ],
                }
            )
        return {
            "schema": "pcai.usb4-fabric-observation/v1",
            "ttl_seconds": 90,
            "hosts": hosts,
        }

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return str(path)

    # Protects: the public command is usable without Share/license or a registry.
    # Detects: missing CLI wiring, accidental host discovery/config writes, false ready.
    # Needs: built native binary only; Breadcrumb: main::topology-plan/build_plan.
    def test_real_cli_plans_unverified_segment_without_writes(self):
        path = self.write("observation.json", self.fixture())
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        run = self.invoke("--input", path, "--compact")
        self.assertEqual(run.returncode, 0, run.stderr)
        plan = json.loads(run.stdout)
        self.assertEqual(plan["schema"], "pcai.usb4-fabric-plan/v1")
        self.assertEqual(len(plan["segments"]), 1)
        self.assertEqual(plan["segments"][0]["state"], "ObservedUnverified")
        self.assertNotIn("routes", plan)
        self.assertNotIn("commands", plan)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    # Protects: observation planning never accepts configuration/apply authority.
    # Detects: forwarding --apply or unrelated mutable command options before input I/O.
    # Needs: no network/hardware; Breadcrumb: main topology-plan option admission.
    def test_apply_and_host_overrides_refuse_before_input_access(self):
        for flag in (
            "--apply",
            "--assign-pairwise-ipv4",
            "--computer",
            "--peer",
            "--registry",
            "--output",
        ):
            extra = [] if flag in ("--apply", "--assign-pairwise-ipv4") else ["fixture"]
            run = self.invoke(flag, *extra, "--input", str(self.root / "absent.json"))
            self.assertNotEqual(run.returncode, 0, flag)
            self.assertIn("unsupported", run.stderr)
            self.assertEqual(run.stdout, "")
        self.assertEqual(list(self.root.iterdir()), [])

    # Protects: unrelated commands' options cannot be silently accepted then ignored.
    # Detects: long-only admission checks missing generic parser short aliases.
    # Needs: absent input path to prove admission precedes input I/O. Breadcrumb:
    # main topology-plan option token walk, separate from global parser defaults.
    def test_unused_long_and_short_options_refuse_before_input_access(self):
        for args in (
            ("-t", "100"),
            ("-n", "2"),
            ("-c", "fixture"),
            ("-p", "fixture"),
            ("-r", "fixture"),
            ("-o", "fixture"),
            ("--mtu", "1500"),
            ("--metric", "15"),
            ("--timeout-ms", "100"),
            ("--iterations", "2"),
            ("--include-disconnected",),
            ("--include-unreachable",),
            ("--skip-legacy",),
        ):
            run = self.invoke(*args, "--input", str(self.root / "absent.json"))
            self.assertNotEqual(run.returncode, 0, args)
            self.assertIn("unsupported", run.stderr, args)
            self.assertEqual(run.stdout, "")

    # Protects: corrupt/oversized evidence cannot create a valid-looking topology.
    # Detects: ignored JSON fields, missing explicit input and unbounded file reads.
    # Needs: scratch files only; Breadcrumb: main::read_topology_json and serde schema.
    def test_missing_malformed_unknown_and_oversized_inputs_are_rejected(self):
        self.assertNotEqual(self.invoke().returncode, 0)
        directory = self.invoke("--input", str(self.root))
        self.assertNotEqual(directory.returncode, 0)
        self.assertIn("regular file", directory.stderr)
        for value in ({}, {**self.fixture(), "authenticated": True}):
            run = self.invoke("--input", self.write("bad.json", value))
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(run.stdout, "")
        large = self.root / "large.json"
        large.write_bytes(b" " * (1024 * 1024 + 1))
        run = self.invoke("--input", str(large))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("1 MiB", run.stderr)

    # Protects: a special input path cannot block the observer before type validation.
    # Detects: waiting for a FIFO writer instead of nonblocking type refusal.
    # Needs: POSIX mkfifo;
    # Windows is explicitly unsupported. Breadcrumb: main::read_topology_json.
    @unittest.skipUnless(
        hasattr(os, "mkfifo"), "UNSUPPORTED: POSIX FIFO control on Windows"
    )
    def test_fifo_input_is_refused_without_waiting_for_a_writer(self):
        fifo = self.root / "input.fifo"
        os.mkfifo(fifo)
        run = self.invoke("--input", str(fifo))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("regular file", run.stderr)

    # Protects: unplug/expiry withdraws observation IDs, never route commands.
    # Detects: historical green edge reuse or accidental write via --previous.
    # Needs: actual first-plan output and new disconnected snapshot;
    # Breadcrumb: build_plan.
    def test_disappeared_interface_withdraws_only_prior_observation_id(self):
        observation = self.fixture()
        run = self.invoke("--input", self.write("first.json", observation))
        self.assertEqual(run.returncode, 0, run.stderr)
        first = json.loads(run.stdout)
        previous = self.write("previous.json", first)
        observation["hosts"][0]["interfaces"] = []
        run = self.invoke(
            "--input",
            self.write("disconnected.json", observation),
            "--previous",
            previous,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        plan = json.loads(run.stdout)
        self.assertEqual(plan["segments"], [])
        self.assertEqual(plan["withdrawn_edge_ids"], [first["segments"][0]["edge_id"]])
        self.assertEqual(json.loads(Path(previous).read_text()), first)
        self.assertNotIn("routes", plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    FabricCliTests.executable = options.exe.resolve(strict=True)
    unittest.main(argv=[__file__, *remaining])
