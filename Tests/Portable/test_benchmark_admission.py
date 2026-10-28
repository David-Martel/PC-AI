"""Exercise actual benchmark admission in pwsh; no timing or hardware claims."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
RUNNER = r"""
param([string]$Suite,[string]$OutputRoot,[string]$RunId)
[IO.File]::WriteAllText($env:PCAI_FIXTURE_BREADCRUMB,'actual owned runner invocation')
if ($env:PCAI_FIXTURE_MODE -eq 'runner-failed') { exit 23 }
if ($env:PCAI_FIXTURE_MODE -eq 'no-report') { exit 0 }
$d=Join-Path $OutputRoot $RunId
New-Item -ItemType Directory -Path $d -Force|Out-Null
$rows=@(
 @{CaseId='one';Backend='native';MeanMs=10;MedianMs=10;Iterations=3},
 @{CaseId='two';Backend='native';MeanMs=10;MedianMs=10;Iterations=3}
)
$repo=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$configPath=Join-Path $repo 'Config/pcai-tooling-benchmarks.json'
$r=@{
 ConfigSha256=(Get-FileHash $configPath).Hash
 RunnerSha256=(Get-FileHash $PSCommandPath).Hash
 SchemaVersion=2
 RunId=$RunId
 GitHead=(& git -C $repo rev-parse HEAD).Trim()
 Suite=$Suite
 MeasurementsComplete=$true
 Machine=[Environment]::MachineName
 Architecture=[Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
 HostOS=[Runtime.InteropServices.RuntimeInformation]::OSDescription
 SelectedCases=@('one','two')
 Results=$rows
}
switch($env:PCAI_FIXTURE_MODE) {
 'stale-run' {$r.RunId='00000000000000000000000000000000'}
 'stale-head' {$r.GitHead='0000000000000000000000000000000000000000'}
 'bad-config' {$r.ConfigSha256='bad'}
 'bad-runner' {$r.RunnerSha256='bad'}
 'foreign-host' {$r.Machine='not-this-machine'}
 'missing-case' {$r.Results=@($rows[0])}
 'missing-backend' {$rows[0].Backend='fallback'}
 'empty' {$r.Results=@()}
 'invalid-number' {$rows[0].MeanMs=0}
 'failed-measurements' {$r.MeasurementsComplete=$false}
 'truthy-status' {$r.MeasurementsComplete='true'}
 'regression' {$rows[0].MeanMs=20}
 'duplicate' {$r.Results=@($rows[0],$rows[1],$rows[0])}
}
$r|ConvertTo-Json -Depth 8|Set-Content (Join-Path $d 'tooling-benchmark-report.json')
"""


class AdmissionTests(unittest.TestCase):
    tmp: tempfile.TemporaryDirectory[str]
    root: Path
    gate: Path
    baseline: Path

    @property
    def breadcrumb(self) -> Path:
        return self.root / "runner-invoked.txt"

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="pcai-admission-")
        self.root = Path(self.tmp.name)
        (self.root / "Tests/Benchmarks").mkdir(parents=True)
        (self.root / "Config").mkdir()
        gitdir = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--absolute-git-dir"], text=True
        ).strip()
        # Read-only metadata pointer, no checkout/source mutation in the real repo.
        _ = (self.root / ".git").write_text("gitdir: " + gitdir + "\n")
        self.gate = self.root / "Tests/Invoke-BenchmarkGate.ps1"
        _ = shutil.copyfile(ROOT / "Tests/Invoke-BenchmarkGate.ps1", self.gate)
        _ = (
            self.root / "Tests/Benchmarks/Invoke-PcaiToolingBenchmarks.ps1"
        ).write_text(RUNNER)
        _ = (self.root / "Config/pcai-tooling-benchmarks.json").write_text(
            json.dumps({"suites": {"quick": ["one", "two"]}})
        )
        self.baseline = self.root / "baseline.json"
        good = self.run_gate("good", save=True)
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_gate(
        self, mode: str, save: bool = False
    ) -> subprocess.CompletedProcess[str]:
        self.breadcrumb.unlink(missing_ok=True)
        args = [
            "pwsh",
            "-NoLogo",
            "-NoProfile",
            "-File",
            str(self.gate),
            "-BaselinePath",
            str(self.baseline),
            "-FailOnRegression",
            "-Format",
            "Json",
        ]
        if save:
            args.append("-SaveBaseline")
        return subprocess.run(
            args,
            env={
                **os.environ,
                "PCAI_FIXTURE_MODE": mode,
                "PCAI_FIXTURE_BREADCRUMB": str(self.breadcrumb),
            },
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    # Protects usable admission; detects rejecting valid fresh evidence or hiding
    # a real slowdown. Needs pwsh/git; breadcrumb: Tests/Invoke-BenchmarkGate.ps1.
    def test_fresh_compatible_evidence_passes_and_regression_fails(self):
        r = self.run_gate("good")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('"regression_count": 0', r.stdout)
        self.assertNotEqual(self.run_gate("regression").returncode, 0)

    # Protects fail-closed admission; detects stale reports, foreign identity,
    # missing measurements, backend fallback, invalid values and truthy status.
    # Needs synthetic runner only, no benchmark hardware; breadcrumb: same gate.
    def test_incomplete_or_incompatible_evidence_is_rejected(self):
        for mode in [
            "runner-failed",
            "no-report",
            "stale-run",
            "stale-head",
            "bad-config",
            "bad-runner",
            "foreign-host",
            "missing-case",
            "missing-backend",
            "empty",
            "invalid-number",
            "failed-measurements",
            "truthy-status",
            "duplicate",
        ]:
            with self.subTest(mode=mode):
                r = self.run_gate(mode)
                self.assertNotEqual(r.returncode, 0, mode + ": " + r.stdout + r.stderr)

    # Protects a missing baseline from certifying a build. Detects previous skip
    # exit0. Needs no runner; breadcrumb: baseline admission in the same gate.
    def test_missing_baseline_fails(self):
        self.baseline.unlink()
        self.assertNotEqual(self.run_gate("good").returncode, 0)

    # The real runner bytes, not just case/config labels, define the clocked
    # workload. These controls would all pass the former admission incorrectly.
    def test_saved_baseline_binds_exact_harness(self):
        saved = cast(
            "dict[str, object]",
            json.loads(self.baseline.read_text(encoding="utf-8-sig")),
        )
        runner = self.root / "Tests/Benchmarks/Invoke-PcaiToolingBenchmarks.ps1"
        self.assertEqual(
            saved["runner_sha256"],
            hashlib.sha256(runner.read_bytes()).hexdigest().upper(),
        )
        result = self.run_gate("good")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unbound_legacy_baseline_is_rejected(self):
        saved = cast(
            "dict[str, object]",
            json.loads(self.baseline.read_text(encoding="utf-8-sig")),
        )
        del saved["runner_sha256"]
        _ = self.baseline.write_text(json.dumps(saved))
        result = self.run_gate("good")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Baseline benchmark harness differs or is unbound", result.stderr)
        self.assertFalse(
            self.breadcrumb.exists(), "Unbound baseline launched a benchmark"
        )

    def test_changed_harness_cannot_compare_identical_means_and_case_names(self):
        runner = self.root / "Tests/Benchmarks/Invoke-PcaiToolingBenchmarks.ps1"
        # All reported numbers/host/config/case identities stay identical. The
        # changed harness represents a changed timing boundary or implementation.
        _ = runner.write_text(RUNNER + "\n# changed measurement implementation\n")
        result = self.run_gate("good")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Baseline benchmark harness differs or is unbound", result.stderr)
        self.assertFalse(
            self.breadcrumb.exists(), "Changed harness launched a benchmark"
        )

    def test_explicit_rebaseline_admits_new_harness(self):
        runner = self.root / "Tests/Benchmarks/Invoke-PcaiToolingBenchmarks.ps1"
        _ = runner.write_text(RUNNER + "\n# explicitly reviewed new harness\n")
        saved = self.run_gate("good", save=True)
        self.assertEqual(saved.returncode, 0, saved.stdout + saved.stderr)
        self.assertTrue(self.breadcrumb.exists(), "Explicit rebaseline did not run")
        result = self.run_gate("good")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.breadcrumb.exists(), "Compatible comparison did not run")


if __name__ == "__main__":
    unittest.main()
