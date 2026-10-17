"""Exercise actual benchmark admission in pwsh; no timing or hardware claims."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = r"""
param([string]$Suite,[string]$OutputRoot,[string]$RunId)
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
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pcai-admission-")
        self.root = Path(self.tmp.name)
        (self.root / "Tests/Benchmarks").mkdir(parents=True)
        (self.root / "Config").mkdir()
        gitdir = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--absolute-git-dir"], text=True
        ).strip()
        # Read-only metadata pointer, no checkout/source mutation in the real repo.
        (self.root / ".git").write_text("gitdir: " + gitdir + "\n")
        self.gate = self.root / "Tests/Invoke-BenchmarkGate.ps1"
        shutil.copyfile(ROOT / "Tests/Invoke-BenchmarkGate.ps1", self.gate)
        (self.root / "Tests/Benchmarks/Invoke-PcaiToolingBenchmarks.ps1").write_text(
            RUNNER
        )
        (self.root / "Config/pcai-tooling-benchmarks.json").write_text(
            json.dumps({"suites": {"quick": ["one", "two"]}})
        )
        self.baseline = self.root / "baseline.json"
        good = self.run_gate("good", save=True)
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)

    def tearDown(self):
        self.tmp.cleanup()

    def run_gate(self, mode, save=False):
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
            env={**os.environ, "PCAI_FIXTURE_MODE": mode},
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


if __name__ == "__main__":
    unittest.main()
