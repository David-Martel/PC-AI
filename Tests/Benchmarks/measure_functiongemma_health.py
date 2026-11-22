"""Measure first, cached, and refresh latency of FunctionGemma /health."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
import urllib.request


# Protects: comparability of isolated health runs. Detects: timeouts, missing
# telemetry, or an unexpected bind. Needs: a CPU runtime binary and loopback.
# Breadcrumb: rust-functiongemma-runtime/src/metrics.rs and handlers.rs.
def measure(binary: Path) -> dict:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PCAI_ROUTER_")
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with tempfile.TemporaryDirectory(prefix="pcai-health-bench-") as folder:
        root = Path(folder)
        config = root / "config.json"
        settings = {"router_addr": "auto", "router_engine": "heuristic"}
        config.write_text(json.dumps({"runtime": settings}), encoding="utf-8")
        child = subprocess.Popen(
            [str(binary), "--config", str(config)],
            cwd=root,
            env=env,
            stdout=subprocess.DEVNULL,
        )
        try:
            receipt = root / "Reports" / "functiongemma-runtime.addr"
            deadline = time.monotonic() + 10
            address = ""
            while time.monotonic() < deadline and child.poll() is None:
                if receipt.is_file():
                    address = receipt.read_text(encoding="utf-8").strip()
                    if address:
                        break
                time.sleep(0.01)
            if not address.startswith("127.0.0.1:"):
                raise RuntimeError("runtime did not publish a loopback address")

            result = {}
            for phase in ("first", "cached", "refresh"):
                if phase == "refresh":
                    # The runtime refreshes health telemetry after five seconds.
                    time.sleep(5.1)
                start = time.perf_counter()
                with opener.open(f"http://{address}/health", timeout=30) as response:
                    body = json.load(response)
                result[f"{phase}_ms"] = round((time.perf_counter() - start) * 1000, 2)
                metrics = body.get("metrics")
                if (
                    body.get("status") != "ok"
                    or not metrics
                    or metrics["total_memory_mb"] <= 0
                ):
                    raise RuntimeError(
                        f"{phase} health sample did not include memory telemetry"
                    )
                result[f"{phase}_cpu_usage"] = metrics["cpu_usage"]
                result["total_memory_mb"] = metrics["total_memory_mb"]
            return result
        finally:
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="Path to a staged CPU runtime binary")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    binary = args.binary.resolve(strict=True)
    with binary.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    samples = [measure(binary) for _ in range(args.trials)]
    medians = {
        key: statistics.median(sample[key] for sample in samples)
        for key in ("first_ms", "cached_ms", "refresh_ms")
    }
    receipt = {"binary_sha256": digest, "samples": samples, "median_ms": medians}
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
