#!/usr/bin/env python3
"""Capture isolated Thunderbolt diagnostics and restore temporary debug settings."""

import argparse
import datetime
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import NoReturn, Protocol, TypedDict, runtime_checkable


@runtime_checkable
class EffectiveIdentity(Protocol):
    def geteuid(self) -> int: ...


class Callsite(TypedDict):
    file: str
    line: int
    flags: str
    p: bool


class PrintingRestoration(TypedDict):
    printingRestored: bool
    errors: list[str]


class FinalRestoration(PrintingRestoration):
    instanceRemoved: bool
    instanceErrors: list[str]
    globalTracingUnchanged: bool


class CommandRecord(TypedDict, total=False):
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    error: str


class Inventory(TypedDict):
    tblist: CommandRecord
    tbadapters: CommandRecord
    tbtunnels: CommandRecord
    kernel: CommandRecord
    thunderboltDevices: list[str]


@dataclass(frozen=True)
class CaptureOptions:
    output_dir: str
    expected_hostname: str
    tools_dir: str | None
    duration: int


class CliOptions(argparse.Namespace):
    apply: bool = False
    dry_run: bool = False
    duration: int = 20
    output_dir: str | None = None
    expected_hostname: str | None = None
    tools_dir: str | None = None


CALLSITE = re.compile(r"^(\S+):(\d+) (\[thunderbolt\]\S+) =(\S+) (.*)$")
KERNEL_FILTER = re.compile(
    r"thunderbolt|usb4|ucsi|typec|USB disconnect|device descriptor|error -",
    re.IGNORECASE,
)


def parse_callsites(text: str) -> dict[str, Callsite]:
    """Return stable callsite identities, source selectors, and printing states."""
    result: dict[str, Callsite] = {}
    for line in text.splitlines():
        if match := CALLSITE.match(line):
            filename, number, function, flags, message = match.groups()
            identity = f"{filename}:{number} {function} {message}"
            if identity in result:
                raise RuntimeError(
                    f"Ambiguous duplicate dynamic-debug callsite: {identity}"
                )
            result[identity] = {
                "file": filename,
                "line": int(number),
                "flags": flags,
                "p": "p" in flags,
            }
    return result


def disabled_selectors(states: dict[str, Callsite]) -> list[tuple[str, int]]:
    """Avoid selectors that could affect a preexisting enabled callsite."""
    groups: dict[tuple[str, int], list[bool]] = {}
    for state in states.values():
        groups.setdefault((state["file"], state["line"]), []).append(state["p"])
    return sorted(selector for selector, enabled in groups.items() if not any(enabled))


def restore_printing(
    control: Path | None,
    selectors: Sequence[tuple[str, int]],
    original: dict[str, Callsite],
    write: Callable[[str], object] | None = None,
    read: Callable[[], str] | None = None,
) -> PrintingRestoration:
    """Attempt every restoration and verify all original printing states exactly."""
    if write is None:

        def default_write(command: str) -> int:
            if control is None:
                raise ValueError("A control path is required to write debug settings")
            return control.write_text(command, encoding="utf-8")

        write = default_write
    if read is None:

        def default_read() -> str:
            if control is None:
                raise ValueError("A control path is required to read debug settings")
            return control.read_text(encoding="utf-8")

        read = default_read
    errors: list[str] = []
    for filename, number in reversed(selectors):
        try:
            _ = write(f"file {filename} line {number} -p\n")
        except OSError as error:
            errors.append(str(error))
    try:
        current = parse_callsites(read())
        restored = {key: state["p"] for key, state in current.items()} == {
            key: state["p"] for key, state in original.items()
        }
    except (OSError, RuntimeError) as error:
        errors.append(str(error))
        restored = False
    return {"printingRestored": restored, "errors": errors}


def command_record(arguments: list[str], timeout: int = 5) -> CommandRecord:
    """Capture command failures as diagnostic evidence without hiding them."""
    try:
        result = subprocess.run(
            arguments, capture_output=True, text=True, check=False, timeout=timeout
        )
        return {
            "command": arguments,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": arguments, "error": str(error)}


def snapshot(tools_dir: Path | None) -> Inventory:
    """Read router inventory, adapter states, tunnels, and filtered kernel logs."""
    commands = [
        ("tblist", ["--all", "--verbose"]),
        ("tbadapters", ["--route", "0"]),
        ("tbtunnels", ["--verbose"]),
    ]
    records: dict[str, CommandRecord] = {}
    for name, arguments in commands:
        candidate = tools_dir / name if tools_dir else None
        executable = (
            str(candidate) if candidate and candidate.is_file() else shutil.which(name)
        )
        records[name] = (
            command_record([executable, *arguments])
            if executable
            else {"error": f"{name} unavailable"}
        )
    logs = command_record(["dmesg", "--ctime"])
    if "stdout" in logs:
        logs["stdout"] = "\n".join(
            line for line in logs["stdout"].splitlines() if KERNEL_FILTER.search(line)
        )
    return {
        "tblist": records["tblist"],
        "tbadapters": records["tbadapters"],
        "tbtunnels": records["tbtunnels"],
        "kernel": logs,
        "thunderboltDevices": sorted(
            path.name for path in Path("/sys/bus/thunderbolt/devices").glob("*")
        ),
    }


def interrupted(signum: int, _frame: FrameType | None) -> NoReturn:
    """Convert termination into an exception so cleanup still runs."""
    raise InterruptedError(f"Capture interrupted by signal {signum}")


def capture(args: CaptureOptions) -> int:
    """Apply a bounded diagnostic session with isolated tracing and cleanup."""
    identity: object = os
    if not isinstance(identity, EffectiveIdentity) or identity.geteuid() != 0:
        raise RuntimeError("--apply requires root")
    if socket.gethostname() != args.expected_hostname:
        raise RuntimeError("Expected hostname does not match this machine")
    output = Path(args.output_dir)
    if not output.is_absolute() or output.exists():
        raise RuntimeError(
            "Output must be an absolute path that does not already exist"
        )
    trace_root = Path("/sys/kernel/tracing")
    control = Path("/sys/kernel/debug/dynamic_debug/control")
    original_text = control.read_text(encoding="utf-8")
    original = parse_callsites(original_text)
    if not original:
        raise RuntimeError("No Thunderbolt dynamic-debug callsites found")
    events = [
        entry.split(":", 1)
        for entry in (trace_root / "available_events").read_text().splitlines()
        if entry.startswith(("ucsi:", "thunderbolt_net:"))
    ]
    if not events:
        raise RuntimeError(
            "No supported UCSI or Thunderbolt-network trace events found"
        )
    global_before = (trace_root / "tracing_on").read_text()
    output.mkdir(parents=True, mode=0o700)
    tools_dir = Path(args.tools_dir) if args.tools_dir else None
    instance = trace_root / "instances" / ("pcai-tb-" + uuid.uuid4().hex)
    summary: dict[str, object] = {
        "hostname": socket.gethostname(),
        "startedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "durationSeconds": args.duration,
        "outputDirectory": str(output),
        "instance": str(instance),
        "enabledEvents": [":".join(event) for event in events],
        "globalTracingBefore": global_before.strip(),
    }
    changed: list[tuple[str, int]] = []
    previous_signals = {
        number: signal.signal(number, interrupted)
        for number in (signal.SIGTERM, signal.SIGINT)
    }
    failure = None
    try:
        _ = (output / "debug-before.txt").write_text(original_text, encoding="utf-8")
        _ = (output / "before.json").write_text(
            json.dumps(snapshot(tools_dir), indent=2), encoding="utf-8"
        )
        instance.mkdir()
        _ = (instance / "tracing_on").write_text("0\n")
        _ = (instance / "buffer_size_kb").write_text("1024\n")
        for group, event in events:
            _ = (instance / "events" / group / event / "enable").write_text("1\n")
        for selector in disabled_selectors(original):
            changed.append(
                selector
            )  # A partially completed write must also be restored.
            filename, number = selector
            _ = control.write_text(
                f"file {filename} line {number} +p\n", encoding="utf-8"
            )
        _ = (instance / "tracing_on").write_text("1\n")
        time.sleep(args.duration)
        _ = (instance / "tracing_on").write_text("0\n")
        trace = (instance / "trace").read_text(encoding="utf-8")
        _ = (output / "trace.txt").write_text(trace, encoding="utf-8")
        summary["traceDataLines"] = sum(
            bool(line.strip()) and not line.lstrip().startswith("#")
            for line in trace.splitlines()
        )
    except (OSError, RuntimeError, InterruptedError) as error:
        failure = str(error)
    finally:
        evidence_errors: list[str] = []
        summary["postCleanupEvidenceErrors"] = evidence_errors
        try:
            # Prevent a second ordinary termination signal from interrupting rollback.
            for number in previous_signals:
                _ = signal.signal(number, signal.SIG_IGN)
            cleanup_errors: list[str] = []
            if instance.exists():
                try:
                    _ = (instance / "tracing_on").write_text("0\n")
                except OSError as error:
                    cleanup_errors.append(str(error))
                try:
                    instance.rmdir()  # Tracefs removes its virtual files automatically.
                except OSError as error:
                    cleanup_errors.append(str(error))
            printing = restore_printing(control, changed, original)
            restoration: FinalRestoration = {
                "printingRestored": printing["printingRestored"],
                "errors": printing["errors"],
                "instanceRemoved": not instance.exists(),
                "instanceErrors": cleanup_errors,
                "globalTracingUnchanged": False,
            }
            summary["restoration"] = restoration
            try:
                summary["globalTracingAfter"] = (
                    (trace_root / "tracing_on").read_text().strip()
                )
                restoration["globalTracingUnchanged"] = (
                    summary["globalTracingBefore"] == summary["globalTracingAfter"]
                )
            except OSError as error:
                evidence_errors.append(f"Read global tracing state: {error}")
            try:
                _ = (output / "debug-after.txt").write_text(
                    control.read_text(encoding="utf-8"), encoding="utf-8"
                )
            except OSError as error:
                evidence_errors.append(f"Save restored debug state: {error}")
            try:
                _ = (output / "after.json").write_text(
                    json.dumps(snapshot(tools_dir), indent=2), encoding="utf-8"
                )
            except (OSError, RuntimeError) as error:
                evidence_errors.append(f"Save final inventory: {error}")
            summary["changedCallsiteSelectors"] = len(changed)
            summary["finishedAt"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            summary["captureError"] = failure
            try:
                _ = (output / "summary.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
            except OSError as error:
                evidence_errors.append(f"Save cleanup summary: {error}")
        finally:
            # Evidence I/O must never leave the caller's handlers set to SIG_IGN.
            for number, handler in previous_signals.items():
                _ = signal.signal(number, handler)
    print(json.dumps(summary, indent=2))
    return (
        1
        if failure
        or not restoration["printingRestored"]
        or not restoration["instanceRemoved"]
        or not restoration["globalTracingUnchanged"]
        or restoration["errors"]
        or cleanup_errors
        or evidence_errors
        else 0
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Print a plan by default; explicit apply enables a temporary capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    _ = action.add_argument(
        "--apply",
        action="store_true",
        help="Require root and execute temporary diagnostic capture",
    )
    _ = action.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without creating files or changing settings (default)",
    )
    _ = parser.add_argument(
        "--duration",
        type=int,
        default=20,
        help="Capture interval from 1 through 60 seconds",
    )
    _ = parser.add_argument(
        "--output-dir",
        help="Explicit absolute, new output directory; required for apply",
    )
    _ = parser.add_argument(
        "--expected-hostname", help="Exact machine hostname; required for apply"
    )
    _ = parser.add_argument(
        "--tools-dir",
        help="Directory containing read-only tblist, tbadapters, and tbtunnels",
    )
    args = CliOptions()
    _ = parser.parse_args(argv, namespace=args)
    if not 1 <= args.duration <= 60:
        parser.error("--duration must be from 1 through 60")
    if args.expected_hostname and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,252}", args.expected_hostname
    ):
        parser.error("Unsafe --expected-hostname")
    if not args.apply:
        print(
            json.dumps(
                {
                    "dryRun": True,
                    "durationSeconds": args.duration,
                    "outputDirectory": args.output_dir,
                    "plan": [
                        "Verify root and exact hostname only on apply",
                        "Create unique tracefs instance; enable only supported "
                        + "ucsi and thunderbolt_net events",
                        "Enable printing only for previously disabled "
                        + "Thunderbolt callsites",
                        "Capture before/after read-only tbtools inventories "
                        + "and filtered kernel log",
                        "Restore original printing states, remove instance, "
                        + "verify global tracing unchanged",
                    ],
                },
                indent=2,
            )
        )
        return 0
    if not args.output_dir or not args.expected_hostname:
        parser.error("--apply requires --output-dir and --expected-hostname")
    try:
        return capture(
            CaptureOptions(
                args.output_dir, args.expected_hostname, args.tools_dir, args.duration
            )
        )
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"Capture failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
