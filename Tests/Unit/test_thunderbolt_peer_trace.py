"""Verify diagnostic opt-in, callsite isolation, and restoration failures."""

import importlib.util
import io
import json
import os
import re
import signal
import socket
import tempfile
import time
import unittest
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable
from unittest.mock import patch


class Callsite(TypedDict):
    file: str
    line: int
    flags: str
    p: bool


class PrintingRestoration(TypedDict):
    printingRestored: bool
    errors: list[str]


class CaptureOptionsShape(Protocol):
    @property
    def output_dir(self) -> str: ...
    @property
    def expected_hostname(self) -> str: ...
    @property
    def tools_dir(self) -> str | None: ...
    @property
    def duration(self) -> int: ...


@runtime_checkable
class TraceModule(Protocol):
    CALLSITE: re.Pattern[str]
    CaptureOptions: Callable[[str, str, str | None, int], CaptureOptionsShape]

    def main(self, argv: Sequence[str] | None = None) -> int: ...
    def capture(self, args: CaptureOptionsShape) -> int: ...
    def parse_callsites(self, text: str) -> dict[str, Callsite]: ...
    def disabled_selectors(
        self, states: dict[str, Callsite]
    ) -> list[tuple[str, int]]: ...
    def restore_printing(
        self,
        control: Path | None,
        selectors: Sequence[tuple[str, int]],
        original: dict[str, Callsite],
        write: Callable[[str], object] | None = None,
        read: Callable[[], str] | None = None,
    ) -> PrintingRestoration: ...


def load_trace_module() -> TraceModule:
    source = (
        Path(__file__).resolve().parents[2]
        / "Tools/SystemScripts/Networking/capture_thunderbolt_peer_trace.py"
    )
    spec = importlib.util.spec_from_file_location("thunderbolt_peer_trace", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load diagnostic source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "main",
        "capture",
        "parse_callsites",
        "disabled_selectors",
        "restore_printing",
        "CaptureOptions",
    ):
        member: object = getattr(module, name, None)
        if not callable(member):
            raise TypeError(f"Diagnostic source lacks callable {name}")
    pattern: object = getattr(module, "CALLSITE", None)
    if not isinstance(pattern, re.Pattern):
        raise TypeError("Diagnostic source lacks its compiled callsite pattern")
    if not isinstance(module, TraceModule):
        raise TypeError("Diagnostic source does not implement the required API")
    return module


TRACE = load_trace_module()

ORIGINAL = """drivers/thunderbolt/tb.c:10 [thunderbolt]one =_ "one"
drivers/thunderbolt/tb.c:11 [thunderbolt]two =pf "two"
drivers/thunderbolt/tb.c:12 [thunderbolt]three =_ "three"
drivers/thunderbolt/tb.c:12 [thunderbolt]four =p "four"
"""


class ThunderboltTraceTests(unittest.TestCase):
    def test_default_and_explicit_dry_run_never_create_output_or_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "absent"
            with patch.object(
                TRACE, "capture", side_effect=AssertionError("capture called")
            ):
                for flags in ([], ["--dry-run"]):
                    stream = io.StringIO()
                    with redirect_stdout(stream):
                        self.assertEqual(
                            TRACE.main([*flags, "--output-dir", str(output)]), 0
                        )
                    self.assertIn('"dryRun": true', stream.getvalue())
                    self.assertFalse(output.exists())

    def test_apply_requires_output_and_hostname(self) -> None:
        with patch.object(
            TRACE, "capture", side_effect=AssertionError("capture called")
        ):
            with self.assertRaises(SystemExit) as error:
                _ = TRACE.main(["--apply"])
            self.assertEqual(error.exception.code, 2)

    def test_duration_and_unsafe_hostname_rejected(self) -> None:
        for arguments in (
            ["--duration", "0"],
            ["--duration", "61"],
            ["--expected-hostname", "host;command"],
        ):
            with self.assertRaises(SystemExit) as error:
                _ = TRACE.main(arguments)
            self.assertEqual(error.exception.code, 2)

    def test_mixed_selector_never_changes_preexisting_enabled_calls(self) -> None:
        states = TRACE.parse_callsites(ORIGINAL)
        self.assertEqual(len(states), 4)
        self.assertEqual(
            TRACE.disabled_selectors(states), [("drivers/thunderbolt/tb.c", 10)]
        )

    def test_restoration_preserves_enabled_printing_and_reports_exact_state(
        self,
    ) -> None:
        commands: list[str] = []
        restored = TRACE.restore_printing(
            None,
            [("drivers/thunderbolt/tb.c", 10)],
            TRACE.parse_callsites(ORIGINAL),
            write=commands.append,
            read=lambda: ORIGINAL,
        )
        self.assertEqual(commands, ["file drivers/thunderbolt/tb.c line 10 -p\n"])
        self.assertEqual(restored, {"printingRestored": True, "errors": []})
        changed = ORIGINAL.replace("one =_", "one =p")
        result = TRACE.restore_printing(
            None, [], TRACE.parse_callsites(ORIGINAL), read=lambda: changed
        )
        self.assertFalse(result["printingRestored"])

    def test_cleanup_attempts_all_selectors_after_write_failure(self) -> None:
        commands: list[str] = []

        def writer(command: str) -> None:
            commands.append(command)
            if "line 20" in command:
                raise OSError("denied")

        result = TRACE.restore_printing(
            None,
            [("drivers/thunderbolt/tb.c", 10), ("drivers/thunderbolt/tb.c", 20)],
            TRACE.parse_callsites(ORIGINAL),
            write=writer,
            read=lambda: ORIGINAL,
        )
        self.assertEqual(len(commands), 2)
        self.assertEqual(result["errors"], ["denied"])

    def test_capture_restores_real_handlers_and_debug_after_late_evidence_errors(
        self,
    ) -> None:
        for failing_file in (
            "debug-after.txt",
            "after.json",
            "summary.json",
            "global-state",
        ):
            with self.subTest(failing_file=failing_file):
                self.exercise_cleanup_failure(failing_file)

    def exercise_cleanup_failure(self, failing_file: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            trace_root = base / "tracefs"
            (trace_root / "instances").mkdir(parents=True)
            _ = (trace_root / "available_events").write_text(
                "ucsi:ucsi_connector_change\n"
            )
            _ = (trace_root / "tracing_on").write_text("1\n")
            control = base / "control"
            _ = control.write_text(ORIGINAL)
            output = base / "output"
            original_mkdir = Path.mkdir
            original_rmdir = Path.rmdir
            original_write = Path.write_text
            original_read = Path.read_text
            signals_before = {
                number: signal.getsignal(number)
                for number in (signal.SIGTERM, signal.SIGINT)
            }
            global_reads: list[Path] = []

            def map_path(value: str | Path) -> Path:
                if str(value) == "/sys/kernel/tracing":
                    return trace_root
                if str(value) == "/sys/kernel/debug/dynamic_debug/control":
                    return control
                return Path(value)

            def virtual_mkdir(
                path: Path,
                mode: int = 0o777,
                parents: bool = False,
                exist_ok: bool = False,
            ) -> None:
                result = original_mkdir(
                    path, mode=mode, parents=parents, exist_ok=exist_ok
                )
                if path.parent == trace_root / "instances":
                    event = path / "events/ucsi/ucsi_connector_change"
                    original_mkdir(event, parents=True)
                    _ = original_write(event / "enable", "0\n")
                    _ = original_write(path / "tracing_on", "0\n")
                    _ = original_write(path / "buffer_size_kb", "0\n")
                return result

            def virtual_rmdir(path: Path) -> None:
                if path.parent == trace_root / "instances":
                    for entry in sorted(
                        path.rglob("*"),
                        key=lambda item: len(item.parts),
                        reverse=True,
                    ):
                        if entry.is_dir():
                            original_rmdir(entry)
                        else:
                            entry.unlink()
                return original_rmdir(path)

            def virtual_write(
                path: Path,
                content: str,
                encoding: str | None = None,
                errors: str | None = None,
                newline: str | None = None,
            ) -> int:
                if path.parent == output and path.name == failing_file:
                    raise OSError("injected final evidence write failure")
                if path == control:
                    _, filename, _, line_number, action = content.split()
                    updated: list[str] = []
                    for line in original_read(control).splitlines():
                        match = TRACE.CALLSITE.match(line)
                        if (
                            match
                            and match.group(1) == filename
                            and match.group(2) == line_number
                        ):
                            flags = match.group(4).replace("_", "")
                            flags = (
                                flags + "p"
                                if action == "+p"
                                else flags.replace("p", "")
                            )
                            line = (
                                line[: match.start(4)]
                                + (flags or "_")
                                + line[match.end(4) :]
                            )
                        updated.append(line)
                    return original_write(control, "\n".join(updated) + "\n")
                return original_write(
                    path, content, encoding=encoding, errors=errors, newline=newline
                )

            def virtual_read(
                path: Path, encoding: str | None = None, errors: str | None = None
            ) -> str:
                if path == trace_root / "tracing_on":
                    global_reads.append(path)
                    if failing_file == "global-state" and len(global_reads) > 1:
                        raise OSError("injected final state read failure")
                return original_read(path, encoding=encoding, errors=errors)

            def interrupt_capture(_duration: float) -> None:
                # Exercise the installed real handler and actual finally path.
                handler = signal.getsignal(signal.SIGTERM)
                if not callable(handler):
                    self.fail("Capture did not install its termination handler")
                handler(signal.SIGTERM, None)

            args = TRACE.CaptureOptions(str(output), "fixture-host", None, 1)
            stream = io.StringIO()
            with (
                patch.object(TRACE, "Path", side_effect=map_path),
                patch.object(os, "geteuid", return_value=0, create=True),
                patch.object(socket, "gethostname", return_value="fixture-host"),
                patch.object(
                    TRACE,
                    "snapshot",
                    return_value={"inventory": "external fixture"},
                ),
                patch.object(time, "sleep", side_effect=interrupt_capture),
                patch.object(Path, "mkdir", virtual_mkdir),
                patch.object(Path, "rmdir", virtual_rmdir),
                patch.object(Path, "write_text", virtual_write),
                patch.object(Path, "read_text", virtual_read),
                redirect_stdout(stream),
            ):
                result = TRACE.capture(args)
            self.assertEqual(result, 1)
            summary = stream.getvalue().strip()
            self.assertIn('"captureError": "Capture interrupted by signal', summary)
            self.assertIn('"printingRestored": true', summary)
            self.assertIn('"instanceRemoved": true', summary)
            self.assertEqual(original_read(control), ORIGINAL)
            self.assertEqual(list((trace_root / "instances").iterdir()), [])
            self.assertEqual(
                {number: signal.getsignal(number) for number in signals_before},
                signals_before,
            )
            expected_errors = {
                "debug-after.txt": "Save restored debug state: "
                + "injected final evidence write failure",
                "after.json": "Save final inventory: "
                + "injected final evidence write failure",
                "summary.json": "Save cleanup summary: "
                + "injected final evidence write failure",
                "global-state": "Read global tracing state: "
                + "injected final state read failure",
            }
            self.assertIn(
                '"postCleanupEvidenceErrors": [\n    '
                + json.dumps(expected_errors[failing_file])
                + "\n  ]",
                summary,
            )
            if failing_file != "summary.json":
                self.assertEqual((output / "summary.json").read_text().strip(), summary)


if __name__ == "__main__":
    _ = unittest.main()
