"""Exercise production native resolver through fresh-process P/Invoke calls."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
OVERRIDES = ("PCAI_CORE_LIB_PATH", "PCAI_TEST_CORE_DLL", "PCAI_NATIVE_DIR")


class NativeOverrideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="pcai-native-override-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        bridge = cls.root / "bridge"
        shutil.copytree(
            ROOT / "Native/PcaiNative",
            bridge,
            ignore=shutil.ignore_patterns("bin", "obj", "*.dll", "*.pdb"),
        )
        project = cls.root / "Probe.csproj"
        project.write_text(
            '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup>'
            "<OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework>"
            "<EnableDefaultCompileItems>false</EnableDefaultCompileItems>"
            '</PropertyGroup><ItemGroup><Compile Include="Program.cs" />'
            '</ItemGroup><ItemGroup><ProjectReference Include="'
            f"{escape(str(bridge / 'PcaiNative.csproj'))}"
            '" /></ItemGroup></Project>'
        )
        # Compile the actual production bridge, including its shared export verifier.
        (cls.root / "Program.cs").write_text("""
using System;
using System.Runtime.InteropServices;
using System.Text.Json;
using PcaiNative;
class Probe {
 [DllImport("pcai_core_lib", EntryPoint="fixture_answer")]
 static extern int Answer();
 [DllImport("pcai_core_lib", EntryPoint="pcai_core_test")]
 static extern uint CoreTest();
 static int Main(string[] args) {
  if (args.Length > 0 && args[0] == "exports") {
   Console.WriteLine(JsonSerializer.Serialize(PcaiCore.RequiredExports)); return 0;
  }
  NativeResolver.EnsureRegistered(typeof(Probe).Assembly);
  try {
   if (args.Length > 0) Console.WriteLine("CORE_TEST=" + CoreTest());
   else Console.WriteLine("ANSWER=" + Answer());
   return 0;
  }
  catch (DllNotFoundException e) {
   Console.Error.WriteLine(e.Message); return 31;
  }
  catch (EntryPointNotFoundException e) {
   Console.Error.WriteLine(e.Message); return 32;
  }
 }
}
""")
        cls.output = cls.root / "out"
        subprocess.run(
            ["dotnet", "build", str(project), "-c", "Release", "-o", str(cls.output)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        cls.filename = "pcai_core_lib.dll" if os.name == "nt" else "libpcai_core_lib.so"
        inventory = subprocess.run(
            ["dotnet", str(cls.output / "Probe.dll"), "exports"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        required_exports = json.loads(inventory.stdout)
        if not required_exports or not all(
            isinstance(name, str) and name.startswith("pcai_")
            for name in required_exports
        ):
            raise AssertionError("Production export inventory is missing or invalid")
        cls.valid = cls.root / "valid"
        cls.valid.mkdir()
        # Distinct sentinels prove selection. Uncalled placeholders satisfy only
        # the actual production symbol inventory, not the complete native ABI.
        for destination, answer in ((cls.output, 7), (cls.valid, 42)):
            rust = cls.root / f"answer{answer}.rs"
            source = (
                '#[no_mangle] pub extern "C" fn fixture_answer() -> i32 {'
                f" {answer} " + "}"
            )
            source += "\n" + "\n".join(
                f'#[no_mangle] pub extern "C" fn {name}() -> usize {{ 0 }}'
                for name in required_exports
            )
            rust.write_text(source)
            subprocess.run(
                [
                    "rustc",
                    "--crate-type",
                    "cdylib",
                    str(rust),
                    "-o",
                    str(destination / cls.filename),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        cls.invalid = cls.root / "invalid"
        cls.invalid.mkdir()
        (cls.invalid / cls.filename).write_bytes(b"not a native library")
        cls.partial = cls.root / "partial"
        cls.partial.mkdir()
        partial_source = cls.root / "partial.rs"
        partial_source.write_text(
            '#[no_mangle] pub extern "C" fn fixture_answer() -> i32 { 99 }'
        )
        subprocess.run(
            [
                "rustc",
                "--crate-type",
                "cdylib",
                str(partial_source),
                "-o",
                str(cls.partial / cls.filename),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def probe(self, override=None, path=None, real=False):
        env = dict(os.environ)
        for key in (*OVERRIDES, "PCAI_NATIVE_BUNDLE_ROOT"):
            env.pop(key, None)
        env["PCAI_ROOT"] = str(self.root)
        if override:
            env[override] = str(path)
        return subprocess.run(
            [
                os.environ.get("PCAI_ADMISSION_DOTNET", "dotnet"),
                str(self.output / "Probe.dll"),
            ]
            + (["real"] if real else []),
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    # Protects explicit artifact custody; detects returning zero from the resolver
    # and allowing CLR fallback. Needs dotnet/rustc; no live production library.
    # Breadcrumb: Native/PcaiNative/NativeResolver.cs (production import resolver).
    def test_invalid_override_blocks_competing_loadable_fallback(self):
        for override in OVERRIDES:
            for path in (self.root / "missing", self.invalid):
                with self.subTest(override=override, path=path.name):
                    result = self.probe(override, path)
                    self.assertEqual(
                        result.returncode, 31, result.stdout + result.stderr
                    )
                    self.assertIn("Authoritative native override", result.stderr)
                    self.assertNotIn("ANSWER=", result.stdout)

    # Protects usable loading and explicit selection; detects unconditional
    # rejection or wrong selection through distinct native outputs7/42.
    # Needs dotnet8, rustc and owned symbol fixtures; no model or hardware.
    # Breadcrumb: Native/PcaiNative/NativeResolver.cs (production import resolver).
    def test_valid_override_and_default_load_real_native_artifacts(self):
        result = self.probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ANSWER=7", result.stdout)
        for override in OVERRIDES:
            with self.subTest(override=override):
                result = self.probe(override, self.valid)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("ANSWER=42", result.stdout)

    # Protects required-export refusal even when the requested symbol exists.
    # Detects executing a partial library or falling back to the complete default.
    # Needs dotnet8, rustc and owned fixtures; no real core, model or GPU.
    # Breadcrumb: NativeResolver.Resolve/VerifyLoadedCoreExports;
    #             PcaiCore.RequiredExports.
    def test_partial_override_refuses_export_before_native_sentinel(self):
        for override in OVERRIDES:
            with self.subTest(override=override):
                result = self.probe(override, self.partial)
                self.assertEqual(result.returncode, 32, result.stdout + result.stderr)
                self.assertIn("missing required export", result.stderr)
                self.assertIn(str(self.partial / self.filename), result.stderr)
                self.assertNotIn("ANSWER=", result.stdout)

    # Protects real-core FFI compatibility; detects failed binding or wrong magic.
    # Needs the CI-built host native core;
    # optional locally and explicitly skipped when no artifact is supplied.
    # Breadcrumb: Native/PcaiNative/NativeResolver.cs (production import resolver).
    def test_real_core_pinvoke_when_artifact_supplied(self):
        path = os.environ.get("PCAI_ADMISSION_REAL_CORE_PATH")
        if not path:
            self.skipTest("PCAI_ADMISSION_REAL_CORE_PATH not supplied")
        self.assertTrue(Path(path).is_file(), path)
        result = self.probe("PCAI_CORE_LIB_PATH", path, real=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"CORE_TEST={0x50434149}", result.stdout)


if __name__ == "__main__":
    unittest.main()
