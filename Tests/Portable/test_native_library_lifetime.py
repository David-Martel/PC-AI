"""Exercise temporary native admission ownership through the production bridge."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]


class NativeLibraryLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="pcai-native-lifetime-")
        cls.root = Path(cls.temp.name)
        # Copy production source unchanged; isolate build outputs from other agents.
        bridge = cls.root / "bridge"
        shutil.copytree(
            ROOT / "Native/PcaiNative",
            bridge,
            ignore=shutil.ignore_patterns("bin", "obj"),
        )
        cls.bridge_output = cls.root / "bridge-output"
        subprocess.run(
            [
                "dotnet",
                "build",
                str(bridge / "PcaiNative.csproj"),
                "-c",
                "Release",
                "-o",
                str(cls.bridge_output),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        probe = cls.root / "probe"
        probe.mkdir()
        (probe / "Probe.csproj").write_text(
            '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup>'
            "<OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework>"
            '</PropertyGroup><ItemGroup><Reference Include="PcaiNative">'
            f"<HintPath>{escape(str(cls.bridge_output / 'PcaiNative.dll'))}</HintPath>"
            "</Reference></ItemGroup></Project>"
        )
        (probe / "Program.cs").write_text(r"""
using System;
using System.Runtime.InteropServices;
using System.Text.Json;
using PcaiNative;
class Probe {
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode, ExactSpelling=true)]
 static extern IntPtr GetModuleHandleW(string name);
 [DllImport("libdl.so.2")] static extern IntPtr dlopen(string name, int flags);
 [DllImport("libdl.so.2")] static extern int dlclose(IntPtr handle);
 static bool Loaded(string path) {
  if (OperatingSystem.IsWindows()) return GetModuleHandleW(path) != IntPtr.Zero;
  var handle = dlopen(path, 6); // RTLD_NOW | RTLD_NOLOAD: observe, never load.
  if (handle == IntPtr.Zero) return false;
  dlclose(handle); return true;
 }
 static int Main(string[] args) {
  if (args[0] == "reject") {
   if (Loaded(args[1])) throw new Exception("Fixture already loaded");
   bool admitted = PcaiCore.VerifyLibraryExports(args[1], out var missing);
   Console.WriteLine(JsonSerializer.Serialize(new {
    admitted, missing = missing.Count, loadedAfter = Loaded(args[1]) }));
  } else {
   Console.WriteLine(JsonSerializer.Serialize(new {
    available = PcaiCore.IsAvailable,
    copied = PcaiCore.TestStringCopy("caf\u00e9 \u4f60\u597d") }));
  }
  return 0;
 }
}
""")
        cls.probe_output = cls.root / "probe-output"
        subprocess.run(
            [
                "dotnet",
                "build",
                str(probe / "Probe.csproj"),
                "-c",
                "Release",
                "-o",
                str(cls.probe_output),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        rust = cls.root / "partial.rs"
        rust.write_text(
            '#[no_mangle] pub extern "C" fn pcai_core_test() -> u32 { 0x50434149 }'
        )
        cls.partial = cls.root / ("partial.dll" if os.name == "nt" else "libpartial.so")
        subprocess.run(
            ["rustc", "--crate-type", "cdylib", str(rust), "-o", str(cls.partial)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def probe(self, mode, path):
        env = dict(os.environ)
        for name in ("PCAI_CORE_LIB_PATH", "PCAI_TEST_CORE_DLL", "PCAI_NATIVE_DIR"):
            env.pop(name, None)
        if mode == "real":
            env["PCAI_CORE_LIB_PATH"] = str(path)
        # communicate() used by run drains both pipes and kills/waits on timeout.
        result = subprocess.run(
            [
                os.environ.get("PCAI_ADMISSION_DOTNET", "dotnet"),
                str(self.probe_output / "Probe.dll"),
                mode,
                str(path),
            ],
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout.strip())

    # Protects temporary loader ownership; detects leaked references after rejection.
    # Needs Ubuntu libdl or Windows kernel32, dotnet8 and rustc; no hardware.
    # Breadcrumb: Native/PcaiNative/PcaiCore.cs VerifyLibraryExports.
    def test_rejected_partial_library_is_unloaded_before_child_exit(self):
        result = self.probe("reject", self.partial)
        self.assertFalse(result["admitted"])
        self.assertGreater(result["missing"], 0)
        self.assertFalse(result["loadedAfter"], result)

    # Protects valid native operations after freeing temporary admission references.
    # Detects freeing the P/Invoke-owned handle or rejecting a valid artifact.
    # Needs host real core path; absence explicitly SKIPs qualification.
    # Breadcrumb: Native/PcaiNative/PcaiCore.cs EvaluateAvailabilityCore/IsAvailable.
    def test_real_core_operations_survive_temporary_admission_release(self):
        path = os.environ.get("PCAI_ADMISSION_REAL_CORE_PATH")
        if not path:
            self.skipTest("PCAI_ADMISSION_REAL_CORE_PATH not supplied")
        self.assertTrue(Path(path).is_file(), path)
        result = self.probe("real", path)
        self.assertTrue(result["available"], result)
        self.assertEqual(result["copied"], "caf\u00e9 \u4f60\u597d")


if __name__ == "__main__":
    unittest.main()
