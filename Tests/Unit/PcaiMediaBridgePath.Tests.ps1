BeforeAll {
    $repo = Join-Path $PSScriptRoot '../..'
    Import-Module (Join-Path $repo 'Modules/PcaiMedia.psm1') -Force
    Add-Type @'
using System.Text;
using System.Runtime.InteropServices;
public static class PcaiMediaPathFixture {
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] public static extern uint GetShortPathName(string path, StringBuilder result, uint size);
}
'@
}

Describe 'Managed bridge existing-file identity' {
    BeforeEach {
        $savedBundle = $env:PCAI_NATIVE_BUNDLE_ROOT
        $savedPath = $env:PATH
        $env:PCAI_NATIVE_BUNDLE_ROOT = $null
        $root = Join-Path $TestDrive 'selected-long-fixture-directory'
        New-Item -ItemType Directory (Join-Path $root 'bin') -Force | Out-Null
        $dll = Join-Path $root 'bin/PcaiNative.dll'
        'fixture bytes only' | Set-Content -LiteralPath $dll
        $foreign = Join-Path $TestDrive 'foreign/PcaiNative.dll'
        New-Item -ItemType Directory (Split-Path $foreign) -Force | Out-Null
        Copy-Item -LiteralPath $dll -Destination $foreign -Force
        InModuleScope PcaiMedia -Parameters @{ Root=$root; Dll=$dll } {
            param($Root, $Dll)
            $script:PathFixtureRoot = $Root
            $script:PathFixtureDll = $Dll
            Mock Get-PcaiProjectRoot { $script:PathFixtureRoot }
            Mock Get-PcaiMediaLoadedBridgePath { $null }
            Mock Import-PcaiMediaManagedBridge { $script:PathFixtureDll }
        }
    }
    AfterEach {
        $env:PCAI_NATIVE_BUNDLE_ROOT = $savedBundle
        $env:PATH = $savedPath
    }
    It 'fails closed on a missing path' {
        InModuleScope PcaiMedia -Parameters @{ Path=(Join-Path $TestDrive 'absent.dll') } {
            param($Path)
            { Resolve-PcaiMediaBridgeFilePath -Path $Path } | Should -Throw
        }
    }
    It 'rejects directories and non-filesystem provider items' {
        InModuleScope PcaiMedia -Parameters @{ Path=$root } {
            param($Path)
            { Resolve-PcaiMediaBridgeFilePath -Path $Path } | Should -Throw '*filesystem file*'
            { Resolve-PcaiMediaBridgeFilePath -Path 'Env:TEMP' } | Should -Throw '*filesystem file*'
        }
    }
    It 'canonicalizes a genuine existing short alias before managed loading' {
        $buffer = [Text.StringBuilder]::new(32768)
        [void][PcaiMediaPathFixture]::GetShortPathName($dll, $buffer, $buffer.Capacity)
        $short = $buffer.ToString()
        if (-not $short -or $short -eq $dll) {
            Set-ItResult -Skipped -Because 'Filesystem did not provide a distinct 8.3 alias.'
            return
        }
        InModuleScope PcaiMedia -Parameters @{ Short=$short; Long=$dll } {
            param($Short, $Long)
            Resolve-PcaiMediaBridgeFilePath -Path $Short | Should -BeExactly (Get-Item -LiteralPath $Long).FullName
            Mock Get-PcaiMediaLoadedBridgePath { $Short }
            Initialize-PcaiMediaFFI | Should -BeTrue
            Should -Invoke Import-PcaiMediaManagedBridge -Times 1 -ParameterFilter { $Path -eq (Get-Item -LiteralPath $Long).FullName }
        }
    }
    It 'rejects an already loaded foreign same-basename file before managed loading and PATH changes' {
        InModuleScope PcaiMedia -Parameters @{ Foreign=$foreign } {
            param($Foreign)
            Mock Get-PcaiMediaLoadedBridgePath { $Foreign }
            Initialize-PcaiMediaFFI -WarningAction SilentlyContinue | Should -BeFalse
            Should -Invoke Import-PcaiMediaManagedBridge -Times 0
        }
        $env:PATH | Should -BeExactly $savedPath
    }
    It 'rejects a foreign same-basename assembly returned by managed loading' {
        InModuleScope PcaiMedia -Parameters @{ Foreign=$foreign } {
            param($Foreign)
            Mock Import-PcaiMediaManagedBridge { $Foreign }
            Initialize-PcaiMediaFFI -WarningAction SilentlyContinue | Should -BeFalse
            Should -Invoke Import-PcaiMediaManagedBridge -Times 1
        }
        $env:PATH | Should -BeExactly $savedPath
    }
}
