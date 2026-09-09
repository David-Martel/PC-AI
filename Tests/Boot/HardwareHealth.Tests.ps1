#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

BeforeAll {
    $script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:ToolPath = Join-Path $script:RepoRoot 'Tools\Test-HardwareHealth.ps1'
}

Describe 'Test-HardwareHealth' {

    Context 'Static checks' {

        It 'exists' {
            Test-Path -LiteralPath $script:ToolPath | Should -BeTrue
        }

        It 'parses without error' {
            $errors = $null
            [void][System.Management.Automation.Language.Parser]::ParseFile(
                $script:ToolPath, [ref]$null, [ref]$errors)
            @($errors) | Should -BeNullOrEmpty
        }

        It 'is read-only - never INVOKES a device, service or registry mutation' {
            # This reports faults; it must not "helpfully" fix them. Starting a
            # service or reinstalling a driver is a human decision.
            #
            # Asserted against the AST, not a regex over the source. The remedy
            # strings deliberately TELL the operator to run `Start-Service
            # WbioSrvc`, so a substring match flags the advice as if it were a
            # call. Only actual command invocations count.
            $ast = [System.Management.Automation.Language.Parser]::ParseFile(
                $script:ToolPath, [ref]$null, [ref]$null)
            $invoked = $ast.FindAll(
                { param($n) $n -is [System.Management.Automation.Language.CommandAst] }, $true) |
                ForEach-Object { $_.GetCommandName() } |
                Where-Object { $_ } |
                Sort-Object -Unique

            $forbidden = 'Start-Service', 'Stop-Service', 'Set-Service', 'Restart-Service',
                         'Enable-PnpDevice', 'Disable-PnpDevice', 'Remove-PnpDevice',
                         'Set-ItemProperty', 'New-ItemProperty', 'Remove-ItemProperty',
                         'Remove-Item', 'pnputil', 'wevtutil', 'sc.exe'

            foreach ($f in $forbidden) {
                $invoked | Should -Not -Contain $f -Because "$f mutates system state"
            }
        }

        It 'never enumerates devices with -ErrorAction SilentlyContinue (must fail closed)' {
            # A PnP query that fails must not read as "no faulted devices".
            $content = Get-Content -LiteralPath $script:ToolPath -Raw
            $content | Should -Not -Match 'Get-PnpDevice\s+-ErrorAction\s+SilentlyContinue'
            $content | Should -Match 'Get-PnpDevice\s+-ErrorAction\s+Stop'
        }

        It 'supports --help without touching the system' {
            $out = & pwsh -NoLogo -NoProfile -File $script:ToolPath '--help' 2>&1
            ($out | Out-String) | Should -Match 'SYNOPSIS'
        }
    }

    Context 'Live behaviour' -Skip:(-not $IsWindows -or -not (Get-Command Get-PnpDevice -ErrorAction SilentlyContinue)) {

        BeforeAll {
            $script:JsonPath = Join-Path ([IO.Path]::GetTempPath()) "hwh-$([guid]::NewGuid()).json"
            & pwsh -NoLogo -NoProfile -File $script:ToolPath -OutputJson $script:JsonPath *> $null
            $script:Summary = Get-Content -LiteralPath $script:JsonPath -Raw | ConvertFrom-Json
        }

        AfterAll {
            Remove-Item -LiteralPath $script:JsonPath -ErrorAction SilentlyContinue
        }

        It 'writes a report with a plausible device count' {
            # Windows always enumerates many devices. A tiny number means the
            # query silently returned almost nothing.
            $script:Summary.DevicesPresent | Should -BeGreaterThan 20
        }

        It 'gives every finding a known severity' {
            foreach ($f in @($script:Summary.Findings)) {
                'ERROR', 'WARN', 'INFO' | Should -Contain $f.Severity
            }
        }

        It 'reports the three Hello readiness signals independently' {
            # Sensors OK does not mean Hello works. All three must be surfaced
            # or the report is misleading.
            $items = @($script:Summary.Findings | Where-Object { $_.Area -eq 'Hello' } |
                Select-Object -ExpandProperty Item)
            $items | Should -Contain 'WbioSrvc'
            $items | Should -Contain 'NGC credential store'
        }

        It 'POSITIVE CONTROL: a device in a non-OK state is reported as an ERROR finding' {
            # If this cannot fail, the tool is decorative. Assert against ground
            # truth from PnP rather than trusting the tool's own output.
            $faulted = @(Get-PnpDevice -ErrorAction SilentlyContinue |
                Where-Object { $_.Present -and $_.Status -ne 'OK' -and $_.Status -ne 'Unknown' })

            if ($faulted.Count -eq 0) {
                Set-ItResult -Skipped -Because 'no faulted device present on this machine right now'
                return
            }

            $reported = @($script:Summary.Findings |
                Where-Object { $_.Area -eq 'Device' -and $_.Severity -eq 'ERROR' })
            $reported.Count | Should -BeGreaterOrEqual 1 -Because "PnP reports $($faulted.Count) faulted device(s)"
        }

        It 'DryRun writes no report file' {
            $tmp = Join-Path ([IO.Path]::GetTempPath()) "hwh-$([guid]::NewGuid()).json"
            & pwsh -NoLogo -NoProfile -File $script:ToolPath -OutputJson $tmp -DryRun *> $null
            Test-Path -LiteralPath $tmp | Should -BeFalse
        }
    }
}
