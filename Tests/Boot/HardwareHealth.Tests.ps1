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
            $commands = @($ast.FindAll(
                { param($n) $n -is [System.Management.Automation.Language.CommandAst] }, $true))
            $invoked = @($commands | ForEach-Object { $_.GetCommandName() } |
                Where-Object { $_ } | Sort-Object -Unique)

            $forbidden = 'Start-Service', 'Stop-Service', 'Set-Service', 'Restart-Service',
                         'Enable-PnpDevice', 'Disable-PnpDevice', 'Remove-PnpDevice',
                         'Set-ItemProperty', 'New-ItemProperty', 'Remove-ItemProperty',
                         'Remove-Item', 'pnputil', 'wevtutil', 'sc.exe'

            foreach ($f in $forbidden) {
                $invoked | Should -Not -Contain $f -Because "$f mutates system state"
            }

            # GetCommandName() returns $null for `& $someVariable ...`, so the
            # filter above silently DROPS every variable-invoked native command -
            # a blind spot in the one check whose entire job is to fail. Anything
            # invoked that way is therefore listed explicitly and reviewed here,
            # so a NEW variable invocation breaks the test until someone looks at
            # it. Fail-closed by construction rather than by vigilance.
            $allowedVariableInvocations = @('certutil')  # read-only: -user -key lists keys

            $variableInvoked = @($commands |
                Where-Object { -not $_.GetCommandName() } |
                ForEach-Object { $_.CommandElements[0].Extent.Text })

            foreach ($v in $variableInvoked) {
                $name = $v.TrimStart('$')
                $allowedVariableInvocations | Should -Contain $name -Because @"
'$v' is invoked as a native command through a variable, which GetCommandName()
cannot resolve. Confirm it is read-only, then add it to
`$allowedVariableInvocations above.
"@
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

        It 'reports the Hello readiness signals independently' {
            # Sensors OK does not mean Hello works, and biometrics are not the
            # only factor - PIN alone is working Hello. Each signal must be
            # surfaced separately or the report is misleading.
            $items = @($script:Summary.Findings | Where-Object { $_.Area -eq 'Hello' } |
                Select-Object -ExpandProperty Item)
            $items | Should -Contain 'WbioSrvc'
            $items | Should -Contain 'NGC credential store'
            $items | Should -Contain 'Biometric enrolment'
            $items | Should -Contain 'PIN and passkeys'
        }

        It 'POSITIVE CONTROL: the PIN/passkey finding matches certutil ground truth' {
            # The previous version of this check read the ACL-protected NGC store
            # and reported an access denial as "nothing enrolled". Assert against
            # the key store directly so a repeat of that failure fails the test.
            $lines = @(& (Join-Path $env:SystemRoot 'System32\certutil.exe') -user -key `
                -csp 'Microsoft Passport Key Storage Provider' 2>&1 | ForEach-Object { "$_".Trim() })
            $keys = @($lines | Where-Object { $_ -like 'S-1-5-*' })

            $finding = @($script:Summary.Findings |
                Where-Object { $_.Area -eq 'Hello' -and $_.Item -eq 'PIN and passkeys' })[0]
            $finding | Should -Not -BeNullOrEmpty

            if ($keys.Count -eq 0) {
                Set-ItResult -Skipped -Because 'no Passport keys on this machine to assert against'
                return
            }
            $finding.Detail | Should -Match "$($keys.Count) Passport key\(s\) total"
            if (@($keys | Where-Object { $_ -match 'uvkey-' }).Count -gt 0) {
                $finding.Detail | Should -Match 'PIN enrolled'
            }
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
