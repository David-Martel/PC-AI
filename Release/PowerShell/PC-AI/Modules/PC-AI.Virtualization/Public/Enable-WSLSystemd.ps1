#Requires -Version 5.1
<#
.SYNOPSIS
    Enables systemd in a WSL distribution by updating /etc/wsl.conf.

.PARAMETER Distribution
    WSL distribution name (default: Ubuntu)

.PARAMETER RestartWSL
    Restart WSL after changing configuration (default: true)

.OUTPUTS
    PSCustomObject with status details
#>
function Enable-WSLSystemd {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$Distribution = 'Ubuntu',

        [Parameter()]
        [switch]$RestartWSL = $true
    )

    $result = [PSCustomObject]@{
        Distribution = $Distribution
        WslConfPath  = "\\wsl$\\$Distribution\\etc\\wsl.conf"
        Updated      = $false
        Restarted    = $false
        Error        = $null
    }

    try {
        # Ensure distro is running so \\wsl$ is available
        & wsl -d $Distribution -- echo "WSL up" 2>$null | Out-Null

        if (-not (Test-Path $result.WslConfPath)) {
            # Create file if missing
            New-Item -Path $result.WslConfPath -ItemType File -Force | Out-Null
        }

        $linesIn = @()
        if (Test-Path $result.WslConfPath) {
            $linesIn = Get-Content -Path $result.WslConfPath -ErrorAction SilentlyContinue
        }

        $linesOut = New-Object System.Collections.Generic.List[string]
        $inBoot = $false
        $bootFound = $false
        $systemdSet = $false

        foreach ($line in $linesIn) {
            if ($line -match '^\s*\[boot\]\s*$') {
                $inBoot = $true
                $bootFound = $true
                $linesOut.Add($line) | Out-Null
                continue
            }

            if ($line -match '^\s*\[.+\]\s*$') {
                if ($inBoot -and -not $systemdSet) {
                    $linesOut.Add('systemd=true') | Out-Null
                    $systemdSet = $true
                }
                $inBoot = $false
                $linesOut.Add($line) | Out-Null
                continue
            }

            if ($inBoot -and $line -match '^\s*systemd\s*=') {
                $linesOut.Add('systemd=true') | Out-Null
                $systemdSet = $true
                continue
            }

            $linesOut.Add($line) | Out-Null
        }

        if (-not $bootFound) {
            if ($linesOut.Count -gt 0) { $linesOut.Add('') | Out-Null }
            $linesOut.Add('[boot]') | Out-Null
            $linesOut.Add('systemd=true') | Out-Null
            $systemdSet = $true
        }
        elseif ($inBoot -and -not $systemdSet) {
            $linesOut.Add('systemd=true') | Out-Null
        }

        $content = ($linesOut -join "`r`n") + "`r`n"
        Set-Content -Path $result.WslConfPath -Value $content -Encoding ASCII
        $result.Updated = $true

        if ($RestartWSL) {
            wsl --shutdown | Out-Null
            Start-Sleep -Seconds 2
            & wsl -d $Distribution -- echo "WSL restarted" 2>$null | Out-Null
            $result.Restarted = $true
        }
    }
    catch {
        $result.Error = $_.Exception.Message
    }

    return $result
}
