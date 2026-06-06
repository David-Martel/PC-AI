[CmdletBinding(SupportsShouldProcess)]
param()

$instanceId = 'PCI\VEN_10DE&DEV_28B8&SUBSYS_223417AA&REV_A1\4&218D282A&0&0008'
$reportRoot = Split-Path -Parent $PSCommandPath

pnputil /enum-devices /instanceid $instanceId |
    Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'nvidia-internal-before-restart.txt')

if ($PSCmdlet.ShouldProcess($instanceId, 'Restart PnP device')) {
    pnputil /restart-device $instanceId |
        Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'nvidia-internal-restart-result.txt')
}

Start-Sleep -Seconds 5

pnputil /enum-devices /instanceid $instanceId |
    Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'nvidia-internal-after-restart.txt')

pnputil /enum-devices /problem |
    Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'pnputil-problems-after-nvidia-restart.txt')
