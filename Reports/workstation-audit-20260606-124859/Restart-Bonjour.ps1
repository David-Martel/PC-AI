[CmdletBinding(SupportsShouldProcess)]
param()

$reportRoot = Split-Path -Parent $PSCommandPath
$service = Get-Service -Name 'Bonjour Service' -ErrorAction Stop
$service | Select-Object Name,DisplayName,Status,StartType |
    Format-List |
    Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'bonjour-before-restart.txt')

if ($PSCmdlet.ShouldProcess('Bonjour Service', 'Restart service')) {
    Restart-Service -Name 'Bonjour Service' -Force -ErrorAction Stop
}

Start-Sleep -Seconds 5
Get-Service -Name 'Bonjour Service' |
    Select-Object Name,DisplayName,Status,StartType |
    Format-List |
    Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'bonjour-after-restart.txt')
