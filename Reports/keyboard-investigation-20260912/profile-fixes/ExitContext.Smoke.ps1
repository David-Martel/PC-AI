param(
    [string]$ProfilePath = 'C:/Users/david/.config/powershell/Microsoft.PowerShell_profile.ps1',
    [string]$OutputDirectory = $PSScriptRoot,
    [ValidateSet('owned','unowned')][string]$WriterMode = 'owned'
)
# Execute only the actual context/registration nodes, with fixture history commands.
$script:SmokeClosePath = Join-Path $OutputDirectory 'exit-smoke.closed'
$script:ReadLineHistorySpoolPath = Join-Path $OutputDirectory 'exit-smoke.spool'
$script:ReadLineHistorySpoolDir = $OutputDirectory
$script:ReadLineHistoryJsonlPath = Join-Path $OutputDirectory 'exit-smoke.merged'
$script:LogBuffer = $null
$script:ProfileHistoryWriterOwned = $WriterMode -eq 'owned'
function Close-JsonlAccelerator { [System.IO.File]::WriteAllText($script:SmokeClosePath, 'fixture closed') }
function Merge-JsonlSpoolFile {
    param($SpoolDirectory, $SharedPath, $CurrentSpoolPath, [switch]$IncludeCurrent)
    if ($IncludeCurrent -and $SpoolDirectory -and $CurrentSpoolPath) {
        [System.IO.File]::WriteAllText($SharedPath, 'fixture merged')
    }
}
$tokens = $null; $parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($ProfilePath, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw 'Profile parse failed.' }
foreach ($variableName in @('$profileExitContext', '$script:ProfileExitJob')) {
    $node = $ast.Find({ param($n)
        $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and
        $n.Left.Extent.Text -eq $variableName -and
        ($variableName -ne '$script:ProfileExitJob' -or $n.Right.Extent.Text -match 'Register-EngineEvent')
    }, $true)
    if (-not $node) { throw 'Required exit fixture node is missing.' }
    . ([scriptblock]::Create($node.Extent.Text))
}
# Normal process exit fires the actual PowerShell.Exiting event.

exit
