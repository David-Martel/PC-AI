#Requires -Version 7.0
<#
.SYNOPSIS
    Checks PowerShell source syntax without executing files.
.DESCRIPTION
    Parses each literal file path and returns a nonzero exit code for missing
    files or parser errors. Used by the powershell-syntax pre-commit hook.
    See [Pre-Commit Hooks](../CI-CD-GUIDE.md#pre-commit-hooks).
.PARAMETER Path
    Literal PowerShell source paths. Multiple filenames are accepted.
.PARAMETER Help
    Shows usage. Also accepts -h and --help.
.EXAMPLE
    pwsh -NoProfile -File Tools/Test-PowerShellSyntax.ps1 ./PC-AI.ps1
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$Path = @(),
    [Alias('h')]
    [switch]$Help
)

Set-StrictMode -Version Latest

if ($Help -or '--help' -in $Path) {
    Write-Output 'Usage: pwsh -NoProfile -File Tools/Test-PowerShellSyntax.ps1 <file.ps1> [file.psm1 ...]'
    exit 0
}

if ($Path.Count -eq 0) {
    Write-Error 'Supply at least one PowerShell source file, or use --help.' -ErrorAction Continue
    exit 1
}

$hasErrors = $false
foreach ($filePath in $Path) {
    try {
        $file = Get-Item -LiteralPath $filePath -Force -ErrorAction Stop
        if ($file.PSProvider.Name -ne 'FileSystem' -or $file.PSIsContainer) {
            throw "Not a source file: $filePath"
        }
        $parseErrors = $null
        $null = [System.Management.Automation.Language.Parser]::ParseFile(
            $file.FullName, [ref]$null, [ref]$parseErrors
        )
        foreach ($parseError in $parseErrors) {
            $hasErrors = $true
            Write-Error ('{0}:{1}:{2}: {3}' -f $file.FullName,
                $parseError.Extent.StartLineNumber, $parseError.Extent.StartColumnNumber,
                $parseError.Message) -ErrorAction Continue
        }
    }
    catch {
        $hasErrors = $true
        Write-Error -ErrorRecord $_ -ErrorAction Continue
    }
}

if ($hasErrors) {
    exit 1
}
exit 0
