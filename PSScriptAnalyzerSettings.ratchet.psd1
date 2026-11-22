# PSScriptAnalyzer ruleset for the static-quality ratchet (Tools/Quality/Invoke-QualityRatchet.ps1).
#
# PSScriptAnalyzerSettings.psd1 (the repo gate, run by Build.ps1 -LintProfile powershell) excludes
# PSAvoidUsingEmptyCatchBlock and never runs PSAvoidOverwritingBuiltInCmdlets (its `Rules` entry is not
# in IncludeRules), so those findings were invisible. This file re-enables them for the ratchet only,
# where they are counted against a baseline rather than failing the build.
#
# Only AST-based rules are listed so the pass takes about a second per hundred files; the repo gate's
# rules need command resolution and take 8-12 minutes over the tree. The ratchet therefore runs this
# ruleset alone by default and runs the repo gate's rules too with -IncludeRepoSettings.
@{
    Severity     = @('Error', 'Warning', 'Information')

    IncludeRules = @(
        'PSAvoidUsingEmptyCatchBlock',
        'PSAvoidOverwritingBuiltInCmdlets'
    )

    # No Rules block: PSAvoidOverwritingBuiltInCmdlets rejects PowerShellVersion values like '5.1' (it wants
    # profile names such as 'core-6.1.0-windows'), and its default profile is what the ratchet wants.
}
