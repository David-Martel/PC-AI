function Get-FunctionDefinitions {
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    $tokens = $null
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors)
    $functions = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)

    $defs = @()
    foreach ($func in $functions) {
        $parent = $func.Parent
        $nested = $false
        while ($parent) {
            if ($parent -is [System.Management.Automation.Language.FunctionDefinitionAst]) {
                $nested = $true
                break
            }
            $parent = $parent.Parent
        }
        if ($nested) { continue }
        $paramNames = @()
        if ($func.Body -and $func.Body.ParamBlock) {
            foreach ($p in $func.Body.ParamBlock.Parameters) {
                if ($p.Name -and $p.Name.VariablePath) {
                    $paramNames += $p.Name.VariablePath.UserPath
                }
            }
        }
        $defs += [PSCustomObject]@{
            Name = $func.Name
            Parameters = $paramNames
        }
    }
    return $defs
}
