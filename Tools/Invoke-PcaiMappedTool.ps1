#Requires -PSEdition Core

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$RepoRoot,

    [Parameter(Mandatory)]
    [string]$ToolsPath,

    [Parameter(Mandatory)]
    [string]$ToolName,

    [Parameter(Mandatory)]
    [string]$ArgumentsPath
)

$ErrorActionPreference = 'Stop'

function ConvertTo-PlainResult {
    param([Parameter(ValueFromPipeline)]$Value)

    if ($null -eq $Value) {
        return ''
    }

    if ($Value -is [string]) {
        return $Value
    }

    if ($Value -is [System.Collections.IEnumerable] -and -not ($Value -is [hashtable]) -and -not ($Value -is [pscustomobject])) {
        return (($Value | ConvertTo-Json -Depth 8 -Compress) | Out-String).Trim()
    }

    return (($Value | ConvertTo-Json -Depth 8 -Compress) | Out-String).Trim()
}

function ConvertTo-ArgumentTable {
    param([Parameter()]$Value)

    $table = @{}
    if ($null -eq $Value) {
        return $table
    }

    if ($Value -is [hashtable]) {
        foreach ($key in $Value.Keys) { $table[$key] = $Value[$key] }
        return $table
    }

    if ($Value -is [System.Collections.IDictionary]) {
        foreach ($key in $Value.Keys) { $table[$key] = $Value[$key] }
        return $table
    }

    foreach ($property in $Value.PSObject.Properties) {
        $table[$property.Name] = $property.Value
    }

    return $table
}

try {
    $toolCatalog = Get-Content -Path $ToolsPath -Raw -Encoding UTF8 | ConvertFrom-Json -Depth 20
    $toolDefinition = $toolCatalog.tools | Where-Object { $_.function.name -eq $ToolName } | Select-Object -First 1
    if (-not $toolDefinition) {
        throw "Tool '$ToolName' is not defined in $ToolsPath"
    }

    $mapping = $toolDefinition.pcai_mapping
    if (-not $mapping) {
        throw "Tool '$ToolName' does not define a pcai_mapping"
    }

    $argumentsObject = $null
    if (Test-Path -Path $ArgumentsPath) {
        $argumentsObject = Get-Content -Path $ArgumentsPath -Raw -Encoding UTF8 | ConvertFrom-Json -Depth 20
    }
    $argumentTable = ConvertTo-ArgumentTable -Value $argumentsObject

    if ($mapping.module) {
        $modulePath = Join-Path $RepoRoot ("Modules/{0}" -f $mapping.module)
        Import-Module $modulePath -Force -ErrorAction Stop | Out-Null
    }

    if ($mapping.cmdlet -eq 'wsl') {
        $wslArgs = @()
        if ($mapping.args) {
            $wslArgs += @($mapping.args)
        }
        foreach ($key in $argumentTable.Keys) {
            if ($null -ne $argumentTable[$key]) {
                $wslArgs += [string]$argumentTable[$key]
            }
        }

        $wslOutput = & wsl @wslArgs 2>&1 | Out-String
        $payload = @{
            ok     = $true
            result = $wslOutput.Trim()
        }
        $payload | ConvertTo-Json -Depth 8 -Compress
        exit 0
    }

    $boundParams = @{}
    if ($mapping.params) {
        foreach ($property in $mapping.params.PSObject.Properties) {
            $targetName = $property.Name
            $sourceValue = $property.Value
            if ($sourceValue -is [string] -and $sourceValue.StartsWith('$')) {
                $lookup = $sourceValue.Substring(1)
                if ($argumentTable.ContainsKey($lookup)) {
                    $boundParams[$targetName] = $argumentTable[$lookup]
                }
            } else {
                $boundParams[$targetName] = $sourceValue
            }
        }
    }

    $result = & $mapping.cmdlet @boundParams
    @{
        ok     = $true
        result = (ConvertTo-PlainResult -Value $result)
    } | ConvertTo-Json -Depth 8 -Compress
    exit 0
}
catch {
    @{
        ok    = $false
        error = $_.Exception.Message
    } | ConvertTo-Json -Depth 8 -Compress
    exit 1
}
