#Requires -Version 5.1

<#
.SYNOPSIS
    Tool parameter validation module for PC-AI.LLM

.DESCRIPTION
    Provides Confirm-ToolParameters function to validate tool call parameters
    against JSON schemas with comprehensive type checking, required parameter
    validation, and enum constraint enforcement.

.NOTES
    Author: PC-AI Development
    Version: 1.0.0
#>

function Confirm-ToolParameters {
    <#
    .SYNOPSIS
        Validates tool call parameters against a JSON schema

    .DESCRIPTION
        Performs comprehensive validation of tool parameters including:
        - Required parameter checking
        - Type validation (string, integer, number, boolean, array, object)
        - Enum value constraints
        - Returns structured validation result with IsValid flag and error list

    .PARAMETER Parameters
        Hashtable of parameters to validate

    .PARAMETER Schema
        JSON schema definition (as hashtable) containing validation rules

    .OUTPUTS
        PSCustomObject with IsValid (boolean) and Errors (ArrayList) properties

    .EXAMPLE
        $schema = @{
            type = "object"
            properties = @{
                name = @{ type = "string" }
                age = @{ type = "integer" }
            }
            required = @("name")
        }
        $params = @{ name = "Alice"; age = 30 }
        $result = Confirm-ToolParameters -Parameters $params -Schema $schema
        # Returns: @{ IsValid = $true; Errors = @() }

    .EXAMPLE
        $schema = @{
            type = "object"
            properties = @{
                status = @{
                    type = "string"
                    enum = @("active", "inactive")
                }
            }
            required = @("status")
        }
        $params = @{ status = "unknown" }
        $result = Confirm-ToolParameters -Parameters $params -Schema $schema
        # Returns: @{ IsValid = $false; Errors = @("Parameter 'status' must be one of: active, inactive") }
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory = $false)]
        [hashtable]$Parameters,

        [Parameter(Mandatory = $false)]
        [hashtable]$Schema
    )

    # Initialize error collection
    $errors = New-Object System.Collections.ArrayList

    # Handle edge cases
    if ($null -eq $Schema -or $Schema.Count -eq 0) {
        return [PSCustomObject]@{
            IsValid = $true
            Errors = $errors
        }
    }

    if ($null -eq $Parameters) {
        $Parameters = @{}
    }

    # Get schema properties and required list safely
    $properties = $null
    $required = $null
    if ($Schema -is [hashtable]) {
        if ($Schema.ContainsKey('properties')) { $properties = $Schema['properties'] }
        if ($Schema.ContainsKey('required')) { $required = $Schema['required'] }
    } else {
        if ($Schema.PSObject.Properties['properties']) { $properties = $Schema.properties }
        if ($Schema.PSObject.Properties['required']) { $required = $Schema.required }
    }

    # Validate required parameters
    if ($required -and $required.Count -gt 0) {
        foreach ($requiredParam in $required) {
            if (-not $Parameters.ContainsKey($requiredParam)) {
                [void]$errors.Add("Required parameter '$requiredParam' is missing")
            }
        }
    }

    # Validate parameter types
    if ($properties -and $properties.Count -gt 0) {
        foreach ($paramName in $Parameters.Keys) {
            # Skip parameters not in schema (allow extra fields)
            if (-not $properties.ContainsKey($paramName)) {
                continue
            }

            $paramValue = $Parameters[$paramName]
            $paramSchema = $properties[$paramName]
            $expectedType = $paramSchema.type

            # Validate type
            $typeValid = Test-ParameterType -Value $paramValue -ExpectedType $expectedType -ParameterName $paramName -Errors $errors

            # Validate enum if present and type is valid
            $enumValues = $null
            if ($paramSchema -is [hashtable]) {
                if ($paramSchema.ContainsKey('enum')) { $enumValues = $paramSchema['enum'] }
            } elseif ($paramSchema.PSObject.Properties['enum']) {
                $enumValues = $paramSchema.enum
            }

            if ($typeValid -and $enumValues) {
                Test-EnumValue -Value $paramValue -AllowedValues $enumValues -ParameterName $paramName -Errors $errors | Out-Null
            }
        }
    }

    # Return validation result
    return [PSCustomObject]@{
        IsValid = ($errors.Count -eq 0)
        Errors = $errors
    }
}

function Test-ParameterType {
    <#
    .SYNOPSIS
        Tests if a parameter value matches the expected type

    .DESCRIPTION
        Internal helper function for type validation
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory = $false)]
        $Value,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedType,

        [Parameter(Mandatory = $true)]
        [string]$ParameterName,

        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [System.Collections.ArrayList]$Errors
    )

    $isValid = $true

    switch ($ExpectedType) {
        'string' {
            if ($Value -isnot [string]) {
                [void]$Errors.Add("Parameter '$ParameterName' must be of type 'string'")
                $isValid = $false
            }
        }

        'integer' {
            # Integer must be a whole number (int32, int64) or a double/decimal that's a whole number
            if ($Value -is [int] -or $Value -is [int64] -or $Value -is [int32]) {
                # Valid integer type
            } elseif ($Value -is [double] -or $Value -is [decimal]) {
                # Check if it's a whole number
                if ($Value -ne [Math]::Floor($Value)) {
                    [void]$Errors.Add("Parameter '$ParameterName' must be of type 'integer'")
                    $isValid = $false
                }
            } else {
                [void]$Errors.Add("Parameter '$ParameterName' must be of type 'integer'")
                $isValid = $false
            }
        }

        'number' {
            # Number can be integer or decimal
            if ($Value -isnot [int] -and $Value -isnot [int32] -and $Value -isnot [int64] -and
                $Value -isnot [double] -and $Value -isnot [decimal] -and $Value -isnot [float]) {
                [void]$Errors.Add("Parameter '$ParameterName' must be of type 'number'")
                $isValid = $false
            }
        }

        'boolean' {
            if ($Value -isnot [bool]) {
                [void]$Errors.Add("Parameter '$ParameterName' must be of type 'boolean'")
                $isValid = $false
            }
        }

        'array' {
            # Arrays can be array, ArrayList, or other collection types
            if ($Value -isnot [array] -and $Value -isnot [System.Collections.ArrayList] -and
                $Value -isnot [System.Collections.IEnumerable]) {
                [void]$Errors.Add("Parameter '$ParameterName' must be of type 'array'")
                $isValid = $false
            } elseif ($Value -is [string] -or $Value -is [hashtable]) {
                # Strings and hashtables implement IEnumerable but shouldn't be treated as arrays
                [void]$Errors.Add("Parameter '$ParameterName' must be of type 'array'")
                $isValid = $false
            }
        }

        'object' {
            # Objects can be hashtable or PSCustomObject
            if ($Value -isnot [hashtable] -and $Value.GetType().Name -ne 'PSCustomObject') {
                [void]$Errors.Add("Parameter '$ParameterName' must be of type 'object'")
                $isValid = $false
            }
        }

        default {
            Write-Warning "Unknown type '$ExpectedType' for parameter '$ParameterName'"
        }
    }

    return $isValid
}

function Test-EnumValue {
    <#
    .SYNOPSIS
        Tests if a value is in the allowed enum list

    .DESCRIPTION
        Internal helper function for enum validation
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory = $false)]
        $Value,

        [Parameter(Mandatory = $true)]
        [array]$AllowedValues,

        [Parameter(Mandatory = $true)]
        [string]$ParameterName,

        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [System.Collections.ArrayList]$Errors
    )

    # Case-sensitive comparison
    if ($AllowedValues -cnotcontains $Value) {
        $allowedList = $AllowedValues -join ', '
        [void]$Errors.Add("Parameter '$ParameterName' must be one of: $allowedList")
        return $false
    }

    return $true
}
