<#
.SYNOPSIS
    Unit tests for tool parameter validation

.DESCRIPTION
    Tests the Confirm-ToolParameters function for validating tool call parameters
    against JSON schemas with type checking, required parameters, and enum validation
#>

BeforeAll {
    # Dot-source the private function directly
    $ModuleRoot = Join-Path $PSScriptRoot '..\..\Modules\PC-AI.LLM'
    . (Join-Path $ModuleRoot 'Private\Validate-ToolParameters.ps1')
}

Describe "Confirm-ToolParameters" -Tag 'Unit', 'Validation', 'Fast', 'Portable' {

    Context "When validating required parameters" {
        BeforeAll {
            $schema = @{
                type = "object"
                properties = @{
                    name = @{ type = "string" }
                    age = @{ type = "integer" }
                }
                required = @("name")
            }
        }

        It "Should pass when all required parameters are present" {
            $params = @{ name = "Alice" }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
            $result.Errors | Should -BeNullOrEmpty
        }

        It "Should fail when required parameter is missing" {
            $params = @{ age = 30 }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $false
            $result.Errors | Should -Contain "Required parameter 'name' is missing"
        }

        It "Should fail when multiple required parameters are missing" {
            $schema.required = @("name", "age")
            $params = @{}
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $false
            $result.Errors.Count | Should -BeGreaterOrEqual 2
        }
    }

    Context "When validating string types" {
        BeforeAll {
            $schema = @{
                type = "object"
                properties = @{
                    name = @{ type = "string" }
                }
            }
        }

        It "Should accept valid string values" {
            $params = @{ name = "Alice" }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }

        It "Should reject non-string values" {
            $params = @{ name = 123 }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $false
            $result.Errors | Should -Match "Parameter 'name' must be of type 'string'"
        }

        It "Should accept empty strings" {
            $params = @{ name = "" }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }
    }

    Context "When validating integer types" {
        BeforeAll {
            $schema = @{
                type = "object"
                properties = @{
                    age = @{ type = "integer" }
                }
            }
        }

        It "Should accept valid integer values" {
            $params = @{ age = 30 }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }

        It "Should reject decimal numbers" {
            $params = @{ age = 30.5 }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $false
            $result.Errors | Should -Match "Parameter 'age' must be of type 'integer'"
        }

        It "Should reject string values" {
            $params = @{ age = "thirty" }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $false
        }
    }

    Context "When validating number types" {
        BeforeAll {
            $schema = @{
                type = "object"
                properties = @{
                    price = @{ type = "number" }
                }
            }
        }

        It "Should accept integer values" {
            $params = @{ price = 100 }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }

        It "Should accept decimal values" {
            $params = @{ price = 99.99 }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }

        It "Should reject string values" {
            $params = @{ price = "expensive" }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $false
            $result.Errors | Should -Match "Parameter 'price' must be of type 'number'"
        }
    }

    Context "When validating boolean types" {
        BeforeAll {
            $schema = @{
                type = "object"
                properties = @{
                    active = @{ type = "boolean" }
                }
            }
        }

        It "Should accept true value" {
            $params = @{ active = $true }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }

        It "Should accept false value" {
            $params = @{ active = $false }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }

        It "Should reject string 'true'" {
            $params = @{ active = "true" }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $false
            $result.Errors | Should -Match "Parameter 'active' must be of type 'boolean'"
        }

        It "Should reject numeric values" {
            $params = @{ active = 1 }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $false
        }
    }

    Context "When validating array types" {
        BeforeAll {
            $schema = @{
                type = "object"
                properties = @{
                    tags = @{ type = "array" }
                }
            }
        }

        It "Should accept array values" {
            $params = @{ tags = @("tag1", "tag2") }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }

        It "Should accept empty arrays" {
            $params = @{ tags = @() }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }

        It "Should reject non-array values" {
            $params = @{ tags = "single-tag" }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $false
            $result.Errors | Should -Match "Parameter 'tags' must be of type 'array'"
        }

        It "Should accept ArrayList as array" {
            $arrayList = New-Object System.Collections.ArrayList
            $arrayList.Add("item1") | Out-Null
            $params = @{ tags = $arrayList }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }
    }

    Context "When validating object types" {
        BeforeAll {
            $schema = @{
                type = "object"
                properties = @{
                    metadata = @{ type = "object" }
                }
            }
        }

        It "Should accept hashtable values" {
            $params = @{ metadata = @{ key = "value" } }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }

        It "Should accept PSCustomObject values" {
            $obj = [PSCustomObject]@{ key = "value" }
            $params = @{ metadata = $obj }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }

        It "Should reject string values" {
            $params = @{ metadata = "not-an-object" }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $false
            $result.Errors | Should -Match "Parameter 'metadata' must be of type 'object'"
        }
    }

    Context "When validating enum values" {
        BeforeAll {
            $schema = @{
                type = "object"
                properties = @{
                    status = @{
                        type = "string"
                        enum = @("active", "inactive", "pending")
                    }
                }
            }
        }

        It "Should accept valid enum values" {
            $params = @{ status = "active" }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }

        It "Should reject invalid enum values" {
            $params = @{ status = "unknown" }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $false
            $result.Errors | Should -Match "Parameter 'status' must be one of: active, inactive, pending"
        }

        It "Should be case-sensitive" {
            $params = @{ status = "ACTIVE" }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $false
        }
    }

    Context "When validating multiple parameters" {
        BeforeAll {
            $schema = @{
                type = "object"
                properties = @{
                    name = @{ type = "string" }
                    age = @{ type = "integer" }
                    active = @{ type = "boolean" }
                    tags = @{ type = "array" }
                }
                required = @("name", "age")
            }
        }

        It "Should validate all parameters and accumulate errors" {
            $params = @{
                age = "not-a-number"
                active = "not-a-boolean"
            }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $false
            $result.Errors.Count | Should -BeGreaterOrEqual 3  # Missing name, wrong age type, wrong active type
        }

        It "Should pass when all parameters are valid" {
            $params = @{
                name = "Alice"
                age = 30
                active = $true
                tags = @("admin", "user")
            }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
            $result.Errors | Should -BeNullOrEmpty
        }
    }

    Context "When schema has no required parameters" {
        BeforeAll {
            $schema = @{
                type = "object"
                properties = @{
                    name = @{ type = "string" }
                }
            }
        }

        It "Should pass with empty parameters" {
            $params = @{}
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }
    }

    Context "When parameters have extra fields not in schema" {
        BeforeAll {
            $schema = @{
                type = "object"
                properties = @{
                    name = @{ type = "string" }
                }
            }
        }

        It "Should ignore extra parameters" {
            $params = @{
                name = "Alice"
                extraField = "extra-value"
            }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }
    }

    Context "When handling edge cases" {
        It "Should handle null parameters gracefully" {
            $schema = @{
                type = "object"
                properties = @{
                    name = @{ type = "string" }
                }
            }
            $params = $null
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }

        It "Should handle null schema gracefully" {
            $params = @{ name = "Alice" }
            $result = Confirm-ToolParameters -Parameters $params -Schema $null

            $result.IsValid | Should -Be $true
        }

        It "Should handle schema without properties" {
            $schema = @{ type = "object" }
            $params = @{ name = "Alice" }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }
    }

    Context "When validating nested structures" {
        BeforeAll {
            $schema = @{
                type = "object"
                properties = @{
                    user = @{
                        type = "object"
                        properties = @{
                            name = @{ type = "string" }
                            age = @{ type = "integer" }
                        }
                        required = @("name")
                    }
                }
                required = @("user")
            }
        }

        It "Should validate required nested object" {
            $params = @{}
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $false
            $result.Errors | Should -Contain "Required parameter 'user' is missing"
        }

        It "Should accept valid nested object" {
            $params = @{
                user = @{
                    name = "Alice"
                    age = 30
                }
            }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.IsValid | Should -Be $true
        }
    }

    Context "When result structure is correct" {
        BeforeAll {
            $schema = @{
                type = "object"
                properties = @{
                    name = @{ type = "string" }
                }
            }
        }

        It "Should return object with IsValid property" {
            $params = @{ name = "Alice" }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.PSObject.Properties.Name | Should -Contain "IsValid"
        }

        It "Should return object with Errors property" {
            $params = @{ name = "Alice" }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            $result.PSObject.Properties.Name | Should -Contain "Errors"
        }

        It "Should return Errors as array type" {
            $params = @{ name = 123 }
            $result = Confirm-ToolParameters -Parameters $params -Schema $schema

            ($result.Errors -is [System.Collections.ArrayList]) | Should -BeTrue
        }
    }
}
