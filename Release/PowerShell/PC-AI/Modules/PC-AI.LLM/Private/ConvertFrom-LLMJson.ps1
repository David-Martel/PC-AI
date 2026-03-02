#Requires -Version 5.1

<#
.SYNOPSIS
    Extracts and parses JSON from LLM responses, prioritizing native performance.
#>
function ConvertFrom-LLMJson {
    [CmdletBinding()]
    param(
        [Parameter(ValueFromPipeline)]
        [AllowEmptyString()]
        [string]$Content,

        [Parameter()]
        [switch]$Strict
    )

    process {
        if ([string]::IsNullOrWhiteSpace($Content)) { return $null }

        # Size guard: 5MB limit to prevent OOM
        if ($Content.Length -gt 5MB) {
            if ($Strict) { throw "Content exceeded 5MB limit" }
            Write-Error "Content too large for JSON extraction ($($Content.Length) bytes)"
            return $Content
        }

        $jsonStr = $null
        $nativeCoreType = ([System.Management.Automation.PSTypeName]'PcaiNative.PcaiCore').Type
        $nativeAvailable = ($nativeCoreType -and [PcaiNative.PcaiCore]::IsAvailable)

        # 1. Try Native Extraction (Highest Performance)
        if ($nativeAvailable) {
            try {
                $jsonStr = [PcaiNative.PcaiCore]::ExtractJson($Content)
                if ($jsonStr) { Write-Verbose 'Natively extracted JSON' }
            } catch {
                Write-Verbose "Native JSON extraction failed: $_"
            }
        }

        # 2. PowerShell Fallback (Regex-based)
        if (-not $jsonStr) {
            if ($Content -match '(?s)```json\s*(?<json>.*?)\s*```') {
                $jsonStr = $Matches['json']
            } elseif ($Content -match '(?s)\{.*\}|\[.*\]') {
                # Attempt to find common JSON boundaries if not in markdown block
                $jsonStr = $Content.Trim()
            } else {
                $jsonStr = $Content.Trim()
            }
        }

        if (-not $jsonStr) { return $null }

        # 3. Parse JSON
        try {
            # Final validation check via native if available (optional but good for strict mode)
            if ($Strict -and $nativeAvailable) {
                if (-not [PcaiNative.PcaiCore]::IsValidJson($jsonStr)) {
                    throw 'Native validation failed: String is not valid JSON'
                }
            }

            return $jsonStr | ConvertFrom-Json
        } catch {
            if ($Strict) { throw "Failed to parse JSON: $_" }
            Write-Warning "JSON parsing failed (length: $($jsonStr.Length)). Returning raw content."
            return $Content
        }
    }
}
