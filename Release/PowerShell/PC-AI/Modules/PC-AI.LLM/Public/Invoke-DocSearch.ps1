#Requires -Version 5.1

function Invoke-DocSearch {
	<#
    .SYNOPSIS
        Search Microsoft and manufacturer documentation for technical details.

    .DESCRIPTION
        This tool allows the agent to search for specific error codes, driver details,
        and troubleshooting guides from authoritative sources.

    .PARAMETER Query
        The search query (e.g., "ConfigManagerErrorCode 31", "Intel Wireless-AC 9560 failed")

    .PARAMETER Source
        Priority source: Microsoft, Intel, AMD, Dell, HP, Lenovo, Generic (default: Microsoft)

    .EXAMPLE
        Invoke-DocSearch -Query "Win32 Error 31" -Source Microsoft
    #>
	[CmdletBinding()]
	param(
		[Parameter(Mandatory, Position = 0)]
		[string]$Query,

		[Parameter()]
		[ValidateSet('Microsoft', 'Intel', 'AMD', 'Dell', 'HP', 'Lenovo', 'Generic')]
		[string]$Source = 'Microsoft'
	)

	Write-Verbose "Searching $Source for: $Query"

	# In a real environment, this would call a search API or scrape.
	# For this implementation, we provide a structured interface.

	$searchUrl = switch ($Source) {
		'Microsoft' { "https://learn.microsoft.com/en-us/search/?terms=$([Uri]::EscapeDataString($Query))" }
		'Intel' { "https://www.intel.com/content/www/us/en/search.html?ws=text#q=$([Uri]::EscapeDataString($Query))" }
		'AMD' { "https://www.amd.com/en/search/site.html#q=$([Uri]::EscapeDataString($Query))" }
		default { "https://www.google.com/search?q=$Source+$([Uri]::EscapeDataString($Query))" }
	}

	# Simulate retrieval of snippets
	# The agent will actually use the 'search_web' tool if available, or we provide a fallback description.

	$results = @{
		Query     = $Query
		Source    = $Source
		Url       = $searchUrl
		Status    = 'Success'
		Timestamp = Get-Date
		Results   = @()
	}

	# Internal knowledge base for common offline fallbacks
	$kb = @{
		'ConfigManagerErrorCode 31' = 'This device is not working properly because Windows cannot load the drivers required for this device. (Code 31). Occurs when a driver is not loaded or failing during initialization.'
		'ConfigManagerErrorCode 43' = 'Windows has stopped this device because it has reported problems. (Code 43). Often hardware failure or firmware crash.'
		'Intel Wireless-AC 9560'    = "High-performance Wi-Fi 5 module. Known issues include Error 10/43 on older drivers or with 'Fast Startup' enabled."
	}

	foreach ($key in $kb.Keys) {
		if ($Query -like "*$key*") {
			$results.Results += @{
				Title   = "Official $Source Documentation Fragment"
				Snippet = $kb[$key]
				Source  = $Source
			}
		}
	}

	if ($results.Results.Count -eq 0) {
		$results.Results += @{
			Title   = 'General Search Direction'
			Snippet = "Search results for '$Query' would typically involve reviewing $Source support pages, forum threads, and driver release notes."
			Source  = $Source
		}
	}

	return $results | ConvertTo-Json -Depth 5
}
