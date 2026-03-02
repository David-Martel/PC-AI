function Get-HelpEntriesFromFiles {
    param(
        [Parameter(Mandatory)]
        [string[]]$Paths
    )

    $entries = @()
    $pattern = '(?s)<#(.*?)#>\s*function\s+([A-Za-z0-9_-]+)'
    foreach ($path in $Paths) {
        if (-not (Test-Path $path)) { continue }
        $defs = Get-FunctionDefinitions -Path $path
        $helpByName = @{}

        $extractor = Get-PcaiHelpExtractorType
        if ($extractor) {
            try {
                $nativeEntries = $extractor::ExtractFromFile($path)
                foreach ($nativeEntry in $nativeEntries) {
                    $nativeParamHelp = @{}
                    $hasParamProperty = $nativeEntry.PSObject.Properties.Name -contains 'Parameters'
                    if ($hasParamProperty -and $nativeEntry.Parameters) {
                        foreach ($key in $nativeEntry.Parameters.Keys) {
                            $nativeParamHelp[$key] = $nativeEntry.Parameters[$key]
                        }
                    }

                    $nativeExamples = @()
                    $hasExamplesProperty = $nativeEntry.PSObject.Properties.Name -contains 'Examples'
                    if ($hasExamplesProperty -and $nativeEntry.Examples) {
                        foreach ($example in $nativeEntry.Examples) {
                            if ($example) {
                                $nativeExamples += $example
                            }
                        }
                    }

                    $helpByName[$nativeEntry.Name] = [PSCustomObject]@{
                        Name = $nativeEntry.Name
                        Synopsis = $nativeEntry.Synopsis
                        Description = $nativeEntry.Description
                        SourcePath = $nativeEntry.SourcePath
                        ParameterHelp = $nativeParamHelp
                        Examples = $nativeExamples
                    }
                }
            } catch {
                # ignore and fall back to regex
            }
        }

        $content = Get-Content -Path $path -Raw -Encoding UTF8
        $matches = [regex]::Matches($content, $pattern)
        foreach ($match in $matches) {
            $helpBlock = $match.Groups[1].Value
            $funcName = $match.Groups[2].Value
            $entry = Convert-HelpBlockToEntry -HelpBlock $helpBlock -FunctionName $funcName -SourcePath $path
            if (-not $helpByName.ContainsKey($funcName)) {
                $helpByName[$funcName] = $entry
                continue
            }

            $existing = $helpByName[$funcName]
            if (-not $existing.Synopsis -and $entry.Synopsis) {
                $existing.Synopsis = $entry.Synopsis
            }
            if (-not $existing.Description -and $entry.Description) {
                $existing.Description = $entry.Description
            }
            if ($entry.ParameterHelp -and $entry.ParameterHelp.Count -gt 0) {
                foreach ($key in $entry.ParameterHelp.Keys) {
                    if (-not $existing.ParameterHelp.ContainsKey($key)) {
                        $existing.ParameterHelp[$key] = $entry.ParameterHelp[$key]
                    }
                }
            }
            if ($entry.Examples -and $entry.Examples.Count -gt 0 -and (-not $existing.Examples -or $existing.Examples.Count -eq 0)) {
                $existing.Examples = $entry.Examples
            }
        }

        foreach ($def in $defs) {
            $entry = $null
            if ($helpByName.ContainsKey($def.Name)) {
                $entry = $helpByName[$def.Name]
            } else {
                $entry = [PSCustomObject]@{
                    Name = $def.Name
                    Synopsis = ''
                    Description = ''
                    SourcePath = $path
                    ParameterHelp = @{}
                }
            }

            $entries += [PSCustomObject]@{
                Name = $entry.Name
                Synopsis = $entry.Synopsis
                Description = $entry.Description
                SourcePath = $entry.SourcePath
                Parameters = $def.Parameters
                ParameterHelp = $entry.ParameterHelp
                Examples = $entry.Examples
            }
        }
    }
    return $entries
}
