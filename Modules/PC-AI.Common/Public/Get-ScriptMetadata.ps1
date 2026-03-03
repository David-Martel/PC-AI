function Get-ScriptMetadata {
	<#
    .SYNOPSIS
        Extracts script and function metadata using native PowerShell AST parsing.
    .DESCRIPTION
        Returns a rich object containing file-level Synopsis/Description and details about all top-level functions such as their names, synopses, defined parameters, and documented parameters. Replaces brittle regex routines.
    #>
	[CmdletBinding()]
	param(
		[Parameter(Mandatory = $true, ValueFromPipeline = $true)]
		[ValidateScript({ Test-Path $_ })]
		[string]$Path
	)
	process {
		$resolved = Resolve-Path $Path
		$tokens = $null
		$errors = $null
		$ast = [System.Management.Automation.Language.Parser]::ParseFile($resolved.Path, [ref]$tokens, [ref]$errors)

		$scriptHelp = $ast.GetHelpContent()

		$functions = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)
		$funcList = @()

		$commonParams = @(
			'WhatIf', 'Confirm', 'Verbose', 'Debug', 'ErrorAction', 'WarningAction',
			'InformationAction', 'ErrorVariable', 'WarningVariable', 'InformationVariable',
			'OutVariable', 'OutBuffer', 'PipelineVariable', 'ProgressAction'
		)

		foreach ($func in $functions) {
			# Skip nested functions
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

			$fHelp = $func.GetHelpContent()
			$fHelpParams = @()
			if ($fHelp -and $fHelp.Parameters) {
				$fHelpParams = @($fHelp.Parameters.Keys)
			}

			$paramNames = @()
			if ($func.Body -and $func.Body.ParamBlock) {
				foreach ($p in $func.Body.ParamBlock.Parameters) {
					if ($p.Name -and $p.Name.VariablePath) {
						$paramNames += $p.Name.VariablePath.UserPath
					}
				}
			}

			$missingHelp = @($paramNames | Where-Object { $fHelpParams -notcontains $_ })
			$extraHelp = @($fHelpParams | Where-Object { ($paramNames -notcontains $_) -and ($commonParams -notcontains $_) })

			$funcList += [PSCustomObject]@{
				Name                  = $func.Name
				Synopsis              = if ($fHelp -and $fHelp.Synopsis) { $fHelp.Synopsis.Trim() } else { '' }
				Description           = if ($fHelp -and $fHelp.Description) { $fHelp.Description.Trim() } else { '' }
				Parameters            = $paramNames
				HelpPresent           = [bool]($fHelp -and ($fHelp.Synopsis -or $fHelp.Description -or $fHelpParams.Count -gt 0))
				HelpParameters        = $fHelpParams
				MissingHelpParameters = $missingHelp
				ExtraHelpParameters   = $extraHelp
				SourcePath            = $resolved.Path
			}
		}

		return [PSCustomObject]@{
			Path        = $resolved.Path
			Name        = Split-Path $resolved.Path -Leaf
			Synopsis    = if ($scriptHelp -and $scriptHelp.Synopsis) { $scriptHelp.Synopsis.Trim() } else { '' }
			Description = if ($scriptHelp -and $scriptHelp.Description) { $scriptHelp.Description.Trim() } else { '' }
			Functions   = $funcList
		}
	}
}
