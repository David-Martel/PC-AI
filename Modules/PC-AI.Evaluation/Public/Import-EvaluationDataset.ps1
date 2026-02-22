function Import-EvaluationDataset {
    [CmdletBinding()]
    param([string]$Path)

    $data = Get-Content $Path | ConvertFrom-Json
    return $data | ForEach-Object {
        # Convert PSCustomObject context to hashtable
        $contextHash = @{}
        if ($_.context) {
            $_.context.PSObject.Properties | ForEach-Object {
                $contextHash[$_.Name] = $_.Value
            }
        }

        [EvaluationTestCase]@{
            Id = $_.id
            Category = $_.category
            Prompt = $_.prompt
            ExpectedOutput = $_.expected
            Context = $contextHash
            Tags = @($_.tags)
        }
    }
}
