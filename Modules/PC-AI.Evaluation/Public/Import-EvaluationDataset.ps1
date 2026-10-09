function Import-EvaluationDataset {
    [CmdletBinding()]
    param([string]$Path)

    $data = @(Get-Content -LiteralPath $Path -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop)
    # Validate the whole dataset before emitting any usable test cases.
    foreach ($record in $data) {
        if ($record.id -isnot [string] -or [string]::IsNullOrWhiteSpace($record.id) -or
            $record.prompt -isnot [string] -or [string]::IsNullOrWhiteSpace($record.prompt)) {
            throw 'Every evaluation dataset record requires a nonempty string id and prompt.'
        }
    }
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
