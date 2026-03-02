function Get-NGrams {
    param([string]$Text, [int]$N = 2)

    $words = $Text -split '\s+'
    $ngrams = @()

    for ($i = 0; $i -le $words.Count - $N; $i++) {
        $ngrams += ($words[$i..($i + $N - 1)] -join ' ')
    }

    return $ngrams
}
