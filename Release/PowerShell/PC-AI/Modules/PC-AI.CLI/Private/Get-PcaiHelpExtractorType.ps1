function Get-PcaiHelpExtractorType {
    if ($script:HelpExtractorType) { return $script:HelpExtractorType }

    $binPath = Join-Path $script:ProjectRoot 'bin\PcaiNative.dll'
    if (-not (Test-Path $binPath)) { return $null }

    try {
        Add-Type -Path $binPath -ErrorAction Stop | Out-Null
        $type = [PcaiNative.HelpExtractor]
        $script:HelpExtractorType = $type
        return $type
    } catch {
        return $null
    }
}
