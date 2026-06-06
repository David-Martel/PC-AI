$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
$out = Join-Path $root '14a-kb-removability.txt'
"# KB removability check $(Get-Date -Format o)" | Set-Content $out

$rawPath = Join-Path $root '14a-all-packages.txt'
$raw = & dism /online /get-packages /format:list 2>&1 | Out-String
$raw | Set-Content -Path $rawPath

$blocks = $raw -split '(?=Package Identity)'

"`n## Touchpad-window relevant KBs (5/1 batch candidates)" | Add-Content $out
$candidates = @('KB5083631','KB5082417','KB5088467','KB5091378','KB5089614','KB5090315')
foreach ($kb in $candidates) {
    "`n### $kb" | Add-Content $out
    $hit = $blocks | Where-Object { $_ -match $kb }
    if ($hit) {
        ($hit -join "`n") | Add-Content $out
    } else {
        "(not present)" | Add-Content $out
    }
}

"`n## All packages installed in last 7 days (with Permanent flag)" | Add-Content $out
$cutoff = (Get-Date).AddDays(-7)
foreach ($b in $blocks) {
    if ($b -match 'Install Time\s*:\s*(\S.+?)\r?\n') {
        $tStr = $matches[1].Trim()
        try {
            $t = [datetime]::Parse($tStr)
            if ($t -ge $cutoff) {
                $b.Trim() | Add-Content $out
                "---" | Add-Content $out
            }
        } catch { }
    }
}

"`n## Permanent=True count vs Permanent=False count (across all packages)" | Add-Content $out
$permTrue = ($raw | Select-String -Pattern 'Permanent\s*:\s*Yes' -AllMatches).Matches.Count
$permFalse = ($raw | Select-String -Pattern 'Permanent\s*:\s*No' -AllMatches).Matches.Count
"Permanent=Yes: $permTrue" | Add-Content $out
"Permanent=No : $permFalse" | Add-Content $out

Write-Host "Wrote $out"
