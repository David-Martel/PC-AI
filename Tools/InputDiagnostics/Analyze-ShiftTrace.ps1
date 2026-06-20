#Requires -Version 7.0
# Analyzes a shift-source-live-*.jsonl capture: reconstructs typed text with raw shift
# state, and flags every letter whose shift state disagrees with normal typing intent.
[CmdletBinding()]
param([string]$Path)

if (-not $Path) {
    $Path = (Get-ChildItem 'C:\codedev\PC_AI\Logs\input-diagnostics\shift-source-live-*.jsonl' |
             Sort-Object LastWriteTime | Select-Object -Last 1).FullName
}
$ev = Get-Content $Path | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json }
"File: $Path"
"Total events: $($ev.Count)"
"By class: " + (($ev | Group-Object cls | ForEach-Object { '{0}={1}' -f $_.Name, $_.Count }) -join ', ')
"By device: " + (($ev | Group-Object dev | ForEach-Object { '{0}x[{1}]' -f $_.Count, $_.Group[0].cls }) -join ', ')

function VkChar([int]$vk) {
    if ($vk -ge 65 -and $vk -le 90) { return [string][char]$vk }
    if ($vk -ge 48 -and $vk -le 57) { return [string][char]$vk }
    switch ($vk) { 32 { '_' } 190 { '.' } 188 { ',' } 186 { ';' } 222 { "'" } default { "<$vk>" } }
}

# Walk in order, track how many shift keys are physically down (raw), reconstruct output.
$held = 0
$sb = [System.Text.StringBuilder]::new()
$timeline = [System.Collections.Generic.List[object]]::new()
foreach ($e in $ev) {
    if ($e.name -in 'LSHIFT', 'RSHIFT') {
        if ($e.dir -eq 'DOWN') { $held++ } elseif ($e.dir -eq 'UP') { $held = [Math]::Max(0, $held - 1) }
        $timeline.Add([pscustomobject]@{ t = $e.t; tok = "[$($e.name) $($e.dir) held=$held]"; ch = '' })
        continue
    }
    if ($e.dir -ne 'DOWN') { continue }
    $raw = VkChar([int]$e.vk)
    $ch = $raw
    if ($raw -match '^[a-zA-Z]$') { $ch = if ($held -gt 0) { $raw.ToUpper() } else { $raw.ToLower() } }
    [void]$sb.Append($ch)
    $timeline.Add([pscustomobject]@{ t = $e.t; tok = "$ch (vk=$($e.vk) shiftHeld=$held)"; ch = $ch })
}

""
"=== RECONSTRUCTED OUTPUT (what the RAW scancodes would produce; _ = space) ==="
$sb.ToString()

""
"=== SHIFT SUMMARY ==="
$sh = $ev | Where-Object { $_.name -in 'LSHIFT', 'RSHIFT' }
"Internal LSHIFT down: " + (($sh | Where-Object { $_.name -eq 'LSHIFT' -and $_.dir -eq 'DOWN' }).Count)
"Internal LSHIFT up  : " + (($sh | Where-Object { $_.name -eq 'LSHIFT' -and $_.dir -eq 'UP' }).Count)
"Internal RSHIFT down: " + (($sh | Where-Object { $_.name -eq 'RSHIFT' -and $_.dir -eq 'DOWN' }).Count)
"Internal RSHIFT up  : " + (($sh | Where-Object { $_.name -eq 'RSHIFT' -and $_.dir -eq 'UP' }).Count)

# Detect asymmetry: a DOWN with no matching UP (or vice versa) = stuck/lost half-stroke.
$dn = ($sh | Where-Object dir -eq 'DOWN').Count
$up = ($sh | Where-Object dir -eq 'UP').Count
"Shift DOWN total=$dn  UP total=$up  (mismatch => a make or break was dropped by the keyboard)"

""
"=== TRIPLE-SPACE FAILURE MARKERS (context = 12 events before each) ==="
$arr = $timeline
for ($i = 0; $i -lt $arr.Count; $i++) {
    # find runs of 3+ consecutive spaces
    if ($arr[$i].ch -eq '_' -and $i -ge 2 -and $arr[$i-1].ch -eq '_' -and $arr[$i-2].ch -eq '_') {
        $start = [Math]::Max(0, $i - 14)
        "--- marker at $($arr[$i].t) ---"
        ($arr[$start..$i] | ForEach-Object { '{0}  {1}' -f $_.t, $_.tok }) -join "`n"
        ""
    }
}
