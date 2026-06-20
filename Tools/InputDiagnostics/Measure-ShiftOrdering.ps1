#Requires -Version 7.0
# Per-device Shift/letter ORDERING metric. For every letter DOWN that the user shifted
# (a shift key was down within +/-250ms), compute delta = letterDown - shiftDown (ms).
# Negative = letter registered BEFORE shift (the late-shift failure signature).
[CmdletBinding()]
param([string]$Path)
if (-not $Path) {
    $Path = (Get-ChildItem 'C:\codedev\PC_AI\Logs\input-diagnostics\shift-source-live-*.jsonl' |
             Sort-Object LastWriteTime | Select-Object -Last 1).FullName
}
$ev = @(Get-Content $Path | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
"File: $([System.IO.Path]::GetFileName($Path))   events=$($ev.Count)"
$t = {param($s) [datetime]::ParseExact($s,'HH:mm:ss.fff',$null)}

foreach ($cls in @('INTERNAL','USB/HID')) {
    $d = @($ev | Where-Object cls -eq $cls)
    if (-not $d.Count) { "`n[$cls] no events"; continue }
    $shDowns = @($d | Where-Object { $_.name -in 'LSHIFT','RSHIFT' -and $_.dir -eq 'DOWN' })
    $letters = @($d | Where-Object { $_.dir -eq 'DOWN' -and $_.vk -ge 65 -and $_.vk -le 90 })
    "`n========== [$cls] ($($d.Count) events; $($shDowns.Count) shift-downs, $($letters.Count) letters) =========="
    $fails = 0; $caps = 0
    foreach ($L in $letters) {
        $lt = & $t $L.t
        # nearest shift-down within 300ms either side
        $near = $shDowns | ForEach-Object { [pscustomobject]@{ s=$_; d=([int]((& $t $_.t)-$lt).TotalMilliseconds) } } |
                Where-Object { [Math]::Abs($_.d) -le 300 } | Sort-Object { [Math]::Abs($_.d) } | Select-Object -First 1
        if ($near) {
            $caps++
            # delta = letterDown - shiftDown ; positive means shift was earlier (good)
            $delta = -1 * $near.d
            $flag = if ($delta -lt 0) { 'FAIL (letter before shift)' } else { 'ok' }
            if ($delta -lt 0) { $fails++ }
            "  {0}  letter '{1}'  shift{2:+0;-0}ms  {3}" -f $L.t, [char][int]$L.vk, $delta, $flag
        }
    }
    "  --> shifted letters=$caps  late-shift FAILURES=$fails"
}
