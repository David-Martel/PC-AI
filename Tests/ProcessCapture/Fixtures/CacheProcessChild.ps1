#Requires -Version 7.0
param(
    [ValidateSet('DualStream', 'Exit23', 'Empty', 'Stall', 'StdoutExact', 'StdoutOver', 'StderrExact', 'StderrOver', 'CancellationStall')]
    [string]$Mode = 'Empty',
    [string]$ReadyEventName = ''
)
# Owned finite child proposal. No provider/network/file/process launches.
$fixtureOut = [Console]::OpenStandardOutput()
$fixtureErr = [Console]::OpenStandardError()
function Write-FixtureBytes {
    param([IO.Stream]$Stream, [long]$Count, [byte]$Value)
    $fixtureChunk = [byte[]]::new(8192)
    [Array]::Fill($fixtureChunk, $Value)
    while ($Count -gt 0) {
        $fixtureTake = [int][Math]::Min($Count, $fixtureChunk.Length)
        $Stream.Write($fixtureChunk, 0, $fixtureTake)
        $Count -= $fixtureTake
    }
    $Stream.Flush()
}
try {
    switch ($Mode) {
        'DualStream' {
            # Protects: complete dual-stream capture above pipe capacity.
            # Detects: producer blockage while parent waits without draining.
            # Needs: actual owned child under original reviewed outer job.
            # Breadcrumb: original Invoke-PcaiProcessCapture WaitForExit/ReadToEnd ordering.
            Write-FixtureBytes $fixtureOut (2 * 1024 * 1024) 65
            Write-FixtureBytes $fixtureErr (512 * 1024) 66
        }
        'Exit23' {
            $fixtureText = [Text.UTF8Encoding]::new($false).GetBytes("  actual-π`r`n")
            $fixtureBom = [byte[]]@(239, 187, 191)
            $fixtureOut.Write($fixtureBom, 0, 3)
            $fixtureOut.Write($fixtureText, 0, $fixtureText.Length)
            $fixtureErr.Write($fixtureText, 0, $fixtureText.Length)
            $fixtureOut.Flush(); $fixtureErr.Flush()
            exit 23
        }
        'Stall' { [Threading.Thread]::Sleep(5000) }
        'CancellationStall' {
            if($ReadyEventName-cnotmatch'^Local\\PcaiCaptureCancellation_[a-f0-9]{32}$'){throw 'Exact owned local readiness event required'}
            $ready=[Threading.EventWaitHandle]::OpenExisting($ReadyEventName)
            try{if(-not$ready.Set()){throw 'Original child readiness signal failed'}}
            finally{$ready.Dispose()}
            [Threading.Thread]::Sleep(5000)
        }
        'StdoutExact' { Write-FixtureBytes $fixtureOut (8 * 1024 * 1024) 65 }
        'StdoutOver' { Write-FixtureBytes $fixtureOut (8 * 1024 * 1024 + 1) 65 }
        'StderrExact' { Write-FixtureBytes $fixtureErr (1024 * 1024) 66 }
        'StderrOver' { Write-FixtureBytes $fixtureErr (1024 * 1024 + 1) 66 }
        'Empty' { }
    }
}
finally { $fixtureOut.Dispose(); $fixtureErr.Dispose() }
