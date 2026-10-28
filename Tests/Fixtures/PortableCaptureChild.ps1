#Requires -Version 7.0
param([ValidateSet('Both','Sleep','Nonzero','Overflow','HeldParent','TreeParent','Hold')][string]$Mode='Both',[string]$PidPath)
$ErrorActionPreference='Stop'
switch($Mode){
    'Both' {
        $bytes=[Text.Encoding]::ASCII.GetBytes(('o'*262144))
        [Console]::OpenStandardOutput().Write($bytes,0,$bytes.Length)
        $bytes=[Text.Encoding]::ASCII.GetBytes(('e'*262144))
        [Console]::OpenStandardError().Write($bytes,0,$bytes.Length)
    }
    'Sleep' { [Console]::Out.Write('started'); [Console]::Out.Flush(); [Threading.Thread]::Sleep(30000) }
    'Nonzero' { [Console]::Out.Write('owned-out'); [Console]::Error.Write('owned-error'); exit 23 }
    'Overflow' {
        $bytes=[Text.Encoding]::ASCII.GetBytes(('x'*65536))
        for($i=0;$i -lt 160;$i++){[Console]::OpenStandardOutput().Write($bytes,0,$bytes.Length)}
    }
    'Hold' { [Threading.Thread]::Sleep(30000) }
    { $_ -in 'HeldParent','TreeParent' } {
        $start=[Diagnostics.ProcessStartInfo]::new((Get-Process -Id $PID).Path)
        $start.UseShellExecute=$false
        $start.CreateNoWindow=$true
        foreach($argument in @('-NoLogo','-NoProfile','-File',$PSCommandPath,'-Mode','Hold')){[void]$start.ArgumentList.Add($argument)}
        $child=[Diagnostics.Process]::Start($start)
        try {
            $identity=@{Pid=$child.Id;StartUtcTicks=$child.StartTime.ToUniversalTime().Ticks}
            if($IsLinux){
                # Raw kernel ticks avoid the per-process boot-time conversion in .NET StartTime.
                $stat=[IO.File]::ReadAllText("/proc/$($child.Id)/stat")
                $fields=$stat.Substring($stat.LastIndexOf(')')+2).Split(' ',[StringSplitOptions]::RemoveEmptyEntries)
                $identity.LinuxStartTicks=$fields[19]
                $identity.LinuxBootId=[IO.File]::ReadAllText('/proc/sys/kernel/random/boot_id').Trim()
            }
            $identity|ConvertTo-Json -Compress|Set-Content -LiteralPath $PidPath
        }finally{$child.Dispose()}
        [Console]::Out.Write('spawned');[Console]::Out.Flush()
        if($Mode -eq 'TreeParent'){[Threading.Thread]::Sleep(30000)}
    }
}
