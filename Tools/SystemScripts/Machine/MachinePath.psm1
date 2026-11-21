#Requires -Version 7.0
<#
.SYNOPSIS
    PATH and binary-location analysis for the Windows fleet.

.DESCRIPTION
    Pure analysis functions (no registry or disk access unless a function says so) plus the thin
    registry and filesystem readers they are fed from. Test-MachinePath.ps1 reports, and
    Repair-MachinePath.ps1 applies, the policy in machine-path.policy.json.

    Why this exists: Claude Code sessions run with USERPROFILE and HOME redirected to an agent home,
    so any PATH entry or wrapper written as %USERPROFILE%\... resolves to the wrong tree there. Every
    function here therefore takes the real profile from the registry (ProfileList), never from
    $env:USERPROFILE or $HOME.

    The cost model measured on dtm-p1gen9: probing 60 PATH directories takes about 25 ms. A missing
    command in PowerShell costs 400 to 800 ms because of module auto-discovery over PSModulePath,
    not because of PATH. Resolve-MachineCommand answers "is this on PATH" in milliseconds.

    See Docs/MACHINE_PATH_POLICY.md for the policy.
#>

Set-StrictMode -Version Latest

function New-MachinePathFinding {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][ValidateSet('Error', 'Warn', 'Info')][string]$Severity,
        [Parameter(Mandatory)][string]$Code,
        [string]$Scope = '',
        [string]$Subject = '',
        [Parameter(Mandatory)][string]$Message
    )
    [pscustomobject]@{ Severity = $Severity; Code = $Code; Scope = $Scope; Subject = $Subject; Message = $Message }
}

function Get-MachinePathContext {
    <#
    .SYNOPSIS
        Real profile, CARGO_HOME and LOCALAPPDATA for the current user, independent of any agent-home redirection.
    #>
    [CmdletBinding()]
    param([string]$UserHome, [string]$CargoHome)

    if (-not $UserHome) {
        $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
        $key = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList\$sid"
        $UserHome = (Get-ItemProperty -LiteralPath $key -Name ProfileImagePath -ErrorAction Stop).ProfileImagePath
    }
    $UserHome = [Environment]::ExpandEnvironmentVariables($UserHome).TrimEnd('\')
    if (-not $CargoHome) {
        $CargoHome = [Environment]::GetEnvironmentVariable('CARGO_HOME', 'User')
        if (-not $CargoHome) { $CargoHome = [Environment]::GetEnvironmentVariable('CARGO_HOME', 'Machine') }
        if (-not $CargoHome) { $CargoHome = Join-Path $UserHome '.cargo' }
    }
    [pscustomobject]@{
        UserHome     = $UserHome
        CargoHome    = $CargoHome.TrimEnd('\')
        LocalAppData = Join-Path $UserHome 'AppData\Local'
        SystemRoot   = $env:SystemRoot
    }
}

function Expand-MachinePathToken {
    [CmdletBinding()]
    param([Parameter(Mandatory)][AllowEmptyString()][string]$Text, [Parameter(Mandatory)]$Context)
    $Text.Replace('{UserHome}', $Context.UserHome).Replace('{CargoHome}', $Context.CargoHome).
        Replace('{LocalAppData}', $Context.LocalAppData).Replace('{SystemRoot}', $Context.SystemRoot)
}

function Read-MachinePathPolicy {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)]$Context, [string]$HostName = $env:COMPUTERNAME)
    $raw = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    # hostOverrides.<COMPUTERNAME>: keepOrder drops the head block (the host's own order is deliberate) and machineAdditions lists
    # directories that must also be on the MACHINE PATH, for hosts whose services and sshd sessions never see the user PATH.
    $override = $null
    if ($raw.PSObject.Properties['hostOverrides'] -and $HostName -and $raw.hostOverrides.PSObject.Properties[$HostName.ToUpperInvariant()]) {
        $override = $raw.hostOverrides.PSObject.Properties[$HostName.ToUpperInvariant()].Value
    }
    $expandList = { param($list) @($list | ForEach-Object { Expand-MachinePathToken -Text $_ -Context $Context }) }
    $reasons = @{}
    if ($raw.PSObject.Properties['forbiddenReasons']) {
        foreach ($p in $raw.forbiddenReasons.PSObject.Properties) { $reasons[(Expand-MachinePathToken -Text $p.Name -Context $Context)] = $p.Value }
    }
    $shadows = @{}
    foreach ($p in $raw.intentionalShadows.PSObject.Properties) { $shadows[$p.Name.ToLowerInvariant()] = $p.Value }
    [pscustomobject]@{
        UserPathHead        = $(if ($override -and $override.PSObject.Properties['keepOrder'] -and $override.keepOrder) { @() } else { & $expandList $raw.userPathHead })
        MachineAdditions    = $(if ($override -and $override.PSObject.Properties['machineAdditions']) { & $expandList $override.machineAdditions } else { @() })
        HostOverrideReason  = $(if ($override -and $override.PSObject.Properties['reason']) { [string]$override.reason } else { '' })
        RequiredUserEntries = & $expandList $raw.requiredUserEntries
        # When CARGO_HOME is the default ~/.cargo, its bin directory is the canonical one and must not be forbidden.
        ForbiddenEntries    = @(& $expandList $raw.forbiddenEntries | Where-Object { $_.TrimEnd('\') -ine (Join-Path $Context.CargoHome 'bin') })
        ForbiddenReasons    = $reasons
        ForbiddenPatterns   = @($(if ($raw.PSObject.Properties['forbiddenPatterns']) { $raw.forbiddenPatterns }))
        ForbiddenPatternReason = $(if ($raw.PSObject.Properties['forbiddenPatternReason']) { [string]$raw.forbiddenPatternReason } else { 'Matches a forbidden pattern.' })
        RequiredUserEnv     = $(if ($raw.PSObject.Properties['requiredUserEnv']) { $raw.requiredUserEnv } else { [pscustomobject]@{} })
        ManagedBinDirs      = & $expandList $raw.managedBinDirs
        LengthWarn          = [int]$raw.lengthWarn
        LengthError         = [int]$raw.lengthError
        RequiredPathExt     = @($raw.requiredPathExt)
        IntentionalShadows  = $shadows
        BaselineCommands    = @($raw.baselineCommands)
    }
}

function Get-MachinePathRegistry {
    <#
    .SYNOPSIS
        Raw (unexpanded) PATH value and registry value kind for one scope.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][ValidateSet('User', 'Machine')][string]$Scope)
    $subKey = if ($Scope -eq 'User') { 'Environment' } else { 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment' }
    $root = if ($Scope -eq 'User') { [Microsoft.Win32.Registry]::CurrentUser } else { [Microsoft.Win32.Registry]::LocalMachine }
    $key = $root.OpenSubKey($subKey)
    try {
        $raw = [string]$key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        $kind = $key.GetValueKind('Path').ToString()
    } finally { $key.Close() }
    [pscustomobject]@{ Scope = $Scope; Raw = $raw; Kind = $kind; Entries = @($raw -split ';' | Where-Object { $_ -ne '' }) }
}

function ConvertTo-MachineLongPath {
    <#
    .SYNOPSIS
        Expands 8.3 short names (C:\PROGRA~1) so two spellings of one directory compare equal. Unchanged if the path does not exist.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)
    if ($Path -notmatch '~\d') { return $Path }
    if (-not ('MachinePath.LongPath' -as [type])) {
        Add-Type -Namespace MachinePath -Name LongPath -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode, SetLastError = true)]
public static extern uint GetLongPathName(string lpszShortPath, System.Text.StringBuilder lpszLongPath, uint cchBuffer);
'@
    }
    $sb = [System.Text.StringBuilder]::new(1024)
    $n = [MachinePath.LongPath]::GetLongPathName($Path, $sb, 1024)
    if ($n -gt 0 -and $n -lt 1024) { $sb.ToString() } else { $Path }
}

function Resolve-MachinePathEntry {
    <#
    .SYNOPSIS
        Expands an entry to the absolute path the registry value means for the REAL user.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Entry, [Parameter(Mandatory)]$Context)
    $e = $Entry.Trim().Trim('"')
    $e = [regex]::Replace($e, '%(USERPROFILE|HOME)%', { $Context.UserHome }, 'IgnoreCase')
    $e = [regex]::Replace($e, '%LOCALAPPDATA%', { $Context.LocalAppData }, 'IgnoreCase')
    $e = [regex]::Replace($e, '%APPDATA%', { Join-Path $Context.UserHome 'AppData\Roaming' }, 'IgnoreCase')
    $e = [Environment]::ExpandEnvironmentVariables($e)
    $e = $e.Replace('/', '\')
    if ($e.Length -gt 3) { $e = $e.TrimEnd('\') }
    if ($e -match '\\\.\.(\\|$)') { $e = [IO.Path]::GetFullPath($e) }   # C:\Program Files\Gpg4win\..\GnuPG\bin
    ConvertTo-MachineLongPath -Path $e
}

function Test-MachinePathEntries {
    <#
    .SYNOPSIS
        Pure check of both scopes' PATH entries against the policy.
    .PARAMETER UserEntries
        Raw user PATH entries (unexpanded).
    .PARAMETER MachineEntries
        Raw machine PATH entries (unexpanded).
    .PARAMETER DirectoryExists
        Injected existence probe, so tests need no disk.
    #>
    [CmdletBinding()]
    param(
        [AllowEmptyCollection()][string[]]$UserEntries = @(),
        [AllowEmptyCollection()][string[]]$MachineEntries = @(),
        [string]$UserKind = 'String',
        [string]$MachineKind = 'ExpandString',
        [Parameter(Mandatory)]$Policy,
        [Parameter(Mandatory)]$Context,
        [scriptblock]$DirectoryExists = { param($p) Test-Path -LiteralPath $p -PathType Container }
    )
    $findings = [System.Collections.Generic.List[object]]::new()
    $cmp = [StringComparer]::OrdinalIgnoreCase
    $seen = @{}   # resolved -> 'Scope#Index'
    $scopes = @(@{ Name = 'User'; Entries = $UserEntries; Kind = $UserKind }, @{ Name = 'Machine'; Entries = $MachineEntries; Kind = $MachineKind })

    foreach ($s in $scopes) {
        $raw = ($s.Entries -join ';')
        if ($raw.Length -gt $Policy.LengthWarn) {
            $findings.Add((New-MachinePathFinding Warn 'PATH-LEN' $s.Name '' "PATH is $($raw.Length) characters; older tools truncate near $($Policy.LengthWarn)."))
        }
        $hasVar = $false
        foreach ($entry in $s.Entries) {
            if ($entry -match '%[^%]+%') { $hasVar = $true }
            if ($entry -match '%(USERPROFILE|HOME|HOMEPATH|HOMEDRIVE)%') {
                $findings.Add((New-MachinePathFinding Warn 'PATH-ENVREL' $s.Name $entry 'Entry depends on USERPROFILE or HOME, which agent homes redirect. Use the absolute profile path.'))
            }
            if ($entry -match '^\s|\s$|^".*"$') {
                $findings.Add((New-MachinePathFinding Info 'PATH-FORMAT' $s.Name $entry 'Entry has surrounding whitespace or quotes.'))
            }
            $resolved = Resolve-MachinePathEntry -Entry $entry -Context $Context
            if ($resolved -notmatch '^[A-Za-z]:\\|^\\\\') {
                $findings.Add((New-MachinePathFinding Warn 'PATH-RELATIVE' $s.Name $entry 'Entry is not an absolute path; it resolves against the current directory.'))
                continue
            }
            if ($Policy.ForbiddenEntries | Where-Object { $cmp.Equals($_.TrimEnd('\'), $resolved) }) {
                $why = $Policy.ForbiddenReasons[($Policy.ForbiddenEntries | Where-Object { $cmp.Equals($_.TrimEnd('\'), $resolved) } | Select-Object -First 1)]
                $findings.Add((New-MachinePathFinding Warn 'PATH-FORBIDDEN' $s.Name $entry "Entry is forbidden by policy. $why"))
            }
            if ($Policy.ForbiddenPatterns | Where-Object { $resolved -match $_ }) {
                $findings.Add((New-MachinePathFinding Warn 'PATH-AGENTHOME' $s.Name $entry $Policy.ForbiddenPatternReason))
            }
            if (-not (& $DirectoryExists $resolved)) {
                $findings.Add((New-MachinePathFinding Warn 'PATH-DEAD' $s.Name $entry 'Directory does not exist.'))
            }
            if ($seen.ContainsKey($resolved.ToLowerInvariant())) {
                $first = $seen[$resolved.ToLowerInvariant()]
                if ($first -like "$($s.Name)#*") { $findings.Add((New-MachinePathFinding Warn 'PATH-DUP' $s.Name $entry "Duplicate of an earlier entry ($first).")) }
                else { $findings.Add((New-MachinePathFinding Info 'PATH-DUP-CROSS' $s.Name $entry "Also present in the other scope ($first); harmless and sometimes deliberate for services.")) }
            } else {
                $seen[$resolved.ToLowerInvariant()] = "$($s.Name)#$($s.Entries.IndexOf($entry))"
            }
        }
        if ($hasVar -and $s.Kind -eq 'String') {
            $findings.Add((New-MachinePathFinding Error 'PATH-KIND' $s.Name '' 'Value uses %VAR% but is REG_SZ, so the variables are never expanded. It must be REG_EXPAND_SZ.'))
        }
    }

    # Required entries and order of the head block, evaluated on resolved user entries that exist.
    $userResolved = @($UserEntries | ForEach-Object { Resolve-MachinePathEntry -Entry $_ -Context $Context })
    foreach ($req in $Policy.RequiredUserEntries) {
        if (-not ($userResolved | Where-Object { $cmp.Equals($_, $req.TrimEnd('\')) })) {
            $findings.Add((New-MachinePathFinding Warn 'PATH-MISSING' 'User' $req 'Required entry is not on the user PATH.'))
        }
    }
    $lastIdx = -1
    foreach ($h in $Policy.UserPathHead) {
        $idx = [array]::FindIndex($userResolved, [Predicate[string]] { param($x) $cmp.Equals($x, $h.TrimEnd('\')) })
        if ($idx -lt 0) { continue }
        if ($idx -lt $lastIdx) {
            $findings.Add((New-MachinePathFinding Info 'PATH-ORDER' 'User' $h 'Entry appears earlier than a policy entry that should precede it.'))
        }
        $lastIdx = [Math]::Max($lastIdx, $idx)
    }
    $findings
}

function Get-MachineCommandIndex {
    <#
    .SYNOPSIS
        Name -> ordered candidates for the given directories. Reads directories once.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string[]]$Directories, [string[]]$Extensions = @('.com', '.exe', '.bat', '.cmd', '.ps1'))
    $index = @{}
    $i = 0
    foreach ($d in $Directories) {
        $i++
        if (-not [IO.Directory]::Exists($d)) { continue }
        try { $files = [IO.Directory]::EnumerateFiles($d) } catch { continue }
        # Within one directory cmd.exe tries PATHEXT in order (.com .exe .bat .cmd); .ps1 is last because only PowerShell runs it first.
        $rank = @{ '.com' = 0; '.exe' = 1; '.bat' = 2; '.cmd' = 3; '.ps1' = 9 }
        foreach ($f in ($files | Sort-Object { $r = $rank[[IO.Path]::GetExtension($_).ToLowerInvariant()]; if ($null -eq $r) { 5 } else { $r } })) {
            $ext = [IO.Path]::GetExtension($f).ToLowerInvariant()
            if ($ext -notin $Extensions) { continue }
            $base = [IO.Path]::GetFileNameWithoutExtension($f).ToLowerInvariant()
            if (-not $index.ContainsKey($base)) { $index[$base] = [System.Collections.Generic.List[object]]::new() }
            $index[$base].Add([pscustomobject]@{ Order = $i; Dir = $d; Path = $f; Ext = $ext })
        }
    }
    $index
}

function Resolve-MachineCommand {
    <#
    .SYNOPSIS
        Fast PATH lookup (no module auto-discovery). Returns the winning path under cmd.exe PATHEXT order, or $null.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Name, [string[]]$Directories = ($env:Path -split ';' | Where-Object { $_ }), [string[]]$PathExt = @('.com', '.exe', '.bat', '.cmd'))
    foreach ($d in $Directories) {
        foreach ($e in $PathExt) {
            $p = [IO.Path]::Combine($d, $Name + $e)
            if ([IO.File]::Exists($p)) { return $p }
        }
    }
    $null
}

function Get-MachineFileHash {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)
    try { (Get-FileHash -LiteralPath $Path -Algorithm SHA256 -ErrorAction Stop).Hash } catch { $null }
}

function Test-MachinePathShadows {
    <#
    .SYNOPSIS
        Same command name in several PATH directories: order dependence, content mismatch and redundant copies.
    .DESCRIPTION
        Windows composes a new logon's PATH as machine then user, while the owner's shells and agent sessions
        put the user entries first. A name whose winner differs between the two compositions behaves
        differently depending on how the process was started. That is PATH-ORDER-DEPENDENT unless the
        policy lists the name as an intentional override.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string[]]$UserDirectories,
        [Parameter(Mandatory)][string[]]$MachineDirectories,
        [Parameter(Mandatory)]$Policy,
        [switch]$Hash
    )
    $findings = [System.Collections.Generic.List[object]]::new()
    $userFirst = @($UserDirectories + $MachineDirectories | Select-Object -Unique)
    $machineFirst = @($MachineDirectories + $UserDirectories | Select-Object -Unique)
    $idxUF = Get-MachineCommandIndex -Directories $userFirst
    $idxMF = Get-MachineCommandIndex -Directories $machineFirst
    $managed = @($Policy.ManagedBinDirs | ForEach-Object { $_.ToLowerInvariant() })
    $pathExtOrder = @('.com', '.exe', '.bat', '.cmd')

    foreach ($name in ($idxUF.Keys | Sort-Object)) {
        $cands = $idxUF[$name]
        if ($cands.Count -lt 2) { continue }
        $touchesManaged = [bool]($cands | Where-Object { $managed -contains $_.Dir.ToLowerInvariant().TrimEnd('\') })
        if (-not $touchesManaged) { continue }

        $winnerUF = ($cands | Where-Object { $_.Ext -in $pathExtOrder } | Select-Object -First 1)
        $winnerMF = ($idxMF[$name] | Where-Object { $_.Ext -in $pathExtOrder } | Select-Object -First 1)
        $dirsOf = @($cands | ForEach-Object { $_.Dir.TrimEnd('\').ToLowerInvariant() } | Select-Object -Unique)

        if ($winnerUF -and $winnerMF -and $winnerUF.Path -ne $winnerMF.Path -and -not $Policy.IntentionalShadows.ContainsKey($name)) {
            $findings.Add((New-MachinePathFinding Warn 'PATH-ORDER-DEPENDENT' '' $name "Resolves to $($winnerUF.Path) in user-first shells but $($winnerMF.Path) when the PATH is composed machine-first. Remove one copy or list the name under intentionalShadows."))
        }
        # A .ps1 next to an .exe/.cmd of the same name runs instead in PowerShell.
        $ps1 = $cands | Where-Object { $_.Ext -eq '.ps1' } | Select-Object -First 1
        if ($ps1 -and $winnerUF -and $winnerUF.Dir -eq $ps1.Dir -and -not $Policy.IntentionalShadows.ContainsKey($name)) {
            $findings.Add((New-MachinePathFinding Info 'PATH-PS1-SHADOWS' '' $name "PowerShell runs $($ps1.Path) while cmd.exe runs $($winnerUF.Path)."))
        }
        if ($Hash) {
            $exes = @($cands | Where-Object { $_.Ext -eq '.exe' })
            if ($exes.Count -ge 2) {
                $hashes = @($exes | ForEach-Object { [pscustomobject]@{ Path = $_.Path; Hash = (Get-MachineFileHash $_.Path) } })
                $distinct = @($hashes | Where-Object { $_.Hash } | Select-Object -ExpandProperty Hash -Unique)
                if ($distinct.Count -gt 1 -and $Policy.IntentionalShadows.ContainsKey($name)) { continue }
                if ($distinct.Count -gt 1) {
                    $findings.Add((New-MachinePathFinding Warn 'PATH-SHADOW-MISMATCH' '' $name ("Different binaries on PATH: " + (($hashes | ForEach-Object { "$($_.Path)=$(if ($_.Hash) { $_.Hash.Substring(0,10) } else { 'unreadable' })" }) -join ' ; '))))
                } elseif ($distinct.Count -eq 1 -and $dirsOf.Count -gt 1) {
                    $findings.Add((New-MachinePathFinding Info 'PATH-SHADOW-DUP' '' $name ("Identical copies in " + ($dirsOf -join ' and '))))
                }
            }
        }
    }
    $findings
}

function Test-MachinePathExtensionless {
    <#
    .SYNOPSIS
        Extensionless files (bash shims) that PowerShell would pick as the command although they cannot run there.
    .DESCRIPTION
        PowerShell lists an extensionless file in a PATH directory as an Application. If no same-named .exe, .cmd, .bat,
        .com or .ps1 sits beside it to win first, `name` in PowerShell fails with "Cannot run a document in the middle of
        a pipeline" even though a real name.exe exists later on PATH.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string[]]$Directories, [Parameter(Mandatory)][string[]]$AllDirectories)
    $index = Get-MachineCommandIndex -Directories $AllDirectories
    foreach ($d in $Directories) {
        if (-not [IO.Directory]::Exists($d)) { continue }
        foreach ($f in [IO.Directory]::EnumerateFiles($d)) {
            if ([IO.Path]::GetExtension($f) -ne '') { continue }
            $name = [IO.Path]::GetFileName($f).ToLowerInvariant()
            if (-not $index.ContainsKey($name)) { continue }
            $same = @($index[$name] | Where-Object { $_.Dir.TrimEnd('') -eq $d.TrimEnd('') })
            $myOrder = [array]::FindIndex($AllDirectories, [Predicate[string]] { param($x) $x.TrimEnd('') -eq $d.TrimEnd('') }) + 1
            $elsewhere = @($index[$name] | Where-Object { $_.Order -gt $myOrder })   # only a LATER directory is shadowed
            $earlier = @($index[$name] | Where-Object { $_.Order -lt $myOrder })   # a runnable in an earlier directory already wins
            if ($same.Count -eq 0 -and $earlier.Count -eq 0 -and $elsewhere.Count -gt 0) {
                New-MachinePathFinding Warn 'PATH-EXTENSIONLESS' '' $f "PowerShell resolves `"$name`" to this extensionless file before $($elsewhere[0].Path) and cannot run it."
            }
        }
    }
}

function Get-MachineVariantFile {
    <#
    .SYNOPSIS
        SHA-suffixed or pre-<sha> copies of a binary that live next to the canonical one.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Directory)
    if (-not [IO.Directory]::Exists($Directory)) { return }
    $rx = '^(?<base>[A-Za-z0-9_.-]+?)[-.](?:pre-)?(?<sha>[0-9a-f]{7,40})\.exe$|^(?<base2>[A-Za-z0-9_-]+)\.pre-[0-9a-f]{7,40}\.exe$|~RF[0-9a-f]+\.TMP$'
    foreach ($f in [IO.Directory]::EnumerateFiles($Directory)) {
        $n = [IO.Path]::GetFileName($f)
        if ($n -match $rx -and $n -match '[0-9a-f]{7,40}|~RF') { $f }
    }
}

function Test-MachinePathWrappers {
    <#
    .SYNOPSIS
        Broken links, wrappers whose targets are all missing, and wrappers that break in an agent home.
    .DESCRIPTION
        A wrapper that builds a path from USERPROFILE or HOME only breaks when that path exists under the
        real profile but not under the redirected one, so the check tests exactly that. Without -AgentHome
        the USERPROFILE references are reported as Info.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string[]]$Directories, [Parameter(Mandatory)]$Context, [string]$AgentHome)
    $findings = [System.Collections.Generic.List[object]]::new()
    $targetRx = [regex]'(?<![A-Za-z])(?<p>[A-Za-z]:[\\/](?![\\/])[^"''<>|*?\r\n;]+?\.(?:exe|cmd|bat|ps1))'
    $homeRx = [regex]'(?i)(?:%USERPROFILE%|%HOME%|\$env:USERPROFILE|\$env:HOME|\$HOME)[\\/]+(?<rest>[^"''\s%;)\r\n]+)'
    foreach ($d in $Directories) {
        if (-not [IO.Directory]::Exists($d)) { continue }
        foreach ($item in (Get-ChildItem -LiteralPath $d -Force -File -ErrorAction SilentlyContinue)) {
            if ($item.LinkType -in 'SymbolicLink', 'Junction') {
                $t = @($item.Target)[0]
                if ($t -and -not [IO.Path]::IsPathRooted($t)) { $t = Join-Path $d $t }
                if (-not $t -or -not (Test-Path -LiteralPath $t)) {
                    $findings.Add((New-MachinePathFinding Warn 'PATH-BROKEN-LINK' '' $item.FullName "Link target is missing: $t"))
                }
                continue
            }
            if ($item.Extension -notin '.cmd', '.bat', '.ps1' -or $item.Length -gt 65536) { continue }
            $text = Get-Content -LiteralPath $item.FullName -Raw -ErrorAction SilentlyContinue
            if (-not $text) { continue }
            $rels = @($homeRx.Matches($text) | ForEach-Object { $_.Groups['rest'].Value.TrimEnd('.', ',', '\', '/') } | Where-Object { $_ } | Select-Object -Unique)
            if ($rels.Count -gt 0) {
                $breaks = @()
                if ($AgentHome) {
                    $breaks = @($rels | Where-Object { (Test-Path -LiteralPath (Join-Path $Context.UserHome $_)) -and -not (Test-Path -LiteralPath (Join-Path $AgentHome $_)) })
                }
                if ($breaks.Count -gt 0) {
                    $findings.Add((New-MachinePathFinding Warn 'PATH-WRAPPER-ENVREL' '' $item.FullName ("Resolves under USERPROFILE or HOME, which an agent home redirects; these exist only in the real profile: " + ($breaks -join ', '))))
                } else {
                    $findings.Add((New-MachinePathFinding Info 'PATH-WRAPPER-HOMEREF' '' $item.FullName 'Builds a path from USERPROFILE or HOME (works in agent homes today).'))
                }
            }
            $targets = @($targetRx.Matches($text) | ForEach-Object { $_.Groups['p'].Value } | Where-Object { $_ -notmatch '[%$]' } | Select-Object -Unique)
            if ($targets.Count -gt 0 -and -not ($targets | Where-Object { Test-Path -LiteralPath $_ })) {
                $findings.Add((New-MachinePathFinding Warn 'PATH-WRAPPER-DEAD' '' $item.FullName ("None of the referenced targets exist: " + ($targets -join ', '))))
            }
        }
    }
    $findings
}

function Get-MachinePathPlan {
    <#
    .SYNOPSIS
        Pure: desired user and machine PATH from the current entries and the policy.
    .OUTPUTS
        UserEntries, MachineEntries (strings to write) and Removed (entry, scope, reason).
    #>
    [CmdletBinding()]
    param(
        [AllowEmptyCollection()][string[]]$UserEntries = @(),
        [AllowEmptyCollection()][string[]]$MachineEntries = @(),
        [Parameter(Mandatory)]$Policy,
        [Parameter(Mandatory)]$Context,
        [scriptblock]$DirectoryExists = { param($p) Test-Path -LiteralPath $p -PathType Container }
    )
    $cmp = [StringComparer]::OrdinalIgnoreCase
    $removed = [System.Collections.Generic.List[object]]::new()
    $forbidden = [System.Collections.Generic.HashSet[string]]::new($cmp)
    foreach ($f in $Policy.ForbiddenEntries) { [void]$forbidden.Add($f.TrimEnd('\')) }

    $build = {
        param($scope, $entries, $head)
        # Duplicates are removed within one scope only. A directory in both scopes is deliberate on hosts whose services and sshd
        # sessions never see the user PATH (dtm-work keeps ~/bin in the machine PATH), and it costs nothing.
        $seen = [System.Collections.Generic.HashSet[string]]::new($cmp)
        $out = [System.Collections.Generic.List[string]]::new()
        $headResolved = [System.Collections.Generic.HashSet[string]]::new($cmp)
        foreach ($h in @($head)) {
            if (-not $h) { continue }
            $r = Resolve-MachinePathEntry -Entry $h -Context $Context
            if ((& $DirectoryExists $r) -and -not $forbidden.Contains($r) -and $seen.Add($r)) { $out.Add($r); [void]$headResolved.Add($r) }
        }
        foreach ($entry in @($entries)) {
            if (-not $entry) { continue }
            $resolved = Resolve-MachinePathEntry -Entry $entry -Context $Context
            if ($forbidden.Contains($resolved)) { $removed.Add([pscustomobject]@{ Scope = $scope; Entry = $entry; Reason = 'forbidden' }); continue }
            if ($Policy.ForbiddenPatterns | Where-Object { $resolved -match $_ }) { $removed.Add([pscustomobject]@{ Scope = $scope; Entry = $entry; Reason = 'agent-home' }); continue }
            if (-not (& $DirectoryExists $resolved)) { $removed.Add([pscustomobject]@{ Scope = $scope; Entry = $entry; Reason = 'dead' }); continue }
            if ($headResolved.Contains($resolved)) { continue }   # moved to the head block, not removed
            if (-not $seen.Add($resolved)) { $removed.Add([pscustomobject]@{ Scope = $scope; Entry = $entry; Reason = 'duplicate' }); continue }
            # Machine entries keep their original text (system-managed, may use %SystemRoot%); user entries become absolute.
            $out.Add($(if ($scope -eq 'Machine') { $entry } else { $resolved }))
        }
        , $out
    }
    $machine = & $build 'Machine' $MachineEntries @()
    $user = & $build 'User' $UserEntries $Policy.UserPathHead
    [pscustomobject]@{ UserEntries = @($user); MachineEntries = @($machine); Removed = @($removed) }
}

function Backup-MachinePath {
    <#
    .SYNOPSIS
        Writes <Host>.<scope>.path.bak-<hash> (raw value, kind on the first line) and returns the file paths.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Directory, [string]$HostName = $env:COMPUTERNAME.ToLowerInvariant())
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
    foreach ($scope in 'User', 'Machine') {
        $reg = Get-MachinePathRegistry -Scope $scope
        $body = "kind=$($reg.Kind)`n$($reg.Raw)"
        $sha = [BitConverter]::ToString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($body))).Replace('-', '').Substring(0, 12).ToLowerInvariant()
        $file = Join-Path $Directory "$HostName.$($scope.ToLowerInvariant()).path.bak-$sha"
        if (-not (Test-Path -LiteralPath $file)) { [IO.File]::WriteAllText($file, $body, [Text.UTF8Encoding]::new($false)) }
        $file
    }
}

function Test-MachineShimReplacement {
    <#
    .SYNOPSIS
        True when, with the shim gone, the command still resolves to the shim's target: the first remaining candidate IS the target.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Target, [AllowEmptyCollection()][string[]]$Candidate = @())
    $norm = { param($p) ([string]$p).Replace('/', '\').TrimEnd('\').ToLowerInvariant() }
    $first = @($Candidate | Select-Object -First 1)
    ($first.Count -eq 1) -and ((& $norm $first[0]) -eq (& $norm $Target))
}

function Get-MachineBuildJobCap {
    <#
    .SYNOPSIS
        Build-job count for a host: the default, lowered to buildPolicy.maxNativeBuildJobs of that host's machine profile.
    #>
    [CmdletBinding()]
    param([string]$HostName = $env:COMPUTERNAME, [string[]]$ProfileDirectories = @(), [int]$Default = 8)
    foreach ($d in $ProfileDirectories) {
        if (-not (Test-Path -LiteralPath $d -PathType Container)) { continue }
        foreach ($f in (Get-ChildItem -LiteralPath $d -Filter '*.json' -File -ErrorAction SilentlyContinue)) {
            try { $p = Get-Content -LiteralPath $f.FullName -Raw | ConvertFrom-Json -ErrorAction Stop } catch { continue }
            if ($p.PSObject.Properties['hostName'] -and $p.hostName -ieq $HostName -and $p.PSObject.Properties['buildPolicy'] -and $p.buildPolicy.PSObject.Properties['maxNativeBuildJobs']) {
                return [Math]::Min($Default, [int]$p.buildPolicy.maxNativeBuildJobs)
            }
        }
    }
    $Default
}

function Get-MachineTaskActionText {
    <#
    .SYNOPSIS
        "Execute Arguments" for every action of the given scheduled tasks. COM-handler and other actions have no Execute property.
    #>
    [CmdletBinding()]
    param([object[]]$Task = @())
    foreach ($tk in $Task) {
        foreach ($a in @($tk.Actions)) {
            $exe = if ($a.PSObject.Properties['Execute']) { $a.Execute } else { '' }
            $arg = if ($a.PSObject.Properties['Arguments']) { $a.Arguments } else { '' }
            if ($exe -or $arg) { "$exe $arg" }
        }
    }
}

function Test-MachineTextReference {
    <#
    .SYNOPSIS
        True when the (lower-cased, slash-normalised) corpus names the path in any spelling a config, hook or script can use.
    .DESCRIPTION
        Recognised: the absolute path, ~\rest, %USERPROFILE%\rest, %HOME%\rest, $HOME\rest, $env:USERPROFILE\rest and $env:HOME\rest,
        where rest is the path below the profile. A path outside the profile is matched absolutely only.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][AllowEmptyString()][string]$CorpusText, [Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$UserHome)
    $needle = $Path.Replace('/', '\').ToLowerInvariant()
    if ($CorpusText.Contains($needle)) { return $true }
    $profileRoot = $UserHome.Replace('/', '\').TrimEnd('\').ToLowerInvariant()
    if (-not $needle.StartsWith($profileRoot + '\')) { return $false }
    $rel = $needle.Substring($profileRoot.Length)
    foreach ($prefix in '~', '%userprofile%', '%home%', '$home', '$env:userprofile', '$env:home') {
        if ($CorpusText.Contains($prefix + $rel)) { return $true }
    }
    $false
}

function Get-MachineReferenceCorpus {
    <#
    .SYNOPSIS
        Lower-cased text of everything that can name a binary by path, plus a list of every source that could not be read.
    .DESCRIPTION
        Callers must treat a non-empty Failures list as "the search is incomplete": an unread source may reference the file,
        so archiving on that basis fails open. Optional files that do not exist are not failures. The providers are injectable
        for tests and default to scheduled tasks, services and running processes.
    #>
    [CmdletBinding()]
    param(
        [string[]]$Files = @(), [string[]]$Directories = @(),
        [string[]]$ExcludePattern = @('\\(backups|archive|logs|imports|migration|\.superseded|node_modules|path-tool)\\'),
        [scriptblock]$TaskProvider = { Get-MachineTaskActionText -Task @(Get-ScheduledTask -ErrorAction Stop) },
        [scriptblock]$ServiceProvider = { Get-CimInstance Win32_Service -ErrorAction Stop | ForEach-Object { $_.PathName } },
        [scriptblock]$ProcessProvider = { Get-CimInstance Win32_Process -ErrorAction Stop | ForEach-Object { "$($_.ExecutablePath) $($_.CommandLine)" } }
    )
    $sb = [System.Text.StringBuilder]::new()
    $failures = [System.Collections.Generic.List[string]]::new()
    $sources = 0
    foreach ($f in $Files) {
        if (-not (Test-Path -LiteralPath $f -PathType Leaf)) { continue }
        try { [void]$sb.AppendLine((Get-Content -LiteralPath $f -Raw -ErrorAction Stop)); $sources++ } catch { $failures.Add("file ${f}: $($_.Exception.Message)") }
    }
    foreach ($d in $Directories) {
        if (-not (Test-Path -LiteralPath $d -PathType Container)) { continue }
        $err = @()
        $found = Get-ChildItem -LiteralPath $d -File -Force -Recurse -Depth 4 -Include *.ps1, *.psm1, *.cmd, *.bat, *.json, *.toml, *.yml, *.xml, *.lua, *.sh, *.conf -ErrorAction SilentlyContinue -ErrorVariable +err
        foreach ($e in $err) { $failures.Add("enumerate ${d}: $($e.Exception.Message)") }
        foreach ($item in $found) {
            if ($item.Length -ge 1MB -or ($ExcludePattern | Where-Object { $item.FullName -match $_ })) { continue }
            try { [void]$sb.AppendLine((Get-Content -LiteralPath $item.FullName -Raw -ErrorAction Stop)); $sources++ } catch { $failures.Add("file $($item.FullName): $($_.Exception.Message)") }
        }
    }
    foreach ($p in @(@{ Name = 'scheduled tasks'; Block = $TaskProvider }, @{ Name = 'services'; Block = $ServiceProvider }, @{ Name = 'processes'; Block = $ProcessProvider })) {
        try { foreach ($line in @(& $p.Block)) { [void]$sb.AppendLine([string]$line) }; $sources++ } catch { $failures.Add("$($p.Name): $($_.Exception.Message)") }
    }
    [pscustomobject]@{ Text = $sb.ToString().Replace('\\', '\').Replace('/', '\').ToLowerInvariant(); Sources = $sources; Failures = @($failures) }
}

function Send-MachineEnvironmentChange {
    [CmdletBinding()]
    param()
    if (-not ('MachinePath.NativeMethods' -as [type])) {
        Add-Type -Namespace MachinePath -Name NativeMethods -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("user32.dll", SetLastError = true, CharSet = System.Runtime.InteropServices.CharSet.Auto)]
public static extern System.IntPtr SendMessageTimeout(System.IntPtr hWnd, uint Msg, System.UIntPtr wParam, string lParam, uint fuFlags, uint uTimeout, out System.UIntPtr lpdwResult);
'@
    }
    $result = [UIntPtr]::Zero
    [void][MachinePath.NativeMethods]::SendMessageTimeout([IntPtr]0xffff, 0x1A, [UIntPtr]::Zero, 'Environment', 2, 5000, [ref]$result)
}

function Set-MachinePathValue {
    <#
    .SYNOPSIS
        Writes one scope's PATH as REG_EXPAND_SZ.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param([Parameter(Mandatory)][ValidateSet('User', 'Machine')][string]$Scope, [Parameter(Mandatory)][string[]]$Entries)
    $value = $Entries -join ';'
    # Always REG_EXPAND_SZ: harmless for absolute entries and correct if a %VAR% is ever added later.
    $kind = [Microsoft.Win32.RegistryValueKind]::ExpandString
    $subKey = if ($Scope -eq 'User') { 'Environment' } else { 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment' }
    $root = if ($Scope -eq 'User') { [Microsoft.Win32.Registry]::CurrentUser } else { [Microsoft.Win32.Registry]::LocalMachine }
    if ($PSCmdlet.ShouldProcess("$Scope PATH", "write $($Entries.Count) entries")) {
        $key = $root.OpenSubKey($subKey, $true)
        try { $key.SetValue('Path', $value, $kind) } finally { $key.Close() }
    }
}

function Get-MachineResolutionBaseline {
    <#
    .SYNOPSIS
        For each command: every PATH candidate in order, the winner, its size and SHA-256.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string[]]$Commands, [Parameter(Mandatory)][string[]]$Directories, [switch]$NoHash)
    $index = Get-MachineCommandIndex -Directories $Directories
    $cache = @{}
    foreach ($c in $Commands) {
        $key = $c.ToLowerInvariant()
        $cands = if ($index.ContainsKey($key)) { @($index[$key]) } else { @() }
        $winner = $cands | Where-Object { $_.Ext -in '.com', '.exe', '.bat', '.cmd' } | Select-Object -First 1
        $hash = $null; $size = $null
        if ($winner) {
            $size = ([IO.FileInfo]::new($winner.Path)).Length
            if (-not $NoHash) {
                if (-not $cache.ContainsKey($winner.Path)) { $cache[$winner.Path] = if ($size -lt 300MB) { Get-MachineFileHash $winner.Path } else { "size:$size" } }
                $hash = $cache[$winner.Path]
            }
        }
        [pscustomobject]@{
            Command    = $c
            Winner     = if ($winner) { $winner.Path } else { $null }
            WinnerPwsh = $(if ($cands) { $ps = $cands | Where-Object { $_.Ext -eq '.ps1' } | Select-Object -First 1; $first = $cands | Select-Object -First 1; if ($ps -and $ps.Dir -eq $first.Dir) { $ps.Path } elseif ($winner) { $winner.Path } else { $null } } else { $null })
            Hash       = $hash
            Size       = $size
            Candidates = @($cands | ForEach-Object { $_.Path })
        }
    }
}

function Compare-MachineResolutionBaseline {
    <#
    .SYNOPSIS
        Rows whose winner path or content changed between two baselines, and commands that stopped resolving.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)]$Before, [Parameter(Mandatory)]$After)
    $afterBy = @{}; foreach ($a in $After) { $afterBy[$a.Command] = $a }
    foreach ($b in $Before) {
        $a = $afterBy[$b.Command]
        if (-not $a) { continue }
        $status = if ($b.Winner -and -not $a.Winner) { 'LOST' }
        elseif (-not $b.Winner -and $a.Winner) { 'GAINED' }
        elseif ($b.Winner -ne $a.Winner) { 'MOVED' }
        elseif ($b.Hash -and $a.Hash -and $b.Hash -ne $a.Hash) { 'CONTENT' }
        else { continue }
        [pscustomobject]@{ Command = $b.Command; Status = $status; Before = $b.Winner; After = $a.Winner }
    }
}

Export-ModuleMember -Function Test-MachineTextReference, Test-MachineShimReplacement, Get-MachineTaskActionText, Get-MachineBuildJobCap, Get-MachineReferenceCorpus, Get-MachineFileHash, ConvertTo-MachineLongPath, New-MachinePathFinding, Get-MachinePathContext, Expand-MachinePathToken, Read-MachinePathPolicy,
    Get-MachinePathRegistry, Resolve-MachinePathEntry, Test-MachinePathEntries, Get-MachineCommandIndex, Resolve-MachineCommand,
    Test-MachinePathShadows, Test-MachinePathExtensionless, Get-MachineVariantFile, Test-MachinePathWrappers, Get-MachinePathPlan, Backup-MachinePath,
    Send-MachineEnvironmentChange, Set-MachinePathValue, Get-MachineResolutionBaseline, Compare-MachineResolutionBaseline
