# Minimal private PowerShell profile for hosts that need PATH ordering and nothing else.
#
# Deploy as %USERPROFILE%\.config\powershell\Microsoft.PowerShell_profile.ps1, the file the public
# bootstrap (Microsoft.PowerShell_profile.ps1 in this folder) delegates to. It does only cheap,
# local work: no module imports, no network, no external processes, no cloud-file or secret reads.
# Secret access stays on demand through Connect-MachineVault.

# Preferred tool directories, highest priority first. Missing directories are skipped so the
# path never carries dead entries.
$minimalPreferred = [Collections.Generic.List[string]]::new()
$minimalCandidates = @(
    (Microsoft.PowerShell.Management\Join-Path $HOME '.local\bin'),
    (Microsoft.PowerShell.Management\Join-Path $HOME 'bin'),
    (Microsoft.PowerShell.Management\Join-Path $HOME '.cargo\bin'),
    (Microsoft.PowerShell.Management\Join-Path $HOME '.dotnet\tools'),
    (Microsoft.PowerShell.Management\Join-Path $HOME 'go\bin')
)
if ($env:LOCALAPPDATA) {
    $minimalCandidates += Microsoft.PowerShell.Management\Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links'
}
foreach ($minimalCandidate in $minimalCandidates) {
    if ([IO.Directory]::Exists($minimalCandidate)) { $minimalPreferred.Add($minimalCandidate) }
}

# Merge newly installed persistent tools into long-lived terminal environments, then put the
# preferred directories first. Entries compare case-insensitively with separators normalized.
$minimalSeen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$minimalPath = [Collections.Generic.List[string]]::new()
$minimalSources = @($minimalPreferred) + @(
    $env:Path -split ';'
    [Environment]::GetEnvironmentVariable('Path', 'Machine') -split ';'
    [Environment]::GetEnvironmentVariable('Path', 'User') -split ';'
)
foreach ($minimalEntry in $minimalSources) {
    if ([string]::IsNullOrWhiteSpace($minimalEntry)) { continue }
    $minimalKey = $minimalEntry.Replace('/', '\').TrimEnd('\')
    if ($minimalSeen.Add($minimalKey)) { $minimalPath.Add($minimalEntry.Replace('/', '\')) }
}
$env:Path = $minimalPath -join ';'

# Isolated or long-lived hosts may inherit only the PowerShell 7 module roots.
$minimalModuleRoots = [Collections.Generic.List[string]]::new()
$minimalLocalModules = Microsoft.PowerShell.Management\Join-Path $HOME '.local\share\powershell\Modules'
if ([IO.Directory]::Exists($minimalLocalModules)) { $minimalModuleRoots.Add($minimalLocalModules) }
$minimalCurrentRoots = @($env:PSModulePath -split [IO.Path]::PathSeparator)
foreach ($minimalRoot in @($minimalModuleRoots)) {
    if ($minimalCurrentRoots -notcontains $minimalRoot) {
        $env:PSModulePath = $minimalRoot + [IO.Path]::PathSeparator + $env:PSModulePath
    }
}

# Defines Import-MachineAgentEnvironment only; nothing runs until it is called.
$minimalAgentEnvironment = Microsoft.PowerShell.Management\Join-Path $HOME '.machine\Import-AgentEnvironment.ps1'
if ([IO.File]::Exists($minimalAgentEnvironment)) { . $minimalAgentEnvironment }

function Connect-MachineVault {
    [CmdletBinding()]
    param([switch]$Refresh)
    if (-not (Microsoft.PowerShell.Core\Get-Command Initialize-BitwardenSessionFromBackends -ErrorAction SilentlyContinue)) {
        $backend = Microsoft.PowerShell.Management\Join-Path $HOME '.machine\SecretBackendUtilities.ps1'
        if (Microsoft.PowerShell.Management\Test-Path -LiteralPath $backend) { . $backend }
    }
    $result = Initialize-BitwardenSessionFromBackends -Quiet -Refresh:$Refresh
    if (-not $result.Success) { throw "Vault initialization unavailable: $($result.Source). Run the workstation vault enrollment tool." }
    [pscustomobject]@{ Success = $true; Source = $result.Source; SessionAvailable = [bool]$env:BW_SESSION }
}

Remove-Variable -Name minimal* -ErrorAction SilentlyContinue
