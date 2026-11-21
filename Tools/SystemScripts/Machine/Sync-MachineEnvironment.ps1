<#
.SYNOPSIS
    Deterministic, idempotent workstation environment and PATH synchronizer for Windows fleet hosts (dtm-p1gen7, dtm-p1gen9).
.DESCRIPTION
    1. Backs up HKCU:\Environment to ~/.machine/backups/hkcu-environment-backup.json.
    2. Disables PowerToys EnvironmentVariables profile toggling (IsEnabled = false) so PowerToys never pollutes HKCU:\Environment with *_PowerToys_* backup keys.
    3. Purges all orphaned *_PowerToys_* backup keys from HKCU:\Environment.
    4. Sets canonical developer, Rust/Cargo/SCCache, CMake, GitGuard, and PowerShell environment variables directly in HKCU:\Environment (with CARGO_PREFLIGHT=0 so cargo/rust-analyzer never block on PowerShell hooks).
    5. Deduplicates, removes dead paths, and deterministically orders HKCU:\Environment\Path with user shim directories first (~/.local/bin, ~/bin, ~/.cargo/bin, ~/.dotnet/tools, ~/go/bin, %LOCALAPPDATA%\agy\bin).
    6. (Not done here.) PSModulePath sanitising and module copies stay in the host's installed script until they get their own reviewed change.
    7. Repairs broken symlinks in ~/bin and ~/.local/bin and ensures git-guard shims exist in ~/bin.
.PARAMETER DryRun
    Print what the PATH step would do and exit before any write: no backup, registry, module, link or file change.
.PARAMETER PassThru
    Return the summary object instead of formatting it.
.PARAMETER Help
    Print this usage (also -h and --help) and exit.
#>
[CmdletBinding()]
param(
    [switch]$PassThru,
    [switch]$DryRun,
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments)][string[]]$RemainingArgs = @()
)

if ($Help -or $RemainingArgs -contains '--help' -or $RemainingArgs -contains '-h') {
    'Usage: Sync-MachineEnvironment.ps1 [-DryRun] [-PassThru] [-h | --help | -Help]'
    '  -DryRun   print the planned PATH change and exit before any write (no backups, registry, modules, links or files)'
    '  -PassThru return the summary object'
    'Writes: HKCU environment and PATH, ~/bin and ~/.local/bin shims and hardlinks, ~/.machine/backups.'
    return
}

$ErrorActionPreference = 'Stop'

# The real profile, never $HOME: Claude Code sessions redirect USERPROFILE and HOME to an agent home, and this script writes
# machine-wide registry state. PATH policy lives in machine-path.policy.json next to MachinePath.psm1 (see Docs/MACHINE_PATH_POLICY.md).
$machinePathDir = @($PSScriptRoot, 'C:\codedev\PC-AI\Tools\SystemScripts\Machine') | Where-Object { $_ -and (Test-Path -LiteralPath (Join-Path $_ 'MachinePath.psm1')) } | Select-Object -First 1
if (-not $machinePathDir) { throw 'MachinePath.psm1 not found next to this script or in C:\codedev\PC-AI\Tools\SystemScripts\Machine.' }
Import-Module (Join-Path $machinePathDir 'MachinePath.psm1') -Force
$pathContext = Get-MachinePathContext
$UserHomeReal = $pathContext.UserHome

# Build concurrency honours the host's machine profile (Config/machine-profiles/<host>.json buildPolicy.maxNativeBuildJobs);
# the NUC and dtm-work cap native builds at two jobs, and a fleet-wide constant would override that.
$profileDirs = @((Join-Path $machinePathDir '..\..\..\Config\machine-profiles'), 'C:\codedev\PC-AI\Config\machine-profiles') | Where-Object { Test-Path -LiteralPath $_ -PathType Container } | ForEach-Object { (Resolve-Path -LiteralPath $_).Path }
$buildJobs = Get-MachineBuildJobCap -HostName $env:COMPUTERNAME -ProfileDirectories $profileDirs -Default 8

if ($DryRun) {
    $plannedCargoHome = if (Test-Path -LiteralPath 'T:\RustCache') { 'T:\RustCache\cargo-home' } else { Join-Path $UserHomeReal '.cargo' }
    $dryContext = Get-MachinePathContext -CargoHome $plannedCargoHome
    $dryPolicy = Read-MachinePathPolicy -Path (Join-Path $machinePathDir 'machine-path.policy.json') -Context $dryContext
    $dryPlan = Get-MachinePathPlan -UserEntries (Get-MachinePathRegistry -Scope User).Entries -MachineEntries (Get-MachinePathRegistry -Scope Machine).Entries -Policy $dryPolicy -Context $dryContext
    "Dry run for $($env:COMPUTERNAME): nothing is written."
    "  CARGO_HOME would be $plannedCargoHome; build jobs would be $buildJobs"
    "  user PATH: $((Get-MachinePathRegistry -Scope User).Entries.Count) -> $($dryPlan.UserEntries.Count) entries"
    foreach ($r in ($dryPlan.Removed | Where-Object Scope -eq 'User')) { "    remove $($r.Entry) ($($r.Reason))" }
    '  not evaluated in a dry run: HKCU backup and environment values, PowerToys keys, link and shim repair.'
    return
}

$backupRoot = Join-Path $UserHomeReal '.machine\backups'
New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null

# ---------------------------------------------------------------------------
# 1. Backup HKCU:\Environment
# ---------------------------------------------------------------------------
$regKey = Get-Item -LiteralPath 'HKCU:\Environment'
$backupData = [ordered]@{
    ComputerName = $env:COMPUTERNAME
    UserName     = $env:USERNAME
    CapturedAt   = (Get-Date).ToString('o')
    Values       = [ordered]@{}
}
foreach ($name in ($regKey.GetValueNames() | Sort-Object)) {
    $backupData.Values[$name] = [ordered]@{
        Value = $regKey.GetValue($name, $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        Kind  = $regKey.GetValueKind($name).ToString()
    }
}
$backupFile = Join-Path $backupRoot 'hkcu-environment-backup.json'
$backupData | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $backupFile -Encoding UTF8

# ---------------------------------------------------------------------------
# 2. Disable PowerToys EnvironmentVariables profile toggling
# ---------------------------------------------------------------------------
$ptProfilesPath = Join-Path $env:LOCALAPPDATA 'Microsoft\PowerToys\EnvironmentVariables\profiles.json'
$ptDisabledCount = 0
if (Test-Path -LiteralPath $ptProfilesPath) {
    $profiles = @(Get-Content -LiteralPath $ptProfilesPath -Raw | ConvertFrom-Json)
    $secretNames = @(
        'ANTHROPIC_API_KEY','OPENAI_API_KEY','GEMINI_API_KEY','VERTEX_AI_API_KEY',
        'CLOUDFLARE_API_KEY','CLOUDFLARE_API_TOKEN','CLOUDFLARE_ACCOUNT_ID','CLOUDFLARE_ZONE_ID',
        'SGITHUB_TOKEN','DOCKER_PAT','HF_TOKEN','GOOGLE_API_KEY','NANOBANANA_GEMINI_API_KEY',
        'CONTEXT7_API_KEY','JULES_API_KEY','OLLAMA_CUST_KEY','STITCH_API_KEY'
    )
    foreach ($p in $profiles) {
        if ($p.IsEnabled) {
            $p.IsEnabled = $false
            $ptDisabledCount++
        }
        if ($p.Variables) {
            $p.Variables = @($p.Variables | Where-Object { $_.Name -notin $secretNames })
        }
    }
    ConvertTo-Json -InputObject $profiles -Depth 10 | Set-Content -LiteralPath $ptProfilesPath -Encoding UTF8
}

# ---------------------------------------------------------------------------
# 3. Purge orphaned *_PowerToys_* backup keys from HKCU:\Environment
# ---------------------------------------------------------------------------
$purgedPtKeys = [System.Collections.Generic.List[string]]::new()
foreach ($name in $regKey.GetValueNames()) {
    if ($name -match '_PowerToys_') {
        Remove-ItemProperty -LiteralPath 'HKCU:\Environment' -Name $name -ErrorAction SilentlyContinue
        Remove-Item -Path "Env:\$name" -ErrorAction SilentlyContinue
        $purgedPtKeys.Add($name)
    }
}

# ---------------------------------------------------------------------------
# 4. Set deterministic developer environment variables in HKCU:\Environment
# ---------------------------------------------------------------------------
$useTRustCache = Test-Path -LiteralPath 'T:\RustCache'
$cargoHome  = if ($useTRustCache) { 'T:\RustCache\cargo-home' } else { Join-Path $UserHomeReal '.cargo' }
$rustupHome = if ($useTRustCache) { 'T:\RustCache\rustup' }     else { Join-Path $UserHomeReal '.rustup' }
$sccacheDir = if ($useTRustCache) { 'T:\RustCache\sccache' }    else { Join-Path $UserHomeReal '.cache\sccache' }
New-Item -ItemType Directory -Force -Path $sccacheDir | Out-Null

$pyo3Python = if (Test-Path -LiteralPath (Join-Path $UserHomeReal '.local\bin\python3.13.exe')) {
    Join-Path $UserHomeReal '.local\bin\python3.13.exe'
} elseif (Test-Path -LiteralPath (Join-Path $UserHomeReal '.local\bin\python3.12.exe')) {
    Join-Path $UserHomeReal '.local\bin\python3.12.exe'
} else {
    $null
}

$canonicalEnv = [ordered]@{
    'CARGO_HOME'                         = $cargoHome
    'RUSTUP_HOME'                        = $rustupHome
    'SCCACHE_DIR'                        = $sccacheDir
    'RUSTC_WRAPPER'                      = 'sccache'
    'CARGO_BUILD_RUSTC_WRAPPER'          = 'sccache'
    'CARGO_BUILD_JOBS'                   = "$buildJobs"
    'CARGO_INCREMENTAL'                  = '0'
    'RUST_BACKTRACE'                     = '1'
    'CARGO_LLD_PATH'                     = 'C:\Program Files\LLVM\bin\lld-link.exe'
    'CARGO_PREFLIGHT'                    = '0'
    'CARGO_RA_PREFLIGHT'                 = '0'
    'CARGO_PREFLIGHT_BLOCKING'           = '0'
    'CARGO_PREFLIGHT_FORCE'              = '0'
    'CARGO_PREFLIGHT_IDE_GUARD'          = '1'
    'CARGO_PREFLIGHT_MODE'               = 'all'
    'CARGO_PREFLIGHT_STRICT'             = '0'
    'CARGO_ROUTE_MACOS'                  = 'docker'
    'CARGO_ROUTE_WASM'                   = 'wsl'
    'CARGO_WSL_CACHE'                    = 'native'
    'RA_SINGLETON'                       = '1'
    'RA_LOG'                             = 'error'
    'RA_DIAGNOSTICS_FLAGS'               = '--disable-build-scripts --disable-proc-macros'
    'SCCACHE_CACHE_SIZE'                 = '30G'
    'SCCACHE_CACHE_COMPRESSION'          = 'zstd'
    'SCCACHE_DIRECT'                     = 'true'
    'SCCACHE_IDLE_TIMEOUT'               = '1800'
    'SCCACHE_SERVER_PORT'                = '4810'
    'CMAKE_GENERATOR'                    = 'Ninja'
    'CMAKE_BUILD_PARALLEL_LEVEL'         = "$buildJobs"
    'MAKEFLAGS'                          = "-j$buildJobs"
    'GIT_GUARD_ROOT'                     = 'C:/Users/david/.local/share/git-guard/current'
    'RIPGREP_CONFIG_PATH'                = (Join-Path $UserHomeReal '.config\ripgrep\config')
    'AST_GREP_GLOBAL_CONFIG'             = (Join-Path $UserHomeReal '.config\ast-grep\sgconfig.yml')
    'PATHEXT'                            = '.COM;.EXE;.BAT;.CMD;.MSC;.PY;.PYW;.CPL'
    'DOTNET_CLI_TELEMETRY_OPTOUT'        = 'true'
    'DOTNET_ADD_GLOBAL_TOOLS_TO_PATH'    = 'false'   # the SDK otherwise appends an agent home's .dotnet\tools to the user PATH on first run
    'CLAUDE_CODE_DISABLE_TERMINAL_TITLE' = '1'
    'CLAUDE_EXE'                         = (Join-Path $UserHomeReal '.local\bin\claude.exe')
    'UVX_PATH'                           = (Join-Path $UserHomeReal '.local\bin\uvx.exe')
    'OLLAMA_FLASH_ATTENTION'             = '1'
    'OLLAMA_KV_CACHE_TYPE'               = 'q8_0'
    'OLLAMA_KEEP_ALIVE'                  = '30m'
    'OLLAMA_NUM_PARALLEL'                = '2'
    'OLLAMA_MAX_LOADED_MODELS'           = '1'
    'CUDA_MODULE_LOADING'                = 'LAZY'
    'PYTHONUTF8'                         = '1'
    'POWERSHELL_PROFILE_ROOT'            = (Join-Path $UserHomeReal '.config\powershell')
    'POWERSHELL_MODULES_PATH'            = (Join-Path $env:LOCALAPPDATA 'PowerShell\Modules')
    'PROFILEUTILITIES_PCAI_AUTO_PATH_OPTIMIZE' = '0'
    'PS_SKIP_PSMODULEPATH_OPTIMIZE'      = '1'
}
if ($pyo3Python) {
    $canonicalEnv['PYO3_PYTHON'] = $pyo3Python
}
if (Test-Path -LiteralPath (Join-Path $UserHomeReal '.dotnet\sdk')) {
    $canonicalEnv['DOTNET_ROOT'] = (Join-Path $UserHomeReal '.dotnet')
} elseif (Test-Path -LiteralPath 'C:\Program Files\dotnet\sdk') {
    $canonicalEnv['DOTNET_ROOT'] = 'C:\Program Files\dotnet'
}

foreach ($kv in $canonicalEnv.GetEnumerator()) {
    [Environment]::SetEnvironmentVariable($kv.Key, $kv.Value, 'User')
    Set-Item -Path "Env:\$($kv.Key)" -Value $kv.Value
}
# CARGO_HOME may just have changed: plan the PATH from the home selected above, not the one captured at startup.
$pathContext = Get-MachinePathContext -CargoHome $cargoHome

# ---------------------------------------------------------------------------
# 5. (removed) PSModulePath sanitising and module copies
# ---------------------------------------------------------------------------
# Module publication and PSModulePath rewriting are not part of the PATH policy and are not performed here. The copy of this
# script installed under ~/.machine keeps doing them until a separate change gives them staged, validated publication with
# replacement checks and configuration merging (see docs/MACHINE_PATH_POLICY.md, Deferred).

# ---------------------------------------------------------------------------
# 6. User PATH from machine-path.policy.json
# ---------------------------------------------------------------------------
# Absolute entries only, dead and duplicate entries removed, policy head block first ({UserHome}\.local\bin, {UserHome}\bin,
# CARGO_HOME\bin, ...), %USERPROFILE% never written. ~/.cargo/bin is a forbidden entry: CARGO_HOME\bin is the canonical
# cargo bin directory (~/.agents/RUST_TOOLING.md).
foreach ($ensureDir in @(
    (Join-Path $UserHomeReal '.local\bin'),
    (Join-Path $UserHomeReal 'bin'),
    (Join-Path $UserHomeReal '.dotnet\tools'),
    (Join-Path $UserHomeReal 'go\bin')
)) {
    New-Item -ItemType Directory -Force -Path $ensureDir | Out-Null
}
$pathPolicy = Read-MachinePathPolicy -Path (Join-Path $machinePathDir 'machine-path.policy.json') -Context $pathContext
$pathPlan = Get-MachinePathPlan -UserEntries (Get-MachinePathRegistry -Scope User).Entries -MachineEntries (Get-MachinePathRegistry -Scope Machine).Entries -Policy $pathPolicy -Context $pathContext
$removedUserPaths = [System.Collections.Generic.List[string]]::new()
foreach ($r in ($pathPlan.Removed | Where-Object Scope -eq 'User')) { $removedUserPaths.Add("$($r.Entry) ($($r.Reason))") }
$cleanUserPath = $pathPlan.UserEntries
if (($cleanUserPath -join ';') -ne [Environment]::GetEnvironmentVariable('Path', 'User')) {
    Backup-MachinePath -Directory (Join-Path $backupRoot 'path') | Out-Null
    Set-MachinePathValue -Scope User -Entries $cleanUserPath
    Send-MachineEnvironmentChange
}
# Current process: user entries first (the owner's shells), then the machine PATH as registered.
$env:Path = [Environment]::ExpandEnvironmentVariables((@($cleanUserPath) + @((Get-MachinePathRegistry -Scope Machine).Entries) | Select-Object -Unique) -join ';')

# ---------------------------------------------------------------------------
# 7. Repair broken links & shims in ~/bin and ~/.local/bin
# ---------------------------------------------------------------------------
$repairedLinks = [System.Collections.Generic.List[string]]::new()
$brokenLinks = [System.Collections.Generic.List[string]]::new()
$binRustup = Join-Path $UserHomeReal 'bin\rustup.exe'
$cargoRustup = Join-Path $UserHomeReal '.cargo\bin\rustup.exe'
# `rustup self update` rewrites only $CARGO_HOME\bin, so that copy is the freshest; ~/.cargo/bin is the legacy fallback.
$cargoHomeRustup = Join-Path $cargoHome 'bin\rustup.exe'
$freshRustup = @($cargoHomeRustup, $cargoRustup) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($freshRustup) {
  try {
    $needsCopy = -not (Test-Path -LiteralPath $binRustup)
    if (-not $needsCopy) {
        # Copy when the content differs and the source is not older, so a stale source never downgrades ~/bin.
        $srcItem = Get-Item -LiteralPath $freshRustup
        $dstItem = Get-Item -LiteralPath $binRustup
        $sameContent = ($srcItem.Length -eq $dstItem.Length) -and
            ((Get-FileHash -LiteralPath $freshRustup -Algorithm SHA256).Hash -eq (Get-FileHash -LiteralPath $binRustup -Algorithm SHA256).Hash)
        $needsCopy = (-not $sameContent) -and ($srcItem.LastWriteTimeUtc -ge $dstItem.LastWriteTimeUtc)
    }
    if ($needsCopy) {
        # Overwrite in place: hardlinked proxies in the same link group follow the new content.
        Copy-Item -LiteralPath $freshRustup -Destination $binRustup -Force
    }
  } catch {
    # A running rustup/proxy can hold the file; the next sync retries. Never abort the whole sync for this.
    Write-Warning "Could not refresh ${binRustup} from ${freshRustup} (in use?): $($_.Exception.Message)"
  }
}
$rustupSource =if (Test-Path -LiteralPath $binRustup) { $binRustup } elseif (Test-Path -LiteralPath $cargoRustup) { $cargoRustup } else { $null }

$rustupProxies = @(
    'cargo.exe','rustc.exe','rustfmt.exe','rustdoc.exe','clippy-driver.exe',
    'cargo-clippy.exe','cargo-fmt.exe','cargo-miri.exe','rls.exe','rust-analyzer.exe','rust-gdb.exe','rust-lldb.exe'
)

foreach ($bdir in @((Join-Path $UserHomeReal 'bin'), (Join-Path $UserHomeReal '.local\bin'))) {
    if (-not (Test-Path -LiteralPath $bdir)) { continue }
    Get-ChildItem -LiteralPath $bdir -Force -ErrorAction SilentlyContinue | ForEach-Object {
        $item = $_
        if ($item.LinkType -in @('SymbolicLink', 'Junction')) {
            $t = $item.Target | Select-Object -First 1
            $resolvedTarget = if ($t -and -not [IO.Path]::IsPathRooted($t)) {
                Join-Path $bdir $t
            } else { $t }

            if (-not $resolvedTarget -or -not (Test-Path -LiteralPath $resolvedTarget)) {
                if ($rustupProxies -contains $item.Name -and $rustupSource) {
                    Remove-Item -LiteralPath $item.FullName -Force
                    New-Item -ItemType HardLink -Path $item.FullName -Target $rustupSource | Out-Null
                    $repairedLinks.Add("Repaired rustup proxy: $($item.FullName) -> $rustupSource")
                } else {
                    # Reported, not deleted: the target may live on a drive that is offline right now, and a file delete is not
                    # covered by the registry backup. Cleanup is deferred (docs/MACHINE_PATH_POLICY.md).
                    $brokenLinks.Add("$($item.FullName) (target: $t)")
                }
            } elseif (-not [IO.Path]::IsPathRooted($t) -and ($rustupProxies -contains $item.Name) -and $rustupSource) {
                Remove-Item -LiteralPath $item.FullName -Force
                New-Item -ItemType HardLink -Path $item.FullName -Target $rustupSource | Out-Null
                $repairedLinks.Add("Converted relative link to HardLink: $($item.FullName) -> $rustupSource")
            }
        }
    }
}

# Ensure rustup proxies exist in ~/bin
if ($rustupSource) {
    foreach ($p in $rustupProxies) {
        $dest = Join-Path $UserHomeReal "bin\$p"
        if (-not (Test-Path -LiteralPath $dest)) {
            New-Item -ItemType HardLink -Path $dest -Target $rustupSource | Out-Null
            $repairedLinks.Add("Restored rustup proxy hardlink: $dest")
        } else {
            # The proxies form hardlink groups that need not include rustup.exe (cargo/rustc/... are a
            # separate group), so refreshing rustup.exe alone leaves them on the old rustup. A proxy is
            # a copy of rustup.exe, so only a file within 10% of its size is treated as one; real
            # toolchain binaries (cargo 27 MB, rustfmt 5 MB) are never touched.
            try {
                $srcItem = Get-Item -LiteralPath $rustupSource
                $dstItem = Get-Item -LiteralPath $dest
                $looksLikeProxy = [math]::Abs($dstItem.Length - $srcItem.Length) -le ($srcItem.Length * 0.10)
                if ($looksLikeProxy -and $dstItem.LastWriteTimeUtc -le $srcItem.LastWriteTimeUtc -and
                    (Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $rustupSource -Algorithm SHA256).Hash) {
                    Copy-Item -LiteralPath $rustupSource -Destination $dest -Force
                    $repairedLinks.Add("Refreshed stale rustup proxy: $dest")
                }
            } catch {
                Write-Warning "Could not refresh rustup proxy ${dest} (in use?): $($_.Exception.Message)"
            }
        }
    }
}

# Ensure the SELECTED CARGO_HOME's bin executables on the same volume are hardlinked in ~/bin. Linking from a legacy ~/.cargo\bin when
# another CARGO_HOME was selected would let an older copy in ~/bin (ahead of the canonical cargo bin on PATH) override the current install.
$cargoBinDir = Join-Path $cargoHome 'bin'
$homeBinDir  = Join-Path $UserHomeReal 'bin'
if ((Test-Path -LiteralPath $cargoBinDir) -and ([IO.Path]::GetPathRoot((Get-Item -LiteralPath $cargoBinDir -Force).FullName) -eq [IO.Path]::GetPathRoot($homeBinDir)) -and ((Get-Item -LiteralPath $cargoBinDir -Force).LinkType -notin 'SymbolicLink', 'Junction')) {
    Get-ChildItem -LiteralPath $cargoBinDir -Filter '*.exe' -File -ErrorAction SilentlyContinue | ForEach-Object {
        $dest = Join-Path $homeBinDir $_.Name
        if (-not (Test-Path -LiteralPath $dest)) {
            New-Item -ItemType HardLink -Path $dest -Target $_.FullName | Out-Null
            $repairedLinks.Add("Linked cargo binary into ~/bin: $($_.Name)")
        }
    }
}

# Ensure pdf-tools (pdftoppm.exe, pdftotext.exe) and uv/uvx/uvw hardlinks exist
foreach ($pdfExe in @('pdftoppm.exe','pdftotext.exe')) {
    $src = Join-Path $UserHomeReal "bin\pdf-tools\poppler-25.07.0-bin\$pdfExe"
    $dst = Join-Path $UserHomeReal "bin\$pdfExe"
    if ((Test-Path -LiteralPath $src) -and -not (Test-Path -LiteralPath $dst)) {
        New-Item -ItemType HardLink -Path $dst -Target $src | Out-Null
        $repairedLinks.Add("Restored pdf-tools hardlink: $dst")
    }
}
foreach ($u in @('uv.exe','uvx.exe','uvw.exe')) {
    $src = Join-Path $UserHomeReal "bin\$u"
    $dst = Join-Path $UserHomeReal ".local\bin\$u"
    if ((Test-Path -LiteralPath $src) -and -not (Test-Path -LiteralPath $dst)) {
        New-Item -ItemType HardLink -Path $dst -Target $src | Out-Null
        $repairedLinks.Add("Restored uv hardlink: $dst")
    }
}

# Stale shadow wrappers in ~/.local/bin that shadow ~/bin (cargo.bat, cargo.cmd, sg.cmd) are REPORTED, not deleted: this unattended
# sync cannot prove a working replacement exists, and a file delete is not covered by the registry backup. Removal is deferred to the
# follow-up PR on applied archival (docs/MACHINE_PATH_POLICY.md); Test-MachinePath.ps1 reports the shadowing itself.
$staleShims = [System.Collections.Generic.List[string]]::new()
foreach ($staleShim in @('cargo.bat','cargo.cmd','sg.cmd')) {
    $stalePath = Join-Path $UserHomeReal ".local\bin\$staleShim"
    if (Test-Path -LiteralPath $stalePath) { $staleShims.Add($stalePath) }
}

# Ensure git-guard wrappers exist in ~/bin
$ggRoot = Join-Path $UserHomeReal '.local\share\git-guard\current\bin\git-guard'
if ((Test-Path -LiteralPath $ggRoot) -or (Test-Path -LiteralPath 'C:\codedev\git-guard\bin\git-guard')) {
    $ggPs1 = Join-Path $UserHomeReal 'bin\git-guard.ps1'
    $ggCmd = Join-Path $UserHomeReal 'bin\git-guard.cmd'
    $ggSh  = Join-Path $UserHomeReal 'bin\git-guard'
    if (-not (Test-Path -LiteralPath $ggPs1)) {
        Get-GitGuardWrapperText -Kind ps1 -UserHome $UserHomeReal | Set-Content -LiteralPath $ggPs1 -Encoding UTF8
        $repairedLinks.Add("Created $ggPs1")
    }
    if (-not (Test-Path -LiteralPath $ggCmd)) {
        @'
@echo off
pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0git-guard.ps1" %*
'@ | Set-Content -LiteralPath $ggCmd -Encoding ASCII
        $repairedLinks.Add("Created $ggCmd")
    }
    if (-not (Test-Path -LiteralPath $ggSh)) {
        (Get-GitGuardWrapperText -Kind sh -UserHome $UserHomeReal) -replace "`r`n", "`n" | Set-Content -LiteralPath $ggSh -Encoding UTF8 -NoNewline
        $repairedLinks.Add("Created $ggSh")
    }
}

# PATH audit (read only): counts only; run Test-MachinePath.ps1 for the findings. A child process because that script exits with its status.
$pathAudit = $null
try {
    $auditJson = & (Get-Command pwsh).Source -NoProfile -File (Join-Path $machinePathDir 'Test-MachinePath.ps1') -Format Json -NoHash 2>$null
    $pathAudit = ($auditJson -join "`n" | ConvertFrom-Json).Counts
} catch { Write-Warning "PATH audit failed: $($_.Exception.Message)" }

$summary = [pscustomobject]@{
    ComputerName          = $env:COMPUTERNAME
    BackupFile            = $backupFile
    PowerToysDisabled     = $ptDisabledCount
    PurgedPowerToysKeys   = $purgedPtKeys.Count
    CanonicalEnvVarsSet   = $canonicalEnv.Count
    UserPathEntries       = $cleanUserPath.Count
    RemovedUserPaths      = $removedUserPaths
    RepairedLinks         = $repairedLinks
    BrokenLinksNotRemoved = $brokenLinks
    StaleShimsNotRemoved  = $staleShims
    PathAudit             = $pathAudit
}

if ($PassThru) {
    return $summary
}
$summary | Format-List
