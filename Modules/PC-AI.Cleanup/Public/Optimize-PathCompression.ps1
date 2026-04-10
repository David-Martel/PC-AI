#Requires -Version 5.1
function Optimize-PathCompression {
    <#
    .SYNOPSIS
        Compresses and optimizes the PATH environment variable below the Windows GUI limit.

    .DESCRIPTION
        Advanced PATH optimizer that goes beyond Repair-MachinePath:

        * Cross-scope deduplication (drops entries from User PATH that live in Machine,
          or vice-versa, based on where each path belongs semantically).
        * Environment variable substitution (%ProgramFiles%, %LocalAppData%, %UserProfile%,
          %CUDA_PATH%) to shrink entries and survive path refactors.
        * CUDA toolkit consolidation: replaces version-pinned paths with %CUDA_PATH% for
          the currently-active version, drops inactive versions entirely.
        * Removes Claude Code agent-ephemeral paths that got promoted to system PATH.
        * Writes the registry value with the correct REG_EXPAND_SZ kind so %VAR% expansion
          actually works (PowerShell's [Environment]::SetEnvironmentVariable always writes
          REG_SZ, which silently breaks substitution — this function bypasses that).
        * Broadcasts WM_SETTINGCHANGE so running Explorer/terminals pick up the new PATH.

        Respects -WhatIf correctly — unlike Repair-MachinePath, -Force does NOT override
        -WhatIf. You can always do a dry run.

    .PARAMETER Target
        Which PATH to optimize: User, Machine, or Both. Machine requires admin.

    .PARAMETER DeduplicateCrossScope
        Remove cross-scope duplicates using this rule:
          - User-specific paths (under %USERPROFILE%) are kept in User, dropped from Machine.
          - System paths (under %ProgramFiles% etc.) are kept in Machine, dropped from User.

    .PARAMETER SubstituteVariables
        Replace literal prefixes with environment variables:
          Machine scope:  %ProgramFiles%, %ProgramFiles(x86)%, %ProgramData%, %WinDir%, %CUDA_PATH%
          User scope:     all of the above PLUS %UserProfile%, %LocalAppData%, %APPDATA%
        Machine PATH NEVER gets user-scoped vars because SYSTEM account has different values.

    .PARAMETER ConsolidateCuda
        Detect multiple CUDA version paths and consolidate to %CUDA_PATH%\bin + %CUDA_PATH%\libnvvp
        for the currently-active version only. Inactive CUDA versions are removed entirely.

    .PARAMETER RemoveAgentEphemeral
        Strip paths matching Claude Code agent-home sandboxes
        (C:\Users\*\AppData\Local\ClaudeCode\agent-homes\<uuid>\...).

    .PARAMETER ReorderForPriority
        Reorder entries into tiers: Windows core → PowerShell → SDKs → languages → editors → rest.
        WARNING: Changing order can change which binary wins if multiple installs exist.
        NOT enabled by default.

    .PARAMETER ConvertToRegExpandSz
        Force the registry value kind to REG_EXPAND_SZ (ExpandString). Required for
        %VAR% substitution to actually work. Enabled automatically if -SubstituteVariables
        is set.

    .PARAMETER Force
        Suppress confirmation prompts (ConfirmImpact=High). Does NOT override -WhatIf.

    .PARAMETER BackupPath
        Optional backup file location. Default is under %LOCALAPPDATA%\PC-AI\Logs\.

    .EXAMPLE
        Optimize-PathCompression -Target User -DeduplicateCrossScope -SubstituteVariables -RemoveAgentEphemeral -WhatIf

        Dry-run User PATH optimization.

    .EXAMPLE
        Optimize-PathCompression -Target Machine -DeduplicateCrossScope -SubstituteVariables -ConsolidateCuda -ConvertToRegExpandSz -Force

        Apply full optimization to Machine PATH without confirmation.

    .OUTPUTS
        PSCustomObject with detailed before/after statistics and change log.
    #>
    [CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [ValidateSet('User', 'Machine', 'Both')]
        [string]$Target = 'User',

        [Parameter()]
        [switch]$DeduplicateCrossScope,

        [Parameter()]
        [switch]$SubstituteVariables,

        [Parameter()]
        [switch]$ConsolidateCuda,

        [Parameter()]
        [switch]$RemoveAgentEphemeral,

        [Parameter()]
        [switch]$ReorderForPriority,

        [Parameter()]
        [switch]$ConvertToRegExpandSz,

        [Parameter()]
        [switch]$Force,

        [Parameter()]
        [string]$BackupPath
    )

    begin {
        # Substitution requires REG_EXPAND_SZ
        if ($SubstituteVariables -and -not $ConvertToRegExpandSz) {
            Write-Verbose 'SubstituteVariables implies ConvertToRegExpandSz'
            $ConvertToRegExpandSz = $true
        }

        # Honor Force by suppressing the high-impact confirmation, NOT WhatIf.
        if ($Force) {
            $ConfirmPreference = 'None'
        }

        # Registry paths
        $regPaths = @{
            Machine = 'Registry::HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment'
            User    = 'Registry::HKEY_CURRENT_USER\Environment'
        }

        # Priority tiers for reordering (lower = earlier)
        $tierRules = @(
            @{ Tier = 1; Pattern = '^C:\\Windows\\System32$|^%SystemRoot%\\System32$|^C:\\WINDOWS\\system32$' }
            @{ Tier = 1; Pattern = '^C:\\Windows$|^%SystemRoot%$|^C:\\WINDOWS$' }
            @{ Tier = 1; Pattern = '^C:\\Windows\\System32\\Wbem|^%SystemRoot%\\System32\\Wbem|^C:\\Windows\\System32\\OpenSSH|^C:\\Windows\\System32\\WindowsPowerShell' }
            @{ Tier = 2; Pattern = 'PowerShell\\7|Microsoft MPI' }
            @{ Tier = 3; Pattern = 'CUDA_PATH|CUDNN|NVIDIA|Git\\cmd|Git LFS|dotnet|PostgreSQL|LLVM\\bin|CMake\\bin|Microsoft Visual Studio' }
            @{ Tier = 4; Pattern = 'Python|\.cargo\\bin|\.local\\bin|Go\\bin|nodejs|Strawberry' }
            @{ Tier = 5; Pattern = 'VSCode|VS Code|Zed|Alacritty|vscode' }
            @{ Tier = 9; Pattern = '.*' }  # default catch-all
        )

        function Get-EntryTier {
            param([string]$Entry)
            foreach ($rule in $tierRules) {
                if ($Entry -match $rule.Pattern) { return $rule.Tier }
            }
            return 9
        }

        function Test-IsUserPath {
            param([string]$Entry)
            $expanded = [Environment]::ExpandEnvironmentVariables($Entry)
            return ($expanded -match '^[A-Za-z]:\\Users\\[^\\]+\\' -or $Entry -match '^%USERPROFILE%|^%LOCALAPPDATA%|^%APPDATA%')
        }

        function Test-IsAgentEphemeral {
            param([string]$Entry)
            return $Entry -match 'ClaudeCode\\agent-homes\\[0-9a-f]{8}-[0-9a-f]{4}'
        }

        function Normalize-Path {
            param([string]$Entry)
            return $Entry.Trim().TrimEnd('\', '/').Replace('/', '\')
        }

        function Get-CudaActivePath {
            # Prefer machine-scope CUDA_PATH, fall back to user
            $cp = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine')
            if (-not $cp) { $cp = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'User') }
            return $cp
        }

        function Get-AllowedSubstitutions {
            param([string]$Scope)

            $subs = [System.Collections.Generic.List[PSObject]]::new()

            # Machine-safe (SYSTEM account has these)
            $subs.Add([PSCustomObject]@{ Token = '%ProgramFiles%';       Literal = $env:ProgramFiles })
            $subs.Add([PSCustomObject]@{ Token = '%ProgramFiles(x86)%';  Literal = ${env:ProgramFiles(x86)} })
            $subs.Add([PSCustomObject]@{ Token = '%ProgramData%';        Literal = $env:ProgramData })
            # %SystemRoot% is safe but swapping it can break hard-coded early-boot lookups; skip by default.

            # CUDA_PATH (if set; applies to both scopes because toolkit is machine-installed)
            $cudaPath = Get-CudaActivePath
            if ($cudaPath) {
                $subs.Add([PSCustomObject]@{ Token = '%CUDA_PATH%'; Literal = $cudaPath })
            }

            # User-only (NEVER in Machine scope — SYSTEM account resolves these differently)
            if ($Scope -eq 'User') {
                $subs.Add([PSCustomObject]@{ Token = '%LOCALAPPDATA%'; Literal = $env:LOCALAPPDATA })
                $subs.Add([PSCustomObject]@{ Token = '%APPDATA%';      Literal = $env:APPDATA })
                $subs.Add([PSCustomObject]@{ Token = '%USERPROFILE%';  Literal = $env:USERPROFILE })
            }

            # Longest literal first — prevents %USERPROFILE% from eating %LOCALAPPDATA%'s prefix
            return $subs | Where-Object { $_.Literal } | Sort-Object { $_.Literal.Length } -Descending
        }

        function Invoke-Substitute {
            param(
                [string]$Entry,
                [object[]]$Substitutions
            )
            foreach ($sub in $Substitutions) {
                if ($Entry -like "$($sub.Literal)*" -or $Entry -like "$($sub.Literal)\*") {
                    return $sub.Token + $Entry.Substring($sub.Literal.Length)
                }
                # Also handle trailing-slash literal
                $withSlash = $sub.Literal + '\'
                if ($Entry -like "$withSlash*") {
                    return $sub.Token + '\' + $Entry.Substring($withSlash.Length)
                }
            }
            return $Entry
        }

        function Test-IsCudaVersionPath {
            param([string]$Entry)
            return $Entry -match 'NVIDIA GPU Computing Toolkit\\CUDA\\v\d+\.\d+'
        }

        function Get-CudaVersionFromPath {
            param([string]$Entry)
            if ($Entry -match 'CUDA\\v(\d+\.\d+)') { return $Matches[1] }
            return $null
        }
    }

    process {
        $scopes = if ($Target -eq 'Both') { @('Machine', 'User') } else { @($Target) }

        $overallResult = [PSCustomObject]@{
            Timestamp    = Get-Date
            Scopes       = @{}
            TotalChanges = 0
            Success      = $true
        }

        foreach ($scope in $scopes) {
            # Admin gate for Machine
            if ($scope -eq 'Machine' -and -not (Test-IsAdministrator)) {
                $msg = "Administrator privileges required to modify Machine PATH."
                Write-CleanupLog -Message $msg -Level Error
                Write-Error $msg
                $overallResult.Success = $false
                continue
            }

            $result = [PSCustomObject]@{
                Scope                  = $scope
                OriginalLength         = 0
                FinalLength            = 0
                OriginalEntries        = 0
                FinalEntries           = 0
                BackupPath             = $null
                OriginalRegKind        = $null
                FinalRegKind           = $null
                AgentEphemeralRemoved  = 0
                CudaConsolidated       = 0
                CrossScopeRemoved      = 0
                Substituted            = 0
                Duplicates             = 0
                Changes                = [System.Collections.Generic.List[PSObject]]::new()
                Success                = $false
            }

            Write-Verbose "Optimizing $scope PATH"
            Write-CleanupLog -Message "Optimize-PathCompression start ($scope)" -Level Info

            # Read current
            $current = [Environment]::GetEnvironmentVariable('PATH', $scope)
            if ([string]::IsNullOrEmpty($current)) {
                Write-Warning "$scope PATH is empty"
                $result.Success = $true
                $overallResult.Scopes[$scope] = $result
                continue
            }
            $result.OriginalLength = $current.Length
            $result.OriginalRegKind = (Get-Item $regPaths[$scope]).GetValueKind('Path').ToString()

            # Backup
            $result.BackupPath = Backup-EnvironmentVariable -Name 'PATH' -Target $scope -BackupPath $BackupPath

            # Parse
            $entries = $current -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { Normalize-Path $_ }
            $result.OriginalEntries = $entries.Count

            # Load "opposite scope" set for cross-scope dedup
            $otherScope = if ($scope -eq 'Machine') { 'User' } else { 'Machine' }
            $otherEntries = @()
            if ($DeduplicateCrossScope) {
                $otherRaw = [Environment]::GetEnvironmentVariable('PATH', $otherScope)
                if ($otherRaw) {
                    $otherEntries = $otherRaw -split ';' |
                        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
                        ForEach-Object { [Environment]::ExpandEnvironmentVariables((Normalize-Path $_)).ToLowerInvariant() }
                }
            }

            # Active CUDA version (for consolidation)
            $cudaActive = Get-CudaActivePath
            $cudaActiveVersion = $null
            if ($cudaActive -and $cudaActive -match 'v(\d+\.\d+)') { $cudaActiveVersion = $Matches[1] }

            $subs = Get-AllowedSubstitutions -Scope $scope

            $seen  = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
            $kept  = [System.Collections.Generic.List[string]]::new()

            foreach ($raw in $entries) {
                $entry  = $raw
                $reason = $null

                # 1. Agent-ephemeral
                if ($RemoveAgentEphemeral -and (Test-IsAgentEphemeral $entry)) {
                    $result.AgentEphemeralRemoved++
                    $result.Changes.Add([PSCustomObject]@{ Action='Removed'; Reason='AgentEphemeral'; Value=$entry })
                    continue
                }

                # 2. CUDA consolidation
                if ($ConsolidateCuda -and (Test-IsCudaVersionPath $entry)) {
                    $ver = Get-CudaVersionFromPath $entry
                    if ($cudaActiveVersion -and $ver -eq $cudaActiveVersion) {
                        # Active version — rewrite to %CUDA_PATH%\...
                        $prefix = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$ver"
                        if ($entry -like "$prefix*") {
                            $newEntry = '%CUDA_PATH%' + $entry.Substring($prefix.Length)
                            $result.Changes.Add([PSCustomObject]@{ Action='Rewritten'; Reason='CudaActive'; Value=$entry; NewValue=$newEntry })
                            $result.CudaConsolidated++
                            $entry = $newEntry
                        }
                    } else {
                        # Inactive version — drop
                        $result.Changes.Add([PSCustomObject]@{ Action='Removed'; Reason="CudaInactive (v$ver, active=v$cudaActiveVersion)"; Value=$entry })
                        $result.CudaConsolidated++
                        continue
                    }
                }

                # 3. Cross-scope dedup
                if ($DeduplicateCrossScope) {
                    $expandedLower = [Environment]::ExpandEnvironmentVariables($entry).ToLowerInvariant()
                    if ($otherEntries -contains $expandedLower) {
                        $isUserish = Test-IsUserPath $entry
                        # Rule: user-ish paths belong in User scope. System paths belong in Machine scope.
                        $shouldDrop = ($scope -eq 'Machine' -and $isUserish) -or ($scope -eq 'User' -and -not $isUserish)
                        if ($shouldDrop) {
                            $result.Changes.Add([PSCustomObject]@{ Action='Removed'; Reason="CrossScope (belongs in $otherScope)"; Value=$entry })
                            $result.CrossScopeRemoved++
                            continue
                        }
                    }
                }

                # 4. Variable substitution
                if ($SubstituteVariables) {
                    $substituted = Invoke-Substitute -Entry $entry -Substitutions $subs
                    if ($substituted -ne $entry) {
                        $result.Changes.Add([PSCustomObject]@{ Action='Substituted'; Reason='Var'; Value=$entry; NewValue=$substituted })
                        $result.Substituted++
                        $entry = $substituted
                    }
                }

                # 5. Intra-scope dedup (on the post-substitution form AND the expanded form)
                $normKey = [Environment]::ExpandEnvironmentVariables($entry).ToLowerInvariant().TrimEnd('\')
                if (-not $seen.Add($normKey)) {
                    $result.Duplicates++
                    $result.Changes.Add([PSCustomObject]@{ Action='Removed'; Reason='Duplicate'; Value=$entry })
                    continue
                }

                $kept.Add($entry)
            }

            # 6. Reorder by tier if requested (stable within tier)
            if ($ReorderForPriority) {
                $indexed = 0..($kept.Count - 1) | ForEach-Object {
                    [PSCustomObject]@{ Index=$_; Entry=$kept[$_]; Tier=(Get-EntryTier $kept[$_]) }
                }
                $sorted = $indexed | Sort-Object Tier, Index
                $kept = [System.Collections.Generic.List[string]]::new()
                foreach ($row in $sorted) { $kept.Add($row.Entry) }
                $result.Changes.Add([PSCustomObject]@{ Action='Reordered'; Reason='PriorityTiers'; Value="$($kept.Count) entries" })
            }

            # Build new PATH
            $newPath = [string]::Join(';', $kept)
            $result.FinalLength  = $newPath.Length
            $result.FinalEntries = $kept.Count

            $summary = "$scope PATH: $($result.OriginalLength) -> $($result.FinalLength) chars, $($result.OriginalEntries) -> $($result.FinalEntries) entries. " +
                       "Removed: $($result.AgentEphemeralRemoved) agent, $($result.CudaConsolidated) cuda, $($result.CrossScopeRemoved) xscope, $($result.Duplicates) dupes. Substituted: $($result.Substituted)."
            Write-Host $summary -ForegroundColor Cyan

            if ($result.FinalLength -eq $result.OriginalLength -and $result.Changes.Count -eq 0) {
                Write-Host "  (no changes)" -ForegroundColor DarkGray
                $result.Success = $true
                $overallResult.Scopes[$scope] = $result
                continue
            }

            # Apply via ShouldProcess — -WhatIf returns false here
            $targetDesc = "$scope PATH ($($result.OriginalLength)→$($result.FinalLength) chars)"
            if ($PSCmdlet.ShouldProcess($targetDesc, "Write optimized PATH")) {
                try {
                    $desiredKind = if ($ConvertToRegExpandSz) { 'ExpandString' } else { $result.OriginalRegKind }

                    # Direct registry write preserves the kind
                    Set-ItemProperty -LiteralPath $regPaths[$scope] -Name 'Path' -Value $newPath -Type $desiredKind -ErrorAction Stop

                    $result.FinalRegKind = (Get-Item $regPaths[$scope]).GetValueKind('Path').ToString()

                    # Update current process PATH too
                    if ($scope -eq 'Machine') {
                        $u = [Environment]::GetEnvironmentVariable('PATH', 'User')
                        $env:PATH = "$newPath;$u"
                    } else {
                        $m = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
                        $env:PATH = "$m;$newPath"
                    }

                    $result.Success = $true
                    Write-CleanupLog -Message "Optimize-PathCompression applied to ${scope}: $($result.OriginalLength) -> $($result.FinalLength)" -Level Info
                } catch {
                    $result.Success = $false
                    $overallResult.Success = $false
                    Write-CleanupLog -Message "Optimize-PathCompression failed ($scope): $_" -Level Error
                    Write-Error "Failed to write $scope PATH: $_"
                }
            } else {
                # WhatIf path
                $result.FinalRegKind = if ($ConvertToRegExpandSz) { 'ExpandString (would-be)' } else { $result.OriginalRegKind }
                $result.Success = $true  # Dry run is a successful no-op
            }

            $overallResult.Scopes[$scope] = $result
            $overallResult.TotalChanges += $result.Changes.Count
        }

        # Broadcast WM_SETTINGCHANGE so running shells pick up the new PATH
        if ($PSCmdlet.ShouldProcess('Windows Explorer and running shells', 'Broadcast WM_SETTINGCHANGE')) {
            try {
                if (-not ('PcaiCleanup.NativeBroadcast' -as [type])) {
                    Add-Type -Namespace PcaiCleanup -Name NativeBroadcast -MemberDefinition @"
                    [System.Runtime.InteropServices.DllImport("user32.dll", SetLastError = true, CharSet = System.Runtime.InteropServices.CharSet.Auto)]
                    public static extern System.IntPtr SendMessageTimeout(
                        System.IntPtr hWnd, uint Msg, System.UIntPtr wParam, string lParam,
                        uint fuFlags, uint uTimeout, out System.UIntPtr lpdwResult);
"@
                }
                $HWND_BROADCAST  = [IntPtr]0xFFFF
                $WM_SETTINGCHANGE = 0x001A
                $SMTO_ABORTIFHUNG = 0x0002
                [UIntPtr]$ret = [UIntPtr]::Zero
                [void][PcaiCleanup.NativeBroadcast]::SendMessageTimeout(
                    $HWND_BROADCAST, $WM_SETTINGCHANGE, [UIntPtr]::Zero, 'Environment',
                    $SMTO_ABORTIFHUNG, 5000, [ref]$ret)
                Write-Verbose 'Broadcast WM_SETTINGCHANGE OK'
            } catch {
                Write-Warning "WM_SETTINGCHANGE broadcast failed: $_"
            }
        }

        return $overallResult
    }
}
