function Import-PcaiAccelerationStack {
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [string[]]$Modules = @('ProfileAccelerator', 'PcaiNativeBridge', 'PC-AI.Acceleration'),
        [string]$RepoRoot,
        [switch]$Force,
        [switch]$RequireAll
    )

    function script:Resolve-AccelerationRepoRoot {
        param([string]$ExplicitRepoRoot)

        if (-not [string]::IsNullOrWhiteSpace($ExplicitRepoRoot) -and (Test-Path -LiteralPath $ExplicitRepoRoot -PathType Container)) {
            return (Resolve-Path -LiteralPath $ExplicitRepoRoot).Path
        }

        if (Get-Command Resolve-PcaiRepoRoot -ErrorAction SilentlyContinue) {
            foreach ($candidateStart in @($PWD.Path, $PSScriptRoot, (Get-Location).Path)) {
                if ([string]::IsNullOrWhiteSpace($candidateStart)) { continue }
                try {
                    $resolved = Resolve-PcaiRepoRoot -StartPath $candidateStart
                    if ($resolved) { return $resolved }
                } catch {}
            }
        }

        foreach ($candidate in @(
            $env:PCAI_ROOT,
            (Join-Path $env:USERPROFILE 'PC_AI'),
            (Join-Path $HOME 'PC_AI'),
            (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
        )) {
            if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
            if (Test-Path -LiteralPath $candidate -PathType Container) {
                try {
                    return (Resolve-Path -LiteralPath $candidate).Path
                } catch {
                    return $candidate
                }
            }
        }

        return $null
    }

    function script:Get-AccelerationModuleCandidates {
        param(
            [Parameter(Mandatory)][string]$ModuleName,
            [string]$ResolvedRepoRoot
        )

        $candidateRoots = @(
            $env:POWERSHELL_MODULES_PATH,
            (Join-Path $HOME 'Documents\PowerShell\Modules'),
            (Join-Path $HOME 'OneDrive\Documents\PowerShell\Modules'),
            (Join-Path $env:LOCALAPPDATA 'PowerShell\Modules')
        ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique

        $candidates = New-Object 'System.Collections.Generic.List[string]'
        foreach ($root in $candidateRoots) {
            $candidates.Add((Join-Path $root "$ModuleName\$ModuleName.psd1")) | Out-Null
            $candidates.Add((Join-Path $root "$ModuleName\$ModuleName.psm1")) | Out-Null
            $candidates.Add((Join-Path $root "PC-AI\Modules\$ModuleName\$ModuleName.psd1")) | Out-Null
            $candidates.Add((Join-Path $root "PC-AI\Modules\$ModuleName\$ModuleName.psm1")) | Out-Null
            $candidates.Add((Join-Path $root "PC-AI\Release\PowerShell\PC-AI\Modules\$ModuleName\$ModuleName.psd1")) | Out-Null
            $candidates.Add((Join-Path $root "PC-AI\Release\PowerShell\PC-AI\Modules\$ModuleName\$ModuleName.psm1")) | Out-Null
        }

        if (-not [string]::IsNullOrWhiteSpace($ResolvedRepoRoot)) {
            $candidates.Add((Join-Path $ResolvedRepoRoot "Modules\$ModuleName\$ModuleName.psd1")) | Out-Null
            $candidates.Add((Join-Path $ResolvedRepoRoot "Modules\$ModuleName\$ModuleName.psm1")) | Out-Null
            $candidates.Add((Join-Path $ResolvedRepoRoot "Release\PowerShell\PC-AI\Modules\$ModuleName\$ModuleName.psd1")) | Out-Null
            $candidates.Add((Join-Path $ResolvedRepoRoot "Release\PowerShell\PC-AI\Modules\$ModuleName\$ModuleName.psm1")) | Out-Null
        }

        return @($candidates | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
    }

    function script:Import-AccelerationModule {
        param(
            [Parameter(Mandatory)][string]$ModuleName,
            [string]$ResolvedRepoRoot,
            [switch]$ForceImport
        )

        $existing = Get-Module -Name $ModuleName -ErrorAction SilentlyContinue
        if ($existing -and -not $ForceImport) {
            return [pscustomobject]@{
                Name = $ModuleName
                Available = $true
                Imported = $false
                Module = $existing
                Path = $existing.Path
                Source = 'loaded'
            }
        }

        try {
            $imported = Import-Module $ModuleName -Force:$ForceImport -PassThru -ErrorAction Stop
            return [pscustomobject]@{
                Name = $ModuleName
                Available = $true
                Imported = $true
                Module = $imported
                Path = $imported.Path
                Source = 'module-path'
            }
        } catch {}

        foreach ($candidate in (Get-AccelerationModuleCandidates -ModuleName $ModuleName -ResolvedRepoRoot $ResolvedRepoRoot)) {
            if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
            try {
                $imported = Import-Module $candidate -Force:$ForceImport -PassThru -ErrorAction Stop
                return [pscustomobject]@{
                    Name = $ModuleName
                    Available = $true
                    Imported = $true
                    Module = $imported
                    Path = $candidate
                    Source = 'candidate'
                }
            } catch {}
        }

        return [pscustomobject]@{
            Name = $ModuleName
            Available = $false
            Imported = $false
            Module = $null
            Path = $null
            Source = 'missing'
        }
    }

    $resolvedRepoRoot = Resolve-AccelerationRepoRoot -ExplicitRepoRoot $RepoRoot
    $moduleStatus = [ordered]@{}

    foreach ($moduleName in @($Modules)) {
        if ([string]::IsNullOrWhiteSpace($moduleName)) { continue }
        $status = Import-AccelerationModule -ModuleName $moduleName -ResolvedRepoRoot $resolvedRepoRoot -ForceImport:$Force
        $moduleStatus[$moduleName] = $status
    }

    $result = [pscustomobject]@{
        TimestampUtc = [datetime]::UtcNow.ToString('o')
        RepoRoot = $resolvedRepoRoot
        Modules = [pscustomobject]$moduleStatus
        CommandLookupAvailable = [bool](Get-Command Find-CommandCached -ErrorAction SilentlyContinue)
        NativeCommandLookupAvailable = [bool](Get-Command Find-CommandNative -ErrorAction SilentlyContinue)
        DispatchAvailable = [bool](Get-Command Invoke-ProfileDispatch -ErrorAction SilentlyContinue)
        FileSearchAvailable = [bool](Get-Command Find-FilesFast -ErrorAction SilentlyContinue)
        ContentSearchAvailable = [bool](Get-Command Search-ContentFast -ErrorAction SilentlyContinue)
        NativeBridgeAvailable = [bool](Get-Command Test-PcaiNativeBridge -ErrorAction SilentlyContinue)
    }

    if ($RequireAll) {
        $missing = @($moduleStatus.GetEnumerator() | Where-Object { -not $_.Value.Available } | ForEach-Object Key)
        if ($missing.Count -gt 0) {
            throw "Required acceleration modules are unavailable: $($missing -join ', ')"
        }
    }

    return $result
}
