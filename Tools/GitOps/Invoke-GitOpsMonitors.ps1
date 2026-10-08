#requires -Version 7.0
<#
.SYNOPSIS
Run bounded GitOps inspections and publish through the configured agent-bus client.
.DESCRIPTION
The detached worker retains inspection and publication failures in its local report.
LibraryOnly loads shared helpers without starting work.
#>
[CmdletBinding()]
param(
    [string]$Repo,
    [ValidateRange(0, 300)][int]$DelaySeconds = 20,
    [switch]$NoDelay,
    [string]$BusUrl,
    [string]$RepoRoot = (Join-Path $PSScriptRoot '../..'),
    [string]$OutputDirectory,
    [ValidateRange(1, 120)][int]$RequestTimeoutSeconds = 15,
    [switch]$LibraryOnly
)

function Invoke-GitOpsProcess {
    [CmdletBinding()]
    param([string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory, [int]$TimeoutSeconds)
    try {
        $runner = Get-Command Invoke-VigilBoundedProcess -ErrorAction SilentlyContinue
        $helper = Join-Path $PSScriptRoot '../SystemScripts/Networking/Invoke-VigilBoundedProcess.ps1'
        if (-not $runner -and $IsWindows -and (Test-Path -LiteralPath $helper)) {
            . $helper
            $runner = Get-Command Invoke-VigilBoundedProcess -ErrorAction Stop
        }
        if ($runner) {
            return Invoke-VigilBoundedProcess -FilePath $FilePath -Arguments $Arguments -WorkingDirectory $WorkingDirectory -TimeoutSeconds $TimeoutSeconds
        }
        # This portable fallback owns only the finite request and its descendants.
        $info = [Diagnostics.ProcessStartInfo]::new()
        $info.FileName = $FilePath
        $info.WorkingDirectory = $WorkingDirectory
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        foreach ($argument in $Arguments) { $info.ArgumentList.Add($argument) }
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $info
        $deadline = [Diagnostics.Stopwatch]::StartNew()
        try {
            if (-not $process.Start()) { throw 'Process launch failed.' }
            $stdout = $process.StandardOutput.ReadToEndAsync()
            $stderr = $process.StandardError.ReadToEndAsync()
            if (-not $process.WaitForExit([Math]::Max(1, $TimeoutSeconds * 1000 - [int]$deadline.ElapsedMilliseconds))) {
                $process.Kill($true)
                $null = $process.WaitForExit(1000)
                throw 'Request deadline exceeded.'
            }
            $capture = [Threading.Tasks.Task]::WhenAll([Threading.Tasks.Task[]]@($stdout, $stderr))
            if (-not $capture.Wait([Math]::Max(1, $TimeoutSeconds * 1000 - [int]$deadline.ElapsedMilliseconds))) { throw 'Output capture deadline exceeded.' }
            return [pscustomobject]@{ ExitCode = $process.ExitCode; Stdout = $stdout.GetAwaiter().GetResult(); Stderr = $stderr.GetAwaiter().GetResult() }
        }
        finally { $process.Dispose() }
    }
    catch {
        # Native stderr and exceptions may contain credentials or private paths.
        return [pscustomobject]@{ ExitCode = -1; Stdout = ''; Stderr = ''; InspectionError = 'Request launch or deadline failed.'; ErrorKind = $_.Exception.GetType().Name }
    }
}

function Get-GitOpsResponse {
    [CmdletBinding()]
    param([string]$Operation, [string[]]$Arguments, [string]$WorkingDirectory, [int]$TimeoutSeconds, [ValidateSet('Array', 'Object')][string]$Shape)
    $response = Invoke-GitOpsProcess -FilePath 'gh' -Arguments $Arguments -WorkingDirectory $WorkingDirectory -TimeoutSeconds $TimeoutSeconds
    $status = 'Available'
    $reason = $null
    $value = $null
    if ($response.ExitCode -ne 0) {
        $status = 'Unavailable'
        $reason = "GitHub inspection failed (exit $($response.ExitCode))."
        if ($Operation -eq 'code-scanning' -and $response.Stderr -match 'HTTP 404' -and $response.Stderr -match 'Code scanning is not enabled') {
            $status = 'Unsupported'
            $reason = 'Code scanning is not enabled.'
        }
    }
    else {
        try {
            $value = ConvertFrom-Json -InputObject $response.Stdout -NoEnumerate -ErrorAction Stop
            if ($null -eq $value -or ($Shape -eq 'Array' -and $value -isnot [array]) -or ($Shape -eq 'Object' -and $value -isnot [pscustomobject])) { throw 'Unexpected response shape.' }
        }
        catch {
            $status = 'Unavailable'
            $reason = 'GitHub inspection returned missing or invalid JSON.'
            $value = $null
        }
    }
    [pscustomobject]@{ operation = $Operation; status = $status; exit_code = $response.ExitCode; reason = $reason; value = $value }
}

function Write-GitOpsReport {
    [CmdletBinding()]
    param([object]$Report, [string]$Directory, [string]$Name)
    $null = New-Item -ItemType Directory -Path $Directory -Force -ErrorAction Stop
    $destination = Join-Path $Directory $Name
    $temporary = Join-Path $Directory ('.gitops-' + [guid]::NewGuid().ToString('N') + '.tmp')
    try {
        [IO.File]::WriteAllText($temporary, ($Report | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
        [IO.File]::Move($temporary, $destination, $true)
    }
    finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -ErrorAction Stop }
    }
}

if ($LibraryOnly) { return }
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $RepoRoot 'Reports/gitops' }
$keyPath = if ($IsWindows) { $RepoRoot.ToLowerInvariant() } else { $RepoRoot }
$hasher = [Security.Cryptography.SHA256]::Create()
try { $leaseKey = [BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($keyPath))).Replace('-', '') }
finally { $hasher.Dispose() }
$lease = [Threading.Mutex]::new($false, "Local\PC_AI_GitOps_$leaseKey")
$acquired = $false
$combined = $null
$exitCode = 2
try {
    try { $acquired = $lease.WaitOne(0) } catch [Threading.AbandonedMutexException] { $acquired = $true }
    if (-not $acquired) { 'gitops-monitors: existing worker owns this checkout'; exit 0 }
    if (-not $NoDelay -and $DelaySeconds -gt 0) { Start-Sleep -Seconds $DelaySeconds }
    $configurationError = $null
    if ($BusUrl) { $configurationError = 'BusUrl is unsupported; configure agent-bus candidates and their credential sources.' }
    if (-not $Repo -and -not $configurationError) {
        $identity = Get-GitOpsResponse -Operation repository -Arguments @('repo', 'view', '--json', 'nameWithOwner') -WorkingDirectory $RepoRoot -TimeoutSeconds $RequestTimeoutSeconds -Shape Object
        if ($identity.status -eq 'Available') { $Repo = $identity.value.nameWithOwner }
    }
    if ($Repo -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') { $configurationError = 'Repository identity is unavailable or invalid.' }
    $combined = [ordered]@{
        repo = $Repo; ts_utc = [datetime]::UtcNow.ToString('o'); inspection_status = 'Unavailable'; configuration_error = $configurationError
        workflow_problems = $null; upstream_items = $null; detail = @{}
        publication = @{ status = 'NotAttempted'; exit_code = $null; reason = $null }
    }
    if (-not $configurationError) {
        $wh = & (Join-Path $PSScriptRoot 'Watch-WorkflowHealth.ps1') -Repo $Repo -RepoRoot $RepoRoot -OutputDirectory $OutputDirectory -RequestTimeoutSeconds $RequestTimeoutSeconds -Quiet | Select-Object -Last 1 | ConvertFrom-Json -ErrorAction Stop
        $ur = & (Join-Path $PSScriptRoot 'Get-UpstreamReviews.ps1') -Repo $Repo -RepoRoot $RepoRoot -OutputDirectory $OutputDirectory -RequestTimeoutSeconds $RequestTimeoutSeconds -Quiet | Select-Object -Last 1 | ConvertFrom-Json -ErrorAction Stop
        if ($wh.inspection_status -notin @('Available', 'Unavailable') -or $ur.inspection_status -notin @('Available', 'Unavailable') -or $null -eq $wh.problem_count -or $null -eq $ur.total) { throw 'Missing child inspection metadata.' }
        $combined.workflow_problems = $wh.problem_count
        $combined.upstream_items = $ur.total
        $combined.detail = @{ workflow = $wh; upstream = $ur }
        $combined.inspection_status = if ($wh.inspection_status -eq 'Available' -and $ur.inspection_status -eq 'Available') { 'Available' } else { 'Unavailable' }
        $exitCode = if ($combined.inspection_status -eq 'Unavailable') { 2 } elseif ($wh.problem_count -gt 0 -or $ur.total -gt 0) { 1 } else { 0 }
        if ($exitCode -ne 0) {
            $client = if ($IsWindows) { Join-Path $HOME 'bin/agent-bus.exe' } else { Join-Path $HOME '.local/bin/agent-bus' }
            if (-not (Test-Path -LiteralPath $client)) {
                $application = Get-Command agent-bus -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
                if ($application) { $client = $application.Source }
            }
            $message = "GitOps monitor [$Repo]: inspection=$($combined.inspection_status), workflows=$($wh.problem_count), upstream=$($ur.total). See the checkout GitOps report."
            $sender = 'gitops-monitor-{0}-{1}' -f [Environment]::MachineName.ToLowerInvariant(), $leaseKey.Substring(0, 12)
            $publication = Invoke-GitOpsProcess -FilePath $client -Arguments @('send', '--from-agent', $sender, '--to-agent', 'all', '--topic', 'status', '--body', $message, '--tag', "repo:$($Repo.Split('/')[1])", '--encoding', 'compact') -WorkingDirectory $RepoRoot -TimeoutSeconds $RequestTimeoutSeconds
            $combined.publication = @{ status = if ($publication.ExitCode -eq 0) { 'Published' } else { 'Failed' }; exit_code = $publication.ExitCode; reason = if ($publication.ExitCode -eq 0) { $null } else { 'Configured agent-bus client failed; inspect its credential/routing configuration.' } }
            if ($publication.ExitCode -ne 0) { $exitCode = 2; Write-Warning 'GitOps publication failed; details are in the local report.' }
        }
    }
}
catch {
    if (-not $combined) { $combined = [ordered]@{ repo = $Repo; ts_utc = [datetime]::UtcNow.ToString('o') } }
    $combined.inspection_status = 'Unavailable'
    $combined.inspection_error = 'GitOps inspection or report preparation failed.'
    $exitCode = 2
}
finally {
    if ($acquired) {
        try {
            $reportName = if ($Repo -match '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') { 'monitors-latest-' + $Repo.Replace('/', '_') + '.json' } else { 'monitors-latest-unavailable.json' }
            Write-GitOpsReport -Report $combined -Directory $OutputDirectory -Name $reportName
        }
        catch {
            Write-Warning 'GitOps report could not be persisted; inspection is unavailable.'
            $exitCode = 2
        }
        finally { $lease.ReleaseMutex() }
    }
    $lease.Dispose()
}
"gitops-monitors $Repo : inspection=$($combined.inspection_status) publication=$($combined.publication.status)"
exit $exitCode
