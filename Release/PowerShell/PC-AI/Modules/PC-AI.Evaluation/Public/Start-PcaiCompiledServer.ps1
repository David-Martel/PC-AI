function Start-PcaiCompiledServer {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('llamacpp', 'mistralrs')]
        [string]$Backend,

        [Parameter(Mandatory)]
        [string]$ModelPath,

        [Parameter(Mandatory)]
        [string]$BaseUrl,

        [int]$GpuLayers = -1,

        [string]$Device,

        [int]$TimeoutSeconds = 60
    )

    if (-not (Test-Path $ModelPath)) {
        throw "Model file not found: $ModelPath"
    }

    $binaryPath = Get-PcaiCompiledBinaryPath -Backend $Backend
    if (-not $binaryPath) {
        throw "Compiled backend binary not found for $Backend. Build and copy to .local\\bin or update Config/llm-config.json evaluation.binSearchPaths."
    }

    $configPath = New-PcaiServerConfigFile -Backend $Backend -ModelPath $ModelPath -BaseUrl $BaseUrl -GpuLayers $GpuLayers -Device $Device

    $process = Start-Process -FilePath $binaryPath `
        -ArgumentList @('--config', $configPath) `
        -WorkingDirectory (Split-Path $binaryPath -Parent) `
        -NoNewWindow -PassThru

    $script:CompiledServerProcess = $process
    $script:CompiledServerConfigPath = $configPath
    $script:CompiledServerBaseUrl = $BaseUrl
    $script:CompiledServerBackend = $Backend

    Write-EvaluationEvent -Type 'backend_start' -Message "Starting compiled backend: $Backend" -Data @{
        backend = $Backend
        baseUrl = $BaseUrl
        pid = $process.Id
    }

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $null = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get -TimeoutSec 3 -ErrorAction Stop
            return $true
        } catch {
            Start-Sleep -Seconds 1
        }
    }

    try {
        if ($process -and (-not $process.HasExited)) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    } catch { }

    if ($configPath -and (Test-Path $configPath)) {
        Remove-Item $configPath -Force -ErrorAction SilentlyContinue
    }

    $script:CompiledServerProcess = $null
    $script:CompiledServerConfigPath = $null
    $script:CompiledServerBaseUrl = $null
    $script:CompiledServerBackend = $null

    throw "Compiled server for $Backend did not become ready within $TimeoutSeconds seconds."
}
