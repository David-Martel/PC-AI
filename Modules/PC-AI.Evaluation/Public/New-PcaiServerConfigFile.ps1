function New-PcaiServerConfigFile {
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

        [string]$Device
    )

    $uri = [Uri]$BaseUrl
    $serverPort = if ($uri.Port -gt 0) { $uri.Port } else { 8080 }

    $backendType = if ($Backend -eq 'llamacpp') { 'llama_cpp' } else { $Backend }
    $config = @{
        backend = @{
            type = $backendType
        }
        model = @{
            path = $ModelPath
            generation = @{
                max_tokens = 512
                temperature = 0.7
                top_p = 0.95
                stop = @()
            }
        }
        server = @{
            host = $uri.Host
            port = $serverPort
            cors = $true
        }
    }

    if ($Backend -eq 'llamacpp' -and $GpuLayers -ge 0) {
        $config.backend.n_gpu_layers = $GpuLayers
    }

    if ($Backend -eq 'mistralrs' -and $Device) {
        $config.backend.device = $Device
    }

    $configPath = Join-Path ([IO.Path]::GetTempPath()) ("pcai-{0}-{1}.json" -f $Backend, [guid]::NewGuid().ToString('N'))
    $config | ConvertTo-Json -Depth 6 | Set-Content -Path $configPath -Encoding UTF8
    return $configPath
}
