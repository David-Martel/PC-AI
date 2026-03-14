#Requires -Version 5.1

function Invoke-TrustedDownload {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)]
        [string]$Url,

        [Parameter(Mandatory)]
        [string]$OutFile,

        [Parameter()]
        [string[]]$TrustedHosts,

        [Parameter()]
        [string]$ExpectedSha256,

        [Parameter()]
        [switch]$ForceDownload
    )

    $uri = [Uri]$Url

    if ($TrustedHosts -and $TrustedHosts.Count -gt 0) {
        $hostMatch = $false
        foreach ($trusted in $TrustedHosts) {
            if ($uri.Host -eq $trusted -or $uri.Host.EndsWith(".$trusted")) {
                $hostMatch = $true
                break
            }
        }
        if (-not $hostMatch) {
            Write-Error "Host '$($uri.Host)' is not in the trusted hosts list."
            return $null
        }
    }

    if ((Test-Path $OutFile) -and (-not $ForceDownload)) {
        Write-Verbose "File already exists, skipping download: $OutFile"
        return $OutFile
    }

    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing -ErrorAction Stop

        if ($ExpectedSha256) {
            $hash = (Get-FileHash -Path $OutFile -Algorithm SHA256).Hash
            if ($hash -ne $ExpectedSha256.ToUpper()) {
                Write-Error "SHA256 mismatch for '$OutFile'. Expected: $ExpectedSha256  Got: $hash"
                Remove-Item -Path $OutFile -Force -ErrorAction SilentlyContinue
                return $null
            }
            Write-Verbose "SHA256 verified: $hash"
        }

        return $OutFile
    }
    catch {
        Write-Error "Download failed for '$Url': $_"
        if (Test-Path $OutFile) {
            Remove-Item -Path $OutFile -Force -ErrorAction SilentlyContinue
        }
        return $null
    }
}
