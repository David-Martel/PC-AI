#Requires -Version 5.1

function Test-NvidiaDownloadUrl {
<#
.SYNOPSIS
    Validates that a given NVIDIA component download URL is trusted and reachable.

.DESCRIPTION
    Checks the supplied URL against an allow-list of known NVIDIA hosts and,
    unless -SkipReachabilityCheck is specified, sends an HTTP HEAD request to
    confirm the resource is accessible.

    The function returns a PSCustomObject rather than a bare [bool] so callers
    can inspect the specific reason for failure (untrusted host, non-200 status,
    etc.) without needing to parse warning messages.

    Trusted host suffixes (any subdomain is accepted):
        nvidia.com
        developer.nvidia.com
        developer.download.nvidia.com
        us.download.nvidia.com

    HTTP status codes treated as success:
        200 OK
        302 Found  (CDN redirect — content exists at the redirect target)

.PARAMETER Url
    The HTTPS URL of the NVIDIA component installer or archive to validate.

.PARAMETER SkipReachabilityCheck
    When specified, only the host trust check is performed. No network request
    is made. IsValid reflects trust status alone; StatusCode and ContentLength
    are set to -1.

.OUTPUTS
    [PSCustomObject] with properties:
        Url            - The URL that was evaluated.
        IsValid        - $true when the URL is trusted and reachable.
        IsTrusted      - $true when the host is in the trusted-host list.
        StatusCode     - HTTP status code returned by the HEAD request, or -1
                         when the check was skipped or a network error occurred.
        ContentLength  - Value of the Content-Length response header in bytes,
                         or -1 when absent or skipped.
        ContentType    - Value of the Content-Type response header, or $null.

.EXAMPLE
    Test-NvidiaDownloadUrl -Url 'https://us.download.nvidia.com/tesla/550.90.07/NVIDIA-Linux-x86_64-550.90.07.run'
    Validates that the URL is on a trusted NVIDIA host and returns 200 OK.

.EXAMPLE
    Test-NvidiaDownloadUrl -Url 'https://developer.nvidia.com/cuda-downloads' -SkipReachabilityCheck
    Returns trust status without making a network request.

.EXAMPLE
    $result = Test-NvidiaDownloadUrl -Url $entry.downloadUrl
    if (-not $result.IsValid) { throw "URL validation failed for $($result.Url)" }

.NOTES
    Phase 3 implementation.
    The HEAD request uses a 30-second timeout. If the server returns a redirect
    (302), the function reports IsValid = $true without following the redirect,
    since CDN-fronted NVIDIA downloads routinely use temporary redirects.
    UseBasicParsing is set to avoid IE engine dependency on Server Core.
#>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory)]
        [ValidateNotNullOrEmpty()]
        [string]$Url,

        [Parameter()]
        [switch]$SkipReachabilityCheck
    )

    $ErrorActionPreference = 'Stop'

    # Trusted NVIDIA host suffixes — any subdomain is accepted.
    [string[]]$trustedHosts = @(
        'nvidia.com',
        'developer.nvidia.com',
        'developer.download.nvidia.com',
        'us.download.nvidia.com'
    )

    # --- Build result skeleton with safe defaults ---
    $result = [PSCustomObject]@{
        Url           = $Url
        IsValid       = $false
        IsTrusted     = $false
        StatusCode    = -1
        ContentLength = -1
        ContentType   = $null
    }

    # --- Host trust check ---
    try {
        $uri = [System.Uri]::new($Url)

        foreach ($trusted in $trustedHosts) {
            if ($uri.Host -eq $trusted -or $uri.Host.EndsWith(".$trusted")) {
                $result.IsTrusted = $true
                break
            }
        }
    }
    catch {
        Write-Warning "Test-NvidiaDownloadUrl: URL parse failed for '$Url': $($_.Exception.Message)"
        return $result
    }

    if (-not $result.IsTrusted) {
        Write-Warning ("Test-NvidiaDownloadUrl: Host '$($uri.Host)' is not in the trusted NVIDIA host list. " +
            "Trusted: $($trustedHosts -join ', ')")
        return $result
    }

    # --- Skip network check if requested ---
    if ($SkipReachabilityCheck) {
        Write-Verbose "Test-NvidiaDownloadUrl: Reachability check skipped (host trust only)."
        $result.IsValid = $true
        return $result
    }

    # --- HTTP HEAD request ---
    Write-Verbose "Test-NvidiaDownloadUrl: Sending HEAD request to '$Url'..."
    try {
        # Disable automatic redirect following so we can inspect the 302 directly.
        $webRequest = [System.Net.HttpWebRequest]::Create($Url)
        $webRequest.Method              = 'HEAD'
        $webRequest.Timeout             = 30000  # 30 seconds
        $webRequest.AllowAutoRedirect   = $false
        $webRequest.UserAgent           = 'PC-AI/1.0 NvidiaInstaller (+https://github.com/David-Martel/PC-AI)'

        try {
            $response = $webRequest.GetResponse()
        }
        catch [System.Net.WebException] {
            # WebException is thrown for non-2xx/3xx codes; extract the response.
            if ($null -ne $_.Exception.Response) {
                $response = $_.Exception.Response
            }
            else {
                throw
            }
        }

        $result.StatusCode = [int]$response.StatusCode

        # Content-Length may be absent (-1 from the framework for chunked responses)
        $contentLen = $response.ContentLength
        if ($contentLen -ge 0) {
            $result.ContentLength = $contentLen
        }

        $result.ContentType = $response.ContentType

        if ($response -is [System.Net.HttpWebResponse]) {
            $response.Close()
        }

        Write-Verbose "Test-NvidiaDownloadUrl: StatusCode=$($result.StatusCode) ContentLength=$($result.ContentLength)"

        # 200 OK or 302 Found are both valid
        if ($result.StatusCode -eq 200 -or $result.StatusCode -eq 302) {
            $result.IsValid = $true
        }
        else {
            Write-Warning ("Test-NvidiaDownloadUrl: URL '$Url' returned HTTP $($result.StatusCode). " +
                "Expected 200 or 302.")
        }
    }
    catch {
        Write-Warning "Test-NvidiaDownloadUrl: HEAD request failed for '$Url': $($_.Exception.Message)"
        # StatusCode remains -1, IsValid remains $false
    }

    return $result
}
