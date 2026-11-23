function Get-PcaiOriginBearerHeader {
    <#
    .SYNOPSIS
        Returns an Authorization bearer header for a request, only if it targets the server the token belongs to.
    .DESCRIPTION
        pcai-inference and pcai-media require a bearer token on every route but GET /health when one
        is configured (docs/http-server-security.md). Clients read that token from the same
        environment variable the server reads (PCAI_INFERENCE_TOKEN, PCAI_MEDIA_TOKEN).

        The token is a credential for one server, so it is attached only when the request URL has the
        same origin (scheme, host and port) as the URL that server is configured at. A request to
        any other origin, such as an Ollama or LM Studio endpoint, gets no header, and so does a
        request when the variable is unset or blank, or when either URL is not an absolute http(s)
        URL. This is the single implementation shared by the LLM module, the evaluation harness and
        the media hardware runner.
    .PARAMETER RequestUrl
        The URL about to be requested.
    .PARAMETER BaseUrl
        The configured base URL of the server that owns the token.
    .PARAMETER TokenVariable
        Name of the environment variable that holds the token.
    .OUTPUTS
        A hashtable with an Authorization entry, or an empty hashtable.
    .EXAMPLE
        $headers = Get-PcaiOriginBearerHeader -RequestUrl "$base/v1/completions" -BaseUrl $base -TokenVariable PCAI_INFERENCE_TOKEN
    #>
    [CmdletBinding()]
    [OutputType([hashtable])]
    param(
        [Parameter(Mandatory)]
        [AllowNull()][AllowEmptyString()]
        [string]$RequestUrl,

        [Parameter(Mandatory)]
        [AllowNull()][AllowEmptyString()]
        [string]$BaseUrl,

        [Parameter(Mandatory)]
        [ValidatePattern('^[A-Za-z_][A-Za-z0-9_]*$')]
        [string]$TokenVariable
    )

    $token = [Environment]::GetEnvironmentVariable($TokenVariable)
    if ([string]::IsNullOrWhiteSpace($token)) { return @{} }

    $request = $null
    $base = $null
    if (-not [uri]::TryCreate($RequestUrl, [UriKind]::Absolute, [ref]$request)) { return @{} }
    if (-not [uri]::TryCreate($BaseUrl, [UriKind]::Absolute, [ref]$base)) { return @{} }
    foreach ($candidate in $request, $base) {
        if ($candidate.Scheme -notin 'http', 'https') { return @{} }
    }

    $sameOrigin = $request.Scheme -eq $base.Scheme -and
        [string]::Equals($request.IdnHost, $base.IdnHost, [StringComparison]::OrdinalIgnoreCase) -and
        $request.Port -eq $base.Port
    if (-not $sameOrigin) { return @{} }

    return @{ Authorization = "Bearer $($token.Trim())" }
}
