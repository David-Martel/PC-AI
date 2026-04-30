<#
.SYNOPSIS
    Legacy wrapper — delegates to Invoke-DtmSsh.ps1 with -HostAlias udmpro.

.DESCRIPTION
    For new code, use Invoke-DtmSsh.ps1 directly:
      . .\scripts\windows\Invoke-DtmSsh.ps1
      Invoke-DtmSsh -HostAlias dtm-work -Command "whoami"
#>

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptDir "Invoke-DtmSsh.ps1")

# Invoke-UdmSsh is already defined by Invoke-DtmSsh.ps1 as a backward-compat wrapper
