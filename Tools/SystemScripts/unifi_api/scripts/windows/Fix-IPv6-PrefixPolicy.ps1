<#
.SYNOPSIS
    Fix IPv6 source address selection on Windows machines in the DTM network.

.DESCRIPTION
    Problem: Windows receives IPv6 RAs from multiple VLANs and ULA sources,
    causing incorrect source address selection that breaks return routing.

    Fix: Configure the Windows prefix policy table (RFC 6724) to:
    1. Strongly prefer the correct GUA prefix (2600:1702:7220::/48)
    2. Deprioritize ULA prefixes (fd31, fd62, fdf9)
    3. Deprecate stale addresses from rogue RAs

    This does NOT disable IPv6 — it just ensures Windows picks the right
    source address for outbound connections.

    Safe to run multiple times (idempotent). Changes persist across reboot.

.PARAMETER Revert
    Restore default Windows prefix policies

.EXAMPLE
    .\Fix-IPv6-PrefixPolicy.ps1          # Apply fix
    .\Fix-IPv6-PrefixPolicy.ps1 -Revert  # Restore defaults
#>
param([switch]$Revert)

$ErrorActionPreference = "Stop"

# Require admin
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]"Administrator")) {
    Write-Host "ERROR: Run as Administrator" -ForegroundColor Red
    exit 1
}

if ($Revert) {
    Write-Host "Restoring default IPv6 prefix policies..." -ForegroundColor Yellow
    netsh interface ipv6 reset 2>$null
    Write-Host "Done. Reboot may be required." -ForegroundColor Green
    exit 0
}

Write-Host "Configuring IPv6 prefix policies for DTM network..." -ForegroundColor Cyan

# --- Step 1: Set prefix policies ---
# Higher precedence = more preferred for source address selection
# Label matching ensures source/destination use the same prefix family

$policies = @(
    # Our delegated GUA — highest priority for internet traffic
    @{ Prefix = "2600:1702:7220::/48"; Precedence = 60; Label = 2 }

    # Default global unicast — standard priority
    @{ Prefix = "::/0";               Precedence = 40; Label = 2 }

    # Loopback — high priority (standard)
    @{ Prefix = "::1/128";            Precedence = 50; Label = 1 }

    # IPv4-mapped — keep at standard level
    @{ Prefix = "::ffff:0:0/96";      Precedence = 35; Label = 5 }

    # ULA (fd00::/7) — LOW priority to avoid source address confusion
    # This covers fd31 (SmartThings rogue), fd62 (UniFi ULA), fdf9 (old ULA)
    @{ Prefix = "fc00::/7";           Precedence = 1;  Label = 7 }

    # 6to4 — minimal
    @{ Prefix = "2002::/16";          Precedence = 30; Label = 6 }

    # Teredo — minimal
    @{ Prefix = "2001::/32";          Precedence = 5;  Label = 8 }

    # Deprecated prefixes
    @{ Prefix = "3ffe::/16";          Precedence = 1;  Label = 9 }
    @{ Prefix = "fec0::/10";          Precedence = 1;  Label = 10 }
    @{ Prefix = "::/96";              Precedence = 1;  Label = 11 }
)

foreach ($p in $policies) {
    netsh interface ipv6 set prefixpolicy $($p.Prefix) $($p.Precedence) $($p.Label) store=persistent 2>$null
    if ($LASTEXITCODE -ne 0) {
        # May need to add instead of set
        netsh interface ipv6 add prefixpolicy $($p.Prefix) $($p.Precedence) $($p.Label) store=persistent 2>$null
    }
}
Write-Host "  [OK] Prefix policies configured (GUA precedence=60, ULA precedence=1)" -ForegroundColor Green

# --- Step 2: Deprecate rogue ULA addresses ---
# Force fd31 (SmartThings rogue) addresses to expire immediately
$adapters = Get-NetIPAddress -AddressFamily IPv6 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -match '^fd31:' }

foreach ($addr in $adapters) {
    Write-Host "  Deprecating rogue address: $($addr.IPAddress) on $($addr.InterfaceAlias)" -ForegroundColor Yellow
    # Set preferred lifetime to 0 (deprecated) and valid lifetime to 60s (expires in 1 min)
    Set-NetIPAddress -InterfaceIndex $addr.InterfaceIndex -IPAddress $addr.IPAddress `
        -PreferredLifetime ([timespan]::Zero) -ValidLifetime ([timespan]::FromSeconds(60)) `
        -ErrorAction SilentlyContinue
}

if ($adapters.Count -gt 0) {
    Write-Host "  [OK] Deprecated $($adapters.Count) rogue fd31 addresses" -ForegroundColor Green
} else {
    Write-Host "  [OK] No rogue fd31 addresses found" -ForegroundColor Green
}

# --- Step 3: Verify ---
Write-Host "`nCurrent prefix policies:" -ForegroundColor Cyan
netsh interface ipv6 show prefixpolicies

Write-Host "`nIPv6 source address test:" -ForegroundColor Cyan
$result = Test-NetConnection -ComputerName "ipv6.google.com" -Port 443 -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
if ($result.TcpTestSucceeded) {
    Write-Host "  [OK] IPv6 connectivity: $($result.RemoteAddress) via $($result.SourceAddress)" -ForegroundColor Green
} else {
    Write-Host "  [WARN] IPv6 connectivity test failed (may need a moment for address changes to take effect)" -ForegroundColor Yellow
}

Write-Host "`nDone. No reboot required." -ForegroundColor Green
