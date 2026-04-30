$ErrorActionPreference = 'SilentlyContinue'
$since = (Get-Date).AddDays(-3)

# Get Hyper-V Compute errors (this is where VSock issues would show up)
$compute = Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-Hyper-V-Compute-Admin'; Level=2,3; StartTime=$since} -MaxEvents 20
if ($compute) {
    Write-Host "=== Hyper-V Compute Errors/Warnings ===" -ForegroundColor Red
    $compute | ForEach-Object {
        "[$($_.TimeCreated)] Level:$($_.LevelDisplayName) ID:$($_.Id)"
        $_.Message.Substring(0, [Math]::Min(250, $_.Message.Length))
        "-" * 80
    }
} else {
    Write-Host "No Hyper-V Compute errors in last 3 days"
}

Write-Host "`n"

# Get .NET Runtime OutOfMemory
$dotnet = Get-WinEvent -FilterHashtable @{LogName='Application'; ProviderName='.NET Runtime'; Level=2; StartTime=$since} -MaxEvents 20 | Where-Object { $_.Message -match 'OutOfMemory' }
if ($dotnet) {
    Write-Host "=== .NET OutOfMemory Errors ===" -ForegroundColor Red
    $dotnet | ForEach-Object {
        "[$($_.TimeCreated)] ID:$($_.Id)"
        $_.Message.Substring(0, [Math]::Min(250, $_.Message.Length))
        "-" * 80
    }
} else {
    Write-Host "No .NET OutOfMemory errors in last 3 days"
}

Write-Host "`n"

# Get Docker/vmcompute errors
$virt = Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=$since} -MaxEvents 1000 | Where-Object { ($_.ProviderName -match 'vmcompute|docker|hcs') -and ($_.LevelDisplayName -in 'Error','Warning') } | Select-Object -First 10
if ($virt) {
    Write-Host "=== Virtualization Stack Errors ===" -ForegroundColor Red
    $virt | ForEach-Object {
        "[$($_.TimeCreated)] $($_.ProviderName) Level:$($_.LevelDisplayName) ID:$($_.Id)"
        $_.Message.Substring(0, [Math]::Min(250, $_.Message.Length))
        "-" * 80
    }
} else {
    Write-Host "No virtualization stack errors in last 3 days"
}
