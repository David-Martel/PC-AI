<#
.SYNOPSIS
    Retrieves and exports details of all network adapters in JSON format.

.DESCRIPTION
    This script uses the Get-NetAdapter cmdlet to retrieve information
    about all network adapters on the system and then exports this
    information as a JSON string to the console.

.NOTES
    - Requires administrator privileges to run for comprehensive information.
    - The output will be a JSON representation of the network adapter objects.
#>
# Retrieve all network adapters
$NetAdapters = Get-NetAdapter

# Select relevant properties for export
$AdapterInfo = $NetAdapters | Select-Object Name, InterfaceDescription, ifIndex, Status, MacAddress, LinkSpeed, PhysicalMediaType, ConnectorPresent, @{Name = 'IPAddress'; Expression = { (Get-NetIPAddress -InterfaceIndex $_.ifIndex -AddressFamily IPv4).IPAddress } }, @{Name = 'IPv6Address'; Expression = { (Get-NetIPAddress -InterfaceIndex $_.ifIndex -AddressFamily IPv6).IPAddress } }

# Convert the selected information to JSON format
$JsonOutput = $AdapterInfo | ConvertTo-Json -Depth 5

# Output the JSON to the console
Write-Host $JsonOutput
