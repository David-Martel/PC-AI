<#
.SYNOPSIS
    Exports XML profiles for all Ethernet connections and converts them to JSON.

.DESCRIPTION
    This script iterates through all Ethernet adapters present on the system,
    exports their network profiles as XML files (named using the adapter name),
    and then converts each exported XML file to JSON format. Both XML and JSON
    files are saved in the current directory.

.NOTES
    - Requires administrator privileges to run.
    - Requires Python to be installed and accessible in the system's PATH for
      the XML to JSON conversion.
#>

# Get all Ethernet adapters
$EthernetAdapters = Get-NetAdapter | Where-Object { $_.PhysicalMediaType -eq "802.3" }

# Iterate through each Ethernet adapter
foreach ($Adapter in $EthernetAdapters) {
	$AdapterName = $Adapter.Name -replace "\s+", "-"
	$XmlFileName = "Ethernet-$AdapterName.xml"
	$JsonFileName = "Ethernet-$AdapterName.json"

	Write-Host "Exporting profile for adapter '$($Adapter.Name)' to '$XmlFileName'..."
	# Export the Ethernet profile using netsh
	netsh lan export profile folder="." interface="$($Adapter.Name)" > $null

	# Check if the XML file was created
	if (Test-Path $XmlFileName) {
		Write-Host "Converting '$XmlFileName' to '$JsonFileName'..."
		python -c "import xml.etree.ElementTree as ET; import json; import sys;

def elem_to_dict(elem):
    d = {}
    for child in elem:
        d[child.tag] = elem_to_dict(child) if len(list(child)) > 0 else (child.text or '')
    return d

xml_file = sys.argv[1]
json_file = sys.argv[2]

try:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = {'LANProfile' if root.tag == 'LANProfile' else 'UnknownRoot': elem_to_dict(root)}
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'Successfully converted {xml_file} to {json_file}')
except FileNotFoundError:
    print(f'Error: XML file {xml_file} not found.')
except ET.ParseError as e:
    print(f'Error parsing XML file {xml_file}: {e}')
except Exception as e:
    print(f'An unexpected error occurred: {e}')

" $XmlFileName $JsonFileName

		# Optional: Remove the XML file after conversion
		# Remove-Item $XmlFileName -Force
	}
 else {
		Write-Warning "Could not export profile for adapter '$($Adapter.Name)'."
	}
}

Write-Host "Script finished."
