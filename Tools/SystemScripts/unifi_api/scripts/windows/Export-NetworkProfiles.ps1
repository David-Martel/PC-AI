<#
.SYNOPSIS
    Exports all Wi-Fi and Ethernet network profiles and converts them to JSON. (Corrected Ethernet Logic)

.DESCRIPTION
    This script exports all available Wi-Fi and Ethernet network profiles
    using netsh and then converts each exported XML file to JSON format.
    Both XML and JSON files are saved in the current directory.

.NOTES
    - Requires administrator privileges to run.
    - Requires Python to be installed and accessible in the system's PATH for
      the XML to JSON conversion.
#>

# Export Wi-Fi Profiles
Write-Host "Exporting Wi-Fi profiles..."
$WifiProfiles = netsh wlan show profiles | Select-String -Pattern "^    All User Profile" | ForEach-Object { $_.Line.Substring(26).Trim() }

foreach ($ProfileName in $WifiProfiles) {
	$SafeProfileName = $ProfileName -replace "\s+", "-"
	$XmlFileName = "Wi-Fi-$SafeProfileName.xml"
	$JsonFileName = "Wi-Fi-$SafeProfileName.json"

	Write-Host "Exporting Wi-Fi profile '$ProfileName' to '$XmlFileName'..."
	netsh wlan export profile name="$ProfileName" folder="." > $null

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
    data = {'WLANProfile' if root.tag == 'WLANProfile' else 'LANProfile': elem_to_dict(root)}
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
	}
 else {
		Write-Warning "Could not export Wi-Fi profile '$ProfileName'."
	}
}

# Export Ethernet Profiles
Write-Host "Exporting Ethernet profiles..."
$EthernetAdapters = Get-NetAdapter | Where-Object { $_.PhysicalMediaType -eq "802.3" }

foreach ($Adapter in $EthernetAdapters) {
	$AdapterName = $Adapter.Name -replace "\s+", "-"
	$XmlFileName = "Ethernet-$AdapterName.xml"
	$JsonFileName = "Ethernet-$AdapterName.json"

	Write-Host "Exporting Ethernet profile for adapter '$($Adapter.Name)' to '$XmlFileName'..."
	netsh lan export profile folder="." interface="$($Adapter.Name)" > $null

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
    data = {'WLANProfile' if root.tag == 'WLANProfile' else 'LANProfile': elem_to_dict(root)}
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
	}
 else {
		Write-Warning "Could not export profile for adapter '$($Adapter.Name)'."
	}
}

Write-Host "Script finished."
