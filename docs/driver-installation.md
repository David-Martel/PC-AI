# Windows driver installation

## Install

Load the signed PC-AI Drivers module in Windows PowerShell 7. An administrator
provisions the local Program Files directory `PC-AI.DriverTrust` and its
`driver-registry.json`. Administrators or SYSTEM must own both. Give regular
users read access only. The [trust check](../Modules/PC-AI.Drivers/Private/Assert-DriverInstallTrust.ps1)
rejects writable or replaceable paths, and checks parent ownership and
permissions back to the drive root.

Use an administrative shell with the installed module when building the machine
registry. Review the package URL, SHA256, and publisher before publishing an
entry. The machine registry uses the [bundled registry](../Config/driver-registry.json)
schema with one additional field in each authorized entry's `driver` object.

- `sha256` is the SHA256 of the exact vendor download, including the ZIP when applicable.
- `publisherCertificateSha256` is the SHA256 of the authorized signer's DER
  certificate. Obtain the certificate through trusted vendor verification.
  Windows must also return a valid Authenticode signature for the staged payload.

Use the [registry editor](../Modules/PC-AI.Drivers/Public/Update-DriverRegistry.ps1)
to update an existing device's URL and package digest in one edit. A URL change
without an accompanying digest invalidates its previous digest. Update the
publisher pin through administrative configuration when the vendor rotates
its signing certificate.

## Build

No native build is required. The
[Drivers module](../Modules/PC-AI.Drivers/PC-AI.Drivers.psm1) loads the
installer and its private trust functions on import.

## Test

Run the [Drivers Pester fixtures](../Tests/Unit/PC-AI.Drivers.Tests.ps1) from the
repository root. The fixtures use local inert files, synthetic ACLs, and
signature mocks.

```powershell
Invoke-Pester ./Tests/Unit/PC-AI.Drivers.Tests.ps1 -Tag Drivers
```

See the [trust review record](../Tests/Unit/driver-install-trust-review.json)
for exact source and fixture bindings.

## Run

[Install-DriverUpdate](../Modules/PC-AI.Drivers/Public/Install-DriverUpdate.ps1)
uses the machine registry for execution. It downloads into a new
administrator-controlled stage, checks the exact staged package digest, and
checks the signature of the executable or MSI. An executable ZIP must contain
exactly one EXE. The installer removes the verified stage on completion and
reports any cleanup failure. INF archives are available for manual acquisition
until the installer can verify their catalog and publisher association.

```powershell
$registryPath = Join-Path ([Environment]::GetFolderPath('ProgramFiles')) 'PC-AI.DriverTrust/driver-registry.json'
Install-DriverUpdate -DeviceId $deviceId -RegistryPath $registryPath -WhatIf
Install-DriverUpdate -DeviceId $deviceId -Confirm
```

`-WhatIf` previews the specified registry entry without touching the
destination or requiring elevation. `-DownloadOnly` saves packages to the
requested cache or the default temporary download directory. It verifies a
digest when one is present and keeps an existing mismatching file unless
`-Force` is used. The installation route never executes the manual cache.
Inspect Windows Update and inbox driver entries through `-WhatIf` or
`-DownloadOnly` to receive a read-only skip result.
