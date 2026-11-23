# Windows driver installation

[Install-DriverUpdate](../Modules/PC-AI.Drivers/Public/Install-DriverUpdate.ps1)
requires administrator elevation and a registry SHA256 digest of exactly 64 hex
characters before it downloads an installer for execution. The expected digest
must identify the exact vendor package. The
[bundled registry](../Config/driver-registry.json) leaves its current packages
unpinned, so they cannot be installed through this command.

Use a digest obtained from a trusted source to update the
[registry](../Modules/PC-AI.Drivers/Public/Update-DriverRegistry.ps1). Supply the
package URL and digest in the same edit. Changing the URL alone clears any
previous digest.

```powershell
Update-DriverRegistry -DeviceId $deviceId -RegistryPath $registryPath -DownloadUrl $vendorUrl -Sha256 $vendorDigest
Install-DriverUpdate -DeviceId $deviceId -RegistryPath $registryPath -WhatIf
```

`-WhatIf` previews the action without touching the destination or requiring
elevation. `-DownloadOnly` saves each unpinned acquisition to a new file for
manual verification. Pinned downloads use the expected digest. A mismatching
cached package is preserved. Use `-Force` to request a replacement. Windows
Update and inbox driver entries return a skip.

An actual install rechecks the package immediately before launch or extraction.
Use an administrator-controlled download directory because the verifier and
Windows installer open the package separately. Archive-backed installers expand into distinct
`install-*` subdirectories under that download directory. Their contents remain
available for inspection after the command completes.

The [Drivers Pester fixtures](../Tests/Unit/PC-AI.Drivers.Tests.ps1) use isolated
registries, local ZIPs with inert text entries, and mocks at the network and
installer launch boundaries. Run from the repository root with Pester 5.

```powershell
Invoke-Pester ./Tests/Unit/PC-AI.Drivers.Tests.ps1 -Tag Drivers
```

The [selector review register](../Tests/Unit/driver-install-review.json) binds
its entries to the test and installer source.
