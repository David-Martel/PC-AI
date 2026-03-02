function Get-PcaiNativeUsbDiagnostics {
    return [PcaiNative.PcaiCore]::GetUsbDeepDiagnostics()
}
