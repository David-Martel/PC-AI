#!/usr/bin/env pwsh
# Quick test: load pcai_media.dll and check FFI exports
$dllPath = Join-Path $PSScriptRoot "..\bin\pcai_media.dll"
if (-not (Test-Path $dllPath)) {
    Write-Error "DLL not found at $dllPath"
    exit 1
}

$handle = [System.Runtime.InteropServices.NativeLibrary]::Load($dllPath)
Write-Host "pcai_media.dll loaded! Handle: $handle"

$exports = @(
    "pcai_media_init",
    "pcai_media_load_model",
    "pcai_media_generate_image",
    "pcai_media_generate_image_bytes",
    "pcai_media_generate_image_async",
    "pcai_media_poll_result",
    "pcai_media_cancel",
    "pcai_media_understand_image",
    "pcai_media_upscale_image",
    "pcai_media_shutdown",
    "pcai_media_last_error",
    "pcai_media_last_error_code",
    "pcai_media_free_string",
    "pcai_media_free_bytes"
)

$found = 0
foreach ($name in $exports) {
    try {
        $addr = [System.Runtime.InteropServices.NativeLibrary]::GetExport($handle, $name)
        Write-Host "  [OK] $name" -ForegroundColor Green
        $found++
    } catch {
        Write-Host "  [MISSING] $name" -ForegroundColor Red
    }
}

Write-Host "`n$found / $($exports.Count) exports found"
[System.Runtime.InteropServices.NativeLibrary]::Free($handle)
Write-Host "DLL unloaded."
