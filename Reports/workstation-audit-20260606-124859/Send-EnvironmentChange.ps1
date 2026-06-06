$signature = @'
[DllImport("user32.dll", SetLastError=true, CharSet=CharSet.Auto)]
public static extern IntPtr SendMessageTimeout(
    IntPtr hWnd,
    uint Msg,
    UIntPtr wParam,
    string lParam,
    uint fuFlags,
    uint uTimeout,
    out UIntPtr lpdwResult);
'@

Add-Type -MemberDefinition $signature -Name NativeMethods -Namespace Win32
$result = [UIntPtr]::Zero
[void][Win32.NativeMethods]::SendMessageTimeout(
    [IntPtr]0xffff,
    0x1A,
    [UIntPtr]::Zero,
    'Environment',
    2,
    5000,
    [ref]$result)

'WM_SETTINGCHANGE sent'
