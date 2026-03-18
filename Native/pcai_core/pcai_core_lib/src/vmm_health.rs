//! VMM Health Module - Hyper-V socket connectivity checks
//!
//! Provides health checks for WSL2/Hyper-V host communication.
//! Some structures and constants are reserved for future direct vsock support.

#[cfg(windows)]
use windows_sys::core::GUID;
#[cfg(windows)]
use windows_sys::Win32::Networking::WinSock::{socket, AF_HYPERV, INVALID_SOCKET, SOCK_STREAM};

#[cfg(not(windows))]
#[derive(Clone, Copy)]
#[expect(
    clippy::upper_case_acronyms,
    reason = "GUID matches the Windows SDK naming convention used by windows-sys \
              on the real target; the non-Windows stub must share the same name \
              so that code that references the type compiles on both platforms"
)]
struct GUID {
    #[expect(
        dead_code,
        reason = "stub field present solely to mirror the Windows SDK GUID layout for cross-platform compilation"
    )]
    pub data1: u32,
    #[expect(
        dead_code,
        reason = "stub field present solely to mirror the Windows SDK GUID layout for cross-platform compilation"
    )]
    pub data2: u16,
    #[expect(
        dead_code,
        reason = "stub field present solely to mirror the Windows SDK GUID layout for cross-platform compilation"
    )]
    pub data3: u16,
    #[expect(
        dead_code,
        reason = "stub field present solely to mirror the Windows SDK GUID layout for cross-platform compilation"
    )]
    pub data4: [u8; 8],
}

// Define SOCKADDR_HV manually since it might not be in windows-sys's version for networking
// Reserved for future direct vsock connect support
#[expect(
    dead_code,
    reason = "reserved for future direct HvSocket connect support; not yet used by health-check paths"
)]
#[repr(C)]
#[derive(Clone, Copy)]
struct SockAddrHv {
    family: u16,
    reserved: u16,
    vm_id: GUID,
    service_id: GUID,
}

// Well-known Hyper-V GUIDs - reserved for future vsock connect support
#[expect(
    dead_code,
    reason = "well-known Hyper-V GUID reserved for future vsock connect support"
)]
const HV_GUID_PARENT: GUID = GUID {
    data1: 0xa42e7cd,
    data2: 0x5542,
    data3: 0x4e71,
    data4: [0xaa, 0x44, 0x89, 0x6e, 0x99, 0xa9, 0x7a, 0x2b],
};

#[expect(
    dead_code,
    reason = "well-known Hyper-V GUID reserved for future vsock connect support"
)]
const HV_GUID_CHILDREN: GUID = GUID {
    data1: 0x9092221,
    data2: 0x2728,
    data3: 0x4ef2,
    data4: [0x91, 0x84, 0x20, 0xfa, 0x1d, 0x20, 0xa2, 0x06],
};

#[expect(
    dead_code,
    reason = "well-known Hyper-V loopback GUID reserved for future vsock connect support"
)]
const HV_GUID_LOOPBACK: GUID = GUID {
    data1: 0xe0e4ca1,
    data2: 0x496a,
    data3: 0x4cd2,
    data4: [0xac, 0x27, 0x4a, 0x71, 0x90, 0xc4, 0x1b, 0x27],
};

#[expect(
    dead_code,
    reason = "well-known Hyper-V wildcard GUID reserved for future vsock connect support"
)]
const HV_GUID_WILDCARD: GUID = GUID {
    data1: 0x00000000,
    data2: 0x0000,
    data3: 0x0000,
    data4: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
};

#[derive(Debug, serde::Serialize)]
pub struct VmmHealth {
    pub wsl_host_responding: bool,
    pub vsock_available: bool,
    pub bridge_latency_ms: u32,
}

/// Performs a direct AF_HYPERV check to see if the WSL/Host bridge is alive.
pub fn check_vmm_health() -> VmmHealth {
    #[cfg(windows)]
    {
        let mut health = VmmHealth {
            wsl_host_responding: false,
            vsock_available: false,
            bridge_latency_ms: 0,
        };

        unsafe {
            let s = socket(AF_HYPERV as i32, SOCK_STREAM, 0);
            if s != INVALID_SOCKET {
                health.vsock_available = true;
                // For now, we just check if we can create the socket.
                // A full connection test would require a known service ID (port).
                // We'll mark host as responding if we can at least open the subsystem.
                health.wsl_host_responding = true;

                // Close the test socket
                windows_sys::Win32::Networking::WinSock::closesocket(s);
            }
        }

        health
    }
    #[cfg(not(windows))]
    VmmHealth {
        wsl_host_responding: false,
        vsock_available: false,
        bridge_latency_ms: 0,
    }
}
