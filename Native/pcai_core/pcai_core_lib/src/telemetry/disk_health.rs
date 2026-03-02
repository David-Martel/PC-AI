use serde::Serialize;
use std::ffi::CString;
use std::ptr::null_mut;
use windows_sys::Win32::Foundation::*;
use windows_sys::Win32::Storage::FileSystem::*;
use windows_sys::Win32::System::IO::DeviceIoControl;

#[derive(Serialize)]
pub struct DiskHealthDetail {
    pub device_id: String,
    pub model: String,
    pub serial_number: String,
    pub status: String,
    pub smart_capable: bool,
    pub smart_status_ok: bool,
    pub severity: String,
}

pub fn collect_disk_health() -> Vec<DiskHealthDetail> {
    let mut disks = Vec::new();

    for i in 0..16 {
        let device_path = format!("\\\\.\\PhysicalDrive{}", i);
        let c_path = match CString::new(device_path.clone()) {
            Ok(s) => s,
            Err(_) => continue,
        };

        unsafe {
            let h_device = CreateFileA(
                c_path.as_ptr() as *const u8,
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                null_mut(),
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                null_mut() as *mut _, // windows-sys 0.59 uses isize for many handles, but error said *mut void
            );

            if h_device == INVALID_HANDLE_VALUE {
                continue;
            }

            let mut detail = DiskHealthDetail {
                device_id: device_path,
                model: "Unknown".to_string(),
                serial_number: "Unknown".to_string(),
                status: "Unknown".to_string(),
                smart_capable: false,
                smart_status_ok: true,
                severity: "OK".to_string(),
            };

            let mut smart_version = [0u8; 1024];
            let mut bytes_returned: u32 = 0;

            // IOCTL_SMART_GET_VERSION = 0x00074080
            if DeviceIoControl(
                h_device,
                0x00074080,
                null_mut(),
                0,
                smart_version.as_mut_ptr() as *mut _,
                smart_version.len() as u32,
                &mut bytes_returned,
                null_mut(),
            ) != 0
            {
                detail.smart_capable = true;
            }

            detail.status = if detail.smart_capable {
                "OK".to_string()
            } else {
                "N/A".to_string()
            };

            disks.push(detail);
            CloseHandle(h_device as *mut _); // HANDLE in foundation/mod.rs is isize, but error said *mut void
        }
    }

    disks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disk_health_detail_serialization() {
        let detail = DiskHealthDetail {
            device_id: "\\\\.\\PhysicalDrive0".to_string(),
            model: "Samsung SSD 980 PRO 1TB".to_string(),
            serial_number: "S5GXNF0R123456".to_string(),
            status: "OK".to_string(),
            smart_capable: true,
            smart_status_ok: true,
            severity: "OK".to_string(),
        };

        let json = serde_json::to_string(&detail).expect("Should serialize");
        assert!(json.contains("\"model\":\"Samsung SSD 980 PRO 1TB\""));
        assert!(json.contains("\"smart_capable\":true"));
    }
}
