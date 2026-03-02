use serde::Serialize;
use std::ptr::null_mut;
use windows_sys::core::GUID;
use windows_sys::Win32::Devices::DeviceAndDriverInstallation::*;
use windows_sys::Win32::Foundation::*;

#[derive(Serialize)]
pub struct UsbDeviceDetail {
    pub name: String,
    pub hardware_id: String,
    pub manufacturer: String,
    pub driver_version: String,
    pub driver_date: String,
    pub install_date: String,
    pub status: String,
    pub config_error_code: u32,
    pub error_summary: String,
    pub help_url: String,
}

pub fn collect_usb_diagnostics() -> Vec<UsbDeviceDetail> {
    let mut devices = Vec::new();

    /* GUID for USB Devices
    // {A5DCBF10-6530-11D2-901F-00C04FB951ED}
    let guid = GUID {
        data1: 0xA5DCBF10,
        data2: 0x6530,
        data3: 0x11D2,
        data4: [0x90, 0x1F, 0x00, 0xC0, 0x4F, 0xB9, 0x51, 0xED],
    }; */

    unsafe {
        // Use DIGCF_ALLCLASSES to see all devices, then filter for USB enumerator.
        // This catches disabled and child devices that DIGCF_DEVICEINTERFACE might miss.
        let h_dev_info = SetupDiGetClassDevsW(null_mut(), null_mut(), null_mut(), DIGCF_PRESENT | DIGCF_ALLCLASSES);

        if h_dev_info as isize == INVALID_HANDLE_VALUE as isize {
            return devices;
        }

        let mut dev_info_data = SP_DEVINFO_DATA {
            cbSize: std::mem::size_of::<SP_DEVINFO_DATA>() as u32,
            ClassGuid: GUID {
                data1: 0,
                data2: 0,
                data3: 0,
                data4: [0; 8],
            },
            DevInst: 0,
            Reserved: 0,
        };

        let mut i = 0;
        while SetupDiEnumDeviceInfo(h_dev_info, i, &mut dev_info_data) != 0 {
            // Filter for USB devices only - check enumerator and hardware ID prefix
            let enumerator = get_device_property(h_dev_info, &mut dev_info_data, SPDRP_ENUMERATOR_NAME);
            let hardware_id = get_device_property(h_dev_info, &mut dev_info_data, SPDRP_HARDWAREID);

            let is_usb = enumerator == "USB"
                || enumerator == "USBVIDEO"
                || enumerator.contains("USB")
                || hardware_id.starts_with("USB\\")
                || hardware_id.starts_with("HDAUDIO\\"); // Match NVIDIA Audio if possible

            if !is_usb {
                i += 1;
                continue;
            }

            let mut detail = UsbDeviceDetail {
                name: "Unknown".to_string(),
                hardware_id,
                manufacturer: "Unknown".to_string(),
                driver_version: "Unknown".to_string(),
                driver_date: "Unknown".to_string(),
                install_date: "Unknown".to_string(),
                status: "Unknown".to_string(),
                config_error_code: 0,
                error_summary: "".to_string(),
                help_url: "".to_string(),
            };

            // Get Friendly Name
            detail.name = get_device_property(h_dev_info, &mut dev_info_data, SPDRP_FRIENDLYNAME);
            if detail.name == "Unknown" {
                detail.name = get_device_property(h_dev_info, &mut dev_info_data, SPDRP_DEVICEDESC);
            }

            detail.manufacturer = get_device_property(h_dev_info, &mut dev_info_data, SPDRP_MFG);

            // Get Status / Error Code
            let mut status: u32 = 0;
            let mut problem_code: u32 = 0;
            if CM_Get_DevNode_Status(&mut status, &mut problem_code, dev_info_data.DevInst, 0) == 0 {
                detail.config_error_code = problem_code;

                if let Some(info) = super::device_codes::get_problem_info(problem_code) {
                    detail.status = format!("Error: {}", info.short_description);
                    detail.error_summary = info.help_summary.to_string();
                    detail.help_url = info.help_url.to_string();
                } else {
                    detail.status = if problem_code == 0 {
                        "OK".to_string()
                    } else {
                        format!("Error {}", problem_code)
                    };
                }
            }

            devices.push(detail);
            i += 1;
        }

        SetupDiDestroyDeviceInfoList(h_dev_info);
    }

    devices
}

unsafe fn get_device_property(h_dev_info: HDEVINFO, dev_info_data: *mut SP_DEVINFO_DATA, property: u32) -> String {
    let mut buffer = [0u16; 512];
    let mut required_size = 0;
    if SetupDiGetDeviceRegistryPropertyW(
        h_dev_info,
        dev_info_data,
        property,
        null_mut(),
        buffer.as_mut_ptr() as *mut u8,
        (buffer.len() * 2) as u32,
        &mut required_size,
    ) != 0
    {
        let len = (0..buffer.len()).find(|&i| buffer[i] == 0).unwrap_or(buffer.len());
        String::from_utf16_lossy(&buffer[..len])
    } else {
        "Unknown".to_string()
    }
}
