use serde::Serialize;
use std::ptr::null_mut;
use windows_sys::core::GUID;
use windows_sys::Win32::Devices::DeviceAndDriverInstallation::*;
use windows_sys::Win32::Foundation::*;

#[derive(Serialize)]
pub struct PnpDeviceDetail {
    pub name: String,
    pub hardware_id: String,
    pub manufacturer: String,
    pub pnp_class: String,
    pub status: String,
    pub config_error_code: u32,
    pub error_summary: String,
    pub help_url: String,
    pub device_id: String,
}

pub fn collect_pnp_devices(class_filter: Option<&str>) -> Vec<PnpDeviceDetail> {
    let mut devices = Vec::new();

    unsafe {
        // Use DIGCF_ALLCLASSES to see all devices, then filter as needed.
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
            let pnp_class = get_device_property(h_dev_info, &mut dev_info_data, SPDRP_CLASS);
            let name = get_device_property(h_dev_info, &mut dev_info_data, SPDRP_FRIENDLYNAME);
            let name = if name == "Unknown" {
                get_device_property(h_dev_info, &mut dev_info_data, SPDRP_DEVICEDESC)
            } else {
                name
            };

            // Apply filter
            if let Some(filter) = class_filter {
                if pnp_class != filter && !name.to_lowercase().contains(&filter.to_lowercase()) {
                    i += 1;
                    continue;
                }
            }

            let hardware_id = get_device_property(h_dev_info, &mut dev_info_data, SPDRP_HARDWAREID);
            let manufacturer = get_device_property(h_dev_info, &mut dev_info_data, SPDRP_MFG);

            // Get Device Instance ID (the true unique ID)
            let mut device_id_buf = [0u16; 512];
            let mut device_id = "Unknown".to_string();
            if CM_Get_Device_IDW(
                dev_info_data.DevInst,
                device_id_buf.as_mut_ptr(),
                device_id_buf.len() as u32,
                0,
            ) == 0
            {
                let len = (0..device_id_buf.len())
                    .find(|&i| device_id_buf[i] == 0)
                    .unwrap_or(device_id_buf.len());
                device_id = String::from_utf16_lossy(&device_id_buf[..len]);
            }

            let mut detail = PnpDeviceDetail {
                name,
                hardware_id,
                manufacturer,
                pnp_class,
                status: "Unknown".to_string(),
                config_error_code: 0,
                error_summary: "".to_string(),
                help_url: "".to_string(),
                device_id,
            };

            // Get Status / Error Code
            let mut status: u32 = 0;
            let mut problem_code: u32 = 0;
            if CM_Get_DevNode_Status(&mut status, &mut problem_code, dev_info_data.DevInst, 0) == 0 {
                detail.config_error_code = problem_code;

                if let Some(info) = super::device_codes::get_problem_info(problem_code) {
                    if problem_code != 0 {
                        detail.status = format!("Error: {}", info.short_description);
                    } else {
                        detail.status = "OK".to_string();
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pnp_detail_serialization() {
        let detail = PnpDeviceDetail {
            name: "Test Device".to_string(),
            hardware_id: "PCI\\VEN_1234&DEV_5678".to_string(),
            manufacturer: "Test Corp".to_string(),
            pnp_class: "USB".to_string(),
            status: "OK".to_string(),
            config_error_code: 0,
            error_summary: "Working normally".to_string(),
            help_url: "http://example.com".to_string(),
            device_id: "DISPLAY\\TEST001\\0".to_string(),
        };

        let json = serde_json::to_string(&detail).expect("Should serialize");
        assert!(json.contains("\"name\":\"Test Device\""));
        assert!(json.contains("\"config_error_code\":0"));
    }
}
