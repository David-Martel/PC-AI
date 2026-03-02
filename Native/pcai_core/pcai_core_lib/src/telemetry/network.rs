use serde::Serialize;
use std::ptr::null_mut;
use windows_sys::Win32::Foundation::*;
use windows_sys::Win32::NetworkManagement::IpHelper::*;

#[derive(Serialize)]
pub struct NetworkInterfaceDetail {
    pub name: String,
    pub description: String,
    pub bytes_sent_total: u64,
    pub bytes_received_total: u64,
    pub octets_in_per_sec: u64,
    pub octets_out_per_sec: u64,
    pub operational_status: u32,
    pub if_index: u32,
}

pub fn collect_network_diagnostics() -> Vec<NetworkInterfaceDetail> {
    let mut interfaces = Vec::new();

    unsafe {
        let mut size = 0;
        if GetIfTable(null_mut(), &mut size, 0) == ERROR_INSUFFICIENT_BUFFER {
            let mut buffer = vec![0u8; size as usize];
            let p_table = buffer.as_mut_ptr() as *mut MIB_IFTABLE;

            if GetIfTable(p_table, &mut size, 0) == NO_ERROR {
                let table = &*p_table;
                for i in 0..table.dwNumEntries as usize {
                    let row_ptr = table.table.as_ptr().add(i);
                    let row = &*row_ptr;

                    // Convert bDescr (8-bit) to String
                    let description = String::from_utf8_lossy(&row.bDescr[..row.dwDescrLen as usize]).into_owned();

                    interfaces.push(NetworkInterfaceDetail {
                        name: format!("Interface {}", row.dwIndex),
                        description,
                        bytes_sent_total: row.dwOutOctets as u64,
                        bytes_received_total: row.dwInOctets as u64,
                        octets_in_per_sec: 0,
                        octets_out_per_sec: 0,
                        operational_status: row.dwOperStatus as u32,
                        if_index: row.dwIndex,
                    });
                }
            }
        }
    }

    interfaces
}
