use crate::identity::local_hostname;
use crate::platform::{
    is_thunderbolt_or_usb4_adapter, AdapterAddressEntry, NeighborCacheEntry, PlatformThunderboltSnapshot,
    ThunderboltAdapterState, Usb4DeviceInfo,
};
use anyhow::Result;
use std::collections::HashMap;
use std::ffi::CStr;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::ptr;
use std::time::Instant;
use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
    CM_Get_Device_ID_ListW, CM_Get_Device_ID_List_SizeW, CM_GETIDLIST_FILTER_ENUMERATOR, CM_GETIDLIST_FILTER_PRESENT,
    CR_SUCCESS,
};
use windows_sys::Win32::Foundation::{ERROR_BUFFER_OVERFLOW, NO_ERROR};
use windows_sys::Win32::NetworkManagement::IpHelper::{
    FreeMibTable, GetAdaptersAddresses, GetIpInterfaceEntry, GetIpNetTable2, GAA_FLAG_INCLUDE_PREFIX,
    IP_ADAPTER_ADDRESSES_LH, MIB_IPINTERFACE_ROW, MIB_IPNET_ROW2, MIB_IPNET_TABLE2,
};
use windows_sys::Win32::NetworkManagement::Ndis::{
    IfOperStatusDormant, IfOperStatusDown, IfOperStatusLowerLayerDown, IfOperStatusNotPresent, IfOperStatusTesting,
    IfOperStatusUp,
};
use windows_sys::Win32::Networking::WinSock::{AF_INET, AF_INET6, AF_UNSPEC, SOCKADDR_IN, SOCKADDR_IN6};

unsafe fn pwstr_to_string(ptr: *const u16) -> String {
    if ptr.is_null() {
        return String::new();
    }
    let mut len = 0usize;
    while *ptr.add(len) != 0 {
        len += 1;
    }
    String::from_utf16_lossy(std::slice::from_raw_parts(ptr, len))
}

unsafe fn pstr_to_string(ptr: *const u8) -> String {
    if ptr.is_null() {
        return String::new();
    }
    CStr::from_ptr(ptr.cast()).to_string_lossy().into_owned()
}

fn oper_status_to_string(status: i32) -> &'static str {
    if status == IfOperStatusUp {
        "Up"
    } else if status == IfOperStatusDown {
        "Down"
    } else if status == IfOperStatusTesting {
        "Testing"
    } else if status == IfOperStatusDormant {
        "Dormant"
    } else if status == IfOperStatusNotPresent {
        "NotPresent"
    } else if status == IfOperStatusLowerLayerDown {
        "LowerLayerDown"
    } else {
        "Unknown"
    }
}

fn neighbor_state_to_string(state: i32) -> &'static str {
    match state {
        0 => "Unreachable",
        1 => "Incomplete",
        2 => "Probe",
        3 => "Delay",
        4 => "Stale",
        5 => "Reachable",
        6 => "Permanent",
        _ => "Unknown",
    }
}

fn format_mac_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02X}")).collect::<Vec<_>>().join("-")
}

pub fn enumerate_usb4_pnp_devices() -> Result<Vec<Usb4DeviceInfo>> {
    super::windows_inventory::collect_present_usb4_devices(|present_only| {
        let filter_usb4: [u16; 5] = [b'U' as u16, b'S' as u16, b'B' as u16, b'4' as u16, 0];
        let flags = CM_GETIDLIST_FILTER_ENUMERATOR | if present_only { CM_GETIDLIST_FILTER_PRESENT } else { 0 };
        unsafe {
            let mut char_len: u32 = 0;
            let cr_size = CM_Get_Device_ID_List_SizeW(&mut char_len, filter_usb4.as_ptr(), flags);
            anyhow::ensure!(
                cr_size == CR_SUCCESS,
                "USB4 device list size failed: CONFIGRET {cr_size}"
            );
            if char_len == 0 {
                return Ok(Vec::new());
            }
            let mut buf = vec![0u16; (char_len as usize) + 2];
            let cr_list = CM_Get_Device_ID_ListW(filter_usb4.as_ptr(), buf.as_mut_ptr(), char_len, flags);
            anyhow::ensure!(cr_list == CR_SUCCESS, "USB4 device list failed: CONFIGRET {cr_list}");
            Ok(buf
                .split(|&ch| ch == 0)
                .filter(|slice| !slice.is_empty())
                .map(String::from_utf16_lossy)
                .collect())
        }
    })
}

pub fn query_interface_mtu_and_metric_checked(if_index: u32, family: u16) -> Option<(u32, u32)> {
    unsafe {
        let mut row: MIB_IPINTERFACE_ROW = std::mem::zeroed();
        row.Family = family;
        row.InterfaceIndex = if_index;
        let rc = GetIpInterfaceEntry(&mut row);
        if rc == NO_ERROR {
            Some((row.NlMtu, row.Metric))
        } else {
            None
        }
    }
}

fn collect_neighbors_by_if_index() -> HashMap<u32, Vec<NeighborCacheEntry>> {
    let mut map: HashMap<u32, Vec<NeighborCacheEntry>> = HashMap::new();
    unsafe {
        let mut table_ptr: *mut MIB_IPNET_TABLE2 = ptr::null_mut();
        let rc = GetIpNetTable2(AF_UNSPEC, &mut table_ptr);
        if rc != NO_ERROR || table_ptr.is_null() {
            return map;
        }
        let num_entries = (*table_ptr).NumEntries as usize;
        let first_row: *const MIB_IPNET_ROW2 = (*table_ptr).Table.as_ptr();
        let rows = std::slice::from_raw_parts(first_row, num_entries);

        for row in rows {
            let state_str = neighbor_state_to_string(row.State);
            if state_str == "Unreachable" || state_str == "Incomplete" {
                continue;
            }
            let family = row.Address.si_family;
            let (ip_str, fam_str) = if family == AF_INET {
                let b = row.Address.Ipv4.sin_addr.S_un.S_un_b;
                let ip = Ipv4Addr::new(b.s_b1, b.s_b2, b.s_b3, b.s_b4);
                if ip.is_loopback() || ip.is_multicast() || ip.is_broadcast() || ip.is_unspecified() {
                    continue;
                }
                (ip.to_string(), "IPv4")
            } else if family == AF_INET6 {
                let bytes = row.Address.Ipv6.sin6_addr.u.Byte;
                let ip = Ipv6Addr::from(bytes);
                if ip.is_loopback() || ip.is_multicast() || ip.is_unspecified() {
                    continue;
                }
                (ip.to_string(), "IPv6")
            } else {
                continue;
            };

            let mac_len = (row.PhysicalAddressLength as usize).min(row.PhysicalAddress.len());
            if mac_len == 0 {
                continue;
            }
            let mac = format_mac_bytes(&row.PhysicalAddress[..mac_len]);
            if mac == "00-00-00-00-00-00" || mac == "FF-FF-FF-FF-FF-FF" {
                continue;
            }
            let is_router = (row.Anonymous.Flags & 0x01) != 0;
            map.entry(row.InterfaceIndex).or_default().push(NeighborCacheEntry {
                interface_index: row.InterfaceIndex,
                ip_address: ip_str,
                address_family: fam_str.to_string(),
                link_layer_address: mac,
                state: state_str.to_string(),
                is_router,
            });
        }

        FreeMibTable(table_ptr.cast());
    }
    map
}

pub fn collect_windows_snapshot(include_disconnected: bool) -> Result<PlatformThunderboltSnapshot> {
    let start = Instant::now();
    let usb4_devices = enumerate_usb4_pnp_devices()?;
    let mut neighbor_map = collect_neighbors_by_if_index();
    let mut adapters = Vec::new();

    unsafe {
        let mut size: u32 = 32 * 1024;
        let mut buffer = vec![0u8; size as usize];

        let mut rc = GetAdaptersAddresses(
            u32::from(AF_UNSPEC),
            GAA_FLAG_INCLUDE_PREFIX,
            ptr::null(),
            buffer.as_mut_ptr().cast::<IP_ADAPTER_ADDRESSES_LH>(),
            &mut size,
        );
        if rc == ERROR_BUFFER_OVERFLOW {
            buffer.resize(size as usize, 0);
            rc = GetAdaptersAddresses(
                u32::from(AF_UNSPEC),
                GAA_FLAG_INCLUDE_PREFIX,
                ptr::null(),
                buffer.as_mut_ptr().cast::<IP_ADAPTER_ADDRESSES_LH>(),
                &mut size,
            );
        }
        if rc != NO_ERROR {
            anyhow::bail!("GetAdaptersAddresses failed with Win32 error {rc}");
        }

        let mut curr = buffer.as_ptr().cast::<IP_ADAPTER_ADDRESSES_LH>();
        while !curr.is_null() {
            let adapter = &*curr;
            let friendly_name = pwstr_to_string(adapter.FriendlyName);
            let description = pwstr_to_string(adapter.Description);

            if is_thunderbolt_or_usb4_adapter(&friendly_name, &description) {
                let status_str = oper_status_to_string(adapter.OperStatus);
                if include_disconnected || status_str == "Up" {
                    let if_index = adapter.Anonymous1.Anonymous.IfIndex;
                    let adapter_guid = pstr_to_string(adapter.AdapterName);
                    let mac_len = (adapter.PhysicalAddressLength as usize).min(adapter.PhysicalAddress.len());
                    let mac_str = format_mac_bytes(&adapter.PhysicalAddress[..mac_len]);

                    let v4_info = query_interface_mtu_and_metric_checked(if_index, AF_INET);
                    let v6_info = query_interface_mtu_and_metric_checked(if_index, AF_INET6);
                    let metric_known = v4_info.is_some() && v6_info.is_some();
                    let (ipv4_mtu, ipv4_metric) = v4_info.unwrap_or((0, 0));
                    let (ipv6_mtu, ipv6_metric) = v6_info.unwrap_or((0, 0));

                    let mut addresses = Vec::new();
                    let mut uni = adapter.FirstUnicastAddress;
                    while !uni.is_null() {
                        let u = &*uni;
                        let sa = u.Address.lpSockaddr;
                        if !sa.is_null() {
                            let fam = (*sa).sa_family;
                            let prefix_len = u.OnLinkPrefixLength;
                            if fam == AF_INET {
                                let sin = &*(sa.cast::<SOCKADDR_IN>());
                                let b = sin.sin_addr.S_un.S_un_b;
                                let ip = Ipv4Addr::new(b.s_b1, b.s_b2, b.s_b3, b.s_b4);
                                let ip_str = ip.to_string();
                                addresses.push(AdapterAddressEntry {
                                    is_apipa: ip_str.starts_with("169.254."),
                                    is_link_local: ip_str.starts_with("169.254."),
                                    is_ula: false,
                                    address: ip_str,
                                    address_family: "IPv4".to_string(),
                                    prefix_length: prefix_len,
                                });
                            } else if fam == AF_INET6 {
                                let sin6 = &*(sa.cast::<SOCKADDR_IN6>());
                                let ip = Ipv6Addr::from(sin6.sin6_addr.u.Byte);
                                let ip_str = ip.to_string();
                                let lower = ip_str.to_ascii_lowercase();
                                addresses.push(AdapterAddressEntry {
                                    is_apipa: false,
                                    is_link_local: lower.starts_with("fe80:"),
                                    is_ula: lower.starts_with("fd") || lower.starts_with("fc"),
                                    address: ip_str,
                                    address_family: "IPv6".to_string(),
                                    prefix_length: prefix_len,
                                });
                            }
                        }
                        uni = u.Next;
                    }

                    let tx_bps = adapter.TransmitLinkSpeed;
                    let rx_bps = adapter.ReceiveLinkSpeed;
                    let speed_known = tx_bps != u64::MAX && tx_bps > 0;
                    let gbps = if speed_known {
                        (tx_bps as f64) / 1_000_000_000.0
                    } else {
                        0.0
                    };

                    adapters.push(ThunderboltAdapterState {
                        interface_index: if_index,
                        interface_alias: friendly_name,
                        interface_description: description,
                        adapter_guid,
                        mac_address: mac_str,
                        operational_status: status_str.to_string(),
                        speed_known,
                        transmit_link_speed_bps: if tx_bps == u64::MAX { 0 } else { tx_bps },
                        receive_link_speed_bps: if rx_bps == u64::MAX { 0 } else { rx_bps },
                        link_speed_gbps: (gbps * 100.0).round() / 100.0,
                        ipv4_mtu_bytes: if ipv4_mtu > 0 { ipv4_mtu } else { adapter.Mtu },
                        ipv6_mtu_bytes: if ipv6_mtu > 0 { ipv6_mtu } else { adapter.Mtu },
                        metric_known,
                        ipv4_metric,
                        ipv6_metric,
                        xdomain_peer_device_name: None,
                        xdomain_peer_unique_id: None,
                        external_owner: None,
                        addresses,
                        neighbors: neighbor_map.remove(&if_index).unwrap_or_default(),
                    });
                }
            }
            curr = adapter.Next;
        }
    }

    let elapsed_us = start.elapsed().as_micros() as u64;
    Ok(PlatformThunderboltSnapshot {
        computer_name: local_hostname(),
        os_family: std::env::consts::OS.to_string(),
        architecture: std::env::consts::ARCH.to_string(),
        collection_latency_us: elapsed_us,
        usb4_devices,
        adapters,
    })
}
