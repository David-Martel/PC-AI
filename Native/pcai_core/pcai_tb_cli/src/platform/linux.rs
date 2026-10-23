use crate::identity::local_hostname;
use crate::platform::{
    discover_xdomain_peers_in_dir, is_thunderbolt_or_usb4_adapter, map_linux_operstate_with_carrier,
    parse_link_speed_bps, AdapterAddressEntry, NeighborCacheEntry, PlatformThunderboltSnapshot,
    ThunderboltAdapterState, Usb4DeviceInfo,
};
use anyhow::Result;
use std::collections::HashMap;
use std::ffi::CStr;
use std::fs;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

fn enumerate_linux_usb4_devices() -> Vec<Usb4DeviceInfo> {
    let mut out = Vec::new();
    let tb_dir = Path::new("/sys/bus/thunderbolt/devices");
    if let Ok(entries) = fs::read_dir(tb_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = entry.path();
            let dev_name = fs::read_to_string(path.join("device_name"))
                .ok()
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "Thunderbolt / USB4 Device".to_string());
            let authorized = fs::read_to_string(path.join("authorized"))
                .ok()
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "1".to_string());
            out.push(Usb4DeviceInfo {
                instance_id: format!("thunderbolt/{name}"),
                friendly_name: dev_name,
                class_name: "Thunderbolt".to_string(),
                status: if authorized == "0" {
                    "Unauthorized".to_string()
                } else {
                    "OK".to_string()
                },
            });
        }
    }
    out
}

fn collect_linux_neighbors_by_ifindex() -> HashMap<u32, Vec<NeighborCacheEntry>> {
    let mut map: HashMap<u32, Vec<NeighborCacheEntry>> = HashMap::new();
    if let Ok(output) = Command::new("ip").args(["-j", "neigh", "show"]).output() {
        if output.status.success() {
            if let Ok(json_val) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                if let Some(arr) = json_val.as_array() {
                    for item in arr {
                        let dst = item.get("dst").and_then(|v| v.as_str()).unwrap_or("");
                        let lladdr = item.get("lladdr").and_then(|v| v.as_str()).unwrap_or("");
                        let dev = item.get("dev").and_then(|v| v.as_str()).unwrap_or("");
                        if dst.is_empty() || lladdr.is_empty() || dev.is_empty() {
                            continue;
                        }
                        let if_idx = fs::read_to_string(format!("/sys/class/net/{dev}/ifindex"))
                            .ok()
                            .and_then(|s| s.trim().parse::<u32>().ok())
                            .unwrap_or(0);
                        if if_idx == 0 {
                            continue;
                        }
                        let state = item
                            .get("state")
                            .and_then(|v| v.as_array())
                            .and_then(|a| a.first())
                            .and_then(|v| v.as_str())
                            .unwrap_or("REACHABLE")
                            .to_string();
                        let fam = if dst.contains(':') { "IPv6" } else { "IPv4" };
                        map.entry(if_idx).or_default().push(NeighborCacheEntry {
                            interface_index: if_idx,
                            ip_address: dst.to_string(),
                            address_family: fam.to_string(),
                            link_layer_address: lladdr.replace(':', "-").to_ascii_uppercase(),
                            state,
                            is_router: item.get("router").is_some(),
                        });
                    }
                }
            }
        }
    }
    map
}

fn query_linux_route_metric(if_name: &str, ipv6: bool) -> Option<u32> {
    let fam_flag = if ipv6 { "-6" } else { "-4" };
    let output = Command::new("ip")
        .args(["-j", fam_flag, "route", "show", "dev", if_name])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let val: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let arr = val.as_array()?;
    arr.iter()
        .filter_map(|item| item.get("metric").and_then(|m| m.as_u64()).map(|m| m as u32))
        .min()
}

fn query_linux_external_owner(if_name: &str, ifalias: &str) -> Option<String> {
    let trimmed_alias = ifalias.trim();
    if trimmed_alias.starts_with("ms01-usb4:") {
        return Some(format!("ms01-usb4-extension:{trimmed_alias}"));
    }
    if let Ok(output) = Command::new("nmcli")
        .args(["-g", "GENERAL.CONNECTION", "device", "show", if_name])
        .output()
    {
        if output.status.success() {
            let conn = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !conn.is_empty() && conn != "--" {
                return Some(format!("NetworkManager:{conn}"));
            }
        }
    }
    None
}

fn resolve_xdomain_peer_for_netdev(
    net_path: &Path,
    all_xdomain_peers: &[crate::platform::XdomainNetworkPeer],
) -> (Option<String>, Option<String>) {
    if let Ok(canon_dev) = fs::canonicalize(net_path.join("device")) {
        if let Some(parent) = canon_dev.parent() {
            let dev_name = fs::read_to_string(parent.join("device_name"))
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            let uid = fs::read_to_string(parent.join("unique_id"))
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            if dev_name.is_some() && uid.is_some() {
                return (dev_name, uid);
            }
        }
    }
    if all_xdomain_peers.len() == 1 {
        let p = &all_xdomain_peers[0];
        return (Some(p.device_name.clone()), Some(p.unique_id.clone()));
    }
    (None, None)
}

fn count_netmask_ones_v4(addr: Ipv4Addr) -> u8 {
    u32::from(addr).count_ones() as u8
}

fn count_netmask_ones_v6(addr: Ipv6Addr) -> u8 {
    u128::from(addr).count_ones() as u8
}

pub fn collect_linux_snapshot(include_disconnected: bool) -> Result<PlatformThunderboltSnapshot> {
    let start = Instant::now();
    let usb4_devices = enumerate_linux_usb4_devices();
    let xdomain_peers = discover_xdomain_peers_in_dir(Path::new("/sys/bus/thunderbolt/devices"));
    let mut neighbor_map = collect_linux_neighbors_by_ifindex();
    let mut addr_map: HashMap<String, Vec<AdapterAddressEntry>> = HashMap::new();

    unsafe {
        let mut ifap: *mut libc::ifaddrs = std::ptr::null_mut();
        if libc::getifaddrs(&mut ifap) == 0 {
            let mut curr = ifap;
            while !curr.is_null() {
                let ifa = &*curr;
                if !ifa.ifa_name.is_null() && !ifa.ifa_addr.is_null() {
                    let if_name = CStr::from_ptr(ifa.ifa_name).to_string_lossy().into_owned();
                    let fam = i32::from((*ifa.ifa_addr).sa_family);
                    if fam == libc::AF_INET {
                        let sin = &*(ifa.ifa_addr.cast::<libc::sockaddr_in>());
                        let ip = Ipv4Addr::from(u32::from_be(sin.sin_addr.s_addr));
                        let prefix_len = if !ifa.ifa_netmask.is_null() {
                            let mask_sin = &*(ifa.ifa_netmask.cast::<libc::sockaddr_in>());
                            count_netmask_ones_v4(Ipv4Addr::from(u32::from_be(mask_sin.sin_addr.s_addr)))
                        } else {
                            24
                        };
                        let ip_str = ip.to_string();
                        addr_map.entry(if_name).or_default().push(AdapterAddressEntry {
                            is_apipa: ip_str.starts_with("169.254."),
                            is_link_local: ip_str.starts_with("169.254."),
                            is_ula: false,
                            address: ip_str,
                            address_family: "IPv4".to_string(),
                            prefix_length: prefix_len,
                        });
                    } else if fam == libc::AF_INET6 {
                        let sin6 = &*(ifa.ifa_addr.cast::<libc::sockaddr_in6>());
                        let ip = Ipv6Addr::from(sin6.sin6_addr.s6_addr);
                        let prefix_len = if !ifa.ifa_netmask.is_null() {
                            let mask_sin6 = &*(ifa.ifa_netmask.cast::<libc::sockaddr_in6>());
                            count_netmask_ones_v6(Ipv6Addr::from(mask_sin6.sin6_addr.s6_addr))
                        } else {
                            64
                        };
                        let ip_str = ip.to_string();
                        let lower = ip_str.to_ascii_lowercase();
                        addr_map.entry(if_name).or_default().push(AdapterAddressEntry {
                            is_apipa: false,
                            is_link_local: lower.starts_with("fe80:"),
                            is_ula: lower.starts_with("fd") || lower.starts_with("fc"),
                            address: ip_str,
                            address_family: "IPv6".to_string(),
                            prefix_length: prefix_len,
                        });
                    }
                }
                curr = ifa.ifa_next;
            }
            libc::freeifaddrs(ifap);
        }
    }

    let mut adapters = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/class/net") {
        for entry in entries.flatten() {
            let if_name = entry.file_name().to_string_lossy().into_owned();
            let net_path = entry.path();
            let driver_link = fs::read_link(net_path.join("device/driver"))
                .ok()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            let ifalias = fs::read_to_string(net_path.join("ifalias"))
                .ok()
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            let description = if driver_link.contains("thunderbolt-net")
                || driver_link.contains("thunderbolt_net")
                || if_name.starts_with("thunderbolt")
                || if_name.starts_with("usb4")
            {
                "Thunderbolt / USB4 P2P Network Adapter (thunderbolt-net)".to_string()
            } else if ifalias.starts_with("ms01-usb4:") || ifalias.starts_with("pcai-usb4:") {
                format!("Thunderbolt / USB4 Bridged Interface ({ifalias})")
            } else {
                driver_link
            };

            if is_thunderbolt_or_usb4_adapter(&if_name, &description) {
                let operstate_raw = fs::read_to_string(net_path.join("operstate"))
                    .ok()
                    .unwrap_or_else(|| "down".to_string());
                let carrier_raw = fs::read_to_string(net_path.join("carrier")).ok();
                let status_str = map_linux_operstate_with_carrier(&operstate_raw, carrier_raw.as_deref());
                if !include_disconnected && status_str != "Up" {
                    continue;
                }
                let if_index = fs::read_to_string(net_path.join("ifindex"))
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .unwrap_or(0);
                let mtu = fs::read_to_string(net_path.join("mtu"))
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .unwrap_or(0);
                let mac = fs::read_to_string(net_path.join("address"))
                    .ok()
                    .map(|s| s.trim().replace(':', "-").to_ascii_uppercase())
                    .unwrap_or_default();
                let speed_raw = fs::read_to_string(net_path.join("speed")).ok();
                let bps = parse_link_speed_bps(speed_raw.as_deref());
                let speed_known = bps > 0;
                let gbps = (bps as f64) / 1_000_000_000.0;

                let v4_metric_opt = query_linux_route_metric(&if_name, false);
                let v6_metric_opt = query_linux_route_metric(&if_name, true);
                let metric_known = v4_metric_opt.is_some() || v6_metric_opt.is_some();

                let (xdomain_peer_device_name, xdomain_peer_unique_id) =
                    resolve_xdomain_peer_for_netdev(&net_path, &xdomain_peers);
                let external_owner = query_linux_external_owner(&if_name, &ifalias);

                adapters.push(ThunderboltAdapterState {
                    interface_index: if_index,
                    interface_alias: if_name.clone(),
                    interface_description: description,
                    adapter_guid: format!("linux-net-{if_name}"),
                    mac_address: mac,
                    operational_status: status_str.to_string(),
                    speed_known,
                    transmit_link_speed_bps: bps,
                    receive_link_speed_bps: bps,
                    link_speed_gbps: (gbps * 100.0).round() / 100.0,
                    ipv4_mtu_bytes: mtu,
                    ipv6_mtu_bytes: mtu,
                    metric_known,
                    ipv4_metric: v4_metric_opt.unwrap_or(0),
                    ipv6_metric: v6_metric_opt.unwrap_or(0),
                    xdomain_peer_device_name,
                    xdomain_peer_unique_id,
                    external_owner,
                    addresses: addr_map.remove(&if_name).unwrap_or_default(),
                    neighbors: neighbor_map.remove(&if_index).unwrap_or_default(),
                });
            }
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

#[cfg(test)]
mod speed_tests {
    use crate::platform::parse_link_speed_bps;

    #[test]
    fn unknown_speed_never_becomes_twenty_gigabit() {
        for raw in [
            None,
            Some("-1"),
            Some(""),
            Some("invalid"),
            Some("0"),
            Some("18446744073709551615"),
        ] {
            assert_eq!(parse_link_speed_bps(raw), 0);
        }
    }

    #[test]
    fn observed_speed_is_converted_without_invention() {
        assert_eq!(parse_link_speed_bps(Some("10000\n")), 10_000_000_000);
    }
}
