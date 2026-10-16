use anyhow::Result;
use serde::{Deserialize, Serialize};

#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(not(target_os = "windows"))]
pub mod linux;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Usb4DeviceInfo {
    pub instance_id: String,
    pub friendly_name: String,
    pub class_name: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct AdapterAddressEntry {
    pub address: String,
    pub address_family: String,
    pub prefix_length: u8,
    pub is_link_local: bool,
    pub is_ula: bool,
    pub is_apipa: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct NeighborCacheEntry {
    pub interface_index: u32,
    pub ip_address: String,
    pub address_family: String,
    pub link_layer_address: String,
    pub state: String,
    pub is_router: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct ThunderboltAdapterState {
    pub interface_index: u32,
    pub interface_alias: String,
    pub interface_description: String,
    pub adapter_guid: String,
    pub mac_address: String,
    pub operational_status: String,
    pub transmit_link_speed_bps: u64,
    pub receive_link_speed_bps: u64,
    pub link_speed_gbps: f64,
    pub ipv4_mtu_bytes: u32,
    pub ipv6_mtu_bytes: u32,
    pub ipv4_metric: u32,
    pub ipv6_metric: u32,
    pub addresses: Vec<AdapterAddressEntry>,
    pub neighbors: Vec<NeighborCacheEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct PlatformThunderboltSnapshot {
    pub computer_name: String,
    pub os_family: String,
    pub architecture: String,
    pub collection_latency_us: u64,
    pub usb4_devices: Vec<Usb4DeviceInfo>,
    pub adapters: Vec<ThunderboltAdapterState>,
}

pub fn is_thunderbolt_or_usb4_adapter(alias: &str, description: &str) -> bool {
    let desc_lower = description.to_ascii_lowercase();
    let alias_lower = alias.to_ascii_lowercase();
    desc_lower.contains("usb4")
        || desc_lower.contains("thunderbolt")
        || desc_lower.contains("p2p network")
        || alias_lower.contains("thunderbolt")
        || alias_lower.contains("usb4")
        || alias_lower.starts_with("thunderbolt")
}

pub fn collect_platform_snapshot(include_disconnected: bool) -> Result<PlatformThunderboltSnapshot> {
    #[cfg(target_os = "windows")]
    {
        windows::collect_windows_snapshot(include_disconnected)
    }
    #[cfg(not(target_os = "windows"))]
    {
        linux::collect_linux_snapshot(include_disconnected)
    }
}
