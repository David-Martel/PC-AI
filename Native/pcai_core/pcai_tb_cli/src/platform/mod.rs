use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Creates a background command without a Windows console window.
pub fn hidden_command(program: impl AsRef<std::ffi::OsStr>) -> std::process::Command {
    let mut command = std::process::Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = &mut command;
    command
}

#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(any(target_os = "windows", test))]
mod windows_inventory;

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
    pub speed_known: bool,
    pub transmit_link_speed_bps: u64,
    pub receive_link_speed_bps: u64,
    pub link_speed_gbps: f64,
    pub ipv4_mtu_bytes: u32,
    pub ipv6_mtu_bytes: u32,
    pub metric_known: bool,
    pub ipv4_metric: u32,
    pub ipv6_metric: u32,
    #[serde(default)]
    pub xdomain_peer_device_name: Option<String>,
    #[serde(default)]
    pub xdomain_peer_unique_id: Option<String>,
    #[serde(default)]
    pub external_owner: Option<String>,
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
        || desc_lower.contains("ms01-usb4")
        || desc_lower.contains("pcai-usb4")
        || alias_lower.contains("thunderbolt")
        || alias_lower.contains("usb4")
        || alias_lower.starts_with("thunderbolt")
        || alias_lower.starts_with("ms01-milly")
}

/// Zero is the schema's unknown-speed sentinel, never an invented measurement.
pub fn parse_link_speed_bps(raw: Option<&str>) -> u64 {
    raw.and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&mbps| mbps > 0)
        .and_then(|mbps| mbps.checked_mul(1_000_000))
        .unwrap_or(0)
}

pub fn map_linux_operstate_with_carrier(raw_operstate: &str, raw_carrier: Option<&str>) -> &'static str {
    match raw_operstate.trim().to_ascii_lowercase().as_str() {
        "up" => "Up",
        "unknown" => {
            if raw_carrier.is_some_and(|c| c.trim() == "1") {
                "Up"
            } else {
                "Unknown"
            }
        }
        "down" => "Down",
        "dormant" => "Dormant",
        "lowerlayerdown" => "LowerLayerDown",
        "notpresent" => "NotPresent",
        "testing" => "Testing",
        _ => "Unknown",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XdomainNetworkPeer {
    pub service_id: String,
    pub parent_id: String,
    pub device_name: String,
    pub unique_id: String,
}

pub fn discover_xdomain_peers_in_dir(tb_devices_dir: &std::path::Path) -> Vec<XdomainNetworkPeer> {
    let mut peers = Vec::new();
    let Ok(entries) = std::fs::read_dir(tb_devices_dir) else {
        return peers;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some((parent_id, _service_idx)) = name.split_once('.') else {
            continue;
        };
        let service_path = entry.path();
        let key = std::fs::read_to_string(service_path.join("key"))
            .ok()
            .map(|s| s.trim().to_ascii_lowercase())
            .unwrap_or_default();
        let modalias = std::fs::read_to_string(service_path.join("modalias"))
            .ok()
            .map(|s| s.trim().to_ascii_lowercase())
            .unwrap_or_default();
        if key != "network" && !modalias.contains("knetwork") {
            continue;
        }
        let parent_path = tb_devices_dir.join(parent_id);
        let device_name = std::fs::read_to_string(parent_path.join("device_name"))
            .ok()
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        let unique_id = std::fs::read_to_string(parent_path.join("unique_id"))
            .ok()
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        if !device_name.is_empty() && !unique_id.is_empty() {
            let parent_id_owned = parent_id.to_string();
            peers.push(XdomainNetworkPeer {
                service_id: name,
                parent_id: parent_id_owned,
                device_name,
                unique_id,
            });
        }
    }
    peers.sort_by(|a, b| a.service_id.cmp(&b.service_id));
    peers
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

pub fn verify_adapter_identity_in_snapshot(
    snapshot: &PlatformThunderboltSnapshot,
    if_index: u32,
    expected_guid: &str,
    expected_mac: &str,
) -> Result<()> {
    let found = snapshot
        .adapters
        .iter()
        .find(|a| a.interface_index == if_index)
        .ok_or_else(|| anyhow::anyhow!("Adapter ifIndex {if_index} is no longer present in live snapshot"))?;
    if !expected_guid.is_empty() && !found.adapter_guid.eq_ignore_ascii_case(expected_guid) {
        anyhow::bail!(
            "Adapter ifIndex {if_index} GUID changed from '{expected_guid}' to '{}' (hotplug race aborted)",
            found.adapter_guid
        );
    }
    if !expected_mac.is_empty() && !found.mac_address.eq_ignore_ascii_case(expected_mac) {
        anyhow::bail!(
            "Adapter ifIndex {if_index} MAC changed from '{expected_mac}' to '{}' (hotplug race aborted)",
            found.mac_address
        );
    }
    Ok(())
}

pub fn verify_live_adapter_identity(if_index: u32, expected_guid: &str, expected_mac: &str) -> Result<()> {
    let snap = collect_platform_snapshot(true)?;
    verify_adapter_identity_in_snapshot(&snap, if_index, expected_guid, expected_mac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_thunderbolt_or_usb4_adapter_matching() {
        assert!(is_thunderbolt_or_usb4_adapter(
            "Ethernet 3",
            "USB4(TM) P2P Network Adapter"
        ));
        assert!(is_thunderbolt_or_usb4_adapter("thunderbolt0", "thunderbolt-net"));
        assert!(is_thunderbolt_or_usb4_adapter(
            "ms01-milly",
            "Thunderbolt / USB4 Bridged Interface (ms01-usb4)"
        ));
        assert!(!is_thunderbolt_or_usb4_adapter(
            "Wi-Fi",
            "Intel(R) Wi-Fi 6E AX211 160MHz"
        ));
    }

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
        assert_eq!(parse_link_speed_bps(Some("10000\n")), 10_000_000_000);
        assert_eq!(parse_link_speed_bps(Some("20000")), 20_000_000_000);
    }

    #[test]
    fn test_discover_xdomain_peers_in_dir_parses_network_service_only() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        std::fs::create_dir_all(root.join("0-0")).unwrap();
        std::fs::write(root.join("0-0/device_name"), "Gen12\n").unwrap();

        std::fs::create_dir_all(root.join("0-1")).unwrap();
        std::fs::write(root.join("0-1/device_name"), "millylaptop1\n").unwrap();
        std::fs::write(root.join("0-1/unique_id"), "bab38780-0019-a9b4-ffff-ffffffffffff\n").unwrap();

        std::fs::create_dir_all(root.join("0-1.0")).unwrap();
        std::fs::write(root.join("0-1.0/key"), "network\n").unwrap();

        std::fs::create_dir_all(root.join("1-1")).unwrap();
        std::fs::write(root.join("1-1/device_name"), "TBT4-UDX1 Dock\n").unwrap();
        std::fs::write(root.join("1-1/unique_id"), "8c5e8780-00d7-986c-ffff-ffffffffffff\n").unwrap();

        let peers = discover_xdomain_peers_in_dir(root);
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].device_name, "millylaptop1");
        assert_eq!(peers[0].unique_id, "bab38780-0019-a9b4-ffff-ffffffffffff");
    }

    #[test]
    fn test_verify_adapter_identity_detects_guid_or_mac_drift() {
        let snap = PlatformThunderboltSnapshot {
            computer_name: "dtm-p1gen7".to_string(),
            os_family: "windows".to_string(),
            architecture: "x86_64".to_string(),
            collection_latency_us: 100,
            usb4_devices: vec![],
            adapters: vec![ThunderboltAdapterState {
                interface_index: 23,
                interface_alias: "Ethernet 3".to_string(),
                interface_description: "USB4(TM) P2P Network Adapter".to_string(),
                adapter_guid: "{11111111-2222-3333-4444-555555555555}".to_string(),
                mac_address: "AA-BB-CC-DD-EE-01".to_string(),
                operational_status: "Up".to_string(),
                speed_known: true,
                transmit_link_speed_bps: 20_000_000_000,
                receive_link_speed_bps: 20_000_000_000,
                link_speed_gbps: 20.0,
                ipv4_mtu_bytes: 62000,
                ipv6_mtu_bytes: 62000,
                metric_known: true,
                ipv4_metric: 5,
                ipv6_metric: 5,
                xdomain_peer_device_name: None,
                xdomain_peer_unique_id: None,
                external_owner: None,
                addresses: vec![],
                neighbors: vec![],
            }],
        };

        assert!(verify_adapter_identity_in_snapshot(
            &snap,
            23,
            "{11111111-2222-3333-4444-555555555555}",
            "AA-BB-CC-DD-EE-01"
        )
        .is_ok());

        let guid_err = verify_adapter_identity_in_snapshot(
            &snap,
            23,
            "{99999999-2222-3333-4444-555555555555}",
            "AA-BB-CC-DD-EE-01",
        )
        .unwrap_err();
        assert!(guid_err.to_string().contains("GUID changed"));

        let mac_err = verify_adapter_identity_in_snapshot(
            &snap,
            23,
            "{11111111-2222-3333-4444-555555555555}",
            "AA-BB-CC-DD-EE-99",
        )
        .unwrap_err();
        assert!(mac_err.to_string().contains("MAC changed"));
    }
}
