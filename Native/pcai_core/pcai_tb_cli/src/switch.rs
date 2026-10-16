// trace: PENDING-FABRIC-F09 [pending-tpm-intake]
// trace: PENDING-FABRIC-F10 [pending-tpm-intake]
// trace: PENDING-FABRIC-F11 [pending-tpm-intake]
// trace: PENDING-FABRIC-F12 [pending-tpm-intake]
// trace: PENDING-FABRIC-F13 [pending-tpm-intake]
// trace: TR-MOB-I01 [implements]
use crate::identity::{
    canonicalize_peer_name, detect_l2_bridge_cycles, load_peer_registry, local_hostname, parse_ipv4_cidr,
    IpamAllocations, SfpSwitchTopology, SharedSwitchSubnet, TopologyLinkSpec,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct SwitchStatusReport {
    pub local_host: String,
    pub os_family: String,
    pub registry_path: Option<String>,
    pub matched_topology: Option<SfpSwitchTopology>,
    pub ipam_allocations: Option<IpamAllocations>,
    pub topology_links: Vec<TopologyLinkSpec>,
    pub local_usb4_routers: Vec<Usb4RouterNode>,
    pub shared_subnet_routes: Vec<SharedSubnetRouteStatus>,
    pub linux_switch_custody: Option<LinuxSwitchCustodyStatus>,
    pub linux_usb4_extension_custody: Option<LinuxUsb4ExtensionCustodyStatus>,
    pub linux_canary_ula_status: Vec<CanaryEndpointLiveStatus>,
    pub windows_metric_hierarchy: Option<WindowsMetricHierarchyReport>,
    pub custody_valid: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Usb4RouterNode {
    pub router_id: String,
    pub parent_router_id: Option<String>,
    pub vendor_name: Option<String>,
    pub device_name: String,
    pub unique_id: Option<String>,
    pub generation: Option<u32>,
    pub rx_lanes: Option<u32>,
    pub tx_lanes: Option<u32>,
    pub rx_speed_gbps: Option<u32>,
    pub tx_speed_gbps: Option<u32>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct SharedSubnetRouteStatus {
    pub subnet: String,
    pub gateway_host: String,
    pub gateway_ipv4: String,
    pub role: String,
    pub is_local_gateway: bool,
    pub route_present: bool,
    pub route_applied: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcaiLinuxUsb4Observation {
    pub schema: String,
    pub status: String,
    pub readiness_scope: String,
    pub network_ready: Option<bool>,
    pub namespace: String,
    pub bridge: String,
    pub usb4_ifname: String,
    pub root_veth: String,
    pub bridge_veth: String,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VigilMs01Usb4CheckReceipt {
    pub schema: String,
    pub phase: String,
    pub ready: bool,
    pub status: String,
    #[serde(default)]
    pub readiness_scope: Option<String>,
    #[serde(default)]
    pub network_ready: Option<bool>,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub bridge: Option<String>,
    #[serde(default)]
    pub control_veth: Option<String>,
    #[serde(default)]
    pub interface: Option<String>,
    #[serde(default)]
    pub peer_uuid: Option<String>,
    #[serde(default)]
    pub boot: Option<String>,
    #[serde(default)]
    pub parent_owner: Option<String>,
    #[serde(default)]
    pub owner: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

pub fn parse_and_validate_ms01_usb4_check_receipt(raw: &str) -> Result<VigilMs01Usb4CheckReceipt> {
    let receipt: VigilMs01Usb4CheckReceipt =
        serde_json::from_str(raw).context("Invalid JSON in ms01-usb4-extension check receipt")?;
    if receipt.schema != "vigil.ms01-usb4-check/v1" {
        anyhow::bail!(
            "Unexpected schema '{}' in ms01-usb4-extension receipt (expected 'vigil.ms01-usb4-check/v1')",
            receipt.schema
        );
    }
    if receipt.ready || receipt.status == "CUSTODY_VERIFIED" {
        if !receipt.ready || receipt.status != "CUSTODY_VERIFIED" || receipt.phase != "running" {
            anyhow::bail!(
                "Inconsistent ready/status/phase in vigil.ms01-usb4-check/v1 receipt: ready={}, status={}, phase={}",
                receipt.ready,
                receipt.status,
                receipt.phase
            );
        }
        for (field_name, val_opt) in [
            ("namespace", &receipt.namespace),
            ("bridge", &receipt.bridge),
            ("control_veth", &receipt.control_veth),
            ("interface", &receipt.interface),
            ("peer_uuid", &receipt.peer_uuid),
            ("boot", &receipt.boot),
            ("parent_owner", &receipt.parent_owner),
            ("owner", &receipt.owner),
        ] {
            if val_opt.as_deref().map(str::trim).unwrap_or("").is_empty() {
                anyhow::bail!("Verified vigil.ms01-usb4-check/v1 receipt missing required field '{field_name}'");
            }
        }
    }
    Ok(receipt)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VnvRequirementGateRow {
    pub trace_id: String,
    pub requirement_id: String,
    pub tpm_admission_status: String,
    pub acceptance_credit: bool,
    pub description: String,
    pub status: String,
    pub technical_verdict: String,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VigilHilDeviceEvidence {
    pub schema: String,
    pub host: String,
    pub os_family: String,
    pub tpm_admission_status: String,
    pub acceptance_credit: bool,
    pub overall_status: String,
    pub custody_valid: bool,
    pub usb4_check_receipt: Option<VigilMs01Usb4CheckReceipt>,
    pub linux_usb4_observation: Option<PcaiLinuxUsb4Observation>,
    pub requirement_gates: Vec<VnvRequirementGateRow>,
    pub local_usb4_routers: Vec<Usb4RouterNode>,
    pub shared_subnet_routes: Vec<SharedSubnetRouteStatus>,
    pub topology_links: Vec<TopologyLinkSpec>,
    pub switch_status: SwitchStatusReport,
    pub output_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct LinuxSwitchCustodyStatus {
    pub state_file_path: String,
    pub state_file_present: bool,
    pub phase_running: bool,
    pub boot_id_matches: bool,
    pub live_boot_id: Option<String>,
    pub recorded_boot_id: Option<String>,
    pub namespace: String,
    pub namespace_inode_matches: bool,
    pub live_namespace_inode: Option<u64>,
    pub recorded_namespace_inode: Option<u64>,
    pub bridge: String,
    pub bridge_mac_matches: bool,
    pub bridge_alias_matches: bool,
    pub live_bridge_mac: Option<String>,
    pub recorded_bridge_mac: Option<String>,
    pub ports: Vec<LinuxSwitchPortCustody>,
    pub all_ports_valid: bool,
    #[serde(default)]
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct LinuxSwitchPortCustody {
    pub name: String,
    pub pci: String,
    pub expected_mac: String,
    pub live_mac: Option<String>,
    pub live_ifindex: Option<u32>,
    pub recorded_ifindex: Option<u32>,
    pub live_master: Option<String>,
    pub live_ifalias: Option<String>,
    pub operstate: Option<String>,
    pub carrier: Option<u32>,
    pub address_count: usize,
    pub custody_valid: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct LinuxUsb4ExtensionCustodyStatus {
    pub state_file_path: String,
    pub state_file_present: bool,
    pub boot_id_matches: bool,
    pub namespace: String,
    pub namespace_inode_matches: bool,
    pub bridge: String,
    pub usb4_ifname: String,
    pub usb4_in_bridge: bool,
    pub usb4_ifalias: Option<String>,
    pub usb4_address_count: usize,
    pub root_veth: String,
    pub root_veth_ifindex: Option<u32>,
    pub root_veth_link_index: Option<u32>,
    pub root_veth_ifalias: Option<String>,
    pub root_veth_addresses: Vec<String>,
    pub bridge_veth: String,
    pub bridge_veth_ifindex: Option<u32>,
    pub bridge_veth_link_index: Option<u32>,
    pub bridge_veth_in_bridge: bool,
    pub bridge_veth_ifalias: Option<String>,
    pub bridge_veth_address_count: usize,
    pub veth_symmetry_valid: bool,
    pub custody_valid: bool,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct CanaryEndpointLiveStatus {
    pub host: String,
    pub interface: String,
    pub expected_ula: String,
    pub role: String,
    pub assigned_on_interface: bool,
    pub policy_table: Option<u32>,
    pub policy_priority: Option<u32>,
    pub policy_rule_present: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct WindowsInterfaceMetricEntry {
    pub interface_index: u32,
    pub interface_alias: String,
    pub media_tier: String,
    pub metric: Option<u32>,
    pub mtu_bytes: Option<u32>,
    pub state: String,
    pub ipv4_subnets: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct WindowsMetricHierarchyReport {
    pub interfaces: Vec<WindowsInterfaceMetricEntry>,
    pub hierarchy_valid: bool,
    pub same_subnet_conflicts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SwitchObserveSnapshot {
    pub schema_version: u32,
    pub payload_capture: bool,
    pub boot_id: String,
    pub namespace: String,
    pub namespace_inode: u64,
    pub bridge: String,
    pub bridge_mac: String,
    pub ports: BTreeMap<String, serde_json::Value>,
    pub links: Vec<serde_json::Value>,
    pub bridge_ports: Vec<serde_json::Value>,
    pub fdb: Vec<serde_json::Value>,
    pub vlans: Vec<serde_json::Value>,
    pub output_path: Option<String>,
    pub recorded: bool,
}

pub fn run_cmd_json(prog: &str, args: &[&str]) -> Result<serde_json::Value> {
    let out = Command::new(prog)
        .args(args)
        .output()
        .with_context(|| format!("Failed to spawn '{prog} {}'", args.join(" ")))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stdout = String::from_utf8_lossy(&out.stdout);
        anyhow::bail!(
            "Command '{prog} {}' failed (exit {:?}): {} {}",
            args.join(" "),
            out.status.code(),
            stderr.trim(),
            stdout.trim()
        );
    }
    let parsed: serde_json::Value = serde_json::from_slice(&out.stdout)
        .with_context(|| format!("Invalid JSON from '{prog} {}'", args.join(" ")))?;
    Ok(parsed)
}

pub fn validate_veth_pair_symmetry(
    root_ifindex: Option<u32>,
    root_link_index: Option<u32>,
    ns_ifindex: Option<u32>,
    ns_link_index: Option<u32>,
) -> bool {
    match (root_ifindex, root_link_index, ns_ifindex, ns_link_index) {
        (Some(r_if), Some(r_peer), Some(n_if), Some(n_peer)) => r_if == n_peer && n_if == r_peer && r_if != n_if,
        _ => false,
    }
}

pub fn evaluate_windows_metric_hierarchy(entries: &[WindowsInterfaceMetricEntry]) -> (bool, Vec<String>) {
    let mut conflicts = Vec::new();
    let connected: Vec<&WindowsInterfaceMetricEntry> = entries
        .iter()
        .filter(|e| e.state.eq_ignore_ascii_case("connected") || e.state.eq_ignore_ascii_case("up"))
        .collect();

    if entries.is_empty() || connected.is_empty() {
        conflicts.push("No connected Windows network interfaces observed in metric census".to_string());
        return (false, conflicts);
    }

    for entry in &connected {
        if matches!(entry.media_tier.as_str(), "usb4-p2p" | "wired-lan" | "wifi") && entry.metric.is_none() {
            conflicts.push(format!(
                "Connected interface '{}' (tier '{}') has missing/unknown metric",
                entry.interface_alias, entry.media_tier
            ));
        }
    }

    let min_usb4_metric = connected
        .iter()
        .filter(|e| e.media_tier == "usb4-p2p")
        .filter_map(|e| e.metric)
        .max();
    let min_wired_metric = connected
        .iter()
        .filter(|e| e.media_tier == "wired-lan")
        .filter_map(|e| e.metric)
        .min();
    let max_wired_metric = connected
        .iter()
        .filter(|e| e.media_tier == "wired-lan")
        .filter_map(|e| e.metric)
        .max();
    let min_wifi_metric = connected
        .iter()
        .filter(|e| e.media_tier == "wifi")
        .filter_map(|e| e.metric)
        .min();

    if let (Some(u_max), Some(w_min)) = (min_usb4_metric, min_wired_metric) {
        if u_max >= w_min {
            conflicts.push(format!(
                "USB4 P2P metric ({u_max}) is not strictly lower than Wired Ethernet/SFP+ metric ({w_min})"
            ));
        }
    }

    if let (Some(w_max), Some(wifi_min)) = (max_wired_metric, min_wifi_metric) {
        if w_max >= wifi_min {
            conflicts.push(format!(
                "Wired Ethernet/SFP+ metric ({w_max}) is not strictly lower than Wi-Fi metric ({wifi_min}); multi-homed LAN traffic may route over Wi-Fi"
            ));
        }
    }

    for wired in connected.iter().filter(|e| e.media_tier == "wired-lan") {
        for wifi in connected.iter().filter(|e| e.media_tier == "wifi") {
            for subnet in &wired.ipv4_subnets {
                if wifi.ipv4_subnets.contains(subnet) {
                    match (wired.metric, wifi.metric) {
                        (Some(wm), Some(wfm)) if wm < wfm => {}
                        (wm, wfm) => {
                            conflicts.push(format!(
                                "Same-subnet multi-homed conflict on {subnet} between '{}' (wired metric {:?}) and '{}' (wifi metric {:?})",
                                wired.interface_alias, wm, wifi.interface_alias, wfm
                            ));
                        }
                    }
                }
            }
        }
    }

    (conflicts.is_empty(), conflicts)
}

pub fn evaluate_linux_switch_custody_from_json(
    topo: &SfpSwitchTopology,
    state_file_path: String,
    state_val: Option<&serde_json::Value>,
    ns_links: Option<&serde_json::Value>,
    ns_addrs: Option<&serde_json::Value>,
    live_boot_id: Option<String>,
    live_ns_inode: Option<u64>,
) -> LinuxSwitchCustodyStatus {
    let Some(state_val) = state_val else {
        return LinuxSwitchCustodyStatus {
            state_file_path,
            state_file_present: false,
            phase_running: false,
            boot_id_matches: false,
            live_boot_id,
            recorded_boot_id: None,
            namespace: topo.namespace.clone(),
            namespace_inode_matches: false,
            live_namespace_inode: live_ns_inode,
            recorded_namespace_inode: None,
            bridge: topo.bridge.clone(),
            bridge_mac_matches: false,
            bridge_alias_matches: false,
            live_bridge_mac: None,
            recorded_bridge_mac: None,
            ports: Vec::new(),
            all_ports_valid: false,
            errors: vec!["Switch custody journal not present or unreadable".to_string()],
        };
    };

    let mut errors = Vec::new();

    if let Some(schema_num) = state_val.get("schema").and_then(|v| v.as_u64()) {
        if schema_num != 1 {
            errors.push(format!("Unsupported switch custody journal schema {schema_num}"));
        }
    }

    let phase = state_val.get("phase").and_then(|v| v.as_str());
    let phase_running = phase == Some("running");
    if !phase_running {
        errors.push(format!(
            "Switch custody phase is not 'running' (got {:?})",
            phase
        ));
    }

    let recorded_owner = state_val
        .get("owner")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());
    if recorded_owner.is_none() {
        errors.push("Switch custody journal missing non-empty 'owner' token".to_string());
    }

    let recorded_boot_id = state_val
        .get("boot_id")
        .or_else(|| state_val.get("boot"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let boot_id_matches = matches!(
        (&live_boot_id, &recorded_boot_id),
        (Some(live), Some(rec)) if live == rec
    );
    if !boot_id_matches {
        errors.push("Switch custody boot_id mismatch".to_string());
    }

    let recorded_ns_inode = state_val
        .get("namespace_inode")
        .or_else(|| state_val.get("inode"))
        .and_then(|v| v.as_u64());
    let namespace_inode_matches = matches!(
        (live_ns_inode, recorded_ns_inode),
        (Some(live), Some(rec)) if live == rec
    );
    if !namespace_inode_matches {
        errors.push("Switch custody namespace_inode mismatch".to_string());
    }

    let recorded_bridge_mac = state_val
        .get("bridge_mac")
        .and_then(|v| v.as_str())
        .map(|s| s.to_ascii_lowercase())
        .or_else(|| topo.sfp_ports.first().map(|p| p.mac.to_ascii_lowercase()));

    let find_ns_link = |ifname: &str| -> Option<serde_json::Value> {
        ns_links?
            .as_array()?
            .iter()
            .find(|item| item.get("ifname").and_then(|v| v.as_str()) == Some(ifname))
            .cloned()
    };

    let count_ns_addrs = |ifname: &str| -> usize {
        ns_addrs
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                arr.iter()
                    .find(|item| item.get("ifname").and_then(|v| v.as_str()) == Some(ifname))
            })
            .and_then(|item| item.get("addr_info"))
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0)
    };

    let bridge_link = find_ns_link(&topo.bridge);
    let live_bridge_mac = bridge_link.as_ref().and_then(|l| {
        l.get("address")
            .and_then(|v| v.as_str())
            .map(|s| s.to_ascii_lowercase())
    });

    let bridge_mac_matches = matches!(
        (&live_bridge_mac, &recorded_bridge_mac),
        (Some(live), Some(rec)) if live == rec
    );
    if !bridge_mac_matches {
        errors.push("Switch custody bridge_mac mismatch".to_string());
    }

    let expected_owner_alias = recorded_owner.map(|owner| format!("ms01-switch:{owner}"));
    let live_bridge_alias = bridge_link
        .as_ref()
        .and_then(|l| l.get("ifalias"))
        .and_then(|v| v.as_str());
    let bridge_alias_matches = match (&expected_owner_alias, live_bridge_alias) {
        (Some(exp), Some(live)) => live == exp,
        _ => false,
    };
    if !bridge_alias_matches {
        errors.push(format!(
            "Bridge '{}' ifalias mismatch: expected {:?}, got {:?}",
            topo.bridge, expected_owner_alias, live_bridge_alias
        ));
    }

    let recorded_ports = state_val
        .get("ports")
        .or_else(|| state_val.get("original"))
        .and_then(|v| v.as_object());
    let mut port_reports = Vec::new();
    let mut all_ports_valid = !topo.sfp_ports.is_empty() && phase_running && bridge_alias_matches && recorded_owner.is_some();

    for spec in &topo.sfp_ports {
        let rec_port = recorded_ports.and_then(|m| m.get(&spec.name));
        let recorded_ifindex = rec_port
            .and_then(|p| p.get("ifindex"))
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        let live_link = find_ns_link(&spec.name);
        let live_ifindex = live_link
            .as_ref()
            .and_then(|l| l.get("ifindex"))
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        let live_mac = live_link
            .as_ref()
            .and_then(|l| l.get("address"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_ascii_lowercase());
        let live_master = live_link
            .as_ref()
            .and_then(|l| l.get("master"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let live_ifalias = live_link
            .as_ref()
            .and_then(|l| l.get("ifalias"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let operstate = live_link
            .as_ref()
            .and_then(|l| l.get("operstate"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let carrier = live_link
            .as_ref()
            .and_then(|l| l.get("carrier"))
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        let addr_count = count_ns_addrs(&spec.name);

        let mut err = None;
        let mac_ok = live_mac.as_deref() == Some(spec.mac.to_ascii_lowercase().as_str());
        let ifindex_ok = matches!((live_ifindex, recorded_ifindex), (Some(a), Some(b)) if a == b);
        let master_ok = live_master.as_deref() == Some(topo.bridge.as_str());
        let alias_ok = match (&expected_owner_alias, live_ifalias.as_deref()) {
            (Some(exp), Some(live)) => live == exp.as_str(),
            _ => false,
        };
        let addr_ok = addr_count == 0;

        if !mac_ok {
            err = Some(format!("MAC mismatch: expected {}, got {:?}", spec.mac, live_mac));
        } else if !ifindex_ok {
            err = Some(format!(
                "ifindex mismatch: live {:?} != recorded {:?}",
                live_ifindex, recorded_ifindex
            ));
        } else if !master_ok {
            err = Some(format!(
                "master mismatch: expected {}, got {:?}",
                topo.bridge, live_master
            ));
        } else if !alias_ok {
            err = Some(format!(
                "missing or mismatched exact ifalias {:?} (got {:?})",
                expected_owner_alias, live_ifalias
            ));
        } else if !addr_ok {
            err = Some(format!("switched SFP+ port has {addr_count} IP addresses assigned"));
        }

        let valid = mac_ok && ifindex_ok && master_ok && alias_ok && addr_ok;
        if !valid {
            all_ports_valid = false;
        }

        port_reports.push(LinuxSwitchPortCustody {
            name: spec.name.clone(),
            pci: spec.pci.clone(),
            expected_mac: spec.mac.to_ascii_lowercase(),
            live_mac,
            live_ifindex,
            recorded_ifindex,
            live_master,
            live_ifalias,
            operstate,
            carrier,
            address_count: addr_count,
            custody_valid: valid,
            error: err,
        });
    }

    LinuxSwitchCustodyStatus {
        state_file_path,
        state_file_present: true,
        phase_running,
        boot_id_matches,
        live_boot_id,
        recorded_boot_id,
        namespace: topo.namespace.clone(),
        namespace_inode_matches,
        live_namespace_inode: live_ns_inode,
        recorded_namespace_inode: recorded_ns_inode,
        bridge: topo.bridge.clone(),
        bridge_mac_matches,
        bridge_alias_matches,
        live_bridge_mac,
        recorded_bridge_mac,
        ports: port_reports,
        all_ports_valid,
        errors,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_linux_usb4_extension_custody_from_json(
    topo: &SfpSwitchTopology,
    state_file_path: String,
    state_val: Option<&serde_json::Value>,
    root_links: Option<&serde_json::Value>,
    root_addrs: Option<&serde_json::Value>,
    ns_links: Option<&serde_json::Value>,
    ns_addrs: Option<&serde_json::Value>,
    live_boot_id: Option<&str>,
    live_ns_inode: Option<u64>,
) -> Option<LinuxUsb4ExtensionCustodyStatus> {
    let ext_spec = topo.usb4_extensions.first()?;
    let Some(state_val) = state_val else {
        return Some(LinuxUsb4ExtensionCustodyStatus {
            state_file_path,
            state_file_present: false,
            boot_id_matches: false,
            namespace: topo.namespace.clone(),
            namespace_inode_matches: false,
            bridge: topo.bridge.clone(),
            usb4_ifname: ext_spec.interface.clone(),
            usb4_in_bridge: false,
            usb4_ifalias: None,
            usb4_address_count: 0,
            root_veth: ext_spec.root_veth.clone(),
            root_veth_ifindex: None,
            root_veth_link_index: None,
            root_veth_ifalias: None,
            root_veth_addresses: Vec::new(),
            bridge_veth: ext_spec.bridge_veth.clone(),
            bridge_veth_ifindex: None,
            bridge_veth_link_index: None,
            bridge_veth_in_bridge: false,
            bridge_veth_ifalias: None,
            bridge_veth_address_count: 0,
            veth_symmetry_valid: false,
            custody_valid: false,
            errors: vec!["State file not present or invalid JSON".to_string()],
        });
    };

    let mut errors = Vec::new();

    if let Some(schema_num) = state_val.get("schema").and_then(|v| v.as_u64()) {
        if schema_num != 1 {
            errors.push(format!("Unsupported usb4-extension journal schema {schema_num}"));
        }
    }

    let phase = state_val.get("phase").and_then(|v| v.as_str());
    if phase != Some("running") {
        errors.push(format!(
            "usb4-extension phase is not 'running' (got {:?})",
            phase
        ));
    }

    let rec_boot_id = state_val
        .get("boot_id")
        .or_else(|| state_val.get("boot"))
        .and_then(|v| v.as_str());
    let boot_id_matches = matches!((live_boot_id, rec_boot_id), (Some(a), Some(b)) if a == b);

    let rec_ns_inode = state_val
        .get("namespace_inode")
        .or_else(|| state_val.get("parent_inode"))
        .or_else(|| state_val.get("inode"))
        .and_then(|v| v.as_u64());
    let namespace_inode_matches = matches!((live_ns_inode, rec_ns_inode), (Some(a), Some(b)) if a == b);
    let rec_owner = state_val
        .get("owner")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let rec_parent_owner = state_val
        .get("parent_owner")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());

    if rec_owner.is_none() {
        errors.push("usb4-extension journal missing non-empty 'owner'".to_string());
    }
    if rec_parent_owner.is_none() {
        errors.push("usb4-extension journal missing non-empty 'parent_owner'".to_string());
    }

    if let Some(rec_peer_uuid) = state_val.get("peer_uuid").and_then(|v| v.as_str()) {
        if !rec_peer_uuid.eq_ignore_ascii_case(&ext_spec.peer_uuid) {
            errors.push(format!(
                "usb4-extension peer_uuid mismatch: expected '{}', got '{rec_peer_uuid}'",
                ext_spec.peer_uuid
            ));
        }
    }
    if let Some(rec_path) = state_val
        .get("path")
        .or_else(|| state_val.get("sysfs_path"))
        .and_then(|v| v.as_str())
    {
        if rec_path != ext_spec.sysfs_path {
            errors.push(format!(
                "usb4-extension sysfs path mismatch: expected '{}', got '{rec_path}'",
                ext_spec.sysfs_path
            ));
        }
    }
    if let Some(rec_driver) = state_val.get("driver").and_then(|v| v.as_str()) {
        if rec_driver != ext_spec.driver {
            errors.push(format!(
                "usb4-extension driver mismatch: expected '{}', got '{rec_driver}'",
                ext_spec.driver
            ));
        }
    }

    let find_link = |links: Option<&serde_json::Value>, ifname: &str| -> Option<serde_json::Value> {
        links?
            .as_array()?
            .iter()
            .find(|item| item.get("ifname").and_then(|v| v.as_str()) == Some(ifname))
            .cloned()
    };

    let get_addrs = |addrs: Option<&serde_json::Value>, ifname: &str| -> Vec<String> {
        addrs
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                arr.iter()
                    .find(|item| item.get("ifname").and_then(|v| v.as_str()) == Some(ifname))
            })
            .and_then(|item| item.get("addr_info"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|a| {
                        let local = a.get("local")?.as_str()?;
                        let plen = a.get("prefixlen")?.as_u64()?;
                        Some(format!("{local}/{plen}"))
                    })
                    .collect()
            })
            .unwrap_or_default()
    };

    // Displaced link checks: USB4 and bridge_veth must NOT exist in root ns; root_veth must NOT exist in switch ns
    if find_link(root_links, &ext_spec.interface).is_some() {
        errors.push(format!(
            "Displaced USB4 interface '{}' still present in root namespace",
            ext_spec.interface
        ));
    }
    if find_link(root_links, &ext_spec.bridge_veth).is_some() {
        errors.push(format!(
            "Displaced bridge veth '{}' present in root namespace",
            ext_spec.bridge_veth
        ));
    }
    if find_link(ns_links, &ext_spec.root_veth).is_some() {
        errors.push(format!(
            "Displaced root veth '{}' present inside switch namespace '{}'",
            ext_spec.root_veth, topo.namespace
        ));
    }

    let usb4_link = find_link(ns_links, &ext_spec.interface);
    let usb4_in_bridge = usb4_link
        .as_ref()
        .and_then(|l| l.get("master"))
        .and_then(|v| v.as_str())
        == Some(topo.bridge.as_str());
    let usb4_ifalias = usb4_link
        .as_ref()
        .and_then(|l| l.get("ifalias"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let usb4_address_count = get_addrs(ns_addrs, &ext_spec.interface).len();

    if let Some(orig) = state_val.get("original").and_then(|v| v.as_object()) {
        if let Some(orig_ifindex) = orig.get("ifindex").and_then(|v| v.as_u64()) {
            let live_usb_ifindex = usb4_link.as_ref().and_then(|l| l.get("ifindex")).and_then(|v| v.as_u64());
            if live_usb_ifindex != Some(orig_ifindex) {
                errors.push(format!(
                    "USB4 interface '{}' ifindex changed: live {:?} != original {}",
                    ext_spec.interface, live_usb_ifindex, orig_ifindex
                ));
            }
        }
        if let Some(orig_mac) = orig.get("address").and_then(|v| v.as_str()) {
            let live_usb_mac = usb4_link.as_ref().and_then(|l| l.get("address")).and_then(|v| v.as_str());
            if live_usb_mac.map(|s| s.eq_ignore_ascii_case(orig_mac)) != Some(true) {
                errors.push(format!(
                    "USB4 interface '{}' MAC changed: live {:?} != original '{}'",
                    ext_spec.interface, live_usb_mac, orig_mac
                ));
            }
        }
    }

    if let Some(rec_bridge_mac) = state_val.get("bridge_mac").and_then(|v| v.as_str()) {
        let live_br_mac = find_link(ns_links, &topo.bridge)
            .and_then(|l| l.get("address").and_then(|v| v.as_str()).map(str::to_string));
        if live_br_mac.as_deref().map(|s| s.eq_ignore_ascii_case(rec_bridge_mac)) != Some(true) {
            errors.push(format!(
                "Parent bridge '{}' MAC mismatch: live {:?} != recorded '{}'",
                topo.bridge, live_br_mac, rec_bridge_mac
            ));
        }
    }
    if let Some(parent_owner) = rec_parent_owner {
        let expected_br_alias = format!("ms01-switch:{parent_owner}");
        let live_br_alias = find_link(ns_links, &topo.bridge)
            .and_then(|l| l.get("ifalias").and_then(|v| v.as_str()).map(str::to_string));
        if live_br_alias.as_deref() != Some(expected_br_alias.as_str()) {
            errors.push(format!(
                "Parent bridge '{}' ifalias mismatch: live {:?} != expected '{}'",
                topo.bridge, live_br_alias, expected_br_alias
            ));
        }
    }

    let root_veth_link = find_link(root_links, &ext_spec.root_veth);
    let root_veth_ifindex = root_veth_link
        .as_ref()
        .and_then(|l| l.get("ifindex"))
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    let root_veth_link_index = root_veth_link
        .as_ref()
        .and_then(|l| l.get("link_index"))
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    let root_veth_ifalias = root_veth_link
        .as_ref()
        .and_then(|l| l.get("ifalias"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let root_veth_addresses = get_addrs(root_addrs, &ext_spec.root_veth);
    if root_veth_link
        .as_ref()
        .and_then(|l| l.get("master"))
        .and_then(|v| v.as_str())
        .is_some()
    {
        errors.push(format!(
            "Root veth '{}' must not be enslaved to a master bridge",
            ext_spec.root_veth
        ));
    }

    let bridge_veth_link = find_link(ns_links, &ext_spec.bridge_veth);
    let bridge_veth_ifindex = bridge_veth_link
        .as_ref()
        .and_then(|l| l.get("ifindex"))
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    let bridge_veth_link_index = bridge_veth_link
        .as_ref()
        .and_then(|l| l.get("link_index"))
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    let bridge_veth_in_bridge = bridge_veth_link
        .as_ref()
        .and_then(|l| l.get("master"))
        .and_then(|v| v.as_str())
        == Some(topo.bridge.as_str());
    let bridge_veth_ifalias = bridge_veth_link
        .as_ref()
        .and_then(|l| l.get("ifalias"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let bridge_veth_address_count = get_addrs(ns_addrs, &ext_spec.bridge_veth).len();

    let veth_symmetry_valid = validate_veth_pair_symmetry(
        root_veth_ifindex,
        root_veth_link_index,
        bridge_veth_ifindex,
        bridge_veth_link_index,
    );

    if !boot_id_matches {
        errors.push("boot_id mismatch in usb4-extension state".to_string());
    }
    if !namespace_inode_matches {
        errors.push("namespace_inode mismatch in usb4-extension state".to_string());
    }
    if !usb4_in_bridge {
        errors.push(format!(
            "USB4 interface '{}' is not enslaved to bridge '{}'",
            ext_spec.interface, topo.bridge
        ));
    }
    if usb4_address_count != 0 {
        errors.push(format!(
            "USB4 interface '{}' has {usb4_address_count} IP addresses inside switch namespace",
            ext_spec.interface
        ));
    }
    if !bridge_veth_in_bridge {
        errors.push(format!(
            "Bridge veth '{}' is not enslaved to bridge '{}'",
            ext_spec.bridge_veth, topo.bridge
        ));
    }
    if bridge_veth_address_count != 0 {
        errors.push(format!(
            "Bridge veth '{}' has {bridge_veth_address_count} IP addresses inside switch namespace",
            ext_spec.bridge_veth
        ));
    }
    if !veth_symmetry_valid {
        errors.push(format!(
            "veth pair symmetry invalid: root ({:?} -> {:?}) vs bridge ({:?} -> {:?})",
            root_veth_ifindex, root_veth_link_index, bridge_veth_ifindex, bridge_veth_link_index
        ));
    }

    let expected_control_cidr = format!("{}/30", ext_spec.local_control_ipv4);
    if root_veth_addresses != vec![expected_control_cidr.clone()] {
        errors.push(format!(
            "Root veth '{}' addresses {:?} do not match exact control address ['{}']",
            ext_spec.root_veth, root_veth_addresses, expected_control_cidr
        ));
    }

    for (label, link_opt) in [
        (&ext_spec.interface, &usb4_link),
        (&ext_spec.root_veth, &root_veth_link),
        (&ext_spec.bridge_veth, &bridge_veth_link),
    ] {
        if let Some(link) = link_opt {
            if let Some(mtu) = link.get("mtu").and_then(|v| v.as_u64()) {
                if mtu != 1500 {
                    errors.push(format!("Interface '{label}' MTU is {mtu}, expected 1500"));
                }
            }
            if let Some(flags) = link.get("flags").and_then(|v| v.as_array()) {
                let is_up = flags.iter().any(|f| f.as_str() == Some("UP"));
                if !is_up {
                    errors.push(format!("Interface '{label}' flags do not include UP"));
                }
            }
        }
    }

    let expected_usb4_alias = rec_owner.map(|owner| format!("ms01-usb4:{owner}"));
    for (label, alias_opt) in [
        (&ext_spec.interface, &usb4_ifalias),
        (&ext_spec.root_veth, &root_veth_ifalias),
        (&ext_spec.bridge_veth, &bridge_veth_ifalias),
    ] {
        let alias_matches = match (&expected_usb4_alias, alias_opt.as_deref()) {
            (Some(exp), Some(live)) => live == exp.as_str(),
            _ => false,
        };
        if !alias_matches {
            errors.push(format!(
                "Interface '{label}' missing or mismatched exact {:?} ifalias (got {:?})",
                expected_usb4_alias, alias_opt
            ));
        }
    }

    let custody_valid = errors.is_empty();
    Some(LinuxUsb4ExtensionCustodyStatus {
        state_file_path,
        state_file_present: true,
        boot_id_matches,
        namespace: topo.namespace.clone(),
        namespace_inode_matches,
        bridge: topo.bridge.clone(),
        usb4_ifname: ext_spec.interface.clone(),
        usb4_in_bridge,
        usb4_ifalias,
        usb4_address_count,
        root_veth: ext_spec.root_veth.clone(),
        root_veth_ifindex,
        root_veth_link_index,
        root_veth_ifalias,
        root_veth_addresses,
        bridge_veth: ext_spec.bridge_veth.clone(),
        bridge_veth_ifindex,
        bridge_veth_link_index,
        bridge_veth_in_bridge,
        bridge_veth_ifalias,
        bridge_veth_address_count,
        veth_symmetry_valid,
        custody_valid,
        errors,
    })
}

#[cfg(target_os = "linux")]
pub fn collect_local_usb4_router_tree() -> Vec<Usb4RouterNode> {
    let mut nodes = Vec::new();
    let tb_dir = Path::new("/sys/bus/thunderbolt/devices");
    let Ok(entries) = fs::read_dir(tb_dir) else {
        return nodes;
    };
    let read_trim = |p: PathBuf| -> Option<String> {
        fs::read_to_string(p)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    let mut dir_names: Vec<String> = entries
        .filter_map(|e| e.ok().map(|de| de.file_name().to_string_lossy().into_owned()))
        .filter(|name| !name.contains(':') && !name.starts_with("domain"))
        .collect();
    dir_names.sort();

    for name in dir_names {
        let dev_path = tb_dir.join(&name);
        let parent_router_id = fs::canonicalize(&dev_path)
            .ok()
            .and_then(|canon| {
                canon
                    .parent()
                    .map(|p| p.file_name().unwrap_or_default().to_string_lossy().into_owned())
            })
            .filter(|p| !p.starts_with("domain") && !p.starts_with("0000:"));
        let vendor_name = read_trim(dev_path.join("vendor_name"));
        let device_name = read_trim(dev_path.join("device_name")).unwrap_or_else(|| name.clone());
        let unique_id = read_trim(dev_path.join("unique_id"));
        let generation = read_trim(dev_path.join("generation")).and_then(|s| s.parse::<u32>().ok());
        let rx_lanes = read_trim(dev_path.join("rx_lanes")).and_then(|s| s.parse::<u32>().ok());
        let tx_lanes = read_trim(dev_path.join("tx_lanes")).and_then(|s| s.parse::<u32>().ok());
        let rx_speed_gbps =
            read_trim(dev_path.join("rx_speed")).and_then(|s| s.split('.').next()?.trim().parse::<u32>().ok());
        let tx_speed_gbps =
            read_trim(dev_path.join("tx_speed")).and_then(|s| s.split('.').next()?.trim().parse::<u32>().ok());
        let status = if name.starts_with("usb4_port") {
            read_trim(dev_path.join("link")).unwrap_or_else(|| "unknown".to_string())
        } else {
            match read_trim(dev_path.join("authorized")).as_deref() {
                Some("1") => "authorized".to_string(),
                Some("0") => "unauthorized".to_string(),
                _ => "present".to_string(),
            }
        };
        nodes.push(Usb4RouterNode {
            router_id: name,
            parent_router_id,
            vendor_name,
            device_name,
            unique_id,
            generation,
            rx_lanes,
            tx_lanes,
            rx_speed_gbps,
            tx_speed_gbps,
            status,
        });
    }
    nodes
}

#[cfg(target_os = "windows")]
#[derive(Default)]
struct RawWinUsb4Node {
    router_id: String,
    leaf: String,
    parent_prefix: Option<String>,
    friendly: Option<String>,
    desc: Option<String>,
    container: Option<String>,
}

#[cfg(target_os = "windows")]
pub fn collect_local_usb4_router_tree() -> Vec<Usb4RouterNode> {
    let Ok(out) = Command::new("reg")
        .args(["query", r"HKLM\SYSTEM\CurrentControlSet\Enum\USB4", "/s"])
        .output()
    else {
        return Vec::new();
    };
    let stdout = String::from_utf8_lossy(&out.stdout);
    let prefix = r"HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Enum\";
    let mut raw_nodes: Vec<RawWinUsb4Node> = Vec::new();
    let mut cur: Option<RawWinUsb4Node> = None;

    for line in stdout.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let parts: Vec<&str> = rest.split('\\').collect();
            if parts.len() == 3 && parts[0].eq_ignore_ascii_case("USB4") {
                if let Some(prev) = cur.take() {
                    raw_nodes.push(prev);
                }
                cur = Some(RawWinUsb4Node {
                    router_id: rest.to_string(),
                    leaf: parts[2].to_ascii_lowercase(),
                    ..Default::default()
                });
            } else if parts.len() > 3 {
                if let Some(prev) = cur.take() {
                    raw_nodes.push(prev);
                }
            }
        } else if let Some(ref mut node) = cur {
            if let Some((k, rest_val)) = trimmed.split_once("REG_SZ") {
                let key_name = k.trim();
                let val = rest_val.trim().to_string();
                match key_name {
                    "ParentIdPrefix" => node.parent_prefix = Some(val.to_ascii_lowercase()),
                    "FriendlyName" => node.friendly = Some(val),
                    "DeviceDesc" => node.desc = Some(val),
                    "ContainerID" => node.container = Some(val),
                    _ => {}
                }
            }
        }
    }
    if let Some(prev) = cur.take() {
        raw_nodes.push(prev);
    }

    let prefix_to_router: BTreeMap<String, String> = raw_nodes
        .iter()
        .filter_map(|n| n.parent_prefix.as_ref().map(|p| (p.clone(), n.router_id.clone())))
        .collect();

    let mut nodes = Vec::new();
    for n in raw_nodes {
        let parent_router_id = prefix_to_router
            .iter()
            .find(|(p, _)| n.leaf.starts_with(p.as_str()) && **p != n.leaf)
            .map(|(_, parent_id)| parent_id.clone());
        let raw_label = n.friendly.or(n.desc).unwrap_or_else(|| n.router_id.clone());
        let clean_label = raw_label.rsplit(';').next().unwrap_or(&raw_label).trim().to_string();
        let vendor_name = if clean_label.starts_with("((") {
            clean_label.split(',').nth(1).map(|s| s.trim().to_string())
        } else {
            None
        };
        nodes.push(Usb4RouterNode {
            router_id: n.router_id,
            parent_router_id,
            vendor_name,
            device_name: clean_label,
            unique_id: n.container,
            generation: None,
            rx_lanes: None,
            tx_lanes: None,
            rx_speed_gbps: None,
            tx_speed_gbps: None,
            status: "registered".to_string(),
        });
    }
    nodes
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
pub fn collect_local_usb4_router_tree() -> Vec<Usb4RouterNode> {
    Vec::new()
}

pub fn prefix_len_to_netmask(prefix_len: u8) -> Ipv4Addr {
    let mask = if prefix_len == 0 {
        0
    } else if prefix_len >= 32 {
        u32::MAX
    } else {
        u32::MAX << (32 - u32::from(prefix_len))
    };
    Ipv4Addr::from(mask)
}

pub fn windows_route_print_has_exact_route(
    stdout: &str,
    net_ip: Ipv4Addr,
    prefix_len: u8,
    expected_gateway: Ipv4Addr,
) -> bool {
    let expected_mask = prefix_len_to_netmask(prefix_len);
    let mut in_ipv4_table = false;
    let mut in_active_routes = false;

    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("IPv4 Route Table") {
            in_ipv4_table = true;
            continue;
        }
        if trimmed.starts_with("IPv6 Route Table") {
            break;
        }
        if in_ipv4_table && trimmed.starts_with("Active Routes:") {
            in_active_routes = true;
            continue;
        }
        if in_active_routes && trimmed.starts_with("Persistent Routes:") {
            in_active_routes = false;
            continue;
        }
        if in_active_routes || !in_ipv4_table {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 3 {
                let Ok(dst) = parts[0].parse::<Ipv4Addr>() else {
                    continue;
                };
                let Ok(mask) = parts[1].parse::<Ipv4Addr>() else {
                    continue;
                };
                let Ok(gw) = parts[2].parse::<Ipv4Addr>() else {
                    continue;
                };
                if dst == net_ip && mask == expected_mask && gw == expected_gateway {
                    return true;
                }
            }
        }
    }
    false
}

pub fn windows_route_print_has_conflicting_destination(
    stdout: &str,
    net_ip: Ipv4Addr,
    prefix_len: u8,
    expected_gateway: Ipv4Addr,
) -> bool {
    let expected_mask = prefix_len_to_netmask(prefix_len);
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let Ok(dst) = parts[0].parse::<Ipv4Addr>() else {
                continue;
            };
            let Ok(mask) = parts[1].parse::<Ipv4Addr>() else {
                continue;
            };
            if dst == net_ip {
                let gw_opt = parts[2].parse::<Ipv4Addr>().ok();
                if mask != expected_mask || gw_opt != Some(expected_gateway) {
                    return true;
                }
            }
        }
    }
    false
}

pub fn linux_ip_route_json_has_exact_route(
    route_json: &serde_json::Value,
    subnet_cidr: &str,
    net_ip: Ipv4Addr,
    prefix_len: u8,
    expected_gateway: Ipv4Addr,
) -> bool {
    let Some(arr) = route_json.as_array() else {
        return false;
    };
    let gw_str = expected_gateway.to_string();
    let net_str = net_ip.to_string();
    arr.iter().any(|entry| {
        let dst = entry.get("dst").and_then(|v| v.as_str()).unwrap_or("");
        let gw = entry.get("gateway").and_then(|v| v.as_str()).unwrap_or("");
        let dst_matches = dst == subnet_cidr || (prefix_len == 32 && dst == net_str);
        dst_matches && gw == gw_str
    })
}

fn check_ipv4_route_present(
    subnet_cidr: &str,
    net_ip: Ipv4Addr,
    prefix_len: u8,
    expected_gateway: &str,
) -> Result<bool> {
    let gw_ip: Ipv4Addr = expected_gateway
        .trim()
        .parse()
        .with_context(|| format!("Invalid gateway IPv4 '{expected_gateway}' for subnet '{subnet_cidr}'"))?;

    #[cfg(target_os = "windows")]
    {
        let _ = subnet_cidr;
        let out = Command::new("route")
            .args(["print", "-4"])
            .output()
            .context("Failed to execute 'route print -4'")?;
        if !out.status.success() {
            anyhow::bail!(
                "'route print -4' exited with status {:?}: {}",
                out.status.code(),
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        Ok(windows_route_print_has_exact_route(&stdout, net_ip, prefix_len, gw_ip))
    }
    #[cfg(target_os = "linux")]
    {
        let route_json = run_cmd_json("ip", &["-4", "-j", "route", "show", subnet_cidr])?;
        Ok(linux_ip_route_json_has_exact_route(
            &route_json,
            subnet_cidr,
            net_ip,
            prefix_len,
            gw_ip,
        ))
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        let _ = (subnet_cidr, net_ip, prefix_len, gw_ip);
        Ok(false)
    }
}

pub fn windows_route_output_succeeded(status_ok: bool, stdout: &str, stderr: &str) -> bool {
    if !status_ok {
        return false;
    }
    let combined = format!("{} {}", stdout.trim(), stderr.trim()).to_ascii_lowercase();
    !combined.contains("failed")
        && !combined.contains("denied")
        && !combined.contains("requires elevation")
        && !combined.contains("not found")
        && !combined.contains("incorrect")
        && !combined.contains("bad argument")
        && (combined.contains("ok!") || combined.trim().is_empty())
}

pub fn windows_netsh_output_has_exact_ipv6(stdout: &str, expected_ip: Ipv6Addr) -> bool {
    for line in stdout.lines() {
        for raw_tok in line.split_whitespace() {
            let trimmed = raw_tok.trim_matches(|c: char| c == '"' || c == '\'' || c == ',' || c == ';');
            let without_prefix = trimmed.split('/').next().unwrap_or(trimmed);
            let without_scope = without_prefix.split('%').next().unwrap_or(without_prefix);
            if let Ok(parsed) = without_scope.parse::<Ipv6Addr>() {
                if parsed == expected_ip {
                    return true;
                }
            }
        }
    }
    false
}

fn apply_ipv4_subnet_route(subnet_cidr: &str, net_ip: Ipv4Addr, prefix_len: u8, gateway_ipv4: &str) -> Result<()> {
    if std::env::var("PC_AI_ALLOW_ROUTE_MUTATION").as_deref() != Ok("1") {
        anyhow::bail!(
            "Route mutation for '{subnet_cidr}' via '{gateway_ipv4}' is blocked: Slice C route apply requires explicit PC_AI_ALLOW_ROUTE_MUTATION=1 and preimage clearance"
        );
    }
    let gw_ip: Ipv4Addr = gateway_ipv4
        .trim()
        .parse()
        .with_context(|| format!("Invalid gateway IPv4 '{gateway_ipv4}'"))?;

    #[cfg(target_os = "windows")]
    {
        let pre_out = Command::new("route")
            .args(["print", "-4"])
            .output()
            .context("Failed to execute preimage 'route print -4'")?;
        if !pre_out.status.success() {
            anyhow::bail!("Preimage 'route print -4' failed with {:?}", pre_out.status.code());
        }
        let pre_stdout = String::from_utf8_lossy(&pre_out.stdout);
        if windows_route_print_has_conflicting_destination(&pre_stdout, net_ip, prefix_len, gw_ip) {
            anyhow::bail!(
                "Refusing to overwrite existing foreign route for destination {} (conflicts with expected /{prefix_len} via {gw_ip})",
                net_ip
            );
        }

        let mask_str = prefix_len_to_netmask(prefix_len).to_string();
        let net_str = net_ip.to_string();
        let add_out = Command::new("route")
            .args(["-p", "add", &net_str, "mask", &mask_str, gateway_ipv4, "metric", "15"])
            .output()
            .context("Failed to execute 'route -p add'")?;
        let add_stdout = String::from_utf8_lossy(&add_out.stdout);
        let add_stderr = String::from_utf8_lossy(&add_out.stderr);
        if !windows_route_output_succeeded(add_out.status.success(), &add_stdout, &add_stderr) {
            anyhow::bail!(
                "route -p add {} mask {} {} failed: {} {}",
                net_str,
                mask_str,
                gateway_ipv4,
                add_stderr.trim(),
                add_stdout.trim()
            );
        }
        Ok(())
    }
    #[cfg(target_os = "linux")]
    {
        let pre_json = run_cmd_json("ip", &["-4", "-j", "route", "show", subnet_cidr])?;
        if let Some(arr) = pre_json.as_array() {
            if !arr.is_empty()
                && !linux_ip_route_json_has_exact_route(&pre_json, subnet_cidr, net_ip, prefix_len, gw_ip)
            {
                anyhow::bail!(
                    "Refusing to overwrite existing foreign Linux route for '{}': {:?}",
                    subnet_cidr,
                    arr
                );
            }
        }
        let out = Command::new("ip")
            .args(["route", "add", subnet_cidr, "via", gateway_ipv4, "metric", "150"])
            .output()
            .context("Failed to execute 'ip route add'")?;
        if !out.status.success() {
            anyhow::bail!(
                "ip route add {} via {} failed: {}",
                subnet_cidr,
                gateway_ipv4,
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Ok(())
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        let _ = (subnet_cidr, net_ip, prefix_len, gw_ip);
        anyhow::bail!("Unsupported OS for route provisioning")
    }
}

fn inspect_and_apply_shared_subnet_routes(
    local_canon: &str,
    shared_subnets: &[SharedSwitchSubnet],
    apply_routes: bool,
) -> Vec<SharedSubnetRouteStatus> {
    let mut out = Vec::new();
    for sub in shared_subnets {
        let is_local_gateway = sub.gateway_host.eq_ignore_ascii_case(local_canon);
        let Ok((net_ip, prefix_len)) = parse_ipv4_cidr(&sub.subnet) else {
            out.push(SharedSubnetRouteStatus {
                subnet: sub.subnet.clone(),
                gateway_host: sub.gateway_host.clone(),
                gateway_ipv4: sub.gateway_ipv4.clone(),
                role: sub.role.clone(),
                is_local_gateway,
                route_present: false,
                route_applied: false,
                error: Some(format!("Invalid CIDR '{}'", sub.subnet)),
            });
            continue;
        };
        let mut err_msg = None;
        let mut route_present = match check_ipv4_route_present(&sub.subnet, net_ip, prefix_len, &sub.gateway_ipv4) {
            Ok(present) => present,
            Err(e) => {
                err_msg = Some(e.to_string());
                false
            }
        };
        let mut route_applied = false;

        if !is_local_gateway && !route_present && err_msg.is_none() && apply_routes {
            match apply_ipv4_subnet_route(&sub.subnet, net_ip, prefix_len, &sub.gateway_ipv4) {
                Ok(()) => {
                    route_applied = true;
                    match check_ipv4_route_present(&sub.subnet, net_ip, prefix_len, &sub.gateway_ipv4) {
                        Ok(true) => {
                            route_present = true;
                        }
                        Ok(false) => {
                            route_present = false;
                            err_msg = Some(format!(
                                "Route command succeeded for '{}' via '{}', but exact route was not found in post-check",
                                sub.subnet, sub.gateway_ipv4
                            ));
                        }
                        Err(e) => {
                            route_present = false;
                            err_msg = Some(format!("Post-check route readback failed: {e}"));
                        }
                    }
                }
                Err(e) => {
                    err_msg = Some(e.to_string());
                }
            }
        }

        out.push(SharedSubnetRouteStatus {
            subnet: sub.subnet.clone(),
            gateway_host: sub.gateway_host.clone(),
            gateway_ipv4: sub.gateway_ipv4.clone(),
            role: sub.role.clone(),
            is_local_gateway,
            route_present,
            route_applied,
            error: err_msg,
        });
    }
    out
}

pub fn validate_safe_custody_directory_and_files(dir: &Path) -> Result<(PathBuf, PathBuf)> {
    let dir_meta =
        fs::symlink_metadata(dir).with_context(|| format!("Custody state directory missing: '{}'", dir.display()))?;
    if dir_meta.file_type().is_symlink() || !dir_meta.file_type().is_dir() {
        anyhow::bail!(
            "Unsafe custody state directory '{}': must be a regular directory, not a symlink",
            dir.display()
        );
    }

    let lock_path = dir.join("lock");
    let lock_meta = fs::symlink_metadata(&lock_path)
        .with_context(|| format!("Custody lock file missing: '{}'", lock_path.display()))?;
    if lock_meta.file_type().is_symlink() || !lock_meta.file_type().is_file() {
        anyhow::bail!(
            "Unsafe custody lock '{}': must be a regular file, not a symlink",
            lock_path.display()
        );
    }

    let journal_path = dir.join("current.json");
    let journal_meta = fs::symlink_metadata(&journal_path)
        .with_context(|| format!("Custody journal missing: '{}'", journal_path.display()))?;
    if journal_meta.file_type().is_symlink() || !journal_meta.file_type().is_file() {
        anyhow::bail!(
            "Unsafe custody journal '{}': must be a regular file, not a symlink",
            journal_path.display()
        );
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if dir_meta.uid() != 0 || (dir_meta.mode() & 0o022) != 0 {
            anyhow::bail!(
                "Unsafe custody directory '{}' (uid={}, mode={:o}): must be root-owned and not group/world writable",
                dir.display(),
                dir_meta.uid(),
                dir_meta.mode() & 0o777
            );
        }
        if lock_meta.uid() != 0 || (lock_meta.mode() & 0o022) != 0 {
            anyhow::bail!(
                "Unsafe custody lock '{}' (uid={}, mode={:o}): must be root-owned and not group/world writable",
                lock_path.display(),
                lock_meta.uid(),
                lock_meta.mode() & 0o777
            );
        }
        if journal_meta.uid() != 0 || (journal_meta.mode() & 0o077) != 0 {
            anyhow::bail!(
                "Unsafe custody journal '{}' (uid={}, mode={:o}): must be root-owned with 0600 permissions",
                journal_path.display(),
                journal_meta.uid(),
                journal_meta.mode() & 0o777
            );
        }
    }

    Ok((lock_path, journal_path))
}

#[cfg(target_os = "linux")]
fn acquire_shared_custody_locks(dirs: &[&Path]) -> Result<Vec<fs::File>> {
    use std::os::unix::io::AsRawFd;
    use std::time::{Duration, Instant};

    let mut guards = Vec::new();
    for dir in dirs {
        let (lock_path, _) = validate_safe_custody_directory_and_files(dir)?;
        let file =
            fs::File::open(&lock_path).with_context(|| format!("Failed to open lock '{}'", lock_path.display()))?;
        let deadline = Instant::now() + Duration::from_millis(2000);
        loop {
            let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) };
            if rc == 0 {
                break;
            }
            let err = std::io::Error::last_os_error();
            if Instant::now() >= deadline {
                anyhow::bail!(
                    "Timed out acquiring shared flock on '{}' within 2000ms: {}",
                    lock_path.display(),
                    err
                );
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        guards.push(file);
    }
    Ok(guards)
}

#[cfg(target_os = "linux")]
fn read_locked_custody_journal(lock_dirs: &[&Path], target_dir: &Path) -> Result<(serde_json::Value, Vec<fs::File>)> {
    let guards = acquire_shared_custody_locks(lock_dirs)?;
    let (_, journal_path) = validate_safe_custody_directory_and_files(target_dir)?;
    let raw = fs::read_to_string(&journal_path)
        .with_context(|| format!("Failed to read custody journal '{}'", journal_path.display()))?;
    let parsed: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("Invalid JSON in custody journal '{}'", journal_path.display()))?;
    Ok((parsed, guards))
}

#[cfg(target_os = "linux")]
fn linux_namespace_inode(namespace: &str) -> Result<u64> {
    use std::os::unix::fs::MetadataExt;
    let path = PathBuf::from("/var/run/netns").join(namespace);
    let meta = fs::metadata(&path).with_context(|| format!("Failed to stat netns '{}'", path.display()))?;
    Ok(meta.ino())
}

#[cfg(target_os = "linux")]
fn inspect_linux_switch_custody(topo: &SfpSwitchTopology) -> LinuxSwitchCustodyStatus {
    let state_dir = PathBuf::from(&topo.state_dir);
    let state_path = state_dir.join("current.json");
    let mut pre_errors = Vec::new();

    let live_boot_id = match fs::read_to_string("/proc/sys/kernel/random/boot_id") {
        Ok(s) => Some(s.trim().to_string()),
        Err(e) => {
            pre_errors.push(format!("Failed to read /proc/sys/kernel/random/boot_id: {e}"));
            None
        }
    };
    let live_ns_inode = match linux_namespace_inode(&topo.namespace) {
        Ok(ino) => Some(ino),
        Err(e) => {
            pre_errors.push(e.to_string());
            None
        }
    };

    let (state_val, _lock_guards) = match read_locked_custody_journal(&[state_dir.as_path()], state_dir.as_path()) {
        Ok((val, guards)) => (Some(val), Some(guards)),
        Err(e) => {
            pre_errors.push(e.to_string());
            (None, None)
        }
    };

    let ns_links = match run_cmd_json("ip", &["-n", &topo.namespace, "-d", "-j", "link"]) {
        Ok(v) => Some(v),
        Err(e) => {
            pre_errors.push(e.to_string());
            None
        }
    };
    let ns_addrs = match run_cmd_json("ip", &["-n", &topo.namespace, "-j", "addr"]) {
        Ok(v) => Some(v),
        Err(e) => {
            pre_errors.push(e.to_string());
            None
        }
    };

    let mut status = evaluate_linux_switch_custody_from_json(
        topo,
        state_path.to_string_lossy().into_owned(),
        state_val.as_ref(),
        ns_links.as_ref(),
        ns_addrs.as_ref(),
        live_boot_id,
        live_ns_inode,
    );
    if !pre_errors.is_empty() {
        status.all_ports_valid = false;
        status.errors.splice(0..0, pre_errors);
    }
    status
}

#[cfg(target_os = "linux")]
fn inspect_linux_usb4_extension_custody(topo: &SfpSwitchTopology) -> Option<LinuxUsb4ExtensionCustodyStatus> {
    if topo.usb4_extensions.is_empty() {
        return None;
    }
    let parent_dir = PathBuf::from(&topo.state_dir);
    let ext_dir = PathBuf::from(
        topo.extension_state_dir
            .as_deref()
            .unwrap_or("/var/lib/ms01-usb4-extension"),
    );
    let state_path = ext_dir.join("current.json");
    let mut pre_errors = Vec::new();

    let (state_val, _lock_guards) =
        match read_locked_custody_journal(&[parent_dir.as_path(), ext_dir.as_path()], ext_dir.as_path()) {
            Ok((val, guards)) => (Some(val), Some(guards)),
            Err(e) => {
                pre_errors.push(e.to_string());
                (None, None)
            }
        };

    let live_boot_id = match fs::read_to_string("/proc/sys/kernel/random/boot_id") {
        Ok(s) => Some(s.trim().to_string()),
        Err(e) => {
            pre_errors.push(format!("Failed to read /proc/sys/kernel/random/boot_id: {e}"));
            None
        }
    };
    let live_ns_inode = match linux_namespace_inode(&topo.namespace) {
        Ok(ino) => Some(ino),
        Err(e) => {
            pre_errors.push(e.to_string());
            None
        }
    };
    let root_links = match run_cmd_json("ip", &["-d", "-j", "link"]) {
        Ok(v) => Some(v),
        Err(e) => {
            pre_errors.push(e.to_string());
            None
        }
    };
    let root_addrs = match run_cmd_json("ip", &["-j", "addr"]) {
        Ok(v) => Some(v),
        Err(e) => {
            pre_errors.push(e.to_string());
            None
        }
    };
    let ns_links = match run_cmd_json("ip", &["-n", &topo.namespace, "-d", "-j", "link"]) {
        Ok(v) => Some(v),
        Err(e) => {
            pre_errors.push(e.to_string());
            None
        }
    };
    let ns_addrs = match run_cmd_json("ip", &["-n", &topo.namespace, "-j", "addr"]) {
        Ok(v) => Some(v),
        Err(e) => {
            pre_errors.push(e.to_string());
            None
        }
    };

    let mut status = evaluate_linux_usb4_extension_custody_from_json(
        topo,
        state_path.to_string_lossy().into_owned(),
        state_val.as_ref(),
        root_links.as_ref(),
        root_addrs.as_ref(),
        ns_links.as_ref(),
        ns_addrs.as_ref(),
        live_boot_id.as_deref(),
        live_ns_inode,
    )?;
    if !pre_errors.is_empty() {
        status.custody_valid = false;
        status.errors.splice(0..0, pre_errors);
    }
    Some(status)
}

#[cfg(target_os = "linux")]
fn run_ms01_usb4_helper_check() -> Option<VigilMs01Usb4CheckReceipt> {
    let helper = Path::new("/usr/local/sbin/ms01-usb4-extension");
    if !helper.exists() {
        return None;
    }
    let out = Command::new(helper).arg("check").output().ok()?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    parse_and_validate_ms01_usb4_check_receipt(stdout.trim()).ok()
}

#[cfg(not(target_os = "linux"))]
fn run_ms01_usb4_helper_check() -> Option<VigilMs01Usb4CheckReceipt> {
    None
}

#[cfg(target_os = "linux")]
fn inspect_linux_canary_endpoints(local_canon: &str, ipam: &IpamAllocations) -> Vec<CanaryEndpointLiveStatus> {
    let root_addrs = run_cmd_json("ip", &["-6", "-j", "addr"]).ok();
    let root_rules = run_cmd_json("ip", &["-6", "-j", "rule"]).ok();

    let mut out = Vec::new();
    for ep in &ipam.bridged_lan_canary_endpoints {
        if !ep.host.eq_ignore_ascii_case(local_canon) {
            continue;
        }
        let expected_ip_opt = ep
            .ula_address
            .split('/')
            .next()
            .unwrap_or(&ep.ula_address)
            .trim()
            .parse::<Ipv6Addr>()
            .ok();
        let assigned = match (&root_addrs, expected_ip_opt) {
            (Some(v), Some(expected_ip)) => v
                .as_array()
                .and_then(|arr| {
                    arr.iter()
                        .find(|item| item.get("ifname").and_then(|v| v.as_str()) == Some(ep.interface.as_str()))
                })
                .and_then(|item| item.get("addr_info"))
                .and_then(|v| v.as_array())
                .map(|infos| {
                    infos.iter().any(|info| {
                        info.get("local")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<Ipv6Addr>().ok())
                            == Some(expected_ip)
                    })
                })
                .unwrap_or(false),
            _ => false,
        };

        let policy_rule_present = ep.policy_table.map(|tbl| {
            let tbl_str = tbl.to_string();
            root_rules
                .as_ref()
                .and_then(|v| v.as_array())
                .map(|rules| {
                    rules.iter().any(|r| {
                        let table_match = r
                            .get("table")
                            .and_then(|v| v.as_str())
                            .map(|s| s == tbl_str)
                            .unwrap_or(false);
                        let prio_match = match ep.policy_priority {
                            Some(p) => r.get("priority").and_then(|v| v.as_u64()) == Some(u64::from(p)),
                            None => true,
                        };
                        table_match && prio_match
                    })
                })
                .unwrap_or(false)
        });

        out.push(CanaryEndpointLiveStatus {
            host: ep.host.clone(),
            interface: ep.interface.clone(),
            expected_ula: ep.ula_address.clone(),
            role: ep.role.clone(),
            assigned_on_interface: assigned,
            policy_table: ep.policy_table,
            policy_priority: ep.policy_priority,
            policy_rule_present,
        });
    }
    out
}

#[cfg(target_os = "windows")]
fn inspect_windows_canary_endpoints(local_canon: &str, ipam: &IpamAllocations) -> Vec<CanaryEndpointLiveStatus> {
    let mut out = Vec::new();
    for ep in &ipam.bridged_lan_canary_endpoints {
        if !ep.host.eq_ignore_ascii_case(local_canon) {
            continue;
        }
        let expected_ip_opt = ep
            .ula_address
            .split('/')
            .next()
            .unwrap_or(&ep.ula_address)
            .trim()
            .parse::<Ipv6Addr>()
            .ok();
        let assigned = match expected_ip_opt {
            Some(expected_ip) => Command::new("netsh")
                .args([
                    "interface",
                    "ipv6",
                    "show",
                    "addresses",
                    &format!("interface={}", ep.interface),
                ])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| {
                    let s = String::from_utf8_lossy(&o.stdout);
                    windows_netsh_output_has_exact_ipv6(&s, expected_ip)
                })
                .unwrap_or(false),
            None => false,
        };

        out.push(CanaryEndpointLiveStatus {
            host: ep.host.clone(),
            interface: ep.interface.clone(),
            expected_ula: ep.ula_address.clone(),
            role: ep.role.clone(),
            assigned_on_interface: assigned,
            policy_table: ep.policy_table,
            policy_priority: ep.policy_priority,
            policy_rule_present: None,
        });
    }
    out
}

#[cfg(target_os = "windows")]
fn collect_windows_metric_hierarchy() -> Result<WindowsMetricHierarchyReport> {
    let out = Command::new("netsh")
        .args(["interface", "ipv4", "show", "interfaces"])
        .output()
        .context("Failed to execute 'netsh interface ipv4 show interfaces'")?;
    if !out.status.success() {
        anyhow::bail!(
            "'netsh interface ipv4 show interfaces' failed with exit {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let stdout = String::from_utf8_lossy(&out.stdout);

    let addr_out = Command::new("netsh")
        .args(["interface", "ipv4", "show", "addresses"])
        .output()
        .context("Failed to execute 'netsh interface ipv4 show addresses'")?;
    if !addr_out.status.success() {
        anyhow::bail!(
            "'netsh interface ipv4 show addresses' failed with exit {:?}: {}",
            addr_out.status.code(),
            String::from_utf8_lossy(&addr_out.stderr).trim()
        );
    }
    let addr_stdout = String::from_utf8_lossy(&addr_out.stdout).into_owned();

    let mut subnet_map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut current_iface: Option<String> = None;
    for line in addr_stdout.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("Configuration for interface \"") {
            if let Some(name) = rest.strip_suffix('"') {
                current_iface = Some(name.to_string());
            }
        } else if let Some(ref iface) = current_iface {
            if let Some(idx) = trimmed.find("Subnet Prefix:") {
                let after = trimmed[idx + "Subnet Prefix:".len()..].trim();
                if let Some(cidr) = after.split_whitespace().next() {
                    if !cidr.starts_with("127.") && !cidr.starts_with("169.254.") {
                        subnet_map.entry(iface.clone()).or_default().push(cidr.to_string());
                    }
                }
            }
        }
    }

    let snap = crate::platform::collect_platform_snapshot(true).ok();
    let usb4_ifindices: Vec<u32> = snap
        .as_ref()
        .map(|s| s.adapters.iter().map(|a| a.interface_index).collect())
        .unwrap_or_default();

    let mut interfaces = Vec::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            continue;
        }
        let Ok(ifindex) = parts[0].parse::<u32>() else {
            continue;
        };
        let metric = parts[1].parse::<u32>().ok();
        let mtu = parts[2].parse::<u32>().ok();
        let state = parts[3].to_string();
        let alias = parts[4..].join(" ");
        let lower_alias = alias.to_ascii_lowercase();
        if lower_alias.contains("loopback")
            || lower_alias.contains("vethernet")
            || lower_alias.contains("bluetooth")
            || lower_alias.contains("local area connection*")
        {
            continue;
        }
        let media_tier = if usb4_ifindices.contains(&ifindex) {
            "usb4-p2p".to_string()
        } else if lower_alias.contains("wi-fi") || lower_alias.contains("wlan") || lower_alias.contains("wireless") {
            "wifi".to_string()
        } else if lower_alias.contains("tailscale") || lower_alias.contains("wg") {
            "overlay".to_string()
        } else {
            "wired-lan".to_string()
        };

        let ipv4_subnets = subnet_map.remove(&alias).unwrap_or_default();
        interfaces.push(WindowsInterfaceMetricEntry {
            interface_index: ifindex,
            interface_alias: alias,
            media_tier,
            metric,
            mtu_bytes: mtu,
            state,
            ipv4_subnets,
        });
    }

    let (hierarchy_valid, same_subnet_conflicts) = evaluate_windows_metric_hierarchy(&interfaces);
    Ok(WindowsMetricHierarchyReport {
        interfaces,
        hierarchy_valid,
        same_subnet_conflicts,
    })
}

pub fn collect_switch_status(explicit_registry: Option<&Path>) -> Result<SwitchStatusReport> {
    collect_switch_status_with_options(explicit_registry, false)
}

pub fn collect_switch_status_with_options(
    explicit_registry: Option<&Path>,
    apply_routes: bool,
) -> Result<SwitchStatusReport> {
    let (reg_opt, loaded_path) = load_peer_registry(explicit_registry)?;
    let raw_host = local_hostname();
    let known_peers = reg_opt.as_ref().map(|r| r.fleet_peers.as_slice()).unwrap_or(&[]);
    let local_canon = canonicalize_peer_name(&raw_host, known_peers);

    let matched_topology = reg_opt.as_ref().and_then(|r| {
        r.sfp_switch_topologies
            .iter()
            .find(|t| t.host.eq_ignore_ascii_case(&local_canon))
            .cloned()
    });
    let ipam_allocations = reg_opt.as_ref().and_then(|r| r.ipam_allocations.clone());
    let topology_links = reg_opt.as_ref().map(|r| r.topology_links.clone()).unwrap_or_default();
    let local_usb4_routers = collect_local_usb4_router_tree();
    let shared_subnet_routes = ipam_allocations
        .as_ref()
        .map(|ipam| inspect_and_apply_shared_subnet_routes(&local_canon, &ipam.shared_switch_subnets, apply_routes))
        .unwrap_or_default();

    #[allow(unused_mut)]
    let mut warnings = Vec::new();
    let mut errors = Vec::new();

    for r in &shared_subnet_routes {
        if let Some(ref e) = r.error {
            errors.push(format!("Shared switch subnet '{}': {e}", r.subnet));
        } else if !r.is_local_gateway && !r.route_present {
            warnings.push(format!(
                "Shared switch subnet '{}' via '{}' ({}) is not currently routed locally",
                r.subnet, r.gateway_ipv4, r.gateway_host
            ));
        }
    }

    #[cfg(target_os = "linux")]
    let (linux_switch_custody, linux_usb4_extension_custody, linux_canary_ula_status) = {
        let sw = matched_topology.as_ref().map(inspect_linux_switch_custody);
        let ext = matched_topology.as_ref().and_then(inspect_linux_usb4_extension_custody);
        let canaries = ipam_allocations
            .as_ref()
            .map(|ipam| inspect_linux_canary_endpoints(&local_canon, ipam))
            .unwrap_or_default();
        (sw, ext, canaries)
    };

    #[cfg(target_os = "windows")]
    let (linux_switch_custody, linux_usb4_extension_custody, linux_canary_ula_status): (
        Option<LinuxSwitchCustodyStatus>,
        Option<LinuxUsb4ExtensionCustodyStatus>,
        Vec<CanaryEndpointLiveStatus>,
    ) = {
        let canaries = ipam_allocations
            .as_ref()
            .map(|ipam| inspect_windows_canary_endpoints(&local_canon, ipam))
            .unwrap_or_default();
        (None, None, canaries)
    };

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    let (linux_switch_custody, linux_usb4_extension_custody, linux_canary_ula_status): (
        Option<LinuxSwitchCustodyStatus>,
        Option<LinuxUsb4ExtensionCustodyStatus>,
        Vec<CanaryEndpointLiveStatus>,
    ) = (None, None, Vec::new());

    #[cfg(target_os = "windows")]
    let windows_metric_hierarchy = Some(collect_windows_metric_hierarchy()?);

    #[cfg(not(target_os = "windows"))]
    let windows_metric_hierarchy: Option<WindowsMetricHierarchyReport> = None;

    if let Some(ref sw) = linux_switch_custody {
        if !sw.state_file_present {
            errors.push(format!("Switch custody state file missing: {}", sw.state_file_path));
        }
        for e in &sw.errors {
            errors.push(e.clone());
        }
        for p in &sw.ports {
            if let Some(ref e) = p.error {
                errors.push(format!("SFP+ port '{}': {e}", p.name));
            }
        }
    }

    if let Some(ref ext) = linux_usb4_extension_custody {
        for e in &ext.errors {
            errors.push(format!("USB4 extension '{}': {e}", ext.usb4_ifname));
        }
    }

    for c in &linux_canary_ula_status {
        if !c.assigned_on_interface {
            warnings.push(format!(
                "Canary ULA '{}' is not currently assigned on {}:{}",
                c.expected_ula, c.host, c.interface
            ));
        }
        if c.policy_rule_present == Some(false) {
            warnings.push(format!(
                "Canary ULA policy rule (table {:?}, priority {:?}) not active on {}:{}",
                c.policy_table, c.policy_priority, c.host, c.interface
            ));
        }
    }

    if let Some(ref win_h) = windows_metric_hierarchy {
        if !win_h.hierarchy_valid {
            for conflict in &win_h.same_subnet_conflicts {
                errors.push(conflict.clone());
            }
        }
    }

    let custody_valid = errors.is_empty();
    Ok(SwitchStatusReport {
        local_host: local_canon,
        os_family: std::env::consts::OS.to_string(),
        registry_path: loaded_path,
        matched_topology,
        ipam_allocations,
        topology_links,
        local_usb4_routers,
        shared_subnet_routes,
        linux_switch_custody,
        linux_usb4_extension_custody,
        linux_canary_ula_status,
        windows_metric_hierarchy,
        custody_valid,
        warnings,
        errors,
    })
}

fn gate_status_from_technical_verdict(technical_verdict: &str) -> String {
    match technical_verdict {
        // Provisional PENDING-FABRIC-* IDs never claim acceptance PASS credit before TPM owner intake
        "PASS" => "HOLD_PENDING_TPM_INTAKE".to_string(),
        other => other.to_string(),
    }
}

pub fn evaluate_vnv_gates_from_status(
    status: &SwitchStatusReport,
) -> (String, bool, Vec<VnvRequirementGateRow>) {
    let mut gates = Vec::new();

    // PENDING-FABRIC-F09: Static declared cycle check + local hardware router or switch topology observation
    let any_unauthorized_router = status
        .local_usb4_routers
        .iter()
        .any(|r| r.status.eq_ignore_ascii_case("unauthorized"));
    let f09_tech = if status.topology_links.is_empty()
        || detect_l2_bridge_cycles(&status.topology_links).is_err()
        || any_unauthorized_router
    {
        "FAIL"
    } else if status.local_usb4_routers.is_empty() && status.matched_topology.is_none() {
        "BLOCKED_MISSING_PROVIDER"
    } else {
        "PASS"
    };
    gates.push(VnvRequirementGateRow {
        trace_id: "PENDING-FABRIC-F09".to_string(),
        requirement_id: "REQ-SPARK-001".to_string(),
        tpm_admission_status: "PENDING_TPM_INTAKE".to_string(),
        acceptance_credit: false,
        description: "Declared multi-hub topology link validation & local USB4/switch router observation".to_string(),
        status: gate_status_from_technical_verdict(f09_tech),
        technical_verdict: f09_tech.to_string(),
        details: format!(
            "topology_links={}, local_usb4_routers={}, unauthorized_routers={}, matched_switch_topology={}",
            status.topology_links.len(),
            status.local_usb4_routers.len(),
            status
                .local_usb4_routers
                .iter()
                .filter(|r| r.status.eq_ignore_ascii_case("unauthorized"))
                .count(),
            status.matched_topology.is_some()
        ),
    });

    // PENDING-FABRIC-F10: Linux switch & USB4 namespace structural custody
    let f10_tech = match (&status.matched_topology, &status.linux_switch_custody) {
        (None, None) if status.os_family != "linux" => "SKIP_NOT_APPLICABLE",
        (None, None) => "SKIP_NOT_APPLICABLE",
        (Some(_), None) => "BLOCKED_MISSING_PROVIDER",
        (_, Some(sw)) => {
            let sw_ok = sw.state_file_present
                && sw.phase_running
                && sw.boot_id_matches
                && sw.namespace_inode_matches
                && sw.bridge_mac_matches
                && sw.bridge_alias_matches
                && sw.all_ports_valid
                && sw.errors.is_empty();
            let ext_expected = status
                .matched_topology
                .as_ref()
                .map(|t| !t.usb4_extensions.is_empty())
                .unwrap_or(false);
            let ext_ok = if ext_expected {
                status
                    .linux_usb4_extension_custody
                    .as_ref()
                    .map(|e| e.custody_valid)
                    .unwrap_or(false)
            } else {
                true
            };
            if sw_ok && ext_ok {
                "PASS"
            } else {
                "FAIL"
            }
        }
    };
    gates.push(VnvRequirementGateRow {
        trace_id: "PENDING-FABRIC-F10".to_string(),
        requirement_id: "REQ-SPARK-002".to_string(),
        tpm_admission_status: "PENDING_TPM_INTAKE".to_string(),
        acceptance_credit: false,
        description:
            "Linux switch & USB4 namespace structural custody verification (boot_id, inode, MAC, exact owner alias, veth symmetry)"
                .to_string(),
        status: gate_status_from_technical_verdict(f10_tech),
        technical_verdict: f10_tech.to_string(),
        details: if let Some(ref sw) = status.linux_switch_custody {
            format!(
                "namespace={}, bridge={}, phase_running={}, all_ports_valid={}, usb4_ext_valid={:?}",
                sw.namespace,
                sw.bridge,
                sw.phase_running,
                sw.all_ports_valid,
                status.linux_usb4_extension_custody.as_ref().map(|e| e.custody_valid)
            )
        } else {
            format!(
                "Not applicable or no local switch topology on host '{}' (os={})",
                status.local_host, status.os_family
            )
        },
    });

    // PENDING-FABRIC-F11: Dual-ULA IPAM separation & live canary endpoint assignment
    let f11_tech = match &status.ipam_allocations {
        None => "FAIL",
        Some(_) => {
            if status.linux_canary_ula_status.is_empty() {
                "SKIP_NOT_APPLICABLE"
            } else if status.linux_canary_ula_status.iter().all(|c| {
                c.assigned_on_interface && c.policy_rule_present.unwrap_or(true)
            }) {
                "PASS"
            } else {
                "FAIL"
            }
        }
    };
    gates.push(VnvRequirementGateRow {
        trace_id: "PENDING-FABRIC-F11".to_string(),
        requirement_id: "REQ-SPARK-003".to_string(),
        tpm_admission_status: "PENDING_TPM_INTAKE".to_string(),
        acceptance_credit: false,
        description:
            "Dual-ULA IPAM separation and live canary endpoint/policy rule verification"
                .to_string(),
        status: gate_status_from_technical_verdict(f11_tech),
        technical_verdict: f11_tech.to_string(),
        details: format!(
            "ipam_present={}, local_canary_endpoints={}, all_assigned={}",
            status.ipam_allocations.is_some(),
            status.linux_canary_ula_status.len(),
            status
                .linux_canary_ula_status
                .iter()
                .all(|c| c.assigned_on_interface && c.policy_rule_present.unwrap_or(true))
        ),
    });

    // PENDING-FABRIC-F12: Windows multi-homed interface metric hierarchy
    let f12_tech = if status.os_family != "windows" {
        "SKIP_NOT_APPLICABLE"
    } else {
        match &status.windows_metric_hierarchy {
            None => "BLOCKED_MISSING_PROVIDER",
            Some(win_h) if win_h.hierarchy_valid && !win_h.interfaces.is_empty() => "PASS",
            Some(_) => "FAIL",
        }
    };
    gates.push(VnvRequirementGateRow {
        trace_id: "PENDING-FABRIC-F12".to_string(),
        requirement_id: "REQ-COMPUTE-001".to_string(),
        tpm_admission_status: "PENDING_TPM_INTAKE".to_string(),
        acceptance_credit: false,
        description: "Windows multi-homed interface metric hierarchy (USB4 P2P < Wired/SFP+ < Wi-Fi)".to_string(),
        status: gate_status_from_technical_verdict(f12_tech),
        technical_verdict: f12_tech.to_string(),
        details: if let Some(ref win_h) = status.windows_metric_hierarchy {
            format!(
                "hierarchy_valid={}, interfaces={}, conflicts={}",
                win_h.hierarchy_valid,
                win_h.interfaces.len(),
                win_h.same_subnet_conflicts.len()
            )
        } else {
            format!("Not applicable on os_family={}", status.os_family)
        },
    });

    // PENDING-FABRIC-F13: Shared switch subnet effective route presence
    let f13_tech = if status.shared_subnet_routes.is_empty() {
        "FAIL"
    } else if status
        .shared_subnet_routes
        .iter()
        .all(|r| r.error.is_none() && (r.is_local_gateway || r.route_present))
    {
        "PASS"
    } else {
        "FAIL"
    };
    gates.push(VnvRequirementGateRow {
        trace_id: "PENDING-FABRIC-F13".to_string(),
        requirement_id: "REQ-MOB-001".to_string(),
        tpm_admission_status: "PENDING_TPM_INTAKE".to_string(),
        acceptance_credit: false,
        description: "Shared 10.* switch subnet effective route verification (netmask + gateway readback)".to_string(),
        status: gate_status_from_technical_verdict(f13_tech),
        technical_verdict: f13_tech.to_string(),
        details: format!(
            "shared_subnets_checked={}, routes_present_or_local_gw={}",
            status.shared_subnet_routes.len(),
            status
                .shared_subnet_routes
                .iter()
                .filter(|r| r.error.is_none() && (r.is_local_gateway || r.route_present))
                .count()
        ),
    });

    let any_failed = !status.custody_valid
        || gates
            .iter()
            .any(|g| g.status == "FAIL" || g.status == "BLOCKED_MISSING_PROVIDER");
    let overall_status = if any_failed {
        "FAIL".to_string()
    } else if gates.iter().any(|g| g.status == "HOLD_PENDING_TPM_INTAKE") {
        "HOLD_PENDING_TPM_INTAKE".to_string()
    } else {
        "SKIP_NOT_APPLICABLE".to_string()
    };

    (overall_status, status.custody_valid && !any_failed, gates)
}

pub fn run_vnv_verification(
    explicit_registry: Option<&Path>,
    output_path: Option<&Path>,
) -> Result<VigilHilDeviceEvidence> {
    let status = collect_switch_status(explicit_registry)?;
    let linux_usb4_observation = status
        .linux_usb4_extension_custody
        .as_ref()
        .map(|ext| PcaiLinuxUsb4Observation {
            schema: "pcai.linux-usb4-observation/v1".to_string(),
            status: if ext.custody_valid {
                "CUSTODY_VERIFIED".to_string()
            } else {
                "FAILED".to_string()
            },
            readiness_scope: "structural_custody_only".to_string(),
            network_ready: None,
            namespace: ext.namespace.clone(),
            bridge: ext.bridge.clone(),
            usb4_ifname: ext.usb4_ifname.clone(),
            root_veth: ext.root_veth.clone(),
            bridge_veth: ext.bridge_veth.clone(),
            errors: ext.errors.clone(),
        });

    let usb4_check_receipt = run_ms01_usb4_helper_check();
    let (overall_status, custody_valid, gates) = evaluate_vnv_gates_from_status(&status);

    let evidence = VigilHilDeviceEvidence {
        schema: "vigil.hil-device-evidence/v1".to_string(),
        host: status.local_host.clone(),
        os_family: status.os_family.clone(),
        tpm_admission_status: "PENDING_TPM_INTAKE".to_string(),
        acceptance_credit: false,
        overall_status,
        custody_valid,
        usb4_check_receipt,
        linux_usb4_observation,
        requirement_gates: gates,
        local_usb4_routers: status.local_usb4_routers.clone(),
        shared_subnet_routes: status.shared_subnet_routes.clone(),
        topology_links: status.topology_links.clone(),
        switch_status: status,
        output_path: output_path.map(|p| p.to_string_lossy().into_owned()),
    };

    if let Some(out_file) = output_path {
        if let Some(parent) = out_file.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("Failed to create parent directory '{}'", parent.display()))?;
            }
        }
        let payload = serde_json::to_string_pretty(&evidence)?;
        fs::write(out_file, format!("{payload}\n"))
            .with_context(|| format!("Failed to write V&V evidence to '{}'", out_file.display()))?;
    }

    Ok(evidence)
}

pub fn run_switch_observe(record: bool, explicit_registry: Option<&Path>) -> Result<SwitchObserveSnapshot> {
    let (reg_opt, _) = load_peer_registry(explicit_registry)?;
    let raw_host = local_hostname();
    let known_peers = reg_opt.as_ref().map(|r| r.fleet_peers.as_slice()).unwrap_or(&[]);
    let local_canon = canonicalize_peer_name(&raw_host, known_peers);
    let topo = reg_opt
        .as_ref()
        .and_then(|r| {
            r.sfp_switch_topologies
                .iter()
                .find(|t| t.host.eq_ignore_ascii_case(&local_canon))
                .cloned()
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Host '{}' does not have an sfpSwitchTopologies entry in thunderbolt-peers.json",
                local_canon
            )
        })?;

    #[cfg(target_os = "linux")]
    let (state_val, _shared_locks) = {
        let state_dir = PathBuf::from(&topo.state_dir);
        let ext_dir = PathBuf::from(
            topo.extension_state_dir
                .as_deref()
                .unwrap_or("/var/lib/ms01-usb4-extension"),
        );
        let mut lock_dirs: Vec<&Path> = vec![state_dir.as_path()];
        if !topo.usb4_extensions.is_empty() {
            lock_dirs.push(ext_dir.as_path());
        }
        read_locked_custody_journal(&lock_dirs, state_dir.as_path())?
    };

    #[cfg(not(target_os = "linux"))]
    let state_val: serde_json::Value = {
        let (_, journal_path) = validate_safe_custody_directory_and_files(Path::new(&topo.state_dir))?;
        serde_json::from_str(&fs::read_to_string(&journal_path)?)?
    };

    let status = collect_switch_status(explicit_registry)?;
    let sw = status.linux_switch_custody.ok_or_else(|| {
        anyhow::anyhow!(
            "Linux switch custody is not available on this OS ({})",
            status.os_family
        )
    })?;
    if !sw.state_file_present
        || !sw.phase_running
        || !sw.boot_id_matches
        || !sw.namespace_inode_matches
        || !sw.bridge_mac_matches
        || !sw.bridge_alias_matches
        || !sw.all_ports_valid
    {
        anyhow::bail!(
            "Refusing switch observe: switch custody verification failed: {}",
            status.errors.join("; ")
        );
    }

    let ports_map: BTreeMap<String, serde_json::Value> = state_val
        .get("ports")
        .or_else(|| state_val.get("original"))
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let links_val = run_cmd_json("ip", &["-n", &topo.namespace, "-s", "-d", "-j", "link"])?;
    let fdb_val = run_cmd_json(
        "bridge",
        &["-n", &topo.namespace, "-j", "fdb", "show", "br", &topo.bridge],
    )?;
    let br_ports_val = run_cmd_json("bridge", &["-n", &topo.namespace, "-s", "-j", "link", "show"])?;
    let vlans_val = run_cmd_json("bridge", &["-n", &topo.namespace, "-j", "vlan", "show"])?;

    let observe_dir = topo
        .observe_state_dir
        .as_deref()
        .unwrap_or("/var/lib/ms01-network-observe");
    let out_path = PathBuf::from(observe_dir).join("latest.json");

    let mut snap = SwitchObserveSnapshot {
        schema_version: 1,
        payload_capture: false,
        boot_id: sw.live_boot_id.unwrap_or_default(),
        namespace: topo.namespace.clone(),
        namespace_inode: sw.live_namespace_inode.unwrap_or(0),
        bridge: topo.bridge.clone(),
        bridge_mac: sw.live_bridge_mac.unwrap_or_default(),
        ports: ports_map,
        links: links_val.as_array().cloned().unwrap_or_default(),
        bridge_ports: br_ports_val.as_array().cloned().unwrap_or_default(),
        fdb: fdb_val.as_array().cloned().unwrap_or_default(),
        vlans: vlans_val.as_array().cloned().unwrap_or_default(),
        output_path: Some(out_path.to_string_lossy().into_owned()),
        recorded: false,
    };

    if record {
        let obs_path = Path::new(observe_dir);
        if let Ok(meta) = fs::symlink_metadata(obs_path) {
            if meta.file_type().is_symlink() {
                anyhow::bail!("Refusing to write observe snapshot to symlinked directory '{observe_dir}'");
            }
        }
        fs::create_dir_all(observe_dir)
            .with_context(|| format!("Failed to create observe directory '{observe_dir}'"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(observe_dir, fs::Permissions::from_mode(0o700))
                .with_context(|| format!("Failed to chmod 0700 '{observe_dir}'"))?;
        }
        let tmp_path = PathBuf::from(observe_dir).join(format!(".latest.json.tmp.{}", std::process::id()));
        snap.recorded = true;
        let payload = serde_json::to_string_pretty(&snap)?;
        fs::write(&tmp_path, format!("{payload}\n"))
            .with_context(|| format!("Failed to write temp observe file '{}'", tmp_path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&tmp_path, fs::Permissions::from_mode(0o600))
                .with_context(|| format!("Failed to chmod 0600 '{}'", tmp_path.display()))?;
        }
        fs::rename(&tmp_path, &out_path).with_context(|| {
            format!(
                "Failed to atomically rename observe snapshot to '{}'",
                out_path.display()
            )
        })?;
    }

    Ok(snap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{SfpPortSpec, Usb4BridgeExtensionSpec};

    #[test]
    fn test_validate_veth_pair_symmetry() {
        assert!(validate_veth_pair_symmetry(Some(10), Some(11), Some(11), Some(10)));
        assert!(!validate_veth_pair_symmetry(Some(10), Some(10), Some(10), Some(10)));
        assert!(!validate_veth_pair_symmetry(Some(10), Some(12), Some(11), Some(10)));
        assert!(!validate_veth_pair_symmetry(None, Some(11), Some(11), Some(10)));
    }

    #[test]
    fn test_evaluate_windows_metric_hierarchy_detects_wifi_wired_conflict_and_empty_or_missing_metric() {
        // Empty census must fail closed (never vacuous true)
        let (empty_valid, empty_conflicts) = evaluate_windows_metric_hierarchy(&[]);
        assert!(!empty_valid);
        assert!(!empty_conflicts.is_empty());

        // Missing metric on a connected interface must fail closed
        let missing_metric_entries = vec![WindowsInterfaceMetricEntry {
            interface_index: 11,
            interface_alias: "Ethernet 2".to_string(),
            media_tier: "wired-lan".to_string(),
            metric: None,
            mtu_bytes: Some(1500),
            state: "connected".to_string(),
            ipv4_subnets: vec!["192.168.50.0/24".to_string()],
        }];
        let (missing_valid, missing_conflicts) = evaluate_windows_metric_hierarchy(&missing_metric_entries);
        assert!(!missing_valid);
        assert!(missing_conflicts.iter().any(|c| c.contains("missing/unknown metric")));

        let bad_entries = vec![
            WindowsInterfaceMetricEntry {
                interface_index: 12,
                interface_alias: "Ethernet 4".to_string(),
                media_tier: "usb4-p2p".to_string(),
                metric: Some(5),
                mtu_bytes: Some(62000),
                state: "connected".to_string(),
                ipv4_subnets: vec!["172.31.241.248/30".to_string()],
            },
            WindowsInterfaceMetricEntry {
                interface_index: 15,
                interface_alias: "Ethernet 2".to_string(),
                media_tier: "wired-lan".to_string(),
                metric: Some(35),
                mtu_bytes: Some(1500),
                state: "connected".to_string(),
                ipv4_subnets: vec!["192.168.50.0/24".to_string()],
            },
            WindowsInterfaceMetricEntry {
                interface_index: 18,
                interface_alias: "Wi-Fi".to_string(),
                media_tier: "wifi".to_string(),
                metric: Some(25),
                mtu_bytes: Some(1500),
                state: "connected".to_string(),
                ipv4_subnets: vec!["192.168.50.0/24".to_string()],
            },
        ];
        let (valid, conflicts) = evaluate_windows_metric_hierarchy(&bad_entries);
        assert!(!valid);
        assert!(
            conflicts.iter().any(|c| c.contains("Same-subnet multi-homed conflict")),
            "expected same-subnet conflict, got: {conflicts:?}"
        );

        let good_entries = vec![
            WindowsInterfaceMetricEntry {
                interface_index: 12,
                interface_alias: "Ethernet 4".to_string(),
                media_tier: "usb4-p2p".to_string(),
                metric: Some(5),
                mtu_bytes: Some(62000),
                state: "connected".to_string(),
                ipv4_subnets: vec!["172.31.241.248/30".to_string()],
            },
            WindowsInterfaceMetricEntry {
                interface_index: 15,
                interface_alias: "Ethernet 2".to_string(),
                media_tier: "wired-lan".to_string(),
                metric: Some(10),
                mtu_bytes: Some(1500),
                state: "connected".to_string(),
                ipv4_subnets: vec!["192.168.50.0/24".to_string()],
            },
            WindowsInterfaceMetricEntry {
                interface_index: 18,
                interface_alias: "Wi-Fi".to_string(),
                media_tier: "wifi".to_string(),
                metric: Some(35),
                mtu_bytes: Some(1500),
                state: "connected".to_string(),
                ipv4_subnets: vec!["192.168.50.0/24".to_string()],
            },
        ];
        let (valid_good, conflicts_good) = evaluate_windows_metric_hierarchy(&good_entries);
        assert!(valid_good, "expected valid hierarchy, got: {conflicts_good:?}");
    }

    #[test]
    fn test_evaluate_linux_switch_and_usb4_extension_custody_from_json_and_negative_controls() {
        let topo = SfpSwitchTopology {
            host: "minisforum".to_string(),
            namespace: "ms01-switch".to_string(),
            bridge: "sw0".to_string(),
            state_dir: "/var/lib/ms01-network-switch".to_string(),
            extension_state_dir: Some("/var/lib/ms01-usb4-extension".to_string()),
            observe_state_dir: Some("/var/lib/ms01-network-observe".to_string()),
            enrollment_path: None,
            usb4_enrollment_path: None,
            sfp_ports: vec![SfpPortSpec {
                name: "enp2s0f0np0".to_string(),
                pci: "0000:02:00.0".to_string(),
                mac: "58:47:ca:7b:6d:98".to_string(),
                vendor: "0x8086".to_string(),
                device: "0x1572".to_string(),
                driver: "i40e".to_string(),
                role: "uplink".to_string(),
                connected_peer: None,
                connected_peer_mac: None,
                connected_peer_ipv4: None,
                speed_gbps: Some(10),
            }],
            usb4_extensions: vec![Usb4BridgeExtensionSpec {
                interface: "usb4-milly".to_string(),
                root_veth: "ms01-milly".to_string(),
                bridge_veth: "milly-host".to_string(),
                sysfs_path: "/sys/devices/pci0000:00/0000:00:0d.2/domain0/0-0/0-1/0-1.0".to_string(),
                peer_uuid: "bab38780-0019-a9b4-ffff-ffffffffffff".to_string(),
                peer_name: "millylaptop1".to_string(),
                driver: "thunderbolt-net".to_string(),
                control_subnet: "10.254.253.0/30".to_string(),
                local_control_ipv4: "10.254.253.1".to_string(),
                peer_control_ipv4: "10.254.253.2".to_string(),
                bridged_lan_ipv4: Some("192.168.50.91".to_string()),
            }],
        };

        let ns_links: serde_json::Value = serde_json::json!([
            {
                "ifindex": 2,
                "ifname": "sw0",
                "address": "58:47:ca:7b:6d:98",
                "ifalias": "ms01-switch:abc"
            },
            {
                "ifindex": 4,
                "ifname": "enp2s0f0np0",
                "address": "58:47:ca:7b:6d:98",
                "master": "sw0",
                "ifalias": "ms01-switch:abc",
                "operstate": "UP",
                "carrier": 1
            },
            {
                "ifindex": 9,
                "ifname": "usb4-milly",
                "address": "02:11:22:33:44:55",
                "master": "sw0",
                "mtu": 1500,
                "flags": ["BROADCAST", "MULTICAST", "UP", "LOWER_UP"],
                "ifalias": "ms01-usb4:def"
            },
            {
                "ifindex": 11,
                "link_index": 10,
                "ifname": "milly-host",
                "master": "sw0",
                "mtu": 1500,
                "flags": ["BROADCAST", "MULTICAST", "UP", "LOWER_UP"],
                "ifalias": "ms01-usb4:def"
            }
        ]);
        let ns_addrs: serde_json::Value = serde_json::json!([
            { "ifname": "enp2s0f0np0", "addr_info": [] },
            { "ifname": "usb4-milly", "addr_info": [] },
            { "ifname": "milly-host", "addr_info": [] }
        ]);
        let root_links: serde_json::Value = serde_json::json!([
            {
                "ifindex": 10,
                "link_index": 11,
                "ifname": "ms01-milly",
                "mtu": 1500,
                "flags": ["BROADCAST", "MULTICAST", "UP", "LOWER_UP"],
                "ifalias": "ms01-usb4:def"
            }
        ]);
        let root_addrs: serde_json::Value = serde_json::json!([
            {
                "ifname": "ms01-milly",
                "addr_info": [{ "local": "10.254.253.1", "prefixlen": 30 }]
            }
        ]);

        let live_sw_journal: serde_json::Value = serde_json::json!({
            "schema": 1,
            "boot": "boot-123",
            "owner": "abc",
            "inode": 4026532830u64,
            "original": {
                "enp2s0f0np0": { "ifindex": 4, "mac": "58:47:ca:7b:6d:98" }
            },
            "phase": "running"
        });
        let sw_live = evaluate_linux_switch_custody_from_json(
            &topo,
            "/var/lib/ms01-network-switch/current.json".to_string(),
            Some(&live_sw_journal),
            Some(&ns_links),
            Some(&ns_addrs),
            Some("boot-123".to_string()),
            Some(4026532830u64),
        );
        assert!(sw_live.phase_running);
        assert!(sw_live.boot_id_matches);
        assert!(sw_live.namespace_inode_matches);
        assert!(sw_live.bridge_mac_matches);
        assert!(sw_live.bridge_alias_matches);
        assert!(sw_live.all_ports_valid, "errors: {:?}", sw_live.errors);

        let live_ext_journal: serde_json::Value = serde_json::json!({
            "schema": 1,
            "boot": "boot-123",
            "owner": "def",
            "parent_owner": "abc",
            "parent_inode": 4026532830u64,
            "bridge_mac": "58:47:ca:7b:6d:98",
            "peer_uuid": "bab38780-0019-a9b4-ffff-ffffffffffff",
            "path": "/sys/devices/pci0000:00/0000:00:0d.2/domain0/0-0/0-1/0-1.0",
            "driver": "thunderbolt-net",
            "original": {
                "ifindex": 9,
                "address": "02:11:22:33:44:55"
            },
            "phase": "running"
        });
        let ext_live = evaluate_linux_usb4_extension_custody_from_json(
            &topo,
            "/var/lib/ms01-usb4-extension/current.json".to_string(),
            Some(&live_ext_journal),
            Some(&root_links),
            Some(&root_addrs),
            Some(&ns_links),
            Some(&ns_addrs),
            Some("boot-123"),
            Some(4026532830u64),
        )
        .expect("live ext custody");
        assert!(ext_live.custody_valid, "errors: {:?}", ext_live.errors);
        assert!(ext_live.veth_symmetry_valid);

        // Negative control 1: owner-prefix collision (owner="de" vs live alias "ms01-usb4:def") must fail closed!
        let prefix_owner_journal: serde_json::Value = serde_json::json!({
            "schema": 1,
            "boot": "boot-123",
            "owner": "de",
            "parent_owner": "abc",
            "parent_inode": 4026532830u64,
            "bridge_mac": "58:47:ca:7b:6d:98",
            "phase": "running"
        });
        let ext_prefix_collision = evaluate_linux_usb4_extension_custody_from_json(
            &topo,
            "/var/lib/ms01-usb4-extension/current.json".to_string(),
            Some(&prefix_owner_journal),
            Some(&root_links),
            Some(&root_addrs),
            Some(&ns_links),
            Some(&ns_addrs),
            Some("boot-123"),
            Some(4026532830u64),
        )
        .expect("prefix collision ext custody");
        assert!(
            !ext_prefix_collision.custody_valid,
            "owner prefix 'de' must not match 'ms01-usb4:def'"
        );

        // Negative control 2: missing owner in journal must fail closed (no fallback to generic prefix)
        let missing_owner_journal: serde_json::Value = serde_json::json!({
            "schema": 1,
            "boot": "boot-123",
            "parent_owner": "abc",
            "parent_inode": 4026532830u64,
            "phase": "running"
        });
        let ext_missing_owner = evaluate_linux_usb4_extension_custody_from_json(
            &topo,
            "/var/lib/ms01-usb4-extension/current.json".to_string(),
            Some(&missing_owner_journal),
            Some(&root_links),
            Some(&root_addrs),
            Some(&ns_links),
            Some(&ns_addrs),
            Some("boot-123"),
            Some(4026532830u64),
        )
        .expect("missing owner ext custody");
        assert!(!ext_missing_owner.custody_valid);

        // Negative control 3: wrong peer_uuid or phase != "running" must fail closed
        let wrong_uuid_journal: serde_json::Value = serde_json::json!({
            "schema": 1,
            "boot": "boot-123",
            "owner": "def",
            "parent_owner": "abc",
            "parent_inode": 4026532830u64,
            "peer_uuid": "00000000-0000-0000-0000-000000000000",
            "phase": "running"
        });
        let ext_wrong_uuid = evaluate_linux_usb4_extension_custody_from_json(
            &topo,
            "/var/lib/ms01-usb4-extension/current.json".to_string(),
            Some(&wrong_uuid_journal),
            Some(&root_links),
            Some(&root_addrs),
            Some(&ns_links),
            Some(&ns_addrs),
            Some("boot-123"),
            Some(4026532830u64),
        )
        .expect("wrong uuid ext custody");
        assert!(!ext_wrong_uuid.custody_valid);
        assert!(ext_wrong_uuid.errors.iter().any(|e| e.contains("peer_uuid mismatch")));
    }

    #[test]
    fn test_route_and_ipv6_exact_matching_rejects_partial_or_wrong_gateway() {
        let win_route_sample = "\
===========================================================================
Interface List
 11...10 e1 8e 5a 26 43 ......Intel(R) Ethernet Controller I226-LM
===========================================================================

IPv4 Route Table
===========================================================================
Active Routes:
Network Destination        Netmask          Gateway       Interface  Metric
          0.0.0.0          0.0.0.0     192.168.50.1    192.168.50.90     35
     10.254.253.0  255.255.255.252  192.168.50.119    192.168.50.90     15
===========================================================================
Persistent Routes:
  None
";
        let net_ip: Ipv4Addr = "10.254.253.0".parse().unwrap();
        let good_gw: Ipv4Addr = "192.168.50.119".parse().unwrap();
        let wrong_gw: Ipv4Addr = "192.168.50.1".parse().unwrap();

        assert!(windows_route_print_has_exact_route(win_route_sample, net_ip, 30, good_gw));
        // Wrong gateway or wrong prefix length must fail closed
        assert!(!windows_route_print_has_exact_route(win_route_sample, net_ip, 30, wrong_gw));
        assert!(!windows_route_print_has_exact_route(win_route_sample, net_ip, 24, good_gw));
        assert!(windows_route_print_has_conflicting_destination(
            win_route_sample,
            net_ip,
            30,
            wrong_gw
        ));

        // IPv6 exact matching: ::9 must NOT match ::99
        let netsh_v6_sample = "\
Address fd5f:743d:6f55:50::99 Parameters
---------------------------------------------------------
Interface Luid     : Ethernet 2
Scope Id           : 0.0
DAD State          : Preferred
";
        let ip_9: Ipv6Addr = "fd5f:743d:6f55:50::9".parse().unwrap();
        let ip_99: Ipv6Addr = "fd5f:743d:6f55:50::99".parse().unwrap();
        assert!(!windows_netsh_output_has_exact_ipv6(netsh_v6_sample, ip_9));
        assert!(windows_netsh_output_has_exact_ipv6(netsh_v6_sample, ip_99));
    }

    #[test]
    fn test_ms01_usb4_check_receipt_validation_and_vnv_pending_intake_gates() {
        // Synthesized receipt missing boot/owner/parent_owner/peer_uuid must be rejected
        let fake_receipt = r#"{
            "schema": "vigil.ms01-usb4-check/v1",
            "phase": "running",
            "ready": true,
            "status": "CUSTODY_VERIFIED",
            "namespace": "ms01-switch",
            "bridge": "sw0"
        }"#;
        assert!(parse_and_validate_ms01_usb4_check_receipt(fake_receipt).is_err());

        let valid_receipt = r#"{
            "schema": "vigil.ms01-usb4-check/v1",
            "phase": "running",
            "ready": true,
            "readiness_scope": "structural_custody_only",
            "network_ready": null,
            "status": "CUSTODY_VERIFIED",
            "namespace": "ms01-switch",
            "bridge": "sw0",
            "control_veth": "ms01-milly",
            "interface": "usb4-milly",
            "peer_uuid": "bab38780-0019-a9b4-ffff-ffffffffffff",
            "boot": "boot-123",
            "parent_owner": "abc",
            "owner": "def"
        }"#;
        assert!(parse_and_validate_ms01_usb4_check_receipt(valid_receipt).is_ok());

        // Verify V&V gates use PENDING-FABRIC-F09..F13, never grant vacuous PASS when providers are None or routes missing
        let win_status = SwitchStatusReport {
            local_host: "dtm-p1gen9".to_string(),
            os_family: "windows".to_string(),
            registry_path: None,
            matched_topology: None,
            ipam_allocations: None,
            topology_links: Vec::new(),
            local_usb4_routers: Vec::new(),
            shared_subnet_routes: Vec::new(),
            linux_switch_custody: None,
            linux_usb4_extension_custody: None,
            linux_canary_ula_status: Vec::new(),
            windows_metric_hierarchy: None,
            custody_valid: true,
            warnings: Vec::new(),
            errors: Vec::new(),
        };
        let (overall, _, gates) = evaluate_vnv_gates_from_status(&win_status);
        assert_eq!(overall, "FAIL");
        assert!(gates.iter().all(|g| g.trace_id.starts_with("PENDING-FABRIC-")));
        assert!(gates.iter().all(|g| !g.acceptance_credit));
        let f10 = gates.iter().find(|g| g.trace_id == "PENDING-FABRIC-F10").unwrap();
        assert_eq!(f10.status, "SKIP_NOT_APPLICABLE");
        let f12 = gates.iter().find(|g| g.trace_id == "PENDING-FABRIC-F12").unwrap();
        assert_eq!(f12.status, "BLOCKED_MISSING_PROVIDER");
        let f13 = gates.iter().find(|g| g.trace_id == "PENDING-FABRIC-F13").unwrap();
        assert_eq!(f13.status, "FAIL");
    }

    #[test]
    fn test_windows_route_output_succeeded_rejects_false_positive_exit_zero() {
        assert!(windows_route_output_succeeded(true, " OK!\r\n", ""));
        assert!(!windows_route_output_succeeded(
            true,
            "",
            "The route change failed: Element not found."
        ));
        assert!(!windows_route_output_succeeded(
            true,
            "The route addition failed: Access is denied.",
            ""
        ));
        assert!(!windows_route_output_succeeded(
            true,
            "The route addition failed: The object already exists.",
            ""
        ));
        assert!(!windows_route_output_succeeded(
            true,
            "The route addition failed: The parameter is incorrect.",
            ""
        ));
        assert!(!windows_route_output_succeeded(
            true,
            "The requested operation requires elevation.",
            ""
        ));
        assert!(!windows_route_output_succeeded(false, " OK!\r\n", ""));
    }
}
