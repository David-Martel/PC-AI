use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::{Path, PathBuf};

// trace: PENDING-FABRIC-F09 [pending-tpm-intake]
// trace: PENDING-FABRIC-F11 [pending-tpm-intake]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Usb4LifecycleState {
    Asleep,
    Disconnected,
    Probing,
    Ready,
}

impl Usb4LifecycleState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Asleep => "asleep",
            Self::Disconnected => "disconnected",
            Self::Probing => "probing",
            Self::Ready => "ready",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usb4LifecyclePolicy {
    #[serde(default)]
    pub states: Vec<Usb4LifecycleState>,
    pub default_evidence_ttl_seconds: u64,
    pub bounded_resume_reprobe_max_attempts: u32,
    pub bounded_resume_reprobe_timeout_ms: u64,
    pub bounded_resume_total_budget_ms: u64,
    pub withdraw_owned_stale_routes_on_expire_or_disconnect: bool,
    pub require_identity_bound_canary: bool,
    pub allow_always_on_transit_claim_when_selective_suspend: bool,
    pub ms01_uplink_cap_mbps: Option<u32>,
    #[serde(default)]
    pub preserved_active_pairs: Vec<String>,
    #[serde(default)]
    pub preserved_bridges: Vec<String>,
}

impl Default for Usb4LifecyclePolicy {
    fn default() -> Self {
        Self {
            states: vec![
                Usb4LifecycleState::Asleep,
                Usb4LifecycleState::Disconnected,
                Usb4LifecycleState::Probing,
                Usb4LifecycleState::Ready,
            ],
            default_evidence_ttl_seconds: 300,
            bounded_resume_reprobe_max_attempts: 3,
            bounded_resume_reprobe_timeout_ms: 250,
            bounded_resume_total_budget_ms: 1500,
            withdraw_owned_stale_routes_on_expire_or_disconnect: true,
            require_identity_bound_canary: true,
            allow_always_on_transit_claim_when_selective_suspend: false,
            ms01_uplink_cap_mbps: Some(1000),
            preserved_active_pairs: vec!["dtm-p1gen7<->dtm-p1gen9".to_string()],
            preserved_bridges: vec!["ms01-milly-usb4-bridge".to_string()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PeerRegistryFile {
    pub schema_version: Option<serde_json::Value>,
    pub ula_prefix: Option<String>,
    pub deterministic_ula_subnet: Option<String>,
    pub pairwise_ipv4_pool: Option<String>,
    pub default_mtu_bytes: Option<u32>,
    pub default_interface_metric: Option<u32>,
    #[serde(default)]
    pub usb4_lifecycle_policy: Option<Usb4LifecyclePolicy>,
    #[serde(default)]
    pub ipam_allocations: Option<IpamAllocations>,
    #[serde(default)]
    pub sfp_switch_topologies: Vec<SfpSwitchTopology>,
    #[serde(default)]
    pub topology_links: Vec<TopologyLinkSpec>,
    #[serde(default)]
    pub peers: Option<serde_json::Value>,
    #[serde(default)]
    pub fleet_peers: Vec<PeerRegistryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IpamAllocations {
    pub p2p_thunderbolt_ula_prefix: String,
    pub p2p_deterministic_ula_subnet: Option<String>,
    pub p2p_link_local_prefix: Option<String>,
    pub bridged_lan_canary_ula_prefix: String,
    pub bridged_lan_policy_table: Option<u32>,
    pub bridged_lan_policy_priority: Option<u32>,
    #[serde(default)]
    pub bridged_lan_canary_endpoints: Vec<BridgedLanCanaryEndpoint>,
    #[serde(default)]
    pub shared_switch_subnets: Vec<SharedSwitchSubnet>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SharedSwitchSubnet {
    pub subnet: String,
    pub gateway_host: String,
    pub gateway_ipv4: String,
    pub role: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TopologyLinkSpec {
    pub link_id: String,
    pub source_host: String,
    pub source_interface: String,
    pub source_hub_model: Option<String>,
    pub source_hub_id: Option<String>,
    pub target_host: String,
    pub target_interface: String,
    pub target_hub_model: Option<String>,
    pub target_hub_id: Option<String>,
    pub link_intent: String,
    pub bridge_domain: Option<String>,
    #[serde(default)]
    pub l3_subnets: Vec<String>,
    #[serde(default)]
    pub lifecycle_state_hint: Option<Usb4LifecycleState>,
    #[serde(default)]
    pub always_on_transit_capable: Option<bool>,
    #[serde(default)]
    pub preserve_existing_pair: Option<bool>,
    #[serde(default)]
    pub preserve_bridge_custody: Option<bool>,
    #[serde(default)]
    pub downstream_link_status: Option<String>,
    #[serde(default)]
    pub windows_xdomain_present: Option<bool>,
    #[serde(default)]
    pub expected_peer_macs: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BridgedLanCanaryEndpoint {
    pub host: String,
    pub interface: String,
    pub ula_address: String,
    pub role: String,
    pub policy_table: Option<u32>,
    pub policy_priority: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SfpSwitchTopology {
    pub host: String,
    pub namespace: String,
    pub bridge: String,
    pub state_dir: String,
    pub extension_state_dir: Option<String>,
    pub observe_state_dir: Option<String>,
    pub enrollment_path: Option<String>,
    pub usb4_enrollment_path: Option<String>,
    #[serde(default)]
    pub sfp_ports: Vec<SfpPortSpec>,
    #[serde(default)]
    pub usb4_extensions: Vec<Usb4BridgeExtensionSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SfpPortSpec {
    pub name: String,
    pub pci: String,
    pub mac: String,
    pub vendor: String,
    pub device: String,
    pub driver: String,
    pub role: String,
    pub connected_peer: Option<String>,
    pub connected_peer_mac: Option<String>,
    pub connected_peer_ipv4: Option<String>,
    pub speed_gbps: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usb4BridgeExtensionSpec {
    pub interface: String,
    pub root_veth: String,
    pub bridge_veth: String,
    pub sysfs_path: String,
    pub peer_uuid: String,
    pub peer_name: String,
    pub driver: String,
    pub control_subnet: String,
    pub local_control_ipv4: String,
    pub peer_control_ipv4: String,
    pub bridged_lan_ipv4: Option<String>,
}

impl PeerRegistryFile {
    pub fn normalize_and_validate(mut self) -> Result<Self> {
        if self.fleet_peers.is_empty() {
            if let Some(serde_json::Value::Array(ref arr)) = self.peers {
                let parsed: Vec<PeerRegistryEntry> = serde_json::from_value(serde_json::Value::Array(arr.clone()))
                    .context("Failed to parse legacy 'peers' array into PeerRegistryEntry list")?;
                self.fleet_peers = parsed;
            }
        }
        validate_peer_registry(&self)?;
        Ok(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PeerRegistryEntry {
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub os_family: Option<String>,
    pub node_id_hex: Option<String>,
    pub short_link_local: Option<String>,
    pub short_ula: Option<String>,
    pub tb_host_alias: Option<String>,
    pub ssh_port: Option<u16>,
    pub ssh_user: Option<String>,
    pub win_rm_port: Option<u16>,
    #[serde(default)]
    pub lan_addresses: Vec<String>,
    #[serde(default)]
    pub default_shares: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub struct PairwiseIpv4Contract {
    pub pair_key: String,
    pub subnet_prefix: String,
    pub prefix_length: u8,
    pub local_ipv4_address: String,
    pub peer_ipv4_address: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct ThunderboltPeerIdentity {
    pub computer_name: String,
    pub input_name: String,
    pub os_family: String,
    pub is_known_fleet_peer: bool,
    pub node_token_hextets: String,
    pub deterministic_link_local: String,
    pub deterministic_ula: String,
    pub short_link_local: Option<String>,
    pub short_ula: Option<String>,
    pub primary_link_local: String,
    pub primary_ula: String,
    pub all_link_local_addresses: Vec<String>,
    pub all_ula_addresses: Vec<String>,
    pub tb_host_alias: String,
    pub ssh_port: u16,
    pub ssh_user: String,
    pub win_rm_port: u16,
    pub lan_addresses: Vec<String>,
    pub default_shares: Vec<String>,
    pub peer_computer_name: Option<String>,
    pub pairwise_ipv4: Option<PairwiseIpv4Contract>,
    pub registry_path: Option<String>,
}

pub fn parse_ipv6_cidr(cidr: &str) -> Result<(Ipv6Addr, u8)> {
    let (ip_str, prefix_str) = cidr
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("Invalid IPv6 CIDR '{cidr}': missing '/prefix'"))?;
    let ip: Ipv6Addr = ip_str
        .trim()
        .parse()
        .with_context(|| format!("Invalid IPv6 address in CIDR '{cidr}'"))?;
    let prefix_len: u8 = prefix_str
        .trim()
        .parse()
        .with_context(|| format!("Invalid prefix length in IPv6 CIDR '{cidr}'"))?;
    if prefix_len == 0 || prefix_len > 128 {
        anyhow::bail!("Invalid IPv6 prefix length {prefix_len} in '{cidr}'");
    }
    Ok((ip, prefix_len))
}

fn ipv6_mask(prefix_len: u8) -> u128 {
    if prefix_len == 0 {
        0
    } else if prefix_len >= 128 {
        u128::MAX
    } else {
        u128::MAX << (128 - u32::from(prefix_len))
    }
}

pub fn ipv6_cidr_overlaps(a: (Ipv6Addr, u8), b: (Ipv6Addr, u8)) -> bool {
    let min_prefix = a.1.min(b.1);
    let mask = ipv6_mask(min_prefix);
    (u128::from(a.0) & mask) == (u128::from(b.0) & mask)
}

pub fn ipv6_in_subnet(addr: Ipv6Addr, subnet: (Ipv6Addr, u8)) -> bool {
    let mask = ipv6_mask(subnet.1);
    (u128::from(addr) & mask) == (u128::from(subnet.0) & mask)
}

pub fn parse_ipv4_cidr(cidr: &str) -> Result<(Ipv4Addr, u8)> {
    let (ip_str, prefix_str) = cidr
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("Invalid IPv4 CIDR '{cidr}': missing '/prefix'"))?;
    let ip: Ipv4Addr = ip_str
        .trim()
        .parse()
        .with_context(|| format!("Invalid IPv4 address in CIDR '{cidr}'"))?;
    let prefix_len: u8 = prefix_str
        .trim()
        .parse()
        .with_context(|| format!("Invalid prefix length in IPv4 CIDR '{cidr}'"))?;
    if prefix_len == 0 || prefix_len > 32 {
        anyhow::bail!("Invalid IPv4 prefix length {prefix_len} in '{cidr}'");
    }
    Ok((ip, prefix_len))
}

fn ipv4_mask(prefix_len: u8) -> u32 {
    if prefix_len == 0 {
        0
    } else if prefix_len >= 32 {
        u32::MAX
    } else {
        u32::MAX << (32 - u32::from(prefix_len))
    }
}

pub fn ipv4_cidr_overlaps(a: (Ipv4Addr, u8), b: (Ipv4Addr, u8)) -> bool {
    let min_prefix = a.1.min(b.1);
    let mask = ipv4_mask(min_prefix);
    (u32::from(a.0) & mask) == (u32::from(b.0) & mask)
}

// trace: PENDING-FABRIC-F09 [pending-tpm-intake]
pub fn detect_l2_bridge_cycles(links: &[TopologyLinkSpec]) -> Result<()> {
    let mut parent: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    fn find(parent: &mut std::collections::HashMap<String, String>, node: &str) -> String {
        let p = parent
            .entry(node.to_string())
            .or_insert_with(|| node.to_string())
            .clone();
        if p == node {
            return p;
        }
        let root = find(parent, &p);
        parent.insert(node.to_string(), root.clone());
        root
    }

    for link in links {
        if link.link_intent != "transparent-bridge" {
            continue;
        }
        let src_host = link.source_host.trim().to_ascii_lowercase();
        let src_if = link.source_interface.trim();
        let dst_host = link.target_host.trim().to_ascii_lowercase();
        let dst_if = link.target_interface.trim();
        if src_host.is_empty() || src_if.is_empty() || dst_host.is_empty() || dst_if.is_empty() {
            anyhow::bail!(
                "transparent-bridge link '{}' has empty sourceHost, sourceInterface, targetHost, or targetInterface",
                link.link_id
            );
        }
        if src_host == dst_host {
            anyhow::bail!(
                "transparent-bridge link '{}' cannot connect host '{}' to itself",
                link.link_id,
                link.source_host
            );
        }
        let domain = link
            .bridge_domain
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "transparent-bridge link '{}' must specify a non-empty bridgeDomain",
                    link.link_id
                )
            })?
            .to_ascii_lowercase();
        let u = format!("{domain}::{src_host}");
        let v = format!("{domain}::{dst_host}");
        let ru = find(&mut parent, &u);
        let rv = find(&mut parent, &v);
        if ru == rv {
            anyhow::bail!(
                "L2 transparent-bridge cycle detected in topologyLinks involving link '{}' between '{}' and '{}' in bridgeDomain '{}' (parallel links must use routed-p2p)",
                link.link_id,
                link.source_host,
                link.target_host,
                domain
            );
        }
        parent.insert(ru, rv);
    }
    Ok(())
}

pub fn validate_peer_registry(reg: &PeerRegistryFile) -> Result<()> {
    match &reg.schema_version {
        Some(serde_json::Value::Number(n)) if n.as_u64() == Some(1) => {}
        Some(serde_json::Value::String(s)) if s == "1" || s == "1.0.0" => {}
        Some(other) => {
            anyhow::bail!("Unsupported thunderbolt-peers.json schemaVersion: {other}");
        }
        None => {
            anyhow::bail!("Missing schemaVersion in thunderbolt-peers.json");
        }
    }

    // Validate Linux /30 profile dictionary if present as an object
    if let Some(serde_json::Value::Object(ref map)) = reg.peers {
        let mut seen_subnets = HashSet::new();
        for (peer_key, val) in map {
            let local_str = val
                .get("localAddress")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("peers.{peer_key} missing localAddress"))?;
            let peer_str = val
                .get("peerAddress")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("peers.{peer_key} missing peerAddress"))?;
            let prefix_len = val
                .get("prefixLength")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow::anyhow!("peers.{peer_key} missing prefixLength"))?;
            if prefix_len != 30 {
                anyhow::bail!("peers.{peer_key} must use prefixLength 30, got {prefix_len}");
            }
            let local_ip: Ipv4Addr = local_str
                .parse()
                .with_context(|| format!("peers.{peer_key} invalid localAddress '{local_str}'"))?;
            let peer_ip: Ipv4Addr = peer_str
                .parse()
                .with_context(|| format!("peers.{peer_key} invalid peerAddress '{peer_str}'"))?;
            if local_ip == peer_ip {
                anyhow::bail!("peers.{peer_key} localAddress and peerAddress cannot be identical ({local_ip})");
            }
            let mask: u32 = !0u32 << (32 - prefix_len as u32);
            let net_local = u32::from(local_ip) & mask;
            let net_peer = u32::from(peer_ip) & mask;
            if net_local != net_peer {
                anyhow::bail!(
                    "peers.{peer_key} localAddress ({local_ip}) and peerAddress ({peer_ip}) are not in the same /30 subnet"
                );
            }
            if !seen_subnets.insert(net_local) {
                anyhow::bail!(
                    "peers.{peer_key} /30 subnet ({}) overlaps with another peer profile",
                    Ipv4Addr::from(net_local)
                );
            }
        }
    }

    let mut seen_names = HashSet::new();
    let mut seen_aliases = HashSet::new();
    let mut seen_short_lla = HashSet::new();
    let mut seen_short_ula = HashSet::new();
    let mut seen_tb_aliases = HashSet::new();
    let mut seen_lan_ips = HashSet::new();

    for entry in &reg.fleet_peers {
        let canon = entry.name.trim().to_ascii_lowercase();
        if canon.is_empty() {
            anyhow::bail!("fleetPeers entry has an empty name");
        }
        if !seen_names.insert(canon.clone()) {
            anyhow::bail!("Duplicate peer name '{}' in thunderbolt-peers.json", entry.name);
        }
        if !seen_aliases.insert(canon.clone()) {
            anyhow::bail!(
                "Peer name '{}' collides with an alias of another host in thunderbolt-peers.json",
                entry.name
            );
        }

        for alias in &entry.aliases {
            let clean_alias = alias.trim().to_ascii_lowercase();
            if clean_alias.is_empty() {
                anyhow::bail!("Empty alias on peer '{}'", entry.name);
            }
            if !seen_aliases.insert(clean_alias.clone()) {
                anyhow::bail!(
                    "Duplicate alias '{}' on peer '{}' in thunderbolt-peers.json",
                    alias,
                    entry.name
                );
            }
            let stripped = strip_domain_suffixes(alias);
            if stripped != canon && seen_names.contains(&stripped) {
                anyhow::bail!(
                    "Alias '{}' on peer '{}' collides with canonical peer '{}'",
                    alias,
                    entry.name,
                    stripped
                );
            }
        }

        if let Some(ref lla) = entry.short_link_local {
            let parsed: Ipv6Addr = lla
                .parse()
                .with_context(|| format!("Invalid shortLinkLocal '{lla}' on peer '{}'", entry.name))?;
            if (parsed.segments()[0] & 0xffc0) != 0xfe80 {
                anyhow::bail!("shortLinkLocal '{lla}' on peer '{}' is not in fe80::/10", entry.name);
            }
            if !seen_short_lla.insert(parsed) {
                anyhow::bail!(
                    "Duplicate shortLinkLocal '{lla}' on peer '{}' in thunderbolt-peers.json",
                    entry.name
                );
            }
        }

        if let Some(ref ula) = entry.short_ula {
            let parsed: Ipv6Addr = ula
                .parse()
                .with_context(|| format!("Invalid shortUla '{ula}' on peer '{}'", entry.name))?;
            if (parsed.segments()[0] & 0xfe00) != 0xfc00 {
                anyhow::bail!("shortUla '{ula}' on peer '{}' is not in fc00::/7", entry.name);
            }
            if !seen_short_ula.insert(parsed) {
                anyhow::bail!(
                    "Duplicate shortUla '{ula}' on peer '{}' in thunderbolt-peers.json",
                    entry.name
                );
            }
        }

        if let Some(ref tb_alias) = entry.tb_host_alias {
            let lower_tb = tb_alias.trim().to_ascii_lowercase();
            if lower_tb.is_empty() || !seen_tb_aliases.insert(lower_tb) {
                anyhow::bail!("Duplicate or empty tbHostAlias '{}' on peer '{}'", tb_alias, entry.name);
            }
        }

        for lan_ip in &entry.lan_addresses {
            let parsed: IpAddr = lan_ip
                .parse()
                .with_context(|| format!("Invalid LAN/Tailscale IP '{lan_ip}' on peer '{}'", entry.name))?;
            if !seen_lan_ips.insert(parsed) {
                anyhow::bail!(
                    "Duplicate LAN/Tailscale IP '{lan_ip}' assigned to peer '{}' (already assigned to another host)",
                    entry.name
                );
            }
        }
    }

    if let Some(ref ipam) = reg.ipam_allocations {
        let p2p_ula = parse_ipv6_cidr(&ipam.p2p_thunderbolt_ula_prefix)
            .context("ipamAllocations.p2pThunderboltUlaPrefix invalid")?;
        if (p2p_ula.0.segments()[0] & 0xfe00) != 0xfc00 {
            anyhow::bail!(
                "ipamAllocations.p2pThunderboltUlaPrefix '{}' is not in fc00::/7",
                ipam.p2p_thunderbolt_ula_prefix
            );
        }

        let canary_ula = parse_ipv6_cidr(&ipam.bridged_lan_canary_ula_prefix)
            .context("ipamAllocations.bridgedLanCanaryUlaPrefix invalid")?;
        if (canary_ula.0.segments()[0] & 0xfe00) != 0xfc00 {
            anyhow::bail!(
                "ipamAllocations.bridgedLanCanaryUlaPrefix '{}' is not in fc00::/7",
                ipam.bridged_lan_canary_ula_prefix
            );
        }

        if ipv6_cidr_overlaps(p2p_ula, canary_ula) {
            anyhow::bail!(
                "ipamAllocations ULA prefixes overlap: p2pThunderboltUlaPrefix '{}' overlaps with bridgedLanCanaryUlaPrefix '{}'",
                ipam.p2p_thunderbolt_ula_prefix,
                ipam.bridged_lan_canary_ula_prefix
            );
        }

        if let Some(table) = ipam.bridged_lan_policy_table {
            if table == 0 || table > 65535 || matches!(table, 253..=255) {
                anyhow::bail!(
                    "ipamAllocations.bridgedLanPolicyTable {table} is invalid or collides with reserved Linux routing tables (253..=255)"
                );
            }
        }

        let mut seen_canary_ips = HashSet::new();
        for ep in &ipam.bridged_lan_canary_endpoints {
            if ep.host.trim().is_empty() || ep.interface.trim().is_empty() || ep.role.trim().is_empty() {
                anyhow::bail!("ipamAllocations.bridgedLanCanaryEndpoints entry has empty host, interface, or role");
            }
            let (ep_ip, ep_prefix) = parse_ipv6_cidr(&ep.ula_address).with_context(|| {
                format!(
                    "Invalid bridgedLanCanaryEndpoints ulaAddress '{}' on {}:{}",
                    ep.ula_address, ep.host, ep.interface
                )
            })?;
            if ep_prefix != canary_ula.1 || !ipv6_in_subnet(ep_ip, canary_ula) {
                anyhow::bail!(
                    "bridgedLanCanaryEndpoints ulaAddress '{}' on {}:{} does not belong to bridgedLanCanaryUlaPrefix '{}'",
                    ep.ula_address,
                    ep.host,
                    ep.interface,
                    ipam.bridged_lan_canary_ula_prefix
                );
            }
            if !seen_canary_ips.insert(ep_ip) || seen_short_ula.contains(&ep_ip) {
                anyhow::bail!(
                    "Duplicate or colliding canary ULA address '{}' on {}:{}",
                    ep.ula_address,
                    ep.host,
                    ep.interface
                );
            }
            if let Some(table) = ep.policy_table {
                if table == 0 || table > 65535 || matches!(table, 253..=255) {
                    anyhow::bail!(
                        "bridgedLanCanaryEndpoints policyTable {table} on {}:{} is invalid or reserved",
                        ep.host,
                        ep.interface
                    );
                }
            }
        }

        let mut parsed_shared_subnets: Vec<(String, (Ipv4Addr, u8))> = Vec::new();
        for sub in &ipam.shared_switch_subnets {
            if sub.gateway_host.trim().is_empty() || sub.role.trim().is_empty() {
                anyhow::bail!("ipamAllocations.sharedSwitchSubnets entry has empty gatewayHost or role");
            }
            let parsed_cidr = parse_ipv4_cidr(&sub.subnet)
                .with_context(|| format!("Invalid sharedSwitchSubnets subnet '{}'", sub.subnet))?;
            let _: Ipv4Addr = sub
                .gateway_ipv4
                .trim()
                .parse()
                .with_context(|| format!("Invalid sharedSwitchSubnets gatewayIpv4 '{}'", sub.gateway_ipv4))?;
            for (prev_raw, prev_cidr) in &parsed_shared_subnets {
                if ipv4_cidr_overlaps(*prev_cidr, parsed_cidr) {
                    anyhow::bail!(
                        "ipamAllocations.sharedSwitchSubnets '{}' overlaps with '{}'",
                        sub.subnet,
                        prev_raw
                    );
                }
            }
            parsed_shared_subnets.push((sub.subnet.clone(), parsed_cidr));
        }
    }

    for topo in &reg.sfp_switch_topologies {
        if topo.host.trim().is_empty()
            || topo.namespace.trim().is_empty()
            || topo.bridge.trim().is_empty()
            || topo.state_dir.trim().is_empty()
        {
            anyhow::bail!("sfpSwitchTopologies entry has empty host, namespace, bridge, or stateDir");
        }
        let mut seen_port_names = HashSet::new();
        let mut seen_port_pcis = HashSet::new();
        let mut seen_port_macs = HashSet::new();
        for port in &topo.sfp_ports {
            let name = port.name.trim();
            let pci = port.pci.trim().to_ascii_lowercase();
            let mac = port.mac.trim().to_ascii_lowercase();
            if name.is_empty() || pci.is_empty() || mac.is_empty() {
                anyhow::bail!("sfpPorts entry on host '{}' has empty name, pci, or mac", topo.host);
            }
            if !seen_port_names.insert(name.to_string()) {
                anyhow::bail!("Duplicate SFP+ port name '{name}' on host '{}'", topo.host);
            }
            if !seen_port_pcis.insert(pci.clone()) {
                anyhow::bail!("Duplicate SFP+ port PCI '{pci}' on host '{}'", topo.host);
            }
            if !seen_port_macs.insert(mac.clone()) {
                anyhow::bail!("Duplicate SFP+ port MAC '{mac}' on host '{}'", topo.host);
            }
            if let Some(ref peer_ip) = port.connected_peer_ipv4 {
                let _: Ipv4Addr = peer_ip.parse().with_context(|| {
                    format!(
                        "Invalid connectedPeerIpv4 '{peer_ip}' on SFP+ port '{name}' ({})",
                        topo.host
                    )
                })?;
            }
        }

        let mut seen_ext_ifaces = HashSet::new();
        for ext in &topo.usb4_extensions {
            let iface = ext.interface.trim();
            let root_veth = ext.root_veth.trim();
            let bridge_veth = ext.bridge_veth.trim();
            if iface.is_empty()
                || root_veth.is_empty()
                || bridge_veth.is_empty()
                || ext.sysfs_path.trim().is_empty()
                || ext.peer_uuid.trim().is_empty()
            {
                anyhow::bail!(
                    "usb4Extensions entry on host '{}' has empty interface, veth, sysfsPath, or peerUuid",
                    topo.host
                );
            }
            if iface == root_veth || iface == bridge_veth || root_veth == bridge_veth {
                anyhow::bail!(
                    "usb4Extensions on host '{}' must use distinct names for interface ('{iface}'), rootVeth ('{root_veth}'), and bridgeVeth ('{bridge_veth}')",
                    topo.host
                );
            }
            for n in [iface, root_veth, bridge_veth] {
                if !seen_ext_ifaces.insert(n.to_string()) || seen_port_names.contains(n) {
                    anyhow::bail!(
                        "Duplicate or colliding interface/veth name '{n}' in sfpSwitchTopologies on host '{}'",
                        topo.host
                    );
                }
            }
            let local_ip: Ipv4Addr = ext
                .local_control_ipv4
                .parse()
                .with_context(|| format!("Invalid localControlIpv4 '{}' on {iface}", ext.local_control_ipv4))?;
            let peer_ip: Ipv4Addr = ext
                .peer_control_ipv4
                .parse()
                .with_context(|| format!("Invalid peerControlIpv4 '{}' on {iface}", ext.peer_control_ipv4))?;
            if local_ip == peer_ip {
                anyhow::bail!("usb4Extensions '{iface}' localControlIpv4 and peerControlIpv4 cannot be identical");
            }
            let mask: u32 = !0u32 << 2;
            if (u32::from(local_ip) & mask) != (u32::from(peer_ip) & mask) {
                anyhow::bail!(
                    "usb4Extensions '{iface}' localControlIpv4 ({local_ip}) and peerControlIpv4 ({peer_ip}) are not in the same /30 subnet"
                );
            }
        }
    }

    if let Some(ref policy) = reg.usb4_lifecycle_policy {
        if policy.default_evidence_ttl_seconds == 0 || policy.default_evidence_ttl_seconds > 86_400 {
            anyhow::bail!(
                "usb4LifecyclePolicy.defaultEvidenceTtlSeconds must be between 1 and 86400 (got {})",
                policy.default_evidence_ttl_seconds
            );
        }
        if policy.bounded_resume_reprobe_max_attempts == 0 || policy.bounded_resume_reprobe_max_attempts > 10 {
            anyhow::bail!(
                "usb4LifecyclePolicy.boundedResumeReprobeMaxAttempts must be between 1 and 10 (got {})",
                policy.bounded_resume_reprobe_max_attempts
            );
        }
        if policy.bounded_resume_reprobe_timeout_ms == 0
            || policy.bounded_resume_total_budget_ms < policy.bounded_resume_reprobe_timeout_ms
        {
            anyhow::bail!(
                "usb4LifecyclePolicy boundedResumeReprobeTimeoutMs ({}) / boundedResumeTotalBudgetMs ({}) invalid",
                policy.bounded_resume_reprobe_timeout_ms,
                policy.bounded_resume_total_budget_ms
            );
        }
        if policy.allow_always_on_transit_claim_when_selective_suspend {
            anyhow::bail!(
                "usb4LifecyclePolicy.allowAlwaysOnTransitClaimWhenSelectiveSuspend must be false (SelectiveSuspend/Unsupported wake prohibits always-on transit claim)"
            );
        }
    }

    let mut seen_link_ids = HashSet::new();
    for link in &reg.topology_links {
        let id = link.link_id.trim();
        let src = link.source_host.trim();
        let dst = link.target_host.trim();
        let src_if = link.source_interface.trim();
        let dst_if = link.target_interface.trim();
        if id.is_empty() || src.is_empty() || dst.is_empty() || src_if.is_empty() || dst_if.is_empty() {
            anyhow::bail!("topologyLinks entry has empty linkId, sourceHost, targetHost, or interface");
        }
        if !seen_link_ids.insert(id.to_ascii_lowercase()) {
            anyhow::bail!("Duplicate topologyLinks linkId '{id}'");
        }
        if src.eq_ignore_ascii_case(dst) {
            anyhow::bail!("topologyLinks '{id}' cannot connect host '{src}' to itself");
        }
        match link.link_intent.as_str() {
            "transparent-bridge" => {
                let dom = link.bridge_domain.as_deref().unwrap_or("").trim();
                if dom.is_empty() {
                    anyhow::bail!(
                        "topologyLinks '{id}' with linkIntent 'transparent-bridge' must specify a non-empty bridgeDomain"
                    );
                }
            }
            "routed-p2p" | "isolated-fabric" => {
                if link.bridge_domain.is_some() {
                    anyhow::bail!(
                        "topologyLinks '{id}' with linkIntent '{}' must have bridgeDomain: null (cannot join L2 bridge)",
                        link.link_intent
                    );
                }
            }
            other => {
                anyhow::bail!(
                    "Unsupported topologyLinks linkIntent '{other}' on '{id}'; expected transparent-bridge, routed-p2p, or isolated-fabric"
                );
            }
        }
        let link_none = link
            .downstream_link_status
            .as_deref()
            .is_some_and(|s| s.trim().eq_ignore_ascii_case("none") || s.trim().eq_ignore_ascii_case("down"));
        let xdomain_absent = link.windows_xdomain_present == Some(false);
        if link_none || xdomain_absent {
            if link.lifecycle_state_hint == Some(Usb4LifecycleState::Ready) {
                anyhow::bail!(
                    "topologyLinks '{id}' cannot claim lifecycleStateHint='ready' when downstreamLinkStatus={:?} or windowsXDomainPresent={:?}",
                    link.downstream_link_status,
                    link.windows_xdomain_present
                );
            }
            if link.always_on_transit_capable == Some(true) {
                anyhow::bail!(
                    "topologyLinks '{id}' cannot claim alwaysOnTransitCapable=true when downstreamLinkStatus={:?} or windowsXDomainPresent={:?}",
                    link.downstream_link_status,
                    link.windows_xdomain_present
                );
            }
        }
    }
    detect_l2_bridge_cycles(&reg.topology_links)?;

    Ok(())
}

pub fn local_hostname() -> String {
    if let Ok(name) = env::var("COMPUTERNAME") {
        if !name.trim().is_empty() {
            return name.trim().to_string();
        }
    }
    if let Ok(name) = env::var("HOSTNAME") {
        if !name.trim().is_empty() {
            return name.trim().to_string();
        }
    }
    if let Ok(name) = fs::read_to_string("/etc/hostname") {
        let trimmed = name.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "unknown-host".to_string()
}

pub fn default_registry_candidates(explicit_path: Option<&Path>) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(p) = explicit_path {
        paths.push(p.to_path_buf());
    }
    let home = env::var("USERPROFILE")
        .or_else(|_| env::var("HOME"))
        .unwrap_or_default();
    if !home.is_empty() {
        let home_path = PathBuf::from(&home);
        paths.push(home_path.join(".machine").join("thunderbolt-peers.json"));
        paths.push(home_path.join(".config").join("pcai").join("thunderbolt-peers.json"));
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(bin_dir) = exe.parent() {
            paths.push(bin_dir.join("thunderbolt-peers.json"));
            if let Some(parent) = bin_dir.parent() {
                paths.push(parent.join("Config").join("thunderbolt-peers.json"));
                if let Some(grand) = parent.parent() {
                    paths.push(grand.join("Config").join("thunderbolt-peers.json"));
                    if let Some(root) = grand.parent() {
                        paths.push(root.join("Config").join("thunderbolt-peers.json"));
                    }
                }
            }
        }
    }
    paths.push(PathBuf::from(r"C:\codedev\PC_AI\Config\thunderbolt-peers.json"));
    paths
}

pub fn load_peer_registry(explicit_path: Option<&Path>) -> Result<(Option<PeerRegistryFile>, Option<String>)> {
    if let Some(explicit) = explicit_path {
        if explicit.is_file() {
            let raw = fs::read_to_string(explicit)
                .with_context(|| format!("Failed to read registry file '{}'", explicit.display()))?;
            let parsed: PeerRegistryFile = serde_json::from_str(&raw)
                .with_context(|| format!("Invalid Thunderbolt peer registry JSON at {}", explicit.display()))?;
            let validated = parsed
                .normalize_and_validate()
                .with_context(|| format!("Registry validation failed for '{}'", explicit.display()))?;
            return Ok((Some(validated), Some(explicit.to_string_lossy().into_owned())));
        }
        anyhow::bail!(
            "Explicit registry does not exist or is not a regular file: {}",
            explicit.display()
        );
    }

    for candidate in default_registry_candidates(None) {
        if candidate.is_file() {
            let raw = fs::read_to_string(&candidate)
                .with_context(|| format!("Failed to read registry file '{}'", candidate.display()))?;
            let parsed: PeerRegistryFile = serde_json::from_str(&raw)
                .with_context(|| format!("Invalid Thunderbolt peer registry JSON at {}", candidate.display()))?;
            let validated = parsed
                .normalize_and_validate()
                .with_context(|| format!("Registry validation failed for '{}'", candidate.display()))?;
            return Ok((Some(validated), Some(candidate.to_string_lossy().into_owned())));
        }
    }
    Ok((None, None))
}

pub fn load_peer_registry_checked(explicit_path: Option<&Path>) -> Result<(Option<PeerRegistryFile>, Option<String>)> {
    load_peer_registry(explicit_path)
}

fn strip_domain_suffixes(raw: &str) -> String {
    let mut s = raw.trim().to_ascii_lowercase();
    for suffix in &[".radius.dtmventures.com", ".dtmventures.com", ".local", ".lan"] {
        if let Some(stripped) = s.strip_suffix(suffix) {
            s = stripped.to_string();
            break;
        }
    }
    if let Some(stripped) = s.strip_suffix("-tb") {
        s = stripped.to_string();
    }
    s
}

pub fn canonicalize_peer_name(raw_name: &str, known_peers: &[PeerRegistryEntry]) -> String {
    let cleaned = strip_domain_suffixes(if raw_name.trim().is_empty() {
        "unknown-host"
    } else {
        raw_name
    });
    for peer in known_peers {
        if peer.name.eq_ignore_ascii_case(&cleaned) {
            return peer.name.to_ascii_lowercase();
        }
        for alias in &peer.aliases {
            if strip_domain_suffixes(alias) == cleaned {
                return peer.name.to_ascii_lowercase();
            }
        }
    }
    match cleaned.as_str() {
        "dtm-supernuc" | "super-nuc" | "supernuc" => "dtm-super-nuc".to_string(),
        "minisforum-ms01" | "ms01" => "minisforum".to_string(),
        "carbon-two" => "dtm-carbon-two".to_string(),
        "p1gen7" => "dtm-p1gen7".to_string(),
        "p1gen9" => "dtm-p1gen9".to_string(),
        _ => cleaned,
    }
}

pub fn compute_deterministic_node_hextets(canonical_name: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("pc-ai:usb4-peer:v1:{canonical_name}").as_bytes());
    let digest = hasher.finalize();

    let mut h0 = u16::from_be_bytes([digest[0], digest[1]]);
    let mut h1 = u16::from_be_bytes([digest[2], digest[3]]);
    let mut h2 = u16::from_be_bytes([digest[4], digest[5]]);
    if h0 == 0 {
        h0 = 1;
    }
    if h1 == 0 {
        h1 = 1;
    }
    if h2 == 0 {
        h2 = 1;
    }
    format!("{h0:04x}:{h1:04x}:{h2:04x}")
}

fn format_ipv4_u32(val: u32) -> String {
    format!(
        "{}.{}.{}.{}",
        (val >> 24) & 0xFF,
        (val >> 16) & 0xFF,
        (val >> 8) & 0xFF,
        val & 0xFF
    )
}

pub fn compute_commutative_pairwise_ipv4(local_canonical: &str, peer_canonical: &str) -> Result<PairwiseIpv4Contract> {
    if local_canonical == peer_canonical {
        anyhow::bail!(
            "LocalComputer ('{local_canonical}') and PeerComputer ('{peer_canonical}') must be distinct hosts to derive a pairwise /30 link."
        );
    }
    let (low, high) = if local_canonical < peer_canonical {
        (local_canonical, peer_canonical)
    } else {
        (peer_canonical, local_canonical)
    };

    let mut hasher = Sha256::new();
    hasher.update(format!("pc-ai:usb4-ipv4-pair:v1:{low}|{high}").as_bytes());
    let digest = hasher.finalize();

    let raw_slot = u32::from(u16::from_be_bytes([digest[0], digest[1]])) % 1022 + 1;
    let base_ip_int: u32 = (172 << 24) | (31 << 16) | (240 << 8);
    let subnet_int = base_ip_int + (raw_slot * 4);
    let low_ip_int = subnet_int + 1;
    let high_ip_int = subnet_int + 2;

    let subnet_str = format_ipv4_u32(subnet_int);
    let low_ip_str = format_ipv4_u32(low_ip_int);
    let high_ip_str = format_ipv4_u32(high_ip_int);

    let is_local_low = local_canonical == low;
    Ok(PairwiseIpv4Contract {
        pair_key: format!("{low}<->{high}"),
        subnet_prefix: format!("{subnet_str}/30"),
        prefix_length: 30,
        local_ipv4_address: if is_local_low {
            low_ip_str.clone()
        } else {
            high_ip_str.clone()
        },
        peer_ipv4_address: if is_local_low { high_ip_str } else { low_ip_str },
    })
}

pub fn resolve_peer_identity(
    computer_name: Option<&str>,
    peer_computer_name: Option<&str>,
    registry_path: Option<&Path>,
) -> Result<ThunderboltPeerIdentity> {
    let raw_input = computer_name.map(str::to_string).unwrap_or_else(local_hostname);
    let (reg_opt, loaded_path) = load_peer_registry_checked(registry_path)?;

    let known_peers = reg_opt.as_ref().map(|r| r.fleet_peers.as_slice()).unwrap_or(&[]);
    let ula_prefix = reg_opt
        .as_ref()
        .and_then(|r| r.ula_prefix.as_deref())
        .unwrap_or("fd00:7462:7032");

    let canonical_name = canonicalize_peer_name(&raw_input, known_peers);
    let matched_peer = known_peers
        .iter()
        .find(|p| p.name.eq_ignore_ascii_case(&canonical_name));

    let hextets = compute_deterministic_node_hextets(&canonical_name);
    let det_link_local = format!("fe80::d700:{hextets}");
    let det_ula = format!("{ula_prefix}:d700::{hextets}");

    let short_link_local = matched_peer.and_then(|p| p.short_link_local.clone());
    let short_ula = matched_peer.and_then(|p| p.short_ula.clone());

    let primary_link_local = short_link_local.clone().unwrap_or_else(|| det_link_local.clone());
    let primary_ula = short_ula.clone().unwrap_or_else(|| det_ula.clone());

    let mut all_lla = Vec::new();
    if let Some(ref s) = short_link_local {
        all_lla.push(s.clone());
    }
    if !all_lla.contains(&det_link_local) {
        all_lla.push(det_link_local.clone());
    }

    let mut all_ula = Vec::new();
    if let Some(ref s) = short_ula {
        all_ula.push(s.clone());
    }
    if !all_ula.contains(&det_ula) {
        all_ula.push(det_ula.clone());
    }

    let (canonical_peer_opt, pairwise_opt) = if let Some(peer_raw) = peer_computer_name {
        if !peer_raw.trim().is_empty() {
            let canon_peer = canonicalize_peer_name(peer_raw, known_peers);
            let pair = compute_commutative_pairwise_ipv4(&canonical_name, &canon_peer)?;
            (Some(canon_peer), Some(pair))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    Ok(ThunderboltPeerIdentity {
        computer_name: canonical_name.clone(),
        input_name: raw_input,
        os_family: matched_peer
            .and_then(|p| p.os_family.clone())
            .unwrap_or_else(|| "windows".to_string()),
        is_known_fleet_peer: matched_peer.is_some(),
        node_token_hextets: hextets,
        deterministic_link_local: det_link_local,
        deterministic_ula: det_ula,
        short_link_local,
        short_ula,
        primary_link_local,
        primary_ula,
        all_link_local_addresses: all_lla,
        all_ula_addresses: all_ula,
        tb_host_alias: matched_peer
            .and_then(|p| p.tb_host_alias.clone())
            .unwrap_or_else(|| format!("{canonical_name}-tb")),
        ssh_port: matched_peer.and_then(|p| p.ssh_port).unwrap_or(22),
        ssh_user: matched_peer
            .and_then(|p| p.ssh_user.clone())
            .unwrap_or_else(|| "david".to_string()),
        win_rm_port: matched_peer.and_then(|p| p.win_rm_port).unwrap_or(5985),
        lan_addresses: matched_peer.map(|p| p.lan_addresses.clone()).unwrap_or_default(),
        default_shares: matched_peer
            .map(|p| p.default_shares.clone())
            .unwrap_or_else(|| vec!["C$".to_string()]),
        peer_computer_name: canonical_peer_opt,
        pairwise_ipv4: pairwise_opt,
        registry_path: loaded_path,
    })
}

pub fn known_fleet_peer_names(explicit_registry: Option<&Path>, exclude_host: &str) -> Result<Vec<String>> {
    let mut names = vec![
        "dtm-p1gen7".to_string(),
        "dtm-p1gen9".to_string(),
        "dtm-carbon-two".to_string(),
        "dtm-super-nuc".to_string(),
        "dtm-work".to_string(),
        "dtm-work-mobo".to_string(),
        "minisforum".to_string(),
        "asuspro13".to_string(),
        "millylaptop1".to_string(),
    ];
    let (reg_opt, _) = load_peer_registry(explicit_registry)?;
    if let Some(reg) = reg_opt {
        for p in reg.fleet_peers {
            if !names.iter().any(|n| n.eq_ignore_ascii_case(&p.name)) {
                names.push(p.name);
            }
        }
    }
    Ok(names
        .into_iter()
        .filter(|n| !n.eq_ignore_ascii_case(exclude_host))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_identity_arbitrary_hosts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let reg_path = dir.path().join("empty.json");
        fs::write(&reg_path, r#"{"schemaVersion":1,"fleetPeers":[]}"#).unwrap();

        let a = resolve_peer_identity(Some("dtm-carbon-two"), None, Some(&reg_path)).expect("resolve carbon-two");
        let b = resolve_peer_identity(Some("dtm-supernuc"), None, Some(&reg_path)).expect("resolve supernuc");
        assert_eq!(a.computer_name, "dtm-carbon-two");
        assert_eq!(b.computer_name, "dtm-super-nuc");
        assert_ne!(a.deterministic_ula, b.deterministic_ula);
        assert!(a.deterministic_link_local.starts_with("fe80::d700:"));
        assert!(a.deterministic_ula.starts_with("fd00:7462:7032:d700::"));
        assert!(a.deterministic_link_local.parse::<std::net::Ipv6Addr>().is_ok());
        assert!(a.deterministic_ula.parse::<std::net::Ipv6Addr>().is_ok());

        assert!(resolve_peer_identity(Some("dtm-carbon-two"), None, Some(Path::new("/nonexistent"))).is_err());
    }

    #[test]
    fn test_commutative_pairwise_ipv4_symmetry_and_disjointness() {
        let c2_to_nuc = compute_commutative_pairwise_ipv4("dtm-carbon-two", "dtm-super-nuc").expect("c2 -> nuc");
        let nuc_to_c2 = compute_commutative_pairwise_ipv4("dtm-super-nuc", "dtm-carbon-two").expect("nuc -> c2");
        assert_eq!(c2_to_nuc.subnet_prefix, nuc_to_c2.subnet_prefix);
        assert_eq!(c2_to_nuc.local_ipv4_address, nuc_to_c2.peer_ipv4_address);
        assert_eq!(c2_to_nuc.peer_ipv4_address, nuc_to_c2.local_ipv4_address);

        let p7_to_p9 = compute_commutative_pairwise_ipv4("dtm-p1gen7", "dtm-p1gen9").expect("p7 -> p9");
        let p7_to_c2 = compute_commutative_pairwise_ipv4("dtm-p1gen7", "dtm-carbon-two").expect("p7 -> c2");
        assert_ne!(p7_to_p9.subnet_prefix, p7_to_c2.subnet_prefix);
    }

    #[test]
    fn test_validate_peer_registry_rejects_duplicate_lan_ip_and_duplicate_ula() {
        let bad_lan_json = r#"{
            "schemaVersion": 1,
            "fleetPeers": [
                {
                    "name": "dtm-p1gen7",
                    "shortLinkLocal": "fe80::d700:7",
                    "shortUla": "fd00:7462:7032::7",
                    "lanAddresses": ["100.64.0.3"]
                },
                {
                    "name": "asuspro13",
                    "shortLinkLocal": "fe80::d700:13",
                    "shortUla": "fd00:7462:7032::13",
                    "lanAddresses": ["100.64.0.3"]
                }
            ]
        }"#;
        let parsed: PeerRegistryFile = serde_json::from_str(bad_lan_json).unwrap();
        let err = parsed.normalize_and_validate().unwrap_err();
        assert!(
            err.to_string().contains("Duplicate LAN/Tailscale IP '100.64.0.3'"),
            "unexpected error: {err}"
        );

        let bad_ula_json = r#"{
            "schemaVersion": 1,
            "fleetPeers": [
                {
                    "name": "host-a",
                    "shortUla": "fd00:7462:7032::7"
                },
                {
                    "name": "host-b",
                    "shortUla": "fd00:7462:7032::7"
                }
            ]
        }"#;
        let parsed_ula: PeerRegistryFile = serde_json::from_str(bad_ula_json).unwrap();
        let err_ula = parsed_ula.normalize_and_validate().unwrap_err();
        assert!(
            err_ula.to_string().contains("Duplicate shortUla"),
            "unexpected error: {err_ula}"
        );
    }

    #[test]
    fn test_validate_peer_registry_rejects_overlapping_linux_30_subnets() {
        let bad_peers_json = r#"{
            "schemaVersion": 1,
            "peers": {
                "peerA": {
                    "localAddress": "172.31.240.1",
                    "peerAddress": "172.31.240.2",
                    "prefixLength": 30
                },
                "peerB": {
                    "localAddress": "172.31.240.2",
                    "peerAddress": "172.31.240.1",
                    "prefixLength": 30
                }
            },
            "fleetPeers": []
        }"#;
        let parsed: PeerRegistryFile = serde_json::from_str(bad_peers_json).unwrap();
        let err = parsed.normalize_and_validate().unwrap_err();
        assert!(
            err.to_string().contains("overlaps with another peer profile"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_validate_peer_registry_ipam_and_sfp_topology_invariants() {
        let overlapping_ula_json = r#"{
            "schemaVersion": 1,
            "ipamAllocations": {
                "p2pThunderboltUlaPrefix": "fd00:7462:7032::/48",
                "bridgedLanCanaryUlaPrefix": "fd00:7462:7032:50::/64",
                "bridgedLanPolicyTable": 6150,
                "bridgedLanCanaryEndpoints": []
            },
            "fleetPeers": []
        }"#;
        let parsed: PeerRegistryFile = serde_json::from_str(overlapping_ula_json).unwrap();
        let err = parsed.normalize_and_validate().unwrap_err();
        assert!(
            err.to_string().contains("ipamAllocations ULA prefixes overlap"),
            "unexpected error: {err}"
        );

        let reserved_table_json = r#"{
            "schemaVersion": 1,
            "ipamAllocations": {
                "p2pThunderboltUlaPrefix": "fd00:7462:7032::/48",
                "bridgedLanCanaryUlaPrefix": "fd5f:743d:6f55:50::/64",
                "bridgedLanPolicyTable": 254,
                "bridgedLanCanaryEndpoints": []
            },
            "fleetPeers": []
        }"#;
        let parsed_tbl: PeerRegistryFile = serde_json::from_str(reserved_table_json).unwrap();
        let err_tbl = parsed_tbl.normalize_and_validate().unwrap_err();
        assert!(
            err_tbl.to_string().contains("reserved Linux routing tables"),
            "unexpected error: {err_tbl}"
        );

        let dup_sfp_mac_json = r#"{
            "schemaVersion": 1,
            "sfpSwitchTopologies": [
                {
                    "host": "minisforum",
                    "namespace": "ms01-switch",
                    "bridge": "sw0",
                    "stateDir": "/var/lib/ms01-network-switch",
                    "sfpPorts": [
                        {
                            "name": "enp2s0f0np0",
                            "pci": "0000:02:00.0",
                            "mac": "58:47:ca:7b:6d:98",
                            "vendor": "0x8086",
                            "device": "0x1572",
                            "driver": "i40e",
                            "role": "uplink"
                        },
                        {
                            "name": "enp2s0f1np1",
                            "pci": "0000:02:00.1",
                            "mac": "58:47:CA:7B:6D:98",
                            "vendor": "0x8086",
                            "device": "0x1572",
                            "driver": "i40e",
                            "role": "downlink"
                        }
                    ]
                }
            ],
            "fleetPeers": []
        }"#;
        let parsed_sfp: PeerRegistryFile = serde_json::from_str(dup_sfp_mac_json).unwrap();
        let err_sfp = parsed_sfp.normalize_and_validate().unwrap_err();
        assert!(
            err_sfp.to_string().contains("Duplicate SFP+ port MAC"),
            "unexpected error: {err_sfp}"
        );
    }

    // trace: PENDING-FABRIC-F09 [pending-tpm-intake]
    #[test]
    fn test_validate_peer_registry_rejects_l2_bridge_cycles_and_overlapping_shared_subnets() {
        let cycle_json = r#"{
            "schemaVersion": 1,
            "topologyLinks": [
                {
                    "linkId": "ms01-gen9-sfp",
                    "sourceHost": "minisforum",
                    "sourceInterface": "enp2s0f1np1",
                    "targetHost": "dtm-p1gen9",
                    "targetInterface": "Ethernet 2",
                    "linkIntent": "transparent-bridge",
                    "bridgeDomain": "ms01-switch/sw0"
                },
                {
                    "linkId": "ms01-gen9-usb4-loop",
                    "sourceHost": "minisforum",
                    "sourceInterface": "domain1/1-1",
                    "targetHost": "dtm-p1gen9",
                    "targetInterface": "Ethernet 4",
                    "linkIntent": "transparent-bridge",
                    "bridgeDomain": "ms01-switch/sw0"
                }
            ],
            "fleetPeers": []
        }"#;
        let parsed_cycle: PeerRegistryFile = serde_json::from_str(cycle_json).unwrap();
        let err_cycle = parsed_cycle.normalize_and_validate().unwrap_err();
        assert!(
            err_cycle.to_string().contains("L2 transparent-bridge cycle detected"),
            "expected L2 bridge cycle error, got: {err_cycle}"
        );
        assert!(
            !err_cycle.to_string().contains("STP is disabled"),
            "error message should not hardcode STP state: {err_cycle}"
        );

        // Direct detect_l2_bridge_cycles negative controls: missing bridge_domain or empty interface must fail closed
        let missing_domain = vec![TopologyLinkSpec {
            link_id: "link-no-domain".to_string(),
            source_host: "minisforum".to_string(),
            source_interface: "enp2s0f1np1".to_string(),
            target_host: "dtm-p1gen9".to_string(),
            target_interface: "Ethernet 2".to_string(),
            link_intent: "transparent-bridge".to_string(),
            bridge_domain: None,
            ..Default::default()
        }];
        assert!(detect_l2_bridge_cycles(&missing_domain).is_err());

        let empty_iface = vec![TopologyLinkSpec {
            link_id: "link-empty-iface".to_string(),
            source_host: "minisforum".to_string(),
            source_interface: "   ".to_string(),
            target_host: "dtm-p1gen9".to_string(),
            target_interface: "Ethernet 2".to_string(),
            link_intent: "transparent-bridge".to_string(),
            bridge_domain: Some("ms01-switch/sw0".to_string()),
            ..Default::default()
        }];
        assert!(detect_l2_bridge_cycles(&empty_iface).is_err());

        // Changing the second link to routed-p2p resolves the L2 loop
        let routed_json = r#"{
            "schemaVersion": 1,
            "topologyLinks": [
                {
                    "linkId": "ms01-gen9-sfp",
                    "sourceHost": "minisforum",
                    "sourceInterface": "enp2s0f1np1",
                    "targetHost": "dtm-p1gen9",
                    "targetInterface": "Ethernet 2",
                    "linkIntent": "transparent-bridge",
                    "bridgeDomain": "ms01-switch/sw0"
                },
                {
                    "linkId": "ms01-gen9-usb4-routed",
                    "sourceHost": "minisforum",
                    "sourceInterface": "domain1/1-1",
                    "targetHost": "dtm-p1gen9",
                    "targetInterface": "Ethernet 4",
                    "linkIntent": "routed-p2p",
                    "bridgeDomain": null
                }
            ],
            "fleetPeers": []
        }"#;
        let parsed_routed: PeerRegistryFile = serde_json::from_str(routed_json).unwrap();
        assert!(parsed_routed.normalize_and_validate().is_ok());

        let overlap_shared_json = r#"{
            "schemaVersion": 1,
            "ipamAllocations": {
                "p2pThunderboltUlaPrefix": "fd00:7462:7032::/48",
                "bridgedLanCanaryUlaPrefix": "fd5f:743d:6f55:50::/64",
                "bridgedLanCanaryEndpoints": [],
                "sharedSwitchSubnets": [
                    {
                        "subnet": "10.254.253.0/30",
                        "gatewayHost": "minisforum",
                        "gatewayIpv4": "192.168.50.119",
                        "role": "usb4-ctrl-a"
                    },
                    {
                        "subnet": "10.254.253.0/29",
                        "gatewayHost": "minisforum",
                        "gatewayIpv4": "192.168.50.119",
                        "role": "usb4-ctrl-b"
                    }
                ]
            },
            "fleetPeers": []
        }"#;
        let parsed_shared: PeerRegistryFile = serde_json::from_str(overlap_shared_json).unwrap();
        let err_shared = parsed_shared.normalize_and_validate().unwrap_err();
        assert!(
            err_shared.to_string().contains("sharedSwitchSubnets") && err_shared.to_string().contains("overlaps"),
            "expected sharedSwitchSubnets overlap error, got: {err_shared}"
        );
    }

    #[test]
    fn test_validate_peer_registry_enforces_usb4_lifecycle_and_link_none_constraints() {
        let false_ready_json = r#"{
            "schemaVersion": 1,
            "topologyLinks": [
                {
                    "linkId": "ms01-plugable-dock-usb4-routed",
                    "sourceHost": "minisforum",
                    "sourceInterface": "domain1/1-1",
                    "targetHost": "dtm-p1gen9",
                    "targetInterface": "Ethernet 4",
                    "linkIntent": "routed-p2p",
                    "bridgeDomain": null,
                    "lifecycleStateHint": "ready",
                    "downstreamLinkStatus": "none",
                    "windowsXDomainPresent": false
                }
            ],
            "fleetPeers": []
        }"#;
        let parsed_false_ready: PeerRegistryFile = serde_json::from_str(false_ready_json).unwrap();
        let err_ready = parsed_false_ready.normalize_and_validate().unwrap_err();
        assert!(
            err_ready.to_string().contains("cannot claim lifecycleStateHint='ready'"),
            "expected false ready rejection, got: {err_ready}"
        );

        let false_transit_json = r#"{
            "schemaVersion": 1,
            "usb4LifecyclePolicy": {
                "states": ["asleep", "disconnected", "probing", "ready"],
                "defaultEvidenceTtlSeconds": 300,
                "boundedResumeReprobeMaxAttempts": 3,
                "boundedResumeReprobeTimeoutMs": 250,
                "boundedResumeTotalBudgetMs": 1500,
                "withdrawOwnedStaleRoutesOnExpireOrDisconnect": true,
                "requireIdentityBoundCanary": true,
                "allowAlwaysOnTransitClaimWhenSelectiveSuspend": true
            },
            "fleetPeers": []
        }"#;
        let parsed_false_transit: PeerRegistryFile = serde_json::from_str(false_transit_json).unwrap();
        let err_transit = parsed_false_transit.normalize_and_validate().unwrap_err();
        assert!(
            err_transit
                .to_string()
                .contains("allowAlwaysOnTransitClaimWhenSelectiveSuspend must be false"),
            "expected false always-on transit policy rejection, got: {err_transit}"
        );
    }
}
