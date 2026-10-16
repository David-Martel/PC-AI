use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::{Path, PathBuf};

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
    pub peers: Option<serde_json::Value>,
    #[serde(default)]
    pub fleet_peers: Vec<PeerRegistryEntry>,
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
    let home = env::var("USERPROFILE")
        .or_else(|_| env::var("HOME"))
        .unwrap_or_default();
    if !home.is_empty() {
        let home_path = PathBuf::from(&home);
        paths.push(home_path.join(".machine").join("thunderbolt-peers.json"));
        paths.push(home_path.join(".config").join("pcai").join("thunderbolt-peers.json"));
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
                .with_context(|| format!("Failed to parse registry JSON '{}'", explicit.display()))?;
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
                .with_context(|| format!("Failed to parse registry JSON '{}'", candidate.display()))?;
            let validated = parsed
                .normalize_and_validate()
                .with_context(|| format!("Registry validation failed for '{}'", candidate.display()))?;
            return Ok((Some(validated), Some(candidate.to_string_lossy().into_owned())));
        }
    }
    Ok((None, None))
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
    let (reg_opt, loaded_path) = load_peer_registry(registry_path)?;

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
        let a = resolve_peer_identity(Some("dtm-carbon-two"), None, None).expect("resolve carbon-two");
        let b = resolve_peer_identity(Some("dtm-supernuc"), None, None).expect("resolve supernuc");
        assert_eq!(a.computer_name, "dtm-carbon-two");
        assert_eq!(b.computer_name, "dtm-super-nuc");
        assert_ne!(a.deterministic_ula, b.deterministic_ula);
        assert!(a.deterministic_link_local.starts_with("fe80::d700:"));
        assert!(a.deterministic_ula.starts_with("fd00:7462:7032:d700::"));
        assert!(a.deterministic_link_local.parse::<std::net::Ipv6Addr>().is_ok());
        assert!(a.deterministic_ula.parse::<std::net::Ipv6Addr>().is_ok());
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
}
