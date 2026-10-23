use crate::identity::{
    canonicalize_peer_name, compute_commutative_pairwise_ipv4, known_fleet_peer_names, load_peer_registry_checked,
    local_hostname, resolve_peer_identity, PairwiseIpv4Contract,
};
use crate::platform::{collect_platform_snapshot, ThunderboltAdapterState};
use anyhow::Result;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

/// Minimal NTLMSSP Negotiate (Type 1) token requesting TargetInfo + Unicode
pub const NTLM_NEGOTIATE_TYPE1_B64: &str = "TlRMTVNTUAABAAAAB4IIogAAAAAAAAAAAAAAAAAAAAAGAbEdAAAADw==";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NtlmTargetIdentity {
    pub netbios_computer_name: Option<String>,
    pub dns_computer_name: Option<String>,
    pub netbios_domain_name: Option<String>,
    pub dns_domain_name: Option<String>,
}

impl NtlmTargetIdentity {
    pub fn best_hostname(&self) -> Option<String> {
        if let Some(ref dns) = self.dns_computer_name {
            if !dns.trim().is_empty() {
                return Some(dns.trim().to_string());
            }
        }
        if let Some(ref nb) = self.netbios_computer_name {
            if !nb.trim().is_empty() {
                return Some(nb.trim().to_string());
            }
        }
        None
    }
}

/// Decodes a binary NTLMSSP Type-2 Challenge message and extracts `AV_PAIR` TargetInfo fields.
pub fn parse_ntlm_challenge_bytes(bytes: &[u8]) -> Option<NtlmTargetIdentity> {
    if bytes.len() < 48 {
        return None;
    }
    if &bytes[0..8] != b"NTLMSSP\0" {
        return None;
    }
    let msg_type = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if msg_type != 2 {
        return None;
    }

    let target_info_len = u16::from_le_bytes([bytes[40], bytes[41]]) as usize;
    let target_info_offset = u32::from_le_bytes([bytes[44], bytes[45], bytes[46], bytes[47]]) as usize;

    if target_info_len == 0 || target_info_offset + target_info_len > bytes.len() {
        return None;
    }

    let mut pos = target_info_offset;
    let end = target_info_offset + target_info_len;
    let mut identity = NtlmTargetIdentity {
        netbios_computer_name: None,
        dns_computer_name: None,
        netbios_domain_name: None,
        dns_domain_name: None,
    };

    while pos + 4 <= end {
        let av_id = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]);
        let av_len = u16::from_le_bytes([bytes[pos + 2], bytes[pos + 3]]) as usize;
        pos += 4;
        if av_id == 0 {
            // MsvAvEOL
            break;
        }
        if pos + av_len > end {
            break;
        }
        if av_len.is_multiple_of(2) {
            let u16_slice: Vec<u16> = bytes[pos..pos + av_len]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let decoded = String::from_utf16_lossy(&u16_slice);
            match av_id {
                1 => identity.netbios_computer_name = Some(decoded), // MsvAvNbComputerName
                2 => identity.netbios_domain_name = Some(decoded),   // MsvAvNbDomainName
                3 => identity.dns_computer_name = Some(decoded),     // MsvAvDnsComputerName
                4 => identity.dns_domain_name = Some(decoded),       // MsvAvDnsDomainName
                _ => {}
            }
        }
        pos += av_len;
    }

    if identity.netbios_computer_name.is_some() || identity.dns_computer_name.is_some() {
        Some(identity)
    } else {
        None
    }
}

pub fn parse_ntlm_challenge_base64(b64_token: &str) -> Option<NtlmTargetIdentity> {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(b64_token.trim())
        .ok()?;
    parse_ntlm_challenge_bytes(&raw)
}

fn resolve_scope_id(scope_str: &str) -> Option<u32> {
    if scope_str.is_empty() {
        return None;
    }
    if let Ok(num) = scope_str.parse::<u32>() {
        return if num > 0 { Some(num) } else { None };
    }
    #[cfg(unix)]
    {
        if let Ok(c_name) = std::ffi::CString::new(scope_str) {
            let idx = unsafe { libc::if_nametoindex(c_name.as_ptr()) };
            if idx > 0 {
                return Some(idx);
            }
        }
    }
    None
}

pub fn parse_sockaddr_with_scope(addr_str: &str, port: u16) -> Option<SocketAddr> {
    let trimmed = addr_str.trim().trim_start_matches('[').trim_end_matches(']');
    if trimmed.is_empty() {
        return None;
    }
    if let Some((ip_part, scope_part)) = trimmed.split_once('%') {
        let ipv6 = ip_part.parse::<Ipv6Addr>().ok()?;
        let scope_id = resolve_scope_id(scope_part)?;
        return Some(SocketAddr::V6(SocketAddrV6::new(ipv6, port, 0, scope_id)));
    }
    if let Ok(ipv4) = trimmed.parse::<Ipv4Addr>() {
        return Some(SocketAddr::V4(SocketAddrV4::new(ipv4, port)));
    }
    if let Ok(ipv6) = trimmed.parse::<Ipv6Addr>() {
        return Some(SocketAddr::V6(SocketAddrV6::new(ipv6, port, 0, 0)));
    }
    if let Ok(ip) = trimmed.parse::<IpAddr>() {
        return Some(SocketAddr::new(ip, port));
    }
    None
}

pub fn peer_ipv4_in_slash30(local_ip_str: &str) -> Option<String> {
    let ip: Ipv4Addr = local_ip_str.trim().parse().ok()?;
    let raw = u32::from(ip);
    let host_bits = raw & 3;
    if host_bits != 1 && host_bits != 2 {
        return None;
    }
    let peer_raw = (raw & 0xFFFF_FFFC) | (host_bits ^ 3);
    Some(Ipv4Addr::from(peer_raw).to_string())
}

pub async fn probe_ssh_banner(addr_str: &str, port: u16, timeout_ms: u64) -> (bool, Option<String>, u64) {
    let start = Instant::now();
    let Some(sockaddr) = parse_sockaddr_with_scope(addr_str, port) else {
        return (false, None, 0);
    };
    let dur = Duration::from_millis(timeout_ms);
    let Ok(Ok(mut stream)) = timeout(dur, TcpStream::connect(sockaddr)).await else {
        return (false, None, start.elapsed().as_micros() as u64);
    };
    let mut buf = [0u8; 128];
    let banner = match timeout(Duration::from_millis(timeout_ms.min(250)), stream.read(&mut buf)).await {
        Ok(Ok(n)) if n > 0 => {
            let s = String::from_utf8_lossy(&buf[..n]);
            let line = s.lines().next().unwrap_or("").trim().to_string();
            if line.is_empty() {
                None
            } else {
                Some(line)
            }
        }
        _ => None,
    };
    (true, banner, start.elapsed().as_micros() as u64)
}

pub async fn probe_winrm_ntlm_identity(
    addr_str: &str,
    port: u16,
    timeout_ms: u64,
) -> (bool, Option<NtlmTargetIdentity>, u64) {
    let start = Instant::now();
    let Some(sockaddr) = parse_sockaddr_with_scope(addr_str, port) else {
        return (false, None, 0);
    };
    let dur = Duration::from_millis(timeout_ms);
    let Ok(Ok(mut stream)) = timeout(dur, TcpStream::connect(sockaddr)).await else {
        return (false, None, start.elapsed().as_micros() as u64);
    };

    let host_hdr = if addr_str.contains(':') {
        let base = addr_str.split('%').next().unwrap_or(addr_str);
        format!("[{base}]:{port}")
    } else {
        format!("{addr_str}:{port}")
    };
    let req = format!(
        "POST /wsman HTTP/1.1\r\nHost: {host_hdr}\r\nAuthorization: Negotiate {NTLM_NEGOTIATE_TYPE1_B64}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    );

    if timeout(dur, stream.write_all(req.as_bytes())).await.is_err() {
        return (true, None, start.elapsed().as_micros() as u64);
    }

    let mut resp_buf = vec![0u8; 4096];
    let Ok(Ok(n)) = timeout(dur, stream.read(&mut resp_buf)).await else {
        return (true, None, start.elapsed().as_micros() as u64);
    };
    if n == 0 {
        return (true, None, start.elapsed().as_micros() as u64);
    }

    let resp_text = String::from_utf8_lossy(&resp_buf[..n]);
    for line in resp_text.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("www-authenticate:") {
            let val = line["www-authenticate:".len()..].trim();
            for prefix in &["Negotiate ", "NTLM ", "negotiate ", "ntlm "] {
                if let Some(rest) = val.strip_prefix(prefix) {
                    let b64_part = rest.split(',').next().unwrap_or(rest).trim();
                    if let Some(id) = parse_ntlm_challenge_base64(b64_part) {
                        return (true, Some(id), start.elapsed().as_micros() as u64);
                    }
                }
            }
        }
    }

    (true, None, start.elapsed().as_micros() as u64)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct DiscoveredThunderboltPeer {
    pub local_computer: String,
    pub peer_name: Option<String>,
    pub is_known_fleet_peer: bool,
    pub interface_index: u32,
    pub interface_alias: String,
    pub interface_description: String,
    pub link_speed_gbps: f64,
    pub candidate_address: String,
    pub address_family: String,
    pub scope_type: String,
    pub discovery_source: String,
    pub neighbor_state: String,
    pub link_layer_address: Option<String>,
    pub ssh_open: bool,
    pub ssh_banner: Option<String>,
    pub win_rm_open: bool,
    pub ntlm_computer_name: Option<String>,
    pub ntlm_dns_name: Option<String>,
    pub verified_handshake: bool,
    pub handshake_method: String,
    pub probe_latency_us: u64,
    pub preferred_transport_address: String,
    pub expected_peer_ula: Option<String>,
    pub expected_peer_link_local: Option<String>,
    pub pairwise_ipv4: Option<PairwiseIpv4Contract>,
}

#[derive(Debug, Clone)]
pub(crate) struct CandidateWorkItem {
    pub interface_index: u32,
    pub interface_alias: String,
    pub interface_description: String,
    pub link_speed_gbps: f64,
    pub connect_target: String,
    pub address_family: String,
    pub scope_type: String,
    pub source: String,
    pub neighbor_state: String,
    pub mac: Option<String>,
    pub expected_peer_hint: Option<String>,
    pub ssh_port: u16,
    pub winrm_port: u16,
}

fn gather_candidates_for_adapter(
    adapter: &ThunderboltAdapterState,
    local_canon: &str,
    known_entries: &[crate::identity::PeerRegistryEntry],
    registry_path: Option<&Path>,
) -> Result<Vec<CandidateWorkItem>> {
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    let if_idx = adapter.interface_index;

    let xdomain_hint = if adapter.xdomain_peer_unique_id.is_some() {
        adapter
            .xdomain_peer_device_name
            .as_deref()
            .filter(|n| !n.trim().is_empty())
            .map(|n| canonicalize_peer_name(n, known_entries))
    } else {
        None
    };

    let slash30_peers: HashSet<String> = adapter
        .addresses
        .iter()
        .filter(|a| a.address_family == "IPv4" && a.prefix_length == 30)
        .filter_map(|a| peer_ipv4_in_slash30(&a.address))
        .collect();

    // 1. Active Neighbor Table entries (NDP / ARP)
    for n in &adapter.neighbors {
        let base_ip = n.ip_address.split('%').next().unwrap_or(&n.ip_address);
        let is_own = adapter
            .addresses
            .iter()
            .any(|a| a.address.eq_ignore_ascii_case(base_ip));
        if is_own {
            continue;
        }
        let is_lla6 = base_ip.to_ascii_lowercase().starts_with("fe80:");
        let is_ula6 = base_ip.to_ascii_lowercase().starts_with("fd") || base_ip.to_ascii_lowercase().starts_with("fc");
        let connect_target = if is_lla6 {
            format!("{base_ip}%{if_idx}")
        } else {
            base_ip.to_string()
        };
        if seen.insert(connect_target.to_ascii_lowercase()) {
            let scope_type = if is_lla6 {
                "IPv6LinkLocal"
            } else if is_ula6 {
                "IPv6ULA"
            } else if base_ip.starts_with("169.254.") {
                "IPv4LinkLocal"
            } else {
                "IPv4Routed"
            };
            // Only bind XDomain hardware peer hint to dedicated /30 P2P neighbors,
            // never to bridged /24 LAN or shared /64 router neighbors.
            let neighbor_hint = if slash30_peers.contains(base_ip) {
                xdomain_hint.clone()
            } else {
                None
            };
            items.push(CandidateWorkItem {
                interface_index: if_idx,
                interface_alias: adapter.interface_alias.clone(),
                interface_description: adapter.interface_description.clone(),
                link_speed_gbps: adapter.link_speed_gbps,
                connect_target,
                address_family: n.address_family.clone(),
                scope_type: scope_type.to_string(),
                source: "NeighborTable".to_string(),
                neighbor_state: n.state.clone(),
                mac: Some(n.link_layer_address.clone()),
                expected_peer_hint: neighbor_hint,
                ssh_port: 22,
                winrm_port: 5985,
            });
        }
    }

    // 1b. Active /30 Point-to-Point Peer IPv4 (e.g. 10.254.253.1/30 <-> 10.254.253.2/30)
    for addr in &adapter.addresses {
        if addr.address_family == "IPv4" && addr.prefix_length == 30 {
            if let Some(peer_v4) = peer_ipv4_in_slash30(&addr.address) {
                if seen.insert(peer_v4.to_ascii_lowercase()) {
                    items.push(CandidateWorkItem {
                        interface_index: if_idx,
                        interface_alias: adapter.interface_alias.clone(),
                        interface_description: adapter.interface_description.clone(),
                        link_speed_gbps: adapter.link_speed_gbps,
                        connect_target: peer_v4,
                        address_family: "IPv4".to_string(),
                        scope_type: "IPv4Slash30".to_string(),
                        source: "ActiveSlash30Peer".to_string(),
                        neighbor_state: "Slash30Peer".to_string(),
                        mac: None,
                        expected_peer_hint: xdomain_hint.clone(),
                        ssh_port: 22,
                        winrm_port: 5985,
                    });
                }
            }
        }
    }

    // 2. Fleet Registry & Deterministic SHA-256 Candidates
    for peer_name in known_fleet_peer_names(registry_path, local_canon)? {
        let Ok(peer_id) = resolve_peer_identity(Some(&peer_name), Some(local_canon), registry_path) else {
            continue;
        };
        for lla in &peer_id.all_link_local_addresses {
            let scoped = format!("{lla}%{if_idx}");
            if seen.insert(scoped.to_ascii_lowercase()) {
                items.push(CandidateWorkItem {
                    interface_index: if_idx,
                    interface_alias: adapter.interface_alias.clone(),
                    interface_description: adapter.interface_description.clone(),
                    link_speed_gbps: adapter.link_speed_gbps,
                    connect_target: scoped,
                    address_family: "IPv6".to_string(),
                    scope_type: "IPv6LinkLocal".to_string(),
                    source: "FleetDeterministicLLA".to_string(),
                    neighbor_state: "DeterministicProbe".to_string(),
                    mac: None,
                    expected_peer_hint: Some(peer_id.computer_name.clone()),
                    ssh_port: peer_id.ssh_port,
                    winrm_port: peer_id.win_rm_port,
                });
            }
        }
        for ula in &peer_id.all_ula_addresses {
            if seen.insert(ula.to_ascii_lowercase()) {
                items.push(CandidateWorkItem {
                    interface_index: if_idx,
                    interface_alias: adapter.interface_alias.clone(),
                    interface_description: adapter.interface_description.clone(),
                    link_speed_gbps: adapter.link_speed_gbps,
                    connect_target: ula.clone(),
                    address_family: "IPv6".to_string(),
                    scope_type: "IPv6ULA".to_string(),
                    source: "FleetDeterministicULA".to_string(),
                    neighbor_state: "DeterministicProbe".to_string(),
                    mac: None,
                    expected_peer_hint: Some(peer_id.computer_name.clone()),
                    ssh_port: peer_id.ssh_port,
                    winrm_port: peer_id.win_rm_port,
                });
            }
        }
        if let Some(ref pair_v4) = peer_id.pairwise_ipv4 {
            let peer_v4 = &pair_v4.local_ipv4_address;
            if seen.insert(peer_v4.to_ascii_lowercase()) {
                items.push(CandidateWorkItem {
                    interface_index: if_idx,
                    interface_alias: adapter.interface_alias.clone(),
                    interface_description: adapter.interface_description.clone(),
                    link_speed_gbps: adapter.link_speed_gbps,
                    connect_target: peer_v4.clone(),
                    address_family: "IPv4".to_string(),
                    scope_type: "IPv4Pairwise30".to_string(),
                    source: "PairwiseIpv4Slot".to_string(),
                    neighbor_state: "DeterministicProbe".to_string(),
                    mac: None,
                    expected_peer_hint: Some(peer_id.computer_name.clone()),
                    ssh_port: peer_id.ssh_port,
                    winrm_port: peer_id.win_rm_port,
                });
            }
        }
    }

    Ok(items)
}

pub(crate) async fn probe_work_batch(
    work_items: Vec<CandidateWorkItem>,
    known_entries: Arc<Vec<crate::identity::PeerRegistryEntry>>,
    local_canon: &str,
    timeout_ms: u64,
    include_unreachable: bool,
    registry_path: Option<&Path>,
) -> Vec<DiscoveredThunderboltPeer> {
    let mut join_set = tokio::task::JoinSet::new();
    let registry_path_buf = registry_path.map(|p| p.to_path_buf());

    for item in work_items {
        let known_clone = Arc::clone(&known_entries);
        let local_clone = local_canon.to_string();
        let reg_buf = registry_path_buf.clone();

        join_set.spawn(async move {
            let probe_start = Instant::now();
            let (ssh_res, winrm_res) = tokio::join!(
                probe_ssh_banner(&item.connect_target, item.ssh_port, timeout_ms),
                probe_winrm_ntlm_identity(&item.connect_target, item.winrm_port, timeout_ms)
            );
            let total_us = probe_start.elapsed().as_micros() as u64;

            let (ssh_open, ssh_banner, _) = ssh_res;
            let (winrm_open, ntlm_id, _) = winrm_res;

            let mut resolved_peer_name: Option<String> = None;
            let mut handshake_method = "None".to_string();

            if let Some(ref id) = ntlm_id {
                if let Some(raw_host) = id.best_hostname() {
                    resolved_peer_name = Some(canonicalize_peer_name(&raw_host, &known_clone));
                    handshake_method = "NTLM-TargetInfo-AVPair".to_string();
                }
            }

            let has_valid_ssh_banner = ssh_open && ssh_banner.as_deref().is_some_and(|b| b.starts_with("SSH-"));

            if resolved_peer_name.is_none() {
                if has_valid_ssh_banner
                    && (item.source == "FleetDeterministicLLA" || item.source == "FleetDeterministicULA")
                    && (item.scope_type == "IPv6LinkLocal" || item.scope_type == "IPv6ULA")
                {
                    if let Some(ref hint) = item.expected_peer_hint {
                        resolved_peer_name = Some(canonicalize_peer_name(hint, &known_clone));
                        handshake_method = "SSH-Banner+DeterministicIPv6".to_string();
                    }
                } else if has_valid_ssh_banner && (item.source == "NeighborTable" || item.source == "ActiveSlash30Peer")
                {
                    if let Some(ref hint) = item.expected_peer_hint {
                        resolved_peer_name = Some(canonicalize_peer_name(hint, &known_clone));
                        handshake_method = "SSH-Banner+XDomainHardwarePeer".to_string();
                    } else {
                        handshake_method = "UnverifiedPortOpen".to_string();
                    }
                } else if ssh_open || winrm_open {
                    handshake_method = "UnverifiedPortOpen".to_string();
                }
            }

            let verified = resolved_peer_name
                .as_ref()
                .is_some_and(|p| !p.eq_ignore_ascii_case(&local_clone))
                && (handshake_method == "NTLM-TargetInfo-AVPair"
                    || handshake_method == "SSH-Banner+DeterministicIPv6"
                    || handshake_method == "SSH-Banner+XDomainHardwarePeer");

            let (is_known_fleet, expected_ula, expected_lla, pairwise_v4) =
                if let Some(ref peer_canon) = resolved_peer_name {
                    if let Ok(pid) = resolve_peer_identity(Some(peer_canon), Some(&local_clone), reg_buf.as_deref()) {
                        let pair_local = compute_commutative_pairwise_ipv4(&local_clone, peer_canon).ok();
                        (
                            pid.is_known_fleet_peer,
                            Some(pid.primary_ula),
                            Some(pid.primary_link_local),
                            pair_local,
                        )
                    } else {
                        (false, None, None, None)
                    }
                } else {
                    (false, None, None, None)
                };

            let preferred_transport = if item.scope_type == "IPv6ULA" && (ssh_open || winrm_open) {
                item.connect_target.clone()
            } else if let Some(ref ula) = expected_ula {
                if item.scope_type == "IPv6ULA" {
                    ula.clone()
                } else {
                    item.connect_target.clone()
                }
            } else {
                item.connect_target.clone()
            };

            DiscoveredThunderboltPeer {
                local_computer: local_clone,
                peer_name: resolved_peer_name,
                is_known_fleet_peer: is_known_fleet,
                interface_index: item.interface_index,
                interface_alias: item.interface_alias,
                interface_description: item.interface_description,
                link_speed_gbps: item.link_speed_gbps,
                candidate_address: item.connect_target,
                address_family: item.address_family,
                scope_type: item.scope_type,
                discovery_source: item.source,
                neighbor_state: item.neighbor_state,
                link_layer_address: item.mac,
                ssh_open,
                ssh_banner,
                win_rm_open: winrm_open,
                ntlm_computer_name: ntlm_id.as_ref().and_then(|i| i.netbios_computer_name.clone()),
                ntlm_dns_name: ntlm_id.as_ref().and_then(|i| i.dns_computer_name.clone()),
                verified_handshake: verified,
                handshake_method,
                probe_latency_us: total_us,
                preferred_transport_address: preferred_transport,
                expected_peer_ula: expected_ula,
                expected_peer_link_local: expected_lla,
                pairwise_ipv4: pairwise_v4,
            }
        });
    }

    let mut results = Vec::new();
    while let Some(Ok(peer)) = join_set.join_next().await {
        if include_unreachable || peer.verified_handshake || peer.ssh_open || peer.win_rm_open {
            results.push(peer);
        }
    }
    results
}

pub async fn discover_thunderbolt_peers(
    include_disconnected: bool,
    timeout_ms: u64,
    include_unreachable: bool,
    registry_path: Option<&Path>,
) -> Result<Vec<DiscoveredThunderboltPeer>> {
    let snapshot = collect_platform_snapshot(include_disconnected)?;
    let (reg_opt, _) = load_peer_registry_checked(registry_path)?;
    let known_entries = Arc::new(reg_opt.map(|r| r.fleet_peers).unwrap_or_default());
    let local_canon = canonicalize_peer_name(&local_hostname(), &known_entries);

    let mut tier1_neighbor_items = Vec::new();
    let mut tier2_fleet_items = Vec::new();

    for adapter in &snapshot.adapters {
        if adapter.operational_status == "Up" || include_disconnected {
            for item in gather_candidates_for_adapter(adapter, &local_canon, &known_entries, registry_path)? {
                if item.source == "NeighborTable" || item.source == "ActiveSlash30Peer" {
                    tier1_neighbor_items.push(item);
                } else {
                    tier2_fleet_items.push(item);
                }
            }
        }
    }

    let mut results = probe_work_batch(
        tier1_neighbor_items,
        Arc::clone(&known_entries),
        &local_canon,
        timeout_ms,
        include_unreachable,
        registry_path,
    )
    .await;

    let verified_interfaces: HashSet<u32> = results
        .iter()
        .filter(|r| r.verified_handshake)
        .map(|r| r.interface_index)
        .collect();

    let remaining_tier2: Vec<CandidateWorkItem> = tier2_fleet_items
        .into_iter()
        .filter(|item| include_unreachable || !verified_interfaces.contains(&item.interface_index))
        .collect();

    if !remaining_tier2.is_empty() {
        let tier2_results = probe_work_batch(
            remaining_tier2,
            Arc::clone(&known_entries),
            &local_canon,
            timeout_ms,
            include_unreachable,
            registry_path,
        )
        .await;
        results.extend(tier2_results);
    }

    results.sort_by(|a, b| {
        b.verified_handshake
            .cmp(&a.verified_handshake)
            .then_with(|| b.ssh_open.cmp(&a.ssh_open))
            .then_with(|| b.win_rm_open.cmp(&a.win_rm_open))
            .then_with(|| a.probe_latency_us.cmp(&b.probe_latency_us))
    });

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    fn encode_utf16le(s: &str) -> Vec<u8> {
        s.encode_utf16().flat_map(|u| u.to_le_bytes()).collect()
    }

    fn build_ntlm_type2_challenge_bytes(nb_host: &str, dns_host: &str) -> Vec<u8> {
        let nb_name = encode_utf16le(nb_host);
        let dns_name = encode_utf16le(dns_host);

        let mut target_info = Vec::new();
        target_info.extend_from_slice(&1u16.to_le_bytes());
        target_info.extend_from_slice(&(nb_name.len() as u16).to_le_bytes());
        target_info.extend_from_slice(&nb_name);
        target_info.extend_from_slice(&3u16.to_le_bytes());
        target_info.extend_from_slice(&(dns_name.len() as u16).to_le_bytes());
        target_info.extend_from_slice(&dns_name);
        target_info.extend_from_slice(&0u16.to_le_bytes());
        target_info.extend_from_slice(&0u16.to_le_bytes());

        let mut msg = vec![0u8; 48];
        msg[0..8].copy_from_slice(b"NTLMSSP\0");
        msg[8..12].copy_from_slice(&2u32.to_le_bytes());
        let ti_len = target_info.len() as u16;
        let ti_off = 48u32;
        msg[40..42].copy_from_slice(&ti_len.to_le_bytes());
        msg[42..44].copy_from_slice(&ti_len.to_le_bytes());
        msg[44..48].copy_from_slice(&ti_off.to_le_bytes());
        msg.extend_from_slice(&target_info);
        msg
    }

    #[test]
    fn test_parse_ntlm_challenge_bytes_extracts_computer_and_dns_names() {
        let msg = build_ntlm_type2_challenge_bytes("DTM-CARBON-TWO", "dtm-carbon-two.local");
        let parsed = parse_ntlm_challenge_bytes(&msg).expect("should parse NTLM Type 2");
        assert_eq!(parsed.netbios_computer_name.as_deref(), Some("DTM-CARBON-TWO"));
        assert_eq!(parsed.dns_computer_name.as_deref(), Some("dtm-carbon-two.local"));
        assert_eq!(parsed.best_hostname().as_deref(), Some("dtm-carbon-two.local"));
    }

    #[test]
    fn test_parse_ntlm_challenge_rejects_invalid_headers() {
        assert!(parse_ntlm_challenge_bytes(b"short").is_none());
        let mut fake = vec![0u8; 64];
        fake[0..8].copy_from_slice(b"NTLMSSP\0");
        fake[8..12].copy_from_slice(&1u32.to_le_bytes()); // Type 1 instead of Type 2
        assert!(parse_ntlm_challenge_bytes(&fake).is_none());
    }

    #[tokio::test]
    async fn test_probe_ssh_banner_against_live_ephemeral_tcp_server() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind ephemeral ssh");
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let _ = sock.write_all(b"SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13\r\n").await;
            }
        });

        let (open, banner, _) = probe_ssh_banner("127.0.0.1", port, 500).await;
        assert!(open);
        assert_eq!(banner.as_deref(), Some("SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13"));
    }

    #[tokio::test]
    async fn test_probe_winrm_ntlm_identity_against_live_ephemeral_http_server() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind ephemeral winrm");
        let port = listener.local_addr().unwrap().port();
        let challenge_b64 = base64::engine::general_purpose::STANDARD
            .encode(build_ntlm_type2_challenge_bytes("DTM-P1GEN9", "dtm-p1gen9.local"));

        tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut req_buf = [0u8; 1024];
                let n = sock.read(&mut req_buf).await.unwrap_or(0);
                let req_str = String::from_utf8_lossy(&req_buf[..n]);
                assert!(req_str.contains("POST /wsman HTTP/1.1"));
                assert!(req_str.contains(NTLM_NEGOTIATE_TYPE1_B64));
                let resp = format!(
                    "HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Negotiate {challenge_b64}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
                let _ = sock.write_all(resp.as_bytes()).await;
            }
        });

        let (open, id_opt, _) = probe_winrm_ntlm_identity("127.0.0.1", port, 500).await;
        assert!(open);
        let id = id_opt.expect("should decode NTLM challenge from HTTP 401 response");
        assert_eq!(id.netbios_computer_name.as_deref(), Some("DTM-P1GEN9"));
        assert_eq!(id.dns_computer_name.as_deref(), Some("dtm-p1gen9.local"));
    }

    #[tokio::test]
    async fn test_probe_work_batch_filters_stale_neighbor_and_rejects_bannerless_port() {
        // 1. Stale NeighborTable entry pointing at an unbound port must be filtered out when include_unreachable=false
        let closed_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let closed_port = closed_listener.local_addr().unwrap().port();
        drop(closed_listener);

        let stale_neighbor = CandidateWorkItem {
            interface_index: 99,
            interface_alias: "Ethernet TB".to_string(),
            interface_description: "USB4(TM) P2P Network Adapter".to_string(),
            link_speed_gbps: 20.0,
            connect_target: "127.0.0.1".to_string(),
            address_family: "IPv4".to_string(),
            scope_type: "IPv4LinkLocal".to_string(),
            source: "NeighborTable".to_string(),
            neighbor_state: "Stale".to_string(),
            mac: Some("AA-BB-CC-DD-EE-FF".to_string()),
            expected_peer_hint: None,
            ssh_port: closed_port,
            winrm_port: closed_port,
        };

        let known = Arc::new(Vec::new());
        let filtered = probe_work_batch(
            vec![stale_neighbor.clone()],
            Arc::clone(&known),
            "dtm-p1gen7",
            150,
            false,
            None,
        )
        .await;
        assert!(
            filtered.is_empty(),
            "Stale unreachable NeighborTable entry must NOT be returned when include_unreachable=false"
        );

        let unfiltered =
            probe_work_batch(vec![stale_neighbor], Arc::clone(&known), "dtm-p1gen7", 150, true, None).await;
        assert_eq!(unfiltered.len(), 1);
        assert!(!unfiltered[0].verified_handshake);

        // 2. Bannerless TCP port accept (no SSH- banner, no NTLM Type-2) must NOT set verified_handshake = true
        let silent_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let silent_port = silent_listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            if let Ok((sock, _)) = silent_listener.accept().await {
                drop(sock);
            }
        });

        let bannerless_candidate = CandidateWorkItem {
            interface_index: 99,
            interface_alias: "Ethernet TB".to_string(),
            interface_description: "USB4(TM) P2P Network Adapter".to_string(),
            link_speed_gbps: 20.0,
            connect_target: "127.0.0.1".to_string(),
            address_family: "IPv6".to_string(),
            scope_type: "IPv6ULA".to_string(),
            source: "FleetDeterministicULA".to_string(),
            neighbor_state: "DeterministicProbe".to_string(),
            mac: None,
            expected_peer_hint: Some("dtm-p1gen9".to_string()),
            ssh_port: silent_port,
            winrm_port: closed_port,
        };

        let res = probe_work_batch(vec![bannerless_candidate], known, "dtm-p1gen7", 200, false, None).await;
        assert_eq!(res.len(), 1);
        assert!(res[0].ssh_open);
        assert!(
            !res[0].verified_handshake,
            "Bannerless TCP accept must never be marked verified_handshake=true"
        );
        assert!(
            res[0].peer_name.is_none(),
            "Bannerless TCP accept must not attribute peer_name"
        );
    }

    #[test]
    fn test_parse_sockaddr_with_scope_and_slash30_peer() {
        let sa_v4 = parse_sockaddr_with_scope("10.254.253.2", 22).expect("IPv4");
        assert!(sa_v4.is_ipv4());
        assert_eq!(sa_v4.port(), 22);

        let sa_ula = parse_sockaddr_with_scope("fd00:7462:7032::9", 22).expect("ULA");
        assert!(sa_ula.is_ipv6());

        let sa_scoped = parse_sockaddr_with_scope("fe80::d700:9%19", 22).expect("Scoped LLA");
        match sa_scoped {
            SocketAddr::V6(v6) => assert_eq!(v6.scope_id(), 19),
            _ => panic!("expected SocketAddr::V6"),
        }

        let sa_bracketed = parse_sockaddr_with_scope("[fe80::d700:21%7]", 22).expect("Bracketed Scoped LLA");
        match sa_bracketed {
            SocketAddr::V6(v6) => assert_eq!(v6.scope_id(), 7),
            _ => panic!("expected SocketAddr::V6"),
        }

        assert!(parse_sockaddr_with_scope("fe80::d700:9%0", 22).is_none());
        assert!(parse_sockaddr_with_scope("fe80::d700:9%", 22).is_none());
        assert!(parse_sockaddr_with_scope("fe80::d700:9%nonexistent_if_999999", 22).is_none());

        assert_eq!(peer_ipv4_in_slash30("10.254.253.1").as_deref(), Some("10.254.253.2"));
        assert_eq!(peer_ipv4_in_slash30("10.254.253.2").as_deref(), Some("10.254.253.1"));
        assert_eq!(peer_ipv4_in_slash30("10.254.253.0"), None);
        assert_eq!(peer_ipv4_in_slash30("10.254.253.3"), None);
    }

    #[tokio::test]
    async fn test_xdomain_hardware_peer_verifies_ssh_on_slash30_link() {
        let ssh_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let ssh_port = ssh_listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = ssh_listener.accept().await {
                let _ = sock.write_all(b"SSH-2.0-OpenSSH_9.6p1 Ubuntu\r\n").await;
            }
        });

        let closed_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let closed_port = closed_listener.local_addr().unwrap().port();
        drop(closed_listener);

        let slash30_candidate = CandidateWorkItem {
            interface_index: 167,
            interface_alias: "usb4-milly".to_string(),
            interface_description: "Thunderbolt / USB4 P2P Network Adapter (thunderbolt-net)".to_string(),
            link_speed_gbps: 20.0,
            connect_target: "127.0.0.1".to_string(),
            address_family: "IPv4".to_string(),
            scope_type: "IPv4Slash30".to_string(),
            source: "ActiveSlash30Peer".to_string(),
            neighbor_state: "Slash30Peer".to_string(),
            mac: None,
            expected_peer_hint: Some("millylaptop1".to_string()),
            ssh_port,
            winrm_port: closed_port,
        };

        let known = Arc::new(Vec::new());
        let res = probe_work_batch(vec![slash30_candidate], known, "minisforum", 300, false, None).await;
        assert_eq!(res.len(), 1);
        assert!(res[0].verified_handshake);
        assert_eq!(res[0].peer_name.as_deref(), Some("millylaptop1"));
        assert_eq!(res[0].handshake_method, "SSH-Banner+XDomainHardwarePeer");
    }
}
