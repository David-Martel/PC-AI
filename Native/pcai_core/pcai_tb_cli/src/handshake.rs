use crate::identity::{
    canonicalize_peer_name, compute_commutative_pairwise_ipv4, known_fleet_peer_names, load_peer_registry,
    local_hostname, resolve_peer_identity, PairwiseIpv4Contract,
};
use crate::platform::{collect_platform_snapshot, ThunderboltAdapterState};
use anyhow::Result;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
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

fn parse_sockaddr_with_scope(addr_str: &str, port: u16) -> Option<SocketAddr> {
    // Handle IPv6 with %scope_id, e.g. fe80::d700:9%95
    let bracketed = if addr_str.contains(':') && !addr_str.starts_with('[') {
        format!("[{addr_str}]:{port}")
    } else {
        format!("{addr_str}:{port}")
    };
    if let Ok(mut iter) = bracketed.to_socket_addrs() {
        if let Some(sa) = iter.next() {
            return Some(sa);
        }
    }
    if let Ok(ip) = addr_str.parse::<IpAddr>() {
        return Some(SocketAddr::new(ip, port));
    }
    None
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
            Some(s.lines().next().unwrap_or("").trim().to_string())
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
struct CandidateWorkItem {
    interface_index: u32,
    interface_alias: String,
    interface_description: String,
    link_speed_gbps: f64,
    connect_target: String,
    address_family: String,
    scope_type: String,
    source: String,
    neighbor_state: String,
    mac: Option<String>,
    expected_peer_hint: Option<String>,
}

fn gather_candidates_for_adapter(
    adapter: &ThunderboltAdapterState,
    local_canon: &str,
    registry_path: Option<&Path>,
) -> Result<Vec<CandidateWorkItem>> {
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    let if_idx = adapter.interface_index;

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
                expected_peer_hint: None,
            });
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
                });
            }
        }
        if let Some(ref pair_v4) = peer_id.pairwise_ipv4 {
            // Note: peer_id was resolved with computer_name=peer_name and peer=local_canon,
            // so peer_id.pairwise_ipv4.local_ipv4_address is the PEER's IPv4 address!
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
                });
            }
        }
    }

    Ok(items)
}

async fn probe_work_batch(
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
                probe_ssh_banner(&item.connect_target, 22, timeout_ms),
                probe_winrm_ntlm_identity(&item.connect_target, 5985, timeout_ms)
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

            if resolved_peer_name.is_none() && (ssh_open || winrm_open) {
                if let Some(ref hint) = item.expected_peer_hint {
                    resolved_peer_name = Some(canonicalize_peer_name(hint, &known_clone));
                    handshake_method = if ssh_open && ssh_banner.is_some() {
                        "SSH-Banner+DeterministicAddress".to_string()
                    } else {
                        "DeterministicAddressProbe".to_string()
                    };
                }
            }

            let verified = resolved_peer_name
                .as_ref()
                .is_some_and(|p| !p.eq_ignore_ascii_case(&local_clone));

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
        if include_unreachable
            || peer.verified_handshake
            || peer.ssh_open
            || peer.win_rm_open
            || peer.discovery_source == "NeighborTable"
        {
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
    let (reg_opt, _) = load_peer_registry(registry_path)?;
    let known_entries = Arc::new(reg_opt.map(|r| r.fleet_peers).unwrap_or_default());
    let local_canon = canonicalize_peer_name(&local_hostname(), &known_entries);

    let mut tier1_neighbor_items = Vec::new();
    let mut tier2_fleet_items = Vec::new();

    for adapter in &snapshot.adapters {
        if adapter.operational_status == "Up" || include_disconnected {
            for item in gather_candidates_for_adapter(adapter, &local_canon, registry_path)? {
                if item.source == "NeighborTable" {
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

    fn encode_utf16le(s: &str) -> Vec<u8> {
        s.encode_utf16().flat_map(|u| u.to_le_bytes()).collect()
    }

    #[test]
    fn test_parse_ntlm_challenge_bytes_extracts_computer_and_dns_names() {
        let nb_name = encode_utf16le("DTM-CARBON-TWO");
        let dns_name = encode_utf16le("dtm-carbon-two.local");

        let mut target_info = Vec::new();
        // MsvAvNbComputerName (1)
        target_info.extend_from_slice(&1u16.to_le_bytes());
        target_info.extend_from_slice(&(nb_name.len() as u16).to_le_bytes());
        target_info.extend_from_slice(&nb_name);
        // MsvAvDnsComputerName (3)
        target_info.extend_from_slice(&3u16.to_le_bytes());
        target_info.extend_from_slice(&(dns_name.len() as u16).to_le_bytes());
        target_info.extend_from_slice(&dns_name);
        // MsvAvEOL (0)
        target_info.extend_from_slice(&0u16.to_le_bytes());
        target_info.extend_from_slice(&0u16.to_le_bytes());

        let mut msg = vec![0u8; 48];
        msg[0..8].copy_from_slice(b"NTLMSSP\0");
        msg[8..12].copy_from_slice(&2u32.to_le_bytes()); // Type 2 Challenge
        let ti_len = target_info.len() as u16;
        let ti_off = 48u32;
        msg[40..42].copy_from_slice(&ti_len.to_le_bytes());
        msg[42..44].copy_from_slice(&ti_len.to_le_bytes());
        msg[44..48].copy_from_slice(&ti_off.to_le_bytes());
        msg.extend_from_slice(&target_info);

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
}
