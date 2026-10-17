use crate::configure::{plan_and_apply_adapter_actions, CommandRunner, PlannedAction, StdCommandRunner};
use crate::handshake::{discover_thunderbolt_peers, DiscoveredThunderboltPeer};
use crate::identity::{
    canonicalize_peer_name, compute_commutative_pairwise_ipv4, load_peer_registry_checked, local_hostname,
    resolve_peer_identity, PeerRegistryFile, TopologyLinkSpec, Usb4LifecyclePolicy, Usb4LifecycleState,
};
use crate::platform::{collect_platform_snapshot, PlatformThunderboltSnapshot, ThunderboltAdapterState};
use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub struct ExpiringLifecycleEvidence {
    pub observed_at_utc: String,
    pub evaluated_at_utc: String,
    pub ttl_seconds: u64,
    pub expires_at_utc: String,
    pub expired: bool,
    pub invalidated_by_standby_506: bool,
    pub invalidation_reason: Option<String>,
}

impl ExpiringLifecycleEvidence {
    pub fn evaluate(
        observed_at: DateTime<Utc>,
        evaluated_at: DateTime<Utc>,
        ttl_seconds: u64,
        last_standby_entry_506: Option<DateTime<Utc>>,
        currently_in_modern_standby: bool,
    ) -> Self {
        let clamped_ttl = ttl_seconds.max(1);
        let expires_at = observed_at + ChronoDuration::seconds(clamped_ttl as i64);
        let ttl_expired = evaluated_at >= expires_at;
        let standby_after_obs = last_standby_entry_506.is_some_and(|t506| t506 >= observed_at);
        let invalidated_by_standby_506 = currently_in_modern_standby || standby_after_obs;

        let invalidation_reason = if currently_in_modern_standby {
            Some("System is currently in S0 Modern Standby (Kernel-Power 506 without subsequent 507)".to_string())
        } else if standby_after_obs {
            Some("Pre-sleep evidence invalidated by Kernel-Power Modern Standby entry (Event 506)".to_string())
        } else if ttl_expired {
            Some(format!(
                "Lifecycle evidence TTL ({clamped_ttl}s) expired at {}",
                expires_at.to_rfc3339()
            ))
        } else {
            None
        };

        Self {
            observed_at_utc: observed_at.to_rfc3339(),
            evaluated_at_utc: evaluated_at.to_rfc3339(),
            ttl_seconds: clamped_ttl,
            expires_at_utc: expires_at.to_rfc3339(),
            expired: ttl_expired || invalidated_by_standby_506,
            invalidated_by_standby_506,
            invalidation_reason,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub struct ModernStandbyTelemetry {
    pub s0_low_power_idle_supported: bool,
    pub s0_network_connected_supported: bool,
    pub last_standby_entry_506_utc: Option<String>,
    pub last_standby_entry_reason: Option<String>,
    pub last_standby_exit_507_utc: Option<String>,
    pub last_standby_exit_reason: Option<String>,
    pub currently_in_modern_standby: bool,
    pub recent_standby_cycle_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub struct Usb4AdapterPowerCapabilities {
    pub interface_index: u32,
    pub interface_alias: String,
    pub interface_description: String,
    pub selective_suspend_enabled: bool,
    pub wake_on_magic_packet: String,
    pub wake_on_pattern: String,
    pub arp_offload: String,
    pub ns_offload: String,
    pub device_sleep_on_disconnect: String,
    pub always_on_transit_capable: bool,
    pub transit_claim_prohibition_reasons: Vec<String>,
}

impl Usb4AdapterPowerCapabilities {
    pub fn evaluate(
        interface_index: u32,
        interface_alias: &str,
        interface_description: &str,
        selective_suspend_enabled: bool,
        wake_on_magic_packet: &str,
        wake_on_pattern: &str,
        arp_offload: &str,
        ns_offload: &str,
        device_sleep_on_disconnect: &str,
        operational_status: &str,
    ) -> Self {
        let mut reasons = Vec::new();
        if selective_suspend_enabled {
            reasons.push("SelectiveSuspend=Enabled".to_string());
        }
        let check_wake = |label: &str, val: &str, out: &mut Vec<String>| {
            let norm = val.trim();
            if norm.eq_ignore_ascii_case("Unsupported")
                || norm.eq_ignore_ascii_case("Disabled")
                || norm.eq_ignore_ascii_case("Unknown")
                || norm.is_empty()
            {
                out.push(format!("{label}={}", if norm.is_empty() { "Unknown" } else { norm }));
            }
        };
        check_wake("WakeOnMagicPacket", wake_on_magic_packet, &mut reasons);
        check_wake("WakeOnPattern", wake_on_pattern, &mut reasons);
        check_wake("ArpOffload", arp_offload, &mut reasons);
        check_wake("NSOffload", ns_offload, &mut reasons);

        if !operational_status.trim().eq_ignore_ascii_case("Up") {
            reasons.push(format!("OperationalStatus={}", operational_status.trim()));
        }

        let always_on_transit_capable = reasons.is_empty();
        Self {
            interface_index,
            interface_alias: interface_alias.to_string(),
            interface_description: interface_description.to_string(),
            selective_suspend_enabled,
            wake_on_magic_packet: wake_on_magic_packet.to_string(),
            wake_on_pattern: wake_on_pattern.to_string(),
            arp_offload: arp_offload.to_string(),
            ns_offload: ns_offload.to_string(),
            device_sleep_on_disconnect: device_sleep_on_disconnect.to_string(),
            always_on_transit_capable,
            transit_claim_prohibition_reasons: reasons,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub struct IdentityBoundCanaryResult {
    pub expected_peer: String,
    pub observed_peer: Option<String>,
    pub interface_index: Option<u32>,
    pub interface_alias: Option<String>,
    pub target_address: Option<String>,
    pub handshake_method: Option<String>,
    pub observed_mac: Option<String>,
    pub probe_latency_us: Option<u64>,
    pub canary_passed: bool,
    pub failure_reason: Option<String>,
}

pub fn evaluate_identity_bound_canary(
    expected_peer: &str,
    expected_interface: Option<&str>,
    expected_peer_macs: &std::collections::BTreeMap<String, String>,
    discovered_peers: &[DiscoveredThunderboltPeer],
    known_peers: &[crate::identity::PeerRegistryEntry],
) -> IdentityBoundCanaryResult {
    let expected_canon = canonicalize_peer_name(expected_peer, known_peers);
    let expected_mac_for_peer: Option<String> = expected_peer_macs
        .iter()
        .find(|(k, _)| canonicalize_peer_name(k, known_peers) == expected_canon)
        .map(|(_, v)| v.trim().replace('-', ":").to_ascii_lowercase())
        .filter(|m| !m.is_empty());

    let candidates: Vec<&DiscoveredThunderboltPeer> = discovered_peers
        .iter()
        .filter(|d| {
            if let Some(exp_if) = expected_interface {
                if !exp_if.trim().is_empty()
                    && !exp_if.contains('/')
                    && !d.interface_alias.eq_ignore_ascii_case(exp_if.trim())
                {
                    return false;
                }
            }
            true
        })
        .collect();

    for cand in &candidates {
        let Some(ref obs_raw) = cand.peer_name else {
            continue;
        };
        let obs_canon = canonicalize_peer_name(obs_raw, known_peers);
        if obs_canon != expected_canon {
            continue;
        }
        if !cand.verified_handshake || !cand.is_known_fleet_peer {
            return IdentityBoundCanaryResult {
                expected_peer: expected_canon,
                observed_peer: Some(obs_canon),
                interface_index: Some(cand.interface_index),
                interface_alias: Some(cand.interface_alias.clone()),
                target_address: Some(cand.preferred_transport_address.clone()),
                handshake_method: Some(cand.handshake_method.clone()),
                observed_mac: cand.link_layer_address.clone(),
                probe_latency_us: Some(cand.probe_latency_us),
                canary_passed: false,
                failure_reason: Some(format!(
                    "Peer '{}' responded without verified cryptographic/NTLM/SSH handshake or fleet registration",
                    obs_raw
                )),
            };
        }
        if let Some(ref exp_mac) = expected_mac_for_peer {
            if let Some(ref obs_mac) = cand.link_layer_address {
                let norm_obs = obs_mac.trim().replace('-', ":").to_ascii_lowercase();
                if !norm_obs.is_empty() && &norm_obs != exp_mac {
                    return IdentityBoundCanaryResult {
                        expected_peer: expected_canon,
                        observed_peer: Some(obs_canon),
                        interface_index: Some(cand.interface_index),
                        interface_alias: Some(cand.interface_alias.clone()),
                        target_address: Some(cand.preferred_transport_address.clone()),
                        handshake_method: Some(cand.handshake_method.clone()),
                        observed_mac: Some(obs_mac.clone()),
                        probe_latency_us: Some(cand.probe_latency_us),
                        canary_passed: false,
                        failure_reason: Some(format!(
                            "Identity-bound canary MAC mismatch for '{}': observed '{}' != expected '{}'",
                            expected_peer, obs_mac, exp_mac
                        )),
                    };
                }
            }
        }
        return IdentityBoundCanaryResult {
            expected_peer: expected_canon,
            observed_peer: Some(obs_canon),
            interface_index: Some(cand.interface_index),
            interface_alias: Some(cand.interface_alias.clone()),
            target_address: Some(cand.preferred_transport_address.clone()),
            handshake_method: Some(cand.handshake_method.clone()),
            observed_mac: cand.link_layer_address.clone(),
            probe_latency_us: Some(cand.probe_latency_us),
            canary_passed: true,
            failure_reason: None,
        };
    }

    // Check if a different peer identity answered on the expected interface (identity spoofing/Cross-plug negative control)
    if let Some(wrong_peer) = candidates.iter().find(|d| d.verified_handshake && d.peer_name.is_some()) {
        let wrong_name = wrong_peer.peer_name.clone().unwrap_or_default();
        let wrong_canon = canonicalize_peer_name(&wrong_name, known_peers);
        if wrong_canon != expected_canon {
            return IdentityBoundCanaryResult {
                expected_peer: expected_canon.clone(),
                observed_peer: Some(wrong_canon.clone()),
                interface_index: Some(wrong_peer.interface_index),
                interface_alias: Some(wrong_peer.interface_alias.clone()),
                target_address: Some(wrong_peer.preferred_transport_address.clone()),
                handshake_method: Some(wrong_peer.handshake_method.clone()),
                observed_mac: wrong_peer.link_layer_address.clone(),
                probe_latency_us: Some(wrong_peer.probe_latency_us),
                canary_passed: false,
                failure_reason: Some(format!(
                    "Identity-bound canary mismatch: expected peer '{expected_canon}', but interface connected to '{wrong_canon}'"
                )),
            };
        }
    }

    IdentityBoundCanaryResult {
        expected_peer: expected_canon,
        observed_peer: None,
        interface_index: None,
        interface_alias: expected_interface.map(str::to_string),
        target_address: None,
        handshake_method: None,
        observed_mac: None,
        probe_latency_us: None,
        canary_passed: false,
        failure_reason: Some(format!(
            "No authenticated identity-bound canary response from peer '{expected_peer}'"
        )),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub struct StaleRouteWithdrawalAction {
    pub link_id: String,
    pub peer_name: String,
    pub interface_index: u32,
    pub interface_alias: String,
    pub route_prefix: String,
    pub address_family: String,
    pub lifecycle_state: Usb4LifecycleState,
    pub reason: String,
    pub preserved: bool,
    pub withdrawn: bool,
    pub command_summary: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct PeerLinkLifecycleStatus {
    pub link_id: String,
    pub local_host: String,
    pub local_interface: String,
    pub peer_host: String,
    pub peer_interface: String,
    pub link_intent: String,
    pub lifecycle_state: Usb4LifecycleState,
    pub always_on_transit_capable: bool,
    pub transit_prohibition_reasons: Vec<String>,
    pub preserve_existing_pair: bool,
    pub preserve_bridge_custody: bool,
    pub downstream_link_status: Option<String>,
    pub windows_xdomain_present: bool,
    pub evidence: ExpiringLifecycleEvidence,
    pub canary: IdentityBoundCanaryResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct BoundedResumeReprobeAttempt {
    pub attempt: u32,
    pub timeout_ms: u64,
    pub elapsed_ms: u64,
    pub state_before: Usb4LifecycleState,
    pub state_after: Usb4LifecycleState,
    pub verified_peers_found: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct BoundedResumeReprobeReport {
    pub triggered: bool,
    pub trigger_reason: String,
    pub max_attempts: u32,
    pub per_attempt_timeout_ms: u64,
    pub total_budget_ms: u64,
    pub total_elapsed_ms: u64,
    pub budget_respected: bool,
    pub attempts: Vec<BoundedResumeReprobeAttempt>,
    pub reapply_actions: Vec<PlannedAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Usb4LifecycleReport {
    pub schema: String,
    pub computer_name: String,
    pub os_family: String,
    pub architecture: String,
    pub evaluated_at_utc: String,
    pub apply_mode: bool,
    pub policy: Usb4LifecyclePolicy,
    pub modern_standby: ModernStandbyTelemetry,
    pub adapter_power_capabilities: Vec<Usb4AdapterPowerCapabilities>,
    pub peer_link_states: Vec<PeerLinkLifecycleStatus>,
    pub stale_route_withdrawals: Vec<StaleRouteWithdrawalAction>,
    pub resume_reprobe: BoundedResumeReprobeReport,
    pub overall_valid: bool,
    pub warnings: Vec<String>,
}

/// Evaluates the lifecycle state for a single topology link using hardware state,
/// Modern Standby telemetry, expiring evidence, and identity-bound canary results.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_link_lifecycle_state(
    link: &TopologyLinkSpec,
    local_host_canon: &str,
    adapter_opt: Option<&ThunderboltAdapterState>,
    power_opt: Option<&Usb4AdapterPowerCapabilities>,
    standby: &ModernStandbyTelemetry,
    evidence: &ExpiringLifecycleEvidence,
    canary: &IdentityBoundCanaryResult,
    reprobing_now: bool,
) -> (Usb4LifecycleState, bool, Vec<String>) {
    let mut transit_reasons = Vec::new();

    if let Some(p) = power_opt {
        transit_reasons.extend(p.transit_claim_prohibition_reasons.clone());
    } else {
        transit_reasons.push("AdapterPowerCapabilities=Unavailable".to_string());
    }

    if link.always_on_transit_capable == Some(false) {
        let marker = "TopologyPolicy=alwaysOnTransitCapable:false".to_string();
        if !transit_reasons.contains(&marker) {
            transit_reasons.push(marker);
        }
    }

    let link_none = link
        .downstream_link_status
        .as_deref()
        .is_some_and(|s| s.trim().eq_ignore_ascii_case("none") || s.trim().eq_ignore_ascii_case("down"));
    let xdomain_present = adapter_opt
        .and_then(|a| a.xdomain_peer_unique_id.as_ref())
        .is_some()
        || link.windows_xdomain_present.unwrap_or(false);

    if link_none {
        transit_reasons.push("DownstreamLinkStatus=none".to_string());
    }
    if link.windows_xdomain_present == Some(false) && !xdomain_present {
        transit_reasons.push("WindowsXDomain=Absent".to_string());
    }

    // 1. If the local host is currently in S0 Modern Standby (506 without 507) or evidence was invalidated by 506 without fresh reprobe:
    if standby.currently_in_modern_standby {
        transit_reasons.push("LifecycleState=asleep".to_string());
        return (Usb4LifecycleState::Asleep, false, transit_reasons);
    }

    // 2. If explicitly reprobing on resume before canary settles:
    if reprobing_now && !canary.canary_passed {
        transit_reasons.push("LifecycleState=probing".to_string());
        return (Usb4LifecycleState::Probing, false, transit_reasons);
    }

    // 3. Preserve Linux MS01 <-> Milly USB4 bridge custody when topology marks preserveBridgeCustody
    if link.preserve_bridge_custody == Some(true)
        && !local_host_canon.eq_ignore_ascii_case(&link.source_host)
        && !local_host_canon.eq_ignore_ascii_case(&link.target_host)
    {
        let state = link.lifecycle_state_hint.unwrap_or(Usb4LifecycleState::Ready);
        return (state, false, transit_reasons);
    }

    // 4. If downstream link is explicitly "none" or local adapter is missing/down:
    if link_none {
        transit_reasons.push("LifecycleState=disconnected".to_string());
        return (Usb4LifecycleState::Disconnected, false, transit_reasons);
    }

    let Some(adapter) = adapter_opt else {
        transit_reasons.push("LocalAdapter=NotPresent".to_string());
        return (Usb4LifecycleState::Disconnected, false, transit_reasons);
    };

    if !adapter.operational_status.trim().eq_ignore_ascii_case("Up") {
        transit_reasons.push(format!("AdapterOperationalStatus={}", adapter.operational_status));
        return (Usb4LifecycleState::Disconnected, false, transit_reasons);
    }

    // 5. If evidence is expired/invalidated by standby 506:
    if evidence.invalidated_by_standby_506 {
        transit_reasons.push("EvidenceInvalidatedByStandby506=true".to_string());
        return (Usb4LifecycleState::Asleep, false, transit_reasons);
    }
    if evidence.expired {
        transit_reasons.push("EvidenceExpired=true".to_string());
        return (Usb4LifecycleState::Disconnected, false, transit_reasons);
    }

    // 6. Adapter is Up and evidence is within TTL: check identity-bound canary
    if canary.canary_passed {
        let always_on = transit_reasons.is_empty() && link.always_on_transit_capable == Some(true);
        (Usb4LifecycleState::Ready, always_on, transit_reasons)
    } else {
        transit_reasons.push(
            canary
                .failure_reason
                .clone()
                .unwrap_or_else(|| "IdentityBoundCanary=Failed".to_string()),
        );
        (Usb4LifecycleState::Disconnected, false, transit_reasons)
    }
}

/// Plans and optionally executes withdrawal of PC-AI-owned stale routes (`/128` ULA in `fd00:7462:7032::/48`
/// and pairwise `/30` or `/32` in `172.31.240.0/20`) for peers whose lifecycle state is not `Ready`,
/// while strictly preserving `millylaptop1` bridge custody and active `dtm-p1gen7<->dtm-p1gen9` pairs.
pub fn plan_and_withdraw_stale_owned_routes<R: CommandRunner>(
    runner: &R,
    local_host_canon: &str,
    peer_links: &[PeerLinkLifecycleStatus],
    adapters: &[ThunderboltAdapterState],
    registry_path: Option<&Path>,
    apply: bool,
) -> Vec<StaleRouteWithdrawalAction> {
    let mut actions = Vec::new();
    let mut active_ready_peers: HashSet<String> = HashSet::new();

    for pl in peer_links {
        if pl.lifecycle_state == Usb4LifecycleState::Ready && pl.canary.canary_passed && !pl.evidence.expired {
            active_ready_peers.insert(pl.peer_host.to_ascii_lowercase());
        }
    }

    for pl in peer_links {
        let peer_lower = pl.peer_host.to_ascii_lowercase();

        // Strictly preserve Milly bridge custody (10.254.253.0/30, ms01-milly-usb4-bridge)
        if pl.preserve_bridge_custody || pl.link_id.eq_ignore_ascii_case("ms01-milly-usb4-bridge") {
            actions.push(StaleRouteWithdrawalAction {
                link_id: pl.link_id.clone(),
                peer_name: pl.peer_host.clone(),
                interface_index: 0,
                interface_alias: pl.local_interface.clone(),
                route_prefix: pl.link_id.clone(),
                address_family: "BridgeCustody".to_string(),
                lifecycle_state: pl.lifecycle_state,
                reason: "Preserving Milly USB4 bridge custody (external owner ms01-usb4-extension)".to_string(),
                preserved: true,
                withdrawn: false,
                command_summary: "No-op: preserve Milly bridge custody".to_string(),
                error: None,
            });
            continue;
        }

        // Preserve active ready pairs (e.g. dtm-p1gen7 <-> dtm-p1gen9)
        if pl.lifecycle_state == Usb4LifecycleState::Ready && pl.canary.canary_passed && !pl.evidence.expired {
            actions.push(StaleRouteWithdrawalAction {
                link_id: pl.link_id.clone(),
                peer_name: pl.peer_host.clone(),
                interface_index: pl.canary.interface_index.unwrap_or(0),
                interface_alias: pl.local_interface.clone(),
                route_prefix: format!("active-pair:{local_host_canon}<->{peer_lower}"),
                address_family: "IPv6+IPv4".to_string(),
                lifecycle_state: pl.lifecycle_state,
                reason: "Active verified USB4 P2P pair with fresh identity-bound canary".to_string(),
                preserved: true,
                withdrawn: false,
                command_summary: format!("Preserve active routes for verified peer '{}'", pl.peer_host),
                error: None,
            });
            continue;
        }

        // If this peer is already Ready on another link, do not withdraw its host routes
        if active_ready_peers.contains(&peer_lower) {
            continue;
        }

        // Resolve PC-AI-owned deterministic ULA and pairwise /30 addresses for the stale/disconnected/asleep peer
        let Ok(peer_id) = resolve_peer_identity(Some(&pl.peer_host), Some(local_host_canon), registry_path) else {
            continue;
        };

        let target_if_idx = adapters
            .iter()
            .find(|a| a.interface_alias.eq_ignore_ascii_case(&pl.local_interface))
            .map(|a| a.interface_index)
            .or_else(|| adapters.first().map(|a| a.interface_index))
            .unwrap_or(0);
        let target_if_alias = adapters
            .iter()
            .find(|a| a.interface_index == target_if_idx)
            .map(|a| a.interface_alias.clone())
            .unwrap_or_else(|| pl.local_interface.clone());

        let withdrawal_reason = if let Some(ref inv) = pl.evidence.invalidation_reason {
            inv.clone()
        } else if let Some(ref fail) = pl.canary.failure_reason {
            fail.clone()
        } else {
            format!("Peer '{}' is in '{}' state", pl.peer_host, pl.lifecycle_state.as_str())
        };

        for ula in &peer_id.all_ula_addresses {
            let prefix = format!("{ula}/128");
            let cmd_summary = format!(
                "Withdraw stale PC-AI ULA host route {prefix} on {target_if_alias} (ifIndex {target_if_idx})"
            );
            let mut withdrawn = false;
            let mut err_msg = None;

            if apply && target_if_idx > 0 {
                #[cfg(target_os = "windows")]
                let (ok, out) = runner.run(
                    "netsh",
                    &[
                        "interface",
                        "ipv6",
                        "delete",
                        "route",
                        &prefix,
                        &target_if_idx.to_string(),
                        "::",
                    ],
                );
                #[cfg(not(target_os = "windows"))]
                let (ok, out) = runner.run("ip", &["-6", "route", "del", &prefix, "dev", &target_if_alias]);

                let benign_absent = out.to_ascii_lowercase().contains("not found")
                    || out.to_ascii_lowercase().contains("element not found")
                    || out.to_ascii_lowercase().contains("no such process");
                withdrawn = ok || benign_absent;
                if !withdrawn {
                    err_msg = Some(out);
                }
            }

            actions.push(StaleRouteWithdrawalAction {
                link_id: pl.link_id.clone(),
                peer_name: pl.peer_host.clone(),
                interface_index: target_if_idx,
                interface_alias: target_if_alias.clone(),
                route_prefix: prefix,
                address_family: "IPv6".to_string(),
                lifecycle_state: pl.lifecycle_state,
                reason: withdrawal_reason.clone(),
                preserved: false,
                withdrawn,
                command_summary: cmd_summary,
                error: err_msg,
            });
        }

        if let Ok(pair_v4) = compute_commutative_pairwise_ipv4(local_host_canon, &pl.peer_host) {
            // Only withdraw if the local adapter actually holds the stale pairwise /30 IPv4 for this disconnected peer
            let has_stale_v4 = adapters
                .iter()
                .filter(|a| a.interface_index == target_if_idx)
                .flat_map(|a| a.addresses.iter())
                .any(|addr| addr.address == pair_v4.local_ipv4_address);

            let prefix = format!("{}/32", pair_v4.peer_ipv4_address);
            let cmd_summary = format!(
                "Withdraw stale PC-AI pairwise IPv4 route/address {} (peer {}) on {target_if_alias} (ifIndex {target_if_idx})",
                pair_v4.subnet_prefix, prefix
            );
            let mut withdrawn = false;
            let mut err_msg = None;

            if apply && has_stale_v4 && target_if_idx > 0 {
                #[cfg(target_os = "windows")]
                let (ok, out) = runner.run(
                    "netsh",
                    &[
                        "interface",
                        "ipv4",
                        "delete",
                        "address",
                        &format!("name={target_if_idx}"),
                        &format!("addr={}", pair_v4.local_ipv4_address),
                    ],
                );
                #[cfg(not(target_os = "windows"))]
                let (ok, out) = runner.run(
                    "ip",
                    &[
                        "-4",
                        "addr",
                        "del",
                        &format!("{}/30", pair_v4.local_ipv4_address),
                        "dev",
                        &target_if_alias,
                    ],
                );
                withdrawn = ok;
                if !ok {
                    err_msg = Some(out);
                }
            }

            actions.push(StaleRouteWithdrawalAction {
                link_id: pl.link_id.clone(),
                peer_name: pl.peer_host.clone(),
                interface_index: target_if_idx,
                interface_alias: target_if_alias,
                route_prefix: prefix,
                address_family: "IPv4".to_string(),
                lifecycle_state: pl.lifecycle_state,
                reason: withdrawal_reason,
                preserved: false,
                withdrawn: withdrawn || !has_stale_v4,
                command_summary: cmd_summary,
                error: err_msg,
            });
        }
    }

    actions
}

#[cfg(target_os = "windows")]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct RawWindowsPowerAndStandbyPayload {
    #[serde(default)]
    pub s0_low_power_idle_supported: bool,
    #[serde(default)]
    pub s0_network_connected_supported: bool,
    pub last_entry_506_utc: Option<String>,
    pub last_entry_reason: Option<String>,
    pub last_exit_507_utc: Option<String>,
    pub last_exit_reason: Option<String>,
    #[serde(default)]
    pub recent_cycle_count: usize,
    #[serde(default)]
    pub adapters: Vec<RawWindowsAdapterPowerItem>,
}

#[cfg(target_os = "windows")]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct RawWindowsAdapterPowerItem {
    pub interface_alias: String,
    pub interface_description: Option<String>,
    pub selective_suspend: Option<String>,
    pub wake_on_magic_packet: Option<String>,
    pub wake_on_pattern: Option<String>,
    pub arp_offload: Option<String>,
    pub ns_offload: Option<String>,
    pub device_sleep_on_disconnect: Option<String>,
}

pub fn collect_modern_standby_and_power_capabilities(
    snapshot: &PlatformThunderboltSnapshot,
) -> (ModernStandbyTelemetry, Vec<Usb4AdapterPowerCapabilities>) {
    #[cfg(target_os = "windows")]
    {
        let ps_script = r#"
$ErrorActionPreference = 'SilentlyContinue'
$ev = @(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power'; Id=506,507} -MaxEvents 12)
$e506 = $ev | Where-Object Id -eq 506 | Select-Object -First 1
$e507 = $ev | Where-Object Id -eq 507 | Select-Object -First 1
$pms = @()
foreach ($n in @(Get-NetAdapter -IncludeHidden | Where-Object { $_.InterfaceDescription -match 'USB4|Thunderbolt|P2P' -or $_.Name -match 'Ethernet 4|Thunderbolt|USB4' })) {
    $pm = Get-NetAdapterPowerManagement -Name $n.Name
    if ($pm) {
        $pms += [pscustomobject]@{
            InterfaceAlias = [string]$n.Name
            InterfaceDescription = [string]$n.InterfaceDescription
            SelectiveSuspend = [string]$pm.SelectiveSuspend
            WakeOnMagicPacket = [string]$pm.WakeOnMagicPacket
            WakeOnPattern = [string]$pm.WakeOnPattern
            ArpOffload = [string]$pm.ArpOffload
            NsOffload = [string]$pm.NSOffload
            DeviceSleepOnDisconnect = [string]$pm.DeviceSleepOnDisconnect
        }
    }
}
[pscustomobject]@{
    S0LowPowerIdleSupported = ($ev.Count -gt 0)
    S0NetworkConnectedSupported = ($ev.Count -gt 0)
    LastEntry506Utc = if ($e506) { $e506.TimeCreated.ToUniversalTime().ToString('o') } else { $null }
    LastEntryReason = if ($e506) { ($e506.Message -replace '\s+', ' ').Trim() } else { $null }
    LastExit507Utc = if ($e507) { $e507.TimeCreated.ToUniversalTime().ToString('o') } else { $null }
    LastExitReason = if ($e507) { ($e507.Message -replace '\s+', ' ').Trim() } else { $null }
    RecentCycleCount = [int](@($ev | Where-Object Id -eq 506).Count)
    Adapters = $pms
} | ConvertTo-Json -Depth 4 -Compress
"#;
        let output_opt = Command::new("pwsh")
            .args(["-NoProfile", "-NonInteractive", "-Command", ps_script])
            .output()
            .or_else(|_| {
                Command::new("powershell")
                    .args(["-NoProfile", "-NonInteractive", "-Command", ps_script])
                    .output()
            })
            .ok();

        if let Some(out) = output_opt {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                if let Ok(parsed) = serde_json::from_str::<RawWindowsPowerAndStandbyPayload>(text.trim()) {
                    let t506 = parsed
                        .last_entry_506_utc
                        .as_deref()
                        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                        .map(|d| d.with_timezone(&Utc));
                    let t507 = parsed
                        .last_exit_507_utc
                        .as_deref()
                        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                        .map(|d| d.with_timezone(&Utc));
                    let in_standby = match (t506, t507) {
                        (Some(entry), Some(exit)) => entry > exit,
                        (Some(_), None) => true,
                        _ => false,
                    };
                    let standby = ModernStandbyTelemetry {
                        s0_low_power_idle_supported: parsed.s0_low_power_idle_supported,
                        s0_network_connected_supported: parsed.s0_network_connected_supported,
                        last_standby_entry_506_utc: parsed.last_entry_506_utc,
                        last_standby_entry_reason: parsed.last_entry_reason,
                        last_standby_exit_507_utc: parsed.last_exit_507_utc,
                        last_standby_exit_reason: parsed.last_exit_reason,
                        currently_in_modern_standby: in_standby,
                        recent_standby_cycle_count: parsed.recent_cycle_count,
                    };

                    let mut caps = Vec::new();
                    for adapter in &snapshot.adapters {
                        let matched = parsed
                            .adapters
                            .iter()
                            .find(|p| p.interface_alias.eq_ignore_ascii_case(&adapter.interface_alias));
                        if let Some(pm) = matched {
                            let sel_susp = pm
                                .selective_suspend
                                .as_deref()
                                .is_some_and(|s| s.trim().eq_ignore_ascii_case("Enabled") || s.trim() == "1");
                            caps.push(Usb4AdapterPowerCapabilities::evaluate(
                                adapter.interface_index,
                                &adapter.interface_alias,
                                pm.interface_description
                                    .as_deref()
                                    .unwrap_or(&adapter.interface_description),
                                sel_susp,
                                pm.wake_on_magic_packet.as_deref().unwrap_or("Unsupported"),
                                pm.wake_on_pattern.as_deref().unwrap_or("Unsupported"),
                                pm.arp_offload.as_deref().unwrap_or("Unsupported"),
                                pm.ns_offload.as_deref().unwrap_or("Unsupported"),
                                pm.device_sleep_on_disconnect.as_deref().unwrap_or("Inactive"),
                                &adapter.operational_status,
                            ));
                        } else {
                            caps.push(Usb4AdapterPowerCapabilities::evaluate(
                                adapter.interface_index,
                                &adapter.interface_alias,
                                &adapter.interface_description,
                                true,
                                "Unsupported",
                                "Unsupported",
                                "Unsupported",
                                "Unsupported",
                                "Inactive",
                                &adapter.operational_status,
                            ));
                        }
                    }
                    return (standby, caps);
                }
            }
        }
    }

    let standby = ModernStandbyTelemetry::default();
    let caps = snapshot
        .adapters
        .iter()
        .map(|a| {
            Usb4AdapterPowerCapabilities::evaluate(
                a.interface_index,
                &a.interface_alias,
                &a.interface_description,
                true,
                "Unsupported",
                "Unsupported",
                "Unsupported",
                "Unsupported",
                "Inactive",
                &a.operational_status,
            )
        })
        .collect();
    (standby, caps)
}

pub async fn run_usb4_lifecycle_evaluation(
    apply: bool,
    timeout_ms: u64,
    registry_path: Option<&Path>,
) -> Result<Usb4LifecycleReport> {
    let now = Utc::now();
    let raw_host = local_hostname();
    let (reg_opt, _) = load_peer_registry_checked(registry_path)?;
    let reg = reg_opt.unwrap_or(PeerRegistryFile {
        schema_version: Some(serde_json::Value::from(1)),
        ula_prefix: None,
        deterministic_ula_subnet: None,
        pairwise_ipv4_pool: None,
        default_mtu_bytes: Some(62000),
        default_interface_metric: Some(5),
        usb4_lifecycle_policy: Some(Usb4LifecyclePolicy::default()),
        ipam_allocations: None,
        sfp_switch_topologies: Vec::new(),
        topology_links: Vec::new(),
        peers: None,
        fleet_peers: Vec::new(),
    });

    let policy = reg.usb4_lifecycle_policy.clone().unwrap_or_default();
    let local_canon = canonicalize_peer_name(&raw_host, &reg.fleet_peers);
    let local_id = resolve_peer_identity(Some(&local_canon), None, registry_path)?;
    let snapshot = collect_platform_snapshot(true)?;
    let (standby, power_caps) = collect_modern_standby_and_power_capabilities(&snapshot);

    // Bounded resume reprobe: obey policy max attempts, per-attempt timeout, and total budget
    let max_attempts = policy.bounded_resume_reprobe_max_attempts.clamp(1, 5);
    let per_attempt_timeout_ms = timeout_ms.min(policy.bounded_resume_reprobe_timeout_ms.max(50));
    let total_budget_ms = policy.bounded_resume_total_budget_ms.max(per_attempt_timeout_ms);

    let reprobe_start = Instant::now();
    let mut attempts = Vec::new();
    let mut discovered_peers = Vec::new();

    for attempt_idx in 1..=max_attempts {
        let elapsed_before = reprobe_start.elapsed().as_millis() as u64;
        if attempt_idx > 1 && elapsed_before >= total_budget_ms {
            break;
        }
        let remaining_budget = total_budget_ms.saturating_sub(elapsed_before).max(50);
        let attempt_timeout = per_attempt_timeout_ms.min(remaining_budget);

        let peers = discover_thunderbolt_peers(false, attempt_timeout, false, registry_path).await?;
        let verified_count = peers
            .iter()
            .filter(|p| p.verified_handshake && p.is_known_fleet_peer)
            .count();
        let elapsed_after = reprobe_start.elapsed().as_millis() as u64;

        attempts.push(BoundedResumeReprobeAttempt {
            attempt: attempt_idx,
            timeout_ms: attempt_timeout,
            elapsed_ms: elapsed_after,
            state_before: if attempt_idx == 1 && standby.last_standby_exit_507_utc.is_some() {
                Usb4LifecycleState::Asleep
            } else {
                Usb4LifecycleState::Probing
            },
            state_after: if verified_count > 0 {
                Usb4LifecycleState::Ready
            } else {
                Usb4LifecycleState::Probing
            },
            verified_peers_found: verified_count,
        });
        discovered_peers = peers;
        if verified_count > 0 || snapshot.adapters.is_empty() {
            break;
        }
    }

    let total_elapsed_ms = reprobe_start.elapsed().as_millis() as u64;
    let obs_time = Utc::now();
    let last_506_dt = standby
        .last_standby_entry_506_utc
        .as_deref()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|d| d.with_timezone(&Utc));

    let mut peer_link_states = Vec::new();
    for link in &reg.topology_links {
        if link.link_intent == "transparent-bridge" && link.link_id == "ms01-gen9-sfp-dock-bridge" {
            // Only evaluate USB4 / bridge links relevant to USB4 lifecycle
            continue;
        }
        let (local_if, peer_host, peer_if) = if link.source_host.eq_ignore_ascii_case(&local_canon) {
            (
                link.source_interface.clone(),
                canonicalize_peer_name(&link.target_host, &reg.fleet_peers),
                link.target_interface.clone(),
            )
        } else if link.target_host.eq_ignore_ascii_case(&local_canon) {
            (
                link.target_interface.clone(),
                canonicalize_peer_name(&link.source_host, &reg.fleet_peers),
                link.source_interface.clone(),
            )
        } else if link.preserve_bridge_custody == Some(true) {
            (
                link.source_interface.clone(),
                canonicalize_peer_name(&link.target_host, &reg.fleet_peers),
                link.target_interface.clone(),
            )
        } else {
            continue;
        };

        let adapter_opt = snapshot
            .adapters
            .iter()
            .find(|a| a.interface_alias.eq_ignore_ascii_case(&local_if));
        let power_opt = power_caps
            .iter()
            .find(|p| p.interface_alias.eq_ignore_ascii_case(&local_if));

        let evidence = ExpiringLifecycleEvidence::evaluate(
            obs_time,
            obs_time,
            policy.default_evidence_ttl_seconds,
            last_506_dt,
            standby.currently_in_modern_standby,
        );

        let canary = if link.preserve_bridge_custody == Some(true)
            && !link.source_host.eq_ignore_ascii_case(&local_canon)
            && !link.target_host.eq_ignore_ascii_case(&local_canon)
        {
            IdentityBoundCanaryResult {
                expected_peer: peer_host.clone(),
                observed_peer: Some(peer_host.clone()),
                interface_index: None,
                interface_alias: Some(local_if.clone()),
                target_address: None,
                handshake_method: Some("ExternalBridgeCustodyPreserved".to_string()),
                observed_mac: None,
                probe_latency_us: Some(0),
                canary_passed: true,
                failure_reason: None,
            }
        } else {
            evaluate_identity_bound_canary(
                &peer_host,
                Some(&local_if),
                &link.expected_peer_macs,
                &discovered_peers,
                &reg.fleet_peers,
            )
        };

        let (state, always_on, reasons) = evaluate_link_lifecycle_state(
            link,
            &local_canon,
            adapter_opt,
            power_opt,
            &standby,
            &evidence,
            &canary,
            false,
        );

        let xdomain_present = adapter_opt
            .and_then(|a| a.xdomain_peer_unique_id.as_ref())
            .is_some()
            || link.windows_xdomain_present.unwrap_or(false);

        peer_link_states.push(PeerLinkLifecycleStatus {
            link_id: link.link_id.clone(),
            local_host: local_canon.clone(),
            local_interface: local_if,
            peer_host,
            peer_interface: peer_if,
            link_intent: link.link_intent.clone(),
            lifecycle_state: state,
            always_on_transit_capable: always_on,
            transit_prohibition_reasons: reasons,
            preserve_existing_pair: link.preserve_existing_pair.unwrap_or(false),
            preserve_bridge_custody: link.preserve_bridge_custody.unwrap_or(false),
            downstream_link_status: link.downstream_link_status.clone(),
            windows_xdomain_present: xdomain_present,
            evidence,
            canary,
        });
    }

    let runner = StdCommandRunner;
    let stale_route_withdrawals = if policy.withdraw_owned_stale_routes_on_expire_or_disconnect {
        plan_and_withdraw_stale_owned_routes(
            &runner,
            &local_canon,
            &peer_link_states,
            &snapshot.adapters,
            registry_path,
            apply,
        )
    } else {
        Vec::new()
    };

    // Reapply idempotent configuration only on Ready links with verified canary
    let mut reapply_actions = Vec::new();
    for pl in &peer_link_states {
        if pl.lifecycle_state == Usb4LifecycleState::Ready && pl.canary.canary_passed {
            if let Some(adapter) = snapshot
                .adapters
                .iter()
                .find(|a| a.interface_alias.eq_ignore_ascii_case(&pl.local_interface))
            {
                let acts = plan_and_apply_adapter_actions(
                    &runner,
                    adapter,
                    &local_id,
                    Some(&pl.peer_host),
                    apply,
                    false,
                    reg.default_mtu_bytes.unwrap_or(62000),
                    reg.default_interface_metric.unwrap_or(5),
                    true,
                    registry_path,
                );
                reapply_actions.extend(acts);
            }
        }
    }

    let resume_reprobe = BoundedResumeReprobeReport {
        triggered: true,
        trigger_reason: if standby.last_standby_exit_507_utc.is_some() {
            "Post-ModernStandby (Kernel-Power 506->507) bounded reprobe and canary verification".to_string()
        } else {
            "Periodic USB4 lifecycle bounded probe and canary verification".to_string()
        },
        max_attempts,
        per_attempt_timeout_ms,
        total_budget_ms,
        total_elapsed_ms,
        budget_respected: total_elapsed_ms <= total_budget_ms.saturating_add(1500),
        attempts,
        reapply_actions,
    };

    let mut warnings = Vec::new();
    for cap in &power_caps {
        if !cap.always_on_transit_capable {
            warnings.push(format!(
                "Adapter '{}' (ifIndex {}) is not always-on-transit capable: {}",
                cap.interface_alias,
                cap.interface_index,
                cap.transit_claim_prohibition_reasons.join(", ")
            ));
        }
    }

    let no_false_transit_claims = peer_link_states.iter().all(|pl| !pl.always_on_transit_capable);
    let withdrawal_errors = stale_route_withdrawals.iter().any(|w| w.error.is_some());

    Ok(Usb4LifecycleReport {
        schema: "pcai.usb4-lifecycle/v1".to_string(),
        computer_name: local_canon,
        os_family: std::env::consts::OS.to_string(),
        architecture: std::env::consts::ARCH.to_string(),
        evaluated_at_utc: now.to_rfc3339(),
        apply_mode: apply,
        policy,
        modern_standby: standby,
        adapter_power_capabilities: power_caps,
        peer_link_states,
        stale_route_withdrawals,
        resume_reprobe,
        overall_valid: no_false_transit_claims && !withdrawal_errors,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::PeerRegistryEntry;
    use crate::platform::AdapterAddressEntry;
    use chrono::TimeZone;
    use std::sync::Mutex;

    struct MockRunner {
        pub calls: Mutex<Vec<(String, Vec<String>)>>,
    }

    impl MockRunner {
        fn new() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    impl CommandRunner for MockRunner {
        fn run(&self, program: &str, args: &[&str]) -> (bool, String) {
            self.calls.lock().unwrap().push((
                program.to_string(),
                args.iter().map(|s| (*s).to_string()).collect(),
            ));
            (true, "OK".to_string())
        }
    }

    fn sample_known_peers() -> Vec<PeerRegistryEntry> {
        vec![
            PeerRegistryEntry {
                name: "dtm-p1gen7".to_string(),
                aliases: vec!["p1gen7".to_string()],
                os_family: Some("windows".to_string()),
                node_id_hex: Some("0007".to_string()),
                short_link_local: Some("fe80::d700:7".to_string()),
                short_ula: Some("fd00:7462:7032::7".to_string()),
                tb_host_alias: Some("dtm-p1gen7-tb".to_string()),
                ssh_port: Some(22),
                ssh_user: Some("david".to_string()),
                win_rm_port: Some(5985),
                lan_addresses: vec!["192.168.50.228".to_string()],
                default_shares: vec!["C$".to_string()],
            },
            PeerRegistryEntry {
                name: "dtm-p1gen9".to_string(),
                aliases: vec!["p1gen9".to_string()],
                os_family: Some("windows".to_string()),
                node_id_hex: Some("0009".to_string()),
                short_link_local: Some("fe80::d700:9".to_string()),
                short_ula: Some("fd00:7462:7032::9".to_string()),
                tb_host_alias: Some("dtm-p1gen9-tb".to_string()),
                ssh_port: Some(22),
                ssh_user: Some("david".to_string()),
                win_rm_port: Some(5985),
                lan_addresses: vec!["192.168.50.90".to_string()],
                default_shares: vec!["C$".to_string()],
            },
            PeerRegistryEntry {
                name: "minisforum".to_string(),
                aliases: vec!["ms01".to_string()],
                os_family: Some("linux".to_string()),
                node_id_hex: Some("0001".to_string()),
                short_link_local: Some("fe80::d700:1".to_string()),
                short_ula: Some("fd00:7462:7032::1".to_string()),
                tb_host_alias: Some("minisforum-tb".to_string()),
                ssh_port: Some(22),
                ssh_user: Some("david".to_string()),
                win_rm_port: Some(5985),
                lan_addresses: vec!["192.168.50.119".to_string()],
                default_shares: Vec::new(),
            },
        ]
    }

    fn sample_gen9_adapter() -> ThunderboltAdapterState {
        ThunderboltAdapterState {
            interface_index: 22,
            interface_alias: "Ethernet 4".to_string(),
            interface_description: "USB4(TM) P2P Network Adapter #2".to_string(),
            adapter_guid: "{423523B4-CE9A-4012-A32E-B68723676418}".to_string(),
            mac_address: "08-93-C1-7B-39-51".to_string(),
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
            addresses: vec![
                AdapterAddressEntry {
                    address: "fd00:7462:7032::9".to_string(),
                    address_family: "IPv6".to_string(),
                    prefix_length: 128,
                    is_link_local: false,
                    is_ula: true,
                    is_apipa: false,
                },
                AdapterAddressEntry {
                    address: "172.31.241.250".to_string(),
                    address_family: "IPv4".to_string(),
                    prefix_length: 30,
                    is_link_local: false,
                    is_ula: false,
                    is_apipa: false,
                },
            ],
            neighbors: Vec::new(),
        }
    }

    #[test]
    fn test_power_capabilities_prohibit_always_on_transit_when_selective_suspend_or_wake_unsupported() {
        let cap = Usb4AdapterPowerCapabilities::evaluate(
            22,
            "Ethernet 4",
            "USB4(TM) P2P Network Adapter #2",
            true,
            "Unsupported",
            "Unsupported",
            "Unsupported",
            "Unsupported",
            "Inactive",
            "Up",
        );
        assert!(!cap.always_on_transit_capable);
        assert!(cap
            .transit_claim_prohibition_reasons
            .contains(&"SelectiveSuspend=Enabled".to_string()));
        assert!(cap
            .transit_claim_prohibition_reasons
            .contains(&"WakeOnMagicPacket=Unsupported".to_string()));
        assert!(cap
            .transit_claim_prohibition_reasons
            .contains(&"WakeOnPattern=Unsupported".to_string()));
        assert!(cap
            .transit_claim_prohibition_reasons
            .contains(&"ArpOffload=Unsupported".to_string()));
        assert!(cap
            .transit_claim_prohibition_reasons
            .contains(&"NSOffload=Unsupported".to_string()));
    }

    #[test]
    fn test_expiring_evidence_ttl_and_standby_506_invalidation() {
        let t0 = Utc.with_ymd_and_hms(2026, 10, 16, 20, 0, 0).unwrap();
        let t_fresh = Utc.with_ymd_and_hms(2026, 10, 16, 20, 2, 0).unwrap();
        let t_expired = Utc.with_ymd_and_hms(2026, 10, 16, 20, 6, 0).unwrap();
        let t_506 = Utc.with_ymd_and_hms(2026, 10, 16, 20, 4, 26).unwrap();

        // Fresh evidence before any 506 event
        let fresh = ExpiringLifecycleEvidence::evaluate(t0, t_fresh, 300, None, false);
        assert!(!fresh.expired);
        assert!(!fresh.invalidated_by_standby_506);
        assert!(fresh.invalidation_reason.is_none());

        // TTL expired (360s > 300s)
        let expired = ExpiringLifecycleEvidence::evaluate(t0, t_expired, 300, None, false);
        assert!(expired.expired);
        assert!(!expired.invalidated_by_standby_506);
        assert!(expired
            .invalidation_reason
            .as_deref()
            .unwrap()
            .contains("TTL (300s) expired"));

        // Pre-sleep evidence invalidated by subsequent Modern Standby Event 506 even before 300s TTL elapses
        let t_after_506 = Utc.with_ymd_and_hms(2026, 10, 16, 20, 4, 40).unwrap();
        let invalidated = ExpiringLifecycleEvidence::evaluate(t0, t_after_506, 300, Some(t_506), false);
        assert!(invalidated.expired);
        assert!(invalidated.invalidated_by_standby_506);
        assert!(invalidated
            .invalidation_reason
            .as_deref()
            .unwrap()
            .contains("Event 506"));
    }

    #[test]
    fn test_identity_bound_canary_rejects_unauthenticated_wrong_peer_or_mac_spoof() {
        let known = sample_known_peers();
        let valid_gen7 = DiscoveredThunderboltPeer {
            local_computer: "dtm-p1gen9".to_string(),
            peer_name: Some("dtm-p1gen7".to_string()),
            is_known_fleet_peer: true,
            interface_index: 22,
            interface_alias: "Ethernet 4".to_string(),
            interface_description: "USB4(TM) P2P Network Adapter #2".to_string(),
            link_speed_gbps: 20.0,
            candidate_address: "fe80::d700:7%22".to_string(),
            address_family: "IPv6".to_string(),
            scope_type: "IPv6LinkLocal".to_string(),
            discovery_source: "NeighborTable".to_string(),
            neighbor_state: "Reachable".to_string(),
            link_layer_address: Some("DE-02-88-56-62-23".to_string()),
            ssh_open: true,
            ssh_banner: Some("SSH-2.0-OpenSSH_for_Windows_9.5".to_string()),
            win_rm_open: true,
            ntlm_computer_name: Some("DTM-P1GEN7".to_string()),
            ntlm_dns_name: Some("DTM-P1GEN7".to_string()),
            verified_handshake: true,
            handshake_method: "WinRM-NTLM-Type2+SSH".to_string(),
            probe_latency_us: 4500,
            preferred_transport_address: "fd00:7462:7032::7".to_string(),
            expected_peer_ula: Some("fd00:7462:7032::7".to_string()),
            expected_peer_link_local: Some("fe80::d700:7".to_string()),
            pairwise_ipv4: None,
        };

        // Positive control: matching peer + MAC + verified NTLM/SSH handshake
        let mut expected_macs_ok = std::collections::BTreeMap::new();
        expected_macs_ok.insert("dtm-p1gen7".to_string(), "de:02:88:56:62:23".to_string());
        let ok_res = evaluate_identity_bound_canary(
            "dtm-p1gen7",
            Some("Ethernet 4"),
            &expected_macs_ok,
            std::slice::from_ref(&valid_gen7),
            &known,
        );
        assert!(ok_res.canary_passed, "expected canary pass: {:?}", ok_res);

        // Negative control 1: MAC mismatch on expected peer
        let mut expected_macs_bad = std::collections::BTreeMap::new();
        expected_macs_bad.insert("dtm-p1gen7".to_string(), "00:11:22:33:44:55".to_string());
        let bad_mac_res = evaluate_identity_bound_canary(
            "dtm-p1gen7",
            Some("Ethernet 4"),
            &expected_macs_bad,
            std::slice::from_ref(&valid_gen7),
            &known,
        );
        assert!(!bad_mac_res.canary_passed);
        assert!(bad_mac_res
            .failure_reason
            .as_deref()
            .unwrap()
            .contains("MAC mismatch"));

        // Negative control 2: Expected minisforum on Ethernet 4, but interface is connected to dtm-p1gen7
        let empty_macs = std::collections::BTreeMap::new();
        let wrong_peer_res = evaluate_identity_bound_canary(
            "minisforum",
            Some("Ethernet 4"),
            &empty_macs,
            std::slice::from_ref(&valid_gen7),
            &known,
        );
        assert!(!wrong_peer_res.canary_passed);
        assert!(wrong_peer_res
            .failure_reason
            .as_deref()
            .unwrap()
            .contains("expected peer 'minisforum', but interface connected to 'dtm-p1gen7'"));
    }

    #[test]
    fn test_lifecycle_transitions_and_stale_route_withdrawal_preserves_gen7_gen9_and_milly_bridge() {
        let known = sample_known_peers();
        let adapter = sample_gen9_adapter();
        let power = Usb4AdapterPowerCapabilities::evaluate(
            22,
            "Ethernet 4",
            "USB4(TM) P2P Network Adapter #2",
            true,
            "Unsupported",
            "Unsupported",
            "Unsupported",
            "Unsupported",
            "Inactive",
            "Up",
        );

        let mut gen7_macs = std::collections::BTreeMap::new();
        gen7_macs.insert("dtm-p1gen7".to_string(), "de:02:88:56:62:23".to_string());
        let gen7_link = TopologyLinkSpec {
            link_id: "gen7-gen9-usb4-hub-chain".to_string(),
            source_host: "dtm-p1gen7".to_string(),
            source_interface: "Ethernet 18".to_string(),
            target_host: "dtm-p1gen9".to_string(),
            target_interface: "Ethernet 4".to_string(),
            link_intent: "routed-p2p".to_string(),
            lifecycle_state_hint: Some(Usb4LifecycleState::Ready),
            always_on_transit_capable: Some(false),
            preserve_existing_pair: Some(true),
            expected_peer_macs: gen7_macs,
            ..Default::default()
        };

        let ms01_plugable_link = TopologyLinkSpec {
            link_id: "ms01-plugable-dock-usb4-routed".to_string(),
            source_host: "minisforum".to_string(),
            source_interface: "domain1/1-1".to_string(),
            target_host: "dtm-p1gen9".to_string(),
            target_interface: "Ethernet 4".to_string(),
            link_intent: "routed-p2p".to_string(),
            lifecycle_state_hint: Some(Usb4LifecycleState::Disconnected),
            always_on_transit_capable: Some(false),
            downstream_link_status: Some("none".to_string()),
            windows_xdomain_present: Some(false),
            ..Default::default()
        };

        let milly_bridge_link = TopologyLinkSpec {
            link_id: "ms01-milly-usb4-bridge".to_string(),
            source_host: "minisforum".to_string(),
            source_interface: "usb4-milly".to_string(),
            target_host: "millylaptop1".to_string(),
            target_interface: "Ethernet 4".to_string(),
            link_intent: "transparent-bridge".to_string(),
            bridge_domain: Some("ms01-switch/sw0".to_string()),
            lifecycle_state_hint: Some(Usb4LifecycleState::Ready),
            always_on_transit_capable: Some(false),
            preserve_bridge_custody: Some(true),
            ..Default::default()
        };

        let t0 = Utc.with_ymd_and_hms(2026, 10, 16, 20, 4, 0).unwrap();
        let t506 = Utc.with_ymd_and_hms(2026, 10, 16, 20, 4, 26).unwrap();
        let t507 = Utc.with_ymd_and_hms(2026, 10, 16, 20, 8, 36).unwrap();
        let t_post_resume = Utc.with_ymd_and_hms(2026, 10, 16, 20, 8, 40).unwrap();

        // Phase A: Host enters Modern Standby (506) -> state must transition to Asleep
        let standby_asleep = ModernStandbyTelemetry {
            s0_low_power_idle_supported: true,
            s0_network_connected_supported: true,
            last_standby_entry_506_utc: Some(t506.to_rfc3339()),
            last_standby_entry_reason: Some("Idle Timeout.".to_string()),
            last_standby_exit_507_utc: None,
            last_standby_exit_reason: None,
            currently_in_modern_standby: true,
            recent_standby_cycle_count: 1,
        };
        let ev_asleep = ExpiringLifecycleEvidence::evaluate(t0, t506, 300, Some(t506), true);
        let empty_macs = std::collections::BTreeMap::new();
        let dummy_canary = evaluate_identity_bound_canary("dtm-p1gen7", Some("Ethernet 4"), &empty_macs, &[], &known);
        let (state_asleep, transit_asleep, _) = evaluate_link_lifecycle_state(
            &gen7_link,
            "dtm-p1gen9",
            Some(&adapter),
            Some(&power),
            &standby_asleep,
            &ev_asleep,
            &dummy_canary,
            false,
        );
        assert_eq!(state_asleep, Usb4LifecycleState::Asleep);
        assert!(!transit_asleep);

        // Phase B: Host exits Modern Standby (507 Input Touchpad) and starts bounded resume reprobe -> Probing
        let standby_awake = ModernStandbyTelemetry {
            s0_low_power_idle_supported: true,
            s0_network_connected_supported: true,
            last_standby_entry_506_utc: Some(t506.to_rfc3339()),
            last_standby_entry_reason: Some("Idle Timeout.".to_string()),
            last_standby_exit_507_utc: Some(t507.to_rfc3339()),
            last_standby_exit_reason: Some("Input Touchpad.".to_string()),
            currently_in_modern_standby: false,
            recent_standby_cycle_count: 1,
        };
        let ev_fresh = ExpiringLifecycleEvidence::evaluate(t_post_resume, t_post_resume, 300, Some(t506), false);
        let (state_probing, transit_probing, _) = evaluate_link_lifecycle_state(
            &gen7_link,
            "dtm-p1gen9",
            Some(&adapter),
            Some(&power),
            &standby_awake,
            &ev_fresh,
            &dummy_canary,
            true,
        );
        assert_eq!(state_probing, Usb4LifecycleState::Probing);
        assert!(!transit_probing);

        // Phase C: Reprobe completes -> gen7-gen9 canary passes (Ready, but always_on_transit_capable=false due to SelectiveSuspend),
        // while ms01-plugable-dock-usb4-routed has link=none / no XDomain (Disconnected).
        let gen7_canary = IdentityBoundCanaryResult {
            expected_peer: "dtm-p1gen7".to_string(),
            observed_peer: Some("dtm-p1gen7".to_string()),
            interface_index: Some(22),
            interface_alias: Some("Ethernet 4".to_string()),
            target_address: Some("fd00:7462:7032::7".to_string()),
            handshake_method: Some("WinRM-NTLM-Type2+SSH".to_string()),
            observed_mac: Some("DE-02-88-56-62-23".to_string()),
            probe_latency_us: Some(4200),
            canary_passed: true,
            failure_reason: None,
        };
        let (state_gen7_ready, transit_gen7, reasons_gen7) = evaluate_link_lifecycle_state(
            &gen7_link,
            "dtm-p1gen9",
            Some(&adapter),
            Some(&power),
            &standby_awake,
            &ev_fresh,
            &gen7_canary,
            false,
        );
        assert_eq!(state_gen7_ready, Usb4LifecycleState::Ready);
        assert!(
            !transit_gen7,
            "SelectiveSuspend=Enabled must prevent always-on transit claim"
        );
        assert!(reasons_gen7.contains(&"SelectiveSuspend=Enabled".to_string()));

        let ms01_canary = evaluate_identity_bound_canary("minisforum", Some("Ethernet 4"), &empty_macs, &[], &known);
        let (state_ms01, transit_ms01, _) = evaluate_link_lifecycle_state(
            &ms01_plugable_link,
            "dtm-p1gen9",
            Some(&adapter),
            Some(&power),
            &standby_awake,
            &ev_fresh,
            &ms01_canary,
            false,
        );
        assert_eq!(state_ms01, Usb4LifecycleState::Disconnected);
        assert!(!transit_ms01);

        let milly_canary = IdentityBoundCanaryResult {
            expected_peer: "millylaptop1".to_string(),
            observed_peer: Some("millylaptop1".to_string()),
            interface_index: None,
            interface_alias: Some("usb4-milly".to_string()),
            target_address: None,
            handshake_method: Some("ExternalBridgeCustodyPreserved".to_string()),
            observed_mac: None,
            probe_latency_us: Some(0),
            canary_passed: true,
            failure_reason: None,
        };
        let (state_milly, _, _) = evaluate_link_lifecycle_state(
            &milly_bridge_link,
            "dtm-p1gen9",
            None,
            None,
            &standby_awake,
            &ev_fresh,
            &milly_canary,
            false,
        );
        assert_eq!(state_milly, Usb4LifecycleState::Ready);

        // Verify route withdrawal withdraws stale minisforum USB4 routes while preserving gen7-gen9 and Milly bridge
        let peer_links = vec![
            PeerLinkLifecycleStatus {
                link_id: gen7_link.link_id.clone(),
                local_host: "dtm-p1gen9".to_string(),
                local_interface: "Ethernet 4".to_string(),
                peer_host: "dtm-p1gen7".to_string(),
                peer_interface: "Ethernet 18".to_string(),
                link_intent: "routed-p2p".to_string(),
                lifecycle_state: state_gen7_ready,
                always_on_transit_capable: transit_gen7,
                transit_prohibition_reasons: reasons_gen7,
                preserve_existing_pair: true,
                preserve_bridge_custody: false,
                downstream_link_status: Some("up".to_string()),
                windows_xdomain_present: false,
                evidence: ev_fresh.clone(),
                canary: gen7_canary,
            },
            PeerLinkLifecycleStatus {
                link_id: ms01_plugable_link.link_id.clone(),
                local_host: "dtm-p1gen9".to_string(),
                local_interface: "Ethernet 4".to_string(),
                peer_host: "minisforum".to_string(),
                peer_interface: "domain1/1-1".to_string(),
                link_intent: "routed-p2p".to_string(),
                lifecycle_state: state_ms01,
                always_on_transit_capable: false,
                transit_prohibition_reasons: vec!["DownstreamLinkStatus=none".to_string()],
                preserve_existing_pair: false,
                preserve_bridge_custody: false,
                downstream_link_status: Some("none".to_string()),
                windows_xdomain_present: false,
                evidence: ev_fresh.clone(),
                canary: ms01_canary,
            },
            PeerLinkLifecycleStatus {
                link_id: milly_bridge_link.link_id.clone(),
                local_host: "dtm-p1gen9".to_string(),
                local_interface: "usb4-milly".to_string(),
                peer_host: "millylaptop1".to_string(),
                peer_interface: "Ethernet 4".to_string(),
                link_intent: "transparent-bridge".to_string(),
                lifecycle_state: state_milly,
                always_on_transit_capable: false,
                transit_prohibition_reasons: Vec::new(),
                preserve_existing_pair: false,
                preserve_bridge_custody: true,
                downstream_link_status: Some("up".to_string()),
                windows_xdomain_present: false,
                evidence: ev_fresh,
                canary: milly_canary,
            },
        ];

        let dir = tempfile::tempdir().expect("tempdir");
        let reg_path = dir.path().join("peers.json");
        let reg_json = serde_json::json!({
            "schemaVersion": 1,
            "fleetPeers": known
        });
        std::fs::write(&reg_path, serde_json::to_string(&reg_json).unwrap()).unwrap();

        let runner = MockRunner::new();
        let withdrawals = plan_and_withdraw_stale_owned_routes(
            &runner,
            "dtm-p1gen9",
            &peer_links,
            std::slice::from_ref(&adapter),
            Some(&reg_path),
            true,
        );

        // gen7-gen9 must be preserved
        assert!(withdrawals
            .iter()
            .any(|w| w.peer_name == "dtm-p1gen7" && w.preserved && !w.withdrawn));
        // Milly bridge must be preserved
        assert!(withdrawals
            .iter()
            .any(|w| w.link_id == "ms01-milly-usb4-bridge" && w.preserved && !w.withdrawn));
        // Disconnected minisforum USB4 ULA route must be withdrawn
        assert!(withdrawals.iter().any(|w| w.peer_name == "minisforum"
            && w.route_prefix == "fd00:7462:7032::1/128"
            && !w.preserved
            && w.withdrawn));
    }
}
