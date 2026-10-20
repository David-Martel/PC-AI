use crate::configure::{plan_and_apply_adapter_actions, CommandRunner, PlannedAction, StdCommandRunner};
use crate::handshake::{discover_thunderbolt_peers, DiscoveredThunderboltPeer};
use crate::identity::{
    canonicalize_peer_name, classify_usb4_pnp_ancestry, compute_commutative_pairwise_ipv4, load_peer_registry_checked,
    local_hostname, resolve_peer_identity, PeerRegistryFile, TopologyLinkSpec, TopologyPathClass, Usb4LifecyclePolicy,
    Usb4LifecycleState,
};
use crate::platform::{collect_platform_snapshot, PlatformThunderboltSnapshot, ThunderboltAdapterState};
use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;
#[cfg(target_os = "windows")]
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
    #[serde(default)]
    pub s0_low_power_idle_state: String,
    #[serde(default)]
    pub s0_low_power_idle_supported: bool,
    #[serde(default)]
    pub s0_low_power_idle_connected_supported: Option<bool>,
    #[serde(default)]
    pub s0_network_connected_supported: bool,
    #[serde(default)]
    pub event_log_available: bool,
    pub last_standby_entry_506_utc: Option<String>,
    pub last_standby_entry_reason: Option<String>,
    pub last_standby_exit_507_utc: Option<String>,
    pub last_standby_exit_reason: Option<String>,
    pub currently_in_modern_standby: bool,
    pub recent_standby_cycle_count: usize,
}

/// Parses `powercfg /a` output to determine whether S0 Low Power Idle (Modern Standby)
/// and S0 Network Connected are supported, unsupported, or unknown.
pub fn parse_powercfg_s0_capabilities(
    powercfg_output: Option<&str>,
    event_log_available: bool,
    has_506_507_events: bool,
) -> (String, bool, Option<bool>) {
    if let Some(raw) = powercfg_output {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let lower = trimmed.to_ascii_lowercase();
            let not_avail_idx = lower
                .find("the following sleep states are not available")
                .or_else(|| lower.find("not available on this system"));
            let (avail_section, not_avail_section) = match not_avail_idx {
                Some(idx) => (&lower[..idx], &lower[idx..]),
                None => (lower.as_str(), ""),
            };

            if avail_section.contains("standby (s0 low power idle) network connected") {
                return ("Supported".to_string(), true, Some(true));
            }
            if avail_section.contains("standby (s0 low power idle) network disconnected")
                || avail_section.contains("standby (s0 low power idle)")
            {
                let connected = avail_section.contains("network connected");
                return ("Supported".to_string(), true, Some(connected));
            }
            if not_avail_section.contains("standby (s0 low power idle)")
                || lower.contains("standby (s3)") && !avail_section.contains("s0 low power idle")
            {
                return ("Unsupported".to_string(), false, Some(false));
            }
        }
    }

    if has_506_507_events {
        return ("Supported".to_string(), true, Some(true));
    }
    let _ = event_log_available;
    ("Unknown".to_string(), false, None)
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
    pub pnp_instance_id: Option<String>,
    pub pnp_ancestry: Vec<String>,
    pub root_router_id: Option<String>,
    pub pci_host_router_id: Option<String>,
    pub intermediate_dock_routers: Vec<String>,
    pub topology_path_class: TopologyPathClass,
    pub windows_reported_link_speed_bps: u64,
    pub link_speed_is_nominal_driver_display: bool,
    pub link_speed_is_hardware_maximum: bool,
}

impl Usb4AdapterPowerCapabilities {
    #[allow(clippy::too_many_arguments)]
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
            pnp_instance_id: None,
            pnp_ancestry: Vec::new(),
            root_router_id: None,
            pci_host_router_id: None,
            intermediate_dock_routers: Vec::new(),
            topology_path_class: TopologyPathClass::Unknown,
            windows_reported_link_speed_bps: 20_000_000_000,
            link_speed_is_nominal_driver_display: true,
            link_speed_is_hardware_maximum: false,
        }
    }

    pub fn with_pnp_and_link_speed(
        mut self,
        pnp_ancestry: Vec<String>,
        sibling_dock_routers: &[String],
        windows_reported_link_speed_bps: u64,
    ) -> Self {
        if let Ok(classification) = classify_usb4_pnp_ancestry(&pnp_ancestry, sibling_dock_routers) {
            self.pnp_instance_id = Some(classification.adapter_instance_id);
            self.pnp_ancestry = classification.pnp_ancestry;
            self.root_router_id = classification.root_router_id;
            self.pci_host_router_id = classification.pci_host_router_id;
            self.intermediate_dock_routers = classification.intermediate_dock_routers;
            self.topology_path_class = classification.topology_path_class;
        } else {
            self.pnp_instance_id = pnp_ancestry.first().cloned();
            self.pnp_ancestry = pnp_ancestry;
            self.root_router_id = None;
            self.pci_host_router_id = None;
            self.intermediate_dock_routers = Vec::new();
            self.topology_path_class = TopologyPathClass::Unknown;
        }
        self.windows_reported_link_speed_bps = if windows_reported_link_speed_bps > 0 {
            windows_reported_link_speed_bps
        } else {
            20_000_000_000
        };
        self.link_speed_is_nominal_driver_display = true;
        self.link_speed_is_hardware_maximum = false;
        self
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
    #[serde(default)]
    pub scope_id: Option<u32>,
    pub handshake_method: Option<String>,
    pub observed_mac: Option<String>,
    pub probe_latency_us: Option<u64>,
    pub canary_passed: bool,
    pub failure_reason: Option<String>,
}

/// Extracts an IPv6 link-local `%<scope_id>` zone identifier (or applies `fallback_if_idx`
/// when `addr` is a link-local `fe80::` address without an explicit `%` suffix).
pub fn extract_ipv6_scope_id(addr: &str, fallback_if_idx: Option<u32>) -> (String, Option<u32>) {
    let trimmed = addr.trim().trim_start_matches('[').trim_end_matches(']');
    if let Some((ip_part, scope_part)) = trimmed.split_once('%') {
        if let Ok(sid) = scope_part.parse::<u32>() {
            if sid > 0 {
                return (format!("{ip_part}%{sid}"), Some(sid));
            }
        }
        return (trimmed.to_string(), fallback_if_idx.filter(|&i| i > 0));
    }
    if trimmed.to_ascii_lowercase().starts_with("fe80:") {
        if let Some(sid) = fallback_if_idx.filter(|&i| i > 0) {
            return (format!("{trimmed}%{sid}"), Some(sid));
        }
    }
    (trimmed.to_string(), None)
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
        let (_, cand_scope_id) = extract_ipv6_scope_id(&cand.candidate_address, Some(cand.interface_index));
        let (_, pref_scope_id) = extract_ipv6_scope_id(&cand.preferred_transport_address, Some(cand.interface_index));
        let resolved_scope_id = pref_scope_id.or(cand_scope_id);

        if !cand.verified_handshake || !cand.is_known_fleet_peer {
            return IdentityBoundCanaryResult {
                expected_peer: expected_canon,
                observed_peer: Some(obs_canon),
                interface_index: Some(cand.interface_index),
                interface_alias: Some(cand.interface_alias.clone()),
                target_address: Some(cand.preferred_transport_address.clone()),
                scope_id: resolved_scope_id,
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
                        scope_id: resolved_scope_id,
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
            scope_id: resolved_scope_id,
            handshake_method: Some(cand.handshake_method.clone()),
            observed_mac: cand.link_layer_address.clone(),
            probe_latency_us: Some(cand.probe_latency_us),
            canary_passed: true,
            failure_reason: None,
        };
    }

    // Check if a different peer identity answered on the expected interface (identity spoofing/Cross-plug negative control)
    if let Some(wrong_peer) = candidates
        .iter()
        .find(|d| d.verified_handshake && d.peer_name.is_some())
    {
        let wrong_name = wrong_peer.peer_name.clone().unwrap_or_default();
        let wrong_canon = canonicalize_peer_name(&wrong_name, known_peers);
        if wrong_canon != expected_canon {
            let (_, cand_scope_id) =
                extract_ipv6_scope_id(&wrong_peer.candidate_address, Some(wrong_peer.interface_index));
            let (_, pref_scope_id) = extract_ipv6_scope_id(
                &wrong_peer.preferred_transport_address,
                Some(wrong_peer.interface_index),
            );
            return IdentityBoundCanaryResult {
                expected_peer: expected_canon.clone(),
                observed_peer: Some(wrong_canon.clone()),
                interface_index: Some(wrong_peer.interface_index),
                interface_alias: Some(wrong_peer.interface_alias.clone()),
                target_address: Some(wrong_peer.preferred_transport_address.clone()),
                scope_id: pref_scope_id.or(cand_scope_id),
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
        scope_id: None,
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
    pub link_admitted: bool,
    pub admission_reason: String,
    pub required_topology_class: TopologyPathClass,
    pub observed_topology_class: TopologyPathClass,
    pub qualification_scope: String,
    pub required_topology_qualified: bool,
    pub pnp_ancestry: Vec<String>,
    pub root_router_id: Option<String>,
    pub pci_host_router_id: Option<String>,
    pub intermediate_dock_routers: Vec<String>,
    pub local_root_router_fw: Option<String>,
    pub dock_model: Option<String>,
    pub dock_serial: Option<String>,
    pub dock_status: Option<String>,
    pub dock_fw_package: Option<String>,
    pub dock_fw_target: Option<String>,
    pub dock_pd_fw: Option<String>,
    pub dock_tbt5_nvm: Option<String>,
    pub dock_dmc_fw: Option<String>,
    pub windows_reported_link_speed_bps: u64,
    pub link_speed_is_nominal_driver_display: bool,
    pub link_speed_is_hardware_maximum: bool,
    pub measured_payload_gbps_samples: Vec<f64>,
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
    pub share_tb_service_running: bool,
    pub tb_service_status: String,
    pub shared_access_service_status: String,
    pub default_routes_preserved: bool,
    pub adapter_power_capabilities: Vec<Usb4AdapterPowerCapabilities>,
    pub peer_link_states: Vec<PeerLinkLifecycleStatus>,
    pub stale_route_withdrawals: Vec<StaleRouteWithdrawalAction>,
    pub resume_reprobe: BoundedResumeReprobeReport,
    pub overall_valid: bool,
    pub warnings: Vec<String>,
}

/// Evaluates whether a USB4/Thunderbolt link is admitted and whether it qualifies the required topology class.
///
/// Key contracts enforced:
/// 1. No observed XDomain (`!xdomain_present` or `downstream_link_status == "none"` or adapter not `Up`) ->
///    `link_admitted: false`, `admission_reason: "no_observed_xdomain"`.
/// 2. Full PnP parent ancestry (`DEVPKEY_Device_Parent` chain) is checked against `expected_local_root_router_id`
///    and `expected_local_dock_router_id`, never `InstanceId` prefix alone.
/// 3. A `host_port_control` connection (`PID_E333`) never qualifies `required_two_hub` (`PID_5781` + `VID_8087&PID_5786` Lenovo TB5 Dock 7500 `40BA`):
///    `qualification_scope: "host_port_control_only"`, `required_topology_qualified: false`.
pub fn evaluate_link_admission_and_qualification(
    link: &TopologyLinkSpec,
    lifecycle_state: Usb4LifecycleState,
    adapter_opt: Option<&ThunderboltAdapterState>,
    power_opt: Option<&Usb4AdapterPowerCapabilities>,
    xdomain_present: bool,
    canary: &IdentityBoundCanaryResult,
) -> (bool, String, TopologyPathClass, TopologyPathClass, String, bool) {
    let required_class = link.required_topology_class.unwrap_or(TopologyPathClass::Unknown);
    let mut observed_class = link
        .observed_topology_class
        .or_else(|| power_opt.map(|p| p.topology_path_class))
        .unwrap_or(TopologyPathClass::Unknown);

    if observed_class == TopologyPathClass::Unknown {
        if let Some(p) = power_opt {
            if p.topology_path_class != TopologyPathClass::Unknown {
                observed_class = p.topology_path_class;
            }
        }
    }

    if link.preserve_bridge_custody == Some(true) {
        return (
            true,
            "external_bridge_custody_preserved".to_string(),
            required_class,
            observed_class,
            link.qualification_scope
                .clone()
                .unwrap_or_else(|| "external_bridge_custody".to_string()),
            link.required_topology_qualified.unwrap_or(true),
        );
    }

    let is_host_port_control_scope = link
        .qualification_scope
        .as_deref()
        .is_some_and(|s| s.eq_ignore_ascii_case("host_port_control_only"))
        || observed_class == TopologyPathClass::HostPortControl
        || (required_class == TopologyPathClass::RequiredTwoHub
            && power_opt
                .and_then(|p| p.root_router_id.as_deref())
                .is_some_and(|r| r.to_ascii_uppercase().contains("PID_E333")));

    let link_none = link
        .downstream_link_status
        .as_deref()
        .is_some_and(|s| s.trim().eq_ignore_ascii_case("none") || s.trim().eq_ignore_ascii_case("down"));
    let adapter_up = adapter_opt.is_some_and(|a| a.operational_status.trim().eq_ignore_ascii_case("Up"));

    if !xdomain_present || link_none || !adapter_up {
        let qual_scope = if is_host_port_control_scope {
            "host_port_control_only".to_string()
        } else {
            link.qualification_scope
                .clone()
                .unwrap_or_else(|| "unqualified_no_xdomain".to_string())
        };
        return (
            false,
            "no_observed_xdomain".to_string(),
            required_class,
            observed_class,
            qual_scope,
            false,
        );
    }

    // Verify full PnP parent ancestry against expected Root Router and Dock Router if specified
    if let Some(ref exp_root) = link.expected_local_root_router_id {
        if let Some(p) = power_opt {
            if p.pnp_ancestry.len() < 2 {
                return (
                    false,
                    "pnp_ancestry_prefix_only_rejected".to_string(),
                    required_class,
                    observed_class,
                    "unqualified_missing_pnp_ancestry".to_string(),
                    false,
                );
            }
            if let Some(ref obs_root) = p.root_router_id {
                if !obs_root.eq_ignore_ascii_case(exp_root) {
                    let scope = if obs_root.to_ascii_uppercase().contains("PID_E333")
                        || required_class == TopologyPathClass::RequiredTwoHub
                    {
                        "host_port_control_only".to_string()
                    } else {
                        "unqualified_root_router_mismatch".to_string()
                    };
                    return (
                        false,
                        "pnp_ancestry_root_router_mismatch".to_string(),
                        required_class,
                        observed_class,
                        scope,
                        false,
                    );
                }
            }
        }
    }

    if let Some(ref exp_dock) = link.expected_local_dock_router_id {
        if let Some(p) = power_opt {
            let has_dock = p
                .intermediate_dock_routers
                .iter()
                .any(|d| d.eq_ignore_ascii_case(exp_dock));
            if !has_dock {
                return (
                    false,
                    "host_port_control_does_not_qualify_required_two_hub".to_string(),
                    required_class,
                    observed_class,
                    "host_port_control_only".to_string(),
                    false,
                );
            }
        }
    }

    if is_host_port_control_scope && required_class == TopologyPathClass::RequiredTwoHub {
        let admitted = lifecycle_state == Usb4LifecycleState::Ready && canary.canary_passed;
        let reason = if admitted {
            "admitted_host_port_control_only".to_string()
        } else {
            "no_observed_xdomain".to_string()
        };
        return (
            admitted,
            reason,
            required_class,
            observed_class,
            "host_port_control_only".to_string(),
            false,
        );
    }

    if lifecycle_state == Usb4LifecycleState::Ready && canary.canary_passed {
        let qualified = link.required_topology_qualified.unwrap_or(true)
            && !(required_class == TopologyPathClass::RequiredTwoHub
                && observed_class == TopologyPathClass::HostPortControl);
        let scope = link.qualification_scope.clone().unwrap_or_else(|| {
            if qualified {
                "qualified".to_string()
            } else {
                "host_port_control_only".to_string()
            }
        });
        (
            true,
            "admitted_verified_xdomain_and_canary".to_string(),
            required_class,
            observed_class,
            scope,
            qualified,
        )
    } else {
        (
            false,
            canary
                .failure_reason
                .clone()
                .unwrap_or_else(|| "no_observed_xdomain".to_string()),
            required_class,
            observed_class,
            link.qualification_scope
                .clone()
                .unwrap_or_else(|| "unqualified".to_string()),
            false,
        )
    }
}

/// Returns true if a route prefix is a default route (`0.0.0.0/0`, `::/0`) or outside PC-AI's
/// deterministic P2P pools (`fd00:7462:7032::/48` and `172.31.240.0/20`).
pub fn is_protected_default_or_non_owned_route(prefix: &str) -> bool {
    let norm = prefix.trim().to_ascii_lowercase();
    if norm.is_empty()
        || norm == "0.0.0.0/0"
        || norm == "::/0"
        || norm == "0.0.0.0"
        || norm == "::"
        || norm.starts_with("0.0.0.0/")
        || norm.starts_with("::/0")
        || norm.starts_with("10.254.253.")
        || norm.starts_with("192.168.")
    {
        return true;
    }
    let owned_v6 = norm.starts_with("fd00:7462:7032:");
    let owned_v4 = norm.starts_with("172.31.24") || norm.starts_with("172.31.25");
    !(owned_v6 || owned_v4)
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
    let xdomain_present = adapter_opt.and_then(|a| a.xdomain_peer_unique_id.as_ref()).is_some()
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

    // 4b. No observed XDomain = no link admission / Ready state when link explicitly requires XDomain
    if link.windows_xdomain_present == Some(false) && !xdomain_present {
        transit_reasons.push("NoObservedXDomain=true".to_string());
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

fn bound_route_adapter<'a>(
    link: &PeerLinkLifecycleStatus,
    adapters: &'a [ThunderboltAdapterState],
) -> Result<&'a ThunderboltAdapterState, String> {
    let matches: Vec<_> = adapters
        .iter()
        .filter(|a| a.interface_alias.eq_ignore_ascii_case(&link.local_interface))
        .collect();
    if matches.len() != 1 {
        return Err("Missing or ambiguous current adapter alias".to_string());
    }
    let adapter = matches[0];
    if adapter.interface_index == 0
        || adapters
            .iter()
            .filter(|a| a.interface_index == adapter.interface_index)
            .count()
            != 1
    {
        return Err("Missing or ambiguous current interface index".to_string());
    }
    if adapter.external_owner.is_some() {
        return Err("Adapter has external owner custody".to_string());
    }
    if link.canary.interface_index != Some(adapter.interface_index) {
        return Err("Historical canary interface index does not match current adapter".to_string());
    }
    if link
        .canary
        .interface_alias
        .as_ref()
        .is_some_and(|alias| !alias.eq_ignore_ascii_case(&adapter.interface_alias))
    {
        return Err("Historical canary alias does not match current adapter".to_string());
    }
    Ok(adapter)
}

/// Plans and optionally executes withdrawal of PC-AI-owned stale routes (`/128` ULA in `fd00:7462:7032::/48`
/// and pairwise `/30` or `/32` in `172.31.240.0/20`) for peers whose lifecycle state is not `Ready`,
/// while strictly preserving `millylaptop1` bridge custody, active `dtm-p1gen7<->dtm-p1gen9` pairs,
/// and default routes (`0.0.0.0/0`, `::/0`).
pub fn plan_and_withdraw_stale_owned_routes<R: CommandRunner>(
    runner: &R,
    local_host_canon: &str,
    peer_links: &[PeerLinkLifecycleStatus],
    adapters: &[ThunderboltAdapterState],
    registry_path: Option<&Path>,
    apply: bool,
) -> Vec<StaleRouteWithdrawalAction> {
    let mut actions = Vec::new();
    let mut active_ready_scopes: HashSet<(String, u32)> = HashSet::new();

    for pl in peer_links {
        if pl.lifecycle_state == Usb4LifecycleState::Ready && pl.canary.canary_passed && !pl.evidence.expired {
            if let Ok(adapter) = bound_route_adapter(pl, adapters) {
                active_ready_scopes.insert((pl.peer_host.to_ascii_lowercase(), adapter.interface_index));
            }
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
        if pl.lifecycle_state == Usb4LifecycleState::Ready
            && pl.canary.canary_passed
            && !pl.evidence.expired
            && bound_route_adapter(pl, adapters).is_ok()
        {
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

        // A healthy alternate edge does not own routes on this stale interface.
        // Only preserve overlapping records for the same peer/current interface.
        let binding = bound_route_adapter(pl, adapters);
        let ownership_error = binding.as_ref().err().cloned();
        let target_adapter = binding.ok();
        let target_if_idx = target_adapter.map(|a| a.interface_index).unwrap_or(0);
        let target_if_alias = target_adapter
            .map(|a| a.interface_alias.clone())
            .unwrap_or_else(|| pl.local_interface.clone());
        if target_if_idx > 0 && active_ready_scopes.contains(&(peer_lower.clone(), target_if_idx)) {
            actions.push(StaleRouteWithdrawalAction {
                link_id: pl.link_id.clone(),
                peer_name: pl.peer_host.clone(),
                interface_index: target_if_idx,
                interface_alias: target_if_alias,
                route_prefix: format!("active-interface:{local_host_canon}<->{peer_lower}"),
                address_family: "IPv6+IPv4".to_string(),
                lifecycle_state: pl.lifecycle_state,
                reason: "Preserve routes shared with a ready record on the same current interface".to_string(),
                preserved: true,
                withdrawn: false,
                command_summary: "No-op: preserve ready peer/interface scope".to_string(),
                error: None,
            });
            continue;
        }

        // Resolve PC-AI-owned deterministic ULA and pairwise /30 addresses for the stale/disconnected/asleep peer
        let Ok(peer_id) = resolve_peer_identity(Some(&pl.peer_host), Some(local_host_canon), registry_path) else {
            continue;
        };

        let withdrawal_reason = if let Some(ref inv) = pl.evidence.invalidation_reason {
            inv.clone()
        } else if let Some(ref fail) = pl.canary.failure_reason {
            fail.clone()
        } else {
            format!("Peer '{}' is in '{}' state", pl.peer_host, pl.lifecycle_state.as_str())
        };

        for ula in &peer_id.all_ula_addresses {
            let prefix = format!("{ula}/128");
            if is_protected_default_or_non_owned_route(&prefix) {
                continue;
            }
            let cmd_summary =
                format!("Withdraw stale PC-AI ULA host route {prefix} on {target_if_alias} (ifIndex {target_if_idx})");
            let mut withdrawn = false;
            let mut err_msg = ownership_error.clone();

            if apply {
                if target_if_idx > 0 {
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
                } else {
                    err_msg = Some(format!(
                        "Interface '{}' lacks bound route ownership: {}; refusing unscoped/foreign route deletion",
                        pl.local_interface,
                        ownership_error.as_deref().unwrap_or("unknown interface")
                    ));
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
            let prefix = format!("{}/32", pair_v4.peer_ipv4_address);
            if !is_protected_default_or_non_owned_route(&prefix) {
                // Only withdraw if the local adapter actually holds the stale pairwise /30 IPv4 for this disconnected peer
                let has_stale_v4 = target_if_idx > 0
                    && adapters
                        .iter()
                        .filter(|a| a.interface_index == target_if_idx)
                        .flat_map(|a| a.addresses.iter())
                        .any(|addr| addr.address == pair_v4.local_ipv4_address);

                if has_stale_v4 || !apply {
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
                        withdrawn,
                        command_summary: cmd_summary,
                        error: err_msg,
                    });
                }
            }
        }
    }

    actions
}

#[derive(Debug, Clone, Default)]
pub struct LocalServiceAndRoutePreservationTelemetry {
    pub share_tb_service_running: bool,
    pub tb_service_status: String,
    pub shared_access_service_status: String,
    pub default_routes_preserved: bool,
}

#[cfg(target_os = "windows")]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct RawWindowsPowerAndStandbyPayload {
    #[serde(default)]
    pub powercfg_output: Option<String>,
    #[serde(default)]
    pub event_log_available: bool,
    #[serde(default)]
    pub s0_low_power_idle_supported: bool,
    pub last_entry_506_utc: Option<String>,
    pub last_entry_reason: Option<String>,
    pub last_exit_507_utc: Option<String>,
    pub last_exit_reason: Option<String>,
    #[serde(default)]
    pub recent_cycle_count: usize,
    #[serde(default)]
    pub tb_service_status: Option<String>,
    #[serde(default)]
    pub shared_access_status: Option<String>,
    #[serde(default)]
    pub dock_routers: Vec<String>,
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
    #[serde(default)]
    pub pnp_ancestry: Vec<String>,
}

pub fn collect_modern_standby_and_power_capabilities(
    snapshot: &PlatformThunderboltSnapshot,
) -> (
    ModernStandbyTelemetry,
    Vec<Usb4AdapterPowerCapabilities>,
    LocalServiceAndRoutePreservationTelemetry,
) {
    #[cfg(target_os = "windows")]
    {
        let ps_script = r#"
$ErrorActionPreference = 'SilentlyContinue'
$powercfgOut = (& powercfg.exe /a 2>$null | Out-String)
$evAvailable = $false
$ev = @()
try {
    $ev = @(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power'; Id=506,507} -MaxEvents 12 -ErrorAction Stop)
    $evAvailable = $true
} catch {
    if ($_.Exception.Message -match 'No events were found') {
        $evAvailable = $true
    }
}
$e506 = $ev | Where-Object Id -eq 506 | Select-Object -First 1
$e507 = $ev | Where-Object Id -eq 507 | Select-Object -First 1
$tbSvc = Get-Service -Name 'TBService','ShareTBService' | Select-Object -First 1
$saSvc = Get-Service -Name 'SharedAccess' | Select-Object -First 1
$dockRouters = @(Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like 'USB4\VID_*' } | ForEach-Object { [string]$_.InstanceId })
$pms = @()
foreach ($n in @(Get-NetAdapter -IncludeHidden | Where-Object { $_.InterfaceDescription -match 'USB4|Thunderbolt|P2P' -or $_.Name -match 'Ethernet 4|Ethernet 5|Thunderbolt|USB4' })) {
    $pm = Get-NetAdapterPowerManagement -Name $n.Name
    $ancestry = @()
    $cur = [string]$n.PnPDeviceID
    $depth = 0
    while ($cur -and $depth -lt 10) {
        $ancestry += $cur
        $parentProp = Get-PnpDeviceProperty -InstanceId $cur -KeyName 'DEVPKEY_Device_Parent'
        $cur = if ($parentProp -and $parentProp.Data) { [string]$parentProp.Data } else { $null }
        $depth++
    }
    $pms += [pscustomobject]@{
        InterfaceAlias = [string]$n.Name
        InterfaceDescription = [string]$n.InterfaceDescription
        SelectiveSuspend = if ($pm) { [string]$pm.SelectiveSuspend } else { 'Enabled' }
        WakeOnMagicPacket = if ($pm) { [string]$pm.WakeOnMagicPacket } else { 'Unsupported' }
        WakeOnPattern = if ($pm) { [string]$pm.WakeOnPattern } else { 'Unsupported' }
        ArpOffload = if ($pm) { [string]$pm.ArpOffload } else { 'Unsupported' }
        NsOffload = if ($pm) { [string]$pm.NSOffload } else { 'Unsupported' }
        DeviceSleepOnDisconnect = if ($pm) { [string]$pm.DeviceSleepOnDisconnect } else { 'Inactive' }
        PnpAncestry = $ancestry
    }
}
[pscustomobject]@{
    PowercfgOutput = $powercfgOut
    EventLogAvailable = $evAvailable
    S0LowPowerIdleSupported = ($ev.Count -gt 0)
    S0NetworkConnectedSupported = ($ev.Count -gt 0)
    LastEntry506Utc = if ($e506) { $e506.TimeCreated.ToUniversalTime().ToString('o') } else { $null }
    LastEntryReason = if ($e506) { ($e506.Message -replace '\s+', ' ').Trim() } else { $null }
    LastExit507Utc = if ($e507) { $e507.TimeCreated.ToUniversalTime().ToString('o') } else { $null }
    LastExitReason = if ($e507) { ($e507.Message -replace '\s+', ' ').Trim() } else { $null }
    RecentCycleCount = [int](@($ev | Where-Object Id -eq 506).Count)
    TbServiceStatus = if ($tbSvc) { [string]$tbSvc.Status } else { 'NotInstalled' }
    SharedAccessStatus = if ($saSvc) { [string]$saSvc.Status } else { 'NotInstalled' }
    DockRouters = $dockRouters
    Adapters = $pms
} | ConvertTo-Json -Depth 5 -Compress
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
                    let has_events = parsed.last_entry_506_utc.is_some()
                        || parsed.last_exit_507_utc.is_some()
                        || parsed.s0_low_power_idle_supported;
                    let (s0_state, s0_supported, s0_connected_opt) = parse_powercfg_s0_capabilities(
                        parsed.powercfg_output.as_deref(),
                        parsed.event_log_available,
                        has_events,
                    );
                    let standby = ModernStandbyTelemetry {
                        s0_low_power_idle_state: s0_state,
                        s0_low_power_idle_supported: s0_supported,
                        s0_low_power_idle_connected_supported: s0_connected_opt,
                        s0_network_connected_supported: s0_connected_opt.unwrap_or(false),
                        event_log_available: parsed.event_log_available,
                        last_standby_entry_506_utc: parsed.last_entry_506_utc,
                        last_standby_entry_reason: parsed.last_entry_reason,
                        last_standby_exit_507_utc: parsed.last_exit_507_utc,
                        last_standby_exit_reason: parsed.last_exit_reason,
                        currently_in_modern_standby: in_standby,
                        recent_standby_cycle_count: parsed.recent_cycle_count,
                    };

                    let tb_status = parsed.tb_service_status.unwrap_or_else(|| "Unknown".to_string());
                    let sa_status = parsed.shared_access_status.unwrap_or_else(|| "Unknown".to_string());
                    let svc_telemetry = LocalServiceAndRoutePreservationTelemetry {
                        share_tb_service_running: tb_status.eq_ignore_ascii_case("Running")
                            || sa_status.eq_ignore_ascii_case("Running"),
                        tb_service_status: tb_status,
                        shared_access_service_status: sa_status,
                        default_routes_preserved: true,
                    };

                    let mut caps = Vec::new();
                    for adapter in &snapshot.adapters {
                        let matched = parsed
                            .adapters
                            .iter()
                            .find(|p| p.interface_alias.eq_ignore_ascii_case(&adapter.interface_alias));
                        let speed_bps = if adapter.transmit_link_speed_bps > 0 {
                            adapter.transmit_link_speed_bps
                        } else {
                            20_000_000_000
                        };
                        if let Some(pm) = matched {
                            let sel_susp = pm
                                .selective_suspend
                                .as_deref()
                                .is_some_and(|s| s.trim().eq_ignore_ascii_case("Enabled") || s.trim() == "1");
                            let base_cap = Usb4AdapterPowerCapabilities::evaluate(
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
                            );
                            caps.push(base_cap.with_pnp_and_link_speed(
                                pm.pnp_ancestry.clone(),
                                &parsed.dock_routers,
                                speed_bps,
                            ));
                        } else {
                            caps.push(
                                Usb4AdapterPowerCapabilities::evaluate(
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
                                )
                                .with_pnp_and_link_speed(
                                    Vec::new(),
                                    &parsed.dock_routers,
                                    speed_bps,
                                ),
                            );
                        }
                    }
                    return (standby, caps, svc_telemetry);
                }
            }
        }
    }

    let standby = ModernStandbyTelemetry {
        s0_low_power_idle_state: "Unknown".to_string(),
        ..Default::default()
    };
    let svc_telemetry = LocalServiceAndRoutePreservationTelemetry {
        share_tb_service_running: true,
        tb_service_status: "Preserved".to_string(),
        shared_access_service_status: "Preserved".to_string(),
        default_routes_preserved: true,
    };
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
    (standby, caps, svc_telemetry)
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
        lifecycle_policy: Some(Usb4LifecyclePolicy::default()),
        ipam_allocations: None,
        sfp_switch_topologies: Vec::new(),
        topology_links: Vec::new(),
        peers: None,
        fleet_peers: Vec::new(),
    });

    let policy = reg
        .usb4_lifecycle_policy
        .clone()
        .or_else(|| reg.lifecycle_policy.clone())
        .unwrap_or_default();
    let local_canon = canonicalize_peer_name(&raw_host, &reg.fleet_peers);
    let local_id = resolve_peer_identity(Some(&local_canon), None, registry_path)?;
    let snapshot = collect_platform_snapshot(true)?;
    let (standby, power_caps, svc_telemetry) = collect_modern_standby_and_power_capabilities(&snapshot);

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
                scope_id: None,
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

        let link_downstream_none = link
            .downstream_link_status
            .as_deref()
            .is_some_and(|s| s.trim().eq_ignore_ascii_case("none") || s.trim().eq_ignore_ascii_case("down"));
        let xdomain_present = if link.windows_xdomain_present == Some(false) || link_downstream_none {
            false
        } else {
            adapter_opt.is_some_and(|a| a.operational_status.trim().eq_ignore_ascii_case("Up"))
                && (adapter_opt.and_then(|a| a.xdomain_peer_unique_id.as_ref()).is_some()
                    || link.windows_xdomain_present.unwrap_or(false)
                    || (state == Usb4LifecycleState::Ready && power_opt.is_some_and(|p| p.pnp_ancestry.len() >= 2)))
        };

        let (
            link_admitted,
            admission_reason,
            required_topology_class,
            observed_topology_class,
            qualification_scope,
            required_topology_qualified,
        ) = evaluate_link_admission_and_qualification(link, state, adapter_opt, power_opt, xdomain_present, &canary);

        let pnp_ancestry = power_opt.map(|p| p.pnp_ancestry.clone()).unwrap_or_default();
        let root_router_id = power_opt
            .and_then(|p| p.root_router_id.clone())
            .or_else(|| link.expected_local_root_router_id.clone());
        let pci_host_router_id = power_opt.and_then(|p| p.pci_host_router_id.clone());
        let intermediate_dock_routers = power_opt
            .map(|p| p.intermediate_dock_routers.clone())
            .unwrap_or_default();
        let windows_reported_link_speed_bps = link
            .windows_reported_link_speed_bps
            .or_else(|| power_opt.map(|p| p.windows_reported_link_speed_bps))
            .unwrap_or(20_000_000_000);
        let link_speed_is_nominal_driver_display = link.link_speed_is_nominal_driver_display.unwrap_or(true);
        let link_speed_is_hardware_maximum = link.link_speed_is_hardware_maximum.unwrap_or(false);
        let measured_payload_gbps_samples = link.measured_payload_gbps_samples.clone();

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
            link_admitted,
            admission_reason,
            required_topology_class,
            observed_topology_class,
            qualification_scope,
            required_topology_qualified,
            pnp_ancestry,
            root_router_id,
            pci_host_router_id,
            intermediate_dock_routers,
            local_root_router_fw: link.local_root_router_fw.clone(),
            dock_model: link.dock_model.clone(),
            dock_serial: link.dock_serial.clone(),
            dock_status: link.dock_status.clone(),
            dock_fw_package: link.dock_fw_package.clone(),
            dock_fw_target: link.dock_fw_target.clone(),
            dock_pd_fw: link.dock_pd_fw.clone(),
            dock_tbt5_nvm: link.dock_tbt5_nvm.clone(),
            dock_dmc_fw: link.dock_dmc_fw.clone(),
            windows_reported_link_speed_bps,
            link_speed_is_nominal_driver_display,
            link_speed_is_hardware_maximum,
            measured_payload_gbps_samples,
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
    let default_routes_preserved = svc_telemetry.default_routes_preserved
        && stale_route_withdrawals
            .iter()
            .filter(|w| !w.preserved)
            .all(|w| !is_protected_default_or_non_owned_route(&w.route_prefix));

    Ok(Usb4LifecycleReport {
        schema: "pcai.usb4-lifecycle/v1".to_string(),
        computer_name: local_canon,
        os_family: std::env::consts::OS.to_string(),
        architecture: std::env::consts::ARCH.to_string(),
        evaluated_at_utc: now.to_rfc3339(),
        apply_mode: apply,
        policy,
        modern_standby: standby,
        share_tb_service_running: svc_telemetry.share_tb_service_running,
        tb_service_status: svc_telemetry.tb_service_status,
        shared_access_service_status: svc_telemetry.shared_access_service_status,
        default_routes_preserved,
        adapter_power_capabilities: power_caps,
        peer_link_states,
        stale_route_withdrawals,
        resume_reprobe,
        overall_valid: no_false_transit_claims && !withdrawal_errors && default_routes_preserved,
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
            self.calls
                .lock()
                .unwrap()
                .push((program.to_string(), args.iter().map(|s| (*s).to_string()).collect()));
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
        assert_eq!(cap.windows_reported_link_speed_bps, 20_000_000_000);
        assert!(cap.link_speed_is_nominal_driver_display);
        assert!(!cap.link_speed_is_hardware_maximum);
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
        assert!(bad_mac_res.failure_reason.as_deref().unwrap().contains("MAC mismatch"));

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
    fn test_no_observed_xdomain_refuses_link_admission_and_host_port_control_never_qualifies_two_hub() {
        let adapter = sample_gen9_adapter();
        let dock_routers = vec!["USB4\\VID_8087&PID_5786\\8&36E6FB2D&0&1".to_string()];

        // 1. Dock-routed adapter (PID_5781 JHL9580 FW 67.67 + Lenovo 40BA dock 1S40BAYJ0019GD FW 1.0.07)
        let dock_power = Usb4AdapterPowerCapabilities::evaluate(
            16,
            "Ethernet 4",
            "USB4(TM) P2P Network Adapter #2",
            true,
            "Unsupported",
            "Unsupported",
            "Unsupported",
            "Unsupported",
            "Inactive",
            "Up",
        )
        .with_pnp_and_link_speed(
            vec![
                "USB4\\VEN_045E&DEV_0000083F&PROT_USB4NET\\8&36E6FB2D&0&3".to_string(),
                "USB4\\ROOT_DEVICE_ROUTER&VID_8086&PID_5781\\7&290E0051&0&0".to_string(),
                "PCI\\VEN_8086&DEV_5781&SUBSYS_234E17AA&REV_02&USB4_MS_CM\\6&52F7B01&0&00E001F0".to_string(),
            ],
            &dock_routers,
            20_000_000_000,
        );
        assert_eq!(
            dock_power.root_router_id.as_deref(),
            Some("USB4\\ROOT_DEVICE_ROUTER&VID_8086&PID_5781\\7&290E0051&0&0")
        );
        assert_eq!(
            dock_power.intermediate_dock_routers,
            vec!["USB4\\VID_8087&PID_5786\\8&36E6FB2D&0&1".to_string()]
        );

        // 2. Host-port control adapter (PID_E333 CPU USB4 Host Router FW 11.7, Ethernet 5 ifIndex=62)
        let hostport_power = Usb4AdapterPowerCapabilities::evaluate(
            62,
            "Ethernet 5",
            "USB4(TM) P2P Network Adapter #3",
            true,
            "Unsupported",
            "Unsupported",
            "Unsupported",
            "Unsupported",
            "Inactive",
            "Up",
        )
        .with_pnp_and_link_speed(
            vec![
                "USB4\\VEN_1D6B&DEV_00000004&PROT_USB4NET\\5&19C8806A&0&1".to_string(),
                "USB4\\ROOT_DEVICE_ROUTER&VID_8086&PID_E333\\4&E448B37&0&0".to_string(),
                "PCI\\VEN_8086&DEV_E333&SUBSYS_235C17AA&REV_00&USB4_MS_CM\\3&11583659&0&6A".to_string(),
            ],
            &dock_routers,
            20_000_000_000,
        );
        assert_eq!(
            hostport_power.root_router_id.as_deref(),
            Some("USB4\\ROOT_DEVICE_ROUTER&VID_8086&PID_E333\\4&E448B37&0&0")
        );
        assert!(hostport_power.intermediate_dock_routers.is_empty());
        assert_eq!(hostport_power.topology_path_class, TopologyPathClass::HostPortControl);

        // Case 2: No observed XDomain on ms01-plugable-dock-usb4-routed -> link_admitted=false, admission_reason="no_observed_xdomain"
        let ms01_plugable_link = TopologyLinkSpec {
            link_id: "ms01-plugable-dock-usb4-routed".to_string(),
            source_host: "minisforum".to_string(),
            source_interface: "domain1/1-1".to_string(),
            target_host: "dtm-p1gen9".to_string(),
            target_interface: "Ethernet 4".to_string(),
            link_intent: "routed-p2p".to_string(),
            downstream_link_status: Some("none".to_string()),
            windows_xdomain_present: Some(false),
            required_topology_class: Some(TopologyPathClass::RequiredTwoHub),
            expected_local_root_router_id: Some(
                "USB4\\ROOT_DEVICE_ROUTER&VID_8086&PID_5781\\7&290E0051&0&0".to_string(),
            ),
            expected_local_dock_router_id: Some("USB4\\VID_8087&PID_5786\\8&36E6FB2D&0&1".to_string()),
            local_root_router_fw: Some("67.67".to_string()),
            dock_model: Some("40BA".to_string()),
            dock_serial: Some("1S40BAYJ0019GD".to_string()),
            dock_fw_package: Some("1.0.07".to_string()),
            dock_fw_target: Some("1.2.99".to_string()),
            dock_pd_fw: Some("0.0.47".to_string()),
            dock_tbt5_nvm: Some("62.62".to_string()),
            dock_dmc_fw: Some("0.6.5".to_string()),
            ..Default::default()
        };
        let failed_canary = IdentityBoundCanaryResult {
            expected_peer: "minisforum".to_string(),
            observed_peer: None,
            interface_index: None,
            interface_alias: Some("Ethernet 4".to_string()),
            target_address: None,
            scope_id: None,
            handshake_method: None,
            observed_mac: None,
            probe_latency_us: None,
            canary_passed: false,
            failure_reason: Some("No observed XDomain".to_string()),
        };
        let (admitted, reason, req_class, _, qual_scope, qualified) = evaluate_link_admission_and_qualification(
            &ms01_plugable_link,
            Usb4LifecycleState::Disconnected,
            Some(&adapter),
            Some(&dock_power),
            false,
            &failed_canary,
        );
        assert!(!admitted);
        assert_eq!(reason, "no_observed_xdomain");
        assert_eq!(req_class, TopologyPathClass::RequiredTwoHub);
        assert_eq!(qual_scope, "unqualified_no_xdomain");
        assert!(!qualified);

        // Case 3: Active host_port_control link (PID_E333) when required_topology_class is RequiredTwoHub
        // must report qualification_scope="host_port_control_only" and required_topology_qualified=false
        let hostport_link = TopologyLinkSpec {
            link_id: "ms01-plugable-gen9-hostport-control".to_string(),
            source_host: "minisforum".to_string(),
            source_interface: "usb4-win".to_string(),
            target_host: "dtm-p1gen9".to_string(),
            target_interface: "Ethernet 5".to_string(),
            link_intent: "routed-p2p".to_string(),
            required_topology_class: Some(TopologyPathClass::RequiredTwoHub),
            observed_topology_class: Some(TopologyPathClass::HostPortControl),
            qualification_scope: Some("host_port_control_only".to_string()),
            required_topology_qualified: Some(false),
            expected_local_root_router_id: Some(
                "USB4\\ROOT_DEVICE_ROUTER&VID_8086&PID_E333\\4&E448B37&0&0".to_string(),
            ),
            local_root_router_fw: Some("11.7".to_string()),
            measured_payload_gbps_samples: vec![7.86, 12.90, 14.12],
            ..Default::default()
        };
        let passed_hostport_canary = IdentityBoundCanaryResult {
            expected_peer: "minisforum".to_string(),
            observed_peer: Some("minisforum".to_string()),
            interface_index: Some(62),
            interface_alias: Some("Ethernet 5".to_string()),
            target_address: Some("fe80::90d7:68ff:fe33:c6c4%62".to_string()),
            scope_id: Some(62),
            handshake_method: Some("SSH".to_string()),
            observed_mac: Some("92-D7-68-33-C6-C4".to_string()),
            probe_latency_us: Some(1200),
            canary_passed: true,
            failure_reason: None,
        };
        let (hp_admitted, hp_reason, hp_req, hp_obs, hp_scope, hp_qualified) =
            evaluate_link_admission_and_qualification(
                &hostport_link,
                Usb4LifecycleState::Ready,
                Some(&adapter),
                Some(&hostport_power),
                true,
                &passed_hostport_canary,
            );
        assert!(hp_admitted);
        assert_eq!(hp_reason, "admitted_host_port_control_only");
        assert_eq!(hp_req, TopologyPathClass::RequiredTwoHub);
        assert_eq!(hp_obs, TopologyPathClass::HostPortControl);
        assert_eq!(hp_scope, "host_port_control_only");
        assert!(
            !hp_qualified,
            "host_port_control must never qualify required_two_hub topology"
        );
    }

    #[test]
    fn test_protected_default_route_guard_never_withdraws_default_or_lan_routes() {
        assert!(is_protected_default_or_non_owned_route("0.0.0.0/0"));
        assert!(is_protected_default_or_non_owned_route("::/0"));
        assert!(is_protected_default_or_non_owned_route("10.254.253.0/30"));
        assert!(is_protected_default_or_non_owned_route("192.168.50.0/24"));
        assert!(!is_protected_default_or_non_owned_route("fd00:7462:7032::1/128"));
        assert!(!is_protected_default_or_non_owned_route("172.31.241.249/32"));
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
            s0_low_power_idle_state: "Supported".to_string(),
            s0_low_power_idle_supported: true,
            s0_low_power_idle_connected_supported: Some(true),
            s0_network_connected_supported: true,
            event_log_available: true,
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
            s0_low_power_idle_state: "Supported".to_string(),
            s0_low_power_idle_supported: true,
            s0_low_power_idle_connected_supported: Some(true),
            s0_network_connected_supported: true,
            event_log_available: true,
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
            scope_id: Some(22),
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

        let mut ms01_canary =
            evaluate_identity_bound_canary("minisforum", Some("Ethernet 4"), &empty_macs, &[], &known);
        // Owned stale-route fixture retains historical interface binding even though
        // the current identity/reachability canary fails. Missing proof is tested separately.
        ms01_canary.interface_index = Some(22);
        ms01_canary.interface_alias = Some("Ethernet 4".to_string());
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
            scope_id: None,
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
                windows_xdomain_present: true,
                link_admitted: true,
                admission_reason: "admitted_verified_xdomain_and_canary".to_string(),
                required_topology_class: TopologyPathClass::SingleHubChain,
                observed_topology_class: TopologyPathClass::SingleHubChain,
                qualification_scope: "active_p2p_pair_qualified".to_string(),
                required_topology_qualified: true,
                pnp_ancestry: Vec::new(),
                root_router_id: Some("USB4\\ROOT_DEVICE_ROUTER&VID_8086&PID_5781\\7&290E0051&0&0".to_string()),
                pci_host_router_id: None,
                intermediate_dock_routers: vec!["USB4\\VID_8087&PID_5786\\8&36E6FB2D&0&1".to_string()],
                local_root_router_fw: Some("67.67".to_string()),
                dock_model: Some("40BA".to_string()),
                dock_serial: Some("1S40BAYJ0019GD".to_string()),
                dock_status: Some("Factory (0x0)".to_string()),
                dock_fw_package: Some("1.0.07".to_string()),
                dock_fw_target: Some("1.2.99".to_string()),
                dock_pd_fw: Some("0.0.47".to_string()),
                dock_tbt5_nvm: Some("62.62".to_string()),
                dock_dmc_fw: Some("0.6.5".to_string()),
                windows_reported_link_speed_bps: 20_000_000_000,
                link_speed_is_nominal_driver_display: true,
                link_speed_is_hardware_maximum: false,
                measured_payload_gbps_samples: Vec::new(),
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
                link_admitted: false,
                admission_reason: "no_observed_xdomain".to_string(),
                required_topology_class: TopologyPathClass::RequiredTwoHub,
                observed_topology_class: TopologyPathClass::Unknown,
                qualification_scope: "unqualified_no_xdomain".to_string(),
                required_topology_qualified: false,
                pnp_ancestry: Vec::new(),
                root_router_id: Some("USB4\\ROOT_DEVICE_ROUTER&VID_8086&PID_5781\\7&290E0051&0&0".to_string()),
                pci_host_router_id: None,
                intermediate_dock_routers: Vec::new(),
                local_root_router_fw: Some("67.67".to_string()),
                dock_model: Some("40BA".to_string()),
                dock_serial: Some("1S40BAYJ0019GD".to_string()),
                dock_status: Some("Factory (0x0)".to_string()),
                dock_fw_package: Some("1.0.07".to_string()),
                dock_fw_target: Some("1.2.99".to_string()),
                dock_pd_fw: Some("0.0.47".to_string()),
                dock_tbt5_nvm: Some("62.62".to_string()),
                dock_dmc_fw: Some("0.6.5".to_string()),
                windows_reported_link_speed_bps: 20_000_000_000,
                link_speed_is_nominal_driver_display: true,
                link_speed_is_hardware_maximum: false,
                measured_payload_gbps_samples: Vec::new(),
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
                link_admitted: true,
                admission_reason: "external_bridge_custody_preserved".to_string(),
                required_topology_class: TopologyPathClass::Unknown,
                observed_topology_class: TopologyPathClass::Unknown,
                qualification_scope: "external_bridge_custody".to_string(),
                required_topology_qualified: true,
                pnp_ancestry: Vec::new(),
                root_router_id: None,
                pci_host_router_id: None,
                intermediate_dock_routers: Vec::new(),
                local_root_router_fw: None,
                dock_model: None,
                dock_serial: None,
                dock_status: None,
                dock_fw_package: None,
                dock_fw_target: None,
                dock_pd_fw: None,
                dock_tbt5_nvm: None,
                dock_dmc_fw: None,
                windows_reported_link_speed_bps: 20_000_000_000,
                link_speed_is_nominal_driver_display: true,
                link_speed_is_hardware_maximum: false,
                measured_payload_gbps_samples: Vec::new(),
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

    struct FailingRunner;
    impl CommandRunner for FailingRunner {
        fn run(&self, _program: &str, _args: &[&str]) -> (bool, String) {
            (false, "Access is denied.".to_string())
        }
    }

    fn stale_withdrawal_fixture(interface: &str) -> PeerLinkLifecycleStatus {
        let t0 = Utc.with_ymd_and_hms(2026, 10, 16, 20, 4, 0).unwrap();
        let ev_fresh = ExpiringLifecycleEvidence::evaluate(t0, t0, 300, None, false);
        PeerLinkLifecycleStatus {
            link_id: "ms01-foreign-if".to_string(),
            local_host: "dtm-p1gen9".to_string(),
            local_interface: interface.to_string(),
            peer_host: "minisforum".to_string(),
            peer_interface: "usb4-win".to_string(),
            link_intent: "routed-p2p".to_string(),
            lifecycle_state: Usb4LifecycleState::Disconnected,
            always_on_transit_capable: false,
            transit_prohibition_reasons: Vec::new(),
            preserve_existing_pair: false,
            preserve_bridge_custody: false,
            downstream_link_status: Some("none".to_string()),
            windows_xdomain_present: false,
            link_admitted: false,
            admission_reason: "no_observed_xdomain".to_string(),
            required_topology_class: TopologyPathClass::RequiredTwoHub,
            observed_topology_class: TopologyPathClass::Unknown,
            qualification_scope: "unqualified_no_xdomain".to_string(),
            required_topology_qualified: false,
            pnp_ancestry: Vec::new(),
            root_router_id: None,
            pci_host_router_id: None,
            intermediate_dock_routers: Vec::new(),
            local_root_router_fw: None,
            dock_model: None,
            dock_serial: None,
            dock_status: None,
            dock_fw_package: None,
            dock_fw_target: None,
            dock_pd_fw: None,
            dock_tbt5_nvm: None,
            dock_dmc_fw: None,
            windows_reported_link_speed_bps: 20_000_000_000,
            link_speed_is_nominal_driver_display: true,
            link_speed_is_hardware_maximum: false,
            measured_payload_gbps_samples: Vec::new(),
            evidence: ev_fresh.clone(),
            canary: IdentityBoundCanaryResult {
                expected_peer: "minisforum".to_string(),
                observed_peer: None,
                interface_index: None,
                interface_alias: Some(interface.to_string()),
                target_address: None,
                scope_id: None,
                handshake_method: None,
                observed_mac: None,
                probe_latency_us: None,
                canary_passed: false,
                failure_reason: Some("Disconnected".to_string()),
            },
        }
    }

    #[test]
    fn test_scoped_route_withdrawal_refuses_foreign_interface_and_reports_deletion_failure() {
        let known = sample_known_peers();
        let adapter_eth4 = sample_gen9_adapter(); // Ethernet 4, ifIndex 22

        // Link bound to Ethernet 99 (not present in adapters): must NOT fall back to Ethernet 4 (ifIndex 22)
        let foreign_link = stale_withdrawal_fixture("Ethernet 99");

        let dir = tempfile::tempdir().expect("tempdir");
        let reg_path = dir.path().join("peers.json");
        let reg_json = serde_json::json!({
            "schemaVersion": 1,
            "fleetPeers": known
        });
        std::fs::write(&reg_path, serde_json::to_string(&reg_json).unwrap()).unwrap();

        let runner = MockRunner::new();
        let foreign_withdrawals = plan_and_withdraw_stale_owned_routes(
            &runner,
            "dtm-p1gen9",
            std::slice::from_ref(&foreign_link),
            std::slice::from_ref(&adapter_eth4),
            Some(&reg_path),
            true,
        );
        assert!(
            runner.calls.lock().unwrap().is_empty(),
            "Must never invoke route deletion on foreign interface Ethernet 4 when Ethernet 99 is missing"
        );
        assert!(foreign_withdrawals
            .iter()
            .all(|w| w.interface_index == 0 && !w.withdrawn && w.error.is_some()));

        // Now test matching interface (Ethernet 4, ifIndex 22) when route deletion fails
        let mut eth4_link = foreign_link;
        eth4_link.local_interface = "Ethernet 4".to_string();
        eth4_link.canary.interface_index = Some(22);
        eth4_link.canary.interface_alias = Some("Ethernet 4".to_string());
        let failing_runner = FailingRunner;
        let failed_withdrawals = plan_and_withdraw_stale_owned_routes(
            &failing_runner,
            "dtm-p1gen9",
            std::slice::from_ref(&eth4_link),
            std::slice::from_ref(&adapter_eth4),
            Some(&reg_path),
            true,
        );
        assert!(failed_withdrawals
            .iter()
            .any(|w| w.interface_index == 22 && !w.withdrawn && w.error.as_deref() == Some("Access is denied.")));
    }

    fn ready_withdrawal_fixture() -> PeerLinkLifecycleStatus {
        let mut link = stale_withdrawal_fixture("Ethernet 4");
        link.link_id = "ready-edge-a".to_string();
        link.lifecycle_state = Usb4LifecycleState::Ready;
        link.canary.canary_passed = true;
        link.canary.interface_index = Some(22);
        link.canary.interface_alias = Some("Ethernet 4".to_string());
        link.canary.failure_reason = None;
        link
    }

    fn withdrawal_registry() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("temp registry");
        let path = dir.path().join("peers.json");
        std::fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({
                "schemaVersion": 1, "fleetPeers": sample_known_peers()
            }))
            .unwrap(),
        )
        .unwrap();
        (dir, path)
    }

    // Protects alternate-edge cleanup; detects peer-wide Ready suppressing stale edge withdrawal.
    // Needs recording CommandRunner only; breadcrumb: plan_and_withdraw_stale_owned_routes.
    #[test]
    fn ready_peer_on_edge_a_does_not_preserve_stale_edge_b_routes() {
        let ready = ready_withdrawal_fixture();
        let mut stale = stale_withdrawal_fixture("Ethernet 5");
        stale.link_id = "stale-edge-b".to_string();
        stale.canary.interface_index = Some(33);
        stale.canary.interface_alias = Some("Ethernet 5".to_string());
        let a = sample_gen9_adapter();
        let mut b = a.clone();
        b.interface_index = 33;
        b.interface_alias = "Ethernet 5".to_string();
        b.addresses.clear();
        let (_dir, registry) = withdrawal_registry();
        let peer = resolve_peer_identity(Some("minisforum"), Some("dtm-p1gen9"), Some(&registry)).unwrap();
        let runner = MockRunner::new();
        let actions = plan_and_withdraw_stale_owned_routes(
            &runner,
            "dtm-p1gen9",
            &[ready, stale],
            &[a, b],
            Some(&registry),
            true,
        );
        let calls = runner.calls.lock().unwrap();
        assert_eq!(
            calls.len(),
            peer.all_ula_addresses.len(),
            "Every stale ULA route must be removed only on edge B"
        );
        for (program, args) in calls.iter() {
            #[cfg(not(target_os = "windows"))]
            {
                assert_eq!(program, "ip");
                assert_eq!(&args[..3], &["-6", "route", "del"]);
                assert_eq!(&args[4..], &["dev", "Ethernet 5"]);
                assert!(peer.all_ula_addresses.iter().any(|ula| args[3] == format!("{ula}/128")));
            }
            #[cfg(target_os = "windows")]
            {
                assert_eq!(program, "netsh");
                assert_eq!(&args[..4], &["interface", "ipv6", "delete", "route"]);
                assert_eq!(&args[5..], &["33", "::"]);
                assert!(peer.all_ula_addresses.iter().any(|ula| args[4] == format!("{ula}/128")));
            }
        }
        assert!(actions.iter().any(|a| a.link_id == "ready-edge-a" && a.preserved));
        assert!(actions
            .iter()
            .filter(|a| a.withdrawn)
            .all(|a| a.interface_index == 33 && a.interface_alias == "Ethernet 5"));
    }

    // Protects a ready route shared by duplicate link records; detects deleting its current interface.
    // Needs recording runner, no hardware; breadcrumb: active_ready_scopes membership.
    #[test]
    fn stale_duplicate_on_ready_peer_interface_never_deletes_routes() {
        let ready = ready_withdrawal_fixture();
        let mut stale = stale_withdrawal_fixture("Ethernet 4");
        stale.canary.interface_index = Some(22);
        let (_dir, registry) = withdrawal_registry();
        let runner = MockRunner::new();
        let actions = plan_and_withdraw_stale_owned_routes(
            &runner,
            "dtm-p1gen9",
            &[ready, stale],
            &[sample_gen9_adapter()],
            Some(&registry),
            true,
        );
        assert!(runner.calls.lock().unwrap().is_empty());
        assert_eq!(actions.len(), 2);
        assert!(actions.iter().all(|a| a.preserved && !a.withdrawn));
    }

    // Protects fail-closed interface ownership after unplug; detects unscoped/fallback deletion.
    // Needs recording runner, no hardware; breadcrumb: target_adapter lookup and index-zero guard.
    #[test]
    fn stale_alternate_without_current_adapter_reports_refusal_and_no_delete() {
        let ready = ready_withdrawal_fixture();
        let stale = stale_withdrawal_fixture("Ethernet 99");
        let (_dir, registry) = withdrawal_registry();
        let runner = MockRunner::new();
        let actions = plan_and_withdraw_stale_owned_routes(
            &runner,
            "dtm-p1gen9",
            &[ready, stale],
            &[sample_gen9_adapter()],
            Some(&registry),
            true,
        );
        assert!(runner.calls.lock().unwrap().is_empty());
        let refused: Vec<_> = actions.iter().filter(|a| !a.preserved).collect();
        assert!(
            !refused.is_empty(),
            "Unknown adapter must report explicit refusal, not peer-wide silent skip"
        );
        assert!(refused
            .iter()
            .all(|a| a.interface_index == 0 && !a.withdrawn && a.error.is_some()));
    }

    // Protects historical/current custody; detects alias reuse, absent proof, external ownership
    // and ambiguous snapshots becoming authority. Needs recording runner only; breadcrumb:
    // bound_route_adapter and plan_and_withdraw_stale_owned_routes.
    #[test]
    fn unbound_or_external_stale_interfaces_never_issue_deletion_commands() {
        let (_dir, registry) = withdrawal_registry();
        for scenario in [
            "mismatched-index",
            "missing-proof",
            "wrong-canary-alias",
            "external-owner",
            "duplicate-alias",
            "duplicate-index",
            "zero-index",
        ] {
            let mut stale = stale_withdrawal_fixture("Ethernet 4");
            stale.canary.interface_index = Some(22);
            let mut adapters = vec![sample_gen9_adapter()];
            match scenario {
                "mismatched-index" => stale.canary.interface_index = Some(7),
                "missing-proof" => stale.canary.interface_index = None,
                "wrong-canary-alias" => stale.canary.interface_alias = Some("foreign".to_string()),
                "external-owner" => adapters[0].external_owner = Some("ms01-usb4-extension".to_string()),
                "duplicate-alias" => {
                    let mut other = adapters[0].clone();
                    other.interface_index = 33;
                    adapters.push(other);
                }
                "duplicate-index" => {
                    let mut other = adapters[0].clone();
                    other.interface_alias = "foreign".to_string();
                    adapters.push(other);
                }
                "zero-index" => {
                    adapters[0].interface_index = 0;
                    stale.canary.interface_index = Some(0);
                }
                _ => unreachable!(),
            }
            let runner = MockRunner::new();
            let actions =
                plan_and_withdraw_stale_owned_routes(&runner, "dtm-p1gen9", &[stale], &adapters, Some(&registry), true);
            assert!(
                runner.calls.lock().unwrap().is_empty(),
                "Unsafe scenario issued deletion: {scenario}"
            );
            assert!(!actions.is_empty(), "Unsafe scenario omitted its refusal: {scenario}");
            assert!(
                actions
                    .iter()
                    .all(|a| !a.withdrawn && a.interface_index == 0 && a.error.is_some()),
                "Unsafe scenario admitted ownership: {scenario}"
            );
        }
    }

    // Protects current ownership from stale ready evidence after alias/index reuse.
    // Detects an unbound ready record preserving a different current interface.
    // Needs recording runner only; breadcrumb: active_ready_scopes and bound_route_adapter.
    #[test]
    fn ready_record_with_reused_alias_cannot_preserve_a_different_current_interface() {
        let ready = ready_withdrawal_fixture(); // historical index22
        let mut stale = stale_withdrawal_fixture("Ethernet 4");
        stale.canary.interface_index = Some(44);
        let mut adapter = sample_gen9_adapter();
        adapter.interface_index = 44;
        adapter.addresses.clear();
        let (_dir, registry) = withdrawal_registry();
        let runner = MockRunner::new();
        let actions = plan_and_withdraw_stale_owned_routes(
            &runner,
            "dtm-p1gen9",
            &[ready, stale],
            &[adapter],
            Some(&registry),
            true,
        );
        assert!(!runner.calls.lock().unwrap().is_empty());
        assert!(actions
            .iter()
            .any(|a| a.interface_index == 0 && a.error.is_some() && !a.withdrawn));
        assert!(actions.iter().filter(|a| a.withdrawn).all(|a| a.interface_index == 44));
    }

    #[test]
    fn test_scoped_ipv6_link_local_scope_id_preserved_across_interfaces() {
        let (addr62, scope62) = extract_ipv6_scope_id("fe80::90d7:68ff:fe33:c6c4%62", None);
        assert_eq!(addr62, "fe80::90d7:68ff:fe33:c6c4%62");
        assert_eq!(scope62, Some(62));

        let (addr22, scope22) = extract_ipv6_scope_id("fe80::90d7:68ff:fe33:c6c4", Some(22));
        assert_eq!(addr22, "fe80::90d7:68ff:fe33:c6c4%22");
        assert_eq!(scope22, Some(22));
        assert_ne!(scope62, scope22);
    }

    #[test]
    fn test_parse_powercfg_s0_capabilities_supported_unsupported_and_unknown() {
        let supported_out = "The following sleep states are available on this system:\n    Standby (S0 Low Power Idle) Network Connected\nThe following sleep states are not available on this system:\n    Standby (S3)";
        let (st1, sup1, conn1) = parse_powercfg_s0_capabilities(Some(supported_out), true, true);
        assert_eq!(st1, "Supported");
        assert!(sup1);
        assert_eq!(conn1, Some(true));

        let unsupported_out = "The following sleep states are available on this system:\n    Standby (S3)\nThe following sleep states are not available on this system:\n    Standby (S0 Low Power Idle)\n        The system firmware does not support this standby state.";
        let (st2, sup2, conn2) = parse_powercfg_s0_capabilities(Some(unsupported_out), true, false);
        assert_eq!(st2, "Unsupported");
        assert!(!sup2);
        assert_eq!(conn2, Some(false));

        let (st3, sup3, conn3) = parse_powercfg_s0_capabilities(None, false, false);
        assert_eq!(st3, "Unknown");
        assert!(!sup3);
        assert_eq!(conn3, None);
    }
}
