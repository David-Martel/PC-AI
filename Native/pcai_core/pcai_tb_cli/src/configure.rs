use crate::handshake::{discover_thunderbolt_peers, DiscoveredThunderboltPeer};
use crate::identity::{
    compute_commutative_pairwise_ipv4, default_registry_candidates, load_peer_registry_checked,
    repo_registry_candidates, resolve_peer_identity, ThunderboltPeerIdentity,
};
use crate::platform::{collect_platform_snapshot, verify_live_adapter_identity, ThunderboltAdapterState};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub trait CommandRunner: Send + Sync {
    fn run(&self, program: &str, args: &[&str]) -> (bool, String);
}

pub struct StdCommandRunner;

impl CommandRunner for StdCommandRunner {
    fn run(&self, program: &str, args: &[&str]) -> (bool, String) {
        match Command::new(program).args(args).output() {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                let combined = if stderr.is_empty() {
                    stdout
                } else if stdout.is_empty() {
                    stderr
                } else {
                    format!("{stdout} | {stderr}")
                };
                (out.status.success(), combined)
            }
            Err(e) => (false, e.to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct PlannedAction {
    pub interface_index: u32,
    pub interface_alias: String,
    pub category: String,
    pub target: String,
    pub already_in_desired_state: bool,
    pub applied: bool,
    pub command_summary: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct ThunderboltRegistrationReport {
    pub computer_name: String,
    pub os_family: String,
    pub architecture: String,
    pub apply_mode: bool,
    pub overall_success: bool,
    pub failed_action_count: usize,
    pub rolled_back: bool,
    pub rollback_actions: Vec<String>,
    pub local_identity: ThunderboltPeerIdentity,
    pub adapters: Vec<ThunderboltAdapterState>,
    pub discovered_peers: Vec<DiscoveredThunderboltPeer>,
    pub actions: Vec<PlannedAction>,
}

pub fn check_applied_actions(actions: &[PlannedAction]) -> Result<()> {
    let failures: Vec<String> = actions
        .iter()
        .filter(|a| a.error.is_some() || (!a.already_in_desired_state && !a.applied))
        .map(|a| {
            format!(
                "{} on {} (ifIndex {}): {}",
                a.category,
                a.interface_alias,
                a.interface_index,
                a.error.as_deref().unwrap_or("action was not applied")
            )
        })
        .collect();
    if !failures.is_empty() {
        anyhow::bail!(
            "One or more adapter configuration actions failed: {}",
            failures.join("; ")
        );
    }
    Ok(())
}

fn ipv6_addrs_equal(a: &str, b: &str) -> bool {
    fn strip(s: &str) -> &str {
        s.split('%').next().unwrap_or(s).split('/').next().unwrap_or(s).trim()
    }
    let sa = strip(a);
    let sb = strip(b);
    if sa.eq_ignore_ascii_case(sb) {
        return true;
    }
    match (sa.parse::<std::net::Ipv6Addr>(), sb.parse::<std::net::Ipv6Addr>()) {
        (Ok(ia), Ok(ib)) => ia == ib,
        _ => false,
    }
}

fn apply_ipv6_address_on_interface<R: CommandRunner>(
    runner: &R,
    if_index: u32,
    if_alias: &str,
    addr: &str,
    prefix_len: u8,
) -> (bool, String) {
    #[cfg(target_os = "windows")]
    {
        let idx_str = if_index.to_string();
        let addr_with_prefix = format!("{addr}/{prefix_len}");
        let _ = if_alias;
        runner.run(
            "netsh",
            &[
                "interface",
                "ipv6",
                "add",
                "address",
                &format!("interface={idx_str}"),
                &format!("address={addr_with_prefix}"),
            ],
        )
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = if_index;
        let cidr = format!("{addr}/{prefix_len}");
        runner.run("ip", &["-6", "addr", "replace", &cidr, "dev", if_alias])
    }
}

fn remove_ipv6_address_on_interface<R: CommandRunner>(
    runner: &R,
    if_index: u32,
    if_alias: &str,
    cidr: &str,
) -> (bool, String) {
    #[cfg(target_os = "windows")]
    {
        let idx_str = if_index.to_string();
        let addr_only = cidr.split('/').next().unwrap_or(cidr);
        let _ = if_alias;
        runner.run(
            "netsh",
            &[
                "interface",
                "ipv6",
                "delete",
                "address",
                &format!("interface={idx_str}"),
                &format!("address={addr_only}"),
            ],
        )
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = if_index;
        runner.run("ip", &["-6", "addr", "del", cidr, "dev", if_alias])
    }
}

fn apply_ipv4_address_on_interface<R: CommandRunner>(
    runner: &R,
    if_index: u32,
    if_alias: &str,
    addr: &str,
    prefix_len: u8,
) -> (bool, String) {
    #[cfg(target_os = "windows")]
    {
        let idx_str = if_index.to_string();
        let mask = if prefix_len == 30 {
            "255.255.255.252"
        } else {
            "255.255.255.0"
        };
        let _ = if_alias;
        runner.run(
            "netsh",
            &[
                "interface",
                "ipv4",
                "add",
                "address",
                &format!("name={idx_str}"),
                &format!("addr={addr}"),
                &format!("mask={mask}"),
            ],
        )
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = if_index;
        let cidr = format!("{addr}/{prefix_len}");
        runner.run("ip", &["-4", "addr", "replace", &cidr, "dev", if_alias])
    }
}

fn remove_ipv4_address_on_interface<R: CommandRunner>(
    runner: &R,
    if_index: u32,
    if_alias: &str,
    cidr: &str,
) -> (bool, String) {
    #[cfg(target_os = "windows")]
    {
        let idx_str = if_index.to_string();
        let addr_only = cidr.split('/').next().unwrap_or(cidr);
        let _ = if_alias;
        runner.run(
            "netsh",
            &[
                "interface",
                "ipv4",
                "delete",
                "address",
                &format!("name={idx_str}"),
                &format!("addr={addr_only}"),
            ],
        )
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = if_index;
        runner.run("ip", &["-4", "addr", "del", cidr, "dev", if_alias])
    }
}

fn apply_interface_mtu<R: CommandRunner>(runner: &R, if_index: u32, if_alias: &str, mtu: u32) -> (bool, String) {
    #[cfg(target_os = "windows")]
    {
        let _ = if_alias;
        let idx_str = if_index.to_string();
        let mtu_arg = format!("mtu={mtu}");
        let (ok4, msg4) = runner.run(
            "netsh",
            &[
                "interface",
                "ipv4",
                "set",
                "subinterface",
                &idx_str,
                &mtu_arg,
                "store=persistent",
            ],
        );
        let (ok6, msg6) = runner.run(
            "netsh",
            &[
                "interface",
                "ipv6",
                "set",
                "subinterface",
                &idx_str,
                &mtu_arg,
                "store=persistent",
            ],
        );
        (ok4 && ok6, format!("v4:{msg4} v6:{msg6}"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = if_index;
        let mtu_str = mtu.to_string();
        runner.run("ip", &["link", "set", "dev", if_alias, "mtu", &mtu_str])
    }
}

fn apply_interface_metric<R: CommandRunner>(runner: &R, if_index: u32, if_alias: &str, metric: u32) -> (bool, String) {
    #[cfg(target_os = "windows")]
    {
        let _ = if_alias;
        let idx_str = if_index.to_string();
        let metric_arg = format!("metric={metric}");
        let (ok4, msg4) = runner.run(
            "netsh",
            &[
                "interface",
                "ipv4",
                "set",
                "interface",
                &idx_str,
                &metric_arg,
                "store=persistent",
            ],
        );
        let (ok6, msg6) = runner.run(
            "netsh",
            &[
                "interface",
                "ipv6",
                "set",
                "interface",
                &idx_str,
                &metric_arg,
                "store=persistent",
            ],
        );
        (ok4 && ok6, format!("v4:{msg4} v6:{msg6}"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = (runner, if_index, if_alias, metric);
        (true, "Linux applies metric on per-route entries".to_string())
    }
}

fn apply_ipv6_host_route<R: CommandRunner>(
    runner: &R,
    if_index: u32,
    if_alias: &str,
    ula_addr: &str,
    metric: u32,
) -> (bool, String) {
    #[cfg(target_os = "windows")]
    {
        let _ = if_alias;
        let idx_str = if_index.to_string();
        let prefix = format!("{ula_addr}/128");
        let metric_arg = format!("metric={metric}");
        let (ok, msg) = runner.run(
            "netsh",
            &[
                "interface",
                "ipv6",
                "add",
                "route",
                &prefix,
                &idx_str,
                "::",
                &metric_arg,
            ],
        );
        if ok {
            (true, msg)
        } else {
            runner.run(
                "netsh",
                &[
                    "interface",
                    "ipv6",
                    "set",
                    "route",
                    &prefix,
                    &idx_str,
                    "::",
                    &metric_arg,
                ],
            )
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = if_index;
        let prefix = format!("{ula_addr}/128");
        let metric_str = metric.to_string();
        runner.run(
            "ip",
            &[
                "-6",
                "route",
                "replace",
                &prefix,
                "dev",
                if_alias,
                "metric",
                &metric_str,
            ],
        )
    }
}

fn remove_ipv6_host_route<R: CommandRunner>(runner: &R, if_index: u32, if_alias: &str, prefix: &str) -> (bool, String) {
    #[cfg(target_os = "windows")]
    {
        let _ = if_alias;
        let idx_str = if_index.to_string();
        runner.run(
            "netsh",
            &["interface", "ipv6", "delete", "route", prefix, &idx_str, "::"],
        )
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = if_index;
        runner.run("ip", &["-6", "route", "del", prefix, "dev", if_alias])
    }
}

fn rollback_applied_adapter_actions<R: CommandRunner>(
    runner: &R,
    adapter: &ThunderboltAdapterState,
    actions: &[PlannedAction],
) -> Vec<String> {
    let mut rollback_log = Vec::new();
    let if_idx = adapter.interface_index;
    let alias = &adapter.interface_alias;

    for act in actions.iter().rev() {
        if !act.applied || act.already_in_desired_state {
            continue;
        }
        match act.category.as_str() {
            "InterfaceMtu" if adapter.ipv4_mtu_bytes > 0 => {
                let (ok, msg) = apply_interface_mtu(runner, if_idx, alias, adapter.ipv4_mtu_bytes);
                rollback_log.push(format!(
                    "Rollback InterfaceMtu to {} on {alias} (ifIndex {if_idx}): ok={ok} ({msg})",
                    adapter.ipv4_mtu_bytes
                ));
            }
            "InterfaceMetric" if adapter.metric_known && adapter.ipv4_metric > 0 => {
                let (ok, msg) = apply_interface_metric(runner, if_idx, alias, adapter.ipv4_metric);
                rollback_log.push(format!(
                    "Rollback InterfaceMetric to {} on {alias} (ifIndex {if_idx}): ok={ok} ({msg})",
                    adapter.ipv4_metric
                ));
            }
            "LocalIPv6LinkLocal" | "LocalIPv6ULA" => {
                let (ok, msg) = remove_ipv6_address_on_interface(runner, if_idx, alias, &act.target);
                rollback_log.push(format!(
                    "Rollback {} ({}) on {alias} (ifIndex {if_idx}): ok={ok} ({msg})",
                    act.category, act.target
                ));
            }
            "PeerIPv6ULARoute128" => {
                let prefix = act.target.split(" ->").next().unwrap_or(&act.target);
                let (ok, msg) = remove_ipv6_host_route(runner, if_idx, alias, prefix);
                rollback_log.push(format!(
                    "Rollback PeerIPv6ULARoute128 ({prefix}) on {alias} (ifIndex {if_idx}): ok={ok} ({msg})"
                ));
            }
            "PairwiseIPv4Subnet30" => {
                let cidr = act.target.split(" (").next().unwrap_or(&act.target);
                let (ok, msg) = remove_ipv4_address_on_interface(runner, if_idx, alias, cidr);
                rollback_log.push(format!(
                    "Rollback PairwiseIPv4Subnet30 ({cidr}) on {alias} (ifIndex {if_idx}): ok={ok} ({msg})"
                ));
            }
            _ => {}
        }
    }
    rollback_log
}

#[allow(clippy::too_many_arguments)]
pub fn plan_and_apply_adapter_actions<R: CommandRunner>(
    runner: &R,
    adapter: &ThunderboltAdapterState,
    local_id: &ThunderboltPeerIdentity,
    peer_name_opt: Option<&str>,
    apply: bool,
    verify_live_identity: bool,
    mtu_bytes: u32,
    interface_metric: u32,
    assign_pairwise_ipv4: bool,
    registry_path: Option<&Path>,
) -> Vec<PlannedAction> {
    let (actions, _) = plan_and_apply_adapter_actions_with_rollback(
        runner,
        adapter,
        local_id,
        peer_name_opt,
        apply,
        verify_live_identity,
        mtu_bytes,
        interface_metric,
        assign_pairwise_ipv4,
        registry_path,
    );
    actions
}

#[allow(clippy::too_many_arguments)]
pub fn plan_and_apply_adapter_actions_with_rollback<R: CommandRunner>(
    runner: &R,
    adapter: &ThunderboltAdapterState,
    local_id: &ThunderboltPeerIdentity,
    peer_name_opt: Option<&str>,
    apply: bool,
    verify_live_identity: bool,
    mtu_bytes: u32,
    interface_metric: u32,
    assign_pairwise_ipv4: bool,
    registry_path: Option<&Path>,
) -> (Vec<PlannedAction>, Vec<String>) {
    let mut actions = Vec::new();
    let if_idx = adapter.interface_index;
    let alias = &adapter.interface_alias;

    // Preserve external owner custody (e.g., ms01-usb4-extension or NetworkManager profile)
    if let Some(ref ext_owner) = adapter.external_owner {
        actions.push(PlannedAction {
            interface_index: if_idx,
            interface_alias: alias.clone(),
            category: "ExternalOwnerCustody".to_string(),
            target: ext_owner.clone(),
            already_in_desired_state: true,
            applied: false,
            command_summary: format!(
                "Preserve external owner custody ({ext_owner}) on {alias} (ifIndex {if_idx}) without conflicting mutations"
            ),
            error: None,
        });
        return (actions, Vec::new());
    }

    if apply && verify_live_identity {
        if let Err(e) = verify_live_adapter_identity(if_idx, &adapter.adapter_guid, &adapter.mac_address) {
            actions.push(PlannedAction {
                interface_index: if_idx,
                interface_alias: alias.clone(),
                category: "AdapterIdentityVerification".to_string(),
                target: format!("GUID={} MAC={}", adapter.adapter_guid, adapter.mac_address),
                already_in_desired_state: false,
                applied: false,
                command_summary: format!("Verify adapter identity before mutating ifIndex {if_idx}"),
                error: Some(e.to_string()),
            });
            return (actions, Vec::new());
        }
    }

    // 1a. Check MTU (idempotent)
    let mtu_ok = adapter.ipv4_mtu_bytes == mtu_bytes && adapter.ipv6_mtu_bytes == mtu_bytes;
    let mut mtu_action = PlannedAction {
        interface_index: if_idx,
        interface_alias: alias.clone(),
        category: "InterfaceMtu".to_string(),
        target: format!("MTU={mtu_bytes}"),
        already_in_desired_state: mtu_ok,
        applied: false,
        command_summary: format!("Set MTU={mtu_bytes} on {alias} (ifIndex {if_idx})"),
        error: None,
    };
    if !mtu_ok && apply {
        let (ok, msg) = apply_interface_mtu(runner, if_idx, alias, mtu_bytes);
        mtu_action.applied = ok;
        if !ok {
            mtu_action.error = Some(msg);
        }
    }
    actions.push(mtu_action);

    // 1b. Check Interface Metric (idempotent — explicitly enforce --metric on both IPv4 & IPv6 interfaces)
    let metric_ok =
        adapter.metric_known && adapter.ipv4_metric == interface_metric && adapter.ipv6_metric == interface_metric;
    let mut metric_action = PlannedAction {
        interface_index: if_idx,
        interface_alias: alias.clone(),
        category: "InterfaceMetric".to_string(),
        target: format!("Metric={interface_metric}"),
        already_in_desired_state: metric_ok,
        applied: false,
        command_summary: format!("Set Metric={interface_metric} on {alias} (ifIndex {if_idx})"),
        error: None,
    };
    if !metric_ok && apply {
        let (ok, msg) = apply_interface_metric(runner, if_idx, alias, interface_metric);
        metric_action.applied = ok;
        if !ok {
            metric_action.error = Some(msg);
        }
    }
    actions.push(metric_action);

    // 2. Local Deterministic Link-Local Addresses (fe80::d700:...)
    for lla in &local_id.all_link_local_addresses {
        let exists = adapter.addresses.iter().any(|a| ipv6_addrs_equal(&a.address, lla));
        let mut act = PlannedAction {
            interface_index: if_idx,
            interface_alias: alias.clone(),
            category: "LocalIPv6LinkLocal".to_string(),
            target: format!("{lla}/64"),
            already_in_desired_state: exists,
            applied: false,
            command_summary: format!("Assign {lla}/64 on {alias} (ifIndex {if_idx})"),
            error: None,
        };
        if !exists && apply {
            let (ok, msg) = apply_ipv6_address_on_interface(runner, if_idx, alias, lla, 64);
            act.applied = ok;
            if !ok {
                act.error = Some(msg);
            }
        }
        actions.push(act);
    }

    // 3. Local Deterministic ULA Addresses (/128 host addresses for multi-port support)
    for ula in &local_id.all_ula_addresses {
        let exists = adapter.addresses.iter().any(|a| ipv6_addrs_equal(&a.address, ula));
        let mut act = PlannedAction {
            interface_index: if_idx,
            interface_alias: alias.clone(),
            category: "LocalIPv6ULA".to_string(),
            target: format!("{ula}/128"),
            already_in_desired_state: exists,
            applied: false,
            command_summary: format!("Assign {ula}/128 on {alias} (ifIndex {if_idx})"),
            error: None,
        };
        if !exists && apply {
            let (ok, msg) = apply_ipv6_address_on_interface(runner, if_idx, alias, ula, 128);
            act.applied = ok;
            if !ok {
                act.error = Some(msg);
            }
        }
        actions.push(act);
    }

    // 4. Per-Port On-Link /128 ULA Routes & Optional Pairwise /30 IPv4 for Authenticated Fleet Peer
    if let Some(peer_name) = peer_name_opt {
        match resolve_peer_identity(Some(peer_name), Some(&local_id.computer_name), registry_path) {
            Ok(peer_id) if peer_id.is_known_fleet_peer => {
                for peer_ula in &peer_id.all_ula_addresses {
                    let mut r_act = PlannedAction {
                        interface_index: if_idx,
                        interface_alias: alias.clone(),
                        category: "PeerIPv6ULARoute128".to_string(),
                        target: format!("{peer_ula}/128 -> ifIndex {if_idx} ({peer_name})"),
                        already_in_desired_state: false,
                        applied: false,
                        command_summary: format!(
                            "Ensure on-link /128 route for {peer_ula}/128 via {alias} (ifIndex {if_idx}) metric={interface_metric}"
                        ),
                        error: None,
                    };
                    if apply {
                        let (ok, msg) = apply_ipv6_host_route(runner, if_idx, alias, peer_ula, interface_metric);
                        r_act.applied = ok;
                        if !ok {
                            r_act.error = Some(msg);
                        }
                    }
                    actions.push(r_act);
                }

                if assign_pairwise_ipv4 {
                    if let Ok(pair) = compute_commutative_pairwise_ipv4(&local_id.computer_name, peer_name) {
                        let v4_exists = adapter.addresses.iter().any(|a| a.address == pair.local_ipv4_address);
                        let mut v4_act = PlannedAction {
                            interface_index: if_idx,
                            interface_alias: alias.clone(),
                            category: "PairwiseIPv4Subnet30".to_string(),
                            target: format!(
                                "{}/{} (peer {}={})",
                                pair.local_ipv4_address, pair.prefix_length, peer_name, pair.peer_ipv4_address
                            ),
                            already_in_desired_state: v4_exists,
                            applied: false,
                            command_summary: format!(
                                "Assign commutative pairwise /30 IPv4 {} on {alias}",
                                pair.local_ipv4_address
                            ),
                            error: None,
                        };
                        if !v4_exists && apply {
                            let (ok, msg) = apply_ipv4_address_on_interface(
                                runner,
                                if_idx,
                                alias,
                                &pair.local_ipv4_address,
                                pair.prefix_length,
                            );
                            v4_act.applied = ok;
                            if !ok {
                                v4_act.error = Some(msg);
                            }
                        }
                        actions.push(v4_act);
                    }
                }
            }
            Ok(_) => {
                actions.push(PlannedAction {
                    interface_index: if_idx,
                    interface_alias: alias.clone(),
                    category: "PeerAuthenticationGate".to_string(),
                    target: peer_name.to_string(),
                    already_in_desired_state: false,
                    applied: false,
                    command_summary: format!("Verify fleet registration for peer '{peer_name}'"),
                    error: Some(format!(
                        "Refusing to admit unregistered/unknown peer '{peer_name}' into Thunderbolt fabric routes"
                    )),
                });
            }
            Err(e) => {
                actions.push(PlannedAction {
                    interface_index: if_idx,
                    interface_alias: alias.clone(),
                    category: "PeerIdentityResolution".to_string(),
                    target: peer_name.to_string(),
                    already_in_desired_state: false,
                    applied: false,
                    command_summary: format!("Resolve peer identity for '{peer_name}'"),
                    error: Some(e.to_string()),
                });
            }
        }
    }

    let has_failure = actions
        .iter()
        .any(|a| a.error.is_some() || (apply && !a.already_in_desired_state && !a.applied));
    let rollback_log = if apply && has_failure {
        rollback_applied_adapter_actions(runner, adapter, &actions)
    } else {
        Vec::new()
    };

    (actions, rollback_log)
}

pub async fn register_local_thunderbolt_fabric(
    apply: bool,
    mtu_bytes: u32,
    interface_metric: u32,
    assign_pairwise_ipv4: bool,
    peer_hint: Option<&str>,
    registry_path: Option<&Path>,
) -> Result<ThunderboltRegistrationReport> {
    let local_id = resolve_peer_identity(None, peer_hint, registry_path)?;
    let snapshot = collect_platform_snapshot(false)?;
    let discovered = discover_thunderbolt_peers(false, 220, false, registry_path).await?;

    let mut peer_by_if: HashMap<u32, String> = HashMap::new();
    for d in &discovered {
        if d.verified_handshake && d.is_known_fleet_peer {
            if let Some(ref p) = d.peer_name {
                peer_by_if.entry(d.interface_index).or_insert_with(|| p.clone());
            }
        }
    }
    if let Some(hint) = peer_hint {
        if !hint.trim().is_empty() && snapshot.adapters.len() == 1 {
            let only_if = snapshot.adapters[0].interface_index;
            let hid = resolve_peer_identity(Some(hint), None, registry_path)?;
            if !hid.is_known_fleet_peer {
                anyhow::bail!(
                    "Explicit --peer '{}' is not a registered fleet peer in thunderbolt-peers.json",
                    hint
                );
            }
            peer_by_if.entry(only_if).or_insert(hid.computer_name);
        }
    }

    let runner = StdCommandRunner;
    let mut actions = Vec::new();
    let mut rollback_actions = Vec::new();

    for adapter in &snapshot.adapters {
        let peer_opt = peer_by_if.get(&adapter.interface_index).map(String::as_str);
        let (adapter_actions, adapter_rollback) = plan_and_apply_adapter_actions_with_rollback(
            &runner,
            adapter,
            &local_id,
            peer_opt,
            apply,
            true,
            mtu_bytes,
            interface_metric,
            assign_pairwise_ipv4,
            registry_path,
        );
        actions.extend(adapter_actions);
        rollback_actions.extend(adapter_rollback);
    }

    let failed_action_count = actions
        .iter()
        .filter(|a| a.error.is_some() || (apply && !a.already_in_desired_state && !a.applied))
        .count();
    let overall_success = failed_action_count == 0;
    let rolled_back = !rollback_actions.is_empty();

    Ok(ThunderboltRegistrationReport {
        computer_name: local_id.computer_name.clone(),
        os_family: env::consts::OS.to_string(),
        architecture: env::consts::ARCH.to_string(),
        apply_mode: apply,
        overall_success,
        failed_action_count,
        rolled_back,
        rollback_actions,
        local_identity: local_id,
        adapters: snapshot.adapters,
        discovered_peers: discovered,
        actions,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct InstallFrameworkReport {
    pub computer_name: String,
    pub os_family: String,
    pub architecture: String,
    pub source_binary: String,
    pub installed_binaries: Vec<String>,
    pub installed_configs: Vec<String>,
    pub apply_mode: bool,
    pub overall_success: bool,
    pub rolled_back: bool,
    pub rollback_actions: Vec<String>,
    pub errors: Vec<String>,
}

fn sha256_file(path: &Path) -> Result<[u8; 32]> {
    let bytes = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(hasher.finalize().into())
}

fn ensure_not_symlink(path: &Path, role: &str) -> Result<()> {
    if let Ok(meta) = fs::symlink_metadata(path) {
        if meta.file_type().is_symlink() {
            anyhow::bail!("Refusing to use symlink {role} path '{}'", path.display());
        }
    }
    Ok(())
}

enum InstallPreimage {
    Created { dst: PathBuf },
    Existed { dst: PathBuf, backup: PathBuf },
}

fn capture_preimage_and_install(src: &Path, dst: &Path) -> Result<InstallPreimage> {
    ensure_not_symlink(src, "source")?;
    if !src.is_file() {
        anyhow::bail!("Source file '{}' does not exist", src.display());
    }
    ensure_not_symlink(dst, "destination")?;
    let parent = dst
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Destination '{}' has no parent directory", dst.display()))?;
    fs::create_dir_all(parent)?;

    let preimage = if dst.exists() {
        if !dst.is_file() {
            anyhow::bail!("Destination '{}' exists and is not a regular file", dst.display());
        }
        let backup_name = format!(
            ".{}.pcai-backup-{}",
            dst.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
            std::process::id()
        );
        let backup_path = parent.join(backup_name);
        fs::copy(dst, &backup_path)?;
        InstallPreimage::Existed {
            dst: dst.to_path_buf(),
            backup: backup_path,
        }
    } else {
        InstallPreimage::Created { dst: dst.to_path_buf() }
    };

    if let Err(e) = staged_atomic_copy_verified(src, dst) {
        // Restore immediate preimage if staged copy failed after creating backup
        if let InstallPreimage::Existed { ref backup, .. } = preimage {
            let _ = fs::remove_file(backup);
        }
        return Err(e);
    }

    Ok(preimage)
}

fn rollback_install_preimages(preimages: &[InstallPreimage]) -> Vec<String> {
    let mut log = Vec::new();
    for item in preimages.iter().rev() {
        match item {
            InstallPreimage::Created { dst } => match fs::remove_file(dst) {
                Ok(()) => log.push(format!("Removed newly created file '{}'", dst.display())),
                Err(e) => log.push(format!("Failed removing newly created file '{}': {e}", dst.display())),
            },
            InstallPreimage::Existed { dst, backup } => match fs::copy(backup, dst) {
                Ok(_) => {
                    let _ = fs::remove_file(backup);
                    log.push(format!("Restored pre-existing file '{}'", dst.display()));
                }
                Err(e) => log.push(format!("Failed restoring pre-existing file '{}': {e}", dst.display())),
            },
        }
    }
    log
}

fn cleanup_install_backups(preimages: &[InstallPreimage]) {
    for item in preimages {
        if let InstallPreimage::Existed { backup, .. } = item {
            let _ = fs::remove_file(backup);
        }
    }
}

fn staged_atomic_copy_verified(src: &Path, dst: &Path) -> Result<()> {
    ensure_not_symlink(src, "source")?;
    if !src.is_file() {
        anyhow::bail!("Source file '{}' does not exist", src.display());
    }
    ensure_not_symlink(dst, "destination")?;
    let parent = dst
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Destination '{}' has no parent directory", dst.display()))?;
    fs::create_dir_all(parent)?;

    let src_hash = sha256_file(src)?;
    let stage_name = format!(
        ".{}.pcai-stage-{}",
        dst.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
        std::process::id()
    );
    let stage_path = parent.join(stage_name);

    fs::copy(src, &stage_path)?;
    let stage_hash = sha256_file(&stage_path)?;
    if src_hash != stage_hash {
        let _ = fs::remove_file(&stage_path);
        anyhow::bail!(
            "SHA-256 mismatch staging '{}' -> '{}'",
            src.display(),
            stage_path.display()
        );
    }

    if let Err(rename_err) = fs::rename(&stage_path, dst) {
        // Fallback if destination is on Windows and rename fails across locks
        if let Err(copy_err) = fs::copy(&stage_path, dst) {
            let _ = fs::remove_file(&stage_path);
            anyhow::bail!(
                "Failed to install '{}' -> '{}' (rename: {rename_err}; copy: {copy_err})",
                src.display(),
                dst.display()
            );
        }
        let _ = fs::remove_file(&stage_path);
    }

    let dst_hash = sha256_file(dst)?;
    if src_hash != dst_hash {
        anyhow::bail!("Post-install SHA-256 verification failed for '{}'", dst.display());
    }
    Ok(())
}

pub fn install_framework_with_paths(
    exe_path: &Path,
    target_bins: &[PathBuf],
    registry_src: Option<&Path>,
    target_configs: &[PathBuf],
    apply: bool,
) -> Result<InstallFrameworkReport> {
    let mut installed_bins = Vec::new();
    let mut installed_configs = Vec::new();
    let mut errors = Vec::new();
    let mut preimages: Vec<InstallPreimage> = Vec::new();

    for dst in target_bins {
        if dst == exe_path {
            installed_bins.push(dst.to_string_lossy().into_owned());
            continue;
        }
        if apply {
            match capture_preimage_and_install(exe_path, dst) {
                Ok(pre) => {
                    installed_bins.push(dst.to_string_lossy().into_owned());
                    preimages.push(pre);
                }
                Err(e) => {
                    errors.push(format!("Binary install failed ({}): {e}", dst.display()));
                    break;
                }
            }
        } else {
            installed_bins.push(format!("(dry-run) {}", dst.to_string_lossy()));
        }
    }

    if errors.is_empty() {
        if let Some(src_path) = registry_src {
            for dst_cfg in target_configs {
                if dst_cfg == src_path {
                    installed_configs.push(dst_cfg.to_string_lossy().into_owned());
                    continue;
                }
                if apply {
                    match capture_preimage_and_install(src_path, dst_cfg) {
                        Ok(pre) => {
                            installed_configs.push(dst_cfg.to_string_lossy().into_owned());
                            preimages.push(pre);
                        }
                        Err(e) => {
                            errors.push(format!("Config install failed ({}): {e}", dst_cfg.display()));
                            break;
                        }
                    }
                } else {
                    installed_configs.push(format!("(dry-run) {}", dst_cfg.to_string_lossy()));
                }
            }
        }
    }

    let rollback_actions = if apply && !errors.is_empty() {
        installed_bins.clear();
        installed_configs.clear();
        rollback_install_preimages(&preimages)
    } else {
        cleanup_install_backups(&preimages);
        Vec::new()
    };

    let overall_success = errors.is_empty();
    let rolled_back = !rollback_actions.is_empty();
    Ok(InstallFrameworkReport {
        computer_name: crate::identity::local_hostname(),
        os_family: env::consts::OS.to_string(),
        architecture: env::consts::ARCH.to_string(),
        source_binary: exe_path.to_string_lossy().into_owned(),
        installed_binaries: installed_bins,
        installed_configs,
        apply_mode: apply,
        overall_success,
        rolled_back,
        rollback_actions,
        errors,
    })
}

pub fn resolve_install_registry_source(explicit_registry: Option<&Path>) -> Result<Option<PathBuf>> {
    resolve_install_registry_source_from_candidates(
        explicit_registry,
        &repo_registry_candidates(),
        &default_registry_candidates(None),
    )
}

fn resolve_install_registry_source_from_candidates(
    explicit_registry: Option<&Path>,
    repository_candidates: &[PathBuf],
    fallback_candidates: &[PathBuf],
) -> Result<Option<PathBuf>> {
    if let Some(explicit) = explicit_registry {
        let (_, loaded_reg_path) = load_peer_registry_checked(Some(explicit))?;
        return Ok(loaded_reg_path.map(PathBuf::from));
    }
    for candidate in repository_candidates.iter().chain(fallback_candidates) {
        if candidate.is_file() {
            let (_, loaded_reg_path) = load_peer_registry_checked(Some(candidate))?;
            return Ok(loaded_reg_path.map(PathBuf::from));
        }
    }
    Ok(None)
}

pub fn install_framework(apply: bool, explicit_registry: Option<&Path>) -> Result<InstallFrameworkReport> {
    let exe_path = env::current_exe()?;
    let bin_name = if cfg!(target_os = "windows") {
        "pcai-tb.exe"
    } else {
        "pcai-tb"
    };

    let home = env::var("USERPROFILE")
        .or_else(|_| env::var("HOME"))
        .unwrap_or_default();

    let mut target_bins = Vec::new();
    if !home.is_empty() {
        let hp = PathBuf::from(&home);
        target_bins.push(hp.join("bin").join(bin_name));
        target_bins.push(hp.join(".local").join("bin").join(bin_name));
    }
    if cfg!(target_os = "windows") {
        for repo_root in [r"C:\codedev\PC-AI", r"C:\codedev\PC_AI"] {
            if Path::new(repo_root).is_dir() {
                target_bins.push(PathBuf::from(repo_root).join("bin").join(bin_name));
            }
        }
    }

    let reg_src_buf = resolve_install_registry_source(explicit_registry)?;
    let mut target_configs = Vec::new();
    if reg_src_buf.is_some() && !home.is_empty() {
        let hp = PathBuf::from(&home);
        target_configs.push(hp.join(".machine").join("thunderbolt-peers.json"));
        target_configs.push(hp.join(".config").join("pcai").join("thunderbolt-peers.json"));
    }

    install_framework_with_paths(&exe_path, &target_bins, reg_src_buf.as_deref(), &target_configs, apply)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn repo_registry_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../Config/thunderbolt-peers.json")
    }

    struct MockFailingCommandRunner {
        fail_category_substring: String,
        invocations: Mutex<Vec<String>>,
    }

    impl CommandRunner for MockFailingCommandRunner {
        fn run(&self, program: &str, args: &[&str]) -> (bool, String) {
            let cmd = format!("{program} {}", args.join(" "));
            self.invocations.lock().unwrap().push(cmd.clone());
            if cmd.contains(&self.fail_category_substring) {
                (false, format!("Simulated OS command failure for: {cmd}"))
            } else {
                (true, "Ok.".to_string())
            }
        }
    }

    #[test]
    fn test_plan_and_apply_adapter_actions_enforces_metric_surfaces_failures_and_rolls_back() {
        let reg_path = repo_registry_path();
        let adapter = ThunderboltAdapterState {
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
            ipv4_mtu_bytes: 1500,
            ipv6_mtu_bytes: 1500,
            metric_known: true,
            ipv4_metric: 25,
            ipv6_metric: 25,
            xdomain_peer_device_name: None,
            xdomain_peer_unique_id: None,
            external_owner: None,
            addresses: vec![],
            neighbors: vec![],
        };

        let local_id =
            resolve_peer_identity(Some("dtm-p1gen7"), Some("dtm-p1gen9"), Some(&reg_path)).expect("resolve local_id");

        let runner = MockFailingCommandRunner {
            fail_category_substring: if cfg!(target_os = "windows") {
                "metric=5".to_string()
            } else {
                "fd00:7462:7032::9/128".to_string()
            },
            invocations: Mutex::new(Vec::new()),
        };

        let (actions, rollback_log) = plan_and_apply_adapter_actions_with_rollback(
            &runner,
            &adapter,
            &local_id,
            Some("dtm-p1gen9"),
            true,
            false,
            62000,
            5,
            true,
            Some(&reg_path),
        );

        let mtu_act = actions
            .iter()
            .find(|a| a.category == "InterfaceMtu")
            .expect("InterfaceMtu action present");
        assert!(mtu_act.applied);
        assert!(mtu_act.error.is_none());

        assert!(
            check_applied_actions(&actions).is_err(),
            "check_applied_actions must fail when a mutating step fails"
        );
        assert!(
            !rollback_log.is_empty(),
            "Rollback log must record restoration of prior adapter settings when apply fails"
        );
        assert!(
            rollback_log
                .iter()
                .any(|entry| entry.contains("Rollback InterfaceMtu to 1500")),
            "Rollback must restore preimage MTU=1500: {rollback_log:?}"
        );
    }

    #[test]
    fn test_plan_and_apply_adapter_actions_preserves_external_owner_custody() {
        let reg_path = repo_registry_path();
        let adapter = ThunderboltAdapterState {
            interface_index: 9,
            interface_alias: "ms01-milly".to_string(),
            interface_description: "USB4/Thunderbolt P2P (veth)".to_string(),
            adapter_guid: "00000009-0000-0000-0000-000000000000".to_string(),
            mac_address: "AA-BB-CC-DD-EE-09".to_string(),
            operational_status: "Up".to_string(),
            speed_known: true,
            transmit_link_speed_bps: 20_000_000_000,
            receive_link_speed_bps: 20_000_000_000,
            link_speed_gbps: 20.0,
            ipv4_mtu_bytes: 1500,
            ipv6_mtu_bytes: 1500,
            metric_known: true,
            ipv4_metric: 10,
            ipv6_metric: 10,
            xdomain_peer_device_name: Some("millylaptop1".to_string()),
            xdomain_peer_unique_id: Some("bab38780-0019-a9b4-ffff-ffffffffffff".to_string()),
            external_owner: Some("ms01-usb4-extension (ifalias=ms01-usb4:owner)".to_string()),
            addresses: vec![],
            neighbors: vec![],
        };

        let local_id =
            resolve_peer_identity(Some("minisforum"), Some("millylaptop1"), Some(&reg_path)).expect("resolve local_id");
        let runner = MockFailingCommandRunner {
            fail_category_substring: "never".to_string(),
            invocations: Mutex::new(Vec::new()),
        };

        let actions = plan_and_apply_adapter_actions(
            &runner,
            &adapter,
            &local_id,
            Some("millylaptop1"),
            true,
            false,
            62000,
            5,
            true,
            Some(&reg_path),
        );

        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].category, "ExternalOwnerCustody");
        assert!(actions[0].already_in_desired_state);
        assert!(
            runner.invocations.lock().unwrap().is_empty(),
            "Externally owned interface must never be mutated by pcai-tb"
        );
    }

    #[test]
    fn test_install_framework_with_paths_stages_verifies_hash_and_rolls_back_on_partial_failure() {
        let temp_root = env::temp_dir().join(format!("pcai_tb_install_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_root);
        fs::create_dir_all(&temp_root).unwrap();

        let src_bin = temp_root.join("src_pcai_tb.exe");
        fs::write(&src_bin, b"pcai-tb-test-binary-payload-v2").unwrap();
        let dst_bin1 = temp_root.join("out_bin1").join("pcai-tb.exe");
        let dst_bin2 = temp_root.join("out_bin2").join("pcai-tb.exe");

        // Pre-populate dst_bin1 with v1 content so we can verify rollback restores v1!
        fs::create_dir_all(dst_bin1.parent().unwrap()).unwrap();
        fs::write(&dst_bin1, b"preexisting-v1-binary").unwrap();

        // Create a directory at dst_bin2 so installing a file to dst_bin2 fails after dst_bin1 succeeded!
        fs::create_dir_all(&dst_bin2).unwrap();

        let partial_fail_report =
            install_framework_with_paths(&src_bin, &[dst_bin1.clone(), dst_bin2.clone()], None, &[], true)
                .expect("install returns structured report");

        assert!(
            !partial_fail_report.overall_success,
            "Partial multi-target install failure must set overall_success=false"
        );
        assert!(
            partial_fail_report.rolled_back,
            "Partial multi-target install failure must trigger rollback"
        );
        assert_eq!(
            fs::read(&dst_bin1).unwrap(),
            b"preexisting-v1-binary",
            "Rollback must restore pre-existing v1 binary on dst_bin1 when dst_bin2 fails"
        );

        // Now remove the directory at dst_bin2 and verify clean multi-target install succeeds
        fs::remove_dir_all(&dst_bin2).unwrap();
        let ok_report = install_framework_with_paths(&src_bin, &[dst_bin1.clone(), dst_bin2.clone()], None, &[], true)
            .expect("install ok");
        assert!(ok_report.overall_success);
        assert!(!ok_report.rolled_back);
        assert!(ok_report.errors.is_empty());
        assert_eq!(fs::read(&dst_bin1).unwrap(), b"pcai-tb-test-binary-payload-v2");
        assert_eq!(fs::read(&dst_bin2).unwrap(), b"pcai-tb-test-binary-payload-v2");

        let _ = fs::remove_dir_all(&temp_root);
    }

    // Protects install precedence without the caller's checkout, cwd or home.
    // Detects fallback precedence and malformed/absent explicit registry regressions.
    // Needs private temporary files; breadcrumb: resolve_install_registry_source_from_candidates.
    #[test]
    fn test_resolve_install_registry_source_hermetic_precedence_and_errors() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("repository.json");
        let fallback = temp.path().join("fallback.json");
        let explicit = temp.path().join("explicit.json");
        let missing = temp.path().join("missing.json");
        let fixture = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../Config/thunderbolt-peers.json"
        ));
        for path in [&repo, &fallback, &explicit] {
            fs::write(path, fixture).unwrap();
        }
        let resolve = |path: Option<&Path>, repos: &[PathBuf], defaults: &[PathBuf]| {
            resolve_install_registry_source_from_candidates(path, repos, defaults)
        };
        assert_eq!(
            resolve(None, std::slice::from_ref(&repo), std::slice::from_ref(&fallback)).unwrap(),
            Some(repo.clone())
        );
        assert_eq!(
            resolve(None, std::slice::from_ref(&missing), std::slice::from_ref(&fallback)).unwrap(),
            Some(fallback.clone())
        );
        assert_eq!(
            resolve(Some(&explicit), std::slice::from_ref(&repo), &[fallback]).unwrap(),
            Some(explicit)
        );
        assert_eq!(resolve(None, std::slice::from_ref(&missing), &[]).unwrap(), None);
        assert!(resolve(Some(&missing), std::slice::from_ref(&repo), &[]).is_err());
        fs::write(&repo, "invalid JSON").unwrap();
        assert!(resolve(None, &[repo], &[]).is_err());
    }

    #[test]
    fn test_ipv6_addrs_equal_handles_rfc5952_single_zero_hextet_and_leading_zeros() {
        assert!(ipv6_addrs_equal(
            "fd00:7462:7032:d700:0:c546:83ee:7cdf",
            "fd00:7462:7032:d700::c546:83ee:7cdf"
        ));
        assert!(ipv6_addrs_equal(
            "fe80::d700:9058:d9e:d2a7%54",
            "fe80::d700:9058:0d9e:d2a7/64"
        ));
        assert!(!ipv6_addrs_equal(
            "fd00:7462:7032:d700::c546:83ee:7cdf",
            "fd00:7462:7032::9"
        ));
    }
}
