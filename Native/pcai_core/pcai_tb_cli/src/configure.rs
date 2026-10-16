use crate::handshake::{discover_thunderbolt_peers, DiscoveredThunderboltPeer};
use crate::identity::{compute_commutative_pairwise_ipv4, resolve_peer_identity, ThunderboltPeerIdentity};
use crate::platform::{collect_platform_snapshot, ThunderboltAdapterState};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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
    pub local_identity: ThunderboltPeerIdentity,
    pub adapters: Vec<ThunderboltAdapterState>,
    pub discovered_peers: Vec<DiscoveredThunderboltPeer>,
    pub actions: Vec<PlannedAction>,
}

fn run_os_command(program: &str, args: &[&str]) -> (bool, String) {
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

fn apply_ipv6_address_on_interface(if_index: u32, if_alias: &str, addr: &str, prefix_len: u8) -> (bool, String) {
    #[cfg(target_os = "windows")]
    {
        let idx_str = if_index.to_string();
        let addr_with_prefix = format!("{addr}/{prefix_len}");
        let _ = if_alias;
        run_os_command(
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
        run_os_command("ip", &["-6", "addr", "replace", &cidr, "dev", if_alias])
    }
}

fn apply_ipv4_address_on_interface(if_index: u32, if_alias: &str, addr: &str, prefix_len: u8) -> (bool, String) {
    #[cfg(target_os = "windows")]
    {
        let idx_str = if_index.to_string();
        let mask = if prefix_len == 30 {
            "255.255.255.252"
        } else {
            "255.255.255.0"
        };
        let _ = if_alias;
        run_os_command(
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
        run_os_command("ip", &["-4", "addr", "replace", &cidr, "dev", if_alias])
    }
}

fn apply_mtu_and_metric(if_index: u32, if_alias: &str, mtu: u32, metric: u32) -> (bool, String) {
    #[cfg(target_os = "windows")]
    {
        let _ = if_alias;
        let idx_str = if_index.to_string();
        let mtu_arg = format!("mtu={mtu}");
        let _ = metric;
        let (ok4, msg4) = run_os_command(
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
        let (ok6, msg6) = run_os_command(
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
        let _ = (if_index, metric);
        let mtu_str = mtu.to_string();
        run_os_command("ip", &["link", "set", "dev", if_alias, "mtu", &mtu_str])
    }
}

fn apply_ipv6_host_route(if_index: u32, if_alias: &str, ula_addr: &str, metric: u32) -> (bool, String) {
    #[cfg(target_os = "windows")]
    {
        let _ = if_alias;
        let idx_str = if_index.to_string();
        let prefix = format!("{ula_addr}/128");
        let metric_arg = format!("metric={metric}");
        // First try add, then set if it already exists
        let (ok, msg) = run_os_command(
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
            run_os_command(
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
        run_os_command(
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

    // Group verified peers by interface_index
    let mut peer_by_if: HashMap<u32, String> = HashMap::new();
    for d in &discovered {
        if d.verified_handshake {
            if let Some(ref p) = d.peer_name {
                peer_by_if.entry(d.interface_index).or_insert_with(|| p.clone());
            }
        }
    }
    if let Some(hint) = peer_hint {
        if !hint.trim().is_empty() && snapshot.adapters.len() == 1 {
            let only_if = snapshot.adapters[0].interface_index;
            if let Ok(hid) = resolve_peer_identity(Some(hint), None, registry_path) {
                peer_by_if.entry(only_if).or_insert(hid.computer_name);
            }
        }
    }

    let mut actions = Vec::new();

    for adapter in &snapshot.adapters {
        let if_idx = adapter.interface_index;
        let alias = &adapter.interface_alias;

        // 1. Check MTU & Metric (idempotent — never reset socket when already matching!)
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
            let (ok, msg) = apply_mtu_and_metric(if_idx, alias, mtu_bytes, interface_metric);
            mtu_action.applied = ok;
            if !ok {
                mtu_action.error = Some(msg);
            }
        }
        actions.push(mtu_action);

        // 2. Local Deterministic Link-Local Addresses (fe80::d700:...)
        for lla in &local_id.all_link_local_addresses {
            let exists = adapter.addresses.iter().any(|a| a.address.eq_ignore_ascii_case(lla));
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
                let (ok, msg) = apply_ipv6_address_on_interface(if_idx, alias, lla, 64);
                act.applied = ok;
                if !ok {
                    act.error = Some(msg);
                }
            }
            actions.push(act);
        }

        // 3. Local Deterministic ULA Addresses (/128 host addresses for multi-port support)
        for ula in &local_id.all_ula_addresses {
            let exists = adapter.addresses.iter().any(|a| a.address.eq_ignore_ascii_case(ula));
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
                let (ok, msg) = apply_ipv6_address_on_interface(if_idx, alias, ula, 128);
                act.applied = ok;
                if !ok {
                    act.error = Some(msg);
                }
            }
            actions.push(act);
        }

        // 4. Per-Port On-Link /128 ULA Routes & Optional Pairwise /30 IPv4 for Discovered Peer
        if let Some(peer_name) = peer_by_if.get(&if_idx) {
            if let Ok(peer_id) = resolve_peer_identity(Some(peer_name), Some(&local_id.computer_name), registry_path) {
                for peer_ula in &peer_id.all_ula_addresses {
                    let mut r_act = PlannedAction {
                        interface_index: if_idx,
                        interface_alias: alias.clone(),
                        category: "PeerIPv6ULARoute128".to_string(),
                        target: format!("{peer_ula}/128 -> ifIndex {if_idx} ({peer_name})"),
                        already_in_desired_state: false,
                        applied: false,
                        command_summary: format!(
                            "Ensure on-link /128 route for {peer_ula}/128 via {alias} (ifIndex {if_idx})"
                        ),
                        error: None,
                    };
                    if apply {
                        let (ok, msg) = apply_ipv6_host_route(if_idx, alias, peer_ula, interface_metric);
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
        }
    }

    Ok(ThunderboltRegistrationReport {
        computer_name: local_id.computer_name.clone(),
        os_family: env::consts::OS.to_string(),
        architecture: env::consts::ARCH.to_string(),
        apply_mode: apply,
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
}

pub fn install_framework(apply: bool, explicit_registry: Option<&Path>) -> Result<InstallFrameworkReport> {
    // Validate before copying any file; malformed explicit input must not partially install.
    let (_, loaded_reg_path) = crate::identity::load_peer_registry(explicit_registry)?;
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
    if cfg!(target_os = "windows") && Path::new(r"C:\codedev\PC_AI").is_dir() {
        target_bins.push(PathBuf::from(r"C:\codedev\PC_AI\bin").join(bin_name));
    }

    let mut installed_bins = Vec::new();
    for dst in target_bins {
        if dst == exe_path {
            installed_bins.push(dst.to_string_lossy().into_owned());
            continue;
        }
        if apply {
            copy_install_file(&exe_path, &dst)?;
            installed_bins.push(dst.to_string_lossy().into_owned());
        } else {
            installed_bins.push(format!("(dry-run) {}", dst.to_string_lossy()));
        }
    }

    let mut installed_configs = Vec::new();
    if let Some(src_reg) = loaded_reg_path {
        let src_path = PathBuf::from(&src_reg);
        if !home.is_empty() {
            let hp = PathBuf::from(&home);
            for dst_cfg in [
                hp.join(".machine").join("thunderbolt-peers.json"),
                hp.join(".config").join("pcai").join("thunderbolt-peers.json"),
            ] {
                if dst_cfg == src_path {
                    installed_configs.push(dst_cfg.to_string_lossy().into_owned());
                    continue;
                }
                if apply {
                    copy_install_file(&src_path, &dst_cfg)?;
                    installed_configs.push(dst_cfg.to_string_lossy().into_owned());
                } else {
                    installed_configs.push(format!("(dry-run) {}", dst_cfg.to_string_lossy()));
                }
            }
        }
    }

    Ok(InstallFrameworkReport {
        computer_name: crate::identity::local_hostname(),
        os_family: env::consts::OS.to_string(),
        architecture: env::consts::ARCH.to_string(),
        source_binary: exe_path.to_string_lossy().into_owned(),
        installed_binaries: installed_bins,
        installed_configs,
        apply_mode: apply,
    })
}

fn copy_install_file(source: &Path, destination: &Path) -> Result<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create installation directory {}", parent.display()))?;
    }
    fs::copy(source, destination)
        .with_context(|| format!("install {} to {}", source.display(), destination.display()))?;
    Ok(())
}

/// Reject failed OS actions after emitting the detailed apply report.
///
/// # Errors
/// Returns an error when any attempted action recorded an OS failure.
pub fn check_applied_actions(actions: &[PlannedAction]) -> Result<()> {
    let failures = actions.iter().filter(|action| action.error.is_some()).count();
    anyhow::ensure!(
        failures == 0,
        "{failures} Thunderbolt apply action(s) failed; inspect the report"
    );
    Ok(())
}

#[cfg(test)]
mod failure_tests {
    use super::*;
    #[test]
    fn failed_copy_is_not_reported_as_install_success() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("missing-source");
        assert!(copy_install_file(&source, &dir.path().join("dest")).is_err());
        assert!(!dir.path().join("dest").exists());
    }
    #[test]
    fn successful_copy_preserves_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        fs::write(&source, b"binary fixture").unwrap();
        let dest = dir.path().join("nested/dest");
        copy_install_file(&source, &dest).unwrap();
        assert_eq!(fs::read(dest).unwrap(), b"binary fixture");
    }
    #[test]
    fn failed_apply_is_an_error_even_if_another_action_succeeded() {
        let mut action = PlannedAction {
            interface_index: 1,
            interface_alias: "fixture".into(),
            category: "fixture".into(),
            target: "fixture".into(),
            already_in_desired_state: false,
            applied: true,
            command_summary: "fixture".into(),
            error: None,
        };
        assert!(check_applied_actions(&[action.clone()]).is_ok());
        action.error = Some("permission denied".into());
        action.applied = false;
        assert!(check_applied_actions(&[action]).is_err());
    }
}
