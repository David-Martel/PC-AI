//! Observe enrolled root-namespace edges without changing network state.
use crate::edge_policy::{derive_edge, derive_edge_plan, DerivedEdge, EdgePolicy};
use anyhow::{Context, Result};
use serde::Serialize;
use std::{fs, path::Path};

#[derive(Debug, Serialize)]
pub struct Observation {
    pub interface: String,
    pub ifindex: u32,
    pub state: String,
    pub reason: String,
    pub edge: Option<DerivedEdge>,
}

/// Inspect sysfs while preserving unknown peers and externally owned interfaces.
pub fn observe(sys: &Path, policy: &EdgePolicy) -> Result<Vec<Observation>> {
    derive_edge_plan(policy, &policy.local_controller_uuids)?;
    let mut result = Vec::new();
    for entry in fs::read_dir(sys.join("class/net")).context("Network inventory unavailable")? {
        let entry = entry?;
        let device = entry.path().join("device");
        let Ok(driver) = fs::read_link(device.join("driver")) else {
            continue;
        };
        if driver.file_name().and_then(|n| n.to_str()) != Some("thunderbolt-net") {
            continue;
        }
        let interface = entry.file_name().to_string_lossy().into_owned();
        let ifindex = fs::read_to_string(entry.path().join("ifindex"))?.trim().parse()?;
        let mut observation = Observation {
            interface,
            ifindex,
            state: "UNKNOWN".into(),
            reason: String::new(),
            edge: None,
        };
        let inspect = || -> Result<DerivedEdge> {
            let canonical = fs::canonicalize(&device)?;
            let remote = fs::read_to_string(canonical.parent().context("Missing XDomain parent")?.join("unique_id"))?;
            let domain = canonical
                .ancestors()
                .filter_map(|p| p.file_name()?.to_str())
                .find_map(|n| {
                    n.strip_prefix("domain")
                        .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
                })
                .context("Missing controller domain")?;
            let local = fs::read_to_string(sys.join(format!("bus/thunderbolt/devices/{domain}-0/unique_id")))?;
            derive_edge(policy, local.trim(), remote.trim())
        };
        match inspect() {
            Ok(edge) => {
                observation.edge = Some(edge);
                if policy.protected_interfaces.contains(&observation.interface) || entry.path().join("master").exists()
                {
                    observation.state = "PROTECTED".into();
                    observation.reason = "Explicit protection or existing master; no adoption allowed".into();
                } else {
                    observation.state = "ENROLLED_UNVERIFIED".into();
                    observation.reason =
                        "Controller identity matches enrollment; address ownership and pinned SSH remain unverified"
                            .into();
                }
            }
            Err(error) => observation.reason = error.to_string(),
        }
        result.push(observation);
    }
    result.sort_by(|a, b| a.interface.cmp(&b.interface));
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    // Protects enrolled ancestry and foreign custody; detects name-based identity and unknown-peer promotion.
    // Needs POSIX symlinks only; breadcrumb: observe, no network devices or privileges required.
    #[test]
    fn actual_sysfs_ancestry_enrollment_protection_and_unknown_peer() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let device = root.join("devices/domain1/1-0/1-301/1-301.0");
        fs::create_dir_all(&device).unwrap();
        let net = root.join("class/net/renamed-edge");
        fs::create_dir_all(&net).unwrap();
        let controller = root.join("bus/thunderbolt/devices/1-0");
        fs::create_dir_all(&controller).unwrap();
        fs::write(controller.join("unique_id"), "22222222-2222-2222-2222-222222222222\n").unwrap();
        fs::write(
            device.parent().unwrap().join("unique_id"),
            "33333333-3333-3333-3333-333333333333\n",
        )
        .unwrap();
        fs::write(net.join("ifindex"), "42\n").unwrap();
        symlink(&device, net.join("device")).unwrap();
        symlink("/drivers/thunderbolt-net", device.join("driver")).unwrap();
        let mut policy: EdgePolicy = serde_json::from_value(serde_json::json!({"schema":1,"epoch":1,"node_id":"11111111-1111-1111-1111-111111111111","node_name":"test","ula_prefix":"fd12:3456:789a::/48","backend":"networkd","local_controller_uuids":["22222222-2222-2222-2222-222222222222"],"ssh_user":null,"peers":[{"node_id":"44444444-4444-4444-4444-444444444444","node_name":"peer","controller_uuids":["33333333-3333-3333-3333-333333333333"],"ssh_alias":null,"mtu":1500}],"protected_interfaces":[],"adopt_profile_uuids":[]})).unwrap();
        let observations = observe(root, &policy).unwrap();
        assert_eq!(observations[0].state, "ENROLLED_UNVERIFIED");
        assert_eq!(observations[0].ifindex, 42);
        policy.protected_interfaces.push("renamed-edge".into());
        assert_eq!(observe(root, &policy).unwrap()[0].state, "PROTECTED");
        policy.protected_interfaces.clear();
        symlink(root, net.join("master")).unwrap();
        assert_eq!(observe(root, &policy).unwrap()[0].state, "PROTECTED");
        fs::write(
            device.parent().unwrap().join("unique_id"),
            "55555555-5555-5555-5555-555555555555",
        )
        .unwrap();
        let unknown = observe(root, &policy).unwrap();
        assert_eq!(unknown[0].state, "UNKNOWN");
        assert!(unknown[0].edge.is_none());
    }
    // Protects provider failure semantics; detects a missing inventory reported as empty success.
    // Needs temporary local storage; breadcrumb: observe/read_dir.
    #[test]
    fn missing_inventory_is_error_not_empty_success() {
        let policy: EdgePolicy = serde_json::from_value(serde_json::json!({"schema":1,"epoch":1,"node_id":"11111111-1111-1111-1111-111111111111","node_name":"test","ula_prefix":"fd12:3456:789a::/48","backend":"networkd","local_controller_uuids":["22222222-2222-2222-2222-222222222222"],"ssh_user":null,"peers":[],"protected_interfaces":[],"adopt_profile_uuids":[]})).unwrap();
        let dir = tempfile::tempdir().unwrap();
        assert!(observe(dir.path(), &policy)
            .unwrap_err()
            .to_string()
            .contains("inventory unavailable"));
    }
}
