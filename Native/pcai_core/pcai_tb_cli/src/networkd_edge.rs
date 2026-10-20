//! Reconcile an exact owned networkd profile after enrolled device re-enumeration.
#[cfg(target_os = "linux")]
use crate::{edge_observer::observe, switch::run_bounded_command_with_deadline};
use crate::{
    edge_observer::Observation,
    edge_policy::{DerivedEdge, EdgePolicy},
};
#[cfg(any(target_os = "linux", test))]
use anyhow::ensure;
#[cfg(any(target_os = "linux", test))]
use anyhow::Context;
use anyhow::Result;
use serde_json::Value;
#[cfg(any(target_os = "linux", test))]
use std::fs;
#[cfg(any(target_os = "linux", test))]
use std::path::Path;

/// Render an IPv6-only profile matched to an enrolled physical edge.
pub fn render(edge: &DerivedEdge, epoch: u64) -> String {
    format!("# PC-AI enrolled USB4 edge; policy epoch {epoch}\n[Match]\nDriver=thunderbolt-net\nProperty=VIGIL_USB4_EDGE_ID={}\n\n[Link]\nMTUBytes={}\nRequiredForOnline=no\n\n[Network]\nAddress={}/64\nLinkLocalAddressing=ipv6\nDHCP=no\nIPv6AcceptRA=no\nIPv6SendRA=no\n", edge.edge_id, edge.mtu, edge.local_ula)
}

#[cfg(any(target_os = "linux", test))]
fn no_dropins(base: &Path, filename: &str) -> Result<()> {
    let mut names = vec![format!("{filename}.d"), "network.d".into()];
    let stem = filename.strip_suffix(".network").context("Expected .network profile")?;
    for (index, _) in stem.match_indices('-') {
        names.push(format!("{}.network.d", &stem[..=index]));
    }
    for name in names {
        match fs::read_dir(base.join(name)) {
            Ok(entries) => {
                for entry in entries {
                    ensure!(
                        entry?.path().extension().and_then(|v| v.to_str()) != Some("conf"),
                        "Effective networkd drop-in requires review"
                    );
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

#[cfg(any(target_os = "linux", test))]
fn custody(value: &Value, profile: &Path, observation: &Observation) -> Result<()> {
    ensure!(
        value["Name"].as_str() == Some(&observation.interface)
            && value["Index"].as_u64() == Some(u64::from(observation.ifindex)),
        "networkd inventory identity mismatch"
    );
    if let Some(owner) = value["NetworkFile"].as_str() {
        ensure!(
            Some(owner) == profile.to_str(),
            "Foreign networkd profile owns the interface"
        );
    } else {
        ensure!(
            value["AdministrativeState"].as_str() == Some("unmanaged"),
            "Networkd ownership is uncertain"
        );
    }
    Ok(())
}

/// Import fresh udev identity and reconfigure only the exact owned enrolled edge.
pub fn reconcile(policy: &EdgePolicy, observation: &Observation) -> Result<Value> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (policy, observation);
        anyhow::bail!("networkd reconciliation requires Linux");
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::MetadataExt;
        ensure!(
            policy.backend == "networkd" && observation.state == "ENROLLED_UNVERIFIED",
            "Only enrolled networkd edges may reconcile"
        );
        let edge = observation.edge.as_ref().context("Missing enrolled edge")?;
        let expected_address = serde_json::json!(edge.local_ula.parse::<std::net::Ipv6Addr>()?.octets());
        let expected = render(edge, policy.epoch);
        let mut profiles = Vec::new();
        for entry in fs::read_dir("/etc/systemd/network")? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("network") {
                continue;
            }
            let meta = fs::symlink_metadata(&path)?;
            if meta.is_file() && meta.uid() == 0 && meta.mode() & 0o022 == 0 && fs::read(&path)? == expected.as_bytes()
            {
                profiles.push(path);
            }
        }
        ensure!(
            profiles.len() == 1,
            "Requires exactly one root-owned exact rendered profile"
        );
        let fresh = || -> Result<()> {
            ensure!(
                fs::read(&profiles[0])? == expected.as_bytes(),
                "Owned networkd profile changed"
            );
            let filename = profiles[0]
                .file_name()
                .and_then(|v| v.to_str())
                .context("Invalid profile filename")?;
            for directory in [
                "/etc/systemd/network",
                "/run/systemd/network",
                "/usr/local/lib/systemd/network",
                "/usr/lib/systemd/network",
            ] {
                no_dropins(Path::new(directory), filename)?;
            }
            ensure!(
                observe(Path::new("/sys"), policy)?
                    .iter()
                    .any(|v| v.interface == observation.interface
                        && v.ifindex == observation.ifindex
                        && v.state == observation.state
                        && v.edge == observation.edge),
                "Enrolled interface changed"
            );
            Ok(())
        };
        let command = |program: &str, args: &[&str]| -> Result<String> {
            let (exit, output, error) = run_bounded_command_with_deadline(Path::new(program), args, 5_000, 256 * 1024)?;
            ensure!(exit == Some(0), "Provider failed: {}", error.trim());
            Ok(output)
        };
        fresh()?;
        ensure!(
            command(
                "/usr/bin/nmcli",
                &["-g", "GENERAL.NM-MANAGED", "device", "show", &observation.interface]
            )?
            .trim()
                == "no",
            "NetworkManager custody must be explicitly unmanaged"
        );
        let before: Value = serde_json::from_str(&command(
            "/usr/bin/networkctl",
            &["--json=short", "status", &observation.interface],
        )?)?;
        custody(&before, &profiles[0], observation)?;
        let properties = command(
            "/usr/bin/udevadm",
            &[
                "info",
                "--query=property",
                &format!("/sys/class/net/{}", observation.interface),
            ],
        )?;
        let property = format!("VIGIL_USB4_EDGE_ID={}", edge.edge_id);
        if before["NetworkFile"].as_str() == profiles[0].to_str()
            && before["MTU"].as_u64() == Some(u64::from(edge.mtu))
            && properties.lines().any(|p| p == property)
            && before["Addresses"].as_array().is_some_and(|addresses| {
                addresses.iter().any(|a| {
                    a["Address"] == expected_address
                        && a["PrefixLength"].as_u64() == Some(64)
                        && a["Family"].as_u64() == Some(10)
                        && a["ConfigSource"].as_str() == Some("static")
                        && a["ConfigState"].as_str() == Some("configured")
                })
            })
        {
            return Ok(
                serde_json::json!({"state":"CONFIGURED_UNVERIFIED","already_configured":true,"interface":observation.interface,"ifindex":observation.ifindex}),
            );
        }
        fresh()?;
        command(
            "/usr/bin/udevadm",
            &[
                "trigger",
                "--action=change",
                &format!("/sys/class/net/{}", observation.interface),
            ],
        )?;
        command("/usr/bin/udevadm", &["settle", "--timeout=3"])?;
        fresh()?;
        let properties = command(
            "/usr/bin/udevadm",
            &[
                "info",
                "--query=property",
                &format!("/sys/class/net/{}", observation.interface),
            ],
        )?;
        ensure!(
            properties.lines().any(|p| p == property),
            "Fresh enrollment property was not imported"
        );
        ensure!(
            command(
                "/usr/bin/nmcli",
                &["-g", "GENERAL.NM-MANAGED", "device", "show", &observation.interface]
            )?
            .trim()
                == "no",
            "NetworkManager acquired foreign custody"
        );
        let current: Value = serde_json::from_str(&command(
            "/usr/bin/networkctl",
            &["--json=short", "status", &observation.interface],
        )?)?;
        custody(&current, &profiles[0], observation)?;
        fresh()?;
        command("/usr/bin/networkctl", &["reconfigure", &observation.interface])?;
        fresh()?;
        Ok(
            serde_json::json!({"state":"CONFIGURING_UNVERIFIED","already_configured":false,"interface":observation.interface,"ifindex":observation.ifindex}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Protect effective config; detects a drop-in overriding an exact main profile.
    // Needs temporary filesystem; breadcrumb: networkd_edge::no_dropins.
    #[test]
    fn refuses_profile_and_prefix_dropins() {
        let root = tempfile::tempdir().unwrap();
        assert!(no_dropins(root.path(), "05-owned.network").is_ok());
        for name in ["05-owned.network.d", "05-.network.d", "network.d"] {
            let directory = root.path().join(name);
            fs::create_dir(&directory).unwrap();
            fs::write(directory.join("override.conf"), "[Network]\nBridge=foreign\n").unwrap();
            assert!(no_dropins(root.path(), "05-owned.network").is_err());
            fs::remove_dir_all(directory).unwrap();
        }
    }
    // Protect manager custody; detects foreign profiles and reused interface indices before mutation.
    // Needs pure JSON fixtures; breadcrumb: networkd_edge::custody.
    #[test]
    fn rejects_foreign_manager_and_identity_drift() {
        let observed = Observation {
            interface: "edge0".into(),
            ifindex: 4,
            state: "ENROLLED_UNVERIFIED".into(),
            reason: String::new(),
            edge: None,
        };
        let mut value = serde_json::json!({"Name":"edge0","Index":4,"NetworkFile":"/etc/owned.network"});
        assert!(custody(&value, Path::new("/etc/owned.network"), &observed).is_ok());
        value["NetworkFile"] = Value::String("/etc/foreign.network".into());
        assert!(custody(&value, Path::new("/etc/owned.network"), &observed).is_err());
        value = serde_json::json!({"Name":"edge0","Index":4,"AdministrativeState":"unmanaged"});
        assert!(custody(&value, Path::new("/etc/owned.network"), &observed).is_ok());
        value["Index"] = Value::from(5);
        assert!(custody(&value, Path::new("/etc/owned.network"), &observed).is_err());
    }
}
