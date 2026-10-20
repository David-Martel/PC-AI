//! Activate an explicitly adopted NetworkManager profile for a freshly enrolled edge.
use crate::{edge_observer::Observation, edge_policy::EdgePolicy};
use anyhow::Result;
#[cfg(any(target_os = "linux", test))]
use anyhow::{ensure, Context};
use serde_json::Value;
#[cfg(any(target_os = "linux", test))]
use std::net::Ipv6Addr;

#[cfg(any(target_os = "linux", test))]
fn validate_profile(output: &str, edge_id: &str, local_ula: &str) -> Result<()> {
    let fields: Vec<_> = output.lines().collect();
    ensure!(fields.len() == 6, "Expected exactly six adopted profile settings");
    ensure!(
        fields[0] == edge_id,
        "Adopted profile stable-id does not match enrolled edge"
    );
    ensure!(fields[1] == "no", "Adopted profile must disable autoconnect");
    ensure!(fields[2] == "disabled", "Adopted profile must disable IPv4");
    ensure!(fields[3] == "manual", "Adopted profile requires manual IPv6");
    let (address, prefix) = fields[4].split_once('/').context("Missing adopted IPv6 prefix")?;
    ensure!(prefix == "64", "Adopted profile requires exactly /64");
    ensure!(
        address.parse::<Ipv6Addr>()? == local_ula.parse::<Ipv6Addr>()?,
        "Adopted IPv6 address differs from enrolled edge"
    );
    ensure!(fields[5] == "yes", "Adopted profile must prohibit default routes");
    Ok(())
}

#[cfg(any(target_os = "linux", test))]
fn validate_inert_settings(output: &str) -> Result<()> {
    let fields: Vec<_> = output.lines().collect();
    ensure!(
        fields.len() == 19,
        "Expected exactly nineteen adopted profile custody, routing and DNS settings"
    );
    for (index, name) in [
        "connection.controller",
        "connection.master",
        "connection.port-type",
        "connection.slave-type",
        "connection.secondaries",
        "ipv4.routes",
        "ipv6.routes",
        "ipv4.gateway",
        "ipv6.gateway",
        "ipv4.routing-rules",
        "ipv6.routing-rules",
        "ipv4.dns",
        "ipv6.dns",
        "ipv4.dns-search",
        "ipv6.dns-search",
    ]
    .iter()
    .enumerate()
    {
        ensure!(fields[index].is_empty(), "Adopted profile must leave {name} empty");
    }
    ensure!(
        fields[15] == "0" && fields[16] == "0",
        "Adopted profile must not change DNS priority"
    );
    ensure!(fields[17] == "yes", "Adopted profile must prohibit IPv4 default routes");
    ensure!(
        fields[18] == "802-3-ethernet",
        "Adopted profile requires Ethernet connection type"
    );
    Ok(())
}

#[cfg(any(target_os = "linux", test))]
fn validate_active(output: &str, uuid: &str, interface: &str) -> Result<()> {
    for line in output.lines() {
        let (active_uuid, device) = line.split_once(':').context("Invalid active profile inventory")?;
        ensure!(!device.contains(':'), "Invalid active device field");
        if active_uuid == uuid {
            ensure!(device == interface, "Adopted profile is active on another device");
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn fresh(policy: &EdgePolicy, expected: &Observation) -> Result<()> {
    let observations = crate::edge_observer::observe(std::path::Path::new("/sys"), policy)?;
    let found = observations
        .iter()
        .find(|o| o.interface == expected.interface && o.ifindex == expected.ifindex)
        .context("Enrolled interface disappeared or changed ifindex")?;
    ensure!(
        found.state == "ENROLLED_UNVERIFIED" && found.edge == expected.edge,
        "Enrolled interface identity or custody changed"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn command(program: &str, args: &[&str], deadline: std::time::Instant) -> Result<String> {
    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
    let millis = u64::try_from(remaining.as_millis()).unwrap_or(5000).min(5000);
    ensure!(millis > 0, "NetworkManager activation deadline exceeded");
    let (status, stdout, stderr) =
        crate::switch::run_bounded_command_with_deadline(std::path::Path::new(program), args, millis, 256 * 1024)?;
    ensure!(status == Some(0), "Provider exited {status:?}: {stderr}");
    Ok(stdout)
}

#[cfg(target_os = "linux")]
fn address_ready(interface: &str, ifindex: u32, address: &str, deadline: std::time::Instant) -> Result<bool> {
    let output = command(
        "/usr/sbin/ip",
        &["-j", "-6", "addr", "show", "dev", interface],
        deadline,
    )?;
    let links: Vec<Value> = serde_json::from_str(&output).context("Invalid IPv6 address inventory")?;
    let expected: Ipv6Addr = address.parse()?;
    for link in links {
        ensure!(
            link.get("ifname").and_then(Value::as_str) == Some(interface)
                && link.get("ifindex").and_then(Value::as_u64) == Some(u64::from(ifindex)),
            "Address inventory interface mismatch"
        );
        for addr in link
            .get("addr_info")
            .and_then(Value::as_array)
            .context("Missing address inventory")?
        {
            if addr
                .get("local")
                .and_then(Value::as_str)
                .and_then(|s| s.parse::<Ipv6Addr>().ok())
                == Some(expected)
                && addr.get("prefixlen").and_then(Value::as_u64) == Some(64)
            {
                let unsafe_flag = |name: &str| {
                    addr.get(name).and_then(Value::as_bool) == Some(true)
                        || addr
                            .get("flags")
                            .and_then(Value::as_array)
                            .is_some_and(|f| f.iter().any(|v| v.as_str() == Some(name)))
                };
                return Ok(!unsafe_flag("tentative") && !unsafe_flag("dadfailed"));
            }
        }
    }
    Ok(false)
}

/// Activate one adopted profile after validating ownership, settings, and live controller identity.
pub fn activate(policy: &EdgePolicy, observation: &Observation) -> Result<Value> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (policy, observation);
        anyhow::bail!("NetworkManager activation requires Linux")
    }
    #[cfg(target_os = "linux")]
    {
        crate::edge_policy::validate_policy(policy)?;
        ensure!(
            policy.backend == "networkmanager",
            "Activation requires networkmanager backend"
        );
        ensure!(
            policy.adopt_profile_uuids.len() == 1,
            "Activation requires exactly one explicitly adopted profile"
        );
        ensure!(
            observation.state == "ENROLLED_UNVERIFIED",
            "Only enrolled unverified edges may activate"
        );
        let edge = observation.edge.as_ref().context("Missing enrolled edge identity")?;
        let interface = observation.interface.as_str();
        ensure!(
            !interface.is_empty()
                && interface.len() <= 15
                && interface
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
                && !interface.starts_with('-'),
            "Unsafe interface name"
        );
        let uuid = policy.adopt_profile_uuids[0].as_str();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        let nm = |args: &[&str]| command("/usr/bin/nmcli", args, deadline);
        let settings = nm(&[
            "--escape",
            "no",
            "-g",
            "connection.stable-id,connection.autoconnect,ipv4.method,ipv6.method,ipv6.addresses,ipv6.never-default",
            "connection",
            "show",
            "uuid",
            uuid,
        ])?;
        validate_profile(&settings, &edge.edge_id, &edge.local_ula)?;
        let inert_settings = nm(&[
            "--escape", "no", "-g",
            "connection.controller,connection.master,connection.port-type,connection.slave-type,connection.secondaries,ipv4.routes,ipv6.routes,ipv4.gateway,ipv6.gateway,ipv4.routing-rules,ipv6.routing-rules,ipv4.dns,ipv6.dns,ipv4.dns-search,ipv6.dns-search,ipv4.dns-priority,ipv6.dns-priority,ipv4.never-default,connection.type",
            "connection", "show", "uuid", uuid,
        ])?;
        validate_inert_settings(&inert_settings)?;
        let current = nm(&["-g", "GENERAL.CON-UUID", "device", "show", interface])?;
        let current = current.trim();
        ensure!(
            current == uuid || current.is_empty() || current == "--",
            "Interface has a foreign active profile"
        );
        let active = nm(&[
            "--escape",
            "no",
            "-t",
            "-f",
            "UUID,DEVICE",
            "connection",
            "show",
            "--active",
        ])?;
        validate_active(&active, uuid, interface)?;
        let binding = nm(&[
            "--escape",
            "no",
            "-g",
            "connection.interface-name",
            "connection",
            "show",
            "uuid",
            uuid,
        ])?;
        fresh(policy, observation)?;
        if current == uuid && address_ready(interface, observation.ifindex, &edge.local_ula, deadline)? {
            return Ok(
                serde_json::json!({"state":"ACTIVE_UNVERIFIED", "profile_uuid":uuid, "interface":interface, "ifindex":observation.ifindex, "edge_id":edge.edge_id, "already_active":true}),
            );
        }
        if binding.trim() != interface {
            nm(&[
                "connection",
                "modify",
                "uuid",
                uuid,
                "connection.interface-name",
                interface,
            ])?;
        }
        fresh(policy, observation)?;
        nm(&["--wait", "8", "connection", "up", "uuid", uuid, "ifname", interface])?;
        fresh(policy, observation)?;
        ensure!(
            nm(&["-g", "GENERAL.CON-UUID", "device", "show", interface])?.trim() == uuid,
            "Activated device does not own the adopted profile"
        );
        ensure!(
            address_ready(interface, observation.ifindex, &edge.local_ula, deadline)?,
            "Activation returned without a usable enrolled IPv6 address"
        );
        Ok(
            serde_json::json!({"state":"ACTIVE_UNVERIFIED", "profile_uuid":uuid, "interface":interface, "ifindex":observation.ifindex, "edge_id":edge.edge_id, "already_active":false}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Protect adopted profile boundaries; detects foreign identity and route/IP drift.
    // Needs pure settings fixtures; breadcrumb: nm_edge::validate_profile.
    #[test]
    fn strict_profile_settings_reject_drift() {
        let good = "edge\nno\ndisabled\nmanual\nfd12::1/64\nyes\n";
        assert!(validate_profile(good, "edge", "fd12::1").is_ok());
        for bad in [
            good.replace("edge", "foreign"),
            good.replace("no", "yes"),
            good.replace("disabled", "auto"),
            good.replace("manual", "auto"),
            good.replace("::1", "::2"),
            good.replace("/64", "/48"),
            good.replace("yes", "no"),
            format!("{good}extra\n"),
        ] {
            assert!(validate_profile(&bad, "edge", "fd12::1").is_err(), "accepted {bad:?}");
        }
    }
    // Protect foreign custody, routing and DNS; detects each injected side effect.
    // Needs pure settings fixtures; breadcrumb: nm_edge::validate_inert_settings.
    #[test]
    fn refuses_routes_controller_and_secondary_profiles() {
        let mut fields = vec![""; 15];
        fields.extend(["0", "0", "yes", "802-3-ethernet"]);
        let good = format!("{}\n", fields.join("\n"));
        assert!(validate_inert_settings(&good).is_ok());
        for index in 0..15 {
            let mut injected = fields.clone();
            injected[index] = "foreign-setting";
            assert!(
                validate_inert_settings(&injected.join("\n")).is_err(),
                "accepted nonempty setting {index}"
            );
        }
        for (index, value) in [(15, "-100"), (16, "-100"), (17, "no"), (18, "bridge")] {
            let mut injected = fields.clone();
            injected[index] = value;
            assert!(validate_inert_settings(&injected.join("\n")).is_err());
        }
        assert!(validate_inert_settings("yes\n802-3-ethernet").is_err());
        assert!(validate_inert_settings(&format!("{good}extra\n")).is_err());
    }
    // Protect other devices from adopted-profile theft; detects mismatched active ownership.
    // Needs pure active inventory; breadcrumb: nm_edge::validate_active.
    #[test]
    fn refuses_profile_active_elsewhere() {
        assert!(validate_active("uuid:usb4edge\nother:eth0\n", "uuid", "usb4edge").is_ok());
        assert!(validate_active("uuid:eth0\n", "uuid", "usb4edge").is_err());
        assert!(validate_active("malformed", "uuid", "usb4edge").is_err());
    }
}
