//! Qualify enrolled edges through source-bound routing and pinned SSH evidence.
use crate::{
    edge_observer::Observation,
    edge_policy::{EdgePolicy, PeerPlatform},
};
use anyhow::Result;
#[cfg(any(target_os = "linux", test))]
use anyhow::{ensure, Context};
use serde_json::Value;
#[cfg(any(target_os = "linux", test))]
use std::net::Ipv6Addr;

#[cfg(any(target_os = "linux", test))]
fn exact_address(value: &str, expected: &str) -> Result<()> {
    ensure!(
        value.parse::<Ipv6Addr>()? == expected.parse::<Ipv6Addr>()?,
        "Address does not match enrolled edge"
    );
    Ok(())
}

#[cfg(any(target_os = "linux", test))]
fn validate_route(output: &str, interface: &str, local: &str, peer: &str) -> Result<()> {
    let routes: Vec<Value> = serde_json::from_str(output)?;
    ensure!(routes.len() == 1, "Route lookup must return exactly one route");
    let route = &routes[0];
    ensure!(
        route.get("gateway").is_none() && route.get("via").is_none() && route.get("multipath").is_none(),
        "Enrolled edge requires a direct route without a gateway"
    );
    ensure!(
        route.get("dev").and_then(Value::as_str) == Some(interface),
        "Wrong route device"
    );
    exact_address(
        route
            .get("dst")
            .and_then(Value::as_str)
            .context("Missing route destination")?,
        peer,
    )?;
    let source = route
        .get("prefsrc")
        .or_else(|| route.get("src"))
        .or_else(|| route.get("from"))
        .and_then(Value::as_str)
        .context("Missing route preferred source")?;
    exact_address(source, local)?;
    for key in ["prefsrc", "src", "from"] {
        if let Some(value) = route.get(key) {
            exact_address(value.as_str().context("Invalid route source")?, local)?;
        }
    }
    ensure!(
        route.get("type").is_none_or(|v| v.as_str() == Some("unicast")),
        "Route must be unicast"
    );
    Ok(())
}

/// Remote evidence command for the peer's login shell: node id, blank line, then
/// `local_ula port peer_ula 22` (the `SSH_CONNECTION` layout `validate_ssh` expects).
#[cfg(any(target_os = "linux", test))]
fn remote_command(platform: PeerPlatform, local: &str, peer: &str) -> String {
    match platform {
        PeerPlatform::Linux => "cat /etc/pcai-usb4/node-id; printf '\n'; printf '%s\n' \"$SSH_CONNECTION\"".into(),
        PeerPlatform::Windows => {
            use base64::{engine::general_purpose::STANDARD, Engine as _};
            // WHY: sshd on Windows hands the string to PowerShell (or cmd), so POSIX
            // quoting and `$SSH_CONNECTION` do not work. -EncodedCommand (base64 UTF-16LE)
            // avoids every shell quoting layer; the interpolated values are canonical
            // IPv6 text from the derived edge, never free-form input.
            // Windows sshd may not export SSH_CONNECTION, so fall back to the established
            // port-22 socket whose endpoints are exactly the enrolled pair. The first match
            // is enough: any such socket proves the peer sees this source address, and
            // `Select-Object -First 1` keeps the query bounded.
            let script = format!(
                "$ErrorActionPreference='Stop';\
                 $id=(Get-Content -Raw -LiteralPath (Join-Path $env:ProgramData 'pcai-usb4\\node-id')).Trim();\
                 [Console]::Out.Write($id+\"`n`n\");\
                 if($env:SSH_CONNECTION){{[Console]::Out.Write($env:SSH_CONNECTION+\"`n\")}}\
                 else{{$c=Get-NetTCPConnection -LocalPort 22 -State Established -LocalAddress '{peer}' -RemoteAddress '{local}'|Select-Object -First 1;\
                 if($c){{[Console]::Out.Write(('{{0}} {{1}} {{2}} {{3}}' -f $c.RemoteAddress,$c.RemotePort,$c.LocalAddress,$c.LocalPort)+\"`n\")}}}}"
            );
            let utf16: Vec<u8> = script.encode_utf16().flat_map(u16::to_le_bytes).collect();
            format!(
                "powershell -NoProfile -NonInteractive -EncodedCommand {}",
                STANDARD.encode(utf16)
            )
        }
    }
}

#[cfg(any(target_os = "linux", test))]
fn validate_ssh(output: &str, node: &str, local: &str, peer: &str) -> Result<()> {
    let lines: Vec<&str> = output.lines().collect();
    ensure!(
        lines.len() == 2 || (lines.len() == 3 && lines[1].is_empty()),
        "Unexpected SSH evidence lines"
    );
    ensure!(lines[0] == node, "Remote node identity does not match enrollment");
    let connection: Vec<&str> = lines
        .last()
        .context("Missing SSH connection")?
        .split_whitespace()
        .collect();
    ensure!(connection.len() == 4, "Invalid SSH connection evidence");
    exact_address(connection[0], local)?;
    let source_port: u16 = connection[1].parse()?;
    ensure!(source_port > 0, "Invalid SSH source port");
    exact_address(connection[2], peer)?;
    ensure!(connection[3] == "22", "Unexpected SSH server port");
    Ok(())
}

/// Require current inventory, exact source routing, and authenticated peer evidence.
pub fn qualify(policy: &EdgePolicy, observation: &Observation) -> Result<Value> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (policy, observation);
        anyhow::bail!("Edge qualification is supported only on Linux");
    }
    #[cfg(target_os = "linux")]
    {
        use crate::{edge_observer::observe, edge_policy::validate_policy, switch::run_bounded_command_with_deadline};
        use std::path::Path;
        let namespace = std::fs::read_link("/proc/self/ns/net")?;
        validate_policy(policy)?;
        ensure!(
            observation.state == "ENROLLED_UNVERIFIED",
            "Only unverified enrolled edges can be qualified"
        );
        let edge = observation.edge.as_ref().context("Missing enrolled edge")?;
        let user = policy.ssh_user.as_deref().context("Enrolled SSH user is required")?;
        let alias = edge
            .ssh_alias
            .as_deref()
            .context("Enrolled pinned SSH alias is required")?;
        ensure!(
            observe(Path::new("/sys"), policy)?.iter().any(|current| {
                current.interface == observation.interface
                    && current.ifindex == observation.ifindex
                    && current.state == "ENROLLED_UNVERIFIED"
                    && current.edge.as_ref() == Some(edge)
            }),
            "Current interface identity or enrollment differs from observation"
        );
        // Five seconds bounds a local lookup; SSH has a three-second connect limit
        // and a ten-second total process deadline, including account switching.
        let (status, route, _) = run_bounded_command_with_deadline(
            Path::new("/usr/sbin/ip"),
            &["-6", "-j", "route", "get", &edge.peer_ula, "from", &edge.local_ula],
            5_000,
            16_384,
        )?;
        ensure!(status == Some(0), "Source-bound route lookup failed");
        validate_route(&route, &observation.interface, &edge.local_ula, &edge.peer_ula)?;
        let platform = policy
            .peers
            .iter()
            .find(|p| p.node_id.eq_ignore_ascii_case(&edge.peer_node_id))
            .map(|p| p.platform)
            .context("Enrolled peer for edge not found")?;
        let command = remote_command(platform, &edge.local_ula, &edge.peer_ula);
        let hostname = format!("HostName={}", edge.peer_ula);
        let bind = format!("BindAddress={}", edge.local_ula);
        let (status, evidence, stderr) = run_bounded_command_with_deadline(
            Path::new("/usr/sbin/runuser"),
            &[
                "-u",
                user,
                "--",
                "/usr/bin/ssh",
                "-T",
                "-oBatchMode=yes",
                "-oStrictHostKeyChecking=yes",
                "-oIdentitiesOnly=yes",
                "-oConnectTimeout=3",
                "-oProxyCommand=none",
                "-oProxyJump=none",
                "-oIdentityAgent=none",
                "-oUpdateHostKeys=no",
                "-oPermitLocalCommand=no",
                "-oControlMaster=no",
                "-oControlPath=none",
                "-oClearAllForwardings=yes",
                "-p",
                "22",
                "-o",
                &hostname,
                "-o",
                &bind,
                alias,
                &command,
            ],
            10_000,
            16_384,
        )?;
        ensure!(status == Some(0), "Pinned SSH qualification failed: {}", stderr.trim());
        validate_ssh(&evidence, &edge.peer_node_id, &edge.local_ula, &edge.peer_ula)?;
        ensure!(
            std::fs::read_link("/proc/self/ns/net")? == namespace,
            "Namespace changed during qualification"
        );
        ensure!(
            observe(Path::new("/sys"), policy)?
                .iter()
                .any(|current| current.interface == observation.interface
                    && current.ifindex == observation.ifindex
                    && current.state == "ENROLLED_UNVERIFIED"
                    && current.edge.as_ref() == Some(edge)),
            "Interface identity changed during qualification"
        );
        Ok(
            serde_json::json!({"state":"AUTHENTICATED", "interface": observation.interface,
            "ifindex":observation.ifindex, "edge":edge, "namespace":namespace, "policy_epoch":policy.epoch,"route":serde_json::from_str::<Value>(&route)?}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const LOCAL: &str = "fd42::1";
    const PEER: &str = "fd42::2";
    // Protects source/egress custody; detects wrong devices, sources, destinations and scoped addresses.
    // Needs JSON fixtures only; breadcrumb: edge_probe::validate_route.
    #[test]
    fn route_requires_exact_device_source_destination_and_unscoped_addresses() {
        let good = serde_json::json!([{"dst":PEER,"dev":"tb0","prefsrc":LOCAL,"from":LOCAL}]);
        validate_route(&good.to_string(), "tb0", LOCAL, PEER).unwrap();
        let from_only = serde_json::json!([{"dst":PEER,"dev":"tb0","from":LOCAL}]);
        validate_route(&from_only.to_string(), "tb0", LOCAL, PEER).unwrap();
        let missing_source = serde_json::json!([{"dst":PEER,"dev":"tb0"}]);
        assert!(validate_route(&missing_source.to_string(), "tb0", LOCAL, PEER).is_err());
        for (key, bad) in [
            ("dev", "tb1"),
            ("prefsrc", PEER),
            ("from", PEER),
            ("dst", LOCAL),
            ("dst", "fd42::2%tb0"),
            ("type", "local"),
            ("gateway", "fd42::3"),
            ("via", "fd42::3"),
        ] {
            let mut value = good.clone();
            value[0][key] = Value::String(bad.into());
            assert!(validate_route(&value.to_string(), "tb0", LOCAL, PEER).is_err(), "{key}");
        }
        assert!(validate_route("[]", "tb0", LOCAL, PEER).is_err());
    }
    // Protects authenticated edge identity; detects wrong node, source, destination, scope and SSH port.
    // Needs fixed SSH stdout fixtures only; breadcrumb: edge_probe::validate_ssh.
    #[test]
    fn ssh_evidence_requires_exact_node_and_connection_endpoints() {
        validate_ssh("node\n\nfd42::1 49152 fd42::2 22\n", "node", LOCAL, PEER).unwrap();
        validate_ssh("node\nfd42::1 49152 fd42::2 22\n", "node", LOCAL, PEER).unwrap();
        for bad in [
            "other\nfd42::1 49152 fd42::2 22\n",
            "node\nfd42::3 49152 fd42::2 22\n",
            "node\nfd42::1 49152 fd42::3 22\n",
            "node\nfd42::1%tb0 49152 fd42::2 22\n",
            "node\nfd42::1 49152 fd42::2 2222\n",
            "node\nfd42::1 0 fd42::2 22\n",
            "node\nfd42::1 49152 fd42::2 22\nextra\n",
            "",
        ] {
            assert!(validate_ssh(bad, "node", LOCAL, PEER).is_err(), "{bad}");
        }
    }

    // Protects the Windows remote command; detects quoting regressions and drift from the Linux command.
    // Needs no processes; breadcrumb: edge_probe::remote_command.
    #[test]
    fn remote_command_is_posix_for_linux_and_encoded_powershell_for_windows() {
        assert_eq!(
            remote_command(PeerPlatform::Linux, LOCAL, PEER),
            "cat /etc/pcai-usb4/node-id; printf '\n'; printf '%s\n' \"$SSH_CONNECTION\""
        );
        let command = remote_command(PeerPlatform::Windows, LOCAL, PEER);
        let encoded = command
            .strip_prefix("powershell -NoProfile -NonInteractive -EncodedCommand ")
            .unwrap();
        assert!(encoded
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=')));
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        let units: Vec<u16> = STANDARD
            .decode(encoded)
            .unwrap()
            .chunks(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let script = String::from_utf16(&units).unwrap();
        assert!(script.contains(r"Join-Path $env:ProgramData 'pcai-usb4\node-id'"));
        assert!(script.contains("-LocalPort 22 -State Established -LocalAddress 'fd42::2' -RemoteAddress 'fd42::1'"));
        assert!(script.contains("Select-Object -First 1"));
    }

    // Protects Windows parity; detects accepting wrong identity/endpoints once CRLF output from PowerShell is allowed.
    // Needs fixed stdout fixtures; positive control: the good case must pass so the rejections are meaningful.
    #[test]
    fn windows_evidence_is_accepted_and_wrong_node_or_endpoints_are_rejected() {
        validate_ssh("node\r\n\r\nfd42::1 49152 fd42::2 22\r\n", "node", LOCAL, PEER).unwrap();
        for bad in [
            "other\r\n\r\nfd42::1 49152 fd42::2 22\r\n",
            "node\r\n\r\nfd42::3 49152 fd42::2 22\r\n",
            "node\r\n\r\nfd42::1 49152 fd42::3 22\r\n",
            // No established socket matched: the script prints only the node id.
            "node\r\n\r\n",
        ] {
            assert!(validate_ssh(bad, "node", LOCAL, PEER).is_err(), "{bad:?}");
        }
    }
}
