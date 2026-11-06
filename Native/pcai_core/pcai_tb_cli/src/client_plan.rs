//! Build offline, source-bound OpenSSH argument plans without inspecting or executing hosts.
//!
//! See docs/usb4/native-peer-client.md. Declarations are not runtime admission.
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::PathBuf;

/// Maximum explicitly supplied policy length, including UTF-8 encoding.
pub const MAX_POLICY_BYTES: usize = 65_536;

/// Client-side path and null-configuration conventions.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ClientPlatform {
    Windows,
    Posix,
}

/// OpenSSH program whose argument grammar is planned.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ClientProtocol {
    Ssh,
    Sftp,
}

/// Caller-declared source identity; no field is discovered or authenticated.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InterfaceDeclaration {
    pub id: String,
    pub mac: String,
    pub index: u32,
}

/// Explicit client inputs; omitted or duplicate required fields fail parsing.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClientPolicy {
    pub schema: u32,
    pub client_platform: ClientPlatform,
    pub protocol: ClientProtocol,
    pub client_executable: String,
    pub account: String,
    pub host_key_alias: String,
    pub identity_file: String,
    pub known_hosts_file: String,
    pub local_interface: InterfaceDeclaration,
    pub source_address: String,
    pub peer_address: String,
    #[serde(default)]
    pub sftp_ssh_executable: Option<String>,
    #[serde(default)]
    pub sftp_batch_file: Option<String>,
}

/// A declaration-only argv array, never a shell command or readiness verdict.
#[derive(Clone, Debug, Serialize)]
pub struct ClientPlan {
    pub schema: &'static str,
    pub state: &'static str,
    pub client_platform: ClientPlatform,
    pub protocol: ClientProtocol,
    pub executable: String,
    pub arguments: Vec<String>,
    pub source_address: String,
    pub peer_address: String,
    pub declared_interface: InterfaceDeclaration,
    pub local_inspection: &'static str,
    pub peer_authentication: &'static str,
    pub executable_identity: &'static str,
    pub ssh_helper_identity: &'static str,
    pub execution_supported: bool,
    pub inspection_supported: bool,
}

/// Complete CLI actions: help or offline plan, without an executor placeholder.
#[derive(Debug, Eq, PartialEq)]
pub enum ClientCommand {
    Help,
    Plan { policy_path: PathBuf },
}

fn text(value: &str, maximum: usize) -> Result<()> {
    ensure!(
        !value.is_empty() && value.len() <= maximum && !value.chars().any(char::is_control),
        "Input is empty, over budget or contains control characters"
    );
    Ok(())
}

fn safe_name(value: &str, maximum: usize, account: bool) -> Result<()> {
    text(value, maximum)?;
    let first = value.as_bytes()[0];
    ensure!(
        first.is_ascii_alphanumeric() || (account && first == b'_'),
        "Name begins with an unsafe character"
    );
    ensure!(
        value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte)),
        "Name contains an unsafe character"
    );
    Ok(())
}

fn reserved_windows_component(component: &str) -> bool {
    let stem = component.split('.').next().unwrap_or_default().to_ascii_uppercase();
    matches!(stem.as_str(), "$NULL" | "AUX" | "CON" | "NUL" | "PRN")
        || stem
            .strip_prefix("COM")
            .or_else(|| stem.strip_prefix("LPT"))
            .is_some_and(|suffix| {
                matches!(
                    suffix,
                    "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
                )
            })
}

/// Validate a literal client path without opening it or expanding environment variables.
///
/// # Errors
/// Refuses relative/device paths, unsafe components and OpenSSH environment substitutions.
pub fn validate_client_path(value: &str, platform: ClientPlatform) -> Result<()> {
    text(value, 4096)?;
    ensure!(
        !value.as_bytes().windows(2).any(|pair| pair == *b"${") && !value.contains(['*', '?', '\0']),
        "Paths must be literal without environment or wildcard substitutions"
    );
    match platform {
        ClientPlatform::Windows => {
            let bytes = value.as_bytes();
            ensure!(
                bytes.len() >= 4
                    && bytes[0].is_ascii_alphabetic()
                    && bytes[1] == b':'
                    && matches!(bytes[2], b'\\' | b'/'),
                "Windows paths require an explicit local drive"
            );
            ensure!(
                !value[2..].contains([':', '<', '>', '"', '|']),
                "Windows path contains an unsafe character"
            );
            for part in value[3..].split(['\\', '/']) {
                ensure!(
                    !part.is_empty()
                        && !matches!(part, "." | "..")
                        && !part.ends_with(['.', ' '])
                        && !reserved_windows_component(part),
                    "Windows path has an unsafe component"
                );
            }
        }
        ClientPlatform::Posix => {
            ensure!(
                value.starts_with('/') && !value.starts_with("//"),
                "POSIX paths must be absolute"
            );
            for part in value[1..].split('/') {
                ensure!(
                    !part.is_empty() && !matches!(part, "." | ".."),
                    "POSIX path has an unsafe component"
                );
            }
        }
    }
    Ok(())
}

fn interface(declaration: &InterfaceDeclaration, platform: ClientPlatform) -> Result<InterfaceDeclaration> {
    ensure!(declaration.index > 0, "Declared interface index must be positive");
    let id = match platform {
        ClientPlatform::Windows => {
            let value = declaration
                .id
                .strip_prefix('{')
                .and_then(|value| value.strip_suffix('}'))
                .unwrap_or(&declaration.id);
            ensure!(
                value.len() == 36
                    && value.bytes().enumerate().all(|(index, byte)| {
                        if [8, 13, 18, 23].contains(&index) {
                            byte == b'-'
                        } else {
                            byte.is_ascii_hexdigit()
                        }
                    })
                    && value.bytes().any(|byte| byte.is_ascii_hexdigit() && byte != b'0'),
                "Invalid declared Windows interface GUID"
            );
            value.to_ascii_lowercase()
        }
        ClientPlatform::Posix => {
            safe_name(&declaration.id, 15, false)?;
            declaration.id.clone()
        }
    };
    let parts: Vec<_> = declaration.mac.split([':', '-']).collect();
    ensure!(
        parts.len() == 6
            && parts
                .iter()
                .all(|part| part.len() == 2 && part.bytes().all(|byte| byte.is_ascii_hexdigit())),
        "Invalid declared MAC"
    );
    ensure!(
        !(declaration.mac.contains(':') && declaration.mac.contains('-')),
        "Mixed MAC separators are refused"
    );
    let bytes: Vec<_> = parts
        .iter()
        .map(|part| u8::from_str_radix(part, 16))
        .collect::<std::result::Result<_, _>>()?;
    ensure!(
        bytes[0] & 1 == 0 && bytes.iter().any(|byte| *byte != 0),
        "MAC must be nonzero unicast"
    );
    Ok(InterfaceDeclaration {
        id,
        mac: bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<Vec<_>>()
            .join(":"),
        index: declaration.index,
    })
}

fn private_v4(address: Ipv4Addr) -> bool {
    address.is_private() || address.is_link_local()
}

fn private_v6(address: Ipv6Addr) -> bool {
    address.is_unique_local() || address.is_unicast_link_local()
}

fn address(value: &str, index: u32) -> Result<IpAddr> {
    text(value, 80)?;
    let (literal, zone) = value
        .split_once('%')
        .map_or((value, None), |(literal, zone)| (literal, Some(zone)));
    let parsed: IpAddr = literal.parse()?;
    ensure!(
        !parsed.is_unspecified() && !parsed.is_loopback() && !parsed.is_multicast(),
        "Invalid unicast address"
    );
    match parsed {
        IpAddr::V4(ip) => ensure!(
            zone.is_none() && private_v4(ip),
            "Only private/link-local IPv4 literals are admitted"
        ),
        IpAddr::V6(ip) => {
            ensure!(
                private_v6(ip) && ip.to_ipv4_mapped().is_none(),
                "Only ULA/link-local IPv6 literals are admitted"
            );
            if let Some(zone) = zone {
                ensure!(
                    ip.is_unicast_link_local()
                        && zone.bytes().all(|byte| byte.is_ascii_digit())
                        && zone.parse::<u32>()? == index,
                    "IPv6 zone must equal the declared local interface"
                );
            }
        }
    }
    Ok(parsed)
}

fn scoped(address: IpAddr, index: u32) -> String {
    match address {
        IpAddr::V6(ip) if ip.is_unicast_link_local() => format!("{ip}%{index}"),
        _ => address.to_string(),
    }
}

fn percent_literal(value: &str) -> String {
    value.replace('%', "%%")
}

fn quoted_option_path(value: &str) -> String {
    percent_literal(value).replace('\\', "\\\\").replace('"', "\\\"")
}

/// Parse a bounded policy while rejecting unknown, duplicate or mistyped fields.
///
/// # Errors
/// Returns an error for oversized UTF-8 JSON or schema/shape violations.
pub fn parse_policy(bytes: &[u8]) -> Result<ClientPolicy> {
    ensure!(bytes.len() <= MAX_POLICY_BYTES, "Policy exceeds the byte budget");
    let policy: ClientPolicy = serde_json::from_slice(bytes)?;
    ensure!(policy.schema == 1, "Unsupported client policy schema");
    Ok(policy)
}

/// Build a source-bound SSH/SFTP plan using declarations only.
///
/// # Errors
/// Refuses mismatched families, unsafe identities/paths and incompatible batch options.
/// No file, NIC, route, process, network or credential provider is consulted.
pub fn build_plan(policy: &ClientPolicy) -> Result<ClientPlan> {
    ensure!(policy.schema == 1, "Unsupported client policy schema");
    safe_name(&policy.account, 64, true)?;
    safe_name(&policy.host_key_alias, 253, false)?;
    let declared_interface = interface(&policy.local_interface, policy.client_platform)?;
    for path in [
        &policy.client_executable,
        &policy.identity_file,
        &policy.known_hosts_file,
    ] {
        validate_client_path(path, policy.client_platform)?;
    }
    let leaf = policy.client_executable.rsplit(['\\', '/']).next().unwrap_or_default();
    let expected = match (policy.protocol, policy.client_platform) {
        (ClientProtocol::Ssh, ClientPlatform::Windows) => "ssh.exe",
        (ClientProtocol::Sftp, ClientPlatform::Windows) => "sftp.exe",
        (ClientProtocol::Ssh, ClientPlatform::Posix) => "ssh",
        (ClientProtocol::Sftp, ClientPlatform::Posix) => "sftp",
    };
    ensure!(
        leaf.eq_ignore_ascii_case(expected),
        "Executable leaf does not match the declared transport"
    );
    let source = address(&policy.source_address, declared_interface.index)?;
    let peer = address(&policy.peer_address, declared_interface.index)?;
    ensure!(
        source.is_ipv4() == peer.is_ipv4() && source != peer,
        "Addresses must be distinct and use the same family"
    );
    let source = scoped(source, declared_interface.index);
    let peer = scoped(peer, declared_interface.index);
    let null_path = match policy.client_platform {
        ClientPlatform::Windows => "NUL",
        ClientPlatform::Posix => "/dev/null",
    };
    let mut arguments = vec![
        if source.contains(':') { "-6" } else { "-4" }.to_owned(),
        "-F".into(),
        null_path.into(),
    ];
    for option in [
        "BatchMode=yes",
        "IdentitiesOnly=yes",
        "IdentityAgent=none",
        "StrictHostKeyChecking=yes",
        "ProxyCommand=none",
        "ProxyJump=none",
        "ControlMaster=no",
        "ControlPath=none",
        "PermitLocalCommand=no",
        "ClearAllForwardings=yes",
        "ForwardAgent=no",
        "ForwardX11=no",
        "KnownHostsCommand=none",
        "UpdateHostKeys=no",
        "CanonicalizeHostname=no",
        "ConnectTimeout=3",
        "ConnectionAttempts=1",
        "Port=22",
    ] {
        arguments.extend(["-o".into(), option.into()]);
    }
    arguments.extend([
        "-o".into(),
        format!("HostKeyAlias={}", policy.host_key_alias),
        "-o".into(),
        format!(
            "UserKnownHostsFile=\"{}\"",
            quoted_option_path(&policy.known_hosts_file)
        ),
        "-o".into(),
        format!("GlobalKnownHostsFile={null_path}"),
        "-o".into(),
        format!("IdentityFile=\"{}\"", quoted_option_path(&policy.identity_file)),
    ]);
    match policy.protocol {
        ClientProtocol::Ssh => {
            ensure!(
                policy.sftp_batch_file.is_none(),
                "SFTP batch input is not an SSH source binding"
            );
            ensure!(
                policy.sftp_ssh_executable.is_none(),
                "SFTP helper input is not used by SSH"
            );
            arguments.extend(["-b".into(), source.clone(), format!("{}@{peer}", policy.account)]);
        }
        ClientProtocol::Sftp => {
            let helper = policy
                .sftp_ssh_executable
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("SFTP requires an explicit SSH helper executable"))?;
            validate_client_path(helper, policy.client_platform)?;
            let expected_helper = if policy.client_platform == ClientPlatform::Windows {
                "ssh.exe"
            } else {
                "ssh"
            };
            ensure!(
                helper
                    .rsplit(['\\', '/'])
                    .next()
                    .unwrap_or_default()
                    .eq_ignore_ascii_case(expected_helper),
                "SFTP helper must be the declared SSH executable"
            );
            arguments.extend(["-S".into(), helper.to_owned()]);
            arguments.extend(["-o".into(), format!("BindAddress={source}")]);
            if let Some(batch) = &policy.sftp_batch_file {
                validate_client_path(batch, policy.client_platform)?;
                arguments.extend(["-b".into(), batch.clone()]);
            }
            let destination = if peer.contains(':') {
                format!("[{peer}]")
            } else {
                peer.clone()
            };
            arguments.push(format!("{}@{destination}", policy.account));
        }
    }
    Ok(ClientPlan {
        schema: "pcai.usb4.client-plan/v1",
        state: "OfflinePlan",
        client_platform: policy.client_platform,
        protocol: policy.protocol,
        executable: policy.client_executable.clone(),
        arguments,
        source_address: source,
        peer_address: peer,
        declared_interface,
        local_inspection: "NotCollected",
        peer_authentication: "NotCollected",
        executable_identity: "NotVerified",
        ssh_helper_identity: if policy.protocol == ClientProtocol::Sftp {
            "NotVerified"
        } else {
            "NotApplicable"
        },
        execution_supported: false,
        inspection_supported: false,
    })
}

/// Parse the small CLI without opening its policy or inspecting the host.
///
/// # Errors
/// Rejects duplicate, missing or unknown options and all inspection/execution commands.
pub fn parse_cli_args(arguments: &[String]) -> Result<ClientCommand> {
    ensure!(
        arguments.len() <= 16 && arguments.iter().all(|argument| argument.len() <= 4096),
        "Argument budget exceeded"
    );
    if arguments.is_empty()
        || arguments
            .iter()
            .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        return Ok(ClientCommand::Help);
    }
    ensure!(
        arguments[0] == "plan",
        "Only offline plan is supported; inspect-local and execute are unavailable"
    );
    let mut policy_path = None;
    let mut index = 1;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--policy" => {
                ensure!(policy_path.is_none(), "Duplicate policy option");
                index += 1;
                let value = arguments
                    .get(index)
                    .ok_or_else(|| anyhow::anyhow!("Missing policy path"))?;
                text(value, 4096)?;
                ensure!(!value.starts_with('-'), "Missing policy path");
                policy_path = Some(PathBuf::from(value));
            }
            "--dry-run" | "--what-if" => {}
            _ => bail!("Unknown option; only policy, dry-run and what-if are supported"),
        }
        index += 1;
    }
    Ok(ClientCommand::Plan {
        policy_path: policy_path.ok_or_else(|| anyhow::anyhow!("Explicit policy path is required"))?,
    })
}
