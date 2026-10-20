//! Validate enrolled identity and derive symmetric, collision-checked USB4 edge addresses.
use anyhow::{anyhow, bail, ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::net::Ipv6Addr;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgePolicy {
    pub schema: u32,
    pub epoch: u64,
    pub node_id: String,
    pub node_name: String,
    pub ula_prefix: String,
    pub backend: String,
    pub local_controller_uuids: Vec<String>,
    #[serde(default)]
    pub namespaces: Vec<String>,
    pub ssh_user: Option<String>,
    pub peers: Vec<EnrolledPeer>,
    pub protected_interfaces: Vec<String>,
    pub adopt_profile_uuids: Vec<String>,
}

/// Peer operating system, which decides how qualification reads remote evidence.
///
/// WHY: a Windows OpenSSH peer defaults to PowerShell, so the POSIX evidence
/// command used for Linux peers cannot run there. Unknown values are rejected
/// at parse time rather than silently falling back to the POSIX command.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PeerPlatform {
    #[default]
    Linux,
    Windows,
}

impl PeerPlatform {
    // Skipped when serializing so policies without the field round-trip unchanged.
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnrolledPeer {
    pub node_id: String,
    pub node_name: String,
    pub controller_uuids: Vec<String>,
    pub ssh_alias: Option<String>,
    pub mtu: u32,
    #[serde(default, skip_serializing_if = "PeerPlatform::is_default")]
    pub platform: PeerPlatform,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivedEdge {
    pub peer_node_id: String,
    pub peer_name: String,
    pub local_controller_uuid: String,
    pub remote_controller_uuid: String,
    pub edge_id: String,
    pub local_ula: String,
    pub peer_ula: String,
    pub prefix_len: u8,
    pub mtu: u32,
    pub ssh_alias: Option<String>,
}

fn uuid(value: &str) -> Result<String> {
    let bytes = value.as_bytes();
    ensure!(bytes.len() == 36, "UUID must use canonical 8-4-4-4-12 syntax");
    for (i, byte) in bytes.iter().enumerate() {
        ensure!(
            if matches!(i, 8 | 13 | 18 | 23) {
                *byte == b'-'
            } else {
                byte.is_ascii_hexdigit()
            },
            "Invalid UUID syntax"
        );
    }
    ensure!(
        bytes.iter().any(|b| b.is_ascii_hexdigit() && *b != b'0'),
        "Nil UUID cannot identify an enrolled object"
    );
    Ok(value.to_ascii_lowercase())
}

fn pool(value: &str) -> Result<Ipv6Addr> {
    let (address, bits) = value
        .split_once('/')
        .ok_or_else(|| anyhow!("ULA pool requires an explicit /48"))?;
    ensure!(bits == "48", "ULA pool must be /48");
    let ip: Ipv6Addr = address.parse()?;
    ensure!(ip.octets()[0] == 0xfd, "ULA pool must use locally assigned fd00::/8");
    ensure!(
        ip.octets()[6..].iter().all(|b| *b == 0),
        "ULA /48 must have zero host/subnet bits"
    );
    Ok(ip)
}

fn safe_alias(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// Validate enrolled policy without discovering or enrolling any host.
pub fn validate_policy(policy: &EdgePolicy) -> Result<()> {
    ensure!(policy.schema == 1, "Unsupported edge policy schema");
    ensure!(policy.epoch > 0, "Policy epoch must be positive");
    let local = uuid(&policy.node_id)?;
    ensure!(!policy.node_name.trim().is_empty(), "Local node name is required");
    ensure!(
        matches!(policy.backend.as_str(), "networkd" | "networkmanager"),
        "Unsupported network backend"
    );
    if let Some(user) = &policy.ssh_user {
        let mut chars = user.chars();
        ensure!(
            matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
                && user.len() <= 32
                && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "Invalid POSIX SSH user"
        );
    }
    for namespace in &policy.namespaces {
        ensure!(safe_alias(namespace), "Invalid approved namespace name");
    }
    pool(&policy.ula_prefix)?;
    let mut nodes = HashSet::from([local]);
    let mut controllers = HashSet::new();
    ensure!(
        !policy.local_controller_uuids.is_empty(),
        "Local controller enrollment is required"
    );
    for controller in &policy.local_controller_uuids {
        ensure!(controllers.insert(uuid(controller)?), "Duplicate local controller UUID");
    }
    for peer in &policy.peers {
        ensure!(nodes.insert(uuid(&peer.node_id)?), "Duplicate or self peer node ID");
        ensure!(!peer.node_name.trim().is_empty(), "Peer node name is required");
        ensure!(
            (1280..=65535).contains(&peer.mtu),
            "Peer MTU must be between 1280 and 65535"
        );
        ensure!(
            !peer.controller_uuids.is_empty(),
            "Enrolled peer needs a controller UUID"
        );
        if let Some(alias) = &peer.ssh_alias {
            ensure!(safe_alias(alias), "Unsafe SSH alias");
        }
        for controller in &peer.controller_uuids {
            ensure!(
                controllers.insert(uuid(controller)?),
                "Duplicate enrolled controller UUID"
            );
        }
    }
    let mut adopted = HashSet::new();
    for profile in &policy.adopt_profile_uuids {
        ensure!(adopted.insert(uuid(profile)?), "Duplicate adopted profile UUID");
    }
    Ok(())
}

fn pair_hash(local: &str, remote: &str) -> [u8; 32] {
    let (a, b) = if local < remote {
        (local, remote)
    } else {
        (remote, local)
    };
    let mut hash = Sha256::new();
    hash.update(b"pcai.usb4.edge/v1\0");
    hash.update(a.as_bytes());
    hash.update(b"\0");
    hash.update(b.as_bytes());
    hash.finalize().into()
}

fn endpoint(node_id: &str) -> [u8; 8] {
    let mut hash = Sha256::new();
    hash.update(b"pcai.usb4.endpoint/v1\0");
    hash.update(node_id.as_bytes());
    let bytes: [u8; 32] = hash.finalize().into();
    let mut iid = [0; 8];
    iid.copy_from_slice(&bytes[..8]);
    iid
}

/// Derive one enrolled edge; use `derive_edge_plan` to reject allocation collisions.
pub fn derive_edge(policy: &EdgePolicy, local_controller_uuid: &str, remote_uuid: &str) -> Result<DerivedEdge> {
    validate_policy(policy)?;
    let local_controller = uuid(local_controller_uuid)?;
    let remote_controller = uuid(remote_uuid)?;
    ensure!(
        local_controller != remote_controller,
        "An edge requires distinct controllers"
    );
    ensure!(
        !policy
            .peers
            .iter()
            .flat_map(|p| &p.controller_uuids)
            .any(|c| c.eq_ignore_ascii_case(&local_controller)),
        "Local controller is enrolled to a remote peer"
    );
    ensure!(
        policy
            .local_controller_uuids
            .iter()
            .any(|c| c.eq_ignore_ascii_case(&local_controller)),
        "Local controller is not enrolled"
    );
    let peer = policy
        .peers
        .iter()
        .find(|p| {
            p.controller_uuids
                .iter()
                .any(|c| c.eq_ignore_ascii_case(&remote_controller))
        })
        .ok_or_else(|| anyhow!("Remote controller is not enrolled"))?;
    let local_node = uuid(&policy.node_id)?;
    let remote_node = uuid(&peer.node_id)?;
    let local_iid = endpoint(&local_node);
    let remote_iid = endpoint(&remote_node);
    ensure!(local_iid != remote_iid, "Endpoint address hash collision");
    ensure!(
        local_iid != [0; 8] && remote_iid != [0; 8],
        "Endpoint hash produced reserved subnet-router anycast IID"
    );
    let hash = pair_hash(&local_controller, &remote_controller);
    let mut address = pool(&policy.ula_prefix)?.octets();
    address[6..8].copy_from_slice(&hash[..2]);
    address[8..].copy_from_slice(&local_iid);
    let local_ula = Ipv6Addr::from(address).to_string();
    address[8..].copy_from_slice(&remote_iid);
    Ok(DerivedEdge {
        peer_node_id: remote_node,
        peer_name: peer.node_name.clone(),
        local_controller_uuid: local_controller,
        remote_controller_uuid: remote_controller,
        edge_id: format!("usb4-{}", hex::encode(hash)),
        local_ula,
        peer_ula: Ipv6Addr::from(address).to_string(),
        prefix_len: 64,
        mtu: peer.mtu,
        ssh_alias: peer.ssh_alias.clone(),
    })
}

/// Derive all enrolled candidates and reject duplicate local identities or /64 collisions.
pub fn derive_edge_plan(policy: &EdgePolicy, local_controller_uuids: &[String]) -> Result<Vec<DerivedEdge>> {
    validate_policy(policy)?;
    let mut local_ids = HashSet::new();
    let mut subnets: HashMap<[u8; 8], String> = HashMap::new();
    let mut edges = Vec::new();
    for local in local_controller_uuids {
        let local = uuid(local)?;
        ensure!(
            policy
                .local_controller_uuids
                .iter()
                .any(|c| c.eq_ignore_ascii_case(&local)),
            "Requested local controller is not enrolled"
        );
        ensure!(local_ids.insert(local), "Duplicate local controller UUID");
    }
    for local in &policy.local_controller_uuids {
        let local = uuid(local)?;
        for remote in policy.peers.iter().flat_map(|p| &p.controller_uuids) {
            let edge = derive_edge(policy, &local, remote)?;
            let bytes = edge.local_ula.parse::<Ipv6Addr>()?.octets();
            let mut subnet = [0; 8];
            subnet.copy_from_slice(&bytes[..8]);
            if let Some(previous) = subnets.insert(subnet, edge.edge_id.clone()) {
                bail!(
                    "Enrolled /64 allocation collision between {previous} and {}",
                    edge.edge_id
                );
            }
            edges.push(edge);
        }
    }
    edges.retain(|edge| local_ids.contains(&edge.local_controller_uuid));
    edges.sort_by(|a, b| a.edge_id.cmp(&b.edge_id));
    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn id(n: u32) -> String {
        format!("00000000-0000-4000-8000-{n:012x}")
    }
    fn policy() -> EdgePolicy {
        EdgePolicy {
            schema: 1,
            epoch: 1,
            node_id: id(1),
            node_name: "local".into(),
            ula_prefix: "fd42:1234:5678::/48".into(),
            backend: "networkmanager".into(),
            local_controller_uuids: vec![id(10), id(11)],
            namespaces: vec![],
            ssh_user: None,
            peers: vec![EnrolledPeer {
                node_id: id(2),
                node_name: "peer".into(),
                controller_uuids: vec![id(20)],
                ssh_alias: Some("peer-pinned".into()),
                mtu: 1500,
                platform: PeerPlatform::Linux,
            }],
            protected_interfaces: vec![],
            adopt_profile_uuids: vec![],
        }
    }

    // Protects interoperable allocation; detects endpoint-order-dependent subnet/host derivation.
    // Needs no hardware; breadcrumb: derive_edge/pair_hash/endpoint.
    #[test]
    fn reversed_endpoints_derive_identical_edge_and_swapped_addresses() {
        let a = policy();
        let mut b = a.clone();
        b.local_controller_uuids = vec![id(20)];
        b.node_id = id(2);
        b.peers[0].node_id = id(1);
        b.peers[0].controller_uuids = vec![id(10)];
        let ab = derive_edge(&a, &id(10), &id(20)).unwrap();
        let ba = derive_edge(&b, &id(20), &id(10)).unwrap();
        assert_eq!(ab.edge_id, ba.edge_id);
        assert_eq!(ab.local_ula, ba.peer_ula);
        assert_eq!(ab.peer_ula, ba.local_ula);
        assert_eq!(ab.prefix_len, 64);
    }

    // Protects parallel edges; detects using peer/node identity alone for the /64.
    // Needs no hardware; breadcrumb: derive_edge_plan.
    #[test]
    fn parallel_controller_pairs_have_distinct_edges_and_prefixes() {
        let p = policy();
        let edges = derive_edge_plan(&p, &[id(10), id(11)]).unwrap();
        assert_eq!(edges.len(), 2);
        assert_ne!(edges[0].edge_id, edges[1].edge_id);
        assert_ne!(
            edges[0].local_ula.parse::<Ipv6Addr>().unwrap().segments()[3],
            edges[1].local_ula.parse::<Ipv6Addr>().unwrap().segments()[3]
        );
    }

    // Protects explicit enrollment; detects name-based or unknown-controller auto-admission.
    // Needs no hardware; breadcrumb: derive_edge/uuid.
    #[test]
    fn unknown_remote_or_self_controller_is_rejected() {
        let p = policy();
        assert!(derive_edge(&p, &id(10), &id(99)).is_err());
        assert!(derive_edge(&p, &id(20), &id(20)).is_err());
        assert!(derive_edge(&p, "not-a-uuid", &id(20)).is_err());
    }

    // Protects allocation and policy boundaries; detects invalid prefix/epoch/schema/MTU acceptance.
    // Needs no hardware; breadcrumb: validate_policy/pool.
    #[test]
    fn invalid_policy_values_are_rejected_before_derivation() {
        for prefix in ["2001:db8::/48", "fc00::/48", "fd42::/64", "fd42::1/48", "fd42::"] {
            let mut p = policy();
            p.ula_prefix = prefix.into();
            assert!(derive_edge(&p, &id(10), &id(20)).is_err(), "{prefix}");
        }
        let mut p = policy();
        p.epoch = 0;
        assert!(validate_policy(&p).is_err());
        p.epoch = 1;
        p.schema = 2;
        assert!(validate_policy(&p).is_err());
        for mtu in [1279, 65536] {
            let mut p = policy();
            p.peers[0].mtu = mtu;
            assert!(validate_policy(&p).is_err());
        }
    }

    // Protects unique ownership; detects duplicate UUIDs including case variants and self peers.
    // Needs no hardware; breadcrumb: validate_policy/derive_edge_plan.
    #[test]
    fn duplicate_nodes_controllers_and_local_inventory_are_rejected() {
        let mut p = policy();
        p.peers.push(p.peers[0].clone());
        assert!(validate_policy(&p).is_err());
        let mut p = policy();
        let mut other = p.peers[0].clone();
        other.node_id = id(3);
        p.peers.push(other);
        assert!(validate_policy(&p).is_err());
        let mut p = policy();
        p.peers[0].node_id = p.node_id.clone();
        assert!(validate_policy(&p).is_err());
        assert!(derive_edge_plan(&policy(), &[id(10), id(10)]).is_err());
    }

    // Protects command-bound enrollment fields; detects option/metacharacter injection and unsupported backends.
    // Needs no processes; breadcrumb: validate_policy/safe_alias.
    #[test]
    fn unsafe_runtime_names_and_unenrolled_local_controllers_are_rejected() {
        for user in ["-root", "user;id", "user name", "user@host", ""] {
            let mut p = policy();
            p.ssh_user = Some(user.into());
            assert!(validate_policy(&p).is_err(), "{user}");
        }
        for alias in ["-oProxyCommand=evil", "host;id", "user@host", "../host", ""] {
            let mut p = policy();
            p.peers[0].ssh_alias = Some(alias.into());
            assert!(validate_policy(&p).is_err(), "{alias}");
        }
        let mut p = policy();
        p.backend = "shell".into();
        assert!(validate_policy(&p).is_err());
        let mut p = policy();
        p.namespaces = vec!["../foreign".into()];
        assert!(validate_policy(&p).is_err());
        assert!(derive_edge(&policy(), &id(999), &id(20)).is_err());
        assert!(derive_edge_plan(&policy(), &[id(999)]).is_err());
        for mtu in [1280, 65535] {
            let mut p = policy();
            p.peers[0].mtu = mtu;
            assert_eq!(derive_edge(&p, &id(10), &id(20)).unwrap().mtu, mtu);
        }
    }

    // Protects enrollment storage compatibility; detects implicit local enrollment or dropping approved namespace defaults.
    // Needs serde JSON only; breadcrumb: EdgePolicy serde and validate_policy.
    #[test]
    fn serialized_policy_requires_local_enrollment_and_defaults_readonly_namespaces() {
        let mut value = serde_json::to_value(policy()).unwrap();
        value.as_object_mut().unwrap().remove("namespaces");
        let decoded: EdgePolicy = serde_json::from_value(value.clone()).unwrap();
        assert!(decoded.namespaces.is_empty());
        validate_policy(&decoded).unwrap();
        value.as_object_mut().unwrap().remove("local_controller_uuids");
        assert!(serde_json::from_value::<EdgePolicy>(value).is_err());
        let mut p = policy();
        p.local_controller_uuids.push(p.local_controller_uuids[0].clone());
        assert!(validate_policy(&p).is_err());
        let mut p = policy();
        p.peers[0].controller_uuids = vec![p.local_controller_uuids[0].clone()];
        assert!(validate_policy(&p).is_err());
    }

    // Protects finite /48 allocation space; detects collision overwrite/silent shared-prefix reuse.
    // Needs deterministic bounded SHA search, no network; breadcrumb: derive_edge_plan.
    #[test]
    fn actual_hash_subnet_collision_fails_closed() {
        let local = id(10);
        let mut seen = HashMap::new();
        let mut collision = None;
        for n in 100..=65_636 {
            let remote = id(n);
            let hash = pair_hash(&local, &remote);
            let subnet = [hash[0], hash[1]];
            if let Some(previous) = seen.insert(subnet, remote.clone()) {
                collision = Some((previous, remote));
                break;
            }
        }
        let (a, b) = collision.expect("65537 candidates must collide in16bits");
        let mut p = policy();
        p.peers[0].controller_uuids = vec![a, b];
        assert!(derive_edge_plan(&p, &[local])
            .unwrap_err()
            .to_string()
            .contains("allocation collision"));
    }

    // Protects Linux-peer compatibility; detects the new field changing parsing, derived edges or rendered profiles.
    // Needs serde JSON only; breadcrumb: EnrolledPeer::platform/networkd_edge::render.
    #[test]
    fn platform_defaults_to_linux_and_leaves_linux_output_unchanged() {
        let without = serde_json::to_value(policy()).unwrap();
        assert!(
            without["peers"][0].get("platform").is_none(),
            "default is not serialized"
        );
        let mut explicit = without.clone();
        explicit["peers"][0]["platform"] = "linux".into();
        let a: EdgePolicy = serde_json::from_value(without).unwrap();
        let b: EdgePolicy = serde_json::from_value(explicit).unwrap();
        assert_eq!(a.peers[0].platform, PeerPlatform::Linux);
        assert_eq!(b.peers[0].platform, PeerPlatform::Linux);
        let ea = derive_edge_plan(&a, &[id(10), id(11)]).unwrap();
        let eb = derive_edge_plan(&b, &[id(10), id(11)]).unwrap();
        assert_eq!(ea, eb);
        assert_eq!(serde_json::to_string(&ea).unwrap(), serde_json::to_string(&eb).unwrap());
        for (x, y) in ea.iter().zip(&eb) {
            assert_eq!(
                crate::networkd_edge::render(x, a.epoch),
                crate::networkd_edge::render(y, b.epoch)
            );
        }
    }

    // Protects the Windows opt-in; detects accepting unknown platforms or dropping the windows value.
    // Needs serde JSON only; breadcrumb: PeerPlatform.
    #[test]
    fn platform_accepts_windows_and_rejects_unknown_values() {
        let mut value = serde_json::to_value(policy()).unwrap();
        value["peers"][0]["platform"] = "windows".into();
        let decoded: EdgePolicy = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(decoded.peers[0].platform, PeerPlatform::Windows);
        validate_policy(&decoded).unwrap();
        assert_eq!(serde_json::to_value(&decoded).unwrap(), value);
        for bad in ["Windows", "macos", "", "posix"] {
            value["peers"][0]["platform"] = bad.into();
            assert!(serde_json::from_value::<EdgePolicy>(value.clone()).is_err(), "{bad}");
        }
    }
}
