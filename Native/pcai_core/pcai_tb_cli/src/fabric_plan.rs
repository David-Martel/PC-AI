//! Plans observed USB4 segments without authorizing routes, sharing or bridges.
//!
//! Input provenance is descriptive, not peer authentication. See the crate's
//! topology documentation for the separate live-device qualification boundary.

use anyhow::{ensure, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::net::IpAddr;

pub const OBSERVATION_SCHEMA: &str = "pcai.usb4-fabric-observation/v1";
pub const PLAN_SCHEMA: &str = "pcai.usb4-fabric-plan/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FabricObservation {
    pub schema: String,
    pub ttl_seconds: u32,
    pub hosts: Vec<HostObservation>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObservationSource {
    #[serde(rename = "local")]
    Local,
    #[serde(rename = "management-ssh")]
    ManagementSsh,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostObservation {
    pub observed_at_utc: DateTime<Utc>,
    pub host: String,
    pub source: ObservationSource,
    pub interfaces: Vec<InterfaceObservation>,
    #[serde(default)]
    pub reference_applications: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterfaceObservation {
    pub interface_guid: String,
    pub mac: String,
    pub pnp_instance_id: String,
    #[serde(default)]
    pub root_router_id: Option<String>,
    pub interface_index: u32,
    pub interface_alias: String,
    pub up: bool,
    #[serde(default)]
    pub addresses: Vec<String>,
    #[serde(default)]
    pub neighbors: Vec<NeighborObservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeighborObservation {
    pub address: String,
    pub mac: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SegmentState {
    ObservedUnverified,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FabricPlan {
    pub schema: String,
    pub evaluated_at_utc: DateTime<Utc>,
    pub ttl_seconds: u32,
    pub segments: Vec<ObservedSegment>,
    pub blocked_endpoints: Vec<BlockedEndpoint>,
    pub warnings: Vec<String>,
    /// Differences from a prior observation only; never instructions to the OS.
    pub withdrawn_edge_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedSegment {
    pub edge_id: String,
    pub state: SegmentState,
    pub endpoints: [SegmentEndpoint; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentEndpoint {
    pub host: String,
    pub interface_guid: String,
    pub mac: String,
    pub root_router_id: String,
    pub interface_index: u32,
    /// Display metadata; neither alias nor source grants identity or trust.
    pub interface_alias: String,
    pub source: ObservationSource,
    /// Only reciprocal neighbor observations; IPv6 LLA zones use this endpoint.
    pub peer_addresses: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockedEndpoint {
    pub host: String,
    pub interface_guid: String,
    pub reasons: Vec<String>,
}

struct Candidate<'a> {
    host: &'a HostObservation,
    interface: &'a InterfaceObservation,
    guid: String,
    mac: String,
    root: String,
    addresses: BTreeSet<IpAddr>,
    neighbors: Vec<(IpAddr, String)>,
}

fn bounded_text(value: &str, maximum: usize) -> bool {
    !value.is_empty() && value.len() <= maximum && !value.chars().any(char::is_control)
}

fn normalized_guid(value: &str) -> Result<String> {
    let value = value
        .strip_prefix('{')
        .and_then(|v| v.strip_suffix('}'))
        .unwrap_or(value);
    ensure!(value.len() == 36, "Invalid interface GUID");
    ensure!(
        value.bytes().enumerate().all(|(index, byte)| {
            if [8, 13, 18, 23].contains(&index) {
                byte == b'-'
            } else {
                byte.is_ascii_hexdigit()
            }
        }) && value.bytes().any(|byte| byte.is_ascii_hexdigit() && byte != b'0'),
        "Invalid or zero interface GUID"
    );
    Ok(value.to_ascii_lowercase())
}

fn normalized_mac(value: &str) -> Result<String> {
    let compact: String = value.chars().filter(|c| *c != ':' && *c != '-').collect();
    ensure!(
        compact.len() == 12 && compact.bytes().all(|b| b.is_ascii_hexdigit()),
        "Invalid MAC"
    );
    // Reject mixed/irregular separators instead of silently accepting malformed input.
    ensure!(
        value.len() == 12
            || (value.len() == 17
                && [':', '-'].iter().any(|separator| {
                    value.split(*separator).count() == 6 && value.split(*separator).all(|part| part.len() == 2)
                })),
        "Invalid MAC separators"
    );
    let bytes = hex::decode(&compact)?;
    ensure!(
        bytes[0] & 1 == 0 && bytes.iter().any(|b| *b != 0),
        "MAC must be nonzero unicast"
    );
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(":"))
}

fn literal_ip(value: &str, local_index: u32) -> Result<IpAddr> {
    ensure!(bounded_text(value, 64), "Invalid address text");
    let (literal, zone) = value
        .split_once('%')
        .map_or((value, None), |(ip, zone)| (ip, Some(zone)));
    let address: IpAddr = literal.parse()?;
    ensure!(
        !address.is_unspecified() && !address.is_loopback() && !address.is_multicast(),
        "Invalid unicast address"
    );
    match address {
        IpAddr::V4(v4) => ensure!(!v4.is_broadcast() && v4.octets()[0] != 0, "Invalid IPv4 address"),
        IpAddr::V6(v6) => ensure!(
            v6.to_ipv4_mapped().is_none(),
            "Mapped IPv4 is not an interface IPv6 address"
        ),
    }
    if let Some(zone) = zone {
        ensure!(
            matches!(address, IpAddr::V6(ip) if ip.is_unicast_link_local()),
            "Zone requires IPv6 LLA"
        );
        ensure!(
            zone.parse::<u32>()? == local_index && local_index > 0,
            "LLA zone differs from local interface"
        );
    }
    Ok(address)
}

fn scoped(address: IpAddr, local_index: u32) -> String {
    match address {
        IpAddr::V6(ip) if ip.is_unicast_link_local() => format!("{ip}%{local_index}"),
        _ => address.to_string(),
    }
}

fn usb4_identity(interface: &InterfaceObservation) -> Option<String> {
    let pnp = interface.pnp_instance_id.to_ascii_uppercase();
    let parts: Vec<_> = pnp.split('\\').collect();
    if parts.len() != 3
        || parts[0] != "USB4"
        || parts[2].is_empty()
        || !parts[1].split('&').any(|part| part == "PROT_USB4NET")
    {
        return None;
    }
    let root = interface.root_router_id.as_ref()?.to_ascii_uppercase();
    let (kind, instance) = root.strip_prefix("USB4\\")?.split_once('\\')?;
    if !(kind == "ROOT_DEVICE_ROUTER" || kind.starts_with("ROOT_DEVICE_ROUTER&"))
        || instance.is_empty()
        || instance.contains('\\')
    {
        return None;
    }
    Some(root)
}

fn endpoint(candidate: &Candidate<'_>, peer: &Candidate<'_>) -> SegmentEndpoint {
    let peer_addresses = candidate
        .neighbors
        .iter()
        .filter(|(ip, mac)| mac == &peer.mac && peer.addresses.contains(ip))
        .map(|(ip, _)| scoped(*ip, candidate.interface.interface_index))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    SegmentEndpoint {
        host: candidate.host.host.to_ascii_lowercase(),
        interface_guid: candidate.guid.clone(),
        mac: candidate.mac.clone(),
        root_router_id: candidate.root.clone(),
        interface_index: candidate.interface.interface_index,
        interface_alias: candidate.interface.interface_alias.clone(),
        source: candidate.host.source,
        peer_addresses,
    }
}

fn identity(endpoint: &SegmentEndpoint) -> (&str, &str, &str, &str) {
    (
        &endpoint.host,
        &endpoint.interface_guid,
        &endpoint.mac,
        &endpoint.root_router_id,
    )
}

/// Builds fresh point-to-point observations; it never configures a fabric.
///
/// # Errors
/// Rejects unsupported schemas, bounds, malformed identity/address values and
/// duplicate host/GUID/MAC/index identities. Missing roots, stale observations,
/// down interfaces and ambiguous or absent reciprocity produce blocked endpoints.
pub fn build_plan(
    observation: &FabricObservation,
    evaluated_at_utc: DateTime<Utc>,
    previous: Option<&FabricPlan>,
) -> Result<FabricPlan> {
    ensure!(
        observation.schema == OBSERVATION_SCHEMA,
        "Unsupported observation schema"
    );
    ensure!(
        (30..=300).contains(&observation.ttl_seconds),
        "TTL must be 30..300 seconds"
    );
    ensure!(
        !observation.hosts.is_empty() && observation.hosts.len() <= 16,
        "Between 1 and 16 hosts are required"
    );
    ensure!(
        observation
            .hosts
            .iter()
            .map(|host| host.interfaces.len())
            .sum::<usize>()
            <= 64,
        "At most 64 interfaces are supported"
    );
    let mut plan = FabricPlan { schema: PLAN_SCHEMA.into(), evaluated_at_utc,
        ttl_seconds: observation.ttl_seconds, segments: vec![], blocked_endpoints: vec![],
        warnings: vec!["Observed segments do not establish authentication, readiness, sharing, throughput or bridge/transit authority.".into()], withdrawn_edge_ids: vec![] };
    let mut hosts = BTreeSet::new();
    let mut candidates = vec![];
    for host in &observation.hosts {
        ensure!(
            bounded_text(&host.host, 63)
                && host.host.as_bytes()[0].is_ascii_alphanumeric()
                && host.host.as_bytes()[host.host.len() - 1].is_ascii_alphanumeric()
                && host
                    .host
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c)),
            "Invalid host identifier"
        );
        ensure!(
            hosts.insert(host.host.to_ascii_lowercase()),
            "Duplicate host identifier"
        );
        ensure!(
            host.reference_applications.len() <= 16
                && host.reference_applications.iter().all(|app| bounded_text(app, 128)),
            "Invalid reference application metadata"
        );
        let mut guids = BTreeSet::new();
        let mut macs = BTreeSet::new();
        let mut indices = BTreeSet::new();
        for interface in &host.interfaces {
            let guid = normalized_guid(&interface.interface_guid)?;
            let mac = normalized_mac(&interface.mac)?;
            ensure!(
                guids.insert(guid.clone()) && macs.insert(mac.clone()) && indices.insert(interface.interface_index),
                "Duplicate interface GUID, MAC or index on host"
            );
            ensure!(
                interface.interface_alias.len() <= 256
                    && !interface.interface_alias.chars().any(char::is_control)
                    && interface.pnp_instance_id.len() <= 512
                    && !interface.pnp_instance_id.chars().any(char::is_control)
                    && interface
                        .root_router_id
                        .as_ref()
                        .is_none_or(|root| bounded_text(root, 512)),
                "Invalid interface metadata"
            );
            ensure!(
                interface.addresses.len() <= 128 && interface.neighbors.len() <= 256,
                "Interface address/neighbor bound exceeded"
            );
            let addresses = interface
                .addresses
                .iter()
                .map(|ip| literal_ip(ip, interface.interface_index))
                .collect::<Result<BTreeSet<_>>>()?;
            let neighbors = interface
                .neighbors
                .iter()
                .map(|neighbor| {
                    Ok((
                        literal_ip(&neighbor.address, interface.interface_index)?,
                        normalized_mac(&neighbor.mac)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut reasons = vec![];
            let age = evaluated_at_utc.signed_duration_since(host.observed_at_utc);
            if age < chrono::Duration::zero() {
                reasons.push("Observation is from the future".into());
            }
            if age >= chrono::Duration::seconds(i64::from(observation.ttl_seconds)) {
                reasons.push("Observation expired".into());
            }
            if !interface.up {
                reasons.push("Interface is down".into());
            }
            if interface.interface_index == 0 {
                reasons.push("Interface index is missing".into());
            }
            if addresses.is_empty() {
                reasons.push("No unicast interface address observed".into());
            }
            let root = usb4_identity(interface);
            if root.is_none() {
                reasons.push("USB4NET PnP/root-router identity is missing or invalid".into());
            }
            if !reasons.is_empty() {
                plan.blocked_endpoints.push(BlockedEndpoint {
                    host: host.host.to_ascii_lowercase(),
                    interface_guid: guid,
                    reasons,
                });
            } else {
                candidates.push(Candidate {
                    host,
                    interface,
                    guid,
                    mac,
                    root: root.unwrap_or_default(),
                    addresses,
                    neighbors,
                });
            }
        }
    }
    let matches: Vec<BTreeSet<usize>> = candidates
        .iter()
        .map(|candidate| {
            candidates
                .iter()
                .enumerate()
                .filter(|(_, remote)| {
                    !candidate.host.host.eq_ignore_ascii_case(&remote.host.host)
                        && candidate
                            .neighbors
                            .iter()
                            .any(|(ip, mac)| mac == &remote.mac && remote.addresses.contains(ip))
                })
                .map(|(index, _)| index)
                .collect()
        })
        .collect();
    for (index, candidate) in candidates.iter().enumerate() {
        let remote_index = matches[index].iter().next().copied();
        let reciprocal = remote_index.filter(|remote| {
            matches[index].len() == 1 && matches[*remote].len() == 1 && matches[*remote].contains(&index)
        });
        if let Some(remote) = reciprocal {
            if index < remote {
                let mut endpoints = [
                    endpoint(candidate, &candidates[remote]),
                    endpoint(&candidates[remote], candidate),
                ];
                endpoints.sort_by(|a, b| identity(a).cmp(&identity(b)));
                let key = serde_json::to_vec(&[identity(&endpoints[0]), identity(&endpoints[1])])?;
                plan.segments.push(ObservedSegment {
                    edge_id: hex::encode(Sha256::digest(key)),
                    state: SegmentState::ObservedUnverified,
                    endpoints,
                });
            }
        } else {
            plan.blocked_endpoints.push(BlockedEndpoint {
                host: candidate.host.host.to_ascii_lowercase(),
                interface_guid: candidate.guid.clone(),
                reasons: vec!["No unique reciprocal neighbor MAC and actual remote interface address".into()],
            });
        }
    }
    plan.segments.sort_by(|a, b| a.edge_id.cmp(&b.edge_id));
    let apipa_segments = plan
        .segments
        .iter()
        .filter(|segment| {
            segment.endpoints.iter().any(|endpoint| {
                endpoint
                    .peer_addresses
                    .iter()
                    .any(|value| value.parse::<std::net::Ipv4Addr>().is_ok_and(|ip| ip.is_link_local()))
            })
        })
        .count();
    let parallel_local_apipa = candidates.iter().any(|candidate| {
        candidates
            .iter()
            .filter(|other| {
                other.host.host.eq_ignore_ascii_case(&candidate.host.host)
                    && other
                        .addresses
                        .iter()
                        .any(|address| matches!(address, IpAddr::V4(ip) if ip.is_link_local()))
            })
            .count()
            > 1
    });
    if apipa_segments > 1 || parallel_local_apipa {
        plan.warnings.push("APIPA /16 is observed on multiple independent segments; overlapping prefixes do not establish a shared LAN or bridge.".into());
    }
    if let Some(previous) = previous {
        ensure!(
            previous.schema == PLAN_SCHEMA && previous.segments.len() <= 32,
            "Invalid previous plan schema or edge bound"
        );
        let current: BTreeSet<_> = plan.segments.iter().map(|segment| segment.edge_id.as_str()).collect();
        for segment in &previous.segments {
            ensure!(
                segment.edge_id.len() == 64 && segment.edge_id.bytes().all(|c| c.is_ascii_hexdigit()),
                "Invalid prior edge identity"
            );
            if !current.contains(segment.edge_id.as_str()) {
                plan.withdrawn_edge_ids.push(segment.edge_id.clone());
            }
        }
        plan.withdrawn_edge_ids.sort();
        plan.withdrawn_edge_ids.dedup();
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> DateTime<Utc> {
        "2026-11-05T11:00:00Z".parse().unwrap()
    }
    fn interface(id: u32, index: u32, address: &str, peer_mac: u32, peer_address: &str) -> InterfaceObservation {
        InterfaceObservation {
            interface_guid: format!("00000000-0000-0000-0000-{id:012x}"),
            mac: format!("02:00:00:00:00:{id:02x}"),
            pnp_instance_id: format!("USB4\\VEN_1D6B&DEV_00000004&PROT_USB4NET\\INSTANCE{id}"),
            root_router_id: Some(format!("USB4\\ROOT_DEVICE_ROUTER\\ROOT{id}")),
            interface_index: index,
            interface_alias: "Thunderbolt connection".into(),
            up: true,
            addresses: vec![address.into()],
            neighbors: vec![NeighborObservation {
                address: peer_address.into(),
                mac: format!("02:00:00:00:00:{peer_mac:02x}"),
            }],
        }
    }
    fn fixture() -> FabricObservation {
        FabricObservation {
            schema: OBSERVATION_SCHEMA.into(),
            ttl_seconds: 90,
            hosts: vec![
                HostObservation {
                    observed_at_utc: now(),
                    host: "gen9".into(),
                    source: ObservationSource::Local,
                    interfaces: vec![interface(1, 17, "169.254.10.1", 2, "169.254.10.2")],
                    reference_applications: vec![],
                },
                HostObservation {
                    observed_at_utc: now(),
                    host: "carbon".into(),
                    source: ObservationSource::ManagementSsh,
                    interfaces: vec![interface(2, 23, "169.254.10.2", 1, "169.254.10.1")],
                    reference_applications: vec![],
                },
            ],
        }
    }

    // Protects: observation is not identity authentication or a licensed-app dependency.
    // Detects: requiring Share metadata or promoting MAC reciprocity to readiness.
    // Needs: pure fixture; Breadcrumb: build_plan and SegmentState.
    #[test]
    fn reciprocal_observation_without_reference_app_is_unverified() {
        let mut observation = fixture();
        let first = build_plan(&observation, now(), None).unwrap();
        assert_eq!(first.segments.len(), 1);
        assert_eq!(first.segments[0].state, SegmentState::ObservedUnverified);
        observation.hosts[0]
            .reference_applications
            .push("Thunderbolt Share".into());
        let second = build_plan(&observation, now(), None).unwrap();
        assert_eq!(first.segments[0].edge_id, second.segments[0].edge_id);
        let encoded = serde_json::to_vec(&observation).unwrap();
        assert_eq!(
            build_plan(&serde_json::from_slice(&encoded).unwrap(), now(), None)
                .unwrap()
                .segments
                .len(),
            1
        );
    }

    // Protects: display aliases never substitute for USB4 physical identifiers.
    // Detects: name-based matches and missing-root/pseudo-USB4 admission.
    // Needs: pure fixture; Breadcrumb: usb4_identity and build_plan.
    #[test]
    fn same_alias_wrong_device_or_missing_root_cannot_form_segment() {
        for bad_identity in ["PCI\\ETHERNET\\DEVICE", "USB4\\VEN_1&NOT_PROT_USB4NET\\INSTANCE"] {
            let mut observation = fixture();
            observation.hosts[0].interfaces[0].pnp_instance_id = bad_identity.into();
            assert!(build_plan(&observation, now(), None).unwrap().segments.is_empty());
        }
        let mut observation = fixture();
        observation.hosts[0].interfaces[0].root_router_id = None;
        let plan = build_plan(&observation, now(), None).unwrap();
        assert!(plan.segments.is_empty());
        assert!(plan
            .blocked_endpoints
            .iter()
            .any(|endpoint| endpoint.reasons.iter().any(|reason| reason.contains("root-router"))));
    }

    // Protects: stale/down/disappeared observations cannot retain a prior edge.
    // Detects: future acceptance, TTL off-by-one and reuse of historical segments.
    // Needs: explicit clock and private prior plan; Breadcrumb: build_plan expiry/difference.
    #[test]
    fn expired_future_down_and_missing_withdraw_only_observation_ids() {
        let observation = fixture();
        let previous = build_plan(&observation, now(), None).unwrap();
        assert_eq!(
            build_plan(&observation, now() + chrono::Duration::seconds(89), None)
                .unwrap()
                .segments
                .len(),
            1
        );
        for evaluated in [
            now() + chrono::Duration::seconds(90),
            now() + chrono::Duration::seconds(91),
            now() - chrono::Duration::nanoseconds(1),
        ] {
            let plan = build_plan(&observation, evaluated, Some(&previous)).unwrap();
            assert!(plan.segments.is_empty());
            assert_eq!(plan.withdrawn_edge_ids, vec![previous.segments[0].edge_id.clone()]);
        }
        let mut down = observation.clone();
        down.hosts[0].interfaces[0].up = false;
        assert!(build_plan(&down, now(), Some(&previous)).unwrap().segments.is_empty());
        let mut missing = observation;
        missing.hosts[0].interfaces.clear();
        assert_eq!(
            build_plan(&missing, now(), Some(&previous))
                .unwrap()
                .withdrawn_edge_ids
                .len(),
            1
        );
    }

    // Protects: aliases/indices are transient while GUID/MAC/root changes are new devices.
    // Detects: unstable renaming IDs and hotplug carrying an old device identity forward.
    // Needs: pure fixture; Breadcrumb: identity, scoped and build_plan hashing.
    #[test]
    fn rename_and_index_change_preserve_id_but_device_changes_replace_it() {
        let original = fixture();
        let previous = build_plan(&original, now(), None).unwrap();
        let id = &previous.segments[0].edge_id;
        for (host_index, _) in original.hosts.iter().enumerate() {
            let mut renamed = original.clone();
            renamed.hosts[host_index].interfaces[0].interface_alias = "Renamed connection".into();
            renamed.hosts[host_index].interfaces[0].interface_index = 77;
            assert_eq!(&build_plan(&renamed, now(), None).unwrap().segments[0].edge_id, id);
            for field in ["guid", "root", "mac"] {
                let mut changed = original.clone();
                match field {
                    "guid" => {
                        changed.hosts[host_index].interfaces[0].interface_guid =
                            "00000000-0000-0000-0000-000000000003".into()
                    }
                    "root" => {
                        changed.hosts[host_index].interfaces[0].root_router_id =
                            Some("USB4\\ROOT_DEVICE_ROUTER\\NEW".into())
                    }
                    _ => {
                        changed.hosts[host_index].interfaces[0].mac = "02:00:00:00:00:03".into();
                        changed.hosts[1 - host_index].interfaces[0].neighbors[0].mac = "02:00:00:00:00:03".into();
                    }
                }
                let plan = build_plan(&changed, now(), Some(&previous)).unwrap();
                assert_ne!(&plan.segments[0].edge_id, id);
                assert_eq!(plan.withdrawn_edge_ids.as_slice(), std::slice::from_ref(id));
            }
        }
    }

    // Protects: both directions name the actual remote interface, not an arbitrary cache IP.
    // Detects: one-way or address-only/alias matching and ambiguous multi-peer admission.
    // Needs: pure fixture; Breadcrumb: build_plan reciprocal match sets.
    #[test]
    fn one_way_wrong_address_and_ambiguous_neighbors_cannot_form_segment() {
        let mut one_way = fixture();
        one_way.hosts[1].interfaces[0].neighbors.clear();
        assert!(build_plan(&one_way, now(), None).unwrap().segments.is_empty());
        let mut wrong_ip = fixture();
        wrong_ip.hosts[0].interfaces[0].neighbors[0].address = "169.254.10.99".into();
        assert!(build_plan(&wrong_ip, now(), None).unwrap().segments.is_empty());
        let mut ambiguous = fixture();
        let mut third = ambiguous.hosts[1].clone();
        third.host = "third".into();
        ambiguous.hosts.push(third);
        assert!(build_plan(&ambiguous, now(), None).unwrap().segments.is_empty());
    }

    // Protects: remote link-local addresses are meaningful only on the local outgoing link.
    // Detects: copying a peer's interface zone into a local connection address.
    // Needs: two distinct interface indices; Breadcrumb: literal_ip and endpoint.
    #[test]
    fn ipv6_link_local_targets_use_each_local_interface_scope() {
        let mut observation = fixture();
        observation.hosts[0].interfaces[0].addresses = vec!["fe80::1%17".into()];
        observation.hosts[0].interfaces[0].neighbors[0].address = "fe80::2%17".into();
        observation.hosts[1].interfaces[0].addresses = vec!["fe80::2%23".into()];
        observation.hosts[1].interfaces[0].neighbors[0].address = "fe80::1%23".into();
        let plan = build_plan(&observation, now(), None).unwrap();
        for endpoint in &plan.segments[0].endpoints {
            assert_eq!(
                endpoint.peer_addresses[0].split('%').next_back().unwrap(),
                endpoint.interface_index.to_string()
            );
        }
        observation.hosts[0].interfaces[0].neighbors[0].address = "fe80::2%23".into();
        assert!(build_plan(&observation, now(), None).is_err());
    }

    // Protects: parallel physical L2 links stay separate, despite shared APIPA prefixes.
    // Detects: automatically collapsing observations into a bridge or one subnet fabric.
    // Needs: two interfaces per host; Breadcrumb: segment derivation/APIPA warning.
    #[test]
    fn parallel_apipa_segments_warn_without_automatic_bridging() {
        let mut observation = fixture();
        observation.hosts[0]
            .interfaces
            .push(interface(3, 18, "169.254.20.1", 4, "169.254.20.2"));
        observation.hosts[1]
            .interfaces
            .push(interface(4, 24, "169.254.20.2", 3, "169.254.20.1"));
        let plan = build_plan(&observation, now(), None).unwrap();
        assert_eq!(plan.segments.len(), 2);
        assert_ne!(plan.segments[0].edge_id, plan.segments[1].edge_id);
        assert!(plan.warnings.iter().any(|warning| warning.contains("APIPA /16")));
        let json = serde_json::to_value(&plan).unwrap();
        assert!(json.get("bridges").is_none() && json.get("commands").is_none() && json.get("routes").is_none());
    }

    // Protects: an unpaired local adapter does not hide overlapping APIPA routes.
    // Detects: warning only after both independent physical segments form edges.
    // Needs: one reciprocal segment and one fresh unmatched USB4 adapter.
    // Breadcrumb: build_plan parallel_local_apipa.
    #[test]
    fn unpaired_local_apipa_adapter_still_warns_about_overlap() {
        let mut observation = fixture();
        observation.hosts[0]
            .interfaces
            .push(interface(3, 18, "169.254.20.1", 4, "169.254.20.2"));
        let plan = build_plan(&observation, now(), None).unwrap();
        assert_eq!(plan.segments.len(), 1);
        assert!(plan.warnings.iter().any(|warning| warning.contains("APIPA /16")));
    }

    // Protects: bounded, distinct device identities and strict serialized contract.
    // Detects: malformed/zero/multicast MACs, duplicate GUID/index/host and invalid TTLs.
    // Needs: pure fixture; Breadcrumb: normalized_guid/normalized_mac/build_plan input checks.
    #[test]
    fn malformed_duplicate_and_out_of_bound_contracts_are_rejected() {
        let mut empty = fixture();
        empty.hosts.clear();
        assert!(build_plan(&empty, now(), None).is_err());
        for mac in [
            "00:00:00:00:00:00",
            "01:00:00:00:00:01",
            "ff:ff:ff:ff:ff:ff",
            "02:0000000001",
        ] {
            let mut input = fixture();
            input.hosts[0].interfaces[0].mac = mac.into();
            assert!(build_plan(&input, now(), None).is_err());
        }
        for ttl in [0, 29, 301] {
            let mut input = fixture();
            input.ttl_seconds = ttl;
            assert!(build_plan(&input, now(), None).is_err());
        }
        let mut input = fixture();
        input.hosts[1].host = "GEN9".into();
        assert!(build_plan(&input, now(), None).is_err());
        let mut input = fixture();
        input.hosts[0].interfaces[0].interface_guid = "00000000-0000-0000-0000-000000000000".into();
        assert!(build_plan(&input, now(), None).is_err());
        for duplicate_field in ["guid", "mac", "index"] {
            let mut input = fixture();
            let mut duplicate = interface(3, 18, "169.254.20.1", 4, "169.254.20.2");
            match duplicate_field {
                "guid" => duplicate.interface_guid = input.hosts[0].interfaces[0].interface_guid.clone(),
                "mac" => duplicate.mac = input.hosts[0].interfaces[0].mac.clone(),
                _ => duplicate.interface_index = input.hosts[0].interfaces[0].interface_index,
            }
            input.hosts[0].interfaces.push(duplicate);
            assert!(build_plan(&input, now(), None).is_err(), "duplicate {duplicate_field}");
        }
        let mut input = fixture();
        input.hosts[0].interfaces[0].neighbors = vec![input.hosts[0].interfaces[0].neighbors[0].clone(); 257];
        assert!(build_plan(&input, now(), None).is_err());
        let mut input = fixture();
        input.hosts[0].interfaces[0].neighbors[0].mac = "01:00:00:00:00:01".into();
        assert!(build_plan(&input, now(), None).is_err());
        let mut json = serde_json::to_value(fixture()).unwrap();
        json["authenticated"] = true.into();
        assert!(serde_json::from_value::<FabricObservation>(json).is_err());
    }
}
