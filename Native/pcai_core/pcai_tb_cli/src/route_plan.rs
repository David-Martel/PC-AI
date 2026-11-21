//! Plans IPv4 addressing, static routes and transit forwarding across observed USB4 hops.
//!
//! A hub, dock or daisy chain gives each host one USB4NET adapter per peer. Windows creates
//! a new adapter instance (new GUID, MAC, ifIndex and alias such as "Ethernet 18" or
//! "USB4(TM) P2P Network Adapter #12") whenever the cable path changes, so per-interface
//! configuration does not survive a hub-port change. This planner therefore:
//!
//! - takes segments only from a fresh [`build_plan`] projection, which pairs endpoints by
//!   reciprocal neighbor MAC and remote address rather than by alias or index;
//! - derives each segment's /30 from the two registered host names
//!   ([`compute_commutative_pairwise_ipv4`]), so the addresses are the same on every plug;
//! - routes between hosts that share no cable through hosts the policy names as transit;
//! - prefixes every host's commands with a guard that the planned ifIndex still carries
//!   the observed MAC and interface GUID, and expires with the observation TTL.
//!
//! It never applies anything. The output is text for an owner to review and run, with a
//! rollback for each action. See `docs/usb4/multi-hop-routing.md`.

use crate::fabric_plan::{build_plan, FabricObservation, ObservedSegment};
use crate::identity::compute_commutative_pairwise_ipv4;
use anyhow::{ensure, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::net::Ipv4Addr;

pub const ROUTE_POLICY_SCHEMA: &str = "pcai.usb4-route-policy/v1";
pub const ROUTE_PLAN_SCHEMA: &str = "pcai.usb4-route-plan/v1";

const PAIRWISE_POOL_BASE: u32 = u32::from_be_bytes([172, 31, 240, 0]);
const PAIRWISE_POOL_MASK: u32 = 0xFFFF_F000; // 172.31.240.0/20

fn default_route_metric() -> u32 {
    5
}

/// Owner-reviewed routing intent. Transit is never inferred from adjacency.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutePolicy {
    pub schema: String,
    /// Hosts allowed to forward IPv4 between their USB4 segments.
    #[serde(default)]
    pub transit_hosts: Vec<String>,
    #[serde(default = "default_route_metric")]
    pub route_metric: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RouteActionKind {
    EdgeAddress,
    HostRoute,
    Forwarding,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PathStatus {
    Direct,
    Routed,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutePlan {
    pub schema: String,
    pub evaluated_at_utc: DateTime<Utc>,
    /// Oldest observation time plus TTL; regenerate after any replug or hub-port change.
    pub expires_at_utc: DateTime<Utc>,
    /// Always false: this command emits text and changes nothing.
    pub apply_supported: bool,
    pub edges: Vec<PlannedEdge>,
    pub paths: Vec<PlannedPath>,
    pub hosts: Vec<HostRoutePlan>,
    pub blocked: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlannedEdge {
    pub edge_id: String,
    pub subnet: String,
    pub endpoints: [PlannedEdgeEnd; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlannedEdgeEnd {
    pub host: String,
    pub interface_index: u32,
    pub interface_guid: String,
    pub mac: String,
    pub interface_alias: String,
    pub ipv4_address: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlannedPath {
    pub from: String,
    pub to: String,
    pub status: PathStatus,
    pub hops: Vec<String>,
    pub destination_ipv4: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostRoutePlan {
    pub host: String,
    pub transit: bool,
    /// Read-only commands: run first and keep the output as the rollback preimage.
    pub preflight: Vec<String>,
    /// Aborts the host's apply when any planned ifIndex now names a different adapter.
    pub guard: Vec<String>,
    pub actions: Vec<RouteAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RouteAction {
    pub kind: RouteActionKind,
    pub interface_index: u32,
    pub target: String,
    pub already_present: bool,
    pub apply: Option<String>,
    pub rollback: Option<String>,
    pub note: Option<String>,
}

#[derive(Debug, Clone)]
struct EdgeEnd {
    host: String,
    interface_index: u32,
    interface_guid: String,
    mac: String,
    alias: String,
    address: Ipv4Addr,
    observed_v4: BTreeSet<Ipv4Addr>,
}

#[derive(Debug, Clone)]
struct Edge {
    edge_id: String,
    subnet: Ipv4Addr,
    ends: [EdgeEnd; 2],
}

impl Edge {
    fn end(&self, host: &str) -> Option<&EdgeEnd> {
        self.ends.iter().find(|end| end.host == host)
    }
    fn other(&self, host: &str) -> Option<&EdgeEnd> {
        self.ends.iter().find(|end| end.host != host)
    }
}

fn windows_mac(normalized: &str) -> String {
    normalized.replace(':', "-").to_ascii_uppercase()
}

fn bare_guid(value: &str) -> String {
    value.trim_matches(|c| c == '{' || c == '}').to_ascii_lowercase()
}

fn in_pairwise_pool(ip: Ipv4Addr) -> bool {
    u32::from(ip) & PAIRWISE_POOL_MASK == PAIRWISE_POOL_BASE
}

fn same_slash30(a: Ipv4Addr, b: Ipv4Addr) -> bool {
    u32::from(a) & 0xFFFF_FFFC == u32::from(b) & 0xFFFF_FFFC
}

fn observed_ipv4(observation: &FabricObservation, host: &str, guid: &str) -> BTreeSet<Ipv4Addr> {
    observation
        .hosts
        .iter()
        .filter(|h| h.host.eq_ignore_ascii_case(host))
        .flat_map(|h| h.interfaces.iter())
        .filter(|i| bare_guid(&i.interface_guid) == guid)
        .flat_map(|i| i.addresses.iter())
        .filter_map(|a| a.parse::<Ipv4Addr>().ok())
        .collect()
}

fn edge_from_segment(
    observation: &FabricObservation,
    segment: &ObservedSegment,
    known_hosts: &BTreeSet<String>,
) -> std::result::Result<Edge, String> {
    let [a, b] = &segment.endpoints;
    for host in [&a.host, &b.host] {
        if !known_hosts.contains(host.as_str()) {
            return Err(format!(
                "Segment {} has unregistered host '{host}'; no deterministic /30 is assigned",
                segment.edge_id
            ));
        }
    }
    let pair = compute_commutative_pairwise_ipv4(&a.host, &b.host).map_err(|e| e.to_string())?;
    let parse = |text: &str| text.parse::<Ipv4Addr>().map_err(|e| e.to_string());
    let a_ip = parse(&pair.local_ipv4_address)?;
    let b_ip = parse(&pair.peer_ipv4_address)?;
    let subnet = parse(pair.subnet_prefix.trim_end_matches("/30"))?;
    let end = |e: &crate::fabric_plan::SegmentEndpoint, address: Ipv4Addr| EdgeEnd {
        host: e.host.clone(),
        interface_index: e.interface_index,
        interface_guid: bare_guid(&e.interface_guid),
        mac: windows_mac(&e.mac),
        alias: e.interface_alias.clone(),
        address,
        observed_v4: observed_ipv4(observation, &e.host, &bare_guid(&e.interface_guid)),
    };
    Ok(Edge {
        edge_id: segment.edge_id.clone(),
        subnet,
        ends: [end(a, a_ip), end(b, b_ip)],
    })
}

/// Shortest host path; only policy transit hosts may appear between the endpoints.
fn shortest_path(
    adjacency: &BTreeMap<String, BTreeMap<String, usize>>,
    from: &str,
    to: &str,
    transit: &BTreeSet<String>,
) -> Option<Vec<String>> {
    let mut previous: BTreeMap<String, String> = BTreeMap::new();
    let mut queue = VecDeque::from([from.to_string()]);
    let mut seen = BTreeSet::from([from.to_string()]);
    while let Some(node) = queue.pop_front() {
        if node == to {
            let mut path = vec![to.to_string()];
            let mut cursor = to.to_string();
            while let Some(prior) = previous.get(&cursor) {
                path.push(prior.clone());
                cursor = prior.clone();
            }
            path.reverse();
            return Some(path);
        }
        if node != from && !transit.contains(&node) {
            continue;
        }
        for next in adjacency.get(&node).into_iter().flat_map(BTreeMap::keys) {
            if seen.insert(next.clone()) {
                previous.insert(next.clone(), node.clone());
                queue.push_back(next.clone());
            }
        }
    }
    None
}

fn guard_lines(ends: &BTreeMap<u32, (String, String)>) -> Vec<String> {
    ends.iter()
        .map(|(index, (mac, guid))| {
            format!(
                "$a = Get-NetAdapter -InterfaceIndex {index} -ErrorAction Stop; if ($a.MacAddress -ne '{mac}' -or [guid]$a.InterfaceGuid -ne [guid]'{guid}') {{ throw 'ifIndex {index} is no longer the observed USB4 endpoint; recollect and regenerate the route plan' }}"
            )
        })
        .collect()
}

/// Builds an owner-reviewed route plan from one fresh bilateral USB4 observation.
///
/// # Errors
/// Propagates [`build_plan`] input errors, rejects an unsupported policy, an unregistered
/// transit host, an out-of-range metric and two segments that derive the same /30.
pub fn build_route_plan(
    observation: &FabricObservation,
    policy: &RoutePolicy,
    known_hosts: &BTreeSet<String>,
    evaluated_at_utc: DateTime<Utc>,
) -> Result<RoutePlan> {
    ensure!(policy.schema == ROUTE_POLICY_SCHEMA, "Unsupported route policy schema");
    ensure!(
        (1..=9999).contains(&policy.route_metric),
        "Route metric must be 1..9999"
    );
    ensure!(
        policy.transit_hosts.len() <= 16,
        "At most 16 transit hosts are supported"
    );
    let transit: BTreeSet<String> = policy.transit_hosts.iter().map(|h| h.to_ascii_lowercase()).collect();
    for host in &transit {
        ensure!(
            known_hosts.contains(host),
            "Transit host '{host}' is not a registered fleet host"
        );
    }
    let fabric = build_plan(observation, evaluated_at_utc, None)?;
    let oldest = observation
        .hosts
        .iter()
        .map(|h| h.observed_at_utc)
        .min()
        .unwrap_or(evaluated_at_utc);
    let mut plan = RoutePlan {
        schema: ROUTE_PLAN_SCHEMA.into(),
        evaluated_at_utc,
        expires_at_utc: oldest + chrono::Duration::seconds(i64::from(observation.ttl_seconds)),
        apply_supported: false,
        edges: vec![],
        paths: vec![],
        hosts: vec![],
        blocked: fabric
            .blocked_endpoints
            .iter()
            .map(|b| format!("{} {}: {}", b.host, b.interface_guid, b.reasons.join("; ")))
            .collect(),
        warnings: vec![
            "Nothing here has been applied. Run each host's guard, then preflight, then actions; keep the preflight output for rollback.".into(),
            "Windows creates a new USB4 adapter (GUID, MAC, ifIndex) when the cable path or hub port changes; recollect and regenerate after any replug.".into(),
            "Forwarding and routes do not change Windows Firewall: a destination's inbound rules must admit the remote /30 source.".into(),
        ],
    };

    // Edges keyed by host pair; parallel segments keep the lowest edge ID.
    let mut edges: BTreeMap<(String, String), Edge> = BTreeMap::new();
    for segment in &fabric.segments {
        match edge_from_segment(observation, segment, known_hosts) {
            Ok(edge) => {
                let key = (edge.ends[0].host.clone(), edge.ends[1].host.clone());
                if let Some(existing) = edges.get(&key) {
                    plan.warnings.push(format!(
                        "Parallel USB4 segments between {} and {}; using {} and leaving {} unaddressed",
                        key.0, key.1, existing.edge_id, edge.edge_id
                    ));
                } else {
                    edges.insert(key, edge);
                }
            }
            Err(reason) => plan.blocked.push(reason),
        }
    }
    let mut subnets = BTreeMap::new();
    for edge in edges.values() {
        if let Some(other) = subnets.insert(edge.subnet, edge.edge_id.clone()) {
            anyhow::bail!(
                "Segments {other} and {} derive the same /30 {}; deterministic pair addressing collides",
                edge.edge_id,
                edge.subnet
            );
        }
    }
    let edge_list: Vec<Edge> = edges.into_values().collect();
    let mut adjacency: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for (index, edge) in edge_list.iter().enumerate() {
        let (a, b) = (&edge.ends[0].host, &edge.ends[1].host);
        adjacency.entry(a.clone()).or_default().insert(b.clone(), index);
        adjacency.entry(b.clone()).or_default().insert(a.clone(), index);
    }

    let mut actions: BTreeMap<String, Vec<RouteAction>> = BTreeMap::new();
    let mut guards: BTreeMap<String, BTreeMap<u32, (String, String)>> = BTreeMap::new();
    let mut transit_used = BTreeSet::new();
    let mut conflicts = Vec::new();
    // One action per (kind, interface, target): the per-pair note differs when a transit
    // host routes the same destination both for itself and for an upstream source. A second
    // next hop for the same destination on the same host is a conflict, never a duplicate.
    let mut push = |host: &str, action: RouteAction| {
        let list = actions.entry(host.to_string()).or_default();
        let destination = |a: &RouteAction| a.target.split(" via ").next().map(str::to_string);
        if list.iter().any(|existing| {
            existing.kind == action.kind
                && existing.interface_index == action.interface_index
                && existing.target == action.target
        }) {
            return;
        }
        if action.kind == RouteActionKind::HostRoute {
            if let Some(existing) = list
                .iter()
                .find(|e| e.kind == RouteActionKind::HostRoute && destination(e) == destination(&action))
            {
                conflicts.push(format!(
                    "{host}: keeping route {} on ifIndex {}; dropped conflicting {} on ifIndex {}",
                    existing.target, existing.interface_index, action.target, action.interface_index
                ));
                return;
            }
        }
        list.push(action);
    };

    for edge in &edge_list {
        for end in &edge.ends {
            guards
                .entry(end.host.clone())
                .or_default()
                .insert(end.interface_index, (end.mac.clone(), end.interface_guid.clone()));
            for stale in end
                .observed_v4
                .iter()
                .filter(|ip| in_pairwise_pool(**ip) && **ip != end.address)
            {
                plan.warnings.push(format!(
                    "{} ifIndex {} holds pool address {stale}, not the derived {}; review before applying",
                    end.host, end.interface_index, end.address
                ));
            }
            let present = end.observed_v4.contains(&end.address);
            let (index, ip) = (end.interface_index, end.address);
            push(
                &end.host,
                RouteAction {
                    kind: RouteActionKind::EdgeAddress,
                    interface_index: index,
                    target: format!("{ip}/30"),
                    already_present: present,
                    apply: (!present).then(|| {
                        format!("New-NetIPAddress -InterfaceIndex {index} -AddressFamily IPv4 -IPAddress {ip} -PrefixLength 30")
                    }),
                    rollback: (!present)
                        .then(|| format!("Remove-NetIPAddress -InterfaceIndex {index} -IPAddress {ip} -Confirm:$false")),
                    note: present.then(|| "Observed on this interface; not owned by this plan".to_string()),
                },
            );
        }
        plan.edges.push(PlannedEdge {
            edge_id: edge.edge_id.clone(),
            subnet: format!("{}/30", edge.subnet),
            endpoints: edge.ends.clone().map(|end| PlannedEdgeEnd {
                host: end.host,
                interface_index: end.interface_index,
                interface_guid: end.interface_guid,
                mac: end.mac,
                interface_alias: end.alias,
                ipv4_address: end.address.to_string(),
            }),
        });
    }

    let hosts: Vec<String> = adjacency.keys().cloned().collect();
    let metric = policy.route_metric;
    for from in &hosts {
        for to in hosts.iter().filter(|to| *to != from) {
            let Some(path) = shortest_path(&adjacency, from, to, &transit) else {
                let reason = if shortest_path(&adjacency, from, to, &hosts.iter().cloned().collect()).is_some() {
                    "Reachable only through a host the policy does not name as transit"
                } else {
                    "No observed USB4 path"
                };
                plan.paths.push(PlannedPath {
                    from: from.clone(),
                    to: to.clone(),
                    status: PathStatus::Blocked,
                    hops: vec![],
                    destination_ipv4: None,
                    reason: Some(reason.into()),
                });
                continue;
            };
            let hop_edge = |i: usize| &edge_list[adjacency[&path[i]][&path[i + 1]]];
            // The destination answers on the address of the last segment it is reached by.
            let last = hop_edge(path.len() - 2);
            let destination = last.end(to).map(|end| end.address);
            if path.len() == 2 {
                plan.paths.push(PlannedPath {
                    from: from.clone(),
                    to: to.clone(),
                    status: PathStatus::Direct,
                    hops: path,
                    destination_ipv4: destination.map(|ip| ip.to_string()),
                    reason: None,
                });
                continue;
            }
            let Some(destination) = destination else { continue };
            // Replies go to the address the source sent from: its end of the first segment.
            // Plan that leg explicitly along the reversed path; the independently computed
            // reverse pair can choose an equal-length path that reaches a different address.
            let source = hop_edge(0).end(from).map(|end| end.address);
            let mut legs = vec![(path.clone(), destination, format!("{from} -> {to}"))];
            if let Some(source) = source {
                legs.push((
                    path.iter().rev().cloned().collect(),
                    source,
                    format!("{to} -> {from} (reply)"),
                ));
            }
            for (leg, target_ip, note) in &legs {
                // Every hop that is not already on the target's /30 needs a host route.
                for (i, node) in leg.iter().enumerate().take(leg.len() - 1) {
                    let edge = &edge_list[adjacency[node][&leg[i + 1]]];
                    let (Some(out), Some(next)) = (edge.end(node), edge.other(node)) else {
                        continue;
                    };
                    if same_slash30(out.address, *target_ip) {
                        continue;
                    }
                    let index = out.interface_index;
                    let gateway = next.address;
                    push(
                        node,
                        RouteAction {
                            kind: RouteActionKind::HostRoute,
                            interface_index: index,
                            target: format!("{target_ip}/32 via {gateway}"),
                            already_present: false,
                            apply: Some(format!(
                                "New-NetRoute -InterfaceIndex {index} -AddressFamily IPv4 -DestinationPrefix {target_ip}/32 -NextHop {gateway} -RouteMetric {metric}"
                            )),
                            rollback: Some(format!(
                                "Remove-NetRoute -InterfaceIndex {index} -DestinationPrefix {target_ip}/32 -NextHop {gateway} -Confirm:$false"
                            )),
                            note: Some(note.clone()),
                        },
                    );
                }
            }
            for (i, node) in path.iter().enumerate().take(path.len() - 1).skip(1) {
                transit_used.insert(node.clone());
                for edge in [hop_edge(i - 1), hop_edge(i)] {
                    if let Some(end) = edge.end(node) {
                        let index = end.interface_index;
                        push(
                            node,
                            RouteAction {
                                kind: RouteActionKind::Forwarding,
                                interface_index: index,
                                target: format!("IPv4 forwarding on ifIndex {index}"),
                                already_present: false,
                                apply: Some(format!(
                                    "Set-NetIPInterface -InterfaceIndex {index} -AddressFamily IPv4 -Forwarding Enabled"
                                )),
                                rollback: Some(format!(
                                    "Set-NetIPInterface -InterfaceIndex {index} -AddressFamily IPv4 -Forwarding Disabled"
                                )),
                                note: Some(
                                    "Forwarding preimage NotCollected: restore Disabled only if the preflight showed Disabled"
                                        .into(),
                                ),
                            },
                        );
                    }
                }
            }
            plan.paths.push(PlannedPath {
                from: from.clone(),
                to: to.clone(),
                status: PathStatus::Routed,
                hops: path,
                destination_ipv4: Some(destination.to_string()),
                reason: None,
            });
        }
    }

    plan.warnings.extend(conflicts);
    for (host, mut list) in actions {
        list.sort_by_key(|a| (a.kind, a.interface_index, a.target.clone()));
        let host_guards = guards.remove(&host).unwrap_or_default();
        let preflight = host_guards
            .keys()
            .flat_map(|index| {
                [
                    format!("Get-NetIPInterface -InterfaceIndex {index} -AddressFamily IPv4 | Select-Object InterfaceIndex, Forwarding, InterfaceMetric, NlMtu"),
                    format!("Get-NetIPAddress -InterfaceIndex {index} -AddressFamily IPv4 | Select-Object InterfaceIndex, IPAddress, PrefixLength, PrefixOrigin"),
                    format!("Get-NetRoute -InterfaceIndex {index} -AddressFamily IPv4 | Select-Object InterfaceIndex, DestinationPrefix, NextHop, RouteMetric"),
                ]
            })
            .collect();
        plan.hosts.push(HostRoutePlan {
            transit: transit_used.contains(&host),
            guard: guard_lines(&host_guards),
            preflight,
            actions: list,
            host,
        });
    }
    for host in transit.iter().filter(|h| !transit_used.contains(*h)) {
        plan.warnings.push(format!(
            "Transit host '{host}' carries no planned path; no forwarding is planned there"
        ));
    }
    if evaluated_at_utc >= plan.expires_at_utc {
        plan.warnings
            .push("Observation TTL has expired; the plan describes stale adapters".into());
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HUB_CHAIN: &str = include_str!("../tests/fixtures/usb4-hub-chain-observation.json");

    fn now() -> DateTime<Utc> {
        "2026-11-21T09:31:00Z".parse().unwrap()
    }
    fn observation() -> FabricObservation {
        serde_json::from_str(HUB_CHAIN).unwrap()
    }
    fn fleet() -> BTreeSet<String> {
        ["dtm-p1gen7", "dtm-p1gen9", "dtm-super-nuc"]
            .into_iter()
            .map(String::from)
            .collect()
    }
    fn policy(transit: &[&str]) -> RoutePolicy {
        RoutePolicy {
            schema: ROUTE_POLICY_SCHEMA.into(),
            transit_hosts: transit.iter().map(|h| (*h).to_string()).collect(),
            route_metric: 5,
        }
    }
    fn host<'a>(plan: &'a RoutePlan, name: &str) -> &'a HostRoutePlan {
        plan.hosts.iter().find(|h| h.host == name).unwrap()
    }
    fn path<'a>(plan: &'a RoutePlan, from: &str, to: &str) -> &'a PlannedPath {
        plan.paths.iter().find(|p| p.from == from && p.to == to).unwrap()
    }

    // Protects: the recorded Gen7 -> NUC -> Gen9 chain (2026-11-21) routes only through the
    // named transit host. Detects: missing return routes, forwarding on an endpoint, or a
    // route on the hop that already shares the destination's /30. Needs: the sanitized
    // recorded fixture. Breadcrumb: build_route_plan path loop.
    #[test]
    fn recorded_hub_chain_routes_through_named_transit_only() {
        let plan = build_route_plan(&observation(), &policy(&["dtm-super-nuc"]), &fleet(), now()).unwrap();
        assert_eq!(plan.edges.len(), 2);
        assert!(!plan.apply_supported);
        let forward = path(&plan, "dtm-p1gen7", "dtm-p1gen9");
        assert_eq!(forward.status, PathStatus::Routed);
        assert_eq!(forward.hops, ["dtm-p1gen7", "dtm-super-nuc", "dtm-p1gen9"]);
        assert_eq!(path(&plan, "dtm-p1gen9", "dtm-p1gen7").status, PathStatus::Routed);
        assert_eq!(path(&plan, "dtm-p1gen7", "dtm-super-nuc").status, PathStatus::Direct);

        let gen9_on_nuc_link = forward.destination_ipv4.clone().unwrap();
        let gen7 = host(&plan, "dtm-p1gen7");
        let gen7_routes: Vec<_> = gen7
            .actions
            .iter()
            .filter(|a| a.kind == RouteActionKind::HostRoute)
            .collect();
        assert_eq!(gen7_routes.len(), 1);
        assert!(gen7_routes[0]
            .target
            .starts_with(&format!("{gen9_on_nuc_link}/32 via ")));
        assert_eq!(gen7_routes[0].interface_index, 31);
        assert!(!gen7.transit);
        assert!(gen7.actions.iter().all(|a| a.kind != RouteActionKind::Forwarding));

        let nuc = host(&plan, "dtm-super-nuc");
        assert!(nuc.transit);
        let forwarding: BTreeSet<_> = nuc
            .actions
            .iter()
            .filter(|a| a.kind == RouteActionKind::Forwarding)
            .map(|a| a.interface_index)
            .collect();
        assert_eq!(forwarding, BTreeSet::from([12, 19]));
        assert!(nuc.actions.iter().all(|a| a.kind != RouteActionKind::HostRoute));
        assert_eq!(
            host(&plan, "dtm-p1gen9")
                .actions
                .iter()
                .filter(|a| a.kind == RouteActionKind::HostRoute)
                .count(),
            1
        );
    }

    // Protects: deterministic addressing agrees with what `register` already applied on the
    // NUC (172.31.255.22/30 toward Gen9) and leaves that address unowned. Detects: a
    // derivation change that would renumber live links, or a rollback that removes state
    // this plan did not create. Needs: the recorded NUC address. Breadcrumb: EdgeAddress.
    #[test]
    fn existing_registered_address_is_derived_and_not_owned() {
        let plan = build_route_plan(&observation(), &policy(&["dtm-super-nuc"]), &fleet(), now()).unwrap();
        let nuc = host(&plan, "dtm-super-nuc");
        let existing = nuc
            .actions
            .iter()
            .find(|a| a.kind == RouteActionKind::EdgeAddress && a.interface_index == 19)
            .unwrap();
        assert_eq!(existing.target, "172.31.255.22/30");
        assert!(existing.already_present && existing.apply.is_none() && existing.rollback.is_none());
        let gen9 = host(&plan, "dtm-p1gen9");
        let missing = gen9
            .actions
            .iter()
            .find(|a| a.kind == RouteActionKind::EdgeAddress)
            .unwrap();
        assert_eq!(missing.target, "172.31.255.21/30");
        assert!(!missing.already_present);
        assert!(missing
            .rollback
            .as_deref()
            .unwrap()
            .starts_with("Remove-NetIPAddress -InterfaceIndex 28 "));
    }

    // Protects: transit is explicit. Detects: forwarding inferred from adjacency (the NUC
    // forwarded nothing on 2026-11-21 and must not be made a router silently). Needs: an
    // empty policy. Breadcrumb: shortest_path transit filter.
    #[test]
    fn no_transit_policy_blocks_multi_hop_and_plans_no_forwarding() {
        let plan = build_route_plan(&observation(), &policy(&[]), &fleet(), now()).unwrap();
        let blocked = path(&plan, "dtm-p1gen7", "dtm-p1gen9");
        assert_eq!(blocked.status, PathStatus::Blocked);
        assert!(blocked.reason.as_deref().unwrap().contains("does not name as transit"));
        assert!(plan
            .hosts
            .iter()
            .flat_map(|h| &h.actions)
            .all(|a| a.kind != RouteActionKind::Forwarding && a.kind != RouteActionKind::HostRoute));
    }

    // Protects: hub replug safety. Detects: commands that address an ifIndex without
    // checking it is still the observed adapter (Gen9's adapter became #12/ifIndex 28 after
    // a hub-port change). Needs: fixture MAC/GUID. Breadcrumb: guard_lines.
    #[test]
    fn every_planned_interface_has_a_mac_and_guid_guard() {
        let plan = build_route_plan(&observation(), &policy(&["dtm-super-nuc"]), &fleet(), now()).unwrap();
        for host in &plan.hosts {
            let guarded: BTreeSet<u32> = host
                .guard
                .iter()
                .map(|g| {
                    g.split_whitespace()
                        .nth(4)
                        .unwrap()
                        .trim_end_matches(';')
                        .parse()
                        .unwrap()
                })
                .collect();
            for action in &host.actions {
                assert!(guarded.contains(&action.interface_index), "{action:?} lacks a guard");
            }
        }
        let gen9 = &host(&plan, "dtm-p1gen9").guard[0];
        assert!(gen9.contains("'02-00-00-09-00-28'") && gen9.contains("a0000000-0000-4000-8000-000000000928"));
    }

    // Protects: stale and unregistered input fail closed. Detects: planning from an expired
    // observation, unregistered transit, or addressing a host with no registry identity.
    // Needs: fixture with shifted clock and reduced fleet. Breadcrumb: build_route_plan gates.
    #[test]
    fn expired_unregistered_and_bad_policy_inputs_are_refused_or_blocked() {
        let late = now() + chrono::Duration::seconds(400);
        let stale = build_route_plan(&observation(), &policy(&["dtm-super-nuc"]), &fleet(), late).unwrap();
        assert!(stale.edges.is_empty() && stale.hosts.is_empty());
        assert!(stale.warnings.iter().any(|w| w.contains("TTL has expired")));

        let mut partial = fleet();
        partial.remove("dtm-p1gen7");
        let plan = build_route_plan(&observation(), &policy(&["dtm-super-nuc"]), &partial, now()).unwrap();
        assert_eq!(plan.edges.len(), 1);
        assert!(plan
            .blocked
            .iter()
            .any(|b| b.contains("unregistered host 'dtm-p1gen7'")));

        assert!(build_route_plan(&observation(), &policy(&["intruder"]), &fleet(), now()).is_err());
        let mut wrong = policy(&[]);
        wrong.schema = "other".into();
        assert!(build_route_plan(&observation(), &wrong, &fleet(), now()).is_err());
        let mut metric = policy(&[]);
        metric.route_metric = 0;
        assert!(build_route_plan(&observation(), &metric, &fleet(), now()).is_err());
    }

    // Protects: a pool address left behind by an earlier plug is surfaced, not silently kept
    // or removed. Detects: missing review warning for a mismatched 172.31.240.0/20 address.
    // Needs: fixture with an extra pool address. Breadcrumb: stale pool warning.
    #[test]
    fn foreign_pool_address_on_an_endpoint_is_reported() {
        let mut obs = observation();
        obs.hosts[2].interfaces[0].addresses.push("172.31.252.133".into());
        let plan = build_route_plan(&obs, &policy(&["dtm-super-nuc"]), &fleet(), now()).unwrap();
        assert!(plan
            .warnings
            .iter()
            .any(|w| w.contains("dtm-p1gen9 ifIndex 28 holds pool address 172.31.252.133")));
    }

    fn chain_interface(
        guid: u32,
        index: u32,
        address: &str,
        peer_mac: u32,
        peer_address: &str,
    ) -> crate::fabric_plan::InterfaceObservation {
        crate::fabric_plan::InterfaceObservation {
            interface_guid: format!("00000000-0000-4000-8000-{guid:012x}"),
            mac: format!("02:00:00:00:00:{guid:02x}"),
            pnp_instance_id: format!("USB4\\VEN_045E&DEV_0000083F&PROT_USB4NET\\CHAIN{guid}"),
            root_router_id: Some(format!("USB4\\ROOT_DEVICE_ROUTER\\CHAINROOT{guid}")),
            interface_index: index,
            interface_alias: "Ethernet".into(),
            up: true,
            addresses: vec![address.into()],
            neighbors: vec![crate::fabric_plan::NeighborObservation {
                address: peer_address.into(),
                mac: format!("02:00:00:00:00:{peer_mac:02x}"),
            }],
        }
    }

    // Protects: chains with two transit hosts (A - T1 - T2 - B). Detects: duplicate
    // New-NetRoute/Remove-NetRoute lines when a transit host routes the same destination for
    // itself and for an upstream source (gemini review of #335), and missing third-hop routes.
    // Needs: synthetic four-host observation. Breadcrumb: build_route_plan push dedup.
    #[test]
    fn two_transit_chain_plans_each_route_once() {
        use crate::fabric_plan::HostObservation;
        let make_host = |name: &str, interfaces| HostObservation {
            observed_at_utc: now(),
            host: name.into(),
            source: crate::fabric_plan::ObservationSource::ManagementSsh,
            interfaces,
            reference_applications: vec![],
        };
        let observation = FabricObservation {
            schema: crate::fabric_plan::OBSERVATION_SCHEMA.into(),
            ttl_seconds: 300,
            hosts: vec![
                make_host("chain-a", vec![chain_interface(1, 11, "169.254.1.1", 2, "169.254.1.2")]),
                make_host(
                    "chain-t1",
                    vec![
                        chain_interface(2, 21, "169.254.1.2", 1, "169.254.1.1"),
                        chain_interface(3, 22, "169.254.2.1", 4, "169.254.2.2"),
                    ],
                ),
                make_host(
                    "chain-t2",
                    vec![
                        chain_interface(4, 31, "169.254.2.2", 3, "169.254.2.1"),
                        chain_interface(5, 32, "169.254.3.1", 6, "169.254.3.2"),
                    ],
                ),
                make_host("chain-b", vec![chain_interface(6, 41, "169.254.3.2", 5, "169.254.3.1")]),
            ],
        };
        let known: BTreeSet<String> = ["chain-a", "chain-t1", "chain-t2", "chain-b"]
            .into_iter()
            .map(String::from)
            .collect();
        let plan = build_route_plan(&observation, &policy(&["chain-t1", "chain-t2"]), &known, now()).unwrap();
        assert_eq!(plan.edges.len(), 3);
        assert_eq!(path(&plan, "chain-a", "chain-b").hops.len(), 4);
        for host in &plan.hosts {
            let mut seen = BTreeSet::new();
            for action in &host.actions {
                assert!(
                    seen.insert((action.kind, action.interface_index, action.target.clone())),
                    "{} repeats {action:?}",
                    host.host
                );
            }
        }
        let t1_routes = host(&plan, "chain-t1")
            .actions
            .iter()
            .filter(|a| a.kind == RouteActionKind::HostRoute)
            .count();
        assert_eq!(t1_routes, 1, "T1 needs exactly one route, to B's far /30");
        assert!(!plan.warnings.iter().any(|w| w.contains("conflicting")));
    }

    fn mesh_observation(links: &[(&str, &str)]) -> FabricObservation {
        use crate::fabric_plan::HostObservation;
        let mut hosts: BTreeMap<String, Vec<crate::fabric_plan::InterfaceObservation>> = BTreeMap::new();
        for (k, (x, y)) in links.iter().enumerate() {
            let k = u32::try_from(k).unwrap() + 1;
            let (gx, gy) = (2 * k + 1, 2 * k + 2);
            let (ax, ay) = (format!("169.254.{k}.1"), format!("169.254.{k}.2"));
            hosts
                .entry((*x).to_string())
                .or_default()
                .push(chain_interface(gx, 10 + gx, &ax, gy, &ay));
            hosts
                .entry((*y).to_string())
                .or_default()
                .push(chain_interface(gy, 10 + gy, &ay, gx, &ax));
        }
        FabricObservation {
            schema: crate::fabric_plan::OBSERVATION_SCHEMA.into(),
            ttl_seconds: 300,
            hosts: hosts
                .into_iter()
                .map(|(host, interfaces)| HostObservation {
                    observed_at_utc: now(),
                    host,
                    source: crate::fabric_plan::ObservationSource::ManagementSsh,
                    interfaces,
                    reference_applications: vec![],
                })
                .collect(),
        }
    }

    /// Follows on-link /30s and planned host routes from `start` toward `target`.
    fn delivers(plan: &RoutePlan, start: &str, target: Ipv4Addr) -> bool {
        let ends: Vec<&PlannedEdgeEnd> = plan.edges.iter().flat_map(|e| e.endpoints.iter()).collect();
        let owner = |ip: Ipv4Addr| {
            ends.iter()
                .find(|e| e.ipv4_address == ip.to_string())
                .map(|e| e.host.clone())
        };
        let mut at = start.to_string();
        for _ in 0..16 {
            if owner(target).as_deref() == Some(at.as_str()) {
                return true;
            }
            let on_link = ends
                .iter()
                .find(|e| e.host == at && same_slash30(e.ipv4_address.parse().unwrap(), target));
            if on_link.is_some() {
                at = owner(target).unwrap();
                continue;
            }
            let Some(route) = plan.hosts.iter().find(|h| h.host == at).and_then(|h| {
                h.actions.iter().find(|a| {
                    a.kind == RouteActionKind::HostRoute && a.target.starts_with(&format!("{target}/32 via "))
                })
            }) else {
                return false;
            };
            let gateway: Ipv4Addr = route.target.rsplit(' ').next().unwrap().parse().unwrap();
            at = owner(gateway).unwrap();
        }
        false
    }

    fn assert_every_routed_pair_round_trips(plan: &RoutePlan) {
        for path in plan.paths.iter().filter(|p| p.status != PathStatus::Blocked) {
            let destination: Ipv4Addr = path.destination_ipv4.as_deref().unwrap().parse().unwrap();
            assert!(
                delivers(plan, &path.from, destination),
                "{} cannot reach {}",
                path.from,
                path.to
            );
            let first = plan
                .edges
                .iter()
                .find(|e| {
                    e.endpoints.iter().any(|x| x.host == path.from)
                        && e.endpoints.iter().any(|x| x.host == path.hops[1])
                })
                .unwrap();
            let source: Ipv4Addr = first
                .endpoints
                .iter()
                .find(|x| x.host == path.from)
                .unwrap()
                .ipv4_address
                .parse()
                .unwrap();
            assert!(
                delivers(plan, &path.to, source),
                "{} cannot reply to {} at {source}",
                path.to,
                path.from
            );
        }
    }

    // Protects: replies reach the address a packet was actually sent from. Detects: return
    // routes planned only from the independently chosen reverse path, which with equal-length
    // alternatives reaches a different source address (codex review of #335: a-e-c-d forward,
    // d-b-f-a reverse). Needs: the reviewer's six-host mesh. Breadcrumb: reply legs.
    #[test]
    fn replies_reach_the_forward_source_address_on_equal_cost_meshes() {
        let links = [("a", "e"), ("a", "f"), ("b", "d"), ("b", "f"), ("c", "d"), ("c", "e")];
        let known: BTreeSet<String> = ["a", "b", "c", "d", "e", "f"].into_iter().map(String::from).collect();
        let all: Vec<&str> = known.iter().map(String::as_str).collect();
        let plan = build_route_plan(&mesh_observation(&links), &policy(&all), &known, now()).unwrap();
        assert_eq!(plan.edges.len(), 6);
        assert_every_routed_pair_round_trips(&plan);

        let chain = [("a", "b"), ("b", "c"), ("c", "d")];
        let plan = build_route_plan(&mesh_observation(&chain), &policy(&["b", "c"]), &known, now()).unwrap();
        assert_every_routed_pair_round_trips(&plan);
        let recorded = build_route_plan(&observation(), &policy(&["dtm-super-nuc"]), &fleet(), now()).unwrap();
        assert_every_routed_pair_round_trips(&recorded);
    }
}
