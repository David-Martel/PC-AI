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

/// Proves the journal storage is writable only by Administrators and SYSTEM, then leaves the
/// journal path in `$j`. Rollback treats journal keys as authority to remove state, so a
/// directory a standard user could create, redirect or write first must never be read.
///
/// `%ProgramData%` lets every user create subdirectories. The leaf `route-plan` directory must
/// be owned by Administrators or SYSTEM, must not be a reparse point and must grant no write,
/// delete or permission right to any other SID. Its parent `pcai-usb4` may keep the inherited
/// right of Users to add entries, because they cannot remove or rename an admin-owned child,
/// but it must still be admin-owned, not a reparse point and grant no other SID delete or
/// permission rights over itself. Apply creates a missing directory with a protected ACL and
/// refuses a new leaf that is not empty. Rollback creates nothing and skips the check when the
/// leaf is absent, since no journal can then exist.
fn journal_guard(plan_id: &str, create: bool) -> String {
    const VERIFY: &str = "$i = Get-Item -LiteralPath $p -Force -ErrorAction Stop; if ($i.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw \"Journal storage $p is a reparse point; review and remove it\" }; $acl = Get-Acl -LiteralPath $p -ErrorAction Stop; if ($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -notin $ok) { throw \"Journal storage $p is not owned by Administrators or SYSTEM; review and remove it\" }; foreach ($r in $acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])) { if ($r.AccessControlType -eq 'Allow' -and $r.IdentityReference.Value -notin $ok -and ($p -eq $d -or -not ($r.PropagationFlags -band [Security.AccessControl.PropagationFlags]::InheritOnly)) -and ([int64]$r.FileSystemRights -band $(if ($p -eq $d) { 0x500D0156 } else { 0x100D0040 }))) { throw \"Journal storage $p lets $($r.IdentityReference.Value) change it; review and remove it\" } }";
    const CREATE: &str = "$s = [Security.AccessControl.DirectorySecurity]::new(); $s.SetAccessRuleProtection($true, $false); $s.SetOwner([Security.Principal.SecurityIdentifier]'S-1-5-32-544'); foreach ($id in $ok) { $s.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]$id, 'FullControl', 'ContainerInherit, ObjectInherit', 'None', 'Allow')) }; if ($p -ne $d) { $s.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]'S-1-5-32-545', 'ReadAndExecute', 'ContainerInherit, ObjectInherit', 'None', 'Allow')) }; New-Item -ItemType Directory -Path $p -ErrorAction Stop | Out-Null; Set-Acl -LiteralPath $p -AclObject $s -ErrorAction Stop";
    // The journal file must not be a reparse point and must be owned by Administrators, SYSTEM or
    // the caller (elevated; client Windows makes the creator the owner of a file it adds).
    const FILE: &str = "if (Test-Path -LiteralPath $j) { $fi = Get-Item -LiteralPath $j -Force -ErrorAction Stop; if ($fi.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw \"Journal $j is a reparse point; review and remove it\" }; $fo = (Get-Acl -LiteralPath $j -ErrorAction Stop).GetOwner([Security.Principal.SecurityIdentifier]).Value; if ($fo -notin $ok -and $fo -ne [Security.Principal.WindowsIdentity]::GetCurrent().User.Value) { throw \"Journal $j is owned by $fo; review and remove it\" } }";
    let head = format!(
        "$ok = 'S-1-5-32-544', 'S-1-5-18'; $d = Join-Path $env:ProgramData 'pcai-usb4\\route-plan'; $j = Join-Path $d '{plan_id}.journal'"
    );
    if create {
        format!(
            "{head}; foreach ($p in (Split-Path $d), $d) {{ $new = -not (Test-Path -LiteralPath $p); if ($new) {{ {CREATE} }}; {VERIFY}; if ($new -and $p -eq $d -and @(Get-ChildItem -LiteralPath $p -Force).Count) {{ throw \"New journal storage $p is not empty\" }} }}; {FILE}"
        )
    } else {
        format!(
            "{head}; if (Test-Path -LiteralPath $d) {{ foreach ($p in (Split-Path $d), $d) {{ {VERIFY} }}; {FILE} }}"
        )
    }
}

/// Rewrites the journal without `key`.
fn retire_key(key: &str) -> String {
    format!(
        "Set-Content -LiteralPath $j -Value @(Get-Content -LiteralPath $j -ErrorAction Stop | Where-Object {{ $_ -ne '{key}' }}) -ErrorAction Stop"
    )
}

/// Bounds a pasted apply command by the original planner's UTC observation window.
fn apply_clock_guard(evaluated_at_utc: DateTime<Utc>, expires_at_utc: DateTime<Utc>) -> String {
    let start = evaluated_at_utc.to_rfc3339();
    let expiry = expires_at_utc.to_rfc3339();
    format!(
        "$pcaiPlanNow = [DateTimeOffset]::UtcNow; if ($pcaiPlanNow -lt [DateTimeOffset]'{start}' -or $pcaiPlanNow -ge [DateTimeOffset]'{expiry}') {{ throw 'USB4 route plan is outside its validity window. Regenerate it from fresh observations.' }}"
    )
}

/// Apply creates state only when it is absent. The key is journaled before the change, so no
/// change can exist without a record; a failed change whose state is still absent retires the
/// key again. State that existed before apply is reported as not owned and never journaled.
fn owned_apply(plan_id: &str, clock_guard: &str, key: &str, present: &str, command: &str) -> String {
    let guard = journal_guard(plan_id, true);
    let retire = retire_key(key);
    format!(
        "{guard}; $owned = @(Get-Content -LiteralPath $j -ErrorAction SilentlyContinue) -contains '{key}'; if ({present}) {{ if ($owned) {{ Write-Output 'Already applied by this plan: {key}' }} else {{ Write-Output 'Present before apply; not owned: {key}' }} }} else {{ if (-not $owned) {{ Add-Content -LiteralPath $j -Value '{key}' -ErrorAction Stop }}; try {{ {clock_guard}; {command} -ErrorAction Stop | Out-Null }} catch {{ if (-not ({present})) {{ {retire} }}; throw }} }}"
    )
}

/// True when `query` (piped through `filter`) finds the object, false when the cmdlet reports it
/// as not found, and a thrown error for any other failure. The NetTCPIP cmdlets report a missing
/// address or route as an ObjectNotFound error, so `-ErrorAction SilentlyContinue` alone would
/// read a failed CIM query as "absent", and rollback would then retire a key whose state is still
/// active.
fn found(query: &str, filter: &str) -> String {
    format!(
        "(& {{ $e = @(); $r = @({query} -ErrorAction SilentlyContinue -ErrorVariable e{filter}); if (@($e | Where-Object {{ $_.CategoryInfo.Category -ne 'ObjectNotFound' }}).Count) {{ throw ('State query failed; nothing changed and the journal key is kept: ' + $e[0]) }}; $r.Count -gt 0 }})"
    )
}

/// A planned host route on the same host to the same destination as `action` but through another
/// interface or next hop. `None` for anything that is not a host route.
fn conflicting_route<'a>(list: &'a [RouteAction], action: &RouteAction) -> Option<&'a RouteAction> {
    if action.kind != RouteActionKind::HostRoute {
        return None;
    }
    let destination = |a: &RouteAction| a.target.split(" via ").next().map(str::to_string);
    list.iter().find(|existing| {
        existing.kind == RouteActionKind::HostRoute
            && destination(existing) == destination(action)
            && (existing.interface_index, &existing.target) != (action.interface_index, &action.target)
    })
}

/// The ifIndex still names the adapter the plan observed (MAC and GUID). Windows reuses an index
/// for a new adapter after a replug, so an index alone never authorizes a change.
fn adapter_matches(index: u32, mac: &str, guid: &str) -> String {
    format!(
        "($a = Get-NetAdapter -InterfaceIndex {index} -ErrorAction SilentlyContinue) -and $a.MacAddress -eq '{mac}' -and [guid]$a.InterfaceGuid -eq [guid]'{guid}'"
    )
}

/// Rollback acts only on keys this plan's apply journaled on this host, and only while the
/// ifIndex still names the adapter the plan observed; otherwise it changes nothing and keeps the
/// key. It removes the state only while it is present, then retires the key, so a later rollback
/// cannot remove state that someone else recreates.
fn owned_rollback(plan_id: &str, key: &str, adapter: &str, present: &str, command: &str) -> String {
    let guard = journal_guard(plan_id, false);
    let retire = retire_key(key);
    format!(
        "{guard}; if (@(Get-Content -LiteralPath $j -ErrorAction SilentlyContinue) -contains '{key}') {{ if (-not ({adapter})) {{ Write-Output 'The ifIndex no longer names the planned adapter; nothing changed and the journal key is kept: {key}' }} else {{ if ({present}) {{ {command} -ErrorAction Stop }}; {retire} }} }}"
    )
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
    let clock_guard = apply_clock_guard(plan.evaluated_at_utc, plan.expires_at_utc);

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
    let plan_id = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(evaluated_at_utc.to_rfc3339().as_bytes());
        for edge in &edge_list {
            hasher.update(edge.edge_id.as_bytes());
        }
        hex::encode(hasher.finalize())[..16].to_string()
    };
    let mut adjacency: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for (index, edge) in edge_list.iter().enumerate() {
        let (a, b) = (&edge.ends[0].host, &edge.ends[1].host);
        adjacency.entry(a.clone()).or_default().insert(b.clone(), index);
        adjacency.entry(b.clone()).or_default().insert(a.clone(), index);
    }

    let mut actions: BTreeMap<String, Vec<RouteAction>> = BTreeMap::new();
    let mut guards: BTreeMap<String, BTreeMap<u32, (String, String)>> = BTreeMap::new();
    let mut prefix_guards: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut transit_used = BTreeSet::new();
    let mut conflicts = Vec::new();
    // One action per (kind, interface, target): the per-pair note differs when a transit
    // host routes the same destination both for itself and for an upstream source. A second
    // next hop for the same destination on the same host is a conflict, never a duplicate.
    // Routes are only ever added, so a route a committed pair relies on is never replaced.
    let push = |actions: &mut BTreeMap<String, Vec<RouteAction>>,
                conflicts: &mut Vec<String>,
                host: &str,
                action: RouteAction| {
        let list = actions.entry(host.to_string()).or_default();
        if list.iter().any(|existing| {
            existing.kind == action.kind
                && existing.interface_index == action.interface_index
                && existing.target == action.target
        }) {
            return;
        }
        if let Some(existing) = conflicting_route(list, &action) {
            conflicts.push(format!(
                "{host}: keeping route {} on ifIndex {}; dropped conflicting {} on ifIndex {}",
                existing.target, existing.interface_index, action.target, action.interface_index
            ));
            return;
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
            let address_present = found(
                &format!("Get-NetIPAddress -InterfaceIndex {index} -IPAddress {ip}"),
                " | Where-Object PrefixLength -eq 30",
            );
            if present {
                // The observation records addresses without prefix lengths; an address the plan
                // relies on but does not own must really be the /30 before anything is applied.
                prefix_guards.entry(end.host.clone()).or_default().push(format!(
                    "$p = Get-NetIPAddress -InterfaceIndex {index} -IPAddress {ip} -ErrorAction Stop; if ($p.PrefixLength -ne 30) {{ throw 'ifIndex {index} holds {ip}/' + $p.PrefixLength + ', not the planned /30; review before applying' }}"
                ));
            }
            push(
                &mut actions,
                &mut conflicts,
                &end.host,
                RouteAction {
                    kind: RouteActionKind::EdgeAddress,
                    interface_index: index,
                    target: format!("{ip}/30"),
                    already_present: present,
                    apply: (!present).then(|| {
                        let key = format!("EdgeAddress|{index}|{ip}/30");
                        // New-NetIPAddress disables DHCP on a DHCP-enabled interface (Microsoft Learn), and
                        // removing the address would not restore it, so such an interface is refused.
                        format!(
                            "if ((Get-NetIPInterface -InterfaceIndex {index} -AddressFamily IPv4 -ErrorAction Stop).Dhcp -eq 'Enabled') {{ throw 'DHCP is enabled on ifIndex {index}; a static address would disable it' }}; {}",
                            owned_apply(
                                &plan_id,
                                &clock_guard,
                                &key,
                                &address_present,
                                &format!("New-NetIPAddress -InterfaceIndex {index} -AddressFamily IPv4 -IPAddress {ip} -PrefixLength 30"),
                            )
                        )
                    }),
                    rollback: (!present).then(|| {
                        owned_rollback(
                            &plan_id,
                            &format!("EdgeAddress|{index}|{ip}/30"),
                            &adapter_matches(index, &end.mac, &end.interface_guid),
                            &address_present,
                            &format!("Remove-NetIPAddress -InterfaceIndex {index} -IPAddress {ip} -Confirm:$false"),
                        )
                    }),
                    note: Some(if present {
                        "Observed on this interface; not owned by this plan".to_string()
                    } else {
                        "Apply refuses a DHCP-enabled interface; rollback removes the address only if this plan's apply created it (journal)".to_string()
                    }),
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
            // The pair's actions are staged and committed together, so a pair that is blocked
            // leaves no routes or forwarding behind. Each entry records whether the route
            // leaves a leg's first host, which sends from a fixed address.
            let mut staged: Vec<(String, RouteAction, bool)> = Vec::new();
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
                    let route_present = found(
                        &format!(
                            "Get-NetRoute -InterfaceIndex {index} -DestinationPrefix {target_ip}/32 -NextHop {gateway}"
                        ),
                        "",
                    );
                    staged.push((
                        node.clone(),
                        RouteAction {
                            kind: RouteActionKind::HostRoute,
                            interface_index: index,
                            target: format!("{target_ip}/32 via {gateway}"),
                            already_present: false,
                            apply: Some(owned_apply(
                                &plan_id,
                                &clock_guard,
                                &format!("HostRoute|{index}|{target_ip}/32 via {gateway}"),
                                &route_present,
                                &format!("New-NetRoute -InterfaceIndex {index} -AddressFamily IPv4 -DestinationPrefix {target_ip}/32 -NextHop {gateway} -RouteMetric {metric}"),
                            )),
                            rollback: Some(owned_rollback(
                                &plan_id,
                                &format!("HostRoute|{index}|{target_ip}/32 via {gateway}"),
                                &adapter_matches(index, &out.mac, &out.interface_guid),
                                &route_present,
                                &format!("Remove-NetRoute -InterfaceIndex {index} -DestinationPrefix {target_ip}/32 -NextHop {gateway} -Confirm:$false"),
                            )),
                            note: Some(note.clone()),
                        },
                        i == 0,
                    ));
                }
            }
            // Windows uses the strong host model by default: a host's own packets, including its
            // replies, leave only through the interface that owns their source address. A leg's
            // first host already routing the target through another interface would keep that
            // route and send the packet nowhere, so the pair is blocked instead of Routed.
            // Transit hosts forward on any route, so a different next hop there is only dropped.
            let endpoint_conflicts: Vec<String> = staged
                .iter()
                .filter(|(_, _, origin)| *origin)
                .filter_map(|(node, action, _)| {
                    let existing = conflicting_route(actions.get(node)?, action)?;
                    Some(format!(
                        "{node} already routes {} on ifIndex {}, but sends from the source address on ifIndex {} ({})",
                        existing.target, existing.interface_index, action.interface_index, action.target
                    ))
                })
                .collect();
            if !endpoint_conflicts.is_empty() {
                plan.paths.push(PlannedPath {
                    from: from.clone(),
                    to: to.clone(),
                    status: PathStatus::Blocked,
                    hops: vec![],
                    destination_ipv4: None,
                    reason: Some(format!(
                        "Strong host routing: a route would not leave through the interface of its source address: {}",
                        endpoint_conflicts.join("; ")
                    )),
                });
                continue;
            }
            for (node, action, _) in staged {
                push(&mut actions, &mut conflicts, &node, action);
            }
            for (i, node) in path.iter().enumerate().take(path.len() - 1).skip(1) {
                transit_used.insert(node.clone());
                for edge in [hop_edge(i - 1), hop_edge(i)] {
                    if let Some(end) = edge.end(node) {
                        let index = end.interface_index;
                        let forwarding_enabled = format!(
                            "(Get-NetIPInterface -InterfaceIndex {index} -AddressFamily IPv4 -ErrorAction Stop).Forwarding -eq 'Enabled'"
                        );
                        push(
                            &mut actions,
                            &mut conflicts,
                            node,
                            RouteAction {
                                kind: RouteActionKind::Forwarding,
                                interface_index: index,
                                target: format!("IPv4 forwarding on ifIndex {index}"),
                                already_present: false,
                                apply: Some(owned_apply(
                                    &plan_id,
                                    &clock_guard,
                                    &format!("Forwarding|{index}"),
                                    &forwarding_enabled,
                                    &format!("Set-NetIPInterface -InterfaceIndex {index} -AddressFamily IPv4 -Forwarding Enabled"),
                                )),
                                rollback: Some(owned_rollback(
                                    &plan_id,
                                    &format!("Forwarding|{index}"),
                                    &adapter_matches(index, &end.mac, &end.interface_guid),
                                    &forwarding_enabled,
                                    &format!("Set-NetIPInterface -InterfaceIndex {index} -AddressFamily IPv4 -Forwarding Disabled"),
                                )),
                                note: Some(
                                    "Rollback disables forwarding only if this plan's apply enabled it (journal)".into(),
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
    let apply_prefix = format!("{clock_guard}; ");
    for (host, mut list) in actions {
        list.sort_by_key(|a| (a.kind, a.interface_index, a.target.clone()));
        // Check time before even an action-specific preflight, such as the DHCP query.
        for action in &mut list {
            if let Some(apply) = &mut action.apply {
                apply.insert_str(0, &apply_prefix);
            }
        }
        let host_guards = guards.remove(&host).unwrap_or_default();
        let preflight = host_guards
            .keys()
            .flat_map(|index| {
                [
                    format!("Get-NetIPInterface -InterfaceIndex {index} -AddressFamily IPv4 | Select-Object InterfaceIndex, Dhcp, Forwarding, InterfaceMetric, NlMtu"),
                    format!("Get-NetIPAddress -InterfaceIndex {index} -AddressFamily IPv4 | Select-Object InterfaceIndex, IPAddress, PrefixLength, PrefixOrigin"),
                    format!("Get-NetRoute -InterfaceIndex {index} -AddressFamily IPv4 | Select-Object InterfaceIndex, DestinationPrefix, NextHop, RouteMetric"),
                ]
            })
            .collect();
        plan.hosts.push(HostRoutePlan {
            transit: transit_used.contains(&host),
            guard: guard_lines(&host_guards)
                .into_iter()
                .chain(prefix_guards.remove(&host).unwrap_or_default())
                .collect(),
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
        assert!(nuc
            .guard
            .iter()
            .any(|g| g.contains("-IPAddress 172.31.255.22 ") && g.contains("PrefixLength -ne 30")));
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
            .contains("Remove-NetIPAddress -InterfaceIndex 28 "));
        assert!(missing
            .apply
            .as_deref()
            .unwrap()
            .contains(".Dhcp -eq 'Enabled') { throw"));
    }

    // Protects: the order that makes the journal trustworthy in every owned command. Detects:
    // reading or writing the journal before its storage is checked, a change made before its key
    // is journaled, and a rollback that undoes state without retiring the key or removes state
    // that is no longer present. Needs: the recorded hub chain. Breadcrumb: journal_guard.
    #[test]
    fn owned_commands_check_storage_journal_first_and_retire_keys() {
        let plan = build_route_plan(&observation(), &policy(&["dtm-super-nuc"]), &fleet(), now()).unwrap();
        let owned: Vec<&RouteAction> = plan
            .hosts
            .iter()
            .flat_map(|h| h.actions.iter())
            .filter(|a| a.apply.is_some())
            .collect();
        assert!(owned.iter().any(|a| a.kind == RouteActionKind::HostRoute));
        assert!(owned.iter().any(|a| a.kind == RouteActionKind::Forwarding));
        for action in owned {
            let apply = action.apply.as_deref().unwrap();
            let rollback = action.rollback.as_deref().unwrap();
            for line in [apply, rollback] {
                let check = line.find("GetOwner(").expect("owner check");
                assert!(line.find("ReparsePoint").unwrap() < check);
                assert!(check < line.find("Get-Content -LiteralPath $j").unwrap(), "{line}");
                assert!(line.contains("S-1-5-32-544', 'S-1-5-18'"));
            }
            let journaled = apply.find("Add-Content -LiteralPath $j").unwrap();
            let change = ["New-NetIPAddress", "New-NetRoute", "-Forwarding Enabled"]
                .iter()
                .filter_map(|c| apply.find(c))
                .min()
                .unwrap();
            assert!(journaled < change, "change before journal: {apply}");
            assert!(apply.contains("SetAccessRuleProtection($true, $false)"));
            assert!(
                !rollback.contains("New-Item"),
                "rollback must create nothing: {rollback}"
            );
            let undo = ["Remove-NetIPAddress", "Remove-NetRoute", "-Forwarding Disabled"]
                .iter()
                .filter_map(|c| rollback.find(c))
                .min()
                .unwrap();
            let retire = rollback.find("Set-Content -LiteralPath $j").unwrap();
            assert!(undo < retire, "key retired before undo: {rollback}");
            let identity = rollback
                .find(&format!("Get-NetAdapter -InterfaceIndex {} ", action.interface_index))
                .expect("rollback checks the adapter identity");
            assert!(identity < undo && rollback[identity..undo].contains("MacAddress -eq '"));
            assert!(rollback[identity..undo].contains("[guid]$a.InterfaceGuid -eq [guid]'"));
            // The presence check between the identity check and the undo must fail loudly: only
            // ObjectNotFound counts as absence for address and route queries, and the forwarding
            // query stops on any error.
            let presence = &rollback[identity..undo];
            assert!(presence.contains("Get-Net"), "{rollback}");
            assert!(
                presence.contains("$_.CategoryInfo.Category -ne 'ObjectNotFound'")
                    || presence.contains("-AddressFamily IPv4 -ErrorAction Stop).Forwarding"),
                "a failed state query would read as absence: {rollback}"
            );
            assert!(!presence.contains("-ErrorAction SilentlyContinue |") || presence.contains("-ErrorVariable e"));
        }
    }

    // Protects: every pasted apply command is valid only within its planner window.
    // Detects: an unguarded action kind, an expired command reaching network mutation after a
    // slow state query, or expiry blocking journal-owned rollback. Needs: the recorded hub chain.
    // Breadcrumb: build_route_plan and owned_apply.
    #[test]
    fn rendered_apply_commands_enforce_plan_window_and_leave_rollback_available() {
        let plan = build_route_plan(&observation(), &policy(&["dtm-super-nuc"]), &fleet(), now()).unwrap();
        let mut guarded_kinds = BTreeSet::new();
        for action in plan.hosts.iter().flat_map(|host| &host.actions) {
            let Some(apply) = action.apply.as_deref() else { continue };
            guarded_kinds.insert(action.kind);
            let clock = "$pcaiPlanNow = [DateTimeOffset]::UtcNow; if ($pcaiPlanNow";
            assert!(
                apply.starts_with(clock),
                "action reaches other cmdlets before checking validity: {apply}"
            );
            let journal_write = apply.find("Add-Content -LiteralPath $j").expect("write-ahead key");
            let mutation_guard = apply
                .find(&format!("try {{ {clock}"))
                .expect("validity checked again before mutation");
            assert!(journal_write < mutation_guard, "{apply}");
            assert!(apply.contains(&format!("-lt [DateTimeOffset]'{}'", plan.evaluated_at_utc.to_rfc3339())));
            assert!(apply.contains(&format!("-ge [DateTimeOffset]'{}'", plan.expires_at_utc.to_rfc3339())));
            let rollback = action.rollback.as_deref().expect("owned action has rollback");
            assert!(
                !rollback.contains(clock),
                "expiry must not prevent rollback: {rollback}"
            );
            assert!(rollback.contains("Get-Content -LiteralPath $j"));
        }
        assert_eq!(
            guarded_kinds,
            BTreeSet::from([
                RouteActionKind::EdgeAddress,
                RouteActionKind::HostRoute,
                RouteActionKind::Forwarding
            ])
        );
        assert!(!plan.apply_supported);
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

    /// Follows on-link /30s and planned host routes from `start`, sending from `source`, toward
    /// `target`. Windows defaults to the strong host model, so the first hop must leave through
    /// the interface that owns `source`; a route or on-link /30 on another interface of `start`
    /// does not carry the packet. Transit hosts forward on any route.
    fn delivers(plan: &RoutePlan, start: &str, source: Ipv4Addr, target: Ipv4Addr) -> bool {
        let ends: Vec<&PlannedEdgeEnd> = plan.edges.iter().flat_map(|e| e.endpoints.iter()).collect();
        let owner = |ip: Ipv4Addr| {
            ends.iter()
                .find(|e| e.ipv4_address == ip.to_string())
                .map(|e| e.host.clone())
        };
        let source_index = ends
            .iter()
            .find(|e| e.host == start && e.ipv4_address == source.to_string())
            .map(|e| e.interface_index)
            .expect("source address belongs to the start host");
        let mut at = start.to_string();
        for hop in 0..16 {
            if owner(target).as_deref() == Some(at.as_str()) {
                return true;
            }
            let on_link = ends
                .iter()
                .find(|e| e.host == at && same_slash30(e.ipv4_address.parse().unwrap(), target));
            if let Some(on_link) = on_link {
                if hop == 0 && on_link.interface_index != source_index {
                    return false;
                }
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
            if hop == 0 && route.interface_index != source_index {
                return false;
            }
            let gateway: Ipv4Addr = route.target.rsplit(' ').next().unwrap().parse().unwrap();
            at = owner(gateway).unwrap();
        }
        false
    }

    fn assert_every_routed_pair_round_trips(plan: &RoutePlan) {
        for path in plan.paths.iter().filter(|p| p.status != PathStatus::Blocked) {
            let destination: Ipv4Addr = path.destination_ipv4.as_deref().unwrap().parse().unwrap();
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
                delivers(plan, &path.from, source, destination),
                "{} cannot reach {} at {destination} from {source}",
                path.from,
                path.to
            );
            // The destination answers from the address the request reached.
            assert!(
                delivers(plan, &path.to, destination, source),
                "{} cannot reply from {destination} to {} at {source}",
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

    // Protects: a pair is reported Routed only when the destination's reply can leave through
    // the interface that owns the address it answers from (Windows strong host model). Detects:
    // a reply route dropped as a conflict because the host already routes the source address
    // through another interface for its own traffic, leaving the path Routed but one-way (codex
    // review of #335 at 5e1346a7: e-b-a-g-c forward while c keeps its c-f route to e). Needs:
    // the reviewer's seven-host mesh with five transit hosts. Breadcrumb: origin conflicts.
    #[test]
    fn a_path_whose_reply_route_conflicts_at_the_endpoint_is_blocked() {
        let links = [
            ("a", "b"),
            ("a", "g"),
            ("b", "d"),
            ("b", "e"),
            ("c", "f"),
            ("c", "g"),
            ("d", "f"),
        ];
        let known: BTreeSet<String> = ["a", "b", "c", "d", "e", "f", "g"]
            .into_iter()
            .map(String::from)
            .collect();
        let plan = build_route_plan(
            &mesh_observation(&links),
            &policy(&["a", "b", "d", "f", "g"]),
            &known,
            now(),
        )
        .unwrap();
        assert_every_routed_pair_round_trips(&plan);
        let blocked: Vec<&PlannedPath> = plan
            .paths
            .iter()
            .filter(|p| p.reason.as_deref().is_some_and(|r| r.contains("source address")))
            .collect();
        assert!(!blocked.is_empty(), "the endpoint conflict must block a path");
        for path in &blocked {
            assert_eq!(path.status, PathStatus::Blocked);
            assert!(path.hops.is_empty() && path.destination_ipv4.is_none());
        }
        // A blocked pair leaves nothing behind: no route carries its forward or reply note.
        for path in &blocked {
            let notes = [
                format!("{} -> {}", path.from, path.to),
                format!("{} -> {} (reply)", path.to, path.from),
            ];
            for host in &plan.hosts {
                assert!(
                    !host
                        .actions
                        .iter()
                        .any(|a| a.note.as_ref().is_some_and(|n| notes.contains(n))),
                    "blocked pair {} -> {} left an action on {}",
                    path.from,
                    path.to,
                    host.host
                );
            }
        }
        // Without the conflict the same mesh still routes most pairs.
        let routed = plan.paths.iter().filter(|p| p.status == PathStatus::Routed).count();
        assert!(routed > blocked.len(), "{routed} routed, {} blocked", blocked.len());
    }
}
