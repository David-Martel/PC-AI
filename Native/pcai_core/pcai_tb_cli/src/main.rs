use anyhow::Result;
use pcai_tb::benchmark::run_architecture_benchmarks;
use pcai_tb::configure::{
    check_applied_actions, install_framework, register_local_thunderbolt_fabric, unregister_local_thunderbolt_fabric,
};
use pcai_tb::fabric_plan::{build_plan, FabricObservation, FabricPlan};
use pcai_tb::handshake::discover_thunderbolt_peers;
use pcai_tb::identity::{load_peer_registry, resolve_peer_identity};
use pcai_tb::lifecycle::run_usb4_lifecycle_evaluation;
use pcai_tb::platform::collect_platform_snapshot;
use pcai_tb::route_plan::{build_route_plan, RoutePolicy};
use pcai_tb::stream_preflight::observe_stream_preflight;
use pcai_tb::switch::{
    collect_switch_status, collect_switch_status_with_options, run_switch_observe, run_vnv_verification,
};
use std::env;
use std::path::PathBuf;

fn print_usage() {
    println!(
        "pcai-tb (USB4 / Thunderbolt 4 / Thunderbolt 5 & SFP+ Switch Engine)
Target: {}-{}

USAGE:
  pcai-tb <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
  status          Inspect USB4/Thunderbolt controllers, P2P adapters, MTU, and neighbors via kernel FFI
  identity        Compute deterministic SHA-256 IPv6 LLA, ULA, and commutative pairwise /30 IPv4
  discover        Run async multiplexed peer discovery (NTLM Type-2 AV_PAIR + SSH banner + NDP/ARP)
  register        Configure local deterministic LLA/ULA, MTU=62000, and per-port /128 ULA host routes
  unregister      Plan (or with --apply, remove) the addresses and /128 routes that register owns; MTU/metric untouched
  lifecycle       Evaluate USB4 lifecycle states (asleep/disconnected/probing/ready), TTL, standby 506/507, and stale route withdrawal
  topology-plan   Project fresh bilateral USB4 observations into separate segments; never changes networking
  stream-preflight Observe local Linux USB4STREAM surfaces without opening a stream device
  route-plan      Plan /30 addressing, host routes and transit forwarding across USB4 hops and hubs; emits commands, never applies
  switch          Inspect SFP+ switch custody, USB4 router trees, shared 10.* routes (--apply provisions routes)
  verify-custody  Enforce SFP+ switch, USB4 veth symmetry, and metric hierarchy invariants (exits non-zero on violation)
  observe         Capture non-payload switch/bridge/FDB/VLAN telemetry (--apply writes latest.json)
  vnv             Emit vigil_ai (DOC-036) V&V evidence (vigil.hil-device-evidence/v1 + vigil.ms01-usb4-check/v1)
  benchmark       Profile NativeKernelFfi vs AsyncMultiplexedHandshake vs LegacyPowerShellCim
  install         Install pcai-tb binary and thunderbolt-peers.json into user/repo paths

COMMON OPTIONS:
  --computer <name>          Target computer name (defaults to local hostname)
  --peer <name>              Peer computer name for commutative pairwise /30 IPv4 or route hint
  --adapter <GUID|MAC>       Select the USB4 adapter for --peer (register/unregister); needed on hub/multi-port hosts
  --registry <path>          Explicit path to thunderbolt-peers.json
  --output <path>            Output file path for V&V evidence artifact (used by vnv)
  --input <path>             Explicit observation JSON for topology-plan (maximum 1 MiB)
  --previous <path>          Previous topology plan for withdrawn-edge data only
  --policy <path>            Route policy for route-plan (schema pcai.usb4-route-policy/v1; names transit hosts)
  --timeout-ms <ms>          Per-socket handshake timeout in milliseconds (default: 220)
  --iterations <n>           Benchmark iteration count (default: 10)
  --mtu <bytes>              Target USB4 P2P MTU in bytes (default: 62000, range: 1280..=65535)
  --metric <n>               Target interface/route metric (default: 5, range: 1..=9999)
  --apply                    Apply changes (without --apply, register, unregister, switch, observe, and install run in dry-run mode)
  --assign-pairwise-ipv4     Also assign deterministic /30 IPv4 address during register
  --include-disconnected     Include disconnected/down USB4 P2P adapters
  --include-unreachable      Include unreachable candidate slots in discover output
  --skip-legacy              Skip the slow PowerShell CIM baseline during benchmark
  --compact                  Emit compact single-line JSON instead of pretty JSON",
        env::consts::OS,
        env::consts::ARCH
    );
}

fn emit_json<T: serde::Serialize>(val: &T, compact: bool) -> Result<()> {
    if compact {
        println!("{}", serde_json::to_string(val)?);
    } else {
        println!("{}", serde_json::to_string_pretty(val)?);
    }
    Ok(())
}

fn require_arg_value<'a>(args: &'a [String], idx: usize, flag: &str) -> Result<&'a str> {
    if idx + 1 >= args.len() {
        anyhow::bail!("Missing required value for option '{flag}'");
    }
    let val = args[idx + 1].trim();
    if val.is_empty() || val.starts_with("--") {
        anyhow::bail!("Missing or invalid value for option '{flag}': '{}'", args[idx + 1]);
    }
    Ok(val)
}

#[derive(Debug, Clone)]
pub struct ParsedCliArgs {
    pub subcommand: String,
    pub computer: Option<String>,
    pub peer: Option<String>,
    pub adapter: Option<String>,
    pub registry: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub input: Option<PathBuf>,
    pub previous: Option<PathBuf>,
    pub policy: Option<PathBuf>,
    pub timeout_ms: u64,
    pub iterations: usize,
    pub mtu_bytes: u32,
    pub metric: u32,
    pub apply: bool,
    pub assign_pairwise_ipv4: bool,
    pub include_disconnected: bool,
    pub include_unreachable: bool,
    pub skip_legacy: bool,
    pub compact: bool,
}

pub fn parse_cli_args(args: &[String]) -> Result<ParsedCliArgs> {
    if args.is_empty() {
        anyhow::bail!("No subcommand provided. Run 'pcai-tb --help' for usage.");
    }
    let subcommand = args[0].to_ascii_lowercase();
    let mut parsed = ParsedCliArgs {
        subcommand,
        computer: None,
        peer: None,
        adapter: None,
        registry: None,
        output: None,
        input: None,
        previous: None,
        policy: None,
        timeout_ms: 220,
        iterations: 10,
        mtu_bytes: 62000,
        metric: 5,
        apply: false,
        assign_pairwise_ipv4: false,
        include_disconnected: false,
        include_unreachable: false,
        skip_legacy: false,
        compact: false,
    };

    let mut i = 1usize;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--computer" | "-c" => {
                let val = require_arg_value(args, i, flag)?;
                parsed.computer = Some(val.to_string());
                i += 1;
            }
            "--peer" | "-p" => {
                let val = require_arg_value(args, i, flag)?;
                parsed.peer = Some(val.to_string());
                i += 1;
            }
            "--adapter" => {
                parsed.adapter = Some(require_arg_value(args, i, flag)?.to_string());
                i += 1;
            }
            "--registry" | "-r" => {
                let val = require_arg_value(args, i, flag)?;
                parsed.registry = Some(PathBuf::from(val));
                i += 1;
            }
            "--output" | "-o" => {
                let val = require_arg_value(args, i, flag)?;
                parsed.output = Some(PathBuf::from(val));
                i += 1;
            }
            "--input" => {
                parsed.input = Some(PathBuf::from(require_arg_value(args, i, flag)?));
                i += 1;
            }
            "--previous" => {
                parsed.previous = Some(PathBuf::from(require_arg_value(args, i, flag)?));
                i += 1;
            }
            "--policy" => {
                parsed.policy = Some(PathBuf::from(require_arg_value(args, i, flag)?));
                i += 1;
            }
            "--timeout-ms" | "-t" => {
                let val = require_arg_value(args, i, flag)?;
                let n: u64 = val
                    .parse()
                    .map_err(|e| anyhow::anyhow!("Invalid integer for '{flag}': '{val}' ({e})"))?;
                if n == 0 || n > 60_000 {
                    anyhow::bail!("Option '{flag}' must be between 1 and 60000 ms (got {n})");
                }
                parsed.timeout_ms = n;
                i += 1;
            }
            "--iterations" | "-n" => {
                let val = require_arg_value(args, i, flag)?;
                let n: usize = val
                    .parse()
                    .map_err(|e| anyhow::anyhow!("Invalid integer for '{flag}': '{val}' ({e})"))?;
                if n == 0 || n > 10_000 {
                    anyhow::bail!("Option '{flag}' must be between 1 and 10000 (got {n})");
                }
                parsed.iterations = n;
                i += 1;
            }
            "--mtu" => {
                let val = require_arg_value(args, i, flag)?;
                let n: u32 = val
                    .parse()
                    .map_err(|e| anyhow::anyhow!("Invalid integer for '{flag}': '{val}' ({e})"))?;
                if !(1280..=65535).contains(&n) {
                    anyhow::bail!("Option '{flag}' must be between 1280 and 65535 bytes (got {n})");
                }
                parsed.mtu_bytes = n;
                i += 1;
            }
            "--metric" => {
                let val = require_arg_value(args, i, flag)?;
                let n: u32 = val
                    .parse()
                    .map_err(|e| anyhow::anyhow!("Invalid integer for '{flag}': '{val}' ({e})"))?;
                if !(1..=9999).contains(&n) {
                    anyhow::bail!("Option '{flag}' must be between 1 and 9999 (got {n})");
                }
                parsed.metric = n;
                i += 1;
            }
            "--apply" => parsed.apply = true,
            "--assign-pairwise-ipv4" => parsed.assign_pairwise_ipv4 = true,
            "--include-disconnected" => parsed.include_disconnected = true,
            "--include-unreachable" => parsed.include_unreachable = true,
            "--skip-legacy" => parsed.skip_legacy = true,
            "--compact" => parsed.compact = true,
            other => {
                anyhow::bail!("Unknown option '{other}'. Run 'pcai-tb --help' for usage.");
            }
        }
        i += 1;
    }

    Ok(parsed)
}

/// Read a bounded explicit observation; never discovers or changes host state.
fn read_topology_json<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<T> {
    use std::io::Read;
    const LIMIT: u64 = 1024 * 1024;
    if !std::fs::metadata(path)?.is_file() {
        anyhow::bail!("Topology input must be a regular file");
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Avoid blocking if a regular path is replaced by a FIFO before open.
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        anyhow::bail!("Topology input must be a regular file");
    }
    let mut bytes = Vec::new();
    file.take(LIMIT + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > LIMIT {
        anyhow::bail!("Topology input exceeds the 1 MiB limit");
    }
    Ok(serde_json::from_slice(&bytes)?)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help" || a == "help") {
        print_usage();
        return Ok(());
    }
    if args.iter().any(|a| a == "-V" || a == "--version" || a == "version") {
        println!(
            "pcai-tb {} ({}-{})",
            env!("CARGO_PKG_VERSION"),
            env::consts::OS,
            env::consts::ARCH
        );
        return Ok(());
    }

    let parsed = parse_cli_args(&args)?;

    if parsed.subcommand == "stream-preflight" {
        if args[1..].iter().any(|arg| arg != "--compact") {
            anyhow::bail!("stream-preflight is read-only; only --compact is accepted");
        }
        return emit_json(&observe_stream_preflight(), parsed.compact);
    }

    if parsed.subcommand == "topology-plan" {
        let mut index = 1;
        while index < args.len() {
            match args[index].as_str() {
                "--input" | "--previous" => index += 2,
                "--compact" => index += 1,
                _ => anyhow::bail!(
                    "topology-plan option is unsupported; only --input, --previous and --compact are accepted"
                ),
            }
        }
        if parsed.apply
            || parsed.assign_pairwise_ipv4
            || parsed.computer.is_some()
            || parsed.peer.is_some()
            || parsed.registry.is_some()
            || parsed.output.is_some()
        {
            anyhow::bail!("topology-plan accepts explicit observations only; apply, registry, host overrides and output writes are unsupported");
        }
        let input = parsed
            .input
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("topology-plan requires --input"))?;
        let observation: FabricObservation = read_topology_json(input)?;
        let previous: Option<FabricPlan> = parsed.previous.as_deref().map(read_topology_json).transpose()?;
        let plan = build_plan(&observation, chrono::Utc::now(), previous.as_ref())?;
        emit_json(&plan, parsed.compact)?;
        return Ok(());
    }
    if parsed.subcommand == "route-plan" {
        let mut index = 1;
        while index < args.len() {
            match args[index].as_str() {
                "--input" | "--policy" | "--registry" => index += 2,
                "--compact" => index += 1,
                _ => anyhow::bail!(
                    "route-plan option is unsupported; only --input, --policy, --registry and --compact are accepted"
                ),
            }
        }
        let input = parsed
            .input
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("route-plan requires --input (a pcai.usb4-fabric-observation/v1 file)"))?;
        let policy_path = parsed
            .policy
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("route-plan requires --policy (a pcai.usb4-route-policy/v1 file)"))?;
        let observation: FabricObservation = read_topology_json(input)?;
        let policy: RoutePolicy = read_topology_json(policy_path)?;
        let known: std::collections::BTreeSet<String> =
            pcai_tb::identity::known_fleet_peer_names(parsed.registry.as_deref(), "")?
                .into_iter()
                .map(|name| name.to_ascii_lowercase())
                .collect();
        let plan = build_route_plan(&observation, &policy, &known, chrono::Utc::now())?;
        emit_json(&plan, parsed.compact)?;
        return Ok(());
    }
    if parsed.input.is_some() || parsed.previous.is_some() || parsed.policy.is_some() {
        anyhow::bail!("--input, --previous and --policy are only supported by topology-plan and route-plan");
    }

    match parsed.subcommand.as_str() {
        "status" | "identity" | "discover" | "register" | "unregister" | "lifecycle" | "switch" | "verify-custody"
        | "observe" | "vnv" | "benchmark" | "install" => {
            load_peer_registry(parsed.registry.as_deref())?;
        }
        other => {
            anyhow::bail!("Unknown subcommand '{other}'. Run 'pcai-tb --help' for usage.");
        }
    }

    if parsed.adapter.is_some() && !matches!(parsed.subcommand.as_str(), "register" | "unregister") {
        anyhow::bail!("--adapter is only supported by register and unregister");
    }

    match parsed.subcommand.as_str() {
        "status" => {
            let snap = collect_platform_snapshot(parsed.include_disconnected)?;
            emit_json(&snap, parsed.compact)?;
        }
        "identity" => {
            let id = resolve_peer_identity(
                parsed.computer.as_deref(),
                parsed.peer.as_deref(),
                parsed.registry.as_deref(),
            )?;
            emit_json(&id, parsed.compact)?;
        }
        "discover" => {
            let peers = discover_thunderbolt_peers(
                parsed.include_disconnected,
                parsed.timeout_ms,
                parsed.include_unreachable,
                parsed.registry.as_deref(),
            )
            .await?;
            emit_json(&peers, parsed.compact)?;
        }
        "register" => {
            if parsed.computer.is_some() {
                anyhow::bail!("register targets this machine; --computer is only supported for identity inspection");
            }
            let report = register_local_thunderbolt_fabric(
                parsed.apply,
                parsed.mtu_bytes,
                parsed.metric,
                parsed.assign_pairwise_ipv4,
                parsed.peer.as_deref(),
                parsed.adapter.as_deref(),
                parsed.registry.as_deref(),
            )
            .await?;
            emit_json(&report, parsed.compact)?;
            if parsed.apply {
                check_applied_actions(&report.actions)?;
            }
        }
        "unregister" => {
            if parsed.computer.is_some() || parsed.assign_pairwise_ipv4 {
                anyhow::bail!("unregister targets this machine and accepts only --peer, --adapter, --registry, --apply and --compact");
            }
            let report = unregister_local_thunderbolt_fabric(
                parsed.apply,
                parsed.peer.as_deref(),
                parsed.adapter.as_deref(),
                parsed.registry.as_deref(),
            )?;
            emit_json(&report, parsed.compact)?;
            if parsed.apply {
                check_applied_actions(&report.actions)?;
            }
        }
        "lifecycle" => {
            let report =
                run_usb4_lifecycle_evaluation(parsed.apply, parsed.timeout_ms, parsed.registry.as_deref()).await?;
            emit_json(&report, parsed.compact)?;
            if !report.overall_valid {
                anyhow::bail!("pcai-tb lifecycle failed validation checks");
            }
        }
        "switch" => {
            let report = collect_switch_status_with_options(parsed.registry.as_deref(), parsed.apply)?;
            emit_json(&report, parsed.compact)?;
            if parsed.apply && !report.custody_valid {
                anyhow::bail!("pcai-tb switch --apply failed: {}", report.errors.join("; "));
            }
        }
        "verify-custody" => {
            let report = collect_switch_status(parsed.registry.as_deref())?;
            emit_json(&report, parsed.compact)?;
            if !report.custody_valid {
                anyhow::bail!("pcai-tb verify-custody failed: {}", report.errors.join("; "));
            }
        }
        "observe" => {
            let snap = run_switch_observe(parsed.apply, parsed.registry.as_deref())?;
            emit_json(&snap, parsed.compact)?;
        }
        "vnv" => {
            let evidence = run_vnv_verification(parsed.registry.as_deref(), parsed.output.as_deref())?;
            emit_json(&evidence, parsed.compact)?;
            if !evidence.custody_valid {
                anyhow::bail!("pcai-tb vnv failed: overall_status={}", evidence.overall_status);
            }
        }
        "benchmark" => {
            let report = run_architecture_benchmarks(
                parsed.iterations,
                parsed.timeout_ms,
                !parsed.skip_legacy,
                parsed.registry.as_deref(),
            )
            .await?;
            emit_json(&report, parsed.compact)?;
        }
        "install" => {
            let report = install_framework(parsed.apply, parsed.registry.as_deref())?;
            emit_json(&report, parsed.compact)?;
            if parsed.apply && !report.overall_success {
                anyhow::bail!("pcai-tb install --apply failed: {}", report.errors.join("; "));
            }
        }
        other => {
            anyhow::bail!("Unknown subcommand '{other}'. Run 'pcai-tb --help' for usage.");
        }
    }

    Ok(())
}
