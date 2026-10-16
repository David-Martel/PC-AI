use anyhow::Result;
use pcai_tb::benchmark::run_architecture_benchmarks;
use pcai_tb::configure::{check_applied_actions, install_framework, register_local_thunderbolt_fabric};
use pcai_tb::handshake::discover_thunderbolt_peers;
use pcai_tb::identity::resolve_peer_identity;
use pcai_tb::platform::collect_platform_snapshot;
use std::env;
use std::path::PathBuf;

fn print_usage() {
    println!(
        "pcai-tb (USB4 / Thunderbolt 4 / Thunderbolt 5 P2P Engine)
Target: {}-{}

USAGE:
  pcai-tb <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
  status      Inspect USB4/Thunderbolt controllers, P2P adapters, MTU, and neighbors via kernel FFI
  identity    Compute deterministic SHA-256 IPv6 LLA, ULA, and commutative pairwise /30 IPv4
  discover    Run async multiplexed peer discovery (NTLM Type-2 AV_PAIR + SSH banner + NDP/ARP)
  register    Configure local deterministic LLA/ULA, MTU=62000, and per-port /128 ULA host routes
  benchmark   Profile NativeKernelFfi vs AsyncMultiplexedHandshake vs LegacyPowerShellCim
  install     Install pcai-tb binary and thunderbolt-peers.json into user/repo paths

COMMON OPTIONS:
  --computer <name>          Target computer name (defaults to local hostname)
  --peer <name>              Peer computer name for commutative pairwise /30 IPv4 or route hint
  --registry <path>          Explicit path to thunderbolt-peers.json
  --timeout-ms <ms>          Per-socket handshake timeout in milliseconds (default: 220)
  --iterations <n>           Benchmark iteration count (default: 10)
  --mtu <bytes>              Target USB4 P2P MTU in bytes (default: 62000)
  --metric <n>               Target interface/route metric (default: 5)
  --apply                    Apply changes (without --apply, register and install run in dry-run mode)
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

    let subcommand = args[0].to_ascii_lowercase();
    let mut computer: Option<String> = None;
    let mut peer: Option<String> = None;
    let mut registry: Option<PathBuf> = None;
    let mut timeout_ms: u64 = 220;
    let mut iterations: usize = 10;
    let mut mtu_bytes: u32 = 62000;
    let mut metric: u32 = 5;
    let mut apply = false;
    let mut assign_pairwise_ipv4 = false;
    let mut include_disconnected = false;
    let mut include_unreachable = false;
    let mut skip_legacy = false;
    let mut compact = false;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--computer" | "-c" => {
                if i + 1 < args.len() {
                    computer = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--peer" | "-p" => {
                if i + 1 < args.len() {
                    peer = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--registry" | "-r" => {
                if i + 1 < args.len() {
                    registry = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--timeout-ms" | "-t" => {
                if i + 1 < args.len() {
                    timeout_ms = args[i + 1].parse().unwrap_or(220);
                    i += 1;
                }
            }
            "--iterations" | "-n" => {
                if i + 1 < args.len() {
                    iterations = args[i + 1].parse().unwrap_or(10);
                    i += 1;
                }
            }
            "--mtu" => {
                if i + 1 < args.len() {
                    mtu_bytes = args[i + 1].parse().unwrap_or(62000);
                    i += 1;
                }
            }
            "--metric" => {
                if i + 1 < args.len() {
                    metric = args[i + 1].parse().unwrap_or(5);
                    i += 1;
                }
            }
            "--apply" => apply = true,
            "--assign-pairwise-ipv4" => assign_pairwise_ipv4 = true,
            "--include-disconnected" => include_disconnected = true,
            "--include-unreachable" => include_unreachable = true,
            "--skip-legacy" => skip_legacy = true,
            "--compact" => compact = true,
            _ => {}
        }
        i += 1;
    }

    if registry.is_some() {
        pcai_tb::identity::load_peer_registry(registry.as_deref())?;
    }
    match subcommand.as_str() {
        "status" => {
            let snap = collect_platform_snapshot(include_disconnected)?;
            emit_json(&snap, compact)?;
        }
        "identity" => {
            let id = resolve_peer_identity(computer.as_deref(), peer.as_deref(), registry.as_deref())?;
            emit_json(&id, compact)?;
        }
        "discover" => {
            let peers = discover_thunderbolt_peers(
                include_disconnected,
                timeout_ms,
                include_unreachable,
                registry.as_deref(),
            )
            .await?;
            emit_json(&peers, compact)?;
        }
        "register" => {
            let report = register_local_thunderbolt_fabric(
                apply,
                mtu_bytes,
                metric,
                assign_pairwise_ipv4,
                peer.as_deref(),
                registry.as_deref(),
            )
            .await?;
            emit_json(&report, compact)?;
            if apply {
                check_applied_actions(&report.actions)?;
            }
        }
        "benchmark" => {
            let report = run_architecture_benchmarks(iterations, timeout_ms, !skip_legacy, registry.as_deref()).await?;
            emit_json(&report, compact)?;
        }
        "install" => {
            let report = install_framework(apply, registry.as_deref())?;
            emit_json(&report, compact)?;
        }
        other => {
            anyhow::bail!("Unknown subcommand '{other}'. Run 'pcai-tb --help' for usage.");
        }
    }

    Ok(())
}
