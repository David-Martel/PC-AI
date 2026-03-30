use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use mimalloc::MiMalloc;
use pcai_core_lib::performance::{disk, process};
use rayon::prelude::*;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Serialize)]
struct ProcessRow {
    #[serde(rename = "PID")]
    pid: u32,
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "CPU")]
    cpu: f64,
    #[serde(rename = "MemoryMB")]
    memory_mb: f64,
    #[serde(rename = "Threads")]
    threads: Option<u32>,
    #[serde(rename = "Handles")]
    handles: Option<u32>,
    #[serde(rename = "Owner")]
    owner: Option<String>,
    #[serde(rename = "Path")]
    path: Option<String>,
    #[serde(rename = "StartTime")]
    start_time: Option<String>,
    #[serde(rename = "Status")]
    status: String,
    #[serde(rename = "Tool")]
    tool: &'static str,
}

#[derive(Debug, Serialize)]
struct DiskRow {
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "SizeBytes")]
    size_bytes: i64,
    #[serde(rename = "SizeMB")]
    size_mb: f64,
    #[serde(rename = "SizeGB")]
    size_gb: f64,
    #[serde(rename = "SizeHuman")]
    size_human: String,
    #[serde(rename = "FileCount")]
    file_count: i64,
    #[serde(rename = "Tool")]
    tool: &'static str,
}

#[derive(Debug, Serialize)]
struct HashRow {
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Hash")]
    hash: Option<String>,
    #[serde(rename = "Algorithm")]
    algorithm: String,
    #[serde(rename = "SizeBytes")]
    size_bytes: i64,
    #[serde(rename = "SizeMB")]
    size_mb: f64,
    #[serde(rename = "Success")]
    success: bool,
    #[serde(rename = "Error")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct WorkerResponse {
    ok: bool,
    result: Option<Value>,
    error: Option<String>,
}

fn main() {
    let exit_code = match run() {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("{err:#}");
            1
        }
    };

    std::process::exit(exit_code);
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or_default();

    match command {
        "processes" => run_processes(&args[2..]),
        "disk" => run_disk(&args[2..]),
        "hash-list" => run_hash_list(&args[2..]),
        "preflight" => run_preflight(&args[2..]),
        "roofline" => run_roofline(&args[2..]),
        "worker" => run_worker(),
        _ => bail!("usage: pcai-perf <processes|disk|hash-list|preflight|roofline|worker> [options]"),
    }
}

fn run_processes(args: &[String]) -> Result<()> {
    let top = parse_usize_flag(args, "--top")?.unwrap_or(10);
    let sort_by = parse_string_flag(args, "--sort-by")
        .unwrap_or_else(|| "memory".to_string())
        .to_ascii_lowercase();

    let sort_key = match sort_by.as_str() {
        "cpu" | "memory" => sort_by.as_str(),
        "mem" => "memory",
        other => bail!("unsupported process sort key: {other}"),
    };

    let rows = collect_process_rows(top, sort_key);

    println!("{}", serde_json::to_string(&rows)?);
    Ok(())
}

fn run_disk(args: &[String]) -> Result<()> {
    let top = parse_usize_flag(args, "--top")?.unwrap_or(10);
    let path = parse_string_flag(args, "--path").ok_or_else(|| anyhow!("disk requires --path"))?;

    let rows = collect_disk_rows(&path, top)?;

    println!("{}", serde_json::to_string(&rows)?);
    Ok(())
}

fn run_hash_list(args: &[String]) -> Result<()> {
    let algorithm = parse_string_flag(args, "--algorithm").unwrap_or_else(|| "SHA256".to_string());
    let file_paths = collect_positional_paths(args);
    if file_paths.is_empty() {
        bail!("hash-list requires at least one file path")
    }

    let rows = collect_hash_rows(&file_paths, &algorithm)?;

    println!("{}", serde_json::to_string(&rows)?);
    Ok(())
}

fn run_preflight(args: &[String]) -> Result<()> {
    let model_path = parse_string_flag(args, "--model");
    let context_length = parse_usize_flag(args, "--ctx")?.unwrap_or(0) as u64;
    let required_mb = parse_usize_flag(args, "--required-mb")?.unwrap_or(0) as u64;

    let result = if let Some(path) = model_path {
        pcai_core_lib::preflight::check_readiness(&path, context_length)?
    } else {
        pcai_core_lib::preflight::check_vram_state(required_mb)?
    };

    println!("{}", serde_json::to_string(&result)?);

    // Exit code matches verdict: 0=go, 1=warn, 2=fail
    match result.verdict {
        pcai_core_lib::preflight::Verdict::Go => Ok(()),
        pcai_core_lib::preflight::Verdict::Warn => std::process::exit(1),
        pcai_core_lib::preflight::Verdict::Fail => std::process::exit(2),
    }
}

fn run_roofline(args: &[String]) -> Result<()> {
    use pcai_core_lib::gpu::roofline::{analyze_roofline, GpuSpecs};

    let model_params = parse_f64_flag(args, "--model-params")?;
    let quant_bits = parse_f64_flag(args, "--quant-bits")?;
    let actual_toks = parse_f64_flag(args, "--actual-toks")?;
    let context_length = parse_usize_flag(args, "--ctx")?.unwrap_or(4096) as u64;

    let gpus = pcai_core_lib::gpu::gpu_inventory().unwrap_or_default();

    // If no model params given, print specs for all detected GPUs.
    if model_params.is_none() {
        let mut specs_list: Vec<GpuSpecs> = Vec::new();
        for gpu in &gpus {
            if let Some(specs) = GpuSpecs::from_compute_capability(&gpu.compute_capability, &gpu.name) {
                specs_list.push(specs);
            }
        }
        if specs_list.is_empty() {
            bail!("no GPUs with known roofline specs detected; pass --model-params to analyse manually");
        }
        println!("{}", serde_json::to_string(&specs_list)?);
        return Ok(());
    }

    // Safety: model_params.is_none() returned early above on line 188.
    let Some(params_b) = model_params else {
        unreachable!()
    };
    let bits = quant_bits.unwrap_or(4.5);
    // Treat 0.0 as "no measurement" for the actual_toks parameter.
    let actual = actual_toks.filter(|&v| v > 0.0);

    let mut analyses: Vec<pcai_core_lib::gpu::roofline::RooflineAnalysis> = Vec::new();
    for gpu in &gpus {
        if let Some(specs) = GpuSpecs::from_compute_capability(&gpu.compute_capability, &gpu.name) {
            analyses.push(analyze_roofline(&specs, params_b, bits, context_length, actual));
        }
    }

    if analyses.is_empty() {
        bail!("no GPUs with known roofline specs detected");
    }

    println!("{}", serde_json::to_string(&analyses)?);
    Ok(())
}

fn run_worker() -> Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = stdin.lock();
    let mut writer = std::io::BufWriter::new(stdout.lock());
    let mut line = String::new();

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let response = match handle_worker_request(trimmed) {
            Ok(result) => WorkerResponse {
                ok: true,
                result: Some(result),
                error: None,
            },
            Err(err) => WorkerResponse {
                ok: false,
                result: None,
                error: Some(err.to_string()),
            },
        };

        serde_json::to_writer(&mut writer, &response)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }

    Ok(())
}

fn handle_worker_request(raw: &str) -> Result<Value> {
    let request: Value = serde_json::from_str(raw).context("parse worker request json")?;
    let command = request
        .get("command")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("worker request missing command"))?;

    match command {
        "hash-list" => {
            let algorithm = request.get("algorithm").and_then(Value::as_str).unwrap_or("SHA256");
            let paths = request
                .get("paths")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("hash-list request missing paths"))?;
            let file_paths: Vec<String> = paths
                .iter()
                .filter_map(|value| value.as_str().map(|text| text.to_string()))
                .collect();
            Ok(serde_json::to_value(collect_hash_rows(&file_paths, algorithm)?)?)
        }
        "processes" => {
            let top = request.get("top").and_then(Value::as_u64).unwrap_or(10) as usize;
            let sort_by = request.get("sort_by").and_then(Value::as_str).unwrap_or("memory");
            Ok(serde_json::to_value(collect_process_rows(top, sort_by))?)
        }
        "disk" => {
            let path = request
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("disk request missing path"))?;
            let top = request.get("top").and_then(Value::as_u64).unwrap_or(10) as usize;
            Ok(serde_json::to_value(collect_disk_rows(path, top)?)?)
        }
        "preflight" => {
            let model_path = request.get("model").and_then(Value::as_str);
            let ctx = request.get("ctx").and_then(Value::as_u64).unwrap_or(0);
            let required_mb = request.get("required_mb").and_then(Value::as_u64).unwrap_or(0);

            let result = if let Some(path) = model_path {
                pcai_core_lib::preflight::check_readiness(path, ctx)?
            } else {
                pcai_core_lib::preflight::check_vram_state(required_mb)?
            };

            Ok(serde_json::to_value(result)?)
        }
        "roofline" => {
            use pcai_core_lib::gpu::roofline::{analyze_roofline, GpuSpecs};

            let model_params = request
                .get("model_params")
                .and_then(Value::as_f64)
                .ok_or_else(|| anyhow!("roofline request missing model_params"))?;
            let quant_bits = request.get("quant_bits").and_then(Value::as_f64).unwrap_or(4.5);
            let ctx = request.get("ctx").and_then(Value::as_u64).unwrap_or(4096);
            let actual_toks = request.get("actual_toks").and_then(Value::as_f64).filter(|&v| v > 0.0);

            let gpus = pcai_core_lib::gpu::gpu_inventory().unwrap_or_default();
            let mut analyses = Vec::new();
            for gpu in &gpus {
                if let Some(specs) = GpuSpecs::from_compute_capability(&gpu.compute_capability, &gpu.name) {
                    analyses.push(analyze_roofline(&specs, model_params, quant_bits, ctx, actual_toks));
                }
            }
            Ok(serde_json::to_value(analyses)?)
        }
        other => bail!("unsupported worker command: {other}"),
    }
}

fn hash_single_file(path: &str, algorithm: &str) -> HashRow {
    let path_ref = Path::new(path);
    let name = path_ref
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());

    match compute_hash(path_ref, algorithm) {
        Ok((hash, size_bytes)) => HashRow {
            path: path.to_string(),
            name,
            hash: Some(hash),
            algorithm: algorithm.to_string(),
            size_bytes: size_bytes as i64,
            size_mb: ((size_bytes as f64 / (1024.0 * 1024.0)) * 100.0).round() / 100.0,
            success: true,
            error: None,
        },
        Err(err) => HashRow {
            path: path.to_string(),
            name,
            hash: None,
            algorithm: algorithm.to_string(),
            size_bytes: 0,
            size_mb: 0.0,
            success: false,
            error: Some(err.to_string()),
        },
    }
}

fn collect_process_rows(top: usize, sort_key: &str) -> Vec<ProcessRow> {
    let (_, processes) = process::get_top_processes(top, sort_key);
    processes
        .into_iter()
        .map(|entry| ProcessRow {
            pid: entry.pid,
            name: entry.name,
            cpu: (entry.cpu_usage as f64 * 100.0).round() / 100.0,
            memory_mb: ((entry.memory_bytes as f64 / (1024.0 * 1024.0)) * 100.0).round() / 100.0,
            threads: None,
            handles: None,
            owner: None,
            path: entry.exe_path,
            start_time: None,
            status: entry.status,
            tool: "pcai_rust",
        })
        .collect()
}

fn collect_disk_rows(path: &str, top: usize) -> Result<Vec<DiskRow>> {
    let (_, entries) = disk::get_disk_usage(path, top).with_context(|| format!("scan disk usage for {path}"))?;
    Ok(entries
        .into_iter()
        .map(|entry| DiskRow {
            path: entry.path,
            size_bytes: entry.size_bytes as i64,
            size_mb: ((entry.size_bytes as f64 / (1024.0 * 1024.0)) * 100.0).round() / 100.0,
            size_gb: ((entry.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) * 100.0).round() / 100.0,
            size_human: entry.size_formatted,
            file_count: entry.file_count as i64,
            tool: "pcai_rust",
        })
        .collect())
}

fn collect_hash_rows(file_paths: &[String], algorithm: &str) -> Result<Vec<HashRow>> {
    if !algorithm.eq_ignore_ascii_case("SHA256") {
        bail!("unsupported hash algorithm for pcai-perf hash-list: {algorithm}");
    }

    Ok(file_paths
        .par_iter()
        .map(|path| hash_single_file(path, algorithm))
        .collect())
}

fn compute_hash(path: &Path, algorithm: &str) -> Result<(String, u64)> {
    let metadata = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut buffer = [0u8; 65536];

    let hash = match algorithm.to_ascii_uppercase().as_str() {
        "SHA256" => hash_reader(&mut reader, &mut buffer, Sha256::new())?,
        other => bail!("unsupported hash algorithm for pcai-perf hash-list: {other}"),
    };

    Ok((hash, metadata.len()))
}

fn hash_reader<T>(reader: &mut BufReader<File>, buffer: &mut [u8; 65536], mut hasher: T) -> Result<String>
where
    T: Digest,
{
    loop {
        let bytes_read = reader.read(buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(hex::encode(hasher.finalize()))
}

fn collect_positional_paths(args: &[String]) -> Vec<String> {
    let mut paths = Vec::new();
    let mut skip_next = false;

    for arg in args {
        if skip_next {
            skip_next = false;
            continue;
        }

        if arg == "--algorithm" {
            skip_next = true;
            continue;
        }

        if arg.starts_with("--") {
            continue;
        }

        paths.push(arg.clone());
    }

    paths
}

fn parse_string_flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|window| window[0] == name)
        .map(|window| window[1].clone())
}

fn parse_usize_flag(args: &[String], name: &str) -> Result<Option<usize>> {
    match parse_string_flag(args, name) {
        Some(value) => Ok(Some(
            value
                .parse::<usize>()
                .with_context(|| format!("parse {name} as usize"))?,
        )),
        None => Ok(None),
    }
}

fn parse_f64_flag(args: &[String], name: &str) -> Result<Option<f64>> {
    match parse_string_flag(args, name) {
        Some(value) => Ok(Some(
            value.parse::<f64>().with_context(|| format!("parse {name} as f64"))?,
        )),
        None => Ok(None),
    }
}
