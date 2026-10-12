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
    #[serde(rename = "CPUUnit")]
    cpu_unit: &'static str,
    #[serde(rename = "CPUPercent")]
    cpu_percent: f64,
    #[serde(rename = "TotalProcessorTimeSeconds")]
    total_processor_time_seconds: Option<f64>,
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
    protocol: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    ok: bool,
    result: Option<Value>,
    error: Option<String>,
}

const WORKER_PROTOCOL: u32 = 1;
const MAX_FRAME_BYTES: usize = 1024 * 1024;
const MAX_RESULTS: usize = 10_000;
const MAX_HASH_PATHS: usize = 4096;

#[derive(Debug, Default)]
struct PerfWorkerState {
    process_sampler: process::ProcessSampler,
}

#[derive(Debug, Default)]
struct BoundedFrame(Vec<u8>);

impl Write for BoundedFrame {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_FRAME_BYTES.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other("worker result exceeds frame byte limit"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
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
    let Some(params_b) = model_params else { unreachable!() };
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
    serve_worker(&mut stdin.lock(), &mut std::io::BufWriter::new(stdout.lock()))
}

/// Reads complete, bounded NDJSON frames; incomplete or oversized frames terminate custody.
fn serve_worker(reader: &mut impl BufRead, writer: &mut impl Write) -> Result<()> {
    let mut state = PerfWorkerState::default();
    loop {
        let mut frame = Vec::new();
        // Taking a bounded reader prevents read_until allocating an unbounded line.
        let bytes = (&mut *reader)
            .take((MAX_FRAME_BYTES + 2) as u64)
            .read_until(b'\n', &mut frame)?;
        if bytes == 0 {
            return Ok(());
        }
        if frame.last() != Some(&b'\n') {
            bail!("worker frame is incomplete or exceeds byte limit");
        }
        frame.pop();
        if frame.last() == Some(&b'\r') {
            frame.pop();
        }
        if frame.len() > MAX_FRAME_BYTES {
            bail!("worker frame exceeds byte limit");
        }
        let text = std::str::from_utf8(&frame).context("worker frame is not UTF-8")?;
        let trimmed = text.trim_start_matches('\u{feff}').trim();
        if trimmed.is_empty() {
            continue;
        }
        let response = make_worker_response(trimmed, &mut state);
        let mut encoded = BoundedFrame::default();
        if serde_json::to_writer(&mut encoded, &response).is_err() {
            encoded.0.clear();
            serde_json::to_writer(
                &mut encoded,
                &WorkerResponse {
                    protocol: WORKER_PROTOCOL,
                    request_id: response.request_id,
                    ok: false,
                    result: None,
                    error: Some("worker result exceeds frame byte limit".into()),
                },
            )?;
        }
        writer.write_all(&encoded.0)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
}

fn make_worker_response(raw: &str, state: &mut PerfWorkerState) -> WorkerResponse {
    let parsed = serde_json::from_str::<Value>(raw).context("parse worker request json");
    let request_id = parsed
        .as_ref()
        .ok()
        .and_then(|request| request.get("request_id"))
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty() && id.len() <= 128)
        .map(str::to_owned);
    let result = parsed.and_then(|request| handle_worker_request(&request, state));
    match result {
        Ok(result) => WorkerResponse {
            protocol: WORKER_PROTOCOL,
            request_id,
            ok: true,
            result: Some(result),
            error: None,
        },
        Err(err) => WorkerResponse {
            protocol: WORKER_PROTOCOL,
            request_id,
            ok: false,
            result: None,
            error: Some(err.to_string()),
        },
    }
}

fn worker_top(request: &Value) -> Result<usize> {
    let top = match request.get("top") {
        Some(value) => value
            .as_u64()
            .ok_or_else(|| anyhow!("top must be a nonnegative integer"))?,
        None => 10,
    };
    if top > MAX_RESULTS as u64 {
        bail!("top exceeds worker result limit");
    }
    Ok(top as usize)
}
fn handle_worker_request(request: &Value, state: &mut PerfWorkerState) -> Result<Value> {
    if !request.is_object() {
        bail!("worker request must be an object");
    }
    if request
        .get("protocol")
        .is_some_and(|value| value.as_u64() != Some(u64::from(WORKER_PROTOCOL)))
    {
        bail!("unsupported worker protocol");
    }
    if let Some(id) = request.get("request_id") {
        if !id.as_str().is_some_and(|text| !text.is_empty() && text.len() <= 128) {
            bail!("request_id must be a string of 1 through 128 bytes");
        }
    }
    let command = request
        .get("command")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("worker request missing command"))?;

    match command {
        "hello" => Ok(serde_json::json!({
            "protocol": WORKER_PROTOCOL, "max_frame_bytes": MAX_FRAME_BYTES,
            "max_results": MAX_RESULTS, "capabilities": ["processes", "disk", "hash-list", "preflight", "roofline"]
        })),
        "hash-list" => {
            let algorithm = request.get("algorithm").and_then(Value::as_str).unwrap_or("SHA256");
            let paths = request
                .get("paths")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("hash-list request missing paths"))?;
            if paths.is_empty() || paths.len() > MAX_HASH_PATHS {
                bail!("paths count must be 1 through 4096");
            }
            let file_paths: Vec<String> = paths
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .filter(|text| !text.is_empty())
                        .map(str::to_owned)
                        .ok_or_else(|| anyhow!("each path must be a nonempty string"))
                })
                .collect::<Result<_>>()?;
            Ok(serde_json::to_value(collect_hash_rows(&file_paths, algorithm)?)?)
        }
        "processes" => {
            let top = worker_top(request)?;
            let sort_by = match request.get("sort_by") {
                Some(value) => value.as_str().ok_or_else(|| anyhow!("sort_by must be a string"))?,
                None => "memory",
            };
            let sort_key = match sort_by.to_ascii_lowercase().as_str() {
                "cpu" => "cpu",
                "mem" | "memory" => "memory",
                _ => bail!("unsupported process sort key"),
            };
            let (_, processes) = state.process_sampler.get_top_processes(top, sort_key);
            Ok(serde_json::to_value(process_rows(processes))?)
        }
        "disk" => {
            let path = request
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("disk request missing path"))?;
            let top = worker_top(request)?;
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
    process_rows(processes)
}

fn process_rows(processes: Vec<process::ProcessInfo>) -> Vec<ProcessRow> {
    processes
        .into_iter()
        .map(|entry| ProcessRow {
            pid: entry.pid,
            name: entry.name,
            cpu: (entry.cpu_usage as f64 * 100.0).round() / 100.0,
            cpu_unit: "percent",
            cpu_percent: (entry.cpu_usage as f64 * 100.0).round() / 100.0,
            total_processor_time_seconds: None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn exchange(input: &[u8]) -> Result<Vec<Value>> {
        let mut output = Vec::new();
        serve_worker(&mut Cursor::new(input), &mut output)?;
        String::from_utf8(output)?
            .lines()
            .map(|line| Ok(serde_json::from_str(line)?))
            .collect()
    }

    #[test]
    fn negotiation_and_errors_preserve_request_correlation() {
        let rows = exchange(b"{\"command\":\"hello\",\"protocol\":1,\"request_id\":\"one\"}\n{\"command\":\"unknown\",\"protocol\":1,\"request_id\":\"two\"}\n{\"command\":\"hello\"}\n").expect("complete frames should work");
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0]["request_id"], "one");
        assert_eq!(rows[0]["protocol"], WORKER_PROTOCOL);
        assert_eq!(rows[0]["result"]["protocol"], WORKER_PROTOCOL);
        assert_eq!(rows[1]["request_id"], "two");
        assert_eq!(rows[1]["ok"], false);
        assert_eq!(rows[2]["ok"], true);
        assert!(rows[2].get("request_id").is_none(), "legacy requests remain supported");
    }

    #[test]
    fn malformed_payloads_reject_instead_of_silently_dropping_values() {
        let mut state = PerfWorkerState::default();
        for request in [
            serde_json::json!({"command":"hash-list", "paths":["file", 7]}),
            serde_json::json!({"command":"hash-list", "paths":[]}),
            serde_json::json!({"command":"processes", "top":-1}),
            serde_json::json!({"command":"processes", "top":10_001}),
            serde_json::json!({"command":"processes", "sort_by":"unsupported"}),
            serde_json::json!({"command":"hello", "protocol":2}),
            serde_json::json!({"command":"hello", "request_id":7}),
            serde_json::json!(["hello"]),
        ] {
            assert!(
                handle_worker_request(&request, &mut state).is_err(),
                "must reject {request}"
            );
        }
    }

    #[test]
    fn framing_is_bounded_complete_utf8_and_recovers_only_at_valid_boundaries() {
        assert!(exchange(b"{\"command\":\"hello\"}").is_err());
        assert!(exchange(&vec![b'x'; MAX_FRAME_BYTES + 3]).is_err());
        assert!(exchange(b"\xff\n").is_err());
        let rows = exchange("\u{feff}{\"command\":\"hello\"}\r\n{invalid}\n{\"command\":\"hello\"}\n".as_bytes())
            .expect("bad complete JSON should yield a correlated error frame");
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0]["ok"], true);
        assert_eq!(rows[1]["ok"], false);
        assert_eq!(rows[2]["ok"], true);
    }

    #[test]
    fn oversized_results_preserve_correlation_without_growing_output_buffer() {
        // A legal frame can produce a larger JSON-escaped error response.
        let input = serde_json::json!({"command":"\\".repeat((MAX_FRAME_BYTES - 64) / 2), "request_id":"bounded"});
        let encoded = format!("{input}\n");
        assert!(encoded.len() < MAX_FRAME_BYTES);
        let rows = exchange(encoded.as_bytes()).expect("bounded response should remain a complete frame");
        assert_eq!(rows[0]["request_id"], "bounded");
        assert_eq!(rows[0]["error"], "worker result exceeds frame byte limit");
        assert_eq!(rows[0]["ok"], false);
        let mut frame = BoundedFrame::default();
        assert!(frame.write_all(&vec![b'x'; MAX_FRAME_BYTES]).is_ok());
        assert!(frame.write_all(b"x").is_err());
        assert_eq!(frame.0.len(), MAX_FRAME_BYTES);
    }

    #[test]
    fn persistent_process_requests_and_cpu_units_keep_public_schema() {
        let rows = exchange(b"{\"command\":\"processes\",\"top\":15,\"sort_by\":\"mem\",\"request_id\":\"first\"}\n{\"command\":\"processes\",\"top\":15,\"sort_by\":\"memory\",\"request_id\":\"second\"}\n").expect("process requests should succeed");
        for response in rows {
            assert_eq!(response["ok"], true);
            let processes = response["result"].as_array().expect("rows should remain arrays");
            assert!(!processes.is_empty());
            assert!(processes.len() <= 15);
            for row in processes {
                assert_eq!(row["CPUUnit"], "percent");
                assert_eq!(row["CPU"], row["CPUPercent"]);
                assert_eq!(row["TotalProcessorTimeSeconds"], Value::Null);
                assert_eq!(row["Tool"], "pcai_rust");
            }
            assert!(processes
                .windows(2)
                .all(|rows| rows[0]["MemoryMB"].as_f64() >= rows[1]["MemoryMB"].as_f64()));
        }
    }
}
