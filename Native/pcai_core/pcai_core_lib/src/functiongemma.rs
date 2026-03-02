//! FunctionGemma dataset utilities exposed via FFI.

use std::os::raw::c_char;
use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;

use rust_functiongemma_train::router_dataset::{
    build_router_dataset, build_tool_test_vectors, write_jsonl, write_test_vectors, RouterDatasetConfig,
};

use crate::string::{c_str_to_rust, json_to_buffer, PcaiStringBuffer};
use crate::PcaiStatus;

#[derive(Debug, Serialize)]
struct RouterDatasetReport {
    status: String,
    output_jsonl: String,
    test_vectors: Option<String>,
    items: u64,
    vectors: u64,
    elapsed_ms: u64,
    include_tool_coverage: bool,
    max_cases: u64,
}

fn parse_required_path(ptr: *const c_char) -> Result<PathBuf, PcaiStatus> {
    if ptr.is_null() {
        return Err(PcaiStatus::NullPointer);
    }
    let raw = unsafe { c_str_to_rust(ptr) }.ok_or(PcaiStatus::InvalidUtf8)?;
    if raw.is_empty() {
        return Err(PcaiStatus::InvalidArgument);
    }
    Ok(PathBuf::from(raw))
}

fn parse_optional_path(ptr: *const c_char) -> Result<Option<PathBuf>, PcaiStatus> {
    if ptr.is_null() {
        return Ok(None);
    }
    let raw = unsafe { c_str_to_rust(ptr) }.ok_or(PcaiStatus::InvalidUtf8)?;
    if raw.is_empty() {
        return Ok(None);
    }
    Ok(Some(PathBuf::from(raw)))
}

fn map_anyhow(err: &anyhow::Error) -> PcaiStatus {
    if let Some(io_err) = err.downcast_ref::<std::io::Error>() {
        return PcaiStatus::from_io_error(io_err);
    }
    if err.downcast_ref::<serde_json::Error>().is_some() {
        return PcaiStatus::JsonError;
    }
    PcaiStatus::InternalError
}

#[no_mangle]
pub extern "C" fn pcai_build_router_dataset_jsonl(
    tools_path: *const c_char,
    scenarios_path: *const c_char,
    output_jsonl: *const c_char,
    output_vectors: *const c_char,
    diagnose_prompt: *const c_char,
    chat_prompt: *const c_char,
    max_cases: u32,
    include_tool_coverage: bool,
) -> PcaiStringBuffer {
    let start = Instant::now();

    let tools_path = match parse_required_path(tools_path) {
        Ok(value) => value,
        Err(status) => return PcaiStringBuffer::error(status),
    };
    let output_jsonl = match parse_required_path(output_jsonl) {
        Ok(value) => value,
        Err(status) => return PcaiStringBuffer::error(status),
    };
    let diagnose_prompt = match parse_required_path(diagnose_prompt) {
        Ok(value) => value,
        Err(status) => return PcaiStringBuffer::error(status),
    };
    let chat_prompt = match parse_required_path(chat_prompt) {
        Ok(value) => value,
        Err(status) => return PcaiStringBuffer::error(status),
    };

    let scenarios_path = match parse_optional_path(scenarios_path) {
        Ok(value) => value,
        Err(status) => return PcaiStringBuffer::error(status),
    };
    let output_vectors = match parse_optional_path(output_vectors) {
        Ok(value) => value,
        Err(status) => return PcaiStringBuffer::error(status),
    };

    let cfg = RouterDatasetConfig {
        output: output_jsonl.clone(),
        tools_path: tools_path.clone(),
        diagnose_prompt,
        chat_prompt,
        scenarios_path,
        include_tool_coverage,
        max_cases: max_cases as usize,
    };

    let items = match build_router_dataset(&cfg) {
        Ok(value) => value,
        Err(err) => return PcaiStringBuffer::error(map_anyhow(&err)),
    };

    if let Err(err) = write_jsonl(&cfg.output, &items) {
        return PcaiStringBuffer::error(map_anyhow(&err));
    }

    let mut vector_count = 0u64;
    if let Some(vectors_path) = output_vectors.as_ref() {
        let vectors = match build_tool_test_vectors(&tools_path, cfg.max_cases) {
            Ok(value) => value,
            Err(err) => return PcaiStringBuffer::error(map_anyhow(&err)),
        };
        if let Err(err) = write_test_vectors(vectors_path, &vectors) {
            return PcaiStringBuffer::error(map_anyhow(&err));
        }
        vector_count = vectors.len() as u64;
    }

    let report = RouterDatasetReport {
        status: "Success".to_string(),
        output_jsonl: output_jsonl.display().to_string(),
        test_vectors: output_vectors.as_ref().map(|p| p.display().to_string()),
        items: items.len() as u64,
        vectors: vector_count,
        elapsed_ms: start.elapsed().as_millis() as u64,
        include_tool_coverage,
        max_cases: max_cases as u64,
    };

    json_to_buffer(&report)
}
