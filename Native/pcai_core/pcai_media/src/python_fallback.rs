//! Python fallback for Janus-Pro image understanding.
//!
//! When the native Janus understanding path is unavailable because the
//! published Janus checkpoint cannot be served by the native Candle loader,
//! this module shells out to a small Python helper that uses the reference
//! Janus implementation.

use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

const PYTHON_EXECUTABLE_ENV: &str = "PCAI_MEDIA_PYTHON_EXECUTABLE";
const PYTHON_SCRIPT_ENV: &str = "PCAI_MEDIA_PYTHON_FALLBACK_SCRIPT";
const PYTHON_FALLBACK_DISABLE_ENV: &str = "PCAI_MEDIA_DISABLE_PYTHON_FALLBACK";

#[derive(Debug, Serialize)]
struct PythonFallbackRequest<'a> {
    model: &'a str,
    image_path: &'a str,
    prompt: &'a str,
    max_tokens: u32,
    temperature: f32,
    device: &'a str,
}

#[derive(Debug, Deserialize)]
struct PythonFallbackResponse {
    text: String,
}

fn parse_python_response(stdout: &[u8]) -> Result<PythonFallbackResponse> {
    if let Ok(response) = serde_json::from_slice::<PythonFallbackResponse>(stdout) {
        return Ok(response);
    }

    let stdout_text = String::from_utf8_lossy(stdout);
    let candidate = stdout_text
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .map(str::trim)
        .context("python fallback produced no stdout")?;

    serde_json::from_str(candidate)
        .with_context(|| format!("parse python fallback JSON response from stdout tail: {candidate}"))
}

fn default_script_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("Tools")
        .join("janus-understand.py")
}

fn python_command() -> (String, Vec<String>) {
    if let Ok(executable) = env::var(PYTHON_EXECUTABLE_ENV) {
        return (executable, Vec::new());
    }

    if cfg!(windows) {
        return ("python".to_string(), Vec::new());
    }

    ("python3".to_string(), Vec::new())
}

fn python_script_path() -> Result<PathBuf> {
    let path = env::var_os(PYTHON_SCRIPT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(default_script_path);
    if !path.is_file() {
        bail!(
            "python Janus fallback script not found at '{}'; set {} to an existing helper path",
            path.display(),
            PYTHON_SCRIPT_ENV
        );
    }
    Ok(path)
}

pub fn python_fallback_enabled() -> bool {
    !matches!(
        env::var(PYTHON_FALLBACK_DISABLE_ENV).ok().as_deref(),
        Some("1" | "true" | "TRUE" | "True" | "yes" | "YES" | "Yes")
    )
}

pub fn understand_image(
    model: &str,
    device: &str,
    image_path: &Path,
    prompt: &str,
    max_tokens: u32,
    temperature: f32,
) -> Result<String> {
    let image_path = image_path
        .to_str()
        .with_context(|| format!("image path '{}' is not valid UTF-8", image_path.display()))?;
    let script_path = python_script_path()?;
    let (python, python_args) = python_command();
    let request = PythonFallbackRequest {
        model,
        image_path,
        prompt,
        max_tokens,
        temperature,
        device,
    };

    let mut command = Command::new(&python);
    command
        .args(&python_args)
        .arg(&script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn().with_context(|| {
        format!(
            "failed to start python Janus fallback using '{}' and '{}'",
            python,
            script_path.display()
        )
    })?;

    {
        let mut stdin = child.stdin.take().context("python fallback stdin unavailable")?;
        serde_json::to_writer(&mut stdin, &request).context("serialize python fallback request")?;
        stdin.flush().context("flush python fallback stdin")?;
    }

    let output = child.wait_with_output().context("wait for python fallback process")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let details = if !stderr.is_empty() {
            stderr
        } else if !stdout.is_empty() {
            stdout
        } else {
            format!("process exited with status {}", output.status)
        };
        bail!("python Janus fallback failed: {details}");
    }

    let response = parse_python_response(&output.stdout)?;
    Ok(response.text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn test_python_fallback_enabled_respects_disable_env() {
        let _guard = env_lock().lock().expect("lock env");
        env::remove_var(PYTHON_FALLBACK_DISABLE_ENV);
        assert!(python_fallback_enabled());

        env::set_var(PYTHON_FALLBACK_DISABLE_ENV, "1");
        assert!(!python_fallback_enabled());
        env::remove_var(PYTHON_FALLBACK_DISABLE_ENV);
    }

    #[test]
    fn test_python_command_defaults_to_py_launcher_on_windows() {
        let _guard = env_lock().lock().expect("lock env");
        env::remove_var(PYTHON_EXECUTABLE_ENV);

        let (python, args) = python_command();

        if cfg!(windows) {
            assert_eq!(python, "python");
            assert!(args.is_empty());
        } else {
            assert_eq!(python, "python3");
            assert!(args.is_empty());
        }
    }

    #[test]
    fn test_understand_image_parses_stubbed_python_response() {
        let _guard = env_lock().lock().expect("lock env");
        let dir = tempdir().expect("tempdir");
        let script_path = dir.path().join("stub.py");
        std::fs::write(
            &script_path,
            r#"import json, sys
request = json.load(sys.stdin)
print(json.dumps({"text": f"stub:{request['prompt']}"}))
"#,
        )
        .expect("write stub");

        env::set_var(PYTHON_EXECUTABLE_ENV, "python");
        env::set_var(PYTHON_SCRIPT_ENV, &script_path);

        let image_path = dir.path().join("test.png");
        std::fs::write(&image_path, b"png").expect("write image");

        let text = understand_image("model", "cpu", &image_path, "describe", 32, 0.7).expect("fallback call");
        assert_eq!(text, "stub:describe");

        env::remove_var(PYTHON_EXECUTABLE_ENV);
        env::remove_var(PYTHON_SCRIPT_ENV);
    }

    #[test]
    fn test_parse_python_response_uses_last_non_empty_line() {
        let stdout = b"noisy prelude\n{\"text\":\"ok\"}\n";
        let response = parse_python_response(stdout).expect("parse response");
        assert_eq!(response.text, "ok");
    }
}
