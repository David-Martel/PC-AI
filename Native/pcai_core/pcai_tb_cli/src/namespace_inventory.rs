//! Collect bounded observations from explicitly enrolled Linux network namespaces.
use crate::edge_policy::EdgePolicy;
#[cfg(any(target_os = "linux", test))]
use crate::switch::run_bounded_command_with_deadline;
use anyhow::Result;
#[cfg(any(target_os = "linux", test))]
use anyhow::{ensure, Context};
use serde_json::{json, Value};
use std::path::Path;
#[cfg(target_os = "linux")]
use std::{
    fs,
    time::{Duration, Instant},
};

#[cfg(target_os = "linux")]
const MAX_NAMESPACES: usize = 16;
#[cfg(target_os = "linux")]
const MAX_OUTPUT_BYTES: usize = 1024 * 1024;
#[cfg(target_os = "linux")]
const COMMAND_TIMEOUT_MS: u64 = 2000;
#[cfg(target_os = "linux")]
const TOTAL_TIMEOUT: Duration = Duration::from_secs(20);

#[cfg(any(target_os = "linux", test))]
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 255
        && name.as_bytes()[0].is_ascii_alphanumeric()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
        && name != "."
        && name != ".."
}

#[cfg(any(target_os = "linux", test))]
fn namespace_inode(identity: &str) -> Result<u64> {
    let digits = identity
        .strip_prefix("net:[")
        .and_then(|s| s.strip_suffix(']'))
        .context("Invalid network namespace identity")?;
    ensure!(
        !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()),
        "Invalid network namespace inode"
    );
    let inode = digits.parse()?;
    ensure!(inode != 0, "Network namespace inode is zero");
    Ok(inode)
}

#[cfg(any(target_os = "linux", test))]
fn decode(stdout: &str, inode: u64, self_inode: u64, epoch: u64, digest: &str) -> Result<Value> {
    let value: Value = serde_json::from_str(stdout).context("Invalid observer JSON")?;
    ensure!(inode != self_inode, "Approved namespace aliases the current namespace");
    let observed = namespace_inode(
        value
            .get("namespace")
            .and_then(Value::as_str)
            .context("Missing namespace identity")?,
    )?;
    ensure!(observed == inode, "Observer namespace inode mismatch");
    ensure!(
        value.get("mode").and_then(Value::as_str) == Some("observe-only"),
        "Unexpected observer mode"
    );
    ensure!(
        value.get("schema").and_then(Value::as_u64) == Some(1),
        "Unexpected observer schema"
    );
    ensure!(
        value.get("epoch").and_then(Value::as_u64) == Some(epoch),
        "Observer policy epoch mismatch"
    );
    ensure!(
        value.get("policy_sha256").and_then(Value::as_str) == Some(digest),
        "Observer policy digest mismatch"
    );
    ensure!(
        value.get("edges").is_some_and(Value::is_array),
        "Missing observer edges array"
    );
    Ok(value)
}

#[cfg(target_os = "linux")]
fn inode(path: &Path) -> Result<u64> {
    use std::os::unix::fs::MetadataExt;
    Ok(fs::metadata(path)
        .with_context(|| format!("Cannot inspect namespace {}", path.display()))?
        .ino())
}

/// Observe approved namespaces and report every unavailable or invalid observation.
///
/// Each subprocess has an output limit and deadline; the entire collection has
/// a twenty-second budget and processes at most sixteen enrolled names.
pub fn collect(policy_path: &Path, policy: &EdgePolicy, expected_digest: &str) -> Result<Value> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (policy_path, expected_digest);
        Ok(json!({"complete": policy.namespaces.is_empty(), "results": [],
            "errors": if policy.namespaces.is_empty() { Vec::<String>::new() } else { vec!["Network namespace inventory requires Linux".to_string()] }}))
    }
    #[cfg(target_os = "linux")]
    {
        let deadline = Instant::now() + TOTAL_TIMEOUT;
        let mut results = Vec::new();
        let mut errors = Vec::new();
        let self_inode = inode(Path::new("/proc/self/ns/net"))?;
        let policy_arg = policy_path.to_str().context("Policy path must be UTF-8")?;
        let mut seen = std::collections::HashSet::new();
        for (index, name) in policy.namespaces.iter().enumerate() {
            let observation = (|| -> Result<Value> {
                ensure!(index < MAX_NAMESPACES, "Namespace collection limit exceeded");
                ensure!(valid_name(name), "Invalid approved namespace name");
                ensure!(seen.insert(name), "Duplicate approved namespace");
                let remaining = deadline.saturating_duration_since(Instant::now());
                ensure!(!remaining.is_zero(), "Namespace collection deadline exceeded");
                let timeout = u64::try_from(remaining.as_millis())
                    .unwrap_or(COMMAND_TIMEOUT_MS)
                    .min(COMMAND_TIMEOUT_MS);
                ensure!(timeout > 0, "Namespace collection deadline exceeded");
                let ns_path = Path::new("/run/netns").join(name);
                let expected = inode(&ns_path)?;
                ensure!(
                    expected != self_inode,
                    "Approved namespace aliases the current namespace"
                );
                let args = [
                    "netns",
                    "exec",
                    name.as_str(),
                    "/usr/local/libexec/pcai-usb4-agent",
                    "observe-current",
                    policy_arg,
                ];
                let (status, stdout, stderr) =
                    run_bounded_command_with_deadline(Path::new("/usr/sbin/ip"), &args, timeout, MAX_OUTPUT_BYTES)?;
                ensure!(status == Some(0), "Namespace observer exited {status:?}: {stderr}");
                ensure!(
                    inode(&ns_path)? == expected,
                    "Approved namespace changed during observation"
                );
                decode(&stdout, expected, self_inode, policy.epoch, expected_digest)
            })();
            match observation {
                Ok(value) => results.push(json!({"name": name, "status": "observed", "observation": value})),
                Err(error) => {
                    let message = format!("{error:#}");
                    errors.push(json!({"name": name, "error": message}));
                    results.push(json!({"name": name, "status": "unavailable", "error": message}));
                }
            }
        }
        Ok(json!({"complete": errors.is_empty(), "results": results, "errors": errors}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Protect command argument/path boundaries; detects unsafe approved names.
    // Needs pure name fixtures; breadcrumb: namespace_inventory::valid_name.
    #[test]
    fn rejects_unsafe_namespace_names() {
        for name in ["", "-n", "../root", "a/b", ".", "..", "with space", "é", "a\n"] {
            assert!(!valid_name(name), "accepted {name:?}");
        }
        assert!(valid_name("ms01-switch_1.test"));
    }

    // Protect namespace/policy binding; detects successful commands reporting another scope.
    // Needs pure JSON fixtures; breadcrumb: namespace_inventory::decode.
    #[test]
    fn binds_observation_to_namespace_and_policy() {
        let value = json!({"schema": 1, "namespace": "net:[42]", "mode": "observe-only", "epoch": 7, "policy_sha256": "abc", "edges": []}).to_string();
        assert!(decode(&value, 42, 1, 7, "abc").is_ok());
        assert!(decode(&value, 43, 1, 7, "abc").is_err());
        assert!(decode(&value, 42, 42, 7, "abc").is_err());
        assert!(decode(&value, 42, 1, 8, "abc").is_err());
        assert!(decode(&value, 42, 1, 7, "wrong").is_err());
        assert!(decode("garbage", 42, 1, 7, "abc").is_err());
        assert!(decode("{}", 42, 1, 7, "abc").is_err());
    }

    // Protect provider failure semantics; detects missing executable reported as success.
    // Needs a missing local path, no live setns; breadcrumb: run_bounded_command_with_deadline.
    #[test]
    fn missing_provider_is_bounded_failure() {
        let dir = tempfile::tempdir().unwrap();
        assert!(run_bounded_command_with_deadline(&dir.path().join("missing-ip"), &[], 100, 1024).is_err());
    }
}
