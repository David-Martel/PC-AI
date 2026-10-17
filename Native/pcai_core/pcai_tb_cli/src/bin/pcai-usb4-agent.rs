//! Emit an enrolled USB4 inventory without mutating network configuration.
use anyhow::{bail, Context, Result};
use pcai_tb::{edge_observer::observe, edge_policy::EdgePolicy};
use sha2::{Digest, Sha256};
use std::{env, fs, io::Read, path::Path};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 2 || args[0] != "observe" {
        bail!("Usage: pcai-usb4-agent observe POLICY.json");
    }
    let path = Path::new(&args[1]);
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let mut file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        bail!("Policy must be a regular file, not a symlink");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.uid() != 0 || metadata.mode() & 0o022 != 0 {
            bail!("Policy must be root-owned and not group/world writable");
        }
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let policy: EdgePolicy = serde_json::from_slice(&bytes).context("Invalid enrollment policy")?;
    let namespace = fs::read_link("/proc/self/ns/net").context("Cannot identify current network namespace")?;
    let boot_id = fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
    let edges = observe(Path::new("/sys"), &policy)?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"schema":1,"epoch":policy.epoch,"policy_sha256":hex::encode(Sha256::digest(&bytes)),"observed_at":chrono::Utc::now().to_rfc3339(),"boot_id":boot_id.trim(),"mode":"observe-only","namespace":namespace,"namespace_inventory_complete":false,"uninspected_namespaces":policy.namespaces,"edges":edges})
        )?
    );
    Ok(())
}
