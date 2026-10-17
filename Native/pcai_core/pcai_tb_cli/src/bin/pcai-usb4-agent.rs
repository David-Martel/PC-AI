//! Emit an enrolled USB4 inventory without mutating network configuration.
use anyhow::{bail, Context, Result};
use pcai_tb::{edge_observer::observe, edge_policy::EdgePolicy};
use sha2::{Digest, Sha256};
use std::{env, fs, io::Read, path::Path};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let profile_mode = args.first().is_some_and(|a| {
        matches!(
            a.as_str(),
            "udev-properties" | "networkd-profile" | "nm-unmanaged-profile"
        )
    });
    if (profile_mode && args.len() != 3)
        || (!profile_mode
            && (args.len() != 2
                || !matches!(
                    args[0].as_str(),
                    "observe" | "observe-current" | "observe-all" | "qualify" | "activate-nm" | "reconcile-networkd"
                )))
    {
        bail!("Usage: pcai-usb4-agent observe|observe-current|observe-all|qualify|activate-nm|reconcile-networkd POLICY.json; udev-properties|networkd-profile|nm-unmanaged-profile POLICY.json INTERFACE");
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
    if profile_mode {
        let selected: Vec<_> = edges
            .iter()
            .filter(|e| e.interface == args[2] && e.state == "ENROLLED_UNVERIFIED")
            .collect();
        if selected.len() != 1 {
            bail!("Interface must uniquely match an enrolled, unprotected edge");
        }
        let edge = selected[0].edge.as_ref().context("Missing enrolled edge")?;
        if args[0] == "nm-unmanaged-profile" {
            if policy.backend != "networkd" {
                bail!("Unmanaged reservation requires networkd enrollment");
            }
            let mac = fs::read_to_string(Path::new("/sys/class/net").join(&selected[0].interface).join("address"))?;
            let mac = mac.trim();
            if mac.len() != 17
                || !mac
                    .split(':')
                    .all(|v| v.len() == 2 && v.bytes().all(|b| b.is_ascii_hexdigit()))
            {
                bail!("Invalid observed MAC");
            }
            println!(
                "# PC-AI audited edge {}\n[device-pcai-{}]\nmatch-device=mac:{}\nmanaged=0",
                edge.edge_id, edge.edge_id, mac
            );
        } else if args[0] == "udev-properties" {
            println!("VIGIL_USB4_EDGE_ID={}", edge.edge_id);
            println!("VIGIL_USB4_BACKEND={}", policy.backend);
        } else {
            if policy.backend != "networkd" {
                bail!("Profile renderer requires networkd enrollment");
            }
            print!("{}", pcai_tb::networkd_edge::render(edge, policy.epoch));
        }
        return Ok(());
    }
    let qualifications = if args[0] == "qualify" {
        edges
            .iter()
            .filter(|e| e.state == "ENROLLED_UNVERIFIED")
            .map(|edge| match pcai_tb::edge_probe::qualify(&policy, edge) {
                Ok(value) => value,
                Err(error) => {
                    serde_json::json!({"interface":edge.interface,"state":"UNVERIFIED","error":error.to_string()})
                }
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let activations = if matches!(args[0].as_str(), "activate-nm" | "reconcile-networkd") {
        edges
            .iter()
            .filter(|e| e.state == "ENROLLED_UNVERIFIED")
            .map(|edge| {
                match if args[0] == "activate-nm" {
                    pcai_tb::nm_edge::activate(&policy, edge)
                } else {
                    pcai_tb::networkd_edge::reconcile(&policy, edge)
                } {
                    Ok(value) => value,
                    Err(error) => {
                        serde_json::json!({"interface":edge.interface,"state":"UNVERIFIED","error":error.to_string()})
                    }
                }
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let namespaces = if args[0] == "observe-all" {
        pcai_tb::namespace_inventory::collect(path, &policy, &hex::encode(Sha256::digest(&bytes)))?
    } else {
        serde_json::json!({"complete":false,"uninspected":policy.namespaces})
    };
    let record = serde_json::to_string_pretty(
        &serde_json::json!({"schema":1,"epoch":policy.epoch,"policy_sha256":hex::encode(Sha256::digest(&bytes)),"observed_at":chrono::Utc::now().to_rfc3339(),"boot_id":boot_id.trim(),"mode":if matches!(args[0].as_str(),"activate-nm"|"reconcile-networkd") {"reconcile-owned-profile"} else {"observe-only"},"namespace":namespace,"namespace_inventory_complete":namespaces["complete"],"namespaces":namespaces,"edges":edges,"qualifications":qualifications,"activations":activations}),
    )?;
    if args[0] != "observe-current"
        && env::var_os("PCAI_USB4_WRITE_STATUS").as_deref() == Some(std::ffi::OsStr::new("1"))
    {
        pcai_tb::status_record::publish(Path::new("/run/pcai-usb4-status"), &args[0], record.as_bytes(), 0)?;
    }
    println!("{record}");
    Ok(())
}
