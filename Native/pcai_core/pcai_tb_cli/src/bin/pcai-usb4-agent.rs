//! Observe enrolled USB4 links and report requested operation outcomes.
use anyhow::{bail, Context, Result};
use pcai_tb::{
    edge_observer::{observe, Observation},
    edge_policy::EdgePolicy,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{Read, Write},
    path::Path,
};

#[derive(Debug, Default)]
struct OperationSummary {
    attempted: usize,
    succeeded: usize,
    failed: usize,
}

fn is_operational(command: &str) -> bool {
    matches!(command, "qualify" | "activate-nm" | "reconcile-networkd")
}

impl OperationSummary {
    fn outcome(&self, command: &str) -> &'static str {
        if !is_operational(command) {
            "NOT_REQUESTED"
        } else if self.attempted == 0 {
            "NO_ELIGIBLE_EDGE"
        } else if self.failed != 0 {
            "FAILED"
        } else {
            "SUCCEEDED"
        }
    }
}

/// Retain each typed operation result without adopting unknown or protected edges.
fn run_enrolled_operation<F>(command: &str, edges: &[Observation], mut action: F) -> (Vec<Value>, OperationSummary)
where
    F: FnMut(&Observation) -> Result<Value>,
{
    let mut summary = OperationSummary::default();
    let mut records = Vec::new();
    if is_operational(command) {
        for edge in edges.iter().filter(|edge| edge.state == "ENROLLED_UNVERIFIED") {
            summary.attempted += 1;
            records.push(match action(edge) {
                Ok(value) => {
                    summary.succeeded += 1;
                    value
                }
                Err(error) => {
                    summary.failed += 1;
                    serde_json::json!({"interface":edge.interface,"state":"UNVERIFIED","error":error.to_string()})
                }
            });
        }
    }
    (records, summary)
}

/// Emit complete evidence to available sinks before returning command status.
fn finish_evidence<W, F>(
    command: &str,
    mut record: Value,
    summary: &OperationSummary,
    output: &mut W,
    publish: F,
) -> Result<()>
where
    W: Write,
    F: FnOnce(&[u8]) -> Result<()>,
{
    let object = record.as_object_mut().context("Evidence record must be an object")?;
    object.insert("operation".into(), command.into());
    object.insert(
        "operation_summary".into(),
        serde_json::json!({"attempted":summary.attempted,"succeeded":summary.succeeded,"failed":summary.failed,"outcome":summary.outcome(command)}),
    );
    let serialized = serde_json::to_string_pretty(&record)?;
    let output_result = (|| -> std::io::Result<()> {
        writeln!(output, "{serialized}")?;
        output.flush()
    })();
    // Attempt publication even if stdout is unavailable; either independent sink
    // may retain the complete evidence needed to diagnose a failed invocation.
    let publication_result = publish(serialized.as_bytes());
    match (output_result, publication_result) {
        (Err(output_error), Err(publication_error)) => {
            bail!("Cannot write evidence: {output_error}; cannot publish status: {publication_error}")
        }
        (Err(error), Ok(())) => return Err(error).context("Cannot write evidence"),
        (Ok(()), Err(error)) => return Err(error).context("Cannot publish status"),
        (Ok(()), Ok(())) => {}
    }
    if is_operational(command) && summary.attempted == 0 {
        bail!("USB4 {command} needs at least one eligible enrolled edge; see emitted evidence");
    }
    if is_operational(command) && summary.failed != 0 {
        bail!(
            "{} of {} USB4 {command} operations failed; see emitted evidence",
            summary.failed,
            summary.attempted
        );
    }
    Ok(())
}

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
    let (records, summary) = run_enrolled_operation(&args[0], &edges, |edge| match args[0].as_str() {
        "qualify" => pcai_tb::edge_probe::qualify(&policy, edge),
        "activate-nm" => pcai_tb::nm_edge::activate(&policy, edge),
        "reconcile-networkd" => pcai_tb::networkd_edge::reconcile(&policy, edge),
        _ => bail!("Unsupported operation"),
    });
    let (qualifications, activations) = if args[0] == "qualify" {
        (records, Vec::new())
    } else {
        (Vec::new(), records)
    };
    let namespaces = if args[0] == "observe-all" {
        pcai_tb::namespace_inventory::collect(path, &policy, &hex::encode(Sha256::digest(&bytes)))?
    } else {
        serde_json::json!({"complete":false,"uninspected":policy.namespaces})
    };
    let record = serde_json::json!({"schema":1,"epoch":policy.epoch,"policy_sha256":hex::encode(Sha256::digest(&bytes)),"observed_at":chrono::Utc::now().to_rfc3339(),"boot_id":boot_id.trim(),"mode":if matches!(args[0].as_str(),"activate-nm"|"reconcile-networkd") {"reconcile-owned-profile"} else {"observe-only"},"namespace":namespace,"namespace_inventory_complete":namespaces["complete"],"namespaces":namespaces,"edges":edges,"qualifications":qualifications,"activations":activations});
    let publish_status = args[0] != "observe-current"
        && env::var_os("PCAI_USB4_WRITE_STATUS").as_deref() == Some(std::ffi::OsStr::new("1"));
    let stdout = std::io::stdout();
    finish_evidence(&args[0], record, &summary, &mut stdout.lock(), |record| {
        if publish_status {
            // The final argument requires root ownership; it is not an exit code.
            pcai_tb::status_record::publish(Path::new("/run/pcai-usb4-status"), &args[0], record, 0)?;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pcai_tb::edge_observer::Observation;
    use serde_json::{json, Value};
    use std::{cell::RefCell, io, rc::Rc};

    #[derive(Default)]
    struct EvidenceWriter {
        bytes: Vec<u8>,
        fail_write: bool,
        fail_flush: bool,
        events: Rc<RefCell<Vec<&'static str>>>,
    }

    impl io::Write for EvidenceWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.fail_write {
                return Err(io::Error::new(io::ErrorKind::BrokenPipe, "fixture write failure"));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            self.events.borrow_mut().push("stdout-flush");
            if self.fail_flush {
                return Err(io::Error::other("fixture flush failure"));
            }
            Ok(())
        }
    }

    fn observations(states: &[&str]) -> Vec<Observation> {
        states
            .iter()
            .enumerate()
            .map(|(index, state)| Observation {
                interface: format!("edge-{index}"),
                ifindex: u32::try_from(index + 1).unwrap(),
                state: (*state).into(),
                reason: "fixture inventory".into(),
                edge: None,
            })
            .collect()
    }

    fn evidence_record(command: &str, edges: &[Observation], records: Vec<Value>) -> Value {
        if command == "qualify" {
            json!({"edges":edges,"qualifications":records,"activations":[],"mode":"observe-only"})
        } else {
            json!({"edges":edges,"qualifications":[],"activations":records,"mode":"reconcile-owned-profile"})
        }
    }

    // Protects failed operational invocations from successful automation status.
    // Detects swallowed mixed/all Result errors after complete stdout/publication.
    // Needs memory writers only; breadcrumb: run_enrolled_operation/finish_evidence.
    #[test]
    fn failed_attempts_emit_all_evidence_before_returning_failure() {
        for command in ["qualify", "activate-nm", "reconcile-networkd"] {
            for all_failed in [false, true] {
                let edges = observations(&["ENROLLED_UNVERIFIED", "UNKNOWN", "ENROLLED_UNVERIFIED"]);
                let (records, summary) = run_enrolled_operation(command, &edges, |edge| {
                    if all_failed || edge.interface == "edge-2" {
                        bail!("fixture operation refused");
                    }
                    Ok(json!({"interface":edge.interface,"state":"ACTIVE_UNVERIFIED"}))
                });
                let mut output = EvidenceWriter::default();
                let events = Rc::clone(&output.events);
                let mut published = Vec::new();
                let result = finish_evidence(
                    command,
                    evidence_record(command, &edges, records),
                    &summary,
                    &mut output,
                    |bytes| {
                        events.borrow_mut().push("published");
                        published.extend_from_slice(bytes);
                        Ok(())
                    },
                );
                assert_eq!(*events.borrow(), ["stdout-flush", "published"]);
                let stdout: Value = serde_json::from_slice(&output.bytes).unwrap();
                let status: Value = serde_json::from_slice(&published).unwrap();
                assert_eq!(stdout, status);
                assert_eq!(status["edges"].as_array().unwrap().len(), 3);
                assert_eq!(status["operation"], command);
                assert_eq!(status["operation_summary"]["attempted"], 2);
                assert_eq!(status["operation_summary"]["failed"], if all_failed { 2 } else { 1 });
                assert_eq!(status["operation_summary"]["outcome"], "FAILED");
                let records = if command == "qualify" {
                    status["qualifications"].as_array().unwrap()
                } else {
                    status["activations"].as_array().unwrap()
                };
                assert_eq!(records.len(), 2);
                assert_eq!(records[1]["state"], "UNVERIFIED");
                assert_eq!(records[1]["error"], "fixture operation refused");
                assert!(
                    result.is_err(),
                    "{command} must not exit successfully after an operation failure"
                );
            }
        }
    }

    // Protects fail-closed operational targeting while retaining foreign inventory.
    // Detects zero eligible targets treated as success or UNKNOWN/PROTECTED actions.
    // Needs memory writers only; breadcrumb: run_enrolled_operation/finish_evidence.
    #[test]
    fn operational_zero_targets_fail_without_adopting_unknown_or_protected_edges() {
        for command in ["qualify", "activate-nm", "reconcile-networkd"] {
            for states in [&[][..], &["UNKNOWN", "PROTECTED"][..]] {
                let edges = observations(states);
                let (records, summary) = run_enrolled_operation(command, &edges, |_| {
                    panic!("Unknown or protected edges must not be operated on")
                });
                let mut output = Vec::new();
                let result = finish_evidence(
                    command,
                    evidence_record(command, &edges, records),
                    &summary,
                    &mut output,
                    |_| Ok(()),
                );
                let value: Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(value["edges"].as_array().unwrap().len(), states.len());
                assert_eq!(value["operation_summary"]["attempted"], 0);
                assert_eq!(value["operation_summary"]["outcome"], "NO_ELIGIBLE_EDGE");
                assert!(result.is_err(), "{command} needs at least one eligible enrolled edge");
            }
        }
    }

    // Protects valid read-only inventory when no operation was requested.
    // Detects empty/unknown observation failures or accidental action invocation.
    // Needs memory writers only; breadcrumb: run_enrolled_operation/finish_evidence.
    #[test]
    fn observation_modes_accept_empty_unknown_and_protected_inventory() {
        for command in ["observe", "observe-current", "observe-all"] {
            for states in [&[][..], &["UNKNOWN", "PROTECTED"][..], &["ENROLLED_UNVERIFIED"][..]] {
                let edges = observations(states);
                let (records, summary) = run_enrolled_operation(command, &edges, |_| {
                    panic!("Observation must not invoke a network operation")
                });
                let mut output = Vec::new();
                finish_evidence(
                    command,
                    json!({"edges":edges,"qualifications":records,"activations":[],"mode":"observe-only"}),
                    &summary,
                    &mut output,
                    |_| Ok(()),
                )
                .unwrap();
                let value: Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(value["edges"].as_array().unwrap().len(), states.len());
                assert_eq!(value["operation_summary"]["outcome"], "NOT_REQUESTED");
                assert_eq!(value["operation_summary"]["attempted"], 0);
                assert_eq!(value["mode"], "observe-only");
            }
        }
    }

    // Protects typed operation success without inventing authentication/readiness.
    // Detects JSON string/error-field classification or rewritten activation states.
    // Needs memory writers only; breadcrumb: run_enrolled_operation/finish_evidence.
    #[test]
    fn successful_typed_results_preserve_unverified_activation_evidence() {
        for (command, state) in [
            ("qualify", "AUTHENTICATED"),
            ("activate-nm", "ACTIVE_UNVERIFIED"),
            ("reconcile-networkd", "CONFIGURED_UNVERIFIED"),
        ] {
            let edges = observations(&["ENROLLED_UNVERIFIED"]);
            let (records, summary) = run_enrolled_operation(command, &edges, |edge| {
                Ok(json!({"interface":edge.interface,"state":state,"error":"informational payload only"}))
            });
            let mut output = Vec::new();
            finish_evidence(
                command,
                evidence_record(command, &edges, records),
                &summary,
                &mut output,
                |_| Ok(()),
            )
            .unwrap();
            let value: Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(value["operation_summary"]["outcome"], "SUCCEEDED");
            assert_eq!(value["operation_summary"]["succeeded"], 1);
            assert_eq!(value["operation_summary"]["failed"], 0);
            let record = if command == "qualify" {
                &value["qualifications"][0]
            } else {
                assert_eq!(value["mode"], "reconcile-owned-profile");
                &value["activations"][0]
            };
            assert_eq!(record["state"], state);
            assert!(value.get("ready").is_none());
        }
    }

    // Protects process failure on broken stdout while retaining status evidence.
    // Detects ignored write/flush errors or publication skipped by a broken pipe.
    // Needs failing memory writers only; breadcrumb: finish_evidence output path.
    #[test]
    fn stdout_write_and_flush_failures_still_publish_available_evidence() {
        for fail_write in [true, false] {
            let mut output = EvidenceWriter {
                fail_write,
                fail_flush: !fail_write,
                ..EvidenceWriter::default()
            };
            let mut published = Vec::new();
            let result = finish_evidence(
                "observe",
                json!({"edges":[]}),
                &OperationSummary::default(),
                &mut output,
                |bytes| {
                    published.extend_from_slice(bytes);
                    Ok(())
                },
            );
            assert!(result.is_err());
            let value: Value = serde_json::from_slice(&published).unwrap();
            assert_eq!(value["operation_summary"]["outcome"], "NOT_REQUESTED");
        }
    }

    // Protects complete stdout evidence and nonzero status on publication failure.
    // Detects swallowed publisher errors or withholding stdout until publication.
    // Needs failing in-memory publisher only; breadcrumb: finish_evidence publication.
    #[test]
    fn publication_failure_preserves_complete_stdout_and_returns_failure() {
        let mut output = EvidenceWriter::default();
        let result = finish_evidence(
            "observe",
            json!({"edges":[]}),
            &OperationSummary::default(),
            &mut output,
            |_| bail!("fixture publication failure"),
        );
        assert!(result.unwrap_err().to_string().contains("publish"));
        let value: Value = serde_json::from_slice(&output.bytes).unwrap();
        assert_eq!(value["operation_summary"]["outcome"], "NOT_REQUESTED");
        assert_eq!(*output.events.borrow(), ["stdout-flush"]);
    }

    // Protects both independent failure causes when neither evidence sink works.
    // Detects error replacement that hides lost stdout or failed publication.
    // Needs failing memory sinks only; breadcrumb: finish_evidence combined errors.
    #[test]
    fn both_evidence_sink_errors_are_reported_without_success() {
        let mut output = EvidenceWriter {
            fail_write: true,
            ..EvidenceWriter::default()
        };
        let result = finish_evidence(
            "observe",
            json!({"edges":[]}),
            &OperationSummary::default(),
            &mut output,
            |_| bail!("fixture publication failure"),
        );
        let error = result.unwrap_err().to_string();
        assert!(error.contains("fixture write failure"));
        assert!(error.contains("fixture publication failure"));
    }
}
