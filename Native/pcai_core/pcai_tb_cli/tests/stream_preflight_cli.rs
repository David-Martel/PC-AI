use pcai_tb::platform::hidden_command;
use serde_json::Value;

// Protects the portable observation contract. Detects schema drift or promotion of a local
// kernel observation to a transfer claim. Needs a built CLI; breadcrumb: src/stream_preflight.rs.
#[test]
fn stream_preflight_reports_capabilities_without_qualifying_a_transfer() {
    let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
        .args(["stream-preflight", "--compact"])
        .output()
        .expect("run read-only preflight");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(String::from_utf8_lossy(&output.stdout).lines().count(), 1);
    let report: Value = serde_json::from_slice(&output.stdout).expect("valid preflight JSON");
    assert_eq!(report["schema"], "pcai.usb4-stream-preflight/v1");
    assert_eq!(report["platform"], std::env::consts::OS);
    assert_eq!(report["transfer_qualification"], "NotTested");
    if cfg!(target_os = "linux") {
        assert_eq!(report["status"], "LocalObservation");
        let linux = &report["linux"];
        assert!(linux.is_object());
        let config = &linux["stream_config"];
        match config["state"].as_str() {
            Some("BuiltIn" | "Module" | "Disabled" | "Unspecified") => {
                assert!(matches!(
                    config["source"].as_str(),
                    Some("ProcConfigGzip" | "BootConfig")
                ));
            }
            Some("Unavailable") => assert!(config.get("source").is_some_and(Value::is_null)),
            state => panic!("unexpected stream configuration state {state:?}"),
        }
        assert!(matches!(
            linux["configfs_mount"].as_str(),
            Some("Mounted" | "Absent" | "UnexpectedType" | "Unavailable")
        ));
    } else {
        assert_eq!(report["status"], "UnsupportedPlatform");
        assert!(report["linux"].is_null());
    }
}

// Protects local, read-only scope. Detects dispatch accepting mutations or alternate inputs.
// Needs a built CLI and temporary directory; breadcrumb: src/main.rs stream-preflight dispatch.
#[test]
fn stream_preflight_rejects_out_of_scope_options_before_output() {
    let dir = tempfile::tempdir().expect("create fixture directory");
    for options in [
        &["--apply"][..],
        &["--output", "out.json"],
        &["--registry", "missing.json"],
        &["--input", "missing.json"],
        &["--computer", "peer"],
        &["--skip-legacy"],
    ] {
        let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
            .arg("stream-preflight")
            .args(options)
            .current_dir(dir.path())
            .output()
            .expect("run rejected preflight");
        assert!(!output.status.success(), "{options:?}");
        assert!(output.stdout.is_empty(), "{options:?}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("only --compact is accepted"));
    }
    assert!(!dir.path().join("out.json").exists());
}
