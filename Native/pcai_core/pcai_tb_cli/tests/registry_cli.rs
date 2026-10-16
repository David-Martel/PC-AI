//! Runtime rejection protects callers from silently falling back to generated identity.
use std::fs;
use std::process::Command;

#[test]
fn every_command_rejects_invalid_explicit_registry_before_applying() {
    let cases = [
        ("unknown-schema", r#"{"schemaVersion":999,"fleetPeers":[]}"#),
        (
            "duplicate-ip",
            r#"{"schemaVersion":1,"fleetPeers":[{"name":"one","lanAddresses":["192.0.2.1"]},{"name":"two","lanAddresses":["192.0.2.1"]}]}"#,
        ),
        ("malformed-json", "{"),
    ];
    let home = tempfile::tempdir().unwrap();
    for (name, text) in cases {
        let registry = home.path().join(format!("{name}.json"));
        fs::write(&registry, text).unwrap();
        for command in ["status", "identity", "discover", "register", "install"] {
            let output = Command::new(env!("CARGO_BIN_EXE_pcai-tb"))
                .args([command, "--apply", "--registry"])
                .arg(&registry)
                .env("HOME", home.path())
                .env("USERPROFILE", home.path())
                .output()
                .unwrap();
            assert!(!output.status.success(), "{command} accepted {name}");
            assert!(output.stdout.is_empty(), "{command} emitted success JSON for {name}");
        }
    }
    assert!(!home.path().join("bin").exists());
    assert!(!home.path().join(".local").exists());
}

#[test]
fn missing_explicit_registry_is_an_error() {
    let home = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_pcai-tb"))
        .args(["identity", "--registry"])
        .arg(home.path().join("missing.json"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Explicit registry"));
}

#[test]
fn valid_explicit_registry_allows_identity_without_network_changes() {
    let home = tempfile::tempdir().unwrap();
    let registry = home.path().join("registry.json");
    fs::write(&registry, r#"{"schemaVersion":1,"fleetPeers":[]}"#).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_pcai-tb"))
        .args(["identity", "--computer", "fixture", "--registry"])
        .arg(registry)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["ComputerName"], "fixture");
}
