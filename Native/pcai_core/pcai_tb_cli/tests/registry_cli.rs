use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("pcai-tb-{name}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn repo_registry_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../Config/thunderbolt-peers.json")
}

#[test]
fn cli_fails_when_explicit_registry_is_missing() {
    let dir = temp_dir("missing-explicit-registry");
    let missing = dir.join("missing.json");

    let output = Command::new(env!("CARGO_BIN_EXE_pcai-tb"))
        .args([
            "identity",
            "--computer",
            "minisforum",
            "--registry",
            missing.to_str().expect("utf8 path"),
            "--compact",
        ])
        .output()
        .expect("run pcai-tb");

    let _ = fs::remove_dir_all(&dir);
    assert!(
        !output.status.success(),
        "expected missing --registry to fail, stdout={}, stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Explicit registry does not exist"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn cli_fails_when_default_registry_is_malformed() {
    let dir = temp_dir("malformed-default-registry");
    let machine_dir = dir.join(".machine");
    fs::create_dir_all(&machine_dir).expect("create .machine");
    fs::write(machine_dir.join("thunderbolt-peers.json"), "{not-json").expect("write invalid json");

    let output = Command::new(env!("CARGO_BIN_EXE_pcai-tb"))
        .args(["identity", "--computer", "minisforum", "--compact"])
        .env("HOME", &dir)
        .env("USERPROFILE", &dir)
        .output()
        .expect("run pcai-tb");

    let _ = fs::remove_dir_all(&dir);
    assert!(
        !output.status.success(),
        "expected malformed default registry to fail, stdout={}, stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Invalid Thunderbolt peer registry JSON"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn status_fails_when_explicit_registry_is_missing() {
    let dir = temp_dir("status-missing-explicit-registry");
    let missing = dir.join("missing.json");

    let output = Command::new(env!("CARGO_BIN_EXE_pcai-tb"))
        .args([
            "status",
            "--registry",
            missing.to_str().expect("utf8 path"),
            "--compact",
        ])
        .output()
        .expect("run pcai-tb");

    let _ = fs::remove_dir_all(&dir);
    assert!(
        !output.status.success(),
        "expected status with missing --registry to fail, stdout={}, stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Explicit registry does not exist"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn cli_rejects_unknown_flags() {
    let reg = repo_registry_path();
    let output = Command::new(env!("CARGO_BIN_EXE_pcai-tb"))
        .args([
            "identity",
            "--registry",
            reg.to_str().expect("utf8 path"),
            "--bogus-nonexistent-flag",
        ])
        .output()
        .expect("run pcai-tb");

    assert!(
        !output.status.success(),
        "expected unknown CLI flag to fail, stdout={}, stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Unknown option '--bogus-nonexistent-flag'"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn cli_rejects_missing_or_invalid_numeric_option_values() {
    let reg = repo_registry_path();
    for bad_args in [
        vec!["register", "--mtu"],
        vec!["register", "--mtu", "not-a-number"],
        vec!["register", "--mtu", "500"],
        vec!["register", "--metric", "0"],
        vec!["discover", "--timeout-ms", "0"],
        vec!["benchmark", "--iterations", "0"],
    ] {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_pcai-tb"));
        cmd.args(&bad_args)
            .args(["--registry", reg.to_str().expect("utf8 path")]);
        let output = cmd.output().expect("run pcai-tb");
        assert!(
            !output.status.success(),
            "expected invalid CLI args {bad_args:?} to fail, stdout={}, stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn cli_resolves_minisforum_and_millylaptop1_pair() {
    let reg = repo_registry_path();
    let output = Command::new(env!("CARGO_BIN_EXE_pcai-tb"))
        .args([
            "identity",
            "--computer",
            "minisforum-ms01",
            "--peer",
            "millylaptop1-usb4",
            "--registry",
            reg.to_str().expect("utf8 path"),
            "--compact",
        ])
        .output()
        .expect("run pcai-tb");

    assert!(
        output.status.success(),
        "expected identity resolution for minisforum<->millylaptop1 to succeed, stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("valid json output");
    assert_eq!(json["ComputerName"], "minisforum");
    assert_eq!(json["IsKnownFleetPeer"], true);
    assert_eq!(json["PeerComputerName"], "millylaptop1");
    assert_eq!(json["PairwiseIpv4"]["PairKey"], "millylaptop1<->minisforum");
}

// trace: PENDING-FABRIC-F09 [pending-tpm-intake]
// trace: PENDING-FABRIC-F11 [pending-tpm-intake]
#[test]
fn cli_switch_subcommand_parses_ipam_and_sfp_topologies() {
    let reg = repo_registry_path();
    let output = Command::new(env!("CARGO_BIN_EXE_pcai-tb"))
        .args(["switch", "--registry", reg.to_str().expect("utf8 path"), "--compact"])
        .output()
        .expect("run pcai-tb switch");

    assert!(
        output.status.success(),
        "expected switch subcommand to succeed, stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("valid json output");
    assert_eq!(
        json["IpamAllocations"]["p2pThunderboltUlaPrefix"],
        "fd00:7462:7032::/48"
    );
    assert_eq!(
        json["IpamAllocations"]["bridgedLanCanaryUlaPrefix"],
        "fd5f:743d:6f55:50::/64"
    );
    assert_eq!(json["IpamAllocations"]["bridgedLanPolicyTable"], 6150);
    assert!(
        json["TopologyLinks"].as_array().map(|a| a.len() >= 4).unwrap_or(false),
        "expected TopologyLinks array in switch output"
    );
    assert!(
        json["SharedSubnetRoutes"]
            .as_array()
            .map(|a| a.len() >= 2)
            .unwrap_or(false),
        "expected SharedSubnetRoutes array in switch output"
    );
}

// trace: PENDING-FABRIC-F10 [pending-tpm-intake]
// trace: PENDING-FABRIC-F12 [pending-tpm-intake]
// trace: PENDING-FABRIC-F13 [pending-tpm-intake]
#[test]
fn cli_vnv_subcommand_emits_vigil_hil_device_evidence() {
    let reg = repo_registry_path();
    let dir = temp_dir("vnv-evidence-output");
    let fail_out_file = dir.join("hil-evidence-fail.json");
    let hold_out_file = dir.join("hil-evidence-hold.json");
    let fail_reg_file = dir.join("fail-registry.json");
    let hold_reg_file = dir.join("hold-registry.json");

    let mut reg_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&reg).expect("read repo registry")).expect("parse repo registry");
    if let Some(arr) = reg_json["sfpSwitchTopologies"].as_array_mut() {
        for item in arr {
            item["host"] = serde_json::Value::String("fixture-switch-host".to_string());
        }
    }
    // 1. Negative control: unrouted remote gateway subnet must fail closed (overall_status == FAIL, exit non-zero)
    reg_json["ipamAllocations"]["sharedSwitchSubnets"] = serde_json::json!([
        {
            "subnet": "198.18.254.0/30",
            "gatewayHost": "remote-nonexistent-gw",
            "gatewayIpv4": "198.18.254.1",
            "role": "unrouted-negative-control"
        }
    ]);
    fs::write(&fail_reg_file, serde_json::to_string_pretty(&reg_json).unwrap()).expect("write fail registry");

    let fail_output = Command::new(env!("CARGO_BIN_EXE_pcai-tb"))
        .args([
            "vnv",
            "--registry",
            fail_reg_file.to_str().expect("utf8 path"),
            "--output",
            fail_out_file.to_str().expect("utf8 path"),
            "--compact",
        ])
        .output()
        .expect("run pcai-tb vnv negative control");

    assert!(
        !fail_output.status.success(),
        "expected vnv with unrouted sharedSwitchSubnets to fail closed"
    );
    assert!(fail_out_file.is_file(), "expected V&V evidence output file on failure");
    let fail_ev: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&fail_out_file).unwrap()).expect("valid fail vnv json");
    assert_eq!(fail_ev["schema"], "vigil.hil-device-evidence/v1");
    assert_eq!(fail_ev["overall_status"], "FAIL");
    assert_eq!(fail_ev["acceptance_credit"], false);
    assert_eq!(fail_ev["tpm_admission_status"], "PENDING_TPM_INTAKE");

    // 2. Positive technical control: when sharedSwitchSubnets are locally hosted by COMPUTERNAME, technical checks
    // succeed and overall_status is HOLD_PENDING_TPM_INTAKE (never claiming acceptance credit before TPM intake).
    reg_json["ipamAllocations"]["bridgedLanCanaryEndpoints"] = serde_json::json!([]);
    reg_json["ipamAllocations"]["sharedSwitchSubnets"] = serde_json::json!([
        {
            "subnet": "10.254.253.0/30",
            "gatewayHost": "dtm-p1gen9",
            "gatewayIpv4": "192.168.50.90",
            "role": "local-gw-fixture"
        }
    ]);
    fs::write(&hold_reg_file, serde_json::to_string_pretty(&reg_json).unwrap()).expect("write hold registry");

    let hold_output = Command::new(env!("CARGO_BIN_EXE_pcai-tb"))
        .env("COMPUTERNAME", "dtm-p1gen9")
        .args([
            "vnv",
            "--registry",
            hold_reg_file.to_str().expect("utf8 path"),
            "--output",
            hold_out_file.to_str().expect("utf8 path"),
            "--compact",
        ])
        .output()
        .expect("run pcai-tb vnv hold control");

    assert!(
        hold_output.status.success(),
        "expected vnv hold control to succeed, stdout={}, stderr={}",
        String::from_utf8_lossy(&hold_output.stdout),
        String::from_utf8_lossy(&hold_output.stderr)
    );
    let hold_ev: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&hold_out_file).unwrap()).expect("valid hold vnv json");
    assert_eq!(hold_ev["schema"], "vigil.hil-device-evidence/v1");
    assert_eq!(hold_ev["overall_status"], "HOLD_PENDING_TPM_INTAKE");
    assert_eq!(hold_ev["acceptance_credit"], false);
    assert_eq!(hold_ev["tpm_admission_status"], "PENDING_TPM_INTAKE");
    assert!(hold_ev["usb4_check_receipt"].is_null());

    let gates = hold_ev["requirement_gates"].as_array().expect("requirement_gates array");
    assert_eq!(gates.len(), 5);
    for expected_trace in [
        "PENDING-FABRIC-F09",
        "PENDING-FABRIC-F10",
        "PENDING-FABRIC-F11",
        "PENDING-FABRIC-F12",
        "PENDING-FABRIC-F13",
    ] {
        assert!(
            gates.iter().any(|g| {
                g["traceId"] == expected_trace
                    && g["acceptanceCredit"] == false
                    && g["tpmAdmissionStatus"] == "PENDING_TPM_INTAKE"
                    && (g["status"] == "HOLD_PENDING_TPM_INTAKE" || g["status"] == "SKIP_NOT_APPLICABLE")
            }),
            "missing or unexpected gate {expected_trace} in {gates:?}"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_lifecycle_subcommand_evaluates_usb4_states_and_never_claims_always_on_transit() {
    let reg = repo_registry_path();
    let output = Command::new(env!("CARGO_BIN_EXE_pcai-tb"))
        .env("COMPUTERNAME", "dtm-p1gen9")
        .args([
            "lifecycle",
            "--registry",
            reg.to_str().expect("utf8 path"),
            "--timeout-ms",
            "150",
            "--compact",
        ])
        .output()
        .expect("run pcai-tb lifecycle");

    assert!(
        output.status.success(),
        "expected lifecycle subcommand to succeed, stdout={}, stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("valid lifecycle json");
    assert_eq!(json["Schema"], "pcai.usb4-lifecycle/v1");
    assert_eq!(json["ComputerName"], "dtm-p1gen9");
    assert_eq!(json["OverallValid"], true);
    assert_eq!(
        json["Policy"]["allowAlwaysOnTransitClaimWhenSelectiveSuspend"],
        false
    );
    let link_states = json["PeerLinkStates"].as_array().expect("PeerLinkStates array");
    assert!(!link_states.is_empty(), "expected non-empty PeerLinkStates");
    assert!(
        link_states.iter().all(|l| l["AlwaysOnTransitCapable"] == false),
        "no USB4 link may claim AlwaysOnTransitCapable=true under SelectiveSuspend/Unsupported wake"
    );
    let plugable = link_states
        .iter()
        .find(|l| l["LinkId"] == "ms01-plugable-dock-usb4-routed")
        .expect("ms01-plugable-dock-usb4-routed entry");
    assert_eq!(plugable["LifecycleState"], "disconnected");
    assert_eq!(plugable["DownstreamLinkStatus"], "none");
    assert_eq!(plugable["WindowsXdomainPresent"], false);

    let milly = link_states
        .iter()
        .find(|l| l["LinkId"] == "ms01-milly-usb4-bridge")
        .expect("ms01-milly-usb4-bridge entry");
    assert_eq!(milly["PreserveBridgeCustody"], true);
}

