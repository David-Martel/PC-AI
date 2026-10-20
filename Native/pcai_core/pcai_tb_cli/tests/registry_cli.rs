use pcai_tb::platform::hidden_command;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

// Protects unattended native probes; detects a console handle in a real hidden
// child. Needs Windows only; breadcrumb: src/platform/mod.rs::hidden_command.
#[cfg(windows)]
#[test]
fn hidden_child_has_no_console_window() {
    const MARKER: &str = "PCAI_TEST_HIDDEN_CHILD";
    if std::env::var_os(MARKER).is_some() {
        // SAFETY: this parameterless Win32 observer does not access Rust memory.
        assert!(unsafe { windows_sys::Win32::System::Console::GetConsoleWindow() }.is_null());
        return;
    }
    let output = hidden_command(std::env::current_exe().unwrap())
        .args(["--exact", "hidden_child_has_no_console_window", "--nocapture"])
        .env(MARKER, "1")
        .output()
        .expect("run hidden child");
    assert!(
        output.status.success(),
        "{} {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

// Protects local-only registration; detects --computer selecting another host.
// Needs fixture registry only; deliberately dry-run so a regression cannot mutate
// real adapters. The dispatch guard is independent of --apply;
// breadcrumb: src/main.rs register dispatch.
#[test]
fn register_rejects_remote_computer_override_in_dry_run() {
    let registry = repo_registry_path();
    let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
        .args([
            "register",
            "--computer",
            "minisforum",
            "--registry",
            registry.to_str().unwrap(),
            "--compact",
        ])
        .output()
        .expect("run rejected dry-run registration");
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("register targets this machine"));
}

// Protects headless Windows identity; detects unknown-host under sanitized child
// environments. Needs Windows only, no attached devices or global env writes;
// breadcrumb: src/identity.rs::local_hostname and operating_system_hostname.
#[cfg(windows)]
#[test]
fn cli_identity_uses_windows_name_without_environment_names() {
    let expected = pcai_tb::identity::operating_system_hostname().expect("Windows hostname");
    let registry = repo_registry_path();
    let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
        .args(["identity", "--registry", registry.to_str().unwrap(), "--compact"])
        .env_remove("COMPUTERNAME")
        .env_remove("HOSTNAME")
        .output()
        .expect("run sanitized CLI");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let identity: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(identity["InputName"].as_str(), Some(expected.as_str()));
    assert_ne!(identity["ComputerName"], "unknown-host");
}

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

    let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
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

    let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
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

    let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
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
    let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
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
        let mut cmd = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"));
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
    let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
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
    let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
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

    let fail_output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
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

    // 2. Local gateway control: route presence is not hardware-provider readiness.
    // Missing providers must remain FAIL; healthy observations remain pending TPM intake.
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

    let hold_output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
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

    let hold_ev: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&hold_out_file).unwrap()).expect("valid hold vnv json");
    assert_eq!(hold_ev["schema"], "vigil.hil-device-evidence/v1");
    // Hosted runners may have no USB4 provider. An on-host local-gateway fixture
    // does not invent that hardware; derive expected readiness from raw observation.
    let observed = &hold_ev["switch_status"];
    let missing_router =
        observed["LocalUsb4Routers"].as_array().unwrap().is_empty() && observed["MatchedTopology"].is_null();
    let windows = hold_ev["os_family"] == "windows";
    let metrics = &observed["WindowsMetricHierarchy"];
    let metrics_ready = !metrics.is_null()
        && metrics["HierarchyValid"] == true
        && metrics["Interfaces"].as_array().is_some_and(|a| !a.is_empty());
    let unauthorized_router = observed["LocalUsb4Routers"].as_array().unwrap().iter().any(|r| {
        r["Status"]
            .as_str()
            .is_some_and(|v| v.eq_ignore_ascii_case("unauthorized"))
    });
    let expected_failure = missing_router || unauthorized_router || (windows && !metrics_ready);
    assert_eq!(hold_output.status.success(), !expected_failure);
    assert_eq!(
        hold_ev["overall_status"],
        if expected_failure {
            "FAIL"
        } else {
            "HOLD_PENDING_TPM_INTAKE"
        }
    );
    let f09 = hold_ev["requirement_gates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|g| g["traceId"] == "PENDING-FABRIC-F09")
        .unwrap();
    assert_eq!(
        f09["technicalVerdict"],
        if unauthorized_router {
            "FAIL"
        } else if missing_router {
            "BLOCKED_MISSING_PROVIDER"
        } else {
            "PASS"
        }
    );
    let f12 = hold_ev["requirement_gates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|g| g["traceId"] == "PENDING-FABRIC-F12")
        .unwrap();
    let metric_verdict = if !windows {
        "SKIP_NOT_APPLICABLE"
    } else if metrics.is_null() {
        "BLOCKED_MISSING_PROVIDER"
    } else if metrics_ready {
        "PASS"
    } else {
        "FAIL"
    };
    assert_eq!(f12["technicalVerdict"], metric_verdict);
    for (id, verdict) in [
        ("PENDING-FABRIC-F10", "SKIP_NOT_APPLICABLE"),
        ("PENDING-FABRIC-F11", "SKIP_NOT_APPLICABLE"),
        ("PENDING-FABRIC-F13", "PASS"),
    ] {
        let gate = hold_ev["requirement_gates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["traceId"] == id)
            .unwrap();
        assert_eq!(gate["technicalVerdict"], verdict, "unexpected fixture gate {id}");
    }
    assert_eq!(hold_ev["acceptance_credit"], false);
    assert_eq!(hold_ev["tpm_admission_status"], "PENDING_TPM_INTAKE");
    assert!(hold_ev["usb4_check_receipt"].is_null());

    let gates = hold_ev["requirement_gates"]
        .as_array()
        .expect("requirement_gates array");
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
                    && g["status"]
                        == if g["technicalVerdict"] == "PASS" {
                            serde_json::json!("HOLD_PENDING_TPM_INTAKE")
                        } else {
                            g["technicalVerdict"].clone()
                        }
            }),
            "missing or unexpected gate {expected_trace} in {gates:?}"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_lifecycle_subcommand_evaluates_usb4_states_and_never_claims_always_on_transit() {
    // This is an absent-interface refusal control, not a claim about attached Windows hardware.
    let fixture_dir = tempfile::tempdir().expect("private lifecycle registry");
    let reg = fixture_dir.path().join("peers.json");
    let mut registry: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(repo_registry_path()).expect("read registry"))
            .expect("parse registry");
    for (index, link) in registry["topologyLinks"].as_array_mut().unwrap().iter_mut().enumerate() {
        link["sourceInterface"] = serde_json::json!(format!("pcai-fixture-{index}"));
        link["targetInterface"] = serde_json::json!(format!("pcai-fixture-{index}"));
    }
    fs::write(&reg, serde_json::to_vec(&registry).unwrap()).expect("write private registry");
    let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
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
        !output.status.success(),
        "Unowned fixture interfaces must refuse readiness"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("valid lifecycle json");
    assert_eq!(json["Schema"], "pcai.usb4-lifecycle/v1");
    assert_eq!(json["ComputerName"], "dtm-p1gen9");
    assert_eq!(json["OverallValid"], false);
    assert_eq!(json["ApplyMode"], false);
    let withdrawals = json["StaleRouteWithdrawals"].as_array().expect("withdrawal decisions");
    let refusals: Vec<_> = withdrawals.iter().filter(|a| !a["Error"].is_null()).collect();
    assert!(!refusals.is_empty(), "Missing adapters must produce explicit refusals");
    assert!(refusals
        .iter()
        .all(|a| a["Error"] == "Missing or ambiguous current adapter alias"));
    assert!(withdrawals.iter().all(|a| a["Withdrawn"] == false));
    assert!(json["ResumeReprobe"]["ReapplyActions"].as_array().unwrap().is_empty());
    assert_eq!(json["DefaultRoutesPreserved"], true);
    assert_eq!(json["Policy"]["allowAlwaysOnTransitClaimWhenSelectiveSuspend"], false);
    let link_states = json["PeerLinkStates"].as_array().expect("PeerLinkStates array");
    assert!(!link_states.is_empty(), "expected non-empty PeerLinkStates");
    assert!(
        link_states.iter().all(|l| l["AlwaysOnTransitCapable"] == false),
        "no USB4 link may claim AlwaysOnTransitCapable=true under SelectiveSuspend/Unsupported wake"
    );

    // Case 2 + Lenovo 40BA dock firmware inventory on ms01-plugable-dock-usb4-routed
    let plugable = link_states
        .iter()
        .find(|l| l["LinkId"] == "ms01-plugable-dock-usb4-routed")
        .expect("ms01-plugable-dock-usb4-routed entry");
    assert_eq!(plugable["LifecycleState"], "disconnected");
    assert_eq!(plugable["DownstreamLinkStatus"], "none");
    assert_eq!(plugable["WindowsXdomainPresent"], false);
    assert_eq!(plugable["LinkAdmitted"], false);
    assert_eq!(plugable["AdmissionReason"], "no_observed_xdomain");
    assert_eq!(plugable["RequiredTopologyClass"], "required_two_hub");
    assert_eq!(plugable["RequiredTopologyQualified"], false);
    assert_eq!(plugable["LocalRootRouterFw"], "67.67");
    assert_eq!(plugable["DockModel"], "40BA");
    assert_eq!(plugable["DockSerial"], "1S40BAYJ0019GD");
    assert_eq!(plugable["DockFwPackage"], "1.0.07");
    assert_eq!(plugable["DockFwTarget"], "1.2.99");
    assert_eq!(plugable["DockPdFw"], "0.0.47");
    assert_eq!(plugable["DockTbt5Nvm"], "62.62");
    assert_eq!(plugable["DockDmcFw"], "0.6.5");
    assert_eq!(plugable["WindowsReportedLinkSpeedBps"], 20_000_000_000u64);
    assert_eq!(plugable["LinkSpeedIsNominalDriverDisplay"], true);
    assert_eq!(plugable["LinkSpeedIsHardwareMaximum"], false);

    // Case 3 + Case 5: ms01-plugable-gen9-hostport-control (PID_E333 FW 11.7)
    let hostport = link_states
        .iter()
        .find(|l| l["LinkId"] == "ms01-plugable-gen9-hostport-control")
        .expect("ms01-plugable-gen9-hostport-control entry");
    assert_eq!(hostport["RequiredTopologyClass"], "required_two_hub");
    assert_eq!(hostport["ObservedTopologyClass"], "host_port_control");
    assert_eq!(hostport["QualificationScope"], "host_port_control_only");
    assert_eq!(hostport["RequiredTopologyQualified"], false);
    assert_eq!(hostport["LocalRootRouterFw"], "11.7");
    assert_eq!(hostport["WindowsReportedLinkSpeedBps"], 20_000_000_000u64);
    assert_eq!(hostport["LinkSpeedIsNominalDriverDisplay"], true);
    assert_eq!(hostport["LinkSpeedIsHardwareMaximum"], false);
    assert_eq!(
        hostport["MeasuredPayloadGbpsSamples"],
        serde_json::json!([7.86, 12.90, 14.12])
    );

    // Case 4: Preserve Milly bridge custody
    let milly = link_states
        .iter()
        .find(|l| l["LinkId"] == "ms01-milly-usb4-bridge")
        .expect("ms01-milly-usb4-bridge entry");
    assert_eq!(milly["PreserveBridgeCustody"], true);
    assert_eq!(milly["LinkAdmitted"], true);
}

// Protects valid observation without declared edges; detects treating every dry-run as failure.
// Needs native CLI only; breadcrumb: lifecycle overall_valid and registry topology selection.
#[test]
fn cli_lifecycle_empty_topology_is_valid_without_route_actions() {
    let dir = tempfile::tempdir().expect("private registry");
    let reg = dir.path().join("peers.json");
    let mut registry: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(repo_registry_path()).unwrap()).unwrap();
    registry["topologyLinks"] = serde_json::json!([]);
    fs::write(&reg, serde_json::to_vec(&registry).unwrap()).unwrap();
    let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
        .env("COMPUTERNAME", "dtm-p1gen9")
        .args([
            "lifecycle",
            "--registry",
            reg.to_str().unwrap(),
            "--timeout-ms",
            "150",
            "--compact",
        ])
        .output()
        .expect("run lifecycle");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["OverallValid"], true);
    assert_eq!(value["ApplyMode"], false);
    assert_eq!(value["DefaultRoutesPreserved"], true);
    assert!(value["StaleRouteWithdrawals"].as_array().unwrap().is_empty());
    assert!(value["PeerLinkStates"].as_array().unwrap().is_empty());
}
