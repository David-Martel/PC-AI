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
    // Protects state precedence against the telemetry returned by this same CLI
    // invocation. Detects a state inconsistent with standby without assuming the
    // Windows event log is awake. Pure lifecycle unit controls cover both inputs.
    let in_standby = json["ModernStandby"]["CurrentlyInModernStandby"]
        .as_bool()
        .expect("current standby must be an explicit boolean");
    assert_eq!(
        plugable["LifecycleState"],
        if in_standby { "asleep" } else { "disconnected" }
    );
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

fn route_plan_inputs(name: &str, transit: &[&str]) -> (PathBuf, PathBuf, PathBuf) {
    let dir = temp_dir(name);
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/usb4-hub-chain-observation.json");
    let mut observation: serde_json::Value = serde_json::from_str(&fs::read_to_string(fixture).unwrap()).unwrap();
    let fresh = chrono::Utc::now().to_rfc3339();
    for host in observation["hosts"].as_array_mut().unwrap() {
        host["observed_at_utc"] = serde_json::Value::String(fresh.clone());
    }
    let input = dir.join("observation.json");
    fs::write(&input, serde_json::to_vec(&observation).unwrap()).unwrap();
    let policy = dir.join("policy.json");
    fs::write(
        &policy,
        serde_json::to_vec(&serde_json::json!({ "schema": "pcai.usb4-route-policy/v1", "transit_hosts": transit }))
            .unwrap(),
    )
    .unwrap();
    (dir, input, policy)
}

// Protects: the actual route-plan entrypoint plans the recorded Gen7 -> NUC -> Gen9 hub
// chain from files, with the repository registry. Detects: dispatch, option parsing or
// registry wiring regressions. Needs: fixture rewritten to a fresh time; no adapters.
// Breadcrumb: src/main.rs route-plan dispatch and src/route_plan.rs.
#[test]
fn cli_route_plan_routes_recorded_hub_chain_through_named_transit() {
    let (dir, input, policy) = route_plan_inputs("route-plan-chain", &["dtm-super-nuc"]);
    let registry = repo_registry_path();
    let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
        .args([
            "route-plan",
            "--input",
            input.to_str().unwrap(),
            "--policy",
            policy.to_str().unwrap(),
            "--registry",
            registry.to_str().unwrap(),
            "--compact",
        ])
        .output()
        .expect("run route-plan");
    let _ = fs::remove_dir_all(&dir);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(plan["schema"], "pcai.usb4-route-plan/v1");
    assert_eq!(plan["apply_supported"], false);
    let routed = plan["paths"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["from"] == "dtm-p1gen7" && p["to"] == "dtm-p1gen9")
        .unwrap();
    assert_eq!(routed["status"], "Routed");
    assert_eq!(routed["hops"][1], "dtm-super-nuc");
}

// Protects: route-plan stays a planner. Detects: accepting --apply, other mutation flags or
// a missing policy. Needs: fixture files only. Breadcrumb: src/main.rs route-plan option gate.
#[test]
fn cli_route_plan_refuses_apply_and_requires_policy() {
    let (dir, input, policy) = route_plan_inputs("route-plan-refusals", &[]);
    let run = |args: &[&str]| {
        hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
            .args(args)
            .output()
            .expect("run route-plan refusal")
    };
    let (input, policy) = (input.to_str().unwrap(), policy.to_str().unwrap());
    for args in [
        vec!["route-plan", "--input", input, "--policy", policy, "--apply"],
        vec![
            "route-plan",
            "--input",
            input,
            "--policy",
            policy,
            "--assign-pairwise-ipv4",
        ],
        vec!["route-plan", "--input", input],
        vec!["topology-plan", "--input", input, "--policy", policy],
        vec!["status", "--policy", policy],
    ] {
        let output = run(&args);
        assert!(!output.status.success(), "{args:?} succeeded");
        assert!(output.stdout.is_empty(), "{args:?} emitted output");
    }
    let _ = fs::remove_dir_all(&dir);
}

// Protects: generated commands are valid PowerShell for the cmdlets they call. Detects:
// syntax errors, unknown cmdlets and unsupported parameters (for example a review claim that
// Get-NetAdapter lacks -InterfaceIndex) without executing anything. Needs: Windows with pwsh
// and the NetAdapter/NetTCPIP modules. Breadcrumb: src/route_plan.rs owned_apply/owned_rollback/guard_lines.
#[cfg(windows)]
#[test]
fn cli_route_plan_commands_parse_and_bind_to_real_cmdlet_parameters() {
    let (dir, input, policy) = route_plan_inputs("route-plan-pwsh", &["dtm-super-nuc"]);
    let registry = repo_registry_path();
    let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
        .args([
            "route-plan",
            "--input",
            input.to_str().unwrap(),
            "--policy",
            policy.to_str().unwrap(),
            "--registry",
            registry.to_str().unwrap(),
            "--compact",
        ])
        .output()
        .expect("run route-plan");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let mut commands = Vec::new();
    for host in plan["hosts"].as_array().unwrap() {
        for field in ["guard", "preflight"] {
            commands.extend(
                host[field]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|c| c.as_str())
                    .map(String::from),
            );
        }
        for action in host["actions"].as_array().unwrap() {
            for field in ["apply", "rollback"] {
                if let Some(command) = action[field].as_str() {
                    commands.push(command.to_string());
                }
            }
        }
    }
    assert!(commands.len() > 10);
    let commands_path = dir.join("commands.json");
    fs::write(&commands_path, serde_json::to_vec(&commands).unwrap()).unwrap();
    let checker = dir.join("check.ps1");
    fs::write(
        &checker,
        r#"param([string]$Path)
$failures = @()
foreach ($text in (Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json)) {
    $tokens = $null; $errors = $null
    $ast = [Management.Automation.Language.Parser]::ParseInput($text, [ref]$tokens, [ref]$errors)
    if ($errors.Count) { $failures += "PARSE: $text"; continue }
    foreach ($call in $ast.FindAll({ param($n) $n -is [Management.Automation.Language.CommandAst] }, $true)) {
        $name = $call.GetCommandName()
        if (-not $name -or $name -notmatch '^[A-Za-z]+-[A-Za-z]+$') { continue }
        $cmd = Get-Command -Name $name -ErrorAction SilentlyContinue
        if (-not $cmd) { $failures += "CMDLET $name"; continue }
        $known = @($cmd.Parameters.Values | ForEach-Object { $_.Name; $_.Aliases })
        foreach ($p in $call.CommandElements | Where-Object { $_ -is [Management.Automation.Language.CommandParameterAst] }) {
            if (@($known | Where-Object { $_ -like "$($p.ParameterName)*" }).Count -eq 0) { $failures += "PARAM $name -$($p.ParameterName)" }
        }
    }
}
if ($failures.Count) { $failures; exit 1 }
'OK'
"#,
    )
    .unwrap();
    let check = hidden_command("pwsh")
        .args(["-NoLogo", "-NoProfile", "-NonInteractive", "-File"])
        .arg(&checker)
        .arg(&commands_path)
        .output()
        .expect("run pwsh command checker");
    let _ = fs::remove_dir_all(&dir);
    assert!(
        check.status.success(),
        "{}{}",
        String::from_utf8_lossy(&check.stdout),
        String::from_utf8_lossy(&check.stderr)
    );
}

// Protects: rollback authority comes only from a journal in admin-only storage, every change is
// journaled before it is made, and rollback retires what it undid. Detects: reading a journal from
// a directory a standard user created (forged keys would authorize removals), a network change
// made after a failed journal write, a key left behind by a failed apply, and a retained key that
// lets a second rollback remove state someone else recreated. Needs: Windows with pwsh; the
// network cmdlets, Get-Acl and Set-Acl are replaced by functions, so nothing is changed and no
// elevation is needed. Breadcrumb: src/route_plan.rs journal_guard/owned_apply/owned_rollback.
#[cfg(windows)]
#[test]
fn cli_route_plan_journal_is_protected_write_ahead_and_retired() {
    let (dir, input, policy) = route_plan_inputs("route-plan-journal", &["dtm-super-nuc"]);
    let registry = repo_registry_path();
    let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
        .args([
            "route-plan",
            "--input",
            input.to_str().unwrap(),
            "--policy",
            policy.to_str().unwrap(),
            "--registry",
            registry.to_str().unwrap(),
            "--compact",
        ])
        .output()
        .expect("run route-plan");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let action = plan["hosts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|h| h["host"] == "dtm-p1gen9")
        .and_then(|h| {
            h["actions"]
                .as_array()
                .unwrap()
                .iter()
                .find(|a| a["kind"] == "EdgeAddress" && a["apply"].is_string())
                .cloned()
        })
        .expect("an owned Gen9 edge address");
    let lines = dir.join("lines.json");
    fs::write(
        &lines,
        serde_json::to_vec(&serde_json::json!({ "apply": action["apply"], "rollback": action["rollback"] })).unwrap(),
    )
    .unwrap();
    let script = dir.join("journal.ps1");
    fs::write(&script, JOURNAL_CHECKER).unwrap();
    let run = hidden_command("pwsh")
        .args(["-NoLogo", "-NoProfile", "-NonInteractive", "-File"])
        .arg(&script)
        .arg(&lines)
        .arg(dir.join("root"))
        .output()
        .expect("run pwsh journal checker");
    let _ = fs::remove_dir_all(&dir);
    assert!(
        run.status.success(),
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}

/// Runs one apply/rollback pair against stub cmdlets under a temporary `$env:ProgramData`.
#[cfg(windows)]
const JOURNAL_CHECKER: &str = r#"param([string]$Lines, [string]$Root)
$cmd = Get-Content -Raw -LiteralPath $Lines | ConvertFrom-Json
$failures = [Collections.Generic.List[string]]::new()
$script:calls = [Collections.Generic.List[string]]::new()
$script:present = $false
$script:failNew = $false
function Get-NetIPInterface { [pscustomobject]@{ Dhcp = 'Disabled'; Forwarding = 'Disabled' } }
function Get-NetIPAddress { if ($script:present) { [pscustomobject]@{ PrefixLength = 30 } } }
function New-NetIPAddress { $script:calls.Add('New'); if ($script:failNew) { throw 'stub New-NetIPAddress failure' }; $script:present = $true }
function Remove-NetIPAddress { $script:calls.Add('Remove'); $script:present = $false }
function Invoke-Line([string]$Text) { try { & ([scriptblock]::Create($Text)) | Out-Null; $null } catch { $_.Exception.Message } }
function Get-Journal { @(Get-ChildItem -LiteralPath (Join-Path $env:ProgramData 'pcai-usb4\route-plan') -Filter *.journal -ErrorAction SilentlyContinue | Get-Content) }
$key = [regex]::Match($cmd.apply, "Value '(EdgeAddress\|[^']+)'").Groups[1].Value
$journalName = [regex]::Match($cmd.apply, "([0-9a-f]{16}\.journal)").Groups[1].Value
if (-not $key -or -not $journalName) { 'no journal key or name in the apply line'; exit 1 }

# A directory a standard user made: the journal in it must be neither trusted nor written.
$env:ProgramData = Join-Path $Root 'forged'
$forged = New-Item -ItemType Directory -Path (Join-Path $env:ProgramData 'pcai-usb4\route-plan') -Force
Set-Content -LiteralPath (Join-Path $forged.FullName $journalName) -Value $key
$script:present = $true
$m = Invoke-Line $cmd.rollback
if ($m -notlike 'Journal storage*') { $failures.Add("forged rollback was not refused: $m") }
$script:present = $false
$m = Invoke-Line $cmd.apply
if ($m -notlike 'Journal storage*') { $failures.Add("forged apply was not refused: $m") }
if ($script:calls.Count) { $failures.Add("forged storage reached the network cmdlets: $($script:calls -join ',')") }

# Trusted storage: ownership and ACL reads report Administrators and SYSTEM only.
function Get-Acl {
    $s = [Security.AccessControl.DirectorySecurity]::new()
    $s.SetOwner([Security.Principal.SecurityIdentifier]'S-1-5-32-544')
    foreach ($id in 'S-1-5-32-544', 'S-1-5-18') {
        $s.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]$id, 'FullControl', 'ContainerInherit, ObjectInherit', 'None', 'Allow'))
    }
    $s
}
function Set-Acl { }

$env:ProgramData = Join-Path $Root 'trusted'
New-Item -ItemType Directory -Path $env:ProgramData | Out-Null
$script:present = $false; $script:calls.Clear()
$m = Invoke-Line $cmd.apply
if ($m) { $failures.Add("apply failed: $m") }
if ((Get-Journal) -notcontains $key -or ($script:calls -join ',') -ne 'New') { $failures.Add("apply did not journal and create: $((Get-Journal) -join ';') / $($script:calls -join ',')") }
$m = Invoke-Line $cmd.rollback
if ($m) { $failures.Add("rollback failed: $m") }
if ((Get-Journal) -contains $key -or $script:present) { $failures.Add('rollback did not remove the address and retire its key') }
$script:present = $true; $script:calls.Clear()   # someone else recreates the address
$m = Invoke-Line $cmd.rollback
if ($m -or $script:calls.Count -or -not $script:present) { $failures.Add("second rollback touched recreated state: $m / $($script:calls -join ',')") }

# A failed journal write must stop apply before any change.
$env:ProgramData = Join-Path $Root 'nowrite'
New-Item -ItemType Directory -Path $env:ProgramData | Out-Null
function Add-Content { throw 'stub journal write failure' }
$script:present = $false; $script:calls.Clear()
$m = Invoke-Line $cmd.apply
if ($m -notlike '*journal write failure*' -or $script:calls.Count) { $failures.Add("apply changed state without a journal record: $m / $($script:calls -join ',')") }
Remove-Item function:Add-Content

# A failed change whose state is still absent retires its key.
$env:ProgramData = Join-Path $Root 'failnew'
New-Item -ItemType Directory -Path $env:ProgramData | Out-Null
$script:failNew = $true; $script:present = $false; $script:calls.Clear()
$m = Invoke-Line $cmd.apply
if ($m -notlike '*New-NetIPAddress failure*' -or (Get-Journal) -contains $key) { $failures.Add("failed apply left its key: $m / $((Get-Journal) -join ';')") }

if ($failures.Count) { $failures; exit 1 }
'OK'
"#;
// Protects: unregister and --adapter stay local and scoped. Detects: unregister accepting a
// remote --computer or register-only flags, and --adapter leaking into read-only commands.
// Needs: repository registry only; every case is refused before any adapter query or Apply.
// Breadcrumb: src/main.rs unregister dispatch and --adapter gate.
#[test]
fn cli_unregister_and_adapter_selector_are_scoped() {
    let registry = repo_registry_path();
    let registry = registry.to_str().unwrap();
    for args in [
        vec!["unregister", "--computer", "minisforum", "--registry", registry],
        vec!["unregister", "--assign-pairwise-ipv4", "--registry", registry],
        vec!["status", "--adapter", "02-00-00-0B-00-19", "--registry", registry],
        vec!["identity", "--adapter", "02-00-00-0B-00-19", "--registry", registry],
    ] {
        let output = hidden_command(env!("CARGO_BIN_EXE_pcai-tb"))
            .args(&args)
            .output()
            .expect("run scoped refusal");
        assert!(!output.status.success(), "{args:?} succeeded");
        assert!(output.stdout.is_empty(), "{args:?} emitted output");
    }
}
