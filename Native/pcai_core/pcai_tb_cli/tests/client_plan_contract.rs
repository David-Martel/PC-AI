//! Detect transport-argument widening and false admission in the offline native client.
//! Review register: client-plan-review.json; guide: docs/usb4/native-peer-client.md.
use pcai_tb::client_plan::{
    build_plan, parse_cli_args, parse_policy, ClientCommand, ClientPlatform, ClientPolicy, ClientProtocol,
    InterfaceDeclaration, MAX_POLICY_BYTES,
};
use serde_json::Value;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

fn policy(protocol: ClientProtocol) -> ClientPolicy {
    ClientPolicy {
        schema: 1,
        client_platform: ClientPlatform::Windows,
        protocol,
        client_executable: format!(
            r"C:\Fixture\{}.exe",
            if protocol == ClientProtocol::Ssh { "ssh" } else { "sftp" }
        ),
        account: "fixture".into(),
        host_key_alias: "fixture-peer".into(),
        identity_file: r"C:\Fixture\identity key".into(),
        known_hosts_file: r"C:\Fixture\known hosts".into(),
        local_interface: InterfaceDeclaration {
            id: "11111111-2222-3333-4444-555555555555".into(),
            mac: "02-00-00-00-00-01".into(),
            index: 33,
        },
        source_address: "169.254.12.34".into(),
        peer_address: "169.254.56.78".into(),
        sftp_ssh_executable: if protocol == ClientProtocol::Sftp {
            Some(r"C:\Fixture\ssh.exe".into())
        } else {
            None
        },
        sftp_batch_file: None,
    }
}

fn option(arguments: &[String], value: &str) -> bool {
    arguments.windows(2).any(|pair| pair[0] == "-o" && pair[1] == value)
}

fn cli(arguments: &[&str]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_pcai-usb4-client"))
        .args(arguments)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    let other = sender.clone();
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stdout.take(65_537).read_to_end(&mut bytes).map(|_| bytes);
        let _ = sender.send((true, result));
    });
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stderr.take(65_537).read_to_end(&mut bytes).map(|_| bytes);
        let _ = other.send((false, result));
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            stop_owned_child(&mut child);
            panic!("Owned offline client exceeded its five-second fixture deadline");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let mut captured = [Vec::new(), Vec::new()];
    for _ in 0..2 {
        let (is_stdout, result) = receiver
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .expect("Owned client pipes did not close within the absolute fixture deadline");
        let bytes = result.unwrap();
        assert!(
            bytes.len() <= 65_536,
            "Owned fixture output exceeded its capture budget"
        );
        captured[usize::from(!is_stdout)] = bytes;
    }
    Output {
        status,
        stdout: std::mem::take(&mut captured[0]),
        stderr: std::mem::take(&mut captured[1]),
    }
}

fn stop_owned_child(child: &mut Child) {
    if child.try_wait().unwrap().is_none() {
        let _ = child.kill();
    }
    let until = Instant::now() + Duration::from_secs(1);
    while child.try_wait().unwrap().is_none() {
        assert!(Instant::now() < until, "Owned fixture cleanup remains unqualified");
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn fixture_root(directory: &Path) -> PathBuf {
    let canonical = fs::canonicalize(directory).unwrap();
    #[cfg(windows)]
    {
        let text = canonical.to_str().unwrap();
        if let Some(local) = text.strip_prefix(r"\\?\") {
            assert!(
                local.as_bytes().get(1) == Some(&b':'),
                "Fixture requires an ordinary local drive"
            );
            return PathBuf::from(local);
        }
    }
    canonical
}

fn write_policy(directory: &Path, name: &str) -> std::path::PathBuf {
    let path = fixture_root(directory).join(name);
    fs::write(&path, serde_json::to_vec(&policy(ClientProtocol::Ssh)).unwrap()).unwrap();
    path
}

// Protects: explicit source, account and isolated pin/config transport boundaries.
// Detects: omitted/overridden options, implicit agent/config trust or a shell command.
// Needs: pure fixture only. Breadcrumb: Native/pcai_core/pcai_tb_cli/src/client_plan.rs#build_plan.
#[test]
fn ssh_ipv4_retains_exact_source_and_every_isolation_guard() {
    let plan = build_plan(&policy(ClientProtocol::Ssh)).unwrap();
    assert_eq!(plan.arguments[0], "-4");
    assert_eq!(plan.arguments[1..3], ["-F", "NUL"]);
    assert_eq!(plan.arguments.last().unwrap(), "fixture@169.254.56.78");
    let bind = plan.arguments.iter().position(|argument| argument == "-b").unwrap();
    assert_eq!(plan.arguments[bind + 1], "169.254.12.34");
    for guard in [
        "BatchMode=yes",
        "IdentitiesOnly=yes",
        "IdentityAgent=none",
        "StrictHostKeyChecking=yes",
        "HostKeyAlias=fixture-peer",
        "GlobalKnownHostsFile=NUL",
        "ProxyCommand=none",
        "ProxyJump=none",
        "ControlMaster=no",
        "ControlPath=none",
        "PermitLocalCommand=no",
        "ClearAllForwardings=yes",
        "ForwardAgent=no",
        "ForwardX11=no",
        "KnownHostsCommand=none",
        "UpdateHostKeys=no",
        "CanonicalizeHostname=no",
        "ConnectTimeout=3",
        "ConnectionAttempts=1",
        "Port=22",
    ] {
        assert!(option(&plan.arguments, guard), "Missing explicit guard: {guard}");
    }
    assert!(!plan.arguments.iter().any(|argument| argument.starts_with("HostName=")));
    assert_eq!(plan.state, "OfflinePlan");
    assert_eq!(plan.local_inspection, "NotCollected");
    assert_eq!(plan.peer_authentication, "NotCollected");
    assert_eq!(plan.executable_identity, "NotVerified");
    assert!(!plan.execution_supported && !plan.inspection_supported);
}

// Protects: family/scope grammar for raw SSH and bracketed SFTP destinations.
// Detects: stale/missing zones, IPv4 forced flags or HostName percent interpolation.
// Needs: pure fixtures only. Breadcrumb: Native/pcai_core/pcai_tb_cli/src/client_plan.rs#build_plan.
#[test]
fn ipv6_scope_and_protocol_specific_destination_grammar_are_exact() {
    for protocol in [ClientProtocol::Ssh, ClientProtocol::Sftp] {
        let mut request = policy(protocol);
        request.source_address = "fe80::1%33".into();
        request.peer_address = "fe80::2".into();
        let plan = build_plan(&request).unwrap();
        assert_eq!(plan.arguments[0], "-6");
        assert_eq!(plan.source_address, "fe80::1%33");
        assert_eq!(plan.peer_address, "fe80::2%33");
        if protocol == ClientProtocol::Ssh {
            assert_eq!(plan.arguments.last().unwrap(), "fixture@fe80::2%33");
        } else {
            assert_eq!(plan.arguments.last().unwrap(), "fixture@[fe80::2%33]");
            assert!(option(&plan.arguments, "BindAddress=fe80::1%33"));
            assert!(!plan.arguments.iter().any(|argument| argument == "-b"));
        }
    }
}

// Protects: SFTP batch grammar separately from source binding.
// Detects: using SFTP -b for a source address or percent-expanding a literal batch path.
// Needs: pure fixtures. Breadcrumb: Native/pcai_core/pcai_tb_cli/src/client_plan.rs#build_plan.
#[test]
fn sftp_source_binding_is_an_option_and_batch_file_stays_literal() {
    let mut request = policy(ClientProtocol::Sftp);
    request.sftp_batch_file = Some(r"C:\Fixture\batch 100%.txt".into());
    let plan = build_plan(&request).unwrap();
    assert!(option(&plan.arguments, "BindAddress=169.254.12.34"));
    let helper = plan.arguments.iter().position(|argument| argument == "-S").unwrap();
    assert_eq!(plan.arguments[helper + 1], r"C:\Fixture\ssh.exe");
    assert_eq!(plan.ssh_helper_identity, "NotVerified");
    let indices: Vec<_> = plan
        .arguments
        .iter()
        .enumerate()
        .filter(|(_, value)| *value == "-b")
        .collect();
    assert_eq!(indices.len(), 1);
    assert_eq!(plan.arguments[indices[0].0 + 1], r"C:\Fixture\batch 100%.txt");
    assert_eq!(plan.arguments.last().unwrap(), "fixture@169.254.56.78");
    request.protocol = ClientProtocol::Ssh;
    request.sftp_ssh_executable = None;
    request.client_executable = r"C:\Fixture\ssh.exe".into();
    assert!(build_plan(&request).is_err());
    let mut missing = policy(ClientProtocol::Sftp);
    missing.sftp_ssh_executable = None;
    assert!(build_plan(&missing).is_err());
    let mut unsafe_helper = policy(ClientProtocol::Sftp);
    unsafe_helper.sftp_ssh_executable = Some(r"C:\Fixture\powershell.exe".into());
    assert!(build_plan(&unsafe_helper).is_err());
}

// Protects: one literal identity/pin path through OpenSSH's config tokenizer/token expansion.
// Detects: unquoted space splitting, lost backslashes, %u expansion or premature -i stat.
// Needs: pure fixtures only. Breadcrumb: Native/pcai_core/pcai_tb_cli/src/client_plan.rs#quoted_option_path.
// Primary OpenSSH ssh.c case i stats before token expansion; IdentityFile uses -o instead.
#[test]
fn identity_and_pin_paths_preserve_spaces_percent_and_apostrophes() {
    let mut request = policy(ClientProtocol::Ssh);
    request.identity_file = r"C:\Peer 100%\O'Connor\%u\identity key".into();
    request.known_hosts_file = r"C:\Peer 100%\O'Connor\known hosts".into();
    let plan = build_plan(&request).unwrap();
    assert!(option(
        &plan.arguments,
        r#"IdentityFile="C:\\Peer 100%%\\O'Connor\\%%u\\identity key""#
    ));
    assert!(option(
        &plan.arguments,
        r#"UserKnownHostsFile="C:\\Peer 100%%\\O'Connor\\known hosts""#
    ));
    assert!(!plan.arguments.iter().any(|argument| argument == "-i"));
}

// Protects: POSIX rendering from Windows null-sink/path semantics.
// Detects: NUL use on POSIX, scope added to ULA or path rewriting.
// Needs: pure declared POSIX policy; not Linux runtime admission. Breadcrumb: Native/pcai_core/pcai_tb_cli/src/client_plan.rs#build_plan.
#[test]
fn posix_plan_uses_explicit_null_config_and_unscoped_ula() {
    let mut request = policy(ClientProtocol::Ssh);
    request.client_platform = ClientPlatform::Posix;
    request.client_executable = "/usr/bin/ssh".into();
    request.identity_file = "/home/fixture/identity 100%".into();
    request.known_hosts_file = "/home/fixture/known hosts".into();
    request.local_interface.id = "thunderbolt0".into();
    request.source_address = "fd12::1".into();
    request.peer_address = "fd12::2".into();
    let plan = build_plan(&request).unwrap();
    assert_eq!(plan.arguments[1..3], ["-F", "/dev/null"]);
    assert!(option(&plan.arguments, "GlobalKnownHostsFile=/dev/null"));
    assert!(option(
        &plan.arguments,
        r#"IdentityFile="/home/fixture/identity 100%%""#
    ));
    assert_eq!(plan.arguments.last().unwrap(), "fixture@fd12::2");
    assert!(!plan.execution_supported && !plan.inspection_supported);
    request.protocol = ClientProtocol::Sftp;
    request.client_executable = "/usr/bin/sftp".into();
    request.sftp_ssh_executable = Some("/usr/bin/ssh".into());
    let sftp = build_plan(&request).unwrap();
    assert_eq!(sftp.arguments.last().unwrap(), "fixture@[fd12::2]");
    assert_eq!(sftp.ssh_helper_identity, "NotVerified");
}

// Protects: explicit interface/address declarations and option boundaries.
// Detects: option injection, unsafe endpoints/paths, stale zones and ambiguous source identity.
// Needs: synthetic values only. Breadcrumb: Native/pcai_core/pcai_tb_cli/src/client_plan.rs#validate_client_path.
#[test]
fn unsafe_declarations_refuse_instead_of_widening_a_plan() {
    let good = policy(ClientProtocol::Ssh);
    for source in [
        "1.1.1.1",
        "127.0.0.1",
        "169.254.12.34/16",
        "localhost",
        "fe80::1%9",
        "::ffff:169.254.12.34",
    ] {
        let mut bad = good.clone();
        bad.source_address = source.into();
        assert!(build_plan(&bad).is_err(), "Unexpected address admission: {source}");
    }
    for path in [
        r"relative\identity",
        r"\\host\share\identity",
        r"C:\Fixture\..\identity",
        r"C:\NUL.txt",
        r"C:\Fixture\\identity",
        r"C:\Fixture\identity.",
        r"C:\Fixture\COM¹.txt",
        r"C:\Fixture\com²",
        r"C:\Fixture\COM³.key",
        r"C:\Fixture\LPT¹.txt",
        r"C:\Fixture\lpt²",
        r"C:\Fixture\LPT³.key",
    ] {
        let mut bad = good.clone();
        bad.identity_file = path.into();
        assert!(build_plan(&bad).is_err(), "Unexpected path admission: {path}");
    }
    let mut bad = good.clone();
    bad.identity_file = format!(r"C:\Fixture\{}{{HOME}}\identity", '$');
    assert!(build_plan(&bad).is_err());
    let mut bad = good.clone();
    bad.account = "-oProxyCommand=bad".into();
    assert!(build_plan(&bad).is_err());
    let mut bad = good.clone();
    bad.host_key_alias = "peer\nProxyCommand=bad".into();
    assert!(build_plan(&bad).is_err());
    let mut bad = good.clone();
    bad.peer_address = bad.source_address.clone();
    assert!(build_plan(&bad).is_err());
    let mut bad = good.clone();
    bad.peer_address = "fe80::2".into();
    assert!(build_plan(&bad).is_err());
    let mut bad = good.clone();
    bad.local_interface.index = 0;
    assert!(build_plan(&bad).is_err());
    let mut bad = good.clone();
    bad.local_interface.id = "00000000-0000-0000-0000-000000000000".into();
    assert!(build_plan(&bad).is_err());
    let mut bad = good.clone();
    bad.local_interface.mac = "FF-FF-FF-FF-FF-FF".into();
    assert!(build_plan(&bad).is_err());
    let mut bad = good.clone();
    bad.local_interface.mac = "+2-00-00-00-00-01".into();
    assert!(build_plan(&bad).is_err());
    let mut bad = good;
    bad.client_executable = r"C:\Fixture\powershell.exe".into();
    assert!(build_plan(&bad).is_err());
}

// Protects: typed, closed bounded policy parsing.
// Detects: duplicate/unknown fields, coercive numbers, malformed UTF-8 and excess input.
// Needs: serialized fixtures only. Breadcrumb: Native/pcai_core/pcai_tb_cli/src/client_plan.rs#parse_policy.
#[test]
fn policy_json_is_bounded_typed_and_closed() {
    let encoded = serde_json::to_vec(&policy(ClientProtocol::Ssh)).unwrap();
    assert!(parse_policy(&encoded).is_ok());
    let duplicate =
        String::from_utf8(encoded.clone())
            .unwrap()
            .replacen("\"schema\":1", "\"schema\":1,\"schema\":1", 1);
    assert!(parse_policy(duplicate.as_bytes()).is_err());
    assert!(parse_policy(&[0xff]).is_err());
    assert!(parse_policy(&vec![b' '; MAX_POLICY_BYTES + 1]).is_err());
    for value in [
        serde_json::json!("1"),
        serde_json::json!(true),
        serde_json::json!(-1),
        serde_json::json!(1.0),
    ] {
        let mut invalid: Value = serde_json::from_slice(&encoded).unwrap();
        invalid["local_interface"]["index"] = value;
        assert!(parse_policy(&serde_json::to_vec(&invalid).unwrap()).is_err());
    }
    let mut unknown: Value = serde_json::from_slice(&encoded).unwrap();
    unknown["execute"] = true.into();
    assert!(parse_policy(&serde_json::to_vec(&unknown).unwrap()).is_err());
}

// Protects: the actual manual parser's offline-only action boundary.
// Detects: execution/inspection acceptance, duplicate policy and unknown client options.
// Needs: pure parser calls. Breadcrumb: Native/pcai_core/pcai_tb_cli/src/client_plan.rs#parse_cli_args.
#[test]
fn parser_rejects_unavailable_actions_and_keeps_plan_flags_offline() {
    assert_eq!(parse_cli_args(&[]).unwrap(), ClientCommand::Help);
    for flag in ["-h", "--help"] {
        assert_eq!(parse_cli_args(&[flag.into()]).unwrap(), ClientCommand::Help);
    }
    for mode in ["inspect-local", "execute", "apply"] {
        assert!(parse_cli_args(&[mode.into(), "--policy".into(), "missing".into()]).is_err());
    }
    for flag in ["--dry-run", "--what-if"] {
        let args = ["plan", "--policy", "missing", flag].map(str::to_owned);
        assert!(matches!(parse_cli_args(&args).unwrap(), ClientCommand::Plan { .. }));
    }
    for args in [
        vec!["plan"],
        vec!["plan", "--policy"],
        vec!["plan", "--policy", "one", "--policy", "two"],
        vec!["plan", "--policy", "one", "--execute"],
        vec!["plan", "--policy", "one", "--ssh-option", "ProxyCommand=bad"],
    ] {
        assert!(parse_cli_args(&args.into_iter().map(str::to_owned).collect::<Vec<_>>()).is_err());
    }
}

// Protects: executable CLI integration and truthful unqualified JSON.
// Detects: parser/builder disconnected from main, hidden writes or invented admission.
// Needs: only private temporary policy file and the built client, no keys/NIC/network.
// Breadcrumb: Native/pcai_core/pcai_tb_cli/src/bin/pcai-usb4-client.rs#run.
#[test]
fn actual_binary_plans_offline_without_opening_declared_credentials_or_program() {
    let directory = tempfile::tempdir().unwrap();
    let path = write_policy(directory.path(), "policy 100%.json");
    for extra in [None, Some("--dry-run"), Some("--what-if")] {
        let mut args = vec!["plan", "--policy", path.to_str().unwrap()];
        if let Some(extra) = extra {
            args.push(extra);
        }
        let result = cli(&args);
        assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
        let plan: Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(plan["state"], "OfflinePlan");
        assert_eq!(plan["arguments"][0], "-4");
        assert_eq!(plan["peer_authentication"], "NotCollected");
        assert_eq!(plan["local_inspection"], "NotCollected");
        assert_eq!(plan["execution_supported"], false);
        assert_eq!(plan["inspection_supported"], false);
        assert!(plan.get("shell_command").is_none());
    }
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
}

// Protects: help/unavailable-action behavior before policy access.
// Detects: reading missing policy while helping or silent success for unavailable modes.
// Needs: the built client only; no policy or host. Breadcrumb: Native/pcai_core/pcai_tb_cli/src/bin/pcai-usb4-client.rs#run.
#[test]
fn actual_help_is_offline_and_unavailable_actions_exit_nonzero() {
    for args in [vec![], vec!["--help"], vec!["plan", "--policy", "missing", "--help"]] {
        let result = cli(&args);
        assert!(result.status.success());
        assert!(String::from_utf8_lossy(&result.stdout).contains("offline"));
    }
    for action in ["inspect-local", "execute"] {
        let result = cli(&[action, "--policy", "missing"]);
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
        assert!(String::from_utf8_lossy(&result.stderr).contains("unavailable"));
    }
}

// Protects: bounded explicit file input and no directory-as-policy admission.
// Detects: oversized input or a directory admitted as an explicit policy.
// Needs: private regular/directory fixtures. Breadcrumb: Native/pcai_core/pcai_tb_cli/src/bin/pcai-usb4-client.rs#read_policy_file.
#[test]
fn actual_binary_refuses_overbudget_and_nonregular_policy_files() {
    let directory = tempfile::tempdir().unwrap();
    let large = directory.path().join("large.json");
    fs::write(&large, vec![b' '; MAX_POLICY_BYTES + 1]).unwrap();
    for path in [large.as_path(), directory.path()] {
        let result = cli(&["plan", "--policy", path.to_str().unwrap()]);
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
    }
}

// Protects: rejection of a FIFO before a potentially blocking policy open.
// Detects: the initial new-reader File::open-before-regular-file guard failure.
// Needs: POSIX mkfifo and only the owned built client; Windows does not compile this selector.
// Breadcrumb: Native/pcai_core/pcai_tb_cli/src/bin/pcai-usb4-client.rs#read_policy_file.
#[cfg(unix)]
#[test]
fn fifo_policy_is_refused_without_waiting_for_a_writer() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::process::Stdio;
    use std::time::{Duration, Instant};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("policy.fifo");
    let c_path = CString::new(path.as_os_str().as_bytes()).unwrap();
    // SAFETY: c_path is terminated and retained; this creates only our private fixture FIFO.
    assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
    let mut child = Command::new(env!("CARGO_BIN_EXE_pcai-usb4-client"))
        .args(["plan", "--policy"])
        .arg(&path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert_eq!(status.code(), Some(2));
            break;
        }
        if Instant::now() >= deadline {
            stop_owned_child(&mut child);
            panic!("Policy FIFO open blocked instead of refusing before reading.");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

// Protects: ancestor-first no-follow input admission on POSIX.
// Detects: following a stable symlink before opening the selected policy.
// Needs: private POSIX symlink fixtures only; Windows does not compile this selector.
// Breadcrumb: Native/pcai_core/pcai_tb_cli/src/bin/pcai-usb4-client.rs#admit_policy_file.
#[cfg(unix)]
#[test]
fn policy_symlinks_and_regular_file_ancestors_are_refused() {
    use std::os::unix::fs::symlink;
    let directory = tempfile::tempdir().unwrap();
    let real = directory.path().join("real");
    fs::create_dir(&real).unwrap();
    let policy = write_policy(&real, "policy.json");
    let link = directory.path().join("link");
    symlink(&real, &link).unwrap();
    for path in [link.join("policy.json"), policy.join("not-a-directory.json")] {
        let result = cli(&["plan", "--policy", path.to_str().unwrap()]);
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
    }
}

// Protects: no-child behavior even if an attempted client launch error is ignored.
// Detects: invoking a real declared program during offline planning; the control
// first proves the owned sentinel runs and writes its marker.
// Needs: POSIX shell, private fixture files and the built planner; Windows does not compile it.
// Breadcrumb: Native/pcai_core/pcai_tb_cli/src/bin/pcai-usb4-client.rs#run.
#[cfg(unix)]
#[test]
fn offline_plan_does_not_launch_a_proven_owned_client_sentinel() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let root = fixture_root(directory.path());
    let program = root.join("ssh");
    let marker = root.join("execution-marker");
    fs::write(
        &program,
        b"#!/bin/sh\nprintf started > \"$PCAI_CLIENT_FIXTURE_MARKER\"\nexit 73\n",
    )
    .unwrap();
    fs::set_permissions(&program, fs::Permissions::from_mode(0o700)).unwrap();
    let mut control_child = Command::new(&program)
        .env("PCAI_CLIENT_FIXTURE_MARKER", &marker)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let control_deadline = Instant::now() + Duration::from_secs(5);
    let control = loop {
        if let Some(status) = control_child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= control_deadline {
            stop_owned_child(&mut control_child);
            panic!("Owned positive sentinel exceeded its deadline");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(control.code(), Some(73));
    assert_eq!(fs::read(&marker).unwrap(), b"started");
    fs::remove_file(&marker).unwrap();
    let mut request = policy(ClientProtocol::Ssh);
    request.client_platform = ClientPlatform::Posix;
    request.client_executable = program.to_str().unwrap().into();
    request.identity_file = root.join("missing-key").to_str().unwrap().into();
    request.known_hosts_file = root.join("missing-pin").to_str().unwrap().into();
    request.local_interface.id = "thunderbolt0".into();
    let path = root.join("sentinel-policy.json");
    fs::write(&path, serde_json::to_vec(&request).unwrap()).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_pcai-usb4-client"))
        .args(["plan", "--policy"])
        .arg(&path)
        .env("PCAI_CLIENT_FIXTURE_MARKER", &marker)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success());
            break;
        }
        if Instant::now() >= deadline {
            stop_owned_child(&mut child);
            panic!("Owned sentinel-plan fixture exceeded its deadline");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        !marker.exists(),
        "Offline planning launched the declared client program"
    );
}
