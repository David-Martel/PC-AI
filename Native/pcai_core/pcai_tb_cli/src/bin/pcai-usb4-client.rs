//! Render bounded offline peer-client plans; no live inspector or executor is installed.
use anyhow::{ensure, Context, Result};
use pcai_tb::client_plan::{
    build_plan, parse_cli_args, parse_policy, validate_client_path, ClientCommand, ClientPlatform, MAX_POLICY_BYTES,
};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

const HELP: &str = "pcai-usb4-client: offline, declaration-only SSH/SFTP argv planner
Usage: pcai-usb4-client plan --policy ABSOLUTE_FILE [--dry-run|--what-if]
       pcai-usb4-client -h|--help
Default plan, dry-run and what-if perform no NIC/route/credential inspection,
subprocess launch, peer access or configuration writes. Only the explicitly
named bounded policy file is read. Output is JSON argv, never a shell string.
inspect-local, execute and apply are unavailable. OfflinePlan is not permission
to execute or evidence of source, pin, program or peer readiness.
";

fn native_policy_platform() -> ClientPlatform {
    if cfg!(windows) {
        ClientPlatform::Windows
    } else {
        ClientPlatform::Posix
    }
}

fn admit_policy_file(path: &Path) -> Result<()> {
    let text = path.to_str().context("Policy path must be valid Unicode")?;
    validate_client_path(text, native_policy_platform())?;
    #[cfg(windows)]
    {
        use windows_sys::Win32::Storage::FileSystem::GetDriveTypeW;
        let root: Vec<u16> = text[..3].encode_utf16().chain(Some(0)).collect();
        // SAFETY: root is a validated drive-root string with a retained UTF-16 terminator.
        let drive_type = unsafe { GetDriveTypeW(root.as_ptr()) };
        ensure!(
            matches!(drive_type, 3 | 6),
            "Policy requires an admitted local fixed/RAM drive"
        );
    }
    let mut cursor = PathBuf::new();
    let mut components = path.components().peekable();
    while let Some(component) = components.next() {
        cursor.push(component.as_os_str());
        if matches!(component, Component::Prefix(_)) {
            continue;
        }
        ensure!(
            !matches!(component, Component::ParentDir | Component::CurDir),
            "Policy traversal is refused"
        );
        let metadata = fs::symlink_metadata(&cursor).context("Policy component is unavailable")?;
        ensure!(!metadata.file_type().is_symlink(), "Policy symbolic links are refused");
        if components.peek().is_none() {
            ensure!(metadata.is_file(), "Policy leaf must be a regular file");
        } else {
            ensure!(metadata.is_dir(), "Policy ancestor must be a directory");
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            ensure!(
                metadata.file_attributes() & (0x400 | 0x1000 | 0x44_0000) == 0,
                "Policy reparse, offline or recall paths are refused"
            );
        }
    }
    Ok(())
}

fn read_policy_file(path: &Path) -> Result<Vec<u8>> {
    admit_policy_file(path)?;
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
    }
    let file = options.open(path).context("Explicit policy cannot be opened")?;
    let metadata = file.metadata().context("Explicit policy metadata is unavailable")?;
    ensure!(
        metadata.is_file() && metadata.len() <= MAX_POLICY_BYTES as u64,
        "Policy must be a bounded regular file"
    );
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        ensure!(
            metadata.file_attributes() & (0x400 | 0x1000 | 0x44_0000) == 0,
            "Opened policy has unsafe attributes"
        );
    }
    let mut bytes = Vec::new();
    file.take(MAX_POLICY_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .context("Explicit policy cannot be read")?;
    ensure!(bytes.len() <= MAX_POLICY_BYTES, "Policy exceeds the byte budget");
    Ok(bytes)
}

fn run() -> Result<()> {
    let arguments: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|argument| {
            argument
                .into_string()
                .map_err(|_| anyhow::anyhow!("Arguments must be valid Unicode"))
        })
        .collect::<Result<_>>()?;
    match parse_cli_args(&arguments)? {
        ClientCommand::Help => std::io::stdout().lock().write_all(HELP.as_bytes())?,
        ClientCommand::Plan { policy_path } => {
            let policy = parse_policy(&read_policy_file(&policy_path)?)?;
            let plan = build_plan(&policy)?;
            let mut output = serde_json::to_vec(&plan)?;
            output.push(b'\n');
            std::io::stdout().lock().write_all(&output)?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // Keep diagnostics bounded and prevent control-character terminal injection.
            let reason: String = error
                .to_string()
                .chars()
                .take(512)
                .flat_map(char::escape_default)
                .collect();
            eprintln!("pcai-usb4-client: {reason}");
            ExitCode::from(2)
        }
    }
}
