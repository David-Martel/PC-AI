//! Publish bounded-name metadata records atomically within a trusted runtime directory.
use anyhow::Result;
use std::path::Path;

fn valid_name(name: &str) -> bool {
    matches!(
        name,
        "observe" | "observe-all" | "qualify" | "activate-nm" | "reconcile-networkd"
    )
}

/// Atomically publish a status record in an existing trusted Linux directory.
///
/// Production callers must pass zero as `expected_uid` to require root ownership.
/// Names are fixed operation identifiers; the stored filename appends `.json`.
pub fn publish(directory: &Path, name: &str, bytes: &[u8], expected_uid: u32) -> Result<()> {
    anyhow::ensure!(valid_name(name), "Unsupported status record name");
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (directory, bytes, expected_uid);
        anyhow::bail!("Status record publication requires Linux")
    }
    #[cfg(target_os = "linux")]
    {
        use anyhow::{ensure, Context};
        use std::{
            ffi::CString,
            fs::OpenOptions,
            io::Write,
            os::{
                fd::AsRawFd,
                unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
            },
        };
        let dir = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(directory)
            .context("Cannot open trusted status directory")?;
        let metadata = dir.metadata()?;
        ensure!(
            metadata.is_dir() && metadata.uid() == expected_uid,
            "Status directory has unexpected owner or type"
        );
        ensure!(
            matches!(metadata.mode() & 0o7777, 0o700 | 0o750),
            "Status directory must have mode 0700 or 0750"
        );
        let target = CString::new(format!("{name}.json"))?;
        let stage_name = format!(".{name}.{}.tmp", std::process::id());
        let stage = CString::new(stage_name.as_str())?;
        let fd = dir.as_raw_fd();
        // fstatat does not follow an existing record symlink; all paths remain relative
        // to the held directory even if its original pathname is moved concurrently.
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        let status = unsafe { libc::fstatat(fd, target.as_ptr(), stat.as_mut_ptr(), libc::AT_SYMLINK_NOFOLLOW) };
        if status == 0 {
            let stat = unsafe { stat.assume_init() };
            ensure!(
                stat.st_mode & libc::S_IFMT == libc::S_IFREG,
                "Existing status target must be a regular file"
            );
        } else {
            let error = std::io::Error::last_os_error();
            ensure!(
                error.kind() == std::io::ErrorKind::NotFound,
                "Cannot inspect status target: {error}"
            );
        }
        let pinned_stage = Path::new("/proc/self/fd").join(fd.to_string()).join(&stage_name);
        // A preexisting stage is foreign to this invocation: create_new fails and
        // never enters the cleanup block, preserving its contents.
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(pinned_stage)
            .context("Cannot exclusively create status stage")?;
        let mut published = false;
        let result = (|| -> Result<()> {
            file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
            file.write_all(bytes)?;
            file.sync_all()?;
            let renamed = unsafe { libc::renameat(fd, stage.as_ptr(), fd, target.as_ptr()) };
            if renamed != 0 {
                return Err(std::io::Error::last_os_error()).context("Cannot publish status stage");
            }
            published = true;
            dir.sync_all().context("Cannot sync published status directory")?;
            Ok(())
        })();
        // Once rename succeeds there is no stage to remove; never unlink target.
        if result.is_err() && !published {
            let _ = unsafe { libc::unlinkat(fd, stage.as_ptr(), 0) };
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Protect fixed output names; pure boundary test requires no filesystem privileges.
    #[test]
    fn accepts_only_operation_names() {
        for name in ["observe", "observe-all", "qualify", "activate-nm"] {
            assert!(valid_name(name));
        }
        for name in [
            "",
            "../observe",
            "observe.json",
            "/qualify",
            "observe/other",
            "ACTIVATE-NM",
        ] {
            assert!(!valid_name(name));
        }
    }
    #[cfg(target_os = "linux")]
    fn trusted_dir() -> (tempfile::TempDir, u32) {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        (dir, unsafe { libc::geteuid() })
    }
    #[cfg(target_os = "linux")]
    // Protect atomic replacement and private mode; real filesystem fixture, no service privileges required.
    #[test]
    fn publishes_two_complete_records() {
        use std::os::unix::fs::MetadataExt;
        let (dir, uid) = trusted_dir();
        publish(dir.path(), "observe", b"{\"record\":1}", uid).unwrap();
        let path = dir.path().join("observe.json");
        let old = std::fs::File::open(&path).unwrap();
        publish(dir.path(), "observe", b"{\"record\":2}", uid).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"{\"record\":2}");
        use std::io::Read;
        let mut previous = String::new();
        (&old).read_to_string(&mut previous).unwrap();
        assert_eq!(previous, "{\"record\":1}");
        assert_eq!(std::fs::metadata(path).unwrap().mode() & 0o777, 0o600);
    }
    #[cfg(target_os = "linux")]
    // Protect symlink targets, directory trust, and foreign stages; real filesystem negative controls.
    #[test]
    fn refuses_symlinks_permissions_owner_and_existing_stage() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let (dir, uid) = trusted_dir();
        let victim = dir.path().join("victim");
        std::fs::write(&victim, b"preserve").unwrap();
        symlink(&victim, dir.path().join("observe.json")).unwrap();
        assert!(publish(dir.path(), "observe", b"replacement", uid).is_err());
        assert_eq!(std::fs::read(&victim).unwrap(), b"preserve");
        assert!(publish(dir.path(), "qualify", b"x", uid.wrapping_add(1)).is_err());
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o770)).unwrap();
        assert!(publish(dir.path(), "qualify", b"x", uid).is_err());
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let stage = dir.path().join(format!(".qualify.{}.tmp", std::process::id()));
        std::fs::write(&stage, b"foreign").unwrap();
        assert!(publish(dir.path(), "qualify", b"replacement", uid).is_err());
        assert_eq!(std::fs::read(stage).unwrap(), b"foreign");
        assert!(!dir.path().join("qualify.json").exists());
        assert!(publish(dir.path(), "../outside", b"x", uid).is_err());
        let alias = dir.path().join("alias");
        symlink(dir.path(), &alias).unwrap();
        assert!(publish(&alias, "qualify", b"x", uid).is_err());
    }
}
