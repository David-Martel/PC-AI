//! Observe Linux USB4STREAM kernel surfaces without opening a stream.
//!
//! See [Linux USB4STREAM preflight](../../../../docs/usb4/rdma-preview.md#linux-and-mixed-fleets).

use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct StreamPreflight {
    schema: &'static str,
    platform: &'static str,
    host: Option<String>,
    status: &'static str,
    transfer_qualification: &'static str,
    linux: Option<LinuxStreamObservation>,
}

#[derive(Debug, Serialize)]
struct LinuxStreamObservation {
    kernel_release: Option<String>,
    stream_config: StreamConfig,
    configfs_mount: ConfigfsMount,
    stream_control: Visibility,
    stream_module: Visibility,
    usb4net_module: Visibility,
    rxe_module: Visibility,
    siw_module: Visibility,
    device_scan: DeviceScan,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct StreamConfig {
    state: StreamConfigState,
    source: Option<StreamConfigSource>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
enum StreamConfigState {
    BuiltIn,
    Module,
    Disabled,
    Unspecified,
    Unavailable,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
enum StreamConfigSource {
    ProcConfigGzip,
    BootConfig,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
enum ConfigfsMount {
    Mounted,
    Absent,
    UnexpectedType,
    Unavailable,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
enum Visibility {
    Present,
    Absent,
    Unavailable,
    UnexpectedType,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
enum ScanCoverage {
    Complete,
    Unavailable,
    LimitReached,
}

#[derive(Debug, Serialize)]
struct DeviceScan {
    coverage: ScanCoverage,
    character_devices: Vec<String>,
    rejected_entries: usize,
}

/// Report local kernel surface visibility and leave transfer qualification untested.
pub fn observe_stream_preflight() -> StreamPreflight {
    #[cfg(target_os = "linux")]
    let (host, linux) = (linux::hostname(), Some(linux::observe()));
    #[cfg(not(target_os = "linux"))]
    let (host, linux) = (crate::identity::operating_system_hostname(), None);

    StreamPreflight {
        schema: "pcai.usb4-stream-preflight/v1",
        platform: std::env::consts::OS,
        host,
        status: if linux.is_some() {
            "LocalObservation"
        } else {
            "UnsupportedPlatform"
        },
        transfer_qualification: "NotTested",
        linux,
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{
        ConfigfsMount, DeviceScan, LinuxStreamObservation, ScanCoverage, StreamConfig, StreamConfigSource,
        StreamConfigState, Visibility,
    };
    use flate2::bufread::GzDecoder;
    use std::fs::{self, OpenOptions};
    use std::io::{ErrorKind, Read};
    use std::os::unix::fs::{FileTypeExt, OpenOptionsExt};
    use std::path::Path;

    const MAX_DIRECTORY_ENTRIES: usize = 4096;
    const MAX_STREAM_NODES: usize = 64;
    const MAX_KERNEL_LABEL_BYTES: usize = 256;
    const MAX_KERNEL_CONFIG_BYTES: usize = 2 * 1024 * 1024;
    const MAX_MOUNTINFO_BYTES: usize = 2 * 1024 * 1024;
    const STREAM_SYMBOL: &str = "CONFIG_USB4_STREAM";
    const CONFIGFS_PATH: &[u8] = b"/sys/kernel/config";

    fn read_limited(reader: impl Read, limit: usize) -> Option<Vec<u8>> {
        let mut bytes = Vec::new();
        reader.take(limit as u64 + 1).read_to_end(&mut bytes).ok()?;
        (bytes.len() <= limit).then_some(bytes)
    }

    fn read_regular_file(path: &Path, limit: usize) -> Option<Vec<u8>> {
        if !fs::symlink_metadata(path).ok()?.is_file() {
            return None;
        }
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)
            .ok()?;
        if !file.metadata().ok()?.is_file() {
            return None;
        }
        read_limited(file, limit)
    }

    fn read_kernel_label(path: &str) -> Option<String> {
        let bytes = read_regular_file(Path::new(path), MAX_KERNEL_LABEL_BYTES)?;
        let text = std::str::from_utf8(&bytes).ok()?.trim();
        (!text.is_empty()).then(|| text.to_owned())
    }

    pub(super) fn hostname() -> Option<String> {
        read_kernel_label("/proc/sys/kernel/hostname")
    }

    pub(super) fn observe() -> LinuxStreamObservation {
        let kernel_release = read_kernel_label("/proc/sys/kernel/osrelease");
        let stream_config = observe_stream_config(
            Path::new("/proc/config.gz"),
            Path::new("/boot"),
            kernel_release.as_deref(),
        );
        LinuxStreamObservation {
            kernel_release,
            stream_config,
            configfs_mount: observe_configfs_mount(Path::new("/proc/self/mountinfo")),
            stream_control: directory_visibility(Path::new("/sys/kernel/config/thunderbolt/stream")),
            stream_module: directory_visibility(Path::new("/sys/module/thunderbolt_stream")),
            usb4net_module: directory_visibility(Path::new("/sys/module/thunderbolt_net")),
            rxe_module: directory_visibility(Path::new("/sys/module/rdma_rxe")),
            siw_module: directory_visibility(Path::new("/sys/module/siw")),
            device_scan: scan_device_nodes(Path::new("/dev"), MAX_DIRECTORY_ENTRIES),
        }
    }

    fn valid_kernel_release(release: &str) -> bool {
        release.len() <= 64
            && release.as_bytes().first().is_some_and(u8::is_ascii_digit)
            && release
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._+-".contains(&byte))
    }

    fn parse_stream_config(bytes: &[u8]) -> Option<StreamConfigState> {
        let text = std::str::from_utf8(bytes).ok()?;
        if text.contains('\0') || !text.ends_with('\n') {
            return None;
        }
        let mut stream_state = None;
        let mut saw_symbol = false;
        for line in text.lines().filter(|line| !line.is_empty()) {
            let (symbol, value) =
                if let Some(comment) = line.strip_prefix("# ").filter(|text| text.starts_with("CONFIG_")) {
                    if let Some(symbol) = comment.strip_suffix(" is not set") {
                        (symbol, "n")
                    } else if comment.split_ascii_whitespace().next() == Some(STREAM_SYMBOL) {
                        return None;
                    } else {
                        continue;
                    }
                } else if line.starts_with('#') {
                    continue;
                } else {
                    line.split_once('=')?
                };
            let name = symbol.strip_prefix("CONFIG_")?;
            if name.is_empty() || !name.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_') {
                return None;
            }
            saw_symbol = true;
            if symbol == STREAM_SYMBOL {
                if stream_state.is_some() {
                    return None;
                }
                stream_state = Some(match value {
                    "y" => StreamConfigState::BuiltIn,
                    "m" => StreamConfigState::Module,
                    "n" => StreamConfigState::Disabled,
                    _ => return None,
                });
            }
        }
        saw_symbol.then_some(stream_state.unwrap_or(StreamConfigState::Unspecified))
    }

    fn read_proc_config(path: &Path) -> Option<StreamConfigState> {
        let compressed = read_regular_file(path, MAX_KERNEL_CONFIG_BYTES)?;
        let mut decoder = GzDecoder::new(compressed.as_slice());
        let config = read_limited(&mut decoder, MAX_KERNEL_CONFIG_BYTES)?;
        if !decoder.into_inner().is_empty() {
            return None;
        }
        parse_stream_config(&config)
    }

    fn observe_stream_config(proc_config: &Path, boot: &Path, release: Option<&str>) -> StreamConfig {
        if let Some(state) = read_proc_config(proc_config) {
            return StreamConfig {
                state,
                source: Some(StreamConfigSource::ProcConfigGzip),
            };
        }
        if let Some(release) = release.filter(|release| valid_kernel_release(release)) {
            let path = boot.join(format!("config-{release}"));
            if let Some(state) = read_regular_file(&path, MAX_KERNEL_CONFIG_BYTES)
                .as_deref()
                .and_then(parse_stream_config)
            {
                return StreamConfig {
                    state,
                    source: Some(StreamConfigSource::BootConfig),
                };
            }
        }
        StreamConfig {
            state: StreamConfigState::Unavailable,
            source: None,
        }
    }

    /// Fields of one mountinfo record, validated through the source field.
    struct MountRecord<'a> {
        path: &'a [u8],
        filesystem: &'a [u8],
        /// Fields after the source. The kernel prints one, but a super-option that holds an
        /// unescaped space (a 9p `path=C:\Program Files\...`) splits into several.
        super_options: Vec<&'a [u8]>,
    }

    /// Validates a record through its mount point, separator, filesystem type and source.
    ///
    /// Returns `None` for any record that is damaged before or at those fields.
    fn parse_mount_record(record: &[u8]) -> Option<MountRecord<'_>> {
        let mut fields = record.split(|byte| *byte == b' ');
        // The first six fields are required; optional fields end at a standalone hyphen.
        let mount_id = fields.next()?;
        let parent_id = fields.next()?;
        let mut device = fields.next()?.split(|byte| *byte == b':');
        let root = fields.next()?;
        let path = fields.next()?;
        let options = fields.next()?;
        let decimal = |field: &[u8]| !field.is_empty() && field.iter().all(u8::is_ascii_digit);
        if !decimal(mount_id)
            || !decimal(parent_id)
            || !decimal(device.next()?)
            || !decimal(device.next()?)
            || device.next().is_some()
            // Namespace bind mounts can report a root such as net:[inode].
            || root.is_empty()
            || !path.starts_with(b"/")
            || options.is_empty()
        {
            return None;
        }
        fields.find(|field| *field == b"-")?;
        let filesystem = fields.next()?;
        let source = fields.next()?;
        if filesystem.is_empty() || source.is_empty() {
            return None;
        }
        Some(MountRecord {
            path,
            filesystem,
            super_options: fields.collect(),
        })
    }

    /// Classifies the configfs mount from a complete `/proc/self/mountinfo` listing.
    ///
    /// The configfs record gets every check: it must parse through the source and carry exactly
    /// one non-empty super-options field, and a second record for the mount point is a conflict.
    /// A record for any other mount point is not evidence about configfs, so one that is damaged
    /// or carries extra super-option fields is skipped instead of discarding the listing. A
    /// skipped record that still names the configfs path as a field cannot be proven unrelated
    /// and fails closed. `Absent` additionally needs at least one usable record, so a listing of
    /// nothing but damage stays `None`.
    fn parse_mountinfo(bytes: &[u8]) -> Option<ConfigfsMount> {
        let records = bytes.strip_suffix(b"\n")?;
        let mut mount = None;
        let mut usable_records = 0_usize;
        for record in records.split(|byte| *byte == b'\n') {
            match parse_mount_record(record) {
                Some(parsed) if parsed.path == CONFIGFS_PATH => {
                    let [super_options] = parsed.super_options.as_slice() else {
                        return None;
                    };
                    if super_options.is_empty() || mount.is_some() {
                        return None;
                    }
                    mount = Some(if parsed.filesystem == b"configfs" {
                        ConfigfsMount::Mounted
                    } else {
                        ConfigfsMount::UnexpectedType
                    });
                    usable_records += 1;
                }
                Some(_) => usable_records += 1,
                None if record.split(|byte| *byte == b' ').any(|field| field == CONFIGFS_PATH) => {
                    return None;
                }
                None => {}
            }
        }
        match (mount, usable_records) {
            (Some(mount), _) => Some(mount),
            (None, 0) => None,
            (None, _) => Some(ConfigfsMount::Absent),
        }
    }

    fn observe_configfs_mount(path: &Path) -> ConfigfsMount {
        read_regular_file(path, MAX_MOUNTINFO_BYTES)
            .as_deref()
            .and_then(parse_mountinfo)
            .unwrap_or(ConfigfsMount::Unavailable)
    }

    fn directory_visibility(path: &Path) -> Visibility {
        match fs::symlink_metadata(path) {
            Ok(metadata) if metadata.is_dir() => Visibility::Present,
            Ok(_) => Visibility::UnexpectedType,
            Err(error) if error.kind() == ErrorKind::NotFound => Visibility::Absent,
            Err(_) => Visibility::Unavailable,
        }
    }

    fn is_stream_node_name(name: &str) -> bool {
        name.strip_prefix("tbstream").is_some_and(|index| {
            !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit()) && index.parse::<u32>().is_ok()
        })
    }

    fn scan_device_nodes(root: &Path, max_directory_entries: usize) -> DeviceScan {
        let mut report = DeviceScan {
            coverage: ScanCoverage::Complete,
            character_devices: Vec::new(),
            rejected_entries: 0,
        };
        let Ok(entries) = fs::read_dir(root) else {
            report.coverage = ScanCoverage::Unavailable;
            return report;
        };
        for (index, entry) in entries.enumerate() {
            if index == max_directory_entries {
                report.coverage = ScanCoverage::LimitReached;
                break;
            }
            let Ok(entry) = entry else {
                report.coverage = ScanCoverage::Unavailable;
                continue;
            };
            let file_name = entry.file_name();
            let Some(name) = file_name.to_str().filter(|name| is_stream_node_name(name)) else {
                continue;
            };
            // file_type does not follow symlinks or open the device, which could allocate rings.
            match entry.file_type() {
                Ok(kind) if kind.is_char_device() => {
                    if report.character_devices.len() == MAX_STREAM_NODES {
                        report.coverage = ScanCoverage::LimitReached;
                        break;
                    }
                    report.character_devices.push(name.to_owned());
                }
                Ok(_) => report.rejected_entries += 1,
                Err(_) => report.coverage = ScanCoverage::Unavailable,
            }
        }
        report.character_devices.sort_unstable();
        report
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use flate2::{write::GzEncoder, Compression};
        use std::io::Write;
        use std::os::unix::fs::symlink;

        fn gzip(bytes: &[u8]) -> Vec<u8> {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
            encoder.write_all(bytes).expect("compress fixture");
            encoder.finish().expect("finish fixture gzip")
        }

        // Protects exact CONFIG_USB4_STREAM tri-state evidence. Detects drift from other
        // symbols and false qualifications from invalid/conflicting values. Needs no hardware.
        // Breadcrumb: parse_stream_config.
        #[test]
        fn stream_config_distinguishes_tristate_from_missing_and_invalid_evidence() {
            for (bytes, expected) in [
                (&b"CONFIG_USB4_STREAM=y\n"[..], Some(StreamConfigState::BuiltIn)),
                (b"CONFIG_USB4_STREAM=m\r\n", Some(StreamConfigState::Module)),
                (b"# CONFIG_USB4_STREAM is not set\n", Some(StreamConfigState::Disabled)),
                (b"CONFIG_USB4_STREAM=n\n", Some(StreamConfigState::Disabled)),
                (b"CONFIG_USB4_STREAM_DEBUG=y\n", Some(StreamConfigState::Unspecified)),
                (
                    b"# CONFIG_MTD_NETtel is not set\n",
                    Some(StreamConfigState::Unspecified),
                ),
                (b"# CONFIG_USB4_NET is not set\n", Some(StreamConfigState::Unspecified)),
                (b"CONFIG_USB4_NET=y\n# CONFIG_USB4_STREAM is set\n", None),
                (b"CONFIG_USB4_STREAM=\"y\"\n", None),
                (b"CONFIG_USB4_STREAM=y\nCONFIG_USB4_STREAM=m\n", None),
                (b"CONFIG_USB4_STREAM=y\n# CONFIG_USB4_STREAM is not set\n", None),
                (b"CONFIG_USB4_STREAM=y", None),
                (b"CONFIG_USB4_STREAM=y\n\xff\n", None),
                (b"# commentary only\n", None),
            ] {
                assert_eq!(parse_stream_config(bytes), expected, "{bytes:?}");
            }
        }

        // Protects precedence of running-kernel evidence and release-bound fallback.
        // Detects fallback overriding valid proc evidence or accepting a path-like release.
        // Needs Linux temporary files and gzip encoder. Breadcrumb: observe_stream_config.
        #[test]
        fn stream_config_prefers_proc_and_falls_back_only_to_validated_boot_file() {
            let dir = tempfile::tempdir().expect("create config fixture root");
            let proc = dir.path().join("config.gz");
            let release = "7.2.0-test+1";
            let boot = dir.path().join(format!("config-{release}"));
            fs::write(&boot, b"CONFIG_USB4_STREAM=m\n").expect("write boot config");
            let expected_boot = StreamConfig {
                state: StreamConfigState::Module,
                source: Some(StreamConfigSource::BootConfig),
            };
            assert_eq!(observe_stream_config(&proc, dir.path(), Some(release)), expected_boot);

            fs::write(&proc, gzip(b"CONFIG_USB4_STREAM=y\n")).expect("write proc config");
            assert_eq!(
                observe_stream_config(&proc, dir.path(), Some(release)),
                StreamConfig {
                    state: StreamConfigState::BuiltIn,
                    source: Some(StreamConfigSource::ProcConfigGzip)
                },
            );
            fs::write(&proc, gzip(b"CONFIG_USB4_NET=y\n")).expect("replace proc config");
            assert_eq!(
                observe_stream_config(&proc, dir.path(), Some(release)),
                StreamConfig {
                    state: StreamConfigState::Unspecified,
                    source: Some(StreamConfigSource::ProcConfigGzip)
                },
            );
            fs::write(&proc, b"invalid gzip").expect("replace proc with corrupt data");
            assert_eq!(observe_stream_config(&proc, dir.path(), Some(release)), expected_boot);

            let unavailable = StreamConfig {
                state: StreamConfigState::Unavailable,
                source: None,
            };
            for invalid in ["", "7/tmp", "7\\tmp", "../7.2", "7.2 more", &"7".repeat(65)] {
                assert!(!valid_kernel_release(invalid), "{invalid:?}");
                assert_eq!(
                    observe_stream_config(&proc, dir.path(), Some(invalid)),
                    unavailable,
                    "{invalid:?}"
                );
            }
            for unknown_release in [None, Some("7.2-other")] {
                assert_eq!(observe_stream_config(&proc, dir.path(), unknown_release), unavailable);
            }
            fs::rename(&boot, dir.path().join("moved-config")).expect("move regular boot fixture");
            symlink(dir.path().join("moved-config"), boot).expect("create boot config alias");
            assert_eq!(observe_stream_config(&proc, dir.path(), Some(release)), unavailable);
        }

        // Protects complete, bounded kernel-config reads. Detects oversized input or output,
        // missing gzip trailer, and trailing input accepted as current-kernel evidence. Needs
        // Linux temporary files and gzip encoder. Breadcrumb: read_proc_config, read_regular_file.
        #[test]
        fn stream_config_rejects_oversized_and_corrupt_sources() {
            let dir = tempfile::tempdir().expect("create compressed config fixture root");
            let proc = dir.path().join("config.gz");
            let release = "7.2.0-test";
            let boot = dir.path().join(format!("config-{release}"));

            let mut data = b"CONFIG_USB4_STREAM=y\n# ".to_vec();
            data.resize(MAX_KERNEL_CONFIG_BYTES, b'x');
            data[MAX_KERNEL_CONFIG_BYTES - 1] = b'\n';
            fs::write(&proc, gzip(&data)).expect("write maximum size config");
            assert_eq!(read_proc_config(&proc), Some(StreamConfigState::BuiltIn));

            for corrupt in [
                {
                    data.push(b'\n');
                    gzip(&data)
                },
                {
                    let mut bytes = gzip(b"CONFIG_USB4_STREAM=y\n");
                    bytes.pop();
                    bytes
                },
                {
                    let mut bytes = gzip(b"CONFIG_USB4_STREAM=y\n");
                    bytes.extend_from_slice(b"tail");
                    bytes
                },
                vec![b'x'; MAX_KERNEL_CONFIG_BYTES + 1],
            ] {
                fs::write(&proc, corrupt).expect("write bad proc source");
                assert_eq!(read_proc_config(&proc), None);
            }
            fs::write(boot, data).expect("write oversized boot config");
            assert_eq!(
                observe_stream_config(&proc, dir.path(), Some(release)),
                StreamConfig {
                    state: StreamConfigState::Unavailable,
                    source: None
                },
            );
        }

        // Protects exact mount-point and filesystem classification. Detects prefix/source
        // look-alikes, stacked records, partial listings and oversized or missing mountinfo.
        // Needs Linux temporary files. Breadcrumb: observe_configfs_mount, parse_mountinfo.
        #[test]
        fn configfs_mount_needs_exact_type_from_complete_mountinfo() {
            let dir = tempfile::tempdir().expect("create mountinfo fixture root");
            let path = dir.path().join("mountinfo");
            assert_eq!(observe_configfs_mount(&path), ConfigfsMount::Unavailable);
            let exact = b"11 10 0:47 / /sys/kernel/config rw shared:10 - configfs none rw\n";
            let prefix = b"11 10 0:47 / /sys/kernel/config-copy rw - configfs none rw\n";
            let other = b"11 10 0:47 / /sys/kernel/config rw - tmpfs configfs rw\n";
            let stacked = b"12 11 0:48 / /sys/kernel/config rw - tmpfs none rw\n";
            let non_unicode = b"12 10 0:48 / /mnt/\xff rw - tmpfs none rw\n";
            let nsfs = b"13 10 0:5 net:[123] /run/docker/netns/test rw - nsfs nsfs rw\n";
            for (bytes, expected) in [
                (exact.to_vec(), ConfigfsMount::Mounted),
                ([non_unicode.as_slice(), exact].concat(), ConfigfsMount::Mounted),
                ([nsfs.as_slice(), exact].concat(), ConfigfsMount::Mounted),
                (prefix.to_vec(), ConfigfsMount::Absent),
                (other.to_vec(), ConfigfsMount::UnexpectedType),
                ([exact.as_slice(), stacked].concat(), ConfigfsMount::Unavailable),
                (exact[..exact.len() - 1].to_vec(), ConfigfsMount::Unavailable),
                (
                    b"11 10 0:47 / /sys/kernel/config rw - configfs\n".to_vec(),
                    ConfigfsMount::Unavailable,
                ),
            ] {
                fs::write(&path, bytes).expect("write mountinfo fixture");
                assert_eq!(observe_configfs_mount(&path), expected);
            }
            let mut oversized = b"11 10 0:47 / /sys/kernel/config rw extension:".to_vec();
            oversized.resize(MAX_MOUNTINFO_BYTES, b'x');
            oversized.extend_from_slice(b" - configfs none rw\n");
            assert_eq!(parse_mountinfo(&oversized), Some(ConfigfsMount::Mounted));
            fs::write(&path, oversized).expect("write oversized mountinfo fixture");
            assert_eq!(observe_configfs_mount(&path), ConfigfsMount::Unavailable);
        }

        /// Writes `lines` as a newline-terminated mountinfo file and classifies it.
        fn classify_mountinfo(lines: &[&str]) -> ConfigfsMount {
            let dir = tempfile::tempdir().expect("create mountinfo fixture root");
            let path = dir.path().join("mountinfo");
            let mut bytes = lines.join("\n").into_bytes();
            bytes.push(b'\n');
            fs::write(&path, bytes).expect("write mountinfo fixture");
            observe_configfs_mount(&path)
        }

        const WSL2_ROOT: &str = "1 0 8:48 / / rw,relatime - ext4 /dev/sdd rw,discard,errors=remount-ro,data=ordered";
        const WSL2_SYSFS: &str = "2 1 0:2 / /sys rw,nosuid,nodev,noexec,noatime shared:2 - sysfs sysfs rw";
        const WSL2_PROC: &str = "3 1 0:5 / /proc rw,nosuid,nodev,noexec,noatime shared:3 - proc proc rw";
        const WSL2_CGROUP: &str =
            "5 2 0:7 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:5 - cgroup2 cgroup2 rw,nsdelegate";
        const WSL2_CONFIGFS: &str =
            "12 2 0:8 / /sys/kernel/config rw,nosuid,nodev,noexec,relatime shared:12 - configfs configfs rw";
        // A 9p drvfs mount as WSL2 reports it. The kernel does not escape the space inside the
        // `path=` super-option, so the record splits into one field more than the format allows.
        const WSL2_9P_WITH_SPACE: &str = r"45 1 0:41 / /Docker/host rw,noatime - 9p drvfs rw,dirsync,aname=drvfs;path=C:\Program Files\Docker\Docker\resources;symlinkroot=/mnt/,mmap,access=client,msize=65536,trans=fd,rfd=8,wfd=8";

        // Protects configfs detection on WSL2, where an unrelated 9p mount carries an unescaped
        // space in its super-options. Detects that record discarding a valid configfs record
        // (the whole file reported Unavailable) in either order. Needs Linux temporary files.
        // Breadcrumb: parse_mountinfo.
        #[test]
        fn configfs_is_found_beside_a_9p_record_with_unescaped_space_in_super_options() {
            let normal = [WSL2_ROOT, WSL2_SYSFS, WSL2_PROC, WSL2_CGROUP];
            let after: Vec<&str> = [&normal[..], &[WSL2_CONFIGFS, WSL2_9P_WITH_SPACE]].concat();
            let before: Vec<&str> = [&normal[..], &[WSL2_9P_WITH_SPACE, WSL2_CONFIGFS]].concat();
            assert_eq!(classify_mountinfo(&after), ConfigfsMount::Mounted);
            assert_eq!(classify_mountinfo(&before), ConfigfsMount::Mounted);
            // The same host without configfs is Absent, not Unavailable.
            let without: Vec<&str> = [&normal[..], &[WSL2_9P_WITH_SPACE]].concat();
            assert_eq!(classify_mountinfo(&without), ConfigfsMount::Absent);
            // A mismatched filesystem type is still reported next to the tolerated record.
            let wrong_type = "12 2 0:8 / /sys/kernel/config rw - tmpfs configfs rw";
            assert_eq!(
                classify_mountinfo(&[WSL2_ROOT, WSL2_9P_WITH_SPACE, wrong_type]),
                ConfigfsMount::UnexpectedType
            );
        }

        // Protects configfs detection against damaged records for other mount points. Detects a
        // missing separator, too few fields, non-decimal identifiers, a missing source and blank
        // lines poisoning the whole file. Needs Linux temporary files.
        // Breadcrumb: parse_mountinfo.
        #[test]
        fn malformed_records_for_other_mount_points_are_skipped() {
            for malformed in [
                "30 1 0:30 / /mnt/a rw shared:1 tmpfs none rw",
                "31 1 0:31",
                "x y 0:32 / /mnt/b rw - tmpfs none rw",
                "32 1 0:33 / /mnt/c rw -",
                "33 1 0:34 / /mnt/d rw - tmpfs",
                "",
            ] {
                assert_eq!(
                    classify_mountinfo(&[WSL2_ROOT, malformed, WSL2_CONFIGFS, malformed]),
                    ConfigfsMount::Mounted,
                    "record {malformed:?} must not hide configfs"
                );
                assert_eq!(
                    classify_mountinfo(&[WSL2_ROOT, malformed]),
                    ConfigfsMount::Absent,
                    "record {malformed:?} must not invent a configfs answer"
                );
            }
        }

        // Protects fail-closed reporting for the configfs mount point itself. Detects any
        // loosening of the checks that apply to a configfs record: extra, missing or non-decimal
        // fields, a missing separator, a shifted mount point, stacked records and conflicting
        // types must all stay Unavailable. Needs Linux temporary files. Breadcrumb: parse_mountinfo.
        #[test]
        fn malformed_or_conflicting_configfs_records_still_fail_closed() {
            for (record, why) in [
                (
                    "12 2 0:8 / /sys/kernel/config rw - configfs configfs rw extra",
                    "extra trailing field",
                ),
                (
                    "12 2 0:8 / /sys/kernel/config rw - tmpfs configfs rw extra",
                    "extra field on wrong type",
                ),
                (
                    "x 2 0:8 / /sys/kernel/config rw - configfs configfs rw",
                    "non-decimal mount id",
                ),
                (
                    "12 2 0:8 / /sys/kernel/config rw configfs configfs rw",
                    "missing separator",
                ),
                (
                    "12 2 0:8 / /sys/kernel/config rw - configfs",
                    "missing source and options",
                ),
                (
                    "12 2 0:8 / /sys/kernel/config rw - configfs configfs",
                    "missing super options",
                ),
                (
                    "12 2 0:8 /sys/kernel/config rw - configfs configfs rw",
                    "shifted mount point",
                ),
                ("12 2 0:8 / /sys/kernel/config", "truncated record"),
            ] {
                assert_eq!(
                    classify_mountinfo(&[WSL2_ROOT, WSL2_9P_WITH_SPACE, record]),
                    ConfigfsMount::Unavailable,
                    "{why}: {record:?}"
                );
            }
            let tmpfs = "13 12 0:9 / /sys/kernel/config rw - tmpfs none rw";
            let damaged = "13 12 0:9 / /sys/kernel/config rw -";
            for conflict in [WSL2_CONFIGFS, tmpfs, damaged] {
                assert_eq!(
                    classify_mountinfo(&[WSL2_ROOT, WSL2_CONFIGFS, WSL2_9P_WITH_SPACE, conflict]),
                    ConfigfsMount::Unavailable,
                    "stacked record {conflict:?}"
                );
            }
        }

        // Protects the rule that absence is only reported from a file with at least one usable
        // record. Detects an all-damaged or empty record list becoming Absent once damaged
        // records are skipped. Needs Linux temporary files. Breadcrumb: parse_mountinfo.
        #[test]
        fn absence_needs_at_least_one_usable_record() {
            for lines in [
                &[""][..],
                &["garbage"],
                &["11 10 0:47"],
                &["11 10 0:47 / /mnt/a rw shared:1 tmpfs none rw"],
            ] {
                assert_eq!(classify_mountinfo(lines), ConfigfsMount::Unavailable, "{lines:?}");
            }
            assert_eq!(classify_mountinfo(&[WSL2_9P_WITH_SPACE]), ConfigfsMount::Absent);
        }

        // Protects distinction between a missing control surface and a directory impostor.
        // Detects path-name-only qualification. Needs temp files; breadcrumb: directory_visibility.
        #[test]
        fn control_path_requires_an_observed_directory() {
            let root = tempfile::tempdir().expect("create fixture directory");
            let missing = root.path().join("missing");
            let directory = root.path().join("stream");
            let file = root.path().join("file");
            let alias = root.path().join("alias");
            fs::create_dir(&directory).expect("create control fixture");
            fs::write(&file, "fixture").expect("create non-directory fixture");
            symlink(&directory, &alias).expect("create alias fixture");
            assert_eq!(directory_visibility(&missing), Visibility::Absent);
            assert_eq!(directory_visibility(&directory), Visibility::Present);
            assert_eq!(directory_visibility(&file), Visibility::UnexpectedType);
            assert_eq!(directory_visibility(&alias), Visibility::UnexpectedType);
        }

        // Protects typed, no-open device inspection. Detects a file or alias treated as a
        // stream, and a missing or over-budget directory reported as fully scanned. Needs Linux temp files;
        // breadcrumb: scan_device_nodes.
        #[test]
        fn device_scan_rejects_non_devices_and_keeps_missing_scan_unknown() {
            let root = tempfile::tempdir().expect("create fixture directory");
            fs::write(root.path().join("tbstream0"), "fixture").expect("create regular node impostor");
            symlink("/dev/null", root.path().join("tbstream1")).expect("create device alias fixture");
            fs::create_dir(root.path().join("tbstream2")).expect("create directory impostor");
            fs::write(root.path().join("tbstream1x"), "fixture").expect("create unrelated fixture");
            let scan = scan_device_nodes(root.path(), MAX_DIRECTORY_ENTRIES);
            assert_eq!(scan.coverage, ScanCoverage::Complete);
            assert_eq!(scan.rejected_entries, 3);
            assert!(scan.character_devices.is_empty());
            let unavailable = scan_device_nodes(&root.path().join("absent"), MAX_DIRECTORY_ENTRIES);
            assert_eq!(unavailable.coverage, ScanCoverage::Unavailable);
            let limited = scan_device_nodes(root.path(), 2);
            assert_eq!(limited.coverage, ScanCoverage::LimitReached);
        }
    }
}
