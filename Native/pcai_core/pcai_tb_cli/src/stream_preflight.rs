//! Observe Linux USB4STREAM kernel surfaces without opening a stream.

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
    stream_control: Visibility,
    stream_module: Visibility,
    usb4net_module: Visibility,
    rxe_module: Visibility,
    siw_module: Visibility,
    device_scan: DeviceScan,
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
    use super::{DeviceScan, LinuxStreamObservation, ScanCoverage, Visibility};
    use std::fs;
    use std::io::ErrorKind;
    use std::os::unix::fs::FileTypeExt;
    use std::path::Path;

    const MAX_DIRECTORY_ENTRIES: usize = 4096;
    const MAX_STREAM_NODES: usize = 64;

    fn read_kernel_label(path: &str) -> Option<String> {
        fs::read_to_string(path)
            .ok()
            .map(|text| text.trim().to_owned())
            .filter(|text| !text.is_empty())
    }

    pub(super) fn hostname() -> Option<String> {
        read_kernel_label("/proc/sys/kernel/hostname")
    }

    pub(super) fn observe() -> LinuxStreamObservation {
        LinuxStreamObservation {
            kernel_release: read_kernel_label("/proc/sys/kernel/osrelease"),
            stream_control: directory_visibility(Path::new("/sys/kernel/config/thunderbolt/stream")),
            stream_module: directory_visibility(Path::new("/sys/module/thunderbolt_stream")),
            usb4net_module: directory_visibility(Path::new("/sys/module/thunderbolt_net")),
            rxe_module: directory_visibility(Path::new("/sys/module/rdma_rxe")),
            siw_module: directory_visibility(Path::new("/sys/module/siw")),
            device_scan: scan_device_nodes(Path::new("/dev"), MAX_DIRECTORY_ENTRIES),
        }
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
        use std::os::unix::fs::symlink;

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
