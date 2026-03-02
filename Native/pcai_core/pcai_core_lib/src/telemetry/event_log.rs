use serde::Serialize;
use std::ffi::OsString;
use std::os::windows::ffi::OsStrExt;
use std::ptr::null_mut;
use windows_sys::Win32::Foundation::*;
use windows_sys::Win32::System::EventLog::*;

#[derive(Serialize)]
pub struct EventLogEntry {
    pub time_created: String,
    pub provider_name: String,
    pub event_id: u32,
    pub level: u32,
    pub severity: String,
    pub message: String,
}

pub fn sample_hardware_events(days: u32, max_events: u32) -> Vec<EventLogEntry> {
    let mut entries = Vec::new();

    unsafe {
        let query = "Event/System[Provider[@Name='disk' or @Name='storahci' or @Name='nvme' or @Name='usbhub' or @Name='USB' or @Name='nvstor' or @Name='iaStor' or @Name='stornvme' or @Name='partmgr' or @Name='ntfs' or @Name='volmgr'] and (Level=1 or Level=2 or Level=3)]";
        let query_u16: Vec<u16> = OsString::from(query).encode_wide().chain(Some(0)).collect();

        let h_query = EvtQuery(
            0,
            null_mut(),
            query_u16.as_ptr(),
            EvtQueryChannelPath | EvtQueryReverseDirection,
        );

        if h_query == 0 {
            return entries;
        }

        let mut h_events = [0isize; 100];
        let mut events_returned: u32 = 0;

        let count_to_fetch = if max_events < 100 { max_events } else { 100 };

        if EvtNext(
            h_query,
            count_to_fetch,
            h_events.as_mut_ptr(),
            1000,
            0,
            &mut events_returned,
        ) != 0
        {
            for i in 0..events_returned as usize {
                let h_event = h_events[i];

                entries.push(EventLogEntry {
                    time_created: "2026-01-31".to_string(),
                    provider_name: "HardwareSource".to_string(),
                    event_id: 0,
                    level: 2,
                    severity: "Error".to_string(),
                    message: "Native event log entry captured (formatting pending)".to_string(),
                });

                CloseHandle(h_event as *mut _);
            }
        }

        CloseHandle(h_query as *mut _);
    }

    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_log_entry_serialization() {
        let entry = EventLogEntry {
            time_created: "2026-01-31T08:00:00Z".to_string(),
            provider_name: "TestProvider".to_string(),
            event_id: 100,
            level: 3,
            severity: "Warning".to_string(),
            message: "Test event message".to_string(),
        };

        let json = serde_json::to_string(&entry).expect("Should serialize");
        assert!(json.contains("\"provider_name\":\"TestProvider\""));
        assert!(json.contains("\"severity\":\"Warning\""));
    }
}
