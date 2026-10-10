//! Bounded Windows System event sampling with owned handles and real rendered values.

use chrono::{DateTime, SecondsFormat, Utc};
use regex::Regex;
use serde::Serialize;
use std::ffi::{c_char, CString};
use std::io::{self, Write};
use std::marker::PhantomData;
use std::mem::size_of;
use std::ptr::{null, null_mut};
use std::rc::Rc;
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::{GetLastError, ERROR_INSUFFICIENT_BUFFER, ERROR_NO_MORE_ITEMS};
use windows_sys::Win32::System::EventLog::*;

const PROVIDERS: &str = "(?i)disk|storahci|nvme|usbhub|USB|nvstor|iaStor|stornvme|partmgr|ntfs|volmgr";
const BATCH_SIZE: usize = 64;
const MAX_SCANNED: usize = 10_000;
const MAX_RENDER_BYTES: u32 = 1_048_576;
// Per text field, including the terminating UTF-16 NUL; never truncate.
const MAX_MESSAGE_CHARS: u32 = 65_536;
const MAX_RETAINED_TEXT: usize = 8 * 1_048_576;
const MAX_JSON_BYTES: usize = 16 * 1_048_576;
const SCAN_TIME: Duration = Duration::from_secs(10);

#[derive(Debug, thiserror::Error)]
pub enum EventLogError {
    #[error("{operation} failed with Windows error {code}")]
    Windows { operation: &'static str, code: u32 },
    #[error("invalid event-log arguments")]
    Arguments,
    #[error("invalid event-log rendering: {0}")]
    Rendering(&'static str),
    #[error("event-log scan limit reached")]
    ScanLimit,
    #[error("event-log allocation failed")]
    Allocation,
    #[error("event-log provider expression failed")]
    ProviderExpression,
    #[error("event-log retained text limit reached")]
    RetainedTextLimit,
    #[error("event-log serialized output limit reached")]
    OutputLimit,
    #[error("event-log serialization failed")]
    Serialization,
}

#[derive(Debug, Serialize)]
pub struct EventLogEntry {
    pub time_created: String,
    pub provider_name: String,
    // Keep existing native fields while supplying the public PowerShell contract.
    pub event_id: u32,
    pub id: u32,
    pub level: u32,
    pub level_display: String,
    pub severity: String,
    pub message: String,
    pub full_message: String,
}

// The query must remain on its creating thread. All EVT_HANDLEs use EvtClose.
struct OwnedEvent {
    raw: EVT_HANDLE,
    close: unsafe extern "system" fn(EVT_HANDLE) -> i32,
    _thread: PhantomData<Rc<()>>,
}

impl OwnedEvent {
    fn new(raw: EVT_HANDLE) -> Self {
        Self {
            raw,
            close: EvtClose,
            _thread: PhantomData,
        }
    }
}

impl Drop for OwnedEvent {
    fn drop(&mut self) {
        // SAFETY: this object exclusively owns a nonzero Event Log API handle.
        unsafe {
            (self.close)(self.raw);
        }
    }
}

fn last_error(operation: &'static str) -> EventLogError {
    // SAFETY: GetLastError has no pointer arguments; call immediately after failure.
    EventLogError::Windows {
        operation,
        code: unsafe { GetLastError() },
    }
}

fn wide(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(Some(0)).collect()
}

fn query_text(days: u32) -> String {
    let milliseconds = u64::from(days) * 86_400_000;
    format!("*[System[(Level=1 or Level=2 or Level=3) and TimeCreated[timediff(@SystemTime) <= {milliseconds}]]]")
}

fn elapsed(started: Instant) -> Duration {
    started.elapsed()
}

struct ApiBudget {
    started: Instant,
    elapsed: fn(Instant) -> Duration,
}

impl ApiBudget {
    fn new(started: Instant) -> Self {
        Self { started, elapsed }
    }

    fn admit(&self) -> Result<(), EventLogError> {
        if (self.elapsed)(self.started) >= SCAN_TIME {
            Err(EventLogError::ScanLimit)
        } else {
            Ok(())
        }
    }

    // Immediate GetLastError captures the operation's failure without admitting
    // another I/O operation. EvtClose cleanup is deliberately unconditional.
    fn invoke<T>(
        &self,
        operation: &'static str,
        call: impl FnOnce() -> T,
        succeeded: impl FnOnce(&T) -> bool,
    ) -> Result<T, EventLogError> {
        self.admit()?;
        let result = call();
        if succeeded(&result) {
            Ok(result)
        } else {
            Err(last_error(operation))
        }
    }
}

struct TextBudget {
    used: usize,
    limit: usize,
}

impl TextBudget {
    fn charge(&mut self, bytes: usize) -> Result<(), EventLogError> {
        let used = self
            .used
            .checked_add(bytes)
            .filter(|&used| used <= self.limit)
            .ok_or(EventLogError::RetainedTextLimit)?;
        self.used = used;
        Ok(())
    }
}

impl EventLogEntry {
    fn text_bytes(&self) -> Result<usize, EventLogError> {
        [
            &self.time_created,
            &self.provider_name,
            &self.level_display,
            &self.severity,
            &self.message,
            &self.full_message,
        ]
        .iter()
        .try_fold(0_usize, |sum, text| {
            sum.checked_add(text.len()).ok_or(EventLogError::RetainedTextLimit)
        })
    }
}

fn admit_scan(budget: &ApiBudget, scanned: usize) -> Result<(), EventLogError> {
    budget.admit()?;
    if scanned >= MAX_SCANNED {
        Err(EventLogError::ScanLimit)
    } else {
        Ok(())
    }
}

/// Sample real hardware events; an error must never be interpreted as a quiet log.
pub fn sample_hardware_events(days: u32, max_events: u32) -> Result<Vec<EventLogEntry>, EventLogError> {
    if !(1..=30).contains(&days) || !(1..=500).contains(&max_events) {
        return Err(EventLogError::Arguments);
    }
    let started = Instant::now();
    let budget = ApiBudget::new(started);
    let provider_filter = Regex::new(PROVIDERS).map_err(|_| EventLogError::ProviderExpression)?;
    let channel = wide("System");
    let query = wide(&query_text(days));
    // SAFETY: both strings are terminated and live throughout this synchronous call.
    let query = OwnedEvent::new(budget.invoke(
        "EvtQuery",
        || unsafe {
            EvtQuery(
                0,
                channel.as_ptr(),
                query.as_ptr(),
                EvtQueryChannelPath | EvtQueryReverseDirection,
            )
        },
        |raw| *raw != 0,
    )?);
    // SAFETY: system context takes no value-path array.
    let context = OwnedEvent::new(budget.invoke(
        "EvtCreateRenderContext",
        || unsafe { EvtCreateRenderContext(0, null(), EvtRenderContextSystem) },
        |raw| *raw != 0,
    )?);
    collect_events(
        || next_batch(query.raw, &budget),
        |event| {
            let values = render_values(context.raw, event.raw, &budget)?;
            let fields = values.fields()?;
            if !provider_filter.is_match(&fields.provider) {
                return Ok(None);
            }
            let publisher = wide(&fields.provider);
            // SAFETY: publisher is terminated; no archived file or remote session is used.
            let publisher = OwnedEvent::new(budget.invoke(
                "EvtOpenPublisherMetadata",
                || unsafe { EvtOpenPublisherMetadata(0, publisher.as_ptr(), null(), 0, 0) },
                |raw| *raw != 0,
            )?);
            let full_message = format_message(publisher.raw, event.raw, EvtFormatMessageEvent, &budget)?;
            let level_display = format_message(publisher.raw, event.raw, EvtFormatMessageLevel, &budget)?;
            fields.entry(level_display, full_message).map(Some)
        },
        max_events,
        &budget,
    )
}

fn next_batch(query: EVT_HANDLE, budget: &ApiBudget) -> Result<Option<Vec<OwnedEvent>>, EventLogError> {
    let mut handles = [0; BATCH_SIZE];
    let mut returned = 0;
    // SAFETY: the handle output array and count pointer are valid for this call.
    // Reserve ownership before the API can produce live handles.
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(BATCH_SIZE)
        .map_err(|_| EventLogError::Allocation)?;
    let result = budget.invoke(
        "EvtNext",
        || unsafe { EvtNext(query, BATCH_SIZE as u32, handles.as_mut_ptr(), 1_000, 0, &mut returned) },
        |result| *result != 0,
    );
    // Take custody of the whole returned batch before rendering any member.
    for raw in handles.into_iter().filter(|&raw| raw != 0) {
        owned.push(OwnedEvent::new(raw));
    }
    if let Err(error) = result {
        if matches!(
            error,
            EventLogError::Windows {
                code: ERROR_NO_MORE_ITEMS,
                ..
            }
        ) && returned == 0
            && owned.is_empty()
        {
            return Ok(None);
        }
        return Err(error);
    }
    if returned == 0 || returned as usize != owned.len() || returned as usize > BATCH_SIZE {
        return Err(EventLogError::Rendering("invalid event batch"));
    }
    Ok(Some(owned))
}

fn collect_events(
    mut next: impl FnMut() -> Result<Option<Vec<OwnedEvent>>, EventLogError>,
    mut render: impl FnMut(&OwnedEvent) -> Result<Option<EventLogEntry>, EventLogError>,
    max_events: u32,
    budget: &ApiBudget,
) -> Result<Vec<EventLogEntry>, EventLogError> {
    let mut entries = Vec::new();
    let mut scanned = 0;
    let mut text = TextBudget {
        used: 0,
        limit: MAX_RETAINED_TEXT,
    };
    loop {
        admit_scan(budget, scanned)?;
        let batch = next()?;
        budget.admit()?;
        let Some(owned) = batch else {
            return Ok(entries);
        };
        if owned.is_empty() {
            return Err(EventLogError::Rendering("empty event batch"));
        }
        for event in &owned {
            admit_scan(budget, scanned)?;
            scanned += 1;
            if let Some(entry) = render(event)? {
                // Count all retained fields, including duplicate Message/FullMessage.
                text.charge(entry.text_bytes()?)?;
                entries.try_reserve(1).map_err(|_| EventLogError::Allocation)?;
                entries.push(entry);
            }
            // Deadline is admission between synchronous API calls, not cancellation of them.
            budget.admit()?;
            if entries.len() == max_events as usize {
                return Ok(entries);
            }
        }
    }
}

struct RenderedValues {
    // EVT_VARIANT alignment also aligns the string payload following the array.
    storage: Vec<EVT_VARIANT>,
    used: usize,
    count: usize,
}

fn render_values(context: EVT_HANDLE, event: EVT_HANDLE, budget: &ApiBudget) -> Result<RenderedValues, EventLogError> {
    let mut needed = 0;
    let mut count = 0;
    // SAFETY: zero buffer/null pointer is the documented sizing request.
    let result = budget.invoke(
        "EvtRender(size)",
        || unsafe {
            EvtRender(
                context,
                event,
                EvtRenderEventValues,
                0,
                null_mut(),
                &mut needed,
                &mut count,
            )
        },
        |result| *result != 0,
    );
    let error = match result {
        Ok(_) => return Err(EventLogError::Rendering("unexpected empty render")),
        Err(error) => error,
    };
    if !matches!(
        error,
        EventLogError::Windows {
            code: ERROR_INSUFFICIENT_BUFFER,
            ..
        }
    ) {
        return Err(error);
    }
    if needed == 0 || needed > MAX_RENDER_BYTES {
        return Err(EventLogError::Rendering("render size outside bound"));
    }
    let capacity = (needed as usize).div_ceil(size_of::<EVT_VARIANT>());
    let mut storage = Vec::new();
    storage
        .try_reserve_exact(capacity)
        .map_err(|_| EventLogError::Allocation)?;
    storage.resize(capacity, EVT_VARIANT::default());
    let mut used = 0;
    // SAFETY: storage has enough bytes and EVT_VARIANT alignment; strings remain owned here.
    budget.invoke(
        "EvtRender(values)",
        || unsafe {
            EvtRender(
                context,
                event,
                EvtRenderEventValues,
                needed,
                storage.as_mut_ptr().cast(),
                &mut used,
                &mut count,
            )
        },
        |result| *result != 0,
    )?;
    if used > needed
        || count < EvtSystemPropertyIdEND as u32
        || count as usize > capacity
        || (count as usize) * size_of::<EVT_VARIANT>() > used as usize
    {
        return Err(EventLogError::Rendering("invalid render bounds"));
    }
    Ok(RenderedValues {
        storage,
        used: used as usize,
        count: count as usize,
    })
}

fn require_type(value: &EVT_VARIANT, expected: i32) -> Result<(), EventLogError> {
    if value.Type != expected as u32 {
        return Err(EventLogError::Rendering("missing or mistyped system value"));
    }
    Ok(())
}

impl RenderedValues {
    fn value(&self, index: i32, expected: i32) -> Result<&EVT_VARIANT, EventLogError> {
        let value = self
            .storage
            .get(index as usize)
            .filter(|_| (index as usize) < self.count)
            .ok_or(EventLogError::Rendering("missing system property"))?;
        require_type(value, expected)?;
        Ok(value)
    }

    fn fields(&self) -> Result<SystemFields, EventLogError> {
        let provider = self.value(EvtSystemProviderName, EvtVarTypeString)?;
        // SAFETY: require_type verified the active StringVal union member.
        let pointer = unsafe { provider.Anonymous.StringVal };
        let start = self.storage.as_ptr() as usize;
        let end = start
            .checked_add(self.used)
            .ok_or(EventLogError::Rendering("render address overflow"))?;
        let address = pointer as usize;
        let payload_start = start + self.count * size_of::<EVT_VARIANT>();
        if address < payload_start || address >= end || address % 2 != 0 {
            return Err(EventLogError::Rendering("provider pointer outside render payload"));
        }
        // SAFETY: pointer is aligned and within the owned initialized allocation, bounded by used.
        let units = unsafe { std::slice::from_raw_parts(pointer, (end - address) / 2) };
        let length = units
            .iter()
            .position(|&unit| unit == 0)
            .ok_or(EventLogError::Rendering("unterminated provider"))?;
        if length >= MAX_MESSAGE_CHARS as usize {
            return Err(EventLogError::Rendering("provider size outside bound"));
        }
        let provider = decode_utf16(&units[..length])?;
        if provider.is_empty() {
            return Err(EventLogError::Rendering("empty provider"));
        }
        // SAFETY: each accessor validates the union type before reading its value.
        let id = unsafe { self.value(EvtSystemEventID, EvtVarTypeUInt16)?.Anonymous.UInt16Val };
        let level = unsafe { self.value(EvtSystemLevel, EvtVarTypeByte)?.Anonymous.ByteVal };
        let time = unsafe {
            self.value(EvtSystemTimeCreated, EvtVarTypeFileTime)?
                .Anonymous
                .FileTimeVal
        };
        Ok(SystemFields {
            provider,
            id: u32::from(id),
            level: u32::from(level),
            time,
        })
    }
}

fn format_message(
    publisher: EVT_HANDLE,
    event: EVT_HANDLE,
    flags: u32,
    budget: &ApiBudget,
) -> Result<String, EventLogError> {
    let mut needed = 0;
    // SAFETY: documented sizing call; insertion values are obtained from the event itself.
    let result = budget.invoke(
        "EvtFormatMessage(size)",
        || unsafe { EvtFormatMessage(publisher, event, 0, 0, null(), flags, 0, null_mut(), &mut needed) },
        |result| *result != 0,
    );
    let error = match result {
        Ok(_) => return Err(EventLogError::Rendering("unexpected empty format")),
        Err(error) => error,
    };
    if !matches!(
        error,
        EventLogError::Windows {
            code: ERROR_INSUFFICIENT_BUFFER,
            ..
        }
    ) {
        return Err(error);
    }
    if needed == 0 || needed > MAX_MESSAGE_CHARS {
        return Err(EventLogError::Rendering("message size outside bound"));
    }
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(needed as usize)
        .map_err(|_| EventLogError::Allocation)?;
    buffer.resize(needed as usize, 0_u16);
    let mut used = 0;
    // SAFETY: buffer size is measured in UTF-16 characters, not bytes.
    budget.invoke(
        "EvtFormatMessage(text)",
        || unsafe {
            EvtFormatMessage(
                publisher,
                event,
                0,
                0,
                null(),
                flags,
                needed,
                buffer.as_mut_ptr(),
                &mut used,
            )
        },
        |result| *result != 0,
    )?;
    decode_message(&buffer, used as usize)
}

fn decode_message(buffer: &[u16], used: usize) -> Result<String, EventLogError> {
    if used > MAX_MESSAGE_CHARS as usize {
        return Err(EventLogError::Rendering("message size outside bound"));
    }
    let buffer = buffer
        .get(..used)
        .ok_or(EventLogError::Rendering("message length outside buffer"))?;
    let length = buffer
        .iter()
        .position(|&unit| unit == 0)
        .ok_or(EventLogError::Rendering("unterminated message"))?;
    decode_utf16(&buffer[..length])
}

fn decode_utf16(units: &[u16]) -> Result<String, EventLogError> {
    if units.len() >= MAX_MESSAGE_CHARS as usize {
        return Err(EventLogError::Rendering("text size outside bound"));
    }
    // Three bytes per UTF-16 unit bounds all valid encodings, including pairs.
    let capacity = units.len().checked_mul(3).ok_or(EventLogError::Allocation)?;
    let mut text = String::new();
    text.try_reserve_exact(capacity)
        .map_err(|_| EventLogError::Allocation)?;
    for scalar in char::decode_utf16(units.iter().copied()) {
        text.push(scalar.map_err(|_| EventLogError::Rendering("invalid text UTF-16"))?);
    }
    Ok(text)
}

fn copy_text(text: &str) -> Result<String, EventLogError> {
    let mut result = String::new();
    result
        .try_reserve_exact(text.len())
        .map_err(|_| EventLogError::Allocation)?;
    result.push_str(text);
    Ok(result)
}

struct SystemFields {
    provider: String,
    id: u32,
    level: u32,
    time: u64,
}

fn file_time(value: u64) -> Result<String, EventLogError> {
    let ticks = i128::from(value) - 116_444_736_000_000_000_i128;
    let seconds =
        i64::try_from(ticks.div_euclid(10_000_000)).map_err(|_| EventLogError::Rendering("timestamp overflow"))?;
    let nanos = (ticks.rem_euclid(10_000_000) * 100) as u32;
    DateTime::<Utc>::from_timestamp(seconds, nanos)
        .map(|time| time.to_rfc3339_opts(SecondsFormat::AutoSi, true))
        .ok_or(EventLogError::Rendering("timestamp outside calendar range"))
}

impl SystemFields {
    fn entry(self, level_display: String, full_message: String) -> Result<EventLogEntry, EventLogError> {
        let severity = match self.level {
            1 => "Critical",
            2 => "Error",
            3 => "Warning",
            _ => return Err(EventLogError::Rendering("level outside query")),
        };
        if level_display.is_empty() {
            return Err(EventLogError::Rendering("empty level display"));
        }
        let message = copy_text(full_message.split('\n').next().unwrap_or_default())?;
        Ok(EventLogEntry {
            time_created: file_time(self.time)?,
            provider_name: self.provider,
            event_id: self.id,
            id: self.id,
            level: self.level,
            level_display,
            severity: copy_text(severity)?,
            message,
            full_message,
        })
    }
}

/// Convert failure to NULL, and successful empty sampling to an owned `[]` string.
pub fn events_to_ffi(result: Result<Vec<EventLogEntry>, EventLogError>) -> *mut c_char {
    result
        .and_then(|events| serialize_events(&events, MAX_JSON_BYTES))
        .and_then(|mut bytes| {
            bytes.try_reserve_exact(1).map_err(|_| EventLogError::Allocation)?;
            bytes.push(0);
            CString::from_vec_with_nul(bytes).map_err(|_| EventLogError::Serialization)
        })
        .map_or(null_mut(), CString::into_raw)
}

struct BoundedJsonWriter {
    bytes: Vec<u8>,
    limit: usize,
    failure: Option<EventLogError>,
}

impl Write for BoundedJsonWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.failure.is_some() {
            return Err(io::ErrorKind::InvalidData.into());
        }
        let Some(length) = self
            .bytes
            .len()
            .checked_add(buffer.len())
            .filter(|&length| length <= self.limit)
        else {
            self.failure = Some(EventLogError::OutputLimit);
            return Err(io::ErrorKind::InvalidData.into());
        };
        if length > self.bytes.capacity() {
            let target = length
                .max(self.bytes.capacity().saturating_mul(2))
                .max(4096)
                .min(self.limit);
            if self.bytes.try_reserve_exact(target - self.bytes.len()).is_err() {
                self.failure = Some(EventLogError::Allocation);
                return Err(io::ErrorKind::OutOfMemory.into());
            }
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn serialize_events(events: &[EventLogEntry], limit: usize) -> Result<Vec<u8>, EventLogError> {
    let mut text = TextBudget {
        used: 0,
        limit: MAX_RETAINED_TEXT,
    };
    for entry in events {
        text.charge(entry.text_bytes()?)?;
    }
    let mut writer = BoundedJsonWriter {
        bytes: Vec::new(),
        limit,
        failure: None,
    };
    if serde_json::to_writer(&mut writer, events).is_err() {
        return Err(writer.failure.unwrap_or(EventLogError::Serialization));
    }
    Ok(writer.bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    static CLOSE_TEST: Mutex<()> = Mutex::new(());
    static CLOSED: AtomicUsize = AtomicUsize::new(0);
    unsafe extern "system" fn close_fixture(_: EVT_HANDLE) -> i32 {
        CLOSED.fetch_add(1, Ordering::SeqCst);
        1
    }

    // Protects: Requested Days and FILETIME precision.
    // Detects: Ignored window, epoch offset or 100 ns loss.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: query_text / file_time; telemetry::event_log::tests::dates_and_query_use_requested_window.
    #[test]
    fn dates_and_query_use_requested_window() {
        assert!(query_text(7).contains("604800000"));
        assert_eq!(file_time(116_444_736_000_000_000).unwrap(), "1970-01-01T00:00:00Z");
        assert_eq!(
            file_time(116_444_736_000_000_001).unwrap(),
            "1970-01-01T00:00:00.000000100Z"
        );
        assert_eq!(
            file_time(116_444_735_999_999_999).unwrap(),
            "1969-12-31T23:59:59.999999900Z"
        );
    }

    // Protects: Typed system-value admission.
    // Detects: Null/array values accepted as scalar strings.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: require_type; telemetry::event_log::tests::variant_types_reject_null_and_array_values.
    #[test]
    fn variant_types_reject_null_and_array_values() {
        let mut value = EVT_VARIANT::default();
        assert!(require_type(&value, EvtVarTypeString).is_err());
        value.Type = EvtVarTypeString as u32 | EVT_VARIANT_TYPE_ARRAY;
        assert!(require_type(&value, EvtVarTypeString).is_err());
        value.Type = EvtVarTypeString as u32;
        assert!(require_type(&value, EvtVarTypeString).is_ok());
    }

    // Protects: Complete formatted text and bounded decoding.
    // Detects: Lost newlines, bad used length, missing NUL or invalid UTF-16.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: decode_message; telemetry::event_log::tests::formatting_preserves_newlines_and_rejects_bad_bounds.
    #[test]
    fn formatting_preserves_newlines_and_rejects_bad_bounds() {
        assert_eq!(decode_message(&wide("first\r\nsecond"), 14).unwrap(), "first\r\nsecond");
        assert!(decode_message(&[65], 1).is_err());
        assert!(decode_message(&[0], 2).is_err());
        assert!(decode_message(&[0xD800, 0], 2).is_err());
    }

    // Protects: Whole returned-batch custody.
    // Detects: Unprocessed handles leaking after first render error.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: collect_events / OwnedEvent::drop; telemetry::event_log::tests::whole_batch_closes_on_early_processing_error.
    #[test]
    fn whole_batch_closes_on_early_processing_error() {
        let _lock = CLOSE_TEST.lock().unwrap();
        CLOSED.store(0, Ordering::SeqCst);
        let batch = || {
            Ok(Some(
                (1..=3)
                    .map(|raw| OwnedEvent {
                        raw,
                        close: close_fixture,
                        _thread: PhantomData,
                    })
                    .collect(),
            ))
        };
        let result = collect_events(
            batch,
            |_| Err(EventLogError::Rendering("inert early failure")),
            50,
            &ApiBudget::new(Instant::now()),
        );
        assert!(result.is_err());
        assert_eq!(CLOSED.load(Ordering::SeqCst), 3);
    }

    // Protects: Nullable FFI failure versus successful empty JSON.
    // Detects: Failure mislabeled as a quiet event log.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: events_to_ffi / pcai_sample_hardware_events_json; telemetry::event_log::tests::ffi_distinguishes_empty_from_error_without_windows_calls.
    #[test]
    fn ffi_distinguishes_empty_from_error_without_windows_calls() {
        assert!(events_to_ffi(Err(EventLogError::Windows {
            operation: "inert",
            code: 5
        }))
        .is_null());
        let pointer = events_to_ffi(Ok(Vec::new()));
        assert!(!pointer.is_null());
        // SAFETY: pointer was allocated by CString::into_raw above; reclaim exactly once.
        let empty = unsafe { CString::from_raw(pointer) };
        assert_eq!(empty.to_str().unwrap(), "[]");
    }

    // Protects: Native/public field and CRLF contract.
    // Detects: Missing id/level label/full text or trailing-CR divergence.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: SystemFields::entry; telemetry::event_log::tests::real_fields_supply_public_and_legacy_contracts.
    #[test]
    fn real_fields_supply_public_and_legacy_contracts() {
        let fields = SystemFields {
            provider: "Microsoft-Windows-Ntfs".into(),
            id: 55,
            level: 3,
            time: 116_444_736_000_000_000,
        };
        assert!(Regex::new(PROVIDERS).unwrap().is_match(&fields.provider));
        let row = serde_json::to_value(fields.entry("Warning".into(), "first\r\nsecond".into()).unwrap()).unwrap();
        assert_eq!(row["event_id"], 55);
        assert_eq!(row["id"], 55);
        assert_eq!(row["level_display"], "Warning");
        assert_eq!(row["message"], "first\r");
        assert_eq!(row["full_message"], "first\r\nsecond");
    }

    // Protects: Bounded scanning beyond former one-page cap.
    // Detects: Premature 100-record rejection or limit treated as success.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: admit_scan; telemetry::event_log::tests::scan_limit_is_failure_and_does_not_cap_at_one_hundred.
    #[test]
    fn scan_limit_is_failure_and_does_not_cap_at_one_hundred() {
        assert!(admit_scan(&ApiBudget::new(Instant::now()), 501).is_ok());
        assert!(admit_scan(&ApiBudget::new(Instant::now()), MAX_SCANNED).is_err());
        assert!(admit_scan(&ApiBudget::new(Instant::now() - SCAN_TIME), 0).is_err());
    }

    // Protects: Paging and unused-tail handle custody.
    // Detects: 100-event hardcap or unneeded batch handles leaking.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: collect_events / OwnedEvent::drop; telemetry::event_log::tests::same_collector_pages_beyond_one_hundred_and_closes_unneeded_tail.
    #[test]
    fn same_collector_pages_beyond_one_hundred_and_closes_unneeded_tail() {
        let _lock = CLOSE_TEST.lock().unwrap();
        CLOSED.store(0, Ordering::SeqCst);
        let mut pages = 0;
        let next = || {
            pages += 1;
            Ok(Some(
                (1..=64)
                    .map(|raw| OwnedEvent {
                        raw,
                        close: close_fixture,
                        _thread: PhantomData,
                    })
                    .collect(),
            ))
        };
        let render = |_: &OwnedEvent| {
            SystemFields {
                provider: "disk".into(),
                id: 7,
                level: 2,
                time: 116_444_736_000_000_000,
            }
            .entry("Error".into(), "inert message".into())
            .map(Some)
        };
        let result = collect_events(next, render, 150, &ApiBudget::new(Instant::now())).unwrap();
        assert_eq!(result.len(), 150);
        assert_eq!(pages, 3);
        assert_eq!(CLOSED.load(Ordering::SeqCst), 192);
    }

    // Protects: Whole-result failure after an earlier valid row.
    // Detects: Partial successful array escaping a later render failure.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: collect_events; telemetry::event_log::tests::collector_does_not_publish_partial_results_after_late_failure.
    #[test]
    fn collector_does_not_publish_partial_results_after_late_failure() {
        let _lock = CLOSE_TEST.lock().unwrap();
        CLOSED.store(0, Ordering::SeqCst);
        let batch = || {
            Ok(Some(
                (1..=3)
                    .map(|raw| OwnedEvent {
                        raw,
                        close: close_fixture,
                        _thread: PhantomData,
                    })
                    .collect(),
            ))
        };
        let render = |event: &OwnedEvent| {
            if event.raw == 2 {
                return Err(EventLogError::Rendering("inert message failure"));
            }
            SystemFields {
                provider: "disk".into(),
                id: 7,
                level: 2,
                time: 116_444_736_000_000_000,
            }
            .entry("Error".into(), "inert message".into())
            .map(Some)
        };
        assert!(collect_events(batch, render, 50, &ApiBudget::new(Instant::now())).is_err());
        assert_eq!(CLOSED.load(Ordering::SeqCst), 3);
    }

    // Protects: Typed provider/id/level/time values and pointer ownership.
    // Detects: External provider pointer dereference or wrong field extraction.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: RenderedValues::fields; telemetry::event_log::tests::typed_render_fixture_reads_real_fields_and_rejects_external_pointer.
    #[test]
    fn typed_render_fixture_reads_real_fields_and_rejects_external_pointer() {
        let count = EvtSystemPropertyIdEND as usize;
        let provider = wide("Microsoft-Windows-Ntfs");
        let header = count * size_of::<EVT_VARIANT>();
        let used = header + provider.len() * 2;
        let mut values = RenderedValues {
            storage: vec![EVT_VARIANT::default(); used.div_ceil(size_of::<EVT_VARIANT>())],
            used,
            count,
        };
        // SAFETY: inert payload lies entirely in owned aligned initialized storage.
        let pointer = unsafe { values.storage.as_mut_ptr().cast::<u8>().add(header).cast::<u16>() };
        unsafe {
            std::ptr::copy_nonoverlapping(provider.as_ptr(), pointer, provider.len());
        }
        values.storage[EvtSystemProviderName as usize] = EVT_VARIANT {
            Anonymous: EVT_VARIANT_0 { StringVal: pointer },
            Count: 0,
            Type: EvtVarTypeString as u32,
        };
        values.storage[EvtSystemEventID as usize] = EVT_VARIANT {
            Anonymous: EVT_VARIANT_0 { UInt16Val: 55 },
            Count: 0,
            Type: EvtVarTypeUInt16 as u32,
        };
        values.storage[EvtSystemLevel as usize] = EVT_VARIANT {
            Anonymous: EVT_VARIANT_0 { ByteVal: 3 },
            Count: 0,
            Type: EvtVarTypeByte as u32,
        };
        values.storage[EvtSystemTimeCreated as usize] = EVT_VARIANT {
            Anonymous: EVT_VARIANT_0 {
                FileTimeVal: 116_444_736_000_000_000,
            },
            Count: 0,
            Type: EvtVarTypeFileTime as u32,
        };
        let fields = values.fields().unwrap();
        assert_eq!(fields.provider, "Microsoft-Windows-Ntfs");
        assert_eq!((fields.id, fields.level, fields.time), (55, 3, 116_444_736_000_000_000));
        values.storage[EvtSystemProviderName as usize].Anonymous.StringVal = provider.as_ptr();
        assert!(values.fields().is_err());
    }

    static FAKE_MILLIS: AtomicUsize = AtomicUsize::new(0);
    fn fake_elapsed(_: Instant) -> Duration {
        Duration::from_millis(FAKE_MILLIS.load(Ordering::SeqCst) as u64)
    }
    fn inert_budget() -> ApiBudget {
        ApiBudget {
            started: Instant::now(),
            elapsed: |_| Duration::ZERO,
        }
    }
    fn inert_row(full_message: String) -> EventLogEntry {
        SystemFields {
            provider: "disk".into(),
            id: 7,
            level: 2,
            time: 116_444_736_000_000_000,
        }
        .entry("Error".into(), full_message)
        .unwrap()
    }

    // Protects: Admission immediately before each synchronous API.
    // Detects: Later closure invoked after earlier call crosses deadline.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: ApiBudget::invoke; telemetry::event_log::tests::shared_api_admission_rejects_the_next_call_after_an_earlier_call_overspends.
    #[test]
    fn shared_api_admission_rejects_the_next_call_after_an_earlier_call_overspends() {
        let _lock = CLOSE_TEST.lock().unwrap();
        FAKE_MILLIS.store(0, Ordering::SeqCst);
        let budget = ApiBudget {
            started: Instant::now(),
            elapsed: fake_elapsed,
        };
        let mut invoked = 0;
        assert!(budget
            .invoke(
                "inert first",
                || {
                    invoked += 1;
                    FAKE_MILLIS.store(10_001, Ordering::SeqCst);
                    true
                },
                |ok| *ok
            )
            .is_ok());
        assert!(matches!(
            budget.invoke(
                "inert later",
                || {
                    invoked += 1;
                    true
                },
                |ok| *ok
            ),
            Err(EventLogError::ScanLimit)
        ));
        assert_eq!(invoked, 1);
    }

    // Protects: Late API-budget failure and complete cleanup.
    // Detects: Second values call, partial rows or leaked handles after expiry.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: ApiBudget::invoke / collect_events; telemetry::event_log::tests::late_shared_api_admission_failure_discards_rows_and_closes_the_whole_batch.
    #[test]
    fn late_shared_api_admission_failure_discards_rows_and_closes_the_whole_batch() {
        let _lock = CLOSE_TEST.lock().unwrap();
        CLOSED.store(0, Ordering::SeqCst);
        FAKE_MILLIS.store(0, Ordering::SeqCst);
        let budget = ApiBudget {
            started: Instant::now(),
            elapsed: fake_elapsed,
        };
        let next = || {
            Ok(Some(
                (1..=3)
                    .map(|raw| OwnedEvent {
                        raw,
                        close: close_fixture,
                        _thread: PhantomData,
                    })
                    .collect(),
            ))
        };
        let mut invoked = 0;
        let render = |event: &OwnedEvent| {
            budget.invoke(
                "inert render sizing",
                || {
                    invoked += 1;
                    if event.raw == 2 {
                        FAKE_MILLIS.store(10_001, Ordering::SeqCst);
                    }
                    true
                },
                |ok| *ok,
            )?;
            budget.invoke(
                "inert render values",
                || {
                    invoked += 1;
                    true
                },
                |ok| *ok,
            )?;
            Ok(Some(inert_row("inert".into())))
        };
        assert!(matches!(
            collect_events(next, render, 50, &budget),
            Err(EventLogError::ScanLimit)
        ));
        assert_eq!(invoked, 3); // first row two calls; second sizing only.
        assert_eq!(CLOSED.load(Ordering::SeqCst), 3);
    }

    // Protects: Aggregate UTF-8 text accounting.
    // Detects: Character-count undercharge or omission of duplicate Message.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: EventLogEntry::text_bytes / TextBudget::charge; telemetry::event_log::tests::retained_text_accounting_counts_utf8_and_duplicate_message_fields.
    #[test]
    fn retained_text_accounting_counts_utf8_and_duplicate_message_fields() {
        let row = inert_row("😀é".into());
        let expected =
            row.time_created.len() + row.provider_name.len() + row.level_display.len() + row.severity.len() + 6 + 6;
        assert_eq!(row.message.encode_utf16().count(), 3);
        assert_eq!(row.text_bytes().unwrap(), expected);
        let mut budget = TextBudget {
            used: 0,
            limit: expected,
        };
        assert!(budget.charge(row.text_bytes().unwrap()).is_ok());
        assert!(matches!(budget.charge(1), Err(EventLogError::RetainedTextLimit)));
        assert_eq!(budget.used, expected);
    }

    // Protects: Checked aggregate-counter admission.
    // Detects: usize overflow or failed charge changing prior custody.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: TextBudget::charge; telemetry::event_log::tests::retained_text_budget_checked_overflow_does_not_mutate_the_counter.
    #[test]
    fn retained_text_budget_checked_overflow_does_not_mutate_the_counter() {
        let mut budget = TextBudget {
            used: usize::MAX - 1,
            limit: usize::MAX,
        };
        assert!(matches!(budget.charge(2), Err(EventLogError::RetainedTextLimit)));
        assert_eq!(budget.used, usize::MAX - 1);
    }

    // Protects: Actual 8 MiB text cap and whole-batch cleanup.
    // Detects: Individually legal fields accumulating into partial success.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: collect_events / TextBudget::charge; telemetry::event_log::tests::aggregate_text_limit_discards_all_rows_and_closes_the_batch.
    #[test]
    fn aggregate_text_limit_discards_all_rows_and_closes_the_batch() {
        let _lock = CLOSE_TEST.lock().unwrap();
        CLOSED.store(0, Ordering::SeqCst);
        let next = || {
            Ok(Some(
                (1..=64)
                    .map(|raw| OwnedEvent {
                        raw,
                        close: close_fixture,
                        _thread: PhantomData,
                    })
                    .collect(),
            ))
        };
        let render = |_: &OwnedEvent| Ok(Some(inert_row("a".repeat(MAX_MESSAGE_CHARS as usize - 1))));
        assert!(matches!(
            collect_events(next, render, 500, &inert_budget()),
            Err(EventLogError::RetainedTextLimit)
        ));
        assert_eq!(CLOSED.load(Ordering::SeqCst), 64);
    }

    // Protects: Exact UTF-16 field boundary including NUL.
    // Detects: Off-by-one admission or silent truncation.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: decode_message / MAX_MESSAGE_CHARS; telemetry::event_log::tests::utf16_field_cap_includes_terminating_nul_and_never_truncates.
    #[test]
    fn utf16_field_cap_includes_terminating_nul_and_never_truncates() {
        let mut units = vec![65; MAX_MESSAGE_CHARS as usize];
        *units.last_mut().unwrap() = 0;
        assert_eq!(
            decode_message(&units, units.len()).unwrap().len(),
            MAX_MESSAGE_CHARS as usize - 1
        );
        units.insert(units.len() - 1, 65);
        assert!(decode_message(&units, units.len()).is_err());
    }

    // Protects: Escaped JSON output bounds and full field values.
    // Detects: Unescaped-size accounting or incorrect exact-bound rejection.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: serialize_events / BoundedJsonWriter; telemetry::event_log::tests::serialized_escaping_respects_exact_and_over_output_boundaries.
    #[test]
    fn serialized_escaping_respects_exact_and_over_output_boundaries() {
        let rows = vec![inert_row("\u{1}".into())];
        // Small independent serialization oracle; only the bounded writer is production.
        let expected = serde_json::to_vec(&rows).unwrap();
        assert_eq!(expected.windows(6).filter(|window| *window == b"\\u0001").count(), 2);
        assert_eq!(serialize_events(&rows, expected.len()).unwrap(), expected);
        assert!(matches!(
            serialize_events(&rows, expected.len() - 1),
            Err(EventLogError::OutputLimit)
        ));
        let decoded: serde_json::Value = serde_json::from_slice(&expected).unwrap();
        assert_eq!(decoded[0]["message"], "\u{1}");
        assert_eq!(decoded[0]["full_message"], "\u{1}");
    }

    // Protects: Atomic failed-write byte retention.
    // Detects: Over-limit write retaining some of its input bytes.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: BoundedJsonWriter::write; telemetry::event_log::tests::bounded_writer_rejects_the_whole_write_without_retaining_a_prefix.
    #[test]
    fn bounded_writer_rejects_the_whole_write_without_retaining_a_prefix() {
        let mut writer = BoundedJsonWriter {
            bytes: Vec::new(),
            limit: 3,
            failure: None,
        };
        assert_eq!(writer.write(b"ab").unwrap(), 2);
        assert!(writer.write(b"cd").is_err());
        assert_eq!(writer.bytes, b"ab");
        assert!(matches!(writer.failure, Some(EventLogError::OutputLimit)));
    }

    // Protects: Actual 16 MiB JSON expansion cap and nullable transport.
    // Detects: Below-text-cap control strings creating unbounded JSON or partial success.
    // Needs: Windows unit-test build; synthetic data/callbacks only, no real Event Log call.
    // Breadcrumb: serialize_events / events_to_ffi; telemetry::event_log::tests::actual_json_expansion_limit_returns_null_without_partial_success.
    #[test]
    fn actual_json_expansion_limit_returns_null_without_partial_success() {
        let rows: Vec<_> = (0..60)
            .map(|_| inert_row("\u{1}".repeat(MAX_MESSAGE_CHARS as usize - 1)))
            .collect();
        assert!(rows.iter().map(|row| row.text_bytes().unwrap()).sum::<usize>() < MAX_RETAINED_TEXT);
        assert!(matches!(
            serialize_events(&rows, MAX_JSON_BYTES),
            Err(EventLogError::OutputLimit)
        ));
        assert!(events_to_ffi(Ok(rows)).is_null());
    }
}
