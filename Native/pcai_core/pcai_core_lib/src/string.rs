//! String buffer utilities for FFI.
use crate::error::PcaiStatus;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[repr(C)]
#[derive(Debug)]
pub struct PcaiStringBuffer {
    pub status: PcaiStatus,
    pub data: *mut c_char,
    pub length: usize,
}

#[repr(C)]
#[derive(Debug)]
pub struct PcaiByteBuffer {
    pub status: PcaiStatus,
    pub data: *mut u8,
    pub length: usize,
}

impl PcaiStringBuffer {
    pub fn from_string(s: &str) -> Self {
        match CString::new(s) {
            Ok(c_string) => {
                let len = c_string.as_bytes().len();
                Self {
                    status: PcaiStatus::Success,
                    data: c_string.into_raw(),
                    length: len,
                }
            }
            Err(_) => Self::error(PcaiStatus::InvalidUtf8),
        }
    }
    pub fn error(status: PcaiStatus) -> Self {
        Self {
            status,
            data: std::ptr::null_mut(),
            length: 0,
        }
    }
    pub fn null() -> Self {
        Self {
            status: PcaiStatus::NullPointer,
            data: std::ptr::null_mut(),
            length: 0,
        }
    }
    pub fn is_valid(&self) -> bool {
        self.status.is_success() && !self.data.is_null()
    }
}

impl Default for PcaiStringBuffer {
    fn default() -> Self {
        Self::null()
    }
}

impl PcaiByteBuffer {
    pub fn from_vec(mut data: Vec<u8>) -> Self {
        let len = data.len();
        let ptr = data.as_mut_ptr();
        std::mem::forget(data);

        Self {
            status: PcaiStatus::Success,
            data: ptr,
            length: len,
        }
    }

    pub fn error(status: PcaiStatus) -> Self {
        Self {
            status,
            data: std::ptr::null_mut(),
            length: 0,
        }
    }

    pub fn null() -> Self {
        Self::error(PcaiStatus::NullPointer)
    }

    pub fn is_valid(&self) -> bool {
        self.status.is_success() && !self.data.is_null()
    }
}

impl Default for PcaiByteBuffer {
    fn default() -> Self {
        Self::null()
    }
}

#[no_mangle]
pub extern "C" fn pcai_create_string_buffer(input: *const c_char) -> PcaiStringBuffer {
    if input.is_null() {
        return PcaiStringBuffer::null();
    }
    let c_str = unsafe { CStr::from_ptr(input) };
    match c_str.to_str() {
        Ok(s) => PcaiStringBuffer::from_string(s),
        Err(_) => PcaiStringBuffer::error(PcaiStatus::InvalidUtf8),
    }
}

#[no_mangle]
pub extern "C" fn pcai_free_string_buffer(buffer: *mut PcaiStringBuffer) {
    if !buffer.is_null() {
        let buf = unsafe { &mut *buffer };
        if !buf.data.is_null() {
            unsafe {
                let _ = CString::from_raw(buf.data);
            }
            buf.data = std::ptr::null_mut();
            buf.length = 0;
        }
    }
}

#[no_mangle]
pub extern "C" fn pcai_free_byte_buffer(buffer: *mut PcaiByteBuffer) {
    if !buffer.is_null() {
        let buf = unsafe { &mut *buffer };
        if !buf.data.is_null() {
            unsafe {
                let _ = Vec::from_raw_parts(buf.data, buf.length, buf.length);
            }
            buf.data = std::ptr::null_mut();
            buf.length = 0;
        }
    }
}

pub fn rust_str_to_c(s: &str) -> *mut c_char {
    match CString::new(s) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Convert a C string to a Rust string slice.
///
/// # Safety
///
/// `ptr` must be a valid, null-terminated C string.
pub unsafe fn c_str_to_rust<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok()
}

pub fn json_to_buffer<T: serde::Serialize>(value: &T) -> PcaiStringBuffer {
    match serde_json::to_string(value) {
        Ok(json) => PcaiStringBuffer::from_string(&json),
        Err(_) => PcaiStringBuffer::error(PcaiStatus::JsonError),
    }
}

pub fn json_to_buffer_pretty<T: serde::Serialize>(value: &T) -> PcaiStringBuffer {
    match serde_json::to_string_pretty(value) {
        Ok(json) => PcaiStringBuffer::from_string(&json),
        Err(_) => PcaiStringBuffer::error(PcaiStatus::JsonError),
    }
}

pub fn rust_str_to_buffer(s: &str) -> PcaiStringBuffer {
    PcaiStringBuffer::from_string(s)
}

pub fn bytes_to_buffer(data: Vec<u8>) -> PcaiByteBuffer {
    PcaiByteBuffer::from_vec(data)
}

pub fn error_buffer(status: PcaiStatus) -> PcaiStringBuffer {
    PcaiStringBuffer::error(status)
}
