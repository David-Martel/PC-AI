use anyhow::{Context, Result};
use std::ffi::c_void;

#[derive(Debug, Clone, Copy)]
pub struct CudaMemSnapshot {
    pub free_bytes: u64,
    pub total_bytes: u64,
    pub used_bytes: u64,
}

impl CudaMemSnapshot {
    pub fn free_mb(self) -> u64 {
        self.free_bytes / 1024 / 1024
    }

    pub fn total_mb(self) -> u64 {
        self.total_bytes / 1024 / 1024
    }

    pub fn used_mb(self) -> u64 {
        self.used_bytes / 1024 / 1024
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CudaMemPoolConfig {
    pub enable: bool,
    pub release_threshold_mb: Option<u64>,
    pub reuse_follow_event_dependencies: bool,
    pub reuse_allow_opportunistic: bool,
    pub reuse_allow_internal_dependencies: bool,
    pub trim_to_mb: Option<u64>,
}

impl Default for CudaMemPoolConfig {
    fn default() -> Self {
        Self {
            enable: false,
            release_threshold_mb: None,
            reuse_follow_event_dependencies: true,
            reuse_allow_opportunistic: true,
            reuse_allow_internal_dependencies: true,
            trim_to_mb: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CudaMemPoolStatus {
    pub release_threshold_bytes: Option<u64>,
    pub reserved_current_bytes: Option<u64>,
    pub used_current_bytes: Option<u64>,
}

pub fn cuda_mem_snapshot(device_index: Option<usize>) -> Option<CudaMemSnapshot> {
    use cudarc::runtime::{result, sys};
    if let Some(idx) = device_index {
        unsafe {
            if sys::cudaSetDevice(idx as i32).result().is_err() {
                return None;
            }
        }
    }
    let (free, total) = result::get_mem_info().ok()?;
    let used = total.saturating_sub(free);
    Some(CudaMemSnapshot {
        free_bytes: free as u64,
        total_bytes: total as u64,
        used_bytes: used as u64,
    })
}

pub fn configure_cuda_mem_pool(
    device_index: usize,
    cfg: CudaMemPoolConfig,
) -> Result<Option<CudaMemPoolStatus>> {
    use cudarc::runtime::sys;

    if !cfg.enable {
        return Ok(None);
    }

    unsafe {
        sys::cudaSetDevice(device_index as i32)
            .result()
            .context("cudaSetDevice failed")?;
    }

    let mut pools_supported: i32 = 0;
    unsafe {
        sys::cudaDeviceGetAttribute(
            &mut pools_supported as *mut i32,
            sys::cudaDeviceAttr::cudaDevAttrMemoryPoolsSupported,
            device_index as i32,
        )
        .result()
        .context("cudaDeviceGetAttribute(cudaDevAttrMemoryPoolsSupported) failed")?;
    }
    if pools_supported == 0 {
        return Ok(None);
    }

    let mut pool: sys::cudaMemPool_t = std::ptr::null_mut();
    unsafe {
        sys::cudaDeviceGetDefaultMemPool(&mut pool as *mut _, device_index as i32)
            .result()
            .context("cudaDeviceGetDefaultMemPool failed")?;
    }

    unsafe {
        let follow = if cfg.reuse_follow_event_dependencies {
            1u32
        } else {
            0u32
        };
        sys::cudaMemPoolSetAttribute(
            pool,
            sys::cudaMemPoolAttr::cudaMemPoolReuseFollowEventDependencies,
            &follow as *const _ as *mut c_void,
        )
        .result()
        .context("cudaMemPoolSetAttribute(reuse_follow_event_dependencies) failed")?;

        let opportunistic = if cfg.reuse_allow_opportunistic {
            1u32
        } else {
            0u32
        };
        sys::cudaMemPoolSetAttribute(
            pool,
            sys::cudaMemPoolAttr::cudaMemPoolReuseAllowOpportunistic,
            &opportunistic as *const _ as *mut c_void,
        )
        .result()
        .context("cudaMemPoolSetAttribute(reuse_allow_opportunistic) failed")?;

        let internal = if cfg.reuse_allow_internal_dependencies {
            1u32
        } else {
            0u32
        };
        sys::cudaMemPoolSetAttribute(
            pool,
            sys::cudaMemPoolAttr::cudaMemPoolReuseAllowInternalDependencies,
            &internal as *const _ as *mut c_void,
        )
        .result()
        .context("cudaMemPoolSetAttribute(reuse_allow_internal_dependencies) failed")?;
    }

    if let Some(threshold_mb) = cfg.release_threshold_mb {
        let threshold_bytes = threshold_mb.saturating_mul(1024 * 1024);
        unsafe {
            sys::cudaMemPoolSetAttribute(
                pool,
                sys::cudaMemPoolAttr::cudaMemPoolAttrReleaseThreshold,
                &threshold_bytes as *const _ as *mut c_void,
            )
            .result()
            .context("cudaMemPoolSetAttribute(release_threshold) failed")?;
        }
    }

    if let Some(trim_mb) = cfg.trim_to_mb {
        let trim_bytes = trim_mb.saturating_mul(1024 * 1024);
        let trim_bytes = usize::try_from(trim_bytes).unwrap_or(usize::MAX);
        unsafe {
            sys::cudaMemPoolTrimTo(pool, trim_bytes)
                .result()
                .context("cudaMemPoolTrimTo failed")?;
        }
    }

    let mut release_threshold: u64 = 0;
    let mut reserved_current: u64 = 0;
    let mut used_current: u64 = 0;
    unsafe {
        let _ = sys::cudaMemPoolGetAttribute(
            pool,
            sys::cudaMemPoolAttr::cudaMemPoolAttrReleaseThreshold,
            &mut release_threshold as *mut _ as *mut c_void,
        );
        let _ = sys::cudaMemPoolGetAttribute(
            pool,
            sys::cudaMemPoolAttr::cudaMemPoolAttrReservedMemCurrent,
            &mut reserved_current as *mut _ as *mut c_void,
        );
        let _ = sys::cudaMemPoolGetAttribute(
            pool,
            sys::cudaMemPoolAttr::cudaMemPoolAttrUsedMemCurrent,
            &mut used_current as *mut _ as *mut c_void,
        );
    }

    Ok(Some(CudaMemPoolStatus {
        release_threshold_bytes: Some(release_threshold),
        reserved_current_bytes: Some(reserved_current),
        used_current_bytes: Some(used_current),
    }))
}
