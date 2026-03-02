use std::collections::HashMap;
use std::sync::OnceLock;

pub struct CmProblemInfo {
    pub code: u32,
    pub short_description: &'static str,
    pub help_summary: &'static str,
    pub help_url: &'static str,
}

static PROBLEM_MAP: OnceLock<HashMap<u32, CmProblemInfo>> = OnceLock::new();

pub fn get_problem_info(code: u32) -> Option<&'static CmProblemInfo> {
    let map = PROBLEM_MAP.get_or_init(|| {
        let mut m = HashMap::new();

        let codes = [
            (
                1,
                "CM_PROB_NOT_CONFIGURED",
                "Device not configured correctly",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-not-configured",
            ),
            (2, "CM_PROB_DEVLOADER_FAILED", "Windows cannot load the driver", ""),
            (3, "CM_PROB_OUT_OF_MEM", "Driver corrupted or system low on memory", ""),
            (
                4,
                "CM_PROB_WRONG_TYPE",
                "Device not working properly (driver or registry issue)",
                "",
            ),
            (
                5,
                "CM_PROB_PARTIAL_LOG_CONF",
                "Driver requires resource Windows cannot manage",
                "",
            ),
            (6, "CM_PROB_NO_VALID_LOG_CONF", "Boot configuration conflict", ""),
            (7, "CM_PROB_INVALID_FILTER_STR", "Cannot filter", ""),
            (8, "CM_PROB_DEVLOADER_NOT_FOUND", "Driver loader missing", ""),
            (9, "CM_PROB_INVALID_ID", "Firmware reporting resources incorrectly", ""),
            (
                10,
                "CM_PROB_FAILED_START",
                "Device cannot start",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-failed-start",
            ),
            (11, "CM_PROB_LIAR", "Device failed", ""),
            (
                12,
                "CM_PROB_NORMAL_CONFLICT",
                "Cannot find enough free resources",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-conflicting-resources",
            ),
            (13, "CM_PROB_NOT_VERIFIED", "Cannot verify resources", ""),
            (
                14,
                "CM_PROB_NEED_RESTART",
                "Restart required",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-need-restart",
            ),
            (15, "CM_PROB_GREEN_RESUME", "Re-enumeration problem", ""),
            (16, "CM_PROB_REGISTRY_SET_FAILED", "Cannot identify all resources", ""),
            (17, "CM_PROB_BAD_CONFIG_ID", "Unknown resource type requested", ""),
            (
                18,
                "CM_PROB_REINSTALL",
                "Reinstall drivers",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-reinstall",
            ),
            (
                19,
                "CM_PROB_REGISTRY_BAD",
                "Registry failure",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-registry-bad",
            ),
            (20, "CM_PROB_DEVICE_NOT_THERE", "VxD loader failure", ""), // Note: PS script says VxD, but code 20 is related to this.
            (
                21,
                "CM_PROB_WILL_BE_REMOVED",
                "System failure",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-will-be-removed",
            ),
            (
                22,
                "CM_PROB_DISABLED",
                "Device is disabled",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-disabled",
            ),
            (23, "CM_PROB_DEVLOADER_NOT_READY", "System failure", ""),
            (
                24,
                "CM_PROB_DEVICE_NOT_THERE",
                "Device missing or not working",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-device-not-there",
            ),
            (25, "CM_PROB_MOVED", "Setup incomplete", ""),
            (26, "CM_PROB_TOO_EARLY", "Setup incomplete", ""),
            (27, "CM_PROB_NO_VALID_LOG_CONF", "Invalid log configuration", ""),
            (
                28,
                "CM_PROB_FAILED_INSTALL",
                "Drivers not installed",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-failed-install",
            ),
            (
                29,
                "CM_PROB_HARDWARE_DISABLED",
                "Firmware did not provide resources",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-hardware-disabled",
            ),
            (30, "CM_PROB_CANT_SHARE_IRQ", "IRQ conflict", ""),
            (
                31,
                "CM_PROB_FAILED_ADD",
                "Device not working properly",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-failed-add",
            ),
            (
                32,
                "CM_PROB_REGISTRY_QUARANTINE",
                "Driver service disabled",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-registry-quarantine",
            ),
            (
                33,
                "CM_PROB_FAIL_REPORTED_WMI",
                "Cannot determine resource requirements",
                "",
            ),
            (
                34,
                "CM_PROB_FAILED_CONFIG",
                "Cannot determine device settings",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-failed-config",
            ),
            (
                35,
                "CM_PROB_PHANTOM",
                "Cannot determine device settings (missing firmware)",
                "",
            ),
            (36, "CM_PROB_RELATIVE_RESOURCE_NOT_FOUND", "PCI IRQ conflict", ""),
            (
                37,
                "CM_PROB_FAILED_DRIVER_ENTRY",
                "Cannot initialize",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-failed-driver-entry",
            ),
            (
                38,
                "CM_PROB_DRIVER_FAILED_PRIOR_UNLOAD",
                "Cannot load driver (already loaded by another device)",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-driver-failed-prior-unload",
            ),
            (
                39,
                "CM_PROB_DRIVER_FAILED_LOAD",
                "Cannot load driver (driver corrupted)",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-driver-failed-load",
            ),
            (
                40,
                "CM_PROB_DRIVER_SERVICE_KEY_INVALID",
                "Service key information missing",
                "",
            ),
            (41, "CM_PROB_LEGACY_SERVICE_ERROR", "Cannot load driver", ""),
            (42, "CM_PROB_NON_SPECIFIC_FAULT", "Duplicate device running", ""),
            (
                43,
                "CM_PROB_DEVICE_REPORTED_FAILURE",
                "Device stopped responding (Code 43)",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-device-reported-failure",
            ),
            (
                44,
                "CM_PROB_FAILED_ENUMERATION",
                "Application or service shut down device",
                "",
            ),
            (
                45,
                "CM_PROB_NOT_CONNECTED",
                "Device not connected",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-not-connected",
            ),
            (
                46,
                "CM_PROB_HARDWARE_REMOVAL",
                "Cannot access device (Windows shutting down)",
                "",
            ),
            (
                47,
                "CM_PROB_WILL_BE_REMOVED",
                "Safe removal prepared",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-will-be-removed",
            ),
            (
                48,
                "CM_PROB_DISABLED_SERVICE",
                "Firmware has blocked device",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-disabled-service",
            ),
            (49, "CM_PROB_RESOURCES_BIT_SET", "Registry size limit exceeded", ""),
            (50, "CM_PROB_RESOURCES_BIT_SET", "Cannot apply properties", ""),
            (51, "CM_PROB_RESOURCES_BIT_SET", "Device waiting on another device", ""),
            (
                52,
                "CM_PROB_UNSIGNED_DRIVER",
                "Cannot verify digital signature",
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/install/cm-prob-unsigned-driver",
            ),
            (53, "CM_PROB_RESERVED", "Reserved for Windows", ""),
            (54, "CM_PROB_RESERVED", "ACPI failure", ""),
        ];

        for (code, short, summary, url) in codes {
            m.insert(
                code,
                CmProblemInfo {
                    code,
                    short_description: short,
                    help_summary: summary,
                    help_url: url,
                },
            );
        }

        m
    });

    map.get(&code)
}
