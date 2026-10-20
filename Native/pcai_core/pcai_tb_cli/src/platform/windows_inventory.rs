//! Presence and classification policy for Windows USB4 instance inventory.
use super::Usb4DeviceInfo;
use anyhow::Result;

/// Collect present instances without inferring NIC identity or device health.
pub(super) fn collect_present_usb4_devices(
    mut enumerate: impl FnMut(bool) -> Result<Vec<String>>,
) -> Result<Vec<Usb4DeviceInfo>> {
    let ids = enumerate(true)?;
    Ok(ids
        .into_iter()
        .filter_map(|id| {
            let upper = id.to_ascii_uppercase();
            if !upper.starts_with("USB4\\") {
                return None;
            }
            let friendly_name = if upper.starts_with("USB4\\VIRTUAL_POWER_PDO") {
                "USB4 Virtual power coordination device"
            } else {
                "USB4 device instance"
            };
            Some(Usb4DeviceInfo {
                instance_id: id,
                friendly_name: friendly_name.to_string(),
                class_name: "Unknown".to_string(),
                status: "Unknown".to_string(),
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Protects current inventory; detects requesting registered ghost devices.
    // Needs an in-memory provider with distinct present/ghost records.
    // Breadcrumb: collect_present_usb4_devices and windows::enumerate_usb4_pnp_devices.
    #[test]
    fn present_inventory_excludes_provider_ghosts() {
        let records = [("USB4\\PRESENT", true), ("USB4\\GHOST", false)];
        let devices = collect_present_usb4_devices(|present_only| {
            Ok(records
                .iter()
                .filter(|(_, present)| !present_only || *present)
                .map(|(id, _)| (*id).to_string())
                .collect())
        })
        .unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].instance_id, "USB4\\PRESENT");
    }

    // Protects honest classification; detects fabricated NIC or healthy status.
    // Needs present virtual-power and protocol instance fixtures.
    // Breadcrumb: collect_present_usb4_devices classification.
    #[test]
    fn virtual_power_and_protocol_instances_do_not_claim_nic_or_health() {
        let devices = collect_present_usb4_devices(|_| {
            Ok(vec![
                "USB4\\VIRTUAL_POWER_PDO\\fixture".into(),
                "USB4\\VEN_1D6B&PROT_USB4NET\\fixture".into(),
            ])
        })
        .unwrap();
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].friendly_name, "USB4 Virtual power coordination device");
        for device in devices {
            assert_eq!(device.class_name, "Unknown");
            assert_eq!(device.status, "Unknown");
            assert!(!device.friendly_name.contains("Network Adapter"));
        }
    }

    // Protects error visibility; detects collapsing failed inventory into empty success.
    // Needs a failing provider and an independently successful empty provider.
    // Breadcrumb: collect_present_usb4_devices and collect_windows_snapshot propagation.
    #[test]
    fn provider_failure_remains_distinct_from_empty_inventory() {
        let failure = collect_present_usb4_devices(|_| anyhow::bail!("CONFIGRET fixture failure"));
        assert!(failure.unwrap_err().to_string().contains("CONFIGRET fixture failure"));
        assert!(collect_present_usb4_devices(|_| Ok(Vec::new())).unwrap().is_empty());
    }
}
