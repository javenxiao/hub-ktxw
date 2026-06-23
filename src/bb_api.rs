//! 基带 API 的 Rust 安全包装
//!
//! 提供类型安全的高级接口，用于与基带芯片 (ar8030) SOC 通信

use std::{cell::RefCell, collections::HashMap, sync::{Arc, Mutex, MutexGuard}, thread, time::{Duration, Instant}};
use crate::ffi;

const REMOTE_SDK_CALL_GAP: Duration = Duration::from_millis(20);
const REMOTE_HOT_UPGRADE_SDK_CALL_GAP: Duration = Duration::from_millis(0);
const REMOTE_DEVICE_SWITCH_GAP: Duration = Duration::from_millis(1200);
const REMOTE_STATUS_SNAPSHOT_REFRESH_GAP: Duration = Duration::from_secs(2);
const UPGRADE_PROGRESS_UPDATE_INTERVAL: Duration = Duration::from_millis(100);
const HOT_UPGRADE_WRITE_SEQ: u16 = 5673;
const HOT_UPGRADE_CRC_SEQ: u16 = 3673;
const BB_USER_0: u8 = 0;
static REMOTE_SDK_LAST_CALL_AT: Mutex<Option<Instant>> = Mutex::new(None);

// ============================================================
// OTA upgrade image format (matching PC Tool ar8030_upgrade_ota.cpp)
// ============================================================
// These constants document the .img file layout and are referenced
// by upgrade_ota_image() calculations below.
#[allow(dead_code)]
const OTA_HDR_SIZE: usize = 256;
#[allow(dead_code)]
const OTA_HASH_SIZE: usize = 32;
#[allow(dead_code)]
const OTA_SIG_SIZE: usize = 256;
#[allow(dead_code)]
const GPT_FLASH_OFFSET: u64 = 0x8000;
#[allow(dead_code)]
const GPT_FLASH_SIZE: u64 = 0x1000;
#[allow(dead_code)]
const HDR_MAGIC: usize = 0;
#[allow(dead_code)]
const HDR_HEADER_EXT_SIZE: usize = 16;
#[allow(dead_code)]
const HDR_HASH_SIZE_OFF: usize = 20;
#[allow(dead_code)]
const HDR_SIG_SIZE_OFF: usize = 24;
#[allow(dead_code)]
const HDR_IMG_SIZE: usize = 32;
#[allow(dead_code)]
const HDR_ROM_SIZE: usize = 40;
#[allow(dead_code)]
const HDR_LOADER_SIZE: usize = 44;
#[allow(dead_code)]
const HDR_PARTITIONS: usize = 48;
#[allow(dead_code)]
const HDR_SEGMENTS: usize = 52;
#[allow(dead_code)]
const HDR_IMAGE_TYPE: usize = 128;
#[allow(dead_code)]
const HDR_PART_STATUS: usize = 136;
#[allow(dead_code)]
const PART_INFO_SIZE: usize = 32;
#[allow(dead_code)]
const SEGMENT_INFO_SIZE: usize = 32;

pub(crate) fn resolve_plot_user(status: &ffi::BbGetStatusSummary) -> u8 {
    status.active_user.or(status.detected_active_user).unwrap_or(BB_USER_0)
}

fn run_remote_sdk_call_with_gap<T>(
    enabled: bool,
    min_gap: Duration,
    operation: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    if enabled && !min_gap.is_zero() {
        let sleep_duration = {
            let last_call = REMOTE_SDK_LAST_CALL_AT.lock().unwrap();
            last_call
                .and_then(|instant| min_gap.checked_sub(instant.elapsed()))
        };

        if let Some(duration) = sleep_duration {
            thread::sleep(duration);
        }
    }

    let result = operation();

    if enabled {
        let mut last_call = REMOTE_SDK_LAST_CALL_AT.lock().unwrap();
        *last_call = Some(Instant::now());
    }

    result
}

fn run_remote_sdk_call<T>(enabled: bool, operation: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    run_remote_sdk_call_with_gap(enabled, REMOTE_SDK_CALL_GAP, operation)
}

#[derive(Debug, Clone)]
struct RemoteHostConfig {
    address: String,
    port: i32,
}

struct RemoteSessionConnectOutcome {
    status: ffi::BbGetStatusSummary,
    device_count: i32,
    daemon_version: Option<String>,
}

#[derive(Debug, Clone)]
struct CachedSystemInfoEntry {
    summary: ffi::BbSystemInfoSummary,
    cached_at: Instant,
}

impl CachedSystemInfoEntry {
    fn new(summary: &ffi::BbSystemInfoSummary) -> Self {
        Self {
            summary: summary.clone(),
            cached_at: Instant::now(),
        }
    }

    fn materialize(&self) -> ffi::BbSystemInfoSummary {
        let mut summary = self.summary.clone();
        summary.uptime = summary.uptime.saturating_add(self.cached_at.elapsed().as_secs());
        summary
    }
}

#[derive(Debug, Clone)]
struct CachedStatusSummaryEntry {
    summary: ffi::BbGetStatusSummary,
    cached_at: Instant,
}

impl CachedStatusSummaryEntry {
    fn new(summary: &ffi::BbGetStatusSummary) -> Self {
        Self {
            summary: summary.clone(),
            cached_at: Instant::now(),
        }
    }

    fn is_fresh(&self, max_age: Duration) -> bool {
        self.cached_at.elapsed() <= max_age
    }
}

fn remote_host_config_from_env() -> Option<RemoteHostConfig> {
    let address = std::env::var("BB_HOST_ADDR").ok()?;
    let port = std::env::var("BB_HOST_PORT")
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(0);

    Some(RemoteHostConfig { address, port })
}

fn non_empty_trimmed(value: String) -> Option<String> {
    let trimmed = value.trim();

    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn normalized_release_value(value: &str) -> Option<String> {
    let trimmed = value.trim();

    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("unavailable") {
        None
    } else {
        Some(trimmed.to_ascii_lowercase())
    }
}

fn release_values_mismatch(current: &ffi::BbSystemInfoSummary, peer: &ffi::BbSystemInfoSummary) -> bool {
    let firmware_mismatch = match (
        normalized_release_value(&current.firmware_version),
        normalized_release_value(&peer.firmware_version),
    ) {
        (Some(left), Some(right)) => left != right,
        _ => false,
    };

    let software_mismatch = match (
        normalized_release_value(&current.software_version),
        normalized_release_value(&peer.software_version),
    ) {
        (Some(left), Some(right)) => left != right,
        _ => false,
    };

    firmware_mismatch || software_mismatch
}

fn release_summary(info: &ffi::BbSystemInfoSummary) -> String {
    let firmware = non_empty_trimmed(info.firmware_version.clone()).unwrap_or_else(|| "--".to_string());
    let software = non_empty_trimmed(info.software_version.clone()).unwrap_or_else(|| "--".to_string());

    format!("FW {} / SW {}", firmware, software)
}

fn role_name_for_warning(role: Option<u8>) -> &'static str {
    match role {
        Some(ffi::BB_ROLE_AP) => "AP",
        Some(ffi::BB_ROLE_DEV) => "DEV",
        _ => "Peer",
    }
}

fn normalized_pair_peer_mac(value: &str) -> Option<String> {
    let normalized = value
        .chars()
        .filter(|character| character.is_ascii_hexdigit())
        .flat_map(|character| character.to_lowercase())
        .collect::<String>();

    if normalized.len() == ffi::BB_MAC_LEN * 2 {
        Some(normalized)
    } else {
        None
    }
}

fn paired_peer_target_for_dev(status: &ffi::BbGetStatusSummary) -> Option<String> {
    let local_mac = normalized_pair_peer_mac(&status.mac_hex);
    let connected_peer = status
        .links
        .iter()
        .find(|link| link.state != 0)
        .and_then(|link| link.peer_mac_hex.as_deref())
        .and_then(normalized_pair_peer_mac)
        .or_else(|| {
            if status.link_state.unwrap_or(0) != 0 {
                status
                    .peer_mac_hex
                    .as_deref()
                    .and_then(normalized_pair_peer_mac)
            } else {
                None
            }
        });

    connected_peer.filter(|peer_mac| local_mac.as_deref() != Some(peer_mac.as_str()))
}

fn paired_peer_targets_for_ap(status: &ffi::BbGetStatusSummary) -> Vec<(u8, String)> {
    let local_mac = normalized_pair_peer_mac(&status.mac_hex);

    status
        .links
        .iter()
        .filter_map(|link| {
            if link.state == 0 {
                return None;
            }

            let slot = u8::try_from(link.slot).ok()?;
            let peer_mac = link
                .peer_mac_hex
                .as_deref()
                .and_then(normalized_pair_peer_mac)?;

            if local_mac.as_deref() == Some(peer_mac.as_str()) {
                return None;
            }

            Some((slot, peer_mac))
        })
        .collect()
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BasebandOperationStatus {
    pub attempted: bool,
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BasebandHostStatus {
    pub configured: bool,
    pub address: Option<String>,
    pub port: Option<i32>,
    pub connected: bool,
    pub daemon_version: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BasebandRuntimeStatus {
    pub detected_device_count: Option<i32>,
    pub status_snapshot: Option<ffi::BbGetStatusSummary>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BasebandHealthStatus {
    pub configured_mode: String,
    pub effective_mode: String,
    pub host: BasebandHostStatus,
    pub runtime: BasebandRuntimeStatus,
    pub sdk: ffi::FfiRuntimeDiagnostics,
    pub device_open: BasebandOperationStatus,
    pub init: BasebandOperationStatus,
    pub start: BasebandOperationStatus,
    pub status_read: BasebandOperationStatus,
    pub socket_init: BasebandOperationStatus,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WirelessRuntimeDetails {
    pub status: ffi::BbGetStatusSummary,
    pub dev_pair_target_mac: Option<String>,
    pub ap_pair_target_macs: Vec<Option<String>>,
    pub available_devices: Vec<ffi::BbDiscoveredDeviceSummary>,
    pub system_info: Option<ffi::BbSystemInfoSummary>,
    pub band_info: Option<ffi::BbBandInfoSummary>,
    pub channel_info: Option<ffi::BbChannelInfoSummary>,
    pub bandwidth_mode: Option<ffi::BbBandwidthModeSummary>,
    pub mcs_mode: Option<ffi::BbMcsModeSummary>,
    pub mcs_value: Option<ffi::BbMcsValueSummary>,
    pub power_mode: Option<ffi::BbPowerModeSummary>,
    pub current_power: Option<ffi::BbCurrentPowerSummary>,
    pub power_auto: Option<ffi::BbPowerAutoSummary>,
    pub power_fallback: Option<ffi::BbPowerFallback>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WirelessConfigurationMinidbDetails {
    pub role: Option<u8>,
    pub band_bitmap: Option<u8>,
    pub local_mac: Option<String>,
    pub ap_mac: Option<String>,
    pub slot_macs: Vec<Option<String>>,
    pub power: Option<ffi::BbMinidbPowerSummary>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WirelessConfigurationDetails {
    pub config_mode: u8,
    pub config_text: String,
    pub minidb: WirelessConfigurationMinidbDetails,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BootDiagnostics {
    pub running_system: String,
    pub boot_reason: String,
}

#[derive(Debug, Clone)]
pub struct FirmwareUpgradeResult {
    pub crc32: u32,
}

/// 固件升级实时进度
#[derive(Debug, Clone, serde::Serialize)]
pub struct FirmwareUpgradeProgress {
    pub state: String,             // "idle" | "uploading" | "flashing" | "done" | "error"
    pub file_name: String,
    pub file_size: usize,
    pub bytes_written: usize,
    pub http_upload_elapsed_ms: u64,
    pub board_write_elapsed_ms: u64,
    pub total_steps: usize,
    pub current_step: usize,
    pub step_label: String,
    pub percent: f64,              // 0.0 ~ 100.0
    pub message: String,
    pub crc32: Option<String>,
    pub reboot_expected: bool,
}

impl BasebandHealthStatus {
    fn new() -> Self {
        let host_address = std::env::var("BB_HOST_ADDR").ok();
        let host_port = std::env::var("BB_HOST_PORT")
            .ok()
            .and_then(|value| value.parse::<i32>().ok());
        let configured = host_address.is_some();

        Self {
            configured_mode: if configured {
                "remote-bb-host".to_string()
            } else {
                "local-sdk".to_string()
            },
            effective_mode: "simulator".to_string(),
            host: BasebandHostStatus {
                configured,
                address: host_address,
                port: host_port,
                connected: false,
                daemon_version: None,
                message: if configured {
                    "BB_HOST_ADDR detected; remote host mode configured".to_string()
                } else {
                    "BB_HOST_ADDR not set; local SDK mode configured".to_string()
                },
            },
            runtime: BasebandRuntimeStatus {
                detected_device_count: None,
                status_snapshot: None,
            },
            sdk: ffi::runtime_diagnostics(),
            device_open: BasebandOperationStatus {
                attempted: false,
                success: false,
                message: "Not attempted".to_string(),
            },
            init: BasebandOperationStatus {
                attempted: false,
                success: false,
                message: "Not attempted".to_string(),
            },
            start: BasebandOperationStatus {
                attempted: false,
                success: false,
                message: "Not attempted".to_string(),
            },
            status_read: BasebandOperationStatus {
                attempted: false,
                success: false,
                message: "Not attempted".to_string(),
            },
            socket_init: BasebandOperationStatus {
                attempted: false,
                success: false,
                message: "Not attempted".to_string(),
            },
        }
    }

    pub fn record_socket_init(&mut self, result: Result<(), String>, socket_id: u32) {
        self.socket_init.attempted = true;

        match result {
            Ok(()) => {
                self.socket_init.success = true;
                self.socket_init.message = format!("Socket {} initialized successfully", socket_id);
            }
            Err(err) => {
                self.socket_init.success = false;
                self.socket_init.message = format!("Socket {} initialization failed: {}", socket_id, err);
            }
        }
    }

    pub fn primary_failure_message(&self) -> String {
        if self.start.attempted && !self.start.success {
            return self.start.message.clone();
        }

        if self.init.attempted && !self.init.success {
            return self.init.message.clone();
        }

        if self.status_read.attempted && !self.status_read.success {
            return self.status_read.message.clone();
        }

        if self.device_open.attempted && !self.device_open.success {
            return self.device_open.message.clone();
        }

        self.host.message.clone()
    }
}

/// 基带 API 管理器 - 线程安全的单例
pub struct BasebandApi {
    initialized: bool,
    started: bool,
    requires_start: bool,
    device_handle: usize,
    host_handle: usize,
    remote_host: Option<RemoteHostConfig>,
    host_devices_cache: RefCell<Vec<ffi::BbDiscoveredDeviceSummary>>,
    plot_subscription_active: bool,
    plot_user: Option<u8>,
    active_device_mac: Option<String>,
    preferred_signal_users: HashMap<String, u8>,
    device_role_cache: HashMap<String, u8>,
    device_sync_role_cache: HashMap<String, (u8, u8)>,
    bandwidth_mode_cache: HashMap<(String, u8), bool>,
    status_summary_cache: HashMap<String, CachedStatusSummaryEntry>,
    channel_info_cache: HashMap<String, ffi::BbChannelInfoSummary>,
    band_selection_bitmap_cache: HashMap<String, Option<u8>>,
    ap_pair_target_cache: HashMap<String, Vec<Option<String>>>,
    boot_diagnostics_cache: HashMap<String, BootDiagnostics>,
    system_info_cache: HashMap<String, CachedSystemInfoEntry>,
    last_remote_device_switch_at: Option<Instant>,
    remote_device_handles: Vec<(String, usize)>,
    /// 固件升级进度追踪器（Arc<Mutex<>> 允许跨锁共享）
    upgrade_progress: Option<Arc<Mutex<FirmwareUpgradeProgress>>>,
    upgrade_board_write_started_at: Option<Instant>,
}

impl BasebandApi {
    fn normalize_mac(value: &str) -> String {
        value
            .chars()
            .filter(|character| character.is_ascii_hexdigit())
            .flat_map(|character| character.to_lowercase())
            .collect()
    }

    fn is_remote_mode(&self) -> bool {
        self.remote_host.is_some()
    }

    fn current_upgrade_board_write_elapsed_ms(&self) -> u64 {
        self.upgrade_board_write_started_at
            .map(|started_at| started_at.elapsed().as_millis().min(u64::MAX as u128) as u64)
            .unwrap_or(0)
    }

    fn empty_ap_pair_target_macs() -> Vec<Option<String>> {
        vec![None; ffi::BB_SLOT_MAX]
    }

    fn band_bitmap_for_target_band(target_band: u8) -> Option<u8> {
        match target_band {
            0 => Some(0x01),
            1 => Some(0x02),
            2 => Some(0x04),
            _ => None,
        }
    }

    fn persist_paired_peer_targets_to_minidb(
        &mut self,
        status: &ffi::BbGetStatusSummary,
    ) -> Result<(), String> {
        let is_remote = self.is_remote_mode();
        let handle = self.handle_ptr();

        match status.role {
            ffi::BB_ROLE_DEV => {
                let Some(peer_mac) = paired_peer_target_for_dev(status) else {
                    return Ok(());
                };

                let current_ap_mac = run_remote_sdk_call(is_remote, || ffi::get_minidb_ap_mac(handle))?;
                if current_ap_mac
                    .as_deref()
                    .and_then(normalized_pair_peer_mac)
                    .as_deref()
                    == Some(peer_mac.as_str())
                {
                    return Ok(());
                }

                run_remote_sdk_call(is_remote, || ffi::set_minidb_ap_mac(handle, &peer_mac))
                    .map_err(|err| format!("Failed to persist paired AP MAC {} to MiniDB: {}", peer_mac, err))
            }
            ffi::BB_ROLE_AP => {
                let mut current_slot_macs = self.read_ap_pair_targets_for_active_device()?;

                for (slot, peer_mac) in paired_peer_targets_for_ap(status) {
                    let current_slot_mac = current_slot_macs
                        .get(slot as usize)
                        .cloned()
                        .unwrap_or(None);

                    if current_slot_mac
                        .as_deref()
                        .and_then(normalized_pair_peer_mac)
                        .as_deref()
                        == Some(peer_mac.as_str())
                    {
                        continue;
                    }

                    run_remote_sdk_call(is_remote, || {
                        ffi::set_minidb_slot_mac(handle, slot, &peer_mac)
                    })
                    .map_err(|err| {
                        format!(
                            "Failed to persist paired DEV MAC {} for slot {} to MiniDB: {}",
                            peer_mac,
                            slot,
                            err
                        )
                    })?;

                    if let Some(entry) = current_slot_macs.get_mut(slot as usize) {
                        *entry = Some(peer_mac.clone());
                    }
                }

                self.cache_ap_pair_targets_for_active_device(&current_slot_macs);

                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn clear_remote_session_state(&mut self) {
        if self.plot_subscription_active && self.device_handle != 0 {
            if let Some(user) = self.plot_user {
                let _ = ffi::unsubscribe_plot_stream(self.handle_ptr(), user);
            }
        }
        self.plot_subscription_active = false;
        self.plot_user = None;

        if self.host_handle != 0 {
            self.close_remote_handles();
        } else if self.device_handle != 0 {
            let _ = ffi::close_device(self.handle_ptr());
            self.device_handle = 0;
        }

        if self.host_handle != 0 {
            let _ = ffi::disconnect_host(self.host_ptr());
            self.host_handle = 0;
        }

        self.initialized = false;
        self.started = false;
        self.host_devices_cache.borrow_mut().clear();
        self.remote_device_handles.clear();
        // Note: device_sync_role_cache is intentionally NOT cleared here.
        // It persists across session resets to preserve M/S labels in the
        // Active Device dropdown even after FSP failures trigger reconnects.
        self.status_summary_cache.clear();
        self.channel_info_cache.clear();
        self.band_selection_bitmap_cache.clear();
        self.ap_pair_target_cache.clear();
        self.boot_diagnostics_cache.clear();
        self.system_info_cache.clear();
    }

    fn invalidate_remote_session_state(&mut self) {
        self.plot_subscription_active = false;
        self.plot_user = None;
        self.initialized = false;
        self.started = false;
        self.device_handle = 0;
        self.host_handle = 0;
        self.host_devices_cache.borrow_mut().clear();
        self.remote_device_handles.clear();
        // Note: device_sync_role_cache is intentionally NOT cleared here.
        self.status_summary_cache.clear();
        self.channel_info_cache.clear();
        self.band_selection_bitmap_cache.clear();
        self.ap_pair_target_cache.clear();
        self.boot_diagnostics_cache.clear();
        self.system_info_cache.clear();
    }

    fn remember_current_device(&mut self, status: &ffi::BbGetStatusSummary) {
        let active_mac = Self::normalize_mac(&status.mac_hex);

        if active_mac.is_empty() {
            self.active_device_mac = None;
            return;
        }

        self.active_device_mac = Some(active_mac.clone());
        self.remember_remote_handle(active_mac, self.device_handle);
        self.cache_role_for_active_device(status.role);
        self.cache_sync_role_for_active_device(status.sync_mode, status.sync_master);
    }

    fn cache_role_for_mac(&mut self, mac: &str, role: u8) {
        if !matches!(role, ffi::BB_ROLE_AP | ffi::BB_ROLE_DEV) {
            return;
        }

        let normalized_mac = Self::normalize_mac(mac);
        if normalized_mac.is_empty() {
            return;
        }

        self.device_role_cache.insert(normalized_mac, role);
    }

    fn cache_role_for_active_device(&mut self, role: u8) {
        if let Some(active_mac) = self.active_device_mac.clone() {
            self.cache_role_for_mac(&active_mac, role);
        }
    }

    fn cache_sync_role_for_mac(&mut self, mac: &str, sync_mode: u8, sync_master: u8) {
        let normalized_mac = Self::normalize_mac(mac);
        if normalized_mac.is_empty() {
            return;
        }

        self.device_sync_role_cache
            .insert(normalized_mac, (sync_mode, sync_master));
    }

    fn cache_sync_role_for_active_device(&mut self, sync_mode: u8, sync_master: u8) {
        if let Some(active_mac) = self.active_device_mac.clone() {
            self.cache_sync_role_for_mac(&active_mac, sync_mode, sync_master);
        }
    }

    fn cached_role_for_mac(&self, normalized_mac: &str) -> Option<u8> {
        self.device_role_cache.get(normalized_mac).copied()
    }

    fn cached_role_for_active_device(&self) -> Option<u8> {
        self.active_device_mac
            .as_ref()
            .and_then(|mac| self.device_role_cache.get(mac))
            .copied()
    }

    fn is_active_device_dev(&self) -> bool {
        self.cached_role_for_active_device() == Some(ffi::BB_ROLE_DEV)
    }

    fn cached_sync_role_for_mac(&self, normalized_mac: &str) -> Option<(u8, u8)> {
        self.device_sync_role_cache.get(normalized_mac).copied()
    }

    fn cache_bandwidth_mode_for_active_device(&mut self, slot: u8, auto_mode: bool) {
        if let Some(active_mac) = self.active_device_mac.clone() {
            self.bandwidth_mode_cache.insert((active_mac, slot), auto_mode);
        }
    }

    fn cache_status_summary_for_active_device(&mut self, status: &ffi::BbGetStatusSummary) {
        if let Some(active_mac) = self.active_device_mac.clone() {
            self.status_summary_cache
                .insert(active_mac, CachedStatusSummaryEntry::new(status));
        }
    }

    fn cached_status_summary_for_active_device(&self) -> Option<ffi::BbGetStatusSummary> {
        let active_mac = self.active_device_mac.as_ref()?;
        let entry = self.status_summary_cache.get(active_mac)?;

        if entry.summary.role != ffi::BB_ROLE_DEV {
            return None;
        }

        Some(entry.summary.clone())
    }

    fn cached_recent_status_summary_for_active_device(&self, max_age: Duration) -> Option<ffi::BbGetStatusSummary> {
        let active_mac = self.active_device_mac.as_ref()?;
        let entry = self.status_summary_cache.get(active_mac)?;

        if entry.summary.role != ffi::BB_ROLE_DEV || !entry.is_fresh(max_age) {
            return None;
        }

        Some(entry.summary.clone())
    }

    fn clear_cached_status_summary_for_active_device(&mut self) {
        if let Some(active_mac) = self.active_device_mac.as_ref() {
            self.status_summary_cache.remove(active_mac);
        }
    }

    fn cache_channel_info_for_active_device(&mut self, channel_info: &ffi::BbChannelInfoSummary) {
        if let Some(active_mac) = self.active_device_mac.clone() {
            self.channel_info_cache.insert(active_mac, channel_info.clone());
        }
    }

    fn cached_channel_info_for_active_device(&self) -> Option<ffi::BbChannelInfoSummary> {
        self.active_device_mac
            .as_ref()
            .and_then(|active_mac| self.channel_info_cache.get(active_mac).cloned())
    }

    fn clear_cached_channel_info_for_active_device(&mut self) {
        if let Some(active_mac) = self.active_device_mac.as_ref() {
            self.channel_info_cache.remove(active_mac);
        }
    }

    fn cache_band_selection_bitmap_for_mac(&mut self, mac: &str, band_bitmap: Option<u8>) {
        let normalized_mac = Self::normalize_mac(mac);
        if normalized_mac.is_empty() {
            return;
        }

        self.band_selection_bitmap_cache.insert(normalized_mac, band_bitmap);
    }

    fn cache_band_selection_bitmap_for_active_device(&mut self, band_bitmap: Option<u8>) {
        if let Some(active_mac) = self.active_device_mac.clone() {
            self.cache_band_selection_bitmap_for_mac(&active_mac, band_bitmap);
        }
    }

    fn cached_band_selection_bitmap_for_mac(&self, normalized_mac: &str) -> Option<Option<u8>> {
        self.band_selection_bitmap_cache.get(normalized_mac).copied()
    }

    fn clear_cached_band_selection_bitmap_for_active_device(&mut self) {
        if let Some(active_mac) = self.active_device_mac.as_ref() {
            self.band_selection_bitmap_cache.remove(active_mac);
        }
    }

    fn cache_boot_diagnostics_for_active_device(&mut self, diagnostics: &BootDiagnostics) {
        if let Some(active_mac) = self.active_device_mac.clone() {
            self.boot_diagnostics_cache.insert(active_mac, diagnostics.clone());
        }
    }

    fn cached_boot_diagnostics_for_active_device(&self) -> Option<BootDiagnostics> {
        self.active_device_mac
            .as_ref()
            .and_then(|active_mac| self.boot_diagnostics_cache.get(active_mac).cloned())
    }

    fn cache_ap_pair_targets_for_mac(&mut self, mac: &str, targets: &[Option<String>]) {
        let normalized_mac = Self::normalize_mac(mac);
        if normalized_mac.is_empty() {
            return;
        }

        let mut normalized_targets = Self::empty_ap_pair_target_macs();
        for (slot, value) in targets.iter().take(ffi::BB_SLOT_MAX).enumerate() {
            normalized_targets[slot] = value.clone();
        }

        self.ap_pair_target_cache.insert(normalized_mac, normalized_targets);
    }

    fn cache_ap_pair_targets_for_active_device(&mut self, targets: &[Option<String>]) {
        if let Some(active_mac) = self.active_device_mac.clone() {
            self.cache_ap_pair_targets_for_mac(&active_mac, targets);
        }
    }

    fn cached_ap_pair_targets_for_mac(&self, normalized_mac: &str) -> Option<Vec<Option<String>>> {
        self.ap_pair_target_cache.get(normalized_mac).cloned()
    }

    fn clear_cached_ap_pair_targets_for_active_device(&mut self) {
        if let Some(active_mac) = self.active_device_mac.as_ref() {
            self.ap_pair_target_cache.remove(active_mac);
        }
    }

    fn update_cached_ap_pair_target_for_active_device(&mut self, slot: u8, mac: Option<&str>) {
        let Some(active_mac) = self.active_device_mac.clone() else {
            return;
        };

        let targets = self
            .ap_pair_target_cache
            .entry(active_mac)
            .or_insert_with(Self::empty_ap_pair_target_macs);

        if let Some(entry) = targets.get_mut(slot as usize) {
            *entry = mac.map(str::to_string);
        }
    }

    fn cache_system_info_for_mac(&mut self, mac: &str, system_info: &ffi::BbSystemInfoSummary) {
        let normalized_mac = Self::normalize_mac(mac);
        if normalized_mac.is_empty() {
            return;
        }

        self.system_info_cache
            .insert(normalized_mac, CachedSystemInfoEntry::new(system_info));
    }

    fn cache_system_info_for_active_device(&mut self, system_info: &ffi::BbSystemInfoSummary) {
        if let Some(active_mac) = self.active_device_mac.clone() {
            self.system_info_cache
                .insert(active_mac, CachedSystemInfoEntry::new(system_info));
        }
    }

    fn cached_system_info_for_mac(&self, normalized_mac: &str) -> Option<ffi::BbSystemInfoSummary> {
        self.system_info_cache
            .get(normalized_mac)
            .map(CachedSystemInfoEntry::materialize)
    }

    fn read_ap_pair_targets_for_active_device(&mut self) -> Result<Vec<Option<String>>, String> {
        if let Some(active_mac) = self.active_device_mac.as_deref() {
            if let Some(cached) = self.cached_ap_pair_targets_for_mac(active_mac) {
                return Ok(cached);
            }
        }

        let is_remote = self.is_remote_mode();
        let mut targets = Self::empty_ap_pair_target_macs();
        for slot in 0..ffi::BB_SLOT_MAX {
            targets[slot] = run_remote_sdk_call(is_remote, || {
                ffi::get_minidb_slot_mac(self.handle_ptr(), slot as u8)
            })?;
        }

        self.cache_ap_pair_targets_for_active_device(&targets);
        Ok(targets)
    }

    fn read_band_selection_bitmap_for_active_device(&mut self) -> Result<Option<u8>, String> {
        if let Some(active_mac) = self.active_device_mac.as_deref() {
            if let Some(cached) = self.cached_band_selection_bitmap_for_mac(active_mac) {
                return Ok(cached);
            }
        }

        let band_bitmap = run_remote_sdk_call(self.is_remote_mode(), || {
            ffi::get_minidb_band_bitmap_optional(self.handle_ptr())
        })?;
        self.cache_band_selection_bitmap_for_active_device(band_bitmap);
        Ok(band_bitmap)
    }

    fn read_channel_info_for_active_device(
        &mut self,
        role: u8,
    ) -> Result<ffi::BbChannelInfoSummary, String> {
        let sdk_result = run_remote_sdk_call(self.is_remote_mode(), || {
            ffi::get_channel_info(self.handle_ptr())
        });

        match sdk_result {
            Ok(channel_info) => {
                self.cache_channel_info_for_active_device(&channel_info);
                Ok(channel_info)
            }
            Err(err) => {
                if self.is_remote_mode() && role == ffi::BB_ROLE_DEV {
                    if let Some(cached) = self.cached_channel_info_for_active_device() {
                        return Ok(cached);
                    }
                }
                Err(err)
            }
        }
    }

    fn read_system_info_for_active_device(&mut self) -> Result<ffi::BbSystemInfoSummary, String> {
        if self.is_remote_mode() {
            if let Some(active_mac) = self.active_device_mac.as_deref() {
                if let Some(cached) = self.cached_system_info_for_mac(active_mac) {
                    return Ok(cached);
                }
            }
        }

        let system_info = run_remote_sdk_call(self.is_remote_mode(), || ffi::get_system_info(self.handle_ptr()))?;
        self.cache_system_info_for_active_device(&system_info);
        Ok(system_info)
    }

    fn cached_bandwidth_mode_for_current_device(&self, slot: u8) -> Option<ffi::BbBandwidthModeSummary> {
        self.active_device_mac
            .as_ref()
            .and_then(|active_mac| self.bandwidth_mode_cache.get(&(active_mac.clone(), slot)).copied())
            .map(|auto_mode| ffi::BbBandwidthModeSummary { slot, auto_mode })
    }

    fn preferred_signal_user_for_mac(&self, normalized_mac: &str) -> Option<u8> {
        self.preferred_signal_users.get(normalized_mac).copied()
    }

    fn preferred_signal_user_for_active_device(&self) -> Option<u8> {
        self.active_device_mac
            .as_deref()
            .and_then(|normalized_mac| self.preferred_signal_user_for_mac(normalized_mac))
    }

    fn ensure_remote_host_connection(&mut self) -> Result<(), String> {
        if !self.is_remote_mode() {
            return Ok(());
        }

        if self.host_handle != 0 {
            return Ok(());
        }

        let remote_host = self
            .remote_host
            .clone()
            .ok_or_else(|| "Remote bb_host mode not configured".to_string())?;

        let host = run_remote_sdk_call(true, || ffi::connect_host(&remote_host.address, remote_host.port))?;
        self.host_handle = host as usize;
        Ok(())
    }

    fn establish_remote_session(&mut self) -> Result<RemoteSessionConnectOutcome, String> {
        let remote_host = self
            .remote_host
            .clone()
            .ok_or_else(|| "Remote bb_host mode not configured".to_string())?;

        let host = run_remote_sdk_call(true, || ffi::connect_host(&remote_host.address, remote_host.port))?;
        self.host_handle = host as usize;

        let daemon_version = ffi::get_daemon_version(host).ok().and_then(non_empty_trimmed);
        let preferred_mac = self.active_device_mac.clone();
        let open_result = if let Some(target_mac) = preferred_mac.as_deref() {
            match run_remote_sdk_call(true, || ffi::open_host_device_by_mac(self.host_ptr(), target_mac)) {
                Ok(result) => Ok(result),
                Err(err) => {
                    tracing::warn!(
                        "Failed to reopen preferred remote device {}: {}. Falling back to first detected device.",
                        target_mac,
                        err
                    );
                    run_remote_sdk_call(true, || ffi::open_first_device_on_host(self.host_ptr()))
                }
            }
        } else {
            run_remote_sdk_call(true, || ffi::open_first_device_on_host(self.host_ptr()))
        };

        let (device_handle, device_count) = match open_result {
            Ok(result) => result,
            Err(err) => {
                let _ = ffi::disconnect_host(self.host_ptr());
                self.host_handle = 0;
                return Err(err);
            }
        };

        let status = match run_remote_sdk_call(true, || {
            ffi::get_status(device_handle, ffi::BB_ALL_DATA_USER_BMP, None)
        }) {
            Ok(status) => status,
            Err(err) => {
                let _ = ffi::close_device(device_handle);
                let _ = ffi::disconnect_host(self.host_ptr());
                self.host_handle = 0;
                return Err(err);
            }
        };

        self.device_handle = device_handle as usize;
        self.initialized = true;
        self.remember_current_device(&status);
        self.cache_status_summary_for_active_device(&status);

        if let Err(err) = self.refresh_host_devices_cache() {
            tracing::warn!("Failed to refresh remote host device cache after connect: {}", err);
        }

        Ok(RemoteSessionConnectOutcome {
            status,
            device_count,
            daemon_version,
        })
    }

    fn ensure_remote_session(&mut self) -> Result<(), String> {
        if !self.is_remote_mode() {
            return Ok(());
        }

        if self.initialized && self.host_handle != 0 && self.device_handle != 0 {
            return Ok(());
        }

        self.clear_remote_session_state();
        let outcome = self.establish_remote_session()?;

        // After reconnection, refresh role/sync caches for ALL remote devices
        // so that the Active Device dropdown shows proper labels (AP/DEV + M/S)
        // immediately, without waiting for the user to click each MAC address.
        // refresh_all_device_status_caches skips already-cached devices, so this
        // is a fast no-op when caches are already populated from a prior session.
        self.refresh_all_device_status_caches();

        let remote_host = self.remote_host.as_ref().expect("remote host config must exist");
        let active_mac = outcome.status.mac_hex.clone();

        if let Some(version) = outcome.daemon_version.as_ref() {
            tracing::info!(
                "Remote bb_host session ready: {}:{} daemon_version={} device_count={} active_device={}",
                remote_host.address,
                remote_host.port,
                version,
                outcome.device_count,
                active_mac
            );
        } else {
            tracing::info!(
                "Remote bb_host session ready: {}:{} device_count={} active_device={}",
                remote_host.address,
                remote_host.port,
                outcome.device_count,
                active_mac
            );
        }

        Ok(())
    }

    fn should_retry_remote_operation(label: &str) -> bool {
        // BB_SET_BANDWIDTH failures can leave the remote SDK in a state where
        // immediate handle teardown/reconnect is not safe.
        // Sweep/FSP operations (configure_sweep, start_sweep, trigger_fsp_scan)
        // are expected to fail on DEV devices — retrying with session reset just
        // makes the UI freeze with no benefit.
        // BB_CFG_CHANNEL (trigger_slave_channel_scan) may not be supported
        // on all firmware versions and also should never retry.
        !matches!(
            label,
            "set_bandwidth"
                | "configure_sweep"
                | "start_sweep"
                | "trigger_fsp_scan"
                | "trigger_slave_channel_scan"
        )
    }

    fn should_invalidate_remote_operation_without_cleanup(label: &str) -> bool {
        matches!(label, "get_status_summary")
    }

    fn should_invalidate_remote_operation_before_retry(err: &str) -> bool {
        let normalized = err.to_ascii_lowercase();
        normalized.contains("bb_get_status")
            || normalized.contains("remote exit")
            || normalized.contains("bb_host_connect")
    }

    fn execute_remote_operation<T>(
        &mut self,
        label: &str,
        mut operation: impl FnMut(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        self.ensure_remote_session()?;

        match operation(self) {
            Ok(value) => Ok(value),
            Err(first_err) => {
                if Self::should_invalidate_remote_operation_without_cleanup(label) {
                    tracing::warn!(
                        "Remote operation '{}' failed: {}. Marking the remote session stale without handle cleanup; a later request will reconnect with fresh handles.",
                        label,
                        first_err
                    );
                    self.invalidate_remote_session_state();
                    return Err(first_err);
                }

                if !Self::should_retry_remote_operation(label) {
                    // trigger_slave_channel_scan / FSP ops 在某些固件版本/角色组合下必然失败，
                    // 失败是预期行为，不应产生 warn 噪声。仅 debug 级别记录。
                    tracing::debug!(
                        "Remote operation '{}' failed (non-retryable): {}. Error is expected for this device/firmware combination.",
                        label,
                        first_err
                    );
                    return Err(first_err);
                }

                tracing::warn!(
                    "Remote operation '{}' failed: {}. Resetting session and retrying once.",
                    label,
                    first_err
                );

                if Self::should_invalidate_remote_operation_before_retry(&first_err) {
                    tracing::warn!(
                        "Remote operation '{}' hit a stale remote session. Retrying without handle cleanup to avoid touching invalid remote handles.",
                        label
                    );
                    self.invalidate_remote_session_state();
                } else {
                    self.clear_remote_session_state();
                }

                self.ensure_remote_session()?;
                operation(self).map_err(|retry_err| {
                    format!(
                        "{}; retry after remote reconnect failed: {}",
                        first_err,
                        retry_err
                    )
                })
            }
        }
    }

    fn with_device_operation<T>(
        &mut self,
        label: &str,
        mut operation: impl FnMut(*mut ffi::bb_dev_handle_t) -> Result<T, String>,
    ) -> Result<T, String> {
        let result = if self.is_remote_mode() {
            self.execute_remote_operation(label, |api| operation(api.handle_ptr()))
        } else {
            if !self.initialized {
                return Err("Baseband API not initialized".to_string());
            }

            operation(self.handle_ptr())
        };

        if result.is_ok() && Self::operation_invalidates_status_summary_cache(label) {
            self.clear_cached_status_summary_for_active_device();
        }

        result
    }

    fn operation_invalidates_status_summary_cache(label: &str) -> bool {
        matches!(
            label,
            "set_pair_mode"
                | "set_pair_candidates"
                | "set_ap_mac"
                | "set_minidb_ap_mac"
                | "set_minidb_local_mac"
                | "set_minidb_role"
                | "set_minidb_power"
                | "save_configuration_text"
                | "clear_flash_configuration"
                | "clear_minidb_configuration"
                | "restore_factory_configuration"
                | "set_minidb_slot_mac"
                | "set_channel_mode"
                | "set_channel"
                | "set_mcs_mode"
                | "set_mcs"
                | "set_power_mode"
                | "set_power"
                | "set_power_auto"
                | "set_band_mode"
                | "set_band"
                | "set_band_selection"
                | "set_bandwidth"
                | "set_bandwidth_mode"
                | "set_baseband_role"
                | "reboot_device"
        )
    }

    fn current_remote_device_count(&self) -> Option<i32> {
        let device_count = self.host_devices_cache.borrow().len();

        if device_count > i32::MAX as usize {
            Some(i32::MAX)
        } else {
            Some(device_count as i32)
        }
    }

    fn snapshot_remote_health(&mut self) -> BasebandHealthStatus {
        let mut health = BasebandHealthStatus::new();
        let remote_host = self.remote_host.clone();

        health.effective_mode = "hardware-remote-bb-host".to_string();
        health.init.message = "Skipped in remote bb_host mode".to_string();
        health.start.message = "Skipped in remote bb_host mode".to_string();
        health.socket_init.message = "Skipped in remote bb_host mode".to_string();
        health.device_open.attempted = true;
        health.status_read.attempted = true;

        match self.get_status_summary() {
            Ok(status) => {
                health.host.connected = self.host_handle != 0 && self.device_handle != 0;

                if let Some(config) = remote_host.as_ref() {
                    health.host.daemon_version = if self.host_handle != 0 {
                        ffi::get_daemon_version(self.host_ptr()).ok().and_then(non_empty_trimmed)
                    } else {
                        None
                    };
                    health.host.message = if let Some(version) = health.host.daemon_version.as_ref() {
                        format!(
                            "Connected to bb_host {}:{} (daemon_version={})",
                            config.address,
                            config.port,
                            version
                        )
                    } else {
                        format!(
                            "Connected to bb_host {}:{} (daemon version unavailable)",
                            config.address,
                            config.port
                        )
                    };
                }

                if let Err(err) = self.refresh_host_devices_cache() {
                    tracing::warn!("Failed to refresh remote host devices while building health snapshot: {}", err);
                }

                health.runtime.detected_device_count = self.current_remote_device_count();
                health.runtime.status_snapshot = Some(status);
                health.device_open.success = self.device_handle != 0;
                health.device_open.message = if let Some(device_count) = health.runtime.detected_device_count {
                    format!("Remote baseband device ready, device_count={}", device_count)
                } else {
                    "Remote baseband device ready".to_string()
                };
                health.status_read.success = true;
                health.status_read.message = "bb_ioctl(BB_GET_STATUS) succeeded".to_string();
            }
            Err(err) => {
                health.host.connected = false;
                health.device_open.success = false;
                health.device_open.message = err.clone();
                health.status_read.success = false;
                health.status_read.message = err.clone();

                if let Some(config) = remote_host.as_ref() {
                    health.host.message = format!(
                        "Failed to connect bb_host {}:{}: {}",
                        config.address,
                        config.port,
                        err
                    );
                } else {
                    health.host.message = err;
                }
            }
        }

        health.sdk = ffi::runtime_diagnostics();
        health
    }

    fn snapshot_local_health(&mut self) -> BasebandHealthStatus {
        let mut health = BasebandHealthStatus::new();

        health.effective_mode = if self.initialized {
            "hardware-local-sdk".to_string()
        } else {
            "simulator".to_string()
        };
        health.device_open.message = "Local SDK mode does not open a remote device handle".to_string();
        health.init.attempted = true;
        health.init.success = self.initialized;
        health.init.message = if self.initialized {
            "bb_init succeeded".to_string()
        } else {
            "bb_init not completed".to_string()
        };
        health.start.attempted = true;
        health.start.success = self.started;
        health.start.message = if self.started {
            "bb_start succeeded".to_string()
        } else {
            "bb_start not completed".to_string()
        };
        health.status_read.attempted = true;

        match self.get_status_summary() {
            Ok(status) => {
                health.status_read.success = true;
                health.status_read.message = "bb_ioctl(BB_GET_STATUS) succeeded".to_string();
                health.runtime.status_snapshot = Some(status);
            }
            Err(err) => {
                health.status_read.success = false;
                health.status_read.message = err;
            }
        }

        health.sdk = ffi::runtime_diagnostics();
        health
    }

    pub fn get_health_status(&mut self) -> BasebandHealthStatus {
        if self.is_remote_mode() {
            self.snapshot_remote_health()
        } else {
            self.snapshot_local_health()
        }
    }

    fn refresh_host_devices_cache(&mut self) -> Result<(), String> {
        if self.host_handle == 0 {
            self.host_devices_cache.borrow_mut().clear();
            return Ok(());
        }

        let devices = run_remote_sdk_call(true, || ffi::list_host_devices(self.host_ptr()))?;
        *self.host_devices_cache.borrow_mut() = devices;
        Ok(())
    }

    fn remember_remote_handle(&mut self, normalized_mac: String, handle: usize) {
        if self.host_handle == 0 || handle == 0 || normalized_mac.is_empty() {
            return;
        }

        if let Some((_, cached_handle)) = self
            .remote_device_handles
            .iter_mut()
            .find(|(mac, _)| mac == &normalized_mac)
        {
            *cached_handle = handle;
            return;
        }

        self.remote_device_handles.push((normalized_mac, handle));
    }

    fn cached_remote_handle(&self, normalized_mac: &str) -> Option<usize> {
        self.remote_device_handles
            .iter()
            .find(|(mac, handle)| mac == normalized_mac && *handle != 0)
            .map(|(_, handle)| *handle)
    }

    fn get_or_open_remote_handle_by_mac(&mut self, target_mac: &str) -> Result<usize, String> {
        if self.host_handle == 0 {
            return Err("Remote device handle requires an active bb_host session".to_string());
        }

        let normalized_target = Self::normalize_mac(target_mac);
        if normalized_target.is_empty() {
            return Err("device_mac is required".to_string());
        }

        if self.active_device_mac.as_deref() == Some(normalized_target.as_str()) && self.device_handle != 0 {
            return Ok(self.device_handle);
        }

        if let Some(handle) = self.cached_remote_handle(&normalized_target) {
            return Ok(handle);
        }

        let (handle, _) = run_remote_sdk_call(true, || {
            ffi::open_host_device_by_mac(self.host_ptr(), &normalized_target)
        })?;
        let handle = handle as usize;
        self.remember_remote_handle(normalized_target, handle);
        Ok(handle)
    }

    fn remote_system_info_for_mac(&mut self, target_mac: &str) -> Result<ffi::BbSystemInfoSummary, String> {
        let normalized_target = Self::normalize_mac(target_mac);
        let handle = self.get_or_open_remote_handle_by_mac(target_mac)? as *mut ffi::bb_dev_handle_t;
        let system_info = run_remote_sdk_call(true, || ffi::get_system_info(handle))?;
        self.cache_system_info_for_mac(&normalized_target, &system_info);
        Ok(system_info)
    }

    fn remote_status_for_mac(&mut self, target_mac: &str) -> Result<ffi::BbGetStatusSummary, String> {
        let normalized_target = Self::normalize_mac(target_mac);
        if normalized_target.is_empty() {
            return Err("device_mac is required".to_string());
        }

        let handle = self.get_or_open_remote_handle_by_mac(target_mac)? as *mut ffi::bb_dev_handle_t;
        let preferred_signal_user = self.preferred_signal_user_for_mac(&normalized_target);
        let status = run_remote_sdk_call(true, || {
            ffi::get_status(handle, ffi::BB_ALL_DATA_USER_BMP, preferred_signal_user)
        })?;

        self.cache_role_for_mac(&normalized_target, status.role);
        self.cache_sync_role_for_mac(&normalized_target, status.sync_mode, status.sync_master);

        Ok(status)
    }

    fn resolve_pair_version_peer(
        &self,
        status: &ffi::BbGetStatusSummary,
        dev_pair_target_mac: Option<&str>,
        available_devices: &[ffi::BbDiscoveredDeviceSummary],
    ) -> Option<(String, Option<u8>)> {
        let active_mac = Self::normalize_mac(&status.mac_hex);
        let expected_peer_role = match status.role {
            ffi::BB_ROLE_AP => Some(ffi::BB_ROLE_DEV),
            ffi::BB_ROLE_DEV => Some(ffi::BB_ROLE_AP),
            _ => None,
        };
        let find_device_by_mac = |mac: &str| {
            available_devices.iter().find(|device| Self::normalize_mac(&device.mac_address) == mac)
        };

        if status.role == ffi::BB_ROLE_DEV {
            let normalized_target = dev_pair_target_mac.map(Self::normalize_mac).filter(|value| !value.is_empty());
            if let Some(target_mac) = normalized_target {
                if let Some(device) = find_device_by_mac(&target_mac) {
                    return Some((device.mac_address.clone(), device.role));
                }

                return Some((target_mac, expected_peer_role));
            }
        }

        let opposite_role_devices = available_devices
            .iter()
            .filter(|device| {
                let normalized_mac = Self::normalize_mac(&device.mac_address);
                normalized_mac != active_mac && expected_peer_role == device.role
            })
            .collect::<Vec<_>>();

        if status.role == ffi::BB_ROLE_DEV && opposite_role_devices.len() == 1 {
            let device = opposite_role_devices[0];
            return Some((device.mac_address.clone(), device.role));
        }

        let connected_peer_mac = status
            .links
            .first()
            .and_then(|link| link.peer_mac_hex.clone())
            .or_else(|| status.peer_mac_hex.clone())
            .map(|value| Self::normalize_mac(&value))
            .filter(|value| !value.is_empty());

        if let Some(peer_mac) = connected_peer_mac {
            if let Some(device) = find_device_by_mac(&peer_mac) {
                return Some((device.mac_address.clone(), device.role));
            }

            if peer_mac.len() == 12 {
                return Some((peer_mac, expected_peer_role));
            }
        }

        if opposite_role_devices.len() == 1 {
            let device = opposite_role_devices[0];
            return Some((device.mac_address.clone(), device.role));
        }

        let other_devices = available_devices
            .iter()
            .filter(|device| Self::normalize_mac(&device.mac_address) != active_mac)
            .collect::<Vec<_>>();

        if other_devices.len() == 1 {
            let device = other_devices[0];
            return Some((device.mac_address.clone(), device.role));
        }

        None
    }

    fn pair_release_mismatch_warning(
        &mut self,
        status: &ffi::BbGetStatusSummary,
        dev_pair_target_mac: Option<&str>,
        available_devices: &[ffi::BbDiscoveredDeviceSummary],
        current_info: &ffi::BbSystemInfoSummary,
    ) -> Result<Option<String>, String> {
        if !self.is_remote_mode() || self.host_handle == 0 || available_devices.len() < 2 {
            return Ok(None);
        }

        let active_mac = Self::normalize_mac(&status.mac_hex);
        let dev_target_peer = if status.role == ffi::BB_ROLE_DEV {
            let peer_devices = available_devices
                .iter()
                .filter(|device| Self::normalize_mac(&device.mac_address) != active_mac)
                .collect::<Vec<_>>();
            let known_peer_macs = peer_devices
                .iter()
                .map(|device| Self::normalize_mac(&device.mac_address))
                .filter(|mac| !mac.is_empty())
                .collect::<Vec<_>>();
            let is_known_peer_mac = |value: &str| {
                let normalized = Self::normalize_mac(value);
                !normalized.is_empty()
                    && (known_peer_macs.is_empty() || known_peer_macs.iter().any(|mac| mac == &normalized))
            };
            let connected_peer_mac = status
                .links
                .iter()
                .find(|link| link.state != 0)
                .and_then(|link| link.peer_mac_hex.clone())
                .or_else(|| {
                    status
                        .link_state
                        .filter(|state| *state != 0)
                        .and_then(|_| status.peer_mac_hex.clone())
                });

            dev_pair_target_mac
                .filter(|value| is_known_peer_mac(value))
                .map(Self::normalize_mac)
                .filter(|value| !value.is_empty())
                .map(|target_mac| {
                    peer_devices
                        .iter()
                        .find(|device| Self::normalize_mac(&device.mac_address) == target_mac)
                        .map(|device| (device.mac_address.clone(), device.role))
                        .unwrap_or((target_mac, Some(ffi::BB_ROLE_AP)))
                })
                .or_else(|| {
                    connected_peer_mac
                        .filter(|value| is_known_peer_mac(value))
                        .map(|peer_mac| {
                            let normalized_peer = Self::normalize_mac(&peer_mac);
                            peer_devices
                                .iter()
                                .find(|device| Self::normalize_mac(&device.mac_address) == normalized_peer)
                                .map(|device| (device.mac_address.clone(), device.role))
                                .unwrap_or((normalized_peer, Some(ffi::BB_ROLE_AP)))
                        })
                })
                .or_else(|| {
                    if peer_devices.len() == 1 {
                        let device = peer_devices[0];
                        Some((device.mac_address.clone(), device.role.or(Some(ffi::BB_ROLE_AP))))
                    } else {
                        None
                    }
                })
        } else {
            None
        };

        let (peer_mac, peer_role) = match dev_target_peer
            .or_else(|| self.resolve_pair_version_peer(status, dev_pair_target_mac, available_devices))
        {
            Some(peer) => peer,
            None => return Ok(None),
        };
        if Self::normalize_mac(&peer_mac) == Self::normalize_mac(&status.mac_hex) {
            return Ok(None);
        }

        let normalized_peer_mac = Self::normalize_mac(&peer_mac);
        let peer_info = if let Some(cached) = self.cached_system_info_for_mac(&normalized_peer_mac) {
            cached
        } else {
            self.remote_system_info_for_mac(&peer_mac).map_err(|error| {
                format!(
                    "Unable to verify whether AP and DEV releases match because peer version info could not be read from {} {}: {}. Switch to the peer once or refresh the bb_host session, then retry Pair.",
                    role_name_for_warning(peer_role),
                    peer_mac,
                    error,
                )
            })?
        };
        if !release_values_mismatch(current_info, &peer_info) {
            return Ok(None);
        }

        let current_role = Some(status.role);

        let (ap_label, ap_mac, ap_info, dev_label, dev_mac, dev_info) = if current_role == Some(ffi::BB_ROLE_AP) {
            (
                "AP",
                status.mac_hex.as_str(),
                current_info,
                "DEV",
                peer_mac.as_str(),
                &peer_info,
            )
        } else if current_role == Some(ffi::BB_ROLE_DEV) {
            (
                "AP",
                peer_mac.as_str(),
                &peer_info,
                "DEV",
                status.mac_hex.as_str(),
                current_info,
            )
        } else if peer_role == Some(ffi::BB_ROLE_AP) {
            (
                "AP",
                peer_mac.as_str(),
                &peer_info,
                role_name_for_warning(current_role),
                status.mac_hex.as_str(),
                current_info,
            )
        } else {
            (
                role_name_for_warning(current_role),
                status.mac_hex.as_str(),
                current_info,
                role_name_for_warning(peer_role),
                peer_mac.as_str(),
                &peer_info,
            )
        };

        Ok(Some(format!(
            "Firmware mismatch detected between {} {} ({}) and {} {} ({}). Pairing requires AP and DEV to run the same release. Upgrade or downgrade one side, then retry Pair.",
            ap_label,
            ap_mac,
            release_summary(ap_info),
            dev_label,
            dev_mac,
            release_summary(dev_info),
        )))
    }

    fn close_remote_handles(&mut self) {
        let mut handles = self
            .remote_device_handles
            .iter()
            .map(|(_, handle)| *handle)
            .collect::<Vec<_>>();

        if self.device_handle != 0 {
            handles.push(self.device_handle);
        }

        handles.sort_unstable();
        handles.dedup();

        for handle in handles.into_iter().filter(|handle| *handle != 0) {
            let _ = ffi::close_device(handle as *mut ffi::bb_dev_handle_t);
        }

        self.remote_device_handles.clear();
        self.device_handle = 0;
    }

    /// 获取或初始化全局基带 API 实例
    pub fn get_with_health() -> (Self, BasebandHealthStatus) {
        let remote_host = remote_host_config_from_env();
        let mut health = BasebandHealthStatus::new();
        let mut api = BasebandApi {
            initialized: false,
            started: false,
            requires_start: true,
            device_handle: 0,
            host_handle: 0,
            remote_host: remote_host.clone(),
            host_devices_cache: RefCell::new(Vec::new()),
            plot_subscription_active: false,
            plot_user: None,
            active_device_mac: None,
            preferred_signal_users: HashMap::new(),
            device_role_cache: HashMap::new(),
            device_sync_role_cache: HashMap::new(),
            bandwidth_mode_cache: HashMap::new(),
            status_summary_cache: HashMap::new(),
            channel_info_cache: HashMap::new(),
            band_selection_bitmap_cache: HashMap::new(),
            ap_pair_target_cache: HashMap::new(),
            boot_diagnostics_cache: HashMap::new(),
            system_info_cache: HashMap::new(),
            last_remote_device_switch_at: None,
            remote_device_handles: Vec::new(),
            upgrade_progress: None,
            upgrade_board_write_started_at: None,
        };

        if let Some(remote_host) = remote_host {
            let host_addr = remote_host.address.clone();
            let host_port = remote_host.port;
            tracing::info!("Using remote bb_host mode: {}:{}", host_addr, host_port);
            api.requires_start = false;
            health.init.message = "Skipped in remote bb_host mode".to_string();
            health.start.message = "Skipped in remote bb_host mode".to_string();

            match api.establish_remote_session() {
                Ok(outcome) => {
                    health.host.connected = true;
                    health.host.daemon_version = outcome.daemon_version.clone();
                    health.host.message = if let Some(version) = health.host.daemon_version.as_ref() {
                        format!("Connected to bb_host {}:{} (daemon_version={})", host_addr, host_port, version)
                    } else {
                        format!("Connected to bb_host {}:{} (daemon version unavailable)", host_addr, host_port)
                    };

                    health.runtime.detected_device_count = Some(outcome.device_count);
                    health.device_open.attempted = true;
                    health.device_open.success = true;
                    health.device_open.message = format!(
                        "Opened baseband device from remote host, device_count={}",
                        outcome.device_count
                    );
                    health.status_read.attempted = true;
                    health.status_read.success = true;
                    health.status_read.message = "bb_ioctl(BB_GET_STATUS) succeeded".to_string();
                    health.effective_mode = "hardware-remote-bb-host".to_string();

                    health.runtime.status_snapshot = Some(outcome.status);
                }
                Err(e) => {
                    health.host.connected = false;
                    health.effective_mode = "hardware-remote-bb-host".to_string();
                    health.host.message = format!("Failed to connect bb_host {}:{}: {}", host_addr, host_port, e);
                    health.device_open.attempted = true;
                    health.device_open.success = false;
                    health.device_open.message = e.clone();
                    health.status_read.attempted = true;
                    health.status_read.success = false;
                    health.status_read.message = e.clone();
                    tracing::error!("Failed to connect bb_host {}:{}: {}", host_addr, host_port, e);
                }
            }

            health.sdk = ffi::runtime_diagnostics();
            return (api, health);
        } else {
            tracing::info!("BB_HOST_ADDR not set; using local SDK mode");
            health.device_open.message = "Local SDK mode does not open a remote device handle".to_string();
        }

        let init_handle = api.handle_ptr();
        health.init.attempted = true;

        match ffi::init(init_handle) {
            Ok(_) => {
                tracing::info!("Baseband SDK initialized successfully");
                api.initialized = true;
                health.init.success = true;
                health.init.message = "bb_init succeeded".to_string();
                health.status_read.attempted = true;

                match ffi::get_status(api.handle_ptr(), ffi::BB_ALL_DATA_USER_BMP, None) {
                    Ok(snapshot) => {
                        health.status_read.success = true;
                        health.status_read.message = "bb_ioctl(BB_GET_STATUS) succeeded".to_string();
                        health.runtime.status_snapshot = Some(snapshot);
                    }
                    Err(err) => {
                        health.status_read.success = false;
                        health.status_read.message = err;
                    }
                }

                health.sdk = ffi::runtime_diagnostics();
                (api, health)
            }
            Err(e) => {
                if api.device_handle != 0 {
                    let _ = ffi::close_device(api.handle_ptr());
                    api.device_handle = 0;
                }

                if api.host_handle != 0 {
                    let _ = ffi::disconnect_host(api.host_ptr());
                    api.host_handle = 0;
                }

                health.init.success = false;
                health.init.message = e.clone();
                health.sdk = ffi::runtime_diagnostics();
                tracing::error!("Failed to initialize baseband SDK: {}", e);
                (api, health)
            }
        }
    }
    fn handle_ptr(&self) -> *mut ffi::bb_dev_handle_t {
        self.device_handle as *mut ffi::bb_dev_handle_t
    }

    fn host_ptr(&self) -> *mut ffi::bb_host_t {
        self.host_handle as *mut ffi::bb_host_t
    }

    /// 检查是否成功初始化
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn requires_start(&self) -> bool {
        self.requires_start
    }

    pub fn start(&mut self) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        ffi::start(self.handle_ptr())?;
        self.started = true;
        Ok(())
    }

    /// 创建数据传输 socket
    pub fn create_socket(&mut self, socket_id: u32, flags: u32, max_size: u32) -> Result<(), String> {
        self.with_device_operation("create_socket", |handle| {
            ffi::create_socket(handle, socket_id, flags, max_size)
        })
    }

    pub fn get_status_summary(&mut self) -> Result<ffi::BbGetStatusSummary, String> {
        let is_remote = self.is_remote_mode();
        let preferred_signal_user = self.preferred_signal_user_for_active_device();

        let result = self.with_device_operation("get_status_summary", |handle| {
            run_remote_sdk_call(is_remote, || {
                ffi::get_status(handle, ffi::BB_ALL_DATA_USER_BMP, preferred_signal_user)
            })
        });

        match result {
            Ok(status) => {
                self.remember_current_device(&status);
                self.cache_status_summary_for_active_device(&status);
                Ok(status)
            }
            Err(err) => {
                if is_remote {
                    if let Some(cached) = self.cached_status_summary_for_active_device() {
                        return Ok(cached);
                    }
                }
                Err(err)
            }
        }
    }

    pub fn get_status_summary_for_snapshot(&mut self) -> Result<ffi::BbGetStatusSummary, String> {
        let is_remote = self.is_remote_mode();
        let preferred_signal_user = self.preferred_signal_user_for_active_device();

        let result = self.with_device_operation("get_status_summary_for_snapshot", |handle| {
            run_remote_sdk_call(is_remote, || {
                ffi::get_status(handle, ffi::BB_ALL_DATA_USER_BMP, preferred_signal_user)
            })
        });

        match result {
            Ok(status) => {
                self.remember_current_device(&status);
                self.cache_status_summary_for_active_device(&status);
                Ok(status)
            }
            Err(err) => {
                if is_remote {
                    if let Some(cached) = self
                        .cached_recent_status_summary_for_active_device(REMOTE_STATUS_SNAPSHOT_REFRESH_GAP)
                    {
                        return Ok(cached);
                    }
                }
                Err(err)
            }
        }
    }

    fn load_wireless_runtime_details(&mut self) -> Result<WirelessRuntimeDetails, String> {
        let is_remote = self.is_remote_mode();
        let status = self.get_status_summary()?;
        let mut warnings = Vec::new();

        if let Err(err) = self.persist_paired_peer_targets_to_minidb(&status) {
            warnings.push(err);
        }

        let dev_pair_target_mac = if status.role == 1 {
            match run_remote_sdk_call(is_remote, || ffi::get_ap_mac(self.handle_ptr())) {
                Ok(value) => value,
                Err(err) => {
                    warnings.push(err);
                    None
                }
            }
        } else {
            None
        };
        let ap_pair_target_macs = if status.role == ffi::BB_ROLE_AP {
            match self.read_ap_pair_targets_for_active_device() {
                Ok(value) => value,
                Err(err) => {
                    warnings.push(err);
                    Self::empty_ap_pair_target_macs()
                }
            }
        } else {
            Vec::new()
        };
        let slot = status.links.first().map(|link| link.slot as u8).unwrap_or(0);
        let user = resolve_plot_user(&status);
        let available_devices = if self.host_handle != 0 {
            if let Err(err) = self.refresh_host_devices_cache() {
                warnings.push(format!(
                    "Failed to refresh remote device list; using cached devices: {}",
                    err
                ));
            }

            self.host_devices_cache
                .borrow()
                .iter()
                .cloned()
                .map(|mut device| {
                    let normalized_mac = Self::normalize_mac(&device.mac_address);
                    if let Some(role) = self.cached_role_for_mac(&normalized_mac) {
                        device.role = Some(role);
                        device.role_label = match role {
                            ffi::BB_ROLE_AP => "AP".to_string(),
                            ffi::BB_ROLE_DEV => "DEV".to_string(),
                            _ => device.role_label,
                        };
                    }
                    if let Some((sync_mode, sync_master)) = self.cached_sync_role_for_mac(&normalized_mac) {
                        device.sync_mode = Some(sync_mode);
                        device.sync_master = Some(sync_master);
                    }
                    device
                })
                .collect()
        } else {
            Vec::new()
        };

        let system_info = match self.read_system_info_for_active_device() {
            Ok(value) => Some(value),
            Err(err) => {
                warnings.push(err);
                None
            }
        };
        let band_selection_bitmap = match self.read_band_selection_bitmap_for_active_device() {
            Ok(value) => value,
            Err(err) => {
                warnings.push(err);
                None
            }
        };
        let band_info = match run_remote_sdk_call(is_remote, || ffi::get_live_band_info(self.handle_ptr())) {
            Ok(mut value) => {
                value.selection_bitmap = band_selection_bitmap;
                Some(value)
            }
            Err(err) => {
                warnings.push(err);
                None
            }
        };
        let channel_info = match self.read_channel_info_for_active_device(status.role) {
            Ok(value) => Some(value),
            Err(err) => {
                warnings.push(err);
                None
            }
        };
        let bandwidth_mode = self.cached_bandwidth_mode_for_current_device(slot);
        let mcs_mode = match run_remote_sdk_call(is_remote, || ffi::get_mcs_mode(self.handle_ptr(), slot)) {
            Ok(value) => Some(value),
            Err(err) => {
                warnings.push(err);
                None
            }
        };
        let preferred_mcs_dir = if status.role == ffi::BB_ROLE_AP {
            ffi::BB_DIR_RX
        } else {
            ffi::BB_DIR_TX
        };
        let mcs_value = match run_remote_sdk_call(is_remote, || ffi::get_mcs(self.handle_ptr(), preferred_mcs_dir, slot)) {
            Ok(value) => Some(value),
            Err(primary_err) => {
                if preferred_mcs_dir != ffi::BB_DIR_TX {
                    match run_remote_sdk_call(is_remote, || ffi::get_mcs(self.handle_ptr(), ffi::BB_DIR_TX, slot)) {
                        Ok(value) => {
                            warnings.push(format!(
                                "Preferred MCS direction read failed; fell back to TX direction: {}",
                                primary_err
                            ));
                            Some(value)
                        }
                        Err(fallback_err) => {
                            warnings.push(primary_err);
                            warnings.push(fallback_err);
                            None
                        }
                    }
                } else {
                    warnings.push(primary_err);
                    None
                }
            }
        };
        let power_mode = match run_remote_sdk_call(is_remote, || ffi::get_power_mode(self.handle_ptr())) {
            Ok(value) => Some(value),
            Err(err) => {
                warnings.push(err);
                None
            }
        };
        let current_power = match run_remote_sdk_call(is_remote, || ffi::get_current_power(self.handle_ptr(), user)) {
            Ok(value) => Some(value),
            Err(err) => {
                warnings.push(err);
                None
            }
        };
        let power_auto = match run_remote_sdk_call(is_remote, || ffi::get_power_auto(self.handle_ptr())) {
            Ok(value) => Some(value),
            Err(err) => {
                warnings.push(err);
                None
            }
        };
        let (power_fallback, mut power_warnings) =
            run_remote_sdk_call(is_remote, || {
                Ok(ffi::read_power_fallback(self.handle_ptr(), status.role))
            }).unwrap_or((None, vec!["Power fallback read failed".to_string()]));
        warnings.append(&mut power_warnings);

        if let Some(current_info) = system_info.as_ref() {
            self.cache_system_info_for_active_device(current_info);

            match self.pair_release_mismatch_warning(
                &status,
                dev_pair_target_mac.as_deref(),
                &available_devices,
                current_info,
            ) {
                Ok(Some(version_warning)) => warnings.push(version_warning),
                Ok(None) => {}
                Err(err) => warnings.push(err),
            }
        }

        Ok(WirelessRuntimeDetails {
            status,
            dev_pair_target_mac,
            ap_pair_target_macs,
            available_devices,
            system_info,
            band_info,
            channel_info,
            bandwidth_mode,
            mcs_mode,
            mcs_value,
            power_mode,
            current_power,
            power_auto,
            power_fallback,
            warnings,
        })
    }

    pub fn get_wireless_runtime_details(&mut self) -> Result<WirelessRuntimeDetails, String> {
        if self.is_remote_mode() {
            self.execute_remote_operation("get_wireless_runtime_details", |api| {
                api.load_wireless_runtime_details()
            })
        } else {
            if !self.initialized {
                return Err("Baseband API not initialized".to_string());
            }

            self.load_wireless_runtime_details()
        }
    }

    pub fn get_detected_remote_devices(&mut self) -> Result<Vec<ffi::BbDiscoveredDeviceSummary>, String> {
        if !self.is_remote_mode() {
            return Ok(Vec::new());
        }

        self.ensure_remote_host_connection()?;
        self.refresh_host_devices_cache()?;

        // Populate role/sync caches for any device we haven't probed yet.
        // ensure_remote_host_connection only connects to the daemon — it does
        // not open any device, so remember_current_device is never called.
        // Without this, the Active Device dropdown shows bare MAC addresses
        // after a device restart (the reboot window calls this function, not
        // get_wireless_runtime_details → ensure_remote_session).
        self.refresh_all_device_status_caches();

        let mut devices = self.host_devices_cache.borrow().clone();
        for device in &mut devices {
            let normalized_mac = Self::normalize_mac(&device.mac_address);
            if let Some(role) = self.cached_role_for_mac(&normalized_mac) {
                device.role = Some(role);
                device.role_label = match role {
                    ffi::BB_ROLE_AP => "AP".to_string(),
                    ffi::BB_ROLE_DEV => "DEV".to_string(),
                    _ => device.role_label.clone(),
                };
            }

            if let Some((sync_mode, sync_master)) = self.cached_sync_role_for_mac(&normalized_mac) {
                device.sync_mode = Some(sync_mode);
                device.sync_master = Some(sync_master);
            }
        }

        Ok(devices)
    }

    /// Refresh role/sync cache for all remote devices by briefly opening each one.
    /// Called during startup after the initial session is established, and after
    /// every session reconnection (e.g. after device reboot) to keep the
    /// Active Device dropdown labels accurate.
    pub fn refresh_all_device_status_caches(&mut self) {
        tracing::info!("refresh_all_device_status_caches: starting (remote={})", self.is_remote_mode());
        if !self.is_remote_mode() {
            tracing::info!("refresh_all_device_status_caches: skipped (not remote)");
            return;
        }
        // 首先填充 host 设备列表缓存
        if let Err(e) = self.refresh_host_devices_cache() {
            tracing::warn!("Failed to refresh host devices cache for status probe: {}", e);
            return;
        }
        let devices = self.host_devices_cache.borrow().clone();
        if devices.is_empty() {
            tracing::info!("No remote devices found to probe");
            return;
        }

        tracing::info!("Probing {} remote device(s) for role/sync info", devices.len());

        // Save current active device handle so we can restore it after scanning
        let saved_active_handle = self.device_handle;
        let saved_active_mac = self.active_device_mac.clone();

        for device in &devices {
            let mac = &device.mac_address;
            let normalized_mac = Self::normalize_mac(mac);

            // Never open a second handle to the currently active device —
            // doing so can interfere with its existing SDK connection and,
            // for AP devices, may trigger an unexpected reset.
            if saved_active_mac.as_deref() == Some(normalized_mac.as_str()) {
                continue;
            }

            // Skip if already cached
            if self.cached_role_for_mac(&normalized_mac).is_some()
                && self.cached_sync_role_for_mac(&normalized_mac).is_some()
            {
                continue;
            }

            tracing::info!("Probing device {} for role/sync info", mac);

            // Open the device temporarily (with a timeout-like fast fail)
            let open_result = match run_remote_sdk_call(true, || {
                ffi::open_host_device_by_mac(self.host_ptr(), mac)
            }) {
                Ok(result) => result,
                Err(e) => {
                    tracing::info!(
                        "Cannot open device {} for probe (will retry on Active Device switch): {}",
                        mac, e
                    );
                    continue;
                }
            };

            let (dev_handle, _) = open_result;
            if dev_handle.is_null() {
                tracing::info!("Probe got null handle for device {}. Skipping.", mac);
                continue;
            }

            // Get status
            match ffi::get_status(dev_handle, ffi::BB_ALL_DATA_USER_BMP, None) {
                Ok(status) => {
                    let role = status.role;
                    if matches!(role, ffi::BB_ROLE_AP | ffi::BB_ROLE_DEV) {
                        self.cache_role_for_mac(&normalized_mac, role);
                    }
                    self.cache_sync_role_for_mac(
                        &normalized_mac,
                        status.sync_mode,
                        status.sync_master,
                    );
                    tracing::info!(
                        "Probed device {}: role={}, sync_mode={}, sync_master={}",
                        mac, role, status.sync_mode, status.sync_master
                    );
                }
                Err(e) => {
                    tracing::info!("Failed to get status for device {}: {}", mac, e);
                }
            }

            // Close the temporary handle silently (best effort)
            let _ = ffi::close_device(dev_handle);
        }

        // Restore active device connection if we switched away
        self.device_handle = saved_active_handle;
        self.active_device_mac = saved_active_mac;
        tracing::info!("Device status probe completed");
    }

    fn load_wireless_configuration_details(&mut self, mode: u8) -> Result<WirelessConfigurationDetails, String> {
        let is_remote = self.is_remote_mode();
        let mut warnings = Vec::new();

        let config_text = match run_remote_sdk_call(is_remote, || ffi::get_configuration_text(self.handle_ptr(), mode)) {
            Ok(value) => value,
            Err(err) => {
                warnings.push(format!("Failed to read configuration text: {}", err));
                String::new()
            }
        };

        let role = match run_remote_sdk_call(is_remote, || ffi::get_minidb_role(self.handle_ptr())) {
            Ok(value) => value,
            Err(err) => {
                warnings.push(format!("Failed to read MiniDB role: {}", err));
                None
            }
        };

        let band_bitmap = match run_remote_sdk_call(is_remote, || ffi::get_minidb_band_bitmap_optional(self.handle_ptr())) {
            Ok(value) => value,
            Err(err) => {
                warnings.push(format!("Failed to read MiniDB band: {}", err));
                None
            }
        };

        let local_mac = match run_remote_sdk_call(is_remote, || ffi::get_minidb_local_mac(self.handle_ptr())) {
            Ok(value) => value,
            Err(err) => {
                warnings.push(format!("Failed to read MiniDB local MAC: {}", err));
                None
            }
        };

        let ap_mac = match run_remote_sdk_call(is_remote, || ffi::get_minidb_ap_mac(self.handle_ptr())) {
            Ok(value) => value,
            Err(err) => {
                warnings.push(format!("Failed to read MiniDB AP MAC: {}", err));
                None
            }
        };

        let power = match run_remote_sdk_call(is_remote, || ffi::get_minidb_power(self.handle_ptr())) {
            Ok(value) => value,
            Err(err) => {
                warnings.push(format!("Failed to read MiniDB power: {}", err));
                None
            }
        };

        let slot_macs = (0..ffi::BB_SLOT_MAX)
            .map(|slot| match run_remote_sdk_call(is_remote, || ffi::get_minidb_slot_mac(self.handle_ptr(), slot as u8)) {
                Ok(value) => value,
                Err(err) => {
                    warnings.push(format!("Failed to read MiniDB slot {} MAC: {}", slot, err));
                    None
                }
            })
            .collect();

        Ok(WirelessConfigurationDetails {
            config_mode: mode,
            config_text,
            minidb: WirelessConfigurationMinidbDetails {
                role,
                band_bitmap,
                local_mac,
                ap_mac,
                slot_macs,
                power,
            },
            warnings,
        })
    }

    pub fn get_wireless_configuration_details(&mut self, mode: u8) -> Result<WirelessConfigurationDetails, String> {
        if self.is_remote_mode() {
            self.execute_remote_operation("get_wireless_configuration_details", |api| {
                api.load_wireless_configuration_details(mode)
            })
        } else {
            if !self.initialized {
                return Err("Baseband API not initialized".to_string());
            }

            self.load_wireless_configuration_details(mode)
        }
    }

    pub fn get_boot_diagnostics(&mut self) -> Result<BootDiagnostics, String> {
        if self.is_remote_mode() {
            if let Some(cached) = self.cached_boot_diagnostics_for_active_device() {
                return Ok(cached);
            }
        }

        let is_remote = self.is_remote_mode();

        let running_system = match run_remote_sdk_call(is_remote, || ffi::get_running_system(self.handle_ptr())) {
            Ok(value) => value,
            Err(err) => format!("Error: {}", err),
        };
        let boot_reason = match run_remote_sdk_call(is_remote, || ffi::get_boot_reason(self.handle_ptr())) {
            Ok(value) => value,
            Err(err) => format!("Error: {}", err),
        };

        let diagnostics = BootDiagnostics {
            running_system,
            boot_reason,
        };
        self.cache_boot_diagnostics_for_active_device(&diagnostics);
        Ok(diagnostics)
    }

    fn switch_remote_device_once(&mut self, target_mac: &str, normalized_target: &str) -> Result<(), String> {
        if self.initialized && self.active_device_mac.as_deref() == Some(normalized_target) {
            return Ok(());
        }

        if let Some(cached_handle) = self.cached_remote_handle(normalized_target) {
            if self.plot_subscription_active {
                if let Some(user) = self.plot_user {
                    let _ = ffi::unsubscribe_plot_stream(self.handle_ptr(), user);
                }
                self.plot_subscription_active = false;
                self.plot_user = None;
            }

            self.device_handle = cached_handle;
            self.initialized = true;
            self.active_device_mac = Some(normalized_target.to_string());

            let status = run_remote_sdk_call(true, || {
                ffi::get_status(
                    self.handle_ptr(),
                    ffi::BB_ALL_DATA_USER_BMP,
                    self.preferred_signal_user_for_mac(normalized_target),
                )
            })?;

            self.remember_current_device(&status);
            self.cache_status_summary_for_active_device(&status);

            return Ok(());
        }

        if let Some(last_switch_at) = self.last_remote_device_switch_at {
            if let Some(remaining) = REMOTE_DEVICE_SWITCH_GAP.checked_sub(last_switch_at.elapsed()) {
                thread::sleep(remaining);
            }
        }
        self.last_remote_device_switch_at = Some(Instant::now());

        let current_handle = self.handle_ptr();
        let (new_handle, _) = run_remote_sdk_call(true, || {
            ffi::open_host_device_by_mac(self.host_ptr(), target_mac)
        })?;
        let preferred_signal_user = self.preferred_signal_user_for_mac(normalized_target);
        let new_status = match run_remote_sdk_call(true, || {
            ffi::get_status(new_handle, ffi::BB_ALL_DATA_USER_BMP, preferred_signal_user)
        }) {
            Ok(status) => status,
            Err(err) => {
                let _ = ffi::close_device(new_handle);
                return Err(err);
            }
        };

        if self.plot_subscription_active {
            if let Some(user) = self.plot_user {
                let _ = ffi::unsubscribe_plot_stream(current_handle, user);
            }
            self.plot_subscription_active = false;
            self.plot_user = None;
        }

        self.device_handle = new_handle as usize;
        self.initialized = true;
        self.active_device_mac = Some(normalized_target.to_string());
        self.remember_current_device(&new_status);
        self.cache_status_summary_for_active_device(&new_status);
        self.remember_remote_handle(Self::normalize_mac(&new_status.mac_hex), self.device_handle);

        Ok(())
    }

    pub fn switch_active_device(&mut self, target_mac: &str) -> Result<(), String> {
        if !self.is_remote_mode() {
            return Err("Device switching requires remote bb_host mode".to_string());
        }

        let normalized_target = Self::normalize_mac(target_mac);
        if normalized_target.is_empty() {
            return Err("device_mac is required".to_string());
        }

        self.execute_remote_operation("switch_active_device", |api| {
            api.switch_remote_device_once(target_mac, &normalized_target)
        })
    }

    pub fn set_pair_mode(&mut self, start: bool, slot_bmp: u8) -> Result<(), String> {
        self.set_pair_mode_with_blacklist(start, slot_bmp, &[])
    }

    pub fn set_pair_mode_with_blacklist(
        &mut self,
        start: bool,
        slot_bmp: u8,
        black_list: &[String],
    ) -> Result<(), String> {
        self.with_device_operation("set_pair_mode", |handle| {
            ffi::set_pair_mode(handle, start, slot_bmp, black_list)
        })
    }

    pub fn get_pair_candidates(&mut self, slot: u8) -> Result<Vec<String>, String> {
        self.with_device_operation("get_pair_candidates", |handle| ffi::get_pair_candidates(handle, slot))
    }

    pub fn set_pair_candidates(&mut self, slot: u8, macs: &[String]) -> Result<(), String> {
        self.with_device_operation("set_pair_candidates", |handle| {
            ffi::set_pair_candidates(handle, slot, macs)
        })
    }

    pub fn set_ap_mac(&mut self, mac: &str) -> Result<(), String> {
        self.with_device_operation("set_ap_mac", |handle| ffi::set_ap_mac(handle, mac))
    }

    pub fn set_minidb_ap_mac(&mut self, mac: &str) -> Result<(), String> {
        self.with_device_operation("set_minidb_ap_mac", |handle| {
            ffi::set_minidb_ap_mac(handle, mac)
        })
    }

    pub fn set_minidb_local_mac(&mut self, mac: &str) -> Result<(), String> {
        self.with_device_operation("set_minidb_local_mac", |handle| {
            ffi::set_minidb_local_mac(handle, mac)
        })
    }

    pub fn set_minidb_role(&mut self, role: u8) -> Result<(), String> {
        self.with_device_operation("set_minidb_role", |handle| ffi::set_minidb_role(handle, role))
    }

    pub fn set_minidb_power(&mut self, power: ffi::bb_phy_pwr_basic_t) -> Result<(), String> {
        self.with_device_operation("set_minidb_power", |handle| ffi::set_minidb_power(handle, power))
    }

    pub fn save_configuration_text(&mut self, text: &str) -> Result<(), String> {
        self.with_device_operation("save_configuration_text", |handle| {
            ffi::set_configuration_text(handle, text)
        })
    }

    pub fn clear_flash_configuration(&mut self) -> Result<(), String> {
        self.with_device_operation("clear_flash_configuration", |handle| ffi::reset_config(handle))
    }

    pub fn clear_minidb_configuration(&mut self) -> Result<(), String> {
        self.with_device_operation("clear_minidb_configuration", |handle| ffi::reset_minidb(handle))?;
        self.clear_cached_channel_info_for_active_device();
        self.clear_cached_band_selection_bitmap_for_active_device();
        self.clear_cached_ap_pair_targets_for_active_device();
        Ok(())
    }

    pub fn restore_factory_configuration(&mut self) -> Result<(), String> {
        self.with_device_operation("restore_factory_configuration", |handle| {
            ffi::reset_config(handle)?;
            ffi::reset_minidb(handle)
        })?;
        self.clear_cached_channel_info_for_active_device();
        self.clear_cached_band_selection_bitmap_for_active_device();
        self.clear_cached_ap_pair_targets_for_active_device();
        Ok(())
    }

    pub fn set_minidb_slot_mac(&mut self, slot: u8, mac: &str) -> Result<(), String> {
        self.with_device_operation("set_minidb_slot_mac", |handle| {
            ffi::set_minidb_slot_mac(handle, slot, mac)
        })?;
        self.update_cached_ap_pair_target_for_active_device(slot, Some(mac));
        Ok(())
    }

    pub fn set_channel_mode(&mut self, auto_mode: bool) -> Result<(), String> {
        self.with_device_operation("set_channel_mode", |handle| ffi::set_channel_mode(handle, auto_mode))?;
        self.clear_cached_channel_info_for_active_device();
        Ok(())
    }

    pub fn set_channel(&mut self, dir: u8, chan_index: u8) -> Result<(), String> {
        self.with_device_operation("set_channel", |handle| ffi::set_channel(handle, dir, chan_index))?;
        self.clear_cached_channel_info_for_active_device();
        Ok(())
    }

    /// Get channel info for sweep. When `known_role` is provided (from the
    /// sweep feeder which already knows the role via get_status_snapshot),
    /// the expensive and potentially session-destroying get_status_summary()
    /// call is skipped. This is the preferred path for the sweep feeder loop.
    pub fn get_sweep_channel_info_for_role(&mut self, known_role: Option<u8>) -> Result<ffi::BbChannelInfoSummary, String> {
        let role = match known_role {
            Some(r) => r,
            None => self.get_status_summary()?.role,
        };
        let result = self.read_channel_info_for_active_device(role);
        if result.is_ok() {
            // cache updated inside read_channel_info_for_active_device on success
        }
        result
    }

    pub fn get_sweep_channel_info(&mut self) -> Result<ffi::BbChannelInfoSummary, String> {
        self.get_sweep_channel_info_for_role(None)
    }

    pub fn configure_sweep(&mut self, mode: u8, bandwidth: u8, frequencies_khz: &[u32]) -> Result<(), String> {
        // DEV does not support BB_SET_FSP_CTRL — calling it corrupts SDK
        // state and causes subsequent BB_GET_CHAN_INFO to fail.  DEV sweep
        // instead uses BB_CFG_CHANNEL (trigger_slave_channel_scan) to
        // refresh the power table.
        if self.is_active_device_dev() {
            // Trigger an immediate channel scan for all DEV devices, including
            // DEV(S).  Even with an empty frequency list, ACS (init_chan = -1)
            // lets the device auto-select channels — this populates the channel
            // table that serial "chan" and BB_GET_CHAN_INFO read from.
            return self.trigger_slave_channel_scan(bandwidth, frequencies_khz);
        }
        self.with_device_operation("configure_sweep", |handle| {
            ffi::configure_sweep(handle, mode, bandwidth, frequencies_khz)
        })?;
        self.clear_cached_channel_info_for_active_device();
        Ok(())
    }

    pub fn start_sweep(&mut self) -> Result<(), String> {
        // DEV does not support BB_SET_FSP_CTRL — sweep feeder uses
        // BB_GET_CHAN_INFO + BB_CFG_CHANNEL instead.
        if self.is_active_device_dev() {
            tracing::debug!("Skipping BB_SET_FSP_CTRL start for DEV (will use BB_CFG_CHANNEL for sweep)");
            return Ok(());
        }
        self.clear_cached_channel_info_for_active_device();
        self.with_device_operation("start_sweep", |handle| ffi::start_sweep(handle))?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn stop_sweep(&mut self) -> Result<(), String> {
        self.with_device_operation("stop_sweep", |handle| ffi::stop_sweep(handle))?;
        Ok(())
    }

    pub fn trigger_fsp_scan(&mut self) -> Result<(), String> {
        self.with_device_operation("trigger_fsp_scan", |handle| ffi::trigger_fsp_scan(handle))
    }

    /// Trigger a real channel scan using BB_CFG_CHANNEL.
    /// This is needed for AP-slave and DEV devices because they lack FSP
    /// event subscriptions, so BB_GET_CHAN_INFO returns stale/zero power
    /// values unless we explicitly trigger a scan via the channel
    /// configuration command.
    pub fn trigger_slave_channel_scan(
        &mut self,
        bandwidth: u8,
        frequencies_khz: &[u32],
    ) -> Result<(), String> {
        self.with_device_operation("trigger_slave_channel_scan", |handle| {
            ffi::trigger_slave_channel_scan(handle, bandwidth, frequencies_khz)
        })
    }

    pub fn set_mcs_mode(&mut self, slot: u8, auto_mode: bool) -> Result<(), String> {
        self.with_device_operation("set_mcs_mode", |handle| ffi::set_mcs_mode(handle, slot, auto_mode))
    }

    pub fn set_mcs(&mut self, slot: u8, mcs: u8) -> Result<(), String> {
        self.with_device_operation("set_mcs", |handle| ffi::set_mcs(handle, slot, mcs))
    }

    /// 通过 BB_SET_TX_MCS 直接设置物理用户的 TX MCS（不经过 slot，可在 DEV 端使用）
    pub fn set_tx_mcs(&mut self, user: u8, mcs: u8) -> Result<(), String> {
        self.with_device_operation("set_tx_mcs", |handle| ffi::set_tx_mcs(handle, user, mcs))
    }

    pub fn set_power_mode(&mut self, pwr_mode: u8) -> Result<(), String> {
        self.with_device_operation("set_power_mode", |handle| ffi::set_power_mode(handle, pwr_mode))
    }

    pub fn set_power(&mut self, user: u8, power_dbm: u8) -> Result<(), String> {
        self.with_device_operation("set_power", |handle| ffi::set_power(handle, user, power_dbm))
    }

    pub fn set_power_auto(&mut self, enabled: bool) -> Result<(), String> {
        self.with_device_operation("set_power_auto", |handle| ffi::set_power_auto(handle, enabled))
    }

    pub fn set_band_mode(&mut self, auto_mode: bool) -> Result<(), String> {
        self.with_device_operation("set_band_mode", |handle| {
            ffi::set_band_mode(handle, auto_mode).and_then(|_| {
                if auto_mode {
                    ffi::set_minidb_band_auto(handle)
                } else {
                    Ok(())
                }
            })
        })?;
        self.clear_cached_channel_info_for_active_device();
        if auto_mode {
            self.cache_band_selection_bitmap_for_active_device(Some(0x07));
        }
        Ok(())
    }

    pub fn set_band(&mut self, target_band: u8) -> Result<(), String> {
        self.with_device_operation("set_band", |handle| {
            ffi::set_band(handle, target_band)
                .and_then(|_| ffi::set_minidb_band(handle, target_band))
        })?;
        self.clear_cached_channel_info_for_active_device();
        self.cache_band_selection_bitmap_for_active_device(Self::band_bitmap_for_target_band(target_band));
        Ok(())
    }

    pub fn set_band_selection(&mut self, band_bitmap: u8) -> Result<(), String> {
        self.with_device_operation("set_band_selection", |handle| {
            ffi::set_band_selection_bitmap(handle, band_bitmap)
        })?;
        self.clear_cached_channel_info_for_active_device();
        self.cache_band_selection_bitmap_for_active_device(Some(band_bitmap));
        Ok(())
    }

    pub fn set_bandwidth(&mut self, slot: u8, dir: u8, bandwidth: u8) -> Result<(), String> {
        self.with_device_operation("set_bandwidth", |handle| {
            ffi::set_bandwidth(handle, slot, dir, bandwidth)
        })?;
        self.cache_bandwidth_mode_for_active_device(slot, false);
        Ok(())
    }

    pub fn set_bandwidth_mode(&mut self, slot: u8, auto_mode: bool) -> Result<(), String> {
        self.with_device_operation("set_bandwidth_mode", |handle| {
            ffi::set_bandwidth_mode(handle, slot, auto_mode)
        })?;
        self.cache_bandwidth_mode_for_active_device(slot, auto_mode);
        Ok(())
    }

    pub fn set_baseband_role(&mut self, role: u8) -> Result<(), String> {
        self.with_device_operation("set_baseband_role", |handle| {
            ffi::set_baseband_role(handle, role)?;
            ffi::reboot_device(handle, 2000)
        })?;
        self.cache_role_for_active_device(role);
        if self.is_remote_mode() {
            self.invalidate_remote_session_state();
        }
        Ok(())
    }

    pub fn reboot_device(&mut self, delay_ms: u32) -> Result<(), String> {
        self.with_device_operation("reboot_device", |handle| ffi::reboot_device(handle, delay_ms))?;
        if self.is_remote_mode() {
            self.invalidate_remote_session_state();
        }
        Ok(())
    }

    /// 将一段分区/段数据写入指定 flash 地址，分块传输后校验 CRC32（对标 PC Tool ar8030_upgrade_partition）
    fn upgrade_partition(&mut self, flash_addr: u64, len: u32, data: &[u8]) -> Result<(), String> {
        if data.is_empty() {
            return Err("Partition data is empty".to_string());
        }

        let is_remote = self.is_remote_mode();
        let partition_crc = crc32fast::hash(data);
        let chunk_count = data.chunks(ffi::BB_HOT_UPGRADE_CHUNK_SIZE).len();

        tracing::info!(
            "Upgrade partition: flash_addr=0x{:X} len={} chunks={} crc32={:08X}",
            flash_addr,
            len,
            chunk_count,
            partition_crc
        );

        // 进度上报辅助
        let progress_arc = self.upgrade_progress.clone();
        let mut last_progress_update_at = None;

        let mut offset = 0u64;
        for (i, chunk) in data.chunks(ffi::BB_HOT_UPGRADE_CHUNK_SIZE).enumerate() {
            let chunk_addr = flash_addr
                .checked_add(offset)
                .ok_or_else(|| "Partition address overflow".to_string())?;
            // BB_SET_HOT_UPGRADE_WRITE 的 addr 字段为 u32，截断高位（flash 地址在 32 位范围内）
            let ioctl_addr = chunk_addr as u32;

            let handle = self.handle_ptr();
            run_remote_sdk_call_with_gap(is_remote, REMOTE_HOT_UPGRADE_SDK_CALL_GAP, || {
                ffi::hot_upgrade_write(handle, HOT_UPGRADE_WRITE_SEQ, ioctl_addr, chunk)
            })
            .map_err(|e| format!("Partition chunk {}/{} write at 0x{:X} failed: {}", i + 1, chunk_count, chunk_addr, e))?;

            offset = offset
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| "Partition offset overflow".to_string())?;

            // 更新进度百分比
            let is_last_chunk = i + 1 == chunk_count;
            let should_update_progress = is_last_chunk
                || last_progress_update_at.map_or(true, |last_update_at: Instant| {
                    last_update_at.elapsed() >= UPGRADE_PROGRESS_UPDATE_INTERVAL
                });

            if should_update_progress {
                last_progress_update_at = Some(Instant::now());
            }

            if should_update_progress {
                if let Some(ref prog) = progress_arc {
                if let Ok(mut guard) = prog.lock() {
                    let chunk_done = i + 1;
                    guard.bytes_written = guard.bytes_written.saturating_add(chunk.len());
                    guard.board_write_elapsed_ms = self.current_upgrade_board_write_elapsed_ms();
                    if guard.file_size > 0 {
                        guard.percent = (guard.bytes_written as f64 / guard.file_size as f64) * 100.0;
                    }
                    guard.current_step = chunk_done;
                    guard.total_steps = chunk_count;
                    if guard.step_label.is_empty() {
                        guard.step_label = "Writing firmware data".to_string();
                    }
                    guard.message = format!(
                        "Board write acknowledged {} / {} bytes",
                        guard.bytes_written,
                        guard.file_size
                    );
                    guard.state = "flashing".to_string();
                }
            }
            }
        }

        // CRC32 校验使用分区的 flash 起始地址和分区数据长度（对标 PC Tool ar8030_upgrade_chk_crc）
        let ioctl_addr = flash_addr as u32;
        let handle = self.handle_ptr();
        run_remote_sdk_call_with_gap(is_remote, REMOTE_HOT_UPGRADE_SDK_CALL_GAP, || {
            ffi::hot_upgrade_crc32(handle, HOT_UPGRADE_CRC_SEQ, len, ioctl_addr, partition_crc)
        })
        .map_err(|e| format!("Partition CRC32 verify at 0x{:X} failed: {}", flash_addr, e))?;

        tracing::info!(
            "Partition upgrade OK: flash_addr=0x{:X} chunks={} crc32={:08X}",
            flash_addr,
            chunk_count,
            partition_crc
        );

        Ok(())
    }

    fn upgrade_raw_firmware(&mut self, firmware: &[u8]) -> Result<FirmwareUpgradeResult, String> {
        let is_remote = self.is_remote_mode();
        let chunk_count = firmware.chunks(ffi::BB_HOT_UPGRADE_CHUNK_SIZE).len();
        tracing::info!(
            "[UPGRADE-RAW] Writing {} bytes in {} chunks ({} bytes each)",
            firmware.len(),
            chunk_count,
            ffi::BB_HOT_UPGRADE_CHUNK_SIZE
        );

        let mut addr = 0_u32;
        let progress_arc = self.upgrade_progress.clone();
        let mut last_progress_update_at = None;

        for (i, chunk) in firmware.chunks(ffi::BB_HOT_UPGRADE_CHUNK_SIZE).enumerate() {
            let handle = self.handle_ptr();
            run_remote_sdk_call_with_gap(is_remote, REMOTE_HOT_UPGRADE_SDK_CALL_GAP, || {
                ffi::hot_upgrade_write(handle, HOT_UPGRADE_WRITE_SEQ, addr, chunk)
            })
            .map_err(|e| format!("Chunk {}/{} write failed: {}", i + 1, chunk_count, e))?;
            addr = addr
                .checked_add(chunk.len() as u32)
                .ok_or_else(|| "Firmware image address overflow".to_string())?;

            // 更新进度
            let is_last_chunk = i + 1 == chunk_count;
            let should_update_progress = is_last_chunk
                || last_progress_update_at.map_or(true, |last_update_at: Instant| {
                    last_update_at.elapsed() >= UPGRADE_PROGRESS_UPDATE_INTERVAL
                });

            if should_update_progress {
                last_progress_update_at = Some(Instant::now());
                if let Some(ref prog) = progress_arc {
                    if let Ok(mut guard) = prog.lock() {
                        guard.current_step = i + 1;
                        guard.total_steps = chunk_count;
                        guard.step_label = "Writing firmware image".to_string();
                        guard.bytes_written = addr as usize;
                        guard.board_write_elapsed_ms = self.current_upgrade_board_write_elapsed_ms();
                        if guard.file_size > 0 {
                            guard.percent = (guard.bytes_written as f64 / guard.file_size as f64) * 100.0;
                        }
                        guard.message = format!(
                            "Board write acknowledged {} / {} bytes",
                            guard.bytes_written,
                            guard.file_size
                        );
                        guard.state = "flashing".to_string();
                    }
                }
            }
        }

        let crc32 = crc32fast::hash(firmware);
        let handle = self.handle_ptr();
        tracing::info!(
            "[UPGRADE-RAW] All chunks written, verifying CRC32={:08X} ({} bytes total)",
            crc32,
            firmware.len()
        );
        run_remote_sdk_call_with_gap(is_remote, REMOTE_HOT_UPGRADE_SDK_CALL_GAP, || {
            ffi::hot_upgrade_crc32(handle, HOT_UPGRADE_CRC_SEQ, firmware.len() as u32, 0, crc32)
        })
        .map_err(|e| format!("CRC verification failed: {}", e))?;
        tracing::info!("[UPGRADE-RAW] CRC verified successfully");

        Ok(FirmwareUpgradeResult { crc32 })
    }

    fn upgrade_partition_by_name_pc<F>(
        &mut self,
        partitions_raw: &[u8],
        segments_raw: &[u8],
        image: &[u8],
        partitions_count: usize,
        segments_count: usize,
        part_name: &str,
        total_bytes_written: &mut usize,
        partition_cnt: &mut usize,
        current_step: &mut usize,
        update_progress: &F,
    ) -> Result<bool, String>
    where
        F: Fn(usize, &str, usize),
    {
        let mut segment_idx: isize = -1;
        let mut selected_part_off: Option<usize> = None;

        for part_idx in 0..partitions_count {
            let part_off = part_idx * ffi::PART_INFO_SIZE;
            if part_off + ffi::PART_INFO_SIZE > partitions_raw.len() {
                break;
            }

            let part_bytes = &partitions_raw[part_off..part_off + ffi::PART_INFO_SIZE];
            let is_upgrade = u32::from_le_bytes([
                part_bytes[ffi::PART_INFO_IS_UPGRADE_OFFSET],
                part_bytes[ffi::PART_INFO_IS_UPGRADE_OFFSET + 1],
                part_bytes[ffi::PART_INFO_IS_UPGRADE_OFFSET + 2],
                part_bytes[ffi::PART_INFO_IS_UPGRADE_OFFSET + 3],
            ]) != 0;
            if is_upgrade {
                segment_idx += 1;
            }

            let current_name = String::from_utf8_lossy(&part_bytes[..ffi::PART_INFO_NAME_SIZE])
                .trim_end_matches('\0')
                .to_string();
            if current_name == part_name {
                selected_part_off = Some(part_off);
                break;
            }
        }

        let Some(part_off) = selected_part_off else {
            tracing::info!("OTA partition '{}' not found in image, skipping", part_name);
            return Ok(false);
        };

        let part_bytes = &partitions_raw[part_off..part_off + ffi::PART_INFO_SIZE];
        let is_upgrade = u32::from_le_bytes([
            part_bytes[ffi::PART_INFO_IS_UPGRADE_OFFSET],
            part_bytes[ffi::PART_INFO_IS_UPGRADE_OFFSET + 1],
            part_bytes[ffi::PART_INFO_IS_UPGRADE_OFFSET + 2],
            part_bytes[ffi::PART_INFO_IS_UPGRADE_OFFSET + 3],
        ]) != 0;
        if !is_upgrade {
            tracing::info!("OTA partition '{}' marked as non-upgrade, skipping", part_name);
            return Ok(false);
        }

        if segment_idx < 0 || segment_idx as usize >= segments_count {
            return Err(format!("OTA partition '{}' has invalid segment index {}", part_name, segment_idx));
        }

        let seg_off = segment_idx as usize * ffi::SEGMENT_INFO_SIZE;
        if seg_off + ffi::SEGMENT_INFO_SIZE > segments_raw.len() {
            return Err(format!("OTA partition '{}' segment {} out of bounds", part_name, segment_idx));
        }
        let seg_bytes = &segments_raw[seg_off..seg_off + ffi::SEGMENT_INFO_SIZE];

        let part_flash_off = u64::from_le_bytes([
            part_bytes[ffi::PART_INFO_FLASH_OFFSET_OFFSET],
            part_bytes[ffi::PART_INFO_FLASH_OFFSET_OFFSET + 1],
            part_bytes[ffi::PART_INFO_FLASH_OFFSET_OFFSET + 2],
            part_bytes[ffi::PART_INFO_FLASH_OFFSET_OFFSET + 3],
            part_bytes[ffi::PART_INFO_FLASH_OFFSET_OFFSET + 4],
            part_bytes[ffi::PART_INFO_FLASH_OFFSET_OFFSET + 5],
            part_bytes[ffi::PART_INFO_FLASH_OFFSET_OFFSET + 6],
            part_bytes[ffi::PART_INFO_FLASH_OFFSET_OFFSET + 7],
        ]);
        let part_length = u64::from_le_bytes([
            part_bytes[ffi::PART_INFO_LENGTH_OFFSET],
            part_bytes[ffi::PART_INFO_LENGTH_OFFSET + 1],
            part_bytes[ffi::PART_INFO_LENGTH_OFFSET + 2],
            part_bytes[ffi::PART_INFO_LENGTH_OFFSET + 3],
            part_bytes[ffi::PART_INFO_LENGTH_OFFSET + 4],
            part_bytes[ffi::PART_INFO_LENGTH_OFFSET + 5],
            part_bytes[ffi::PART_INFO_LENGTH_OFFSET + 6],
            part_bytes[ffi::PART_INFO_LENGTH_OFFSET + 7],
        ]);
        let seg_flash_off = u64::from_le_bytes([
            seg_bytes[8], seg_bytes[9], seg_bytes[10], seg_bytes[11],
            seg_bytes[12], seg_bytes[13], seg_bytes[14], seg_bytes[15],
        ]);
        if seg_flash_off != part_flash_off {
            return Err(format!(
                "OTA partition '{}' flash offset mismatch: partition=0x{:X}, segment=0x{:X}, segment_idx={}",
                part_name,
                part_flash_off,
                seg_flash_off,
                segment_idx
            ));
        }

        let img_off = u64::from_le_bytes([
            seg_bytes[0], seg_bytes[1], seg_bytes[2], seg_bytes[3],
            seg_bytes[4], seg_bytes[5], seg_bytes[6], seg_bytes[7],
        ]);
        let size_dec = u64::from_le_bytes([
            seg_bytes[24], seg_bytes[25], seg_bytes[26], seg_bytes[27],
            seg_bytes[28], seg_bytes[29], seg_bytes[30], seg_bytes[31],
        ]);

        if seg_flash_off + size_dec > part_flash_off + part_length {
            return Err(format!(
                "OTA partition '{}' segment exceeds partition length: segment=0x{:X}+0x{:X}, partition=0x{:X}+0x{:X}",
                part_name,
                seg_flash_off,
                size_dec,
                part_flash_off,
                part_length
            ));
        }

        if size_dec == 0 {
            tracing::info!("OTA partition '{}' has zero-sized segment, skipping", part_name);
            return Ok(false);
        }
        if size_dec > u32::MAX as u64 {
            return Err(format!("OTA segment for '{}' too large: {}", part_name, size_dec));
        }

        let data_start = img_off as usize;
        let data_end = data_start + size_dec as usize;
        if data_end > image.len() {
            return Err(format!("OTA segment '{}' data out of bounds: offset=0x{:X} size=0x{:X}", part_name, img_off, size_dec));
        }

        let seg_data = &image[data_start..data_end];
        let flash_addr = seg_flash_off
            .checked_add(ffi::GPT_FLASH_OFFSET)
            .ok_or_else(|| format!("Flash address overflow for partition '{}'", part_name))?;

        tracing::info!(
            "OTA partition '{}': flash_addr=0x{:X} size={} file_offset=0x{:X} segment_idx={}",
            part_name, flash_addr, size_dec, img_off, segment_idx
        );

        let (progress_label, complete_label) = match part_name {
            "app0" | "app1" => ("Writing application image", "Application image written"),
            _ => ("Writing firmware data", "Firmware segment written"),
        };

        update_progress(*current_step, progress_label, *total_bytes_written);
        self.upgrade_partition(flash_addr, size_dec as u32, seg_data)?;
        *total_bytes_written += size_dec as usize;
        *partition_cnt += 1;
        *current_step += 1;
        update_progress(*current_step, complete_label, *total_bytes_written);
        Ok(true)
    }

    /// 解析 .img 格式 OTA 升级镜像并逐分区写入 flash（精确对标 PC Tool AR8030_Upgrade_Thread_OTA::ar8030_upgrade）
    fn upgrade_ota_image(&mut self, image: &[u8]) -> Result<FirmwareUpgradeResult, String> {
        let is_remote = self.is_remote_mode();
        if is_remote {
            self.ensure_remote_session()?;
        } else if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        // 校验最小长度：header(256) + hash(32) + sig(256) = 544
        if image.len() < ffi::UPGRADE_HDR_SIZE + ffi::OTA_HASH_SIZE + ffi::OTA_SIG_SIZE {
            return Err(format!(
                "Firmware image too small for OTA format: {} bytes (need at least {})",
                image.len(),
                ffi::UPGRADE_HDR_SIZE + ffi::OTA_HASH_SIZE + ffi::OTA_SIG_SIZE
            ));
        }

        // 读取 upgrade_hdr 头部（256 字节）。
        // 结构体布局严格对齐 PC Tool common.h struct upgrade_hdr (packed, LE):
        //   0x00: magic            u32   (4 bytes, 0x4152544F)
        //   0x04: hdr_version      u8    (1 byte)
        //   0x05: compressed       u8    (1 byte)
        //   0x06: flashtype        u8    (1 byte)
        //   0x07: part_status      u8    (1 byte)
        //   0x08: header_ext_size  u16   (2 bytes)
        //   0x0A: hash_size        u16   (2 bytes, must be 32)
        //   0x0C: sig_size         u16   (2 bytes, must be 256)
        //   0x0E: sig_realsize     u16   (2 bytes)
        //   0x10: img_size         u64   (8 bytes)
        //   0x18: rom_size         u32   (4 bytes, ≤ 0x8000)
        //   0x1C: loader_size      u32   (4 bytes, ≤ 0x100000)
        //   0x20: partitions       u16   (2 bytes)
        //   0x22: segments         u16   (2 bytes)
        //   0x24: object_version   u32   (4 bytes)
        //   0x28: depend_version   u32   (4 bytes)
        //   0x2C: reserve[20]            (20 bytes)
        //   0x40: part_flag[64]          (64 bytes)
        //   0x80: sdk_version[128]       (128 bytes)
        //   Total: 256 bytes
        let hdr = &image[..ffi::UPGRADE_HDR_SIZE];

        // Diagnostic: print first 16 u32 values as hex dump
        let mut hdr_dump = String::new();
        for i in 0..(ffi::UPGRADE_HDR_SIZE / 4).min(16) {
            let val = u32::from_le_bytes([hdr[i * 4], hdr[i * 4 + 1], hdr[i * 4 + 2], hdr[i * 4 + 3]]);
            use std::fmt::Write;
            if i > 0 && i % 8 == 0 {
                let _ = write!(hdr_dump, "\n  ");
            }
            let _ = write!(hdr_dump, " {:02X}:{:08X}", i * 4, val);
        }
        tracing::info!("OTA header scan (u32 LE, offset:value):\n  {}", hdr_dump);

        // Parse fields at the CORRECT offsets matching PC Tool common.h
        let magic = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
        let _hdr_version = hdr[4];
        let _compressed  = hdr[5];
        let _flashtype   = hdr[6];
        let _part_status = hdr[7];
        let header_ext_size = u16::from_le_bytes([hdr[8], hdr[9]]) as usize;
        let hash_size       = u16::from_le_bytes([hdr[10], hdr[11]]) as usize;
        let sig_size        = u16::from_le_bytes([hdr[12], hdr[13]]) as usize;
        let _sig_realsize   = u16::from_le_bytes([hdr[14], hdr[15]]) as usize;
        let img_size    = u64::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19], hdr[20], hdr[21], hdr[22], hdr[23]]);
        let rom_size    = u32::from_le_bytes([hdr[24], hdr[25], hdr[26], hdr[27]]) as u64;
        let loader_size = u32::from_le_bytes([hdr[28], hdr[29], hdr[30], hdr[31]]) as u64;
        let partitions_count = u16::from_le_bytes([hdr[32], hdr[33]]) as usize;
        let segments_count   = u16::from_le_bytes([hdr[34], hdr[35]]) as usize;

        tracing::info!(
            "OTA header: magic=0x{:08X} hdr_ver={} compressed={} flashtype={} part_status={} hdr_ext_sz={} hash_sz={} sig_sz={} sig_real={} img_size={} rom={} loader={} partitions={} segments={}",
            magic,
            _hdr_version,
            _compressed,
            _flashtype,
            _part_status,
            header_ext_size,
            hash_size,
            sig_size,
            _sig_realsize,
            img_size,
            rom_size,
            loader_size,
            partitions_count,
            segments_count
        );

        if magic != ffi::OTA_IMG_MAGIC {
            return Err(format!("Not a valid OTA image: magic 0x{:08X}", magic));
        }

        // Validate hash/sig sizes (must match the hardcoded 32/256 used by PC Tool)
        if hash_size != ffi::OTA_HASH_SIZE || sig_size != ffi::OTA_SIG_SIZE {
            return Err(format!(
                "OTA image hash/sig size mismatch: hash={}, sig={} (expected hash=32, sig=256). Header scan:\n{}",
                hash_size,
                sig_size,
                hdr_dump
            ));
        }

        if rom_size > 0x8000 {
            return Err(format!("OTA image rom_size too large: {} (max 0x8000)", rom_size));
        }
        if loader_size > 0x100000 {
            return Err(format!("OTA image loader_size too large: {} (max 0x100000)", loader_size));
        }

        // Offset calculations — exactly matching PC Tool ar8030_upgrade_ota.cpp:
        //   hdr_ext   = data + sizeof(upgrade_hdr)        = data + 256
        //   romcode   = hdr_ext + 32 + 256 + header_ext   = data + 544 + header_ext
        //   bootloader= romcode + rom_size
        //   gpt       = bootloader + loader_size
        //   partitions= gpt + GPT_FLASH_SIZE
        //   segments  = partitions + partitions * sizeof(part_info)
        //   img_data  = segments + segments * sizeof(segment_info)
        let romcode_offset    = ffi::UPGRADE_HDR_SIZE + ffi::OTA_HASH_SIZE + ffi::OTA_SIG_SIZE + header_ext_size;
        let bootloader_offset = romcode_offset + rom_size as usize;
        let gpt_offset        = bootloader_offset + loader_size as usize;
        let partitions_offset = gpt_offset + ffi::GPT_FLASH_SIZE as usize;
        let segments_offset   = partitions_offset + partitions_count * ffi::PART_INFO_SIZE;
        // Segment data: img_offset is **absolute** file offset (matching PC Tool data += segments[idx].img_offset)
        let image_data_offset = segments_offset + segments_count * ffi::SEGMENT_INFO_SIZE;

        if image_data_offset > image.len() {
            return Err(format!("OTA image header overflow: data at {} but file is {}", image_data_offset, image.len()));
        }

        let mut total_bytes_written: usize = 0;
        let mut partition_cnt: usize = 0;
        let overall_crc = crc32fast::hash(image);

        // 对齐 PC Tool 实际升级顺序: ROM -> GPT1 -> app1 -> GPT0 -> app0 -> reboot
        let total_steps = 5;
        let mut current_step: usize = 0;

        // 初始化进度（如果已设置进度追踪器）
        let progress_arc = self.upgrade_progress.clone();
        let update_progress = |step: usize, label: &str, bytes: usize| {
            if let Some(ref prog) = progress_arc {
                if let Ok(mut guard) = prog.lock() {
                    guard.current_step = step;
                    guard.total_steps = total_steps;
                    guard.step_label = label.to_string();
                    guard.bytes_written = bytes;
                    if guard.file_size > 0 {
                        guard.percent = (bytes as f64 / guard.file_size as f64) * 100.0;
                    }
                    guard.message = format!(
                        "Board write acknowledged {} / {} bytes",
                        guard.bytes_written,
                        guard.file_size
                    );
                    guard.state = "flashing".to_string();
                }
            }
        };

        // ----- 1) 写入 romcode -----
        if rom_size > 0 {
            let end = romcode_offset + rom_size as usize;
            if end > image.len() { return Err("OTA image truncated at romcode".to_string()); }
            let rom_data = &image[romcode_offset..end];
            self.upgrade_partition(0x0, rom_size as u32, rom_data)?;
            total_bytes_written += rom_size as usize;
            partition_cnt += 1;
            tracing::info!("OTA romcode: 0x0 {} bytes", rom_size);
        }

        // ----- 2) 写入 GPT table 1（地址 = 0 + rom_size + GPT_FLASH_SIZE）-----
        // GPT 数据在 .img 文件中的位置已在上面 gpt_offset 计算
        let gpt_data_end = gpt_offset + ffi::GPT_FLASH_SIZE as usize;
        if gpt_data_end > image.len() { return Err("OTA image truncated at GPT".to_string()); }
        let gpt_data = &image[gpt_offset..gpt_data_end];

        let gpt_table1_addr = (0u64)
            .checked_add(rom_size)
            .and_then(|a| a.checked_add(ffi::GPT_FLASH_SIZE))
            .ok_or_else(|| "GPT table 1 address overflow".to_string())?;
        self.upgrade_partition(gpt_table1_addr, ffi::GPT_FLASH_SIZE as u32, gpt_data)?;
        total_bytes_written += ffi::GPT_FLASH_SIZE as usize;
        partition_cnt += 1;
        tracing::info!("OTA gpt table1: 0x{:X} {} bytes", gpt_table1_addr, ffi::GPT_FLASH_SIZE);

        // ----- 3) 写入 app1 分区（严格对齐 PC Tool ar8030_upgrade_partition_byname）-----
        let partitions_raw = &image[partitions_offset..segments_offset];
        let segments_raw   = &image[segments_offset..image_data_offset];
        let _ = self.upgrade_partition_by_name_pc(
            partitions_raw,
            segments_raw,
            image,
            partitions_count,
            segments_count,
            "app1",
            &mut total_bytes_written,
            &mut partition_cnt,
            &mut current_step,
            &update_progress,
        )?;

        // ----- 4) 写入 GPT table 0（地址 = 0 + rom_size）-----
        let gpt_table0_addr = rom_size;
        update_progress(current_step, "Writing GPT table 0", total_bytes_written);
        self.upgrade_partition(gpt_table0_addr, ffi::GPT_FLASH_SIZE as u32, gpt_data)?;
        total_bytes_written += ffi::GPT_FLASH_SIZE as usize;
        partition_cnt += 1;
        current_step += 1;
        update_progress(current_step, "GPT table 0 written", total_bytes_written);
        tracing::info!("OTA gpt table0: 0x{:X} {} bytes", gpt_table0_addr, ffi::GPT_FLASH_SIZE);

        let _ = self.upgrade_partition_by_name_pc(
            partitions_raw,
            segments_raw,
            image,
            partitions_count,
            segments_count,
            "app0",
            &mut total_bytes_written,
            &mut partition_cnt,
            &mut current_step,
            &update_progress,
        )?;

        tracing::info!(
            "OTA upgrade done: {} partitions, {} bytes, overall_crc={:08X}",
            partition_cnt, total_bytes_written, overall_crc
        );

        Ok(FirmwareUpgradeResult { crc32: overall_crc })
    }

    pub fn upgrade_firmware(&mut self, firmware: &[u8]) -> Result<FirmwareUpgradeResult, String> {
        if firmware.is_empty() {
            return Err("Firmware image is empty".to_string());
        }

        if firmware.len() > u32::MAX as usize {
            return Err(format!(
                "Firmware image is too large: {} bytes exceeds SDK limit {}",
                firmware.len(),
                u32::MAX
            ));
        }

        let is_remote = self.is_remote_mode();
        tracing::info!(
            "[UPGRADE] Starting: {} bytes, remote_mode={}, handle_valid={}",
            firmware.len(),
            is_remote,
            self.device_handle != 0
        );

        // OTA (.img) partition-based upgrade path is gated behind the OTA_IMG_MAGIC
        // detection below.  Because the header layout assumptions in upgrade_ota_image()
        // may not match the .img format produced by every toolchain version, OTA
        // parsing is opt-in: only firmware images that start with the OTA magic
        // (0x4152544F, "ARTO") are routed to the partition-aware path.
        // All other firmware files use the raw BB_SET_HOT_UPGRADE_WRITE flat-binary path.
        let is_ota_image = firmware.len() >= 4
            && u32::from_le_bytes([firmware[0], firmware[1], firmware[2], firmware[3]]) == ffi::OTA_IMG_MAGIC;

        if is_ota_image {
            tracing::info!("[UPGRADE] OTA image detected (magic=0x{:08X}), trying partition-based upgrade", ffi::OTA_IMG_MAGIC);
            match self.upgrade_ota_image(firmware) {
                Ok(result) => {
                    tracing::info!("[UPGRADE] OTA image written successfully, triggering reboot");

                    let handle = self.handle_ptr();
                    run_remote_sdk_call(is_remote, || {
                        ffi::reboot_device(handle, 2000)
                    }).map_err(|e| {
                        tracing::error!("[UPGRADE] Reboot after OTA upgrade failed: {}", e);
                        format!("OTA firmware written but reboot failed: {}", e)
                    })?;

                    tracing::info!("[UPGRADE] Complete, device rebooting");
                    return Ok(result);
                }
                Err(ota_err) => {
                    // OTA parsing failed — likely the header layout does not match
                    // this toolchain's .img format.  Fall back to the raw path with
                    // a warning so the upgrade can still proceed.
                    tracing::warn!(
                        "[UPGRADE] OTA partition parsing failed ({}). Falling back to raw flat-binary path. \
                         The OTA header/metadata bytes will be written to flash address 0, \
                         which may not be correct for all devices.",
                        ota_err
                    );
                }
            }
        }

        // Raw flat-binary path (also serves as fallback when OTA parsing fails)
        {
            tracing::info!("[UPGRADE] Using flat binary upgrade path");
            if is_remote {
                self.ensure_remote_session().map_err(|e| {
                    tracing::error!("[UPGRADE] Remote session failed: {}", e);
                    format!("Remote session error: {}", e)
                })?;
            } else if !self.initialized {
                return Err("Baseband API not initialized".to_string());
            }

            let result = self.upgrade_raw_firmware(firmware).map_err(|e| {
                tracing::error!("[UPGRADE] Raw firmware write failed: {}", e);
                e
            })?;
            tracing::info!("[UPGRADE] Raw firmware written and CRC verified, triggering reboot");

            // Small delay to let the flash controller finalize before reboot
            if is_remote {
                std::thread::sleep(Duration::from_millis(500));
            }

            let handle = self.handle_ptr();
            run_remote_sdk_call(is_remote, || {
                ffi::reboot_device(handle, 2000)
            }).map_err(|e| {
                tracing::error!("[UPGRADE] Reboot after raw upgrade failed: {}", e);
                format!("Firmware written but reboot failed: {}", e)
            })?;

            tracing::info!("[UPGRADE] Complete, device rebooting");
            Ok(result)
        }
    }

    pub fn set_signal_user_preference(&mut self, user: u8) -> Result<(), String> {
        if usize::from(user) >= ffi::BB_DATA_USER_MAX {
            return Err(format!(
                "Unsupported signal user '{}'; expected 0-{}",
                user,
                ffi::BB_DATA_USER_MAX.saturating_sub(1)
            ));
        }

        if self.active_device_mac.is_none() {
            let status = self.get_status_summary()?;
            self.remember_current_device(&status);
        }

        let Some(active_device_mac) = self.active_device_mac.clone() else {
            return Err("No active device is available for signal user selection".to_string());
        };

        self.preferred_signal_users.insert(active_device_mac, user);
        self.clear_cached_status_summary_for_active_device();
        Ok(())
    }

}

impl Drop for BasebandApi {
    fn drop(&mut self) {
        if self.plot_subscription_active {
            if let Some(user) = self.plot_user {
                let _ = ffi::unsubscribe_plot_stream(self.handle_ptr(), user);
            }
            self.plot_subscription_active = false;
            self.plot_user = None;
        }

        if self.started {
            let _ = ffi::stop(self.handle_ptr());
            self.started = false;
        }

        if self.requires_start {
            if self.initialized {
                let _ = ffi::deinit(self.handle_ptr());

                if self.device_handle != 0 {
                    let _ = ffi::close_device(self.handle_ptr());
                    self.device_handle = 0;
                }

                tracing::info!("Baseband SDK deinitialized");
            }
        } else if self.is_remote_mode() {
            self.clear_remote_session_state();
            tracing::info!("Baseband remote host disconnected");
        }
    }
}

/// 管理基带通信的核心模块
pub struct BasebandManager {
    api: Arc<Mutex<BasebandApi>>,
    upgrade_progress: Mutex<Option<Arc<Mutex<FirmwareUpgradeProgress>>>>,
}

impl BasebandManager {
    pub fn initialize_with_health() -> (Option<Self>, BasebandHealthStatus) {
        let (mut api, mut health) = BasebandApi::get_with_health();

        if !api.is_initialized() {
            if api.is_remote_mode() {
                health.effective_mode = "hardware-remote-bb-host".to_string();
                return (
                    Some(BasebandManager {
                        api: Arc::new(Mutex::new(api)),
                        upgrade_progress: Mutex::new(None),
                    }),
                    health,
                );
            }

            return (None, health);
        }

        if !api.requires_start() {
            health.effective_mode = "hardware-remote-bb-host".to_string();
            return (
                Some(BasebandManager {
                    api: Arc::new(Mutex::new(api)),
                    upgrade_progress: Mutex::new(None),
                }),
                health,
            );
        }

        health.start.attempted = true;

        match api.start() {
            Ok(()) => {
                health.start.success = true;
                health.start.message = "bb_start succeeded".to_string();
                health.effective_mode = "hardware-local-sdk".to_string();
                (
                    Some(BasebandManager {
                        api: Arc::new(Mutex::new(api)),
                        upgrade_progress: Mutex::new(None),
                    }),
                    health,
                )
            }
            Err(err) => {
                health.start.success = false;
                health.start.message = err;
                (None, health)
            }
        }
    }

    /// 创建新的基带管理器并初始化 SDK
    pub fn new() -> Result<Self, String> {
        let (manager, health) = Self::initialize_with_health();

        manager.ok_or_else(|| {
            if health.start.attempted {
                health.start.message
            } else {
                health.init.message
            }
        })
    }

    fn lock_api(&self, operation: &str) -> MutexGuard<'_, BasebandApi> {
        match self.api.lock() {
            Ok(api) => api,
            Err(poisoned) => {
                tracing::warn!(
                    "BasebandManager API mutex was poisoned during '{}'; clearing cached session state and recovering the manager.",
                    operation
                );
                let mut api = poisoned.into_inner();
                api.clear_remote_session_state();
                // Immediately attempt to re-establish a valid session so that
                // the caller doesn't operate on an uninitialized (handle=0) API.
                if let Err(e) = api.ensure_remote_session() {
                    tracing::error!(
                        "Failed to re-establish remote session after mutex poison recovery: {}",
                        e
                    );
                }
                api
            }
        }
    }

    fn with_api<T>(&self, operation: &str, action: impl FnOnce(&mut BasebandApi) -> T) -> T {
        let mut api = self.lock_api(operation);
        action(&mut api)
    }

    /// 初始化通信 socket
    pub fn initialize_socket(&self, socket_id: u32) -> Result<(), String> {
        self.with_api("initialize_socket", |api| {
            // TX + RX 双向通信
            let flags = ffi::BB_SOCK_FLAG_TX | ffi::BB_SOCK_FLAG_RX;
            api.create_socket(socket_id, flags, 4096)
        })
    }

    pub fn get_status_snapshot(&self) -> Result<ffi::BbGetStatusSummary, String> {
        tracing::debug!("BasebandManager::get_status_snapshot acquiring SDK status");
        let result = self.with_api("get_status_snapshot", |api| api.get_status_summary_for_snapshot());
        tracing::debug!(success = result.is_ok(), "BasebandManager::get_status_snapshot finished");
        result
    }

    pub fn get_status_snapshot_for_device(&self, target_mac: &str) -> Result<ffi::BbGetStatusSummary, String> {
        self.with_api("get_status_snapshot_for_device", |api| {
            if api.is_remote_mode() {
                return api.remote_status_for_mac(target_mac);
            }

            let normalized_target = BasebandApi::normalize_mac(target_mac);
            if normalized_target.is_empty() {
                return Err("device_mac is required".to_string());
            }

            if api.active_device_mac.as_deref() == Some(normalized_target.as_str()) {
                api.get_status_summary_for_snapshot()
            } else {
                Err("Device-specific snapshot requires remote bb_host mode".to_string())
            }
        })
    }

    pub fn get_wireless_runtime_details(&self) -> Result<WirelessRuntimeDetails, String> {
        self.with_api("get_wireless_runtime_details", |api| api.get_wireless_runtime_details())
    }

    pub fn get_detected_remote_devices(&self) -> Result<Vec<ffi::BbDiscoveredDeviceSummary>, String> {
        self.with_api("get_detected_remote_devices", |api| api.get_detected_remote_devices())
    }

    pub fn get_wireless_configuration_details(&self, mode: u8) -> Result<WirelessConfigurationDetails, String> {
        self.with_api("get_wireless_configuration_details", |api| api.get_wireless_configuration_details(mode))
    }

    pub fn get_boot_diagnostics(&self) -> Result<BootDiagnostics, String> {
        self.with_api("get_boot_diagnostics", |api| api.get_boot_diagnostics())
    }

    pub fn get_health_status(&self) -> BasebandHealthStatus {
        self.with_api("get_health_status", |api| api.get_health_status())
    }

    pub fn set_signal_user_preference(&self, user: u8) -> Result<(), String> {
        self.with_api("set_signal_user_preference", |api| api.set_signal_user_preference(user))
    }

    pub fn set_pair_mode(&self, start: bool, slot_bmp: u8) -> Result<(), String> {
        self.with_api("set_pair_mode", |api| api.set_pair_mode(start, slot_bmp))
    }

    pub fn get_pair_candidates(&self, slot: u8) -> Result<Vec<String>, String> {
        self.with_api("get_pair_candidates", |api| api.get_pair_candidates(slot))
    }

    pub fn set_pair_candidates(&self, slot: u8, macs: &[String]) -> Result<(), String> {
        self.with_api("set_pair_candidates", |api| api.set_pair_candidates(slot, macs))
    }

    pub fn set_ap_mac(&self, mac: &str) -> Result<(), String> {
        self.with_api("set_ap_mac", |api| api.set_ap_mac(mac))
    }

    pub fn set_minidb_ap_mac(&self, mac: &str) -> Result<(), String> {
        self.with_api("set_minidb_ap_mac", |api| api.set_minidb_ap_mac(mac))
    }

    pub fn set_minidb_local_mac(&self, mac: &str) -> Result<(), String> {
        self.with_api("set_minidb_local_mac", |api| api.set_minidb_local_mac(mac))
    }

    pub fn set_minidb_role(&self, role: u8) -> Result<(), String> {
        self.with_api("set_minidb_role", |api| api.set_minidb_role(role))
    }

    pub fn set_minidb_power(&self, power: ffi::bb_phy_pwr_basic_t) -> Result<(), String> {
        self.with_api("set_minidb_power", |api| api.set_minidb_power(power))
    }

    pub fn save_configuration_text(&self, text: &str) -> Result<(), String> {
        self.with_api("save_configuration_text", |api| api.save_configuration_text(text))
    }

    pub fn clear_flash_configuration(&self) -> Result<(), String> {
        self.with_api("clear_flash_configuration", |api| api.clear_flash_configuration())
    }

    pub fn clear_minidb_configuration(&self) -> Result<(), String> {
        self.with_api("clear_minidb_configuration", |api| api.clear_minidb_configuration())
    }

    pub fn restore_factory_configuration(&self) -> Result<(), String> {
        self.with_api("restore_factory_configuration", |api| api.restore_factory_configuration())
    }

    pub fn set_minidb_slot_mac(&self, slot: u8, mac: &str) -> Result<(), String> {
        self.with_api("set_minidb_slot_mac", |api| api.set_minidb_slot_mac(slot, mac))
    }

    pub fn set_channel_mode(&self, auto_mode: bool) -> Result<(), String> {
        self.with_api("set_channel_mode", |api| api.set_channel_mode(auto_mode))
    }

    pub fn set_channel(&self, dir: u8, chan_index: u8) -> Result<(), String> {
        self.with_api("set_channel", |api| api.set_channel(dir, chan_index))
    }

    pub fn get_sweep_channel_info(&self) -> Result<ffi::BbChannelInfoSummary, String> {
        self.with_api("get_sweep_channel_info", |api| api.get_sweep_channel_info())
    }

    /// Role-aware variant for the sweep feeder — passes known_role directly
    /// to avoid the expensive / session-destroying get_status_summary() call.
    pub fn get_sweep_channel_info_for_role(&self, known_role: Option<u8>) -> Result<ffi::BbChannelInfoSummary, String> {
        self.with_api("get_sweep_channel_info_for_role", |api| api.get_sweep_channel_info_for_role(known_role))
    }

    pub fn configure_sweep(&self, mode: u8, bandwidth: u8, frequencies_khz: &[u32]) -> Result<(), String> {
        self.with_api("configure_sweep", |api| api.configure_sweep(mode, bandwidth, frequencies_khz))
    }

    pub fn start_sweep(&self) -> Result<(), String> {
        self.with_api("start_sweep", |api| api.start_sweep())
    }

    #[allow(dead_code)]
    pub fn stop_sweep(&self) -> Result<(), String> {
        self.with_api("stop_sweep", |api| api.stop_sweep())
    }

    pub fn trigger_fsp_scan(&self) -> Result<(), String> {
        self.with_api("trigger_fsp_scan", |api| api.trigger_fsp_scan())
    }

    /// Trigger a real channel scan using BB_CFG_CHANNEL.
    /// See BasebandApi::trigger_slave_channel_scan for details.
    pub fn trigger_slave_channel_scan(&self, bandwidth: u8, frequencies_khz: &[u32]) -> Result<(), String> {
        self.with_api("trigger_slave_channel_scan", |api| api.trigger_slave_channel_scan(bandwidth, frequencies_khz))
    }

    /// Probe all remote devices for role/sync info and populate caches.
    /// Call this once after the initial session is established so the
    /// Active Device dropdown shows proper labels immediately.
    pub fn refresh_all_device_status_caches(&self) {
        let mut api = self.api.lock().unwrap();
        api.refresh_all_device_status_caches();
    }

    pub fn set_mcs_mode(&self, slot: u8, auto_mode: bool) -> Result<(), String> {
        self.with_api("set_mcs_mode", |api| api.set_mcs_mode(slot, auto_mode))
    }

    pub fn set_mcs(&self, slot: u8, mcs: u8) -> Result<(), String> {
        self.with_api("set_mcs", |api| api.set_mcs(slot, mcs))
    }

    pub fn set_tx_mcs(&self, user: u8, mcs: u8) -> Result<(), String> {
        self.with_api("set_tx_mcs", |api| api.set_tx_mcs(user, mcs))
    }

    pub fn set_power_mode(&self, pwr_mode: u8) -> Result<(), String> {
        self.with_api("set_power_mode", |api| api.set_power_mode(pwr_mode))
    }

    pub fn set_power(&self, user: u8, power_dbm: u8) -> Result<(), String> {
        self.with_api("set_power", |api| api.set_power(user, power_dbm))
    }

    pub fn set_power_auto(&self, enabled: bool) -> Result<(), String> {
        self.with_api("set_power_auto", |api| api.set_power_auto(enabled))
    }

    pub fn switch_active_device(&self, target_mac: &str) -> Result<(), String> {
        self.with_api("switch_active_device", |api| api.switch_active_device(target_mac))
    }

    pub fn set_band_mode(&self, auto_mode: bool) -> Result<(), String> {
        self.with_api("set_band_mode", |api| api.set_band_mode(auto_mode))
    }

    pub fn set_band(&self, target_band: u8) -> Result<(), String> {
        self.with_api("set_band", |api| api.set_band(target_band))
    }

    pub fn set_band_selection(&self, band_bitmap: u8) -> Result<(), String> {
        self.with_api("set_band_selection", |api| api.set_band_selection(band_bitmap))
    }

    pub fn set_bandwidth(&self, slot: u8, dir: u8, bandwidth: u8) -> Result<(), String> {
        self.with_api("set_bandwidth", |api| api.set_bandwidth(slot, dir, bandwidth))
    }

    pub fn set_bandwidth_mode(&self, slot: u8, auto_mode: bool) -> Result<(), String> {
        self.with_api("set_bandwidth_mode", |api| api.set_bandwidth_mode(slot, auto_mode))
    }

    pub fn set_baseband_role(&self, role: u8) -> Result<(), String> {
        self.with_api("set_baseband_role", |api| api.set_baseband_role(role))
    }

    pub fn reboot_device(&self, delay_ms: u32) -> Result<(), String> {
        self.with_api("reboot_device", |api| api.reboot_device(delay_ms))
    }

    /// 在后台线程中执行固件升级，立即返回（通过 progress 轮询进度）
    pub fn start_upgrade_background(
        self: &Arc<Self>,
        firmware: Vec<u8>,
        file_name: String,
        http_upload_elapsed_ms: u64,
    ) {
        let progress = Arc::new(Mutex::new(FirmwareUpgradeProgress {
            state: "idle".to_string(),
            file_name: file_name.clone(),
            file_size: firmware.len(),
            bytes_written: 0,
            http_upload_elapsed_ms,
            board_write_elapsed_ms: 0,
            total_steps: 0,
            current_step: 0,
            step_label: String::new(),
            percent: 0.0,
            message: "Waiting for first board write acknowledgement...".to_string(),
            crc32: None,
            reboot_expected: true,
        }));
        *self.upgrade_progress.lock().unwrap() = Some(Arc::clone(&progress));

        let mgr = Arc::clone(self);
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // 将进度追踪器注入 BasebandApi
                {
                    let mut api = mgr.lock_api("start_upgrade_background progress setup");
                    api.upgrade_progress = Some(Arc::clone(&progress));
                }

                let (result, board_write_elapsed_ms) = {
                    let mut api = mgr.lock_api("start_upgrade_background upgrade_firmware");
                    api.upgrade_board_write_started_at = Some(Instant::now());
                    let result = api.upgrade_firmware(&firmware);
                    let board_write_elapsed_ms = api.current_upgrade_board_write_elapsed_ms();
                    api.upgrade_board_write_started_at = None;
                    (result, board_write_elapsed_ms)
                };

                // 更新最终进度
                if let Ok(mut s) = progress.lock() {
                    s.board_write_elapsed_ms = board_write_elapsed_ms;
                    match &result {
                        Ok(r) => {
                            s.state = "done".to_string();
                            s.percent = 100.0;
                            s.crc32 = Some(format!("{:08X}", r.crc32));
                            s.step_label = "Upgrade complete".to_string();
                            s.message = "Firmware written and verified. Device is rebooting...".to_string();
                        }
                        Err(e) => {
                            s.state = "error".to_string();
                            s.message = format!("Upgrade failed: {}", e);
                        }
                    }
                }

                result
            }));

            let result = match result {
                Ok(r) => r,
                Err(panic_payload) => {
                    let msg = panic_payload
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| panic_payload.downcast_ref::<String>().map(|s| s.as_str()))
                        .unwrap_or("unknown panic");
                    tracing::error!("Background firmware upgrade panicked: {}", msg);
                    if let Ok(mut s) = progress.lock() {
                        s.state = "error".to_string();
                        s.message = format!("Internal server error during upgrade: {}", msg);
                    }
                    Err(format!("Background upgrade panicked: {}", msg))
                }
            };

            // 清理 BasebandApi 中的进度引用
            if let Ok(mut api) = mgr.api.lock() {
                api.upgrade_progress = None;
            }

            tracing::info!("Background firmware upgrade finished: {:?}", result);
        });
    }

    pub fn get_upgrade_progress(&self) -> Option<FirmwareUpgradeProgress> {
        self.upgrade_progress
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|progress| progress.lock().ok().map(|guard| guard.clone()))
    }

    /// 发送数据到 SOC
    pub fn send_data(&self, socket_id: u32, data: &[u8]) -> Result<usize, String> {
        if data.is_empty() {
            return Err("Cannot send empty data".to_string());
        }

        let bytes_sent = unsafe {
            ffi::bb_socket_write(
                socket_id as std::os::raw::c_int,
                data.as_ptr() as *const std::os::raw::c_void,
                data.len() as std::os::raw::c_uint,
                -1,
            )
        };

        if bytes_sent < 0 {
            Err(format!("Failed to send data: error code {}", bytes_sent))
        } else {
            Ok(bytes_sent as usize)
        }
    }

}

impl Default for BasebandManager {
    fn default() -> Self {
        BasebandManager::new().expect("Failed to create BasebandManager")
    }
}

impl Clone for BasebandManager {
    fn clone(&self) -> Self {
        BasebandManager {
            api: Arc::clone(&self.api),
            upgrade_progress: Mutex::new(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_status(role: u8, local_mac: &str, links: Vec<ffi::BbLinkStatusSummary>) -> ffi::BbGetStatusSummary {
        ffi::BbGetStatusSummary {
            role,
            mode: 0,
            sync_mode: 0,
            sync_master: 0,
            cfg_sbmp: 0,
            rt_sbmp: 0,
            active_user: None,
            detected_active_user: None,
            tx_status: None,
            rx_status: None,
            slot_tx_status: None,
            slot_rx_status: None,
            br_tx_status: None,
            br_rx_status: None,
            mac_bytes: [0; ffi::BB_MAC_LEN],
            mac_hex: local_mac.to_string(),
            frequency_khz: None,
            bandwidth: None,
            tx_mcs: None,
            rx_mcs: None,
            link_state: links.first().map(|link| link.state),
            pair_state: None,
            snr_db: None,
            br_snr_db: None,
            ldpc_err: None,
            ldpc_num: None,
            signal_main: None,
            signal_aux: None,
            br_signal_main: None,
            br_signal_aux: None,
            peer_mac_bytes: None,
            peer_mac_hex: links.first().and_then(|link| link.peer_mac_hex.clone()),
            links,
        }
    }

    #[test]
    #[ignore = "requires hardware or remote bb_host"]
    fn test_baseband_manager_initialization() {
        let _ = BasebandManager::new();
    }

    #[test]
    fn paired_peer_target_for_dev_uses_connected_peer_mac() {
        let status = sample_status(
            ffi::BB_ROLE_DEV,
            "A5:54:F6:2C",
            vec![ffi::BbLinkStatusSummary {
                slot: 0,
                state: 2,
                rx_mcs: None,
                pair_state: true,
                candidate_macs: Vec::new(),
                snr_db: None,
                ldpc_err: None,
                ldpc_num: None,
                signal_main: None,
                signal_aux: None,
                peer_mac_bytes: None,
                peer_mac_hex: Some("A5:68:B0:33".to_string()),
            }],
        );

        assert_eq!(paired_peer_target_for_dev(&status).as_deref(), Some("a568b033"));
    }

    #[test]
    fn paired_peer_targets_for_ap_only_keeps_connected_slots() {
        let status = sample_status(
            ffi::BB_ROLE_AP,
            "A5:68:B0:33",
            vec![
                ffi::BbLinkStatusSummary {
                    slot: 0,
                    state: 0,
                    rx_mcs: None,
                    pair_state: false,
                    candidate_macs: Vec::new(),
                    snr_db: None,
                    ldpc_err: None,
                    ldpc_num: None,
                    signal_main: None,
                    signal_aux: None,
                    peer_mac_bytes: None,
                    peer_mac_hex: Some("A5:54:F6:2C".to_string()),
                },
                ffi::BbLinkStatusSummary {
                    slot: 1,
                    state: 3,
                    rx_mcs: None,
                    pair_state: true,
                    candidate_macs: Vec::new(),
                    snr_db: None,
                    ldpc_err: None,
                    ldpc_num: None,
                    signal_main: None,
                    signal_aux: None,
                    peer_mac_bytes: None,
                    peer_mac_hex: Some("A5:54:F6:2C".to_string()),
                },
            ],
        );

        assert_eq!(paired_peer_targets_for_ap(&status), vec![(1, "a554f62c".to_string())]);
    }

}
