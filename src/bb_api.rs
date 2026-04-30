//! 基带 API 的 Rust 安全包装
//!
//! 提供类型安全的高级接口，用于与基带芯片 (ar8030) SOC 通信

use std::{cell::RefCell, collections::HashMap, sync::{Arc, Mutex}, thread, time::{Duration, Instant}};
use crate::ffi;

const REMOTE_SDK_CALL_GAP: Duration = Duration::from_millis(20);
const REMOTE_DEVICE_SWITCH_GAP: Duration = Duration::from_millis(1200);
const BB_USER_0: u8 = 0;
static REMOTE_SDK_LAST_CALL_AT: Mutex<Option<Instant>> = Mutex::new(None);

pub(crate) fn resolve_plot_user(status: &ffi::BbGetStatusSummary) -> u8 {
    status.active_user.or(status.detected_active_user).unwrap_or(BB_USER_0)
}

fn run_remote_sdk_call<T>(enabled: bool, operation: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    if enabled {
        let sleep_duration = {
            let last_call = REMOTE_SDK_LAST_CALL_AT.lock().unwrap();
            last_call
                .and_then(|instant| REMOTE_SDK_CALL_GAP.checked_sub(instant.elapsed()))
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
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WirelessConfigurationSlotMac {
    pub slot: u8,
    pub mac_address: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WirelessConfigurationDetails {
    pub mode: u8,
    pub config_file: Option<ffi::BbConfigTextSummary>,
    pub role: Option<u8>,
    pub band_bitmap: Option<u8>,
    pub power: Option<ffi::BbMinidbPowerSummary>,
    pub local_mac_address: Option<String>,
    pub ap_mac_address: Option<String>,
    pub slot_macs: Vec<WirelessConfigurationSlotMac>,
    pub warnings: Vec<String>,
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
    bandwidth_mode_cache: HashMap<(String, u8), bool>,
    last_remote_device_switch_at: Option<Instant>,
    remote_device_handles: Vec<(String, usize)>,
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
    }

    fn remember_current_device(&mut self, status: &ffi::BbGetStatusSummary) {
        let active_mac = Self::normalize_mac(&status.mac_hex);

        if active_mac.is_empty() {
            self.active_device_mac = None;
            return;
        }

        self.active_device_mac = Some(active_mac.clone());
        self.remember_remote_handle(active_mac, self.device_handle);
    }

    fn cache_bandwidth_mode_for_active_device(&mut self, slot: u8, auto_mode: bool) {
        if let Some(active_mac) = self.active_device_mac.clone() {
            self.bandwidth_mode_cache.insert((active_mac, slot), auto_mode);
        }
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

    fn execute_remote_operation<T>(
        &mut self,
        label: &str,
        mut operation: impl FnMut(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        self.ensure_remote_session()?;

        match operation(self) {
            Ok(value) => Ok(value),
            Err(first_err) => {
                tracing::warn!(
                    "Remote operation '{}' failed: {}. Resetting session and retrying once.",
                    label,
                    first_err
                );
                self.clear_remote_session_state();
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
        if self.is_remote_mode() {
            self.execute_remote_operation(label, |api| operation(api.handle_ptr()))
        } else {
            if !self.initialized {
                return Err("Baseband API not initialized".to_string());
            }

            operation(self.handle_ptr())
        }
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
            bandwidth_mode_cache: HashMap::new(),
            last_remote_device_switch_at: None,
            remote_device_handles: Vec::new(),
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

        self.with_device_operation("get_status_summary", |handle| {
            run_remote_sdk_call(is_remote, || {
                ffi::get_status(handle, ffi::BB_ALL_DATA_USER_BMP, preferred_signal_user)
            })
        })
        .map(|status| {
            self.remember_current_device(&status);
            status
        })
    }

    fn load_wireless_runtime_details(&mut self) -> Result<WirelessRuntimeDetails, String> {
        let is_remote = self.is_remote_mode();
        let preferred_signal_user = self.preferred_signal_user_for_active_device();
        let status = run_remote_sdk_call(is_remote, || {
            ffi::get_status(self.handle_ptr(), ffi::BB_ALL_DATA_USER_BMP, preferred_signal_user)
        })?;
        self.remember_current_device(&status);
        let slot = status.links.first().map(|link| link.slot as u8).unwrap_or(0);
        let user = resolve_plot_user(&status);
        let mut warnings = Vec::new();
        let available_devices = if self.host_handle != 0 {
            if let Err(err) = self.refresh_host_devices_cache() {
                warnings.push(format!(
                    "Failed to refresh remote device list; using cached devices: {}",
                    err
                ));
            }

            self.host_devices_cache.borrow().clone()
        } else {
            Vec::new()
        };

        let system_info = match run_remote_sdk_call(is_remote, || ffi::get_system_info(self.handle_ptr())) {
            Ok(value) => Some(value),
            Err(err) => {
                warnings.push(err);
                None
            }
        };
        let band_info = match run_remote_sdk_call(is_remote, || ffi::get_band_info(self.handle_ptr())) {
            Ok(value) => Some(value),
            Err(err) => {
                warnings.push(err);
                None
            }
        };
        let channel_info = match run_remote_sdk_call(is_remote, || ffi::get_channel_info(self.handle_ptr())) {
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
        let mcs_value = match run_remote_sdk_call(is_remote, || ffi::get_mcs(self.handle_ptr(), ffi::BB_DIR_TX, slot)) {
            Ok(value) => Some(value),
            Err(err) => {
                warnings.push(err);
                None
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

        Ok(WirelessRuntimeDetails {
            status,
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

    fn load_wireless_configuration_details(&mut self, mode: u8) -> Result<WirelessConfigurationDetails, String> {
        let is_remote = self.is_remote_mode();
        let mut warnings = Vec::new();

        let config_file = match run_remote_sdk_call(is_remote, || ffi::get_config_text(self.handle_ptr(), mode)) {
            Ok(value) => Some(value),
            Err(err) => {
                warnings.push(err);
                None
            }
        };
        let role = match run_remote_sdk_call(is_remote, || ffi::get_minidb_role(self.handle_ptr())) {
            Ok(value) => value,
            Err(err) => {
                warnings.push(err);
                None
            }
        };
        let band_bitmap = match run_remote_sdk_call(is_remote, || ffi::get_minidb_band(self.handle_ptr())) {
            Ok(value) => value,
            Err(err) => {
                warnings.push(err);
                None
            }
        };
        let power = match run_remote_sdk_call(is_remote, || ffi::get_minidb_power(self.handle_ptr())) {
            Ok(value) => value,
            Err(err) => {
                warnings.push(err);
                None
            }
        };
        let local_mac_address = match run_remote_sdk_call(is_remote, || ffi::get_minidb_local_mac(self.handle_ptr())) {
            Ok(value) => value,
            Err(err) => {
                warnings.push(err);
                None
            }
        };
        let ap_mac_address = match run_remote_sdk_call(is_remote, || ffi::get_minidb_ap_mac(self.handle_ptr())) {
            Ok(value) => value,
            Err(err) => {
                warnings.push(err);
                None
            }
        };

        let mut slot_macs = Vec::with_capacity(ffi::BB_SLOT_MAX);
        for slot in 0..ffi::BB_SLOT_MAX as u8 {
            let mac_address = match run_remote_sdk_call(is_remote, || ffi::get_minidb_slot_mac(self.handle_ptr(), slot)) {
                Ok(value) => value,
                Err(err) => {
                    warnings.push(format!("Failed to read MiniDB slot {} MAC: {}", slot, err));
                    None
                }
            };

            slot_macs.push(WirelessConfigurationSlotMac { slot, mac_address });
        }

        Ok(WirelessConfigurationDetails {
            mode,
            config_file,
            role,
            band_bitmap,
            power,
            local_mac_address,
            ap_mac_address,
            slot_macs,
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

            run_remote_sdk_call(true, || {
                ffi::get_status(
                    self.handle_ptr(),
                    ffi::BB_ALL_DATA_USER_BMP,
                    self.preferred_signal_user_for_mac(normalized_target),
                )
            })?;

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

    pub fn set_channel_mode(&mut self, auto_mode: bool) -> Result<(), String> {
        self.with_device_operation("set_channel_mode", |handle| ffi::set_channel_mode(handle, auto_mode))
    }

    pub fn set_channel(&mut self, dir: u8, chan_index: u8) -> Result<(), String> {
        self.with_device_operation("set_channel", |handle| ffi::set_channel(handle, dir, chan_index))
    }

    pub fn set_mcs_mode(&mut self, slot: u8, auto_mode: bool) -> Result<(), String> {
        self.with_device_operation("set_mcs_mode", |handle| ffi::set_mcs_mode(handle, slot, auto_mode))
    }

    pub fn set_mcs(&mut self, slot: u8, mcs: u8) -> Result<(), String> {
        self.with_device_operation("set_mcs", |handle| ffi::set_mcs(handle, slot, mcs))
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
        self.with_device_operation("set_band_mode", |handle| ffi::set_band_mode(handle, auto_mode))
    }

    pub fn set_band(&mut self, target_band: u8) -> Result<(), String> {
        self.with_device_operation("set_band", |handle| ffi::set_band(handle, target_band))
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
        })
    }

    pub fn set_configuration_text(&mut self, content: &str) -> Result<(), String> {
        self.with_device_operation("set_configuration_text", |handle| {
            ffi::set_config_text(handle, content)
        })
    }

    pub fn reset_configuration(&mut self) -> Result<(), String> {
        self.with_device_operation("reset_configuration", |handle| ffi::reset_config(handle))
    }

    pub fn reset_minidb(&mut self) -> Result<(), String> {
        self.with_device_operation("reset_minidb", |handle| ffi::reset_minidb(handle))
    }

    pub fn restore_factory_settings(&mut self) -> Result<(), String> {
        self.with_device_operation("restore_factory_settings", |handle| {
            ffi::reset_config(handle)?;
            ffi::reset_minidb(handle)
        })
    }

    pub fn set_minidb_role(&mut self, role: u8) -> Result<(), String> {
        self.with_device_operation("set_minidb_role", |handle| ffi::set_minidb_role(handle, role))
    }

    pub fn set_minidb_band(&mut self, band_bitmap: u8) -> Result<(), String> {
        self.with_device_operation("set_minidb_band", |handle| ffi::set_minidb_band(handle, band_bitmap))
    }

    pub fn set_minidb_power(&mut self, power: ffi::bb_phy_pwr_basic_t) -> Result<(), String> {
        self.with_device_operation("set_minidb_power", |handle| ffi::set_minidb_power(handle, power))
    }

    pub fn set_minidb_local_mac(&mut self, mac: &str) -> Result<(), String> {
        self.with_device_operation("set_minidb_local_mac", |handle| ffi::set_minidb_local_mac(handle, mac))
    }

    pub fn set_minidb_ap_mac(&mut self, mac: &str) -> Result<(), String> {
        self.with_device_operation("set_minidb_ap_mac", |handle| ffi::set_minidb_ap_mac(handle, mac))
    }

    pub fn set_minidb_slot_mac(&mut self, slot: u8, mac: &str) -> Result<(), String> {
        self.with_device_operation("set_minidb_slot_mac", |handle| ffi::set_minidb_slot_mac(handle, slot, mac))
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

    /// 初始化通信 socket
    pub fn initialize_socket(&self, socket_id: u32) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        // TX + RX 双向通信
        let flags = ffi::BB_SOCK_FLAG_TX | ffi::BB_SOCK_FLAG_RX;
        api.create_socket(socket_id, flags, 4096)
    }

    pub fn get_status_snapshot(&self) -> Result<ffi::BbGetStatusSummary, String> {
        tracing::debug!("BasebandManager::get_status_snapshot acquiring SDK status");
        let mut api = self.api.lock().unwrap();
        let result = api.get_status_summary();
        tracing::debug!(success = result.is_ok(), "BasebandManager::get_status_snapshot finished");
        result
    }

    pub fn get_wireless_runtime_details(&self) -> Result<WirelessRuntimeDetails, String> {
        let mut api = self.api.lock().unwrap();
        api.get_wireless_runtime_details()
    }

    pub fn get_wireless_configuration_details(&self, mode: u8) -> Result<WirelessConfigurationDetails, String> {
        let mut api = self.api.lock().unwrap();
        api.get_wireless_configuration_details(mode)
    }

    pub fn get_health_status(&self) -> BasebandHealthStatus {
        let mut api = self.api.lock().unwrap();
        api.get_health_status()
    }

    pub fn set_signal_user_preference(&self, user: u8) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_signal_user_preference(user)
    }

    pub fn set_pair_mode(&self, start: bool, slot_bmp: u8) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_pair_mode(start, slot_bmp)
    }

    pub fn set_pair_mode_with_blacklist(
        &self,
        start: bool,
        slot_bmp: u8,
        black_list: &[String],
    ) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_pair_mode_with_blacklist(start, slot_bmp, black_list)
    }

    pub fn get_pair_candidates(&self, slot: u8) -> Result<Vec<String>, String> {
        let mut api = self.api.lock().unwrap();
        api.get_pair_candidates(slot)
    }

    pub fn set_channel_mode(&self, auto_mode: bool) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_channel_mode(auto_mode)
    }

    pub fn set_channel(&self, dir: u8, chan_index: u8) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_channel(dir, chan_index)
    }

    pub fn set_mcs_mode(&self, slot: u8, auto_mode: bool) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_mcs_mode(slot, auto_mode)
    }

    pub fn set_mcs(&self, slot: u8, mcs: u8) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_mcs(slot, mcs)
    }

    pub fn set_power_mode(&self, pwr_mode: u8) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_power_mode(pwr_mode)
    }

    pub fn set_power(&self, user: u8, power_dbm: u8) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_power(user, power_dbm)
    }

    pub fn set_power_auto(&self, enabled: bool) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_power_auto(enabled)
    }

    pub fn switch_active_device(&self, target_mac: &str) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.switch_active_device(target_mac)
    }

    pub fn set_band_mode(&self, auto_mode: bool) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_band_mode(auto_mode)
    }

    pub fn set_band(&self, target_band: u8) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_band(target_band)
    }

    pub fn set_bandwidth(&self, slot: u8, dir: u8, bandwidth: u8) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_bandwidth(slot, dir, bandwidth)
    }

    pub fn set_bandwidth_mode(&self, slot: u8, auto_mode: bool) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_bandwidth_mode(slot, auto_mode)
    }

    pub fn set_baseband_role(&self, role: u8) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_baseband_role(role)
    }

    pub fn set_configuration_text(&self, content: &str) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_configuration_text(content)
    }

    pub fn reset_configuration(&self) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.reset_configuration()
    }

    pub fn reset_minidb(&self) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.reset_minidb()
    }

    pub fn restore_factory_settings(&self) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.restore_factory_settings()
    }

    pub fn set_minidb_role(&self, role: u8) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_minidb_role(role)
    }

    pub fn set_minidb_band(&self, band_bitmap: u8) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_minidb_band(band_bitmap)
    }

    pub fn set_minidb_power(&self, power: ffi::bb_phy_pwr_basic_t) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_minidb_power(power)
    }

    pub fn set_minidb_local_mac(&self, mac: &str) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_minidb_local_mac(mac)
    }

    pub fn set_minidb_ap_mac(&self, mac: &str) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_minidb_ap_mac(mac)
    }

    pub fn set_minidb_slot_mac(&self, slot: u8, mac: &str) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.set_minidb_slot_mac(slot, mac)
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseband_api_initialization() {
        // 测试初始化 - 需要实际的 ar8030 硬件或模拟
        let api = BasebandApi::get();
        println!("API Initialized: {}", api.is_initialized());
    }

}
