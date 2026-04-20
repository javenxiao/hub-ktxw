//! 基带 API 的 Rust 安全包装
//!
//! 提供类型安全的高级接口，用于与基带芯片 (ar8030) SOC 通信

use std::{cell::RefCell, sync::{Arc, Mutex}, thread, time::{Duration, Instant}};
use crate::ffi;

const REMOTE_SDK_CALL_GAP: Duration = Duration::from_millis(20);
const REMOTE_DEVICE_SWITCH_GAP: Duration = Duration::from_millis(1200);
static REMOTE_SDK_LAST_CALL_AT: Mutex<Option<Instant>> = Mutex::new(None);

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
    pub mcs_mode: Option<ffi::BbMcsModeSummary>,
    pub mcs_value: Option<ffi::BbMcsValueSummary>,
    pub power_mode: Option<ffi::BbPowerModeSummary>,
    pub current_power: Option<ffi::BbCurrentPowerSummary>,
    pub power_auto: Option<ffi::BbPowerAutoSummary>,
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
    host_devices_cache: RefCell<Vec<ffi::BbDiscoveredDeviceSummary>>,
    plot_subscription_active: bool,
    plot_user: Option<u8>,
    active_device_mac: Option<String>,
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
        let mut api = BasebandApi {
            initialized: false,
            started: false,
            requires_start: true,
            device_handle: 0,
            host_handle: 0,
            host_devices_cache: RefCell::new(Vec::new()),
            plot_subscription_active: false,
            plot_user: None,
            active_device_mac: None,
            last_remote_device_switch_at: None,
            remote_device_handles: Vec::new(),
        };
        let mut health = BasebandHealthStatus::new();

        if let Ok(host_addr) = std::env::var("BB_HOST_ADDR") {
            let host_port = std::env::var("BB_HOST_PORT")
                .ok()
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or(0);

            tracing::info!("Using remote bb_host mode: {}:{}", host_addr, host_port);
            api.requires_start = false;
            health.init.message = "Skipped in remote bb_host mode".to_string();
            health.start.message = "Skipped in remote bb_host mode".to_string();

            match ffi::connect_host(&host_addr, host_port) {
                Ok(host) => {
                    api.host_handle = host as usize;
                    health.host.connected = true;
                    health.host.daemon_version = ffi::get_daemon_version(host).ok().and_then(|value| {
                        let trimmed = value.trim();

                        if trimmed.is_empty() {
                            None
                        } else {
                            Some(trimmed.to_string())
                        }
                    });
                    health.host.message = if let Some(version) = health.host.daemon_version.as_ref() {
                        format!("Connected to bb_host {}:{} (daemon_version={})", host_addr, host_port, version)
                    } else {
                        format!("Connected to bb_host {}:{} (daemon version unavailable)", host_addr, host_port)
                    };

                    match ffi::open_first_device_on_host(host) {
                        Ok((device, device_count)) => {
                            api.device_handle = device as usize;
                            api.initialized = true;
                            health.runtime.detected_device_count = Some(device_count);
                            health.device_open.attempted = true;
                            health.device_open.success = true;
                            health.device_open.message = format!("Opened first baseband device from remote host, device_count={}", device_count);
                            health.status_read.attempted = true;
                            health.effective_mode = "hardware-remote-bb-host".to_string();

                            match ffi::get_status(api.handle_ptr(), ffi::BB_ALL_DATA_USER_BMP) {
                                Ok(snapshot) => {
                                    health.status_read.success = true;
                                    health.status_read.message = "bb_ioctl(BB_GET_STATUS) succeeded".to_string();
                                    let active_mac = Self::normalize_mac(&snapshot.mac_hex);
                                    api.active_device_mac = Some(active_mac.clone());
                                    api.remember_remote_handle(active_mac, api.device_handle);
                                    tracing::info!("Skipping plot stream subscription in remote bb_host mode");
                                    if let Err(err) = api.refresh_host_devices_cache() {
                                        tracing::warn!("Failed to cache remote host devices: {}", err);
                                    }
                                    health.runtime.status_snapshot = Some(snapshot);
                                }
                                Err(err) => {
                                    health.status_read.success = false;
                                    health.status_read.message = err;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = ffi::disconnect_host(host);
                            api.host_handle = 0;
                            health.host.connected = false;
                            health.host.message = format!("Connected to host but failed to open device: {}", e);
                            health.device_open.attempted = true;
                            health.device_open.success = false;
                            health.device_open.message = e.clone();
                            tracing::error!("Failed to open baseband device from host {}:{}: {}", host_addr, host_port, e);
                        }
                    }
                }
                Err(e) => {
                    health.host.connected = false;
                    health.host.message = format!("Failed to connect bb_host {}:{}: {}", host_addr, host_port, e);
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

                match ffi::get_status(api.handle_ptr(), ffi::BB_ALL_DATA_USER_BMP) {
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

    pub fn enable_plot_stream(&mut self, user: u8) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        if self.plot_subscription_active && self.plot_user == Some(user) {
            return Ok(());
        }

        if self.plot_subscription_active {
            if let Some(previous_user) = self.plot_user {
                let _ = ffi::unsubscribe_plot_stream(self.handle_ptr(), previous_user);
            }
            self.plot_subscription_active = false;
            self.plot_user = None;
        }

        ffi::subscribe_plot_stream(self.handle_ptr(), user, ffi::BB_PLOT_POINT_MAX as u8)?;
        self.plot_subscription_active = true;
        self.plot_user = Some(user);
        Ok(())
    }

    /// 创建数据传输 socket
    pub fn create_socket(&self, socket_id: u32, flags: u32, max_size: u32) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }
        ffi::create_socket(self.handle_ptr(), socket_id, flags, max_size)
    }

    pub fn get_status_summary(&self) -> Result<ffi::BbGetStatusSummary, String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        run_remote_sdk_call(self.host_handle != 0, || {
            ffi::get_status(self.handle_ptr(), ffi::BB_ALL_DATA_USER_BMP)
        })
    }

    pub fn get_wireless_runtime_details(&self) -> Result<WirelessRuntimeDetails, String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        let is_remote = self.host_handle != 0;
        let status = run_remote_sdk_call(is_remote, || {
            ffi::get_status(self.handle_ptr(), ffi::BB_ALL_DATA_USER_BMP)
        })?;
        let slot = status.links.first().map(|link| link.slot as u8).unwrap_or(0);
        let user = status.active_user.unwrap_or(0);
        let mut warnings = Vec::new();
        let available_devices = if self.host_handle != 0 {
            if self.host_devices_cache.borrow().is_empty() {
                match run_remote_sdk_call(true, || ffi::list_host_devices(self.host_ptr())) {
                    Ok(devices) => {
                        *self.host_devices_cache.borrow_mut() = devices;
                    }
                    Err(err) => warnings.push(err),
                }
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
            mcs_mode,
            mcs_value,
            power_mode,
            current_power,
            power_auto,
            warnings,
        })
    }

    pub fn get_plot_snapshot(&self) -> Option<ffi::BbPlotSnapshotSummary> {
        if !self.initialized {
            return None;
        }

        ffi::latest_plot_snapshot()
    }

    pub fn switch_active_device(&mut self, target_mac: &str) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        if self.host_handle == 0 {
            return Err("Device switching requires remote bb_host mode".to_string());
        }

        let normalized_target = Self::normalize_mac(target_mac);
        if normalized_target.is_empty() {
            return Err("device_mac is required".to_string());
        }

        if self.active_device_mac.as_deref() == Some(normalized_target.as_str()) {
            return Ok(());
        }

        if let Some(cached_handle) = self.cached_remote_handle(&normalized_target) {
            self.device_handle = cached_handle;
            self.initialized = true;
            self.active_device_mac = Some(normalized_target);
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
        let new_status = match run_remote_sdk_call(true, || ffi::get_status(new_handle, ffi::BB_ALL_DATA_USER_BMP)) {
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
        self.active_device_mac = Some(normalized_target);
        self.remember_remote_handle(Self::normalize_mac(&new_status.mac_hex), self.device_handle);

        if self.host_handle == 0 {
            if let Err(err) = self.enable_plot_stream(new_status.active_user.unwrap_or(0)) {
                tracing::warn!("Failed to enable plot stream after device switch: {}", err);
            }
        }

        Ok(())
    }

    pub fn set_pair_mode(&self, start: bool, slot_bmp: u8) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        ffi::set_pair_mode(self.handle_ptr(), start, slot_bmp)
    }

    pub fn set_channel_mode(&self, auto_mode: bool) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        ffi::set_channel_mode(self.handle_ptr(), auto_mode)
    }

    pub fn set_channel(&self, dir: u8, chan_index: u8) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        ffi::set_channel(self.handle_ptr(), dir, chan_index)
    }

    pub fn set_mcs_mode(&self, slot: u8, auto_mode: bool) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        ffi::set_mcs_mode(self.handle_ptr(), slot, auto_mode)
    }

    pub fn set_mcs(&self, slot: u8, mcs: u8) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        ffi::set_mcs(self.handle_ptr(), slot, mcs)
    }

    pub fn set_power_mode(&self, pwr_mode: u8) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        ffi::set_power_mode(self.handle_ptr(), pwr_mode)
    }

    pub fn set_power(&self, user: u8, power_dbm: u8) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        ffi::set_power(self.handle_ptr(), user, power_dbm)
    }

    pub fn set_power_auto(&self, enabled: bool) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        ffi::set_power_auto(self.handle_ptr(), enabled)
    }

    pub fn set_band_mode(&self, auto_mode: bool) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        ffi::set_band_mode(self.handle_ptr(), auto_mode)
    }

    pub fn set_band(&self, target_band: u8) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        ffi::set_band(self.handle_ptr(), target_band)
    }

    pub fn set_bandwidth(&self, slot: u8, dir: u8, bandwidth: u8) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        ffi::set_bandwidth(self.handle_ptr(), slot, dir, bandwidth)
    }

    pub fn set_bandwidth_mode(&self, slot: u8, auto_mode: bool) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        ffi::set_bandwidth_mode(self.handle_ptr(), slot, auto_mode)
    }

    /// 获取通信统计信息
    pub fn get_stats(&self) -> CommunicationStats {
        CommunicationStats {
            recv_packets: 0,
            send_packets: 0,
            recv_bytes: 0,
            send_bytes: 0,
        }
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

        if self.initialized {
            if self.requires_start {
                let _ = ffi::deinit(self.handle_ptr());
            }

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

            if self.requires_start {
                tracing::info!("Baseband SDK deinitialized");
            } else {
                tracing::info!("Baseband remote host disconnected");
            }
        }
    }
}

/// 通信统计数据结构
#[derive(Debug, Clone, serde::Serialize)]
pub struct CommunicationStats {
    /// 接收数据包计数
    pub recv_packets: u32,
    /// 发送数据包计数
    pub send_packets: u32,
    /// 接收字节数
    pub recv_bytes: u64,
    /// 发送字节数
    pub send_bytes: u64,
}

/// 管理基带通信的核心模块
pub struct BasebandManager {
    api: Arc<Mutex<BasebandApi>>,
}

impl BasebandManager {
    pub fn initialize_with_health() -> (Option<Self>, BasebandHealthStatus) {
        let (mut api, mut health) = BasebandApi::get_with_health();

        if !api.is_initialized() {
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
                let plot_user = api
                    .get_status_summary()
                    .ok()
                    .and_then(|status| status.active_user)
                    .unwrap_or(0);

                if let Err(err) = api.enable_plot_stream(plot_user) {
                    tracing::warn!("Failed to enable plot stream in local SDK mode: {}", err);
                }

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
        let api = self.api.lock().unwrap();
        // TX + RX 双向通信
        let flags = ffi::BB_SOCK_FLAG_TX | ffi::BB_SOCK_FLAG_RX;
        api.create_socket(socket_id, flags, 4096)
    }

    /// 获取基带统计信息
    pub fn get_communication_stats(&self) -> CommunicationStats {
        let api = self.api.lock().unwrap();
        api.get_stats()
    }

    pub fn get_status_snapshot(&self) -> Result<ffi::BbGetStatusSummary, String> {
        tracing::debug!("BasebandManager::get_status_snapshot acquiring SDK status");
        let api = self.api.lock().unwrap();
        let result = api.get_status_summary();
        tracing::debug!(success = result.is_ok(), "BasebandManager::get_status_snapshot finished");
        result
    }

    pub fn get_wireless_runtime_details(&self) -> Result<WirelessRuntimeDetails, String> {
        let api = self.api.lock().unwrap();
        api.get_wireless_runtime_details()
    }

    pub fn get_plot_snapshot(&self) -> Option<ffi::BbPlotSnapshotSummary> {
        let api = self.api.lock().unwrap();
        api.get_plot_snapshot()
    }

    pub fn set_pair_mode(&self, start: bool, slot_bmp: u8) -> Result<(), String> {
        let api = self.api.lock().unwrap();
        api.set_pair_mode(start, slot_bmp)
    }

    pub fn set_channel_mode(&self, auto_mode: bool) -> Result<(), String> {
        let api = self.api.lock().unwrap();
        api.set_channel_mode(auto_mode)
    }

    pub fn set_channel(&self, dir: u8, chan_index: u8) -> Result<(), String> {
        let api = self.api.lock().unwrap();
        api.set_channel(dir, chan_index)
    }

    pub fn set_mcs_mode(&self, slot: u8, auto_mode: bool) -> Result<(), String> {
        let api = self.api.lock().unwrap();
        api.set_mcs_mode(slot, auto_mode)
    }

    pub fn set_mcs(&self, slot: u8, mcs: u8) -> Result<(), String> {
        let api = self.api.lock().unwrap();
        api.set_mcs(slot, mcs)
    }

    pub fn set_power_mode(&self, pwr_mode: u8) -> Result<(), String> {
        let api = self.api.lock().unwrap();
        api.set_power_mode(pwr_mode)
    }

    pub fn set_power(&self, user: u8, power_dbm: u8) -> Result<(), String> {
        let api = self.api.lock().unwrap();
        api.set_power(user, power_dbm)
    }

    pub fn set_power_auto(&self, enabled: bool) -> Result<(), String> {
        let api = self.api.lock().unwrap();
        api.set_power_auto(enabled)
    }

    pub fn switch_active_device(&self, target_mac: &str) -> Result<(), String> {
        let mut api = self.api.lock().unwrap();
        api.switch_active_device(target_mac)
    }

    pub fn set_band_mode(&self, auto_mode: bool) -> Result<(), String> {
        let api = self.api.lock().unwrap();
        api.set_band_mode(auto_mode)
    }

    pub fn set_band(&self, target_band: u8) -> Result<(), String> {
        let api = self.api.lock().unwrap();
        api.set_band(target_band)
    }

    pub fn set_bandwidth(&self, slot: u8, dir: u8, bandwidth: u8) -> Result<(), String> {
        let api = self.api.lock().unwrap();
        api.set_bandwidth(slot, dir, bandwidth)
    }

    pub fn set_bandwidth_mode(&self, slot: u8, auto_mode: bool) -> Result<(), String> {
        let api = self.api.lock().unwrap();
        api.set_bandwidth_mode(slot, auto_mode)
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

    #[test]
    fn test_communication_stats() {
        let api = BasebandApi::get();
        let stats = api.get_stats();
        println!("Stats: {:?}", stats);
    }
}
