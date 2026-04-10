//! 基带 API 的 Rust 安全包装
//!
//! 提供类型安全的高级接口，用于与基带芯片 (ar8030) SOC 通信

use std::sync::{Arc, Mutex};
use crate::ffi;

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
    pub message: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BasebandHealthStatus {
    pub configured_mode: String,
    pub effective_mode: String,
    pub host: BasebandHostStatus,
    pub sdk: ffi::FfiRuntimeDiagnostics,
    pub device_open: BasebandOperationStatus,
    pub init: BasebandOperationStatus,
    pub start: BasebandOperationStatus,
    pub socket_init: BasebandOperationStatus,
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
                message: if configured {
                    "BB_HOST_ADDR detected; remote host mode configured".to_string()
                } else {
                    "BB_HOST_ADDR not set; local SDK mode configured".to_string()
                },
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
}

/// 基带 API 管理器 - 线程安全的单例
pub struct BasebandApi {
    initialized: bool,
    started: bool,
    device_handle: usize,
    host_handle: usize,
}

impl BasebandApi {
    /// 获取或初始化全局基带 API 实例
    pub fn get_with_health() -> (Self, BasebandHealthStatus) {
        let mut api = BasebandApi {
            initialized: false,
            started: false,
            device_handle: 0,
            host_handle: 0,
        };
        let mut health = BasebandHealthStatus::new();

        if let Ok(host_addr) = std::env::var("BB_HOST_ADDR") {
            let host_port = std::env::var("BB_HOST_PORT")
                .ok()
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or(0);

            tracing::info!("Using remote bb_host mode: {}:{}", host_addr, host_port);

            match ffi::connect_host(&host_addr, host_port) {
                Ok(host) => {
                    api.host_handle = host as usize;
                    health.host.connected = true;
                    health.host.message = format!("Connected to bb_host {}:{}", host_addr, host_port);

                    match ffi::open_first_device_on_host(host) {
                        Ok(device) => {
                            api.device_handle = device as usize;
                            health.device_open.attempted = true;
                            health.device_open.success = true;
                            health.device_open.message = "Opened first baseband device from remote host".to_string();
                        }
                        Err(e) => {
                            let _ = ffi::disconnect_host(host);
                            health.host.connected = false;
                            health.host.message = format!("Connected to host but failed to open device: {}", e);
                            health.device_open.attempted = true;
                            health.device_open.success = false;
                            health.device_open.message = e.clone();
                            health.sdk = ffi::runtime_diagnostics();
                            tracing::error!("Failed to open baseband device from host {}:{}: {}", host_addr, host_port, e);
                            return (api, health);
                        }
                    }
                }
                Err(e) => {
                    health.host.connected = false;
                    health.host.message = format!("Failed to connect bb_host {}:{}: {}", host_addr, host_port, e);
                    health.sdk = ffi::runtime_diagnostics();
                    tracing::error!("Failed to connect bb_host {}:{}: {}", host_addr, host_port, e);
                    return (api, health);
                }
            }
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

    pub fn start(&mut self) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }

        ffi::start(self.handle_ptr())?;
        self.started = true;
        Ok(())
    }

    /// 创建数据传输 socket
    pub fn create_socket(&self, socket_id: u32, flags: u32, max_size: u32) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }
        ffi::create_socket(self.handle_ptr(), socket_id, flags, max_size)
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
        if self.started {
            let _ = ffi::stop(self.handle_ptr());
            self.started = false;
        }

        if self.initialized {
            let _ = ffi::deinit(self.handle_ptr());

            if self.device_handle != 0 {
                let _ = ffi::close_device(self.handle_ptr());
                self.device_handle = 0;
            }

            if self.host_handle != 0 {
                let _ = ffi::disconnect_host(self.host_ptr());
                self.host_handle = 0;
            }

            tracing::info!("Baseband SDK deinitialized");
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

/// 基带状态数据 - 用于存储从 SOC 获取的原始数据
#[derive(Debug, Clone, serde::Serialize)]
pub struct BasebandStatus {
    /// 运行状态标志
    pub state_flags: u32,
    /// 信道频率 (MHz)
    pub frequency: u32,
    /// 带宽 (MHz)
    pub bandwidth: u16,
    /// 发射功率 (dBm)
    pub tx_power: i16,
    /// 接收信号强度 (dBm)
    pub rssi: i16,
}

/// 连接统计信息
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConnectionStats {
    /// 连接状态
    pub is_connected: bool,
    /// 远程 MAC 地址
    pub remote_mac: String,
    /// 信噪比 (dB)
    pub snr: i32,
    /// 主天线 RSSI (dBm)
    pub rssi_main: i32,
    /// 副天线 RSSI (dBm)
    pub rssi_aux: i32,
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

        health.start.attempted = true;

        match api.start() {
            Ok(()) => {
                health.start.success = true;
                health.start.message = "bb_start succeeded".to_string();
                health.effective_mode = "hardware".to_string();
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

    /// 接收来自 SOC 的数据
    pub fn receive_data(&self, socket_id: u32, buffer: &mut [u8]) -> Result<usize, String> {
        if buffer.is_empty() {
            return Err("Buffer is empty".to_string());
        }

        let bytes_read = unsafe {
            ffi::bb_socket_read(
                socket_id as std::os::raw::c_int,
                buffer.as_mut_ptr() as *mut std::os::raw::c_void,
                buffer.len() as std::os::raw::c_uint,
                -1,
            )
        };

        if bytes_read < 0 {
            Err(format!("Failed to receive data: error code {}", bytes_read))
        } else {
            Ok(bytes_read as usize)
        }
    }

    /// 检查是否有数据可读
    pub fn has_data_available(&self, socket_id: u32) -> bool {
        let mut dummy = [0u8; 1];
        // 尝试非阻塞读取 - 这是一个简化的检查方法
        let _result = unsafe {
            ffi::bb_socket_read(
                socket_id as std::os::raw::c_int,
                dummy.as_mut_ptr() as *mut std::os::raw::c_void,
                1 as std::os::raw::c_uint,
                0,
            )
        };
        false // 实际实现需要扩展
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
