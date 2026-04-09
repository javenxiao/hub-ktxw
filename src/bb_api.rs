//! 基带 API 的 Rust 安全包装
//! 
//! 提供类型安全的高级接口，用于与基带芯片 (ar8030) SOC 通信

use std::sync::{Arc, Mutex, Once};
use crate::ffi;

/// 全局初始化标志
static INIT: Once = Once::new();
static mut BB_INITIALIZED: bool = false;

/// 基带 API 管理器 - 线程安全的单例
pub struct BasebandApi {
    initialized: bool,
}

impl BasebandApi {
    /// 获取或初始化全局基带 API 实例
    pub fn get() -> Self {
        unsafe {
            INIT.call_once(|| {
                match ffi::init() {
                    Ok(_) => {
                        BB_INITIALIZED = true;
                        tracing::info!("Baseband SDK initialized successfully");
                    }
                    Err(e) => {
                        tracing::error!("Failed to initialize baseband SDK: {}", e);
                        BB_INITIALIZED = false;
                    }
                }
            });

            BasebandApi {
                initialized: BB_INITIALIZED,
            }
        }
    }

    /// 检查是否成功初始化
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// 创建数据传输 socket
    pub fn create_socket(&self, socket_id: u32, flags: u32, max_size: u32) -> Result<(), String> {
        if !self.initialized {
            return Err("Baseband API not initialized".to_string());
        }
        ffi::create_socket(socket_id, flags, max_size)
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
        if self.initialized {
            let _ = ffi::deinit();
            unsafe {
                BB_INITIALIZED = false;
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
    /// 创建新的基带管理器并初始化 SDK
    pub fn new() -> Result<Self, String> {
        let api = BasebandApi::get();

        if !api.is_initialized() {
            return Err("Failed to initialize baseband SDK".to_string());
        }

        Ok(BasebandManager {
            api: Arc::new(Mutex::new(api)),
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
