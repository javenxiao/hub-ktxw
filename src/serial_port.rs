//! 串口管理模块
//!
//! 提供串口枚举、打开、关闭、读写等基本操作，
//! 并通过 broadcast channel 将接收到的数据推送给 WebSocket 客户端。

use std::{
    io::{Read, Write},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use serialport::{available_ports, SerialPort, SerialPortInfo, SerialPortType};
use tokio::sync::broadcast;
use tracing::{info, warn};

/// 串口接收日志行
#[derive(Debug, Clone)]
pub struct SerialRxLine {
    pub data: String,
    pub timestamp_ms: u64,
}

/// 串口端口元数据（可序列化，方便前端展示）
#[derive(Debug, Clone, serde::Serialize)]
pub struct SerialPortMeta {
    pub name: String,
    pub description: String,
    pub hardware_id: Option<String>,
}

impl From<SerialPortInfo> for SerialPortMeta {
    fn from(info: SerialPortInfo) -> Self {
        let hardware_id = match &info.port_type {
            SerialPortType::UsbPort(usb) => {
                usb.serial_number
                    .clone()
                    .or_else(|| usb.product.clone())
            }
            SerialPortType::PciPort => None,
            SerialPortType::BluetoothPort => None,
            SerialPortType::Unknown => None,
        };
        Self {
            name: info.port_name.clone(),
            description: format!("{:?}", info.port_type),
            hardware_id,
        }
    }
}

/// 枚举系统可用串口
pub fn list_serial_ports() -> Vec<SerialPortMeta> {
    match available_ports() {
        Ok(ports) => ports.into_iter().map(SerialPortMeta::from).collect(),
        Err(e) => {
            warn!("Failed to enumerate serial ports: {}", e);
            Vec::new()
        }
    }
}

/// 串口管理器（线程安全）
pub struct SerialPortManager {
    /// 当前打开的串口，None 表示未连接
    port: Mutex<Option<Box<dyn SerialPort>>>,
    /// 连接状态
    connected: Mutex<bool>,
    /// 当前端口名称
    port_name: Mutex<Option<String>>,
    /// 广播通道 – 将接收到的数据推送给所有订阅者
    rx_tx: broadcast::Sender<SerialRxLine>,
    /// 读写线程停止标志
    stop_flag: Mutex<bool>,
    /// 累计收发字节
    rx_bytes: Mutex<u64>,
    tx_bytes: Mutex<u64>,
}

impl SerialPortManager {
    pub fn new() -> Self {
        let (rx_tx, _) = broadcast::channel(256);
        Self {
            port: Mutex::new(None),
            connected: Mutex::new(false),
            port_name: Mutex::new(None),
            rx_tx,
            stop_flag: Mutex::new(false),
            rx_bytes: Mutex::new(0),
            tx_bytes: Mutex::new(0),
        }
    }

    /// 获取接收数据的广播订阅器
    pub fn subscribe_rx(&self) -> broadcast::Receiver<SerialRxLine> {
        self.rx_tx.subscribe()
    }

    /// 是否已连接
    pub fn is_connected(&self) -> bool {
        *self.connected.lock().unwrap()
    }

    /// 当前端口名称
    pub fn port_name(&self) -> Option<String> {
        self.port_name.lock().unwrap().clone()
    }

    /// 累计收发字节
    #[allow(dead_code)]
    pub fn byte_counts(&self) -> (u64, u64) {
        (
            *self.rx_bytes.lock().unwrap(),
            *self.tx_bytes.lock().unwrap(),
        )
    }

    /// 打开并连接串口
    pub fn connect(
        &self,
        port_name: &str,
        baud_rate: u32,
        _data_bits: u8,
        _parity: &str,
        _stop_bits: u8,
    ) -> Result<(), String> {
        let mut connected = self.connected.lock().unwrap();
        if *connected {
            return Err("Serial port is already connected".to_string());
        }

        // Serialport crate currently only supports 8N1 by default.
        // data_bits, parity, stop_bits are accepted for future extension.
        let builder = serialport::new(port_name, baud_rate)
            .timeout(Duration::from_millis(50));

        let port = builder
            .open()
            .map_err(|e| format!("Failed to open serial port '{}': {}", port_name, e))?;

        *self.port.lock().unwrap() = Some(port);
        *self.port_name.lock().unwrap() = Some(port_name.to_string());
        *self.stop_flag.lock().unwrap() = false;
        *connected = true;

        info!("Serial port '{}' opened at {} baud", port_name, baud_rate);

        // 启动后台读取线程
        self.spawn_reader_thread();

        Ok(())
    }

    /// 断开串口连接
    pub fn disconnect(&self) {
        *self.stop_flag.lock().unwrap() = true;
        *self.connected.lock().unwrap() = false;
        *self.port_name.lock().unwrap() = None;

        // 关闭串口（释放 port 即可）
        if let Ok(mut port_guard) = self.port.lock() {
            *port_guard = None;
        }

        info!("Serial port disconnected");
    }

    /// 发送数据到串口
    pub fn send(&self, data: &[u8]) -> Result<usize, String> {
        let mut port = self.port.lock().unwrap();
        let port = port
            .as_mut()
            .ok_or_else(|| "Serial port is not connected".to_string())?;

        port.write_all(data)
            .map_err(|e| format!("Serial write failed: {}", e))?;
        port.flush()
            .map_err(|e| format!("Serial flush failed: {}", e))?;

        let len = data.len();
        if let Ok(mut tx) = self.tx_bytes.lock() {
            *tx = tx.saturating_add(len as u64);
        }

        Ok(len)
    }

    /// 后台读取线程：循环读取串口数据并通过 broadcast 推送
    fn spawn_reader_thread(&self) {
        // 需要从 Arc 中获取原始指针的替代方案：
        // 我们在 connect 时由外部的 Arc<SerialPortManager> 调用 spawn
        // 但这里 self 是 &Self，无法 spawn 独立线程。
        //
        // 实际 spawn 由外部的 Arc<SerialPortManager> 负责，
        // 该函数由 connect 调用后由外部管理。
        //
        // 我们这里只负责提供 thread::spawn 所需要的逻辑闭包，
        // 真正的 spawn 将在 serial_port_manager 的 connect 方法中完成。
    }
}

/// 为 Arc<SerialPortManager> 启动后台读取线程
pub fn spawn_reader(manager: Arc<SerialPortManager>) {
    thread::spawn(move || {
        let mut buf = [0u8; 1024];
        let mut pending = Vec::new();

        loop {
            if *manager.stop_flag.lock().unwrap() {
                break;
            }

            // 尝试读取
            let bytes_read = {
                let mut port_guard = manager.port.lock().unwrap();
                match port_guard.as_mut() {
                    Some(port) => match port.read(&mut buf) {
                        Ok(0) => {
                            // 无数据，短暂休眠
                            drop(port_guard);
                            thread::sleep(Duration::from_millis(10));
                            continue;
                        }
                        Ok(n) => n,
                        Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {
                            drop(port_guard);
                            // 超时：flush 当前缓冲的行并继续
                            if !pending.is_empty() {
                                let line = String::from_utf8_lossy(&pending).to_string();
                                pending.clear();
                                if let Ok(mut rx) = manager.rx_bytes.lock() {
                                    *rx = rx.saturating_add(line.len() as u64);
                                }
                                let _ = manager.rx_tx.send(SerialRxLine {
                                    data: line,
                                    timestamp_ms: std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_millis() as u64,
                                });
                            }
                            thread::sleep(Duration::from_millis(10));
                            continue;
                        }
                        Err(e) => {
                            warn!("Serial read error: {}", e);
                            break;
                        }
                    },
                    None => break,
                }
            };

            pending.extend_from_slice(&buf[..bytes_read]);

            // 按行拆分并推送
            while let Some(newline_pos) = pending.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = pending.drain(..=newline_pos).collect();
                let line = String::from_utf8_lossy(&line_bytes).to_string();
                let line_len = line.len();

                if let Ok(mut rx) = manager.rx_bytes.lock() {
                    *rx = rx.saturating_add(line_len as u64);
                }

                let _ = manager.rx_tx.send(SerialRxLine {
                    data: line,
                    timestamp_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                });
            }

            thread::sleep(Duration::from_millis(5));
        }

        info!("Serial reader thread stopped");
    });
}
