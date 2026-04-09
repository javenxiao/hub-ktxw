//! FFI 模块：与 ar8030_client.dll 的 C 语言接口绑定
//!
//! 该模块通过运行时动态加载 SDK，避免在缺少第三方依赖时进程启动失败。

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::{
    env,
    os::raw::{c_int, c_uint, c_void},
    path::PathBuf,
    sync::OnceLock,
};

use libloading::Library;

#[repr(C)]
pub struct bb_sock_opt_t {
    pub tx_buf_size: u32,
    pub rx_buf_size: u32,
}

#[repr(C)]
pub struct bb_dev_handle_t {
    _private: [u8; 0],
}

type BbInitFn = unsafe extern "C" fn() -> c_int;
type BbDeinitFn = unsafe extern "C" fn() -> c_int;
type BbSocketOpenFn = unsafe extern "C" fn(
    dev: *mut bb_dev_handle_t,
    slot: c_int,
    port: c_uint,
    flag: c_uint,
    opt: *mut bb_sock_opt_t,
) -> c_int;
type BbSocketWriteFn = unsafe extern "C" fn(socket_id: c_int, data: *const c_void, size: c_uint, timeout: c_int) -> c_int;
type BbSocketReadFn = unsafe extern "C" fn(socket_id: c_int, data: *mut c_void, size: c_uint, timeout: c_int) -> c_int;

struct Ar8030Sdk {
    _library: Library,
    bb_init: BbInitFn,
    bb_deinit: BbDeinitFn,
    bb_socket_open: BbSocketOpenFn,
    bb_socket_write: BbSocketWriteFn,
    bb_socket_read: BbSocketReadFn,
}

static SDK: OnceLock<Result<Ar8030Sdk, String>> = OnceLock::new();

// ============ 常数定义 ============
pub const BB_MAC_LEN: usize = 4;
pub const BB_REG_PAGE_NUM: usize = 16;
pub const BB_REG_PAGE_SIZE: usize = 256;
pub const BB_CFG_PAGE_SIZE: usize = 1024;
pub const BB_PLOT_POINT_MAX: usize = 10;
pub const BB_BLACK_LIST_SIZE: usize = 3;
pub const BB_RC_FREQ_NUM: usize = 4;
pub const BB_SOCK_INFO_NUM: usize = 8;
pub const BB_REMOTE_CMD_WAIT_MAX: usize = 8;

// Socket 选项标志
pub const BB_SOCK_FLAG_RX: u32 = 1 << 0;
pub const BB_SOCK_FLAG_TX: u32 = 1 << 1;
pub const BB_SOCK_FLAG_TROC: u32 = 1 << 2;
pub const BB_SOCK_FLAG_DATAGRAM: u32 = 1 << 3;
pub const BB_SOCK_FLAG_SBUS: u32 = 1 << 4;

// 信道配置标志
pub const BB_CHAN_HOP_AUTO: u32 = 1 << 0;
pub const BB_CHAN_BAND_HOP_AUTO: u32 = 1 << 1;
pub const BB_CHAN_COMPLIANCE: u32 = 1 << 2;
pub const BB_CHAN_MULTI_MODE: u32 = 1 << 3;
pub const BB_CHAN_SUBCHAN_ENABLE: u32 = 1 << 4;

// MCS 配置标志
pub const BB_MCS_SWITCH_ENABLE: u32 = 1 << 0;
pub const BB_MCS_SWITCH_AUTO: u32 = 1 << 1;

// ioctl 请求类型定义
pub const BB_REQ_CFG: u32 = 0;
pub const BB_REQ_GET: u32 = 1;
pub const BB_REQ_SET: u32 = 2;
pub const BB_REQ_CB: u32 = 3;
pub const BB_REQ_SOCKET: u32 = 4;
pub const BB_REQ_DBG: u32 = 5;
pub const BB_REQ_REMOTE: u32 = 6;
pub const BB_REQ_RPC: u32 = 10;
pub const BB_REQ_RPC_IOCTL: u32 = 11;
pub const BB_REQ_PLAT_CTL: u32 = 12;

// 配置命令字
pub const BB_CFG_AP_BASIC: u32 = bb_request(BB_REQ_CFG, 0);
pub const BB_CFG_DEV_BASIC: u32 = bb_request(BB_REQ_CFG, 1);
pub const BB_CFG_CHANNEL: u32 = bb_request(BB_REQ_CFG, 2);
pub const BB_CFG_CANDIDATES: u32 = bb_request(BB_REQ_CFG, 3);
pub const BB_CFG_USER_PARA: u32 = bb_request(BB_REQ_CFG, 4);
pub const BB_CFG_SLOT_RX_MCS: u32 = bb_request(BB_REQ_CFG, 5);
pub const BB_CFG_ANY_CHANNEL: u32 = bb_request(BB_REQ_CFG, 6);
pub const BB_CFG_DISTC: u32 = bb_request(BB_REQ_CFG, 7);
pub const BB_CFG_AP_SYNC_MODE: u32 = bb_request(BB_REQ_CFG, 9);
pub const BB_CFG_BR_HOP_POLICY: u32 = bb_request(BB_REQ_CFG, 10);
pub const BB_CFG_PWR_BASIC: u32 = bb_request(BB_REQ_CFG, 11);
pub const BB_CFG_RC_HOP_POLICY: u32 = bb_request(BB_REQ_CFG, 12);
pub const BB_CFG_POWER_SAVE: u32 = bb_request(BB_REQ_CFG, 13);
pub const BB_CFG_LNA: u32 = bb_request(BB_REQ_CFG, 14);
pub const BB_CFG_RF_POLICY: u32 = bb_request(BB_REQ_CFG, 15);
pub const BB_CFG_SHARE_SLOT: u32 = bb_request(BB_REQ_CFG, 16);
pub const BB_CFG_PWR_EXTEND: u32 = bb_request(BB_REQ_CFG, 17);

// 获取命令字
pub const BB_GET_STATUS: u32 = bb_request(BB_REQ_GET, 0);
pub const BB_GET_PAIR_RESULT: u32 = bb_request(BB_REQ_GET, 1);
pub const BB_GET_AP_MAC: u32 = bb_request(BB_REQ_GET, 2);
pub const BB_GET_CANDIDATES: u32 = bb_request(BB_REQ_GET, 3);
pub const BB_GET_USER_QUALITY: u32 = bb_request(BB_REQ_GET, 4);
pub const BB_GET_DISTC_RESULT: u32 = bb_request(BB_REQ_GET, 5);

// ============ 辅助函数 ============
#[inline]
pub const fn bb_request(req_type: u32, order: u32) -> u32 {
    (req_type << 24) | order
}

#[inline]
pub const fn bb_request_type(req: u32) -> u32 {
    req >> 24
}

fn sdk() -> Result<&'static Ar8030Sdk, String> {
    match SDK.get_or_init(load_sdk) {
        Ok(sdk) => Ok(sdk),
        Err(err) => Err(err.clone()),
    }
}

fn load_sdk() -> Result<Ar8030Sdk, String> {
    let candidates = dll_candidates();

    for dll_path in &candidates {
        if !dll_path.exists() {
            continue;
        }

        unsafe {
            let library = Library::new(dll_path)
                .map_err(|err| format!("failed to load {}: {}", dll_path.display(), err))?;

            let bb_init = load_symbol::<BbInitFn>(&library, b"bb_init\0")?;
            let bb_deinit = load_symbol::<BbDeinitFn>(&library, b"bb_deinit\0")?;
            let bb_socket_open = load_symbol::<BbSocketOpenFn>(&library, b"bb_socket_open\0")?;
            let bb_socket_write = load_symbol::<BbSocketWriteFn>(&library, b"bb_socket_write\0")?;
            let bb_socket_read = load_symbol::<BbSocketReadFn>(&library, b"bb_socket_read\0")?;

            return Ok(Ar8030Sdk {
                _library: library,
                bb_init,
                bb_deinit,
                bb_socket_open,
                bb_socket_write,
                bb_socket_read,
            });
        }
    }

    Err(format!(
        "ar8030_client.dll not found in any expected location: {}",
        candidates
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

unsafe fn load_symbol<T: Copy>(library: &Library, symbol_name: &[u8]) -> Result<T, String> {
    library
        .get::<T>(symbol_name)
        .map(|symbol| *symbol)
        .map_err(|err| format!("failed to resolve symbol {}: {}", String::from_utf8_lossy(symbol_name), err))
}

fn dll_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(exe_path) = env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidates.push(exe_dir.join("ar8030_client.dll"));
        }
    }

    if let Ok(current_dir) = env::current_dir() {
        candidates.push(current_dir.join("ar8030_client.dll"));
        candidates.push(current_dir.join("third_party").join("lib").join("ar8030_client.dll"));
    }

    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("third_party").join("lib").join("ar8030_client.dll"));
    candidates
}

// ============ 安全的 Rust 包装函数 ============

/// 初始化基带 SDK
pub fn init() -> Result<(), String> {
    let sdk = sdk()?;

    unsafe {
        match (sdk.bb_init)() {
            0 => Ok(()),
            e => Err(format!("bb_init failed with code: {}", e)),
        }
    }
}

/// 反初始化基带 SDK
pub fn deinit() -> Result<(), String> {
    let sdk = sdk()?;

    unsafe {
        match (sdk.bb_deinit)() {
            0 => Ok(()),
            e => Err(format!("bb_deinit failed with code: {}", e)),
        }
    }
}

/// 创建数据传输 socket
pub fn create_socket(socket_id: u32, flags: u32, max_size: u32) -> Result<(), String> {
    let sdk = sdk()?;
    let mut opt = bb_sock_opt_t {
        tx_buf_size: if flags & BB_SOCK_FLAG_TX != 0 { max_size } else { 0 },
        rx_buf_size: if flags & BB_SOCK_FLAG_RX != 0 { max_size } else { 0 },
    };

    unsafe {
        match (sdk.bb_socket_open)(std::ptr::null_mut(), 0, socket_id as c_uint, flags as c_uint, &mut opt) {
            fd if fd >= 0 => Ok(()),
            e => Err(format!("Open socket for port {} failed with code: {}", socket_id, e)),
        }
    }
}

pub unsafe fn bb_socket_write(socket_id: c_int, data: *const c_void, size: c_uint, timeout: c_int) -> c_int {
    match sdk() {
        Ok(sdk) => (sdk.bb_socket_write)(socket_id, data, size, timeout),
        Err(_) => -1,
    }
}

pub unsafe fn bb_socket_read(socket_id: c_int, data: *mut c_void, size: c_uint, timeout: c_int) -> c_int {
    match sdk() {
        Ok(sdk) => (sdk.bb_socket_read)(socket_id, data, size, timeout),
        Err(_) => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bb_request_macro() {
        assert_eq!(BB_CFG_AP_BASIC, 0x00000000);
        assert_eq!(BB_CFG_DEV_BASIC, 0x00000001);
        assert_eq!(BB_GET_STATUS, 0x01000000);
    }
}
