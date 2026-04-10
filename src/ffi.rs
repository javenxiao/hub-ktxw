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
    path::{Path, PathBuf},
    sync::OnceLock,
};

use libloading::Library;

#[derive(Debug, Clone, serde::Serialize)]
pub struct RuntimeFileStatus {
    pub path: String,
    pub exists: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FfiRuntimeDiagnostics {
    pub load_attempted: bool,
    pub sdk_loaded: bool,
    pub selected_dll_path: Option<String>,
    pub sdk_load_error: Option<String>,
    pub dll_candidates: Vec<RuntimeFileStatus>,
    pub dependency_candidates: Vec<RuntimeFileStatus>,
    pub loaded_dependency_paths: Vec<String>,
}

#[repr(C)]
pub struct bb_sock_opt_t {
    pub tx_buf_size: u32,
    pub rx_buf_size: u32,
}

#[repr(C)]
pub struct bb_dev_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct bb_dev_handle_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct bb_host_t {
    _private: [u8; 0],
}

type BbInitFn = unsafe extern "C" fn(handle: *mut bb_dev_handle_t) -> c_int;
type BbDeinitFn = unsafe extern "C" fn(handle: *mut bb_dev_handle_t) -> c_int;
type BbStartFn = unsafe extern "C" fn(handle: *mut bb_dev_handle_t) -> c_int;
type BbStopFn = unsafe extern "C" fn(handle: *mut bb_dev_handle_t) -> c_int;
type BbHostConnectFn = unsafe extern "C" fn(phost: *mut *mut bb_host_t, addr: *const i8, port: c_int) -> c_int;
type BbHostDisconnectFn = unsafe extern "C" fn(phost: *mut bb_host_t) -> c_int;
type BbDevGetListFn = unsafe extern "C" fn(phost: *mut bb_host_t, plist: *mut *mut bb_dev_t) -> c_int;
type BbDevOpenFn = unsafe extern "C" fn(devs: *mut bb_dev_t) -> *mut bb_dev_handle_t;
type BbDevCloseFn = unsafe extern "C" fn(handle: *mut bb_dev_handle_t) -> c_int;
type BbDevFreeListFn = unsafe extern "C" fn(plist: *mut bb_dev_t) -> c_int;
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
    _dependencies: Vec<Library>,
    _library: Library,
    loaded_dll_path: PathBuf,
    loaded_dependency_paths: Vec<PathBuf>,
    bb_init: BbInitFn,
    bb_deinit: BbDeinitFn,
    bb_start: BbStartFn,
    bb_stop: BbStopFn,
    bb_host_connect: BbHostConnectFn,
    bb_host_disconnect: BbHostDisconnectFn,
    bb_dev_getlist: BbDevGetListFn,
    bb_dev_open: BbDevOpenFn,
    bb_dev_close: BbDevCloseFn,
    bb_dev_freelist: BbDevFreeListFn,
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
            let (dependencies, loaded_dependency_paths) = preload_dependencies(dll_path.parent())?;
            let library = Library::new(dll_path)
                .map_err(|err| format!("failed to load {}: {}", dll_path.display(), err))?;

            let bb_init = load_symbol::<BbInitFn>(&library, b"bb_init\0")?;
            let bb_deinit = load_symbol::<BbDeinitFn>(&library, b"bb_deinit\0")?;
            let bb_start = load_symbol::<BbStartFn>(&library, b"bb_start\0")?;
            let bb_stop = load_symbol::<BbStopFn>(&library, b"bb_stop\0")?;
            let bb_host_connect = load_symbol::<BbHostConnectFn>(&library, b"bb_host_connect\0")?;
            let bb_host_disconnect = load_symbol::<BbHostDisconnectFn>(&library, b"bb_host_disconnect\0")?;
            let bb_dev_getlist = load_symbol::<BbDevGetListFn>(&library, b"bb_dev_getlist\0")?;
            let bb_dev_open = load_symbol::<BbDevOpenFn>(&library, b"bb_dev_open\0")?;
            let bb_dev_close = load_symbol::<BbDevCloseFn>(&library, b"bb_dev_close\0")?;
            let bb_dev_freelist = load_symbol::<BbDevFreeListFn>(&library, b"bb_dev_freelist\0")?;
            let bb_socket_open = load_symbol::<BbSocketOpenFn>(&library, b"bb_socket_open\0")?;
            let bb_socket_write = load_symbol::<BbSocketWriteFn>(&library, b"bb_socket_write\0")?;
            let bb_socket_read = load_symbol::<BbSocketReadFn>(&library, b"bb_socket_read\0")?;

            return Ok(Ar8030Sdk {
                _dependencies: dependencies,
                _library: library,
                loaded_dll_path: dll_path.clone(),
                loaded_dependency_paths,
                bb_init,
                bb_deinit,
                bb_start,
                bb_stop,
                bb_host_connect,
                bb_host_disconnect,
                bb_dev_getlist,
                bb_dev_open,
                bb_dev_close,
                bb_dev_freelist,
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

pub fn runtime_diagnostics() -> FfiRuntimeDiagnostics {
    let dll_candidates = dll_candidates()
        .into_iter()
        .map(|path| RuntimeFileStatus {
            exists: path.exists(),
            path: path.display().to_string(),
        })
        .collect::<Vec<_>>();

    let dependency_candidates = diagnostic_dependency_candidates()
        .into_iter()
        .map(|path| RuntimeFileStatus {
            exists: path.exists(),
            path: path.display().to_string(),
        })
        .collect::<Vec<_>>();

    match SDK.get() {
        Some(Ok(sdk)) => FfiRuntimeDiagnostics {
            load_attempted: true,
            sdk_loaded: true,
            selected_dll_path: Some(sdk.loaded_dll_path.display().to_string()),
            sdk_load_error: None,
            dll_candidates,
            dependency_candidates,
            loaded_dependency_paths: sdk
                .loaded_dependency_paths
                .iter()
                .map(|path| path.display().to_string())
                .collect(),
        },
        Some(Err(err)) => FfiRuntimeDiagnostics {
            load_attempted: true,
            sdk_loaded: false,
            selected_dll_path: None,
            sdk_load_error: Some(err.clone()),
            dll_candidates,
            dependency_candidates,
            loaded_dependency_paths: Vec::new(),
        },
        None => FfiRuntimeDiagnostics {
            load_attempted: false,
            sdk_loaded: false,
            selected_dll_path: None,
            sdk_load_error: None,
            dll_candidates,
            dependency_candidates,
            loaded_dependency_paths: Vec::new(),
        },
    }
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
    dedupe_paths(candidates)
}

unsafe fn preload_dependencies(base_dir: Option<&Path>) -> Result<(Vec<Library>, Vec<PathBuf>), String> {
    let mut libraries = Vec::new();
    let mut loaded_paths = Vec::new();

    for dependency in dependency_candidates(base_dir) {
        if !dependency.exists() {
            continue;
        }

        match Library::new(&dependency) {
            Ok(library) => {
                loaded_paths.push(dependency.clone());
                libraries.push(library);
            }
            Err(err) => {
                return Err(format!("failed to preload dependency {}: {}", dependency.display(), err));
            }
        }
    }

    Ok((libraries, loaded_paths))
}

fn dependency_candidates(base_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Some(dir) = base_dir {
        candidates.push(dir.join("pthread.dll"));
        candidates.push(dir.join("vcruntime140d.dll"));
        candidates.push(dir.join("ucrtbased.dll"));
    }

    if let Ok(exe_path) = env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidates.push(exe_dir.join("pthread.dll"));
            candidates.push(exe_dir.join("vcruntime140d.dll"));
            candidates.push(exe_dir.join("ucrtbased.dll"));
        }
    }

    let manifest_lib_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("third_party").join("lib");
    candidates.push(manifest_lib_dir.join("pthread.dll"));

    if let Ok(current_dir) = env::current_dir() {
        candidates.push(current_dir.join("third_party").join("lib").join("pthread.dll"));
    }

    candidates.push(PathBuf::from(r"C:\Windows\System32\vcruntime140d.dll"));
    candidates.push(PathBuf::from(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC\14.44.35112\debug_nonredist\x64\Microsoft.VC143.DebugCRT\vcruntime140d.dll"));
    candidates.push(PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\ucrt\ucrtbased.dll"));
    candidates.push(PathBuf::from(r"C:\Program Files (x86)\Microsoft SDKs\Windows Kits\10\ExtensionSDKs\Microsoft.UniversalCRT.Debug\10.0.26100.0\Redist\Debug\x64\ucrtbased.dll"));
    dedupe_paths(candidates)
}

fn diagnostic_dependency_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    for dll_path in dll_candidates() {
        if let Some(parent) = dll_path.parent() {
            candidates.extend(dependency_candidates(Some(parent)));
        }
    }

    candidates.extend(dependency_candidates(None));
    dedupe_paths(candidates)
}

fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut unique = Vec::new();

    for path in paths {
        if !unique.iter().any(|existing: &PathBuf| existing == &path) {
            unique.push(path);
        }
    }

    unique
}

// ============ 安全的 Rust 包装函数 ============

/// 初始化基带 SDK
pub fn connect_host(addr: &str, port: i32) -> Result<*mut bb_host_t, String> {
    let sdk = sdk()?;
    let address = std::ffi::CString::new(addr)
        .map_err(|_| format!("Invalid host address: {}", addr))?;

    unsafe {
        let mut host: *mut bb_host_t = std::ptr::null_mut();
        match (sdk.bb_host_connect)(&mut host, address.as_ptr(), port as c_int) {
            0 if !host.is_null() => Ok(host),
            0 => Err("bb_host_connect succeeded but returned null host handle".to_string()),
            e => Err(format!("bb_host_connect failed with code: {}", e)),
        }
    }
}

pub fn disconnect_host(host: *mut bb_host_t) -> Result<(), String> {
    let sdk = sdk()?;

    if host.is_null() {
        return Ok(());
    }

    unsafe {
        match (sdk.bb_host_disconnect)(host) {
            0 => Ok(()),
            e => Err(format!("bb_host_disconnect failed with code: {}", e)),
        }
    }
}

pub fn open_first_device_on_host(host: *mut bb_host_t) -> Result<*mut bb_dev_handle_t, String> {
    let sdk = sdk()?;

    unsafe {
        let mut device_list: *mut bb_dev_t = std::ptr::null_mut();
        let device_count = (sdk.bb_dev_getlist)(host, &mut device_list);

        if device_count < 0 {
            return Err(format!("bb_dev_getlist failed with code: {}", device_count));
        }

        if device_count == 0 || device_list.is_null() {
            return Err("No baseband devices detected".to_string());
        }

        let handle = (sdk.bb_dev_open)(device_list);
        let _ = (sdk.bb_dev_freelist)(device_list);

        if handle.is_null() {
            Err("bb_dev_open returned null handle".to_string())
        } else {
            Ok(handle)
        }
    }
}

pub fn close_device(handle: *mut bb_dev_handle_t) -> Result<(), String> {
    let sdk = sdk()?;

    if handle.is_null() {
        return Ok(());
    }

    unsafe {
        match (sdk.bb_dev_close)(handle) {
            0 => Ok(()),
            e => Err(format!("bb_dev_close failed with code: {}", e)),
        }
    }
}

pub fn init(handle: *mut bb_dev_handle_t) -> Result<(), String> {
    let sdk = sdk()?;

    unsafe {
        match (sdk.bb_init)(handle) {
            0 => Ok(()),
            e => Err(format!("bb_init failed with code: {}", e)),
        }
    }
}

pub fn start(handle: *mut bb_dev_handle_t) -> Result<(), String> {
    let sdk = sdk()?;

    unsafe {
        match (sdk.bb_start)(handle) {
            value if value >= 0 => Ok(()),
            e => Err(format!("bb_start failed with code: {}", e)),
        }
    }
}

/// 反初始化基带 SDK
pub fn deinit(handle: *mut bb_dev_handle_t) -> Result<(), String> {
    let sdk = sdk()?;

    unsafe {
        match (sdk.bb_deinit)(handle) {
            0 => Ok(()),
            e => Err(format!("bb_deinit failed with code: {}", e)),
        }
    }
}

pub fn stop(handle: *mut bb_dev_handle_t) -> Result<(), String> {
    let sdk = sdk()?;

    unsafe {
        match (sdk.bb_stop)(handle) {
            0 => Ok(()),
            e => Err(format!("bb_stop failed with code: {}", e)),
        }
    }
}

/// 创建数据传输 socket
pub fn create_socket(
    handle: *mut bb_dev_handle_t,
    socket_id: u32,
    flags: u32,
    max_size: u32,
) -> Result<(), String> {
    let sdk = sdk()?;
    let mut opt = bb_sock_opt_t {
        tx_buf_size: if flags & BB_SOCK_FLAG_TX != 0 { max_size } else { 0 },
        rx_buf_size: if flags & BB_SOCK_FLAG_RX != 0 { max_size } else { 0 },
    };

    unsafe {
        match (sdk.bb_socket_open)(handle, 0, socket_id as c_uint, flags as c_uint, &mut opt) {
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
