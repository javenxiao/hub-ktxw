//! FFI 模块：与 ar8030_client.dll 的 C 语言接口绑定
//!
//! 该模块通过运行时动态加载 SDK，避免在缺少第三方依赖时进程启动失败。

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::{
    env,
    ffi::CStr,
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

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbLinkStatusSummary {
    pub slot: usize,
    pub state: u8,
    pub rx_mcs: Option<u8>,
    pub pair_state: bool,
    pub peer_mac_bytes: Option<[u8; BB_MAC_LEN]>,
    pub peer_mac_hex: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbGetStatusSummary {
    pub role: u8,
    pub mode: u8,
    pub sync_mode: u8,
    pub sync_master: u8,
    pub cfg_sbmp: u8,
    pub rt_sbmp: u8,
    pub mac_bytes: [u8; BB_MAC_LEN],
    pub mac_hex: String,
    pub frequency_khz: Option<u32>,
    pub bandwidth: Option<u8>,
    pub tx_mcs: Option<u8>,
    pub rx_mcs: Option<u8>,
    pub link_state: Option<u8>,
    pub pair_state: Option<bool>,
    pub peer_mac_bytes: Option<[u8; BB_MAC_LEN]>,
    pub peer_mac_hex: Option<String>,
    pub links: Vec<BbLinkStatusSummary>,
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

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct bb_mac_t {
    pub addr: [u8; BB_MAC_LEN],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct bb_phy_status_t {
    pub mcs: u8,
    pub rf_mode: u8,
    pub tintlv_enable: u8,
    pub tintlv_num: u8,
    pub tintlv_len: u8,
    pub bandwidth: u8,
    pub freq_khz: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct bb_link_status_t {
    pub state: u8,
    pub rx_mcs_pair_state: u8,
    pub peer_mac: bb_mac_t,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct bb_user_status_t {
    pub tx_status: bb_phy_status_t,
    pub rx_status: bb_phy_status_t,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct bb_get_status_in_t {
    pub user_bmp: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct bb_get_status_out_t {
    pub role: u8,
    pub mode: u8,
    pub sync_mode: u8,
    pub sync_master: u8,
    pub cfg_sbmp: u8,
    pub rt_sbmp: u8,
    pub mac: bb_mac_t,
    pub user_status: [bb_user_status_t; BB_DATA_USER_MAX],
    pub link_status: [bb_link_status_t; BB_SLOT_MAX],
}

impl Default for bb_get_status_out_t {
    fn default() -> Self {
        Self {
            role: 0,
            mode: 0,
            sync_mode: 0,
            sync_master: 0,
            cfg_sbmp: 0,
            rt_sbmp: 0,
            mac: bb_mac_t::default(),
            user_status: [bb_user_status_t::default(); BB_DATA_USER_MAX],
            link_status: [bb_link_status_t::default(); BB_SLOT_MAX],
        }
    }
}

type BbInitFn = unsafe extern "C" fn(handle: *mut bb_dev_handle_t) -> c_int;
type BbDeinitFn = unsafe extern "C" fn(handle: *mut bb_dev_handle_t) -> c_int;
type BbStartFn = unsafe extern "C" fn(handle: *mut bb_dev_handle_t) -> c_int;
type BbStopFn = unsafe extern "C" fn(handle: *mut bb_dev_handle_t) -> c_int;
type BbIoctlFn = unsafe extern "C" fn(dev: *mut bb_dev_handle_t, request: c_uint, input: *const c_void, output: *mut c_void) -> c_int;
type BbGetDaemonVerFn = unsafe extern "C" fn(phost: *mut bb_host_t) -> *const i8;
type BbHostConnectFn = unsafe extern "C" fn(phost: *mut *mut bb_host_t, addr: *const i8, port: c_int) -> c_int;
type BbHostDisconnectFn = unsafe extern "C" fn(phost: *mut bb_host_t) -> c_int;
type BbDevGetListFn = unsafe extern "C" fn(phost: *mut bb_host_t, plist: *mut *mut *mut bb_dev_t) -> c_int;
type BbDevOpenFn = unsafe extern "C" fn(devs: *mut bb_dev_t) -> *mut bb_dev_handle_t;
type BbDevCloseFn = unsafe extern "C" fn(handle: *mut bb_dev_handle_t) -> c_int;
type BbDevFreeListFn = unsafe extern "C" fn(plist: *mut *mut bb_dev_t) -> c_int;
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
    bb_ioctl: BbIoctlFn,
    bb_get_daemon_ver: BbGetDaemonVerFn,
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
pub const BB_SLOT_MAX: usize = 8;
pub const BB_DATA_USER_MAX: usize = 10;
pub const BB_ALL_DATA_USER_BMP: u16 = ((1_u32 << BB_DATA_USER_MAX) - 1) as u16;

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
            let bb_ioctl = load_symbol::<BbIoctlFn>(&library, b"bb_ioctl\0")?;
            let bb_get_daemon_ver = load_symbol::<BbGetDaemonVerFn>(&library, b"bb_get_daemon_ver\0")?;
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
                bb_ioctl,
                bb_get_daemon_ver,
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

pub fn get_daemon_version(host: *mut bb_host_t) -> Result<String, String> {
    let sdk = sdk()?;

    if host.is_null() {
        return Err("bb_get_daemon_ver requires a valid host handle".to_string());
    }

    unsafe {
        let version_ptr = (sdk.bb_get_daemon_ver)(host);

        if version_ptr.is_null() {
            Err("bb_get_daemon_ver returned null".to_string())
        } else {
            Ok(CStr::from_ptr(version_ptr).to_string_lossy().into_owned())
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

pub fn open_first_device_on_host(host: *mut bb_host_t) -> Result<(*mut bb_dev_handle_t, i32), String> {
    let sdk = sdk()?;

    unsafe {
        let mut device_list: *mut *mut bb_dev_t = std::ptr::null_mut();
        let device_count = (sdk.bb_dev_getlist)(host, &mut device_list);

        if device_count < 0 {
            return Err(format!("bb_dev_getlist failed with code: {}", device_count));
        }

        if device_count == 0 || device_list.is_null() {
            return Err("No baseband devices detected".to_string());
        }

        let first_device = *device_list;
        if first_device.is_null() {
            let _ = (sdk.bb_dev_freelist)(device_list);
            return Err("bb_dev_getlist returned a null first device entry".to_string());
        }

        let handle = (sdk.bb_dev_open)(first_device);
        let _ = (sdk.bb_dev_freelist)(device_list);

        if handle.is_null() {
            Err("bb_dev_open returned null handle".to_string())
        } else {
            Ok((handle, device_count))
        }
    }
}

pub fn get_status(handle: *mut bb_dev_handle_t, user_bmp: u16) -> Result<BbGetStatusSummary, String> {
    let sdk = sdk()?;
    let input = bb_get_status_in_t { user_bmp };
    let mut output = bb_get_status_out_t::default();

    unsafe {
        match (sdk.bb_ioctl)(
            handle,
            BB_GET_STATUS as c_uint,
            &input as *const bb_get_status_in_t as *const c_void,
            &mut output as *mut bb_get_status_out_t as *mut c_void,
        ) {
            0 => {
                let active_user = output
                    .user_status
                    .iter()
                    .find(|user| user.tx_status.freq_khz != 0 || user.rx_status.freq_khz != 0);
                let links = output
                    .link_status
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, link)| {
                        summarize_link_status(slot, link, output.cfg_sbmp, output.rt_sbmp)
                    })
                    .collect::<Vec<_>>();
                let active_link = links.first();

                let frequency_khz = active_user.and_then(preferred_frequency_khz);
                let bandwidth = active_user.and_then(preferred_bandwidth);
                let tx_mcs = active_user.map(|user| user.tx_status.mcs);
                let rx_mcs = active_link.and_then(|link| link.rx_mcs);
                let link_state = active_link.map(|link| link.state);
                let pair_state = active_link.map(|link| link.pair_state);
                let peer_mac_bytes = active_link.and_then(|link| link.peer_mac_bytes);
                let peer_mac_hex = active_link.and_then(|link| link.peer_mac_hex.clone());

                Ok(BbGetStatusSummary {
                    role: output.role,
                    mode: output.mode,
                    sync_mode: output.sync_mode,
                    sync_master: output.sync_master,
                    cfg_sbmp: output.cfg_sbmp,
                    rt_sbmp: output.rt_sbmp,
                    mac_bytes: output.mac.addr,
                    mac_hex: format_bb_mac(&output.mac),
                    frequency_khz,
                    bandwidth,
                    tx_mcs,
                    rx_mcs,
                    link_state,
                    pair_state,
                    peer_mac_bytes,
                    peer_mac_hex,
                    links,
                })
            }
            e => Err(format!("bb_ioctl(BB_GET_STATUS) failed with code: {}", e)),
        }
    }
}

fn format_bb_mac(mac: &bb_mac_t) -> String {
    mac.addr
        .iter()
        .map(|value| format!("{:02X}", value))
        .collect::<Vec<_>>()
        .join(":")
}

fn preferred_frequency_khz(user: &bb_user_status_t) -> Option<u32> {
    if user.tx_status.freq_khz != 0 {
        Some(user.tx_status.freq_khz)
    } else if user.rx_status.freq_khz != 0 {
        Some(user.rx_status.freq_khz)
    } else {
        None
    }
}

fn preferred_bandwidth(user: &bb_user_status_t) -> Option<u8> {
    if user.tx_status.freq_khz != 0 {
        Some(user.tx_status.bandwidth)
    } else if user.rx_status.freq_khz != 0 {
        Some(user.rx_status.bandwidth)
    } else {
        None
    }
}

fn decode_rx_mcs(packed: u8) -> u8 {
    packed & 0x1F
}

fn decode_pair_state(packed: u8) -> bool {
    ((packed >> 5) & 0x01) != 0
}

fn summarize_link_status(
    slot: usize,
    link: &bb_link_status_t,
    cfg_sbmp: u8,
    rt_sbmp: u8,
) -> Option<BbLinkStatusSummary> {
    let pair_state = decode_pair_state(link.rx_mcs_pair_state);
    let peer_mac_bytes = if is_zero_mac(&link.peer_mac.addr) {
        None
    } else {
        Some(link.peer_mac.addr)
    };
    let slot_mask = 1_u8.checked_shl(slot as u32).unwrap_or(0);
    let slot_declared = (cfg_sbmp & slot_mask) != 0 || (rt_sbmp & slot_mask) != 0;

    if link.state == 0 && !pair_state && peer_mac_bytes.is_none() && !slot_declared {
        return None;
    }

    let peer_mac_hex = peer_mac_bytes.map(|addr| format_bb_mac(&bb_mac_t { addr }));

    Some(BbLinkStatusSummary {
        slot,
        state: link.state,
        rx_mcs: if link.state == 0 {
            None
        } else {
            Some(decode_rx_mcs(link.rx_mcs_pair_state))
        },
        pair_state,
        peer_mac_bytes,
        peer_mac_hex,
    })
}

fn is_zero_mac(mac: &[u8; BB_MAC_LEN]) -> bool {
    mac.iter().all(|value| *value == 0)
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
