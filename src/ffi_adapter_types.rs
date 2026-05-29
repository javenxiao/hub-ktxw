use super::BB_MAC_LEN;

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
pub struct BbQualitySummary {
    pub snr_linear: u16,
    pub snr_db: Option<i32>,
    pub ldpc_err: u16,
    pub ldpc_num: u16,
    pub gain_a: u8,
    pub gain_b: u8,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbLinkStatusSummary {
    pub slot: usize,
    pub state: u8,
    pub rx_mcs: Option<u8>,
    pub pair_state: bool,
    pub candidate_macs: Vec<String>,
    pub snr_db: Option<i32>,
    pub ldpc_err: Option<i32>,
    pub ldpc_num: Option<i32>,
    pub signal_main: Option<i32>,
    pub signal_aux: Option<i32>,
    pub peer_mac_bytes: Option<[u8; BB_MAC_LEN]>,
    pub peer_mac_hex: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbPhyStatusSummary {
    pub mcs: u8,
    pub rf_mode: u8,
    pub tintlv_enable: u8,
    pub tintlv_num: u8,
    pub tintlv_len: u8,
    pub bandwidth: u8,
    pub freq_khz: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbGetStatusSummary {
    pub role: u8,
    pub mode: u8,
    pub sync_mode: u8,
    pub sync_master: u8,
    pub cfg_sbmp: u8,
    pub rt_sbmp: u8,
    pub active_user: Option<u8>,
    pub detected_active_user: Option<u8>,
    pub tx_status: Option<BbPhyStatusSummary>,
    pub rx_status: Option<BbPhyStatusSummary>,
    pub slot_tx_status: Option<BbPhyStatusSummary>,
    pub slot_rx_status: Option<BbPhyStatusSummary>,
    pub br_tx_status: Option<BbPhyStatusSummary>,
    pub br_rx_status: Option<BbPhyStatusSummary>,
    pub mac_bytes: [u8; BB_MAC_LEN],
    pub mac_hex: String,
    pub frequency_khz: Option<u32>,
    pub bandwidth: Option<u8>,
    pub tx_mcs: Option<u8>,
    pub rx_mcs: Option<u8>,
    pub link_state: Option<u8>,
    pub pair_state: Option<bool>,
    pub snr_db: Option<i32>,
    pub br_snr_db: Option<i32>,
    pub ldpc_err: Option<i32>,
    pub ldpc_num: Option<i32>,
    pub signal_main: Option<i32>,
    pub signal_aux: Option<i32>,
    pub br_signal_main: Option<i32>,
    pub br_signal_aux: Option<i32>,
    pub peer_mac_bytes: Option<[u8; BB_MAC_LEN]>,
    pub peer_mac_hex: Option<String>,
    pub links: Vec<BbLinkStatusSummary>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbSystemInfoSummary {
    pub uptime: u64,
    pub compile_time: String,
    pub software_version: String,
    pub hardware_version: String,
    pub firmware_version: String,
    pub running_system: Option<String>,
    pub boot_reason: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbBandInfoSummary {
    pub band_auto: bool,
    pub work_band: u8,
    pub selection_bitmap: Option<u8>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbDiscoveredDeviceSummary {
    pub mac_address: String,
    pub role: Option<u8>,
    pub role_label: String,
    pub sync_mode: Option<u8>,
    pub sync_master: Option<u8>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbSetBandModeSummary {
    pub auto_mode: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbSetBandSummary {
    pub target_band: u8,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbChannelEntrySummary {
    pub index: u8,
    pub frequency_khz: u32,
    pub power_dbm: i32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbChannelInfoSummary {
    pub chan_num: u8,
    pub auto_mode: bool,
    pub work_chan: u8,
    pub work_frequency_khz: Option<u32>,
    pub channels: Vec<BbChannelEntrySummary>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbMcsValueSummary {
    pub slot: u8,
    pub dir: u8,
    pub mcs: u8,
    pub throughput_kbps: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbMcsModeSummary {
    pub slot: u8,
    pub auto_mode: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbPowerModeSummary {
    pub pwr_mode: u8,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbCurrentPowerSummary {
    pub user: u8,
    pub power_dbm: u8,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbPowerAutoSummary {
    pub enabled: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbMinidbPowerSummary {
    pub pwr_mode: u8,
    pub pwr_init: u8,
    pub pwr_auto: bool,
    pub pwr_min: u8,
    pub pwr_max: u8,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbBandwidthModeSummary {
    pub slot: u8,
    pub auto_mode: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbPlotPointSummary {
    pub snr: i32,
    pub ldpc_err: i32,
    pub ldpc_num: i32,
    pub gain_a: i32,
    pub gain_b: i32,
    pub mcs_rx: i32,
    pub fch_lock: i32,
    pub br_power: i32,
    pub ap_power: i32,
    pub dev_power: i32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BbPlotSnapshotSummary {
    pub user: u8,
    pub sample_count: usize,
    pub snr: Vec<i32>,
    pub ldpc_err: Vec<i32>,
    pub ldpc_num: Vec<i32>,
    pub gain_a: Vec<i32>,
    pub gain_b: Vec<i32>,
    pub mcs_rx: Vec<i32>,
    pub fch_lock: Vec<i32>,
    pub br_power: Vec<i32>,
    pub ap_power: Vec<i32>,
    pub dev_power: Vec<i32>,
}