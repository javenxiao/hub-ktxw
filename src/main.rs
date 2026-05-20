mod ffi;
mod bb_api;

use std::{net::SocketAddr, sync::Arc, time::{Duration, Instant}};

use axum::{
    extract::{
        Multipart,
        Query,
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::{
        header::{CACHE_CONTROL, EXPIRES, PRAGMA},
        HeaderValue,
    },
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Notify, RwLock};
use tower_http::{
    cors::CorsLayer,
    limit::RequestBodyLimitLayer,
    services::ServeDir,
    set_header::SetResponseHeaderLayer,
};
use tracing::{error, info, warn};

use bb_api::{
    BasebandHealthStatus,
    BasebandManager,
    WirelessConfigurationDetails,
    WirelessRuntimeDetails,
};
use ffi::BbGetStatusSummary;

const DEFAULT_RUST_LOG: &str = "info";
const DEFAULT_BB_HOST_ADDR: &str = "127.0.0.1";
const DEFAULT_BB_HOST_PORT: &str = "50000";
const DEFAULT_SERVER_PORT: &str = "8080";
const IMMEDIATE_REBOOT_DELAY_MS: u32 = 2_000;
const REMOTE_RUNTIME_REFRESH_INTERVAL_SECS: u64 = 2;
const LOCAL_RUNTIME_REFRESH_INTERVAL_SECS: u64 = 2;
const MAX_REBOOT_DELAY_SECONDS: u32 = (30 * 24 * 60 * 60) + (24 * 60 * 60) + (60 * 60) + 60;
const SCHEDULED_REBOOT_TRIGGER_DELAY_MS: u32 = 0;
const CHANNEL_EFFECT_RETRY_ATTEMPTS: usize = 8;
const CHANNEL_EFFECT_RETRY_INTERVAL_MS: u64 = 500;

fn set_default_env_var(key: &str, value: &str) {
    let should_set = std::env::var(key)
        .map(|existing| existing.trim().is_empty())
        .unwrap_or(true);

    if should_set {
        std::env::set_var(key, value);
    }
}

fn apply_default_runtime_env() {
    set_default_env_var("RUST_LOG", DEFAULT_RUST_LOG);
    set_default_env_var("BB_HOST_ADDR", DEFAULT_BB_HOST_ADDR);
    set_default_env_var("BB_HOST_PORT", DEFAULT_BB_HOST_PORT);
    set_default_env_var("SERVER_PORT", DEFAULT_SERVER_PORT);
}

#[derive(Debug, Clone, Serialize)]
struct SystemInfo {
    host_name: String,
    product_name: String,
    description: String,
    system_date: String,
    system_uptime: String,
    hardware_version: String,
    software_version: String,
    software_build: String,
    build_date: String,
    build_time: String,
    mac_address: String,
    ip_address: String,
    subnet_mask: String,
    connection_type: String,
    gateway: String,
    lan_mac_address: String,
    lan_ip_address: String,
    lan_subnet_mask: String,
    lan_connection_type: String,
    lan_gateway: String,
    wan_mac_address: String,
    wan_ip_address: String,
    wan_subnet_mask: String,
    wan_connection_type: String,
    wan_gateway: String,
    wan_primary_dns: String,
    wan_secondary_dns: String,
}

#[derive(Debug, Clone, Serialize)]
struct MaintenanceInfoResponse {
    success: bool,
    message: String,
    version: MaintenanceVersionInfo,
    upgrade_supported: bool,
}

#[derive(Debug, Clone, Serialize)]
struct MaintenanceVersionInfo {
    product_name: String,
    host_name: String,
    active_device: String,
    operation_mode: String,
    access_mode: String,
    local_mac_address: String,
    system_uptime: String,
    hardware_version: String,
    firmware_version: String,
    software_version: String,
    software_build: String,
    compile_time: String,
    running_system: String,
    boot_reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct BootDiagnosticsResponse {
    success: bool,
    running_system: String,
    boot_reason: String,
    message: String,
}

#[derive(Debug, Clone, Deserialize)]
struct RebootActionRequest {
    delay_seconds: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
struct RebootActionResponse {
    success: bool,
    message: String,
    delay_seconds: u32,
    reboot_expected: bool,
}

#[derive(Debug, Clone, Serialize)]
struct FirmwareUpgradeActionResponse {
    success: bool,
    message: String,
    file_name: String,
    file_size: usize,
    http_upload_elapsed_ms: u64,
    bytes_written: usize,
    chunk_count: usize,
    crc32: String,
    reboot_expected: bool,
}

#[derive(Debug, Clone, Serialize)]
struct BasebandTestResponse {
    available: bool,
    socket_initialized: bool,
    bytes_sent: Option<usize>,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
struct BasebandLinkExerciseResponse {
    available: bool,
    success: bool,
    message: String,
    initial_link_state: String,
    final_link_state: String,
    states_seen: Vec<String>,
    final_peer_mac: Option<String>,
    socket_initialized: bool,
    total_bytes_sent: usize,
    total_send_attempts: usize,
}

#[derive(Debug, Clone, Serialize)]
struct WirelessRuntimeResponse {
    available: bool,
    message: String,
    current: Option<WirelessRuntimeView>,
    available_devices: Vec<WirelessDeviceOption>,
}

#[derive(Debug, Clone, Deserialize)]
struct WirelessConfigurationQuery {
    mode: Option<u8>,
}

#[derive(Debug, Clone, Serialize)]
struct WirelessConfigurationPowerView {
    pwr_mode: u8,
    auto_mode: bool,
    init_dbm: u8,
    min_dbm: u8,
    max_dbm: u8,
}

#[derive(Debug, Clone, Serialize)]
struct WirelessConfigurationMinidbView {
    role: Option<u8>,
    band_bitmap: Option<u8>,
    local_mac: Option<String>,
    ap_mac: Option<String>,
    slot_macs: Vec<Option<String>>,
    power: Option<WirelessConfigurationPowerView>,
}

#[derive(Debug, Clone, Serialize)]
struct WirelessConfigurationView {
    mode: u8,
    config_text: String,
    minidb: WirelessConfigurationMinidbView,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct WirelessConfigurationResponse {
    success: bool,
    message: String,
    current: Option<WirelessConfigurationView>,
}

#[derive(Debug, Clone, Serialize)]
struct WirelessConfiguredBandView {
    bitmap: Option<u8>,
    label: String,
    auto: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
struct WirelessLiveRfView {
    band_code: Option<u8>,
    band: String,
    channel_auto: Option<bool>,
    channel_count: Option<u8>,
    channel_index: Option<u8>,
    channel_frequency: String,
}

#[derive(Debug, Clone, Serialize)]
struct WirelessRuntimeView {
    local_mac_address: String,
    operation_mode: String,
    dev_pair_target_mac: Option<String>,
    ap_pair_target_macs: Vec<Option<String>>,
    available_devices: Vec<WirelessDeviceOption>,
    selected_signal_user: Option<u8>,
    detected_signal_user: Option<u8>,
    compatibility_mode: String,
    configured_band: WirelessConfiguredBandView,
    live_rf: WirelessLiveRfView,
    work_band_code: Option<u8>,
    band_bitmap: Option<u8>,
    bandwidth_code: Option<u8>,
    bandwidth: String,
    frequency_khz: Option<u32>,
    frequency: String,
    system_uptime: String,
    compile_time: String,
    software_version: String,
    hardware_version: String,
    firmware_version: String,
    running_system: String,
    boot_reason: String,
    band_auto: Option<bool>,
    work_band: String,
    channel_auto: Option<bool>,
    channel_count: Option<u8>,
    work_channel_index: Option<u8>,
    work_channel_frequency: String,
    channels: Vec<WirelessChannelOption>,
    bandwidth_auto: Option<bool>,
    current_slot: Option<u8>,
    current_mcs_direction: String,
    current_mcs_auto: Option<bool>,
    current_mcs_value: Option<u8>,
    current_mcs_label: String,
    current_mcs_throughput_kbps: Option<u32>,
    current_power_user: Option<u8>,
    current_power_mode: String,
    current_power_auto: Option<bool>,
    current_power_dbm: Option<u8>,
    br_power_dbm: Option<u8>,
    ap_power_dbm: Option<u8>,
    dev_power_dbm: Option<u8>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct WirelessChannelOption {
    index: u8,
    frequency: String,
    power_dbm: i32,
}

#[derive(Debug, Clone, Serialize)]
struct WirelessDeviceOption {
    role: String,
    mac_address: String,
    label: String,
    selected: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct WirelessSettingRequest {
    action: String,
    auto_mode: Option<bool>,
    band_bitmap: Option<u8>,
    device_mac: Option<String>,
    pair_start: Option<bool>,
    pair_target_mac: Option<String>,
    slot: Option<u8>,
    user: Option<u8>,
    target_band: Option<u8>,
    direction: Option<String>,
    channel_index: Option<u8>,
    mcs: Option<u8>,
    power_dbm: Option<u8>,
    bandwidth: Option<u8>,
    power_mode: Option<String>,
    role: Option<u8>,
}

#[derive(Debug, Clone, Serialize)]
struct WirelessSettingResponse {
    success: bool,
    message: String,
    current: Option<WirelessRuntimeView>,
}

#[derive(Debug, Clone, Deserialize)]
struct WirelessConfigurationActionRequest {
    action: String,
    mode: Option<u8>,
    config_text: Option<String>,
    role: Option<u8>,
    band_bitmap: Option<u8>,
    slot: Option<u8>,
    mac_address: Option<String>,
    power_auto: Option<bool>,
    power_init: Option<u8>,
    power_min: Option<u8>,
    power_max: Option<u8>,
}

#[derive(Debug, Clone, Deserialize)]
struct PairCandidatesQuery {
    slot: u8,
}

#[derive(Debug, Clone, Serialize)]
struct PairCandidatesResponse {
    success: bool,
    slot: u8,
    candidates: Vec<String>,
    message: String,
}

#[derive(Debug, Clone, Deserialize)]
struct PlotRefreshSettingsRequest {
    update_interval_ms: u64,
    sample_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct PlotRefreshSettingsResponse {
    success: bool,
    update_interval_ms: u64,
    sample_count: usize,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
struct WirelessSnapshot {
    sequence: u64,
    general: GeneralStatus,
    connections: Vec<ConnectionStatus>,
    chart: RssiChart,
}

#[derive(Debug, Clone, Serialize)]
struct GeneralStatus {
    role: String,
    mac_address: String,
    master_slave_mode: String,
    networking_mode: String,
    band_mode: String,
}

#[derive(Debug, Clone, Serialize)]
struct ConnectionStatus {
    link_slot: String,
    slot_type: String,
    direction: String,
    duration: String,
    frequency: String,
    bandwidth: String,
    mcs: String,
    antenna_mode: String,
    throughput: String,
    link_state: String,
    pair_state: String,
    pairing_active: bool,
    mac_address: String,
    snr_db: i32,
    signal_level: u8,
    rssi_main_history: Vec<i32>,
    rssi_aux_history: Vec<i32>,
}

#[derive(Debug, Clone, Serialize)]
struct ChartSeries {
    key: String,
    label: String,
    unit: String,
    current_value: Option<i32>,
    min_value: Option<i32>,
    max_value: Option<i32>,
    points: Vec<i32>,
}

#[derive(Debug, Clone, Serialize)]
struct RssiChart {
    title: String,
    target_mac_address: String,
    history_context_key: String,
    series: Vec<ChartSeries>,
}

const CONNECTION_HISTORY_POINTS: usize = 18;
const DEFAULT_AP_PLOT_SAMPLE_POINTS: usize = 200;
const RSSI_UNAVAILABLE_DBM: i32 = -127;
const SNR_UNAVAILABLE_DB: i32 = -1;
const REBOOT_REFRESH_QUIET_WINDOW_SECS: u64 = 15;
const REBOOT_RUNTIME_RECOVERY_PROBE_DELAY_SECS: u64 = 3;
const REBOOT_RUNTIME_UNAVAILABLE_MESSAGE: &str = "Device reboot in progress. Waiting for reconnect.";
const REBOOT_PAIR_RESUME_WINDOW_SECS: u64 = 120;
const REBOOT_PAIR_RESUME_RETRY_INTERVAL_SECS: u64 = 5;

struct PendingRebootPairResume {
    target_mac: String,
    expires_at: Instant,
    last_attempt_at: Option<Instant>,
}

struct AppState {
    snapshot: RwLock<WirelessSnapshot>,
    wireless_runtime: RwLock<WirelessRuntimeResponse>,
    expected_reboot_until: RwLock<Option<Instant>>,
    pending_reboot_pair_resume: RwLock<Option<PendingRebootPairResume>>,
    plot_refresh_interval_ms: RwLock<u64>,
    plot_sample_count: RwLock<usize>,
    plot_refresh_interval_notify: Notify,
    tx: broadcast::Sender<WirelessSnapshot>,
    baseband: Option<Arc<BasebandManager>>,
    baseband_health: BasebandHealthStatus,
}

impl AppState {
    fn new(
        initial: WirelessSnapshot,
        initial_runtime: WirelessRuntimeResponse,
        initial_plot_refresh_interval_ms: u64,
        initial_plot_sample_count: usize,
        baseband: Option<Arc<BasebandManager>>,
        baseband_health: BasebandHealthStatus,
    ) -> Self {
        let (tx, _) = broadcast::channel(128);
        Self {
            snapshot: RwLock::new(initial),
            wireless_runtime: RwLock::new(initial_runtime),
            expected_reboot_until: RwLock::new(None),
            pending_reboot_pair_resume: RwLock::new(None),
            plot_refresh_interval_ms: RwLock::new(clamp_plot_refresh_interval_ms(initial_plot_refresh_interval_ms)),
            plot_sample_count: RwLock::new(clamp_plot_sample_count(initial_plot_sample_count)),
            plot_refresh_interval_notify: Notify::new(),
            tx,
            baseband,
            baseband_health,
        }
    }

    async fn begin_expected_reboot_window(&self, duration: Duration) {
        {
            let mut guard = self.expected_reboot_until.write().await;
            *guard = Some(Instant::now() + duration);
        }

        {
            let mut guard = self.wireless_runtime.write().await;
            *guard = runtime_unavailable_response(REBOOT_RUNTIME_UNAVAILABLE_MESSAGE.to_string());
        }
    }

    async fn clear_expected_reboot_window(&self) {
        let mut guard = self.expected_reboot_until.write().await;
        *guard = None;
    }

    async fn stage_reboot_pair_resume(&self, target_mac: &str) {
        let normalized_target = normalize_device_mac(target_mac);
        let mut guard = self.pending_reboot_pair_resume.write().await;

        if normalized_target.is_empty() {
            *guard = None;
            return;
        }

        *guard = Some(PendingRebootPairResume {
            target_mac: normalized_target,
            expires_at: Instant::now() + Duration::from_secs(REBOOT_PAIR_RESUME_WINDOW_SECS),
            last_attempt_at: None,
        });
    }

    async fn clear_reboot_pair_resume(&self) {
        let mut guard = self.pending_reboot_pair_resume.write().await;
        *guard = None;
    }

    async fn clear_reboot_pair_resume_if_matches(&self, current_mac: &str) {
        let normalized_current = normalize_device_mac(current_mac);
        let mut guard = self.pending_reboot_pair_resume.write().await;
        if guard
            .as_ref()
            .map(|pending| pending.target_mac.as_str())
            == Some(normalized_current.as_str())
        {
            *guard = None;
        }
    }

    async fn reboot_pair_resume_pending_for(&self, current_mac: &str) -> bool {
        let normalized_current = normalize_device_mac(current_mac);
        let now = Instant::now();
        let mut guard = self.pending_reboot_pair_resume.write().await;

        if guard
            .as_ref()
            .map(|pending| pending.expires_at <= now)
            .unwrap_or(false)
        {
            *guard = None;
            return false;
        }

        guard
            .as_ref()
            .map(|pending| pending.target_mac == normalized_current)
            .unwrap_or(false)
    }

    async fn mark_reboot_pair_resume_attempt(&self, current_mac: &str) -> bool {
        let normalized_current = normalize_device_mac(current_mac);
        let now = Instant::now();
        let mut guard = self.pending_reboot_pair_resume.write().await;

        let Some(pending) = guard.as_mut() else {
            return false;
        };

        if pending.expires_at <= now {
            *guard = None;
            return false;
        }

        if pending.target_mac != normalized_current {
            return false;
        }

        if pending
            .last_attempt_at
            .map(|last| now.saturating_duration_since(last) < Duration::from_secs(REBOOT_PAIR_RESUME_RETRY_INTERVAL_SECS))
            .unwrap_or(false)
        {
            return false;
        }

        pending.last_attempt_at = Some(now);
        true
    }

    async fn expected_reboot_window_active(&self) -> bool {
        let deadline = {
            let guard = self.expected_reboot_until.read().await;
            *guard
        };

        match deadline {
            Some(until) if until > Instant::now() => true,
            Some(_) => {
                self.clear_expected_reboot_window().await;
                false
            }
            None => false,
        }
    }

    async fn expected_reboot_recovery_probe_due(&self) -> bool {
        let deadline = {
            let guard = self.expected_reboot_until.read().await;
            *guard
        };

        match deadline {
            Some(until) if until > Instant::now() => {
                let quiet_window = Duration::from_secs(REBOOT_REFRESH_QUIET_WINDOW_SECS);
                let probe_delay = Duration::from_secs(REBOOT_RUNTIME_RECOVERY_PROBE_DELAY_SECS);
                until.saturating_duration_since(Instant::now())
                    <= quiet_window.saturating_sub(probe_delay)
            }
            Some(_) => {
                self.clear_expected_reboot_window().await;
                true
            }
            None => false,
        }
    }
}

fn clamp_plot_refresh_interval_ms(value: u64) -> u64 {
    value.clamp(100, 10_000)
}

fn clamp_plot_sample_count(value: usize) -> usize {
    value.max(10)
}

fn default_plot_refresh_interval_ms(baseband_health: &BasebandHealthStatus) -> u64 {
    let _ = baseband_health;
    100
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    apply_default_runtime_env();

    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("========== RSHTML Server Starting ==========");

    // 初始化基带 API
    let (baseband, baseband_health) = match BasebandManager::initialize_with_health() {
        (Some(bb), health) => {
            let mut health = health;

            if health.effective_mode == "hardware-remote-bb-host" {
                if health.host.connected {
                    info!("[ok] Remote bb_host session initialized successfully");
                } else {
                    warn!(
                        "Remote bb_host manager initialized without an active daemon session: {}",
                        health.host.message
                    );
                    info!("Remote bb_host auto-reconnect remains enabled; waiting for daemon availability");
                }

                info!("[ok] Baseband API initialized successfully");
                health.socket_init.message = "Skipped in remote bb_host mode".to_string();
            } else {
                info!("[ok] Baseband API initialized successfully");
                let socket_result = bb.initialize_socket(0);
                health.record_socket_init(socket_result.clone(), 0);

                if let Err(e) = socket_result {
                    warn!("Failed to initialize socket: {}", e);
                } else {
                    info!("[ok] Socket 0 initialized for data communication");
                }
            }

            (Some(Arc::new(bb)), health)
        }
        (None, health) => {
            let failure_message = health.primary_failure_message();

            warn!("Failed to initialize baseband API: {}", failure_message);
            warn!("Using simulator mode without hardware communication");
            (None, health)
        }
    };

    let initial_runtime = match baseband.as_ref() {
        Some(baseband) => match baseband.get_wireless_runtime_details() {
            Ok(details) => runtime_response_from_details(&details),
            Err(err) => runtime_unavailable_response(format_runtime_fetch_error_message(&err)),
        },
        None => runtime_unavailable_response(
            "Baseband SDK not available; runtime controls require real hardware mode".to_string(),
        ),
    };
    let initial = match (baseband.as_ref(), baseband_health.runtime.status_snapshot.as_ref()) {
        (Some(_), Some(status)) => {
            build_hardware_snapshot(
                0,
                status,
                None,
                DEFAULT_AP_PLOT_SAMPLE_POINTS,
                initial_runtime.current.as_ref(),
                None,
            )
        }
        _ => build_simulated_snapshot(0, DEFAULT_AP_PLOT_SAMPLE_POINTS),
    };
    let initial_plot_refresh_interval_ms = default_plot_refresh_interval_ms(&baseband_health);
    let initial_plot_sample_count = DEFAULT_AP_PLOT_SAMPLE_POINTS;
    ffi::set_plot_history_limit(initial_plot_sample_count);
    let state = Arc::new(AppState::new(
        initial,
        initial_runtime,
        initial_plot_refresh_interval_ms,
        initial_plot_sample_count,
        baseband.clone(),
        baseband_health,
    ));

    spawn_data_feeder(state.clone());
    spawn_runtime_feeder(state.clone());

    let app = Router::new()
        .route("/api/wireless/status", get(get_wireless_status))
        .route("/api/wireless/runtime", get(get_wireless_runtime))
        .route("/api/wireless/configuration", get(get_wireless_configuration))
        .route("/api/wireless/pair-candidates", get(get_pair_candidates))
        .route("/api/wireless/runtime/apply", post(apply_wireless_setting))
        .route(
            "/api/wireless/configuration/action",
            post(apply_wireless_configuration_action),
        )
        .route(
            "/api/wireless/plot/settings",
            get(get_plot_refresh_settings).post(apply_plot_refresh_settings),
        )
        .route("/api/system/info", get(get_system_info))
        .route("/api/system/reboot", post(request_system_reboot))
        .route("/api/system/maintenance", get(get_maintenance_info))
        .route("/api/system/maintenance/boot-diagnostics", get(get_boot_diagnostics))
        .route("/api/system/maintenance/upgrade", post(apply_firmware_upgrade))
        .route("/api/system/maintenance/upgrade-progress", get(get_upgrade_progress))
        .route("/api/baseband/health", get(get_baseband_health))
        .route("/api/baseband/test", get(test_baseband_communication))
        .route("/api/baseband/link/exercise", post(exercise_baseband_link))
        .route("/ws", get(ws_handler))
        .nest_service("/", ServeDir::new("static").append_index_html_on_directories(true))
        .layer(SetResponseHeaderLayer::overriding(
            CACHE_CONTROL,
            HeaderValue::from_static("no-store, no-cache, must-revalidate"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            PRAGMA,
            HeaderValue::from_static("no-cache"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            EXPIRES,
            HeaderValue::from_static("0"),
        ))
        .layer(CorsLayer::permissive())
        .layer(RequestBodyLimitLayer::new(200 * 1024 * 1024))
        .with_state(state);

    let server_port = std::env::var("SERVER_PORT")
        .ok()
        .and_then(|value| value.trim().parse::<u16>().ok())
        .unwrap_or(8080);
    let bind_addr = SocketAddr::from(([0, 0, 0, 0], server_port));
    let local_browser_addr = SocketAddr::from(([127, 0, 0, 1], server_port));
    info!("wireless status server listening on {}", bind_addr);
    info!("browser access on this PC: http://{}", local_browser_addr);
    info!("for remote access, use this PC's LAN IP with port {}", server_port);
    info!("========== Server Ready ==========\n");

    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn get_wireless_status(State(state): State<Arc<AppState>>) -> Json<WirelessSnapshot> {
    Json(state.snapshot.read().await.clone())
}

async fn get_wireless_runtime(State(state): State<Arc<AppState>>) -> Json<WirelessRuntimeResponse> {
    let cached = state.wireless_runtime.read().await.clone();

    if !cached.available
        && cached.current.is_none()
        && cached.message == REBOOT_RUNTIME_UNAVAILABLE_MESSAGE
    {
        if let Some(baseband) = state.baseband.as_ref() {
            let reboot_window_active = state.expected_reboot_window_active().await;
            let should_probe = if reboot_window_active {
                state.expected_reboot_recovery_probe_due().await
            } else {
                true
            };

            if should_probe {
                match baseband.get_wireless_runtime_details() {
                    Ok(details) => {
                        let response = runtime_response_from_details(&details);
                        {
                            let mut guard = state.wireless_runtime.write().await;
                            *guard = response;
                        }
                        state.clear_expected_reboot_window().await;
                        refresh_snapshot_from_baseband(&state, baseband).await;
                    }
                    Err(err) if !reboot_window_active => {
                        let response = runtime_unavailable_response(
                            format_runtime_fetch_error_message(&err),
                        );
                        let mut guard = state.wireless_runtime.write().await;
                        *guard = response;
                    }
                    Err(_) => {
                        if reboot_window_active {
                            if let Ok(devices) = baseband.get_detected_remote_devices() {
                                if !devices.is_empty() {
                                    let response = runtime_unavailable_response_with_devices(
                                        REBOOT_RUNTIME_UNAVAILABLE_MESSAGE.to_string(),
                                        build_wireless_device_options(None, None, None, None, None, &devices),
                                    );
                                    let mut guard = state.wireless_runtime.write().await;
                                    *guard = response;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Json(state.wireless_runtime.read().await.clone())
}

async fn get_wireless_configuration(
    State(state): State<Arc<AppState>>,
    Query(query): Query<WirelessConfigurationQuery>,
) -> Json<WirelessConfigurationResponse> {
    if state.expected_reboot_window_active().await {
        return Json(WirelessConfigurationResponse {
            success: false,
            message: "Device reboot in progress. Configuration will refresh after reconnect.".to_string(),
            current: None,
        });
    }

    let Some(baseband) = state.baseband.as_ref() else {
        return Json(WirelessConfigurationResponse {
            success: false,
            message: "Baseband SDK not available; cannot load configuration in simulator mode".to_string(),
            current: None,
        });
    };

    let mode = query.mode.unwrap_or(0);

    match baseband.get_wireless_configuration_details(mode) {
        Ok(details) => {
            let message = if details.warnings.is_empty() {
                format!("Wireless configuration loaded (mode {})", mode)
            } else {
                format!(
                    "Wireless configuration loaded with {} warning(s)",
                    details.warnings.len()
                )
            };

            Json(WirelessConfigurationResponse {
                success: true,
                message,
                current: Some(build_wireless_configuration_view(&details)),
            })
        }
        Err(err) => Json(WirelessConfigurationResponse {
            success: false,
            message: format!("Failed to load wireless configuration: {}", err),
            current: None,
        }),
    }
}

async fn get_pair_candidates(
    State(state): State<Arc<AppState>>,
    Query(query): Query<PairCandidatesQuery>,
) -> Json<PairCandidatesResponse> {
    let Some(baseband) = state.baseband.as_ref() else {
        return Json(PairCandidatesResponse {
            success: false,
            slot: query.slot,
            candidates: Vec::new(),
            message: "Baseband SDK not available; cannot load pair candidates in simulator mode".to_string(),
        });
    };

    if query.slot >= 8 {
        return Json(PairCandidatesResponse {
            success: false,
            slot: query.slot,
            candidates: Vec::new(),
            message: format!("Unsupported slot '{}'; expected 0-7", query.slot),
        });
    }

    let runtime_role = state
        .wireless_runtime
        .read()
        .await
        .current
        .as_ref()
        .and_then(resolve_runtime_role);

    if runtime_role == Some(1) {
        return Json(PairCandidatesResponse {
            success: true,
            slot: query.slot,
            candidates: Vec::new(),
            message: "DEV pair uses AP MAC and does not expose candidate DEV list".to_string(),
        });
    }

    match baseband.get_pair_candidates(query.slot) {
        Ok(candidates) => Json(PairCandidatesResponse {
            success: true,
            slot: query.slot,
            message: format!("Loaded {} pair candidates for slot {}", candidates.len(), query.slot),
            candidates,
        }),
        Err(err) => Json(PairCandidatesResponse {
            success: false,
            slot: query.slot,
            candidates: Vec::new(),
            message: format!("Failed to load pair candidates for slot {}: {}", query.slot, err),
        }),
    }
}

async fn get_plot_refresh_settings(
    State(state): State<Arc<AppState>>,
) -> Json<PlotRefreshSettingsResponse> {
    let update_interval_ms = clamp_plot_refresh_interval_ms(*state.plot_refresh_interval_ms.read().await);
    let sample_count = clamp_plot_sample_count(*state.plot_sample_count.read().await);
    Json(PlotRefreshSettingsResponse {
        success: true,
        update_interval_ms,
        sample_count,
        message: "AP Plot refresh interval loaded".to_string(),
    })
}

async fn apply_plot_refresh_settings(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PlotRefreshSettingsRequest>,
) -> Json<PlotRefreshSettingsResponse> {
    let update_interval_ms = clamp_plot_refresh_interval_ms(request.update_interval_ms);
    let sample_count = clamp_plot_sample_count(request.sample_count);

    {
        let mut guard = state.plot_refresh_interval_ms.write().await;
        *guard = update_interval_ms;
    }
    {
        let mut guard = state.plot_sample_count.write().await;
        *guard = sample_count;
    }
    ffi::set_plot_history_limit(sample_count);
    state.plot_refresh_interval_notify.notify_waiters();

    Json(PlotRefreshSettingsResponse {
        success: true,
        update_interval_ms,
        sample_count,
        message: format!(
            "AP Plot settings updated: refresh {} ms, samples {}",
            update_interval_ms,
            sample_count
        ),
    })
}

fn action_requires_runtime_context(action: &str, request: &WirelessSettingRequest) -> bool {
    match action {
        "set_pair_mode" | "set_mcs_mode" | "set_mcs" | "set_bandwidth_mode" | "set_bandwidth" => {
            request.slot.is_none()
        }
        "set_power" => request.user.is_none(),
        _ => false,
    }
}

fn resolve_runtime_role(current: &WirelessRuntimeView) -> Option<u8> {
    let operation_mode = current.operation_mode.to_ascii_uppercase();
    if operation_mode.contains("AP") {
        Some(0)
    } else if operation_mode.contains("DEV") {
        Some(1)
    } else {
        None
    }
}

fn async_runtime_control_unsupported_message(
    current: Option<&WirelessRuntimeView>,
    action: &str,
) -> Option<String> {
    let current = current?;
    if !current.operation_mode.to_ascii_uppercase().contains("ASYNC") {
        return None;
    }

    match action {
        "set_bandwidth" | "set_bandwidth_mode" => Some(format!(
            "Bandwidth control is not supported on {}. The current Async runtime keeps bandwidth fixed, so manual bandwidth changes are disabled.",
            current.operation_mode
        )),
        _ => None,
    }
}

async fn apply_wireless_setting(
    State(state): State<Arc<AppState>>,
    Json(request): Json<WirelessSettingRequest>,
) -> Json<WirelessSettingResponse> {
    let Some(baseband) = state.baseband.as_ref() else {
        return Json(WirelessSettingResponse {
            success: false,
            message: "Baseband SDK not available; cannot apply wireless settings in simulator mode".to_string(),
            current: None,
        });
    };

    let baseband = Arc::clone(baseband);
    let cached_current = state.wireless_runtime.read().await.current.clone();
    let current = if action_requires_runtime_context(&request.action, &request) && cached_current.is_none() {
        match baseband.get_wireless_runtime_details() {
            Ok(details) => Some(details),
            Err(err) => {
                return Json(WirelessSettingResponse {
                    success: false,
                    message: format!("Failed to read current wireless runtime before applying setting: {}", err),
                    current: None,
                });
            }
        }
    } else {
        None
    };

    let default_slot = current
        .as_ref()
        .and_then(|details| details.mcs_value.as_ref().map(|value| value.slot))
        .or_else(|| current.as_ref().and_then(|details| details.status.links.first().map(|link| link.slot as u8)))
        .or_else(|| cached_current.as_ref().and_then(|runtime| runtime.current_slot))
        .unwrap_or(0);
    let default_user = current
        .as_ref()
        .and_then(|details| details.current_power.as_ref().map(|value| value.user))
        .or_else(|| current.as_ref().and_then(|details| details.status.active_user))
        .or_else(|| cached_current.as_ref().and_then(|runtime| runtime.current_power_user))
        .unwrap_or(0);
    let runtime_view_for_constraints = cached_current
        .clone()
        .or_else(|| current.as_ref().map(build_wireless_runtime_view));

    if let Some(message) = async_runtime_control_unsupported_message(
        runtime_view_for_constraints.as_ref(),
        &request.action,
    ) {
        return Json(WirelessSettingResponse {
            success: false,
            message,
            current: runtime_view_for_constraints,
        });
    }

    let result = match request.action.as_str() {
        "set_signal_user" => request
            .user
            .ok_or_else(|| "user is required".to_string())
            .and_then(|user| baseband.set_signal_user_preference(user)),
        "select_device" => request
            .device_mac
            .as_deref()
            .ok_or_else(|| "device_mac is required".to_string())
            .and_then(|device_mac| baseband.switch_active_device(device_mac)),
        "set_pair_target" => {
            let slot = request.slot.unwrap_or(default_slot);
            let current_role = cached_current
                .as_ref()
                .and_then(resolve_runtime_role)
                .or_else(|| current.as_ref().map(|details| details.status.role))
                .or_else(|| baseband.get_wireless_runtime_details().ok().map(|details| details.status.role));

            request
                .pair_target_mac
                .as_deref()
                .map(normalize_device_mac)
                .filter(|mac| !mac.is_empty())
                .ok_or_else(|| "pair_target_mac is required".to_string())
                .and_then(|target_mac| {
                    if current_role == Some(ffi::BB_ROLE_DEV) {
                        baseband
                            .set_ap_mac(&target_mac)
                            .and_then(|_| baseband.set_minidb_ap_mac(&target_mac))
                    } else {
                        resolve_pair_slot_bitmap(slot, current_role)?;
                        baseband
                            .set_pair_candidates(slot, std::slice::from_ref(&target_mac))
                            .and_then(|_| baseband.set_minidb_slot_mac(slot, &target_mac))
                    }
                })
        }
        "set_pair_mode" => request
            .pair_start
            .ok_or_else(|| "pair_start is required".to_string())
            .and_then(|pair_start| {
                if pair_start {
                    let runtime = baseband.get_wireless_runtime_details().map_err(|err| {
                        format!(
                            "Failed to verify whether Pair can start before applying setting: {}",
                            err
                        )
                    })?;

                    if let Some(blocking_notice) = pair_blocking_notice_from_runtime(&runtime) {
                        return Err(blocking_notice);
                    }
                }

                let slot = request.slot.unwrap_or(default_slot);
                let current_role = cached_current
                    .as_ref()
                    .and_then(resolve_runtime_role)
                    .or_else(|| current.as_ref().map(|details| details.status.role))
                    .or_else(|| baseband.get_wireless_runtime_details().ok().map(|details| details.status.role));
                let slot_bmp = resolve_pair_slot_bitmap(slot, current_role)?;

                let requested_target = request
                    .pair_target_mac
                    .as_deref()
                    .map(normalize_device_mac)
                    .filter(|mac| !mac.is_empty());

                if current_role == Some(1) {
                    if pair_start {
                        if let Some(target_mac) = requested_target {
                            baseband.set_ap_mac(&target_mac)?;
                            baseband.set_minidb_ap_mac(&target_mac)?;
                        }
                    }

                    return baseband.set_pair_mode(pair_start, slot_bmp);
                }

                if !pair_start {
                    return baseband.set_pair_mode(pair_start, slot_bmp);
                }

                if let Some(target_mac) = requested_target {
                    baseband.set_pair_candidates(slot, std::slice::from_ref(&target_mac))?;
                    baseband.set_minidb_slot_mac(slot, &target_mac)?;
                    return baseband.set_pair_mode(pair_start, slot_bmp);
                }

                baseband.set_pair_mode(pair_start, slot_bmp)
            }),
        "set_channel_mode" => request
            .auto_mode
            .ok_or_else(|| "auto_mode is required".to_string())
            .and_then(|auto_mode| baseband.set_channel_mode(auto_mode)),
        "set_channel" => request
            .channel_index
            .ok_or_else(|| "channel_index is required".to_string())
            .and_then(|chan_index| {
                let dir = parse_direction(request.direction.as_deref().unwrap_or("rx"))?;
                baseband.set_channel(dir, chan_index)
            }),
        "set_mcs_mode" => request
            .auto_mode
            .ok_or_else(|| "auto_mode is required".to_string())
            .and_then(|auto_mode| baseband.set_mcs_mode(request.slot.unwrap_or(default_slot), auto_mode)),
        "set_mcs" => request
            .mcs
            .ok_or_else(|| "mcs is required".to_string())
            .and_then(|mcs| baseband.set_mcs(request.slot.unwrap_or(default_slot), mcs)),
        "set_power_mode" => request
            .power_mode
            .as_deref()
            .ok_or_else(|| "power_mode is required".to_string())
            .and_then(parse_power_mode)
            .and_then(|mode| baseband.set_power_mode(mode)),
        "set_power" => request
            .power_dbm
            .ok_or_else(|| "power_dbm is required".to_string())
            .and_then(|power_dbm| baseband.set_power(request.user.unwrap_or(default_user), power_dbm)),
        "set_power_auto" => request
            .auto_mode
            .ok_or_else(|| "auto_mode is required".to_string())
            .and_then(|enabled| baseband.set_power_auto(enabled)),
        "set_band_mode" => request
            .auto_mode
            .ok_or_else(|| "auto_mode is required".to_string())
            .and_then(|auto_mode| baseband.set_band_mode(auto_mode)),
        "set_band_selection" => request
            .band_bitmap
            .ok_or_else(|| "band_bitmap is required".to_string())
            .and_then(parse_band_bitmap)
            .and_then(|band_bitmap| baseband.set_band_selection(band_bitmap)),
        "set_band" => request
            .target_band
            .ok_or_else(|| "target_band is required".to_string())
            .and_then(parse_band)
            .and_then(|target_band| baseband.set_band(target_band)),
        "set_bandwidth_mode" => request
            .auto_mode
            .ok_or_else(|| "auto_mode is required".to_string())
            .and_then(|auto_mode| baseband.set_bandwidth_mode(request.slot.unwrap_or(default_slot), auto_mode)),
        "set_bandwidth" => request
            .bandwidth
            .ok_or_else(|| "bandwidth is required".to_string())
            .and_then(|bandwidth| {
                let dir = parse_direction(request.direction.as_deref().unwrap_or("rx"))?;
                baseband.set_bandwidth(request.slot.unwrap_or(default_slot), dir, bandwidth)
            }),
        "set_role" => request
            .role
            .ok_or_else(|| "role is required".to_string())
            .and_then(|role| baseband.set_baseband_role(role)),
        _ => Err(format!("Unsupported wireless setting action: {}", request.action)),
    };

    match result {
        Ok(()) => {
            let is_role_switch = request.action == "set_role";

            if request.action == "select_device" {
                state.clear_expected_reboot_window().await;
            }

            let current = if is_role_switch {
                state
                    .begin_expected_reboot_window(Duration::from_secs(REBOOT_REFRESH_QUIET_WINDOW_SECS))
                    .await;
                None
            } else {
                refresh_runtime_until_effect_or_timeout(&state, &baseband, &request).await
            };

            if !is_role_switch {
                if let Err(err) = verify_wireless_setting_effect(&request, current.as_ref()) {
                    return Json(WirelessSettingResponse {
                        success: false,
                        message: err,
                        current,
                    });
                }
            }

            let message = if request.action == "set_role" {
                "Baseband role switch requested; device rebooting to apply the new role".to_string()
            } else {
                format!("Wireless setting action '{}' applied successfully", request.action)
            };

            Json(WirelessSettingResponse {
                success: true,
                message,
                current,
            })
        }
        Err(err) => Json(WirelessSettingResponse {
            success: false,
            message: err,
            current: cached_current.or_else(|| current.as_ref().map(build_wireless_runtime_view)),
        }),
    }
}

async fn apply_wireless_configuration_action(
    State(state): State<Arc<AppState>>,
    Json(request): Json<WirelessConfigurationActionRequest>,
) -> Json<WirelessConfigurationResponse> {
    if state.expected_reboot_window_active().await {
        return Json(WirelessConfigurationResponse {
            success: false,
            message: "Device reboot in progress. Configuration actions are temporarily unavailable.".to_string(),
            current: None,
        });
    }

    let Some(baseband) = state.baseband.as_ref() else {
        return Json(WirelessConfigurationResponse {
            success: false,
            message: "Baseband SDK not available; cannot apply configuration in simulator mode".to_string(),
            current: None,
        });
    };

    let baseband = Arc::clone(baseband);
    let mode = request.mode.unwrap_or(0);

    let result = match request.action.as_str() {
        "save_config" => request
            .config_text
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .ok_or_else(|| "config_text is required".to_string())
            .and_then(|text| baseband.save_configuration_text(text)),
        "clear_flash_config" => baseband.clear_flash_configuration(),
        "clear_minidb_config" => baseband.clear_minidb_configuration(),
        "restore_factory_config" => baseband.restore_factory_configuration(),
        "set_minidb_role" => request
            .role
            .ok_or_else(|| "role is required".to_string())
            .and_then(|role| baseband.set_minidb_role(role)),
        "set_minidb_band" => request
            .band_bitmap
            .ok_or_else(|| "band_bitmap is required".to_string())
            .and_then(parse_band_bitmap)
            .and_then(|band_bitmap| baseband.set_band_selection(band_bitmap)),
        "set_minidb_power" => request
            .power_auto
            .ok_or_else(|| "power_auto is required".to_string())
            .and_then(|power_auto| {
                let power = ffi::bb_phy_pwr_basic_t {
                    pwr_mode: 0,
                    pwr_init: if power_auto {
                        0
                    } else {
                        request
                            .power_init
                            .ok_or_else(|| "power_init is required when power_auto is false".to_string())?
                    },
                    pwr_auto: u8::from(power_auto),
                    pwr_min: if power_auto {
                        request
                            .power_min
                            .ok_or_else(|| "power_min is required when power_auto is true".to_string())?
                    } else {
                        0
                    },
                    pwr_max: if power_auto {
                        request
                            .power_max
                            .ok_or_else(|| "power_max is required when power_auto is true".to_string())?
                    } else {
                        0
                    },
                };

                baseband.set_minidb_power(power)
            }),
        "set_minidb_local_mac" => request
            .mac_address
            .as_deref()
            .map(normalize_device_mac)
            .filter(|mac| !mac.is_empty())
            .ok_or_else(|| "mac_address is required".to_string())
            .and_then(|mac| baseband.set_minidb_local_mac(&mac)),
        "set_minidb_ap_mac" => request
            .mac_address
            .as_deref()
            .map(normalize_device_mac)
            .filter(|mac| !mac.is_empty())
            .ok_or_else(|| "mac_address is required".to_string())
            .and_then(|mac| baseband.set_minidb_ap_mac(&mac)),
        "set_minidb_slot_mac" => request
            .slot
            .ok_or_else(|| "slot is required".to_string())
            .and_then(|slot| {
                if slot >= ffi::BB_SLOT_MAX as u8 {
                    Err(format!("Unsupported slot '{}'; expected 0-7", slot))
                } else {
                    Ok(slot)
                }
            })
            .and_then(|slot| {
                request
                    .mac_address
                    .as_deref()
                    .map(normalize_device_mac)
                    .filter(|mac| !mac.is_empty())
                    .ok_or_else(|| "mac_address is required".to_string())
                    .and_then(|mac| baseband.set_minidb_slot_mac(slot, &mac))
            }),
        other => Err(format!("Unsupported wireless configuration action: {}", other)),
    };

    let current = match baseband.get_wireless_configuration_details(mode) {
        Ok(details) => Some(build_wireless_configuration_view(&details)),
        Err(_) => None,
    };

    match result {
        Ok(()) => {
            refresh_runtime_from_baseband(&state, &baseband).await;
            refresh_snapshot_from_baseband(&state, &baseband).await;

            Json(WirelessConfigurationResponse {
                success: true,
                message: format!("Wireless configuration action '{}' applied successfully", request.action),
                current,
            })
        }
        Err(err) => Json(WirelessConfigurationResponse {
            success: false,
            message: err,
            current,
        }),
    }
}

async fn get_system_info() -> Json<SystemInfo> {
    Json(build_system_info())
}

async fn fetch_boot_diagnostics_response(
    baseband: Option<&Arc<BasebandManager>>,
) -> BootDiagnosticsResponse {
    let Some(baseband) = baseband else {
        return BootDiagnosticsResponse {
            success: false,
            running_system: "Unavailable".to_string(),
            boot_reason: "Unavailable".to_string(),
            message: "Baseband SDK not available; boot diagnostics are disabled in simulator mode".to_string(),
        };
    };

    let baseband = Arc::clone(baseband);
    let diagnostics_task = tokio::task::spawn_blocking(move || baseband.get_boot_diagnostics());

    match tokio::time::timeout(Duration::from_secs(2), diagnostics_task).await {
        Ok(Ok(Ok(diagnostics))) => BootDiagnosticsResponse {
            success: true,
            running_system: diagnostics.running_system,
            boot_reason: diagnostics.boot_reason,
            message: "Boot diagnostics loaded".to_string(),
        },
        Ok(Ok(Err(err))) => BootDiagnosticsResponse {
            success: false,
            running_system: format!("Error: {}", err),
            boot_reason: format!("Error: {}", err),
            message: format!("Failed to load boot diagnostics: {}", err),
        },
        Ok(Err(err)) => BootDiagnosticsResponse {
            success: false,
            running_system: format!("Join error: {}", err),
            boot_reason: format!("Join error: {}", err),
            message: format!("Boot diagnostics task failed: {}", err),
        },
        Err(_) => BootDiagnosticsResponse {
            success: false,
            running_system: "Timeout".to_string(),
            boot_reason: "Timeout".to_string(),
            message: "Boot diagnostics timed out after 2 seconds".to_string(),
        },
    }
}

async fn get_maintenance_info(
    State(state): State<Arc<AppState>>,
) -> Json<MaintenanceInfoResponse> {
    let system_info = build_system_info();
    let current = state.wireless_runtime.read().await.current.clone();
    let mut version = build_maintenance_version_info(state.as_ref(), &system_info, current.as_ref());
    let actions_supported = state.baseband.is_some();
    let reboot_window_active = state.expected_reboot_window_active().await;

    if reboot_window_active {
        version.running_system = "Rebooting".to_string();
        version.boot_reason = "Rebooting".to_string();

        return Json(MaintenanceInfoResponse {
            success: true,
            message: "Device reboot in progress. Maintenance information will refresh after reconnect.".to_string(),
            version,
            upgrade_supported: actions_supported,
        });
    }

    let diagnostics = fetch_boot_diagnostics_response(state.baseband.as_ref()).await;
    version.running_system = diagnostics.running_system;
    version.boot_reason = diagnostics.boot_reason;
    let message = if actions_supported {
        "Maintenance information loaded".to_string()
    } else {
        "Baseband SDK not available; maintenance actions are disabled in simulator mode".to_string()
    };

    Json(MaintenanceInfoResponse {
        success: true,
        message,
        version,
        upgrade_supported: actions_supported,
    })
}

async fn get_boot_diagnostics(
    State(state): State<Arc<AppState>>,
) -> Json<BootDiagnosticsResponse> {
    if state.expected_reboot_window_active().await {
        return Json(BootDiagnosticsResponse {
            success: false,
            running_system: "Rebooting".to_string(),
            boot_reason: "Rebooting".to_string(),
            message: "Device reboot in progress. Boot diagnostics will refresh after reconnect.".to_string(),
        });
    }

    Json(fetch_boot_diagnostics_response(state.baseband.as_ref()).await)
}

async fn apply_firmware_upgrade(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Json<FirmwareUpgradeActionResponse> {
    info!("Firmware upgrade request received");
    let upload_started_at = Instant::now();
    let http_upload_elapsed_ms = || upload_started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;

    let Some(baseband) = state.baseband.as_ref() else {
        warn!("Firmware upgrade rejected: baseband SDK not available");
        return Json(FirmwareUpgradeActionResponse {
            success: false,
            message: "Baseband SDK not available; firmware upgrade is disabled in simulator mode".to_string(),
            file_name: String::new(),
            file_size: 0,
            http_upload_elapsed_ms: 0,
            bytes_written: 0,
            chunk_count: 0,
            crc32: String::new(),
            reboot_expected: false,
        });
    };

    let mut file_name = String::new();
    let mut firmware = None;

    loop {
        let next_field = match multipart.next_field().await {
            Ok(field) => field,
            Err(err) => {
                warn!("Multipart next_field error: {}", err);
                return Json(FirmwareUpgradeActionResponse {
                    success: false,
                    message: format!("Failed to read firmware upload: {}", err),
                    file_name,
                    file_size: 0,
                    http_upload_elapsed_ms: http_upload_elapsed_ms(),
                    bytes_written: 0,
                    chunk_count: 0,
                    crc32: String::new(),
                    reboot_expected: false,
                });
            }
        };

        let Some(field) = next_field else {
            break;
        };

        let field_name = field.name().unwrap_or_default().to_string();
        let detected_file_name = field.file_name().unwrap_or("firmware.bin").to_string();

        info!(
            "Multipart field: name='{}', file_name='{}', content_type={:?}",
            field_name,
            detected_file_name,
            field.content_type()
        );

        if !file_name.is_empty() || (!field_name.is_empty() && field_name != "firmware") {
            continue;
        }

        let mut field = field;
        let mut buffered_firmware = Vec::new();
        loop {
            match field.chunk().await {
                Ok(Some(chunk)) => buffered_firmware.extend_from_slice(&chunk),
                Ok(None) => break,
                Err(err) => {
                    return Json(FirmwareUpgradeActionResponse {
                        success: false,
                        message: format!("Failed to read uploaded firmware file: {}", err),
                        file_name: detected_file_name,
                        file_size: 0,
                        http_upload_elapsed_ms: http_upload_elapsed_ms(),
                        bytes_written: 0,
                        chunk_count: 0,
                        crc32: String::new(),
                        reboot_expected: false,
                    });
                }
            }
        }

        match buffered_firmware.is_empty() {
            false => {
                file_name = detected_file_name;
                firmware = Some(buffered_firmware);
                break;
            }
            true => {}
        }
    }

    let firmware = match firmware {
        Some(bytes) if !bytes.is_empty() => bytes,
        Some(_) => {
            return Json(FirmwareUpgradeActionResponse {
                success: false,
                message: "The selected firmware file is empty".to_string(),
                file_name,
                file_size: 0,
                http_upload_elapsed_ms: http_upload_elapsed_ms(),
                bytes_written: 0,
                chunk_count: 0,
                crc32: String::new(),
                reboot_expected: false,
            });
        }
        None => {
            return Json(FirmwareUpgradeActionResponse {
                success: false,
                message: "No firmware file was uploaded".to_string(),
                file_name,
                file_size: 0,
                http_upload_elapsed_ms: http_upload_elapsed_ms(),
                bytes_written: 0,
                chunk_count: 0,
                crc32: String::new(),
                reboot_expected: false,
            });
        }
    };

    let file_size = firmware.len();
    let upload_elapsed_ms = http_upload_elapsed_ms();
    info!(
        "Firmware file '{}' received: {} bytes in {} ms, dispatching to background upgrade",
        file_name,
        file_size,
        upload_elapsed_ms
    );

    // 后台异步升级
    baseband.start_upgrade_background(firmware, file_name.clone(), upload_elapsed_ms);

    Json(FirmwareUpgradeActionResponse {
        success: true,
        message: format!("Firmware '{}' upload accepted. Flashing in background...", file_name),
        file_name,
        file_size,
        http_upload_elapsed_ms: upload_elapsed_ms,
        bytes_written: 0,
        chunk_count: 0,
        crc32: String::new(),
        reboot_expected: true,
    })
}

/// 轮询固件升级进度
async fn get_upgrade_progress(
    State(state): State<Arc<AppState>>,
) -> Json<bb_api::FirmwareUpgradeProgress> {
    let default_progress = bb_api::FirmwareUpgradeProgress {
        state: "idle".to_string(),
        file_name: String::new(),
        file_size: 0,
        bytes_written: 0,
        http_upload_elapsed_ms: 0,
        board_write_elapsed_ms: 0,
        total_steps: 0,
        current_step: 0,
        step_label: String::new(),
        percent: 0.0,
        message: "No upgrade in progress".to_string(),
        crc32: None,
        reboot_expected: false,
    };

    let Some(baseband) = state.baseband.as_ref() else {
        return Json(default_progress);
    };

    Json(baseband.get_upgrade_progress().unwrap_or(default_progress))
}

async fn request_system_reboot(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RebootActionRequest>,
) -> Json<RebootActionResponse> {
    let requested_delay_seconds = request.delay_seconds.unwrap_or(0);
    let reboot_quiet_window = Duration::from_secs(REBOOT_REFRESH_QUIET_WINDOW_SECS);

    let Some(baseband) = state.baseband.as_ref() else {
        return Json(RebootActionResponse {
            success: false,
            message: "Baseband SDK not available; reboot is disabled in simulator mode".to_string(),
            delay_seconds: requested_delay_seconds,
            reboot_expected: false,
        });
    };

    if requested_delay_seconds > MAX_REBOOT_DELAY_SECONDS {
        return Json(RebootActionResponse {
            success: false,
            message: format!(
                "Scheduled reboot delay '{}' exceeds the supported limit {} seconds",
                requested_delay_seconds,
                MAX_REBOOT_DELAY_SECONDS
            ),
            delay_seconds: requested_delay_seconds,
            reboot_expected: false,
        });
    }

    if requested_delay_seconds == 0 {
        let reboot_target_mac = baseband
            .get_wireless_runtime_details()
            .ok()
            .map(|details| normalize_device_mac(&details.status.mac_hex))
            .filter(|mac| !mac.is_empty());

        if let Some(target_mac) = reboot_target_mac.as_deref() {
            state.stage_reboot_pair_resume(target_mac).await;
        } else {
            state.clear_reboot_pair_resume().await;
        }

        state.begin_expected_reboot_window(reboot_quiet_window).await;

        return match baseband.reboot_device(IMMEDIATE_REBOOT_DELAY_MS) {
            Ok(()) => Json(RebootActionResponse {
                success: true,
                message: "Immediate reboot request accepted. Device is restarting now.".to_string(),
                delay_seconds: 0,
                reboot_expected: true,
            }),
            Err(err) => {
                state.clear_expected_reboot_window().await;
                state.clear_reboot_pair_resume().await;
                Json(RebootActionResponse {
                    success: false,
                    message: format!("Failed to submit reboot request: {}", err),
                    delay_seconds: 0,
                    reboot_expected: false,
                })
            }
        };
    }

    let scheduled_target_mac = match baseband.get_wireless_runtime_details() {
        Ok(details) => {
            let target_mac = details.status.mac_hex.trim().to_string();
            if target_mac.is_empty() || target_mac == "--" {
                return Json(RebootActionResponse {
                    success: false,
                    message: "Failed to resolve the currently selected Active Device for scheduled reboot.".to_string(),
                    delay_seconds: requested_delay_seconds,
                    reboot_expected: false,
                });
            }
            target_mac
        }
        Err(err) => {
            return Json(RebootActionResponse {
                success: false,
                message: format!(
                    "Failed to resolve the currently selected Active Device for scheduled reboot: {}",
                    err
                ),
                delay_seconds: requested_delay_seconds,
                reboot_expected: false,
            });
        }
    };

    let scheduled_baseband = Arc::clone(baseband);
    let scheduled_state = Arc::clone(&state);
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(requested_delay_seconds as u64)).await;
        scheduled_state.begin_expected_reboot_window(reboot_quiet_window).await;

        if let Err(err) = scheduled_baseband.switch_active_device(&scheduled_target_mac) {
            warn!(
                "Scheduled reboot failed to reselect target device {} before execution: {}",
                scheduled_target_mac,
                err
            );
            scheduled_state.clear_expected_reboot_window().await;
            scheduled_state.clear_reboot_pair_resume().await;
            return;
        }

        scheduled_state.stage_reboot_pair_resume(&scheduled_target_mac).await;

        if let Err(err) = scheduled_baseband.reboot_device(SCHEDULED_REBOOT_TRIGGER_DELAY_MS) {
            warn!(
                "Scheduled reboot failed for target device {}: {}",
                scheduled_target_mac,
                err
            );
            scheduled_state.clear_expected_reboot_window().await;
            scheduled_state.clear_reboot_pair_resume().await;
            return;
        }

        info!(
            "Scheduled reboot executed for target device {} after {} seconds",
            scheduled_target_mac,
            requested_delay_seconds
        );
    });

    Json(RebootActionResponse {
        success: true,
        message: format!(
            "Scheduled reboot request accepted. Server will reboot the selected device in {} seconds.",
            requested_delay_seconds
        ),
        delay_seconds: requested_delay_seconds,
        reboot_expected: true,
    })
}

fn build_system_info() -> SystemInfo {
    SystemInfo {
        host_name: "UserDevice".to_string(),
        product_name: "MIMO Wireless Bridge".to_string(),
        description: "mypDDL-MIMO".to_string(),
        system_date: "2026-04-09 10:00:00".to_string(),
        system_uptime: "5 days 08:16:42".to_string(),
        hardware_version: "Rev A".to_string(),
        software_version: "v1.4.0".to_string(),
        software_build: "1005".to_string(),
        build_date: "2026-04-08".to_string(),
        build_time: "17:30:00".to_string(),
        mac_address: "00:0F:92:FA:37:CE".to_string(),
        ip_address: "192.168.1.2".to_string(),
        subnet_mask: "255.255.255.0".to_string(),
        connection_type: "Static".to_string(),
        gateway: "192.168.1.1".to_string(),
        lan_mac_address: "00:0F:92:FA:37:CE".to_string(),
        lan_ip_address: "192.168.1.2".to_string(),
        lan_subnet_mask: "255.255.255.0".to_string(),
        lan_connection_type: "Static".to_string(),
        lan_gateway: "192.168.1.1".to_string(),
        wan_mac_address: "00:0F:92:FA:37:CF".to_string(),
        wan_ip_address: "10.10.10.2".to_string(),
        wan_subnet_mask: "255.255.255.0".to_string(),
        wan_connection_type: "DHCP".to_string(),
        wan_gateway: "10.10.10.1".to_string(),
        wan_primary_dns: "8.8.8.8".to_string(),
        wan_secondary_dns: "8.8.4.4".to_string(),
    }
}

fn build_maintenance_version_info(
    state: &AppState,
    system_info: &SystemInfo,
    current: Option<&WirelessRuntimeView>,
) -> MaintenanceVersionInfo {
    let active_device = current
        .and_then(|runtime| {
            runtime
                .available_devices
                .iter()
                .find(|device| device.selected)
                .map(|device| device.label.clone())
        })
        .or_else(|| {
            current
                .map(|runtime| runtime.local_mac_address.trim().to_string())
                .filter(|value| !value.is_empty() && value != "--")
        })
        .unwrap_or_else(|| "--".to_string());

    let compile_time = current
        .map(|runtime| runtime.compile_time.trim().to_string())
        .filter(|value| !value.is_empty() && value != "Unavailable")
        .unwrap_or_else(|| "Unavailable".to_string());
    let running_system = current
        .map(|runtime| runtime.running_system.clone())
        .filter(|value| !value.is_empty() && value != "Unavailable")
        .unwrap_or_else(|| "Unavailable".to_string());
    let boot_reason = current
        .map(|runtime| runtime.boot_reason.clone())
        .filter(|value| !value.is_empty() && value != "Unavailable")
        .unwrap_or_else(|| "Unavailable".to_string());

    MaintenanceVersionInfo {
        product_name: system_info.product_name.clone(),
        host_name: system_info.host_name.clone(),
        active_device,
        operation_mode: current
            .map(|runtime| runtime.operation_mode.clone())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "--".to_string()),
        access_mode: state.baseband_health.effective_mode.clone(),
        local_mac_address: current
            .map(|runtime| runtime.local_mac_address.clone())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "--".to_string()),
        system_uptime: current
            .map(|runtime| runtime.system_uptime.clone())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "--".to_string()),
        hardware_version: current
            .map(|runtime| runtime.hardware_version.clone())
            .filter(|value| !value.is_empty() && value != "Unavailable")
            .unwrap_or_else(|| "Unavailable".to_string()),
        firmware_version: current
            .map(|runtime| runtime.firmware_version.clone())
            .filter(|value| !value.is_empty() && value != "Unavailable")
            .unwrap_or_else(|| "Unavailable".to_string()),
        software_version: system_info.software_version.clone(),
        software_build: system_info.software_build.clone(),
        compile_time,
        running_system,
        boot_reason,
    }
}

async fn get_baseband_health(State(state): State<Arc<AppState>>) -> Json<BasebandHealthStatus> {
    let response = match state.baseband.as_ref() {
        Some(baseband) => {
            let mut health = baseband.get_health_status();
            health.socket_init = state.baseband_health.socket_init.clone();
            health
        }
        None => state.baseband_health.clone(),
    };

    Json(response)
}

async fn test_baseband_communication(
    State(state): State<Arc<AppState>>,
) -> Json<BasebandTestResponse> {
    let response = match state.baseband.as_ref() {
        Some(baseband) => match baseband.send_data(0, b"ping") {
            Ok(bytes_sent) => BasebandTestResponse {
                available: true,
                socket_initialized: true,
                bytes_sent: Some(bytes_sent),
                message: "Baseband communication test completed".to_string(),
            },
            Err(err) => BasebandTestResponse {
                available: true,
                socket_initialized: false,
                bytes_sent: None,
                message: format!("Baseband communication test failed: {}", err),
            },
        },
        None => BasebandTestResponse {
            available: false,
            socket_initialized: false,
            bytes_sent: None,
            message: "Baseband SDK not available; running in simulator mode".to_string(),
        },
    };

    Json(response)
}

async fn exercise_baseband_link(
    State(state): State<Arc<AppState>>,
) -> Json<BasebandLinkExerciseResponse> {
    let Some(baseband) = state.baseband.as_ref() else {
        return Json(BasebandLinkExerciseResponse {
            available: false,
            success: false,
            message: "Baseband SDK not available; cannot exercise link in simulator mode".to_string(),
            initial_link_state: "Unavailable".to_string(),
            final_link_state: "Unavailable".to_string(),
            states_seen: Vec::new(),
            final_peer_mac: None,
            socket_initialized: false,
            total_bytes_sent: 0,
            total_send_attempts: 0,
        });
    };

    let baseband = Arc::clone(baseband);
    let initial_status = match baseband.get_status_snapshot() {
        Ok(status) => status,
        Err(err) => {
            return Json(BasebandLinkExerciseResponse {
                available: true,
                success: false,
                message: format!("Failed to read initial link status: {}", err),
                initial_link_state: "Unavailable".to_string(),
                final_link_state: "Unavailable".to_string(),
                states_seen: Vec::new(),
                final_peer_mac: None,
                socket_initialized: false,
                total_bytes_sent: 0,
                total_send_attempts: 0,
            });
        }
    };

    let initial_link_state = initial_status
        .links
        .first()
        .map(|link| format_link_state(link.state).to_string())
        .unwrap_or_else(|| "Unavailable".to_string());
    let mut final_status = initial_status.clone();
    let mut states_seen = vec![initial_link_state.clone()];
    let mut notes = Vec::new();
    let mut total_bytes_sent = 0usize;
    let mut total_send_attempts = 0usize;

    if let Err(err) = baseband.set_pair_mode(true, 0x01) {
        return Json(BasebandLinkExerciseResponse {
            available: true,
            success: false,
            message: format!("Failed to enable pair mode on slot bitmap 0x01: {}", err),
            initial_link_state,
            final_link_state: "Unavailable".to_string(),
            states_seen,
            final_peer_mac: None,
            socket_initialized: false,
            total_bytes_sent,
            total_send_attempts,
        });
    }
    notes.push("Enabled pair mode on slot bitmap 0x01".to_string());

    let socket_initialized = match baseband.initialize_socket(0) {
        Ok(()) => {
            notes.push("Initialized socket 0 for traffic exercise".to_string());
            true
        }
        Err(err) => {
            notes.push(format!("Socket 0 initialization failed or was already unavailable: {}", err));
            false
        }
    };

    for poll_index in 0..20 {
        tokio::time::sleep(Duration::from_secs(1)).await;

        match baseband.get_status_snapshot() {
            Ok(status) => {
                final_status = status.clone();

                let state_name = status
                    .links
                    .first()
                    .map(|link| format_link_state(link.state).to_string())
                    .unwrap_or_else(|| "Unavailable".to_string());
                if states_seen.last().map(String::as_str) != Some(state_name.as_str()) {
                    states_seen.push(state_name.clone());
                }

                if status.links.first().map(|link| link.state >= 1).unwrap_or(false) && socket_initialized {
                    total_send_attempts = total_send_attempts.saturating_add(1);

                    match baseband.send_data(0, b"plot-probe") {
                        Ok(bytes) => {
                            total_bytes_sent = total_bytes_sent.saturating_add(bytes);
                            notes.push(format!(
                                "Poll {} sent {} bytes on socket 0 while link state was {}",
                                poll_index + 1,
                                bytes,
                                state_name
                            ));
                        }
                        Err(err) => {
                            notes.push(format!(
                                "Poll {} failed to send traffic on socket 0 while link state was {}: {}",
                                poll_index + 1,
                                state_name,
                                err
                            ));
                        }
                    }
                }
            }
            Err(err) => notes.push(format!("Poll {} failed to read link status: {}", poll_index + 1, err)),
        }
    }

    if let Err(err) = baseband.set_pair_mode(false, 0x01) {
        notes.push(format!("Failed to disable pair mode after exercise: {}", err));
    } else {
        notes.push("Disabled pair mode after exercise".to_string());
    }

    refresh_snapshot_from_baseband(&state, &baseband).await;

    let final_link_state = final_status
        .links
        .first()
        .map(|link| format_link_state(link.state).to_string())
        .unwrap_or_else(|| "Unavailable".to_string());
    let success = final_status
        .links
        .first()
        .map(|link| link.state >= 1)
        .unwrap_or(false);
    let final_peer_mac = final_status.links.first().and_then(|link| link.peer_mac_hex.clone());

    Json(BasebandLinkExerciseResponse {
        available: true,
        success,
        message: notes.join(" | "),
        initial_link_state,
        final_link_state,
        states_seen,
        final_peer_mac,
        socket_initialized,
        total_bytes_sent,
        total_send_attempts,
    })
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.tx.subscribe();

    if let Ok(initial) = serde_json::to_string(&state.snapshot.read().await.clone()) {
        if sender.send(Message::Text(initial.into())).await.is_err() {
            return;
        }
    }

    let send_task = tokio::spawn(async move {
        while let Ok(snapshot) = rx.recv().await {
            match serde_json::to_string(&snapshot) {
                Ok(payload) => {
                    if sender.send(Message::Text(payload.into())).await.is_err() {
                        break;
                    }
                }
                Err(err) => error!("failed to serialize snapshot: {}", err),
            }
        }
    });

    let recv_task = tokio::spawn(async move {
        while let Some(message) = receiver.next().await {
            match message {
                Ok(Message::Close(_)) => break,
                Ok(Message::Ping(_)) => {}
                Ok(_) => {}
                Err(_) => break,
            }
        }
    });

    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }
}

fn spawn_data_feeder(state: Arc<AppState>) {
    tokio::spawn(async move {
        let mut tick = 1_u64;
        loop {
            let interval_ms = clamp_plot_refresh_interval_ms(*state.plot_refresh_interval_ms.read().await);
            let sleep = tokio::time::sleep(Duration::from_millis(interval_ms));
            tokio::pin!(sleep);

            tokio::select! {
                _ = &mut sleep => {}
                _ = state.plot_refresh_interval_notify.notified() => {
                    continue;
                }
            }

            tracing::debug!(tick, "spawn_data_feeder tick started");

            let previous = state.snapshot.read().await.clone();
            if state.expected_reboot_window_active().await {
                let snapshot = carry_forward_snapshot(previous, tick);

                {
                    let mut guard = state.snapshot.write().await;
                    *guard = snapshot.clone();
                }
                let _ = state.tx.send(snapshot);
                tick = tick.wrapping_add(1);
                continue;
            }

            let sample_count = *state.plot_sample_count.read().await;
            let runtime_current = state.wireless_runtime.read().await.current.clone();
            let power_fallback = runtime_current.as_ref().and_then(|r| {
                if r.br_power_dbm.is_some() || r.ap_power_dbm.is_some() || r.dev_power_dbm.is_some() {
                    Some(ffi::BbPowerFallback {
                        br_power_dbm: r.br_power_dbm,
                        ap_power_dbm: r.ap_power_dbm,
                        dev_power_dbm: r.dev_power_dbm,
                    })
                } else {
                    None
                }
            });
            let snapshot = match state.baseband.as_ref() {
                Some(baseband) => match baseband.get_status_snapshot() {
                    Ok(status) => {
                        build_hardware_snapshot(
                            tick,
                            &status,
                            Some(&previous),
                            sample_count,
                            runtime_current.as_ref(),
                            power_fallback.as_ref(),
                        )
                    }
                    Err(err) => {
                        if tick % 30 == 1 {
                            warn!("Failed to refresh wireless status from baseband: {}", err);
                        }
                        carry_forward_snapshot(previous, tick)
                    }
                },
                None => build_simulated_snapshot(tick, sample_count),
            };
            tracing::debug!(tick, "spawn_data_feeder tick completed");

            {
                let mut guard = state.snapshot.write().await;
                *guard = snapshot.clone();
            }
            let _ = state.tx.send(snapshot);
            tick = tick.wrapping_add(1);
        }
    });
}

fn spawn_runtime_feeder(state: Arc<AppState>) {
    tokio::spawn(async move {
        let interval_secs = if state.baseband_health.effective_mode == "hardware-remote-bb-host" {
            REMOTE_RUNTIME_REFRESH_INTERVAL_SECS
        } else {
            LOCAL_RUNTIME_REFRESH_INTERVAL_SECS
        };
        let mut ticker = tokio::time::interval_at(
            tokio::time::Instant::now() + Duration::from_secs(interval_secs),
            Duration::from_secs(interval_secs),
        );

        loop {
            ticker.tick().await;

            let Some(baseband) = state.baseband.as_ref() else {
                continue;
            };

            refresh_runtime_from_baseband(&state, baseband).await;
        }
    });
}

fn build_simulated_snapshot(sequence: u64, plot_sample_count: usize) -> WirelessSnapshot {
    let plot_prefix = plot_series_prefix_for_role(0);
    let peer_plot_prefix = plot_series_prefix_for_role(1);
    let peer_main_rssi = -65 + oscillate(sequence + 4, 4, 13);
    let peer_aux_rssi = -69 + oscillate(sequence + 7, 5, 15);
    let peer_current_main_rssi = peer_main_rssi.clamp(-74, -52);
    let peer_current_aux_rssi = peer_aux_rssi.clamp(-78, -56);

    let peer_main_history = build_history(sequence + 9, peer_current_main_rssi, CONNECTION_HISTORY_POINTS, 5);
    let peer_aux_history = build_history(sequence + 12, peer_current_aux_rssi, CONNECTION_HISTORY_POINTS, 4);
    let ap_snr = 20 + oscillate(sequence, 2, 7);
    let ap_ldpc_err = (sequence % 7) as i32 + oscillate(sequence + 2, 2, 5).max(0);
    let ap_ldpc_num = 360 + oscillate(sequence + 5, 24, 17);
    let ap_gain_a = 72 + oscillate(sequence + 1, 7, 9);
    let ap_gain_b = 68 + oscillate(sequence + 4, 6, 11);
    let ap_mcs_rx = 6 + ((sequence % 6) as i32);
    let ap_fch_lock = if sequence % 9 == 0 { 0 } else { 1 };
    let dev_snr = 31 + oscillate(sequence + 2, 3, 7);
    let dev_gain_a = 66 + oscillate(sequence + 6, 6, 11);
    let dev_gain_b = 62 + oscillate(sequence + 8, 5, 13);

    WirelessSnapshot {
        sequence,
        general: GeneralStatus {
            role: "AP".to_string(),
            mac_address: "00:0F:92:FA:37:CE".to_string(),
            master_slave_mode: "Master".to_string(),
            networking_mode: "1V1".to_string(),
            band_mode: "Auto (2G)".to_string(),
        },
        connections: vec![ConnectionStatus {
            link_slot: "SLOT 0".to_string(),
            slot_type: "slot0".to_string(),
            direction: "Recv".to_string(),
            duration: "Unavailable".to_string(),
            frequency: "2.422 GHz".to_string(),
            bandwidth: "10 MHz".to_string(),
            mcs: "MCS 10".to_string(),
            antenna_mode: "2T2R_STBC".to_string(),
            throughput: "15372 kbps".to_string(),
            link_state: if sequence % 3 == 0 {
                "Connect".to_string()
            } else {
                "Lock".to_string()
            },
            pair_state: if sequence % 5 == 0 {
                "Pairing".to_string()
            } else {
                "Stable".to_string()
            },
            pairing_active: sequence % 5 == 0,
            mac_address: "00:0F:92:FA:37:C5".to_string(),
            snr_db: 34 + oscillate(sequence, 2, 7),
            signal_level: map_signal_level_from_snr(Some(34 + oscillate(sequence, 2, 7))),
            rssi_main_history: peer_main_history,
            rssi_aux_history: peer_aux_history,
        }],
        chart: RssiChart {
            title: "RSSI Graph".to_string(),
            target_mac_address: "00:0F:92:FA:37:CE".to_string(),
            history_context_key: "simulator|role:0|signal_user:0".to_string(),
            series: vec![
                build_chart_series(
                    "ap_snr",
                    &plot_series_label(plot_prefix, "snr"),
                    "",
                    Some(ap_snr),
                    build_metric_history(sequence, ap_snr, plot_sample_count, 2, 10, 30),
                ),
                build_chart_series(
                    "ap_ldpc_err",
                    &plot_series_label(plot_prefix, "ldpc_err"),
                    "",
                    Some(ap_ldpc_err),
                    build_metric_history(sequence + 1, ap_ldpc_err, plot_sample_count, 2, 0, 20),
                ),
                build_chart_series(
                    "ap_ldpc_num",
                    &plot_series_label(plot_prefix, "ldpc_num"),
                    "",
                    Some(ap_ldpc_num),
                    build_metric_history(sequence + 2, ap_ldpc_num, plot_sample_count, 18, 250, 420),
                ),
                build_chart_series(
                    "ap_gain_a",
                    &plot_series_label(plot_prefix, "gain_a"),
                    "",
                    Some(ap_gain_a),
                    build_metric_history(sequence + 3, ap_gain_a, plot_sample_count, 6, 40, 100),
                ),
                build_chart_series(
                    "ap_gain_b",
                    &plot_series_label(plot_prefix, "gain_b"),
                    "",
                    Some(ap_gain_b),
                    build_metric_history(sequence + 4, ap_gain_b, plot_sample_count, 6, 40, 100),
                ),
                build_chart_series(
                    "ap_mcs_rx",
                    &plot_series_label(plot_prefix, "mcs_rx"),
                    "",
                    Some(ap_mcs_rx),
                    build_metric_history(sequence + 5, ap_mcs_rx, plot_sample_count, 2, 0, 24),
                ),
                build_chart_series(
                    "ap_fch_lock",
                    &plot_series_label(plot_prefix, "fch_lock"),
                    "",
                    Some(ap_fch_lock),
                    build_metric_history(sequence + 6, ap_fch_lock, plot_sample_count, 1, 0, 1),
                ),
                build_chart_series(
                    "dev_snr",
                    &plot_series_label(peer_plot_prefix, "snr"),
                    "",
                    Some(dev_snr),
                    build_metric_history(sequence + 7, dev_snr, plot_sample_count, 2, 10, 30),
                ),
                build_chart_series(
                    "dev_gain_a",
                    &plot_series_label(peer_plot_prefix, "gain_a"),
                    "",
                    Some(dev_gain_a),
                    build_metric_history(sequence + 8, dev_gain_a, plot_sample_count, 6, 40, 100),
                ),
                build_chart_series(
                    "dev_gain_b",
                    &plot_series_label(peer_plot_prefix, "gain_b"),
                    "",
                    Some(dev_gain_b),
                    build_metric_history(sequence + 9, dev_gain_b, plot_sample_count, 6, 40, 100),
                ),
            ],
        },
    }
}

fn build_hardware_snapshot(
    sequence: u64,
    status: &BbGetStatusSummary,
    previous: Option<&WirelessSnapshot>,
    plot_sample_count: usize,
    runtime_current: Option<&WirelessRuntimeView>,
    power_fallback: Option<&ffi::BbPowerFallback>,
) -> WirelessSnapshot {
    let built_power_fallback;
    let power_fallback = match power_fallback {
        Some(pf) => Some(pf),
        None => {
            built_power_fallback = runtime_current.and_then(|r| {
                if r.br_power_dbm.is_some() || r.ap_power_dbm.is_some() || r.dev_power_dbm.is_some() {
                    Some(ffi::BbPowerFallback {
                        br_power_dbm: r.br_power_dbm,
                        ap_power_dbm: r.ap_power_dbm,
                        dev_power_dbm: r.dev_power_dbm,
                    })
                } else {
                    None
                }
            });
            built_power_fallback.as_ref()
        }
    };
    let plot_prefix = plot_series_prefix_for_role(status.role);
    let _peer_plot_prefix = plot_series_prefix_for_role(1);
    let fch_lock_value = fch_lock_value_from_status(status);
    let chart_history_context = chart_history_context_key(status);
    let has_selected_user_signal = status.snr_db.is_some()
        || status.ldpc_err.is_some()
        || status.ldpc_num.is_some()
        || status.signal_main.is_some()
        || status.signal_aux.is_some();
    let render_chart_series = has_selected_user_signal && should_render_chart_series(status);
    let mut connections = Vec::new();

    let (primary_direction, primary_phy) = resolve_primary_connection_status(status);

    if let Some(br_connection) = build_br_connection_status(status, runtime_current, previous) {
        connections.push(br_connection);
    }

    for link in &status.links {
        connections.push(build_link_connection_status(
            status,
            link,
            runtime_current,
            previous,
            primary_direction,
            primary_phy,
        ));
    }
    let chart_target = status.mac_hex.clone();
    let chart_series = if render_chart_series {
        vec![
            build_chart_series_from_source(
                "ap_snr",
                &plot_series_label(plot_prefix, "snr"),
                "",
                None,
                previous,
                status.snr_db,
                plot_sample_count,
                &chart_history_context,
            ),
            build_chart_series_from_source(
                "ap_ldpc_err",
                &plot_series_label(plot_prefix, "ldpc_err"),
                "",
                None,
                previous,
                status.ldpc_err,
                plot_sample_count,
                &chart_history_context,
            ),
            build_chart_series_from_source(
                "ap_ldpc_num",
                &plot_series_label(plot_prefix, "ldpc_num"),
                "",
                None,
                previous,
                status.ldpc_num,
                plot_sample_count,
                &chart_history_context,
            ),
            build_chart_series_from_source(
                "ap_gain_a",
                &plot_series_label(plot_prefix, "gain_a"),
                "",
                None,
                previous,
                status.signal_main,
                plot_sample_count,
                &chart_history_context,
            ),
            build_chart_series_from_source(
                "ap_gain_b",
                &plot_series_label(plot_prefix, "gain_b"),
                "",
                None,
                previous,
                status.signal_aux,
                plot_sample_count,
                &chart_history_context,
            ),
            build_chart_series_from_source(
                "ap_mcs_rx",
                &plot_series_label(plot_prefix, "mcs_rx"),
                "",
                None,
                previous,
                status.rx_mcs.map(i32::from),
                plot_sample_count,
                &chart_history_context,
            ),
            build_chart_series_from_source(
                "ap_fch_lock",
                &plot_series_label(plot_prefix, "fch_lock"),
                "",
                None,
                previous,
                fch_lock_value,
                plot_sample_count,
                &chart_history_context,
            ),
            build_chart_series_from_source(
                "br_power",
                "br_power",
                "",
                None,
                previous,
                power_fallback.and_then(|p| p.br_power_dbm.map(i32::from)),
                plot_sample_count,
                &chart_history_context,
            ),
            build_chart_series_from_source(
                "ap_power",
                "ap_power",
                "",
                None,
                previous,
                power_fallback.and_then(|p| p.ap_power_dbm.map(i32::from)),
                plot_sample_count,
                &chart_history_context,
            ),
            build_chart_series_from_source(
                "dev_power",
                "dev_power",
                "",
                None,
                previous,
                power_fallback.and_then(|p| p.dev_power_dbm.map(i32::from)),
                plot_sample_count,
                &chart_history_context,
            ),
        ]
    } else {
        Vec::new()
    };

    WirelessSnapshot {
        sequence,
        general: GeneralStatus {
            role: format_role(status.role).to_string(),
            mac_address: status.mac_hex.clone(),
            master_slave_mode: format_master_slave_mode(status).to_string(),
            networking_mode: format_networking_mode(status.mode).to_string(),
            band_mode: format_general_band_mode(runtime_current, status),
        },
        connections,
        chart: RssiChart {
            title: "RSSI Graph".to_string(),
            target_mac_address: chart_target,
            history_context_key: chart_history_context,
            series: chart_series,
        },
    }
}

fn build_link_connection_status(
    status: &BbGetStatusSummary,
    link: &ffi::BbLinkStatusSummary,
    runtime_current: Option<&WirelessRuntimeView>,
    previous: Option<&WirelessSnapshot>,
    direction: &str,
    phy: Option<&ffi::BbPhyStatusSummary>,
) -> ConnectionStatus {
    let current_main = link.signal_main.unwrap_or(RSSI_UNAVAILABLE_DBM);
    let current_aux = link.signal_aux.unwrap_or(RSSI_UNAVAILABLE_DBM);
    let current_snr = link.snr_db;
    let signal_level = map_signal_level_from_snr(current_snr);
    let slot_type = format_connection_slot_type(link.slot);

    ConnectionStatus {
        link_slot: format!("SLOT {}", link.slot),
        slot_type: slot_type.clone(),
        direction: direction.to_string(),
        duration: format_connection_duration(status, link.slot),
        frequency: phy
            .and_then(|value| normalize_connection_frequency(value.freq_khz))
            .map(format_frequency_khz)
            .unwrap_or_else(|| "Unavailable".to_string()),
        bandwidth: phy
            .map(|value| format_bandwidth(value.bandwidth))
            .unwrap_or_else(|| "Unavailable".to_string()),
        mcs: format_connection_mcs(status, link, phy, direction),
        antenna_mode: format_connection_antenna_mode(phy, direction).to_string(),
        throughput: format_connection_throughput(runtime_current, false, direction),
        link_state: format_link_state(link.state).to_string(),
        pair_state: format_pair_state(link),
        pairing_active: link.pair_state,
        mac_address: resolve_connection_mac_address(status, link, runtime_current),
        snr_db: current_snr.unwrap_or(SNR_UNAVAILABLE_DB),
        signal_level,
        rssi_main_history: history_from_previous(
            previous_connection_history(previous, &slot_type, direction, true),
            current_main,
            CONNECTION_HISTORY_POINTS,
        ),
        rssi_aux_history: history_from_previous(
            previous_connection_history(previous, &slot_type, direction, false),
            current_aux,
            CONNECTION_HISTORY_POINTS,
        ),
    }
}

fn previous_connection_history<'a>(
    previous: Option<&'a WirelessSnapshot>,
    slot_type: &str,
    direction: &str,
    primary: bool,
) -> Option<&'a [i32]> {
    previous
        .and_then(|snapshot| {
            snapshot
                .connections
                .iter()
                .find(|connection| {
                    connection.slot_type.eq_ignore_ascii_case(slot_type)
                        && connection.direction.eq_ignore_ascii_case(direction)
                })
        })
        .map(|connection| {
            if primary {
                connection.rssi_main_history.as_slice()
            } else {
                connection.rssi_aux_history.as_slice()
            }
        })
}

fn build_chart_series(
    key: &str,
    label: &str,
    unit: &str,
    current_value: Option<i32>,
    mut points: Vec<i32>,
) -> ChartSeries {
    if points.is_empty() {
        if let Some(value) = current_value {
            points.push(value);
        }
    }

    let (min_value, max_value) = chart_range(&points);

    ChartSeries {
        key: key.to_string(),
        label: label.to_string(),
        unit: unit.to_string(),
        current_value: current_value.or_else(|| points.last().copied()),
        min_value,
        max_value,
        points,
    }
}

fn build_chart_series_from_source(
    key: &str,
    label: &str,
    unit: &str,
    source: Option<&[i32]>,
    previous: Option<&WirelessSnapshot>,
    fallback_current_value: Option<i32>,
    plot_sample_count: usize,
    history_context_key: &str,
) -> ChartSeries {
    let source_points = source
        .map(|values| take_tail_points(values, plot_sample_count))
        .filter(|values| !values.is_empty());
    let points = if let Some(values) = source_points {
        values
    } else if let Some(value) = fallback_current_value {
        let previous_points = previous_chart_points(previous, key, history_context_key);
        history_from_previous(previous_points.as_deref(), value, plot_sample_count)
    } else {
        Vec::new()
    };

    let current_value = points.last().copied().or(fallback_current_value);

    build_chart_series(key, label, unit, current_value, points)
}

#[allow(dead_code)]
fn build_optional_chart_series_from_source(
    key: &str,
    label: &str,
    unit: &str,
    previous: Option<&WirelessSnapshot>,
    fallback_current_value: Option<i32>,
    plot_sample_count: usize,
    history_context_key: &str,
) -> Option<ChartSeries> {
    if fallback_current_value.is_none() {
        return None;
    }

    Some(build_chart_series_from_source(
        key,
        label,
        unit,
        None,
        previous,
        fallback_current_value,
        plot_sample_count,
        history_context_key,
    ))
}

fn plot_series_prefix_for_role(role: u8) -> &'static str {
    match role {
        0 => "AP",
        1 => "DEV",
        _ => "UNKNOWN",
    }
}

fn plot_series_label(prefix: &str, metric_name: &str) -> String {
    format!("{}_{}", prefix, metric_name.to_ascii_uppercase())
}

fn fch_lock_value_from_status(status: &BbGetStatusSummary) -> Option<i32> {
    status
        .link_state
        .or_else(|| status.links.first().map(|link| link.state))
        .map(|state| if state == 0 { 0 } else { 1 })
}

fn chart_history_context_key(status: &BbGetStatusSummary) -> String {
    let signal_user = status
        .active_user
        .or(status.detected_active_user)
        .map(|user| user.to_string())
        .unwrap_or_else(|| "none".to_string());

    format!("{}|role:{}|signal_user:{}", status.mac_hex, status.role, signal_user)
}

fn previous_chart_points(previous: Option<&WirelessSnapshot>, key: &str, history_context_key: &str) -> Option<Vec<i32>> {
    previous.and_then(|snapshot| {
        if snapshot.chart.history_context_key != history_context_key {
            return None;
        }

        snapshot
            .chart
            .series
            .iter()
            .find(|series| series.key == key)
            .map(|series| series.points.clone())
    })
}

fn take_tail_points(values: &[i32], limit: usize) -> Vec<i32> {
    let start = values.len().saturating_sub(limit);
    values[start..].to_vec()
}

fn chart_range(points: &[i32]) -> (Option<i32>, Option<i32>) {
    let min_value = points.iter().copied().min();
    let max_value = points.iter().copied().max();

    (min_value, max_value)
}

fn carry_forward_snapshot(mut previous: WirelessSnapshot, sequence: u64) -> WirelessSnapshot {
    previous.sequence = sequence;
    previous
}

fn should_render_chart_series(status: &BbGetStatusSummary) -> bool {
    if status.role == ffi::BB_ROLE_DEV && status.active_user.or(status.detected_active_user) == Some(0) {
        return false;
    }

    true
}

fn runtime_view_matches_status(current: &WirelessRuntimeView, status: &BbGetStatusSummary) -> bool {
    let runtime_mac = normalize_device_mac(&current.local_mac_address);
    let status_mac = normalize_device_mac(&status.mac_hex);

    !runtime_mac.is_empty() && runtime_mac == status_mac
}

fn resolve_connection_mac_address(
    status: &BbGetStatusSummary,
    link: &ffi::BbLinkStatusSummary,
    runtime_current: Option<&WirelessRuntimeView>,
) -> String {
    if let Some(peer_mac) = link.peer_mac_hex.clone().filter(|value| !normalize_device_mac(value).is_empty()) {
        return peer_mac;
    }

    if status.role == ffi::BB_ROLE_DEV {
        if let Some(current) = runtime_current.filter(|current| runtime_view_matches_status(current, status)) {
            if let Some(target_mac) = resolve_dev_pair_target_mac(
                current.dev_pair_target_mac.as_deref(),
                status,
                &current.available_devices,
            ) {
                return target_mac;
            }
        }
    } else if status.role == ffi::BB_ROLE_AP {
        if let Some(current) = runtime_current.filter(|current| runtime_view_matches_status(current, status)) {
            if let Some(target_mac) = current
                .ap_pair_target_macs
                .get(link.slot as usize)
                .and_then(|value| value.clone())
                .filter(|value| !normalize_device_mac(value).is_empty())
            {
                return target_mac;
            }
        }
    }

    format!("SLOT {} Peer Unknown", link.slot)
}

fn build_wireless_runtime_view(details: &WirelessRuntimeDetails) -> WirelessRuntimeView {
    let channels = details
        .channel_info
        .as_ref()
        .map(|info| info.channels.as_slice())
        .unwrap_or(&[])
        .iter()
        .map(|entry| WirelessChannelOption {
            index: entry.index,
            frequency: format_frequency_khz(entry.frequency_khz),
            power_dbm: entry.power_dbm,
        })
        .collect::<Vec<_>>();
    let connected_peer_mac_normalized = resolve_connected_peer_mac(&details.status)
        .map(|value| normalize_device_mac(&value))
        .unwrap_or_default();
    let available_devices = build_wireless_device_options(
        Some(details.status.role),
        Some(&details.status.mac_hex),
        Some(details.status.sync_mode),
        Some(details.status.sync_master),
        Some(&connected_peer_mac_normalized),
        &details.available_devices,
    );

    let configured_band = build_configured_band_view(details.band_info.as_ref());
    let live_rf = build_live_rf_view(details.band_info.as_ref(), details.channel_info.as_ref());

    WirelessRuntimeView {
        local_mac_address: details.status.mac_hex.clone(),
        operation_mode: format_operation_mode(&details.status),
        dev_pair_target_mac: resolve_dev_pair_target_mac(
            details.dev_pair_target_mac.as_deref(),
            &details.status,
            &available_devices,
        ),
        ap_pair_target_macs: details.ap_pair_target_macs.clone(),
        available_devices,
        selected_signal_user: details.status.active_user,
        detected_signal_user: details.status.detected_active_user,
        compatibility_mode: format_baseband_mode(details.status.mode).to_string(),
        configured_band: configured_band.clone(),
        live_rf: live_rf.clone(),
        work_band_code: live_rf.band_code,
        band_bitmap: configured_band.bitmap,
        bandwidth_code: details.status.bandwidth,
        bandwidth: details
            .status
            .bandwidth
            .map(format_bandwidth)
            .unwrap_or_else(|| "Unavailable".to_string()),
        frequency_khz: details.status.frequency_khz,
        frequency: details
            .status
            .frequency_khz
            .map(format_frequency_khz)
            .unwrap_or_else(|| "Unavailable".to_string()),
        system_uptime: details
            .system_info
            .as_ref()
            .map(|info| format_uptime_ms(info.uptime))
            .unwrap_or_else(|| "Unavailable".to_string()),
        compile_time: details
            .system_info
            .as_ref()
            .map(|info| info.compile_time.clone())
            .unwrap_or_else(|| "Unavailable".to_string()),
        software_version: details
            .system_info
            .as_ref()
            .map(|info| info.software_version.clone())
            .unwrap_or_else(|| "Unavailable".to_string()),
        hardware_version: details
            .system_info
            .as_ref()
            .map(|info| info.hardware_version.clone())
            .unwrap_or_else(|| "Unavailable".to_string()),
        firmware_version: details
            .system_info
            .as_ref()
            .map(|info| info.firmware_version.clone())
            .unwrap_or_else(|| "Unavailable".to_string()),
        running_system: details
            .system_info
            .as_ref()
            .and_then(|info| info.running_system.clone())
            .unwrap_or_else(|| "Unavailable".to_string()),
        boot_reason: details
            .system_info
            .as_ref()
            .and_then(|info| info.boot_reason.clone())
            .unwrap_or_else(|| "Unavailable".to_string()),
        band_auto: configured_band.auto,
        work_band: live_rf.band.clone(),
        channel_auto: live_rf.channel_auto,
        channel_count: live_rf.channel_count,
        work_channel_index: live_rf.channel_index,
        work_channel_frequency: live_rf.channel_frequency.clone(),
        channels,
        bandwidth_auto: details.bandwidth_mode.as_ref().map(|info| info.auto_mode),
        current_slot: details.mcs_value.as_ref().map(|info| info.slot),
        current_mcs_direction: details
            .mcs_value
            .as_ref()
            .map(|info| format_direction(info.dir).to_string())
            .unwrap_or_else(|| "Unavailable".to_string()),
        current_mcs_auto: details.mcs_mode.as_ref().map(|info| info.auto_mode),
        current_mcs_value: details.mcs_value.as_ref().map(|info| info.mcs),
        current_mcs_label: details
            .mcs_value
            .as_ref()
            .map(|info| format_mcs(info.mcs))
            .unwrap_or_else(|| "Unavailable".to_string()),
        current_mcs_throughput_kbps: details.mcs_value.as_ref().map(|info| info.throughput_kbps),
        current_power_user: details.current_power.as_ref().map(|info| info.user),
        current_power_mode: details
            .power_mode
            .as_ref()
            .map(|info| format_power_mode(info.pwr_mode).to_string())
            .unwrap_or_else(|| "Unavailable".to_string()),
        current_power_auto: details.power_auto.as_ref().map(|info| info.enabled),
        current_power_dbm: details.current_power.as_ref().map(|info| info.power_dbm),
        br_power_dbm: details.power_fallback.as_ref().and_then(|p| p.br_power_dbm),
        ap_power_dbm: details.power_fallback.as_ref().and_then(|p| p.ap_power_dbm),
        dev_power_dbm: details.power_fallback.as_ref().and_then(|p| p.dev_power_dbm),
        warnings: details.warnings.clone(),
    }
}

fn build_wireless_configuration_view(details: &WirelessConfigurationDetails) -> WirelessConfigurationView {
    WirelessConfigurationView {
        mode: details.config_mode,
        config_text: details.config_text.clone(),
        minidb: WirelessConfigurationMinidbView {
            role: details.minidb.role,
            band_bitmap: details.minidb.band_bitmap,
            local_mac: details.minidb.local_mac.clone(),
            ap_mac: details.minidb.ap_mac.clone(),
            slot_macs: details.minidb.slot_macs.clone(),
            power: details.minidb.power.as_ref().map(|power| WirelessConfigurationPowerView {
                pwr_mode: power.pwr_mode,
                auto_mode: power.pwr_auto,
                init_dbm: power.pwr_init,
                min_dbm: power.pwr_min,
                max_dbm: power.pwr_max,
            }),
        },
        warnings: details.warnings.clone(),
    }
}

fn build_wireless_device_options(
    active_role: Option<u8>,
    active_mac: Option<&str>,
    active_sync_mode: Option<u8>,
    active_sync_master: Option<u8>,
    connected_peer_mac_normalized: Option<&str>,
    available_devices: &[ffi::BbDiscoveredDeviceSummary],
) -> Vec<WirelessDeviceOption> {
    let current_role = active_role.map(format_role).unwrap_or("Unknown").to_string();
    let current_mac_normalized = active_mac.map(normalize_device_mac).unwrap_or_default();
    let connected_peer_mac_normalized = connected_peer_mac_normalized
        .map(normalize_device_mac)
        .unwrap_or_default();

    if available_devices.is_empty() {
        return active_mac
            .filter(|value| !normalize_device_mac(value).is_empty())
            .map(|mac_address| WirelessDeviceOption {
                role: current_role.clone(),
                mac_address: mac_address.to_string(),
                label: format_device_selector_label_with_sync(
                    &current_role,
                    mac_address,
                    active_sync_mode,
                    active_sync_master,
                ),
                selected: true,
            })
            .into_iter()
            .collect();
    }

    let mut options: Vec<WirelessDeviceOption> = Vec::with_capacity(available_devices.len());

    for device in available_devices {
            let normalized_mac = normalize_device_mac(&device.mac_address);
            let selected = !current_mac_normalized.is_empty() && normalized_mac == current_mac_normalized;
            let role = if selected {
                current_role.clone()
            } else if let Some(role_code) = device.role {
                format_role(role_code).to_string()
            } else if !device.role_label.eq_ignore_ascii_case("Unknown") {
                device.role_label.clone()
            } else if !connected_peer_mac_normalized.is_empty() && normalized_mac == connected_peer_mac_normalized {
                match active_role {
                    Some(ffi::BB_ROLE_AP) => "DEV".to_string(),
                    Some(ffi::BB_ROLE_DEV) => "AP".to_string(),
                    _ => "Unknown".to_string(),
                }
            } else {
                "Unknown".to_string()
            };
            let sync_mode = if selected {
                active_sync_mode.or(device.sync_mode)
            } else {
                device.sync_mode
            };
            let sync_master = if selected {
                active_sync_master.or(device.sync_master)
            } else {
                device.sync_master
            };

            let candidate = WirelessDeviceOption {
                role: role.clone(),
                mac_address: device.mac_address.clone(),
                label: format_device_selector_label_with_sync(
                    &role,
                    &device.mac_address,
                    sync_mode,
                    sync_master,
                ),
                selected,
            };

            if let Some(existing) = options
                .iter_mut()
                .find(|option| normalize_device_mac(&option.mac_address) == normalized_mac)
            {
                let replace_existing = should_replace_wireless_device_option(existing, &candidate);
                existing.selected |= candidate.selected;
                if replace_existing {
                    existing.role = candidate.role;
                    existing.label = candidate.label;
                }
            } else {
                options.push(candidate);
            }
    }

    options
}

fn should_replace_wireless_device_option(
    existing: &WirelessDeviceOption,
    candidate: &WirelessDeviceOption,
) -> bool {
    if candidate.selected != existing.selected {
        return candidate.selected;
    }

    let existing_role_known = !existing.role.eq_ignore_ascii_case("unknown");
    let candidate_role_known = !candidate.role.eq_ignore_ascii_case("unknown");
    if candidate_role_known != existing_role_known {
        return candidate_role_known;
    }

    candidate.label.len() > existing.label.len()
}

fn runtime_response_from_details(details: &WirelessRuntimeDetails) -> WirelessRuntimeResponse {
    let current = build_wireless_runtime_view(details);
    WirelessRuntimeResponse {
        available: true,
        message: "Wireless runtime details fetched successfully".to_string(),
        available_devices: current.available_devices.clone(),
        current: Some(current),
    }
}

fn pair_blocking_notice_from_runtime(details: &WirelessRuntimeDetails) -> Option<String> {
    details.warnings.iter().find_map(|warning| {
        let text = warning.trim();
        if text.is_empty() {
            return None;
        }

        let normalized = text.to_ascii_lowercase();
        if normalized.contains("firmware mismatch")
            || normalized.contains("version mismatch")
            || normalized.contains("same release")
        {
            Some(text.to_string())
        } else {
            None
        }
    })
}

fn has_persisted_pair_mac(value: Option<&str>) -> bool {
    value
        .map(normalize_device_mac)
        .filter(|mac| !mac.is_empty() && mac != "00000000")
        .is_some()
}

fn persisted_ap_slot_bitmap(config: &WirelessConfigurationDetails) -> u8 {
    config
        .minidb
        .slot_macs
        .iter()
        .enumerate()
        .fold(0u8, |bitmap, (slot, mac)| {
            if has_persisted_pair_mac(mac.as_deref()) {
                bitmap | 1_u8.checked_shl(slot as u32).unwrap_or(0)
            } else {
                bitmap
            }
        })
}

fn pair_connected(details: &WirelessRuntimeDetails) -> bool {
    details.status.links.iter().any(|link| link.state != 0)
        || details.status.link_state.map(|state| state != 0).unwrap_or(false)
}

fn pairing_in_progress(details: &WirelessRuntimeDetails) -> bool {
    details.status.links.iter().any(|link| link.pair_state)
}

fn reboot_pair_resume_slot_bitmap(
    details: &WirelessRuntimeDetails,
    config: &WirelessConfigurationDetails,
) -> Option<u8> {
    if pair_connected(details) || pairing_in_progress(details) {
        return None;
    }

    match details.status.role {
        ffi::BB_ROLE_DEV => has_persisted_pair_mac(config.minidb.ap_mac.as_deref()).then_some(0),
        ffi::BB_ROLE_AP => {
            let slot_bitmap = persisted_ap_slot_bitmap(config);
            (slot_bitmap != 0).then_some(slot_bitmap)
        }
        _ => None,
    }
}

async fn maybe_resume_pair_after_reboot(
    state: &Arc<AppState>,
    baseband: &Arc<BasebandManager>,
    details: &WirelessRuntimeDetails,
) {
    let current_mac = details.status.mac_hex.trim();
    if current_mac.is_empty() || current_mac == "--" {
        return;
    }

    if !state.reboot_pair_resume_pending_for(current_mac).await {
        return;
    }

    if pair_connected(details) {
        state.clear_reboot_pair_resume_if_matches(current_mac).await;
        return;
    }

    if pairing_in_progress(details) {
        return;
    }

    if let Some(blocking_notice) = pair_blocking_notice_from_runtime(details) {
        warn!(
            "Skip reboot pair resume for {} because Pair is blocked: {}",
            current_mac,
            blocking_notice
        );
        return;
    }

    let config = match baseband.get_wireless_configuration_details(0) {
        Ok(config) => config,
        Err(err) => {
            warn!(
                "Failed to read configuration before reboot pair resume for {}: {}",
                current_mac,
                err
            );
            return;
        }
    };

    let Some(slot_bmp) = reboot_pair_resume_slot_bitmap(details, &config) else {
        state.clear_reboot_pair_resume_if_matches(current_mac).await;
        return;
    };

    if !state.mark_reboot_pair_resume_attempt(current_mac).await {
        return;
    }

    match baseband.set_pair_mode(true, slot_bmp) {
        Ok(()) => info!(
            "Auto-resumed Pair after reboot for {} (role={}, slot_bmp=0x{:02X})",
            current_mac,
            format_role(details.status.role),
            slot_bmp
        ),
        Err(err) => warn!(
            "Failed to auto-resume Pair after reboot for {}: {}",
            current_mac,
            err
        ),
    }
}

fn format_runtime_fetch_error_message(error: &str) -> String {
    let normalized_error = error.trim();
    let lower_error = normalized_error.to_ascii_lowercase();

    if lower_error.contains("bb_host_connect failed")
        || lower_error.contains("failed to connect bb_host")
        || lower_error.contains("remote bb_host session not established")
    {
        let host_addr = std::env::var("BB_HOST_ADDR")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_BB_HOST_ADDR.to_string());
        let host_port = std::env::var("BB_HOST_PORT")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_BB_HOST_PORT.to_string());

        return format!(
            "Remote bb_host daemon is unavailable. Start daemon.exe and ensure {}:{} is listening, then refresh the RF page. SDK detail: {}",
            host_addr, host_port, normalized_error
        );
    }

    format!("Failed to fetch wireless runtime details: {}", normalized_error)
}

fn runtime_unavailable_response(message: String) -> WirelessRuntimeResponse {
    runtime_unavailable_response_with_devices(message, Vec::new())
}

fn runtime_unavailable_response_with_devices(
    message: String,
    available_devices: Vec<WirelessDeviceOption>,
) -> WirelessRuntimeResponse {
    WirelessRuntimeResponse {
        available: false,
        message,
        current: None,
        available_devices,
    }
}

async fn refresh_snapshot_from_baseband(state: &Arc<AppState>, baseband: &Arc<BasebandManager>) {
    if state.expected_reboot_window_active().await {
        return;
    }

    if let Ok(status) = baseband.get_status_snapshot() {
        let previous = state.snapshot.read().await.clone();
        let next_sequence = previous.sequence.wrapping_add(1);
        let sample_count = *state.plot_sample_count.read().await;
        let runtime_current = state.wireless_runtime.read().await.current.clone();
        let power_fallback = runtime_current.as_ref().and_then(|r| {
            if r.br_power_dbm.is_some() || r.ap_power_dbm.is_some() || r.dev_power_dbm.is_some() {
                Some(ffi::BbPowerFallback {
                    br_power_dbm: r.br_power_dbm,
                    ap_power_dbm: r.ap_power_dbm,
                    dev_power_dbm: r.dev_power_dbm,
                })
            } else {
                None
            }
        });
        let snapshot = build_hardware_snapshot(
            next_sequence,
            &status,
            Some(&previous),
            sample_count,
            runtime_current.as_ref(),
            power_fallback.as_ref(),
        );

        {
            let mut guard = state.snapshot.write().await;
            *guard = snapshot.clone();
        }

        let _ = state.tx.send(snapshot);
    }
}

async fn refresh_runtime_from_baseband(state: &Arc<AppState>, baseband: &Arc<BasebandManager>) {
    if state.expected_reboot_window_active().await {
        return;
    }

    let response = match baseband.get_wireless_runtime_details() {
        Ok(details) => {
            maybe_resume_pair_after_reboot(state, baseband, &details).await;
            if pair_connected(&details) {
                state
                    .clear_reboot_pair_resume_if_matches(&details.status.mac_hex)
                    .await;
            }
            runtime_response_from_details(&details)
        }
        Err(err) => runtime_unavailable_response(format_runtime_fetch_error_message(&err)),
    };

    let mut guard = state.wireless_runtime.write().await;
    *guard = response;
}

fn wireless_setting_retry_budget(action: &str) -> Option<(usize, Duration)> {
    match action {
        "set_channel" => Some((
            CHANNEL_EFFECT_RETRY_ATTEMPTS,
            Duration::from_millis(CHANNEL_EFFECT_RETRY_INTERVAL_MS),
        )),
        _ => None,
    }
}

async fn refresh_runtime_until_effect_or_timeout(
    state: &Arc<AppState>,
    baseband: &Arc<BasebandManager>,
    request: &WirelessSettingRequest,
) -> Option<WirelessRuntimeView> {
    refresh_runtime_from_baseband(state, baseband).await;
    refresh_snapshot_from_baseband(state, baseband).await;

    let mut current = state.wireless_runtime.read().await.current.clone();
    let Some((attempts, retry_interval)) = wireless_setting_retry_budget(&request.action) else {
        return current;
    };

    if verify_wireless_setting_effect(request, current.as_ref()).is_ok() {
        return current;
    }

    for _ in 0..attempts {
        tokio::time::sleep(retry_interval).await;
        refresh_runtime_from_baseband(state, baseband).await;
        refresh_snapshot_from_baseband(state, baseband).await;
        current = state.wireless_runtime.read().await.current.clone();

        if verify_wireless_setting_effect(request, current.as_ref()).is_ok() {
            break;
        }
    }

    current
}

fn verify_wireless_setting_effect(
    request: &WirelessSettingRequest,
    current: Option<&WirelessRuntimeView>,
) -> Result<(), String> {
    let Some(current) = current else {
        return Ok(());
    };

    match request.action.as_str() {
        "set_channel_mode" => {
            if let Some(requested) = request.auto_mode {
                verify_bool_effect(
                    "Channel auto mode",
                    requested,
                    current.channel_auto,
                    &current.operation_mode,
                    None,
                )?;
            }
            Ok(())
        }
        "set_channel" => {
            if let Some(requested) = request.channel_index {
                if current.work_channel_index != Some(requested) {
                    return Err(format!(
                        "Manual channel did not take effect on {}. Requested channel index {}, but runtime still reports {}.",
                        current.operation_mode,
                        requested,
                        format_optional_u8(current.work_channel_index)
                    ));
                }
            }
            Ok(())
        }
        "set_power_auto" => {
            if let Some(requested) = request.auto_mode {
                verify_bool_effect(
                    "Power auto mode",
                    requested,
                    current.current_power_auto,
                    &current.operation_mode,
                    None,
                )?;
            }
            Ok(())
        }
        "set_power" => {
            if let Some(requested) = request.power_dbm {
                if current.current_power_dbm != Some(requested) {
                    return Err(format!(
                        "Manual power did not take effect on {}. Requested {} dB, but runtime still reports {} dB.",
                        current.operation_mode,
                        requested,
                        format_optional_u8(current.current_power_dbm)
                    ));
                }
            }
            Ok(())
        }
        "set_mcs_mode" => {
            if let Some(requested) = request.auto_mode {
                verify_bool_effect(
                    "MCS auto mode",
                    requested,
                    current.current_mcs_auto,
                    &current.operation_mode,
                    Some("Use the AP side to control MCS in the current runtime."),
                )?;
            }
            Ok(())
        }
        "set_mcs" => {
            if let Some(requested) = request.mcs {
                if current.current_mcs_value != Some(requested) {
                    let mut message = format!(
                        "Manual MCS did not take effect on {}. Requested {}, but runtime still reports {}.",
                        current.operation_mode,
                        requested,
                        format_optional_u8(current.current_mcs_value)
                    );

                    if current.operation_mode.to_ascii_uppercase().contains("DEV") {
                        message.push_str(" Use the AP side to control MCS in the current runtime.");
                    }

                    return Err(message);
                }
            }
            Ok(())
        }
        "set_bandwidth" => {
            if let Some(requested) = request.bandwidth {
                if current.bandwidth_code != Some(requested) {
                    return Err(format!(
                        "Manual bandwidth did not take effect on {}. Requested {}, but runtime still reports {}.",
                        current.operation_mode,
                        requested,
                        format_optional_u8(current.bandwidth_code)
                    ));
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn verify_bool_effect(
    label: &str,
    requested: bool,
    reported: Option<bool>,
    operation_mode: &str,
    hint: Option<&str>,
) -> Result<(), String> {
    if reported == Some(requested) {
        return Ok(());
    }

    let mut message = format!(
        "{} did not take effect on {}. Requested {}, but runtime still reports {}.",
        label,
        operation_mode,
        format_bool_mode(requested),
        format_optional_bool_mode(reported)
    );

    if let Some(hint) = hint {
        message.push(' ');
        message.push_str(hint);
    }

    Err(message)
}

fn format_bool_mode(value: bool) -> &'static str {
    if value {
        "auto"
    } else {
        "manual"
    }
}

fn format_optional_bool_mode(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "auto",
        Some(false) => "manual",
        None => "unavailable",
    }
}

fn format_optional_u8(value: Option<u8>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unavailable".to_string())
}

fn parse_direction(value: &str) -> Result<u8, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "tx" | "0" => Ok(ffi::BB_DIR_TX),
        "rx" | "1" => Ok(ffi::BB_DIR_RX),
        other => Err(format!("Unsupported direction '{}'; expected tx or rx", other)),
    }
}

fn parse_power_mode(value: &str) -> Result<u8, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "openloop" | "open" | "0" => Ok(0),
        "closeloop" | "close" | "closed" | "1" => Ok(1),
        other => Err(format!("Unsupported power_mode '{}'; expected openloop or closeloop", other)),
    }
}

fn parse_band(value: u8) -> Result<u8, String> {
    match value {
        0..=2 => Ok(value),
        other => Err(format!("Unsupported target_band '{}'; expected 0(600M), 1(2.4G), or 2(5.8G)", other)),
    }
}

fn parse_band_bitmap(value: u8) -> Result<u8, String> {
    match value {
        0x01 | 0x02 | 0x04 | 0x07 => Ok(value),
        other => Err(format!(
            "Unsupported band_bitmap '{}'; expected 0x01(600M), 0x02(2.4G), 0x04(5.8G), or 0x07(Auto)",
            other
        )),
    }
}

fn format_band_bitmap_name(value: u8) -> &'static str {
    match value {
        0x01 => "600M",
        0x02 => "2.4G",
        0x04 => "5.8G",
        0x07 => "Auto",
        _ => "Unknown",
    }
}

fn resolve_band_bitmap(band_info: Option<&ffi::BbBandInfoSummary>) -> Option<u8> {
    let info = band_info?;

    if let Some(selection_bitmap) = info.selection_bitmap {
        return Some(selection_bitmap);
    }

    if info.band_auto {
        return Some(0x07);
    }

    match info.work_band {
        0 => Some(0x01),
        1 => Some(0x02),
        2 => Some(0x04),
        _ => None,
    }
}

fn build_configured_band_view(
    band_info: Option<&ffi::BbBandInfoSummary>,
) -> WirelessConfiguredBandView {
    let bitmap = resolve_band_bitmap(band_info);
    let auto = band_info.map(|info| info.band_auto);
    let label = if let Some(bitmap) = bitmap {
        format_band_bitmap_name(bitmap).to_string()
    } else if auto == Some(true) {
        "Auto".to_string()
    } else if let Some(info) = band_info {
        format_band_name(info.work_band).to_string()
    } else {
        "Unavailable".to_string()
    };

    WirelessConfiguredBandView {
        bitmap,
        label,
        auto,
    }
}

fn build_live_rf_view(
    band_info: Option<&ffi::BbBandInfoSummary>,
    channel_info: Option<&ffi::BbChannelInfoSummary>,
) -> WirelessLiveRfView {
    WirelessLiveRfView {
        band_code: band_info.map(|info| info.work_band),
        band: band_info
            .map(|info| format_band_name(info.work_band).to_string())
            .unwrap_or_else(|| "Unavailable".to_string()),
        channel_auto: channel_info.map(|info| info.auto_mode),
        channel_count: channel_info.map(|info| info.chan_num),
        channel_index: channel_info.map(|info| info.work_chan),
        channel_frequency: channel_info
            .and_then(|info| info.work_frequency_khz)
            .map(format_frequency_khz)
            .unwrap_or_else(|| "Unavailable".to_string()),
    }
}

fn format_uptime_ms(uptime_ms: u64) -> String {
    let total_seconds = uptime_ms / 1000;
    let days = total_seconds / 86_400;
    let hours = (total_seconds % 86_400) / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let seconds = total_seconds % 60;

    format!("{}d {:02}:{:02}:{:02}", days, hours, minutes, seconds)
}

fn history_from_previous(previous: Option<&[i32]>, next: i32, len: usize) -> Vec<i32> {
    let mut history = previous.map(|values| values.to_vec()).unwrap_or_default();
    history.push(next);

    if history.len() > len {
        let drain_count = history.len() - len;
        history.drain(0..drain_count);
    }

    if history.is_empty() {
        vec![next; len]
    } else {
        while history.len() < len {
            history.insert(0, next);
        }
        history
    }
}

fn build_history(sequence: u64, current: i32, len: usize, amplitude: i32) -> Vec<i32> {
    (0..len)
        .map(|offset| {
            let step = sequence.saturating_sub((len - offset) as u64);
            (current + oscillate(step, amplitude, 9) - oscillate(step / 2 + 3, 2, 5)).clamp(-82, -48)
        })
        .collect()
}

fn build_metric_history(
    sequence: u64,
    current: i32,
    len: usize,
    amplitude: i32,
    min_value: i32,
    max_value: i32,
) -> Vec<i32> {
    let drift_amplitude = (amplitude / 3).max(1);

    (0..len)
        .map(|offset| {
            let step = sequence.saturating_sub((len - offset) as u64);
            (current + oscillate(step, amplitude.max(1), 9) - oscillate(step / 2 + 3, drift_amplitude, 5))
                .clamp(min_value, max_value)
        })
        .collect()
}

fn oscillate(step: u64, amplitude: i32, period: u64) -> i32 {
    let cycle = (step % period) as i32;
    let pivot = (period as i32) / 2;
    let distance = (cycle - pivot).abs();
    amplitude - (distance * amplitude / pivot.max(1))
}

fn map_signal_level_from_snr(snr_db: Option<i32>) -> u8 {
    match snr_db {
        None => 0,
        Some(value) if value >= 20 => 4,
        Some(value) if value >= 15 => 3,
        Some(value) if value >= 9 => 2,
        Some(_) => 1,
    }
}

fn format_pair_state(link: &ffi::BbLinkStatusSummary) -> String {
    if matches!(link.state, 1 | 2) {
        "Paired".to_string()
    } else if link.pair_state {
        "Pairing".to_string()
    } else {
        "Stable".to_string()
    }
}

fn resolve_pair_slot_bitmap(slot: u8, current_role: Option<u8>) -> Result<u8, String> {
    let slot_bmp = 1_u8.checked_shl(u32::from(slot)).unwrap_or(0);
    if slot_bmp == 0 {
        return Err(format!("Unsupported slot '{}'; expected 0-7", slot));
    }

    if current_role == Some(ffi::BB_ROLE_DEV) {
        Ok(0)
    } else {
        Ok(slot_bmp)
    }
}

fn format_operation_mode(status: &BbGetStatusSummary) -> String {
    let role = format_role(status.role);

    let sync_role = if status.sync_mode == 1 {
        if status.sync_master == 1 {
            "Master"
        } else {
            "Slave"
        }
    } else {
        "Async"
    };

    format!("{} ({})", role, sync_role)
}

fn format_master_slave_mode(status: &BbGetStatusSummary) -> &'static str {
    match status.sync_master {
        1 => "Master",
        0 => "Slave",
        _ => "Unknown",
    }
}

fn format_networking_mode(mode: u8) -> &'static str {
    match mode {
        0 => "1V1",
        1 => "1VN",
        2 => "Relay",
        3 => "Director",
        _ => "Unknown",
    }
}

fn format_role(role: u8) -> &'static str {
    match role {
        0 => "AP",
        1 => "DEV",
        _ => "Unknown",
    }
}

fn resolve_primary_connection_status(
    status: &BbGetStatusSummary,
) -> (&'static str, Option<&ffi::BbPhyStatusSummary>) {
    if status.role == ffi::BB_ROLE_AP {
        if let Some(phy) = status.slot_rx_status.as_ref() {
            ("Recv", Some(phy))
        } else if let Some(phy) = status.slot_tx_status.as_ref() {
            ("Send", Some(phy))
        } else {
            ("Recv", None)
        }
    } else {
        if let Some(phy) = status.slot_tx_status.as_ref() {
            ("Send", Some(phy))
        } else if let Some(phy) = status.slot_rx_status.as_ref() {
            ("Recv", Some(phy))
        } else {
            ("Send", None)
        }
    }
}

fn normalize_connection_frequency(freq_khz: u32) -> Option<u32> {
    if freq_khz == 0 {
        None
    } else {
        Some(freq_khz)
    }
}

fn format_connection_slot_type(slot: usize) -> String {
    format!("slot{}", slot)
}

fn format_connection_duration(status: &BbGetStatusSummary, slot: usize) -> String {
    let _ = (status, slot);
    "Unavailable".to_string()
}

fn build_br_connection_status(
    status: &BbGetStatusSummary,
    runtime_current: Option<&WirelessRuntimeView>,
    previous: Option<&WirelessSnapshot>,
) -> Option<ConnectionStatus> {
    let fallback_link = status.links.first();

    let (direction, phy, current_main, current_aux, current_snr, link_state, pair_state, pairing_active, mac_address) =
        if status.role == ffi::BB_ROLE_AP {
            let (direction, phy) = if let Some(phy) = status.br_tx_status.as_ref() {
                ("Send", phy)
            } else if let Some(phy) = status.br_rx_status.as_ref() {
                ("Recv", phy)
            } else {
                return None;
            };

            (
                direction,
                phy,
                status.br_signal_main.unwrap_or(RSSI_UNAVAILABLE_DBM),
                status.br_signal_aux.unwrap_or(RSSI_UNAVAILABLE_DBM),
                status.br_snr_db,
                "Stable".to_string(),
                "Stable".to_string(),
                false,
                status.mac_hex.clone(),
            )
        } else {
            let (direction, phy) = if let Some(phy) = status.br_rx_status.as_ref() {
                ("Recv", phy)
            } else if let Some(phy) = status.br_tx_status.as_ref() {
                ("Send", phy)
            } else {
                return None;
            };

            let current_main = status
                .br_signal_main
                .or_else(|| fallback_link.and_then(|link| link.signal_main))
                .unwrap_or(RSSI_UNAVAILABLE_DBM);
            let current_aux = status
                .br_signal_aux
                .or_else(|| fallback_link.and_then(|link| link.signal_aux))
                .unwrap_or(RSSI_UNAVAILABLE_DBM);
            let current_snr = status.br_snr_db.or_else(|| fallback_link.and_then(|link| link.snr_db));
            let link_state = fallback_link
                .map(|link| format_link_state(link.state).to_string())
                .unwrap_or_else(|| "Stable".to_string());
            let pair_state = fallback_link
                .map(format_pair_state)
                .unwrap_or_else(|| "Stable".to_string());
            let pairing_active = fallback_link.map(|link| link.pair_state).unwrap_or(false);
            let mac_address = fallback_link
                .map(|link| resolve_connection_mac_address(status, link, runtime_current))
                .unwrap_or_else(|| status.mac_hex.clone());

            (
                direction,
                phy,
                current_main,
                current_aux,
                current_snr,
                link_state,
                pair_state,
                pairing_active,
                mac_address,
            )
        };

    let signal_level = map_signal_level_from_snr(current_snr);

    Some(ConnectionStatus {
        link_slot: "BR".to_string(),
        slot_type: "BR".to_string(),
        direction: direction.to_string(),
        duration: "Unavailable".to_string(),
        frequency: normalize_connection_frequency(phy.freq_khz)
            .map(format_frequency_khz)
            .unwrap_or_else(|| "Unavailable".to_string()),
        bandwidth: format_bandwidth(phy.bandwidth),
        mcs: format_mcs(phy.mcs),
        antenna_mode: if direction == "Send" {
            format_tx_rf_mode(phy.rf_mode).to_string()
        } else {
            format_rx_rf_mode(phy.rf_mode).to_string()
        },
        throughput: format_connection_throughput(runtime_current, true, direction),
        link_state,
        pair_state,
        pairing_active,
        mac_address,
        snr_db: current_snr.unwrap_or(SNR_UNAVAILABLE_DB),
        signal_level,
        rssi_main_history: history_from_previous(
            previous_connection_history(previous, "BR", direction, true),
            current_main,
            CONNECTION_HISTORY_POINTS,
        ),
        rssi_aux_history: history_from_previous(
            previous_connection_history(previous, "BR", direction, false),
            current_aux,
            CONNECTION_HISTORY_POINTS,
        ),
    })
}

fn format_connection_mcs(
    status: &BbGetStatusSummary,
    link: &ffi::BbLinkStatusSummary,
    phy: Option<&ffi::BbPhyStatusSummary>,
    direction: &str,
) -> String {
    if direction.eq_ignore_ascii_case("Recv") {
        link.rx_mcs
            .or_else(|| phy.map(|value| value.mcs))
            .map(format_mcs)
            .unwrap_or_else(|| "Unavailable".to_string())
    } else {
        phy.map(|value| value.mcs)
            .or_else(|| {
                if status.active_user.or(status.detected_active_user) == Some(ffi::BB_USER_0 as u8) {
                    status.tx_mcs
                } else {
                    None
                }
            })
            .map(format_mcs)
            .unwrap_or_else(|| "Unavailable".to_string())
    }
}

fn format_connection_antenna_mode(
    phy: Option<&ffi::BbPhyStatusSummary>,
    direction: &str,
) -> &'static str {
    let Some(phy) = phy else {
        return "Unavailable";
    };

    if direction.eq_ignore_ascii_case("Send") {
        format_tx_rf_mode(phy.rf_mode)
    } else {
        format_rx_rf_mode(phy.rf_mode)
    }
}

fn format_connection_throughput(
    runtime_current: Option<&WirelessRuntimeView>,
    is_br: bool,
    direction: &str,
) -> String {
    if is_br {
        return "Unavailable".to_string();
    }

    runtime_current
        .filter(|current| {
            let expected_direction = if direction.eq_ignore_ascii_case("Send") {
                "TX"
            } else {
                "RX"
            };

            current.current_mcs_direction.eq_ignore_ascii_case(expected_direction)
        })
        .and_then(|current| current.current_mcs_throughput_kbps)
        .map(|value| format!("{} kbps", value))
        .unwrap_or_else(|| "Unavailable".to_string())
}

fn format_tx_rf_mode(mode: u8) -> &'static str {
    match mode {
        0 => "1TX",
        1 => "2TX_STBC",
        2 => "2TX_MIMO",
        _ => "Unknown",
    }
}

fn format_rx_rf_mode(mode: u8) -> &'static str {
    match mode {
        0 => "1T1R",
        1 => "1T2R",
        2 => "2T2R_STBC",
        3 => "2T2R_MIMO",
        _ => "Unknown",
    }
}

fn normalize_device_mac(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_hexdigit())
        .collect::<String>()
        .to_lowercase()
}

fn resolve_connected_peer_mac(status: &BbGetStatusSummary) -> Option<String> {
    status
        .links
        .iter()
        .find(|link| link.state != 0)
        .and_then(|link| link.peer_mac_hex.clone())
        .or_else(|| {
            status
                .link_state
                .filter(|state| *state != 0)
                .and_then(|_| status.peer_mac_hex.clone())
        })
}

fn resolve_dev_pair_target_mac(
    configured_target: Option<&str>,
    status: &BbGetStatusSummary,
    available_devices: &[WirelessDeviceOption],
) -> Option<String> {
    if status.role != ffi::BB_ROLE_DEV {
        return None;
    }

    let ap_device_macs = available_devices
        .iter()
        .filter(|device| device.role.eq_ignore_ascii_case("AP"))
        .map(|device| normalize_device_mac(&device.mac_address))
        .filter(|mac| !mac.is_empty())
        .collect::<Vec<_>>();

    let is_known_ap_mac = |value: &str| {
        let normalized = normalize_device_mac(value);
        !normalized.is_empty() && (ap_device_macs.is_empty() || ap_device_macs.iter().any(|mac| mac == &normalized))
    };

    if let Some(target) = configured_target.filter(|value| is_known_ap_mac(value)) {
        return Some(target.to_string());
    }

    if let Some(peer_mac) = resolve_connected_peer_mac(status).filter(|value| is_known_ap_mac(value)) {
        return Some(peer_mac);
    }

    if ap_device_macs.len() == 1 {
        return available_devices
            .iter()
            .find(|device| device.role.eq_ignore_ascii_case("AP"))
            .map(|device| device.mac_address.clone());
    }

    None
}

fn format_device_selector_sync_suffix(sync_mode: Option<u8>, sync_master: Option<u8>) -> &'static str {
    let _ = sync_mode;

    match sync_master {
        Some(1) => "(M)",
        Some(0) => "(S)",
        _ => "",
    }
}

fn format_device_selector_label_with_sync(
    role: &str,
    mac_address: &str,
    sync_mode: Option<u8>,
    sync_master: Option<u8>,
) -> String {
    let normalized_mac = normalize_device_mac(mac_address);
    let role_with_sync = if role.eq_ignore_ascii_case("Unknown") {
        role.to_string()
    } else {
        format!("{}{}", role, format_device_selector_sync_suffix(sync_mode, sync_master))
    };

    if normalized_mac.is_empty() {
        role_with_sync
    } else if role.eq_ignore_ascii_case("Unknown") {
        normalized_mac
    } else {
        format!("{}:{}", role_with_sync, normalized_mac)
    }
}

fn format_link_state(state: u8) -> &'static str {
    match state {
        0 => "Idle",
        1 => "Lock",
        2 => "Connect",
        _ => "Unknown",
    }
}

fn format_baseband_mode(mode: u8) -> &'static str {
    match mode {
        0 => "Single User",
        1 => "Multi User",
        2 => "Relay",
        3 => "Director",
        _ => "Unknown",
    }
}

fn normalize_general_status_value(value: &str) -> Option<&str> {
    let trimmed = value.trim();

    if trimmed.is_empty() || trimmed == "--" || trimmed.eq_ignore_ascii_case("Unavailable") {
        None
    } else {
        Some(trimmed)
    }
}

fn infer_band_name_from_frequency_khz(freq_khz: u32) -> &'static str {
    if freq_khz < 2_000_000 {
        "1G"
    } else if freq_khz < 5_000_000 {
        "2G"
    } else {
        "5G"
    }
}

fn format_general_band_mode(
    runtime_current: Option<&WirelessRuntimeView>,
    status: &BbGetStatusSummary,
) -> String {
    let configured_label = runtime_current
        .and_then(|current| normalize_general_status_value(&current.configured_band.label))
        .map(str::to_string);
    let configured_is_auto = runtime_current
        .and_then(|current| current.configured_band.auto)
        .unwrap_or(false)
        || configured_label
            .as_deref()
            .map(|label| label.eq_ignore_ascii_case("Auto"))
            .unwrap_or(false);
    let live_band = runtime_current
        .and_then(|current| {
            normalize_general_status_value(&current.live_rf.band)
                .or_else(|| normalize_general_status_value(&current.work_band))
        })
        .map(str::to_string)
        .or_else(|| {
            status
                .frequency_khz
                .map(|freq_khz| infer_band_name_from_frequency_khz(freq_khz).to_string())
        });

    if configured_is_auto {
        if let Some(live_band) = live_band {
            return format!("Auto ({})", live_band);
        }

        return "Auto".to_string();
    }

    configured_label
        .or(live_band)
        .unwrap_or_else(|| "Unavailable".to_string())
}

fn format_band_name(band: u8) -> &'static str {
    match band {
        0 => "1G",
        1 => "2G",
        2 => "5G",
        _ => "Unknown",
    }
}

fn format_direction(dir: u8) -> &'static str {
    match dir {
        0 => "TX",
        1 => "RX",
        _ => "Unknown",
    }
}

fn format_power_mode(mode: u8) -> &'static str {
    match mode {
        0 => "Open Loop",
        1 => "Closed Loop",
        _ => "Unknown",
    }
}

fn format_bandwidth(bandwidth: u8) -> String {
    match bandwidth {
        0 => "1.25 MHz".to_string(),
        1 => "2.5 MHz".to_string(),
        2 => "5 MHz".to_string(),
        3 => "10 MHz".to_string(),
        4 => "20 MHz".to_string(),
        value => format!("Unknown Bandwidth ({})", value),
    }
}

fn format_frequency_khz(freq_khz: u32) -> String {
    if freq_khz >= 1_000_000 {
        format!("{:.3} GHz", freq_khz as f64 / 1_000_000.0)
    } else {
        format!("{:.3} MHz", freq_khz as f64 / 1_000.0)
    }
}

fn format_mcs(mcs: u8) -> String {
    match mcs {
        0 => "MCS-2 BPSK 1/2 REP4".to_string(),
        1 => "MCS-1 BPSK 1/2 REP2".to_string(),
        2 => "MCS0 BPSK 1/2".to_string(),
        3 => "MCS1 BPSK 2/3".to_string(),
        4 => "MCS2 BPSK 3/4".to_string(),
        5 => "MCS3 QPSK 1/2".to_string(),
        6 => "MCS4 QPSK 2/3".to_string(),
        7 => "MCS5 QPSK 3/4".to_string(),
        8 => "MCS6 16QAM 1/2".to_string(),
        9 => "MCS7 16QAM 2/3".to_string(),
        10 => "MCS8 16QAM 3/4".to_string(),
        11 => "MCS9 64QAM 1/2".to_string(),
        12 => "MCS10 64QAM 2/3".to_string(),
        13 => "MCS11 64QAM 3/4".to_string(),
        14 => "MCS12 256QAM 1/2".to_string(),
        15 => "MCS13 256QAM 2/3".to_string(),
        16 => "MCS14 QPSK 1/2 Dual".to_string(),
        17 => "MCS15 QPSK 2/3 Dual".to_string(),
        18 => "MCS16 QPSK 3/4 Dual".to_string(),
        19 => "MCS17 16QAM 1/2 Dual".to_string(),
        20 => "MCS18 16QAM 2/3 Dual".to_string(),
        21 => "MCS19 16QAM 3/4 Dual".to_string(),
        22 => "MCS20 64QAM 1/2 Dual".to_string(),
        23 => "MCS21 64QAM 2/3 Dual".to_string(),
        24 => "MCS22 64QAM 3/4 Dual".to_string(),
        value => format!("Unknown MCS ({})", value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_phy_status(freq_khz: u32, bandwidth: u8, mcs: u8, rf_mode: u8) -> ffi::BbPhyStatusSummary {
        ffi::BbPhyStatusSummary {
            mcs,
            rf_mode,
            tintlv_enable: 0,
            tintlv_num: 0,
            tintlv_len: 0,
            bandwidth,
            freq_khz,
        }
    }

    fn sample_link_status(state: u8, pair_state: bool, peer_mac_hex: Option<&str>) -> ffi::BbLinkStatusSummary {
        ffi::BbLinkStatusSummary {
            slot: 0,
            state,
            rx_mcs: None,
            pair_state,
            candidate_macs: Vec::new(),
            snr_db: None,
            ldpc_err: None,
            ldpc_num: None,
            signal_main: None,
            signal_aux: None,
            peer_mac_bytes: None,
            peer_mac_hex: peer_mac_hex.map(str::to_string),
        }
    }

    fn sample_dev_status(configured_peer_mac: Option<&str>) -> BbGetStatusSummary {
        BbGetStatusSummary {
            role: ffi::BB_ROLE_DEV,
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
            mac_hex: "A5:54:F6:2C".to_string(),
            frequency_khz: None,
            bandwidth: None,
            tx_mcs: None,
            rx_mcs: None,
            link_state: Some(0),
            pair_state: Some(true),
            snr_db: None,
            br_snr_db: None,
            ldpc_err: None,
            ldpc_num: None,
            signal_main: None,
            signal_aux: None,
            br_signal_main: None,
            br_signal_aux: None,
            peer_mac_bytes: None,
            peer_mac_hex: configured_peer_mac.map(str::to_string),
            links: vec![sample_link_status(0, true, configured_peer_mac)],
        }
    }

    fn sample_available_devices() -> Vec<WirelessDeviceOption> {
        vec![
            WirelessDeviceOption {
                role: "DEV".to_string(),
                mac_address: "A5:54:F6:2C".to_string(),
                label: "DEV:a554f62c".to_string(),
                selected: true,
            },
            WirelessDeviceOption {
                role: "AP".to_string(),
                mac_address: "A5:68:B0:33".to_string(),
                label: "AP:a568b033".to_string(),
                selected: false,
            },
        ]
    }

    fn sample_discovered_device(
        mac_address: &str,
        role: Option<u8>,
        role_label: &str,
        sync_mode: Option<u8>,
        sync_master: Option<u8>,
    ) -> ffi::BbDiscoveredDeviceSummary {
        ffi::BbDiscoveredDeviceSummary {
            mac_address: mac_address.to_string(),
            role,
            role_label: role_label.to_string(),
            sync_mode,
            sync_master,
        }
    }

    fn sample_runtime_view(local_mac_address: &str, dev_pair_target_mac: Option<&str>) -> WirelessRuntimeView {
        WirelessRuntimeView {
            local_mac_address: local_mac_address.to_string(),
            operation_mode: "DEV (Async)".to_string(),
            dev_pair_target_mac: dev_pair_target_mac.map(str::to_string),
            ap_pair_target_macs: Vec::new(),
            available_devices: sample_available_devices(),
            selected_signal_user: None,
            detected_signal_user: None,
            compatibility_mode: "Single User".to_string(),
            configured_band: WirelessConfiguredBandView {
                bitmap: None,
                label: "Unavailable".to_string(),
                auto: None,
            },
            live_rf: WirelessLiveRfView {
                band_code: None,
                band: "Unavailable".to_string(),
                channel_auto: None,
                channel_count: None,
                channel_index: None,
                channel_frequency: "Unavailable".to_string(),
            },
            work_band_code: None,
            band_bitmap: None,
            bandwidth_code: None,
            bandwidth: "Unavailable".to_string(),
            frequency_khz: None,
            frequency: "Unavailable".to_string(),
            system_uptime: "Unavailable".to_string(),
            compile_time: "Unavailable".to_string(),
            software_version: "Unavailable".to_string(),
            hardware_version: "Unavailable".to_string(),
            firmware_version: "Unavailable".to_string(),
            running_system: "Unavailable".to_string(),
            boot_reason: "Unavailable".to_string(),
            band_auto: None,
            work_band: "Unavailable".to_string(),
            channel_auto: None,
            channel_count: None,
            work_channel_index: None,
            work_channel_frequency: "Unavailable".to_string(),
            channels: Vec::new(),
            bandwidth_auto: None,
            current_slot: None,
            current_mcs_direction: "Unavailable".to_string(),
            current_mcs_auto: None,
            current_mcs_value: None,
            current_mcs_label: "Unavailable".to_string(),
            current_mcs_throughput_kbps: None,
            current_power_user: None,
            current_power_mode: "Unavailable".to_string(),
            current_power_auto: None,
            current_power_dbm: None,
            br_power_dbm: None,
            ap_power_dbm: None,
            dev_power_dbm: None,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn resolve_dev_pair_target_mac_filters_stale_idle_peer() {
        let status = sample_dev_status(Some("66:00:00:00"));

        let resolved = resolve_dev_pair_target_mac(
            Some("66:00:00:00"),
            &status,
            &sample_available_devices(),
        );

        assert_eq!(resolved.as_deref(), Some("A5:68:B0:33"));
    }

    #[test]
    fn resolve_dev_pair_target_mac_keeps_known_ap_target() {
        let status = sample_dev_status(Some("A5:68:B0:33"));

        let resolved = resolve_dev_pair_target_mac(
            Some("A5:68:B0:33"),
            &status,
            &sample_available_devices(),
        );

        assert_eq!(resolved.as_deref(), Some("A5:68:B0:33"));
    }

    #[test]
    fn build_wireless_device_options_deduplicates_same_mac() {
        let options = build_wireless_device_options(
            Some(ffi::BB_ROLE_DEV),
            Some("A5:54:F6:2C"),
            Some(0),
            Some(1),
            Some("A568B033"),
            &[
                sample_discovered_device(
                    "A5:68:B0:33",
                    Some(ffi::BB_ROLE_AP),
                    "AP",
                    Some(0),
                    Some(1),
                ),
                sample_discovered_device(
                    "A5:68:B0:33",
                    Some(ffi::BB_ROLE_AP),
                    "AP",
                    Some(0),
                    Some(1),
                ),
                sample_discovered_device(
                    "A5:54:F6:2C",
                    Some(ffi::BB_ROLE_DEV),
                    "DEV",
                    Some(0),
                    Some(0),
                ),
            ],
        );

        assert_eq!(options.len(), 2);
        assert_eq!(
            options
                .iter()
                .filter(|option| normalize_device_mac(&option.mac_address) == "a568b033")
                .count(),
            1,
        );
    }

    #[test]
    fn resolve_connection_mac_address_prefers_dev_ap_target_for_matching_runtime() {
        let status = sample_dev_status(Some("66:00:00:00"));
        let link = sample_link_status(1, true, Some("66:00:00:00"));
        let runtime = sample_runtime_view("A5:54:F6:2C", Some("66:00:00:00"));

        let resolved = resolve_connection_mac_address(&status, &link, Some(&runtime));

        assert_eq!(resolved, "A5:68:B0:33");
    }

    #[test]
    fn resolve_connection_mac_address_ignores_runtime_for_other_device() {
        let status = sample_dev_status(Some("66:00:00:00"));
        let link = sample_link_status(1, true, Some("66:00:00:00"));
        let runtime = sample_runtime_view("A5:68:B0:33", Some("A5:68:B0:33"));

        let resolved = resolve_connection_mac_address(&status, &link, Some(&runtime));

        assert_eq!(resolved, "66:00:00:00");
    }

    #[test]
    fn resolve_connection_mac_address_uses_saved_ap_slot_target_when_peer_is_offline() {
        let mut status = sample_dev_status(None);
        status.role = ffi::BB_ROLE_AP;
        status.mac_hex = "A5:68:B0:33".to_string();
        let link = sample_link_status(0, false, None);
        let mut runtime = sample_runtime_view("A5:68:B0:33", None);
        runtime.operation_mode = "AP (Async)".to_string();
        runtime.ap_pair_target_macs = vec![Some("A5:54:F6:2C".to_string())];

        let resolved = resolve_connection_mac_address(&status, &link, Some(&runtime));

        assert_eq!(resolved, "A5:54:F6:2C");
    }

    #[test]
    fn should_render_chart_series_hides_dev_user_zero() {
        let mut status = sample_dev_status(None);
        status.active_user = Some(0);
        status.snr_db = Some(24);

        assert!(!should_render_chart_series(&status));
    }

    #[test]
    fn should_render_chart_series_keeps_dev_nonzero_user() {
        let mut status = sample_dev_status(None);
        status.active_user = Some(8);
        status.snr_db = Some(24);

        assert!(should_render_chart_series(&status));
    }

    #[test]
    fn map_signal_level_from_snr_treats_20_db_as_full_signal() {
        assert_eq!(map_signal_level_from_snr(Some(20)), 4);
        assert_eq!(map_signal_level_from_snr(Some(19)), 3);
    }

    #[test]
    fn format_pair_state_reports_pairing_for_idle_slot_with_cached_peer_mac() {
        let link = sample_link_status(0, true, Some("A5:68:B0:33"));

        assert_eq!(format_pair_state(&link), "Pairing");
    }

    #[test]
    fn format_pair_state_reports_paired_for_locked_slot() {
        let link = sample_link_status(1, false, Some("A5:68:B0:33"));

        assert_eq!(format_pair_state(&link), "Paired");
    }

    #[test]
    fn resolve_pair_slot_bitmap_uses_zero_for_dev_role() {
        assert_eq!(resolve_pair_slot_bitmap(0, Some(ffi::BB_ROLE_DEV)).unwrap(), 0);
    }

    #[test]
    fn resolve_pair_slot_bitmap_uses_slot_mask_for_ap_role() {
        assert_eq!(resolve_pair_slot_bitmap(0, Some(ffi::BB_ROLE_AP)).unwrap(), 0x01);
        assert_eq!(resolve_pair_slot_bitmap(3, Some(ffi::BB_ROLE_AP)).unwrap(), 0x08);
    }

    #[test]
    fn resolve_primary_connection_status_uses_slot0_phy_when_active_user_is_br() {
        let mut status = sample_dev_status(Some("A5:68:B0:33"));
        status.active_user = Some(ffi::BB_USER_BR_CS as u8);
        status.tx_status = Some(sample_phy_status(2418000, 1, 25, 1));
        status.rx_status = Some(sample_phy_status(2418000, 1, 25, 2));
        status.slot_tx_status = Some(sample_phy_status(2477000, 3, 12, 1));
        status.slot_rx_status = Some(sample_phy_status(2407200, 1, 25, 2));

        let (direction, phy) = resolve_primary_connection_status(&status);

        assert_eq!(direction, "Send");
        assert_eq!(phy.unwrap().freq_khz, 2477000);
        assert_eq!(phy.unwrap().bandwidth, 3);
    }

    fn sample_runtime_details(role: u8, link_state: u8, pairing_active: bool) -> WirelessRuntimeDetails {
        let mut status = sample_dev_status(None);
        status.role = role;
        status.mac_hex = if role == ffi::BB_ROLE_AP {
            "A5:68:B0:33".to_string()
        } else {
            "A5:54:F6:2C".to_string()
        };
        status.link_state = Some(link_state);
        status.pair_state = Some(pairing_active);
        status.links = vec![sample_link_status(link_state, pairing_active, None)];

        WirelessRuntimeDetails {
            status,
            dev_pair_target_mac: None,
            ap_pair_target_macs: Vec::new(),
            available_devices: Vec::new(),
            system_info: None,
            band_info: None,
            channel_info: None,
            bandwidth_mode: None,
            mcs_mode: None,
            mcs_value: None,
            power_mode: None,
            current_power: None,
            power_auto: None,
            power_fallback: None,
            warnings: Vec::new(),
        }
    }

    fn sample_configuration_details(ap_mac: Option<&str>, slot_macs: &[Option<&str>]) -> WirelessConfigurationDetails {
        WirelessConfigurationDetails {
            config_mode: 0,
            config_text: String::new(),
            minidb: bb_api::WirelessConfigurationMinidbDetails {
                role: None,
                band_bitmap: None,
                local_mac: None,
                ap_mac: ap_mac.map(str::to_string),
                slot_macs: slot_macs
                    .iter()
                    .map(|value| value.map(str::to_string))
                    .collect(),
                power: None,
            },
            warnings: Vec::new(),
        }
    }

    #[test]
    fn reboot_pair_resume_slot_bitmap_uses_saved_ap_mac_for_dev() {
        let details = sample_runtime_details(ffi::BB_ROLE_DEV, 0, false);
        let config = sample_configuration_details(Some("A568B033"), &[]);

        assert_eq!(reboot_pair_resume_slot_bitmap(&details, &config), Some(0));
    }

    #[test]
    fn reboot_pair_resume_slot_bitmap_collects_saved_slots_for_ap() {
        let details = sample_runtime_details(ffi::BB_ROLE_AP, 0, false);
        let config = sample_configuration_details(None, &[Some("A554F62C"), None, Some("A5660001")]);

        assert_eq!(reboot_pair_resume_slot_bitmap(&details, &config), Some(0x05));
    }

    #[test]
    fn reboot_pair_resume_slot_bitmap_skips_connected_or_pairing_links() {
        let connected = sample_runtime_details(ffi::BB_ROLE_AP, 1, false);
        let pairing = sample_runtime_details(ffi::BB_ROLE_DEV, 0, true);
        let config = sample_configuration_details(Some("A568B033"), &[Some("A554F62C")]);

        assert_eq!(reboot_pair_resume_slot_bitmap(&connected, &config), None);
        assert_eq!(reboot_pair_resume_slot_bitmap(&pairing, &config), None);
    }

    #[test]
    fn format_runtime_fetch_error_message_mentions_daemon_when_bb_host_is_unreachable() {
        let message = format_runtime_fetch_error_message("bb_host_connect failed with code: -1");

        assert!(message.contains("Remote bb_host daemon is unavailable"));
        assert!(message.contains("127.0.0.1:50000"));
        assert!(message.contains("Start daemon.exe"));
    }
}
