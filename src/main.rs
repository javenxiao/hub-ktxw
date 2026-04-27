mod ffi;
mod bb_api;

use std::{net::SocketAddr, sync::Arc, time::Duration};

use axum::{
    extract::{
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
    services::ServeDir,
    set_header::SetResponseHeaderLayer,
};
use tracing::{error, info, warn};

use bb_api::{resolve_plot_user, BasebandHealthStatus, BasebandManager, WirelessRuntimeDetails};
use ffi::{BbGetStatusSummary, BbPlotSnapshotSummary};

const DEFAULT_RUST_LOG: &str = "info";
const DEFAULT_BB_HOST_ADDR: &str = "127.0.0.1";
const DEFAULT_BB_HOST_PORT: &str = "50000";

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
}

#[derive(Debug, Clone, Serialize)]
struct WirelessRuntimeView {
    local_mac_address: String,
    operation_mode: String,
    available_devices: Vec<WirelessDeviceOption>,
    selected_signal_user: Option<u8>,
    detected_signal_user: Option<u8>,
    compatibility_mode: String,
    work_band_code: Option<u8>,
    bandwidth_code: Option<u8>,
    bandwidth: String,
    frequency_khz: Option<u32>,
    frequency: String,
    system_uptime: String,
    compile_time: String,
    software_version: String,
    hardware_version: String,
    firmware_version: String,
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
    device_mac: Option<String>,
    pair_start: Option<bool>,
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
    mac_address: String,
    operation_mode: String,
    compatibility_mode: String,
    bandwidth: String,
    frequency: String,
    tx_power: String,
}

#[derive(Debug, Clone, Serialize)]
struct ConnectionStatus {
    link_slot: String,
    link_state: String,
    pair_state: String,
    mac_address: String,
    tx_mod: String,
    rx_mod: String,
    snr_db: i32,
    signal_level_main: u8,
    signal_level_aux: u8,
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
    series: Vec<ChartSeries>,
}

const CONNECTION_HISTORY_POINTS: usize = 18;
const DEFAULT_AP_PLOT_SAMPLE_POINTS: usize = 200;
const RSSI_UNAVAILABLE_DBM: i32 = -127;
const SNR_UNAVAILABLE_DB: i32 = -1;

struct AppState {
    snapshot: RwLock<WirelessSnapshot>,
    wireless_runtime: RwLock<WirelessRuntimeResponse>,
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
            plot_refresh_interval_ms: RwLock::new(clamp_plot_refresh_interval_ms(initial_plot_refresh_interval_ms)),
            plot_sample_count: RwLock::new(clamp_plot_sample_count(initial_plot_sample_count)),
            plot_refresh_interval_notify: Notify::new(),
            tx,
            baseband,
            baseband_health,
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
    if baseband_health.effective_mode == "hardware-remote-bb-host" {
        3_000
    } else {
        1_000
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    apply_default_runtime_env();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("========== RSHTML Server Starting ==========");

    // 初始化基带 API
    let (baseband, baseband_health) = match BasebandManager::initialize_with_health() {
        (Some(bb), health) => {
            let mut health = health;

            if health.effective_mode == "hardware-remote-bb-host" {
                if health.host.connected {
                    info!("✓ Remote bb_host session initialized successfully");
                } else {
                    warn!(
                        "Remote bb_host manager initialized without an active daemon session: {}",
                        health.host.message
                    );
                    info!("Remote bb_host auto-reconnect remains enabled; waiting for daemon availability");
                }
                health.socket_init.message = "Skipped in remote bb_host mode".to_string();
            } else {
                info!("✓ Baseband API initialized successfully");
                // 初始化通信 socket
                let socket_result = bb.initialize_socket(0);
                health.record_socket_init(socket_result.clone(), 0);

                if let Err(e) = socket_result {
                    warn!("Failed to initialize socket: {}", e);
                } else {
                    info!("✓ Socket 0 initialized for data communication");
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

    let initial = match (baseband.as_ref(), baseband_health.runtime.status_snapshot.as_ref()) {
        (Some(baseband), Some(status)) => {
            let plot_snapshot = baseband.get_plot_snapshot();
            build_hardware_snapshot(
                0,
                status,
                plot_snapshot.as_ref(),
                None,
                DEFAULT_AP_PLOT_SAMPLE_POINTS,
            )
        }
        _ => build_simulated_snapshot(0, DEFAULT_AP_PLOT_SAMPLE_POINTS),
    };
    let initial_runtime = match baseband.as_ref() {
        Some(baseband) => match baseband.get_wireless_runtime_details() {
            Ok(details) => runtime_response_from_details(&details),
            Err(err) => runtime_unavailable_response(format!(
                "Failed to fetch wireless runtime details: {}",
                err
            )),
        },
        None => runtime_unavailable_response(
            "Baseband SDK not available; runtime controls require real hardware mode".to_string(),
        ),
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
        .route("/api/wireless/runtime/apply", post(apply_wireless_setting))
        .route(
            "/api/wireless/plot/settings",
            get(get_plot_refresh_settings).post(apply_plot_refresh_settings),
        )
        .route("/api/system/info", get(get_system_info))
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
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("wireless status server listening on http://{}", addr);
    info!("========== Server Ready ==========\n");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn get_wireless_status(State(state): State<Arc<AppState>>) -> Json<WirelessSnapshot> {
    Json(state.snapshot.read().await.clone())
}

async fn get_wireless_runtime(State(state): State<Arc<AppState>>) -> Json<WirelessRuntimeResponse> {
    Json(state.wireless_runtime.read().await.clone())
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
        "set_pair_mode" => request
            .pair_start
            .ok_or_else(|| "pair_start is required".to_string())
            .and_then(|pair_start| {
                let slot = request.slot.unwrap_or(default_slot);
                let slot_bmp = 1_u8.checked_shl(u32::from(slot)).unwrap_or(0);
                if slot_bmp == 0 {
                    return Err(format!("Unsupported slot '{}'; expected 0-7", slot));
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
            refresh_snapshot_from_baseband(&state, &baseband).await;
            refresh_runtime_from_baseband(&state, &baseband).await;
            let current = state.wireless_runtime.read().await.current.clone();
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

async fn get_system_info() -> Json<SystemInfo> {
    Json(build_system_info())
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
        let mut last_plot_sample_count: Option<usize> = None;
        let mut plot_stall_ticks = 0_u8;

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
            let sample_count = *state.plot_sample_count.read().await;
            let snapshot = match state.baseband.as_ref() {
                Some(baseband) => match baseband.get_status_snapshot() {
                    Ok(status) => {
                        let plot_user = resolve_plot_user(&status);

                        if let Err(err) = baseband.ensure_plot_stream(plot_user) {
                            if tick % 30 == 1 {
                                warn!("Failed to keep AP plot stream enabled: {}", err);
                            }
                        }

                        let mut plot_snapshot = baseband.get_plot_snapshot();
                        let current_plot_sample_count = plot_snapshot.as_ref().map(|plot| plot.sample_count);
                        let plot_has_progress = match (current_plot_sample_count, last_plot_sample_count) {
                            (Some(current), Some(previous)) => current > previous,
                            (Some(current), None) => current > 0,
                            (None, _) => false,
                        };

                        if plot_has_progress {
                            plot_stall_ticks = 0;
                        } else {
                            plot_stall_ticks = plot_stall_ticks.saturating_add(1);
                            if plot_stall_ticks >= 2 {
                                match baseband.rebind_plot_stream(plot_user) {
                                    Ok(()) => {
                                        tracing::info!(tick, plot_user, "Rebound AP plot stream after stalled samples");
                                        plot_snapshot = baseband.get_plot_snapshot();
                                    }
                                    Err(err) => {
                                        warn!("Failed to rebind AP plot stream: {}", err);
                                    }
                                }
                                plot_stall_ticks = 0;
                            }
                        }

                        last_plot_sample_count = plot_snapshot.as_ref().map(|plot| plot.sample_count);

                        build_hardware_snapshot(
                            tick,
                            &status,
                            plot_snapshot.as_ref(),
                            Some(&previous),
                            sample_count,
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
            5
        } else {
            2
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

    WirelessSnapshot {
        sequence,
        general: GeneralStatus {
            mac_address: "00:0F:92:FA:37:CE".to_string(),
            operation_mode: "Master".to_string(),
            compatibility_mode: "PDDL".to_string(),
            bandwidth: "4 MHz".to_string(),
            frequency: format!("2.{:03} GHz", 438 + ((sequence % 5) as u16)),
            tx_power: format!("{} dBm", 20 + (sequence % 2)),
        },
        connections: vec![ConnectionStatus {
            link_slot: "SLOT 0".to_string(),
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
            mac_address: "00:0F:92:FA:37:C5".to_string(),
            tx_mod: if sequence % 2 == 0 {
                "QPSK FEC 1/2".to_string()
            } else {
                "16-QAM FEC 3/4".to_string()
            },
            rx_mod: if sequence % 3 == 0 {
                "64-QAM FEC 5/6".to_string()
            } else {
                "16-QAM FEC 3/4".to_string()
            },
            snr_db: 34 + oscillate(sequence, 2, 7),
            signal_level_main: map_signal_level(peer_current_main_rssi),
            signal_level_aux: map_signal_level(peer_current_aux_rssi),
            rssi_main_history: peer_main_history,
            rssi_aux_history: peer_aux_history,
        }],
        chart: RssiChart {
            title: "AP Plot Data".to_string(),
            target_mac_address: "00:0F:92:FA:37:CE".to_string(),
            series: vec![
                build_chart_series(
                    "ap_snr",
                    "ap_snr",
                    "",
                    Some(ap_snr),
                    build_metric_history(sequence, ap_snr, plot_sample_count, 2, 10, 30),
                ),
                build_chart_series(
                    "ap_ldpc_err",
                    "ap_ldpc_err",
                    "",
                    Some(ap_ldpc_err),
                    build_metric_history(sequence + 1, ap_ldpc_err, plot_sample_count, 2, 0, 20),
                ),
                build_chart_series(
                    "ap_ldpc_num",
                    "ap_ldpc_num",
                    "",
                    Some(ap_ldpc_num),
                    build_metric_history(sequence + 2, ap_ldpc_num, plot_sample_count, 18, 250, 420),
                ),
                build_chart_series(
                    "ap_gain_a",
                    "ap_gain_a",
                    "",
                    Some(ap_gain_a),
                    build_metric_history(sequence + 3, ap_gain_a, plot_sample_count, 6, 40, 100),
                ),
                build_chart_series(
                    "ap_gain_b",
                    "ap_gain_b",
                    "",
                    Some(ap_gain_b),
                    build_metric_history(sequence + 4, ap_gain_b, plot_sample_count, 6, 40, 100),
                ),
                build_chart_series(
                    "ap_mcs_rx",
                    "ap_mcs_rx",
                    "",
                    Some(ap_mcs_rx),
                    build_metric_history(sequence + 5, ap_mcs_rx, plot_sample_count, 2, 0, 24),
                ),
                build_chart_series(
                    "ap_fch_lock",
                    "ap_fch_lock",
                    "",
                    Some(ap_fch_lock),
                    build_metric_history(sequence + 6, ap_fch_lock, plot_sample_count, 1, 0, 1),
                ),
            ],
        },
    }
}

fn build_hardware_snapshot(
    sequence: u64,
    status: &BbGetStatusSummary,
    plot_snapshot: Option<&BbPlotSnapshotSummary>,
    previous: Option<&WirelessSnapshot>,
    plot_sample_count: usize,
) -> WirelessSnapshot {
    let plot_snr_points = plot_snapshot.map(|plot| convert_plot_snr_points_to_db(&plot.snr));
    let connections = status
        .links
        .iter()
        .map(|link| {
            let current_main = link.signal_main.unwrap_or(RSSI_UNAVAILABLE_DBM);
            let current_aux = link.signal_aux.unwrap_or(RSSI_UNAVAILABLE_DBM);

            ConnectionStatus {
                link_slot: format!("SLOT {}", link.slot),
                link_state: format_link_state(link.state).to_string(),
                pair_state: format_pair_state(link),
                mac_address: link
                    .peer_mac_hex
                    .clone()
                    .unwrap_or_else(|| format!("SLOT {} Peer Unknown", link.slot)),
                tx_mod: status
                    .tx_mcs
                    .map(format_mcs)
                    .unwrap_or_else(|| "Unavailable".to_string()),
                rx_mod: link
                    .rx_mcs
                    .map(format_mcs)
                    .unwrap_or_else(|| "Unavailable".to_string()),
                snr_db: link.snr_db.unwrap_or(SNR_UNAVAILABLE_DB),
                signal_level_main: map_signal_level(current_main),
                signal_level_aux: map_signal_level(current_aux),
                rssi_main_history: history_from_previous(
                    previous_connection_history(previous, link.slot, true),
                    current_main,
                    CONNECTION_HISTORY_POINTS,
                ),
                rssi_aux_history: history_from_previous(
                    previous_connection_history(previous, link.slot, false),
                    current_aux,
                    CONNECTION_HISTORY_POINTS,
                ),
            }
        })
        .collect::<Vec<_>>();
    let chart_target = status.mac_hex.clone();
    let chart_series = vec![
        build_chart_series_from_source(
            "ap_snr",
            "ap_snr",
            "",
            plot_snr_points.as_deref(),
            previous,
            status.snr_db,
            plot_sample_count,
        ),
        build_chart_series_from_source(
            "ap_ldpc_err",
            "ap_ldpc_err",
            "",
            plot_snapshot.map(|plot| plot.ldpc_err.as_slice()),
            previous,
            status.ldpc_err,
            plot_sample_count,
        ),
        build_chart_series_from_source(
            "ap_ldpc_num",
            "ap_ldpc_num",
            "",
            plot_snapshot.map(|plot| plot.ldpc_num.as_slice()),
            previous,
            None,
            plot_sample_count,
        ),
        build_chart_series_from_source(
            "ap_gain_a",
            "ap_gain_a",
            "",
            plot_snapshot.map(|plot| plot.gain_a.as_slice()),
            previous,
            status.signal_main,
            plot_sample_count,
        ),
        build_chart_series_from_source(
            "ap_gain_b",
            "ap_gain_b",
            "",
            plot_snapshot.map(|plot| plot.gain_b.as_slice()),
            previous,
            status.signal_aux,
            plot_sample_count,
        ),
        build_chart_series_from_source(
            "ap_mcs_rx",
            "ap_mcs_rx",
            "",
            plot_snapshot.map(|plot| plot.mcs_rx.as_slice()),
            previous,
            status.rx_mcs.map(i32::from),
            plot_sample_count,
        ),
        build_chart_series_from_source(
            "ap_fch_lock",
            "ap_fch_lock",
            "",
            plot_snapshot.map(|plot| plot.fch_lock.as_slice()),
            previous,
            None,
            plot_sample_count,
        ),
    ];

    WirelessSnapshot {
        sequence,
        general: GeneralStatus {
            mac_address: status.mac_hex.clone(),
            operation_mode: format_operation_mode(status),
            compatibility_mode: format_baseband_mode(status.mode).to_string(),
            bandwidth: status
                .bandwidth
                .map(format_bandwidth)
                .unwrap_or_else(|| "Unavailable".to_string()),
            frequency: status
                .frequency_khz
                .map(format_frequency_khz)
                .unwrap_or_else(|| "Unavailable".to_string()),
            tx_power: "Unavailable".to_string(),
        },
        connections,
        chart: RssiChart {
            title: "AP Plot Data".to_string(),
            target_mac_address: chart_target,
            series: chart_series,
        },
    }
}

fn previous_connection_history(
    previous: Option<&WirelessSnapshot>,
    slot: usize,
    primary: bool,
) -> Option<&[i32]> {
    previous
        .and_then(|snapshot| {
            snapshot
                .connections
                .iter()
                .find(|connection| connection.link_slot == format!("SLOT {}", slot))
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
) -> ChartSeries {
    let source_points = source
        .map(|values| take_tail_points(values, plot_sample_count))
        .filter(|values| !values.is_empty());
    let source_missing = source_points.is_none();

    let mut points = source_points
        .or_else(|| previous_chart_points(previous, key))
        .unwrap_or_default();

    if source_missing {
        if let Some(value) = fallback_current_value {
            points = history_from_previous(Some(points.as_slice()), value, plot_sample_count);
        }
    }

    let current_value = points.last().copied().or(fallback_current_value);

    build_chart_series(key, label, unit, current_value, points)
}

fn convert_plot_snr_points_to_db(values: &[i32]) -> Vec<i32> {
    values.iter().copied().map(plot_snr_linear_to_db).collect()
}

fn plot_snr_linear_to_db(snr_linear: i32) -> i32 {
    if snr_linear <= 0 {
        return 0;
    }

    (10.0 * ((snr_linear as f64) / 36.0).log10()).round() as i32
}

fn previous_chart_points(previous: Option<&WirelessSnapshot>, key: &str) -> Option<Vec<i32>> {
    previous.and_then(|snapshot| {
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
    let current_role = format_role(details.status.role).to_string();
    let current_mac_normalized = normalize_device_mac(&details.status.mac_hex);
    let peer_mac_normalized = details
        .status
        .peer_mac_hex
        .as_deref()
        .map(normalize_device_mac)
        .unwrap_or_default();
    let available_devices = if details.available_devices.is_empty() {
        vec![WirelessDeviceOption {
            role: current_role.clone(),
            mac_address: details.status.mac_hex.clone(),
            label: format_device_selector_label(&current_role, &details.status.mac_hex),
            selected: true,
        }]
    } else {
        details
            .available_devices
            .iter()
            .map(|device| {
                let normalized_mac = normalize_device_mac(&device.mac_address);
                let selected = !current_mac_normalized.is_empty() && normalized_mac == current_mac_normalized;
                let role = if selected {
                    current_role.clone()
                } else if !peer_mac_normalized.is_empty() && normalized_mac == peer_mac_normalized {
                    match details.status.role {
                        0 => "DEV".to_string(),
                        1 => "AP".to_string(),
                        _ => "Unknown".to_string(),
                    }
                } else if device.role_label == "Unknown" {
                    // role_label is always "Unknown" from list_host_devices (SDK only exposes MAC).
                    // When the RF link is not yet established (peer_mac_hex absent), infer the
                    // opposite role from the current device: AP peer is DEV and vice-versa.
                    match details.status.role {
                        0 => "DEV".to_string(),
                        1 => "AP".to_string(),
                        _ => "Unknown".to_string(),
                    }
                } else {
                    device.role_label.clone()
                };

                WirelessDeviceOption {
                    role: role.clone(),
                    mac_address: device.mac_address.clone(),
                    label: format_device_selector_label(&role, &device.mac_address),
                    selected,
                }
            })
            .collect::<Vec<_>>()
    };

    WirelessRuntimeView {
        local_mac_address: details.status.mac_hex.clone(),
        operation_mode: format_operation_mode(&details.status),
        available_devices,
        selected_signal_user: details.status.active_user,
        detected_signal_user: details.status.detected_active_user,
        compatibility_mode: format_baseband_mode(details.status.mode).to_string(),
        work_band_code: details.band_info.as_ref().map(|info| info.work_band),
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
        band_auto: details.band_info.as_ref().map(|info| info.band_auto),
        work_band: details
            .band_info
            .as_ref()
            .map(|info| format_band_name(info.work_band).to_string())
            .unwrap_or_else(|| "Unavailable".to_string()),
        channel_auto: details.channel_info.as_ref().map(|info| info.auto_mode),
        channel_count: details.channel_info.as_ref().map(|info| info.chan_num),
        work_channel_index: details.channel_info.as_ref().map(|info| info.work_chan),
        work_channel_frequency: details
            .channel_info
            .as_ref()
            .and_then(|info| info.work_frequency_khz)
            .map(format_frequency_khz)
            .unwrap_or_else(|| "Unavailable".to_string()),
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
        warnings: details.warnings.clone(),
    }
}

fn runtime_response_from_details(details: &WirelessRuntimeDetails) -> WirelessRuntimeResponse {
    WirelessRuntimeResponse {
        available: true,
        message: "Wireless runtime details fetched successfully".to_string(),
        current: Some(build_wireless_runtime_view(details)),
    }
}

fn runtime_unavailable_response(message: String) -> WirelessRuntimeResponse {
    WirelessRuntimeResponse {
        available: false,
        message,
        current: None,
    }
}

async fn refresh_snapshot_from_baseband(state: &Arc<AppState>, baseband: &Arc<BasebandManager>) {
    if let Ok(status) = baseband.get_status_snapshot() {
        let previous = state.snapshot.read().await.clone();
        let next_sequence = previous.sequence.wrapping_add(1);
        let plot_snapshot = baseband.get_plot_snapshot();
        let sample_count = *state.plot_sample_count.read().await;
        let snapshot = build_hardware_snapshot(
            next_sequence,
            &status,
            plot_snapshot.as_ref(),
            Some(&previous),
            sample_count,
        );

        {
            let mut guard = state.snapshot.write().await;
            *guard = snapshot.clone();
        }

        let _ = state.tx.send(snapshot);
    }
}

async fn refresh_runtime_from_baseband(state: &Arc<AppState>, baseband: &Arc<BasebandManager>) {
    let response = match baseband.get_wireless_runtime_details() {
        Ok(details) => runtime_response_from_details(&details),
        Err(err) => runtime_unavailable_response(format!(
            "Failed to fetch wireless runtime details: {}",
            err
        )),
    };

    let mut guard = state.wireless_runtime.write().await;
    *guard = response;
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

fn map_signal_level(rssi_dbm: i32) -> u8 {
    match rssi_dbm {
        i32::MIN..=-100 => 0,
        96..=i32::MAX => 4,
        64..=95 => 3,
        32..=63 => 2,
        1..=31 => 1,
        -58..=-1 => 4,
        -64..=-59 => 3,
        -70..=-65 => 2,
        _ => 1,
    }
}

fn format_pair_state(link: &ffi::BbLinkStatusSummary) -> String {
    if link.pair_state {
        "Pairing".to_string()
    } else if link.peer_mac_hex.is_some() {
        "Paired".to_string()
    } else {
        "Stable".to_string()
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

fn format_role(role: u8) -> &'static str {
    match role {
        0 => "AP",
        1 => "DEV",
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

fn format_device_selector_label(role: &str, mac_address: &str) -> String {
    let normalized_mac = normalize_device_mac(mac_address);
    if normalized_mac.is_empty() {
        role.to_string()
    } else {
        format!("{}:{}", role, normalized_mac)
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
        1 => "Close Loop",
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
        5 => "40 MHz".to_string(),
        value => format!("Unknown ({})", value),
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
