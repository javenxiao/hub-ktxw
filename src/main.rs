mod ffi;
mod bb_api;

use std::{net::SocketAddr, sync::Arc, time::Duration};

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use tokio::sync::{broadcast, RwLock};
use tower_http::{cors::CorsLayer, services::ServeDir};
use tracing::{error, info, warn};

use bb_api::{BasebandHealthStatus, BasebandManager, CommunicationStats};
use ffi::BbGetStatusSummary;

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
struct BasebandStatsResponse {
    available: bool,
    stats: Option<CommunicationStats>,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
struct BasebandTestResponse {
    available: bool,
    socket_initialized: bool,
    bytes_sent: Option<usize>,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
struct WirelessSnapshot {
    sequence: u64,
    general: GeneralStatus,
    traffic: TrafficStatus,
    connections: Vec<ConnectionStatus>,
    chart: RssiChart,
}

#[derive(Debug, Clone, Serialize)]
struct GeneralStatus {
    mac_address: String,
    operation_mode: String,
    network_id: String,
    compatibility_mode: String,
    bandwidth: String,
    frequency: String,
    tx_power: String,
    encryption_type: String,
}

#[derive(Debug, Clone, Serialize)]
struct TrafficStatus {
    receive_bytes: u64,
    receive_packets: u64,
    transmit_bytes: u64,
    transmit_packets: u64,
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
    rssi_main_dbm: i32,
    rssi_aux_dbm: i32,
    signal_level_main: u8,
    signal_level_aux: u8,
    rssi_main_history: Vec<i32>,
    rssi_aux_history: Vec<i32>,
}

#[derive(Debug, Clone, Serialize)]
struct RssiChart {
    target_mac_address: String,
    primary_label: String,
    secondary_label: String,
    current_primary_rssi_dbm: i32,
    current_secondary_rssi_dbm: i32,
    min_rssi_dbm: i32,
    max_rssi_dbm: i32,
    primary_points: Vec<i32>,
    secondary_points: Vec<i32>,
}

const HISTORY_POINTS: usize = 18;
const RSSI_UNAVAILABLE_DBM: i32 = -127;
const SNR_UNAVAILABLE_DB: i32 = -1;

struct AppState {
    snapshot: RwLock<WirelessSnapshot>,
    tx: broadcast::Sender<WirelessSnapshot>,
    baseband: Option<Arc<BasebandManager>>,
    baseband_health: BasebandHealthStatus,
}

impl AppState {
    fn new(
        initial: WirelessSnapshot,
        baseband: Option<Arc<BasebandManager>>,
        baseband_health: BasebandHealthStatus,
    ) -> Self {
        let (tx, _) = broadcast::channel(128);
        Self {
            snapshot: RwLock::new(initial),
            tx,
            baseband,
            baseband_health,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("========== RSHTML Server Starting ==========");

    // 初始化基带 API
    let (baseband, baseband_health) = match BasebandManager::initialize_with_health() {
        (Some(bb), health) => {
            info!("✓ Baseband API initialized successfully");
            let mut health = health;

            if health.effective_mode == "hardware-remote-bb-host" {
                health.socket_init.message = "Skipped in remote bb_host mode".to_string();
            } else {
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
            build_hardware_snapshot(0, status, &baseband.get_communication_stats(), None)
        }
        _ => build_simulated_snapshot(0),
    };
    let state = Arc::new(AppState::new(initial, baseband.clone(), baseband_health));

    spawn_data_feeder(state.clone());

    let app = Router::new()
        .route("/api/wireless/status", get(get_wireless_status))
        .route("/api/system/info", get(get_system_info))
        .route("/api/baseband/health", get(get_baseband_health))
        .route("/api/baseband/stats", get(get_baseband_stats))
        .route("/api/baseband/test", get(test_baseband_communication))
        .route("/ws", get(ws_handler))
        .nest_service("/", ServeDir::new("static").append_index_html_on_directories(true))
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

async fn get_system_info() -> Json<SystemInfo> {
    Json(SystemInfo {
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
    })
}

async fn get_baseband_health(State(state): State<Arc<AppState>>) -> Json<BasebandHealthStatus> {
    Json(state.baseband_health.clone())
}

async fn get_baseband_stats(State(state): State<Arc<AppState>>) -> Json<BasebandStatsResponse> {
    let response = match state.baseband.as_ref() {
        Some(baseband) => BasebandStatsResponse {
            available: true,
            stats: Some(baseband.get_communication_stats()),
            message: "Baseband statistics fetched successfully".to_string(),
        },
        None => BasebandStatsResponse {
            available: false,
            stats: None,
            message: "Baseband SDK not available; running in simulator mode".to_string(),
        },
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
        let mut ticker = tokio::time::interval(Duration::from_secs(1));
        let mut tick = 1_u64;

        loop {
            ticker.tick().await;

            let previous = state.snapshot.read().await.clone();
            let snapshot = match state.baseband.as_ref() {
                Some(baseband) => match baseband.get_status_snapshot() {
                    Ok(status) => {
                        let stats = baseband.get_communication_stats();
                        build_hardware_snapshot(tick, &status, &stats, Some(&previous))
                    }
                    Err(err) => {
                        if tick % 30 == 1 {
                            warn!("Failed to refresh wireless status from baseband: {}", err);
                        }
                        carry_forward_snapshot(previous, tick)
                    }
                },
                None => build_simulated_snapshot(tick),
            };

            {
                let mut guard = state.snapshot.write().await;
                *guard = snapshot.clone();
            }
            let _ = state.tx.send(snapshot);
            tick = tick.wrapping_add(1);
        }
    });
}

fn build_simulated_snapshot(sequence: u64) -> WirelessSnapshot {
    let main_rssi = -72 + oscillate(sequence, 5, 17) - oscillate(sequence / 3, 3, 11);
    let aux_rssi = -76 + oscillate(sequence + 5, 4, 19) - oscillate(sequence / 2, 2, 7);
    let peer_main_rssi = -65 + oscillate(sequence + 4, 4, 13);
    let peer_aux_rssi = -69 + oscillate(sequence + 7, 5, 15);
    let current_main_rssi = main_rssi.clamp(-79, -58);
    let current_aux_rssi = aux_rssi.clamp(-83, -60);
    let peer_current_main_rssi = peer_main_rssi.clamp(-74, -52);
    let peer_current_aux_rssi = peer_aux_rssi.clamp(-78, -56);

    let main_history = build_history(sequence, current_main_rssi, HISTORY_POINTS, 6);
    let aux_history = build_history(sequence + 3, current_aux_rssi, HISTORY_POINTS, 5);
    let peer_main_history = build_history(sequence + 9, peer_current_main_rssi, HISTORY_POINTS, 5);
    let peer_aux_history = build_history(sequence + 12, peer_current_aux_rssi, HISTORY_POINTS, 4);

    let receive_bytes = 3_965_000 + sequence * 18_400;
    let transmit_bytes = 10_085_000 + sequence * 24_900;

    WirelessSnapshot {
        sequence,
        general: GeneralStatus {
            mac_address: "00:0F:92:FA:37:CE".to_string(),
            operation_mode: "Master".to_string(),
            network_id: "TEST_ID".to_string(),
            compatibility_mode: "PDDL".to_string(),
            bandwidth: "4 MHz".to_string(),
            frequency: format!("2.{:03} GHz", 438 + ((sequence % 5) as u16)),
            tx_power: format!("{} dBm", 20 + (sequence % 2)),
            encryption_type: "AES-128".to_string(),
        },
        traffic: TrafficStatus {
            receive_bytes,
            receive_packets: 42_117 + sequence * 33,
            transmit_bytes,
            transmit_packets: 65_437 + sequence * 47,
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
            rssi_main_dbm: peer_current_main_rssi,
            rssi_aux_dbm: peer_current_aux_rssi,
            signal_level_main: map_signal_level(peer_current_main_rssi),
            signal_level_aux: map_signal_level(peer_current_aux_rssi),
            rssi_main_history: peer_main_history,
            rssi_aux_history: peer_aux_history,
        }],
        chart: RssiChart {
            target_mac_address: "00:0F:92:FA:37:C5".to_string(),
            primary_label: "RSSI Main".to_string(),
            secondary_label: "RSSI Aux".to_string(),
            current_primary_rssi_dbm: current_main_rssi,
            current_secondary_rssi_dbm: current_aux_rssi,
            min_rssi_dbm: main_history
                .iter()
                .chain(aux_history.iter())
                .copied()
                .min()
                .unwrap_or(current_main_rssi.min(current_aux_rssi)),
            max_rssi_dbm: main_history
                .iter()
                .chain(aux_history.iter())
                .copied()
                .max()
                .unwrap_or(current_main_rssi.max(current_aux_rssi)),
            primary_points: main_history,
            secondary_points: aux_history,
        },
    }
}

fn build_hardware_snapshot(
    sequence: u64,
    status: &BbGetStatusSummary,
    stats: &CommunicationStats,
    previous: Option<&WirelessSnapshot>,
) -> WirelessSnapshot {
    let local_main_history = history_from_previous(
        previous.map(|snapshot| snapshot.chart.primary_points.as_slice()),
        RSSI_UNAVAILABLE_DBM,
        HISTORY_POINTS,
    );
    let local_aux_history = history_from_previous(
        previous.map(|snapshot| snapshot.chart.secondary_points.as_slice()),
        RSSI_UNAVAILABLE_DBM,
        HISTORY_POINTS,
    );
    let connections = status
        .links
        .iter()
        .map(|link| ConnectionStatus {
            link_slot: format!("SLOT {}", link.slot),
            link_state: format_link_state(link.state).to_string(),
            pair_state: if link.pair_state {
                "Pairing".to_string()
            } else {
                "Stable".to_string()
            },
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
            snr_db: SNR_UNAVAILABLE_DB,
            rssi_main_dbm: RSSI_UNAVAILABLE_DBM,
            rssi_aux_dbm: RSSI_UNAVAILABLE_DBM,
            signal_level_main: map_signal_level(RSSI_UNAVAILABLE_DBM),
            signal_level_aux: map_signal_level(RSSI_UNAVAILABLE_DBM),
            rssi_main_history: vec![RSSI_UNAVAILABLE_DBM; HISTORY_POINTS],
            rssi_aux_history: vec![RSSI_UNAVAILABLE_DBM; HISTORY_POINTS],
        })
        .collect::<Vec<_>>();
    let peer_mac = connections
        .first()
        .map(|connection| connection.mac_address.clone())
        .unwrap_or_else(|| "Unavailable".to_string());

    WirelessSnapshot {
        sequence,
        general: GeneralStatus {
            mac_address: status.mac_hex.clone(),
            operation_mode: format_operation_mode(status),
            network_id: format!(
                "Unavailable (cfg_sbmp=0x{:02X}, rt_sbmp=0x{:02X})",
                status.cfg_sbmp, status.rt_sbmp
            ),
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
            encryption_type: "Unavailable".to_string(),
        },
        traffic: TrafficStatus {
            receive_bytes: stats.recv_bytes,
            receive_packets: stats.recv_packets as u64,
            transmit_bytes: stats.send_bytes,
            transmit_packets: stats.send_packets as u64,
        },
        connections,
        chart: RssiChart {
            target_mac_address: peer_mac,
            primary_label: "RSSI Main".to_string(),
            secondary_label: "RSSI Aux".to_string(),
            current_primary_rssi_dbm: RSSI_UNAVAILABLE_DBM,
            current_secondary_rssi_dbm: RSSI_UNAVAILABLE_DBM,
            min_rssi_dbm: RSSI_UNAVAILABLE_DBM,
            max_rssi_dbm: RSSI_UNAVAILABLE_DBM,
            primary_points: local_main_history,
            secondary_points: local_aux_history,
        },
    }
}

fn carry_forward_snapshot(mut previous: WirelessSnapshot, sequence: u64) -> WirelessSnapshot {
    previous.sequence = sequence;
    previous
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

fn oscillate(step: u64, amplitude: i32, period: u64) -> i32 {
    let cycle = (step % period) as i32;
    let pivot = (period as i32) / 2;
    let distance = (cycle - pivot).abs();
    amplitude - (distance * amplitude / pivot.max(1))
}

fn map_signal_level(rssi_dbm: i32) -> u8 {
    match rssi_dbm {
        i32::MIN..=-100 => 0,
        -58..=-1 => 4,
        -64..=-59 => 3,
        -70..=-65 => 2,
        _ => 1,
    }
}

fn format_operation_mode(status: &BbGetStatusSummary) -> String {
    let role = match status.role {
        0 => "AP",
        1 => "DEV",
        _ => "Unknown",
    };

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
