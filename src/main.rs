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
#[derive(Debug, Clone, Serialize)]
struct SystemInfo {
    host_name: String,
    product_name: String,
    description: String,
    system_date: String,
    system_uptime: String,
    build_date: String,
    build_time: String,
    mac_address: String,
    ip_address: String,
    subnet_mask: String,
    connection_type: String,
    gateway: String,
}

use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use tokio::sync::{broadcast, RwLock};
use tower_http::{cors::CorsLayer, services::ServeDir};
use tracing::{error, info};

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

struct AppState {
    snapshot: RwLock<WirelessSnapshot>,
    tx: broadcast::Sender<WirelessSnapshot>,
}

impl AppState {
    fn new(initial: WirelessSnapshot) -> Self {
        let (tx, _) = broadcast::channel(128);
        Self {
            snapshot: RwLock::new(initial),
            tx,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let initial = build_snapshot(0);
    let state = Arc::new(AppState::new(initial));

    spawn_data_feeder(state.clone());

    let app = Router::new()
        .route("/api/wireless/status", get(get_wireless_status))
        .route("/api/system/info", get(get_system_info))
        .route("/ws", get(ws_handler))
        .nest_service("/", ServeDir::new("static").append_index_html_on_directories(true))
        .layer(CorsLayer::permissive())
        .with_state(state);
async fn get_system_info() -> Json<SystemInfo> {
    Json(SystemInfo {
        host_name: "UserDevice".to_string(),
        product_name: "pDDL-MIMO".to_string(),
        description: "mypDDL-MIMO".to_string(),
        system_date: "2026-03-31 10:00:00".to_string(),
        system_uptime: "5 min".to_string(),
        build_date: "2026-03-31".to_string(),
        build_time: "09:00:00".to_string(),
        mac_address: "00:0F:92:FA:94:CF".to_string(),
        ip_address: "192.168.168.1".to_string(),
        subnet_mask: "255.255.255.0".to_string(),
        connection_type: "static".to_string(),
        gateway: "192.168.168.1".to_string(),
    })
}

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("wireless status server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn get_wireless_status(State(state): State<Arc<AppState>>) -> Json<WirelessSnapshot> {
    Json(state.snapshot.read().await.clone())
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

            let snapshot = build_snapshot(tick);
            {
                let mut guard = state.snapshot.write().await;
                *guard = snapshot.clone();
            }
            let _ = state.tx.send(snapshot);
            tick = tick.wrapping_add(1);
        }
    });
}

fn build_snapshot(sequence: u64) -> WirelessSnapshot {
    let main_rssi = -72 + oscillate(sequence, 5, 17) - oscillate(sequence / 3, 3, 11);
    let aux_rssi = -76 + oscillate(sequence + 5, 4, 19) - oscillate(sequence / 2, 2, 7);
    let peer_main_rssi = -65 + oscillate(sequence + 4, 4, 13);
    let peer_aux_rssi = -69 + oscillate(sequence + 7, 5, 15);
    let current_main_rssi = main_rssi.clamp(-79, -58);
    let current_aux_rssi = aux_rssi.clamp(-83, -60);
    let peer_current_main_rssi = peer_main_rssi.clamp(-74, -52);
    let peer_current_aux_rssi = peer_aux_rssi.clamp(-78, -56);

    let main_history = build_history(sequence, current_main_rssi, 18, 6);
    let aux_history = build_history(sequence + 3, current_aux_rssi, 18, 5);
    let peer_main_history = build_history(sequence + 9, peer_current_main_rssi, 18, 5);
    let peer_aux_history = build_history(sequence + 12, peer_current_aux_rssi, 18, 4);

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
        -58..=-1 => 4,
        -64..=-59 => 3,
        -70..=-65 => 2,
        _ => 1,
    }
}
