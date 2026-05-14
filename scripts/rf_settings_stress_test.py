from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


RUNTIME_URL = "/api/wireless/runtime"
APPLY_URL = "/api/wireless/runtime/apply"
CONFIGURATION_URL = "/api/wireless/configuration"
MIN_ITERATIONS = 5
MAX_ITERATIONS = 20


@dataclass
class ActionStep:
    action: str
    label: str
    payload: dict[str, Any]
    verification_key: str | None = None
    verification_scope: str = "runtime"


@dataclass
class ActionResult:
    phase: str
    iteration: int
    device_label: str
    device_mac: str
    device_role: str
    action: str
    label: str
    payload: dict[str, Any]
    http_status: int
    response_success: bool
    response_message: str
    followup_runtime_available: bool
    followup_runtime_message: str | None
    verification_scope: str
    verification_key: str | None
    verification_expected: Any = None
    verification_observed: Any = None
    verification_matched: bool | None = None
    outcome: str = ""
    response_current: dict[str, Any] | None = None
    followup_current: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class DeviceStressResult:
    device_label: str
    device_mac: str
    device_role: str
    baseline: dict[str, Any]
    actions: list[ActionResult] = field(default_factory=list)


@dataclass
class StressSummary:
    total_actions: int = 0
    apply_success_count: int = 0
    apply_failure_count: int = 0
    board_failure_count: int = 0
    readback_failed_count: int = 0
    apply_success_readback_mismatch_count: int = 0
    verified_match_count: int = 0
    verified_mismatch_count: int = 0
    runtime_unavailable_count: int = 0


@dataclass
class AggregateRow:
    phase: str
    action: str | None = None
    device_label: str | None = None
    device_role: str | None = None
    total_actions: int = 0
    apply_success_count: int = 0
    apply_failure_count: int = 0
    board_failure_count: int = 0
    readback_failed_count: int = 0
    apply_success_readback_mismatch_count: int = 0
    verified_match_count: int = 0
    verified_mismatch_count: int = 0
    runtime_unavailable_count: int = 0


@dataclass
class StressResult:
    base_url: str
    iterations: int
    pause_ms: int
    devices: list[DeviceStressResult] = field(default_factory=list)
    request_failures: list[str] = field(default_factory=list)
    summary: StressSummary = field(default_factory=StressSummary)
    stress_by_action: list[AggregateRow] = field(default_factory=list)
    stress_by_device: list[AggregateRow] = field(default_factory=list)
    stress_by_device_action: list[AggregateRow] = field(default_factory=list)


def build_url(base_url: str, path: str) -> str:
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def normalize_mac(value: str | None) -> str:
    return "".join(ch for ch in str(value or "") if ch.lower() in "0123456789abcdef").upper()


def parse_iterations(value: str) -> int:
    parsed = int(value)
    if parsed < MIN_ITERATIONS or parsed > MAX_ITERATIONS:
        raise argparse.ArgumentTypeError(
            f"iterations must be between {MIN_ITERATIONS} and {MAX_ITERATIONS}"
        )
    return parsed


def request_json(base_url: str, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(build_url(base_url, path), data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            text = response.read().decode("utf-8")
            return response.getcode(), json.loads(text) if text else {}
    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {"success": False, "message": text}
        return error.code, payload


def get_runtime(base_url: str) -> tuple[int, dict[str, Any]]:
    return request_json(base_url, RUNTIME_URL)


def get_configuration(base_url: str, mode: int = 0) -> tuple[int, dict[str, Any]]:
    query = urllib.parse.urlencode({"mode": mode})
    return request_json(base_url, f"{CONFIGURATION_URL}?{query}")


def get_runtime_current(payload: dict[str, Any]) -> dict[str, Any] | None:
    current = payload.get("current")
    return current if isinstance(current, dict) else None


def get_runtime_devices(payload: dict[str, Any]) -> list[dict[str, Any]]:
    devices = payload.get("available_devices")
    if isinstance(devices, list):
        return [device for device in devices if isinstance(device, dict)]

    current = get_runtime_current(payload)
    if current and isinstance(current.get("available_devices"), list):
        return [device for device in current["available_devices"] if isinstance(device, dict)]

    return []


def resolve_available_device_role(device: dict[str, Any]) -> str:
    role = str(device.get("role") or device.get("role_label") or "").upper()
    label = str(device.get("label") or "").upper()
    if "DEV" in role or label.startswith("DEV:"):
        return "DEV"
    if "AP" in role or label.startswith("AP:"):
        return "AP"
    return "UNKNOWN"


def describe_available_device(device: dict[str, Any]) -> str:
    label = str(device.get("label") or "Unknown")
    mac = normalize_mac(device.get("mac_address") or device.get("local_mac_address"))
    role = resolve_available_device_role(device)
    return f"{label}[role={role}, mac={mac or 'unknown'}]"


def validate_runtime_topology(payload: dict[str, Any]) -> list[dict[str, Any]]:
    devices = get_runtime_devices(payload)
    if len(devices) != 2:
        labels = ", ".join(describe_available_device(device) for device in devices) or "none"
        raise RuntimeError(
            "RF settings stress requires exactly 2 available devices arranged as 1 AP + 1 DEV; "
            f"runtime reported {len(devices)} device(s): {labels}"
        )

    ap_count = 0
    dev_count = 0
    for device in devices:
        role = resolve_available_device_role(device)
        if role == "AP":
            ap_count += 1
        elif role == "DEV":
            dev_count += 1

    if ap_count != 1 or dev_count != 1:
        labels = ", ".join(describe_available_device(device) for device in devices)
        raise RuntimeError(
            "RF settings stress requires exactly 1 AP and 1 DEV before execution; "
            f"runtime reported AP={ap_count}, DEV={dev_count}: {labels}"
        )

    return devices


def runtime_summary(current: dict[str, Any] | None) -> dict[str, Any] | None:
    if not current:
        return None

    configured_band = current.get("configured_band") or {}
    live_rf = current.get("live_rf") or {}
    return {
        "local_mac_address": current.get("local_mac_address"),
        "operation_mode": current.get("operation_mode"),
        "band_bitmap": configured_band.get("bitmap", current.get("band_bitmap")),
        "configured_band_label": configured_band.get("label"),
        "channel_auto": current.get("channel_auto"),
        "work_channel_index": current.get("work_channel_index"),
        "work_channel_frequency": current.get("work_channel_frequency") or live_rf.get("channel_frequency"),
        "bandwidth_auto": current.get("bandwidth_auto"),
        "bandwidth_code": current.get("bandwidth_code"),
        "power_auto": current.get("current_power_auto"),
        "power_mode": current.get("current_power_mode"),
        "power_dbm": current.get("current_power_dbm"),
        "mcs_auto": current.get("current_mcs_auto"),
        "mcs_value": current.get("current_mcs_value"),
        "current_slot": current.get("current_slot"),
        "current_power_user": current.get("current_power_user"),
    }


def resolve_runtime_role(current: dict[str, Any]) -> str:
    mode = str(current.get("operation_mode") or "").upper()
    if "DEV" in mode:
        return "DEV"
    if "AP" in mode:
        return "AP"
    return "UNKNOWN"


def choose_distinct_values(values: list[int], fallback: int, *, limit: int = 3) -> list[int]:
    unique: list[int] = []
    for value in values:
        if value not in unique:
            unique.append(value)
        if len(unique) >= limit:
            return unique

    if fallback not in unique:
        unique.append(fallback)
    return unique[:limit]


def parse_frequency_band(frequency_text: Any) -> str | None:
    token = str(frequency_text or "").strip().split(" ", 1)[0]
    try:
        value = float(token)
    except ValueError:
        return None

    if 0.5 <= value < 1.0:
        return "600m"
    if 2.0 <= value < 3.0:
        return "2g"
    if 5.0 <= value < 6.0:
        return "5g"
    return None


def allowed_bands_for_bitmap(band_bitmap: int | None) -> set[str] | None:
    mapping = {
        0x01: {"600m"},
        0x02: {"2g"},
        0x04: {"5g"},
        0x07: {"600m", "2g", "5g"},
    }
    return mapping.get(band_bitmap)


def candidate_channel_indices(current: dict[str, Any], band_bitmap: int | None = None) -> list[int]:
    allowed_bands = allowed_bands_for_bitmap(band_bitmap)
    channel_entries = [
        entry
        for entry in current.get("channels", [])
        if isinstance(entry, dict) and isinstance(entry.get("index"), int)
    ]
    if allowed_bands is not None:
        filtered_entries = [
            entry
            for entry in channel_entries
            if parse_frequency_band(entry.get("frequency")) in allowed_bands
        ]
    else:
        filtered_entries = channel_entries

    if not filtered_entries:
        return []

    filtered_indices = [int(entry["index"]) for entry in filtered_entries]
    current_index = current.get("work_channel_index")
    fallback = current_index if isinstance(current_index, int) and current_index in filtered_indices else filtered_indices[0]
    return choose_distinct_values([fallback, filtered_indices[0], filtered_indices[-1]], fallback)


def resolve_power_range(configuration_payload: dict[str, Any]) -> tuple[int, int] | None:
    current = configuration_payload.get("current")
    if not isinstance(current, dict):
        return None

    minidb = current.get("minidb")
    if isinstance(minidb, dict):
        power = minidb.get("power")
        if isinstance(power, dict):
            min_dbm = power.get("min_dbm")
            max_dbm = power.get("max_dbm")
            if isinstance(min_dbm, int) and isinstance(max_dbm, int):
                return min(min_dbm, max_dbm), max(min_dbm, max_dbm)

    config_text = current.get("config_text")
    if not isinstance(config_text, str) or not config_text.strip():
        return None

    try:
        config_payload = json.loads(config_text)
    except json.JSONDecodeError:
        return None

    baseband = config_payload.get("baseband")
    if not isinstance(baseband, dict):
        return None

    basic = baseband.get("basic")
    if not isinstance(basic, dict):
        return None

    power = basic.get("power")
    if not isinstance(power, dict):
        return None

    power_range = power.get("pwr_range")
    if not isinstance(power_range, list) or len(power_range) < 2:
        return None

    try:
        min_dbm = int(power_range[0])
        max_dbm = int(power_range[1])
    except (TypeError, ValueError):
        return None

    return min(min_dbm, max_dbm), max(min_dbm, max_dbm)


def candidate_power_values(current: dict[str, Any], power_range: tuple[int, int] | None = None) -> list[int]:
    current_power = current.get("current_power_dbm")
    if not isinstance(current_power, int):
        current_power = 25

    if power_range is not None:
        min_power, max_power = power_range
        midpoint = (min_power + max_power) // 2
        current_power = min(max(current_power, min_power), max_power)
        values = [current_power, min_power, midpoint, max_power]
    else:
        values = [max(0, current_power - 3), current_power, min(30, current_power + 3)]
    return choose_distinct_values(values, current_power)


def candidate_mcs_values(current: dict[str, Any]) -> list[int]:
    current_mcs = current.get("current_mcs_value")
    if not isinstance(current_mcs, int):
        current_mcs = 2
    values = [max(0, current_mcs - 1), current_mcs, min(24, current_mcs + 1)]
    return choose_distinct_values(values, current_mcs)


def candidate_bandwidth_values(current: dict[str, Any]) -> list[int]:
    current_bandwidth = current.get("bandwidth_code")
    if not isinstance(current_bandwidth, int):
        current_bandwidth = 3
    values = [max(0, current_bandwidth - 1), current_bandwidth, min(4, current_bandwidth + 1)]
    return choose_distinct_values(values, current_bandwidth)


def normalized_power_mode(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if "open" in text:
        return "openloop"
    if "close" in text:
        return "closeloop"
    return text


def build_stress_plan(
    current: dict[str, Any],
    iteration: int,
    power_range: tuple[int, int] | None = None,
) -> list[ActionStep]:
    band_bitmaps = [0x07, 0x02, 0x04, 0x01]
    target_band_bitmap = band_bitmaps[iteration % len(band_bitmaps)]
    channel_indices = candidate_channel_indices(current, target_band_bitmap)
    power_values = candidate_power_values(current, power_range)
    mcs_values = candidate_mcs_values(current)
    bandwidth_values = candidate_bandwidth_values(current)
    slot = int(current.get("current_slot") or 0)
    user = int(current.get("current_power_user") or 0)
    power_mode = normalized_power_mode(current.get("current_power_mode"))
    next_power_mode = "openloop" if power_mode == "closeloop" else "closeloop"

    steps = [
        ActionStep(
            action="set_band_selection",
            label=f"Band bitmap -> 0x{target_band_bitmap:02X}",
            payload={"band_bitmap": target_band_bitmap},
            verification_key="band_bitmap",
        ),
        ActionStep(
            action="set_channel_mode",
            label="Channel Auto -> Off",
            payload={"auto_mode": False},
            verification_key="channel_auto",
        ),
        ActionStep(
            action="set_channel_mode",
            label="Channel Auto -> On",
            payload={"auto_mode": True},
            verification_key="channel_auto",
        ),
        ActionStep(
            action="set_power_auto",
            label="Power Auto -> Off",
            payload={"auto_mode": False},
            verification_key="current_power_auto",
        ),
        ActionStep(
            action="set_power_mode",
            label=f"Power Mode -> {next_power_mode}",
            payload={"power_mode": next_power_mode},
            verification_key="current_power_mode",
        ),
        ActionStep(
            action="set_power",
            label=f"Power -> {power_values[iteration % len(power_values)]} dB",
            payload={"user": user, "power_dbm": power_values[iteration % len(power_values)]},
            verification_key="current_power_dbm",
        ),
        ActionStep(
            action="set_power_auto",
            label="Power Auto -> On",
            payload={"auto_mode": True},
            verification_key="current_power_auto",
        ),
        ActionStep(
            action="set_mcs_mode",
            label="MCS Auto -> Off",
            payload={"slot": slot, "auto_mode": False},
            verification_key="current_mcs_auto",
        ),
        ActionStep(
            action="set_mcs",
            label=f"MCS -> {mcs_values[iteration % len(mcs_values)]}",
            payload={"slot": slot, "mcs": mcs_values[iteration % len(mcs_values)]},
            verification_key="current_mcs_value",
        ),
        ActionStep(
            action="set_mcs_mode",
            label="MCS Auto -> On",
            payload={"slot": slot, "auto_mode": True},
            verification_key="current_mcs_auto",
        ),
        ActionStep(
            action="set_bandwidth_mode",
            label="Bandwidth Auto -> Off",
            payload={"slot": slot, "auto_mode": False},
            verification_key="bandwidth_auto",
            verification_scope="service_cache",
        ),
        ActionStep(
            action="set_bandwidth",
            label=f"Bandwidth -> {bandwidth_values[iteration % len(bandwidth_values)]}",
            payload={"slot": slot, "direction": "rx", "bandwidth": bandwidth_values[iteration % len(bandwidth_values)]},
            verification_key="bandwidth_code",
        ),
        ActionStep(
            action="set_bandwidth_mode",
            label="Bandwidth Auto -> On",
            payload={"slot": slot, "auto_mode": True},
            verification_key="bandwidth_auto",
            verification_scope="service_cache",
        ),
    ]

    if channel_indices:
        steps.insert(
            2,
            ActionStep(
                action="set_channel",
                label=f"Channel Index -> {channel_indices[iteration % len(channel_indices)]}",
                payload={"direction": "rx", "channel_index": channel_indices[iteration % len(channel_indices)]},
                verification_key="work_channel_index",
            ),
        )

    return steps


def build_restore_plan(current: dict[str, Any]) -> list[ActionStep]:
    steps: list[ActionStep] = []

    band_bitmap = current.get("configured_band", {}).get("bitmap", current.get("band_bitmap"))
    if isinstance(band_bitmap, int):
        steps.append(
            ActionStep(
                action="set_band_selection",
                label=f"Restore Band bitmap -> 0x{band_bitmap:02X}",
                payload={"band_bitmap": band_bitmap},
                verification_key="band_bitmap",
            )
        )

    channel_auto = current.get("channel_auto")
    if isinstance(channel_auto, bool):
        steps.append(
            ActionStep(
                action="set_channel_mode",
                label=f"Restore Channel Auto -> {channel_auto}",
                payload={"auto_mode": channel_auto},
                verification_key="channel_auto",
            )
        )
        if not channel_auto and isinstance(current.get("work_channel_index"), int):
            steps.append(
                ActionStep(
                    action="set_channel",
                    label=f"Restore Channel Index -> {current['work_channel_index']}",
                    payload={"direction": "rx", "channel_index": current["work_channel_index"]},
                    verification_key="work_channel_index",
                )
            )

    power_auto = current.get("current_power_auto")
    if isinstance(power_auto, bool):
        steps.append(
            ActionStep(
                action="set_power_auto",
                label=f"Restore Power Auto -> {power_auto}",
                payload={"auto_mode": power_auto},
                verification_key="current_power_auto",
            )
        )
        if not power_auto:
            power_mode = normalized_power_mode(current.get("current_power_mode")) or "closeloop"
            steps.append(
                ActionStep(
                    action="set_power_mode",
                    label=f"Restore Power Mode -> {power_mode}",
                    payload={"power_mode": power_mode},
                    verification_key="current_power_mode",
                )
            )
            if isinstance(current.get("current_power_dbm"), int):
                steps.append(
                    ActionStep(
                        action="set_power",
                        label=f"Restore Power -> {current['current_power_dbm']} dB",
                        payload={
                            "user": int(current.get("current_power_user") or 0),
                            "power_dbm": current["current_power_dbm"],
                        },
                        verification_key="current_power_dbm",
                    )
                )

    mcs_auto = current.get("current_mcs_auto")
    if isinstance(mcs_auto, bool):
        steps.append(
            ActionStep(
                action="set_mcs_mode",
                label=f"Restore MCS Auto -> {mcs_auto}",
                payload={"slot": int(current.get("current_slot") or 0), "auto_mode": mcs_auto},
                verification_key="current_mcs_auto",
            )
        )
        if not mcs_auto and isinstance(current.get("current_mcs_value"), int):
            steps.append(
                ActionStep(
                    action="set_mcs",
                    label=f"Restore MCS -> {current['current_mcs_value']}",
                    payload={"slot": int(current.get("current_slot") or 0), "mcs": current["current_mcs_value"]},
                    verification_key="current_mcs_value",
                )
            )

    bandwidth_auto = current.get("bandwidth_auto")
    if isinstance(bandwidth_auto, bool):
        steps.append(
            ActionStep(
                action="set_bandwidth_mode",
                label=f"Restore Bandwidth Auto -> {bandwidth_auto}",
                payload={"slot": int(current.get("current_slot") or 0), "auto_mode": bandwidth_auto},
                verification_key="bandwidth_auto",
                verification_scope="service_cache",
            )
        )
        if not bandwidth_auto and isinstance(current.get("bandwidth_code"), int):
            steps.append(
                ActionStep(
                    action="set_bandwidth",
                    label=f"Restore Bandwidth -> {current['bandwidth_code']}",
                    payload={
                        "slot": int(current.get("current_slot") or 0),
                        "direction": "rx",
                        "bandwidth": current["bandwidth_code"],
                    },
                    verification_key="bandwidth_code",
                )
            )

    return steps


def lookup_value(current: dict[str, Any] | None, verification_key: str | None) -> Any:
    if not current or not verification_key:
        return None

    mapping = {
        "band_bitmap": current.get("configured_band", {}).get("bitmap", current.get("band_bitmap")),
        "channel_auto": current.get("channel_auto"),
        "work_channel_index": current.get("work_channel_index"),
        "current_power_auto": current.get("current_power_auto"),
        "current_power_mode": normalized_power_mode(current.get("current_power_mode")),
        "current_power_dbm": current.get("current_power_dbm"),
        "current_mcs_auto": current.get("current_mcs_auto"),
        "current_mcs_value": current.get("current_mcs_value"),
        "bandwidth_auto": current.get("bandwidth_auto"),
        "bandwidth_code": current.get("bandwidth_code"),
    }
    return mapping.get(verification_key)


def expected_value(step: ActionStep) -> Any:
    if step.verification_key == "current_power_mode":
        return normalized_power_mode(step.payload.get("power_mode"))
    if step.verification_key == "band_bitmap":
        return step.payload.get("band_bitmap")
    if step.verification_key == "work_channel_index":
        return step.payload.get("channel_index")
    if step.verification_key == "current_power_dbm":
        return step.payload.get("power_dbm")
    if step.verification_key == "current_mcs_value":
        return step.payload.get("mcs")
    if step.verification_key == "bandwidth_code":
        return step.payload.get("bandwidth")
    if step.verification_key in {"channel_auto", "current_power_auto", "current_mcs_auto", "bandwidth_auto"}:
        return step.payload.get("auto_mode")
    return None


def classify_failure(message: str) -> str:
    text = message.lower()
    if "failed with code" in text:
        return "board_failed"
    if "did not take effect" in text:
        return "readback_failed"
    if "unsupported" in text:
        return "unsupported"
    return "apply_failed"


def poll_runtime(base_url: str, *, expected_mac: str | None = None, attempts: int = 5, delay_s: float = 0.4) -> tuple[int, dict[str, Any]]:
    last_status = 0
    last_payload: dict[str, Any] = {}
    normalized_expected = normalize_mac(expected_mac)
    for _ in range(attempts):
        last_status, last_payload = get_runtime(base_url)
        current = get_runtime_current(last_payload)
        if current and last_payload.get("available"):
            if not normalized_expected or normalize_mac(current.get("local_mac_address")) == normalized_expected:
                return last_status, last_payload
        time.sleep(delay_s)
    return last_status, last_payload


def execute_step(
    base_url: str,
    *,
    device_label: str,
    device_mac: str,
    device_role: str,
    phase: str,
    iteration: int,
    step: ActionStep,
    pause_ms: int,
) -> ActionResult:
    http_status, response = request_json(base_url, APPLY_URL, method="POST", payload={"action": step.action, **step.payload})
    response_success = bool(response.get("success"))
    response_message = str(response.get("message") or "")
    response_current = response.get("current") if isinstance(response.get("current"), dict) else None

    time.sleep(max(pause_ms, 0) / 1000.0)
    runtime_status, runtime_payload = poll_runtime(base_url, expected_mac=device_mac)
    del runtime_status
    followup_current = get_runtime_current(runtime_payload)
    followup_available = bool(runtime_payload.get("available") and followup_current)
    if not response_success and response_current is not None:
        current_for_verify = response_current
    else:
        current_for_verify = followup_current or response_current
    expected = expected_value(step)
    observed = lookup_value(current_for_verify, step.verification_key)

    verification_matched: bool | None = None
    notes: list[str] = []
    if step.verification_key is not None:
        if current_for_verify is None:
            notes.append("runtime current unavailable after apply")
        elif expected is None:
            notes.append("no expected verification value was derived")
        elif observed is None:
            notes.append("verification field unavailable in runtime readback")
        else:
            verification_matched = observed == expected

    if not response_success:
        outcome = classify_failure(response_message)
    elif step.verification_key is None:
        outcome = "apply_succeeded_unverified"
    elif verification_matched is True:
        outcome = "apply_succeeded_readback_matched"
    elif verification_matched is False:
        outcome = "apply_succeeded_readback_mismatch"
    elif not followup_available:
        outcome = "apply_succeeded_runtime_unavailable"
    else:
        outcome = "apply_succeeded_unverified"

    return ActionResult(
        phase=phase,
        iteration=iteration,
        device_label=device_label,
        device_mac=device_mac,
        device_role=device_role,
        action=step.action,
        label=step.label,
        payload=step.payload,
        http_status=http_status,
        response_success=response_success,
        response_message=response_message,
        followup_runtime_available=followup_available,
        followup_runtime_message=str(runtime_payload.get("message") or "") if isinstance(runtime_payload, dict) else None,
        verification_scope=step.verification_scope,
        verification_key=step.verification_key,
        verification_expected=expected,
        verification_observed=observed,
        verification_matched=verification_matched,
        outcome=outcome,
        response_current=runtime_summary(response_current),
        followup_current=runtime_summary(followup_current),
        notes=notes,
    )


def select_device(base_url: str, device: dict[str, Any]) -> dict[str, Any]:
    target_mac = normalize_mac(device.get("mac_address"))
    if not target_mac:
        raise RuntimeError(f"Device has no usable MAC address: {device}")

    status, response = request_json(
        base_url,
        APPLY_URL,
        method="POST",
        payload={"action": "select_device", "device_mac": target_mac},
    )
    if status >= 400 or not response.get("success"):
        raise RuntimeError(f"Failed to select device {device.get('label')}: {response.get('message')}")

    _, runtime_payload = poll_runtime(base_url, expected_mac=target_mac, attempts=10, delay_s=0.6)
    current = get_runtime_current(runtime_payload)
    if not current or normalize_mac(current.get("local_mac_address")) != target_mac:
        raise RuntimeError(f"Selected device {device.get('label')} did not become active")
    return current


def accumulate_stats(target: StressSummary | AggregateRow, action: ActionResult) -> None:
    target.total_actions += 1
    if action.response_success:
        target.apply_success_count += 1
    else:
        target.apply_failure_count += 1
        if action.outcome == "board_failed":
            target.board_failure_count += 1

    if action.outcome == "readback_failed":
        target.readback_failed_count += 1
    elif action.outcome == "apply_succeeded_readback_mismatch":
        target.apply_success_readback_mismatch_count += 1

    if action.verification_matched is True:
        target.verified_match_count += 1
    elif action.verification_matched is False:
        target.verified_mismatch_count += 1

    if not action.followup_runtime_available:
        target.runtime_unavailable_count += 1


def update_summary(summary: StressSummary, action: ActionResult) -> None:
    accumulate_stats(summary, action)


def build_stress_aggregates(result: StressResult) -> None:
    by_action: dict[str, AggregateRow] = {}
    by_device: dict[tuple[str, str], AggregateRow] = {}
    by_device_action: dict[tuple[str, str, str], AggregateRow] = {}

    for device in result.devices:
        for action in device.actions:
            if action.phase != "stress":
                continue

            action_row = by_action.setdefault(
                action.action,
                AggregateRow(phase="stress", action=action.action),
            )
            accumulate_stats(action_row, action)

            device_key = (action.device_label, action.device_role)
            device_row = by_device.setdefault(
                device_key,
                AggregateRow(
                    phase="stress",
                    device_label=action.device_label,
                    device_role=action.device_role,
                ),
            )
            accumulate_stats(device_row, action)

            device_action_key = (action.device_label, action.device_role, action.action)
            device_action_row = by_device_action.setdefault(
                device_action_key,
                AggregateRow(
                    phase="stress",
                    action=action.action,
                    device_label=action.device_label,
                    device_role=action.device_role,
                ),
            )
            accumulate_stats(device_action_row, action)

    result.stress_by_action = sorted(by_action.values(), key=lambda row: row.action or "")
    result.stress_by_device = sorted(
        by_device.values(),
        key=lambda row: (row.device_label or "", row.device_role or ""),
    )
    result.stress_by_device_action = sorted(
        by_device_action.values(),
        key=lambda row: (row.device_label or "", row.action or ""),
    )


def run_stress(base_url: str, iterations: int, pause_ms: int) -> StressResult:
    result = StressResult(base_url=base_url, iterations=iterations, pause_ms=pause_ms)

    _, runtime_payload = get_runtime(base_url)
    if not runtime_payload.get("available") or not get_runtime_current(runtime_payload):
        raise RuntimeError(f"Wireless runtime is unavailable: {runtime_payload.get('message')}")

    available_devices = validate_runtime_topology(runtime_payload)

    for device in available_devices:
        current = select_device(base_url, device)
        _, configuration_payload = get_configuration(base_url)
        power_range = resolve_power_range(configuration_payload)
        device_label = str(device.get("label") or current.get("local_mac_address") or "Unknown device")
        device_mac = normalize_mac(current.get("local_mac_address"))
        device_role = resolve_runtime_role(current)
        baseline = runtime_summary(current) or {}
        if power_range is not None:
            baseline["power_range_dbm"] = {"min": power_range[0], "max": power_range[1]}
        device_result = DeviceStressResult(
            device_label=device_label,
            device_mac=device_mac,
            device_role=device_role,
            baseline=baseline,
        )

        for iteration in range(iterations):
            for step in build_stress_plan(current, iteration, power_range):
                action_result = execute_step(
                    base_url,
                    device_label=device_label,
                    device_mac=device_mac,
                    device_role=device_role,
                    phase="stress",
                    iteration=iteration,
                    step=step,
                    pause_ms=pause_ms,
                )
                device_result.actions.append(action_result)
                update_summary(result.summary, action_result)

        for step in build_restore_plan(current):
            action_result = execute_step(
                base_url,
                device_label=device_label,
                device_mac=device_mac,
                device_role=device_role,
                phase="restore",
                iteration=iterations,
                step=step,
                pause_ms=pause_ms,
            )
            device_result.actions.append(action_result)
            update_summary(result.summary, action_result)

        result.devices.append(device_result)

    build_stress_aggregates(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/")
    parser.add_argument("--iterations", type=parse_iterations, default=MIN_ITERATIONS)
    parser.add_argument("--pause-ms", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("rf-settings-stress-result.json"))
    args = parser.parse_args()

    try:
        result = run_stress(args.url, args.iterations, max(0, args.pause_ms))
        text = json.dumps(asdict(result), indent=2, ensure_ascii=False)
        args.output.write_text(text, encoding="utf-8")
        print(text)
        return 0
    except Exception as error:
        payload = {"fatal_error": str(error)}
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        args.output.write_text(text, encoding="utf-8")
        print(text)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())