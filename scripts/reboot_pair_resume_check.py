import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from playwright.async_api import Error, TimeoutError as PlaywrightTimeoutError, async_playwright


def normalize_mac(value: str | None) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch in "0123456789abcdef")


@dataclass
class PairStateSnapshot:
    elapsed_ms: int
    runtime_mac: str
    runtime_mac_normalized: str
    selected_mac: str
    pair_state: str
    pair_button: str
    pair_value: str
    dev_pair_target_mac: str
    reboot_status: str
    connections: list[dict] = field(default_factory=list)


@dataclass
class IterationResult:
    iteration: int
    dev_mac: str
    success: bool
    reboot_response: dict
    transition_observed: bool
    reselected: bool
    pair_ok: bool
    pair_settled: bool = True
    failure_reason: str | None = None
    initial_state: PairStateSnapshot | None = None
    transition_timeline: list[PairStateSnapshot] = field(default_factory=list)
    pair_timeline: list[PairStateSnapshot] = field(default_factory=list)
    settle_timeline: list[PairStateSnapshot] = field(default_factory=list)


@dataclass
class RebootPairResumeResult:
    iterations: int
    completed_iterations: int = 0
    success: bool = False
    console_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    request_failures: list[str] = field(default_factory=list)
    response_failures: list[str] = field(default_factory=list)
    iteration_results: list[IterationResult] = field(default_factory=list)


async def goto_rf_settings(page, base_url: str) -> None:
    await page.goto(base_url, wait_until="networkidle")
    await page.locator('label[for="tab-wireless"]').click()
    await page.locator('[data-wireless-tab="rf"]').click()
    await page.locator("#rf-pair-list").wait_for(state="visible")


async def get_device_options(page) -> list[dict[str, str]]:
    return await page.eval_on_selector_all(
        "#rf-operation-mode option",
        "options => options.map(option => ({ value: option.value, label: (option.textContent || '').trim() })).filter(option => option.value)",
    )


async def get_dev_option(page) -> dict[str, str]:
    options = await get_device_options(page)
    for option in options:
        if option["label"].upper().startswith("DEV"):
            return option
    raise RuntimeError("Unable to find a DEV device in Active Device options")


async def select_device(page, target_mac: str, timeout_ms: int = 10000) -> None:
    normalized_target = normalize_mac(target_mac)
    await page.locator("#rf-operation-mode").select_option(target_mac)
    await page.wait_for_function(
        """
        target => {
            const value = document.querySelector('#runtime-mac')?.textContent || '';
            const normalized = String(value).toLowerCase().replace(/[^0-9a-f]/g, '');
            return normalized === target;
        }
        """,
        arg=normalized_target,
        timeout=timeout_ms,
    )


async def trigger_immediate_reboot(page) -> dict:
    return await page.evaluate(
        """
        async () => {
            const response = await fetch('/api/system/reboot', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ delay_seconds: 0 }),
            });
            return await response.json();
        }
        """
    )


async def capture_pair_state(page, started_at: float) -> PairStateSnapshot:
    payload = await page.evaluate(
        """
        async ({ startedAt }) => {
            const normalize = (value) => String(value || '').toLowerCase().replace(/[^0-9a-f]/g, '');
            const dot = document.querySelector('#rf-pair-list .rf-pair-row .rf-pair-dot');
            let pairState = 'missing';
            if (dot) {
                if (dot.classList.contains('pairing')) {
                    pairState = 'pairing';
                } else if (dot.classList.contains('paired')) {
                    pairState = 'paired';
                } else if (dot.classList.contains('idle')) {
                    pairState = 'idle';
                } else {
                    pairState = Array.from(dot.classList).join(' ');
                }
            }

            const button = document.querySelector('#rf-pair-list [data-pair-slot="0"]');
            const input = document.querySelector('#rf-pair-list [data-pair-mac-slot="0"]');
            const runtimeMac = String(document.querySelector('#runtime-mac')?.textContent || '').trim();
            const selectedMac = String(document.querySelector('#rf-operation-mode')?.value || '').trim();
            const rebootStatus = String(document.querySelector('#reboot-action-status')?.textContent || '').trim();
            const connections = Array.isArray(globalThis.latestWirelessSnapshot?.connections)
                ? globalThis.latestWirelessSnapshot.connections.map((connection) => ({
                    link_slot: connection.link_slot,
                    link_state: connection.link_state,
                    pair_state: connection.pair_state,
                    pairing_active: Boolean(connection.pairing_active),
                    mac_address: connection.mac_address,
                }))
                : [];

            return {
                elapsed_ms: Math.round(Date.now() - startedAt),
                runtime_mac: runtimeMac,
                runtime_mac_normalized: normalize(runtimeMac),
                selected_mac: normalize(selectedMac),
                pair_state: pairState,
                pair_button: String(button?.textContent || '').trim(),
                pair_value: String(input?.value || '').trim(),
                dev_pair_target_mac: String(globalThis.wirelessRuntimeCurrent?.dev_pair_target_mac || '').trim(),
                reboot_status: rebootStatus,
                connections,
            };
        }
        """,
        {"startedAt": started_at},
    )
    return PairStateSnapshot(**payload)


def timeline_append(timeline: list[PairStateSnapshot], state: PairStateSnapshot) -> None:
    if not timeline or asdict(timeline[-1]) != asdict(state):
        timeline.append(state)


async def wait_for_reboot_transition(page, target_mac: str, started_at: float, timeout_ms: int) -> tuple[bool, list[PairStateSnapshot]]:
    deadline = time.monotonic() + (timeout_ms / 1000)
    timeline: list[PairStateSnapshot] = []

    while time.monotonic() < deadline:
        state = await capture_pair_state(page, started_at)
        timeline_append(timeline, state)
        if not state.runtime_mac_normalized or state.runtime_mac_normalized != normalize_mac(target_mac):
            return True, timeline
        await page.wait_for_timeout(1000)

    return False, timeline


async def reselect_dev_after_reboot(page, target_mac: str, timeout_ms: int) -> bool:
    deadline = time.monotonic() + (timeout_ms / 1000)
    normalized_target = normalize_mac(target_mac)

    while time.monotonic() < deadline:
        options = await get_device_options(page)
        if any(normalize_mac(option["value"]) == normalized_target for option in options):
            try:
                await select_device(page, target_mac, timeout_ms=4000)
                return True
            except PlaywrightTimeoutError:
                pass
        await page.wait_for_timeout(1000)

    return False


async def wait_for_non_idle_pair(page, target_mac: str, started_at: float, timeout_ms: int) -> tuple[bool, list[PairStateSnapshot]]:
    deadline = time.monotonic() + (timeout_ms / 1000)
    timeline: list[PairStateSnapshot] = []
    normalized_target = normalize_mac(target_mac)

    while time.monotonic() < deadline:
        state = await capture_pair_state(page, started_at)
        timeline_append(timeline, state)
        if state.runtime_mac_normalized == normalized_target and state.pair_state in ("pairing", "paired"):
            return True, timeline
        await page.wait_for_timeout(1000)

    return False, timeline


async def wait_for_pair_settle(page, target_mac: str, started_at: float, timeout_ms: int) -> tuple[bool, list[PairStateSnapshot]]:
    deadline = time.monotonic() + (timeout_ms / 1000)
    timeline: list[PairStateSnapshot] = []
    normalized_target = normalize_mac(target_mac)

    while time.monotonic() < deadline:
        state = await capture_pair_state(page, started_at)
        timeline_append(timeline, state)
        if state.runtime_mac_normalized == normalized_target and state.pair_state == "paired":
            return True, timeline
        await page.wait_for_timeout(1000)

    return False, timeline


def classify_failure(transition_observed: bool, reselected: bool, pair_ok: bool, reboot_response: dict) -> str:
    if not reboot_response.get("success"):
        return f"reboot request failed: {reboot_response}"
    if not transition_observed:
        return "reboot transition was not observed"
    if not reselected:
        return "device could not be reselected after reboot"
    if not pair_ok:
        return "pair row stayed idle after DEV reconnected"
    return "unknown"


async def run_iteration(page, iteration: int, transition_timeout_ms: int, pair_timeout_ms: int) -> IterationResult:
    dev_option = await get_dev_option(page)
    dev_mac = normalize_mac(dev_option["value"])
    await select_device(page, dev_option["value"])

    started_at = await page.evaluate("Date.now()")
    initial_state = await capture_pair_state(page, started_at)
    reboot_response = await trigger_immediate_reboot(page)

    transition_observed, transition_timeline = await wait_for_reboot_transition(
        page,
        dev_option["value"],
        started_at,
        transition_timeout_ms,
    )
    reselected = await reselect_dev_after_reboot(page, dev_option["value"], transition_timeout_ms)
    pair_ok, pair_timeline = await wait_for_non_idle_pair(page, dev_option["value"], started_at, pair_timeout_ms) if reselected else (False, [])

    success = reboot_response.get("success") and transition_observed and reselected and pair_ok
    return IterationResult(
        iteration=iteration,
        dev_mac=dev_mac,
        success=success,
        reboot_response=reboot_response,
        transition_observed=transition_observed,
        reselected=reselected,
        pair_ok=pair_ok,
        failure_reason=None if success else classify_failure(transition_observed, reselected, pair_ok, reboot_response),
        initial_state=initial_state,
        transition_timeline=transition_timeline,
        pair_timeline=pair_timeline,
    )


async def run_check(base_url: str, iterations: int, headless: bool, transition_timeout_ms: int, pair_timeout_ms: int) -> RebootPairResumeResult:
    result = RebootPairResumeResult(iterations=iterations)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel="msedge", headless=headless)
        page = await browser.new_page()

        page.on(
            "console",
            lambda message: result.console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: result.page_errors.append(str(error)))
        page.on(
            "requestfailed",
            lambda request: result.request_failures.append(
                f"{request.method} {request.url} -> {getattr(request, 'failure', None)!r}"
            ),
        )
        page.on(
            "response",
            lambda response: result.response_failures.append(
                f"{response.status} {response.request.method} {response.url}"
            )
            if response.status >= 400
            else None,
        )

        await goto_rf_settings(page, base_url)

        for iteration in range(1, iterations + 1):
            iteration_result = await run_iteration(
                page,
                iteration,
                transition_timeout_ms,
                pair_timeout_ms,
            )
            result.iteration_results.append(iteration_result)
            result.completed_iterations = iteration
            if not iteration_result.success:
                break

            if iteration < iterations:
                started_at = await page.evaluate("Date.now()")
                pair_settled, settle_timeline = await wait_for_pair_settle(
                    page,
                    iteration_result.dev_mac,
                    started_at,
                    pair_timeout_ms,
                )
                iteration_result.pair_settled = pair_settled
                iteration_result.settle_timeline = settle_timeline
                if not pair_settled:
                    iteration_result.success = False
                    iteration_result.failure_reason = "pair did not settle to paired before the next reboot"
                    break

        result.success = result.completed_iterations == iterations and all(
            item.success for item in result.iteration_results
        )
        await browser.close()

    return result


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--transition-timeout-ms", type=int, default=45000)
    parser.add_argument("--pair-timeout-ms", type=int, default=15000)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        result = await run_check(
            args.url,
            args.iterations,
            headless=not args.headed,
            transition_timeout_ms=args.transition_timeout_ms,
            pair_timeout_ms=args.pair_timeout_ms,
        )
    except Error as error:
        payload = {"fatal_error": str(error)}
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        print(text)
        return 1
    except Exception as error:
        payload = {"fatal_error": str(error)}
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        print(text)
        return 1

    text = json.dumps(asdict(result), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))