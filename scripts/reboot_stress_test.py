import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from playwright.async_api import Error, async_playwright


def normalize_mac(value: str | None) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch in "0123456789abcdef")


@dataclass
class IterationState:
    elapsed_ms: int
    selected_mac: str
    current_mac: str
    runtime_mac_text: str
    maintenance_local_mac: str
    live_indicator: str
    reboot_status: str
    reboot_status_state: str
    scheduled_enabled: bool
    in_flight: bool
    queued: str
    info_ok: bool
    info_error: str | None = None


@dataclass
class IterationResult:
    iteration: int
    initial_mac: str
    success: bool
    failure_reason: str | None = None
    final_state: IterationState | None = None
    timeline: list[IterationState] = field(default_factory=list)


@dataclass
class RebootStressResult:
    iterations: int
    completed_iterations: int = 0
    success: bool = False
    console_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    request_failures: list[str] = field(default_factory=list)
    response_failures: list[str] = field(default_factory=list)
    iteration_results: list[IterationResult] = field(default_factory=list)


async def capture_iteration_state(page, started_at: float) -> IterationState:
    payload = await page.evaluate(
        """
        async ({ startedAt }) => {
            const normalize = (value) => String(value || '').toLowerCase().replace(/[^0-9a-f]/g, '');

            let infoOk = false;
            let infoError = '';
            try {
                const response = await fetch('/api/system/info', { cache: 'no-store' });
                infoOk = response.ok;
            } catch (error) {
                infoError = String(error?.message || error || '');
            }

            const select = document.getElementById('rf-operation-mode');
            const runtimeMacText = String(document.getElementById('runtime-mac')?.textContent || '').trim();
            const maintenanceLocalMac = String(
                document.getElementById('maintenance-local-mac-config')?.textContent || ''
            ).trim();
            const currentMac = normalize(runtimeMacText || maintenanceLocalMac);
            return {
                elapsed_ms: Math.round(Date.now() - startedAt),
                selected_mac: normalize(select?.value || ''),
                current_mac: currentMac,
                runtime_mac_text: runtimeMacText,
                maintenance_local_mac: maintenanceLocalMac,
                live_indicator: document.querySelector('img[alt^="RF Radio"]')?.getAttribute('alt')
                    || document.querySelector('.live-state')?.getAttribute('aria-label')
                    || '',
                reboot_status: String(document.getElementById('reboot-action-status')?.textContent || '').trim(),
                reboot_status_state: String(document.getElementById('reboot-action-status')?.dataset?.state || '').trim(),
                scheduled_enabled: Boolean(document.getElementById('reboot-scheduled-toggle')?.checked),
                in_flight: Boolean(globalThis.deviceSelectionInFlight),
                queued: normalize(globalThis.pendingDeviceSelectionMac || ''),
                info_ok: infoOk,
                info_error: infoError,
            };
        }
        """,
        {"startedAt": started_at},
    )
    return IterationState(**payload)


async def trigger_immediate_reboot(page) -> None:
    await page.evaluate(
        """
        async () => {
            const originalConfirm = window.confirm;
            try {
                window.confirm = () => true;
                globalThis.setScheduledRebootEnabled(false);
                globalThis.updateRebootControlsState();
                await globalThis.triggerRebootAction();
            } finally {
                window.confirm = originalConfirm;
            }
        }
        """
    )


def timeline_append(timeline: list[IterationState], state: IterationState) -> None:
    if not timeline or asdict(timeline[-1]) != asdict(state):
        timeline.append(state)


def classify_failure(initial_mac: str, final_state: IterationState) -> str:
    if not final_state.info_ok and final_state.info_error:
        return f"web api unreachable: {final_state.info_error}"
    if not final_state.current_mac:
        return "runtime stayed unavailable"
    if final_state.current_mac == initial_mac:
        return "current device did not switch away from reboot target"
    if normalize_mac(final_state.maintenance_local_mac) != final_state.current_mac:
        return "maintenance local MAC did not align with current device"
    if "online" not in final_state.live_indicator.lower():
        return "RF Radio did not stay online"
    if final_state.reboot_status_state != "success":
        return "reboot status did not settle to success"
    return "timed out waiting for reboot recovery"


async def run_iteration(page, iteration: int, timeout_ms: int, poll_interval_ms: int) -> IterationResult:
    started_at = await page.evaluate("Date.now()")
    initial_state = await capture_iteration_state(page, started_at)
    timeline = [initial_state]

    await trigger_immediate_reboot(page)

    deadline = started_at + timeout_ms
    while True:
        await page.wait_for_timeout(poll_interval_ms)
        current_state = await capture_iteration_state(page, started_at)
        timeline_append(timeline, current_state)

        success = (
            current_state.info_ok
            and current_state.current_mac
            and current_state.current_mac != initial_state.current_mac
            and current_state.selected_mac == current_state.current_mac
            and normalize_mac(current_state.maintenance_local_mac) == current_state.current_mac
            and "online" in current_state.live_indicator.lower()
            and current_state.reboot_status_state == "success"
            and not current_state.in_flight
            and not current_state.queued
        )
        if success:
            return IterationResult(
                iteration=iteration,
                initial_mac=initial_state.current_mac,
                success=True,
                final_state=current_state,
                timeline=timeline,
            )

        if current_state.elapsed_ms + started_at >= deadline:
            return IterationResult(
                iteration=iteration,
                initial_mac=initial_state.current_mac,
                success=False,
                failure_reason=classify_failure(initial_state.current_mac, current_state),
                final_state=current_state,
                timeline=timeline,
            )


async def run_stress_test(base_url: str, iterations: int, headless: bool, timeout_ms: int, poll_interval_ms: int) -> RebootStressResult:
    result = RebootStressResult(iterations=iterations)

    def handle_request_failed(request) -> None:
        try:
            result.request_failures.append(
                f"{request.method} {request.url} -> {getattr(request, 'failure', None)!r}"
            )
        except Exception as error:
            result.request_failures.append(f"requestfailed handler error: {error}")

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
        page.on("requestfailed", handle_request_failed)
        page.on(
            "response",
            lambda response: result.response_failures.append(
                f"{response.status} {response.request.method} {response.url}"
            )
            if response.status >= 400
            else None,
        )

        await page.goto(base_url, wait_until="networkidle")
        await page.locator('label[for="tab-system"]').click()
        await page.locator('[data-tab="reboot"]').click()
        await page.wait_for_selector('#reboot-submit-btn')

        for iteration in range(1, iterations + 1):
            iteration_result = await run_iteration(page, iteration, timeout_ms, poll_interval_ms)
            result.iteration_results.append(iteration_result)
            result.completed_iterations = iteration
            if not iteration_result.success:
                break

        result.success = result.completed_iterations == iterations and all(
            item.success for item in result.iteration_results
        )
        await browser.close()

    return result


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timeout-ms", type=int, default=45000)
    parser.add_argument("--poll-interval-ms", type=int, default=1000)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        result = await run_stress_test(
            args.url,
            args.iterations,
            headless=not args.headed,
            timeout_ms=args.timeout_ms,
            poll_interval_ms=args.poll_interval_ms,
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