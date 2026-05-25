import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from playwright.async_api import Error, Page, async_playwright


PAIR_APPLY_URL = "/api/wireless/runtime/apply"
PAIR_CANDIDATES_URL = "/api/wireless/pair-candidates"
UPGRADE_URL = "/api/system/maintenance/upgrade"
UPGRADE_PROGRESS_URL = "/api/system/maintenance/upgrade-progress"


@dataclass
class ScenarioResult:
    name: str
    success: bool = False
    status_text: str | None = None
    pair_row_count: int = 0
    pair_row_label: str | None = None
    pair_row_value: str | None = None
    pair_row_button: str | None = None
    apply_calls: list[dict[str, Any]] = field(default_factory=list)
    candidate_calls: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class PairRegressionResult:
    base_url: str
    console_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    request_failures: list[str] = field(default_factory=list)
    response_failures: list[str] = field(default_factory=list)
    scenarios: list[ScenarioResult] = field(default_factory=list)


@dataclass
class FirmwareUpgradeTestResult:
    """固件升级自动化测试结果"""
    base_url: str
    firmware_file: str
    upload_success: bool = False
    upload_message: str = ""
    upload_elapsed_ms: int = 0
    progress_states: list[dict[str, Any]] = field(default_factory=list)
    final_state: str = "unknown"
    final_message: str = ""
    final_crc32: str = ""
    total_elapsed_ms: int = 0
    success: bool = False
    error: str | None = None
    console_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    request_failures: list[str] = field(default_factory=list)
    response_failures: list[str] = field(default_factory=list)


async def run_firmware_upgrade_test(
    base_url: str,
    firmware_path: str,
    poll_interval_ms: int = 2000,
    timeout_ms: int = 300000,
    headless: bool = True,
) -> FirmwareUpgradeTestResult:
    """测试固件升级功能的自动化脚本。
    
    通过 Playwright 在浏览器中操作文件上传、触发升级、轮询进度，
    验证整个升级链路是否正常工作。
    """
    result = FirmwareUpgradeTestResult(
        base_url=base_url,
        firmware_file=firmware_path,
    )

    if not os.path.isfile(firmware_path):
        result.error = f"Firmware file not found: {firmware_path}"
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
        return result

    firmware_size = os.path.getsize(firmware_path)
    print(f"Firmware file: {firmware_path} ({firmware_size} bytes)")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel="msedge", headless=headless)
        page = await browser.new_page()

        # 记录错误
        page.on("console", lambda msg: result.console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: result.page_errors.append(str(err)))
        page.on("requestfailed", lambda request: result.request_failures.append(
            f"{request.method} {request.url} -> {getattr(request, 'failure', None)!r}"))
        page.on("response", lambda response: result.response_failures.append(
            f"{response.status} {response.request.method} {response.url}") if response.status >= 400 else None)

        # 导航到维护页面
        print(f"Navigating to {base_url}")
        await page.goto(base_url, wait_until="networkidle")
        await page.locator('label[for="tab-system"]').click()
        await page.locator('[data-tab="maintenance"]').click()
        await page.wait_for_selector('#maintenance-upgrade-btn', timeout=10000)

        # 等 maintenance info 加载完毕
        await page.wait_for_function(
            "() => globalThis.maintenanceInfoCache !== null && globalThis.maintenanceInfoCache !== undefined",
            timeout=15000,
        )
        print("Maintenance info loaded, upgrade button should be visible")

        # 上传文件
        file_input = page.locator('#maintenance-firmware-file')
        await file_input.set_input_files(firmware_path)
        await page.wait_for_timeout(500)

        # 检查升级按钮是否可用
        upgrade_button = page.locator('#maintenance-upgrade-btn')
        is_disabled = await upgrade_button.is_disabled()
        if is_disabled:
            result.error = "Upgrade button is disabled (maybe upgrade_supported=false or no file selected)"
            print(f"ERROR: {result.error}")
            await browser.close()
            print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
            return result

        # 触发升级（自动确认 confirm 对话框）
        print("Triggering firmware upgrade...")
        started_at = time.monotonic()

        # 覆盖 confirm 然后点击升级按钮
        await page.evaluate("window.confirm = () => true")
        await upgrade_button.click()

        # 等待上传响应
        try:
            await page.wait_for_function(
                """(expectedFile) => {
                    const status = globalThis.maintenanceStatusStore?.['maintenance-action-status'];
                    return status && (status.state === 'success' || status.state === 'error' || status.state === 'pending');
                }""",
                timeout=30000,
            )
        except Exception as e:
            result.error = f"Timeout waiting for upload response: {e}"
            print(f"ERROR: {result.error}")
            await browser.close()
            print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
            return result

        # 读取上传响应状态
        upload_status = await page.evaluate(
            """() => {
                const status = globalThis.maintenanceStatusStore?.['maintenance-action-status'];
                return status ? { state: status.state, message: status.message } : { state: 'unknown', message: '' };
            }"""
        )
        result.upload_message = upload_status.get("message", "")
        upload_state = upload_status.get("state", "unknown")

        if upload_state == "error":
            result.error = f"Upload rejected: {result.upload_message}"
            result.success = False
            print(f"FAIL: {result.error}")
            await browser.close()
            print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
            return result

        if upload_state == "success":
            result.upload_success = True
            result.upload_elapsed_ms = int((time.monotonic() - started_at) * 1000)
            print(f"Upload accepted in {result.upload_elapsed_ms}ms. Starting progress polling...")
        else:
            # 可能在 pending 状态（后台还在处理）
            print(f"Upload state: {upload_state}. Starting progress polling...")

        # 轮询升级进度
        deadline = time.monotonic() + (timeout_ms / 1000)
        last_state = ""

        while time.monotonic() < deadline:
            progress = await page.evaluate(
                """async () => {
                    try {
                        const resp = await fetch('/api/system/maintenance/upgrade-progress', { cache: 'no-store' });
                        return await resp.json();
                    } catch (e) {
                        return { state: 'poll_error', message: String(e) };
                    }
                }"""
            )

            if progress and progress.get("state"):
                current_state = progress["state"]
                if current_state != last_state:
                    result.progress_states.append({
                        "elapsed_ms": int((time.monotonic() - started_at) * 1000),
                        "state": current_state,
                        "percent": progress.get("percent", 0),
                        "step_label": progress.get("step_label", ""),
                        "message": progress.get("message", ""),
                        "crc32": progress.get("crc32", ""),
                    })
                    last_state = current_state
                    print(f"  Progress: {current_state} ({progress.get('percent', 0):.1f}%) - {progress.get('message', '')}")

                if current_state == "done":
                    result.success = True
                    result.final_state = "done"
                    result.final_message = progress.get("message", "")
                    result.final_crc32 = progress.get("crc32", "")
                    result.total_elapsed_ms = int((time.monotonic() - started_at) * 1000) if not result.total_elapsed_ms else result.total_elapsed_ms
                    print(f"DONE: CRC32={result.final_crc32}")
                    break
                elif current_state == "error":
                    result.error = progress.get("message", "Unknown upgrade error")
                    result.final_state = "error"
                    result.final_message = progress.get("message", "")
                    print(f"FAIL: {result.error}")
                    break
                elif current_state == "idle" and len(result.progress_states) > 1:
                    # 从非 idle 回到 idle 说明已完成但没有 done 状态
                    result.error = "Upgrade progress returned to idle without done"
                    print(f"WARNING: {result.error}")
                    break

            await page.wait_for_timeout(poll_interval_ms)
        else:
            result.error = f"Upgrade timed out after {timeout_ms}ms"
            result.final_state = "timeout"
            print(f"TIMEOUT: {result.error}")

        if not result.total_elapsed_ms:
            result.total_elapsed_ms = int((time.monotonic() - started_at) * 1000)

        await browser.close()

    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return result


async def install_pair_probe(page: Page) -> None:
    await page.evaluate(
        """
        () => {
            window.__pairApplyCalls = [];
            window.__pairCandidateCalls = [];

            if (window.__pairFetchPatched) {
                return;
            }

            window.__pairFetchPatched = true;
            window.__pairOriginalFetch = window.fetch.bind(window);
            window.fetch = async (...args) => {
                const [input, init] = args;
                const url = typeof input === 'string' ? input : input.url;
                const method = init?.method || 'GET';
                const body = typeof init?.body === 'string' ? init.body : null;
                const response = await window.__pairOriginalFetch(...args);
                const text = await response.clone().text();
                const record = { url, method, body, status: response.status, text };

                if (url.includes('/api/wireless/runtime/apply')) {
                    window.__pairApplyCalls.push(record);
                }
                if (url.includes('/api/wireless/pair-candidates')) {
                    window.__pairCandidateCalls.push(record);
                }

                return response;
            };
        }
        """
    )


async def reset_pair_probe(page: Page) -> None:
    await page.evaluate(
        """
        () => {
            window.__pairApplyCalls = [];
            window.__pairCandidateCalls = [];
        }
        """
    )


async def read_pair_probe(page: Page) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = await page.evaluate(
        """
        () => ({
            applyCalls: window.__pairApplyCalls || [],
            candidateCalls: window.__pairCandidateCalls || [],
        })
        """
    )
    return payload["applyCalls"], payload["candidateCalls"]


async def goto_rf_settings(page: Page, base_url: str) -> None:
    await page.goto(base_url, wait_until="networkidle")
    await page.locator('label[for="tab-wireless"]').click()
    await page.locator('[data-wireless-tab="rf"]').click()
    await page.locator("#rf-pair-list").wait_for(state="visible")
    await install_pair_probe(page)


async def get_device_options(page: Page) -> list[dict[str, str]]:
    return await page.eval_on_selector_all(
        "#rf-operation-mode option",
        "options => options.map(option => ({ value: option.value, label: (option.textContent || '').trim() })).filter(option => option.value)",
    )


async def select_device_by_prefix(page: Page, prefix: str) -> dict[str, str]:
    options = await get_device_options(page)
    match = next((option for option in options if option["label"].upper().startswith(prefix.upper() + ":")), None)
    if not match:
        raise RuntimeError(f"Unable to find device with prefix {prefix!r}")

    await page.locator("#rf-operation-mode").select_option(match["value"])
    await page.wait_for_function(
        """
        target => {
            const text = document.querySelector('#runtime-mac')?.textContent || '';
            return text.replace(/[^0-9A-F]/gi, '').toUpperCase() === String(target || '').toUpperCase();
        }
        """,
        arg=match["value"],
    )
    return match


async def wait_for_pair_button(page: Page, expected: str = "Pair") -> None:
    await page.wait_for_function(
        """
        expectedText => {
            const button = document.querySelector('#rf-pair-list [data-pair-slot="0"]');
            return Boolean(button) && button.textContent?.trim() === expectedText;
        }
        """,
        arg=expected,
    )


async def read_first_pair_row(page: Page) -> tuple[int, str | None, str | None, str | None]:
    row_count = await page.locator(".rf-pair-row").count()
    if row_count == 0:
        return 0, None, None, None

    row = page.locator(".rf-pair-row").first
    return (
        row_count,
        await row.locator(".rf-pair-user").text_content(),
        await row.locator("[data-pair-mac-slot]").input_value(),
        await row.locator("[data-pair-slot]").text_content(),
    )


async def stop_pair_if_running(page: Page) -> None:
    button = page.locator('#rf-pair-list [data-pair-slot="0"]')
    if not await button.count():
        return

    if (await button.text_content() or "").strip() == "Stop":
        await button.click()
        await page.wait_for_timeout(1500)


def normalize_pair_mac(value: str | None) -> str:
    return "".join(ch for ch in (value or "") if ch.lower() in "0123456789abcdef").upper()


def parse_apply_body(call: dict[str, Any]) -> dict[str, Any]:
    body = call.get("body")
    if not body:
        return {}
    return json.loads(body)


async def attach_page_listeners(page: Page, result: PairRegressionResult) -> None:
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


async def run_ap_auto_pair(page: Page, base_url: str) -> ScenarioResult:
    scenario = ScenarioResult(name="ap_auto_pair")
    await goto_rf_settings(page, base_url)
    selected = await select_device_by_prefix(page, "AP")
    scenario.notes.append(f"Selected device: {selected['label']}")
    await stop_pair_if_running(page)
    await wait_for_pair_button(page, "Pair")
    await reset_pair_probe(page)

    row_count, label, value, button_text = await read_first_pair_row(page)
    scenario.pair_row_count = row_count
    scenario.pair_row_label = label
    scenario.pair_row_value = value
    scenario.pair_row_button = button_text

    await page.locator('#rf-pair-list [data-pair-slot="0"]').click()
    await page.wait_for_function("() => (window.__pairApplyCalls || []).length >= 1")
    await page.wait_for_timeout(800)

    apply_calls, candidate_calls = await read_pair_probe(page)
    scenario.apply_calls = apply_calls
    scenario.candidate_calls = candidate_calls
    scenario.status_text = await page.locator("#wireless-control-status").text_content()

    if not apply_calls:
        raise RuntimeError("AP auto pair did not send any apply request")

    payload = parse_apply_body(apply_calls[0])
    if payload.get("pair_target_mac"):
        raise RuntimeError(f"AP auto pair unexpectedly sent pair_target_mac: {payload}")

    await stop_pair_if_running(page)
    scenario.success = True
    return scenario


async def run_ap_targeted_pair(page: Page, base_url: str) -> ScenarioResult:
    scenario = ScenarioResult(name="ap_targeted_pair")
    await goto_rf_settings(page, base_url)
    selected = await select_device_by_prefix(page, "AP")
    scenario.notes.append(f"Selected device: {selected['label']}")
    await stop_pair_if_running(page)
    await wait_for_pair_button(page, "Pair")
    await reset_pair_probe(page)

    input_locator = page.locator('#rf-pair-list [data-pair-mac-slot="0"]')
    await input_locator.focus()
    await page.wait_for_timeout(800)

    device_options = await get_device_options(page)
    dev_option = next(
        (option for option in device_options if option["label"].upper().startswith("DEV:")),
        None,
    )
    if not dev_option:
        raise RuntimeError("AP targeted pair could not find any DEV device in the runtime device list")

    target_mac = dev_option["value"]
    scenario.notes.append(f"Target DEV MAC: {target_mac}")
    await input_locator.fill(target_mac)
    await page.locator('#rf-pair-list [data-pair-slot="0"]').click()
    await page.wait_for_function("() => (window.__pairApplyCalls || []).length >= 1")
    await page.wait_for_timeout(800)

    apply_calls, candidate_calls = await read_pair_probe(page)
    scenario.apply_calls = apply_calls
    scenario.candidate_calls = candidate_calls
    scenario.status_text = await page.locator("#wireless-control-status").text_content()

    if not apply_calls:
        raise RuntimeError("AP targeted pair did not send any apply request")

    payload = parse_apply_body(apply_calls[0])
    if payload.get("pair_target_mac") != normalize_pair_mac(target_mac):
        raise RuntimeError(
            f"AP targeted pair sent unexpected target MAC: expected {normalize_pair_mac(target_mac)}, got {payload}"
        )

    scenario.notes.append(
        f"Candidate list requests observed: {len(candidate_calls)}"
    )

    await stop_pair_if_running(page)
    row_count, label, value, button_text = await read_first_pair_row(page)
    scenario.pair_row_count = row_count
    scenario.pair_row_label = label
    scenario.pair_row_value = value
    scenario.pair_row_button = button_text
    scenario.success = True
    return scenario


async def run_dev_single_address_pair(page: Page, base_url: str) -> ScenarioResult:
    scenario = ScenarioResult(name="dev_single_address_pair")
    await goto_rf_settings(page, base_url)
    selected = await select_device_by_prefix(page, "DEV")
    scenario.notes.append(f"Selected device: {selected['label']}")
    await stop_pair_if_running(page)
    await wait_for_pair_button(page, "Pair")
    await reset_pair_probe(page)

    row_count, label, value, button_text = await read_first_pair_row(page)
    scenario.pair_row_count = row_count
    scenario.pair_row_label = label
    scenario.pair_row_value = value
    scenario.pair_row_button = button_text

    if row_count != 1:
        raise RuntimeError(f"DEV pair panel should render exactly one row, got {row_count}")
    if (label or "").strip() != "AP MAC":
        raise RuntimeError(f"DEV pair row should be labeled AP MAC, got {label!r}")

    input_locator = page.locator('#rf-pair-list [data-pair-mac-slot="0"]')
    await input_locator.focus()
    await page.wait_for_timeout(800)

    apply_calls, candidate_calls = await read_pair_probe(page)
    if candidate_calls:
        raise RuntimeError(f"DEV pair should not request candidate list, got {candidate_calls}")

    target_mac = (await input_locator.input_value()).strip()
    if not normalize_pair_mac(target_mac):
        raise RuntimeError("DEV pair does not expose a usable AP MAC target value")

    await page.locator('#rf-pair-list [data-pair-slot="0"]').click()
    await page.wait_for_function("() => (window.__pairApplyCalls || []).length >= 1")
    await page.wait_for_timeout(800)

    apply_calls, candidate_calls = await read_pair_probe(page)
    scenario.apply_calls = apply_calls
    scenario.candidate_calls = candidate_calls
    scenario.status_text = await page.locator("#wireless-control-status").text_content()

    if not apply_calls:
        raise RuntimeError("DEV pair did not send any apply request")

    payload = parse_apply_body(apply_calls[0])
    if payload.get("pair_target_mac") != normalize_pair_mac(target_mac):
        raise RuntimeError(
            f"DEV pair should send AP MAC as pair_target_mac: expected {normalize_pair_mac(target_mac)}, got {payload}"
        )

    if candidate_calls:
        raise RuntimeError(f"DEV pair should not request candidate list during apply, got {candidate_calls}")

    await stop_pair_if_running(page)
    scenario.success = True
    return scenario


async def run_dev_pair_with_cleared_ap_mac(page: Page, base_url: str) -> ScenarioResult:
    scenario = ScenarioResult(name="dev_pair_with_cleared_ap_mac")
    await goto_rf_settings(page, base_url)
    selected = await select_device_by_prefix(page, "DEV")
    scenario.notes.append(f"Selected device: {selected['label']}")
    await stop_pair_if_running(page)
    await wait_for_pair_button(page, "Pair")
    await reset_pair_probe(page)

    row_count, label, value, button_text = await read_first_pair_row(page)
    scenario.pair_row_count = row_count
    scenario.pair_row_label = label
    scenario.pair_row_value = value
    scenario.pair_row_button = button_text

    if row_count != 1:
        raise RuntimeError(f"DEV cleared-AP-MAC pair panel should render exactly one row, got {row_count}")
    if (label or "").strip() != "AP MAC":
        raise RuntimeError(f"DEV cleared-AP-MAC pair row should be labeled AP MAC, got {label!r}")

    input_locator = page.locator('#rf-pair-list [data-pair-mac-slot="0"]')
    await input_locator.fill("")
    await page.wait_for_timeout(200)

    apply_calls, candidate_calls = await read_pair_probe(page)
    if candidate_calls:
        raise RuntimeError(f"DEV cleared-AP-MAC pair should not request candidate list, got {candidate_calls}")

    await page.locator('#rf-pair-list [data-pair-slot="0"]').click()
    await page.wait_for_function("() => (window.__pairApplyCalls || []).length >= 1")
    await page.wait_for_timeout(800)

    apply_calls, candidate_calls = await read_pair_probe(page)
    scenario.apply_calls = apply_calls
    scenario.candidate_calls = candidate_calls
    scenario.status_text = await page.locator("#wireless-control-status").text_content()

    if not apply_calls:
        raise RuntimeError("DEV cleared-AP-MAC pair did not send any apply request")

    payload = parse_apply_body(apply_calls[0])
    if payload.get("pair_target_mac"):
        raise RuntimeError(
            f"DEV cleared-AP-MAC pair should not send pair_target_mac when input is blank, got {payload}"
        )

    if candidate_calls:
        raise RuntimeError(f"DEV cleared-AP-MAC pair should not request candidate list during apply, got {candidate_calls}")

    await stop_pair_if_running(page)
    scenario.success = True
    return scenario


async def execute_scenario(browser, result: PairRegressionResult, base_url: str, runner) -> None:
    page = await browser.new_page()
    await attach_page_listeners(page, result)
    try:
        scenario_result = await runner(page, base_url)
    except Exception as error:
        scenario_name = getattr(runner, "__name__", "unknown_scenario")
        result.scenarios.append(
            ScenarioResult(name=scenario_name, success=False, error=str(error))
        )
        raise
    else:
        result.scenarios.append(scenario_result)
    finally:
        await page.close()


async def run_pair_regression(base_url: str, headless: bool) -> PairRegressionResult:
    result = PairRegressionResult(base_url=base_url)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel="msedge", headless=headless)
        try:
            await execute_scenario(browser, result, base_url, run_ap_auto_pair)
            await execute_scenario(browser, result, base_url, run_ap_targeted_pair)
            await execute_scenario(browser, result, base_url, run_dev_single_address_pair)
            await execute_scenario(browser, result, base_url, run_dev_pair_with_cleared_ap_mac)
        finally:
            await browser.close()

    return result


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        result = await run_pair_regression(args.url, headless=not args.headed)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))