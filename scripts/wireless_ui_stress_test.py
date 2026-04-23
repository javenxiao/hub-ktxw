import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from playwright.async_api import Error, async_playwright


@dataclass
class StressResult:
    iterations: int
    completed_iterations: int = 0
    console_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    request_failures: list[str] = field(default_factory=list)
    response_failures: list[str] = field(default_factory=list)
    status_text: str | None = None
    final_series_count: str | None = None
    final_chart_title: str | None = None
    final_target_device: str | None = None
    final_stream_state: str | None = None
    final_legend_count: int = 0
    ap_plot_checks: list[str] = field(default_factory=list)


async def collect_option_values(page) -> list[str]:
    return await page.eval_on_selector_all(
        "#rf-operation-mode option",
        "options => options.map(option => option.value).filter(Boolean)",
    )


async def ensure_axes_panel_open(page) -> None:
    toggle = page.locator("#plot-settings-toggle")
    await toggle.wait_for(state="visible")
    if await toggle.get_attribute("aria-expanded") != "true":
        await toggle.click()
    await page.locator("#plot-settings-panel").wait_for(state="visible")


async def set_plot_input(page, setting_key: str, value: str) -> None:
    input_locator = page.locator(f'[data-plot-setting="{setting_key}"]').first
    await input_locator.wait_for(state="visible")
    await input_locator.click()
    await input_locator.press("Control+A")
    await input_locator.fill(value)
    await input_locator.press("Enter")


async def exercise_ap_plot_data(page, index: int, result: StressResult) -> None:
    await page.locator('[data-wireless-tab="status"]').click()
    await page.locator("#plot-series-grid").wait_for(state="visible")
    await page.wait_for_function(
        "() => document.querySelectorAll('.plot-legend-item').length > 0 && Boolean(document.getElementById('plot-surface'))"
    )

    if index % 25 == 0:
        await ensure_axes_panel_open(page)
        await set_plot_input(page, "xMax", str(120 + (index % 5) * 40))
        await set_plot_input(page, "updateIntervalMs", str(200 + (index % 4) * 100))
        await set_plot_input(page, "yMax", str(100 + (index % 3) * 20))

    if index % 40 == 0:
        legend_items = page.locator(".plot-legend-item")
        legend_count = await legend_items.count()
        if legend_count > 0:
            await legend_items.nth(index % legend_count).click()
            await page.wait_for_timeout(10)
            await legend_items.nth(index % legend_count).click()

    if index % 60 == 0:
        await page.locator("#plot-pause-btn").click()
        await page.wait_for_timeout(10)
        await page.locator("#plot-start-btn").click()

    if index % 90 == 0:
        await page.locator("#plot-stop-btn").click()
        await page.wait_for_timeout(10)
        await page.locator("#plot-start-btn").click()

    surface = page.locator("#plot-surface")
    box = await surface.bounding_box()
    if box:
        await page.mouse.move(box["x"] + box["width"] * 0.35, box["y"] + box["height"] * 0.45)
        await page.wait_for_timeout(5)

    if index % 100 == 0:
        stream_state = await page.locator("#plot-stream-state").text_content()
        if stream_state:
            result.ap_plot_checks.append(f"iter={index}:stream={stream_state}")


def extract_target_from_title(title: str | None) -> str | None:
    if not title:
        return None
    if "(" in title and title.endswith(")"):
        return title.rsplit("(", 1)[-1].rstrip(")").strip()
    if " - " in title:
        return title.rsplit(" - ", 1)[-1].strip()
    return title.strip()


async def run_stress_test(base_url: str, iterations: int, headless: bool) -> StressResult:
    result = StressResult(iterations=iterations)

    def handle_request_failed(request) -> None:
        try:
            result.request_failures.append(
                f"{request.method} {request.url} -> {getattr(request, 'failure', None)!r}"
            )
        except Exception as error:
            result.request_failures.append(f"requestfailed handler error: {error}")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            channel="msedge",
            headless=headless,
        )
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
        await page.locator('label[for="tab-wireless"]').click()
        await page.locator('[data-wireless-tab="status"]').click()
        await page.wait_for_selector("#plot-series-grid")

        for index in range(iterations):
            await exercise_ap_plot_data(page, index, result)
            await page.wait_for_timeout(15)
            result.completed_iterations = index + 1

        await page.locator('[data-wireless-tab="status"]').click()
        await page.wait_for_timeout(500)

        result.status_text = await page.locator("#wireless-control-status").text_content()
        result.final_chart_title = await page.locator("#chart-title").text_content()
        result.final_stream_state = await page.locator("#plot-stream-state").text_content()
        result.final_legend_count = await page.locator(".plot-legend-item").count()
        result.final_series_count = f"Series: {result.final_legend_count}"
        result.final_target_device = extract_target_from_title(result.final_chart_title)

        await browser.close()

    return result


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        result = await run_stress_test(args.url, args.iterations, headless=not args.headed)
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