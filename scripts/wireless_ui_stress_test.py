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


async def collect_option_values(page) -> list[str]:
    return await page.eval_on_selector_all(
        "#rf-operation-mode option",
        "options => options.map(option => option.value).filter(Boolean)",
    )


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
        await page.locator('[data-wireless-tab="rf"]').click()
        await page.wait_for_selector("#rf-operation-mode")

        for index in range(iterations):
            await page.locator('[data-wireless-tab="status"]').click()
            await page.locator('[data-wireless-tab="rf"]').click()

            select = page.locator("#rf-operation-mode")
            await select.wait_for(state="attached")
            await page.wait_for_function("() => { const select = document.getElementById('rf-operation-mode'); return Boolean(select) && !select.disabled; }")
            option_values = await collect_option_values(page)
            if len(option_values) < 2:
                await page.wait_for_timeout(50)
                continue

            current_value = await select.input_value()
            next_value = next(
                (value for value in option_values if value != current_value),
                option_values[index % len(option_values)],
            )
            await select.select_option(next_value)
            await page.wait_for_timeout(35)
            result.completed_iterations = index + 1

        await page.locator('[data-wireless-tab="status"]').click()
        await page.wait_for_timeout(1000)

        result.status_text = await page.locator("#wireless-control-status").text_content()
        result.final_series_count = await page.locator("#chart-series-count").text_content()
        result.final_chart_title = await page.locator("#chart-title").text_content()
        result.final_target_device = await page.locator("#chart-target-device").text_content()

        await browser.close()

    return result


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/")
    parser.add_argument("--iterations", type=int, default=200)
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