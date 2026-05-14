# Wireless Status Server PC Test Package

## 1. Package purpose

This package is for Windows PC validation and test execution.

It includes:

- `wireless_status_server.exe`
- runtime DLL files copied from `target/release`
- `static` page resources
- `start.cmd` for one-click launch

## 2. Basic usage

1. Unzip the package to any local folder.
2. Keep `wireless_status_server.exe`, `start.cmd`, and the `static` folder at the same level.
3. Double-click `start.cmd` so the console starts with the expected code page and working directory.
4. Open `http://127.0.0.1:8080` in a browser.

## 3. Real device validation

If you need real board data instead of fallback or mock behavior:

1. Start the external `daemon.exe` first.
2. Confirm `127.0.0.1:50000` is listening.
3. Then double-click `start.cmd`.
4. Open `http://127.0.0.1:8080/api/baseband/health` to check whether `host.connected` is `true`.

## 4. Stop the service

- Press `Ctrl + C` in the console window.
- Or close the console window directly.

## 5. Common issues

- If the page cannot load, first confirm the `static` folder was not removed or moved.
- If port `8080` is occupied, edit `start.cmd` and change `SERVER_PORT`.
- If wireless runtime data is unavailable, check whether `daemon.exe` is running and whether `127.0.0.1:50000` is reachable.
- Do not launch the EXE from another working directory. Use `start.cmd` so relative paths stay correct.