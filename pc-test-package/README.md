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

## 6. 当前版本完成情况

当前 PC 包可用于 wireless web 已完成功能的联调和验证。

已完成并可测试的内容如下：

| 功能区域 | 当前状态 | 测试说明 |
| --- | --- | --- |
| 服务启动与页面访问 | 可测试 | 可通过 `start.cmd` 启动，并用浏览器打开 `http://127.0.0.1:8080` |
| 设备连接状态 | 可测试 | 可通过 `/api/baseband/health` 检查 bb_host 是否连接成功 |
| Active Device 切换 | 可测试 | 页面顶部设备下拉框可切换当前活动设备 |
| Role 显示与切换 | 可测试 | 支持 AP / DEV 角色显示与切换；切换后会进入重启恢复流程 |
| 重启恢复后的设备列表 | 可测试 | 设备重启恢复期间，页面仍能保留可识别的设备列表与恢复提示 |
| System 基础信息 | 可测试 | Summary、Maintenance、Reboot 相关基础展示和切换联动可做验证 |
| RF Settings - Band | 可测试 | 支持读取和切换频段/频段选择 |
| RF Settings - Channel | 可测试 | 支持 Auto / Manual 切换和信道设置；在 `1 AP + 1 DEV` 拓扑下已验证可正常生效 |
| RF Settings - Power | 可测试 | 支持 Power Auto、Power Mode、Power 数值设置 |
| RF Settings - MCS | 可测试 | 支持 MCS Auto 切换；手动 MCS 允许发送 |
| RF Settings - Bandwidth | 可测试 | 当前 Async 运行态下会直接提示 `unsupported` 并禁用控件，这属于当前设计行为 |
| RSSI Graph | 可测试 | 图表默认刷新间隔为 `100 ms`，可用于观察实时曲线 |
| Pair | 可测试 | 支持 AP / DEV 配对；DEV 侧可按 AP MAC 执行配对 |

## 7. 当前不纳入的内容

以下内容暂不建议纳入本次 PC 包测试范围：

| 功能区域 | 状态 | 说明 |
| --- | --- | --- |
| Configuration 页面 | 暂不纳入 | 当前刚开始联调，功能尚未完成，不作为本次测试范围 |