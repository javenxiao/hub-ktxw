# Wireless Status Rust Server

这个目录现在是一个 Rust Web 服务骨架，整体组织方式参考了 ar_dbg_client：

- 前端静态资源放在 static 目录
- 后端使用 axum 提供 HTTP 接口和 WebSocket
- 页面中的无线状态内容由服务端下发，不再是纯静态硬编码

## 目录说明

- static/index.html：页面入口
- static/styles.css：页面样式
- src/main.rs：服务端入口、接口、WebSocket 和模拟数据源
- ar_dbg_client：参考实现

## 当前接口

- GET /api/wireless/status：返回当前无线状态快照
- GET /ws：推送实时无线状态更新

## 运行方式

```bash
cargo run
```

默认监听：

- http://127.0.0.1:8080

## 后续接真实数据的方向

现在 src/main.rs 里是一个模拟数据生成器，后面可以直接替换成真实采集逻辑：

1. 从设备状态接口抓取 RF/Traffic/Connection 数据
2. 把采集结果组装成 WirelessSnapshot
3. 写回共享状态并通过 WebSocket 广播给前端

这样前端页面结构基本不用再改，只需要换后端数据来源。
