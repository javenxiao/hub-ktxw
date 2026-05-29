# Wireless Status Rust Server

这个目录是一个面向设备管理页面的 Rust Web 服务。它不是前后端分离工程，而是由 Rust 服务直接托管静态页面，并通过 HTTP 接口和 WebSocket 向页面下发设备状态。

当前仓库已经接入第三方基带 SDK 的库文件和头文件，在此基础上可以继续完成页面开发、接口开发和硬件数据接入。

## 0. 最近版本更新

### 2026-05-20

- Wireless -> Status / Connection Info 已重新对齐 frame 语义：AP 与 DEV 统一展示 `BR + slot0` 两条记录；后端不再混用当前 effective user 的 PHY 状态，而是拆分固定 `User 0` 的 `slot_tx_status/slot_rx_status` 与固定 `User BR/CS` 的 `br_tx_status/br_rx_status`，避免 slot0 行的频率、带宽、MCS 被 BR 用户污染。
- remote bb_host 模式下的 DEV 状态刷新已进一步分层：runtime 路径继续复用按设备缓存以压低串口噪声，但状态快照额外增加了独立的 2 秒节流刷新，恢复 Connection Info 的 RSSI 小图与 RSSI Graph 更新，而不会重新回到 100ms 级别高频打 `BB_GET_STATUS`。
- RSSI Graph 交互已改为角色感知：AP 只允许从 slot0 行进入，并切到 `User 0`；DEV 只允许从 BR 行进入，并切到 `User BR/CS`。同时，DEV 手动切到 `User 0` 时后端会返回空 `chart.series`，避免出现与产品语义不一致的 DEV User0 曲线。
- Wireless 页在高频实时刷新下的可操作性已补强：Connection Info 的 RSSI 入口从双击改为 `pointerdown` 立即打开，并为 Wireless 控件增加了短暂的交互保护窗，避免下拉框、按钮或 RSSI 入口在用户操作中被运行态刷新重绘打断。
- Connection Info 的窄屏行为已修正为与 General Status 一致：中等窄屏优先使用横向滚动，不再把列宽继续压缩到文字重叠；同时信号条等级重新定义为 `SNR >= 20` 即显示满格。

### 2026-05-19

- Pair 持久化链路已补齐：自动对频成功后，板端实际锁定到的 peer MAC 会自动回写到 MiniDB，避免只保存用户手工输入的目标地址。
- Pair 重启恢复已补齐自动续连：AP 或 DEV 重启后，如果设备已保存 Pair 目标且当前链路未恢复，服务端会在设备重新上线后自动补发一次普通 Pair，恢复到 Connect / Paired，无需重新输入 MAC，也无需手工再点 Pair。
- AP 模式下的 Pair 面板已修正离线显示逻辑：当匹配设备离线、重启或页面处于 runtime/snapshot 切换窗口时，输入框会优先显示之前匹配过的目标 MAC，不再误显示“无匹配设备”或把当前 AP 自己的 MAC 当成匹配方显示。
- remote bb_host 模式下的 Active Device 轮询已做分层降噪：AP 侧高频 PRJ_DISPATCH 读取已缓存，DEV 侧 channel/status 热路径已改为按设备复用缓存，不再因 Web 后台 runtime、status、plot 刷新而周期性触发同类 SDK 读取。
- 顶部全局工具栏顺序已调整为 Active Device -> Role -> Reboot，减少 Role 切换与即时重启操作的视觉打断。
- 小屏顶部工具栏已进一步优化：Active Device、Role、Reboot 改为等宽三列单排展示，每项采用“标题在上、控件在下”的布局；Reboot 按钮也同步缩小，避免小屏出现横向滚动条。

### 2026-05-18

- Active Device 下拉已对齐板端状态，支持显示 AP(M)、AP(S)、DEV(M)、DEV(S) 主从标识。
- Maintenance 页面已精简顶部与版本信息展示，移除了 RF Product Console、RF Radio online 状态显示，并去掉了 Version Details 中的 Hardware Version 卡片；Firmware Version 已恢复为同组统一背景样式。
- Firmware Upgrade 已补齐性能与可观测性优化：
	- 远端热升级 write/CRC 路径不再套用通用 20ms SDK 调用间隔。
	- 升级进度已拆分为 HTTP upload 与 Board write 两段耗时，便于和 PC Tool 对比定位瓶颈。
	- 后端上传改为流式读取，避免整包二次拷贝。
	- 板端写入进度更新改为时间节流，减少升级热路径中的加锁和字符串构造开销。
	- 升级期间暂停 Maintenance 自动刷新，降低与升级线程的资源竞争。

## 1. 项目定位

项目分成 4 层：

- 页面层：static 目录下的 HTML、CSS、原生 JavaScript
- 接口层：src/main.rs 中的 HTTP 路由、WebSocket、返回结构
- 业务封装层：src/bb_api.rs 中对基带 SDK 的安全 Rust 封装
- FFI 层：src/ffi.rs 中对 C 接口、常量、结构体的声明

建议开发人员始终沿着这条链路理解和改动系统：

页面展示
-> Rust 接口
-> Rust 业务封装
-> FFI
-> 第三方动态库

## 2. 目录说明

- static/index.html：页面入口，包含页面结构和一部分原生 JavaScript 逻辑
- static/styles.css：页面样式
- src/main.rs：服务端入口、接口定义、WebSocket 推送、状态组装
- src/bb_api.rs：对基带 SDK 的安全包装，屏蔽底层错误码和裸指针
- src/ffi.rs：C 接口绑定、常量、底层调用声明
- build.rs：在本机构建时复制第三方 DLL 到目标目录，SDK 运行时加载逻辑位于 src/ffi.rs
- third_party/include：第三方头文件目录
- third_party/lib：第三方库文件目录
- scripts/package-a7.py：A7 目标打包脚本

## 3. 环境准备

### 3.1 Windows 本地开发环境

建议至少准备以下环境：

- Rust 工具链，建议使用 stable
- Visual Studio C++ Build Tools 或完整 Visual Studio 开发工具链
- 能正常执行 cargo build 的本地编译环境

如果需要在 Windows 下运行并联调第三方 SDK，还需要确保以下文件存在：

- third_party/lib/ar8030_client.dll
- third_party/lib/ar8030_client.lib
- third_party/include/bb_api.h
- third_party/include/bb_config.h

当前仓库已经包含这些文件，build.rs 会在构建时处理链接，并尝试将 DLL 复制到目标输出目录。

### 3.2 可选的 A7 交叉编译环境

如果需要构建 ARM A7 目标，准备交叉工具链后执行：

```bash
rustup target add armv7-unknown-linux-gnueabihf
cargo build-a7
```

如果交叉链接器名称不同，可以覆盖环境变量：

```bash
CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_LINKER=/path/to/arm-linux-gnueabihf-gcc cargo build-a7
```

发布打包：

```bash
python3 ./scripts/package-a7.py
```

如需覆盖 strip 或 upx 路径：

```bash
CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_STRIP=/path/to/arm-linux-gnueabihf-strip UPX=/path/to/upx python3 ./scripts/package-a7.py
```

## 4. 本地运行

### 4.1 启动与运行

当前代码已经内置以下默认启动参数：

```text
RUST_LOG=info
BB_HOST_ADDR=127.0.0.1
BB_HOST_PORT=50000
```

因此日常启动时不需要每次手动输入环境变量，直接运行即可。

开发模式运行：

```powershell
cd e:\Artosyn\10_project\rshtml
cargo run
```

如果已经完成编译，也可以直接运行可执行文件：

```powershell
cd e:\Artosyn\10_project\rshtml
.\target\debug\wireless_status_server.exe
```

如果某次需要临时覆盖默认值，仍然可以在启动前手动设置：

```powershell
$env:RUST_LOG = "debug"
$env:BB_HOST_ADDR = "192.168.1.10"
$env:BB_HOST_PORT = "50000"
cargo run
```

或在 VS Code 中运行已有任务：

- Debug Rust Project

默认监听地址：

- http://127.0.0.1:8080

常见检查点：

- 页面是否可以打开
- GET /api/system/info 是否返回 JSON
- GET /api/wireless/status 是否返回 JSON
- WebSocket /ws 是否能收到推送数据
- 运行目录下是否存在 ar8030_client.dll

### 4.2 结束程序

如果程序是从当前终端以前台方式启动的，最简单的结束方式是：

```text
Ctrl + C
```

如果程序已经在后台运行，或者当前终端已经关闭，可以执行：

```powershell
Get-Process wireless_status_server -ErrorAction SilentlyContinue | Stop-Process -Force
```

或者：

```powershell
taskkill /F /IM wireless_status_server.exe
```

### 4.3 当前基线验证结果

已在本仓库执行过一次基线验证，当前结果如下：

- cargo build 成功
- cargo run 成功
- 页面入口 / 可正常返回 static/index.html
- GET /api/system/info 可正常返回 JSON
- GET /api/wireless/status 可正常返回 JSON
- 当 SDK 无法完成加载时，服务会自动回退到模拟模式继续运行

当前 Windows 本地环境中的已知现象：

- ar8030_client.dll 已复制到 target/debug
- 但第三方 DLL 的依赖仍可能缺失，导致 SDK 初始化失败
- 这种情况下，Web 服务本身仍可启动，只是基带能力不可用

说明：

- 当 BB_HOST_ADDR 存在时，程序会尝试先连接远程 bb_host，再走设备枚举和设备打开流程
- 当 BB_HOST_ADDR 不存在时，程序走本地 SDK 模式
- 当前仓库已经修复了远程 RPC 设备路径的崩溃问题，但是否能真正进入真实模式，仍取决于外部 bb_host、设备、驱动和 SDK 环境是否满足要求

这意味着：

- “cargo run 成功”不等于“真实基带模式已成功启动”
- 当前项目已经支持在 SDK 不可用时继续做页面和接口开发

## 5. 开发人员如何开始工作

建议按下面顺序上手，而不是一开始就直接修改底层 FFI：

1. 先运行现有服务，确认页面、接口和 DLL 加载都正常。
2. 先熟悉 static/index.html 和 static/styles.css，理解页面结构和 DOM 更新方式。
3. 再阅读 src/main.rs，理解接口、状态结构和 WebSocket 广播。
4. 再阅读 src/bb_api.rs，理解当前基带 SDK 是如何被包装成 Rust 方法的。
5. 最后再查看 src/ffi.rs 和 third_party/include 中的头文件，理解底层能力来源。

推荐第一个练手任务是：

- 先给已有页面增加一个只读展示字段
- 再让后端返回对应的模拟字段
- 最后再把模拟字段替换成真实硬件字段

这样可以把“页面问题”“接口问题”“硬件问题”分开定位。

## 6. 当前接口

当前已经使用或暴露的接口包括：

- GET /api/system/info：返回系统信息
- GET /api/wireless/status：返回当前无线状态快照
- GET /api/baseband/health：返回基带运行模式、依赖状态、host 配置和 init/start 结果
- GET /api/baseband/stats：返回基带通信统计
- GET /api/baseband/test：测试基带通信
- GET /ws：推送实时无线状态更新

其中：

- System 页签主要通过 fetch 获取一次性信息
- Wireless 页签更适合消费持续更新的数据
- WebSocket 适合推送周期性状态，避免页面主动轮询

## 7. 接口分层原则

为避免后续维护失控，新增功能时建议遵守下面的职责边界。

### 7.1 static/index.html

负责：

- 页面 DOM 结构
- fetch 和 WebSocket 消费
- 接口字段渲染到页面
- 轻量级交互逻辑

不建议负责：

- 复杂业务判断
- 设备协议细节
- 与第三方 SDK 直接耦合

### 7.2 src/main.rs

负责：

- 定义页面接口返回结构
- 组织 HTTP 路由和 WebSocket
- 调用业务封装层拿数据
- 把业务数据组装成前端可直接消费的 JSON

不建议负责：

- 裸 FFI 调用细节
- 裸指针、缓冲区、错误码解析

### 7.3 src/bb_api.rs

负责：

- 封装 SDK 初始化、socket 创建、发送、接收、统计
- 将底层错误码转换为 Rust Result
- 对上层暴露稳定、语义化的方法

### 7.4 src/ffi.rs

负责：

- extern "C" 函数声明
- 常量声明
- C 结构体映射
- 最薄的一层底层绑定

注意：

- 当前项目没有启用 bindgen 自动生成绑定
- FFI 定义目前是手工维护的
- 每次 SDK 头文件变动后，都要同步检查 src/ffi.rs

## 7.x 第三方头文件 / 库文件更新流程

当需要替换 `third_party/include` 或 `third_party/lib` 中的 SDK 版本时，请按下面流程执行，避免出现 ABI 失配。

### 7.x.1 替换文件

1. 将新版头文件覆盖到 `third_party/include/`（bb_api.h、bb_config.h、prj_rpc.h 等）。
2. 将新版库文件覆盖到 `third_party/lib/`（ar8030_client.dll、ar8030_client.lib 等）。

### 7.x.2 再生 FFI 绑定（推荐）

如果已安装 LLVM（并配置了 RSHTML_LIBCLANG_PATH），建议先再生绑定作为对照基准：

```powershell
$env:RSHTML_REGENERATE_BINDINGS = '1'
$env:RSHTML_BINDINGS_OUT = 'target/generated/ffi_bindings.rs'
cargo check
```

或直接使用封装脚本：

```powershell
.\scripts\regenerate-ffi-bindings.ps1
```

### 7.x.3 编译检查

```powershell
cargo check
```

### 7.x.4 运行 ABI 守卫测试

```powershell
cargo test abi_ -- --nocapture
```

这组测试会锁定：
- PRJ 命令字常量
- PRJ dispatch 缓冲区大小、payload 偏移
- 核心 SDK 常量（BB_MAC_LEN、BB_DATA_USER_MAX、BB_SLOT_MAX 等）
- BB ioctl 请求宏
- 固件升级 chunk 与 data 尺寸约束

如果头文件更新引入了 ABI 破坏，这些测试会第一时间失败。

### 7.x.5 运行全部测试

```powershell
cargo test
```

### 7.x.6 手工核对关键结构体

对比 `target/generated/ffi_bindings.rs` 与 `src/ffi.rs` 中以下结构：
- `bb_get_status_out_t` 及其依赖
- `bb_get_sys_info_out_t`
- `bb_dev_info_t`
- `bb_set_hot_upgrade_write_in_t`
- `prj_rpc_hdr_t`（如已通过 PACK 展开）
- 所有 `BB_SET_*`、`BB_GET_*` 常量

确保手写 FFI 与生成绑定在 size、align、字段偏移上一致。

### 7.x.7 功能验证

启动服务并在实际硬件上验证受影响的页面功能正常工作。

### 7.x.8 SDK 升级流水线（一键执行）

如果已经从 SDK 提供方获得了新版头文件和库文件（按 `include/` 和 `lib/` 目录组织），可以直接执行流水线脚本：

```powershell
.\scripts\sdk-upgrade-pipeline.ps1 -SourceDir "C:\path\to\new-sdk"
```

如果再生绑定需要指定 libclang 路径：

```powershell
.\scripts\sdk-upgrade-pipeline.ps1 -SourceDir "C:\path\to\new-sdk" -LibclangPath "C:\Program Files\LLVM\bin"
```

脚本会自动完成：复制文件 → 记录哈希 → 再生绑定 → cargo check → ABI 测试 → 生成变更摘要 → 追加版本清单。

### 7.x.9 CI 门禁

当 `third_party/include/**` 或 `third_party/lib/**` 在 PR 中发生变化时，`.github/workflows/sdk-guard.yml` 会自动触发：
- LLVM 安装
- FFI 绑定再生
- ABI 守卫测试
- 全部单元测试

漏同步 SDK 头文件的 PR 会被 CI 阻断合并。

### 7.x.10 版本清单

每次 SDK 升级后，流水线脚本会自动向 `docs/sdk-version-manifest.md` 追加一条记录，包含来源、各文件 SHA256 和兼容性备注。出现线上问题时可直接根据该清单定位对应 SDK 版本。

## 8. 新增字段的标准步骤

这是开发人员最常见的工作场景。无论是 System 页面还是 Wireless 页面，建议都按同一套流程执行。

### 场景 A：只新增页面展示字段，先用模拟数据

1. 在 src/main.rs 中给对应的返回结构增加字段。
2. 在返回 JSON 的逻辑里填入模拟值。
3. 在 static/index.html 中增加对应 DOM 节点。
4. 在页面脚本中读取新字段并渲染。
5. 如有需要，在 static/styles.css 中补样式。

适用情况：

- 页面联调阶段
- UI 验证阶段
- 硬件接口尚未就绪

### 场景 B：把模拟字段替换成真实 SDK 数据

1. 先查看 third_party/include/bb_api.h 和 third_party/include/bb_config.h，确认目标字段来自哪个函数或结构体。
2. 如果 src/ffi.rs 中没有对应声明，则先补 FFI 声明和结构体映射。
3. 在 src/bb_api.rs 中增加语义化封装方法，对错误码和数据类型做统一处理。
4. 在 src/main.rs 中调用该封装方法，并将结果转换成页面所需 JSON 字段。
5. 保持前端字段名稳定，尽量不要让页面直接感知底层 SDK 结构。
6. 本地运行后验证接口返回和页面展示是否一致。

### 场景 C：新增一个完整接口

1. 在 src/main.rs 中定义新的返回结构。
2. 增加新的 route。
3. 如果数据来自硬件，则通过 src/bb_api.rs 调用，不要直接写 FFI。
4. 在 static/index.html 中增加 fetch 或 WebSocket 消费逻辑。
5. 如果接口会被多个页面使用，优先统一字段命名和单位。

## 9. 开发建议

### 9.1 优先保持前端字段稳定

页面依赖的是 JSON 字段和 DOM id。即使底层 SDK 结构变化，也尽量把变化收敛在 Rust 层，不要把 SDK 命名直接暴露给页面。

### 9.2 不要在 main.rs 堆积底层细节

如果把大量指针处理、C 结构体转换、错误码判断都写进 src/main.rs，后续页面开发和接口开发会越来越难维护。

### 9.3 先用模拟数据跑通页面，再接真值

这能把问题拆成两部分：

- 页面是否正确展示
- 数据是否正确读取

比一步到位更容易定位问题。

### 9.4 关注 Windows 运行时 DLL 问题

如果编译成功但运行失败，优先检查：

- DLL 是否被复制到 target/debug 或 target/release
- DLL 位数是否和当前构建目标一致
- SDK 是否在运行时被 src/ffi.rs 正常加载

当前已确认的实际行为：

- 本项目使用运行时动态加载 SDK，而不是在进程启动前强依赖静态链接成功
- 如果 ar8030_client.dll 的外部依赖缺失，SDK 初始化会失败
- 即使 SDK 初始化失败，Web 服务仍会回退到模拟模式继续运行

如果本机仍无法进入真实模式，优先检查第三方 DLL 的外部依赖，例如：

- pthread.dll
- VCRUNTIME140D.dll
- ucrtbased.dll

如果 DLL 依赖已经满足，但 `bb_init failed with code: -1` 仍然出现，优先继续排查：

- 当前是否应该使用远程 bb_host 模式而不是本地模式
- BB_HOST_ADDR 和 BB_HOST_PORT 是否配置正确
- 目标 8030 设备是否真实在线
- SDK 所需驱动或宿主服务是否已经启动

## 10. 常见问题

### 10.1 页面改了为什么没生效

当前页面是静态文件直出，先确认：

- 修改的是 static 目录下文件
- 服务已重新启动或浏览器已强制刷新缓存

### 10.2 新字段应该加在哪一层

判断原则：

- 纯展示问题：改 static
- JSON 字段和接口问题：改 src/main.rs
- 硬件数据语义封装：改 src/bb_api.rs
- C 函数、结构体、常量映射：改 src/ffi.rs

### 10.3 什么时候需要看头文件

当你发现现有 Rust 封装没有你要的能力，或者要确认某个字段的真实含义、单位、返回码时，就应该回到 third_party/include 中查头文件。

## 11. 后续接真实数据的方向

当前 src/main.rs 中仍然存在模拟数据生成逻辑。后续可以按下面方向替换：

1. 从基带接口或设备状态接口获取 RF、Traffic、Connection 数据。
2. 在 Rust 层把原始数据转换成页面需要的结构。
3. 写回共享状态，并通过 WebSocket 广播给前端。

这样前端结构可以保持基本稳定，只需要逐步把模拟值替换为真实值。
