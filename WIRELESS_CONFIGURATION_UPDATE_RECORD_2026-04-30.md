# Wireless Configuration Update Record

日期：2026-04-30

## Git Baseline

- 操作前基线提交：5203cab
- 建议基线标记：pre-wireless-configuration-20260430

说明：

- 该基线对应 Configuration 页面与板端配置功能接入之前的仓库状态。
- 本次新增功能提交应从该基线继续追踪与回退。

## 本次更新目标

1. 在 Wireless 中新增 Configuration 子页，并与 Status / RF Settings 同层并排切换。
2. 对齐 PQ Tool 的 Configuration 能力：
   - Auto / Memory / Flash 配置文本读取
   - Configuration 文本保存
   - Clear Flash Configuration
   - Clear MiniDB Configuration
   - Restore Factory Settings
   - MiniDB Role / Frequency Band / Power / Local MAC / AP MAC / Slot MAC
3. 基于板端 API 与 PQ Tool 源码补齐 Rust 后端封装与 HTTP 接口。
4. 补充实机联调检查文档，便于后续逐项验收。

## 本次更新内容

### 1. 前端页面

- 在 Wireless 二级页签中新增 Configuration。
- 新增 Configuration 主布局：
  - 左侧 Config Text 编辑区
  - 右侧 MiniDB 配置区
  - 顶部 Mode / Read / Save / Clear Flash / Clear MiniDB / Restore Factory / Help 按钮组
- 新增前端交互：
  - 读取指定 mode 配置
  - 保存配置文本
  - MiniDB role / band 即时下发
  - MiniDB power / MAC / slot MAC 按钮下发
  - 操作成功后联动刷新 runtime 与 snapshot
- 修正 MiniDB 空值回显：reset 后 `role` / `band_bitmap` 为 null 时不再错误显示为 AP / Auto。

### 2. 后端 HTTP 接口

- 新增 GET `/api/wireless/configuration?mode=0|1|2`
- 新增 POST `/api/wireless/configuration/apply`
- 新增请求动作：
  - `save_config`
  - `reset_config`
  - `reset_minidb`
  - `restore_factory`
  - `set_role`
  - `set_band`
  - `set_power`
  - `set_local_mac`
  - `set_ap_mac`
  - `set_slot_macs`

### 3. Rust SDK / FFI 封装

- 新增 Configuration 文本读写：
  - `BB_GET_CFG`
  - `BB_SET_CFG`
  - `BB_RESET_CFG`
- 新增 PRJ Dispatch 能力：
  - `BB_GET_PRJ_DISPATCH`
  - `BB_SET_PRJ_DISPATCH`
- 新增 MiniDB 相关结构与命令：
  - role
  - band bitmap
  - power
  - local MAC
  - AP MAC
  - slot MAC
  - reset db
- 新增 CRC16-CCITT 分块写入逻辑，与 PQ Tool 配置写入路径对齐。

### 4. 文档

- 新增实机联调清单：Auto / Memory / Flash、MiniDB 各项、破坏性操作、接口探针、故障定位顺序。

## 影响文件

- `src/ffi.rs`
- `src/bb_api.rs`
- `src/main.rs`
- `static/index.html`
- `static/styles.css`
- `WIRELESS_CONFIGURATION_HARDWARE_CHECKLIST.md`

## 验证结果

- `get_errors`：前端与 Rust 相关变更文件无错误
- `cargo check --quiet`：通过

## 备注

- 工作区中存在未纳入本次提交的临时文件：`.tmp_do_not_use`
- 该临时文件不属于本次 Configuration 功能的一部分，提交时应排除