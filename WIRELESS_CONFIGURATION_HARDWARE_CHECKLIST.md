# Wireless Configuration Hardware Checklist

本清单用于对 Wireless -> Configuration 页面做一轮实机联调。

范围包括：

- Configuration 文本读取：Auto / Memory / Flash
- Configuration 文本写入：Read / Save
- MiniDB：Role / Band / Power / Local MAC / AP MAC / Slot MAC 0-7
- 破坏性操作：Clear Flash / Clear MiniDB / Restore Factory

不在本轮强制验收范围内：

- 运行态 RF 行为是否即时切换到新的持久化值
- 板端重启后的长期稳定性压测
- 配置文件语义级正确性

## 1. 联调前提

执行前请先确认：

1. 当前板卡是测试板，不是生产板。
2. 你已经能访问 Web 页面，并能打开 Wireless -> Configuration。
3. 如果使用 remote bb_host 模式，bb_host / daemon 已启动。
4. 联调前已做好基线备份：
   - Flash 模式读出一份配置文本
   - MiniDB 当前值截图或导出一份 JSON
5. 若要执行 Clear Flash / Clear MiniDB / Restore Factory，必须先确认回滚路径。

## 2. 启动与健康检查

推荐启动方式：

- VS Code task: Debug Rust Project

或者使用 PowerShell：

```powershell
cargo build
if ($LASTEXITCODE -eq 0) { .\target\debug\wireless_status_server.exe }
```

启动后先验证健康状态：

```powershell
Invoke-RestMethod http://127.0.0.1:8080/api/baseband/health | ConvertTo-Json -Depth 6
```

通过标准：

- effective_mode 为 hardware-local-sdk 或 hardware-remote-bb-host
- 不是 simulator
- remote 模式下 host.connected 为 true
- 没有持续性的 bb_host_connect failed / bb_ioctl failed

如果这里不过，不要继续做 Configuration 联调，先解决底层连接问题。

## 3. 记录基线

建议先记录以下信息：

- 日期
- 板卡编号
- 本机 MAC
- 当前运行角色
- effective_mode
- software / hardware / firmware version

建议保存三份基线：

```powershell
Invoke-RestMethod "http://127.0.0.1:8080/api/wireless/configuration?mode=0" | ConvertTo-Json -Depth 8
Invoke-RestMethod "http://127.0.0.1:8080/api/wireless/configuration?mode=1" | ConvertTo-Json -Depth 8
Invoke-RestMethod "http://127.0.0.1:8080/api/wireless/configuration?mode=2" | ConvertTo-Json -Depth 8
```

重点记录：

- current.config_total_length
- current.config_total_crc16
- current.role
- current.band_bitmap
- current.power_mode / power_auto / power_initial_dbm / power_min_dbm / power_max_dbm
- current.local_mac_address
- current.ap_mac_address
- current.slot_macs

## 4. 判定规则

本轮联调的判定顺序如下：

1. 先看接口 JSON 是否 success = true。
2. 再看 current 字段是否按预期变化。
3. 最后再看 UI 是否正确回显。

注意：

- MiniDB 是持久化配置，不一定等价于当前运行态。
- Role / Band / Power 的“写入成功”，本轮以 Configuration 页回读一致为准，不以 Status 页是否即时变化作为唯一标准。
- Clear MiniDB 后，如果 current.role 或 current.band_bitmap 为 null，表示“未设置/已清空”。当前 UI 会保持未选中状态，这是正确现象。

## 5. 安全项：Auto / Memory / Flash 逐项验证

### 5.1 Auto Read

操作：

1. 打开 Wireless -> Configuration。
2. Mode 选择 Auto。
3. 点击 Read。

通过标准：

- 状态条显示读取成功
- Config Text 有内容或明确的空文本结果
- Size 和 CRC 有合理值
- 接口返回 success = true

记录项：

- Auto length
- Auto crc16
- Auto 首行文本

### 5.2 Memory Read

操作：

1. Mode 选择 Memory。
2. 点击 Read。

通过标准：

- 读取成功
- 返回数据结构完整
- 如果与 Auto 不同，差异可解释为内存缓存未落盘或板端优先级不同

记录项：

- Memory length
- Memory crc16
- 是否与 Auto 一致

### 5.3 Flash Read

操作：

1. Mode 选择 Flash。
2. 点击 Read。

通过标准：

- 读取成功
- 返回数据结构完整
- 返回结果可作为后续回滚基线

记录项：

- Flash length
- Flash crc16
- 是否与 Auto 一致

### 5.4 三模式一致性判定

优先按下面逻辑判断：

1. Auto = Flash = Memory：最理想
2. Auto = Memory，Flash 不同：说明内存已有变更但未落盘
3. Auto = Flash，Memory 不同：说明内存缓存异常或未刷新
4. 三者都不同：先暂停写入验证，排查板端模式语义

建议留一份对比表：

| 项 | Auto | Memory | Flash | 结论 |
| --- | --- | --- | --- | --- |
| length |  |  |  |  |
| crc16 |  |  |  |  |
| 文本前 64 字节 |  |  |  |  |

## 6. 安全项：Configuration Save 验证

### 6.1 保守验证：原文回写

适用场景：不想修改任何业务字段。

操作：

1. 先用 Flash Read 读出文本。
2. 不修改文本，直接点击 Save。
3. 再次分别读取 Auto / Memory / Flash。

通过标准：

- Save 成功
- Flash 模式仍可正常读回
- 不出现空文本、乱码或明显截断

### 6.2 轻量写入验证：仅测试板执行

适用场景：允许在测试板上做一次最小改动。

建议方式：

- 只做一处可回退的小改动
- 优先选择无业务风险字段
- 如果不确定字段语义，不做文本修改，只做 6.1 的原文回写验证

通过标准：

- Save 后再次 Read，文本与修改一致
- Flash 回读与预期一致
- Auto / Memory 与 Flash 的差异可解释

## 7. MiniDB 验证总则

MiniDB 建议按以下顺序验证：

1. Role
2. Band
3. Power
4. Local MAC
5. AP MAC
6. Slot MAC 0-7

每一项都执行：

1. 读基线
2. 写新值
3. 再读回
4. 与接口 JSON 比较
5. 记录是否影响运行态页面

## 8. MiniDB 逐项验证

### 8.1 Role

操作：

1. 记录 current.role 基线。
2. 在 Configuration 页切换到另一角色。
3. 等待写入成功。
4. 重新 Read 当前 mode。

通过标准：

- current.role 与下发值一致
- UI 单选项回显一致
- 不要求 Status 页运行态立即切换

备注：

- 这是 MiniDB 持久化值，不是运行态 role 切换。

### 8.2 Band

依次验证 4 个值：

- 600M = 0x01
- 2.4G = 0x02
- 5.8G = 0x04
- Auto = 0x07

每个值的操作：

1. 点击对应单选项。
2. 等待写入成功。
3. 重新 Read 当前 mode。

通过标准：

- current.band_bitmap 与目标值一致
- UI 回显正确

备注：

- 这是 MiniDB band bitmap，不是当前 RF 实时 band。

### 8.3 Power

本轮只验证以下字段：

- power_auto
- power_initial_dbm
- power_min_dbm
- power_max_dbm

注意：

- 当前页面中的 power_mode 是只读来源值，不在本轮前端可编辑范围内。

验证 1：Auto Policy

1. Policy 选 Auto。
2. 设置 Min / Max。
3. 点击 Apply Power。
4. 重新 Read。

通过标准：

- current.power_auto = true
- current.power_min_dbm / power_max_dbm 回读一致

验证 2：Manual Policy

1. Policy 选 Manual。
2. 设置 Init dBm。
3. 点击 Apply Power。
4. 重新 Read。

通过标准：

- current.power_auto = false
- current.power_initial_dbm 回读一致

### 8.4 Local MAC

操作：

1. 记录基线值。
2. 输入测试 MAC。
3. 点击 Apply。
4. 重新 Read。

通过标准：

- current.local_mac_address 回读一致
- 文本格式正常

建议：

- 测试后恢复原值

### 8.5 AP MAC

操作与 Local MAC 相同。

通过标准：

- current.ap_mac_address 回读一致

### 8.6 Slot MAC 0-7

建议最少验证：

- slot 0
- slot 7

如果要做全覆盖，则 0-7 全部执行一遍。

每个 slot 的操作：

1. 记录基线值。
2. 修改指定 slot MAC。
3. 点击 Apply Slot MACs。
4. 重新 Read。

通过标准：

- current.slot_macs 中对应 slot 的 mac_address 回读一致
- 其他 slot 未被意外覆盖

建议表：

| Slot | 基线 | 目标值 | 回读值 | 结果 |
| --- | --- | --- | --- | --- |
| 0 |  |  |  |  |
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |
| 6 |  |  |  |  |
| 7 |  |  |  |  |

## 9. 破坏性项：最后执行

以下操作都可能破坏当前板上配置，只能放到最后：

- Clear Flash
- Clear MiniDB
- Restore Factory

执行前必须：

1. 保存 Flash 文本基线
2. 保存 MiniDB 基线 JSON
3. 明确回滚方法

### 9.1 Clear Flash

操作：

1. 点击 Clear Flash。
2. 再做 Flash Read。
3. 再做 Auto Read。

通过标准：

- 接口 success = true
- Flash 模式结果发生变化，或返回空/默认内容
- Auto 模式结果符合板端优先级语义

注意：

- 如果 Auto 立刻没有变化，不一定是失败，可能仍在读 Memory。
- 如有必要，重进页面或重启板端后再次对比 Auto 与 Flash。

### 9.2 Clear MiniDB

操作：

1. 点击 Clear MiniDB。
2. 重新 Read 当前 mode。
3. 再用接口 JSON 检查 current.role / current.band_bitmap / MAC / power 字段。

通过标准：

- 接口 success = true
- current.role 可能为 null
- current.band_bitmap 可能为 null
- MAC 字段为空
- power 相关字段为空或 null

注意：

- 这里一定要以接口 JSON 为准，不要只看 UI 单选状态。

### 9.3 Restore Factory

操作：

1. 点击 Restore Factory。
2. 重新 Read Flash。
3. 再检查 MiniDB 各字段。

通过标准：

- Flash 和 MiniDB 都回到默认/工厂状态
- 后续 Read 不报错

## 10. 常用接口探针

读取配置：

```powershell
Invoke-RestMethod "http://127.0.0.1:8080/api/wireless/configuration?mode=0" | ConvertTo-Json -Depth 8
Invoke-RestMethod "http://127.0.0.1:8080/api/wireless/configuration?mode=1" | ConvertTo-Json -Depth 8
Invoke-RestMethod "http://127.0.0.1:8080/api/wireless/configuration?mode=2" | ConvertTo-Json -Depth 8
```

下发角色：

```powershell
Invoke-RestMethod "http://127.0.0.1:8080/api/wireless/configuration/apply" -Method Post -ContentType "application/json" -Body '{"action":"set_role","mode":0,"role":1}' | ConvertTo-Json -Depth 8
```

下发频段：

```powershell
Invoke-RestMethod "http://127.0.0.1:8080/api/wireless/configuration/apply" -Method Post -ContentType "application/json" -Body '{"action":"set_band","mode":0,"band_bitmap":7}' | ConvertTo-Json -Depth 8
```

读取运行态：

```powershell
Invoke-RestMethod "http://127.0.0.1:8080/api/wireless/runtime" | ConvertTo-Json -Depth 8
```

## 11. 故障定位顺序

如果某项失败，按下面顺序排查：

1. /api/baseband/health 是否仍然健康
2. configuration 接口返回是否 success = false
3. current 是否为 null
4. warnings 是否出现 bb_ioctl 失败
5. 失败是否只出现在某个 mode
6. 写入成功但回读不一致时，优先比较 Auto / Memory / Flash 是否读到同一来源

## 12. 本轮验收建议

建议以以下标准作为“本轮通过”：

1. Auto / Memory / Flash 三种读取都成功。
2. 原文 Save 能成功并稳定回读。
3. MiniDB 的 Role / Band / Power / Local MAC / AP MAC / Slot MAC 至少各完成一轮写入回读。
4. Clear Flash / Clear MiniDB / Restore Factory 至少在测试板上各成功一次。
5. 任何失败项都能通过接口 JSON 留下可定位证据。