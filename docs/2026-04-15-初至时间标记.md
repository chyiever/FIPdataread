# 2026-4-15 mark_arrival_time 开发记录

## 1. 开发目标

- 在时域图中支持“初至时间”手动标记。
- 支持通过键盘微调初至线位置，满足高精度定位需求。
- 导出可见波形为 `npz` 时写入 `arrival_time` 字段。
- 读取 `npz` 时在状态栏展示头信息：采样率、初至时间、总时长。

## 2. 交互设计与实现

### 2.1 初至标记交互

- `Ctrl + 鼠标右键`：
  - 在时域图当前位置创建/更新一条蓝色竖线。
  - 该竖线对应时刻定义为 `First arrival time`。

- `Ctrl + ← / Ctrl + →`：
  - 蓝色竖线向左/向右移动。
  - 固定步长：`0.1 ms`。
  - 实现方式：按当前采样率换算为样本步长 `step_samples = 0.0001 * sample_rate`。

### 2.2 状态栏反馈

- 标记或微调后，状态栏显示：
  - `First arrival time: YYYYMMDDHHMMSS.ffff`
- 精度：`0.0001 s`（4 位小数）。

## 3. 数据结构变更

### 3.1 NPZ 导出结构新增字段

在导出可见原始波形为 `npz` 时，新增：

- 顶层字段 `arrival_time`
  - 有标记：保存为绝对时间字符串，格式 `YYYYMMDDHHMMSS.ffff`
  - 无标记：保存 `None`

同时在 `data_info` 中同步写入：

- `arrival_time`

现有字段保持兼容：

- `phase_data`
- `sample_rate`
- `comm_count`
- `npts`
- `timestamp`
- `starttime`
- `data_info`

### 3.2 NPZ 读取结构扩展

读取 `npz` 时新增解析：

- 顶层 `arrival_time`（优先）
- 若顶层不存在，则尝试 `data_info.arrival_time`

兼容格式：

- `YYYYMMDDHHMMSS.ffff...`（推荐）
- `YYYYMMDDTHHMMSS.fff...`（旧格式兼容）
- ISO 字符串（兼容）

解析失败不会阻断主流程，仅记录 warning。

## 4. 状态栏头信息显示

读取文件成功后，状态栏显示头信息（重点覆盖 `npz`）：

- 文件名
- `sample_rate`
- `arrival_time`
- 总时长 `duration`（秒）

示例：

- `xxx.npz | sample_rate=200000 Hz | arrival_time=20260304155621.2333 | duration=60.000000 s`

## 5. 关键代码变更

- `src/main_window.py`
  - `TimePlotWidget` 新增 `arrivalMarkRequested` 信号
  - 新增 `Ctrl+右键` 标记逻辑
  - 新增 `Ctrl+左右` 微调快捷键
  - 新增蓝色初至线创建/更新/清除逻辑
  - 新增加载后头信息状态栏消息构造
  - 导出 `npz` 时传入 `arrival_time`

- `src/data_access.py`
  - 新增 `format_arrival_time_token()`
  - 新增 `parse_arrival_time_token()`
  - `save_npz_waveform()` 新增可选参数 `arrival_time`
  - `npz` 导出结构新增 `arrival_time`
  - `npz` 读取流程新增 `arrival_time` 解析

- `src/models.py`
  - `LoadedWaveform` 新增字段 `arrival_time: Optional[datetime]`

## 6. 本地验证

- 语法检查：
  - `python -m py_compile src/main_window.py src/data_access.py src/models.py`

- 读写冒烟验证：
  - 导出 `npz` 后，确认顶层存在 `arrival_time`
  - 重新读取该 `npz`，确认 `arrival_time` 能解析回模型并用于状态显示

## 7. 说明

- 初至时间使用绝对时间保存，不使用相对秒偏移。
- 若当前无标记，导出 `npz` 的 `arrival_time` 为 `None`，保持结构统一。
