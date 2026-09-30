# DATA Structure README

本文档描述 FIPread 当前版本的数据读取与导出结构（对应代码：`src/data_access.py`、`src/main_window.py`、`src/processing.py`）。

## 1. 可读取的数据类型

当前仅支持两种输入文件：

- `.npz`
- `.tdms`

扩展名匹配不区分大小写。

## 2. 读取结构说明

### 2.1 NPZ 读取结构

读取 `.npz` 时使用以下顶层键：

| 键名 | 必需 | 类型 | 说明 |
|---|---|---|---|
| `phase_data` | 是 | `float64` 一维数组 | 主波形数据，读取时会拉平为一维 |
| `sample_rate` | 是 | `float` | 采样率（Hz） |
| `comm_count` | 是 | `int` | 通信计数 |
| `timestamp` | 是 | `float` | 时间戳 |
| `data_info` | 否 | 任意（常见 `dict`） | 附加元信息，读取失败仅 warning，不阻止加载 |

说明：

- 缺少必需键会导致文件加载失败。
- `data_info` 是可选扩展字段，不参与核心读取校验。

### 2.2 TDMS 读取结构

读取 `.tdms` 时：

1. 查找第一个有通道的 group。
2. 读取该 group 的第一个 channel 作为波形。
3. 波形统一转成 `float64` 一维数组。

## 3. 文件名要求（读取）

### 3.1 时间 token（NPZ/TDMS 通用）

文件名中会解析如下时间格式作为起始时间：

- `YYYYMMDDTHHMMSS`
- `YYYYMMDDTHHMMSS.fff...`（小数秒 1~6 位）

示例：

- `20260323T102944.379`
- `20260324T033931.619`

若文件名缺少时间 token，则回退到文件修改时间（`mtime`）。

### 3.2 采样率 token（TDMS 必需）

`.tdms` 文件名必须含 `-<rate>K-` 片段（大小写不敏感），例如：

- `-200K-`
- `-500K-`
- `-1000K-`

若缺失会报错：`Cannot determine sample rate from file name`。

`.npz` 的采样率来自文件内 `sample_rate` 字段，不依赖文件名解析。

## 4. 原始数据导出（Export Visible Raw Data）

### 4.1 导出格式选择

在 `Files Management` 区域：

- `Refresh` 按钮右侧新增下拉框，格式选项：
  - `NPZ`
  - `TDMS`
- 默认选中：`NPZ`

点击 `Export Visible Raw Data` 时，按下拉框当前选择导出。

样本类型输入框（`Files Management`）：

- 为“下拉列表 + 可编辑输入框”。
- 为空表示不标记样本类型。
- 下拉中内置五大类与细分代码：
  - 断丝：`BK14`、`BK12`、`BK40`
  - 平稳流噪声：`F0`、`F052`、`F065`、`F130`
  - 非平稳流噪声：`F0a`、`F052a`、`F065a`、`F130a`
  - 锤击：`HM12`、`HM14`
  - 其他：`OT`
- 支持手动输入自定义代码。
- 鼠标在大类标题项上悬停时，会弹出该大类的小类代码提示。

### 4.2 导出数据来源

- 导出片段来源于当前可见窗口对应的原始波形切片（`phase_data[start:end]`）。
- 导出片段起始时间 = 原始文件起始时间 + 可见窗口起始采样点偏移。

### 4.3 导出为 NPZ 时的结构

顶层键如下：

| 键名 | 类型 | 说明 |
|---|---|---|
| `phase_data` | `float64` 一维数组 | 可见窗口波形 |
| `sample_rate` | `float` | 采样率（Hz） |
| `comm_count` | `int` | 当前导出片段采样点数 |
| `npts` | `int` | 当前导出文件总采样点数 |
| `timestamp` | `float` | 片段起始时间对应 Unix 时间戳 |
| `starttime` | `str` | 片段起始时间字符串（`YYYYMMDDTHHMMSS.mmm`） |
| `arrival_time` | `str \| None` | 初至时间绝对时间字符串（`YYYYMMDDHHMMSS.ffff`），未标记则为 `None` |
| `type` | `str \| None` | 样本类型代码（如 `BK14`、`F052a`、`HM12`、`OT`），未标记则为 `None` |
| `data_info` | `dict` | 导出附加信息（见下） |

`data_info` 当前包含：

- `type`: `phase_data_export_visible_segment`
- `length`: 采样点数
- `npts`: 采样点数
- `duration_seconds`: 片段时长（秒）
- `save_time`: 导出时刻（ISO 字符串）
- `starttime`: 与顶层 `starttime` 相同
- `arrival_time`: 与顶层 `arrival_time` 相同
- `sample_type`: 与顶层 `type` 相同

说明：

- 你要求的 `starttime`（文件名时间）已写入顶层键和 `data_info`。
- 你要求的 `npts`（总采样点数）已写入顶层键和 `data_info`。

### 4.4 导出为 TDMS 时的结构

- Root properties:
  - `start_time`
  - `sample_rate`
  - `channel_name = phase_data`
- Group: `FIP`
- Channel: `phase_data`
- Channel properties:
  - `start_time`
  - `sample_rate`
  - `unit_string = rad`

### 4.5 原始数据导出文件名默认规范

- `NPZ`：
  - 未标记样本类型：`FIP-<rateK>-<start>.npz`
  - 已标记样本类型：`<type_code>-FIP-<rateK>-<start>.npz`
  - 其中 `<type_code>` 为完整样本类型代码（如 `BK14`、`F053`、`F052a`、`HM12`、`OT`）
- `TDMS`：`FIP-<rateK>-<start>.tdms`
- `<start>` 格式：`YYYYMMDDTHHMMSS.mmm`

示例：

- `FIP-200K-20260323T103125.631.npz`
- `BK14-FIP-200K-20260323T103125.631.npz`
- `F053-FIP-200K-20260323T103125.631.npz`
- `FIP-200K-20260323T103125.631.tdms`

默认导出目录：`<项目当前工作目录>/exports`

## 5. 音频导出（Export Visible Audio）

### 5.1 音频格式与结构

- 输出格式：`.wav`
- 数据类型：`int16`（PCM16）
- 声道：单声道
- 数据来源：当前可见显示波形（可能已应用显示滤波）

导出前处理：

1. 去均值
2. 按 `Audio Downsample` 降采样（默认 10）
3. 峰值归一化并限幅
4. 转 `int16` 写入 WAV

音频采样率：

- `audio_sample_rate = round(original_sample_rate / downsample_factor)`，最小 1 Hz。

### 5.2 音频文件名默认规范

默认路径：

- `<项目当前工作目录>/exports/<YYYYMMDDHHMM>.wav`

示例：

- `exports/202603231031.wav`

用户可在 UI 中手动修改音频路径与文件名。

## 6. 快速结论

- 读取：支持 `.npz` / `.tdms`。
- 原始数据导出：支持 `NPZ` / `TDMS`，下拉默认 `NPZ`。
- 导出 `npz` 新增 `npts` 字段，表示该导出文件总采样点数。
- 导出 `npz` 包含 `starttime`，即文件名中使用的起始时间 token。
- 导出 `npz` 包含 `arrival_time`（绝对时间，精度 0.0001 s），未标记时为 `None`。
- 导出 `npz` 包含 `type`（样本类型代码）；已标记时文件名前缀使用完整类型代码（例如 `BK14-`、`F053-`）。
