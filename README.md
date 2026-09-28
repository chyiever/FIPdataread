# FIPread

FIPread 是一款用于浏览和分析 FIP 波形文件（`.npz`、`.tdms`、`.txt`）的桌面工具。

## 功能特性

- 文件列表：支持排序、分页、基于幅值阈值的过滤
- 时域波形显示：支持可选的滤波预处理
- TDMS 输入支持 1 或 2 个通道；上方时域图默认使用 CH1
- TXT 输入支持 1 或 2 列数值，每列作为一个通道
- Plot 2 模式切换：`SVM Prediction`、`ST Energy Ratio`、`ST Energy`、`ST Energy Energy`、`ST PSD sum`、`ST-energy-max-num`
- Plot 2 也可显示第二个 TDMS 通道的波形（采用与上方时域图相同的显示滤波预处理）
- PSD 分析：基于选定的可见窗口，可选择 CH1、CH2 或 CH1+CH2 显示
- PSD 的 X/Y 显示范围可在 `Display Controls` 中调整
- 可见波形音频播放（`Play`、`Stop`、`Replay`）
- 可见波形导出为 `.wav`
- 可见原始波形导出

## 环境要求

推荐环境：

- Windows
- Python 3.9+

批量安装依赖：

```powershell
pip install -r requirements.txt
```

完整库列表及版本说明见 `docs/required-libraries.md`。

## 运行

在项目根目录执行：

```powershell
python .\run.py
```

## 打包 EXE

安装打包依赖：

```powershell
pip install -r requirements-build.txt
```

在项目根目录执行：

```powershell
python .\build_exe.py
```

默认输出：

```text
dist\FIP.YYYY.MM.DD.exe
```

示例：

```text
dist\FIP.2026.08.01.exe
```

打包行为说明：

- `dist` 中已有的 exe 默认保留。
- 若 `dist\FIP.YYYY.MM.DD.exe` 已存在，新 exe 会以时间后缀保存，例如 `FIP.2026.08.01.203012.exe`。
- PyInstaller 中间文件暂存在 `build` 下，构建成功后自动删除。
- 运行时资源 `logo.png` 与 `models/saved_models` 会打包进 exe。
- 包含 Matplotlib 颜色映射，确保 `t-f Plot` 色标与源码运行模式显示一致。

常用变体：

```powershell
python .\build_exe.py --console
python .\build_exe.py --clean-only
python .\build_exe.py --keep-intermediate
python .\build_exe.py --overwrite
python .\build_exe.py --collect-sklearn
```

## 基本操作流程

1. 选择数据目录。
2. 在列表中选择一个文件，或用 Ctrl/扩展选择选中多个兼容文件。
3. 在上方时域图中查看波形。多选文件会按起始时间排序并首尾拼接后再绘制。
4. 如需滤波，在 `Display Controls` 中调整滤波参数。
5. 使用 `矩形放大`（Zoom Mode）或 `计算PSD`（Window PSD Mode）进行交互。
6. 使用 `应用窗宽` 设置可见时长。
7. 对双通道 TDMS 文件，可在 Plot 2 中选择 `CH2` 对比第二通道（采用相同的显示预处理）。
8. PSD 通过 `PSD` 下拉框选择 `CH1`、`CH2` 或 `CH1+CH2`，默认始终为 `CH1`。
9. 时频分析：打开 `t-f Plot` 页签，用其紧凑 `CH` 下拉框选择 `CH1` 或 `CH2`，所选通道同时驱动该页签内的单一时域图和下方的时频图。
10. 在 Plot 2 中选择 `SVM Prediction`、`ST Energy Ratio`、`ST Energy`、`ST Energy Energy`、`ST PSD sum` 或 `ST-energy-max-num` 等特征工作流；左侧 `ST-feature` 面板会自动切换到对应参数。
11. 使用音频控制播放/导出当前可见片段。

## 界面与功能详解

### 显示滤波（Display Controls）

`apply_display_filter` 对波形做零相位滤波（`scipy.signal.sosfiltfilt`），滤波器为 4 阶 Butterworth，支持三种模式：

- `Band-pass`（带通）：`low_cut_hz < f < high_cut_hz`
- `High-pass`（高通）：`f > low_cut_hz`
- `Low-pass`（低通）：`f < high_cut_hz`

`Display Controls` 中还包含 PSD 的 X/Y 显示范围、`t-f` 模式/窗宽/重叠率/频率范围/色标等参数。

### PSD 分析

`compute_window_psd` 对选定的可见窗口计算 PSD：

```
scipy.signal.welch(
    signal,
    fs=sample_rate,
    window="hann",
    nperseg=窗口长度,
    noverlap=nperseg // 2,
    detrend="linear",
    scaling="density",
    return_onesided=True,
)
```

输出以 dB 显示：`psd_db = 10 · log10(max(psd, tiny))`。

### 时频分析（t-f Plot）

`compute_time_frequency_map` 基于 `scipy.signal.spectrogram`，Hann 窗，`nperseg = t-f Window (s) × 采样率`，重叠率 `noverlap = round(window_samples × overlap)`（上限 95%）。支持两种模式：

- `PSD`：`mode="psd"`，`scaling="density"`（功率谱密度）
- `Amplitude`：`mode="magnitude"`，`scaling="spectrum"`（幅度谱）

时频图 X 轴为时间（`times × sample_rate`，即样本索引），Y 轴为频率（Hz）。

### 音频播放与导出

`prepare_audio_waveform` 将波形转为 16-bit PCM：

1. 按 `Audio Downsample` 因子降采样（`scipy.signal.resample_poly`）；
2. 去均值；
3. 以 99.5 分位幅值为参考归一化到 `target_peak = 0.95`；
4. 裁切到 [-1, 1] 后量化到 int16。

`Play` / `Stop` / `Replay` 播放当前可见片段，`Export Visible Audio` 导出为 `.wav`。

### Plot 2 特征图

Plot 2 是下方第二幅图，用于显示多种短时特征或通道波形。所有特征均基于 CH1（相位数据，单位 rad）计算，输出曲线的 X 轴为时间（样本索引），X 与上方时域图同步。选择不同的特征时，左侧 `ST-feature` 页签自动切换到对应参数页；`Feature Y Min/Max` 与 `Apply Feature Y Range` 对所有特征通用。

特征模式列表：

| Plot 2 下拉项 | 数据标识 | 说明 |
| --- | --- | --- |
| None | `none` | 不显示特征图 |
| CH1 / CH2 | `channel_1_waveform` / `channel_2_waveform` | 通道波形（CH2 采用显示滤波预处理） |
| SVM Prediction | `svm_prediction` | 滑窗 SVM 分类预测 |
| ST Energy Ratio | `short_time_energy` | 两频带能量密度比（dB） |
| ST Energy | `short_time_band_energy` | 指定频带滤波后时域能量（线性） |
| ST Energy Energy | `short_time_energy_energy` | ST Energy 曲线在二级滑窗内的能量 |
| ST PSD sum | `short_time_psd_sum` | 扣除本底后的短时 PSD 带内求和 |
| ST-energy-max-num | `st_energy_max_num` | 二级滑窗内，子窗最大能量超过阈值的个数 |

> 各特征的完整公式、物理意义与函数代码见《[短时特征算法详解](docs/特征算法详解.md)》。

## 短时特征算法详解

所有短时特征共享相同的滑窗采样机制，仅在预处理与每窗统计方式上不同。

> 本文档为概述；完整的**计算公式、物理意义与函数定义代码**见《[docs/特征算法详解.md](docs/特征算法详解.md)》。

### 公共滑窗机制

设采样率为 `fs`，ST Energy 类特征的**窗口宽度以 ms 为单位**（输入范围 0.001 ~ 100000 ms，内部换算为秒后乘以 `fs`）：

- 窗长（样本数）：`window_samples = max(2, round(window_seconds_ms / 1000 × fs))`
- 步长（样本数）：`hop_samples = max(1, round(window_samples × step_percent / 100))`
- 窗起点：`0, hop, 2·hop, ...`，直到 `signal.size − window_samples`
- 曲线 X 坐标：窗中心样本索引 `start + window_samples / 2`

**幅度门控（Amplitude Gate）**：对每个窗，取门控信号（默认即当前显示波形 `_current_display_values`）在窗内的幅值 `max(|gate|)`；若小于 `Amplitude Gate` 阈值，该窗输出 `0.0`。门控信号跟随显示滤波，因此特征曲线会随显示设置变化。

### ST Energy Ratio（短时能量密度比）

默认参数：Band 1 = 4000–10000 Hz，Band 2 = 20000–40000 Hz，Window = 30 ms，Step = 50%，Amplitude Gate = 0.02。

算法步骤：

1. **预处理**：对 CH1 相位数据做 4 阶 Butterworth **100 Hz 高通** 零相位滤波（`butter(4, 100/nyquist, 'highpass')` + `sosfiltfilt`）。
2. **逐窗 FFT**：每个窗先乘 **Hann 窗**，再作 `rfft`，得到功率谱 `power = |FFT|²`。
3. **频带选择**：频率轴为 `numpy.fft.rfftfreq(window_samples, d=1/fs)`；频带 `[low, high]` 选择所有满足 `low ≤ f ≤ high` 的频点。
4. **能量密度**：
   - `numerator_density = Σ power[Band 1] / (Band 1 High − Band 1 Low)`
   - `denominator_density = Σ power[Band 2] / (Band 2 High − Band 2 Low)`
5. **输出**（dB）：

```
10 · log10( numerator_density / denominator_density )
```

取对数前加极小下限 `numpy.finfo(float).tiny` 以避免 `log10(0)`。纵轴标签为 `Band Energy Density Ratio (dB)`。

### ST Energy（短时频带能量）

默认参数：Band = 4000–10000 Hz，Window = 30 ms，Step = 50%，Amplitude Gate = 0.02。

算法步骤：

1. **预处理**：对 CH1 相位数据做 4 阶 Butterworth **带通** 滤波，通带为 `[Band Low, Band High]` Hz（`sosfiltfilt` 零相位）。若 `Band Low = 0`，退化为在 `Band High` 处的低通滤波。
2. **逐窗输出**（线性，非 dB）：

```
Σ x[n]²
```

即窗内带通滤波后时域信号的平方和（时域能量）。纵轴标签为 `Band Energy (Σx²)`。

### ST Energy Energy（ST Energy 的能量）

两阶段特征：先计算 `ST Energy` 曲线，再对该曲线做二级滑窗求和。

默认参数：Band = 4000–10000 Hz，Stage 1 Window = 0.1 ms，Stage 1 Step = 50%，Amplitude Gate = 0.02，Stage 2 Window = 20 ms，Stage 2 Step = 50%。

**阶段 1**：与 `ST Energy` 完全相同——对 CH1 在 `[Band Low, Band High]` 带通滤波后，逐窗输出 `Σ x[n]²`，得到曲线 `(y[i], t[i])`（`t[i]` 为窗中心样本索引）。

**阶段 2**：以 `Stage 2 Window (ms)` 为窗宽、`Stage 2 Step (% of window)` 为步长，在阶段 1 曲线上滑动。每个二级窗输出：

```
Σ y[i]²     （对窗内所有阶段1窗中心 y[i] 的平方求和）
```

X 坐标为二级窗中心时间。输出为线性，非 dB。纵轴标签为 `ST Energy Energy (Σy²)`。

### ST PSD sum（扣除本底后短时 PSD 带内求和）

默认参数：Band = 4000–10000 Hz，PSD Window = 0.025 s（25 ms），Step = 0.015 s（15 ms），Background Window = 1.0 s。

算法步骤：

1. **本底 PSD**：取 CH1 相位数据前 `Background Window (s)`（默认 1 s）作为本底噪声段，用 `scipy.signal.welch` 估计本底 PSD：
   ```
   welch(background_segment, fs=fs, window="hann", nperseg=PSD Window 样本数,
         noverlap=nperseg // 2, detrend="linear", scaling="density")
   ```
   得到线性 PSD（rad²/Hz，非 dB），频率网格为 `nperseg = round(PSD Window × fs)` 对应的单边谱。
2. **短时 PSD**：以 `PSD Window (s)` 为窗宽、`Step (s)` 为步长滑动，每个窗用相同 `welch` 设置计算短时 PSD。因 `nperseg` 相同，短时 PSD 与本底 PSD 的频率网格完全一致。
3. **相减**：逐频点 `short_time_psd − background_psd`。
4. **带内求和**：在 `[Band Low, Band High]` Hz 内对相减后 PSD 逐点求和：

```
Σ (psd_i − background_i)     （i 遍历频带内所有频点）
```

输入信号单位为 rad，故每个频点为 rad²/Hz，求和后的特征输出单位为 rad²/Hz（线性）。X 坐标为窗中心。纵轴标签为 `Band PSD Sum (rad²/Hz)`。

### ST-energy-max-num（ST Energy 超阈值子窗计数）

两阶段特征：先计算 `ST Energy` 曲线，再统计二级滑窗内超过阈值的 1 ms 子窗个数。

默认参数：Band = 4000–10000 Hz，Stage 1 Window = 0.1 ms，Stage 1 Step = 50%，Amplitude Gate = 0.02，Stage 2 Window = 70 ms，Stage 2 Step = 0.015 s（15 ms），Sub Window = 1 ms，Max Threshold = 400（单位为 ×1e-6，输入范围 0.1 ~ 1000000）。

**阶段 1**：与 `ST Energy` 完全相同——对 CH1 在 `[Band Low, Band High]` 带通滤波后，逐窗输出 `Σ x[n]²`，得到曲线 `(y[i], t[i])`。

**阶段 2**：以 `Stage 2 Window (ms)` 为窗宽、`Stage 2 Step (s)` 为步长在阶段 1 曲线上滑动。每个二级窗被切成 `Sub Window (ms)`（默认 1 ms）宽、互不重叠的若干子窗（默认 70 个），对每个子窗取阶段 1 值的最大值：

```
count = #{ k ∈ 子窗 : max(y[i]) 在子窗 k 内 > Max Threshold }
```

输出为该 70 ms 窗内小窗最大值大于 `Max Threshold` 的小窗个数（范围 0~70）。X 坐标为二级窗中心。纵轴标签为 `ST Energy Max Num`。

### SVM Prediction（滑窗 SVM 预测）

从 `models/saved_models` 加载 sklearn Pipeline（`svm_model.joblib`）及其元数据（`svm_model_metadata.json`，包含所选特征、能量频带、统计频带）。

滑窗默认 `window_seconds = 0.04 s`、`hop_ratio = 0.5`。每个窗：

1. **门控**：信号经 20 kHz 高通后，若窗内峰值 `max(|gate|) ≤ 0.05`，则跳过（输出 0）。
2. **特征提取**：对活跃窗计算元数据中指定的能量频带能量、能量密度比（参考频带 1000–60000 Hz）与统计频带统计量。
3. **预测**：用加载的 SVM 模型输出 0/1 预测。

输出纵轴标签为 `SVM Prediction`，固定 Y 范围 [-0.1, 1.1]。

## 数据格式与解析说明

- X 轴时间标签由「文件名时间戳 + 采样率」推导。
- 旧版 TDMS/TXT 采样率标识：如 `-200K-`。
- 新版 TDMS/TXT 采样率标识：如 `-1MHz-`、`_1000k`。
- 旧版起始时间标识：如 `20260323T103125.631`。
- 新版 TDMS 起始时间标识：如 `2026-8-1-12-43-36`。
- 紧凑版 TXT 起始时间标识：如 `20260820170538.780`。
- 可见原始波形导出会保留所有加载通道；上方时域图、音频播放/导出、SVM 预测、ST Energy Ratio、ST Energy、ST PSD sum、ST-energy-max-num 默认均使用 CH1。
- 文件列表 Ctrl/扩展多选会按解析出的起始时间拼接所选文件；所选文件必须具有相同的采样率与通道数。
- t-f 图默认 CH1，加载双通道文件后可在 `t-f Plot` 页签切换到 CH2。
- Plot 2 的 `CH2 Waveform` 采用与上方时域图相同的显示滤波设置。
- PSD 使用选定的原始窗口，可绘制 CH1、CH2 或 CH1+CH2。
- 当 PSD X 范围跨度至少一个数量级时，X 轴标签限制为 10 的幂，例如 `100Hz`、`1000Hz`、`10000Hz`。