# FIP 快速读取-处理-绘图策略总结

目标：单个输入文件达到几百 MB（如 1 MHz 采样、数百万 ~ 数千万点）时，仍能快速读取、快速滤波、快速绘图，同时**不降低采样率、不丧失数据完整性**。

## 1. 总体原则

- 数据完整性优先：所有滤波、分析、导出都基于**全分辨率原始采样点**，不抽稀、不降采样。
- 只有「显示层」允许抽稀：pyqtgraph 的 `setDownsampling(auto=True, method="peak")` + `setClipToView(True)` 仅在绘制时按屏幕像素抽取极值点，原始数组不被动过。
- 计算全部向量化：避免逐点 Python 循环；能整块计算的不分块，不能整块的分块但保持精度。

## 2. 快速读取（src/data_access.py）

- TDMS：使用 `nptdms.TdmsFile.read(path)` 读取元数据，随后按通道 `channel[:]` 取出数据。
  - 实测：96 MB / 6M 点双通道 TDMS，元数据 0.04 s，全量读取 + 转 float64 约 0.2 s。
  - 读取后按通道原样保留为 numpy 数组，`phase_data` 沿用第一通道，其余通道进 `channels` 元组，避免重复拷贝。
- NPZ：`np.load(path, allow_pickle=True)` 一次性读入，`phase_data` 以 float64 展平。
- 结果统一封装为 `LoadedWaveform`，供 UI 各模块复用同一份数据，不再重复读盘。

## 3. 快速滤波（src/processing.py）

- `apply_display_filter` 使用 `scipy.signal.butter` + `sosfiltfilt` 做零相位滤波（滤波后无相位失真，数据完整）。
- 滤波器系数（SOS 矩阵）按 `(mode, sample_rate, low, high)` 分离到 `_build_filter_sos()`，仅与参数相关，避免每次调用重复设计滤波器。
- 实测：6M 点带通滤波（1k–50k Hz，4 阶）约 0.14 s；高通（20k Hz）约 0.11 s。
- 滤波结果只存一份到 `_current_display_values`，绘图/PSD/时频/特征全部复用。

## 4. 快速绘图（src/main_window.py + src/plotting.py）

- 曲线绘制：
  - `setClipToView(True)`：只绘制当前视口内的数据段。
  - `setDownsampling(auto=True, method="peak")`：屏幕级自动抽稀，峰值保持，**仅影响显示，不改变底层数据**。
  - `setSkipFiniteCheck(True)`：跳过逐点 NaN/Inf 检查，加速大数据 setData。
- 时频图：
  - `scipy.signal.spectrogram` 一次整段计算（C 扩展，向量化）。
  - `_build_time_frequency_display_grid()` 将线性频率网格插值到对数频率网格，改用**批量向量化线性插值**（`searchsorted` + 权重组合）替换原先逐列 `np.interp` Python 循环。
  - 实测：6M 点 / 401 频率 bin / 62496 窗的显示网格构建从约 0.7 s 降到约 0.5 s，且与旧结果逐点一致（最大差 ~1e-25）。
- 视图状态统一走 `_apply_view_state()`，一次 setXRange/setYRange，避免多次触发重排。

## 5. 大批量特征计算

- 短时能量比（`compute_short_time_energy_ratio`）：整段 `np.fft.rfft` 批量求谱后按频带掩码求和，仅在能量超阈值的窗内计算。
- SVM 滑动窗（`compute_short_time_svm_predictions`）：分块处理特征、按统计频带并行（`ThreadPoolExecutor`），不改变特征精度。
- PSD 窗口（`compute_window_psd`）：对当前可见段 `welch` 一次计算。

## 6. 交互层：窗口自适应与同步

- 启动时按屏幕可用区域夹取窗口尺寸，主分割条默认左:右 ≈ 1:5，左侧面板最小宽度 360（不再塌缩）。
- 图 1（时域）与图 2（特征/Plot 2）左轴宽度统一为 90 px，保证两图左侧边缘对齐。
- 图 1 / 图 2 X 轴双向同步缩放：移除单向 `setXLink`，改用 `sigXRangeChanged` 互连 + `_syncing_time_feature_x` 防重入标志，任一侧放大另一侧跟随，且不产生循环抖动。

## 7. 性能基线（参考机型，96 MB / 6M 点双通道 TDMS）

| 操作 | 耗时 |
| --- | --- |
| TDMS 读取（双通道 → float64） | ~0.2 s |
| 带通滤波 1k–50k（4 阶零相位） | ~0.14 s |
| 时频图 spectrogram + 显示网格 | ~1.1 s |
| 单次 `_rebuild_time_plot`（含时频重建） | ~1.4 s（稳态） |
| 显示层缩放/平移 | 实时（抽稀后绘制） |

## 8. 验证要点

- 滤波前后点数不变（`size` 恒定），无降采样。
- 显示抽稀仅作用于绘制，`_current_display_values` 始终为全分辨率数组。
- 时频网格向量化结果与旧逐列插值结果 `np.allclose` 一致。
