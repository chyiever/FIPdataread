# FIPread 开发日志

> 本文档用于长期记录 FIPread 项目的开发、优化与问题修复过程，供追溯、复盘与回归验证。
> 后续新增记录请严格按下方《一、开发日志编写规范》补充，并放在对应的日期位置；《二、编写示例》提供可直接复制的模板。

## 一、开发日志编写规范

### 1. 排序规则

- 全局按日期倒序排列，最新条目在最上方。
- 同一天有多条记录时，自上而下依次为：
  - `## YYYY-MM-DD`（当天第一条）
  - `## YYYY-MM-DD (续)`（当天第二条）
  - `## YYYY-MM-DD (续 2)`（当天第三条，依此类推）

### 2. 标题格式

- 日期使用 `YYYY-MM-DD`，例如 `## 2026-08-29`。
- 标题不使用具体时刻；同一天多条用 `(续)` 后缀区分撰写顺序。

### 3. 条目结构

每条记录按下列小节组织，不适用的小节可以省略：

1. 功能 / 变更描述：以 `-` 列表逐条说明“改了什么、为什么改”。
2. 问题修复（如本次含 bug 修复）：
   - `根因：` 说明问题根本原因，尽量给出可复现条件与量化数据。
   - `修复：` 说明解决方案与关键取舍。
3. 关键代码：列出涉及的文件与函数，格式 `src/module.py::func_name()`。
4. 验证：给出可复验的命令与结果（如 `python -m py_compile ...`、冒烟测试、启动实测观察）。
5. 收尾（推荐）：以 `Problem solved: ` 一句话概括本次改动解决的最终问题。

### 4. 语言与术语

- 用户可见的界面文字（按钮 / 标签 / 页签等）使用界面原文，如 `1D Curve`、`t-f Plot`、`Short-Time Energy`、`应用窗宽`。
- 代码符号用反引号标注：`_apply_initial_window_size()`、`LoadedWaveform.channels`。
- 数字与单位明确书写，如 `0.3 ms`、`10 kHz ~ 50 kHz`、`1500x920`。
- 描述性正文中英文皆可，建议英文术语 + 中文说明。

### 5. 其他约定

- 每次功能开发或问题修复完成后，先补充对应日期日志，再提交代码。
- 性能或数值结论需写明可复现条件与实测数值，不使用“感觉变快了”这类无依据描述。
- 凡是改动会波及数据格式、交互方式或默认参数时，同步更新 README / 数据说明文档，并在日志中注明。

## 二、编写示例

### 2.1 模板（可直接复制）

```text
## YYYY-MM-DD
- <一句话总述本次改动>。
- <功能 / 变更点 1>。
- <功能 / 变更点 2>。

（如本次包含缺陷修复，补充：）
- 根因：
  - <问题根本原因，含可复现条件与量化数据>。
- 修复：
  - <解决方案与关键取舍>。
- 关键代码：
  - `src/xxx.py::func_name()`
- 验证：
  - <可复验命令>（如 `python -m py_compile src/xxx.py`）
  - <冒烟 / 启动实测观察结果>。
- Problem solved: <本次改动解决的最终问题>。
```

### 2.2 对照实例

当前最新一条真实记录（`2026-08-29 (续)`）即按本规范编写，可作为完整参照，详见下方《三、开发日志》。

## 三、开发日志

### 2026-09-29 (续)
- 文档整理：
  - `README.md` 重写为中文，新增“界面与功能详解”“短时特征算法详解”等章节，补充显示滤波、PSD、t-f、音频处理等技术细节。
  - 新增 `docs/特征算法详解.md`：为每个短时特征给出计算公式、中文含义、物理意义、界面参数默认值，以及**完整的函数实现代码**（函数体内用 `# ==== 模块名 ====` 分隔行划分输入校验 / 预处理 / 滑窗生成 / 逐窗计算 / 返回等模块，便于阅读）。
- 关键代码：
  - `docs/特征算法详解.md`
  - `README.md`
- 验证：
  - `python -m py_compile src\main_window.py src\processing.py src\config.py`
- Problem solved: 短时特征的公式、物理意义与实现代码集中成文档，便于理解与回归。

### 2026-09-29
- Plot 2 新增多个短时特征，并按下拉选择自动切换 `ST-feature` 参数页：
  - `ST Energy Ratio`（原 `Short-Time Energy` 更名）：两频带能量密度比（dB）。100 Hz 高通预处理 + Hann 窗 FFT，频带功率除以带宽后取比值再取对数。
  - `ST Energy`：指定频带带通滤波（4 阶 Butterworth，`Band Low=0` 时退化为低通）后，逐窗输出时域信号平方和 `Σx[n]²`，线性坐标。
  - `ST Energy Energy`：两阶段。阶段 1 同 `ST Energy`（默认 0.1 ms 窗）；阶段 2 用 20 ms 二级窗对阶段 1 曲线做 `Σy[i]²`。
  - `ST PSD sum`：以信号前 1 s 用 `welch`（25 ms 窗）估计本底 PSD，25 ms 窗 15 ms 步长短时 PSD 逐点相减后在指定频带求和，输出 rad²/Hz。
  - `ST-energy-max-num`：两阶段。阶段 1 同 `ST Energy`；阶段 2 用 70 ms 二级窗、15 ms 步长滑动，将二级窗切成 1 ms 子窗，统计子窗最大能量大于阈值（默认 400，单位 ×1e-6）的子窗个数（0~70）。
- `ST-feature` 面板改为 `QStackedWidget`（5 个参数页：Ratio / Energy / Energy Energy / PSD sum / max-num），选不同特征自动切换；共享的 `Window / Step / Amplitude Gate` 控件在不需要时隐藏。
- 窗口宽度单位统一为 **ms**（输入范围 0.001 ~ 100000 ms，界面显示 ms，内部除以 1000 换算为秒）；`ST-energy-max-num` 的 `Max Threshold` 单位为 **×1e-6**（输入范围 0.1 ~ 1000000，内部乘以 1e-6）。
- 新增特征计算函数：`compute_short_time_band_energy`、`compute_short_time_energy_sum`、`compute_short_time_psd_sum`、`compute_short_time_max_num`；`compute_short_time_energy_ratio` 逻辑不变。
- 关键代码：
  - `src/processing.py::compute_short_time_energy_ratio()`
  - `src/processing.py::compute_short_time_band_energy()`
  - `src/processing.py::compute_short_time_energy_sum()`
  - `src/processing.py::compute_short_time_psd_sum()`
  - `src/processing.py::compute_short_time_max_num()`
  - `src/main_window.py::_rebuild_short_time_feature_plot()`
  - `src/main_window.py::_update_feature_params_page()`
  - `src/main_window.py::_rebuild_short_time_energy_ratio_plot()`
  - `src/main_window.py::_rebuild_short_time_band_energy_plot()`
  - `src/main_window.py::_rebuild_short_time_energy_energy_plot()`
  - `src/main_window.py::_rebuild_short_time_psd_sum_plot()`
  - `src/main_window.py::_rebuild_short_time_max_num_plot()`
  - `src/config.py::FeaturePanelDefaults`
- 验证：
  - `python -m py_compile src\main_window.py src\processing.py src\config.py`
  - 离屏 UI 冒烟：`feature_plot_mode_combo` 共 9 项、`feature_params_stack` 共 5 页，各模式切换后页索引与 `feature_window_gate_widget` 显隐正确；
  - 数值冒烟：`compute_short_time_band_energy`（0.1 ms 窗）→ `compute_short_time_energy_sum`（20 ms 二级窗）两阶段 19999→99 窗；`compute_short_time_max_num` 信号段计数升至 70。
- Problem solved: Plot 2 现支持 5 类短时特征，参数单位统一为 ms / ×1e-6，各特征参数页随下拉自动切换。

### 2026-09-05
- 修复双通道文件在 `1D Curve` 页显示 `CH1` / `CH2` 两个时域图时高度仍不相等的问题。
- 根因：
  - 之前 `_apply_right_plot_splitter_sizes()` 用固定 `overhead_time=108` 和 `overhead_tabs=42` 估算顶部工具栏、信息行、滚动条、tab 页框与 splitter handle 的高度；
  - 真实 Qt 布局高度会随窗口尺寸、DPI、字体、tab 页框和 splitter handle 变化，固定估算值偏离后，外层 `right_panel` 与内层 `curve_splitter` 的比例不能保证 Plot 1 / Plot 2 / PSD 的实际绘图区满足 `1:1:2`；
  - 因此虽然上一次设置了补偿公式，在部分窗口尺寸下仍会出现两个时域图高度不一致，或两个时域图高度之和不等于 PSD 图高度。
- 修复：
  - `_apply_right_plot_splitter_sizes()` 改为读取当前布局实测开销：顶部面板非 `time_plot` 部分高度、`analysis_tabs` 相对 `curve_splitter` 的页框/标签开销，以及外层 splitter handle 宽度；
  - 在可用绘图高度中按 4 份分配：Plot 1 占 1 份，Plot 2 占 1 份，PSD 占 2 份；
  - 同步设置外层 `right_panel` 和内层 `_curve_splitter`，确保 Plot 1 与 Plot 2 等高，且二者之和等于 PSD 高度；
  - 在窗口 resize 与 `1D Curve` / `t-f Plot` 页签切换后用 `QTimer.singleShot(0, ...)` 延迟重算一次，避免布局尚未完成时使用旧高度。
- 关键代码：
  - `src/main_window.py::resizeEvent()`
  - `src/main_window.py::_apply_right_plot_splitter_sizes()`
  - `src/main_window.py::_measure_time_panel_overhead()`
  - `src/main_window.py::_update_time_tf_alignment_for_tab()`
- 验证：
  - `python -m py_compile src\main_window.py`
- Problem solved: 双通道 `CH1` / `CH2` 时域图按实际布局动态保持等高，两个时域图高度之和与 PSD 图高度一致。

### 2026-09-04
- 文件列表支持 Ctrl/扩展多选；选中多个可读波形文件后，按文件起始时间从早到晚首尾拼接为一个连续波形，再复用现有绘图、滤波、PSD、t-f、音频和导出流程。
- 多文件拼接会校验采样率与通道数一致，并按通道分别拼接，保留双通道数据参与后续分析。
- PSD 横轴刻度线改为朝外（向下），刻度值后不再显示 Hz 单位。
- 修复双通道时两个时域图（Plot 1 与 Plot 2）高度不一致的问题。
- 功能 / 变更点：
  - 新增 `load_waveforms_concatenated()`，加载多个文件后按 `start_time` 排序并构造合成 `LoadedWaveform`；
  - `QListWidget` 改为 `ExtendedSelection`，通过 `itemSelectionChanged` 获取当前选中路径；
  - `LoadWaveformWorker` 从单路径加载改为路径列表加载；
  - `LogPowerFrequencyAxis` 使用与时间轴一致的朝外刻度配置；
  - `_update_psd_x_axis_ticks()` 输出纯数字（如 `1`、`10`、`100`、`1000`），不带 `Hz` 后缀；
  - `_apply_right_plot_splitter_sizes()` 在 `Plot 2` 显示 `CH1` / `CH2` 时补偿 `mode_row`、`info_row`、`time_scrollbar` 的固定占用高度，使 Plot 1 与 Plot 2 的可视绘图区高度一致。
- 关键代码：
  - `src/data_access.py::load_waveforms_concatenated()`
  - `src/main_window.py::LoadWaveformWorker`
  - `src/main_window.py::_handle_file_selection()`
  - `src/main_window.py::_start_waveform_load()`
  - `src/main_window.py::_apply_right_plot_splitter_sizes()`
  - `src/main_window.py::_update_psd_x_axis_ticks()`
  - `src/plotting.py::LogPowerFrequencyAxis`
- 验证：
  - `python -m py_compile src\data_access.py src\main_window.py src\plotting.py`
- Problem solved: 可通过 Ctrl 多选把同采样率、同通道数的文件按时间拼接分析；PSD 横轴刻度朝外且无 Hz 单位；双通道时域图高度一致。

### 2026-08-29 (续)
- 修复主窗口启动时的 `QWindowsWindow::setGeometry: Unable to set geometry ...` 警告与窗口超出屏幕的问题。
- 根因：
  - 右侧模式工具栏原为单行布局（6 个模式按钮 + `窗宽(s)` + `PSD / CH / Plot 2` 等控件），其最小宽度约 `1305` 逻辑像素，叠加左侧面板后窗口 `minimumSizeHint` 达到约 `1686` 逻辑像素；
  - 高 DPI（如 150%）下该最小宽度随字体同步放大，超过屏幕可用宽度（如逻辑宽 `1707`）后，`show()` 触发布局把窗口强制撑到最小尺寸，超出屏幕，被 Windows 截断并打印 geometry 警告。
- 修复：
  - 将模式工具栏拆分为两行：按钮行 `mode_row` 与信息控件行 `info_row`，右侧面板最小宽度从约 `1305` 降至约 `811`，窗口 `minimumSizeHint` 由约 `1686` 降至约 `1192`；
  - 主布局设置 `QLayout.SetNoConstraint`，允许窗口小于内容最小尺寸，避免小屏被强制撑大；
  - `_apply_initial_window_size()` 扣减窗口边框（水平约 `24`、标题栏约 `40`）后按屏幕可用区域裁剪，并设置窗口最小尺寸 `800x600`。
- 关键代码：
  - `src/main_window.py::_build_ui()`——工具栏分行与布局约束
  - `src/main_window.py::_apply_initial_window_size()`
- 验证：
  - `python -m py_compile src/main_window.py`
  - 启动实测：初始尺寸保持 `1500x920`，控制台不再出现 geometry 警告，右侧工具栏控件完整可见、不被截断。
- Problem solved: 主窗口在不同 DPI / 分辨率下均可正确适配屏幕，无 geometry 警告，界面控件完整可见。

### 2026-08-29 (续 2)
- 修复 `1D Curve` 页顶部时域图与 `t-f Plot` 页下方时频图的横向时间轴未对齐的问题。
- 根因：
  - 顶部时域图所在 `time_panel` 占满右侧面板全宽；而 `t-f Plot` 页内容区 `tf_plot` 右侧被颜色条（`tf_histogram`，固定宽 `88`）、页内右边距与间距（`6+6`）以及 `QTabWidget` 页框（约 `2`）占据，绘制区比顶部时域图窄约 `102` 逻辑像素；
  - 上下两图 X 轴时间范围相同、左边缘对齐，但同一时间刻度在下方图中整体左移，左右边缘不齐，造成“上下时间没对齐”的观感。
- 修复：
  - 恢复顶部面板右缘补偿占位 `time_right_spacer`：仅当当前页签为 `t-f Plot`（index==1）时将其宽度设为 `self._tf_time_axis_compensation = 102`，使顶部时域图绘制区宽度与 `tf_plot` 一致；切换回 `1D Curve` 时置回 `0` 以复用全宽。
- 关键代码：
  - `src/main_window.py::_update_time_tf_alignment_for_tab()`
- 验证：
  - 用 ViewBox `geometry()` 量化对比两图 plot-area 右边缘：`t-f Plot` 页签下右侧差由约 `102` 降至 `0.00`；在 `1500x920`、`900x600`、`1200x800` 三种窗口尺寸下均为 `0.00`，与窗口宽度无关；
  - `python -m py_compile src/main_window.py`。
- Problem solved: `1D Curve` 顶部时域图与 `t-f Plot` 时频图在任意窗口尺寸下左右边缘时间轴完全对齐。

### 2026-08-29
- Refined the right plotting splitter ratios for two-channel 1D analysis:
  - Plot 1 / Plot 2 / PSD now use a 1:1:2 height ratio when Plot 2 is visible.
  - Plot 2 remains collapsed when the current file has only one channel and Plot 2 is `None`.
- Renamed the left-side file tab from `Flie List` to `File`.
- Optimized full-screen switching from `1D Curve` to `t-f Plot` by removing the fixed top-panel right spacer and narrowing the t-f colorbar to reduce right-edge clipping.
- Restored the shared top time-domain panel for both `1D Curve` and `t-f Plot`; switching to `t-f Plot` no longer hides the Zoom/PSD toolbar or replaces it with a standalone `CH` row.
- Moved the t-f `CH` selector back to the main toolbar between `PSD` and `Plot 2`, and kept it as the source selector for both the top time plot and the time-frequency map.
- Changed Plot 2 to use the same interactive time-plot widget as Plot 1, with bidirectional X-axis synchronization:
  - Dragging, rectangular zoom, reset, undo, visible-window application, and 2x zoom-out now keep Plot 1 and Plot 2 aligned in time.
  - Fixed PSD selection regions are mirrored on both time-domain plots.
- Added `CH1` and `CH2` waveform options to `Plot 2`; two-channel files default to `CH2`, while single-channel files switch to `None` and collapse Plot 2 height to zero.
- Set the Plot 1 / Plot 2 splitter to equal heights when Plot 2 is visible.
- Updated toolbar labels:
  - `Zoom Mode` -> `矩形放大`
  - `Window PSD Mode` -> `计算PSD`
  - `Back View` -> `撤销放大`
  - `PSD WINDOWS` -> `固定PSD窗`
  - `Zoom Out 2x` -> `缩小2倍`
  - `Reset View` -> `重置窗口`
  - `Apply Visible Window` -> `应用窗宽`
- Switched the application font to `Times New Roman` for controls and reduced the left control-tab width while keeping tab text visible.
- Combined file management, file list, display controls, short-time feature controls, and audio controls into one left-side tab group: `Flie List`, `Display`, `ST-feature`, and `Audio`.
- Reduced the t-f colorbar alignment width and stopped resetting the right vertical splitter on analysis-tab changes, improving full-screen switching between `1D Curve` and `t-f Plot`.

### 2026-08-28
- Added TXT waveform input compatibility for files such as `20260820170538.780_SemiPhase_1000k.txt`.
- The file list now includes `.txt` together with `.npz` and `.tdms`.
- TXT loading supports one or two numeric columns; each column is mapped to one waveform channel and stored in `LoadedWaveform.channels`.
- `LoadedWaveform.phase_data` remains column/CH1 so existing first-channel workflows continue unchanged.
- Extended filename parsing to support compact start-time tokens like `YYYYMMDDHHMMSS.fff` and sample-rate tokens at the end of the file stem such as `_1000k`.
- Added `TXT` to the visible raw data export-format dropdown and implemented single-column txt export for the current visible CH1 segment.
- Updated README and data-structure documentation for TXT read/export behavior.
- Problem solved: txt captures with one or two columns can be opened from the normal file list, carry filename-derived start time/sample rate metadata, and participate in the existing two-channel plot/PSD workflows.
- Added PSD X-axis range controls in `Display Controls`.
  - `PSD X Min (Hz)` and `PSD X Max (Hz)` use `0 / 0` as the automatic range, otherwise values are applied in Hz and clamped to the current file Nyquist frequency.
  - The existing range apply button is now labeled `Apply Ranges` and applies phase Y, PSD X/Y, feature Y, t-f Y, and t-f color ranges.
  - Added a PSD-specific log-frequency bottom axis. When the visible PSD X range spans at least one decade, only powers of ten are labeled, for example `100Hz`, `1000Hz`, and `10000Hz`.
- Problem solved: users can inspect a chosen PSD frequency band without dense intermediate tick labels crowding the logarithmic frequency axis.
- Added a compact `CH` dropdown with `CH1` and `CH2` for the `t-f Plot` tab.
  - CH2 is enabled only when the loaded file has at least two channels.
  - Switching the t-f source recomputes the time-frequency map from the selected channel using the same display filter settings.
  - The t-f plot status message now reports the selected channel label.
- Problem solved: two-channel TDMS/TXT files can compare time-frequency behavior by channel from the `t-f Plot` tab selector.
- Changed the Plot 2 `CH2 Waveform` left-axis label from `CH2 Phase (rad)` to `Phase (rad)` so the second time-domain waveform keeps the same Y-axis label as the first time-domain plot.
- The `t-f Plot` tab now owns a single internal time-domain plot above the t-f image; the external top time-domain panel is collapsed on this tab.
- Removed the extra margins around the `t-f Plot` content, collapsed the hidden external time-domain panel to zero height, and changed tab-specific right-panel splitter sizing:
  - `1D Curve`: keeps the taller time-domain panel and lower Plot 2 / PSD analysis area.
  - `t-f Plot`: uses a smaller single time-domain panel and gives the remaining height to the t-f image.
- Problem solved: the `t-f Plot` view shows only one time-domain plot, no longer leaves unnecessary blank space under the hidden external panel, and the single visible time-domain plot has a compact CH1/CH2 switch.

### 2026-08-15
- Fixed the main-splitter layout so the left control panel no longer collapses to a narrow strip and is hard to pull open.
  - Root cause: the right panel `mode_row` (6 mode buttons + labels + 2 combo boxes + spin box + Apply button) forced a ~1539 px minimum width, and together with the left panel minimum it made the window minimum wider than the available screen width, collapsing the left pane.
  - Fix: reduced mode button minimum width to 56 px with `Preferred` policy, trimmed combo-box minimum width and row spacings, kept the left scroll area at a 360 px minimum, and set the initial splitter ratio to approximately 1:5 at startup.
  - Problem solved: the left panel stays at a usable width, the right controls remain fully visible, and the window fits the available screen (e.g. 1667 px on a 1707 px logical screen).
- Aligned the left edges of Plot 1 (time-domain) and Plot 2 (feature) by giving both plots the same left-axis width (90 px).
- Made Plot 1 / Plot 2 X-axis zoom synchronized in both directions.
  - Previously `feature_plot.setXLink(time_plot)` only let Plot 1 drive Plot 2.
  - Replaced it with `sigXRangeChanged` handlers in both directions plus a `_syncing_time_feature_x` re-entry guard, so zooming either plot updates the other without feedback loops.
- Optimized large-file handling (hundreds of MB) without any sampling-rate reduction or data-integrity loss.
  - Verified the existing pipeline already keeps full-resolution arrays: reading (~0.2 s for 6M points), zero-phase `sosfiltfilt` filtering (~0.14 s), and display-only decimation (`setDownsampling` peak + `setClipToView`) that never touches the underlying data.
  - Moved Butterworth SOS coefficient design into `_build_filter_sos()` so filters are not re-designed on every call.
  - Vectorized the time-frequency display-grid interpolation: replaced the per-column `np.interp` loop with `searchsorted` + vectorized lerp, cutting grid build time from ~0.7 s to ~0.5 s while keeping results bitwise-consistent.
- Added `FIP快速读取-处理-绘图策略总结.md` summarizing the fast read/filter/plot strategy for large TDMS files.
- Problem solved: the UI now stays responsive for multi-million-point files, Plot 1/Plot 2 align and zoom together, and the layout no longer collapses on lower-resolution screens.

### 2026-08-14 (续)
- Added `docs/required-libraries.md` as the required-library manifest.
  - Lists all runtime dependencies (numpy, scipy, PyQt5, pyqtgraph, nptdms, pandas, joblib, scikit-learn, matplotlib) and build-only dependencies (pyinstaller, Pillow) with the versions verified in the development environment.
- Added `requirements.txt` for runtime dependencies and `requirements-build.txt` for packaging dependencies.
- Updated `README.md` to install dependencies in batch via `pip install -r requirements.txt` and `pip install -r requirements-build.txt`, replacing the previous single-line install command.
- Problem solved: a fresh machine can now restore the full environment with two pip commands instead of remembering the library list manually.

### 2026-08-14
- Fixed GUI font-size inconsistency across machines where the UI text appears too large and overflows buttons after moving the app to another computer.
- Root cause:
  - the Qt application was created without enabling high-DPI scaling
  - on displays with Windows scaling above 100%, Qt sized point-based fonts (SimSun/Times New Roman) using the physical DPI while widget geometry stayed in logical pixels
  - the mismatch made button text render larger than the button bounds, so labels got clipped or displayed incompletely
- Fix:
  - `run.py` now enables `Qt.AA_EnableHighDpiScaling` and `Qt.AA_UseHighDpiPixmaps` before creating the `QApplication`
  - with high-DPI scaling on, Qt scales fonts and layouts by the same device-pixel ratio, so text always fits its widgets regardless of the host display scaling
  - applied the same change to `sig_mark/run.py`
- Problem solved: the GUI now renders with consistent font/widget proportions on different computers, and button labels are no longer cut off.
- Rebuilt the packaged exe with the existing naming convention (`FIP.2026.08.14.exe`).

### 2026-08-01
- Added `build_exe.py` for PyInstaller-based Windows packaging.
- Default exe name is `FIP.YYYY.MM.DD.exe`, for example `FIP.2026.08.01.exe`.
- The build script stages PyInstaller outputs under `build/`, moves only the final exe into `dist/`, and removes intermediate files after a successful build.
- Existing `dist/*.exe` files are preserved by default; if the same output name already exists, the new exe receives a time suffix instead of overwriting the old one.
- Bundled runtime resources required by the packaged app:
  - `logo.png`
  - `models/saved_models`
- Updated app resource lookup so logo and SVM model paths work both from source and from a PyInstaller onefile runtime.
- Updated `README.md` with exe build dependencies, default output naming, old-exe preservation behavior, and common packaging command variants.
- Fixed packaged exe startup failure caused by a non-`pyqtgraph.ColorMap` object being passed into `ImageItem.setColorMap()` during t-f plot initialization.
- Added a matplotlib-to-`pyqtgraph.ColorMap` conversion path and packaged matplotlib colormap modules so the t-f color bar stays visually consistent with source-mode display.
- Kept local fallback color maps for emergency startup only when matplotlib colormap loading is unavailable.
- Narrowed the default sklearn packaging scope to the modules required by the bundled `Pipeline(StandardScaler, SVC)` model; `--collect-sklearn` remains available as a slower compatibility fallback.
- Moved the generated `.ico` file out of the cleaned build directory so PyInstaller can still find it during the final exe assembly step.
- Added compatibility for TDMS files named like `SemiPhase-1MHz-2026-8-1-12-43-36.tdms`.
- Extended filename parsing to support `K/KHz/M/MHz` sample-rate tokens and `YYYY-M-D-H-M-S` start-time tokens.
- Extended `LoadedWaveform` to retain all TDMS channels while preserving `phase_data` as CH1 for existing first-channel workflows.
- Added `CH2 Waveform` to the Plot 2 dropdown; it applies the same display filter preprocessing as CH1.
- Added a PSD source dropdown with `CH1`, `CH2`, and `CH1+CH2`; CH1 remains the default for all files.
- Updated README and data-structure documentation for the new TDMS dual-channel behavior.
- Problem solved: two-channel TDMS files can now be inspected without losing the existing first-channel workflows, and PSD comparison between CH1 and CH2 is available from the UI.
- Problem solved: FIPread now has a repeatable exe packaging workflow that keeps historical builds while cleaning temporary packaging artifacts.

### 2026-04-16
- Fixed `t-f Plot` axis-tick rendering after multiple failed attempts that confused axis ticks with grid/reference lines.
- Corrected log-frequency minor ticks to standard base-10 positions (`2..9 x 10^n`) instead of equal subdivisions within each decade.
- Removed the earlier `InfiniteLine`-style pseudo minor-tick idea from the final solution path and separated axis ticks from plot-area guide lines.
- Added custom short-tick drawing on top of `pyqtgraph.AxisItem.generateDrawSpecs()`:
  - `LogFrequencyAxis.generateDrawSpecs()` now supplements outward short major/minor ticks on the left frequency axis.
  - `AbsoluteTimeAxis.generateDrawSpecs()` now supplements outward short ticks on the bottom time axis.
- Added shared helpers in `src/plotting.py`:
  - `_manual_tick_levels()` extracts visible tick levels from the current axis range.
  - `_append_axis_tick_stubs()` draws short tick stubs at the axis edge independent of grid rendering.
- Updated `t-f` grid policy to avoid visual ambiguity:
  - keep vertical time grid lines
  - disable horizontal frequency grid lines
  - keep short ticks as an axis-only visual element
- Key related code:
  - `src/main_window.py::_update_time_frequency_axis_ticks()`
  - `src/main_window.py::_handle_tf_y_range_changed()`
  - `src/main_window.py::_apply_time_frequency_y_range()`
  - `src/plotting.py::_manual_tick_levels()`
  - `src/plotting.py::_append_axis_tick_stubs()`
  - `src/plotting.py::AbsoluteTimeAxis.generateDrawSpecs()`
  - `src/plotting.py::LogFrequencyAxis.generateDrawSpecs()`
- Fixed a separate `t-f Plot` frequency-axis mapping bug:
  - the spectrogram output frequency bins are linearly spaced in Hz
  - but the image had been placed directly into a log-frequency axis using one affine `ImageItem.setRect(...)`
  - this made real high-frequency energy appear at much lower displayed frequencies
- Root cause:
  - `ImageItem` supports only uniformly spaced rows/columns under affine mapping
  - therefore it cannot directly represent a linear-frequency matrix on a log-frequency y-axis
- Final fix:
  - added `_build_time_frequency_display_grid()` in `src/main_window.py`
  - resampled each spectrogram column from linear-frequency bins onto an evenly spaced `log10(f)` grid before rendering
  - reused the same display grid for color-level computation to keep the rendered image and histogram consistent
- Key related code:
  - `src/main_window.py::_build_time_frequency_display_grid()`
  - `src/main_window.py::_render_time_frequency_image()`
  - `src/main_window.py::_apply_time_frequency_color_levels()`
- Problem solved: the `t-f Plot` y-axis labels and the rendered energy distribution now refer to the same physical frequencies; a `10 kHz ~ 50 kHz` band-pass no longer appears falsely concentrated around `1~3 kHz`.
- Problem solved: `t-f Plot` now shows correct log-frequency major/minor ticks and outward short axis ticks on both left and bottom axes without mistaking them for in-plot grid lines.

### 2026-04-15
- Refactored the lower-right plotting area into analysis tabs:
  - `1D Curve` now contains previous Plot 2 + Plot 3.
  - Added `t-f Plot` for short-time time-frequency visualization.
- Added dedicated t-f controls in `Display Controls`:
  - Mode (`PSD` / `Amplitude`)
  - Value scale (`Log` / `Linear`)
  - Window length (default `0.005 s`) and overlap (default `50%`)
  - t-f Y range, colormap, and color level auto/manual controls
- Implemented short-time t-f computation pipeline in `processing.py`:
  - `PSD` path via `welch`
  - `Amplitude` path via one-sided FFT
- Implemented pyqtgraph-based rendering (`ImageItem + HistogramLUTWidget`) with log-frequency axis.
- Added two-way X-axis synchronization between time-domain plot and t-f plot.
- Updated tab selected-state styling to clearly differentiate active tab text/background.
- Problem solved: one app view now supports both legacy 1D curves and interactive t-f analysis with synchronized navigation.

### 2026-04-14 (续)
- Optimized UI layout and visual style for the left control panel and top header.
- Added top branding header with logo + centered title.
- Refactored control tabs to three sections: `Display Controls`, `Short-Time Feature`, and `Audio`.
- Moved audio controls into the dedicated `Audio` tab without changing audio behavior.
- Increased spacing and border contrast for better visual grouping across left-side modules.
- Standardized button visual feedback and set all button text to bold.
- Updated visible-time precision to 3 decimal places for label and input consistency.
- Problem solved: improved readability, clearer module separation, and more consistent interaction feedback.

### 2026-04-14
- Merged `scripts` + `scripts_svm` into one program based on the `scripts_svm` branch.
- Added Plot 2 mode switch (`SVM Prediction` / `Short-Time Energy`).
- Merged `Display Controls` and `Short-Time Feature` into left-side tabs.
- Enabled manual width resize for left panel with splitter layout.
- Problem solved: one unified app now supports both feature views and better panel ergonomics.

### 2026-03-31
- Upgraded `scripts_svm` with visible-waveform audio playback/export.
- Added `Play / Stop / Replay`, `Audio Path`, and `Audio Downsample`.
- Problem solved: users can directly listen to and export visible waveform segments.

### 2026-03-26
- Added threshold filtering workflow for file list.
- Increased page size and improved paging UX.
- Added page jump and progress feedback.
- Problem solved: large-file browsing and quick file screening became easier.
