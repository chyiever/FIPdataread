# FIPread Development Log

Purpose: keep a simple long-term update log for this project.

## 2026-08-29
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

## 2026-08-28
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

## 2026-03-26 14:00
- Added threshold filtering workflow for file list.
- Increased page size and improved paging UX.
- Added page jump and progress feedback.
- Problem solved: large-file browsing and quick file screening became easier.

## 2026-03-31 18:00
- Upgraded `scripts_svm` with visible-waveform audio playback/export.
- Added `Play / Stop / Replay`, `Audio Path`, and `Audio Downsample`.
- Problem solved: users can directly listen to and export visible waveform segments.

## 2026-04-14 21:29
- Merged `scripts` + `scripts_svm` into one program based on the `scripts_svm` branch.
- Added Plot 2 mode switch (`SVM Prediction` / `Short-Time Energy`).
- Merged `Display Controls` and `Short-Time Feature` into left-side tabs.
- Enabled manual width resize for left panel with splitter layout.
- Problem solved: one unified app now supports both feature views and better panel ergonomics.

## 2026-04-14 22:10
- Optimized UI layout and visual style for the left control panel and top header.
- Added top branding header with logo + centered title.
- Refactored control tabs to three sections: `Display Controls`, `Short-Time Feature`, and `Audio`.
- Moved audio controls into the dedicated `Audio` tab without changing audio behavior.
- Increased spacing and border contrast for better visual grouping across left-side modules.
- Standardized button visual feedback and set all button text to bold.
- Updated visible-time precision to 3 decimal places for label and input consistency.
- Problem solved: improved readability, clearer module separation, and more consistent interaction feedback.

## 2026-04-15 15:40
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

## 2026-04-16
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

## 2026-08-01
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

## 2026-08-14
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

## 2026-08-14 (续)
- Added `docs/required-libraries.md` as the required-library manifest.
  - Lists all runtime dependencies (numpy, scipy, PyQt5, pyqtgraph, nptdms, pandas, joblib, scikit-learn, matplotlib) and build-only dependencies (pyinstaller, Pillow) with the versions verified in the development environment.
- Added `requirements.txt` for runtime dependencies and `requirements-build.txt` for packaging dependencies.
- Updated `README.md` to install dependencies in batch via `pip install -r requirements.txt` and `pip install -r requirements-build.txt`, replacing the previous single-line install command.
- Problem solved: a fresh machine can now restore the full environment with two pip commands instead of remembering the library list manually.

## 2026-08-15
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
