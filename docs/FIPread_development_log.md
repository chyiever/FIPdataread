# FIPread Development Log

Purpose: keep a simple long-term update log for this project.

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
